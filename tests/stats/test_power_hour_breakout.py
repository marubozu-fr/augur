"""Tests for stats.power_hour_breakout.standard.

All data is synthetic — no real market files required. Expected values are
hand-calculated before each assertion.

Framing (single condition ``power_hour``): given the range established during the
pre-power-hour window [09:30, 15:15), how does the power hour [15:15, 16:15) behave?
Four mutually exclusive outcomes partition every countable day (strict inequality —
touching the level exactly is NOT a new extreme):
  - ``made_high`` : ph_high > pre_high AND ph_low >= pre_low
  - ``made_low``  : ph_low  < pre_low  AND ph_high <= pre_high
  - ``made_both`` : ph_high > pre_high AND ph_low < pre_low
  - ``neither``   : ph_high <= pre_high AND ph_low >= pre_low

RTH for NQ: 09:30–16:15 → rth_start_min=570, rth_end_min=975.
Power hour = last 60 min → ph_start_min = 975 - 60 = 915 (= 15:15).
Pre-PH window: [570, 915). PH window: [915, 975).
Resolution threshold: last bar >= rth_end - close_tolerance_min = 975 - 15 = 960 (16:00).
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.power_hour_breakout.standard import PowerHourBreakout

# ---------------------------------------------------------------------------
# Minimal InstrumentConfig (does not depend on NQ.yaml)
# ---------------------------------------------------------------------------
_NY = "America/New_York"
_RTH_SESSION = Session(start="09:30", end="16:15")
_TEST_CONFIG = InstrumentConfig(
  instrument="NQ",
  working_timezone=_NY,
  sessions={"rth": _RTH_SESSION},
  timeframes=["1h"],
  parquet_path=Path("data/NQ_1min.parquet"),
)

# Minute-of-day constants
_RTH_START = 570   # 09:30
_RTH_END = 975     # 16:15 (exclusive upper bound)
_PH_START = 915    # 15:15 — power-hour open
_RTH_LAST = 974    # 16:14, last bar before 16:15; >= 960 so resolved


def _make_ph_day(
  date: str,
  *,
  pre_high: float,
  pre_low: float,
  ph_high: float,
  ph_low: float,
  ph_open: float | None = None,
  last_mod: int = _RTH_LAST,
) -> pd.DataFrame:
  """One trading day of 1-min RTH bars with a controllable pre-PH and PH window.

  The pre-PH window [09:30, 15:15) carries ``pre_high`` / ``pre_low`` as wicks
  on dedicated bars; all other pre-PH bars are neutral at the pre-PH midpoint.

  The PH window [15:15, 16:15) carries ``ph_high`` / ``ph_low`` as wicks on
  dedicated bars. The first PH bar (15:15) always has its open set to
  ``ph_open`` (defaults to the pre-PH midpoint, which lands exactly ON the
  midpoint so ``ph_open_above`` is True).

  ``last_mod`` controls the last bar minute; set < 960 to simulate an early close
  (unresolved day).
  """
  base = pd.Timestamp(date, tz=_NY)
  pre_mid = (pre_high + pre_low) / 2.0
  if ph_open is None:
    # Default to pre-PH midpoint (ph_open >= mid → ph_open_above = True).
    ph_open = pre_mid

  records = []
  for mod in range(_RTH_START, last_mod + 1):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)

    if mod < _PH_START:
      # Pre-PH window: place extremes on two dedicated bars.
      o = pre_mid
      c = pre_mid
      if mod == _RTH_START + 1:
        hi, lo = pre_high, pre_mid
      elif mod == _RTH_START + 2:
        hi, lo = pre_mid, pre_low
      else:
        hi, lo = pre_mid, pre_mid
    else:
      # PH window.
      if mod == _PH_START:
        # First PH bar: open = ph_open; place ph_high here as the wick. Clamp the
        # bar's high/low around the open so the bar stays OHLC-valid even when
        # ph_open sits below pre_mid (the aggregated ph_high / ph_low are
        # unchanged: ph_high remains the window max and ph_low comes from the
        # second PH bar).
        o = ph_open
        c = pre_mid
        hi, lo = max(ph_high, ph_open, pre_mid), min(ph_open, pre_mid)
      elif mod == _PH_START + 1:
        # Second PH bar: place ph_low as the wick.
        o = pre_mid
        c = pre_mid
        hi, lo = pre_mid, ph_low
      else:
        o = pre_mid
        c = pre_mid
        hi, lo = pre_mid, pre_mid

    records.append({
      "timestamp": ts,
      "open": o,
      "high": hi,
      "low": lo,
      "close": c,
      "volume": 1000,
    })
  return pd.DataFrame(records)


def _make_no_ph_open_day(date: str) -> pd.DataFrame:
  """A day that has pre-PH bars but NO bar at exactly 15:15 → excluded."""
  base = pd.Timestamp(date, tz=_NY)
  records = []
  for mod in range(_RTH_START, _RTH_LAST + 1):
    if mod == _PH_START:
      # Skip the 15:15 bar explicitly.
      continue
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    records.append({
      "timestamp": ts, "open": 100.0, "high": 100.25, "low": 99.75,
      "close": 100.0, "volume": 500,
    })
  return pd.DataFrame(records)


def _make_no_ph_bars_day(date: str) -> pd.DataFrame:
  """A day that ends at 15:10 — has no PH bars at all → excluded."""
  base = pd.Timestamp(date, tz=_NY)
  records = []
  # End at mod 910 (15:10), well before ph_start=915. This also makes it
  # unresolved (910 < 960), which excludes it at the resolution stage.
  for mod in range(_RTH_START, 911):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    records.append({
      "timestamp": ts, "open": 100.0, "high": 100.25, "low": 99.75,
      "close": 100.0, "volume": 500,
    })
  return pd.DataFrame(records)


def _make_early_close_day(date: str) -> pd.DataFrame:
  """A day whose last RTH bar is 15:30 (mod 930 < 960) — unresolved, excluded."""
  base = pd.Timestamp(date, tz=_NY)
  records = []
  for mod in range(_RTH_START, 931):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    records.append({
      "timestamp": ts, "open": 100.0, "high": 100.25, "low": 99.75,
      "close": 100.0, "volume": 500,
    })
  return pd.DataFrame(records)


def _concat(days: list[pd.DataFrame]) -> pd.DataFrame:
  df = pd.concat(days, ignore_index=True)
  return df.sort_values("timestamp").reset_index(drop=True)


def _empty_df() -> pd.DataFrame:
  df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
  df["timestamp"] = pd.array([], dtype="datetime64[ns, America/New_York]")
  return df


def _stat(timeframe: str = "1h") -> PowerHourBreakout:
  return PowerHourBreakout(
    instrument="NQ",
    timeframe=timeframe,
    config=_TEST_CONFIG,
  )


def _rows(stat: PowerHourBreakout, candles: pd.DataFrame) -> dict[str, StatResultRow]:
  result = stat.compute(candles)
  return {r.outcome: r for r in result.instruments["NQ"][stat.timeframe].results}


# A canonical 5-day set:
#   pre-PH range always pre_high=110, pre_low=90 (midpoint=100, range=20)
#   Day 1: ph_high=115 > 110, ph_low=95 >= 90 → made_high
#   Day 2: ph_high=105 <= 110, ph_low=85 < 90  → made_low
#   Day 3: ph_high=115 > 110, ph_low=85 < 90   → made_both
#   Day 4: ph_high=108 <= 110, ph_low=92 >= 90  → neither
#   Day 5: ph_high=110, ph_low=90 (boundary touch, strict ineq) → neither
_CANON_SPECS = [
  dict(date="2024-01-02", pre_high=110, pre_low=90, ph_high=115, ph_low=95),
  dict(date="2024-01-03", pre_high=110, pre_low=90, ph_high=105, ph_low=85),
  dict(date="2024-01-04", pre_high=110, pre_low=90, ph_high=115, ph_low=85),
  dict(date="2024-01-05", pre_high=110, pre_low=90, ph_high=108, ph_low=92),
  dict(date="2024-01-08", pre_high=110, pre_low=90, ph_high=110, ph_low=90),
]


def _canon_candles() -> pd.DataFrame:
  return _concat([_make_ph_day(**spec) for spec in _CANON_SPECS])


# ===========================================================================
# Core classification
# ===========================================================================

def test_made_high_only() -> None:
  # ph_high=115 > pre_high=110, ph_low=95 >= pre_low=90 → made_high only.
  candles = _concat([
    _make_ph_day(date="2024-01-02", pre_high=110, pre_low=90, ph_high=115, ph_low=95),
  ])
  rows = _rows(_stat(), candles)
  # Hand-calc: 1 countable day, outcome = made_high.
  assert rows["made_high"].count == 1
  assert rows["made_low"].count == 0
  assert rows["made_both"].count == 0
  assert rows["neither"].count == 0
  assert sum(r.count for r in rows.values()) == 1


def test_made_low_only() -> None:
  # ph_high=105 <= pre_high=110, ph_low=85 < pre_low=90 → made_low only.
  candles = _concat([
    _make_ph_day(date="2024-01-02", pre_high=110, pre_low=90, ph_high=105, ph_low=85),
  ])
  rows = _rows(_stat(), candles)
  # Hand-calc: 1 countable day, outcome = made_low.
  assert rows["made_high"].count == 0
  assert rows["made_low"].count == 1
  assert rows["made_both"].count == 0
  assert rows["neither"].count == 0
  assert sum(r.count for r in rows.values()) == 1


def test_made_both() -> None:
  # ph_high=115 > 110 AND ph_low=85 < 90 → made_both.
  candles = _concat([
    _make_ph_day(date="2024-01-02", pre_high=110, pre_low=90, ph_high=115, ph_low=85),
  ])
  rows = _rows(_stat(), candles)
  # Hand-calc: 1 countable day, outcome = made_both.
  assert rows["made_high"].count == 0
  assert rows["made_low"].count == 0
  assert rows["made_both"].count == 1
  assert rows["neither"].count == 0
  assert sum(r.count for r in rows.values()) == 1


def test_neither() -> None:
  # ph_high=108 <= 110 AND ph_low=92 >= 90 → neither.
  candles = _concat([
    _make_ph_day(date="2024-01-02", pre_high=110, pre_low=90, ph_high=108, ph_low=92),
  ])
  rows = _rows(_stat(), candles)
  # Hand-calc: 1 countable day, outcome = neither.
  assert rows["made_high"].count == 0
  assert rows["made_low"].count == 0
  assert rows["made_both"].count == 0
  assert rows["neither"].count == 1
  assert sum(r.count for r in rows.values()) == 1


def test_four_outcomes_partition_countable_days() -> None:
  rows = _rows(_stat(), _canon_candles())
  # 5 countable days: 1 made_high, 1 made_low, 1 made_both, 2 neither (incl. boundary touch).
  assert rows["made_high"].count == 1
  assert rows["made_low"].count == 1
  assert rows["made_both"].count == 1
  assert rows["neither"].count == 2
  # Every row's total is the countable-day count.
  totals = {r.total for r in rows.values()}
  assert totals == {5}
  # Counts sum to total.
  assert sum(r.count for r in rows.values()) == 5


# ===========================================================================
# Strict-inequality boundary
# ===========================================================================

def test_boundary_touch_is_not_a_new_extreme() -> None:
  # ph_high == pre_high (110 == 110) and ph_low == pre_low (90 == 90):
  # strict inequality not satisfied → neither.
  candles = _concat([
    _make_ph_day(date="2024-01-02", pre_high=110, pre_low=90, ph_high=110, ph_low=90),
  ])
  rows = _rows(_stat(), candles)
  # Hand-calc: boundary touch → neither.
  assert rows["neither"].count == 1
  assert rows["made_high"].count == 0
  assert rows["made_low"].count == 0
  assert rows["made_both"].count == 0


def test_one_tick_above_is_made_high() -> None:
  # ph_high = pre_high + 0.25 (one tick above) → strict inequality satisfied → made_high.
  # ph_low = pre_low (touching, not new low).
  candles = _concat([
    _make_ph_day(date="2024-01-02", pre_high=110, pre_low=90, ph_high=110.25, ph_low=90),
  ])
  rows = _rows(_stat(), candles)
  # Hand-calc: 110.25 > 110 → made_high; 90 >= 90 (not strictly below) → high only.
  assert rows["made_high"].count == 1
  assert rows["made_low"].count == 0
  assert rows["neither"].count == 0


# ===========================================================================
# Pending / resolution discipline
# ===========================================================================

def test_early_close_day_excluded() -> None:
  # Day 1: resolved → countable.
  # Day 2: last bar at 15:30 (mod 930 < 960) → unresolved → excluded.
  candles = _concat([
    _make_ph_day(date="2024-01-02", pre_high=110, pre_low=90, ph_high=115, ph_low=95),
    _make_early_close_day("2024-01-03"),
  ])
  result = _stat().compute(candles)
  tf = result.instruments["NQ"]["1h"]
  # Only the resolved day counts; early-close day is dropped.
  assert tf.total_samples == 1
  rows = {r.outcome: r for r in tf.results}
  assert rows["made_high"].count == 1
  assert rows["made_high"].total == 1


def test_missing_ph_open_bar_excluded() -> None:
  # Day 1: resolved.
  # Day 2: has pre-PH and PH bars but the 15:15 bar is missing → excluded from countable days.
  candles = _concat([
    _make_ph_day(date="2024-01-02", pre_high=110, pre_low=90, ph_high=115, ph_low=95),
    _make_no_ph_open_day("2024-01-03"),
  ])
  result = _stat().compute(candles)
  tf = result.instruments["NQ"]["1h"]
  # Day 2 is excluded (no 15:15 open bar) despite being "resolved" (bars run to 16:14).
  assert tf.total_samples == 1


def test_no_ph_bars_day_excluded() -> None:
  # Day 2 ends before the PH window starts → excluded (also unresolved).
  candles = _concat([
    _make_ph_day(date="2024-01-02", pre_high=110, pre_low=90, ph_high=115, ph_low=95),
    _make_no_ph_bars_day("2024-01-03"),
  ])
  result = _stat().compute(candles)
  tf = result.instruments["NQ"]["1h"]
  assert tf.total_samples == 1


# ===========================================================================
# Probabilities and total_samples
# ===========================================================================

def test_probabilities_are_count_over_total() -> None:
  rows = _rows(_stat(), _canon_candles())
  # Hand-calc: 5 countable days.
  # made_high=1/5=0.2, made_low=1/5=0.2, made_both=1/5=0.2, neither=2/5=0.4.
  assert rows["made_high"].probability == pytest.approx(1 / 5)
  assert rows["made_low"].probability == pytest.approx(1 / 5)
  assert rows["made_both"].probability == pytest.approx(1 / 5)
  assert rows["neither"].probability == pytest.approx(2 / 5)


def test_probabilities_sum_to_one() -> None:
  rows = _rows(_stat(), _canon_candles())
  assert sum(r.probability for r in rows.values()) == pytest.approx(1.0)


def test_total_samples_counts_resolved_days() -> None:
  result = _stat().compute(_canon_candles())
  # 5 canonical days all resolve cleanly.
  assert result.instruments["NQ"]["1h"].total_samples == 5
  assert result.instruments["NQ"]["1h"].data_range == ["2024-01-02", "2024-01-08"]


def test_each_row_total_equals_countable_days() -> None:
  rows = _rows(_stat(), _canon_candles())
  for r in rows.values():
    assert r.total == 5


# ===========================================================================
# Baseline (random directional null via reflection)
# ===========================================================================

def test_baseline_is_deterministic_for_fixed_seed() -> None:
  stat = _stat()
  table = stat.build_day_table(_canon_candles())
  a = stat.baseline_rows(table, seed=42)
  b = stat.baseline_rows(table, seed=42)
  # Same seed → identical outcome/count pairs.
  assert [(r.outcome, r.count) for r in a] == [(r.outcome, r.count) for r in b]


def test_different_seeds_may_differ() -> None:
  stat = _stat()
  table = stat.build_day_table(_canon_candles())
  a_counts = [(r.outcome, r.count) for r in stat.baseline_rows(table, seed=42)]
  b_counts = [(r.outcome, r.count) for r in stat.baseline_rows(table, seed=99)]
  # With 5 days and a random coin flip, different seeds can produce different results.
  # (This is probabilistic; with these specific seeds and 5 days they should differ.)
  # We only assert they both stay in [0, 5].
  for _, c in a_counts:
    assert 0 <= c <= 5
  for _, c in b_counts:
    assert 0 <= c <= 5


def test_baseline_preserves_made_both_and_neither() -> None:
  # On a symmetric construction: reflection swaps made_high<->made_low but leaves
  # made_both and neither invariant (both extremes reflect symmetrically).
  stat = _stat()
  table = stat.build_day_table(_canon_candles())
  bl = {r.outcome: r for r in stat.baseline_rows(table, seed=42)}
  actual = {r.outcome: r for r in stat.compute_rows(table)}
  # Directional mass is conserved between high and low.
  assert (bl["made_high"].count + bl["made_low"].count) == (
    actual["made_high"].count + actual["made_low"].count
  )
  # made_both and neither are direction-symmetric: baseline total equals actual total.
  assert bl["made_both"].count + bl["neither"].count == (
    actual["made_both"].count + actual["neither"].count
  )


def test_baseline_no_directional_bias_on_symmetric_set() -> None:
  # Build a set where made_high and made_low are equally likely by construction.
  # With a large enough symmetric dataset, baseline_prob(made_high) ≈ baseline_prob(made_low).
  # We use 100 days: 50 made_high, 50 made_low, 0 made_both, 0 neither.
  def _next_weekday(cursor: pd.Timestamp) -> pd.Timestamp:
    """Advance one day, skipping weekends — always strictly forward so dates stay unique."""
    cursor += pd.Timedelta(days=1)
    while cursor.weekday() >= 5:
      cursor += pd.Timedelta(days=1)
    return cursor

  days = []
  date = pd.Timestamp("2024-01-01")
  for _ in range(50):
    date = _next_weekday(date)
    days.append(_make_ph_day(
      date=date.strftime("%Y-%m-%d"),
      pre_high=110, pre_low=90, ph_high=115, ph_low=92,
    ))
  for _ in range(50):
    date = _next_weekday(date)
    days.append(_make_ph_day(
      date=date.strftime("%Y-%m-%d"),
      pre_high=110, pre_low=90, ph_high=108, ph_low=85,
    ))
  candles = _concat(days)
  stat = _stat()
  table = stat.build_day_table(candles)
  bl = {r.outcome: r for r in stat.baseline_rows(table, seed=42)}
  # After reflection, made_high and made_low should be approximately equal (≈ 50 each).
  # Allow ±15 tolerance for a 100-day sample with seed=42.
  assert abs(bl["made_high"].count - bl["made_low"].count) <= 15


def test_baseline_merged_into_result_rows() -> None:
  result = _stat().compute(_canon_candles())
  for r in result.instruments["NQ"]["1h"].results:
    # baseline_n is the countable-day count carried from the baseline rows.
    assert r.baseline_n == 5


# ===========================================================================
# Slices
# ===========================================================================

def test_declared_slices_are_present() -> None:
  result = _stat().compute(_canon_candles())
  slices = result.instruments["NQ"]["1h"].slices
  # The module declares ("weekday", PowerHourOpen()) → two slice dimensions.
  assert set(slices) == {"weekday", "open"}


def test_weekday_slice_partitions_correctly() -> None:
  # 5 canonical days span Mon-Fri of the same week plus one Monday the next week.
  # 2024-01-02 = Tuesday, 2024-01-03 = Wednesday, 2024-01-04 = Thursday,
  # 2024-01-05 = Friday, 2024-01-08 = Monday.
  result = _stat().compute(_canon_candles())
  weekday_groups = result.instruments["NQ"]["1h"].slices["weekday"].groups
  # Each group's total_samples must sum to the overall total_samples.
  total_in_slices = sum(g.total_samples for g in weekday_groups.values())
  assert total_in_slices == 5
  # Each of the 5 days should appear in exactly one weekday bucket.
  assert weekday_groups["tuesday"].total_samples == 1
  assert weekday_groups["wednesday"].total_samples == 1
  assert weekday_groups["thursday"].total_samples == 1
  assert weekday_groups["friday"].total_samples == 1
  assert weekday_groups["monday"].total_samples == 1


def test_weekday_slice_counts_re_partition_overall() -> None:
  result = _stat().compute(_canon_candles())
  weekday_groups = result.instruments["NQ"]["1h"].slices["weekday"].groups
  # Per-weekday counts of made_high must sum to the overall made_high count.
  # Overall made_high = 1 (2024-01-02, a Tuesday).
  high_total = sum(
    next(r.count for r in g.results if r.outcome == "made_high")
    for g in weekday_groups.values()
  )
  assert high_total == 1


def test_open_slice_groups_sum_to_total() -> None:
  # The "open" slice splits on ph_open_above (above/below pre-PH midpoint).
  # Default ph_open equals pre-PH midpoint → ph_open_above = True (>= mid) → "above".
  result = _stat().compute(_canon_candles())
  open_groups = result.instruments["NQ"]["1h"].slices["open"].groups
  total_in_slices = sum(g.total_samples for g in open_groups.values())
  # All 5 canonical days use the default ph_open = midpoint → all "above".
  assert total_in_slices == 5


def test_open_slice_above_when_ph_open_at_midpoint() -> None:
  # ph_open = pre-PH midpoint (100 when pre_high=110, pre_low=90).
  # ph_open >= mid → ph_open_above = True → classified as "above".
  candles = _concat([
    _make_ph_day(
      date="2024-01-02", pre_high=110, pre_low=90, ph_high=115, ph_low=95,
      ph_open=100,  # exactly at midpoint
    ),
  ])
  result = _stat().compute(candles)
  open_groups = result.instruments["NQ"]["1h"].slices["open"].groups
  assert "above" in open_groups
  assert open_groups["above"].total_samples == 1
  assert "below" not in open_groups


def test_open_slice_below_when_ph_open_below_midpoint() -> None:
  # pre_high=110, pre_low=90 → midpoint=100. ph_open=95 < 100 → ph_open_above=False → "below".
  candles = _concat([
    _make_ph_day(
      date="2024-01-02", pre_high=110, pre_low=90, ph_high=115, ph_low=95,
      ph_open=95,  # below midpoint
    ),
  ])
  result = _stat().compute(candles)
  open_groups = result.instruments["NQ"]["1h"].slices["open"].groups
  assert "below" in open_groups
  assert open_groups["below"].total_samples == 1
  assert "above" not in open_groups


def test_open_slice_above_when_ph_open_above_midpoint() -> None:
  # ph_open=105 > 100 (midpoint) → ph_open_above=True → "above".
  candles = _concat([
    _make_ph_day(
      date="2024-01-02", pre_high=110, pre_low=90, ph_high=115, ph_low=95,
      ph_open=105,
    ),
  ])
  result = _stat().compute(candles)
  open_groups = result.instruments["NQ"]["1h"].slices["open"].groups
  assert "above" in open_groups
  assert open_groups["above"].total_samples == 1


def test_open_slice_splits_mixed_days() -> None:
  # Day 1: ph_open=105 > 100 → above.
  # Day 2: ph_open=95 < 100 → below.
  # Both are made_high days (ph_high=115 > 110, ph_low=95 >= 90).
  candles = _concat([
    _make_ph_day(
      date="2024-01-02", pre_high=110, pre_low=90, ph_high=115, ph_low=95,
      ph_open=105,
    ),
    _make_ph_day(
      date="2024-01-03", pre_high=110, pre_low=90, ph_high=115, ph_low=95,
      ph_open=95,
    ),
  ])
  result = _stat().compute(candles)
  open_groups = result.instruments["NQ"]["1h"].slices["open"].groups
  assert open_groups["above"].total_samples == 1
  assert open_groups["below"].total_samples == 1
  assert open_groups["above"].total_samples + open_groups["below"].total_samples == 2


# ===========================================================================
# Empty input
# ===========================================================================

def test_empty_input_yields_empty_day_table() -> None:
  table = _stat().build_day_table(_empty_df())
  assert table.empty


def test_empty_input_yields_zero_count_rows_no_crash() -> None:
  result = _stat().compute(_empty_df())
  tf = result.instruments["NQ"]["1h"]
  assert tf.total_samples == 0
  assert tf.data_range == []
  for r in tf.results:
    assert r.count == 0
    assert r.total == 0
    assert r.probability == 0.0


def test_empty_input_slices_empty() -> None:
  result = _stat().compute(_empty_df())
  slices = result.instruments["NQ"]["1h"].slices
  # Slicers must not crash on empty input; groups should be empty dicts.
  for _, sr in slices.items():
    assert sr.groups == {}


# ===========================================================================
# build_day_table columns
# ===========================================================================

def test_day_table_columns() -> None:
  table = _stat().build_day_table(_canon_candles())
  expected_cols = {"pre_high", "pre_low", "ph_high", "ph_low", "ph_open_above"}
  assert expected_cols.issubset(set(table.columns))


def test_day_table_pre_high_low_values() -> None:
  # pre_high=110, pre_low=90 for all canonical days.
  table = _stat().build_day_table(_canon_candles())
  assert (table["pre_high"] == 110.0).all()
  assert (table["pre_low"] == 90.0).all()


def test_day_table_ph_open_above_at_midpoint() -> None:
  # Default ph_open = pre-PH midpoint (100). mid=100. ph_open >= mid → True.
  table = _stat().build_day_table(_canon_candles())
  assert table["ph_open_above"].all()


def test_day_table_index_is_normalized_date() -> None:
  # The day table index should be normalized dates (no time component).
  table = _stat().build_day_table(_canon_candles())
  for idx in table.index:
    assert idx == idx.normalize()


# ===========================================================================
# Invalid constructor arguments
# ===========================================================================

def test_invalid_timeframe_raises() -> None:
  with pytest.raises(ValueError, match="Unsupported timeframe"):
    PowerHourBreakout(instrument="NQ", timeframe="30min", config=_TEST_CONFIG)


def test_unsupported_timeframe_15min_raises() -> None:
  with pytest.raises(ValueError, match="Unsupported timeframe"):
    PowerHourBreakout(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)


# ===========================================================================
# StatRunResult validation
# ===========================================================================

def test_compute_returns_valid_stat_run_result() -> None:
  result = _stat().compute(_canon_candles())
  # Pydantic validation: re-validate round-tripped dict (no ValidationError raised).
  StatRunResult.model_validate(result.model_dump())


def test_result_has_correct_stat_name() -> None:
  result = _stat().compute(_canon_candles())
  assert result.stat_name == "power_hour_breakout"


def test_result_has_correct_condition_key() -> None:
  rows = _rows(_stat(), _canon_candles())
  for r in rows.values():
    assert r.condition == "power_hour"


def test_result_has_correct_outcome_keys() -> None:
  rows = _rows(_stat(), _canon_candles())
  assert set(rows.keys()) == {"made_high", "made_low", "made_both", "neither"}


def test_result_has_expected_slice_dimensions() -> None:
  result = _stat().compute(_canon_candles())
  slices = result.instruments["NQ"]["1h"].slices
  assert "weekday" in slices
  assert "open" in slices


# ===========================================================================
# JSON round-trip
# ===========================================================================

def test_result_validates_and_writes(tmp_path: Path) -> None:
  result = _stat().compute(_canon_candles())
  out = write_results(result, results_dir=tmp_path)
  assert out.exists()
  data = json.loads(out.read_text(encoding="utf-8"))
  assert data["stat_name"] == "power_hour_breakout"
  assert data["title"]["en"] == "Power Hour Breakout"
  # Re-validate the written payload through the Pydantic model.
  StatRunResult.model_validate(data)
