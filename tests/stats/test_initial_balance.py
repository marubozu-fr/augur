"""Tests for stats.initial_balance.standard.

All data is synthetic — no real market files required. Expected values are
hand-calculated before each assertion.

Framing (single condition ``ib``): given the initial balance formed from the first
``ib_period`` minutes of the session, which direction does price break during the
REST of the session? Four mutually exclusive outcomes partition every countable
day (strict inequality defines a break — touching the level is NOT a break):
  - ``broke_high``: post_high  >  ib_high AND post_low  >= ib_low
  - ``broke_low`` : post_low   <  ib_low  AND post_high <= ib_high
  - ``broke_both``: post_high  >  ib_high AND post_low  <  ib_low
  - ``neither``   : post_high  <= ib_high AND post_low  >= ib_low

With ``breakout_criteria='wick'`` (default) the breakout window's intraday
extremes are used; with ``'close'`` its extreme bar CLOSES are used instead.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.initial_balance.standard import InitialBalanceBreakout

# ---------------------------------------------------------------------------
# Minimal InstrumentConfig (does not depend on NQ.yaml)
# ---------------------------------------------------------------------------
_NY = "America/New_York"
_RTH_SESSION = Session(start="09:30", end="16:15")
_TEST_CONFIG = InstrumentConfig(
  instrument="NQ",
  working_timezone=_NY,
  sessions={"rth": _RTH_SESSION},
  timeframes=["30min", "1h"],
  parquet_path=Path("data/NQ_1min.parquet"),
)

_RTH_START = 570   # 09:30
_RTH_LAST = 974    # 16:14 (last bar before 16:15); resolved needs last mod >= 960


def _make_ib_day(
  date: str,
  *,
  ib_high: float,
  ib_low: float,
  post_high: float,
  post_low: float,
  session_open: float | None = None,
  session_close: float | None = None,
  post_close_high: float | None = None,
  post_close_low: float | None = None,
  ib_minutes: int = 30,
  last_mod: int = _RTH_LAST,
) -> pd.DataFrame:
  """One trading day of 1-min RTH bars with a controllable initial balance.

  The IB window ``[09:30, 09:30 + ib_minutes)`` carries ``ib_high`` / ``ib_low``
  as wicks; the breakout window ``[09:30 + ib_minutes, last_mod]`` carries
  ``post_high`` / ``post_low`` as wicks and ``post_close_high`` / ``post_close_low``
  as bar CLOSES (defaulting to the IB midpoint, i.e. no close-based break).

  ``session_open`` defaults to the IB midpoint (an open inside the range);
  ``session_close`` (the last bar's close) likewise, so colour is neutral unless set.
  """
  base = pd.Timestamp(date, tz=_NY)
  ib_mid = (ib_high + ib_low) / 2.0
  if session_open is None:
    session_open = ib_mid
  if session_close is None:
    session_close = ib_mid
  if post_close_high is None:
    post_close_high = ib_mid
  if post_close_low is None:
    post_close_low = ib_mid

  ib_end = _RTH_START + ib_minutes
  records = []
  for mod in range(_RTH_START, last_mod + 1):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    if mod < ib_end:
      # IB window: open on the first bar, extremes as wicks on the next two bars.
      o = session_open if mod == _RTH_START else ib_mid
      c = ib_mid
      hi = ib_high if mod == _RTH_START + 1 else max(o, c)
      lo = ib_low if mod == _RTH_START + 2 else min(o, c)
    else:
      # Breakout window: wick extremes, close extremes, then the session close.
      o = ib_mid
      c = ib_mid
      if mod == ib_end:
        hi, lo = post_high, ib_mid
      elif mod == ib_end + 1:
        hi, lo = ib_mid, post_low
      elif mod == ib_end + 2:
        c = post_close_high
        hi, lo = max(ib_mid, c), ib_mid
      elif mod == ib_end + 3:
        c = post_close_low
        hi, lo = ib_mid, min(ib_mid, c)
      else:
        hi, lo = max(o, c), min(o, c)
      if mod == last_mod:
        c = session_close
        hi, lo = max(hi, c), min(lo, c)
    records.append({
      "timestamp": ts,
      "open": o,
      "high": hi,
      "low": lo,
      "close": c,
      "volume": 1000,
    })
  return pd.DataFrame(records)


def _make_truncated_day(date: str) -> pd.DataFrame:
  """A day whose last RTH bar is 09:50 (mod 590 < 960) — unresolved, excluded."""
  base = pd.Timestamp(date, tz=_NY)
  records = []
  for mod in range(_RTH_START, 591):
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


def _stat(timeframe: str = "30min", breakout_criteria: str = "wick") -> InitialBalanceBreakout:
  return InitialBalanceBreakout(
    instrument="NQ",
    timeframe=timeframe,
    config=_TEST_CONFIG,
    breakout_criteria=breakout_criteria,
  )


def _row(result: StatRunResult, outcome: str, timeframe: str = "30min") -> StatResultRow:
  for r in result.instruments["NQ"][timeframe].results:
    if r.condition == "ib" and r.outcome == outcome:
      return r
  raise KeyError(("ib", outcome))


def _rows(stat: InitialBalanceBreakout, candles: pd.DataFrame) -> dict[str, StatResultRow]:
  result = stat.compute(candles)
  return {r.outcome: r for r in result.instruments["NQ"][stat.timeframe].results}


# A canonical 5-day set for the 30-min IB (ib_high=110, ib_low=90):
#   broke_high, broke_low, broke_both, neither, boundary-touch (neither).
_CANON = [
  # broke_high: post_high 115 > 110, post_low 95 >= 90
  dict(date="2024-01-02", ib_high=110, ib_low=90, post_high=115, post_low=95),
  # broke_low: post_high 105 <= 110, post_low 85 < 90
  dict(date="2024-01-03", ib_high=110, ib_low=90, post_high=105, post_low=85),
  # broke_both: post_high 115 > 110, post_low 85 < 90
  dict(date="2024-01-04", ib_high=110, ib_low=90, post_high=115, post_low=85),
  # neither: post_high 108 <= 110, post_low 92 >= 90
  dict(date="2024-01-05", ib_high=110, ib_low=90, post_high=108, post_low=92),
  # boundary touch (not a break): post_high == 110, post_low == 90 -> neither
  dict(date="2024-01-08", ib_high=110, ib_low=90, post_high=110, post_low=90),
]


def _canon_candles() -> pd.DataFrame:
  return _concat([_make_ib_day(**spec) for spec in _CANON])


# ===========================================================================
# Core classification
# ===========================================================================
def test_four_outcomes_partition_countable_days() -> None:
  rows = _rows(_stat(), _canon_candles())
  # 5 countable days: 1 high, 1 low, 1 both, 2 neither (incl. the boundary touch).
  assert rows["broke_high"].count == 1
  assert rows["broke_low"].count == 1
  assert rows["broke_both"].count == 1
  assert rows["neither"].count == 2
  # Every row's total is the countable-day count; counts sum to total.
  totals = {r.total for r in rows.values()}
  assert totals == {5}
  assert sum(r.count for r in rows.values()) == 5


def test_boundary_touch_is_not_a_break() -> None:
  # A day that only touches ib_high / ib_low exactly classifies as neither.
  candles = _concat([
    _make_ib_day(date="2024-01-02", ib_high=110, ib_low=90, post_high=110, post_low=90),
  ])
  rows = _rows(_stat(), candles)
  assert rows["neither"].count == 1
  assert rows["broke_high"].count == 0
  assert rows["broke_low"].count == 0
  assert rows["broke_both"].count == 0


def test_probabilities_are_count_over_total() -> None:
  rows = _rows(_stat(), _canon_candles())
  assert rows["broke_high"].probability == pytest.approx(1 / 5)
  assert rows["broke_low"].probability == pytest.approx(1 / 5)
  assert rows["broke_both"].probability == pytest.approx(1 / 5)
  assert rows["neither"].probability == pytest.approx(2 / 5)
  assert sum(r.probability for r in rows.values()) == pytest.approx(1.0)


def test_total_samples_counts_resolved_days() -> None:
  result = _stat().compute(_canon_candles())
  assert result.instruments["NQ"]["30min"].total_samples == 5
  assert result.instruments["NQ"]["30min"].data_range == ["2024-01-02", "2024-01-08"]


# ===========================================================================
# Breakout criteria: wick vs close
# ===========================================================================
def test_wick_break_does_not_count_as_close_break() -> None:
  # Wick pokes above ib_high (115 > 110) but every post bar CLOSES inside the
  # range (post_close_high = 100). Wick criteria -> broke_high; close -> neither.
  candles = _concat([
    _make_ib_day(
      date="2024-01-02", ib_high=110, ib_low=90,
      post_high=115, post_low=95, post_close_high=100,
    ),
  ])
  assert _rows(_stat(breakout_criteria="wick"), candles)["broke_high"].count == 1
  assert _rows(_stat(breakout_criteria="close"), candles)["neither"].count == 1


def test_close_beyond_level_counts_as_close_break() -> None:
  # A post bar CLOSES above ib_high (post_close_high = 112 > 110), so the close
  # criteria classifies broke_high. The wick reaches the same value so wick agrees.
  candles = _concat([
    _make_ib_day(
      date="2024-01-02", ib_high=110, ib_low=90,
      post_high=112, post_low=95, post_close_high=112,
    ),
  ])
  rows = _rows(_stat(breakout_criteria="close"), candles)
  assert rows["broke_high"].count == 1
  assert rows["broke_low"].count == 0


def test_invalid_breakout_criteria_raises() -> None:
  with pytest.raises(ValueError, match="breakout_criteria"):
    _stat(breakout_criteria="bogus")


def test_invalid_timeframe_raises() -> None:
  with pytest.raises(ValueError, match="Unsupported timeframe"):
    _stat(timeframe="7min")


def test_15min_is_not_a_supported_ib_length() -> None:
  # The initial balance is conventionally >= 30 minutes; 15min is rejected.
  with pytest.raises(ValueError, match="Unsupported timeframe"):
    _stat(timeframe="15min")


# ===========================================================================
# IB window length per timeframe
# ===========================================================================
def test_ib_window_length_tracks_timeframe() -> None:
  # The same day, read with a 1h IB, includes a 10:15 spike in the INITIAL
  # balance (so it is part of ib_high, not a breakout). With a 30-min IB the same
  # 10:15 spike falls in the breakout window and breaks the high.
  spike_day = _make_ib_day(
    date="2024-01-02", ib_high=110, ib_low=90,
    post_high=115, post_low=95, ib_minutes=30,
  )
  candles = _concat([spike_day])
  # 30-min IB: ib_high stays 110, the 115 spike at 10:00 breaks it.
  assert _rows(_stat("30min"), candles)["broke_high"].count == 1
  # 1h IB: the window now swallows the 10:00 spike, so ib_high = 115 and the
  # post window (post_low 95 only) holds the range -> neither.
  assert _rows(_stat("1h"), candles)["neither"].count == 1


def test_ib_size_column_is_range_width() -> None:
  table = _stat().build_day_table(_canon_candles())
  # ib_high - ib_low = 110 - 90 = 20 on every canonical day.
  assert (table["ib_size"] == 20.0).all()


# ===========================================================================
# Baseline (random directional null via reflection)
# ===========================================================================
def test_baseline_preserves_direction_symmetric_outcomes() -> None:
  # Reflection swaps broke_high <-> broke_low but leaves broke_both and neither
  # unchanged, so their baseline counts equal the actual counts exactly.
  stat = _stat()
  table = stat.build_day_table(_canon_candles())
  bl = {r.outcome: r for r in stat.baseline_rows(table, seed=42)}
  actual = {r.outcome: r for r in stat.compute_rows(table)}
  assert bl["broke_both"].count == actual["broke_both"].count
  assert bl["neither"].count == actual["neither"].count
  # Directional mass is conserved, only redistributed between high and low.
  assert (bl["broke_high"].count + bl["broke_low"].count) == (
    actual["broke_high"].count + actual["broke_low"].count
  )


def test_baseline_is_deterministic_for_fixed_seed() -> None:
  stat = _stat()
  table = stat.build_day_table(_canon_candles())
  a = stat.baseline_rows(table, seed=42)
  b = stat.baseline_rows(table, seed=42)
  assert [(r.outcome, r.count) for r in a] == [(r.outcome, r.count) for r in b]


def test_baseline_merged_into_result_rows() -> None:
  result = _stat().compute(_canon_candles())
  for r in result.instruments["NQ"]["30min"].results:
    # baseline_n is the countable-day count carried from the baseline rows.
    assert r.baseline_n == 5


# ===========================================================================
# Pending / resolution discipline
# ===========================================================================
def test_unresolved_day_is_excluded() -> None:
  candles = _concat([
    _make_ib_day(date="2024-01-02", ib_high=110, ib_low=90, post_high=115, post_low=95),
    _make_truncated_day("2024-01-03"),
  ])
  result = _stat().compute(candles)
  # Only the resolved day counts; the truncated day is dropped entirely.
  assert result.instruments["NQ"]["30min"].total_samples == 1
  rows = {r.outcome: r for r in result.instruments["NQ"]["30min"].results}
  assert rows["broke_high"].count == 1
  assert rows["broke_high"].total == 1


def test_empty_input_yields_zero_rows() -> None:
  result = _stat().compute(_empty_df())
  tf = result.instruments["NQ"]["30min"]
  assert tf.total_samples == 0
  assert tf.data_range == []
  for r in tf.results:
    assert r.count == 0
    assert r.total == 0
    assert r.probability == 0.0


# ===========================================================================
# Slices
# ===========================================================================
def test_declared_slices_are_present() -> None:
  result = _stat().compute(_canon_candles())
  slices = result.instruments["NQ"]["30min"].slices
  assert set(slices) == {
    "weekday",
    "close",
    "prev_candle",
    "overnight",
    "size",
    "size_pct",
    "levels",
  }


def test_close_slice_splits_by_session_colour() -> None:
  # Two green days (close > open) and one red day (close < open), all broke_high.
  candles = _concat([
    _make_ib_day(date="2024-01-02", ib_high=110, ib_low=90, post_high=115,
                 post_low=95, session_open=100, session_close=105),
    _make_ib_day(date="2024-01-03", ib_high=110, ib_low=90, post_high=115,
                 post_low=95, session_open=100, session_close=104),
    _make_ib_day(date="2024-01-04", ib_high=110, ib_low=90, post_high=115,
                 post_low=95, session_open=100, session_close=95),
  ])
  groups = _stat().compute(candles).instruments["NQ"]["30min"].slices["close"].groups
  assert groups["green"].total_samples == 2
  assert groups["red"].total_samples == 1


def test_size_slice_buckets_by_ib_size() -> None:
  # Four distinct IB sizes -> quartile buckets q1..q4, one day each.
  candles = _concat([
    _make_ib_day(date="2024-01-02", ib_high=105, ib_low=95, post_high=120, post_low=80),
    _make_ib_day(date="2024-01-03", ib_high=110, ib_low=90, post_high=120, post_low=80),
    _make_ib_day(date="2024-01-04", ib_high=115, ib_low=85, post_high=120, post_low=80),
    _make_ib_day(date="2024-01-05", ib_high=120, ib_low=80, post_high=130, post_low=70),
  ])
  groups = _stat().compute(candles).instruments["NQ"]["30min"].slices["size"].groups
  assert sum(g.total_samples for g in groups.values()) == 4


# ---------------------------------------------------------------------------
# New extension slices: overnight, size_pct, levels
# ---------------------------------------------------------------------------
def test_ib_size_pct_column_is_size_over_open() -> None:
  # session_open defaults to the IB midpoint (100 here), so ib_size_pct == ib_size.
  table = _stat().build_day_table(_canon_candles())
  # ib_size = 20, open = 100 -> 20%.
  assert (table["ib_size_pct"] == 20.0).all()


def test_extension_column_is_furthest_break_past_balance() -> None:
  # extension = max(post_high - ib_high, ib_low - post_low, 0), criteria-aware (wick).
  table = _stat().build_day_table(_canon_candles())
  ext = {d.strftime("%Y-%m-%d"): v for d, v in table["extension"].items()}
  assert ext["2024-01-02"] == pytest.approx(5.0)   # broke_high: 115 - 110
  assert ext["2024-01-03"] == pytest.approx(5.0)   # broke_low: 90 - 85
  assert ext["2024-01-04"] == pytest.approx(5.0)   # broke_both: max(5, 5)
  assert ext["2024-01-05"] == pytest.approx(0.0)   # neither: stayed inside
  assert ext["2024-01-08"] == pytest.approx(0.0)   # boundary touch


def test_overnight_slice_splits_by_gap_direction() -> None:
  # Day 1 has no prior close (excluded). Day 2 opens above day 1's close (green
  # overnight); day 3 opens below day 2's close (red overnight). All broke_high.
  candles = _concat([
    _make_ib_day(date="2024-01-02", ib_high=110, ib_low=90, post_high=115,
                 post_low=95, session_open=100, session_close=100),
    _make_ib_day(date="2024-01-03", ib_high=110, ib_low=90, post_high=115,
                 post_low=95, session_open=105, session_close=100),
    _make_ib_day(date="2024-01-04", ib_high=110, ib_low=90, post_high=115,
                 post_low=95, session_open=95, session_close=100),
  ])
  groups = _stat().compute(candles).instruments["NQ"]["30min"].slices["overnight"].groups
  assert groups["green"].total_samples == 1   # 2024-01-03 opened above prior close
  assert groups["red"].total_samples == 1     # 2024-01-04 opened below prior close
  # The first day has no prior close and is excluded from both groups.
  assert sum(g.total_samples for g in groups.values()) == 2


def test_size_pct_slice_uses_preset_price_relative_bands() -> None:
  # session_open defaults to the IB midpoint (100), so ib_size_pct == ib_size.
  # Sizes 0.1 / 0.5 / 1.2 land in the <0.2 / 0.4–0.6 / >0.9 preset bands.
  candles = _concat([
    _make_ib_day(date="2024-01-02", ib_high=100.05, ib_low=99.95,
                 post_high=101, post_low=99),
    _make_ib_day(date="2024-01-03", ib_high=100.25, ib_low=99.75,
                 post_high=101, post_low=99),
    _make_ib_day(date="2024-01-04", ib_high=100.6, ib_low=99.4,
                 post_high=102, post_low=98),
  ])
  groups = _stat().compute(candles).instruments["NQ"]["30min"].slices["size_pct"].groups
  # Three distinct preset bands, one day each (empty bands are omitted).
  assert sum(g.total_samples for g in groups.values()) == 3
  assert len(groups) == 3


def test_levels_slice_buckets_by_extension_multiple() -> None:
  # ib_size = 20. Extensions 5 / 35 give ratios 0.25 / 1.75 -> bands [0,0.5) and
  # [1.5,2). A neither day (extension 0) falls in the first band.
  candles = _concat([
    _make_ib_day(date="2024-01-02", ib_high=110, ib_low=90, post_high=115, post_low=95),
    _make_ib_day(date="2024-01-03", ib_high=110, ib_low=90, post_high=145, post_low=95),
    _make_ib_day(date="2024-01-04", ib_high=110, ib_low=90, post_high=108, post_low=92),
  ])
  groups = _stat().compute(candles).instruments["NQ"]["30min"].slices["levels"].groups
  assert sum(g.total_samples for g in groups.values()) == 3
  # The 1.75x extension lands in its own higher band, distinct from the small ones.
  assert groups["1_5_2x"].total_samples == 1


# ===========================================================================
# JSON round-trip
# ===========================================================================
def test_result_validates_and_writes(tmp_path: Path) -> None:
  result = _stat().compute(_canon_candles())
  out = write_results(result, results_dir=tmp_path)
  assert out.exists()
  data = json.loads(out.read_text(encoding="utf-8"))
  assert data["stat_name"] == "initial_balance"
  assert data["title"]["en"] == "Initial Balance Breakout"
  # Re-validate the written payload through the Pydantic model.
  StatRunResult.model_validate(data)
