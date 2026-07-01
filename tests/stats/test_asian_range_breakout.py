"""Tests for stats.asian_range_breakout.standard.

All data is synthetic — no real market files required. Expected values are
hand-calculated before each assertion.

Framing: a single ``asian_range`` condition with four mutually exclusive
outcomes (``broke_high`` / ``broke_low`` / ``broke_both`` / ``neither``) that
partition every countable day. The "range" is the PRIOR Asian (Tokyo) session's
high/low; the breakout window is the whole RTH session. Strict inequality defines
a break (touching the level is not a break).

The Asian session crosses midnight (18:00 ET prev evening → 03:00 ET). Each
synthetic session carries one Asian bar plus RTH bars 09:30–16:14. The Asian bar
is placed at 01:00 on the RTH date by default (mod 60 < 180 = asia_end_min →
same-day ``_cycle`` attribution). A flag (``asian_prev_evening``) places it at
20:00 on the prior calendar day instead (mod 1200 >= 1080 = asia_start_min →
next-day ``_cycle`` = the RTH date), which exercises the cross-midnight path
through ``session_bars``.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.asian_range_breakout.standard import AsianRangeBreakout
from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session

# ---------------------------------------------------------------------------
# Minimal InstrumentConfig (does not depend on NQ.yaml)
# ---------------------------------------------------------------------------
_NY = "America/New_York"
_RTH_SESSION = Session(start="09:30", end="16:15")
_ASIA_SESSION = Session(start="18:00", end="03:00")
_TEST_CONFIG = InstrumentConfig(
  instrument="NQ",
  working_timezone=_NY,
  sessions={"rth": _RTH_SESSION, "asia": _ASIA_SESSION},
  timeframes=["daily"],
  parquet_path=Path("data/NQ_1min.parquet"),
)

_RTH_START = 570   # 09:30
_RTH_LAST = 974    # 16:14 (last bar before 16:15); resolved needs last mod >= 960


def _make_session(
  date: str,
  ar_high: float,
  ar_low: float,
  day_high: float,
  day_low: float,
  session_open: float | None = None,
  session_close: float | None = None,
  asian_prev_evening: bool = False,
) -> pd.DataFrame:
  """One trading session: one Asian bar plus RTH bars (09:30–16:14).

  The Asian bar carries ``ar_high`` / ``ar_low``. By default it sits at 01:00
  on ``date`` (mod 60 < 180 = asia_end_min → same-day cycle = ``date``). With
  ``asian_prev_evening=True`` it sits at 20:00 on the prior calendar day (mod
  1200 >= 1080 = asia_start_min → next-day cycle = ``date``), exercising the
  cross-midnight attribution inside ``session_bars``.

  The 09:30 bar carries ``day_high`` / ``day_low`` (RTH wick extremes) and
  ``session_open``; the 16:14 bar carries ``session_close``. Intermediate RTH
  bars sit at the midpoint so they never affect the extremes.
  ``session_open`` / ``session_close`` default to the RTH midpoint (flat
  session). Callers must keep ``session_close`` within ``[day_low, day_high]``
  to preserve OHLC consistency.
  """
  base = pd.Timestamp(date, tz=_NY)
  mid = (day_high + day_low) / 2.0
  if session_open is None:
    session_open = mid
  if session_close is None:
    session_close = mid

  records: list[dict] = []

  # Asian bar.
  if asian_prev_evening:
    # 20:00 on the prior day: mod 1200 >= 1080 (asia_start=18:00)
    # → session_bars tags _cycle = _date + 1 day = ``date``.
    asia_ts = (base - pd.Timedelta(days=1)).replace(
      hour=20, minute=0, second=0, microsecond=0
    )
  else:
    # 01:00 on ``date``: mod 60 < 180 (asia_end=03:00)
    # → session_bars tags _cycle = _date = ``date``.
    asia_ts = base.replace(hour=1, minute=0, second=0, microsecond=0)
  records.append({
    "timestamp": asia_ts,
    "open": ar_low,
    "high": ar_high,
    "low": ar_low,
    "close": ar_high,
    "volume": 100,
  })

  # RTH bars 09:30–16:14.
  for mod in range(_RTH_START, _RTH_LAST + 1):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    if mod == _RTH_START:
      o, hi, lo, c = session_open, day_high, day_low, mid
    elif mod == _RTH_LAST:
      # Keep OHLC-consistent: RTH extremes are driven by the 09:30 bar;
      # session_close is always within [day_low, day_high].
      o, hi, lo, c = mid, max(mid, session_close), min(mid, session_close), session_close
    else:
      o, hi, lo, c = mid, mid, mid, mid
    records.append({
      "timestamp": ts, "open": o, "high": hi, "low": lo, "close": c, "volume": 1000,
    })

  return pd.DataFrame(records)


def make_candles(sessions: list[dict]) -> pd.DataFrame:
  """Concatenate per-session specs into a single sorted 1-min OHLCV DataFrame."""
  frames = [_make_session(**s) for s in sessions]
  df = pd.concat(frames, ignore_index=True)
  return df.sort_values("timestamp").reset_index(drop=True)


def _make_truncated_session(date: str) -> pd.DataFrame:
  """A session whose last RTH bar is 09:50 (mod 590 < 960) — unresolved, excluded."""
  base = pd.Timestamp(date, tz=_NY)
  records: list[dict] = [{
    "timestamp": base.replace(hour=1, minute=0),
    "open": 90.0, "high": 110.0, "low": 90.0, "close": 110.0, "volume": 100,
  }]
  for mod in range(_RTH_START, 591):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    records.append({
      "timestamp": ts, "open": 100.0, "high": 120.0, "low": 80.0,
      "close": 100.0, "volume": 500,
    })
  return pd.DataFrame(records)


def _stat(breakout_criteria: str = "wick") -> AsianRangeBreakout:
  return AsianRangeBreakout(
    instrument="NQ", config=_TEST_CONFIG, breakout_criteria=breakout_criteria
  )


def _row(result: StatRunResult, outcome: str) -> StatResultRow:
  for r in result.instruments["NQ"]["daily"].results:
    if r.condition == "asian_range" and r.outcome == outcome:
      return r
  raise KeyError(outcome)


# ===========================================================================
# Core 4-outcome partition
#
# Asian range fixed at [90, 110] (ar_size = 20) for every day. The RTH
# extremes select each outcome (strict break):
#   broke_high : day_high 115 > 110, day_low 95 >= 90        (up only)
#   broke_low  : day_high 105 <= 110, day_low 85 < 90        (down only)
#   broke_both : day_high 115 > 110, day_low 85 < 90         (both)
#   neither    : day_high 108 <= 110, day_low 92 >= 90       (inside)
# Two of each → every outcome count = 2, total = 8.
# ===========================================================================

_SEQ = [
  # broke_high: high 115 > 110, low 95 >= 90 (up only)
  {"date": "2024-01-02", "ar_high": 110, "ar_low": 90, "day_high": 115, "day_low": 95},
  # broke_low: high 105 <= 110, low 85 < 90 (down only)
  {"date": "2024-01-03", "ar_high": 110, "ar_low": 90, "day_high": 105, "day_low": 85},
  # broke_both: high 115 > 110, low 85 < 90 (both)
  {"date": "2024-01-04", "ar_high": 110, "ar_low": 90, "day_high": 115, "day_low": 85},
  # neither: high 108 <= 110, low 92 >= 90 (inside)
  {"date": "2024-01-05", "ar_high": 110, "ar_low": 90, "day_high": 108, "day_low": 92},
  # broke_high
  {"date": "2024-01-08", "ar_high": 110, "ar_low": 90, "day_high": 115, "day_low": 95},
  # broke_low
  {"date": "2024-01-09", "ar_high": 110, "ar_low": 90, "day_high": 105, "day_low": 85},
  # broke_both
  {"date": "2024-01-10", "ar_high": 110, "ar_low": 90, "day_high": 115, "day_low": 85},
  # neither
  {"date": "2024-01-11", "ar_high": 110, "ar_low": 90, "day_high": 108, "day_low": 92},
]


def test_total_samples_counts_all_resolved_days() -> None:
  """All 8 sessions are resolved AND have a prior Asian range → 8 countable."""
  result = _stat().compute(make_candles(_SEQ))
  assert result.instruments["NQ"]["daily"].total_samples == 8


def test_four_outcomes_partition() -> None:
  """Each outcome count = 2, total = 8, probabilities = 0.25 each."""
  result = _stat().compute(make_candles(_SEQ))
  for outcome in ("broke_high", "broke_low", "broke_both", "neither"):
    row = _row(result, outcome)
    assert row.count == 2, outcome
    assert row.total == 8, outcome
    assert row.probability == pytest.approx(0.25), outcome


def test_outcome_counts_sum_to_total() -> None:
  """The four mutually exclusive outcomes sum to the countable total."""
  result = _stat().compute(make_candles(_SEQ))
  rows = result.instruments["NQ"]["daily"].results
  assert sum(r.count for r in rows) == 8
  assert {r.total for r in rows} == {8}


def test_probabilities_sum_to_one() -> None:
  """Probabilities of the four outcomes sum exactly to 1 (complete partition)."""
  # Hand-calc: 2/8 + 2/8 + 2/8 + 2/8 = 8/8 = 1.0.
  result = _stat().compute(make_candles(_SEQ))
  probs = [
    _row(result, out).probability
    for out in ("broke_high", "broke_low", "broke_both", "neither")
  ]
  assert sum(probs) == pytest.approx(1.0)


def test_four_rows_only() -> None:
  """Exactly the four breakout outcomes are reported under one condition."""
  result = _stat().compute(make_candles(_SEQ))
  rows = result.instruments["NQ"]["daily"].results
  assert {(r.condition, r.outcome) for r in rows} == {
    ("asian_range", "broke_high"),
    ("asian_range", "broke_low"),
    ("asian_range", "broke_both"),
    ("asian_range", "neither"),
  }


def test_data_range() -> None:
  """data_range spans from the first to the last resolved session date."""
  result = _stat().compute(make_candles(_SEQ))
  assert result.instruments["NQ"]["daily"].data_range == ["2024-01-02", "2024-01-11"]


# ===========================================================================
# Strict-break boundary: touching a level exactly is NOT a break
# ===========================================================================

def test_touching_high_level_is_not_a_break() -> None:
  """day_high == ar_high exactly → NOT a break (strict inequality required)."""
  # day_high 110 == ar_high 110, day_low 92 >= ar_low 90 → neither.
  # P(neither) = 1/1 = 1.0; all other outcomes = 0.
  days = [{"date": "2024-02-01", "ar_high": 110, "ar_low": 90, "day_high": 110, "day_low": 92}]
  result = _stat().compute(make_candles(days))
  assert _row(result, "neither").count == 1
  assert _row(result, "broke_high").count == 0


def test_touching_low_level_is_not_a_break() -> None:
  """day_low == ar_low exactly → NOT a break (strict inequality required)."""
  # day_high 108 <= ar_high 110, day_low 90 == ar_low 90 → neither.
  days = [{"date": "2024-02-02", "ar_high": 110, "ar_low": 90, "day_high": 108, "day_low": 90}]
  result = _stat().compute(make_candles(days))
  assert _row(result, "neither").count == 1
  assert _row(result, "broke_low").count == 0


def test_touching_both_levels_is_neither() -> None:
  """day_high == ar_high AND day_low == ar_low → neither (boundary is inclusive inside)."""
  days = [{"date": "2024-02-03", "ar_high": 110, "ar_low": 90, "day_high": 110, "day_low": 90}]
  result = _stat().compute(make_candles(days))
  assert _row(result, "neither").count == 1
  assert _row(result, "broke_both").count == 0
  assert _row(result, "broke_high").count == 0
  assert _row(result, "broke_low").count == 0


# ===========================================================================
# Cross-midnight Asian session attribution
# ===========================================================================

def test_asian_range_from_prior_evening() -> None:
  """An Asian bar at 20:00 on the prior evening forms the next RTH day's range.

  session_bars cross-midnight logic: mod 1200 >= 1080 (asia_start=18:00) →
  _cycle = _date + 1 day. The Asian bar placed on 2024-03-03 at 20:00 is
  therefore attributed to the 2024-03-04 RTH cycle.
  """
  days = [{
    "date": "2024-03-04", "ar_high": 110, "ar_low": 90,
    "day_high": 115, "day_low": 95, "asian_prev_evening": True,
  }]
  result = _stat().compute(make_candles(days))
  # Asian range [90, 110] from the prior evening; RTH high 115 > 110 → broke_high.
  assert result.instruments["NQ"]["daily"].total_samples == 1
  assert _row(result, "broke_high").count == 1


def test_cross_midnight_and_early_bar_produce_identical_classification() -> None:
  """Evening bar (prev day 20:00) and early bar (same day 01:00) both attribute
  to the correct RTH cycle and yield the same breakout outcome."""
  # Both supply Asian range [90, 110]; RTH breaks only the high.
  # Hand-calc: day_high 115 > ar_high 110, day_low 95 >= ar_low 90 → broke_high.
  spec_early = {
    "date": "2024-04-01", "ar_high": 110, "ar_low": 90,
    "day_high": 115, "day_low": 95,
  }
  spec_evening = dict(spec_early, asian_prev_evening=True)

  result_early = _stat().compute(make_candles([spec_early]))
  result_evening = _stat().compute(make_candles([spec_evening]))

  assert result_early.instruments["NQ"]["daily"].total_samples == 1
  assert _row(result_early, "broke_high").count == 1
  assert result_evening.instruments["NQ"]["daily"].total_samples == 1
  assert _row(result_evening, "broke_high").count == 1


def test_asian_range_present_in_day_table() -> None:
  """build_day_table exposes ar_high / ar_low / ar_size columns for the session."""
  day_table = _stat().build_day_table(make_candles(_SEQ))
  d = pd.Timestamp("2024-01-02", tz=_NY).normalize()
  assert day_table.loc[d, "ar_high"] == 110
  assert day_table.loc[d, "ar_low"] == 90
  assert day_table.loc[d, "ar_size"] == 20


# ===========================================================================
# close vs wick breakout criteria
# ===========================================================================

def test_close_criteria_uses_bar_closes() -> None:
  """With close criteria a wick beyond the Asian high does not count; close must."""
  # day_high=115, day_low=95 → mid = (115+95)/2 = 105.
  # All intermediate closes = mid = 105. Last bar close = session_close = 105.
  # Wick:  day_high 115 > ar_high 110 → broke_high.
  # Close: day_close_high = max(105, 105) = 105 <= ar_high 110 → neither.
  days = [{
    "date": "2024-04-15", "ar_high": 110, "ar_low": 90,
    "day_high": 115, "day_low": 95, "session_open": 100.0, "session_close": 105.0,
  }]
  candles = make_candles(days)
  # Wick: high 115 > 110 → broke_high.
  assert _row(_stat("wick").compute(candles), "broke_high").count == 1
  # Close: max close = 105 <= 110 → neither.
  result_close = _stat("close").compute(candles)
  assert _row(result_close, "neither").count == 1
  assert _row(result_close, "broke_high").count == 0


def test_close_criteria_low_wick_only() -> None:
  """A wick that pierces ar_low but closes above it is not a break under close."""
  # day_high=108, day_low=85 → mid = (108+85)/2 = 96.5.
  # All intermediate closes = 96.5 >= ar_low=90. session_close = 96.5.
  # Wick:  day_low 85 < ar_low 90 → broke_low.
  # Close: day_close_low = min(96.5, 96.5) = 96.5 >= ar_low 90 → neither.
  days = [{
    "date": "2024-04-16", "ar_high": 110, "ar_low": 90,
    "day_high": 108, "day_low": 85,
  }]
  candles = make_candles(days)
  assert _row(_stat("wick").compute(candles), "broke_low").count == 1
  result_close = _stat("close").compute(candles)
  assert _row(result_close, "neither").count == 1
  assert _row(result_close, "broke_low").count == 0


def test_invalid_breakout_criteria_raises() -> None:
  with pytest.raises(ValueError):
    AsianRangeBreakout(instrument="NQ", config=_TEST_CONFIG, breakout_criteria="bogus")


# ===========================================================================
# Slices
# ===========================================================================

def test_declared_slice_dimensions_present() -> None:
  """All declared slice dimensions appear in the timeframe slices."""
  result = _stat().compute(make_candles(_SEQ))
  slices = result.instruments["NQ"]["daily"].slices
  assert set(slices) == {"weekday", "close", "prev_candle", "size", "levels"}


def test_weekday_slice_counts_reconcile_with_total() -> None:
  """Sum of weekday-group sample counts equals the unsliced countable total.

  Hand-calc: 8 days distributed across at most 5 weekdays; each day lands in
  exactly one weekday group, so their total_samples must sum to 8.
  """
  result = _stat().compute(make_candles(_SEQ))
  weekday = result.instruments["NQ"]["daily"].slices["weekday"]
  sliced_total = sum(g.total_samples for g in weekday.groups.values())
  assert sliced_total == result.instruments["NQ"]["daily"].total_samples == 8


def test_size_slice_buckets_asian_range() -> None:
  """The size slice buckets days by Asian-range size (ar_size)."""
  # Four distinct Asian-range sizes → four quartile buckets, one day each.
  # ar_size values: 5, 10, 20, 40.
  days = [
    {"date": "2024-05-01", "ar_high": 105, "ar_low": 100, "day_high": 110, "day_low": 95},
    {"date": "2024-05-02", "ar_high": 110, "ar_low": 100, "day_high": 115, "day_low": 95},
    {"date": "2024-05-03", "ar_high": 120, "ar_low": 100, "day_high": 125, "day_low": 95},
    {"date": "2024-05-06", "ar_high": 140, "ar_low": 100, "day_high": 145, "day_low": 95},
  ]
  result = _stat().compute(make_candles(days))
  size = result.instruments["NQ"]["daily"].slices["size"]
  # qcut on 4 distinct values → 4 single-day groups.
  assert len(size.groups) == 4
  assert all(g.total_samples == 1 for g in size.groups.values())


def test_levels_slice_uses_extension_multiples() -> None:
  """The levels slice bands the breakout extension in multiples of ar_size."""
  # ar_size = 20. Extensions chosen to land in distinct bands:
  #   day_high 115 → ext = 115 - 110 = 5  → 5/20 = 0.25x → 0–0.5x band
  #   day_high 125 → ext = 125 - 110 = 15 → 15/20 = 0.75x → 0.5–1x band
  #   day_high 140 → ext = 140 - 110 = 30 → 30/20 = 1.5x  → 1.5–2x band
  days = [
    {"date": "2024-06-03", "ar_high": 110, "ar_low": 90, "day_high": 115, "day_low": 95},
    {"date": "2024-06-04", "ar_high": 110, "ar_low": 90, "day_high": 125, "day_low": 95},
    {"date": "2024-06-05", "ar_high": 110, "ar_low": 90, "day_high": 140, "day_low": 95},
  ]
  result = _stat().compute(make_candles(days))
  levels = result.instruments["NQ"]["daily"].slices["levels"]
  keys = set(levels.groups)
  assert "0_0_5x" in keys
  assert "0_5_1x" in keys
  assert "1_5_2x" in keys


def test_close_slice_splits_by_session_color() -> None:
  """The close slice partitions days into green / red by RTH session color."""
  # green: session_close (115) >= session_open (95) → session_green = True.
  # red:   session_close (95)  <  session_open (115) → session_green = False.
  days = [
    {
      "date": "2024-07-01", "ar_high": 110, "ar_low": 90,
      "day_high": 120, "day_low": 90,
      "session_open": 95.0, "session_close": 115.0,
    },
    {
      "date": "2024-07-02", "ar_high": 110, "ar_low": 90,
      "day_high": 120, "day_low": 90,
      "session_open": 115.0, "session_close": 95.0,
    },
  ]
  result = _stat().compute(make_candles(days))
  close = result.instruments["NQ"]["daily"].slices["close"]
  assert close.groups["green"].total_samples == 1
  assert close.groups["red"].total_samples == 1


# ===========================================================================
# Baseline ≈ direction-symmetric and deterministic
# ===========================================================================

def _long_seq() -> pd.DataFrame:
  """~250 weekdays with a strong upside-break bias (mostly broke_high)."""
  dates: list[str] = []
  d = pd.Timestamp("2020-01-01", tz=_NY)
  while len(dates) < 250:
    if d.weekday() < 5:
      dates.append(d.strftime("%Y-%m-%d"))
    d += pd.Timedelta(days=1)

  days = []
  for i, date in enumerate(dates):
    if i % 5 == 0:
      # Occasional downside-only break to keep both outcomes populated.
      days.append({
        "date": date, "ar_high": 110, "ar_low": 90, "day_high": 105, "day_low": 80,
      })
    else:
      # Upside-only break.
      days.append({
        "date": date, "ar_high": 110, "ar_low": 90, "day_high": 120, "day_low": 95,
      })
  return make_candles(days)


def test_baseline_symmetric_high_low() -> None:
  """Reflection baseline removes directional bias: broke_high ≈ broke_low baseline.

  The null flips each day's RTH move around the Asian-range midpoint with P=0.5.
  With no directional preference, P(broke_high) and P(broke_low) converge to
  the same value. Tolerance abs=0.12 is conservative for ~250 samples.
  """
  stat = _stat()
  rows = {r.outcome: r for r in stat.baseline(_long_seq(), seed=42)}
  assert rows["broke_high"].probability == pytest.approx(
    rows["broke_low"].probability, abs=0.12
  )


def test_baseline_deterministic() -> None:
  """Same seed produces identical baseline results across two calls."""
  stat = _stat()
  df = _long_seq()
  rows_a = stat.baseline(df, seed=7)
  rows_b = stat.baseline(df, seed=7)
  for a, b in zip(rows_a, rows_b):
    assert (a.condition, a.outcome) == (b.condition, b.outcome)
    assert a.probability == pytest.approx(b.probability)


def test_real_directional_signal_beats_baseline() -> None:
  """Strong upside-break bias: broke_high real prob (≈0.80) > baseline (≈0.50)."""
  # Real broke_high = 200/250 = 0.80. Reflection baseline ≈ 0.50.
  result = _stat().compute(_long_seq())
  bh = _row(result, "broke_high")
  assert bh.probability > bh.baseline_prob


def test_baseline_embedded_in_compute() -> None:
  """After compute(), every row carries a positive baseline_n."""
  result = _stat().compute(_long_seq())
  for row in result.instruments["NQ"]["daily"].results:
    assert row.baseline_n > 0


# ===========================================================================
# Pending-sample discipline
# ===========================================================================

def test_pending_day_excluded_from_total_samples() -> None:
  """A truncated/early-close session does not appear in total_samples."""
  stat = _stat()
  base_df = make_candles(_SEQ)
  total_base = stat.compute(base_df).instruments["NQ"]["daily"].total_samples
  truncated = _make_truncated_session("2024-01-15")
  combined = (
    pd.concat([base_df, truncated], ignore_index=True)
    .sort_values("timestamp")
    .reset_index(drop=True)
  )
  total_with = stat.compute(combined).instruments["NQ"]["daily"].total_samples
  assert total_with == total_base == 8


def test_pending_day_absent_from_day_table() -> None:
  """build_day_table excludes the pending (truncated) session entirely."""
  stat = _stat()
  day_table = stat.build_day_table(_make_truncated_session("2024-01-15"))
  pending = pd.Timestamp("2024-01-15", tz=_NY).normalize()
  assert pending not in day_table.index


def test_session_without_asian_range_excluded() -> None:
  """A resolved RTH session with no Asian bars is dropped by the inner join.

  The inner join of ``resolved`` with ``ar_high`` / ``ar_low`` silently drops
  dates that have no Asian session bars, enforcing pending-sample discipline for
  days whose prior Asian range is unknown.
  """
  # Only RTH bars for 2024-08-01 — no Asian bar → ar_high / ar_low NaN → excluded.
  base = pd.Timestamp("2024-08-01", tz=_NY)
  mid = 100.0
  records = []
  for mod in range(_RTH_START, _RTH_LAST + 1):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    hi = 120.0 if mod == _RTH_START else mid
    lo = 80.0 if mod == _RTH_START else mid
    records.append({
      "timestamp": ts, "open": mid, "high": hi, "low": lo, "close": mid, "volume": 1000,
    })
  result = _stat().compute(pd.DataFrame(records))
  # Resolved RTH day but no Asian bars → not countable.
  assert result.instruments["NQ"]["daily"].total_samples == 0


# ===========================================================================
# Edge cases & validation
# ===========================================================================

def _empty_df() -> pd.DataFrame:
  df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
  df["timestamp"] = pd.array([], dtype="datetime64[ns, America/New_York]")
  return df


def test_empty_dataframe_no_crash() -> None:
  """Empty input → zero samples, empty data_range, all four rows zeroed."""
  result = _stat().compute(_empty_df())
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 0
  assert tf.data_range == []
  for row in tf.results:
    assert row.count == 0
    assert row.total == 0
    assert row.probability == pytest.approx(0.0)


def test_single_day_all_four_rows_present() -> None:
  """One countable day → exactly four rows, totals = 1, one outcome carries it."""
  days = [{"date": "2024-09-03", "ar_high": 110, "ar_low": 90, "day_high": 115, "day_low": 95}]
  result = _stat().compute(make_candles(days))
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 1
  # day_high 115 > ar_high 110, day_low 95 >= ar_low 90 → broke_high.
  assert _row(result, "broke_high").count == 1
  assert {r.total for r in tf.results} == {1}


# ===========================================================================
# i18n
# ===========================================================================

def test_i18n_title_and_definition() -> None:
  """title and definition both have non-empty en and fr strings."""
  result = _stat().compute(_empty_df())
  assert result.title.en != "" and result.title.fr != ""
  assert result.definition.en != "" and result.definition.fr != ""


def test_i18n_labels_have_en_and_fr() -> None:
  """Every condition and outcome label has non-empty en and fr; correct keys."""
  result = _stat().compute(_empty_df())
  for mapping in (result.labels.conditions, result.labels.outcomes):
    for key, i18n in mapping.items():
      assert i18n.en != "", f"{key}.en empty"
      assert i18n.fr != "", f"{key}.fr empty"
  assert set(result.labels.conditions) == {"asian_range"}
  assert set(result.labels.outcomes) == {"broke_high", "broke_low", "broke_both", "neither"}


# ===========================================================================
# stat_name and write_results round-trip
# ===========================================================================

def test_stat_name() -> None:
  assert _stat().compute(_empty_df()).stat_name == "asian_range_breakout"


def test_write_results_round_trip(tmp_path: Path) -> None:
  """write_results produces asian_range_breakout.json that re-validates."""
  result = _stat().compute(make_candles(_SEQ))
  written = write_results(result, results_dir=tmp_path)
  assert written.name == "asian_range_breakout.json"

  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)
  assert validated.instruments["NQ"]["daily"].total_samples == 8


def test_write_results_french_accent_literal(tmp_path: Path) -> None:
  """French text is stored as literal UTF-8 characters, not escaped unicode."""
  result = _stat().compute(_empty_df())
  written = write_results(result, results_dir=tmp_path)
  raw = written.read_text(encoding="utf-8")
  assert "é" in raw
  assert "\\u00e9" not in raw


# ===========================================================================
# classify_samples
#
# Reusing _SEQ (see the 4-outcome partition table above): all 8 days are
# countable (resolved RTH day + prior Asian range).
# ===========================================================================

def test_classify_samples_exact_list() -> None:
  """classify_samples emits one SampleRow per countable day, matching the hand-calc."""
  stat = _stat()
  table = stat.build_day_table(make_candles(_SEQ))
  samples = stat.classify_samples(table)
  assert [(s.date, s.condition, s.outcome) for s in samples] == [
    ("2024-01-02", "asian_range", "broke_high"),
    ("2024-01-03", "asian_range", "broke_low"),
    ("2024-01-04", "asian_range", "broke_both"),
    ("2024-01-05", "asian_range", "neither"),
    ("2024-01-08", "asian_range", "broke_high"),
    ("2024-01-09", "asian_range", "broke_low"),
    ("2024-01-10", "asian_range", "broke_both"),
    ("2024-01-11", "asian_range", "neither"),
  ]
  assert all(s.value is None for s in samples)


def test_classify_samples_matches_compute_rows_counts() -> None:
  """For every StatResultRow, the matching SampleRow count equals r.count."""
  stat = _stat()
  table = stat.build_day_table(make_candles(_SEQ))
  samples = stat.classify_samples(table)
  rows = stat.compute_rows(table)
  for r in rows:
    n = sum(1 for s in samples if s.condition == r.condition and s.outcome == r.outcome)
    assert n == r.count


def test_classify_samples_empty_day_table() -> None:
  """Empty day_table -> classify_samples returns []."""
  stat = _stat()
  assert stat.classify_samples(_empty_df().iloc[:0]) == []
  assert stat.classify_samples(pd.DataFrame()) == []


def test_classify_samples_excludes_noncountable_day() -> None:
  """A day with a NaN Asian range (non-countable) produces no SampleRow.

  ``build_day_table`` always inner-joins away such days, so this constructs
  the table manually to directly exercise the ``countable`` mask that
  ``classify_samples`` shares with ``compute_rows``.
  """
  stat = _stat()
  index = pd.DatetimeIndex(
    [pd.Timestamp("2024-02-01", tz=_NY), pd.Timestamp("2024-02-02", tz=_NY)]
  )
  table = pd.DataFrame(
    {
      "ar_high": [110.0, float("nan")],
      "ar_low": [90.0, 90.0],
      "day_high": [115.0, 115.0],
      "day_low": [95.0, 95.0],
      "day_close_high": [115.0, 115.0],
      "day_close_low": [95.0, 95.0],
    },
    index=index,
  )
  samples = stat.classify_samples(table)
  assert [(s.date, s.condition, s.outcome) for s in samples] == [
    ("2024-02-01", "asian_range", "broke_high"),
  ]
  rows = stat.compute_rows(table)
  for r in rows:
    n = sum(1 for s in samples if s.condition == r.condition and s.outcome == r.outcome)
    assert n == r.count
