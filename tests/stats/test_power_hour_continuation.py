"""Tests for stats.power_hour_continuation.standard.

All data is synthetic — no real market files required. Expected values are
hand-calculated before each assertion.

Matrix framing: condition = pre-power-hour candle color (pre_close vs session_open),
outcome = power-hour candle color (session_close vs ph_open).

RTH for NQ: 09:30–16:15 → rth_start_min=570, rth_end_min=975.
Power hour = last 60 min → ph_start_min = 975 - 60 = 915 (= 15:15).
Pre-PH window: [570, 915). PH window: [915, 975).
Resolution threshold: last bar >= 975 - 15 = 960 (16:00).
Last valid bar = 16:14 (mod 974).

  pre_green: pre_close (last bar close before 15:15) >= session_open (open of 09:30 bar)
  ph_green:  session_close (close of last RTH bar)   >= ph_open (open of 15:15 bar)
  ph_open_above: ph_open >= (pre_high + pre_low) / 2
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.power_hour_continuation.standard import PowerHourContinuation

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
_PH_START = 915    # 15:15 — power-hour open bar
_RTH_LAST = 974    # 16:14, last bar before 16:15; mod >= 960 so resolved


def _make_cont_day(
  date: str,
  *,
  session_open: float,
  pre_close: float,
  pre_high: float,
  pre_low: float,
  ph_open: float,
  session_close: float,
  last_mod: int = _RTH_LAST,
) -> pd.DataFrame:
  """One trading day of 1-min RTH bars with controlled pre-PH and PH windows.

  The 09:30 bar carries ``session_open`` as its open price.
  The last bar before 15:15 (mod 914) carries ``pre_close`` as its close.
  The pre-PH range extremes (``pre_high``, ``pre_low``) are placed as wicks on
  dedicated bars within the pre-PH window so the midpoint is deterministic.
  The 15:15 bar carries ``ph_open`` as its open.
  The last RTH bar (``last_mod``) carries ``session_close`` as its close.

  ``last_mod`` can be set < 960 to simulate an early-close / unresolved day.
  """
  base = pd.Timestamp(date, tz=_NY)
  neutral = 100.0  # price for bars that should not affect the measured values

  records = []
  for mod in range(_RTH_START, last_mod + 1):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)

    if mod < _PH_START:
      # Pre-PH window [09:30, 15:15)
      if mod == _RTH_START:
        # Session-open bar: open carries session_open
        o = session_open
        c = neutral
        hi = max(o, c)
        lo = min(o, c)
      elif mod == _RTH_START + 1:
        # Place pre_high wick here
        o = neutral
        c = neutral
        hi, lo = pre_high, neutral
      elif mod == _RTH_START + 2:
        # Place pre_low wick here
        o = neutral
        c = neutral
        hi, lo = neutral, pre_low
      elif mod == _PH_START - 1:
        # Last pre-PH bar (15:14, mod 914): close carries pre_close
        o = neutral
        c = pre_close
        hi = max(o, c)
        lo = min(o, c)
      else:
        o = neutral
        c = neutral
        hi, lo = neutral, neutral
    else:
      # PH window [15:15, 16:15)
      if mod == _PH_START:
        # First PH bar: open carries ph_open
        o = ph_open
        c = neutral
        hi = max(o, c)
        lo = min(o, c)
      elif mod == last_mod:
        # Last bar: close carries session_close
        o = neutral
        c = session_close
        hi = max(o, c)
        lo = min(o, c)
      else:
        o = neutral
        c = neutral
        hi, lo = neutral, neutral

    records.append({
      "timestamp": ts,
      "open": o,
      "high": hi,
      "low": lo,
      "close": c,
      "volume": 1000,
    })

  return pd.DataFrame(records)


def _make_no_rth_open_day(date: str) -> pd.DataFrame:
  """A resolved day missing the 09:30 open bar → excluded by build_resolved_days."""
  base = pd.Timestamp(date, tz=_NY)
  records = []
  for mod in range(_RTH_START + 1, _RTH_LAST + 1):  # skip 09:30
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


def _make_no_pre_ph_day(date: str) -> pd.DataFrame:
  """A day whose bars start at 15:15 — no pre-PH bars → excluded (also unresolved)."""
  base = pd.Timestamp(date, tz=_NY)
  records = []
  # Start at PH window only; this day is also unresolved (no 09:30 bar)
  for mod in range(_PH_START, _RTH_LAST + 1):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    records.append({
      "timestamp": ts, "open": 100.0, "high": 100.25, "low": 99.75,
      "close": 100.0, "volume": 500,
    })
  return pd.DataFrame(records)


def _make_no_ph_open_bar_day(date: str) -> pd.DataFrame:
  """A resolved day missing the 15:15 bar exactly → excluded from countable days."""
  base = pd.Timestamp(date, tz=_NY)
  records = []
  for mod in range(_RTH_START, _RTH_LAST + 1):
    if mod == _PH_START:
      continue  # skip 15:15 bar
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


def _stat() -> PowerHourContinuation:
  return PowerHourContinuation(
    instrument="NQ",
    timeframe="1h",
    config=_TEST_CONFIG,
  )


def _row(result: StatRunResult, condition: str, outcome: str) -> StatResultRow:
  for r in result.instruments["NQ"]["1h"].results:
    if r.condition == condition and r.outcome == outcome:
      return r
  raise KeyError((condition, outcome))


# ===========================================================================
# Core matrix (6-day synthetic set)
#
# Day-by-day derivation:
#   For all days:
#     pre_high=110, pre_low=90 (midpoint=100)
#     session_open = 100 (the 09:30 bar open)
#
#   Day 1 (2024-01-02): pre_close=110 (>=100 → pre_green), ph_open=95, session_close=100
#     ph_green: 100 >= 95 → True → green  →  pre_green → green
#
#   Day 2 (2024-01-03): pre_close=110 (>=100 → pre_green), ph_open=95, session_close=90
#     ph_green: 90 >= 95 → False → red    →  pre_green → red
#
#   Day 3 (2024-01-04): pre_close=110 (>=100 → pre_green), ph_open=95, session_close=100
#     ph_green: 100 >= 95 → True → green  →  pre_green → green
#
#   Day 4 (2024-01-05): pre_close=80 (<100 → pre_red), ph_open=95, session_close=100
#     ph_green: 100 >= 95 → True → green  →  pre_red → green
#
#   Day 5 (2024-01-08): pre_close=80 (<100 → pre_red), ph_open=105, session_close=95
#     ph_green: 95 >= 105 → False → red   →  pre_red → red
#
#   Day 6 (2024-01-09): pre_close=80 (<100 → pre_red), ph_open=105, session_close=95
#     ph_green: 95 >= 105 → False → red   →  pre_red → red
#
# All 6 days are resolved (bars run to 16:14, mod=974 >= 960).
# All have a 15:15 bar (ph_open present). All have pre-PH bars.
# All have the 09:30 bar (session_open present).
# → total_samples = 6, all 6 are countable.
#
# pre_green days: 1, 2, 3 → total=3
#   pre_green → green: 1, 3      → count=2, P=2/3
#   pre_green → red:   2         → count=1, P=1/3
#
# pre_red days:   4, 5, 6 → total=3
#   pre_red → green:  4         → count=1, P=1/3
#   pre_red → red:    5, 6      → count=2, P=2/3
# ===========================================================================

_SEQ = [
  dict(
    date="2024-01-02", session_open=100.0, pre_close=110.0,
    pre_high=110.0, pre_low=90.0, ph_open=95.0, session_close=100.0,
  ),  # pre_green → ph_green
  dict(
    date="2024-01-03", session_open=100.0, pre_close=110.0,
    pre_high=110.0, pre_low=90.0, ph_open=95.0, session_close=90.0,
  ),  # pre_green → ph_red
  dict(
    date="2024-01-04", session_open=100.0, pre_close=110.0,
    pre_high=110.0, pre_low=90.0, ph_open=95.0, session_close=100.0,
  ),  # pre_green → ph_green
  dict(
    date="2024-01-05", session_open=100.0, pre_close=80.0,
    pre_high=110.0, pre_low=90.0, ph_open=95.0, session_close=100.0,
  ),  # pre_red → ph_green
  dict(
    date="2024-01-08", session_open=100.0, pre_close=80.0,
    pre_high=110.0, pre_low=90.0, ph_open=105.0, session_close=95.0,
  ),  # pre_red → ph_red
  dict(
    date="2024-01-09", session_open=100.0, pre_close=80.0,
    pre_high=110.0, pre_low=90.0, ph_open=105.0, session_close=95.0,
  ),  # pre_red → ph_red
]


def _canon_candles() -> pd.DataFrame:
  return _concat([_make_cont_day(**spec) for spec in _SEQ])


# ===========================================================================
# Core matrix tests
# ===========================================================================

def test_total_samples_counts_all_resolved_days() -> None:
  """total_samples = 6 resolved sessions (all have 09:30 bar and run to 16:14)."""
  result = _stat().compute(_canon_candles())
  assert result.instruments["NQ"]["1h"].total_samples == 6


def test_pre_green_conditional() -> None:
  """Given a green pre-move: P(ph_green)=2/3, P(ph_red)=1/3 over 3 countable days."""
  result = _stat().compute(_canon_candles())
  green = _row(result, "pre_green", "green")
  red = _row(result, "pre_green", "red")
  # pre_green total = 3 (days 1, 2, 3)
  assert green.total == red.total == 3
  assert green.count == 2
  assert green.probability == pytest.approx(2 / 3)
  assert red.count == 1
  assert red.probability == pytest.approx(1 / 3)


def test_pre_red_conditional() -> None:
  """Given a red pre-move: P(ph_green)=1/3, P(ph_red)=2/3 over 3 countable days."""
  result = _stat().compute(_canon_candles())
  green = _row(result, "pre_red", "green")
  red = _row(result, "pre_red", "red")
  # pre_red total = 3 (days 4, 5, 6)
  assert green.total == red.total == 3
  assert green.count == 1
  assert green.probability == pytest.approx(1 / 3)
  assert red.count == 2
  assert red.probability == pytest.approx(2 / 3)


def test_condition_totals_sum_to_total_samples() -> None:
  """Sum of condition totals == total_samples (every resolved day is countable here)."""
  result = _stat().compute(_canon_candles())
  tf = result.instruments["NQ"]["1h"]
  pre_green_total = _row(result, "pre_green", "green").total
  pre_red_total = _row(result, "pre_red", "green").total
  # All 6 days are countable (have a 15:15 bar and pre-PH bars)
  assert pre_green_total + pre_red_total == tf.total_samples == 6


def test_outcomes_partition_each_condition() -> None:
  """green.count + red.count == total for each pre-move condition."""
  result = _stat().compute(_canon_candles())
  for cond in ("pre_green", "pre_red"):
    green = _row(result, cond, "green")
    red = _row(result, cond, "red")
    assert green.count + red.count == green.total == red.total


def test_four_rows_only() -> None:
  """Exactly four rows: the 2x2 (pre-move color × power-hour color) matrix."""
  result = _stat().compute(_canon_candles())
  rows = result.instruments["NQ"]["1h"].results
  assert {(r.condition, r.outcome) for r in rows} == {
    ("pre_green", "green"), ("pre_green", "red"),
    ("pre_red", "green"), ("pre_red", "red"),
  }


def test_data_range() -> None:
  """data_range spans from the first to the last resolved session date."""
  result = _stat().compute(_canon_candles())
  assert result.instruments["NQ"]["1h"].data_range == ["2024-01-02", "2024-01-09"]


# ===========================================================================
# Boundary / tie: >= is green for both conditions and outcomes
# ===========================================================================

def test_boundary_pre_close_equals_session_open_is_pre_green() -> None:
  """pre_close == session_open → pre_green (>= classifies as green)."""
  # pre_close = session_open = 100 → pre_green = True
  # ph_green: session_close=105 >= ph_open=100 → True
  candles = _concat([_make_cont_day(
    date="2024-01-02",
    session_open=100.0,
    pre_close=100.0,   # tie → pre_green
    pre_high=110.0,
    pre_low=90.0,
    ph_open=100.0,
    session_close=105.0,
  )])
  result = _stat().compute(candles)
  pg_green = _row(result, "pre_green", "green")
  pr_green = _row(result, "pre_red", "green")
  # The day must land in pre_green, not pre_red
  assert pg_green.total == 1
  assert pg_green.count == 1
  assert pr_green.total == 0


def test_boundary_session_close_equals_ph_open_is_ph_green() -> None:
  """session_close == ph_open → ph_green (>= classifies as green)."""
  # pre_close=110 > session_open=100 → pre_green
  # session_close = ph_open = 100 → ph_green (tie, >=)
  candles = _concat([_make_cont_day(
    date="2024-01-02",
    session_open=100.0,
    pre_close=110.0,
    pre_high=110.0,
    pre_low=90.0,
    ph_open=100.0,
    session_close=100.0,  # tie → ph_green
  )])
  result = _stat().compute(candles)
  pg_green = _row(result, "pre_green", "green")
  pg_red = _row(result, "pre_green", "red")
  # Must land in green outcome (tie = green)
  assert pg_green.count == 1
  assert pg_red.count == 0


def test_boundary_both_ties_are_both_green() -> None:
  """Both pre_close == session_open and session_close == ph_open → pre_green → ph_green."""
  candles = _concat([_make_cont_day(
    date="2024-01-02",
    session_open=100.0,
    pre_close=100.0,    # tie → pre_green
    pre_high=110.0,
    pre_low=90.0,
    ph_open=100.0,
    session_close=100.0,  # tie → ph_green
  )])
  result = _stat().compute(candles)
  pg_green = _row(result, "pre_green", "green")
  assert pg_green.count == 1
  assert pg_green.total == 1


# ===========================================================================
# Pending / unresolved discipline
# ===========================================================================

def test_missing_09_30_bar_excluded() -> None:
  """A day missing the 09:30 open bar is excluded (no session_open → not resolved)."""
  stat = _stat()
  base_df = _canon_candles()
  total_base = stat.compute(base_df).instruments["NQ"]["1h"].total_samples
  no_open = _make_no_rth_open_day("2024-01-15")
  combined = _concat([base_df, no_open])
  total_with = stat.compute(combined).instruments["NQ"]["1h"].total_samples
  # The missing-09:30 day must not increase the count
  assert total_with == total_base == 6


def test_early_close_day_excluded() -> None:
  """A day whose last bar is at 15:30 (mod 930 < 960) is excluded as unresolved."""
  stat = _stat()
  base_df = _canon_candles()
  total_base = stat.compute(base_df).instruments["NQ"]["1h"].total_samples
  early = _make_early_close_day("2024-01-15")
  combined = _concat([base_df, early])
  total_with = stat.compute(combined).instruments["NQ"]["1h"].total_samples
  assert total_with == total_base == 6


def test_no_pre_ph_bars_day_excluded() -> None:
  """A day that only has PH bars (no 09:30 bar) is excluded — no pre-PH window."""
  stat = _stat()
  base_df = _canon_candles()
  total_base = stat.compute(base_df).instruments["NQ"]["1h"].total_samples
  no_pre = _make_no_pre_ph_day("2024-01-15")
  combined = _concat([base_df, no_pre])
  total_with = stat.compute(combined).instruments["NQ"]["1h"].total_samples
  assert total_with == total_base == 6


def test_missing_ph_open_bar_drops_from_day_table() -> None:
  """A resolved day missing the 15:15 bar fails the inner join and is excluded
  from build_day_table entirely. Since total_samples == len(day_table), it
  does not appear in total_samples either."""
  stat = _stat()
  # 1 canonical day + 1 day missing 15:15 bar
  good_day = _make_cont_day(
    date="2024-01-02",
    session_open=100.0, pre_close=110.0,
    pre_high=110.0, pre_low=90.0,
    ph_open=95.0, session_close=100.0,
  )
  no_ph_open = _make_no_ph_open_bar_day("2024-01-03")
  combined = _concat([good_day, no_ph_open])
  result = stat.compute(combined)
  tf = result.instruments["NQ"]["1h"]
  # Only the day with a clean 15:15 bar survives the inner join
  assert tf.total_samples == 1
  # That one countable day was pre_green (pre_close=110 >= session_open=100)
  # and ph_green (session_close=100 >= ph_open=95)
  pre_green_total = _row(result, "pre_green", "green").total
  pre_red_total = _row(result, "pre_red", "green").total
  assert pre_green_total + pre_red_total == 1
  assert pre_green_total == 1


# ===========================================================================
# Empty input
# ===========================================================================

def test_empty_dataframe_no_crash() -> None:
  """Empty input → zero samples, empty data_range, all rows zeroed."""
  result = _stat().compute(_empty_df())
  tf = result.instruments["NQ"]["1h"]
  assert tf.total_samples == 0
  assert tf.data_range == []
  for row in tf.results:
    assert row.count == 0
    assert row.total == 0
    assert row.probability == pytest.approx(0.0)


def test_empty_dataframe_four_rows_produced() -> None:
  """Empty input → still emits 4 zeroed rows (no crash, no missing keys)."""
  result = _stat().compute(_empty_df())
  rows = result.instruments["NQ"]["1h"].results
  assert len(rows) == 4
  assert {(r.condition, r.outcome) for r in rows} == {
    ("pre_green", "green"), ("pre_green", "red"),
    ("pre_red", "green"), ("pre_red", "red"),
  }


# ===========================================================================
# Single day
# ===========================================================================

def test_single_day_one_countable_row() -> None:
  """One resolved day with pre-PH bars and a 15:15 bar → 1 countable day total."""
  # pre_close=110 > session_open=100 → pre_green; session_close=105 > ph_open=95 → ph_green
  candles = _concat([_make_cont_day(
    date="2024-01-02",
    session_open=100.0, pre_close=110.0,
    pre_high=110.0, pre_low=90.0,
    ph_open=95.0, session_close=105.0,
  )])
  result = _stat().compute(candles)
  tf = result.instruments["NQ"]["1h"]
  assert tf.total_samples == 1
  pg_green = _row(result, "pre_green", "green")
  assert pg_green.total == 1
  assert pg_green.count == 1
  # pre_red rows are empty
  pr_green = _row(result, "pre_red", "green")
  assert pr_green.total == 0
  assert pr_green.probability == pytest.approx(0.0)


def test_all_pre_green_no_pre_red_rows() -> None:
  """All days are pre_green → pre_red rows are empty (no zero-division)."""
  days = [
    dict(date="2024-01-02", session_open=100.0, pre_close=110.0,
         pre_high=110.0, pre_low=90.0, ph_open=95.0, session_close=105.0),
    dict(date="2024-01-03", session_open=100.0, pre_close=110.0,
         pre_high=110.0, pre_low=90.0, ph_open=95.0, session_close=90.0),
  ]
  candles = _concat([_make_cont_day(**d) for d in days])
  result = _stat().compute(candles)
  pr_green = _row(result, "pre_red", "green")
  pr_red = _row(result, "pre_red", "red")
  assert pr_green.total == 0
  assert pr_red.total == 0
  assert pr_green.probability == pytest.approx(0.0)
  assert pr_red.probability == pytest.approx(0.0)


def test_all_pre_red_no_pre_green_rows() -> None:
  """All days are pre_red → pre_green rows are empty (no zero-division)."""
  days = [
    dict(date="2024-01-02", session_open=100.0, pre_close=90.0,
         pre_high=110.0, pre_low=80.0, ph_open=95.0, session_close=100.0),
    dict(date="2024-01-03", session_open=100.0, pre_close=90.0,
         pre_high=110.0, pre_low=80.0, ph_open=105.0, session_close=95.0),
  ]
  candles = _concat([_make_cont_day(**d) for d in days])
  result = _stat().compute(candles)
  pg_green = _row(result, "pre_green", "green")
  assert pg_green.total == 0
  assert pg_green.probability == pytest.approx(0.0)


# ===========================================================================
# Probabilities sum to 1 within each condition
# ===========================================================================

def test_probabilities_sum_to_one_per_condition() -> None:
  """green.probability + red.probability == 1.0 for each non-empty condition."""
  result = _stat().compute(_canon_candles())
  for cond in ("pre_green", "pre_red"):
    green = _row(result, cond, "green")
    red = _row(result, cond, "red")
    assert green.probability + red.probability == pytest.approx(1.0)


# ===========================================================================
# Baseline: determinism and ≈ 0.5
# ===========================================================================

def _long_seq() -> pd.DataFrame:
  """~200 weekdays of synthetic days with mixed pre conditions.

  Half the days are pre_green (pre_close > session_open),
  half are pre_red (pre_close < session_open), alternating.
  Both have a mix of ph outcomes. Builds a balanced dataset so the
  random baseline hovers near 0.5 for each row.
  """
  dates: list[str] = []
  d = pd.Timestamp("2020-01-01", tz=_NY)
  while len(dates) < 200:
    if d.weekday() < 5:
      dates.append(d.strftime("%Y-%m-%d"))
    d += pd.Timedelta(days=1)

  days = []
  for i, date in enumerate(dates):
    if i % 2 == 0:
      # pre_green day (pre_close > session_open), alternating ph outcome
      pre_close = 110.0
      session_open = 100.0
      if i % 4 == 0:
        ph_open, session_close = 95.0, 105.0  # ph_green
      else:
        ph_open, session_close = 105.0, 95.0  # ph_red
    else:
      # pre_red day (pre_close < session_open), alternating ph outcome
      pre_close = 90.0
      session_open = 100.0
      if i % 4 == 1:
        ph_open, session_close = 95.0, 105.0  # ph_green
      else:
        ph_open, session_close = 105.0, 95.0  # ph_red

    days.append(_make_cont_day(
      date=date,
      session_open=session_open,
      pre_close=pre_close,
      pre_high=110.0,
      pre_low=90.0,
      ph_open=ph_open,
      session_close=session_close,
    ))

  return _concat(days)


def test_baseline_deterministic() -> None:
  """Same seed produces identical baseline results across two calls."""
  stat = _stat()
  df = _long_seq()
  rows_a = stat.baseline(df, seed=7)
  rows_b = stat.baseline(df, seed=7)
  for a, b in zip(rows_a, rows_b):
    assert (a.condition, a.outcome) == (b.condition, b.outcome)
    assert a.probability == pytest.approx(b.probability)


def test_baseline_different_seeds_may_differ() -> None:
  """Different seeds may produce different baseline results (probabilistic check)."""
  stat = _stat()
  df = _long_seq()
  day_table = stat.build_day_table(df)
  rows_42 = stat.baseline_rows(day_table, seed=42)
  rows_99 = stat.baseline_rows(day_table, seed=99)
  # Collect all probabilities from both runs; they should not all be identical
  probs_42 = [r.probability for r in rows_42]
  probs_99 = [r.probability for r in rows_99]
  # With 200 days and different seeds, at least one row should differ
  assert probs_42 != probs_99


def test_baseline_approx_fifty_percent() -> None:
  """Random baseline probability is near 0.5 for every matrix row (large sample)."""
  stat = _stat()
  rows = stat.baseline(_long_seq(), seed=42)
  for row in rows:
    assert row.probability == pytest.approx(0.5, abs=0.12)


def test_baseline_embedded_in_compute() -> None:
  """After compute(), every row carries a positive baseline_n."""
  result = _stat().compute(_long_seq())
  for row in result.instruments["NQ"]["1h"].results:
    assert row.baseline_n > 0


def test_baseline_preserves_pre_condition_counts() -> None:
  """The baseline only randomizes ph_green; pre_green proportions are unchanged."""
  stat = _stat()
  df = _long_seq()
  day_table = stat.build_day_table(df)
  bl_rows = stat.baseline_rows(day_table, seed=42)
  real_rows = stat.compute_rows(day_table)
  # For each condition, the total (condition count) must be the same
  # between baseline and real rows (only ph_green was randomized)
  for cond in ("pre_green", "pre_red"):
    bl_total = next(r.total for r in bl_rows if r.condition == cond and r.outcome == "green")
    real_total = next(r.total for r in real_rows if r.condition == cond and r.outcome == "green")
    assert bl_total == real_total


# ===========================================================================
# "open" slice: ph_open above/below pre-range midpoint
# ===========================================================================

def test_open_slice_dimensions_present() -> None:
  """compute() exposes 'weekday' and 'open' slice dimensions."""
  result = _stat().compute(_canon_candles())
  slices = result.instruments["NQ"]["1h"].slices
  assert set(slices) == {"weekday", "open"}


def test_open_slice_above_when_ph_open_at_midpoint() -> None:
  """ph_open == midpoint (= (pre_high+pre_low)/2) → ph_open_above=True → 'above' group."""
  # pre_high=110, pre_low=90 → midpoint=100. ph_open=100 → ph_open_above=True
  candles = _concat([_make_cont_day(
    date="2024-01-02",
    session_open=100.0, pre_close=110.0,
    pre_high=110.0, pre_low=90.0,
    ph_open=100.0,     # exactly at midpoint
    session_close=105.0,
  )])
  result = _stat().compute(candles)
  open_groups = result.instruments["NQ"]["1h"].slices["open"].groups
  assert "above" in open_groups
  assert open_groups["above"].total_samples == 1
  assert "below" not in open_groups


def test_open_slice_below_when_ph_open_below_midpoint() -> None:
  """ph_open < midpoint → ph_open_above=False → 'below' group."""
  # pre_high=110, pre_low=90 → midpoint=100. ph_open=99 < 100 → ph_open_above=False
  candles = _concat([_make_cont_day(
    date="2024-01-02",
    session_open=100.0, pre_close=110.0,
    pre_high=110.0, pre_low=90.0,
    ph_open=99.0,      # below midpoint
    session_close=105.0,
  )])
  result = _stat().compute(candles)
  open_groups = result.instruments["NQ"]["1h"].slices["open"].groups
  assert "below" in open_groups
  assert open_groups["below"].total_samples == 1
  assert "above" not in open_groups


def test_open_slice_above_when_ph_open_above_midpoint() -> None:
  """ph_open > midpoint → ph_open_above=True → 'above' group."""
  # midpoint=100, ph_open=101 > 100 → above
  candles = _concat([_make_cont_day(
    date="2024-01-02",
    session_open=100.0, pre_close=110.0,
    pre_high=110.0, pre_low=90.0,
    ph_open=101.0,
    session_close=105.0,
  )])
  result = _stat().compute(candles)
  open_groups = result.instruments["NQ"]["1h"].slices["open"].groups
  assert "above" in open_groups
  assert "below" not in open_groups


def test_open_slice_splits_mixed_days_correctly() -> None:
  """Days split into above and below by ph_open vs pre-range midpoint.

  Scenario:
    Day 1: pre_high=110, pre_low=90 → mid=100; ph_open=105 → above;
           pre_close=110 > session_open=100 → pre_green; session_close=110 > 105 → ph_green
    Day 2: pre_high=110, pre_low=90 → mid=100; ph_open=95  → below;
           pre_close=90  < session_open=100 → pre_red;   session_close=85 < 95 → ph_red
    Day 3: pre_high=110, pre_low=90 → mid=100; ph_open=101 → above;
           pre_close=90  < session_open=100 → pre_red;   session_close=105 > 101 → ph_green
  """
  candles = _concat([
    _make_cont_day(
      date="2024-01-02",
      session_open=100.0, pre_close=110.0,
      pre_high=110.0, pre_low=90.0,
      ph_open=105.0, session_close=110.0,  # above, pre_green→ph_green
    ),
    _make_cont_day(
      date="2024-01-03",
      session_open=100.0, pre_close=90.0,
      pre_high=110.0, pre_low=90.0,
      ph_open=95.0, session_close=85.0,   # below, pre_red→ph_red
    ),
    _make_cont_day(
      date="2024-01-04",
      session_open=100.0, pre_close=90.0,
      pre_high=110.0, pre_low=90.0,
      ph_open=101.0, session_close=105.0,  # above, pre_red→ph_green
    ),
  ])
  result = _stat().compute(candles)
  open_groups = result.instruments["NQ"]["1h"].slices["open"].groups

  # 2 days above (day 1 and day 3), 1 day below (day 2)
  assert open_groups["above"].total_samples == 2
  assert open_groups["below"].total_samples == 1

  # Within "above": day 1 = pre_green→ph_green, day 3 = pre_red→ph_green
  above_rows = {(r.condition, r.outcome): r for r in open_groups["above"].results}
  # Day 1 contributes pre_green→green count=1, day 3 contributes pre_red→green count=1
  assert above_rows[("pre_green", "green")].count == 1
  assert above_rows[("pre_green", "red")].count == 0
  assert above_rows[("pre_red", "green")].count == 1
  assert above_rows[("pre_red", "red")].count == 0

  # Within "below": day 2 = pre_red→ph_red
  below_rows = {(r.condition, r.outcome): r for r in open_groups["below"].results}
  assert below_rows[("pre_red", "red")].count == 1
  assert below_rows[("pre_red", "green")].count == 0


def test_open_slice_groups_sum_to_total_samples() -> None:
  """Total samples across all open-slice groups == total_samples."""
  result = _stat().compute(_canon_candles())
  tf = result.instruments["NQ"]["1h"]
  open_groups = tf.slices["open"].groups
  total_in_slices = sum(g.total_samples for g in open_groups.values())
  assert total_in_slices == tf.total_samples


# ===========================================================================
# "weekday" slice
# ===========================================================================

def test_weekday_slice_partitions_correctly() -> None:
  """Each of the 6 canonical days lands in the correct weekday bucket.

  2024-01-02 = Tuesday, 2024-01-03 = Wednesday, 2024-01-04 = Thursday,
  2024-01-05 = Friday, 2024-01-08 = Monday, 2024-01-09 = Tuesday.
  """
  result = _stat().compute(_canon_candles())
  weekday_groups = result.instruments["NQ"]["1h"].slices["weekday"].groups
  # Total across all weekday groups must equal overall total_samples
  total_in_slices = sum(g.total_samples for g in weekday_groups.values())
  assert total_in_slices == 6
  # Per-weekday checks
  assert weekday_groups["tuesday"].total_samples == 2   # 01-02 and 01-09
  assert weekday_groups["wednesday"].total_samples == 1  # 01-03
  assert weekday_groups["thursday"].total_samples == 1   # 01-04
  assert weekday_groups["friday"].total_samples == 1     # 01-05
  assert weekday_groups["monday"].total_samples == 1     # 01-08


def test_weekday_slice_counts_re_partition_overall() -> None:
  """Sum of per-weekday counts for each matrix cell equals the overall count."""
  result = _stat().compute(_canon_candles())
  weekday_groups = result.instruments["NQ"]["1h"].slices["weekday"].groups

  # Overall pre_green→green count = 2 (days 1 and 3: 2024-01-02 Tue, 2024-01-04 Thu)
  overall_pg_green = _row(result, "pre_green", "green").count
  weekday_pg_green_total = sum(
    next(r.count for r in g.results if r.condition == "pre_green" and r.outcome == "green")
    for g in weekday_groups.values()
  )
  assert weekday_pg_green_total == overall_pg_green


# ===========================================================================
# Edge: empty slices on empty input
# ===========================================================================

def test_empty_input_slices_present_but_empty() -> None:
  """Slicers produce no groups on empty input (empty groups dict, no crash)."""
  result = _stat().compute(_empty_df())
  slices = result.instruments["NQ"]["1h"].slices
  # Slice dimensions are registered even for empty input
  for _, sr in slices.items():
    assert sr.groups == {}


# ===========================================================================
# Stat metadata: stat_name, title, definition, labels
# ===========================================================================

def test_stat_name() -> None:
  assert _stat().compute(_empty_df()).stat_name == "power_hour_continuation"


def test_i18n_title_and_definition() -> None:
  """title and definition both have non-empty en and fr strings."""
  result = _stat().compute(_empty_df())
  assert result.title.en != "" and result.title.fr != ""
  assert result.definition.en != "" and result.definition.fr != ""


def test_i18n_labels_have_correct_keys() -> None:
  """Condition keys are pre_green/pre_red; outcome keys are green/red."""
  result = _stat().compute(_empty_df())
  assert set(result.labels.conditions) == {"pre_green", "pre_red"}
  assert set(result.labels.outcomes) == {"green", "red"}


def test_i18n_labels_have_en_and_fr() -> None:
  """Every condition and outcome label has non-empty en and fr strings."""
  result = _stat().compute(_empty_df())
  for mapping in (result.labels.conditions, result.labels.outcomes):
    for key, i18n in mapping.items():
      assert i18n.en != "", f"{key}.en empty"
      assert i18n.fr != "", f"{key}.fr empty"


# ===========================================================================
# Invalid constructor arguments
# ===========================================================================

def test_invalid_timeframe_raises() -> None:
  """Unsupported timeframe raises ValueError."""
  with pytest.raises(ValueError, match="Unsupported timeframe"):
    PowerHourContinuation(
      instrument="NQ",
      timeframe="15min",
      config=_TEST_CONFIG,
    )


def test_unsupported_timeframe_30min_raises() -> None:
  with pytest.raises(ValueError, match="Unsupported timeframe"):
    PowerHourContinuation(
      instrument="NQ",
      timeframe="30min",
      config=_TEST_CONFIG,
    )


# ===========================================================================
# StatRunResult validation and JSON round-trip
# ===========================================================================

# ===========================================================================
# classify_samples
# ===========================================================================

def test_classify_samples_exact_list() -> None:
  """One SampleRow per resolved day, matching the 6-day canonical set's matrix cell."""
  stat = _stat()
  day_table = stat.build_day_table(_canon_candles())
  samples = stat.classify_samples(day_table)
  assert [(s.date, s.condition, s.outcome) for s in samples] == [
    ("2024-01-02", "pre_green", "green"),
    ("2024-01-03", "pre_green", "red"),
    ("2024-01-04", "pre_green", "green"),
    ("2024-01-05", "pre_red", "green"),
    ("2024-01-08", "pre_red", "red"),
    ("2024-01-09", "pre_red", "red"),
  ]
  assert all(s.value is None for s in samples)


def test_classify_samples_matches_compute_rows_counts() -> None:
  """For every StatResultRow, the matching SampleRow count equals r.count and the
  matching condition's total sample count equals r.total."""
  stat = _stat()
  day_table = stat.build_day_table(_canon_candles())
  samples = stat.classify_samples(day_table)
  rows = stat.compute_rows(day_table)
  for r in rows:
    n = sum(1 for s in samples if s.condition == r.condition and s.outcome == r.outcome)
    assert n == r.count
    cond_total = sum(1 for s in samples if s.condition == r.condition)
    assert cond_total == r.total


def test_classify_samples_empty_day_table() -> None:
  """Empty day_table -> classify_samples returns []."""
  stat = _stat()
  assert stat.classify_samples(stat.build_day_table(_empty_df())) == []


def test_classify_samples_excludes_unresolved_day() -> None:
  """An unresolved day (missing the 15:15 open bar) never yields a sample."""
  good_day = _make_cont_day(
    date="2024-01-02",
    session_open=100.0, pre_close=110.0,
    pre_high=110.0, pre_low=90.0,
    ph_open=95.0, session_close=100.0,
  )
  no_ph_open = _make_no_ph_open_bar_day("2024-01-03")
  combined = _concat([good_day, no_ph_open])
  stat = _stat()
  day_table = stat.build_day_table(combined)
  samples = stat.classify_samples(day_table)
  assert {s.date for s in samples} == {"2024-01-02"}
  assert len(samples) == 1


def test_compute_returns_valid_stat_run_result() -> None:
  """Pydantic re-validation of the serialized result must not raise."""
  result = _stat().compute(_canon_candles())
  StatRunResult.model_validate(result.model_dump())


def test_write_results_round_trip(tmp_path: Path) -> None:
  """write_results produces power_hour_continuation.json that re-validates."""
  result = _stat().compute(_canon_candles())
  written = write_results(result, results_dir=tmp_path)
  assert written.name == "power_hour_continuation.json"

  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)
  assert validated.instruments["NQ"]["1h"].total_samples == 6


def test_write_results_french_accent_literal(tmp_path: Path) -> None:
  """French text is stored as literal UTF-8, not escaped unicode."""
  result = _stat().compute(_empty_df())
  written = write_results(result, results_dir=tmp_path)
  raw = written.read_text(encoding="utf-8")
  # The French definition or labels contain accented characters
  assert "é" in raw or "è" in raw or "ê" in raw or "à" in raw
  assert "\\u00e9" not in raw
