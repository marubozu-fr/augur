"""Tests for stats.fomc_intraday.standard.

All data is synthetic — no real market files required.
Expected values are hand-calculated before each assertion.

The stat builds a per-day table for qualifying FOMC dates, indexed by date,
with 3 metrics per 15-minute RTH interval:
  pct_{key}    = (interval_close - interval_open) / interval_open
  dollar_{key} = interval_close - interval_open
  vol_{key}    = sum of 1-min volumes in [t, t+15)

Where:
  interval_open  = open  of the bar at exactly the interval start minute
  interval_close = close of the last bar in [t, t+15)

A date qualifies iff:
  (a) RTH is resolved: bar at mod=570 (09:30) AND last RTH bar mod >= 960
      (rth_end_min=975 - close_tolerance_min=15)
  (b) The date is in event_dates

Per-interval values are NaN when the interval has no bar at exactly its
start minute (pending-sample discipline at the interval level).

reaction_positive:
  1.0 = close > open for the 14:00 bar (strictly up)
  0.0 = flat or down (close <= open)
  NaN = no bar at exactly mod=840 (14:00)

With rth 09:30–16:15 the grid has 27 intervals (570..960 in steps of 15),
so compute_rows always emits 27 × 3 = 81 rows.

Key minute-of-day constants used throughout:
  _RTH_OPEN_MOD  = 570  (09:30 — RTH open bar; also i0930 open bar)
  _I0930_LAST    = 584  (09:44 — last bar in i0930 window [570, 585))
  _I0945_START   = 585  (09:45 — i0945 open bar)
  _REACT_MOD     = 840  (14:00 — i1400 open bar; also reaction open)
  _REACT_LAST    = 854  (14:14 — last bar in i1400 window [840, 855))
  _RTH_LAST_MOD  = 974  (16:14 — last RTH bar; 974 >= 960 -> resolved)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.fomc_intraday.standard import FOMCIntraday

# ---------------------------------------------------------------------------
# Minimal InstrumentConfig — does not depend on NQ.yaml
# ---------------------------------------------------------------------------
_NY = "America/New_York"
_RTH_SESSION = Session(start="09:30", end="16:15")
_TEST_CONFIG = InstrumentConfig(
  instrument="NQ",
  working_timezone=_NY,
  sessions={"rth": _RTH_SESSION},
  timeframes=["daily"],
  parquet_path=Path("data/NQ_1min.parquet"),
)

# Minute-of-day constants
_RTH_OPEN_MOD = 570   # 09:30
_I0930_LAST   = 584   # 09:44 — last bar in i0930 window [570, 585)
_I0945_START  = 585   # 09:45 — first bar of i0945
_REACT_MOD    = 840   # 14:00 — reaction open bar
_REACT_LAST   = 854   # 14:14 — last bar in i1400 window [840, 855)
_RTH_LAST_MOD = 974   # 16:14 — last RTH bar; mod 974 >= resolved_min 960

# Derived grid constants
_N_INTERVALS = 27     # range(570, 975, 15) has 27 elements
_N_ROWS = _N_INTERVALS * 3  # 81 rows per compute_rows call


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _bar(
  ts: pd.Timestamp,
  open_: float,
  close: float,
  volume: int = 500,
) -> dict:
  """Build a single OHLCV record with high/low derived from open/close."""
  return {
    "timestamp": ts,
    "open": open_,
    "high": max(open_, close) + 0.25,
    "low": min(open_, close) - 0.25,
    "close": close,
    "volume": volume,
  }


def _make_fomc_day(
  date: str,
  open_0930: float,
  close_0944: float,
  vol_0930: int,
  vol_0944: int,
  open_1400: float,
  close_1414: float,
  vol_1400: int,
  vol_1414: int,
) -> pd.DataFrame:
  """Build one fully-qualifying FOMC day with bars for i0930, i1400, and RTH close.

  Bars emitted (minute-of-day):
    570 (09:30): open=open_0930, vol=vol_0930   — RTH open + i0930 open
    584 (09:44): close=close_0944, vol=vol_0944  — last bar of i0930
    840 (14:00): open=open_1400, vol=vol_1400    — i1400 open + reaction open
    854 (14:14): close=close_1414, vol=vol_1414  — i1400 last bar + reaction close
    974 (16:14): close=130.0, vol=800            — RTH resolve bar (mod=974 >= 960)

  All other intervals have no bars, so pct/dollar/vol will be NaN for them.
  """
  base = pd.Timestamp(date, tz=_NY)

  def _ts(mod: int) -> pd.Timestamp:
    h, m = divmod(mod, 60)
    return base.replace(hour=h, minute=m, second=0, microsecond=0)

  records = [
    _bar(_ts(_RTH_OPEN_MOD), open_=open_0930, close=open_0930 + 5, volume=vol_0930),
    _bar(_ts(_I0930_LAST),   open_=open_0930 + 4, close=close_0944, volume=vol_0944),
    _bar(_ts(_REACT_MOD),    open_=open_1400, close=open_1400 + 5, volume=vol_1400),
    _bar(_ts(_REACT_LAST),   open_=open_1400 + 4, close=close_1414, volume=vol_1414),
    _bar(_ts(_RTH_LAST_MOD), open_=105.0, close=130.0, volume=800),
  ]
  return pd.DataFrame(records)


def _make_rth_day(date: str) -> pd.DataFrame:
  """Non-FOMC resolved RTH day (09:30 + 16:14 bars only)."""
  base = pd.Timestamp(date, tz=_NY)

  def _ts(mod: int) -> pd.Timestamp:
    h, m = divmod(mod, 60)
    return base.replace(hour=h, minute=m, second=0, microsecond=0)

  records = [
    _bar(_ts(_RTH_OPEN_MOD), open_=100.0, close=105.0, volume=1000),
    _bar(_ts(_RTH_LAST_MOD), open_=105.0, close=110.0, volume=1000),
  ]
  return pd.DataFrame(records)


def _make_unresolved_day(date: str) -> pd.DataFrame:
  """FOMC day where RTH is truncated: last bar mod=590 < 960 → not resolved."""
  base = pd.Timestamp(date, tz=_NY)

  def _ts(mod: int) -> pd.Timestamp:
    h, m = divmod(mod, 60)
    return base.replace(hour=h, minute=m, second=0, microsecond=0)

  records = [
    _bar(_ts(_RTH_OPEN_MOD), open_=100.0, close=101.0, volume=1000),  # 09:30
    _bar(_ts(590),           open_=101.0, close=102.0, volume=500),    # 09:50, mod<960
  ]
  return pd.DataFrame(records)


def _make_no_reaction_day(date: str) -> pd.DataFrame:
  """FOMC day resolved but with NO bar at 14:00 → reaction_positive=NaN."""
  base = pd.Timestamp(date, tz=_NY)

  def _ts(mod: int) -> pd.Timestamp:
    h, m = divmod(mod, 60)
    return base.replace(hour=h, minute=m, second=0, microsecond=0)

  records = [
    _bar(_ts(_RTH_OPEN_MOD), open_=100.0, close=105.0, volume=1000),  # 09:30
    _bar(_ts(_I0930_LAST),   open_=104.0, close=110.0, volume=1000),  # 09:44
    # NO 14:00 bar → reaction_positive = NaN
    _bar(_ts(_RTH_LAST_MOD), open_=105.0, close=110.0, volume=800),   # 16:14
  ]
  return pd.DataFrame(records)


def _make_thin_day(date: str) -> pd.DataFrame:
  """FOMC day with only 09:30, 09:44, and 16:14 bars — no bar at 09:45 (i0945).

  i0930: open bar at 570, close bar at 584 → pct/dollar/vol NOT NaN.
  i0945: no bar at 585 → pct/dollar/vol NaN for this day.
  No 14:00 bar → reaction_positive NaN.
  """
  base = pd.Timestamp(date, tz=_NY)

  def _ts(mod: int) -> pd.Timestamp:
    h, m = divmod(mod, 60)
    return base.replace(hour=h, minute=m, second=0, microsecond=0)

  records = [
    _bar(_ts(_RTH_OPEN_MOD), open_=120.0, close=125.0, volume=1200),
    _bar(_ts(_I0930_LAST),   open_=124.0, close=130.0, volume=1200),  # 09:44
    _bar(_ts(_RTH_LAST_MOD), open_=105.0, close=110.0, volume=800),   # 16:14
  ]
  return pd.DataFrame(records)


def _make_candles(frames: list[pd.DataFrame]) -> pd.DataFrame:
  """Concatenate per-day DataFrames into a single sorted 1-min candle DataFrame."""
  df = pd.concat(frames, ignore_index=True)
  return df.sort_values("timestamp").reset_index(drop=True)


def _empty_df() -> pd.DataFrame:
  df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
  df["timestamp"] = pd.array([], dtype="datetime64[ns, America/New_York]")
  return df


# ---------------------------------------------------------------------------
# Stat factory helpers
# ---------------------------------------------------------------------------

def _stat(event_dates: list[str]) -> FOMCIntraday:
  return FOMCIntraday(
    instrument="NQ",
    config=_TEST_CONFIG,
    event_dates=[pd.Timestamp(d) for d in event_dates],
  )


def _row(result: StatRunResult, condition: str, outcome: str) -> StatResultRow:
  for r in result.instruments["NQ"]["daily"].results:
    if r.condition == condition and r.outcome == outcome:
      return r
  raise KeyError((condition, outcome))


def _slice_row(
  result: StatRunResult, group: str, condition: str, outcome: str
) -> StatResultRow:
  for r in result.instruments["NQ"]["daily"].slices["reaction"].groups[group].results:
    if r.condition == condition and r.outcome == outcome:
      return r
  raise KeyError((group, condition, outcome))


# ===========================================================================
# 1. Happy path: 3 FOMC days with known per-interval values
#
# Day  Date        i0930 (open→close)  pct    dollar  vol   i1400 (open→close)  pct    dollar  vol    reaction
#  1   2024-03-20  100→110             +0.10  +10     2000  100→120             +0.20  +20     1000   positive
#  2   2024-05-01  200→180             -0.10  -20     4000  100→90              -0.10  -10     1200   negative
#  3   2024-07-31  150→150              0.00    0     3000  100→100              0.00    0     1400   negative (flat)
#
# Bar volumes:
#   Day 1 i0930: vol_0930=1000, vol_0944=1000 → sum=2000
#   Day 2 i0930: vol_0930=2000, vol_0944=2000 → sum=4000
#   Day 3 i0930: vol_0930=1500, vol_0944=1500 → sum=3000
#   Day 1 i1400: vol_1400=500,  vol_1414=500  → sum=1000
#   Day 2 i1400: vol_1400=600,  vol_1414=600  → sum=1200
#   Day 3 i1400: vol_1400=700,  vol_1414=700  → sum=1400
#
# Hand-calculated (i0930):
#   pct vals = [0.10, -0.10, 0.00]  mean=0.0  count(>=0)=2  prob=2/3
#   dollar vals = [10, -20, 0]       mean=-10/3  count(>=0)=2  prob=2/3
#   vol vals = [2000, 4000, 3000]    mean=3000.0  count=total=3  prob=1.0
#
# Hand-calculated (i1400):
#   pct vals = [0.20, -0.10, 0.00]  mean=1/30  count(>=0)=2  prob=2/3
#   dollar vals = [20, -10, 0]       mean=10/3   count(>=0)=2  prob=2/3
#   vol vals = [1000, 1200, 1400]    mean=1200.0  count=total=3  prob=1.0
#
#   total_samples = 3
#   data_range = ["2024-03-20", "2024-07-31"]
# ===========================================================================

_FOMC_DATES_3 = ["2024-03-20", "2024-05-01", "2024-07-31"]


def _build_3day_df() -> pd.DataFrame:
  return _make_candles([
    # Day 1: i0930 open=100→110, i1400 open=100→120 (positive reaction)
    _make_fomc_day("2024-03-20",
                   open_0930=100.0, close_0944=110.0, vol_0930=1000, vol_0944=1000,
                   open_1400=100.0, close_1414=120.0, vol_1400=500,  vol_1414=500),
    # Day 2: i0930 open=200→180, i1400 open=100→90 (negative reaction)
    _make_fomc_day("2024-05-01",
                   open_0930=200.0, close_0944=180.0, vol_0930=2000, vol_0944=2000,
                   open_1400=100.0, close_1414=90.0,  vol_1400=600,  vol_1414=600),
    # Day 3: i0930 open=150→150, i1400 open=100→100 (flat = not strictly up → negative)
    _make_fomc_day("2024-07-31",
                   open_0930=150.0, close_0944=150.0, vol_0930=1500, vol_0944=1500,
                   open_1400=100.0, close_1414=100.0, vol_1400=700,  vol_1414=700),
  ])


def test_total_samples_three_days() -> None:
  """total_samples == 3: all three FOMC days resolve."""
  result = _stat(_FOMC_DATES_3).compute(_build_3day_df())
  assert result.instruments["NQ"]["daily"].total_samples == 3


def test_data_range() -> None:
  """data_range spans first..last qualifying date."""
  result = _stat(_FOMC_DATES_3).compute(_build_3day_df())
  assert result.instruments["NQ"]["daily"].data_range == [
    "2024-03-20", "2024-07-31",
  ]


def test_i0930_pct_change_mean() -> None:
  # vals=[0.10, -0.10, 0.00] → mean=0.0
  result = _stat(_FOMC_DATES_3).compute(_build_3day_df())
  r = _row(result, "i0930", "pct_change")
  assert r.value == pytest.approx(0.0)


def test_i0930_pct_change_count_and_total() -> None:
  # count(>=0)=2 (days 1 and 3), total=3
  result = _stat(_FOMC_DATES_3).compute(_build_3day_df())
  r = _row(result, "i0930", "pct_change")
  assert r.total == 3
  assert r.count == 2
  assert r.probability == pytest.approx(2 / 3)


def test_i0930_dollar_change_mean() -> None:
  # vals=[10, -20, 0] → mean=-10/3
  result = _stat(_FOMC_DATES_3).compute(_build_3day_df())
  r = _row(result, "i0930", "dollar_change")
  assert r.value == pytest.approx(-10 / 3)


def test_i0930_dollar_change_count_and_total() -> None:
  # count(>=0)=2, total=3
  result = _stat(_FOMC_DATES_3).compute(_build_3day_df())
  r = _row(result, "i0930", "dollar_change")
  assert r.total == 3
  assert r.count == 2
  assert r.probability == pytest.approx(2 / 3)


def test_i0930_volume_mean() -> None:
  # vals=[2000, 4000, 3000] → mean=3000.0
  result = _stat(_FOMC_DATES_3).compute(_build_3day_df())
  r = _row(result, "i0930", "volume")
  assert r.value == pytest.approx(3000.0)


def test_i0930_volume_count_equals_total() -> None:
  # For volume: count always == total
  result = _stat(_FOMC_DATES_3).compute(_build_3day_df())
  r = _row(result, "i0930", "volume")
  assert r.total == 3
  assert r.count == 3
  assert r.probability == pytest.approx(1.0)


def test_i1400_pct_change_mean() -> None:
  # vals=[0.20, -0.10, 0.00] → mean=(0.20-0.10+0.00)/3=0.10/3=1/30
  result = _stat(_FOMC_DATES_3).compute(_build_3day_df())
  r = _row(result, "i1400", "pct_change")
  assert r.value == pytest.approx(1 / 30)


def test_i1400_pct_change_count_and_total() -> None:
  # count(>=0)=2 (days 1 and 3), total=3
  result = _stat(_FOMC_DATES_3).compute(_build_3day_df())
  r = _row(result, "i1400", "pct_change")
  assert r.total == 3
  assert r.count == 2
  assert r.probability == pytest.approx(2 / 3)


def test_i1400_dollar_change_mean() -> None:
  # vals=[20, -10, 0] → mean=10/3
  result = _stat(_FOMC_DATES_3).compute(_build_3day_df())
  r = _row(result, "i1400", "dollar_change")
  assert r.value == pytest.approx(10 / 3)


def test_i1400_volume_mean() -> None:
  # vals=[1000, 1200, 1400] → mean=1200.0
  result = _stat(_FOMC_DATES_3).compute(_build_3day_df())
  r = _row(result, "i1400", "volume")
  assert r.value == pytest.approx(1200.0)


def test_i1400_volume_count_equals_total() -> None:
  result = _stat(_FOMC_DATES_3).compute(_build_3day_df())
  r = _row(result, "i1400", "volume")
  assert r.count == r.total == 3


# ===========================================================================
# 2. total_samples and data_range (covered above; supplement with edge values)
# ===========================================================================

def test_total_samples_equals_len_day_table() -> None:
  """total_samples == len(build_day_table(df))."""
  stat = _stat(_FOMC_DATES_3)
  df = _build_3day_df()
  result = stat.compute(df)
  table = stat.build_day_table(df)
  assert result.instruments["NQ"]["daily"].total_samples == len(table)


# ===========================================================================
# 3. Reaction split: positive vs negative vs NaN
#
# Using the 3-day dataset:
#   Day 1 (2024-03-20): reaction_positive=True  → positive group
#   Day 2 (2024-05-01): reaction_positive=False → negative group
#   Day 3 (2024-07-31): reaction_positive=False (flat, not strictly up) → negative
#
# Positive group (N=1):
#   i0930 pct: [0.10] → mean=0.10, count>=0=1, total=1, prob=1.0
# Negative group (N=2):
#   i0930 pct: [-0.10, 0.00] → mean=-0.05, count>=0=1, total=2, prob=0.5
# ===========================================================================

def test_reaction_positive_group_size() -> None:
  result = _stat(_FOMC_DATES_3).compute(_build_3day_df())
  pos = result.instruments["NQ"]["daily"].slices["reaction"].groups["positive"]
  assert pos.total_samples == 1


def test_reaction_negative_group_size() -> None:
  result = _stat(_FOMC_DATES_3).compute(_build_3day_df())
  neg = result.instruments["NQ"]["daily"].slices["reaction"].groups["negative"]
  assert neg.total_samples == 2


def test_flat_reaction_goes_to_negative_group() -> None:
  """close == open at 14:00 is NOT strictly up → reaction_positive=False → negative."""
  # Day 3 has close=open=100 at 14:00; it must appear in negative, not positive
  result = _stat(_FOMC_DATES_3).compute(_build_3day_df())
  neg = result.instruments["NQ"]["daily"].slices["reaction"].groups["negative"]
  # N=2 (days 2 and 3, the flat day being one of them)
  assert neg.total_samples == 2


def test_reaction_positive_group_i0930_pct() -> None:
  # Positive group (Day 1 only): i0930 pct=0.10, count>=0=1, total=1
  result = _stat(_FOMC_DATES_3).compute(_build_3day_df())
  r = _slice_row(result, "positive", "i0930", "pct_change")
  assert r.total == 1
  assert r.count == 1
  assert r.value == pytest.approx(0.10)
  assert r.probability == pytest.approx(1.0)


def test_reaction_negative_group_i0930_pct() -> None:
  # Negative group (Days 2,3): vals=[-0.10, 0.00] → mean=-0.05, count>=0=1, prob=0.5
  result = _stat(_FOMC_DATES_3).compute(_build_3day_df())
  r = _slice_row(result, "negative", "i0930", "pct_change")
  assert r.total == 2
  assert r.count == 1
  assert r.value == pytest.approx(-0.05)
  assert r.probability == pytest.approx(0.5)


def test_reaction_groups_differ_on_i0930_pct() -> None:
  """i0930 pct_change value differs between positive and negative groups."""
  result = _stat(_FOMC_DATES_3).compute(_build_3day_df())
  pos_val = _slice_row(result, "positive", "i0930", "pct_change").value
  neg_val = _slice_row(result, "negative", "i0930", "pct_change").value
  assert pos_val != pytest.approx(neg_val)


def test_reaction_slice_exists_in_result() -> None:
  result = _stat(_FOMC_DATES_3).compute(_build_3day_df())
  slices = result.instruments["NQ"]["daily"].slices
  assert "reaction" in slices
  assert "positive" in slices["reaction"].groups
  assert "negative" in slices["reaction"].groups


def test_reaction_slice_group_results_count() -> None:
  """Each slice group has exactly 81 result rows."""
  result = _stat(_FOMC_DATES_3).compute(_build_3day_df())
  slices = result.instruments["NQ"]["daily"].slices
  for grp_name in ("positive", "negative"):
    grp = slices["reaction"].groups[grp_name]
    assert len(grp.results) == _N_ROWS, (
      f"group '{grp_name}' has {len(grp.results)} rows, expected {_N_ROWS}"
    )


# ===========================================================================
# 4a. Pending discipline: unresolved RTH day is dropped
#
# FOMC date 2024-03-20 has RTH session truncated at 09:50 (mod=590 < 960).
# It must be absent from the day table and not count in total_samples.
# ===========================================================================

def test_unresolved_rth_dropped_from_day_table() -> None:
  """FOMC day with truncated RTH is absent from build_day_table output."""
  date = "2024-03-20"
  df = _make_candles([_make_unresolved_day(date)])
  stat = _stat([date])
  table = stat.build_day_table(df)
  assert table.empty


def test_unresolved_rth_not_in_total_samples() -> None:
  """Unresolved FOMC date does not increase total_samples."""
  good = "2024-03-19"
  bad = "2024-03-20"
  df = _make_candles([
    _make_fomc_day(good,
                   open_0930=100.0, close_0944=110.0, vol_0930=1000, vol_0944=1000,
                   open_1400=100.0, close_1414=120.0, vol_1400=500, vol_1414=500),
    _make_unresolved_day(bad),
  ])
  result = _stat([good, bad]).compute(df)
  assert result.instruments["NQ"]["daily"].total_samples == 1


# ===========================================================================
# 4b. Pending discipline: missing interval open bar → NaN for THAT interval only
#
# Use a "thin" day (2024-03-20) that has bars at mod=570, mod=584, and mod=974
# but NO bar at mod=585 (i0945 start).
# Combined with one full FOMC day (2024-03-19):
#   - i0930 total = 2 (both days have a bar at mod=570)
#   - i0945 total = 1 (only the full day has a bar at mod=585)
# ===========================================================================

def _build_missing_interval_df() -> pd.DataFrame:
  # Day A (full): i0930 and i0945 both have open bars
  day_a = "2024-03-19"
  day_b = "2024-03-20"  # thin: no i0945 bar
  frames = [
    _make_fomc_day(day_a,
                   open_0930=100.0, close_0944=110.0, vol_0930=1000, vol_0944=1000,
                   open_1400=100.0, close_1414=120.0, vol_1400=500, vol_1414=500),
    _make_thin_day(day_b),
  ]
  # Add i0945 bar only to day_a
  base_a = pd.Timestamp(day_a, tz=_NY)
  extra = pd.DataFrame([
    _bar(base_a.replace(hour=9, minute=45, second=0, microsecond=0),
         open_=110.0, close=115.0, volume=900),
  ])
  return _make_candles(frames + [extra])


def test_missing_interval_bar_nan_for_that_interval() -> None:
  """Day with no bar at i0945 start contributes NaN to i0945, not to i0930."""
  stat = _stat(["2024-03-19", "2024-03-20"])
  table = stat.build_day_table(_build_missing_interval_df())
  # Both days qualify
  assert len(table) == 2
  # i0930 has no NaN (both days have bar at mod=570)
  assert table["pct_i0930"].notna().all()
  # i0945: one NaN (the thin day, 2024-03-20)
  assert table["pct_i0945"].isna().sum() == 1


def test_missing_interval_bar_reduces_that_interval_total() -> None:
  """i0945 total==1 (thin day excluded), i0930 total==2 (both days have it)."""
  stat = _stat(["2024-03-19", "2024-03-20"])
  result = stat.compute(_build_missing_interval_df())
  r0930 = _row(result, "i0930", "pct_change")
  r0945 = _row(result, "i0945", "pct_change")
  assert r0930.total == 2
  assert r0945.total == 1


# ===========================================================================
# 4c. Pending discipline: missing 14:00 bar → reaction_positive=NaN
#
# A day with no bar at mod=840 stays in total_samples but reaction_positive=NaN.
# Such a day is excluded from BOTH reaction groups (positive and negative).
# ===========================================================================

def test_missing_reaction_bar_keeps_day_in_total() -> None:
  """FOMC day with no 14:00 bar stays in total_samples."""
  date = "2024-03-20"
  df = _make_candles([_make_no_reaction_day(date)])
  result = _stat([date]).compute(df)
  assert result.instruments["NQ"]["daily"].total_samples == 1


def test_missing_reaction_bar_excluded_from_both_groups() -> None:
  """Day with reaction_positive=NaN appears in neither positive nor negative group."""
  date = "2024-03-20"
  df = _make_candles([_make_no_reaction_day(date)])
  stat = _stat([date])
  table = stat.build_day_table(df)
  assert len(table) == 1
  # reaction_positive must be NaN
  assert table["reaction_positive"].isna().all()

  # Slicer sees no non-NaN reactions → no groups emitted
  result = stat.compute(df)
  slices = result.instruments["NQ"]["daily"].slices
  assert slices["reaction"].groups == {}


def test_missing_reaction_bar_combined_with_full_days() -> None:
  """Day with NaN reaction does not inflate either group, only the overall total.

  Days used:
    2024-03-20: i1400 close=120 > open=100 → reaction_positive=True  (positive)
    2024-05-01: i1400 close=80  < open=100 → reaction_positive=False (negative)
    2024-07-31: i1400 close=100 == open=100 → reaction_positive=False (negative)
    2024-03-21: no 14:00 bar    → reaction_positive=NaN  (excluded from both)

  Expected: positive N=1, negative N=2, total_samples=4, pos_n+neg_n=3.
  """
  no_react_date = "2024-03-21"
  df = _make_candles([
    # 2024-03-20: positive reaction
    _make_fomc_day("2024-03-20",
                   open_0930=100.0, close_0944=110.0, vol_0930=1000, vol_0944=1000,
                   open_1400=100.0, close_1414=120.0, vol_1400=500, vol_1414=500),
    # 2024-05-01: negative reaction
    _make_fomc_day("2024-05-01",
                   open_0930=100.0, close_0944=110.0, vol_0930=1000, vol_0944=1000,
                   open_1400=100.0, close_1414=80.0, vol_1400=500, vol_1414=500),
    # 2024-07-31: flat reaction (close==open → not strictly up → negative)
    _make_fomc_day("2024-07-31",
                   open_0930=100.0, close_0944=110.0, vol_0930=1000, vol_0944=1000,
                   open_1400=100.0, close_1414=100.0, vol_1400=500, vol_1414=500),
    # no_react_date: no 14:00 bar → NaN
    _make_no_reaction_day(no_react_date),
  ])
  all_dates = _FOMC_DATES_3 + [no_react_date]
  result = _stat(all_dates).compute(df)
  tf = result.instruments["NQ"]["daily"]
  # Total: 4 qualifying days
  assert tf.total_samples == 4
  # Reaction groups sum: 1 positive + 2 negative = 3 (no_react_date excluded)
  pos_n = tf.slices["reaction"].groups["positive"].total_samples
  neg_n = tf.slices["reaction"].groups["negative"].total_samples
  assert pos_n + neg_n == 3


# ===========================================================================
# 5. Non-FOMC trading days excluded
#
# 3 regular trading days (not in event_dates) + 3 FOMC days.
# total_samples must equal 3 (only FOMC dates).
# ===========================================================================

_NON_FOMC_DATES = ["2024-03-18", "2024-03-19", "2024-03-21"]


def _build_mixed_df() -> pd.DataFrame:
  frames = [
    _make_rth_day("2024-03-18"),
    _make_rth_day("2024-03-19"),
    _make_fomc_day("2024-03-20",
                   open_0930=100.0, close_0944=110.0, vol_0930=1000, vol_0944=1000,
                   open_1400=100.0, close_1414=120.0, vol_1400=500, vol_1414=500),
    _make_rth_day("2024-03-21"),
    _make_fomc_day("2024-05-01",
                   open_0930=200.0, close_0944=180.0, vol_0930=2000, vol_0944=2000,
                   open_1400=100.0, close_1414=90.0, vol_1400=600, vol_1414=600),
    _make_fomc_day("2024-07-31",
                   open_0930=150.0, close_0944=150.0, vol_0930=1500, vol_0944=1500,
                   open_1400=100.0, close_1414=100.0, vol_1400=700, vol_1414=700),
  ]
  return _make_candles(frames)


def test_non_fomc_days_excluded_from_total_samples() -> None:
  """3 non-FOMC trading days are present in candles but not in event_dates."""
  result = _stat(_FOMC_DATES_3).compute(_build_mixed_df())
  assert result.instruments["NQ"]["daily"].total_samples == 3


def test_non_fomc_days_do_not_change_interval_totals() -> None:
  """Non-FOMC days do not inflate interval totals."""
  result = _stat(_FOMC_DATES_3).compute(_build_mixed_df())
  r = _row(result, "i0930", "pct_change")
  assert r.total == 3


# ===========================================================================
# 6. Volume: value_baseline is None; count == total for every volume row
# ===========================================================================

def test_volume_value_baseline_is_none() -> None:
  """value_baseline must be None for every volume row after compute()."""
  result = _stat(_FOMC_DATES_3).compute(_build_3day_df())
  for r in result.instruments["NQ"]["daily"].results:
    if r.outcome == "volume":
      assert r.value_baseline is None, (
        f"volume row {r.condition} has non-None value_baseline"
      )


def test_volume_count_equals_total_for_all_rows() -> None:
  """For volume rows: count == total (every non-NaN day has volume)."""
  result = _stat(_FOMC_DATES_3).compute(_build_3day_df())
  for r in result.instruments["NQ"]["daily"].results:
    if r.outcome == "volume" and r.total > 0:
      assert r.count == r.total, (
        f"volume row {r.condition}: count={r.count} != total={r.total}"
      )


# ===========================================================================
# 7. Baseline reproducibility
# ===========================================================================

def test_baseline_same_seed_deterministic() -> None:
  """baseline_rows(seed=42) is byte-for-byte identical on two calls."""
  stat = _stat(_FOMC_DATES_3)
  df = _build_3day_df()
  table = stat.build_day_table(df)
  rows_a = stat.baseline_rows(table, seed=42)
  rows_b = stat.baseline_rows(table, seed=42)
  assert len(rows_a) == len(rows_b) == _N_ROWS
  for a, b in zip(rows_a, rows_b):
    assert (a.condition, a.outcome) == (b.condition, b.outcome)
    assert a.count == b.count
    assert a.total == b.total
    assert a.probability == pytest.approx(b.probability)


def test_baseline_different_seeds_may_differ() -> None:
  """seed=42 and seed=99 produce at least one row with a different probability."""
  stat = _stat(_FOMC_DATES_3)
  table = stat.build_day_table(_build_3day_df())
  rows_42 = stat.baseline_rows(table, seed=42)
  rows_99 = stat.baseline_rows(table, seed=99)
  any_diff = any(
    abs(a.probability - b.probability) > 1e-9
    for a, b in zip(rows_42, rows_99)
  )
  assert any_diff, "seed=42 and seed=99 produced identical baseline probabilities"


def test_baseline_preserves_interval_totals() -> None:
  """Randomising sign does not change per-interval totals."""
  stat = _stat(_FOMC_DATES_3)
  table = stat.build_day_table(_build_3day_df())
  real_rows = stat.compute_rows(table)
  bl_rows = stat.baseline_rows(table, seed=42)
  for real, bl in zip(real_rows, bl_rows):
    assert (real.condition, real.outcome) == (bl.condition, bl.outcome)
    assert real.total == bl.total


def test_baseline_pct_dollar_not_none_after_compute() -> None:
  """After compute(), pct/dollar rows have baseline_n>0 and value_baseline not None."""
  result = _stat(_FOMC_DATES_3).compute(_build_3day_df())
  for r in result.instruments["NQ"]["daily"].results:
    if r.outcome in ("pct_change", "dollar_change") and r.total > 0:
      assert r.baseline_n > 0, (
        f"baseline_n=0 for {r.condition}/{r.outcome}"
      )
      assert r.value_baseline is not None, (
        f"value_baseline is None for {r.condition}/{r.outcome}"
      )


def test_baseline_volume_value_baseline_none_after_compute() -> None:
  """Volume rows always have value_baseline=None regardless of seed."""
  result = _stat(_FOMC_DATES_3).compute(_build_3day_df())
  for r in result.instruments["NQ"]["daily"].results:
    if r.outcome == "volume":
      assert r.value_baseline is None


# ===========================================================================
# 8. Empty cases: empty event_dates and empty candles DataFrame
# ===========================================================================

def test_empty_event_dates_day_table_empty() -> None:
  """event_dates={} → build_day_table returns empty DataFrame."""
  stat = _stat([])
  table = stat.build_day_table(_build_3day_df())
  assert table.empty


def test_empty_event_dates_zero_total_samples() -> None:
  result = _stat([]).compute(_build_3day_df())
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 0
  assert tf.data_range == []


def test_empty_event_dates_all_zero_rows() -> None:
  """event_dates={} → compute() returns 81 zeroed rows, no crash."""
  result = _stat([]).compute(_build_3day_df())
  for r in result.instruments["NQ"]["daily"].results:
    assert r.count == 0
    assert r.total == 0
    assert r.probability == pytest.approx(0.0)


def test_empty_event_dates_no_reaction_groups() -> None:
  """event_dates={} → no days → reaction groups are empty."""
  result = _stat([]).compute(_build_3day_df())
  slices = result.instruments["NQ"]["daily"].slices
  assert slices["reaction"].groups == {}


def test_empty_candles_day_table_empty() -> None:
  """Empty candles DataFrame → build_day_table returns empty, no crash."""
  stat = _stat(["2024-03-20"])
  table = stat.build_day_table(_empty_df())
  assert table.empty


def test_empty_candles_zero_total_samples() -> None:
  result = _stat(["2024-03-20"]).compute(_empty_df())
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 0
  assert tf.data_range == []


def test_empty_candles_all_zero_rows() -> None:
  """Empty candles → compute() returns 81 zeroed rows, no crash."""
  result = _stat(["2024-03-20"]).compute(_empty_df())
  for r in result.instruments["NQ"]["daily"].results:
    assert r.count == 0
    assert r.total == 0
    assert r.probability == pytest.approx(0.0)


def test_empty_candles_no_reaction_groups() -> None:
  result = _stat(["2024-03-20"]).compute(_empty_df())
  slices = result.instruments["NQ"]["daily"].slices
  assert slices["reaction"].groups == {}


def test_both_empty_no_crash() -> None:
  """No event_dates and empty candles → no crash, 81 zero rows."""
  result = _stat([]).compute(_empty_df())
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 0
  assert len(tf.results) == _N_ROWS


# ===========================================================================
# 9. Result-shape invariants
# ===========================================================================

def test_exactly_n_rows() -> None:
  """compute() always returns exactly 81 rows (27 intervals × 3 metrics)."""
  result = _stat(_FOMC_DATES_3).compute(_build_3day_df())
  rows = result.instruments["NQ"]["daily"].results
  assert len(rows) == _N_ROWS


def test_result_condition_outcome_keys() -> None:
  """The (condition, outcome) pairs match the 27-interval × 3-metric specification."""
  result = _stat(_FOMC_DATES_3).compute(_build_3day_df())
  rows = result.instruments["NQ"]["daily"].results
  pairs = {(r.condition, r.outcome) for r in rows}

  # Expected interval keys: range(570, 975, 15)
  expected_keys = [
    f"i{t // 60:02d}{t % 60:02d}" for t in range(570, 975, 15)
  ]
  expected_pairs = {
    (key, outcome)
    for key in expected_keys
    for outcome in ("pct_change", "dollar_change", "volume")
  }
  assert pairs == expected_pairs


def test_result_timeframe_is_daily() -> None:
  result = _stat(_FOMC_DATES_3).compute(_build_3day_df())
  assert "daily" in result.instruments["NQ"]


def test_interval_keys_include_i0930_and_i1600() -> None:
  """Grid starts at i0930 (09:30) and ends at i1600 (16:00)."""
  result = _stat(_FOMC_DATES_3).compute(_build_3day_df())
  rows = result.instruments["NQ"]["daily"].results
  conditions = {r.condition for r in rows}
  assert "i0930" in conditions
  assert "i1600" in conditions
  assert "i1615" not in conditions  # rth_end_min not included


def test_no_extra_conditions() -> None:
  """Conditions are exactly the 27 interval keys — nothing more, nothing less."""
  result = _stat(_FOMC_DATES_3).compute(_build_3day_df())
  conditions = {r.condition for r in result.instruments["NQ"]["daily"].results}
  assert len(conditions) == _N_INTERVALS


# ===========================================================================
# 10. stat_name, i18n, and write_results round-trip
# ===========================================================================

def test_stat_name() -> None:
  result = _stat([]).compute(_empty_df())
  assert result.stat_name == "fomc_intraday"


def test_i18n_title_non_empty() -> None:
  result = _stat([]).compute(_empty_df())
  assert result.title.en != ""
  assert result.title.fr != ""


def test_i18n_definition_non_empty() -> None:
  result = _stat([]).compute(_empty_df())
  assert result.definition.en != ""
  assert result.definition.fr != ""


def test_i18n_labels_conditions_cover_all_intervals() -> None:
  """Every interval key has a condition label in both en and fr."""
  result = _stat([]).compute(_empty_df())
  expected_keys = {
    f"i{t // 60:02d}{t % 60:02d}" for t in range(570, 975, 15)
  }
  assert set(result.labels.conditions.keys()) == expected_keys
  for key, label in result.labels.conditions.items():
    assert label.en != "", f"{key}.en is empty"
    assert label.fr != "", f"{key}.fr is empty"


def test_i18n_labels_outcomes_cover_three_metrics() -> None:
  """Outcome labels cover pct_change, dollar_change, volume in both languages."""
  result = _stat([]).compute(_empty_df())
  assert set(result.labels.outcomes.keys()) == {
    "pct_change", "dollar_change", "volume"
  }
  for key, label in result.labels.outcomes.items():
    assert label.en != "", f"{key}.en is empty"
    assert label.fr != "", f"{key}.fr is empty"


def test_write_results_round_trip(tmp_path: Path) -> None:
  """write_results produces fomc_intraday.json that re-validates correctly."""
  stat = _stat(_FOMC_DATES_3)
  result = stat.compute(_build_3day_df())
  written = write_results(result, results_dir=tmp_path)
  assert written.name == "fomc_intraday.json"

  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)
  assert validated.stat_name == "fomc_intraday"
  assert "NQ" in validated.instruments
  tf = validated.instruments["NQ"]["daily"]
  assert tf.total_samples == 3
  assert len(tf.results) == _N_ROWS
  conditions = {r.condition for r in tf.results}
  assert "i0930" in conditions
  assert "i1400" in conditions


def test_write_results_french_accents_not_escaped(tmp_path: Path) -> None:
  """French labels are stored as UTF-8 literals, not \\uXXXX escape sequences."""
  result = _stat([]).compute(_empty_df())
  written = write_results(result, results_dir=tmp_path)
  raw = written.read_text(encoding="utf-8")
  # French definition and slice labels contain accented characters (é, è)
  assert "é" in raw
  assert "\\u00e9" not in raw


def test_compute_result_validates_as_stat_run_result() -> None:
  """compute() output re-validates cleanly as a StatRunResult model."""
  result = _stat(_FOMC_DATES_3).compute(_build_3day_df())
  revalidated = StatRunResult.model_validate(result.model_dump())
  assert revalidated.stat_name == "fomc_intraday"
  rows = revalidated.instruments["NQ"]["daily"].results
  assert len(rows) == _N_ROWS
