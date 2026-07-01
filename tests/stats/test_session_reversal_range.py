"""Tests for stats.session_reversal_range.standard.

All data is synthetic — no real market files required.
Expected values are hand-calculated before each assertion.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.session_reversal_range.standard import SessionReversalRange
from tests.stats.range_helpers import (
  _NY,
  _TEST_CONFIG,
  _empty_df,
  _make_truncated_day,
)

# RTH window: [570, 975).  09:30 = 570, 16:15 = 975.
# Resolution: bar at exactly 570 AND last bar mod >= 975 - 15 = 960 (>= 16:00).
_RTH_START = 570   # 09:30
_RTH_LAST = 974    # 16:14  (last bar before 16:15)
# Number of 1-min bars in a full RTH session: 974 - 570 + 1 = 405
_N_BARS = _RTH_LAST - _RTH_START + 1  # 405


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

def _make_day(
  date: str,
  session_open: float,
  day_high: float,
  day_low: float,
  session_close: float,
) -> pd.DataFrame:
  """Build one full RTH trading day of 1-min OHLCV bars (09:30 – 16:14).

  The 09:30 bar sets session_open and its high is forced to day_high.
  The 16:14 bar's low is forced to day_low and its close is set to
  session_close (the framework reads session_close = close of last RTH bar).
  Interior bars use neutral OHLC strictly inside (day_low, day_high) so they
  never violate the intended extremes.

  Guarantees: max(high) == day_high, min(low) == day_low, open of 09:30 bar
  == session_open, close of 16:14 bar == session_close.  session_open and
  session_close must lie within [day_low, day_high] in the caller's test data.
  """
  base = pd.Timestamp(date, tz=_NY)
  mid = (day_high + day_low) / 2.0
  neutral_high = mid + 0.25
  neutral_low = mid - 0.25
  records = []
  for mod in range(_RTH_START, _RTH_LAST + 1):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    is_first = mod == _RTH_START
    is_last = mod == _RTH_LAST
    o = session_open if is_first else mid
    c = session_close if is_last else mid
    bar_high = day_high if is_first else neutral_high
    bar_low = day_low if is_last else neutral_low
    records.append({
      "timestamp": ts,
      "open": o,
      "high": bar_high,
      "low": bar_low,
      "close": c,
      "volume": 100,
    })
  return pd.DataFrame(records)


def make_candles(days: list[dict]) -> pd.DataFrame:
  """Concatenate per-day DataFrames into a sorted multi-day OHLCV frame."""
  frames = [
    _make_day(
      d["date"],
      d["session_open"],
      d["day_high"],
      d["day_low"],
      d["session_close"],
    )
    for d in days
  ]
  df = pd.concat(frames, ignore_index=True)
  return df.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main synthetic dataset: 10 resolved days, 2 per weekday.
#
# Dates and weekdays (2024 calendar):
#   2024-01-01 Monday    open=100, high=112, low= 88, close=105  GREEN
#   2024-01-02 Tuesday   open=200, high=225, low=182, close=195  RED
#   2024-01-03 Wednesday open=150, high=168, low=130, close=158  GREEN
#   2024-01-04 Thursday  open=100, high=118, low= 84, close= 92  RED
#   2024-01-05 Friday    open=200, high=214, low=192, close=206  GREEN
#   2024-01-08 Monday    open=100, high=122, low= 78, close=112  GREEN
#   2024-01-09 Tuesday   open=200, high=235, low=172, close=188  RED
#   2024-01-10 Wednesday open=150, high=182, low=118, close=162  GREEN
#   2024-01-11 Thursday  open=100, high=114, low= 88, close= 96  RED
#   2024-01-12 Friday    open=200, high=245, low=162, close=215  GREEN
#
# Per-day derived values:
#   down_excursion = session_open - day_low
#   up_excursion   = day_high - session_open
#   reversal       = down_excursion (GREEN) or up_excursion (RED)
#   reversal_pct   = reversal / session_open
#
#   01-01 GREEN: down=12, up=12, rev=12, rev_pct= 12/100 = 0.12
#   01-02 RED:   down=18, up=25, rev=25, rev_pct= 25/200 = 0.125
#   01-03 GREEN: down=20, up=18, rev=20, rev_pct= 20/150 = 0.13333...
#   01-04 RED:   down=16, up=18, rev=18, rev_pct= 18/100 = 0.18
#   01-05 GREEN: down= 8, up=14, rev= 8, rev_pct=  8/200 = 0.04
#   01-08 GREEN: down=22, up=22, rev=22, rev_pct= 22/100 = 0.22
#   01-09 RED:   down=28, up=35, rev=35, rev_pct= 35/200 = 0.175
#   01-10 GREEN: down=32, up=32, rev=32, rev_pct= 32/150 = 0.21333...
#   01-11 RED:   down=12, up=14, rev=14, rev_pct= 14/100 = 0.14
#   01-12 GREEN: down=38, up=45, rev=38, rev_pct= 38/200 = 0.19
#
# Overall (10 days):
#   mean_reversal     = (12+25+20+18+8+22+35+32+14+38) / 10 = 224 / 10 = 22.4
#   mean_reversal_pct = 461/3000 = 0.15366...  (exact fraction sum)
#   max_reversal      = 38
#   max_reversal_pct  = 22/100 = 0.22  (day 01-08)
#
# Close slice — GREEN (6 days: 01-01,01-03,01-05,01-08,01-10,01-12):
#   reversals:     [12, 20, 8, 22, 32, 38]
#   mean_reversal  = 132 / 6 = 22.0
#   max_reversal   = 38
#   mean_rev_pct   = (12/100 + 20/150 + 8/200 + 22/100 + 32/150 + 38/200) / 6
#                  = 11/72 = 0.15277...
#   max_rev_pct    = 22/100 = 0.22
#
# Close slice — RED (4 days: 01-02,01-04,01-09,01-11):
#   reversals:     [25, 18, 35, 14]
#   mean_reversal  = 92 / 4 = 23.0
#   max_reversal   = 35
#   mean_rev_pct   = (25/200 + 18/100 + 35/200 + 14/100) / 4
#                  = 31/200 = 0.155
#   max_rev_pct    = 18/100 = 0.18
#
# Weekday slice (2 days each):
#   Monday:    reversals=[12,22]  mean=17.0  rev_pcts=[0.12,0.22]  mean_pct=0.17  max=22  max_pct=0.22
#   Tuesday:   reversals=[25,35]  mean=30.0  rev_pcts=[0.125,0.175] mean_pct=0.15  max=35  max_pct=0.175
#   Wednesday: reversals=[20,32]  mean=26.0  rev_pcts=[20/150,32/150] mean_pct=52/300=0.17333 max=32 max_pct=32/150=0.21333
#   Thursday:  reversals=[18,14]  mean=16.0  rev_pcts=[0.18,0.14]  mean_pct=0.16  max=18  max_pct=0.18
#   Friday:    reversals=[8,38]   mean=23.0  rev_pcts=[0.04,0.19]  mean_pct=0.115 max=38  max_pct=0.19
# ---------------------------------------------------------------------------

_DAYS = [
  {"date": "2024-01-01", "session_open": 100.0, "day_high": 112.0, "day_low":  88.0, "session_close": 105.0},
  {"date": "2024-01-02", "session_open": 200.0, "day_high": 225.0, "day_low": 182.0, "session_close": 195.0},
  {"date": "2024-01-03", "session_open": 150.0, "day_high": 168.0, "day_low": 130.0, "session_close": 158.0},
  {"date": "2024-01-04", "session_open": 100.0, "day_high": 118.0, "day_low":  84.0, "session_close":  92.0},
  {"date": "2024-01-05", "session_open": 200.0, "day_high": 214.0, "day_low": 192.0, "session_close": 206.0},
  {"date": "2024-01-08", "session_open": 100.0, "day_high": 122.0, "day_low":  78.0, "session_close": 112.0},
  {"date": "2024-01-09", "session_open": 200.0, "day_high": 235.0, "day_low": 172.0, "session_close": 188.0},
  {"date": "2024-01-10", "session_open": 150.0, "day_high": 182.0, "day_low": 118.0, "session_close": 162.0},
  {"date": "2024-01-11", "session_open": 100.0, "day_high": 114.0, "day_low":  88.0, "session_close":  96.0},
  {"date": "2024-01-12", "session_open": 200.0, "day_high": 245.0, "day_low": 162.0, "session_close": 215.0},
]


def _stat() -> SessionReversalRange:
  return SessionReversalRange(instrument="NQ", config=_TEST_CONFIG)


def _row(rows: list[StatResultRow], outcome: str) -> StatResultRow:
  for r in rows:
    if r.condition == "session_reversal_range" and r.outcome == outcome:
      return r
  raise KeyError(outcome)


def _overall(result: StatRunResult, outcome: str) -> StatResultRow:
  return _row(result.instruments["NQ"]["daily"].results, outcome)


def _weekday_rows(result: StatRunResult, day_key: str) -> list[StatResultRow]:
  return result.instruments["NQ"]["daily"].slices["weekday"].groups[day_key].results


def _close_rows(result: StatRunResult, color_key: str) -> list[StatResultRow]:
  return result.instruments["NQ"]["daily"].slices["close"].groups[color_key].results


# ===========================================================================
# 1. build_day_table: per-row correctness, resolution filtering, index
# ===========================================================================

def test_build_day_table_row_count() -> None:
  """All 10 fully resolved days produce exactly 10 rows."""
  dt = _stat().build_day_table(make_candles(_DAYS))
  assert len(dt) == 10


def test_build_day_table_columns_present() -> None:
  """The table must carry the six metric/flag columns."""
  dt = _stat().build_day_table(make_candles(_DAYS))
  assert set(dt.columns) >= {
    "session_open", "down_excursion", "up_excursion",
    "session_green", "reversal", "reversal_pct",
  }


def test_build_day_table_index_is_session_date() -> None:
  """Index is normalized timestamps, one per resolved day."""
  dt = _stat().build_day_table(make_candles(_DAYS))
  expected = {
    pd.Timestamp("2024-01-01", tz=_NY).normalize(),
    pd.Timestamp("2024-01-12", tz=_NY).normalize(),
  }
  assert expected <= set(dt.index)


def test_build_day_table_green_day_reversal_is_open_to_low() -> None:
  """Green session (01-01): reversal == session_open - day_low = 100 - 88 = 12."""
  dt = _stat().build_day_table(make_candles(_DAYS))
  date = pd.Timestamp("2024-01-01", tz=_NY).normalize()
  # session_green=True, down_excursion = 100 - 88 = 12
  assert dt.loc[date, "session_green"] is True or bool(dt.loc[date, "session_green"])
  # reversal = down_excursion = 12
  assert dt.loc[date, "reversal"] == pytest.approx(12.0)
  assert dt.loc[date, "reversal_pct"] == pytest.approx(12.0 / 100.0)


def test_build_day_table_red_day_reversal_is_open_to_high() -> None:
  """Red session (01-02): reversal == day_high - session_open = 225 - 200 = 25."""
  dt = _stat().build_day_table(make_candles(_DAYS))
  date = pd.Timestamp("2024-01-02", tz=_NY).normalize()
  # session_green=False (close=195 < open=200), up_excursion = 225 - 200 = 25
  assert not bool(dt.loc[date, "session_green"])
  # reversal = up_excursion = 25
  assert dt.loc[date, "reversal"] == pytest.approx(25.0)
  assert dt.loc[date, "reversal_pct"] == pytest.approx(25.0 / 200.0)


def test_build_day_table_asymmetric_red_day() -> None:
  """01-09 RED: down=28, up=35; reversal must pick up_excursion=35, not down."""
  dt = _stat().build_day_table(make_candles(_DAYS))
  date = pd.Timestamp("2024-01-09", tz=_NY).normalize()
  # open=200, high=235, low=172, close=188; red (188<200)
  # up_excursion = 235 - 200 = 35
  assert dt.loc[date, "reversal"] == pytest.approx(35.0)


def test_build_day_table_asymmetric_green_day() -> None:
  """01-03 GREEN: down=20, up=18; reversal must pick down_excursion=20, not up."""
  dt = _stat().build_day_table(make_candles(_DAYS))
  date = pd.Timestamp("2024-01-03", tz=_NY).normalize()
  # open=150, high=168, low=130, close=158; green (158>150)
  # down_excursion = 150 - 130 = 20
  assert dt.loc[date, "reversal"] == pytest.approx(20.0)


def test_build_day_table_reversal_pct_is_reversal_over_open() -> None:
  """01-03 GREEN: rev_pct = 20 / 150 = 0.13333."""
  dt = _stat().build_day_table(make_candles(_DAYS))
  date = pd.Timestamp("2024-01-03", tz=_NY).normalize()
  # reversal = 20, session_open = 150 => rev_pct = 20/150
  assert dt.loc[date, "reversal_pct"] == pytest.approx(20.0 / 150.0)


def test_build_day_table_excludes_truncated_day() -> None:
  """A day ending at 09:50 (mod 590 < 960) is excluded as unresolved."""
  base_df = make_candles(_DAYS)
  trunc = _make_truncated_day("2024-01-15")
  df = pd.concat([base_df, trunc], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
  dt = _stat().build_day_table(df)
  excluded = pd.Timestamp("2024-01-15", tz=_NY).normalize()
  assert excluded not in dt.index
  assert len(dt) == 10


def test_build_day_table_sorted_chronologically() -> None:
  """Index must be sorted in ascending date order."""
  dt = _stat().build_day_table(make_candles(_DAYS))
  assert list(dt.index) == sorted(dt.index)


def test_build_day_table_zero_down_excursion_green_session() -> None:
  """A green day where session_open == day_low: down_excursion=0, reversal=0."""
  # session_open=100 == day_low=100, day_high=110, session_close=105 (green)
  day = [{"date": "2024-02-01", "session_open": 100.0, "day_high": 110.0, "day_low": 100.0, "session_close": 105.0}]
  dt = _stat().build_day_table(make_candles(day))
  date = pd.Timestamp("2024-02-01", tz=_NY).normalize()
  # down_excursion = 100 - 100 = 0; session is green => reversal = 0
  assert dt.loc[date, "down_excursion"] == pytest.approx(0.0)
  assert dt.loc[date, "reversal"] == pytest.approx(0.0)
  assert dt.loc[date, "reversal_pct"] == pytest.approx(0.0)


# ===========================================================================
# 2. compute_rows: overall magnitudes, counts, probability channel
# ===========================================================================

def test_overall_four_rows() -> None:
  """Exactly four outcome rows under the single session_reversal_range condition."""
  result = _stat().compute(make_candles(_DAYS))
  rows = result.instruments["NQ"]["daily"].results
  assert len(rows) == 4
  assert {r.outcome for r in rows} == {
    "mean_reversal", "mean_reversal_pct", "max_reversal", "max_reversal_pct",
  }
  assert {r.condition for r in rows} == {"session_reversal_range"}


def test_overall_mean_reversal() -> None:
  """mean_reversal = (12+25+20+18+8+22+35+32+14+38) / 10 = 224 / 10 = 22.4."""
  result = _stat().compute(make_candles(_DAYS))
  row = _overall(result, "mean_reversal")
  # sum: 12+25+20+18+8+22+35+32+14+38 = 224; mean = 22.4
  assert row.value == pytest.approx(22.4)
  assert row.count == 10
  assert row.total == 10
  assert row.probability == pytest.approx(0.0)
  assert row.baseline_prob == pytest.approx(0.0)


def test_overall_mean_reversal_pct() -> None:
  """mean_reversal_pct = 461/3000 = 0.153666...."""
  result = _stat().compute(make_candles(_DAYS))
  row = _overall(result, "mean_reversal_pct")
  # rev_pcts: 0.12, 0.125, 20/150, 0.18, 0.04, 0.22, 0.175, 32/150, 0.14, 0.19
  # sum = 461/300; mean = 461/3000 = 0.15366...
  assert row.value == pytest.approx(461.0 / 3000.0)
  assert row.count == 10
  assert row.total == 10
  assert row.probability == pytest.approx(0.0)


def test_overall_max_reversal() -> None:
  """max_reversal = 38 (day 2024-01-12, green, down_excursion=200-162=38)."""
  result = _stat().compute(make_candles(_DAYS))
  row = _overall(result, "max_reversal")
  # reversals: [12,25,20,18,8,22,35,32,14,38]; max = 38
  assert row.value == pytest.approx(38.0)
  assert row.count == 10
  assert row.total == 10
  assert row.probability == pytest.approx(0.0)


def test_overall_max_reversal_pct() -> None:
  """max_reversal_pct = 0.22 (day 2024-01-08, green, rev=22, open=100)."""
  result = _stat().compute(make_candles(_DAYS))
  row = _overall(result, "max_reversal_pct")
  # max of [0.12, 0.125, 20/150, 0.18, 0.04, 0.22, 0.175, 32/150, 0.14, 0.19]
  # = 0.22 (22/100 from 2024-01-08)
  assert row.value == pytest.approx(0.22)
  assert row.count == 10
  assert row.total == 10
  assert row.probability == pytest.approx(0.0)


def test_overall_data_range() -> None:
  """data_range spans first to last resolved session date."""
  result = _stat().compute(make_candles(_DAYS))
  assert result.instruments["NQ"]["daily"].data_range == ["2024-01-01", "2024-01-12"]


def test_overall_total_samples() -> None:
  """total_samples equals the number of resolved days."""
  result = _stat().compute(make_candles(_DAYS))
  assert result.instruments["NQ"]["daily"].total_samples == 10


def test_overall_count_equals_total_equals_total_samples() -> None:
  """Every row: count == total == total_samples (no warm-up window)."""
  result = _stat().compute(make_candles(_DAYS))
  tf = result.instruments["NQ"]["daily"]
  for row in tf.results:
    assert row.count == 10
    assert row.total == 10


# ===========================================================================
# 3. close slice: green vs red group correctness
# ===========================================================================

def test_close_slice_two_groups() -> None:
  """Two groups under the close slice: green and red."""
  result = _stat().compute(make_candles(_DAYS))
  groups = result.instruments["NQ"]["daily"].slices["close"].groups
  assert set(groups.keys()) == {"green", "red"}


def test_close_slice_sample_counts() -> None:
  """Green group: 6 days; red group: 4 days."""
  result = _stat().compute(make_candles(_DAYS))
  groups = result.instruments["NQ"]["daily"].slices["close"].groups
  # GREEN days: 01-01,01-03,01-05,01-08,01-10,01-12 = 6
  # RED days:   01-02,01-04,01-09,01-11             = 4
  assert groups["green"].total_samples == 6
  assert groups["red"].total_samples == 4


def test_close_green_mean_reversal() -> None:
  """Green group mean_reversal = (12+20+8+22+32+38) / 6 = 132 / 6 = 22.0."""
  result = _stat().compute(make_candles(_DAYS))
  row = _row(_close_rows(result, "green"), "mean_reversal")
  # Green reversals: [12, 20, 8, 22, 32, 38]; sum=132; mean=22.0
  assert row.value == pytest.approx(22.0)
  assert row.count == 6
  assert row.total == 6
  assert row.probability == pytest.approx(0.0)


def test_close_green_max_reversal() -> None:
  """Green group max_reversal = 38 (2024-01-12)."""
  result = _stat().compute(make_candles(_DAYS))
  row = _row(_close_rows(result, "green"), "max_reversal")
  # max([12, 20, 8, 22, 32, 38]) = 38
  assert row.value == pytest.approx(38.0)


def test_close_green_mean_reversal_pct() -> None:
  """Green group mean_reversal_pct = 11/72 = 0.152777...."""
  result = _stat().compute(make_candles(_DAYS))
  row = _row(_close_rows(result, "green"), "mean_reversal_pct")
  # pcts: 12/100, 20/150, 8/200, 22/100, 32/150, 38/200
  # = 0.12 + 0.1333... + 0.04 + 0.22 + 0.2133... + 0.19
  # sum = 11/12; mean = 11/72 = 0.15277...
  assert row.value == pytest.approx(11.0 / 72.0)


def test_close_green_max_reversal_pct() -> None:
  """Green group max_reversal_pct = 0.22 (2024-01-08, rev=22, open=100)."""
  result = _stat().compute(make_candles(_DAYS))
  row = _row(_close_rows(result, "green"), "max_reversal_pct")
  # max([0.12, 20/150, 0.04, 0.22, 32/150, 0.19]) = 0.22
  assert row.value == pytest.approx(0.22)


def test_close_red_mean_reversal() -> None:
  """Red group mean_reversal = (25+18+35+14) / 4 = 92 / 4 = 23.0."""
  result = _stat().compute(make_candles(_DAYS))
  row = _row(_close_rows(result, "red"), "mean_reversal")
  # Red reversals: [25, 18, 35, 14]; sum=92; mean=23.0
  assert row.value == pytest.approx(23.0)
  assert row.count == 4
  assert row.total == 4
  assert row.probability == pytest.approx(0.0)


def test_close_red_max_reversal() -> None:
  """Red group max_reversal = 35 (2024-01-09, up_excursion=235-200=35)."""
  result = _stat().compute(make_candles(_DAYS))
  row = _row(_close_rows(result, "red"), "max_reversal")
  # max([25, 18, 35, 14]) = 35
  assert row.value == pytest.approx(35.0)


def test_close_red_mean_reversal_pct() -> None:
  """Red group mean_reversal_pct = 31/200 = 0.155."""
  result = _stat().compute(make_candles(_DAYS))
  row = _row(_close_rows(result, "red"), "mean_reversal_pct")
  # pcts: 25/200, 18/100, 35/200, 14/100 = 0.125+0.18+0.175+0.14 = 0.62
  # mean = 0.62/4 = 0.155 = 31/200
  assert row.value == pytest.approx(31.0 / 200.0)


def test_close_red_max_reversal_pct() -> None:
  """Red group max_reversal_pct = 0.18 (2024-01-04, rev=18, open=100)."""
  result = _stat().compute(make_candles(_DAYS))
  row = _row(_close_rows(result, "red"), "max_reversal_pct")
  # max([0.125, 0.18, 0.175, 0.14]) = 0.18
  assert row.value == pytest.approx(0.18)


# ===========================================================================
# 4. weekday slice: correct groups and per-weekday means / maxes
# ===========================================================================

def test_weekday_slice_five_groups() -> None:
  """Five weekday groups are present, each with 2 days."""
  result = _stat().compute(make_candles(_DAYS))
  groups = result.instruments["NQ"]["daily"].slices["weekday"].groups
  assert set(groups.keys()) == {"monday", "tuesday", "wednesday", "thursday", "friday"}
  for g in groups.values():
    assert g.total_samples == 2


def test_weekday_mean_reversals() -> None:
  """Per-weekday average reversals match hand-calculated values."""
  result = _stat().compute(make_candles(_DAYS))
  # Monday:    (12+22)/2 = 17.0
  # Tuesday:   (25+35)/2 = 30.0
  # Wednesday: (20+32)/2 = 26.0
  # Thursday:  (18+14)/2 = 16.0
  # Friday:    (8+38)/2  = 23.0
  expected = {
    "monday": 17.0,
    "tuesday": 30.0,
    "wednesday": 26.0,
    "thursday": 16.0,
    "friday": 23.0,
  }
  for day_key, exp in expected.items():
    row = _row(_weekday_rows(result, day_key), "mean_reversal")
    assert row.value == pytest.approx(exp), f"{day_key}: expected {exp}, got {row.value}"


def test_weekday_mean_reversal_pcts() -> None:
  """Per-weekday average reversal_pct values match hand-calculated values."""
  result = _stat().compute(make_candles(_DAYS))
  # Monday:    (12/100 + 22/100) / 2 = 34/200 = 0.17
  # Tuesday:   (25/200 + 35/200) / 2 = 60/400 = 0.15
  # Wednesday: (20/150 + 32/150) / 2 = 52/300 = 0.17333...
  # Thursday:  (18/100 + 14/100) / 2 = 32/200 = 0.16
  # Friday:    (8/200  + 38/200) / 2 = 46/400 = 0.115
  expected = {
    "monday":    0.17,
    "tuesday":   0.15,
    "wednesday": 52.0 / 300.0,
    "thursday":  0.16,
    "friday":    0.115,
  }
  for day_key, exp in expected.items():
    row = _row(_weekday_rows(result, day_key), "mean_reversal_pct")
    assert row.value == pytest.approx(exp), f"{day_key}: expected {exp}, got {row.value}"


def test_weekday_max_reversals() -> None:
  """Per-weekday max reversals match hand-calculated values."""
  result = _stat().compute(make_candles(_DAYS))
  # Monday:    max(12,22) = 22
  # Tuesday:   max(25,35) = 35
  # Wednesday: max(20,32) = 32
  # Thursday:  max(18,14) = 18
  # Friday:    max(8,38)  = 38
  expected = {
    "monday": 22.0,
    "tuesday": 35.0,
    "wednesday": 32.0,
    "thursday": 18.0,
    "friday": 38.0,
  }
  for day_key, exp in expected.items():
    row = _row(_weekday_rows(result, day_key), "max_reversal")
    assert row.value == pytest.approx(exp), f"{day_key}: expected {exp}, got {row.value}"


def test_weekday_slice_count_and_total() -> None:
  """Each weekday slice row carries count == total == 2, probability == 0.0."""
  result = _stat().compute(make_candles(_DAYS))
  for day_key in ("monday", "tuesday", "wednesday", "thursday", "friday"):
    for outcome in ("mean_reversal", "mean_reversal_pct", "max_reversal", "max_reversal_pct"):
      row = _row(_weekday_rows(result, day_key), outcome)
      assert row.count == 2, f"{day_key}/{outcome}: count"
      assert row.total == 2, f"{day_key}/{outcome}: total"
      assert row.probability == pytest.approx(0.0), f"{day_key}/{outcome}: probability"


# ===========================================================================
# 5. Determinism / reproducibility
# ===========================================================================

def test_compute_twice_same_result() -> None:
  """compute() with the default seed returns identical value_baseline on both calls."""
  df = make_candles(_DAYS)
  stat = _stat()
  result_a = stat.compute(df)
  result_b = stat.compute(df)
  rows_a = result_a.instruments["NQ"]["daily"].results
  rows_b = result_b.instruments["NQ"]["daily"].results
  assert len(rows_a) == len(rows_b)
  for a, b in zip(rows_a, rows_b):
    assert a.outcome == b.outcome
    assert a.value == pytest.approx(b.value)
    if a.value_baseline is None:
      assert b.value_baseline is None
    else:
      assert a.value_baseline == pytest.approx(b.value_baseline)


def test_baseline_deterministic_fixed_seed() -> None:
  """Same seed -> identical baseline_rows output."""
  df = make_candles(_DAYS)
  stat = _stat()
  day_table = stat.build_day_table(df)
  rows_a = stat.baseline_rows(day_table, seed=7)
  rows_b = stat.baseline_rows(day_table, seed=7)
  for a, b in zip(rows_a, rows_b):
    assert a.outcome == b.outcome
    if a.value is None:
      assert b.value is None
    else:
      assert a.value == pytest.approx(b.value)


def test_overall_baseline_populated() -> None:
  """value_baseline is not None on every overall row."""
  result = _stat().compute(make_candles(_DAYS))
  for row in result.instruments["NQ"]["daily"].results:
    assert row.value_baseline is not None, f"Missing value_baseline for {row.outcome}"


def test_different_seeds_differ() -> None:
  """Different seeds produce different baseline values (overwhelmingly likely)."""
  df = make_candles(_DAYS)
  stat = _stat()
  day_table = stat.build_day_table(df)
  rows_seed1 = stat.baseline_rows(day_table, seed=1)
  rows_seed2 = stat.baseline_rows(day_table, seed=2)
  all_same = all(
    abs((a.value or 0.0) - (b.value or 0.0)) < 1e-9
    for a, b in zip(rows_seed1, rows_seed2)
  )
  assert not all_same


# ===========================================================================
# 6. Baseline correctness: independently reproduce the coin-flip
# ===========================================================================

def test_baseline_values_match_independent_coin_flip() -> None:
  """Independently reproduce the coin-flip baseline for a 3-day dataset.

  The stat baseline_rows() uses np.random.default_rng(seed).random(n) < 0.5
  to assign each row to down_excursion (True) or up_excursion (False).  We
  replicate the exact same RNG call here to confirm the stat's output matches.

  3-day dataset (chronological order per build_resolved_days sort):
    01-01 GREEN: open=100, high=112, low=88;  down=12, up=12
    01-02 RED:   open=200, high=225, low=182; down=18, up=25
    01-03 GREEN: open=150, high=168, low=130; down=20, up=18

  Seed=42: rng.random(3) = [0.7739, 0.4388, 0.8585]
    pick_down = [False, True, False]
    reversal_bl = [up[0]=12, down[1]=18, up[2]=18] = [12, 18, 18]
    rev_pct_bl  = [12/100, 18/200, 18/150] = [0.12, 0.09, 0.12]
    mean_reversal_bl     = (12+18+18)/3 = 16.0
    mean_reversal_pct_bl = (0.12+0.09+0.12)/3 = 0.11
    max_reversal_bl      = 18
    max_reversal_pct_bl  = 0.12
  """
  three_days = _DAYS[:3]  # 01-01, 01-02, 01-03 in chronological order
  df = make_candles(three_days)
  stat = _stat()
  day_table = stat.build_day_table(df)

  # Independently replicate the coin flip with seed=42
  n = len(day_table)
  assert n == 3
  down = np.array([100.0 - 88.0, 200.0 - 182.0, 150.0 - 130.0])   # [12, 18, 20]
  up   = np.array([112.0 - 100.0, 225.0 - 200.0, 168.0 - 150.0])  # [12, 25, 18]
  s_open = np.array([100.0, 200.0, 150.0])

  seed = 42
  rng = np.random.default_rng(seed)
  pick_down = rng.random(n) < 0.5
  # pick_down = [False, True, False]
  reversal_bl = np.where(pick_down, down, up)
  rev_pct_bl = reversal_bl / s_open

  expected_mean_reversal     = float(reversal_bl.mean())  # 16.0
  expected_mean_reversal_pct = float(rev_pct_bl.mean())   # 0.11
  expected_max_reversal      = float(reversal_bl.max())   # 18.0
  expected_max_reversal_pct  = float(rev_pct_bl.max())    # 0.12

  bl_rows = stat.baseline_rows(day_table, seed=42)
  bl_map = {r.outcome: r for r in bl_rows}

  assert bl_map["mean_reversal"].value == pytest.approx(expected_mean_reversal)
  assert bl_map["mean_reversal_pct"].value == pytest.approx(expected_mean_reversal_pct)
  assert bl_map["max_reversal"].value == pytest.approx(expected_max_reversal)
  assert bl_map["max_reversal_pct"].value == pytest.approx(expected_max_reversal_pct)


def test_overall_baseline_values_seed42() -> None:
  """Verify compute(seed=42) value_baseline matches independent coin-flip for 10 days.

  Seed=42 on 10 rows:
    coins:      [0.7739, 0.4388, 0.8585, 0.6973, 0.0941, 0.9756, 0.7611, 0.7860, 0.1281, 0.4503]
    pick_down:  [False, True, False, False, True, False, False, False, True, True]
    reversal_bl = [up[0]=12, down[1]=18, up[2]=18, up[3]=18, down[4]=8, up[5]=22,
                   up[6]=35, up[7]=32, down[8]=12, down[9]=38]
               = [12, 18, 18, 18, 8, 22, 35, 32, 12, 38]
    mean_reversal_bl     = 213/10 = 21.3
    max_reversal_bl      = 38
    rev_pct_bl = [12/100,18/200,18/150,18/100,8/200,22/100,35/200,32/150,12/100,38/200]
    mean_reversal_pct_bl = sum above / 10 = 0.146833...
    max_reversal_pct_bl  = 22/100 = 0.22
  """
  df = make_candles(_DAYS)
  result = _stat().compute(df)
  rows = result.instruments["NQ"]["daily"].results
  bl_map = {r.outcome: r.value_baseline for r in rows}

  # mean_reversal baseline: (12+18+18+18+8+22+35+32+12+38)/10 = 213/10 = 21.3
  assert bl_map["mean_reversal"] == pytest.approx(21.3)

  # max_reversal baseline: max([12,18,18,18,8,22,35,32,12,38]) = 38
  assert bl_map["max_reversal"] == pytest.approx(38.0)

  # mean_reversal_pct baseline: reproduce independently
  down_arr = np.array([12., 18., 20., 16.,  8., 22., 28., 32., 12., 38.])
  up_arr   = np.array([12., 25., 18., 18., 14., 22., 35., 32., 14., 45.])
  s_open_arr = np.array([100., 200., 150., 100., 200., 100., 200., 150., 100., 200.])
  rng2 = np.random.default_rng(42)
  pick2 = rng2.random(10) < 0.5
  bl2 = np.where(pick2, down_arr, up_arr)
  expected_mean_pct_bl = float((bl2 / s_open_arr).mean())

  assert bl_map["mean_reversal_pct"] == pytest.approx(expected_mean_pct_bl)

  # max_reversal_pct baseline: max of bl2/s_open_arr
  expected_max_pct_bl = float((bl2 / s_open_arr).max())
  assert bl_map["max_reversal_pct"] == pytest.approx(expected_max_pct_bl)


# ===========================================================================
# 7. Edge cases
# ===========================================================================

def test_empty_dataframe_no_crash() -> None:
  """Empty input: 0 samples, empty data_range, no crash."""
  result = _stat().compute(_empty_df())
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 0
  assert tf.data_range == []


def test_empty_dataframe_zero_counts() -> None:
  """All rows report count == total == 0, probability == 0.0, value == 0.0."""
  result = _stat().compute(_empty_df())
  for row in result.instruments["NQ"]["daily"].results:
    assert row.count == 0
    assert row.total == 0
    assert row.probability == pytest.approx(0.0)
    # Magnitude rows fall back to 0.0 when there are no days.
    assert row.value == pytest.approx(0.0)


def test_empty_dataframe_four_rows_present() -> None:
  """Even with no data, all four outcome rows are still emitted."""
  result = _stat().compute(_empty_df())
  outcomes = {r.outcome for r in result.instruments["NQ"]["daily"].results}
  assert outcomes == {"mean_reversal", "mean_reversal_pct", "max_reversal", "max_reversal_pct"}


def test_empty_dataframe_no_slice_groups() -> None:
  """No weekday or close groups when there is no data."""
  result = _stat().compute(_empty_df())
  tf = result.instruments["NQ"]["daily"]
  assert tf.slices["weekday"].groups == {}
  assert tf.slices["close"].groups == {}


def test_single_day_no_crash() -> None:
  """A single resolved day works without division errors."""
  single = [_DAYS[0]]  # 2024-01-01 Monday GREEN
  result = _stat().compute(make_candles(single))
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 1


def test_single_day_values_correct() -> None:
  """With one green day, all four metrics equal the single-day reversal values.

  2024-01-01: open=100, low=88, close=105 (GREEN)
  down_excursion = 100 - 88 = 12
  reversal = 12 (green uses down_excursion)
  reversal_pct = 12/100 = 0.12
  mean and max collapse to the single value.
  """
  single = [_DAYS[0]]
  result = _stat().compute(make_candles(single))
  # mean_reversal = max_reversal = 12.0
  assert _overall(result, "mean_reversal").value == pytest.approx(12.0)
  assert _overall(result, "max_reversal").value == pytest.approx(12.0)
  # mean_reversal_pct = max_reversal_pct = 0.12
  assert _overall(result, "mean_reversal_pct").value == pytest.approx(0.12)
  assert _overall(result, "max_reversal_pct").value == pytest.approx(0.12)


def test_single_day_weekday_group_only_monday() -> None:
  """Single Monday day produces only a monday weekday group."""
  single = [_DAYS[0]]
  result = _stat().compute(make_candles(single))
  groups = result.instruments["NQ"]["daily"].slices["weekday"].groups
  assert set(groups.keys()) == {"monday"}
  assert groups["monday"].total_samples == 1


def test_single_day_close_slice_only_green() -> None:
  """Single green day produces only a green close group."""
  single = [_DAYS[0]]  # 2024-01-01 GREEN
  result = _stat().compute(make_candles(single))
  groups = result.instruments["NQ"]["daily"].slices["close"].groups
  assert set(groups.keys()) == {"green"}
  assert groups["green"].total_samples == 1


def test_truncated_day_excluded_from_total() -> None:
  """Appending a truncated day does not change total_samples."""
  base_df = make_candles(_DAYS)
  base_total = _stat().compute(base_df).instruments["NQ"]["daily"].total_samples

  trunc = _make_truncated_day("2024-01-15")
  extended = (
    pd.concat([base_df, trunc], ignore_index=True)
    .sort_values("timestamp")
    .reset_index(drop=True)
  )
  new_total = _stat().compute(extended).instruments["NQ"]["daily"].total_samples
  assert new_total == base_total == 10


def test_truncated_day_absent_from_day_table() -> None:
  """A standalone truncated day produces an empty day table."""
  trunc = _make_truncated_day("2024-01-15")
  dt = _stat().build_day_table(trunc)
  excluded = pd.Timestamp("2024-01-15", tz=_NY).normalize()
  assert excluded not in dt.index
  assert len(dt) == 0


def test_zero_down_excursion_reversal_zero_for_green() -> None:
  """Green session where session_open == day_low: reversal == 0."""
  # open=100 == day_low=100, high=110, close=105 (green)
  day = [{"date": "2024-02-05", "session_open": 100.0, "day_high": 110.0, "day_low": 100.0, "session_close": 105.0}]
  result = _stat().compute(make_candles(day))
  # reversal = down_excursion = 100 - 100 = 0
  assert _overall(result, "mean_reversal").value == pytest.approx(0.0)
  assert _overall(result, "max_reversal").value == pytest.approx(0.0)
  assert _overall(result, "mean_reversal_pct").value == pytest.approx(0.0)
  assert _overall(result, "max_reversal_pct").value == pytest.approx(0.0)


# ===========================================================================
# 7b. classify_samples
#
# Reusing _DAYS (10 days). Every resolved day is countable under the single
# "session_reversal_range" condition, so each day yields exactly four
# SampleRows — one per outcome — carrying that day's raw metric:
#   mean_reversal / max_reversal         -> value = reversal
#   mean_reversal_pct / max_reversal_pct -> value = reversal_pct
# 10 days * 4 outcomes = 40 samples total.
# ===========================================================================

def test_classify_samples_exact_list() -> None:
  """classify_samples emits four SampleRows per day, matching the hand-calc."""
  stat = _stat()
  day_table = stat.build_day_table(make_candles(_DAYS))
  samples = stat.classify_samples(day_table)

  # (date, reversal, reversal_pct) per the module docstring hand-calc table.
  expected_days = [
    ("2024-01-01", 12.0, 0.12),
    ("2024-01-02", 25.0, 0.125),
    ("2024-01-03", 20.0, 20.0 / 150.0),
    ("2024-01-04", 18.0, 0.18),
    ("2024-01-05", 8.0, 0.04),
    ("2024-01-08", 22.0, 0.22),
    ("2024-01-09", 35.0, 0.175),
    ("2024-01-10", 32.0, 32.0 / 150.0),
    ("2024-01-11", 14.0, 0.14),
    ("2024-01-12", 38.0, 0.19),
  ]
  expected: list[tuple[str, str, str, float]] = []
  for date, reversal, reversal_pct in expected_days:
    expected.append((date, "session_reversal_range", "mean_reversal", reversal))
    expected.append((date, "session_reversal_range", "mean_reversal_pct", reversal_pct))
    expected.append((date, "session_reversal_range", "max_reversal", reversal))
    expected.append((date, "session_reversal_range", "max_reversal_pct", reversal_pct))

  assert len(samples) == len(expected) == 40
  for s, (date, condition, outcome, value) in zip(samples, expected):
    assert s.date == date
    assert s.condition == condition
    assert s.outcome == outcome
    assert s.value == pytest.approx(value)


def test_classify_samples_matches_compute_rows_counts() -> None:
  """For every StatResultRow, the matching SampleRow count equals r.count."""
  stat = _stat()
  day_table = stat.build_day_table(make_candles(_DAYS))
  samples = stat.classify_samples(day_table)
  rows = stat.compute_rows(day_table)
  for r in rows:
    n = sum(1 for s in samples if s.condition == r.condition and s.outcome == r.outcome)
    assert n == r.count


def test_classify_samples_mean_outcomes_average_matches_compute_rows() -> None:
  """Mean of mean_reversal / mean_reversal_pct SampleRow values matches compute_rows."""
  stat = _stat()
  day_table = stat.build_day_table(make_candles(_DAYS))
  samples = stat.classify_samples(day_table)
  rows = {r.outcome: r for r in stat.compute_rows(day_table)}

  for outcome in ("mean_reversal", "mean_reversal_pct"):
    values = [s.value for s in samples if s.outcome == outcome]
    assert sum(values) / len(values) == pytest.approx(rows[outcome].value)


def test_classify_samples_max_outcomes_max_matches_compute_rows() -> None:
  """Max of max_reversal / max_reversal_pct SampleRow values matches compute_rows."""
  stat = _stat()
  day_table = stat.build_day_table(make_candles(_DAYS))
  samples = stat.classify_samples(day_table)
  rows = {r.outcome: r for r in stat.compute_rows(day_table)}

  for outcome in ("max_reversal", "max_reversal_pct"):
    values = [s.value for s in samples if s.outcome == outcome]
    assert max(values) == pytest.approx(rows[outcome].value)


def test_classify_samples_empty_day_table() -> None:
  """Empty day_table -> classify_samples returns []."""
  stat = _stat()
  assert stat.classify_samples(stat.build_day_table(_empty_df())) == []


def test_classify_samples_single_day() -> None:
  """A single resolved day yields exactly four samples, one per outcome."""
  single = [_DAYS[0]]  # 2024-01-01 GREEN, reversal=12, reversal_pct=0.12
  stat = _stat()
  day_table = stat.build_day_table(make_candles(single))
  samples = stat.classify_samples(day_table)
  assert len(samples) == 4
  by_outcome = {s.outcome: s.value for s in samples}
  assert by_outcome == {
    "mean_reversal": pytest.approx(12.0),
    "mean_reversal_pct": pytest.approx(0.12),
    "max_reversal": pytest.approx(12.0),
    "max_reversal_pct": pytest.approx(0.12),
  }
  assert {s.date for s in samples} == {"2024-01-01"}
  assert {s.condition for s in samples} == {"session_reversal_range"}


# ===========================================================================
# 8. End-to-end: StatRunResult validity and write_results round-trip
# ===========================================================================

def test_stat_name() -> None:
  """stat_name attribute must equal 'session_reversal_range'."""
  result = _stat().compute(_empty_df())
  assert result.stat_name == "session_reversal_range"


def test_result_is_stat_run_result_instance() -> None:
  """compute() returns a valid StatRunResult."""
  result = _stat().compute(make_candles(_DAYS))
  assert isinstance(result, StatRunResult)


def test_i18n_title_and_definition() -> None:
  """Both title and definition carry non-empty en/fr strings."""
  result = _stat().compute(_empty_df())
  assert result.title.en != "" and result.title.fr != ""
  assert result.definition.en != "" and result.definition.fr != ""


def test_i18n_labels_outcomes_complete() -> None:
  """Labels cover all four outcomes plus the session_reversal_range condition."""
  result = _stat().compute(_empty_df())
  assert set(result.labels.outcomes) == {
    "mean_reversal", "mean_reversal_pct", "max_reversal", "max_reversal_pct",
  }
  assert "session_reversal_range" in result.labels.conditions


def test_i18n_labels_have_en_and_fr() -> None:
  """Every condition and outcome label has non-empty en and fr strings."""
  result = _stat().compute(_empty_df())
  for mapping in (result.labels.conditions, result.labels.outcomes):
    for key, i18n in mapping.items():
      assert i18n.en != "", f"{key}.en empty"
      assert i18n.fr != "", f"{key}.fr empty"


def test_labels_dimensions_contain_weekday_and_close() -> None:
  """Both weekday and close dimension labels are present."""
  result = _stat().compute(make_candles(_DAYS))
  assert "weekday" in result.labels.dimensions
  assert "close" in result.labels.dimensions
  assert result.labels.dimensions["weekday"].en != ""
  assert result.labels.dimensions["close"].en != ""


def test_write_results_round_trip(tmp_path: Path) -> None:
  """write_results serialises to JSON; the file validates back to StatRunResult."""
  result = _stat().compute(make_candles(_DAYS))
  written = write_results(result, results_dir=tmp_path)

  assert written.name == "session_reversal_range.json"
  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)

  tf = validated.instruments["NQ"]["daily"]
  assert tf.total_samples == 10

  # Spot-check Monday weekday slice
  monday = tf.slices["weekday"].groups["monday"]
  mean_row = next(r for r in monday.results if r.outcome == "mean_reversal")
  # Monday mean_reversal = (12+22)/2 = 17.0
  assert mean_row.value == pytest.approx(17.0)

  # Spot-check green close slice
  green = tf.slices["close"].groups["green"]
  max_row = next(r for r in green.results if r.outcome == "max_reversal")
  # Green max_reversal = 38
  assert max_row.value == pytest.approx(38.0)
