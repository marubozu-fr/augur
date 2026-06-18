"""Tests for stats.high_low_weekday.standard.

All data is synthetic — no real market files required.
Expected values are hand-calculated before each assertion.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, TimeframeResult, write_results
from stats.config import InstrumentConfig, Session
from stats.high_low_weekday.standard import HighLowWeekday

# ---------------------------------------------------------------------------
# Minimal InstrumentConfig — does NOT depend on NQ.yaml
# ---------------------------------------------------------------------------
_NY = "America/New_York"
_RTH_SESSION = Session(start="09:30", end="16:15")
_TEST_CONFIG = InstrumentConfig(
  instrument="NQ",
  working_timezone=_NY,
  sessions={"rth": _RTH_SESSION},
  timeframes=["weekly"],
  parquet_path=Path("data/NQ_1min.parquet"),
)

# RTH minutes: 09:30 = 570, 16:15 = 975
# Resolved when last bar mod >= 975 - 15 = 960  (i.e., >= 16:00)
_RTH_START = 570   # 09:30
_RTH_LAST = 974    # 16:14  (last 1-min bar before 16:15)


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

def _make_day(
  date: str,
  session_open: float,
  session_close: float,
  day_high: float,
  day_low: float,
) -> pd.DataFrame:
  """Build one trading day of 1-min RTH OHLCV bars (09:30–16:14).

  Constraints enforced by caller: day_high >= max(session_open, session_close)
  and day_low <= min(session_open, session_close).

  - The 09:30 bar uses session_open as its open.
  - The 16:14 bar uses session_close as its close.
  - The 09:31 bar carries day_high as its high (above all other bars).
  - The 09:32 bar carries day_low as its low (below all other bars).
  - All other bars are flat at 100.0 with high=100.25, low=99.75.

  This guarantees max(high over RTH) == day_high and min(low over RTH) == day_low.
  """
  assert day_high > max(session_open, session_close) + 0.25, "day_high too close to open/close"
  assert day_low < min(session_open, session_close) - 0.25, "day_low too close to open/close"
  base = pd.Timestamp(date, tz=_NY)
  records = []
  for mod in range(_RTH_START, _RTH_LAST + 1):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)

    if mod == _RTH_START:
      # session open bar
      o = session_open
      c = 100.0
      # For the open bar, high/low stay conservative (the special bars handle extremes)
      bar_high = max(session_open, 100.0) + 0.25
      bar_low = min(session_open, 100.0) - 0.25
    elif mod == _RTH_LAST:
      # session close bar
      o = 100.0
      c = session_close
      bar_high = max(session_close, 100.0) + 0.25
      bar_low = min(session_close, 100.0) - 0.25
    elif mod == _RTH_START + 1:
      # bar carrying the day_high
      o = 100.0
      c = 100.0
      bar_high = day_high
      bar_low = 99.75
    elif mod == _RTH_START + 2:
      # bar carrying the day_low
      o = 100.0
      c = 100.0
      bar_high = 100.25
      bar_low = day_low
    else:
      o = 100.0
      c = 100.0
      bar_high = 100.25
      bar_low = 99.75

    records.append({
      "timestamp": ts,
      "open": o,
      "high": bar_high,
      "low": bar_low,
      "close": c,
      "volume": 1000,
    })

  return pd.DataFrame(records)


def _make_truncated_day(date: str) -> pd.DataFrame:
  """A day whose last RTH bar is at 09:50 (mod 590 < 960) — excluded as unresolved."""
  base = pd.Timestamp(date, tz=_NY)
  records = []
  for mod in range(_RTH_START, 591):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    records.append({
      "timestamp": ts,
      "open": 100.0,
      "high": 100.25,
      "low": 99.75,
      "close": 100.0,
      "volume": 500,
    })
  return pd.DataFrame(records)


def _concat_days(frames: list[pd.DataFrame]) -> pd.DataFrame:
  """Concatenate per-day DataFrames into a single sorted 1-min DataFrame."""
  df = pd.concat(frames, ignore_index=True)
  return df.sort_values("timestamp").reset_index(drop=True)


def _stat() -> HighLowWeekday:
  return HighLowWeekday(instrument="NQ", config=_TEST_CONFIG)


# ---------------------------------------------------------------------------
# Three-week synthetic dataset
#
# Calendar dates and their ISO (year, week) + weekday:
#
#  Week 1 — ISO 2024w01 (Mon–Fri):
#    2024-01-01 Mon(0)  open=100 close=155 day_high=200 day_low=90
#    2024-01-02 Tue(1)  open=100 close=145 day_high=150 day_low=85
#    2024-01-03 Wed(2)  open=100 close=165 day_high=170 day_low=95
#    2024-01-04 Thu(3)  open=100 close=135 day_high=140 day_low=88
#    2024-01-05 Fri(4)  open=100 close=125 day_high=130 day_low=92
#    => weekly_high on Mon(0): max day_high = 200
#    => weekly_low  on Tue(1): min day_low  = 85
#    => high_close  on Wed(2): max close    = 165
#    => low_close   on Fri(4): min close    = 125
#    => weekly candle: open=100 (Mon open), close=125 (Fri close) -> green (125>=100)
#
#  Week 2 — ISO 2024w02 (Mon–Fri):
#    2024-01-08 Mon(0)  open=100 close=110 day_high=130 day_low=80
#    2024-01-09 Tue(1)  open=100 close=120 day_high=140 day_low=90
#    2024-01-10 Wed(2)  open=100 close=105 day_high=150 day_low=75
#    2024-01-11 Thu(3)  open=100 close=130 day_high=180 day_low=85
#    2024-01-12 Fri(4)  open=100 close=95  day_high=120 day_low=88
#    => weekly_high on Thu(3): max day_high = 180
#    => weekly_low  on Wed(2): min day_low  = 75
#    => high_close  on Thu(3): max close    = 130
#    => low_close   on Fri(4): min close    = 95
#    => weekly candle: open=100 (Mon open), close=95 (Fri close) -> red (95<100)
#
#  Week 3 — ISO 2024w03 (Mon–Fri):
#    2024-01-15 Mon(0)  open=100 close=108 day_high=110 day_low=85
#    2024-01-16 Tue(1)  open=100 close=155 day_high=160 day_low=80
#    2024-01-17 Wed(2)  open=100 close=118 day_high=120 day_low=90
#    2024-01-18 Thu(3)  open=100 close=112 day_high=115 day_low=88
#    2024-01-19 Fri(4)  open=100 close=122 day_high=125 day_low=86
#    => weekly_high on Tue(1): max day_high = 160
#    => weekly_low  on Tue(1): min day_low  = 80
#    => high_close  on Tue(1): max close    = 155
#    => low_close   on Mon(0): min close    = 108
#    => weekly candle: open=100 (Mon open), close=122 (Fri close) -> green (122>=100)
#
#  Week 4 — ISO 2024w04 — PENDING (always dropped as final ISO week):
#    2024-01-22 Mon(0)  open=100 close=110 day_high=115 day_low=95
#    [Only Mon present — still the final ISO week, must be dropped]
#
# Aggregated over resolved weeks (1, 2, 3):
#   weekly_high:      Mon=1, Tue=1, Thu=1  -> each = 1/3
#   weekly_low:       Tue=2, Wed=1         -> Tue=2/3, Wed=1/3
#   weekly_high_close:Tue=1, Wed=1, Thu=1  -> each = 1/3
#   weekly_low_close: Mon=1, Fri=2         -> Mon=1/3, Fri=2/3
#   total_samples = 3
#
#   By_close differs from intraday (Week 1): weekly_high on Mon(0) but high_close on Wed(2)
#
#   WeeklyCandle slicer groups:
#     green (weeks 1,3): weekly_high Mon=1,Tue=1; weekly_low Tue=2; high_close Wed=1,Tue=1; low_close Fri=1,Mon=1
#     red   (week 2):    weekly_high Thu=1;       weekly_low Wed=1; high_close Thu=1;        low_close Fri=1
# ---------------------------------------------------------------------------

_W1 = [
  ("2024-01-01", 100.0, 155.0, 200.0, 90.0),   # Mon ISO 2024w01
  ("2024-01-02", 100.0, 145.0, 150.0, 85.0),   # Tue ISO 2024w01
  ("2024-01-03", 100.0, 165.0, 170.0, 95.0),   # Wed ISO 2024w01
  ("2024-01-04", 100.0, 135.0, 140.0, 88.0),   # Thu ISO 2024w01
  ("2024-01-05", 100.0, 125.0, 130.0, 92.0),   # Fri ISO 2024w01
]
_W2 = [
  ("2024-01-08", 100.0, 110.0, 130.0, 80.0),   # Mon ISO 2024w02
  ("2024-01-09", 100.0, 120.0, 140.0, 90.0),   # Tue ISO 2024w02
  ("2024-01-10", 100.0, 105.0, 150.0, 75.0),   # Wed ISO 2024w02
  ("2024-01-11", 100.0, 130.0, 180.0, 85.0),   # Thu ISO 2024w02
  ("2024-01-12", 100.0, 95.0,  120.0, 88.0),   # Fri ISO 2024w02
]
_W3 = [
  ("2024-01-15", 100.0, 108.0, 110.0, 85.0),   # Mon ISO 2024w03
  ("2024-01-16", 100.0, 155.0, 160.0, 80.0),   # Tue ISO 2024w03
  ("2024-01-17", 100.0, 118.0, 120.0, 90.0),   # Wed ISO 2024w03
  ("2024-01-18", 100.0, 112.0, 115.0, 88.0),   # Thu ISO 2024w03
  ("2024-01-19", 100.0, 122.0, 125.0, 86.0),   # Fri ISO 2024w03
]
_W4_PENDING = [
  ("2024-01-22", 100.0, 110.0, 115.0, 95.0),   # Mon ISO 2024w04 — pending
]


def _make_three_weeks() -> pd.DataFrame:
  """Three resolved weeks (2024w01–w03) + one trailing pending week (2024w04)."""
  frames = []
  for specs in (_W1, _W2, _W3, _W4_PENDING):
    for date, o, c, h, day_lo in specs:
      frames.append(_make_day(date, o, c, h, day_lo))
  return _concat_days(frames)


def _row(rows: list[StatResultRow], condition: str, outcome: str) -> StatResultRow:
  for r in rows:
    if r.condition == condition and r.outcome == outcome:
      return r
  raise KeyError(f"({condition}, {outcome})")


def _tf(result: StatRunResult) -> TimeframeResult:
  return result.instruments["NQ"]["weekly"]


# ===========================================================================
# 1. build_day_table: per-week extreme attribution
# ===========================================================================

def test_build_day_table_three_weeks_resolved() -> None:
  """Three full ISO weeks + one trailing pending week -> table has 3 rows."""
  day_table = _stat().build_day_table(_make_three_weeks())
  # Pending week 2024w04 is always dropped; 3 resolved weeks remain.
  assert len(day_table) == 3


def test_build_day_table_week1_extremes() -> None:
  """Week 1 (ISO 2024w01): high on Mon(0), low on Tue(1), high_close on Wed(2), low_close on Fri(4)."""
  day_table = _stat().build_day_table(_make_three_weeks())
  # Index = first session date of each week. Week 1 starts 2024-01-01.
  w1 = pd.Timestamp("2024-01-01", tz=_NY).normalize()
  assert int(day_table.loc[w1, "high_weekday"]) == 0      # Monday
  assert int(day_table.loc[w1, "low_weekday"]) == 1       # Tuesday
  assert int(day_table.loc[w1, "high_close_weekday"]) == 2  # Wednesday
  assert int(day_table.loc[w1, "low_close_weekday"]) == 4   # Friday


def test_build_day_table_week2_extremes() -> None:
  """Week 2 (ISO 2024w02): high on Thu(3), low on Wed(2), high_close on Thu(3), low_close on Fri(4)."""
  day_table = _stat().build_day_table(_make_three_weeks())
  w2 = pd.Timestamp("2024-01-08", tz=_NY).normalize()
  assert int(day_table.loc[w2, "high_weekday"]) == 3       # Thursday
  assert int(day_table.loc[w2, "low_weekday"]) == 2        # Wednesday
  assert int(day_table.loc[w2, "high_close_weekday"]) == 3  # Thursday
  assert int(day_table.loc[w2, "low_close_weekday"]) == 4   # Friday


def test_build_day_table_week3_extremes() -> None:
  """Week 3 (ISO 2024w03): high on Tue(1), low on Tue(1), high_close on Tue(1), low_close on Mon(0)."""
  day_table = _stat().build_day_table(_make_three_weeks())
  w3 = pd.Timestamp("2024-01-15", tz=_NY).normalize()
  assert int(day_table.loc[w3, "high_weekday"]) == 1        # Tuesday
  assert int(day_table.loc[w3, "low_weekday"]) == 1         # Tuesday
  assert int(day_table.loc[w3, "high_close_weekday"]) == 1  # Tuesday
  assert int(day_table.loc[w3, "low_close_weekday"]) == 0   # Monday


def test_build_day_table_weekly_green_flags() -> None:
  """Week 1: green (125>=100), Week 2: red (95<100), Week 3: green (122>=100)."""
  day_table = _stat().build_day_table(_make_three_weeks())
  w1 = pd.Timestamp("2024-01-01", tz=_NY).normalize()
  w2 = pd.Timestamp("2024-01-08", tz=_NY).normalize()
  w3 = pd.Timestamp("2024-01-15", tz=_NY).normalize()
  assert bool(day_table.loc[w1, "weekly_green"]) is True
  assert bool(day_table.loc[w2, "weekly_green"]) is False
  assert bool(day_table.loc[w3, "weekly_green"]) is True


def test_build_day_table_present_weekdays() -> None:
  """All five days present in each resolved week."""
  day_table = _stat().build_day_table(_make_three_weeks())
  for row_ts in day_table.index:
    pw = day_table.loc[row_ts, "present_weekdays"]
    assert tuple(sorted(pw)) == (0, 1, 2, 3, 4), f"Row {row_ts}: {pw}"


# ===========================================================================
# 2. compute_rows: aggregated counts and probabilities
# ===========================================================================

def test_compute_rows_total_samples() -> None:
  """3 resolved weeks -> total_samples == 3."""
  result = _stat().compute(_make_three_weeks())
  assert _tf(result).total_samples == 3


def test_compute_rows_weekly_high_counts() -> None:
  """weekly_high: Mon=1, Tue=1, Thu=1 out of 3 weeks.

  Week1->Mon(0), Week2->Thu(3), Week3->Tue(1) -> each count=1, total=3.
  """
  result = _stat().compute(_make_three_weeks())
  rows = _tf(result).results
  # P = 1/3 each
  mon = _row(rows, "weekly_high", "monday")
  tue = _row(rows, "weekly_high", "tuesday")
  thu = _row(rows, "weekly_high", "thursday")
  assert mon.count == 1
  assert tue.count == 1
  assert thu.count == 1
  assert mon.total == 3
  assert mon.probability == pytest.approx(1 / 3)
  assert tue.probability == pytest.approx(1 / 3)
  assert thu.probability == pytest.approx(1 / 3)


def test_compute_rows_weekly_low_counts() -> None:
  """weekly_low: Tue=2, Wed=1 out of 3 weeks.

  Week1->Tue(1), Week2->Wed(2), Week3->Tue(1).
  """
  result = _stat().compute(_make_three_weeks())
  rows = _tf(result).results
  tue = _row(rows, "weekly_low", "tuesday")
  wed = _row(rows, "weekly_low", "wednesday")
  assert tue.count == 2
  assert wed.count == 1
  assert tue.total == 3
  assert tue.probability == pytest.approx(2 / 3)
  assert wed.probability == pytest.approx(1 / 3)


def test_compute_rows_weekly_high_close_counts() -> None:
  """weekly_high_close: Tue=1, Wed=1, Thu=1 out of 3 weeks.

  Week1->Wed(2), Week2->Thu(3), Week3->Tue(1).
  """
  result = _stat().compute(_make_three_weeks())
  rows = _tf(result).results
  tue = _row(rows, "weekly_high_close", "tuesday")
  wed = _row(rows, "weekly_high_close", "wednesday")
  thu = _row(rows, "weekly_high_close", "thursday")
  assert tue.count == 1
  assert wed.count == 1
  assert thu.count == 1
  assert tue.probability == pytest.approx(1 / 3)
  assert wed.probability == pytest.approx(1 / 3)
  assert thu.probability == pytest.approx(1 / 3)


def test_compute_rows_weekly_low_close_counts() -> None:
  """weekly_low_close: Mon=1, Fri=2 out of 3 weeks.

  Week1->Fri(4), Week2->Fri(4), Week3->Mon(0).
  """
  result = _stat().compute(_make_three_weeks())
  rows = _tf(result).results
  mon = _row(rows, "weekly_low_close", "monday")
  fri = _row(rows, "weekly_low_close", "friday")
  assert mon.count == 1
  assert fri.count == 2
  assert mon.total == 3
  assert mon.probability == pytest.approx(1 / 3)
  assert fri.probability == pytest.approx(2 / 3)


# ===========================================================================
# 3. Probabilities sum to ~1.0 per condition
# ===========================================================================

def test_probabilities_sum_to_one_per_condition() -> None:
  """Each week contributes exactly one weekday per condition -> probs sum to 1.0."""
  result = _stat().compute(_make_three_weeks())
  rows = _tf(result).results
  conditions = ["weekly_high", "weekly_low", "weekly_high_close", "weekly_low_close"]
  for cond in conditions:
    cond_rows = [r for r in rows if r.condition == cond]
    total_prob = sum(r.probability for r in cond_rows)
    assert total_prob == pytest.approx(1.0), f"{cond}: sum={total_prob}"


# ===========================================================================
# 4. by_close differs from intraday
# ===========================================================================

def test_by_close_differs_from_intraday_weekly_high() -> None:
  """Week 1: intraday high on Mon(0), but high close on Wed(2). Aggregated rows differ.

  weekly_high has Mon=1 and Thu=1 and Tue=1 (one each).
  weekly_high_close has Wed=1 and Thu=1 and Tue=1 (one each) — Wednesday appears
  in high_close but NOT in weekly_high; Monday appears in weekly_high but NOT in
  weekly_high_close.
  """
  result = _stat().compute(_make_three_weeks())
  rows = _tf(result).results
  # Monday appears in weekly_high (count=1) but should have count=0 in weekly_high_close
  mon_high = _row(rows, "weekly_high", "monday")
  assert mon_high.count == 1
  # Monday should not appear in weekly_high_close rows (or count==0 if present)
  mon_high_close = next(
    (r for r in rows if r.condition == "weekly_high_close" and r.outcome == "monday"),
    None,
  )
  # Either Monday is absent (count=0) or has an explicit count=0 row.
  assert mon_high_close is None or mon_high_close.count == 0

  # Wednesday appears in weekly_high_close (count=1) but NOT in weekly_high.
  wed_high_close = _row(rows, "weekly_high_close", "wednesday")
  assert wed_high_close.count == 1
  wed_high = next(
    (r for r in rows if r.condition == "weekly_high" and r.outcome == "wednesday"),
    None,
  )
  assert wed_high is None or wed_high.count == 0


def test_by_close_differs_from_intraday_low() -> None:
  """weekly_low has Tue=2,Wed=1; weekly_low_close has Mon=1,Fri=2. Entirely different sets."""
  result = _stat().compute(_make_three_weeks())
  rows = _tf(result).results
  # Tue/Wed should NOT appear in low_close
  for outcome in ("tuesday", "wednesday"):
    r = next(
      (r for r in rows if r.condition == "weekly_low_close" and r.outcome == outcome),
      None,
    )
    assert r is None or r.count == 0, f"weekly_low_close[{outcome}] should have count=0"
  # Mon/Fri should NOT appear in weekly_low
  for outcome in ("monday", "friday"):
    r = next(
      (r for r in rows if r.condition == "weekly_low" and r.outcome == outcome),
      None,
    )
    assert r is None or r.count == 0, f"weekly_low[{outcome}] should have count=0"


# ===========================================================================
# 5. Pending exclusion — final ISO week always dropped
# ===========================================================================

def test_pending_final_week_dropped() -> None:
  """Week 4 (ISO 2024w04) is the final ISO week and must be excluded.

  3 weeks (w01, w02, w03) resolve; w04 is dropped -> total_samples == 3.
  """
  result = _stat().compute(_make_three_weeks())
  assert _tf(result).total_samples == 3


def test_single_week_data_yields_empty_after_pending_drop() -> None:
  """A single ISO week of data -> after pending-week drop, nothing remains."""
  frames = [_make_day(date, o, c, h, day_lo) for date, o, c, h, day_lo in _W1]
  single_week_df = _concat_days(frames)
  result = _stat().compute(single_week_df)
  assert _tf(result).total_samples == 0


def test_pending_drop_after_two_weeks_leaves_one() -> None:
  """Two ISO weeks: the second (most recent) is dropped; only the first counts."""
  frames = []
  for specs in (_W1, _W2):
    for date, o, c, h, day_lo in specs:
      frames.append(_make_day(date, o, c, h, day_lo))
  df = _concat_days(frames)
  result = _stat().compute(df)
  # After dropping week 2 (pending), only week 1 remains.
  assert _tf(result).total_samples == 1


def test_truncated_day_excluded_from_week_extreme() -> None:
  """A truncated (unresolved) day in week 4 is excluded; week 4 still dropped as pending.

  The truncated day should not contaminate the extreme computation because
  build_resolved_days excludes it (last bar at 09:50 < threshold).
  """
  frames = []
  for specs in (_W1, _W2, _W3, _W4_PENDING):
    for date, o, c, h, day_lo in specs:
      frames.append(_make_day(date, o, c, h, day_lo))
  # Add a truncated day in week 4 with extreme values that would distort if included.
  truncated = _make_truncated_day("2024-01-23")  # Tue ISO 2024w04
  frames.append(truncated)
  df = _concat_days(frames)
  result = _stat().compute(df)
  # Total samples still 3 (weeks 1-3 resolved, week 4 still pending/dropped).
  assert _tf(result).total_samples == 3
  # The truncated day's extreme values should not affect weeks 1-3.
  rows = _tf(result).results
  assert sum(r.count for r in rows if r.condition == "weekly_high") == 3


# ===========================================================================
# 6. WeeklyCandle slicer
# ===========================================================================

def test_weekly_candle_slicer_groups_present() -> None:
  """WeeklyCandle slicer produces 'green' and 'red' groups (2 green, 1 red)."""
  result = _stat().compute(_make_three_weeks())
  groups = _tf(result).slices["weekly_candle"].groups
  assert set(groups.keys()) == {"green", "red"}


def test_weekly_candle_green_sample_count() -> None:
  """Green group: weeks 1 and 3 -> 2 samples."""
  result = _stat().compute(_make_three_weeks())
  green_grp = _tf(result).slices["weekly_candle"].groups["green"]
  # 2 green weeks (w01 close=125>=100 open, w03 close=122>=100 open)
  assert green_grp.total_samples == 2


def test_weekly_candle_red_sample_count() -> None:
  """Red group: week 2 only -> 1 sample."""
  result = _stat().compute(_make_three_weeks())
  red_grp = _tf(result).slices["weekly_candle"].groups["red"]
  # 1 red week (w02 close=95 < 100 open)
  assert red_grp.total_samples == 1


def test_weekly_candle_green_high_weekday_probs() -> None:
  """Green weeks: weekly_high on Mon(w1) and Tue(w3) -> Mon=1/2, Tue=1/2."""
  result = _stat().compute(_make_three_weeks())
  green_rows = _tf(result).slices["weekly_candle"].groups["green"].results
  # P(weekly_high on Monday | green week) = 1/2
  mon = _row(green_rows, "weekly_high", "monday")
  tue = _row(green_rows, "weekly_high", "tuesday")
  assert mon.count == 1
  assert tue.count == 1
  assert mon.probability == pytest.approx(0.5)
  assert tue.probability == pytest.approx(0.5)


def test_weekly_candle_green_low_weekday_probs() -> None:
  """Green weeks: weekly_low on Tue(w1) and Tue(w3) -> Tue=2/2=1.0."""
  result = _stat().compute(_make_three_weeks())
  green_rows = _tf(result).slices["weekly_candle"].groups["green"].results
  # Both green weeks have their weekly low on Tuesday.
  tue = _row(green_rows, "weekly_low", "tuesday")
  assert tue.count == 2
  assert tue.probability == pytest.approx(1.0)


def test_weekly_candle_red_high_weekday_probs() -> None:
  """Red weeks (1 week): weekly_high on Thu(w2) -> Thu=1/1=1.0."""
  result = _stat().compute(_make_three_weeks())
  red_rows = _tf(result).slices["weekly_candle"].groups["red"].results
  thu = _row(red_rows, "weekly_high", "thursday")
  assert thu.count == 1
  assert thu.probability == pytest.approx(1.0)


def test_weekly_candle_red_low_weekday_probs() -> None:
  """Red weeks (1 week): weekly_low on Wed(w2) -> Wed=1/1=1.0."""
  result = _stat().compute(_make_three_weeks())
  red_rows = _tf(result).slices["weekly_candle"].groups["red"].results
  wed = _row(red_rows, "weekly_low", "wednesday")
  assert wed.count == 1
  assert wed.probability == pytest.approx(1.0)


def test_weekly_candle_group_probs_sum_to_one() -> None:
  """Per-group probabilities for each condition sum to ~1.0."""
  result = _stat().compute(_make_three_weeks())
  slc = _tf(result).slices["weekly_candle"]
  conditions = ["weekly_high", "weekly_low", "weekly_high_close", "weekly_low_close"]
  for grp_key, grp in slc.groups.items():
    for cond in conditions:
      cond_rows = [r for r in grp.results if r.condition == cond]
      total_prob = sum(r.probability for r in cond_rows)
      assert total_prob == pytest.approx(1.0), (
        f"group={grp_key} condition={cond}: sum={total_prob}"
      )


# ===========================================================================
# 7. Baseline determinism / reproducibility
# ===========================================================================

def test_baseline_rows_deterministic_same_seed() -> None:
  """Same seed -> identical baseline_rows output."""
  stat = _stat()
  df = _make_three_weeks()
  day_table = stat.build_day_table(df)
  rows_a = stat.baseline_rows(day_table, seed=42)
  rows_b = stat.baseline_rows(day_table, seed=42)
  assert len(rows_a) == len(rows_b)
  for a, b in zip(rows_a, rows_b):
    assert a.condition == b.condition
    assert a.outcome == b.outcome
    assert a.count == b.count
    assert a.probability == pytest.approx(b.probability)


def test_baseline_rows_different_seeds_may_differ() -> None:
  """Different seeds typically produce different baseline distributions."""
  stat = _stat()
  df = _make_three_weeks()
  day_table = stat.build_day_table(df)
  rows_42 = stat.baseline_rows(day_table, seed=42)
  rows_99 = stat.baseline_rows(day_table, seed=99)
  # With only 3 weeks the counts are small; assert at least the totals are consistent.
  assert len(rows_42) == len(rows_99)
  totals_42 = {(r.condition, r.outcome): r.total for r in rows_42}
  totals_99 = {(r.condition, r.outcome): r.total for r in rows_99}
  assert totals_42 == totals_99


def test_baseline_probs_populated_in_compute() -> None:
  """compute() embeds baseline: every result row has baseline_n > 0 and baseline_prob set."""
  result = _stat().compute(_make_three_weeks())
  for row in _tf(result).results:
    assert row.baseline_n > 0, f"({row.condition},{row.outcome}) has baseline_n=0"
    # baseline_prob is a float (may be 0.0 if the random draw never picked that weekday)
    assert isinstance(row.baseline_prob, float)


def test_compute_twice_identical_results() -> None:
  """Running compute() twice on the same data yields identical results (deterministic)."""
  stat = _stat()
  df = _make_three_weeks()
  result_a = stat.compute(df)
  result_b = stat.compute(df)
  rows_a = _tf(result_a).results
  rows_b = _tf(result_b).results
  assert len(rows_a) == len(rows_b)
  for a, b in zip(rows_a, rows_b):
    assert a.condition == b.condition
    assert a.outcome == b.outcome
    assert a.probability == pytest.approx(b.probability)
    assert a.baseline_prob == pytest.approx(b.baseline_prob)


def test_baseline_probs_sum_near_one_per_condition() -> None:
  """Baseline probabilities for each condition also sum to 1.0 (random reassignment).

  Each week contributes exactly one weekday to the baseline draw per condition.
  """
  stat = _stat()
  day_table = stat.build_day_table(_make_three_weeks())
  bl_rows = stat.baseline_rows(day_table, seed=42)
  conditions = ["weekly_high", "weekly_low", "weekly_high_close", "weekly_low_close"]
  for cond in conditions:
    cond_rows = [r for r in bl_rows if r.condition == cond]
    total_prob = sum(r.probability for r in cond_rows)
    assert total_prob == pytest.approx(1.0), f"baseline {cond}: sum={total_prob}"


# ===========================================================================
# 8. Edge cases: empty DataFrame, single ISO week
# ===========================================================================

def _empty_df() -> pd.DataFrame:
  df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
  df["timestamp"] = pd.array([], dtype="datetime64[ns, America/New_York]")
  return df


def test_empty_dataframe_build_day_table() -> None:
  """Empty input -> build_day_table returns an empty DataFrame."""
  day_table = _stat().build_day_table(_empty_df())
  assert len(day_table) == 0


def test_empty_dataframe_compute_no_crash() -> None:
  """Empty input -> compute() produces a valid StatRunResult with 0 samples."""
  result = _stat().compute(_empty_df())
  tf = _tf(result)
  assert tf.total_samples == 0
  assert tf.data_range == []
  assert tf.results == []


def test_empty_dataframe_no_slice_groups() -> None:
  """Empty input -> no WeeklyCandle slice groups."""
  result = _stat().compute(_empty_df())
  assert _tf(result).slices["weekly_candle"].groups == {}


def test_single_week_build_day_table_empty() -> None:
  """Single ISO week: after pending drop, build_day_table returns empty."""
  frames = [_make_day(date, o, c, h, day_lo) for date, o, c, h, day_lo in _W1]
  df = _concat_days(frames)
  day_table = _stat().build_day_table(df)
  assert len(day_table) == 0


def test_single_week_compute_zero_samples() -> None:
  """Single ISO week of full RTH data: after pending drop, total_samples == 0."""
  frames = [_make_day(date, o, c, h, day_lo) for date, o, c, h, day_lo in _W1]
  df = _concat_days(frames)
  result = _stat().compute(df)
  assert _tf(result).total_samples == 0
  assert _tf(result).data_range == []


def test_two_weeks_first_is_resolved() -> None:
  """Two ISO weeks: week 2 dropped (pending), week 1 resolved -> total_samples==1."""
  frames = []
  for specs in (_W1, _W2):
    for date, o, c, h, day_lo in specs:
      frames.append(_make_day(date, o, c, h, day_lo))
  df = _concat_days(frames)
  result = _stat().compute(df)
  assert _tf(result).total_samples == 1
  # The one resolved week is week 1 -> weekly_high on Monday
  rows = _tf(result).results
  mon = _row(rows, "weekly_high", "monday")
  assert mon.count == 1
  assert mon.probability == pytest.approx(1.0)


def test_data_range_reflects_resolved_weeks() -> None:
  """data_range shows the first and last date of the resolved day table index."""
  result = _stat().compute(_make_three_weeks())
  # Resolved weeks: 2024-01-01 (first day of w01) to 2024-01-15 (first day of w03).
  tf = _tf(result)
  assert tf.data_range[0] == "2024-01-01"
  assert tf.data_range[1] == "2024-01-15"


# ===========================================================================
# 9. write_results round-trip
# ===========================================================================

def test_write_results_round_trip(tmp_path: Path) -> None:
  """write_results serialises the result; model_validate round-trips correctly."""
  result = _stat().compute(_make_three_weeks())
  written = write_results(result, results_dir=tmp_path)

  assert written.name == "high_low_weekday.json"
  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)

  tf = validated.instruments["NQ"]["weekly"]
  assert tf.total_samples == 3

  # Spot-check: weekly_low tuesday count == 2 after round-trip
  rows = tf.results
  tue = _row(rows, "weekly_low", "tuesday")
  assert tue.count == 2
  assert tue.probability == pytest.approx(2 / 3)


def test_write_results_stat_name(tmp_path: Path) -> None:
  """Serialised file name matches stat_name."""
  result = _stat().compute(_make_three_weeks())
  written = write_results(result, results_dir=tmp_path)
  assert written.stem == "high_low_weekday"
  raw = json.loads(written.read_text(encoding="utf-8"))
  assert raw["stat_name"] == "high_low_weekday"


# ===========================================================================
# 10. i18n
# ===========================================================================

def test_i18n_title_and_definition() -> None:
  """Title and definition carry non-empty en and fr strings."""
  result = _stat().compute(_empty_df())
  assert result.title.en != "" and result.title.fr != ""
  assert result.definition.en != "" and result.definition.fr != ""


def test_i18n_labels_have_en_and_fr() -> None:
  """All condition and outcome labels carry non-empty en and fr strings."""
  result = _stat().compute(_empty_df())
  for mapping in (result.labels.conditions, result.labels.outcomes):
    for key, i18n in mapping.items():
      assert i18n.en != "", f"{key}.en is empty"
      assert i18n.fr != "", f"{key}.fr is empty"


def test_i18n_labels_conditions_correct_keys() -> None:
  """The four expected condition keys are present in labels.conditions."""
  result = _stat().compute(_empty_df())
  expected = {"weekly_high", "weekly_low", "weekly_high_close", "weekly_low_close"}
  assert set(result.labels.conditions.keys()) == expected


def test_i18n_labels_weekly_candle_dimension() -> None:
  """WeeklyCandle slicer dimension label is surfaced in labels.dimensions."""
  result = _stat().compute(_make_three_weeks())
  assert "weekly_candle" in result.labels.dimensions
  dim = result.labels.dimensions["weekly_candle"]
  assert dim.en != ""
  assert dim.fr != ""
