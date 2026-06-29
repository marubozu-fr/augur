"""Tests for stats.weekly_open_retracement.standard.

All data is synthetic — no real market files required.
Expected values are hand-calculated before each assertion.

Framing:
  - Weekly opening price L = open of the first RTH bar (09:30 bar of Monday).
  - Direction from the first bar's close vs L:
      close > L → opened_above (direction_up=True)
      close < L → opened_below (direction_up=False)
      close == L → doji (direction_up=NA, excluded from denominators, in total_samples)
  - Retracement: any post-first-bar RTH bar within the week has low <= L
    (opened_above) or high >= L (opened_below). Exact touch counts.
  - retrace_weekday: python weekday (0=Mon..4=Fri) of the chronologically first
    retracing bar.
  - spike_pts: max favorable excursion from L over [first_bar .. first_retrace_bar]
    inclusive, or [first_bar .. last_bar] when no retracement.
  - spike_pct: 100 * spike_pts / L.
  - Tier 1: 2x2 matrix opened_above/opened_below × retraced/not_retraced.
  - Tier 2: condition retraced × outcomes monday..friday.
  - Last ISO week always dropped (pending discipline).
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.weekly_open_retracement.standard import WeeklyOpenRetracement

# ---------------------------------------------------------------------------
# Minimal InstrumentConfig — does not touch NQ.yaml
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

_RTH_START = 570  # minute-of-day for 09:30
_RTH_NEXT = 571   # minute-of-day for 09:31


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

def _bar(
  date: str,
  mod: int,
  open_p: float,
  close_p: float,
  high: float,
  low: float,
) -> dict:
  """Single 1-min OHLCV bar at minute-of-day ``mod`` on ``date`` (NY tz)."""
  h, m = divmod(mod, 60)
  ts = pd.Timestamp(date, tz=_NY).replace(hour=h, minute=m, second=0, microsecond=0)
  return {
    "timestamp": ts,
    "open": open_p,
    "high": high,
    "low": low,
    "close": close_p,
    "volume": 1000,
  }


def _make_rth_day(
  date: str,
  open_price: float,
  first_close: float,
  day_high: float,
  day_low: float,
  first_high: float | None = None,
  first_low: float | None = None,
) -> pd.DataFrame:
  """Two RTH bars for one trading day.

  Bar 1 (09:30): open=open_price, close=first_close.
    high defaults to max(open_price, first_close); override with first_high.
    low  defaults to min(open_price, first_close); override with first_low.
  Bar 2 (09:31): open=first_close, close=first_close, high=day_high, low=day_low.

  The 09:30 bar controls L (= open_price for Monday) and direction (first_close vs
  open_price). The 09:31 bar controls subsequent extremes for retracement / spike.
  Passing first_high / first_low overrides the 09:30 bar's extremes explicitly —
  used when the 09:30 bar itself must trigger retracement or carry a specific spike.
  """
  fh = first_high if first_high is not None else max(open_price, first_close)
  fl = first_low if first_low is not None else min(open_price, first_close)
  return pd.DataFrame([
    _bar(date, _RTH_START, open_price, first_close, fh, fl),
    _bar(date, _RTH_NEXT,  first_close, first_close, day_high, day_low),
  ])


def make_candles(days: list[dict]) -> pd.DataFrame:
  """Concatenate per-day specs into a sorted 1-min OHLCV DataFrame.

  Each dict requires: "date", "open" (first-bar open = L for Mon),
  "close" (first-bar close, determines direction when on Monday),
  "high" (09:31-bar high), "low" (09:31-bar low).
  Optional: "first_high", "first_low" override the 09:30-bar extremes.
  """
  frames = [
    _make_rth_day(
      d["date"], d["open"], d["close"], d["high"], d["low"],
      first_high=d.get("first_high"),
      first_low=d.get("first_low"),
    )
    for d in days
  ]
  df = pd.concat(frames, ignore_index=True)
  return df.sort_values("timestamp").reset_index(drop=True)


def _empty_df() -> pd.DataFrame:
  df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
  df["timestamp"] = pd.array([], dtype="datetime64[ns, America/New_York]")
  return df


# ---------------------------------------------------------------------------
# Stat / result helpers
# ---------------------------------------------------------------------------

def _stat() -> WeeklyOpenRetracement:
  return WeeklyOpenRetracement(instrument="NQ", config=_TEST_CONFIG)


def _row(result: StatRunResult, condition: str, outcome: str) -> StatResultRow:
  for r in result.instruments["NQ"]["weekly"].results:
    if r.condition == condition and r.outcome == outcome:
      return r
  raise KeyError((condition, outcome))


def _tf(result: StatRunResult):
  return result.instruments["NQ"]["weekly"]


# ---------------------------------------------------------------------------
# Main fixture — 5 resolved weeks + 1 pending
#
# Week 1 (2024-01-01, Mon): opened_above, NOT retraced
#   L=100, first_close=105 → direction_up=True
#   Mon 09:31 high=108, low=103 → 103 > 100, no retrace
#   spike window = all bars (not retraced): max(105, 108) - 100 = 8 pts; pct=8.0%
#
# Week 2 (2024-01-08, Mon): opened_above, retraced on Monday (retrace_weekday=0)
#   L=100, first_close=105 → direction_up=True
#   Mon 09:31 high=112, low=100 → 100 <= 100 (exact touch), retraced, weekday=0
#   spike window = [Mon 09:30, Mon 09:31]: max(105, 112) - 100 = 12 pts; pct=12.0%
#
# Week 3 (2024-01-15, Mon): opened_below, NOT retraced
#   L=100, first_close=95 → direction_up=False
#   Mon 09:31 high=98, low=93 → 98 < 100, no retrace
#   spike window = all bars: 100 - min(95, 93) = 7 pts; pct=7.0%
#
# Week 4 (2024-01-22, Mon + 2024-01-23, Tue): opened_below, retraced on Tuesday
#   L=100, first_close=95 → direction_up=False
#   Mon 09:30: high=100, low=95; Mon 09:31: high=98, low=93 → 98 < 100, no retrace
#   Tue 09:30: first_high=101 >= 100 → retraced, retrace_weekday=1; first_low=92
#   Tue 09:31: high=98, low=90 — OUTSIDE spike window
#   spike window = [Mon09:30, Mon09:31, Tue09:30]: 100 - min(95, 93, 92) = 8 pts; pct=8.0%
#
# Week 5 (2024-01-29, Mon): doji (first_close == L == 100)
#   direction_up=pd.NA, spike_pts=pd.NA, spike_pct=pd.NA — not countable
#
# Week 6 (2024-02-05, Mon): PENDING — dropped by pending discipline
#
# Expected Tier-1 results:
#   opened_above total=2 (W1,W2): retraced=1(W2), not_retraced=1(W1) → P=0.5 each
#   opened_below total=2 (W3,W4): retraced=1(W4), not_retraced=1(W3) → P=0.5 each
#
# Expected Tier-2 (retraced_total=2: W2+W4):
#   monday: count=1(W2), total=2, P=0.5
#   tuesday: count=1(W4), total=2, P=0.5
#   wednesday/thursday/friday: count=0, total=2, P=0.0
#
# total_samples = 5 (W1-W5 in week_table; W6 dropped)
# ---------------------------------------------------------------------------
_MAIN_SEQ: list[dict] = [
  # W1 — opened_above, not retraced
  {"date": "2024-01-01", "open": 100.0, "close": 105.0, "high": 108.0, "low": 103.0},
  # W2 — opened_above, retraced Mon (exact touch)
  {"date": "2024-01-08", "open": 100.0, "close": 105.0, "high": 112.0, "low": 100.0},
  # W3 — opened_below, not retraced
  {"date": "2024-01-15", "open": 100.0, "close": 95.0, "high": 98.0, "low": 93.0},
  # W4 Mon — opened_below, no retrace on Monday
  {"date": "2024-01-22", "open": 100.0, "close": 95.0, "high": 98.0, "low": 93.0},
  # W4 Tue — retrace on Tuesday (first_high=101 >= L=100)
  {
    "date": "2024-01-23", "open": 95.0, "close": 97.0, "high": 98.0, "low": 90.0,
    "first_high": 101.0, "first_low": 92.0,
  },
  # W5 — doji (first_close == open == L=100)
  {"date": "2024-01-29", "open": 100.0, "close": 100.0, "high": 105.0, "low": 95.0},
  # W6 — pending (dropped)
  {"date": "2024-02-05", "open": 100.0, "close": 105.0, "high": 108.0, "low": 103.0},
]


def _make_alternating_weeks(n_resolved: int) -> pd.DataFrame:
  """Generate n_resolved alternating above/below weeks with mixed retracement.

  Pattern (cycling): above-ret, above-not, below-ret(Tue), below-not.
  Appends one pending stub week at the end. Base Monday = 2020-01-06.

  With n_resolved=8: 2 above-ret, 2 above-not, 2 below-ret, 2 below-not.
  Retraced total = 4 of 8 countable.
  """
  days: list[dict] = []
  base = pd.Timestamp("2020-01-06", tz=_NY)  # known Monday
  for i in range(n_resolved + 1):
    monday = base + pd.Timedelta(weeks=i)
    mon_str = monday.strftime("%Y-%m-%d")
    if i == n_resolved:
      # pending stub
      days.append({"date": mon_str, "open": 100.0, "close": 105.0, "high": 108.0, "low": 103.0})
      continue
    pattern = i % 4
    if pattern == 0:  # above-ret (Mon 09:31 low==100 == L, exact touch)
      days.append({"date": mon_str, "open": 100.0, "close": 105.0, "high": 112.0, "low": 100.0})
    elif pattern == 1:  # above-not (Mon 09:31 low=103 > 100)
      days.append({"date": mon_str, "open": 100.0, "close": 105.0, "high": 108.0, "low": 103.0})
    elif pattern == 2:  # below-ret (Mon no retrace; Tue 09:30 retraces)
      days.append({"date": mon_str, "open": 100.0, "close": 95.0, "high": 98.0, "low": 93.0})
      tue_str = (monday + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
      days.append({
        "date": tue_str, "open": 95.0, "close": 97.0, "high": 90.0, "low": 88.0,
        "first_high": 101.0, "first_low": 93.0,
      })
    else:  # pattern == 3: below-not (Mon 09:31 high=98 < 100)
      days.append({"date": mon_str, "open": 100.0, "close": 95.0, "high": 98.0, "low": 93.0})
  return make_candles(days)


# ===========================================================================
# 1. build_week_table structure
# ===========================================================================

def test_build_week_table_correct_row_count() -> None:
  """5 resolved weeks in _MAIN_SEQ (W1-W5); W6 pending → 5 rows in week_table."""
  # W1 Jan1, W2 Jan8, W3 Jan15, W4 Jan22, W5 Jan29 → 5 resolved; W6 Feb5 dropped
  stat = _stat()
  week_table = stat.build_week_table(make_candles(_MAIN_SEQ))
  assert len(week_table) == 5


def test_build_week_table_index_is_monday() -> None:
  """All week_table index entries are Mondays (dayofweek == 0)."""
  stat = _stat()
  week_table = stat.build_week_table(make_candles(_MAIN_SEQ))
  assert all(idx.dayofweek == 0 for idx in week_table.index)


def test_build_week_table_pending_week_dropped() -> None:
  """The last ISO week (W6 = 2024-02-05) is absent from the week_table index."""
  stat = _stat()
  week_table = stat.build_week_table(make_candles(_MAIN_SEQ))
  w6 = pd.Timestamp("2024-02-05", tz=_NY).normalize()
  assert w6 not in week_table.index


def test_build_week_table_weekly_open_is_first_rth_bar_open() -> None:
  """weekly_open for W1 equals the open of the Monday 09:30 bar (100.0)."""
  # W1 Mon 09:30 bar: open_price=100.0 → weekly_open=100.0
  stat = _stat()
  week_table = stat.build_week_table(make_candles(_MAIN_SEQ))
  w1 = pd.Timestamp("2024-01-01", tz=_NY).normalize()
  assert float(week_table.loc[w1, "weekly_open"]) == pytest.approx(100.0)


def test_build_week_table_direction_up_nullable_boolean_dtype() -> None:
  """direction_up column must use pandas BooleanDtype (nullable bool)."""
  stat = _stat()
  week_table = stat.build_week_table(make_candles(_MAIN_SEQ))
  assert week_table["direction_up"].dtype == pd.BooleanDtype()


def test_build_week_table_retrace_weekday_int64_dtype() -> None:
  """retrace_weekday column must use pandas Int64Dtype (nullable int)."""
  stat = _stat()
  week_table = stat.build_week_table(make_candles(_MAIN_SEQ))
  assert week_table["retrace_weekday"].dtype == pd.Int64Dtype()


def test_build_week_table_columns_match_declared_schema() -> None:
  """week_table has exactly the six declared columns in order."""
  expected = ["weekly_open", "direction_up", "retraced", "retrace_weekday", "spike_pts", "spike_pct"]
  stat = _stat()
  week_table = stat.build_week_table(make_candles(_MAIN_SEQ))
  assert list(week_table.columns) == expected


def test_build_week_table_empty_input_returns_empty_with_columns() -> None:
  """Empty candles DataFrame → empty week_table with the six declared columns."""
  stat = _stat()
  week_table = stat.build_week_table(_empty_df())
  assert week_table.empty
  assert set(week_table.columns) == {
    "weekly_open", "direction_up", "retraced", "retrace_weekday", "spike_pts", "spike_pct"
  }


def test_build_week_table_single_iso_week_returns_empty() -> None:
  """Only one ISO week in the data → it is the pending week → empty week_table."""
  # One ISO week = last week → dropped; need 2+ to have 1 resolved.
  candles = make_candles([
    {"date": "2024-01-01", "open": 100.0, "close": 105.0, "high": 108.0, "low": 103.0},
  ])
  stat = _stat()
  week_table = stat.build_week_table(candles)
  assert week_table.empty


def test_build_week_table_two_weeks_yields_one_resolved() -> None:
  """Two ISO weeks → one resolved, one pending → week_table has 1 row."""
  candles = make_candles([
    {"date": "2024-01-01", "open": 100.0, "close": 105.0, "high": 108.0, "low": 103.0},
    {"date": "2024-01-08", "open": 100.0, "close": 103.0, "high": 106.0, "low": 102.0},
  ])
  stat = _stat()
  week_table = stat.build_week_table(candles)
  assert len(week_table) == 1


def test_build_day_table_delegates_to_build_week_table() -> None:
  """build_day_table and build_week_table return identical DataFrames."""
  candles = make_candles(_MAIN_SEQ)
  stat = _stat()
  pd.testing.assert_frame_equal(stat.build_day_table(candles), stat.build_week_table(candles))


# ===========================================================================
# 2. Direction classification
# ===========================================================================

def test_direction_opened_above_is_true() -> None:
  """first_close > weekly_open → direction_up=True (W1: close=105 > open=100)."""
  stat = _stat()
  week_table = stat.build_week_table(make_candles(_MAIN_SEQ))
  w1 = pd.Timestamp("2024-01-01", tz=_NY).normalize()
  assert week_table.loc[w1, "direction_up"] is True or bool(week_table.loc[w1, "direction_up"]) is True


def test_direction_opened_below_is_false() -> None:
  """first_close < weekly_open → direction_up=False (W3: close=95 < open=100)."""
  stat = _stat()
  week_table = stat.build_week_table(make_candles(_MAIN_SEQ))
  w3 = pd.Timestamp("2024-01-15", tz=_NY).normalize()
  assert bool(week_table.loc[w3, "direction_up"]) is False


def test_direction_doji_is_na() -> None:
  """first_close == weekly_open → direction_up=pd.NA (W5: close=100 == open=100)."""
  stat = _stat()
  week_table = stat.build_week_table(make_candles(_MAIN_SEQ))
  w5 = pd.Timestamp("2024-01-29", tz=_NY).normalize()
  assert pd.isna(week_table.loc[w5, "direction_up"])


def test_doji_excluded_from_condition_totals() -> None:
  """Doji week (W5) excluded from every denominator; above+below total = 4 (not 5)."""
  # 5 resolved: 2 above (W1,W2), 2 below (W3,W4), 1 doji (W5 excluded)
  result = _stat().compute(make_candles(_MAIN_SEQ))
  above_total = _row(result, "opened_above", "retraced").total
  below_total = _row(result, "opened_below", "retraced").total
  assert above_total + below_total == 4


def test_doji_counted_in_total_samples() -> None:
  """Doji week (W5) appears in total_samples even though excluded from denominators."""
  # 5 resolved weeks including doji → total_samples=5
  result = _stat().compute(make_candles(_MAIN_SEQ))
  assert _tf(result).total_samples == 5


# ===========================================================================
# 3. Retracement detection
# ===========================================================================

def test_retracement_above_exact_touch_counts() -> None:
  """opened_above: low == L (exact touch) → retraced=True (W2: Mon 09:31 low=100=L)."""
  # W2 Mon 09:31 bar: low=100 == L=100 → 100 <= 100 → retraced
  stat = _stat()
  week_table = stat.build_week_table(make_candles(_MAIN_SEQ))
  w2 = pd.Timestamp("2024-01-08", tz=_NY).normalize()
  assert bool(week_table.loc[w2, "retraced"]) is True


def test_retracement_above_miss_is_not_retraced() -> None:
  """opened_above: all post-first-bar lows > L → retraced=False (W1: low=103 > 100)."""
  # W1 Mon 09:31 bar: low=103 > 100 → not retraced
  stat = _stat()
  week_table = stat.build_week_table(make_candles(_MAIN_SEQ))
  w1 = pd.Timestamp("2024-01-01", tz=_NY).normalize()
  assert bool(week_table.loc[w1, "retraced"]) is False


def test_retracement_below_exact_touch_counts() -> None:
  """opened_below: high == L (exact touch) → retraced=True (W4: Tue first_high=101 >= 100)."""
  # W4 Tue 09:30 bar: high=101 >= L=100 → retraced
  stat = _stat()
  week_table = stat.build_week_table(make_candles(_MAIN_SEQ))
  w4 = pd.Timestamp("2024-01-22", tz=_NY).normalize()
  assert bool(week_table.loc[w4, "retraced"]) is True


def test_retracement_below_miss_is_not_retraced() -> None:
  """opened_below: all post-first-bar highs < L → retraced=False (W3: high=98 < 100)."""
  # W3 Mon 09:31 bar: high=98 < 100 → not retraced
  stat = _stat()
  week_table = stat.build_week_table(make_candles(_MAIN_SEQ))
  w3 = pd.Timestamp("2024-01-15", tz=_NY).normalize()
  assert bool(week_table.loc[w3, "retraced"]) is False


def test_retracement_above_one_tick_above_l_is_not_retraced() -> None:
  """opened_above: low strictly above L by 1 tick → not retraced.

  Single-week fixture: L=100, first_close=105, Mon 09:31 low=100.25 > 100 → not retraced.
  """
  # Hand-calc: 100.25 > 100 → not retraced; P(retraced|above)=0/1=0.0
  candles = make_candles([
    {"date": "2024-03-04", "open": 100.0, "close": 105.0, "high": 108.0, "low": 100.25},
    {"date": "2024-03-11", "open": 100.0, "close": 103.0, "high": 106.0, "low": 102.0},
  ])
  result = _stat().compute(candles)
  r = _row(result, "opened_above", "retraced")
  assert r.count == 0
  assert r.total == 1
  assert r.probability == pytest.approx(0.0)


# ===========================================================================
# 4. retrace_weekday
# ===========================================================================

def test_retrace_weekday_monday_value() -> None:
  """W2 retraces on Monday → retrace_weekday=0."""
  # W2 Mon 09:31 bar is the first post-bar; dayofweek(2024-01-08)=0
  stat = _stat()
  week_table = stat.build_week_table(make_candles(_MAIN_SEQ))
  w2 = pd.Timestamp("2024-01-08", tz=_NY).normalize()
  assert int(week_table.loc[w2, "retrace_weekday"]) == 0


def test_retrace_weekday_tuesday_value() -> None:
  """W4 retraces on Tuesday → retrace_weekday=1."""
  # W4 Tue 09:30 bar is first retrace bar; dayofweek(2024-01-23)=1
  stat = _stat()
  week_table = stat.build_week_table(make_candles(_MAIN_SEQ))
  w4 = pd.Timestamp("2024-01-22", tz=_NY).normalize()
  assert int(week_table.loc[w4, "retrace_weekday"]) == 1


def test_retrace_weekday_na_when_not_retraced() -> None:
  """W1 (not retraced) → retrace_weekday=pd.NA."""
  stat = _stat()
  week_table = stat.build_week_table(make_candles(_MAIN_SEQ))
  w1 = pd.Timestamp("2024-01-01", tz=_NY).normalize()
  assert pd.isna(week_table.loc[w1, "retrace_weekday"])


def test_tier2_weekday_monday_count_and_probability() -> None:
  """Tier-2 monday row: count=1 (W2), total=2 (W2+W4 both retraced), P=0.5."""
  # Retraced: W2 (Mon) and W4 (Tue). monday count=1.
  result = _stat().compute(make_candles(_MAIN_SEQ))
  r = _row(result, "retraced", "monday")
  assert r.count == 1
  assert r.total == 2
  assert r.probability == pytest.approx(0.5)


def test_tier2_weekday_tuesday_count_and_probability() -> None:
  """Tier-2 tuesday row: count=1 (W4), total=2, P=0.5."""
  result = _stat().compute(make_candles(_MAIN_SEQ))
  r = _row(result, "retraced", "tuesday")
  assert r.count == 1
  assert r.total == 2
  assert r.probability == pytest.approx(0.5)


def test_tier2_weekday_counts_sum_to_retraced_total() -> None:
  """Sum of monday..friday counts == total retraced countable weeks."""
  # Retraced total = 2; monday=1, tuesday=1, wed/thu/fri=0 → sum=2
  result = _stat().compute(make_candles(_MAIN_SEQ))
  days = ("monday", "tuesday", "wednesday", "thursday", "friday")
  total_count = sum(_row(result, "retraced", d).count for d in days)
  retraced_total = _row(result, "retraced", "monday").total
  assert total_count == retraced_total == 2


# ===========================================================================
# 5. spike_pts / spike_pct
# ===========================================================================

def test_spike_pts_above_not_retraced() -> None:
  """W1 (above, not retraced): spike = max high over all week bars - L.

  Mon 09:30 high=105, Mon 09:31 high=108. spike_pts = 108 - 100 = 8.
  """
  stat = _stat()
  week_table = stat.build_week_table(make_candles(_MAIN_SEQ))
  w1 = pd.Timestamp("2024-01-01", tz=_NY).normalize()
  assert float(week_table.loc[w1, "spike_pts"]) == pytest.approx(8.0)


def test_spike_pct_above_not_retraced() -> None:
  """W1 spike_pct = 100 * 8 / 100 = 8.0%."""
  stat = _stat()
  week_table = stat.build_week_table(make_candles(_MAIN_SEQ))
  w1 = pd.Timestamp("2024-01-01", tz=_NY).normalize()
  assert float(week_table.loc[w1, "spike_pct"]) == pytest.approx(8.0)


def test_spike_pts_above_retraced() -> None:
  """W2 (above, retraced Mon): spike = max high in [Mon09:30..Mon09:31] - L.

  Mon 09:30 high=105, Mon 09:31 high=112. spike_pts = 112 - 100 = 12.
  """
  stat = _stat()
  week_table = stat.build_week_table(make_candles(_MAIN_SEQ))
  w2 = pd.Timestamp("2024-01-08", tz=_NY).normalize()
  assert float(week_table.loc[w2, "spike_pts"]) == pytest.approx(12.0)


def test_spike_pts_below_not_retraced() -> None:
  """W3 (below, not retraced): spike = L - min low over all week bars.

  Mon 09:30 low=95, Mon 09:31 low=93. spike_pts = 100 - 93 = 7.
  """
  stat = _stat()
  week_table = stat.build_week_table(make_candles(_MAIN_SEQ))
  w3 = pd.Timestamp("2024-01-15", tz=_NY).normalize()
  assert float(week_table.loc[w3, "spike_pts"]) == pytest.approx(7.0)


def test_spike_pts_below_retraced_on_tuesday() -> None:
  """W4 (below, retraced Tue): spike = L - min low in [Mon09:30..Tue09:30].

  Mon 09:30 low=95, Mon 09:31 low=93, Tue 09:30 low=92 (first_low=92).
  spike_pts = 100 - 92 = 8. Tue 09:31 low=90 is OUTSIDE the window.
  """
  stat = _stat()
  week_table = stat.build_week_table(make_candles(_MAIN_SEQ))
  w4 = pd.Timestamp("2024-01-22", tz=_NY).normalize()
  assert float(week_table.loc[w4, "spike_pts"]) == pytest.approx(8.0)


def test_spike_pts_doji_is_na() -> None:
  """W5 (doji): spike_pts=pd.NA (no direction)."""
  stat = _stat()
  week_table = stat.build_week_table(make_candles(_MAIN_SEQ))
  w5 = pd.Timestamp("2024-01-29", tz=_NY).normalize()
  assert pd.isna(week_table.loc[w5, "spike_pts"])


def test_spike_stops_at_first_retrace_bar() -> None:
  """Spike is capped at the first retracing bar; a larger excursion AFTER must not count.

  Setup (1 resolved week + 1 pending):
    L=1000, first_close=1010 → opened_above.
    Mon 09:30: high=1010, low=1000.
    Mon 09:31: high=1015, low=1000 → retraced (low==L, weekday=0).
    Tue 09:30: high=1030 (first_high override) → OUTSIDE window; must not inflate spike.
    Tue 09:31: high=1025, low=990 → also outside.

  spike window = [Mon09:30, Mon09:31]: max(1010, 1015) - 1000 = 15.
  If window were extended: max(1010, 1015, 1030) - 1000 = 30 (wrong).
  """
  candles = make_candles([
    {"date": "2024-01-01", "open": 1000.0, "close": 1010.0, "high": 1015.0, "low": 1000.0},
    # Tue 09:30 first_high=1030 (after retrace) must not appear in spike window
    {
      "date": "2024-01-02", "open": 1010.0, "close": 1010.0,
      "high": 1010.0, "low": 990.0,
      "first_high": 1030.0, "first_low": 990.0,
    },
    # pending
    {"date": "2024-01-08", "open": 1000.0, "close": 1005.0, "high": 1010.0, "low": 1000.0},
  ])
  stat = _stat()
  week_table = stat.build_week_table(candles)
  w1 = pd.Timestamp("2024-01-01", tz=_NY).normalize()
  spike = float(week_table.loc[w1, "spike_pts"])
  # Correct: 15.0. Buggy (all bars): 30.0.
  assert spike == pytest.approx(15.0)
  assert spike < 30.0


# ===========================================================================
# 6. compute_rows partition invariants
# ===========================================================================

def test_opened_above_retraced_count_and_probability() -> None:
  """opened_above retraced: 1 of 2 above weeks (W2) → count=1, total=2, P=0.5."""
  result = _stat().compute(make_candles(_MAIN_SEQ))
  r = _row(result, "opened_above", "retraced")
  assert r.count == 1
  assert r.total == 2
  assert r.probability == pytest.approx(0.5)


def test_opened_above_not_retraced_count_and_probability() -> None:
  """opened_above not_retraced: 1 of 2 above weeks (W1) → count=1, total=2, P=0.5."""
  result = _stat().compute(make_candles(_MAIN_SEQ))
  r = _row(result, "opened_above", "not_retraced")
  assert r.count == 1
  assert r.total == 2
  assert r.probability == pytest.approx(0.5)


def test_opened_below_retraced_count_and_probability() -> None:
  """opened_below retraced: 1 of 2 below weeks (W4) → count=1, total=2, P=0.5."""
  result = _stat().compute(make_candles(_MAIN_SEQ))
  r = _row(result, "opened_below", "retraced")
  assert r.count == 1
  assert r.total == 2
  assert r.probability == pytest.approx(0.5)


def test_opened_below_not_retraced_count_and_probability() -> None:
  """opened_below not_retraced: 1 of 2 below weeks (W3) → count=1, total=2, P=0.5."""
  result = _stat().compute(make_candles(_MAIN_SEQ))
  r = _row(result, "opened_below", "not_retraced")
  assert r.count == 1
  assert r.total == 2
  assert r.probability == pytest.approx(0.5)


def test_tier1_outcomes_partition_each_direction() -> None:
  """retraced.count + not_retraced.count == total for each direction."""
  result = _stat().compute(make_candles(_MAIN_SEQ))
  for cond in ("opened_above", "opened_below"):
    ret = _row(result, cond, "retraced")
    not_ret = _row(result, cond, "not_retraced")
    assert ret.count + not_ret.count == ret.total == not_ret.total


def test_tier1_probabilities_sum_to_one_per_direction() -> None:
  """P(retraced) + P(not_retraced) == 1.0 for each direction with data."""
  result = _stat().compute(make_candles(_MAIN_SEQ))
  for cond in ("opened_above", "opened_below"):
    p_ret = _row(result, cond, "retraced").probability
    p_not = _row(result, cond, "not_retraced").probability
    assert p_ret + p_not == pytest.approx(1.0)


def test_nine_rows_exactly() -> None:
  """compute_rows emits exactly 9 rows: 4 tier-1 (2x2) + 5 tier-2 (weekdays)."""
  result = _stat().compute(make_candles(_MAIN_SEQ))
  rows = _tf(result).results
  assert len(rows) == 9
  expected_keys = {
    ("opened_above", "retraced"),
    ("opened_above", "not_retraced"),
    ("opened_below", "retraced"),
    ("opened_below", "not_retraced"),
    ("retraced", "monday"),
    ("retraced", "tuesday"),
    ("retraced", "wednesday"),
    ("retraced", "thursday"),
    ("retraced", "friday"),
  }
  assert {(r.condition, r.outcome) for r in rows} == expected_keys


def test_all_probabilities_in_unit_interval() -> None:
  """Every row's probability is in [0, 1]."""
  result = _stat().compute(make_candles(_MAIN_SEQ))
  for r in _tf(result).results:
    assert 0.0 <= r.probability <= 1.0


# ===========================================================================
# 7. Baseline
# ===========================================================================

def test_baseline_deterministic_same_seed() -> None:
  """Two baseline calls with the same seed return identical rows."""
  stat = _stat()
  df = make_candles(_MAIN_SEQ)
  rows_a = stat.baseline(df, seed=42)
  rows_b = stat.baseline(df, seed=42)
  assert len(rows_a) == len(rows_b)
  for a, b in zip(rows_a, rows_b):
    assert (a.condition, a.outcome) == (b.condition, b.outcome)
    assert a.probability == pytest.approx(b.probability)


def test_baseline_different_seeds_differ() -> None:
  """Different seeds produce different baseline results on a large fixture."""
  # 8 resolved alternating weeks (4 retraced) + 1 pending; enough permutations to differ
  stat = _stat()
  candles = _make_alternating_weeks(8)
  week_table = stat.build_week_table(candles)
  rows_42 = {(r.condition, r.outcome): r.count for r in stat.baseline_rows(week_table, seed=42)}
  rows_99 = {(r.condition, r.outcome): r.count for r in stat.baseline_rows(week_table, seed=99)}
  assert rows_42 != rows_99


def test_baseline_tier1_pooled_retracement_count_preserved() -> None:
  """Tier-1 permutation preserves total retraced count across all countable weeks.

  _MAIN_SEQ: 4 countable (W1-W4), 2 retraced. Permuting [T,T,F,F] keeps sum=2
  for every seed.
  """
  stat = _stat()
  week_table = stat.build_week_table(make_candles(_MAIN_SEQ))
  for seed in (0, 1, 42, 99):
    bl = stat.baseline_rows(week_table, seed=seed)
    bl_map = {(r.condition, r.outcome): r for r in bl}
    total_ret = (
      bl_map[("opened_above", "retraced")].count
      + bl_map[("opened_below", "retraced")].count
    )
    assert total_ret == 2, f"seed={seed}: expected 2 retraced, got {total_ret}"


def test_baseline_tier2_weekday_total_equals_retraced_total() -> None:
  """Tier-2 weekday rows have total == retraced_total (unchanged by randomization)."""
  # Retraced countable = 2 (W2, W4); tier-2 baseline keeps the 'retraced' column as-is
  stat = _stat()
  week_table = stat.build_week_table(make_candles(_MAIN_SEQ))
  bl = stat.baseline_rows(week_table, seed=42)
  bl_map = {(r.condition, r.outcome): r for r in bl}
  for day in ("monday", "tuesday", "wednesday", "thursday", "friday"):
    assert bl_map[("retraced", day)].total == 2


def test_baseline_n_embedded_after_compute() -> None:
  """After compute(), rows with total>0 carry baseline_n > 0."""
  result = _stat().compute(make_candles(_MAIN_SEQ))
  for r in _tf(result).results:
    if r.total > 0:
      assert r.baseline_n > 0, f"{r.condition}/{r.outcome}: baseline_n=0"


def test_baseline_reproducibility_two_compute_calls() -> None:
  """Two compute() calls with the same seed on the same data return identical rows."""
  stat = _stat()
  df = make_candles(_MAIN_SEQ)
  rows_a = stat.compute(df, seed=7).instruments["NQ"]["weekly"].results
  rows_b = stat.compute(df, seed=7).instruments["NQ"]["weekly"].results
  for a, b in zip(rows_a, rows_b):
    assert a.count == b.count
    assert a.probability == pytest.approx(b.probability)
    assert a.baseline_prob == pytest.approx(b.baseline_prob)


# ===========================================================================
# 8. Slices
# ===========================================================================

def test_slices_spike_pts_key_present() -> None:
  """Result carries a 'spike_pts' slice (declared SizeBucket)."""
  result = _stat().compute(make_candles(_MAIN_SEQ))
  assert "spike_pts" in _tf(result).slices


def test_slices_spike_pct_key_present() -> None:
  """Result carries a 'spike_pct' slice (declared SizeBucket)."""
  result = _stat().compute(make_candles(_MAIN_SEQ))
  assert "spike_pct" in _tf(result).slices


def test_slices_exactly_two_declared() -> None:
  """Exactly spike_pts and spike_pct slice dimensions are declared."""
  result = _stat().compute(make_candles(_MAIN_SEQ))
  assert set(_tf(result).slices.keys()) == {"spike_pts", "spike_pct"}


def test_slices_spike_pts_has_groups() -> None:
  """spike_pts slice produces at least one group from the 4 countable weeks."""
  result = _stat().compute(make_candles(_MAIN_SEQ))
  groups = _tf(result).slices["spike_pts"].groups
  assert len(groups) >= 1


def test_slices_doji_week_excluded_from_groups() -> None:
  """Doji week (spike_pts=NA) does not appear in any spike_pts slice group.

  Sum of group total_samples across all groups <= 4 (countable weeks only).
  """
  result = _stat().compute(make_candles(_MAIN_SEQ))
  groups = _tf(result).slices["spike_pts"].groups
  group_total = sum(g.total_samples for g in groups.values())
  # 5 resolved but doji (W5) has spike_pts=NA → excluded; max=4
  assert group_total <= 4


def test_slices_group_tier1_outcomes_partition() -> None:
  """For every spike_pts slice group: retraced+not_retraced count == total for each direction."""
  result = _stat().compute(make_candles(_MAIN_SEQ))
  for group in _tf(result).slices["spike_pts"].groups.values():
    results_map = {(r.condition, r.outcome): r for r in group.results}
    for cond in ("opened_above", "opened_below"):
      ret = results_map.get((cond, "retraced"))
      not_ret = results_map.get((cond, "not_retraced"))
      if ret is not None and not_ret is not None and ret.total > 0:
        assert ret.count + not_ret.count == ret.total


# ===========================================================================
# 9. End-to-end compute()
# ===========================================================================

def test_compute_stat_name() -> None:
  """stat_name attribute equals 'weekly_open_retracement'."""
  result = _stat().compute(make_candles(_MAIN_SEQ))
  assert result.stat_name == "weekly_open_retracement"


def test_compute_timeframe_key_is_weekly() -> None:
  """Instrument result is keyed under 'weekly' timeframe."""
  result = _stat().compute(make_candles(_MAIN_SEQ))
  assert "weekly" in result.instruments["NQ"]


def test_compute_total_samples() -> None:
  """total_samples == 5 (W1-W5 resolved; W6 pending dropped)."""
  result = _stat().compute(make_candles(_MAIN_SEQ))
  assert _tf(result).total_samples == 5


def test_compute_data_range() -> None:
  """data_range spans the Monday of the first to last resolved week."""
  # W1=Jan1, W5=Jan29 (both Mondays); W6=Feb5 dropped
  result = _stat().compute(make_candles(_MAIN_SEQ))
  assert _tf(result).data_range == ["2024-01-01", "2024-01-29"]


def test_compute_title_and_definition_non_empty() -> None:
  """title and definition have non-empty en and fr strings."""
  result = _stat().compute(_empty_df())
  assert result.title.en != "" and result.title.fr != ""
  assert result.definition.en != "" and result.definition.fr != ""


def test_compute_labels_correct_condition_keys() -> None:
  """Labels.conditions carries opened_above, opened_below, retraced with en/fr."""
  result = _stat().compute(_empty_df())
  assert set(result.labels.conditions) >= {"opened_above", "opened_below", "retraced"}
  for key, label in result.labels.conditions.items():
    assert label.en != "", f"conditions[{key}].en empty"
    assert label.fr != "", f"conditions[{key}].fr empty"


def test_compute_labels_correct_outcome_keys() -> None:
  """Labels.outcomes carries all 7 expected keys with en/fr strings."""
  result = _stat().compute(_empty_df())
  expected = {"retraced", "not_retraced", "monday", "tuesday", "wednesday", "thursday", "friday"}
  assert set(result.labels.outcomes) == expected
  for key, label in result.labels.outcomes.items():
    assert label.en != "", f"outcomes[{key}].en empty"
    assert label.fr != "", f"outcomes[{key}].fr empty"


def test_compute_labels_dimensions_contain_slicers() -> None:
  """Labels.dimensions carries entries for both declared slice dimensions."""
  result = _stat().compute(make_candles(_MAIN_SEQ))
  assert set(result.labels.dimensions.keys()) >= {"spike_pts", "spike_pct"}


# ===========================================================================
# 10. Pending discipline
# ===========================================================================

def test_pending_last_week_not_counted() -> None:
  """The last ISO week is always dropped even if it would affect results.

  Two-week fixture: W1 (above, retraced) would be W1=resolved, W2=pending.
  If W2 were counted, total_samples=2. With pending rule, total_samples=1.
  """
  # W1=resolved, W2=pending; opened_above total=1 (W1 only)
  candles = make_candles([
    {"date": "2024-01-01", "open": 100.0, "close": 105.0, "high": 112.0, "low": 100.0},
    {"date": "2024-01-08", "open": 100.0, "close": 108.0, "high": 115.0, "low": 99.0},
  ])
  result = _stat().compute(candles)
  assert _tf(result).total_samples == 1
  assert _row(result, "opened_above", "retraced").total == 1


def test_pending_in_progress_week_excluded_from_probabilities() -> None:
  """The in-progress week's retracement outcome does not affect probabilities.

  Three-week fixture: W1=above-not, W2=above-retraced (would give P=0.5),
  W3=pending (would be above-retraced if counted, boosting P to 2/3).
  Correct result: P=0.5 (W1 and W2 only).
  """
  # W1: above, not retraced; W2: above, retraced; W3: pending above-retraced
  candles = make_candles([
    {"date": "2024-01-01", "open": 100.0, "close": 105.0, "high": 108.0, "low": 103.0},
    {"date": "2024-01-08", "open": 100.0, "close": 105.0, "high": 112.0, "low": 100.0},
    {"date": "2024-01-15", "open": 100.0, "close": 106.0, "high": 115.0, "low": 100.0},
  ])
  result = _stat().compute(candles)
  # Only W1 and W2 resolved. opened_above total=2, retraced=1 (W2) → P=0.5
  r = _row(result, "opened_above", "retraced")
  assert r.total == 2
  assert r.count == 1
  assert r.probability == pytest.approx(0.5)


# ===========================================================================
# 11. Empty / edge cases
# ===========================================================================

def test_empty_dataframe_returns_zero_total_samples() -> None:
  """Empty input → total_samples=0, data_range=[]."""
  result = _stat().compute(_empty_df())
  tf = _tf(result)
  assert tf.total_samples == 0
  assert tf.data_range == []


def test_empty_dataframe_nine_zero_rows() -> None:
  """Empty input → nine zero rows with count=0, total=0, probability=0.0."""
  result = _stat().compute(_empty_df())
  rows = _tf(result).results
  assert len(rows) == 9
  for r in rows:
    assert r.count == 0
    assert r.total == 0
    assert r.probability == pytest.approx(0.0)


def test_all_opened_above_below_has_zero_total() -> None:
  """All weeks open above → opened_below has total=0 and P=0.0."""
  # 2 resolved above + 1 pending (all above)
  candles = make_candles([
    {"date": "2024-01-01", "open": 100.0, "close": 105.0, "high": 108.0, "low": 103.0},
    {"date": "2024-01-08", "open": 100.0, "close": 106.0, "high": 110.0, "low": 101.0},
    {"date": "2024-01-15", "open": 100.0, "close": 104.0, "high": 107.0, "low": 103.0},
  ])
  result = _stat().compute(candles)
  r_below = _row(result, "opened_below", "retraced")
  assert r_below.total == 0
  assert r_below.probability == pytest.approx(0.0)
  assert _row(result, "opened_above", "retraced").total == 2


def test_all_opened_below_above_has_zero_total() -> None:
  """All weeks open below → opened_above has total=0 and P=0.0."""
  candles = make_candles([
    {"date": "2024-01-01", "open": 100.0, "close": 95.0, "high": 98.0, "low": 93.0},
    {"date": "2024-01-08", "open": 100.0, "close": 94.0, "high": 97.0, "low": 92.0},
    {"date": "2024-01-15", "open": 100.0, "close": 96.0, "high": 99.0, "low": 94.0},
  ])
  result = _stat().compute(candles)
  assert _row(result, "opened_above", "retraced").total == 0
  assert _row(result, "opened_below", "retraced").total == 2


# ===========================================================================
# 12. write_results round-trip
# ===========================================================================

def test_write_results_round_trip(tmp_path: Path) -> None:
  """write_results produces weekly_open_retracement.json that re-validates correctly."""
  result = _stat().compute(make_candles(_MAIN_SEQ))
  written = write_results(result, results_dir=tmp_path)
  assert written.name == "weekly_open_retracement.json"

  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)
  tf = validated.instruments["NQ"]["weekly"]
  assert tf.total_samples == 5

  oa_ret = next(
    r for r in tf.results if r.condition == "opened_above" and r.outcome == "retraced"
  )
  assert oa_ret.count == 1
  assert oa_ret.total == 2
  assert oa_ret.probability == pytest.approx(0.5)


def test_write_results_utf8_literals(tmp_path: Path) -> None:
  """French text is stored as literal UTF-8, not escaped unicode (e.g. \\u00e9)."""
  result = _stat().compute(_empty_df())
  written = write_results(result, results_dir=tmp_path)
  raw = written.read_text(encoding="utf-8")
  # French text contains accented chars like "é" (e.g. "Retracé" / "fréquence")
  assert "é" in raw
  assert "\\u00e9" not in raw


def test_write_results_has_both_slice_dimensions(tmp_path: Path) -> None:
  """Serialised JSON slices dict contains spike_pts and spike_pct."""
  result = _stat().compute(make_candles(_MAIN_SEQ))
  written = write_results(result, results_dir=tmp_path)
  raw = json.loads(written.read_text(encoding="utf-8"))
  slices = raw["instruments"]["NQ"]["weekly"]["slices"]
  assert set(slices.keys()) == {"spike_pts", "spike_pct"}
