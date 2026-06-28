"""Tests for stats.fibonacci_levels.standard.

All data is synthetic — no real market files required. Expected values are
hand-calculated before each assertion.

Fib anchoring by prior candle direction:
  GREEN prior: level(f) = prev_high - f * rng   (f=0 at top, f=1 at bottom)
  RED prior:   level(f) = prev_low  + f * rng   (f=0 at bottom, f=1 at top)

Opening zone fraction (same anchored scale):
  GREEN prior: open_frac = (prev_high - session_open) / rng
  RED prior:   open_frac = (session_open - prev_low)  / rng

Zone boundaries (half-open, 786_100 inclusive at 1.0):
  below_0   : open_frac < 0
  0_236     : 0      <= open_frac < 0.236
  236_382   : 0.236  <= open_frac < 0.382
  382_500   : 0.382  <= open_frac < 0.5
  500_618   : 0.5    <= open_frac < 0.618
  618_786   : 0.618  <= open_frac < 0.786
  786_100   : 0.786  <= open_frac <= 1.0
  above_100 : open_frac > 1.0

Touch rule: day_low <= level <= day_high.
Outcomes are INDEPENDENT (one session can touch multiple levels).
Zone outcomes DO partition (exactly one zone per session).
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.fibonacci_levels.standard import FibonacciLevels

# ---------------------------------------------------------------------------
# Minimal InstrumentConfig (does not depend on NQ.yaml)
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

_RTH_START = 570   # 09:30
_RTH_LAST = 974    # 16:14 (last bar before 16:15); resolved needs last mod >= 960


# ---------------------------------------------------------------------------
# Synthetic day builder (mirrors test_prev_days_range._make_day exactly)
# ---------------------------------------------------------------------------

def _make_day(
  date: str,
  session_open: float,
  session_close: float,
  day_high: float,
  day_low: float,
) -> pd.DataFrame:
  """One trading day of 1-min RTH bars (09:30–16:14).

  The opening bar carries ``session_open`` as its open; the last bar carries
  ``session_close`` as its close. The day's RTH extremes ``day_high`` /
  ``day_low`` are placed on a neutral mid-session bar so they are independent
  of the open/close prices.

  IMPORTANT: callers must ensure ``day_high >= max(session_open, session_close)``
  and ``day_low <= min(session_open, session_close)`` so that the grouped max/min
  in ``build_day_table_with_prior_range`` returns exactly the requested extremes.
  """
  base = pd.Timestamp(date, tz=_NY)
  mid = (_RTH_START + _RTH_LAST) // 2
  records = []
  for mod in range(_RTH_START, _RTH_LAST + 1):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    o = session_open if mod == _RTH_START else session_close
    c = session_close
    hi = day_high if mod == mid else max(o, c)
    lo = day_low if mod == mid else min(o, c)
    records.append({
      "timestamp": ts,
      "open": o,
      "high": hi,
      "low": lo,
      "close": c,
      "volume": 1000,
    })
  return pd.DataFrame(records)


def make_candles(days: list[dict]) -> pd.DataFrame:
  """Concatenate per-day specs into a single sorted 1-min OHLCV DataFrame."""
  frames = [
    _make_day(d["date"], d["open"], d["close"], d["high"], d["low"])
    for d in days
  ]
  df = pd.concat(frames, ignore_index=True)
  return df.sort_values("timestamp").reset_index(drop=True)


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


def _empty_df() -> pd.DataFrame:
  df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
  df["timestamp"] = pd.array([], dtype="datetime64[ns, America/New_York]")
  return df


# ---------------------------------------------------------------------------
# Stat factory and result accessors
# ---------------------------------------------------------------------------

def _stat() -> FibonacciLevels:
  return FibonacciLevels(instrument="NQ", config=_TEST_CONFIG)


def _row(result: StatRunResult, condition: str, outcome: str) -> StatResultRow:
  for r in result.instruments["NQ"]["daily"].results:
    if r.condition == condition and r.outcome == outcome:
      return r
  raise KeyError((condition, outcome))


# ---------------------------------------------------------------------------
# Canonical prior days
#
# GREEN prior: open=100, close=200, high=200, low=100
#   → prev_high=200, prev_low=100, rng=100, prev_session_green=True
# RED prior:   open=200, close=100, high=200, low=100
#   → prev_high=200, prev_low=100, rng=100, prev_session_green=False
#
# In _make_day, actual day_high = max(day_high_param, max(open, close)).
# Setting day_high_param = max(open, close) ensures the param is the effective
# extreme. For GREEN: max(100, 200)=200 → prev_high=200 ✓
# For RED:            max(200, 100)=200 → prev_high=200 ✓
# ---------------------------------------------------------------------------

_PRIOR_GREEN_DATE = "2024-01-02"
_PRIOR_GREEN = {
  "date": _PRIOR_GREEN_DATE,
  "open": 100.0, "close": 200.0, "high": 200.0, "low": 100.0,
}
# prev_session_green=True (close=200 >= open=100)

_PRIOR_RED_DATE = "2024-01-02"
_PRIOR_RED = {
  "date": _PRIOR_RED_DATE,
  "open": 200.0, "close": 100.0, "high": 200.0, "low": 100.0,
}
# prev_session_green=False (close=100 < open=200)

_CURRENT_DATE = "2024-01-03"


# ===========================================================================
# 1. Fibonacci level price computation
#
# Both anchor orientations use prev_low=100, prev_high=200, rng=100.
#
# GREEN prior levels (retrace down from prev_high):
#   f=0.000 → 200 - 0.000*100 = 200.0
#   f=0.236 → 200 - 0.236*100 = 176.4
#   f=0.382 → 200 - 0.382*100 = 161.8
#   f=0.500 → 200 - 0.500*100 = 150.0
#   f=0.618 → 200 - 0.618*100 = 138.2
#   f=0.786 → 200 - 0.786*100 = 121.4
#   f=1.000 → 200 - 1.000*100 = 100.0
#
# RED prior levels (retrace up from prev_low):
#   f=0.000 → 100 + 0.000*100 = 100.0
#   f=0.236 → 100 + 0.236*100 = 123.6
#   f=0.382 → 100 + 0.382*100 = 138.2
#   f=0.500 → 100 + 0.500*100 = 150.0
#   f=0.618 → 100 + 0.618*100 = 161.8
#   f=0.786 → 100 + 0.786*100 = 178.6
#   f=1.000 → 100 + 1.000*100 = 200.0
# ===========================================================================

def test_green_prior_touch_0_level_is_200() -> None:
  """GREEN prior: f=0 anchor is prev_high=200. Session straddling 200 triggers touch_0."""
  # day_high=201, day_low=199 straddles level=200 exclusively.
  # Levels: 200, 176.4, 161.8, 150, 138.2, 121.4, 100.
  # Only 200 is in [199, 201]; the nearest others (176.4, 100) are outside.
  days = [
    _PRIOR_GREEN,
    {"date": _CURRENT_DATE, "open": 200.0, "close": 200.5, "high": 201.0, "low": 199.0},
  ]
  result = _stat().compute(make_candles(days))
  assert _row(result, "fib_levels", "touch_0").count == 1
  assert _row(result, "fib_levels", "touch_236").count == 0
  assert _row(result, "fib_levels", "touch_100").count == 0


def test_green_prior_touch_236_level_is_176_4() -> None:
  """GREEN prior: f=0.236 → 200 - 0.236*100 = 176.4. Day band [176, 177] touches it."""
  # Levels outside [176, 177]: 200, 161.8, 150, 138.2, 121.4, 100. ✓
  days = [
    _PRIOR_GREEN,
    {"date": _CURRENT_DATE, "open": 176.5, "close": 176.8, "high": 177.0, "low": 176.0},
  ]
  result = _stat().compute(make_candles(days))
  assert _row(result, "fib_levels", "touch_0").count == 0
  assert _row(result, "fib_levels", "touch_236").count == 1
  assert _row(result, "fib_levels", "touch_382").count == 0


def test_green_prior_touch_382_level_is_161_8() -> None:
  """GREEN prior: f=0.382 → 161.8. Day band [161, 162] touches it."""
  # [161, 162]: 161.8 in range; 176.4 above, 150 below. ✓
  days = [
    _PRIOR_GREEN,
    {"date": _CURRENT_DATE, "open": 161.5, "close": 161.8, "high": 162.0, "low": 161.0},
  ]
  result = _stat().compute(make_candles(days))
  assert _row(result, "fib_levels", "touch_236").count == 0
  assert _row(result, "fib_levels", "touch_382").count == 1
  assert _row(result, "fib_levels", "touch_500").count == 0


def test_green_prior_touch_100_level_is_100() -> None:
  """GREEN prior: f=1.0 far extreme is prev_low=100. Day band [99, 101] touches it."""
  # [99, 101]: 100 in range; nearest other is 121.4. ✓
  days = [
    _PRIOR_GREEN,
    {"date": _CURRENT_DATE, "open": 100.0, "close": 100.5, "high": 101.0, "low": 99.0},
  ]
  result = _stat().compute(make_candles(days))
  assert _row(result, "fib_levels", "touch_786").count == 0
  assert _row(result, "fib_levels", "touch_100").count == 1


def test_red_prior_touch_0_level_is_100() -> None:
  """RED prior: f=0 anchor is prev_low=100. Day band [99, 101] triggers touch_0."""
  # Levels: 100, 123.6, 138.2, 150, 161.8, 178.6, 200.
  # [99, 101]: only 100 in range; nearest other 123.6 is above. ✓
  days = [
    _PRIOR_RED,
    {"date": _CURRENT_DATE, "open": 100.0, "close": 100.5, "high": 101.0, "low": 99.0},
  ]
  result = _stat().compute(make_candles(days))
  assert _row(result, "fib_levels", "touch_0").count == 1
  assert _row(result, "fib_levels", "touch_236").count == 0


def test_red_prior_touch_236_level_is_123_6() -> None:
  """RED prior: f=0.236 → 100 + 0.236*100 = 123.6. Day band [123, 124] touches it."""
  # [123, 124]: 123.6 in range; 100 below, 138.2 above. ✓
  days = [
    _PRIOR_RED,
    {"date": _CURRENT_DATE, "open": 123.5, "close": 123.7, "high": 124.0, "low": 123.0},
  ]
  result = _stat().compute(make_candles(days))
  assert _row(result, "fib_levels", "touch_0").count == 0
  assert _row(result, "fib_levels", "touch_236").count == 1
  assert _row(result, "fib_levels", "touch_382").count == 0


def test_red_prior_touch_786_level_is_178_6() -> None:
  """RED prior: f=0.786 → 100 + 0.786*100 = 178.6. Day band [178, 179] touches it."""
  # [178, 179]: 178.6 in range; 161.8 below, 200 above. ✓
  days = [
    _PRIOR_RED,
    {"date": _CURRENT_DATE, "open": 178.5, "close": 178.8, "high": 179.0, "low": 178.0},
  ]
  result = _stat().compute(make_candles(days))
  assert _row(result, "fib_levels", "touch_618").count == 0
  assert _row(result, "fib_levels", "touch_786").count == 1
  assert _row(result, "fib_levels", "touch_100").count == 0


def test_red_prior_touch_100_level_is_200() -> None:
  """RED prior: f=1.0 far extreme is prev_high=200. Day band [199, 201] touches it."""
  days = [
    _PRIOR_RED,
    {"date": _CURRENT_DATE, "open": 200.0, "close": 200.5, "high": 201.0, "low": 199.0},
  ]
  result = _stat().compute(make_candles(days))
  assert _row(result, "fib_levels", "touch_786").count == 0
  assert _row(result, "fib_levels", "touch_100").count == 1


# ===========================================================================
# 2. Touch detection: boundary and multi-level scenarios
#
# Canonical GREEN prior: prev_high=200, prev_low=100, rng=100.
# Levels: 200.0, 176.4, 161.8, 150.0, 138.2, 121.4, 100.0
# ===========================================================================

def test_touch_nothing_when_band_between_two_levels() -> None:
  """Day band [162, 174] spans the gap between L382=161.8 and L236=176.4 — no touch.

  No fib level falls in [162, 174]:
    L236=176.4 > 174, L382=161.8 < 162. All other levels outside too.
  """
  days = [
    _PRIOR_GREEN,
    {"date": _CURRENT_DATE, "open": 165.0, "close": 168.0, "high": 174.0, "low": 162.0},
  ]
  result = _stat().compute(make_candles(days))
  for key in ("touch_0", "touch_236", "touch_382", "touch_500", "touch_618", "touch_786", "touch_100"):
    assert _row(result, "fib_levels", key).count == 0, f"{key} should be 0"


def test_touch_all_levels_when_band_spans_full_range() -> None:
  """Day band [99, 201] brackets all seven fib levels → every touch_* count = 1.

  countable_n = 1. All levels 100.0–200.0 lie within [99, 201]. ✓
  """
  days = [
    _PRIOR_GREEN,
    {"date": _CURRENT_DATE, "open": 150.0, "close": 155.0, "high": 201.0, "low": 99.0},
  ]
  result = _stat().compute(make_candles(days))
  for key in ("touch_0", "touch_236", "touch_382", "touch_500", "touch_618", "touch_786", "touch_100"):
    row = _row(result, "fib_levels", key)
    assert row.count == 1, f"{key} should be 1"
    assert row.total == 1


def test_touch_only_500_level() -> None:
  """Day band [149, 151] touches only L500=150.

  L382=161.8 > 151, L618=138.2 < 149 — only L500=150 in [149, 151]. ✓
  countable_n = 1.
  """
  days = [
    _PRIOR_GREEN,
    {"date": _CURRENT_DATE, "open": 150.0, "close": 150.5, "high": 151.0, "low": 149.0},
  ]
  result = _stat().compute(make_candles(days))
  assert _row(result, "fib_levels", "touch_500").count == 1
  for key in ("touch_0", "touch_236", "touch_382", "touch_618", "touch_786", "touch_100"):
    assert _row(result, "fib_levels", key).count == 0, f"{key} should be 0"


def test_touch_exactly_at_boundary_is_a_touch() -> None:
  """day_low == level → touched (inclusive boundary).

  Level L500=150. Day where day_low exactly equals 150 → touch_500=True.
  day_high must be >= 150 (= day_low). Use day_high=155, day_low=150.
  """
  days = [
    _PRIOR_GREEN,
    {"date": _CURRENT_DATE, "open": 152.0, "close": 153.0, "high": 155.0, "low": 150.0},
  ]
  result = _stat().compute(make_candles(days))
  assert _row(result, "fib_levels", "touch_500").count == 1


def test_touch_just_above_level_is_not_a_touch() -> None:
  """day_low = 150.1 (above L500=150) with day_high=155 → L500 NOT touched.

  day_low=150.1 > 150 and day_high=155 > 150 → level 150 not in [150.1, 155].
  """
  days = [
    _PRIOR_GREEN,
    {"date": _CURRENT_DATE, "open": 152.0, "close": 153.0, "high": 155.0, "low": 150.1},
  ]
  result = _stat().compute(make_candles(days))
  assert _row(result, "fib_levels", "touch_500").count == 0


def test_touch_outcomes_independent_same_session_counts_in_multiple() -> None:
  """A session touching L500=150 AND L382=161.8 is counted in both touch rows.

  Day band [149, 162]: 150 in range ✓, 161.8 in range ✓.
  L236=176.4 > 162, L618=138.2 < 149 → not touched. ✓
  """
  days = [
    _PRIOR_GREEN,
    {"date": _CURRENT_DATE, "open": 155.0, "close": 156.0, "high": 162.0, "low": 149.0},
  ]
  result = _stat().compute(make_candles(days))
  assert _row(result, "fib_levels", "touch_382").count == 1
  assert _row(result, "fib_levels", "touch_500").count == 1
  # Both share the same total (1 countable session)
  assert _row(result, "fib_levels", "touch_382").total == 1
  assert _row(result, "fib_levels", "touch_500").total == 1


def test_all_touch_rows_share_same_total() -> None:
  """total is countable_n for every touch row (one prior + two current sessions)."""
  days = [
    _PRIOR_GREEN,
    # Day 1: touches nothing
    {"date": "2024-01-03", "open": 165.0, "close": 168.0, "high": 174.0, "low": 162.0},
    # Day 2: touches all
    {"date": "2024-01-04", "open": 150.0, "close": 155.0, "high": 201.0, "low": 99.0},
  ]
  # Day 2's prior is Day 1 (prev_high=174, prev_low=162, rng=12 > 0) → countable.
  # Day 1's prior is Day 0 (prev_high=200, prev_low=100, rng=100) → countable.
  # countable_n = 2 (Day 1 and Day 2).
  result = _stat().compute(make_candles(days))
  for key in ("touch_0", "touch_236", "touch_382", "touch_500", "touch_618", "touch_786", "touch_100"):
    row = _row(result, "fib_levels", key)
    assert row.total == 2, f"{key}.total should be 2"


# ===========================================================================
# 3. Opening zone — one test per zone (GREEN prior, 2-day scenarios)
#
# GREEN prior: prev_high=200, prev_low=100, rng=100.
# open_frac = (200 - session_open) / 100
#
# Zone       | open_frac range      | session_open range
# -----------+----------------------+-------------------
# below_0    | < 0                  | > 200
# 0_236      | [0, 0.236)           | (176.4, 200]
# 236_382    | [0.236, 0.382)       | (161.8, 176.4]
# 382_500    | [0.382, 0.5)         | (150, 161.8]
# 500_618    | [0.5, 0.618)         | (138.2, 150]
# 618_786    | [0.618, 0.786)       | (121.4, 138.2]
# 786_100    | [0.786, 1.0]         | [100, 121.4]
# above_100  | > 1.0                | < 100
# ===========================================================================

_ALL_ZONE_KEYS = (
  "below_0", "0_236", "236_382", "382_500",
  "500_618", "618_786", "786_100", "above_100",
)


def _zone_two_days(
  session_open: float,
  session_close: float,
  day_high: float,
  day_low: float,
) -> StatRunResult:
  """Build [prior_green, current_session] and compute."""
  days = [
    _PRIOR_GREEN,
    {"date": _CURRENT_DATE, "open": session_open, "close": session_close,
     "high": day_high, "low": day_low},
  ]
  return _stat().compute(make_candles(days))


def _assert_only_zone(result: StatRunResult, expected_zone: str) -> None:
  """Assert exactly one zone has count=1 and all others have count=0."""
  for key in _ALL_ZONE_KEYS:
    row = _row(result, "opening_zone", key)
    if key == expected_zone:
      assert row.count == 1, f"expected {key}.count=1, got {row.count}"
      assert row.total == 1
    else:
      assert row.count == 0, f"expected {key}.count=0, got {row.count} (expected zone={expected_zone})"


def test_opening_zone_below_0() -> None:
  """session_open=205 → open_frac=(200-205)/100=-0.05 → below_0."""
  # day_high/low chosen to not accidentally trigger other zones.
  result = _zone_two_days(205.0, 206.0, 210.0, 204.0)
  _assert_only_zone(result, "below_0")


def test_opening_zone_0_236() -> None:
  """session_open=190 → open_frac=(200-190)/100=0.10 → 0_236 (0 <= 0.10 < 0.236)."""
  result = _zone_two_days(190.0, 191.0, 195.0, 188.0)
  _assert_only_zone(result, "0_236")


def test_opening_zone_236_382() -> None:
  """session_open=170 → open_frac=(200-170)/100=0.30 → 236_382 (0.236 <= 0.30 < 0.382)."""
  result = _zone_two_days(170.0, 171.0, 175.0, 168.0)
  _assert_only_zone(result, "236_382")


def test_opening_zone_382_500() -> None:
  """session_open=155 → open_frac=(200-155)/100=0.45 → 382_500 (0.382 <= 0.45 < 0.5)."""
  result = _zone_two_days(155.0, 156.0, 160.0, 153.0)
  _assert_only_zone(result, "382_500")


def test_opening_zone_500_618() -> None:
  """session_open=145 → open_frac=(200-145)/100=0.55 → 500_618 (0.5 <= 0.55 < 0.618)."""
  result = _zone_two_days(145.0, 146.0, 148.0, 143.0)
  _assert_only_zone(result, "500_618")


def test_opening_zone_618_786() -> None:
  """session_open=130 → open_frac=(200-130)/100=0.70 → 618_786 (0.618 <= 0.70 < 0.786)."""
  result = _zone_two_days(130.0, 131.0, 135.0, 128.0)
  _assert_only_zone(result, "618_786")


def test_opening_zone_786_100() -> None:
  """session_open=110 → open_frac=(200-110)/100=0.90 → 786_100 (0.786 <= 0.90 <= 1.0)."""
  result = _zone_two_days(110.0, 111.0, 115.0, 108.0)
  _assert_only_zone(result, "786_100")


def test_opening_zone_above_100() -> None:
  """session_open=90 → open_frac=(200-90)/100=1.10 → above_100 (> 1.0)."""
  result = _zone_two_days(90.0, 91.0, 95.0, 88.0)
  _assert_only_zone(result, "above_100")


def test_opening_zone_boundary_frac_exactly_one_lands_in_786_100() -> None:
  """open_frac=1.0 belongs to 786_100 (not above_100).

  GREEN prior: session_open=100 → open_frac=(200-100)/100=1.0.
  Zone 786_100 uses (open_frac <= 1.0); above_100 uses (open_frac > 1.0).
  """
  # session_open=100, session_close=101.
  # day_high=105 (>=101), day_low=98 (<=100). ✓
  days = [
    _PRIOR_GREEN,
    {"date": _CURRENT_DATE, "open": 100.0, "close": 101.0, "high": 105.0, "low": 98.0},
  ]
  result = _stat().compute(make_candles(days))
  assert _row(result, "opening_zone", "786_100").count == 1
  assert _row(result, "opening_zone", "above_100").count == 0


def test_opening_zone_boundary_frac_just_above_one_lands_in_above_100() -> None:
  """open_frac=1.01 (session_open=99) belongs to above_100.

  GREEN prior: open_frac=(200-99)/100=1.01 > 1.0 → above_100. ✓
  """
  days = [
    _PRIOR_GREEN,
    {"date": _CURRENT_DATE, "open": 99.0, "close": 99.5, "high": 103.0, "low": 98.0},
  ]
  result = _stat().compute(make_candles(days))
  assert _row(result, "opening_zone", "above_100").count == 1
  assert _row(result, "opening_zone", "786_100").count == 0


def test_opening_zone_red_prior_below_0_is_open_below_prev_low() -> None:
  """RED prior: open_frac = (session_open - prev_low) / rng.

  session_open=90 < prev_low=100 → open_frac=(90-100)/100=-0.1 → below_0.
  """
  # RED prior: prev_low=100, prev_high=200.
  days = [
    _PRIOR_RED,
    {"date": _CURRENT_DATE, "open": 90.0, "close": 91.0, "high": 95.0, "low": 88.0},
  ]
  result = _stat().compute(make_candles(days))
  assert _row(result, "opening_zone", "below_0").count == 1


def test_opening_zone_partition_sums_to_countable_n() -> None:
  """Sum of all zone counts == countable_n for any valid input.

  Build 8 sessions after one GREEN prior (8 distinct zones, 8 countable days).
  Note: Day 2 uses Day 1 as its prior, Day 3 uses Day 2 etc. The zones are
  determined by each day's actual prior, so we only verify the partition sum,
  not the specific zone assignment per day.
  """
  days = [
    # Prior
    {"date": "2024-01-02", "open": 100.0, "close": 200.0, "high": 200.0, "low": 100.0},
    # 8 current sessions (various opens spread around each prior's range)
    {"date": "2024-01-03", "open": 205.0, "close": 206.0, "high": 210.0, "low": 204.0},
    {"date": "2024-01-04", "open": 190.0, "close": 191.0, "high": 195.0, "low": 188.0},
    {"date": "2024-01-05", "open": 170.0, "close": 171.0, "high": 175.0, "low": 168.0},
    {"date": "2024-01-08", "open": 155.0, "close": 156.0, "high": 160.0, "low": 153.0},
    {"date": "2024-01-09", "open": 145.0, "close": 146.0, "high": 148.0, "low": 143.0},
    {"date": "2024-01-10", "open": 130.0, "close": 131.0, "high": 135.0, "low": 128.0},
    {"date": "2024-01-11", "open": 110.0, "close": 111.0, "high": 115.0, "low": 108.0},
    {"date": "2024-01-12", "open":  90.0, "close":  91.0, "high":  95.0, "low":  88.0},
  ]
  result = _stat().compute(make_candles(days))
  tf = result.instruments["NQ"]["daily"]
  countable_n = _row(result, "fib_levels", "touch_0").total  # total == countable_n
  zone_total = sum(
    r.count for r in tf.results if r.condition == "opening_zone"
  )
  assert zone_total == countable_n, (
    f"Zone counts sum to {zone_total} but countable_n={countable_n}"
  )
  assert countable_n == 8  # 8 days have a prior resolved day


# ===========================================================================
# 4. Pending-sample discipline
# ===========================================================================

def test_first_day_excluded_from_touch_denominator() -> None:
  """The first resolved day (NaN prior) never appears in any touch row's total.

  With 1 prior + 2 current sessions → total_samples=3, countable_n=2.
  total on every touch/zone row must be 2 (not 3).
  """
  # Day 0: prior (no prior itself → NaN prev_high → not countable)
  # Day 1: countable (prior = Day 0)
  # Day 2: countable (prior = Day 1)
  days = [
    {"date": "2024-01-02", "open": 100.0, "close": 200.0, "high": 200.0, "low": 100.0},
    {"date": "2024-01-03", "open": 165.0, "close": 168.0, "high": 174.0, "low": 162.0},
    {"date": "2024-01-04", "open": 150.0, "close": 155.0, "high": 160.0, "low": 148.0},
  ]
  result = _stat().compute(make_candles(days))
  assert result.instruments["NQ"]["daily"].total_samples == 3
  for key in ("touch_0", "touch_236", "touch_382", "touch_500", "touch_618", "touch_786", "touch_100"):
    assert _row(result, "fib_levels", key).total == 2, f"{key}.total should be 2"
  for key in _ALL_ZONE_KEYS:
    assert _row(result, "opening_zone", key).total == 2, f"{key}.total should be 2"


def test_total_samples_counts_all_resolved_days() -> None:
  """total_samples = all resolved days including the first (which has no prior)."""
  days = [
    {"date": "2024-01-02", "open": 100.0, "close": 200.0, "high": 200.0, "low": 100.0},
    {"date": "2024-01-03", "open": 165.0, "close": 168.0, "high": 174.0, "low": 162.0},
    {"date": "2024-01-04", "open": 150.0, "close": 155.0, "high": 160.0, "low": 148.0},
    {"date": "2024-01-05", "open": 145.0, "close": 148.0, "high": 152.0, "low": 143.0},
  ]
  result = _stat().compute(make_candles(days))
  assert result.instruments["NQ"]["daily"].total_samples == 4


def test_zero_range_prior_is_not_countable() -> None:
  """A prior day with prev_high == prev_low (zero range) is excluded from totals.

  Day 0: high=100, low=100 (zero range)
  Day 1: has prior range = 0 → NOT countable
  Day 2: prior is Day 1 (check Day 1's range)

  With Day 0's range=0, Day 1 has prev_high=100, prev_low=100 → not countable.
  Day 2's prior is Day 1 (range depends on Day 1's values).
  total_samples=3, countable_n depends on Day 1's day_high/day_low.
  We just verify Day 1 is excluded from the denominator.
  """
  # Day 0: flat session (zero range)
  # Day 1: prev = Day 0 → rng=0 → not countable
  # Day 2: prev = Day 1 → Day 1 has day_high=160, day_low=148, rng=12 > 0 → countable
  days = [
    {"date": "2024-01-02", "open": 100.0, "close": 100.0, "high": 100.0, "low": 100.0},
    {"date": "2024-01-03", "open": 150.0, "close": 155.0, "high": 160.0, "low": 148.0},
    {"date": "2024-01-04", "open": 155.0, "close": 158.0, "high": 162.0, "low": 153.0},
  ]
  result = _stat().compute(make_candles(days))
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 3
  # Only Day 2 is countable (Day 0 has no prior; Day 1's prior=Day 0 has rng=0)
  assert _row(result, "fib_levels", "touch_0").total == 1


def test_truncated_day_absent_from_day_table() -> None:
  """A session that ends before 16:00 (unresolved) never appears in the day table."""
  stat = _stat()
  truncated = _make_truncated_day("2024-01-05")
  day_table = stat.build_day_table(truncated)
  pending_date = pd.Timestamp("2024-01-05", tz=_NY).normalize()
  assert pending_date not in day_table.index


def test_truncated_day_excluded_from_total_samples() -> None:
  """Adding a truncated day to a resolved sequence does not increase total_samples."""
  stat = _stat()
  base_days = [
    {"date": "2024-01-02", "open": 100.0, "close": 200.0, "high": 200.0, "low": 100.0},
    {"date": "2024-01-03", "open": 165.0, "close": 168.0, "high": 174.0, "low": 162.0},
  ]
  base_df = make_candles(base_days)
  base_total = stat.compute(base_df).instruments["NQ"]["daily"].total_samples

  truncated = _make_truncated_day("2024-01-04")
  combined = (
    pd.concat([base_df, truncated], ignore_index=True)
    .sort_values("timestamp")
    .reset_index(drop=True)
  )
  combined_total = stat.compute(combined).instruments["NQ"]["daily"].total_samples
  assert combined_total == base_total == 2


# ===========================================================================
# 5. Baseline — determinism and reproducibility
# ===========================================================================

def _long_seq() -> pd.DataFrame:
  """~200 weekdays with alternating wide/narrow sessions (non-trivial patterns)."""
  dates: list[str] = []
  d = pd.Timestamp("2020-01-01", tz=_NY)
  while len(dates) < 200:
    if d.weekday() < 5:
      dates.append(d.strftime("%Y-%m-%d"))
    d += pd.Timedelta(days=1)

  days = []
  base = 100.0
  for i, date in enumerate(dates):
    if i % 2 == 0:
      hi, lo = base + 30.0, base - 30.0
      open_, close = base - 10.0, base + 20.0  # green
    else:
      hi, lo = base + 8.0, base - 8.0
      open_, close = base + 3.0, base - 3.0    # red
    days.append({"date": date, "open": open_, "close": close, "high": hi, "low": lo})
    base += 0.5

  return make_candles(days)


def test_baseline_deterministic_same_seed() -> None:
  """Same seed produces identical baseline rows across two independent calls."""
  stat = _stat()
  df = _long_seq()
  rows_a = stat.baseline(df, seed=42)
  rows_b = stat.baseline(df, seed=42)
  assert len(rows_a) == len(rows_b)
  for a, b in zip(rows_a, rows_b):
    assert (a.condition, a.outcome) == (b.condition, b.outcome)
    assert a.probability == pytest.approx(b.probability)
    assert a.count == b.count
    assert a.total == b.total


def test_baseline_different_seeds_produce_different_results() -> None:
  """Different seeds produce at least some different probabilities."""
  stat = _stat()
  df = _long_seq()
  rows_42 = stat.baseline(df, seed=42)
  rows_99 = stat.baseline(df, seed=99)
  probs_42 = [r.probability for r in rows_42]
  probs_99 = [r.probability for r in rows_99]
  assert probs_42 != probs_99, "Two different seeds should yield different permutations"


def test_baseline_embedded_in_compute() -> None:
  """After compute(), every row with total > 0 carries a positive baseline_n."""
  result = _stat().compute(_long_seq())
  for row in result.instruments["NQ"]["daily"].results:
    if row.total > 0:
      assert row.baseline_n > 0, f"baseline_n=0 for {row.condition}/{row.outcome}"


def test_baseline_total_matches_compute_total() -> None:
  """Baseline countable_n equals the main computation's countable_n.

  Shuffling the prior triple (prev_high, prev_low, prev_session_green) moves the
  lone NaN row to a random position but preserves the count of NaN rows, so the
  number of countable days in the shuffled table equals countable_n in the real
  table. touch_0.total must be identical.
  """
  stat = _stat()
  df = _long_seq()
  day_table = stat.build_day_table(df)
  real_rows = stat.compute_rows(day_table)
  bl_rows = stat.baseline_rows(day_table, seed=7)

  real_total = next(r.total for r in real_rows if r.condition == "fib_levels" and r.outcome == "touch_0")
  bl_total = next(r.total for r in bl_rows if r.condition == "fib_levels" and r.outcome == "touch_0")
  assert real_total == bl_total


# ===========================================================================
# 6. Slices
# ===========================================================================

def test_declared_slices_present() -> None:
  """The stat declares weekday and prev_candle slices."""
  result = _stat().compute(make_candles([_PRIOR_GREEN]))
  slices = result.instruments["NQ"]["daily"].slices
  assert set(slices) == {"weekday", "prev_candle"}


def test_weekday_slice_touch_counts_sum_to_overall() -> None:
  """Sum of touch_0 counts across weekday groups equals the overall touch_0 count."""
  result = _stat().compute(_long_seq())
  wk = result.instruments["NQ"]["daily"].slices["weekday"]
  slice_total = sum(
    r.count
    for grp in wk.groups.values()
    for r in grp.results
    if r.condition == "fib_levels" and r.outcome == "touch_0"
  )
  overall = _row(result, "fib_levels", "touch_0").count
  assert slice_total == overall


def test_weekday_slice_zone_partition_holds_per_group() -> None:
  """Within each weekday group, opening_zone counts sum to that group's countable_n."""
  result = _stat().compute(_long_seq())
  wk = result.instruments["NQ"]["daily"].slices["weekday"]
  for day_key, grp in wk.groups.items():
    countable_n = next(
      (r.total for r in grp.results if r.condition == "fib_levels"), 0
    )
    zone_sum = sum(
      r.count for r in grp.results if r.condition == "opening_zone"
    )
    assert zone_sum == countable_n, (
      f"Weekday group '{day_key}': zone_sum={zone_sum} != countable_n={countable_n}"
    )


def test_prev_candle_slice_has_green_and_red_groups() -> None:
  """prev_candle slice exposes 'green' and/or 'red' groups."""
  result = _stat().compute(_long_seq())
  pc = result.instruments["NQ"]["daily"].slices["prev_candle"]
  assert set(pc.groups).issubset({"green", "red"})
  assert len(pc.groups) >= 1


def test_prev_candle_green_subset_matches_green_prior_orientation() -> None:
  """prev_candle 'green' group uses only days with a green prior candle.

  Build a sequence where Day 0 is GREEN and Day 1 is the only current session.
  The 'green' prev_candle slice must contain exactly 1 countable day;
  the 'red' slice must be empty (absent or 0 countable).
  """
  days = [
    _PRIOR_GREEN,  # GREEN prior
    {"date": _CURRENT_DATE, "open": 150.0, "close": 155.0, "high": 160.0, "low": 148.0},
  ]
  result = _stat().compute(make_candles(days))
  pc = result.instruments["NQ"]["daily"].slices["prev_candle"]
  # Only 'green' group should exist (1 countable day, all have green prior)
  assert "green" in pc.groups
  assert pc.groups["green"].total_samples == 1
  assert "red" not in pc.groups or pc.groups.get("red", None) is None


def test_prev_candle_slice_groups_sum_to_overall_count() -> None:
  """Sum of touch_0 counts across prev_candle groups equals the overall count."""
  result = _stat().compute(_long_seq())
  pc = result.instruments["NQ"]["daily"].slices["prev_candle"]
  slice_total = sum(
    r.count
    for grp in pc.groups.values()
    for r in grp.results
    if r.condition == "fib_levels" and r.outcome == "touch_0"
  )
  overall = _row(result, "fib_levels", "touch_0").count
  assert slice_total == overall


# ===========================================================================
# 7. Edge cases
# ===========================================================================

def test_empty_dataframe_no_crash() -> None:
  """Empty input → zero samples, empty data_range, all rows zeroed."""
  result = _stat().compute(_empty_df())
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 0
  assert tf.data_range == []
  for row in tf.results:
    assert row.count == 0
    assert row.total == 0
    assert row.probability == pytest.approx(0.0)


def test_single_resolved_day_no_countable_rows() -> None:
  """One resolved session → no prior day → all totals zero, no crash."""
  days = [{"date": "2024-03-01", "open": 100.0, "close": 200.0, "high": 200.0, "low": 100.0}]
  result = _stat().compute(make_candles(days))
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 1
  for row in tf.results:
    assert row.total == 0
    assert row.count == 0


def test_correct_number_of_result_rows() -> None:
  """Result contains exactly 7 touch rows + 8 zone rows = 15 rows total."""
  result = _stat().compute(_empty_df())
  tf = result.instruments["NQ"]["daily"]
  assert len(tf.results) == 15
  touch_rows = [r for r in tf.results if r.condition == "fib_levels"]
  zone_rows = [r for r in tf.results if r.condition == "opening_zone"]
  assert len(touch_rows) == 7
  assert len(zone_rows) == 8


def test_result_row_keys_complete() -> None:
  """All expected condition/outcome combinations are present."""
  result = _stat().compute(_empty_df())
  pairs = {(r.condition, r.outcome) for r in result.instruments["NQ"]["daily"].results}
  for key in ("touch_0", "touch_236", "touch_382", "touch_500", "touch_618", "touch_786", "touch_100"):
    assert ("fib_levels", key) in pairs, f"missing fib_levels/{key}"
  for key in _ALL_ZONE_KEYS:
    assert ("opening_zone", key) in pairs, f"missing opening_zone/{key}"


def test_data_range_spans_first_to_last_resolved_session() -> None:
  """data_range spans from the first to the last resolved session date."""
  days = [
    {"date": "2024-01-02", "open": 100.0, "close": 200.0, "high": 200.0, "low": 100.0},
    {"date": "2024-01-03", "open": 165.0, "close": 168.0, "high": 174.0, "low": 162.0},
    {"date": "2024-01-04", "open": 150.0, "close": 155.0, "high": 160.0, "low": 148.0},
  ]
  result = _stat().compute(make_candles(days))
  assert result.instruments["NQ"]["daily"].data_range == ["2024-01-02", "2024-01-04"]


# ===========================================================================
# 8. i18n metadata
# ===========================================================================

def test_stat_name() -> None:
  """stat_name is 'fibonacci_levels'."""
  assert _stat().compute(_empty_df()).stat_name == "fibonacci_levels"


def test_i18n_title_and_definition() -> None:
  """title and definition both have non-empty en and fr strings."""
  result = _stat().compute(_empty_df())
  assert result.title.en != "" and result.title.fr != ""
  assert result.definition.en != "" and result.definition.fr != ""


def test_i18n_labels_have_en_and_fr() -> None:
  """Every condition and outcome label has non-empty en and fr; correct key sets."""
  result = _stat().compute(_empty_df())
  for mapping in (result.labels.conditions, result.labels.outcomes):
    for key, i18n in mapping.items():
      assert i18n.en != "", f"{key}.en empty"
      assert i18n.fr != "", f"{key}.fr empty"
  assert set(result.labels.conditions) == {"fib_levels", "opening_zone"}
  expected_outcomes = {
    "touch_0", "touch_236", "touch_382", "touch_500", "touch_618", "touch_786", "touch_100",
    "below_0", "0_236", "236_382", "382_500", "500_618", "618_786", "786_100", "above_100",
  }
  assert set(result.labels.outcomes) == expected_outcomes


# ===========================================================================
# 9. write_results round-trip
# ===========================================================================

def test_write_results_round_trip(tmp_path: Path) -> None:
  """write_results produces fibonacci_levels.json that re-validates correctly."""
  days = [
    {"date": "2024-01-02", "open": 100.0, "close": 200.0, "high": 200.0, "low": 100.0},
    {"date": "2024-01-03", "open": 165.0, "close": 168.0, "high": 174.0, "low": 162.0},
  ]
  result = _stat().compute(make_candles(days))
  written = write_results(result, results_dir=tmp_path)
  assert written.name == "fibonacci_levels.json"

  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)
  assert validated.instruments["NQ"]["daily"].total_samples == 2


def test_write_results_french_accent_literal(tmp_path: Path) -> None:
  """French text is stored as literal UTF-8, not escaped unicode."""
  result = _stat().compute(_empty_df())
  written = write_results(result, results_dir=tmp_path)
  raw = written.read_text(encoding="utf-8")
  # The French definition contains "précédente" (with é).
  assert "é" in raw
  assert "\\u00e9" not in raw
