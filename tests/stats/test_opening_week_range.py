"""Tests for stats.opening_week_range.standard.

All data is synthetic — no real market files required. Expected values are
hand-calculated before each assertion.

Framing:
  - Tier 1 (condition ``opening_range``): a four-way partition of countable
    weeks against the opening week range — ``break_high_only``,
    ``break_low_only``, ``break_both``, ``inside``. The four counts sum to
    countable_n (weeks that have a non-empty breakout window).
  - Tier 2 (condition ``break_both``): among both-break weeks, which level was
    taken first — ``high_first`` / ``low_first`` (a partition of break_both).

Week boundaries are ISO weeks keyed by their Monday (tz-aware, normalized).
Opening window = the first ``open_days`` trading sessions of the week.
Breakout window = the remaining sessions of the same week (strictly after).
Break is strict inequality: touching the level is NOT a break.
The LAST week in the data is always dropped (pending discipline).
A week with no breakout window (≤ open_days sessions) is non-countable.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.opening_week_range.standard import OpeningWeekRange

# ---------------------------------------------------------------------------
# Minimal InstrumentConfig (does not depend on NQ.yaml)
# ---------------------------------------------------------------------------
_NY = "America/New_York"
_RTH_SESSION = Session(start="09:30", end="16:15")
_TEST_CONFIG = InstrumentConfig(
  instrument="NQ",
  working_timezone=_NY,
  sessions={"rth": _RTH_SESSION},
  timeframes=["1d", "2d"],
  parquet_path=Path("data/NQ_1min.parquet"),
)

# minutes-since-midnight constants
_RTH_START = 570   # 09:30
_RTH_LAST = 974    # 16:14 (last bar inside the 09:30–16:15 RTH window)


# ---------------------------------------------------------------------------
# Synthetic candle builders
# ---------------------------------------------------------------------------

def _make_day(
  date: str,
  session_open: float,
  session_close: float,
  day_high: float,
  day_low: float,
  high_at: int | None = None,
  low_at: int | None = None,
) -> pd.DataFrame:
  """One trading day of 1-min RTH bars (09:30–16:14).

  The opening bar carries ``session_open`` as its open; the last bar carries
  ``session_close`` as its close. ``day_high`` is placed on ``high_at`` (default
  mid-session) and ``day_low`` on ``low_at``; all other bars stay between the
  open/close extremes. Controlling ``high_at`` / ``low_at`` lets tests fix the
  intra-day order of level breaks for the sequence-tier assertions.
  """
  base = pd.Timestamp(date, tz=_NY)
  mid = (_RTH_START + _RTH_LAST) // 2
  if high_at is None:
    high_at = mid
  if low_at is None:
    low_at = mid
  records = []
  for mod in range(_RTH_START, _RTH_LAST + 1):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    o = session_open if mod == _RTH_START else session_close
    c = session_close
    hi = max(o, c)
    lo = min(o, c)
    if mod == high_at:
      hi = day_high
    if mod == low_at:
      lo = day_low
    records.append({
      "timestamp": ts, "open": o, "high": hi, "low": lo, "close": c, "volume": 1000,
    })
  return pd.DataFrame(records)


def make_candles(days: list[dict]) -> pd.DataFrame:
  """Concatenate per-day specs into a single sorted 1-min OHLCV DataFrame."""
  frames = [
    _make_day(
      d["date"], d["open"], d["close"], d["high"], d["low"],
      high_at=d.get("high_at"),
      low_at=d.get("low_at"),
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
# Stat factory helpers
# ---------------------------------------------------------------------------

def _stat(timeframe: str = "1d") -> OpeningWeekRange:
  return OpeningWeekRange(instrument="NQ", timeframe=timeframe, config=_TEST_CONFIG)


def _row(result: StatRunResult, condition: str, outcome: str, timeframe: str = "1d") -> StatResultRow:
  for r in result.instruments["NQ"][timeframe].results:
    if r.condition == condition and r.outcome == outcome:
      return r
  raise KeyError((condition, outcome))


def _tf(result: StatRunResult, timeframe: str = "1d"):
  return result.instruments["NQ"][timeframe]


# ===========================================================================
# 1. break_high_only classification
#
# Setup (timeframe "1d", open_days=1):
#   Week 1 (2024-01-01): Mon opening session: high=110, low=90 → opening range.
#     Breakout window (Tue–Fri): post_high=115 > 110, post_low=95 NOT < 90 → break_high_only.
#   Week 2 (2024-01-08): pending, dropped.
#
# Countable weeks: 1 (W1 only). break_high_only=1, all others=0.
# P(break_high_only) = 1/1 = 1.0.
# ===========================================================================

def _week_break_high_only() -> pd.DataFrame:
  """One resolved week: opening high broken, opening low held."""
  return make_candles([
    # W1 opening session (Mon): high=110, low=90
    {"date": "2024-01-01", "open": 100.0, "close": 101.0, "high": 110.0, "low": 90.0},
    # W1 breakout session (Tue): post_high=115>110 ✓, post_low=95 NOT <90 ✗
    {"date": "2024-01-02", "open": 100.0, "close": 101.0, "high": 115.0, "low": 95.0},
    # W1 breakout (Wed–Fri): neutral, well within range
    {"date": "2024-01-03", "open": 100.0, "close": 101.0, "high": 108.0, "low": 93.0},
    {"date": "2024-01-04", "open": 100.0, "close": 101.0, "high": 107.0, "low": 93.0},
    {"date": "2024-01-05", "open": 100.0, "close": 101.0, "high": 107.0, "low": 93.0},
    # W2 (pending): dropped
    {"date": "2024-01-08", "open": 100.0, "close": 101.0, "high": 105.0, "low": 95.0},
  ])


def test_break_high_only_count() -> None:
  """Week whose post-high exceeds opening high but post-low holds → break_high_only."""
  result = _stat("1d").compute(_week_break_high_only())
  # Hand-calc: countable_n=1, break_high_only=1, P=1.0
  r = _row(result, "opening_range", "break_high_only")
  assert r.count == 1
  assert r.total == 1
  assert r.probability == pytest.approx(1.0)


def test_break_high_only_other_outcomes_zero() -> None:
  """break_low_only, break_both, inside all zero when only high breaks."""
  result = _stat("1d").compute(_week_break_high_only())
  for outcome in ("break_low_only", "break_both", "inside"):
    assert _row(result, "opening_range", outcome).count == 0


# ===========================================================================
# 2. break_low_only classification
#
# Setup (timeframe "1d", open_days=1):
#   W1 opening (Mon): high=110, low=90.
#   Breakout (Tue+): post_high=105 NOT >110, post_low=85 < 90 → break_low_only.
#   W2: pending.
#
# Countable weeks: 1. break_low_only=1, P=1.0.
# ===========================================================================

def _week_break_low_only() -> pd.DataFrame:
  return make_candles([
    {"date": "2024-01-01", "open": 100.0, "close": 101.0, "high": 110.0, "low": 90.0},
    {"date": "2024-01-02", "open": 100.0, "close": 101.0, "high": 105.0, "low": 85.0},
    {"date": "2024-01-03", "open": 100.0, "close": 101.0, "high": 108.0, "low": 92.0},
    {"date": "2024-01-04", "open": 100.0, "close": 101.0, "high": 107.0, "low": 93.0},
    {"date": "2024-01-05", "open": 100.0, "close": 101.0, "high": 107.0, "low": 93.0},
    # W2 pending
    {"date": "2024-01-08", "open": 100.0, "close": 101.0, "high": 105.0, "low": 95.0},
  ])


def test_break_low_only_count() -> None:
  """Week whose post-low falls below opening low but post-high holds → break_low_only."""
  result = _stat("1d").compute(_week_break_low_only())
  # Hand-calc: countable_n=1, break_low_only=1, P=1.0
  r = _row(result, "opening_range", "break_low_only")
  assert r.count == 1
  assert r.total == 1
  assert r.probability == pytest.approx(1.0)


def test_break_low_only_other_outcomes_zero() -> None:
  """break_high_only, break_both, inside all zero when only low breaks."""
  result = _stat("1d").compute(_week_break_low_only())
  for outcome in ("break_high_only", "break_both", "inside"):
    assert _row(result, "opening_range", outcome).count == 0


# ===========================================================================
# 3. break_both classification
#
# Setup (timeframe "1d", open_days=1):
#   W1 opening (Mon): high=110, low=90.
#   Breakout: post_high=115>110 AND post_low=85<90 → break_both.
#   W2: pending.
#
# Countable weeks: 1. break_both=1, inside=0.
# ===========================================================================

def _week_break_both() -> pd.DataFrame:
  return make_candles([
    {"date": "2024-01-01", "open": 100.0, "close": 101.0, "high": 110.0, "low": 90.0},
    # Tue breaks both: post_high=115>110, post_low=85<90
    {"date": "2024-01-02", "open": 100.0, "close": 101.0, "high": 115.0, "low": 85.0},
    {"date": "2024-01-03", "open": 100.0, "close": 101.0, "high": 108.0, "low": 93.0},
    {"date": "2024-01-04", "open": 100.0, "close": 101.0, "high": 107.0, "low": 93.0},
    {"date": "2024-01-05", "open": 100.0, "close": 101.0, "high": 107.0, "low": 93.0},
    # W2 pending
    {"date": "2024-01-08", "open": 100.0, "close": 101.0, "high": 105.0, "low": 95.0},
  ])


def test_break_both_count() -> None:
  """Week breaking both opening high and opening low → break_both, inside=0."""
  result = _stat("1d").compute(_week_break_both())
  # Hand-calc: countable_n=1, break_both=1, inside=0
  r = _row(result, "opening_range", "break_both")
  assert r.count == 1
  assert r.total == 1
  assert r.probability == pytest.approx(1.0)
  assert _row(result, "opening_range", "inside").count == 0


# ===========================================================================
# 4. inside classification + strict-inequality boundary
#
# Setup (timeframe "1d", open_days=1):
#   W1 opening (Mon): high=110, low=90.
#   Case A — strictly inside: post_high=108<=110, post_low=92>=90 → inside.
#   Case B — exact touch: post_high=110 (==110, NOT >110), post_low=90 (==90, NOT <90) → inside.
#
# For Case A: countable_n=1, inside=1, P=1.0.
# For Case B: touching the level exactly is NOT a break → inside=1.
# ===========================================================================

def _week_inside() -> pd.DataFrame:
  """Breakout window stays strictly inside the opening range."""
  return make_candles([
    {"date": "2024-01-01", "open": 100.0, "close": 101.0, "high": 110.0, "low": 90.0},
    {"date": "2024-01-02", "open": 100.0, "close": 101.0, "high": 108.0, "low": 92.0},
    {"date": "2024-01-03", "open": 100.0, "close": 101.0, "high": 107.0, "low": 93.0},
    {"date": "2024-01-04", "open": 100.0, "close": 101.0, "high": 106.0, "low": 94.0},
    {"date": "2024-01-05", "open": 100.0, "close": 101.0, "high": 105.0, "low": 95.0},
    # W2 pending
    {"date": "2024-01-08", "open": 100.0, "close": 101.0, "high": 105.0, "low": 95.0},
  ])


def _week_exact_touch() -> pd.DataFrame:
  """Breakout window exactly touches opening high and low (==, not > or <) → inside."""
  return make_candles([
    {"date": "2024-01-01", "open": 100.0, "close": 101.0, "high": 110.0, "low": 90.0},
    # Tue: post_high==110 (NOT >110), post_low==90 (NOT <90) → inside
    {"date": "2024-01-02", "open": 100.0, "close": 101.0, "high": 110.0, "low": 90.0},
    {"date": "2024-01-03", "open": 100.0, "close": 101.0, "high": 105.0, "low": 95.0},
    {"date": "2024-01-04", "open": 100.0, "close": 101.0, "high": 105.0, "low": 95.0},
    {"date": "2024-01-05", "open": 100.0, "close": 101.0, "high": 105.0, "low": 95.0},
    # W2 pending
    {"date": "2024-01-08", "open": 100.0, "close": 101.0, "high": 104.0, "low": 96.0},
  ])


def test_inside_count() -> None:
  """Breakout staying strictly within the opening range → inside=1."""
  result = _stat("1d").compute(_week_inside())
  # Hand-calc: countable_n=1, inside=1, P=1.0
  r = _row(result, "opening_range", "inside")
  assert r.count == 1
  assert r.total == 1
  assert r.probability == pytest.approx(1.0)


def test_exact_touch_is_not_a_break() -> None:
  """post_high == opening_high and post_low == opening_low → inside (strict inequality)."""
  result = _stat("1d").compute(_week_exact_touch())
  # Touching 110 exactly (not > 110) and touching 90 exactly (not < 90) → inside
  assert _row(result, "opening_range", "inside").count == 1
  assert _row(result, "opening_range", "break_high_only").count == 0
  assert _row(result, "opening_range", "break_low_only").count == 0
  assert _row(result, "opening_range", "break_both").count == 0


# ===========================================================================
# 5. Partition invariant: four outcomes sum to total, probabilities sum to 1.0
#
# Multi-week fixture (timeframe "1d", open_days=1):
#   W1 (Jan 1–5):  opening Mon high=110, low=90; breakout: high=115>110, low=95 → break_high_only
#   W2 (Jan 8–12): opening Mon high=110, low=90; breakout: high=105, low=85<90 → break_low_only
#   W3 (Jan 15–19):opening Mon high=110, low=90; breakout: high=115, low=85 → break_both
#   W4 (Jan 22–26):opening Mon high=110, low=90; breakout: high=108, low=92 → inside
#   W5 (Jan 29): pending, dropped.
#
# Countable: 4 weeks. Partition: 1+1+1+1=4. Each P=0.25.
# ===========================================================================

_FOUR_WEEKS = [
  # W1: break_high_only (Mon opening; Tue break high only)
  {"date": "2024-01-01", "open": 100.0, "close": 101.0, "high": 110.0, "low": 90.0},
  {"date": "2024-01-02", "open": 100.0, "close": 101.0, "high": 115.0, "low": 95.0},
  {"date": "2024-01-03", "open": 100.0, "close": 101.0, "high": 108.0, "low": 93.0},
  {"date": "2024-01-04", "open": 100.0, "close": 101.0, "high": 107.0, "low": 93.0},
  {"date": "2024-01-05", "open": 100.0, "close": 101.0, "high": 107.0, "low": 93.0},
  # W2: break_low_only
  {"date": "2024-01-08", "open": 100.0, "close": 101.0, "high": 110.0, "low": 90.0},
  {"date": "2024-01-09", "open": 100.0, "close": 101.0, "high": 105.0, "low": 85.0},
  {"date": "2024-01-10", "open": 100.0, "close": 101.0, "high": 108.0, "low": 92.0},
  {"date": "2024-01-11", "open": 100.0, "close": 101.0, "high": 107.0, "low": 93.0},
  {"date": "2024-01-12", "open": 100.0, "close": 101.0, "high": 107.0, "low": 93.0},
  # W3: break_both (Jan 15–19)
  {"date": "2024-01-15", "open": 100.0, "close": 101.0, "high": 110.0, "low": 90.0},
  {"date": "2024-01-16", "open": 100.0, "close": 101.0, "high": 115.0, "low": 85.0},
  {"date": "2024-01-17", "open": 100.0, "close": 101.0, "high": 108.0, "low": 92.0},
  {"date": "2024-01-18", "open": 100.0, "close": 101.0, "high": 107.0, "low": 93.0},
  {"date": "2024-01-19", "open": 100.0, "close": 101.0, "high": 107.0, "low": 93.0},
  # W4: inside (Jan 22–26)
  {"date": "2024-01-22", "open": 100.0, "close": 101.0, "high": 110.0, "low": 90.0},
  {"date": "2024-01-23", "open": 100.0, "close": 101.0, "high": 108.0, "low": 92.0},
  {"date": "2024-01-24", "open": 100.0, "close": 101.0, "high": 107.0, "low": 93.0},
  {"date": "2024-01-25", "open": 100.0, "close": 101.0, "high": 106.0, "low": 94.0},
  {"date": "2024-01-26", "open": 100.0, "close": 101.0, "high": 105.0, "low": 95.0},
  # W5: pending (Jan 29)
  {"date": "2024-01-29", "open": 100.0, "close": 101.0, "high": 130.0, "low": 70.0},
]


def test_partition_counts_each_outcome_once() -> None:
  """Each of the four outcomes appears exactly once in a balanced four-week fixture."""
  result = _stat("1d").compute(make_candles(_FOUR_WEEKS))
  # Hand-calc: countable_n=4 (W1–W4 resolved, W5 pending)
  for outcome in ("break_high_only", "break_low_only", "break_both", "inside"):
    r = _row(result, "opening_range", outcome)
    assert r.count == 1, f"{outcome}: expected 1"
    assert r.total == 4, f"{outcome}: expected total 4"
    assert r.probability == pytest.approx(0.25), f"{outcome}: expected 0.25"


def test_partition_counts_sum_to_total() -> None:
  """Four outcome counts sum to the countable-week total."""
  result = _stat("1d").compute(make_candles(_FOUR_WEEKS))
  counts = sum(
    _row(result, "opening_range", o).count
    for o in ("break_high_only", "break_low_only", "break_both", "inside")
  )
  assert counts == 4


def test_partition_probabilities_sum_to_one() -> None:
  """Four outcome probabilities sum to 1.0 (partition over countable weeks)."""
  result = _stat("1d").compute(make_candles(_FOUR_WEEKS))
  probs = sum(
    _row(result, "opening_range", o).probability
    for o in ("break_high_only", "break_low_only", "break_both", "inside")
  )
  assert probs == pytest.approx(1.0)


def test_total_samples_equals_resolved_weeks() -> None:
  """total_samples = number of resolved weeks (excluding the pending last week)."""
  result = _stat("1d").compute(make_candles(_FOUR_WEEKS))
  # W1–W4 resolved, W5 pending → 4 resolved weeks in the week table
  assert _tf(result).total_samples == 4


# ===========================================================================
# 6. Sequence tier: high_first / low_first + same-bar tiebreak
#
# All scenarios use "1d" (open_days=1).
# Opening week range (Mon): high=110, low=90.
# Breakout session has controllable high_at / low_at to fix chronological order.
#
# Case A (high_at=600, low_at=700): high broken at mod 600, low at mod 700 → high_first.
# Case B (high_at=700, low_at=600): low broken first → low_first.
# Case C (same bar, bearish bar_open>bar_close): high printed first → high_first.
# Case D (same bar, bullish bar_close>bar_open): low printed first → low_first.
# ===========================================================================

def _both_break_week_1d(
  high_at: int,
  low_at: int,
  bar_open: float = 100.0,
  bar_close: float = 101.0,
) -> pd.DataFrame:
  """W1 with opening Mon (high=110, low=90) and Tue breaking both, with controllable timing.

  A pending W2 (Jan 8) is appended so the stat has a non-last resolved week.
  """
  return make_candles([
    # W1: Mon opening session
    {"date": "2024-01-01", "open": 100.0, "close": 101.0, "high": 110.0, "low": 90.0},
    # W1: Tue breakout — breaks both. high_at/low_at control which bar triggers each break.
    {
      "date": "2024-01-02",
      "open": bar_open,
      "close": bar_close,
      "high": 120.0,    # post_high=120 > 110 ✓
      "low": 80.0,      # post_low=80  < 90  ✓
      "high_at": high_at,
      "low_at": low_at,
    },
    {"date": "2024-01-03", "open": 100.0, "close": 101.0, "high": 108.0, "low": 93.0},
    {"date": "2024-01-04", "open": 100.0, "close": 101.0, "high": 107.0, "low": 93.0},
    {"date": "2024-01-05", "open": 100.0, "close": 101.0, "high": 107.0, "low": 93.0},
    # W2: pending (dropped)
    {"date": "2024-01-08", "open": 100.0, "close": 101.0, "high": 105.0, "low": 95.0},
  ])


def test_sequence_high_first() -> None:
  """Opening high broken before opening low (earlier bar) → high_first=1, low_first=0."""
  # high_at=600 (≈09:30+30), low_at=700 (≈09:30+∼2h): high is broken first
  result = _stat("1d").compute(_both_break_week_1d(high_at=600, low_at=700))
  # Hand-calc: both_n=1, high_first=1, low_first=0
  hf = _row(result, "break_both", "high_first")
  lf = _row(result, "break_both", "low_first")
  assert hf.count == 1
  assert hf.total == 1
  assert hf.probability == pytest.approx(1.0)
  assert lf.count == 0


def test_sequence_low_first() -> None:
  """Opening low broken before opening high → low_first=1, high_first=0."""
  # low_at=600, high_at=700: low is broken on an earlier bar
  result = _stat("1d").compute(_both_break_week_1d(high_at=700, low_at=600))
  hf = _row(result, "break_both", "high_first")
  lf = _row(result, "break_both", "low_first")
  assert lf.count == 1
  assert lf.total == 1
  assert lf.probability == pytest.approx(1.0)
  assert hf.count == 0


def test_sequence_same_bar_bearish_is_high_first() -> None:
  """Both broken on the same bar (close < open = bearish) → high printed first → high_first."""
  # Both high_at and low_at point to the opening bar (mod _RTH_START).
  # bar_open=101, bar_close=100 → bearish → high_first
  result = _stat("1d").compute(
    _both_break_week_1d(
      high_at=_RTH_START,
      low_at=_RTH_START,
      bar_open=101.0,
      bar_close=100.0,
    )
  )
  hf = _row(result, "break_both", "high_first")
  lf = _row(result, "break_both", "low_first")
  assert hf.count == 1
  assert lf.count == 0


def test_sequence_same_bar_bullish_is_low_first() -> None:
  """Both broken on the same bar (close > open = bullish) → low printed first → low_first."""
  # bar_open=100, bar_close=101 → bullish → low_first
  result = _stat("1d").compute(
    _both_break_week_1d(
      high_at=_RTH_START,
      low_at=_RTH_START,
      bar_open=100.0,
      bar_close=101.0,
    )
  )
  hf = _row(result, "break_both", "high_first")
  lf = _row(result, "break_both", "low_first")
  assert lf.count == 1
  assert hf.count == 0


def test_sequence_partitions_break_both() -> None:
  """high_first + low_first == break_both total (partition invariant for sequence tier)."""
  result = _stat("1d").compute(_both_break_week_1d(high_at=600, low_at=700))
  hf = _row(result, "break_both", "high_first")
  lf = _row(result, "break_both", "low_first")
  bb = _row(result, "opening_range", "break_both")
  assert hf.count + lf.count == hf.total == lf.total == bb.count


# ===========================================================================
# 7. open_days difference: same candles, timeframe "1d" vs "2d"
#
# Fixture (two resolved weeks, each with 5 sessions):
#   W1 (Jan 1–5):
#     Mon: high=110, low=90   (always part of opening window for both 1d and 2d)
#     Tue: high=115, low=95   (part of OPENING for 2d; part of BREAKOUT for 1d)
#     Wed–Fri: neutral inside [90, 110]
#   W2: pending.
#
# Under 1d (open_days=1):
#   Opening = Mon only: opening_high=110, opening_low=90.
#   Breakout = Tue–Fri: Tue's high=115 > 110 → break_high_only.
#
# Under 2d (open_days=2):
#   Opening = Mon+Tue: opening_high=max(110,115)=115, opening_low=min(90,95)=90.
#   Breakout = Wed–Fri: max high=108<115, min low=92>90 → inside.
#
# So "1d" → break_high_only; "2d" → inside. Different classifications.
# ===========================================================================

_TIMEFRAME_DIFF_WEEKS = [
  # W1: Mon opening
  {"date": "2024-01-01", "open": 100.0, "close": 101.0, "high": 110.0, "low": 90.0},
  # W1: Tue — high spikes above Mon's high
  {"date": "2024-01-02", "open": 100.0, "close": 101.0, "high": 115.0, "low": 95.0},
  # W1: Wed–Fri all inside [90, 110] (definitely inside [90, 115] too)
  {"date": "2024-01-03", "open": 100.0, "close": 101.0, "high": 108.0, "low": 92.0},
  {"date": "2024-01-04", "open": 100.0, "close": 101.0, "high": 107.0, "low": 93.0},
  {"date": "2024-01-05", "open": 100.0, "close": 101.0, "high": 106.0, "low": 94.0},
  # W2: pending
  {"date": "2024-01-08", "open": 100.0, "close": 101.0, "high": 105.0, "low": 95.0},
]


def test_timeframe_1d_vs_2d_different_classification() -> None:
  """Same candles: 1d classifies break_high_only while 2d classifies inside."""
  candles = make_candles(_TIMEFRAME_DIFF_WEEKS)

  # 1d: opening = Mon only (high=110, low=90); Tue spike at 115 → break_high_only
  result_1d = _stat("1d").compute(candles)
  assert _row(result_1d, "opening_range", "break_high_only").count == 1
  assert _row(result_1d, "opening_range", "inside").count == 0

  # 2d: opening = Mon+Tue (high=115, low=90); Wed–Fri: max=108<115 → inside
  result_2d = _stat("2d").compute(candles)
  assert _row(result_2d, "opening_range", "inside", "2d").count == 1
  assert _row(result_2d, "opening_range", "break_high_only", "2d").count == 0


def test_timeframe_2d_opening_range_aggregates_two_sessions() -> None:
  """Under 2d, opening_high = max(Mon, Tue) and opening_low = min(Mon, Tue)."""
  candles = make_candles(_TIMEFRAME_DIFF_WEEKS)
  stat = _stat("2d")
  week_table = stat.build_week_table(candles)
  # opening_high should be max(110, 115) = 115; opening_low = min(90, 95) = 90
  assert len(week_table) == 1
  assert float(week_table["opening_high"].iloc[0]) == pytest.approx(115.0)
  assert float(week_table["opening_low"].iloc[0]) == pytest.approx(90.0)
  assert float(week_table["opening_size"].iloc[0]) == pytest.approx(25.0)


# ===========================================================================
# 8. Pending discipline
#
# a) The last ISO week is always dropped, even if fully resolved.
# b) A short week (≤ open_days sessions, no breakout window) is non-countable:
#    its post_high/post_low are NaN and it does not appear in any outcome total.
# ===========================================================================

def test_last_week_always_dropped() -> None:
  """The final week in the data is always excluded (pending discipline)."""
  # W1 (Jan 1–5): fully resolved break_high_only.
  # W2 (Jan 8–12): would be break_both — but it's the LAST week and must be dropped.
  candles = make_candles([
    {"date": "2024-01-01", "open": 100.0, "close": 101.0, "high": 110.0, "low": 90.0},
    {"date": "2024-01-02", "open": 100.0, "close": 101.0, "high": 115.0, "low": 95.0},
    {"date": "2024-01-03", "open": 100.0, "close": 101.0, "high": 108.0, "low": 92.0},
    {"date": "2024-01-04", "open": 100.0, "close": 101.0, "high": 107.0, "low": 93.0},
    {"date": "2024-01-05", "open": 100.0, "close": 101.0, "high": 107.0, "low": 93.0},
    # W2 (last week, dropped): would break both (high 130 > 110, low 70 < 90)
    {"date": "2024-01-08", "open": 100.0, "close": 101.0, "high": 110.0, "low": 90.0},
    {"date": "2024-01-09", "open": 100.0, "close": 101.0, "high": 130.0, "low": 70.0},
    {"date": "2024-01-10", "open": 100.0, "close": 101.0, "high": 108.0, "low": 92.0},
    {"date": "2024-01-11", "open": 100.0, "close": 101.0, "high": 107.0, "low": 93.0},
    {"date": "2024-01-12", "open": 100.0, "close": 101.0, "high": 107.0, "low": 93.0},
  ])
  result = _stat("1d").compute(candles)
  # Only W1 resolved; W2 dropped. break_both must be 0.
  assert _row(result, "opening_range", "break_both").count == 0
  assert _row(result, "opening_range", "break_high_only").count == 1
  assert _tf(result).total_samples == 1


def test_short_week_is_non_countable_1d() -> None:
  """A week with only the opening session (no breakout window) has NaN post_* and is non-countable.

  Under 1d (open_days=1): a week that contains exactly 1 trading session has
  opening = that session, post_dates = {} → NaN post values → excluded from total.
  """
  # W1 (Jan 1): only Monday — no breakout window for 1d
  # W2 (Jan 8–12): normal full week — break_high_only
  # W3 (Jan 15): pending
  candles = make_candles([
    # W1: only Mon, no breakout sessions
    {"date": "2024-01-01", "open": 100.0, "close": 101.0, "high": 110.0, "low": 90.0},
    # W2: full week, break_high_only
    {"date": "2024-01-08", "open": 100.0, "close": 101.0, "high": 110.0, "low": 90.0},
    {"date": "2024-01-09", "open": 100.0, "close": 101.0, "high": 115.0, "low": 95.0},
    {"date": "2024-01-10", "open": 100.0, "close": 101.0, "high": 108.0, "low": 92.0},
    {"date": "2024-01-11", "open": 100.0, "close": 101.0, "high": 107.0, "low": 93.0},
    {"date": "2024-01-12", "open": 100.0, "close": 101.0, "high": 107.0, "low": 93.0},
    # W3: pending
    {"date": "2024-01-15", "open": 100.0, "close": 101.0, "high": 105.0, "low": 95.0},
  ])
  stat = _stat("1d")
  week_table = stat.build_week_table(candles)

  # W1 appears in the table (resolved) but with NaN post values → non-countable
  assert len(week_table) == 2  # W1 + W2 (W3 pending, dropped)
  w1_row = week_table.iloc[0]
  assert pd.isna(w1_row["post_high"])
  assert pd.isna(w1_row["post_low"])

  result = stat.compute(candles)
  # countable_n = 1 (only W2); W1 excluded from every outcome total
  for outcome in ("break_high_only", "break_low_only", "break_both", "inside"):
    assert _row(result, "opening_range", outcome).total == 1, outcome
  assert _row(result, "opening_range", "break_high_only").count == 1


def test_short_week_is_non_countable_2d() -> None:
  """Under 2d (open_days=2): a week with exactly 2 sessions has no breakout window → non-countable."""
  # W1 (Jan 1–2): Mon+Tue only, open_days=2 → no breakout sessions → NaN post
  # W2 (Jan 8–12): full 5-session week → countable
  # W3: pending
  candles = make_candles([
    {"date": "2024-01-01", "open": 100.0, "close": 101.0, "high": 110.0, "low": 90.0},
    {"date": "2024-01-02", "open": 100.0, "close": 101.0, "high": 115.0, "low": 95.0},
    # W2 full
    {"date": "2024-01-08", "open": 100.0, "close": 101.0, "high": 110.0, "low": 90.0},
    {"date": "2024-01-09", "open": 100.0, "close": 101.0, "high": 112.0, "low": 91.0},
    {"date": "2024-01-10", "open": 100.0, "close": 101.0, "high": 116.0, "low": 92.0},
    {"date": "2024-01-11", "open": 100.0, "close": 101.0, "high": 108.0, "low": 93.0},
    {"date": "2024-01-12", "open": 100.0, "close": 101.0, "high": 107.0, "low": 93.0},
    # W3: pending
    {"date": "2024-01-15", "open": 100.0, "close": 101.0, "high": 105.0, "low": 95.0},
  ])
  stat = _stat("2d")
  week_table = stat.build_week_table(candles)

  # W1 is in the table but has NaN post values
  w1_row = week_table.iloc[0]
  assert pd.isna(w1_row["post_high"])
  assert pd.isna(w1_row["post_low"])

  result = stat.compute(candles)
  # countable_n = 1 (only W2). W2 2d opening: Mon+Tue high=max(110,112)=112, low=min(90,91)=90
  # Breakout (Wed–Fri): max_high=max(116,108,107)=116 > 112 → break_high_only
  assert _row(result, "opening_range", "break_high_only", "2d").total == 1
  assert _row(result, "opening_range", "break_high_only", "2d").count == 1


# ===========================================================================
# 9. Baseline determinism and symmetry
#
# For a fixture where EVERY resolved week is break_high_only:
#   - baseline_rows(table, seed=42) is reproducible (same result every call).
#   - Reflection: break_high_only ↔ break_low_only swap; break_both & inside preserved.
#     So: baseline break_high_only < observed break_high_only (some flip to low_only).
#         baseline break_low_only > 0 (some high_only weeks become low_only).
#         baseline break_both == observed break_both (both=0 stays 0).
#         baseline inside == observed inside (inside=0 stays 0).
#   - Sequence-tier total (both_n) is preserved in the baseline.
# ===========================================================================

def _all_high_only_weeks(n: int) -> pd.DataFrame:
  """n resolved weeks, each break_high_only, plus 1 pending week.

  Opening (Mon): high=110, low=90.
  Breakout (Tue): post_high=115>110, post_low=95 (NOT <90) → break_high_only.
  Days Wed–Fri are neutral. The last week is a pending stub.
  """
  days = []
  # Use Jan 2020 onwards; 2020-01-06 is a Monday.
  base = pd.Timestamp("2020-01-06", tz=_NY)
  for i in range(n + 1):  # +1 for the pending last week
    monday = base + pd.Timedelta(weeks=i)
    tue = monday + pd.Timedelta(days=1)
    wed = monday + pd.Timedelta(days=2)
    thu = monday + pd.Timedelta(days=3)
    fri = monday + pd.Timedelta(days=4)
    is_last = i == n
    days += [
      {
        "date": monday.strftime("%Y-%m-%d"),
        "open": 100.0, "close": 101.0, "high": 110.0, "low": 90.0,
      },
    ]
    if not is_last:
      days += [
        {
          "date": tue.strftime("%Y-%m-%d"),
          "open": 100.0, "close": 101.0, "high": 115.0, "low": 95.0,
        },
        {
          "date": wed.strftime("%Y-%m-%d"),
          "open": 100.0, "close": 101.0, "high": 108.0, "low": 93.0,
        },
        {
          "date": thu.strftime("%Y-%m-%d"),
          "open": 100.0, "close": 101.0, "high": 107.0, "low": 93.0,
        },
        {
          "date": fri.strftime("%Y-%m-%d"),
          "open": 100.0, "close": 101.0, "high": 106.0, "low": 94.0,
        },
      ]
  return make_candles(days)


def test_baseline_deterministic_same_seed() -> None:
  """Same seed produces identical baseline rows across two calls."""
  stat = _stat("1d")
  candles = _all_high_only_weeks(20)
  week_table = stat.build_week_table(candles)
  rows_a = stat.baseline_rows(week_table, seed=42)
  rows_b = stat.baseline_rows(week_table, seed=42)
  for a, b in zip(rows_a, rows_b):
    assert (a.condition, a.outcome) == (b.condition, b.outcome)
    assert a.count == b.count
    assert a.probability == pytest.approx(b.probability)


def test_baseline_different_seed_gives_different_result() -> None:
  """Different seeds produce different baseline rows on a large enough fixture."""
  stat = _stat("1d")
  candles = _all_high_only_weeks(40)
  week_table = stat.build_week_table(candles)
  rows_42 = {(r.condition, r.outcome): r.count for r in stat.baseline_rows(week_table, seed=42)}
  rows_99 = {(r.condition, r.outcome): r.count for r in stat.baseline_rows(week_table, seed=99)}
  # With 40 weeks the chance of identical flip vectors is negligible.
  assert rows_42 != rows_99


def test_baseline_reflection_symmetry_all_high_only() -> None:
  """On an all-break_high_only fixture the reflection baseline moves mass to break_low_only.

  Reflection maps each week's post extremes around the opening midpoint with
  probability 0.5. For a break_high_only week, the reflected version becomes
  break_low_only. So:
    baseline break_high_only < observed break_high_only (n weeks)
    baseline break_low_only  > 0  (some flipped)
    baseline break_both      == 0 (both=0 stays 0 — symmetric outcome)
    baseline inside          == 0 (inside=0 stays 0 — symmetric outcome)
  """
  n = 20
  stat = _stat("1d")
  candles = _all_high_only_weeks(n)
  week_table = stat.build_week_table(candles)
  bl = {(r.condition, r.outcome): r for r in stat.baseline_rows(week_table, seed=42)}
  actual = {(r.condition, r.outcome): r for r in stat.compute_rows(week_table)}

  # Observed: all n weeks are break_high_only
  assert actual[("opening_range", "break_high_only")].count == n

  # Baseline: fewer high_only (some reflected to low_only)
  assert bl[("opening_range", "break_high_only")].count < n
  # Baseline: some low_only now (> 0)
  assert bl[("opening_range", "break_low_only")].count > 0
  # break_both/inside observed=0, and symmetric under reflection → still 0
  assert bl[("opening_range", "break_both")].count == actual[("opening_range", "break_both")].count
  assert bl[("opening_range", "inside")].count == actual[("opening_range", "inside")].count


def test_baseline_both_direction_mass_conserved() -> None:
  """Reflection swaps high_only ↔ low_only; their combined count is invariant."""
  n = 20
  stat = _stat("1d")
  candles = _all_high_only_weeks(n)
  week_table = stat.build_week_table(candles)
  bl = {(r.condition, r.outcome): r for r in stat.baseline_rows(week_table, seed=42)}
  actual = {(r.condition, r.outcome): r for r in stat.compute_rows(week_table)}

  actual_directional = (
    actual[("opening_range", "break_high_only")].count
    + actual[("opening_range", "break_low_only")].count
  )
  bl_directional = (
    bl[("opening_range", "break_high_only")].count
    + bl[("opening_range", "break_low_only")].count
  )
  assert actual_directional == bl_directional


def test_baseline_sequence_total_preserved() -> None:
  """Baseline sequence tier: both_n (total) equals observed both_n.

  The sequence baseline only re-randomizes seq_high_first while keeping the
  opening ranges (and thus the break_both set) unchanged.
  """
  # Build a fixture with both-break weeks.
  candles = make_candles(_FOUR_WEEKS)
  stat = _stat("1d")
  week_table = stat.build_week_table(candles)
  bl_map = {(r.condition, r.outcome): r for r in stat.baseline_rows(week_table, seed=42)}
  actual_map = {(r.condition, r.outcome): r for r in stat.compute_rows(week_table)}

  # break_both total must be identical in baseline and observed
  bl_bb_total = bl_map[("break_both", "high_first")].total
  actual_bb_total = actual_map[("break_both", "high_first")].total
  assert bl_bb_total == actual_bb_total


def test_baseline_embedded_in_compute_rows() -> None:
  """After compute(), every row with total>0 carries a positive baseline_n."""
  result = _stat("1d").compute(make_candles(_FOUR_WEEKS))
  for r in _tf(result).results:
    if r.total > 0:
      assert r.baseline_n > 0, f"{r.condition}/{r.outcome}"


# ===========================================================================
# 10. Edge cases: empty DataFrame, single week, unsupported timeframe
# ===========================================================================

def test_empty_dataframe_build_week_table_returns_empty() -> None:
  """Empty input → build_week_table returns an empty DataFrame with the declared columns."""
  stat = _stat("1d")
  table = stat.build_week_table(_empty_df())
  assert table.empty
  expected_cols = {
    "opening_high", "opening_low", "opening_size", "post_high", "post_low", "seq_high_first"
  }
  assert set(table.columns) == expected_cols


def test_empty_dataframe_compute_returns_zero_rows() -> None:
  """Empty input → compute_rows returns the six zero rows with total=0, probability=0.0."""
  stat = _stat("1d")
  rows = stat.compute_rows(stat.build_week_table(_empty_df()))
  expected_keys = {
    ("opening_range", "break_high_only"),
    ("opening_range", "break_low_only"),
    ("opening_range", "break_both"),
    ("opening_range", "inside"),
    ("break_both", "high_first"),
    ("break_both", "low_first"),
  }
  assert {(r.condition, r.outcome) for r in rows} == expected_keys
  for r in rows:
    assert r.count == 0
    assert r.total == 0
    assert r.probability == pytest.approx(0.0)


def test_empty_dataframe_full_compute() -> None:
  """Empty input → full compute() pipeline returns zeroed TimeframeResult."""
  result = _stat("1d").compute(_empty_df())
  tf = _tf(result)
  assert tf.total_samples == 0
  assert tf.data_range == []
  for r in tf.results:
    assert r.count == 0
    assert r.total == 0
    assert r.probability == pytest.approx(0.0)


def test_single_week_is_pending() -> None:
  """A single week in the data is the last (pending) week → zero resolved weeks."""
  candles = make_candles([
    {"date": "2024-01-01", "open": 100.0, "close": 101.0, "high": 110.0, "low": 90.0},
    {"date": "2024-01-02", "open": 100.0, "close": 101.0, "high": 115.0, "low": 95.0},
  ])
  result = _stat("1d").compute(candles)
  assert _tf(result).total_samples == 0


def test_two_weeks_only_one_resolved_and_non_countable() -> None:
  """Two weeks: one resolved (but short, no breakout window for 1d) and one pending.

  W1 (Jan 1): only Monday → opening only, no breakout → non-countable.
  W2 (Jan 8): pending → dropped.
  Result: total_samples=1 (one row in week_table), but countable_n=0.
  """
  candles = make_candles([
    {"date": "2024-01-01", "open": 100.0, "close": 101.0, "high": 110.0, "low": 90.0},
    {"date": "2024-01-08", "open": 100.0, "close": 101.0, "high": 115.0, "low": 95.0},
  ])
  result = _stat("1d").compute(candles)
  tf = _tf(result)
  assert tf.total_samples == 1   # 1 row in the week table (W1 resolved)
  for r in tf.results:
    assert r.total == 0  # but no countable weeks → zero denominators


def test_unsupported_timeframe_raises_value_error() -> None:
  """Passing an unsupported timeframe to __init__ raises ValueError."""
  with pytest.raises(ValueError, match="Unsupported timeframe"):
    OpeningWeekRange(instrument="NQ", timeframe="3d", config=_TEST_CONFIG)


def test_unsupported_timeframe_weekly_raises() -> None:
  """'weekly' is not a valid timeframe for OpeningWeekRange."""
  with pytest.raises(ValueError, match="Unsupported timeframe"):
    OpeningWeekRange(instrument="NQ", timeframe="weekly", config=_TEST_CONFIG)


# ===========================================================================
# 11. End-to-end: compute() on a small fixture returns a valid StatRunResult
# ===========================================================================

def test_end_to_end_stat_run_result_structure() -> None:
  """compute() on a multi-week fixture returns a StatRunResult with the size slice."""
  result = _stat("1d").compute(make_candles(_FOUR_WEEKS))
  assert result.stat_name == "opening_week_range"
  assert result.title.en != ""
  assert result.definition.en != ""
  # Timeframe key must be "1d"
  assert "1d" in result.instruments["NQ"]
  tf = result.instruments["NQ"]["1d"]
  # 4 resolved weeks (W1–W4), W5 pending
  assert tf.total_samples == 4
  # 6 rows total (4 partition + 2 sequence)
  assert len(tf.results) == 6
  # "size" slice must be present (declared in slices = (SizeBucket(..., name="size"),))
  assert "size" in tf.slices


def test_end_to_end_total_samples_correct() -> None:
  """total_samples equals the number of rows in the week table (resolved weeks)."""
  candles = make_candles(_FOUR_WEEKS)
  stat = _stat("1d")
  week_table = stat.build_week_table(candles)
  result = stat.compute(candles)
  assert _tf(result).total_samples == len(week_table)


def test_end_to_end_write_results_round_trip(tmp_path: Path) -> None:
  """write_results produces opening_week_range.json that re-validates correctly."""
  result = _stat("1d").compute(make_candles(_FOUR_WEEKS))
  written = write_results(result, results_dir=tmp_path)
  assert written.name == "opening_week_range.json"

  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)
  assert validated.instruments["NQ"]["1d"].total_samples == 4


def test_six_rows_exactly() -> None:
  """Exactly the four partition rows plus the two sequence rows."""
  rows = _stat("1d").compute(make_candles(_FOUR_WEEKS)).instruments["NQ"]["1d"].results
  assert {(r.condition, r.outcome) for r in rows} == {
    ("opening_range", "break_high_only"),
    ("opening_range", "break_low_only"),
    ("opening_range", "break_both"),
    ("opening_range", "inside"),
    ("break_both", "high_first"),
    ("break_both", "low_first"),
  }


# ===========================================================================
# 12. Data range and week-table indexing
# ===========================================================================

def test_data_range_spans_first_to_last_resolved_week() -> None:
  """data_range reflects the Monday dates of first and last resolved weeks."""
  result = _stat("1d").compute(make_candles(_FOUR_WEEKS))
  # Resolved weeks: W1 (Jan 1) through W4 (Jan 22). W5 (Jan 29) is pending.
  assert _tf(result).data_range == ["2024-01-01", "2024-01-22"]


def test_week_table_indexed_by_monday() -> None:
  """build_week_table index entries are the ISO Monday of each resolved week."""
  candles = make_candles(_FOUR_WEEKS)
  week_table = _stat("1d").build_week_table(candles)
  # All index entries must be Mondays (dayofweek == 0)
  assert all(idx.dayofweek == 0 for idx in week_table.index)


def test_build_day_table_delegates_to_build_week_table() -> None:
  """build_day_table and build_week_table produce identical results."""
  candles = make_candles(_FOUR_WEEKS)
  stat = _stat("1d")
  from_day = stat.build_day_table(candles)
  from_week = stat.build_week_table(candles)
  pd.testing.assert_frame_equal(from_day, from_week)


# ===========================================================================
# 13. i18n
# ===========================================================================

def test_i18n_title_and_definition_non_empty() -> None:
  """title and definition both have non-empty en and fr strings."""
  result = _stat("1d").compute(_empty_df())
  assert result.title.en != ""
  assert result.title.fr != ""
  assert result.definition.en != ""
  assert result.definition.fr != ""


def test_i18n_labels_correct_keys() -> None:
  """Labels carry the expected condition and outcome keys with non-empty i18n strings."""
  result = _stat("1d").compute(_empty_df())
  assert set(result.labels.conditions) == {"opening_range", "break_both"}
  assert set(result.labels.outcomes) == {
    "break_high_only", "break_low_only", "break_both", "inside",
    "high_first", "low_first",
  }
  for key, label in {**result.labels.conditions, **result.labels.outcomes}.items():
    assert label.en != "", f"{key}.en empty"
    assert label.fr != "", f"{key}.fr empty"


# ===========================================================================
# 14. classify_samples()
#
# Reusing _FOUR_WEEKS (1d, 4 countable weeks, one per outcome; W5 pending):
#   W1 (2024-01-01): break_high_only
#   W2 (2024-01-08): break_low_only
#   W3 (2024-01-15): break_both, seq_high_first=False -> low_first
#   W4 (2024-01-22): inside
# W3 emits TWO SampleRows (opening_range + break_both); the others emit ONE.
# ===========================================================================

def test_classify_samples_exact_list() -> None:
  """classify_samples emits opening_range samples for every countable week, plus a
  break_both sequence sample for the one both-break week."""
  stat = _stat("1d")
  day_table = stat.build_day_table(make_candles(_FOUR_WEEKS))
  samples = stat.classify_samples(day_table)
  assert [(s.date, s.condition, s.outcome) for s in samples] == [
    ("2024-01-01", "opening_range", "break_high_only"),
    ("2024-01-08", "opening_range", "break_low_only"),
    ("2024-01-15", "opening_range", "break_both"),
    ("2024-01-15", "break_both", "low_first"),
    ("2024-01-22", "opening_range", "inside"),
  ]
  assert all(s.value is None for s in samples)


def test_classify_samples_matches_compute_rows_counts() -> None:
  """For every StatResultRow, the matching SampleRow count equals r.count and total."""
  stat = _stat("1d")
  day_table = stat.build_day_table(make_candles(_FOUR_WEEKS))
  samples = stat.classify_samples(day_table)
  rows = stat.compute_rows(day_table)
  for r in rows:
    n = sum(1 for s in samples if s.condition == r.condition and s.outcome == r.outcome)
    assert n == r.count
  opening_range_total = sum(1 for s in samples if s.condition == "opening_range")
  assert opening_range_total == 4
  break_both_total = sum(1 for s in samples if s.condition == "break_both")
  assert break_both_total == 1


def test_classify_samples_empty_day_table() -> None:
  """Empty day_table -> classify_samples returns []."""
  stat = _stat("1d")
  assert stat.classify_samples(stat.build_day_table(_empty_df())) == []


def test_classify_samples_excludes_pending_week() -> None:
  """The trailing pending week (already dropped by build_day_table) yields no samples."""
  stat = _stat("1d")
  day_table = stat.build_day_table(make_candles(_FOUR_WEEKS))
  samples = stat.classify_samples(day_table)
  sample_dates = {s.date for s in samples}
  assert "2024-01-29" not in sample_dates
  assert sample_dates == {"2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22"}
