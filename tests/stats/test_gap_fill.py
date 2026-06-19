"""Tests for stats.gap_fill.standard.

All data is synthetic — no real market files required.
Expected values are hand-calculated before each assertion.

Matrix framing:
  condition = gap direction (gap_up: session_open > prev_session_close;
              gap_down: session_open < prev_session_close)
  outcome   = whether the gap filled to fill_threshold_pct

The first resolved day has no prior close and is excluded from every
denominator (pending-sample discipline). Days where open == prev_close
(zero-gap) are also excluded from all denominators and condition totals.
total_samples counts ALL resolved days, regardless of gap status.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.gap_fill.standard import GapFill

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
_RTH_LAST = 974    # 16:14 (last bar before 16:15); resolved needs last_mod >= 960


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

def _make_day(
  date: str,
  session_open: float,
  session_close: float,
  day_high: float | None = None,
  day_low: float | None = None,
) -> pd.DataFrame:
  """Build one full RTH trading day of 1-min OHLCV bars (09:30–16:14).

  The 09:30 bar sets session_open as its open price and records day_high as
  that bar's high (so day_high is always the maximum of all bar highs).
  The 16:14 bar carries session_close as its close and records day_low as its
  low (so day_low is always the minimum of all bar lows).
  All intermediate bars sit inside (day_low, day_high) using a neutral mid-
  price, so they never violate the intended extremes.

  When day_high / day_low are omitted, they default to:
    day_high = max(session_open, session_close) + 0.25
    day_low  = min(session_open, session_close) - 0.25

  The OHLC pre-conditions (day_high >= max(open, close) and
  day_low <= min(open, close)) are enforced by assertion below, so every
  generated bar keeps open and close within [low, high].
  """
  if day_high is None:
    day_high = max(session_open, session_close) + 0.25
  if day_low is None:
    day_low = min(session_open, session_close) - 0.25

  assert day_high >= max(session_open, session_close), (
    f"day_high ({day_high}) < max(open, close) ({max(session_open, session_close)})"
  )
  assert day_low <= min(session_open, session_close), (
    f"day_low ({day_low}) > min(open, close) ({min(session_open, session_close)})"
  )

  base = pd.Timestamp(date, tz=_NY)
  mid = (day_high + day_low) / 2.0
  neutral_high = mid + 0.10
  neutral_low = mid - 0.10

  records = []
  for mod in range(_RTH_START, _RTH_LAST + 1):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    is_first = mod == _RTH_START
    is_last = mod == _RTH_LAST
    o = session_open if is_first else mid
    c = session_close if is_last else mid
    bar_high = day_high if is_first else (max(neutral_high, c) if is_last else neutral_high)
    bar_low = (min(neutral_low, o) if is_first else (day_low if is_last else neutral_low))
    records.append({
      "timestamp": ts,
      "open": o,
      "high": bar_high,
      "low": bar_low,
      "close": c,
      "volume": 1000,
    })
  return pd.DataFrame(records)


def make_candles(days: list[dict]) -> pd.DataFrame:
  """Concatenate per-day specs into a single sorted 1-min OHLCV DataFrame."""
  frames = [
    _make_day(
      d["date"],
      d["session_open"],
      d["session_close"],
      d.get("day_high"),
      d.get("day_low"),
    )
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
      "timestamp": ts,
      "open": 100.0,
      "high": 100.25,
      "low": 99.75,
      "close": 100.0,
      "volume": 500,
    })
  return pd.DataFrame(records)


def _empty_df() -> pd.DataFrame:
  df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
  df["timestamp"] = pd.array([], dtype="datetime64[ns, America/New_York]")
  return df


def _stat(fill_threshold_pct: float = 100.0) -> GapFill:
  return GapFill(
    instrument="NQ",
    config=_TEST_CONFIG,
    fill_threshold_pct=fill_threshold_pct,
  )


def _row(result: StatRunResult, condition: str, outcome: str) -> StatResultRow:
  for r in result.instruments["NQ"]["daily"].results:
    if r.condition == condition and r.outcome == outcome:
      return r
  raise KeyError((condition, outcome))


# ===========================================================================
# 1. Core matrix
#
# Day-by-day derivation (pct=100, full fill back to prev_close):
#
#   idx  date        open    close   prev_close  gap        gap_size  gap_up
#    0   2024-01-02  100.00  105.00  (none)      —          —         —  ← excluded (no prior)
#    1   2024-01-03  110.00   95.00  105.00      +5         5.00      T
#    2   2024-01-04   98.00  102.00   95.00      +3         3.00      T
#    3   2024-01-05   88.00   92.00  102.00      -14        14.00     F
#    4   2024-01-08  105.00   98.00   92.00      +13        13.00     T
#    5   2024-01-09   80.00   85.00   98.00      -18        18.00     F
#    6   2024-01-10   90.00  100.00   85.00      +5         5.00      T
#    7   2024-01-11   78.00   72.00  100.00      -22        22.00     F
#    8   2024-01-12   85.00   80.00   72.00      +13        13.00     T
#
# Fill detection (pct=100 → target = open ± 1.0 * gap_size = prev_close):
#
#   idx  gap_up  prev_close  target           day_low / day_high         filled?
#    1   T       105.00      open-5=105.00    day_low= 94.50  ≤ 105.00   YES
#    2   T        95.00      open-3= 95.00    day_low= 96.00  > 95.00    NO
#    3   F       102.00      open+14=102.00   day_high=103.00 ≥ 102.00   YES
#    4   T        92.00      open-13= 92.00   day_low= 91.50  ≤ 92.00    YES
#    5   F        98.00      open+18= 98.00   day_high= 97.00 < 98.00    NO
#    6   T        85.00      open-5 = 85.00   day_low= 84.50  ≤ 85.00    YES
#    7   F       100.00      open+22=100.00   day_high=100.50 ≥ 100.00   YES
#    8   T        72.00      open-13= 72.00   day_low= 73.00  > 72.00    NO
#
# Matrix:
#   gap_up   days: 1,2,4,6,8  → total=5
#     filled:     1,4,6      → count=3, P=3/5=0.6
#     not_filled: 2,8        → count=2, P=2/5=0.4
#
#   gap_down days: 3,5,7     → total=3
#     filled:     3,7        → count=2, P=2/3≈0.6667
#     not_filled: 5          → count=1, P=1/3≈0.3333
#
# total_samples = 9 (all 9 resolved days, including day 0 which has no prior close)
# ===========================================================================

_SEQ = [
  # idx 0: anchor, excluded from denominators (no prior close)
  {
    "date": "2024-01-02",
    "session_open": 100.00,
    "session_close": 105.00,
    # default high/low: 105.25 / 99.75
  },
  # idx 1: gap_up (+5), target=105.00, day_low=94.50 ≤ 105.00 → FILLED
  {
    "date": "2024-01-03",
    "session_open": 110.00,
    "session_close": 95.00,
    "day_high": 111.00,
    "day_low": 94.50,
  },
  # idx 2: gap_up (+3), target=95.00, day_low=96.00 > 95.00 → NOT FILLED
  {
    "date": "2024-01-04",
    "session_open": 98.00,
    "session_close": 102.00,
    "day_high": 103.00,
    "day_low": 96.00,
  },
  # idx 3: gap_down (-14), target=102.00, day_high=103.00 ≥ 102.00 → FILLED
  {
    "date": "2024-01-05",
    "session_open": 88.00,
    "session_close": 92.00,
    "day_high": 103.00,
    "day_low": 87.00,
  },
  # idx 4: gap_up (+13), target=92.00, day_low=91.50 ≤ 92.00 → FILLED
  {
    "date": "2024-01-08",
    "session_open": 105.00,
    "session_close": 98.00,
    "day_high": 106.00,
    "day_low": 91.50,
  },
  # idx 5: gap_down (-18), target=98.00, day_high=97.00 < 98.00 → NOT FILLED
  {
    "date": "2024-01-09",
    "session_open": 80.00,
    "session_close": 85.00,
    "day_high": 97.00,
    "day_low": 79.00,
  },
  # idx 6: gap_up (+5), target=85.00, day_low=84.50 ≤ 85.00 → FILLED
  {
    "date": "2024-01-10",
    "session_open": 90.00,
    "session_close": 100.00,
    "day_high": 101.00,
    "day_low": 84.50,
  },
  # idx 7: gap_down (-22), target=100.00, day_high=100.50 ≥ 100.00 → FILLED
  {
    "date": "2024-01-11",
    "session_open": 78.00,
    "session_close": 72.00,
    "day_high": 100.50,
    "day_low": 71.00,
  },
  # idx 8: gap_up (+13), target=72.00, day_low=73.00 > 72.00 → NOT FILLED
  {
    "date": "2024-01-12",
    "session_open": 85.00,
    "session_close": 80.00,
    "day_high": 86.00,
    "day_low": 73.00,
  },
]


def test_total_samples_counts_all_resolved_days() -> None:
  """total_samples = 9: all 9 resolved sessions, including the first (no prior close)."""
  result = _stat().compute(make_candles(_SEQ))
  assert result.instruments["NQ"]["daily"].total_samples == 9


def test_gap_up_filled_count_and_probability() -> None:
  """gap_up filled: 3 of 5 gap-up days → P=3/5=0.6."""
  # gap_up days: idx 1(filled),2(not),4(filled),6(filled),8(not) → total=5, filled=3
  result = _stat().compute(make_candles(_SEQ))
  r = _row(result, "gap_up", "filled")
  assert r.total == 5
  assert r.count == 3
  assert r.probability == pytest.approx(3 / 5)


def test_gap_up_not_filled_count_and_probability() -> None:
  """gap_up not_filled: 2 of 5 gap-up days → P=2/5=0.4."""
  result = _stat().compute(make_candles(_SEQ))
  r = _row(result, "gap_up", "not_filled")
  assert r.total == 5
  assert r.count == 2
  assert r.probability == pytest.approx(2 / 5)


def test_gap_down_filled_count_and_probability() -> None:
  """gap_down filled: 2 of 3 gap-down days → P=2/3≈0.6667."""
  # gap_down days: idx 3(filled),5(not),7(filled) → total=3, filled=2
  result = _stat().compute(make_candles(_SEQ))
  r = _row(result, "gap_down", "filled")
  assert r.total == 3
  assert r.count == 2
  assert r.probability == pytest.approx(2 / 3)


def test_gap_down_not_filled_count_and_probability() -> None:
  """gap_down not_filled: 1 of 3 gap-down days → P=1/3≈0.3333."""
  result = _stat().compute(make_candles(_SEQ))
  r = _row(result, "gap_down", "not_filled")
  assert r.total == 3
  assert r.count == 1
  assert r.probability == pytest.approx(1 / 3)


def test_outcomes_partition_each_condition() -> None:
  """filled.count + not_filled.count == total for each gap direction."""
  result = _stat().compute(make_candles(_SEQ))
  for cond in ("gap_up", "gap_down"):
    filled = _row(result, cond, "filled")
    not_filled = _row(result, cond, "not_filled")
    assert filled.count + not_filled.count == filled.total == not_filled.total


def test_four_rows_only() -> None:
  """Exactly four rows: the 2x2 (gap direction × filled/not_filled) matrix."""
  result = _stat().compute(make_candles(_SEQ))
  rows = result.instruments["NQ"]["daily"].results
  assert {(r.condition, r.outcome) for r in rows} == {
    ("gap_up", "filled"),
    ("gap_up", "not_filled"),
    ("gap_down", "filled"),
    ("gap_down", "not_filled"),
  }


def test_first_day_excluded_from_denominators() -> None:
  """Sum of all condition totals = 8 = total_samples - 1 (first day has no prior close)."""
  # 9 resolved days; day 0 excluded → 8 countable (5 gap_up + 3 gap_down)
  result = _stat().compute(make_candles(_SEQ))
  gap_up_total = _row(result, "gap_up", "filled").total
  gap_down_total = _row(result, "gap_down", "filled").total
  assert gap_up_total + gap_down_total == 9 - 1


def test_data_range_spans_first_to_last_resolved_day() -> None:
  """data_range reflects the first and last resolved session dates."""
  result = _stat().compute(make_candles(_SEQ))
  assert result.instruments["NQ"]["daily"].data_range == ["2024-01-02", "2024-01-12"]


# ===========================================================================
# 2. Pending discipline — first resolved day and zero-gap days excluded
#
# Zero-gap day: open == prev_close → gap_pts == 0 → excluded from denominators.
# The first resolved day already has no prior close; this test verifies that a
# zero-gap day that appears between two regular days is also excluded.
#
# Scenario:
#   day 0: open=100, close=110       → anchor, no prior close, excluded
#   day 1: open=110, close=115       → open == prev_close (110==110) → zero-gap, excluded
#   day 2: open=120, close=115       → gap_up (+5), not filled (day_low=116 > 115)
#
# Countable gap days: 1 (day 2 only).
# total_samples = 3 (all resolved).
# ===========================================================================

def test_zero_gap_day_excluded_from_denominators() -> None:
  """A day where open == prev_close has zero gap and is excluded from all totals."""
  # day 1: open==prev_close → gap_pts==0 → excluded from denominators
  days = [
    {
      "date": "2024-02-01",
      "session_open": 100.00,
      "session_close": 110.00,
    },
    # open=110 == prev_close=110 → zero gap → excluded
    {
      "date": "2024-02-02",
      "session_open": 110.00,
      "session_close": 115.00,
    },
    # gap_up (+5 from 115), target=115.00, day_low=116.00 > 115.00 → NOT FILLED
    {
      "date": "2024-02-05",
      "session_open": 120.00,
      "session_close": 118.00,
      "day_high": 121.00,
      "day_low": 116.00,
    },
  ]
  result = _stat().compute(make_candles(days))
  tf = result.instruments["NQ"]["daily"]
  # All 3 are resolved → total_samples = 3
  assert tf.total_samples == 3
  # Only day 2 is countable (day 0 has no prior, day 1 has zero gap)
  gap_up_filled = _row(result, "gap_up", "filled")
  gap_up_not_filled = _row(result, "gap_up", "not_filled")
  assert gap_up_filled.total == 1
  assert gap_up_not_filled.total == 1
  # day 2 is not filled
  assert gap_up_filled.count == 0
  assert gap_up_not_filled.count == 1
  # gap_down has 0 countable days
  gap_down_filled = _row(result, "gap_down", "filled")
  assert gap_down_filled.total == 0
  assert gap_down_filled.probability == pytest.approx(0.0)


# ===========================================================================
# 3. Partial-fill threshold
#
# Construct a day with a gap that retraces exactly 60% of the gap distance.
#
# Setup (pct varies):
#   day 0: open=100, close=100             → anchor
#   day 1: open=110, close=108             → gap_up = +10 (prev_close=100)
#
# Fill targets by threshold:
#   pct=50:  target = 110 - 0.50 * 10 = 105.00
#   pct=100: target = 110 - 1.00 * 10 = 100.00
#
# Day 1 day_low = 104.00
#   pct=50:  104.00 ≤ 105.00 → FILLED
#   pct=100: 104.00 > 100.00 → NOT FILLED
# ===========================================================================

_PARTIAL_FILL_DAYS = [
  {
    "date": "2024-03-01",
    "session_open": 100.00,
    "session_close": 100.00,
  },
  # gap_up of 10; day_low=104 retraces 60% (104 ≤ 105 for pct=50, but 104 > 100 for pct=100)
  {
    "date": "2024-03-04",
    "session_open": 110.00,
    "session_close": 108.00,
    "day_high": 111.00,
    "day_low": 104.00,
  },
]


def test_partial_fill_at_50pct_threshold_is_filled() -> None:
  """A 60%-retrace gap counts as filled at fill_threshold_pct=50."""
  # gap=10, day_low=104.00, target(50%)=105.00; 104.00 ≤ 105.00 → filled
  result = _stat(fill_threshold_pct=50.0).compute(make_candles(_PARTIAL_FILL_DAYS))
  r = _row(result, "gap_up", "filled")
  assert r.count == 1
  assert r.total == 1
  assert r.probability == pytest.approx(1.0)


def test_partial_fill_at_100pct_threshold_is_not_filled() -> None:
  """The same 60%-retrace gap is NOT filled at the default fill_threshold_pct=100."""
  # gap=10, day_low=104.00, target(100%)=100.00; 104.00 > 100.00 → not filled
  result = _stat(fill_threshold_pct=100.0).compute(make_candles(_PARTIAL_FILL_DAYS))
  r = _row(result, "gap_up", "filled")
  assert r.count == 0
  r_nf = _row(result, "gap_up", "not_filled")
  assert r_nf.count == 1
  assert r_nf.probability == pytest.approx(1.0)


# ===========================================================================
# 4. Unresolved (truncated) days excluded
# ===========================================================================

def test_pending_day_excluded_from_total_samples() -> None:
  """A truncated day (last bar mod 590 < 960) does not appear in total_samples."""
  base_df = make_candles(_SEQ)
  total_base = _stat().compute(base_df).instruments["NQ"]["daily"].total_samples
  truncated = _make_truncated_day("2024-01-15")
  combined = (
    pd.concat([base_df, truncated], ignore_index=True)
    .sort_values("timestamp")
    .reset_index(drop=True)
  )
  total_with = _stat().compute(combined).instruments["NQ"]["daily"].total_samples
  # Truncated day must not increase the count
  assert total_with == total_base == 9


def test_pending_day_absent_from_day_table() -> None:
  """build_day_table excludes the pending (truncated) day entirely."""
  stat = _stat()
  day_table = stat.build_day_table(_make_truncated_day("2024-01-15"))
  pending = pd.Timestamp("2024-01-15", tz=_NY).normalize()
  assert pending not in day_table.index


def test_pending_day_does_not_affect_matrix_counts() -> None:
  """Adding a truncated day after the sequence does not change condition counts."""
  base_result = _stat().compute(make_candles(_SEQ))
  truncated = _make_truncated_day("2024-01-15")
  combined = (
    pd.concat([make_candles(_SEQ), truncated], ignore_index=True)
    .sort_values("timestamp")
    .reset_index(drop=True)
  )
  extended_result = _stat().compute(combined)
  for cond in ("gap_up", "gap_down"):
    for out in ("filled", "not_filled"):
      base_row = _row(base_result, cond, out)
      ext_row = _row(extended_result, cond, out)
      assert base_row.count == ext_row.count
      assert base_row.total == ext_row.total


# ===========================================================================
# 5. Slices present and correct
#
# GapFill declares slices: weekday, close, prev_candle, size_pts, size_pct
# We spot-check with _SEQ (9 days, days 1-8 countable):
#
# Session-color (close) slice — from session_green:
#   Green sessions (session_close >= session_open): idx 2,3,4,5,6  (close≥open)
#     idx 2: open=98, close=102  → green
#     idx 3: open=88, close=92   → green
#     idx 4: open=105, close=98  → red (98<105)
#     idx 5: open=80, close=85   → green
#     idx 6: open=90, close=100  → green
#     idx 7: open=78, close=72   → red
#     idx 8: open=85, close=80   → red
#   (idx 0: open=100, close=105 → green, but excluded from denominators)
#   (idx 1: open=110, close=95  → red)
#
#   Day 0 IS included in total_samples but has no prior close. It IS in
#   the day_table (9 rows), so the close slicer reads its session_green.
#   However, its gap_size_pts is NaN → compute_rows' countable mask
#   excludes it. So for slices, totals still only reflect countable days.
#
#   All 9 resolved days appear in the day_table; session_green is set for all.
#   Close slicer groups by session_green:
#     green: idx 0(excluded),2,3,5,6    but idx 0 is NOT countable → 4 countable
#     red:   idx 1,4,7,8                but all 4 are countable → 4 countable
#
#   Wait — total_samples in slice groups counts ALL rows in the subset (len(sub)),
#   not just countable ones. The close slicer operates on the full day_table.
#   Day 0 has session_green=True (close=105 ≥ open=100).
#
#   Green slice total_samples = count of all rows where session_green=True:
#     idx 0: green=True (close=105≥open=100) ← included in group even though no prior close
#     idx 2: green=True (102≥98)
#     idx 3: green=True (92≥88)
#     idx 5: green=True (85≥80)
#     idx 6: green=True (100≥90)
#     → 5 days in green slice (total_samples=5 for the group)
#
#   Red slice total_samples = count of rows where session_green=False:
#     idx 1: red (95<110)
#     idx 4: red (98<105)
#     idx 7: red (72<78)
#     idx 8: red (80<85)
#     → 4 days in red slice (total_samples=4 for the group)
#
#   For the "close" slice, compute_rows over the green subset:
#     Countable green days (has prior close AND gap != 0):
#       idx 2: gap_up, not_filled   → gap_up  +3
#       idx 3: gap_down, filled     → gap_down, target=102, high=103 ≥ 102 → filled
#       idx 5: gap_down, not_filled → gap_down, target=98, high=97 < 98 → not_filled
#       idx 6: gap_up, filled       → gap_up, target=85, low=84.5 ≤ 85 → filled
#     Green countable: 4 days (idx 0 is not countable — no prior close)
#
#     green gap_up: idx 2(not filled), 6(filled) → total=2, filled=1, P=0.5
#     green gap_down: idx 3(filled), 5(not filled) → total=2, filled=1, P=0.5
#
#   For the "close" slice, compute_rows over the red subset:
#     Countable red days:
#       idx 1: gap_up, filled   → gap_up, target=105, low=94.50 ≤ 105 → filled
#       idx 4: gap_up, filled   → gap_up, target=92, low=91.5 ≤ 92 → filled
#       idx 7: gap_down, filled → gap_down, target=100, high=100.5 ≥ 100 → filled
#       idx 8: gap_up, not_filled → gap_up, target=72, low=73 > 72 → not filled
#     Red countable: 4 days
#
#     red gap_up: idx 1(filled), 4(filled), 8(not_filled) → total=3, filled=2, P=2/3
#     red gap_down: idx 7(filled) → total=1, filled=1, P=1.0
# ===========================================================================

def test_slices_keys_present() -> None:
  """GapFill result must carry all five declared slice dimensions."""
  result = _stat().compute(make_candles(_SEQ))
  slices = result.instruments["NQ"]["daily"].slices
  assert set(slices.keys()) == {"weekday", "close", "prev_candle", "size_pts", "size_pct"}


def test_close_slice_has_green_and_red_groups() -> None:
  """The close slice produces green and red groups."""
  result = _stat().compute(make_candles(_SEQ))
  groups = result.instruments["NQ"]["daily"].slices["close"].groups
  assert "green" in groups
  assert "red" in groups


def test_close_slice_group_total_samples() -> None:
  """Green close group: 5 days; red close group: 4 days."""
  # Green days (session_close >= session_open): idx 0,2,3,5,6 → 5
  # Red days  (session_close < session_open):  idx 1,4,7,8   → 4
  result = _stat().compute(make_candles(_SEQ))
  groups = result.instruments["NQ"]["daily"].slices["close"].groups
  assert groups["green"].total_samples == 5
  assert groups["red"].total_samples == 4


def test_close_slice_green_gap_up_counts() -> None:
  """Green close group — gap_up: 2 countable (idx 2 not filled, idx 6 filled) → P=0.5."""
  # Countable green gap-up days: idx 2 (not filled), idx 6 (filled) → total=2
  result = _stat().compute(make_candles(_SEQ))
  groups = result.instruments["NQ"]["daily"].slices["close"].groups
  green_results = groups["green"].results
  green_up_filled = next(r for r in green_results if r.condition == "gap_up" and r.outcome == "filled")
  green_up_not = next(r for r in green_results if r.condition == "gap_up" and r.outcome == "not_filled")
  assert green_up_filled.total == 2
  assert green_up_filled.count == 1
  assert green_up_filled.probability == pytest.approx(0.5)
  assert green_up_not.count == 1


def test_close_slice_red_gap_up_counts() -> None:
  """Red close group — gap_up: idx 1(filled),4(filled),8(not_filled) → total=3, P=2/3."""
  result = _stat().compute(make_candles(_SEQ))
  groups = result.instruments["NQ"]["daily"].slices["close"].groups
  red_results = groups["red"].results
  red_up_filled = next(r for r in red_results if r.condition == "gap_up" and r.outcome == "filled")
  assert red_up_filled.total == 3
  assert red_up_filled.count == 2
  assert red_up_filled.probability == pytest.approx(2 / 3)


def test_close_slice_red_gap_down_filled() -> None:
  """Red close group — gap_down: only idx 7 (filled) → total=1, P=1.0."""
  result = _stat().compute(make_candles(_SEQ))
  groups = result.instruments["NQ"]["daily"].slices["close"].groups
  red_results = groups["red"].results
  red_down_filled = next(r for r in red_results if r.condition == "gap_down" and r.outcome == "filled")
  assert red_down_filled.total == 1
  assert red_down_filled.count == 1
  assert red_down_filled.probability == pytest.approx(1.0)


def test_weekday_slice_present() -> None:
  """The weekday slice produces at least one group."""
  result = _stat().compute(make_candles(_SEQ))
  groups = result.instruments["NQ"]["daily"].slices["weekday"].groups
  assert len(groups) > 0


def test_prev_candle_slice_present() -> None:
  """The prev_candle slice produces at least one group."""
  result = _stat().compute(make_candles(_SEQ))
  groups = result.instruments["NQ"]["daily"].slices["prev_candle"].groups
  assert len(groups) > 0


def test_size_pts_and_size_pct_slices_present() -> None:
  """Both SizeBucket slices (size_pts, size_pct) produce groups with 8 countable days."""
  result = _stat().compute(make_candles(_SEQ))
  slices = result.instruments["NQ"]["daily"].slices
  # 8 countable days → quartiles should yield 2 groups (some ties may collapse bins)
  assert len(slices["size_pts"].groups) >= 1
  assert len(slices["size_pct"].groups) >= 1


# ===========================================================================
# 6. Baseline: determinism, fill-count preservation, baseline_n
# ===========================================================================

def test_baseline_deterministic_same_seed() -> None:
  """Two calls with the same seed return identical baseline row probabilities."""
  stat = _stat()
  df = make_candles(_SEQ)
  rows_a = stat.baseline(df, seed=7)
  rows_b = stat.baseline(df, seed=7)
  assert len(rows_a) == len(rows_b)
  for a, b in zip(rows_a, rows_b):
    assert (a.condition, a.outcome) == (b.condition, b.outcome)
    assert a.probability == pytest.approx(b.probability)


def test_baseline_different_seeds_differ() -> None:
  """Two calls with different seeds produce different baseline probabilities.

  Seeds 7 and 0 are verified to yield distinct permutations of the filled
  column, resulting in different gap_up vs gap_down fill distributions.
  (seed=7 → gap_up filled=5/5, gap_down filled=0/3;
   seed=0 → gap_up filled=3/5, gap_down filled=2/3)
  """
  stat = _stat()
  df = make_candles(_SEQ)
  rows_7 = stat.baseline(df, seed=7)
  rows_0 = stat.baseline(df, seed=0)
  all_same = all(
    abs(a.probability - b.probability) < 1e-9
    for a, b in zip(rows_7, rows_0)
  )
  assert not all_same


def test_baseline_preserves_pooled_fill_count() -> None:
  """The permutation baseline preserves the total filled count across all countable days.

  The baseline permutes the 'filled' column across countable days, so the
  total number of filled outcomes (summed over gap_up and gap_down) must equal
  the real total filled count. Only the association with gap direction is
  destroyed.

  Real filled: idx 1(up), 3(down), 4(up), 6(up), 7(down) → 5 filled in 8 countable.
  Baseline: permuted filled values → sum(filled_up.count + filled_down.count) == 5.
  """
  stat = _stat()
  df = make_candles(_SEQ)
  day_table = stat.build_day_table(df)
  # Independently compute the real pooled fill count
  real_filled_up = _row(_stat().compute(df), "gap_up", "filled").count
  real_filled_down = _row(_stat().compute(df), "gap_down", "filled").count
  real_total_filled = real_filled_up + real_filled_down
  # Real: idx 1(up,F), 3(down,F), 4(up,F), 6(up,F), 7(down,F) → 5 filled
  assert real_total_filled == 5

  # The baseline permutes filled within countable days → pooled count preserved
  for seed in (0, 1, 42, 123):
    bl_rows = stat.baseline_rows(day_table, seed=seed)
    bl_map = {(r.condition, r.outcome): r for r in bl_rows}
    bl_filled_total = (
      bl_map[("gap_up", "filled")].count
      + bl_map[("gap_down", "filled")].count
    )
    assert bl_filled_total == real_total_filled, (
      f"seed={seed}: baseline filled={bl_filled_total}, expected {real_total_filled}"
    )


def test_baseline_n_equals_condition_total() -> None:
  """After compute(), each row's baseline_n equals the condition's real total."""
  result = _stat().compute(make_candles(_SEQ))
  for cond in ("gap_up", "gap_down"):
    filled = _row(result, cond, "filled")
    not_filled = _row(result, cond, "not_filled")
    # baseline_n is the condition total from the baseline rows
    assert filled.baseline_n == filled.total
    assert not_filled.baseline_n == not_filled.total


def test_baseline_embedded_in_compute() -> None:
  """After compute(), every row has a positive baseline_n (baseline ran successfully)."""
  result = _stat().compute(make_candles(_SEQ))
  for row in result.instruments["NQ"]["daily"].results:
    if row.total > 0:
      assert row.baseline_n > 0


# ===========================================================================
# 7. Empty input → four zero rows, total_samples 0, empty data_range
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


def test_empty_dataframe_four_rows_present() -> None:
  """Even with no data, all four matrix rows are emitted."""
  result = _stat().compute(_empty_df())
  outcomes = {(r.condition, r.outcome) for r in result.instruments["NQ"]["daily"].results}
  assert outcomes == {
    ("gap_up", "filled"),
    ("gap_up", "not_filled"),
    ("gap_down", "filled"),
    ("gap_down", "not_filled"),
  }


def test_single_resolved_day_no_countable_rows() -> None:
  """One resolved session → no prior close → all condition totals zero, no crash."""
  days = [{"date": "2024-03-01", "session_open": 100.0, "session_close": 110.0}]
  result = _stat().compute(make_candles(days))
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 1
  for row in tf.results:
    assert row.total == 0
    assert row.probability == pytest.approx(0.0)


def test_all_gap_up_no_gap_down_condition() -> None:
  """All gaps up → gap_down condition is empty (total=0, probability=0.0, no crash)."""
  # Each day opens well above the prior close
  days = [
    {"date": "2024-01-02", "session_open": 100.0, "session_close": 105.0},
    {"date": "2024-01-03", "session_open": 112.0, "session_close": 115.0, "day_high": 116.0, "day_low": 104.0},
    {"date": "2024-01-04", "session_open": 122.0, "session_close": 125.0, "day_high": 126.0, "day_low": 114.0},
    {"date": "2024-01-05", "session_open": 132.0, "session_close": 135.0, "day_high": 136.0, "day_low": 124.0},
  ]
  result = _stat().compute(make_candles(days))
  # gap_down condition must be empty
  r = _row(result, "gap_down", "filled")
  assert r.total == 0
  assert r.probability == pytest.approx(0.0)
  # gap_up must have 3 countable days
  r_up = _row(result, "gap_up", "filled")
  assert r_up.total == 3


def test_all_gap_down_no_gap_up_condition() -> None:
  """All gaps down → gap_up condition is empty (total=0, probability=0.0, no crash)."""
  days = [
    {"date": "2024-01-02", "session_open": 120.0, "session_close": 115.0},
    {"date": "2024-01-03", "session_open": 108.0, "session_close": 105.0, "day_high": 114.0, "day_low": 104.50},
    {"date": "2024-01-04", "session_open": 98.0, "session_close": 96.0, "day_high": 104.0, "day_low": 95.50},
  ]
  result = _stat().compute(make_candles(days))
  r = _row(result, "gap_up", "filled")
  assert r.total == 0
  assert r.probability == pytest.approx(0.0)
  r_down = _row(result, "gap_down", "filled")
  assert r_down.total == 2


# ===========================================================================
# 8. write_results round-trip via Pydantic
# ===========================================================================

def test_write_results_round_trip(tmp_path: Path) -> None:
  """write_results produces gap_fill.json that re-validates as StatRunResult."""
  result = _stat().compute(make_candles(_SEQ))
  written = write_results(result, results_dir=tmp_path)
  assert written.name == "gap_fill.json"

  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)
  tf = validated.instruments["NQ"]["daily"]
  assert tf.total_samples == 9
  # Spot-check the gap_up filled row
  gu_filled = next(r for r in tf.results if r.condition == "gap_up" and r.outcome == "filled")
  assert gu_filled.count == 3
  assert gu_filled.total == 5
  assert gu_filled.probability == pytest.approx(3 / 5)


def test_write_results_has_all_slices(tmp_path: Path) -> None:
  """Serialised JSON contains all five declared slice dimensions."""
  result = _stat().compute(make_candles(_SEQ))
  written = write_results(result, results_dir=tmp_path)
  raw = json.loads(written.read_text(encoding="utf-8"))
  slices = raw["instruments"]["NQ"]["daily"]["slices"]
  assert set(slices.keys()) == {"weekday", "close", "prev_candle", "size_pts", "size_pct"}


def test_write_results_utf8_literals(tmp_path: Path) -> None:
  """French text is stored as literal UTF-8 characters, not escaped unicode."""
  result = _stat().compute(_empty_df())
  written = write_results(result, results_dir=tmp_path)
  raw = written.read_text(encoding="utf-8")
  # French definition contains "intraday" and accented characters
  assert "é" in raw
  assert "\\u00e9" not in raw


def test_stat_name() -> None:
  """stat_name attribute must equal 'gap_fill'."""
  result = _stat().compute(_empty_df())
  assert result.stat_name == "gap_fill"


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
  assert set(result.labels.conditions) == {"gap_up", "gap_down"}
  assert set(result.labels.outcomes) == {"filled", "not_filled"}


def test_i18n_labels_dimensions_contain_declared_slicers() -> None:
  """Labels.dimensions carries entries for all five declared slice dimensions."""
  result = _stat().compute(make_candles(_SEQ))
  assert set(result.labels.dimensions.keys()) >= {"weekday", "close", "prev_candle", "size_pts", "size_pct"}
  for key, i18n in result.labels.dimensions.items():
    assert i18n.en != "", f"dimensions[{key}].en empty"
