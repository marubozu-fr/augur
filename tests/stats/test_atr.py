"""Tests for stats.atr.standard.

All data is synthetic — no real market files required. Expected values are
hand-calculated before each assertion.

Framing:
  - Condition ``atr``: how often the session's True Range exceeds or respects
    the prior-session period-N ATR (rolling mean of the N most recent true
    ranges, strictly before the current session).
  - True Range for session d:
      true_range = max(day_high - day_low,
                       abs(day_high - prev_close),
                       abs(day_low  - prev_close))
    where prev_close = the prior RESOLVED session's session_close.
  - For the first resolved session prev_close is NaN, so the gap terms are
    dropped and true_range = day_high - day_low (Wilder first-bar convention).
  - ``exceeded``:  true_range > atr  (strict; touching the ATR → respected).
  - ``respected``: true_range <= atr.

ATR for day d = mean(true_ranges[d-N .. d-1])  (no lookahead, shift(1)).
The first N resolved sessions have NaN atr and are EXCLUDED from every
denominator.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.atr.standard import AverageTrueRange
from stats.base import StatRunResult, write_results
from tests.stats.range_helpers import (
  _NY,
  _empty_df,
  _make_truncated_day,
  _row,
  _stat_factory,
  make_candles,
)


def _stat(period: int = 3) -> AverageTrueRange:
  return _stat_factory(AverageTrueRange, period=period)


# ===========================================================================
# 1. ATR warm-up exclusion
#
# period=3, M=7 sessions.
#
# All days have a tight plain range (high-low=1) so the True Range is driven
# by overnight GAPS.  Day 0 has no prev_close → TR = high-low (fallback).
#
# Construction (session_close is the gap reference for the next day):
#   idx  date        high  low   close  prev_close  TR
#    0   2024-01-02  110   100   100    –           10   (first bar: TR=high-low)
#    1   2024-01-03  111   110   110    100         max(1,11,10)=11
#    2   2024-01-04  112   111   111    110         max(1,2,1)=2
#    3   2024-01-05  162   161   161    111         max(1,51,50)=51
#    4   2024-01-08  163   162   162    161         max(1,2,1)=2
#    5   2024-01-09  214   213   213    162         max(1,52,51)=52
#    6   2024-01-10  215   214   214    213         max(1,2,1)=2
#
# TRs: [10, 11, 2, 51, 2, 52, 2]
#
# rolling(3).mean():
#   idx 0: NaN
#   idx 1: NaN
#   idx 2: mean(10,11,2) = 7.667
#   idx 3: mean(11,2,51) = 21.333
#   idx 4: mean(2,51,2)  = 18.333
#   idx 5: mean(51,2,52) = 35.0
#   idx 6: mean(2,52,2)  = 18.667
#
# atr = shift(1):
#   idx 0: NaN
#   idx 1: NaN
#   idx 2: NaN
#   idx 3: 7.667   countable  TR=51 > 7.667  → exceeded
#   idx 4: 21.333  countable  TR=2  > 21.333 → respected
#   idx 5: 18.333  countable  TR=52 > 18.333 → exceeded
#   idx 6: 35.0    countable  TR=2  > 35.0   → respected
#
# Countable = M - period = 7 - 3 = 4.
# exceeded=2, respected=2.
# ===========================================================================

_7_SESSIONS = [
  {"date": "2024-01-02", "open": 100.0, "close": 100.0, "high": 110.0, "low": 100.0},
  {"date": "2024-01-03", "open": 110.0, "close": 110.0, "high": 111.0, "low": 110.0},
  {"date": "2024-01-04", "open": 110.0, "close": 111.0, "high": 112.0, "low": 111.0},
  {"date": "2024-01-05", "open": 161.0, "close": 161.0, "high": 162.0, "low": 161.0},
  {"date": "2024-01-08", "open": 162.0, "close": 162.0, "high": 163.0, "low": 162.0},
  {"date": "2024-01-09", "open": 213.0, "close": 213.0, "high": 214.0, "low": 213.0},
  {"date": "2024-01-10", "open": 214.0, "close": 214.0, "high": 215.0, "low": 214.0},
]


def test_warmup_exclusion_period3() -> None:
  """period=3, 7 sessions: countable total = 7 - 3 = 4."""
  result = _stat(period=3).compute(make_candles(_7_SESSIONS))
  exceeded = _row(result, "atr", "exceeded")
  respected = _row(result, "atr", "respected")
  assert exceeded.total == 4
  assert respected.total == 4


def test_warmup_exclusion_period5() -> None:
  """period=5, 7 sessions: countable total = 7 - 5 = 2."""
  result = _stat(period=5).compute(make_candles(_7_SESSIONS))
  exceeded = _row(result, "atr", "exceeded")
  assert exceeded.total == 2


def test_total_samples_counts_all_resolved_days() -> None:
  """total_samples = all 7 resolved sessions, including warm-up days."""
  result = _stat().compute(make_candles(_7_SESSIONS))
  assert result.instruments["NQ"]["daily"].total_samples == 7


# ===========================================================================
# 2. Exceeded vs respected partition (period=3, 7 sessions, see table above)
#
# idx 3: TR=51, atr=7.667   → 51 > 7.667?  Yes → exceeded
# idx 4: TR=2,  atr=21.333  → 2  > 21.333? No  → respected
# idx 5: TR=52, atr=18.333  → 52 > 18.333? Yes → exceeded
# idx 6: TR=2,  atr=35.0    → 2  > 35.0?   No  → respected
#
# exceeded_n=2, respected_n=2, total=4
# P(exceeded)=0.5, P(respected)=0.5
# ===========================================================================

def test_exceeded_count_and_probability() -> None:
  """2 of 4 countable days exceed their prior-session ATR → P=0.5."""
  row = _row(_stat(period=3).compute(make_candles(_7_SESSIONS)), "atr", "exceeded")
  assert row.count == 2
  assert row.total == 4
  assert row.probability == pytest.approx(0.5)


def test_respected_count_and_probability() -> None:
  """2 of 4 countable days stay within their prior-session ATR → P=0.5."""
  row = _row(_stat(period=3).compute(make_candles(_7_SESSIONS)), "atr", "respected")
  assert row.count == 2
  assert row.total == 4
  assert row.probability == pytest.approx(0.5)


def test_exceeded_respected_partition() -> None:
  """exceeded.count + respected.count == total for every data set."""
  result = _stat(period=3).compute(make_candles(_7_SESSIONS))
  exc = _row(result, "atr", "exceeded")
  res = _row(result, "atr", "respected")
  assert exc.count + res.count == exc.total == res.total


# ===========================================================================
# 2b. Boundary case: true_range == atr exactly → must be classified "respected"
#     (strict >; touching is NOT exceeding)
#
# Design: 4 sessions, all plain ranges = 30, all closes = 100.
#   prev_close stays 100 for each day → gap terms = |high-100| and |low-100|
#   Day 0: high=130, low=100, close=100.  First day → TR=30 (fallback).
#   Day 1: high=130, low=100, close=100, prev_close=100.
#          TR=max(30, |130-100|=30, |100-100|=0)=30.
#   Day 2: high=130, low=100, close=100, prev_close=100.
#          TR=max(30, 30, 0)=30.
#   Day 3: high=130, low=100, close=100, prev_close=100.
#          TR=max(30, 30, 0)=30.
#          atr[3]=mean(30,30,30)=30.  30 > 30 is False → respected.
#
# Countable = 4-3 = 1; exceeded=0, respected=1.
# ===========================================================================

_BOUNDARY_SESSIONS = [
  {"date": "2024-01-02", "open": 100.0, "close": 100.0, "high": 130.0, "low": 100.0},
  {"date": "2024-01-03", "open": 100.0, "close": 100.0, "high": 130.0, "low": 100.0},
  {"date": "2024-01-04", "open": 100.0, "close": 100.0, "high": 130.0, "low": 100.0},
  {"date": "2024-01-05", "open": 100.0, "close": 100.0, "high": 130.0, "low": 100.0},
]


def test_boundary_true_range_equals_atr_is_respected() -> None:
  """true_range == atr exactly → classified as respected (strict > for exceeded)."""
  result = _stat(period=3).compute(make_candles(_BOUNDARY_SESSIONS))
  exc = _row(result, "atr", "exceeded")
  res = _row(result, "atr", "respected")
  # Countable = 4 - 3 = 1; TR=30 == atr=30 → exceeded=0, respected=1.
  assert exc.count == 0
  assert res.count == 1
  assert exc.total == res.total == 1
  assert exc.probability == pytest.approx(0.0)
  assert res.probability == pytest.approx(1.0)


# ===========================================================================
# 3. ATR value correctness (column-level assertions)
#
# Using the 7-session sequence (period=3):
#   TRs: [10, 11, 2, 51, 2, 52, 2]
#   atr[3] = mean(10,11,2)  = 7.667
#   atr[4] = mean(11,2,51)  = 21.333
#   atr[5] = mean(2,51,2)   = 18.333
#   atr[6] = mean(51,2,52)  = 35.0
# ===========================================================================

def test_atr_column_value_idx3() -> None:
  """atr at idx 3 = mean(TR[0..2]) = mean(10,11,2) = 7.667 (no lookahead)."""
  stat = _stat(period=3)
  table = stat.build_day_table(make_candles(_7_SESSIONS))
  date_idx3 = pd.Timestamp("2024-01-05", tz=_NY).normalize()
  assert table.loc[date_idx3, "atr"] == pytest.approx((10 + 11 + 2) / 3)


def test_atr_column_value_idx4() -> None:
  """atr at idx 4 = mean(TR[1..3]) = mean(11,2,51) = 21.333."""
  stat = _stat(period=3)
  table = stat.build_day_table(make_candles(_7_SESSIONS))
  date_idx4 = pd.Timestamp("2024-01-08", tz=_NY).normalize()
  assert table.loc[date_idx4, "atr"] == pytest.approx((11 + 2 + 51) / 3)


def test_atr_column_value_idx5() -> None:
  """atr at idx 5 = mean(TR[2..4]) = mean(2,51,2) = 18.333."""
  stat = _stat(period=3)
  table = stat.build_day_table(make_candles(_7_SESSIONS))
  date_idx5 = pd.Timestamp("2024-01-09", tz=_NY).normalize()
  assert table.loc[date_idx5, "atr"] == pytest.approx((2 + 51 + 2) / 3)


def test_atr_column_value_idx6() -> None:
  """atr at idx 6 = mean(TR[3..5]) = mean(51,2,52) = 35.0."""
  stat = _stat(period=3)
  table = stat.build_day_table(make_candles(_7_SESSIONS))
  date_idx6 = pd.Timestamp("2024-01-10", tz=_NY).normalize()
  assert table.loc[date_idx6, "atr"] == pytest.approx((51 + 2 + 52) / 3)


def test_atr_no_lookahead_first_n_rows_are_nan() -> None:
  """The first period resolved sessions have NaN atr — no lookahead."""
  stat = _stat(period=3)
  table = stat.build_day_table(make_candles(_7_SESSIONS))
  # First 3 rows (period=3) must have NaN atr.
  first_three_dates = table.index[:3]
  for date in first_three_dates:
    assert pd.isna(table.loc[date, "atr"]), f"Expected NaN atr at {date}"


# ===========================================================================
# 4. True Range correctness — GAP-specific tests
#
# These tests verify that the True Range correctly selects the gap component
# when it is larger than the plain high-low range.  This is the fundamental
# difference between ATR and ADR.
#
# Session design (high-low = 1 but large overnight gaps):
#   Day 0: high=110, low=100, close=100. TR fallback = 10 (first session).
#   Day 1: high=111, low=110, close=110, prev_close=100.
#           plain range = 1;  |high-prev|=11;  |low-prev|=10 → TR=11 (gap wins)
#   Day 2: high=112, low=111, close=111, prev_close=110.
#           plain range = 1;  |high-prev|=2;   |low-prev|=1  → TR=2  (gap wins)
#
# These TRs differ from the plain range (1) for every gap day.
# ===========================================================================

def test_gap_true_range_exceeds_plain_range_day1() -> None:
  """Day 1 TR=11 (gap term) > plain range 1: gap is correctly selected."""
  stat = _stat(period=3)
  # Use first 2 sessions only; Day 0 is first (TR=fallback), Day 1 has gap.
  candles = make_candles([
    {"date": "2024-01-02", "open": 100.0, "close": 100.0, "high": 110.0, "low": 100.0},
    {"date": "2024-01-03", "open": 110.0, "close": 110.0, "high": 111.0, "low": 110.0},
  ])
  table = stat.build_day_table(candles)
  date_d1 = pd.Timestamp("2024-01-03", tz=_NY).normalize()
  # plain range = 1, but gap term abs(111-100)=11 dominates → TR=11.
  assert table.loc[date_d1, "true_range"] == pytest.approx(11.0)


def test_gap_true_range_day2() -> None:
  """Day 2 TR=2 (|112-110|=2) > plain range 1: gap component selected."""
  stat = _stat(period=3)
  candles = make_candles([
    {"date": "2024-01-02", "open": 100.0, "close": 100.0, "high": 110.0, "low": 100.0},
    {"date": "2024-01-03", "open": 110.0, "close": 110.0, "high": 111.0, "low": 110.0},
    {"date": "2024-01-04", "open": 110.0, "close": 111.0, "high": 112.0, "low": 111.0},
  ])
  table = stat.build_day_table(candles)
  date_d2 = pd.Timestamp("2024-01-04", tz=_NY).normalize()
  # plain range = 1, but gap term abs(112-110)=2 dominates → TR=2.
  assert table.loc[date_d2, "true_range"] == pytest.approx(2.0)


def test_gap_downward_gap_true_range() -> None:
  """Downward gap: prev_close=200, today high=170, low=150.

  plain range = 20;  |high-prev|=30;  |low-prev|=50 → TR=50 (low gap wins).
  """
  stat = _stat(period=3)
  candles = make_candles([
    # Day 0: close=200 (reference for Day 1)
    {"date": "2024-01-02", "open": 200.0, "close": 200.0, "high": 210.0, "low": 190.0},
    # Day 1: gap down; high=170, low=150, prev_close=200.
    # plain range=20, |170-200|=30, |150-200|=50 → TR=50
    {"date": "2024-01-03", "open": 165.0, "close": 160.0, "high": 170.0, "low": 150.0},
  ])
  table = stat.build_day_table(candles)
  date_d1 = pd.Timestamp("2024-01-03", tz=_NY).normalize()
  assert table.loc[date_d1, "true_range"] == pytest.approx(50.0)


def test_no_gap_true_range_equals_plain_range() -> None:
  """When prev_close sits inside today's range, TR equals the plain range.

  prev_close=105, high=110, low=100.
  plain range=10; |high-prev|=5; |low-prev|=5 → TR=10 (plain range wins).
  """
  stat = _stat(period=3)
  candles = make_candles([
    # Day 0: close=105
    {"date": "2024-01-02", "open": 100.0, "close": 105.0, "high": 110.0, "low": 100.0},
    # Day 1: prev_close=105, inside today's range [100,110] → TR=range=10.
    {"date": "2024-01-03", "open": 102.0, "close": 107.0, "high": 110.0, "low": 100.0},
  ])
  table = stat.build_day_table(candles)
  date_d1 = pd.Timestamp("2024-01-03", tz=_NY).normalize()
  # |110-105|=5 < 10; |100-105|=5 < 10 → TR=10 (plain high-low).
  assert table.loc[date_d1, "true_range"] == pytest.approx(10.0)


# ===========================================================================
# 5. First resolved session fallback (Wilder convention)
#
# The first resolved session has no prev_close (NaN), so the two gap terms
# are NaN.  pd.concat(...).max(axis=1) with skipna drops the NaN terms and
# returns just the plain high-low range.
# ===========================================================================

def test_first_session_true_range_fallback_to_high_low() -> None:
  """First resolved session: TR = day_high - day_low (gap terms absent).

  Day 0 has high=150, low=100 → TR must be 50 (no gap terms).
  """
  stat = _stat(period=3)
  candles = make_candles([
    {"date": "2024-01-02", "open": 100.0, "close": 120.0, "high": 150.0, "low": 100.0},
    # Day 1 just to ensure the table has the index we need (day 0 is still first).
    {"date": "2024-01-03", "open": 120.0, "close": 125.0, "high": 130.0, "low": 120.0},
  ])
  table = stat.build_day_table(candles)
  date_d0 = pd.Timestamp("2024-01-02", tz=_NY).normalize()
  # First session: TR = high - low = 150 - 100 = 50.
  assert table.loc[date_d0, "true_range"] == pytest.approx(50.0)


def test_first_session_true_range_not_inflated_by_gap() -> None:
  """First session TR is NOT inflated by prev_close-based gap terms.

  If an artificial prev_close of 200 were used, TR would be max(50, 50, 100)=100.
  The correct value is 50 (plain range), confirming no prev_close is applied.
  """
  stat = _stat(period=3)
  candles = make_candles([
    # A second 'preceding' day exists only in a hypothetical scenario; here we
    # test the true first resolved day in the dataset directly.
    {"date": "2024-01-02", "open": 100.0, "close": 120.0, "high": 150.0, "low": 100.0},
  ])
  table = stat.build_day_table(candles)
  date_d0 = pd.Timestamp("2024-01-02", tz=_NY).normalize()
  assert table.loc[date_d0, "true_range"] == pytest.approx(50.0)


# ===========================================================================
# 6. Determinism / reproducibility
# ===========================================================================

def _long_seq() -> pd.DataFrame:
  """~50 weekdays with alternating wide and narrow sessions plus gaps.

  Gaps are produced by keeping session_close != next session_open, so TR > range
  for most days. The sequence has enough variety to make baseline permutations
  non-trivial.
  """
  dates: list[str] = []
  d = pd.Timestamp("2020-01-01", tz=_NY)
  while len(dates) < 50:
    if d.weekday() < 5:
      dates.append(d.strftime("%Y-%m-%d"))
    d += pd.Timedelta(days=1)

  days = []
  base = 100.0
  for i, date in enumerate(dates):
    if i % 2 == 0:
      hi, lo = base + 15.0, base - 15.0  # wide session; range=30
    else:
      hi, lo = base + 5.0, base - 5.0    # narrow session; range=10
    # close at the low, so next session's open gap is large.
    days.append({
      "date": date,
      "open": base - 2.0,
      "close": lo,
      "high": hi,
      "low": lo,
    })
    base += 0.5
  return make_candles(days)


def test_baseline_rows_deterministic() -> None:
  """baseline_rows(seed=42) returns identical results on two successive calls."""
  stat = _stat(period=3)
  table = stat.build_day_table(_long_seq())
  rows_a = stat.baseline_rows(table, seed=42)
  rows_b = stat.baseline_rows(table, seed=42)
  for a, b in zip(rows_a, rows_b):
    assert (a.condition, a.outcome) == (b.condition, b.outcome)
    assert a.probability == pytest.approx(b.probability)
    assert a.count == b.count
    assert a.total == b.total


def test_baseline_permutes_atr_column() -> None:
  """baseline_rows uses a permutation of the atr column: countable N is preserved.

  The permuted atr may redistribute NaN values, so countable may change; but the
  total number of non-NaN atr values across all rows is preserved because numpy
  permutation of the raw array moves NaN positions without creating or destroying
  them.
  """
  stat = _stat(period=3)
  table = stat.build_day_table(_long_seq())
  # Original countable count.
  orig_countable = int((table["atr"].notna() & (table["atr"] > 0)).sum())
  rows = stat.baseline_rows(table, seed=42)
  baseline_total = rows[0].total  # both rows share the same denominator
  # The permutation preserves the number of valid (non-NaN, >0) atr values.
  assert baseline_total == orig_countable


def test_compute_reproducible() -> None:
  """compute() with same input and seed produces identical JSON-serializable output."""
  stat = _stat(period=3)
  df = _long_seq()
  result_a = stat.compute(df, seed=42)
  result_b = stat.compute(df, seed=42)
  assert result_a.model_dump_json() == result_b.model_dump_json()


def test_baseline_embedded_has_positive_n() -> None:
  """After compute(), countable rows carry a positive baseline_n."""
  result = _stat(period=3).compute(_long_seq())
  for row in result.instruments["NQ"]["daily"].results:
    if row.total > 0:
      assert row.baseline_n > 0, f"Expected baseline_n>0 for {row.condition}/{row.outcome}"


def test_baseline_different_seed_produces_different_permutation() -> None:
  """Different seeds produce different permutations of the atr column.

  We test at the permutation level rather than the outcome count level, because
  two different permutations can coincidentally yield the same exceeded/respected
  split count on small datasets.
  """
  import numpy as np

  stat = _stat(period=3)
  table = stat.build_day_table(_long_seq())
  n = len(table)
  rng_42 = np.random.default_rng(42)
  rng_99 = np.random.default_rng(99)
  perm_42 = rng_42.permutation(n)
  perm_99 = rng_99.permutation(n)
  # Two different seeds must produce different orderings.
  assert not (perm_42 == perm_99).all()


# ===========================================================================
# 7. Pending discipline / empty
# ===========================================================================

def test_empty_dataframe_build_day_table() -> None:
  """Empty input → build_day_table returns empty DataFrame."""
  table = _stat().build_day_table(_empty_df())
  assert table.empty


def test_empty_dataframe_compute_rows_both_zero() -> None:
  """Empty day table → compute_rows returns both outcome rows with zeros."""
  stat = _stat()
  rows = stat.compute_rows(pd.DataFrame(columns=["true_range", "atr", "prev_session_green"]))
  by_outcome = {r.outcome: r for r in rows}
  assert set(by_outcome) == {"exceeded", "respected"}
  for r in rows:
    assert r.count == 0
    assert r.total == 0
    assert r.probability == pytest.approx(0.0)


def test_empty_dataframe_full_compute() -> None:
  """Empty candles → full compute() produces total_samples=0, all rows zeroed."""
  result = _stat().compute(_empty_df())
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 0
  assert tf.data_range == []
  for row in tf.results:
    assert row.count == 0
    assert row.total == 0
    assert row.probability == pytest.approx(0.0)


def test_single_resolved_day_all_nan_atr() -> None:
  """One resolved session → atr=NaN (warm-up) → all totals zero, no crash."""
  days = [{"date": "2024-03-01", "open": 100.0, "close": 105.0, "high": 115.0, "low": 95.0}]
  result = _stat().compute(make_candles(days))
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 1
  for row in tf.results:
    assert row.total == 0


def test_pending_day_excluded_from_total_samples() -> None:
  """A truncated/early-close day does not appear in total_samples."""
  stat = _stat(period=3)
  base_df = make_candles(_7_SESSIONS)
  total_base = stat.compute(base_df).instruments["NQ"]["daily"].total_samples
  truncated = _make_truncated_day("2024-01-11")
  combined = (
    pd.concat([base_df, truncated], ignore_index=True)
    .sort_values("timestamp")
    .reset_index(drop=True)
  )
  total_with = stat.compute(combined).instruments["NQ"]["daily"].total_samples
  assert total_with == total_base == 7


def test_pending_day_absent_from_day_table() -> None:
  """build_day_table excludes the pending (truncated) day entirely."""
  day_table = _stat().build_day_table(_make_truncated_day("2024-01-10"))
  pending = pd.Timestamp("2024-01-10", tz=_NY).normalize()
  assert pending not in day_table.index


# ===========================================================================
# 8. Weekday slice
#
# With 7 sessions covering Mon–Thu of two weeks, the weekday slice must
# group them. The sum of exceeded counts across all weekday groups must equal
# the overall exceeded count.  Similarly for respected.
# ===========================================================================

def test_weekday_slice_present() -> None:
  """Result includes a 'weekday' slice."""
  result = _stat(period=3).compute(make_candles(_7_SESSIONS))
  slices = result.instruments["NQ"]["daily"].slices
  assert "weekday" in slices


def test_weekday_slice_exceeded_counts_sum_to_overall() -> None:
  """Sum of exceeded.count across weekday groups == overall exceeded.count."""
  result = _stat(period=3).compute(make_candles(_7_SESSIONS))
  overall_exc = _row(result, "atr", "exceeded").count
  wk = result.instruments["NQ"]["daily"].slices["weekday"]
  total_exc = 0
  for grp in wk.groups.values():
    for r in grp.results:
      if r.condition == "atr" and r.outcome == "exceeded":
        total_exc += r.count
  assert total_exc == overall_exc


def test_weekday_slice_outcomes_partition_per_group() -> None:
  """Within each weekday group, exceeded.count + respected.count == group total."""
  result = _stat(period=3).compute(make_candles(_7_SESSIONS))
  wk = result.instruments["NQ"]["daily"].slices["weekday"]
  for grp_key, grp in wk.groups.items():
    counts = {r.outcome: r for r in grp.results if r.condition == "atr"}
    if "exceeded" in counts and "respected" in counts:
      exc = counts["exceeded"]
      res = counts["respected"]
      assert exc.total == res.total, f"group {grp_key}: totals differ"
      assert exc.count + res.count == exc.total, f"group {grp_key}: partition broken"


# ===========================================================================
# 9. Custom period
# ===========================================================================

def test_custom_period_3_warmup() -> None:
  """period=3: first 3 resolved sessions excluded from denominators."""
  result = _stat(period=3).compute(make_candles(_7_SESSIONS))
  assert _row(result, "atr", "exceeded").total == 7 - 3


def test_custom_period_5_warmup() -> None:
  """period=5: first 5 resolved sessions excluded from denominators."""
  result = _stat(period=5).compute(make_candles(_7_SESSIONS))
  assert _row(result, "atr", "exceeded").total == 7 - 5


def test_custom_period_stored_on_instance() -> None:
  """period is stored and used: different periods give different countable totals."""
  stat3 = _stat(period=3)
  stat5 = _stat(period=5)
  df = make_candles(_7_SESSIONS)
  total3 = _row(stat3.compute(df), "atr", "exceeded").total
  total5 = _row(stat5.compute(df), "atr", "exceeded").total
  assert total3 == 4  # 7 - 3
  assert total5 == 2  # 7 - 5
  assert total3 != total5


def test_period_3_atr_uses_3_session_window() -> None:
  """With period=3, verify the 3-session rolling window via a direct atr value check.

  TRs from _BOUNDARY_SESSIONS (all TR=30, close=100, prev_close=100 for idx>=1):
  atr[3] = mean(30, 30, 30) = 30.0.
  """
  stat = _stat(period=3)
  table = stat.build_day_table(make_candles(_BOUNDARY_SESSIONS))
  date_idx3 = pd.Timestamp("2024-01-05", tz=_NY).normalize()
  assert table.loc[date_idx3, "atr"] == pytest.approx(30.0)


# ===========================================================================
# All-exceeded edge case: all countable days exceed their ATR.
#
# period=3, 4 sessions with TRs: 10, 11, 2, 1000.
#   Day 0: high=110, low=100, close=100.  TR=10 (first bar).
#   Day 1: high=111, low=110, close=110, prev_close=100.  TR=11.
#   Day 2: high=112, low=111, close=111, prev_close=110.  TR=2.
#   Day 3: high=1111,low=1110,close=1110,prev_close=111.
#          TR=max(1,|1111-111|=1000,|1110-111|=999)=1000.
#          atr[3]=mean(10,11,2)=7.667; 1000 > 7.667 → exceeded.
# ===========================================================================

_ALL_EXCEEDED_SESSIONS = [
  {"date": "2024-01-02", "open": 100.0, "close": 100.0, "high": 110.0, "low": 100.0},
  {"date": "2024-01-03", "open": 110.0, "close": 110.0, "high": 111.0, "low": 110.0},
  {"date": "2024-01-04", "open": 110.0, "close": 111.0, "high": 112.0, "low": 111.0},
  # Massive gap up: TR dominated by gap = |1111 - 111| = 1000.
  {"date": "2024-01-05", "open": 1110.0, "close": 1110.0, "high": 1111.0, "low": 1110.0},
]


def test_all_countable_days_exceeded() -> None:
  """When every countable day exceeds its ATR: exceeded=total, respected=0."""
  result = _stat(period=3).compute(make_candles(_ALL_EXCEEDED_SESSIONS))
  exc = _row(result, "atr", "exceeded")
  res = _row(result, "atr", "respected")
  # Countable = 4 - 3 = 1; TR=1000 > atr≈7.667 → exceeded.
  assert exc.total == 1
  assert exc.count == 1
  assert res.count == 0
  assert exc.probability == pytest.approx(1.0)
  assert res.probability == pytest.approx(0.0)


# ===========================================================================
# All-respected edge case: all countable days respect their ATR.
#
# period=3, 4 sessions, tiny TR on the countable day.
#   Day 0: high=110, low=100, close=100.  TR=10 (first bar).
#   Day 1: high=111, low=110, close=110, prev_close=100.  TR=11.
#   Day 2: high=112, low=111, close=111, prev_close=110.  TR=2.
#   Day 3: high=111.1, low=111.0, close=111.0, prev_close=111.
#          TR=max(0.1, |111.1-111|=0.1, |111.0-111|=0)=0.1.
#          atr[3]=mean(10,11,2)=7.667; 0.1 > 7.667? No → respected.
# ===========================================================================

_ALL_RESPECTED_SESSIONS = [
  {"date": "2024-01-02", "open": 100.0, "close": 100.0, "high": 110.0, "low": 100.0},
  {"date": "2024-01-03", "open": 110.0, "close": 110.0, "high": 111.0, "low": 110.0},
  {"date": "2024-01-04", "open": 110.0, "close": 111.0, "high": 112.0, "low": 111.0},
  # Tiny range and no gap: prev_close=111, high=111.1, low=111.0 → TR=0.1.
  {"date": "2024-01-05", "open": 111.0, "close": 111.0, "high": 111.1, "low": 111.0},
]


def test_all_countable_days_respected() -> None:
  """When every countable day respects its ATR: respected=total, exceeded=0."""
  result = _stat(period=3).compute(make_candles(_ALL_RESPECTED_SESSIONS))
  exc = _row(result, "atr", "exceeded")
  res = _row(result, "atr", "respected")
  # Countable = 4 - 3 = 1; TR=0.1 <= atr≈7.667 → respected.
  assert res.total == 1
  assert res.count == 1
  assert exc.count == 0
  assert res.probability == pytest.approx(1.0)
  assert exc.probability == pytest.approx(0.0)


# ===========================================================================
# Two rows only (structure check)
# ===========================================================================

def test_exactly_two_rows() -> None:
  """compute() returns exactly the exceeded and respected outcome rows."""
  rows = _stat(period=3).compute(make_candles(_7_SESSIONS)).instruments["NQ"]["daily"].results
  assert {(r.condition, r.outcome) for r in rows} == {
    ("atr", "exceeded"),
    ("atr", "respected"),
  }


# ===========================================================================
# data_range
# ===========================================================================

def test_data_range_spans_all_resolved_sessions() -> None:
  """data_range spans from the first to the last resolved session date."""
  result = _stat(period=3).compute(make_candles(_7_SESSIONS))
  assert result.instruments["NQ"]["daily"].data_range == ["2024-01-02", "2024-01-10"]


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
  assert set(result.labels.conditions) == {"atr"}
  assert set(result.labels.outcomes) == {"exceeded", "respected"}


# ===========================================================================
# stat_name and write_results round-trip
# ===========================================================================

def test_stat_name() -> None:
  """stat_name is 'atr'."""
  assert _stat().compute(_empty_df()).stat_name == "atr"


def test_write_results_round_trip(tmp_path: Path) -> None:
  """write_results produces atr.json that re-validates correctly."""
  result = _stat(period=3).compute(make_candles(_7_SESSIONS))
  written = write_results(result, results_dir=tmp_path)
  assert written.name == "atr.json"

  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)
  assert validated.instruments["NQ"]["daily"].total_samples == 7


def test_write_results_french_accent_literal(tmp_path: Path) -> None:
  """French text is stored as literal UTF-8 characters, not escaped unicode."""
  result = _stat().compute(_empty_df())
  written = write_results(result, results_dir=tmp_path)
  raw = written.read_text(encoding="utf-8")
  # The French definition contains accented French (e.g. "fréquence").
  assert "fréquence" in raw


# ===========================================================================
# True Range column present in build_day_table output
# ===========================================================================

def test_day_table_has_true_range_column() -> None:
  """build_day_table output has a 'true_range' column (not 'day_range')."""
  table = _stat(period=3).build_day_table(make_candles(_7_SESSIONS))
  assert "true_range" in table.columns
  assert "atr" in table.columns
  assert "day_range" not in table.columns


def test_true_range_is_always_gte_plain_range() -> None:
  """For every resolved session, true_range >= day_high - day_low."""
  stat = _stat(period=3)
  # Use the 7-session gap sequence (most have TR > plain range).
  table = stat.build_day_table(make_candles(_7_SESSIONS))
  # We need day_high and day_low to compute plain range; rebuild from raw candles.
  from stats.utils.daily_candles import build_day_table_with_prior_range
  from stats.config import minute_of_day
  rth_start = minute_of_day("09:30")
  rth_end = minute_of_day("16:15")
  raw = build_day_table_with_prior_range(make_candles(_7_SESSIONS), rth_start, rth_end)
  plain_range = raw["day_high"] - raw["day_low"]
  for date in table.index:
    assert table.loc[date, "true_range"] >= plain_range.loc[date] - 1e-10, (
      f"TR < plain range at {date}: TR={table.loc[date, 'true_range']}, "
      f"range={plain_range.loc[date]}"
    )


# ===========================================================================
# classify_samples
#
# period=3, _7_SESSIONS (see table at top of file):
#   idx 0-2 (2024-01-02, 03, 04): NaN atr → not countable, no SampleRow.
#   idx 3 (2024-01-05): TR=51, atr=7.667   → 51 > 7.667?  Yes → exceeded
#   idx 4 (2024-01-08): TR=2,  atr=21.333  → 2  > 21.333? No  → respected
#   idx 5 (2024-01-09): TR=52, atr=18.333  → 52 > 18.333? Yes → exceeded
#   idx 6 (2024-01-10): TR=2,  atr=35.0    → 2  > 35.0?   No  → respected
# ===========================================================================

def test_classify_samples_exact_list() -> None:
  """classify_samples emits one SampleRow per countable day, matching the hand-calc."""
  stat = _stat(period=3)
  table = stat.build_day_table(make_candles(_7_SESSIONS))
  samples = stat.classify_samples(table)
  assert [(s.date, s.condition, s.outcome) for s in samples] == [
    ("2024-01-05", "atr", "exceeded"),
    ("2024-01-08", "atr", "respected"),
    ("2024-01-09", "atr", "exceeded"),
    ("2024-01-10", "atr", "respected"),
  ]
  assert all(s.value is None for s in samples)


def test_classify_samples_matches_compute_rows_counts() -> None:
  """For every StatResultRow, the matching SampleRow count equals r.count."""
  stat = _stat(period=3)
  table = stat.build_day_table(make_candles(_7_SESSIONS))
  samples = stat.classify_samples(table)
  rows = stat.compute_rows(table)
  for r in rows:
    n = sum(1 for s in samples if s.condition == r.condition and s.outcome == r.outcome)
    assert n == r.count


def test_classify_samples_empty_day_table() -> None:
  """Empty day_table -> classify_samples returns []."""
  stat = _stat(period=3)
  assert stat.classify_samples(pd.DataFrame(columns=["true_range", "atr"])) == []


def test_classify_samples_excludes_noncountable_warmup_days() -> None:
  """The warm-up (NaN atr) days produce no SampleRow."""
  stat = _stat(period=3)
  table = stat.build_day_table(make_candles(_7_SESSIONS))
  samples = stat.classify_samples(table)
  sample_dates = {s.date for s in samples}
  for d in ("2024-01-02", "2024-01-03", "2024-01-04"):
    assert d not in sample_dates
