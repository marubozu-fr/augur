"""Tests for stats.opening_stats.standard.

All data is synthetic — no real market files required.
Expected values are hand-calculated before each assertion.

Module under test
-----------------
OpeningStats computes a marginal 3-way distribution of where the daily RTH
session open lands relative to the PRIOR resolved session's RTH intraday range:

  above_high  — session_open > prev_high
  inside_range — prev_low <= session_open <= prev_high  (ties → inside_range)
  below_low   — session_open < prev_low

Single condition: "open". Three outcome rows that sum to 1.0.
First resolved day (no prior session) excluded from every denominator.
Declared slices: weekday, prev_candle, close.
Baseline: uniform random across the three outcomes (≈1/3 each), seed=42.

Synthetic-data builder conventions (from range_helpers.py)
-----------------------------------------------------------
  make_candles(days) — each day spec has: date, open, close, high, low
  _make_truncated_day(date) — last bar at 09:50, unresolved
  _empty_df() — timezone-aware empty DataFrame
  _make_ohlc_day used internally by make_candles

RTH: 09:30–16:15, last bar at mod 974 (16:14); resolved iff last mod >= 960.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.opening_stats.standard import OpeningStats
from tests.stats.range_helpers import (
  _empty_df,
  _make_truncated_day,
  make_candles,
)

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


def _stat() -> OpeningStats:
  return OpeningStats(instrument="NQ", config=_TEST_CONFIG)


def _row(result: StatRunResult, outcome: str) -> StatResultRow:
  """Retrieve the row for condition='open' and the given outcome."""
  for r in result.instruments["NQ"]["daily"].results:
    if r.condition == "open" and r.outcome == outcome:
      return r
  raise KeyError(outcome)


# ===========================================================================
# 1. Classification correctness
#
# Build 4 consecutive resolved sessions so that days 1–3 each represent one
# outcome. Day 0 is the anchor (excluded, no prior session).
#
# Day layout:
#   idx  date        open    high    low     close   prev_high  prev_low  location
#    0   2024-01-02  100.00  105.00  95.00   102.00  —          —         (anchor, excluded)
#    1   2024-01-03  106.00  108.00  96.00   104.00  105.00     95.00     above_high (106 > 105)
#    2   2024-01-04  100.00  110.00  93.00   107.00  108.00     96.00     inside_range (96 ≤ 100 ≤ 108)
#    3   2024-01-05   90.00  112.00  88.00   108.00  110.00     93.00     below_low (90 < 93)
#
# Countable days: idx 1, 2, 3 → total = 3
# above_high: 1 → P = 1/3
# inside_range: 1 → P = 1/3
# below_low: 1 → P = 1/3
# ===========================================================================

_CLASSIFICATION_SEQ = [
  # idx 0: anchor — no prior session; high=105, low=95
  {
    "date": "2024-01-02",
    "open": 100.00,
    "close": 102.00,
    "high": 105.00,
    "low": 95.00,
  },
  # idx 1: above_high — open=106 > prev_high=105
  {
    "date": "2024-01-03",
    "open": 106.00,
    "close": 104.00,
    "high": 108.00,
    "low": 96.00,
  },
  # idx 2: inside_range — open=100, prev_high=108, prev_low=96 → 96 ≤ 100 ≤ 108
  {
    "date": "2024-01-04",
    "open": 100.00,
    "close": 107.00,
    "high": 110.00,
    "low": 93.00,
  },
  # idx 3: below_low — open=90 < prev_low=93
  {
    "date": "2024-01-05",
    "open": 90.00,
    "close": 108.00,
    "high": 112.00,
    "low": 88.00,
  },
]


def test_classification_above_high() -> None:
  """Day idx 1: open=106 > prev_high=105 → classified as above_high."""
  stat = _stat()
  day_table = stat.build_day_table(make_candles(_CLASSIFICATION_SEQ))
  date = pd.Timestamp("2024-01-03", tz=_NY).normalize()
  assert day_table.loc[date, "open_location"] == "above_high"


def test_classification_inside_range() -> None:
  """Day idx 2: open=100 is within [96, 108] → classified as inside_range."""
  stat = _stat()
  day_table = stat.build_day_table(make_candles(_CLASSIFICATION_SEQ))
  date = pd.Timestamp("2024-01-04", tz=_NY).normalize()
  assert day_table.loc[date, "open_location"] == "inside_range"


def test_classification_below_low() -> None:
  """Day idx 3: open=90 < prev_low=93 → classified as below_low."""
  stat = _stat()
  day_table = stat.build_day_table(make_candles(_CLASSIFICATION_SEQ))
  date = pd.Timestamp("2024-01-05", tz=_NY).normalize()
  assert day_table.loc[date, "open_location"] == "below_low"


def test_classification_counts_and_probabilities() -> None:
  """3 countable days: one per outcome → each count=1, probability=1/3."""
  # Hand-calc: total=3, above=1, inside=1, below=1 → P=1/3 each
  result = _stat().compute(make_candles(_CLASSIFICATION_SEQ))
  assert result.instruments["NQ"]["daily"].total_samples == 3
  for out_key in ("above_high", "inside_range", "below_low"):
    r = _row(result, out_key)
    assert r.total == 3
    assert r.count == 1
    assert r.probability == pytest.approx(1 / 3)


def test_three_rows_only() -> None:
  """Exactly three rows: one condition 'open' × three outcomes."""
  result = _stat().compute(make_candles(_CLASSIFICATION_SEQ))
  rows = result.instruments["NQ"]["daily"].results
  assert {(r.condition, r.outcome) for r in rows} == {
    ("open", "above_high"),
    ("open", "inside_range"),
    ("open", "below_low"),
  }


# ===========================================================================
# 2. Tie / boundary convention
#
# A day whose open equals prev_high must be inside_range (not above_high).
# A day whose open equals prev_low must be inside_range (not below_low).
#
# Setup:
#   idx 0: anchor — high=110, low=90
#   idx 1: open=110 (ties prev_high=110) → inside_range
#   idx 2: open=90  (ties prev_low=90)   → inside_range
#            (prev_high = high from idx 1 = 112, prev_low = low from idx 1 = 88)
#
# So we need idx 1 to have high=112, low=88 so idx 2 can tie prev_low=88.
# Let's use a more direct approach with three explicit anchor + two tie days:
#
#   idx 0: anchor — high=110, low=90
#   idx 1: open=110 (== prev_high=110) → inside_range
#   idx 2: anchor2 — high=100, low=80
#   idx 3: open=80  (== prev_low=80)   → inside_range
# ===========================================================================

_BOUNDARY_SEQ = [
  # idx 0: anchor, high=110, low=90
  {
    "date": "2024-02-01",
    "open": 100.00,
    "close": 102.00,
    "high": 110.00,
    "low": 90.00,
  },
  # idx 1: open=110 == prev_high → inside_range (not above_high)
  {
    "date": "2024-02-02",
    "open": 110.00,
    "close": 105.00,
    "high": 115.00,
    "low": 85.00,
  },
  # idx 2: open=85 == prev_low=85 → inside_range (not below_low)
  {
    "date": "2024-02-05",
    "open": 85.00,
    "close": 90.00,
    "high": 95.00,
    "low": 80.00,
  },
]


def test_tie_on_prev_high_is_inside_range() -> None:
  """open == prev_high (110) must be classified as inside_range, not above_high."""
  stat = _stat()
  day_table = stat.build_day_table(make_candles(_BOUNDARY_SEQ))
  date = pd.Timestamp("2024-02-02", tz=_NY).normalize()
  assert day_table.loc[date, "open_location"] == "inside_range"


def test_tie_on_prev_low_is_inside_range() -> None:
  """open == prev_low (85) must be classified as inside_range, not below_low."""
  stat = _stat()
  day_table = stat.build_day_table(make_candles(_BOUNDARY_SEQ))
  date = pd.Timestamp("2024-02-05", tz=_NY).normalize()
  assert day_table.loc[date, "open_location"] == "inside_range"


def test_no_above_high_or_below_low_when_all_ties() -> None:
  """Both tie days → above_high.count=0 and below_low.count=0."""
  # 2 countable days, both inside_range by tie: inside=2, above=0, below=0
  result = _stat().compute(make_candles(_BOUNDARY_SEQ))
  assert _row(result, "above_high").count == 0
  assert _row(result, "below_low").count == 0
  assert _row(result, "inside_range").count == 2


# ===========================================================================
# 3. First-day exclusion / total_samples discipline
#
# The first resolved day has no prior session → excluded from build_day_table.
# total_samples (in TimeframeResult) == len(day_table) == resolved_days - 1.
#
# Using _CLASSIFICATION_SEQ: 4 resolved days → total_samples = 3.
# Using a 1-day input: total_samples = 0.
# Using a 2-day input: 1 resolved excluded, 1 countable → total_samples = 1.
# ===========================================================================

def test_first_day_excluded_from_day_table() -> None:
  """The anchor day (no prev session) must not appear in build_day_table."""
  stat = _stat()
  day_table = stat.build_day_table(make_candles(_CLASSIFICATION_SEQ))
  anchor = pd.Timestamp("2024-01-02", tz=_NY).normalize()
  assert anchor not in day_table.index


def test_total_samples_equals_resolved_minus_one() -> None:
  """total_samples == 3 for a 4-resolved-day sequence."""
  # 4 resolved days in _CLASSIFICATION_SEQ → total_samples = 4 - 1 = 3
  result = _stat().compute(make_candles(_CLASSIFICATION_SEQ))
  assert result.instruments["NQ"]["daily"].total_samples == 3


def test_row_total_equals_total_samples() -> None:
  """Each row's total equals total_samples (single condition, marginal distribution)."""
  result = _stat().compute(make_candles(_CLASSIFICATION_SEQ))
  ts = result.instruments["NQ"]["daily"].total_samples
  for out_key in ("above_high", "inside_range", "below_low"):
    assert _row(result, out_key).total == ts


def test_two_resolved_days_yields_one_countable() -> None:
  """2 resolved days → 1 countable → total_samples=1, one outcome count=1."""
  days = [
    {"date": "2024-03-01", "open": 100.0, "close": 105.0, "high": 108.0, "low": 98.0},
    # open=120 > prev_high=108 → above_high
    {"date": "2024-03-04", "open": 120.0, "close": 115.0, "high": 125.0, "low": 110.0},
  ]
  result = _stat().compute(make_candles(days))
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 1
  assert _row(result, "above_high").count == 1
  assert _row(result, "inside_range").count == 0
  assert _row(result, "below_low").count == 0


# ===========================================================================
# 4. Probabilities sum to 1.0
#
# The three outcomes are mutually exclusive and exhaustive for a countable day,
# so their probabilities must sum to exactly 1.0 (within float tolerance).
# We test this for the overall result and for each non-empty weekday slice group.
# ===========================================================================

def test_probabilities_sum_to_one_overall() -> None:
  """above_high + inside_range + below_low probabilities must sum to 1.0."""
  result = _stat().compute(make_candles(_CLASSIFICATION_SEQ))
  total_prob = sum(_row(result, out).probability for out in ("above_high", "inside_range", "below_low"))
  assert total_prob == pytest.approx(1.0)


def _make_weekday_seq() -> list[dict]:
  """10 days across Mon–Fri (2 per weekday) with diverse open locations.

  Spans the weeks of 2024-01-08 (Mon) and 2024-01-15 (Mon). Each measured day's
  location is read against the PRIOR resolved session's RTH range, giving a 3/3/3
  split across above_high / inside_range / below_low:
    Tue1 (01-09): open=115 > prev_high=110          → above_high
    Wed1 (01-10): open=108 inside [95, 120]         → inside_range
    Thu1 (01-11): open=88  < prev_low=92            → below_low
    Fri1 (01-12): open=106 > prev_high=103          → above_high
    Mon2 (01-15): open=105 inside [100, 112]        → inside_range
    Tue2 (01-16): open=84  < prev_low=90            → below_low
    Wed2 (01-17): open=110 > prev_high=92           → above_high
    Thu2 (01-18): open=100 inside [98, 115]         → inside_range
    Fri2 (01-19): open=85  < prev_low=94            → below_low
  Mon1 (01-08) is the anchor (no prior session, excluded).
  """
  return [
    # Mon1 anchor: high=110, low=90.
    {"date": "2024-01-08", "open": 100.0, "close": 105.0, "high": 110.0, "low": 90.0},
    # Tue1: prev=Mon1 [90, 110]; open=115 > 110 → above_high.
    {"date": "2024-01-09", "open": 115.0, "close": 108.0, "high": 120.0, "low": 95.0},
    # Wed1: prev=Tue1 [95, 120]; 95 ≤ 108 ≤ 120 → inside_range.
    {"date": "2024-01-10", "open": 108.0, "close": 112.0, "high": 118.0, "low": 92.0},
    # Thu1: prev=Wed1 [92, 118]; open=88 < 92 → below_low.
    {"date": "2024-01-11", "open": 88.0, "close": 95.0, "high": 103.0, "low": 88.0},
    # Fri1: prev=Thu1 [88, 103]; open=106 > 103 → above_high.
    {"date": "2024-01-12", "open": 106.0, "close": 109.0, "high": 112.0, "low": 100.0},
    # Mon2: prev=Fri1 [100, 112]; 100 ≤ 105 ≤ 112 → inside_range.
    {"date": "2024-01-15", "open": 105.0, "close": 98.0, "high": 109.0, "low": 90.0},
    # Tue2: prev=Mon2 [90, 109]; open=84 < 90 → below_low.
    {"date": "2024-01-16", "open": 84.0, "close": 88.0, "high": 92.0, "low": 82.0},
    # Wed2: prev=Tue2 [82, 92]; open=110 > 92 → above_high.
    {"date": "2024-01-17", "open": 110.0, "close": 105.0, "high": 115.0, "low": 98.0},
    # Thu2: prev=Wed2 [98, 115]; 98 ≤ 100 ≤ 115 → inside_range.
    {"date": "2024-01-18", "open": 100.0, "close": 103.0, "high": 108.0, "low": 94.0},
    # Fri2: prev=Thu2 [94, 108]; open=85 < 94 → below_low.
    {"date": "2024-01-19", "open": 85.0, "close": 90.0, "high": 95.0, "low": 82.0},
  ]


def test_probabilities_sum_to_one_per_weekday_group() -> None:
  """In every non-empty weekday slice group, the three outcome probabilities sum to 1.0."""
  result = _stat().compute(make_candles(_make_weekday_seq()))
  groups = result.instruments["NQ"]["daily"].slices["weekday"].groups
  assert len(groups) > 0, "Expected at least one weekday group"
  for day_key, group in groups.items():
    total_prob = sum(r.probability for r in group.results)
    assert total_prob == pytest.approx(1.0), (
      f"weekday={day_key}: probabilities sum to {total_prob}, expected 1.0"
    )


# ===========================================================================
# 5. Pending / unresolved day discipline
#
# A truncated day (last bar at 09:50, mod=590 < 960) must be excluded from
# all computations. Adding it after a full sequence must not change
# total_samples, open_location counts, or any probabilities.
# ===========================================================================

def test_pending_day_absent_from_day_table() -> None:
  """A truncated day must not appear in build_day_table's index."""
  stat = _stat()
  day_table = stat.build_day_table(_make_truncated_day("2024-01-15"))
  pending = pd.Timestamp("2024-01-15", tz=_NY).normalize()
  assert pending not in day_table.index


def test_pending_day_does_not_increase_total_samples() -> None:
  """Adding a truncated final day must leave total_samples unchanged."""
  base_df = make_candles(_CLASSIFICATION_SEQ)
  base_result = _stat().compute(base_df)
  base_total = base_result.instruments["NQ"]["daily"].total_samples

  truncated = _make_truncated_day("2024-01-08")
  combined = (
    pd.concat([base_df, truncated], ignore_index=True)
    .sort_values("timestamp")
    .reset_index(drop=True)
  )
  extended_result = _stat().compute(combined)
  extended_total = extended_result.instruments["NQ"]["daily"].total_samples

  assert extended_total == base_total == 3


def test_pending_day_does_not_affect_outcome_counts() -> None:
  """Outcome counts must be identical with and without a trailing truncated day."""
  base_df = make_candles(_CLASSIFICATION_SEQ)
  base_result = _stat().compute(base_df)

  truncated = _make_truncated_day("2024-01-08")
  combined = (
    pd.concat([base_df, truncated], ignore_index=True)
    .sort_values("timestamp")
    .reset_index(drop=True)
  )
  extended_result = _stat().compute(combined)

  for out_key in ("above_high", "inside_range", "below_low"):
    base_r = _row(base_result, out_key)
    ext_r = _row(extended_result, out_key)
    assert base_r.count == ext_r.count, f"{out_key}: count changed"
    assert base_r.total == ext_r.total, f"{out_key}: total changed"


# ===========================================================================
# 6. Reproducibility / determinism
#
# Two compute() calls on the same input must return identical results.
# Baseline must be ≈ 1/3 per outcome on a large sample (N ≥ 300, tolerance ±0.05).
# Different seeds must yield different probabilities.
# ===========================================================================

def test_compute_is_deterministic() -> None:
  """Two compute() calls on the same input yield identical results."""
  df = make_candles(_CLASSIFICATION_SEQ)
  result_a = _stat().compute(df)
  result_b = _stat().compute(df)
  for out_key in ("above_high", "inside_range", "below_low"):
    a = _row(result_a, out_key)
    b = _row(result_b, out_key)
    assert a.count == b.count
    assert a.total == b.total
    assert a.probability == pytest.approx(b.probability)
    assert a.baseline_prob == pytest.approx(b.baseline_prob)


def test_baseline_is_deterministic_same_seed() -> None:
  """Same seed produces identical baseline probabilities."""
  stat = _stat()
  df = make_candles(_CLASSIFICATION_SEQ)
  rows_a = stat.baseline(df, seed=42)
  rows_b = stat.baseline(df, seed=42)
  for a, b in zip(rows_a, rows_b):
    assert (a.condition, a.outcome) == (b.condition, b.outcome)
    assert a.probability == pytest.approx(b.probability)


def _make_large_seq(n_days: int = 300) -> list[dict]:
  """Build N+1 consecutive weekday sessions with alternating open locations.

  Day 0 is an anchor. Days 1..n_days cycle through above/inside/below so
  the true distribution is balanced. The anchor has high=110, low=90; each
  subsequent day uses that day's high/low so the classification is controlled
  via the open price only.
  """
  days: list[dict] = []
  d = pd.Timestamp("2020-01-02", tz="America/New_York")
  anchor_high = 110.0
  anchor_low = 90.0
  days.append({
    "date": d.strftime("%Y-%m-%d"),
    "open": 100.0,
    "close": 102.0,
    "high": anchor_high,
    "low": anchor_low,
  })
  prev_high = anchor_high
  prev_low = anchor_low
  i = 0
  while len(days) <= n_days:
    d += pd.Timedelta(days=1)
    if d.weekday() >= 5:
      continue
    cycle = i % 3
    if cycle == 0:
      # above_high: open above prev_high
      o = prev_high + 5.0
    elif cycle == 1:
      # inside_range: open at midpoint
      o = (prev_high + prev_low) / 2.0
    else:
      # below_low: open below prev_low
      o = prev_low - 5.0
    new_high = o + 10.0
    new_low = o - 10.0
    days.append({
      "date": d.strftime("%Y-%m-%d"),
      "open": o,
      "close": o + 2.0,
      "high": new_high,
      "low": new_low,
    })
    prev_high = new_high
    prev_low = new_low
    i += 1
  return days


def test_baseline_approx_one_third_per_outcome() -> None:
  """Over 300 countable days with seed=42, each baseline outcome ≈ 1/3 (±0.05)."""
  stat = _stat()
  df = make_candles(_make_large_seq(300))
  rows = stat.baseline(df, seed=42)
  for row in rows:
    assert row.probability == pytest.approx(1 / 3, abs=0.05), (
      f"outcome={row.outcome}: baseline_prob={row.probability:.4f}, expected ≈1/3"
    )


def test_baseline_different_seeds_differ() -> None:
  """Different seeds produce different baseline probabilities on a larger dataset."""
  stat = _stat()
  df = make_candles(_make_large_seq(60))
  rows_42 = stat.baseline(df, seed=42)
  rows_99 = stat.baseline(df, seed=99)
  probs_42 = [r.probability for r in rows_42]
  probs_99 = [r.probability for r in rows_99]
  all_same = all(abs(a - b) < 1e-9 for a, b in zip(probs_42, probs_99))
  assert not all_same, "Different seeds should produce different baseline probabilities"


def test_baseline_embedded_in_compute_rows() -> None:
  """After compute(), each result row has a positive baseline_n (baseline ran)."""
  result = _stat().compute(make_candles(_CLASSIFICATION_SEQ))
  for row in result.instruments["NQ"]["daily"].results:
    if row.total > 0:
      assert row.baseline_n > 0


# ===========================================================================
# 7. Empty input edge cases
#
# Empty DataFrame → build_day_table returns empty frame with expected columns.
# compute_rows on empty table → all three outcome rows with total=0, prob=0.0.
# Single resolved day → excluded (no prior) → total_samples=0, all rows zeroed.
# All-above dataset → below_low.count=0, above_high.count == total_samples.
# All-below dataset → above_high.count=0, below_low.count == total_samples.
# ===========================================================================

def test_empty_dataframe_build_day_table_columns() -> None:
  """Empty input → build_day_table returns empty frame with expected columns."""
  stat = _stat()
  day_table = stat.build_day_table(_empty_df())
  assert day_table.empty
  for col in ("session_open", "prev_high", "prev_low", "open_location", "session_green", "prev_session_green"):
    assert col in day_table.columns, f"Missing column: {col}"


def test_empty_dataframe_compute_three_zero_rows() -> None:
  """Empty input → three outcome rows all with total=0, count=0, probability=0.0."""
  result = _stat().compute(_empty_df())
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 0
  assert tf.data_range == []
  for out_key in ("above_high", "inside_range", "below_low"):
    r = _row(result, out_key)
    assert r.count == 0
    assert r.total == 0
    assert r.probability == pytest.approx(0.0)


def test_empty_dataframe_three_rows_present() -> None:
  """Even with no data, all three outcome rows are emitted."""
  result = _stat().compute(_empty_df())
  outcomes = {(r.condition, r.outcome) for r in result.instruments["NQ"]["daily"].results}
  assert outcomes == {
    ("open", "above_high"),
    ("open", "inside_range"),
    ("open", "below_low"),
  }


def test_single_resolved_day_total_samples_zero() -> None:
  """One resolved session → no prior session → total_samples=0, all rows zeroed."""
  days = [{"date": "2024-03-01", "open": 100.0, "close": 110.0, "high": 115.0, "low": 95.0}]
  result = _stat().compute(make_candles(days))
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 0
  for r in tf.results:
    assert r.total == 0
    assert r.probability == pytest.approx(0.0)


def test_all_above_high() -> None:
  """All opens above prior high → above_high.count == total_samples, below_low=0, inside=0."""
  # Anchor: high=105, low=95; each subsequent opens above the prior high.
  days = [
    {"date": "2024-01-02", "open": 100.0, "close": 105.0, "high": 110.0, "low": 95.0},
    {"date": "2024-01-03", "open": 115.0, "close": 112.0, "high": 120.0, "low": 108.0},
    {"date": "2024-01-04", "open": 125.0, "close": 122.0, "high": 130.0, "low": 118.0},
    {"date": "2024-01-05", "open": 135.0, "close": 132.0, "high": 140.0, "low": 128.0},
  ]
  result = _stat().compute(make_candles(days))
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 3
  assert _row(result, "above_high").count == 3
  assert _row(result, "inside_range").count == 0
  assert _row(result, "below_low").count == 0
  assert _row(result, "above_high").probability == pytest.approx(1.0)


def test_all_below_low() -> None:
  """All opens below prior low → below_low.count == total_samples, above=0, inside=0."""
  days = [
    {"date": "2024-01-02", "open": 100.0, "close": 95.0, "high": 105.0, "low": 90.0},
    {"date": "2024-01-03", "open": 85.0, "close": 82.0, "high": 88.0, "low": 78.0},
    {"date": "2024-01-04", "open": 73.0, "close": 70.0, "high": 76.0, "low": 66.0},
    {"date": "2024-01-05", "open": 61.0, "close": 58.0, "high": 64.0, "low": 54.0},
  ]
  result = _stat().compute(make_candles(days))
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 3
  assert _row(result, "below_low").count == 3
  assert _row(result, "above_high").count == 0
  assert _row(result, "inside_range").count == 0
  assert _row(result, "below_low").probability == pytest.approx(1.0)


def test_all_inside_range() -> None:
  """All opens inside prior range → inside_range.count == total_samples."""
  # Each open lands exactly at the midpoint of the prior range.
  days = [
    {"date": "2024-01-02", "open": 100.0, "close": 100.0, "high": 120.0, "low": 80.0},
    # prev_high=120, prev_low=80; open=100 → inside
    {"date": "2024-01-03", "open": 100.0, "close": 100.0, "high": 115.0, "low": 85.0},
    # prev_high=115, prev_low=85; open=100 → inside
    {"date": "2024-01-04", "open": 100.0, "close": 100.0, "high": 110.0, "low": 90.0},
  ]
  result = _stat().compute(make_candles(days))
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 2
  assert _row(result, "inside_range").count == 2
  assert _row(result, "above_high").count == 0
  assert _row(result, "below_low").count == 0


# ===========================================================================
# 8. Slices present and correct
#
# OpeningStats declares slices: ("weekday", "prev_candle", "close")
# All three must appear in the result slices dict.
# Group probabilities within each slice must also sum to 1.0.
# ===========================================================================

def test_slice_keys_present() -> None:
  """Result must carry all three declared slice dimensions: weekday, prev_candle, close."""
  result = _stat().compute(make_candles(_make_weekday_seq()))
  slices = result.instruments["NQ"]["daily"].slices
  assert set(slices.keys()) == {"weekday", "prev_candle", "close"}


def test_weekday_slice_present_and_non_empty() -> None:
  """weekday slice must produce at least one group."""
  result = _stat().compute(make_candles(_CLASSIFICATION_SEQ))
  groups = result.instruments["NQ"]["daily"].slices["weekday"].groups
  assert len(groups) > 0


def test_prev_candle_slice_present_and_non_empty() -> None:
  """prev_candle slice must produce at least one group (green or red)."""
  result = _stat().compute(make_candles(_CLASSIFICATION_SEQ))
  groups = result.instruments["NQ"]["daily"].slices["prev_candle"].groups
  assert len(groups) > 0


def test_close_slice_present_and_non_empty() -> None:
  """close slice must produce at least one group (green or red)."""
  result = _stat().compute(make_candles(_CLASSIFICATION_SEQ))
  groups = result.instruments["NQ"]["daily"].slices["close"].groups
  assert len(groups) > 0


def test_weekday_slice_group_probabilities_sum_to_one() -> None:
  """Each non-empty weekday group must have outcome probabilities summing to 1.0."""
  result = _stat().compute(make_candles(_make_weekday_seq()))
  groups = result.instruments["NQ"]["daily"].slices["weekday"].groups
  for day_key, group in groups.items():
    s = sum(r.probability for r in group.results)
    assert s == pytest.approx(1.0), f"weekday={day_key}: {s} != 1.0"


def test_close_slice_group_probabilities_sum_to_one() -> None:
  """Each non-empty close slice group must have outcome probabilities summing to 1.0."""
  result = _stat().compute(make_candles(_CLASSIFICATION_SEQ))
  groups = result.instruments["NQ"]["daily"].slices["close"].groups
  for grp_key, group in groups.items():
    s = sum(r.probability for r in group.results)
    assert s == pytest.approx(1.0), f"close={grp_key}: {s} != 1.0"


def test_prev_candle_slice_group_probabilities_sum_to_one() -> None:
  """Each non-empty prev_candle slice group must have outcome probabilities summing to 1.0."""
  result = _stat().compute(make_candles(_CLASSIFICATION_SEQ))
  groups = result.instruments["NQ"]["daily"].slices["prev_candle"].groups
  for grp_key, group in groups.items():
    s = sum(r.probability for r in group.results)
    assert s == pytest.approx(1.0), f"prev_candle={grp_key}: {s} != 1.0"


def test_close_slice_has_green_and_red_groups() -> None:
  """close slice must have both green and red groups for _CLASSIFICATION_SEQ."""
  # _CLASSIFICATION_SEQ: day1 close=104>106 → red; day2 close=107>100 → green;
  # day3 close=108>90 → green. Both colors present.
  result = _stat().compute(make_candles(_CLASSIFICATION_SEQ))
  groups = result.instruments["NQ"]["daily"].slices["close"].groups
  assert "green" in groups
  assert "red" in groups


def test_prev_candle_slice_values() -> None:
  """Verify the prev_session_green column is correctly populated in the day table.

  In _CLASSIFICATION_SEQ:
    idx 0 (anchor): session_green = (102 >= 100) = True
    idx 1: prev_session_green = True (anchor was green); session_green = (104 >= 106) = False
    idx 2: prev_session_green = False (idx1 was red);   session_green = (107 >= 100) = True
    idx 3: prev_session_green = True  (idx2 was green); session_green = (108 >= 90) = True
  """
  stat = _stat()
  day_table = stat.build_day_table(make_candles(_CLASSIFICATION_SEQ)).sort_index()

  d1 = pd.Timestamp("2024-01-03", tz=_NY).normalize()
  d2 = pd.Timestamp("2024-01-04", tz=_NY).normalize()
  d3 = pd.Timestamp("2024-01-05", tz=_NY).normalize()

  # idx 1: anchor was green → prev_session_green=True; close(104)<open(106) → red
  assert bool(day_table.loc[d1, "prev_session_green"]) is True
  assert bool(day_table.loc[d1, "session_green"]) is False

  # idx 2: idx1 was red → prev_session_green=False; close(107)>open(100) → green
  assert bool(day_table.loc[d2, "prev_session_green"]) is False
  assert bool(day_table.loc[d2, "session_green"]) is True

  # idx 3: idx2 was green → prev_session_green=True; close(108)>open(90) → green
  assert bool(day_table.loc[d3, "prev_session_green"]) is True
  assert bool(day_table.loc[d3, "session_green"]) is True


def test_weekday_slice_dimension_label_present() -> None:
  """labels.dimensions must contain 'weekday' with non-empty en and fr."""
  result = _stat().compute(make_candles(_CLASSIFICATION_SEQ))
  assert "weekday" in result.labels.dimensions
  assert result.labels.dimensions["weekday"].en != ""
  assert result.labels.dimensions["weekday"].fr != ""


# ===========================================================================
# 9. Larger sequence: known counts for a mixed dataset
#
# A deterministic 10-day sequence with a balanced 3/3/3 split. Each day's own
# high/low define the prior range read by the NEXT day, so the outcomes are
# fully controlled. idx 0 is the anchor (no prior session, excluded), leaving
# 9 countable days:
#
#   idx  date        open    prev range      outcome
#    0   2024-03-04  100.00  —               anchor (excluded)
#    1   2024-03-05  115.00  [ 90, 110]      above_high (115 > 110)
#    2   2024-03-06  110.00  [100, 120]      inside_range (100 ≤ 110 ≤ 120)
#    3   2024-03-07   95.00  [ 98, 125]      below_low (95 < 98)
#    4   2024-03-08  112.00  [ 88, 105]      above_high (112 > 105)
#    5   2024-03-11  110.00  [104, 118]      inside_range (104 ≤ 110 ≤ 118)
#    6   2024-03-12   98.00  [102, 115]      below_low (98 < 102)
#    7   2024-03-13  115.00  [ 92, 108]      above_high (115 > 108)
#    8   2024-03-14  112.00  [108, 120]      inside_range (108 ≤ 112 ≤ 120)
#    9   2024-03-15  104.00  [106, 122]      below_low (104 < 106)
#
# above_high: idx 1, 4, 7 → 3 | inside_range: idx 2, 5, 8 → 3 | below_low: idx 3, 6, 9 → 3
# ===========================================================================

_MIXED_SEQ = [
  {"date": "2024-03-04", "open": 100.00, "close": 102.00, "high": 110.00, "low":  90.00},
  {"date": "2024-03-05", "open": 115.00, "close": 110.00, "high": 120.00, "low": 100.00},
  {"date": "2024-03-06", "open": 110.00, "close": 105.00, "high": 125.00, "low":  98.00},
  {"date": "2024-03-07", "open":  95.00, "close":  92.00, "high": 105.00, "low":  88.00},
  {"date": "2024-03-08", "open": 112.00, "close": 108.00, "high": 118.00, "low": 104.00},
  {"date": "2024-03-11", "open": 110.00, "close": 108.00, "high": 115.00, "low": 102.00},
  {"date": "2024-03-12", "open":  98.00, "close":  95.00, "high": 108.00, "low":  92.00},
  {"date": "2024-03-13", "open": 115.00, "close": 112.00, "high": 120.00, "low": 108.00},
  {"date": "2024-03-14", "open": 112.00, "close": 110.00, "high": 122.00, "low": 106.00},
  {"date": "2024-03-15", "open": 104.00, "close": 100.00, "high": 110.00, "low":  98.00},
]


def test_mixed_seq_total_samples() -> None:
  """_MIXED_SEQ: 10 resolved days → 9 countable → total_samples=9."""
  result = _stat().compute(make_candles(_MIXED_SEQ))
  assert result.instruments["NQ"]["daily"].total_samples == 9


def test_mixed_seq_above_high_count() -> None:
  """_MIXED_SEQ: idx 1(115>110), idx 4(112>105), idx 7(115>108) → above_high count=3.

  Hand-check:
    idx 1: open=115, prev_high=110 → above (115>110) ✓
    idx 4: open=112, prev_high=105 → above (112>105) ✓
    idx 7: open=115, prev_high=108 → above (115>108) ✓
  """
  result = _stat().compute(make_candles(_MIXED_SEQ))
  r = _row(result, "above_high")
  assert r.count == 3
  assert r.probability == pytest.approx(3 / 9)


def test_mixed_seq_below_low_count() -> None:
  """_MIXED_SEQ: idx 3(95<98), idx 6(98<102), idx 9(104<106) → below_low count=3.

  Hand-check:
    idx 3: open=95,  prev_low=98  → below (95<98)   ✓
    idx 6: open=98,  prev_low=102 → below (98<102)  ✓
    idx 9: open=104, prev_low=106 → below (104<106) ✓
  """
  result = _stat().compute(make_candles(_MIXED_SEQ))
  r = _row(result, "below_low")
  assert r.count == 3
  assert r.probability == pytest.approx(3 / 9)


def test_mixed_seq_inside_range_count() -> None:
  """_MIXED_SEQ: idx 2,5,8 → inside_range count=3.

  Hand-check:
    idx 2: open=110, prev_high=120, prev_low=100 → inside (100≤110≤120) ✓
    idx 5: open=110, prev_high=118, prev_low=104 → inside (104≤110≤118) ✓
    idx 8: open=112, prev_high=120, prev_low=108 → inside (108≤112≤120) ✓
  """
  result = _stat().compute(make_candles(_MIXED_SEQ))
  r = _row(result, "inside_range")
  assert r.count == 3
  assert r.probability == pytest.approx(3 / 9)


def test_mixed_seq_counts_partition() -> None:
  """above + inside + below == total_samples for _MIXED_SEQ."""
  result = _stat().compute(make_candles(_MIXED_SEQ))
  ts = result.instruments["NQ"]["daily"].total_samples
  total_count = sum(_row(result, out).count for out in ("above_high", "inside_range", "below_low"))
  assert total_count == ts == 9


def test_data_range_spans_first_to_last_date() -> None:
  """data_range reflects the first and last dates in the day table (countable days)."""
  result = _stat().compute(make_candles(_CLASSIFICATION_SEQ))
  # Countable days: 2024-01-03 to 2024-01-05
  assert result.instruments["NQ"]["daily"].data_range == ["2024-01-03", "2024-01-05"]


# ===========================================================================
# 10. Stat metadata and i18n
# ===========================================================================

def test_stat_name() -> None:
  """stat_name must be 'opening_stats'."""
  result = _stat().compute(_empty_df())
  assert result.stat_name == "opening_stats"


def test_i18n_title_and_definition() -> None:
  """title and definition must carry non-empty en and fr strings."""
  result = _stat().compute(_empty_df())
  assert result.title.en != "" and result.title.fr != ""
  assert result.definition.en != "" and result.definition.fr != ""


def test_i18n_labels_have_en_and_fr() -> None:
  """Every condition and outcome label must have non-empty en and fr."""
  result = _stat().compute(_empty_df())
  for mapping in (result.labels.conditions, result.labels.outcomes):
    for key, i18n in mapping.items():
      assert i18n.en != "", f"{key}.en empty"
      assert i18n.fr != "", f"{key}.fr empty"
  assert set(result.labels.conditions) == {"open"}
  assert set(result.labels.outcomes) == {"above_high", "inside_range", "below_low"}


def test_labels_dimensions_contain_declared_slicers() -> None:
  """labels.dimensions must contain entries for weekday, prev_candle, and close."""
  result = _stat().compute(make_candles(_CLASSIFICATION_SEQ))
  assert set(result.labels.dimensions.keys()) >= {"weekday", "prev_candle", "close"}
  for key in ("weekday", "prev_candle", "close"):
    assert result.labels.dimensions[key].en != "", f"dimensions[{key}].en empty"
    assert result.labels.dimensions[key].fr != "", f"dimensions[{key}].fr empty"


# ===========================================================================
# 11. write_results round-trip via Pydantic
# ===========================================================================

def test_write_results_round_trip(tmp_path: Path) -> None:
  """write_results produces opening_stats.json that re-validates as StatRunResult."""
  result = _stat().compute(make_candles(_MIXED_SEQ))
  written = write_results(result, results_dir=tmp_path)
  assert written.name == "opening_stats.json"

  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)
  tf = validated.instruments["NQ"]["daily"]
  assert tf.total_samples == 9
  # Spot-check above_high row
  above = next(r for r in tf.results if r.condition == "open" and r.outcome == "above_high")
  assert above.count == 3
  assert above.total == 9
  assert above.probability == pytest.approx(3 / 9)


def test_write_results_has_all_slices(tmp_path: Path) -> None:
  """Serialised JSON contains all three declared slice dimensions."""
  result = _stat().compute(make_candles(_CLASSIFICATION_SEQ))
  written = write_results(result, results_dir=tmp_path)
  raw = json.loads(written.read_text(encoding="utf-8"))
  slices = raw["instruments"]["NQ"]["daily"]["slices"]
  assert set(slices.keys()) == {"weekday", "prev_candle", "close"}


def test_write_results_utf8_literals(tmp_path: Path) -> None:
  """French text is stored as literal UTF-8 characters, not escaped unicode."""
  result = _stat().compute(_empty_df())
  written = write_results(result, results_dir=tmp_path)
  raw = written.read_text(encoding="utf-8")
  # French definition contains accented characters
  assert "é" in raw
  assert "\\u00e9" not in raw
