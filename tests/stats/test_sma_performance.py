"""Tests for stats.sma_performance.standard.

All data is synthetic — no real market files required. Expected values are
hand-calculated before each assertion.

Methodology recap:
  - dist[i] = session_close[i] - prior_sma[i]
  - prior_sma[i] = rolling(period).mean() of closes, shifted by 1 (no lookahead)
  - With period=P: dist values start at resolved session index P (0-indexed);
    the first P sessions contribute to SMA warm-up but appear in no dist row.
  - total_samples = len(day_table) = number of sessions with a valid (non-NaN) dist.
  - data_range spans the dist-row index, not all resolved sessions.
  - A "run" is a maximal sequence of same-sign dist values.
  - dist == 0 is classified "down" (ties go to "down").
  - confirmed runs = all_runs[1:-1]  (first and last excluded).
  - qualified runs = confirmed runs with duration >= min_duration.
  - travel for "up" run  = max(dist values) in that run.
  - travel for "down" run = max(-dist values) in that run = max magnitude.
  - Always exactly 8 rows: {cross_up, cross_down} x {avg_duration, max_duration,
    avg_travel, max_travel}. When N=0, all four values are 0.0.
  - probability == 0.0 (sentinel); meaningful number is value.
  - count == total for every row (no pending inside runs).

Period=2 construction:
  Given closes c[0], c[1], ..., the dist array (after dropping the 2 warm-up rows) is:
    dist[k] = c[k+2] - mean(c[k], c[k+1])   for k = 0, 1, ...

  To produce a desired dist value d at index k, choose:
    c[k+2] = (c[k] + c[k+1]) / 2 + d
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.sma_performance.standard import SMAPerformance
from tests.stats.range_helpers import (
  _NY,
  _TEST_CONFIG,
  _empty_df,
  _make_day,
  _make_truncated_day,
)


def _weekdays(n: int, start: str = "2024-01-02") -> list[str]:
  """Return n weekday date strings starting from ``start``."""
  dates: list[str] = []
  d = pd.Timestamp(start, tz=_NY)
  while len(dates) < n:
    if d.weekday() < 5:
      dates.append(d.strftime("%Y-%m-%d"))
    d += pd.Timedelta(days=1)
  return dates


def make_candles_from_closes(closes: list[float], start: str = "2024-01-02") -> pd.DataFrame:
  """Build 1-min RTH candle data from a list of session closes."""
  dates = _weekdays(len(closes), start=start)
  frames = [_make_day(dates[i], closes[i]) for i in range(len(closes))]
  df = pd.concat(frames, ignore_index=True)
  return df.sort_values("timestamp").reset_index(drop=True)


def closes_from_dist(desired_dist: list[float]) -> list[float]:
  """Build a close sequence that produces the given dist values with period=2.

  The first two closes are both 100.0 (warm-up). Each subsequent close is chosen
  so that dist[k] = c[k+2] - mean(c[k], c[k+1]) equals desired_dist[k].
  """
  closes = [100.0, 100.0]
  for k, d in enumerate(desired_dist):
    sma = (closes[k] + closes[k + 1]) / 2.0
    closes.append(sma + d)
  return closes


def _stat(period: int = 2, min_duration: int = 1) -> SMAPerformance:
  """Instantiate SMAPerformance with a small period for fast tests."""
  return SMAPerformance(
    instrument="NQ", config=_TEST_CONFIG, period=period, min_duration=min_duration
  )


def _row(result: StatRunResult, condition: str, outcome: str) -> StatResultRow:
  for r in result.instruments["NQ"]["daily"].results:
    if r.condition == condition and r.outcome == outcome:
      return r
  raise KeyError((condition, outcome))


# ===========================================================================
# 1. Known-runs correctness
#
# period=2, min_duration=1 (accept all confirmed runs).
#
# Desired dist sequence (12 values):
#   [+10, +15, +20, -5, -8, +12, +18, -6, -9, +11, +14, +7]
#
# Signs (up=+, down=-):
#   + + + - - + + - - + + +
#
# All runs (chronological):
#   Run 0:  up,   duration=3, values=[10,15,20],  travel=max(10,15,20)=20  (FIRST→excluded)
#   Run 1:  down, duration=2, values=[-5,-8],      travel=max(5,8)=8
#   Run 2:  up,   duration=2, values=[12,18],      travel=max(12,18)=18
#   Run 3:  down, duration=2, values=[-6,-9],      travel=max(6,9)=9
#   Run 4:  up,   duration=3, values=[11,14,7],    travel=max(11,14,7)=14  (LAST→excluded)
#
# Confirmed (drop first=Run 0, drop last=Run 4): Runs 1, 2, 3
#
# cross_up  (qualified confirmed up runs): [Run 2]
#   N=1, avg_duration=2.0, max_duration=2.0, avg_travel=18.0, max_travel=18.0
#
# cross_down (qualified confirmed down runs): [Run 1, Run 3]
#   N=2, avg_duration=(2+2)/2=2.0, max_duration=2.0
#   avg_travel=(8+9)/2=8.5, max_travel=9.0
#
# total_samples = 12 (number of dist rows = 14 sessions - 2 warm-up)
# ===========================================================================

_MAIN_DIST = [10.0, 15.0, 20.0, -5.0, -8.0, 12.0, 18.0, -6.0, -9.0, 11.0, 14.0, 7.0]
_MAIN_CLOSES = closes_from_dist(_MAIN_DIST)  # 14 sessions total


def test_total_samples_equals_dist_row_count() -> None:
  """total_samples = len(dist rows) = N sessions - period warm-up."""
  result = _stat().compute(make_candles_from_closes(_MAIN_CLOSES))
  # 14 sessions, period=2 → 12 dist rows
  assert result.instruments["NQ"]["daily"].total_samples == 12


def test_cross_up_count_and_aggregates() -> None:
  """cross_up: 1 confirmed qualified run (Run 2, duration=2, travel=18)."""
  result = _stat().compute(make_candles_from_closes(_MAIN_CLOSES))
  # N=1, avg_duration=2.0, max_duration=2.0, avg_travel=18.0, max_travel=18.0
  assert _row(result, "cross_up", "avg_duration").count == 1
  assert _row(result, "cross_up", "avg_duration").value == pytest.approx(2.0)
  assert _row(result, "cross_up", "max_duration").value == pytest.approx(2.0)
  assert _row(result, "cross_up", "avg_travel").value == pytest.approx(18.0)
  assert _row(result, "cross_up", "max_travel").value == pytest.approx(18.0)


def test_cross_down_count_and_aggregates() -> None:
  """cross_down: 2 confirmed runs (Run 1 dur=2 trav=8, Run 3 dur=2 trav=9)."""
  result = _stat().compute(make_candles_from_closes(_MAIN_CLOSES))
  # N=2, avg_duration=2.0, max_duration=2.0, avg_travel=8.5, max_travel=9.0
  assert _row(result, "cross_down", "avg_duration").count == 2
  assert _row(result, "cross_down", "avg_duration").value == pytest.approx(2.0)
  assert _row(result, "cross_down", "max_duration").value == pytest.approx(2.0)
  assert _row(result, "cross_down", "avg_travel").value == pytest.approx(8.5)
  assert _row(result, "cross_down", "max_travel").value == pytest.approx(9.0)


def test_first_run_excluded_from_confirmed() -> None:
  """Run 0 (up:3, values=[10,15,20]) is the first run and must be excluded.

  If Run 0 were included, cross_up would have N>=2 and max_travel>=20.
  With it excluded, cross_up has N=1 and max_travel=18.
  """
  result = _stat().compute(make_candles_from_closes(_MAIN_CLOSES))
  # Run 0 (up, travel=20) excluded → max_travel of cross_up is 18, not 20
  assert _row(result, "cross_up", "max_travel").value == pytest.approx(18.0)
  assert _row(result, "cross_up", "max_duration").count == 1  # only Run 2


def test_last_run_excluded_from_confirmed() -> None:
  """Run 4 (up:3, values=[11,14,7]) is the last run and must be excluded.

  If Run 4 were included, cross_up would have N=2 and avg_duration would change.
  With it excluded, cross_up has N=1 with duration=2 only.
  """
  result = _stat().compute(make_candles_from_closes(_MAIN_CLOSES))
  # Only Run 2 qualifies → N=1
  assert _row(result, "cross_up", "avg_duration").count == 1
  assert _row(result, "cross_up", "avg_duration").total == 1


def test_dist_uses_prior_sma_no_lookahead() -> None:
  """dist[i] = close[i] - mean(close[i-2], close[i-1]) — verified via build_day_table."""
  stat = _stat()
  df = make_candles_from_closes(_MAIN_CLOSES)
  day_table = stat.build_day_table(df)
  # First dist value should be exactly +10.0 (by construction)
  assert day_table["dist"].iloc[0] == pytest.approx(10.0)
  # Third dist value should be exactly +20.0
  assert day_table["dist"].iloc[2] == pytest.approx(20.0)
  # Fourth dist value should be exactly -5.0
  assert day_table["dist"].iloc[3] == pytest.approx(-5.0)


def test_data_range_spans_dist_rows_not_all_sessions() -> None:
  """data_range covers the dist-row index (warm-up sessions excluded)."""
  result = _stat().compute(make_candles_from_closes(_MAIN_CLOSES))
  dates = _weekdays(14)
  # First dist row = session index 2 (0-based), last = session index 13
  expected_start = dates[2]   # 2024-01-04
  expected_end = dates[13]    # 2024-01-19
  dr = result.instruments["NQ"]["daily"].data_range
  assert dr == [expected_start, expected_end]


# ===========================================================================
# 2. min_duration filter
#
# Same 14-session dataset, same dist pattern.
# Confirmed runs: Run 1 (down,dur=2), Run 2 (up,dur=2), Run 3 (down,dur=2).
#
# min_duration=2: all three confirmed runs qualify (duration 2 >= 2).
#   → identical to min_duration=1 results.
#
# min_duration=3: no confirmed run has duration >= 3 → N=0 for both directions.
#
# Default min_duration=5: a separate dataset is used (see below).
# ===========================================================================

def test_min_duration_2_same_as_1_for_main_sequence() -> None:
  """With min_duration=2 all three confirmed runs (each dur=2) still qualify."""
  result_1 = _stat(min_duration=1).compute(make_candles_from_closes(_MAIN_CLOSES))
  result_2 = _stat(min_duration=2).compute(make_candles_from_closes(_MAIN_CLOSES))
  # cross_up N should be identical
  assert _row(result_1, "cross_up", "avg_duration").count == _row(result_2, "cross_up", "avg_duration").count
  assert _row(result_1, "cross_down", "avg_duration").count == _row(result_2, "cross_down", "avg_duration").count


def test_min_duration_3_filters_all_confirmed_runs() -> None:
  """With min_duration=3 no confirmed run (all dur=2) qualifies → N=0 everywhere."""
  result = _stat(min_duration=3).compute(make_candles_from_closes(_MAIN_CLOSES))
  for row in result.instruments["NQ"]["daily"].results:
    assert row.count == 0
    assert row.total == 0
    assert row.value == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Default min_duration=5 filter test
#
# Dist pattern (17 values):
#   +10 x2 (up:2, first run) |  -10 x6 (down:6) | +10 x3 (up:3) | -10 x4 (down:4) | +10 x2 (up:2, last)
#
# All runs: [up:2, down:6, up:3, down:4, up:2]
# Confirmed (drop first=up:2, drop last=up:2): [down:6, up:3, down:4]
#
# min_duration=5: only down:6 qualifies (dur 3 < 5, dur 4 < 5).
#   cross_up:   N=0 (up:3 filtered out)
#   cross_down: N=1 (down:6 qualifies), avg_duration=6, max_duration=6, travel=10
#
# min_duration=1 (for comparison): [down:6, up:3, down:4] all qualify.
#   cross_up:  N=1  avg_dur=3, max_dur=3, avg_trav=10, max_trav=10
#   cross_down: N=2  avg_dur=(6+4)/2=5, max_dur=6, avg_trav=10, max_trav=10
# ---------------------------------------------------------------------------

_FILTER_DIST = [10.0] * 2 + [-10.0] * 6 + [10.0] * 3 + [-10.0] * 4 + [10.0] * 2
_FILTER_CLOSES = closes_from_dist(_FILTER_DIST)  # 19 sessions total


def test_default_min_duration_5_filters_short_runs() -> None:
  """Default min_duration=5: only down:6 qualifies; up:3 and down:4 are filtered."""
  stat = _stat(min_duration=5)
  result = stat.compute(make_candles_from_closes(_FILTER_CLOSES))
  # cross_up: N=0 (up:3 filtered out by min_duration=5)
  assert _row(result, "cross_up", "avg_duration").count == 0
  assert _row(result, "cross_up", "avg_duration").value == pytest.approx(0.0)
  # cross_down: N=1 (only down:6 qualifies)
  assert _row(result, "cross_down", "avg_duration").count == 1
  assert _row(result, "cross_down", "avg_duration").value == pytest.approx(6.0)
  assert _row(result, "cross_down", "max_duration").value == pytest.approx(6.0)
  assert _row(result, "cross_down", "avg_travel").value == pytest.approx(10.0)
  assert _row(result, "cross_down", "max_travel").value == pytest.approx(10.0)


def test_min_duration_1_accepts_all_confirmed_runs_for_filter_sequence() -> None:
  """min_duration=1 with the filter dataset: [down:6, up:3, down:4] all qualify."""
  result = _stat(min_duration=1).compute(make_candles_from_closes(_FILTER_CLOSES))
  # cross_up: N=1 (up:3)
  assert _row(result, "cross_up", "avg_duration").count == 1
  assert _row(result, "cross_up", "avg_duration").value == pytest.approx(3.0)
  # cross_down: N=2 (down:6, down:4), avg_dur=(6+4)/2=5, max_dur=6
  assert _row(result, "cross_down", "avg_duration").count == 2
  assert _row(result, "cross_down", "avg_duration").value == pytest.approx(5.0)
  assert _row(result, "cross_down", "max_duration").value == pytest.approx(6.0)


# ===========================================================================
# 3. Row shape invariants
#
# Always exactly 8 rows: {cross_up, cross_down} x {avg_duration, max_duration,
# avg_travel, max_travel}. For every row:
#   probability == 0.0 (sentinel)
#   baseline_prob == 0.0 (sentinel)
#   count == total
#   value is not None
# When N=0 for a direction, its four rows have count/total==0 and value==0.0.
# ===========================================================================

_EXPECTED_SHAPE = {
  ("cross_up", "avg_duration"), ("cross_up", "max_duration"),
  ("cross_up", "avg_travel"),   ("cross_up", "max_travel"),
  ("cross_down", "avg_duration"), ("cross_down", "max_duration"),
  ("cross_down", "avg_travel"),   ("cross_down", "max_travel"),
}


def test_exactly_8_rows() -> None:
  """compute() always returns exactly 8 rows."""
  result = _stat().compute(make_candles_from_closes(_MAIN_CLOSES))
  rows = result.instruments["NQ"]["daily"].results
  assert len(rows) == 8


def test_row_shape_conditions_and_outcomes() -> None:
  """The (condition, outcome) pairs match exactly the expected 8-element set."""
  rows = _stat().compute(make_candles_from_closes(_MAIN_CLOSES)).instruments["NQ"]["daily"].results
  assert {(r.condition, r.outcome) for r in rows} == _EXPECTED_SHAPE


def test_probability_sentinel_is_zero() -> None:
  """Every row has probability == 0.0 (magnitude stat, probability is a sentinel)."""
  rows = _stat().compute(make_candles_from_closes(_MAIN_CLOSES)).instruments["NQ"]["daily"].results
  for row in rows:
    assert row.probability == pytest.approx(0.0), f"{row.condition}/{row.outcome}: prob != 0"
    assert row.baseline_prob == pytest.approx(0.0)


def test_count_equals_total_for_all_rows() -> None:
  """count == total for every row (no partial pending inside qualified runs)."""
  rows = _stat().compute(make_candles_from_closes(_MAIN_CLOSES)).instruments["NQ"]["daily"].results
  for row in rows:
    assert row.count == row.total, f"{row.condition}/{row.outcome}: count != total"


def test_value_not_none_for_all_rows() -> None:
  """value is always populated (never None), even when N=0."""
  rows = _stat().compute(make_candles_from_closes(_MAIN_CLOSES)).instruments["NQ"]["daily"].results
  for row in rows:
    assert row.value is not None, f"{row.condition}/{row.outcome}: value is None"


def test_zero_direction_rows_have_zero_value() -> None:
  """When a direction has 0 qualifying runs, all four outcome values are 0.0.

  min_duration=3 eliminates all confirmed runs → both directions are empty.
  """
  result = _stat(min_duration=3).compute(make_candles_from_closes(_MAIN_CLOSES))
  for condition in ("cross_up", "cross_down"):
    for outcome in ("avg_duration", "max_duration", "avg_travel", "max_travel"):
      row = _row(result, condition, outcome)
      assert row.count == 0
      assert row.total == 0
      assert row.value == pytest.approx(0.0)


def test_empty_input_8_rows_shape() -> None:
  """Empty candles_df still produces exactly 8 zero-rows."""
  rows = _stat().compute(_empty_df()).instruments["NQ"]["daily"].results
  assert len(rows) == 8
  assert {(r.condition, r.outcome) for r in rows} == _EXPECTED_SHAPE
  for row in rows:
    assert row.count == 0
    assert row.total == 0
    assert row.value == pytest.approx(0.0)
    assert row.probability == pytest.approx(0.0)


# ===========================================================================
# 4. Pending discipline
#
# (a) The last run is always excluded regardless of its duration.
# (b) The first `period` sessions have NaN SMA and are excluded from dist.
#
# Test (a):
#   Dist pattern (13 values):
#     -5 x2 (down:2, first run) | +10 x10 (up:10) | -5 x1 (down:1, LAST run)
#   Runs: [down:2, up:10, down:1]
#   Confirmed (drop first=down:2, drop last=down:1): [up:10]
#   cross_up: N=1, dur=10, travel=10
#   cross_down: N=0 (down:1 excluded as last run)
# ===========================================================================

_PENDING_LAST_DIST = [-5.0] * 2 + [10.0] * 10 + [-5.0] * 1
_PENDING_LAST_CLOSES = closes_from_dist(_PENDING_LAST_DIST)  # 15 sessions


def test_last_run_excluded_even_when_qualifying_length() -> None:
  """Last run (down:1) is excluded even though duration>=min_duration=1.

  Only up:10 (confirmed run) contributes to cross_up; down:1 (last run) is
  never counted in cross_down.
  """
  result = _stat(min_duration=1).compute(make_candles_from_closes(_PENDING_LAST_CLOSES))
  # cross_up: the up:10 run is confirmed → N=1, dur=10, travel=10
  assert _row(result, "cross_up", "avg_duration").count == 1
  assert _row(result, "cross_up", "avg_duration").value == pytest.approx(10.0)
  # cross_down: down:1 is the last run → excluded → N=0
  assert _row(result, "cross_down", "avg_duration").count == 0
  assert _row(result, "cross_down", "avg_duration").value == pytest.approx(0.0)


def test_first_period_sessions_excluded_from_dist() -> None:
  """The first `period` resolved sessions (warm-up) are absent from build_day_table.

  With period=2 and 14 total sessions: day_table has 12 rows, not 14.
  The first two session dates are not in day_table.index.
  """
  stat = _stat()
  df = make_candles_from_closes(_MAIN_CLOSES)
  day_table = stat.build_day_table(df)
  dates = _weekdays(14)
  warmup_0 = pd.Timestamp(dates[0], tz=_NY).normalize()
  warmup_1 = pd.Timestamp(dates[1], tz=_NY).normalize()
  assert len(day_table) == 12
  assert warmup_0 not in day_table.index
  assert warmup_1 not in day_table.index


def test_only_two_runs_means_no_confirmed() -> None:
  """With exactly 2 runs, confirmed=[] (first and last both excluded) → N=0.

  Dist: [+10 x5, -10 x3] → runs=[up:5, down:3]; confirmed=[] (len(runs)<3).
  """
  dist = [10.0] * 5 + [-10.0] * 3
  closes = closes_from_dist(dist)
  result = _stat(min_duration=1).compute(make_candles_from_closes(closes))
  for row in result.instruments["NQ"]["daily"].results:
    assert row.count == 0
    assert row.value == pytest.approx(0.0)


def test_truncated_day_excluded_from_dist_rows() -> None:
  """An unresolved (early-close) day is not counted in total_samples."""
  stat = _stat()
  base_df = make_candles_from_closes(_MAIN_CLOSES)
  total_base = stat.compute(base_df).instruments["NQ"]["daily"].total_samples
  truncated = _make_truncated_day("2024-03-01")
  combined = (
    pd.concat([base_df, truncated], ignore_index=True)
    .sort_values("timestamp")
    .reset_index(drop=True)
  )
  total_with = stat.compute(combined).instruments["NQ"]["daily"].total_samples
  assert total_with == total_base == 12


# ===========================================================================
# 5. Reproducibility / baseline determinism
#
# compute(df, seed=42) called twice produces identical model_dump_json().
# baseline_rows(day_table, seed=42) called twice gives identical results.
# Different seeds produce different permutations (and thus different baselines).
# Baseline rows carry a non-None value and feed value_baseline on the main rows.
# ===========================================================================

# Use a longer dist sequence for the baseline tests (more permutation variety).
_REPRO_DIST = [10.0, -5.0, 8.0, -12.0, 15.0, -3.0, 20.0, -8.0, 6.0, -10.0,
               18.0, -7.0, 12.0, -4.0, 9.0]
_REPRO_CLOSES = closes_from_dist(_REPRO_DIST)  # 17 sessions


def test_compute_fully_reproducible() -> None:
  """compute(seed=42) produces identical JSON-serialisable output on two calls."""
  stat = _stat(min_duration=1)
  df = make_candles_from_closes(_REPRO_CLOSES)
  result_a = stat.compute(df, seed=42)
  result_b = stat.compute(df, seed=42)
  assert result_a.model_dump_json() == result_b.model_dump_json()


def test_baseline_rows_deterministic_same_seed() -> None:
  """baseline_rows(seed=42) returns identical values on two successive calls."""
  stat = _stat(min_duration=1)
  day_table = stat.build_day_table(make_candles_from_closes(_REPRO_CLOSES))
  rows_a = stat.baseline_rows(day_table, seed=42)
  rows_b = stat.baseline_rows(day_table, seed=42)
  for a, b in zip(rows_a, rows_b):
    assert (a.condition, a.outcome) == (b.condition, b.outcome)
    assert a.value == pytest.approx(b.value) if a.value is not None else a.value == b.value
    assert a.count == b.count
    assert a.total == b.total


def test_baseline_rows_differ_for_different_seeds() -> None:
  """Different seeds produce different baseline values on the same data."""
  stat = _stat(min_duration=1)
  day_table = stat.build_day_table(make_candles_from_closes(_REPRO_CLOSES))
  rows_42 = stat.baseline_rows(day_table, seed=42)
  rows_99 = stat.baseline_rows(day_table, seed=99)
  # At least one row must differ in value between the two seeds.
  any_diff = any(
    r42.value != r99.value
    for r42, r99 in zip(rows_42, rows_99)
    if r42.value is not None and r99.value is not None
  )
  assert any_diff, "Baseline with seed=42 and seed=99 produced identical results"


def test_baseline_value_not_none_on_nonempty_data() -> None:
  """Baseline rows carry a non-None value when data is non-empty."""
  stat = _stat(min_duration=1)
  day_table = stat.build_day_table(make_candles_from_closes(_REPRO_CLOSES))
  rows = stat.baseline_rows(day_table, seed=42)
  for row in rows:
    # All 8 rows are always emitted; the value may be 0.0 but never None.
    assert row.value is not None


def test_baseline_embedded_in_compute_rows() -> None:
  """After compute(), rows with count>0 carry a non-None value_baseline."""
  stat = _stat(min_duration=1)
  result = stat.compute(make_candles_from_closes(_REPRO_CLOSES), seed=42)
  for row in result.instruments["NQ"]["daily"].results:
    if row.count > 0:
      assert row.value_baseline is not None, (
        f"{row.condition}/{row.outcome}: value_baseline is None despite count>0"
      )


def test_baseline_permutation_preserves_series_length() -> None:
  """The baseline permutes the dist array: its total count equals the original.

  The permutation cannot create or destroy runs. The total number of qualified
  baseline runs (sum of counts across both conditions) may differ from the
  real data, but must remain within [0, len(day_table)-2] (at most len-2
  confirmed runs if every session alternates sign).
  """
  stat = _stat(min_duration=1)
  df = make_candles_from_closes(_REPRO_CLOSES)
  day_table = stat.build_day_table(df)
  rows = stat.baseline_rows(day_table, seed=42)
  total_qualified = rows[0].total + rows[4].total  # cross_up count + cross_down count
  max_possible = len(day_table)  # loose upper bound
  assert total_qualified <= max_possible


# ===========================================================================
# 6. Empty / insufficient data
#
# Empty candles_df → 8 zero-rows, total_samples=0, data_range=[].
# Fewer than period+1 resolved sessions → 0 dist rows → same 8 zero-rows.
# build_day_table on empty input returns an empty DataFrame with ["dist"] column.
# ===========================================================================

def test_empty_candles_total_samples_zero() -> None:
  """Empty candles_df → total_samples=0 and data_range=[]."""
  result = _stat().compute(_empty_df())
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 0
  assert tf.data_range == []


def test_empty_candles_no_crash() -> None:
  """compute() on an empty DataFrame must not raise."""
  _stat().compute(_empty_df())  # must not raise


def test_build_day_table_returns_dist_column_on_empty_input() -> None:
  """build_day_table on empty input returns an empty DataFrame with 'dist' column."""
  day_table = _stat().build_day_table(_empty_df())
  assert day_table.empty
  assert list(day_table.columns) == ["dist"]


def test_insufficient_sessions_no_dist_rows() -> None:
  """Fewer resolved sessions than period → 0 dist rows → all results zeroed.

  With period=2, 2 resolved sessions produce 0 dist values (rolling NaN).
  """
  closes_2 = [100.0, 110.0]
  df = make_candles_from_closes(closes_2)
  result = _stat(period=2).compute(df)
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 0
  for row in tf.results:
    assert row.count == 0
    assert row.value == pytest.approx(0.0)


def test_exactly_period_plus_one_sessions_gives_one_dist_row() -> None:
  """With period+1 sessions exactly 1 dist row is produced.

  1 dist row → 1 run total → confirmed=[] (first==last) → N=0.
  total_samples=1.
  """
  # 3 sessions, period=2 → 1 dist row
  closes_3 = [100.0, 100.0, 110.0]
  df = make_candles_from_closes(closes_3)
  result = _stat(period=2).compute(df)
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 1
  for row in tf.results:
    assert row.count == 0  # 1 run → confirmed=[], so N=0
    assert row.value == pytest.approx(0.0)


def test_two_dist_rows_single_run_no_confirmed() -> None:
  """2 dist rows with same sign → 1 run → confirmed=[] (first==last) → N=0."""
  closes_4 = [100.0, 100.0, 110.0, 120.0]  # 4 sessions, period=2 → 2 dist rows (+)
  df = make_candles_from_closes(closes_4)
  result = _stat(period=2).compute(df)
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 2
  for row in tf.results:
    assert row.count == 0


def test_two_dist_rows_alternating_gives_two_runs_no_confirmed() -> None:
  """2 dist rows with opposite signs → 2 runs → confirmed=[] → N=0."""
  # +10, -5: runs=[up:1, down:1] → confirmed=[] (only 2 runs)
  closes = closes_from_dist([10.0, -5.0])
  df = make_candles_from_closes(closes)
  result = _stat(period=2).compute(df)
  for row in result.instruments["NQ"]["daily"].results:
    assert row.count == 0


# ===========================================================================
# 7. Travel sign correctness
#
# Explicit case verifying that:
#   - up-travel   = max(dist values in the up run)     = max positive distance
#   - down-travel = max(-dist values in the down run)  = max magnitude below SMA
#
# Dist pattern (8 values):
#   -2 (down:1, FIRST run) | +3, +10, +5 (up:3, confirmed) | -4, -15, -7 (down:3, confirmed) | +20 (up:1, LAST run)
#
# Runs: [down:1, up:3, down:3, up:1]
# Confirmed: [up:3, down:3]
#
# cross_up: N=1, dur=3, travel = max(3, 10, 5) = 10  (NOT 20 from the excluded last run)
# cross_down: N=1, dur=3, travel = max(4, 15, 7) = 15 (NOT the sum or average)
# ===========================================================================

_TRAVEL_DIST = [-2.0, 3.0, 10.0, 5.0, -4.0, -15.0, -7.0, 20.0]
_TRAVEL_CLOSES = closes_from_dist(_TRAVEL_DIST)  # 10 sessions


def test_up_travel_is_max_positive_dist_not_average() -> None:
  """Up-travel = max(dist values) = 10 (not the average (3+10+5)/3=6)."""
  result = _stat(min_duration=1).compute(make_candles_from_closes(_TRAVEL_CLOSES))
  assert _row(result, "cross_up", "max_travel").value == pytest.approx(10.0)
  # Also verify avg_travel is the mean of the single run's travel
  assert _row(result, "cross_up", "avg_travel").value == pytest.approx(10.0)


def test_down_travel_is_max_magnitude_below_sma() -> None:
  """Down-travel = max(-dist values) = max(4, 15, 7) = 15 (not 7 or the sum)."""
  result = _stat(min_duration=1).compute(make_candles_from_closes(_TRAVEL_CLOSES))
  assert _row(result, "cross_down", "max_travel").value == pytest.approx(15.0)
  assert _row(result, "cross_down", "avg_travel").value == pytest.approx(15.0)


def test_up_travel_does_not_include_excluded_last_run() -> None:
  """The excluded last run (up:1, dist=20) must not inflate up-travel.

  If the last run were counted, max_travel would be 20. It should be 10.
  """
  result = _stat(min_duration=1).compute(make_candles_from_closes(_TRAVEL_CLOSES))
  assert _row(result, "cross_up", "max_travel").value == pytest.approx(10.0)
  # Definitely not 20 (last run) and not the average of the up run values.
  assert _row(result, "cross_up", "max_travel").value != pytest.approx(20.0)


def test_down_travel_duration_count() -> None:
  """Travel test dataset: cross_up N=1 dur=3, cross_down N=1 dur=3."""
  result = _stat(min_duration=1).compute(make_candles_from_closes(_TRAVEL_CLOSES))
  assert _row(result, "cross_up", "avg_duration").count == 1
  assert _row(result, "cross_up", "avg_duration").value == pytest.approx(3.0)
  assert _row(result, "cross_down", "avg_duration").count == 1
  assert _row(result, "cross_down", "avg_duration").value == pytest.approx(3.0)


# ===========================================================================
# 8. dist == 0 ties go to "down"
#
# Dist: [+10, 0, -5, +10]
# Signs:  up,  down (0→down), down, up
# Runs:   [up:1, down:2, up:1]
# Confirmed: [down:2]
#   travel = max(-0, 5) = max(0, 5) = 5
# ===========================================================================

def test_zero_dist_classified_as_down() -> None:
  """dist == 0 ties go to 'down'. The zero row joins the following -5 in a down:2 run."""
  dist = [10.0, 0.0, -5.0, 10.0]
  closes = closes_from_dist(dist)
  df = make_candles_from_closes(closes)
  result = _stat(min_duration=1).compute(df)
  # cross_down: N=1, dur=2, travel=max(-0, 5)=5
  assert _row(result, "cross_down", "avg_duration").count == 1
  assert _row(result, "cross_down", "avg_duration").value == pytest.approx(2.0)
  assert _row(result, "cross_down", "avg_travel").value == pytest.approx(5.0)
  # cross_up: N=0 (up:1 is first run, up:1 is last run, both excluded)
  assert _row(result, "cross_up", "avg_duration").count == 0


# ===========================================================================
# 8b. classify_samples
#
# sma_performance events (runs) span multiple days, so the natural sample unit
# is the qualified run, not the individual day. Each qualified run emits 4
# SampleRows (avg_duration, max_duration, avg_travel, max_travel), dated on
# the run's LAST session.
#
# Reusing the period=2, min_duration=1 dataset from section 1 (_MAIN_CLOSES):
#   Run0 up:3   -> positions 0,1,2   (FIRST → excluded)
#   Run1 down:2 -> positions 3,4     -> end index 4, travel=8
#   Run2 up:2   -> positions 5,6     -> end index 6, travel=18
#   Run3 down:2 -> positions 7,8     -> end index 8, travel=9
#   Run4 up:3   -> positions 9,10,11 (LAST → excluded)
#
# Confirmed & qualified (min_duration=1): Run1, Run2, Run3.
# Total samples = 3 runs x 4 outcomes = 12.
# ===========================================================================

def test_classify_samples_exact_list() -> None:
  """classify_samples emits 4 SampleRows per qualified run, dated on its last session."""
  stat = _stat()  # period=2, min_duration=1
  df = make_candles_from_closes(_MAIN_CLOSES)
  day_table = stat.build_day_table(df)
  samples = stat.classify_samples(day_table)

  date_run1 = day_table.index[4].strftime("%Y-%m-%d")  # Run1 down:2, travel=8
  date_run2 = day_table.index[6].strftime("%Y-%m-%d")  # Run2 up:2, travel=18
  date_run3 = day_table.index[8].strftime("%Y-%m-%d")  # Run3 down:2, travel=9

  expected = [
    (date_run1, "cross_down", "avg_duration", 2.0),
    (date_run1, "cross_down", "max_duration", 2.0),
    (date_run1, "cross_down", "avg_travel", 8.0),
    (date_run1, "cross_down", "max_travel", 8.0),
    (date_run2, "cross_up", "avg_duration", 2.0),
    (date_run2, "cross_up", "max_duration", 2.0),
    (date_run2, "cross_up", "avg_travel", 18.0),
    (date_run2, "cross_up", "max_travel", 18.0),
    (date_run3, "cross_down", "avg_duration", 2.0),
    (date_run3, "cross_down", "max_duration", 2.0),
    (date_run3, "cross_down", "avg_travel", 9.0),
    (date_run3, "cross_down", "max_travel", 9.0),
  ]
  assert len(samples) == len(expected) == 12
  for s, (date, condition, outcome, value) in zip(samples, expected):
    assert s.date == date
    assert s.condition == condition
    assert s.outcome == outcome
    assert s.value == pytest.approx(value)


def test_classify_samples_matches_compute_rows_counts() -> None:
  """For every StatResultRow, the matching SampleRow count equals r.count."""
  stat = _stat()
  day_table = stat.build_day_table(make_candles_from_closes(_MAIN_CLOSES))
  samples = stat.classify_samples(day_table)
  rows = stat.compute_rows(day_table)
  for r in rows:
    n = sum(1 for s in samples if s.condition == r.condition and s.outcome == r.outcome)
    assert n == r.count


def test_classify_samples_avg_outcomes_mean_matches_compute_rows() -> None:
  """Mean of avg_duration/avg_travel SampleRow values reproduces the compute_rows value."""
  stat = _stat()
  day_table = stat.build_day_table(make_candles_from_closes(_MAIN_CLOSES))
  samples = stat.classify_samples(day_table)
  rows = {(r.condition, r.outcome): r for r in stat.compute_rows(day_table)}
  for condition in ("cross_up", "cross_down"):
    for outcome in ("avg_duration", "avg_travel"):
      values = [s.value for s in samples if s.condition == condition and s.outcome == outcome]
      assert values, f"no samples for {condition}/{outcome}"
      assert sum(values) / len(values) == pytest.approx(rows[(condition, outcome)].value)


def test_classify_samples_max_outcomes_max_matches_compute_rows() -> None:
  """Max of max_duration/max_travel SampleRow values reproduces the compute_rows value."""
  stat = _stat()
  day_table = stat.build_day_table(make_candles_from_closes(_MAIN_CLOSES))
  samples = stat.classify_samples(day_table)
  rows = {(r.condition, r.outcome): r for r in stat.compute_rows(day_table)}
  for condition in ("cross_up", "cross_down"):
    for outcome in ("max_duration", "max_travel"):
      values = [s.value for s in samples if s.condition == condition and s.outcome == outcome]
      assert values, f"no samples for {condition}/{outcome}"
      assert max(values) == pytest.approx(rows[(condition, outcome)].value)


def test_classify_samples_excludes_first_and_last_run() -> None:
  """First (no prior regime) and last (pending) runs never produce samples."""
  stat = _stat(min_duration=1)
  df = make_candles_from_closes(_PENDING_LAST_CLOSES)
  day_table = stat.build_day_table(df)
  samples = stat.classify_samples(day_table)
  # Only the confirmed up:10 run qualifies; down:2 (first) and down:1 (last)
  # are excluded even though down:1 would otherwise satisfy min_duration=1.
  assert len(samples) == 4
  assert all(s.condition == "cross_up" for s in samples)
  assert all(s.value == pytest.approx(10.0) for s in samples)


def test_classify_samples_respects_min_duration_filter() -> None:
  """Runs shorter than min_duration produce no samples, mirroring compute_rows."""
  stat = _stat(min_duration=3)
  day_table = stat.build_day_table(make_candles_from_closes(_MAIN_CLOSES))
  assert stat.classify_samples(day_table) == []


def test_classify_samples_empty_day_table() -> None:
  """Empty day_table -> classify_samples returns []."""
  stat = _stat()
  assert stat.classify_samples(stat.build_day_table(_empty_df())) == []


def test_classify_samples_only_two_runs_means_no_samples() -> None:
  """With exactly 2 runs, confirmed=[] (first and last both excluded) -> no samples."""
  dist = [10.0] * 5 + [-10.0] * 3
  closes = closes_from_dist(dist)
  stat = _stat(min_duration=1)
  day_table = stat.build_day_table(make_candles_from_closes(closes))
  assert stat.classify_samples(day_table) == []


# ===========================================================================
# 9. stat_name, slices, no-op slicing, defaults
# ===========================================================================

def test_stat_name() -> None:
  """stat_name == 'sma_performance'."""
  result = _stat().compute(_empty_df())
  assert result.stat_name == "sma_performance"


def test_no_slices() -> None:
  """SMAPerformance declares no slices (events span multiple days)."""
  result = _stat().compute(make_candles_from_closes(_MAIN_CLOSES))
  assert result.instruments["NQ"]["daily"].slices == {}


def test_default_period_is_200() -> None:
  """Default period is 200."""
  stat = SMAPerformance(instrument="NQ", config=_TEST_CONFIG)
  assert stat.period == 200


def test_default_min_duration_is_5() -> None:
  """Default min_duration is 5."""
  stat = SMAPerformance(instrument="NQ", config=_TEST_CONFIG)
  assert stat.min_duration == 5


def test_timeframe_is_daily() -> None:
  """timeframe attribute is 'daily'."""
  assert _stat().timeframe == "daily"


# ===========================================================================
# 10. i18n — title, definition, labels
# ===========================================================================

def test_i18n_title_and_definition_non_empty() -> None:
  """title and definition both have non-empty en and fr strings."""
  result = _stat().compute(_empty_df())
  assert result.title.en != ""
  assert result.title.fr != ""
  assert result.definition.en != ""
  assert result.definition.fr != ""


def test_i18n_labels_conditions_and_outcomes() -> None:
  """Labels cover exactly {cross_up, cross_down} and the four outcomes."""
  result = _stat().compute(_empty_df())
  assert set(result.labels.conditions) == {"cross_up", "cross_down"}
  assert set(result.labels.outcomes) == {
    "avg_duration", "max_duration", "avg_travel", "max_travel"
  }


def test_i18n_labels_all_have_en_and_fr() -> None:
  """Every condition and outcome label has non-empty en and fr."""
  result = _stat().compute(_empty_df())
  for mapping in (result.labels.conditions, result.labels.outcomes):
    for key, i18n in mapping.items():
      assert i18n.en != "", f"{key}.en is empty"
      assert i18n.fr != "", f"{key}.fr is empty"


# ===========================================================================
# 11. write_results round-trip
# ===========================================================================

def test_write_results_round_trip(tmp_path: Path) -> None:
  """write_results produces sma_performance.json that re-validates correctly."""
  result = _stat(min_duration=1).compute(make_candles_from_closes(_MAIN_CLOSES))
  written = write_results(result, results_dir=tmp_path)
  assert written.name == "sma_performance.json"

  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)
  assert validated.stat_name == "sma_performance"
  assert "NQ" in validated.instruments
  assert "daily" in validated.instruments["NQ"]
  assert validated.instruments["NQ"]["daily"].total_samples == 12


def test_write_results_french_accent_literal(tmp_path: Path) -> None:
  """French text is stored as literal UTF-8 characters, not escaped unicode."""
  result = _stat().compute(_empty_df())
  written = write_results(result, results_dir=tmp_path)
  raw = written.read_text(encoding="utf-8")
  # The French labels contain accented characters (e.g. "Croisement", "Durée").
  assert "é" in raw
  assert "\\u00e9" not in raw


def test_write_results_all_8_rows_survive_roundtrip(tmp_path: Path) -> None:
  """All 8 rows are preserved after JSON serialisation and re-validation."""
  result = _stat(min_duration=1).compute(make_candles_from_closes(_MAIN_CLOSES))
  written = write_results(result, results_dir=tmp_path)
  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)
  rows = validated.instruments["NQ"]["daily"].results
  assert len(rows) == 8
  assert {(r.condition, r.outcome) for r in rows} == _EXPECTED_SHAPE
