"""Tests for stats.avg_consecutive_bars.standard.

All data is synthetic — no real market files required.
Expected values are hand-calculated before each assertion.

Resolution criterion (build_resolved_days):
  - Bar at exactly mod=570 (09:30 RTH open).
  - Last RTH bar mod >= 975 - 15 = 960.

Bucket arithmetic (5min, RTH start=09:30=570):
  bucket_idx = (mod - 570) // 5
  bucket  0 → mod 570 → "0930"
  bucket  1 → mod 575 → "0935"
  bucket  6 → mod 600 → "1000"
  bucket 78 → mod 960 → "1600"  (close-tolerance sentinel)

Test data builder (_make_streak_day):
  - Places 1 bar per explicit color at consecutive buckets from mod=570.
  - Appends a GREEN sentinel bar at mod=960 if the explicit colors
    don't already reach mod >= 960.
  - green bar : open=100, close=102  (close > open  → True)
  - red bar   : open=100, close=98   (close < open  → False)
  - sentinel  : open=100, close=100  (flat, close >= open → True = green)

Three-day base dataset:
  Day A 2024-01-08 (Mon): colors=[G,G,R,G,G,G,R]
    + sentinel(G)  → bar_colors=[T,T,F,T,T,T,F,T]
    Runs: TT(2), F(1), TTT(3), F(1), T(1)
    max_green=3, max_red=1

  Day B 2024-01-09 (Tue): colors=[G,R,R,G,R]
    + sentinel(G)  → bar_colors=[T,F,F,T,F,T]
    Runs: T(1), FF(2), T(1), F(1), T(1)
    max_green=1, max_red=2

  Day C 2024-01-10 (Wed): colors=[R,R,R,R,G]
    + sentinel(G)  → bar_colors=[F,F,F,F,T,T]
    Runs: FFFF(4), TT(2)
    max_green=2, max_red=4

  mean green streak = (3+1+2) / 3 = 2.0
  mean red   streak = (1+2+4) / 3 = 7/3 ≈ 2.333...
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stats.avg_consecutive_bars.standard import AvgConsecutiveBars, _longest_run
from stats.base import SampleRow, StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session

# ---------------------------------------------------------------------------
# Minimal InstrumentConfig — does NOT depend on NQ.yaml
# ---------------------------------------------------------------------------
_NY = "America/New_York"
_RTH_SESSION = Session(start="09:30", end="16:15")
_TEST_CONFIG = InstrumentConfig(
  instrument="NQ",
  working_timezone=_NY,
  sessions={"rth": _RTH_SESSION},
  timeframes=["5min"],
  parquet_path=Path("data/NQ_1min.parquet"),
)
_TEST_CONFIG_15 = InstrumentConfig(
  instrument="NQ",
  working_timezone=_NY,
  sessions={"rth": _RTH_SESSION},
  timeframes=["15min"],
  parquet_path=Path("data/NQ_1min.parquet"),
)
_TEST_CONFIG_1H = InstrumentConfig(
  instrument="NQ",
  working_timezone=_NY,
  sessions={"rth": _RTH_SESSION},
  timeframes=["1h"],
  parquet_path=Path("data/NQ_1min.parquet"),
)

_RTH_START = 570    # 09:30
_RTH_END = 975      # 16:15 (exclusive)
_RESOLVED_MIN = 960  # rth_end - close_tolerance = 975 - 15


# ---------------------------------------------------------------------------
# Synthetic day builders
# ---------------------------------------------------------------------------

def _make_streak_day(
  date: str,
  colors: list[bool],
  bucket_min: int = 5,
) -> pd.DataFrame:
  """Build one resolved RTH day with controlled per-bucket colors.

  Each entry in ``colors`` corresponds to one consecutive bucket starting at
  bucket 0 (mod=570).  The sentinel bar at mod=960 is appended (as a flat
  green bar) when the explicit colors do not already reach mod >= 960 —
  this satisfies the close-tolerance resolution criterion.

  True  → green bucket: open=100, close=102
  False → red   bucket: open=100, close=98
  sentinel (flat)     : open=100, close=100 → close >= open → True (green)
  """
  base = pd.Timestamp(date, tz=_NY)
  records = []

  last_explicit_mod = _RTH_START + (len(colors) - 1) * bucket_min if colors else _RTH_START - 1

  for i, green in enumerate(colors):
    mod = _RTH_START + i * bucket_min
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    o, c = (100.0, 102.0) if green else (100.0, 98.0)
    records.append({
      "timestamp": ts,
      "open": o,
      "high": max(o, c) + 1.0,
      "low": min(o, c) - 1.0,
      "close": c,
      "volume": 100,
    })

  if last_explicit_mod < _RESOLVED_MIN:
    h, m = divmod(_RESOLVED_MIN, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    # Flat sentinel → green (close >= open)
    records.append({
      "timestamp": ts,
      "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 100,
    })

  return pd.DataFrame(records)


def _make_truncated_day(date: str) -> pd.DataFrame:
  """A day whose last RTH bar is at mod=590 (09:50) < 960 — excluded."""
  base = pd.Timestamp(date, tz=_NY)
  records = []
  for mod in range(_RTH_START, 591):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    records.append({
      "timestamp": ts, "open": 100.0, "high": 101.0, "low": 99.0,
      "close": 100.0, "volume": 100,
    })
  return pd.DataFrame(records)


def _concat(*frames: pd.DataFrame) -> pd.DataFrame:
  df = pd.concat(frames, ignore_index=True)
  return df.sort_values("timestamp").reset_index(drop=True)


def _empty_df() -> pd.DataFrame:
  df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
  df["timestamp"] = pd.array([], dtype="datetime64[ns, America/New_York]")
  return df


# Three-day base data (see module docstring for hand-calculated streak values)
_COLORS_A = [True, True, False, True, True, True, False]   # max_green=3, max_red=1
_COLORS_B = [True, False, False, True, False]               # max_green=1, max_red=2
_COLORS_C = [False, False, False, False, True]              # max_green=2, max_red=4

_DAY_A = "2024-01-08"
_DAY_B = "2024-01-09"
_DAY_C = "2024-01-10"
_DAY_D = "2024-01-15"  # truncated — excluded


def _make_three_days() -> pd.DataFrame:
  return _concat(
    _make_streak_day(_DAY_A, _COLORS_A),
    _make_streak_day(_DAY_B, _COLORS_B),
    _make_streak_day(_DAY_C, _COLORS_C),
  )


def _make_three_days_with_truncated() -> pd.DataFrame:
  return _concat(
    _make_streak_day(_DAY_A, _COLORS_A),
    _make_streak_day(_DAY_B, _COLORS_B),
    _make_streak_day(_DAY_C, _COLORS_C),
    _make_truncated_day(_DAY_D),
  )


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def _stat(timeframe: str = "5min", close_tolerance_min: int = 15) -> AvgConsecutiveBars:
  cfg = _TEST_CONFIG if timeframe == "5min" else (
    _TEST_CONFIG_15 if timeframe == "15min" else _TEST_CONFIG_1H
  )
  return AvgConsecutiveBars(
    instrument="NQ",
    config=cfg,
    timeframe=timeframe,
    close_tolerance_min=close_tolerance_min,
  )


def _tf(result: StatRunResult, timeframe: str = "5min"):  # type: ignore[return]
  return result.instruments["NQ"][timeframe]


def _row(rows: list[StatResultRow], condition: str) -> StatResultRow:
  """Find the single row with the given condition (outcome is always max_streak)."""
  for r in rows:
    if r.condition == condition:
      return r
  raise KeyError(f"condition {condition!r} not found")


# ===========================================================================
# 1. _longest_run unit tests
# ===========================================================================

def test_longest_run_empty_array_returns_zero() -> None:
  """Empty array → 0 regardless of color."""
  arr: np.ndarray = np.array([], dtype=bool)
  assert _longest_run(arr, want_green=True) == 0
  assert _longest_run(arr, want_green=False) == 0


def test_longest_run_all_green_want_green() -> None:
  """All-green array: longest green run = len(array)."""
  arr = np.array([True, True, True, True], dtype=bool)
  # Single run of 4 greens → 4
  assert _longest_run(arr, want_green=True) == 4


def test_longest_run_all_green_want_red() -> None:
  """All-green array: no red bars → longest red run = 0."""
  arr = np.array([True, True, True], dtype=bool)
  assert _longest_run(arr, want_green=False) == 0


def test_longest_run_all_red_want_green() -> None:
  """All-red array: no green bars → longest green run = 0."""
  arr = np.array([False, False, False], dtype=bool)
  assert _longest_run(arr, want_green=True) == 0


def test_longest_run_all_red_want_red() -> None:
  """All-red array: longest red run = len(array)."""
  arr = np.array([False, False, False, False], dtype=bool)
  # Single run of 4 reds → 4
  assert _longest_run(arr, want_green=False) == 4


def test_longest_run_alternating_want_green() -> None:
  """Alternating G/R: every run has length 1 → longest green = 1."""
  arr = np.array([True, False, True, False, True], dtype=bool)
  # Runs: T(1), F(1), T(1), F(1), T(1) → max green = 1
  assert _longest_run(arr, want_green=True) == 1


def test_longest_run_alternating_want_red() -> None:
  """Alternating G/R: every run has length 1 → longest red = 1."""
  arr = np.array([True, False, True, False, True], dtype=bool)
  assert _longest_run(arr, want_green=False) == 1


def test_longest_run_known_mixed_sequence_green() -> None:
  """[G,G,R,G,G,G,R] → longest green run = 3 (indices 3-5)."""
  arr = np.array([True, True, False, True, True, True, False], dtype=bool)
  # Runs: TT(2), F(1), TTT(3), F(1) → max green = 3
  assert _longest_run(arr, want_green=True) == 3


def test_longest_run_known_mixed_sequence_red() -> None:
  """[G,G,R,G,G,G,R] → longest red run = 1 (each F is isolated)."""
  arr = np.array([True, True, False, True, True, True, False], dtype=bool)
  # F at index 2, F at index 6 → both isolated → max red = 1
  assert _longest_run(arr, want_green=False) == 1


def test_longest_run_single_green_bar() -> None:
  """Single green bar: longest green = 1, longest red = 0."""
  arr = np.array([True], dtype=bool)
  assert _longest_run(arr, want_green=True) == 1
  assert _longest_run(arr, want_green=False) == 0


def test_longest_run_single_red_bar() -> None:
  """Single red bar: longest red = 1, longest green = 0."""
  arr = np.array([False], dtype=bool)
  assert _longest_run(arr, want_green=False) == 1
  assert _longest_run(arr, want_green=True) == 0


def test_longest_run_red_streak_in_middle() -> None:
  """[G,R,R,G,R] → longest red run = 2 (indices 1-2)."""
  arr = np.array([True, False, False, True, False], dtype=bool)
  # Runs: T(1), FF(2), T(1), F(1) → max red = 2
  assert _longest_run(arr, want_green=False) == 2
  assert _longest_run(arr, want_green=True) == 1


def test_longest_run_long_red_start() -> None:
  """[R,R,R,R,G,G] → longest red = 4, longest green = 2."""
  arr = np.array([False, False, False, False, True, True], dtype=bool)
  # Runs: FFFF(4), TT(2) → max red=4, max green=2
  assert _longest_run(arr, want_green=False) == 4
  assert _longest_run(arr, want_green=True) == 2


# ===========================================================================
# 2. build_day_table correctness — single day
# ===========================================================================

def test_build_day_table_single_day_columns() -> None:
  """Day table has exactly the three required columns."""
  dt = _stat().build_day_table(_make_streak_day(_DAY_A, _COLORS_A))
  assert set(dt.columns) == {"max_green_streak", "max_red_streak", "bar_colors"}


def test_build_day_table_single_day_index() -> None:
  """Day table index is the normalized session date."""
  dt = _stat().build_day_table(_make_streak_day(_DAY_A, _COLORS_A))
  expected = pd.Timestamp(_DAY_A, tz=_NY).normalize()
  assert expected in dt.index
  assert len(dt) == 1


def test_build_day_table_single_day_max_green() -> None:
  """Day A colors=[G,G,R,G,G,G,R] + sentinel(G) → max_green=3.

  bar_colors=[T,T,F,T,T,T,F,T]: runs TT(2), F(1), TTT(3), F(1), T(1)
  Longest green run = 3 (explicit buckets 3-5).
  """
  dt = _stat().build_day_table(_make_streak_day(_DAY_A, _COLORS_A))
  date_a = pd.Timestamp(_DAY_A, tz=_NY).normalize()
  # Runs: TT(2), F(1), TTT(3), F(1), T(1) → max_green = 3
  assert dt.loc[date_a, "max_green_streak"] == pytest.approx(3.0)


def test_build_day_table_single_day_max_red() -> None:
  """Day A colors=[G,G,R,G,G,G,R] + sentinel(G) → max_red=1.

  F at bucket 2 (isolated) and F at bucket 6 (isolated): max red run = 1.
  """
  dt = _stat().build_day_table(_make_streak_day(_DAY_A, _COLORS_A))
  date_a = pd.Timestamp(_DAY_A, tz=_NY).normalize()
  # Both red bars are isolated → max_red = 1
  assert dt.loc[date_a, "max_red_streak"] == pytest.approx(1.0)


def test_build_day_table_single_day_bar_colors_type() -> None:
  """bar_colors is a numpy bool array."""
  dt = _stat().build_day_table(_make_streak_day(_DAY_A, _COLORS_A))
  date_a = pd.Timestamp(_DAY_A, tz=_NY).normalize()
  bc = dt.loc[date_a, "bar_colors"]
  assert isinstance(bc, np.ndarray)
  assert bc.dtype == bool


def test_build_day_table_single_day_bar_colors_values() -> None:
  """Day A: bar_colors=[T,T,F,T,T,T,F,T] (7 explicit + 1 green sentinel)."""
  dt = _stat().build_day_table(_make_streak_day(_DAY_A, _COLORS_A))
  date_a = pd.Timestamp(_DAY_A, tz=_NY).normalize()
  bc = dt.loc[date_a, "bar_colors"]
  # _COLORS_A=[G,G,R,G,G,G,R] explicit + sentinel(G) = 8 bars
  expected = np.array([True, True, False, True, True, True, False, True], dtype=bool)
  assert np.array_equal(bc, expected), f"bar_colors={bc}, expected={expected}"


def test_build_day_table_all_green_max_streaks() -> None:
  """All-green day: max_green=N_bars, max_red=0.

  colors=[G,G,G,G,G] (5 explicit) + sentinel(G) = 6 bars all green.
  max_green=6, max_red=0.
  """
  colors = [True] * 5
  dt = _stat().build_day_table(_make_streak_day(_DAY_A, colors))
  date_a = pd.Timestamp(_DAY_A, tz=_NY).normalize()
  # 5 explicit greens + 1 green sentinel = 6 consecutive greens
  assert dt.loc[date_a, "max_green_streak"] == pytest.approx(6.0)
  assert dt.loc[date_a, "max_red_streak"] == pytest.approx(0.0)


def test_build_day_table_all_red_max_streaks() -> None:
  """All-red explicit bars + green sentinel: max_red=5, max_green=1.

  colors=[R,R,R,R,R] (5 explicit) + sentinel(G)
  bar_colors=[F,F,F,F,F,T]: FFFFF(5), T(1)
  max_red=5, max_green=1.
  """
  colors = [False] * 5
  dt = _stat().build_day_table(_make_streak_day(_DAY_A, colors))
  date_a = pd.Timestamp(_DAY_A, tz=_NY).normalize()
  # FFFFF(5) + T(1): max_red=5, max_green=1
  assert dt.loc[date_a, "max_red_streak"] == pytest.approx(5.0)
  assert dt.loc[date_a, "max_green_streak"] == pytest.approx(1.0)


# ===========================================================================
# 3. Multi-day: compute_rows values match hand-calculated means
# ===========================================================================

def test_multi_day_table_has_three_rows() -> None:
  """Three fully resolved days → day table has exactly 3 rows."""
  dt = _stat().build_day_table(_make_three_days())
  assert len(dt) == 3


def test_multi_day_table_day_a_max_green() -> None:
  """Day A (2024-01-08): max_green_streak=3."""
  dt = _stat().build_day_table(_make_three_days())
  date_a = pd.Timestamp(_DAY_A, tz=_NY).normalize()
  # [T,T,F,T,T,T,F,T]: TTT run → max_green=3
  assert dt.loc[date_a, "max_green_streak"] == pytest.approx(3.0)


def test_multi_day_table_day_a_max_red() -> None:
  """Day A (2024-01-08): max_red_streak=1."""
  dt = _stat().build_day_table(_make_three_days())
  date_a = pd.Timestamp(_DAY_A, tz=_NY).normalize()
  # Both F bars isolated → max_red=1
  assert dt.loc[date_a, "max_red_streak"] == pytest.approx(1.0)


def test_multi_day_table_day_b_max_green() -> None:
  """Day B (2024-01-09): max_green_streak=1."""
  dt = _stat().build_day_table(_make_three_days())
  date_b = pd.Timestamp(_DAY_B, tz=_NY).normalize()
  # [T,F,F,T,F,T]: T runs all length 1 → max_green=1
  assert dt.loc[date_b, "max_green_streak"] == pytest.approx(1.0)


def test_multi_day_table_day_b_max_red() -> None:
  """Day B (2024-01-09): max_red_streak=2."""
  dt = _stat().build_day_table(_make_three_days())
  date_b = pd.Timestamp(_DAY_B, tz=_NY).normalize()
  # [T,F,F,T,F,T]: FF run → max_red=2
  assert dt.loc[date_b, "max_red_streak"] == pytest.approx(2.0)


def test_multi_day_table_day_c_max_green() -> None:
  """Day C (2024-01-10): max_green_streak=2 (last explicit G + sentinel G)."""
  dt = _stat().build_day_table(_make_three_days())
  date_c = pd.Timestamp(_DAY_C, tz=_NY).normalize()
  # [F,F,F,F,T,T]: TT run (last explicit G + sentinel G) → max_green=2
  assert dt.loc[date_c, "max_green_streak"] == pytest.approx(2.0)


def test_multi_day_table_day_c_max_red() -> None:
  """Day C (2024-01-10): max_red_streak=4."""
  dt = _stat().build_day_table(_make_three_days())
  date_c = pd.Timestamp(_DAY_C, tz=_NY).normalize()
  # [F,F,F,F,T,T]: FFFF run → max_red=4
  assert dt.loc[date_c, "max_red_streak"] == pytest.approx(4.0)


def test_compute_rows_mean_green_streak() -> None:
  """compute_rows green value = mean daily-max green = (3+1+2)/3 = 2.0."""
  stat = _stat()
  dt = stat.build_day_table(_make_three_days())
  rows = stat.compute_rows(dt)
  r = _row(rows, "green")
  # mean([3, 1, 2]) = 6/3 = 2.0
  assert r.value == pytest.approx(2.0)


def test_compute_rows_mean_red_streak() -> None:
  """compute_rows red value = mean daily-max red = (1+2+4)/3 = 7/3."""
  stat = _stat()
  dt = stat.build_day_table(_make_three_days())
  rows = stat.compute_rows(dt)
  r = _row(rows, "red")
  # mean([1, 2, 4]) = 7/3 ≈ 2.333...
  assert r.value == pytest.approx(7 / 3)


def test_compute_rows_count_equals_total_equals_n_days() -> None:
  """count == total == 3 (one contributing day per row, 3 days total)."""
  stat = _stat()
  dt = stat.build_day_table(_make_three_days())
  rows = stat.compute_rows(dt)
  for r in rows:
    assert r.count == 3
    assert r.total == 3


def test_compute_rows_probability_and_baseline_prob_zero() -> None:
  """Magnitude stat: probability and baseline_prob are always 0.0."""
  stat = _stat()
  dt = stat.build_day_table(_make_three_days())
  rows = stat.compute_rows(dt)
  for r in rows:
    assert r.probability == pytest.approx(0.0)
    assert r.baseline_prob == pytest.approx(0.0)


def test_compute_rows_outcome_key_is_max_streak() -> None:
  """Both rows carry outcome='max_streak'."""
  stat = _stat()
  dt = stat.build_day_table(_make_three_days())
  rows = stat.compute_rows(dt)
  for r in rows:
    assert r.outcome == "max_streak"


def test_compute_rows_exactly_two_rows() -> None:
  """compute_rows always returns exactly 2 rows: green and red."""
  stat = _stat()
  dt = stat.build_day_table(_make_three_days())
  rows = stat.compute_rows(dt)
  assert len(rows) == 2
  assert {r.condition for r in rows} == {"green", "red"}


def test_compute_rows_value_baseline_none_without_baseline() -> None:
  """Without passing baseline_rows, value_baseline is None on each row."""
  stat = _stat()
  dt = stat.build_day_table(_make_three_days())
  rows = stat.compute_rows(dt, baseline_rows=None)
  for r in rows:
    assert r.value_baseline is None


def test_compute_rows_value_baseline_merged_from_baseline() -> None:
  """When baseline_rows is provided, value_baseline is filled from it."""
  stat = _stat()
  dt = stat.build_day_table(_make_three_days())
  bl = stat.baseline_rows(dt, seed=42)
  rows = stat.compute_rows(dt, baseline_rows=bl)
  for r in rows:
    assert r.value_baseline is not None


# ===========================================================================
# 4. Pending / early-close exclusion
# ===========================================================================

def test_truncated_day_absent_from_day_table() -> None:
  """Truncated 2024-01-15 (last bar mod=590 < 960) not in day_table index."""
  dt = _stat().build_day_table(_make_three_days_with_truncated())
  excluded = pd.Timestamp(_DAY_D, tz=_NY).normalize()
  assert excluded not in dt.index


def test_truncated_day_day_table_still_has_three_rows() -> None:
  """Adding a truncated day keeps the resolved count at 3."""
  dt = _stat().build_day_table(_make_three_days_with_truncated())
  assert len(dt) == 3


def test_truncated_day_does_not_change_mean_green() -> None:
  """Truncated day doesn't affect the mean green streak value."""
  stat = _stat()
  dt_clean = stat.build_day_table(_make_three_days())
  dt_trunc = stat.build_day_table(_make_three_days_with_truncated())
  rows_clean = stat.compute_rows(dt_clean)
  rows_trunc = stat.compute_rows(dt_trunc)
  # mean([3,1,2]) = 2.0 in both cases
  assert _row(rows_trunc, "green").value == pytest.approx(_row(rows_clean, "green").value)


def test_truncated_day_does_not_change_total_samples() -> None:
  """total_samples == 3 whether or not the truncated day is in the data."""
  result_clean = _stat().compute(_make_three_days())
  result_trunc = _stat().compute(_make_three_days_with_truncated())
  assert _tf(result_clean).total_samples == 3
  assert _tf(result_trunc).total_samples == 3


def test_single_resolved_day_only() -> None:
  """Single resolved day: day_table has 1 row, value=max streak for that day."""
  df = _make_streak_day(_DAY_A, _COLORS_A)
  stat = _stat()
  dt = stat.build_day_table(df)
  assert len(dt) == 1
  rows = stat.compute_rows(dt)
  # Day A: max_green=3, max_red=1 → mean over 1 day = the value itself
  assert _row(rows, "green").value == pytest.approx(3.0)
  assert _row(rows, "red").value == pytest.approx(1.0)


# ===========================================================================
# 5. Reproducibility / determinism
# ===========================================================================

def test_compute_same_seed_identical_value() -> None:
  """compute(df, seed=42) twice yields identical value on both calls."""
  df = _make_three_days()
  stat = _stat()
  r1 = stat.compute(df, seed=42)
  r2 = stat.compute(df, seed=42)
  rows1 = _tf(r1).results
  rows2 = _tf(r2).results
  for a, b in zip(rows1, rows2):
    assert a.condition == b.condition
    assert a.value == pytest.approx(b.value)
    assert a.value_baseline == pytest.approx(b.value_baseline)


def test_compute_same_seed_identical_model_dump() -> None:
  """Full model_dump() is identical across two calls with the same seed."""
  df = _make_three_days()
  stat = _stat()
  assert stat.compute(df, seed=42).model_dump() == stat.compute(df, seed=42).model_dump()


def test_baseline_rows_same_seed_deterministic() -> None:
  """baseline_rows with seed=42 gives the same values on repeated calls."""
  stat = _stat()
  dt = stat.build_day_table(_make_three_days())
  bl_a = stat.baseline_rows(dt, seed=42)
  bl_b = stat.baseline_rows(dt, seed=42)
  assert len(bl_a) == len(bl_b)
  for a, b in zip(bl_a, bl_b):
    assert a.condition == b.condition
    assert a.value == pytest.approx(b.value)
    assert a.count == b.count
    assert a.total == b.total


def test_baseline_rows_different_seeds_can_differ() -> None:
  """Different seeds can produce different baseline values.

  With 3 days each having a non-trivial color distribution, two seeds should
  produce different shuffled streak averages with high probability.
  """
  stat = _stat()
  dt = stat.build_day_table(_make_three_days())
  bl_1 = stat.baseline_rows(dt, seed=1)
  bl_99 = stat.baseline_rows(dt, seed=99)
  any_diff = any(
    abs((a.value or 0.0) - (b.value or 0.0)) > 1e-9
    for a, b in zip(bl_1, bl_99)
  )
  assert any_diff, "All baseline values are identical for seed=1 vs seed=99"


# ===========================================================================
# 6. Baseline preserves per-day color counts / correct N
# ===========================================================================

def test_baseline_rows_n_equals_day_table_length() -> None:
  """baseline_rows count and total equal the number of contributing days."""
  stat = _stat()
  dt = stat.build_day_table(_make_three_days())
  bl = stat.baseline_rows(dt, seed=42)
  # 3 resolved days → count=total=3 for both green and red baseline rows
  for r in bl:
    assert r.count == 3
    assert r.total == 3


def test_baseline_rows_value_not_none_for_non_empty() -> None:
  """baseline_rows returns non-None values when day_table is non-empty."""
  stat = _stat()
  dt = stat.build_day_table(_make_three_days())
  bl = stat.baseline_rows(dt, seed=42)
  for r in bl:
    assert r.value is not None


def test_baseline_rows_value_is_positive() -> None:
  """Baseline streak means are positive (each day has at least 1 green or red bar)."""
  stat = _stat()
  dt = stat.build_day_table(_make_three_days())
  bl = stat.baseline_rows(dt, seed=42)
  for r in bl:
    assert r.value is not None
    assert r.value > 0.0


def test_baseline_rows_outcome_key() -> None:
  """baseline_rows rows carry outcome='max_streak'."""
  stat = _stat()
  dt = stat.build_day_table(_make_three_days())
  for r in stat.baseline_rows(dt, seed=42):
    assert r.outcome == "max_streak"


def test_baseline_rows_two_rows() -> None:
  """baseline_rows returns exactly 2 rows: green and red."""
  stat = _stat()
  dt = stat.build_day_table(_make_three_days())
  bl = stat.baseline_rows(dt, seed=42)
  assert len(bl) == 2
  assert {r.condition for r in bl} == {"green", "red"}


def test_baseline_probability_channels_zero() -> None:
  """baseline_rows rows: probability=0.0, baseline_prob=0.0."""
  stat = _stat()
  dt = stat.build_day_table(_make_three_days())
  for r in stat.baseline_rows(dt, seed=42):
    assert r.probability == pytest.approx(0.0)
    assert r.baseline_prob == pytest.approx(0.0)


# ===========================================================================
# 7. Empty input edge cases
# ===========================================================================

def test_empty_build_day_table_returns_empty_dataframe() -> None:
  """build_day_table on empty input returns empty DataFrame with the 3 columns."""
  dt = _stat().build_day_table(_empty_df())
  assert len(dt) == 0
  assert set(dt.columns) == {"max_green_streak", "max_red_streak", "bar_colors"}


def test_empty_compute_rows_two_rows_value_none() -> None:
  """compute_rows on empty day_table → 2 rows with value=None, count=0."""
  stat = _stat()
  dt = stat.build_day_table(_empty_df())
  rows = stat.compute_rows(dt)
  assert len(rows) == 2
  for r in rows:
    assert r.value is None
    assert r.count == 0
    assert r.total == 0


def test_empty_baseline_rows_two_rows_value_none() -> None:
  """baseline_rows on empty day_table → 2 rows with value=None, count=0."""
  stat = _stat()
  dt = stat.build_day_table(_empty_df())
  bl = stat.baseline_rows(dt, seed=42)
  assert len(bl) == 2
  for r in bl:
    assert r.value is None
    assert r.count == 0
    assert r.total == 0


def test_empty_compute_total_samples_zero() -> None:
  """compute() on empty input → total_samples=0."""
  assert _tf(_stat().compute(_empty_df())).total_samples == 0


def test_empty_compute_results_two_rows() -> None:
  """compute() on empty input still returns exactly 2 result rows."""
  rows = _tf(_stat().compute(_empty_df())).results
  assert len(rows) == 2
  for r in rows:
    assert r.value is None
    assert r.count == 0


def test_empty_compute_data_range_empty() -> None:
  """compute() on empty input → data_range=[]."""
  assert _tf(_stat().compute(_empty_df())).data_range == []


def test_empty_compute_no_slices() -> None:
  """avg_consecutive_bars declares no slices → slices={}."""
  assert _tf(_stat().compute(_empty_df())).slices == {}


# ===========================================================================
# 8. Multiple timeframes / invalid timeframe
# ===========================================================================

def test_invalid_timeframe_raises_value_error() -> None:
  """Unknown timeframe string raises ValueError."""
  with pytest.raises(ValueError, match="Unknown timeframe"):
    AvgConsecutiveBars(instrument="NQ", config=_TEST_CONFIG, timeframe="2h")


def test_invalid_timeframe_message_mentions_name() -> None:
  """ValueError message includes the offending timeframe string."""
  with pytest.raises(ValueError, match="badtf"):
    AvgConsecutiveBars(instrument="NQ", config=_TEST_CONFIG, timeframe="badtf")


def test_15min_timeframe_accepted() -> None:
  """15min is a valid timeframe."""
  stat = _stat(timeframe="15min")
  assert stat.bucket_min == 15


def test_1h_timeframe_accepted() -> None:
  """1h is a valid timeframe."""
  stat = _stat(timeframe="1h")
  assert stat.bucket_min == 60


def test_15min_fewer_bars_than_5min() -> None:
  """15min buckets produce fewer bars per day than 5min buckets.

  With 1 bar per 5min bucket (explicit + sentinel) the 5min config sees ~8
  color entries for Day A.  At 15min, the same bars collapse into fewer buckets
  (bucket_idx=(mod-570)//15), so bar_colors is shorter.
  """
  df = _make_streak_day(_DAY_A, _COLORS_A)  # mods at 570,575,580,585,590,595,600 + 960

  stat_5 = AvgConsecutiveBars(instrument="NQ", config=_TEST_CONFIG, timeframe="5min")
  stat_15 = AvgConsecutiveBars(instrument="NQ", config=_TEST_CONFIG_15, timeframe="15min")

  dt_5 = stat_5.build_day_table(df)
  dt_15 = stat_15.build_day_table(df)

  date_a = pd.Timestamp(_DAY_A, tz=_NY).normalize()
  bc_5 = dt_5.loc[date_a, "bar_colors"]
  bc_15 = dt_15.loc[date_a, "bar_colors"]

  # 5min: 8 colour entries (7 explicit + sentinel at bucket 78)
  # 15min: bars at mods 570,575,580 → all in bucket 0 (first bar open, last bar close)
  #         mods 585,590,595 → bucket 1; mod 600 → bucket 2; sentinel mod=960 → bucket 26
  #        So 15min has 4 colour entries vs 8 for 5min
  assert len(bc_15) < len(bc_5), (
    f"15min bars ({len(bc_15)}) should be fewer than 5min bars ({len(bc_5)})"
  )


def test_15min_bucket_aggregation_correct() -> None:
  """15min bucket aggregates first bar's open and last bar's close.

  Day A mods in bucket 0 (15min): 570 (open=100, G), 575 (open=100, G), 580 (open=100, R)
    bucket_open  = first bar's open = 100 (bar at mod=570)
    bucket_close = last bar's close = 98  (bar at mod=580, red)
    → bucket is RED (98 < 100)

  Day A mods in bucket 1 (15min): 585 (G), 590 (G), 595 (G)
    bucket_open=100, bucket_close=102 → GREEN

  Day A mods in bucket 2 (15min): 600 (R)
    bucket_open=100, bucket_close=98 → RED

  Sentinel mod=960 → 15min bucket 26: GREEN (open=close=100)

  bar_colors for Day A at 15min = [F, T, F, T]  (4 entries: buckets 0,1,2,26)
  max_green = 1, max_red = 2  (bucket 0 and bucket 2 are red but separated by green)
  Wait, that's: [F, T, F, T]:
    Runs: F(1), T(1), F(1), T(1) → max_red=1, max_green=1
  Actually max_red=1 not 2. Let me re-verify: F(1), T(1), F(1), T(1) → longest F=1.
  """
  df = _make_streak_day(_DAY_A, _COLORS_A)
  stat_15 = AvgConsecutiveBars(instrument="NQ", config=_TEST_CONFIG_15, timeframe="15min")
  dt = stat_15.build_day_table(df)
  date_a = pd.Timestamp(_DAY_A, tz=_NY).normalize()

  # bar_colors=[F,T,F,T]: Runs: F(1),T(1),F(1),T(1) → max_green=1, max_red=1
  assert dt.loc[date_a, "max_green_streak"] == pytest.approx(1.0)
  assert dt.loc[date_a, "max_red_streak"] == pytest.approx(1.0)


def test_1h_timeframe_compute_single_day() -> None:
  """1h timeframe computes on a single resolved day without error.

  At 1h bucket_min=60, bars at mods 570,575,...,600 + sentinel at 960:
    bucket 0 (570-629): bars at 570,575,580,585,590,595,600
      bucket_open=100 (first bar mod=570), bucket_close=98 (last bar mod=600, red)
      → bucket 0 is RED
    bucket 6 (930-989): sentinel at mod=960, close=100=open → GREEN

  So bar_colors=[F, T] → max_green=1, max_red=1.
  """
  df = _make_streak_day(_DAY_A, _COLORS_A)
  stat_1h = AvgConsecutiveBars(instrument="NQ", config=_TEST_CONFIG_1H, timeframe="1h")
  result = stat_1h.compute(df)
  tf = result.instruments["NQ"]["1h"]
  assert tf.total_samples == 1
  rows = tf.results
  # bar_colors=[F, T]: max_green=1, max_red=1 → value=1.0 for both
  g = _row(rows, "green")
  r = _row(rows, "red")
  assert g.value == pytest.approx(1.0)
  assert r.value == pytest.approx(1.0)


# ===========================================================================
# 9. compute() shape / StatRunResult validation
# ===========================================================================

def test_compute_returns_stat_run_result_instance() -> None:
  """compute() returns a StatRunResult instance."""
  assert isinstance(_stat().compute(_make_three_days()), StatRunResult)


def test_compute_stat_name() -> None:
  """stat_name == 'avg_consecutive_bars'."""
  assert _stat().compute(_make_three_days()).stat_name == "avg_consecutive_bars"


def test_compute_instrument_key() -> None:
  """Result instruments dict contains 'NQ'."""
  result = _stat().compute(_make_three_days())
  assert "NQ" in result.instruments


def test_compute_timeframe_key() -> None:
  """Result instruments['NQ'] contains '5min'."""
  result = _stat().compute(_make_three_days())
  assert "5min" in result.instruments["NQ"]


def test_compute_total_samples_equals_resolved_days() -> None:
  """total_samples == 3 for 3 resolved days."""
  assert _tf(_stat().compute(_make_three_days())).total_samples == 3


def test_compute_data_range() -> None:
  """data_range spans 2024-01-08 to 2024-01-10."""
  dr = _tf(_stat().compute(_make_three_days())).data_range
  assert dr[0] == "2024-01-08"
  assert dr[1] == "2024-01-10"


def test_compute_exactly_two_result_rows() -> None:
  """compute() result has exactly 2 rows: one green, one red."""
  rows = _tf(_stat().compute(_make_three_days())).results
  assert len(rows) == 2
  assert {r.condition for r in rows} == {"green", "red"}


def test_compute_outcome_keys() -> None:
  """All result rows carry outcome='max_streak'."""
  for r in _tf(_stat().compute(_make_three_days())).results:
    assert r.outcome == "max_streak"


def test_compute_probability_channels_zero() -> None:
  """probability and baseline_prob are 0.0 on every result row."""
  for r in _tf(_stat().compute(_make_three_days())).results:
    assert r.probability == pytest.approx(0.0)
    assert r.baseline_prob == pytest.approx(0.0)


def test_compute_no_slices() -> None:
  """avg_consecutive_bars declares no slices → slices == {}."""
  assert _tf(_stat().compute(_make_three_days())).slices == {}


def test_compute_green_value_correct() -> None:
  """compute() green value = (3+1+2)/3 = 2.0."""
  rows = _tf(_stat().compute(_make_three_days())).results
  # mean([3,1,2]) = 2.0
  assert _row(rows, "green").value == pytest.approx(2.0)


def test_compute_red_value_correct() -> None:
  """compute() red value = (1+2+4)/3 = 7/3."""
  rows = _tf(_stat().compute(_make_three_days())).results
  # mean([1,2,4]) = 7/3
  assert _row(rows, "red").value == pytest.approx(7 / 3)


def test_compute_value_baseline_non_none_for_non_empty() -> None:
  """compute() merges baseline into value_baseline (non-None when data exists)."""
  rows = _tf(_stat().compute(_make_three_days())).results
  for r in rows:
    assert r.value_baseline is not None


def test_compute_baseline_n_equals_resolved_days() -> None:
  """baseline_n == 3 (all resolved days contribute to the baseline)."""
  rows = _tf(_stat().compute(_make_three_days())).results
  for r in rows:
    assert r.baseline_n == 3


def test_compute_i18n_title_non_empty() -> None:
  """title has non-empty en and fr strings."""
  result = _stat().compute(_make_three_days())
  assert result.title.en != ""
  assert result.title.fr != ""


def test_compute_i18n_definition_non_empty() -> None:
  """definition has non-empty en and fr strings."""
  result = _stat().compute(_make_three_days())
  assert result.definition.en != ""
  assert result.definition.fr != ""


def test_compute_i18n_condition_labels() -> None:
  """labels.conditions has 'green' and 'red' with non-empty en/fr."""
  result = _stat().compute(_make_three_days())
  assert set(result.labels.conditions) == {"green", "red"}
  for key, lbl in result.labels.conditions.items():
    assert lbl.en != "", f"condition {key!r}.en empty"
    assert lbl.fr != "", f"condition {key!r}.fr empty"


def test_compute_i18n_outcome_labels() -> None:
  """labels.outcomes has 'max_streak' with non-empty en/fr."""
  result = _stat().compute(_make_three_days())
  assert "max_streak" in result.labels.outcomes
  lbl = result.labels.outcomes["max_streak"]
  assert lbl.en != ""
  assert lbl.fr != ""


# ===========================================================================
# 10. write_results round-trip
# ===========================================================================

def test_write_results_file_named_correctly(tmp_path: Path) -> None:
  """write_results produces 'avg_consecutive_bars.json'."""
  result = _stat().compute(_make_three_days())
  written = write_results(result, results_dir=tmp_path)
  assert written.name == "avg_consecutive_bars.json"


def test_write_results_round_trip_stat_name(tmp_path: Path) -> None:
  """JSON contains stat_name == 'avg_consecutive_bars'."""
  result = _stat().compute(_make_three_days())
  written = write_results(result, results_dir=tmp_path)
  raw = json.loads(written.read_text(encoding="utf-8"))
  assert raw["stat_name"] == "avg_consecutive_bars"


def test_write_results_round_trip_values(tmp_path: Path) -> None:
  """Values survive JSON serialisation and Pydantic re-validation."""
  result = _stat().compute(_make_three_days())
  written = write_results(result, results_dir=tmp_path)
  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)
  tf = validated.instruments["NQ"]["5min"]
  assert tf.total_samples == 3
  g = _row(tf.results, "green")
  r = _row(tf.results, "red")
  # mean([3,1,2])=2.0; mean([1,2,4])=7/3
  assert g.value == pytest.approx(2.0)
  assert r.value == pytest.approx(7 / 3)
  assert g.probability == pytest.approx(0.0)
  assert r.probability == pytest.approx(0.0)


def test_write_results_french_accent_preserved(tmp_path: Path) -> None:
  """French accents in the JSON are stored as literals, not \\uXXXX escapes."""
  result = _stat().compute(_empty_df())
  written = write_results(result, results_dir=tmp_path)
  raw_text = written.read_text(encoding="utf-8")
  # The French definition contains accented characters (é, è, etc.)
  assert "\\u00e9" not in raw_text, "French é stored as unicode escape instead of literal"


# ===========================================================================
# 11. classify_samples()
#
# Reuses the three-day base dataset (see module docstring):
#   Day A 2024-01-08: max_green=3, max_red=1
#   Day B 2024-01-09: max_green=1, max_red=2
#   Day C 2024-01-10: max_green=2, max_red=4
# ===========================================================================

def test_classify_samples_exact_rows() -> None:
  """classify_samples() emits exactly the 6 expected SampleRows."""
  stat = _stat()
  dt = stat.build_day_table(_make_three_days())
  samples = stat.classify_samples(dt)

  expected = [
    SampleRow(date="2024-01-08", condition="green", outcome="max_streak", value=3.0),
    SampleRow(date="2024-01-08", condition="red", outcome="max_streak", value=1.0),
    SampleRow(date="2024-01-09", condition="green", outcome="max_streak", value=1.0),
    SampleRow(date="2024-01-09", condition="red", outcome="max_streak", value=2.0),
    SampleRow(date="2024-01-10", condition="green", outcome="max_streak", value=2.0),
    SampleRow(date="2024-01-10", condition="red", outcome="max_streak", value=4.0),
  ]

  assert len(samples) == len(expected)
  for got, want in zip(samples, expected):
    assert got.date == want.date
    assert got.condition == want.condition
    assert got.outcome == want.outcome
    assert got.value == pytest.approx(want.value)


def test_classify_samples_matches_compute_rows() -> None:
  """Per-condition sample count/mean reconstruct the max_streak row."""
  stat = _stat()
  dt = stat.build_day_table(_make_three_days())
  samples = stat.classify_samples(dt)
  rows = stat.compute_rows(dt)

  for condition in ("green", "red"):
    values = [s.value for s in samples if s.condition == condition]
    row = _row(rows, condition)
    assert len(values) == row.total
    assert sum(values) / len(values) == pytest.approx(row.value)


def test_classify_samples_empty_day_table() -> None:
  """Empty day_table -> []."""
  stat = _stat()
  dt = stat.build_day_table(_empty_df())
  assert stat.classify_samples(dt) == []


def test_classify_samples_skips_nan_column_independently() -> None:
  """A NaN in one streak column produces no sample for that condition only.

  ``build_day_table`` currently drops a day entirely if either streak column is
  NaN, but ``classify_samples`` must still honor per-column NaN independently
  (defensive against future per-color pending columns).
  """
  stat = _stat()
  dt = pd.DataFrame(
    {
      "max_green_streak": [3.0, np.nan],
      "max_red_streak": [np.nan, 4.0],
      "bar_colors": [None, None],
    },
    index=pd.DatetimeIndex(["2024-01-08", "2024-01-09"]),
  )
  samples = stat.classify_samples(dt)
  assert len(samples) == 2
  assert samples[0].date == "2024-01-08"
  assert samples[0].condition == "green"
  assert samples[0].value == pytest.approx(3.0)
  assert samples[1].date == "2024-01-09"
  assert samples[1].condition == "red"
  assert samples[1].value == pytest.approx(4.0)
