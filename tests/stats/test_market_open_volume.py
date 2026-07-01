"""Tests for stats.market_open_volume.standard.

All data is synthetic — no real market files required.
Expected values are hand-calculated before each assertion.

RTH constants:
  RTH start = 09:30 = 570 min
  RTH end   = 16:15 = 975 min (exclusive)
  Last RTH bar mod = 974 (16:14)
  Close threshold  = 960 (16:00); last bar mod must be >= this for resolution.

Window arithmetic for supported timeframes (open_bar_min):
  "15min" -> open_bar_min=15: open=[570,585)=15 bars, rest=[585,975)=390 bars
  "30min" -> open_bar_min=30: open=[570,600)=30 bars, rest=[600,975)=375 bars
  "1h"    -> open_bar_min=60: open=[570,630)=60 bars, rest=[630,975)=345 bars

Per-day volume sums (constant vol_per_bar across window):
  open_volume = open_vol_per_bar  * n_open_bars
  rest_volume = rest_vol_per_bar  * n_rest_bars

Resolution criterion (build_resolved_days):
  - bar at mod=570 must exist (clean session open)
  - last RTH bar mod >= rth_end - close_tolerance_min = 975 - 15 = 960
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import SampleRow, StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.market_open_volume.standard import MarketOpenVolume

# ---------------------------------------------------------------------------
# Minimal InstrumentConfig — does NOT depend on NQ.yaml
# ---------------------------------------------------------------------------
_NY = "America/New_York"
_RTH_SESSION = Session(start="09:30", end="16:15")
_TEST_CONFIG = InstrumentConfig(
  instrument="NQ",
  working_timezone=_NY,
  sessions={"rth": _RTH_SESSION},
  timeframes=["15min", "30min", "1h"],
  parquet_path=Path("data/NQ_1min.parquet"),
)

# RTH constants
_RTH_START = 570   # 09:30
_RTH_END = 975     # 16:15 (exclusive)
_RTH_LAST = 974    # 16:14 (last 1-min bar in RTH)
_CLOSE_THRESH = 960  # 16:00 — last bar mod must be >= this


# ---------------------------------------------------------------------------
# Helper accessors
# ---------------------------------------------------------------------------

def _stat(timeframe: str = "1h") -> MarketOpenVolume:
  return MarketOpenVolume(instrument="NQ", config=_TEST_CONFIG, timeframe=timeframe)


def _tf(result: StatRunResult, timeframe: str = "1h"):  # type: ignore[return]
  return result.instruments["NQ"][timeframe]


def _row(rows: list[StatResultRow], condition: str, outcome: str) -> StatResultRow:
  for r in rows:
    if r.condition == condition and r.outcome == outcome:
      return r
  raise KeyError(f"({condition!r}, {outcome!r}) not found in rows")


# ---------------------------------------------------------------------------
# Synthetic day builders
# ---------------------------------------------------------------------------

def _make_rth_day(
  date: str,
  open_vol_per_bar: int,
  rest_vol_per_bar: int,
  open_bar_min: int = 60,
  price: float = 100.0,
) -> pd.DataFrame:
  """Build one fully resolved RTH day (mods 570..974) with controlled volume.

  - open window = [570, 570+open_bar_min): each bar carries open_vol_per_bar
  - rest window = [570+open_bar_min, 975): each bar carries rest_vol_per_bar
  - OHLC is flat/neutral at ``price`` with tiny spread for validity.
  - The day spans all 405 RTH bars (mods 570..974 inclusive) so it is always
    resolved: bar at mod=570 exists and last bar mod=974 >= 960.

  Hand-calculated volume sums:
    open_volume = open_vol_per_bar  * open_bar_min
    rest_volume = rest_vol_per_bar  * (975 - 570 - open_bar_min)
  """
  base = pd.Timestamp(date, tz=_NY)
  open_end_mod = _RTH_START + open_bar_min
  records = []
  for mod in range(_RTH_START, _RTH_END):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    vol = open_vol_per_bar if mod < open_end_mod else rest_vol_per_bar
    records.append({
      "timestamp": ts,
      "open": price,
      "high": price + 0.25,
      "low": price - 0.25,
      "close": price,
      "volume": vol,
    })
  return pd.DataFrame(records)


def _make_truncated_day(date: str, last_mod: int = 590) -> pd.DataFrame:
  """A day whose last RTH bar is at ``last_mod`` < 960 — must be excluded.

  Default: last bar at mod=590 (09:50 ET).
  """
  base = pd.Timestamp(date, tz=_NY)
  records = []
  for mod in range(_RTH_START, last_mod + 1):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    records.append({
      "timestamp": ts,
      "open": 100.0,
      "high": 101.0,
      "low": 99.0,
      "close": 100.0,
      "volume": 500,
    })
  return pd.DataFrame(records)


def _concat_days(frames: list[pd.DataFrame]) -> pd.DataFrame:
  """Concatenate per-day DataFrames into a single sorted 1-min OHLCV frame."""
  df = pd.concat(frames, ignore_index=True)
  return df.sort_values("timestamp").reset_index(drop=True)


def _empty_df() -> pd.DataFrame:
  df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
  df["timestamp"] = pd.array([], dtype="datetime64[ns, America/New_York]")
  return df


# ---------------------------------------------------------------------------
# Concrete 3-day datasets with hand-calculated expectations
#
# For "1h" timeframe (open_bar_min=60):
#   n_open_bars = 60, n_rest_bars = 975-570-60 = 345
#
#   Day A  2024-01-08 Mon: open_vol_per_bar=10, rest_vol_per_bar=5
#     open_volume_A = 10 * 60 = 600
#     rest_volume_A = 5  * 345 = 1725
#
#   Day B  2024-01-09 Tue: open_vol_per_bar=20, rest_vol_per_bar=10
#     open_volume_B = 20 * 60 = 1200
#     rest_volume_B = 10 * 345 = 3450
#
#   Day C  2024-01-10 Wed: open_vol_per_bar=30, rest_vol_per_bar=15
#     open_volume_C = 30 * 60 = 1800
#     rest_volume_C = 15 * 345 = 5175
#
#   rest_volume = 2.875 * open_volume for all three days:
#     1725/600=2.875, 3450/1200=2.875, 5175/1800=2.875
#   => perfectly positively correlated: r = 1.0
#
# For "15min" timeframe (open_bar_min=15):
#   n_open_bars = 15, n_rest_bars = 975-570-15 = 390
#
#   Day A  2024-01-08: open_vol_per_bar=10, rest_vol_per_bar=5
#     open_volume_A = 10 * 15 = 150
#     rest_volume_A = 5  * 390 = 1950
#
#   Day B  2024-01-09: open_vol_per_bar=20, rest_vol_per_bar=10
#     open_volume_B = 20 * 15 = 300
#     rest_volume_B = 10 * 390 = 3900
#
#   Day C  2024-01-10: open_vol_per_bar=30, rest_vol_per_bar=15
#     open_volume_C = 30 * 15 = 450
#     rest_volume_C = 15 * 390 = 5850
#
#   rest_volume = 13 * open_volume => r = 1.0
#
# For perfect negative correlation (1h timeframe):
#   open_volumes:  600, 1200, 1800  (increasing)
#   rest_volumes:  5175, 3450, 1725 (decreasing)
#   => r = -1.0
#
# For zero-variance open (1h timeframe):
#   All days have same open_volume (std=0) -> r = 0.0
# ---------------------------------------------------------------------------

# 1h constants
_N_OPEN_1H = 60
_N_REST_1H = 975 - 570 - 60  # 345

# 15min constants
_N_OPEN_15 = 15
_N_REST_15 = 975 - 570 - 15  # 390

# 30min constants
_N_OPEN_30 = 30
_N_REST_30 = 975 - 570 - 30  # 375

# Perfectly positively correlated days (1h timeframe)
_OV_A_1H = 10 * _N_OPEN_1H   # 600
_OV_B_1H = 20 * _N_OPEN_1H   # 1200
_OV_C_1H = 30 * _N_OPEN_1H   # 1800
_RV_A_1H = 5  * _N_REST_1H   # 1725
_RV_B_1H = 10 * _N_REST_1H   # 3450
_RV_C_1H = 15 * _N_REST_1H   # 5175

# Perfectly positively correlated days (15min timeframe)
_OV_A_15 = 10 * _N_OPEN_15   # 150
_OV_B_15 = 20 * _N_OPEN_15   # 300
_OV_C_15 = 30 * _N_OPEN_15   # 450
_RV_A_15 = 5  * _N_REST_15   # 1950
_RV_B_15 = 10 * _N_REST_15   # 3900
_RV_C_15 = 15 * _N_REST_15   # 5850


def _make_pos_corr_days_1h() -> pd.DataFrame:
  """Three days perfectly positively correlated for 1h timeframe.

  open_volumes:  600, 1200, 1800  (all scale factors differ)
  rest_volumes: 1725, 3450, 5175  (same scale factor 2.875x)
  Pearson r = 1.0
  """
  return _concat_days([
    _make_rth_day("2024-01-08", open_vol_per_bar=10, rest_vol_per_bar=5,  open_bar_min=60),
    _make_rth_day("2024-01-09", open_vol_per_bar=20, rest_vol_per_bar=10, open_bar_min=60),
    _make_rth_day("2024-01-10", open_vol_per_bar=30, rest_vol_per_bar=15, open_bar_min=60),
  ])


def _make_neg_corr_days_1h() -> pd.DataFrame:
  """Three days perfectly negatively correlated for 1h timeframe.

  open_volumes:  600, 1200, 1800  (increasing)
  rest_volumes: 5175, 3450, 1725  (decreasing, same magnitudes reversed)
  Pearson r = -1.0
  """
  return _concat_days([
    _make_rth_day("2024-01-08", open_vol_per_bar=10, rest_vol_per_bar=15, open_bar_min=60),
    _make_rth_day("2024-01-09", open_vol_per_bar=20, rest_vol_per_bar=10, open_bar_min=60),
    _make_rth_day("2024-01-10", open_vol_per_bar=30, rest_vol_per_bar=5,  open_bar_min=60),
  ])


def _make_const_open_days_1h() -> pd.DataFrame:
  """Three days with constant open_volume (zero variance) for 1h timeframe.

  All open_volumes identical -> std=0 -> r = 0.0
  rest_volumes vary to make the zero-variance case non-trivial.
  """
  return _concat_days([
    _make_rth_day("2024-01-08", open_vol_per_bar=20, rest_vol_per_bar=5,  open_bar_min=60),
    _make_rth_day("2024-01-09", open_vol_per_bar=20, rest_vol_per_bar=10, open_bar_min=60),
    _make_rth_day("2024-01-10", open_vol_per_bar=20, rest_vol_per_bar=15, open_bar_min=60),
  ])


def _make_pos_corr_days_15min() -> pd.DataFrame:
  """Three days perfectly positively correlated for 15min timeframe."""
  return _concat_days([
    _make_rth_day("2024-01-08", open_vol_per_bar=10, rest_vol_per_bar=5,  open_bar_min=15),
    _make_rth_day("2024-01-09", open_vol_per_bar=20, rest_vol_per_bar=10, open_bar_min=15),
    _make_rth_day("2024-01-10", open_vol_per_bar=30, rest_vol_per_bar=15, open_bar_min=15),
  ])


# ===========================================================================
# 1. build_day_table correctness
# ===========================================================================

def test_build_day_table_1h_index_is_resolved_dates() -> None:
  """Day table (1h) index contains exactly the 3 resolved session dates."""
  df = _make_pos_corr_days_1h()
  dt = _stat("1h").build_day_table(df)
  expected = {
    pd.Timestamp("2024-01-08", tz=_NY).normalize(),
    pd.Timestamp("2024-01-09", tz=_NY).normalize(),
    pd.Timestamp("2024-01-10", tz=_NY).normalize(),
  }
  assert set(dt.index) == expected


def test_build_day_table_1h_open_volume_day_a() -> None:
  """Day A (1h): open_volume = 10 bars_per_min * 60 bars = 600."""
  # open_volume_A = 10 * 60 = 600
  dt = _stat("1h").build_day_table(_make_pos_corr_days_1h())
  date_a = pd.Timestamp("2024-01-08", tz=_NY).normalize()
  assert dt.loc[date_a, "open_volume"] == pytest.approx(600.0)


def test_build_day_table_1h_rest_volume_day_a() -> None:
  """Day A (1h): rest_volume = 5 * 345 = 1725."""
  # rest_volume_A = 5 * 345 = 1725
  dt = _stat("1h").build_day_table(_make_pos_corr_days_1h())
  date_a = pd.Timestamp("2024-01-08", tz=_NY).normalize()
  assert dt.loc[date_a, "rest_volume"] == pytest.approx(float(_RV_A_1H))  # 1725


def test_build_day_table_1h_open_volume_day_b() -> None:
  """Day B (1h): open_volume = 20 * 60 = 1200."""
  dt = _stat("1h").build_day_table(_make_pos_corr_days_1h())
  date_b = pd.Timestamp("2024-01-09", tz=_NY).normalize()
  assert dt.loc[date_b, "open_volume"] == pytest.approx(float(_OV_B_1H))  # 1200


def test_build_day_table_1h_rest_volume_day_b() -> None:
  """Day B (1h): rest_volume = 10 * 345 = 3450."""
  dt = _stat("1h").build_day_table(_make_pos_corr_days_1h())
  date_b = pd.Timestamp("2024-01-09", tz=_NY).normalize()
  assert dt.loc[date_b, "rest_volume"] == pytest.approx(float(_RV_B_1H))  # 3450


def test_build_day_table_15min_open_volume_day_a() -> None:
  """Day A (15min): open_volume = 10 * 15 = 150."""
  # open_volume_A = 10 * 15 = 150
  dt = _stat("15min").build_day_table(_make_pos_corr_days_15min())
  date_a = pd.Timestamp("2024-01-08", tz=_NY).normalize()
  assert dt.loc[date_a, "open_volume"] == pytest.approx(float(_OV_A_15))  # 150


def test_build_day_table_15min_rest_volume_day_c() -> None:
  """Day C (15min): rest_volume = 15 * 390 = 5850."""
  # rest_volume_C = 15 * 390 = 5850
  dt = _stat("15min").build_day_table(_make_pos_corr_days_15min())
  date_c = pd.Timestamp("2024-01-10", tz=_NY).normalize()
  assert dt.loc[date_c, "rest_volume"] == pytest.approx(float(_RV_C_15))  # 5850


def test_build_day_table_columns() -> None:
  """build_day_table returns DataFrame with exactly open_volume and rest_volume."""
  dt = _stat("1h").build_day_table(_make_pos_corr_days_1h())
  assert list(dt.columns) == ["open_volume", "rest_volume"]


def test_build_day_table_truncated_day_excluded() -> None:
  """A truncated day (last bar mod=590 < 960) is excluded from the day table.

  Three resolved days + one truncated -> day table has exactly 3 rows.
  """
  df = _concat_days([
    _make_pos_corr_days_1h(),
    _make_truncated_day("2024-01-15"),
  ])
  dt = _stat("1h").build_day_table(df)
  assert len(dt) == 3
  excluded = pd.Timestamp("2024-01-15", tz=_NY).normalize()
  assert excluded not in dt.index


def test_build_day_table_empty_input_returns_empty_frame() -> None:
  """build_day_table on empty input returns an empty DataFrame."""
  dt = _stat("1h").build_day_table(_empty_df())
  assert len(dt) == 0
  assert list(dt.columns) == ["open_volume", "rest_volume"]


# ===========================================================================
# 2. Known correlation values
# ===========================================================================

def test_perfect_positive_correlation_1h() -> None:
  """Perfectly positively correlated days -> r = 1.0.

  open_volumes: 600, 1200, 1800 (A, B, C)
  rest_volumes: 1725, 3450, 5175 (all = 2.875 * open_volume)
  Pearson r between two perfectly linearly related series = 1.0.
  """
  result = _stat("1h").compute(_make_pos_corr_days_1h())
  r = _row(_tf(result, "1h").results, "any_day", "correlation")
  assert r.value == pytest.approx(1.0)


def test_perfect_positive_correlation_15min() -> None:
  """Perfectly positively correlated days (15min timeframe) -> r = 1.0.

  open_volumes: 150, 300, 450
  rest_volumes: 1950, 3900, 5850 (all = 13 * open_volume)
  Pearson r = 1.0.
  """
  result = _stat("15min").compute(_make_pos_corr_days_15min())
  r = _row(_tf(result, "15min").results, "any_day", "correlation")
  assert r.value == pytest.approx(1.0)


def test_perfect_negative_correlation_1h() -> None:
  """Perfectly negatively correlated days -> r = -1.0.

  open_volumes: 600, 1200, 1800 (A, B, C — increasing)
  rest_volumes: 5175, 3450, 1725 (C, B, A — decreasing same magnitudes)
  Pearson r between two perfectly anti-linear series = -1.0.
  """
  result = _stat("1h").compute(_make_neg_corr_days_1h())
  r = _row(_tf(result, "1h").results, "any_day", "correlation")
  assert r.value == pytest.approx(-1.0)


def test_zero_variance_open_produces_r_zero() -> None:
  """Constant open_volume (zero variance) -> r = 0.0 (degenerate guard).

  All days: open_vol_per_bar=20 -> open_volume = 20*60 = 1200 (constant)
  std(open_volume) = 0 -> correlation undefined -> return 0.0.
  """
  result = _stat("1h").compute(_make_const_open_days_1h())
  r = _row(_tf(result, "1h").results, "any_day", "correlation")
  assert r.value == pytest.approx(0.0)


# ===========================================================================
# 3. Row shape
# ===========================================================================

def test_exactly_one_result_row() -> None:
  """compute() returns exactly ONE result row (no slices, single condition/outcome)."""
  result = _stat("1h").compute(_make_pos_corr_days_1h())
  rows = _tf(result, "1h").results
  assert len(rows) == 1


def test_result_row_condition_is_any_day() -> None:
  """The single result row has condition == 'any_day'."""
  result = _stat("1h").compute(_make_pos_corr_days_1h())
  r = _tf(result, "1h").results[0]
  assert r.condition == "any_day"


def test_result_row_outcome_is_correlation() -> None:
  """The single result row has outcome == 'correlation'."""
  result = _stat("1h").compute(_make_pos_corr_days_1h())
  r = _tf(result, "1h").results[0]
  assert r.outcome == "correlation"


def test_result_row_probability_is_zero() -> None:
  """probability == 0.0: this is a magnitude (correlation) stat, not a probability stat."""
  result = _stat("1h").compute(_make_pos_corr_days_1h())
  r = _tf(result, "1h").results[0]
  assert r.probability == pytest.approx(0.0)


def test_result_row_value_not_none() -> None:
  """The result row's value is not None (magnitude channel populated)."""
  result = _stat("1h").compute(_make_pos_corr_days_1h())
  r = _tf(result, "1h").results[0]
  assert r.value is not None


def test_result_row_count_equals_resolved_days() -> None:
  """count == total == number of resolved days (3 for 3-day dataset)."""
  result = _stat("1h").compute(_make_pos_corr_days_1h())
  r = _tf(result, "1h").results[0]
  # 3 resolved days -> both count and total = 3
  assert r.count == 3
  assert r.total == 3


def test_result_row_count_equals_total() -> None:
  """count and total are always equal (no distinction for this stat)."""
  result = _stat("1h").compute(_make_pos_corr_days_1h())
  r = _tf(result, "1h").results[0]
  assert r.count == r.total


def test_no_slices_in_result() -> None:
  """slices dict is empty (MarketOpenVolume declares slices=())."""
  result = _stat("1h").compute(_make_pos_corr_days_1h())
  assert _tf(result, "1h").slices == {}


# ===========================================================================
# 4. N<2 degenerate cases
# ===========================================================================

def test_single_resolved_day_value_is_zero() -> None:
  """Single resolved day -> N=1 < 2 -> r = 0.0 (degenerate guard)."""
  df = _make_rth_day("2024-01-08", open_vol_per_bar=10, rest_vol_per_bar=5, open_bar_min=60)
  result = _stat("1h").compute(df)
  r = _row(_tf(result, "1h").results, "any_day", "correlation")
  assert r.value == pytest.approx(0.0)


def test_single_resolved_day_count_is_one() -> None:
  """Single resolved day -> count == total == 1."""
  df = _make_rth_day("2024-01-08", open_vol_per_bar=10, rest_vol_per_bar=5, open_bar_min=60)
  result = _stat("1h").compute(df)
  r = _row(_tf(result, "1h").results, "any_day", "correlation")
  assert r.count == 1
  assert r.total == 1


def test_zero_days_value_is_zero() -> None:
  """No resolved days (only truncated) -> count=0, r = 0.0."""
  df = _make_truncated_day("2024-01-08")
  result = _stat("1h").compute(df)
  r = _row(_tf(result, "1h").results, "any_day", "correlation")
  assert r.value == pytest.approx(0.0)
  assert r.count == 0
  assert r.total == 0


# ===========================================================================
# 5. Baseline
# ===========================================================================

def test_result_row_has_non_none_value_baseline() -> None:
  """The result row carries a non-None value_baseline after compute()."""
  result = _stat("1h").compute(_make_pos_corr_days_1h())
  r = _tf(result, "1h").results[0]
  assert r.value_baseline is not None


def test_baseline_rows_returns_one_row() -> None:
  """baseline_rows returns exactly one (any_day, correlation) row."""
  stat = _stat("1h")
  day_table = stat.build_day_table(_make_pos_corr_days_1h())
  rows = stat.baseline_rows(day_table, seed=42)
  assert len(rows) == 1
  assert rows[0].condition == "any_day"
  assert rows[0].outcome == "correlation"


def test_baseline_r_within_valid_range() -> None:
  """|baseline r| <= 1.0 (Pearson r is always in [-1, 1])."""
  stat = _stat("1h")
  day_table = stat.build_day_table(_make_pos_corr_days_1h())
  rows = stat.baseline_rows(day_table, seed=42)
  assert rows[0].value is not None
  assert abs(rows[0].value) <= 1.0 + 1e-9


def test_baseline_same_seed_deterministic() -> None:
  """Two baseline_rows calls with the same seed produce identical output."""
  stat = _stat("1h")
  day_table = stat.build_day_table(_make_pos_corr_days_1h())
  rows_a = stat.baseline_rows(day_table, seed=7)
  rows_b = stat.baseline_rows(day_table, seed=7)
  assert len(rows_a) == len(rows_b)
  assert rows_a[0].condition == rows_b[0].condition
  assert rows_a[0].outcome == rows_b[0].outcome
  assert rows_a[0].value == pytest.approx(rows_b[0].value)  # type: ignore[arg-type]


def test_baseline_different_seeds_can_differ() -> None:
  """Different seeds typically produce different permutations -> different r.

  With 3 days there are 3! = 6 distinct permutations. Seeds 1 and 2 will
  almost certainly yield different permutations for this dataset.
  """
  stat = _stat("1h")
  day_table = stat.build_day_table(_make_pos_corr_days_1h())
  rows_s1 = stat.baseline_rows(day_table, seed=1)
  rows_s2 = stat.baseline_rows(day_table, seed=2)
  # With 6 possible permutations, same outcome is rare; we compare values.
  # Both seeds must still return valid float values:
  assert rows_s1[0].value is not None
  assert rows_s2[0].value is not None


def test_baseline_single_day_returns_zero() -> None:
  """baseline_rows with 1 resolved day -> N<2 degenerate path -> r=0.0."""
  stat = _stat("1h")
  df = _make_rth_day("2024-01-08", open_vol_per_bar=10, rest_vol_per_bar=5, open_bar_min=60)
  day_table = stat.build_day_table(df)
  rows = stat.baseline_rows(day_table, seed=42)
  assert rows[0].value == pytest.approx(0.0)


# ===========================================================================
# 6. Reproducibility
# ===========================================================================

def test_compute_twice_identical_model_dump() -> None:
  """compute() twice on the same input -> identical model_dump()."""
  df = _make_pos_corr_days_1h()
  stat = _stat("1h")
  assert stat.compute(df).model_dump() == stat.compute(df).model_dump()


def test_compute_twice_identical_row_values() -> None:
  """compute() twice with default seed returns identical row values."""
  df = _make_pos_corr_days_1h()
  stat = _stat("1h")
  result_a = stat.compute(df)
  result_b = stat.compute(df)
  rows_a = _tf(result_a, "1h").results
  rows_b = _tf(result_b, "1h").results
  assert len(rows_a) == len(rows_b)
  for a, b in zip(rows_a, rows_b):
    assert a.condition == b.condition
    assert a.outcome == b.outcome
    assert a.value == pytest.approx(b.value)  # type: ignore[arg-type]
    if a.value_baseline is None:
      assert b.value_baseline is None
    else:
      assert a.value_baseline == pytest.approx(b.value_baseline)


def test_baseline_rows_same_seed_reproducible_across_stat_instances() -> None:
  """Two separate stat instances with the same seed produce identical baselines."""
  df = _make_pos_corr_days_1h()
  stat_x = _stat("1h")
  stat_y = _stat("1h")
  dt_x = stat_x.build_day_table(df)
  dt_y = stat_y.build_day_table(df)
  rows_x = stat_x.baseline_rows(dt_x, seed=99)
  rows_y = stat_y.baseline_rows(dt_y, seed=99)
  assert rows_x[0].value == pytest.approx(rows_y[0].value)  # type: ignore[arg-type]


# ===========================================================================
# 7. Empty input
# ===========================================================================

def test_empty_input_total_samples_zero() -> None:
  """Empty candles -> total_samples == 0."""
  result = _stat("1h").compute(_empty_df())
  assert _tf(result, "1h").total_samples == 0


def test_empty_input_results_list() -> None:
  """Empty candles -> exactly one result row (correlation=0.0, N=0)."""
  result = _stat("1h").compute(_empty_df())
  rows = _tf(result, "1h").results
  # compute_rows always emits one row; with 0 valid rows it emits r=0, n=0.
  assert len(rows) == 1
  r = rows[0]
  assert r.condition == "any_day"
  assert r.outcome == "correlation"
  assert r.value == pytest.approx(0.0)
  assert r.count == 0
  assert r.total == 0


def test_empty_input_data_range_empty() -> None:
  """Empty candles -> data_range == []."""
  result = _stat("1h").compute(_empty_df())
  assert _tf(result, "1h").data_range == []


# ===========================================================================
# 8. Validation / round-trip
# ===========================================================================

def test_stat_name_is_market_open_volume() -> None:
  """stat_name == 'market_open_volume'."""
  result = _stat("1h").compute(_empty_df())
  assert result.stat_name == "market_open_volume"


def test_result_is_stat_run_result_instance() -> None:
  """compute() returns a StatRunResult instance."""
  result = _stat("1h").compute(_make_pos_corr_days_1h())
  assert isinstance(result, StatRunResult)


def test_i18n_title_non_empty_en_fr() -> None:
  """title carries non-empty en and fr strings."""
  result = _stat("1h").compute(_empty_df())
  assert result.title.en != ""
  assert result.title.fr != ""


def test_i18n_definition_non_empty_en_fr() -> None:
  """definition carries non-empty en and fr strings."""
  result = _stat("1h").compute(_empty_df())
  assert result.definition.en != ""
  assert result.definition.fr != ""


def test_i18n_condition_labels_have_en_fr() -> None:
  """Every condition label has non-empty en and fr strings."""
  result = _stat("1h").compute(_empty_df())
  for key, i18n in result.labels.conditions.items():
    assert i18n.en != "", f"condition {key!r}.en is empty"
    assert i18n.fr != "", f"condition {key!r}.fr is empty"


def test_i18n_outcome_labels_have_en_fr() -> None:
  """Every outcome label has non-empty en and fr strings."""
  result = _stat("1h").compute(_empty_df())
  for key, i18n in result.labels.outcomes.items():
    assert i18n.en != "", f"outcome {key!r}.en is empty"
    assert i18n.fr != "", f"outcome {key!r}.fr is empty"


def test_write_results_round_trip(tmp_path: Path) -> None:
  """write_results serialises to market_open_volume.json; re-validates correctly."""
  result = _stat("1h").compute(_make_pos_corr_days_1h())
  written = write_results(result, results_dir=tmp_path)

  assert written.name == "market_open_volume.json"
  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)

  tf = validated.instruments["NQ"]["1h"]
  assert tf.total_samples == 3

  r = _row(tf.results, "any_day", "correlation")
  # 3 perfectly correlated days -> r = 1.0 after round-trip
  assert r.value == pytest.approx(1.0)
  assert r.count == 3


def test_write_results_file_stem_matches_stat_name(tmp_path: Path) -> None:
  """The written JSON file stem equals stat_name 'market_open_volume'."""
  result = _stat("1h").compute(_make_pos_corr_days_1h())
  written = write_results(result, results_dir=tmp_path)
  raw = json.loads(written.read_text(encoding="utf-8"))
  assert written.stem == "market_open_volume"
  assert raw["stat_name"] == "market_open_volume"


# ===========================================================================
# 9. Timeframe handling
# ===========================================================================

def test_unknown_timeframe_raises_value_error() -> None:
  """Unknown timeframe raises ValueError containing 'Unknown timeframe'."""
  with pytest.raises(ValueError, match="Unknown timeframe"):
    MarketOpenVolume(instrument="NQ", config=_TEST_CONFIG, timeframe="2h")


def test_unknown_timeframe_error_message_names_bad_key() -> None:
  """The ValueError for an unknown timeframe includes the bad key in the message."""
  with pytest.raises(ValueError, match="badtf"):
    MarketOpenVolume(instrument="NQ", config=_TEST_CONFIG, timeframe="badtf")


def test_timeframe_15min_sets_open_bar_min() -> None:
  """'15min' timeframe sets open_bar_min == 15."""
  stat = MarketOpenVolume(instrument="NQ", config=_TEST_CONFIG, timeframe="15min")
  assert stat.open_bar_min == 15


def test_timeframe_30min_sets_open_bar_min() -> None:
  """'30min' timeframe sets open_bar_min == 30."""
  stat = MarketOpenVolume(instrument="NQ", config=_TEST_CONFIG, timeframe="30min")
  assert stat.open_bar_min == 30


def test_timeframe_1h_sets_open_bar_min() -> None:
  """'1h' timeframe sets open_bar_min == 60."""
  stat = MarketOpenVolume(instrument="NQ", config=_TEST_CONFIG, timeframe="1h")
  assert stat.open_bar_min == 60


def test_timeframe_30min_correct_open_volume() -> None:
  """'30min' timeframe: open_volume = vol_per_bar * 30 bars.

  Day A 30min: open_vol_per_bar=10 -> open_volume = 10*30 = 300
  rest_vol_per_bar=5 -> rest_volume = 5*375 = 1875
  """
  # open_volume_A = 10 * 30 = 300
  # rest_volume_A = 5  * 375 = 1875
  df = _make_rth_day("2024-01-08", open_vol_per_bar=10, rest_vol_per_bar=5, open_bar_min=30)
  dt = _stat("30min").build_day_table(df)
  date_a = pd.Timestamp("2024-01-08", tz=_NY).normalize()
  assert dt.loc[date_a, "open_volume"] == pytest.approx(10 * _N_OPEN_30)   # 300
  assert dt.loc[date_a, "rest_volume"] == pytest.approx(5  * _N_REST_30)   # 1875


def test_timeframe_30min_perfect_positive_correlation() -> None:
  """30min timeframe: three positively correlated days -> r = 1.0.

  Day A: open=10*30=300,  rest=5*375=1875
  Day B: open=20*30=600,  rest=10*375=3750
  Day C: open=30*30=900,  rest=15*375=5625
  rest_volume = 6.25 * open_volume for all days -> r = 1.0.
  """
  df = _concat_days([
    _make_rth_day("2024-01-08", open_vol_per_bar=10, rest_vol_per_bar=5,  open_bar_min=30),
    _make_rth_day("2024-01-09", open_vol_per_bar=20, rest_vol_per_bar=10, open_bar_min=30),
    _make_rth_day("2024-01-10", open_vol_per_bar=30, rest_vol_per_bar=15, open_bar_min=30),
  ])
  result = _stat("30min").compute(df)
  r = _row(_tf(result, "30min").results, "any_day", "correlation")
  assert r.value == pytest.approx(1.0)


# ===========================================================================
# 10. data_range reflects resolved days
# ===========================================================================

def test_data_range_three_resolved_days() -> None:
  """Three resolved days Mon-Wed -> data_range = ['2024-01-08', '2024-01-10']."""
  result = _stat("1h").compute(_make_pos_corr_days_1h())
  dr = _tf(result, "1h").data_range
  assert dr[0] == "2024-01-08"
  assert dr[1] == "2024-01-10"


def test_data_range_single_day_equals_that_date() -> None:
  """Single resolved day -> data_range = [date, date]."""
  df = _make_rth_day("2024-01-08", open_vol_per_bar=10, rest_vol_per_bar=5, open_bar_min=60)
  result = _stat("1h").compute(df)
  dr = _tf(result, "1h").data_range
  assert dr == ["2024-01-08", "2024-01-08"]


def test_data_range_excludes_truncated_day() -> None:
  """Truncated day does not affect data_range start/end."""
  df = _concat_days([
    _make_pos_corr_days_1h(),
    _make_truncated_day("2024-01-15"),
  ])
  result = _stat("1h").compute(df)
  dr = _tf(result, "1h").data_range
  # Truncated day on 2024-01-15 must not appear in data_range
  assert dr[0] == "2024-01-08"
  assert dr[1] == "2024-01-10"


def test_total_samples_three_resolved_days() -> None:
  """Three resolved sessions -> total_samples == 3."""
  result = _stat("1h").compute(_make_pos_corr_days_1h())
  assert _tf(result, "1h").total_samples == 3


def test_total_samples_excludes_truncated_day() -> None:
  """Appending a truncated day does not increase total_samples."""
  df = _concat_days([
    _make_pos_corr_days_1h(),
    _make_truncated_day("2024-01-15"),
  ])
  result = _stat("1h").compute(df)
  # Only 3 resolved days; truncated is excluded
  assert _tf(result, "1h").total_samples == 3


# ===========================================================================
# 11. Additional correlation math verification
# ===========================================================================

def test_correlation_value_is_float() -> None:
  """The computed correlation value is a Python float."""
  result = _stat("1h").compute(_make_pos_corr_days_1h())
  r = _tf(result, "1h").results[0]
  assert isinstance(r.value, float)


def test_positive_corr_value_is_positive() -> None:
  """Positively correlated data -> value > 0."""
  result = _stat("1h").compute(_make_pos_corr_days_1h())
  r = _tf(result, "1h").results[0]
  assert r.value is not None
  assert r.value > 0.0


def test_negative_corr_value_is_negative() -> None:
  """Negatively correlated data -> value < 0."""
  result = _stat("1h").compute(_make_neg_corr_days_1h())
  r = _tf(result, "1h").results[0]
  assert r.value is not None
  assert r.value < 0.0


def test_correlation_value_bounds() -> None:
  """|r| <= 1.0 for the positive correlation dataset."""
  result = _stat("1h").compute(_make_pos_corr_days_1h())
  r = _tf(result, "1h").results[0]
  assert r.value is not None
  assert -1.0 - 1e-9 <= r.value <= 1.0 + 1e-9


def test_baseline_value_baseline_is_float() -> None:
  """value_baseline is a float on the result row (not int or None)."""
  result = _stat("1h").compute(_make_pos_corr_days_1h())
  r = _tf(result, "1h").results[0]
  assert isinstance(r.value_baseline, float)


def test_baseline_r_does_not_equal_actual_r_for_pos_corr() -> None:
  """For perfectly correlated data, baseline r should not equal 1.0.

  Permuting rest_volume destroys the perfect correlation. With seed=42
  and 3 days, the permuted series is almost certainly not identical to
  the original ordering, so baseline r != 1.0.
  """
  result = _stat("1h").compute(_make_pos_corr_days_1h(), seed=42)
  r = _tf(result, "1h").results[0]
  assert r.value_baseline is not None
  # The baseline should not reproduce the perfect r=1.0
  assert r.value_baseline != pytest.approx(1.0)


def test_baseline_n_equals_count() -> None:
  """baseline_n == count == total (same N for both actual and baseline)."""
  result = _stat("1h").compute(_make_pos_corr_days_1h())
  r = _tf(result, "1h").results[0]
  # The baseline is computed from the same valid rows
  assert r.baseline_n == r.count


# ===========================================================================
# 12. classify_samples()
#
# Reuses the three-day positively-correlated dataset (1h timeframe):
#   Day A 2024-01-08: open_volume=600
#   Day B 2024-01-09: open_volume=1200
#   Day C 2024-01-10: open_volume=1800
# ===========================================================================

def test_classify_samples_exact_rows() -> None:
  """classify_samples() emits exactly the 3 expected SampleRows."""
  stat = _stat("1h")
  dt = stat.build_day_table(_make_pos_corr_days_1h())
  samples = stat.classify_samples(dt)

  expected = [
    SampleRow(date="2024-01-08", condition="any_day", outcome="correlation", value=float(_OV_A_1H)),
    SampleRow(date="2024-01-09", condition="any_day", outcome="correlation", value=float(_OV_B_1H)),
    SampleRow(date="2024-01-10", condition="any_day", outcome="correlation", value=float(_OV_C_1H)),
  ]

  assert len(samples) == len(expected)
  for got, want in zip(samples, expected):
    assert got.date == want.date
    assert got.condition == want.condition
    assert got.outcome == want.outcome
    assert got.value == pytest.approx(want.value)


def test_classify_samples_matches_compute_rows_count_and_values() -> None:
  """Sample count equals the row's N; sample values reproduce open_volume."""
  stat = _stat("1h")
  dt = stat.build_day_table(_make_pos_corr_days_1h())
  samples = stat.classify_samples(dt)
  rows = stat.compute_rows(dt)
  row = _row(rows, "any_day", "correlation")

  assert len(samples) == row.total
  valid = dt[["open_volume", "rest_volume"]].dropna()
  got_values = sorted(s.value for s in samples)
  want_values = sorted(float(v) for v in valid["open_volume"])
  assert got_values == pytest.approx(want_values)


def test_classify_samples_empty_day_table() -> None:
  """Empty day_table -> []."""
  stat = _stat("1h")
  dt = stat.build_day_table(_empty_df())
  assert stat.classify_samples(dt) == []


def test_classify_samples_excludes_row_with_nan_column() -> None:
  """A day with a NaN in either volume column is excluded (non-countable).

  Mirrors ``compute_rows``' ``valid = day_table[[...]].dropna()`` filter:
  only days with both ``open_volume`` and ``rest_volume`` non-NaN are
  countable and thus classified.
  """
  stat = _stat("1h")
  dt = pd.DataFrame(
    {
      "open_volume": [600.0, 1200.0, float("nan")],
      "rest_volume": [1725.0, float("nan"), 5175.0],
    },
    index=pd.DatetimeIndex(["2024-01-08", "2024-01-09", "2024-01-10"]),
  )
  samples = stat.classify_samples(dt)
  assert len(samples) == 1
  assert samples[0].date == "2024-01-08"
  assert samples[0].condition == "any_day"
  assert samples[0].outcome == "correlation"
  assert samples[0].value == pytest.approx(600.0)
