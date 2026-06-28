"""Tests for stats.candle_body_ratio.standard.

All data is synthetic — no real market files required.
Expected values are hand-calculated before each assertion.

Bucket arithmetic reference (RTH start = 09:30 = 570 min, 15min buckets):
  n_buckets = ceil(405 / 15) = 27
  bucket_idx = (mod - 570) // 15
  bucket 0  = "0930"  mod 570
  bucket 1  = "0945"  mod 585
  bucket 26 = "1600"  mod 960  (partial last bucket)

Resolution criterion (build_resolved_days):
  - Bar at exactly mod=570 (clean session open).
  - Last RTH bar mod >= 975 - 15 = 960.

Builder note: each bucket gets exactly 1 bar at its start_mod.
With 1 bar: bucket candle == that bar's own OHLC.
  body  = |close - open|
  range = high - low
  ratio = body / range  (0.0 when range == 0)
  large body when ratio >= candle_size / 100
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.candle_body_ratio.standard import CandleBodyRatio
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
  timeframes=["15min"],
  parquet_path=Path("data/NQ_1min.parquet"),
)

_RTH_START = 570   # 09:30
_RTH_END = 975     # 16:15 (exclusive)
_N_BUCKETS_15 = 27  # ceil(405/15)


# ---------------------------------------------------------------------------
# Synthetic day builders
# ---------------------------------------------------------------------------

def _make_body_day(
  date: str,
  bucket_specs: dict[int, dict],
  n_buckets: int = _N_BUCKETS_15,
  bucket_min: int = 15,
) -> pd.DataFrame:
  """Build one fully resolved RTH day with controlled per-bucket OHLCV.

  Each bucket gets exactly 1 bar placed at its start_mod. Since there is only
  one bar per bucket:
    bucket open  = bar.open
    bucket close = bar.close
    bucket high  = bar.high
    bucket low   = bar.low
    body  = |close - open|
    range = high - low
    ratio = body / range  (0.0 when range == 0)

  The caller supplies bucket_idx → spec where spec has:
    open  : bar open price
    close : bar close price
    high  : bar high (caller must ensure >= max(open, close))
    low   : bar low  (caller must ensure <= min(open, close))

  Buckets not listed in bucket_specs use a flat neutral bar:
    open=close=high=low=100.0 → ratio=0.0 → not large at any positive threshold.

  With bucket_min=15 and n_buckets=27 (the default):
    - Bar at mod=570 satisfies the session-open requirement.
    - Bar at bucket 26 (mod=960) satisfies close tolerance (960 >= 960).
  """
  base = pd.Timestamp(date, tz=_NY)
  records = []
  for bucket_idx in range(n_buckets):
    start_mod = _RTH_START + bucket_idx * bucket_min
    h, m = divmod(start_mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    spec = bucket_specs.get(bucket_idx)
    if spec is not None:
      o = float(spec["open"])
      c = float(spec["close"])
      hi = max(float(spec["high"]), o, c)
      lo = min(float(spec["low"]), o, c)
    else:
      o = c = hi = lo = 100.0
    records.append({
      "timestamp": ts,
      "open": o,
      "high": hi,
      "low": lo,
      "close": c,
      "volume": 100,
    })
  return pd.DataFrame(records)


def _make_truncated_day(date: str, last_mod: int = 590) -> pd.DataFrame:
  """A day whose last RTH bar is at last_mod < 960 — must be excluded."""
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
      "volume": 100,
    })
  return pd.DataFrame(records)


def _concat_days(frames: list[pd.DataFrame]) -> pd.DataFrame:
  df = pd.concat(frames, ignore_index=True)
  return df.sort_values("timestamp").reset_index(drop=True)


def _empty_df() -> pd.DataFrame:
  df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
  df["timestamp"] = pd.array([], dtype="datetime64[ns, America/New_York]")
  return df


def _stat(timeframe: str = "15min", candle_size: float = 50.0) -> CandleBodyRatio:
  return CandleBodyRatio(
    instrument="NQ",
    config=_TEST_CONFIG,
    timeframe=timeframe,
    candle_size=candle_size,
  )


def _tf(result: StatRunResult, timeframe: str = "15min"):  # type: ignore[return]
  return result.instruments["NQ"][timeframe]


def _row(rows: list[StatResultRow], condition: str, outcome: str) -> StatResultRow:
  for r in rows:
    if r.condition == condition and r.outcome == outcome:
      return r
  raise KeyError(f"({condition!r}, {outcome!r}) not found in rows")


# ---------------------------------------------------------------------------
# Main synthetic dataset: 3 resolved days + 1 truncated day
#
# Each day: 1 bar per 15min bucket (mods 570, 585, ..., 960).
# Bucket body:  |close - open|
# Bucket range: high - low
# Bucket ratio: body / range
# Large body when ratio >= 0.50 (default candle_size=50).
#
# Day A  2024-01-08 Mon
#   bucket 0 "0930": open=100, close=108, high=110, low=100
#     body=8, range=10, ratio=0.80 → LARGE
#   bucket 1 "0945": open=100, close=102, high=110, low=100
#     body=2, range=10, ratio=0.20 → not large
#   buckets 2-26: neutral (ratio=0.0) → not large
#
# Day B  2024-01-09 Tue
#   bucket 0 "0930": open=100, close=103, high=110, low=100
#     body=3, range=10, ratio=0.30 → not large
#   bucket 1 "0945": open=100, close=106, high=110, low=100
#     body=6, range=10, ratio=0.60 → LARGE
#   buckets 2-26: neutral → not large
#
# Day C  2024-01-10 Wed
#   bucket 0 "0930": open=100, close=109, high=110, low=100
#     body=9, range=10, ratio=0.90 → LARGE
#   bucket 1 "0945": open=100, close=101, high=110, low=100
#     body=1, range=10, ratio=0.10 → not large
#   buckets 2-26: neutral → not large
#
# Day D  2024-01-15 Mon — TRUNCATED (last bar mod=590 < 960) — excluded
#
# Expected at candle_size=50 (threshold=0.50):
#   "0930": large on A, C → count=2, total=3, probability=2/3
#   "0945": large on B   → count=1, total=3, probability=1/3
#   buckets 2-26:         count=0, total=3, probability=0.0
# ---------------------------------------------------------------------------

_SPEC_A = {
  0: {"open": 100.0, "close": 108.0, "high": 110.0, "low": 100.0},  # ratio=0.80 LARGE
  1: {"open": 100.0, "close": 102.0, "high": 110.0, "low": 100.0},  # ratio=0.20
}
_SPEC_B = {
  0: {"open": 100.0, "close": 103.0, "high": 110.0, "low": 100.0},  # ratio=0.30
  1: {"open": 100.0, "close": 106.0, "high": 110.0, "low": 100.0},  # ratio=0.60 LARGE
}
_SPEC_C = {
  0: {"open": 100.0, "close": 109.0, "high": 110.0, "low": 100.0},  # ratio=0.90 LARGE
  1: {"open": 100.0, "close": 101.0, "high": 110.0, "low": 100.0},  # ratio=0.10
}


def _make_three_days() -> pd.DataFrame:
  return _concat_days([
    _make_body_day("2024-01-08", _SPEC_A),
    _make_body_day("2024-01-09", _SPEC_B),
    _make_body_day("2024-01-10", _SPEC_C),
  ])


def _make_three_days_with_truncated() -> pd.DataFrame:
  return _concat_days([
    _make_body_day("2024-01-08", _SPEC_A),
    _make_body_day("2024-01-09", _SPEC_B),
    _make_body_day("2024-01-10", _SPEC_C),
    _make_truncated_day("2024-01-15"),
  ])


# ===========================================================================
# 1. Known per-bucket probabilities at default 50% threshold
# ===========================================================================

def test_bucket_0930_count_and_probability() -> None:
  """Bucket "0930" large on days A and C → count=2, total=3, probability=2/3."""
  result = _stat().compute(_make_three_days())
  # A: ratio=0.80 (large), B: ratio=0.30 (not large), C: ratio=0.90 (large)
  r = _row(_tf(result).results, "0930", "large_body")
  assert r.count == 2
  assert r.total == 3
  assert r.probability == pytest.approx(2 / 3)


def test_bucket_0945_count_and_probability() -> None:
  """Bucket "0945" large on day B only → count=1, total=3, probability=1/3."""
  result = _stat().compute(_make_three_days())
  # A: ratio=0.20, B: ratio=0.60 (large), C: ratio=0.10
  r = _row(_tf(result).results, "0945", "large_body")
  assert r.count == 1
  assert r.total == 3
  assert r.probability == pytest.approx(1 / 3)


def test_neutral_buckets_count_zero() -> None:
  """Neutral buckets (indices 2-26) have count=0, total=3, probability=0.0."""
  result = _stat().compute(_make_three_days())
  # Build the set of neutral bucket keys (mods 600, 615, ..., 960)
  neutral_keys = set()
  for i in range(2, _N_BUCKETS_15):
    start = _RTH_START + i * 15
    hh, mm = divmod(start, 60)
    neutral_keys.add(f"{hh:02d}{mm:02d}")

  for r in _tf(result).results:
    if r.condition in neutral_keys:
      # All neutral bars are flat (ratio=0.0 < 0.50) → never large
      assert r.count == 0, f"bucket {r.condition}: count={r.count}"
      assert r.total == 3, f"bucket {r.condition}: total={r.total}"
      assert r.probability == pytest.approx(0.0)


def test_outcome_key_is_large_body() -> None:
  """Every result row carries outcome='large_body' (single outcome key)."""
  result = _stat().compute(_make_three_days())
  for r in _tf(result).results:
    assert r.outcome == "large_body"


def test_each_bucket_yields_exactly_one_row() -> None:
  """Each present bucket produces exactly 1 row (single outcome)."""
  from collections import Counter
  result = _stat().compute(_make_three_days())
  counts = Counter(r.condition for r in _tf(result).results)
  for key, cnt in counts.items():
    assert cnt == 1, f"bucket {key!r} has {cnt} rows, expected 1"


def test_27_rows_total() -> None:
  """With 1 bar in every bucket on every day, all 27 buckets appear → 27 rows."""
  result = _stat().compute(_make_three_days())
  # 27 buckets × 1 outcome = 27 rows
  assert len(_tf(result).results) == _N_BUCKETS_15


def test_value_is_none_on_all_rows() -> None:
  """Probability stat: value == None on every result row."""
  result = _stat().compute(_make_three_days())
  for r in _tf(result).results:
    assert r.value is None, f"({r.condition}): value={r.value!r}, expected None"


def test_value_baseline_is_none_on_all_rows() -> None:
  """Probability stat: value_baseline == None on every result row."""
  result = _stat().compute(_make_three_days())
  for r in _tf(result).results:
    assert r.value_baseline is None, (
      f"({r.condition}): value_baseline={r.value_baseline!r}, expected None"
    )


def test_probability_equals_count_over_total() -> None:
  """probability == count / total for every result row."""
  result = _stat().compute(_make_three_days())
  for r in _tf(result).results:
    expected = r.count / r.total if r.total > 0 else 0.0
    assert r.probability == pytest.approx(expected), (
      f"({r.condition}): probability={r.probability}, count/total={expected}"
    )


# ===========================================================================
# 2. candle_size threshold matters
# ===========================================================================

def test_candle_size_30_makes_ratio_05_large() -> None:
  """Ratio=0.5 is large at candle_size=30 (threshold=0.30): 0.5 >= 0.30 → LARGE.

  Bucket 0: open=100, close=105, high=110, low=100
    body=5, range=10, ratio=0.5
  """
  spec = {0: {"open": 100.0, "close": 105.0, "high": 110.0, "low": 100.0}}
  df = _make_body_day("2024-01-08", spec)
  result = _stat(candle_size=30.0).compute(df)
  r = _row(_tf(result).results, "0930", "large_body")
  # 0.5 >= 0.30 → large
  assert r.count == 1
  assert r.total == 1
  assert r.probability == pytest.approx(1.0)


def test_candle_size_70_makes_ratio_05_not_large() -> None:
  """Ratio=0.5 is NOT large at candle_size=70 (threshold=0.70): 0.5 < 0.70.

  Same bucket spec: open=100, close=105, high=110, low=100 → ratio=0.5
  """
  spec = {0: {"open": 100.0, "close": 105.0, "high": 110.0, "low": 100.0}}
  df = _make_body_day("2024-01-08", spec)
  result = _stat(candle_size=70.0).compute(df)
  r = _row(_tf(result).results, "0930", "large_body")
  # 0.5 < 0.70 → not large
  assert r.count == 0
  assert r.total == 1
  assert r.probability == pytest.approx(0.0)


def test_threshold_boundary_ratio_equals_threshold_is_large() -> None:
  """ratio == threshold counts as large (condition is ratio >= threshold).

  open=100, close=105, high=110, low=100 → ratio=0.5
  candle_size=50 → threshold=0.50 → 0.5 >= 0.50 → LARGE.
  """
  spec = {0: {"open": 100.0, "close": 105.0, "high": 110.0, "low": 100.0}}
  df = _make_body_day("2024-01-08", spec)
  result = _stat(candle_size=50.0).compute(df)
  r = _row(_tf(result).results, "0930", "large_body")
  assert r.count == 1
  assert r.probability == pytest.approx(1.0)


def test_candle_size_zero_every_bucket_large() -> None:
  """candle_size=0.0: ratio=0.0 >= 0.0 → every bucket (including flat) is large."""
  df = _make_body_day("2024-01-08", {})  # all neutral → ratio=0.0
  result = _stat(candle_size=0.0).compute(df)
  for r in _tf(result).results:
    # 0.0 >= 0.0 → large
    assert r.probability == pytest.approx(1.0), (
      f"bucket {r.condition}: P={r.probability}, expected 1.0 at candle_size=0"
    )


def test_candle_size_100_only_full_body_large() -> None:
  """candle_size=100: only ratio=1.0 qualifies.

  Bucket 0: open=100, close=110, high=110, low=100 → body=10, range=10, ratio=1.0 → LARGE.
  Neutral buckets: ratio=0.0 < 1.0 → not large.
  """
  spec = {0: {"open": 100.0, "close": 110.0, "high": 110.0, "low": 100.0}}
  df = _make_body_day("2024-01-08", spec)
  result = _stat(candle_size=100.0).compute(df)
  r_large = _row(_tf(result).results, "0930", "large_body")
  # ratio=1.0 >= 1.0 → large
  assert r_large.count == 1
  assert r_large.probability == pytest.approx(1.0)

  r_neutral = _row(_tf(result).results, "0945", "large_body")
  # ratio=0.0 < 1.0 → not large
  assert r_neutral.count == 0
  assert r_neutral.probability == pytest.approx(0.0)


def test_threshold_crossover_across_three_days() -> None:
  """Same day-table, different candle_size changes which days are 'large'.

  Dataset: 3 days, all in bucket 0 only.
    Day A: ratio=0.80 → large at 50% and 70%, not large at 90%
    Day B: ratio=0.30 → large at 0%, not large at 50% and 70%
    Day C: ratio=0.90 → large at 50%, 70%, and 90%

  At candle_size=70 (threshold=0.70):
    large days: A (0.80>=0.70), C (0.90>=0.70) → count=2, probability=2/3
  At candle_size=90 (threshold=0.90):
    large days: C only (0.90>=0.90) → count=1, probability=1/3
  """
  result_70 = _stat(candle_size=70.0).compute(_make_three_days())
  r_70 = _row(_tf(result_70).results, "0930", "large_body")
  # A: 0.80>=0.70 large, B: 0.30<0.70 not, C: 0.90>=0.70 large → count=2
  assert r_70.count == 2
  assert r_70.probability == pytest.approx(2 / 3)

  result_90 = _stat(candle_size=90.0).compute(_make_three_days())
  r_90 = _row(_tf(result_90).results, "0930", "large_body")
  # A: 0.80<0.90 not, B: 0.30<0.90 not, C: 0.90>=0.90 large → count=1
  assert r_90.count == 1
  assert r_90.probability == pytest.approx(1 / 3)


# ===========================================================================
# 3. Zero-range (flat) candle → ratio=0.0 → not large at positive threshold
# ===========================================================================

def test_zero_range_candle_not_large_at_positive_threshold() -> None:
  """Flat candle (open=close=high=low=100): range=0 → ratio=0.0 → not large at 50%."""
  spec = {0: {"open": 100.0, "close": 100.0, "high": 100.0, "low": 100.0}}
  df = _make_body_day("2024-01-08", spec)
  result = _stat(candle_size=50.0).compute(df)
  r = _row(_tf(result).results, "0930", "large_body")
  # range=0 → ratio=0.0 < 0.5 → not large
  assert r.count == 0
  assert r.probability == pytest.approx(0.0)


def test_zero_range_candle_still_counted_in_total() -> None:
  """Flat candle: bucket IS present (has a bar) → total=1, not excluded."""
  spec = {0: {"open": 100.0, "close": 100.0, "high": 100.0, "low": 100.0}}
  df = _make_body_day("2024-01-08", spec)
  result = _stat(candle_size=50.0).compute(df)
  r = _row(_tf(result).results, "0930", "large_body")
  # bucket is present → counted in total
  assert r.total == 1


def test_zero_range_candle_large_at_candle_size_zero() -> None:
  """Flat candle (ratio=0.0) IS large at candle_size=0 (threshold=0.0)."""
  spec = {0: {"open": 100.0, "close": 100.0, "high": 100.0, "low": 100.0}}
  df = _make_body_day("2024-01-08", spec)
  result = _stat(candle_size=0.0).compute(df)
  r = _row(_tf(result).results, "0930", "large_body")
  # 0.0 >= 0.0 → large
  assert r.count == 1
  assert r.probability == pytest.approx(1.0)


# ===========================================================================
# 4. Partial last bucket "1600" present in results
# ===========================================================================

def test_partial_last_bucket_1600_is_present() -> None:
  """The 16:00 bucket ('1600') appears in result rows."""
  result = _stat().compute(_make_three_days())
  keys = {r.condition for r in _tf(result).results}
  assert "1600" in keys


def test_partial_last_bucket_1600_total_is_three() -> None:
  """The '1600' bucket has a bar on all 3 resolved days → total=3."""
  result = _stat().compute(_make_three_days())
  r = _row(_tf(result).results, "1600", "large_body")
  # All 3 days have a neutral bar at mod=960 (bucket 26) → total=3, count=0
  assert r.total == 3
  assert r.count == 0


def test_partial_last_bucket_1600_probability_zero() -> None:
  """The '1600' bucket uses neutral bars (ratio=0.0) → probability=0.0."""
  result = _stat().compute(_make_three_days())
  r = _row(_tf(result).results, "1600", "large_body")
  assert r.probability == pytest.approx(0.0)


def test_partial_last_bucket_1600_large_when_spec_set() -> None:
  """When bucket 26 has a large body, '1600' probability reflects it.

  Bucket 26 (mod=960): open=100, close=110, high=110, low=100
    body=10, range=10, ratio=1.0 → LARGE at 50%.
  """
  spec = {26: {"open": 100.0, "close": 110.0, "high": 110.0, "low": 100.0}}
  df = _make_body_day("2024-01-08", spec)
  result = _stat().compute(df)
  r = _row(_tf(result).results, "1600", "large_body")
  assert r.count == 1
  assert r.probability == pytest.approx(1.0)


# ===========================================================================
# 5. Pending/truncated day excluded
# ===========================================================================

def test_truncated_day_excluded_from_total_samples() -> None:
  """Truncated day (last bar mod=590 < 960) does not count in total_samples."""
  result = _stat().compute(_make_three_days_with_truncated())
  # 3 resolved + 1 truncated: only 3 pass the close-tolerance filter
  assert _tf(result).total_samples == 3


def test_truncated_day_absent_from_day_table() -> None:
  """build_day_table excludes the truncated 2024-01-15 from its index."""
  df = _make_three_days_with_truncated()
  dt = _stat().build_day_table(df)
  excluded = pd.Timestamp("2024-01-15", tz=_NY).normalize()
  assert excluded not in dt.index
  assert len(dt) == 3


def test_truncated_day_does_not_affect_bucket_probabilities() -> None:
  """Appending a truncated day leaves all bucket probabilities unchanged."""
  result_clean = _stat().compute(_make_three_days())
  result_trunc = _stat().compute(_make_three_days_with_truncated())
  r_clean = _row(_tf(result_clean).results, "0930", "large_body")
  r_trunc = _row(_tf(result_trunc).results, "0930", "large_body")
  assert r_trunc.probability == pytest.approx(r_clean.probability)
  assert r_trunc.count == r_clean.count == 2
  assert r_trunc.total == r_clean.total == 3


def test_truncated_day_excluded_from_bucket_totals() -> None:
  """No bucket's total exceeds 3 when a truncated day is appended."""
  result = _stat().compute(_make_three_days_with_truncated())
  for r in _tf(result).results:
    assert r.total <= 3, (
      f"bucket {r.condition}: total={r.total} > 3, truncated day leaked in"
    )


# ===========================================================================
# 6. Weekday slice
# ===========================================================================

def test_weekday_slice_groups_present() -> None:
  """Mon/Tue/Wed 3-day dataset → weekday slicer has exactly 3 groups."""
  result = _stat().compute(_make_three_days())
  groups = _tf(result).slices["weekday"].groups
  assert set(groups.keys()) == {"monday", "tuesday", "wednesday"}


def test_weekday_each_group_total_samples() -> None:
  """Each weekday group has total_samples=1 (one day per weekday)."""
  result = _stat().compute(_make_three_days())
  for key, grp in _tf(result).slices["weekday"].groups.items():
    assert grp.total_samples == 1, f"{key}: total_samples={grp.total_samples}"


def test_weekday_monday_bucket_0930() -> None:
  """Monday = Day A: bucket '0930' ratio=0.80 → count=1, total=1, P=1.0."""
  result = _stat().compute(_make_three_days())
  mon = _tf(result).slices["weekday"].groups["monday"].results
  r = _row(mon, "0930", "large_body")
  # Day A bucket 0: body=8, range=10, ratio=0.80 >= 0.50 → large
  assert r.count == 1
  assert r.total == 1
  assert r.probability == pytest.approx(1.0)


def test_weekday_monday_bucket_0945() -> None:
  """Monday = Day A: bucket '0945' ratio=0.20 → count=0, total=1, P=0.0."""
  result = _stat().compute(_make_three_days())
  mon = _tf(result).slices["weekday"].groups["monday"].results
  r = _row(mon, "0945", "large_body")
  # Day A bucket 1: body=2, range=10, ratio=0.20 < 0.50 → not large
  assert r.count == 0
  assert r.total == 1
  assert r.probability == pytest.approx(0.0)


def test_weekday_tuesday_bucket_0930() -> None:
  """Tuesday = Day B: bucket '0930' ratio=0.30 → count=0, P=0.0."""
  result = _stat().compute(_make_three_days())
  tue = _tf(result).slices["weekday"].groups["tuesday"].results
  r = _row(tue, "0930", "large_body")
  # Day B bucket 0: body=3, range=10, ratio=0.30 < 0.50 → not large
  assert r.count == 0
  assert r.probability == pytest.approx(0.0)


def test_weekday_tuesday_bucket_0945() -> None:
  """Tuesday = Day B: bucket '0945' ratio=0.60 → count=1, P=1.0."""
  result = _stat().compute(_make_three_days())
  tue = _tf(result).slices["weekday"].groups["tuesday"].results
  r = _row(tue, "0945", "large_body")
  # Day B bucket 1: body=6, range=10, ratio=0.60 >= 0.50 → large
  assert r.count == 1
  assert r.probability == pytest.approx(1.0)


def test_weekday_wednesday_bucket_0930() -> None:
  """Wednesday = Day C: bucket '0930' ratio=0.90 → count=1, P=1.0."""
  result = _stat().compute(_make_three_days())
  wed = _tf(result).slices["weekday"].groups["wednesday"].results
  r = _row(wed, "0930", "large_body")
  # Day C bucket 0: body=9, range=10, ratio=0.90 >= 0.50 → large
  assert r.count == 1
  assert r.probability == pytest.approx(1.0)


def test_weekday_wednesday_bucket_0945() -> None:
  """Wednesday = Day C: bucket '0945' ratio=0.10 → count=0, P=0.0."""
  result = _stat().compute(_make_three_days())
  wed = _tf(result).slices["weekday"].groups["wednesday"].results
  r = _row(wed, "0945", "large_body")
  # Day C bucket 1: body=1, range=10, ratio=0.10 < 0.50 → not large
  assert r.count == 0
  assert r.probability == pytest.approx(0.0)


def test_weekday_slice_value_channels_are_none() -> None:
  """Weekday slice rows: probability stat → value=None, value_baseline=None."""
  result = _stat().compute(_make_three_days())
  for key, grp in _tf(result).slices["weekday"].groups.items():
    for r in grp.results:
      assert r.value is None, (
        f"weekday={key} ({r.condition}): value={r.value!r}"
      )
      assert r.value_baseline is None, (
        f"weekday={key} ({r.condition}): value_baseline={r.value_baseline!r}"
      )


def test_weekday_per_group_probabilities_cross_check() -> None:
  """Cross-check all 3 weekday groups for both controlled buckets.

  Mon (Day A): 0930=LARGE, 0945=not → P(0930)=1.0, P(0945)=0.0
  Tue (Day B): 0930=not, 0945=LARGE → P(0930)=0.0, P(0945)=1.0
  Wed (Day C): 0930=LARGE, 0945=not → P(0930)=1.0, P(0945)=0.0
  """
  result = _stat().compute(_make_three_days())
  grps = _tf(result).slices["weekday"].groups

  assert _row(grps["monday"].results, "0930", "large_body").probability == pytest.approx(1.0)
  assert _row(grps["monday"].results, "0945", "large_body").probability == pytest.approx(0.0)
  assert _row(grps["tuesday"].results, "0930", "large_body").probability == pytest.approx(0.0)
  assert _row(grps["tuesday"].results, "0945", "large_body").probability == pytest.approx(1.0)
  assert _row(grps["wednesday"].results, "0930", "large_body").probability == pytest.approx(1.0)
  assert _row(grps["wednesday"].results, "0945", "large_body").probability == pytest.approx(0.0)


# ===========================================================================
# 7. Baseline: present on every row, deterministic, seed-dependent
# ===========================================================================

def test_baseline_present_on_every_overall_row() -> None:
  """Every overall result row has baseline_n > 0 and baseline_prob in [0, 1]."""
  result = _stat().compute(_make_three_days())
  for r in _tf(result).results:
    assert r.baseline_n > 0, f"({r.condition}): baseline_n=0"
    assert 0.0 <= r.baseline_prob <= 1.0, (
      f"({r.condition}): baseline_prob={r.baseline_prob} out of [0,1]"
    )


def test_baseline_rows_same_seed_deterministic() -> None:
  """Two baseline_rows calls with the same seed produce identical probabilities."""
  stat = _stat()
  dt = stat.build_day_table(_make_three_days())
  rows_a = stat.baseline_rows(dt, seed=7)
  rows_b = stat.baseline_rows(dt, seed=7)
  assert len(rows_a) == len(rows_b)
  for a, b in zip(rows_a, rows_b):
    assert a.condition == b.condition
    assert a.outcome == b.outcome
    assert a.probability == pytest.approx(b.probability)
    assert a.total == b.total


def test_baseline_different_seeds_can_differ() -> None:
  """Different seeds yield different baseline probabilities (with high probability).

  Pool: 27 buckets × 3 days = 81 flags, drawing n=3 without replacement.
  The pool has 3 ones and 78 zeros; different seeds produce different samples.
  """
  stat = _stat()
  dt = stat.build_day_table(_make_three_days())
  rows_1 = stat.baseline_rows(dt, seed=1)
  rows_99 = stat.baseline_rows(dt, seed=99)
  any_diff = any(
    abs(a.probability - b.probability) > 1e-9
    for a, b in zip(rows_1, rows_99)
    if a.condition == b.condition
  )
  assert any_diff, "All baseline probabilities identical for seed=1 vs seed=99"


def test_baseline_n_equals_contributing_days() -> None:
  """baseline_n == 3 for each bucket (pool draw of size n=3 per bucket).

  3 resolved full-session days → each bucket has n=3 contributing days.
  pool has 81 values; sample_size = min(3, 81) = 3 → baseline_n=3.
  """
  result = _stat().compute(_make_three_days())
  for r in _tf(result).results:
    assert r.baseline_n == 3, f"({r.condition}): baseline_n={r.baseline_n}"


def test_baseline_outcome_key_is_large_body() -> None:
  """baseline_rows rows carry outcome='large_body'."""
  stat = _stat()
  dt = stat.build_day_table(_make_three_days())
  for r in stat.baseline_rows(dt, seed=42):
    assert r.outcome == "large_body"


def test_baseline_prob_is_float() -> None:
  """baseline_prob is a float on every result row (not int, not None)."""
  result = _stat().compute(_make_three_days())
  for r in _tf(result).results:
    assert isinstance(r.baseline_prob, float), (
      f"({r.condition}): baseline_prob type={type(r.baseline_prob)}"
    )


# ===========================================================================
# 8. Reproducibility
# ===========================================================================

def test_compute_twice_identical_model_dump() -> None:
  """compute() twice on the same input → identical model_dump()."""
  df = _make_three_days()
  stat = _stat()
  assert stat.compute(df).model_dump() == stat.compute(df).model_dump()


def test_compute_twice_identical_probabilities() -> None:
  """compute() with the same seed returns identical probabilities both calls."""
  df = _make_three_days()
  stat = _stat()
  result_a = stat.compute(df)
  result_b = stat.compute(df)
  rows_a = _tf(result_a).results
  rows_b = _tf(result_b).results
  assert len(rows_a) == len(rows_b)
  for a, b in zip(rows_a, rows_b):
    assert a.condition == b.condition
    assert a.probability == pytest.approx(b.probability)
    assert a.baseline_prob == pytest.approx(b.baseline_prob)
    assert a.baseline_n == b.baseline_n


# ===========================================================================
# 9. Empty input
# ===========================================================================

def test_empty_dataframe_total_samples_zero() -> None:
  """Empty candles → total_samples == 0."""
  assert _tf(_stat().compute(_empty_df())).total_samples == 0


def test_empty_dataframe_results_empty() -> None:
  """Empty candles → results == []."""
  assert _tf(_stat().compute(_empty_df())).results == []


def test_empty_dataframe_data_range_empty() -> None:
  """Empty candles → data_range == []."""
  assert _tf(_stat().compute(_empty_df())).data_range == []


def test_empty_dataframe_no_weekday_groups() -> None:
  """Empty candles → weekday slicer produces no groups."""
  assert _tf(_stat().compute(_empty_df())).slices["weekday"].groups == {}


def test_empty_build_day_table_returns_zero_rows() -> None:
  """build_day_table on empty input returns a DataFrame with 0 rows."""
  dt = _stat().build_day_table(_empty_df())
  assert len(dt) == 0


# ===========================================================================
# 10. write_results round-trip → file named candle_body_ratio.json
# ===========================================================================

def test_write_results_file_named_correctly(tmp_path: Path) -> None:
  """write_results produces 'candle_body_ratio.json' in the results dir."""
  result = _stat().compute(_make_three_days())
  written = write_results(result, results_dir=tmp_path)
  assert written.name == "candle_body_ratio.json"


def test_write_results_round_trip_probabilities(tmp_path: Path) -> None:
  """write_results → JSON → StatRunResult: probabilities survive round-trip."""
  result = _stat().compute(_make_three_days())
  written = write_results(result, results_dir=tmp_path)
  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)
  tf = validated.instruments["NQ"]["15min"]
  assert tf.total_samples == 3

  # "0930": count=2, total=3, probability=2/3
  r = _row(tf.results, "0930", "large_body")
  assert r.count == 2
  assert r.total == 3
  assert r.probability == pytest.approx(2 / 3)
  assert r.value is None
  assert r.value_baseline is None


def test_write_results_stat_name_in_file(tmp_path: Path) -> None:
  """Serialised JSON carries stat_name == 'candle_body_ratio'."""
  result = _stat().compute(_make_three_days())
  written = write_results(result, results_dir=tmp_path)
  raw = json.loads(written.read_text(encoding="utf-8"))
  assert raw["stat_name"] == "candle_body_ratio"


def test_write_results_weekday_slice_survives_round_trip(tmp_path: Path) -> None:
  """Weekday slice data survives JSON serialisation and Pydantic validation."""
  result = _stat().compute(_make_three_days())
  validated = StatRunResult.model_validate(
    json.loads(write_results(result, results_dir=tmp_path).read_text(encoding="utf-8"))
  )
  tf = validated.instruments["NQ"]["15min"]
  mon = tf.slices["weekday"].groups["monday"]
  r = _row(mon.results, "0930", "large_body")
  # Monday = Day A, bucket 0: ratio=0.80 → large → P=1.0
  assert r.probability == pytest.approx(1.0)
  assert r.count == 1


# ===========================================================================
# 11. i18n: title / definition / labels non-empty en+fr; outcome embeds threshold
# ===========================================================================

def test_stat_name_is_candle_body_ratio() -> None:
  """stat_name == 'candle_body_ratio'."""
  assert _stat().compute(_empty_df()).stat_name == "candle_body_ratio"


def test_result_is_stat_run_result_instance() -> None:
  """compute() returns a StatRunResult instance."""
  assert isinstance(_stat().compute(_make_three_days()), StatRunResult)


def test_i18n_title_non_empty_en_and_fr() -> None:
  """title has non-empty en and fr strings."""
  result = _stat().compute(_empty_df())
  assert result.title.en != ""
  assert result.title.fr != ""


def test_i18n_definition_non_empty_en_and_fr() -> None:
  """definition has non-empty en and fr strings."""
  result = _stat().compute(_empty_df())
  assert result.definition.en != ""
  assert result.definition.fr != ""


def test_i18n_outcome_label_embeds_threshold_50() -> None:
  """The 'large_body' label at candle_size=50 contains '50' in en and fr."""
  result = _stat(candle_size=50.0).compute(_empty_df())
  lbl = result.labels.outcomes["large_body"]
  assert "50" in lbl.en, f"expected '50' in en label: {lbl.en!r}"
  assert "50" in lbl.fr, f"expected '50' in fr label: {lbl.fr!r}"


def test_i18n_outcome_label_embeds_threshold_30() -> None:
  """The 'large_body' label at candle_size=30 contains '30' in en and fr."""
  result = _stat(candle_size=30.0).compute(_empty_df())
  lbl = result.labels.outcomes["large_body"]
  assert "30" in lbl.en
  assert "30" in lbl.fr


def test_i18n_outcome_labels_have_en_and_fr() -> None:
  """Every outcome label has non-empty en and fr strings."""
  result = _stat().compute(_empty_df())
  for key, lbl in result.labels.outcomes.items():
    assert lbl.en != "", f"outcome {key!r}.en is empty"
    assert lbl.fr != "", f"outcome {key!r}.fr is empty"


def test_i18n_condition_labels_have_en_and_fr() -> None:
  """Every condition (bucket) label has non-empty en and fr strings."""
  result = _stat().compute(_empty_df())
  for key, lbl in result.labels.conditions.items():
    assert lbl.en != "", f"condition {key!r}.en is empty"
    assert lbl.fr != "", f"condition {key!r}.fr is empty"


def test_labels_conditions_contains_all_27_buckets() -> None:
  """labels.conditions contains all 27 15-min bucket keys."""
  result = _stat().compute(_empty_df())
  assert "0930" in result.labels.conditions
  assert "0945" in result.labels.conditions
  assert "1600" in result.labels.conditions
  assert len(result.labels.conditions) == _N_BUCKETS_15


def test_labels_dimensions_contains_weekday() -> None:
  """The weekday dimension label appears in labels.dimensions."""
  result = _stat().compute(_make_three_days())
  assert "weekday" in result.labels.dimensions
  assert result.labels.dimensions["weekday"].en != ""
  assert result.labels.dimensions["weekday"].fr != ""


# ===========================================================================
# 12. ValueError: unknown timeframe / candle_size out of [0, 100]
# ===========================================================================

def test_unknown_timeframe_raises_value_error() -> None:
  """Unknown timeframe raises ValueError."""
  with pytest.raises(ValueError, match="Unknown timeframe"):
    CandleBodyRatio(instrument="NQ", config=_TEST_CONFIG, timeframe="2h")


def test_unknown_timeframe_message_mentions_name() -> None:
  """The ValueError message includes the invalid timeframe string."""
  with pytest.raises(ValueError, match="badtf"):
    CandleBodyRatio(instrument="NQ", config=_TEST_CONFIG, timeframe="badtf")


def test_candle_size_negative_raises_value_error() -> None:
  """candle_size < 0.0 raises ValueError."""
  with pytest.raises(ValueError, match="candle_size"):
    CandleBodyRatio(instrument="NQ", config=_TEST_CONFIG, candle_size=-0.1)


def test_candle_size_above_100_raises_value_error() -> None:
  """candle_size > 100.0 raises ValueError."""
  with pytest.raises(ValueError, match="candle_size"):
    CandleBodyRatio(instrument="NQ", config=_TEST_CONFIG, candle_size=100.1)


def test_candle_size_0_accepted() -> None:
  """candle_size=0.0 is a valid boundary value."""
  stat = CandleBodyRatio(instrument="NQ", config=_TEST_CONFIG, candle_size=0.0)
  assert stat.candle_size == 0.0
  assert stat.threshold == pytest.approx(0.0)


def test_candle_size_100_accepted() -> None:
  """candle_size=100.0 is a valid boundary value."""
  stat = CandleBodyRatio(instrument="NQ", config=_TEST_CONFIG, candle_size=100.0)
  assert stat.candle_size == 100.0
  assert stat.threshold == pytest.approx(1.0)


# ===========================================================================
# 13. 1h bucket layout
# ===========================================================================

_TEST_CONFIG_1H = InstrumentConfig(
  instrument="NQ",
  working_timezone=_NY,
  sessions={"rth": _RTH_SESSION},
  timeframes=["1h"],
  parquet_path=Path("data/NQ_1min.parquet"),
)


def _stat_1h(candle_size: float = 50.0) -> CandleBodyRatio:
  return CandleBodyRatio(
    instrument="NQ", config=_TEST_CONFIG_1H, timeframe="1h", candle_size=candle_size
  )


def _make_1h_day(date: str, bucket0_spec: dict | None = None) -> pd.DataFrame:
  """Build a resolved RTH day using 1h bucket spacing.

  Places 1 bar at each 1h bucket start (mods 570, 630, ..., 930) PLUS an extra
  neutral bar at mod=960 so that the close-tolerance criterion (>= 960) is met.

  The extra bar at mod=960 falls inside 1h bucket 6 ((960-570)//60 = 6), making
  bucket 6's candle: open = bar at 930, close = bar at 960 (neutral, close=100).

  Bucket 0 "0930" contains only the bar at mod=570, so its candle equals that
  bar's OHLC directly (single bar → open=bar.open, close=bar.close, high=bar.high,
  low=bar.low).
  """
  base = pd.Timestamp(date, tz=_NY)
  records = []

  # 7 bars at 1h bucket starts: mods 570, 630, 690, 750, 810, 870, 930
  for bucket_idx in range(7):
    mod = 570 + bucket_idx * 60
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    if bucket_idx == 0 and bucket0_spec is not None:
      o = float(bucket0_spec["open"])
      c = float(bucket0_spec["close"])
      hi = max(float(bucket0_spec["high"]), o, c)
      lo = min(float(bucket0_spec["low"]), o, c)
    else:
      o = c = hi = lo = 100.0
    records.append({
      "timestamp": ts,
      "open": o, "high": hi, "low": lo, "close": c, "volume": 100,
    })

  # Extra neutral bar at mod=960 to satisfy close-tolerance (960 >= 960)
  records.append({
    "timestamp": base.replace(hour=16, minute=0, second=0, microsecond=0),
    "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 100,
  })
  return pd.DataFrame(records)


def test_1h_n_buckets() -> None:
  """1h timeframe: n_buckets = ceil(405/60) = 7."""
  # ceil(405/60) = ceil(6.75) = 7
  assert _stat_1h().n_buckets == 7


def test_1h_first_bucket_key_is_0930() -> None:
  """1h timeframe: first bucket key is '0930'."""
  keys = [key for _, key in _stat_1h()._bucket_order]
  assert keys[0] == "0930"


def test_1h_last_bucket_key_is_1530() -> None:
  """1h timeframe: last bucket (index 6) starts at 570+6*60=930 → '1530'."""
  keys = [key for _, key in _stat_1h()._bucket_order]
  # 930 min → hh=15, mm=30 → "1530"
  assert keys[-1] == "1530"


def test_1h_compute_on_three_days_no_error() -> None:
  """1h stat compute() on the 3-day 15min-grid dataset: 3 total_samples."""
  result = _stat_1h().compute(_make_three_days())
  assert result.instruments["NQ"]["1h"].total_samples == 3


def test_1h_explicit_large_body_bucket_0930() -> None:
  """1h bucket '0930': single bar with ratio=1.0 → count=1, P=1.0.

  Bucket 0 bar at mod=570: open=100, close=110, high=110, low=100
    body=10, range=10, ratio=1.0 → LARGE at 50%.
  """
  spec = {"open": 100.0, "close": 110.0, "high": 110.0, "low": 100.0}
  df = _make_1h_day("2024-01-08", bucket0_spec=spec)
  result = _stat_1h().compute(df)
  r = _row(result.instruments["NQ"]["1h"].results, "0930", "large_body")
  # bucket 0: single bar → body=10, range=10, ratio=1.0 → large
  assert r.count == 1
  assert r.total == 1
  assert r.probability == pytest.approx(1.0)


def test_1h_neutral_bucket_not_large() -> None:
  """1h bucket '0930' with flat bar (ratio=0.0) → P=0.0 at candle_size=50."""
  df = _make_1h_day("2024-01-08", bucket0_spec=None)
  result = _stat_1h().compute(df)
  r = _row(result.instruments["NQ"]["1h"].results, "0930", "large_body")
  # neutral: ratio=0.0 < 0.50 → not large
  assert r.count == 0
  assert r.probability == pytest.approx(0.0)


# ===========================================================================
# 14. 30min bucket layout
# ===========================================================================

_TEST_CONFIG_30 = InstrumentConfig(
  instrument="NQ",
  working_timezone=_NY,
  sessions={"rth": _RTH_SESSION},
  timeframes=["30min"],
  parquet_path=Path("data/NQ_1min.parquet"),
)


def _stat_30m(candle_size: float = 50.0) -> CandleBodyRatio:
  return CandleBodyRatio(
    instrument="NQ", config=_TEST_CONFIG_30, timeframe="30min", candle_size=candle_size
  )


def test_30min_n_buckets() -> None:
  """30min timeframe: n_buckets = ceil(405/30) = 14."""
  # ceil(405/30) = ceil(13.5) = 14
  assert _stat_30m().n_buckets == 14


def test_30min_first_bucket_key_is_0930() -> None:
  """30min timeframe: first bucket key is '0930'."""
  keys = [key for _, key in _stat_30m()._bucket_order]
  assert keys[0] == "0930"


def test_30min_last_bucket_key_is_1600() -> None:
  """30min timeframe: last bucket (index 13) starts at 570+13*30=960 → '1600'."""
  keys = [key for _, key in _stat_30m()._bucket_order]
  # 960 min → hh=16, mm=0 → "1600"
  assert keys[-1] == "1600"


def test_30min_explicit_large_body() -> None:
  """30min bucket '0930' with ratio=0.8 → count=1, total=1, P=1.0.

  Bucket 0 (30min builder, n_buckets=14, bucket_min=30):
    open=100, close=108, high=110, low=100 → body=8, range=10, ratio=0.8 → LARGE.
  """
  spec = {0: {"open": 100.0, "close": 108.0, "high": 110.0, "low": 100.0}}
  df = _make_body_day("2024-01-08", spec, n_buckets=14, bucket_min=30)
  result = _stat_30m().compute(df)
  r = _row(result.instruments["NQ"]["30min"].results, "0930", "large_body")
  # body=8, range=10, ratio=0.8 >= 0.5 → large
  assert r.count == 1
  assert r.total == 1
  assert r.probability == pytest.approx(1.0)


def test_30min_compute_single_day_resolved() -> None:
  """30min stat resolves the single-day 30min dataset (last bar at mod=960)."""
  # 30min builder: n_buckets=14, bucket_min=30; last bar at mod=960 >= 960 ✓
  df = _make_body_day("2024-01-08", {}, n_buckets=14, bucket_min=30)
  result = _stat_30m().compute(df)
  assert result.instruments["NQ"]["30min"].total_samples == 1


def test_30min_last_bucket_1600_is_present() -> None:
  """The '1600' bucket appears in 30min results."""
  df = _make_body_day("2024-01-08", {}, n_buckets=14, bucket_min=30)
  result = _stat_30m().compute(df)
  keys = {r.condition for r in result.instruments["NQ"]["30min"].results}
  assert "1600" in keys


# ===========================================================================
# 15. build_day_table column structure and values
# ===========================================================================

def test_build_day_table_has_body_columns_for_all_buckets() -> None:
  """build_day_table has columns body_0 .. body_26 for all 27 buckets."""
  dt = _stat().build_day_table(_make_three_days())
  for i in range(_N_BUCKETS_15):
    assert f"body_{i}" in dt.columns, f"column body_{i} missing"


def test_build_day_table_index_is_normalized_dates() -> None:
  """build_day_table index consists of the 3 normalized session dates."""
  dt = _stat().build_day_table(_make_three_days())
  expected = {
    pd.Timestamp("2024-01-08", tz=_NY).normalize(),
    pd.Timestamp("2024-01-09", tz=_NY).normalize(),
    pd.Timestamp("2024-01-10", tz=_NY).normalize(),
  }
  assert expected == set(dt.index)


def test_build_day_table_bucket0_day_a_flag_is_one() -> None:
  """Day A bucket 0: ratio=0.80 >= 0.50 → body_0=1.0."""
  dt = _stat().build_day_table(_make_three_days())
  date_a = pd.Timestamp("2024-01-08", tz=_NY).normalize()
  assert dt.loc[date_a, "body_0"] == pytest.approx(1.0)


def test_build_day_table_bucket0_day_b_flag_is_zero() -> None:
  """Day B bucket 0: ratio=0.30 < 0.50 → body_0=0.0."""
  dt = _stat().build_day_table(_make_three_days())
  date_b = pd.Timestamp("2024-01-09", tz=_NY).normalize()
  assert dt.loc[date_b, "body_0"] == pytest.approx(0.0)


def test_build_day_table_bucket1_day_b_flag_is_one() -> None:
  """Day B bucket 1: ratio=0.60 >= 0.50 → body_1=1.0."""
  dt = _stat().build_day_table(_make_three_days())
  date_b = pd.Timestamp("2024-01-09", tz=_NY).normalize()
  assert dt.loc[date_b, "body_1"] == pytest.approx(1.0)


def test_build_day_table_neutral_bucket_flag_is_zero() -> None:
  """Neutral bucket 2 (ratio=0.0 < 0.50): body_2=0.0 (present but not large)."""
  dt = _stat().build_day_table(_make_three_days())
  date_a = pd.Timestamp("2024-01-08", tz=_NY).normalize()
  # bucket 2 is neutral → ratio=0.0 → flag=0.0, not NaN
  assert dt.loc[date_a, "body_2"] == pytest.approx(0.0)


def test_build_day_table_no_nan_for_full_session_days() -> None:
  """With 1 bar per bucket in every day, no NaN appears in the day table."""
  dt = _stat().build_day_table(_make_three_days())
  assert not dt.isnull().any().any(), "Unexpected NaN in day_table"


# ===========================================================================
# 16. Single-day edge cases
# ===========================================================================

def test_single_day_total_samples_is_one() -> None:
  """Single resolved day → total_samples == 1."""
  df = _make_body_day("2024-01-08", _SPEC_A)
  assert _tf(_stat().compute(df)).total_samples == 1


def test_single_day_data_range_is_one_date() -> None:
  """Single resolved day → data_range == ['2024-01-08', '2024-01-08']."""
  df = _make_body_day("2024-01-08", _SPEC_A)
  assert _tf(_stat().compute(df)).data_range == ["2024-01-08", "2024-01-08"]


def test_single_day_one_weekday_group() -> None:
  """Single day (Mon 2024-01-08) → only 'monday' weekday group, total_samples=1."""
  df = _make_body_day("2024-01-08", _SPEC_A)
  groups = _tf(_stat().compute(df)).slices["weekday"].groups
  assert set(groups.keys()) == {"monday"}
  assert groups["monday"].total_samples == 1


def test_single_day_probabilities_are_zero_or_one() -> None:
  """With 1 day, every bucket probability is exactly 0.0 or 1.0 (no fractions)."""
  df = _make_body_day("2024-01-08", _SPEC_A)
  for r in _tf(_stat().compute(df)).results:
    assert r.probability in (pytest.approx(0.0), pytest.approx(1.0)), (
      f"({r.condition}): probability={r.probability}"
    )


# ===========================================================================
# 17. Data range and total_samples
# ===========================================================================

def test_data_range_spans_first_to_last_day() -> None:
  """Three resolved Mon–Wed → data_range = ['2024-01-08', '2024-01-10']."""
  result = _stat().compute(_make_three_days())
  dr = _tf(result).data_range
  assert dr[0] == "2024-01-08"
  assert dr[1] == "2024-01-10"


def test_total_samples_equals_resolved_day_count() -> None:
  """Three resolved sessions → total_samples == 3."""
  assert _tf(_stat().compute(_make_three_days())).total_samples == 3
