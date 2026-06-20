"""Tests for stats.intraday_volume_range.standard.

All data is synthetic — no real market files required.
Expected values are hand-calculated before each assertion.

Bucket arithmetic reference (RTH start = 09:30 = 570 min, 15min buckets):
  n_buckets = ceil(405 / 15) = 27
  bucket_idx = (mod - 570) // 15
  bucket 0  = "0930"  mod 570-584   (15 bars, full)
  bucket 1  = "0945"  mod 585-599   (15 bars, full)
  bucket 2  = "1000"  mod 600-614   (15 bars, full)
  bucket 26 = "1600"  mod 960-974   (15 bars, partial/last)

Resolution criterion:
  - A bar at mod=570 must exist (clean session open).
  - The day's last RTH bar mod >= 975 - 15 = 960.

Per-bucket metrics:
  vol     = sum of all bars' volume in the bucket
  rng     = max(high) - min(low) across the bucket's bars
  pct     = rng / bucket_open   (bucket_open = open of first bar in bucket)

Day-table averages are NaN-skipping means across resolved days.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.intraday_volume_range.standard import IntradayVolumeRange

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

# RTH constants
_RTH_START = 570   # 09:30
_RTH_END = 975     # 16:15 (exclusive)
_RTH_LAST = 974    # 16:14 (last 1-min bar in RTH)
_CLOSE_THRESH = 960  # 16:00 — last bar mod must be >= this

# With 15min buckets, each full bucket has 15 bars.
_BARS_PER_BUCKET = 15


# ---------------------------------------------------------------------------
# Per-bucket specification type
# ---------------------------------------------------------------------------

def _bucket_start_mod(bucket_idx: int, bucket_min: int = 15) -> int:
  """Minute-of-day for the first bar in a given bucket index."""
  return _RTH_START + bucket_idx * bucket_min


# ---------------------------------------------------------------------------
# Synthetic day builders
# ---------------------------------------------------------------------------

def _make_day_from_bucket_specs(
  date: str,
  bucket_specs: dict[int, dict],
  n_buckets: int = 27,
  bucket_min: int = 15,
) -> pd.DataFrame:
  """Build one fully resolved RTH day with controlled per-bucket OHLCV.

  Each bucket covers ``bucket_min`` 1-min bars. The caller supplies a dict of
  bucket_idx -> spec where spec has:
    vol_per_bar : volume placed on each bar in the bucket
    high        : the high value placed on the bucket's FIRST bar; this is
                  also the ONLY high extreme — all other bars use
                  ``high - 0.01`` as their high, ensuring max(high) == spec high.
    low         : the low value placed on the bucket's LAST bar; this is
                  also the ONLY low extreme — all other bars use
                  ``low + 0.01`` as their low, ensuring min(low) == spec low.
    bucket_open : the open price of the bucket's first bar

  Buckets not listed in ``bucket_specs`` get a flat neutral bar with:
    open=100, high=100.25, low=99.75, close=100, volume=50

  The day runs from mod=570 to mod=974 (all 27 * 15 = 405 1-min bars).

  Hand-calculation of per-bucket metrics for a spec:
    vol = vol_per_bar * bucket_min          (all bars carry vol_per_bar)
    rng = high - low                         (max high is first bar, min low is last bar)
    pct = (high - low) / bucket_open
  """
  base = pd.Timestamp(date, tz=_NY)
  records = []

  for bucket_idx in range(n_buckets):
    start_mod = _RTH_START + bucket_idx * bucket_min
    # Last bucket may have fewer bars: mods from start_mod to _RTH_LAST inclusive.
    end_mod = min(start_mod + bucket_min, _RTH_END) - 1
    spec = bucket_specs.get(bucket_idx)

    for mod in range(start_mod, end_mod + 1):
      h, m = divmod(mod, 60)
      ts = base.replace(hour=h, minute=m, second=0, microsecond=0)

      if spec is not None:
        spec_high = float(spec["high"])
        spec_low = float(spec["low"])
        o = float(spec["bucket_open"]) if mod == start_mod else (spec_low + spec_high) / 2.0
        c = (spec_low + spec_high) / 2.0
        vol = int(spec["vol_per_bar"])
        # First bar carries the unique maximum high; last bar carries the unique minimum low.
        # Middle bars use slightly-inside values so they never become the extremes.
        if mod == start_mod:
          bar_high = spec_high
          bar_low = spec_low + 0.01  # not the minimum
        elif mod == end_mod:
          bar_high = spec_high - 0.01  # not the maximum
          bar_low = spec_low
        else:
          bar_high = spec_high - 0.01
          bar_low = spec_low + 0.01
        # Ensure OHLC validity: high >= max(open,close), low <= min(open,close).
        bar_high = max(o, c, bar_high)
        bar_low = min(o, c, bar_low)
      else:
        o = 100.0
        c = 100.0
        vol = 50
        bar_high = 100.25
        bar_low = 99.75

      records.append({
        "timestamp": ts,
        "open": o,
        "high": bar_high,
        "low": bar_low,
        "close": c,
        "volume": vol,
      })

  df = pd.DataFrame(records)
  return df


def _make_truncated_day(date: str, last_mod: int = 590) -> pd.DataFrame:
  """A day whose last RTH bar is at ``last_mod`` < 960 — must be excluded.

  Default: last bar at mod=590 (09:50).
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


def _stat(timeframe: str = "15min") -> IntradayVolumeRange:
  return IntradayVolumeRange(instrument="NQ", config=_TEST_CONFIG, timeframe=timeframe)


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
# We control buckets 0 ("0930") and 1 ("0945") explicitly; all other
# buckets use neutral bars (vol=50/bar, high=100.25, low=99.75, open=100).
#
# Day A  2024-01-08 Mon — bucket specs:
#   bucket 0 "0930": vol_per_bar=100, high=110.0, low=100.0, bucket_open=100.0
#     -> vol_0  = 100 * 15 = 1500
#     -> rng_0  = 110.0 - 100.0 = 10.0
#     -> pct_0  = 10.0 / 100.0 = 0.10
#   bucket 1 "0945": vol_per_bar=200, high=108.0, low=102.0, bucket_open=105.0
#     -> vol_1  = 200 * 15 = 3000
#     -> rng_1  = 108.0 - 102.0 = 6.0
#     -> pct_1  = 6.0 / 105.0 ≈ 0.05714286
#
# Day B  2024-01-09 Tue — bucket specs:
#   bucket 0 "0930": vol_per_bar=300, high=120.0, low=95.0, bucket_open=100.0
#     -> vol_0  = 300 * 15 = 4500
#     -> rng_0  = 120.0 - 95.0 = 25.0
#     -> pct_0  = 25.0 / 100.0 = 0.25
#   bucket 1 "0945": vol_per_bar=400, high=115.0, low=103.0, bucket_open=110.0
#     -> vol_1  = 400 * 15 = 6000
#     -> rng_1  = 115.0 - 103.0 = 12.0
#     -> pct_1  = 12.0 / 110.0 ≈ 0.10909091
#
# Day C  2024-01-10 Wed — bucket specs:
#   bucket 0 "0930": vol_per_bar=500, high=130.0, low=90.0, bucket_open=100.0
#     -> vol_0  = 500 * 15 = 7500
#     -> rng_0  = 130.0 - 90.0 = 40.0
#     -> pct_0  = 40.0 / 100.0 = 0.40
#   bucket 1 "0945": vol_per_bar=600, high=125.0, low=105.0, bucket_open=115.0
#     -> vol_1  = 600 * 15 = 9000
#     -> rng_1  = 125.0 - 105.0 = 20.0
#     -> pct_1  = 20.0 / 115.0 ≈ 0.17391304
#
# Day D  2024-01-15 Mon — TRUNCATED (last bar mod=590 < 960) — excluded
#
# Overall averages across A, B, C (3 days):
#   bucket "0930":
#     mean_volume    = (1500 + 4500 + 7500) / 3 = 13500 / 3 = 4500.0
#     mean_range     = (10.0 + 25.0 + 40.0) / 3 = 75.0 / 3 = 25.0
#     mean_range_pct = (0.10 + 0.25 + 0.40) / 3 = 0.75 / 3 = 0.25
#
#   bucket "0945":
#     mean_volume    = (3000 + 6000 + 9000) / 3 = 18000 / 3 = 6000.0
#     mean_range     = (6.0 + 12.0 + 20.0) / 3 = 38.0 / 3 ≈ 12.6667
#     mean_range_pct = (6/105 + 12/110 + 20/115) / 3
#                    = (0.05714286 + 0.10909091 + 0.17391304) / 3
#                    = 0.34014681 / 3 ≈ 0.11338227
# ---------------------------------------------------------------------------

_SPEC_A = {
  0: {"vol_per_bar": 100, "high": 110.0, "low": 100.0, "bucket_open": 100.0},
  1: {"vol_per_bar": 200, "high": 108.0, "low": 102.0, "bucket_open": 105.0},
}
_SPEC_B = {
  0: {"vol_per_bar": 300, "high": 120.0, "low": 95.0,  "bucket_open": 100.0},
  1: {"vol_per_bar": 400, "high": 115.0, "low": 103.0, "bucket_open": 110.0},
}
_SPEC_C = {
  0: {"vol_per_bar": 500, "high": 130.0, "low": 90.0,  "bucket_open": 100.0},
  1: {"vol_per_bar": 600, "high": 125.0, "low": 105.0, "bucket_open": 115.0},
}

# Pre-computed per-bucket metric expectations for the 3-day dataset
_VOL_A_B0   = 100 * 15      # 1500
_VOL_B_B0   = 300 * 15      # 4500
_VOL_C_B0   = 500 * 15      # 7500
_VOL_A_B1   = 200 * 15      # 3000
_VOL_B_B1   = 400 * 15      # 6000
_VOL_C_B1   = 600 * 15      # 9000

_RNG_A_B0 = 110.0 - 100.0   # 10.0
_RNG_B_B0 = 120.0 - 95.0    # 25.0
_RNG_C_B0 = 130.0 - 90.0    # 40.0
_RNG_A_B1 = 108.0 - 102.0   # 6.0
_RNG_B_B1 = 115.0 - 103.0   # 12.0
_RNG_C_B1 = 125.0 - 105.0   # 20.0

_PCT_A_B0 = _RNG_A_B0 / 100.0   # 0.10
_PCT_B_B0 = _RNG_B_B0 / 100.0   # 0.25
_PCT_C_B0 = _RNG_C_B0 / 100.0   # 0.40
_PCT_A_B1 = _RNG_A_B1 / 105.0   # 6/105
_PCT_B_B1 = _RNG_B_B1 / 110.0   # 12/110
_PCT_C_B1 = _RNG_C_B1 / 115.0   # 20/115

# Mean across 3 days
_MEAN_VOL_B0  = (_VOL_A_B0 + _VOL_B_B0 + _VOL_C_B0) / 3   # 4500.0
_MEAN_VOL_B1  = (_VOL_A_B1 + _VOL_B_B1 + _VOL_C_B1) / 3   # 6000.0
_MEAN_RNG_B0  = (_RNG_A_B0 + _RNG_B_B0 + _RNG_C_B0) / 3   # 25.0
_MEAN_RNG_B1  = (_RNG_A_B1 + _RNG_B_B1 + _RNG_C_B1) / 3   # 38/3
_MEAN_PCT_B0  = (_PCT_A_B0 + _PCT_B_B0 + _PCT_C_B0) / 3   # 0.25
_MEAN_PCT_B1  = (_PCT_A_B1 + _PCT_B_B1 + _PCT_C_B1) / 3   # (6/105+12/110+20/115)/3


def _make_three_days() -> pd.DataFrame:
  """Three fully resolved RTH sessions (Mon/Tue/Wed 2024-01-08..10)."""
  return _concat_days([
    _make_day_from_bucket_specs("2024-01-08", _SPEC_A),
    _make_day_from_bucket_specs("2024-01-09", _SPEC_B),
    _make_day_from_bucket_specs("2024-01-10", _SPEC_C),
  ])


def _make_three_days_with_truncated() -> pd.DataFrame:
  """Three resolved days + one truncated day on 2024-01-15."""
  return _concat_days([
    _make_day_from_bucket_specs("2024-01-08", _SPEC_A),
    _make_day_from_bucket_specs("2024-01-09", _SPEC_B),
    _make_day_from_bucket_specs("2024-01-10", _SPEC_C),
    _make_truncated_day("2024-01-15"),
  ])


# ===========================================================================
# 1. Known per-bucket averages (standard / overall)
# ===========================================================================

def test_bucket_0930_mean_volume() -> None:
  """Bucket "0930" mean_volume = (1500+4500+7500)/3 = 4500.0 across 3 resolved days."""
  result = _stat().compute(_make_three_days())
  # mean_volume = (100*15 + 300*15 + 500*15) / 3 = (1500+4500+7500)/3 = 4500
  r = _row(_tf(result).results, "0930", "mean_volume")
  assert r.value == pytest.approx(_MEAN_VOL_B0)  # 4500.0


def test_bucket_0930_mean_range() -> None:
  """Bucket "0930" mean_range = (10+25+40)/3 = 25.0 across 3 resolved days."""
  result = _stat().compute(_make_three_days())
  # rng per day: 110-100=10, 120-95=25, 130-90=40; mean=(10+25+40)/3=25.0
  r = _row(_tf(result).results, "0930", "mean_range")
  assert r.value == pytest.approx(_MEAN_RNG_B0)  # 25.0


def test_bucket_0930_mean_range_pct() -> None:
  """Bucket "0930" mean_range_pct = (0.10+0.25+0.40)/3 = 0.25 across 3 days."""
  result = _stat().compute(_make_three_days())
  # pct per day: 10/100=0.10, 25/100=0.25, 40/100=0.40; mean=0.75/3=0.25
  r = _row(_tf(result).results, "0930", "mean_range_pct")
  assert r.value == pytest.approx(_MEAN_PCT_B0)  # 0.25


def test_bucket_0945_mean_volume() -> None:
  """Bucket "0945" mean_volume = (3000+6000+9000)/3 = 6000.0."""
  result = _stat().compute(_make_three_days())
  # vol per day: 200*15=3000, 400*15=6000, 600*15=9000; mean=6000.0
  r = _row(_tf(result).results, "0945", "mean_volume")
  assert r.value == pytest.approx(_MEAN_VOL_B1)  # 6000.0


def test_bucket_0945_mean_range() -> None:
  """Bucket "0945" mean_range = (6+12+20)/3 = 38/3 ≈ 12.6667."""
  result = _stat().compute(_make_three_days())
  # rng: 108-102=6, 115-103=12, 125-105=20; mean=38/3
  r = _row(_tf(result).results, "0945", "mean_range")
  assert r.value == pytest.approx(_MEAN_RNG_B1)  # 38/3


def test_bucket_0945_mean_range_pct() -> None:
  """Bucket "0945" mean_range_pct = (6/105 + 12/110 + 20/115) / 3."""
  result = _stat().compute(_make_three_days())
  # pct: 6/105, 12/110, 20/115; mean = sum/3
  r = _row(_tf(result).results, "0945", "mean_range_pct")
  assert r.value == pytest.approx(_MEAN_PCT_B1)


def test_bucket_count_equals_n_resolved_days() -> None:
  """Each bucket's count and total equal the number of resolved days (3)."""
  result = _stat().compute(_make_three_days())
  for outcome in ("mean_volume", "mean_range", "mean_range_pct"):
    r = _row(_tf(result).results, "0930", outcome)
    assert r.count == 3
    assert r.total == 3


# ===========================================================================
# 2. Condition/outcome shape and probability channel
# ===========================================================================

def test_each_present_bucket_yields_three_rows() -> None:
  """Every present bucket produces exactly 3 rows (one per outcome)."""
  result = _stat().compute(_make_three_days())
  rows = _tf(result).results
  from collections import Counter
  bucket_counts = Counter(r.condition for r in rows)
  # Every bucket seen in the dataset should have exactly 3 rows.
  for bucket_key, cnt in bucket_counts.items():
    assert cnt == 3, f"bucket {bucket_key!r} has {cnt} rows, expected 3"


def test_probability_is_zero_on_all_rows() -> None:
  """This is a magnitude stat: probability == 0.0 on every result row."""
  result = _stat().compute(_make_three_days())
  for r in _tf(result).results:
    assert r.probability == pytest.approx(0.0), (
      f"({r.condition}, {r.outcome}): probability={r.probability}"
    )


def test_value_is_not_none_on_all_rows() -> None:
  """Every result row carries a non-None value (magnitude channel)."""
  result = _stat().compute(_make_three_days())
  for r in _tf(result).results:
    assert r.value is not None, f"({r.condition}, {r.outcome}): value is None"


def test_labels_conditions_contains_all_present_buckets() -> None:
  """labels.conditions must contain a key for every present bucket."""
  result = _stat().compute(_make_three_days())
  # All 27 bucket keys are registered on the stat's Labels.
  assert "0930" in result.labels.conditions
  assert "0945" in result.labels.conditions
  assert "1600" in result.labels.conditions
  assert len(result.labels.conditions) == 27  # all 27 15-min buckets


def test_labels_outcomes_has_three_metric_keys() -> None:
  """labels.outcomes carries exactly the 3 outcome keys."""
  result = _stat().compute(_make_three_days())
  assert set(result.labels.outcomes) == {"mean_volume", "mean_range", "mean_range_pct"}


# ===========================================================================
# 3. Partial last bucket ("1600") is present and computed
# ===========================================================================

def test_partial_last_bucket_1600_is_present() -> None:
  """The 16:00 bucket ("1600") appears in the result rows (even though partial)."""
  result = _stat().compute(_make_three_days())
  bucket_keys = {r.condition for r in _tf(result).results}
  assert "1600" in bucket_keys


def test_partial_last_bucket_1600_metrics_computed() -> None:
  """The "1600" bucket has valid (non-zero) metric values computed from its bars.

  Neutral bars in bucket 26 ("1600"):
    vol_per_bar=50, high=100.25, low=99.75, open=100 (neutral spec)
    vol = 50 * 15 = 750
    rng = 100.25 - 99.75 = 0.5
    pct = 0.5 / 100 = 0.005
  All 3 days use neutral bars for bucket 26 so the averages equal those values.
  """
  result = _stat().compute(_make_three_days())
  # Bucket 26 "1600" uses neutral bars: vol=50/bar*15=750, rng=0.5, pct=0.005
  r_vol = _row(_tf(result).results, "1600", "mean_volume")
  assert r_vol.value == pytest.approx(750.0)

  r_rng = _row(_tf(result).results, "1600", "mean_range")
  assert r_rng.value == pytest.approx(0.5)

  r_pct = _row(_tf(result).results, "1600", "mean_range_pct")
  assert r_pct.value == pytest.approx(0.005)


def test_partial_last_bucket_1600_count_is_three() -> None:
  """The "1600" bucket has count=3 from 3 resolved days."""
  result = _stat().compute(_make_three_days())
  r = _row(_tf(result).results, "1600", "mean_volume")
  assert r.count == 3


# ===========================================================================
# 4. Pending/truncated exclusion
# ===========================================================================

def test_truncated_day_excluded_from_total_samples() -> None:
  """A truncated day (last bar mod=590 < 960) is not counted in total_samples."""
  result = _stat().compute(_make_three_days_with_truncated())
  # Only 3 resolved days; the 2024-01-15 truncated day must be excluded.
  assert _tf(result).total_samples == 3


def test_truncated_day_absent_from_day_table() -> None:
  """build_day_table excludes the truncated day from its index."""
  df = _make_three_days_with_truncated()
  dt = _stat().build_day_table(df)
  excluded = pd.Timestamp("2024-01-15", tz=_NY).normalize()
  assert excluded not in dt.index
  assert len(dt) == 3


def test_truncated_day_does_not_affect_bucket_averages() -> None:
  """Appending a truncated day leaves bucket averages unchanged (same 3 days)."""
  result_clean = _stat().compute(_make_three_days())
  result_trunc = _stat().compute(_make_three_days_with_truncated())

  r_clean = _row(_tf(result_clean).results, "0930", "mean_volume")
  r_trunc = _row(_tf(result_trunc).results, "0930", "mean_volume")
  assert r_trunc.value == pytest.approx(r_clean.value)
  assert r_trunc.count == r_clean.count == 3


def test_truncated_day_excluded_from_bucket_counts() -> None:
  """No bucket's count increases due to the truncated day."""
  result = _stat().compute(_make_three_days_with_truncated())
  for r in _tf(result).results:
    assert r.count <= 3, f"bucket {r.condition} {r.outcome} has count={r.count}"


# ===========================================================================
# 5. Weekday slice
# ===========================================================================

def test_weekday_slice_groups_present() -> None:
  """3-day Mon/Tue/Wed dataset -> weekday slicer has exactly 3 groups."""
  result = _stat().compute(_make_three_days())
  groups = _tf(result).slices["weekday"].groups
  assert set(groups.keys()) == {"monday", "tuesday", "wednesday"}


def test_weekday_each_group_total_samples() -> None:
  """Each weekday group has total_samples == 1 (one day per weekday)."""
  result = _stat().compute(_make_three_days())
  for key, grp in _tf(result).slices["weekday"].groups.items():
    assert grp.total_samples == 1, f"{key}: total_samples={grp.total_samples}"


def test_weekday_monday_bucket_0930_mean_volume() -> None:
  """Monday = Day A: bucket "0930" mean_volume equals single-day value 1500.

  With only 1 day in the group, the group mean == that day's bucket vol.
  Day A bucket 0: vol = 100 * 15 = 1500
  """
  result = _stat().compute(_make_three_days())
  mon_rows = _tf(result).slices["weekday"].groups["monday"].results
  # Only Day A falls on Monday; bucket "0930" vol = 100*15 = 1500
  r = _row(mon_rows, "0930", "mean_volume")
  assert r.value == pytest.approx(float(_VOL_A_B0))  # 1500


def test_weekday_tuesday_bucket_0930_mean_range() -> None:
  """Tuesday = Day B: bucket "0930" mean_range = 120-95 = 25.0."""
  result = _stat().compute(_make_three_days())
  tue_rows = _tf(result).slices["weekday"].groups["tuesday"].results
  # Day B bucket 0: rng = 120-95 = 25
  r = _row(tue_rows, "0930", "mean_range")
  assert r.value == pytest.approx(_RNG_B_B0)  # 25.0


def test_weekday_wednesday_bucket_0945_mean_range_pct() -> None:
  """Wednesday = Day C: bucket "0945" mean_range_pct = 20/115."""
  result = _stat().compute(_make_three_days())
  wed_rows = _tf(result).slices["weekday"].groups["wednesday"].results
  # Day C bucket 1: pct = (125-105)/115 = 20/115
  r = _row(wed_rows, "0945", "mean_range_pct")
  assert r.value == pytest.approx(_PCT_C_B1)  # 20/115


def test_weekday_slice_probability_zero() -> None:
  """Weekday slice rows also carry probability == 0.0 (magnitude stat)."""
  result = _stat().compute(_make_three_days())
  for key, grp in _tf(result).slices["weekday"].groups.items():
    for r in grp.results:
      assert r.probability == pytest.approx(0.0), (
        f"weekday={key} ({r.condition},{r.outcome}): probability={r.probability}"
      )


# ===========================================================================
# 6. Reproducibility
# ===========================================================================

def test_compute_twice_identical_results() -> None:
  """compute() with default seed returns identical rows on both calls."""
  df = _make_three_days()
  stat = _stat()
  result_a = stat.compute(df)
  result_b = stat.compute(df)
  rows_a = _tf(result_a).results
  rows_b = _tf(result_b).results
  assert len(rows_a) == len(rows_b)
  for a, b in zip(rows_a, rows_b):
    assert a.condition == b.condition
    assert a.outcome == b.outcome
    assert a.value == pytest.approx(b.value)
    if a.value_baseline is None:
      assert b.value_baseline is None
    else:
      assert a.value_baseline == pytest.approx(b.value_baseline)


def test_compute_twice_identical_model_dump() -> None:
  """compute() twice on the same input -> identical model_dump()."""
  df = _make_three_days()
  stat = _stat()
  assert stat.compute(df).model_dump() == stat.compute(df).model_dump()


def test_baseline_rows_same_seed_deterministic() -> None:
  """Two baseline_rows calls with the same seed produce identical output."""
  stat = _stat()
  day_table = stat.build_day_table(_make_three_days())
  rows_a = stat.baseline_rows(day_table, seed=7)
  rows_b = stat.baseline_rows(day_table, seed=7)
  assert len(rows_a) == len(rows_b)
  for a, b in zip(rows_a, rows_b):
    assert a.condition == b.condition
    assert a.outcome == b.outcome
    assert a.value == pytest.approx(b.value)


# ===========================================================================
# 7. Baseline sanity
# ===========================================================================

def test_baseline_rows_exist_in_overall_result() -> None:
  """Every overall result row has a non-None value_baseline."""
  result = _stat().compute(_make_three_days())
  for r in _tf(result).results:
    assert r.value_baseline is not None, (
      f"({r.condition},{r.outcome}) missing value_baseline"
    )


def test_baseline_n_populated() -> None:
  """Every overall result row has baseline_n > 0."""
  result = _stat().compute(_make_three_days())
  for r in _tf(result).results:
    assert r.baseline_n > 0, f"({r.condition},{r.outcome}): baseline_n=0"


def test_baseline_value_is_float() -> None:
  """value_baseline is a float on every row."""
  result = _stat().compute(_make_three_days())
  for r in _tf(result).results:
    assert isinstance(r.value_baseline, float), (
      f"({r.condition},{r.outcome}): value_baseline type={type(r.value_baseline)}"
    )


def test_baseline_mean_volume_within_observed_range() -> None:
  """For mean_volume, baseline draws from the pool of all bucket observations.

  The pool contains all per-(day,bucket) volumes; the baseline mean must lie
  within [min(pool), max(pool)]. With 3 days and 27 buckets, the pool is
  large and the baseline is well within bounds.

  Observed mean_volume for "0930": range [min_bucket_vol, max_bucket_vol].
  All bucket vol values across 3 days: varies per bucket (neutral=750, specified).
  We check the "0930" baseline is a float in a reasonable range.
  """
  result = _stat().compute(_make_three_days())
  r = _row(_tf(result).results, "0930", "mean_volume")
  assert isinstance(r.value_baseline, float)
  # The pool spans from neutral bucket vol (50*15=750) to max specified (500*15=7500).
  # The baseline mean must fall within the full pool's min/max.
  assert r.value_baseline >= 750.0
  assert r.value_baseline <= 7500.0


def test_baseline_mean_range_within_observed_range() -> None:
  """For mean_range, baseline value lies within [min_pool, max_pool].

  Neutral bucket range = 100.25-99.75 = 0.5.
  Max specified range = 130-90 = 40.
  """
  result = _stat().compute(_make_three_days())
  r = _row(_tf(result).results, "0930", "mean_range")
  assert isinstance(r.value_baseline, float)
  assert r.value_baseline >= 0.5
  assert r.value_baseline <= 40.0


def test_baseline_different_seeds_can_differ() -> None:
  """Different seeds produce different baseline values (with high probability).

  With 3 days and 27 buckets, the pool for each metric has 81 observations.
  Drawing n=3 without replacement with different seeds will almost always differ.
  """
  stat = _stat()
  day_table = stat.build_day_table(_make_three_days())
  rows_seed1 = stat.baseline_rows(day_table, seed=1)
  rows_seed2 = stat.baseline_rows(day_table, seed=2)
  # At least one bucket/outcome pair should differ between seeds.
  any_diff = any(
    abs((a.value or 0.0) - (b.value or 0.0)) > 1e-9
    for a, b in zip(rows_seed1, rows_seed2)
    if a.condition == b.condition and a.outcome == b.outcome
  )
  assert any_diff, "All baseline values were identical for seed=1 vs seed=2"


# ===========================================================================
# 8. Empty input
# ===========================================================================

def test_empty_dataframe_total_samples_zero() -> None:
  """Empty candles -> total_samples == 0."""
  result = _stat().compute(_empty_df())
  assert _tf(result).total_samples == 0


def test_empty_dataframe_results_empty() -> None:
  """Empty candles -> results == [] (no rows produced)."""
  result = _stat().compute(_empty_df())
  assert _tf(result).results == []


def test_empty_dataframe_data_range_empty() -> None:
  """Empty candles -> data_range == []."""
  result = _stat().compute(_empty_df())
  assert _tf(result).data_range == []


def test_empty_dataframe_no_weekday_groups() -> None:
  """Empty candles -> weekday slicer produces no groups."""
  result = _stat().compute(_empty_df())
  assert _tf(result).slices["weekday"].groups == {}


def test_empty_build_day_table_returns_empty_frame() -> None:
  """build_day_table on empty input returns an empty DataFrame."""
  dt = _stat().build_day_table(_empty_df())
  assert len(dt) == 0


# ===========================================================================
# 9. Validation / round-trip
# ===========================================================================

def test_stat_name_is_intraday_volume_range() -> None:
  """stat_name attribute must equal 'intraday_volume_range'."""
  result = _stat().compute(_empty_df())
  assert result.stat_name == "intraday_volume_range"


def test_result_is_stat_run_result_instance() -> None:
  """compute() returns a valid StatRunResult instance."""
  result = _stat().compute(_make_three_days())
  assert isinstance(result, StatRunResult)


def test_i18n_title_and_definition_non_empty() -> None:
  """title and definition carry non-empty en/fr strings."""
  result = _stat().compute(_empty_df())
  assert result.title.en != ""
  assert result.title.fr != ""
  assert result.definition.en != ""
  assert result.definition.fr != ""


def test_i18n_outcome_labels_have_en_and_fr() -> None:
  """Every outcome label has non-empty en and fr strings."""
  result = _stat().compute(_empty_df())
  for key, i18n in result.labels.outcomes.items():
    assert i18n.en != "", f"outcome {key!r}.en is empty"
    assert i18n.fr != "", f"outcome {key!r}.fr is empty"


def test_i18n_condition_labels_have_en_and_fr() -> None:
  """Every condition (bucket) label has non-empty en and fr strings."""
  result = _stat().compute(_empty_df())
  for key, i18n in result.labels.conditions.items():
    assert i18n.en != "", f"condition {key!r}.en is empty"
    assert i18n.fr != "", f"condition {key!r}.fr is empty"


def test_labels_dimensions_contains_weekday() -> None:
  """The weekday dimension label is surfaced in labels.dimensions."""
  result = _stat().compute(_make_three_days())
  assert "weekday" in result.labels.dimensions
  assert result.labels.dimensions["weekday"].en != ""
  assert result.labels.dimensions["weekday"].fr != ""


def test_write_results_round_trip(tmp_path: Path) -> None:
  """write_results serialises to JSON; the file validates back to StatRunResult."""
  result = _stat().compute(_make_three_days())
  written = write_results(result, results_dir=tmp_path)

  assert written.name == "intraday_volume_range.json"
  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)

  tf = validated.instruments["NQ"]["15min"]
  assert tf.total_samples == 3

  # Spot-check: "0930" mean_volume = 4500.0 after round-trip.
  r = _row(tf.results, "0930", "mean_volume")
  # mean_volume = (1500+4500+7500)/3 = 4500
  assert r.value == pytest.approx(4500.0)
  assert r.count == 3

  # Weekday slice must survive round-trip.
  mon = tf.slices["weekday"].groups["monday"]
  mon_vol = _row(mon.results, "0930", "mean_volume")
  # Monday only: 100*15 = 1500
  assert mon_vol.value == pytest.approx(1500.0)


def test_write_results_stat_name_in_file(tmp_path: Path) -> None:
  """The serialised JSON file carries stat_name == 'intraday_volume_range'."""
  result = _stat().compute(_make_three_days())
  written = write_results(result, results_dir=tmp_path)
  assert written.stem == "intraday_volume_range"
  raw = json.loads(written.read_text(encoding="utf-8"))
  assert raw["stat_name"] == "intraday_volume_range"


# ===========================================================================
# 10. Unknown timeframe raises ValueError
# ===========================================================================

def test_unknown_timeframe_raises_value_error() -> None:
  """IntradayVolumeRange with an unknown timeframe raises ValueError."""
  with pytest.raises(ValueError, match="Unknown timeframe"):
    IntradayVolumeRange(instrument="NQ", config=_TEST_CONFIG, timeframe="2h")


def test_unknown_timeframe_message_includes_choices() -> None:
  """The ValueError message lists the valid timeframe choices."""
  with pytest.raises(ValueError, match="15min"):
    IntradayVolumeRange(instrument="NQ", config=_TEST_CONFIG, timeframe="badtf")


# ===========================================================================
# 11. Additional correctness: single-day edge case
# ===========================================================================

def test_single_day_bucket_averages_equal_day_metrics() -> None:
  """With one resolved day, bucket average == that day's bucket metric exactly.

  Day A only: bucket "0930" vol=1500, rng=10, pct=0.10.
  """
  df = _make_day_from_bucket_specs("2024-01-08", _SPEC_A)
  result = _stat().compute(df)
  tf = _tf(result)
  assert tf.total_samples == 1

  # vol = 100*15 = 1500
  assert _row(tf.results, "0930", "mean_volume").value == pytest.approx(1500.0)
  # rng = 110-100 = 10.0
  assert _row(tf.results, "0930", "mean_range").value == pytest.approx(10.0)
  # pct = 10/100 = 0.10
  assert _row(tf.results, "0930", "mean_range_pct").value == pytest.approx(0.10)


def test_single_day_data_range() -> None:
  """Single resolved day -> data_range is [date, date]."""
  df = _make_day_from_bucket_specs("2024-01-08", _SPEC_A)
  result = _stat().compute(df)
  assert _tf(result).data_range == ["2024-01-08", "2024-01-08"]


def test_single_day_one_weekday_group() -> None:
  """Single day (Monday 2024-01-08) -> only 'monday' weekday group."""
  df = _make_day_from_bucket_specs("2024-01-08", _SPEC_A)
  result = _stat().compute(df)
  groups = _tf(result).slices["weekday"].groups
  assert set(groups.keys()) == {"monday"}
  assert groups["monday"].total_samples == 1


# ===========================================================================
# 12. Data range reflects resolved day table
# ===========================================================================

def test_data_range_three_days() -> None:
  """Three days Mon–Wed -> data_range spans 2024-01-08 to 2024-01-10."""
  result = _stat().compute(_make_three_days())
  dr = _tf(result).data_range
  assert dr[0] == "2024-01-08"
  assert dr[1] == "2024-01-10"


def test_total_samples_three_resolved() -> None:
  """Three resolved sessions -> total_samples == 3."""
  result = _stat().compute(_make_three_days())
  assert _tf(result).total_samples == 3


# ===========================================================================
# 13. Supported timeframe: 1h bucket layout
# ===========================================================================

_TEST_CONFIG_1H = InstrumentConfig(
  instrument="NQ",
  working_timezone=_NY,
  sessions={"rth": _RTH_SESSION},
  timeframes=["1h"],
  parquet_path=Path("data/NQ_1min.parquet"),
)


def _stat_1h() -> IntradayVolumeRange:
  return IntradayVolumeRange(instrument="NQ", config=_TEST_CONFIG_1H, timeframe="1h")


def test_1h_n_buckets() -> None:
  """1h timeframe: n_buckets = ceil(405/60) = 7."""
  stat = _stat_1h()
  # ceil(405/60) = ceil(6.75) = 7
  assert stat.n_buckets == 7


def test_1h_first_bucket_key() -> None:
  """1h timeframe: first bucket key is "0930"."""
  stat = _stat_1h()
  keys = [key for _, key in stat._bucket_order]
  assert keys[0] == "0930"


def test_1h_compute_runs_without_error() -> None:
  """1h timeframe compute() on 3-day data runs without errors."""
  result = _stat_1h().compute(_make_three_days())
  assert _tf(result, "1h").total_samples == 3


def test_1h_bucket_0930_mean_volume() -> None:
  """1h timeframe: bucket "0930" aggregates 60 bars per day.

  With bucket_min=60, bucket 0 spans mods 570-629 (60 bars).
  Our test data places specified bars in 15-min buckets 0 and 1 (mods 570-599),
  and neutral bars in the remaining 15-min buckets within the 1h window.

  1h bucket 0 "0930" (mods 570-629) per day:
    - Buckets 0..3 in 15min terms (mods 570-629) but our spec only defines
      15min buckets 0 (570-584) and 1 (585-599). Mods 600-614 (15min bucket 2)
      and 615-629 (15min bucket 3) are neutral.

  Day A: vol = (100*15) + (200*15) + (50*15) + (50*15) = 1500+3000+750+750 = 6000
  Day B: vol = (300*15) + (400*15) + (50*15) + (50*15) = 4500+6000+750+750 = 12000
  Day C: vol = (500*15) + (600*15) + (50*15) + (50*15) = 7500+9000+750+750 = 18000
  Mean = (6000+12000+18000)/3 = 36000/3 = 12000.0
  """
  result = _stat_1h().compute(_make_three_days())
  r = _row(_tf(result, "1h").results, "0930", "mean_volume")
  # mean = (6000+12000+18000)/3 = 12000.0
  assert r.value == pytest.approx(12000.0)


# ===========================================================================
# 14. build_day_table column structure
# ===========================================================================

def test_build_day_table_has_vol_col_per_bucket() -> None:
  """build_day_table has columns vol_0, vol_1, ... for 27 buckets."""
  dt = _stat().build_day_table(_make_three_days())
  for i in range(27):
    assert f"vol_{i}" in dt.columns, f"col vol_{i} missing"


def test_build_day_table_has_rng_col_per_bucket() -> None:
  """build_day_table has columns rng_0, rng_1, ... for 27 buckets."""
  dt = _stat().build_day_table(_make_three_days())
  for i in range(27):
    assert f"rng_{i}" in dt.columns, f"col rng_{i} missing"


def test_build_day_table_has_pct_col_per_bucket() -> None:
  """build_day_table has columns pct_0, pct_1, ... for 27 buckets."""
  dt = _stat().build_day_table(_make_three_days())
  for i in range(27):
    assert f"pct_{i}" in dt.columns, f"col pct_{i} missing"


def test_build_day_table_index_is_normalized_date() -> None:
  """build_day_table index is normalized session dates."""
  dt = _stat().build_day_table(_make_three_days())
  expected_dates = {
    pd.Timestamp("2024-01-08", tz=_NY).normalize(),
    pd.Timestamp("2024-01-09", tz=_NY).normalize(),
    pd.Timestamp("2024-01-10", tz=_NY).normalize(),
  }
  assert expected_dates == set(dt.index)


def test_build_day_table_bucket0_vol_day_a() -> None:
  """Day A bucket 0 (mods 570-584): vol_0 = 100 * 15 = 1500."""
  dt = _stat().build_day_table(_make_three_days())
  date_a = pd.Timestamp("2024-01-08", tz=_NY).normalize()
  # bucket 0: 15 bars at vol_per_bar=100 -> 1500
  assert dt.loc[date_a, "vol_0"] == pytest.approx(1500.0)


def test_build_day_table_bucket0_rng_day_b() -> None:
  """Day B bucket 0: rng_0 = 120 - 95 = 25.0."""
  dt = _stat().build_day_table(_make_three_days())
  date_b = pd.Timestamp("2024-01-09", tz=_NY).normalize()
  # bucket 0: high=120, low=95 -> rng=25
  assert dt.loc[date_b, "rng_0"] == pytest.approx(25.0)


def test_build_day_table_bucket0_pct_day_c() -> None:
  """Day C bucket 0: pct_0 = (130-90)/100 = 0.40."""
  dt = _stat().build_day_table(_make_three_days())
  date_c = pd.Timestamp("2024-01-10", tz=_NY).normalize()
  # pct = rng / bucket_open = 40 / 100 = 0.40
  assert dt.loc[date_c, "pct_0"] == pytest.approx(0.40)


def test_build_day_table_bucket1_pct_day_a() -> None:
  """Day A bucket 1: pct_1 = (108-102)/105 = 6/105."""
  dt = _stat().build_day_table(_make_three_days())
  date_a = pd.Timestamp("2024-01-08", tz=_NY).normalize()
  # pct = (108-102)/105 = 6/105
  assert dt.loc[date_a, "pct_1"] == pytest.approx(6.0 / 105.0)


# ===========================================================================
# 15. Valid timeframe strings are accepted
# ===========================================================================

def test_valid_timeframe_1min_accepted() -> None:
  """'1min' timeframe is accepted without error."""
  cfg = InstrumentConfig(
    instrument="NQ",
    working_timezone=_NY,
    sessions={"rth": _RTH_SESSION},
    timeframes=["1min"],
    parquet_path=Path("data/NQ_1min.parquet"),
  )
  stat = IntradayVolumeRange(instrument="NQ", config=cfg, timeframe="1min")
  assert stat.bucket_min == 1


def test_valid_timeframe_5min_accepted() -> None:
  """'5min' timeframe is accepted without error."""
  cfg = InstrumentConfig(
    instrument="NQ",
    working_timezone=_NY,
    sessions={"rth": _RTH_SESSION},
    timeframes=["5min"],
    parquet_path=Path("data/NQ_1min.parquet"),
  )
  stat = IntradayVolumeRange(instrument="NQ", config=cfg, timeframe="5min")
  assert stat.bucket_min == 5


def test_valid_timeframe_30min_n_buckets() -> None:
  """'30min' timeframe: n_buckets = ceil(405/30) = 14."""
  cfg = InstrumentConfig(
    instrument="NQ",
    working_timezone=_NY,
    sessions={"rth": _RTH_SESSION},
    timeframes=["30min"],
    parquet_path=Path("data/NQ_1min.parquet"),
  )
  stat = IntradayVolumeRange(instrument="NQ", config=cfg, timeframe="30min")
  # ceil(405/30) = ceil(13.5) = 14
  assert stat.n_buckets == 14
