"""Tests for stats.economic_data_volume.standard.

All data is synthetic — no real market files required.
Expected values are hand-calculated before each assertion.

Synthetic dataset — 4 resolved RTH days + 1 unresolved day:

  Day A: 2024-01-02 — CPI + FOMC (dual-event), RTH volume = 1000 + 2000 = 3000
  Day B: 2024-01-03 — NFP event,                RTH volume = 800  + 1200 = 2000
  Day C: 2024-01-04 — GDP event,                RTH volume = 1500 + 2500 = 4000
  Day D: 2024-01-05 — non-event,                RTH volume = 400  + 600  = 1000
  Day E: 2024-01-08 — UNRESOLVED (last RTH bar at 09:50, mod 590 < 960)

Event dates:
  cpi:  {2024-01-02}
  fomc: {2024-01-02}   (same date as CPI — dual-event day)
  nfp:  {2024-01-03}
  gdp:  {2024-01-04}

Hand-calculated expected values (pending-discipline: Day E excluded):
  resolved days: A, B, C, D — total_samples=4
  data_range: ["2024-01-02", "2024-01-05"]

  cpi_day:       N=1, mean_volume=3000.0
  fomc_day:      N=1, mean_volume=3000.0
  nfp_day:       N=1, mean_volume=2000.0
  gdp_day:       N=1, mean_volume=4000.0
  any_event_day: N=3, mean_volume=(3000+2000+4000)/3=3000.0
  non_event_day: N=1, mean_volume=1000.0

RTH session: 09:30-16:15. Resolution requires:
  - bar at mod=570 (09:30) — session open
  - last RTH bar at mod>=960 (e.g. mod=974 = 16:14)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.economic_data_volume.standard import EconomicDataVolume, load_gdp_release_dates

# ---------------------------------------------------------------------------
# Minimal InstrumentConfig — does not depend on NQ.yaml
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

# Minute-of-day constants
_RTH_OPEN_MOD = 570  # 09:30 — must be present for resolution
_RTH_LAST_MOD = 974  # 16:14 — mod 974 >= resolved_min 960
_TRUNC_MOD = 590     # 09:50 — mod 590 < 960 → unresolved


# ---------------------------------------------------------------------------
# Bar and day builders
# ---------------------------------------------------------------------------

def _bar(ts: pd.Timestamp, volume: int) -> dict:
  """One OHLCV bar with constant OHLC and given volume."""
  return {
    "timestamp": ts,
    "open": 100.0,
    "high": 100.25,
    "low": 99.75,
    "close": 100.0,
    "volume": volume,
  }


def _resolved_day(date: str, open_vol: int, last_vol: int) -> list[dict]:
  """Two-bar resolved RTH day: 09:30 open + 16:14 last bar.

  Both bars are inside [570, 975); last mod=974 >= 960 → resolved.
  RTH volume = open_vol + last_vol.
  """
  base = pd.Timestamp(date, tz=_NY)
  return [
    _bar(base.replace(hour=9, minute=30, second=0, microsecond=0), open_vol),
    _bar(base.replace(hour=16, minute=14, second=0, microsecond=0), last_vol),
  ]


def _unresolved_day(date: str) -> list[dict]:
  """RTH day with a session-open bar but truncated before mod>=960.

  09:30 bar (mod=570) is present (clean open); 09:50 bar (mod=590 < 960)
  is the last bar, so build_resolved_days excludes the day.
  """
  base = pd.Timestamp(date, tz=_NY)
  return [
    _bar(base.replace(hour=9, minute=30, second=0, microsecond=0), 500),
    _bar(base.replace(hour=9, minute=50, second=0, microsecond=0), 200),
  ]


def _make_candles(bar_lists: list[list[dict]]) -> pd.DataFrame:
  """Concatenate bar lists into a sorted 1-min OHLCV DataFrame."""
  records = [bar for bars in bar_lists for bar in bars]
  df = pd.DataFrame(records)
  return df.sort_values("timestamp").reset_index(drop=True)


def _empty_df() -> pd.DataFrame:
  df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
  df["timestamp"] = pd.array([], dtype="datetime64[ns, America/New_York]")
  return df


# ---------------------------------------------------------------------------
# Stat factory and result helpers
# ---------------------------------------------------------------------------

def _stat(
  cpi_dates: list[str] | None = None,
  fomc_dates: list[str] | None = None,
  nfp_dates: list[str] | None = None,
  gdp_dates: list[str] | None = None,
) -> EconomicDataVolume:
  """Build an EconomicDataVolume instance with synthetic event dates."""
  return EconomicDataVolume(
    instrument="NQ",
    config=_TEST_CONFIG,
    event_dates={
      "cpi":  [pd.Timestamp(d) for d in (cpi_dates or [])],
      "fomc": [pd.Timestamp(d) for d in (fomc_dates or [])],
      "nfp":  [pd.Timestamp(d) for d in (nfp_dates or [])],
      "gdp":  [pd.Timestamp(d) for d in (gdp_dates or [])],
    },
  )


def _row(result: StatRunResult, condition: str) -> StatResultRow:
  """Extract the single mean_volume row for a given condition."""
  for r in result.instruments["NQ"]["daily"].results:
    if r.condition == condition and r.outcome == "mean_volume":
      return r
  raise KeyError(condition)


# ---------------------------------------------------------------------------
# Main 4-day resolved + 1 unresolved dataset
#
#   Day A: 2024-01-02 — CPI + FOMC, volume = 1000 + 2000 = 3000
#   Day B: 2024-01-03 — NFP,         volume = 800  + 1200 = 2000
#   Day C: 2024-01-04 — GDP,         volume = 1500 + 2500 = 4000
#   Day D: 2024-01-05 — non-event,   volume = 400  + 600  = 1000
#   Day E: 2024-01-08 — UNRESOLVED   (excluded by build_resolved_days)
# ---------------------------------------------------------------------------

def _build_main_stat() -> EconomicDataVolume:
  return _stat(
    cpi_dates=["2024-01-02"],
    fomc_dates=["2024-01-02"],
    nfp_dates=["2024-01-03"],
    gdp_dates=["2024-01-04"],
  )


def _build_main_candles() -> pd.DataFrame:
  return _make_candles([
    _resolved_day("2024-01-02", 1000, 2000),  # Day A: CPI+FOMC, vol=3000
    _resolved_day("2024-01-03", 800, 1200),   # Day B: NFP,       vol=2000
    _resolved_day("2024-01-04", 1500, 2500),  # Day C: GDP,       vol=4000
    _resolved_day("2024-01-05", 400, 600),    # Day D: non-event, vol=1000
    _unresolved_day("2024-01-08"),            # Day E: UNRESOLVED
  ])


# ===========================================================================
# 1. build_day_table — shape, columns, volumes
# ===========================================================================

def test_build_day_table_excludes_unresolved_day() -> None:
  """Day E (unresolved: last bar at 09:50, mod 590 < 960) is absent; 4 rows remain."""
  stat = _build_main_stat()
  table = stat.build_day_table(_build_main_candles())
  assert len(table) == 4
  unresolved_date = pd.Timestamp("2024-01-08").normalize()
  assert unresolved_date not in table.index


def test_build_day_table_columns_present() -> None:
  """Day table carries exactly the expected boolean flag columns plus volume."""
  stat = _build_main_stat()
  table = stat.build_day_table(_build_main_candles())
  expected_cols = {
    "volume", "is_cpi", "is_fomc", "is_nfp", "is_gdp", "is_any_event", "is_non_event",
  }
  assert set(table.columns) == expected_cols


def test_build_day_table_rth_volume_correct() -> None:
  """Per-day RTH volume equals the sum of all RTH bar volumes for that day.

  Day A: 1000 + 2000 = 3000
  Day B: 800  + 1200 = 2000
  Day C: 1500 + 2500 = 4000
  Day D: 400  + 600  = 1000
  """
  stat = _build_main_stat()
  table = stat.build_day_table(_build_main_candles())
  assert table.loc[pd.Timestamp("2024-01-02"), "volume"] == pytest.approx(3000.0)
  assert table.loc[pd.Timestamp("2024-01-03"), "volume"] == pytest.approx(2000.0)
  assert table.loc[pd.Timestamp("2024-01-04"), "volume"] == pytest.approx(4000.0)
  assert table.loc[pd.Timestamp("2024-01-05"), "volume"] == pytest.approx(1000.0)


def test_build_day_table_index_tz_naive() -> None:
  """Day table index is tz-naive (normalized midnight Timestamps)."""
  stat = _build_main_stat()
  table = stat.build_day_table(_build_main_candles())
  assert table.index.tz is None


# ===========================================================================
# 2. Dual-event day: CPI + FOMC on 2024-01-02
# ===========================================================================

def test_dual_event_day_both_individual_flags_set() -> None:
  """Day A (2024-01-02) has is_cpi=True and is_fomc=True simultaneously."""
  stat = _build_main_stat()
  table = stat.build_day_table(_build_main_candles())
  row = table.loc[pd.Timestamp("2024-01-02")]
  assert bool(row["is_cpi"]) is True
  assert bool(row["is_fomc"]) is True


def test_dual_event_day_any_event_true() -> None:
  """is_any_event is True for a day carrying both CPI and FOMC flags."""
  stat = _build_main_stat()
  table = stat.build_day_table(_build_main_candles())
  assert bool(table.loc[pd.Timestamp("2024-01-02"), "is_any_event"]) is True


def test_dual_event_day_non_event_false() -> None:
  """is_non_event is False for a dual-event day (it is an event day)."""
  stat = _build_main_stat()
  table = stat.build_day_table(_build_main_candles())
  assert bool(table.loc[pd.Timestamp("2024-01-02"), "is_non_event"]) is False


def test_dual_event_day_counts_once_in_any_event() -> None:
  """any_event_day N=3 (Days A, B, C); Day A counted once despite two flags."""
  # any_event = Day A (CPI+FOMC) + Day B (NFP) + Day C (GDP) = 3 days
  result = _build_main_stat().compute(_build_main_candles())
  r = _row(result, "any_event_day")
  assert r.count == 3
  assert r.total == 3


# ===========================================================================
# 3. Event flag assignment per event type
# ===========================================================================

def test_event_flags_nfp_day() -> None:
  """Day B (2024-01-03): is_nfp=True, all others False, is_any_event=True."""
  stat = _build_main_stat()
  table = stat.build_day_table(_build_main_candles())
  row = table.loc[pd.Timestamp("2024-01-03")]
  assert bool(row["is_nfp"]) is True
  assert bool(row["is_cpi"]) is False
  assert bool(row["is_fomc"]) is False
  assert bool(row["is_gdp"]) is False
  assert bool(row["is_any_event"]) is True
  assert bool(row["is_non_event"]) is False


def test_event_flags_gdp_day() -> None:
  """Day C (2024-01-04): is_gdp=True, all others False, is_any_event=True."""
  stat = _build_main_stat()
  table = stat.build_day_table(_build_main_candles())
  row = table.loc[pd.Timestamp("2024-01-04")]
  assert bool(row["is_gdp"]) is True
  assert bool(row["is_cpi"]) is False
  assert bool(row["is_fomc"]) is False
  assert bool(row["is_nfp"]) is False
  assert bool(row["is_any_event"]) is True
  assert bool(row["is_non_event"]) is False


def test_event_flags_non_event_day() -> None:
  """Day D (2024-01-05): all flags False, is_any_event=False, is_non_event=True."""
  stat = _build_main_stat()
  table = stat.build_day_table(_build_main_candles())
  row = table.loc[pd.Timestamp("2024-01-05")]
  assert bool(row["is_cpi"]) is False
  assert bool(row["is_fomc"]) is False
  assert bool(row["is_nfp"]) is False
  assert bool(row["is_gdp"]) is False
  assert bool(row["is_any_event"]) is False
  assert bool(row["is_non_event"]) is True


def test_any_event_is_union_of_four_flags() -> None:
  """is_any_event == (is_cpi | is_fomc | is_nfp | is_gdp) for all rows."""
  stat = _build_main_stat()
  table = stat.build_day_table(_build_main_candles())
  union = (
    table["is_cpi"] | table["is_fomc"] | table["is_nfp"] | table["is_gdp"]
  )
  pd.testing.assert_series_equal(
    table["is_any_event"].astype(bool),
    union.astype(bool),
    check_names=False,
  )


def test_non_event_is_complement_of_any_event() -> None:
  """is_non_event == ~is_any_event for all rows."""
  stat = _build_main_stat()
  table = stat.build_day_table(_build_main_candles())
  pd.testing.assert_series_equal(
    table["is_non_event"].astype(bool),
    (~table["is_any_event"]).astype(bool),
    check_names=False,
  )


# ===========================================================================
# 4. compute_rows — 6 rows, exact mean_volume values, probability==0.0
#
# cpi_day:       N=1, mean = 3000.0
# fomc_day:      N=1, mean = 3000.0
# nfp_day:       N=1, mean = 2000.0
# gdp_day:       N=1, mean = 4000.0
# any_event_day: N=3, mean = (3000+2000+4000)/3 = 3000.0
# non_event_day: N=1, mean = 1000.0
# ===========================================================================

def test_exactly_six_result_rows() -> None:
  """compute() always returns exactly 6 result rows."""
  result = _build_main_stat().compute(_build_main_candles())
  rows = result.instruments["NQ"]["daily"].results
  assert len(rows) == 6


def test_result_conditions_and_outcome_keys() -> None:
  """The (condition, outcome) pairs match the exact 6-row specification."""
  result = _build_main_stat().compute(_build_main_candles())
  rows = result.instruments["NQ"]["daily"].results
  pairs = {(r.condition, r.outcome) for r in rows}
  expected = {
    ("cpi_day", "mean_volume"),
    ("fomc_day", "mean_volume"),
    ("nfp_day", "mean_volume"),
    ("gdp_day", "mean_volume"),
    ("any_event_day", "mean_volume"),
    ("non_event_day", "mean_volume"),
  }
  assert pairs == expected


def test_cpi_day_mean_volume() -> None:
  # Day A is the only CPI day: mean = 3000.0
  result = _build_main_stat().compute(_build_main_candles())
  r = _row(result, "cpi_day")
  assert r.count == 1
  assert r.total == 1
  assert r.value == pytest.approx(3000.0)
  assert r.probability == pytest.approx(0.0)


def test_fomc_day_mean_volume() -> None:
  # Day A is the only FOMC day (same as CPI): mean = 3000.0
  result = _build_main_stat().compute(_build_main_candles())
  r = _row(result, "fomc_day")
  assert r.count == 1
  assert r.total == 1
  assert r.value == pytest.approx(3000.0)
  assert r.probability == pytest.approx(0.0)


def test_nfp_day_mean_volume() -> None:
  # Day B is the only NFP day: mean = 2000.0
  result = _build_main_stat().compute(_build_main_candles())
  r = _row(result, "nfp_day")
  assert r.count == 1
  assert r.total == 1
  assert r.value == pytest.approx(2000.0)
  assert r.probability == pytest.approx(0.0)


def test_gdp_day_mean_volume() -> None:
  # Day C is the only GDP day: mean = 4000.0
  result = _build_main_stat().compute(_build_main_candles())
  r = _row(result, "gdp_day")
  assert r.count == 1
  assert r.total == 1
  assert r.value == pytest.approx(4000.0)
  assert r.probability == pytest.approx(0.0)


def test_any_event_day_mean_volume() -> None:
  # Days A, B, C: mean = (3000+2000+4000)/3 = 3000.0
  result = _build_main_stat().compute(_build_main_candles())
  r = _row(result, "any_event_day")
  assert r.count == 3
  assert r.total == 3
  assert r.value == pytest.approx(3000.0)
  assert r.probability == pytest.approx(0.0)


def test_non_event_day_mean_volume() -> None:
  # Day D is the only non-event day: mean = 1000.0
  result = _build_main_stat().compute(_build_main_candles())
  r = _row(result, "non_event_day")
  assert r.count == 1
  assert r.total == 1
  assert r.value == pytest.approx(1000.0)
  assert r.probability == pytest.approx(0.0)


def test_count_equals_total_all_rows() -> None:
  """For all rows, count == total (magnitude stat: no separate numerator/denominator)."""
  result = _build_main_stat().compute(_build_main_candles())
  for row in result.instruments["NQ"]["daily"].results:
    assert row.count == row.total, f"{row.condition}: count != total"


def test_probability_is_zero_for_all_rows() -> None:
  """All rows carry probability==0.0 (magnitude stat, not a probability stat)."""
  result = _build_main_stat().compute(_build_main_candles())
  for row in result.instruments["NQ"]["daily"].results:
    assert row.probability == pytest.approx(0.0), f"{row.condition}: probability != 0.0"


# ===========================================================================
# 5. total_samples and data_range
#
# 4 resolved days → total_samples=4
# data_range = ["2024-01-02", "2024-01-05"]
# ===========================================================================

def test_total_samples_equals_resolved_days() -> None:
  """total_samples == 4 (Day E excluded as unresolved)."""
  result = _build_main_stat().compute(_build_main_candles())
  assert result.instruments["NQ"]["daily"].total_samples == 4


def test_total_samples_equals_len_day_table() -> None:
  """total_samples == len(build_day_table(df))."""
  stat = _build_main_stat()
  df = _build_main_candles()
  result = stat.compute(df)
  table = stat.build_day_table(df)
  assert result.instruments["NQ"]["daily"].total_samples == len(table)


def test_data_range_spans_first_to_last_resolved_day() -> None:
  """data_range = ["2024-01-02", "2024-01-05"] (first to last resolved day)."""
  result = _build_main_stat().compute(_build_main_candles())
  assert result.instruments["NQ"]["daily"].data_range == ["2024-01-02", "2024-01-05"]


# ===========================================================================
# 6. Baseline reproducibility
# ===========================================================================

def test_baseline_same_seed_deterministic() -> None:
  """baseline_rows(seed=42) is byte-for-byte identical on two calls."""
  stat = _build_main_stat()
  df = _build_main_candles()
  table = stat.build_day_table(df)
  rows_a = stat.baseline_rows(table, seed=42)
  rows_b = stat.baseline_rows(table, seed=42)
  assert len(rows_a) == len(rows_b) == 6
  for a, b in zip(rows_a, rows_b):
    assert (a.condition, a.outcome) == (b.condition, b.outcome)
    assert a.count == b.count
    assert a.total == b.total
    assert a.value == pytest.approx(b.value)


def test_baseline_different_seeds_may_differ() -> None:
  """seed=42 and seed=99 produce at least one row with a different value."""
  stat = _build_main_stat()
  table = stat.build_day_table(_build_main_candles())
  rows_42 = stat.baseline_rows(table, seed=42)
  rows_99 = stat.baseline_rows(table, seed=99)
  any_diff = any(
    abs((a.value or 0.0) - (b.value or 0.0)) > 1e-9
    for a, b in zip(rows_42, rows_99)
  )
  assert any_diff, "seed=42 and seed=99 produced identical baseline values"


def test_baseline_condition_sample_sizes_match_real() -> None:
  """Baseline rows have the same count/total as the real compute_rows rows."""
  stat = _build_main_stat()
  df = _build_main_candles()
  table = stat.build_day_table(df)
  real_rows = stat.compute_rows(table)
  bl_rows = stat.baseline_rows(table, seed=42)
  assert len(real_rows) == len(bl_rows) == 6
  for real, bl in zip(real_rows, bl_rows):
    assert (real.condition, real.outcome) == (bl.condition, bl.outcome)
    assert real.total == bl.total


def test_baseline_embedded_in_compute_sets_value_baseline() -> None:
  """After compute(), every result row carries a non-None value_baseline."""
  result = _build_main_stat().compute(_build_main_candles(), seed=42)
  for row in result.instruments["NQ"]["daily"].results:
    assert row.value_baseline is not None, (
      f"Missing value_baseline for {row.condition}"
    )


def test_compute_reproducible_same_seed() -> None:
  """Same input + same seed → identical values and value_baselines."""
  stat = _build_main_stat()
  df = _build_main_candles()
  result_a = stat.compute(df, seed=42)
  result_b = stat.compute(df, seed=42)
  for a, b in zip(
    result_a.instruments["NQ"]["daily"].results,
    result_b.instruments["NQ"]["daily"].results,
  ):
    assert (a.condition, a.outcome) == (b.condition, b.outcome)
    assert a.value == pytest.approx(b.value)
    assert a.value_baseline == pytest.approx(b.value_baseline)


# ===========================================================================
# 7. Empty and edge cases
# ===========================================================================

def test_empty_candles_build_day_table_returns_empty() -> None:
  """Empty candles DataFrame → build_day_table returns empty, no crash."""
  stat = _build_main_stat()
  table = stat.build_day_table(_empty_df())
  assert table.empty


def test_empty_candles_six_zero_rows() -> None:
  """Empty candles → compute() returns 6 zeroed rows, no crash."""
  result = _build_main_stat().compute(_empty_df())
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 0
  assert tf.data_range == []
  assert len(tf.results) == 6
  for r in tf.results:
    assert r.count == 0
    assert r.total == 0
    assert r.value == pytest.approx(0.0)
    assert r.probability == pytest.approx(0.0)


def test_no_event_dates_all_days_appear_as_non_event() -> None:
  """With no event dates, all resolved days appear with is_non_event=True."""
  stat = _stat()  # all event date lists empty
  table = stat.build_day_table(_build_main_candles())
  # 4 resolved days returned (Day E excluded)
  assert len(table) == 4
  assert not bool(table["is_any_event"].any())
  assert table["is_non_event"].all()


def test_no_event_dates_non_event_condition_has_all_days() -> None:
  """With no event dates, non_event_day N=4 and mean=(3000+2000+4000+1000)/4=2500."""
  stat = _stat()
  result = stat.compute(_build_main_candles())
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 4
  # Individual event conditions: N=0
  for cond in ("cpi_day", "fomc_day", "nfp_day", "gdp_day", "any_event_day"):
    r = _row(result, cond)
    assert r.count == 0
    assert r.total == 0
    assert r.value == pytest.approx(0.0)
  # non_event_day: all 4 resolved days; mean = (3000+2000+4000+1000)/4 = 2500.0
  ne = _row(result, "non_event_day")
  assert ne.count == 4
  assert ne.total == 4
  assert ne.value == pytest.approx(2500.0)


def test_single_resolved_event_day() -> None:
  """Single resolved CPI day: cpi_day N=1, mean equals that day's RTH volume."""
  # Day A only: open_vol=1000, last_vol=2000 → RTH volume=3000
  df = _make_candles([_resolved_day("2024-01-02", 1000, 2000)])
  stat = _stat(cpi_dates=["2024-01-02"])
  result = stat.compute(df)
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 1
  r = _row(result, "cpi_day")
  assert r.count == 1
  assert r.value == pytest.approx(3000.0)
  # non_event_day: N=0 (no non-event days)
  ne = _row(result, "non_event_day")
  assert ne.count == 0
  assert ne.value == pytest.approx(0.0)


def test_unresolved_day_excluded_from_day_table() -> None:
  """A day whose last RTH bar is at 09:50 (mod<960) is excluded from build_day_table."""
  df = _make_candles([
    _resolved_day("2024-01-02", 1000, 2000),
    _unresolved_day("2024-01-03"),
  ])
  stat = _stat(cpi_dates=["2024-01-02", "2024-01-03"])
  table = stat.build_day_table(df)
  # Only 2024-01-02 is resolved; 2024-01-03 is excluded
  assert len(table) == 1
  assert pd.Timestamp("2024-01-03").normalize() not in table.index


def test_event_date_outside_data_range_not_in_table() -> None:
  """An event date with no matching candles in the data is silently excluded."""
  # event date 2024-01-07 (Sunday) has no candles; only Day A has candles
  df = _make_candles([_resolved_day("2024-01-02", 1000, 2000)])
  stat = _stat(cpi_dates=["2024-01-07"])
  table = stat.build_day_table(df)
  # 2024-01-02 is resolved but NOT a CPI date; 2024-01-07 has no candles
  # is_cpi will be False for 2024-01-02; table has 1 row (not filtered by CPI)
  assert len(table) == 1
  # The 2024-01-02 row is present but has is_cpi=False
  assert bool(table.loc[pd.Timestamp("2024-01-02"), "is_cpi"]) is False


def test_event_date_entirely_outside_data_no_crash() -> None:
  """All event dates are outside the data range; compute() returns all-zero rows."""
  stat = _stat(cpi_dates=["2023-01-01"])
  df = _make_candles([_resolved_day("2024-01-02", 1000, 2000)])
  result = stat.compute(df)
  # Day A is resolved but 2024-01-02 is not a CPI date; cpi_day N=0
  r = _row(result, "cpi_day")
  assert r.count == 0
  assert r.value == pytest.approx(0.0)


# ===========================================================================
# 8. load_gdp_release_dates
#
# CSV rows:
#   2024-01-26, USD, Advance GDP q/q   → INCLUDE
#   2024-01-26, USD, Advance GDP q/q   → DUPLICATE → deduped
#   2024-01-27, USD, GDP Price Index q/q → EXCLUDE (deflator, not a real-GDP vintage)
#   2024-01-26, EUR, Advance GDP q/q   → EXCLUDE (not USD)
#   2024-04-25, USD, Prelim GDP q/q    → INCLUDE
#   2024-07-25, USD, Final GDP q/q     → INCLUDE
#   2024-07-25, USD, FOMC Statement    → EXCLUDE (wrong event)
#
# Expected: {Timestamp("2024-01-26"), Timestamp("2024-04-25"), Timestamp("2024-07-25")}
# ===========================================================================

_GDP_CSV_CONTENT = """\
date,currency,event
2024-01-26,USD,Advance GDP q/q
2024-01-26,USD,Advance GDP q/q
2024-01-27,USD,GDP Price Index q/q
2024-01-26,EUR,Advance GDP q/q
2024-04-25,USD,Prelim GDP q/q
2024-07-25,USD,Final GDP q/q
2024-07-25,USD,FOMC Statement
"""


def test_load_gdp_advance_included(tmp_path: Path) -> None:
  """Advance GDP q/q (USD) rows are included."""
  csv = tmp_path / "cal.csv"
  csv.write_text(_GDP_CSV_CONTENT, encoding="utf-8")
  dates = load_gdp_release_dates(csv)
  assert pd.Timestamp("2024-01-26") in dates


def test_load_gdp_prelim_included(tmp_path: Path) -> None:
  """Prelim GDP q/q (USD) rows are included."""
  csv = tmp_path / "cal.csv"
  csv.write_text(_GDP_CSV_CONTENT, encoding="utf-8")
  dates = load_gdp_release_dates(csv)
  assert pd.Timestamp("2024-04-25") in dates


def test_load_gdp_final_included(tmp_path: Path) -> None:
  """Final GDP q/q (USD) rows are included."""
  csv = tmp_path / "cal.csv"
  csv.write_text(_GDP_CSV_CONTENT, encoding="utf-8")
  dates = load_gdp_release_dates(csv)
  assert pd.Timestamp("2024-07-25") in dates


def test_load_gdp_price_index_excluded(tmp_path: Path) -> None:
  """GDP Price Index q/q rows are excluded; their date does not appear in results."""
  csv = tmp_path / "cal.csv"
  csv.write_text(_GDP_CSV_CONTENT, encoding="utf-8")
  dates = load_gdp_release_dates(csv)
  # 2024-01-27 has only a GDP Price Index row — it must be absent
  assert pd.Timestamp("2024-01-27") not in dates


def test_load_gdp_total_count_three(tmp_path: Path) -> None:
  """Exactly 3 distinct dates survive filtering and deduplication."""
  csv = tmp_path / "cal.csv"
  csv.write_text(_GDP_CSV_CONTENT, encoding="utf-8")
  dates = load_gdp_release_dates(csv)
  assert len(dates) == 3


def test_load_gdp_non_usd_excluded(tmp_path: Path) -> None:
  """EUR Advance GDP q/q rows are excluded."""
  content = "date,currency,event\n2024-01-26,EUR,Advance GDP q/q\n"
  csv = tmp_path / "eur_only.csv"
  csv.write_text(content, encoding="utf-8")
  dates = load_gdp_release_dates(csv)
  assert pd.Timestamp("2024-01-26") not in dates
  assert len(dates) == 0


def test_load_gdp_fomc_excluded(tmp_path: Path) -> None:
  """FOMC Statement rows on a GDP date are excluded; date retained from GDP row."""
  csv = tmp_path / "cal.csv"
  csv.write_text(_GDP_CSV_CONTENT, encoding="utf-8")
  dates = load_gdp_release_dates(csv)
  # 2024-07-25 is present because of Final GDP q/q; FOMC Statement must not inflate count
  assert len(dates) == 3


def test_load_gdp_deduplicates_same_date(tmp_path: Path) -> None:
  """Two USD Advance GDP q/q rows on the same date produce one Timestamp."""
  csv = tmp_path / "cal.csv"
  csv.write_text(_GDP_CSV_CONTENT, encoding="utf-8")
  dates = load_gdp_release_dates(csv)
  # 2024-01-26 appears twice for Advance GDP q/q; dedup → still 3 unique dates
  assert len(dates) == 3


def test_load_gdp_returns_normalized_tz_naive(tmp_path: Path) -> None:
  """Returned Timestamps are tz-naive and normalized to midnight."""
  csv = tmp_path / "cal.csv"
  csv.write_text(_GDP_CSV_CONTENT, encoding="utf-8")
  dates = load_gdp_release_dates(csv)
  for ts in dates:
    assert ts.tzinfo is None
    assert ts == ts.normalize()


def test_load_gdp_case_insensitive_event(tmp_path: Path) -> None:
  """Event matching is case-insensitive and strips whitespace."""
  content = "date,currency,event\n2024-02-28, USD , ADVANCE GDP Q/Q \n"
  csv = tmp_path / "upper.csv"
  csv.write_text(content, encoding="utf-8")
  dates = load_gdp_release_dates(csv)
  assert pd.Timestamp("2024-02-28") in dates


def test_load_gdp_all_three_vintages_from_one_csv(tmp_path: Path) -> None:
  """Advance, Prelim, and Final GDP q/q are all matched from a single CSV."""
  content = (
    "date,currency,event\n"
    "2024-01-26,USD,Advance GDP q/q\n"
    "2024-04-25,USD,Prelim GDP q/q\n"
    "2024-07-25,USD,Final GDP q/q\n"
  )
  csv = tmp_path / "three.csv"
  csv.write_text(content, encoding="utf-8")
  dates = load_gdp_release_dates(csv)
  assert len(dates) == 3
  assert pd.Timestamp("2024-01-26") in dates
  assert pd.Timestamp("2024-04-25") in dates
  assert pd.Timestamp("2024-07-25") in dates


def test_load_gdp_empty_result_when_no_match(tmp_path: Path) -> None:
  """CSV with no matching rows returns an empty set."""
  content = "date,currency,event\n2024-01-12,USD,Core CPI m/m\n"
  csv = tmp_path / "nomatch.csv"
  csv.write_text(content, encoding="utf-8")
  dates = load_gdp_release_dates(csv)
  assert dates == set()


# ===========================================================================
# 9. Result shape invariants
# ===========================================================================

def test_result_timeframe_is_daily() -> None:
  """Results are stored under the 'daily' timeframe key."""
  result = _build_main_stat().compute(_build_main_candles())
  assert "daily" in result.instruments["NQ"]


def test_result_slices_empty() -> None:
  """EconomicDataVolume declares no slices; slices dict must be empty."""
  result = _build_main_stat().compute(_build_main_candles())
  assert result.instruments["NQ"]["daily"].slices == {}


def test_result_slices_empty_on_empty_input() -> None:
  """slices is {} even with empty candles (no crash on slice computation)."""
  result = _build_main_stat().compute(_empty_df())
  assert result.instruments["NQ"]["daily"].slices == {}


# ===========================================================================
# 10. Metadata — stat_name, i18n, and write_results round-trip
# ===========================================================================

def test_stat_name() -> None:
  """stat_name must equal 'economic_data_volume'."""
  result = _build_main_stat().compute(_empty_df())
  assert result.stat_name == "economic_data_volume"


def test_i18n_title_non_empty() -> None:
  result = _build_main_stat().compute(_empty_df())
  assert result.title.en != ""
  assert result.title.fr != ""


def test_i18n_definition_non_empty() -> None:
  result = _build_main_stat().compute(_empty_df())
  assert result.definition.en != ""
  assert result.definition.fr != ""


def test_i18n_labels_conditions_complete() -> None:
  """Labels carry all six condition keys."""
  result = _build_main_stat().compute(_empty_df())
  expected_conditions = {
    "cpi_day", "fomc_day", "nfp_day", "gdp_day", "any_event_day", "non_event_day",
  }
  assert set(result.labels.conditions) == expected_conditions


def test_i18n_labels_outcomes_complete() -> None:
  """Labels carry the 'mean_volume' outcome key."""
  result = _build_main_stat().compute(_empty_df())
  assert "mean_volume" in result.labels.outcomes


def test_i18n_labels_all_have_en_and_fr() -> None:
  """Every condition and outcome label has non-empty en and fr strings."""
  result = _build_main_stat().compute(_empty_df())
  for mapping in (result.labels.conditions, result.labels.outcomes):
    for key, i18n in mapping.items():
      assert i18n.en != "", f"{key}.en is empty"
      assert i18n.fr != "", f"{key}.fr is empty"


def test_write_results_round_trip(tmp_path: Path) -> None:
  """write_results produces economic_data_volume.json that re-validates correctly."""
  stat = _build_main_stat()
  result = stat.compute(_build_main_candles())
  written = write_results(result, results_dir=tmp_path)
  assert written.name == "economic_data_volume.json"

  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)
  assert validated.stat_name == "economic_data_volume"
  assert "NQ" in validated.instruments
  tf = validated.instruments["NQ"]["daily"]
  assert tf.total_samples == 4
  assert len(tf.results) == 6
  pairs = {(r.condition, r.outcome) for r in tf.results}
  assert pairs == {
    ("cpi_day", "mean_volume"),
    ("fomc_day", "mean_volume"),
    ("nfp_day", "mean_volume"),
    ("gdp_day", "mean_volume"),
    ("any_event_day", "mean_volume"),
    ("non_event_day", "mean_volume"),
  }


def test_write_results_french_accents_not_escaped(tmp_path: Path) -> None:
  """French labels are stored as UTF-8 literals, not \\uXXXX escape sequences."""
  result = _build_main_stat().compute(_empty_df())
  written = write_results(result, results_dir=tmp_path)
  raw = written.read_text(encoding="utf-8")
  assert "é" in raw
  assert "\\u00e9" not in raw


def test_compute_result_validates_as_stat_run_result() -> None:
  """compute() output re-validates cleanly as a StatRunResult."""
  result = _build_main_stat().compute(_build_main_candles())
  revalidated = StatRunResult.model_validate(result.model_dump())
  assert revalidated.stat_name == "economic_data_volume"
  rows = revalidated.instruments["NQ"]["daily"].results
  assert len(rows) == 6
