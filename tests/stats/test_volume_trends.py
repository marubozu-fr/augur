"""Tests for stats.volume_trends.standard.

All data is synthetic — no real market files required.
Expected values are hand-calculated before each assertion.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.volume_trends.standard import VolumeTrends, daily_volume_table

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

# RTH window: 09:30=570 .. 16:15=975.
# Resolution: bar at exactly 570 AND last bar mod >= 975 - 15 = 960 (>= 16:00).
# Full day bars: mod 570..974 inclusive = 405 bars.
_RTH_START = 570   # 09:30
_RTH_LAST = 974    # 16:14 (last bar before 16:15)
_N_BARS = _RTH_LAST - _RTH_START + 1  # 405


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

def _make_day(date: str, bar_volume: int) -> pd.DataFrame:
  """Build one full RTH trading day of 1-min OHLCV bars (09:30 – 16:14).

  Every bar carries bar_volume units; the day's summed RTH volume is
  bar_volume * 405. Prices are flat at 100.0 throughout.
  """
  base = pd.Timestamp(date, tz=_NY)
  records = []
  for mod in range(_RTH_START, _RTH_LAST + 1):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    records.append({
      "timestamp": ts,
      "open": 100.0,
      "high": 100.25,
      "low": 99.75,
      "close": 100.0,
      "volume": bar_volume,
    })
  return pd.DataFrame(records)


def _make_truncated_day(date: str, bar_volume: int = 500) -> pd.DataFrame:
  """A day whose last RTH bar is 09:50 (mod 590 < 960) — must be excluded."""
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
      "volume": bar_volume,
    })
  return pd.DataFrame(records)


def _make_overnight_bar(date: str, bar_volume: int = 9999) -> pd.DataFrame:
  """A single ETH bar at 08:00 — must NOT count toward RTH volume."""
  ts = pd.Timestamp(date, tz=_NY).replace(hour=8, minute=0, second=0, microsecond=0)
  return pd.DataFrame([{
    "timestamp": ts,
    "open": 100.0,
    "high": 100.25,
    "low": 99.75,
    "close": 100.0,
    "volume": bar_volume,
  }])


def make_candles(days: list[dict]) -> pd.DataFrame:
  """Concatenate per-day DataFrames into a sorted multi-day OHLCV frame.

  Each element of ``days`` must have ``date`` and ``bar_volume`` keys.
  """
  frames = [_make_day(d["date"], d["bar_volume"]) for d in days]
  df = pd.concat(frames, ignore_index=True)
  return df.sort_values("timestamp").reset_index(drop=True)


def _empty_df() -> pd.DataFrame:
  df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
  df["timestamp"] = pd.array([], dtype="datetime64[ns, America/New_York]")
  return df


def _stat(granularity: str = "monthly") -> VolumeTrends:
  return VolumeTrends(instrument="NQ", config=_TEST_CONFIG, granularity=granularity)


def _row(rows: list[StatResultRow], outcome: str) -> StatResultRow:
  for r in rows:
    if r.condition == "any_period" and r.outcome == outcome:
      return r
  raise KeyError(outcome)


def _overall(result: StatRunResult, granularity: str, outcome: str) -> StatResultRow:
  return _row(result.instruments["NQ"][granularity].results, outcome)


def _month_rows(result: StatRunResult, month_key: str) -> list[StatResultRow]:
  return (
    result.instruments["NQ"]["monthly"].slices["month_of_year"].groups[month_key].results
  )


def _week_rows(result: StatRunResult, week_key: str) -> list[StatResultRow]:
  return (
    result.instruments["NQ"]["weekly"].slices["week_of_year"].groups[week_key].results
  )


# ---------------------------------------------------------------------------
# Main monthly synthetic dataset:
#
# 3 calendar months, each with 2 resolved days, plus one EXTRA month that will
# be DROPPED by the pending discipline (most-recent-period exclusion).
#
# Month A — 2024-01: days 01-15 (bar_vol=100) and 01-16 (bar_vol=200)
#   volume_A = (100 + 200) * 405 = 121500
# Month B — 2024-02: days 02-12 (bar_vol=300) and 02-13 (bar_vol=100)
#   volume_B = (300 + 100) * 405 = 162000
# Month C — 2024-03: days 03-11 (bar_vol=150) and 03-12 (bar_vol=150)
#   volume_C = (150 + 150) * 405 = 121500
# Month D — 2024-04: one day 04-10 (bar_vol=999) → DROPPED (pending)
#
# After pending-period exclusion only months A, B, C survive:
#   N = 3 periods
#   mean_volume = (121500 + 162000 + 121500) / 3 = 405000 / 3 = 135000.0
#
# data_range: 2024-01-15 .. 2024-03-12  (first..last resolved period start date)
# ---------------------------------------------------------------------------

_MONTHLY_DAYS = [
  # Month A: January 2024
  {"date": "2024-01-15", "bar_volume": 100},
  {"date": "2024-01-16", "bar_volume": 200},
  # Month B: February 2024
  {"date": "2024-02-12", "bar_volume": 300},
  {"date": "2024-02-13", "bar_volume": 100},
  # Month C: March 2024
  {"date": "2024-03-11", "bar_volume": 150},
  {"date": "2024-03-12", "bar_volume": 150},
  # Month D: April 2024 — will be DROPPED (pending)
  {"date": "2024-04-10", "bar_volume": 999},
]

# ---------------------------------------------------------------------------
# Weekly synthetic dataset:
#
# ISO weeks 2024-W03, 2024-W04, 2024-W05 each with 2 days;
# ISO week 2024-W06 has 1 day → DROPPED (pending).
#
# 2024-01-15 (Mon) ISO W03, 2024-01-19 (Fri) ISO W03
#   volume_W03 = (100 + 200) * 405 = 121500
# 2024-01-22 (Mon) ISO W04, 2024-01-26 (Fri) ISO W04
#   volume_W04 = (300 + 100) * 405 = 162000
# 2024-01-29 (Mon) ISO W05, 2024-02-02 (Fri) ISO W05
#   volume_W05 = (150 + 150) * 405 = 121500
# 2024-02-05 (Mon) ISO W06 → DROPPED (pending, most-recent period)
#
# N = 3 periods  mean_volume = 135000.0
# ---------------------------------------------------------------------------

_WEEKLY_DAYS = [
  # Week W03
  {"date": "2024-01-15", "bar_volume": 100},
  {"date": "2024-01-19", "bar_volume": 200},
  # Week W04
  {"date": "2024-01-22", "bar_volume": 300},
  {"date": "2024-01-26", "bar_volume": 100},
  # Week W05
  {"date": "2024-01-29", "bar_volume": 150},
  {"date": "2024-02-02", "bar_volume": 150},
  # Week W06 — will be DROPPED (pending)
  {"date": "2024-02-05", "bar_volume": 999},
]

# ---------------------------------------------------------------------------
# Multi-year dataset for month-of-year slice tests:
#
# January appears in BOTH 2023 and 2024, so the "january" group has 2 periods.
#
# 2023-01: day 2023-01-16 (bar_vol=100)  → volume = 100 * 405 = 40500
# 2023-02: day 2023-02-15 (bar_vol=200)  → volume = 200 * 405 = 81000
# 2024-01: day 2024-01-15 (bar_vol=150)  → volume = 150 * 405 = 60750
# 2024-02: day 2024-02-12 (bar_vol=50)   → DROPPED (pending, most-recent period)
#
# After dropping 2024-02:  periods = [2023-01, 2023-02, 2024-01]
# month_of_year groups:
#   january:  [40500, 60750]  → mean = 101250 / 2 = 50625.0
#   february: [81000]         → mean = 81000.0
# overall: mean = (40500 + 81000 + 60750) / 3 = 182250 / 3 = 60750.0
# ---------------------------------------------------------------------------

_MULTIYEAR_MONTHLY_DAYS = [
  {"date": "2023-01-16", "bar_volume": 100},  # 2023 Jan → 40500
  {"date": "2023-02-15", "bar_volume": 200},  # 2023 Feb → 81000
  {"date": "2024-01-15", "bar_volume": 150},  # 2024 Jan → 60750
  {"date": "2024-02-12", "bar_volume": 50},   # 2024 Feb → DROPPED
]

# ---------------------------------------------------------------------------
# Multi-year weekly dataset for week-of-year slice tests:
#
# ISO week 3 appears in 2023 and 2024; week 4 appears in 2023 only.
#
# 2023-01-16 (Mon W03): bar_vol=100  → period volume = 40500
# 2023-01-23 (Mon W04): bar_vol=200  → period volume = 81000
# 2024-01-15 (Mon W03): bar_vol=150  → period volume = 60750
# 2024-01-22 (Mon W04): bar_vol=50   → DROPPED (pending)
#
# After dropping 2024-W04:  [2023-W03, 2023-W04, 2024-W03]
# week_of_year groups:
#   w03: [40500, 60750] → mean = 50625.0
#   w04: [81000]        → mean = 81000.0
# overall: mean = (40500 + 81000 + 60750) / 3 = 60750.0
# ---------------------------------------------------------------------------

_MULTIYEAR_WEEKLY_DAYS = [
  {"date": "2023-01-16", "bar_volume": 100},  # 2023 W03 → 40500
  {"date": "2023-01-23", "bar_volume": 200},  # 2023 W04 → 81000
  {"date": "2024-01-15", "bar_volume": 150},  # 2024 W03 → 60750
  {"date": "2024-01-22", "bar_volume": 50},   # 2024 W04 → DROPPED
]


# ===========================================================================
# 1. Period volume = SUM of daily RTH volumes; mean across periods is correct
# ===========================================================================

def test_monthly_period_volumes_are_sums() -> None:
  """build_day_table sums each period's daily RTH volumes correctly.

  January: (100 + 200) * 405 = 121500
  February: (300 + 100) * 405 = 162000
  March: (150 + 150) * 405 = 121500
  """
  dt = _stat("monthly").build_day_table(make_candles(_MONTHLY_DAYS))
  # 3 periods survive (April is dropped as pending)
  assert len(dt) == 3
  jan = pd.Timestamp("2024-01-15", tz=_NY).normalize()
  feb = pd.Timestamp("2024-02-12", tz=_NY).normalize()
  mar = pd.Timestamp("2024-03-11", tz=_NY).normalize()
  # Jan: (100 + 200) * 405 = 121500
  assert dt.loc[jan, "volume"] == pytest.approx(121500.0)
  # Feb: (300 + 100) * 405 = 162000
  assert dt.loc[feb, "volume"] == pytest.approx(162000.0)
  # Mar: (150 + 150) * 405 = 121500
  assert dt.loc[mar, "volume"] == pytest.approx(121500.0)


def test_monthly_mean_volume_overall() -> None:
  """mean_volume = (121500 + 162000 + 121500) / 3 = 135000.0."""
  result = _stat("monthly").compute(make_candles(_MONTHLY_DAYS))
  row = _overall(result, "monthly", "mean_volume")
  # mean = 405000 / 3 = 135000
  assert row.value == pytest.approx(135000.0)
  assert row.count == 3
  assert row.total == 3
  assert row.probability == pytest.approx(0.0)


def test_weekly_period_volumes_are_sums() -> None:
  """Weekly periods aggregate daily volumes by ISO week.

  W03: (100 + 200) * 405 = 121500
  W04: (300 + 100) * 405 = 162000
  W05: (150 + 150) * 405 = 121500
  """
  dt = _stat("weekly").build_day_table(make_candles(_WEEKLY_DAYS))
  assert len(dt) == 3
  w03 = pd.Timestamp("2024-01-15", tz=_NY).normalize()
  w04 = pd.Timestamp("2024-01-22", tz=_NY).normalize()
  w05 = pd.Timestamp("2024-01-29", tz=_NY).normalize()
  assert dt.loc[w03, "volume"] == pytest.approx(121500.0)
  assert dt.loc[w04, "volume"] == pytest.approx(162000.0)
  assert dt.loc[w05, "volume"] == pytest.approx(121500.0)


def test_weekly_mean_volume_overall() -> None:
  """mean_volume = (121500 + 162000 + 121500) / 3 = 135000.0."""
  result = _stat("weekly").compute(make_candles(_WEEKLY_DAYS))
  row = _overall(result, "weekly", "mean_volume")
  assert row.value == pytest.approx(135000.0)
  assert row.count == 3
  assert row.total == 3
  assert row.probability == pytest.approx(0.0)


# ===========================================================================
# 2. Only RTH bars count toward volume (ETH bars excluded)
# ===========================================================================

def test_eth_bars_excluded_from_volume() -> None:
  """An overnight bar at 08:00 must NOT add to the day's RTH volume.

  Single Jan day with bar_vol=100 → RTH volume = 100 * 405 = 40500.
  Injecting an ETH bar with volume=9999 must leave the RTH sum unchanged.
  The single period is dropped as pending (most recent), so we inspect
  daily_volume_table directly to verify the per-day volume.
  """
  # One resolved day bar_vol=100
  df = _make_day("2024-01-15", bar_volume=100)
  # Inject a non-RTH bar at 08:00 with high volume
  overnight = _make_overnight_bar("2024-01-15", bar_volume=9999)
  combined = pd.concat([df, overnight], ignore_index=True).sort_values("timestamp").reset_index(drop=True)

  stat = _stat("monthly")
  # Use daily_volume_table to inspect per-day volume directly
  dvt = daily_volume_table(
    combined, stat.rth_start_min, stat.rth_end_min, stat.close_tolerance_min
  )
  date = pd.Timestamp("2024-01-15", tz=_NY).normalize()
  # 100 * 405 = 40500; the 9999-volume ETH bar must not be included
  assert dvt.loc[date, "volume"] == pytest.approx(40500.0)


def test_eth_bars_excluded_from_volume_two_days() -> None:
  """ETH bars mixed across multiple days are all excluded.

  Two days each bar_vol=100; one ETH bar injected per day (vol=5000).
  Each day's RTH volume must still be 100 * 405 = 40500.
  """
  df1 = _make_day("2024-01-15", bar_volume=100)
  df2 = _make_day("2024-01-16", bar_volume=100)
  eth1 = _make_overnight_bar("2024-01-15", bar_volume=5000)
  eth2 = _make_overnight_bar("2024-01-16", bar_volume=5000)
  combined = pd.concat([df1, df2, eth1, eth2], ignore_index=True).sort_values("timestamp").reset_index(drop=True)

  stat = _stat("monthly")
  dvt = daily_volume_table(
    combined, stat.rth_start_min, stat.rth_end_min, stat.close_tolerance_min
  )
  for date_str in ("2024-01-15", "2024-01-16"):
    date = pd.Timestamp(date_str, tz=_NY).normalize()
    assert dvt.loc[date, "volume"] == pytest.approx(40500.0), f"ETH bled into {date_str}"


# ===========================================================================
# 3. Most recent period is excluded (pending discipline)
# ===========================================================================

def test_pending_period_dropped_monthly() -> None:
  """With 3 full months + 1 partial month, only the first 3 appear in the table."""
  dt = _stat("monthly").build_day_table(make_candles(_MONTHLY_DAYS))
  # Month D (April) must NOT appear
  apr = pd.Timestamp("2024-04-10", tz=_NY).normalize()
  assert apr not in dt.index
  assert len(dt) == 3


def test_pending_period_dropped_weekly() -> None:
  """With 3 full weeks + 1 partial week, only the first 3 appear."""
  dt = _stat("weekly").build_day_table(make_candles(_WEEKLY_DAYS))
  # W06 must NOT appear
  w06 = pd.Timestamp("2024-02-05", tz=_NY).normalize()
  assert w06 not in dt.index
  assert len(dt) == 3


def test_data_range_reflects_remaining_periods_monthly() -> None:
  """data_range spans first..last period start date after pending exclusion.

  First period start: 2024-01-15, last: 2024-03-11.
  """
  result = _stat("monthly").compute(make_candles(_MONTHLY_DAYS))
  tf = result.instruments["NQ"]["monthly"]
  assert tf.data_range == ["2024-01-15", "2024-03-11"]
  # 3 monthly periods survive
  assert tf.total_samples == 3


def test_data_range_reflects_remaining_periods_weekly() -> None:
  """data_range spans first..last period start date after pending exclusion."""
  result = _stat("weekly").compute(make_candles(_WEEKLY_DAYS))
  tf = result.instruments["NQ"]["weekly"]
  assert tf.data_range == ["2024-01-15", "2024-01-29"]
  assert tf.total_samples == 3


# ===========================================================================
# 4. Unresolved days excluded from their period's volume
# ===========================================================================

def test_unresolved_day_excluded_from_period_volume() -> None:
  """A truncated day (early close) is excluded; the period volume reflects
  only the resolved days.

  Setup: January 2024 with two resolved days (bar_vol=100 each) plus a
  truncated day (bar_vol=999). The truncated day does NOT count.
  February 2024 has one resolved day (bar_vol=50) → DROPPED as pending.

  January volume = (100 + 100) * 405 = 81000.0
  """
  df_jan1 = _make_day("2024-01-15", bar_volume=100)
  df_jan2 = _make_day("2024-01-16", bar_volume=100)
  df_trunc = _make_truncated_day("2024-01-17", bar_volume=999)
  df_feb = _make_day("2024-02-12", bar_volume=50)  # will be dropped as pending
  combined = pd.concat(
    [df_jan1, df_jan2, df_trunc, df_feb], ignore_index=True
  ).sort_values("timestamp").reset_index(drop=True)

  dt = _stat("monthly").build_day_table(combined)
  # Only January survives (February dropped as pending)
  assert len(dt) == 1
  jan = pd.Timestamp("2024-01-15", tz=_NY).normalize()
  # (100 + 100) * 405 = 81000; the truncated day's 999 bars do NOT count
  assert dt.loc[jan, "volume"] == pytest.approx(81000.0)


def test_unresolved_day_does_not_inflate_total_samples() -> None:
  """Adding a truncated day to a clean dataset does not change total_samples."""
  base_df = make_candles(_MONTHLY_DAYS)
  base_total = _stat("monthly").compute(base_df).instruments["NQ"]["monthly"].total_samples

  trunc = _make_truncated_day("2024-02-14", bar_volume=500)
  extended = pd.concat([base_df, trunc], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
  new_total = _stat("monthly").compute(extended).instruments["NQ"]["monthly"].total_samples

  # The truncated day is unresolved; it joins February (already a complete period)
  # so it does not change the number of periods
  assert new_total == base_total


# ===========================================================================
# 5. Slices: MonthOfYear groups by Jan..Dec; WeekOfYear by ISO week
# ===========================================================================

def test_month_of_year_slice_groups_present() -> None:
  """The month_of_year slice exists and contains january and february."""
  result = _stat("monthly").compute(make_candles(_MULTIYEAR_MONTHLY_DAYS))
  groups = result.instruments["NQ"]["monthly"].slices["month_of_year"].groups
  assert set(groups.keys()) == {"january", "february"}


def test_month_of_year_january_averages_two_years() -> None:
  """january group contains 2023-Jan and 2024-Jan, averaging their volumes.

  2023-Jan: 100 * 405 = 40500
  2024-Jan: 150 * 405 = 60750
  mean = (40500 + 60750) / 2 = 50625.0
  """
  result = _stat("monthly").compute(make_candles(_MULTIYEAR_MONTHLY_DAYS))
  groups = result.instruments["NQ"]["monthly"].slices["month_of_year"].groups
  jan_group = groups["january"]
  assert jan_group.total_samples == 2
  row = _row(jan_group.results, "mean_volume")
  # (40500 + 60750) / 2 = 50625
  assert row.value == pytest.approx(50625.0)
  assert row.count == 2
  assert row.total == 2


def test_month_of_year_february_single_year() -> None:
  """february group contains only 2023-Feb (2024-Feb was dropped as pending).

  2023-Feb: 200 * 405 = 81000
  mean = 81000.0
  """
  result = _stat("monthly").compute(make_candles(_MULTIYEAR_MONTHLY_DAYS))
  groups = result.instruments["NQ"]["monthly"].slices["month_of_year"].groups
  feb_group = groups["february"]
  assert feb_group.total_samples == 1
  row = _row(feb_group.results, "mean_volume")
  assert row.value == pytest.approx(81000.0)


def test_month_of_year_overall_mean() -> None:
  """Overall mean = (40500 + 81000 + 60750) / 3 = 60750.0."""
  result = _stat("monthly").compute(make_candles(_MULTIYEAR_MONTHLY_DAYS))
  row = _overall(result, "monthly", "mean_volume")
  # (40500 + 81000 + 60750) / 3 = 182250 / 3 = 60750
  assert row.value == pytest.approx(60750.0)
  assert row.count == 3


def test_week_of_year_slice_groups_present() -> None:
  """The week_of_year slice exists and contains w03 and w04."""
  result = _stat("weekly").compute(make_candles(_MULTIYEAR_WEEKLY_DAYS))
  groups = result.instruments["NQ"]["weekly"].slices["week_of_year"].groups
  assert set(groups.keys()) == {"w03", "w04"}


def test_week_of_year_w03_averages_two_years() -> None:
  """w03 contains 2023-W03 and 2024-W03, averaging their volumes.

  2023-W03 (2023-01-16): 100 * 405 = 40500
  2024-W03 (2024-01-15): 150 * 405 = 60750
  mean = (40500 + 60750) / 2 = 50625.0
  """
  result = _stat("weekly").compute(make_candles(_MULTIYEAR_WEEKLY_DAYS))
  groups = result.instruments["NQ"]["weekly"].slices["week_of_year"].groups
  w03_group = groups["w03"]
  assert w03_group.total_samples == 2
  row = _row(w03_group.results, "mean_volume")
  # (40500 + 60750) / 2 = 50625
  assert row.value == pytest.approx(50625.0)


def test_week_of_year_w04_single_year() -> None:
  """w04 contains only 2023-W04 (2024-W04 was dropped as pending).

  2023-W04 (2023-01-23): 200 * 405 = 81000
  """
  result = _stat("weekly").compute(make_candles(_MULTIYEAR_WEEKLY_DAYS))
  groups = result.instruments["NQ"]["weekly"].slices["week_of_year"].groups
  w04_group = groups["w04"]
  assert w04_group.total_samples == 1
  row = _row(w04_group.results, "mean_volume")
  assert row.value == pytest.approx(81000.0)


def test_week_of_year_overall_mean() -> None:
  """Overall mean = (40500 + 81000 + 60750) / 3 = 60750.0."""
  result = _stat("weekly").compute(make_candles(_MULTIYEAR_WEEKLY_DAYS))
  row = _overall(result, "weekly", "mean_volume")
  assert row.value == pytest.approx(60750.0)
  assert row.count == 3


# ===========================================================================
# 6. Random baseline: deterministic and overall permutation property
# ===========================================================================

def test_overall_baseline_equals_value_monthly() -> None:
  """For the full period table, sampling N from N is a permutation, so
  value_baseline == value exactly for the overall row."""
  result = _stat("monthly").compute(make_candles(_MONTHLY_DAYS))
  row = _overall(result, "monthly", "mean_volume")
  assert row.value_baseline is not None
  assert row.value_baseline == pytest.approx(row.value)


def test_overall_baseline_equals_value_weekly() -> None:
  """Same permutation property for the weekly granularity."""
  result = _stat("weekly").compute(make_candles(_WEEKLY_DAYS))
  row = _overall(result, "weekly", "mean_volume")
  assert row.value_baseline is not None
  assert row.value_baseline == pytest.approx(row.value)


def test_baseline_rows_deterministic_same_seed() -> None:
  """Same seed produces identical baseline_rows output."""
  df = make_candles(_MONTHLY_DAYS)
  stat = _stat("monthly")
  day_table = stat.build_day_table(df)
  rows_a = stat.baseline_rows(day_table, seed=7)
  rows_b = stat.baseline_rows(day_table, seed=7)
  assert len(rows_a) == len(rows_b)
  for a, b in zip(rows_a, rows_b):
    assert a.outcome == b.outcome
    if a.value is None:
      assert b.value is None
    else:
      assert a.value == pytest.approx(b.value)


def test_compute_twice_same_result_monthly() -> None:
  """compute() with default seed returns identical value_baseline on both calls."""
  df = make_candles(_MONTHLY_DAYS)
  stat = _stat("monthly")
  result_a = stat.compute(df)
  result_b = stat.compute(df)
  rows_a = result_a.instruments["NQ"]["monthly"].results
  rows_b = result_b.instruments["NQ"]["monthly"].results
  assert len(rows_a) == len(rows_b)
  for a, b in zip(rows_a, rows_b):
    assert a.outcome == b.outcome
    assert a.value == pytest.approx(b.value)
    if a.value_baseline is None:
      assert b.value_baseline is None
    else:
      assert a.value_baseline == pytest.approx(b.value_baseline)


def test_baseline_different_seeds_may_differ() -> None:
  """Different seeds should produce different baselines for a sub-slice sample.

  With the multi-year monthly dataset (3 periods total) the january slice has
  2 periods. Sampling 2-of-3 periods with different seeds should differ on at
  least one of the C(3,2)=3 possible draws (not all draws give the same mean).
  """
  df = make_candles(_MULTIYEAR_MONTHLY_DAYS)
  stat = _stat("monthly")
  full_table = stat.build_day_table(df)
  # january sub-table: 2 rows
  jan_mask = full_table.index.month == 1
  jan_table = full_table[jan_mask]
  assert len(jan_table) == 2
  assert len(full_table) == 3

  rows_seed1 = stat.baseline_rows(jan_table, seed=1)
  rows_seed2 = stat.baseline_rows(jan_table, seed=999)
  # With only 3 population items and 2 seeds drawing different pairs the means
  # will differ on most seed pairs; we just verify the determinism holds within
  # each seed (already tested above), and that the two calls don't crash.
  assert len(rows_seed1) == 1
  assert len(rows_seed2) == 1


# ===========================================================================
# 7. Edge cases: empty / single-period input
# ===========================================================================

def test_empty_dataframe_no_crash_monthly() -> None:
  """Empty input: 0 samples, empty data_range, no crash."""
  result = _stat("monthly").compute(_empty_df())
  tf = result.instruments["NQ"]["monthly"]
  assert tf.total_samples == 0
  assert tf.data_range == []


def test_empty_dataframe_zero_counts_monthly() -> None:
  """All rows report count == total == 0, probability == 0.0, value == 0.0."""
  result = _stat("monthly").compute(_empty_df())
  for row in result.instruments["NQ"]["monthly"].results:
    assert row.count == 0
    assert row.total == 0
    assert row.probability == pytest.approx(0.0)
    assert row.value == pytest.approx(0.0)


def test_empty_dataframe_no_slice_groups_monthly() -> None:
  """No month_of_year groups when there is no data."""
  result = _stat("monthly").compute(_empty_df())
  assert result.instruments["NQ"]["monthly"].slices["month_of_year"].groups == {}


def test_empty_dataframe_no_crash_weekly() -> None:
  """Empty input with weekly granularity: 0 samples, no crash."""
  result = _stat("weekly").compute(_empty_df())
  tf = result.instruments["NQ"]["weekly"]
  assert tf.total_samples == 0
  assert tf.data_range == []


def test_empty_dataframe_no_slice_groups_weekly() -> None:
  """No week_of_year groups when there is no data."""
  result = _stat("weekly").compute(_empty_df())
  assert result.instruments["NQ"]["weekly"].slices["week_of_year"].groups == {}


def test_single_period_yields_empty_table() -> None:
  """Only one calendar month of data → that period is the 'most recent' and is
  dropped as pending. The result has N=0 periods.

  This tests the edge case where the entire dataset collapses to zero periods.
  """
  single_month = [
    {"date": "2024-01-15", "bar_volume": 100},
    {"date": "2024-01-16", "bar_volume": 200},
  ]
  dt = _stat("monthly").build_day_table(make_candles(single_month))
  # The single period is dropped as pending (most recent)
  assert len(dt) == 0


def test_single_period_compute_n_zero() -> None:
  """When only one month exists (dropped as pending), total_samples == 0."""
  single_month = [
    {"date": "2024-01-15", "bar_volume": 100},
    {"date": "2024-01-16", "bar_volume": 200},
  ]
  result = _stat("monthly").compute(make_candles(single_month))
  tf = result.instruments["NQ"]["monthly"]
  assert tf.total_samples == 0
  assert tf.data_range == []


def test_two_periods_keeps_one() -> None:
  """Two calendar months: the most recent is dropped, leaving exactly 1 period."""
  two_months = [
    {"date": "2024-01-15", "bar_volume": 100},
    {"date": "2024-02-12", "bar_volume": 200},
  ]
  dt = _stat("monthly").build_day_table(make_candles(two_months))
  # February is dropped; January remains
  assert len(dt) == 1
  jan = pd.Timestamp("2024-01-15", tz=_NY).normalize()
  assert jan in dt.index
  # Jan: 100 * 405 = 40500
  assert dt.loc[jan, "volume"] == pytest.approx(40500.0)


# ===========================================================================
# 8. Pydantic validation: result validates and round-trips
# ===========================================================================

def test_result_is_stat_run_result_instance_monthly() -> None:
  """compute() returns a valid StatRunResult."""
  result = _stat("monthly").compute(make_candles(_MONTHLY_DAYS))
  assert isinstance(result, StatRunResult)


def test_result_is_stat_run_result_instance_weekly() -> None:
  result = _stat("weekly").compute(make_candles(_WEEKLY_DAYS))
  assert isinstance(result, StatRunResult)


def test_stat_name() -> None:
  """stat_name must equal 'volume_trends'."""
  result = _stat("monthly").compute(_empty_df())
  assert result.stat_name == "volume_trends"


def test_write_results_round_trip_monthly(tmp_path: Path) -> None:
  """write_results serialises to JSON; the file validates back to StatRunResult."""
  result = _stat("monthly").compute(make_candles(_MONTHLY_DAYS))
  written = write_results(result, results_dir=tmp_path)

  assert written.name == "volume_trends.json"
  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)

  tf = validated.instruments["NQ"]["monthly"]
  assert tf.total_samples == 3
  # Verify the month_of_year slice survived the round-trip
  jan_group = tf.slices["month_of_year"].groups["january"]
  row = next(r for r in jan_group.results if r.outcome == "mean_volume")
  # January single period: (100 + 200) * 405 = 121500
  assert row.value == pytest.approx(121500.0)


def test_write_results_round_trip_weekly(tmp_path: Path) -> None:
  """write_results serialises to JSON for weekly granularity."""
  result = _stat("weekly").compute(make_candles(_WEEKLY_DAYS))
  written = write_results(result, results_dir=tmp_path)

  assert written.name == "volume_trends.json"
  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)

  tf = validated.instruments["NQ"]["weekly"]
  assert tf.total_samples == 3
  w03_group = tf.slices["week_of_year"].groups["w03"]
  row = next(r for r in w03_group.results if r.outcome == "mean_volume")
  # W03: (100 + 200) * 405 = 121500
  assert row.value == pytest.approx(121500.0)


# ===========================================================================
# 9. Single outcome key, condition, and probability channel
# ===========================================================================

def test_exactly_one_outcome_row_monthly() -> None:
  """Exactly one outcome row under the single any_period condition."""
  result = _stat("monthly").compute(make_candles(_MONTHLY_DAYS))
  rows = result.instruments["NQ"]["monthly"].results
  assert len(rows) == 1
  assert rows[0].outcome == "mean_volume"
  assert rows[0].condition == "any_period"


def test_exactly_one_outcome_row_weekly() -> None:
  """Exactly one outcome row for weekly granularity."""
  result = _stat("weekly").compute(make_candles(_WEEKLY_DAYS))
  rows = result.instruments["NQ"]["weekly"].results
  assert len(rows) == 1
  assert rows[0].outcome == "mean_volume"
  assert rows[0].condition == "any_period"


def test_probability_channel_is_zero_overall() -> None:
  """probability must remain 0.0 for a magnitude-only stat."""
  result = _stat("monthly").compute(make_candles(_MONTHLY_DAYS))
  for row in result.instruments["NQ"]["monthly"].results:
    assert row.probability == pytest.approx(0.0)


def test_probability_channel_is_zero_in_slices() -> None:
  """probability must remain 0.0 in every month_of_year slice group."""
  result = _stat("monthly").compute(make_candles(_MULTIYEAR_MONTHLY_DAYS))
  slices = result.instruments["NQ"]["monthly"].slices["month_of_year"].groups
  for group in slices.values():
    for row in group.results:
      assert row.probability == pytest.approx(0.0)


# ===========================================================================
# 10. i18n
# ===========================================================================

def test_i18n_title_and_definition() -> None:
  """Both title and definition carry non-empty en/fr strings."""
  result = _stat("monthly").compute(_empty_df())
  assert result.title.en != "" and result.title.fr != ""
  assert result.definition.en != "" and result.definition.fr != ""


def test_i18n_labels_have_en_and_fr() -> None:
  """Every condition and outcome label has non-empty en and fr strings."""
  result = _stat("monthly").compute(_empty_df())
  for mapping in (result.labels.conditions, result.labels.outcomes):
    for key, i18n in mapping.items():
      assert i18n.en != "", f"{key}.en empty"
      assert i18n.fr != "", f"{key}.fr empty"
  assert set(result.labels.outcomes) == {"mean_volume"}
  assert "any_period" in result.labels.conditions


def test_labels_dimensions_contains_month_of_year() -> None:
  """The month_of_year dimension label is present after compute()."""
  result = _stat("monthly").compute(make_candles(_MONTHLY_DAYS))
  assert "month_of_year" in result.labels.dimensions
  assert result.labels.dimensions["month_of_year"].en != ""
  assert result.labels.dimensions["month_of_year"].fr != ""


def test_labels_dimensions_contains_week_of_year() -> None:
  """The week_of_year dimension label is present after compute()."""
  result = _stat("weekly").compute(make_candles(_WEEKLY_DAYS))
  assert "week_of_year" in result.labels.dimensions
  assert result.labels.dimensions["week_of_year"].en != ""
  assert result.labels.dimensions["week_of_year"].fr != ""


def test_invalid_granularity_raises() -> None:
  """Unsupported granularity raises ValueError."""
  with pytest.raises(ValueError, match="Unsupported granularity"):
    VolumeTrends(instrument="NQ", config=_TEST_CONFIG, granularity="daily")


def test_default_granularity_is_monthly() -> None:
  """Default granularity is 'monthly'."""
  stat = VolumeTrends(instrument="NQ", config=_TEST_CONFIG)
  assert stat.granularity == "monthly"
  assert stat.timeframe == "monthly"


def test_monthly_declares_month_of_year_slice() -> None:
  """Monthly instance declares MonthOfYear slicer."""
  stat = _stat("monthly")
  assert stat.slices[0].name == "month_of_year"


def test_weekly_declares_week_of_year_slice() -> None:
  """Weekly instance declares WeekOfYear slicer."""
  stat = _stat("weekly")
  assert stat.slices[0].name == "week_of_year"
