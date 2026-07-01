"""Tests for stats.session_volume_weekday.standard.

All data is synthetic — no real market files required. Expected values are
hand-calculated before each assertion.

Three geographic sessions under test:
  - asia:   18:00 prev day → 03:00 next day (cross-midnight, duration=540 min)
  - london: 03:00 → 09:30 (intraday, duration=390 min)
  - ny:     09:30 → 16:15 (intraday, duration=405 min)

Resolution requires:
  - Clean open bar at offset 0 from session start
  - Last bar offset >= duration - 15 (close_tolerance_min=15)
    Asia:   offset >= 525  (bar at or after 02:45 on cycle date)
    London: offset >= 375  (bar at or after 09:15 on cycle date)
    NY:     offset >= 390  (bar at or after 16:00 on cycle date)

A cycle is countable only when ALL three sessions are resolved; an inner join
enforces the pending-sample discipline uniformly across all three outcomes.

Volume per bar uses a [40%, 30%, 20%, 10%] split of the session's total volume,
making each bar's contribution distinct and the sum non-trivially verifiable.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.session_volume_weekday.standard import SessionVolumeByWeekday

# ---------------------------------------------------------------------------
# Minimal InstrumentConfig — does not depend on NQ.yaml
# ---------------------------------------------------------------------------
_NY = "America/New_York"
_TEST_CONFIG = InstrumentConfig(
  instrument="NQ",
  working_timezone=_NY,
  sessions={
    "asia": Session(start="18:00", end="03:00"),
    "london": Session(start="03:00", end="09:30"),
    "ny": Session(start="09:30", end="16:15"),
  },
  timeframes=["daily"],
  parquet_path=Path("data/NQ_1min.parquet"),
)


# ---------------------------------------------------------------------------
# Bar and session builders
# ---------------------------------------------------------------------------

def _bar(ts: pd.Timestamp, vol: int = 100) -> dict:
  """One OHLCV bar with hardcoded OHLC and a given volume."""
  return {
    "timestamp": ts,
    "open": 100.0,
    "high": 100.25,
    "low": 99.75,
    "close": 100.0,
    "volume": vol,
  }


def _split_vol(total_vol: int) -> tuple[int, int, int, int]:
  """Split total_vol across 4 bars as [40%, 30%, 20%, 10%].

  Each bar gets a distinct share; the fourth bar absorbs any rounding
  remainder so bar1+bar2+bar3+bar4 == total_vol exactly.
  """
  v0 = int(total_vol * 0.4)
  v1 = int(total_vol * 0.3)
  v2 = int(total_vol * 0.2)
  v3 = total_vol - v0 - v1 - v2
  return v0, v1, v2, v3


def _asia_bars(cycle_date: str, total_vol: int) -> list[dict]:
  """Asia session bars attributed to ``cycle_date`` (cross-midnight session).

  Evening bars live on the prior calendar day (18:00–23:59); early bars live on
  cycle_date (00:00–02:59). session_bars() maps both sets to cycle_date via the
  cross-midnight attribution rule (mod >= start_min → _cycle = _date + 1 day).

  Offsets (start_min=1080):
    18:00 prev → offset (1080-1080)%1440 = 0          clean open ✓
    20:00 prev → offset (1200-1080)%1440 = 120
    01:00 day  → offset (60-1080)%1440  = 420
    02:59 day  → offset (179-1080)%1440 = 539 >= 525  coverage ✓

  Volume distributed [40%, 30%, 20%, 10%] of total_vol.
  """
  base = pd.Timestamp(cycle_date, tz=_NY)
  prev = base - pd.Timedelta(days=1)
  v0, v1, v2, v3 = _split_vol(total_vol)
  return [
    _bar(prev.replace(hour=18, minute=0), v0),   # clean open, offset=0
    _bar(prev.replace(hour=20, minute=0), v1),   # offset=120
    _bar(base.replace(hour=1, minute=0), v2),    # offset=420
    _bar(base.replace(hour=2, minute=59), v3),   # coverage, offset=539
  ]


def _london_bars(cycle_date: str, total_vol: int) -> list[dict]:
  """London session bars on ``cycle_date`` (intraday, start_min=180).

  Offsets:
    03:00 → offset 0           clean open ✓
    05:00 → offset 120
    07:00 → offset 240
    09:29 → offset 389 >= 375  coverage ✓

  Volume distributed [40%, 30%, 20%, 10%] of total_vol.
  """
  base = pd.Timestamp(cycle_date, tz=_NY)
  v0, v1, v2, v3 = _split_vol(total_vol)
  return [
    _bar(base.replace(hour=3, minute=0), v0),    # clean open, offset=0
    _bar(base.replace(hour=5, minute=0), v1),    # offset=120
    _bar(base.replace(hour=7, minute=0), v2),    # offset=240
    _bar(base.replace(hour=9, minute=29), v3),   # coverage, offset=389
  ]


def _ny_bars(cycle_date: str, total_vol: int) -> list[dict]:
  """NY session bars on ``cycle_date`` (intraday, start_min=570).

  Offsets:
    09:30 → offset 0           clean open ✓
    12:00 → offset 150
    13:00 → offset 210
    16:14 → offset 404 >= 390  coverage ✓

  Volume distributed [40%, 30%, 20%, 10%] of total_vol.
  """
  base = pd.Timestamp(cycle_date, tz=_NY)
  v0, v1, v2, v3 = _split_vol(total_vol)
  return [
    _bar(base.replace(hour=9, minute=30), v0),   # clean open, offset=0
    _bar(base.replace(hour=12, minute=0), v1),   # offset=150
    _bar(base.replace(hour=13, minute=0), v2),   # offset=210
    _bar(base.replace(hour=16, minute=14), v3),  # coverage, offset=404
  ]


def _full_cycle(
  cycle_date: str,
  asia_vol: int,
  london_vol: int,
  ny_vol: int,
) -> list[dict]:
  """One fully-resolved cycle across all three sessions."""
  return (
    _asia_bars(cycle_date, asia_vol)
    + _london_bars(cycle_date, london_vol)
    + _ny_bars(cycle_date, ny_vol)
  )


def make_candles(cycles: list[list[dict]]) -> pd.DataFrame:
  """Concatenate cycle bar lists into a sorted OHLCV DataFrame."""
  records = [r for cycle in cycles for r in cycle]
  df = pd.DataFrame(records)
  return df.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stat factory and result accessors
# ---------------------------------------------------------------------------

def _stat() -> SessionVolumeByWeekday:
  return SessionVolumeByWeekday(instrument="NQ", config=_TEST_CONFIG)


def _row(rows: list[StatResultRow], outcome: str) -> StatResultRow:
  for r in rows:
    if r.condition == "any_day" and r.outcome == outcome:
      return r
  raise KeyError(outcome)


def _overall(result: StatRunResult, outcome: str) -> StatResultRow:
  return _row(result.instruments["NQ"]["daily"].results, outcome)


def _weekday_rows(result: StatRunResult, day_key: str) -> list[StatResultRow]:
  return result.instruments["NQ"]["daily"].slices["weekday"].groups[day_key].results


def _empty_df() -> pd.DataFrame:
  df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
  df["timestamp"] = pd.array([], dtype="datetime64[ns, America/New_York]")
  return df


# ---------------------------------------------------------------------------
# Main synthetic dataset — 10 fully-resolved cycles, 2 per weekday.
#
# Volumes follow k * {1000, 2000, 3000} for cycle k=1..10, distributed
# [40%, 30%, 20%, 10%] across the 4 bars of each session.
#
# Dates and weekdays (2024 calendar):
#   2024-01-02 Tuesday   k=1:  asia=1000,  london=2000,  ny=3000
#   2024-01-03 Wednesday k=2:  asia=2000,  london=4000,  ny=6000
#   2024-01-04 Thursday  k=3:  asia=3000,  london=6000,  ny=9000
#   2024-01-05 Friday    k=4:  asia=4000,  london=8000,  ny=12000
#   2024-01-08 Monday    k=5:  asia=5000,  london=10000, ny=15000
#   2024-01-09 Tuesday   k=6:  asia=6000,  london=12000, ny=18000
#   2024-01-10 Wednesday k=7:  asia=7000,  london=14000, ny=21000
#   2024-01-11 Thursday  k=8:  asia=8000,  london=16000, ny=24000
#   2024-01-12 Friday    k=9:  asia=9000,  london=18000, ny=27000
#   2024-01-15 Monday    k=10: asia=10000, london=20000, ny=30000
#
# Overall (10 cycles):
#   mean_asia_volume   = (1+2+...+10)*1000 / 10 = 55000/10 = 5500
#   mean_london_volume = (1+2+...+10)*2000 / 10 = 110000/10 = 11000
#   mean_ny_volume     = (1+2+...+10)*3000 / 10 = 165000/10 = 16500
#
# Per-weekday means (2 cycles each):
#   Tuesday   (k=1, k=6):  asia=(1000+6000)/2=3500   london=(2000+12000)/2=7000   ny=(3000+18000)/2=10500
#   Wednesday (k=2, k=7):  asia=(2000+7000)/2=4500   london=(4000+14000)/2=9000   ny=(6000+21000)/2=13500
#   Thursday  (k=3, k=8):  asia=(3000+8000)/2=5500   london=(6000+16000)/2=11000  ny=(9000+24000)/2=16500
#   Friday    (k=4, k=9):  asia=(4000+9000)/2=6500   london=(8000+18000)/2=13000  ny=(12000+27000)/2=19500
#   Monday    (k=5, k=10): asia=(5000+10000)/2=7500  london=(10000+20000)/2=15000 ny=(15000+30000)/2=22500
# ---------------------------------------------------------------------------

_CYCLES: list[tuple] = [
  # (date,          asia_vol, london_vol, ny_vol)
  ("2024-01-02",   1000,   2000,   3000),  # Tue  k=1
  ("2024-01-03",   2000,   4000,   6000),  # Wed  k=2
  ("2024-01-04",   3000,   6000,   9000),  # Thu  k=3
  ("2024-01-05",   4000,   8000,  12000),  # Fri  k=4
  ("2024-01-08",   5000,  10000,  15000),  # Mon  k=5
  ("2024-01-09",   6000,  12000,  18000),  # Tue  k=6
  ("2024-01-10",   7000,  14000,  21000),  # Wed  k=7
  ("2024-01-11",   8000,  16000,  24000),  # Thu  k=8
  ("2024-01-12",   9000,  18000,  27000),  # Fri  k=9
  ("2024-01-15",  10000,  20000,  30000),  # Mon  k=10
]


def _make_all_cycles() -> pd.DataFrame:
  return make_candles([_full_cycle(*c) for c in _CYCLES])


# ===========================================================================
# 1. Known means: overall magnitudes
# ===========================================================================

def test_overall_three_outcome_rows() -> None:
  """Exactly three outcome rows under the single any_day condition."""
  result = _stat().compute(_make_all_cycles())
  rows = result.instruments["NQ"]["daily"].results
  assert len(rows) == 3
  assert {r.outcome for r in rows} == {
    "mean_asia_volume", "mean_london_volume", "mean_ny_volume"
  }
  assert all(r.condition == "any_day" for r in rows)


def test_overall_mean_asia_volume() -> None:
  """mean_asia_volume = (1+2+...+10)*1000 / 10 = 5500."""
  result = _stat().compute(_make_all_cycles())
  row = _overall(result, "mean_asia_volume")
  # asia volumes: 1000+2000+...+10000 = 55000; mean = 5500
  assert row.value == pytest.approx(5500.0)
  assert row.count == 10
  assert row.total == 10
  assert row.probability == pytest.approx(0.0)


def test_overall_mean_london_volume() -> None:
  """mean_london_volume = (1+2+...+10)*2000 / 10 = 11000."""
  result = _stat().compute(_make_all_cycles())
  row = _overall(result, "mean_london_volume")
  # london volumes: 2000+4000+...+20000 = 110000; mean = 11000
  assert row.value == pytest.approx(11000.0)
  assert row.count == 10
  assert row.total == 10
  assert row.probability == pytest.approx(0.0)


def test_overall_mean_ny_volume() -> None:
  """mean_ny_volume = (1+2+...+10)*3000 / 10 = 16500."""
  result = _stat().compute(_make_all_cycles())
  row = _overall(result, "mean_ny_volume")
  # ny volumes: 3000+6000+...+30000 = 165000; mean = 16500
  assert row.value == pytest.approx(16500.0)
  assert row.count == 10
  assert row.total == 10
  assert row.probability == pytest.approx(0.0)


def test_total_samples() -> None:
  """total_samples equals the number of fully-resolved cycles."""
  result = _stat().compute(_make_all_cycles())
  assert result.instruments["NQ"]["daily"].total_samples == 10


def test_n_uniform_across_outcomes() -> None:
  """Inner join enforces uniform N: all three outcomes share the same count/total."""
  result = _stat().compute(_make_all_cycles())
  rows = result.instruments["NQ"]["daily"].results
  counts = {r.count for r in rows}
  totals = {r.total for r in rows}
  assert counts == {10}
  assert totals == {10}


def test_data_range_spans_first_to_last_cycle() -> None:
  """data_range = [first_cycle_date, last_cycle_date]."""
  result = _stat().compute(_make_all_cycles())
  assert result.instruments["NQ"]["daily"].data_range == ["2024-01-02", "2024-01-15"]


# ===========================================================================
# 2. Weekday slice: groups, per-weekday means, sample sizes
# ===========================================================================

def test_weekday_slice_five_groups() -> None:
  """Five weekday groups are present — one per trading weekday."""
  result = _stat().compute(_make_all_cycles())
  groups = result.instruments["NQ"]["daily"].slices["weekday"].groups
  assert set(groups.keys()) == {"monday", "tuesday", "wednesday", "thursday", "friday"}


def test_weekday_sample_sizes() -> None:
  """Each weekday group contains exactly 2 resolved cycles."""
  result = _stat().compute(_make_all_cycles())
  groups = result.instruments["NQ"]["daily"].slices["weekday"].groups
  for day_key, group in groups.items():
    assert group.total_samples == 2, f"{day_key}: expected 2, got {group.total_samples}"


def test_weekday_mean_asia_volumes() -> None:
  """Per-weekday mean Asia volumes match hand-calculated values."""
  result = _stat().compute(_make_all_cycles())
  # Tuesday   (k=1, k=6):  (1000+6000)/2  = 3500
  # Wednesday (k=2, k=7):  (2000+7000)/2  = 4500
  # Thursday  (k=3, k=8):  (3000+8000)/2  = 5500
  # Friday    (k=4, k=9):  (4000+9000)/2  = 6500
  # Monday    (k=5, k=10): (5000+10000)/2 = 7500
  expected = {
    "tuesday": 3500.0,
    "wednesday": 4500.0,
    "thursday": 5500.0,
    "friday": 6500.0,
    "monday": 7500.0,
  }
  for day_key, exp in expected.items():
    row = _row(_weekday_rows(result, day_key), "mean_asia_volume")
    assert row.value == pytest.approx(exp), f"{day_key}: expected {exp}, got {row.value}"


def test_weekday_mean_london_volumes() -> None:
  """Per-weekday mean London volumes match hand-calculated values."""
  result = _stat().compute(_make_all_cycles())
  # Tuesday   (k=1, k=6):  (2000+12000)/2  = 7000
  # Wednesday (k=2, k=7):  (4000+14000)/2  = 9000
  # Thursday  (k=3, k=8):  (6000+16000)/2  = 11000
  # Friday    (k=4, k=9):  (8000+18000)/2  = 13000
  # Monday    (k=5, k=10): (10000+20000)/2 = 15000
  expected = {
    "tuesday": 7000.0,
    "wednesday": 9000.0,
    "thursday": 11000.0,
    "friday": 13000.0,
    "monday": 15000.0,
  }
  for day_key, exp in expected.items():
    row = _row(_weekday_rows(result, day_key), "mean_london_volume")
    assert row.value == pytest.approx(exp), f"{day_key}: expected {exp}, got {row.value}"


def test_weekday_mean_ny_volumes() -> None:
  """Per-weekday mean NY volumes match hand-calculated values."""
  result = _stat().compute(_make_all_cycles())
  # Tuesday   (k=1, k=6):  (3000+18000)/2  = 10500
  # Wednesday (k=2, k=7):  (6000+21000)/2  = 13500
  # Thursday  (k=3, k=8):  (9000+24000)/2  = 16500
  # Friday    (k=4, k=9):  (12000+27000)/2 = 19500
  # Monday    (k=5, k=10): (15000+30000)/2 = 22500
  expected = {
    "tuesday": 10500.0,
    "wednesday": 13500.0,
    "thursday": 16500.0,
    "friday": 19500.0,
    "monday": 22500.0,
  }
  for day_key, exp in expected.items():
    row = _row(_weekday_rows(result, day_key), "mean_ny_volume")
    assert row.value == pytest.approx(exp), f"{day_key}: expected {exp}, got {row.value}"


def test_weekday_count_total_probability_uniform() -> None:
  """Within each weekday group, count==total==2 and probability==0.0 for all outcomes."""
  result = _stat().compute(_make_all_cycles())
  for day_key in ("monday", "tuesday", "wednesday", "thursday", "friday"):
    for outcome in ("mean_asia_volume", "mean_london_volume", "mean_ny_volume"):
      row = _row(_weekday_rows(result, day_key), outcome)
      assert row.count == 2, f"{day_key}/{outcome}: count"
      assert row.total == 2, f"{day_key}/{outcome}: total"
      assert row.probability == pytest.approx(0.0), f"{day_key}/{outcome}: probability"


# ===========================================================================
# 3. Resolution / pending discipline
# ===========================================================================

def test_truncated_ny_excludes_entire_cycle() -> None:
  """Cycle with NY ending at 09:50 (offset 20 < 390) is dropped via inner join.

  Even though asia and london are fully resolved, the missing ny resolution
  removes the cycle from all three outcomes. N drops from 2 to 1.
  """
  good = _full_cycle("2024-01-02", 1000, 2000, 3000)
  base = pd.Timestamp("2024-01-03", tz=_NY)
  # Asia and london are resolved; ny stops at 09:50 → offset 20 < 390 → not resolved
  truncated = (
    _asia_bars("2024-01-03", 2000)
    + _london_bars("2024-01-03", 4000)
    + [
      _bar(base.replace(hour=9, minute=30), 100),   # clean open
      _bar(base.replace(hour=9, minute=50), 100),   # last bar offset=20 < 390
    ]
  )
  result = _stat().compute(make_candles([good, truncated]))
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 1
  # Inner join uniformity: all three outcomes reflect the same exclusion
  for outcome in ("mean_asia_volume", "mean_london_volume", "mean_ny_volume"):
    assert _overall(result, outcome).count == 1


def test_missing_asia_open_excludes_cycle() -> None:
  """Cycle whose asia session has no bar at offset 0 (18:00) is not resolved.

  The first asia bar is at 20:00 (offset 120, not 0), so _has_open=False →
  asia table drops that cycle → inner join excludes it from all outcomes.
  """
  good = _full_cycle("2024-01-02", 1000, 2000, 3000)
  base = pd.Timestamp("2024-01-03", tz=_NY)
  prev = base - pd.Timedelta(days=1)
  # No 18:00 bar; first asia bar is at 20:00 → offset (1200-1080)%1440 = 120 ≠ 0
  no_open_asia = [
    _bar(prev.replace(hour=20, minute=0), 300),    # offset 120 (not a clean open)
    _bar(base.replace(hour=1, minute=0), 200),     # offset 420
    _bar(base.replace(hour=2, minute=59), 100),    # coverage
  ]
  cycle_no_open = (
    no_open_asia
    + _london_bars("2024-01-03", 4000)
    + _ny_bars("2024-01-03", 6000)
  )
  result = _stat().compute(make_candles([good, cycle_no_open]))
  assert result.instruments["NQ"]["daily"].total_samples == 1


def test_truncated_london_excludes_cycle() -> None:
  """Cycle whose london session ends at 04:00 (offset 60 < 375) is excluded."""
  good = _full_cycle("2024-01-02", 1000, 2000, 3000)
  base = pd.Timestamp("2024-01-03", tz=_NY)
  # London open at 03:00 (offset 0) but last bar at 04:00 (offset 60 < 375)
  truncated_london = (
    _asia_bars("2024-01-03", 2000)
    + [
      _bar(base.replace(hour=3, minute=0), 800),   # clean open, offset=0
      _bar(base.replace(hour=4, minute=0), 600),   # last bar offset=60 < 375
    ]
    + _ny_bars("2024-01-03", 6000)
  )
  result = _stat().compute(make_candles([good, truncated_london]))
  assert result.instruments["NQ"]["daily"].total_samples == 1


def test_pending_cycle_n_uniform_across_outcomes() -> None:
  """N is the same for all three outcomes — the inner join is enforced globally.

  2 fully-resolved cycles + 1 cycle with truncated NY: N must be 2, not 3,
  for every outcome.
  """
  cycle_a = _full_cycle("2024-01-02", 1000, 2000, 3000)
  cycle_b = _full_cycle("2024-01-03", 2000, 4000, 6000)
  base = pd.Timestamp("2024-01-04", tz=_NY)
  cycle_c = (
    _asia_bars("2024-01-04", 3000)
    + _london_bars("2024-01-04", 6000)
    + [
      _bar(base.replace(hour=9, minute=30), 100),   # clean open
      _bar(base.replace(hour=10, minute=0), 100),   # offset 30 < 390
    ]
  )
  result = _stat().compute(make_candles([cycle_a, cycle_b, cycle_c]))
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 2
  counts = {
    _overall(result, out).count
    for out in ("mean_asia_volume", "mean_london_volume", "mean_ny_volume")
  }
  assert counts == {2}


# ===========================================================================
# 4. Cross-midnight attribution
# ===========================================================================

def test_cross_midnight_attributes_to_next_cycle() -> None:
  """Asia evening bars (>=18:00 on prev day) must be attributed to the next cycle.

  Build one fully-resolved cycle for 2024-01-03 (Wednesday). The asia session
  bars are placed at 2024-01-02 18:00+ (Tuesday evening). If attribution is
  correct, the cycle resolves as Wednesday (N=1, weekday=wednesday). If
  attribution were wrong (evening bars land on 2024-01-02 Tuesday), the london/ny
  tables would have no 2024-01-02 entry and the inner join would produce N=0.
  """
  cycle_wed = _full_cycle("2024-01-03", 2000, 4000, 6000)
  result = _stat().compute(make_candles([cycle_wed]))
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 1
  groups = tf.slices["weekday"].groups
  assert "wednesday" in groups, "cycle should be attributed to Wednesday"
  assert groups["wednesday"].total_samples == 1
  assert "tuesday" not in groups, "Tuesday evening bars must not create a Tuesday cycle"


def test_cross_midnight_volume_lands_on_correct_weekday() -> None:
  """Asia volume is reported under the cycle date, not the prior evening's calendar date.

  Two cycles with distinct asia volumes:
    2024-01-02 Tuesday  : asia_vol=1000  (asia bars at 2024-01-01 18:00)
    2024-01-03 Wednesday: asia_vol=3000  (asia bars at 2024-01-02 18:00)

  Verifies that asia_volume=1000 lands on Tuesday and asia_volume=3000 on Wednesday.
  """
  cycle_tue = _full_cycle("2024-01-02", 1000, 2000, 3000)
  cycle_wed = _full_cycle("2024-01-03", 3000, 4000, 6000)
  result = _stat().compute(make_candles([cycle_tue, cycle_wed]))
  groups = result.instruments["NQ"]["daily"].slices["weekday"].groups
  tue_asia = _row(_weekday_rows(result, "tuesday"), "mean_asia_volume")
  wed_asia = _row(_weekday_rows(result, "wednesday"), "mean_asia_volume")
  # asia_vol=1000 belongs to Tuesday cycle (01-02), asia_vol=3000 to Wednesday (01-03)
  assert tue_asia.value == pytest.approx(1000.0)
  assert wed_asia.value == pytest.approx(3000.0)
  assert "tuesday" in groups and "wednesday" in groups


# ===========================================================================
# 5. Baseline determinism / reproducibility
# ===========================================================================

def test_baseline_reproducible_same_seed() -> None:
  """Same input + same seed => identical value_baseline on both calls."""
  df = _make_all_cycles()
  stat = _stat()
  result_a = stat.compute(df, seed=42)
  result_b = stat.compute(df, seed=42)
  for outcome in ("mean_asia_volume", "mean_london_volume", "mean_ny_volume"):
    a = _overall(result_a, outcome)
    b = _overall(result_b, outcome)
    assert a.value == pytest.approx(b.value)
    assert a.value_baseline is not None
    assert b.value_baseline is not None
    assert a.value_baseline == pytest.approx(b.value_baseline)


def test_baseline_value_baseline_present_overall() -> None:
  """Every outcome row carries a non-None value_baseline after compute()."""
  result = _stat().compute(_make_all_cycles())
  for row in result.instruments["NQ"]["daily"].results:
    assert row.value_baseline is not None, f"Missing value_baseline for {row.outcome}"


def test_baseline_value_baseline_present_weekday_groups() -> None:
  """Every weekday sub-group row also carries value_baseline."""
  result = _stat().compute(_make_all_cycles())
  for day_key in ("monday", "tuesday", "wednesday", "thursday", "friday"):
    for outcome in ("mean_asia_volume", "mean_london_volume", "mean_ny_volume"):
      row = _row(_weekday_rows(result, day_key), outcome)
      assert row.value_baseline is not None, f"{day_key}/{outcome}: missing value_baseline"


def test_baseline_direct_call_reproducible() -> None:
  """baseline_rows() called directly with the same seed returns identical values."""
  df = _make_all_cycles()
  stat = _stat()
  full_table = stat.build_day_table(df)
  rows_a = stat.baseline_rows(full_table, seed=7)
  rows_b = stat.baseline_rows(full_table, seed=7)
  for a, b in zip(rows_a, rows_b):
    assert a.outcome == b.outcome
    assert (a.value or 0.0) == pytest.approx(b.value or 0.0)


def test_baseline_weekday_different_seeds_differ() -> None:
  """Different seeds draw different 2-of-10 samples, producing different averages.

  There are C(10,2)=45 possible 2-element samples from the 10-cycle population;
  the probability that seeds 1 and 3 draw the same pair is 1/45. For this dataset
  seeds 1 and 3 are verified to draw distinct index pairs (empirically confirmed).
  """
  df = _make_all_cycles()
  stat = _stat()
  full_table = stat.build_day_table(df)
  # Isolate Monday rows (2 of them) to pass a sub-table to baseline_rows
  monday_mask = full_table.index.dayofweek == 0
  monday_table = full_table[monday_mask]
  assert len(monday_table) == 2
  assert len(full_table) == 10
  rows_1 = stat.baseline_rows(monday_table, seed=1)
  rows_2 = stat.baseline_rows(monday_table, seed=3)
  all_same = all(
    abs((a.value or 0.0) - (b.value or 0.0)) < 1e-9
    for a, b in zip(rows_1, rows_2)
  )
  assert not all_same


# ===========================================================================
# 6. Empty input
# ===========================================================================

def test_empty_input_no_crash() -> None:
  """Empty DataFrame must not raise; all counters are zero."""
  result = _stat().compute(_empty_df())
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 0
  assert tf.data_range == []


def test_empty_input_three_outcome_rows() -> None:
  """Empty input still produces three outcome rows with zero counts and value=0.0."""
  result = _stat().compute(_empty_df())
  rows = result.instruments["NQ"]["daily"].results
  assert len(rows) == 3
  assert {r.outcome for r in rows} == {
    "mean_asia_volume", "mean_london_volume", "mean_ny_volume"
  }
  for row in rows:
    assert row.count == 0
    assert row.total == 0
    assert row.probability == pytest.approx(0.0)
    assert row.value == pytest.approx(0.0)


def test_empty_input_no_weekday_groups() -> None:
  """No weekday groups when there is no data."""
  result = _stat().compute(_empty_df())
  assert result.instruments["NQ"]["daily"].slices["weekday"].groups == {}


def test_single_cycle_no_crash() -> None:
  """A single resolved cycle works without division errors."""
  cycle = _full_cycle("2024-01-02", 1000, 2000, 3000)
  result = _stat().compute(make_candles([cycle]))
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 1
  # 2024-01-02 is Tuesday
  groups = tf.slices["weekday"].groups
  assert set(groups.keys()) == {"tuesday"}
  assert groups["tuesday"].total_samples == 1


def test_single_cycle_values_correct() -> None:
  """With one cycle, each mean equals that cycle's session volume exactly.

  2024-01-02: asia_vol=1000, london_vol=2000, ny_vol=3000.
  Bar volumes are distributed [40%,30%,20%,10%]; sums must equal the totals.
  """
  cycle = _full_cycle("2024-01-02", 1000, 2000, 3000)
  result = _stat().compute(make_candles([cycle]))
  assert _overall(result, "mean_asia_volume").value == pytest.approx(1000.0)
  assert _overall(result, "mean_london_volume").value == pytest.approx(2000.0)
  assert _overall(result, "mean_ny_volume").value == pytest.approx(3000.0)


# ===========================================================================
# 7. Metadata, validation, and serialisation
# ===========================================================================

def test_stat_name() -> None:
  """stat_name must equal 'session_volume_weekday'."""
  result = _stat().compute(_empty_df())
  assert result.stat_name == "session_volume_weekday"


def test_result_is_stat_run_result_instance() -> None:
  """compute() returns a valid StatRunResult."""
  result = _stat().compute(_make_all_cycles())
  assert isinstance(result, StatRunResult)


def test_i18n_title_and_definition() -> None:
  """Both title and definition carry non-empty en/fr strings."""
  result = _stat().compute(_empty_df())
  assert result.title.en != "" and result.title.fr != ""
  assert result.definition.en != "" and result.definition.fr != ""


def test_i18n_labels_outcomes_complete() -> None:
  """Labels cover all three outcomes plus the any_day condition."""
  result = _stat().compute(_empty_df())
  assert set(result.labels.outcomes) == {
    "mean_asia_volume", "mean_london_volume", "mean_ny_volume"
  }
  assert "any_day" in result.labels.conditions


def test_i18n_labels_have_en_and_fr() -> None:
  """Every condition and outcome label has non-empty en and fr strings."""
  result = _stat().compute(_empty_df())
  for mapping in (result.labels.conditions, result.labels.outcomes):
    for key, i18n in mapping.items():
      assert i18n.en != "", f"{key}.en empty"
      assert i18n.fr != "", f"{key}.fr empty"


def test_labels_dimensions_contains_weekday() -> None:
  """The weekday dimension label is present in labels.dimensions."""
  result = _stat().compute(_make_all_cycles())
  assert "weekday" in result.labels.dimensions
  assert result.labels.dimensions["weekday"].en != ""
  assert result.labels.dimensions["weekday"].fr != ""


def test_unknown_session_raises() -> None:
  """Constructing with an unknown session name raises ValueError."""
  with pytest.raises(ValueError, match="Unknown session"):
    SessionVolumeByWeekday(
      instrument="NQ",
      config=_TEST_CONFIG,
      asia="tokyo",
    )


# ===========================================================================
# 7b. classify_samples
#
# Reusing _CYCLES (10 fully-resolved cycles). Each cycle emits exactly three
# SampleRows — one per outcome (mean_asia_volume, mean_london_volume,
# mean_ny_volume) — carrying that cycle's session volume as ``value``.
# 10 cycles * 3 outcomes = 30 samples total.
# ===========================================================================

def test_classify_samples_exact_list() -> None:
  """classify_samples emits three SampleRows per cycle, matching the hand-calc."""
  stat = _stat()
  day_table = stat.build_day_table(_make_all_cycles())
  samples = stat.classify_samples(day_table)
  expected = [
    (date, "any_day", outcome, value)
    for date, asia_vol, london_vol, ny_vol in _CYCLES
    for outcome, value in (
      ("mean_asia_volume", float(asia_vol)),
      ("mean_london_volume", float(london_vol)),
      ("mean_ny_volume", float(ny_vol)),
    )
  ]
  assert len(samples) == len(expected) == 30
  for s, (date, condition, outcome, value) in zip(samples, expected):
    assert s.date == date
    assert s.condition == condition
    assert s.outcome == outcome
    assert s.value == pytest.approx(value)


def test_classify_samples_matches_compute_rows_counts() -> None:
  """For every StatResultRow, the matching SampleRow count equals r.count."""
  stat = _stat()
  day_table = stat.build_day_table(_make_all_cycles())
  samples = stat.classify_samples(day_table)
  rows = stat.compute_rows(day_table)
  for r in rows:
    n = sum(1 for s in samples if s.condition == r.condition and s.outcome == r.outcome)
    assert n == r.count


def test_classify_samples_means_match_compute_rows_values() -> None:
  """Mean of each outcome's SampleRow values reproduces the compute_rows value."""
  stat = _stat()
  day_table = stat.build_day_table(_make_all_cycles())
  samples = stat.classify_samples(day_table)
  rows = stat.compute_rows(day_table)
  for r in rows:
    values = [
      s.value for s in samples if s.condition == r.condition and s.outcome == r.outcome
    ]
    assert sum(values) / len(values) == pytest.approx(r.value)


def test_classify_samples_empty_day_table() -> None:
  """Empty day_table -> classify_samples returns []."""
  stat = _stat()
  assert stat.classify_samples(stat.build_day_table(_empty_df())) == []


def test_classify_samples_excludes_pending_cycle() -> None:
  """A cycle with an unresolved session never appears among the SampleRows."""
  good = _full_cycle("2024-01-02", 1000, 2000, 3000)
  base = pd.Timestamp("2024-01-03", tz=_NY)
  truncated = (
    _asia_bars("2024-01-03", 2000)
    + _london_bars("2024-01-03", 4000)
    + [
      _bar(base.replace(hour=9, minute=30), 100),   # clean open
      _bar(base.replace(hour=9, minute=50), 100),   # last bar offset=20 < 390
    ]
  )
  stat = _stat()
  day_table = stat.build_day_table(make_candles([good, truncated]))
  samples = stat.classify_samples(day_table)
  sample_dates = {s.date for s in samples}
  assert sample_dates == {"2024-01-02"}
  assert len(samples) == 3


def test_write_results_round_trip(tmp_path: Path) -> None:
  """write_results serialises to JSON; the file round-trips through StatRunResult."""
  result = _stat().compute(_make_all_cycles())
  written = write_results(result, results_dir=tmp_path)

  assert written.name == "session_volume_weekday.json"
  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)

  tf = validated.instruments["NQ"]["daily"]
  assert tf.total_samples == 10

  monday = tf.slices["weekday"].groups["monday"]
  asia_row = next(r for r in monday.results if r.outcome == "mean_asia_volume")
  # Monday mean_asia_volume = (5000+10000)/2 = 7500
  assert asia_row.value == pytest.approx(7500.0)
  ny_row = next(r for r in monday.results if r.outcome == "mean_ny_volume")
  # Monday mean_ny_volume = (15000+30000)/2 = 22500
  assert ny_row.value == pytest.approx(22500.0)
