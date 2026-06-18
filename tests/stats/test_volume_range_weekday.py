"""Tests for stats.volume_range_weekday.standard.

All data is synthetic — no real market files required.
Expected values are hand-calculated before each assertion.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.volume_range_weekday.standard import VolumeRangeByWeekday

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

# RTH window: [570, 975).  09:30 = 570, 16:15 = 975.
# Resolution: bar at exactly 570 AND last bar mod >= 975 - 15 = 960 (>= 16:00).
_RTH_START = 570   # 09:30
_RTH_LAST = 974    # 16:14  (last bar before 16:15)
# Number of 1-min bars in a full RTH session: 974 - 570 + 1 = 405
_N_BARS = _RTH_LAST - _RTH_START + 1  # 405


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

def _make_day(
  date: str,
  session_open: float,
  day_high: float,
  day_low: float,
  bar_volume: int,
) -> pd.DataFrame:
  """Build one full RTH trading day of 1-min OHLCV bars (09:30 – 16:14).

  The 09:30 bar sets session_open.  The day's high/low extremes are controlled
  explicitly: every bar uses a neutral OHLC inside the (day_low, day_high)
  interval, with the exception of the first bar whose ``high`` is set to
  ``day_high`` and the last bar whose ``low`` is set to ``day_low``.  This
  guarantees max(high) == day_high and min(low) == day_low exactly, while the
  neutral intermediate bars never violate those bounds.

  Every bar carries ``bar_volume`` units; the day's summed RTH volume is
  therefore ``bar_volume * 405``.
  """
  base = pd.Timestamp(date, tz=_NY)
  # Neutral OHLC: a flat bar in the middle of the range so it never touches
  # the intended extremes.
  mid = (day_high + day_low) / 2.0
  neutral_high = mid + 0.25
  neutral_low = mid - 0.25
  records = []
  for mod in range(_RTH_START, _RTH_LAST + 1):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    o = session_open if mod == _RTH_START else mid
    c = mid
    bar_high = day_high if mod == _RTH_START else neutral_high
    bar_low = day_low if mod == _RTH_LAST else neutral_low
    records.append({
      "timestamp": ts,
      "open": o,
      "high": bar_high,
      "low": bar_low,
      "close": c,
      "volume": bar_volume,
    })
  return pd.DataFrame(records)


def _make_truncated_day(date: str) -> pd.DataFrame:
  """A day whose last RTH bar is 09:50 (mod 590 < 960) — must be excluded."""
  base = pd.Timestamp(date, tz=_NY)
  records = []
  for mod in range(_RTH_START, 591):
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


def make_candles(days: list[dict]) -> pd.DataFrame:
  """Concatenate per-day DataFrames into a sorted multi-day OHLCV frame."""
  frames = [
    _make_day(
      d["date"],
      d["session_open"],
      d["day_high"],
      d["day_low"],
      d["bar_volume"],
    )
    for d in days
  ]
  df = pd.concat(frames, ignore_index=True)
  return df.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main synthetic dataset: 10 resolved days, 2 per weekday.
#
# Dates and weekdays (2024 calendar):
#   2024-01-01 Monday  session_open=100, day_high=110, day_low=90,  bar_vol=100
#   2024-01-02 Tuesday session_open=200, day_high=220, day_low=180, bar_vol=200
#   2024-01-03 Wed     session_open=150, day_high=165, day_low=135, bar_vol=150
#   2024-01-04 Thu     session_open=100, day_high=115, day_low=85,  bar_vol=100
#   2024-01-05 Fri     session_open=200, day_high=210, day_low=190, bar_vol=50
#   2024-01-08 Monday  session_open=100, day_high=120, day_low=80,  bar_vol=200
#   2024-01-09 Tuesday session_open=200, day_high=230, day_low=170, bar_vol=100
#   2024-01-10 Wed     session_open=150, day_high=180, day_low=120, bar_vol=200
#   2024-01-11 Thu     session_open=100, day_high=110, day_low=90,  bar_vol=300
#   2024-01-12 Fri     session_open=200, day_high=240, day_low=160, bar_vol=150
#
# Per-day derived values (N_BARS = 405):
#   volume      = bar_vol * 405
#   range       = day_high - day_low
#   range_pct   = range / session_open
#
#   Day 01-01: vol=40500,  range=20,  range_pct=0.20
#   Day 01-02: vol=81000,  range=40,  range_pct=0.20
#   Day 01-03: vol=60750,  range=30,  range_pct=0.20
#   Day 01-04: vol=40500,  range=30,  range_pct=0.30
#   Day 01-05: vol=20250,  range=20,  range_pct=0.10
#   Day 01-08: vol=81000,  range=40,  range_pct=0.40
#   Day 01-09: vol=40500,  range=60,  range_pct=0.30
#   Day 01-10: vol=81000,  range=60,  range_pct=0.40
#   Day 01-11: vol=121500, range=20,  range_pct=0.20
#   Day 01-12: vol=60750,  range=80,  range_pct=0.40
#
# Overall (10 days):
#   mean_volume   = 627750 / 10 = 62775.0
#   mean_range    = 400 / 10    = 40.0
#   mean_range_pct= 2.70 / 10   = 0.27
#
# Per weekday (2 days each):
#   Monday:    vol=(40500+81000)/2=60750,   range=(20+40)/2=30,  range_pct=(0.20+0.40)/2=0.30
#   Tuesday:   vol=(81000+40500)/2=60750,   range=(40+60)/2=50,  range_pct=(0.20+0.30)/2=0.25
#   Wednesday: vol=(60750+81000)/2=70875,   range=(30+60)/2=45,  range_pct=(0.20+0.40)/2=0.30
#   Thursday:  vol=(40500+121500)/2=81000,  range=(30+20)/2=25,  range_pct=(0.30+0.20)/2=0.25
#   Friday:    vol=(20250+60750)/2=40500,   range=(20+80)/2=50,  range_pct=(0.10+0.40)/2=0.25
# ---------------------------------------------------------------------------

_DAYS = [
  {"date": "2024-01-01", "session_open": 100.0, "day_high": 110.0, "day_low": 90.0,  "bar_volume": 100},
  {"date": "2024-01-02", "session_open": 200.0, "day_high": 220.0, "day_low": 180.0, "bar_volume": 200},
  {"date": "2024-01-03", "session_open": 150.0, "day_high": 165.0, "day_low": 135.0, "bar_volume": 150},
  {"date": "2024-01-04", "session_open": 100.0, "day_high": 115.0, "day_low": 85.0,  "bar_volume": 100},
  {"date": "2024-01-05", "session_open": 200.0, "day_high": 210.0, "day_low": 190.0, "bar_volume": 50},
  {"date": "2024-01-08", "session_open": 100.0, "day_high": 120.0, "day_low": 80.0,  "bar_volume": 200},
  {"date": "2024-01-09", "session_open": 200.0, "day_high": 230.0, "day_low": 170.0, "bar_volume": 100},
  {"date": "2024-01-10", "session_open": 150.0, "day_high": 180.0, "day_low": 120.0, "bar_volume": 200},
  {"date": "2024-01-11", "session_open": 100.0, "day_high": 110.0, "day_low": 90.0,  "bar_volume": 300},
  {"date": "2024-01-12", "session_open": 200.0, "day_high": 240.0, "day_low": 160.0, "bar_volume": 150},
]


def _stat() -> VolumeRangeByWeekday:
  return VolumeRangeByWeekday(instrument="NQ", config=_TEST_CONFIG)


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


# ===========================================================================
# 1. build_day_table: per-row correctness, resolution filtering, index
# ===========================================================================

def test_build_day_table_row_count() -> None:
  """All 10 fully resolved days produce exactly 10 rows."""
  dt = _stat().build_day_table(make_candles(_DAYS))
  assert len(dt) == 10


def test_build_day_table_columns_present() -> None:
  """The table must carry the three metric columns."""
  dt = _stat().build_day_table(make_candles(_DAYS))
  assert set(dt.columns) >= {"volume", "range", "range_pct"}


def test_build_day_table_index_is_session_date() -> None:
  """Index is normalized timestamps, one per resolved day."""
  dt = _stat().build_day_table(make_candles(_DAYS))
  expected = {
    pd.Timestamp("2024-01-01", tz=_NY).normalize(),
    pd.Timestamp("2024-01-12", tz=_NY).normalize(),
  }
  assert expected <= set(dt.index)


def test_build_day_table_volume_correct() -> None:
  """Day 2024-01-01: bar_volume=100, 405 bars => volume=40500."""
  dt = _stat().build_day_table(make_candles(_DAYS))
  date = pd.Timestamp("2024-01-01", tz=_NY).normalize()
  # volume = bar_volume * N_BARS = 100 * 405 = 40500
  assert dt.loc[date, "volume"] == pytest.approx(40500.0)


def test_build_day_table_range_correct() -> None:
  """Day 2024-01-02: day_high=220, day_low=180 => range=40."""
  dt = _stat().build_day_table(make_candles(_DAYS))
  date = pd.Timestamp("2024-01-02", tz=_NY).normalize()
  # range = day_high - day_low = 220 - 180 = 40
  assert dt.loc[date, "range"] == pytest.approx(40.0)


def test_build_day_table_range_pct_correct() -> None:
  """Day 2024-01-04: range=30, session_open=100 => range_pct=0.30."""
  dt = _stat().build_day_table(make_candles(_DAYS))
  date = pd.Timestamp("2024-01-04", tz=_NY).normalize()
  # range_pct = (115 - 85) / 100 = 0.30
  assert dt.loc[date, "range_pct"] == pytest.approx(0.30)


def test_build_day_table_excludes_truncated_day() -> None:
  """A day ending at 09:50 (mod 590 < 960) is excluded as unresolved."""
  base_df = make_candles(_DAYS)
  trunc = _make_truncated_day("2024-01-15")
  df = pd.concat([base_df, trunc], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
  dt = _stat().build_day_table(df)
  excluded = pd.Timestamp("2024-01-15", tz=_NY).normalize()
  assert excluded not in dt.index
  assert len(dt) == 10


def test_build_day_table_sorted_chronologically() -> None:
  """Index must be sorted in ascending date order."""
  dt = _stat().build_day_table(make_candles(_DAYS))
  assert list(dt.index) == sorted(dt.index)


# ===========================================================================
# 2. compute_rows: overall magnitudes, counts, probability channel
# ===========================================================================

def test_overall_three_rows() -> None:
  """Exactly three outcome rows under the single any_day condition."""
  result = _stat().compute(make_candles(_DAYS))
  rows = result.instruments["NQ"]["daily"].results
  assert len(rows) == 3
  assert {r.outcome for r in rows} == {"mean_volume", "mean_range", "mean_range_pct"}
  assert {r.condition for r in rows} == {"any_day"}


def test_overall_mean_volume() -> None:
  """mean_volume = 627750 / 10 = 62775.0 over 10 resolved days."""
  result = _stat().compute(make_candles(_DAYS))
  row = _overall(result, "mean_volume")
  # sum volumes: 40500+81000+60750+40500+20250+81000+40500+81000+121500+60750 = 627750
  # mean = 627750 / 10 = 62775
  assert row.value == pytest.approx(62775.0)
  assert row.count == 10
  assert row.total == 10
  assert row.probability == pytest.approx(0.0)


def test_overall_mean_range() -> None:
  """mean_range = 400 / 10 = 40.0 points."""
  result = _stat().compute(make_candles(_DAYS))
  row = _overall(result, "mean_range")
  # ranges: 20+40+30+30+20+40+60+60+20+80 = 400; mean = 40.0
  assert row.value == pytest.approx(40.0)
  assert row.count == 10
  assert row.total == 10
  assert row.probability == pytest.approx(0.0)


def test_overall_mean_range_pct() -> None:
  """mean_range_pct = 2.70 / 10 = 0.27."""
  result = _stat().compute(make_candles(_DAYS))
  row = _overall(result, "mean_range_pct")
  # range_pcts: 0.20+0.20+0.20+0.30+0.10+0.40+0.30+0.40+0.20+0.40 = 2.70; mean = 0.27
  assert row.value == pytest.approx(0.27)
  assert row.count == 10
  assert row.total == 10
  assert row.probability == pytest.approx(0.0)


def test_overall_data_range() -> None:
  """data_range spans first to last resolved session date."""
  result = _stat().compute(make_candles(_DAYS))
  assert result.instruments["NQ"]["daily"].data_range == ["2024-01-01", "2024-01-12"]


def test_overall_total_samples() -> None:
  """total_samples equals the number of resolved days."""
  result = _stat().compute(make_candles(_DAYS))
  assert result.instruments["NQ"]["daily"].total_samples == 10


# ===========================================================================
# 3. Weekday slice: correct groups and per-weekday means
# ===========================================================================

def test_weekday_slice_five_groups() -> None:
  """Five weekday groups are present, each with 2 days."""
  result = _stat().compute(make_candles(_DAYS))
  groups = result.instruments["NQ"]["daily"].slices["weekday"].groups
  assert set(groups.keys()) == {"monday", "tuesday", "wednesday", "thursday", "friday"}
  for g in groups.values():
    assert g.total_samples == 2


def test_weekday_mean_volumes() -> None:
  """Per-weekday average volumes match hand-calculated values."""
  result = _stat().compute(make_candles(_DAYS))
  # Monday:    (40500+81000)/2  = 60750
  # Tuesday:   (81000+40500)/2  = 60750
  # Wednesday: (60750+81000)/2  = 70875
  # Thursday:  (40500+121500)/2 = 81000
  # Friday:    (20250+60750)/2  = 40500
  expected = {
    "monday": 60750.0,
    "tuesday": 60750.0,
    "wednesday": 70875.0,
    "thursday": 81000.0,
    "friday": 40500.0,
  }
  for day_key, exp in expected.items():
    row = _row(_weekday_rows(result, day_key), "mean_volume")
    assert row.value == pytest.approx(exp), f"{day_key}: expected {exp}, got {row.value}"


def test_weekday_mean_ranges() -> None:
  """Per-weekday average price ranges match hand-calculated values."""
  result = _stat().compute(make_candles(_DAYS))
  # Monday:    (20+40)/2 = 30
  # Tuesday:   (40+60)/2 = 50
  # Wednesday: (30+60)/2 = 45
  # Thursday:  (30+20)/2 = 25
  # Friday:    (20+80)/2 = 50
  expected = {
    "monday": 30.0,
    "tuesday": 50.0,
    "wednesday": 45.0,
    "thursday": 25.0,
    "friday": 50.0,
  }
  for day_key, exp in expected.items():
    row = _row(_weekday_rows(result, day_key), "mean_range")
    assert row.value == pytest.approx(exp), f"{day_key}: expected {exp}, got {row.value}"


def test_weekday_mean_range_pcts() -> None:
  """Per-weekday average range_pct values match hand-calculated values."""
  result = _stat().compute(make_candles(_DAYS))
  # Monday:    (0.20+0.40)/2 = 0.30
  # Tuesday:   (0.20+0.30)/2 = 0.25
  # Wednesday: (0.20+0.40)/2 = 0.30
  # Thursday:  (0.30+0.20)/2 = 0.25
  # Friday:    (0.10+0.40)/2 = 0.25
  expected = {
    "monday": 0.30,
    "tuesday": 0.25,
    "wednesday": 0.30,
    "thursday": 0.25,
    "friday": 0.25,
  }
  for day_key, exp in expected.items():
    row = _row(_weekday_rows(result, day_key), "mean_range_pct")
    assert row.value == pytest.approx(exp), f"{day_key}: expected {exp}, got {row.value}"


def test_weekday_slice_count_and_total() -> None:
  """Each weekday slice row carries count == total == 2."""
  result = _stat().compute(make_candles(_DAYS))
  for day_key in ("monday", "tuesday", "wednesday", "thursday", "friday"):
    for outcome in ("mean_volume", "mean_range", "mean_range_pct"):
      row = _row(_weekday_rows(result, day_key), outcome)
      assert row.count == 2, f"{day_key}/{outcome}: count"
      assert row.total == 2, f"{day_key}/{outcome}: total"
      assert row.probability == pytest.approx(0.0), f"{day_key}/{outcome}: probability"


# ===========================================================================
# 4. Baseline: overall permutation property (N sampled from N == identity)
# ===========================================================================

def test_overall_baseline_equals_value() -> None:
  """For the full table, sampling N from N without replacement is a
  permutation, so the baseline averages equal the true averages exactly."""
  result = _stat().compute(make_candles(_DAYS))
  for row in result.instruments["NQ"]["daily"].results:
    # value_baseline is set when baseline_rows is merged into compute_rows.
    assert row.value_baseline is not None, f"Missing value_baseline for {row.outcome}"
    assert row.value_baseline == pytest.approx(row.value), (
      f"{row.outcome}: value={row.value}, value_baseline={row.value_baseline}"
    )


# ===========================================================================
# 5. Determinism / reproducibility
# ===========================================================================

def test_compute_twice_same_result() -> None:
  """compute() with the default seed returns identical value_baseline on both calls."""
  df = make_candles(_DAYS)
  stat = _stat()
  result_a = stat.compute(df)
  result_b = stat.compute(df)
  rows_a = result_a.instruments["NQ"]["daily"].results
  rows_b = result_b.instruments["NQ"]["daily"].results
  assert len(rows_a) == len(rows_b)
  for a, b in zip(rows_a, rows_b):
    assert a.outcome == b.outcome
    assert a.value == pytest.approx(b.value)
    if a.value_baseline is None:
      assert b.value_baseline is None
    else:
      assert a.value_baseline == pytest.approx(b.value_baseline)


def test_baseline_deterministic_fixed_seed() -> None:
  """Same seed → identical baseline_rows output."""
  df = make_candles(_DAYS)
  stat = _stat()
  day_table = stat.build_day_table(df)
  rows_a = stat.baseline_rows(day_table, seed=7)
  rows_b = stat.baseline_rows(day_table, seed=7)
  for a, b in zip(rows_a, rows_b):
    assert a.outcome == b.outcome
    if a.value is None:
      assert b.value is None
    else:
      assert a.value == pytest.approx(b.value)


def test_baseline_weekday_slice_different_seeds_differ() -> None:
  """For a weekday sub-slice (n=2, population=10), different seeds draw
  different 2-day samples from the full 10-day table, producing different
  averages (with overwhelmingly high probability for this dataset)."""
  df = make_candles(_DAYS)
  stat = _stat()
  # Build the day table first so _full_day_table is populated.
  full_table = stat.build_day_table(df)
  # Simulate what _slice_results does: pass just Monday's 2-row sub-table
  # while the full 10-row table is stashed on stat._full_day_table.
  monday_mask = full_table.index.dayofweek == 0
  monday_table = full_table[monday_mask]
  assert len(monday_table) == 2
  assert len(full_table) == 10
  # Different seeds draw different 2-of-10 samples → different averages
  # (there are C(10,2)=45 possible samples; collisions are negligible here).
  rows_seed1 = stat.baseline_rows(monday_table, seed=1)
  rows_seed2 = stat.baseline_rows(monday_table, seed=2)
  all_same = all(
    abs((a.value or 0.0) - (b.value or 0.0)) < 1e-9
    for a, b in zip(rows_seed1, rows_seed2)
  )
  assert not all_same


# ===========================================================================
# 6. Edge cases
# ===========================================================================

def test_empty_dataframe_no_crash() -> None:
  """Empty input: 0 samples, empty data_range, no crash."""
  result = _stat().compute(_empty_df())
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 0
  assert tf.data_range == []


def test_empty_dataframe_zero_counts() -> None:
  """All rows report count == total == 0 and probability == 0.0."""
  result = _stat().compute(_empty_df())
  for row in result.instruments["NQ"]["daily"].results:
    assert row.count == 0
    assert row.total == 0
    assert row.probability == pytest.approx(0.0)
    # Magnitude rows fall back to 0.0 when there are no days.
    assert row.value == pytest.approx(0.0)


def test_empty_dataframe_no_weekday_groups() -> None:
  """No weekday groups when there is no data."""
  result = _stat().compute(_empty_df())
  assert result.instruments["NQ"]["daily"].slices["weekday"].groups == {}


def test_single_day_no_crash() -> None:
  """A single resolved day works without division errors."""
  single = [_DAYS[0]]
  result = _stat().compute(make_candles(single))
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 1
  # Only one weekday group (Monday).
  groups = tf.slices["weekday"].groups
  assert set(groups.keys()) == {"monday"}
  assert groups["monday"].total_samples == 1


def test_single_day_values_correct() -> None:
  """With one day, per-day values equal the overall averages."""
  single = [_DAYS[0]]  # Mon 2024-01-01: vol=40500, range=20, range_pct=0.20
  result = _stat().compute(make_candles(single))
  assert _overall(result, "mean_volume").value == pytest.approx(40500.0)
  assert _overall(result, "mean_range").value == pytest.approx(20.0)
  assert _overall(result, "mean_range_pct").value == pytest.approx(0.20)


def test_pending_day_excluded_from_total() -> None:
  """Appending a truncated day does not change total_samples."""
  base_df = make_candles(_DAYS)
  base_total = _stat().compute(base_df).instruments["NQ"]["daily"].total_samples

  trunc = _make_truncated_day("2024-01-15")
  extended = pd.concat([base_df, trunc], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
  new_total = _stat().compute(extended).instruments["NQ"]["daily"].total_samples
  assert new_total == base_total == 10


def test_pending_day_absent_from_day_table() -> None:
  """A truncated day is absent from the day table index."""
  trunc = _make_truncated_day("2024-01-15")
  dt = _stat().build_day_table(trunc)
  excluded = pd.Timestamp("2024-01-15", tz=_NY).normalize()
  assert excluded not in dt.index
  assert len(dt) == 0


# ===========================================================================
# 7. End-to-end: StatRunResult validity and write_results round-trip
# ===========================================================================

def test_stat_name() -> None:
  """stat_name attribute must equal 'volume_range_weekday'."""
  result = _stat().compute(_empty_df())
  assert result.stat_name == "volume_range_weekday"


def test_result_is_stat_run_result_instance() -> None:
  """compute() returns a valid StatRunResult."""
  result = _stat().compute(make_candles(_DAYS))
  assert isinstance(result, StatRunResult)


def test_i18n_title_and_definition() -> None:
  """Both title and definition carry non-empty en/fr strings."""
  result = _stat().compute(_empty_df())
  assert result.title.en != "" and result.title.fr != ""
  assert result.definition.en != "" and result.definition.fr != ""


def test_i18n_labels_outcomes_complete() -> None:
  """Labels cover all three outcomes plus the any_day condition."""
  result = _stat().compute(_empty_df())
  assert set(result.labels.outcomes) == {"mean_volume", "mean_range", "mean_range_pct"}
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
  result = _stat().compute(make_candles(_DAYS))
  assert "weekday" in result.labels.dimensions
  assert result.labels.dimensions["weekday"].en != ""
  assert result.labels.dimensions["weekday"].fr != ""


def test_write_results_round_trip(tmp_path: Path) -> None:
  """write_results serialises to JSON; the file validates back to StatRunResult."""
  result = _stat().compute(make_candles(_DAYS))
  written = write_results(result, results_dir=tmp_path)

  assert written.name == "volume_range_weekday.json"
  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)

  tf = validated.instruments["NQ"]["daily"]
  assert tf.total_samples == 10
  monday = tf.slices["weekday"].groups["monday"]
  vol_row = next(r for r in monday.results if r.outcome == "mean_volume")
  # Monday mean_volume = 60750.0
  assert vol_row.value == pytest.approx(60750.0)
  range_row = next(r for r in monday.results if r.outcome == "mean_range")
  # Monday mean_range = 30.0
  assert range_row.value == pytest.approx(30.0)
