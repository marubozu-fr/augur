"""Tests for stats.performance_weekday.standard.

All data is synthetic — no real market files required.
Expected values are hand-calculated before each assertion.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.performance_weekday.standard import PerformanceByWeekday

# ---------------------------------------------------------------------------
# Minimal InstrumentConfig used by all tests (does not depend on NQ.yaml)
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

# RTH minutes: 09:30 = 570, 16:15 = 975
# Resolved if last bar mod >= 975 - 15 = 960  (i.e., >= 16:00)
_RTH_START = 570   # 09:30
_RTH_LAST = 974    # 16:14  (last bar before 16:15)


# ---------------------------------------------------------------------------
# Synthetic data builder
# ---------------------------------------------------------------------------

def _make_day(date: str, session_open: float, session_close: float) -> pd.DataFrame:
  """Build one trading day of 1-min RTH OHLCV bars (09:30–16:14).

  The 09:30 bar opens at session_open; the 16:14 bar closes at session_close.
  Intermediate bars are flat at 100.0.
  """
  base = pd.Timestamp(date, tz=_NY)
  records = []
  for mod in range(_RTH_START, _RTH_LAST + 1):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    o = 100.0
    c = 100.0
    if mod == _RTH_START:
      o = session_open
    if mod == _RTH_LAST:
      c = session_close
    records.append({
      "timestamp": ts,
      "open": o,
      "high": max(o, c) + 0.25,
      "low": min(o, c) - 0.25,
      "close": c,
      "volume": 1000,
    })
  return pd.DataFrame(records)


def make_candles(days: list[dict]) -> pd.DataFrame:
  """Build a multi-day 1-min OHLCV DataFrame from a list of day specs."""
  frames = [_make_day(d["date"], d["session_open"], d["session_close"]) for d in days]
  df = pd.concat(frames, ignore_index=True)
  return df.sort_values("timestamp").reset_index(drop=True)


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
      "high": 100.25,
      "low": 99.75,
      "close": 100.0,
      "volume": 500,
    })
  return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# open_to_close dataset: 10 days, 2 per weekday, session_open=100 always.
#
#   Mon 120 (+0.20), Tue 80 (-0.20), Wed 120 (+0.20), Thu 120 (+0.20), Fri 80 (-0.20)
#   Mon 110 (+0.10), Tue 90 (-0.10), Wed 80 (-0.20),  Thu 120 (+0.20), Fri 90 (-0.10)
#
# Overall returns: [.20,-.20,.20,.20,-.20,.10,-.10,-.20,.20,-.10] → sum 0.10, mean 0.01
#   green = 5 (>=0), red = 5 → P(green)=0.5
#   mean_green = mean(.20,.20,.20,.10,.20) = 0.18
#   mean_red   = mean(-.20,-.20,-.20,-.10,-.10) = -0.16
# Per weekday (2 each):
#   Mon: returns .20,.10 → mean 0.15, green 2/2, mean_green 0.15, mean_red 0.0
#   Tue: -.20,-.10       → mean -0.15, red 2/2, mean_green 0.0, mean_red -0.15
#   Wed: .20,-.20        → mean 0.0,  green 1/2, mean_green 0.20, mean_red -0.20
#   Thu: .20,.20         → mean 0.20, green 2/2, mean_green 0.20, mean_red 0.0
#   Fri: -.20,-.10       → mean -0.15, red 2/2, mean_green 0.0, mean_red -0.15
# ---------------------------------------------------------------------------

_OTC_DAYS = [
  {"date": "2024-01-01", "session_open": 100.0, "session_close": 120.0},  # Mon +0.20
  {"date": "2024-01-02", "session_open": 100.0, "session_close": 80.0},   # Tue -0.20
  {"date": "2024-01-03", "session_open": 100.0, "session_close": 120.0},  # Wed +0.20
  {"date": "2024-01-04", "session_open": 100.0, "session_close": 120.0},  # Thu +0.20
  {"date": "2024-01-05", "session_open": 100.0, "session_close": 80.0},   # Fri -0.20
  {"date": "2024-01-08", "session_open": 100.0, "session_close": 110.0},  # Mon +0.10
  {"date": "2024-01-09", "session_open": 100.0, "session_close": 90.0},   # Tue -0.10
  {"date": "2024-01-10", "session_open": 100.0, "session_close": 80.0},   # Wed -0.20
  {"date": "2024-01-11", "session_open": 100.0, "session_close": 120.0},  # Thu +0.20
  {"date": "2024-01-12", "session_open": 100.0, "session_close": 90.0},   # Fri -0.10
]


def _otc_stat() -> PerformanceByWeekday:
  return PerformanceByWeekday(
    instrument="NQ", config=_TEST_CONFIG, performance="open_to_close"
  )


def _ctc_stat() -> PerformanceByWeekday:
  return PerformanceByWeekday(
    instrument="NQ", config=_TEST_CONFIG, performance="close_to_close"
  )


def _row(rows: list[StatResultRow], outcome: str) -> StatResultRow:
  for r in rows:
    if r.condition == "any_day" and r.outcome == outcome:
      return r
  raise KeyError(outcome)


def _overall(result: StatRunResult, outcome: str) -> StatResultRow:
  return _row(result.instruments["NQ"]["daily"].results, outcome)


def _weekday_rows(result: StatRunResult, day_key: str) -> list[StatResultRow]:
  return result.instruments["NQ"]["daily"].slices["weekday"].groups[day_key].results


# ===========================================================================
# 1. open_to_close: overall magnitudes and counts
# ===========================================================================

def test_overall_five_rows() -> None:
  """Exactly five outcome rows under the single any_day condition."""
  result = _otc_stat().compute(make_candles(_OTC_DAYS))
  rows = result.instruments["NQ"]["daily"].results
  assert len(rows) == 5
  assert {r.outcome for r in rows} == {
    "mean_return", "green_day", "red_day", "mean_green_move", "mean_red_move",
  }
  assert {r.condition for r in rows} == {"any_day"}


def test_overall_mean_return() -> None:
  """Overall average return = 0.01 (sum 0.10 over 10 days), in `value`."""
  result = _otc_stat().compute(make_candles(_OTC_DAYS))
  mr = _overall(result, "mean_return")
  assert mr.value == pytest.approx(0.01)
  assert mr.count == 10
  assert mr.total == 10
  # Magnitude rows do not use the probability channel.
  assert mr.probability == pytest.approx(0.0)


def test_overall_green_red_counts() -> None:
  """5 green + 5 red → P(green)=0.5, counts carried in count/total."""
  result = _otc_stat().compute(make_candles(_OTC_DAYS))
  green = _overall(result, "green_day")
  red = _overall(result, "red_day")
  assert green.count == 5
  assert green.total == 10
  assert green.probability == pytest.approx(0.5)
  assert red.count == 5
  assert red.probability == pytest.approx(0.5)
  assert green.count + red.count == 10
  # Count rows carry no continuous metric.
  assert green.value is None
  assert red.value is None


def test_overall_mean_moves() -> None:
  """mean_green_move = 0.18, mean_red_move = -0.16; counts match green/red."""
  result = _otc_stat().compute(make_candles(_OTC_DAYS))
  mg = _overall(result, "mean_green_move")
  mr = _overall(result, "mean_red_move")
  assert mg.value == pytest.approx(0.18)
  assert mg.count == 5  # number of green days
  assert mg.total == 10
  assert mr.value == pytest.approx(-0.16)
  assert mr.count == 5


def test_overall_data_range() -> None:
  result = _otc_stat().compute(make_candles(_OTC_DAYS))
  assert result.instruments["NQ"]["daily"].data_range == ["2024-01-01", "2024-01-12"]


# ===========================================================================
# 2. Weekday slice magnitudes
# ===========================================================================

def test_weekday_slice_five_groups() -> None:
  result = _otc_stat().compute(make_candles(_OTC_DAYS))
  groups = result.instruments["NQ"]["daily"].slices["weekday"].groups
  assert set(groups.keys()) == {"monday", "tuesday", "wednesday", "thursday", "friday"}
  for g in groups.values():
    assert g.total_samples == 2


def test_weekday_mean_returns() -> None:
  """Per-weekday average return."""
  result = _otc_stat().compute(make_candles(_OTC_DAYS))
  expected = {
    "monday": 0.15,
    "tuesday": -0.15,
    "wednesday": 0.0,
    "thursday": 0.20,
    "friday": -0.15,
  }
  for day_key, exp in expected.items():
    mr = _row(_weekday_rows(result, day_key), "mean_return")
    assert mr.value == pytest.approx(exp), f"{day_key}: expected {exp}, got {mr.value}"


def test_weekday_green_probabilities() -> None:
  """Per-weekday P(green): Mon=1.0, Tue=0.0, Wed=0.5, Thu=1.0, Fri=0.0."""
  result = _otc_stat().compute(make_candles(_OTC_DAYS))
  expected = {
    "monday": 1.0, "tuesday": 0.0, "wednesday": 0.5, "thursday": 1.0, "friday": 0.0,
  }
  for day_key, exp in expected.items():
    green = _row(_weekday_rows(result, day_key), "green_day")
    assert green.probability == pytest.approx(exp), f"{day_key}: got {green.probability}"


def test_weekday_mean_moves() -> None:
  """Per-weekday average green/red move size."""
  result = _otc_stat().compute(make_candles(_OTC_DAYS))
  # (mean_green_move, mean_red_move)
  expected = {
    "monday": (0.15, 0.0),
    "tuesday": (0.0, -0.15),
    "wednesday": (0.20, -0.20),
    "thursday": (0.20, 0.0),
    "friday": (0.0, -0.15),
  }
  for day_key, (exp_g, exp_r) in expected.items():
    mg = _row(_weekday_rows(result, day_key), "mean_green_move")
    mr = _row(_weekday_rows(result, day_key), "mean_red_move")
    assert mg.value == pytest.approx(exp_g), f"{day_key} green: got {mg.value}"
    assert mr.value == pytest.approx(exp_r), f"{day_key} red: got {mr.value}"


def test_weekday_returns_average_to_overall() -> None:
  """Mean of per-weekday mean_returns (equal weekday counts) equals overall."""
  result = _otc_stat().compute(make_candles(_OTC_DAYS))
  groups = result.instruments["NQ"]["daily"].slices["weekday"].groups
  per_weekday = [
    _row(g.results, "mean_return").value for g in groups.values()
  ]
  assert sum(per_weekday) / len(per_weekday) == pytest.approx(0.01)


# ===========================================================================
# 3. close_to_close: prior-close basis, first day excluded
# ===========================================================================

# Closes: day0=100(excluded), day1=110, day2=105, day3=105, day4=90
_CTC_DAYS = [
  {"date": "2024-01-01", "session_open": 100.0, "session_close": 100.0},  # excluded
  {"date": "2024-01-02", "session_open": 100.0, "session_close": 110.0},  # +0.10 G
  {"date": "2024-01-03", "session_open": 100.0, "session_close": 105.0},  # (105-110)/110 R
  {"date": "2024-01-04", "session_open": 100.0, "session_close": 105.0},  # 0.0 G (equal)
  {"date": "2024-01-05", "session_open": 100.0, "session_close": 90.0},   # (90-105)/105 R
]


def test_close_to_close_excludes_first_day() -> None:
  day_table = _ctc_stat().build_day_table(make_candles(_CTC_DAYS))
  assert len(day_table) == 4
  first_date = pd.Timestamp("2024-01-01", tz=_NY).normalize()
  assert first_date not in day_table.index


def test_close_to_close_return_formula() -> None:
  """day1 return uses the prior close: (110-100)/100 = 0.10."""
  day_table = _ctc_stat().build_day_table(make_candles(_CTC_DAYS)).sort_index()
  day1 = pd.Timestamp("2024-01-02", tz=_NY).normalize()
  assert day_table.loc[day1, "return_pct"] == pytest.approx(0.10)


def test_close_to_close_equal_close_is_green() -> None:
  """day3 close (105) equals day2 close (105) → return 0.0 → green."""
  day_table = _ctc_stat().build_day_table(make_candles(_CTC_DAYS)).sort_index()
  day3 = pd.Timestamp("2024-01-04", tz=_NY).normalize()
  assert day_table.loc[day3, "return_pct"] == pytest.approx(0.0)
  assert bool(day_table.loc[day3, "day_green"])


def test_close_to_close_counts() -> None:
  """green=2 (day1,day3), red=2 (day2,day4), total=4."""
  result = _ctc_stat().compute(make_candles(_CTC_DAYS))
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 4
  assert _overall(result, "green_day").count == 2
  assert _overall(result, "red_day").count == 2


# ===========================================================================
# 4. Baseline: random sign → mean ≈ 0, deterministic
# ===========================================================================

def _make_200_day_spec() -> list[dict]:
  """200 weekdays with alternating ±10% closes (open_to_close)."""
  dates: list[str] = []
  d = pd.Timestamp("2020-01-02", tz=_NY)
  while len(dates) < 200:
    if d.weekday() < 5:
      dates.append(d.strftime("%Y-%m-%d"))
    d += pd.Timedelta(days=1)
  return [
    {
      "date": date,
      "session_open": 100.0,
      "session_close": 110.0 if i % 2 == 0 else 90.0,
    }
    for i, date in enumerate(dates)
  ]


def test_baseline_mean_return_near_zero() -> None:
  """Random-sign baseline drives the average return toward 0."""
  rows = _otc_stat().baseline(make_candles(_make_200_day_spec()), seed=42)
  mr = _row(rows, "mean_return")
  assert mr.value == pytest.approx(0.0, abs=0.03)


def test_baseline_green_prob_near_half() -> None:
  """Random-sign baseline yields ~50% green days."""
  rows = _otc_stat().baseline(make_candles(_make_200_day_spec()), seed=42)
  assert _row(rows, "green_day").probability == pytest.approx(0.5, abs=0.12)


def test_baseline_deterministic() -> None:
  """Same seed => identical baseline values."""
  df = make_candles(_make_200_day_spec())
  stat = _otc_stat()
  rows_a = stat.baseline(df, seed=7)
  rows_b = stat.baseline(df, seed=7)
  for a, b in zip(rows_a, rows_b):
    assert a.outcome == b.outcome
    assert a.probability == pytest.approx(b.probability)
    if a.value is None:
      assert b.value is None
    else:
      assert a.value == pytest.approx(b.value)


def test_baseline_embedded_in_compute() -> None:
  """compute() must embed baselines: probability rows get baseline_prob/n,
  magnitude rows get value_baseline."""
  result = _otc_stat().compute(make_candles(_make_200_day_spec()))
  for row in result.instruments["NQ"]["daily"].results:
    if row.outcome in ("green_day", "red_day"):
      assert row.baseline_n > 0
      assert row.value is None
      assert row.value_baseline is None
    else:
      assert row.value_baseline is not None


# ===========================================================================
# 5. Pending exclusion
# ===========================================================================

def test_pending_day_excluded() -> None:
  stat = _otc_stat()
  base_df = make_candles(_OTC_DAYS)
  total_base = stat.compute(base_df).instruments["NQ"]["daily"].total_samples

  truncated = _make_truncated_day("2024-01-15")
  df = pd.concat([base_df, truncated], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
  total_with = stat.compute(df).instruments["NQ"]["daily"].total_samples
  assert total_with == total_base == 10


def test_pending_day_absent_from_day_table() -> None:
  day_table = _otc_stat().build_day_table(_make_truncated_day("2024-01-15"))
  pending = pd.Timestamp("2024-01-15", tz=_NY).normalize()
  assert pending not in day_table.index


# ===========================================================================
# 6. Edge cases
# ===========================================================================

def _empty_df() -> pd.DataFrame:
  df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
  df["timestamp"] = pd.array([], dtype="datetime64[ns, America/New_York]")
  return df


def test_empty_dataframe_no_crash() -> None:
  """Empty input: 0 samples, empty range, zero/None metrics, no division error."""
  result = _otc_stat().compute(_empty_df())
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 0
  assert tf.data_range == []
  for row in tf.results:
    assert row.count == 0
    assert row.total == 0
    assert row.probability == pytest.approx(0.0)
    if row.outcome in ("green_day", "red_day"):
      assert row.value is None
    else:
      assert row.value == pytest.approx(0.0)


def test_empty_dataframe_no_slice_groups() -> None:
  result = _otc_stat().compute(_empty_df())
  assert result.instruments["NQ"]["daily"].slices["weekday"].groups == {}


def test_all_green_days() -> None:
  """All-green dataset: P(green)=1.0, mean_red_move falls back to 0.0."""
  days = [
    {"date": d["date"], "session_open": 100.0, "session_close": 120.0}
    for d in _OTC_DAYS
  ]
  result = _otc_stat().compute(make_candles(days))
  assert _overall(result, "green_day").probability == pytest.approx(1.0)
  assert _overall(result, "red_day").probability == pytest.approx(0.0)
  assert _overall(result, "mean_red_move").value == pytest.approx(0.0)
  assert _overall(result, "mean_red_move").count == 0
  assert _overall(result, "mean_green_move").value == pytest.approx(0.20)


def test_invalid_performance_raises() -> None:
  with pytest.raises(ValueError, match="Unsupported performance"):
    PerformanceByWeekday(instrument="NQ", config=_TEST_CONFIG, performance="tick_to_tick")


def test_default_performance_is_close_to_close() -> None:
  stat = PerformanceByWeekday(instrument="NQ", config=_TEST_CONFIG)
  assert stat.performance == "close_to_close"
  assert stat.timeframe == "daily"


# ===========================================================================
# 7. i18n
# ===========================================================================

def test_i18n_title_and_definition() -> None:
  result = _otc_stat().compute(_empty_df())
  assert result.title.en != "" and result.title.fr != ""
  assert result.definition.en != "" and result.definition.fr != ""


def test_i18n_french_accent_in_definition() -> None:
  """French definition must contain an accented word."""
  result = _otc_stat().compute(_empty_df())
  assert "rendement" in result.definition.fr


def test_i18n_labels_have_en_and_fr() -> None:
  result = _otc_stat().compute(_empty_df())
  for mapping in (result.labels.conditions, result.labels.outcomes):
    for key, i18n in mapping.items():
      assert i18n.en != "", f"{key}.en empty"
      assert i18n.fr != "", f"{key}.fr empty"
  assert set(result.labels.outcomes) == {
    "mean_return", "green_day", "red_day", "mean_green_move", "mean_red_move",
  }
  assert "any_day" in result.labels.conditions


def test_labels_dimensions_contains_weekday() -> None:
  result = _otc_stat().compute(make_candles(_OTC_DAYS))
  assert "weekday" in result.labels.dimensions
  assert result.labels.dimensions["weekday"].en != ""
  assert result.labels.dimensions["weekday"].fr != ""


# ===========================================================================
# 8. write_results round-trip
# ===========================================================================

def test_stat_name() -> None:
  assert _otc_stat().compute(_empty_df()).stat_name == "performance_weekday"


def test_write_results_round_trip(tmp_path: Path) -> None:
  result = _otc_stat().compute(make_candles(_OTC_DAYS))
  written = write_results(result, results_dir=tmp_path)

  assert written.name == "performance_weekday.json"
  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)

  tf = validated.instruments["NQ"]["daily"]
  assert tf.total_samples == 10
  monday = tf.slices["weekday"].groups["monday"]
  mr = next(r for r in monday.results if r.outcome == "mean_return")
  assert mr.value == pytest.approx(0.15)
