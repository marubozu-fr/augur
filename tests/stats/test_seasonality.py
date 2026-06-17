"""Tests for stats.seasonality.standard.

All data is synthetic — no real market files required.
Expected values are hand-calculated before each assertion.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.seasonality.standard import Seasonality

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
# open_to_close monthly dataset: one trading day per month for 2 years.
#
# Each month's single day collapses to that month's period (period_open=100,
# period_close as below). open_to_close return = (close - 100) / 100.
#
# Year 2023:  Jan 120(+0.20) Feb 80(-0.20) Mar 110(+0.10)
# Year 2024:  Jan 110(+0.10) Feb 90(-0.10) Mar 80(-0.20)
#
# month-of-year averages (open_to_close):
#   January : mean(.20, .10) = 0.15  | green 2/2
#   February: mean(-.20, -.10) = -0.15 | red 2/2
#   March   : mean(.10, -.20) = -0.05 | green 1/2
# overall (6 periods): returns [.20,-.20,.10,.10,-.10,-.20] sum -0.10 mean -0.0166..
#   green = 3 (>=0), red = 3 → P(green)=0.5
#   mean_green = mean(.20,.10,.10) = 0.1333..  mean_red = mean(-.20,-.10,-.20) = -0.1666..
# ---------------------------------------------------------------------------

_OTC_MONTHLY_DAYS = [
  {"date": "2023-01-16", "session_open": 100.0, "session_close": 120.0},  # Jan +0.20
  {"date": "2023-02-15", "session_open": 100.0, "session_close": 80.0},   # Feb -0.20
  {"date": "2023-03-15", "session_open": 100.0, "session_close": 110.0},  # Mar +0.10
  {"date": "2024-01-16", "session_open": 100.0, "session_close": 110.0},  # Jan +0.10
  {"date": "2024-02-15", "session_open": 100.0, "session_close": 90.0},   # Feb -0.10
  {"date": "2024-03-15", "session_open": 100.0, "session_close": 80.0},   # Mar -0.20
]


def _otc(granularity: str = "monthly") -> Seasonality:
  return Seasonality(
    instrument="NQ", config=_TEST_CONFIG, granularity=granularity, performance="open_to_close"
  )


def _ctc(granularity: str = "monthly") -> Seasonality:
  return Seasonality(
    instrument="NQ", config=_TEST_CONFIG, granularity=granularity, performance="close_to_close"
  )


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


# ===========================================================================
# 1. open_to_close monthly: overall magnitudes and counts
# ===========================================================================

def test_overall_five_rows() -> None:
  """Exactly five outcome rows under the single any_period condition."""
  result = _otc().compute(make_candles(_OTC_MONTHLY_DAYS))
  rows = result.instruments["NQ"]["monthly"].results
  assert len(rows) == 5
  assert {r.outcome for r in rows} == {
    "mean_return", "green_period", "red_period", "mean_green_move", "mean_red_move",
  }
  assert {r.condition for r in rows} == {"any_period"}


def test_overall_mean_return() -> None:
  """Overall average return = -0.10/6, in `value`."""
  result = _otc().compute(make_candles(_OTC_MONTHLY_DAYS))
  mr = _overall(result, "monthly", "mean_return")
  assert mr.value == pytest.approx(-0.10 / 6)
  assert mr.count == 6
  assert mr.total == 6
  assert mr.probability == pytest.approx(0.0)


def test_overall_green_red_counts() -> None:
  """3 green + 3 red → P(green)=0.5, counts carried in count/total."""
  result = _otc().compute(make_candles(_OTC_MONTHLY_DAYS))
  green = _overall(result, "monthly", "green_period")
  red = _overall(result, "monthly", "red_period")
  assert green.count == 3
  assert green.total == 6
  assert green.probability == pytest.approx(0.5)
  assert red.count == 3
  assert red.probability == pytest.approx(0.5)
  assert green.value is None
  assert red.value is None


def test_overall_mean_moves() -> None:
  """mean_green_move = 0.40/3, mean_red_move = -0.50/3."""
  result = _otc().compute(make_candles(_OTC_MONTHLY_DAYS))
  mg = _overall(result, "monthly", "mean_green_move")
  mr = _overall(result, "monthly", "mean_red_move")
  assert mg.value == pytest.approx((0.20 + 0.10 + 0.10) / 3)
  assert mg.count == 3
  assert mr.value == pytest.approx((-0.20 - 0.10 - 0.20) / 3)
  assert mr.count == 3


def test_overall_data_range() -> None:
  result = _otc().compute(make_candles(_OTC_MONTHLY_DAYS))
  assert result.instruments["NQ"]["monthly"].data_range == ["2023-01-16", "2024-03-15"]


# ===========================================================================
# 2. month-of-year slice magnitudes
# ===========================================================================

def test_month_slice_three_groups() -> None:
  result = _otc().compute(make_candles(_OTC_MONTHLY_DAYS))
  groups = result.instruments["NQ"]["monthly"].slices["month_of_year"].groups
  assert set(groups.keys()) == {"january", "february", "march"}
  for g in groups.values():
    assert g.total_samples == 2


def test_month_mean_returns() -> None:
  """Per-month average return across the two years."""
  result = _otc().compute(make_candles(_OTC_MONTHLY_DAYS))
  expected = {
    "january": 0.15,
    "february": -0.15,
    "march": -0.05,
  }
  for month_key, exp in expected.items():
    mr = _row(_month_rows(result, month_key), "mean_return")
    assert mr.value == pytest.approx(exp), f"{month_key}: expected {exp}, got {mr.value}"


def test_month_green_probabilities() -> None:
  """Per-month P(green): Jan=1.0, Feb=0.0, Mar=0.5."""
  result = _otc().compute(make_candles(_OTC_MONTHLY_DAYS))
  expected = {"january": 1.0, "february": 0.0, "march": 0.5}
  for month_key, exp in expected.items():
    green = _row(_month_rows(result, month_key), "green_period")
    assert green.probability == pytest.approx(exp), f"{month_key}: got {green.probability}"


# ===========================================================================
# 3. close_to_close monthly: prior-close basis, first period excluded
# ===========================================================================

# One day per month; close_to_close uses the PREVIOUS month's close.
# Closes: 2023-01=100 (excluded), 2023-02=110, 2023-03=105, 2024-01=105, 2024-02=90
_CTC_MONTHLY_DAYS = [
  {"date": "2023-01-16", "session_open": 100.0, "session_close": 100.0},  # excluded
  {"date": "2023-02-15", "session_open": 100.0, "session_close": 110.0},  # (110-100)/100 = +0.10 G
  {"date": "2023-03-15", "session_open": 100.0, "session_close": 105.0},  # (105-110)/110 R
  {"date": "2024-01-16", "session_open": 100.0, "session_close": 105.0},  # 0.0 G (equal)
  {"date": "2024-02-15", "session_open": 100.0, "session_close": 90.0},   # (90-105)/105 R
]


def test_close_to_close_excludes_first_period() -> None:
  day_table = _ctc().build_day_table(make_candles(_CTC_MONTHLY_DAYS))
  assert len(day_table) == 4
  first_date = pd.Timestamp("2023-01-16", tz=_NY).normalize()
  assert first_date not in day_table.index


def test_close_to_close_return_formula() -> None:
  """Feb 2023 return uses the prior month's close: (110-100)/100 = 0.10."""
  day_table = _ctc().build_day_table(make_candles(_CTC_MONTHLY_DAYS)).sort_index()
  feb = pd.Timestamp("2023-02-15", tz=_NY).normalize()
  assert day_table.loc[feb, "return_pct"] == pytest.approx(0.10)


def test_close_to_close_equal_close_is_green() -> None:
  """Jan 2024 close (105) equals Mar 2023 close (105) → return 0.0 → green."""
  day_table = _ctc().build_day_table(make_candles(_CTC_MONTHLY_DAYS)).sort_index()
  jan24 = pd.Timestamp("2024-01-16", tz=_NY).normalize()
  assert day_table.loc[jan24, "return_pct"] == pytest.approx(0.0)
  assert bool(day_table.loc[jan24, "period_green"])


def test_close_to_close_counts() -> None:
  """green=2, red=2, total=4 resolved periods."""
  result = _ctc().compute(make_candles(_CTC_MONTHLY_DAYS))
  tf = result.instruments["NQ"]["monthly"]
  assert tf.total_samples == 4
  assert _overall(result, "monthly", "green_period").count == 2
  assert _overall(result, "monthly", "red_period").count == 2


# ===========================================================================
# 4. Weekly granularity: period aggregation and week-of-year slice
# ===========================================================================

# Two days within the same ISO week aggregate into one weekly period:
# period_open = first day's open, period_close = last day's close.
# 2024-01-01 (Mon) .. 2024-01-05 (Fri) is ISO week 1 of 2024.
# 2024-01-08 (Mon) .. 2024-01-12 (Fri) is ISO week 2 of 2024.
_OTC_WEEKLY_DAYS = [
  {"date": "2024-01-01", "session_open": 100.0, "session_close": 105.0},  # week 1 open
  {"date": "2024-01-05", "session_open": 102.0, "session_close": 120.0},  # week 1 close -> +0.20
  {"date": "2024-01-08", "session_open": 100.0, "session_close": 101.0},  # week 2 open
  {"date": "2024-01-12", "session_open": 103.0, "session_close": 80.0},   # week 2 close -> -0.20
]


def test_weekly_aggregates_days_into_periods() -> None:
  """Two weeks → two periods; period_open/close span the week."""
  day_table = _otc("weekly").build_day_table(make_candles(_OTC_WEEKLY_DAYS)).sort_index()
  assert len(day_table) == 2
  week1 = pd.Timestamp("2024-01-01", tz=_NY).normalize()
  assert day_table.loc[week1, "period_open"] == pytest.approx(100.0)
  assert day_table.loc[week1, "period_close"] == pytest.approx(120.0)
  assert day_table.loc[week1, "return_pct"] == pytest.approx(0.20)


def test_weekly_week_of_year_slice() -> None:
  """The weekly granularity slices by week-of-year (ISO week 1 and 2)."""
  result = _otc("weekly").compute(make_candles(_OTC_WEEKLY_DAYS))
  groups = result.instruments["NQ"]["weekly"].slices["week_of_year"].groups
  assert set(groups.keys()) == {"w01", "w02"}
  w1 = _row(groups["w01"].results, "mean_return")
  w2 = _row(groups["w02"].results, "mean_return")
  assert w1.value == pytest.approx(0.20)
  assert w2.value == pytest.approx(-0.20)


# ===========================================================================
# 5. Baseline: random sign → mean ≈ 0, deterministic
# ===========================================================================

def _make_200_month_spec() -> list[dict]:
  """~200 monthly periods (one day each) with alternating ±10% closes."""
  specs: list[dict] = []
  d = pd.Timestamp("2000-01-15", tz=_NY)
  for i in range(200):
    specs.append({
      "date": d.strftime("%Y-%m-%d"),
      "session_open": 100.0,
      "session_close": 110.0 if i % 2 == 0 else 90.0,
    })
    # advance roughly one month, staying mid-month so each lands in a fresh month
    month = d.month + 1
    year = d.year + (1 if month > 12 else 0)
    month = month - 12 if month > 12 else month
    d = pd.Timestamp(f"{year}-{month:02d}-15", tz=_NY)
  return specs


def test_baseline_mean_return_near_zero() -> None:
  """Random-sign baseline drives the average return toward 0."""
  rows = _otc().baseline(make_candles(_make_200_month_spec()), seed=42)
  mr = _row(rows, "mean_return")
  assert mr.value == pytest.approx(0.0, abs=0.03)


def test_baseline_green_prob_near_half() -> None:
  """Random-sign baseline yields ~50% green periods."""
  rows = _otc().baseline(make_candles(_make_200_month_spec()), seed=42)
  assert _row(rows, "green_period").probability == pytest.approx(0.5, abs=0.12)


def test_baseline_deterministic() -> None:
  """Same seed => identical baseline values."""
  df = make_candles(_make_200_month_spec())
  stat = _otc()
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
  result = _otc().compute(make_candles(_make_200_month_spec()))
  for row in result.instruments["NQ"]["monthly"].results:
    if row.outcome in ("green_period", "red_period"):
      assert row.baseline_n > 0
      assert row.value is None
      assert row.value_baseline is None
    else:
      assert row.value_baseline is not None


# ===========================================================================
# 6. Pending exclusion
# ===========================================================================

def test_pending_day_excluded() -> None:
  """A truncated (early-close) day is excluded from its period's aggregation."""
  stat = _otc()
  base_df = make_candles(_OTC_MONTHLY_DAYS)
  total_base = stat.compute(base_df).instruments["NQ"]["monthly"].total_samples

  truncated = _make_truncated_day("2024-04-15")
  df = pd.concat([base_df, truncated], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
  total_with = stat.compute(df).instruments["NQ"]["monthly"].total_samples
  # The truncated April day is unresolved → no April period is created.
  assert total_with == total_base == 6


# ===========================================================================
# 7. Edge cases
# ===========================================================================

def _empty_df() -> pd.DataFrame:
  df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
  df["timestamp"] = pd.array([], dtype="datetime64[ns, America/New_York]")
  return df


def test_empty_dataframe_no_crash() -> None:
  """Empty input: 0 samples, empty range, zero/None metrics, no division error."""
  result = _otc().compute(_empty_df())
  tf = result.instruments["NQ"]["monthly"]
  assert tf.total_samples == 0
  assert tf.data_range == []
  for row in tf.results:
    assert row.count == 0
    assert row.total == 0
    assert row.probability == pytest.approx(0.0)
    if row.outcome in ("green_period", "red_period"):
      assert row.value is None
    else:
      assert row.value == pytest.approx(0.0)


def test_empty_dataframe_no_slice_groups() -> None:
  result = _otc().compute(_empty_df())
  assert result.instruments["NQ"]["monthly"].slices["month_of_year"].groups == {}


def test_invalid_granularity_raises() -> None:
  with pytest.raises(ValueError, match="Unsupported granularity"):
    Seasonality(instrument="NQ", config=_TEST_CONFIG, granularity="daily")


def test_invalid_performance_raises() -> None:
  with pytest.raises(ValueError, match="Unsupported performance"):
    Seasonality(instrument="NQ", config=_TEST_CONFIG, performance="tick_to_tick")


def test_default_granularity_and_performance() -> None:
  stat = Seasonality(instrument="NQ", config=_TEST_CONFIG)
  assert stat.granularity == "monthly"
  assert stat.timeframe == "monthly"
  assert stat.performance == "close_to_close"


def test_monthly_declares_month_slice_weekly_declares_week_slice() -> None:
  monthly = Seasonality(instrument="NQ", config=_TEST_CONFIG, granularity="monthly")
  weekly = Seasonality(instrument="NQ", config=_TEST_CONFIG, granularity="weekly")
  assert monthly.slices[0].name == "month_of_year"
  assert weekly.slices[0].name == "week_of_year"


# ===========================================================================
# 8. i18n
# ===========================================================================

def test_i18n_title_and_definition() -> None:
  result = _otc().compute(_empty_df())
  assert result.title.en != "" and result.title.fr != ""
  assert result.definition.en != "" and result.definition.fr != ""


def test_i18n_french_accent_in_definition() -> None:
  """French definition must contain an accented word."""
  result = _otc().compute(_empty_df())
  assert "rendement" in result.definition.fr


def test_i18n_labels_have_en_and_fr() -> None:
  result = _otc().compute(_empty_df())
  for mapping in (result.labels.conditions, result.labels.outcomes):
    for key, i18n in mapping.items():
      assert i18n.en != "", f"{key}.en empty"
      assert i18n.fr != "", f"{key}.fr empty"
  assert set(result.labels.outcomes) == {
    "mean_return", "green_period", "red_period", "mean_green_move", "mean_red_move",
  }
  assert "any_period" in result.labels.conditions


def test_labels_dimensions_contains_month_of_year() -> None:
  result = _otc().compute(make_candles(_OTC_MONTHLY_DAYS))
  assert "month_of_year" in result.labels.dimensions
  assert result.labels.dimensions["month_of_year"].en != ""
  assert result.labels.dimensions["month_of_year"].fr != ""


# ===========================================================================
# 9. write_results round-trip
# ===========================================================================

def test_stat_name() -> None:
  assert _otc().compute(_empty_df()).stat_name == "seasonality"


def test_write_results_round_trip(tmp_path: Path) -> None:
  result = _otc().compute(make_candles(_OTC_MONTHLY_DAYS))
  written = write_results(result, results_dir=tmp_path)

  assert written.name == "seasonality.json"
  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)

  tf = validated.instruments["NQ"]["monthly"]
  assert tf.total_samples == 6
  january = tf.slices["month_of_year"].groups["january"]
  mr = next(r for r in january.results if r.outcome == "mean_return")
  assert mr.value == pytest.approx(0.15)
