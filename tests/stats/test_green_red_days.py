"""Tests for stats.green_red_days.by_weekday.

All data is synthetic — no real market files required.
Expected values are hand-calculated before each assertion.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.green_red_days.by_weekday import GreenRedDaysByWeekday

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
  """Build a multi-day 1-min OHLCV DataFrame from a list of day specs.

  Each spec dict must have: date (str), session_open (float), session_close (float).
  """
  frames = [_make_day(d["date"], d["session_open"], d["session_close"]) for d in days]
  df = pd.concat(frames, ignore_index=True)
  return df.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# open_to_close dataset: 10 days, 2 per weekday, session_open=100 always.
#
#   Mon1 120 (G), Tue1 80 (R), Wed1 120 (G), Thu1 120 (G), Fri1 80 (R),
#   Mon2 120 (G), Tue2 80 (R), Wed2 80 (R),  Thu2 120 (G), Fri2 80 (R)
#
# Overall: green = 5 (Mon1,Wed1,Thu1,Mon2,Thu2), red = 5 → P(green)=0.5
# Weekday P(green): Mon=1.0, Tue=0.0, Wed=0.5, Thu=1.0, Fri=0.0
# ---------------------------------------------------------------------------

_OTC_DAYS = [
  {"date": "2024-01-01", "session_open": 100.0, "session_close": 120.0},  # Mon G
  {"date": "2024-01-02", "session_open": 100.0, "session_close": 80.0},   # Tue R
  {"date": "2024-01-03", "session_open": 100.0, "session_close": 120.0},  # Wed G
  {"date": "2024-01-04", "session_open": 100.0, "session_close": 120.0},  # Thu G
  {"date": "2024-01-05", "session_open": 100.0, "session_close": 80.0},   # Fri R
  {"date": "2024-01-08", "session_open": 100.0, "session_close": 120.0},  # Mon G
  {"date": "2024-01-09", "session_open": 100.0, "session_close": 80.0},   # Tue R
  {"date": "2024-01-10", "session_open": 100.0, "session_close": 80.0},   # Wed R
  {"date": "2024-01-11", "session_open": 100.0, "session_close": 120.0},  # Thu G
  {"date": "2024-01-12", "session_open": 100.0, "session_close": 80.0},   # Fri R
]


def _otc_stat() -> GreenRedDaysByWeekday:
  return GreenRedDaysByWeekday(
    instrument="NQ", config=_TEST_CONFIG, performance="open_to_close"
  )


def _ctc_stat() -> GreenRedDaysByWeekday:
  return GreenRedDaysByWeekday(
    instrument="NQ", config=_TEST_CONFIG, performance="close_to_close"
  )


def _get_row(result: StatRunResult, tf: str, outcome: str) -> StatResultRow:
  for row in result.instruments["NQ"][tf].results:
    if row.condition == "any_day" and row.outcome == outcome:
      return row
  raise KeyError(outcome)


# ===========================================================================
# 1. open_to_close: overall exact counts
# ===========================================================================

def test_open_to_close_overall_counts() -> None:
  """5 green + 5 red days → P(green)=0.5, P(red)=0.5."""
  stat = _otc_stat()
  result = stat.compute(make_candles(_OTC_DAYS))

  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 10

  green = _get_row(result, "daily", "green_day")
  red = _get_row(result, "daily", "red_day")
  assert green.count == 5
  assert green.total == 10
  assert green.probability == pytest.approx(0.5)
  assert red.count == 5
  assert red.probability == pytest.approx(0.5)
  # Outcomes must partition the total
  assert green.count + red.count == tf.total_samples


def test_open_to_close_two_rows_only() -> None:
  """Exactly two outcome rows (green_day, red_day) under one condition."""
  stat = _otc_stat()
  result = stat.compute(make_candles(_OTC_DAYS))
  rows = result.instruments["NQ"]["daily"].results

  assert len(rows) == 2
  assert {(r.condition, r.outcome) for r in rows} == {
    ("any_day", "green_day"),
    ("any_day", "red_day"),
  }


def test_open_to_close_data_range() -> None:
  """data_range must span first to last date."""
  stat = _otc_stat()
  result = stat.compute(make_candles(_OTC_DAYS))
  assert result.instruments["NQ"]["daily"].data_range == ["2024-01-01", "2024-01-12"]


def test_close_equals_open_counts_as_green() -> None:
  """Green = close >= open. A flat day (close == open) is green."""
  stat = _otc_stat()
  result = stat.compute(make_candles([
    {"date": "2024-02-01", "session_open": 100.0, "session_close": 100.0},
  ]))
  green = _get_row(result, "daily", "green_day")
  assert green.count == 1
  assert green.total == 1


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


# ===========================================================================
# 2. close_to_close: prior-close basis, first day excluded
# ===========================================================================

# Closes: day0=100, day1=110(G), day2=105(R), day3=105(G, equal), day4=90(R)
# First day excluded (no prior close) → 4 resolved days, green=2, red=2.
_CTC_DAYS = [
  {"date": "2024-01-01", "session_open": 100.0, "session_close": 100.0},  # excluded
  {"date": "2024-01-02", "session_open": 100.0, "session_close": 110.0},  # G (110>=100)
  {"date": "2024-01-03", "session_open": 100.0, "session_close": 105.0},  # R (105>=110 no)
  {"date": "2024-01-04", "session_open": 100.0, "session_close": 105.0},  # G (105>=105)
  {"date": "2024-01-05", "session_open": 100.0, "session_close": 90.0},   # R (90>=105 no)
]


def test_close_to_close_excludes_first_day() -> None:
  """First resolved day has no prior close → excluded from the day table."""
  stat = _ctc_stat()
  day_table = stat.build_day_table(make_candles(_CTC_DAYS))

  assert len(day_table) == 4
  first_date = pd.Timestamp("2024-01-01", tz=_NY).normalize()
  assert first_date not in day_table.index


def test_close_to_close_overall_counts() -> None:
  """green=2 (day1,day3), red=2 (day2,day4), total=4 → P(green)=0.5."""
  stat = _ctc_stat()
  result = stat.compute(make_candles(_CTC_DAYS))

  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 4

  green = _get_row(result, "daily", "green_day")
  red = _get_row(result, "daily", "red_day")
  assert green.count == 2
  assert red.count == 2
  assert green.probability == pytest.approx(0.5)


def test_close_to_close_equal_close_is_green() -> None:
  """day3 close (105) equals day2 close (105) → counts as green.

  If equality were treated as red, green would be 1 instead of 2.
  """
  stat = _ctc_stat()
  day_table = stat.build_day_table(make_candles(_CTC_DAYS)).sort_index()
  day3 = pd.Timestamp("2024-01-04", tz=_NY).normalize()
  assert bool(day_table.loc[day3, "day_green"])


def test_close_to_close_prev_skips_unresolved_day() -> None:
  """prev_session_close references the previous RESOLVED day, not a pending one.

  An early-close day inserted between two full days must not become the
  reference close: the later full day compares against the earlier full day.
  """
  stat = _ctc_stat()
  full_days = [
    {"date": "2024-01-02", "session_open": 100.0, "session_close": 100.0},  # excluded (first)
    {"date": "2024-01-03", "session_open": 100.0, "session_close": 200.0},  # prev close = 100
    {"date": "2024-01-05", "session_open": 100.0, "session_close": 150.0},  # prev close = 200 → R
  ]
  df = make_candles(full_days)
  # Insert a truncated (unresolved) day between 01-03 and 01-05.
  truncated = _make_truncated_day("2024-01-04")
  df = pd.concat([df, truncated], ignore_index=True).sort_values("timestamp").reset_index(drop=True)

  day_table = stat.build_day_table(df).sort_index()
  day_0105 = pd.Timestamp("2024-01-05", tz=_NY).normalize()
  # prev resolved close is 01-03's close (200), not the truncated day.
  assert day_table.loc[day_0105, "prev_session_close"] == pytest.approx(200.0)
  assert not bool(day_table.loc[day_0105, "day_green"])  # 150 < 200


# ===========================================================================
# 3. Weekday slice (open_to_close dataset)
# ===========================================================================

def test_weekday_slice_present() -> None:
  """The stat declares slices=('weekday',) → slices['weekday'] is populated."""
  stat = _otc_stat()
  result = stat.compute(make_candles(_OTC_DAYS))

  tf = result.instruments["NQ"]["daily"]
  assert "weekday" in tf.slices
  assert tf.slices["weekday"].dimension == "weekday"


def test_weekday_slice_five_groups() -> None:
  """Mon–Fri all present (2 each) → 5 groups."""
  stat = _otc_stat()
  result = stat.compute(make_candles(_OTC_DAYS))
  groups = result.instruments["NQ"]["daily"].slices["weekday"].groups

  assert set(groups.keys()) == {"monday", "tuesday", "wednesday", "thursday", "friday"}
  for g in groups.values():
    assert g.total_samples == 2


def test_weekday_slice_green_probabilities() -> None:
  """Per-weekday P(green): Mon=1.0, Tue=0.0, Wed=0.5, Thu=1.0, Fri=0.0."""
  stat = _otc_stat()
  result = stat.compute(make_candles(_OTC_DAYS))
  groups = result.instruments["NQ"]["daily"].slices["weekday"].groups

  expected = {
    "monday": 1.0,
    "tuesday": 0.0,
    "wednesday": 0.5,
    "thursday": 1.0,
    "friday": 0.0,
  }
  for day_key, exp_prob in expected.items():
    green = next(r for r in groups[day_key].results if r.outcome == "green_day")
    assert green.probability == pytest.approx(exp_prob), (
      f"{day_key}: expected P(green)={exp_prob}, got {green.probability}"
    )


def test_weekday_slice_totals_sum_to_overall() -> None:
  """Sum of per-weekday green_day totals equals the overall total."""
  stat = _otc_stat()
  result = stat.compute(make_candles(_OTC_DAYS))
  tf = result.instruments["NQ"]["daily"]
  groups = tf.slices["weekday"].groups

  overall = _get_row(result, "daily", "green_day").total
  per_weekday = sum(
    next(r for r in g.results if r.outcome == "green_day").total
    for g in groups.values()
  )
  assert per_weekday == overall == 10


# ===========================================================================
# 4. Baseline ≈ 50% and deterministic
# ===========================================================================

def _make_200_day_spec() -> list[dict]:
  """200 weekdays with alternating green/red closes (open_to_close)."""
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
      "session_close": 120.0 if i % 2 == 0 else 80.0,
    }
    for i, date in enumerate(dates)
  ]


def test_baseline_approx_fifty_percent() -> None:
  """With seed=42 over 200 days, baseline_prob values should be near 0.5."""
  stat = _otc_stat()
  rows = stat.baseline(make_candles(_make_200_day_spec()), seed=42)
  for row in rows:
    assert row.probability == pytest.approx(0.5, abs=0.12)


def test_baseline_deterministic() -> None:
  """Same seed => identical baseline probabilities."""
  stat = _otc_stat()
  df = make_candles(_make_200_day_spec())
  rows_a = stat.baseline(df, seed=7)
  rows_b = stat.baseline(df, seed=7)
  for a, b in zip(rows_a, rows_b):
    assert a.outcome == b.outcome
    assert a.probability == pytest.approx(b.probability)


def test_baseline_embedded_in_compute() -> None:
  """compute() must embed baseline_prob (not 0.0) into result rows."""
  stat = _otc_stat()
  result = stat.compute(make_candles(_make_200_day_spec()))
  for row in result.instruments["NQ"]["daily"].results:
    assert row.baseline_prob != 0.0
    assert row.baseline_n > 0


# ===========================================================================
# 5. Pending exclusion
# ===========================================================================

def test_pending_day_excluded() -> None:
  """A truncated day must not change total_samples (open_to_close)."""
  stat = _otc_stat()
  base_df = make_candles(_OTC_DAYS)
  total_base = stat.compute(base_df).instruments["NQ"]["daily"].total_samples

  truncated = _make_truncated_day("2024-01-15")
  df = pd.concat([base_df, truncated], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
  total_with = stat.compute(df).instruments["NQ"]["daily"].total_samples

  assert total_with == total_base == 10


def test_pending_day_absent_from_day_table() -> None:
  """The truncated day's date must not appear in build_day_table's index."""
  stat = _otc_stat()
  day_table = stat.build_day_table(_make_truncated_day("2024-01-15"))
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
  """Empty input returns 0 total_samples, empty data_range, zero probabilities."""
  stat = _otc_stat()
  result = stat.compute(_empty_df())
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 0
  assert tf.data_range == []
  for row in tf.results:
    assert row.probability == pytest.approx(0.0)
    assert row.count == 0
    assert row.total == 0


def test_empty_dataframe_no_slice_groups() -> None:
  """Empty input must produce an empty weekday slice (no groups)."""
  stat = _otc_stat()
  result = stat.compute(_empty_df())
  assert result.instruments["NQ"]["daily"].slices["weekday"].groups == {}


def test_all_green_days() -> None:
  """All-green dataset: P(green)=1.0, P(red)=0.0, no ZeroDivisionError."""
  stat = _otc_stat()
  days = [
    {"date": d["date"], "session_open": 100.0, "session_close": 120.0}
    for d in _OTC_DAYS
  ]
  result = stat.compute(make_candles(days))
  assert _get_row(result, "daily", "green_day").probability == pytest.approx(1.0)
  assert _get_row(result, "daily", "red_day").probability == pytest.approx(0.0)


def test_invalid_performance_raises() -> None:
  """Unsupported performance mode must raise ValueError."""
  with pytest.raises(ValueError, match="Unsupported performance"):
    GreenRedDaysByWeekday(instrument="NQ", config=_TEST_CONFIG, performance="tick_to_tick")


def test_default_performance_is_close_to_close() -> None:
  """Default performance mode must be close_to_close."""
  stat = GreenRedDaysByWeekday(instrument="NQ", config=_TEST_CONFIG)
  assert stat.performance == "close_to_close"
  assert stat.timeframe == "daily"


# ===========================================================================
# 7. i18n
# ===========================================================================

def test_i18n_title_and_definition() -> None:
  """Title and definition must carry non-empty en and fr."""
  stat = _otc_stat()
  result = stat.compute(_empty_df())
  assert result.title.en != "" and result.title.fr != ""
  assert result.definition.en != "" and result.definition.fr != ""


def test_i18n_french_accent_in_definition() -> None:
  """French definition must contain the accented word 'clôture'."""
  stat = _otc_stat()
  result = stat.compute(_empty_df())
  assert "clôture" in result.definition.fr


def test_i18n_labels_have_en_and_fr() -> None:
  """Conditions and outcomes labels must define en and fr for every key."""
  stat = _otc_stat()
  result = stat.compute(_empty_df())
  for mapping in (result.labels.conditions, result.labels.outcomes):
    for key, i18n in mapping.items():
      assert i18n.en != "", f"{key}.en empty"
      assert i18n.fr != "", f"{key}.fr empty"

  assert set(result.labels.outcomes) == {"green_day", "red_day"}
  assert "any_day" in result.labels.conditions


def test_labels_dimensions_contains_weekday() -> None:
  """labels.dimensions must contain 'weekday' after compute()."""
  stat = _otc_stat()
  result = stat.compute(make_candles(_OTC_DAYS))
  assert "weekday" in result.labels.dimensions
  assert result.labels.dimensions["weekday"].en != ""
  assert result.labels.dimensions["weekday"].fr != ""


# ===========================================================================
# 8. write_results round-trip
# ===========================================================================

def test_stat_name() -> None:
  """stat_name must be 'green_red_days_by_weekday'."""
  stat = _otc_stat()
  assert stat.compute(_empty_df()).stat_name == "green_red_days_by_weekday"


def test_write_results_round_trip(tmp_path: Path) -> None:
  """compute() → write_results → JSON → validate preserves structure."""
  stat = _otc_stat()
  result = stat.compute(make_candles(_OTC_DAYS))
  written = write_results(result, results_dir=tmp_path)

  assert written.name == "green_red_days_by_weekday.json"
  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)

  tf = validated.instruments["NQ"]["daily"]
  assert tf.total_samples == 10
  assert "weekday" in tf.slices
  monday = tf.slices["weekday"].groups["monday"]
  green = next(r for r in monday.results if r.outcome == "green_day")
  assert green.probability == pytest.approx(1.0)


def test_write_results_french_accent_literal(tmp_path: Path) -> None:
  """JSON must store accented French as literal UTF-8, not \\u escapes."""
  stat = _otc_stat()
  result = stat.compute(_empty_df())
  written = write_results(result, results_dir=tmp_path)
  raw = written.read_text(encoding="utf-8")
  assert "clôture" in raw
  assert "\\u00f4" not in raw
