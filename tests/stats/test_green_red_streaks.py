"""Tests for stats.green_red_streaks.standard.

All data is synthetic — no real market files required.
Expected values are hand-calculated before each assertion.

Continuation framing: for a period of a given color, "continue" means the next
resolved period shares that color. The last resolved period has no next period
and is excluded from every denominator (pending discipline).
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.green_red_streaks.standard import GreenRedStreaks

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

_RTH_START = 570   # 09:30
_RTH_LAST = 974    # 16:14 (last bar before 16:15); resolved needs last mod >= 960


def _make_day(date: str, session_open: float, session_close: float) -> pd.DataFrame:
  """One trading day of 1-min RTH bars (09:30–16:14)."""
  base = pd.Timestamp(date, tz=_NY)
  records = []
  for mod in range(_RTH_START, _RTH_LAST + 1):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    o = session_open if mod == _RTH_START else 100.0
    c = session_close if mod == _RTH_LAST else 100.0
    records.append({
      "timestamp": ts,
      "open": o,
      "high": max(o, c) + 0.25,
      "low": min(o, c) - 0.25,
      "close": c,
      "volume": 1000,
    })
  return pd.DataFrame(records)


def _green(date: str) -> dict:
  return {"date": date, "session_open": 100.0, "session_close": 120.0}


def _red(date: str) -> dict:
  return {"date": date, "session_open": 100.0, "session_close": 80.0}


def make_candles(days: list[dict]) -> pd.DataFrame:
  frames = [_make_day(d["date"], d["session_open"], d["session_close"]) for d in days]
  df = pd.concat(frames, ignore_index=True)
  return df.sort_values("timestamp").reset_index(drop=True)


def _make_truncated_day(date: str) -> pd.DataFrame:
  """A day whose last RTH bar is 09:50 (mod 590 < 960) — unresolved, excluded."""
  base = pd.Timestamp(date, tz=_NY)
  records = []
  for mod in range(_RTH_START, 591):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    records.append({
      "timestamp": ts, "open": 100.0, "high": 100.25, "low": 99.75,
      "close": 100.0, "volume": 500,
    })
  return pd.DataFrame(records)


def _daily(performance: str = "open_to_close") -> GreenRedStreaks:
  return GreenRedStreaks(
    instrument="NQ", config=_TEST_CONFIG, granularity="daily", performance=performance
  )


def _row(result: StatRunResult, tf: str, condition: str, outcome: str) -> StatResultRow:
  for r in result.instruments["NQ"][tf].results:
    if r.condition == condition and r.outcome == outcome:
      return r
  raise KeyError((condition, outcome))


# ===========================================================================
# Daily continuation (open_to_close)
#
# Color sequence over 10 weekdays (G=close 120, R=close 80):
#   idx:  0 1 2 3 4 5 6 7 8 9
#   col:  G G R G R R R G R G
#   next: G R G R R R G R G -      (idx 9 has no next → excluded)
#
# Green periods with a next: 0,1,3,7  (idx 9 excluded)
#   continue(next green): 0→G yes, 1→R no, 3→R no, 7→R no  => 1/4 = 0.25
# Red periods with a next: 2,4,5,6,8
#   continue(next red):  2→G no, 4→R yes, 5→R yes, 6→G no, 8→G no => 2/5 = 0.4
# ===========================================================================

_SEQ_DATES = [
  "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
  "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
]
_SEQ_COLORS = "GGRGRRRGRG"
_DAILY_SEQ = [
  _green(d) if c == "G" else _red(d) for d, c in zip(_SEQ_DATES, _SEQ_COLORS)
]


def test_daily_total_samples_counts_all_periods() -> None:
  """total_samples = all 10 resolved periods (the last one is still resolved)."""
  result = _daily().compute(make_candles(_DAILY_SEQ))
  assert result.instruments["NQ"]["daily"].total_samples == 10


def test_daily_green_continuation() -> None:
  """P(green streak continues) = 1/4 = 0.25 over 4 countable green periods."""
  result = _daily().compute(make_candles(_DAILY_SEQ))
  cont = _row(result, "daily", "green", "continue")
  brk = _row(result, "daily", "green", "break")
  assert cont.total == 4
  assert cont.count == 1
  assert cont.probability == pytest.approx(0.25)
  assert brk.count == 3
  assert cont.count + brk.count == cont.total


def test_daily_red_continuation() -> None:
  """P(red streak continues) = 2/5 = 0.4 over 5 countable red periods."""
  result = _daily().compute(make_candles(_DAILY_SEQ))
  cont = _row(result, "daily", "red", "continue")
  brk = _row(result, "daily", "red", "break")
  assert cont.total == 5
  assert cont.count == 2
  assert cont.probability == pytest.approx(0.4)
  assert brk.count == 3


def test_daily_last_period_excluded_from_denominators() -> None:
  """Countable periods (green + red totals) = 9 = periods - 1 (last has no next)."""
  result = _daily().compute(make_candles(_DAILY_SEQ))
  green_total = _row(result, "daily", "green", "continue").total
  red_total = _row(result, "daily", "red", "continue").total
  assert green_total + red_total == 9


def test_daily_four_rows_only() -> None:
  result = _daily().compute(make_candles(_DAILY_SEQ))
  rows = result.instruments["NQ"]["daily"].results
  assert {(r.condition, r.outcome) for r in rows} == {
    ("green", "continue"), ("green", "break"),
    ("red", "continue"), ("red", "break"),
  }


def test_daily_data_range() -> None:
  result = _daily().compute(make_candles(_DAILY_SEQ))
  assert result.instruments["NQ"]["daily"].data_range == ["2024-01-01", "2024-01-12"]


def test_all_green_streak_always_continues() -> None:
  """An all-green run: every countable green period continues → P=1.0, red empty."""
  days = [_green(d) for d in _SEQ_DATES]
  result = _daily().compute(make_candles(days))
  green_cont = _row(result, "daily", "green", "continue")
  assert green_cont.probability == pytest.approx(1.0)
  assert green_cont.total == 9  # 10 periods, last excluded
  red_cont = _row(result, "daily", "red", "continue")
  assert red_cont.total == 0
  assert red_cont.probability == pytest.approx(0.0)  # no ZeroDivisionError


# ===========================================================================
# close_to_close direction basis
#
# Closes: 100, 110, 105, 105, 90  (first period dropped: no prior close)
#   period1 110>=100 G ; period2 105>=110 R ; period3 105>=105 G ; period4 90>=105 R
#   colors after drop: G R G R  (4 periods), next: R G R -
#   green countable: p1,p3 → p1 next R no, p3 next R no => continue 0/2 = 0.0
# ===========================================================================

_CTC_DAYS = [
  {"date": "2024-01-01", "session_open": 100.0, "session_close": 100.0},  # dropped
  {"date": "2024-01-02", "session_open": 100.0, "session_close": 110.0},  # G
  {"date": "2024-01-03", "session_open": 100.0, "session_close": 105.0},  # R
  {"date": "2024-01-04", "session_open": 100.0, "session_close": 105.0},  # G (equal)
  {"date": "2024-01-05", "session_open": 100.0, "session_close": 90.0},   # R
]


def test_close_to_close_drops_first_period() -> None:
  stat = _daily(performance="close_to_close")
  day_table = stat.build_day_table(make_candles(_CTC_DAYS))
  assert len(day_table) == 4
  first = pd.Timestamp("2024-01-01", tz=_NY).normalize()
  assert first not in day_table.index


def test_close_to_close_colors_and_continuation() -> None:
  stat = _daily(performance="close_to_close")
  result = stat.compute(make_candles(_CTC_DAYS))
  green_cont = _row(result, "daily", "green", "continue")
  # green periods = p1,p3 (both followed by red) → 0 continue out of 2
  assert green_cont.total == 2
  assert green_cont.count == 0
  assert green_cont.probability == pytest.approx(0.0)


def test_close_to_close_equal_close_is_green() -> None:
  stat = _daily(performance="close_to_close")
  day_table = stat.build_day_table(make_candles(_CTC_DAYS)).sort_index()
  day3 = pd.Timestamp("2024-01-04", tz=_NY).normalize()  # 105 vs prev 105
  assert bool(day_table.loc[day3, "period_green"])


# ===========================================================================
# Weekly aggregation (open_to_close)
#
# 6 ISO weeks, each with Mon (open=100) + Tue (close controls color):
#   colors: G G R G R G   next: G R G R G -
#   green countable: W1,W2,W4 (W6 last, excluded) → W1→G yes, W2→R no, W4→R no = 1/3
# ===========================================================================

_WEEK_TUES = ["2024-01-02", "2024-01-09", "2024-01-16", "2024-01-23", "2024-01-30", "2024-02-06"]
_WEEK_MONS = ["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22", "2024-01-29", "2024-02-05"]
_WEEK_COLORS = "GGRGRG"


def _weekly_candles() -> pd.DataFrame:
  days = []
  for mon, tue, col in zip(_WEEK_MONS, _WEEK_TUES, _WEEK_COLORS):
    days.append(_green(mon))  # Monday open is what matters; color set by Tuesday close
    days.append(_green(tue) if col == "G" else _red(tue))
  return make_candles(days)


def test_weekly_period_count() -> None:
  stat = GreenRedStreaks(instrument="NQ", config=_TEST_CONFIG, granularity="weekly",
                         performance="open_to_close")
  result = stat.compute(_weekly_candles())
  assert result.instruments["NQ"]["weekly"].total_samples == 6


def test_weekly_period_open_close_aggregation() -> None:
  """week_open = first day's open, week_close = last day's close."""
  stat = GreenRedStreaks(instrument="NQ", config=_TEST_CONFIG, granularity="weekly",
                         performance="open_to_close")
  table = stat.build_day_table(_weekly_candles()).sort_index()
  w1 = pd.Timestamp("2024-01-01", tz=_NY).normalize()  # indexed by first day of week
  assert table.loc[w1, "period_open"] == pytest.approx(100.0)
  assert table.loc[w1, "period_close"] == pytest.approx(120.0)  # Tue green close
  assert bool(table.loc[w1, "period_green"])


def test_weekly_green_continuation() -> None:
  stat = GreenRedStreaks(instrument="NQ", config=_TEST_CONFIG, granularity="weekly",
                         performance="open_to_close")
  result = stat.compute(_weekly_candles())
  green_cont = _row(result, "weekly", "green", "continue")
  assert green_cont.total == 3
  assert green_cont.count == 1
  assert green_cont.probability == pytest.approx(1 / 3)


# ===========================================================================
# Monthly aggregation (open_to_close)
#
# Same data as weekly: Jan holds weeks 1–5, Feb holds week 6.
#   Jan: open = 2024-01-01 (100), close = 2024-01-30 Tue (R, 80) → red
#   Feb: open = 2024-02-05 (100), close = 2024-02-06 Tue (G, 120) → green
#   colors: R G  → red countable=1 (next green) → continue 0/1
# ===========================================================================

def test_monthly_period_count_and_colors() -> None:
  stat = GreenRedStreaks(instrument="NQ", config=_TEST_CONFIG, granularity="monthly",
                         performance="open_to_close")
  table = stat.build_day_table(_weekly_candles()).sort_index()
  assert len(table) == 2
  jan = pd.Timestamp("2024-01-01", tz=_NY).normalize()
  feb = pd.Timestamp("2024-02-05", tz=_NY).normalize()
  assert not bool(table.loc[jan, "period_green"])  # Jan closes red (80 < 100)
  assert bool(table.loc[feb, "period_green"])       # Feb closes green


def test_monthly_continuation() -> None:
  stat = GreenRedStreaks(instrument="NQ", config=_TEST_CONFIG, granularity="monthly",
                         performance="open_to_close")
  result = stat.compute(_weekly_candles())
  red_cont = _row(result, "monthly", "red", "continue")
  assert red_cont.total == 1  # Jan (red) has a next (Feb); Feb is last → excluded
  assert red_cont.count == 0  # next period (Feb) is green
  assert red_cont.probability == pytest.approx(0.0)


# ===========================================================================
# Baseline ≈ 50% and deterministic
# ===========================================================================

def _long_seq() -> pd.DataFrame:
  """250 weekdays, deterministic but streaky color pattern."""
  dates: list[str] = []
  d = pd.Timestamp("2020-01-01", tz=_NY)
  while len(dates) < 250:
    if d.weekday() < 5:
      dates.append(d.strftime("%Y-%m-%d"))
    d += pd.Timedelta(days=1)
  # Runs of 3 green then 2 red, repeated — strong real continuation signal.
  days = []
  for i, date in enumerate(dates):
    days.append(_green(date) if (i % 5) < 3 else _red(date))
  return make_candles(days)


def test_baseline_approx_fifty_percent() -> None:
  stat = _daily()
  rows = stat.baseline(_long_seq(), seed=42)
  for row in rows:
    assert row.probability == pytest.approx(0.5, abs=0.12)


def test_baseline_deterministic() -> None:
  stat = _daily()
  df = _long_seq()
  rows_a = stat.baseline(df, seed=7)
  rows_b = stat.baseline(df, seed=7)
  for a, b in zip(rows_a, rows_b):
    assert (a.condition, a.outcome) == (b.condition, b.outcome)
    assert a.probability == pytest.approx(b.probability)


def test_baseline_embedded_in_compute() -> None:
  result = _daily().compute(_long_seq())
  for row in result.instruments["NQ"]["daily"].results:
    assert row.baseline_n > 0


def test_real_continuation_beats_baseline() -> None:
  """The streaky pattern continues far more than the random baseline."""
  result = _daily().compute(_long_seq())
  green_cont = _row(result, "daily", "green", "continue")
  assert green_cont.probability > green_cont.baseline_prob


# ===========================================================================
# Pending exclusion
# ===========================================================================

def test_pending_day_excluded() -> None:
  stat = _daily()
  base_df = make_candles(_DAILY_SEQ)
  total_base = stat.compute(base_df).instruments["NQ"]["daily"].total_samples
  truncated = _make_truncated_day("2024-01-15")
  df = pd.concat([base_df, truncated], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
  total_with = stat.compute(df).instruments["NQ"]["daily"].total_samples
  assert total_with == total_base == 10


def test_pending_day_absent_from_day_table() -> None:
  stat = _daily()
  day_table = stat.build_day_table(_make_truncated_day("2024-01-15"))
  pending = pd.Timestamp("2024-01-15", tz=_NY).normalize()
  assert pending not in day_table.index


# ===========================================================================
# Edge cases & validation
# ===========================================================================

def _empty_df() -> pd.DataFrame:
  df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
  df["timestamp"] = pd.array([], dtype="datetime64[ns, America/New_York]")
  return df


def test_empty_dataframe_no_crash() -> None:
  result = _daily().compute(_empty_df())
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 0
  assert tf.data_range == []
  for row in tf.results:
    assert row.count == 0
    assert row.total == 0
    assert row.probability == pytest.approx(0.0)


def test_empty_dataframe_no_slices() -> None:
  """The streaks stat declares no slices."""
  result = _daily().compute(_empty_df())
  assert result.instruments["NQ"]["daily"].slices == {}


def test_single_period_no_countable() -> None:
  """One resolved period has no next → all denominators zero, no crash."""
  result = _daily().compute(make_candles([_green("2024-03-01")]))
  for row in result.instruments["NQ"]["daily"].results:
    assert row.total == 0


def test_invalid_granularity_raises() -> None:
  with pytest.raises(ValueError, match="Unsupported granularity"):
    GreenRedStreaks(instrument="NQ", config=_TEST_CONFIG, granularity="hourly")


def test_invalid_performance_raises() -> None:
  with pytest.raises(ValueError, match="Unsupported performance"):
    GreenRedStreaks(instrument="NQ", config=_TEST_CONFIG, performance="tick_to_tick")


def test_defaults() -> None:
  stat = GreenRedStreaks(instrument="NQ", config=_TEST_CONFIG)
  assert stat.performance == "close_to_close"
  assert stat.granularity == "daily"
  assert stat.timeframe == "daily"


# ===========================================================================
# i18n
# ===========================================================================

def test_i18n_title_and_definition() -> None:
  result = _daily().compute(_empty_df())
  assert result.title.en != "" and result.title.fr != ""
  assert result.definition.en != "" and result.definition.fr != ""


def test_i18n_labels_have_en_and_fr() -> None:
  result = _daily().compute(_empty_df())
  for mapping in (result.labels.conditions, result.labels.outcomes):
    for key, i18n in mapping.items():
      assert i18n.en != "", f"{key}.en empty"
      assert i18n.fr != "", f"{key}.fr empty"
  assert set(result.labels.conditions) == {"green", "red"}
  assert set(result.labels.outcomes) == {"continue", "break"}


# ===========================================================================
# write_results round-trip
# ===========================================================================

def test_stat_name() -> None:
  assert _daily().compute(_empty_df()).stat_name == "green_red_streaks"


def _merged_three_granularities() -> StatRunResult:
  """Mirror run(): compute all three granularities and merge into one result."""
  from stats.base import TimeframeResult
  merged: dict[str, TimeframeResult] = {}
  for g in ("daily", "weekly", "monthly"):
    stat = GreenRedStreaks(instrument="NQ", config=_TEST_CONFIG, granularity=g,
                           performance="open_to_close")
    merged[g] = stat.compute(_weekly_candles()).instruments["NQ"][g]
  return StatRunResult(
    stat_name="green_red_streaks",
    title=_daily().title,
    definition=_daily().definition,
    labels=_daily().labels,
    instruments={"NQ": merged},
  )


def test_write_results_round_trip(tmp_path: Path) -> None:
  result = _merged_three_granularities()
  written = write_results(result, results_dir=tmp_path)
  assert written.name == "green_red_streaks.json"

  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)
  assert set(validated.instruments["NQ"].keys()) == {"daily", "weekly", "monthly"}
  assert validated.instruments["NQ"]["weekly"].total_samples == 6


def test_write_results_french_accent_literal(tmp_path: Path) -> None:
  result = _daily().compute(_empty_df())
  written = write_results(result, results_dir=tmp_path)
  raw = written.read_text(encoding="utf-8")
  assert "é" in raw  # "Séries" / "période"
  assert "\\u00e9" not in raw


# ===========================================================================
# classify_samples()
# ===========================================================================

def test_classify_samples_exact_daily_sequence() -> None:
  """Exact (date, condition, outcome) tuples for the daily sequence GGRGRRRGRG.

  next: G R G R R R G R G -  (idx 9 has no next → excluded)
  idx0 G next G -> continue | idx1 G next R -> break
  idx2 R next G -> break    | idx3 G next R -> break
  idx4 R next R -> continue | idx5 R next R -> continue
  idx6 R next G -> break    | idx7 G next R -> break
  idx8 R next G -> break    | idx9 excluded (no next)
  """
  stat = _daily()
  day_table = stat.build_day_table(make_candles(_DAILY_SEQ))
  samples = stat.classify_samples(day_table)

  expected = [
    ("2024-01-01", "green", "continue"),
    ("2024-01-02", "green", "break"),
    ("2024-01-03", "red", "break"),
    ("2024-01-04", "green", "break"),
    ("2024-01-05", "red", "continue"),
    ("2024-01-08", "red", "continue"),
    ("2024-01-09", "red", "break"),
    ("2024-01-10", "green", "break"),
    ("2024-01-11", "red", "break"),
  ]
  assert [(s.date, s.condition, s.outcome) for s in samples] == expected
  assert all(s.value is None for s in samples)
  assert "2024-01-12" not in {s.date for s in samples}  # last period excluded


def test_classify_samples_consistency_with_compute_rows() -> None:
  """Every StatResultRow.count equals the matching SampleRow (condition, outcome) tally."""
  stat = _daily()
  day_table = stat.build_day_table(make_candles(_DAILY_SEQ))
  rows = stat.compute_rows(day_table)
  samples = stat.classify_samples(day_table)

  for row in rows:
    tally = sum(1 for s in samples if s.condition == row.condition and s.outcome == row.outcome)
    assert tally == row.count, (row.condition, row.outcome)

  # Total countable samples across both conditions/outcomes matches sum of totals
  # of one outcome per condition (continue + break = total per condition).
  green_total = next(r.total for r in rows if r.condition == "green" and r.outcome == "continue")
  red_total = next(r.total for r in rows if r.condition == "red" and r.outcome == "continue")
  assert len(samples) == green_total + red_total


def test_classify_samples_empty_day_table() -> None:
  stat = _daily()
  day_table = stat.build_day_table(_empty_df())
  assert stat.classify_samples(day_table) == []


def test_classify_samples_single_period_no_countable() -> None:
  """One resolved period has no next -> not countable -> no samples."""
  stat = _daily()
  day_table = stat.build_day_table(make_candles([_green("2024-03-01")]))
  assert stat.classify_samples(day_table) == []


def test_classify_samples_pending_day_excluded() -> None:
  """A truncated (unresolved) day never appears in the day_table, so it can never
  produce a sample."""
  stat = _daily()
  day_table = stat.build_day_table(_make_truncated_day("2024-01-15"))
  samples = stat.classify_samples(day_table)
  assert samples == []
  assert "2024-01-15" not in {s.date for s in samples}


def test_classify_samples_embedded_in_compute() -> None:
  """compute() embeds the same samples in TimeframeResult.samples, sorted by date."""
  stat = _daily()
  df = make_candles(_DAILY_SEQ)
  day_table = stat.build_day_table(df)
  expected = sorted(stat.classify_samples(day_table), key=lambda s: s.date)

  result = stat.compute(df)
  actual = result.instruments["NQ"]["daily"].samples
  assert [(s.date, s.condition, s.outcome) for s in actual] == [
    (s.date, s.condition, s.outcome) for s in expected
  ]
