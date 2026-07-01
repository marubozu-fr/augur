"""Tests for stats.prev_session_correlation.standard.

All data is synthetic — no real market files required.
Expected values are hand-calculated before each assertion.

Matrix framing: condition = the prior resolved session's color, outcome = the
current session's color. The first usable session has no prior session and is
excluded from every denominator (pending discipline).
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.prev_session_correlation.standard import PrevSessionCorrelation

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


def _stat(performance: str = "open_to_close") -> PrevSessionCorrelation:
  return PrevSessionCorrelation(
    instrument="NQ", config=_TEST_CONFIG, performance=performance
  )


def _row(result: StatRunResult, condition: str, outcome: str) -> StatResultRow:
  for r in result.instruments["NQ"]["daily"].results:
    if r.condition == condition and r.outcome == outcome:
      return r
  raise KeyError((condition, outcome))


# ===========================================================================
# Daily correlation matrix (open_to_close)
#
# Color sequence over 10 weekdays (G=close 120, R=close 80):
#   idx:  0 1 2 3 4 5 6 7 8 9
#   col:  G G R G R R R G R G
#   prev: -  G G R G R R R G R   (idx 0 has no prior → excluded)
#
# Countable sessions = idx 1..9 (9). For each, (prior color, current color):
#   1:(G,G) 2:(G,R) 3:(R,G) 4:(G,R) 5:(R,R) 6:(R,R) 7:(R,G) 8:(G,R) 9:(R,G)
#   prev_green: idx 1,2,4,8  → current G:1 R:3  → P(green|prevG)=1/4
#   prev_red:   idx 3,5,6,7,9 → current G:3 R:2 → P(green|prevR)=3/5
# ===========================================================================

_SEQ_DATES = [
  "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
  "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
]
_SEQ_COLORS = "GGRGRRRGRG"
_DAILY_SEQ = [
  _green(d) if c == "G" else _red(d) for d, c in zip(_SEQ_DATES, _SEQ_COLORS)
]


def test_total_samples_counts_all_resolved_sessions() -> None:
  """total_samples = all 10 resolved sessions (the first one is still resolved)."""
  result = _stat().compute(make_candles(_DAILY_SEQ))
  assert result.instruments["NQ"]["daily"].total_samples == 10


def test_prev_green_conditional() -> None:
  """Given a prior green session: P(green)=1/4, P(red)=3/4 over 4 countable rows."""
  result = _stat().compute(make_candles(_DAILY_SEQ))
  green = _row(result, "prev_green", "green")
  red = _row(result, "prev_green", "red")
  assert green.total == red.total == 4
  assert green.count == 1
  assert green.probability == pytest.approx(0.25)
  assert red.count == 3
  assert red.probability == pytest.approx(0.75)


def test_prev_red_conditional() -> None:
  """Given a prior red session: P(green)=3/5, P(red)=2/5 over 5 countable rows."""
  result = _stat().compute(make_candles(_DAILY_SEQ))
  green = _row(result, "prev_red", "green")
  red = _row(result, "prev_red", "red")
  assert green.total == red.total == 5
  assert green.count == 3
  assert green.probability == pytest.approx(0.6)
  assert red.count == 2
  assert red.probability == pytest.approx(0.4)


def test_first_session_excluded_from_denominators() -> None:
  """Countable sessions (all condition totals) = 9 = sessions - 1 (first has no prior)."""
  result = _stat().compute(make_candles(_DAILY_SEQ))
  prev_green_total = _row(result, "prev_green", "green").total
  prev_red_total = _row(result, "prev_red", "green").total
  assert prev_green_total + prev_red_total == 9


def test_outcomes_partition_each_condition() -> None:
  """green + red counts equal the condition total for each prior color."""
  result = _stat().compute(make_candles(_DAILY_SEQ))
  for cond in ("prev_green", "prev_red"):
    green = _row(result, cond, "green")
    red = _row(result, cond, "red")
    assert green.count + red.count == green.total == red.total


def test_four_rows_only() -> None:
  result = _stat().compute(make_candles(_DAILY_SEQ))
  rows = result.instruments["NQ"]["daily"].results
  assert {(r.condition, r.outcome) for r in rows} == {
    ("prev_green", "green"), ("prev_green", "red"),
    ("prev_red", "green"), ("prev_red", "red"),
  }


def test_data_range() -> None:
  result = _stat().compute(make_candles(_DAILY_SEQ))
  assert result.instruments["NQ"]["daily"].data_range == ["2024-01-01", "2024-01-12"]


def test_all_green_only_prev_green_condition() -> None:
  """An all-green run: prev_green→green is certain; prev_red condition is empty."""
  days = [_green(d) for d in _SEQ_DATES]
  result = _stat().compute(make_candles(days))
  pg_green = _row(result, "prev_green", "green")
  assert pg_green.probability == pytest.approx(1.0)
  assert pg_green.total == 9  # 10 sessions, first excluded
  pr_green = _row(result, "prev_red", "green")
  assert pr_green.total == 0
  assert pr_green.probability == pytest.approx(0.0)  # no ZeroDivisionError


# ===========================================================================
# close_to_close direction basis
#
# Closes: 100, 110, 105, 105, 90  (first session dropped: no prior close)
#   s1 110>=100 G ; s2 105>=110 R ; s3 105>=105 G ; s4 90>=105 R
#   colors after drop: G R G R, prev: - G R G  (s1 has no prior → excluded)
#   countable: s2(prevG,R) s3(prevR,G) s4(prevG,R)
#     prev_green: s2,s4 → green 0/2 ; prev_red: s3 → green 1/1
# ===========================================================================

_CTC_DAYS = [
  {"date": "2024-01-01", "session_open": 100.0, "session_close": 100.0},  # dropped
  {"date": "2024-01-02", "session_open": 100.0, "session_close": 110.0},  # G
  {"date": "2024-01-03", "session_open": 100.0, "session_close": 105.0},  # R
  {"date": "2024-01-04", "session_open": 100.0, "session_close": 105.0},  # G (equal)
  {"date": "2024-01-05", "session_open": 100.0, "session_close": 90.0},   # R
]


def test_close_to_close_drops_first_session() -> None:
  stat = _stat(performance="close_to_close")
  day_table = stat.build_day_table(make_candles(_CTC_DAYS))
  assert len(day_table) == 4
  first = pd.Timestamp("2024-01-01", tz=_NY).normalize()
  assert first not in day_table.index


def test_close_to_close_equal_close_is_green() -> None:
  stat = _stat(performance="close_to_close")
  day_table = stat.build_day_table(make_candles(_CTC_DAYS)).sort_index()
  s3 = pd.Timestamp("2024-01-04", tz=_NY).normalize()  # 105 vs prev 105
  assert bool(day_table.loc[s3, "session_green"])


def test_close_to_close_conditionals() -> None:
  stat = _stat(performance="close_to_close")
  result = stat.compute(make_candles(_CTC_DAYS))
  pg_green = _row(result, "prev_green", "green")
  assert pg_green.total == 2  # s2, s4 follow a green session
  assert pg_green.count == 0
  assert pg_green.probability == pytest.approx(0.0)
  pr_green = _row(result, "prev_red", "green")
  assert pr_green.total == 1  # s3 follows a red session
  assert pr_green.count == 1
  assert pr_green.probability == pytest.approx(1.0)


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
  # Runs of 3 green then 2 red, repeated — strong real follow-through signal.
  days = []
  for i, date in enumerate(dates):
    days.append(_green(date) if (i % 5) < 3 else _red(date))
  return make_candles(days)


def test_baseline_approx_fifty_percent() -> None:
  stat = _stat()
  rows = stat.baseline(_long_seq(), seed=42)
  for row in rows:
    assert row.probability == pytest.approx(0.5, abs=0.12)


def test_baseline_deterministic() -> None:
  stat = _stat()
  df = _long_seq()
  rows_a = stat.baseline(df, seed=7)
  rows_b = stat.baseline(df, seed=7)
  for a, b in zip(rows_a, rows_b):
    assert (a.condition, a.outcome) == (b.condition, b.outcome)
    assert a.probability == pytest.approx(b.probability)


def test_baseline_embedded_in_compute() -> None:
  result = _stat().compute(_long_seq())
  for row in result.instruments["NQ"]["daily"].results:
    assert row.baseline_n > 0


def test_real_follow_through_beats_baseline() -> None:
  """The streaky pattern follows through far more than the random baseline."""
  result = _stat().compute(_long_seq())
  pg_green = _row(result, "prev_green", "green")
  assert pg_green.probability > pg_green.baseline_prob


# ===========================================================================
# Pending exclusion
# ===========================================================================

def test_pending_day_excluded() -> None:
  stat = _stat()
  base_df = make_candles(_DAILY_SEQ)
  total_base = stat.compute(base_df).instruments["NQ"]["daily"].total_samples
  truncated = _make_truncated_day("2024-01-15")
  df = pd.concat([base_df, truncated], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
  total_with = stat.compute(df).instruments["NQ"]["daily"].total_samples
  assert total_with == total_base == 10


def test_pending_day_absent_from_day_table() -> None:
  stat = _stat()
  day_table = stat.build_day_table(_make_truncated_day("2024-01-15"))
  pending = pd.Timestamp("2024-01-15", tz=_NY).normalize()
  assert pending not in day_table.index


# ===========================================================================
# classify_samples()
# ===========================================================================
def test_classify_samples_exact_sequence() -> None:
  """Exact (date, condition, outcome, value) tuples for the 10-day _DAILY_SEQ.

  Sequence recap (see the module-level comment above): col GGRGRRRGRG, idx 0 has
  no prior session and is excluded from every countable row.
  """
  stat = _stat()
  day_table = stat.build_day_table(make_candles(_DAILY_SEQ))
  samples = stat.classify_samples(day_table)

  expected = [
    ("2024-01-02", "prev_green", "green", 1.0),
    ("2024-01-03", "prev_green", "red", 0.0),
    ("2024-01-04", "prev_red", "green", 1.0),
    ("2024-01-05", "prev_green", "red", 0.0),
    ("2024-01-08", "prev_red", "red", 0.0),
    ("2024-01-09", "prev_red", "red", 0.0),
    ("2024-01-10", "prev_red", "green", 1.0),
    ("2024-01-11", "prev_green", "red", 0.0),
    ("2024-01-12", "prev_red", "green", 1.0),
  ]
  assert [(s.date, s.condition, s.outcome, s.value) for s in samples] == expected


def test_classify_samples_consistency_with_compute_rows() -> None:
  """Every StatResultRow.count equals the matching SampleRow (condition, outcome) tally."""
  stat = _stat()
  day_table = stat.build_day_table(make_candles(_DAILY_SEQ))
  rows = stat.compute_rows(day_table)
  samples = stat.classify_samples(day_table)

  for row in rows:
    tally = sum(1 for s in samples if s.condition == row.condition and s.outcome == row.outcome)
    assert tally == row.count, (row.condition, row.outcome)

  # day_table keeps all 10 resolved sessions; the first has no prior session (NaN
  # prev_green) and is excluded from samples, matching compute_rows's countable mask.
  assert day_table.shape[0] == 10
  assert len(samples) == 9


def test_classify_samples_empty_day_table() -> None:
  stat = _stat()
  day_table = stat.build_day_table(_empty_df())
  assert stat.classify_samples(day_table) == []


def test_classify_samples_first_session_excluded() -> None:
  """The first resolved session has no prior color and never produces a sample."""
  stat = _stat()
  day_table = stat.build_day_table(make_candles(_DAILY_SEQ))
  samples = stat.classify_samples(day_table)
  assert "2024-01-01" not in {s.date for s in samples}


def test_classify_samples_embedded_in_compute() -> None:
  """compute() embeds the same samples in TimeframeResult.samples, sorted by date."""
  stat = _stat()
  df = make_candles(_DAILY_SEQ)
  day_table = stat.build_day_table(df)
  expected = sorted(stat.classify_samples(day_table), key=lambda s: s.date)
  result = stat.compute(df)
  assert result.instruments["NQ"]["daily"].samples == expected


# ===========================================================================
# Edge cases & validation
# ===========================================================================

def _empty_df() -> pd.DataFrame:
  df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
  df["timestamp"] = pd.array([], dtype="datetime64[ns, America/New_York]")
  return df


def test_empty_dataframe_no_crash() -> None:
  result = _stat().compute(_empty_df())
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 0
  assert tf.data_range == []
  for row in tf.results:
    assert row.count == 0
    assert row.total == 0
    assert row.probability == pytest.approx(0.0)


def test_empty_dataframe_no_slices() -> None:
  """The prev-session-correlation stat declares no slices."""
  result = _stat().compute(_empty_df())
  assert result.instruments["NQ"]["daily"].slices == {}


def test_single_session_no_countable() -> None:
  """One resolved session has no prior → all denominators zero, no crash."""
  result = _stat().compute(make_candles([_green("2024-03-01")]))
  for row in result.instruments["NQ"]["daily"].results:
    assert row.total == 0


def test_invalid_performance_raises() -> None:
  with pytest.raises(ValueError, match="Unsupported performance"):
    PrevSessionCorrelation(instrument="NQ", config=_TEST_CONFIG, performance="tick_to_tick")


def test_defaults() -> None:
  stat = PrevSessionCorrelation(instrument="NQ", config=_TEST_CONFIG)
  assert stat.performance == "close_to_close"
  assert stat.timeframe == "daily"


# ===========================================================================
# i18n
# ===========================================================================

def test_i18n_title_and_definition() -> None:
  result = _stat().compute(_empty_df())
  assert result.title.en != "" and result.title.fr != ""
  assert result.definition.en != "" and result.definition.fr != ""


def test_i18n_labels_have_en_and_fr() -> None:
  result = _stat().compute(_empty_df())
  for mapping in (result.labels.conditions, result.labels.outcomes):
    for key, i18n in mapping.items():
      assert i18n.en != "", f"{key}.en empty"
      assert i18n.fr != "", f"{key}.fr empty"
  assert set(result.labels.conditions) == {"prev_green", "prev_red"}
  assert set(result.labels.outcomes) == {"green", "red"}


# ===========================================================================
# write_results round-trip
# ===========================================================================

def test_stat_name() -> None:
  assert _stat().compute(_empty_df()).stat_name == "prev_session_correlation"


def test_write_results_round_trip(tmp_path: Path) -> None:
  result = _stat().compute(make_candles(_DAILY_SEQ))
  written = write_results(result, results_dir=tmp_path)
  assert written.name == "prev_session_correlation.json"

  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)
  assert validated.instruments["NQ"]["daily"].total_samples == 10


def test_write_results_french_accent_literal(tmp_path: Path) -> None:
  result = _stat().compute(_empty_df())
  written = write_results(result, results_dir=tmp_path)
  raw = written.read_text(encoding="utf-8")
  assert "é" in raw  # "précédente"
  assert "\\u00e9" not in raw
