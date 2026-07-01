"""Tests for stats.prev_weeks_range.standard.

All data is synthetic — no real market files required. Expected values are
hand-calculated before each assertion.

Framing:
  - Tier 1 (condition ``prior_range``): a four-way partition of the countable
    weeks against the prior resolved week's range — ``break_high_only``,
    ``break_low_only``, ``break_both``, ``inside``. The four counts sum to the
    countable weeks.
  - Tier 2 (condition ``break_both``): among both-break weeks, which prior level
    was taken first — ``high_first`` / ``low_first`` (a partition of break_both).

Week boundaries are ISO weeks keyed by their Monday. A week's range is built from
the RTH bars of every trading day in the week. The LAST week in the data is
pending and excluded; the FIRST resolved week has no prior week and is excluded
from every denominator.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.prev_weeks_range.standard import PrevWeeksRange

# ---------------------------------------------------------------------------
# Minimal InstrumentConfig (does not depend on NQ.yaml)
# ---------------------------------------------------------------------------
_NY = "America/New_York"
_RTH_SESSION = Session(start="09:30", end="16:15")
_TEST_CONFIG = InstrumentConfig(
  instrument="NQ",
  working_timezone=_NY,
  sessions={"rth": _RTH_SESSION},
  timeframes=["weekly"],
  parquet_path=Path("data/NQ_1min.parquet"),
)

_RTH_START = 570   # 09:30
_RTH_LAST = 974    # 16:14 (last bar before 16:15)


def _make_day(
  date: str,
  session_open: float,
  session_close: float,
  day_high: float,
  day_low: float,
  high_at: int | None = None,
  low_at: int | None = None,
) -> pd.DataFrame:
  """One trading day of 1-min RTH bars (09:30–16:14).

  The opening bar carries ``session_open`` as its open; the last bar carries
  ``session_close`` as its close. ``day_high`` is placed on the ``high_at`` bar
  (default mid-session) and ``day_low`` on the ``low_at`` bar; all other bars sit
  between the open/close so they never produce a spurious extreme. Controlling
  ``high_at`` / ``low_at`` lets a test fix the intra-day order of the breaks.
  """
  base = pd.Timestamp(date, tz=_NY)
  mid = (_RTH_START + _RTH_LAST) // 2
  high_at = mid if high_at is None else high_at
  low_at = mid if low_at is None else low_at
  records = []
  for mod in range(_RTH_START, _RTH_LAST + 1):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    o = session_open if mod == _RTH_START else session_close
    c = session_close
    hi = max(o, c)
    lo = min(o, c)
    if mod == high_at:
      hi = day_high
    if mod == low_at:
      lo = day_low
    records.append({
      "timestamp": ts, "open": o, "high": hi, "low": lo, "close": c, "volume": 1000,
    })
  return pd.DataFrame(records)


def make_candles(days: list[dict]) -> pd.DataFrame:
  """Concatenate per-day specs into a single sorted 1-min OHLCV DataFrame."""
  frames = [
    _make_day(
      d["date"], d["open"], d["close"], d["high"], d["low"],
      high_at=d.get("high_at"), low_at=d.get("low_at"),
    )
    for d in days
  ]
  df = pd.concat(frames, ignore_index=True)
  return df.sort_values("timestamp").reset_index(drop=True)


def _stat() -> PrevWeeksRange:
  return PrevWeeksRange(instrument="NQ", config=_TEST_CONFIG)


def _row(result: StatRunResult, condition: str, outcome: str) -> StatResultRow:
  for r in result.instruments["NQ"]["weekly"].results:
    if r.condition == condition and r.outcome == outcome:
      return r
  raise KeyError((condition, outcome))


# ===========================================================================
# Core derivation — one trading day per week so each week's range == day range.
#
#   week  monday       high  low   | prev_high prev_low | outcome
#   W1    2024-01-01   110   90    |   —         —       | excluded (no prior)
#   W2    2024-01-08   115   95    |  110       90       | high_only (115>110, 95!<90)
#   W3    2024-01-15   113   85    |  115       95       | low_only  (113!>115, 85<95)
#   W4    2024-01-22   120   80    |  113       85       | both      (120>113, 80<85)
#   W5    2024-01-29   118   82    |  120       80       | inside    (118!>120, 82!<80)
#   W6    2024-02-05   130   70    |  (pending — dropped, never used)
#
# Resolved weeks = W1..W5 (W6 is the last week → pending).  total_samples = 5.
# Countable = W2..W5 (4 weeks; W1 has no prior).
# Partition: high_only=1, low_only=1, both=1, inside=1.
# ===========================================================================

_SEQ = [
  {"date": "2024-01-01", "open": 100.0, "close": 101.0, "high": 110.0, "low": 90.0},
  {"date": "2024-01-08", "open": 100.0, "close": 101.0, "high": 115.0, "low": 95.0},
  {"date": "2024-01-15", "open": 100.0, "close": 101.0, "high": 113.0, "low": 85.0},
  {"date": "2024-01-22", "open": 100.0, "close": 101.0, "high": 120.0, "low": 80.0},
  {"date": "2024-01-29", "open": 100.0, "close": 101.0, "high": 118.0, "low": 82.0},
  {"date": "2024-02-05", "open": 100.0, "close": 101.0, "high": 130.0, "low": 70.0},
]


def test_total_samples_counts_resolved_weeks() -> None:
  """total_samples = 5 resolved weeks (the pending last week is excluded)."""
  result = _stat().compute(make_candles(_SEQ))
  assert result.instruments["NQ"]["weekly"].total_samples == 5


def test_outcome_partition_counts() -> None:
  """Each of the four outcomes has exactly one countable week; total = 4."""
  result = _stat().compute(make_candles(_SEQ))
  for outcome in ("break_high_only", "break_low_only", "break_both", "inside"):
    row = _row(result, "prior_range", outcome)
    assert row.total == 4, outcome
    assert row.count == 1, outcome
    assert row.probability == pytest.approx(0.25), outcome


def test_outcome_partition_sums_to_countable() -> None:
  """The four outcome counts sum to the countable-week total."""
  result = _stat().compute(make_candles(_SEQ))
  counts = sum(
    _row(result, "prior_range", o).count
    for o in ("break_high_only", "break_low_only", "break_both", "inside")
  )
  assert counts == _row(result, "prior_range", "inside").total == 4


def test_first_week_excluded_from_denominators() -> None:
  """Countable total = 4 = resolved_weeks - 1 (first week has no prior)."""
  result = _stat().compute(make_candles(_SEQ))
  assert _row(result, "prior_range", "break_both").total == 5 - 1


def test_pending_last_week_excluded() -> None:
  """The wide-range final week (W6) is dropped, so it never produces a break."""
  result = _stat().compute(make_candles(_SEQ))
  # If W6 (high 130, low 70) had counted, it would break both vs W5 → both=2.
  assert _row(result, "prior_range", "break_both").count == 1


def test_data_range_uses_week_mondays() -> None:
  """data_range spans the first to the last RESOLVED week's Monday."""
  result = _stat().compute(make_candles(_SEQ))
  assert result.instruments["NQ"]["weekly"].data_range == ["2024-01-01", "2024-01-29"]


def test_six_rows_only() -> None:
  """Exactly the four partition rows plus the two sequence rows."""
  rows = _stat().compute(make_candles(_SEQ)).instruments["NQ"]["weekly"].results
  assert {(r.condition, r.outcome) for r in rows} == {
    ("prior_range", "break_high_only"), ("prior_range", "break_low_only"),
    ("prior_range", "break_both"), ("prior_range", "inside"),
    ("break_both", "high_first"), ("break_both", "low_first"),
  }


# ===========================================================================
# Multi-day weeks: range aggregates over every trading day in the week.
# ===========================================================================

def test_week_range_aggregates_over_all_days() -> None:
  """Week extremes are the max high / min low across the week's trading days."""
  days = [
    # W1 (prior): spread across the week, overall high 110, low 90.
    {"date": "2024-01-01", "open": 100.0, "close": 101.0, "high": 105.0, "low": 95.0},
    {"date": "2024-01-02", "open": 100.0, "close": 101.0, "high": 110.0, "low": 98.0},
    {"date": "2024-01-03", "open": 100.0, "close": 101.0, "high": 103.0, "low": 90.0},
    # W2: high 111 (Tue) breaks 110; low 89 (Wed) breaks 90 → both.
    {"date": "2024-01-08", "open": 100.0, "close": 101.0, "high": 104.0, "low": 96.0},
    {"date": "2024-01-09", "open": 100.0, "close": 101.0, "high": 111.0, "low": 97.0},
    {"date": "2024-01-10", "open": 100.0, "close": 101.0, "high": 102.0, "low": 89.0},
    # W3: pending, dropped.
    {"date": "2024-01-15", "open": 100.0, "close": 101.0, "high": 100.5, "low": 100.0},
  ]
  result = _stat().compute(make_candles(days))
  assert _row(result, "prior_range", "break_both").count == 1
  assert _row(result, "prior_range", "break_both").total == 1
  # High broken Tuesday (Jan 9), low broken Wednesday (Jan 10) → high first.
  assert _row(result, "break_both", "high_first").count == 1
  assert _row(result, "break_both", "low_first").count == 0


# ===========================================================================
# Break definition: strict inequality (touching the level is not a break).
# ===========================================================================

def test_touching_prior_level_is_not_a_break() -> None:
  """week_high == prev_high and week_low == prev_low → inside, not a break."""
  days = [
    {"date": "2024-01-01", "open": 100.0, "close": 100.0, "high": 110.0, "low": 90.0},
    # W2 exactly retests the prior extremes: 110 not > 110, 90 not < 90.
    {"date": "2024-01-08", "open": 100.0, "close": 100.0, "high": 110.0, "low": 90.0},
    {"date": "2024-01-15", "open": 100.0, "close": 100.0, "high": 110.0, "low": 90.0},  # pending
  ]
  result = _stat().compute(make_candles(days))
  assert _row(result, "prior_range", "inside").count == 1
  assert _row(result, "prior_range", "break_high_only").count == 0
  assert _row(result, "prior_range", "break_low_only").count == 0
  assert _row(result, "prior_range", "break_both").count == 0


# ===========================================================================
# Double-break sequence: which level is taken first within the week.
# ===========================================================================

def _both_break_week(high_at: int, low_at: int, *, bar_open: float = 100.0,
                     bar_close: float = 101.0) -> pd.DataFrame:
  """Prior week (110/90) followed by a both-break week with controllable timing.

  The breaking week is a single day whose high (120) is placed on ``high_at`` and
  low (80) on ``low_at``; a trailing pending week is appended.
  """
  days = [
    {"date": "2024-01-01", "open": 100.0, "close": 101.0, "high": 110.0, "low": 90.0},
    {"date": "2024-01-08", "open": bar_open, "close": bar_close, "high": 120.0,
     "low": 80.0, "high_at": high_at, "low_at": low_at},
    {"date": "2024-01-15", "open": 100.0, "close": 101.0, "high": 100.5, "low": 100.0},
  ]
  return make_candles(days)


def test_sequence_high_first() -> None:
  """High break (mod 600) precedes low break (mod 700) → high_first."""
  result = _stat().compute(_both_break_week(high_at=600, low_at=700))
  assert _row(result, "break_both", "high_first").count == 1
  assert _row(result, "break_both", "low_first").count == 0
  assert _row(result, "break_both", "high_first").total == 1
  assert _row(result, "break_both", "high_first").probability == pytest.approx(1.0)


def test_sequence_low_first() -> None:
  """Low break (mod 600) precedes high break (mod 700) → low_first."""
  result = _stat().compute(_both_break_week(high_at=700, low_at=600))
  assert _row(result, "break_both", "low_first").count == 1
  assert _row(result, "break_both", "high_first").count == 0


def test_sequence_partitions_break_both() -> None:
  """high_first + low_first == break_both total for the sequence rows."""
  result = _stat().compute(_both_break_week(high_at=600, low_at=700))
  hf = _row(result, "break_both", "high_first")
  lf = _row(result, "break_both", "low_first")
  assert hf.count + lf.count == hf.total == lf.total == 1


def test_sequence_same_bar_bullish_is_low_first() -> None:
  """Both levels broken on one bullish bar (close > open) → low printed first.

  The break is placed on the opening bar (mod 570) so the bar carries the
  controllable session open/close that the candle-path tiebreak reads.
  """
  result = _stat().compute(_both_break_week(high_at=_RTH_START, low_at=_RTH_START,
                                            bar_open=100.0, bar_close=101.0))
  assert _row(result, "break_both", "low_first").count == 1
  assert _row(result, "break_both", "high_first").count == 0


def test_sequence_same_bar_bearish_is_high_first() -> None:
  """Both levels broken on one bearish bar (close < open) → high printed first."""
  result = _stat().compute(_both_break_week(high_at=_RTH_START, low_at=_RTH_START,
                                            bar_open=101.0, bar_close=100.0))
  assert _row(result, "break_both", "high_first").count == 1
  assert _row(result, "break_both", "low_first").count == 0


# ===========================================================================
# Baseline ≈ neutral and deterministic.
# ===========================================================================

def _long_weeks() -> pd.DataFrame:
  """~120 single-day weeks alternating wide/narrow so both-breaks are frequent."""
  mondays: list[str] = []
  d = pd.Timestamp("2020-01-06", tz=_NY)  # a Monday
  while len(mondays) < 120:
    mondays.append(d.strftime("%Y-%m-%d"))
    d += pd.Timedelta(days=7)

  # Fixed center so wide weeks reliably engulf narrow ones — after the baseline
  # permutes prior ranges, both-breaks stay plentiful (low-variance ~0.5 check).
  base = 100.0
  days = []
  for i, date in enumerate(mondays):
    if i % 2 == 0:
      hi, lo = base + 20.0, base - 20.0
    else:
      hi, lo = base + 5.0, base - 5.0
    days.append({"date": date, "open": base, "close": base + 1.0, "high": hi, "low": lo})

  return make_candles(days)


def test_baseline_sequence_near_fifty_percent() -> None:
  """Sequence baseline (randomized order) is near 0.5 for both directions."""
  rows = _stat().baseline(_long_weeks(), seed=42)
  for r in rows:
    if r.condition == "break_both" and r.total > 0:
      assert r.probability == pytest.approx(0.5, abs=0.2)


def test_baseline_deterministic() -> None:
  """Same seed produces identical baseline results across two calls."""
  stat = _stat()
  df = _long_weeks()
  rows_a = stat.baseline(df, seed=7)
  rows_b = stat.baseline(df, seed=7)
  for a, b in zip(rows_a, rows_b):
    assert (a.condition, a.outcome) == (b.condition, b.outcome)
    assert a.probability == pytest.approx(b.probability)


def test_baseline_embedded_in_compute() -> None:
  """After compute(), every countable row carries a positive baseline_n."""
  result = _stat().compute(_long_weeks())
  for row in result.instruments["NQ"]["weekly"].results:
    if row.total > 0:
      assert row.baseline_n > 0


# ===========================================================================
# Edge cases & validation.
# ===========================================================================

def _empty_df() -> pd.DataFrame:
  df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
  df["timestamp"] = pd.array([], dtype="datetime64[ns, America/New_York]")
  return df


def test_empty_dataframe_no_crash() -> None:
  """Empty input → zero samples, empty data_range, all rows zeroed."""
  result = _stat().compute(_empty_df())
  tf = result.instruments["NQ"]["weekly"]
  assert tf.total_samples == 0
  assert tf.data_range == []
  for row in tf.results:
    assert row.count == 0
    assert row.total == 0
    assert row.probability == pytest.approx(0.0)


def test_single_week_is_pending() -> None:
  """A single week of data is the last (pending) week → zero resolved weeks."""
  days = [{"date": "2024-03-04", "open": 100.0, "close": 110.0, "high": 115.0, "low": 95.0}]
  result = _stat().compute(make_candles(days))
  assert result.instruments["NQ"]["weekly"].total_samples == 0


def test_two_weeks_first_has_no_prior() -> None:
  """Two weeks → one resolved week (the second), but it has no prior → no counts."""
  days = [
    {"date": "2024-03-04", "open": 100.0, "close": 110.0, "high": 115.0, "low": 95.0},
    {"date": "2024-03-11", "open": 100.0, "close": 110.0, "high": 120.0, "low": 90.0},  # pending
  ]
  result = _stat().compute(make_candles(days))
  tf = result.instruments["NQ"]["weekly"]
  assert tf.total_samples == 1
  for row in tf.results:
    assert row.total == 0


# ===========================================================================
# i18n.
# ===========================================================================

def test_i18n_title_and_definition() -> None:
  """title and definition both have non-empty en and fr strings."""
  result = _stat().compute(_empty_df())
  assert result.title.en != "" and result.title.fr != ""
  assert result.definition.en != "" and result.definition.fr != ""


def test_i18n_labels_have_en_and_fr() -> None:
  """Every condition and outcome label has non-empty en and fr; correct keys."""
  result = _stat().compute(_empty_df())
  for mapping in (result.labels.conditions, result.labels.outcomes):
    for key, i18n in mapping.items():
      assert i18n.en != "", f"{key}.en empty"
      assert i18n.fr != "", f"{key}.fr empty"
  assert set(result.labels.conditions) == {"prior_range", "break_both"}
  assert set(result.labels.outcomes) == {
    "break_high_only", "break_low_only", "break_both", "inside",
    "high_first", "low_first",
  }


# ===========================================================================
# stat_name and write_results round-trip.
# ===========================================================================

def test_stat_name() -> None:
  assert _stat().compute(_empty_df()).stat_name == "prev_weeks_range"


def test_timeframe_is_weekly() -> None:
  """The single timeframe entry is keyed 'weekly'."""
  result = _stat().compute(make_candles(_SEQ))
  assert set(result.instruments["NQ"]) == {"weekly"}


def test_write_results_round_trip(tmp_path: Path) -> None:
  """write_results produces prev_weeks_range.json that re-validates correctly."""
  result = _stat().compute(make_candles(_SEQ))
  written = write_results(result, results_dir=tmp_path)
  assert written.name == "prev_weeks_range.json"

  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)
  assert validated.instruments["NQ"]["weekly"].total_samples == 5


def test_write_results_french_accent_literal(tmp_path: Path) -> None:
  """French text is stored as literal UTF-8 characters, not escaped unicode."""
  result = _stat().compute(_empty_df())
  written = write_results(result, results_dir=tmp_path)
  raw = written.read_text(encoding="utf-8")
  # The French definition contains "précédente" (with é).
  assert "é" in raw
  assert "\\u00e9" not in raw


# ===========================================================================
# classify_samples()
#
# Reusing _SEQ (W1..W5 resolved, W6 pending, one countable week per outcome):
#   W1 (2024-01-01): excluded (no prior week)
#   W2 (2024-01-08): break_high_only
#   W3 (2024-01-15): break_low_only
#   W4 (2024-01-22): break_both, single-bar tiebreak — bullish bar
#                     (open=100 < close=101) -> low printed first -> low_first
#   W5 (2024-01-29): inside
# W4 emits TWO SampleRows (prior_range + break_both); the others emit ONE.
# ===========================================================================

def test_classify_samples_exact_list() -> None:
  """classify_samples emits prior_range samples for every countable week, plus a
  break_both sequence sample for the one both-break week."""
  stat = _stat()
  week_table = stat.build_day_table(make_candles(_SEQ))
  samples = stat.classify_samples(week_table)
  assert [(s.date, s.condition, s.outcome) for s in samples] == [
    ("2024-01-08", "prior_range", "break_high_only"),
    ("2024-01-15", "prior_range", "break_low_only"),
    ("2024-01-22", "prior_range", "break_both"),
    ("2024-01-22", "break_both", "low_first"),
    ("2024-01-29", "prior_range", "inside"),
  ]
  assert all(s.value is None for s in samples)


def test_classify_samples_matches_compute_rows_counts() -> None:
  """For every StatResultRow, the matching SampleRow count equals r.count."""
  stat = _stat()
  week_table = stat.build_day_table(make_candles(_SEQ))
  samples = stat.classify_samples(week_table)
  rows = stat.compute_rows(week_table)
  for r in rows:
    n = sum(1 for s in samples if s.condition == r.condition and s.outcome == r.outcome)
    assert n == r.count
  prior_range_total = sum(1 for s in samples if s.condition == "prior_range")
  assert prior_range_total == 4
  break_both_total = sum(1 for s in samples if s.condition == "break_both")
  assert break_both_total == 1


def test_classify_samples_high_first_order() -> None:
  """Sequence tier-2 outcome mirrors the week table's seq_high_first column."""
  stat = _stat()
  week_table = stat.build_day_table(_both_break_week(high_at=600, low_at=700))
  samples = stat.classify_samples(week_table)
  assert [(s.date, s.condition, s.outcome) for s in samples] == [
    ("2024-01-08", "prior_range", "break_both"),
    ("2024-01-08", "break_both", "high_first"),
  ]
  rows = stat.compute_rows(week_table)
  for r in rows:
    n = sum(1 for s in samples if s.condition == r.condition and s.outcome == r.outcome)
    assert n == r.count


def test_classify_samples_low_first_order() -> None:
  """Low-first sequence order is preserved in the tier-2 sample."""
  stat = _stat()
  week_table = stat.build_day_table(_both_break_week(high_at=700, low_at=600))
  samples = stat.classify_samples(week_table)
  assert [(s.date, s.condition, s.outcome) for s in samples] == [
    ("2024-01-08", "prior_range", "break_both"),
    ("2024-01-08", "break_both", "low_first"),
  ]


def test_classify_samples_empty_week_table() -> None:
  """Empty week_table -> classify_samples returns []."""
  stat = _stat()
  assert stat.classify_samples(stat.build_day_table(_empty_df())) == []


def test_classify_samples_excludes_first_and_pending_weeks() -> None:
  """The prior-less first week and the trailing pending week yield no samples."""
  stat = _stat()
  week_table = stat.build_day_table(make_candles(_SEQ))
  samples = stat.classify_samples(week_table)
  sample_dates = {s.date for s in samples}
  assert "2024-01-01" not in sample_dates
  assert "2024-02-05" not in sample_dates
  assert sample_dates == {"2024-01-08", "2024-01-15", "2024-01-22", "2024-01-29"}
