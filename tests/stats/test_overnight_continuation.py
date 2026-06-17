"""Tests for stats.overnight_continuation.standard.

All data is synthetic — no real market files required.
Expected values are hand-calculated before each assertion.

Matrix framing: condition = overnight gap color (today's session_open vs the
prior resolved session's session_close), outcome = intraday day color
(session_close vs session_open). The first resolved day has no prior close and
is excluded from every denominator (pending discipline).
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.overnight_continuation.standard import OvernightContinuation

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
  """One trading day of 1-min RTH bars (09:30–16:14).

  The opening bar carries session_open as its open price; the last bar carries
  session_close as its close price. All intermediate bars use neutral values
  (100.0) so they do not influence the stat.
  """
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


def make_candles(days: list[dict]) -> pd.DataFrame:
  """Concatenate per-day specs into a single sorted 1-min OHLCV DataFrame."""
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


def _stat() -> OvernightContinuation:
  return OvernightContinuation(instrument="NQ", config=_TEST_CONFIG)


def _row(result: StatRunResult, condition: str, outcome: str) -> StatResultRow:
  for r in result.instruments["NQ"]["daily"].results:
    if r.condition == condition and r.outcome == outcome:
      return r
  raise KeyError((condition, outcome))


# ===========================================================================
# Core matrix (9 days)
#
# Day-by-day derivation:
#   idx  date        open   close  prev_close  gap_color       intraday_color
#    0   2024-01-02  100.0  120.0  (none)      — (excluded)    G (close≥open)
#    1   2024-01-03  125.0   90.0  120.0       G (125≥120)     R (90<125)
#    2   2024-01-04   85.0   95.0   90.0       R (85<90)       G (95≥85)
#    3   2024-01-05  100.0  110.0   95.0       G (100≥95)      G (110≥100)
#    4   2024-01-08   80.0   75.0  110.0       R (80<110)      R (75<80)
#    5   2024-01-09  115.0  130.0   75.0       G (115≥75)      G (130≥115)
#    6   2024-01-10  140.0  120.0  130.0       G (140≥130)     R (120<140)
#    7   2024-01-11   70.0   80.0  120.0       R (70<120)      G (80≥70)
#    8   2024-01-12   90.0   85.0   80.0       G (90≥80)       R (85<90)
#
# Countable = idx 1..8 (8 days; idx 0 has no prior close → excluded from totals)
#
# gap_green days: idx 1(G→R), 3(G→G), 5(G→G), 6(G→R), 8(G→R)  → total=5
#   gap_green → green: 3, 5          → count=2, P=2/5=0.4
#   gap_green → red:   1, 6, 8       → count=3, P=3/5=0.6
#
# gap_red days:   idx 2(R→G), 4(R→R), 7(R→G)                   → total=3
#   gap_red → green:  2, 7           → count=2, P=2/3≈0.6667
#   gap_red → red:    4              → count=1, P=1/3≈0.3333
#
# total_samples = 9 (all resolved days including day 0)
# ===========================================================================

_SEQ = [
  {"date": "2024-01-02", "session_open": 100.0, "session_close": 120.0},  # idx 0 — excluded
  {"date": "2024-01-03", "session_open": 125.0, "session_close":  90.0},  # gap_green → red
  {"date": "2024-01-04", "session_open":  85.0, "session_close":  95.0},  # gap_red   → green
  {"date": "2024-01-05", "session_open": 100.0, "session_close": 110.0},  # gap_green → green
  {"date": "2024-01-08", "session_open":  80.0, "session_close":  75.0},  # gap_red   → red
  {"date": "2024-01-09", "session_open": 115.0, "session_close": 130.0},  # gap_green → green
  {"date": "2024-01-10", "session_open": 140.0, "session_close": 120.0},  # gap_green → red
  {"date": "2024-01-11", "session_open":  70.0, "session_close":  80.0},  # gap_red   → green
  {"date": "2024-01-12", "session_open":  90.0, "session_close":  85.0},  # gap_green → red
]


def test_total_samples_counts_all_resolved_days() -> None:
  """total_samples = 9 resolved sessions (the first one is still resolved)."""
  result = _stat().compute(make_candles(_SEQ))
  assert result.instruments["NQ"]["daily"].total_samples == 9


def test_gap_green_conditional() -> None:
  """Given a green gap: P(green day)=2/5, P(red day)=3/5 over 5 countable days."""
  result = _stat().compute(make_candles(_SEQ))
  green = _row(result, "gap_green", "green")
  red = _row(result, "gap_green", "red")
  # gap_green total = 5 (idx 1,3,5,6,8)
  assert green.total == red.total == 5
  assert green.count == 2
  assert green.probability == pytest.approx(2 / 5)
  assert red.count == 3
  assert red.probability == pytest.approx(3 / 5)


def test_gap_red_conditional() -> None:
  """Given a red gap: P(green day)=2/3, P(red day)=1/3 over 3 countable days."""
  result = _stat().compute(make_candles(_SEQ))
  green = _row(result, "gap_red", "green")
  red = _row(result, "gap_red", "red")
  # gap_red total = 3 (idx 2,4,7)
  assert green.total == red.total == 3
  assert green.count == 2
  assert green.probability == pytest.approx(2 / 3)
  assert red.count == 1
  assert red.probability == pytest.approx(1 / 3)


def test_first_day_excluded_from_denominators() -> None:
  """Sum of all condition totals = 8 = n_resolved - 1 (first day has no prior close)."""
  result = _stat().compute(make_candles(_SEQ))
  gap_green_total = _row(result, "gap_green", "green").total
  gap_red_total = _row(result, "gap_red", "green").total
  # 9 resolved days; first excluded → 8 countable
  assert gap_green_total + gap_red_total == 9 - 1


def test_outcomes_partition_each_condition() -> None:
  """green.count + red.count == total for each gap color condition."""
  result = _stat().compute(make_candles(_SEQ))
  for cond in ("gap_green", "gap_red"):
    green = _row(result, cond, "green")
    red = _row(result, cond, "red")
    assert green.count + red.count == green.total == red.total


def test_four_rows_only() -> None:
  """Exactly four rows: the 2x2 (gap color × day color) matrix."""
  result = _stat().compute(make_candles(_SEQ))
  rows = result.instruments["NQ"]["daily"].results
  assert {(r.condition, r.outcome) for r in rows} == {
    ("gap_green", "green"), ("gap_green", "red"),
    ("gap_red", "green"), ("gap_red", "red"),
  }


def test_data_range() -> None:
  """data_range spans from the first to the last resolved session date."""
  result = _stat().compute(make_candles(_SEQ))
  assert result.instruments["NQ"]["daily"].data_range == ["2024-01-02", "2024-01-12"]


# ===========================================================================
# Boundary: flat gap and flat intraday are both classified green
# ===========================================================================

def test_flat_gap_classified_green() -> None:
  """session_open == prev_session_close → gap_green (>= is green)."""
  # Day 0: close=100; Day 1: open=100 (flat gap, == prev_close → gap_green)
  days = [
    {"date": "2024-01-02", "session_open": 100.0, "session_close": 100.0},
    {"date": "2024-01-03", "session_open": 100.0, "session_close": 110.0},
  ]
  result = _stat().compute(make_candles(days))
  # Only 1 countable day (idx 1); gap must be classified gap_green
  gg_green = _row(result, "gap_green", "green")
  assert gg_green.total == 1
  assert gg_green.count == 1  # intraday also green (110 >= 100)


def test_flat_intraday_classified_green_day() -> None:
  """session_close == session_open → green day (>= is green)."""
  # Day 0: close=100; Day 1: open=110 (gap_green: 110≥100), close=110 (flat intraday)
  days = [
    {"date": "2024-01-02", "session_open": 100.0, "session_close": 100.0},
    {"date": "2024-01-03", "session_open": 110.0, "session_close": 110.0},
  ]
  result = _stat().compute(make_candles(days))
  # Day 1: gap_green (110≥100), intraday green (110≥110)
  gg_green = _row(result, "gap_green", "green")
  assert gg_green.total == 1
  assert gg_green.count == 1


# ===========================================================================
# Edge: all-green-gap sequence
# ===========================================================================

def test_all_green_gap_no_red_gap_condition() -> None:
  """All gaps green → gap_red condition is empty (total=0, probability=0.0, no crash)."""
  # 5 days where each day opens above the prior close
  days = [
    {"date": "2024-01-02", "session_open": 100.0, "session_close": 110.0},
    {"date": "2024-01-03", "session_open": 115.0, "session_close": 120.0},
    {"date": "2024-01-04", "session_open": 125.0, "session_close": 130.0},
    {"date": "2024-01-05", "session_open": 135.0, "session_close": 140.0},
    {"date": "2024-01-08", "session_open": 145.0, "session_close": 150.0},
  ]
  result = _stat().compute(make_candles(days))
  # All 4 countable days have gap_green
  gg_green = _row(result, "gap_green", "green")
  assert gg_green.total == 4
  assert gg_green.probability == pytest.approx(1.0)
  # gap_red is empty
  gr_green = _row(result, "gap_red", "green")
  assert gr_green.total == 0
  assert gr_green.probability == pytest.approx(0.0)  # no ZeroDivisionError


# ===========================================================================
# Baseline ≈ 50% and deterministic
# ===========================================================================

def _long_seq() -> pd.DataFrame:
  """~250 weekdays, deterministic pattern with strong gap continuation signal."""
  dates: list[str] = []
  d = pd.Timestamp("2020-01-01", tz=_NY)
  while len(dates) < 250:
    if d.weekday() < 5:
      dates.append(d.strftime("%Y-%m-%d"))
    d += pd.Timedelta(days=1)

  days = []
  # Alternating rising/falling pattern: each open is above/below the prior close
  # by a fixed amount, making the gap direction perfectly predictable.
  close = 100.0
  for i, date in enumerate(dates):
    if i % 4 < 2:
      # Two gap_green days: open above prior close, then close above open (green day)
      open_ = close + 5.0
      close_new = open_ + 3.0
    else:
      # Two gap_red days: open below prior close, then close below open (red day)
      open_ = close - 5.0
      close_new = open_ - 3.0
    days.append({"date": date, "session_open": open_, "session_close": close_new})
    close = close_new

  return make_candles(days)


def test_baseline_approx_fifty_percent() -> None:
  """Random baseline probability is near 0.5 for every matrix row."""
  stat = _stat()
  rows = stat.baseline(_long_seq(), seed=42)
  for row in rows:
    assert row.probability == pytest.approx(0.5, abs=0.12)


def test_baseline_deterministic() -> None:
  """Same seed produces identical baseline results across two calls."""
  stat = _stat()
  df = _long_seq()
  rows_a = stat.baseline(df, seed=7)
  rows_b = stat.baseline(df, seed=7)
  for a, b in zip(rows_a, rows_b):
    assert (a.condition, a.outcome) == (b.condition, b.outcome)
    assert a.probability == pytest.approx(b.probability)


def test_baseline_embedded_in_compute() -> None:
  """After compute(), every row carries a positive baseline_n."""
  result = _stat().compute(_long_seq())
  for row in result.instruments["NQ"]["daily"].results:
    assert row.baseline_n > 0


def test_real_signal_beats_baseline() -> None:
  """Strong continuation pattern: gap_green→green probability exceeds its baseline."""
  # _long_seq has a perfect gap→intraday continuation pattern
  result = _stat().compute(_long_seq())
  gg_green = _row(result, "gap_green", "green")
  # The real signal (continuation) must exceed the ≈0.5 random baseline
  assert gg_green.probability > gg_green.baseline_prob


# ===========================================================================
# Pending exclusion
# ===========================================================================

def test_pending_day_excluded_from_total_samples() -> None:
  """A truncated/early-close day does not appear in total_samples."""
  stat = _stat()
  base_df = make_candles(_SEQ)
  total_base = stat.compute(base_df).instruments["NQ"]["daily"].total_samples
  truncated = _make_truncated_day("2024-01-15")
  combined = (
    pd.concat([base_df, truncated], ignore_index=True)
    .sort_values("timestamp")
    .reset_index(drop=True)
  )
  total_with = stat.compute(combined).instruments["NQ"]["daily"].total_samples
  # Truncated day must not increase the count
  assert total_with == total_base == 9


def test_pending_day_absent_from_day_table() -> None:
  """build_day_table excludes the pending (truncated) day entirely."""
  stat = _stat()
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
  """Empty input → zero samples, empty data_range, all rows zeroed."""
  result = _stat().compute(_empty_df())
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 0
  assert tf.data_range == []
  for row in tf.results:
    assert row.count == 0
    assert row.total == 0
    assert row.probability == pytest.approx(0.0)


def test_empty_dataframe_no_slices() -> None:
  """overnight_continuation declares no slices → slices == {}."""
  result = _stat().compute(_empty_df())
  assert result.instruments["NQ"]["daily"].slices == {}


def test_slices_empty_on_real_data() -> None:
  """slices is {} even with a full dataset (no slice dimensions declared)."""
  result = _stat().compute(make_candles(_SEQ))
  assert result.instruments["NQ"]["daily"].slices == {}


def test_single_resolved_day_no_countable_rows() -> None:
  """One resolved session → no prior close → all condition totals zero, no crash."""
  days = [{"date": "2024-03-01", "session_open": 100.0, "session_close": 110.0}]
  result = _stat().compute(make_candles(days))
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 1
  for row in tf.results:
    assert row.total == 0


# ===========================================================================
# i18n
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
  assert set(result.labels.conditions) == {"gap_green", "gap_red"}
  assert set(result.labels.outcomes) == {"green", "red"}


# ===========================================================================
# stat_name and write_results round-trip
# ===========================================================================

def test_stat_name() -> None:
  assert _stat().compute(_empty_df()).stat_name == "overnight_continuation"


def test_write_results_round_trip(tmp_path: Path) -> None:
  """write_results produces overnight_continuation.json that re-validates correctly."""
  result = _stat().compute(make_candles(_SEQ))
  written = write_results(result, results_dir=tmp_path)
  assert written.name == "overnight_continuation.json"

  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)
  assert validated.instruments["NQ"]["daily"].total_samples == 9


def test_write_results_french_accent_literal(tmp_path: Path) -> None:
  """French text is stored as literal UTF-8 characters, not escaped unicode."""
  result = _stat().compute(_empty_df())
  written = write_results(result, results_dir=tmp_path)
  raw = written.read_text(encoding="utf-8")
  # The French definition contains "précédente" (with é)
  assert "é" in raw
  assert "\\u00e9" not in raw
