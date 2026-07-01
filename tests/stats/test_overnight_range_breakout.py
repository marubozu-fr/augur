"""Tests for stats.overnight_range_breakout.standard.

All data is synthetic — no real market files required. Expected values are
hand-calculated before each assertion.

Framing: a single ``overnight_range`` condition with four mutually exclusive
outcomes (``broke_high`` / ``broke_low`` / ``broke_both`` / ``neither``) that
partition every countable day. The "range" is the PRIOR overnight session's
high/low; the breakout window is the whole RTH session. Strict inequality defines
a break (touching the level is not a break).

Each synthetic session carries an overnight bar (controlling ``on_high`` /
``on_low``) plus RTH bars 09:30–16:14 (controlling ``day_high`` / ``day_low`` and
the session open/close). By default the overnight bar is placed at 08:00 on the
session date itself (minute-of-day 480 < 09:30 → it belongs to that date's
overnight session); a flag places it on the prior evening instead to exercise the
cross-midnight attribution.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.overnight_range_breakout.standard import OvernightRangeBreakout

# ---------------------------------------------------------------------------
# Minimal InstrumentConfig (does not depend on NQ.yaml)
# ---------------------------------------------------------------------------
_NY = "America/New_York"
_RTH_SESSION = Session(start="09:30", end="16:15")
_ON_SESSION = Session(start="18:00", end="09:30")
_TEST_CONFIG = InstrumentConfig(
  instrument="NQ",
  working_timezone=_NY,
  sessions={"rth": _RTH_SESSION, "overnight": _ON_SESSION},
  timeframes=["daily"],
  parquet_path=Path("data/NQ_1min.parquet"),
)

_RTH_START = 570   # 09:30
_RTH_LAST = 974    # 16:14 (last bar before 16:15); resolved needs last mod >= 960


def _make_session(
  date: str,
  on_high: float,
  on_low: float,
  day_high: float,
  day_low: float,
  session_open: float | None = None,
  session_close: float | None = None,
  overnight_prev_evening: bool = False,
) -> pd.DataFrame:
  """One trading session: an overnight bar plus RTH bars (09:30–16:14).

  The overnight bar carries ``on_high`` / ``on_low``. The 09:30 bar carries
  ``day_high`` / ``day_low`` (the RTH extremes) and ``session_open``; the 16:14
  bar carries ``session_close``. Intermediate bars sit at the RTH midpoint so they
  never affect the extremes. ``session_open`` / ``session_close`` default to the
  RTH midpoint (a flat green session).
  """
  base = pd.Timestamp(date, tz=_NY)
  mid = (day_high + day_low) / 2.0
  if session_open is None:
    session_open = mid
  if session_close is None:
    session_close = mid

  records = []
  # Overnight bar. Default: 08:00 on `date` (mod 480 < 09:30 → same-day overnight).
  # overnight_prev_evening: 19:00 on the prior day (mod 1140 >= 18:00 → next-day
  # overnight, which is `date`) — exercises the cross-midnight attribution.
  if overnight_prev_evening:
    on_ts = (base - pd.Timedelta(days=1)).replace(hour=19, minute=0, second=0, microsecond=0)
  else:
    on_ts = base.replace(hour=8, minute=0, second=0, microsecond=0)
  records.append({
    "timestamp": on_ts, "open": on_low, "high": on_high, "low": on_low,
    "close": on_high, "volume": 100,
  })

  for mod in range(_RTH_START, _RTH_LAST + 1):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    if mod == _RTH_START:
      o, hi, lo, c = session_open, day_high, day_low, mid
    elif mod == _RTH_LAST:
      # Keep the bar OHLC-consistent when session_close != mid; the RTH extremes
      # are still driven by the 09:30 bar (day_high/day_low), and every caller
      # keeps session_close within [day_low, day_high], so high/low here never
      # exceed those extremes.
      o, hi, lo, c = mid, max(mid, session_close), min(mid, session_close), session_close
    else:
      o, hi, lo, c = mid, mid, mid, mid
    records.append({
      "timestamp": ts, "open": o, "high": hi, "low": lo, "close": c, "volume": 1000,
    })
  return pd.DataFrame(records)


def make_candles(sessions: list[dict]) -> pd.DataFrame:
  """Concatenate per-session specs into a single sorted 1-min OHLCV DataFrame."""
  frames = [_make_session(**s) for s in sessions]
  df = pd.concat(frames, ignore_index=True)
  return df.sort_values("timestamp").reset_index(drop=True)


def _make_truncated_session(date: str) -> pd.DataFrame:
  """A session whose last RTH bar is 09:50 (mod 590 < 960) — unresolved, excluded."""
  base = pd.Timestamp(date, tz=_NY)
  records = [{
    "timestamp": base.replace(hour=8, minute=0), "open": 90.0, "high": 110.0,
    "low": 90.0, "close": 110.0, "volume": 100,
  }]
  for mod in range(_RTH_START, 591):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    records.append({
      "timestamp": ts, "open": 100.0, "high": 120.0, "low": 80.0,
      "close": 100.0, "volume": 500,
    })
  return pd.DataFrame(records)


def _stat(breakout_criteria: str = "wick") -> OvernightRangeBreakout:
  return OvernightRangeBreakout(
    instrument="NQ", config=_TEST_CONFIG, breakout_criteria=breakout_criteria
  )


def _row(result: StatRunResult, outcome: str) -> StatResultRow:
  for r in result.instruments["NQ"]["daily"].results:
    if r.condition == "overnight_range" and r.outcome == outcome:
      return r
  raise KeyError(outcome)


# ===========================================================================
# Core 4-outcome partition
#
# Overnight range fixed at [90, 110] (on_size = 20) for every day. The RTH
# extremes select each outcome (strict break):
#   broke_high : day_high 115 > 110, day_low 95 >= 90        (up only)
#   broke_low  : day_high 105 <= 110, day_low 85 < 90        (down only)
#   broke_both : day_high 115 > 110, day_low 85 < 90         (both)
#   neither    : day_high 108 <= 110, day_low 92 >= 90       (inside)
# Two of each → every outcome count = 2, total = 8.
# ===========================================================================

_SEQ = [
  {"date": "2024-01-02", "on_high": 110, "on_low": 90, "day_high": 115, "day_low": 95},  # broke_high
  {"date": "2024-01-03", "on_high": 110, "on_low": 90, "day_high": 105, "day_low": 85},  # broke_low
  {"date": "2024-01-04", "on_high": 110, "on_low": 90, "day_high": 115, "day_low": 85},  # broke_both
  {"date": "2024-01-05", "on_high": 110, "on_low": 90, "day_high": 108, "day_low": 92},  # neither
  {"date": "2024-01-08", "on_high": 110, "on_low": 90, "day_high": 115, "day_low": 95},  # broke_high
  {"date": "2024-01-09", "on_high": 110, "on_low": 90, "day_high": 105, "day_low": 85},  # broke_low
  {"date": "2024-01-10", "on_high": 110, "on_low": 90, "day_high": 115, "day_low": 85},  # broke_both
  {"date": "2024-01-11", "on_high": 110, "on_low": 90, "day_high": 108, "day_low": 92},  # neither
]


def test_total_samples_counts_all_resolved_days() -> None:
  """All 8 sessions are resolved AND have a same-day overnight range → 8 countable."""
  result = _stat().compute(make_candles(_SEQ))
  assert result.instruments["NQ"]["daily"].total_samples == 8


def test_four_outcomes_partition() -> None:
  """Each outcome count = 2, total = 8, probabilities = 0.25 each."""
  result = _stat().compute(make_candles(_SEQ))
  for outcome in ("broke_high", "broke_low", "broke_both", "neither"):
    row = _row(result, outcome)
    assert row.count == 2, outcome
    assert row.total == 8, outcome
    assert row.probability == pytest.approx(0.25), outcome


def test_outcome_counts_sum_to_total() -> None:
  """The four mutually exclusive outcomes sum to the countable total."""
  result = _stat().compute(make_candles(_SEQ))
  rows = result.instruments["NQ"]["daily"].results
  assert sum(r.count for r in rows) == 8
  assert {r.total for r in rows} == {8}


def test_four_rows_only() -> None:
  """Exactly the four breakout outcomes are reported under one condition."""
  result = _stat().compute(make_candles(_SEQ))
  rows = result.instruments["NQ"]["daily"].results
  assert {(r.condition, r.outcome) for r in rows} == {
    ("overnight_range", "broke_high"),
    ("overnight_range", "broke_low"),
    ("overnight_range", "broke_both"),
    ("overnight_range", "neither"),
  }


def test_data_range() -> None:
  """data_range spans from the first to the last resolved session date."""
  result = _stat().compute(make_candles(_SEQ))
  assert result.instruments["NQ"]["daily"].data_range == ["2024-01-02", "2024-01-11"]


# ===========================================================================
# Strict-break boundary: touching a level exactly is NOT a break
# ===========================================================================

def test_touching_level_is_not_a_break() -> None:
  """day_high == on_high and day_low == on_low → neither (boundaries inclusive inside)."""
  days = [{"date": "2024-02-01", "on_high": 110, "on_low": 90, "day_high": 110, "day_low": 90}]
  result = _stat().compute(make_candles(days))
  assert _row(result, "neither").count == 1
  assert _row(result, "broke_high").count == 0
  assert _row(result, "broke_low").count == 0


# ===========================================================================
# Cross-midnight overnight attribution
# ===========================================================================

def test_overnight_range_from_prior_evening() -> None:
  """An overnight bar on the prior evening (19:00) forms this session's range."""
  days = [{
    "date": "2024-03-04", "on_high": 110, "on_low": 90,
    "day_high": 115, "day_low": 95, "overnight_prev_evening": True,
  }]
  result = _stat().compute(make_candles(days))
  # Range [90, 110] sourced from the prior evening; RTH high 115 > 110 → broke_high.
  assert result.instruments["NQ"]["daily"].total_samples == 1
  assert _row(result, "broke_high").count == 1


def test_overnight_range_present_in_day_table() -> None:
  """build_day_table exposes the on_high/on_low/on_size columns for the session."""
  day_table = _stat().build_day_table(make_candles(_SEQ))
  d = pd.Timestamp("2024-01-02", tz=_NY).normalize()
  assert day_table.loc[d, "on_high"] == 110
  assert day_table.loc[d, "on_low"] == 90
  assert day_table.loc[d, "on_size"] == 20


# ===========================================================================
# close vs wick breakout criteria
# ===========================================================================

def test_close_criteria_uses_bar_closes() -> None:
  """With close criteria a wick beyond the range does not count; the close must."""
  # RTH wick reaches 115 (> 110) but every close stays at/below 110, so under the
  # close criteria the day is `neither`; under wick it would be `broke_high`.
  days = [{
    "date": "2024-04-01", "on_high": 110, "on_low": 90,
    "day_high": 115, "day_low": 95, "session_open": 100.0, "session_close": 105.0,
  }]
  candles = make_candles(days)
  # Wick: high 115 > 110 → broke_high.
  assert _row(_stat("wick").compute(candles), "broke_high").count == 1
  # Close: max close = 105 (mid) <= 110 → neither.
  result_close = _stat("close").compute(candles)
  assert _row(result_close, "neither").count == 1
  assert _row(result_close, "broke_high").count == 0


def test_invalid_breakout_criteria_raises() -> None:
  with pytest.raises(ValueError):
    OvernightRangeBreakout(instrument="NQ", config=_TEST_CONFIG, breakout_criteria="bogus")


# ===========================================================================
# Slices
# ===========================================================================

def test_declared_slice_dimensions_present() -> None:
  """All declared slice dimensions appear in the timeframe slices."""
  result = _stat().compute(make_candles(_SEQ))
  slices = result.instruments["NQ"]["daily"].slices
  assert set(slices) == {"weekday", "close", "prev_candle", "size", "levels"}


def test_size_slice_buckets_overnight_range() -> None:
  """The size slice buckets days by overnight-range size (on_size)."""
  # Four distinct overnight-range sizes → four quartile buckets, one day each.
  days = [
    {"date": "2024-05-01", "on_high": 105, "on_low": 100, "day_high": 110, "day_low": 95},   # size 5
    {"date": "2024-05-02", "on_high": 110, "on_low": 100, "day_high": 115, "day_low": 95},   # size 10
    {"date": "2024-05-03", "on_high": 120, "on_low": 100, "day_high": 125, "day_low": 95},   # size 20
    {"date": "2024-05-06", "on_high": 140, "on_low": 100, "day_high": 145, "day_low": 95},   # size 40
  ]
  result = _stat().compute(make_candles(days))
  size = result.instruments["NQ"]["daily"].slices["size"]
  # qcut on 4 distinct values → 4 single-day groups.
  assert len(size.groups) == 4
  assert all(g.total_samples == 1 for g in size.groups.values())


def test_levels_slice_uses_extension_multiples() -> None:
  """The levels slice bands the breakout extension in multiples of on_size."""
  # on_size = 20. Extensions chosen to land in distinct bands:
  #   ext 5  → 0.25x  → 0–0.5x band
  #   ext 15 → 0.75x  → 0.5–1x band
  #   ext 30 → 1.5x   → 1.5–2x band
  days = [
    {"date": "2024-06-03", "on_high": 110, "on_low": 90, "day_high": 115, "day_low": 95},   # ext 5
    {"date": "2024-06-04", "on_high": 110, "on_low": 90, "day_high": 125, "day_low": 95},   # ext 15
    {"date": "2024-06-05", "on_high": 110, "on_low": 90, "day_high": 140, "day_low": 95},   # ext 30
  ]
  result = _stat().compute(make_candles(days))
  levels = result.instruments["NQ"]["daily"].slices["levels"]
  keys = set(levels.groups)
  assert "0_0_5x" in keys
  assert "0_5_1x" in keys
  assert "1_5_2x" in keys


def test_close_slice_splits_by_session_color() -> None:
  """The close slice partitions days into green / red by session color."""
  days = [
    # green: close (115) >= open (95)
    {"date": "2024-07-01", "on_high": 110, "on_low": 90, "day_high": 120, "day_low": 90,
     "session_open": 95.0, "session_close": 115.0},
    # red: close (95) < open (115)
    {"date": "2024-07-02", "on_high": 110, "on_low": 90, "day_high": 120, "day_low": 90,
     "session_open": 115.0, "session_close": 95.0},
  ]
  result = _stat().compute(make_candles(days))
  close = result.instruments["NQ"]["daily"].slices["close"]
  assert close.groups["green"].total_samples == 1
  assert close.groups["red"].total_samples == 1


# ===========================================================================
# Baseline ≈ direction-symmetric and deterministic
# ===========================================================================

def _long_seq() -> pd.DataFrame:
  """~250 weekdays with a strong upside-break bias (mostly broke_high)."""
  dates: list[str] = []
  d = pd.Timestamp("2020-01-01", tz=_NY)
  while len(dates) < 250:
    if d.weekday() < 5:
      dates.append(d.strftime("%Y-%m-%d"))
    d += pd.Timedelta(days=1)

  days = []
  for i, date in enumerate(dates):
    if i % 5 == 0:
      # occasional downside-only break to keep both outcomes populated
      days.append({"date": date, "on_high": 110, "on_low": 90, "day_high": 105, "day_low": 80})
    else:
      # upside-only break
      days.append({"date": date, "on_high": 110, "on_low": 90, "day_high": 120, "day_low": 95})
  return make_candles(days)


def test_baseline_symmetric_high_low() -> None:
  """Reflection baseline removes directional bias: broke_high ≈ broke_low baseline."""
  stat = _stat()
  rows = {r.outcome: r for r in stat.baseline(_long_seq(), seed=42)}
  assert rows["broke_high"].probability == pytest.approx(rows["broke_low"].probability, abs=0.12)


def test_baseline_deterministic() -> None:
  """Same seed produces identical baseline results across two calls."""
  stat = _stat()
  df = _long_seq()
  rows_a = stat.baseline(df, seed=7)
  rows_b = stat.baseline(df, seed=7)
  for a, b in zip(rows_a, rows_b):
    assert (a.condition, a.outcome) == (b.condition, b.outcome)
    assert a.probability == pytest.approx(b.probability)


def test_real_directional_signal_beats_baseline() -> None:
  """Strong upside-break bias: broke_high probability exceeds its reflection baseline."""
  result = _stat().compute(_long_seq())
  bh = _row(result, "broke_high")
  assert bh.probability > bh.baseline_prob


def test_baseline_embedded_in_compute() -> None:
  """After compute(), every row carries a positive baseline_n."""
  result = _stat().compute(_long_seq())
  for row in result.instruments["NQ"]["daily"].results:
    assert row.baseline_n > 0


# ===========================================================================
# Pending exclusion
# ===========================================================================

def test_pending_day_excluded_from_total_samples() -> None:
  """A truncated/early-close session does not appear in total_samples."""
  stat = _stat()
  base_df = make_candles(_SEQ)
  total_base = stat.compute(base_df).instruments["NQ"]["daily"].total_samples
  truncated = _make_truncated_session("2024-01-15")
  combined = (
    pd.concat([base_df, truncated], ignore_index=True)
    .sort_values("timestamp")
    .reset_index(drop=True)
  )
  total_with = stat.compute(combined).instruments["NQ"]["daily"].total_samples
  assert total_with == total_base == 8


def test_pending_day_absent_from_day_table() -> None:
  """build_day_table excludes the pending (truncated) session entirely."""
  stat = _stat()
  day_table = stat.build_day_table(_make_truncated_session("2024-01-15"))
  pending = pd.Timestamp("2024-01-15", tz=_NY).normalize()
  assert pending not in day_table.index


def test_session_without_overnight_range_excluded() -> None:
  """A resolved RTH session with no overnight bars is dropped (inner join)."""
  base = pd.Timestamp("2024-08-01", tz=_NY)
  mid = 100.0
  records = []
  for mod in range(_RTH_START, _RTH_LAST + 1):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    hi = 120.0 if mod == _RTH_START else mid
    lo = 80.0 if mod == _RTH_START else mid
    records.append({
      "timestamp": ts, "open": mid, "high": hi, "low": lo, "close": mid, "volume": 1000,
    })
  result = _stat().compute(pd.DataFrame(records))
  # Resolved RTH day but no overnight bars → not countable.
  assert result.instruments["NQ"]["daily"].total_samples == 0


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


def test_single_day_all_outcomes_present() -> None:
  """One countable day → exactly four rows, totals = 1, one outcome carries it."""
  days = [{"date": "2024-09-03", "on_high": 110, "on_low": 90, "day_high": 115, "day_low": 95}]
  result = _stat().compute(make_candles(days))
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 1
  assert _row(result, "broke_high").count == 1
  assert {r.total for r in tf.results} == {1}


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
  assert set(result.labels.conditions) == {"overnight_range"}
  assert set(result.labels.outcomes) == {"broke_high", "broke_low", "broke_both", "neither"}


# ===========================================================================
# stat_name and write_results round-trip
# ===========================================================================

def test_stat_name() -> None:
  assert _stat().compute(_empty_df()).stat_name == "overnight_range_breakout"


def test_write_results_round_trip(tmp_path: Path) -> None:
  """write_results produces overnight_range_breakout.json that re-validates."""
  result = _stat().compute(make_candles(_SEQ))
  written = write_results(result, results_dir=tmp_path)
  assert written.name == "overnight_range_breakout.json"

  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)
  assert validated.instruments["NQ"]["daily"].total_samples == 8


def test_write_results_french_accent_literal(tmp_path: Path) -> None:
  """French text is stored as literal UTF-8 characters, not escaped unicode."""
  result = _stat().compute(_empty_df())
  written = write_results(result, results_dir=tmp_path)
  raw = written.read_text(encoding="utf-8")
  assert "é" in raw
  assert "\\u00e9" not in raw


# ===========================================================================
# classify_samples
#
# Reusing _SEQ (8 countable days, hand-calculated outcomes above):
#   2024-01-02 broke_high, 2024-01-03 broke_low, 2024-01-04 broke_both,
#   2024-01-05 neither, 2024-01-08 broke_high, 2024-01-09 broke_low,
#   2024-01-10 broke_both, 2024-01-11 neither
# ===========================================================================

def test_classify_samples_exact_list() -> None:
  """classify_samples emits one SampleRow per countable day, matching the hand-calc."""
  stat = _stat()
  day_table = stat.build_day_table(make_candles(_SEQ))
  samples = stat.classify_samples(day_table)
  assert [(s.date, s.condition, s.outcome) for s in samples] == [
    ("2024-01-02", "overnight_range", "broke_high"),
    ("2024-01-03", "overnight_range", "broke_low"),
    ("2024-01-04", "overnight_range", "broke_both"),
    ("2024-01-05", "overnight_range", "neither"),
    ("2024-01-08", "overnight_range", "broke_high"),
    ("2024-01-09", "overnight_range", "broke_low"),
    ("2024-01-10", "overnight_range", "broke_both"),
    ("2024-01-11", "overnight_range", "neither"),
  ]
  assert all(s.value is None for s in samples)


def test_classify_samples_matches_compute_rows_counts() -> None:
  """For every StatResultRow, the matching SampleRow count equals r.count and total."""
  stat = _stat()
  day_table = stat.build_day_table(make_candles(_SEQ))
  samples = stat.classify_samples(day_table)
  rows = stat.compute_rows(day_table)
  for r in rows:
    n = sum(1 for s in samples if s.condition == r.condition and s.outcome == r.outcome)
    assert n == r.count
  total = sum(1 for s in samples if s.condition == "overnight_range")
  assert total == rows[0].total == 8


def test_classify_samples_empty_day_table() -> None:
  """Empty day_table -> classify_samples returns []."""
  stat = _stat()
  assert stat.classify_samples(stat.build_day_table(_empty_df())) == []


def test_classify_samples_excludes_pending_session() -> None:
  """A trailing truncated (unresolved) session yields no samples."""
  base_df = make_candles(_SEQ)
  truncated = _make_truncated_session("2024-01-12")
  combined = (
    pd.concat([base_df, truncated], ignore_index=True)
    .sort_values("timestamp")
    .reset_index(drop=True)
  )
  stat = _stat()
  day_table = stat.build_day_table(combined)
  samples = stat.classify_samples(day_table)
  sample_dates = {s.date for s in samples}
  assert "2024-01-12" not in sample_dates
  assert len(samples) == 8
