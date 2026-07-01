"""Tests for stats.initial_balance.time (InitialBalanceTime).

All data is synthetic — no real market files required. Expected breakout times and
bucket assignments are hand-calculated before each assertion.

Framing (two partitions over the same denominator): given the initial balance
formed from the first ``ib_period`` minutes of the session, WHEN does price first
break either side of it during the breakout window? Reported as a coarse early/late
split around a threshold (``timing`` condition) and as the full first-breakout-time
histogram (``bucket`` condition). Sessions that never break either side are excluded
from every denominator.

A flexible day builder (``_make_day``) fills a dense RTH session with neutral bars
at ``base_price`` (inside the IB) and overrides specific breakout-window minutes
with explicit ``(high, low, close)`` bars, so each break minute can be scripted.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.initial_balance.time import InitialBalanceTime

# ---------------------------------------------------------------------------
# Minimal InstrumentConfig (does not depend on NQ.yaml)
# ---------------------------------------------------------------------------
_NY = "America/New_York"
_RTH_SESSION = Session(start="09:30", end="16:15")
_TEST_CONFIG = InstrumentConfig(
  instrument="NQ",
  working_timezone=_NY,
  sessions={"rth": _RTH_SESSION},
  timeframes=["30min", "1h"],
  parquet_path=Path("data/NQ_1min.parquet"),
)

_RTH_START = 570   # 09:30
_RTH_LAST = 974    # 16:14 (last bar before 16:15); resolved needs last mod >= 960
_IB_END_30 = _RTH_START + 30   # 10:00 -> 600
_THRESHOLD = _RTH_START + 150  # 12:00 -> 720 (default early/late boundary)
_BASE = 100.0


def _make_day(
  date: str,
  *,
  ib_high: float = 110.0,
  ib_low: float = 90.0,
  events: dict[int, tuple[float, float, float]] | None = None,
  base_price: float = _BASE,
  session_open: float | None = None,
  session_close: float | None = None,
  last_mod: int = _RTH_LAST,
) -> pd.DataFrame:
  """One dense day of 1-min RTH bars with a controllable initial balance.

  The 09:30 bar carries ``ib_high`` / ``ib_low`` as wicks (so they define the IB
  extremes) and opens at ``session_open``. Every other minute defaults to a neutral
  bar at ``base_price`` (kept strictly inside the IB so it never breaks). ``events``
  overrides specific minutes with an explicit ``(high, low, close)`` bar — used to
  script the breakout window. The last bar's close is set to ``session_close`` (kept
  inside the IB by callers so it does not itself break).
  """
  events = events or {}
  base = pd.Timestamp(date, tz=_NY)
  if session_open is None:
    session_open = base_price
  if session_close is None:
    session_close = base_price

  records = []
  for mod in range(_RTH_START, last_mod + 1):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    if mod == _RTH_START:
      o, hi, lo, c = session_open, ib_high, ib_low, base_price
    elif mod in events:
      ehi, elo, ec = events[mod]
      hi, lo, c = ehi, elo, ec
      o = max(min(base_price, hi), lo)
    else:
      o, hi, lo, c = base_price, base_price, base_price, base_price
    if mod == last_mod:
      c = session_close
      hi = max(hi, c)
      lo = min(lo, c)
    records.append(
      {"timestamp": ts, "open": o, "high": hi, "low": lo, "close": c, "volume": 1000}
    )
  return pd.DataFrame(records)


def _make_truncated_day(date: str) -> pd.DataFrame:
  """A day whose last RTH bar is 09:50 (mod 590 < 960) — unresolved, excluded."""
  base = pd.Timestamp(date, tz=_NY)
  records = []
  for mod in range(_RTH_START, 591):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    records.append(
      {"timestamp": ts, "open": 100.0, "high": 100.25, "low": 99.75,
       "close": 100.0, "volume": 500}
    )
  return pd.DataFrame(records)


def _concat(days: list[pd.DataFrame]) -> pd.DataFrame:
  df = pd.concat(days, ignore_index=True)
  return df.sort_values("timestamp").reset_index(drop=True)


def _empty_df() -> pd.DataFrame:
  df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
  df["timestamp"] = pd.array([], dtype="datetime64[ns, America/New_York]")
  return df


def _stat(timeframe: str = "30min", breakout_criteria: str = "wick") -> InitialBalanceTime:
  return InitialBalanceTime(
    instrument="NQ",
    timeframe=timeframe,
    config=_TEST_CONFIG,
    breakout_criteria=breakout_criteria,
  )


def _rows(stat: InitialBalanceTime, candles: pd.DataFrame) -> dict[tuple[str, str], StatResultRow]:
  result = stat.compute(candles)
  return {
    (r.condition, r.outcome): r
    for r in result.instruments["NQ"][stat.timeframe].results
  }


def _break_mod(stat: InitialBalanceTime, candles: pd.DataFrame) -> dict[str, float]:
  table = stat.build_day_table(candles)
  return {d.strftime("%Y-%m-%d"): v for d, v in table["first_break_mod"].items()}


# ===========================================================================
# First-breakout time: which bar, which side
# ===========================================================================
def test_single_up_breakout_time() -> None:
  # 10:00 bar wicks to 115 (> ib_high 110). First (only) break at mod 600.
  candles = _concat([_make_day("2024-01-02", events={_IB_END_30: (115.0, 100.0, 100.0)})])
  assert _break_mod(_stat(), candles)["2024-01-02"] == 600
  rows = _rows(_stat(), candles)
  assert rows[("timing", "early")].count == 1
  assert rows[("timing", "early")].total == 1
  assert rows[("timing", "late")].count == 0
  # 10:00 falls in bucket_1 = [600, 630).
  assert rows[("bucket", "bucket_1")].count == 1
  assert rows[("bucket", "bucket_0")].count == 0


def test_single_down_breakout_time() -> None:
  # 10:05 bar wicks to 85 (< ib_low 90). First break at mod 605 (bucket_1).
  candles = _concat([_make_day("2024-01-02", events={605: (100.0, 85.0, 100.0)})])
  assert _break_mod(_stat(), candles)["2024-01-02"] == 605
  assert _rows(_stat(), candles)[("bucket", "bucket_1")].count == 1


def test_first_break_is_earliest_of_either_side() -> None:
  # Down break at 10:00 (mod 600), up break later at 11:00 (mod 660). The first
  # break is the chronologically earliest, regardless of direction -> mod 600.
  candles = _concat([
    _make_day("2024-01-02", events={
      _IB_END_30: (100.0, 85.0, 100.0),   # down break, 10:00
      660: (115.0, 100.0, 100.0),         # up break, 11:00
    }),
  ])
  assert _break_mod(_stat(), candles)["2024-01-02"] == 600


# ===========================================================================
# Early vs late split around the threshold (12:00 = mod 720)
# ===========================================================================
def test_break_before_threshold_is_early() -> None:
  # Break at 11:55 (mod 715 < 720) -> early; bucket_4 = [690, 720).
  candles = _concat([_make_day("2024-01-02", events={715: (115.0, 100.0, 100.0)})])
  rows = _rows(_stat(), candles)
  assert rows[("timing", "early")].count == 1
  assert rows[("timing", "late")].count == 0
  assert rows[("bucket", "bucket_4")].count == 1


def test_break_at_threshold_is_late() -> None:
  # Break exactly at 12:00 (mod 720) -> late (at/after threshold); bucket_5 = [720, 750).
  candles = _concat([_make_day("2024-01-02", events={_THRESHOLD: (115.0, 100.0, 100.0)})])
  rows = _rows(_stat(), candles)
  assert rows[("timing", "late")].count == 1
  assert rows[("timing", "early")].count == 0
  assert rows[("bucket", "bucket_5")].count == 1


def test_early_late_and_buckets_partition_the_denominator() -> None:
  # Mixed sessions: two early, one late. Both partitions must sum to the countable
  # total (3) and agree with each other.
  candles = _concat([
    _make_day("2024-01-02", events={_IB_END_30: (115.0, 100.0, 100.0)}),  # 10:00 early
    _make_day("2024-01-03", events={715: (115.0, 100.0, 100.0)}),         # 11:55 early
    _make_day("2024-01-04", events={_THRESHOLD: (115.0, 100.0, 100.0)}),  # 12:00 late
  ])
  rows = _rows(_stat(), candles)
  total = rows[("timing", "early")].total
  assert total == 3
  assert rows[("timing", "early")].count == 2
  assert rows[("timing", "late")].count == 1
  bucket_sum = sum(
    r.count for (cond, _), r in rows.items() if cond == "bucket"
  )
  assert bucket_sum == total
  # Early buckets (0..4) sum to the early count; late buckets (5..) to the late count.
  early_bucket_sum = sum(rows[("bucket", f"bucket_{i}")].count for i in range(5))
  assert early_bucket_sum == rows[("timing", "early")].count


# ===========================================================================
# Non-breaking sessions are excluded from the denominator
# ===========================================================================
def test_no_breakout_session_is_excluded() -> None:
  candles = _concat([
    _make_day("2024-01-02", events={_IB_END_30: (115.0, 100.0, 100.0)}),
    _make_day("2024-01-03"),  # no events -> never breaks
  ])
  result = _stat().compute(candles)
  tf = result.instruments["NQ"]["30min"]
  assert tf.total_samples == 2  # both resolved...
  rows = {(r.condition, r.outcome): r for r in tf.results}
  assert rows[("timing", "early")].total == 1  # ...but only the breaking day counts
  # The excluded day carries NaN first_break_mod.
  bm = _break_mod(_stat(), candles)
  assert bm["2024-01-02"] == 600
  assert pd.isna(bm["2024-01-03"])


# ===========================================================================
# Breakout criteria: wick vs close
# ===========================================================================
def test_wick_break_is_not_a_close_break() -> None:
  # 10:00 wicks to 115 (> 110) but CLOSES inside (100). A later bar at 10:30 CLOSES
  # at 112 (> 110). wick sees the first break at 10:00; close sees it at 10:30.
  candles = _concat([
    _make_day("2024-01-02", events={
      _IB_END_30: (115.0, 100.0, 100.0),       # wick break only, 10:00
      _IB_END_30 + 30: (112.0, 100.0, 112.0),  # close break, 10:30
    }),
  ])
  assert _break_mod(_stat(breakout_criteria="wick"), candles)["2024-01-02"] == 600
  assert _break_mod(_stat(breakout_criteria="close"), candles)["2024-01-02"] == 630


def test_close_criteria_with_no_close_break_excludes_session() -> None:
  # Only a wick beyond the level, never a close -> close criteria sees no break.
  candles = _concat([_make_day("2024-01-02", events={_IB_END_30: (115.0, 100.0, 100.0)})])
  rows = _rows(_stat(breakout_criteria="close"), candles)
  assert rows[("timing", "early")].total == 0
  assert pd.isna(_break_mod(_stat(breakout_criteria="close"), candles)["2024-01-02"])


# ===========================================================================
# IB window length per timeframe (30min vs 1h)
# ===========================================================================
def test_ib_window_length_tracks_timeframe() -> None:
  # A 10:15 (mod 615) spike to 115. With a 30-min IB it is in the breakout window
  # and breaks ib_high -> first break at 615 (bucket_1). With a 1h IB (09:30-10:30)
  # the same bar is INSIDE the IB, lifting ib_high to 115, so nothing breaks.
  candles = _concat([_make_day("2024-01-02", events={615: (115.0, 100.0, 100.0)})])
  assert _break_mod(_stat("30min"), candles)["2024-01-02"] == 615
  assert _rows(_stat("30min"), candles)[("bucket", "bucket_1")].count == 1

  assert pd.isna(_break_mod(_stat("1h"), candles)["2024-01-02"])
  assert _rows(_stat("1h"), candles)[("timing", "early")].total == 0


def test_one_hour_ib_buckets_before_ib_end_are_zero() -> None:
  # For a 1h IB the breakout window starts at 10:30; bucket_0 (09:30-10:00) and
  # bucket_1 (10:00-10:30) are unreachable. A 10:30 break lands in bucket_2.
  candles = _concat([_make_day("2024-01-02", events={630: (115.0, 100.0, 100.0)})])
  rows = _rows(_stat("1h"), candles)
  assert rows[("bucket", "bucket_0")].count == 0
  assert rows[("bucket", "bucket_1")].count == 0
  assert rows[("bucket", "bucket_2")].count == 1


# ===========================================================================
# Validation
# ===========================================================================
def test_invalid_breakout_criteria_raises() -> None:
  with pytest.raises(ValueError, match="breakout_criteria"):
    _stat(breakout_criteria="bogus")


def test_invalid_timeframe_raises() -> None:
  with pytest.raises(ValueError, match="Unsupported timeframe"):
    _stat(timeframe="7min")


def test_invalid_threshold_raises() -> None:
  with pytest.raises(ValueError, match="early_threshold_min"):
    InitialBalanceTime("NQ", "30min", _TEST_CONFIG, early_threshold_min=0)


def test_invalid_bin_minutes_raises() -> None:
  with pytest.raises(ValueError, match="bin_minutes"):
    InitialBalanceTime("NQ", "30min", _TEST_CONFIG, bin_minutes=0)


# ===========================================================================
# Baseline (uniform random breakout time)
# ===========================================================================
def test_baseline_is_deterministic_for_fixed_seed() -> None:
  stat = _stat()
  table = stat.build_day_table(_concat([
    _make_day("2024-01-02", events={_IB_END_30: (115.0, 100.0, 100.0)}),
    _make_day("2024-01-03", events={_THRESHOLD: (115.0, 100.0, 100.0)}),
    _make_day("2024-01-04", events={715: (115.0, 100.0, 100.0)}),
  ]))
  a = stat.baseline_rows(table, seed=42)
  b = stat.baseline_rows(table, seed=42)
  assert [(r.condition, r.outcome, r.count) for r in a] == \
    [(r.condition, r.outcome, r.count) for r in b]


def test_baseline_excludes_non_breaking_sessions() -> None:
  # The non-breaking day must stay out of the baseline denominator too.
  candles = _concat([
    _make_day("2024-01-02", events={_IB_END_30: (115.0, 100.0, 100.0)}),
    _make_day("2024-01-03"),  # never breaks
  ])
  stat = _stat()
  table = stat.build_day_table(candles)
  baseline = {(r.condition, r.outcome): r for r in stat.baseline_rows(table, seed=42)}
  assert baseline[("timing", "early")].total == 1


def test_baseline_merged_into_result_rows() -> None:
  candles = _concat([_make_day("2024-01-02", events={_IB_END_30: (115.0, 100.0, 100.0)})])
  rows = _rows(_stat(), candles)
  for r in rows.values():
    assert r.baseline_n == 1
    # Probability channel carries a baseline rate (may be 0.0 for an empty bucket).
    assert r.baseline_prob >= 0.0


# ===========================================================================
# Reproducibility & resolution discipline
# ===========================================================================
def test_compute_is_reproducible() -> None:
  candles = _concat([
    _make_day("2024-01-02", events={_IB_END_30: (115.0, 100.0, 100.0)}),
    _make_day("2024-01-03", events={715: (100.0, 85.0, 100.0)}),
  ])
  a = _stat().compute(candles).model_dump()
  b = _stat().compute(candles).model_dump()
  assert a == b


def test_unresolved_day_is_excluded() -> None:
  candles = _concat([
    _make_day("2024-01-02", events={_IB_END_30: (115.0, 100.0, 100.0)}),
    _make_truncated_day("2024-01-03"),
  ])
  tf = _stat().compute(candles).instruments["NQ"]["30min"]
  assert tf.total_samples == 1
  rows = {(r.condition, r.outcome): r for r in tf.results}
  assert rows[("timing", "early")].total == 1


def test_empty_input_yields_zero_rows() -> None:
  stat = _stat()
  table = stat.build_day_table(_empty_df())
  assert table.empty
  assert "first_break_mod" in table.columns

  tf = stat.compute(_empty_df()).instruments["NQ"]["30min"]
  assert tf.total_samples == 0
  assert tf.data_range == []
  for r in tf.results:
    assert r.count == 0
    assert r.total == 0
    assert r.probability == 0.0


# ===========================================================================
# Slices
# ===========================================================================
def test_declared_slices_are_present() -> None:
  candles = _concat([_make_day("2024-01-02", events={_IB_END_30: (115.0, 100.0, 100.0)})])
  slices = _stat().compute(candles).instruments["NQ"]["30min"].slices
  assert set(slices) == {"weekday", "close", "prev_candle", "overnight", "size", "size_pct"}


def test_close_slice_splits_by_session_colour() -> None:
  # Two green days (close 105 > open 100) and one red (close 95 < open 100), all
  # breaking up at 10:00. session_close stays inside the IB so it does not break.
  candles = _concat([
    _make_day("2024-01-02", events={_IB_END_30: (115.0, 100.0, 100.0)},
              session_open=100.0, session_close=105.0),
    _make_day("2024-01-03", events={_IB_END_30: (115.0, 100.0, 100.0)},
              session_open=100.0, session_close=104.0),
    _make_day("2024-01-04", events={_IB_END_30: (115.0, 100.0, 100.0)},
              session_open=100.0, session_close=95.0),
  ])
  groups = _stat().compute(candles).instruments["NQ"]["30min"].slices["close"].groups
  assert groups["green"].total_samples == 2
  assert groups["red"].total_samples == 1
  green_rows = {(r.condition, r.outcome): r for r in groups["green"].results}
  assert green_rows[("bucket", "bucket_1")].count == 2


# ===========================================================================
# classify_samples
# ===========================================================================
def test_classify_samples_matches_compute_rows() -> None:
  stat = _stat()
  candles = _concat([
    _make_day("2024-01-02", events={_IB_END_30: (115.0, 100.0, 100.0)}),  # 10:00 early
    _make_day("2024-01-03", events={715: (115.0, 100.0, 100.0)}),         # 11:55 early
    _make_day("2024-01-04", events={_THRESHOLD: (115.0, 100.0, 100.0)}),  # 12:00 late
  ])
  table = stat.build_day_table(candles)
  samples = stat.classify_samples(table)
  # Two rows per countable day: one under 'timing', one under 'bucket'. The
  # implementation emits every 'timing' row (date order) before every 'bucket'
  # row (bucket order); ``compute()`` re-sorts by date before writing results.
  assert [(s.date, s.condition, s.outcome) for s in samples] == [
    ("2024-01-02", "timing", "early"),
    ("2024-01-03", "timing", "early"),
    ("2024-01-04", "timing", "late"),
    ("2024-01-02", "bucket", "bucket_1"),
    ("2024-01-03", "bucket", "bucket_4"),
    ("2024-01-04", "bucket", "bucket_5"),
  ]


def test_classify_samples_invariant_matches_compute_rows_counts() -> None:
  stat = _stat()
  candles = _concat([
    _make_day("2024-01-02", events={_IB_END_30: (115.0, 100.0, 100.0)}),
    _make_day("2024-01-03", events={715: (115.0, 100.0, 100.0)}),
    _make_day("2024-01-04", events={_THRESHOLD: (115.0, 100.0, 100.0)}),
  ])
  table = stat.build_day_table(candles)
  samples = stat.classify_samples(table)
  rows = stat.compute_rows(table)
  for r in rows:
    matching = sum(1 for s in samples if s.condition == r.condition and s.outcome == r.outcome)
    assert matching == r.count


def test_classify_samples_empty_table_yields_empty_list() -> None:
  stat = _stat()
  assert stat.classify_samples(_empty_df()) == []
  table = stat.build_day_table(_empty_df())
  assert stat.classify_samples(table) == []


def test_classify_samples_excludes_non_breaking_and_unresolved_days() -> None:
  candles = _concat([
    _make_day("2024-01-02", events={_IB_END_30: (115.0, 100.0, 100.0)}),
    _make_day("2024-01-03"),  # never breaks
    _make_truncated_day("2024-01-04"),
  ])
  stat = _stat()
  table = stat.build_day_table(candles)
  samples = stat.classify_samples(table)
  assert {s.date for s in samples} == {"2024-01-02"}


def test_classify_samples_included_in_compute_result() -> None:
  candles = _concat([_make_day("2024-01-02", events={_IB_END_30: (115.0, 100.0, 100.0)})])
  result = _stat().compute(candles)
  tf = result.instruments["NQ"]["30min"]
  assert len(tf.samples) == 2  # one 'timing' + one 'bucket' row


# ===========================================================================
# JSON round-trip
# ===========================================================================
def test_result_validates_and_writes(tmp_path: Path) -> None:
  candles = _concat([_make_day("2024-01-02", events={_IB_END_30: (115.0, 100.0, 100.0)})])
  result = _stat().compute(candles)
  out = write_results(result, results_dir=tmp_path)
  assert out.exists()
  data = json.loads(out.read_text(encoding="utf-8"))
  assert data["stat_name"] == "initial_balance_time"
  assert data["title"]["en"] == "Initial Balance Breakout — by Time"
  StatRunResult.model_validate(data)


def test_bucket_outcome_labels_are_clock_ranges() -> None:
  # The data-dependent histogram-bucket labels run() merges into labels.outcomes
  # must be human-readable clock ranges so the JSON is self-documenting.
  from stats.initial_balance import time as time_mod

  stat = _stat()
  labels = {k: v.en for k, v in time_mod._bucket_outcome_labels(stat.buckets).items()}
  assert labels["bucket_0"] == "09:30–10:00"
  assert labels["bucket_1"] == "10:00–10:30"
  assert labels["bucket_13"] == "16:00–16:15"
