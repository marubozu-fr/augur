"""Tests for stats.fair_value_gaps.standard.

All data is synthetic — no real market files required.
Expected values are hand-calculated before each assertion.

FVG geometry (15-min candles built from 1-min bars):
  Bullish: c1.high < c3.low  →  gap_pts = c3.low - c1.high
  Bearish: c1.low  > c3.high →  gap_pts = c1.low - c3.high
  gap_size_pct = 100 * gap_pts / c3.close

Fill (default 100% threshold = full mitigation):
  Bullish: suffix_min_low(after c3) <= c3.low - gap_pts  ≡  <= c1.high
  Bearish: suffix_max_high(after c3) >= c3.high + gap_pts ≡  >= c1.low

Pending discipline:
  FVG where c3 is the last 15-min candle of its session →
  suffix is NaN → excluded from both numerator and denominator.

Bucket index: k = (mod - 570) // 15
  Bucket 0:  mod=570  (09:30)  ← required for resolved open check
  Bucket 26: mod=960  (16:00)  ← satisfies resolved close threshold (>= 975-15=960)
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.fair_value_gaps.standard import FairValueGaps

# ---------------------------------------------------------------------------
# Minimal InstrumentConfig — no NQ.yaml needed
# ---------------------------------------------------------------------------
_NY = "America/New_York"
_RTH_START = 570   # 09:30
_RTH_END = 975     # 16:15

_TEST_CONFIG = InstrumentConfig(
  instrument="NQ",
  working_timezone=_NY,
  sessions={"rth": Session(start="09:30", end="16:15")},
  timeframes=["15min"],
  parquet_path=Path("data/NQ_1min.parquet"),
)


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

def _make_session(
  date: str,
  buckets: list[tuple[int, float, float, float, float]],
) -> pd.DataFrame:
  """Build 1-min RTH bars from explicit per-bucket (k, open, high, low, close) tuples.

  Each tuple places a single bar at mod = 570 + k*15 (the bucket's first minute).
  That single bar fully controls the 15-min aggregate OHLC for that bucket.

  Caller must include bucket 0 (for the resolved open bar at mod=570) and at
  least one bucket with mod >= 960 (for the resolved close threshold).
  """
  base = pd.Timestamp(date, tz=_NY)
  records = []
  for (k, o, h, lo, c) in buckets:
    mod = _RTH_START + k * 15
    hr, mn = divmod(mod, 60)
    ts = base.replace(hour=hr, minute=mn, second=0, microsecond=0)
    records.append({
      "timestamp": ts,
      "open": o,
      "high": h,
      "low": lo,
      "close": c,
      "volume": 1000,
    })
  return pd.DataFrame(records)


def _empty_df() -> pd.DataFrame:
  df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
  df["timestamp"] = pd.array([], dtype="datetime64[ns, America/New_York]")
  return df


def _stat(fill_threshold_pct: float = 100.0) -> FairValueGaps:
  return FairValueGaps(
    instrument="NQ",
    config=_TEST_CONFIG,
    fill_threshold_pct=fill_threshold_pct,
  )


def _row(result: StatRunResult, condition: str, outcome: str) -> StatResultRow:
  for r in result.instruments["NQ"]["15min"].results:
    if r.condition == condition and r.outcome == outcome:
      return r
  raise KeyError((condition, outcome))


# ---------------------------------------------------------------------------
# Session fixtures — OHLC per bucket
# ---------------------------------------------------------------------------

# Bullish FVG at bucket 2 (c3), c1=bucket 0:
#   c1.high=100.0, c3.low=102.0  →  c1.high < c3.low  ✓
#   gap_pts = 102 - 100 = 2.0
#   gap_size_pct = 100 * 2 / 103 (c3.close=103)
#   fill target (100%) = c3.low - gap_pts = 102 - 2 = 100.0 = c1.high
#   FILL session: bucket26.low = 99.0 <= 100.0  →  FILLED
#   NOFILL session: bucket26.low = 101.0 > 100.0 →  NOT FILLED
_BULL_FILL = [
  (0,  100.0, 100.0,  99.0, 100.0),  # c1: high=100
  (1,  101.0, 101.0,  99.5, 101.0),  # c2: neutral
  (2,  103.0, 104.0, 102.0, 103.0),  # c3: low=102 > c1.high=100 → BULLISH FVG
  (26, 100.0, 100.5,  99.0, 100.0),  # fill: low=99 <= 100 → FILLED
]
_BULL_NOFILL = [
  (0,  100.0, 100.0,  99.0, 100.0),
  (1,  101.0, 101.0,  99.5, 101.0),
  (2,  103.0, 104.0, 102.0, 103.0),
  (26, 102.0, 102.5, 101.0, 102.0),  # low=101 > 100 → NOT FILLED
]

# Bearish FVG at bucket 2, c1=bucket 0:
#   c1.low=105.0, c3.high=103.0  →  c1.low > c3.high  ✓
#   gap_pts = 105 - 103 = 2.0
#   gap_size_pct = 100 * 2 / 103 (c3.close=103)
#   fill target (100%) = c3.high + gap_pts = 103 + 2 = 105.0 = c1.low
#   FILL session: bucket26.high = 106.0 >= 105.0  →  FILLED
#   NOFILL session: bucket26.high = 104.5 < 105.0 →  NOT FILLED
_BEAR_FILL = [
  (0,  110.0, 110.0, 105.0, 110.0),  # c1: low=105
  (1,  107.0, 108.0, 106.0, 107.0),  # c2: neutral
  (2,  102.0, 103.0, 101.0, 103.0),  # c3: high=103 < c1.low=105 → BEARISH FVG
  (26, 105.0, 106.0, 104.5, 105.0),  # fill: high=106 >= 105 → FILLED
]
_BEAR_NOFILL = [
  (0,  110.0, 110.0, 105.0, 110.0),
  (1,  107.0, 108.0, 106.0, 107.0),
  (2,  102.0, 103.0, 101.0, 103.0),
  (26, 103.0, 104.5, 102.5, 103.0),  # high=104.5 < 105 → NOT FILLED
]

# Overlapping candles — no gap can form between c1 and c3:
#   Triple (bucket0, bucket1, bucket2):
#     Bullish? c1.high=101 < c3.low=99.5? NO
#     Bearish? c1.low=99  > c3.high=101.5? NO
#   Triple (bucket1, bucket2, bucket26):
#     Bullish? c1.high=102 < c3.low=99.5? NO
#     Bearish? c1.low=100 > c3.high=100.5? NO
_NO_FVG = [
  (0,  100.0, 101.0,  99.0, 100.0),
  (1,  100.5, 102.0, 100.0, 100.5),
  (2,  100.5, 101.5,  99.5, 100.5),
  (26, 100.0, 100.5,  99.5, 100.0),
]

# Pending: bullish FVG at last bucket (bucket 26 = last candle in session)
#   Triple (bucket0, bucket24, bucket25): c1=bucket0, c3=bucket25
#     Bullish? c1.high=100.5 < c3.low=99.5? NO
#     Bearish? c1.low=99.5  > c3.high=100.5? NO
#   Triple (bucket24, bucket25, bucket26): c1=bucket24, c3=bucket26 (LAST CANDLE)
#     Bullish? c1.high=100 < c3.low=102? YES → FVG detected but c3=last → PENDING
#     suffix_min_low for bucket26 = NaN → excluded
_PENDING = [
  (0,  100.0, 100.5,  99.5, 100.0),   # open check
  (24, 100.0, 100.0,  99.0, 100.0),   # c1 for last triple
  (25, 100.0, 100.5,  99.5, 100.0),   # c2 for last triple
  (26, 102.0, 103.0, 102.0, 102.0),   # c3 at last bucket → PENDING
]

_MIXED_DF = pd.concat([
  _make_session("2024-01-02", _BULL_FILL),
  _make_session("2024-01-03", _BEAR_NOFILL),
], ignore_index=True)


# ===========================================================================
# 1. Bullish FVG — FILLS
# ===========================================================================

def test_bullish_fvg_fills_single_event() -> None:
  """Exactly one bullish FVG, fills: direction=bullish, filled=True."""
  dt = _stat().build_day_table(_make_session("2024-01-02", _BULL_FILL))
  assert len(dt) == 1
  assert dt["direction"].iloc[0] == "bullish"
  assert bool(dt["filled"].iloc[0])


def test_bullish_fvg_fills_gap_size_pct() -> None:
  """gap_size_pct = 100 * (c3.low - c1.high) / c3.close = 100*2/103."""
  # c1.high=100, c3.low=102, c3.close=103 → gap_pts=2
  dt = _stat().build_day_table(_make_session("2024-01-02", _BULL_FILL))
  assert dt["gap_size_pct"].iloc[0] == pytest.approx(100.0 * 2.0 / 103.0)


# ===========================================================================
# 2. Bullish FVG — does NOT fill
# ===========================================================================

def test_bullish_fvg_does_not_fill() -> None:
  """Suffix min low above c1.high → filled=False."""
  # suffix_min_low=101.0 > fill_target=100.0 → NOT FILLED
  dt = _stat().build_day_table(_make_session("2024-01-02", _BULL_NOFILL))
  assert len(dt) == 1
  assert dt["direction"].iloc[0] == "bullish"
  assert not bool(dt["filled"].iloc[0])


# ===========================================================================
# 3. Bearish FVG — FILLS
# ===========================================================================

def test_bearish_fvg_fills() -> None:
  """Exactly one bearish FVG, fills: direction=bearish, filled=True."""
  dt = _stat().build_day_table(_make_session("2024-01-02", _BEAR_FILL))
  assert len(dt) == 1
  assert dt["direction"].iloc[0] == "bearish"
  assert bool(dt["filled"].iloc[0])


def test_bearish_fvg_fills_gap_size_pct() -> None:
  """gap_size_pct = 100 * (c1.low - c3.high) / c3.close = 100*2/103."""
  # c1.low=105, c3.high=103, c3.close=103 → gap_pts=2
  dt = _stat().build_day_table(_make_session("2024-01-02", _BEAR_FILL))
  assert dt["gap_size_pct"].iloc[0] == pytest.approx(100.0 * 2.0 / 103.0)


# ===========================================================================
# 4. Bearish FVG — does NOT fill
# ===========================================================================

def test_bearish_fvg_does_not_fill() -> None:
  """Suffix max high below c1.low → filled=False."""
  # suffix_max_high=104.5 < fill_target=105 → NOT FILLED
  dt = _stat().build_day_table(_make_session("2024-01-02", _BEAR_NOFILL))
  assert len(dt) == 1
  assert dt["direction"].iloc[0] == "bearish"
  assert not bool(dt["filled"].iloc[0])


# ===========================================================================
# 5. No FVG — overlapping / continuous candles
# ===========================================================================

def test_no_fvg_empty_event_table() -> None:
  """Overlapping candles → build_day_table returns empty DataFrame with correct columns."""
  dt = _stat().build_day_table(_make_session("2024-01-02", _NO_FVG))
  assert len(dt) == 0
  assert list(dt.columns) == ["direction", "gap_size_pct", "filled"]


def test_no_fvg_compute_totals_zero() -> None:
  """No FVG session → all condition totals are 0 and total_samples=0."""
  result = _stat().compute(_make_session("2024-01-02", _NO_FVG))
  tf = result.instruments["NQ"]["15min"]
  assert tf.total_samples == 0
  for row in tf.results:
    assert row.total == 0
    assert row.count == 0
    assert row.probability == pytest.approx(0.0)


# ===========================================================================
# 6. Pending exclusion: FVG on the last 15-min candle is excluded
# ===========================================================================

def test_pending_fvg_on_last_candle_excluded_from_event_table() -> None:
  """FVG at bucket 26 (last candle) → suffix NaN → excluded; event table empty."""
  # Triple (bucket24, bucket25, bucket26): bucket24.high=100 < bucket26.low=102
  # → Bullish FVG at the last candle; suffix_min_low=NaN → pending → excluded.
  dt = _stat().build_day_table(_make_session("2024-01-02", _PENDING))
  assert len(dt) == 0, f"Expected 0 events (pending excluded), got {len(dt)}"


def test_pending_fvg_not_counted_in_total_samples() -> None:
  """Session with only a pending FVG → total_samples=0 in compute() result."""
  result = _stat().compute(_make_session("2024-01-02", _PENDING))
  assert result.instruments["NQ"]["15min"].total_samples == 0


# ===========================================================================
# 7. Reproducibility / determinism
# ===========================================================================

def test_same_input_gives_identical_output() -> None:
  """compute() is deterministic: two calls on same data produce identical rows."""
  stat = _stat()
  df = _make_session("2024-01-02", _BULL_FILL)
  rows_a = stat.compute(df).instruments["NQ"]["15min"].results
  rows_b = stat.compute(df).instruments["NQ"]["15min"].results
  for a, b in zip(rows_a, rows_b):
    assert (a.condition, a.outcome) == (b.condition, b.outcome)
    assert a.count == b.count
    assert a.total == b.total
    assert a.probability == pytest.approx(b.probability)
    assert a.baseline_prob == pytest.approx(b.baseline_prob)


def test_baseline_deterministic_same_seed() -> None:
  """baseline_rows() with the same seed returns identical probabilities."""
  stat = _stat()
  df = pd.concat([
    _make_session("2024-01-02", _BULL_FILL),
    _make_session("2024-01-03", _BEAR_NOFILL),
  ], ignore_index=True)
  dt = stat.build_day_table(df)
  rows_a = stat.baseline_rows(dt, seed=42)
  rows_b = stat.baseline_rows(dt, seed=42)
  for a, b in zip(rows_a, rows_b):
    assert a.probability == pytest.approx(b.probability)
    assert a.count == b.count


# ===========================================================================
# 8. Baseline populated + outcomes partition
# ===========================================================================

def test_baseline_populated_when_events_exist() -> None:
  """After compute(), rows with total>0 have baseline_n > 0."""
  result = _stat().compute(_make_session("2024-01-02", _BULL_FILL))
  for row in result.instruments["NQ"]["15min"].results:
    if row.total > 0:
      assert row.baseline_n > 0, f"baseline_n=0 for {row.condition}/{row.outcome}"


def test_outcomes_partition_each_condition() -> None:
  """filled.count + not_filled.count == total for bullish and bearish."""
  # 4 events: bull-fill, bull-nofill, bear-fill, bear-nofill → total=2 each direction
  df = pd.concat([
    _make_session("2024-01-02", _BULL_FILL),
    _make_session("2024-01-03", _BULL_NOFILL),
    _make_session("2024-01-04", _BEAR_FILL),
    _make_session("2024-01-05", _BEAR_NOFILL),
  ], ignore_index=True)
  result = _stat().compute(df)
  for cond in ("bullish", "bearish"):
    f = _row(result, cond, "filled")
    nf = _row(result, cond, "not_filled")
    assert f.total == nf.total
    assert f.count + nf.count == f.total


def test_baseline_preserves_pooled_fill_count() -> None:
  """Permutation baseline preserves total filled count across all events."""
  # 4 events: 2 filled (bull-fill + bear-fill) + 2 not-filled
  stat = _stat()
  df = pd.concat([
    _make_session("2024-01-02", _BULL_FILL),
    _make_session("2024-01-03", _BULL_NOFILL),
    _make_session("2024-01-04", _BEAR_FILL),
    _make_session("2024-01-05", _BEAR_NOFILL),
  ], ignore_index=True)
  dt = stat.build_day_table(df)
  assert len(dt) == 4
  real_filled = int(dt["filled"].sum())
  assert real_filled == 2  # one bullish fill + one bearish fill

  for seed in (0, 1, 42, 123):
    bl_rows = stat.baseline_rows(dt, seed=seed)
    bl_map = {(r.condition, r.outcome): r for r in bl_rows}
    bl_total = (
      bl_map[("bullish", "filled")].count
      + bl_map[("bearish", "filled")].count
    )
    assert bl_total == real_filled, (
      f"seed={seed}: baseline filled={bl_total}, expected {real_filled}"
    )


# ===========================================================================
# 9. gap_size_pct correctness (SizeBucket slice exercises the column too)
# ===========================================================================

def test_gap_size_pct_bullish_formula() -> None:
  """Bullish gap_size_pct = 100 * (c3.low - c1.high) / c3.close."""
  # c1.high=100, c3.low=102, c3.close=103 → 100*2/103 ≈ 1.9417
  dt = _stat().build_day_table(_make_session("2024-01-02", _BULL_FILL))
  assert dt["gap_size_pct"].iloc[0] == pytest.approx(100.0 * 2.0 / 103.0)


def test_gap_size_pct_bearish_formula() -> None:
  """Bearish gap_size_pct = 100 * (c1.low - c3.high) / c3.close."""
  # c1.low=105, c3.high=103, c3.close=103 → 100*2/103 ≈ 1.9417
  dt = _stat().build_day_table(_make_session("2024-01-02", _BEAR_FILL))
  assert dt["gap_size_pct"].iloc[0] == pytest.approx(100.0 * 2.0 / 103.0)


def test_size_slice_buckets_present() -> None:
  """The 'size' SizeBucket slice is computed and contains at least one group."""
  result = _stat().compute(_make_session("2024-01-02", _BULL_FILL))
  slices = result.instruments["NQ"]["15min"].slices
  assert "size" in slices
  assert len(slices["size"].groups) >= 1


# ===========================================================================
# 10. Edge cases: empty DF, single session, unresolved session
# ===========================================================================

def test_empty_dataframe_no_crash() -> None:
  """Empty input → total_samples=0, data_range=[], all rows zeroed."""
  result = _stat().compute(_empty_df())
  tf = result.instruments["NQ"]["15min"]
  assert tf.total_samples == 0
  assert tf.data_range == []
  for row in tf.results:
    assert row.count == 0
    assert row.total == 0
    assert row.probability == pytest.approx(0.0)


def test_four_rows_always_emitted() -> None:
  """The 2x2 matrix (all four condition/outcome rows) is emitted even with no data."""
  result = _stat().compute(_empty_df())
  outcomes = {
    (r.condition, r.outcome) for r in result.instruments["NQ"]["15min"].results
  }
  assert outcomes == {
    ("bullish", "filled"),
    ("bullish", "not_filled"),
    ("bearish", "filled"),
    ("bearish", "not_filled"),
  }


def test_unresolved_session_excluded() -> None:
  """Session without bar at mod>=960 is excluded; event table stays empty."""
  # Buckets 0,1,2 → last bar at mod=600 < 960 → NOT resolved
  # Would be a bullish FVG (c1.high=100 < c3.low=102) if resolved
  truncated = _make_session("2024-01-02", [
    (0,  100.0, 100.0,  99.0, 100.0),
    (1,  101.0, 101.0,  99.5, 101.0),
    (2,  103.0, 104.0, 102.0, 103.0),
  ])
  result = _stat().compute(truncated)
  assert result.instruments["NQ"]["15min"].total_samples == 0


# ===========================================================================
# 11. End-to-end compute() with two sessions
#
# Session A (2024-01-02): bullish FVG, fills
# Session B (2024-01-03): bearish FVG, does NOT fill
#
# Expected matrix:
#   bullish/filled:     count=1, total=1, P=1.0
#   bullish/not_filled: count=0, total=1, P=0.0
#   bearish/filled:     count=0, total=1, P=0.0
#   bearish/not_filled: count=1, total=1, P=1.0
#
# total_samples (FVG event rows) = 2
# ===========================================================================

def test_compute_mixed_sessions_total_samples() -> None:
  """Two FVG events across two sessions → total_samples=2."""
  assert _stat().compute(_MIXED_DF).instruments["NQ"]["15min"].total_samples == 2


def test_compute_mixed_bullish_filled() -> None:
  """bullish/filled: 1 of 1 → count=1, P=1.0."""
  r = _row(_stat().compute(_MIXED_DF), "bullish", "filled")
  assert r.count == 1
  assert r.total == 1
  assert r.probability == pytest.approx(1.0)


def test_compute_mixed_bullish_not_filled() -> None:
  """bullish/not_filled: 0 of 1 → count=0, P=0.0."""
  r = _row(_stat().compute(_MIXED_DF), "bullish", "not_filled")
  assert r.count == 0
  assert r.total == 1
  assert r.probability == pytest.approx(0.0)


def test_compute_mixed_bearish_not_filled() -> None:
  """bearish/not_filled: 1 of 1 → count=1, P=1.0."""
  r = _row(_stat().compute(_MIXED_DF), "bearish", "not_filled")
  assert r.count == 1
  assert r.total == 1
  assert r.probability == pytest.approx(1.0)


def test_compute_mixed_data_range() -> None:
  """data_range spans both session dates."""
  tf = _stat().compute(_MIXED_DF).instruments["NQ"]["15min"]
  assert tf.data_range == ["2024-01-02", "2024-01-03"]


# ===========================================================================
# 12. Slices
# ===========================================================================

def test_slices_keys_present() -> None:
  """FairValueGaps declares exactly 'weekday' and 'size' slices."""
  result = _stat().compute(_make_session("2024-01-02", _BULL_FILL))
  slices = result.instruments["NQ"]["15min"].slices
  assert set(slices.keys()) == {"weekday", "size"}


def test_weekday_slice_present() -> None:
  """weekday slice produces at least one group for a session with FVGs."""
  result = _stat().compute(_make_session("2024-01-02", _BULL_FILL))
  groups = result.instruments["NQ"]["15min"].slices["weekday"].groups
  assert len(groups) > 0


# ===========================================================================
# 13. Stat metadata
# ===========================================================================

def test_stat_name() -> None:
  """stat_name must equal 'fair_value_gaps'."""
  assert _stat().compute(_empty_df()).stat_name == "fair_value_gaps"


def test_timeframe_attribute() -> None:
  """timeframe must equal '15min'."""
  assert _stat().timeframe == "15min"


def test_i18n_title_and_definition() -> None:
  """title and definition both have non-empty en and fr strings."""
  result = _stat().compute(_empty_df())
  assert result.title.en != "" and result.title.fr != ""
  assert result.definition.en != "" and result.definition.fr != ""


def test_i18n_labels_correct_keys() -> None:
  """Labels carry bullish/bearish conditions and filled/not_filled outcomes with en+fr."""
  result = _stat().compute(_empty_df())
  assert set(result.labels.conditions) == {"bullish", "bearish"}
  assert set(result.labels.outcomes) == {"filled", "not_filled"}
  for key, i18n in {**result.labels.conditions, **result.labels.outcomes}.items():
    assert i18n.en != "", f"{key}.en is empty"
    assert i18n.fr != "", f"{key}.fr is empty"


# ===========================================================================
# 14. write_results round-trip
# ===========================================================================

def test_write_results_round_trip(tmp_path: Path) -> None:
  """write_results produces fair_value_gaps.json that re-validates via Pydantic."""
  result = _stat().compute(_MIXED_DF)
  written = write_results(result, results_dir=tmp_path)
  assert written.name == "fair_value_gaps.json"
  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)
  tf = validated.instruments["NQ"]["15min"]
  assert tf.total_samples == 2
  bull_filled = next(
    r for r in tf.results if r.condition == "bullish" and r.outcome == "filled"
  )
  assert bull_filled.count == 1


def test_write_results_utf8_literals(tmp_path: Path) -> None:
  """French text is stored as literal UTF-8 characters, not escaped unicode."""
  result = _stat().compute(_empty_df())
  raw = write_results(result, results_dir=tmp_path).read_text(encoding="utf-8")
  assert "é" in raw
  assert "\\u00e9" not in raw
