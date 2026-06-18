"""Tests for stats.engulfing_candles.standard.

All data is synthetic — no real market files required. Expected values are
hand-calculated before each assertion.

Framing:
  - Tier 1 (condition ``engulfing``): how often the RTH daily candle is a
    ``bullish`` or ``bearish`` engulfing of the prior resolved day's body
    (opposite color, current body fully engulfs the prior body, inclusive).
    total = countable days (those with a prior resolved day).
  - Tier 2 (conditions ``bullish`` / ``bearish``): the ``avg_continuation`` and
    ``max_continuation`` percent of the engulfing close, measured forward until
    price reclaims the engulfing open. Magnitude rows carry the metric in
    ``value``; pending (never-invalidated) patterns are excluded.

The first resolved day has no prior day and is excluded from every denominator.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.engulfing_candles.standard import EngulfingCandles

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


def _make_day(
  date: str,
  session_open: float,
  session_close: float,
  day_high: float,
  day_low: float,
) -> pd.DataFrame:
  """One trading day of 1-min RTH bars (09:30–16:14).

  The opening bar carries ``session_open`` as its open; the last bar carries
  ``session_close`` as its close. The day's RTH extremes ``day_high`` /
  ``day_low`` are placed on a neutral mid-session bar so they are independent of
  the open/close prices. All other bars sit between the extremes.
  """
  base = pd.Timestamp(date, tz=_NY)
  mid = (_RTH_START + _RTH_LAST) // 2
  records = []
  for mod in range(_RTH_START, _RTH_LAST + 1):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    o = session_open if mod == _RTH_START else session_close
    c = session_close
    hi = day_high if mod == mid else max(o, c)
    lo = day_low if mod == mid else min(o, c)
    records.append({
      "timestamp": ts,
      "open": o,
      "high": hi,
      "low": lo,
      "close": c,
      "volume": 1000,
    })
  return pd.DataFrame(records)


def make_candles(days: list[dict]) -> pd.DataFrame:
  """Concatenate per-day specs into a single sorted 1-min OHLCV DataFrame."""
  frames = [
    _make_day(d["date"], d["open"], d["close"], d["high"], d["low"]) for d in days
  ]
  df = pd.concat(frames, ignore_index=True)
  return df.sort_values("timestamp").reset_index(drop=True)


def _stat() -> EngulfingCandles:
  return EngulfingCandles(instrument="NQ", config=_TEST_CONFIG)


def _row(result: StatRunResult, condition: str, outcome: str) -> StatResultRow:
  for r in result.instruments["NQ"]["daily"].results:
    if r.condition == condition and r.outcome == outcome:
      return r
  raise KeyError((condition, outcome))


# ===========================================================================
# Core derivation (6 days)
#
#  idx date        open close high  low   color  body         classification
#   0  01-02 Tue   100   95   101   94    red    [95,100]     excluded (no prior)
#   1  01-03 Wed    94  101   101   94    green  [94,101]     BULLISH engulf of 0
#                                                              (prev red; 101>=100, 94<=95)
#   2  01-04 Thu   102  103   110   96    green  [102,103]    neither (prev green)
#   3  01-05 Fri   103  101   108   90    red    [101,103]    BEARISH engulf of 2
#                                                              (prev green; 103>=103, 101<=102)
#   4  01-08 Mon   101  102   104   99    green  [101,102]    neither (top 102 < 103)
#   5  01-09 Tue   100   99   101   98    red    [99,100]     neither (top 100 < 102)
#
# Continuations:
#   day1 bullish: ref=close=101, inval=open=94. Walk forward:
#     day2 low=96 > 94 (no reclaim), high=110 -> run=110
#     day3 low=90 <= 94 (invalidate), high=108 -> run=max(110,108)=110
#     cont = (110-101)/101*100 = 8.910891%   (resolved)
#   day3 bearish: ref=close=101, inval=open=103. Walk forward:
#     day4 high=104 >= 103 (invalidate), low=99 -> run_min=99
#     cont = (101-99)/101*100 = 1.980198%     (resolved)
#
# countable = idx 1..5 (n=5)
#   bullish_n=1 (idx 1), bearish_n=1 (idx 3)
# total_samples = 6 (all resolved days incl. idx 0)
# ===========================================================================

_SEQ = [
  {"date": "2024-01-02", "open": 100.0, "close": 95.0, "high": 101.0, "low": 94.0},
  {"date": "2024-01-03", "open": 94.0, "close": 101.0, "high": 101.0, "low": 94.0},
  {"date": "2024-01-04", "open": 102.0, "close": 103.0, "high": 110.0, "low": 96.0},
  {"date": "2024-01-05", "open": 103.0, "close": 101.0, "high": 108.0, "low": 90.0},
  {"date": "2024-01-08", "open": 101.0, "close": 102.0, "high": 104.0, "low": 99.0},
  {"date": "2024-01-09", "open": 100.0, "close": 99.0, "high": 101.0, "low": 98.0},
]

_BULL_CONT = 9.0 / 101.0 * 100.0    # 8.910891%
_BEAR_CONT = 2.0 / 101.0 * 100.0    # 1.980198%


def test_total_samples_counts_all_resolved_days() -> None:
  """total_samples = 6 resolved sessions (the first one is still resolved)."""
  result = _stat().compute(make_candles(_SEQ))
  assert result.instruments["NQ"]["daily"].total_samples == 6


def test_bullish_frequency() -> None:
  """1 of 5 countable days is a bullish engulfing → P=1/5."""
  row = _row(_stat().compute(make_candles(_SEQ)), "engulfing", "bullish")
  assert row.total == 5
  assert row.count == 1
  assert row.probability == pytest.approx(1 / 5)


def test_bearish_frequency() -> None:
  """1 of 5 countable days is a bearish engulfing → P=1/5."""
  row = _row(_stat().compute(make_candles(_SEQ)), "engulfing", "bearish")
  assert row.total == 5
  assert row.count == 1
  assert row.probability == pytest.approx(1 / 5)


def test_bullish_continuation_value() -> None:
  """Single bullish pattern: avg == max == (110-101)/101."""
  result = _stat().compute(make_candles(_SEQ))
  avg = _row(result, "bullish", "avg_continuation")
  mx = _row(result, "bullish", "max_continuation")
  assert avg.total == 1
  assert avg.value == pytest.approx(_BULL_CONT)
  assert mx.value == pytest.approx(_BULL_CONT)
  assert avg.probability == 0.0  # magnitude row leaves probability at 0


def test_bearish_continuation_value() -> None:
  """Single bearish pattern: avg == max == (101-99)/101."""
  result = _stat().compute(make_candles(_SEQ))
  avg = _row(result, "bearish", "avg_continuation")
  mx = _row(result, "bearish", "max_continuation")
  assert avg.total == 1
  assert avg.value == pytest.approx(_BEAR_CONT)
  assert mx.value == pytest.approx(_BEAR_CONT)


def test_avg_vs_max_continuation() -> None:
  """With two bullish patterns of different continuation, avg < max."""
  # Two reds then two greens that each engulf the prior red, with different MFE.
  #  idx date       open close high low  color  classification
  #   0  01-02 Tue  100   95  101   94   red    excluded
  #   1  01-03 Wed   94  101  101   94   green  BULLISH engulf of 0
  #   2  01-04 Thu  101   96  102   95   red    neither (prev green)
  #   3  01-05 Fri   95  103  103   95   green  BULLISH engulf of 2 (103>=101, 95<=96)
  #   4  01-08 Mon  120  118  121   90   red    neither — but reclaims everything
  #
  # day1 bullish: ref=101, inval=94. fwd: day2 low=95>94 high=102 run=102;
  #   day3 low=95>94 high=103 run=103; day4 low=90<=94 invalidate high=121 run=121.
  #   cont = (121-101)/101*100 = 19.801980%
  # day3 bullish: ref=103, inval=95. fwd: day4 low=90<=95 invalidate high=121 run=121.
  #   cont = (121-103)/103*100 = 17.475728%
  seq = [
    {"date": "2024-01-02", "open": 100.0, "close": 95.0, "high": 101.0, "low": 94.0},
    {"date": "2024-01-03", "open": 94.0, "close": 101.0, "high": 101.0, "low": 94.0},
    {"date": "2024-01-04", "open": 101.0, "close": 96.0, "high": 102.0, "low": 95.0},
    {"date": "2024-01-05", "open": 95.0, "close": 103.0, "high": 103.0, "low": 95.0},
    {"date": "2024-01-08", "open": 120.0, "close": 118.0, "high": 121.0, "low": 90.0},
  ]
  cont1 = 20.0 / 101.0 * 100.0   # 19.801980%
  cont3 = 18.0 / 103.0 * 100.0   # 17.475728%
  result = _stat().compute(make_candles(seq))
  avg = _row(result, "bullish", "avg_continuation")
  mx = _row(result, "bullish", "max_continuation")
  assert avg.total == 2
  assert avg.value == pytest.approx((cont1 + cont3) / 2)
  assert mx.value == pytest.approx(max(cont1, cont3))


def test_pending_pattern_excluded_from_magnitude() -> None:
  """A bullish engulfing as the LAST day never invalidates → pending.

  Its occurrence is still counted (frequency), but it contributes no continuation.
  """
  seq = [
    {"date": "2024-01-02", "open": 100.0, "close": 95.0, "high": 101.0, "low": 94.0},
    {"date": "2024-01-03", "open": 94.0, "close": 101.0, "high": 101.0, "low": 94.0},
  ]
  result = _stat().compute(make_candles(seq))
  freq = _row(result, "engulfing", "bullish")
  assert freq.total == 1
  assert freq.count == 1  # the pattern is observed
  avg = _row(result, "bullish", "avg_continuation")
  assert avg.total == 0      # but excluded from the magnitude denominator
  assert avg.value is None


def test_non_invalidated_but_resolved_uses_full_window() -> None:
  """Continuation includes the invalidation day's favorable extreme."""
  # day1 bullish ref=101 inval=94; the invalidation day (day2) carries the MFE.
  seq = [
    {"date": "2024-01-02", "open": 100.0, "close": 95.0, "high": 101.0, "low": 94.0},
    {"date": "2024-01-03", "open": 94.0, "close": 101.0, "high": 101.0, "low": 94.0},
    {"date": "2024-01-04", "open": 101.0, "close": 100.0, "high": 130.0, "low": 90.0},
  ]
  # day2 high=130 (MFE) AND low=90<=94 (invalidate) on the same day.
  result = _stat().compute(make_candles(seq))
  avg = _row(result, "bullish", "avg_continuation")
  assert avg.total == 1
  assert avg.value == pytest.approx((130.0 - 101.0) / 101.0 * 100.0)


def test_color_requires_opposite_prior() -> None:
  """A green candle engulfing a prior GREEN body is not a bullish engulfing."""
  #  idx date       open close high low  color
  #   0  01-02 Tue   95  100  101   94   green  excluded
  #   1  01-03 Wed   93  102  103   92   green  engulfs [95,100] but prev is GREEN → neither
  seq = [
    {"date": "2024-01-02", "open": 95.0, "close": 100.0, "high": 101.0, "low": 94.0},
    {"date": "2024-01-03", "open": 93.0, "close": 102.0, "high": 103.0, "low": 92.0},
  ]
  result = _stat().compute(make_candles(seq))
  assert _row(result, "engulfing", "bullish").count == 0
  assert _row(result, "engulfing", "bearish").count == 0


def test_partial_engulf_is_not_a_pattern() -> None:
  """A green candle whose body does not fully cover the prior red body fails."""
  #  idx date       open close high low  color  body
  #   0  01-02 Tue  100   95  101   94   red    [95,100]
  #   1  01-03 Wed   96  100  101   95   green  [96,100] — bot 96 > 95, not engulfed
  seq = [
    {"date": "2024-01-02", "open": 100.0, "close": 95.0, "high": 101.0, "low": 94.0},
    {"date": "2024-01-03", "open": 96.0, "close": 100.0, "high": 101.0, "low": 95.0},
  ]
  assert _row(_stat().compute(make_candles(seq)), "engulfing", "bullish").count == 0


def test_weekday_slice_isolates_pattern() -> None:
  """The bullish engulfing (idx 1) falls on a Wednesday; its slice carries it."""
  result = _stat().compute(make_candles(_SEQ))
  weekday = result.instruments["NQ"]["daily"].slices["weekday"]
  wed = weekday.groups["wednesday"]
  rows = {(r.condition, r.outcome): r for r in wed.results}
  assert rows[("engulfing", "bullish")].count == 1
  assert rows[("engulfing", "bullish")].total == 1
  # Continuation is precomputed on the full table, so the slice keeps the value.
  assert rows[("bullish", "avg_continuation")].value == pytest.approx(_BULL_CONT)


def test_baseline_is_present_and_deterministic() -> None:
  """Every row carries a baseline, and the result is reproducible for a seed."""
  candles = make_candles(_SEQ)
  r1 = _stat().compute(candles, seed=42)
  r2 = _stat().compute(candles, seed=42)
  freq1 = _row(r1, "engulfing", "bullish")
  freq2 = _row(r2, "engulfing", "bullish")
  assert freq1.baseline_n == freq2.baseline_n
  assert freq1.baseline_prob == freq2.baseline_prob
  # Magnitude rows expose a value_baseline (may be None if no baseline pattern).
  mag1 = _row(r1, "bullish", "avg_continuation")
  mag2 = _row(r2, "bullish", "avg_continuation")
  assert mag1.value_baseline == mag2.value_baseline


def test_empty_data_yields_zero_rows() -> None:
  """No candles → all six rows present with zero counts and no values."""
  empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
  empty["timestamp"] = pd.to_datetime(empty["timestamp"], utc=True)
  result = _stat().compute(empty)
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 0
  assert tf.data_range == []
  assert _row(result, "engulfing", "bullish").total == 0
  assert _row(result, "bullish", "avg_continuation").value is None


def test_write_results_round_trip(tmp_path: Path) -> None:
  """Result validates and round-trips through write_results to JSON."""
  result = _stat().compute(make_candles(_SEQ))
  path = write_results(result, results_dir=tmp_path)
  with open(path, encoding="utf-8") as f:
    data = json.load(f)
  assert data["stat_name"] == "engulfing_candles"
  loaded = StatRunResult.model_validate(data)
  assert loaded.instruments["NQ"]["daily"].total_samples == 6
