"""Tests for stats.market_session_breakout.standard.

All data is synthetic — no real market files required. Expected values are
hand-calculated before each assertion.

Framing: a single ``session1_range`` condition with four mutually exclusive
outcomes (``broke_high`` / ``broke_low`` / ``broke_both`` / ``neither``) that
partition every countable cycle. The "range" is session 1's high/low; the
breakout window is session 2. Strict inequality defines a break (touching the
level is not a break). The default pair is ``london`` (session 1) → ``ny``
(session 2), both on the same calendar-day cycle.

Each synthetic cycle carries london bars (03:00–09:29, controlling ``s1_high`` /
``s1_low`` and which extreme formed first) plus ny bars (09:30–16:14, controlling
``s2_high`` / ``s2_low`` and the extreme closes). A flag instead builds session 1
as the cross-midnight ``asia`` session (18:00 prior evening → 02:59) to exercise
the cycle attribution.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.market_session_breakout.standard import MarketSessionBreakout

# ---------------------------------------------------------------------------
# Minimal InstrumentConfig (does not depend on NQ.yaml)
# ---------------------------------------------------------------------------
_NY = "America/New_York"
_TEST_CONFIG = InstrumentConfig(
  instrument="NQ",
  working_timezone=_NY,
  sessions={
    "asia": Session(start="18:00", end="03:00"),
    "london": Session(start="03:00", end="09:30"),
    "ny": Session(start="09:30", end="16:15"),
  },
  timeframes=["daily"],
  parquet_path=Path("data/NQ_1min.parquet"),
)


def _bar(ts: pd.Timestamp, o: float, h: float, lo: float, c: float) -> dict:
  return {"timestamp": ts, "open": o, "high": h, "low": lo, "close": c, "volume": 100}


def _london_bars(
  date: str, s1_high: float, s1_low: float, high_first: bool = True, same_bar: bool = False
) -> list[dict]:
  """London session 1 bars (03:00–09:29). Extremes at 04:00 / 05:00.

  ``high_first`` puts the high bar before the low bar (else the reverse).
  ``same_bar`` packs both extremes into one bar (order undetermined → NaN).
  """
  base = pd.Timestamp(date, tz=_NY)
  mid = (s1_high + s1_low) / 2.0
  bars = [_bar(base.replace(hour=3, minute=0), mid, mid, mid, mid)]  # clean open
  if same_bar:
    bars.append(_bar(base.replace(hour=4, minute=0), mid, s1_high, s1_low, mid))
  else:
    # The earlier bar (04:00) holds whichever extreme forms first.
    hi_hour, lo_hour = (4, 5) if high_first else (5, 4)
    bars.append(_bar(base.replace(hour=hi_hour, minute=0), mid, s1_high, mid, mid))
    bars.append(_bar(base.replace(hour=lo_hour, minute=0), mid, mid, s1_low, mid))
  bars.append(_bar(base.replace(hour=9, minute=29), mid, mid, mid, mid))  # coverage
  return bars


def _asia_bars(date: str, s1_high: float, s1_low: float) -> list[dict]:
  """Asia session 1 bars for cycle ``date``: 18:00 prior evening → 02:59.

  Exercises cross-midnight attribution: evening bars (prior day) and early bars
  (cycle day) both belong to ``date``'s cycle.
  """
  base = pd.Timestamp(date, tz=_NY)
  prev = base - pd.Timedelta(days=1)
  mid = (s1_high + s1_low) / 2.0
  return [
    _bar(prev.replace(hour=18, minute=0), mid, mid, mid, mid),  # clean open
    _bar(prev.replace(hour=20, minute=0), mid, s1_high, mid, mid),  # high (evening)
    _bar(base.replace(hour=1, minute=0), mid, mid, s1_low, mid),  # low (early)
    _bar(base.replace(hour=2, minute=59), mid, mid, mid, mid),  # coverage
  ]


def _ny_bars(
  date: str,
  s2_high: float,
  s2_low: float,
  s2_open: float | None = None,
  s2_close: float | None = None,
) -> list[dict]:
  """NY session 2 bars (09:30–16:14). The 09:30 bar carries the wick extremes."""
  base = pd.Timestamp(date, tz=_NY)
  mid = (s2_high + s2_low) / 2.0
  if s2_open is None:
    s2_open = mid
  if s2_close is None:
    s2_close = mid
  return [
    _bar(base.replace(hour=9, minute=30), s2_open, s2_high, s2_low, mid),  # open + wicks
    _bar(base.replace(hour=12, minute=0), mid, mid, mid, mid),  # filler
    _bar(  # coverage bar carries the session close
      base.replace(hour=16, minute=14),
      mid,
      max(mid, s2_close),
      min(mid, s2_close),
      s2_close,
    ),
  ]


def _cycle(
  date: str,
  s1_high: float,
  s1_low: float,
  s2_high: float,
  s2_low: float,
  *,
  high_first: bool = True,
  same_bar: bool = False,
  s2_close: float | None = None,
  asia: bool = False,
) -> list[dict]:
  s1 = (
    _asia_bars(date, s1_high, s1_low)
    if asia
    else _london_bars(date, s1_high, s1_low, high_first=high_first, same_bar=same_bar)
  )
  return s1 + _ny_bars(date, s2_high, s2_low, s2_close=s2_close)


def make_candles(cycles: list[list[dict]]) -> pd.DataFrame:
  records = [r for cycle in cycles for r in cycle]
  df = pd.DataFrame(records)
  return df.sort_values("timestamp").reset_index(drop=True)


def _stat(
  session1: str = "london", session2: str = "ny", breakout_criteria: str = "wick"
) -> MarketSessionBreakout:
  return MarketSessionBreakout(
    instrument="NQ",
    config=_TEST_CONFIG,
    session1=session1,
    session2=session2,
    breakout_criteria=breakout_criteria,
  )


def _row(result: StatRunResult, outcome: str) -> StatResultRow:
  for r in result.instruments["NQ"]["daily"].results:
    if r.condition == "session1_range" and r.outcome == outcome:
      return r
  raise KeyError(outcome)


# ===========================================================================
# Core 4-outcome partition
#
# Session 1 range fixed at [90, 110] (s1_size = 20) for every cycle. Session 2
# extremes select each outcome (strict break):
#   broke_high : s2_high 115 > 110, s2_low 95 >= 90        (up only)
#   broke_low  : s2_high 105 <= 110, s2_low 85 < 90        (down only)
#   broke_both : s2_high 115 > 110, s2_low 85 < 90         (both)
#   neither    : s2_high 108 <= 110, s2_low 92 >= 90       (inside)
# Two of each → every outcome count = 2, total = 8.
# ===========================================================================
_SEQ = [
  _cycle("2024-01-02", 110, 90, 115, 95),  # broke_high
  _cycle("2024-01-03", 110, 90, 105, 85),  # broke_low
  _cycle("2024-01-04", 110, 90, 115, 85),  # broke_both
  _cycle("2024-01-05", 110, 90, 108, 92),  # neither
  _cycle("2024-01-08", 110, 90, 115, 95),  # broke_high
  _cycle("2024-01-09", 110, 90, 105, 85),  # broke_low
  _cycle("2024-01-10", 110, 90, 115, 85),  # broke_both
  _cycle("2024-01-11", 110, 90, 108, 92),  # neither
]


def test_four_outcomes_partition_countable_cycles():
  result = _stat().compute(make_candles(_SEQ))
  counts = {out: _row(result, out).count for out in ("broke_high", "broke_low", "broke_both", "neither")}
  assert counts == {"broke_high": 2, "broke_low": 2, "broke_both": 2, "neither": 2}
  totals = {_row(result, out).total for out in counts}
  assert totals == {8}
  assert sum(counts.values()) == 8
  assert result.instruments["NQ"]["daily"].total_samples == 8


def test_probabilities_sum_to_one():
  result = _stat().compute(make_candles(_SEQ))
  probs = [_row(result, out).probability for out in ("broke_high", "broke_low", "broke_both", "neither")]
  assert sum(probs) == pytest.approx(1.0)


def test_touching_level_is_not_a_break():
  # s2_high == s1_high and s2_low == s1_low exactly → neither (strict inequality).
  result = _stat().compute(make_candles([_cycle("2024-01-02", 110, 90, 110, 90)]))
  assert _row(result, "neither").count == 1
  assert _row(result, "broke_both").count == 0


# ===========================================================================
# Close vs wick criteria
# ===========================================================================
def test_close_criteria_ignores_wick_only_breaks():
  # Wick pierces both sides (115 / 85) but the close stays inside [90, 110].
  cycles = [_cycle("2024-01-02", 110, 90, 115, 85, s2_close=100)]
  wick = _stat(breakout_criteria="wick").compute(make_candles(cycles))
  close = _stat(breakout_criteria="close").compute(make_candles(cycles))
  assert _row(wick, "broke_both").count == 1
  assert _row(close, "neither").count == 1


# ===========================================================================
# Rejection slice (which session 1 extreme formed first)
# ===========================================================================
def test_rejection_slice_splits_by_first_extreme():
  cycles = [
    _cycle("2024-01-02", 110, 90, 115, 95, high_first=True),  # high first
    _cycle("2024-01-03", 110, 90, 105, 85, high_first=False),  # low first
    _cycle("2024-01-04", 110, 90, 115, 85, high_first=True),  # high first
  ]
  result = _stat().compute(make_candles(cycles))
  rej = result.instruments["NQ"]["daily"].slices["rejection"].groups
  assert rej["high_first"].total_samples == 2
  assert rej["low_first"].total_samples == 1


def test_undetermined_first_extreme_excluded_from_rejection():
  # One cycle has both extremes in a single bar → high_first NaN → excluded from
  # the rejection slice, but still counted in the overall result.
  cycles = [
    _cycle("2024-01-02", 110, 90, 115, 95, high_first=True),
    _cycle("2024-01-03", 110, 90, 105, 85, same_bar=True),
  ]
  result = _stat().compute(make_candles(cycles))
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 2
  rej = tf.slices["rejection"].groups
  assert sum(g.total_samples for g in rej.values()) == 1
  # high_first is read only by the rejection slicer; the undetermined cycle still
  # lands in the other slices (weekday does not read high_first).
  assert sum(g.total_samples for g in tf.slices["weekday"].groups.values()) == 2


# ===========================================================================
# Size and levels slices
# ===========================================================================
def test_size_and_levels_slices_present():
  result = _stat().compute(make_candles(_SEQ))
  slices = result.instruments["NQ"]["daily"].slices
  assert set(slices) == {"weekday", "size", "levels", "rejection"}
  # Every countable cycle lands in exactly one size bucket and one weekday group.
  assert sum(g.total_samples for g in slices["weekday"].groups.values()) == 8


def test_levels_extension_uses_break_distance():
  # s1 range [90, 110] (size 20). s2_high 130 → extension 20 = 1.0x of s1_size.
  result = _stat().compute(make_candles([_cycle("2024-01-02", 110, 90, 130, 95)]))
  levels = result.instruments["NQ"]["daily"].slices["levels"].groups
  assert sum(g.total_samples for g in levels.values()) == 1
  # 1.0x falls in the 1.0–1.5x band.
  assert any(key.startswith("1_") for key in levels)


# ===========================================================================
# Cross-midnight session 1 (asia → ny)
# ===========================================================================
def test_asia_session_cross_midnight_attribution():
  cycles = [
    _cycle("2024-01-02", 110, 90, 115, 95, asia=True),  # broke_high
    _cycle("2024-01-03", 110, 90, 108, 92, asia=True),  # neither
  ]
  result = _stat(session1="asia").compute(make_candles(cycles))
  assert result.instruments["NQ"]["daily"].total_samples == 2
  assert _row(result, "broke_high").count == 1
  assert _row(result, "neither").count == 1


# ===========================================================================
# Pending-sample discipline
# ===========================================================================
def test_unresolved_session2_excluded():
  # A cycle whose ny session stops at 09:50 (no end coverage) is dropped.
  base = pd.Timestamp("2024-01-03", tz=_NY)
  truncated = _london_bars("2024-01-03", 110, 90) + [
    _bar(base.replace(hour=9, minute=30), 100, 100, 100, 100),
    _bar(base.replace(hour=9, minute=50), 100, 100, 100, 100),
  ]
  cycles = [_cycle("2024-01-02", 110, 90, 115, 95), truncated]
  result = _stat().compute(make_candles(cycles))
  assert result.instruments["NQ"]["daily"].total_samples == 1


@pytest.mark.parametrize(
  "last_hour,last_minute,expected",
  [
    (16, 0, 1),   # offset 390 == duration(405) - tolerance(15) → resolved
    (15, 59, 0),  # offset 389 < 390 → unresolved, dropped
  ],
)
def test_session2_resolution_boundary(last_hour, last_minute, expected):
  base = pd.Timestamp("2024-01-03", tz=_NY)
  ny = [
    _bar(base.replace(hour=9, minute=30), 100, 115, 95, 100),  # clean open
    _bar(base.replace(hour=last_hour, minute=last_minute), 100, 100, 100, 100),
  ]
  cycles = [_london_bars("2024-01-03", 110, 90) + ny]
  result = _stat().compute(make_candles(cycles))
  assert result.instruments["NQ"]["daily"].total_samples == expected


def test_missing_session1_excluded():
  # A cycle with only ny bars (no london range) is not countable.
  cycles = [
    _cycle("2024-01-02", 110, 90, 115, 95),
    _ny_bars("2024-01-03", 115, 95),
  ]
  result = _stat().compute(make_candles(cycles))
  assert result.instruments["NQ"]["daily"].total_samples == 1


# ===========================================================================
# Baseline
# ===========================================================================
def test_baseline_is_reproducible():
  candles = make_candles(_SEQ)
  a = _stat().compute(candles, seed=42)
  b = _stat().compute(candles, seed=42)
  assert _row(a, "broke_high").baseline_prob == _row(b, "broke_high").baseline_prob


def test_baseline_preserves_symmetric_outcomes():
  # broke_both and neither are unchanged by reflection around the midpoint, so the
  # random baseline reproduces their observed rate exactly (2/8 each here). Only
  # the directional outcomes (broke_high/broke_low) are reshuffled by the flips.
  result = _stat().compute(make_candles(_SEQ), seed=42)
  for outcome in ("broke_both", "neither"):
    row = _row(result, outcome)
    assert row.baseline_prob == pytest.approx(row.probability)
  assert _row(result, "broke_high").baseline_n == 8


# ===========================================================================
# Validation and empty input
# ===========================================================================
def test_rejects_unknown_session():
  with pytest.raises(ValueError, match="Unknown session"):
    _stat(session1="tokyo")


def test_rejects_identical_sessions():
  with pytest.raises(ValueError, match="different"):
    _stat(session1="ny", session2="ny")


def test_empty_input_yields_zero_rows():
  empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
  result = _stat().compute(empty)
  assert result.instruments["NQ"]["daily"].total_samples == 0
  assert _row(result, "neither").total == 0


# ===========================================================================
# Result serialization
# ===========================================================================
def test_result_round_trips_through_write(tmp_path):
  result = _stat().compute(make_candles(_SEQ))
  path = write_results(result, results_dir=tmp_path)
  with open(path, encoding="utf-8") as f:
    data = json.load(f)
  assert data["stat_name"] == "market_session_breakout"
  assert data["title"]["en"] == "Market Session Breakout"
  rows = data["instruments"]["NQ"]["daily"]["results"]
  assert {r["outcome"] for r in rows} == {"broke_high", "broke_low", "broke_both", "neither"}
