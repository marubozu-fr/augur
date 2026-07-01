"""Tests for stats.initial_balance.rejection (InitialBalanceRejection).

All data is synthetic — no real market files required. Expected values are
hand-calculated before each assertion.

Framing (2x3 contingency): which IB edge formed FIRST during the IB window
(condition: formed_high vs formed_low) cross-tabulated with which IB edge
broke FIRST in the breakout window (outcome: broke_high / broke_low / neither).

Key definitions:
- IB window: [09:30, 09:30 + ib_minutes). Default 30-min IB ends at 10:00.
- Breakout window: [ib_end, 16:15).
- Formation order: which bar's minute-of-day is smaller — the bar reaching ib_high
  first or the bar reaching ib_low first. Single bar holds both -> NaN -> excluded.
- Break order (wick): first breakout-window bar with high > ib_high (broke_high)
  or low < ib_low (broke_low). Strict inequality; same-minute tie -> excluded.
- Break order (close): same but uses bar close instead of intraday extreme.
- neither: no bar in the breakout window breaks either side.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.initial_balance.rejection import InitialBalanceRejection

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
_RTH_LAST = 974    # 16:14 (last 1-min bar before 16:15)
_IB_END_30 = 600   # 10:00 — end of 30-min IB
_IB_END_1H = 630   # 10:30 — end of 1-h IB
_BASE = 100.0


# ---------------------------------------------------------------------------
# Day builder
# ---------------------------------------------------------------------------

def _make_day(
  date: str,
  *,
  ib_high: float = 110.0,
  ib_low: float = 90.0,
  # Minute-of-day -> (high, low, close) for IB-window bars to override.
  ib_overrides: dict[int, tuple[float, float, float]] | None = None,
  # Minute-of-day -> (high, low, close) for breakout-window bars to override.
  post_events: dict[int, tuple[float, float, float]] | None = None,
  session_open: float = _BASE,
  session_close: float = _BASE,
  last_mod: int = _RTH_LAST,
  ib_minutes: int = 30,
) -> pd.DataFrame:
  """One dense day of 1-min RTH bars with a controllable IB and breakout window.

  Default IB window bars:
  - The first bar (09:30) carries ib_high as wicks high and ib_low as wick low,
    so both extremes are reached simultaneously (formed_high_first = NaN unless
    caller uses ib_overrides to spread them across separate minutes).
  - All subsequent IB bars and all breakout-window bars default to neutral at
    _BASE (strictly inside the range so nothing breaks by default).

  ``ib_overrides`` / ``post_events`` replace specific bars with explicit OHLC.
  """
  ib_overrides = ib_overrides or {}
  post_events = post_events or {}
  base = pd.Timestamp(date, tz=_NY)
  ib_end = _RTH_START + ib_minutes

  records = []
  for mod in range(_RTH_START, last_mod + 1):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)

    if mod < ib_end:
      # IB window
      if mod in ib_overrides:
        ehi, elo, ec = ib_overrides[mod]
        o = max(min(_BASE, ehi), elo)
        hi, lo, c = ehi, elo, ec
      elif mod == _RTH_START:
        # Default: first bar holds both extremes — use ib_high/ib_low as wicks.
        o, hi, lo, c = session_open, ib_high, ib_low, _BASE
      else:
        o = hi = lo = c = _BASE
    else:
      # Breakout window
      if mod in post_events:
        ehi, elo, ec = post_events[mod]
        o = max(min(_BASE, ehi), elo)
        hi, lo, c = ehi, elo, ec
      else:
        o = hi = lo = c = _BASE

    if mod == last_mod:
      c = session_close
      hi = max(hi, c)
      lo = min(lo, c)

    records.append({
      "timestamp": ts,
      "open": o,
      "high": hi,
      "low": lo,
      "close": c,
      "volume": 1000,
    })

  return pd.DataFrame(records)


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


def _concat(days: list[pd.DataFrame]) -> pd.DataFrame:
  df = pd.concat(days, ignore_index=True)
  return df.sort_values("timestamp").reset_index(drop=True)


def _empty_df() -> pd.DataFrame:
  df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
  df["timestamp"] = pd.array([], dtype="datetime64[ns, America/New_York]")
  return df


def _stat(
  timeframe: str = "30min",
  breakout_criteria: str = "wick",
) -> InitialBalanceRejection:
  return InitialBalanceRejection(
    instrument="NQ",
    timeframe=timeframe,
    config=_TEST_CONFIG,
    breakout_criteria=breakout_criteria,
  )


def _rows(
  stat: InitialBalanceRejection,
  candles: pd.DataFrame,
) -> dict[tuple[str, str], StatResultRow]:
  result = stat.compute(candles)
  return {
    (r.condition, r.outcome): r
    for r in result.instruments["NQ"][stat.timeframe].results
  }


# ---------------------------------------------------------------------------
# Helper to build a day where the IB extremes are on SEPARATE bars
# (so formation order is determined).
#
# Strategy:
#   - 09:30 bar: neutral at _BASE (wicks stay inside by default).
#   - 09:31 bar: reaches ib_high (high_first=True) or ib_low (high_first=False).
#   - 09:32 bar: reaches the other extreme.
# ---------------------------------------------------------------------------

def _make_split_ib_day(
  date: str,
  *,
  high_first: bool,
  ib_high: float = 110.0,
  ib_low: float = 90.0,
  post_events: dict[int, tuple[float, float, float]] | None = None,
  session_open: float = _BASE,
  session_close: float = _BASE,
  last_mod: int = _RTH_LAST,
  ib_minutes: int = 30,
) -> pd.DataFrame:
  """Day with a 30-min (or 1h) IB whose extremes are on separate bars.

  The 09:30 bar is neutral. If high_first=True: the 09:31 bar reaches ib_high
  and the 09:32 bar reaches ib_low (so formed_high_first=True). Reversed when
  high_first=False.
  """
  if high_first:
    hi_bar = (_RTH_START + 1, (ib_high, _BASE, _BASE))
    lo_bar = (_RTH_START + 2, (_BASE, ib_low, _BASE))
  else:
    lo_bar = (_RTH_START + 1, (_BASE, ib_low, _BASE))
    hi_bar = (_RTH_START + 2, (ib_high, _BASE, _BASE))

  ib_overrides = {
    _RTH_START: (_BASE, _BASE, _BASE),  # neutral open bar
    hi_bar[0]: hi_bar[1],
    lo_bar[0]: lo_bar[1],
  }
  return _make_day(
    date,
    ib_high=ib_high,
    ib_low=ib_low,
    ib_overrides=ib_overrides,
    post_events=post_events,
    session_open=session_open,
    session_close=session_close,
    last_mod=last_mod,
    ib_minutes=ib_minutes,
  )


# ===========================================================================
# 1. Formation order detection
# ===========================================================================

def test_formation_high_first_detected() -> None:
  # IB high bar at 09:31 (mod 571) comes before IB low bar at 09:32 (mod 572).
  # formed_high_first must be True.
  candles = _concat([_make_split_ib_day("2024-01-02", high_first=True)])
  stat = _stat()
  table = stat.build_day_table(candles)
  assert bool(table.loc[table.index[0], "formed_high_first"]) is True


def test_formation_low_first_detected() -> None:
  # IB low bar at 09:31 (mod 571) precedes IB high bar at 09:32 (mod 572).
  # formed_high_first must be False.
  candles = _concat([_make_split_ib_day("2024-01-02", high_first=False)])
  stat = _stat()
  table = stat.build_day_table(candles)
  assert bool(table.loc[table.index[0], "formed_high_first"]) is False


def test_same_bar_holds_both_extremes_is_excluded() -> None:
  # Default _make_day: the 09:30 bar carries both ib_high and ib_low as wicks.
  # formed_high_first must be NaN -> day excluded from countable rows.
  candles = _concat([_make_day("2024-01-02")])  # default: single bar, both extremes
  stat = _stat()
  table = stat.build_day_table(candles)
  date_key = table.index[0]
  assert pd.isna(table.loc[date_key, "formed_high_first"])
  # Confirm: no countable samples in the result.
  rows = _rows(stat, candles)
  assert rows[("formed_high", "broke_high")].total == 0
  assert rows[("formed_low", "broke_high")].total == 0


# ===========================================================================
# 2. Break order: broke_high, broke_low, neither, same-bar tie excluded
# ===========================================================================

def test_broke_high_first() -> None:
  # Breakout window: 10:01 (mod 601) breaks high (high > 110), 10:02 breaks low.
  # 601 < 602 -> broke_high.
  # formed_high_first=True (high_first split IB).
  # Expected: formed_high -> broke_high = 1, total = 1.
  candles = _concat([
    _make_split_ib_day("2024-01-02", high_first=True, post_events={
      601: (115.0, 100.0, 100.0),  # breaks high at 10:01
      602: (100.0, 85.0, 100.0),   # breaks low at 10:02
    }),
  ])
  rows = _rows(_stat(), candles)
  assert rows[("formed_high", "broke_high")].count == 1
  assert rows[("formed_high", "broke_high")].total == 1
  assert rows[("formed_high", "broke_low")].count == 0
  assert rows[("formed_high", "neither")].count == 0


def test_broke_low_first() -> None:
  # Breakout window: 10:01 breaks low (low < 90), 10:02 breaks high.
  # 601 < 602 -> broke_low.
  # formed_high_first=False (low_first split IB).
  # Expected: formed_low -> broke_low = 1, total = 1.
  candles = _concat([
    _make_split_ib_day("2024-01-02", high_first=False, post_events={
      601: (100.0, 85.0, 100.0),  # breaks low at 10:01
      602: (115.0, 100.0, 100.0), # breaks high at 10:02
    }),
  ])
  rows = _rows(_stat(), candles)
  assert rows[("formed_low", "broke_low")].count == 1
  assert rows[("formed_low", "broke_low")].total == 1
  assert rows[("formed_low", "broke_high")].count == 0
  assert rows[("formed_low", "neither")].count == 0


def test_neither_breaks() -> None:
  # Breakout window: all neutral bars (high=100, low=100 — strictly inside range).
  # Neither side ever breaks -> neither outcome.
  # formed_high_first=True.
  # Expected: formed_high -> neither = 1, total = 1.
  candles = _concat([
    _make_split_ib_day("2024-01-02", high_first=True),  # no post_events -> all neutral
  ])
  rows = _rows(_stat(), candles)
  assert rows[("formed_high", "neither")].count == 1
  assert rows[("formed_high", "neither")].total == 1
  assert rows[("formed_high", "broke_high")].count == 0
  assert rows[("formed_high", "broke_low")].count == 0


def test_same_bar_breaks_both_sides_excluded() -> None:
  # Breakout window: 10:01 bar has high > 110 AND low < 90 simultaneously.
  # Same-minute break on both sides -> tie -> excluded from countable.
  # formed_high_first=True on its IB. Day has a determined formation order
  # but an undetermined break order -> day is excluded.
  candles = _concat([
    _make_split_ib_day("2024-01-02", high_first=True, post_events={
      601: (115.0, 85.0, 100.0),  # breaks both at 10:01
    }),
  ])
  rows = _rows(_stat(), candles)
  # Tie in break order: day must be excluded, total = 0.
  assert rows[("formed_high", "broke_high")].total == 0
  assert rows[("formed_high", "broke_low")].total == 0
  assert rows[("formed_high", "neither")].total == 0


def test_break_touching_level_exactly_is_not_a_break() -> None:
  # high == ib_high (100 is NOT > 110) and low == ib_low (100 is NOT < 90).
  # Strict inequality: touch is not a break -> neither.
  candles = _concat([
    _make_split_ib_day("2024-01-02", high_first=True, post_events={
      601: (110.0, 90.0, 100.0),  # touches exactly but does NOT break
    }),
  ])
  rows = _rows(_stat(), candles)
  assert rows[("formed_high", "neither")].count == 1
  assert rows[("formed_high", "broke_high")].count == 0
  assert rows[("formed_high", "broke_low")].count == 0


# ===========================================================================
# 3. Contingency: three outcomes partition condition total; probabilities sum ~1
# ===========================================================================

def test_contingency_outcomes_sum_to_total() -> None:
  # Four days, all formed_high_first=True:
  #   Day 1 -> broke_high
  #   Day 2 -> broke_low
  #   Day 3 -> neither
  #   Day 4 -> neither
  # formed_high total = 4; counts 1+1+2 = 4.
  # formed_low total = 0 (no formed_low days).
  candles = _concat([
    _make_split_ib_day("2024-01-02", high_first=True, post_events={601: (115.0, 100.0, 100.0)}),
    _make_split_ib_day("2024-01-03", high_first=True, post_events={601: (100.0, 85.0, 100.0)}),
    _make_split_ib_day("2024-01-04", high_first=True),  # neither
    _make_split_ib_day("2024-01-05", high_first=True),  # neither
  ])
  rows = _rows(_stat(), candles)
  total = rows[("formed_high", "broke_high")].total
  assert total == 4
  count_sum = (
    rows[("formed_high", "broke_high")].count
    + rows[("formed_high", "broke_low")].count
    + rows[("formed_high", "neither")].count
  )
  assert count_sum == total
  prob_sum = (
    rows[("formed_high", "broke_high")].probability
    + rows[("formed_high", "broke_low")].probability
    + rows[("formed_high", "neither")].probability
  )
  assert prob_sum == pytest.approx(1.0)


def test_contingency_both_conditions_populated() -> None:
  # Two formed_high days and two formed_low days.
  # formed_high: 1 broke_high, 1 neither.
  # formed_low:  1 broke_low, 1 neither.
  candles = _concat([
    _make_split_ib_day("2024-01-02", high_first=True, post_events={601: (115.0, 100.0, 100.0)}),
    _make_split_ib_day("2024-01-03", high_first=True),  # neither
    _make_split_ib_day("2024-01-04", high_first=False, post_events={601: (100.0, 85.0, 100.0)}),
    _make_split_ib_day("2024-01-05", high_first=False),  # neither
  ])
  rows = _rows(_stat(), candles)

  # formed_high condition: total = 2
  assert rows[("formed_high", "broke_high")].total == 2
  assert rows[("formed_high", "broke_high")].count == 1
  assert rows[("formed_high", "broke_low")].count == 0
  assert rows[("formed_high", "neither")].count == 1

  # formed_low condition: total = 2
  assert rows[("formed_low", "broke_low")].total == 2
  assert rows[("formed_low", "broke_low")].count == 1
  assert rows[("formed_low", "broke_high")].count == 0
  assert rows[("formed_low", "neither")].count == 1

  # Probabilities sum to 1.0 per condition
  ph_sum = sum(rows[("formed_high", o)].probability for o in ("broke_high", "broke_low", "neither"))
  pl_sum = sum(rows[("formed_low", o)].probability for o in ("broke_high", "broke_low", "neither"))
  assert ph_sum == pytest.approx(1.0)
  assert pl_sum == pytest.approx(1.0)


def test_known_probability_values() -> None:
  # formed_high: 3 days -> 2 broke_high, 1 broke_low, 0 neither
  # P(broke_high|formed_high) = 2/3
  # formed_low: 2 days -> 0 broke_high, 1 broke_low, 1 neither
  # P(broke_low|formed_low) = 1/2
  candles = _concat([
    _make_split_ib_day("2024-01-02", high_first=True, post_events={601: (115.0, 100.0, 100.0)}),
    _make_split_ib_day("2024-01-03", high_first=True, post_events={601: (115.0, 100.0, 100.0)}),
    _make_split_ib_day("2024-01-04", high_first=True, post_events={601: (100.0, 85.0, 100.0)}),
    _make_split_ib_day("2024-01-05", high_first=False, post_events={601: (100.0, 85.0, 100.0)}),
    _make_split_ib_day("2024-01-08", high_first=False),  # neither
  ])
  rows = _rows(_stat(), candles)
  # Hand-calculated:
  assert rows[("formed_high", "broke_high")].probability == pytest.approx(2 / 3)
  assert rows[("formed_high", "broke_low")].probability == pytest.approx(1 / 3)
  assert rows[("formed_high", "neither")].probability == pytest.approx(0.0)
  assert rows[("formed_low", "broke_low")].probability == pytest.approx(1 / 2)
  assert rows[("formed_low", "neither")].probability == pytest.approx(1 / 2)
  assert rows[("formed_low", "broke_high")].probability == pytest.approx(0.0)


# ===========================================================================
# 4. Wick vs close criteria
# ===========================================================================

def test_wick_break_not_a_close_break() -> None:
  # 10:01 bar: high = 115 (> 110 ib_high) but close = 100 (inside range).
  # A later bar at 10:05 closes at 112 (> 110).
  # wick: first break at 10:01 (mod 601) -> broke_high.
  # close: first close-break at 10:05 (mod 605) -> broke_high (still broke_high,
  #        but the key is the wick stat sees no close break at 10:01).
  # To confirm they use different bars, verify neither breaks on the CLOSE stat
  # for a day that only has a wick-break (no close > ib_high ever).
  candles = _concat([
    _make_split_ib_day("2024-01-02", high_first=True, post_events={
      601: (115.0, 100.0, 100.0),  # wick break only (close inside)
    }),
  ])
  rows_wick = _rows(_stat(breakout_criteria="wick"), candles)
  rows_close = _rows(_stat(breakout_criteria="close"), candles)
  # wick: broke_high detected
  assert rows_wick[("formed_high", "broke_high")].count == 1
  # close: no close ever exceeded ib_high -> neither
  assert rows_close[("formed_high", "neither")].count == 1
  assert rows_close[("formed_high", "broke_high")].count == 0


def test_close_break_detected_by_close_criteria() -> None:
  # 10:01 bar: close = 112 > 110 (ib_high). Both wick and close detect broke_high.
  candles = _concat([
    _make_split_ib_day("2024-01-02", high_first=True, post_events={
      601: (115.0, 100.0, 112.0),  # close = 112 > 110
    }),
  ])
  rows_wick = _rows(_stat(breakout_criteria="wick"), candles)
  rows_close = _rows(_stat(breakout_criteria="close"), candles)
  assert rows_wick[("formed_high", "broke_high")].count == 1
  assert rows_close[("formed_high", "broke_high")].count == 1


def test_wick_break_below_detected_only_by_wick() -> None:
  # 10:01 bar: low = 85 (< 90 ib_low) but close = 100 (inside range).
  # wick -> broke_low; close -> neither.
  candles = _concat([
    _make_split_ib_day("2024-01-02", high_first=False, post_events={
      601: (100.0, 85.0, 100.0),  # wick below ib_low, close inside
    }),
  ])
  rows_wick = _rows(_stat(breakout_criteria="wick"), candles)
  rows_close = _rows(_stat(breakout_criteria="close"), candles)
  assert rows_wick[("formed_low", "broke_low")].count == 1
  assert rows_close[("formed_low", "neither")].count == 1


def test_criteria_changes_break_order() -> None:
  # Two bars in the breakout window:
  #   10:01: wick high = 115 > 110, close inside -> wick broke_high at 601
  #   10:02: low = 85 < 90, close inside -> wick broke_low at 602 (after 601)
  # wick sees broke_high first. close sees neither (no close crosses either level).
  candles = _concat([
    _make_split_ib_day("2024-01-02", high_first=True, post_events={
      601: (115.0, 100.0, 100.0),  # wick high break only
      602: (100.0, 85.0, 100.0),   # wick low break only
    }),
  ])
  rows_wick = _rows(_stat(breakout_criteria="wick"), candles)
  rows_close = _rows(_stat(breakout_criteria="close"), candles)
  assert rows_wick[("formed_high", "broke_high")].count == 1
  assert rows_close[("formed_high", "neither")].count == 1


# ===========================================================================
# 5. Pending-sample discipline
# ===========================================================================

def test_unresolved_day_excluded() -> None:
  # A truncated day (last bar at 09:50) must not appear in the day table.
  candles = _concat([
    _make_split_ib_day("2024-01-02", high_first=True, post_events={601: (115.0, 100.0, 100.0)}),
    _make_truncated_day("2024-01-03"),
  ])
  stat = _stat()
  tf = stat.compute(candles).instruments["NQ"]["30min"]
  # Only the resolved day survives.
  assert tf.total_samples == 1
  rows = {(r.condition, r.outcome): r for r in tf.results}
  assert rows[("formed_high", "broke_high")].count == 1
  assert rows[("formed_high", "broke_high")].total == 1


def test_day_without_breakout_window_excluded() -> None:
  # A day whose last bar is at ib_end - 1 (09:59) has no breakout window.
  # It must be excluded (inner join on post_any drops it).
  ib_end = _IB_END_30  # 600 for 30-min IB
  candles = _concat([
    _make_split_ib_day(
      "2024-01-02", high_first=True, last_mod=ib_end - 1
    ),
  ])
  stat = _stat()
  table = stat.build_day_table(candles)
  # The day has no breakout-window bars -> must be absent from the day table.
  assert table.empty


def test_day_with_undetermined_formation_excluded_from_contingency() -> None:
  # Default _make_day: single bar holds both extremes -> formed_high_first = NaN.
  # The day is in the day table but excluded from all condition counts.
  candles = _concat([_make_day("2024-01-02")])  # single bar holds both extremes
  stat = _stat()
  table = stat.build_day_table(candles)
  # Day exists in the table (it's resolved and has a breakout window).
  assert len(table) == 1
  assert pd.isna(table.iloc[0]["formed_high_first"])
  # But it contributes to no condition total.
  rows = _rows(stat, candles)
  assert rows[("formed_high", "broke_high")].total == 0
  assert rows[("formed_low", "broke_high")].total == 0


def test_same_bar_tie_excluded_from_contingency() -> None:
  # formed_high_first is determined (True) but break order is tied (same minute).
  # Only the break-order tie excludes the day, not the formation order.
  candles = _concat([
    _make_split_ib_day("2024-01-02", high_first=True, post_events={
      601: (115.0, 85.0, 100.0),  # breaks both sides at the same minute
    }),
  ])
  rows = _rows(_stat(), candles)
  assert rows[("formed_high", "broke_high")].total == 0
  assert rows[("formed_high", "broke_low")].total == 0
  assert rows[("formed_high", "neither")].total == 0


# ===========================================================================
# 6. Reproducibility and baseline determinism
# ===========================================================================

def test_baseline_is_deterministic_for_fixed_seed() -> None:
  # Two calls with the same seed must return identical row lists.
  candles = _concat([
    _make_split_ib_day("2024-01-02", high_first=True, post_events={601: (115.0, 100.0, 100.0)}),
    _make_split_ib_day("2024-01-03", high_first=False, post_events={601: (100.0, 85.0, 100.0)}),
    _make_split_ib_day("2024-01-04", high_first=True),
    _make_split_ib_day("2024-01-05", high_first=False),
  ])
  stat = _stat()
  table = stat.build_day_table(candles)
  a = stat.baseline_rows(table, seed=42)
  b = stat.baseline_rows(table, seed=42)
  assert [(r.condition, r.outcome, r.count) for r in a] == [
    (r.condition, r.outcome, r.count) for r in b
  ]


def test_baseline_different_seed_yields_different_result() -> None:
  # With distinct seeds the permutation differs (almost certainly) on > 1 row.
  # Build enough days that the distribution is non-trivial.
  days = []
  for i, (hf, mod) in enumerate([
    (True, 601), (True, None), (False, 601), (False, None),
    (True, 601), (False, None), (True, None), (False, 601),
  ]):
    date = f"2024-01-{i + 2:02d}"
    pe = {mod: (115.0, 100.0, 100.0)} if mod is not None else {}
    days.append(_make_split_ib_day(date, high_first=hf, post_events=pe))
  stat = _stat()
  table = stat.build_day_table(_concat(days))
  a_counts = [r.count for r in stat.baseline_rows(table, seed=1)]
  b_counts = [r.count for r in stat.baseline_rows(table, seed=9999)]
  # Different seeds almost certainly produce different permutations.
  assert a_counts != b_counts


def test_baseline_preserves_break_outcome_marginal() -> None:
  # Permuting formed_high_first does NOT change break outcomes, so the total
  # number of broke_high, broke_low, and neither days is preserved.
  candles = _concat([
    _make_split_ib_day("2024-01-02", high_first=True, post_events={601: (115.0, 100.0, 100.0)}),
    _make_split_ib_day("2024-01-03", high_first=True, post_events={602: (100.0, 85.0, 100.0)}),
    _make_split_ib_day("2024-01-04", high_first=False),
    _make_split_ib_day("2024-01-05", high_first=False, post_events={601: (100.0, 85.0, 100.0)}),
  ])
  stat = _stat()
  table = stat.build_day_table(candles)

  # Actual marginals from the actual rows
  actual_rows = stat.compute_rows(table)
  actual_bh = sum(r.count for r in actual_rows if r.outcome == "broke_high")
  actual_bl = sum(r.count for r in actual_rows if r.outcome == "broke_low")
  actual_n = sum(r.count for r in actual_rows if r.outcome == "neither")

  # Baseline marginals (permuted formation order, break outcomes unchanged)
  bl_rows = stat.baseline_rows(table, seed=42)
  bl_bh = sum(r.count for r in bl_rows if r.outcome == "broke_high")
  bl_bl = sum(r.count for r in bl_rows if r.outcome == "broke_low")
  bl_n = sum(r.count for r in bl_rows if r.outcome == "neither")

  assert bl_bh == actual_bh
  assert bl_bl == actual_bl
  assert bl_n == actual_n


def test_baseline_merged_into_result_rows() -> None:
  # Every row in compute() output must have baseline_prob >= 0 and baseline_n > 0
  # (when the day table is non-empty and countable).
  candles = _concat([
    _make_split_ib_day("2024-01-02", high_first=True, post_events={601: (115.0, 100.0, 100.0)}),
    _make_split_ib_day("2024-01-03", high_first=False, post_events={602: (100.0, 85.0, 100.0)}),
  ])
  result = _stat().compute(candles)
  for r in result.instruments["NQ"]["30min"].results:
    assert r.baseline_n > 0
    assert r.baseline_prob >= 0.0


def test_compute_is_reproducible() -> None:
  candles = _concat([
    _make_split_ib_day("2024-01-02", high_first=True, post_events={601: (115.0, 100.0, 100.0)}),
    _make_split_ib_day("2024-01-03", high_first=False, post_events={602: (100.0, 85.0, 100.0)}),
  ])
  a = _stat().compute(candles).model_dump()
  b = _stat().compute(candles).model_dump()
  assert a == b


# ===========================================================================
# 7. Both timeframes run and produce the full 6-row result set
# ===========================================================================

def test_30min_timeframe_produces_six_rows() -> None:
  candles = _concat([
    _make_split_ib_day("2024-01-02", high_first=True, post_events={601: (115.0, 100.0, 100.0)}),
    _make_split_ib_day("2024-01-03", high_first=False),
  ])
  result = _stat(timeframe="30min").compute(candles)
  rows = result.instruments["NQ"]["30min"].results
  assert len(rows) == 6
  conditions = {r.condition for r in rows}
  outcomes = {r.outcome for r in rows}
  assert conditions == {"formed_high", "formed_low"}
  assert outcomes == {"broke_high", "broke_low", "neither"}


def test_1h_timeframe_produces_six_rows() -> None:
  # For the 1h IB the window ends at 10:30 (mod 630). Events before 10:30 are
  # inside the IB window and shift ib_high/ib_low; break events must be >= 630.
  candles = _concat([
    _make_split_ib_day("2024-01-02", high_first=True, ib_minutes=60,
                       post_events={631: (115.0, 100.0, 100.0)}),
    _make_split_ib_day("2024-01-03", high_first=False, ib_minutes=60),
  ])
  result = _stat(timeframe="1h").compute(candles)
  rows = result.instruments["NQ"]["1h"].results
  assert len(rows) == 6
  conditions = {r.condition for r in rows}
  outcomes = {r.outcome for r in rows}
  assert conditions == {"formed_high", "formed_low"}
  assert outcomes == {"broke_high", "broke_low", "neither"}


def test_1h_ib_bar_before_ib_end_inside_ib_not_a_break() -> None:
  # A spike at mod 615 (10:15) with high = 115 is inside the 1h IB window (ends at
  # 10:30, mod 630). This lifts ib_high to 115. Formation order: the bar reaching
  # ib_high (115) is at mod 615; the bar reaching ib_low (90) is at mod 572.
  # Since 615 > 572, the LOW formed first -> condition is formed_low.
  # Breakout window (>= 630): a bar at 631 with wick = 114 < 115 -> does NOT break.
  # Expected: formed_low -> neither.
  candles = _concat([
    _make_day(
      "2024-01-02",
      ib_overrides={
        _RTH_START: (_BASE, _BASE, _BASE),       # neutral open — no extremes yet
        _RTH_START + 2: (_BASE, 90.0, _BASE),    # low formed at mod 572
        615: (115.0, _BASE, _BASE),              # high formed at mod 615, inside 1h IB
      },
      post_events={631: (114.0, 100.0, 100.0)},  # wick = 114 < ib_high 115 -> no break
      ib_minutes=60,
    ),
  ])
  stat = _stat(timeframe="1h")
  table = stat.build_day_table(candles)
  # IB high must be 115 (lifted by the 10:15 bar inside the IB window).
  assert table.iloc[0]["ib_high"] == pytest.approx(115.0)
  # Low (90) was reached at mod 572, high (115) at mod 615: low formed first.
  assert bool(table.iloc[0]["formed_high_first"]) is False
  rows = _rows(stat, candles)
  assert rows[("formed_low", "neither")].count == 1
  assert rows[("formed_low", "broke_high")].count == 0


# ===========================================================================
# 8. Empty input -> six zero-count rows
# ===========================================================================

def test_empty_input_yields_six_zero_rows() -> None:
  stat = _stat()
  table = stat.build_day_table(_empty_df())
  assert table.empty

  result = stat.compute(_empty_df())
  tf = result.instruments["NQ"]["30min"]
  assert tf.total_samples == 0
  assert tf.data_range == []
  assert len(tf.results) == 6
  for r in tf.results:
    assert r.count == 0
    assert r.total == 0
    assert r.probability == 0.0


def test_empty_input_row_conditions_and_outcomes_still_correct() -> None:
  # Even with zero data the 6 rows must carry the right (condition, outcome) keys.
  stat = _stat()
  result = stat.compute(_empty_df())
  keys = {(r.condition, r.outcome) for r in result.instruments["NQ"]["30min"].results}
  expected = {
    ("formed_high", "broke_high"),
    ("formed_high", "broke_low"),
    ("formed_high", "neither"),
    ("formed_low", "broke_high"),
    ("formed_low", "broke_low"),
    ("formed_low", "neither"),
  }
  assert keys == expected


# ===========================================================================
# 8b. classify_samples
# ===========================================================================

def test_classify_samples_matches_compute_rows() -> None:
  # Mirrors test_contingency_both_conditions_sum_to_total: 2 formed_high days
  # (broke_high, neither) + 2 formed_low days (broke_low, neither).
  stat = _stat()
  candles = _concat([
    _make_split_ib_day("2024-01-02", high_first=True, post_events={601: (115.0, 100.0, 100.0)}),
    _make_split_ib_day("2024-01-03", high_first=True),  # neither
    _make_split_ib_day("2024-01-04", high_first=False, post_events={601: (100.0, 85.0, 100.0)}),
    _make_split_ib_day("2024-01-05", high_first=False),  # neither
  ])
  table = stat.build_day_table(candles)
  samples = stat.classify_samples(table)
  assert [(s.date, s.condition, s.outcome) for s in samples] == [
    ("2024-01-02", "formed_high", "broke_high"),
    ("2024-01-03", "formed_high", "neither"),
    ("2024-01-04", "formed_low", "broke_low"),
    ("2024-01-05", "formed_low", "neither"),
  ]


def test_classify_samples_invariant_matches_compute_rows_counts() -> None:
  stat = _stat()
  candles = _concat([
    _make_split_ib_day("2024-01-02", high_first=True, post_events={601: (115.0, 100.0, 100.0)}),
    _make_split_ib_day("2024-01-03", high_first=True, post_events={601: (100.0, 85.0, 100.0)}),
    _make_split_ib_day("2024-01-04", high_first=False, post_events={601: (100.0, 85.0, 100.0)}),
    _make_split_ib_day("2024-01-05", high_first=False),
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


def test_classify_samples_excludes_undetermined_and_tied_and_unresolved_days() -> None:
  candles = _concat([
    _make_split_ib_day("2024-01-02", high_first=True, post_events={601: (115.0, 100.0, 100.0)}),
    _make_day("2024-01-03"),  # single bar holds both extremes -> formation undetermined
    _make_split_ib_day("2024-01-04", high_first=True, post_events={
      601: (115.0, 85.0, 100.0),  # breaks both sides at the same minute -> tie
    }),
    _make_truncated_day("2024-01-05"),
  ])
  stat = _stat()
  table = stat.build_day_table(candles)
  samples = stat.classify_samples(table)
  assert {s.date for s in samples} == {"2024-01-02"}


def test_classify_samples_included_in_compute_result() -> None:
  candles = _concat([
    _make_split_ib_day("2024-01-02", high_first=True, post_events={601: (115.0, 100.0, 100.0)}),
    _make_split_ib_day("2024-01-03", high_first=False),
  ])
  result = _stat().compute(candles)
  tf = result.instruments["NQ"]["30min"]
  assert len(tf.samples) == 2


# ===========================================================================
# 9. Pydantic validation + JSON round-trip
# ===========================================================================

def test_result_validates_against_pydantic_model() -> None:
  candles = _concat([
    _make_split_ib_day("2024-01-02", high_first=True, post_events={601: (115.0, 100.0, 100.0)}),
    _make_split_ib_day("2024-01-03", high_first=False),
  ])
  result = _stat().compute(candles)
  # model_validate on the dumped dict must succeed without exception.
  StatRunResult.model_validate(result.model_dump())


def test_result_writes_and_reads_json(tmp_path: Path) -> None:
  candles = _concat([
    _make_split_ib_day("2024-01-02", high_first=True, post_events={601: (115.0, 100.0, 100.0)}),
    _make_split_ib_day("2024-01-03", high_first=False),
  ])
  result = _stat().compute(candles)
  out = write_results(result, results_dir=tmp_path)
  assert out.exists()
  data = json.loads(out.read_text(encoding="utf-8"))
  assert data["stat_name"] == "initial_balance_rejection"
  assert data["title"]["en"] == "Initial Balance Breakout — by Rejection"
  StatRunResult.model_validate(data)


def test_result_data_range_and_total_samples() -> None:
  # Three resolved days, one unresolved: total_samples=3, data_range correct.
  candles = _concat([
    _make_split_ib_day("2024-01-02", high_first=True),
    _make_split_ib_day("2024-01-03", high_first=False),
    _make_split_ib_day("2024-01-04", high_first=True),
    _make_truncated_day("2024-01-05"),
  ])
  tf = _stat().compute(candles).instruments["NQ"]["30min"]
  assert tf.total_samples == 3
  assert tf.data_range == ["2024-01-02", "2024-01-04"]


# ===========================================================================
# 10. Declared slices
# ===========================================================================

def test_declared_slices_are_present() -> None:
  candles = _concat([
    _make_split_ib_day("2024-01-02", high_first=True, post_events={601: (115.0, 100.0, 100.0)}),
  ])
  slices = _stat().compute(candles).instruments["NQ"]["30min"].slices
  assert set(slices) == {"weekday", "close", "prev_candle", "overnight", "size", "size_pct"}


def test_close_slice_splits_by_session_colour() -> None:
  # Two green days (close 105 > open 100) and one red (close 95 < 100), all formed_high
  # and broke_high.
  candles = _concat([
    _make_split_ib_day("2024-01-02", high_first=True,
                       session_open=100.0, session_close=105.0,
                       post_events={601: (115.0, 100.0, 100.0)}),
    _make_split_ib_day("2024-01-03", high_first=True,
                       session_open=100.0, session_close=104.0,
                       post_events={601: (115.0, 100.0, 100.0)}),
    _make_split_ib_day("2024-01-04", high_first=True,
                       session_open=100.0, session_close=95.0,
                       post_events={601: (115.0, 100.0, 100.0)}),
  ])
  groups = _stat().compute(candles).instruments["NQ"]["30min"].slices["close"].groups
  assert groups["green"].total_samples == 2
  assert groups["red"].total_samples == 1


# ===========================================================================
# 11. Invalid arguments raise ValueError
# ===========================================================================

def test_invalid_timeframe_raises() -> None:
  with pytest.raises(ValueError, match="Unsupported timeframe"):
    _stat(timeframe="15min")


def test_invalid_breakout_criteria_raises() -> None:
  with pytest.raises(ValueError, match="breakout_criteria"):
    _stat(breakout_criteria="bogus")
