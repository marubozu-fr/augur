"""Tests for stats.cpi_reaction.standard.

All data is synthetic — no real market files required.
Expected values are hand-calculated before each assertion.

The stat classifies each qualifying CPI release date into one of four cells:
  reaction color (green/red) x RTH day color (green/red).

A date qualifies only when BOTH:
  (a) the RTH session is resolved: has a bar at 09:30 (mod=570) AND last bar
      mod >= 960 (rth_end_min 975 - close_tolerance 15 = 960)
  (b) a 1-min bar exists at exactly 08:30 ET (mod=510) for the reaction open

Definitions:
  reaction_green = reaction_close >= reaction_open   (>= is green)
  day_green      = session_close  >= session_open    (>= is green)

Key minute-of-day constants used throughout:
  _REACT_OPEN_MOD = 510  (08:30 — reaction open bar, in [510, 570))
  _REACT_LAST_MOD = 569  (09:29 — last bar in [510, 570); sets reaction_close)
  _RTH_OPEN_MOD   = 570  (09:30 — RTH open bar)
  _RTH_LAST_MOD   = 974  (16:14 — last RTH bar; 974 >= 960 -> resolves)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.cpi_reaction.standard import CPIReaction

# ---------------------------------------------------------------------------
# Minimal InstrumentConfig — does not depend on NQ.yaml
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

_REACT_OPEN_MOD = 510  # 08:30
_REACT_LAST_MOD = 569  # 09:29 — last bar in the default reaction window [510, 570)
_RTH_OPEN_MOD = 570    # 09:30 — RTH session open
_RTH_LAST_MOD = 974    # 16:14 — last RTH bar; mod 974 >= resolved_min 960


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _bar(
  ts: pd.Timestamp,
  open_: float,
  close: float,
  volume: int = 500,
) -> dict:
  """Build a single OHLCV record with high/low derived from open/close."""
  return {
    "timestamp": ts,
    "open": open_,
    "high": max(open_, close) + 0.25,
    "low": min(open_, close) - 0.25,
    "close": close,
    "volume": volume,
  }


def _make_cpi_day(
  date: str,
  reaction_open: float,
  reaction_close: float,
  session_open: float,
  session_close: float,
) -> pd.DataFrame:
  """One fully-qualifying CPI day: reaction bars (08:30, 09:29) + RTH bars (09:30, 16:14).

  The stat reads:
    reaction_open  = open  of the 08:30 bar     (mod=510)
    reaction_close = close of the 09:29 bar     (mod=569, max mod in [510, 570))
    session_open   = open  of the 09:30 bar     (mod=570)
    session_close  = close of the 16:14 bar     (mod=974, max mod in [570, 975))

  Both the RTH session (mod 570 + last mod 974 >= 960) and the reaction open
  (bar at mod=510) qualify, so the date is counted in every denominator.
  """
  base = pd.Timestamp(date, tz=_NY)
  h_r, m_r = divmod(_REACT_OPEN_MOD, 60)
  h_rl, m_rl = divmod(_REACT_LAST_MOD, 60)
  h_o, m_o = divmod(_RTH_OPEN_MOD, 60)
  h_l, m_l = divmod(_RTH_LAST_MOD, 60)
  records = [
    _bar(base.replace(hour=h_r, minute=m_r, second=0, microsecond=0),
         open_=reaction_open, close=100.0),           # reaction open bar; close neutral
    _bar(base.replace(hour=h_rl, minute=m_rl, second=0, microsecond=0),
         open_=100.0, close=reaction_close),           # last reaction bar; close sets reaction_close
    _bar(base.replace(hour=h_o, minute=m_o, second=0, microsecond=0),
         open_=session_open, close=100.0, volume=1000),  # RTH open bar
    _bar(base.replace(hour=h_l, minute=m_l, second=0, microsecond=0),
         open_=100.0, close=session_close, volume=1000),  # last RTH bar; close sets session_close
  ]
  return pd.DataFrame(records)


def _make_rth_only_day(
  date: str,
  session_open: float = 100.0,
  session_close: float = 110.0,
) -> pd.DataFrame:
  """A resolved RTH day with NO pre-market reaction bars.

  Used for non-CPI trading days to verify they are excluded from denominators.
  """
  base = pd.Timestamp(date, tz=_NY)
  h_o, m_o = divmod(_RTH_OPEN_MOD, 60)
  h_l, m_l = divmod(_RTH_LAST_MOD, 60)
  records = [
    _bar(base.replace(hour=h_o, minute=m_o, second=0, microsecond=0),
         open_=session_open, close=100.0, volume=1000),
    _bar(base.replace(hour=h_l, minute=m_l, second=0, microsecond=0),
         open_=100.0, close=session_close, volume=1000),
  ]
  return pd.DataFrame(records)


def _make_missing_reaction_open_day(date: str) -> pd.DataFrame:
  """CPI day: RTH resolves, but NO bar at 08:30 (mod=510).

  The stat requires a bar at reaction_start_min; without it the reaction is
  undefined and the date is dropped (pending-sample discipline).
  """
  base = pd.Timestamp(date, tz=_NY)
  h_o, m_o = divmod(_RTH_OPEN_MOD, 60)
  h_l, m_l = divmod(_RTH_LAST_MOD, 60)
  records = [
    _bar(base.replace(hour=h_o, minute=m_o, second=0, microsecond=0),
         open_=100.0, close=100.0, volume=1000),
    _bar(base.replace(hour=h_l, minute=m_l, second=0, microsecond=0),
         open_=100.0, close=110.0, volume=1000),
  ]
  return pd.DataFrame(records)


def _make_unresolved_rth_cpi_day(date: str) -> pd.DataFrame:
  """CPI day: has reaction open at 08:30, but RTH is truncated.

  Last RTH bar is at 09:50 (mod=590 < resolved_min=960) so build_resolved_days
  excludes the day; the inner join drops it.
  """
  base = pd.Timestamp(date, tz=_NY)
  h_r, m_r = divmod(_REACT_OPEN_MOD, 60)
  records = [
    _bar(base.replace(hour=h_r, minute=m_r, second=0, microsecond=0),
         open_=100.0, close=105.0),                    # reaction open bar present
    _bar(base.replace(hour=9, minute=30, second=0, microsecond=0),
         open_=100.0, close=100.0, volume=1000),       # RTH open bar
    _bar(base.replace(hour=9, minute=50, second=0, microsecond=0),
         open_=100.0, close=100.0, volume=500),        # mod=590 < 960 -> unresolved
  ]
  return pd.DataFrame(records)


def _make_candles(frames: list[pd.DataFrame]) -> pd.DataFrame:
  """Concatenate per-day DataFrames into a single sorted 1-min candle DataFrame."""
  df = pd.concat(frames, ignore_index=True)
  return df.sort_values("timestamp").reset_index(drop=True)


def _empty_df() -> pd.DataFrame:
  df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
  df["timestamp"] = pd.array([], dtype="datetime64[ns, America/New_York]")
  return df


# ---------------------------------------------------------------------------
# Stat factory helpers
# ---------------------------------------------------------------------------

def _stat(
  event_dates: list[str],
  reaction_start: str = "08:30",
  reaction_end: str = "09:30",
) -> CPIReaction:
  return CPIReaction(
    instrument="NQ",
    config=_TEST_CONFIG,
    event_dates=[pd.Timestamp(d) for d in event_dates],
    reaction_start=reaction_start,
    reaction_end=reaction_end,
  )


def _row(result: StatRunResult, condition: str, outcome: str) -> StatResultRow:
  for r in result.instruments["NQ"]["daily"].results:
    if r.condition == condition and r.outcome == outcome:
      return r
  raise KeyError((condition, outcome))


# ===========================================================================
# 1. Happy path: 2x2 matrix with known counts in all four cells
#
# 8 CPI days — all qualify (reaction open present + RTH resolved):
#
#   Day  Date        reaction_open  reaction_close  reaction_color  session_open  session_close  day_color
#    1   2024-01-02    100            110             green           100            115           green
#    2   2024-01-03    100            110             green           100            115           green
#    3   2024-01-04    100            110             green           100            115           green
#    4   2024-01-05    100            110             green           115            100           red
#    5   2024-01-08    100            110             green           115            100           red
#    6   2024-01-09    110            100             red             100            115           green
#    7   2024-01-10    110            100             red             115            100           red
#    8   2024-01-11    110            100             red             115            100           red
#
# Hand-calculated:
#   reaction_green total = 5 (days 1–5)
#     reaction_green -> green: 3 (days 1,2,3)  P = 3/5 = 0.600
#     reaction_green -> red:   2 (days 4,5)    P = 2/5 = 0.400
#   reaction_red   total = 3 (days 6–8)
#     reaction_red   -> green: 1 (day 6)       P = 1/3 ≈ 0.333
#     reaction_red   -> red:   2 (days 7,8)    P = 2/3 ≈ 0.667
#
#   total_samples = 8
#   data_range = ["2024-01-02", "2024-01-11"]
# ===========================================================================

_CPI_DATES_8 = [
  "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
  "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11",
]


def _build_8day_df() -> pd.DataFrame:
  return _make_candles([
    _make_cpi_day("2024-01-02", 100.0, 110.0, 100.0, 115.0),  # rG, dG
    _make_cpi_day("2024-01-03", 100.0, 110.0, 100.0, 115.0),  # rG, dG
    _make_cpi_day("2024-01-04", 100.0, 110.0, 100.0, 115.0),  # rG, dG
    _make_cpi_day("2024-01-05", 100.0, 110.0, 115.0, 100.0),  # rG, dR
    _make_cpi_day("2024-01-08", 100.0, 110.0, 115.0, 100.0),  # rG, dR
    _make_cpi_day("2024-01-09", 110.0, 100.0, 100.0, 115.0),  # rR, dG
    _make_cpi_day("2024-01-10", 110.0, 100.0, 115.0, 100.0),  # rR, dR
    _make_cpi_day("2024-01-11", 110.0, 100.0, 115.0, 100.0),  # rR, dR
  ])


def test_total_samples() -> None:
  """total_samples == 8: all eight CPI days qualify."""
  result = _stat(_CPI_DATES_8).compute(_build_8day_df())
  assert result.instruments["NQ"]["daily"].total_samples == 8


def test_reaction_green_to_green() -> None:
  # 3 out of 5 reaction-green days closed green: P = 3/5 = 0.6
  result = _stat(_CPI_DATES_8).compute(_build_8day_df())
  r = _row(result, "reaction_green", "green")
  assert r.total == 5
  assert r.count == 3
  assert r.probability == pytest.approx(3 / 5)


def test_reaction_green_to_red() -> None:
  # 2 out of 5 reaction-green days closed red: P = 2/5 = 0.4
  result = _stat(_CPI_DATES_8).compute(_build_8day_df())
  r = _row(result, "reaction_green", "red")
  assert r.total == 5
  assert r.count == 2
  assert r.probability == pytest.approx(2 / 5)


def test_reaction_red_to_green() -> None:
  # 1 out of 3 reaction-red days closed green: P = 1/3
  result = _stat(_CPI_DATES_8).compute(_build_8day_df())
  r = _row(result, "reaction_red", "green")
  assert r.total == 3
  assert r.count == 1
  assert r.probability == pytest.approx(1 / 3)


def test_reaction_red_to_red() -> None:
  # 2 out of 3 reaction-red days closed red: P = 2/3
  result = _stat(_CPI_DATES_8).compute(_build_8day_df())
  r = _row(result, "reaction_red", "red")
  assert r.total == 3
  assert r.count == 2
  assert r.probability == pytest.approx(2 / 3)


def test_condition_totals_sum_to_total_samples() -> None:
  """reaction_green total + reaction_red total == total_samples."""
  result = _stat(_CPI_DATES_8).compute(_build_8day_df())
  rg_total = _row(result, "reaction_green", "green").total
  rr_total = _row(result, "reaction_red", "green").total
  assert rg_total + rr_total == 8


def test_outcomes_partition_each_condition() -> None:
  """green.count + red.count == total for every condition."""
  result = _stat(_CPI_DATES_8).compute(_build_8day_df())
  for cond in ("reaction_green", "reaction_red"):
    g = _row(result, cond, "green")
    r = _row(result, cond, "red")
    assert g.count + r.count == g.total == r.total


def test_data_range() -> None:
  """data_range spans the first to the last qualifying CPI date."""
  result = _stat(_CPI_DATES_8).compute(_build_8day_df())
  assert result.instruments["NQ"]["daily"].data_range == [
    "2024-01-02", "2024-01-11",
  ]


# ===========================================================================
# 2. Flat-reaction and flat-day edge: >= means green
#
# reaction_close == reaction_open  -> reaction_green = True
# session_close  == session_open   -> day_green      = True
# ===========================================================================

def test_flat_reaction_classified_green() -> None:
  """reaction_close == reaction_open -> reaction_green (>= is green)."""
  # reaction_open=100, reaction_close=100 -> green reaction
  # session_open=100, session_close=110   -> green day
  df = _make_candles([_make_cpi_day("2024-03-12", 100.0, 100.0, 100.0, 110.0)])
  result = _stat(["2024-03-12"]).compute(df)
  # 1 reaction-green day that closed green
  rg_green = _row(result, "reaction_green", "green")
  assert rg_green.total == 1
  assert rg_green.count == 1
  assert rg_green.probability == pytest.approx(1.0)
  # reaction_red is empty
  rr_green = _row(result, "reaction_red", "green")
  assert rr_green.total == 0
  assert rr_green.probability == pytest.approx(0.0)


def test_flat_day_classified_green() -> None:
  """session_close == session_open -> day_green (>= is green)."""
  # reaction_open=110, reaction_close=100 -> red reaction (110 > 100, so 100 < 110 -> red)
  # session_open=100, session_close=100   -> green day (flat)
  df = _make_candles([_make_cpi_day("2024-03-12", 110.0, 100.0, 100.0, 100.0)])
  result = _stat(["2024-03-12"]).compute(df)
  # 1 reaction-red day that closed green (flat)
  rr_green = _row(result, "reaction_red", "green")
  assert rr_green.total == 1
  assert rr_green.count == 1
  assert rr_green.probability == pytest.approx(1.0)


# ===========================================================================
# 3. Window boundary: 09:30 bar (mod=570) must NOT be part of the reaction
#
# The reaction window is [reaction_start_min, reaction_end_min) = [510, 570).
# A bar at exactly mod=570 (09:30) lies outside the reaction window.
#
# Setup:
#   08:30 bar: open=100  (reaction_open=100)
#   09:29 bar: close=90  (would make reaction RED since 90 < 100)
#   09:30 bar: open=105, close=150  (RTH open; if included in reaction,
#              close=150 > 100 -> GREEN — discriminates the two interpretations)
#   16:14 bar: close=110 (session_close)
#
# Expected: reaction is RED (reaction_close=90 from 09:29, not 150 from 09:30).
# ===========================================================================

def _make_window_boundary_day(date: str) -> pd.DataFrame:
  """Day with 08:30 (open=100), 09:29 (close=90), 09:30 (close=150), 16:14 (close=110)."""
  base = pd.Timestamp(date, tz=_NY)
  records = [
    _bar(base.replace(hour=8, minute=30, second=0, microsecond=0),
         open_=100.0, close=100.0),           # reaction open bar
    _bar(base.replace(hour=9, minute=29, second=0, microsecond=0),
         open_=100.0, close=90.0),            # last reaction bar: close=90 -> red
    _bar(base.replace(hour=9, minute=30, second=0, microsecond=0),
         open_=105.0, close=150.0, volume=1000),  # RTH open; close=150 excluded from reaction
    _bar(base.replace(hour=16, minute=14, second=0, microsecond=0),
         open_=100.0, close=110.0, volume=1000),   # last RTH bar
  ]
  return pd.DataFrame(records)


def test_reaction_end_bar_excluded_from_window() -> None:
  """Bar at reaction_end_min (09:30, mod=570) is outside [510, 570) and not used.

  If 09:30 were included, reaction_close=150 and reaction would be GREEN.
  Since 09:30 is correctly excluded, reaction_close=90 (from 09:29) and
  reaction is RED.
  """
  df = _make_candles([_make_window_boundary_day("2024-03-12")])
  stat = _stat(["2024-03-12"])
  table = stat.build_day_table(df)
  assert len(table) == 1
  # reaction_green must be False (red reaction): close=90 < open=100
  assert bool(table["reaction_green"].iloc[0]) is False


# ===========================================================================
# 4. Non-CPI days excluded: total_samples counts only event_dates
#
# 3 qualifying CPI days + 5 non-CPI trading days.
# total_samples must equal 3 (only CPI dates count in any denominator).
# ===========================================================================

_NON_CPI_DATES = [
  "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09",
]
_CPI_DATES_3 = ["2024-01-02", "2024-01-10", "2024-01-11"]


def _build_mixed_df() -> pd.DataFrame:
  frames = [
    _make_cpi_day("2024-01-02", 100.0, 110.0, 100.0, 115.0),   # CPI, rG, dG
    _make_rth_only_day("2024-01-03"),                           # non-CPI
    _make_rth_only_day("2024-01-04"),                           # non-CPI
    _make_rth_only_day("2024-01-05"),                           # non-CPI
    _make_rth_only_day("2024-01-08"),                           # non-CPI
    _make_rth_only_day("2024-01-09"),                           # non-CPI
    _make_cpi_day("2024-01-10", 100.0, 110.0, 115.0, 100.0),   # CPI, rG, dR
    _make_cpi_day("2024-01-11", 110.0, 100.0, 100.0, 115.0),   # CPI, rR, dG
  ]
  return _make_candles(frames)


def test_non_cpi_days_excluded_from_total() -> None:
  """5 non-CPI days present in data but not in event_dates; total_samples == 3."""
  result = _stat(_CPI_DATES_3).compute(_build_mixed_df())
  assert result.instruments["NQ"]["daily"].total_samples == 3


def test_non_cpi_days_do_not_inflate_condition_totals() -> None:
  """Condition totals sum to 3 (number of CPI dates), not 8 (all trading days)."""
  result = _stat(_CPI_DATES_3).compute(_build_mixed_df())
  rg_total = _row(result, "reaction_green", "green").total
  rr_total = _row(result, "reaction_red", "green").total
  # 2 reaction-green (Jan 2, Jan 10) + 1 reaction-red (Jan 11) = 3
  assert rg_total + rr_total == 3
  assert rg_total == 2
  assert rr_total == 1


# ===========================================================================
# 5. Pending discipline — missing reaction open
#
# A CPI date with a resolved RTH session but NO bar at 08:30 must be dropped.
# The reaction is undefined; the date contributes 0 to every denominator.
# ===========================================================================

def test_missing_reaction_open_dropped_from_day_table() -> None:
  """CPI date with no 08:30 bar is absent from build_day_table output."""
  date = "2024-03-12"
  df = _make_candles([_make_missing_reaction_open_day(date)])
  stat = _stat([date])
  table = stat.build_day_table(df)
  # Day must be excluded; day_table should be empty
  assert table.empty


def test_missing_reaction_open_not_in_total_samples() -> None:
  """CPI date without reaction open does not increase total_samples."""
  # 1 good CPI day + 1 CPI date missing reaction open
  good_date = "2024-03-11"
  bad_date = "2024-03-12"
  df = _make_candles([
    _make_cpi_day(good_date, 100.0, 110.0, 100.0, 115.0),
    _make_missing_reaction_open_day(bad_date),
  ])
  stat = _stat([good_date, bad_date])
  result = stat.compute(df)
  # Only the good day qualifies
  assert result.instruments["NQ"]["daily"].total_samples == 1


# ===========================================================================
# 6. Pending discipline — unresolved RTH session
#
# A CPI date with a reaction open bar but a truncated RTH session (last bar
# mod=590 < resolved_min=960) is excluded by build_resolved_days; the inner
# join then drops it from the qualifying set.
# ===========================================================================

def test_unresolved_rth_dropped_from_day_table() -> None:
  """CPI date with truncated RTH is absent from build_day_table output."""
  date = "2024-03-12"
  df = _make_candles([_make_unresolved_rth_cpi_day(date)])
  stat = _stat([date])
  table = stat.build_day_table(df)
  assert table.empty


def test_unresolved_rth_not_in_total_samples() -> None:
  """CPI date with unresolved RTH does not increase total_samples."""
  good_date = "2024-03-11"
  bad_date = "2024-03-12"
  df = _make_candles([
    _make_cpi_day(good_date, 100.0, 110.0, 100.0, 115.0),
    _make_unresolved_rth_cpi_day(bad_date),
  ])
  stat = _stat([good_date, bad_date])
  result = stat.compute(df)
  assert result.instruments["NQ"]["daily"].total_samples == 1


# ===========================================================================
# 7. CPI date outside data / on a non-trading day
#
# event_dates contains dates for which no candles exist in the DataFrame.
# build_day_table must return an empty table without crashing.
# ===========================================================================

def test_event_date_with_no_candles_dropped() -> None:
  """CPI date with no candles at all is silently excluded — no crash."""
  # event_dates has 2024-01-07 (Sunday), but no candles exist for that date
  stat = _stat(["2024-01-07"])
  df = _make_candles([_make_cpi_day("2024-01-02", 100.0, 110.0, 100.0, 115.0)])
  table = stat.build_day_table(df)
  assert table.empty


def test_event_dates_entirely_outside_data_range() -> None:
  """All event dates have no candles; compute() returns all-zero rows, no crash."""
  stat = _stat(["2023-01-01", "2023-01-02"])
  df = _make_candles([_make_cpi_day("2024-01-02", 100.0, 110.0, 100.0, 115.0)])
  result = stat.compute(df)
  assert result.instruments["NQ"]["daily"].total_samples == 0
  for r in result.instruments["NQ"]["daily"].results:
    assert r.count == 0
    assert r.total == 0


# ===========================================================================
# 8. Baseline reproducibility
#
# baseline_rows(table, seed=X) must be identical across two calls with the same
# seed. Different seeds may differ (verified probabilistically on 8-row table).
# ===========================================================================

def test_baseline_same_seed_deterministic() -> None:
  """baseline_rows(seed=42) is byte-for-byte identical on two calls."""
  stat = _stat(_CPI_DATES_8)
  df = _build_8day_df()
  table = stat.build_day_table(df)
  rows_a = stat.baseline_rows(table, seed=42)
  rows_b = stat.baseline_rows(table, seed=42)
  assert len(rows_a) == len(rows_b) == 4
  for a, b in zip(rows_a, rows_b):
    assert (a.condition, a.outcome) == (b.condition, b.outcome)
    assert a.count == b.count
    assert a.total == b.total
    assert a.probability == pytest.approx(b.probability)


def test_baseline_different_seeds_may_differ() -> None:
  """seed=42 and seed=99 produce at least one row with a different probability."""
  stat = _stat(_CPI_DATES_8)
  table = stat.build_day_table(_build_8day_df())
  rows_42 = stat.baseline_rows(table, seed=42)
  rows_99 = stat.baseline_rows(table, seed=99)
  any_diff = any(
    abs(a.probability - b.probability) > 1e-9
    for a, b in zip(rows_42, rows_99)
  )
  assert any_diff, "seed=42 and seed=99 produced identical baseline probabilities"


def test_baseline_preserves_condition_totals() -> None:
  """Randomising day_green does not change the condition totals.

  reaction_green is held fixed; only day_green is randomised. Therefore the
  number of days in each condition (reaction_green / reaction_red) is unchanged.
  """
  stat = _stat(_CPI_DATES_8)
  table = stat.build_day_table(_build_8day_df())
  real_rows = stat.compute_rows(table)
  bl_rows = stat.baseline_rows(table, seed=42)
  # Real rows and baseline rows must have the same totals
  for real, bl in zip(real_rows, bl_rows):
    assert (real.condition, real.outcome) == (bl.condition, bl.outcome)
    assert real.total == bl.total


def test_baseline_embedded_in_compute_sets_baseline_n() -> None:
  """After compute(), every result row carries a positive baseline_n (N > 0)."""
  result = _stat(_CPI_DATES_8).compute(_build_8day_df())
  for row in result.instruments["NQ"]["daily"].results:
    assert row.baseline_n > 0, f"baseline_n=0 for {row.condition} -> {row.outcome}"


# ===========================================================================
# 9. Empty cases: empty event_dates and empty candles DataFrame
# ===========================================================================

def test_empty_event_dates_build_day_table_returns_empty() -> None:
  """event_dates={} -> build_day_table returns an empty DataFrame."""
  stat = _stat([])
  table = stat.build_day_table(_build_8day_df())
  assert table.empty


def test_empty_event_dates_four_zero_rows() -> None:
  """event_dates={} -> compute() returns 4 rows all zeroed, no crash."""
  result = _stat([]).compute(_build_8day_df())
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 0
  assert tf.data_range == []
  for r in tf.results:
    assert r.count == 0
    assert r.total == 0
    assert r.probability == pytest.approx(0.0)


def test_empty_candles_build_day_table_returns_empty() -> None:
  """Empty candles DataFrame -> build_day_table returns empty, no crash."""
  stat = _stat(["2024-01-02"])
  table = stat.build_day_table(_empty_df())
  assert table.empty


def test_empty_candles_four_zero_rows() -> None:
  """Empty candles -> compute() returns 4 zeroed rows, no crash."""
  result = _stat(["2024-01-02"]).compute(_empty_df())
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 0
  assert tf.data_range == []
  for r in tf.results:
    assert r.count == 0
    assert r.total == 0
    assert r.probability == pytest.approx(0.0)


def test_both_empty_no_crash() -> None:
  """No event_dates and empty candles -> no crash, all-zero rows."""
  result = _stat([]).compute(_empty_df())
  assert result.instruments["NQ"]["daily"].total_samples == 0
  assert len(result.instruments["NQ"]["daily"].results) == 4


# ===========================================================================
# 10. Custom reaction window
#
# reaction_start="08:30", reaction_end="09:00" -> window [510, 540).
# The last bar inside the window is 08:59 (mod=539); a bar at 09:00 (mod=540)
# must NOT be included.
#
# Day layout:
#   08:30 (mod=510): open=100                      -> reaction_open=100
#   08:59 (mod=539): close=90                      -> reaction_close=90 with custom window
#   09:00 (mod=540): close=150                     -> reaction_close=150 with DEFAULT window
#   09:30 (mod=570): RTH open bar
#   16:14 (mod=974): RTH close bar
#
# Custom window [510, 540): max mod=539 -> reaction_close=90 < 100 -> RED
# Default window [510, 570): max mod=540 -> reaction_close=150 > 100 -> GREEN
# ===========================================================================

def _make_custom_window_day(date: str) -> pd.DataFrame:
  base = pd.Timestamp(date, tz=_NY)
  records = [
    _bar(base.replace(hour=8, minute=30, second=0, microsecond=0),
         open_=100.0, close=100.0),           # mod=510 — reaction open
    _bar(base.replace(hour=8, minute=59, second=0, microsecond=0),
         open_=100.0, close=90.0),            # mod=539 — last bar in [510, 540)
    _bar(base.replace(hour=9, minute=0, second=0, microsecond=0),
         open_=100.0, close=150.0),           # mod=540 — excluded in custom window
    _bar(base.replace(hour=9, minute=30, second=0, microsecond=0),
         open_=100.0, close=100.0, volume=1000),  # RTH open
    _bar(base.replace(hour=16, minute=14, second=0, microsecond=0),
         open_=100.0, close=110.0, volume=1000),  # RTH close (mod=974 >= 960)
  ]
  return pd.DataFrame(records)


def test_custom_window_reaction_close_from_last_bar_in_window() -> None:
  """Custom window 08:30-09:00 uses 08:59 bar as reaction_close, not 09:00."""
  date = "2024-03-12"
  df = _make_candles([_make_custom_window_day(date)])

  stat_custom = _stat([date], reaction_start="08:30", reaction_end="09:00")
  table_custom = stat_custom.build_day_table(df)
  assert len(table_custom) == 1
  # With [510,540): max mod=539, reaction_close=90 < 100=reaction_open -> red
  assert bool(table_custom["reaction_green"].iloc[0]) is False


def test_custom_window_vs_default_window_differ() -> None:
  """The two window configs classify the same day differently.

  Default [510,570): max mod=540, reaction_close=150 > 100 -> green.
  Custom  [510,540): max mod=539, reaction_close=90  < 100 -> red.
  """
  date = "2024-03-12"
  df = _make_candles([_make_custom_window_day(date)])

  stat_default = _stat([date])  # reaction_end="09:30" -> [510, 570)
  stat_custom = _stat([date], reaction_start="08:30", reaction_end="09:00")

  table_default = stat_default.build_day_table(df)
  table_custom = stat_custom.build_day_table(df)

  assert len(table_default) == 1
  assert len(table_custom) == 1
  # Default: green; custom: red
  assert bool(table_default["reaction_green"].iloc[0]) is True
  assert bool(table_custom["reaction_green"].iloc[0]) is False


def test_reaction_end_equal_to_start_raises_value_error() -> None:
  """reaction_end == reaction_start must raise ValueError."""
  with pytest.raises(ValueError):
    _stat([], reaction_start="08:30", reaction_end="08:30")


def test_reaction_end_before_start_raises_value_error() -> None:
  """reaction_end < reaction_start must raise ValueError."""
  with pytest.raises(ValueError):
    _stat([], reaction_start="09:30", reaction_end="08:30")


# ===========================================================================
# 11. Result shape invariants (via stat.compute())
# ===========================================================================

def test_exactly_four_result_rows() -> None:
  """compute() always returns exactly 4 result rows (the 2x2 matrix)."""
  result = _stat(_CPI_DATES_8).compute(_build_8day_df())
  rows = result.instruments["NQ"]["daily"].results
  assert len(rows) == 4


def test_result_condition_outcome_keys() -> None:
  """The (condition, outcome) pairs match the exact 2x2 specification."""
  result = _stat(_CPI_DATES_8).compute(_build_8day_df())
  rows = result.instruments["NQ"]["daily"].results
  pairs = {(r.condition, r.outcome) for r in rows}
  assert pairs == {
    ("reaction_green", "green"),
    ("reaction_green", "red"),
    ("reaction_red", "green"),
    ("reaction_red", "red"),
  }


def test_result_timeframe_is_daily() -> None:
  """Results are stored under the 'daily' timeframe key."""
  result = _stat(_CPI_DATES_8).compute(_build_8day_df())
  assert "daily" in result.instruments["NQ"]


def test_result_slices_empty() -> None:
  """CPIReaction declares no slices; slices dict must be empty."""
  result = _stat(_CPI_DATES_8).compute(_build_8day_df())
  assert result.instruments["NQ"]["daily"].slices == {}


def test_result_slices_empty_when_no_data() -> None:
  """slices is {} even with empty candles (no crash on slice computation)."""
  result = _stat(["2024-01-02"]).compute(_empty_df())
  assert result.instruments["NQ"]["daily"].slices == {}


def test_total_samples_equals_len_day_table() -> None:
  """total_samples == len(build_day_table(df))."""
  stat = _stat(_CPI_DATES_8)
  df = _build_8day_df()
  result = stat.compute(df)
  table = stat.build_day_table(df)
  assert result.instruments["NQ"]["daily"].total_samples == len(table)


# ===========================================================================
# 12. stat_name, i18n, and write_results round-trip
# ===========================================================================

def test_stat_name() -> None:
  result = _stat([]).compute(_empty_df())
  assert result.stat_name == "cpi_reaction"


def test_i18n_title_non_empty() -> None:
  result = _stat([]).compute(_empty_df())
  assert result.title.en != ""
  assert result.title.fr != ""


def test_i18n_definition_non_empty() -> None:
  result = _stat([]).compute(_empty_df())
  assert result.definition.en != ""
  assert result.definition.fr != ""


def test_i18n_labels_conditions() -> None:
  """condition labels carry both en and fr for every key."""
  result = _stat([]).compute(_empty_df())
  assert set(result.labels.conditions) == {"reaction_green", "reaction_red"}
  for key, label in result.labels.conditions.items():
    assert label.en != "", f"{key}.en is empty"
    assert label.fr != "", f"{key}.fr is empty"


def test_i18n_labels_outcomes() -> None:
  """outcome labels carry both en and fr for every key."""
  result = _stat([]).compute(_empty_df())
  assert set(result.labels.outcomes) == {"green", "red"}
  for key, label in result.labels.outcomes.items():
    assert label.en != "", f"{key}.en is empty"
    assert label.fr != "", f"{key}.fr is empty"


def test_write_results_round_trip(tmp_path: Path) -> None:
  """write_results produces cpi_reaction.json that re-validates correctly."""
  stat = _stat(_CPI_DATES_8)
  result = stat.compute(_build_8day_df())
  written = write_results(result, results_dir=tmp_path)
  assert written.name == "cpi_reaction.json"

  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)
  assert validated.stat_name == "cpi_reaction"
  assert "NQ" in validated.instruments
  tf = validated.instruments["NQ"]["daily"]
  assert tf.total_samples == 8
  assert len(tf.results) == 4
  assert {(r.condition, r.outcome) for r in tf.results} == {
    ("reaction_green", "green"), ("reaction_green", "red"),
    ("reaction_red", "green"), ("reaction_red", "red"),
  }


def test_write_results_french_accents_not_escaped(tmp_path: Path) -> None:
  """French labels are stored as UTF-8 literals, not \\uXXXX escape sequences."""
  result = _stat([]).compute(_empty_df())
  written = write_results(result, results_dir=tmp_path)
  raw = written.read_text(encoding="utf-8")
  # French labels contain accented characters (e.g. 'Réaction', 'bougie')
  assert "é" in raw
  assert "\\u00e9" not in raw


def test_compute_result_validates_as_stat_run_result() -> None:
  """compute() output re-validates cleanly as a StatRunResult model."""
  result = _stat(_CPI_DATES_8).compute(_build_8day_df())
  revalidated = StatRunResult.model_validate(result.model_dump())
  assert revalidated.stat_name == "cpi_reaction"
  rows = revalidated.instruments["NQ"]["daily"].results
  assert len(rows) == 4


# ===========================================================================
# classify_samples
#
# Reusing _build_8day_df / _CPI_DATES_8 (see the 8-day matrix table above).
# ===========================================================================

def test_classify_samples_exact_list() -> None:
  """classify_samples emits one SampleRow per qualifying CPI day, matching the hand-calc."""
  stat = _stat(_CPI_DATES_8)
  table = stat.build_day_table(_build_8day_df())
  samples = stat.classify_samples(table)
  assert [(s.date, s.condition, s.outcome) for s in samples] == [
    ("2024-01-02", "reaction_green", "green"),
    ("2024-01-03", "reaction_green", "green"),
    ("2024-01-04", "reaction_green", "green"),
    ("2024-01-05", "reaction_green", "red"),
    ("2024-01-08", "reaction_green", "red"),
    ("2024-01-09", "reaction_red", "green"),
    ("2024-01-10", "reaction_red", "red"),
    ("2024-01-11", "reaction_red", "red"),
  ]
  assert all(s.value is None for s in samples)


def test_classify_samples_matches_compute_rows_counts() -> None:
  """For every StatResultRow, the matching SampleRow count equals r.count."""
  stat = _stat(_CPI_DATES_8)
  table = stat.build_day_table(_build_8day_df())
  samples = stat.classify_samples(table)
  rows = stat.compute_rows(table)
  for r in rows:
    n = sum(1 for s in samples if s.condition == r.condition and s.outcome == r.outcome)
    assert n == r.count


def test_classify_samples_empty_day_table() -> None:
  """Empty day_table -> classify_samples returns []."""
  stat = _stat(_CPI_DATES_8)
  assert stat.classify_samples(pd.DataFrame(columns=["reaction_green", "day_green"])) == []


def test_classify_samples_excludes_non_qualifying_day() -> None:
  """A CPI day missing the reaction open bar is dropped by build_day_table,
  so classify_samples emits no SampleRow for it."""
  bad_date = "2024-02-01"
  stat = _stat(["2024-01-02", bad_date])
  df = _make_candles([
    _make_cpi_day("2024-01-02", 100.0, 110.0, 100.0, 115.0),
    _make_missing_reaction_open_day(bad_date),
  ])
  table = stat.build_day_table(df)
  samples = stat.classify_samples(table)
  sample_dates = {s.date for s in samples}
  assert bad_date not in sample_dates
  assert "2024-01-02" in sample_dates
