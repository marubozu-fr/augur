"""Tests for stats.opening_range_breakout.standard.

All data is synthetic — no real market files required. Expected values are
hand-calculated before each assertion.

Framing (single condition ``orb``): given the opening range formed from the first
``orb_period`` minutes of the session, which direction does price break during the
REST of the session? Four mutually exclusive outcomes partition every countable
day (strict inequality defines a break — touching the level is NOT a break):
  - ``broke_high``: post_high  >  orb_high AND post_low  >= orb_low
  - ``broke_low`` : post_low   <  orb_low  AND post_high <= orb_high
  - ``broke_both``: post_high  >  orb_high AND post_low  <  orb_low
  - ``neither``   : post_high  <= orb_high AND post_low  >= orb_low

With ``breakout_criteria='wick'`` (default) the breakout window's intraday
extremes are used; with ``'close'`` its extreme bar CLOSES are used instead.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.opening_range_breakout.standard import OpeningRangeBreakout

# ---------------------------------------------------------------------------
# Minimal InstrumentConfig (does not depend on NQ.yaml)
# ---------------------------------------------------------------------------
_NY = "America/New_York"
_RTH_SESSION = Session(start="09:30", end="16:15")
_TEST_CONFIG = InstrumentConfig(
  instrument="NQ",
  working_timezone=_NY,
  sessions={"rth": _RTH_SESSION},
  timeframes=["15min", "30min", "1h"],
  parquet_path=Path("data/NQ_1min.parquet"),
)

_RTH_START = 570   # 09:30
_RTH_LAST = 974    # 16:14 (last bar before 16:15); resolved needs last mod >= 960


def _make_orb_day(
  date: str,
  *,
  orb_high: float,
  orb_low: float,
  post_high: float,
  post_low: float,
  session_open: float | None = None,
  session_close: float | None = None,
  post_close_high: float | None = None,
  post_close_low: float | None = None,
  orb_minutes: int = 15,
  last_mod: int = _RTH_LAST,
) -> pd.DataFrame:
  """One trading day of 1-min RTH bars with a controllable opening range.

  The ORB window ``[09:30, 09:30 + orb_minutes)`` carries ``orb_high`` / ``orb_low``
  as wicks; the breakout window ``[09:30 + orb_minutes, last_mod]`` carries
  ``post_high`` / ``post_low`` as wicks and ``post_close_high`` / ``post_close_low``
  as bar CLOSES (defaulting to the ORB midpoint, i.e. no close-based break).

  ``session_open`` defaults to the ORB midpoint (an open inside the range);
  ``session_close`` (the last bar's close) likewise, so colour is neutral unless set.
  """
  base = pd.Timestamp(date, tz=_NY)
  orb_mid = (orb_high + orb_low) / 2.0
  if session_open is None:
    session_open = orb_mid
  if session_close is None:
    session_close = orb_mid
  if post_close_high is None:
    post_close_high = orb_mid
  if post_close_low is None:
    post_close_low = orb_mid

  orb_end = _RTH_START + orb_minutes
  records = []
  for mod in range(_RTH_START, last_mod + 1):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    if mod < orb_end:
      # ORB window: open on the first bar, extremes as wicks on the next two bars.
      o = session_open if mod == _RTH_START else orb_mid
      c = orb_mid
      hi = orb_high if mod == _RTH_START + 1 else max(o, c)
      lo = orb_low if mod == _RTH_START + 2 else min(o, c)
    else:
      # Breakout window: wick extremes, close extremes, then the session close.
      o = orb_mid
      c = orb_mid
      if mod == orb_end:
        hi, lo = post_high, orb_mid
      elif mod == orb_end + 1:
        hi, lo = orb_mid, post_low
      elif mod == orb_end + 2:
        c = post_close_high
        hi, lo = max(orb_mid, c), orb_mid
      elif mod == orb_end + 3:
        c = post_close_low
        hi, lo = orb_mid, min(orb_mid, c)
      else:
        hi, lo = max(o, c), min(o, c)
      if mod == last_mod:
        c = session_close
        hi, lo = max(hi, c), min(lo, c)
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


def _stat(timeframe: str = "15min", breakout_criteria: str = "wick") -> OpeningRangeBreakout:
  return OpeningRangeBreakout(
    instrument="NQ",
    timeframe=timeframe,
    config=_TEST_CONFIG,
    breakout_criteria=breakout_criteria,
  )


def _row(result: StatRunResult, outcome: str, timeframe: str = "15min") -> StatResultRow:
  for r in result.instruments["NQ"][timeframe].results:
    if r.condition == "orb" and r.outcome == outcome:
      return r
  raise KeyError(("orb", outcome))


def _rows(stat: OpeningRangeBreakout, candles: pd.DataFrame) -> dict[str, StatResultRow]:
  result = stat.compute(candles)
  return {r.outcome: r for r in result.instruments["NQ"][stat.timeframe].results}


# A canonical 5-day set for the 15-min ORB (orb_high=110, orb_low=90):
#   broke_high, broke_low, broke_both, neither, boundary-touch (neither).
_CANON = [
  # broke_high: post_high 115 > 110, post_low 95 >= 90
  dict(date="2024-01-02", orb_high=110, orb_low=90, post_high=115, post_low=95),
  # broke_low: post_high 105 <= 110, post_low 85 < 90
  dict(date="2024-01-03", orb_high=110, orb_low=90, post_high=105, post_low=85),
  # broke_both: post_high 115 > 110, post_low 85 < 90
  dict(date="2024-01-04", orb_high=110, orb_low=90, post_high=115, post_low=85),
  # neither: post_high 108 <= 110, post_low 92 >= 90
  dict(date="2024-01-05", orb_high=110, orb_low=90, post_high=108, post_low=92),
  # boundary touch (not a break): post_high == 110, post_low == 90 -> neither
  dict(date="2024-01-08", orb_high=110, orb_low=90, post_high=110, post_low=90),
]


def _canon_candles() -> pd.DataFrame:
  return _concat([_make_orb_day(**spec) for spec in _CANON])


# ===========================================================================
# Core classification
# ===========================================================================
def test_four_outcomes_partition_countable_days() -> None:
  rows = _rows(_stat(), _canon_candles())
  # 5 countable days: 1 high, 1 low, 1 both, 2 neither (incl. the boundary touch).
  assert rows["broke_high"].count == 1
  assert rows["broke_low"].count == 1
  assert rows["broke_both"].count == 1
  assert rows["neither"].count == 2
  # Every row's total is the countable-day count; counts sum to total.
  totals = {r.total for r in rows.values()}
  assert totals == {5}
  assert sum(r.count for r in rows.values()) == 5


def test_boundary_touch_is_not_a_break() -> None:
  # A day that only touches orb_high / orb_low exactly classifies as neither.
  candles = _concat([
    _make_orb_day(date="2024-01-02", orb_high=110, orb_low=90, post_high=110, post_low=90),
  ])
  rows = _rows(_stat(), candles)
  assert rows["neither"].count == 1
  assert rows["broke_high"].count == 0
  assert rows["broke_low"].count == 0
  assert rows["broke_both"].count == 0


def test_probabilities_are_count_over_total() -> None:
  rows = _rows(_stat(), _canon_candles())
  assert rows["broke_high"].probability == pytest.approx(1 / 5)
  assert rows["broke_low"].probability == pytest.approx(1 / 5)
  assert rows["broke_both"].probability == pytest.approx(1 / 5)
  assert rows["neither"].probability == pytest.approx(2 / 5)
  assert sum(r.probability for r in rows.values()) == pytest.approx(1.0)


def test_total_samples_counts_resolved_days() -> None:
  result = _stat().compute(_canon_candles())
  assert result.instruments["NQ"]["15min"].total_samples == 5
  assert result.instruments["NQ"]["15min"].data_range == ["2024-01-02", "2024-01-08"]


# ===========================================================================
# Breakout criteria: wick vs close
# ===========================================================================
def test_wick_break_does_not_count_as_close_break() -> None:
  # Wick pokes above orb_high (115 > 110) but every post bar CLOSES inside the
  # range (post_close_high = 100). Wick criteria -> broke_high; close -> neither.
  candles = _concat([
    _make_orb_day(
      date="2024-01-02", orb_high=110, orb_low=90,
      post_high=115, post_low=95, post_close_high=100,
    ),
  ])
  assert _rows(_stat(breakout_criteria="wick"), candles)["broke_high"].count == 1
  assert _rows(_stat(breakout_criteria="close"), candles)["neither"].count == 1


def test_close_beyond_level_counts_as_close_break() -> None:
  # A post bar CLOSES above orb_high (post_close_high = 112 > 110), so the close
  # criteria classifies broke_high. The wick reaches the same value so wick agrees.
  candles = _concat([
    _make_orb_day(
      date="2024-01-02", orb_high=110, orb_low=90,
      post_high=112, post_low=95, post_close_high=112,
    ),
  ])
  rows = _rows(_stat(breakout_criteria="close"), candles)
  assert rows["broke_high"].count == 1
  assert rows["broke_low"].count == 0


def test_invalid_breakout_criteria_raises() -> None:
  with pytest.raises(ValueError, match="breakout_criteria"):
    _stat(breakout_criteria="bogus")


def test_invalid_timeframe_raises() -> None:
  with pytest.raises(ValueError, match="Unsupported timeframe"):
    _stat(timeframe="7min")


# ===========================================================================
# ORB window length per timeframe
# ===========================================================================
def test_orb_window_length_tracks_timeframe() -> None:
  # The same day, read with a 30-min ORB, includes a 09:50 spike in the OPENING
  # range (so it is part of orb_high, not a breakout). With a 15-min ORB the same
  # 09:50 spike falls in the breakout window and breaks the high.
  spike_day = _make_orb_day(
    date="2024-01-02", orb_high=110, orb_low=90,
    post_high=115, post_low=95, orb_minutes=15,
  )
  candles = _concat([spike_day])
  # 15-min ORB: orb_high stays 110, the 115 spike at 09:45 breaks it.
  assert _rows(_stat("15min"), candles)["broke_high"].count == 1
  # 30-min ORB: the window now swallows the 09:45 spike, so orb_high = 115 and the
  # post window (post_low 95 only) holds the range -> neither.
  assert _rows(_stat("30min"), candles)["neither"].count == 1


def test_orb_size_column_is_range_width() -> None:
  table = _stat().build_day_table(_canon_candles())
  # orb_high - orb_low = 110 - 90 = 20 on every canonical day.
  assert (table["orb_size"] == 20.0).all()


# ===========================================================================
# Baseline (random directional null via reflection)
# ===========================================================================
def test_baseline_preserves_direction_symmetric_outcomes() -> None:
  # Reflection swaps broke_high <-> broke_low but leaves broke_both and neither
  # unchanged, so their baseline counts equal the actual counts exactly.
  stat = _stat()
  table = stat.build_day_table(_canon_candles())
  bl = {r.outcome: r for r in stat.baseline_rows(table, seed=42)}
  actual = {r.outcome: r for r in stat.compute_rows(table)}
  assert bl["broke_both"].count == actual["broke_both"].count
  assert bl["neither"].count == actual["neither"].count
  # Directional mass is conserved, only redistributed between high and low.
  assert (bl["broke_high"].count + bl["broke_low"].count) == (
    actual["broke_high"].count + actual["broke_low"].count
  )


def test_baseline_is_deterministic_for_fixed_seed() -> None:
  stat = _stat()
  table = stat.build_day_table(_canon_candles())
  a = stat.baseline_rows(table, seed=42)
  b = stat.baseline_rows(table, seed=42)
  assert [(r.outcome, r.count) for r in a] == [(r.outcome, r.count) for r in b]


def test_baseline_merged_into_result_rows() -> None:
  result = _stat().compute(_canon_candles())
  for r in result.instruments["NQ"]["15min"].results:
    # baseline_n is the countable-day count carried from the baseline rows.
    assert r.baseline_n == 5


# ===========================================================================
# Pending / resolution discipline
# ===========================================================================
def test_unresolved_day_is_excluded() -> None:
  candles = _concat([
    _make_orb_day(date="2024-01-02", orb_high=110, orb_low=90, post_high=115, post_low=95),
    _make_truncated_day("2024-01-03"),
  ])
  result = _stat().compute(candles)
  # Only the resolved day counts; the truncated day is dropped entirely.
  assert result.instruments["NQ"]["15min"].total_samples == 1
  rows = {r.outcome: r for r in result.instruments["NQ"]["15min"].results}
  assert rows["broke_high"].count == 1
  assert rows["broke_high"].total == 1


def test_empty_input_yields_zero_rows() -> None:
  result = _stat().compute(_empty_df())
  tf = result.instruments["NQ"]["15min"]
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
  result = _stat().compute(_canon_candles())
  slices = result.instruments["NQ"]["15min"].slices
  assert set(slices) == {"weekday", "close", "prev_candle", "size"}


def test_close_slice_splits_by_session_colour() -> None:
  # Two green days (close > open) and one red day (close < open), all broke_high.
  candles = _concat([
    _make_orb_day(date="2024-01-02", orb_high=110, orb_low=90, post_high=115,
                  post_low=95, session_open=100, session_close=105),
    _make_orb_day(date="2024-01-03", orb_high=110, orb_low=90, post_high=115,
                  post_low=95, session_open=100, session_close=104),
    _make_orb_day(date="2024-01-04", orb_high=110, orb_low=90, post_high=115,
                  post_low=95, session_open=100, session_close=95),
  ])
  groups = _stat().compute(candles).instruments["NQ"]["15min"].slices["close"].groups
  assert groups["green"].total_samples == 2
  assert groups["red"].total_samples == 1


def test_size_slice_buckets_by_orb_size() -> None:
  # Four distinct ORB sizes -> quartile buckets q1..q4, one day each.
  candles = _concat([
    _make_orb_day(date="2024-01-02", orb_high=105, orb_low=95, post_high=120, post_low=80),
    _make_orb_day(date="2024-01-03", orb_high=110, orb_low=90, post_high=120, post_low=80),
    _make_orb_day(date="2024-01-04", orb_high=115, orb_low=85, post_high=120, post_low=80),
    _make_orb_day(date="2024-01-05", orb_high=120, orb_low=80, post_high=130, post_low=70),
  ])
  groups = _stat().compute(candles).instruments["NQ"]["15min"].slices["size"].groups
  assert sum(g.total_samples for g in groups.values()) == 4


# ===========================================================================
# JSON round-trip
# ===========================================================================
def test_result_validates_and_writes(tmp_path: Path) -> None:
  result = _stat().compute(_canon_candles())
  out = write_results(result, results_dir=tmp_path)
  assert out.exists()
  data = json.loads(out.read_text(encoding="utf-8"))
  assert data["stat_name"] == "opening_range_breakout"
  assert data["title"]["en"] == "Opening Range Breakout"
  # Re-validate the written payload through the Pydantic model.
  StatRunResult.model_validate(data)
