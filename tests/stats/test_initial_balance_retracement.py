"""Tests for stats.initial_balance.retracement (InitialBalanceRetracement).

All data is synthetic — no real market files required. Expected retracement
fractions are hand-calculated before each assertion.

Framing (single condition ``single_break``, probability channel): on single-break
days only, after price breaks one side of the initial balance (IB), how deep does
it pull back into the balance range, as a fraction of IB size? Reported as the
share of single-break sessions whose retracement reaches at least each configured
threshold (default 0.25 / 0.50 / 0.75 of IB size). Non-single-break sessions (both
sides broke, or neither) are excluded from every denominator.

The IB is fixed at 90–110 (size 20) by the 09:30 bar's wicks. ``base_price`` sets
both the neutral fill bars and the controllable retrace floor/ceiling: because the
deepest pull-back is the min low (up-break) or max high (down-break) from the first
break bar onward, putting the neutral bars AT the target retrace level lets each
test pin ``depth_frac`` exactly. ``events`` overrides specific breakout-window
minutes with an explicit ``(high, low, close)`` bar.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.initial_balance.retracement import (
  InitialBalanceRetracement,
  _outcome_key,
)

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
_IB_HIGH = 110.0
_IB_LOW = 90.0
_IB_SIZE = 20.0


def _make_day(
  date: str,
  *,
  base_price: float,
  events: dict[int, tuple[float, float, float]] | None = None,
  ib_high: float = _IB_HIGH,
  ib_low: float = _IB_LOW,
  session_open: float | None = None,
  session_close: float | None = None,
  last_mod: int = _RTH_LAST,
) -> pd.DataFrame:
  """One dense day of 1-min RTH bars with a controllable initial balance.

  The 09:30 bar carries ``ib_high`` / ``ib_low`` as wicks (defining the IB) and
  opens at ``session_open``. Every other minute defaults to a neutral bar at
  ``base_price`` (kept inside the IB so it never breaks). ``events`` overrides
  specific minutes with an explicit ``(high, low, close)`` bar. The last bar's
  close is set to ``session_close``.
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


def _stat(
  timeframe: str = "30min",
  breakout_criteria: str = "wick",
  thresholds: tuple[float, ...] = (0.25, 0.50, 0.75),
) -> InitialBalanceRetracement:
  return InitialBalanceRetracement(
    instrument="NQ",
    timeframe=timeframe,
    config=_TEST_CONFIG,
    breakout_criteria=breakout_criteria,
    thresholds=thresholds,
  )


def _rows(stat: InitialBalanceRetracement, candles: pd.DataFrame) -> dict:
  result = stat.compute(candles)
  return {r.outcome: r for r in result.instruments["NQ"][stat.timeframe].results}


def _table_col(
  stat: InitialBalanceRetracement, candles: pd.DataFrame, column: str
) -> dict:
  table = stat.build_day_table(candles)
  return {d.strftime("%Y-%m-%d"): v for d, v in table[column].items()}


def _up_day(date: str, retrace_low: float) -> pd.DataFrame:
  """Single up-break day whose deepest pull-back floor is ``retrace_low``.

  Neutral bars sit at ``retrace_low`` and the 10:00 event wicks up to 115 with its
  low at ``retrace_low`` — so the min low from the break onward is exactly
  ``retrace_low``. ``depth_frac = (ib_high - retrace_low) / ib_size``.
  """
  return _make_day(
    date,
    base_price=retrace_low,
    events={_IB_END_30: (115.0, retrace_low, retrace_low)},
  )


def _down_day(date: str, retrace_high: float) -> pd.DataFrame:
  """Single down-break day whose deepest pull-back ceiling is ``retrace_high``.

  Neutral bars sit at ``retrace_high`` and the 10:00 event wicks down to 85 with
  its high at ``retrace_high``. ``depth_frac = (retrace_high - ib_low) / ib_size``.
  """
  return _make_day(
    date,
    base_price=retrace_high,
    events={_IB_END_30: (retrace_high, 85.0, retrace_high)},
  )


# ===========================================================================
# Retracement depth on single-break days
# ===========================================================================
def test_up_break_retracement_fraction() -> None:
  # Break up to 115, pull back down to 100 -> depth 10 / size 20 = 0.50.
  candles = _concat([_up_day("2024-01-02", retrace_low=100.0)])
  assert _table_col(_stat(), candles, "depth_frac")["2024-01-02"] == pytest.approx(0.50)
  assert _table_col(_stat(), candles, "break_up")["2024-01-02"] is True
  rows = _rows(_stat(), candles)
  # Nested thresholds: 0.50 hits retrace_25 and retrace_50, not retrace_75.
  assert rows["retrace_25"].count == 1
  assert rows["retrace_50"].count == 1
  assert rows["retrace_75"].count == 0
  assert rows["retrace_25"].total == 1


def test_down_break_retracement_fraction() -> None:
  # Break down to 85, pull back up to 105 -> depth 15 / size 20 = 0.75.
  candles = _concat([_down_day("2024-01-02", retrace_high=105.0)])
  assert _table_col(_stat(), candles, "depth_frac")["2024-01-02"] == pytest.approx(0.75)
  assert _table_col(_stat(), candles, "break_up")["2024-01-02"] is False
  rows = _rows(_stat(), candles)
  assert rows["retrace_25"].count == 1
  assert rows["retrace_50"].count == 1
  assert rows["retrace_75"].count == 1


def test_shallow_retracement_hits_no_threshold() -> None:
  # Break up to 115, pull back only to 107 -> depth 3 / 20 = 0.15 < 0.25.
  candles = _concat([_up_day("2024-01-02", retrace_low=107.0)])
  assert _table_col(_stat(), candles, "depth_frac")["2024-01-02"] == pytest.approx(0.15)
  rows = _rows(_stat(), candles)
  assert rows["retrace_25"].count == 0
  # Still a countable single-break session.
  assert rows["retrace_25"].total == 1


def test_no_pull_back_is_zero_depth() -> None:
  # Break up to 115 and never come back below ib_high (low stays at 112) -> the
  # raw penetration is negative and clipped to 0.
  candles = _concat([
    _make_day("2024-01-02", base_price=112.0,
              events={_IB_END_30: (115.0, 112.0, 112.0)}),
  ])
  # base_price 112 is above ib_high (110): neutral bars never re-enter the range.
  assert _table_col(_stat(), candles, "depth_frac")["2024-01-02"] == pytest.approx(0.0)
  rows = _rows(_stat(), candles)
  assert rows["retrace_25"].count == 0
  assert rows["retrace_25"].total == 1


# ===========================================================================
# Retracement is measured from the FIRST break bar onward (chronology)
# ===========================================================================
def test_pre_break_low_is_not_counted() -> None:
  # 10:00 dips to 92 (inside IB, no break), THEN 10:05 breaks up to 115 and only
  # pulls back to 105 afterward. The pre-break dip to 92 must be ignored, so the
  # depth is 110 - 105 = 5 (frac 0.25), NOT 110 - 92 = 18.
  candles = _concat([
    _make_day("2024-01-02", base_price=106.0, events={
      _IB_END_30: (106.0, 92.0, 106.0),        # pre-break dip, low 92 (holds)
      _IB_END_30 + 5: (115.0, 105.0, 106.0),   # up break, pull-back floor 105
    }),
  ])
  assert _table_col(_stat(), candles, "depth_frac")["2024-01-02"] == pytest.approx(0.25)


def test_retracement_uses_wick_not_close() -> None:
  # After breaking up, a later bar WICKS down to 95 (deep) but CLOSES at 108. The
  # pull-back is measured by the intraday low (95) -> depth 15 / 20 = 0.75, not by
  # the close (108) which would give 0.10.
  candles = _concat([
    _make_day("2024-01-02", base_price=108.0, events={
      _IB_END_30: (115.0, 108.0, 108.0),       # up break
      _IB_END_30 + 1: (108.0, 95.0, 108.0),    # deep wick to 95, closes 108
    }),
  ])
  assert _table_col(_stat(), candles, "depth_frac")["2024-01-02"] == pytest.approx(0.75)


# ===========================================================================
# Nested threshold counting across several days
# ===========================================================================
def test_nested_threshold_hit_counts() -> None:
  # Three single-break days at fracs 0.25, 0.50, 0.75.
  candles = _concat([
    _up_day("2024-01-02", retrace_low=105.0),   # 0.25
    _up_day("2024-01-03", retrace_low=100.0),   # 0.50
    _down_day("2024-01-04", retrace_high=105.0),  # 0.75
  ])
  rows = _rows(_stat(), candles)
  assert rows["retrace_25"].count == 3   # all three reach >= 0.25
  assert rows["retrace_50"].count == 2   # the 0.50 and 0.75 days
  assert rows["retrace_75"].count == 1   # only the 0.75 day
  assert rows["retrace_25"].total == 3
  assert rows["retrace_25"].probability == pytest.approx(1.0)
  assert rows["retrace_50"].probability == pytest.approx(2 / 3)
  assert rows["retrace_75"].probability == pytest.approx(1 / 3)


# ===========================================================================
# Non-single-break sessions are excluded from the denominator
# ===========================================================================
def test_both_sides_break_is_excluded() -> None:
  candles = _concat([
    _up_day("2024-01-02", retrace_low=100.0),  # single up break, frac 0.50
    _make_day("2024-01-03", base_price=100.0, events={
      _IB_END_30: (115.0, 100.0, 100.0),       # up break
      _IB_END_30 + 1: (100.0, 85.0, 100.0),    # down break -> both sides
    }),
  ])
  result = _stat().compute(candles)
  tf = result.instruments["NQ"]["30min"]
  assert tf.total_samples == 2          # both are resolved sessions
  rows = {r.outcome: r for r in tf.results}
  assert rows["retrace_25"].total == 1  # only the single-break day is countable
  # The both-sides day carries NaN depth and is not a single break.
  table = _stat().build_day_table(candles)
  assert pd.isna(table.loc[table.index[1], "depth_frac"])
  assert pd.isna(table.loc[table.index[1], "break_up"])


def test_no_break_session_is_excluded() -> None:
  candles = _concat([
    _up_day("2024-01-02", retrace_low=100.0),
    _make_day("2024-01-03", base_price=100.0),  # no events -> never breaks
  ])
  rows = _rows(_stat(), candles)
  assert rows["retrace_25"].total == 1
  depth = _table_col(_stat(), candles, "depth_frac")
  assert depth["2024-01-02"] == pytest.approx(0.50)
  assert pd.isna(depth["2024-01-03"])


# ===========================================================================
# Breakout criteria: wick vs close
# ===========================================================================
def test_wick_break_is_not_a_close_break() -> None:
  # High pokes to 115 (> ib_high 110) but the bar CLOSES inside at 100.
  candles = _concat([_up_day("2024-01-02", retrace_low=100.0)])
  # wick: a single up break, frac 0.50.
  assert _rows(_stat(breakout_criteria="wick"), candles)["retrace_50"].count == 1
  # close: no bar closes beyond the level -> not a single break -> nothing countable.
  close_rows = _rows(_stat(breakout_criteria="close"), candles)
  assert close_rows["retrace_25"].total == 0


def test_close_break_uses_wick_for_retracement() -> None:
  # A bar CLOSES at 112 (> ib_high 110) so the close criteria sees an up break;
  # its low (100) sets the pull-back floor -> depth 10 / 20 = 0.50.
  candles = _concat([
    _make_day("2024-01-02", base_price=100.0,
              events={_IB_END_30: (112.0, 100.0, 112.0)}),
  ])
  rows = _rows(_stat(breakout_criteria="close"), candles)
  assert rows["retrace_50"].count == 1
  assert rows["retrace_50"].total == 1


# ===========================================================================
# IB window length per timeframe (30min vs 1h)
# ===========================================================================
def test_ib_window_length_tracks_timeframe() -> None:
  # A 10:15 (mod 615) up break. With a 30-min IB it is in the breakout window and
  # breaks; with a 1h IB (09:30-10:30) the same bar is INSIDE the IB, lifting
  # ib_high, so nothing breaks afterward -> not countable.
  candles = _concat([
    _make_day("2024-01-02", base_price=100.0, events={615: (115.0, 100.0, 100.0)}),
  ])
  assert _rows(_stat("30min"), candles)["retrace_50"].total == 1
  assert _rows(_stat("1h"), candles)["retrace_25"].total == 0


# ===========================================================================
# Validation
# ===========================================================================
def test_invalid_breakout_criteria_raises() -> None:
  with pytest.raises(ValueError, match="breakout_criteria"):
    _stat(breakout_criteria="bogus")


def test_invalid_timeframe_raises() -> None:
  with pytest.raises(ValueError, match="Unsupported timeframe"):
    _stat(timeframe="7min")


def test_empty_thresholds_raises() -> None:
  with pytest.raises(ValueError, match="non-empty"):
    _stat(thresholds=())


def test_non_positive_threshold_raises() -> None:
  with pytest.raises(ValueError, match="must be > 0"):
    _stat(thresholds=(0.0, 0.5))


# ===========================================================================
# Configurable thresholds
# ===========================================================================
def test_custom_thresholds_produce_matching_outcomes_and_labels() -> None:
  candles = _concat([_up_day("2024-01-02", retrace_low=95.0)])  # frac 0.75
  stat = _stat(thresholds=(0.10, 0.90))
  result = stat.compute(candles)
  rows = {r.outcome: r for r in result.instruments["NQ"]["30min"].results}
  assert set(rows) == {"retrace_10", "retrace_90"}
  assert rows["retrace_10"].count == 1   # 0.75 >= 0.10
  assert rows["retrace_90"].count == 0   # 0.75 < 0.90
  # Labels stay self-documenting for the custom thresholds.
  assert set(result.labels.outcomes) == {"retrace_10", "retrace_90"}
  assert _outcome_key(0.9) == "retrace_90"


# ===========================================================================
# Baseline (uniform null)
# ===========================================================================
def test_baseline_is_deterministic_for_fixed_seed() -> None:
  stat = _stat()
  table = stat.build_day_table(_concat([
    _up_day("2024-01-02", retrace_low=105.0),
    _up_day("2024-01-03", retrace_low=100.0),
    _down_day("2024-01-04", retrace_high=105.0),
  ]))
  a = stat.baseline_rows(table, seed=42)
  b = stat.baseline_rows(table, seed=42)
  assert [(r.outcome, r.count, r.probability) for r in a] == \
    [(r.outcome, r.count, r.probability) for r in b]


def test_baseline_excludes_non_single_break_sessions() -> None:
  # One single-break day plus one no-break day. The baseline denominator must be
  # the single-break count only.
  candles = _concat([
    _up_day("2024-01-02", retrace_low=100.0),
    _make_day("2024-01-03", base_price=100.0),  # no break
  ])
  stat = _stat()
  table = stat.build_day_table(candles)
  baseline = {r.outcome: r for r in stat.baseline_rows(table, seed=42)}
  assert baseline["retrace_25"].total == 1


def test_baseline_converges_to_one_minus_threshold() -> None:
  # Many single-break days, each with a different (but irrelevant) retrace floor;
  # the uniform null assigns depth ~ U(0,1) so P(hit >= t) -> 1 - t. With a fixed
  # seed and a large sample the baseline lands near (0.75, 0.50, 0.25).
  days = [
    _up_day((pd.Timestamp("2020-01-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
            retrace_low=100.0)
    for i in range(400)
  ]
  stat = _stat()
  table = stat.build_day_table(_concat(days))
  baseline = {r.outcome: r for r in stat.baseline_rows(table, seed=42)}
  assert baseline["retrace_25"].probability == pytest.approx(0.75, abs=0.05)
  assert baseline["retrace_50"].probability == pytest.approx(0.50, abs=0.05)
  assert baseline["retrace_75"].probability == pytest.approx(0.25, abs=0.05)


def test_baseline_merged_into_result_rows() -> None:
  candles = _concat([_up_day("2024-01-02", retrace_low=100.0)])
  rows = _rows(_stat(), candles)
  for r in rows.values():
    assert r.baseline_n == 1
    # baseline_prob is populated (a uniform draw produced a 0/1 hit rate).
    assert r.baseline_prob in (0.0, 1.0)


# ===========================================================================
# Reproducibility & resolution discipline
# ===========================================================================
def test_compute_is_reproducible() -> None:
  candles = _concat([
    _up_day("2024-01-02", retrace_low=100.0),
    _down_day("2024-01-03", retrace_high=105.0),
  ])
  a = _stat().compute(candles).model_dump()
  b = _stat().compute(candles).model_dump()
  assert a == b


def test_unresolved_day_is_excluded() -> None:
  candles = _concat([
    _up_day("2024-01-02", retrace_low=100.0),
    _make_truncated_day("2024-01-03"),
  ])
  tf = _stat().compute(candles).instruments["NQ"]["30min"]
  assert tf.total_samples == 1
  rows = {r.outcome: r for r in tf.results}
  assert rows["retrace_50"].total == 1


def test_empty_input_yields_zero_rows() -> None:
  stat = _stat()
  table = stat.build_day_table(_empty_df())
  assert table.empty
  assert "depth_frac" in table.columns
  assert "break_up" in table.columns

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
  candles = _concat([_up_day("2024-01-02", retrace_low=100.0)])
  slices = _stat().compute(candles).instruments["NQ"]["30min"].slices
  assert set(slices) == {
    "weekday", "close", "prev_candle", "overnight", "size", "size_pct"
  }


def test_close_slice_splits_by_session_colour() -> None:
  # Two green days (close 105 > open 100) and one red (close 95 < open 100), all
  # single up breaks pulling back to 100 (frac 0.50). session_close stays inside
  # the IB so it does not itself break.
  candles = _concat([
    _make_day("2024-01-02", base_price=100.0,
              events={_IB_END_30: (115.0, 100.0, 100.0)},
              session_open=100.0, session_close=105.0),
    _make_day("2024-01-03", base_price=100.0,
              events={_IB_END_30: (115.0, 100.0, 100.0)},
              session_open=100.0, session_close=104.0),
    _make_day("2024-01-04", base_price=100.0,
              events={_IB_END_30: (115.0, 100.0, 100.0)},
              session_open=100.0, session_close=95.0),
  ])
  groups = _stat().compute(candles).instruments["NQ"]["30min"].slices["close"].groups
  assert groups["green"].total_samples == 2
  assert groups["red"].total_samples == 1
  green_rows = {r.outcome: r for r in groups["green"].results}
  assert green_rows["retrace_50"].count == 2


def test_size_slice_buckets_by_ib_size() -> None:
  # Four distinct IB sizes -> quartile buckets, one single up-break day each.
  candles = _concat([
    _make_day("2024-01-02", base_price=100.0, ib_high=105.0, ib_low=95.0,
              events={_IB_END_30: (130.0, 100.0, 100.0)}),
    _make_day("2024-01-03", base_price=100.0, ib_high=110.0, ib_low=90.0,
              events={_IB_END_30: (130.0, 100.0, 100.0)}),
    _make_day("2024-01-04", base_price=100.0, ib_high=115.0, ib_low=85.0,
              events={_IB_END_30: (130.0, 100.0, 100.0)}),
    _make_day("2024-01-05", base_price=100.0, ib_high=120.0, ib_low=80.0,
              events={_IB_END_30: (130.0, 100.0, 100.0)}),
  ])
  groups = _stat().compute(candles).instruments["NQ"]["30min"].slices["size"].groups
  assert sum(g.total_samples for g in groups.values()) == 4


# ===========================================================================
# JSON round-trip
# ===========================================================================
def test_result_validates_and_writes(tmp_path: Path) -> None:
  candles = _concat([_up_day("2024-01-02", retrace_low=100.0)])
  result = _stat().compute(candles)
  out = write_results(result, results_dir=tmp_path)
  assert out.exists()
  data = json.loads(out.read_text(encoding="utf-8"))
  assert data["stat_name"] == "initial_balance_retracement"
  assert data["title"]["en"] == "Initial Balance Breakout — Retracement"
  StatRunResult.model_validate(data)
