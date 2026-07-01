"""Tests for stats.initial_balance.performance (InitialBalancePerformance).

All data is synthetic — no real market files required. Expected extensions are
hand-calculated before each assertion.

Framing (single condition ``ib``, magnitude channel): given the initial balance
formed from the first ``ib_period`` minutes of the session, how far does price
extend past the IB high/low on its *chronologically first* excursion, before
trading back into the IB range? The first-breakout extension is reported as mean
/ max over countable sessions, in points (``extension``) and as a decimal of the
session open (``extension_pct``). Sessions that never break either side are
excluded from every denominator.

A flexible day builder (``_make_day``) fills a dense RTH session with neutral bars
at ``base_price`` (inside the IB) and overrides specific breakout-window minutes
with explicit ``(high, low, close)`` bars, so each excursion can be hand-computed.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.initial_balance.performance import InitialBalancePerformance

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
  extremes) and opens at ``session_open``. Every other minute defaults to a
  neutral bar at ``base_price`` (kept strictly inside the IB so it never breaks).
  ``events`` overrides specific minutes with an explicit ``(high, low, close)``
  bar — used to script the breakout window. The last bar's close is set to
  ``session_close`` (kept inside the IB by callers so it does not itself break).
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
      # Clamp the open into [low, high] so every event bar is OHLC-valid even when
      # the scripted bar sits entirely above/below base_price (the stat reads only
      # high/low/close of breakout bars, but synthetic bars must stay valid).
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
  timeframe: str = "30min", breakout_criteria: str = "wick"
) -> InitialBalancePerformance:
  return InitialBalancePerformance(
    instrument="NQ",
    timeframe=timeframe,
    config=_TEST_CONFIG,
    breakout_criteria=breakout_criteria,
  )


def _rows(stat: InitialBalancePerformance, candles: pd.DataFrame) -> dict[str, StatResultRow]:
  result = stat.compute(candles)
  return {r.outcome: r for r in result.instruments["NQ"][stat.timeframe].results}


def _table_col(
  stat: InitialBalancePerformance, candles: pd.DataFrame, column: str
) -> dict[str, float]:
  table = stat.build_day_table(candles)
  return {d.strftime("%Y-%m-%d"): v for d, v in table[column].items()}


# ===========================================================================
# Single clean breakouts (wick) — extension magnitude and direction
# ===========================================================================
def test_single_up_breakout_extension() -> None:
  # 10:00 bar wicks to 115 (ib_high 110) then re-enters (low 100) -> ext 5.
  candles = _concat([
    _make_day("2024-01-02", events={_IB_END_30: (115.0, 100.0, 100.0)}),
  ])
  rows = _rows(_stat(), candles)
  assert rows["mean_extension"].value == pytest.approx(5.0)
  assert rows["max_extension"].value == pytest.approx(5.0)
  assert rows["mean_extension"].count == 1
  assert rows["mean_extension"].total == 1
  # Magnitude stat: the probability channel stays at 0.
  assert rows["mean_extension"].probability == 0.0
  # extension_pct = 5 / session_open (default base 100) = 0.05.
  assert rows["mean_extension_pct"].value == pytest.approx(0.05)
  # Direction recorded as up, down side never broke.
  assert _table_col(_stat(), candles, "first_dir_up")["2024-01-02"] is True
  assert _table_col(_stat(), candles, "down_ext")["2024-01-02"] == pytest.approx(0.0)


def test_single_down_breakout_extension() -> None:
  # 10:00 bar wicks to 85 (ib_low 90) then re-enters (high 100) -> ext 5.
  candles = _concat([
    _make_day("2024-01-02", events={_IB_END_30: (100.0, 85.0, 100.0)}),
  ])
  rows = _rows(_stat(), candles)
  assert rows["mean_extension"].value == pytest.approx(5.0)
  assert _table_col(_stat(), candles, "first_dir_up")["2024-01-02"] is False
  assert _table_col(_stat(), candles, "up_ext")["2024-01-02"] == pytest.approx(0.0)


# ===========================================================================
# Non-breaking sessions are excluded from the denominator
# ===========================================================================
def test_no_breakout_session_is_excluded() -> None:
  # One breaking day (ext 5) and one that stays entirely inside the IB.
  candles = _concat([
    _make_day("2024-01-02", events={_IB_END_30: (115.0, 100.0, 100.0)}),
    _make_day("2024-01-03"),  # no events -> never breaks
  ])
  result = _stat().compute(candles)
  tf = result.instruments["NQ"]["30min"]
  # The non-breaking day is still a resolved session...
  assert tf.total_samples == 2
  rows = {r.outcome: r for r in tf.results}
  # ...but only the breaking day is countable.
  assert rows["mean_extension"].total == 1
  assert rows["mean_extension"].value == pytest.approx(5.0)
  # The excluded day carries NaN extension in the day table.
  ext = _table_col(_stat(), candles, "extension")
  assert ext["2024-01-02"] == pytest.approx(5.0)
  assert pd.isna(ext["2024-01-03"])


# ===========================================================================
# First-direction ordering when both sides break
# ===========================================================================
def test_up_breaks_first_uses_up_extension() -> None:
  # Up break at 10:00 (ext 5), down break later at 10:01 (ext 5). Up came first.
  candles = _concat([
    _make_day("2024-01-02", events={
      _IB_END_30: (115.0, 100.0, 100.0),       # up break, ext 5
      _IB_END_30 + 1: (100.0, 85.0, 100.0),    # down break, ext 5
    }),
  ])
  assert _table_col(_stat(), candles, "first_dir_up")["2024-01-02"] is True
  assert _table_col(_stat(), candles, "extension")["2024-01-02"] == pytest.approx(5.0)
  # The unchosen down side is still computed independently (for the baseline).
  assert _table_col(_stat(), candles, "down_ext")["2024-01-02"] == pytest.approx(5.0)
  assert _table_col(_stat(), candles, "up_ext")["2024-01-02"] == pytest.approx(5.0)


def test_down_breaks_first_uses_down_extension() -> None:
  # Down break at 10:00, up break later at 10:01. Down came first.
  candles = _concat([
    _make_day("2024-01-02", events={
      _IB_END_30: (100.0, 80.0, 100.0),        # down break, ext 10
      _IB_END_30 + 1: (115.0, 100.0, 100.0),   # up break, ext 5
    }),
  ])
  assert _table_col(_stat(), candles, "first_dir_up")["2024-01-02"] is False
  assert _table_col(_stat(), candles, "extension")["2024-01-02"] == pytest.approx(10.0)
  assert _table_col(_stat(), candles, "up_ext")["2024-01-02"] == pytest.approx(5.0)


# ===========================================================================
# The excursion stops at the first re-entry ("before breaking back")
# ===========================================================================
def test_extension_is_first_excursion_only() -> None:
  # First excursion runs to 115 (peak1) staying above ib_high, then bar 10:02
  # trades back into the range. A LATER bar reaches 120, but only the FIRST
  # excursion counts -> extension is 5 (115 - 110), NOT 10.
  candles = _concat([
    _make_day("2024-01-02", events={
      _IB_END_30: (113.0, 111.0, 112.0),       # break up, stays out (low 111 > 110)
      _IB_END_30 + 1: (115.0, 111.0, 114.0),   # peak1 = 115, still out
      _IB_END_30 + 2: (112.0, 108.0, 109.0),   # low 108 < 110 -> re-enters, excursion ends
      _IB_END_30 + 3: (120.0, 109.0, 119.0),   # later spike to 120 -> ignored
    }),
  ])
  assert _table_col(_stat(), candles, "extension")["2024-01-02"] == pytest.approx(5.0)


# ===========================================================================
# Mean vs max aggregation
# ===========================================================================
def test_mean_and_max_aggregation() -> None:
  # Three up-only sessions with extensions 5, 10, 15.
  candles = _concat([
    _make_day("2024-01-02", events={_IB_END_30: (115.0, 100.0, 100.0)}),  # 5
    _make_day("2024-01-03", events={_IB_END_30: (120.0, 100.0, 100.0)}),  # 10
    _make_day("2024-01-04", events={_IB_END_30: (125.0, 100.0, 100.0)}),  # 15
  ])
  rows = _rows(_stat(), candles)
  assert rows["mean_extension"].value == pytest.approx((5 + 10 + 15) / 3)
  assert rows["max_extension"].value == pytest.approx(15.0)
  assert rows["mean_extension"].count == 3
  # extension_pct aggregates the same way (each over open 100).
  assert rows["mean_extension_pct"].value == pytest.approx((0.05 + 0.10 + 0.15) / 3)
  assert rows["max_extension_pct"].value == pytest.approx(0.15)


# ===========================================================================
# Breakout criteria: wick vs close
# ===========================================================================
def test_wick_break_is_not_a_close_break() -> None:
  # High pokes to 115 (> ib_high 110) but the bar CLOSES inside (100).
  candles = _concat([
    _make_day("2024-01-02", events={_IB_END_30: (115.0, 100.0, 100.0)}),
  ])
  # wick: an extension of 5.
  assert _rows(_stat(breakout_criteria="wick"), candles)["mean_extension"].value == \
    pytest.approx(5.0)
  # close: no bar closes beyond the level -> no breakout -> nothing countable.
  close_rows = _rows(_stat(breakout_criteria="close"), candles)
  assert close_rows["mean_extension"].total == 0
  assert close_rows["mean_extension"].value == 0.0


def test_close_beyond_level_counts_as_close_break() -> None:
  # A bar CLOSES at 112 (> ib_high 110). Both criteria see a break of size 2.
  candles = _concat([
    _make_day("2024-01-02", events={_IB_END_30: (112.0, 100.0, 112.0)}),
  ])
  assert _rows(_stat(breakout_criteria="close"), candles)["mean_extension"].value == \
    pytest.approx(2.0)
  assert _rows(_stat(breakout_criteria="wick"), candles)["mean_extension"].value == \
    pytest.approx(2.0)


# ===========================================================================
# IB window length per timeframe (30min vs 1h)
# ===========================================================================
def test_ib_window_length_tracks_timeframe() -> None:
  # A 10:15 (mod 615) spike to 115. With a 30-min IB it falls in the breakout
  # window and breaks ib_high (110) -> extension 5. With a 1h IB (09:30-10:30)
  # the same bar is INSIDE the IB, lifting ib_high to 115, so the breakout window
  # (>= 10:30) never breaks -> not countable.
  candles = _concat([
    _make_day("2024-01-02", events={615: (115.0, 100.0, 100.0)}),
  ])
  rows_30 = _rows(_stat("30min"), candles)
  assert rows_30["mean_extension"].value == pytest.approx(5.0)
  assert rows_30["mean_extension"].total == 1

  rows_1h = _rows(_stat("1h"), candles)
  assert rows_1h["mean_extension"].total == 0
  assert rows_1h["mean_extension"].value == 0.0


def test_invalid_breakout_criteria_raises() -> None:
  with pytest.raises(ValueError, match="breakout_criteria"):
    _stat(breakout_criteria="bogus")


def test_invalid_timeframe_raises() -> None:
  with pytest.raises(ValueError, match="Unsupported timeframe"):
    _stat(timeframe="7min")


# ===========================================================================
# Baseline (random coin-flip side)
# ===========================================================================
def test_baseline_is_deterministic_for_fixed_seed() -> None:
  stat = _stat()
  table = stat.build_day_table(_concat([
    _make_day("2024-01-02", events={_IB_END_30: (115.0, 100.0, 100.0)}),
    _make_day("2024-01-03", events={_IB_END_30: (120.0, 100.0, 100.0)}),
    _make_day("2024-01-04", events={_IB_END_30: (125.0, 100.0, 100.0)}),
  ]))
  a = stat.baseline_rows(table, seed=42)
  b = stat.baseline_rows(table, seed=42)
  assert [(r.outcome, r.value) for r in a] == [(r.outcome, r.value) for r in b]


def test_baseline_at_or_below_actual_for_one_sided_sessions() -> None:
  # Every session breaks only up (down_ext 0). The coin sometimes selects the 0
  # side, so the baseline mean can only be <= the actual mean.
  candles = _concat([
    _make_day("2024-01-02", events={_IB_END_30: (115.0, 100.0, 100.0)}),  # 5
    _make_day("2024-01-03", events={_IB_END_30: (120.0, 100.0, 100.0)}),  # 10
    _make_day("2024-01-04", events={_IB_END_30: (125.0, 100.0, 100.0)}),  # 15
  ])
  stat = _stat()
  table = stat.build_day_table(candles)
  actual = {r.outcome: r for r in stat.compute_rows(table)}
  baseline = {r.outcome: r for r in stat.baseline_rows(table, seed=42)}
  assert baseline["mean_extension"].value <= actual["mean_extension"].value
  assert baseline["max_extension"].value <= actual["max_extension"].value


def test_baseline_merged_into_result_rows() -> None:
  candles = _concat([
    _make_day("2024-01-02", events={_IB_END_30: (115.0, 100.0, 100.0)}),
  ])
  rows = _rows(_stat(), candles)
  for r in rows.values():
    # baseline_n carries the countable count; value_baseline is populated.
    assert r.baseline_n == 1
    assert r.value_baseline is not None


# ===========================================================================
# Reproducibility & resolution discipline
# ===========================================================================
def test_compute_is_reproducible() -> None:
  candles = _concat([
    _make_day("2024-01-02", events={_IB_END_30: (115.0, 100.0, 100.0)}),
    _make_day("2024-01-03", events={_IB_END_30: (100.0, 80.0, 100.0)}),
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
  rows = {r.outcome: r for r in tf.results}
  assert rows["mean_extension"].total == 1
  assert rows["mean_extension"].value == pytest.approx(5.0)


def test_empty_input_yields_zero_rows() -> None:
  stat = _stat()
  table = stat.build_day_table(_empty_df())
  assert table.empty
  assert "extension" in table.columns
  assert "up_ext" in table.columns

  tf = stat.compute(_empty_df()).instruments["NQ"]["30min"]
  assert tf.total_samples == 0
  assert tf.data_range == []
  for r in tf.results:
    assert r.count == 0
    assert r.total == 0
    assert r.value == 0.0
    assert r.probability == 0.0


# ===========================================================================
# Slices
# ===========================================================================
def test_declared_slices_are_present() -> None:
  candles = _concat([
    _make_day("2024-01-02", events={_IB_END_30: (115.0, 100.0, 100.0)}),
  ])
  slices = _stat().compute(candles).instruments["NQ"]["30min"].slices
  assert set(slices) == {
    "weekday", "close", "prev_candle", "overnight", "size", "size_pct"
  }


def test_close_slice_splits_by_session_colour() -> None:
  # Two green days (close 105 > open 100) and one red (close 95 < open 100), all
  # breaking up by 5. session_close stays inside the IB so it does not itself break.
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
  green_rows = {r.outcome: r for r in groups["green"].results}
  assert green_rows["mean_extension"].value == pytest.approx(5.0)


def test_size_slice_buckets_by_ib_size() -> None:
  # Four distinct IB sizes -> quartile buckets, one day each, all breaking up.
  candles = _concat([
    _make_day("2024-01-02", ib_high=105.0, ib_low=95.0,
              events={_IB_END_30: (130.0, 100.0, 100.0)}),
    _make_day("2024-01-03", ib_high=110.0, ib_low=90.0,
              events={_IB_END_30: (130.0, 100.0, 100.0)}),
    _make_day("2024-01-04", ib_high=115.0, ib_low=85.0,
              events={_IB_END_30: (130.0, 100.0, 100.0)}),
    _make_day("2024-01-05", ib_high=120.0, ib_low=80.0,
              events={_IB_END_30: (130.0, 100.0, 100.0)}),
  ])
  groups = _stat().compute(candles).instruments["NQ"]["30min"].slices["size"].groups
  assert sum(g.total_samples for g in groups.values()) == 4


# ===========================================================================
# classify_samples
# ===========================================================================
def test_classify_samples_matches_compute_rows() -> None:
  # Mirrors test_mean_and_max_aggregation: three up-only sessions, ext 5/10/15.
  stat = _stat()
  candles = _concat([
    _make_day("2024-01-02", events={_IB_END_30: (115.0, 100.0, 100.0)}),  # 5
    _make_day("2024-01-03", events={_IB_END_30: (120.0, 100.0, 100.0)}),  # 10
    _make_day("2024-01-04", events={_IB_END_30: (125.0, 100.0, 100.0)}),  # 15
  ])
  table = stat.build_day_table(candles)
  samples = stat.classify_samples(table)
  # Four SampleRows per day (one per magnitude outcome), grouped by outcome key.
  assert [(s.date, s.condition, s.outcome, s.value) for s in samples] == [
    ("2024-01-02", "ib", "mean_extension", 5.0),
    ("2024-01-03", "ib", "mean_extension", 10.0),
    ("2024-01-04", "ib", "mean_extension", 15.0),
    ("2024-01-02", "ib", "mean_extension_pct", 0.05),
    ("2024-01-03", "ib", "mean_extension_pct", 0.10),
    ("2024-01-04", "ib", "mean_extension_pct", 0.15),
    ("2024-01-02", "ib", "max_extension", 5.0),
    ("2024-01-03", "ib", "max_extension", 10.0),
    ("2024-01-04", "ib", "max_extension", 15.0),
    ("2024-01-02", "ib", "max_extension_pct", 0.05),
    ("2024-01-03", "ib", "max_extension_pct", 0.10),
    ("2024-01-04", "ib", "max_extension_pct", 0.15),
  ]


def test_classify_samples_value_reproduces_row_aggregate() -> None:
  stat = _stat()
  candles = _concat([
    _make_day("2024-01-02", events={_IB_END_30: (115.0, 100.0, 100.0)}),  # 5
    _make_day("2024-01-03", events={_IB_END_30: (120.0, 100.0, 100.0)}),  # 10
    _make_day("2024-01-04", events={_IB_END_30: (125.0, 100.0, 100.0)}),  # 15
  ])
  table = stat.build_day_table(candles)
  samples = stat.classify_samples(table)
  rows = {r.outcome: r for r in stat.compute_rows(table)}

  mean_vals = [s.value for s in samples if s.outcome == "mean_extension"]
  assert sum(mean_vals) / len(mean_vals) == pytest.approx(rows["mean_extension"].value)
  max_vals = [s.value for s in samples if s.outcome == "max_extension"]
  assert max(max_vals) == pytest.approx(rows["max_extension"].value)

  matching = sum(1 for s in samples if s.condition == "ib" and s.outcome == "mean_extension")
  assert matching == rows["mean_extension"].count


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
  # One countable day * four magnitude outcomes.
  assert len(tf.samples) == 4


# ===========================================================================
# JSON round-trip
# ===========================================================================
def test_result_validates_and_writes(tmp_path: Path) -> None:
  candles = _concat([
    _make_day("2024-01-02", events={_IB_END_30: (115.0, 100.0, 100.0)}),
  ])
  result = _stat().compute(candles)
  out = write_results(result, results_dir=tmp_path)
  assert out.exists()
  data = json.loads(out.read_text(encoding="utf-8"))
  assert data["stat_name"] == "initial_balance_performance"
  assert data["title"]["en"] == "Initial Balance Breakout — Performance"
  StatRunResult.model_validate(data)
