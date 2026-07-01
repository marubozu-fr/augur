"""Cross-family spot-checks for the `agg` field on StatResultRow (issue #185).

All data is synthetic — no real market files required. Each family's
``compute_rows()`` is exercised directly against a hand-built day table (the
column shape each family's own ``build_day_table`` documents), sidestepping
1-min candle synthesis since only the `agg` field + its aggregation is under
test here — the underlying values/probabilities are already covered by each
family's dedicated test file.

Covers all four decomposable aggregation values plus the two "agg=None"
cases (probability rows, non-decomposable Pearson r rows):
  - intraday_range_window: range_avg -> "mean", range_max -> "max",
    range_min -> "min", range_median -> "median" (and the range_pct_*
    variants), all four in a single family.
  - engulfing_candles: avg_continuation -> "mean", max_continuation -> "max";
    its tier-1 frequency rows (probability channel) are agg=None.
  - session_reversal_range: mean_reversal(_pct) -> "mean",
    max_reversal(_pct) -> "max".
  - market_open_volume: the lone Pearson-r `correlation` row is agg=None
    (non-decomposable magnitude row).
"""

from pathlib import Path

import pandas as pd
import pytest

from stats.config import InstrumentConfig, Session
from stats.engulfing_candles.standard import EngulfingCandles
from stats.intraday_range_window.standard import IntradayRangeWindow
from stats.market_open_volume.standard import MarketOpenVolume
from stats.session_reversal_range.standard import SessionReversalRange

_NY = "America/New_York"
_RTH_SESSION = Session(start="09:30", end="16:15")
_TEST_CONFIG = InstrumentConfig(
  instrument="NQ",
  working_timezone=_NY,
  sessions={"rth": _RTH_SESSION},
  timeframes=["daily"],
  parquet_path=Path("data/NQ_1min.parquet"),
)


def _by_outcome(rows: list) -> dict[str, object]:
  return {r.outcome: r for r in rows}


def _by_key(rows: list) -> dict[tuple[str, str], object]:
  return {(r.condition, r.outcome): r for r in rows}


# ===========================================================================
# intraday_range_window: all four agg values in one family
# ===========================================================================


def test_intraday_range_window_agg_covers_all_four_types() -> None:
  # 4 days, range = [10, 20, 30, 100] -> mean=40, max=100, min=10, median=25
  # (median of [10,20,30,100] is the average of the two middle values 20,30).
  # range_pct = range / 100 -> [0.1, 0.2, 0.3, 1.0] -> mean=0.4, max=1.0,
  # min=0.1, median=0.25.
  day_table = pd.DataFrame({
    "range": [10.0, 20.0, 30.0, 100.0],
    "range_pct": [0.1, 0.2, 0.3, 1.0],
  })
  stat = IntradayRangeWindow("NQ", _TEST_CONFIG)
  by = _by_outcome(stat.compute_rows(day_table))

  assert by["range_avg"].agg == "mean"
  assert by["range_avg"].value == pytest.approx(40.0)
  assert by["range_max"].agg == "max"
  assert by["range_max"].value == pytest.approx(100.0)
  assert by["range_min"].agg == "min"
  assert by["range_min"].value == pytest.approx(10.0)
  assert by["range_median"].agg == "median"
  assert by["range_median"].value == pytest.approx(25.0)

  assert by["range_pct_avg"].agg == "mean"
  assert by["range_pct_avg"].value == pytest.approx(0.4)
  assert by["range_pct_max"].agg == "max"
  assert by["range_pct_max"].value == pytest.approx(1.0)
  assert by["range_pct_min"].agg == "min"
  assert by["range_pct_min"].value == pytest.approx(0.1)
  assert by["range_pct_median"].agg == "median"
  assert by["range_pct_median"].value == pytest.approx(0.25)


def test_intraday_range_window_empty_table_still_tags_agg() -> None:
  # An empty day table falls back to value=0.0 for every outcome, but the
  # `agg` field must still be populated (it is set unconditionally per outcome).
  empty = pd.DataFrame(columns=["range", "range_pct"])
  stat = IntradayRangeWindow("NQ", _TEST_CONFIG)
  by = _by_outcome(stat.compute_rows(empty))
  assert by["range_avg"].agg == "mean"
  assert by["range_max"].agg == "max"
  assert by["range_min"].agg == "min"
  assert by["range_median"].agg == "median"


# ===========================================================================
# engulfing_candles: mean / max magnitude rows + agg=None probability rows
# ===========================================================================


def test_engulfing_candles_continuation_agg_mean_and_max() -> None:
  # 3 resolved bullish engulfing patterns with cont_pct = [5, 10, 15]:
  #   avg_continuation = (5+10+15)/3 = 10.0   -> agg "mean"
  #   max_continuation = max(5,10,15) = 15.0  -> agg "max"
  # Body check: current [94, 101] (green) engulfs prior [95, 100] (red):
  # curr_top=101 >= prev_top=100 and curr_bot=94 <= prev_bot=95 -> bullish.
  day_table = pd.DataFrame({
    "session_open": [94.0, 94.0, 94.0],
    "session_close": [101.0, 101.0, 101.0],
    "prev_open": [100.0, 100.0, 100.0],
    "prev_close": [95.0, 95.0, 95.0],
    "cont_pct": [5.0, 10.0, 15.0],
    "cont_resolved": [True, True, True],
  })
  stat = EngulfingCandles("NQ", _TEST_CONFIG)
  by = _by_key(stat.compute_rows(day_table))

  assert by[("bullish", "avg_continuation")].agg == "mean"
  assert by[("bullish", "avg_continuation")].value == pytest.approx(10.0)
  assert by[("bullish", "max_continuation")].agg == "max"
  assert by[("bullish", "max_continuation")].value == pytest.approx(15.0)


def test_engulfing_candles_probability_rows_have_agg_none() -> None:
  # Same 3 days as above: all 3 are countable and all 3 are bullish.
  #   P(bullish | countable) = 3/3 = 1.0, P(bearish | countable) = 0/3 = 0.0
  day_table = pd.DataFrame({
    "session_open": [94.0, 94.0, 94.0],
    "session_close": [101.0, 101.0, 101.0],
    "prev_open": [100.0, 100.0, 100.0],
    "prev_close": [95.0, 95.0, 95.0],
    "cont_pct": [5.0, 10.0, 15.0],
    "cont_resolved": [True, True, True],
  })
  stat = EngulfingCandles("NQ", _TEST_CONFIG)
  by = _by_key(stat.compute_rows(day_table))

  bullish_freq = by[("engulfing", "bullish")]
  bearish_freq = by[("engulfing", "bearish")]
  assert bullish_freq.agg is None
  assert bullish_freq.probability == pytest.approx(1.0)
  assert bearish_freq.agg is None
  assert bearish_freq.probability == pytest.approx(0.0)


# ===========================================================================
# session_reversal_range: mean / max magnitude rows (points and pct)
# ===========================================================================


def test_session_reversal_range_agg_mean_and_max() -> None:
  # reversal = [2, 4, 6] -> mean=4.0, max=6.0
  # reversal_pct = [0.02, 0.04, 0.06] -> mean=0.04, max=0.06
  day_table = pd.DataFrame({
    "reversal": [2.0, 4.0, 6.0],
    "reversal_pct": [0.02, 0.04, 0.06],
  })
  stat = SessionReversalRange("NQ", _TEST_CONFIG)
  by = _by_outcome(stat.compute_rows(day_table))

  assert by["mean_reversal"].agg == "mean"
  assert by["mean_reversal"].value == pytest.approx(4.0)
  assert by["max_reversal"].agg == "max"
  assert by["max_reversal"].value == pytest.approx(6.0)
  assert by["mean_reversal_pct"].agg == "mean"
  assert by["mean_reversal_pct"].value == pytest.approx(0.04)
  assert by["max_reversal_pct"].agg == "max"
  assert by["max_reversal_pct"].value == pytest.approx(0.06)


# ===========================================================================
# market_open_volume: non-decomposable Pearson r row is agg=None
# ===========================================================================


def test_market_open_volume_correlation_row_has_agg_none() -> None:
  # Perfectly correlated open/rest volume -> Pearson r = 1.0 exactly.
  day_table = pd.DataFrame({
    "open_volume": [100.0, 200.0, 300.0],
    "rest_volume": [1000.0, 2000.0, 3000.0],
  })
  stat = MarketOpenVolume("NQ", _TEST_CONFIG, timeframe="1h")
  rows = stat.compute_rows(day_table)
  assert len(rows) == 1
  row = rows[0]
  assert row.condition == "any_day"
  assert row.outcome == "correlation"
  assert row.agg is None
  assert row.value == pytest.approx(1.0)
