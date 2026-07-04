"""Tests for stats.opening_candle.extreme_reclaim.

All data is synthetic — no real market files required. Each day's opening candle
extremes (opening_high / opening_low) and the 1-min highs/lows of the outcome
window are injected explicitly, so every reclaim outcome is hand-calculated
before the assertion.

The v1 base-stat machinery (resolution filter, slices, day-table construction) is
inherited and covered by the v1 suite; here we assert only the 5-way reclaim
partition, its ordering, doji exclusion, marubozu handling, and the baseline.
"""

from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.opening_candle.extreme_reclaim import OpeningCandleExtremeReclaim

# ---------------------------------------------------------------------------
# Config + timing constants (mirror the v1/v2 test suites)
# ---------------------------------------------------------------------------
_NY = "America/New_York"
_TEST_CONFIG = InstrumentConfig(
  instrument="NQ",
  working_timezone=_NY,
  sessions={"rth": Session(start="09:30", end="16:15")},
  timeframes=["15min", "30min", "1h"],
  parquet_path=Path("data/NQ_1min.parquet"),
)

_RTH_START = 570  # 09:30
_RTH_END = 975    # 16:15 (exclusive upper bound of RTH bars)


# ---------------------------------------------------------------------------
# Synthetic builder with fully controllable opening extremes + outcome bars
# ---------------------------------------------------------------------------
def _bar(base: pd.Timestamp, mod: int, o: float, high: float, low: float, c: float) -> dict:
  hh, mm = divmod(mod, 60)
  return {
    "timestamp": base.replace(hour=hh, minute=mm, second=0, microsecond=0),
    "open": o,
    "high": high,
    "low": low,
    "close": c,
    "volume": 1000,
  }


def _make_day(
  date: str,
  opening_open: float,
  opening_close: float,
  opening_high: float,
  opening_low: float,
  outcome_bars: list[tuple[int, float, float]],
  tf_minutes: int = 15,
) -> pd.DataFrame:
  """One trading day with a controlled opening candle and explicit outcome bars.

  Args:
    date: "YYYY-MM-DD".
    opening_open: open of the opening-candle open bar (= session open).
    opening_close: close of the opening candle's last bar.
    opening_high / opening_low: wick extremes of the opening candle window.
    outcome_bars: list of (mod, high, low) for bars in the outcome window
      [candle_open + tf, 16:15). At least one must sit at mod >= 960 (16:00) so
      the day resolves; its close becomes the session close.
    tf_minutes: 15, 30, or 60.
  """
  base = pd.Timestamp(date, tz=_NY)
  candle_open = (_RTH_START // tf_minutes) * tf_minutes
  oc_last = candle_open + tf_minutes - 1

  # Opening candle: the candle-open bar carries the wick extremes, the last bar
  # fixes the close. For 1h the candle opens at 09:00 (540) — before the RTH
  # open — so v1's resolution filter also needs a clean session-open bar exactly
  # at 09:30 (570); emit one inside the candle window (body-ranged, so it never
  # overrides the wick extremes).
  body_hi = max(opening_open, opening_close)
  body_lo = min(opening_open, opening_close)
  records = [_bar(base, candle_open, opening_open, opening_high, opening_low, opening_open)]
  if candle_open != _RTH_START:
    records.append(_bar(base, _RTH_START, opening_open, body_hi, body_lo, opening_open))
  records.append(_bar(base, oc_last, opening_open, body_hi, body_lo, opening_close))
  for mod, high, low in outcome_bars:
    mid = (high + low) / 2
    records.append(_bar(base, mod, mid, high, low, mid))
  return pd.DataFrame(records)


def _candles(days: list[pd.DataFrame]) -> pd.DataFrame:
  return pd.concat(days, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def _stat(tf: str = "15min") -> OpeningCandleExtremeReclaim:
  return OpeningCandleExtremeReclaim(instrument="NQ", timeframe=tf, config=_TEST_CONFIG)


def _empty_df() -> pd.DataFrame:
  df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
  df["timestamp"] = pd.array([], dtype="datetime64[ns, America/New_York]")
  return df


def _outcome_for(day: pd.DataFrame) -> str:
  """Build the day table for a single day and return its classified outcome."""
  table = _stat().build_day_table(day)
  assert len(table) == 1
  return table.iloc[0]["outcome"]


# A default resolving tail bar that touches neither extreme (opening 100/95..110).
def _neutral_tail() -> tuple[int, float, float]:
  return (970, 108, 96)


# ---------------------------------------------------------------------------
# Outcome classification (green opening candle: open 100, close 105, hi 110, lo 95)
# ---------------------------------------------------------------------------
_G = dict(opening_open=100.0, opening_close=105.0, opening_high=110.0, opening_low=95.0)


def test_high_only():
  # High (110) reclaimed at 600, low (95) never reached.
  day = _make_day("2020-01-06", **_G, outcome_bars=[(600, 111, 100), (970, 108, 101)])
  assert _outcome_for(day) == "high_only"


def test_low_only():
  # Low (95) reclaimed at 600, high (110) never reached.
  day = _make_day("2020-01-06", **_G, outcome_bars=[(600, 108, 94), (970, 109, 96)])
  assert _outcome_for(day) == "low_only"


def test_high_first():
  # Both reclaimed; high touched at 600 (index 3), low at 700 (index 7).
  day = _make_day("2020-01-06", **_G, outcome_bars=[(600, 111, 100), (700, 105, 94), (970, 108, 100)])
  assert _outcome_for(day) == "high_first"


def test_low_first():
  # Both reclaimed; low touched at 600 (index 2), high at 700 (index 10).
  day = _make_day("2020-01-06", **_G, outcome_bars=[(600, 105, 94), (700, 111, 100), (970, 108, 100)])
  assert _outcome_for(day) == "low_first"


def test_neither():
  # Neither extreme reached (all highs < 110, all lows > 95).
  day = _make_day("2020-01-06", **_G, outcome_bars=[(600, 108, 96), (970, 109, 97)])
  assert _outcome_for(day) == "neither"


def test_same_bar_double_touch_is_high_first():
  # One bar spans both extremes -> first_touch "same" -> high_first tiebreak.
  day = _make_day("2020-01-06", **_G, outcome_bars=[(600, 111, 94), (970, 108, 100)])
  table = _stat().build_day_table(day)
  assert table.iloc[0]["first_touch"] == "same"
  assert table.iloc[0]["outcome"] == "high_first"


def test_no_outcome_bars_is_neither():
  # A resolving day whose only outcome bar touches nothing is classified neither.
  day = _make_day("2020-01-06", **_G, outcome_bars=[_neutral_tail()])
  assert _outcome_for(day) == "neither"


# ---------------------------------------------------------------------------
# Red candle direction (mirrored): open 105, close 100, hi 110, lo 95
# ---------------------------------------------------------------------------
_R = dict(opening_open=105.0, opening_close=100.0, opening_high=110.0, opening_low=95.0)


def test_red_high_only():
  day = _make_day("2020-01-06", **_R, outcome_bars=[(600, 111, 100), (970, 108, 101)])
  table = _stat().build_day_table(day)
  assert not table.iloc[0]["opening_green"]  # red condition
  assert table.iloc[0]["outcome"] == "high_only"


def test_red_low_first():
  day = _make_day("2020-01-06", **_R, outcome_bars=[(600, 105, 94), (700, 111, 100), (970, 108, 100)])
  table = _stat().build_day_table(day)
  assert not table.iloc[0]["opening_green"]
  assert table.iloc[0]["outcome"] == "low_first"


# ---------------------------------------------------------------------------
# Edge cases: doji + marubozu
# ---------------------------------------------------------------------------
def test_doji_excluded():
  # opening_close == opening_open -> no direction -> dropped from the day table.
  doji = _make_day(
    "2020-01-06",
    opening_open=100.0,
    opening_close=100.0,
    opening_high=110.0,
    opening_low=95.0,
    outcome_bars=[(600, 111, 94), (970, 108, 100)],
  )
  green = _make_day("2020-01-07", **_G, outcome_bars=[(600, 111, 100), (970, 108, 101)])
  table = _stat().build_day_table(_candles([doji, green]))
  assert len(table) == 1  # doji dropped
  assert table.index[0] == pd.Timestamp("2020-01-07", tz=_NY)


def test_marubozu_high_reclaimed_not_excluded():
  # Green marubozu: close == high (opening_high == opening_close), no upper wick.
  # The first outcome bar reaching that level trivially reclaims the high.
  maru = _make_day(
    "2020-01-06",
    opening_open=100.0,
    opening_close=110.0,
    opening_high=110.0,  # == close, no wick above the body
    opening_low=95.0,
    outcome_bars=[(600, 110, 105), (970, 108, 106)],
  )
  table = _stat().build_day_table(maru)
  assert len(table) == 1  # not excluded
  assert bool(table.iloc[0]["reclaims_high"])
  assert table.iloc[0]["outcome"] in ("high_only", "high_first")


# ---------------------------------------------------------------------------
# Partition: the five outcome probabilities sum to 1.0 within each condition
# ---------------------------------------------------------------------------
def test_partition_sums_to_one_per_condition():
  days = [
    _make_day("2020-01-06", **_G, outcome_bars=[(600, 111, 100), (970, 108, 101)]),   # high_only
    _make_day("2020-01-07", **_G, outcome_bars=[(600, 108, 94), (970, 109, 96)]),     # low_only
    _make_day("2020-01-08", **_G, outcome_bars=[(600, 111, 100), (700, 105, 94), (970, 108, 100)]),  # high_first
    _make_day("2020-01-09", **_G, outcome_bars=[(600, 108, 96), (970, 109, 97)]),     # neither
    _make_day("2020-01-13", **_R, outcome_bars=[(600, 105, 94), (700, 111, 100), (970, 108, 100)]),  # red low_first
    _make_day("2020-01-14", **_R, outcome_bars=[(600, 111, 100), (970, 108, 101)]),   # red high_only
  ]
  stat = _stat()
  table = stat.build_day_table(_candles(days))
  rows = stat.compute_rows(table)
  for cond in ("green_open", "red_open"):
    cond_rows = [r for r in rows if r.condition == cond]
    assert len(cond_rows) == 5
    total = cond_rows[0].total
    assert total > 0
    assert sum(r.count for r in cond_rows) == total
    assert sum(r.probability for r in cond_rows) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Baseline: two independent fair coins -> 25/25/12.5/12.5/25 %
# ---------------------------------------------------------------------------
def test_baseline_distribution():
  # Build a day table directly with many green days; the baseline ignores the
  # actual reclaims (pure coin flips), so only the row count and opening_green
  # matter here.
  n = 8000
  idx = pd.date_range("2005-01-03", periods=n, freq="D", tz=_NY)
  table = pd.DataFrame({"opening_green": [True] * n}, index=idx)
  rows = _stat().baseline_rows(table, seed=42)
  probs = {r.outcome: r.probability for r in rows if r.condition == "green_open"}
  assert probs["high_only"] == pytest.approx(0.25, abs=0.02)
  assert probs["low_only"] == pytest.approx(0.25, abs=0.02)
  assert probs["high_first"] == pytest.approx(0.125, abs=0.02)
  assert probs["low_first"] == pytest.approx(0.125, abs=0.02)
  assert probs["neither"] == pytest.approx(0.25, abs=0.02)
  assert sum(probs.values()) == pytest.approx(1.0)


def test_baseline_deterministic():
  n = 500
  idx = pd.date_range("2005-01-03", periods=n, freq="D", tz=_NY)
  table = pd.DataFrame({"opening_green": [True] * n}, index=idx)
  stat = _stat()
  a = stat.baseline_rows(table, seed=42)
  b = stat.baseline_rows(table, seed=42)
  assert [(r.outcome, r.count) for r in a] == [(r.outcome, r.count) for r in b]


# ---------------------------------------------------------------------------
# Samples + base-stat consistency
# ---------------------------------------------------------------------------
def test_samples_match_rows():
  days = [
    _make_day("2020-01-06", **_G, outcome_bars=[(600, 111, 100), (970, 108, 101)]),   # high_only
    _make_day("2020-01-07", **_G, outcome_bars=[(600, 108, 94), (970, 109, 96)]),     # low_only
    _make_day("2020-01-08", **_R, outcome_bars=[(600, 105, 94), (700, 111, 100), (970, 108, 100)]),  # red low_first
  ]
  stat = _stat()
  table = stat.build_day_table(_candles(days))
  samples = stat.classify_samples(table)
  assert len(samples) == 3
  by_date = {s.date: (s.condition, s.outcome) for s in samples}
  assert by_date["2020-01-06"] == ("green_open", "high_only")
  assert by_date["2020-01-07"] == ("green_open", "low_only")
  assert by_date["2020-01-08"] == ("red_open", "low_first")

  # Sample counts reconcile with the aggregated rows.
  rows = stat.compute_rows(table)
  for r in rows:
    n = sum(1 for s in samples if s.condition == r.condition and s.outcome == r.outcome)
    assert n == r.count


def test_inherits_v1_slices():
  # Slices are inherited verbatim from v1 (weekday, close location, size bucket).
  from stats.opening_candle.standard import OpeningCandleContinuation

  assert OpeningCandleExtremeReclaim.slices is OpeningCandleContinuation.slices
  names = {getattr(s, "name", s) for s in _stat().slices}
  assert names == {"weekday", "close", "size"}


# ---------------------------------------------------------------------------
# Empty / degenerate inputs
# ---------------------------------------------------------------------------
def test_empty_dataframe():
  stat = _stat()
  table = stat.build_day_table(_empty_df())
  assert table.empty
  assert stat.classify_samples(table) == []
  rows = stat.compute_rows(table, baseline_rows=stat.baseline_rows(table, seed=42))
  assert all(r.count == 0 and r.total == 0 for r in rows)


def test_single_day():
  day = _make_day("2020-01-06", **_G, outcome_bars=[(600, 111, 100), (970, 108, 101)])
  result = _stat().compute(_candles([day]))
  tf = result.instruments["NQ"]["15min"]
  assert tf.total_samples == 1
  green = [r for r in tf.results if r.condition == "green_open"]
  assert sum(r.count for r in green) == 1
  assert next(r for r in green if r.outcome == "high_only").count == 1


def test_all_doji_yields_no_samples():
  days = [
    _make_day(
      f"2020-01-0{d}",
      opening_open=100.0,
      opening_close=100.0,
      opening_high=110.0,
      opening_low=95.0,
      outcome_bars=[(600, 111, 94), (970, 108, 100)],
    )
    for d in (6, 7, 8)
  ]
  stat = _stat()
  table = stat.build_day_table(_candles(days))
  assert table.empty
  assert stat.classify_samples(table) == []


# ---------------------------------------------------------------------------
# Timeframe smoke tests — 30min and 1h resolve and classify like 15min.
#
# 1h is the important case: candle_open_min = 540 (09:00) not 570, so the
# outcome window starts at 600 and v1's resolution needs a distinct 09:30 bar.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tf", ["30min", "1h"])
def test_timeframe_high_first(tf):
  tf_min = {"30min": 30, "1h": 60}[tf]
  # Both extremes reclaimed, high (110) first at 605, low (95) at 700.
  day = _make_day(
    "2020-01-06", **_G, outcome_bars=[(605, 111, 100), (700, 105, 94), (970, 108, 100)], tf_minutes=tf_min
  )
  table = _stat(tf).build_day_table(day)
  assert len(table) == 1  # resolves despite the 09:00 candle open (1h)
  assert table.iloc[0]["outcome"] == "high_first"


def test_1h_high_only_and_neither():
  # 1h opening candle opens at 09:00 (540); the outcome window is [600, 975).
  stat = _stat("1h")
  high_only = _make_day(
    "2020-01-06", **_G, outcome_bars=[(600, 111, 100), (970, 108, 101)], tf_minutes=60
  )
  neither = _make_day(
    "2020-01-07", **_G, outcome_bars=[(600, 108, 96), (970, 109, 97)], tf_minutes=60
  )
  table = stat.build_day_table(_candles([high_only, neither]))
  assert len(table) == 2
  outcomes = dict(zip(table.index.strftime("%Y-%m-%d"), table["outcome"]))
  assert outcomes["2020-01-06"] == "high_only"
  assert outcomes["2020-01-07"] == "neither"


# ---------------------------------------------------------------------------
# End-to-end: compute() produces a valid, writable result
# ---------------------------------------------------------------------------
def test_compute_result_validates_and_writes(tmp_path):
  days = [
    _make_day("2020-01-06", **_G, outcome_bars=[(600, 111, 100), (970, 108, 101)]),
    _make_day("2020-01-07", **_R, outcome_bars=[(600, 105, 94), (700, 111, 100), (970, 108, 100)]),
  ]
  result = _stat().compute(_candles(days))
  assert isinstance(result, StatRunResult)
  assert result.stat_name == "opening_candle_extreme_reclaim"
  # Round-trips through Pydantic validation on write.
  out = write_results(result, results_dir=tmp_path)
  loaded = StatRunResult.model_validate_json(Path(out).read_text(encoding="utf-8"))
  assert loaded.stat_name == "opening_candle_extreme_reclaim"
  assert set(loaded.labels.outcomes) == {
    "high_only", "low_only", "high_first", "low_first", "neither"
  }
