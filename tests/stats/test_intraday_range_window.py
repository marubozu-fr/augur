"""Tests for stats.intraday_range_window.standard.

All data is synthetic — no real market files required. Expected values are
hand-calculated before each assertion.

RTH reference (start = 09:30 = 570 min, end = 16:15 = 975 exclusive):
  - A resolved day needs a bar at mod=570 (clean open) and a last RTH bar at
    mod >= 975 - 15 = 960.
  - Default window 09:30-10:30 covers mods 570..630 (61 bars); window_open is the
    open of the bar at mod 570.

Key property used for the baseline: on a FLAT session (every RTH bar shares the
same high / low / open), every same-length window has the identical range and
percentage range, so the random baseline provably equals the actual aggregates.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stats.base import SampleRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.intraday_range_window.standard import IntradayRangeWindow

# ---------------------------------------------------------------------------
# Minimal InstrumentConfig — does NOT depend on NQ.yaml
# ---------------------------------------------------------------------------
_NY = "America/New_York"
_RTH_SESSION = Session(start="09:30", end="16:15")
_TEST_CONFIG = InstrumentConfig(
  instrument="NQ",
  working_timezone=_NY,
  sessions={"rth": _RTH_SESSION},
  timeframes=["1min"],
  parquet_path=Path("data/NQ_1min.parquet"),
)

_RTH_START = 570   # 09:30
_RTH_LAST = 974    # 16:14 (last 1-min bar in RTH)
_RESOLVED_LAST = 960  # last bar mod must be >= this to resolve


# ---------------------------------------------------------------------------
# Synthetic day builders
# ---------------------------------------------------------------------------
def _make_day(
  date: str,
  open_: float,
  high: float,
  low: float,
  last_mod: int = _RTH_LAST,
  overrides: dict[int, tuple[float, float]] | None = None,
) -> pd.DataFrame:
  """One RTH day with a flat (high, low, open) on every bar, plus overrides.

  ``overrides`` maps a minute-of-day to a ``(high, low)`` pair placed only on
  that bar — used to inject a spike inside or outside the measured window.
  """
  base = pd.Timestamp(date, tz=_NY)
  records = []
  for mod in range(_RTH_START, last_mod + 1):
    h, m = divmod(mod, 60)
    hi, lo = high, low
    if overrides and mod in overrides:
      hi, lo = overrides[mod]
    records.append({
      "timestamp": base.replace(hour=h, minute=m, second=0, microsecond=0),
      "open": open_,
      "high": hi,
      "low": lo,
      "close": open_,
      "volume": 1000,
    })
  return pd.DataFrame(records)


def _candles(*days: pd.DataFrame) -> pd.DataFrame:
  return pd.concat(days, ignore_index=True)


def _rows_by_outcome(results: list) -> dict[str, object]:
  return {r.outcome: r for r in results}


_OUTCOME_KEYS = [
  "range_avg", "range_max", "range_min", "range_median",
  "range_pct_avg", "range_pct_max", "range_pct_min", "range_pct_median",
]


# ---------------------------------------------------------------------------
# Basic single-window computation
# ---------------------------------------------------------------------------
def test_single_flat_day_range_and_pct():
  # open=100, high=105, low=95 -> range=10, pct=10/100=0.1.
  candles = _candles(_make_day("2026-01-05", open_=100.0, high=105.0, low=95.0))
  stat = IntradayRangeWindow("NQ", _TEST_CONFIG)
  result = stat.compute(candles)

  tf = result.instruments["NQ"]["0930-1030"]
  assert tf.total_samples == 1
  by = _rows_by_outcome(tf.results)
  assert set(by) == set(_OUTCOME_KEYS)
  for agg in ("range_avg", "range_max", "range_min", "range_median"):
    assert by[agg].value == pytest.approx(10.0)
    assert by[agg].count == 1
    assert by[agg].total == 1
    assert by[agg].probability == 0.0
  for agg in ("range_pct_avg", "range_pct_max", "range_pct_min", "range_pct_median"):
    assert by[agg].value == pytest.approx(0.1)


def test_aggregates_over_multiple_days():
  # Daily ranges 10 / 20 / 30 (low fixed at 95, open 100).
  candles = _candles(
    _make_day("2026-01-05", open_=100.0, high=105.0, low=95.0),  # range 10
    _make_day("2026-01-06", open_=100.0, high=115.0, low=95.0),  # range 20
    _make_day("2026-01-07", open_=100.0, high=125.0, low=95.0),  # range 30
  )
  stat = IntradayRangeWindow("NQ", _TEST_CONFIG)
  by = _rows_by_outcome(stat.compute(candles).instruments["NQ"]["0930-1030"].results)

  assert by["range_avg"].value == pytest.approx(20.0)
  assert by["range_max"].value == pytest.approx(30.0)
  assert by["range_min"].value == pytest.approx(10.0)
  assert by["range_median"].value == pytest.approx(20.0)
  assert by["range_pct_avg"].value == pytest.approx(0.2)
  assert by["range_pct_max"].value == pytest.approx(0.3)
  assert by["range_pct_min"].value == pytest.approx(0.1)
  assert by["range_pct_median"].value == pytest.approx(0.2)
  for r in by.values():
    assert r.count == 3


def test_percentage_uses_window_open():
  # Same absolute range (20) but a higher open -> smaller percentage.
  candles = _candles(_make_day("2026-01-05", open_=200.0, high=210.0, low=190.0))
  by = _rows_by_outcome(
    IntradayRangeWindow("NQ", _TEST_CONFIG).compute(candles).instruments["NQ"]["0930-1030"].results
  )
  assert by["range_avg"].value == pytest.approx(20.0)
  assert by["range_pct_avg"].value == pytest.approx(0.1)  # 20 / 200


# ---------------------------------------------------------------------------
# Window isolation
# ---------------------------------------------------------------------------
def test_window_isolates_to_its_bars():
  # Flat range 10 inside the window; a huge spike at 12:00 (mod 720), well
  # OUTSIDE the 09:30-10:30 window, must not affect the window range.
  candles = _candles(
    _make_day(
      "2026-01-05", open_=100.0, high=105.0, low=95.0,
      overrides={720: (500.0, 95.0)},
    )
  )
  stat = IntradayRangeWindow("NQ", _TEST_CONFIG)
  by = _rows_by_outcome(stat.compute(candles).instruments["NQ"]["0930-1030"].results)
  assert by["range_max"].value == pytest.approx(10.0)
  assert by["range_pct_max"].value == pytest.approx(0.1)

  # But the spike DOES enter some sliding candidate windows (used by baseline),
  # so the day's candidate-range pool contains values above the window's 10.
  day_table = stat.build_day_table(candles)
  cand = day_table["_cand_range"].iloc[0]
  assert cand.max() > 10.0


def test_custom_window_selects_correct_bars():
  # Flat range 5 everywhere, but 10:00-11:00 (mods 600..660) widened to range 15.
  overrides = {mod: (110.0, 95.0) for mod in range(600, 661)}
  candles = _candles(
    _make_day("2026-01-05", open_=100.0, high=100.0, low=95.0, overrides=overrides)
  )
  stat = IntradayRangeWindow("NQ", _TEST_CONFIG, start_window="10:00", end_window="11:00")
  tf = stat.compute(candles).instruments["NQ"]["1000-1100"]
  by = _rows_by_outcome(tf.results)
  assert by["range_avg"].value == pytest.approx(15.0)
  assert by["range_pct_avg"].value == pytest.approx(0.15)  # 15 / 100
  assert stat.window_key == "1000-1100"
  assert "1000-1100" in stat.labels.conditions


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------
def test_baseline_equals_actual_on_flat_sessions():
  # On flat sessions every candidate window equals the actual window, so the
  # random baseline must equal the actual aggregates exactly.
  candles = _candles(
    _make_day("2026-01-05", open_=100.0, high=105.0, low=95.0),   # range 10
    _make_day("2026-01-06", open_=100.0, high=115.0, low=95.0),   # range 20
  )
  stat = IntradayRangeWindow("NQ", _TEST_CONFIG)
  by = _rows_by_outcome(stat.compute(candles).instruments["NQ"]["0930-1030"].results)
  for r in by.values():
    assert r.value_baseline is not None
    assert r.value_baseline == pytest.approx(r.value)
    assert r.baseline_n == r.total


def test_baseline_population_aligned_with_actual():
  # Custom window 10:00-11:00. The second day is resolved (clean 09:30 open + a
  # late close) but every bar inside the window is missing, so its actual range
  # is NaN. It must contribute to NEITHER the actual nor the baseline population:
  # count and baseline_n stay equal to the single valid day.
  good = _make_day("2026-01-05", open_=100.0, high=110.0, low=95.0)
  missing = _make_day("2026-01-06", open_=100.0, high=120.0, low=95.0)
  missing = missing[(missing["timestamp"].dt.hour * 60 + missing["timestamp"].dt.minute)
                    .lt(600) | (missing["timestamp"].dt.hour * 60
                                + missing["timestamp"].dt.minute).gt(660)]
  candles = _candles(good, missing)

  stat = IntradayRangeWindow("NQ", _TEST_CONFIG, start_window="10:00", end_window="11:00")
  tf = stat.compute(candles).instruments["NQ"]["1000-1100"]
  # Both days resolve, but only the first has a valid window.
  assert tf.total_samples == 2
  by = _rows_by_outcome(tf.results)
  for r in by.values():
    assert r.count == 1
    assert r.baseline_n == 1  # baseline population matches the actual population
  assert by["range_avg"].value == pytest.approx(15.0)


def test_baseline_reproducible():
  candles = _candles(
    _make_day("2026-01-05", open_=100.0, high=105.0, low=95.0, overrides={720: (300.0, 95.0)}),
    _make_day("2026-01-06", open_=100.0, high=115.0, low=95.0, overrides={800: (250.0, 95.0)}),
  )
  stat = IntradayRangeWindow("NQ", _TEST_CONFIG)
  first = stat.compute(candles).model_dump()
  second = stat.compute(candles).model_dump()
  assert first == second


# ---------------------------------------------------------------------------
# Slices
# ---------------------------------------------------------------------------
def test_weekday_slice():
  # 2026-01-05 Mon, 2026-01-12 Mon, 2026-01-06 Tue.
  candles = _candles(
    _make_day("2026-01-05", open_=100.0, high=110.0, low=95.0),  # Mon range 15
    _make_day("2026-01-12", open_=100.0, high=120.0, low=95.0),  # Mon range 25
    _make_day("2026-01-06", open_=100.0, high=105.0, low=95.0),  # Tue range 10
  )
  stat = IntradayRangeWindow("NQ", _TEST_CONFIG)
  result = stat.compute(candles)
  tf = result.instruments["NQ"]["0930-1030"]
  assert "weekday" in tf.slices

  groups = tf.slices["weekday"].groups
  assert groups["monday"].total_samples == 2
  assert groups["tuesday"].total_samples == 1

  mon = _rows_by_outcome(groups["monday"].results)
  assert mon["range_avg"].value == pytest.approx(20.0)  # (15 + 25) / 2
  assert mon["range_max"].value == pytest.approx(25.0)
  tue = _rows_by_outcome(groups["tuesday"].results)
  assert tue["range_avg"].value == pytest.approx(10.0)
  assert "weekday" in result.labels.dimensions


# ---------------------------------------------------------------------------
# Pending-sample discipline & edge cases
# ---------------------------------------------------------------------------
def test_early_close_day_excluded():
  candles = _candles(
    _make_day("2026-01-05", open_=100.0, high=105.0, low=95.0),               # resolved
    _make_day("2026-01-06", open_=100.0, high=200.0, low=95.0, last_mod=940),  # early close
  )
  stat = IntradayRangeWindow("NQ", _TEST_CONFIG)
  tf = stat.compute(candles).instruments["NQ"]["0930-1030"]
  assert tf.total_samples == 1
  by = _rows_by_outcome(tf.results)
  # Only the resolved day's range (10) counts; the early-close 105 is excluded.
  assert by["range_max"].value == pytest.approx(10.0)


def test_empty_input():
  empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
  stat = IntradayRangeWindow("NQ", _TEST_CONFIG)
  tf = stat.compute(empty).instruments["NQ"]["0930-1030"]
  assert tf.total_samples == 0
  assert tf.data_range == []
  by = _rows_by_outcome(tf.results)
  assert set(by) == set(_OUTCOME_KEYS)
  for r in by.values():
    assert r.count == 0
    assert r.value == 0.0


@pytest.mark.parametrize(
  "start,end",
  [
    ("10:30", "09:30"),  # start after end
    ("09:30", "09:30"),  # zero-length
    ("09:00", "10:30"),  # start before RTH
    ("09:30", "16:30"),  # end after RTH
  ],
)
def test_invalid_window_raises(start: str, end: str):
  with pytest.raises(ValueError):
    IntradayRangeWindow("NQ", _TEST_CONFIG, start_window=start, end_window=end)


# ---------------------------------------------------------------------------
# Schema / roundtrip
# ---------------------------------------------------------------------------
def test_write_and_reload_roundtrip(tmp_path: Path):
  candles = _candles(
    _make_day("2026-01-05", open_=100.0, high=105.0, low=95.0),
    _make_day("2026-01-06", open_=100.0, high=115.0, low=95.0),
  )
  stat = IntradayRangeWindow("NQ", _TEST_CONFIG)
  result = stat.compute(candles)
  path = write_results(result, results_dir=tmp_path)

  with open(path, encoding="utf-8") as f:
    data = json.load(f)
  reloaded = StatRunResult.model_validate(data)
  assert reloaded.stat_name == "intraday_range_window"
  assert reloaded.title.en == "Intraday Range Window"
  assert "0930-1030" in reloaded.labels.conditions
  assert set(reloaded.labels.outcomes) == set(_OUTCOME_KEYS)
  tf = reloaded.instruments["NQ"]["0930-1030"]
  assert tf.data_range == ["2026-01-05", "2026-01-06"]


# ---------------------------------------------------------------------------
# classify_samples()
# ---------------------------------------------------------------------------
_AGG_FUNCS = {"avg": np.mean, "max": np.max, "min": np.min, "median": np.median}


def test_classify_samples_exact_rows():
  # Single flat day: open=100, high=105, low=95 -> range=10, pct=0.1.
  candles = _candles(_make_day("2026-01-05", open_=100.0, high=105.0, low=95.0))
  stat = IntradayRangeWindow("NQ", _TEST_CONFIG)
  day_table = stat.build_day_table(candles)
  samples = stat.classify_samples(day_table)

  expected = [
    SampleRow(date="2026-01-05", condition="0930-1030", outcome=out, value=val)
    for out, val in (
      ("range_avg", 10.0), ("range_max", 10.0), ("range_min", 10.0), ("range_median", 10.0),
      ("range_pct_avg", 0.1), ("range_pct_max", 0.1), ("range_pct_min", 0.1), ("range_pct_median", 0.1),
    )
  ]

  assert len(samples) == len(expected)
  for got, want in zip(samples, expected):
    assert got.date == want.date
    assert got.condition == want.condition
    assert got.outcome == want.outcome
    assert got.value == pytest.approx(want.value)


def test_classify_samples_matches_compute_rows():
  # 3 days: ranges 10 / 20 / 30, pct 0.1 / 0.2 / 0.3 (see test_aggregates_over_multiple_days).
  candles = _candles(
    _make_day("2026-01-05", open_=100.0, high=105.0, low=95.0),
    _make_day("2026-01-06", open_=100.0, high=115.0, low=95.0),
    _make_day("2026-01-07", open_=100.0, high=125.0, low=95.0),
  )
  stat = IntradayRangeWindow("NQ", _TEST_CONFIG)
  day_table = stat.build_day_table(candles)
  samples = stat.classify_samples(day_table)
  rows_by_outcome = _rows_by_outcome(stat.compute_rows(day_table))

  for out_key in _OUTCOME_KEYS:
    values = [s.value for s in samples if s.outcome == out_key]
    row = rows_by_outcome[out_key]
    assert len(values) == row.total
    agg = out_key.rsplit("_", 1)[-1]
    assert _AGG_FUNCS[agg](values) == pytest.approx(row.value)


def test_classify_samples_empty_day_table():
  empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
  stat = IntradayRangeWindow("NQ", _TEST_CONFIG)
  day_table = stat.build_day_table(empty)
  assert stat.classify_samples(day_table) == []


def test_classify_samples_excludes_pending_window():
  # Second day is resolved but every bar inside the window is missing, so its
  # actual range/pct are NaN — it must NOT produce any samples.
  good = _make_day("2026-01-05", open_=100.0, high=110.0, low=95.0)
  missing = _make_day("2026-01-06", open_=100.0, high=120.0, low=95.0)
  missing = missing[(missing["timestamp"].dt.hour * 60 + missing["timestamp"].dt.minute)
                    .lt(600) | (missing["timestamp"].dt.hour * 60
                                + missing["timestamp"].dt.minute).gt(660)]
  candles = _candles(good, missing)

  stat = IntradayRangeWindow("NQ", _TEST_CONFIG, start_window="10:00", end_window="11:00")
  day_table = stat.build_day_table(candles)
  samples = stat.classify_samples(day_table)

  assert len(samples) == 8
  assert {s.date for s in samples} == {"2026-01-05"}
  assert {s.outcome for s in samples} == set(_OUTCOME_KEYS)


def test_metric_independence_from_numpy_warnings():
  # A sporadic missing minute inside the window must not raise; the window range
  # is still computed over the bars that are present.
  day = _make_day("2026-01-05", open_=100.0, high=105.0, low=95.0)
  day = day[day["timestamp"].dt.minute != 45]  # drop the 09:45 bar
  stat = IntradayRangeWindow("NQ", _TEST_CONFIG)
  by = _rows_by_outcome(stat.compute(day).instruments["NQ"]["0930-1030"].results)
  assert by["range_avg"].value == pytest.approx(10.0)
  assert np.isfinite(by["range_pct_avg"].value)
