"""Tests for stats.opening_range_indicator.standard.

All data is synthetic — no real market files required. Expected values are
hand-calculated before each assertion.

Framing (single condition ``opening_range``, magnitude stat): the opening range
is the high-low of the first N minutes of the RTH session; the remaining range is
the high-low of the rest of the session. Four value-channel outcomes are reported:
``mean_opening_range``, ``mean_remaining_range``, ``opening_range_share`` (opening
range / full session range), and ``correlation`` (Pearson r of opening vs
remaining range across days).
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.opening_range_indicator.standard import OpeningRangeIndicator

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


def _make_or_day(
  date: str,
  *,
  orb_high: float,
  orb_low: float,
  rest_high: float,
  rest_low: float,
  session_open: float | None = None,
  session_close: float | None = None,
  orb_minutes: int = 15,
  spike_mod: int | None = None,
  spike_high: float | None = None,
  last_mod: int = _RTH_LAST,
) -> pd.DataFrame:
  """One trading day of 1-min RTH bars with a controllable opening / remaining range.

  The opening window ``[09:30, 09:30 + orb_minutes)`` carries ``orb_high`` /
  ``orb_low`` as wicks; the remaining window ``[09:30 + orb_minutes, last_mod]``
  carries ``rest_high`` / ``rest_low`` as wicks. ``session_open`` and
  ``session_close`` default to the opening / remaining midpoints (kept inside the
  ranges so they do not perturb the extremes). An optional ``spike_high`` at
  ``spike_mod`` injects a single high wick at a fixed clock minute, used to test
  that the opening-window length tracks the timeframe.
  """
  base = pd.Timestamp(date, tz=_NY)
  orb_mid = (orb_high + orb_low) / 2.0
  rest_mid = (rest_high + rest_low) / 2.0
  if session_open is None:
    session_open = orb_mid
  if session_close is None:
    session_close = rest_mid

  orb_end = _RTH_START + orb_minutes
  records = []
  for mod in range(_RTH_START, last_mod + 1):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    if mod < orb_end:
      # Opening window: open on the first bar, extremes as wicks on the next two.
      o = session_open if mod == _RTH_START else orb_mid
      c = orb_mid
      hi = orb_high if mod == _RTH_START + 1 else max(o, c)
      lo = orb_low if mod == _RTH_START + 2 else min(o, c)
    else:
      # Remaining window: wick extremes, then the session close on the last bar.
      o = rest_mid
      c = rest_mid
      if mod == orb_end:
        hi, lo = rest_high, rest_mid
      elif mod == orb_end + 1:
        hi, lo = rest_mid, rest_low
      else:
        hi, lo = max(o, c), min(o, c)
      if mod == last_mod:
        c = session_close
        hi, lo = max(hi, c), min(lo, c)
    if spike_mod is not None and mod == spike_mod and spike_high is not None:
      hi = max(hi, spike_high)
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


def _stat(timeframe: str = "15min") -> OpeningRangeIndicator:
  return OpeningRangeIndicator(instrument="NQ", timeframe=timeframe, config=_TEST_CONFIG)


def _rows(stat: OpeningRangeIndicator, candles: pd.DataFrame) -> dict[str, StatResultRow]:
  result = stat.compute(candles)
  return {r.outcome: r for r in result.instruments["NQ"][stat.timeframe].results}


# A canonical 4-day set: opening_range = 10,20,30,40 and remaining_range = 20,40,60,80
# (remaining = 2 x opening every day, so the correlation is a perfect +1). Each day's
# full session range = remaining range, so opening_range_share = 0.5 on every day.
_CANON = [
  dict(date="2024-01-02", orb_high=105, orb_low=95, rest_high=110, rest_low=90),
  dict(date="2024-01-03", orb_high=110, orb_low=90, rest_high=120, rest_low=80),
  dict(date="2024-01-04", orb_high=115, orb_low=85, rest_high=130, rest_low=70),
  dict(date="2024-01-05", orb_high=120, orb_low=80, rest_high=140, rest_low=60),
]


def _canon_candles() -> pd.DataFrame:
  return _concat([_make_or_day(**spec) for spec in _CANON])


# ===========================================================================
# Core magnitude computation
# ===========================================================================
def test_mean_ranges_are_averages_over_days() -> None:
  rows = _rows(_stat(), _canon_candles())
  # opening = 10,20,30,40 -> mean 25 ; remaining = 20,40,60,80 -> mean 50.
  assert rows["mean_opening_range"].value == pytest.approx(25.0)
  assert rows["mean_remaining_range"].value == pytest.approx(50.0)


def test_opening_range_share_is_fraction_of_session_range() -> None:
  rows = _rows(_stat(), _canon_candles())
  # Every day: session range == remaining range, opening == half of it -> share 0.5.
  assert rows["opening_range_share"].value == pytest.approx(0.5)


def test_correlation_is_perfect_for_linear_data() -> None:
  rows = _rows(_stat(), _canon_candles())
  # remaining = 2 x opening on every day -> Pearson r == 1.0.
  assert rows["correlation"].value == pytest.approx(1.0)


def test_probability_channel_is_unused() -> None:
  rows = _rows(_stat(), _canon_candles())
  for r in rows.values():
    assert r.probability == 0.0
    assert r.baseline_prob == 0.0
    assert r.value is not None


def test_every_resolved_day_is_countable() -> None:
  result = _stat().compute(_canon_candles())
  tf = result.instruments["NQ"]["15min"]
  assert tf.total_samples == 4
  assert tf.data_range == ["2024-01-02", "2024-01-05"]
  for r in tf.results:
    assert r.count == 4
    assert r.total == 4


# ===========================================================================
# Opening-window length per timeframe
# ===========================================================================
def test_opening_window_length_tracks_timeframe() -> None:
  # A 09:50 (mod 590) spike to 150. With a 15-min opening window it falls in the
  # REMAINING window; with a 30-min window it is part of the OPENING range.
  day = _make_or_day(
    date="2024-01-02", orb_high=110, orb_low=90,
    rest_high=100, rest_low=100, spike_mod=590, spike_high=150,
  )
  candles = _concat([day])
  # 15-min: opening range stays 110-90 = 20; the spike lands in the remaining window.
  assert _rows(_stat("15min"), candles)["mean_opening_range"].value == pytest.approx(20.0)
  # 30-min: the window swallows the 09:50 spike, so opening range = 150-90 = 60.
  assert _rows(_stat("30min"), candles)["mean_opening_range"].value == pytest.approx(60.0)


def test_opening_range_column_is_window_width() -> None:
  table = _stat().build_day_table(_canon_candles())
  # opening_range = orb_high - orb_low = 10, 20, 30, 40 across the canonical days.
  assert list(table["opening_range"]) == [10.0, 20.0, 30.0, 40.0]


# ===========================================================================
# Baseline (permutation null for the correlation)
# ===========================================================================
def test_baseline_breaks_correlation() -> None:
  stat = _stat()
  table = stat.build_day_table(_canon_candles())
  bl = {r.outcome: r for r in stat.baseline_rows(table, seed=42)}
  # Permuting remaining_range against opening_range destroys the perfect pairing.
  assert bl["correlation"].value < 1.0


def test_baseline_leaves_means_and_share_invariant() -> None:
  stat = _stat()
  table = stat.build_day_table(_canon_candles())
  bl = {r.outcome: r for r in stat.baseline_rows(table, seed=42)}
  # A permutation of remaining_range cannot change its mean, the opening mean, or
  # the per-day opening/session share.
  assert bl["mean_opening_range"].value == pytest.approx(25.0)
  assert bl["mean_remaining_range"].value == pytest.approx(50.0)
  assert bl["opening_range_share"].value == pytest.approx(0.5)


def test_baseline_merged_into_result_rows() -> None:
  rows = _rows(_stat(), _canon_candles())
  for r in rows.values():
    assert r.baseline_n == 4
    assert r.value_baseline is not None


def test_baseline_is_deterministic_for_fixed_seed() -> None:
  stat = _stat()
  table = stat.build_day_table(_canon_candles())
  a = stat.baseline_rows(table, seed=42)
  b = stat.baseline_rows(table, seed=42)
  assert [(r.outcome, r.value) for r in a] == [(r.outcome, r.value) for r in b]


def test_correlation_undefined_for_single_day_is_zero() -> None:
  # A single day gives no variance / pairing, so the correlation is reported as 0.0.
  candles = _concat([
    _make_or_day(date="2024-01-02", orb_high=110, orb_low=90, rest_high=120, rest_low=80),
  ])
  rows = _rows(_stat(), candles)
  assert rows["correlation"].value == 0.0
  assert rows["mean_opening_range"].value == pytest.approx(20.0)


# ===========================================================================
# Pending / resolution discipline
# ===========================================================================
def test_unresolved_day_is_excluded() -> None:
  candles = _concat([
    _make_or_day(date="2024-01-02", orb_high=110, orb_low=90, rest_high=120, rest_low=80),
    _make_truncated_day("2024-01-03"),
  ])
  result = _stat().compute(candles)
  # Only the resolved day counts; the truncated day is dropped entirely.
  assert result.instruments["NQ"]["15min"].total_samples == 1
  rows = {r.outcome: r for r in result.instruments["NQ"]["15min"].results}
  assert rows["mean_opening_range"].total == 1


def test_empty_input_yields_zero_rows() -> None:
  result = _stat().compute(_empty_df())
  tf = result.instruments["NQ"]["15min"]
  assert tf.total_samples == 0
  assert tf.data_range == []
  for r in tf.results:
    assert r.count == 0
    assert r.total == 0
    assert r.value == 0.0


# ===========================================================================
# Slices
# ===========================================================================
def test_declared_slices_are_present() -> None:
  result = _stat().compute(_canon_candles())
  slices = result.instruments["NQ"]["15min"].slices
  assert set(slices) == {"weekday", "close", "size"}


def test_close_slice_splits_by_session_colour() -> None:
  # Two green days (close > open) and one red day (close < open).
  candles = _concat([
    _make_or_day(date="2024-01-02", orb_high=110, orb_low=90, rest_high=120,
                 rest_low=80, session_open=100, session_close=105),
    _make_or_day(date="2024-01-03", orb_high=110, orb_low=90, rest_high=120,
                 rest_low=80, session_open=100, session_close=104),
    _make_or_day(date="2024-01-04", orb_high=110, orb_low=90, rest_high=120,
                 rest_low=80, session_open=100, session_close=95),
  ])
  groups = _stat().compute(candles).instruments["NQ"]["15min"].slices["close"].groups
  assert groups["green"].total_samples == 2
  assert groups["red"].total_samples == 1


def test_size_slice_buckets_by_opening_range() -> None:
  # Four distinct opening ranges -> quartile buckets, one day each.
  groups = _stat().compute(_canon_candles()).instruments["NQ"]["15min"].slices["size"].groups
  assert sum(g.total_samples for g in groups.values()) == 4


# ===========================================================================
# Validation / errors
# ===========================================================================
def test_invalid_timeframe_raises() -> None:
  with pytest.raises(ValueError, match="Unsupported timeframe"):
    _stat(timeframe="7min")


def test_result_validates_and_writes(tmp_path: Path) -> None:
  result = _stat().compute(_canon_candles())
  out = write_results(result, results_dir=tmp_path)
  assert out.exists()
  data = json.loads(out.read_text(encoding="utf-8"))
  assert data["stat_name"] == "opening_range_indicator"
  assert data["title"]["en"] == "Opening Range Indicator"
  # Re-validate the written payload through the Pydantic model.
  StatRunResult.model_validate(data)
