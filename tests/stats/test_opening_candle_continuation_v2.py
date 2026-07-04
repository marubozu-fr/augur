"""Tests for stats.opening_candle.standard_v2 (tradable layer).

All data is synthetic — no real market files required. Expected tradable values
(MFE, MAE, TP-ladder probabilities, conditional MAE before first touch) are
hand-calculated before each assertion.

The base stat (probability rows, slices, baseline) is inherited verbatim from v1
and covered by tests/test_opening_candle_continuation.py; here we assert only the
v2 additions plus a base-stat consistency check.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.opening_candle.standard import OpeningCandleContinuation
from stats.opening_candle.standard_v2 import OpeningCandleContinuationV2

# ---------------------------------------------------------------------------
# Config + timing constants (mirror the v1 test suite)
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
# Synthetic builder with fully controllable outcome-window bars
#
# Unlike the v1 builder (flat filler bars), this lets each test inject the exact
# 1-min highs/lows of the outcome window so MFE/MAE/first-touch are deterministic.
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


def _make_v2_day(
  date: str,
  session_open: float,
  opening_close: float,
  outcome_bars: list[tuple[int, float, float, float]],
  tf_minutes: int = 15,
) -> pd.DataFrame:
  """One trading day with a controlled opening candle and explicit outcome bars.

  Args:
    date: "YYYY-MM-DD".
    session_open: open of the grid-aligned opening-candle open bar (= session open).
    opening_close: close of the opening candle's last bar (= the anchor).
    outcome_bars: list of (mod, high, low, close) for bars in the outcome window
      [candle_open + tf, 16:15). The highest-mod bar's close becomes the session
      close and must be at mod >= 960 (16:00) so the day resolves.
    tf_minutes: 15, 30, or 60.
  """
  base = pd.Timestamp(date, tz=_NY)
  candle_open = (_RTH_START // tf_minutes) * tf_minutes
  oc_last = candle_open + tf_minutes - 1

  hi = max(session_open, opening_close) + 0.25
  lo = min(session_open, opening_close) - 0.25
  records = [
    # opening-candle open bar (also the session open at 09:30 for 15/30min)
    _bar(base, candle_open, session_open, hi, lo, session_open),
    # opening-candle close bar (fixes the anchor = opening_close)
    _bar(base, oc_last, session_open, hi, lo, opening_close),
  ]
  for mod, high, low, c in outcome_bars:
    records.append(_bar(base, mod, c, high, low, c))
  return pd.DataFrame(records)


def _candles(days: list[pd.DataFrame]) -> pd.DataFrame:
  return (
    pd.concat(days, ignore_index=True)
    .sort_values("timestamp")
    .reset_index(drop=True)
  )


def _stat(tf: str = "15min") -> OpeningCandleContinuationV2:
  return OpeningCandleContinuationV2(instrument="NQ", timeframe=tf, config=_TEST_CONFIG)


def _empty_df() -> pd.DataFrame:
  df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
  df["timestamp"] = pd.array([], dtype="datetime64[ns, America/New_York]")
  return df


# ---------------------------------------------------------------------------
# Shared 4-day GREEN dataset (anchor = 100 on every day, hand-calculated below)
#
#   day  outcome bars (mod, high, low, close)                    mfe  mae  rem  range
#   G1   (600,110,98,108) (700,130,90,120) (970,106,100,104)      30   10    4    40
#   G2   (600,120,95,118) (900,140,100,138) (970,130,120,128)     40    5   28    45
#   G3   (600,108,80,82)  (970,100,70,72)                          8   30  -28    38
#   G4   (600,160,99,150) (970,155,100,152)                       60    1   52    61
#
# session_open = 95, opening_close = 100  => green, anchor = 100
# mfe (green) = max(high) - 100 ; mae (green) = 100 - min(low)
# remaining   = session_close - 100
# ---------------------------------------------------------------------------

_GREEN_DAYS = [
  _make_v2_day("2024-01-02", 95.0, 100.0,
               [(600, 110, 98, 108), (700, 130, 90, 120), (970, 106, 100, 104)]),
  _make_v2_day("2024-01-03", 95.0, 100.0,
               [(600, 120, 95, 118), (900, 140, 100, 138), (970, 130, 120, 128)]),
  _make_v2_day("2024-01-04", 95.0, 100.0,
               [(600, 108, 80, 82), (970, 100, 70, 72)]),
  _make_v2_day("2024-01-05", 95.0, 100.0,
               [(600, 160, 99, 150), (970, 155, 100, 152)]),
]


def _td(tf) -> dict:
  """Dump a TimeframeResult's tradable layer to plain dicts for assertions."""
  return {k: v.model_dump() for k, v in (tf.tradable or {}).items()}


def _green_tradable() -> dict:
  stat = _stat()
  tf = stat.compute(_candles(_GREEN_DAYS)).instruments["NQ"]["15min"]
  return _td(tf)


# ===========================================================================
# 1. build_day_table enrichment: anchor/close/mfe/mae/is_doji
# ===========================================================================

def test_build_day_table_tradable_columns_present() -> None:
  stat = _stat()
  day = stat.build_day_table(_candles(_GREEN_DAYS))
  required = {"anchor_price", "close_price", "is_doji", "mfe_pts", "mae_pts"}
  assert required.issubset(set(day.columns))


def test_build_day_table_mfe_mae_green() -> None:
  """Green: mfe = max(high) - anchor ; mae = anchor - min(low)."""
  stat = _stat()
  day = stat.build_day_table(_candles(_GREEN_DAYS)).sort_index()

  assert list(day["anchor_price"]) == [100.0, 100.0, 100.0, 100.0]
  assert list(day["close_price"]) == [104.0, 128.0, 72.0, 152.0]
  assert list(day["mfe_pts"]) == pytest.approx([30.0, 40.0, 8.0, 60.0])
  assert list(day["mae_pts"]) == pytest.approx([10.0, 5.0, 30.0, 1.0])


def test_build_day_table_mfe_mae_red_direction() -> None:
  """Red: favorable = down, so mfe = anchor - min(low), mae = max(high) - anchor."""
  # session_open=105, opening_close=100 => red, anchor=100
  red = _make_v2_day("2024-02-01", 105.0, 100.0,
                     [(600, 102, 88, 90), (970, 101, 85, 86)])
  stat = _stat()
  day = stat.build_day_table(_candles([red])).sort_index()

  assert bool(day["opening_green"].iloc[0]) is False
  # mfe (red) = 100 - 85 = 15 ; mae (red) = 102 - 100 = 2
  assert day["mfe_pts"].iloc[0] == pytest.approx(15.0)
  assert day["mae_pts"].iloc[0] == pytest.approx(2.0)


# ===========================================================================
# 2. Block 1 — tradable residual
# ===========================================================================

def test_residual_block_green() -> None:
  """remaining = [4, 28, -28, 52] on anchor 100 across the 4 green days."""
  block = _green_tradable()["green_open"]
  assert block["n"] == 4
  assert block["win_rate"] == pytest.approx(3 / 4)
  assert block["remaining_mean_pts"] == pytest.approx(14.0)
  assert block["remaining_median_pts"] == pytest.approx(16.0)  # median([-28,4,28,52])
  assert block["remaining_mean_pct"] == pytest.approx(0.14)    # mean of remaining/100


def test_residual_block_red_signs() -> None:
  """Red win = price falls below anchor: remaining = anchor - close."""
  # R1 close 86 -> remaining 14 (win) ; R2 close 108 -> remaining -8 (loss)
  r1 = _make_v2_day("2024-02-01", 105.0, 100.0, [(600, 102, 88, 90), (970, 101, 85, 86)])
  r2 = _make_v2_day("2024-02-02", 105.0, 100.0, [(600, 112, 99, 108), (970, 110, 100, 108)])
  stat = _stat()
  tf = stat.compute(_candles([r1, r2])).instruments["NQ"]["15min"]
  block = _td(tf)["red_open"]
  assert block["n"] == 2
  assert block["win_rate"] == pytest.approx(0.5)
  assert block["remaining_mean_pts"] == pytest.approx((14 + -8) / 2)


# ===========================================================================
# 3. Block 2 — adaptive TP ladder
# ===========================================================================

def test_tp_ladder_levels_and_probs_green() -> None:
  """median_range = median([40,45,38,61]) = 42.5.

  Levels = 42.5 * [0.25,0.5,0.75,1.0,1.5] = [10.625,21.25,31.875,42.5,63.75].
  mfe = [30,40,8,60]; P(mfe>=level) = [0.75,0.75,0.5,0.25,0.0].
  """
  ladder = _green_tradable()["green_open"]["tp_ladder"]
  assert [lvl["level_x_range"] for lvl in ladder] == [0.25, 0.5, 0.75, 1.0, 1.5]
  assert [lvl["level_pts"] for lvl in ladder] == pytest.approx(
    [10.625, 21.25, 31.875, 42.5, 63.75]
  )
  assert [lvl["prob"] for lvl in ladder] == pytest.approx([0.75, 0.75, 0.5, 0.25, 0.0])
  assert all(lvl["n"] == 4 for lvl in ladder)


# ===========================================================================
# 4. Block 3 — MAE percentiles (points)
# ===========================================================================

def test_mae_percentiles_green() -> None:
  """mae = [10,5,30,1] -> sorted [1,5,10,30]; numpy linear interpolation."""
  mae = _green_tradable()["green_open"]["mae"]
  assert mae["p50"] == pytest.approx(7.5)
  assert mae["p75"] == pytest.approx(15.0)
  assert mae["p90"] == pytest.approx(24.0)


# ===========================================================================
# 5. Block 3 — conditional MAE before first touch (fraction of anchor)
# ===========================================================================

def test_conditional_mae_before_first_touch_green() -> None:
  """Per level, among days reaching it, MAE up to & incl the first touching bar.

  anchor = 100, so pct = mae_before / 100.

  Level 10.625 (touch high>=110.625): G1 idx1 minlow(98,90)=90 ->0.10;
    G2 idx0 low95 ->0.05; G4 idx0 low99 ->0.01  => p50=0.05, p75=0.075
  Level 31.875 (touch high>=131.875): G2 idx1 minlow(95,100)=95 ->0.05;
    G4 idx0 low99 ->0.01                          => p50=0.03, p75=0.04
  Level 42.5 (touch high>=142.5): only G4 idx0 low99 ->0.01 => p50=p75=0.01
  Level 63.75: no day reaches it                  => p50=p75=0.0
  """
  cond = _green_tradable()["green_open"]["mae"]["conditional"]
  by_level = {round(c["level_pts"], 3): c for c in cond}

  c1 = by_level[10.625]
  assert c1["mae_pct_p50"] == pytest.approx(0.05)
  assert c1["mae_pct_p75"] == pytest.approx(0.075)

  c3 = by_level[31.875]
  assert c3["mae_pct_p50"] == pytest.approx(0.03)
  assert c3["mae_pct_p75"] == pytest.approx(0.04)

  c4 = by_level[42.5]
  assert c4["mae_pct_p50"] == pytest.approx(0.01)
  assert c4["mae_pct_p75"] == pytest.approx(0.01)

  c5 = by_level[63.75]
  assert c5["mae_pct_p50"] == 0.0
  assert c5["mae_pct_p75"] == 0.0


def test_conditional_mae_covers_every_tp_level() -> None:
  """The MAE conditional must have one entry per TP level (product rule)."""
  block = _green_tradable()["green_open"]
  tp_levels = [lvl["level_pts"] for lvl in block["tp_ladder"]]
  cond_levels = [c["level_pts"] for c in block["mae"]["conditional"]]
  assert cond_levels == pytest.approx(tp_levels)


# ===========================================================================
# 6. Doji exclusion + excluded_days
# ===========================================================================

def test_doji_excluded_from_tradable_but_in_base_stat() -> None:
  """A doji (opening_close == opening_open) counts for the base stat (v1 treats
  '>=' as green) but is excluded from every tradable metric."""
  doji = _make_v2_day("2024-03-15", 100.0, 100.0,
                      [(600, 130, 90, 120), (970, 125, 95, 120)])
  stat = _stat()
  days = _GREEN_DAYS + [doji]
  tf = stat.compute(_candles(days)).instruments["NQ"]["15min"]

  # Base stat sees 5 green-open days (4 real + doji).
  green_row = next(
    r for r in tf.results if r.condition == "green_open" and r.outcome == "green_close"
  )
  assert green_row.total == 5

  # Tradable still only counts the 4 non-doji green days.
  block = _td(tf)["green_open"]
  assert block["n"] == 4
  assert "2024-03-15" in block["meta"]["excluded_days"]


def test_excluded_days_list_contents() -> None:
  """excluded_days lists exactly the doji dates for that condition, sorted."""
  doji_a = _make_v2_day("2024-03-15", 100.0, 100.0, [(600, 110, 90, 105), (970, 108, 95, 105)])
  doji_b = _make_v2_day("2024-01-01", 100.0, 100.0, [(600, 110, 90, 105), (970, 108, 95, 105)])
  stat = _stat()
  tf = stat.compute(_candles(_GREEN_DAYS + [doji_a, doji_b])).instruments["NQ"]["15min"]
  assert _td(tf)["green_open"]["meta"]["excluded_days"] == ["2024-01-01", "2024-03-15"]


def test_doji_sample_has_null_tradable_fields() -> None:
  """The doji still produces a SampleRow, but with tradable fields = None."""
  doji = _make_v2_day("2024-03-15", 100.0, 100.0, [(600, 130, 90, 120), (970, 125, 95, 120)])
  stat = _stat()
  tf = stat.compute(_candles(_GREEN_DAYS + [doji])).instruments["NQ"]["15min"]
  doji_sample = next(s for s in tf.samples if s.date == "2024-03-15")
  assert doji_sample.condition == "green_open"  # v1 classifies '>=' as green
  assert doji_sample.anchor_price is None
  assert doji_sample.close_price is None
  assert doji_sample.mfe_pts is None
  assert doji_sample.mae_pts is None


# ===========================================================================
# 7. Samples carry tradable enrichment for non-doji days
# ===========================================================================

def test_samples_carry_tradable_fields() -> None:
  stat = _stat()
  tf = stat.compute(_candles(_GREEN_DAYS)).instruments["NQ"]["15min"]
  s = next(s for s in tf.samples if s.date == "2024-01-02")  # G1
  assert s.anchor_price == pytest.approx(100.0)
  assert s.close_price == pytest.approx(104.0)
  assert s.mfe_pts == pytest.approx(30.0)
  assert s.mae_pts == pytest.approx(10.0)


# ===========================================================================
# 8. Tradable meta
# ===========================================================================

def test_tradable_meta_fields() -> None:
  meta = _green_tradable()["green_open"]["meta"]
  assert meta["anchor"] == "condition_candle_close"
  assert meta["outcome_window"] == "anchor→16:15"
  assert meta["overlap_free"] is True


# ===========================================================================
# 9. Base-stat consistency: v2 results rows must equal v1's
# ===========================================================================

def test_base_stat_rows_match_v1() -> None:
  """The inherited probability rows must not regress relative to v1."""
  # Mixed data: 4 green + 2 red days (no doji so both stats agree on classification).
  red1 = _make_v2_day("2024-04-01", 105.0, 100.0, [(600, 101, 90, 92), (970, 100, 88, 90)])
  red2 = _make_v2_day("2024-04-02", 105.0, 100.0, [(600, 112, 99, 110), (970, 113, 101, 111)])
  df = _candles(_GREEN_DAYS + [red1, red2])

  v1 = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  v2 = _stat()
  rows_v1 = {(r.condition, r.outcome): (r.count, r.total) for r in
             v1.compute(df).instruments["NQ"]["15min"].results}
  rows_v2 = {(r.condition, r.outcome): (r.count, r.total) for r in
             v2.compute(df).instruments["NQ"]["15min"].results}
  assert rows_v1 == rows_v2


def test_base_stat_slices_present() -> None:
  """Slices (inherited from v1) must still be produced."""
  stat = _stat()
  tf = stat.compute(_candles(_GREEN_DAYS)).instruments["NQ"]["15min"]
  assert "weekday" in tf.slices
  assert "close" in tf.slices
  assert "size" in tf.slices


# ===========================================================================
# 10. Edge cases
# ===========================================================================

def test_empty_dataframe_no_crash() -> None:
  stat = _stat()
  tf = stat.compute(_empty_df()).instruments["NQ"]["15min"]
  assert tf.total_samples == 0
  assert tf.tradable == {}


def test_single_day_tradable_degenerate_percentiles() -> None:
  """n=1: percentiles collapse to the single value."""
  one = _make_v2_day("2024-05-01", 95.0, 100.0, [(600, 130, 90, 120), (970, 125, 95, 118)])
  stat = _stat()
  tf = stat.compute(_candles([one])).instruments["NQ"]["15min"]
  block = _td(tf)["green_open"]
  assert block["n"] == 1
  # mfe = 130-100 = 30 ; mae = 100-90 = 10
  assert block["mae"]["p50"] == pytest.approx(10.0)
  assert block["mae"]["p75"] == pytest.approx(10.0)
  assert block["mae"]["p90"] == pytest.approx(10.0)
  assert block["win_rate"] == pytest.approx(1.0)  # close 118 > anchor 100


def test_all_green_only_green_condition_present() -> None:
  stat = _stat()
  tradable = stat.compute(_candles(_GREEN_DAYS)).instruments["NQ"]["15min"].tradable
  assert set(tradable.keys()) == {"green_open"}


def test_all_doji_tradable_empty() -> None:
  """All-doji data: base stat has rows, tradable is empty (everything excluded)."""
  dojis = [
    _make_v2_day(f"2024-06-0{i}", 100.0, 100.0, [(600, 110, 90, 105), (970, 108, 95, 105)])
    for i in range(1, 5)
  ]
  stat = _stat()
  tf = stat.compute(_candles(dojis)).instruments["NQ"]["15min"]
  assert tf.total_samples == 4  # base stat still counts them
  assert tf.tradable == {}


# ===========================================================================
# 11. write_results round-trip with the tradable layer
# ===========================================================================

def test_write_results_roundtrip_tradable(tmp_path: Path) -> None:
  stat = _stat()
  result = stat.compute(_candles(_GREEN_DAYS))
  path = write_results(result, results_dir=tmp_path)
  assert path.name == "opening_candle_continuation_v2.json"

  reloaded = StatRunResult.model_validate(json.loads(path.read_text(encoding="utf-8")))
  block = reloaded.instruments["NQ"]["15min"].tradable["green_open"]
  assert block.win_rate == pytest.approx(0.75)
  assert block.tp_ladder[0].prob == pytest.approx(0.75)
  assert block.mae.p50 == pytest.approx(7.5)


def test_stat_name_is_v2() -> None:
  stat = _stat()
  assert stat.compute(_empty_df()).stat_name == "opening_candle_continuation_v2"
