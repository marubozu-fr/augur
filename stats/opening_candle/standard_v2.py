"""Opening Candle Continuation — v2 with a tradable layer.

The v1 (``standard.py``) measures P(session closes same direction as the opening
candle). Its outcome window (session open → RTH end) *contains* the condition
window (the opening candle), so most of the edge above 50 % is self-counting.

This v2 keeps the v1 base stat verbatim (inherited) and adds a tradable layer
anchored **after** the opening candle closes. From that anchor to RTH end it
measures the tradable residual only:

  - Block 1 — residual: how far price still travels favorably after the anchor.
  - Block 2 — TP ladder: P(MFE >= adaptive level) at 5 levels.
  - Block 3 — drawdown: MAE distribution + MAE-before-first-touch per TP level.

All MFE/MAE measurements use 1-min bar highs/lows over the outcome window. Doji
opening candles contribute to the base stat (v1 classifies them green) but are
excluded from every tradable metric.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stats.base import (
  ConditionalMae,
  I18nString,
  MaeResult,
  SampleRow,
  StatRunResult,
  TimeframeResult,
  TpLadderLevel,
  TradableMeta,
  TradableResult,
  write_results,
)
from stats.config import InstrumentConfig, load_config
from stats.opening_candle.standard import OpeningCandleContinuation

# ---------------------------------------------------------------------------
# i18n content (v2-specific title/definition; conditions/outcomes inherited)
# ---------------------------------------------------------------------------
_TITLE_V2 = I18nString(
  en="Opening Candle Continuation (tradable)",
  fr="Continuation de la bougie d'ouverture (tradable)",
)
_DEFINITION_V2 = I18nString(
  en=(
    "Base stat identical to v1, plus a tradable layer anchored at the opening "
    "candle close: residual travel, take-profit ladder, and drawdown from the "
    "anchor to the NY session close."
  ),
  fr=(
    "Statistique de base identique à v1, plus une couche tradable ancrée à la "
    "clôture de la bougie d'ouverture : résidu de mouvement, échelle de "
    "take-profit et drawdown de l'ancre à la clôture de session NY."
  ),
)

# Adaptive TP multipliers applied to the median outcome-window range.
_TP_MULTIPLIERS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5)


class OpeningCandleContinuationV2(OpeningCandleContinuation):
  """v1 base stat plus an overlap-free tradable layer.

  Inherits ``compute_rows`` / ``baseline_rows`` / slices unchanged, extends
  ``build_day_table`` with tradable columns, and overrides ``compute`` to attach
  the tradable layer.
  """

  stat_name = "opening_candle_continuation_v2"
  title = _TITLE_V2
  definition = _DEFINITION_V2

  # -- helpers --------------------------------------------------------------

  @property
  def _outcome_start_min(self) -> int:
    """Minute-of-day of the first outcome-window bar (the bar after the opening
    candle closes)."""
    return self.candle_open_min + self.tf_minutes

  def _outcome_sequences(
    self, candles_df: pd.DataFrame
  ) -> dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]]:
    """Per-day ordered (highs, lows) over the outcome window [anchor+1 .. RTH end).

    Keyed by normalized session date. Used for the conditional-MAE first-touch
    computation, which needs the intra-day bar order that a groupby-max cannot
    provide.
    """
    if candles_df.empty:
      return {}
    df = candles_df.copy()
    df["mod"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["date"] = df["timestamp"].dt.normalize()
    mask = (df["mod"] >= self._outcome_start_min) & (df["mod"] < self.rth_end_min)
    ow = df[mask].sort_values(["date", "mod"])
    return {
      date: (grp["high"].to_numpy(dtype=float), grp["low"].to_numpy(dtype=float))
      for date, grp in ow.groupby("date")
    }

  # -- overrides ------------------------------------------------------------

  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """v1 day table enriched with tradable columns.

    Adds ``anchor_price`` (= opening_close), ``close_price`` (= session_close),
    ``is_doji`` (opening_close == opening_open), and direction-adjusted
    ``mfe_pts`` / ``mae_pts`` measured on 1-min bars over the outcome window.
    Tradable magnitudes are NaN for doji days (excluded from the tradable layer).
    """
    day = super().build_day_table(candles_df)

    if day.empty:
      day["anchor_price"] = pd.Series(dtype="float64")
      day["close_price"] = pd.Series(dtype="float64")
      day["is_doji"] = pd.Series(dtype="bool")
      day["mfe_pts"] = pd.Series(dtype="float64")
      day["mae_pts"] = pd.Series(dtype="float64")
      return day

    day["anchor_price"] = day["opening_close"]
    day["close_price"] = day["session_close"]
    day["is_doji"] = day["opening_close"] == day["opening_open"]

    # Outcome-window extremes (favorable/adverse depend on candle direction).
    df = candles_df.copy()
    df["mod"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["date"] = df["timestamp"].dt.normalize()
    ow_mask = (df["mod"] >= self._outcome_start_min) & (df["mod"] < self.rth_end_min)
    ow = df[ow_mask]
    ow_high = ow.groupby("date")["high"].max().reindex(day.index)
    ow_low = ow.groupby("date")["low"].min().reindex(day.index)

    green = day["opening_green"].to_numpy(dtype=bool)
    anchor = day["anchor_price"].to_numpy(dtype=float)
    high = ow_high.to_numpy(dtype=float)
    low = ow_low.to_numpy(dtype=float)
    # Green: favorable = up. Red: favorable = down.
    day["mfe_pts"] = np.where(green, high - anchor, anchor - low)
    day["mae_pts"] = np.where(green, anchor - low, high - anchor)

    # Doji days are excluded from the tradable layer entirely.
    doji = day["is_doji"].to_numpy(dtype=bool)
    day.loc[doji, ["mfe_pts", "mae_pts"]] = np.nan

    return day

  def classify_samples(self, day_table: pd.DataFrame) -> list[SampleRow]:
    """One SampleRow per resolved day, carrying the tradable enrichment fields.

    Doji days still produce a sample (base-stat consistency) but with tradable
    fields set to ``None``.
    """
    if day_table.empty:
      return []
    samples: list[SampleRow] = []
    for ts, row in day_table.iterrows():
      condition = "green_open" if bool(row["opening_green"]) else "red_open"
      outcome = "green_close" if bool(row["session_green"]) else "red_close"
      if bool(row["is_doji"]):
        anchor = close = mfe = mae = None
      else:
        anchor = float(row["anchor_price"])
        close = float(row["close_price"])
        mfe = float(row["mfe_pts"])
        mae = float(row["mae_pts"])
      samples.append(
        SampleRow(
          date=ts.strftime("%Y-%m-%d"),
          condition=condition,
          outcome=outcome,
          anchor_price=anchor,
          close_price=close,
          mfe_pts=mfe,
          mae_pts=mae,
        )
      )
    return samples

  def compute(self, candles_df: pd.DataFrame, seed: int = 42) -> StatRunResult:
    """Reproduce BaseStat's compute flow (single ``build_day_table`` call) and
    attach the tradable layer."""
    day_table = self.build_day_table(candles_df)
    baseline = self.baseline_rows(day_table, seed)
    rows = self.compute_rows(day_table, baseline_rows=baseline)
    slice_results, dimension_labels = self._slice_results(day_table, seed)
    samples = sorted(self.classify_samples(day_table), key=lambda s: s.date)
    tradable = self._compute_tradable(candles_df, day_table)

    if len(day_table) > 0:
      dates = day_table.index
      data_range = [dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")]
    else:
      data_range = []

    tf_result = TimeframeResult(
      data_range=data_range,
      total_samples=len(day_table),
      results=rows,
      slices=slice_results,
      samples=samples,
      tradable=tradable,
    )

    labels = self.labels.model_copy(deep=True)
    labels.dimensions = {**labels.dimensions, **dimension_labels}

    return StatRunResult(
      stat_name=self.stat_name,
      title=self.title,
      definition=self.definition,
      labels=labels,
      instruments={self.instrument: {self.timeframe: tf_result}},
    )

  # -- tradable layer -------------------------------------------------------

  def _compute_tradable(
    self, candles_df: pd.DataFrame, day_table: pd.DataFrame
  ) -> dict[str, TradableResult]:
    """Compute the tradable layer for each condition with non-doji days.

    Conditions with no non-doji day are omitted (e.g. all-red data has no
    ``green_open`` entry). Returns an empty dict when nothing is tradable.
    """
    if day_table.empty:
      return {}

    sequences = self._outcome_sequences(candles_df)
    rth_end = self.config.sessions["rth"].end
    out: dict[str, TradableResult] = {}

    for cond_key, cond_green in (("green_open", True), ("red_open", False)):
      cond = day_table[day_table["opening_green"] == cond_green]
      doji_mask = cond["is_doji"].to_numpy(dtype=bool)
      excluded_days = sorted(ts.strftime("%Y-%m-%d") for ts in cond.index[doji_mask])
      non_doji = cond[~doji_mask]
      if len(non_doji) == 0:
        continue

      anchor = non_doji["anchor_price"].to_numpy(dtype=float)
      close = non_doji["close_price"].to_numpy(dtype=float)
      mfe = non_doji["mfe_pts"].to_numpy(dtype=float)
      mae = non_doji["mae_pts"].to_numpy(dtype=float)
      n = len(non_doji)

      # Block 1 — tradable residual (signed: positive = favorable).
      remaining = (close - anchor) if cond_green else (anchor - close)

      # Block 2 — adaptive TP ladder + Block 3 conditional MAE per level.
      median_range = float(np.median(mfe + mae))
      tp_ladder: list[TpLadderLevel] = []
      conditional: list[ConditionalMae] = []
      for multiplier in _TP_MULTIPLIERS:
        level_pts = median_range * multiplier
        prob = float(np.mean(mfe >= level_pts))
        tp_ladder.append(
          TpLadderLevel(level_x_range=multiplier, level_pts=level_pts, prob=prob, n=n)
        )
        mae_pct = self._conditional_mae_pct(
          non_doji.index, anchor, mfe, level_pts, cond_green, sequences
        )
        conditional.append(
          ConditionalMae(
            level_pts=level_pts,
            mae_pct_p50=float(np.percentile(mae_pct, 50)) if mae_pct.size else 0.0,
            mae_pct_p75=float(np.percentile(mae_pct, 75)) if mae_pct.size else 0.0,
          )
        )

      out[cond_key] = TradableResult(
        win_rate=float(np.mean(remaining > 0)),
        remaining_mean_pts=float(np.mean(remaining)),
        remaining_median_pts=float(np.median(remaining)),
        remaining_mean_pct=float(np.mean(remaining / anchor)),
        n=n,
        tp_ladder=tp_ladder,
        mae=MaeResult(
          p50=float(np.percentile(mae, 50)),
          p75=float(np.percentile(mae, 75)),
          p90=float(np.percentile(mae, 90)),
          conditional=conditional,
        ),
        meta=TradableMeta(
          anchor="condition_candle_close",
          outcome_window=f"anchor→{rth_end}",
          overlap_free=True,
          excluded_days=excluded_days,
        ),
      )

    return out

  def _conditional_mae_pct(
    self,
    dates: pd.Index,
    anchor: np.ndarray,
    mfe: np.ndarray,
    level_pts: float,
    cond_green: bool,
    sequences: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]],
  ) -> np.ndarray:
    """MAE-before-first-touch (fraction of anchor) for days that reach the level.

    Among days where ``mfe >= level_pts`` (i.e. the level is touched), compute the
    worst excursion from the anchor up to and including the first bar that touches
    the level. Green: touch = first high >= anchor+level, drawdown below anchor.
    Red: touch = first low <= anchor-level, drawdown above anchor.
    """
    pcts: list[float] = []
    for i, ts in enumerate(dates):
      if mfe[i] < level_pts:
        continue
      highs, lows = sequences[ts]
      a = anchor[i]
      if cond_green:
        touch_idx = int(np.argmax(highs >= a + level_pts))
        mae_before = a - lows[: touch_idx + 1].min()
      else:
        touch_idx = int(np.argmax(lows <= a - level_pts))
        mae_before = highs[: touch_idx + 1].max() - a
      pcts.append(mae_before / a)
    return np.asarray(pcts, dtype=float)


def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
) -> Path:
  """Load data and compute Opening Candle Continuation v2 for all three timeframes."""
  config: InstrumentConfig = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  timeframes = ["15min", "30min", "1h"]
  merged_tf: dict[str, TimeframeResult] = {}
  labels = _labels_seed()

  for tf in timeframes:
    stat = OpeningCandleContinuationV2(instrument=instrument, timeframe=tf, config=config)
    result = stat.compute(candles_df)
    merged_tf[tf] = result.instruments[instrument][tf]
    labels.dimensions.update(result.labels.dimensions)

  final_result = StatRunResult(
    stat_name="opening_candle_continuation_v2",
    title=_TITLE_V2,
    definition=_DEFINITION_V2,
    labels=labels,
    instruments={instrument: merged_tf},
  )

  return write_results(final_result)


def _labels_seed():
  """Deep copy of the inherited condition/outcome labels for merging dimensions."""
  return OpeningCandleContinuationV2.labels.model_copy(deep=True)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Opening Candle Continuation v2 stat")
  parser.add_argument("--instrument", default="NQ", help="Instrument name (default: NQ)")
  parser.add_argument("--data-path", default=None, help="Override parquet file path")
  parser.add_argument("--config-dir", default="config", help="Config directory (default: config)")
  args = parser.parse_args()

  output_path = run(
    instrument=args.instrument,
    config_dir=args.config_dir,
    data_path=args.data_path,
  )

  print(f"Written: {output_path}")

  import json

  with open(output_path, encoding="utf-8") as f:
    data = json.load(f)

  instrument_data = data["instruments"][args.instrument]
  for tf, tf_data in instrument_data.items():
    print(f"\n  {tf}: {tf_data['total_samples']} resolved days | {tf_data['data_range']}")
    tradable = tf_data.get("tradable") or {}
    for cond, block in tradable.items():
      ladder = ", ".join(
        f"{lvl['level_pts']:.0f}pt={lvl['prob']:.2f}" for lvl in block["tp_ladder"]
      )
      print(
        f"    {cond}: n={block['n']} win_rate={block['win_rate']:.3f} "
        f"remaining_median={block['remaining_median_pts']:.1f}pt"
      )
      print(f"      TP ladder: {ladder}")
      print(
        f"      MAE p50/p75/p90: {block['mae']['p50']:.0f}/"
        f"{block['mae']['p75']:.0f}/{block['mae']['p90']:.0f} pts"
      )
