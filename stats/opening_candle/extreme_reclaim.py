"""Opening Candle Extreme Reclaim stat.

After the opening candle closes, does the session reclaim the candle's high,
its low, both (and in what order), or neither?

This is a pure 5-way probability partition per condition (no tradable layer, no
magnitude channel). It inherits the v1 (``standard.py``) day-table construction
— opening candle OHLC, session open/close, resolution filter — and adds a scan
of the 1-min bars in the outcome window ``[candle_open + tf, rth_end)`` (the same
post-candle window used by the reverted v2, which excludes the opening candle
itself).

Outcomes (a partition — every non-doji resolved day falls in exactly one):
  high_only  — reclaims opening_high, never reclaims opening_low
  low_only   — reclaims opening_low, never reclaims opening_high
  high_first — reclaims both; opening_high touched first
  low_first  — reclaims both; opening_low touched first
  neither    — reclaims neither extreme

Doji opening candles (``opening_close == opening_open``) have no direction to
classify and are excluded entirely — dropped from the day table so they never
enter any count, slice, or sample.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stats.base import (
  I18nString,
  Labels,
  SampleRow,
  StatResultRow,
  StatRunResult,
  TimeframeResult,
  write_results,
)
from stats.config import load_config
from stats.opening_candle.standard import OpeningCandleContinuation

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="Opening Candle Extreme Reclaim",
  fr="Reclaim des extrêmes de la bougie d'ouverture",
)
_DEFINITION = I18nString(
  en=(
    "After the opening candle closes, does the session reclaim the candle's "
    "high, low, both (and in what order), or neither?"
  ),
  fr=(
    "Après la clôture de la bougie d'ouverture, la session reclaime-t-elle le "
    "high, le low, les deux (et dans quel ordre), ou aucun ?"
  ),
)
# Conditions are identical to v1 (opening candle direction). Doji is excluded, so
# only green_open / red_open apply.
_LABELS = Labels(
  conditions={
    "green_open": I18nString(en="Green opening candle", fr="Bougie d'ouverture verte"),
    "red_open": I18nString(en="Red opening candle", fr="Bougie d'ouverture rouge"),
  },
  outcomes={
    "high_only": I18nString(en="Reclaims high only", fr="Reclaim du high uniquement"),
    "low_only": I18nString(en="Reclaims low only", fr="Reclaim du low uniquement"),
    "high_first": I18nString(en="Reclaims both, high first", fr="Reclaim des deux, high en premier"),
    "low_first": I18nString(en="Reclaims both, low first", fr="Reclaim des deux, low en premier"),
    "neither": I18nString(en="Reclaims neither", fr="Aucun reclaim"),
  },
)

# Ordered outcome partition. Every non-doji resolved day maps to exactly one.
_OUTCOMES: tuple[str, ...] = ("high_only", "low_only", "high_first", "low_first", "neither")


def _classify_outcome(
  reclaims_high: np.ndarray,
  reclaims_low: np.ndarray,
  first_touch: np.ndarray,
) -> np.ndarray:
  """Map (reclaims_high, reclaims_low, first_touch) arrays to the 5-way outcome.

  ``first_touch`` is only consulted when both extremes are reclaimed. A same-bar
  double touch (``first_touch == "same"``, extremely rare) is classified as
  ``high_first`` by the tiebreak convention.
  """
  rh = np.asarray(reclaims_high, dtype=bool)
  rl = np.asarray(reclaims_low, dtype=bool)
  ft = np.asarray(first_touch, dtype=object)
  both = rh & rl
  return np.select(
    [rh & ~rl, rl & ~rh, both & (ft != "low"), both & (ft == "low")],
    ["high_only", "low_only", "high_first", "low_first"],
    default="neither",  # ~rh & ~rl
  )


class OpeningCandleExtremeReclaim(OpeningCandleContinuation):
  """5-way partition of post-opening-candle extreme reclaims.

  Inherits v1's day-table construction and its slices (weekday, close location,
  size bucket). Overrides the row/sample/baseline hooks to classify each day into
  one of the five reclaim outcomes. The base ``compute`` flow is used unchanged —
  there is no tradable layer to attach.
  """

  stat_name = "opening_candle_extreme_reclaim"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS

  @property
  def _outcome_start_min(self) -> int:
    """Minute-of-day of the first outcome-window bar (the bar after the opening
    candle closes)."""
    return self.candle_open_min + self.tf_minutes

  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """v1 day table enriched with reclaim columns; doji days dropped.

    Adds ``reclaims_high`` / ``reclaims_low`` (bool), ``first_touch``
    ("high" / "low" / "same" / "none"), and ``outcome`` (one of ``_OUTCOMES``),
    then drops doji opening candles (``opening_close == opening_open``) so they
    never enter any statistic.

    "Reaches" means at least one 1-min bar in the outcome window
    ``[candle_open + tf, rth_end)`` has ``high >= opening_high`` (high reclaim) or
    ``low <= opening_low`` (low reclaim). Ordering compares the first such bar's
    minute-of-day; a same-bar double touch is recorded as ``first_touch == "same"``.
    """
    day = super().build_day_table(candles_df)

    if day.empty:
      day["reclaims_high"] = pd.Series(dtype="bool")
      day["reclaims_low"] = pd.Series(dtype="bool")
      day["first_touch"] = pd.Series(dtype="object")
      day["outcome"] = pd.Series(dtype="object")
      return day

    is_doji = (day["opening_close"] == day["opening_open"]).to_numpy(dtype=bool)

    # Scan the outcome window per day for reclaims and the first-touch order.
    df = candles_df.copy()
    df["mod"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["date"] = df["timestamp"].dt.normalize()
    ow_mask = (df["mod"] >= self._outcome_start_min) & (df["mod"] < self.rth_end_min)
    ow = df.loc[ow_mask, ["date", "mod", "high", "low"]].copy()
    # Attach each day's opening extremes; drop bars for days absent from the table.
    ow["opening_high"] = ow["date"].map(day["opening_high"])
    ow["opening_low"] = ow["date"].map(day["opening_low"])
    ow = ow.dropna(subset=["opening_high", "opening_low"])
    ow["hit_high"] = ow["high"] >= ow["opening_high"]
    ow["hit_low"] = ow["low"] <= ow["opening_low"]

    reclaims_high = ow.groupby("date")["hit_high"].any().reindex(day.index, fill_value=False)
    reclaims_low = ow.groupby("date")["hit_low"].any().reindex(day.index, fill_value=False)
    # First bar (by minute-of-day) that touches each extreme; NaN when never touched.
    first_high = ow.loc[ow["hit_high"]].groupby("date")["mod"].min().reindex(day.index)
    first_low = ow.loc[ow["hit_low"]].groupby("date")["mod"].min().reindex(day.index)

    rh = reclaims_high.to_numpy(dtype=bool)
    rl = reclaims_low.to_numpy(dtype=bool)
    fh = first_high.to_numpy(dtype=float)
    fl = first_low.to_numpy(dtype=float)
    both = rh & rl

    first_touch = np.full(len(day), "none", dtype=object)
    first_touch[rh & ~rl] = "high"
    first_touch[rl & ~rh] = "low"
    first_touch[both & (fh < fl)] = "high"
    first_touch[both & (fl < fh)] = "low"
    first_touch[both & (fh == fl)] = "same"

    day["reclaims_high"] = rh
    day["reclaims_low"] = rl
    day["first_touch"] = first_touch
    day["outcome"] = _classify_outcome(rh, rl, first_touch)

    # Doji opening candles have no direction — excluded entirely.
    return day[~is_doji].copy()

  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the 2×5 condition/outcome rows over a (possibly sliced) day table.

    Denominator per condition = resolved non-doji days with that opening
    direction; the five outcome probabilities within a condition sum to 1.0.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    has_rows = not day_table.empty
    rows: list[StatResultRow] = []
    for cond_key, cond_green in (("green_open", True), ("red_open", False)):
      cond_mask = (
        day_table["opening_green"] == cond_green
        if has_rows
        else pd.Series([], dtype=bool)
      )
      total = int(cond_mask.sum()) if has_rows else 0

      for out_key in _OUTCOMES:
        count = (
          int(((day_table["outcome"] == out_key) & cond_mask).sum()) if has_rows else 0
        )
        probability = count / total if total > 0 else 0.0

        bl = baseline_map.get((cond_key, out_key))
        rows.append(
          StatResultRow(
            condition=cond_key,
            outcome=out_key,
            count=count,
            total=total,
            probability=probability,
            baseline_prob=bl.probability if bl else 0.0,
            baseline_n=bl.total if bl else 0,
          )
        )

    return rows

  def classify_samples(self, day_table: pd.DataFrame) -> list[SampleRow]:
    """One SampleRow per resolved non-doji day (condition + outcome)."""
    if day_table.empty:
      return []
    conditions = np.where(day_table["opening_green"].to_numpy(dtype=bool), "green_open", "red_open")
    outcomes = day_table["outcome"].to_numpy()
    return [
      SampleRow(date=ts.strftime("%Y-%m-%d"), condition=cond, outcome=out)
      for ts, cond, out in zip(day_table.index, conditions, outcomes)
    ]

  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random baseline: two independent coin flips per day.

    P(high touched) = P(low touched) = 0.5; when both are touched the order is a
    third fair coin. Expected baseline probabilities: high_only 25%, low_only 25%,
    high_first 12.5%, low_first 12.5%, neither 25%. Deterministic for a fixed seed.
    """
    n = len(day_table)
    if n == 0:
      return self.compute_rows(day_table, baseline_rows=None)

    rng = np.random.default_rng(seed)
    rand_high = rng.random(n) < 0.5
    rand_low = rng.random(n) < 0.5
    high_first = rng.random(n) < 0.5
    # first_touch only matters where both are touched; "high"/"low" pick the order.
    first_touch = np.where(high_first, "high", "low").astype(object)

    tmp = day_table.copy()
    tmp["outcome"] = _classify_outcome(rand_high, rand_low, first_touch)
    return self.compute_rows(tmp, baseline_rows=None)


def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
) -> Path:
  """Load data and compute Opening Candle Extreme Reclaim for all three timeframes.

  Merges per-timeframe TimeframeResults into a single StatRunResult and writes
  the consolidated JSON to results/.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  timeframes = ["15min", "30min", "1h"]
  merged_tf: dict[str, TimeframeResult] = {}
  labels = _LABELS.model_copy(deep=True)

  for tf in timeframes:
    stat = OpeningCandleExtremeReclaim(instrument=instrument, timeframe=tf, config=config)
    result = stat.compute(candles_df)
    merged_tf[tf] = result.instruments[instrument][tf]
    # Merge framework-enriched slice dimension labels across timeframes.
    labels.dimensions.update(result.labels.dimensions)

  final_result = StatRunResult(
    stat_name="opening_candle_extreme_reclaim",
    title=_TITLE,
    definition=_DEFINITION,
    labels=labels,
    instruments={instrument: merged_tf},
  )

  return write_results(final_result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Opening Candle Extreme Reclaim stat")
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
    for row in tf_data["results"]:
      print(
        f"    {row['condition']} -> {row['outcome']}: "
        f"{row['probability']:.3f} (N={row['total']}, baseline={row['baseline_prob']:.3f})"
      )
