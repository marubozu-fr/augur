"""Market Open Volume stat.

Measures the Pearson correlation between the opening bar's traded volume and
the rest-of-session volume across resolved RTH trading days. A positive
correlation indicates that high-volume opens tend to precede high-volume
sessions; a near-zero correlation suggests the open gives no predictive
information about session activity.

Methodology
-----------
For each resolved RTH day, two quantities are computed from the 1-min bars:

  open_volume  — sum of volume in [rth_start, rth_start + open_bar_min)
  rest_volume  — sum of volume in [rth_start + open_bar_min, rth_end)

The stat reports the Pearson r across all resolved days that have both
quantities non-NaN.

The ``open_bar_min`` is governed by the chosen ``timeframe`` (15min, 30min,
or 1h), mirroring the Opening Candle Continuation convention where the
opening bar duration equals the chart timeframe.

Baseline
--------
The null hypothesis is that the opening bar carries no information about the
rest of the session. The random baseline permutes ``rest_volume`` across days
(using a fixed seed for deterministic output) while holding ``open_volume``
fixed, then recomputes Pearson r. Under the null, the expected r ≈ 0.

Pending-sample discipline
-------------------------
``build_resolved_days`` excludes:
  - days with no bar at rth_start_min (no clean session open),
  - early-close days (last RTH bar before rth_end - close_tolerance_min),
  - the final incomplete day in the data.
Only fully resolved days enter the day table and thus the correlation.

Multi-timeframe merge
---------------------
The ``run()`` function computes all three supported timeframes (15min, 30min,
1h) in one pass and merges their TimeframeResults into a single JSON file
``results/market_open_volume.json`` under
``instruments.{INSTRUMENT}.{timeframe}``, mirroring the Opening Candle
Continuation merge pattern.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stats.base import (
  BaseStat,
  I18nString,
  Labels,
  StatResultRow,
  StatRunResult,
  TimeframeResult,
  write_results,
)
from stats.config import InstrumentConfig, load_config, minute_of_day
from stats.utils.daily_candles import build_resolved_days

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="Market Open Volume",
  fr="Volume à l'ouverture",
)
_DEFINITION = I18nString(
  en="Pearson correlation between the opening bar's volume and the rest-of-session volume: does a high-volume open predict a high-volume session?",
  fr="Corrélation de Pearson entre le volume de la bougie d'ouverture et le volume du reste de la séance : un volume élevé à l'ouverture annonce-t-il une séance active ?",
)
_LABELS = Labels(
  conditions={
    "any_day": I18nString(en="All days", fr="Tous les jours"),
  },
  outcomes={
    "correlation": I18nString(
      en="Open vs rest-of-session volume correlation",
      fr="Corrélation volume ouverture / reste de séance",
    ),
  },
)

# Single condition/outcome keys used throughout.
_CONDITION = "any_day"
_OUTCOME = "correlation"

# Supported opening-bar timeframes: label -> minutes.
_TIMEFRAME_MINUTES: dict[str, int] = {
  "15min": 15,
  "30min": 30,
  "1h": 60,
}


class MarketOpenVolume(BaseStat):
  """Pearson correlation between opening-bar volume and rest-of-session volume.

  One instance covers a single opening-bar duration (``timeframe``). The
  ``run()`` function computes all three supported timeframes and merges their
  results into one JSON file.
  """

  stat_name = "market_open_volume"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = ()

  def __init__(
    self,
    instrument: str,
    config: InstrumentConfig,
    timeframe: str = "1h",
    close_tolerance_min: int = 15,
  ) -> None:
    if timeframe not in _TIMEFRAME_MINUTES:
      raise ValueError(
        f"Unknown timeframe '{timeframe}'. Choose from {list(_TIMEFRAME_MINUTES)}."
      )
    self.instrument = instrument
    self.timeframe = timeframe
    self.config = config
    self.open_bar_min: int = _TIMEFRAME_MINUTES[timeframe]
    self.close_tolerance_min = close_tolerance_min

    rth = config.sessions["rth"]
    self.rth_start_min: int = minute_of_day(rth.start)
    self.rth_end_min: int = minute_of_day(rth.end)

    self.labels = _LABELS

  # -------------------------------------------------------------------------
  # Day table
  # -------------------------------------------------------------------------
  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-day table with ``open_volume`` and ``rest_volume`` columns.

    Each row is one resolved trading day, indexed by the normalized session
    date. Columns:
      open_volume  — sum of RTH bar volumes in [rth_start, rth_start + open_bar_min)
      rest_volume  — sum of RTH bar volumes in [rth_start + open_bar_min, rth_end)

    Days where either window has no bars produce NaN for that column. The
    resolved-days index (from ``build_resolved_days``) ensures only clean,
    fully-closed sessions are included.
    """
    columns = ["open_volume", "rest_volume"]
    empty = pd.DataFrame(columns=columns)

    if candles_df.empty:
      return empty

    resolved = build_resolved_days(
      candles_df, self.rth_start_min, self.rth_end_min, self.close_tolerance_min
    )
    if resolved.empty:
      return empty

    df = candles_df.copy()
    df["_mod"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["_date"] = df["timestamp"].dt.normalize()

    open_end_min = self.rth_start_min + self.open_bar_min

    # Opening window: [rth_start, rth_start + open_bar_min)
    open_mask = (df["_mod"] >= self.rth_start_min) & (df["_mod"] < open_end_min)
    open_volume = (
      df[open_mask].groupby("_date")["volume"].sum().rename("open_volume")
    )

    # Rest window: [rth_start + open_bar_min, rth_end)
    rest_mask = (df["_mod"] >= open_end_min) & (df["_mod"] < self.rth_end_min)
    rest_volume = (
      df[rest_mask].groupby("_date")["volume"].sum().rename("rest_volume")
    )

    # Assemble on the resolved index so only resolved dates pass through.
    # Use reindex (not join) so absent windows produce NaN rather than being
    # silently dropped.
    out = pd.DataFrame(index=resolved.index)
    out["open_volume"] = open_volume.reindex(resolved.index)
    out["rest_volume"] = rest_volume.reindex(resolved.index)

    return out[columns]

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the single Pearson-r magnitude row over a day table.

    Only rows where both ``open_volume`` and ``rest_volume`` are non-NaN
    contribute. ``count`` and ``total`` both equal that valid-row count N.

    The correlation is set to 0.0 when N < 2 or either column has zero
    variance (degenerate cases where r is undefined).

    If ``baseline_rows`` is provided, the matching row's ``value`` / ``total``
    are merged into ``value_baseline`` / ``baseline_n``.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    # Work on rows that have both columns valid.
    valid = day_table[["open_volume", "rest_volume"]].dropna()
    n = len(valid)

    if n < 2:
      r = 0.0
    else:
      ov = valid["open_volume"].astype(float)
      rv = valid["rest_volume"].astype(float)
      # Guard against zero variance (constant series).
      if float(ov.std()) == 0.0 or float(rv.std()) == 0.0:
        r = 0.0
      else:
        r = float(ov.corr(rv))
        # corr() can return NaN in edge cases (e.g. identical values after
        # floating-point arithmetic); treat as zero.
        if np.isnan(r):
          r = 0.0

    bl = baseline_map.get((_CONDITION, _OUTCOME))
    return [
      StatResultRow(
        condition=_CONDITION,
        outcome=_OUTCOME,
        count=n,
        total=n,
        probability=0.0,
        baseline_prob=0.0,
        baseline_n=bl.total if bl else 0,
        value=r,
        value_baseline=bl.value if bl else None,
      )
    ]

  # -------------------------------------------------------------------------
  # Baseline
  # -------------------------------------------------------------------------
  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random null: permute ``rest_volume`` across days, recompute Pearson r.

    Holding ``open_volume`` fixed while randomly reassigning ``rest_volume``
    destroys any open→rest relationship. Under the null hypothesis the
    expected r ≈ 0. Uses ``np.random.default_rng`` with the supplied seed for
    deterministic, reproducible output.

    Only the both-non-NaN rows participate (pending-sample discipline). If
    fewer than two valid rows exist, the degenerate path in ``compute_rows``
    returns r = 0.0.
    """
    # Isolate the valid subset.
    valid = day_table[["open_volume", "rest_volume"]].dropna()
    n = len(valid)

    if n < 2:
      # Degenerate: return the zero-r row directly (no shuffling possible).
      return self.compute_rows(valid, baseline_rows=None)

    rng = np.random.default_rng(seed)
    shuffled = valid.copy()
    perm = rng.permutation(n)
    shuffled["rest_volume"] = valid["rest_volume"].to_numpy()[perm]

    return self.compute_rows(shuffled, baseline_rows=None)


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
) -> Path:
  """Compute Market Open Volume for all three opening-bar timeframes.

  Loads config and parquet data once, then computes each timeframe and merges
  the per-timeframe TimeframeResults into a single StatRunResult. Writes the
  consolidated JSON to ``results/market_open_volume.json``.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  timeframes = list(_TIMEFRAME_MINUTES)  # ["15min", "30min", "1h"]
  merged_tf: dict[str, TimeframeResult] = {}

  for tf in timeframes:
    stat = MarketOpenVolume(instrument=instrument, config=config, timeframe=tf)
    result = stat.compute(candles_df)
    merged_tf[tf] = result.instruments[instrument][tf]

  final_result = StatRunResult(
    stat_name="market_open_volume",
    title=_TITLE,
    definition=_DEFINITION,
    labels=_LABELS,
    instruments={instrument: merged_tf},
  )

  return write_results(final_result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Market Open Volume stat")
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

  with open(output_path, encoding="utf-8") as f:
    data = json.load(f)

  instrument_data = data["instruments"][args.instrument]
  for tf, tf_data in instrument_data.items():
    rows = {r["outcome"]: r for r in tf_data["results"]}
    corr_row = rows.get("correlation", {})
    r_val = corr_row.get("value", float("nan"))
    r_bl = corr_row.get("value_baseline")
    bl_str = f"{r_bl:.4f}" if r_bl is not None else "n/a"
    print(
      f"\n  {tf}: {tf_data['total_samples']} resolved days"
      f" | {tf_data['data_range']}"
      f" | r={r_val:.4f} (baseline={bl_str})"
    )
