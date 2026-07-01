"""Volume & Range by Weekday stat.

Measures, for each weekday: the average RTH session volume, the average price
range (high - low, in points), and the average percentage range
((high - low) / session_open).

The overall result aggregates all resolved days; the per-weekday breakdown is
produced by the framework's declared ``weekday`` slice.

This is a **magnitude** stat: every outcome carries its metric in
``StatResultRow.value`` (and its random baseline in ``value_baseline``). The
ordinary ``probability`` channel is left at ``0.0`` for every row.

A "day" is the RTH daily candle (session open to session close). For each
resolved day:
  - ``volume``    = sum of the RTH bars' volume.
  - ``range``     = day_high - day_low (intraday RTH extremes), in points.
  - ``range_pct`` = (day_high - day_low) / session_open, a decimal (e.g. ``0.012``
                    = 1.2%).
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
  SampleRow,
  StatResultRow,
  write_results,
)
from stats.config import InstrumentConfig, load_config, minute_of_day
from stats.utils.daily_candles import build_resolved_days

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="Volume & Range by Weekday",
  fr="Volume & amplitude par jour de la semaine",
)
_DEFINITION = I18nString(
  en="What is the average volume, the average price range, and the average percentage range for each weekday?",
  fr="Quels sont le volume moyen, l'amplitude moyenne en points et l'amplitude moyenne en pourcentage pour chaque jour de la semaine ?",
)
_LABELS = Labels(
  conditions={
    "any_day": I18nString(en="All trading days", fr="Tous les jours de bourse"),
  },
  outcomes={
    "mean_volume": I18nString(en="Average volume", fr="Volume moyen"),
    "mean_range": I18nString(en="Average range (points)", fr="Amplitude moyenne (points)"),
    "mean_range_pct": I18nString(en="Average range (%)", fr="Amplitude moyenne (%)"),
  },
)

# Single condition: every resolved day belongs to it; the weekday slice does the
# per-day breakdown.
_CONDITION = "any_day"

# Outcome keys and the day-table column carrying each metric, in display order.
_OUTCOMES: list[tuple[str, str]] = [
  ("mean_volume", "volume"),
  ("mean_range", "range"),
  ("mean_range_pct", "range_pct"),
]


class VolumeRangeByWeekday(BaseStat):
  """Average volume, price range, and percentage range, sliced by weekday."""

  stat_name = "volume_range_weekday"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = ("weekday",)

  def __init__(
    self,
    instrument: str,
    config: InstrumentConfig,
    close_tolerance_min: int = 15,
  ) -> None:
    self.instrument = instrument
    self.timeframe = "daily"
    self.config = config
    self.close_tolerance_min = close_tolerance_min

    rth = config.sessions["rth"]
    self.rth_start_min: int = minute_of_day(rth.start)
    self.rth_end_min: int = minute_of_day(rth.end)

    # Full resolved-day table, stashed by build_day_table so the per-slice random
    # baseline can sample from the whole population of days (see baseline_rows).
    self._full_day_table: pd.DataFrame | None = None

  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-day RTH summary table from 1-min OHLCV data.

    Each row is one resolved trading day, indexed by the normalized session date,
    with columns: ``volume``, ``range``, ``range_pct``.

    The resolved-days core (RTH filter, session open extraction, resolution
    filter, chronological sort) is shared via ``build_resolved_days``; the RTH
    intraday high/low and the summed volume are joined on per day.

    The full table is stashed on ``self._full_day_table`` so the random baseline
    can sample from the whole population of resolved days.
    """
    columns = ["volume", "range", "range_pct"]
    empty = pd.DataFrame(columns=columns)

    if candles_df.empty:
      self._full_day_table = empty
      return empty

    resolved = build_resolved_days(
      candles_df, self.rth_start_min, self.rth_end_min, self.close_tolerance_min
    )
    if resolved.empty:
      self._full_day_table = empty
      return empty

    # Per-day RTH high, low and summed volume.
    df = candles_df.copy()
    df["_mod"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["_date"] = df["timestamp"].dt.normalize()

    rth_mask = (df["_mod"] >= self.rth_start_min) & (df["_mod"] < self.rth_end_min)
    rth = df[rth_mask]

    day_high = rth.groupby("_date")["high"].max().rename("day_high")
    day_low = rth.groupby("_date")["low"].min().rename("day_low")
    day_volume = rth.groupby("_date")["volume"].sum().rename("volume")

    # Inner join onto the resolved index: only resolved dates pass through.
    day = (
      resolved.join(day_high, how="inner")
      .join(day_low, how="inner")
      .join(day_volume, how="inner")
    )
    if day.empty:
      self._full_day_table = empty
      return empty

    day["range"] = day["day_high"] - day["day_low"]
    day["range_pct"] = day["range"] / day["session_open"]

    table = day[columns]
    self._full_day_table = table
    return table

  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the three magnitude rows over a (possibly sliced) day subset.

    Each row carries its average metric in ``value``; the ``probability`` channel
    is left at ``0.0``. Every row reports its sample size in ``count`` / ``total``.

    If ``baseline_rows`` is provided, the matching row's ``value`` is merged into
    each row's ``value_baseline``.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    total = len(day_table)

    rows: list[StatResultRow] = []
    for out_key, column in _OUTCOMES:
      mean_value = float(day_table[column].astype(float).mean()) if total > 0 else 0.0
      bl = baseline_map.get((_CONDITION, out_key))
      rows.append(
        StatResultRow(
          condition=_CONDITION,
          outcome=out_key,
          count=total,
          total=total,
          probability=0.0,
          baseline_prob=0.0,
          baseline_n=bl.total if bl else 0,
          value=mean_value,
          value_baseline=bl.value if bl else None,
        )
      )
    return rows

  def classify_samples(self, day_table: pd.DataFrame) -> list[SampleRow]:
    """Per-day SampleRows mirroring the three outcomes ``compute_rows`` aggregates.

    Every resolved day (all rows are already countable; pending-sample discipline
    is enforced in ``build_day_table``) yields one SampleRow per outcome in
    ``_OUTCOMES`` — ``mean_volume``, ``mean_range``, ``mean_range_pct`` — each
    carrying that day's own metric as ``value``. All three outcomes cover every
    day (they are not mutually exclusive), so each day emits exactly three
    samples, all under the single condition ``any_day``.
    """
    if day_table.empty:
      return []

    samples: list[SampleRow] = []
    for ts, row in day_table.iterrows():
      date_str = ts.strftime("%Y-%m-%d")
      for out_key, column in _OUTCOMES:
        samples.append(
          SampleRow(
            date=date_str,
            condition=_CONDITION,
            outcome=out_key,
            value=float(row[column]),
          )
        )
    return samples

  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random baseline: a random sample of N days drawn from all resolved days.

    For a subset of ``N`` days (one weekday, or the whole table), draw ``N`` days
    uniformly at random — without replacement — from the **full** resolved-day
    table, and compute the same three averages on that random sample. This is the
    null hypothesis that the weekday carries no information: its expected metrics
    equal the grand mean across all days. Uses ``np.random.default_rng`` with a
    fixed seed for deterministic output.

    Falls back to the passed ``day_table`` as the population when the full table
    has not been stashed (e.g. ``baseline_rows`` called in isolation).
    """
    population = self._full_day_table
    if population is None:
      population = day_table

    n = len(day_table)
    if n == 0 or len(population) == 0:
      return self.compute_rows(day_table.iloc[0:0], baseline_rows=None)

    rng = np.random.default_rng(seed)
    # Sample N rows from the population. The subset is always a sub-population, so
    # sampling without replacement is well-defined (n <= len(population)).
    sample_size = min(n, len(population))
    idx = rng.choice(len(population), size=sample_size, replace=False)
    sample = population.iloc[idx]
    return self.compute_rows(sample, baseline_rows=None)


def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
) -> Path:
  """Load data and compute Volume & Range by Weekday for the daily timeframe.

  Writes the result JSON to results/.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  stat = VolumeRangeByWeekday(instrument=instrument, config=config)
  result = stat.compute(candles_df)

  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Volume & Range by Weekday stat")
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

  # Print a brief summary of results
  with open(output_path, encoding="utf-8") as f:
    data = json.load(f)

  instrument_data = data["instruments"][args.instrument]
  for tf, tf_data in instrument_data.items():
    print(f"\n  {tf}: {tf_data['total_samples']} resolved days | {tf_data['data_range']}")
    weekday = tf_data["slices"].get("weekday", {}).get("groups", {})
    for day_key, group in weekday.items():
      rows = {r["outcome"]: r for r in group["results"]}
      vol = rows["mean_volume"]["value"]
      rng_pts = rows["mean_range"]["value"]
      rng_pct = rows["mean_range_pct"]["value"]
      print(
        f"    {day_key}: avg volume={vol:,.0f} "
        f"| avg range={rng_pts:.2f} pts ({rng_pct:.4%}) (N={group['total_samples']})"
      )
