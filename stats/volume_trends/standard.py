"""Volume Trends stat.

Measures historical volume seasonality: the average traded volume per calendar
month and per ISO week of the year.

Raw 1-min OHLCV is collapsed into resolved RTH days, each carrying its summed
session volume. Days are then aggregated into **periods** at the configured
granularity:
  - ``monthly`` granularity, sliced by **month of year** (January…December):
    "what is the typical volume of a January, averaged across years?"
  - ``weekly`` granularity, sliced by **week of year** (ISO week 1…53).

A period's volume is the **sum** of its resolved days' RTH volume (the month's or
week's total traded volume). The framework's slice then averages those period
totals across years for each calendar position.

This is a **magnitude** stat: the single ``mean_volume`` outcome carries its
metric in ``StatResultRow.value`` (and its random baseline in ``value_baseline``);
the ordinary ``probability`` channel is left at ``0.0``.

Pending-sample discipline: the most recent period is **excluded**. Data typically
ends mid-month/mid-week, so that period's total is incomplete and would skew the
average downward. Resolved days
already drop the final incomplete trading day upstream.

The two granularities are merged into a single result file under
``instruments.{INSTRUMENT}.{granularity}``.
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
  StatRunResult,
  TimeframeResult,
  write_results,
)
from stats.config import InstrumentConfig, load_config, minute_of_day
from stats.seasonality.standard import MonthOfYear, WeekOfYear
from stats.utils.daily_candles import build_resolved_days

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="Volume Trends",
  fr="Tendances de volume",
)
_DEFINITION = I18nString(
  en="What is the average traded volume for each calendar month and each ISO week of the year?",
  fr="Quel est le volume échangé moyen pour chaque mois civil et chaque semaine ISO de l'année ?",
)
_LABELS = Labels(
  conditions={
    "any_period": I18nString(en="All periods", fr="Toutes les périodes"),
  },
  outcomes={
    "mean_volume": I18nString(en="Average volume", fr="Volume moyen"),
  },
)

# Granularities computed and written under instruments.{INSTRUMENT}.{granularity}.
_GRANULARITIES = ("monthly", "weekly")

# Single condition: every resolved period belongs to it; the month/week-of-year
# slice does the seasonal breakdown.
_CONDITION = "any_period"

# Single magnitude outcome carried in the `value` channel.
_OUTCOME = "mean_volume"


class VolumeTrends(BaseStat):
  """Average traded volume per period, sliced by position in the year.

  One instance computes a single granularity (``monthly`` or ``weekly``); the
  declared slice (month-of-year or week-of-year) is chosen accordingly.
  """

  stat_name = "volume_trends"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS

  def __init__(
    self,
    instrument: str,
    config: InstrumentConfig,
    granularity: str = "monthly",
    close_tolerance_min: int = 15,
  ) -> None:
    if granularity not in _GRANULARITIES:
      raise ValueError(
        f"Unsupported granularity '{granularity}'. Choose from {list(_GRANULARITIES)}"
      )
    self.instrument = instrument
    self.granularity = granularity
    self.timeframe = granularity
    self.config = config
    self.close_tolerance_min = close_tolerance_min
    # The seasonal breakdown depends on the granularity: monthly periods split by
    # month-of-year, weekly periods by week-of-year.
    self.slices = (MonthOfYear(),) if granularity == "monthly" else (WeekOfYear(),)
    # Optional precomputed resolved-days-with-volume table shared across
    # granularities (see run()). When set, build_day_table reuses it instead of
    # rescanning the candles.
    self.daily_table: pd.DataFrame | None = None
    # Full period table, stashed by build_day_table so the per-slice random
    # baseline can sample from the whole population of periods (see baseline_rows).
    self._full_day_table: pd.DataFrame | None = None

    rth = config.sessions["rth"]
    self.rth_start_min: int = minute_of_day(rth.start)
    self.rth_end_min: int = minute_of_day(rth.end)

  # -------------------------------------------------------------------------
  # Period table construction
  # -------------------------------------------------------------------------
  def _aggregate_periods(self, daily: pd.DataFrame) -> pd.DataFrame:
    """Collapse resolved days into periods at the configured granularity.

    ``daily`` is indexed by the normalized session date and carries a ``volume``
    column. Returns a frame indexed by each period's first session date, with a
    single ``volume`` column holding that period's summed RTH volume. The index
    date carries the period's calendar position, so a month-of-year /
    week-of-year slicer can read it directly.

    The most recent period is dropped (pending-sample discipline): data usually
    ends mid-period, so its total is incomplete and would bias the average.
    """
    if daily.empty:
      return pd.DataFrame(columns=["volume"])

    idx = daily.index
    if self.granularity == "weekly":
      iso = idx.isocalendar()
      key = list(zip(iso["year"].to_numpy(), iso["week"].to_numpy()))
    else:  # monthly
      key = list(zip(idx.year.to_numpy(), idx.month.to_numpy()))

    tmp = daily.copy()
    tmp["_key"] = key
    tmp["_date"] = idx
    agg = tmp.groupby("_key", sort=False).agg(
      volume=("volume", "sum"),
      period_date=("_date", "min"),
    )
    periods = agg.set_index("period_date").sort_index()
    # Drop the most recent period: it is the one likely unresolved at end of data.
    return periods.iloc[:-1]

  def build_day_table(
    self,
    candles_df: pd.DataFrame,
    daily_table: pd.DataFrame | None = None,
  ) -> pd.DataFrame:
    """Build the per-period table: one row per period with its summed volume.

    Each row is one resolved period, indexed by the period's first session date,
    with a single ``volume`` column.

    The resolved-days-with-volume table is built from ``candles_df`` unless one
    is supplied (via the ``daily_table`` argument or the ``self.daily_table``
    attribute), which lets ``run()`` scan the candles once for both granularities.

    The full period table is stashed on ``self._full_day_table`` so the random
    baseline can sample from the whole population of resolved periods.
    """
    columns = ["volume"]
    if daily_table is None:
      daily_table = self.daily_table
    if daily_table is None:
      daily_table = daily_volume_table(
        candles_df, self.rth_start_min, self.rth_end_min, self.close_tolerance_min
      )
    periods = self._aggregate_periods(daily_table)
    table = periods[columns] if not periods.empty else pd.DataFrame(columns=columns)
    self._full_day_table = table
    return table

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the single magnitude row over a (possibly sliced) period subset.

    The row carries the average period volume in ``value``; the ``probability``
    channel is left at ``0.0``. ``count`` / ``total`` report the sample size.

    If ``baseline_rows`` is provided, the matching row's ``value`` is merged into
    ``value_baseline``.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    total = len(day_table)
    mean_volume = float(day_table["volume"].astype(float).mean()) if total > 0 else 0.0

    bl = baseline_map.get((_CONDITION, _OUTCOME))
    return [
      StatResultRow(
        condition=_CONDITION,
        outcome=_OUTCOME,
        count=total,
        total=total,
        probability=0.0,
        baseline_prob=0.0,
        baseline_n=bl.total if bl else 0,
        value=mean_volume,
        value_baseline=bl.value if bl else None,
        agg="mean",
      )
    ]

  def classify_samples(self, day_table: pd.DataFrame) -> list[SampleRow]:
    """One SampleRow per resolved period, mirroring the single ``mean_volume``
    outcome ``compute_rows()`` averages.

    Each row of ``day_table`` is one resolved period (month or week); every
    period contributes exactly one sample under the single ``any_period``
    condition, carrying that period's summed RTH volume in ``value`` (the
    same per-period figure ``compute_rows()`` averages into ``mean_volume``).
    The sample's ``date`` is the period's first session date, matching the
    table's index.
    """
    if day_table.empty:
      return []

    return [
      SampleRow(
        date=ts.strftime("%Y-%m-%d"),
        condition=_CONDITION,
        outcome=_OUTCOME,
        value=float(volume),
      )
      for ts, volume in zip(day_table.index, day_table["volume"])
    ]

  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random baseline: a random sample of N periods drawn from all periods.

    For a subset of ``N`` periods (one month/week of year, or the whole table),
    draw ``N`` periods uniformly at random — without replacement — from the
    **full** period table, and compute the same average on that random sample.
    This is the null hypothesis that the calendar position carries no information:
    its expected volume equals the grand mean across all periods. Uses
    ``np.random.default_rng`` with a fixed seed for deterministic output.

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
    # The subset is always a sub-population, so sampling without replacement is
    # well-defined (n <= len(population)).
    sample_size = min(n, len(population))
    idx = rng.choice(len(population), size=sample_size, replace=False)
    sample = population.iloc[idx]
    return self.compute_rows(sample, baseline_rows=None)


def daily_volume_table(
  candles_df: pd.DataFrame,
  rth_start_min: int,
  rth_end_min: int,
  close_tolerance_min: int,
) -> pd.DataFrame:
  """Resolved trading days with their summed RTH volume.

  Returns a DataFrame indexed by the normalized session date with a single
  ``volume`` column (sum of the day's RTH bars' volume), restricted to resolved
  days via ``build_resolved_days``.
  """
  empty = pd.DataFrame(columns=["volume"])
  if candles_df.empty:
    return empty

  resolved = build_resolved_days(
    candles_df, rth_start_min, rth_end_min, close_tolerance_min
  )
  if resolved.empty:
    return empty

  df = candles_df.copy()
  df["_mod"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
  df["_date"] = df["timestamp"].dt.normalize()

  rth_mask = (df["_mod"] >= rth_start_min) & (df["_mod"] < rth_end_min)
  day_volume = df[rth_mask].groupby("_date")["volume"].sum().rename("volume")

  # Inner join onto the resolved index: only resolved dates pass through.
  day = resolved.join(day_volume, how="inner")
  if day.empty:
    return empty
  return day[["volume"]]


def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
) -> Path:
  """Compute Volume Trends for the monthly and weekly granularities.

  Merges the per-granularity TimeframeResults into a single StatRunResult and
  writes the consolidated JSON to results/.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  # Scan the 1-min candles into the resolved-days-with-volume table once, then
  # reuse it for both granularities (each merely re-aggregates the same days).
  stats = [
    VolumeTrends(instrument=instrument, config=config, granularity=granularity)
    for granularity in _GRANULARITIES
  ]
  ref = stats[0]
  daily_table = daily_volume_table(
    candles_df, ref.rth_start_min, ref.rth_end_min, ref.close_tolerance_min
  )

  merged_tf: dict[str, TimeframeResult] = {}
  dimensions: dict[str, I18nString] = {}
  for stat in stats:
    stat.daily_table = daily_table
    result = stat.compute(candles_df)
    merged_tf[stat.granularity] = result.instruments[instrument][stat.granularity]
    dimensions.update(result.labels.dimensions)

  labels = _LABELS.model_copy(deep=True)
  labels.dimensions = dimensions

  final_result = StatRunResult(
    stat_name="volume_trends",
    title=_TITLE,
    definition=_DEFINITION,
    labels=labels,
    instruments={instrument: merged_tf},
  )

  return write_results(final_result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Volume Trends stat")
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
    print(f"\n  {tf}: {tf_data['total_samples']} periods | {tf_data['data_range']}")
    dimension = "month_of_year" if tf == "monthly" else "week_of_year"
    groups = tf_data["slices"].get(dimension, {}).get("groups", {})
    for group_key, group in groups.items():
      rows = {r["outcome"]: r for r in group["results"]}
      vol = rows["mean_volume"]["value"]
      print(f"    {group_key}: avg volume={vol:,.0f} (N={group['total_samples']})")
