"""Seasonality stat.

Measures month-of-year and week-of-year average performance patterns from
monthly and weekly close-to-close returns.

For each period (a calendar month or an ISO week) the signed percent return is
computed, then the periods are grouped by their position in the year:
  - ``monthly`` granularity, sliced by **month of year** (January…December):
    "what does a typical January return, averaged across years?"
  - ``weekly`` granularity, sliced by **week of year** (ISO week 1…53).

Like Performance by Weekday, this is a **magnitude** stat: three of the five
outcome rows carry a continuous metric in ``StatResultRow.value`` (a signed
decimal return, e.g. ``0.012`` = +1.2%) and its random baseline in
``value_baseline``; the green/red period count rows use the ordinary
``probability`` channel.

A period's percent return is controlled by ``performance``:
  - ``close_to_close`` (default): (period_close - PREVIOUS resolved period's
    close) / previous close. The first resolved period has no prior close and is
    excluded (pending-sample discipline).
  - ``open_to_close``: (period_close - period_open) / period_open.

A period is **green** when its return is >= 0, otherwise **red**.

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
  SliceGroup,
  Slicer,
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
  en="Seasonality",
  fr="Saisonnalité",
)
_DEFINITION = I18nString(
  en="What is the average percent return and green/red period split for each month of the year and each week of the year?",
  fr="Quels sont le rendement moyen en pourcentage et la répartition des périodes vertes/rouges pour chaque mois de l'année et chaque semaine de l'année ?",
)
_LABELS = Labels(
  conditions={
    "any_period": I18nString(en="All periods", fr="Toutes les périodes"),
  },
  outcomes={
    "mean_return": I18nString(en="Average return", fr="Rendement moyen"),
    "green_period": I18nString(en="Green period (up)", fr="Période verte (hausse)"),
    "red_period": I18nString(en="Red period (down)", fr="Période rouge (baisse)"),
    "mean_green_move": I18nString(en="Average green move", fr="Mouvement vert moyen"),
    "mean_red_move": I18nString(en="Average red move", fr="Mouvement rouge moyen"),
  },
)

# Performance modes: how a period's percent return is computed.
_PERFORMANCE_MODES = ("close_to_close", "open_to_close")

# Granularities computed and written under instruments.{INSTRUMENT}.{granularity}.
_GRANULARITIES = ("monthly", "weekly")

# Single condition: every resolved period belongs to it; the month/week-of-year
# slice does the seasonal breakdown.
_CONDITION = "any_period"

# Outcomes whose payload is a continuous metric in `value` (not a probability).
_MAGNITUDE_OUTCOMES = frozenset({"mean_return", "mean_green_move", "mean_red_move"})

_MONTHS: list[tuple[int, str, str, str]] = [
  (1, "january", "January", "Janvier"),
  (2, "february", "February", "Février"),
  (3, "march", "March", "Mars"),
  (4, "april", "April", "Avril"),
  (5, "may", "May", "Mai"),
  (6, "june", "June", "Juin"),
  (7, "july", "July", "Juillet"),
  (8, "august", "August", "Août"),
  (9, "september", "September", "Septembre"),
  (10, "october", "October", "Octobre"),
  (11, "november", "November", "Novembre"),
  (12, "december", "December", "Décembre"),
]


class MonthOfYear(Slicer):
  """Group periods by calendar month (January…December).

  Reads the DatetimeIndex of the period table. Used with monthly periods so each
  group averages, e.g., every January across all years in the data.
  """

  name = "month_of_year"

  def dimension_label(self) -> I18nString:
    return I18nString(en="Month of year", fr="Mois de l'année")

  def split(self, day_table: pd.DataFrame) -> list[SliceGroup]:
    if len(day_table) == 0:
      return []
    month = day_table.index.month
    groups: list[SliceGroup] = []
    for num, key, en, fr in _MONTHS:
      mask = pd.Series(month == num, index=day_table.index)
      if bool(mask.any()):
        groups.append(SliceGroup(key=key, label=I18nString(en=en, fr=fr), mask=mask))
    return groups


class WeekOfYear(Slicer):
  """Group periods by ISO week number (1…53).

  Reads the DatetimeIndex of the period table. Used with weekly periods so each
  group averages, e.g., every ISO week 1 across all years in the data.
  """

  name = "week_of_year"

  def dimension_label(self) -> I18nString:
    return I18nString(en="Week of year", fr="Semaine de l'année")

  def split(self, day_table: pd.DataFrame) -> list[SliceGroup]:
    if len(day_table) == 0:
      return []
    week = day_table.index.isocalendar()["week"].to_numpy()
    groups: list[SliceGroup] = []
    for wk in range(1, 54):
      mask = pd.Series(week == wk, index=day_table.index)
      if bool(mask.any()):
        groups.append(
          SliceGroup(
            key=f"w{wk:02d}",
            label=I18nString(en=f"Week {wk}", fr=f"Semaine {wk}"),
            mask=mask,
          )
        )
    return groups


class Seasonality(BaseStat):
  """Average return and green/red split per period, sliced by position in year.

  One instance computes a single granularity (``monthly`` or ``weekly``); the
  declared slice (month-of-year or week-of-year) is chosen accordingly.
  """

  stat_name = "seasonality"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS

  def __init__(
    self,
    instrument: str,
    config: InstrumentConfig,
    granularity: str = "monthly",
    performance: str = "close_to_close",
    close_tolerance_min: int = 15,
  ) -> None:
    if granularity not in _GRANULARITIES:
      raise ValueError(
        f"Unsupported granularity '{granularity}'. Choose from {list(_GRANULARITIES)}"
      )
    if performance not in _PERFORMANCE_MODES:
      raise ValueError(
        f"Unsupported performance '{performance}'. Choose from {list(_PERFORMANCE_MODES)}"
      )
    self.instrument = instrument
    self.granularity = granularity
    self.timeframe = granularity
    self.config = config
    self.performance = performance
    self.close_tolerance_min = close_tolerance_min
    # The seasonal breakdown depends on the granularity: monthly periods split by
    # month-of-year, weekly periods by week-of-year.
    self.slices = (MonthOfYear(),) if granularity == "monthly" else (WeekOfYear(),)
    # Optional precomputed resolved-days table shared across granularities (see
    # run()). When set, build_day_table reuses it instead of rescanning candles.
    self.daily_table: pd.DataFrame | None = None

    rth = config.sessions["rth"]
    self.rth_start_min: int = minute_of_day(rth.start)
    self.rth_end_min: int = minute_of_day(rth.end)

  # -------------------------------------------------------------------------
  # Period table construction
  # -------------------------------------------------------------------------
  def _aggregate_periods(self, daily: pd.DataFrame) -> pd.DataFrame:
    """Collapse resolved days into periods at the configured granularity.

    Returns a frame indexed by the period's first session date, with columns
    period_open (first day's session_open) and period_close (last day's
    session_close). The index date carries the period's calendar position, so a
    month-of-year / week-of-year slicer can read it directly.
    """
    if daily.empty:
      return pd.DataFrame(columns=["period_open", "period_close"])

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
      period_open=("session_open", "first"),
      period_close=("session_close", "last"),
      period_date=("_date", "min"),
    )
    return agg.set_index("period_date").sort_index()

  def build_day_table(
    self,
    candles_df: pd.DataFrame,
    daily_table: pd.DataFrame | None = None,
  ) -> pd.DataFrame:
    """Build the per-period table with each period's signed return and color.

    Each row is one resolved period, indexed by the period's first session date,
    with columns:
      period_open, period_close, prev_period_close, return_pct, period_green

    ``return_pct`` is the signed decimal return per the ``performance`` mode;
    ``period_green`` is ``return_pct >= 0``.

    The resolved-days table is built from ``candles_df`` unless one is supplied
    (via the ``daily_table`` argument or the ``self.daily_table`` attribute),
    which lets ``run()`` scan the candles once for both granularities.

    In ``close_to_close`` mode the first resolved period has no previous close
    and is excluded (pending-sample discipline).
    """
    columns = ["period_open", "period_close", "prev_period_close", "return_pct", "period_green"]
    if daily_table is None:
      daily_table = self.daily_table
    if daily_table is None:
      daily_table = build_resolved_days(
        candles_df, self.rth_start_min, self.rth_end_min, self.close_tolerance_min
      )
    periods = self._aggregate_periods(daily_table)
    if periods.empty:
      return pd.DataFrame(columns=columns)

    periods = periods.sort_index()
    # _aggregate_periods returns periods in chronological order, so prev_period_close
    # references the previous resolved period.
    periods["prev_period_close"] = periods["period_close"].shift(1)

    if self.performance == "open_to_close":
      periods["return_pct"] = (
        (periods["period_close"] - periods["period_open"]) / periods["period_open"]
      )
    else:  # close_to_close
      # First resolved period has no prior close: excluded (pending discipline).
      periods = periods[periods["prev_period_close"].notna()].copy()
      periods["return_pct"] = (
        (periods["period_close"] - periods["prev_period_close"]) / periods["prev_period_close"]
      )

    periods["period_green"] = periods["return_pct"] >= 0
    return periods[columns]

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the five outcome rows for one (possibly sliced) period subset.

    Magnitude rows (``mean_return``, ``mean_green_move``, ``mean_red_move``) carry
    their metric in ``value``; the count rows (``green_period``, ``red_period``)
    use ``probability``. Every row reports its sample size in ``count`` / ``total``.

    If ``baseline_rows`` is provided, merges each row's baseline into the matching
    ``baseline_prob`` / ``baseline_n`` (probability rows) and ``value_baseline``
    (magnitude rows).
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    total = len(day_table)
    if total > 0:
      returns = day_table["return_pct"].astype(float)
      green_mask = day_table["period_green"].astype(bool)
    else:
      returns = pd.Series([], dtype=float)
      green_mask = pd.Series([], dtype=bool)

    green_count = int(green_mask.sum())
    red_count = total - green_count

    mean_return = float(returns.mean()) if total > 0 else 0.0
    mean_green = float(returns[green_mask].mean()) if green_count > 0 else 0.0
    mean_red = float(returns[~green_mask].mean()) if red_count > 0 else 0.0

    # (outcome, count, probability, value, agg) — value/agg are None for
    # probability rows.
    specs: list[tuple[str, int, float, float | None, str | None]] = [
      ("mean_return", total, 0.0, mean_return, "mean"),
      ("green_period", green_count, green_count / total if total > 0 else 0.0, None, None),
      ("red_period", red_count, red_count / total if total > 0 else 0.0, None, None),
      ("mean_green_move", green_count, 0.0, mean_green, "mean"),
      ("mean_red_move", red_count, 0.0, mean_red, "mean"),
    ]

    rows: list[StatResultRow] = []
    for out_key, count, probability, value, agg in specs:
      bl = baseline_map.get((_CONDITION, out_key))
      rows.append(
        StatResultRow(
          condition=_CONDITION,
          outcome=out_key,
          count=count,
          total=total,
          probability=probability,
          baseline_prob=bl.probability if bl else 0.0,
          baseline_n=bl.total if bl else 0,
          value=value,
          value_baseline=(bl.value if bl else None) if value is not None else None,
          agg=agg,
        )
      )
    return rows

  def classify_samples(self, day_table: pd.DataFrame) -> list[SampleRow]:
    """Per-period SampleRows mirroring the five outcomes ``compute_rows`` aggregates.

    Every resolved period (all rows are already countable; pending-sample
    discipline is enforced in ``build_day_table``) yields:
      - one ``mean_return`` sample carrying that period's ``return_pct`` as
        ``value`` (every period counts toward this outcome);
      - one ``green_period`` OR ``red_period`` sample (mutually exclusive, no
        ``value``);
      - one ``mean_green_move`` sample (only green periods) OR
        ``mean_red_move`` sample (only red periods), again carrying
        ``return_pct`` as ``value``.
    All samples share the single condition ``any_period``. ``date`` is the
    period's index date (its first session date).
    """
    if day_table.empty:
      return []

    samples: list[SampleRow] = []
    for ts, return_pct, period_green in zip(
      day_table.index, day_table["return_pct"], day_table["period_green"]
    ):
      date_str = ts.strftime("%Y-%m-%d")
      value = float(return_pct)
      samples.append(
        SampleRow(date=date_str, condition=_CONDITION, outcome="mean_return", value=value)
      )
      if period_green:
        samples.append(SampleRow(date=date_str, condition=_CONDITION, outcome="green_period"))
        samples.append(
          SampleRow(date=date_str, condition=_CONDITION, outcome="mean_green_move", value=value)
        )
      else:
        samples.append(SampleRow(date=date_str, condition=_CONDITION, outcome="red_period"))
        samples.append(
          SampleRow(date=date_str, condition=_CONDITION, outcome="mean_red_move", value=value)
        )
    return samples

  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random baseline: each period's return direction is a coin flip.

    Magnitudes are held fixed; only the sign of each period's return is randomized
    (p=0.5). Expected average return ≈ 0; expected green/red period counts ≈ 50/50.
    Uses a fixed seed for deterministic output. Operates directly on the period
    table so the framework can compute a baseline per slice group as well as
    overall.
    """
    rng = np.random.default_rng(seed)
    n = len(day_table)
    tmp = day_table.copy()
    if n > 0:
      sign = rng.integers(0, 2, size=n) * 2 - 1  # ±1
      random_return = sign * tmp["return_pct"].abs().to_numpy()
      tmp["return_pct"] = random_return
      tmp["period_green"] = tmp["return_pct"] >= 0
    return self.compute_rows(tmp, baseline_rows=None)


def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
  performance: str = "close_to_close",
) -> Path:
  """Compute Seasonality for the monthly and weekly granularities.

  Merges the per-granularity TimeframeResults into a single StatRunResult and
  writes the consolidated JSON to results/.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  # Scan the 1-min candles into the resolved-days table once, then reuse it for
  # both granularities (each merely re-aggregates the same daily candles).
  stats = [
    Seasonality(
      instrument=instrument,
      config=config,
      granularity=granularity,
      performance=performance,
    )
    for granularity in _GRANULARITIES
  ]
  ref = stats[0]
  daily_table = build_resolved_days(
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
    stat_name="seasonality",
    title=_TITLE,
    definition=_DEFINITION,
    labels=labels,
    instruments={instrument: merged_tf},
  )

  return write_results(final_result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Seasonality stat")
  parser.add_argument("--instrument", default="NQ", help="Instrument name (default: NQ)")
  parser.add_argument("--data-path", default=None, help="Override parquet file path")
  parser.add_argument("--config-dir", default="config", help="Config directory (default: config)")
  parser.add_argument(
    "--performance",
    default="close_to_close",
    choices=list(_PERFORMANCE_MODES),
    help="Return basis (default: close_to_close)",
  )
  args = parser.parse_args()

  output_path = run(
    instrument=args.instrument,
    config_dir=args.config_dir,
    data_path=args.data_path,
    performance=args.performance,
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
      mean_ret = rows["mean_return"]["value"]
      green = rows["green_period"]
      print(
        f"    {group_key}: avg return={mean_ret:+.4%} "
        f"| P(green)={green['probability']:.3f} (N={green['total']})"
      )
