"""Green & Red Streaks stat.

Measures: once a period is green (or red), how likely is the streak to continue
for one more period of the same color?

The issue frames this as "consecutive green and red periods" with average and max
streak lengths. We express it as a **continuation probability**: among periods of
a given color that have a following resolved period, how often is that next period
the same color. This fits the probability-row framework natively (each row carries
its sample size and a random baseline). The average streak length is recoverable by
consumers as ``1 / (1 - P(continue))``.

A "period" is a green/red candle at one of three granularities:
  - ``daily``: the RTH daily candle (session open to session close).
  - ``weekly``: trading days aggregated by ISO week (week_open = first day's
    session_open, week_close = last day's session_close).
  - ``monthly``: trading days aggregated by calendar month (analogous).

Color is controlled by ``performance``:
  - ``close_to_close`` (default): period close vs the PREVIOUS resolved period's
    close. The first resolved period has no prior close and is excluded.
  - ``open_to_close``: period close vs that same period's open.

Pending discipline: the final resolved period has no following period, so its
continuation outcome is unresolved and it is excluded from every denominator.
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
  en="Green & Red Streaks",
  fr="Séries vertes et rouges",
)
_DEFINITION = I18nString(
  en="Once a period is green (or red), how likely is the streak to continue for one more period of the same color?",
  fr="Une fois qu'une période est verte (ou rouge), quelle est la probabilité que la série se poursuive d'une période supplémentaire de la même couleur ?",
)
_LABELS = Labels(
  conditions={
    "green": I18nString(en="Green period", fr="Période verte"),
    "red": I18nString(en="Red period", fr="Période rouge"),
  },
  outcomes={
    "continue": I18nString(en="Streak continues", fr="La série se poursuit"),
    "break": I18nString(en="Streak breaks", fr="La série se rompt"),
  },
)

# Performance modes: how a period's direction (green/red) is determined.
_PERFORMANCE_MODES = ("close_to_close", "open_to_close")

# Granularities computed and written under instruments.{INSTRUMENT}.{granularity}.
_GRANULARITIES = ("daily", "weekly", "monthly")


class GreenRedStreaks(BaseStat):
  """Continuation probability of a green/red streak at one granularity."""

  stat_name = "green_red_streaks"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = ()

  def __init__(
    self,
    instrument: str,
    config: InstrumentConfig,
    granularity: str = "daily",
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
    # Optional precomputed resolved-days table shared across granularities (see
    # run()). When set, build_day_table reuses it instead of rescanning candles.
    self.daily_table: pd.DataFrame | None = None

    rth = config.sessions["rth"]
    self.rth_start_min: int = minute_of_day(rth.start)
    self.rth_end_min: int = minute_of_day(rth.end)

  # -------------------------------------------------------------------------
  # Day table construction
  # -------------------------------------------------------------------------
  def _aggregate_periods(self, daily: pd.DataFrame) -> pd.DataFrame:
    """Collapse resolved days into periods at the configured granularity.

    Returns a frame indexed by the period's first session date, with columns
    period_open (first day's session_open) and period_close (last day's
    session_close). For daily, each day is its own period.
    """
    if daily.empty:
      return pd.DataFrame(columns=["period_open", "period_close"])

    if self.granularity == "daily":
      return daily.rename(
        columns={"session_open": "period_open", "session_close": "period_close"}
      )

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
    """Build the per-period table with color and the next period's color.

    Columns:
      period_open, period_close, period_green (bool), next_green (1.0/0.0/NaN).

    ``next_green`` is the color of the chronologically following resolved period;
    it is NaN for the last period (unresolved continuation, excluded downstream).
    In close_to_close mode the first period has no prior close and is dropped.

    The resolved-days table is built from ``candles_df`` unless one is supplied
    (via the ``daily_table`` argument or the ``self.daily_table`` attribute),
    which lets ``run()`` scan the candles once for all three granularities.
    """
    columns = ["period_open", "period_close", "period_green", "next_green"]
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
    if self.performance == "open_to_close":
      periods["period_green"] = periods["period_close"] >= periods["period_open"]
    else:  # close_to_close
      periods["prev_close"] = periods["period_close"].shift(1)
      periods = periods[periods["prev_close"].notna()].copy()
      periods["period_green"] = periods["period_close"] >= periods["prev_close"]

    periods["next_green"] = periods["period_green"].astype(float).shift(-1)
    return periods[columns]

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the four continuation rows (green/red × continue/break).

    Only periods with a following resolved period (next_green not NaN) count.
    For a green period, "continue" means the next period is also green; for a
    red period, "continue" means the next period is also red.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    if len(day_table) > 0:
      period_green = day_table["period_green"].astype(bool)
      next_green = day_table["next_green"]
      countable = next_green.notna()
      next_is_green = next_green == 1.0
    else:
      period_green = pd.Series([], dtype=bool)
      countable = pd.Series([], dtype=bool)
      next_is_green = pd.Series([], dtype=bool)

    rows: list[StatResultRow] = []
    for cond_key, cond_is_green in (("green", True), ("red", False)):
      cond_mask = countable & (period_green == cond_is_green)
      total = int(cond_mask.sum())
      # "continue" = next period shares the current period's color.
      next_same = next_is_green if cond_is_green else ~next_is_green
      cont_count = int((cond_mask & next_same).sum())
      outcomes = [
        ("continue", cont_count),
        ("break", total - cont_count),
      ]
      for out_key, count in outcomes:
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

  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random baseline: independent green/red colors (p=0.5) destroy any streak.

    Each period is recolored at random, then the next-period color is recomputed
    from the randomized chronological sequence, so continuation is independent of
    the current color. Expected baseline_prob ≈ 0.5. Deterministic for a fixed seed.
    """
    rng = np.random.default_rng(seed)
    n = len(day_table)
    random_green = rng.integers(0, 2, size=n).astype(bool)

    tmp = day_table.copy()
    tmp["period_green"] = random_green
    tmp["next_green"] = pd.Series(random_green.astype(float), index=tmp.index).shift(-1)
    return self.compute_rows(tmp, baseline_rows=None)


def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
  performance: str = "close_to_close",
) -> Path:
  """Compute Green & Red Streaks for daily, weekly, and monthly granularities.

  Merges the three per-granularity TimeframeResults into a single StatRunResult
  and writes the consolidated JSON to results/.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  # Scan the 1-min candles into the resolved-days table once, then reuse it for
  # every granularity (each merely re-aggregates the same daily candles).
  stats = [
    GreenRedStreaks(
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
  for stat in stats:
    stat.daily_table = daily_table
    result = stat.compute(candles_df)
    merged_tf[stat.granularity] = result.instruments[instrument][stat.granularity]

  final_result = StatRunResult(
    stat_name="green_red_streaks",
    title=_TITLE,
    definition=_DEFINITION,
    labels=_LABELS,
    instruments={instrument: merged_tf},
  )

  return write_results(final_result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Green & Red Streaks stat")
  parser.add_argument("--instrument", default="NQ", help="Instrument name (default: NQ)")
  parser.add_argument("--data-path", default=None, help="Override parquet file path")
  parser.add_argument("--config-dir", default="config", help="Config directory (default: config)")
  parser.add_argument(
    "--performance",
    default="close_to_close",
    choices=list(_PERFORMANCE_MODES),
    help="Period direction basis (default: close_to_close)",
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
    for row in tf_data["results"]:
      print(
        f"    {row['condition']} -> {row['outcome']}: "
        f"{row['probability']:.3f} (N={row['total']}, baseline={row['baseline_prob']:.3f})"
      )
