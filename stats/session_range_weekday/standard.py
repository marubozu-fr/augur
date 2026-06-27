"""Session Range by Weekday stat.

Measures, for each weekday, the average price range (high − low, in points)
of each of three named geographic market sessions — by default ``asia``,
``london``, and ``ny`` — read from the instrument config.

This is a **magnitude** stat: every outcome carries its metric in
``StatResultRow.value`` (and its random baseline in ``value_baseline``). The
ordinary ``probability`` channel is left at ``0.0`` for every row.

A "cycle" is the RTH session date to which bars are attributed.
Cross-midnight sessions (``start >= end``, e.g. ``asia`` 18:00→03:00) have
their evening bars attributed to the NEXT calendar day's cycle. A cycle is
countable only when ALL three sessions are resolved for that cycle — the three
session tables are inner-joined, enforcing the pending-sample discipline
uniformly across all outcomes.
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
  write_results,
)
from stats.config import InstrumentConfig, load_config, minute_of_day
from stats.utils.session_candles import session_bars

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="Session Range by Weekday",
  fr="Amplitude de session par jour de la semaine",
)
_DEFINITION = I18nString(
  en="What is the average price range (high − low, in points) of the Asia, London, and NY sessions for each weekday?",
  fr="Quelle est l'amplitude moyenne (plus haut − plus bas, en points) des sessions asiatique, de Londres et de New York pour chaque jour de la semaine ?",
)
_LABELS = Labels(
  conditions={
    "any_day": I18nString(en="All trading days", fr="Tous les jours de bourse"),
  },
  outcomes={
    "mean_asia_range": I18nString(
      en="Average Asia session range (points)",
      fr="Amplitude moyenne de la session asiatique (points)",
    ),
    "mean_london_range": I18nString(
      en="Average London session range (points)",
      fr="Amplitude moyenne de la session de Londres (points)",
    ),
    "mean_ny_range": I18nString(
      en="Average NY session range (points)",
      fr="Amplitude moyenne de la session de New York (points)",
    ),
  },
)

# Single condition: every resolved cycle belongs to it; the weekday slice does
# the per-day breakdown.
_CONDITION = "any_day"

# Outcome keys and the day-table column carrying each metric, in display order.
_OUTCOMES: list[tuple[str, str]] = [
  ("mean_asia_range", "asia_range"),
  ("mean_london_range", "london_range"),
  ("mean_ny_range", "ny_range"),
]

_MINUTES_PER_DAY = 24 * 60


class SessionRangeByWeekday(BaseStat):
  """Average Asia, London, and NY session ranges, sliced by weekday."""

  stat_name = "session_range_weekday"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = ("weekday",)

  def __init__(
    self,
    instrument: str,
    config: InstrumentConfig,
    asia: str = "asia",
    london: str = "london",
    ny: str = "ny",
    close_tolerance_min: int = 15,
  ) -> None:
    for name in (asia, london, ny):
      if name not in config.sessions:
        raise ValueError(
          f"Unknown session '{name}'. Configured sessions: {list(config.sessions)}"
        )

    self.instrument = instrument
    self.timeframe = "daily"
    self.config = config
    self.asia = asia
    self.london = london
    self.ny = ny
    self.close_tolerance_min = close_tolerance_min

    s_asia = config.sessions[asia]
    self.asia_start_min: int = minute_of_day(s_asia.start)
    self.asia_end_min: int = minute_of_day(s_asia.end)

    s_london = config.sessions[london]
    self.london_start_min: int = minute_of_day(s_london.start)
    self.london_end_min: int = minute_of_day(s_london.end)

    s_ny = config.sessions[ny]
    self.ny_start_min: int = minute_of_day(s_ny.start)
    self.ny_end_min: int = minute_of_day(s_ny.end)

    # Full resolved-cycle table, stashed by build_day_table so the per-slice
    # random baseline can sample from the whole population of cycles.
    self._full_day_table: pd.DataFrame | None = None

  # -------------------------------------------------------------------------
  # Day table construction
  # -------------------------------------------------------------------------
  def _session_table(
    self, df: pd.DataFrame, start_min: int, end_min: int
  ) -> pd.DataFrame:
    """Per-cycle high/low aggregates for one session, indexed by cycle date.

    Columns: ``high``, ``low``. Only resolved cycles are returned — those with
    a clean open bar (offset 0 from session start) AND a last bar at or after
    ``duration - close_tolerance_min`` minutes from session start.
    """
    bars = session_bars(df, start_min, end_min)
    if bars.empty:
      return pd.DataFrame()

    duration = (end_min - start_min) % _MINUTES_PER_DAY or _MINUTES_PER_DAY
    bars["_offset"] = (bars["_mod"] - start_min) % _MINUTES_PER_DAY

    grouped = bars.groupby("_cycle")
    table = pd.DataFrame(
      {
        "high": grouped["high"].max(),
        "low": grouped["low"].min(),
        "_has_open": grouped["_offset"].min() == 0,
        "_last_offset": grouped["_offset"].max(),
      }
    )
    resolved = table["_has_open"] & (
      table["_last_offset"] >= duration - self.close_tolerance_min
    )
    table = table[resolved]
    return table.drop(columns=["_has_open", "_last_offset"])

  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-cycle session-range table from 1-min OHLCV data.

    Each row is one countable cycle — a cycle where ALL three sessions
    (asia, london, ny) are resolved. The table is indexed by the normalized
    RTH cycle date (required by the ``weekday`` slicer) and has columns:
    ``asia_range``, ``london_range``, ``ny_range`` (each = session high − low,
    in points).

    The full table is stashed on ``self._full_day_table`` so the random
    baseline can sample from the whole population of countable cycles.
    """
    columns = ["asia_range", "london_range", "ny_range"]
    empty = pd.DataFrame(columns=columns)

    if candles_df.empty:
      self._full_day_table = empty
      return empty

    # reset_index guarantees unique, positional labels for groupby operations.
    df = candles_df.sort_values("timestamp").reset_index(drop=True)
    df["_mod"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["_date"] = df["timestamp"].dt.normalize()

    asia_tbl = self._session_table(df, self.asia_start_min, self.asia_end_min)
    london_tbl = self._session_table(
      df, self.london_start_min, self.london_end_min
    )
    ny_tbl = self._session_table(df, self.ny_start_min, self.ny_end_min)

    if asia_tbl.empty or london_tbl.empty or ny_tbl.empty:
      self._full_day_table = empty
      return empty

    # Inner join across all three sessions: a cycle is countable only when
    # every session is resolved for that cycle (pending-sample discipline).
    day = (
      pd.DataFrame(
        {"asia_high": asia_tbl["high"], "asia_low": asia_tbl["low"]}
      )
      .join(
        pd.DataFrame(
          {"london_high": london_tbl["high"], "london_low": london_tbl["low"]}
        ),
        how="inner",
      )
      .join(
        pd.DataFrame({"ny_high": ny_tbl["high"], "ny_low": ny_tbl["low"]}),
        how="inner",
      )
    )

    if day.empty:
      self._full_day_table = empty
      return empty

    day = day.sort_index()
    day["asia_range"] = day["asia_high"] - day["asia_low"]
    day["london_range"] = day["london_high"] - day["london_low"]
    day["ny_range"] = day["ny_high"] - day["ny_low"]

    table = day[columns]
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
    """Compute the three magnitude rows over a (possibly sliced) cycle subset.

    Each row carries its average session range in ``value``; the
    ``probability`` channel is left at ``0.0``. Every row reports its sample
    size in ``count`` / ``total``.

    If ``baseline_rows`` is provided, the matching row's ``value`` is merged
    into each row's ``value_baseline``.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    total = len(day_table)

    rows: list[StatResultRow] = []
    for out_key, column in _OUTCOMES:
      mean_value = (
        float(day_table[column].astype(float).mean()) if total > 0 else 0.0
      )
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

  # -------------------------------------------------------------------------
  # Baseline
  # -------------------------------------------------------------------------
  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random baseline: a random sample of N cycles drawn from all countable cycles.

    For a subset of ``N`` cycles (one weekday, or the whole table), draw ``N``
    cycles uniformly at random — without replacement — from the **full**
    countable-cycle table, and compute the same three averages on that random
    sample. This is the null hypothesis that the weekday carries no information:
    its expected session ranges equal the grand mean across all cycles. Uses
    ``np.random.default_rng`` with a fixed seed for deterministic output.

    Falls back to the passed ``day_table`` as the population when the full
    table has not been stashed (e.g. ``baseline_rows`` called in isolation).
    """
    population = self._full_day_table
    if population is None:
      population = day_table

    n = len(day_table)
    if n == 0 or len(population) == 0:
      return self.compute_rows(day_table.iloc[0:0], baseline_rows=None)

    rng = np.random.default_rng(seed)
    sample_size = min(n, len(population))
    idx = rng.choice(len(population), size=sample_size, replace=False)
    sample = population.iloc[idx]
    return self.compute_rows(sample, baseline_rows=None)


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
  asia: str = "asia",
  london: str = "london",
  ny: str = "ny",
) -> Path:
  """Load data and compute Session Range by Weekday for the daily timeframe.

  Writes the result JSON to results/.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  stat = SessionRangeByWeekday(
    instrument=instrument,
    config=config,
    asia=asia,
    london=london,
    ny=ny,
  )
  result = stat.compute(candles_df)

  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Compute Session Range by Weekday stat"
  )
  parser.add_argument(
    "--instrument", default="NQ", help="Instrument name (default: NQ)"
  )
  parser.add_argument(
    "--data-path", default=None, help="Override parquet file path"
  )
  parser.add_argument(
    "--config-dir", default="config", help="Config directory (default: config)"
  )
  parser.add_argument(
    "--asia", default="asia", help="Asia session name in config (default: asia)"
  )
  parser.add_argument(
    "--london",
    default="london",
    help="London session name in config (default: london)",
  )
  parser.add_argument(
    "--ny", default="ny", help="NY session name in config (default: ny)"
  )
  args = parser.parse_args()

  output_path = run(
    instrument=args.instrument,
    config_dir=args.config_dir,
    data_path=args.data_path,
    asia=args.asia,
    london=args.london,
    ny=args.ny,
  )

  print(f"Written: {output_path}")

  # Print a brief summary of results
  with open(output_path, encoding="utf-8") as f:
    data = json.load(f)

  instrument_data = data["instruments"][args.instrument]
  for tf, tf_data in instrument_data.items():
    print(
      f"\n  {tf}: {tf_data['total_samples']} countable cycles | {tf_data['data_range']}"
    )
    weekday = tf_data["slices"].get("weekday", {}).get("groups", {})
    for day_key, group in weekday.items():
      rows = {r["outcome"]: r for r in group["results"]}
      asia_r = rows["mean_asia_range"]["value"]
      london_r = rows["mean_london_range"]["value"]
      ny_r = rows["mean_ny_range"]["value"]
      print(
        f"    {day_key}: asia={asia_r:.2f} pts"
        f" | london={london_r:.2f} pts"
        f" | ny={ny_r:.2f} pts"
        f" (N={group['total_samples']})"
      )
