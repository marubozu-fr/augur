"""CPI Reaction stat.

On CPI release dates, given the initial reaction candle's direction (the
08:30–09:30 ET window that captures the immediate post-release price action),
how often does the RTH day close in the same direction?

A 2×2 conditional matrix:
  - P(green day | green reaction), P(red day | green reaction)
  - P(green day | red reaction),   P(red day | red reaction)

Definitions:
  - Reaction candle color: the [reaction_start, reaction_end) ET window on the
    CPI release date (default 08:30–09:30, capturing the pre-RTH reaction).
      reaction_open  = open of the bar at exactly reaction_start_min.
      reaction_close = close of the last bar in [reaction_start_min, reaction_end_min).
      reaction_green = reaction_close >= reaction_open.
    If the bar at reaction_start_min is missing for a release date, the reaction
    is undefined and the date is dropped (pending-sample discipline).
  - RTH day color: session_close vs session_open (intraday direction).
      green: session_close >= session_open.
    The RTH session must be resolved (clean session-open bar and a last RTH bar
    at or after session_end - close_tolerance_min). Unresolved release dates are
    dropped.

Only release dates that satisfy BOTH requirements (resolved RTH session AND
complete reaction candle) are counted in any denominator.
total_samples = number of qualifying CPI release dates.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
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
from stats.cpi_performance.standard import load_cpi_release_dates
from stats.event_performance_base import _DEFAULT_CALENDAR_PATH
from stats.utils.daily_candles import build_resolved_days

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="CPI Reaction",
  fr="Réaction CPI",
)
_DEFINITION = I18nString(
  en=(
    "On CPI release dates, given the initial reaction candle's direction "
    "(08:30–09:30 ET move after the release), how often does the RTH day "
    "close in the same direction?"
  ),
  fr=(
    "Les jours de publication du CPI, selon la direction de la bougie de "
    "réaction initiale (mouvement 08:30–09:30 ET après la publication), "
    "à quelle fréquence la séance RTH clôture-t-elle dans la même direction ?"
  ),
)
_LABELS = Labels(
  conditions={
    "reaction_green": I18nString(
      en="Green reaction (up)", fr="Réaction verte (hausse)"
    ),
    "reaction_red": I18nString(
      en="Red reaction (down)", fr="Réaction rouge (baisse)"
    ),
  },
  outcomes={
    "green": I18nString(en="Green day (up)", fr="Jour vert (hausse)"),
    "red": I18nString(en="Red day (down)", fr="Jour rouge (baisse)"),
  },
)

# Condition / outcome enumeration: reaction color -> RTH day color.
_CONDITIONS = (("reaction_green", True), ("reaction_red", False))
_OUTCOMES = (("green", True), ("red", False))


class CPIReaction(BaseStat):
  """Conditional probability of the RTH day color given the CPI reaction candle color."""

  stat_name = "cpi_reaction"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = ()  # Events are scattered across the calendar; per-day slicing is meaningless.

  def __init__(
    self,
    instrument: str,
    config: InstrumentConfig,
    event_dates: Iterable[pd.Timestamp],
    reaction_start: str = "08:30",
    reaction_end: str = "09:30",
    close_tolerance_min: int = 15,
  ) -> None:
    self.instrument = instrument
    self.timeframe = "daily"
    self.config = config
    self.close_tolerance_min = close_tolerance_min

    self.reaction_start_min: int = minute_of_day(reaction_start)
    self.reaction_end_min: int = minute_of_day(reaction_end)
    if self.reaction_end_min <= self.reaction_start_min:
      raise ValueError(
        f"reaction_end ({reaction_end}) must be after reaction_start ({reaction_start})"
      )

    rth = config.sessions["rth"]
    self.rth_start_min: int = minute_of_day(rth.start)
    self.rth_end_min: int = minute_of_day(rth.end)

    # Normalize to a set of midnight, tz-naive Timestamps for date alignment.
    self.event_dates: set[pd.Timestamp] = {
      pd.Timestamp(d).normalize().tz_localize(None) for d in event_dates
    }

  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-CPI-release table with reaction and RTH day direction columns.

    Columns: reaction_green (bool), day_green (bool).
    Index = release date (DatetimeIndex, tz-naive normalized Timestamps).

    A release date is included only when BOTH (a) the RTH session is resolved
    AND (b) the reaction candle has a bar at exactly reaction_start_min. Dates
    that fail either requirement are dropped (pending-sample discipline).

    Returns an empty DataFrame with the expected columns when event_dates is
    empty or no dates qualify.
    """
    columns = ["reaction_green", "day_green"]
    empty = pd.DataFrame(columns=columns)

    if not self.event_dates:
      return empty

    # Build resolved RTH days, indexed by normalized date.
    resolved = build_resolved_days(
      candles_df, self.rth_start_min, self.rth_end_min, self.close_tolerance_min
    )
    if resolved.empty:
      return empty

    resolved = resolved.sort_index()

    # Strip timezone from resolved index for alignment with tz-naive event_dates.
    if resolved.index.tz is not None:
      resolved.index = resolved.index.tz_localize(None)

    # RTH day color: session_close >= session_open (intraday direction).
    day_green_series = (
      (resolved["session_close"] >= resolved["session_open"]).rename("day_green")
    )

    # Compute the reaction candle from the 1-min candles.
    df = candles_df.copy()
    df["mod"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute

    # Normalize date to tz-naive midnight for alignment with event_dates.
    ts_norm = df["timestamp"].dt.normalize()
    df["date"] = ts_norm.dt.tz_localize(None) if ts_norm.dt.tz is not None else ts_norm

    # Keep bars in the reaction window: [reaction_start_min, reaction_end_min)
    reaction_mask = (
      (df["mod"] >= self.reaction_start_min) & (df["mod"] < self.reaction_end_min)
    )
    reaction_bars = df[reaction_mask].copy()

    if reaction_bars.empty:
      return empty

    # reaction_open = open of the bar at exactly reaction_start_min.
    open_bars = (
      reaction_bars[reaction_bars["mod"] == self.reaction_start_min]
      .set_index("date")["open"]
      .rename("reaction_open")
    )
    # Defensive: guard against duplicate bars at reaction_start_min for the same day.
    open_bars = open_bars[~open_bars.index.duplicated(keep="first")]

    # reaction_close = close of the last bar in the reaction window.
    last_bars = (
      reaction_bars.loc[
        reaction_bars.groupby("date")["mod"].idxmax(), ["date", "close"]
      ]
      .set_index("date")["close"]
      .rename("reaction_close")
    )

    # Merge open and close; only dates with a bar at reaction_start_min qualify.
    reaction = pd.concat([open_bars, last_bars], axis=1, sort=False)
    # Drop dates missing the open bar (no bar at reaction_start_min → undefined reaction).
    reaction = reaction[reaction["reaction_open"].notna()].copy()
    if reaction.empty:
      return empty

    reaction["reaction_green"] = reaction["reaction_close"] >= reaction["reaction_open"]

    # Restrict to CPI release dates (tz-naive DatetimeIndex for alignment).
    event_index = pd.DatetimeIndex(sorted(self.event_dates))
    reaction = reaction[reaction.index.isin(event_index)]
    day_green_aligned = day_green_series[day_green_series.index.isin(event_index)]

    # Inner join: keep only dates present in BOTH the resolved RTH table and the
    # reaction table. Dates failing either requirement are excluded.
    result = reaction[["reaction_green"]].join(day_green_aligned, how="inner")
    if result.empty:
      return empty

    return result[columns]

  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the four matrix rows (reaction green/red × day green/red).

    All four rows are always emitted (even when the table is empty) so that the
    result shape is deterministic. If baseline_rows is provided, merges
    baseline_prob / baseline_n from it.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    if day_table.empty:
      return [
        StatResultRow(
          condition=cond_key,
          outcome=out_key,
          count=0,
          total=0,
          probability=0.0,
          baseline_prob=0.0,
          baseline_n=0,
        )
        for cond_key, _ in _CONDITIONS
        for out_key, _ in _OUTCOMES
      ]

    reaction_green = day_table["reaction_green"].astype(bool)
    day_green = day_table["day_green"].astype(bool)

    rows: list[StatResultRow] = []
    for cond_key, cond_is_green in _CONDITIONS:
      cond_mask = reaction_green == cond_is_green
      total = int(cond_mask.sum())
      for out_key, out_is_green in _OUTCOMES:
        out_match = day_green if out_is_green else ~day_green
        count = int((cond_mask & out_match).sum())
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
    """One SampleRow per qualifying CPI release date, mirroring ``compute_rows``.

    Every row in the day table is countable (non-qualifying dates are already
    excluded by ``build_day_table``).
    """
    if day_table.empty:
      return []

    samples: list[SampleRow] = []
    for ts, reaction_green, day_green in zip(
      day_table.index, day_table["reaction_green"], day_table["day_green"]
    ):
      condition = "reaction_green" if bool(reaction_green) else "reaction_red"
      outcome = "green" if bool(day_green) else "red"
      samples.append(SampleRow(date=ts.strftime("%Y-%m-%d"), condition=condition, outcome=outcome))
    return samples

  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random baseline: randomize RTH day color (p=0.5) to destroy any correlation.

    The reaction candle condition stays as-is from real data; only ``day_green``
    is replaced with an independent random sequence. Expected baseline_prob ~ 0.5
    for every row. Deterministic for a fixed seed.
    """
    rng = np.random.default_rng(seed)
    n = len(day_table)
    random_green = rng.integers(0, 2, size=n).astype(bool)

    tmp = day_table.copy()
    tmp["day_green"] = random_green
    return self.compute_rows(tmp, baseline_rows=None)


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
  calendar_path: str | Path = _DEFAULT_CALENDAR_PATH,
  reaction_start: str = "08:30",
  reaction_end: str = "09:30",
) -> Path:
  """Load data and the CPI calendar, compute CPI Reaction, write the result."""
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)
  event_dates = load_cpi_release_dates(calendar_path)

  stat = CPIReaction(
    instrument=instrument,
    config=config,
    event_dates=event_dates,
    reaction_start=reaction_start,
    reaction_end=reaction_end,
  )
  result = stat.compute(candles_df)
  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute CPI Reaction stat")
  parser.add_argument("--instrument", default="NQ", help="Instrument name (default: NQ)")
  parser.add_argument("--data-path", default=None, help="Override parquet file path")
  parser.add_argument("--config-dir", default="config", help="Config directory (default: config)")
  parser.add_argument(
    "--calendar-path",
    default=str(_DEFAULT_CALENDAR_PATH),
    help=f"Economic calendar CSV path (default: {_DEFAULT_CALENDAR_PATH})",
  )
  parser.add_argument(
    "--reaction-start",
    default="08:30",
    help="Reaction window start time HH:MM ET (default: 08:30)",
  )
  parser.add_argument(
    "--reaction-end",
    default="09:30",
    help="Reaction window end time HH:MM ET (default: 09:30)",
  )
  args = parser.parse_args()

  output_path = run(
    instrument=args.instrument,
    config_dir=args.config_dir,
    data_path=args.data_path,
    calendar_path=args.calendar_path,
    reaction_start=args.reaction_start,
    reaction_end=args.reaction_end,
  )

  print(f"Written: {output_path}")

  with open(output_path, encoding="utf-8") as f:
    data = json.load(f)

  instrument_data = data["instruments"][args.instrument]
  for tf, tf_data in instrument_data.items():
    print(
      f"\n  {tf}: {tf_data['total_samples']} qualifying CPI days"
      f" | {tf_data['data_range']}"
    )
    for row in tf_data["results"]:
      print(
        f"    {row['condition']} -> {row['outcome']}: "
        f"{row['probability']:.3f} (N={row['total']}, baseline={row['baseline_prob']:.3f})"
      )
