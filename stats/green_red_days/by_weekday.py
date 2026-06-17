"""Green & Red Days by Weekday stat.

Measures: how often does each weekday close green (up) versus red (down)?

The marginal probability over all resolved days is the overall result; the
per-weekday breakdown is produced by the framework's declared ``weekday`` slice.

A "day" is the RTH daily candle (session open to session close). Direction is
controlled by ``performance``:
  - ``close_to_close`` (default): today's session close vs the PREVIOUS resolved
    day's session close. The first resolved day has no prior close and is
    excluded (pending-sample discipline).
  - ``open_to_close``: today's session close vs today's session open.
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

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="Green & Red Days by Weekday",
  fr="Jours verts et rouges par jour de la semaine",
)
_DEFINITION = I18nString(
  en="How often does each weekday close green (up) versus red (down)?",
  fr="À quelle fréquence chaque jour de la semaine clôture-t-il en vert (hausse) plutôt qu'en rouge (baisse) ?",
)
_LABELS = Labels(
  conditions={
    "any_day": I18nString(en="All trading days", fr="Tous les jours de bourse"),
  },
  outcomes={
    "green_day": I18nString(en="Green day (up)", fr="Jour vert (hausse)"),
    "red_day": I18nString(en="Red day (down)", fr="Jour rouge (baisse)"),
  },
)

# Performance modes: how a day's direction (green/red) is determined.
_PERFORMANCE_MODES = ("close_to_close", "open_to_close")

# Single condition: every resolved day belongs to it; the weekday slice does the
# per-day breakdown.
_CONDITION = "any_day"


class GreenRedDaysByWeekday(BaseStat):
  """Marginal probability of a green vs red day, sliced by weekday."""

  stat_name = "green_red_days_by_weekday"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = ("weekday",)

  def __init__(
    self,
    instrument: str,
    config: InstrumentConfig,
    performance: str = "close_to_close",
    close_tolerance_min: int = 15,
  ) -> None:
    if performance not in _PERFORMANCE_MODES:
      raise ValueError(
        f"Unsupported performance '{performance}'. Choose from {list(_PERFORMANCE_MODES)}"
      )
    self.instrument = instrument
    self.timeframe = "daily"
    self.config = config
    self.performance = performance
    self.close_tolerance_min = close_tolerance_min

    rth = config.sessions["rth"]
    self.rth_start_min: int = minute_of_day(rth.start)
    self.rth_end_min: int = minute_of_day(rth.end)

  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-day RTH summary table from 1-min OHLCV data (vectorized).

    Each row corresponds to one resolved trading day, indexed by the normalized
    session date, with columns:
      session_open, session_close, last_minute, prev_session_close, day_green

    A day is "resolved" (confirmed) if:
      - It has a bar exactly at rth_start_min (clean session open)
      - Its last RTH bar is at or after rth_end_min - close_tolerance_min
        (excludes early-close days and the final incomplete day in the data)

    In ``close_to_close`` mode the first resolved day has no previous session
    close and is excluded (pending-sample discipline).
    """
    columns = ["session_open", "session_close", "last_minute", "prev_session_close", "day_green"]
    if candles_df.empty:
      return pd.DataFrame(columns=columns)

    df = candles_df.copy()
    df["mod"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["date"] = df["timestamp"].dt.normalize()

    # Keep only RTH bars: [rth_start_min, rth_end_min)
    rth_mask = (df["mod"] >= self.rth_start_min) & (df["mod"] < self.rth_end_min)
    rth = df[rth_mask].copy()
    if rth.empty:
      return pd.DataFrame(columns=columns)

    # session_open = open of the bar at exactly rth_start_min
    open_bars = (
      rth[rth["mod"] == self.rth_start_min].set_index("date")["open"].rename("session_open")
    )
    # Defensive: guard against duplicate bars at rth_start_min for the same day.
    open_bars = open_bars[~open_bars.index.duplicated(keep="first")]
    # session_close = close of the last RTH bar for that day
    last_bars = (
      rth.loc[rth.groupby("date")["mod"].idxmax(), ["date", "close", "mod"]]
      .set_index("date")
      .rename(columns={"close": "session_close", "mod": "last_minute"})
    )

    day = pd.concat([open_bars, last_bars], axis=1, sort=False)

    # Resolution filter: clean session open AND sufficient close coverage.
    resolved_min = self.rth_end_min - self.close_tolerance_min
    day = day[day["session_open"].notna() & (day["last_minute"] >= resolved_min)].copy()

    # Chronological order so prev_session_close references the previous resolved day.
    day = day.sort_index()
    day["prev_session_close"] = day["session_close"].shift(1)

    if self.performance == "open_to_close":
      day["day_green"] = day["session_close"] >= day["session_open"]
    else:  # close_to_close
      # First resolved day has no prior close: excluded (pending discipline).
      day = day[day["prev_session_close"].notna()].copy()
      day["day_green"] = day["session_close"] >= day["prev_session_close"]

    return day

  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the two outcome rows (green day / red day) for one day subset.

    If baseline_rows is provided, merges baseline_prob/baseline_n from it.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    total = len(day_table)
    green_count = int(day_table["day_green"].sum()) if total > 0 else 0
    outcomes = [
      ("green_day", green_count),
      ("red_day", total - green_count),
    ]

    rows: list[StatResultRow] = []
    for out_key, count in outcomes:
      probability = count / total if total > 0 else 0.0
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
        )
      )
    return rows

  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random baseline: randomize each day's direction (p=0.5).

    Uses a fixed seed for deterministic output. Expected baseline_prob ≈ 0.5.
    Operates directly on the day table so the framework can compute a baseline
    per slice group as well as overall.
    """
    rng = np.random.default_rng(seed)
    n = len(day_table)
    random_green = rng.integers(0, 2, size=n).astype(bool)

    tmp = day_table.copy()
    tmp["day_green"] = random_green
    return self.compute_rows(tmp, baseline_rows=None)


def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
  performance: str = "close_to_close",
) -> Path:
  """Load data and compute Green & Red Days by Weekday for the daily timeframe.

  Writes the result JSON to results/.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  stat = GreenRedDaysByWeekday(instrument=instrument, config=config, performance=performance)
  result = stat.compute(candles_df)

  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Green & Red Days by Weekday stat")
  parser.add_argument("--instrument", default="NQ", help="Instrument name (default: NQ)")
  parser.add_argument("--data-path", default=None, help="Override parquet file path")
  parser.add_argument("--config-dir", default="config", help="Config directory (default: config)")
  parser.add_argument(
    "--performance",
    default="close_to_close",
    choices=list(_PERFORMANCE_MODES),
    help="Day direction basis (default: close_to_close)",
  )
  args = parser.parse_args()

  output_path = run(
    instrument=args.instrument,
    config_dir=args.config_dir,
    data_path=args.data_path,
    performance=args.performance,
  )

  print(f"Written: {output_path}")

  # Print a brief summary of results
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
    weekday = tf_data["slices"].get("weekday", {}).get("groups", {})
    for day_key, group in weekday.items():
      green = next((r for r in group["results"] if r["outcome"] == "green_day"), None)
      if green:
        print(
          f"      {day_key}: P(green)={green['probability']:.3f} "
          f"(N={green['total']})"
        )
