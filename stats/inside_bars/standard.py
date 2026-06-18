"""Inside Bars stat.

Measures how often a session OPENS INSIDE the prior trading day's range — within
its RTH high-low range (inclusive) — and, given such an open, which direction
price breaks out during the session.

Two tiers of rows are reported:

  1. Inside-open frequency (condition ``inside_open``): over all countable days,
     how often does the session open inside the prior day's RTH range
     (``session_open >= prev_low`` AND ``session_open <= prev_high``)? An open
     exactly AT ``prev_high`` or ``prev_low`` counts as inside (inclusive
     inequalities), making this the exact complement of the outside_days
     strict-outside definition.

  2. Breakout direction (condition ``inside``): among the days that opened inside,
     which direction did price break during the session? Four MUTUALLY EXCLUSIVE
     outcomes partition the inside set exhaustively:
       - ``broke_high``  — day_high >  prev_high AND day_low  >= prev_low
                           (broke above the prior range only)
       - ``broke_low``   — day_low  <  prev_low  AND day_high <= prev_high
                           (broke below the prior range only)
       - ``broke_both``  — day_high >  prev_high AND day_low  <  prev_low
                           (broke both sides during the session)
       - ``contained``   — day_high <= prev_high AND day_low  >= prev_low
                           (held entirely within the prior range all session)
     Strict inequalities define a break (touching the level exactly is NOT a
     break), mirroring the strict re-entry convention of the sibling
     ``outside_days`` stat.

An inside open uses inclusive inequality: ``prev_low <= open <= prev_high``.
The "prior day" is the chronologically previous RESOLVED day, so it skips any
excluded/early-close day. The FIRST resolved day has no prior day and is excluded
from every denominator (pending-sample discipline). ``total_samples`` counts ALL
resolved days; each row's ``total`` counts only the relevant countable days.

Declared slices re-run the whole computation per subset:
  - ``weekday``     — the "by weekday" breakdown.
  - ``prev_candle`` — split by the prior session's color (green/red).
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
from stats.utils.daily_candles import build_day_table_with_prior_range

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="Inside Bars",
  fr="Inside bars",
)
_DEFINITION = I18nString(
  en="How often does a session open inside the prior day's range (between its RTH low and RTH high, inclusive), and given such an open, does price break above, below, both sides, or stay contained within the prior range during the session?",
  fr="À quelle fréquence une session ouvre-t-elle à l'intérieur du range du jour précédent (entre son plus bas RTH et son plus haut RTH, bornes incluses), et après une telle ouverture, le prix casse-t-il vers le haut, vers le bas, des deux côtés, ou reste-t-il contenu dans le range précédent pendant la séance ?",
)
_LABELS = Labels(
  conditions={
    "inside_open": I18nString(en="Inside open", fr="Ouverture dans le range"),
    "inside": I18nString(
      en="Inside open (breakout direction)",
      fr="Ouverture dans le range (direction de la cassure)",
    ),
  },
  outcomes={
    "inside": I18nString(
      en="Open inside prior range",
      fr="Ouverture à l'intérieur du range précédent",
    ),
    "broke_high": I18nString(
      en="Broke above prior high only",
      fr="Cassure au-dessus du plus haut précédent uniquement",
    ),
    "broke_low": I18nString(
      en="Broke below prior low only",
      fr="Cassure en-dessous du plus bas précédent uniquement",
    ),
    "broke_both": I18nString(
      en="Broke both sides",
      fr="Cassure des deux côtés",
    ),
    "contained": I18nString(
      en="Contained within prior range",
      fr="Contenu dans le range précédent",
    ),
  },
)


class InsideBars(BaseStat):
  """Inside-open frequency plus breakout-direction breakdown given an inside open."""

  stat_name = "inside_bars"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = ("weekday", "prev_candle")

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

  # -------------------------------------------------------------------------
  # Day table construction
  # -------------------------------------------------------------------------
  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-session table with open, intraday extremes and prior range.

    Columns:
      session_open, day_high, day_low (RTH intraday extremes), prev_high,
      prev_low (the prior RESOLVED day's extremes; NaN for the first resolved
      day), prev_session_green (the prior session's color; NaN for the first
      resolved day).

    ``prev_high`` / ``prev_low`` are NaN for the first resolved day (no prior
    day), so it is excluded from every denominator downstream.
    """
    columns = [
      "session_open",
      "day_high",
      "day_low",
      "prev_high",
      "prev_low",
      "prev_session_green",
    ]
    daily = build_day_table_with_prior_range(
      candles_df, self.rth_start_min, self.rth_end_min, self.close_tolerance_min
    )
    if daily.empty:
      return pd.DataFrame(columns=columns)

    return daily[columns]

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the inside-open frequency and breakout-direction rows.

    Only days with a prior resolved day (``prev_high`` not NaN) are countable.
    If ``baseline_rows`` is provided, merges ``baseline_prob`` / ``baseline_n``.

    The four breakout-direction outcomes (``broke_high``, ``broke_low``,
    ``broke_both``, ``contained``) are mutually exclusive and partition the
    set of inside-open days exhaustively.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    def _make(condition: str, outcome: str, count: int, total: int) -> StatResultRow:
      bl = baseline_map.get((condition, outcome))
      return StatResultRow(
        condition=condition,
        outcome=outcome,
        count=count,
        total=total,
        probability=count / total if total > 0 else 0.0,
        baseline_prob=bl.probability if bl else 0.0,
        baseline_n=bl.total if bl else 0,
      )

    if day_table.empty:
      return [
        _make("inside_open", "inside", 0, 0),
        _make("inside", "broke_high", 0, 0),
        _make("inside", "broke_low", 0, 0),
        _make("inside", "broke_both", 0, 0),
        _make("inside", "contained", 0, 0),
      ]

    prev_high = day_table["prev_high"]
    prev_low = day_table["prev_low"]
    session_open = day_table["session_open"]
    day_high = day_table["day_high"]
    day_low = day_table["day_low"]

    countable = prev_high.notna() & prev_low.notna()
    # Inclusive inequalities: opening exactly at the prior high or low is inside.
    inside = countable & (session_open >= prev_low) & (session_open <= prev_high)

    countable_n = int(countable.sum())
    inside_n = int(inside.sum())

    # Four mutually exclusive outcomes that partition inside days.
    # Strict inequality defines a break (touching the level is NOT a break).
    broke_high = inside & (day_high > prev_high) & (day_low >= prev_low)
    broke_low = inside & (day_low < prev_low) & (day_high <= prev_high)
    broke_both = inside & (day_high > prev_high) & (day_low < prev_low)
    contained = inside & (day_high <= prev_high) & (day_low >= prev_low)

    return [
      # Tier 1 — inside-open frequency over all countable days.
      _make("inside_open", "inside", inside_n, countable_n),
      # Tier 2 — breakout direction given an inside open.
      _make("inside", "broke_high", int(broke_high.sum()), inside_n),
      _make("inside", "broke_low", int(broke_low.sum()), inside_n),
      _make("inside", "broke_both", int(broke_both.sum()), inside_n),
      _make("inside", "contained", int(contained.sum()), inside_n),
    ]

  # -------------------------------------------------------------------------
  # Baseline
  # -------------------------------------------------------------------------
  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random null baseline. Deterministic for a fixed seed.

    The prior-range pair (``prev_high``, ``prev_low``) is **permuted together**
    across days, so each day's open and intraday extremes are compared against an
    UNRELATED day's range. This is the joint null for both tiers at once: it
    destroys the temporal adjacency that the stat measures (does opening inside
    *yesterday's* range carry information?), while keeping each prior range
    internally consistent (``low <= high``). The lone NaN pair (first day) simply
    moves to a random day, preserving the countable count.
    """
    n = len(day_table)
    if n == 0:
      return self.compute_rows(day_table, baseline_rows=None)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    tmp = day_table.copy()
    tmp["prev_high"] = day_table["prev_high"].to_numpy()[perm]
    tmp["prev_low"] = day_table["prev_low"].to_numpy()[perm]

    return self.compute_rows(tmp, baseline_rows=None)


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
) -> Path:
  """Load data and compute Inside Bars for the daily timeframe.

  Writes the result JSON to results/.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  stat = InsideBars(instrument=instrument, config=config)
  result = stat.compute(candles_df)

  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Inside Bars stat")
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
    print(f"\n  {tf}: {tf_data['total_samples']} resolved sessions | {tf_data['data_range']}")
    for row in tf_data["results"]:
      print(
        f"    {row['condition']} -> {row['outcome']}: "
        f"{row['probability']:.3f} (N={row['total']}, baseline={row['baseline_prob']:.3f})"
      )
