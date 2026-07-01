"""Previous Session Correlation stat.

Measures: how often does the current session close green (or red) given the prior
session's color? Day-over-day color follow-through, expressed as a 2x2 conditional
matrix.

Unlike Green & Red Streaks (which collapses the same data into a single
continue/break outcome), this stat reports both outcomes per prior color, so the
asymmetry between "green follows green" and "red follows red" is visible directly:
  - P(green session | prior green), P(red session | prior green)
  - P(green session | prior red),   P(red session | prior red)

A "session" is the RTH daily candle (session open to session close). Direction is
controlled by ``performance``:
  - ``close_to_close`` (default): a session is green when its close is at or above
    the PREVIOUS resolved session's close. The first resolved session has no prior
    close and is excluded (pending-sample discipline).
  - ``open_to_close``: a session is green when its close is at or above its open.

The prior-session color of a session is the color of the chronologically previous
resolved session. The first usable session has no prior session, so its condition
is undefined and it is excluded from every denominator (countable rows only).
``total_samples`` counts all resolved sessions; each row's ``total`` counts only the
countable sessions (those with a prior session) carrying that prior color.
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
  en="Previous Session Correlation",
  fr="Corrélation de la session précédente",
)
_DEFINITION = I18nString(
  en="How often does the current session close green or red given the prior session's color?",
  fr="À quelle fréquence la session actuelle clôture-t-elle en vert ou en rouge selon la couleur de la session précédente ?",
)
_LABELS = Labels(
  conditions={
    "prev_green": I18nString(en="Prior session green", fr="Session précédente verte"),
    "prev_red": I18nString(en="Prior session red", fr="Session précédente rouge"),
  },
  outcomes={
    "green": I18nString(en="Green session (up)", fr="Session verte (hausse)"),
    "red": I18nString(en="Red session (down)", fr="Session rouge (baisse)"),
  },
)

# Performance modes: how a session's direction (green/red) is determined.
_PERFORMANCE_MODES = ("close_to_close", "open_to_close")

# Condition / outcome enumeration: prior color -> current color.
_CONDITIONS = (("prev_green", True), ("prev_red", False))
_OUTCOMES = (("green", True), ("red", False))


class PrevSessionCorrelation(BaseStat):
  """Conditional probability of the current session color given the prior one."""

  stat_name = "prev_session_correlation"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = ()

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
    """Build the per-session table with each session's color and its prior color.

    Columns:
      session_open, session_close, session_green (bool), prev_green (1.0/0.0/NaN).

    ``prev_green`` is the color of the chronologically previous resolved session;
    it is NaN for the first usable session (no prior session), which is therefore
    excluded from every denominator downstream. In ``close_to_close`` mode the very
    first resolved session has no prior close and is dropped here (pending
    discipline) before prior colors are assigned.
    """
    columns = ["session_open", "session_close", "session_green", "prev_green"]
    day = build_resolved_days(
      candles_df, self.rth_start_min, self.rth_end_min, self.close_tolerance_min
    )
    if day.empty:
      return pd.DataFrame(columns=columns)

    # build_resolved_days returns rows in chronological order.
    if self.performance == "open_to_close":
      day["session_green"] = day["session_close"] >= day["session_open"]
    else:  # close_to_close
      day["prev_close"] = day["session_close"].shift(1)
      # First resolved session has no prior close: excluded (pending discipline).
      day = day[day["prev_close"].notna()].copy()
      day["session_green"] = day["session_close"] >= day["prev_close"]

    day["prev_green"] = day["session_green"].astype(float).shift(1)
    return day[columns]

  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the four matrix rows (prior green/red x current green/red).

    Only sessions with a prior session (prev_green not NaN) count. If
    baseline_rows is provided, merges baseline_prob/baseline_n from it.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    if len(day_table) > 0:
      session_green = day_table["session_green"].astype(bool)
      prev_green = day_table["prev_green"]
      countable = prev_green.notna()
      prev_is_green = prev_green == 1.0
    else:
      session_green = pd.Series([], dtype=bool)
      countable = pd.Series([], dtype=bool)
      prev_is_green = pd.Series([], dtype=bool)

    rows: list[StatResultRow] = []
    for cond_key, cond_is_green in _CONDITIONS:
      cond_mask = countable & (prev_is_green == cond_is_green)
      total = int(cond_mask.sum())
      for out_key, out_is_green in _OUTCOMES:
        out_match = session_green if out_is_green else ~session_green
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
    """One SampleRow per countable session, mirroring ``compute_rows``'s classification.

    ``condition`` is the prior session's color (``prev_green``/``prev_red``);
    ``outcome`` is the current session's color (``green``/``red``) — the same
    pairing ``compute_rows`` aggregates into the 2x2 matrix. Sessions with no
    prior session (``prev_green`` is NaN — the first usable session) are excluded,
    matching ``compute_rows``'s countable mask.

    SPECIAL — correlation family (see ``BaseStat.classify_samples`` docstring):
    this stat and ``market_session_correlation`` report a conditional-probability
    matrix rather than a literal Pearson r, but both are named/treated as
    "correlation" families. ``value`` stores the current session's color as a
    float (1.0 green / 0.0 red) — the "y" side of the boolean pairing whose "x"
    side is ``condition``. A future iteration could decode (x, y) pairs from
    (condition, value) to recompute a full Pearson/point-biserial r over a
    date-filtered subset of days; that recomputation itself is deferred.
    """
    if day_table.empty:
      return []

    prev_green = day_table["prev_green"]
    countable = prev_green.notna()
    if not countable.any():
      return []

    countable_session = day_table.loc[countable, "session_green"].astype(bool)
    countable_prev = prev_green[countable] == 1.0

    samples: list[SampleRow] = []
    for ts, prev_is_green, session_is_green in zip(
      countable_prev.index, countable_prev, countable_session
    ):
      condition = "prev_green" if prev_is_green else "prev_red"
      outcome = "green" if session_is_green else "red"
      samples.append(
        SampleRow(
          date=ts.strftime("%Y-%m-%d"),
          condition=condition,
          outcome=outcome,
          value=1.0 if session_is_green else 0.0,
        )
      )
    return samples

  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random baseline: independent green/red colors (p=0.5) destroy any correlation.

    Each session is recolored at random, then the prior-session color is recomputed
    from the randomized chronological sequence, so the current color is independent
    of the prior one. Expected baseline_prob ~ 0.5 for every row. Deterministic for
    a fixed seed.
    """
    rng = np.random.default_rng(seed)
    n = len(day_table)
    random_green = rng.integers(0, 2, size=n).astype(bool)

    tmp = day_table.copy()
    tmp["session_green"] = random_green
    tmp["prev_green"] = pd.Series(random_green.astype(float), index=tmp.index).shift(1)
    return self.compute_rows(tmp, baseline_rows=None)


def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
  performance: str = "close_to_close",
) -> Path:
  """Load data and compute Previous Session Correlation for the daily timeframe.

  Writes the result JSON to results/.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  stat = PrevSessionCorrelation(instrument=instrument, config=config, performance=performance)
  result = stat.compute(candles_df)

  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Previous Session Correlation stat")
  parser.add_argument("--instrument", default="NQ", help="Instrument name (default: NQ)")
  parser.add_argument("--data-path", default=None, help="Override parquet file path")
  parser.add_argument("--config-dir", default="config", help="Config directory (default: config)")
  parser.add_argument(
    "--performance",
    default="close_to_close",
    choices=list(_PERFORMANCE_MODES),
    help="Session direction basis (default: close_to_close)",
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
    print(f"\n  {tf}: {tf_data['total_samples']} resolved sessions | {tf_data['data_range']}")
    for row in tf_data["results"]:
      print(
        f"    {row['condition']} -> {row['outcome']}: "
        f"{row['probability']:.3f} (N={row['total']}, baseline={row['baseline_prob']:.3f})"
      )
