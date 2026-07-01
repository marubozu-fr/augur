"""Overnight Continuation stat.

Measures: given the overnight gap's direction (today's open vs the prior session's
close), how often does the day close in the same direction intraday?

A 2x2 conditional matrix:
  - P(green day | green overnight gap), P(red day | green overnight gap)
  - P(green day | red overnight gap),   P(red day | red overnight gap)

Definitions:
  - Overnight gap color: today's ``session_open`` vs the PREVIOUS resolved
    session's ``session_close``.
      gap green: session_open >= prev_session_close
      gap red:   session_open <  prev_session_close
  - Intraday day color: ``session_close`` vs ``session_open`` (close-vs-open).
      green: session_close >= session_open
      red:   session_close <  session_open

"Continuation" means the day closes in the same direction as the gap (gap green
→ day green, gap red → day red). All four matrix cells are reported so the
asymmetry is directly visible.

The FIRST resolved day has no prior close, so its gap is undefined and it is
excluded from every denominator (pending-sample discipline). ``total_samples``
counts ALL resolved days; each row's ``total`` counts only the days that carry
a gap direction (i.e. have a prior resolved close).
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
  en="Overnight Continuation",
  fr="Continuation overnight",
)
_DEFINITION = I18nString(
  en="Given the overnight gap's direction (today's open vs the prior session's close), how often does the day close in the same direction (intraday)?",
  fr="Selon la direction du gap overnight (ouverture du jour vs clôture de la session précédente), à quelle fréquence le jour clôture-t-il dans la même direction (intraday) ?",
)
_LABELS = Labels(
  conditions={
    "gap_green": I18nString(en="Green overnight gap (up)", fr="Gap overnight vert (hausse)"),
    "gap_red": I18nString(en="Red overnight gap (down)", fr="Gap overnight rouge (baisse)"),
  },
  outcomes={
    "green": I18nString(en="Green day (up)", fr="Jour vert (hausse)"),
    "red": I18nString(en="Red day (down)", fr="Jour rouge (baisse)"),
  },
)

# Condition / outcome enumeration: gap color -> intraday day color.
_CONDITIONS = (("gap_green", True), ("gap_red", False))
_OUTCOMES = (("green", True), ("red", False))


class OvernightContinuation(BaseStat):
  """Conditional probability of the intraday day color given the overnight gap color."""

  stat_name = "overnight_continuation"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = ()

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

  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-session table with gap and intraday direction columns.

    Columns:
      session_open, session_close, prev_session_close (NaN for first day),
      day_green (bool: session_close >= session_open).

    ``prev_session_close`` is NaN for the first resolved day (no prior close),
    so that day is excluded from every denominator downstream.
    """
    columns = ["session_open", "session_close", "prev_session_close", "day_green"]
    day = build_resolved_days(
      candles_df, self.rth_start_min, self.rth_end_min, self.close_tolerance_min
    )
    if day.empty:
      return pd.DataFrame(columns=columns)

    # build_resolved_days returns rows in chronological order.
    day["prev_session_close"] = day["session_close"].shift(1)
    day["day_green"] = day["session_close"] >= day["session_open"]
    return day[columns]

  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the four matrix rows (gap green/red x day green/red).

    Only sessions with a prior session close (prev_session_close not NaN) count.
    If baseline_rows is provided, merges baseline_prob/baseline_n from it.
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

    day_green = day_table["day_green"].astype(bool)
    prev_session_close = day_table["prev_session_close"]
    session_open = day_table["session_open"]
    countable = prev_session_close.notna()
    gap_is_green = session_open >= prev_session_close

    rows: list[StatResultRow] = []
    for cond_key, cond_is_green in _CONDITIONS:
      cond_mask = countable & (gap_is_green == cond_is_green)
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
    """One SampleRow per countable day, mirroring ``compute_rows``.

    Each day with a prior session close (``prev_session_close`` not NaN)
    contributes exactly one sample: ``condition`` is its gap color
    (``gap_green`` / ``gap_red``) and ``outcome`` is its intraday day color
    (``green`` / ``red``). The 2x2 matrix's four cells are all reproduced
    directly, and the two outcomes per condition sum to that condition's total.
    """
    if day_table.empty:
      return []

    countable = day_table["prev_session_close"].notna()
    if not bool(countable.any()):
      return []

    sub = day_table[countable]
    gap_is_green = sub["session_open"] >= sub["prev_session_close"]
    day_green = sub["day_green"].astype(bool)

    samples: list[SampleRow] = []
    for ts, gap_green, is_green in zip(sub.index, gap_is_green, day_green):
      samples.append(
        SampleRow(
          date=ts.strftime("%Y-%m-%d"),
          condition="gap_green" if gap_green else "gap_red",
          outcome="green" if is_green else "red",
        )
      )
    return samples

  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random baseline: randomize intraday day color (p=0.5) to destroy any correlation.

    The overnight gap condition stays as-is from real data; only ``day_green``
    is replaced with an independent random sequence. Expected baseline_prob ~ 0.5
    for every row. Deterministic for a fixed seed.
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
) -> Path:
  """Load data and compute Overnight Continuation for the daily timeframe.

  Writes the result JSON to results/.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  stat = OvernightContinuation(instrument=instrument, config=config)
  result = stat.compute(candles_df)

  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Overnight Continuation stat")
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
