"""Previous Day's Range stat.

Measures how often intraday price breaks the prior trading day's range (its RTH
high or low), and — given a break — whether the session followed through by
closing in the break's direction.

Two tiers of rows are reported:

  1. Break frequency (condition ``prior_range``): over all countable days, how
     often does the day's RTH high exceed the prior day's RTH high
     (``break_high``) or the day's RTH low fall below the prior day's RTH low
     (``break_low``)? These two outcomes are independent break rates — a single
     session may break both, one, or neither — so they do not partition.

  2. Follow-through (conditions ``break_high`` / ``break_low``): among the days
     that broke the prior high (resp. low), how often did the session close
     green (close >= open) versus red (close < open)? For a high break, green is
     follow-through (continuation up); for a low break, red is follow-through
     (continuation down). The green/red split partitions each break condition.

Follow-through here is the *session direction* (close vs open). The distinct
"close outside vs back inside the prior range" classification is a separate
variant (``by_outside_close``) and is not computed by this standard module.

A break uses strict inequality: ``day_high > prev_high`` / ``day_low < prev_low``
(touching the level is not a break). The "prior day" is the chronologically
previous RESOLVED day, so it skips over any excluded/early-close day. The FIRST
resolved day has no prior day and is excluded from every denominator
(pending-sample discipline). ``total_samples`` counts ALL resolved days; each
row's ``total`` counts only the relevant countable days.

Declared slices re-run the whole computation per subset:
  - ``weekday``     — the "by weekday" breakdown.
  - ``prev_candle`` — the "by prev close" breakdown (prior session green/red).
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
from stats.utils.daily_candles import build_day_table_with_prior_range

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="Previous Day's Range",
  fr="Range du jour précédent",
)
_DEFINITION = I18nString(
  en="How often does intraday price break the prior day's range (RTH high or low), and given a break, does the session follow through by closing in the break's direction?",
  fr="À quelle fréquence le prix intraday casse-t-il le range du jour précédent (plus haut ou plus bas RTH), et après une cassure, la session poursuit-elle en clôturant dans la direction de la cassure ?",
)
_LABELS = Labels(
  conditions={
    "prior_range": I18nString(en="Prior day's range", fr="Range du jour précédent"),
    "break_high": I18nString(en="Broke prior high", fr="A cassé le plus haut précédent"),
    "break_low": I18nString(en="Broke prior low", fr="A cassé le plus bas précédent"),
  },
  outcomes={
    "break_high": I18nString(en="Break prior high", fr="Casse le plus haut précédent"),
    "break_low": I18nString(en="Break prior low", fr="Casse le plus bas précédent"),
    "green": I18nString(en="Green day (up)", fr="Jour vert (hausse)"),
    "red": I18nString(en="Red day (down)", fr="Jour rouge (baisse)"),
  },
)


class PrevDaysRange(BaseStat):
  """Break frequency of the prior day's range plus directional follow-through."""

  stat_name = "prev_days_range"
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
    """Build the per-session table with prior-range and direction columns.

    Columns:
      day_high, day_low (RTH intraday extremes), prev_high, prev_low (the prior
      RESOLVED day's extremes; NaN for the first resolved day), day_green
      (session_close >= session_open), prev_session_green (the prior session's
      color; NaN for the first resolved day).

    ``prev_high`` / ``prev_low`` are NaN for the first resolved day (no prior
    day), so it is excluded from every denominator downstream.
    """
    columns = [
      "day_high",
      "day_low",
      "prev_high",
      "prev_low",
      "day_green",
      "prev_session_green",
    ]
    daily = build_day_table_with_prior_range(
      candles_df, self.rth_start_min, self.rth_end_min, self.close_tolerance_min
    )
    if daily.empty:
      return pd.DataFrame(columns=columns)

    # This stat's compute reads the current day's color directly; derive it from
    # the resolved open/close (prev_session_green is already provided upstream).
    daily = daily.copy()
    daily["day_green"] = daily["session_close"] >= daily["session_open"]

    return daily[columns]

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the break-frequency and follow-through rows.

    Only days with a prior resolved day (``prev_high`` not NaN) are countable.
    If ``baseline_rows`` is provided, merges ``baseline_prob`` / ``baseline_n``.
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
        _make("prior_range", "break_high", 0, 0),
        _make("prior_range", "break_low", 0, 0),
        _make("break_high", "green", 0, 0),
        _make("break_high", "red", 0, 0),
        _make("break_low", "green", 0, 0),
        _make("break_low", "red", 0, 0),
      ]

    countable = day_table["prev_high"].notna() & day_table["prev_low"].notna()
    day_green = day_table["day_green"].astype(bool)
    broke_high = countable & (day_table["day_high"] > day_table["prev_high"])
    broke_low = countable & (day_table["day_low"] < day_table["prev_low"])

    countable_n = int(countable.sum())
    high_n = int(broke_high.sum())
    low_n = int(broke_low.sum())

    return [
      # Tier 1 — break frequency over all countable days.
      _make("prior_range", "break_high", high_n, countable_n),
      _make("prior_range", "break_low", low_n, countable_n),
      # Tier 2 — directional follow-through given a break.
      _make("break_high", "green", int((broke_high & day_green).sum()), high_n),
      _make("break_high", "red", int((broke_high & ~day_green).sum()), high_n),
      _make("break_low", "green", int((broke_low & day_green).sum()), low_n),
      _make("break_low", "red", int((broke_low & ~day_green).sum()), low_n),
    ]

  # -------------------------------------------------------------------------
  # Per-day sample classification
  # -------------------------------------------------------------------------
  def classify_samples(self, day_table: pd.DataFrame) -> list[SampleRow]:
    """Per-day SampleRows mirroring ``compute_rows``'s two tiers.

    Tier 1 (condition ``prior_range``) — INDEPENDENT break rates: a countable
    day (a prior resolved day exists, i.e. ``prev_high`` / ``prev_low`` not NaN)
    yields one sample per break it makes — ``break_high`` if ``day_high >
    prev_high``, ``break_low`` if ``day_low < prev_low`` — so a day produces 0,
    1, or 2 tier-1 samples (they do not partition, so a non-breaking day simply
    contributes nothing to this tier).

    Tier 2 (conditions ``break_high`` / ``break_low``) — directional
    follow-through: each day that broke the prior high (resp. low) yields
    exactly one additional sample under that break's condition, ``green`` or
    ``red`` (``day_green`` from the day table), partitioning that break
    condition's subset exactly as ``compute_rows`` does.

    Non-countable days (no prior resolved day, e.g. the first resolved day)
    contribute nothing.
    """
    if day_table.empty:
      return []

    samples: list[SampleRow] = []
    for ts, day_high, day_low, prev_high, prev_low, day_green in zip(
      day_table.index,
      day_table["day_high"],
      day_table["day_low"],
      day_table["prev_high"],
      day_table["prev_low"],
      day_table["day_green"],
    ):
      if pd.isna(prev_high) or pd.isna(prev_low):
        continue

      date_str = ts.strftime("%Y-%m-%d")
      broke_high = day_high > prev_high
      broke_low = day_low < prev_low
      direction = "green" if bool(day_green) else "red"

      if broke_high:
        samples.append(SampleRow(date=date_str, condition="prior_range", outcome="break_high"))
        samples.append(SampleRow(date=date_str, condition="break_high", outcome=direction))
      if broke_low:
        samples.append(SampleRow(date=date_str, condition="prior_range", outcome="break_low"))
        samples.append(SampleRow(date=date_str, condition="break_low", outcome=direction))

    return samples

  # -------------------------------------------------------------------------
  # Baseline
  # -------------------------------------------------------------------------
  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random null baseline. Deterministic for a fixed seed.

    Two independent randomizations destroy the two signals this stat measures:
      - The prior-range pair (``prev_high``, ``prev_low``) is permuted across
        days, so each day is compared against an UNRELATED day's range. This is
        the null for break frequency: temporal adjacency carries no information.
        The pair is permuted together so each prior range stays internally
        consistent (low <= high), and the lone NaN pair simply moves to a random
        day, preserving the countable count.
      - ``day_green`` is reassigned by a fair coin (p=0.5), so direction is
        independent of any break. This is the null for follow-through (~0.5).
    """
    n = len(day_table)
    if n == 0:
      return self.compute_rows(day_table, baseline_rows=None)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    tmp = day_table.copy()
    tmp["prev_high"] = day_table["prev_high"].to_numpy()[perm]
    tmp["prev_low"] = day_table["prev_low"].to_numpy()[perm]
    tmp["day_green"] = rng.integers(0, 2, size=n).astype(bool)

    return self.compute_rows(tmp, baseline_rows=None)


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
) -> Path:
  """Load data and compute Previous Day's Range for the daily timeframe.

  Writes the result JSON to results/.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  stat = PrevDaysRange(instrument=instrument, config=config)
  result = stat.compute(candles_df)

  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Previous Day's Range stat")
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
