"""Outside Days stat.

Measures how often a session OPENS outside the prior trading day's range — above
its RTH high (bullish outside) or below its RTH low (bearish outside) — and, given
such an open, whether the move continued away from the range or reversed back into
it during the session.

Two tiers of rows are reported:

  1. Outside-open frequency (condition ``outside_open``): over all countable days,
     how often does the session open above the prior day's RTH high (``bullish``)
     or below the prior day's RTH low (``bearish``)? These two outcomes are
     mutually exclusive (a single open cannot be both above the high and below the
     low) but do not partition — most days open inside the prior range.

  2. Continuation vs reversal (conditions ``bullish`` / ``bearish``): among the
     days that opened outside, did price hold outside the prior range for the whole
     session (``continuation``) or trade back into it (``reversal``)?
       - bullish outside (open > prev_high):
           reversal     ⇐ day_low  <  prev_high (price re-entered the range)
           continuation ⇐ day_low  >= prev_high (held above all session)
       - bearish outside (open < prev_low):
           reversal     ⇐ day_high >  prev_low
           continuation ⇐ day_high <= prev_low
     The continuation/reversal split partitions each outside condition.

An outside open uses strict inequality: ``open > prev_high`` / ``open < prev_low``
(opening exactly at the level is not outside). Re-entry into the range likewise
uses strict inequality, mirroring the strict-break convention of the sibling
``prev_days_range`` stat. The "prior day" is the chronologically previous RESOLVED
day, so it skips any excluded/early-close day. The FIRST resolved day has no prior
day and is excluded from every denominator (pending-sample discipline).
``total_samples`` counts ALL resolved days; each row's ``total`` counts only the
relevant countable days.

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
from stats.utils.daily_candles import build_resolved_days

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="Outside Days",
  fr="Outside days",
)
_DEFINITION = I18nString(
  en="How often does a session open outside the prior day's range (above its RTH high or below its RTH low), and given such an open, does price continue away from the range or reverse back into it during the session?",
  fr="À quelle fréquence une session ouvre-t-elle hors du range du jour précédent (au-dessus de son plus haut RTH ou en-dessous de son plus bas RTH), et après une telle ouverture, le prix poursuit-il hors du range ou y revient-il (reversal) pendant la séance ?",
)
_LABELS = Labels(
  conditions={
    "outside_open": I18nString(en="Outside open", fr="Ouverture hors range"),
    "bullish": I18nString(
      en="Bullish outside (open above prior high)",
      fr="Outside haussier (ouverture au-dessus du plus haut)",
    ),
    "bearish": I18nString(
      en="Bearish outside (open below prior low)",
      fr="Outside baissier (ouverture en-dessous du plus bas)",
    ),
  },
  outcomes={
    "bullish": I18nString(
      en="Open above prior high",
      fr="Ouverture au-dessus du plus haut précédent",
    ),
    "bearish": I18nString(
      en="Open below prior low",
      fr="Ouverture en-dessous du plus bas précédent",
    ),
    "continuation": I18nString(
      en="Continuation (held outside)",
      fr="Continuation (maintien hors range)",
    ),
    "reversal": I18nString(
      en="Reversal (back into range)",
      fr="Reversal (retour dans le range)",
    ),
  },
)


class OutsideDays(BaseStat):
  """Outside-open frequency plus continuation-vs-reversal of the outside move."""

  stat_name = "outside_days"
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
    empty = pd.DataFrame(columns=columns)

    if candles_df.empty:
      return empty

    resolved = build_resolved_days(
      candles_df, self.rth_start_min, self.rth_end_min, self.close_tolerance_min
    )
    if resolved.empty:
      return empty

    # Per-day RTH high/low from the same RTH bar filter the resolution uses.
    df = candles_df.copy()
    df["_mod"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["_date"] = df["timestamp"].dt.normalize()
    rth_mask = (df["_mod"] >= self.rth_start_min) & (df["_mod"] < self.rth_end_min)
    rth = df[rth_mask]

    day_high = rth.groupby("_date")["high"].max().rename("day_high")
    day_low = rth.groupby("_date")["low"].min().rename("day_low")

    # Inner join keeps only resolved dates; result stays chronologically sorted.
    daily = resolved.join(day_high, how="inner").join(day_low, how="inner")
    if daily.empty:
      return empty

    day_green = daily["session_close"] >= daily["session_open"]
    # Prior RESOLVED day's values (shift over the resolved-only, sorted index).
    daily["prev_high"] = daily["day_high"].shift(1)
    daily["prev_low"] = daily["day_low"].shift(1)
    daily["prev_session_green"] = day_green.shift(1)

    return daily[columns]

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the outside-open frequency and continuation/reversal rows.

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
        _make("outside_open", "bullish", 0, 0),
        _make("outside_open", "bearish", 0, 0),
        _make("bullish", "continuation", 0, 0),
        _make("bullish", "reversal", 0, 0),
        _make("bearish", "continuation", 0, 0),
        _make("bearish", "reversal", 0, 0),
      ]

    prev_high = day_table["prev_high"]
    prev_low = day_table["prev_low"]
    session_open = day_table["session_open"]
    day_high = day_table["day_high"]
    day_low = day_table["day_low"]

    countable = prev_high.notna() & prev_low.notna()
    bullish = countable & (session_open > prev_high)
    bearish = countable & (session_open < prev_low)

    countable_n = int(countable.sum())
    bullish_n = int(bullish.sum())
    bearish_n = int(bearish.sum())

    # Reversal = price re-entered the prior range (strict); continuation = held out.
    bull_reversal = bullish & (day_low < prev_high)
    bull_continuation = bullish & (day_low >= prev_high)
    bear_reversal = bearish & (day_high > prev_low)
    bear_continuation = bearish & (day_high <= prev_low)

    return [
      # Tier 1 — outside-open frequency over all countable days.
      _make("outside_open", "bullish", bullish_n, countable_n),
      _make("outside_open", "bearish", bearish_n, countable_n),
      # Tier 2 — continuation vs reversal given an outside open.
      _make("bullish", "continuation", int(bull_continuation.sum()), bullish_n),
      _make("bullish", "reversal", int(bull_reversal.sum()), bullish_n),
      _make("bearish", "continuation", int(bear_continuation.sum()), bearish_n),
      _make("bearish", "reversal", int(bear_reversal.sum()), bearish_n),
    ]

  # -------------------------------------------------------------------------
  # Baseline
  # -------------------------------------------------------------------------
  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random null baseline. Deterministic for a fixed seed.

    The prior-range pair (``prev_high``, ``prev_low``) is **permuted together**
    across days, so each day's open and intraday extremes are compared against an
    UNRELATED day's range. This is the joint null for both tiers at once: it
    destroys the temporal adjacency that the stat measures (does opening outside
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
  """Load data and compute Outside Days for the daily timeframe.

  Writes the result JSON to results/.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  stat = OutsideDays(instrument=instrument, config=config)
  result = stat.compute(candles_df)

  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Outside Days stat")
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
