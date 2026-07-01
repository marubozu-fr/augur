"""Previous Week's Range stat.

Measures how a trading week resolves against the prior week's RTH range, and —
when both sides of that range are taken — which level was hit first.

Two tiers of rows are reported:

  1. Outcome partition (condition ``prior_range``): over all countable weeks,
     each week falls into exactly ONE of four buckets relative to the prior
     week's range —
       - ``break_high_only`` : week high exceeds prior high, low stays inside.
       - ``break_low_only``  : week low falls below prior low, high stays inside.
       - ``break_both``      : both the prior high and the prior low are taken.
       - ``inside``          : neither level is taken (week stays inside).
     The four outcomes partition the countable weeks (counts sum to ``total``).

  2. Double-break sequence (condition ``break_both``): among the weeks that took
     both levels, which one was reached FIRST in chronological (intra-week) order?
       - ``high_first`` : the prior high was broken before the prior low.
       - ``low_first``  : the prior low was broken before the prior high.
     This split partitions the ``break_both`` weeks.

A break uses strict inequality: ``week_high > prev_high`` / ``week_low < prev_low``
(touching the level is not a break). The "prior week" is the chronologically
previous RESOLVED week, so it skips over any gap in the data.

Week boundaries are ISO weeks (Monday–Sunday); a week is keyed by its Monday
date. A week's range is built from the RTH bars of EVERY trading day in the week
(early-close days still contribute valid highs/lows). The LAST week present in
the data is treated as pending (its range may not be final) and is excluded from
every statistic. The FIRST resolved week has no prior week and is excluded from
every denominator (pending-sample discipline). ``total_samples`` counts ALL
resolved weeks; each row's ``total`` counts only the relevant countable weeks.

Sequence tiebreak: if a single 1-minute bar is the first to break BOTH levels,
the order is inferred from that bar's candle path — a bearish bar
(``close < open``) is assumed to print its high before its low (``high_first``),
a bullish bar its low before its high (``low_first``).
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

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="Previous Week's Range",
  fr="Range de la semaine précédente",
)
_DEFINITION = I18nString(
  en="How does a trading week resolve against the prior week's range — does it break the high only, the low only, both, or stay inside? And when both are taken, which level is hit first?",
  fr="Comment une semaine de bourse se résout-elle face au range de la semaine précédente — casse-t-elle seulement le plus haut, seulement le plus bas, les deux, ou reste-t-elle à l'intérieur ? Et quand les deux sont pris, quel niveau est atteint en premier ?",
)
_LABELS = Labels(
  conditions={
    "prior_range": I18nString(en="Prior week's range", fr="Range de la semaine précédente"),
    "break_both": I18nString(en="Broke both levels", fr="A cassé les deux niveaux"),
  },
  outcomes={
    "break_high_only": I18nString(en="Break prior high only", fr="Casse seulement le plus haut précédent"),
    "break_low_only": I18nString(en="Break prior low only", fr="Casse seulement le plus bas précédent"),
    "break_both": I18nString(en="Break both levels", fr="Casse les deux niveaux"),
    "inside": I18nString(en="Stay inside", fr="Reste à l'intérieur"),
    "high_first": I18nString(en="High taken first", fr="Plus haut atteint en premier"),
    "low_first": I18nString(en="Low taken first", fr="Plus bas atteint en premier"),
  },
)


class PrevWeeksRange(BaseStat):
  """Weekly outcome partition against the prior week's range plus break sequence."""

  stat_name = "prev_weeks_range"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = ()

  def __init__(
    self,
    instrument: str,
    config: InstrumentConfig,
  ) -> None:
    self.instrument = instrument
    self.timeframe = "weekly"
    self.config = config

    rth = config.sessions["rth"]
    self.rth_start_min: int = minute_of_day(rth.start)
    self.rth_end_min: int = minute_of_day(rth.end)

  # -------------------------------------------------------------------------
  # Week table construction
  # -------------------------------------------------------------------------
  def build_week_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-week table with prior-range and break-sequence columns.

    Indexed by the week's Monday date. Columns:
      week_high, week_low (RTH extremes over the whole week), prev_high,
      prev_low (the prior RESOLVED week's extremes; NaN for the first resolved
      week), seq_high_first (nullable bool: for a both-break week, whether the
      prior high was taken before the prior low; NA otherwise).

    The last week in the data is dropped (pending). ``prev_high`` / ``prev_low``
    are NaN for the first resolved week, so it is excluded downstream.
    """
    columns = ["week_high", "week_low", "prev_high", "prev_low", "seq_high_first"]
    empty = pd.DataFrame(columns=columns)

    if candles_df.empty:
      return empty

    df = candles_df.copy()
    df["_mod"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    rth_mask = (df["_mod"] >= self.rth_start_min) & (df["_mod"] < self.rth_end_min)
    rth = df[rth_mask].copy()
    if rth.empty:
      return empty

    # Week key = the Monday of the bar's ISO week (normalized, tz-aware).
    date = rth["timestamp"].dt.normalize()
    rth["_week"] = date - pd.to_timedelta(date.dt.dayofweek, unit="D")

    weekly = pd.DataFrame(
      {
        "week_high": rth.groupby("_week")["high"].max(),
        "week_low": rth.groupby("_week")["low"].min(),
      }
    ).sort_index()

    # Drop the last week: it may still be in progress (pending discipline).
    resolved = weekly.iloc[:-1].copy()
    if resolved.empty:
      return empty

    # Prior RESOLVED week's range (shift over the resolved-only, sorted index).
    resolved["prev_high"] = resolved["week_high"].shift(1)
    resolved["prev_low"] = resolved["week_low"].shift(1)

    # Break sequence: only computable for weeks that took both prior levels.
    both = (
      resolved["prev_high"].notna()
      & resolved["prev_low"].notna()
      & (resolved["week_high"] > resolved["prev_high"])
      & (resolved["week_low"] < resolved["prev_low"])
    )
    seq = pd.Series(pd.NA, index=resolved.index, dtype="boolean")
    if bool(both.any()):
      week_groups = {wk: g for wk, g in rth.groupby("_week")}
      for wk in resolved.index[both]:
        prev_high = resolved.at[wk, "prev_high"]
        prev_low = resolved.at[wk, "prev_low"]
        seq.at[wk] = _high_taken_first(week_groups[wk], prev_high, prev_low)
    resolved["seq_high_first"] = seq

    return resolved[columns]

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def compute_rows(
    self,
    week_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the outcome-partition and break-sequence rows.

    Only weeks with a prior resolved week (``prev_high`` not NaN) are countable.
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

    def _zero() -> list[StatResultRow]:
      return [
        _make("prior_range", "break_high_only", 0, 0),
        _make("prior_range", "break_low_only", 0, 0),
        _make("prior_range", "break_both", 0, 0),
        _make("prior_range", "inside", 0, 0),
        _make("break_both", "high_first", 0, 0),
        _make("break_both", "low_first", 0, 0),
      ]

    if week_table.empty:
      return _zero()

    countable = week_table["prev_high"].notna() & week_table["prev_low"].notna()
    broke_high = countable & (week_table["week_high"] > week_table["prev_high"])
    broke_low = countable & (week_table["week_low"] < week_table["prev_low"])
    both = broke_high & broke_low

    high_only = broke_high & ~broke_low
    low_only = broke_low & ~broke_high
    inside = countable & ~broke_high & ~broke_low

    countable_n = int(countable.sum())
    both_n = int(both.sum())

    # Sequence among both-break weeks (seq_high_first is non-null there).
    seq = week_table["seq_high_first"]
    high_first_n = int((both & seq.fillna(False).astype(bool)).sum())
    low_first_n = int((both & ~seq.fillna(True).astype(bool)).sum())

    return [
      # Tier 1 — four-way outcome partition over all countable weeks.
      _make("prior_range", "break_high_only", int(high_only.sum()), countable_n),
      _make("prior_range", "break_low_only", int(low_only.sum()), countable_n),
      _make("prior_range", "break_both", both_n, countable_n),
      _make("prior_range", "inside", int(inside.sum()), countable_n),
      # Tier 2 — which level was taken first, given both broke.
      _make("break_both", "high_first", high_first_n, both_n),
      _make("break_both", "low_first", low_first_n, both_n),
    ]

  def classify_samples(self, week_table: pd.DataFrame) -> list[SampleRow]:
    """One or two SampleRows per countable week, mirroring ``compute_rows``.

    Every countable week (a prior resolved week exists, i.e. ``prev_high`` /
    ``prev_low`` not NaN) yields exactly one tier-1 sample under condition
    ``prior_range``, whose outcome is the four-way partition of that week
    against the prior week's range: ``break_high_only`` / ``break_low_only`` /
    ``break_both`` / ``inside``.

    Each ``break_both`` week additionally yields exactly one tier-2 sample
    under condition ``break_both``, whose outcome (``high_first`` /
    ``low_first``) mirrors the week table's ``seq_high_first`` column — the
    same column ``compute_rows`` reads, so no sequence logic is recomputed
    here.

    Non-countable weeks (no prior resolved week) contribute nothing.
    """
    if week_table.empty:
      return []

    countable = week_table["prev_high"].notna() & week_table["prev_low"].notna()
    if not bool(countable.any()):
      return []

    samples: list[SampleRow] = []
    for ts, row in week_table[countable].iterrows():
      date_str = ts.strftime("%Y-%m-%d")
      broke_high = row["week_high"] > row["prev_high"]
      broke_low = row["week_low"] < row["prev_low"]
      if broke_high and broke_low:
        outcome = "break_both"
      elif broke_high:
        outcome = "break_high_only"
      elif broke_low:
        outcome = "break_low_only"
      else:
        outcome = "inside"
      samples.append(SampleRow(date=date_str, condition="prior_range", outcome=outcome))

      if broke_high and broke_low:
        seq_outcome = "high_first" if bool(row["seq_high_first"]) else "low_first"
        samples.append(SampleRow(date=date_str, condition="break_both", outcome=seq_outcome))

    return samples

  # -------------------------------------------------------------------------
  # Baseline
  # -------------------------------------------------------------------------
  def baseline_rows(self, week_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random null baseline. Deterministic for a fixed seed.

    The two tiers need different nulls, so each is computed on its own surrogate
    table and the relevant rows are stitched together:

      - Outcome partition (``prior_range``): the prior-range pair (``prev_high``,
        ``prev_low``) is permuted together across weeks, so each week is compared
        against an UNRELATED week's range. This is the null for the partition —
        temporal adjacency carries no information. The pair is permuted jointly
        so each prior range stays internally consistent (low <= high); the lone
        NaN pair simply moves to a random week, preserving the countable count.

      - Double-break sequence (``break_both``): the REAL prior ranges are kept
        (so the both-break set, hence ``total``, is preserved at full size) and
        only ``seq_high_first`` is reassigned by a fair coin — the null for the
        order in which the two levels are taken (~0.5 each). Permuting the prior
        ranges here would be wrong: over a long, trending history it dissolves
        almost every both-break, leaving a meaninglessly small baseline N.
    """
    n = len(week_table)
    if n == 0:
      return self.compute_rows(week_table, baseline_rows=None)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    partition = week_table.copy()
    partition["prev_high"] = week_table["prev_high"].to_numpy()[perm]
    partition["prev_low"] = week_table["prev_low"].to_numpy()[perm]
    partition_rows = self.compute_rows(partition, baseline_rows=None)

    sequence = week_table.copy()
    sequence["seq_high_first"] = rng.integers(0, 2, size=n).astype(bool)
    sequence_rows = self.compute_rows(sequence, baseline_rows=None)

    by_tier = {(r.condition, r.outcome): r for r in partition_rows}
    by_tier.update({
      (r.condition, r.outcome): r for r in sequence_rows if r.condition == "break_both"
    })
    return [by_tier[(r.condition, r.outcome)] for r in partition_rows]

  # BaseStat hooks delegate to the weekly table builder.
  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    return self.build_week_table(candles_df)


def _high_taken_first(week_bars: pd.DataFrame, prev_high: float, prev_low: float) -> bool:
  """Return True if the prior high is broken before the prior low within a week.

  ``week_bars`` are the week's RTH bars. The first bar (by timestamp) whose high
  exceeds ``prev_high`` and the first whose low falls below ``prev_low`` are
  compared. If a single bar is the first to break both, the order is inferred
  from that bar's candle path: a bearish bar (``close < open``) prints its high
  first; a bullish bar prints its low first.
  """
  bars = week_bars.sort_values("timestamp")
  high_break = bars[bars["high"] > prev_high]["timestamp"]
  low_break = bars[bars["low"] < prev_low]["timestamp"]
  t_high = high_break.iloc[0]
  t_low = low_break.iloc[0]
  if t_high < t_low:
    return True
  if t_low < t_high:
    return False
  # Same bar broke both — infer from candle direction.
  bar = bars[bars["timestamp"] == t_high].iloc[0]
  return bool(bar["close"] < bar["open"])


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
) -> Path:
  """Load data and compute Previous Week's Range for the weekly timeframe.

  Writes the result JSON to results/.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  stat = PrevWeeksRange(instrument=instrument, config=config)
  result = stat.compute(candles_df)

  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Previous Week's Range stat")
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
    print(f"\n  {tf}: {tf_data['total_samples']} resolved weeks | {tf_data['data_range']}")
    for row in tf_data["results"]:
      print(
        f"    {row['condition']} -> {row['outcome']}: "
        f"{row['probability']:.3f} (N={row['total']}, baseline={row['baseline_prob']:.3f})"
      )
