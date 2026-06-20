"""Opening Week Range stat — standard variant.

Measures the high-low RTH range formed by the FIRST ``open_days`` trading
sessions of each ISO week (default 2), then classifies how the REST of that
week resolves against it.

Definitions:
  - **Opening week range**: the high-low range across the first ``open_days``
    RTH sessions of the week. ``opening_high`` is the max high and
    ``opening_low`` the min low over those sessions; ``opening_size`` is
    ``opening_high - opening_low``.
  - **Breakout window**: the remaining trading sessions of the same week
    (strictly after the opening sessions). The opening sessions themselves are
    excluded so the range can never break its own bars.
  - **Break**: strict inequality — ``post_high > opening_high`` or
    ``post_low < opening_low`` (touching the level exactly is NOT a break).

Two tiers of rows are reported:

  1. Outcome partition (condition ``opening_range``): over all countable weeks,
     each week falls into exactly ONE of four buckets —
       - ``break_high_only`` : post high exceeds opening high, low held inside.
       - ``break_low_only``  : post low falls below opening low, high held.
       - ``break_both``      : both the opening high and the opening low are taken.
       - ``inside``          : neither level is taken (week stays inside).
     The four outcomes partition the countable weeks (counts sum to ``total``).

  2. Double-break sequence (condition ``break_both``): among the weeks that
     broke both levels, which one was reached FIRST in chronological order?
       - ``high_first`` : the opening high was broken before the opening low.
       - ``low_first``  : the opening low was broken before the opening high.
     This split partitions the ``break_both`` weeks.

A week is countable only when it has at least ``open_days + 1`` trading
sessions (a full opening window AND a non-empty breakout window). The LAST week
present in the data is always excluded (pending discipline — it may still be in
progress). Sequence tiebreak: if a single bar first breaks both levels, the
order is inferred from candle direction — a bearish bar (``close < open``)
prints its high first (``high_first``), a bullish bar its low first
(``low_first``).

Results are computed for two opening lengths, each emitted as its own timeframe
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
  SizeBucket,
  StatResultRow,
  StatRunResult,
  TimeframeResult,
  write_results,
)
from stats.config import InstrumentConfig, load_config, minute_of_day

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="Opening Week Range",
  fr="Range de la semaine d'ouverture",
)
_DEFINITION = I18nString(
  en="After the opening week range forms from the first N sessions of the week, how often does the rest of the week break above the range high, below the range low, both sides, or neither — and when both are taken, which level is reached first?",
  fr="Après la formation du range de la semaine d'ouverture sur les N premières séances, à quelle fréquence le reste de la semaine casse-t-il au-dessus du plus haut du range, en-dessous du plus bas du range, des deux côtés, ou ni l'un ni l'autre — et quand les deux sont pris, quel niveau est atteint en premier ?",
)
_LABELS = Labels(
  conditions={
    "opening_range": I18nString(
      en="Opening week range",
      fr="Range de la semaine d'ouverture",
    ),
    "break_both": I18nString(
      en="Broke both levels",
      fr="A cassé les deux niveaux",
    ),
  },
  outcomes={
    "break_high_only": I18nString(
      en="Break opening high only",
      fr="Casse seulement le plus haut d'ouverture",
    ),
    "break_low_only": I18nString(
      en="Break opening low only",
      fr="Casse seulement le plus bas d'ouverture",
    ),
    "break_both": I18nString(
      en="Break both levels",
      fr="Casse les deux niveaux",
    ),
    "inside": I18nString(
      en="Stay inside",
      fr="Reste à l'intérieur",
    ),
    "high_first": I18nString(
      en="High taken first",
      fr="Plus haut atteint en premier",
    ),
    "low_first": I18nString(
      en="Low taken first",
      fr="Plus bas atteint en premier",
    ),
  },
)

# Timeframe string → number of opening sessions
_TF_OPEN_DAYS: dict[str, int] = {
  "1d": 1,
  "2d": 2,
}


def _high_taken_first(
  post_bars: pd.DataFrame,
  opening_high: float,
  opening_low: float,
) -> bool:
  """Return True if the opening high is broken before the opening low.

  ``post_bars`` are the breakout-window RTH bars sorted by timestamp. The first
  bar whose high exceeds ``opening_high`` and the first whose low falls below
  ``opening_low`` are compared. If a single bar is the first to break both, the
  order is inferred from that bar's candle path: a bearish bar (``close < open``)
  prints its high first; a bullish bar prints its low first.
  """
  bars = post_bars.sort_values("timestamp")
  high_break = bars[bars["high"] > opening_high]["timestamp"]
  low_break = bars[bars["low"] < opening_low]["timestamp"]
  t_high = high_break.iloc[0]
  t_low = low_break.iloc[0]
  if t_high < t_low:
    return True
  if t_low < t_high:
    return False
  # Same bar broke both — infer from candle direction.
  bar = bars[bars["timestamp"] == t_high].iloc[0]
  return bool(bar["close"] < bar["open"])


class OpeningWeekRange(BaseStat):
  """Weekly outcome partition against the opening week range plus break sequence."""

  stat_name = "opening_week_range"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = (SizeBucket(column="opening_size", preset="quartiles", name="size"),)

  def __init__(
    self,
    instrument: str,
    timeframe: str,
    config: InstrumentConfig,
    close_tolerance_min: int = 15,
  ) -> None:
    if timeframe not in _TF_OPEN_DAYS:
      raise ValueError(
        f"Unsupported timeframe '{timeframe}'. Choose from {list(_TF_OPEN_DAYS)}"
      )
    self.instrument = instrument
    self.timeframe = timeframe
    self.config = config
    self.open_days = _TF_OPEN_DAYS[timeframe]
    self.close_tolerance_min = close_tolerance_min

    rth = config.sessions["rth"]
    self.rth_start_min: int = minute_of_day(rth.start)
    self.rth_end_min: int = minute_of_day(rth.end)

  # -------------------------------------------------------------------------
  # Week table construction
  # -------------------------------------------------------------------------
  def build_week_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-week table with opening-range and breakout-window columns.

    Indexed by the week's Monday date (normalized, tz-aware). Columns:
      ``opening_high`` / ``opening_low`` — RTH extremes over the opening sessions.
      ``opening_size``                   — ``opening_high - opening_low``.
      ``post_high`` / ``post_low``       — RTH extremes over the breakout window.
      ``seq_high_first``                 — nullable bool; True when the opening
                                           high was taken before the opening low
                                           (only set for ``break_both`` weeks).

    The last week in the data is dropped (pending). A week with no breakout
    window (``<= open_days`` trading sessions) has ``post_high`` / ``post_low``
    as NaN and is excluded downstream (pending-sample discipline).
    """
    columns = [
      "opening_high",
      "opening_low",
      "opening_size",
      "post_high",
      "post_low",
      "seq_high_first",
    ]
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
    rth["_date"] = date

    # All unique week keys, sorted.
    all_weeks = sorted(rth["_week"].unique())

    # Drop the last week: it may still be in progress (pending discipline).
    if len(all_weeks) <= 1:
      return empty
    resolved_weeks = all_weeks[:-1]

    # Build a lookup: week key → group of RTH bars.
    week_groups = {wk: grp for wk, grp in rth.groupby("_week")}

    rows: list[dict] = []
    for wk in resolved_weeks:
      bars = week_groups[wk]
      # Sorted unique trading dates for this week.
      trade_dates = sorted(bars["_date"].unique())
      opening_dates = set(trade_dates[: self.open_days])
      post_dates = set(trade_dates[self.open_days :])

      opening_bars = bars[bars["_date"].isin(opening_dates)]
      opening_high = float(opening_bars["high"].max())
      opening_low = float(opening_bars["low"].min())
      opening_size = opening_high - opening_low

      if not post_dates:
        # No breakout window — non-countable (pending-sample discipline).
        rows.append({
          "week": wk,
          "opening_high": opening_high,
          "opening_low": opening_low,
          "opening_size": opening_size,
          "post_high": float("nan"),
          "post_low": float("nan"),
          "seq_high_first": pd.NA,
        })
        continue

      post_bars = bars[bars["_date"].isin(post_dates)]
      post_high = float(post_bars["high"].max())
      post_low = float(post_bars["low"].min())

      # Sequence: only for weeks that broke both levels.
      broke_both = (post_high > opening_high) and (post_low < opening_low)
      if broke_both:
        seq = _high_taken_first(post_bars, opening_high, opening_low)
      else:
        seq = pd.NA

      rows.append({
        "week": wk,
        "opening_high": opening_high,
        "opening_low": opening_low,
        "opening_size": opening_size,
        "post_high": post_high,
        "post_low": post_low,
        "seq_high_first": seq,
      })

    if not rows:
      return empty

    week_table = (
      pd.DataFrame(rows)
      .set_index("week")
      .sort_index()
    )
    week_table["seq_high_first"] = week_table["seq_high_first"].astype("boolean")
    return week_table[columns]

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def compute_rows(
    self,
    week_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the outcome-partition and break-sequence rows.

    A week is countable when it has a valid opening range AND a non-empty
    breakout window (all four price columns are non-NaN). If ``baseline_rows``
    is provided, merges ``baseline_prob`` / ``baseline_n``.
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
        _make("opening_range", "break_high_only", 0, 0),
        _make("opening_range", "break_low_only", 0, 0),
        _make("opening_range", "break_both", 0, 0),
        _make("opening_range", "inside", 0, 0),
        _make("break_both", "high_first", 0, 0),
        _make("break_both", "low_first", 0, 0),
      ]

    if week_table.empty:
      return _zero()

    countable = (
      week_table["opening_high"].notna()
      & week_table["opening_low"].notna()
      & week_table["post_high"].notna()
      & week_table["post_low"].notna()
    )
    broke_high = countable & (week_table["post_high"] > week_table["opening_high"])
    broke_low = countable & (week_table["post_low"] < week_table["opening_low"])
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
      _make("opening_range", "break_high_only", int(high_only.sum()), countable_n),
      _make("opening_range", "break_low_only", int(low_only.sum()), countable_n),
      _make("opening_range", "break_both", both_n, countable_n),
      _make("opening_range", "inside", int(inside.sum()), countable_n),
      # Tier 2 — which level was taken first, given both broke.
      _make("break_both", "high_first", high_first_n, both_n),
      _make("break_both", "low_first", low_first_n, both_n),
    ]

  # -------------------------------------------------------------------------
  # Baseline
  # -------------------------------------------------------------------------
  def baseline_rows(self, week_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random null baseline. Deterministic for a fixed seed.

    Two tiers are computed on separate surrogate tables and stitched together:

      - Outcome partition (``opening_range``): each week's breakout extremes are
        REFLECTED around the opening-range midpoint with probability 0.5. A price
        ``p`` maps to ``2*mid - p``, swapping ``post_high`` and ``post_low``.
        Reflection turns a ``break_high_only`` week into a ``break_low_only``
        week and vice versa, while ``break_both`` and ``inside`` are
        direction-symmetric and unchanged. The null has NO directional bias, so
        the comparison reveals whether the week breaks UP more often than DOWN
        beyond a coin flip. NaN post values propagate through the arithmetic,
        preserving the countable set.

      - Double-break sequence (``break_both``): the REAL opening ranges are kept
        (so the both-break set, hence ``total``, stays at full size) and only
        ``seq_high_first`` is reassigned by a fair coin — the null for the order
        in which the two levels are taken. Uses ``np.random.default_rng(seed)``.
    """
    n = len(week_table)
    if n == 0:
      return self.compute_rows(week_table, baseline_rows=None)

    rng = np.random.default_rng(seed)
    flip = rng.integers(0, 2, size=n).astype(bool)

    mid = (
      week_table["opening_high"].to_numpy() + week_table["opening_low"].to_numpy()
    ) / 2.0
    post_high = week_table["post_high"].to_numpy(dtype=float, na_value=float("nan"))
    post_low = week_table["post_low"].to_numpy(dtype=float, na_value=float("nan"))

    refl_post_high = 2.0 * mid - post_low
    refl_post_low = 2.0 * mid - post_high

    partition = week_table.copy()
    partition["post_high"] = np.where(flip, refl_post_high, post_high)
    partition["post_low"] = np.where(flip, refl_post_low, post_low)
    partition_rows = self.compute_rows(partition, baseline_rows=None)

    sequence = week_table.copy()
    sequence["seq_high_first"] = rng.integers(0, 2, size=n).astype(bool)
    sequence_rows = self.compute_rows(sequence, baseline_rows=None)

    by_tier = {(r.condition, r.outcome): r for r in partition_rows}
    by_tier.update({
      (r.condition, r.outcome): r
      for r in sequence_rows
      if r.condition == "break_both"
    })
    return [by_tier[(r.condition, r.outcome)] for r in partition_rows]

  # BaseStat hook delegates to the weekly table builder.
  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    return self.build_week_table(candles_df)


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
) -> Path:
  """Load data and compute Opening Week Range for both opening lengths.

  Merges the per-timeframe TimeframeResults into one StatRunResult and writes
  the consolidated JSON to results/.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  timeframes = ["1d", "2d"]
  merged_tf: dict[str, TimeframeResult] = {}
  labels = _LABELS.model_copy(deep=True)

  for tf in timeframes:
    stat = OpeningWeekRange(
      instrument=instrument,
      timeframe=tf,
      config=config,
    )
    result = stat.compute(candles_df)
    merged_tf[tf] = result.instruments[instrument][tf]
    # Merge the framework-enriched slice dimension labels so no timeframe's
    # dimensions overwrite an earlier one's.
    labels.dimensions.update(result.labels.dimensions)

  final_result = StatRunResult(
    stat_name="opening_week_range",
    title=_TITLE,
    definition=_DEFINITION,
    labels=labels,
    instruments={instrument: merged_tf},
  )

  return write_results(final_result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Opening Week Range stat")
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
