"""High & Low by Weekday stat.

For each ISO week, which weekday produced the weekly high (RTH intraday high),
the weekly low (RTH intraday low), the weekly high close, and the weekly low
close? Aggregated across weeks: for each weekday, the fraction of weeks whose
extreme fell on that weekday. Probabilities for a given condition sum to ~100%
across weekdays (each week contributes exactly one weekday per condition).

The stat operates at ``weekly`` granularity. Each resolved ISO week contributes
exactly one observation per condition — the weekday that produced the extreme.
The final ISO week present in the data is ALWAYS excluded as a pending sample
(it may be in progress and its extremes not yet final).

Four conditions are reported:
  - ``weekly_high``:       weekday with the highest RTH intraday high.
  - ``weekly_low``:        weekday with the lowest RTH intraday low.
  - ``weekly_high_close``: weekday with the highest RTH session close.
  - ``weekly_low_close``:  weekday with the lowest RTH session close.

A custom ``WeeklyCandle`` slicer (subclass of ``_ColorSlicer``) splits weeks by
whether the weekly candle is green (close >= open) or red.
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
  _ColorSlicer,
  _WEEKDAYS,
  write_results,
)
from stats.config import InstrumentConfig, load_config, minute_of_day
from stats.utils.daily_candles import build_resolved_days

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="High & Low by Weekday",
  fr="Plus haut & plus bas par jour de la semaine",
)
_DEFINITION = I18nString(
  en="For each ISO week, which weekday produced the weekly high, weekly low, weekly high close, and weekly low close? Aggregated as the fraction of weeks each weekday claims each extreme.",
  fr="Pour chaque semaine ISO, quel jour de la semaine a produit le plus haut hebdomadaire, le plus bas hebdomadaire, le plus haut de clôture hebdomadaire et le plus bas de clôture hebdomadaire ? Agrégé sous forme de fraction des semaines où chaque jour revendique chaque extrême.",
)

# Condition keys and their i18n labels, in display order.
_CONDITIONS: list[tuple[str, I18nString]] = [
  ("weekly_high", I18nString(en="Weekly high", fr="Plus haut hebdomadaire")),
  ("weekly_low", I18nString(en="Weekly low", fr="Plus bas hebdomadaire")),
  ("weekly_high_close", I18nString(en="Weekly high by close", fr="Plus haut hebdomadaire (clôture)")),
  ("weekly_low_close", I18nString(en="Weekly low by close", fr="Plus bas hebdomadaire (clôture)")),
]

# Maps each condition key to the day_table column that carries its weekday int.
_CONDITION_COLUMN: dict[str, str] = {
  "weekly_high": "high_weekday",
  "weekly_low": "low_weekday",
  "weekly_high_close": "high_close_weekday",
  "weekly_low_close": "low_close_weekday",
}

# Build outcome labels from the canonical _WEEKDAYS list in base.py.
_OUTCOME_LABELS: dict[str, I18nString] = {
  key: I18nString(en=en, fr=fr) for _, key, en, fr in _WEEKDAYS
}

_LABELS = Labels(
  conditions={key: label for key, label in _CONDITIONS},
  outcomes=_OUTCOME_LABELS,
)

# Ordered list of (int, key) for weekday iteration.
_WEEKDAY_ORDER: list[tuple[int, str]] = [(num, key) for num, key, _, _ in _WEEKDAYS]


# ---------------------------------------------------------------------------
# Custom slicer
# ---------------------------------------------------------------------------
class WeeklyCandle(_ColorSlicer):
  """Split weeks by the weekly candle color (green = close >= open)."""

  def __init__(self) -> None:
    super().__init__(column="weekly_green", name="weekly_candle")

  def dimension_label(self) -> I18nString:
    return I18nString(en="Weekly candle", fr="Bougie hebdomadaire")


# ---------------------------------------------------------------------------
# Stat implementation
# ---------------------------------------------------------------------------
class HighLowWeekday(BaseStat):
  """Fraction of weeks whose extreme (high/low, intraday or close) fell on each weekday.

  One instance covers the single ``weekly`` timeframe. The ``WeeklyCandle``
  slicer re-runs the core computation over the green-week and red-week subsets.
  """

  stat_name = "high_low_weekday"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = (WeeklyCandle(),)

  def __init__(
    self,
    instrument: str,
    config: InstrumentConfig,
    close_tolerance_min: int = 15,
  ) -> None:
    self.instrument = instrument
    self.timeframe = "weekly"
    self.config = config
    self.close_tolerance_min = close_tolerance_min

    rth = config.sessions["rth"]
    self.rth_start_min: int = minute_of_day(rth.start)
    self.rth_end_min: int = minute_of_day(rth.end)

  # -------------------------------------------------------------------------
  # Day / week table construction
  # -------------------------------------------------------------------------
  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-week table from raw 1-min OHLCV data.

    Steps:
    1. Build the resolved-days table (session_open, session_close) via
       ``build_resolved_days``.
    2. Compute per-day RTH ``day_high`` (max of high over RTH bars) and
       ``day_low`` (min of low over RTH bars), then join onto the resolved-days
       index (inner join — only resolved dates pass through).
    3. Group resolved days by ISO (year, week). For each week compute the four
       extreme-weekday columns, the weekly_green flag, and present_weekdays.
    4. Exclude the final ISO week (max year+week key) as a pending sample.

    Returns a DataFrame indexed by the first session date of each week, with
    columns: high_weekday, low_weekday, high_close_weekday, low_close_weekday,
    weekly_green, present_weekdays.
    """
    columns = [
      "high_weekday",
      "low_weekday",
      "high_close_weekday",
      "low_close_weekday",
      "weekly_green",
      "present_weekdays",
    ]
    empty = pd.DataFrame(columns=columns)

    if candles_df.empty:
      return empty

    # --- Step 1: resolved daily summaries ---
    resolved = build_resolved_days(
      candles_df, self.rth_start_min, self.rth_end_min, self.close_tolerance_min
    )
    if resolved.empty:
      return empty

    # --- Step 2: per-day RTH high/low ---
    df = candles_df.copy()
    df["_mod"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["_date"] = df["timestamp"].dt.normalize()

    rth_mask = (df["_mod"] >= self.rth_start_min) & (df["_mod"] < self.rth_end_min)
    rth = df[rth_mask]

    day_high = rth.groupby("_date")["high"].max().rename("day_high")
    day_low = rth.groupby("_date")["low"].min().rename("day_low")

    # Join onto resolved index: keeps only resolved dates.
    daily = resolved.join(day_high, how="inner").join(day_low, how="inner")
    if daily.empty:
      return empty

    # --- Step 3: group by ISO (year, week) ---
    idx = daily.index
    iso = idx.isocalendar()
    year_arr = iso["year"].to_numpy()
    week_arr = iso["week"].to_numpy()
    iso_keys = list(zip(year_arr.tolist(), week_arr.tolist()))

    daily = daily.copy()
    daily["_iso_key"] = iso_keys
    daily["_dow"] = idx.dayofweek
    daily["_date_col"] = idx

    def _agg_week(grp: pd.DataFrame) -> pd.Series:
      high_idx = grp["day_high"].idxmax()
      low_idx = grp["day_low"].idxmin()
      hc_idx = grp["session_close"].idxmax()
      lc_idx = grp["session_close"].idxmin()
      # The group is already chronological (inherited from build_resolved_days'
      # sort), so positional access gives the week's first/last trading day.
      week_open = grp["session_open"].iloc[0]
      week_close = grp["session_close"].iloc[-1]
      present = tuple(sorted(grp["_dow"].unique().tolist()))
      return pd.Series(
        {
          "high_weekday": int(grp.loc[high_idx, "_dow"]),
          "low_weekday": int(grp.loc[low_idx, "_dow"]),
          "high_close_weekday": int(grp.loc[hc_idx, "_dow"]),
          "low_close_weekday": int(grp.loc[lc_idx, "_dow"]),
          "weekly_green": bool(week_close >= week_open),
          "present_weekdays": present,
          "week_first_date": grp["_date_col"].iloc[0],
        }
      )

    weekly = daily.groupby("_iso_key", sort=False).apply(_agg_week, include_groups=False)

    # Use week_first_date as the index (first session date of each week).
    weekly.index = pd.DatetimeIndex(weekly["week_first_date"])
    weekly = weekly.drop(columns=["week_first_date"]).sort_index()

    # --- Step 4: exclude the final ISO week (pending discipline) ---
    # The last ISO week may still be in progress; its extremes are not final.
    # We identify the max (year, week) key that was present.
    if len(weekly) <= 1:
      # 0 or 1 weeks: after excluding the final week, nothing remains.
      return empty

    weekly = weekly.iloc[:-1]  # drop the last row (already sorted chronologically)

    return weekly[columns]

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute condition/outcome rows over a (possibly sliced) week table.

    For each of the four conditions and each weekday present in the subset,
    counts how many weeks had their extreme on that weekday, and divides by
    the total number of weeks in the subset.

    Probabilities for a given condition sum to 1.0 across the weekdays that
    appear in the subset (each week contributes exactly one weekday per condition).

    ``value`` / ``value_baseline`` are left as None (probability-only stat).
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    total = len(day_table)

    # Determine the union of present weekdays across all weeks, in _WEEKDAYS order.
    if total == 0:
      return []

    all_present: set[int] = set()
    for pw in day_table["present_weekdays"]:
      all_present.update(pw)
    # Filter _WEEKDAY_ORDER to those present, preserving Mon→Sun order.
    present_ordered = [(num, key) for num, key in _WEEKDAY_ORDER if num in all_present]

    rows: list[StatResultRow] = []
    for cond_key, _ in _CONDITIONS:
      col = _CONDITION_COLUMN[cond_key]
      counts = day_table[col].value_counts()
      for wd_num, wd_key in present_ordered:
        count = int(counts.get(wd_num, 0))
        probability = count / total if total > 0 else 0.0
        bl = baseline_map.get((cond_key, wd_key))
        rows.append(
          StatResultRow(
            condition=cond_key,
            outcome=wd_key,
            count=count,
            total=total,
            probability=probability,
            baseline_prob=bl.probability if bl else 0.0,
            baseline_n=bl.total if bl else 0,
            value=None,
            value_baseline=None,
          )
        )
    return rows

  # -------------------------------------------------------------------------
  # Baseline
  # -------------------------------------------------------------------------
  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random null baseline: each week's extreme weekday is drawn uniformly at random.

    For each week, four independent uniform draws are made from that week's
    ``present_weekdays`` — one per condition. This represents the null hypothesis
    that extremes fall uniformly across trading days. Uses ``np.random.default_rng``
    with a fixed seed for deterministic output. Operates on the passed table so the
    framework can compute a per-slice-group baseline as well as the overall one.
    """
    rng = np.random.default_rng(seed)
    n = len(day_table)
    if n == 0:
      return self.compute_rows(day_table, baseline_rows=None)

    tmp = day_table.copy()
    condition_cols = [_CONDITION_COLUMN[cond_key] for cond_key, _ in _CONDITIONS]

    random_weekdays: dict[str, list[int]] = {col: [] for col in condition_cols}
    for pw in tmp["present_weekdays"]:
      pw_list = list(pw)
      for col in condition_cols:
        random_weekdays[col].append(int(rng.choice(pw_list)))

    for col in condition_cols:
      tmp[col] = random_weekdays[col]

    return self.compute_rows(tmp, baseline_rows=None)


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
) -> Path:
  """Compute High & Low by Weekday for the weekly timeframe.

  Writes the result JSON to results/.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  stat = HighLowWeekday(instrument=instrument, config=config)
  result = stat.compute(candles_df)

  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute High & Low by Weekday stat")
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
      cond = row["condition"]
      outcome = row["outcome"]
      prob = row["probability"]
      n = row["total"]
      count = row["count"]
      if count == max(
        r["count"] for r in tf_data["results"] if r["condition"] == cond
      ):
        print(
          f"    {cond}: peak weekday={outcome} P={prob:.3f} (count={count}, N={n})"
        )
