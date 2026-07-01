"""Average Consecutive Bars stat.

For each resolved trading day, on the intraday chart at a given timeframe,
find the longest run of consecutive green bars and the longest run of
consecutive red bars within the RTH session. Then average those per-day
maximums across all resolved days.

A "bar" is a time-bucket candle built from the 1-min bars in that
``(day, bucket)``: ``open`` is the first bar's open, ``close`` the last bar's
close. A bar is **green** when ``close >= open``, **red** otherwise.

This is a **magnitude** stat: each row's ``value`` holds the mean of daily
maximum streak lengths; ``value_baseline`` holds the permutation-null baseline.
``probability`` and ``baseline_prob`` are always ``0.0`` (unused channel).
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
  en="Average Consecutive Bars",
  fr="Bougies consécutives moyennes",
)
_DEFINITION = I18nString(
  en="For each trading day, the longest run of consecutive green and of consecutive red intraday bars, averaged across days. Shows the typical length of intraday color streaks.",
  fr="Pour chaque jour de bourse, la plus longue série de bougies intraday vertes consécutives et de rouges consécutives, moyennée sur l'ensemble des jours. Montre la longueur typique des séries de couleur intraday.",
)
_LABELS = Labels(
  conditions={
    "green": I18nString(en="Green bars", fr="Bougies vertes"),
    "red": I18nString(en="Red bars", fr="Bougies rouges"),
  },
  outcomes={
    "max_streak": I18nString(
      en="Avg longest run (bars)",
      fr="Série la plus longue moy. (bougies)",
    ),
  },
)

_OUTCOME_KEY = "max_streak"

# Supported bucket granularities, in minutes.
_TIMEFRAME_MINUTES: dict[str, int] = {
  "1min": 1,
  "5min": 5,
  "15min": 15,
  "30min": 30,
  "1h": 60,
}


def _longest_run(colors: np.ndarray, want_green: bool) -> int:
  """Return the length of the longest consecutive run of the target color.

  ``colors`` is a boolean array where ``True`` means green and ``False`` means
  red. When the array is empty or contains no bars of the wanted color,
  returns 0. Uses a vectorized cumsum-based grouping for efficiency.
  """
  if len(colors) == 0:
    return 0
  target = colors.astype(bool) if want_green else ~colors.astype(bool)
  if not target.any():
    return 0
  # Identify run boundaries: a new run starts whenever the color changes.
  changed = np.concatenate([[True], target[1:] != target[:-1]])
  groups = np.cumsum(changed)
  # Count bars in each run that matches the wanted color.
  counts = np.bincount(groups[target])
  return int(counts.max())


class AvgConsecutiveBars(BaseStat):
  """Mean daily-max run of consecutive green / red intraday bars.

  One instance covers a single bucket granularity (``timeframe``). The RTH
  session is partitioned into fixed-width time buckets; each bucket becomes one
  bar whose color is determined by close vs open. The longest consecutive green
  run and longest consecutive red run are found for each resolved day; the stat
  reports the mean of those per-day maxima.
  """

  stat_name = "avg_consecutive_bars"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = ()

  def __init__(
    self,
    instrument: str,
    config: InstrumentConfig,
    timeframe: str = "5min",
    close_tolerance_min: int = 15,
  ) -> None:
    if timeframe not in _TIMEFRAME_MINUTES:
      raise ValueError(
        f"Unknown timeframe '{timeframe}'. Choose from {list(_TIMEFRAME_MINUTES)}."
      )
    self.instrument = instrument
    self.timeframe = timeframe
    self.config = config
    self.bucket_min: int = _TIMEFRAME_MINUTES[timeframe]
    self.close_tolerance_min = close_tolerance_min

    rth = config.sessions["rth"]
    self.rth_start_min: int = minute_of_day(rth.start)
    self.rth_end_min: int = minute_of_day(rth.end)

  # -------------------------------------------------------------------------
  # Day table construction
  # -------------------------------------------------------------------------
  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-day table of max consecutive streak lengths from 1-min data.

    Each row is one resolved trading day, indexed by the normalized session
    date. Columns:
      - ``max_green_streak`` (float): longest run of consecutive green bucket
        bars that day.
      - ``max_red_streak`` (float): longest run of consecutive red bucket bars
        that day.
      - ``bar_colors`` (object): numpy bool array of that day's bar colors in
        chronological bucket order (True = green). Used by the baseline to
        permute within each day.

    Days without any RTH bucket bars are excluded (they produce NaN streaks
    after the reindex and are dropped). Pending days (those not in the
    resolved-days table) are excluded via the reindex-then-dropna discipline.
    """
    _columns = ["max_green_streak", "max_red_streak", "bar_colors"]
    empty = pd.DataFrame(columns=_columns)

    if candles_df.empty:
      return empty

    resolved = build_resolved_days(
      candles_df, self.rth_start_min, self.rth_end_min, self.close_tolerance_min
    )
    if resolved.empty:
      return empty

    df = candles_df.copy()
    df["_mod"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["_date"] = df["timestamp"].dt.normalize()

    rth_mask = (df["_mod"] >= self.rth_start_min) & (df["_mod"] < self.rth_end_min)
    rth = df[rth_mask].copy()
    if rth.empty:
      return empty

    rth["_bucket"] = (rth["_mod"] - self.rth_start_min) // self.bucket_min

    # Per (date, bucket): open = first bar's open, close = last bar's close.
    sorted_rth = rth.sort_values("_mod")
    bucket_open = sorted_rth.groupby(["_date", "_bucket"])["open"].first()
    bucket_close = sorted_rth.groupby(["_date", "_bucket"])["close"].last()
    bucket_df = pd.DataFrame({"_open": bucket_open, "_close": bucket_close}).reset_index()
    bucket_df["_green"] = bucket_df["_close"] >= bucket_df["_open"]

    # ---- Vectorized streak computation ------------------------------------
    bucket_df = bucket_df.sort_values(["_date", "_bucket"]).reset_index(drop=True)

    # A new run starts when the color changes or when we cross a day boundary.
    color_changed = bucket_df["_green"] != bucket_df["_green"].shift(1)
    day_changed = bucket_df["_date"] != bucket_df["_date"].shift(1)
    bucket_df["_is_new_run"] = (color_changed | day_changed).astype(int)
    bucket_df["_run_id"] = bucket_df["_is_new_run"].cumsum()

    # Run length = count of bars sharing the same run ID.
    run_len = bucket_df.groupby("_run_id").size().rename("_run_len")
    bucket_df = bucket_df.join(run_len, on="_run_id")

    # Per (date, color): max run length. Unstack the color dimension.
    # fill_value=0 handles days that are entirely one color (the other streak = 0).
    day_color_max = (
      bucket_df.groupby(["_date", "_green"])["_run_len"]
      .max()
      .unstack("_green", fill_value=0)
    )
    # Ensure both color columns exist even if all bars are the same color.
    for col in [True, False]:
      if col not in day_color_max.columns:
        day_color_max[col] = 0

    # ---- bar_colors per day (object column for baseline permutations) -----
    bar_colors_per_day: pd.Series = (
      bucket_df.sort_values(["_date", "_bucket"])
      .groupby("_date")["_green"]
      .apply(lambda x: x.to_numpy(dtype=bool))
    )

    # ---- Assemble output aligned to the resolved-days index --------------
    out = pd.DataFrame(index=resolved.index)
    out["max_green_streak"] = day_color_max[True].reindex(resolved.index).astype(float)
    out["max_red_streak"] = day_color_max[False].reindex(resolved.index).astype(float)
    out["bar_colors"] = bar_colors_per_day.reindex(resolved.index)

    # Drop days with no RTH bucket bars (max_green_streak / max_red_streak are NaN).
    out = out.dropna(subset=["max_green_streak", "max_red_streak"])

    return out[_columns]

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute two magnitude rows: mean daily-max green streak and red streak.

    ``value`` is the mean of ``max_green_streak`` / ``max_red_streak`` over all
    contributing days (non-NaN entries). ``count`` = ``total`` = number of
    contributing days. ``probability`` and ``baseline_prob`` are ``0.0``
    (magnitude-only stat). ``value_baseline`` is merged from ``baseline_rows``
    when provided.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    def mag_row(condition: str, col: str) -> StatResultRow:
      series = (
        day_table[col].dropna()
        if (not day_table.empty and col in day_table.columns)
        else pd.Series([], dtype=float)
      )
      n = len(series)
      value: float | None = float(series.mean()) if n > 0 else None
      bl = baseline_map.get((condition, _OUTCOME_KEY))
      return StatResultRow(
        condition=condition,
        outcome=_OUTCOME_KEY,
        count=n,
        total=n,
        probability=0.0,
        baseline_prob=0.0,
        baseline_n=bl.total if bl else 0,
        value=value,
        value_baseline=bl.value if bl else None,
      )

    return [
      mag_row("green", "max_green_streak"),
      mag_row("red", "max_red_streak"),
    ]

  # -------------------------------------------------------------------------
  # Baseline
  # -------------------------------------------------------------------------
  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Permutation null: shuffle each day's bar colors, recompute max streaks.

    For each resolved day, the ``bar_colors`` array (which preserves the day's
    green/red counts) is randomly permuted using a single seeded RNG. The max
    green and max red streaks are recomputed on the shuffled array. Averaging
    these shuffled maxima across days gives the null: what streak length would
    occur if bar colors within each day were randomly ordered (no clustering).
    Days are iterated in stable index order for reproducibility. Uses
    ``np.random.default_rng(seed)`` for deterministic output.
    """
    def _empty_row(cond: str) -> StatResultRow:
      return StatResultRow(
        condition=cond,
        outcome=_OUTCOME_KEY,
        count=0,
        total=0,
        probability=0.0,
        baseline_prob=0.0,
        baseline_n=0,
        value=None,
        value_baseline=None,
      )

    if day_table.empty or "bar_colors" not in day_table.columns:
      return [_empty_row("green"), _empty_row("red")]

    rng = np.random.default_rng(seed)

    green_maxes: list[float] = []
    red_maxes: list[float] = []

    # Iterate in stable index order (chronological, same as day_table.index).
    for colors in day_table["bar_colors"]:
      if not isinstance(colors, np.ndarray) or len(colors) == 0:
        continue
      shuffled = colors.copy()
      rng.shuffle(shuffled)
      green_maxes.append(float(_longest_run(shuffled, True)))
      red_maxes.append(float(_longest_run(shuffled, False)))

    n_green = len(green_maxes)
    n_red = len(red_maxes)

    bl_green_val: float | None = float(np.mean(green_maxes)) if n_green > 0 else None
    bl_red_val: float | None = float(np.mean(red_maxes)) if n_red > 0 else None

    return [
      StatResultRow(
        condition="green",
        outcome=_OUTCOME_KEY,
        count=n_green,
        total=n_green,
        probability=0.0,
        baseline_prob=0.0,
        baseline_n=0,
        value=bl_green_val,
        value_baseline=None,
      ),
      StatResultRow(
        condition="red",
        outcome=_OUTCOME_KEY,
        count=n_red,
        total=n_red,
        probability=0.0,
        baseline_prob=0.0,
        baseline_n=0,
        value=bl_red_val,
        value_baseline=None,
      ),
    ]

  def classify_samples(self, day_table: pd.DataFrame) -> list[SampleRow]:
    """One SampleRow per (day, color) with a non-NaN max streak that day."""
    if day_table.empty:
      return []

    samples: list[SampleRow] = []
    for ts, row in day_table.iterrows():
      date = ts.strftime("%Y-%m-%d")
      green = row["max_green_streak"]
      if pd.notna(green):
        samples.append(SampleRow(date=date, condition="green", outcome=_OUTCOME_KEY, value=float(green)))
      red = row["max_red_streak"]
      if pd.notna(red):
        samples.append(SampleRow(date=date, condition="red", outcome=_OUTCOME_KEY, value=float(red)))
    return samples


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  timeframe: str = "5min",
  config_dir: str = "config",
  data_path: str | None = None,
) -> Path:
  """Compute Average Consecutive Bars for one bucket granularity. Writes JSON."""
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  stat = AvgConsecutiveBars(
    instrument=instrument,
    config=config,
    timeframe=timeframe,
  )
  result = stat.compute(candles_df)

  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Average Consecutive Bars stat")
  parser.add_argument("--instrument", default="NQ", help="Instrument name (default: NQ)")
  parser.add_argument(
    "--timeframe",
    default="5min",
    choices=list(_TIMEFRAME_MINUTES),
    help="Bucket granularity (default: 5min)",
  )
  parser.add_argument("--data-path", default=None, help="Override parquet file path")
  parser.add_argument("--config-dir", default="config", help="Config directory (default: config)")
  args = parser.parse_args()

  output_path = run(
    instrument=args.instrument,
    timeframe=args.timeframe,
    config_dir=args.config_dir,
    data_path=args.data_path,
  )

  print(f"Written: {output_path}")

  with open(output_path, encoding="utf-8") as f:
    data = json.load(f)

  instrument_data = data["instruments"][args.instrument]
  for tf, tf_data in instrument_data.items():
    print(f"\n  {tf}: {tf_data['total_samples']} resolved days | {tf_data['data_range']}")
    for r in tf_data["results"]:
      v = r.get("value")
      vbl = r.get("value_baseline")
      v_str = f"{v:.3f}" if v is not None else "N/A"
      vbl_str = f"{vbl:.3f}" if vbl is not None else "N/A"
      print(
        f"    {r['condition']}: avg_max_streak={v_str} "
        f"(N={r['total']}) "
        f"baseline={vbl_str}"
      )
