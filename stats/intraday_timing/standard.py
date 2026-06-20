"""Intraday Timing stat.

For each resolved RTH session, which intraday time bucket produced the day's
high and which produced the day's low? Aggregated across days: for each bucket,
the fraction of days whose extreme fell in that bucket. Probabilities for a
given condition sum to ~100% across buckets (each day contributes exactly one
bucket per condition).

The stat operates at a single ``timeframe`` granularity (the bucket size, e.g.
``15min``). The RTH session is partitioned into fixed-width buckets starting at
the session open; a bar's bucket is ``(minute_of_day - rth_start) // bucket_min``.
The final RTH bucket may be partial when the session length is not a whole
multiple of the bucket size (e.g. NQ 09:30–16:15 with 30-minute buckets ends
with a 15-minute 16:00 bucket).

Two conditions are reported:
  - ``intraday_high``: the bucket containing the RTH session high.
  - ``intraday_low``:  the bucket containing the RTH session low.

Only resolved days (clean session open + sufficient close coverage) participate;
early-close days and the final incomplete day are excluded as pending samples by
``build_resolved_days``.

A custom ``SessionColor`` slicer (subclass of ``_ColorSlicer``) splits days by
whether the RTH session candle is green (close >= open) or red — the issue's
``session_color`` parameter, expressed as a slice so a single run yields the
overall (``all``) result plus the green and red breakdowns. The shared
``Weekday`` slicer provides the "by weekday" variant.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from stats.base import (
  BaseStat,
  I18nString,
  Labels,
  StatResultRow,
  Weekday,
  _ColorSlicer,
  write_results,
)
from stats.config import InstrumentConfig, load_config, minute_of_day
from stats.utils.daily_candles import build_resolved_days

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="Intraday Timing",
  fr="Timing intraday",
)
_DEFINITION = I18nString(
  en="For each RTH session, which intraday time bucket produced the day's high and which produced the day's low? Aggregated as the fraction of days each bucket claims each extreme.",
  fr="Pour chaque séance RTH, quelle tranche horaire intraday a produit le plus haut et le plus bas du jour ? Agrégé sous forme de fraction des jours où chaque tranche revendique chaque extrême.",
)

# Condition keys and their i18n labels, in display order.
_CONDITIONS: list[tuple[str, I18nString]] = [
  ("intraday_high", I18nString(en="Intraday high", fr="Plus haut intraday")),
  ("intraday_low", I18nString(en="Intraday low", fr="Plus bas intraday")),
]

# Maps each condition key to the day_table column that carries its bucket int.
_CONDITION_COLUMN: dict[str, str] = {
  "intraday_high": "high_bucket",
  "intraday_low": "low_bucket",
}

# Supported bucket granularities, in minutes.
_TIMEFRAME_MINUTES: dict[str, int] = {
  "1min": 1,
  "5min": 5,
  "15min": 15,
  "30min": 30,
  "1h": 60,
}


# ---------------------------------------------------------------------------
# Custom slicer
# ---------------------------------------------------------------------------
class SessionColor(_ColorSlicer):
  """Split days by the RTH session candle color (green = close >= open)."""

  def __init__(self) -> None:
    super().__init__(column="session_green", name="session_color")

  def dimension_label(self) -> I18nString:
    return I18nString(en="Session color", fr="Couleur de séance")


# ---------------------------------------------------------------------------
# Stat implementation
# ---------------------------------------------------------------------------
class IntradayTiming(BaseStat):
  """Fraction of days whose high/low fell in each intraday time bucket.

  One instance covers a single bucket granularity (``timeframe``). The
  ``SessionColor`` and ``Weekday`` slicers re-run the core computation over the
  green/red and per-weekday subsets respectively.
  """

  stat_name = "intraday_timing"
  title = _TITLE
  definition = _DEFINITION
  slices = (Weekday(), SessionColor())

  def __init__(
    self,
    instrument: str,
    config: InstrumentConfig,
    timeframe: str = "15min",
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

    # Build the full bucket grid for this session + granularity. The last bucket
    # may be partial when the session length is not a whole multiple of bucket_min.
    span = self.rth_end_min - self.rth_start_min
    self.n_buckets: int = math.ceil(span / self.bucket_min)
    self._bucket_order: list[tuple[int, str]] = []
    outcomes: dict[str, I18nString] = {}
    for i in range(self.n_buckets):
      start = self.rth_start_min + i * self.bucket_min
      hh, mm = divmod(start, 60)
      key = f"{hh:02d}{mm:02d}"
      self._bucket_order.append((i, key))
      label = f"{hh:02d}:{mm:02d}"
      outcomes[key] = I18nString(en=label, fr=label)

    # Outcome labels are timeframe-dependent, so build the Labels per instance.
    self.labels = Labels(
      conditions={key: label for key, label in _CONDITIONS},
      outcomes=outcomes,
    )

  # -------------------------------------------------------------------------
  # Day table construction
  # -------------------------------------------------------------------------
  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-day table from raw 1-min OHLCV data.

    Steps:
    1. Build the resolved-days table (session_open, session_close) via
       ``build_resolved_days`` — only clean, fully-closed sessions pass.
    2. Over the same RTH bar filter, find for each day the minute of the bar
       that printed the session high (first occurrence on ties) and the minute
       of the bar that printed the session low, then map each to its bucket
       index. Also collect the set of buckets that actually contain a bar that
       day (``present_buckets``) for the baseline draw.
    3. Join onto the resolved index (inner join) and add ``session_green``.

    Returns a DataFrame indexed by the normalized session date with columns:
    high_bucket, low_bucket, session_green, present_buckets.
    """
    columns = ["high_bucket", "low_bucket", "session_green", "present_buckets"]
    empty = pd.DataFrame(columns=columns)

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

    # Minute (and thus bucket) of the day's high/low. idxmax/idxmin return the
    # first occurrence on ties, giving a deterministic result.
    high_idx = rth.groupby("_date")["high"].idxmax()
    low_idx = rth.groupby("_date")["low"].idxmin()
    high_bucket = (
      rth.loc[high_idx].set_index("_date")["_bucket"].astype(int).rename("high_bucket")
    )
    low_bucket = (
      rth.loc[low_idx].set_index("_date")["_bucket"].astype(int).rename("low_bucket")
    )

    present_buckets = (
      rth.groupby("_date")["_bucket"]
      .apply(lambda s: tuple(sorted(int(b) for b in s.unique())))
      .rename("present_buckets")
    )

    daily = (
      resolved.join(high_bucket, how="inner")
      .join(low_bucket, how="inner")
      .join(present_buckets, how="inner")
    )
    if daily.empty:
      return empty

    daily["session_green"] = daily["session_close"] >= daily["session_open"]

    return daily[columns]

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute condition/outcome rows over a (possibly sliced) day table.

    For each condition and each bucket present in the subset, counts how many
    days had their extreme in that bucket, divided by the number of days in the
    subset. Probabilities for a given condition sum to 1.0 across present
    buckets. ``value`` / ``value_baseline`` are left None (probability-only).
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    total = len(day_table)
    if total == 0:
      return []

    # Union of buckets present across all days in the subset, in grid order.
    all_present: set[int] = set()
    for pb in day_table["present_buckets"]:
      all_present.update(pb)
    present_ordered = [(i, key) for i, key in self._bucket_order if i in all_present]

    rows: list[StatResultRow] = []
    for cond_key, _ in _CONDITIONS:
      col = _CONDITION_COLUMN[cond_key]
      counts = day_table[col].value_counts()
      for bucket_idx, bucket_key in present_ordered:
        count = int(counts.get(bucket_idx, 0))
        probability = count / total
        bl = baseline_map.get((cond_key, bucket_key))
        rows.append(
          StatResultRow(
            condition=cond_key,
            outcome=bucket_key,
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
    """Random null baseline: each day's extreme bucket is drawn uniformly.

    For each day, two independent uniform draws are made from that day's
    ``present_buckets`` — one for the high, one for the low. This is the null
    hypothesis that extremes fall uniformly across the session's buckets. Uses
    ``np.random.default_rng`` with a fixed seed for deterministic output.
    """
    n = len(day_table)
    if n == 0:
      return self.compute_rows(day_table, baseline_rows=None)

    rng = np.random.default_rng(seed)
    tmp = day_table.copy()
    condition_cols = [_CONDITION_COLUMN[cond_key] for cond_key, _ in _CONDITIONS]

    random_buckets: dict[str, list[int]] = {col: [] for col in condition_cols}
    for pb in tmp["present_buckets"]:
      pb_list = list(pb)
      for col in condition_cols:
        random_buckets[col].append(int(rng.choice(pb_list)))

    for col in condition_cols:
      tmp[col] = random_buckets[col]

    return self.compute_rows(tmp, baseline_rows=None)


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  timeframe: str = "15min",
  config_dir: str = "config",
  data_path: str | None = None,
) -> Path:
  """Compute Intraday Timing for one bucket granularity. Writes JSON to results/."""
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  stat = IntradayTiming(instrument=instrument, config=config, timeframe=timeframe)
  result = stat.compute(candles_df)

  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Intraday Timing stat")
  parser.add_argument("--instrument", default="NQ", help="Instrument name (default: NQ)")
  parser.add_argument(
    "--timeframe",
    default="15min",
    choices=list(_TIMEFRAME_MINUTES),
    help="Bucket granularity (default: 15min)",
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
    for cond, _ in _CONDITIONS:
      cond_rows = [r for r in tf_data["results"] if r["condition"] == cond]
      if not cond_rows:
        continue
      peak = max(cond_rows, key=lambda r: r["count"])
      print(
        f"    {cond}: peak bucket={peak['outcome']} "
        f"P={peak['probability']:.3f} (count={peak['count']}, N={peak['total']})"
      )
