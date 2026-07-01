"""Intraday Volume & Range stat.

For each intraday time bucket of the RTH session: the average traded volume, the
average price range (high - low, in points), and the average percentage range
((high - low) / bucket_open). Shows when volume and volatility build or fade
through the session.

The stat operates at a single ``timeframe`` granularity (the bucket size, e.g.
``15min``). The RTH session is partitioned into fixed-width buckets starting at
the session open; a bar's bucket is ``(minute_of_day - rth_start) // bucket_min``.
The final RTH bucket may be partial when the session length is not a whole
multiple of the bucket size (e.g. NQ 09:30-16:15 with 30-minute buckets ends with
a 15-minute 16:00 bucket).

This is a **magnitude** stat: every outcome carries its metric in
``StatResultRow.value`` (and its random baseline in ``value_baseline``); the
ordinary ``probability`` channel is left at ``0.0`` for every row.

Each intraday bucket is a **condition** (its key is ``HHMM``, e.g. ``0930``). The
three metrics are the **outcomes**:
  - ``mean_volume``    — average summed volume of the bucket's bars.
  - ``mean_range``     — average ``bucket_high - bucket_low`` in points.
  - ``mean_range_pct`` — average ``(bucket_high - bucket_low) / bucket_open``
                         (a decimal, e.g. ``0.004`` = 0.4%), where ``bucket_open``
                         is the open of the first bar in the bucket that day.

Only resolved days (clean session open + sufficient close coverage) participate;
early-close days and the final incomplete day are excluded as pending samples by
``build_resolved_days``. The shared ``Weekday`` slicer provides the "by weekday"
variant.
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
  en="Intraday Volume & Range",
  fr="Volume & amplitude intraday",
)
_DEFINITION = I18nString(
  en="For each intraday time bucket: the average traded volume, the average price range, and the average percentage range. Shows when volume and volatility build or fade through the session.",
  fr="Pour chaque tranche horaire intraday : le volume échangé moyen, l'amplitude moyenne en points et l'amplitude moyenne en pourcentage. Montre quand le volume et la volatilité montent ou retombent au fil de la séance.",
)

# Outcome keys and the day-table column prefix carrying each metric, in display
# order. Each bucket b stores its metrics in columns f"{prefix}_{b}".
_OUTCOMES: list[tuple[str, str, I18nString]] = [
  ("mean_volume", "vol", I18nString(en="Average volume", fr="Volume moyen")),
  ("mean_range", "rng", I18nString(en="Average range (points)", fr="Amplitude moyenne (points)")),
  ("mean_range_pct", "pct", I18nString(en="Average range (%)", fr="Amplitude moyenne (%)")),
]

# Supported bucket granularities, in minutes.
_TIMEFRAME_MINUTES: dict[str, int] = {
  "1min": 1,
  "5min": 5,
  "15min": 15,
  "30min": 30,
  "1h": 60,
}


class IntradayVolumeRange(BaseStat):
  """Average volume, range, and percentage range per intraday time bucket.

  One instance covers a single bucket granularity (``timeframe``). The ``Weekday``
  slicer re-runs the core computation over each per-weekday subset.
  """

  stat_name = "intraday_volume_range"
  title = _TITLE
  definition = _DEFINITION
  slices = ("weekday",)

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
    conditions: dict[str, I18nString] = {}
    for i in range(self.n_buckets):
      start = self.rth_start_min + i * self.bucket_min
      hh, mm = divmod(start, 60)
      key = f"{hh:02d}{mm:02d}"
      self._bucket_order.append((i, key))
      label = f"{hh:02d}:{mm:02d}"
      conditions[key] = I18nString(en=label, fr=label)

    # Bucket conditions are timeframe-dependent, so build the Labels per instance.
    self.labels = Labels(
      conditions=conditions,
      outcomes={out_key: label for out_key, _, label in _OUTCOMES},
    )

  # -------------------------------------------------------------------------
  # Day table construction
  # -------------------------------------------------------------------------
  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-day wide table of per-bucket metrics from 1-min OHLCV data.

    Each row is one resolved trading day, indexed by the normalized session date,
    with three columns per bucket ``b``: ``vol_{b}`` (summed volume), ``rng_{b}``
    (high - low, points), ``pct_{b}`` ((high - low) / bucket_open). A bucket with
    no bars on a given day is NaN and excluded from that bucket's averages.

    Steps:
    1. Build the resolved-days table via ``build_resolved_days`` (only clean,
       fully-closed sessions pass).
    2. Over the same RTH bar filter, assign each bar a bucket index, then per
       (day, bucket) compute summed volume, the high/low extremes, and the open
       of the bucket's first bar. Derive range and range_pct.
    3. Pivot to one row per day with the per-bucket columns, reindexed onto the
       resolved-days index (only resolved dates pass through).
    """
    columns = self._all_columns()
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

    grouped = rth.groupby(["_date", "_bucket"])
    agg = grouped.agg(
      _vol=("volume", "sum"),
      _high=("high", "max"),
      _low=("low", "min"),
    )
    # Bucket open = open of the earliest bar in the bucket (first by minute).
    bucket_open = (
      rth.sort_values("_mod").groupby(["_date", "_bucket"])["open"].first()
    )
    agg["_open"] = bucket_open
    agg["_rng"] = agg["_high"] - agg["_low"]
    agg["_pct"] = agg["_rng"] / agg["_open"]

    # Pivot each metric to a day x bucket wide frame, then assemble the per-day
    # table on the resolved-days index (alignment drops non-resolved dates and
    # fills absent buckets with NaN).
    vol_w = agg["_vol"].unstack("_bucket")
    rng_w = agg["_rng"].unstack("_bucket")
    pct_w = agg["_pct"].unstack("_bucket")

    out = pd.DataFrame(index=resolved.index)
    for i, _ in self._bucket_order:
      out[f"vol_{i}"] = vol_w[i] if i in vol_w.columns else np.nan
      out[f"rng_{i}"] = rng_w[i] if i in rng_w.columns else np.nan
      out[f"pct_{i}"] = pct_w[i] if i in pct_w.columns else np.nan

    return out[columns]

  def _all_columns(self) -> list[str]:
    """The wide table's column names, in (bucket, metric) display order."""
    return [
      f"{prefix}_{i}"
      for i, _ in self._bucket_order
      for _, prefix, _ in _OUTCOMES
    ]

  def _present_buckets(self, day_table: pd.DataFrame) -> list[tuple[int, str]]:
    """Buckets with at least one observation in this (possibly sliced) subset."""
    present: list[tuple[int, str]] = []
    for i, key in self._bucket_order:
      col = f"vol_{i}"
      if col in day_table.columns and bool(day_table[col].notna().any()):
        present.append((i, key))
    return present

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the three magnitude rows per present bucket over a day subset.

    For each bucket present in the subset, each metric's row carries its average
    (over the days that had that bucket) in ``value``; the ``probability`` channel
    is left at ``0.0``. ``count`` / ``total`` report the number of contributing
    days for that bucket.

    If ``baseline_rows`` is provided, the matching row's ``value`` / ``total`` are
    merged into each row's ``value_baseline`` / ``baseline_n``.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    rows: list[StatResultRow] = []
    for i, bucket_key in self._present_buckets(day_table):
      for out_key, prefix, _ in _OUTCOMES:
        series = day_table[f"{prefix}_{i}"].astype(float)
        n = int(series.notna().sum())
        mean_value = float(series.mean()) if n > 0 else 0.0
        bl = baseline_map.get((bucket_key, out_key))
        rows.append(
          StatResultRow(
            condition=bucket_key,
            outcome=out_key,
            count=n,
            total=n,
            probability=0.0,
            baseline_prob=0.0,
            baseline_n=bl.total if bl else 0,
            value=mean_value,
            value_baseline=bl.value if bl else None,
          )
        )
    return rows

  # -------------------------------------------------------------------------
  # Baseline
  # -------------------------------------------------------------------------
  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random null: each bucket's metric resampled from all buckets pooled.

    For each metric, all per-(day, bucket) observations in the subset are pooled.
    For a bucket with ``n`` contributing days, ``n`` observations are drawn
    uniformly at random — without replacement — from that pool, and their mean is
    the bucket's baseline. This is the null hypothesis that the intraday time
    bucket carries no information: its expected metric equals the grand mean across
    all buckets. Uses ``np.random.default_rng`` with a fixed seed for deterministic
    output.
    """
    present = self._present_buckets(day_table)
    if not present:
      return []

    rng = np.random.default_rng(seed)

    # Pool of all observations per metric (across every present bucket and day).
    pools: dict[str, np.ndarray] = {}
    counts: dict[tuple[int, str], int] = {}
    for out_key, prefix, _ in _OUTCOMES:
      arrays = []
      for i, _ in present:
        values = day_table[f"{prefix}_{i}"].dropna().to_numpy(dtype=float)
        counts[(i, out_key)] = len(values)
        arrays.append(values)
      pools[out_key] = np.concatenate(arrays) if arrays else np.empty(0)

    rows: list[StatResultRow] = []
    for i, bucket_key in present:
      for out_key, _, _ in _OUTCOMES:
        pool = pools[out_key]
        n = counts[(i, out_key)]
        # The pool contains this bucket's own observations, so n <= len(pool) and
        # sampling without replacement is well-defined.
        sample_size = min(n, len(pool))
        if sample_size > 0:
          idx = rng.choice(len(pool), size=sample_size, replace=False)
          mean_value = float(pool[idx].mean())
        else:
          mean_value = 0.0
        rows.append(
          StatResultRow(
            condition=bucket_key,
            outcome=out_key,
            count=sample_size,
            total=sample_size,
            probability=0.0,
            baseline_prob=0.0,
            baseline_n=0,
            value=mean_value,
            value_baseline=None,
          )
        )
    return rows

  def classify_samples(self, day_table: pd.DataFrame) -> list[SampleRow]:
    """One SampleRow per (day, bucket, outcome) with a non-NaN metric that day.

    Mirrors ``compute_rows``'s classification exactly: only buckets present in
    the day table (``_present_buckets``) are considered, and within each bucket
    only the days whose metric is not NaN contribute — the same days
    ``compute_rows`` counts in ``n`` and averages into ``value``.
    """
    if day_table.empty:
      return []

    samples: list[SampleRow] = []
    for i, bucket_key in self._present_buckets(day_table):
      for out_key, prefix, _ in _OUTCOMES:
        series = day_table[f"{prefix}_{i}"]
        for ts, val in series.items():
          if pd.notna(val):
            samples.append(
              SampleRow(
                date=ts.strftime("%Y-%m-%d"),
                condition=bucket_key,
                outcome=out_key,
                value=float(val),
              )
            )
    return samples


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  timeframe: str = "15min",
  config_dir: str = "config",
  data_path: str | None = None,
) -> Path:
  """Compute Intraday Volume & Range for one bucket granularity. Writes JSON."""
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  stat = IntradayVolumeRange(instrument=instrument, config=config, timeframe=timeframe)
  result = stat.compute(candles_df)

  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Intraday Volume & Range stat")
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
    by_bucket: dict[str, dict[str, float]] = {}
    for r in tf_data["results"]:
      by_bucket.setdefault(r["condition"], {})[r["outcome"]] = r["value"]
    for bucket_key, metrics in by_bucket.items():
      vol = metrics.get("mean_volume", 0.0)
      rng_pts = metrics.get("mean_range", 0.0)
      rng_pct = metrics.get("mean_range_pct", 0.0)
      print(
        f"    {bucket_key}: avg volume={vol:,.0f} "
        f"| avg range={rng_pts:.2f} pts ({rng_pct:.4%})"
      )
