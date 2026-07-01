"""Candle Body Ratio stat.

For each intraday time bucket of the RTH session: the fraction of resolved days
whose bucket candle has a **body** (``|close - open|``) covering at least
``candle_size`` percent of its full **range** (``high - low``). A high body ratio
marks a decisive, trend-like candle; a low ratio marks an indecisive, wick-heavy
one. Bucketed by time slot, the stat shows when in the session decisive candles
are most (and least) likely to print.

The stat operates at a single ``timeframe`` granularity (the bucket size, e.g.
``15min``). The RTH session is partitioned into fixed-width buckets starting at
the session open; a bar's bucket is ``(minute_of_day - rth_start) // bucket_min``.
The final RTH bucket may be partial when the session length is not a whole
multiple of the bucket size (e.g. NQ 09:30-16:15 with 30-minute buckets ends with
a 15-minute 16:00 bucket).

Each bucket candle is built from the 1-min bars in that ``(day, bucket)``:
``open`` is the first bar's open, ``close`` the last bar's close, ``high`` /
``low`` the bucket's extremes. Its body ratio is ``|close - open| / (high - low)``
(``0.0`` for a flat candle with zero range); the candle counts as a *large body*
when that ratio is at least ``candle_size / 100``.

This is a **probability** stat. Each intraday bucket is a **condition** (its key
is ``HHMM``, e.g. ``0930``); the single **outcome** ``large_body`` carries the
fraction of that bucket's days whose candle had a large body. ``count`` / ``total``
report the number of large-body days and the number of contributing days for that
bucket.

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
  en="Candle Body Ratio",
  fr="Ratio de corps de bougie",
)
_DEFINITION = I18nString(
  en="For each intraday time bucket: the fraction of days whose candle body covers at least the configured percentage of its high-to-low range. Shows when in the session decisive, trend-like candles are most likely to print.",
  fr="Pour chaque tranche horaire intraday : la fraction des jours dont le corps de la bougie couvre au moins le pourcentage configuré de son amplitude haut-bas. Montre à quel moment de la séance les bougies décisives, de tendance, sont les plus probables.",
)

# The single outcome key. Its label embeds the configured threshold, so it is
# built per instance.
_OUTCOME_KEY = "large_body"

# Supported bucket granularities, in minutes.
_TIMEFRAME_MINUTES: dict[str, int] = {
  "1min": 1,
  "5min": 5,
  "15min": 15,
  "30min": 30,
  "1h": 60,
}


class CandleBodyRatio(BaseStat):
  """Fraction of days whose bucket candle has a large body, per intraday bucket.

  One instance covers a single bucket granularity (``timeframe``) and a single
  ``candle_size`` threshold. The ``Weekday`` slicer re-runs the core computation
  over each per-weekday subset.
  """

  stat_name = "candle_body_ratio"
  title = _TITLE
  definition = _DEFINITION
  slices = ("weekday",)

  def __init__(
    self,
    instrument: str,
    config: InstrumentConfig,
    timeframe: str = "15min",
    candle_size: float = 50.0,
    close_tolerance_min: int = 15,
  ) -> None:
    if timeframe not in _TIMEFRAME_MINUTES:
      raise ValueError(
        f"Unknown timeframe '{timeframe}'. Choose from {list(_TIMEFRAME_MINUTES)}."
      )
    if not 0.0 <= candle_size <= 100.0:
      raise ValueError(f"candle_size must be in [0, 100], got {candle_size}.")
    self.instrument = instrument
    self.timeframe = timeframe
    self.config = config
    self.bucket_min: int = _TIMEFRAME_MINUTES[timeframe]
    self.candle_size = float(candle_size)
    self.threshold = candle_size / 100.0
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

    pct_label = f"{self.candle_size:g}"
    outcome_label = I18nString(
      en=f"Body ≥ {pct_label}% of range",
      fr=f"Corps ≥ {pct_label}% de l'amplitude",
    )

    # Bucket conditions are timeframe-dependent and the outcome label is
    # threshold-dependent, so build the Labels per instance.
    self.labels = Labels(
      conditions=conditions,
      outcomes={_OUTCOME_KEY: outcome_label},
    )

  # -------------------------------------------------------------------------
  # Day table construction
  # -------------------------------------------------------------------------
  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-day wide table of per-bucket large-body flags from 1-min data.

    Each row is one resolved trading day, indexed by the normalized session date,
    with one column per bucket ``b``: ``body_{b}`` is ``1.0`` when that day's
    bucket candle had a large body (ratio ``>= threshold``), ``0.0`` when it did
    not, and ``NaN`` when the bucket had no bars that day (excluded from the
    bucket's denominator).

    Steps:
    1. Build the resolved-days table via ``build_resolved_days`` (only clean,
       fully-closed sessions pass).
    2. Over the same RTH bar filter, assign each bar a bucket index, then per
       (day, bucket) build the candle: open of the first bar, close of the last
       bar, max high, min low. Derive the body ratio and the large-body flag.
    3. Pivot to one row per day with the per-bucket ``body_{b}`` columns,
       reindexed onto the resolved-days index (only resolved dates pass through).
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
      _high=("high", "max"),
      _low=("low", "min"),
    )
    # Candle open / close = open of the bucket's first bar / close of its last bar.
    sorted_rth = rth.sort_values("_mod")
    agg["_open"] = sorted_rth.groupby(["_date", "_bucket"])["open"].first()
    agg["_close"] = sorted_rth.groupby(["_date", "_bucket"])["close"].last()

    rng = (agg["_high"] - agg["_low"]).to_numpy(dtype=float)
    body = (agg["_close"] - agg["_open"]).abs().to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
      ratio = np.where(rng > 0, body / rng, 0.0)
    agg["_large"] = (ratio >= self.threshold).astype(float)

    # Pivot the flag to a day x bucket wide frame, then assemble the per-day table
    # on the resolved-days index (alignment drops non-resolved dates and fills
    # absent buckets with NaN).
    large_w = agg["_large"].unstack("_bucket")

    out = pd.DataFrame(index=resolved.index)
    for i, _ in self._bucket_order:
      out[f"body_{i}"] = large_w[i] if i in large_w.columns else np.nan

    return out[columns]

  def _all_columns(self) -> list[str]:
    """The wide table's column names, in bucket display order."""
    return [f"body_{i}" for i, _ in self._bucket_order]

  def _present_buckets(self, day_table: pd.DataFrame) -> list[tuple[int, str]]:
    """Buckets with at least one observation in this (possibly sliced) subset."""
    present: list[tuple[int, str]] = []
    for i, key in self._bucket_order:
      col = f"body_{i}"
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
    """Compute the large-body probability per present bucket over a day subset.

    For each bucket present in the subset, ``probability`` is the fraction of
    contributing days (days that had bars in that bucket) whose candle had a
    large body. ``count`` is the number of large-body days; ``total`` is the
    number of contributing days. ``value`` / ``value_baseline`` are left None
    (probability-only).

    If ``baseline_rows`` is provided, each row's ``baseline_prob`` / ``baseline_n``
    is merged from the matching baseline row.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    rows: list[StatResultRow] = []
    for i, bucket_key in self._present_buckets(day_table):
      series = day_table[f"body_{i}"].dropna()
      total = int(len(series))
      count = int(series.sum())
      probability = count / total if total > 0 else 0.0
      bl = baseline_map.get((bucket_key, _OUTCOME_KEY))
      rows.append(
        StatResultRow(
          condition=bucket_key,
          outcome=_OUTCOME_KEY,
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

  def classify_samples(self, day_table: pd.DataFrame) -> list[SampleRow]:
    """One SampleRow per (day, present bucket) contributing observation.

    Mirrors ``compute_rows``: for every present bucket, every contributing day
    (non-NaN ``body_{i}``) yields a sample whose outcome is ``large_body`` or
    ``not_large_body``. The "large_body" samples reproduce each row's ``count``;
    the total samples per bucket reproduce its ``total``.
    """
    if day_table.empty:
      return []

    samples: list[SampleRow] = []
    for i, bucket_key in self._present_buckets(day_table):
      series = day_table[f"body_{i}"]
      for ts, flag in series.items():
        if pd.isna(flag):
          continue
        outcome = "large_body" if bool(flag) else "not_large_body"
        samples.append(
          SampleRow(date=ts.strftime("%Y-%m-%d"), condition=bucket_key, outcome=outcome)
        )
    return samples

  # -------------------------------------------------------------------------
  # Baseline
  # -------------------------------------------------------------------------
  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random null: each bucket's large-body rate resampled from all buckets pooled.

    All per-(day, bucket) large-body flags in the subset are pooled. For a bucket
    with ``n`` contributing days, ``n`` flags are drawn uniformly at random —
    without replacement — from that pool, and their mean is the bucket's baseline
    probability. This is the null hypothesis that the intraday time bucket carries
    no information: its expected large-body rate equals the grand mean across all
    buckets. Uses ``np.random.default_rng`` with a fixed seed for deterministic
    output.
    """
    present = self._present_buckets(day_table)
    if not present:
      return []

    rng = np.random.default_rng(seed)

    # Pool of all large-body flags across every present bucket and day.
    arrays = []
    counts: dict[int, int] = {}
    for i, _ in present:
      values = day_table[f"body_{i}"].dropna().to_numpy(dtype=float)
      counts[i] = len(values)
      arrays.append(values)
    pool = np.concatenate(arrays) if arrays else np.empty(0)

    rows: list[StatResultRow] = []
    for i, bucket_key in present:
      n = counts[i]
      # The pool contains this bucket's own observations, so n <= len(pool) and
      # sampling without replacement is well-defined.
      sample_size = min(n, len(pool))
      if sample_size > 0:
        idx = rng.choice(len(pool), size=sample_size, replace=False)
        probability = float(pool[idx].mean())
      else:
        probability = 0.0
      rows.append(
        StatResultRow(
          condition=bucket_key,
          outcome=_OUTCOME_KEY,
          count=int(round(probability * sample_size)),
          total=sample_size,
          probability=probability,
          baseline_prob=0.0,
          baseline_n=0,
          value=None,
          value_baseline=None,
        )
      )
    return rows


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  timeframe: str = "15min",
  candle_size: float = 50.0,
  config_dir: str = "config",
  data_path: str | None = None,
) -> Path:
  """Compute Candle Body Ratio for one bucket granularity. Writes JSON."""
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  stat = CandleBodyRatio(
    instrument=instrument,
    config=config,
    timeframe=timeframe,
    candle_size=candle_size,
  )
  result = stat.compute(candles_df)

  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Candle Body Ratio stat")
  parser.add_argument("--instrument", default="NQ", help="Instrument name (default: NQ)")
  parser.add_argument(
    "--timeframe",
    default="15min",
    choices=list(_TIMEFRAME_MINUTES),
    help="Bucket granularity (default: 15min)",
  )
  parser.add_argument(
    "--candle-size",
    type=float,
    default=50.0,
    help="Minimum body-to-range ratio percentage (default: 50)",
  )
  parser.add_argument("--data-path", default=None, help="Override parquet file path")
  parser.add_argument("--config-dir", default="config", help="Config directory (default: config)")
  args = parser.parse_args()

  output_path = run(
    instrument=args.instrument,
    timeframe=args.timeframe,
    candle_size=args.candle_size,
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
      print(
        f"    {r['condition']}: P(large body)={r['probability']:.3f} "
        f"(count={r['count']}, N={r['total']}) "
        f"baseline={r['baseline_prob']:.3f}"
      )
