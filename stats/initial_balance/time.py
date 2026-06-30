"""Initial Balance Breakout — by Time stat.

Measures *when* the first breakout of the initial balance (IB) happens within the
RTH session: the distribution of the **first-breakout time** across the session,
summarised both as a coarse early/late split around a configurable threshold and
as a fine-grained time histogram.

Definitions
-----------
- **Initial balance**: the high-low range of the bars in the IB window
  ``[rth_start, rth_start + ib_period)``. ``ib_high`` is the max high and
  ``ib_low`` the min low over that window.
- **Breakout window**: the rest of the session ``[rth_start + ib_period,
  rth_end)``. The IB itself is excluded so the range can never break its own bars.
- **First breakout**: the *chronologically first* bar in the breakout window that
  breaks EITHER side of the IB, irrespective of direction. With
  ``breakout_criteria = "wick"`` (default) a break uses the intraday extreme
  (``high > ib_high`` or ``low < ib_low``); with ``"close"`` it requires a bar to
  CLOSE beyond a level (``close > ib_high`` or ``close < ib_low``). Strict
  inequality — touching a level exactly is NOT a break, mirroring the sibling
  ``initial_balance`` / ``opening_range_breakout`` convention.
- **First-breakout time**: the minute-of-day of that first-breakout bar
  (``first_break_mod``). The "early vs late" split compares it to the
  ``early_threshold_min``-th minute from the session open (default 150 → 12:00 ET).

Sessions where price never breaks either side during the breakout window are
excluded from every denominator (pending-sample discipline).

Outcomes
--------
Two outcome partitions are reported over the same countable denominator (every
session that broke at least once), each summing to that denominator:

- Condition ``timing`` — coarse split: ``early`` (first break before the
  threshold) vs ``late`` (at/after it).
- Condition ``bucket`` — the breakout-time histogram: fixed ``bin_minutes``-wide
  buckets spanning ``[rth_start, rth_end)`` (default 30 min → 14 buckets for a
  09:30–16:15 NQ session). Buckets entirely inside the IB window are unreachable
  and always read zero; they are kept so the outcome set is identical across
  timeframes. The threshold is a bucket boundary at the default 30-min width, so
  ``early`` / ``late`` are exactly the sums of the buckets on each side.

Results are computed for two IB lengths (30 min and 1 h) emitted as separate
timeframe entries, mirroring the ``initial_balance`` standard stat.

Declared slices
---------------
- ``weekday``     — the "by weekday" breakdown.
- ``close``       — split by session close color (green / red).
- ``prev_candle`` — split by the prior session's color.
- ``overnight``   — split by overnight gap direction.
- ``size``        — quartile buckets of the IB size (absolute points).
- ``size_pct``    — preset buckets of the IB size as a percent of price.
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
from stats.initial_balance.common import build_ib_day_base

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="Initial Balance Breakout — by Time",
  fr="Cassure de l'initial balance — par heure",
)
_DEFINITION = I18nString(
  en=(
    "After the initial balance forms, WHEN does price first break above its high "
    "or below its low? Reported as the share of sessions breaking early vs late "
    "around a configurable threshold, plus the full first-breakout-time histogram."
  ),
  fr=(
    "Après la formation de l'initial balance, QUAND le prix casse-t-il pour la "
    "première fois au-dessus de son haut ou en-dessous de son bas ? Exprimé comme "
    "la proportion des séances cassant tôt ou tard autour d'un seuil configurable, "
    "ainsi que l'histogramme complet de l'heure de première cassure."
  ),
)

# Two condition keys, each holding an outcome partition over the same denominator.
_CONDITION_TIMING = "timing"
_CONDITION_BUCKET = "bucket"

_LABELS = Labels(
  conditions={
    _CONDITION_TIMING: I18nString(en="Breakout timing", fr="Moment de la cassure"),
    _CONDITION_BUCKET: I18nString(en="Breakout-time bucket", fr="Tranche horaire de cassure"),
  },
  outcomes={
    "early": I18nString(
      en="First breakout before the threshold",
      fr="Première cassure avant le seuil",
    ),
    "late": I18nString(
      en="First breakout at or after the threshold",
      fr="Première cassure au seuil ou après",
    ),
    # Bucket outcome labels (bucket_0, bucket_1, …) are clock-range dependent and
    # are injected at run() time from the instrument's session times.
  },
)

# Timeframe string → IB window length in minutes (mirrors standard.py exactly).
_TF_MINUTES: dict[str, int] = {
  "30min": 30,
  "1h": 60,
}

_BREAKOUT_CRITERIA: tuple[str, ...] = ("wick", "close")


def _fmt_clock(mod: int) -> str:
  """Format a minute-of-day as a ``HH:MM`` clock string."""
  h, m = divmod(mod, 60)
  return f"{h:02d}:{m:02d}"


def _make_buckets(
  rth_start_min: int, rth_end_min: int, bin_minutes: int
) -> list[tuple[str, int, int]]:
  """Build the histogram buckets as ``(key, lo_mod, hi_mod)`` over the session.

  Buckets are ``bin_minutes``-wide, aligned to the session open, spanning
  ``[rth_start, rth_end)``. The final bucket is clipped at ``rth_end`` so a
  session that does not divide evenly still ends cleanly. Edges are absolute
  minute-of-day so they match ``first_break_mod`` directly.
  """
  buckets: list[tuple[str, int, int]] = []
  idx = 0
  lo = rth_start_min
  while lo < rth_end_min:
    hi = min(lo + bin_minutes, rth_end_min)
    buckets.append((f"bucket_{idx}", lo, hi))
    lo = hi
    idx += 1
  return buckets


def _bucket_outcome_labels(buckets: list[tuple[str, int, int]]) -> dict[str, I18nString]:
  """Clock-range i18n labels for each histogram bucket (e.g. ``09:30–10:00``)."""
  labels: dict[str, I18nString] = {}
  for key, lo, hi in buckets:
    span = f"{_fmt_clock(lo)}–{_fmt_clock(hi)}"
    labels[key] = I18nString(en=span, fr=span)
  return labels


class InitialBalanceTime(BaseStat):
  """Distribution of the first-breakout time of the initial balance."""

  stat_name = "initial_balance_time"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = (
    "weekday",
    "close",
    "prev_candle",
    "overnight",
    SizeBucket(column="ib_size", preset="quartiles", name="size"),
    # 0.6–0.9%, >0.9%); upper edge is open (inf).
    SizeBucket(
      column="ib_size_pct",
      buckets=[0.0, 0.2, 0.4, 0.6, 0.9, float("inf")],
      name="size_pct",
    ),
  )

  def __init__(
    self,
    instrument: str,
    timeframe: str,
    config: InstrumentConfig,
    breakout_criteria: str = "wick",
    early_threshold_min: int = 150,
    bin_minutes: int = 30,
    close_tolerance_min: int = 15,
  ) -> None:
    if timeframe not in _TF_MINUTES:
      raise ValueError(
        f"Unsupported timeframe '{timeframe}'. Choose from {list(_TF_MINUTES)}"
      )
    if breakout_criteria not in _BREAKOUT_CRITERIA:
      raise ValueError(
        f"Unsupported breakout_criteria '{breakout_criteria}'. "
        f"Choose from {list(_BREAKOUT_CRITERIA)}"
      )
    if early_threshold_min <= 0:
      raise ValueError(f"early_threshold_min must be positive, got {early_threshold_min}")
    if bin_minutes <= 0:
      raise ValueError(f"bin_minutes must be positive, got {bin_minutes}")
    self.instrument = instrument
    self.timeframe = timeframe
    self.config = config
    self.ib_minutes = _TF_MINUTES[timeframe]
    self.breakout_criteria = breakout_criteria
    self.early_threshold_min = early_threshold_min
    self.bin_minutes = bin_minutes
    self.close_tolerance_min = close_tolerance_min

    rth = config.sessions["rth"]
    self.rth_start_min: int = minute_of_day(rth.start)
    self.rth_end_min: int = minute_of_day(rth.end)
    # Initial-balance window: [rth_start, ib_end). Breakout window: [ib_end, rth_end).
    self.ib_end_min: int = self.rth_start_min + self.ib_minutes
    # Absolute minute-of-day threshold separating early from late breakouts.
    self.threshold_mod: int = self.rth_start_min + self.early_threshold_min
    # Histogram buckets, aligned to the session open (absolute minute-of-day edges).
    self.buckets: list[tuple[str, int, int]] = _make_buckets(
      self.rth_start_min, self.rth_end_min, self.bin_minutes
    )

  # -------------------------------------------------------------------------
  # Day table construction
  # -------------------------------------------------------------------------
  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-session table with the IB extremes and first-breakout time.

    Columns returned:
      ``ib_high`` / ``ib_low``   — IB extremes from the IB window.
      ``ib_size``                — ``ib_high - ib_low`` (read by the ``size`` slicer).
      ``ib_size_pct``            — ``ib_size`` as % of the session open, in percent
                                   (e.g. 0.3 = 0.3%), read by the ``size_pct`` slicer.
      ``first_break_mod``        — minute-of-day of the chronologically first
                                   breakout-window bar that breaks either side of the
                                   IB; NaN where price never breaks (excluded).
      ``session_green``          — session color (``session_close >= session_open``),
                                   read by the ``close`` slicer.
      ``prev_session_green``     — prior session's color (NaN for the first resolved
                                   day), read by the ``prev_candle`` slicer.
      ``overnight_green``        — overnight gap direction: open above prior close
                                   (NaN for the first resolved day), read by the
                                   ``overnight`` slicer.

    The resolved-days core is shared via ``build_resolved_days``. A day is included
    only when it has both a clean IB window AND a non-empty breakout window (inner
    join). Days where price never breaks either side keep ``first_break_mod = NaN``
    and are excluded from every denominator. The DatetimeIndex (normalised session
    date) is required by the ``weekday`` slicer.
    """
    columns = [
      "ib_high",
      "ib_low",
      "ib_size",
      "ib_size_pct",
      "first_break_mod",
      "session_green",
      "prev_session_green",
      "overnight_green",
    ]
    empty = pd.DataFrame(columns=columns)

    if candles_df.empty:
      return empty

    base = build_ib_day_base(
      candles_df,
      self.rth_start_min,
      self.ib_end_min,
      self.rth_end_min,
      self.close_tolerance_min,
    )
    if base.day.empty:
      return empty

    day = base.day
    post = base.post

    # First-breakout time: the minute-of-day of the earliest breakout-window bar
    # that breaks either side of that day's IB (criteria-aware, strict inequality).
    day["first_break_mod"] = self._first_break_mod(post, day["ib_high"], day["ib_low"])

    return day[columns]

  def _first_break_mod(
    self, post: pd.DataFrame, ib_high: pd.Series, ib_low: pd.Series
  ) -> pd.Series:
    """Minute-of-day of the first IB break per session, aligned to ``ib_high.index``.

    ``post`` is the full set of breakout-window bars (all dates). ``ib_high`` /
    ``ib_low`` are indexed by the surviving session dates. A break uses strict
    inequality; the ``breakout_criteria`` selects wick (intraday extreme) or close.
    Sessions with no breaking bar are left as NaN (excluded downstream).
    """
    if post.empty:
      return pd.Series(np.nan, index=ib_high.index, name="first_break_mod")

    p = post[["_date", "_mod", "high", "low", "close"]].copy()
    p["_ib_high"] = p["_date"].map(ib_high)
    p["_ib_low"] = p["_date"].map(ib_low)
    # Only bars on surviving sessions (both IB extremes present) can break.
    p = p[p["_ib_high"].notna() & p["_ib_low"].notna()]
    if p.empty:
      return pd.Series(np.nan, index=ib_high.index, name="first_break_mod")

    if self.breakout_criteria == "close":
      broke = (p["close"] > p["_ib_high"]) | (p["close"] < p["_ib_low"])
    else:  # wick
      broke = (p["high"] > p["_ib_high"]) | (p["low"] < p["_ib_low"])

    broke_bars = p[broke]
    first = broke_bars.groupby("_date")["_mod"].min()
    return first.reindex(ib_high.index).rename("first_break_mod")

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the timing and histogram rows over the (sliced) day table.

    Countable sessions are those that broke at least once (``first_break_mod`` is
    not NaN). Both the ``timing`` (early/late) and ``bucket`` (histogram)
    partitions share that denominator, so each partition's counts sum to ``total``.
    If ``baseline_rows`` is provided, merges its ``probability`` / ``total`` into
    each row's ``baseline_prob`` / ``baseline_n``.
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
      mod = pd.Series([], dtype=float)
    else:
      mod = day_table["first_break_mod"]
    countable = mod.notna()
    total = int(countable.sum())

    rows: list[StatResultRow] = []
    # Timing partition: early (before threshold) vs late (at/after).
    early = int((countable & (mod < self.threshold_mod)).sum())
    rows.append(_make(_CONDITION_TIMING, "early", early, total))
    rows.append(_make(_CONDITION_TIMING, "late", total - early, total))

    # Histogram partition: one bucket per [lo, hi) minute-of-day band.
    for key, lo, hi in self.buckets:
      count = int((countable & (mod >= lo) & (mod < hi)).sum())
      rows.append(_make(_CONDITION_BUCKET, key, count, total))

    return rows

  # -------------------------------------------------------------------------
  # Baseline
  # -------------------------------------------------------------------------
  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random null for the first-breakout time. Deterministic for a fixed seed.

    The null hypothesis is that the first breakout is equally likely at any minute
    of the breakout window — its timing carries no information. Each countable
    session is assigned a uniform random minute-of-day in ``[ib_end, rth_end)``;
    non-breaking sessions keep their NaN so they stay excluded (pending-sample
    discipline). The same counting path then yields the early/late split and the
    histogram a purely uniform breakout time would produce. Comparing the actual
    distribution to this null reveals whether breakouts cluster earlier or later
    than chance. Uses ``np.random.default_rng(seed)``.
    """
    n = len(day_table)
    if n == 0:
      return self.compute_rows(day_table, baseline_rows=None)

    rng = np.random.default_rng(seed)
    # Uniform integer minute in [ib_end, rth_end) for every row.
    rand_mod = rng.integers(self.ib_end_min, self.rth_end_min, size=n).astype(float)

    # Restore NaN for sessions that never broke; they must remain excluded.
    no_break = day_table["first_break_mod"].isna().to_numpy()
    rand_mod[no_break] = np.nan

    tmp = day_table.copy()
    tmp["first_break_mod"] = rand_mod

    return self.compute_rows(tmp, baseline_rows=None)


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
  breakout_criteria: str = "wick",
  early_threshold_min: int = 150,
  bin_minutes: int = 30,
) -> Path:
  """Load data and compute IB-by-time for both initial-balance lengths.

  Merges the per-timeframe TimeframeResults into one StatRunResult and writes the
  consolidated JSON to results/. Slice dimension labels and the data-dependent
  histogram-bucket outcome labels are merged across timeframes so no timeframe's
  labels overwrite another's.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  timeframes = ["30min", "1h"]
  merged_tf: dict[str, TimeframeResult] = {}
  labels = _LABELS.model_copy(deep=True)

  for tf in timeframes:
    stat = InitialBalanceTime(
      instrument=instrument,
      timeframe=tf,
      config=config,
      breakout_criteria=breakout_criteria,
      early_threshold_min=early_threshold_min,
      bin_minutes=bin_minutes,
    )
    result = stat.compute(candles_df)
    merged_tf[tf] = result.instruments[instrument][tf]
    # Merge framework-enriched slice dimension labels and the histogram-bucket
    # outcome labels so each timeframe contributes without overwriting earlier ones.
    labels.dimensions.update(result.labels.dimensions)
    labels.outcomes.update(_bucket_outcome_labels(stat.buckets))

  final_result = StatRunResult(
    stat_name="initial_balance_time",
    title=_TITLE,
    definition=_DEFINITION,
    labels=labels,
    instruments={instrument: merged_tf},
  )

  return write_results(final_result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Compute Initial Balance Breakout — by Time stat"
  )
  parser.add_argument("--instrument", default="NQ", help="Instrument name (default: NQ)")
  parser.add_argument("--data-path", default=None, help="Override parquet file path")
  parser.add_argument("--config-dir", default="config", help="Config directory (default: config)")
  parser.add_argument(
    "--breakout-criteria",
    default="wick",
    choices=list(_BREAKOUT_CRITERIA),
    help="Break detection: 'wick' (intraday extreme) or 'close' (bar close) (default: wick)",
  )
  parser.add_argument(
    "--early-threshold-min",
    type=int,
    default=150,
    help="Minutes from session open separating early from late breakouts (default: 150)",
  )
  parser.add_argument(
    "--bin-minutes",
    type=int,
    default=30,
    help="Histogram bucket width in minutes (default: 30)",
  )
  args = parser.parse_args()

  output_path = run(
    instrument=args.instrument,
    config_dir=args.config_dir,
    data_path=args.data_path,
    breakout_criteria=args.breakout_criteria,
    early_threshold_min=args.early_threshold_min,
    bin_minutes=args.bin_minutes,
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
