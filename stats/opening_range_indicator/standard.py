"""Opening Range Indicator stat — standard variant.

Measures the **size of the opening range** (the first N minutes of the RTH
session) against the **range of the rest of the session**, and quantifies how
strongly the two are related.

Definitions:
  - **Opening range**: the high-low range of the bars in the opening window
    ``[rth_start, rth_start + orb_period)``. ``opening_range = orb_high - orb_low``.
  - **Remaining range**: the high-low range of the rest of the session,
    ``[rth_start + orb_period, rth_end)``. ``remaining_range = rest_high - rest_low``.
    The opening window is excluded so the two ranges never share bars.
  - **Session range**: the full RTH high-low range. Because the opening window and
    the remaining window exactly partition the RTH session, the full range is
    ``max(orb_high, rest_high) - min(orb_low, rest_low)``.

This is a **magnitude** stat (like ``session_reversal_range``): every outcome
carries its metric in ``StatResultRow.value`` (and its random baseline in
``value_baseline``); the ``probability`` channel is left at ``0.0``. A single
condition ``opening_range`` reports four metrics over the (sliced) day table:

  - ``mean_opening_range``   — average opening range in points.
  - ``mean_remaining_range`` — average remaining-session range in points.
  - ``opening_range_share``  — average of ``opening_range / session_range``, the
                               fraction of the full day's range already formed in
                               the opening window (a decimal in [0, 1]).
  - ``correlation``          — Pearson correlation coefficient between the opening
                               range and the remaining range across days (in
                               [-1, 1]). This is the headline: does a wide opening
                               range tend to precede a wide rest-of-session?

A day is countable only when it has a clean opening range AND a non-empty
remaining window; days missing either drop out via the inner join (pending-sample
discipline). ``total_samples`` counts all resolved sessions; each row's
``count`` / ``total`` equals the countable-day count.

Results are computed for three opening-range lengths, each emitted as its own
timeframe entry: 15min (09:30–09:45), 30min (09:30–10:00), 1h (09:30–10:30).

Declared slices re-run the whole computation per subset:
  - ``weekday`` — the "by weekday" breakdown.
  - ``close``   — split by session close color (green/red).
  - ``size``    — quartile buckets of the opening-range size; the per-bucket
                  ``mean_remaining_range`` makes the relationship visible directly.
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
from stats.utils.daily_candles import build_resolved_days

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="Opening Range Indicator",
  fr="Indicateur du range d'ouverture",
)
_DEFINITION = I18nString(
  en="How big is the opening range (the first N minutes of the session) relative to the rest-of-session range, and how strongly are the two correlated?",
  fr="Quelle est la taille du range d'ouverture (les N premières minutes de la séance) par rapport au range du reste de la séance, et à quel point les deux sont-ils corrélés ?",
)
_LABELS = Labels(
  conditions={
    "opening_range": I18nString(en="Opening range", fr="Range d'ouverture"),
  },
  outcomes={
    "mean_opening_range": I18nString(
      en="Average opening range (points)",
      fr="Range d'ouverture moyen (points)",
    ),
    "mean_remaining_range": I18nString(
      en="Average remaining-session range (points)",
      fr="Range moyen du reste de la séance (points)",
    ),
    "opening_range_share": I18nString(
      en="Opening range as a share of the session range",
      fr="Range d'ouverture en proportion du range de la séance",
    ),
    "correlation": I18nString(
      en="Correlation of opening range with remaining range",
      fr="Corrélation du range d'ouverture avec le range restant",
    ),
  },
)

# Single condition: every countable session belongs to it; the declared slices do
# the per-weekday / per-color / per-size breakdowns.
_CONDITION = "opening_range"

# Outcome enumeration, in display order.
_OUTCOMES: tuple[str, ...] = (
  "mean_opening_range",
  "mean_remaining_range",
  "opening_range_share",
  "correlation",
)

# ---------------------------------------------------------------------------
# Timeframe string → opening-range window length in minutes
# ---------------------------------------------------------------------------
_TF_MINUTES: dict[str, int] = {
  "15min": 15,
  "30min": 30,
  "1h": 60,
}


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
  """Pearson correlation between two arrays, 0.0 when undefined.

  Returns 0.0 for fewer than two points or when either series has zero variance
  (``np.corrcoef`` would return NaN, which JSON cannot represent).
  """
  if a.size < 2:
    return 0.0
  if a.std() == 0.0 or b.std() == 0.0:
    return 0.0
  return float(np.corrcoef(a, b)[0, 1])


class OpeningRangeIndicator(BaseStat):
  """Size of the opening range vs the remaining session range, and their correlation."""

  stat_name = "opening_range_indicator"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = (
    "weekday",
    "close",
    SizeBucket(column="opening_range", preset="quartiles", name="size"),
  )

  def __init__(
    self,
    instrument: str,
    timeframe: str,
    config: InstrumentConfig,
    close_tolerance_min: int = 15,
  ) -> None:
    if timeframe not in _TF_MINUTES:
      raise ValueError(f"Unsupported timeframe '{timeframe}'. Choose from {list(_TF_MINUTES)}")
    self.instrument = instrument
    self.timeframe = timeframe
    self.config = config
    self.orb_minutes = _TF_MINUTES[timeframe]
    self.close_tolerance_min = close_tolerance_min

    rth = config.sessions["rth"]
    self.rth_start_min: int = minute_of_day(rth.start)
    self.rth_end_min: int = minute_of_day(rth.end)
    # Opening window: [rth_start, orb_end). The remaining window is the rest of the
    # session, [orb_end, rth_end).
    self.orb_end_min: int = self.rth_start_min + self.orb_minutes

  # -------------------------------------------------------------------------
  # Day table construction
  # -------------------------------------------------------------------------
  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-session table with the opening, remaining, and session ranges.

    Columns returned:
      ``opening_range``   — ``orb_high - orb_low`` over the opening window
                            (read by the ``size`` slicer).
      ``remaining_range`` — ``rest_high - rest_low`` over the remaining window.
      ``session_range``   — full RTH high-low range (the share denominator).
      ``session_green``   — session color (``session_close >= session_open``),
                            read by the ``close`` slicer.

    The resolved-days core (RTH filter, session open/close, resolution filter,
    chronological sort) is shared via ``build_resolved_days``; the opening and
    remaining window aggregates are joined on per day. A day is countable only when
    it has a clean opening range AND a non-empty remaining window; days missing
    either drop out via the inner join (pending-sample discipline). The
    DatetimeIndex (normalized session date) is required by the ``weekday`` slicer.
    """
    columns = ["opening_range", "remaining_range", "session_range", "session_green"]
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

    # Opening-window aggregates: [rth_start, orb_end).
    orb_mask = (df["_mod"] >= self.rth_start_min) & (df["_mod"] < self.orb_end_min)
    orb = df[orb_mask]
    orb_high = orb.groupby("_date")["high"].max().rename("orb_high")
    orb_low = orb.groupby("_date")["low"].min().rename("orb_low")

    # Remaining-window aggregates: [orb_end, rth_end).
    rest_mask = (df["_mod"] >= self.orb_end_min) & (df["_mod"] < self.rth_end_min)
    rest = df[rest_mask]
    rest_high = rest.groupby("_date")["high"].max().rename("rest_high")
    rest_low = rest.groupby("_date")["low"].min().rename("rest_low")

    # Inner join onto the resolved index: a day survives only with a full opening
    # range AND a non-empty remaining window.
    day = (
      resolved.join(orb_high, how="inner")
      .join(orb_low, how="inner")
      .join(rest_high, how="inner")
      .join(rest_low, how="inner")
    )
    if day.empty:
      return empty

    day["opening_range"] = day["orb_high"] - day["orb_low"]
    day["remaining_range"] = day["rest_high"] - day["rest_low"]
    # The opening and remaining windows partition the RTH session, so the full RTH
    # range is the union of the two windows' extremes.
    session_high = day[["orb_high", "rest_high"]].max(axis=1)
    session_low = day[["orb_low", "rest_low"]].min(axis=1)
    day["session_range"] = session_high - session_low
    day["session_green"] = day["session_close"] >= day["session_open"]

    return day[columns]

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the four magnitude rows over a (possibly sliced) day subset.

    Each row carries its metric in ``value``; the ``probability`` channel is left
    at ``0.0``. Every row reports its sample size in ``count`` / ``total``. If
    ``baseline_rows`` is provided, the matching row's ``value`` is merged into each
    row's ``value_baseline``.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    total = len(day_table)
    values = self._metric_values(day_table)

    rows: list[StatResultRow] = []
    for out_key in _OUTCOMES:
      bl = baseline_map.get((_CONDITION, out_key))
      rows.append(
        StatResultRow(
          condition=_CONDITION,
          outcome=out_key,
          count=total,
          total=total,
          probability=0.0,
          baseline_prob=0.0,
          baseline_n=bl.total if bl else 0,
          value=values[out_key],
          value_baseline=bl.value if bl else None,
        )
      )
    return rows

  def _metric_values(self, day_table: pd.DataFrame) -> dict[str, float]:
    """Reduce a (sliced) day table to one value per outcome.

    Empty tables yield 0.0 for every metric. ``opening_range_share`` averages the
    per-day ratio only over days with a positive session range; ``correlation`` is
    0.0 when undefined (fewer than two days or zero variance).
    """
    if day_table.empty:
      return {out: 0.0 for out in _OUTCOMES}

    opening = day_table["opening_range"].to_numpy(dtype=float)
    remaining = day_table["remaining_range"].to_numpy(dtype=float)
    session = day_table["session_range"].to_numpy(dtype=float)

    valid_share = session > 0.0
    share = float((opening[valid_share] / session[valid_share]).mean()) if valid_share.any() else 0.0

    return {
      "mean_opening_range": float(opening.mean()),
      "mean_remaining_range": float(remaining.mean()),
      "opening_range_share": share,
      "correlation": _pearson(opening, remaining),
    }

  # -------------------------------------------------------------------------
  # Baseline
  # -------------------------------------------------------------------------
  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random null baseline. Deterministic for a fixed seed.

    The ``remaining_range`` column is **permuted** across days while
    ``opening_range`` and ``session_range`` are held fixed. This pairs each day's
    opening range with an unrelated day's remaining range, destroying the
    same-day link that the ``correlation`` metric measures.

    Expected baseline: ``correlation`` collapses toward 0 (the headline test —
    the actual correlation should sit well above its permuted null if opening-range
    size carries information about the rest of the session). ``mean_opening_range``,
    ``mean_remaining_range`` and ``opening_range_share`` are invariant under a
    permutation of ``remaining_range``, so their baselines equal their actual
    values by construction — they are descriptive context, not hypothesis-tested.
    Uses ``np.random.default_rng(seed)``.
    """
    n = len(day_table)
    if n == 0:
      return self.compute_rows(day_table, baseline_rows=None)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    tmp = day_table.copy()
    tmp["remaining_range"] = day_table["remaining_range"].to_numpy()[perm]

    return self.compute_rows(tmp, baseline_rows=None)


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
) -> Path:
  """Load data and compute Opening Range Indicator for all three opening lengths.

  Merges the per-timeframe TimeframeResults into one StatRunResult and writes the
  consolidated JSON to results/.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  timeframes = ["15min", "30min", "1h"]
  merged_tf: dict[str, TimeframeResult] = {}
  labels = _LABELS.model_copy(deep=True)

  for tf in timeframes:
    stat = OpeningRangeIndicator(instrument=instrument, timeframe=tf, config=config)
    result = stat.compute(candles_df)
    merged_tf[tf] = result.instruments[instrument][tf]
    # Merge the framework-enriched slice dimension labels so no timeframe's
    # dimensions overwrite an earlier one's.
    labels.dimensions.update(result.labels.dimensions)

  final_result = StatRunResult(
    stat_name="opening_range_indicator",
    title=_TITLE,
    definition=_DEFINITION,
    labels=labels,
    instruments={instrument: merged_tf},
  )

  return write_results(final_result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Opening Range Indicator stat")
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
    rows = {r["outcome"]: r for r in tf_data["results"]}
    open_pts = rows["mean_opening_range"]["value"]
    rest_pts = rows["mean_remaining_range"]["value"]
    share = rows["opening_range_share"]["value"]
    corr = rows["correlation"]["value"]
    corr_bl = rows["correlation"]["value_baseline"]
    print(
      f"    avg opening={open_pts:.2f} pts | avg remaining={rest_pts:.2f} pts "
      f"| share={share:.2%} | corr={corr:.3f} (baseline={corr_bl:.3f})"
    )
