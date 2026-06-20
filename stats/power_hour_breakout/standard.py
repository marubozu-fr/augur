"""Power Hour Breakout stat — standard variant.

Measures the **power hour** — the last ``ph_period`` minutes of the RTH session —
and classifies whether price makes a new high of day, a new low of day, both, or
neither relative to the range established EARLIER in the session.

Definitions:
  - **Power hour window**: the last ``ph_period`` minutes of the session,
    ``[rth_end - ph_period, rth_end)`` (default 60 minutes, the conventional
    final hour).
  - **Pre-power-hour window**: everything before it, ``[rth_start, ph_start)``.
    ``pre_high`` is the max high and ``pre_low`` the min low over that window —
    the session's high/low of day going into the power hour.
  - **New extreme**: how the power-hour window's price relates to the pre-power-hour
    range. A new high of day requires the power-hour high to exceed ``pre_high``;
    a new low of day requires the power-hour low to fall below ``pre_low``. Uses
    the window's intraday extremes (a wick beyond the level counts).

Four MUTUALLY EXCLUSIVE outcomes partition every countable day exhaustively
(strict inequality defines a new extreme — touching the level exactly is NOT a
break, mirroring the sibling ``initial_balance`` / ``opening_range_breakout``
convention):
  - ``made_high`` — made a new high of day only (pre-PH low held)
  - ``made_low``  — made a new low of day only (pre-PH high held)
  - ``made_both`` — made both a new high and a new low during the power hour
  - ``neither``   — stayed entirely within the pre-power-hour range

Reported under a single ``power_hour`` condition; the four outcomes sum to the
countable day count, so their probabilities sum to 1. ``total_samples`` counts ALL
resolved days; each row's ``total`` is the countable days (a non-empty
pre-power-hour window AND a clean power-hour open AND a non-empty power-hour
window). The marginal new-high / new-low rates are recoverable as
``made_high + made_both`` and ``made_low + made_both``.

The power hour is conventionally the final hour, so results are computed for the
60-minute window, emitted as the ``1h`` timeframe entry.

Declared slices re-run the whole computation per subset:
  - ``weekday`` — the "by weekday" breakdown.
  - ``open``    — the "by open" breakdown: where the power-hour open falls relative
                  to the pre-power-hour range midpoint (upper / lower half).
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
  SliceGroup,
  Slicer,
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
  en="Power Hour Breakout",
  fr="Cassure du power hour",
)
_DEFINITION = I18nString(
  en="During the last hour of the session (the power hour), how often does price make a new high of day, a new low of day, both, or neither relative to the range established earlier in the session?",
  fr="Pendant la dernière heure de la séance (le power hour), à quelle fréquence le prix inscrit-il un nouveau plus haut du jour, un nouveau plus bas du jour, les deux, ou ni l'un ni l'autre par rapport à l'amplitude établie plus tôt dans la séance ?",
)
_LABELS = Labels(
  conditions={
    "power_hour": I18nString(en="Power hour", fr="Power hour"),
  },
  outcomes={
    "made_high": I18nString(
      en="Made a new high of day only",
      fr="Nouveau plus haut du jour uniquement",
    ),
    "made_low": I18nString(
      en="Made a new low of day only",
      fr="Nouveau plus bas du jour uniquement",
    ),
    "made_both": I18nString(
      en="Made a new high and a new low",
      fr="Nouveau plus haut et nouveau plus bas",
    ),
    "neither": I18nString(
      en="Stayed within the prior range",
      fr="Resté dans l'amplitude antérieure",
    ),
  },
)

# Outcome enumeration, in display order. All four partition the countable days.
_OUTCOMES: tuple[str, ...] = ("made_high", "made_low", "made_both", "neither")

# ---------------------------------------------------------------------------
# Timeframe string → power-hour window length in minutes
# ---------------------------------------------------------------------------
_TF_MINUTES: dict[str, int] = {
  "1h": 60,
}


# ---------------------------------------------------------------------------
# Power-hour-specific slicer ("by open" variant)
# ---------------------------------------------------------------------------
class PowerHourOpen(Slicer):
  """Split by where the power-hour open falls relative to the pre-power-hour
  range midpoint: the upper half (``above``) or the lower half (``below``).

  Reads the precomputed ``ph_open_above`` boolean column. Rows missing it are
  excluded from every group (pending-sample discipline)."""

  name = "open"

  def dimension_label(self) -> I18nString:
    return I18nString(en="Power hour open", fr="Ouverture du power hour")

  def split(self, day_table: pd.DataFrame) -> list[SliceGroup]:
    if len(day_table) == 0 or "ph_open_above" not in day_table.columns:
      return []
    col = day_table["ph_open_above"]
    present = col.notna()
    groups: list[SliceGroup] = []
    specs = [
      (
        "above",
        I18nString(en="Open in upper half", fr="Ouverture dans la moitié haute"),
        present & col.fillna(False).astype(bool),
      ),
      (
        "below",
        I18nString(en="Open in lower half", fr="Ouverture dans la moitié basse"),
        present & ~col.fillna(True).astype(bool),
      ),
    ]
    for key, label, mask in specs:
      if bool(mask.any()):
        groups.append(SliceGroup(key=key, label=label, mask=mask))
    return groups


class PowerHourBreakout(BaseStat):
  """New-extreme distribution of the power hour relative to the prior session range."""

  stat_name = "power_hour_breakout"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = (
    "weekday",
    PowerHourOpen(),
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
    self.ph_minutes = _TF_MINUTES[timeframe]
    self.close_tolerance_min = close_tolerance_min

    rth = config.sessions["rth"]
    self.rth_start_min: int = minute_of_day(rth.start)
    self.rth_end_min: int = minute_of_day(rth.end)
    # Power-hour window: [ph_start, rth_end). The pre-power-hour window is the rest
    # of the session that came before it, [rth_start, ph_start).
    self.ph_start_min: int = self.rth_end_min - self.ph_minutes

  # -------------------------------------------------------------------------
  # Day table construction
  # -------------------------------------------------------------------------
  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-session table with the pre-power-hour range and power-hour extremes.

    Columns returned:
      ``pre_high`` / ``pre_low``   — pre-power-hour extremes (the high/low of day
                                     going into the power hour).
      ``ph_high`` / ``ph_low``     — power-hour intraday extremes.
      ``ph_open_above``            — power-hour open above the pre-power-hour range
                                     midpoint (read by the ``open`` slicer).

    The resolved-days core (RTH filter, session open/close, resolution filter,
    chronological sort) is shared via ``build_resolved_days``; the pre-power-hour
    and power-hour aggregates are joined on per day. A day is countable only when
    it has a non-empty pre-power-hour window, a clean power-hour open bar, AND a
    non-empty power-hour window; days missing any drop out via the inner join
    (pending-sample discipline). The DatetimeIndex (normalized session date) is
    required by the ``weekday`` slicer.
    """
    columns = ["pre_high", "pre_low", "ph_high", "ph_low", "ph_open_above"]
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

    # Pre-power-hour window aggregates: [rth_start, ph_start).
    pre_mask = (df["_mod"] >= self.rth_start_min) & (df["_mod"] < self.ph_start_min)
    pre = df[pre_mask]
    pre_high = pre.groupby("_date")["high"].max().rename("pre_high")
    pre_low = pre.groupby("_date")["low"].min().rename("pre_low")

    # Power-hour window aggregates: [ph_start, rth_end).
    ph_mask = (df["_mod"] >= self.ph_start_min) & (df["_mod"] < self.rth_end_min)
    ph = df[ph_mask]
    ph_high = ph.groupby("_date")["high"].max().rename("ph_high")
    ph_low = ph.groupby("_date")["low"].min().rename("ph_low")

    # Power-hour open = open of the bar at exactly ph_start (mirrors the session-open
    # convention in build_resolved_days). A day without that bar is not countable.
    ph_open = (
      df[df["_mod"] == self.ph_start_min]
      .drop_duplicates("_date")
      .set_index("_date")["open"]
      .rename("ph_open")
    )

    # Inner join onto the resolved index: a day survives only with a full
    # pre-power-hour window, a clean power-hour open, AND a power-hour window.
    day = (
      resolved.join(pre_high, how="inner")
      .join(pre_low, how="inner")
      .join(ph_high, how="inner")
      .join(ph_low, how="inner")
      .join(ph_open, how="inner")
    )
    if day.empty:
      return empty

    # Power-hour open relative to the pre-power-hour range midpoint (upper/lower half).
    mid = (day["pre_high"] + day["pre_low"]) / 2.0
    day["ph_open_above"] = day["ph_open"] >= mid

    return day[columns]

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the four new-extreme rows over the (sliced) day table.

    Only days with a valid pre-power-hour range and power-hour extremes are
    countable. The four outcomes are mutually exclusive and partition the
    countable days, so their counts sum to ``total``. If ``baseline_rows`` is
    provided, merges its ``probability`` / ``total`` into each row's
    ``baseline_prob`` / ``baseline_n``.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    def _make(outcome: str, count: int, total: int) -> StatResultRow:
      bl = baseline_map.get(("power_hour", outcome))
      return StatResultRow(
        condition="power_hour",
        outcome=outcome,
        count=count,
        total=total,
        probability=count / total if total > 0 else 0.0,
        baseline_prob=bl.probability if bl else 0.0,
        baseline_n=bl.total if bl else 0,
      )

    if day_table.empty:
      return [_make(out, 0, 0) for out in _OUTCOMES]

    pre_high = day_table["pre_high"]
    pre_low = day_table["pre_low"]
    ph_high = day_table["ph_high"]
    ph_low = day_table["ph_low"]

    countable = pre_high.notna() & pre_low.notna() & ph_high.notna() & ph_low.notna()
    total = int(countable.sum())

    # Strict inequality defines a new extreme (touching the level exactly is not).
    above = countable & (ph_high > pre_high)
    below = countable & (ph_low < pre_low)

    counts = {
      "made_high": int((above & ~below).sum()),
      "made_low": int((below & ~above).sum()),
      "made_both": int((above & below).sum()),
      "neither": int((countable & ~above & ~below).sum()),
    }

    return [_make(out, counts[out], total) for out in _OUTCOMES]

  # -------------------------------------------------------------------------
  # Baseline
  # -------------------------------------------------------------------------
  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random directional baseline. Deterministic for a fixed seed.

    Each countable day's power-hour move is REFLECTED around its pre-power-hour
    range midpoint with probability 0.5 (a price ``p`` maps to ``2*mid - p``,
    which swaps the high and low extremes). Reflection turns a ``made_high`` day
    into a ``made_low`` day and vice versa, while ``made_both`` and ``neither``
    are direction-symmetric and unchanged. The null therefore has NO directional
    bias: ``made_high`` and ``made_low`` converge to their shared mean, so the
    comparison reveals whether the power hour breaks UP more often than DOWN
    beyond a coin flip. Uses ``np.random.default_rng(seed)``.
    """
    n = len(day_table)
    if n == 0:
      return self.compute_rows(day_table, baseline_rows=None)

    rng = np.random.default_rng(seed)
    flip = rng.integers(0, 2, size=n).astype(bool)

    mid = (day_table["pre_high"].to_numpy() + day_table["pre_low"].to_numpy()) / 2.0
    tmp = day_table.copy()

    # Reflect the (high, low) extremes around the midpoint, then keep the reflected
    # value only on flipped days.
    hi = day_table["ph_high"].to_numpy()
    lo = day_table["ph_low"].to_numpy()
    refl_hi = 2.0 * mid - lo
    refl_lo = 2.0 * mid - hi
    tmp["ph_high"] = np.where(flip, refl_hi, hi)
    tmp["ph_low"] = np.where(flip, refl_lo, lo)

    return self.compute_rows(tmp, baseline_rows=None)


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
) -> Path:
  """Load data and compute the power hour breakout for the 1h power-hour window.

  Writes the result JSON to results/ and returns its path.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  timeframes = list(_TF_MINUTES)
  merged_tf: dict[str, TimeframeResult] = {}
  labels = _LABELS.model_copy(deep=True)

  for tf in timeframes:
    stat = PowerHourBreakout(instrument=instrument, timeframe=tf, config=config)
    result = stat.compute(candles_df)
    merged_tf[tf] = result.instruments[instrument][tf]
    # Merge the framework-enriched slice dimension labels so no timeframe's
    # dimensions overwrite an earlier one's.
    labels.dimensions.update(result.labels.dimensions)

  final_result = StatRunResult(
    stat_name="power_hour_breakout",
    title=_TITLE,
    definition=_DEFINITION,
    labels=labels,
    instruments={instrument: merged_tf},
  )

  return write_results(final_result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Power Hour Breakout stat")
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
