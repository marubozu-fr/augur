"""Power Hour Continuation stat — standard variant.

Cross-tabulates the **pre-power-hour move** (the session's direction before the
power hour) against the **power-hour candle direction**. Does the power hour
continue the earlier move or reverse it?

Definitions:
  - **Power-hour window**: the last ``ph_period`` minutes of the session,
    ``[ph_start, rth_end)`` where ``ph_start = rth_end - ph_period`` (default 60
    minutes, the conventional final hour).
  - **Pre-power-hour window**: everything before it, ``[rth_start, ph_start)``.
  - **Pre-power-hour move** (the condition): the pre-power-hour candle color,
    ``pre_close`` (close of the last bar before the power hour) vs ``pre_open``
    (the session open, the open of the bar at ``rth_start``).
      pre green: pre_close >= pre_open
      pre red:   pre_close <  pre_open
  - **Power-hour direction** (the outcome): the power-hour candle color, the
    power-hour close (the session's last RTH bar close) vs the **reference open**.
    The reference open is the **power-hour open** (``ph_open``, the open of the bar
    at ``ph_start``) — the power-hour candle's own color.
      green: ph_close >= ph_open
      red:   ph_close <  ph_open

A 2x2 conditional matrix (mirrors ``overnight_continuation``):
  - P(green power hour | green pre-move), P(red power hour | green pre-move)
  - P(green power hour | red pre-move),   P(red power hour | red pre-move)

"Continuation" means the power hour closes in the same direction as the pre-move
(pre green → power hour green, pre red → power hour red). All four matrix cells
are reported so the asymmetry is directly visible.

``total_samples`` counts the **countable** sessions — those with a non-empty
pre-power-hour window AND a clean power-hour open bar (days missing either drop
out via the inner join, pending-sample discipline). Each row's ``total`` counts
the countable days in that condition (``pre_green`` or ``pre_red``), so the two
conditions' totals sum to ``total_samples``.

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
  SampleRow,
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
  en="Power Hour Continuation",
  fr="Continuation du power hour",
)
_DEFINITION = I18nString(
  en="Given the session's direction going into the power hour (the pre-power-hour move, green or red), how often does the power hour itself close in the same direction (continuation) versus reverse?",
  fr="Selon la direction de la séance à l'entrée du power hour (le mouvement pré-power-hour, vert ou rouge), à quelle fréquence le power hour clôture-t-il dans la même direction (continuation) plutôt que de se retourner ?",
)
_LABELS = Labels(
  conditions={
    "pre_green": I18nString(
      en="Green pre-power-hour move (up)",
      fr="Mouvement pré-power-hour vert (hausse)",
    ),
    "pre_red": I18nString(
      en="Red pre-power-hour move (down)",
      fr="Mouvement pré-power-hour rouge (baisse)",
    ),
  },
  outcomes={
    "green": I18nString(en="Green power hour (up)", fr="Power hour vert (hausse)"),
    "red": I18nString(en="Red power hour (down)", fr="Power hour rouge (baisse)"),
  },
)

# Condition / outcome enumeration: pre-power-hour move color -> power-hour color.
_CONDITIONS: tuple[tuple[str, bool], ...] = (("pre_green", True), ("pre_red", False))
_OUTCOMES: tuple[tuple[str, bool], ...] = (("green", True), ("red", False))

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


class PowerHourContinuation(BaseStat):
  """Conditional probability of the power-hour color given the pre-power-hour move."""

  stat_name = "power_hour_continuation"
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
    """Build the per-session table with the pre-power-hour and power-hour colors.

    Columns returned:
      ``pre_green``      — pre-power-hour candle color (pre_close >= session_open).
      ``ph_green``       — power-hour candle color (session_close >= ph_open).
      ``ph_open_above``  — power-hour open above the pre-power-hour range midpoint
                           (read by the ``open`` slicer).

    The resolved-days core (RTH filter, session open/close, resolution filter,
    chronological sort) is shared via ``build_resolved_days``: its ``session_open``
    is the pre-power-hour candle open and its ``session_close`` is the power-hour
    candle close. The pre-power-hour close and the power-hour open are joined on
    per day. A day is countable only when it has a non-empty pre-power-hour window
    AND a clean power-hour open bar; days missing either drop out via the inner
    join (pending-sample discipline). The DatetimeIndex (normalized session date)
    is required by the ``weekday`` slicer.
    """
    columns = ["pre_green", "ph_green", "ph_open_above"]
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

    # Pre-power-hour window [rth_start, ph_start): range extremes (for the open
    # slicer midpoint) and the closing bar (the pre-power-hour candle close).
    pre = df[(df["_mod"] >= self.rth_start_min) & (df["_mod"] < self.ph_start_min)]
    pre_high = pre.groupby("_date")["high"].max().rename("pre_high")
    pre_low = pre.groupby("_date")["low"].min().rename("pre_low")
    pre_close = (
      pre.sort_values("timestamp").groupby("_date")["close"].last().rename("pre_close")
    )

    # Power-hour open = open of the bar at exactly ph_start (mirrors the session-open
    # convention in build_resolved_days). A day without that bar is not countable.
    ph_open = (
      df[df["_mod"] == self.ph_start_min]
      .drop_duplicates("_date")
      .set_index("_date")["open"]
      .rename("ph_open")
    )

    # Inner join onto the resolved index: a day survives only with a full
    # pre-power-hour window AND a clean power-hour open bar.
    day = (
      resolved.join(pre_high, how="inner")
      .join(pre_low, how="inner")
      .join(pre_close, how="inner")
      .join(ph_open, how="inner")
    )
    if day.empty:
      return empty

    # Pre-power-hour candle: session_open is the pre-window open; pre_close its close.
    day["pre_green"] = day["pre_close"] >= day["session_open"]
    # Power-hour candle: session_close is the power-hour close; ph_open its open.
    day["ph_green"] = day["session_close"] >= day["ph_open"]
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
    """Compute the four matrix rows (pre green/red x power hour green/red).

    Every row in ``day_table`` is countable (the inner join already dropped days
    without a pre-power-hour window or a clean power-hour open). If
    ``baseline_rows`` is provided, merges its ``probability`` / ``total`` into each
    row's ``baseline_prob`` / ``baseline_n``.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    if day_table.empty:
      return [
        StatResultRow(
          condition=cond_key,
          outcome=out_key,
          count=0,
          total=0,
          probability=0.0,
          baseline_prob=0.0,
          baseline_n=0,
        )
        for cond_key, _ in _CONDITIONS
        for out_key, _ in _OUTCOMES
      ]

    pre_green = day_table["pre_green"].astype(bool)
    ph_green = day_table["ph_green"].astype(bool)

    rows: list[StatResultRow] = []
    for cond_key, cond_is_green in _CONDITIONS:
      cond_mask = pre_green if cond_is_green else ~pre_green
      total = int(cond_mask.sum())
      for out_key, out_is_green in _OUTCOMES:
        out_match = ph_green if out_is_green else ~ph_green
        count = int((cond_mask & out_match).sum())
        bl = baseline_map.get((cond_key, out_key))
        rows.append(
          StatResultRow(
            condition=cond_key,
            outcome=out_key,
            count=count,
            total=total,
            probability=count / total if total > 0 else 0.0,
            baseline_prob=bl.probability if bl else 0.0,
            baseline_n=bl.total if bl else 0,
          )
        )
    return rows

  # -------------------------------------------------------------------------
  # Per-day sample classification
  # -------------------------------------------------------------------------
  def classify_samples(self, day_table: pd.DataFrame) -> list[SampleRow]:
    """One SampleRow per resolved day, mirroring ``compute_rows``.

    Every row in ``day_table`` is countable (the inner join in
    ``build_day_table`` already dropped days without a pre-power-hour window or
    a clean power-hour open). Each day belongs to exactly one condition
    (``pre_green`` / ``pre_red``, from ``pre_green``) and exactly one outcome
    (``green`` / ``red``, from ``ph_green``) — the 2x2 matrix is a single
    partition of countable days, so each day yields exactly one sample.
    """
    if day_table.empty:
      return []

    samples: list[SampleRow] = []
    for ts, is_pre_green, is_ph_green in zip(
      day_table.index, day_table["pre_green"], day_table["ph_green"]
    ):
      date_str = ts.strftime("%Y-%m-%d")
      condition = "pre_green" if is_pre_green else "pre_red"
      outcome = "green" if is_ph_green else "red"
      samples.append(SampleRow(date=date_str, condition=condition, outcome=outcome))
    return samples

  # -------------------------------------------------------------------------
  # Baseline
  # -------------------------------------------------------------------------
  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random baseline: randomize the power-hour color (p=0.5), destroying any
    correlation with the pre-power-hour move.

    The pre-power-hour condition stays as-is from real data; only ``ph_green`` is
    replaced with an independent random sequence. Expected ``baseline_prob`` ~ 0.5
    for every row, so the comparison reveals whether the pre-move predicts the
    power-hour direction beyond a coin flip. Uses ``np.random.default_rng(seed)``.
    """
    rng = np.random.default_rng(seed)
    n = len(day_table)
    random_green = rng.integers(0, 2, size=n).astype(bool)

    tmp = day_table.copy()
    tmp["ph_green"] = random_green
    return self.compute_rows(tmp, baseline_rows=None)


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
) -> Path:
  """Load data and compute the power hour continuation for the 1h power-hour window.

  Writes the result JSON to results/ and returns its path.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  timeframes = list(_TF_MINUTES)
  merged_tf: dict[str, TimeframeResult] = {}
  labels = _LABELS.model_copy(deep=True)

  for tf in timeframes:
    stat = PowerHourContinuation(instrument=instrument, timeframe=tf, config=config)
    result = stat.compute(candles_df)
    merged_tf[tf] = result.instruments[instrument][tf]
    # Merge the framework-enriched slice dimension labels so no timeframe's
    # dimensions overwrite an earlier one's.
    labels.dimensions.update(result.labels.dimensions)

  final_result = StatRunResult(
    stat_name="power_hour_continuation",
    title=_TITLE,
    definition=_DEFINITION,
    labels=labels,
    instruments={instrument: merged_tf},
  )

  return write_results(final_result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Power Hour Continuation stat")
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
