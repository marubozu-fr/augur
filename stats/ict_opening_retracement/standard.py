"""ICT Opening Retracement stat — standard variant.

Measures how often a session opens above or below an ICT reference candle level
and whether intraday RTH price retraces back to touch that level before the
session ends.

Definitions:
  - **ICT reference candle level**: the price of a single bar whose minute-of-day
    matches ``reference_time`` (default ``00:00`` ET — the midnight candle), using
    the ``reference_price`` field (default ``open``).  The bar is looked up in the
    FULL candle series (not restricted to RTH), since the reference time may fall
    outside regular trading hours.
  - **Date alignment**: the midnight (``00:00``) bar on calendar date D has
    ``timestamp.normalize() == D``, which is the same normalized date as the RTH
    session that opens on D at 09:30.  This one-to-one alignment by calendar date
    is the reason the default ``reference_time = "00:00"`` works without any
    date-offset arithmetic.  Other reference times outside RTH follow the same
    convention: a bar at 18:00 on date D aligns with the RTH session on date D.
  - **Direction** (strict; opening exactly AT the reference level → no direction,
    excluded from denominators):
      opened above: ``session_open > reference_level``
      opened below: ``session_open < reference_level``
  - **Retracement**: intraday RTH price touches the reference level.
      opened above → retraced when ``day_low <= reference_level``.
      opened below → retraced when ``day_high >= reference_level``.
    Touching the level exactly counts as a retracement (``<=`` / ``>=``).

Reported as a 2x2 conditional matrix: P(retraced | opened above),
P(not retraced | opened above), P(retraced | opened below),
P(not retraced | opened below). The two outcomes partition each direction, so
they sum to 1.

A resolved day with no reference bar for that calendar date has
``reference_level = NaN`` and is excluded from every denominator (but still
counted in ``total_samples``). The first resolved session is always included in
``total_samples``; only the directional denominators (sessions with a valid,
non-zero gap) matter for counting.
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
  write_results,
)
from stats.config import InstrumentConfig, load_config, minute_of_day
from stats.utils.daily_candles import build_resolved_days

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="ICT Opening Retracement",
  fr="Retracement d'ouverture ICT",
)
_DEFINITION = I18nString(
  en="When a session opens above or below the ICT reference candle level (default the midnight open), how often does intraday price retrace back to that level?",
  fr="Lorsqu'une session ouvre au-dessus ou en dessous du niveau de la bougie de référence ICT (par défaut l'ouverture de minuit), à quelle fréquence le prix intraday revient-il à ce niveau ?",
)
_LABELS = Labels(
  conditions={
    "opened_above": I18nString(en="Opened above", fr="Ouverture au-dessus"),
    "opened_below": I18nString(en="Opened below", fr="Ouverture en dessous"),
  },
  outcomes={
    "retraced": I18nString(en="Retraced", fr="Retracé"),
    "not_retraced": I18nString(en="Not retraced", fr="Non retracé"),
  },
)

# Condition / outcome enumeration. Condition: direction (opened_above == gap_pts > 0).
# Outcome: whether intraday price retraced back to the reference level.
_CONDITIONS: tuple[tuple[str, bool], ...] = (
  ("opened_above", True),
  ("opened_below", False),
)
_OUTCOMES: tuple[tuple[str, bool], ...] = (
  ("retraced", True),
  ("not_retraced", False),
)


class IctOpeningRetracement(BaseStat):
  """Conditional probability that RTH intraday price retraces to the ICT reference
  candle level, by session open direction relative to that level."""

  stat_name = "ict_opening_retracement"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = (
    "weekday",
    "close",
    "prev_candle",
    SizeBucket(column="gap_size_pts", preset="quartiles", name="size_pts"),
    SizeBucket(column="gap_size_pct", preset="quartiles", name="size_pct"),
  )

  def __init__(
    self,
    instrument: str,
    config: InstrumentConfig,
    reference_time: str = "00:00",
    reference_price: str = "open",
    close_tolerance_min: int = 15,
  ) -> None:
    if reference_price not in {"open", "high", "low", "close"}:
      raise ValueError(
        f"reference_price must be one of open/high/low/close, got '{reference_price}'"
      )
    self.instrument = instrument
    self.timeframe = "daily"
    self.config = config
    self.reference_time = reference_time
    self.reference_price = reference_price
    self.reference_time_min: int = minute_of_day(reference_time)
    self.close_tolerance_min = close_tolerance_min

    rth = config.sessions["rth"]
    self.rth_start_min: int = minute_of_day(rth.start)
    self.rth_end_min: int = minute_of_day(rth.end)

  # -------------------------------------------------------------------------
  # Day table construction
  # -------------------------------------------------------------------------
  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-session table with direction, size, and retracement columns.

    Columns returned:
      ``session_open``       — open of the RTH session-open bar.
      ``reference_level``    — price of the reference bar for that calendar date
                               (``NaN`` when no reference bar exists for the date).
      ``gap_pts``            — signed distance ``session_open - reference_level``.
      ``gap_size_pts``       — absolute distance in points; ``NaN`` when there is no
                               reference level or no direction (so non-countable days
                               drop out of the size slicers and the denominators).
      ``gap_size_pct``       — distance as a **percent** of the reference level
                               (``100 * gap_size_pts / reference_level``, e.g.
                               ``0.8`` = 0.8%).  Stored in percent units for legible
                               size-bucket labels — a label-only column, never
                               reported as a value.
      ``opened_above``       — ``gap_pts > 0`` (only meaningful where countable).
      ``retraced``           — whether intraday RTH price touched the reference level.
      ``session_green``      — session color (``session_close >= session_open``),
                               read by the ``close`` slicer.
      ``prev_session_green`` — prior resolved session color, read by the
                               ``prev_candle`` slicer (``NaN`` for the first
                               resolved day).

    The reference bar is looked up from the FULL ``candles_df`` (not restricted
    to RTH) so that reference times outside regular trading hours (e.g. midnight)
    are found. The bar at ``reference_time_min`` is matched by minute-of-day and
    keyed by normalized calendar date, then left-joined onto the resolved RTH
    day index. Because ``timestamp.normalize()`` maps both the midnight bar and
    the 09:30 open bar to the same calendar date D, no date-offset is required.

    The resolved-days core (RTH filter, session open/close extraction, resolution
    filter, chronological sort) is shared via ``build_resolved_days``; the RTH
    intraday high/low are joined per day for retracement detection. The
    DatetimeIndex (normalized session date) is required by the ``weekday`` slicer.
    """
    columns = [
      "session_open",
      "reference_level",
      "gap_pts",
      "gap_size_pts",
      "gap_size_pct",
      "opened_above",
      "retraced",
      "session_green",
      "prev_session_green",
    ]
    empty = pd.DataFrame(columns=columns)

    if candles_df.empty:
      return empty

    resolved = build_resolved_days(
      candles_df, self.rth_start_min, self.rth_end_min, self.close_tolerance_min
    )
    if resolved.empty:
      return empty

    # Per-day RTH high/low from the same RTH bar filter the resolution uses.
    df = candles_df.copy()
    df["_mod"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["_date"] = df["timestamp"].dt.normalize()
    rth_mask = (df["_mod"] >= self.rth_start_min) & (df["_mod"] < self.rth_end_min)
    rth = df[rth_mask]

    day_high = rth.groupby("_date")["high"].max().rename("day_high")
    day_low = rth.groupby("_date")["low"].min().rename("day_low")

    # Inner join onto the resolved index: only resolved dates pass through.
    day = resolved.join(day_high, how="inner").join(day_low, how="inner")
    if day.empty:
      return empty

    day["session_green"] = day["session_close"] >= day["session_open"]
    # Prior RESOLVED session (shift over the resolved-only, sorted index, so it
    # skips over any excluded/early-close day).
    day["prev_session_green"] = day["session_green"].shift(1)

    # Reference level: from the FULL candles_df (NOT just RTH bars), select bars
    # whose minute-of-day matches reference_time_min, keyed by normalized date.
    ref_bars = df[df["_mod"] == self.reference_time_min].copy()
    ref_level = (
      ref_bars.set_index("_date")[self.reference_price].rename("reference_level")
    )
    # Defensive: guard against duplicate reference bars for the same date.
    ref_level = ref_level[~ref_level.index.duplicated(keep="first")]

    # Left join by calendar date: resolved days with no reference bar get NaN.
    day = day.join(ref_level, how="left")

    # Direction (strict): opening exactly AT the reference level has no direction
    # and is excluded from every denominator (pending-sample discipline).
    day["gap_pts"] = day["session_open"] - day["reference_level"]
    has_direction = day["reference_level"].notna() & (day["gap_pts"] != 0)
    day["gap_size_pts"] = day["gap_pts"].abs().where(has_direction)
    day["gap_size_pct"] = 100.0 * day["gap_size_pts"] / day["reference_level"]
    day["opened_above"] = day["gap_pts"] > 0

    # Retracement: touching the reference level exactly counts (<=, >=).
    # Non-countable days' retraced value is irrelevant — excluded via gap_size_pts.
    retraced_above = day["opened_above"] & (day["day_low"] <= day["reference_level"])
    retraced_below = ~day["opened_above"] & (day["day_high"] >= day["reference_level"])
    day["retraced"] = retraced_above | retraced_below

    return day[columns]

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the four matrix rows (opened above/below x retraced/not retraced).

    Only days with a direction (``gap_size_pts`` not NaN) are countable. If
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

    countable = day_table["gap_size_pts"].notna()
    opened_above = day_table["opened_above"].astype(bool)
    retraced = day_table["retraced"].astype(bool)

    rows: list[StatResultRow] = []
    for cond_key, cond_is_above in _CONDITIONS:
      cond_mask = countable & (opened_above == cond_is_above)
      total = int(cond_mask.sum())
      for out_key, out_is_retraced in _OUTCOMES:
        out_match = retraced if out_is_retraced else ~retraced
        count = int((cond_mask & out_match).sum())
        probability = count / total if total > 0 else 0.0
        bl = baseline_map.get((cond_key, out_key))
        rows.append(
          StatResultRow(
            condition=cond_key,
            outcome=out_key,
            count=count,
            total=total,
            probability=probability,
            baseline_prob=bl.probability if bl else 0.0,
            baseline_n=bl.total if bl else 0,
          )
        )
    return rows

  # -------------------------------------------------------------------------
  # Baseline
  # -------------------------------------------------------------------------
  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random null baseline. Deterministic for a fixed seed.

    The ``retraced`` outcome is **permuted** across countable days, so the overall
    retracement rate is preserved but its association with the open direction is
    destroyed.  Each condition's ``baseline_prob`` therefore converges to the
    pooled retracement rate, and the comparison reveals whether opening above and
    opening below retrace at *different* rates than the market does on average — a
    fair-coin baseline would instead anchor to 0.5, which is not the relevant null
    for an event that may retrace well above or below half the time.  Uses
    ``np.random.default_rng(seed)``.
    """
    n = len(day_table)
    if n == 0:
      return self.compute_rows(day_table, baseline_rows=None)

    countable = day_table["gap_size_pts"].notna().to_numpy()
    tmp = day_table.copy()
    if countable.any():
      rng = np.random.default_rng(seed)
      idx = day_table.index[countable]
      vals = day_table.loc[idx, "retraced"].to_numpy()
      tmp.loc[idx, "retraced"] = vals[rng.permutation(len(vals))]

    return self.compute_rows(tmp, baseline_rows=None)


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
  reference_time: str = "00:00",
  reference_price: str = "open",
) -> Path:
  """Load data and compute ICT Opening Retracement for the daily timeframe.

  Writes the result JSON to results/ and returns its path.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  stat = IctOpeningRetracement(
    instrument=instrument,
    config=config,
    reference_time=reference_time,
    reference_price=reference_price,
  )
  result = stat.compute(candles_df)

  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute ICT Opening Retracement stat")
  parser.add_argument("--instrument", default="NQ", help="Instrument name (default: NQ)")
  parser.add_argument("--data-path", default=None, help="Override parquet file path")
  parser.add_argument("--config-dir", default="config", help="Config directory (default: config)")
  parser.add_argument(
    "--reference-time",
    default="00:00",
    help="Reference bar time as HH:MM in instrument timezone (default: 00:00)",
  )
  parser.add_argument(
    "--reference-price",
    default="open",
    choices=["open", "high", "low", "close"],
    help="OHLC field of the reference bar to use as the level (default: open)",
  )
  args = parser.parse_args()

  output_path = run(
    instrument=args.instrument,
    config_dir=args.config_dir,
    data_path=args.data_path,
    reference_time=args.reference_time,
    reference_price=args.reference_price,
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
