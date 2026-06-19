"""Gap Fill stat — standard variant.

Measures how often a session opens with a gap from the prior session's close and
whether intraday price retraces back through a configurable percentage of that gap
("fills" it) before the session ends.

Definitions:
  - **Overnight gap**: today's ``session_open`` vs the PREVIOUS resolved session's
    ``session_close``.
      gap up:   ``session_open > prev_session_close``
      gap down: ``session_open < prev_session_close``
    A session that opens exactly at the prior close has no gap and is excluded.
  - **Fill**: intraday RTH price retraces back toward the prior close by at least
    ``fill_threshold_pct`` percent of the gap distance. The fill target is measured
    from the open back toward the prior close:
      gap up:   ``target = session_open - (pct/100) * gap_size``; filled when
                ``day_low <= target``.
      gap down: ``target = session_open + (pct/100) * gap_size``; filled when
                ``day_high >= target``.
    At the default ``pct = 100`` a full fill means price trades all the way back to
    the prior close (gap up → ``day_low <= prev_session_close``; gap down →
    ``day_high >= prev_session_close``).

Reported as a 2x2 conditional matrix: P(fill | gap up), P(no fill | gap up),
P(fill | gap down), P(no fill | gap down). The two outcomes partition each gap
direction, so they sum to 1.

The FIRST resolved session has no prior close, so its gap is undefined and it is
excluded from every denominator (pending-sample discipline). ``total_samples``
counts ALL resolved days; each row's ``total`` counts only the days that carry a
gap of that direction (a prior resolved close and a non-zero gap).
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
  en="Gap Fill",
  fr="Comblement de gap",
)
_DEFINITION = I18nString(
  en="When a session opens with a gap up or down from the prior session's close, how often does intraday price retrace back through the gap and fill it?",
  fr="Lorsqu'une session ouvre avec un gap à la hausse ou à la baisse par rapport à la clôture de la session précédente, à quelle fréquence le prix intraday revient-il combler le gap ?",
)
_LABELS = Labels(
  conditions={
    "gap_up": I18nString(en="Gap up", fr="Gap haussier"),
    "gap_down": I18nString(en="Gap down", fr="Gap baissier"),
  },
  outcomes={
    "filled": I18nString(en="Filled", fr="Comblé"),
    "not_filled": I18nString(en="Not filled", fr="Non comblé"),
  },
)

# Condition / outcome enumeration. Condition: gap direction (gap_up == gap_pts > 0).
# Outcome: whether the gap filled to the threshold.
_CONDITIONS: tuple[tuple[str, bool], ...] = (("gap_up", True), ("gap_down", False))
_OUTCOMES: tuple[tuple[str, bool], ...] = (("filled", True), ("not_filled", False))


class GapFill(BaseStat):
  """Conditional probability that an opening gap fills intraday, by gap direction."""

  stat_name = "gap_fill"
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
    fill_threshold_pct: float = 100.0,
    close_tolerance_min: int = 15,
  ) -> None:
    self.instrument = instrument
    self.timeframe = "daily"
    self.config = config
    self.fill_threshold_pct = fill_threshold_pct
    self.close_tolerance_min = close_tolerance_min

    rth = config.sessions["rth"]
    self.rth_start_min: int = minute_of_day(rth.start)
    self.rth_end_min: int = minute_of_day(rth.end)

  # -------------------------------------------------------------------------
  # Day table construction
  # -------------------------------------------------------------------------
  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-session table with gap direction, size, and fill columns.

    Columns returned:
      ``session_open``         — open of the RTH session-open bar.
      ``prev_session_close``   — prior resolved session's close (NaN for the first
                                 resolved day).
      ``gap_pts``              — signed gap ``session_open - prev_session_close``.
      ``gap_size_pts``         — absolute gap in points; NaN when there is no prior
                                 close or no gap (so non-countable days drop out of
                                 the size slicers and the denominators).
      ``gap_size_pct``         — gap as a **percent** of the prior close
                                 (``100 * gap_size_pts / prev_session_close``, e.g.
                                 ``0.8`` = 0.8%). Stored in percent units, not as a
                                 decimal, so the ``size_pct`` bucket-edge labels are
                                 legible (a label-only column — it is never reported
                                 as a value).
      ``gap_up``               — ``gap_pts > 0`` (only meaningful where countable).
      ``filled``               — whether the gap filled to ``fill_threshold_pct``.
      ``session_green``        — session color (``session_close >= session_open``),
                                 read by the ``close`` slicer.
      ``prev_session_green``   — prior session color, read by the ``prev_candle``
                                 slicer (NaN for the first resolved day).

    The resolved-days core (RTH filter, session open/close extraction, resolution
    filter, chronological sort) is shared via ``build_resolved_days``; the RTH
    intraday high/low are joined on per day for the fill detection. The
    DatetimeIndex (normalized session date) is required by the ``weekday`` slicer.
    """
    columns = [
      "session_open",
      "prev_session_close",
      "gap_pts",
      "gap_size_pts",
      "gap_size_pct",
      "gap_up",
      "filled",
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
    day["prev_session_close"] = day["session_close"].shift(1)
    day["prev_session_green"] = day["session_green"].shift(1)

    day["gap_pts"] = day["session_open"] - day["prev_session_close"]
    # A day is countable only with a prior close AND a non-zero gap.
    has_gap = day["prev_session_close"].notna() & (day["gap_pts"] != 0)
    day["gap_size_pts"] = day["gap_pts"].abs().where(has_gap)
    day["gap_size_pct"] = 100.0 * day["gap_size_pts"] / day["prev_session_close"]
    day["gap_up"] = day["gap_pts"] > 0

    # Fill detection: retrace from the open back toward the prior close by at least
    # the threshold fraction of the gap. Targets are NaN for non-countable days,
    # so their comparisons are False — they are excluded by ``gap_size_pts`` anyway.
    frac = self.fill_threshold_pct / 100.0
    up_target = day["session_open"] - frac * day["gap_size_pts"]
    down_target = day["session_open"] + frac * day["gap_size_pts"]
    filled_up = day["day_low"] <= up_target
    filled_down = day["day_high"] >= down_target
    day["filled"] = (day["gap_up"] & filled_up) | (~day["gap_up"] & filled_down)

    return day[columns]

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the four matrix rows (gap up/down x filled/not filled).

    Only days with a gap (``gap_size_pts`` not NaN) are countable. If
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
    gap_up = day_table["gap_up"].astype(bool)
    filled = day_table["filled"].astype(bool)

    rows: list[StatResultRow] = []
    for cond_key, cond_is_up in _CONDITIONS:
      cond_mask = countable & (gap_up == cond_is_up)
      total = int(cond_mask.sum())
      for out_key, out_is_filled in _OUTCOMES:
        out_match = filled if out_is_filled else ~filled
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

    The ``filled`` outcome is **permuted** across countable days, so the overall
    fill rate is preserved but its association with the gap direction is destroyed.
    Each condition's ``baseline_prob`` therefore converges to the pooled fill rate,
    and the comparison reveals whether gap up and gap down fill at *different* rates
    than the market does on average — a fair-coin baseline would instead anchor to
    0.5, which is not the relevant null for an event that fills well above half the
    time. Uses ``np.random.default_rng(seed)``.
    """
    n = len(day_table)
    if n == 0:
      return self.compute_rows(day_table, baseline_rows=None)

    countable = day_table["gap_size_pts"].notna().to_numpy()
    tmp = day_table.copy()
    if countable.any():
      rng = np.random.default_rng(seed)
      idx = day_table.index[countable]
      vals = day_table.loc[idx, "filled"].to_numpy()
      tmp.loc[idx, "filled"] = vals[rng.permutation(len(vals))]

    return self.compute_rows(tmp, baseline_rows=None)


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
  fill_threshold_pct: float = 100.0,
) -> Path:
  """Load data and compute Gap Fill for the daily timeframe.

  Writes the result JSON to results/ and returns its path.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  stat = GapFill(
    instrument=instrument, config=config, fill_threshold_pct=fill_threshold_pct
  )
  result = stat.compute(candles_df)

  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Gap Fill stat")
  parser.add_argument("--instrument", default="NQ", help="Instrument name (default: NQ)")
  parser.add_argument("--data-path", default=None, help="Override parquet file path")
  parser.add_argument("--config-dir", default="config", help="Config directory (default: config)")
  parser.add_argument(
    "--fill-threshold-pct",
    type=float,
    default=100.0,
    help="Fill threshold as a percent of the gap (default: 100 = full fill)",
  )
  args = parser.parse_args()

  output_path = run(
    instrument=args.instrument,
    config_dir=args.config_dir,
    data_path=args.data_path,
    fill_threshold_pct=args.fill_threshold_pct,
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
