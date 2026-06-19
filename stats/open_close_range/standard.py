"""Open to Close Range stat — standard variant.

Measures how often a session's RTH daily candle closes **within** a given
percentage range of its open, versus closing **outside** that range.

For each resolved session the open-to-close move is measured as a percentage of
the open::

    oc_move_pct = abs(session_close - session_open) / session_open * 100

A single condition ``open_close_range`` is reported with two outcomes that
partition every resolved session:

  - ``within``:  ``oc_move_pct <= range_percentage``  (the close lands inside the
                 band around the open).
  - ``outside``: ``oc_move_pct > range_percentage``.

Every resolved session is countable — there is no warm-up window — so each row's
``total`` equals ``total_samples``.

The issue's ``day_type`` filter (green vs red session) is exposed idiomatically
via the framework's ``close`` slice, which splits the result by session color in
addition to the ``weekday`` breakdown.

Declared slices:

  - ``weekday`` — the "by weekday" breakdown.
  - ``close``   — the "by session color" breakdown (the ``day_type`` dimension).
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
  StatResultRow,
  write_results,
)
from stats.config import InstrumentConfig, load_config, minute_of_day
from stats.utils.daily_candles import build_resolved_days

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="Open to Close Range",
  fr="Range ouverture-clôture",
)
_DEFINITION = I18nString(
  en="How often does a session close within a given percentage range of its open?",
  fr="À quelle fréquence une session clôture-t-elle dans une plage de pourcentage donnée par rapport à son ouverture ?",
)
_LABELS = Labels(
  conditions={
    "open_close_range": I18nString(
      en="Open-to-close move vs range threshold",
      fr="Mouvement ouverture-clôture vs seuil de plage",
    ),
  },
  outcomes={
    "within": I18nString(en="Within range", fr="Dans la plage"),
    "outside": I18nString(en="Outside range", fr="Hors de la plage"),
  },
)

# Single condition: every resolved session belongs to it; the declared slices do
# the per-weekday / per-color breakdowns.
_CONDITION = "open_close_range"


class OpenCloseRange(BaseStat):
  """How often a session closes within a percentage range of its open."""

  stat_name = "open_close_range"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = ("weekday", "close")

  def __init__(
    self,
    instrument: str,
    config: InstrumentConfig,
    range_percentage: float = 1.0,
    close_tolerance_min: int = 15,
  ) -> None:
    if range_percentage <= 0:
      raise ValueError(f"range_percentage must be > 0, got {range_percentage}")
    self.instrument = instrument
    self.timeframe = "daily"
    self.config = config
    self.range_percentage = float(range_percentage)
    self.close_tolerance_min = close_tolerance_min

    rth = config.sessions["rth"]
    self.rth_start_min: int = minute_of_day(rth.start)
    self.rth_end_min: int = minute_of_day(rth.end)

  # -------------------------------------------------------------------------
  # Day table construction
  # -------------------------------------------------------------------------
  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-session table with the open-to-close move and color.

    Columns returned:
      ``session_open``  — open of the RTH session-open bar.
      ``session_close`` — close of the last RTH bar.
      ``oc_move_pct``   — absolute open-to-close move as a percent of the open::

                             abs(session_close - session_open) / session_open * 100

      ``session_green`` — session color (``session_close >= session_open``), read
                          by the ``close`` slicer (the ``day_type`` dimension).

    The resolved-days core (RTH filter, session open/close extraction, resolution
    filter, chronological sort) is shared via ``build_resolved_days``. Every
    resolved session is countable: there is no warm-up window. The DatetimeIndex
    (normalized session date) is required by the ``weekday`` slicer.
    """
    columns = ["session_open", "session_close", "oc_move_pct", "session_green"]

    day = build_resolved_days(
      candles_df, self.rth_start_min, self.rth_end_min, self.close_tolerance_min
    )
    if day.empty:
      return pd.DataFrame(columns=columns)

    day = day.copy()
    day["oc_move_pct"] = (
      (day["session_close"] - day["session_open"]).abs() / day["session_open"] * 100.0
    )
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
    """Compute the within / outside rows for one (possibly sliced) day subset.

    The two outcomes partition every resolved session exactly. If
    ``baseline_rows`` is provided, merges ``baseline_prob`` / ``baseline_n`` into
    the rows.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    total = len(day_table)
    if total > 0:
      within_count = int((day_table["oc_move_pct"] <= self.range_percentage).sum())
    else:
      within_count = 0

    outcomes = [
      ("within", within_count),
      ("outside", total - within_count),
    ]

    rows: list[StatResultRow] = []
    for out_key, count in outcomes:
      bl = baseline_map.get((_CONDITION, out_key))
      rows.append(
        StatResultRow(
          condition=_CONDITION,
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
  # Baseline
  # -------------------------------------------------------------------------
  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random null baseline. Deterministic for a fixed seed.

    The ``session_close`` column is **permuted** across all rows while
    ``session_open`` is held fixed, and the open-to-close move is recomputed
    against each session's own open. This pairs every session's open with an
    unrelated session's close, destroying the same-session open-to-close link.

    Expected baseline: a random historical close lands within the (typically
    tight) ``range_percentage`` band of a given session's open far less often
    than the session's own close does, so the within-rate is much lower than the
    actual — confirming that the same-session move is genuinely contained rather
    than an artifact of the band width. Fixed seed for reproducibility.
    """
    n = len(day_table)
    if n == 0:
      return self.compute_rows(day_table, baseline_rows=None)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    opens = day_table["session_open"].to_numpy()
    shuffled_close = day_table["session_close"].to_numpy()[perm]

    tmp = day_table.copy()
    tmp["oc_move_pct"] = np.abs(shuffled_close - opens) / opens * 100.0

    return self.compute_rows(tmp, baseline_rows=None)


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
  range_percentage: float = 1.0,
) -> Path:
  """Load data and compute Open to Close Range for the daily timeframe.

  Writes the result JSON to results/ and returns its path.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  stat = OpenCloseRange(
    instrument=instrument, config=config, range_percentage=range_percentage
  )
  result = stat.compute(candles_df)

  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Open to Close Range stat")
  parser.add_argument("--instrument", default="NQ", help="Instrument name (default: NQ)")
  parser.add_argument("--data-path", default=None, help="Override parquet file path")
  parser.add_argument("--config-dir", default="config", help="Config directory (default: config)")
  parser.add_argument(
    "--range-percentage",
    type=float,
    default=1.0,
    help="Open-to-close band as a percent of the open (default: 1.0)",
  )
  args = parser.parse_args()

  output_path = run(
    instrument=args.instrument,
    config_dir=args.config_dir,
    data_path=args.data_path,
    range_percentage=args.range_percentage,
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
