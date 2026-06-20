"""Opening Stats (standard) — where does the daily session open land?

Measures the marginal 3-way distribution of where the RTH session open falls
relative to the PRIOR resolved session's intraday range (RTH high and low):

  above_high  — session_open >  prev_high  (open is above the prior high)
  inside_range — prev_low <= session_open <= prev_high  (open is inside the range)
  below_low   — session_open <  prev_low   (open is below the prior low)

Tie / boundary convention
  Opening exactly on the prior high OR the prior low is classified as
  ``inside_range`` — both boundaries are inclusive on the inside side. This is
  consistent with the ``inside_bars`` stat and the natural complement of the strict
  inequality used by ``outside_days``.

First-day exclusion (pending-sample discipline)
  The first resolved day has no prior resolved session, so ``prev_high`` and
  ``prev_low`` are NaN. It is excluded from every denominator.

Baseline
  Each day's ``open_location`` is reassigned uniformly at random across the three
  outcomes (``above_high``, ``inside_range``, ``below_low``) using
  ``np.random.default_rng(seed)``. The expected baseline probability is ≈ 1/3 per
  outcome — the null hypothesis is that the daily open is equally likely to land
  above, inside, or below the prior range.

Declared slices: ``weekday``, ``prev_candle``, ``close``.
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
  en="Opening Stats",
  fr="Statistiques d'ouverture",
)
_DEFINITION = I18nString(
  en="Where does the daily session open land relative to the prior session's high and low — above the prior high, inside the prior range, or below the prior low?",
  fr="Où se situe l'ouverture quotidienne par rapport au plus haut et au plus bas de la session précédente — au-dessus du plus haut, dans la fourchette ou en dessous du plus bas ?",
)
_LABELS = Labels(
  conditions={
    "open": I18nString(en="Session open", fr="Ouverture de session"),
  },
  outcomes={
    "above_high": I18nString(
      en="Above prior high", fr="Au-dessus du plus haut précédent"
    ),
    "inside_range": I18nString(
      en="Inside prior range", fr="Dans la fourchette précédente"
    ),
    "below_low": I18nString(
      en="Below prior low", fr="En dessous du plus bas précédent"
    ),
  },
)

# Single condition: all resolved countable days share it.
_CONDITION = "open"

# Ordered outcomes (order is preserved in the output).
_OUTCOMES: tuple[str, ...] = ("above_high", "inside_range", "below_low")

# Integer codes used by numpy for the random baseline draw.
_OUTCOME_CODES: dict[int, str] = {0: "above_high", 1: "inside_range", 2: "below_low"}


class OpeningStats(BaseStat):
  """Marginal 3-way distribution of where the daily open lands vs the prior range."""

  stat_name = "opening_stats"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = ("weekday", "prev_candle", "close")

  def __init__(
    self,
    instrument: str,
    config: InstrumentConfig,
    close_tolerance_min: int = 15,
  ) -> None:
    self.instrument = instrument
    self.timeframe = "daily"
    self.config = config
    self.close_tolerance_min = close_tolerance_min

    rth = config.sessions["rth"]
    self.rth_start_min: int = minute_of_day(rth.start)
    self.rth_end_min: int = minute_of_day(rth.end)

  # -------------------------------------------------------------------------
  # Day table construction
  # -------------------------------------------------------------------------
  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-session table with open-location classification.

    Each row corresponds to one resolved, countable trading day (the first
    resolved day has no prior session and is excluded), indexed by the
    normalized session date. Columns returned:

      session_open       — open of the RTH 09:30 bar.
      prev_high          — prior resolved session's RTH intraday high.
      prev_low           — prior resolved session's RTH intraday low.
      open_location      — one of ``above_high``, ``inside_range``, ``below_low``.
      session_green      — bool, ``session_close >= session_open`` (for ``close`` slicer).
      prev_session_green — prior session's color (for ``prev_candle`` slicer).

    Ties (open exactly on ``prev_high`` or ``prev_low``) count as ``inside_range``
    — both boundaries are inclusive on the inside.

    The first resolved day (NaN ``prev_high`` / ``prev_low``) is excluded
    (pending-sample discipline).
    """
    columns = [
      "session_open",
      "prev_high",
      "prev_low",
      "open_location",
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

    # Inner join: only resolved dates pass through.
    day = resolved.join(day_high, how="inner").join(day_low, how="inner")
    if day.empty:
      return empty

    # Session color (for close slicer).
    day["session_green"] = day["session_close"] >= day["session_open"]

    # Prior RESOLVED session values (shift over the resolved-only, sorted index,
    # so they skip over any excluded/early-close day).
    day["prev_high"] = day["day_high"].shift(1)
    day["prev_low"] = day["day_low"].shift(1)
    day["prev_session_green"] = day["session_green"].shift(1)

    # Exclude the first resolved day: its prev_high / prev_low are NaN
    # (pending-sample discipline).
    day = day[day["prev_high"].notna() & day["prev_low"].notna()].copy()
    if day.empty:
      return empty

    # Classify open_location.
    # Ties (open == prev_high or open == prev_low) count as inside_range.
    above = day["session_open"] > day["prev_high"]
    below = day["session_open"] < day["prev_low"]
    day["open_location"] = "inside_range"
    day.loc[above, "open_location"] = "above_high"
    day.loc[below, "open_location"] = "below_low"

    return day[columns]

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the three outcome rows for one (possibly sliced) day subset.

    Always emits all three rows (above_high, inside_range, below_low) even when
    total == 0. Merges ``baseline_prob`` / ``baseline_n`` from ``baseline_rows``
    if provided.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    total = len(day_table)

    rows: list[StatResultRow] = []
    for out_key in _OUTCOMES:
      count = int((day_table["open_location"] == out_key).sum()) if total > 0 else 0
      probability = count / total if total > 0 else 0.0
      bl = baseline_map.get((_CONDITION, out_key))
      rows.append(
        StatResultRow(
          condition=_CONDITION,
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
    """Random null baseline: uniform draw across the three outcomes.

    Each day's ``open_location`` is replaced by a draw from
    {above_high, inside_range, below_low} with equal probability 1/3.
    Expected ``baseline_prob`` ≈ 1/3 per outcome. Uses
    ``np.random.default_rng(seed)`` for deterministic output.
    """
    n = len(day_table)
    if n == 0:
      return self.compute_rows(day_table, baseline_rows=None)

    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 3, size=n)
    random_locations = pd.Series(
      [_OUTCOME_CODES[int(c)] for c in codes], index=day_table.index
    )

    tmp = day_table.copy()
    tmp["open_location"] = random_locations
    return self.compute_rows(tmp, baseline_rows=None)


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
) -> Path:
  """Load data and compute Opening Stats for the daily timeframe.

  Writes the result JSON to results/ and returns its path.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  stat = OpeningStats(instrument=instrument, config=config)
  result = stat.compute(candles_df)

  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Opening Stats")
  parser.add_argument("--instrument", default="NQ", help="Instrument name (default: NQ)")
  parser.add_argument("--data-path", default=None, help="Override parquet file path")
  parser.add_argument(
    "--config-dir", default="config", help="Config directory (default: config)"
  )
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
    print(f"\n  {tf}: {tf_data['total_samples']} resolved days | {tf_data['data_range']}")
    for row in tf_data["results"]:
      print(
        f"    {row['condition']} -> {row['outcome']}: "
        f"{row['probability']:.3f} (N={row['total']}, baseline={row['baseline_prob']:.3f})"
      )
    weekday = tf_data["slices"].get("weekday", {}).get("groups", {})
    if weekday:
      print("  Weekday slice:")
      for day_key, group in weekday.items():
        above = next(
          (r for r in group["results"] if r["outcome"] == "above_high"), None
        )
        inside = next(
          (r for r in group["results"] if r["outcome"] == "inside_range"), None
        )
        below = next(
          (r for r in group["results"] if r["outcome"] == "below_low"), None
        )
        if above and inside and below:
          print(
            f"    {day_key}: above={above['probability']:.3f} "
            f"inside={inside['probability']:.3f} "
            f"below={below['probability']:.3f} "
            f"(N={above['total']})"
          )
