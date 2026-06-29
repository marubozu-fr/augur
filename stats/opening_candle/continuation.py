"""Opening Candle Continuation stat.

Measures: after the first N-minute candle of the RTH session, how often does
the session close in the same direction as that opening candle?
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stats.base import (
  BaseStat,
  I18nString,
  Labels,
  SizeBucket,
  SliceGroup,
  Slicer,
  StatResultRow,
  StatRunResult,
  TimeframeResult,
  write_results,
)
from stats.config import InstrumentConfig, load_config, minute_of_day

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="Opening Candle Continuation",
  fr="Continuation de la bougie d'ouverture",
)
_DEFINITION = I18nString(
  en="After the first N-minute candle of the NY session, how often does the session close in the same direction?",
  fr="Après la première bougie de N minutes de la session NY, à quelle fréquence la session clôture-t-elle dans la même direction ?",
)
_LABELS = Labels(
  conditions={
    "green_open": I18nString(en="Green opening candle", fr="Bougie d'ouverture verte"),
    "red_open": I18nString(en="Red opening candle", fr="Bougie d'ouverture rouge"),
  },
  outcomes={
    "green_close": I18nString(en="Session closes green", fr="Session clôture en vert"),
    "red_close": I18nString(en="Session closes red", fr="Session clôture en rouge"),
  },
)

# ---------------------------------------------------------------------------
# Timeframe string → integer minutes
# ---------------------------------------------------------------------------
_TF_MINUTES: dict[str, int] = {
  "15min": 15,
  "30min": 30,
  "1h": 60,
}


class _CloseLocation(Slicer):
  """Split by where the session closes relative to the opening candle range.

  Unlike the generic ``Close`` slicer (which merely splits days by session
  color), this is a distinct metric: it groups days by how far the session
  travels from the opening candle, using the candle's wick extremes as the
  reference. Three ordered groups:

    above  — session_close >  opening_high (closes beyond the candle's high)
    inside — opening_low <= session_close <= opening_high (closes within it)
    below  — session_close <  opening_low (closes beyond the candle's low)

  Reads the ``close_location`` column produced by ``build_day_table``.
  """

  name = "close"

  _GROUPS: tuple[tuple[str, I18nString], ...] = (
    (
      "above",
      I18nString(
        en="Closes above opening candle",
        fr="Clôture au-dessus de la bougie d'ouverture",
      ),
    ),
    (
      "inside",
      I18nString(
        en="Closes inside opening candle",
        fr="Clôture dans la bougie d'ouverture",
      ),
    ),
    (
      "below",
      I18nString(
        en="Closes below opening candle",
        fr="Clôture en dessous de la bougie d'ouverture",
      ),
    ),
  )

  def dimension_label(self) -> I18nString:
    return I18nString(
      en="Session close vs opening candle",
      fr="Clôture de session vs bougie d'ouverture",
    )

  def split(self, day_table: pd.DataFrame) -> list[SliceGroup]:
    if len(day_table) == 0 or "close_location" not in day_table.columns:
      return []
    col = day_table["close_location"]
    groups: list[SliceGroup] = []
    for key, label in self._GROUPS:
      mask = pd.Series(col == key, index=day_table.index)
      if bool(mask.any()):
        groups.append(SliceGroup(key=key, label=label, mask=mask))
    return groups


class OpeningCandleContinuation(BaseStat):
  """Conditional probability of session direction given opening candle direction."""

  stat_name = "opening_candle_continuation"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = (
    "weekday",
    _CloseLocation(),
    SizeBucket(column="opening_body", preset="quartiles", name="size"),
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
    self.tf_minutes = _TF_MINUTES[timeframe]
    self.close_tolerance_min = close_tolerance_min

    rth = config.sessions["rth"]
    self.rth_start_min: int = minute_of_day(rth.start)
    self.rth_end_min: int = minute_of_day(rth.end)

    # Opening candle aligned to the timeframe grid: the candle that CONTAINS the
    # RTH open, not necessarily one that starts on it. Chart timeframes are
    # clock-aligned, so on a 09:30 open the opening 1h candle is 09:00–10:00
    # (floor(570 / 60) * 60 = 540), while 15min and 30min still start at 09:30.
    self.candle_open_min: int = (self.rth_start_min // self.tf_minutes) * self.tf_minutes

  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-day summary table from 1-min OHLCV data (vectorized).

    Each row in the output corresponds to one resolved trading day with columns:
      date, session_open, session_close, opening_open, opening_close,
      opening_green, session_green

    A day is "resolved" (confirmed) if:
      - It has a bar exactly at rth_start_min (clean session open)
      - It has a bar exactly at candle_open_min (clean opening-candle open)
      - Its last RTH bar is at or after rth_end_min - close_tolerance_min
        (this excludes early-close days and the final incomplete day in the data)
    Unresolved days are excluded from all counts (pending sample rule).
    """
    if candles_df.empty:
      return pd.DataFrame(
        columns=[
          "session_open", "session_close", "last_minute",
          "opening_open", "opening_close", "opening_high", "opening_low",
          "opening_green", "session_green", "opening_body", "close_location",
        ]
      )
    df = candles_df.copy()

    # Compute minute-of-day and date for each bar
    df["mod"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["date"] = df["timestamp"].dt.normalize()

    # Keep only RTH bars: [rth_start_min, rth_end_min)
    rth_mask = (df["mod"] >= self.rth_start_min) & (df["mod"] < self.rth_end_min)
    rth = df[rth_mask].copy()

    # --- Session open and close (per day) ---
    # session_open = open of the bar at exactly rth_start_min
    open_bars = rth[rth["mod"] == self.rth_start_min].set_index("date")["open"].rename("session_open")

    # session_close = close of the last RTH bar for that day
    last_bars = (
      rth.loc[rth.groupby("date")["mod"].idxmax(), ["date", "close", "mod"]]
      .set_index("date")
      .rename(columns={"close": "session_close", "mod": "last_minute"})
    )

    # --- Opening candle (per day) ---
    # The opening candle window is [candle_open_min, candle_open_min + tf).
    # For 1h this starts at 09:00, i.e. BEFORE rth_start_min, so it is computed
    # from the full frame (df) rather than the RTH-only subset.
    oc_start_min = self.candle_open_min
    oc_end_min = self.candle_open_min + self.tf_minutes
    oc_mask = (df["mod"] >= oc_start_min) & (df["mod"] < oc_end_min)
    oc_bars = df[oc_mask].copy()

    # opening_open = open of the first bar (must be at candle_open_min)
    oc_first = oc_bars[oc_bars["mod"] == oc_start_min].set_index("date")["open"].rename(
      "opening_open"
    )
    # opening_close = close of the last bar of the opening candle window
    oc_last = (
      oc_bars.loc[oc_bars.groupby("date")["mod"].idxmax(), ["date", "close"]]
      .set_index("date")["close"]
      .rename("opening_close")
    )
    # opening_high / opening_low = wick extremes over the opening candle window
    oc_high = oc_bars.groupby("date")["high"].max().rename("opening_high")
    oc_low = oc_bars.groupby("date")["low"].min().rename("opening_low")

    # --- Join everything ---
    day = pd.concat(
      [open_bars, last_bars, oc_first, oc_last, oc_high, oc_low], axis=1, sort=False
    )

    # Resolution filter: must have clean session open AND sufficient close coverage.
    # Days missing the open bar will have NaN in session_open / opening_open.
    resolved_min = self.rth_end_min - self.close_tolerance_min
    day = day[
      day["session_open"].notna()
      & day["opening_open"].notna()
      & (day["last_minute"] >= resolved_min)
    ].copy()

    # Direction flags
    day["opening_green"] = day["opening_close"] >= day["opening_open"]
    day["session_green"] = day["session_close"] >= day["session_open"]

    # Slice metrics
    # opening_body = absolute opening-candle body size (drives the `size` slice).
    day["opening_body"] = (day["opening_close"] - day["opening_open"]).abs()
    # close_location = session close relative to the opening candle range
    # (drives the `close` slice). Wick extremes are the reference; boundaries are
    # inclusive on the inside.
    day["close_location"] = np.select(
      [day["session_close"] > day["opening_high"], day["session_close"] < day["opening_low"]],
      ["above", "below"],
      default="inside",
    )

    return day

  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the four condition/outcome StatResultRows.

    If baseline_rows is provided, merges baseline_prob/baseline_n from it.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    rows: list[StatResultRow] = []
    conditions = [
      ("green_open", True),
      ("red_open", False),
    ]
    outcomes = [
      ("green_close", True),
      ("red_close", False),
    ]

    for cond_key, cond_green in conditions:
      cond_mask = day_table["opening_green"] == cond_green
      total = int(cond_mask.sum())

      for out_key, out_green in outcomes:
        out_mask = day_table["session_green"] == out_green
        count = int((cond_mask & out_mask).sum())
        probability = count / total if total > 0 else 0.0

        # baseline_rows come from baseline(): their .probability field holds
        # the randomized rate (since they were computed with no nested baseline).
        # We read that as the baseline_prob for the real result rows.
        bl = baseline_map.get((cond_key, out_key))
        baseline_prob = bl.probability if bl else 0.0
        baseline_n = bl.total if bl else 0

        rows.append(
          StatResultRow(
            condition=cond_key,
            outcome=out_key,
            count=count,
            total=total,
            probability=probability,
            baseline_prob=baseline_prob,
            baseline_n=baseline_n,
          )
        )

    return rows

  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random baseline: keep actual opening direction, randomize session direction (p=0.5).

    Uses a fixed seed for deterministic output. Expected baseline_prob ≈ 0.5.
    Operates directly on the day table so the framework can compute a baseline
    per slice group as well as overall.
    """
    rng = np.random.default_rng(seed)
    n = len(day_table)
    # Randomly assign session direction (True=green, False=red) with p=0.5
    random_session_green = rng.integers(0, 2, size=n).astype(bool)

    # Build a temporary frame with randomized session direction
    tmp = day_table.copy()
    tmp["session_green"] = random_session_green

    # Compute rows from the randomized frame (no nested baseline call)
    return self.compute_rows(tmp, baseline_rows=None)


def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
) -> Path:
  """Load data and compute Opening Candle Continuation for all three timeframes.

  Merges per-timeframe TimeframeResults into a single StatRunResult and writes
  the consolidated JSON to results/.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  timeframes = ["15min", "30min", "1h"]
  merged_tf: dict[str, TimeframeResult] = {}
  labels = _LABELS.model_copy(deep=True)

  for tf in timeframes:
    stat = OpeningCandleContinuation(instrument=instrument, timeframe=tf, config=config)
    result = stat.compute(candles_df)
    merged_tf[tf] = result.instruments[instrument][tf]
    # Merge the framework-enriched slice dimension labels across timeframes so
    # no timeframe's dimensions overwrite an earlier one's.
    labels.dimensions.update(result.labels.dimensions)

  final_result = StatRunResult(
    stat_name="opening_candle_continuation",
    title=_TITLE,
    definition=_DEFINITION,
    labels=labels,
    instruments={instrument: merged_tf},
  )

  return write_results(final_result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Opening Candle Continuation stat")
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

  # Print a brief summary of results
  import json
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
