"""Session Reversal Range stat — standard variant.

Measures how far each session moves **against its eventual direction** before it
closes — the adverse intraday excursion from the session open:

  - **Green** session (``session_close >= session_open``): the open-to-low move
    (``session_open - day_low``). The session finished up, so the downside dip
    below the open is the reversal.
  - **Red** session (``session_close < session_open``): the open-to-high move
    (``day_high - session_open``). The session finished down, so the upside pop
    above the open is the reversal.

Both excursions are non-negative because the RTH session open always lies within
the RTH ``[day_low, day_high]`` interval.

This is a **magnitude** stat (like ``volume_range_weekday``): every outcome
carries its metric in ``StatResultRow.value`` (and its random baseline in
``value_baseline``); the ``probability`` channel is left at ``0.0``. The reversal
range is reported as both an **average** and a **maximum**, each in **points**
and in **percent** of the session open:

  - ``mean_reversal``     — average reversal range in points.
  - ``mean_reversal_pct`` — average reversal range as a decimal of the open
                            (e.g. ``0.012`` = 1.2%).
  - ``max_reversal``      — maximum reversal range in points.
  - ``max_reversal_pct``  — maximum reversal range as a decimal of the open.

Every resolved session is countable — there is no warm-up window — so each row's
``count`` / ``total`` equals ``total_samples``.

Declared slices:

  - ``weekday`` — the "by weekday" breakdown.
  - ``close``   — the "by session color" breakdown (green / red), the central
                  green-vs-red split of this stat.
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
  StatResultRow,
  write_results,
)
from stats.config import InstrumentConfig, load_config, minute_of_day
from stats.utils.daily_candles import build_resolved_days

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="Session Reversal Range",
  fr="Amplitude de retournement de session",
)
_DEFINITION = I18nString(
  en="How far does a session move against its eventual direction (open-to-low for green sessions, open-to-high for red sessions) before it closes?",
  fr="De combien une session évolue-t-elle à contre-sens de sa direction finale (ouverture-bas pour les sessions vertes, ouverture-haut pour les rouges) avant de clôturer ?",
)
_LABELS = Labels(
  conditions={
    "session_reversal_range": I18nString(
      en="All resolved sessions",
      fr="Toutes les sessions résolues",
    ),
  },
  outcomes={
    "mean_reversal": I18nString(
      en="Average reversal range (points)",
      fr="Amplitude de retournement moyenne (points)",
    ),
    "mean_reversal_pct": I18nString(
      en="Average reversal range (%)",
      fr="Amplitude de retournement moyenne (%)",
    ),
    "max_reversal": I18nString(
      en="Maximum reversal range (points)",
      fr="Amplitude de retournement maximale (points)",
    ),
    "max_reversal_pct": I18nString(
      en="Maximum reversal range (%)",
      fr="Amplitude de retournement maximale (%)",
    ),
  },
)

# Single condition: every resolved session belongs to it; the declared slices do
# the per-weekday / per-color breakdowns.
_CONDITION = "session_reversal_range"

# Outcome keys with the day-table column they average and the aggregation, in
# display order. "mean" and "max" both reduce the same two metric columns.
_OUTCOMES: list[tuple[str, str, str]] = [
  ("mean_reversal", "reversal", "mean"),
  ("mean_reversal_pct", "reversal_pct", "mean"),
  ("max_reversal", "reversal", "max"),
  ("max_reversal_pct", "reversal_pct", "max"),
]


class SessionReversalRange(BaseStat):
  """Average and maximum adverse intraday excursion from the session open."""

  stat_name = "session_reversal_range"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = ("weekday", "close")

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
    """Build the per-session table with the directional reversal range.

    Columns returned:
      ``session_open``   — open of the RTH session-open bar (the pct denominator).
      ``down_excursion`` — open-to-low move ``session_open - day_low`` (points).
      ``up_excursion``   — open-to-high move ``day_high - session_open`` (points).
      ``session_green``  — session color (``session_close >= session_open``), read
                           by the ``close`` slicer and by the reversal selection.
      ``reversal``       — the adverse excursion in points: ``down_excursion`` for
                           green sessions, ``up_excursion`` for red sessions.
      ``reversal_pct``   — ``reversal / session_open`` (a decimal of the open).

    The resolved-days core (RTH filter, session open/close extraction, resolution
    filter, chronological sort) is shared via ``build_resolved_days``; the RTH
    intraday high/low are joined on per day. Every resolved session is countable:
    there is no warm-up window. The ``down_excursion`` / ``up_excursion``
    components are kept so the random baseline can pick the adverse side at
    random. The DatetimeIndex (normalized session date) is required by the
    ``weekday`` slicer.
    """
    columns = [
      "session_open",
      "down_excursion",
      "up_excursion",
      "session_green",
      "reversal",
      "reversal_pct",
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

    day["down_excursion"] = day["session_open"] - day["day_low"]
    day["up_excursion"] = day["day_high"] - day["session_open"]
    day["session_green"] = day["session_close"] >= day["session_open"]
    # Adverse side: open-to-low for green sessions, open-to-high for red.
    day["reversal"] = day["down_excursion"].where(
      day["session_green"], day["up_excursion"]
    )
    day["reversal_pct"] = day["reversal"] / day["session_open"]

    return day[columns]

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the magnitude rows over a (possibly sliced) day subset.

    Each row carries its average / maximum metric in ``value``; the
    ``probability`` channel is left at ``0.0``. Every row reports its sample size
    in ``count`` / ``total``. If ``baseline_rows`` is provided, the matching
    row's ``value`` is merged into each row's ``value_baseline``.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    total = len(day_table)

    rows: list[StatResultRow] = []
    for out_key, column, agg in _OUTCOMES:
      if total > 0:
        series = day_table[column].astype(float)
        value = float(series.mean() if agg == "mean" else series.max())
      else:
        value = 0.0
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
          value=value,
          value_baseline=bl.value if bl else None,
          agg=agg,
        )
      )
    return rows

  # -------------------------------------------------------------------------
  # Per-day sample classification
  # -------------------------------------------------------------------------
  def classify_samples(self, day_table: pd.DataFrame) -> list[SampleRow]:
    """Per-day SampleRows mirroring the four outcomes ``compute_rows`` aggregates.

    Every resolved session is countable under the single ``session_reversal_range``
    condition (no warm-up window), so each day yields exactly four SampleRows —
    one per outcome in ``_OUTCOMES`` — each carrying that day's raw metric
    (``reversal`` for ``mean_reversal`` / ``max_reversal``, ``reversal_pct`` for
    ``mean_reversal_pct`` / ``max_reversal_pct``) in ``value``. ``compute_rows``
    reduces these same per-day values with ``mean`` or ``max`` depending on the
    outcome; re-aggregating a date-filtered subset of these samples with the
    matching reducer reproduces the corresponding ``StatResultRow.value``.
    """
    if day_table.empty:
      return []

    samples: list[SampleRow] = []
    for ts, row in day_table.iterrows():
      date_str = ts.strftime("%Y-%m-%d")
      for out_key, column, _agg in _OUTCOMES:
        samples.append(
          SampleRow(
            date=date_str,
            condition=_CONDITION,
            outcome=out_key,
            value=float(row[column]),
          )
        )
    return samples

  # -------------------------------------------------------------------------
  # Baseline
  # -------------------------------------------------------------------------
  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random null baseline. Deterministic for a fixed seed.

    For each session the adverse side is chosen by a **fair coin** — the
    open-to-low or the open-to-high excursion, independent of the actual close
    direction — and the same averages / maxima are computed on that random-side
    reversal. This is the null hypothesis that the session's direction carries no
    information about which side is adverse.

    Expected baseline: the true adverse excursion (the side opposite the close)
    tends to be smaller than a randomly chosen side, so the actual mean reversal
    is **below** the baseline — confirming the directional containment is real
    rather than an artifact of intraday range. The coin flip randomizes within
    the overall table and within every slice (including the green-only / red-only
    ``close`` groups). Uses ``np.random.default_rng`` with a fixed seed.
    """
    n = len(day_table)
    if n == 0:
      return self.compute_rows(day_table, baseline_rows=None)

    rng = np.random.default_rng(seed)
    pick_down = rng.random(n) < 0.5

    down = day_table["down_excursion"].to_numpy()
    up = day_table["up_excursion"].to_numpy()
    reversal_bl = np.where(pick_down, down, up)

    tmp = day_table.copy()
    tmp["reversal"] = reversal_bl
    tmp["reversal_pct"] = reversal_bl / day_table["session_open"].to_numpy()

    return self.compute_rows(tmp, baseline_rows=None)


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
) -> Path:
  """Load data and compute Session Reversal Range for the daily timeframe.

  Writes the result JSON to results/ and returns its path.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  stat = SessionReversalRange(instrument=instrument, config=config)
  result = stat.compute(candles_df)

  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Session Reversal Range stat")
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
    mean_pts = rows["mean_reversal"]["value"]
    mean_pct = rows["mean_reversal_pct"]["value"]
    max_pts = rows["max_reversal"]["value"]
    bl_pts = rows["mean_reversal"]["value_baseline"]
    print(
      f"    avg reversal={mean_pts:.2f} pts ({mean_pct:.4%}) "
      f"| max reversal={max_pts:.2f} pts | baseline avg={bl_pts:.2f} pts"
    )
