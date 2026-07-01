"""FOMC Intraday stat.

Measures intraday performance on FOMC decision days, grouped by 15-minute RTH
intervals. Days are split into positive/negative reaction groups based on the
2pm ET (14:00) interval direction — the 15-minute window that captures the
immediate market response to the Fed's rate announcement.

Each interval reports three metrics:
  - Average % change  (interval_close - interval_open) / interval_open
  - Average $ change   interval_close - interval_open
  - Average volume     sum of 1-minute bar volumes in [t, t+15)

Where:
  interval_open  = open of the bar at exactly minute t
  interval_close = close of the last bar in [t, t+15)

Only FOMC dates with a resolved RTH session are included. Per-interval values
are NaN when the interval lacks a bar at exactly its start minute (pending-sample
discipline applied at the interval level, not the day level). The 2pm reaction
flag is NaN when the 14:00 bar is absent; such days stay in the overall results
but are excluded from both reaction groups by the slicer.

total_samples = number of qualifying FOMC decision days.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from stats.base import (
  BaseStat,
  I18nString,
  Labels,
  SampleRow,
  Slicer,
  SliceGroup,
  StatResultRow,
  write_results,
)
from stats.config import InstrumentConfig, load_config, minute_of_day
from stats.event_performance_base import _DEFAULT_CALENDAR_PATH
from stats.fomc_performance.standard import load_fomc_release_dates
from stats.utils.daily_candles import build_resolved_days

# ---------------------------------------------------------------------------
# i18n content (static, class-level)
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="FOMC Intraday",
  fr="FOMC Intrajournalier",
)
_DEFINITION = I18nString(
  en=(
    "On FOMC decision days, how does price perform within each 15-minute RTH "
    "interval? Results are split by the direction of the 2pm ET reaction window "
    "(the bar that captures the immediate response to the Fed's rate decision)."
  ),
  fr=(
    "Les jours de décision du FOMC, comment le prix évolue-t-il dans chaque "
    "intervalle de 15 minutes de la session RTH ? Les résultats sont divisés selon "
    "la direction de la fenêtre de réaction de 14h ET (la bougie capturant la "
    "réponse immédiate à la décision de taux de la Fed)."
  ),
)

# FOMC announcement is conventionally at 14:00 ET (2pm).
_REACTION_TIME = "14:00"

# Metric tuple: (outcome key, day-table column prefix).
_METRICS: tuple[tuple[str, str], ...] = (
  ("pct_change", "pct_"),
  ("dollar_change", "dollar_"),
  ("volume", "vol_"),
)

# Conditions shown in the CLI summary (open, announcement, close).
_SUMMARY_CONDITIONS = {"i0930", "i1400", "i1600"}


# ---------------------------------------------------------------------------
# Reaction slicer
# ---------------------------------------------------------------------------
class _ReactionSplit(Slicer):
  """Split FOMC days by the 2pm ET interval direction.

  Reads the ``reaction_positive`` column (float: 1.0 = up, 0.0 = down/flat,
  NaN = missing 14:00 bar). Days with NaN are excluded from both groups.
  """

  name = "reaction"

  def dimension_label(self) -> I18nString:
    return I18nString(en="2pm ET reaction", fr="Réaction 14h ET")

  def split(self, day_table: pd.DataFrame) -> list[SliceGroup]:
    if len(day_table) == 0 or "reaction_positive" not in day_table.columns:
      return []
    col = day_table["reaction_positive"]
    present = col.notna()
    groups: list[SliceGroup] = []
    specs: list[tuple[str, I18nString, pd.Series]] = [
      (
        "positive",
        I18nString(en="Positive reaction", fr="Réaction positive"),
        present & col.fillna(False).astype(bool),
      ),
      (
        "negative",
        I18nString(en="Negative/flat reaction", fr="Réaction négative/plate"),
        present & ~col.fillna(True).astype(bool),
      ),
    ]
    for key, label, mask in specs:
      if bool(mask.any()):
        groups.append(SliceGroup(key=key, label=label, mask=mask))
    return groups


# ---------------------------------------------------------------------------
# FOMCIntraday stat
# ---------------------------------------------------------------------------
class FOMCIntraday(BaseStat):
  """Intraday performance on FOMC decision days by 15-minute RTH interval."""

  stat_name = "fomc_intraday"
  title = _TITLE
  definition = _DEFINITION
  # labels is set dynamically in __init__ (instance attr shadows class attr).
  slices = (_ReactionSplit(),)

  def __init__(
    self,
    instrument: str,
    config: InstrumentConfig,
    event_dates: Iterable[pd.Timestamp],
    close_tolerance_min: int = 15,
  ) -> None:
    self.instrument = instrument
    self.timeframe = "daily"
    self.config = config
    self.close_tolerance_min = close_tolerance_min

    rth = config.sessions["rth"]
    self.rth_start_min: int = minute_of_day(rth.start)
    self.rth_end_min: int = minute_of_day(rth.end)

    # Normalize to a set of midnight, tz-naive Timestamps for date alignment.
    self.event_dates: set[pd.Timestamp] = {
      pd.Timestamp(d).normalize().tz_localize(None) for d in event_dates
    }

    # 15-minute interval grid: [rth_start_min, rth_end_min) in steps of 15.
    self._interval_starts: list[int] = list(
      range(self.rth_start_min, self.rth_end_min, 15)
    )
    self._interval_keys: list[str] = [
      f"i{t // 60:02d}{t % 60:02d}" for t in self._interval_starts
    ]

    # Minute of the 2pm ET bar (FOMC announcement).
    self._reaction_min: int = minute_of_day(_REACTION_TIME)

    # Dynamic labels: one condition per interval (e.g. "i0930" -> "09:30–09:45").
    conditions: dict[str, I18nString] = {}
    for t in self._interval_starts:
      key = f"i{t // 60:02d}{t % 60:02d}"
      t_end = t + 15
      hh, mm = divmod(t, 60)
      hh_end, mm_end = divmod(t_end, 60)
      time_range = f"{hh:02d}:{mm:02d}–{hh_end:02d}:{mm_end:02d}"
      conditions[key] = I18nString(en=time_range, fr=time_range)

    self.labels = Labels(
      conditions=conditions,
      outcomes={
        "pct_change": I18nString(
          en="Average % change", fr="Variation % moyenne"
        ),
        "dollar_change": I18nString(
          en="Average $ change", fr="Variation $ moyenne"
        ),
        "volume": I18nString(en="Average volume", fr="Volume moyen"),
      },
    )

  # ---------------------------------------------------------------------------
  # Day table
  # ---------------------------------------------------------------------------
  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-FOMC-date table with per-interval metrics.

    Index = qualifying FOMC date (tz-naive normalized Timestamps).
    Columns (per interval key, in grid order): pct_{key}, dollar_{key},
    vol_{key}; plus ``reaction_positive``.

    A date qualifies iff it is an RTH-resolved session AND an FOMC event date.
    Per-interval values are NaN when the interval has no bar at exactly its
    start minute (pending-sample discipline at the interval level).

    ``reaction_positive``:
      1.0 = 2pm interval strictly up (close > open)
      0.0 = flat or down
      NaN = no bar at exactly the reaction minute, or no close in window

    Returns an empty DataFrame (with all expected columns) when no dates qualify.
    """
    columns: list[str] = []
    for key in self._interval_keys:
      columns += [f"pct_{key}", f"dollar_{key}", f"vol_{key}"]
    columns.append("reaction_positive")
    empty = pd.DataFrame(columns=columns)

    if not self.event_dates:
      return empty

    resolved = build_resolved_days(
      candles_df, self.rth_start_min, self.rth_end_min, self.close_tolerance_min
    )
    if resolved.empty:
      return empty

    resolved = resolved.sort_index()
    if resolved.index.tz is not None:
      resolved.index = resolved.index.tz_localize(None)

    # Prepare 1-minute candles: add mod (minute of day) and tz-naive date.
    df = candles_df.copy()
    df["mod"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    ts_norm = df["timestamp"].dt.normalize()
    df["date"] = (
      ts_norm.dt.tz_localize(None) if ts_norm.dt.tz is not None else ts_norm
    )

    # RTH bars only: [rth_start_min, rth_end_min).
    rth_mask = (df["mod"] >= self.rth_start_min) & (df["mod"] < self.rth_end_min)
    rth_bars = df[rth_mask].copy()

    if rth_bars.empty:
      return empty

    # Qualifying dates: resolved RTH session AND FOMC event date.
    event_index = pd.DatetimeIndex(sorted(self.event_dates))
    resolved_event = resolved[resolved.index.isin(event_index)]
    if resolved_event.empty:
      return empty

    qualifying_dates = resolved_event.index  # DatetimeIndex, tz-naive
    fomc_rth = rth_bars[rth_bars["date"].isin(qualifying_dates)].copy()

    if fomc_rth.empty:
      return empty

    # Build per-interval series (vectorized over all qualifying dates).
    series_dict: dict[str, pd.Series] = {}

    for key, t in zip(self._interval_keys, self._interval_starts):
      t_end = t + 15

      # Open bar: bar at exactly minute t (interval_open).
      open_s = (
        fomc_rth[fomc_rth["mod"] == t]
        .groupby("date")["open"]
        .first()
        .reindex(qualifying_dates)
      )

      # Window bars: [t, t+15).
      win = fomc_rth[(fomc_rth["mod"] >= t) & (fomc_rth["mod"] < t_end)]

      if win.empty:
        nan_s = pd.Series(np.nan, index=qualifying_dates)
        series_dict[f"pct_{key}"] = nan_s.copy()
        series_dict[f"dollar_{key}"] = nan_s.copy()
        series_dict[f"vol_{key}"] = nan_s.copy()
        continue

      # Close of last bar in window (interval_close).
      last_idx = win.groupby("date")["mod"].idxmax()
      close_s = (
        win.loc[last_idx].set_index("date")["close"].reindex(qualifying_dates)
      )

      # Volume sum in window.
      vol_s = win.groupby("date")["volume"].sum().reindex(qualifying_dates)

      # pct and dollar: NaN where interval open bar is missing.
      diff = close_s - open_s
      valid = open_s.notna()

      series_dict[f"pct_{key}"] = (diff / open_s).where(valid, other=np.nan)
      series_dict[f"dollar_{key}"] = diff.where(valid, other=np.nan)
      # Volume is NaN when the open bar is missing (consistent pending discipline).
      series_dict[f"vol_{key}"] = vol_s.where(valid, other=np.nan)

    # reaction_positive: direction of the 2pm ET interval (independent of grid).
    t_2pm = self._reaction_min
    rp_open_s = (
      fomc_rth[fomc_rth["mod"] == t_2pm]
      .groupby("date")["open"]
      .first()
      .reindex(qualifying_dates)
    )
    rp_win = fomc_rth[
      (fomc_rth["mod"] >= t_2pm) & (fomc_rth["mod"] < t_2pm + 15)
    ]

    if rp_win.empty:
      series_dict["reaction_positive"] = pd.Series(np.nan, index=qualifying_dates)
    else:
      last_idx_rp = rp_win.groupby("date")["mod"].idxmax()
      rp_close_s = (
        rp_win.loc[last_idx_rp].set_index("date")["close"].reindex(qualifying_dates)
      )
      diff_rp = rp_close_s - rp_open_s
      # True (1.0) = strictly up, False (0.0) = flat/down, NaN = missing bars.
      # (NaN > 0) evaluates to False in pandas; .where restores NaN for those entries.
      series_dict["reaction_positive"] = (
        (diff_rp > 0).astype(float).where(diff_rp.notna(), other=np.nan)
      )

    result = pd.DataFrame(series_dict, index=qualifying_dates)
    return result[columns]

  # ---------------------------------------------------------------------------
  # Core computation
  # ---------------------------------------------------------------------------
  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute per-interval metric rows (27 intervals × 3 metrics = 81 rows for NQ).

    All rows are always emitted (even when empty) for a deterministic shape.
    For pct_change / dollar_change: count = number of non-negative observations;
    for volume: count = total (all observations have volume data).
    value = mean of non-NaN observations (signed return or average volume).

    Baseline merging:
      - pct_change / dollar_change: value_baseline = bl.value (signed baseline mean)
      - volume: value_baseline = None (no directional baseline)
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    rows: list[StatResultRow] = []
    for key in self._interval_keys:
      for outcome, col_prefix in _METRICS:
        col = f"{col_prefix}{key}"
        if len(day_table) > 0 and col in day_table.columns:
          vals = day_table[col].dropna()
        else:
          vals = pd.Series([], dtype=float)

        total = len(vals)

        if outcome == "volume":
          count = total
        else:
          count = int((vals >= 0).sum())

        probability = count / total if total > 0 else 0.0
        value = float(vals.mean()) if total > 0 else 0.0

        bl = baseline_map.get((key, outcome))
        rows.append(
          StatResultRow(
            condition=key,
            outcome=outcome,
            count=count,
            total=total,
            probability=probability,
            baseline_prob=bl.probability if bl else 0.0,
            baseline_n=bl.total if bl else 0,
            value=value,
            value_baseline=(
              (bl.value if bl else None) if outcome != "volume" else None
            ),
            agg="mean",
          )
        )
    return rows

  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random baseline: sign-flip pct and dollar columns; leave volume unchanged.

    Magnitudes are held fixed and only the sign of each non-NaN observation is
    randomized (p=0.5). NaN (pending) entries stay NaN. vol_ columns are left
    as-is (no directional baseline for volume). Deterministic for a fixed seed.
    """
    rng = np.random.default_rng(seed)
    tmp = day_table.copy()

    for key in self._interval_keys:
      for col_prefix in ("pct_", "dollar_"):
        col = f"{col_prefix}{key}"
        if col not in tmp.columns:
          continue
        values = tmp[col].to_numpy(dtype=float).copy()
        mask = ~np.isnan(values)
        if mask.any():
          k = int(mask.sum())
          signs = rng.integers(0, 2, size=k) * 2 - 1  # ±1
          values[mask] = signs * np.abs(values[mask])
        tmp[col] = values

    return self.compute_rows(tmp, baseline_rows=None)

  def classify_samples(self, day_table: pd.DataFrame) -> list[SampleRow]:
    """One SampleRow per (day, interval, metric) with a non-NaN observation.

    ``condition`` is the interval key (e.g. "i0930") and ``outcome`` is the
    metric key (pct_change / dollar_change / volume), matching ``compute_rows``.
    """
    if day_table.empty:
      return []

    samples: list[SampleRow] = []
    for ts, row in day_table.iterrows():
      date = ts.strftime("%Y-%m-%d")
      for key in self._interval_keys:
        for outcome, col_prefix in _METRICS:
          val = row[f"{col_prefix}{key}"]
          if pd.notna(val):
            samples.append(SampleRow(date=date, condition=key, outcome=outcome, value=float(val)))
    return samples


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
  calendar_path: str | Path = _DEFAULT_CALENDAR_PATH,
) -> Path:
  """Load data and the FOMC calendar, compute FOMC Intraday, write the result."""
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)
  event_dates = load_fomc_release_dates(calendar_path)

  stat = FOMCIntraday(
    instrument=instrument,
    config=config,
    event_dates=event_dates,
  )
  result = stat.compute(candles_df)
  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute FOMC Intraday stat")
  parser.add_argument(
    "--instrument", default="NQ", help="Instrument name (default: NQ)"
  )
  parser.add_argument(
    "--data-path", default=None, help="Override parquet file path"
  )
  parser.add_argument(
    "--config-dir", default="config", help="Config directory (default: config)"
  )
  parser.add_argument(
    "--calendar-path",
    default=str(_DEFAULT_CALENDAR_PATH),
    help=f"Economic calendar CSV path (default: {_DEFAULT_CALENDAR_PATH})",
  )
  args = parser.parse_args()

  output_path = run(
    instrument=args.instrument,
    config_dir=args.config_dir,
    data_path=args.data_path,
    calendar_path=args.calendar_path,
  )

  print(f"Written: {output_path}")

  with open(output_path, encoding="utf-8") as f:
    data = json.load(f)

  instrument_data = data["instruments"][args.instrument]
  for tf, tf_data in instrument_data.items():
    print(
      f"\n  {tf}: {tf_data['total_samples']} qualifying FOMC days"
      f" | {tf_data['data_range']}"
    )

    # Overall: show a few key interval rows (open, announcement, close).
    print("  Overall (pct_change):")
    for row in tf_data["results"]:
      if row["condition"] in _SUMMARY_CONDITIONS and row["outcome"] == "pct_change":
        print(
          f"    {row['condition']}: avg={row['value']:+.4%}"
          f"  up={row['probability']:.3f}"
          f"  N={row['total']}"
          f"  baseline_up={row['baseline_prob']:.3f}"
        )

    # Reaction slices.
    reaction = tf_data.get("slices", {}).get("reaction", {})
    for grp_key, grp in reaction.get("groups", {}).items():
      print(f"  {grp['label']['en']} (N={grp['total_samples']}, pct_change):")
      for row in grp["results"]:
        if row["condition"] in _SUMMARY_CONDITIONS and row["outcome"] == "pct_change":
          print(
            f"    {row['condition']}: avg={row['value']:+.4%}"
            f"  up={row['probability']:.3f}"
            f"  N={row['total']}"
          )
