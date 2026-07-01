"""Intraday Range Window stat.

The high-low range over a single, configurable intraday window (e.g. 09:30-10:30)
measured each trading day, then summarized across days. For both the **absolute**
range (``high - low``, in points) and the **percentage** range
(``(high - low) / window_open``) the stat reports four aggregates: average,
maximum, minimum, and median. It shows how wide a given time-of-day window tends
to travel, and how that varies day to day.

This is a **magnitude** stat: every outcome carries its metric in
``StatResultRow.value`` (and its random baseline in ``value_baseline``); the
ordinary ``probability`` channel is left at ``0.0`` for every row.

There is a single **condition**: the measured window, keyed ``HHMM-HHMM`` (e.g.
``0930-1030``). The eight aggregates are the **outcomes**:
  - ``range_avg`` / ``range_max`` / ``range_min`` / ``range_median`` — the
    average / maximum / minimum / median of the daily absolute range, in points.
  - ``range_pct_avg`` / ``range_pct_max`` / ``range_pct_min`` /
    ``range_pct_median`` — the same four aggregates of the daily percentage range
    (a decimal, e.g. ``0.004`` = 0.4%), where ``window_open`` is the open of the
    first bar in the window that day.

Only resolved days (clean session open + sufficient close coverage) participate;
early-close days and the final incomplete day are excluded as pending samples by
``build_resolved_days``. The shared ``Weekday`` slicer provides the "by weekday"
variant.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

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
  en="Intraday Range Window",
  fr="Amplitude d'une fenêtre intraday",
)
_DEFINITION = I18nString(
  en="The high-low range over a configurable intraday window measured each day, summarized across days. Reports the average, maximum, minimum, and median of both the absolute range (points) and the percentage range.",
  fr="L'amplitude haut-bas sur une fenêtre intraday configurable mesurée chaque jour, résumée sur l'ensemble des jours. Donne la moyenne, le maximum, le minimum et la médiane de l'amplitude en points et de l'amplitude en pourcentage.",
)

# Aggregate functions over the per-day metric values, keyed by aggregate suffix.
_AGG = {
  "avg": np.mean,
  "max": np.max,
  "min": np.min,
  "median": np.median,
}

_RANGE_LABELS: dict[str, I18nString] = {
  "avg": I18nString(en="Average range (points)", fr="Amplitude moyenne (points)"),
  "max": I18nString(en="Maximum range (points)", fr="Amplitude maximale (points)"),
  "min": I18nString(en="Minimum range (points)", fr="Amplitude minimale (points)"),
  "median": I18nString(en="Median range (points)", fr="Amplitude médiane (points)"),
}
_PCT_LABELS: dict[str, I18nString] = {
  "avg": I18nString(en="Average range (%)", fr="Amplitude moyenne (%)"),
  "max": I18nString(en="Maximum range (%)", fr="Amplitude maximale (%)"),
  "min": I18nString(en="Minimum range (%)", fr="Amplitude minimale (%)"),
  "median": I18nString(en="Median range (%)", fr="Amplitude médiane (%)"),
}

# Outcome rows, in display order: (outcome_key, metric_column, aggregate, label).
# ``metric_column`` selects which per-day series the aggregate is applied to.
_OUTCOMES: list[tuple[str, str, str, I18nString]] = [
  *[(f"range_{agg}", "range", agg, _RANGE_LABELS[agg]) for agg in _AGG],
  *[(f"range_pct_{agg}", "range_pct", agg, _PCT_LABELS[agg]) for agg in _AGG],
]


class IntradayRangeWindow(BaseStat):
  """Average / max / min / median of the daily high-low range over one window.

  One instance covers a single ``[start_window, end_window]`` window. The
  ``Weekday`` slicer re-runs the core computation over each per-weekday subset.
  """

  stat_name = "intraday_range_window"
  title = _TITLE
  definition = _DEFINITION
  slices = ("weekday",)

  def __init__(
    self,
    instrument: str,
    config: InstrumentConfig,
    start_window: str = "09:30",
    end_window: str = "10:30",
    close_tolerance_min: int = 15,
  ) -> None:
    self.instrument = instrument
    self.config = config
    self.close_tolerance_min = close_tolerance_min

    rth = config.sessions["rth"]
    self.rth_start_min: int = minute_of_day(rth.start)
    self.rth_end_min: int = minute_of_day(rth.end)

    self.start_min: int = minute_of_day(start_window)
    self.end_min: int = minute_of_day(end_window)
    if not (self.rth_start_min <= self.start_min < self.end_min <= self.rth_end_min):
      raise ValueError(
        f"Window {start_window}-{end_window} must satisfy "
        f"{rth.start} <= start < end <= {rth.end}."
      )

    # Grid of RTH bar minutes [rth_start_min, rth_end_min); the last bar sits one
    # minute before rth_end_min (e.g. 16:14 for an RTH ending at 16:15).
    self._grid_len: int = self.rth_end_min - self.rth_start_min
    # The window's last bar is clamped to the last RTH bar, so an end of exactly
    # rth_end_min (no bar there) still measures through the final RTH bar.
    last_bar = min(self.end_min, self.rth_end_min - 1)
    self._win_len: int = last_bar - self.start_min + 1
    self._win_start_idx: int = self.start_min - self.rth_start_min
    self._n_cand: int = self._grid_len - self._win_len + 1

    sh, sm = divmod(self.start_min, 60)
    eh, em = divmod(self.end_min, 60)
    self.window_key: str = f"{sh:02d}{sm:02d}-{eh:02d}{em:02d}"
    self.timeframe = self.window_key
    window_label = f"{sh:02d}:{sm:02d}–{eh:02d}:{em:02d}"

    self.labels = Labels(
      conditions={self.window_key: I18nString(en=window_label, fr=window_label)},
      outcomes={out_key: label for out_key, _, _, label in _OUTCOMES},
    )

  # -------------------------------------------------------------------------
  # Day table construction
  # -------------------------------------------------------------------------
  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-day table of the window's range, plus baseline candidates.

    Each row is one resolved trading day, indexed by the normalized session date,
    with columns:
      - ``range``      — the window's ``high - low`` in points (NaN if invalid).
      - ``range_pct``  — the window's ``range / window_open`` (NaN if invalid).
      - ``_cand_range``, ``_cand_pct`` — per-day 1-D arrays of the range / pct of
        every same-length window that fits in the RTH session, used by the random
        baseline. The two arrays are paired (same candidate index) and contain
        only candidates where both metrics are finite.

    Steps:
    1. Build the resolved-days index via ``build_resolved_days``.
    2. Pivot RTH bars onto a (day x minute) grid for high / low / open.
    3. Slide a window of ``_win_len`` bars across the grid; each position's range
       is ``max(high) - min(low)`` and its pct is ``range / open`` at the window's
       first bar. The actual window is the slide position at ``_win_start_idx``.
    """
    columns = ["range", "range_pct", "_cand_range", "_cand_pct"]
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

    rth_mask = (df["_mod"] >= self.rth_start_min) & (df["_mod"] < self.rth_end_min)
    rth = df[rth_mask]
    if rth.empty:
      return empty

    grid = list(range(self.rth_start_min, self.rth_end_min))
    high_w = rth.pivot_table(index="_date", columns="_mod", values="high", aggfunc="max")
    low_w = rth.pivot_table(index="_date", columns="_mod", values="low", aggfunc="min")
    open_w = rth.pivot_table(index="_date", columns="_mod", values="open", aggfunc="first")

    high_w = high_w.reindex(index=resolved.index, columns=grid)
    low_w = low_w.reindex(index=resolved.index, columns=grid)
    open_w = open_w.reindex(index=resolved.index, columns=grid)

    high = high_w.to_numpy(dtype=float)
    low = low_w.to_numpy(dtype=float)
    opens = open_w.to_numpy(dtype=float)

    # Sliding windows of _win_len bars: (n_days, n_cand, _win_len). nanmax/nanmin
    # ignore the sporadic missing minute; an all-NaN window yields NaN (filtered).
    high_sw = sliding_window_view(high, self._win_len, axis=1)
    low_sw = sliding_window_view(low, self._win_len, axis=1)
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", category=RuntimeWarning)
      cand_high = np.nanmax(high_sw, axis=2)
      cand_low = np.nanmin(low_sw, axis=2)
    cand_range = cand_high - cand_low

    # Open of each candidate window = open at its first bar (grid index == cand).
    cand_open = opens[:, : self._n_cand]
    with np.errstate(divide="ignore", invalid="ignore"):
      cand_pct = np.where(cand_open > 0, cand_range / cand_open, np.nan)

    out = pd.DataFrame(index=resolved.index)
    actual_range = cand_range[:, self._win_start_idx]
    actual_pct = cand_pct[:, self._win_start_idx]
    both = np.isfinite(actual_range) & np.isfinite(actual_pct)
    out["range"] = np.where(both, actual_range, np.nan)
    out["range_pct"] = np.where(both, actual_pct, np.nan)

    valid = np.isfinite(cand_range) & np.isfinite(cand_pct)
    out["_cand_range"] = [cand_range[i][valid[i]] for i in range(len(out))]
    out["_cand_pct"] = [cand_pct[i][valid[i]] for i in range(len(out))]

    return out[columns]

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the eight aggregate rows for the window over a day subset.

    A day counts only when both its ``range`` and ``range_pct`` are valid, so all
    eight rows share the same ``count`` / ``total`` (the number of contributing
    days). If ``baseline_rows`` is provided, each row's ``value_baseline`` /
    ``baseline_n`` is merged from the matching baseline row.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    metrics: dict[str, np.ndarray] = {}
    if not day_table.empty:
      metrics["range"] = day_table["range"].dropna().to_numpy(dtype=float)
      metrics["range_pct"] = day_table["range_pct"].dropna().to_numpy(dtype=float)
    n = len(metrics["range"]) if metrics else 0

    rows: list[StatResultRow] = []
    for out_key, metric, agg, _ in _OUTCOMES:
      value = float(_AGG[agg](metrics[metric])) if n > 0 else 0.0
      bl = baseline_map.get((self.window_key, out_key))
      rows.append(
        StatResultRow(
          condition=self.window_key,
          outcome=out_key,
          count=n,
          total=n,
          probability=0.0,
          baseline_prob=0.0,
          baseline_n=bl.total if bl else 0,
          value=value,
          value_baseline=bl.value if bl else None,
        )
      )
    return rows

  # -------------------------------------------------------------------------
  # Baseline
  # -------------------------------------------------------------------------
  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random null: one random same-length window per day, aggregated identically.

    For each day, a single same-length window is drawn uniformly at random from
    all windows that fit the RTH session (the precomputed ``_cand_range`` /
    ``_cand_pct`` arrays), keeping range and pct paired. The eight aggregates are
    then computed over those random draws exactly as ``compute_rows`` does. This
    is the null hypothesis that the chosen time-of-day window carries no
    information beyond a random window of the same length. Uses
    ``np.random.default_rng(seed)`` for deterministic output.
    """
    # Only days whose actual window is valid contribute to compute_rows, so the
    # baseline must draw from exactly the same days — otherwise a resolved day
    # with a NaN actual range but valid candidates elsewhere would inflate the
    # baseline population (possible only for exotic windows that can miss every
    # bar on some day; the default window always covers the clean 09:30 open).
    day_table = day_table.dropna(subset=["range"])
    if day_table.empty:
      return []

    rng = np.random.default_rng(seed)
    range_draws: list[float] = []
    pct_draws: list[float] = []
    for cand_range, cand_pct in zip(day_table["_cand_range"], day_table["_cand_pct"]):
      k = len(cand_range)
      if k == 0:
        continue
      j = int(rng.integers(k))
      range_draws.append(float(cand_range[j]))
      pct_draws.append(float(cand_pct[j]))

    n = len(range_draws)
    samples = {
      "range": np.asarray(range_draws, dtype=float),
      "range_pct": np.asarray(pct_draws, dtype=float),
    }

    rows: list[StatResultRow] = []
    for out_key, metric, agg, _ in _OUTCOMES:
      value = float(_AGG[agg](samples[metric])) if n > 0 else 0.0
      rows.append(
        StatResultRow(
          condition=self.window_key,
          outcome=out_key,
          count=n,
          total=n,
          probability=0.0,
          baseline_prob=0.0,
          baseline_n=0,
          value=value,
          value_baseline=None,
        )
      )
    return rows

  # -------------------------------------------------------------------------
  # Per-day sample classification
  # -------------------------------------------------------------------------
  def classify_samples(self, day_table: pd.DataFrame) -> list[SampleRow]:
    """One SampleRow per resolved day, per outcome, mirroring ``compute_rows``.

    A day counts only when both its ``range`` and ``range_pct`` are valid (the
    same pairing ``compute_rows`` relies on). Each such day emits eight
    ``SampleRow``s — one per outcome in ``_OUTCOMES`` — carrying that day's raw
    per-day metric (``range`` for the four ``range_*`` outcomes, ``range_pct``
    for the four ``range_pct_*`` outcomes) as ``value``. The outcome's
    aggregate (avg/max/min/median) is applied across days by the caller, not
    here, so all four outcomes sharing a metric carry the identical per-day
    value.
    """
    if day_table.empty:
      return []

    samples: list[SampleRow] = []
    for ts, row in day_table.iterrows():
      rng_val = row["range"]
      pct_val = row["range_pct"]
      if pd.isna(rng_val) or pd.isna(pct_val):
        continue
      date = ts.strftime("%Y-%m-%d")
      for out_key, metric, _, _ in _OUTCOMES:
        value = float(rng_val) if metric == "range" else float(pct_val)
        samples.append(
          SampleRow(date=date, condition=self.window_key, outcome=out_key, value=value)
        )
    return samples


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  start_window: str = "09:30",
  end_window: str = "10:30",
  config_dir: str = "config",
  data_path: str | None = None,
) -> Path:
  """Compute Intraday Range Window for one window. Writes JSON."""
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  stat = IntradayRangeWindow(
    instrument=instrument,
    config=config,
    start_window=start_window,
    end_window=end_window,
  )
  result = stat.compute(candles_df)

  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Intraday Range Window stat")
  parser.add_argument("--instrument", default="NQ", help="Instrument name (default: NQ)")
  parser.add_argument("--start-window", default="09:30", help="Window start HH:MM (default: 09:30)")
  parser.add_argument("--end-window", default="10:30", help="Window end HH:MM (default: 10:30)")
  parser.add_argument("--data-path", default=None, help="Override parquet file path")
  parser.add_argument("--config-dir", default="config", help="Config directory (default: config)")
  args = parser.parse_args()

  output_path = run(
    instrument=args.instrument,
    start_window=args.start_window,
    end_window=args.end_window,
    config_dir=args.config_dir,
    data_path=args.data_path,
  )

  print(f"Written: {output_path}")

  with open(output_path, encoding="utf-8") as f:
    data = json.load(f)

  instrument_data = data["instruments"][args.instrument]
  for tf, tf_data in instrument_data.items():
    print(f"\n  {tf}: {tf_data['total_samples']} resolved days | {tf_data['data_range']}")
    metrics = {r["outcome"]: r["value"] for r in tf_data["results"]}
    print(
      f"    range (points): avg={metrics['range_avg']:.2f} "
      f"max={metrics['range_max']:.2f} min={metrics['range_min']:.2f} "
      f"median={metrics['range_median']:.2f}"
    )
    print(
      f"    range (%):      avg={metrics['range_pct_avg']:.4%} "
      f"max={metrics['range_pct_max']:.4%} min={metrics['range_pct_min']:.4%} "
      f"median={metrics['range_pct_median']:.4%}"
    )
