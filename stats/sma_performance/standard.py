"""SMA Performance stat.

Detects price cross-ups and cross-downs against the N-period simple moving
average (SMA) of session closes, then measures:
  - Duration: how many consecutive sessions the move lasts before reversing.
  - Travel: the peak favorable excursion (furthest close from the SMA, in
    points, in the direction of the move).

This is a magnitude stat — it populates the ``value`` / ``value_baseline``
channel of ``StatResultRow`` rather than the probability channel.

Methodology
-----------
For each resolved session, compute ``dist = session_close - sma`` where the
SMA uses only the ``period`` sessions that strictly precede the current one
(no lookahead, via ``shift(1)`` on the rolling mean).

A "move" is a maximal run of consecutive sessions with the same sign of dist:
  - "up"   when dist > 0 (close is above the SMA).
  - "down" when dist <= 0 (close is on or below the SMA; ties go to "down"
    for determinism).

A run is a **confirmed cross event** only when it is both preceded and followed
by an opposite-regime run (it has genuinely crossed in *and* back out). The
first run (no prior regime → not a genuine cross) and the last run (has not
yet crossed back → pending) are excluded.

Among confirmed runs, only those with ``duration >= min_duration`` sessions
are kept. Rows are emitted for:
  - conditions: cross_up, cross_down
  - outcomes:   avg_duration, max_duration, avg_travel, max_travel

``probability = 0.0`` is a sentinel; the meaningful number is ``value``.
All 8 rows are always emitted (even when N = 0) for deterministic shape.

Pending discipline: the first ``period`` resolved sessions (for which the
prior-window SMA is undefined) are excluded from the dist sequence. The last
run of any sign is always excluded (unresolved continuation).
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
  en="SMA Performance",
  fr="Performance SMA",
)
_DEFINITION = I18nString(
  en=(
    "When price crosses above or below the N-period simple moving average, "
    "how long does the move last (in bars) and how far does price travel "
    "from the SMA before crossing back?"
  ),
  fr=(
    "Lorsque le prix croise au-dessus ou en dessous de la moyenne mobile "
    "simple sur N périodes, combien de temps dure le mouvement (en barres) "
    "et jusqu'où le prix s'éloigne-t-il de la SMA avant de retraverser ?"
  ),
)
_LABELS = Labels(
  conditions={
    "cross_up": I18nString(en="Cross above SMA", fr="Croisement au-dessus de la SMA"),
    "cross_down": I18nString(en="Cross below SMA", fr="Croisement en dessous de la SMA"),
  },
  outcomes={
    "avg_duration": I18nString(
      en="Average duration (bars)", fr="Durée moyenne (barres)"
    ),
    "max_duration": I18nString(
      en="Maximum duration (bars)", fr="Durée maximale (barres)"
    ),
    "avg_travel": I18nString(
      en="Average travel (points)", fr="Distance moyenne (points)"
    ),
    "max_travel": I18nString(
      en="Maximum travel (points)", fr="Distance maximale (points)"
    ),
  },
)

# Fixed ordering for the 8 emitted rows.
_CONDITIONS = ("cross_up", "cross_down")
_OUTCOMES = ("avg_duration", "max_duration", "avg_travel", "max_travel")

# Outcome -> aggregation function name, for the `agg` field.
_AGG_BY_OUTCOME = {
  "avg_duration": "mean",
  "max_duration": "max",
  "avg_travel": "mean",
  "max_travel": "max",
}


class SMAPerformance(BaseStat):
  """SMA cross duration and travel magnitude stat (daily timeframe)."""

  stat_name = "sma_performance"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = ()  # Events span multiple days — per-day slicing is meaningless.

  def __init__(
    self,
    instrument: str,
    config: InstrumentConfig,
    period: int = 200,
    min_duration: int = 5,
    close_tolerance_min: int = 15,
  ) -> None:
    self.instrument = instrument
    self.timeframe = "daily"
    self.config = config
    self.period = period
    self.min_duration = min_duration
    self.close_tolerance_min = close_tolerance_min

    rth = config.sessions["rth"]
    self.rth_start_min: int = minute_of_day(rth.start)
    self.rth_end_min: int = minute_of_day(rth.end)

  # -------------------------------------------------------------------------
  # Day table construction
  # -------------------------------------------------------------------------
  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-session dist table.

    Returns a DataFrame indexed by the normalized session date with a single
    column ``dist`` = session_close - prior_sma (signed distance from the
    N-period SMA of session closes). The first ``period`` resolved sessions,
    for which the prior-window SMA is undefined, are dropped.

    Returns an empty DataFrame with column ``["dist"]`` when there are no
    rows after filtering.
    """
    empty = pd.DataFrame(columns=["dist"])

    resolved = build_resolved_days(
      candles_df, self.rth_start_min, self.rth_end_min, self.close_tolerance_min
    )
    if resolved.empty:
      return empty

    resolved = resolved.sort_index()
    close = resolved["session_close"]

    # SMA of the prior `period` session closes (no lookahead).
    # rolling(period).mean() at position i = mean of [i-period, i-1].
    # shift(1) pushes that value to position i+1, so the SMA available on day i
    # uses only sessions strictly before day i — identical to the ATR convention.
    sma = close.rolling(self.period).mean().shift(1)

    dist = close - sma

    # Drop rows where sma is NaN (the first `period` resolved sessions).
    mask = sma.notna()
    if not mask.any():
      return empty

    return pd.DataFrame({"dist": dist[mask]}, index=resolved.index[mask])

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  @staticmethod
  def _extract_runs(dist: pd.Series) -> list[dict]:
    """Extract all maximal runs of same-sign sessions from ``dist``.

    Returns a list of dicts with keys:
      regime    "up" | "down"
      duration  int  number of sessions in the run
      travel    float  peak excursion magnitude (points from SMA)
    Ordered chronologically.
    """
    if dist.empty:
      return []

    values = dist.to_numpy()
    regimes = np.where(values > 0, "up", "down")

    runs: list[dict] = []
    i = 0
    n = len(regimes)
    while i < n:
      regime = regimes[i]
      j = i
      while j < n and regimes[j] == regime:
        j += 1
      # Slice of dist values for this run.
      run_vals = values[i:j]
      if regime == "up":
        travel = float(np.max(run_vals))
      else:
        travel = float(np.max(-run_vals))
      runs.append({"regime": regime, "duration": j - i, "travel": travel})
      i = j

    return runs

  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the 8 magnitude rows (condition × outcome).

    Confirmed cross events = all runs except the first (no prior regime) and
    the last (pending reversal). Among those, keep only runs with
    ``duration >= self.min_duration``.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    all_runs = self._extract_runs(day_table["dist"] if len(day_table) > 0 else pd.Series([], dtype=float))

    # Confirmed = not the first and not the last run.
    confirmed = all_runs[1:-1] if len(all_runs) >= 3 else []

    # Filter by minimum duration.
    qualified = [r for r in confirmed if r["duration"] >= self.min_duration]

    # Partition by regime.
    up_runs = [r for r in qualified if r["regime"] == "up"]
    down_runs = [r for r in qualified if r["regime"] == "down"]

    def _aggregate(
      runs: list[dict], condition: str
    ) -> list[StatResultRow]:
      n = len(runs)
      rows: list[StatResultRow] = []
      for outcome in _OUTCOMES:
        if n == 0:
          value = 0.0
        elif outcome == "avg_duration":
          value = float(np.mean([r["duration"] for r in runs]))
        elif outcome == "max_duration":
          value = float(np.max([r["duration"] for r in runs]))
        elif outcome == "avg_travel":
          value = float(np.mean([r["travel"] for r in runs]))
        else:  # max_travel
          value = float(np.max([r["travel"] for r in runs]))

        bl = baseline_map.get((condition, outcome))
        rows.append(
          StatResultRow(
            condition=condition,
            outcome=outcome,
            count=n,
            total=n,
            probability=0.0,
            baseline_prob=0.0,
            baseline_n=bl.total if bl else 0,
            value=value,
            value_baseline=bl.value if bl else None,
            agg=_AGG_BY_OUTCOME[outcome],
          )
        )
      return rows

    result: list[StatResultRow] = []
    result.extend(_aggregate(up_runs, "cross_up"))
    result.extend(_aggregate(down_runs, "cross_down"))
    return result

  def classify_samples(self, day_table: pd.DataFrame) -> list[SampleRow]:
    """One SampleRow set per qualified cross-event run, mirroring ``compute_rows``.

    Confirmed cross events (``all_runs[1:-1]``) filtered by ``min_duration`` are
    the same qualified runs ``compute_rows`` aggregates — a run, not a single
    day, is the natural sample unit here (duration/travel are properties of the
    whole move). Each qualified run emits 4 SampleRows, one per outcome, all
    dated on the run's LAST session (the last day the move persisted before the
    opposite-regime run that confirmed it). ``avg_duration`` and
    ``max_duration`` both carry the run's ``duration`` as ``value``;
    ``avg_travel`` and ``max_travel`` both carry its ``travel`` — the
    aggregation method (mean vs. max) differs only at the ``compute_rows``
    level, not per sample. ``condition`` is ``cross_up``/``cross_down`` from
    the run's regime.
    """
    if day_table.empty:
      return []

    all_runs = self._extract_runs(day_table["dist"])
    if len(all_runs) < 3:
      return []

    # Each run's last-session position, via cumulative durations, so the
    # confirmed slice (all_runs[1:-1]) can be dated consistently with
    # compute_rows' chronological ordering.
    end_indices: list[int] = []
    cursor = 0
    for r in all_runs:
      cursor += r["duration"]
      end_indices.append(cursor - 1)

    confirmed_runs = all_runs[1:-1]
    confirmed_end_indices = end_indices[1:-1]

    samples: list[SampleRow] = []
    for run, end_idx in zip(confirmed_runs, confirmed_end_indices):
      if run["duration"] < self.min_duration:
        continue
      condition = "cross_up" if run["regime"] == "up" else "cross_down"
      date_str = day_table.index[end_idx].strftime("%Y-%m-%d")
      duration = float(run["duration"])
      travel = float(run["travel"])
      for outcome in ("avg_duration", "max_duration"):
        samples.append(
          SampleRow(date=date_str, condition=condition, outcome=outcome, value=duration)
        )
      for outcome in ("avg_travel", "max_travel"):
        samples.append(
          SampleRow(date=date_str, condition=condition, outcome=outcome, value=travel)
        )
    return samples

  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random baseline destroying serial autocorrelation.

    Permutes the ``dist`` column (destroying trend/memory), then calls
    ``compute_rows`` on the shuffled table. This yields the expected
    duration and travel if daily distances were i.i.d. (no persistence).
    Deterministic for a fixed seed.
    """
    if day_table.empty:
      return self.compute_rows(day_table, baseline_rows=None)

    rng = np.random.default_rng(seed)
    shuffled_dist = rng.permutation(day_table["dist"].to_numpy())
    shuffled = pd.DataFrame({"dist": shuffled_dist}, index=day_table.index)
    return self.compute_rows(shuffled, baseline_rows=None)


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
  period: int = 200,
  min_duration: int = 5,
) -> Path:
  """Load data and compute SMA Performance for the daily timeframe."""
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  stat = SMAPerformance(
    instrument=instrument,
    config=config,
    period=period,
    min_duration=min_duration,
  )
  result = stat.compute(candles_df)
  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute SMA Performance stat")
  parser.add_argument("--instrument", default="NQ", help="Instrument name (default: NQ)")
  parser.add_argument("--data-path", default=None, help="Override parquet file path")
  parser.add_argument("--config-dir", default="config", help="Config directory (default: config)")
  parser.add_argument(
    "--period",
    type=int,
    default=200,
    help="SMA period in sessions (default: 200)",
  )
  parser.add_argument(
    "--min-duration",
    type=int,
    default=5,
    help="Minimum run length to count as a qualified cross event (default: 5)",
  )
  args = parser.parse_args()

  output_path = run(
    instrument=args.instrument,
    config_dir=args.config_dir,
    data_path=args.data_path,
    period=args.period,
    min_duration=args.min_duration,
  )

  print(f"Written: {output_path}")

  with open(output_path, encoding="utf-8") as f:
    data = json.load(f)

  instrument_data = data["instruments"][args.instrument]
  for tf, tf_data in instrument_data.items():
    print(f"\n  {tf}: {tf_data['total_samples']} samples | {tf_data['data_range']}")
    for row in tf_data["results"]:
      bl_val = row.get("value_baseline")
      bl_str = f"{bl_val:.4f}" if bl_val is not None else "None"
      print(
        f"    {row['condition']} -> {row['outcome']}: "
        f"value={row['value']:.4f} (N={row['total']}, value_baseline={bl_str})"
      )
