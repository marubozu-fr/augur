"""Shared base for event-performance stats (CPI, NFP).

Both CPI Performance and NFP Performance ask the same question — what is the
close-to-close percent return in three windows around a scheduled economic
release? — and the only things that differ between them are:

  - ``event_condition_key`` — the condition string reported for the release day
    itself (``"cpi_day"`` vs ``"nfp_day"``).
  - ``event_return_col`` — the day-table column holding the release-day return
    (``"cpi_return"`` vs ``"nfp_return"``).
  - the i18n ``title`` / ``definition`` / ``labels`` metadata and ``stat_name``.
  - the calendar loader that turns the economic-calendar CSV into the set of
    release dates (kept as a standalone function in each module).

This base holds the shared ``build_day_table`` (per-window pending discipline),
``compute_rows`` (mean_return / green / red over each window) and ``baseline_rows``
(per-window sign flip), all parameterized by the two names above, plus a shared
``run`` entry point and CLI ``main`` parameterized by the stat class and loader.

This is a **magnitude** stat: the ``mean_return`` rows carry a signed decimal
return (e.g. ``0.012`` = +1.2%) in ``StatResultRow.value`` and its random
baseline in ``value_baseline``; the green/red event-count rows use the ordinary
``probability`` channel.

Methodology
-----------
A "day" is the RTH daily candle (session open to session close). Each release
date ``D`` is mapped to its trading session in the resolved-day sequence;
releases falling on a non-resolved session (holiday / early close / outside the
data range) are dropped. Using the resolved-day positions, "N trading days
before/after" naturally skips non-trading days.

Let ``c[i]`` be the session close of the resolved day at position ``i`` and let
``D`` sit at position ``i``. With ``pre = pre_announcement`` and
``post = post_announcement`` (in trading sessions), the three windows use
**adjacent closes** (contiguous, non-overlapping):

  - ``pre_announcement``:  (c[i-1] - c[i-pre]) / c[i-pre]   -- the run-up baseline
  - ``<event>_day``:       (c[i]   - c[i-1])  / c[i-1]      -- the release-day move
  - ``post_announcement``: (c[i+post] - c[i]) / c[i]        -- the follow-through

Pending discipline is **per window**: an event contributes to a window only
when every close that window needs exists. Each window therefore reports its own
sample size ``N``. A window observation is **green** when its return is >= 0,
otherwise **red**.

The release dates are not derived from the OHLCV data; they are read from an
external economic calendar CSV and injected into the stat, so the stat stays a
pure function of (candles, release dates).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from stats.base import BaseStat, SampleRow, StatResultRow, write_results
from stats.config import InstrumentConfig, load_config, minute_of_day
from stats.utils.daily_candles import build_resolved_days

# Default economic calendar CSV (shared by every event-performance stat).
_DEFAULT_CALENDAR_PATH = Path("data/forex_factory_calendar.csv")


class EventPerformanceStat(BaseStat):
  """Close-to-close performance across the three windows around an economic event.

  Subclasses set ``event_condition_key`` (the release-day condition string),
  ``event_return_col`` (the day-table column for the release-day return), the
  i18n metadata (``stat_name``, ``title``, ``definition``, ``labels``) and the
  declarative ``slices`` tuple. All computation lives here.
  """

  # Names — set by each subclass.
  event_condition_key: str
  event_return_col: str

  def __init__(
    self,
    instrument: str,
    config: InstrumentConfig,
    event_dates: Iterable[pd.Timestamp],
    pre_announcement: int = 5,
    post_announcement: int = 5,
    close_tolerance_min: int = 15,
  ) -> None:
    if pre_announcement < 1 or post_announcement < 1:
      raise ValueError("pre_announcement and post_announcement must be >= 1")
    self.instrument = instrument
    self.timeframe = "daily"
    self.config = config
    self.pre_announcement = pre_announcement
    self.post_announcement = post_announcement
    self.close_tolerance_min = close_tolerance_min
    # Normalize to a set of midnight, tz-naive Timestamps for date alignment.
    self.event_dates: set[pd.Timestamp] = {
      pd.Timestamp(d).normalize().tz_localize(None) for d in event_dates
    }

    rth = config.sessions["rth"]
    self.rth_start_min: int = minute_of_day(rth.start)
    self.rth_end_min: int = minute_of_day(rth.end)

  # -------------------------------------------------------------------------
  # Window layout (derived from the subclass names)
  # -------------------------------------------------------------------------
  @property
  def windows(self) -> tuple[tuple[str, str], ...]:
    """(condition, return-column) for the three windows, in chronological order."""
    return (
      ("pre_announcement", "pre_return"),
      (self.event_condition_key, self.event_return_col),
      ("post_announcement", "post_return"),
    )

  @property
  def return_columns(self) -> list[str]:
    return [col for _, col in self.windows]

  # -------------------------------------------------------------------------
  # Day table construction
  # -------------------------------------------------------------------------
  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-event table of the three window returns.

    Each row is one release that maps to a resolved trading session, indexed by
    the release date. The three return columns hold the signed decimal return for
    each window, or ``NaN`` when that window runs past the available resolved
    sessions (pending). Rows whose three windows are all pending are dropped.
    """
    columns = self.return_columns
    empty = pd.DataFrame(columns=columns)

    if not self.event_dates:
      return empty

    resolved = build_resolved_days(
      candles_df, self.rth_start_min, self.rth_end_min, self.close_tolerance_min
    )
    if resolved.empty:
      return empty

    resolved = resolved.sort_index()
    closes = resolved["session_close"].to_numpy(dtype=float)
    n = len(closes)
    # Resolved sessions keyed by their (tz-naive) date for release alignment.
    pos_by_date = {
      ts.normalize().tz_localize(None): i for i, ts in enumerate(resolved.index)
    }

    pre, post = self.pre_announcement, self.post_announcement
    records: list[tuple[float, float, float]] = []
    index: list[pd.Timestamp] = []
    for d in sorted(self.event_dates):
      i = pos_by_date.get(d)
      if i is None:
        continue  # release fell on a non-resolved session (holiday / outside data)

      pre_ret = (closes[i - 1] - closes[i - pre]) / closes[i - pre] if i >= pre else np.nan
      event_ret = (closes[i] - closes[i - 1]) / closes[i - 1] if i >= 1 else np.nan
      post_ret = (closes[i + post] - closes[i]) / closes[i] if i + post <= n - 1 else np.nan

      if np.isnan(pre_ret) and np.isnan(event_ret) and np.isnan(post_ret):
        continue
      records.append((pre_ret, event_ret, post_ret))
      index.append(d)

    if not records:
      return empty

    return pd.DataFrame(records, columns=columns, index=pd.DatetimeIndex(index))

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the three rows (mean_return, green, red) for each window.

    Each window aggregates only its non-NaN observations (pending discipline),
    so the three windows can have different sample sizes. The nine rows are
    always emitted (even when ``N == 0``) for a deterministic result shape.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    rows: list[StatResultRow] = []
    for condition, column in self.windows:
      if len(day_table) > 0 and column in day_table.columns:
        # Window columns are built as floats (np.nan + float divisions), so a
        # plain dropna() is enough — no numeric coercion needed.
        vals = day_table[column].dropna()
      else:
        vals = pd.Series([], dtype=float)

      total = len(vals)
      green_count = int((vals >= 0).sum())
      red_count = total - green_count
      mean_return = float(vals.mean()) if total > 0 else 0.0

      # (outcome, count, probability, value) -- value is None for probability rows.
      specs: list[tuple[str, int, float, float | None]] = [
        ("mean_return", total, 0.0, mean_return),
        ("green", green_count, green_count / total if total > 0 else 0.0, None),
        ("red", red_count, red_count / total if total > 0 else 0.0, None),
      ]
      for out_key, count, probability, value in specs:
        bl = baseline_map.get((condition, out_key))
        rows.append(
          StatResultRow(
            condition=condition,
            outcome=out_key,
            count=count,
            total=total,
            probability=probability,
            baseline_prob=bl.probability if bl else 0.0,
            baseline_n=bl.total if bl else 0,
            value=value,
            value_baseline=(bl.value if bl else None) if value is not None else None,
          )
        )
    return rows

  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random baseline: each observation's return direction is a coin flip.

    Per window, magnitudes are held fixed and only the sign of each non-NaN
    return is randomized (p=0.5), so the expected average return is ~0 and the
    expected green/red split is ~50/50. NaN (pending) entries stay NaN.
    Deterministic for a fixed seed.
    """
    rng = np.random.default_rng(seed)
    tmp = day_table.copy()
    for column in self.return_columns:
      if column not in tmp.columns:
        continue
      values = tmp[column].to_numpy(dtype=float).copy()
      mask = ~np.isnan(values)
      if mask.any():
        signs = rng.integers(0, 2, size=int(mask.sum())) * 2 - 1  # ±1
        values[mask] = signs * np.abs(values[mask])
      tmp[column] = values
    return self.compute_rows(tmp, baseline_rows=None)

  def classify_samples(self, day_table: pd.DataFrame) -> list[SampleRow]:
    """One SampleRow per non-pending window observation, mirroring ``compute_rows``.

    Each event row contributes up to three samples (one per window), skipping
    windows whose return is NaN (pending). The ``mean_return`` row that
    ``compute_rows`` emits is a derived aggregate over the green/red samples,
    not a sample outcome itself.
    """
    if day_table.empty:
      return []

    samples: list[SampleRow] = []
    for ts, row in day_table.iterrows():
      date = ts.strftime("%Y-%m-%d")
      for condition, column in self.windows:
        ret = row[column]
        if pd.notna(ret):
          samples.append(
            SampleRow(
              date=date,
              condition=condition,
              outcome="green" if ret >= 0 else "red",
              value=float(ret),
            )
          )
    return samples


# ---------------------------------------------------------------------------
# Shared run() entry point and CLI
# ---------------------------------------------------------------------------
def run_event_performance_stat(
  stat_cls: type[EventPerformanceStat],
  load_dates: Callable[[str | Path], Iterable[pd.Timestamp]],
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
  calendar_path: str | Path = _DEFAULT_CALENDAR_PATH,
  pre_announcement: int = 5,
  post_announcement: int = 5,
) -> Path:
  """Load data and the calendar, compute an event-performance stat, write it.

  ``load_dates`` reads the release dates from the calendar CSV; it is the only
  per-stat piece of the run, so each module passes its own loader.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)
  event_dates = load_dates(calendar_path)

  stat = stat_cls(
    instrument=instrument,
    config=config,
    event_dates=event_dates,
    pre_announcement=pre_announcement,
    post_announcement=post_announcement,
  )
  result = stat.compute(candles_df)
  return write_results(result)


def main(
  stat_cls: type[EventPerformanceStat],
  load_dates: Callable[[str | Path], Iterable[pd.Timestamp]],
  description: str,
  event_noun: str,
) -> None:
  """Shared CLI: parse args, run ``stat_cls``, and print a per-window summary."""
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument("--instrument", default="NQ", help="Instrument name (default: NQ)")
  parser.add_argument("--data-path", default=None, help="Override parquet file path")
  parser.add_argument("--config-dir", default="config", help="Config directory (default: config)")
  parser.add_argument(
    "--calendar-path",
    default=str(_DEFAULT_CALENDAR_PATH),
    help=f"Economic calendar CSV path (default: {_DEFAULT_CALENDAR_PATH})",
  )
  parser.add_argument(
    "--pre-announcement",
    type=int,
    default=5,
    help="Trading sessions before the release in the pre window (default: 5)",
  )
  parser.add_argument(
    "--post-announcement",
    type=int,
    default=5,
    help="Trading sessions after the release in the post window (default: 5)",
  )
  args = parser.parse_args()

  output_path = run_event_performance_stat(
    stat_cls,
    load_dates,
    instrument=args.instrument,
    config_dir=args.config_dir,
    data_path=args.data_path,
    calendar_path=args.calendar_path,
    pre_announcement=args.pre_announcement,
    post_announcement=args.post_announcement,
  )

  print(f"Written: {output_path}")

  with open(output_path, encoding="utf-8") as f:
    data = json.load(f)

  conditions = ["pre_announcement", stat_cls.event_condition_key, "post_announcement"]
  instrument_data = data["instruments"][args.instrument]
  for tf, tf_data in instrument_data.items():
    print(f"\n  {tf}: {tf_data['total_samples']} {event_noun} events | {tf_data['data_range']}")
    by_window: dict[str, dict] = {}
    for row in tf_data["results"]:
      by_window.setdefault(row["condition"], {})[row["outcome"]] = row
    for condition in conditions:
      rows = by_window.get(condition, {})
      mean_ret = rows.get("mean_return", {}).get("value", 0.0)
      green = rows.get("green", {})
      print(
        f"    {condition}: avg return={mean_ret:+.4%} "
        f"| P(green)={green.get('probability', 0.0):.3f} (N={green.get('total', 0)})"
      )
