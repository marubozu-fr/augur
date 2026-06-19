"""CPI Performance stat.

Measures close-to-close price performance across three windows around each CPI
release: a pre-announcement window, the CPI release day itself, and a
post-announcement window.

This is a **magnitude** stat: the ``mean_return`` rows carry a signed decimal
return (e.g. ``0.012`` = +1.2%) in ``StatResultRow.value`` and its random
baseline in ``value_baseline``; the green/red event-count rows use the ordinary
``probability`` channel.

Methodology
-----------
A "day" is the RTH daily candle (session open to session close). Each CPI
release date ``D`` is mapped to its trading session in the resolved-day
sequence; releases falling on a non-resolved session (holiday / early close /
outside the data range) are dropped. Using the resolved-day positions, "N
trading days before/after" naturally skips non-trading days.

Let ``c[i]`` be the session close of the resolved day at position ``i`` and let
``D`` sit at position ``i``. With ``pre = pre_announcement`` and
``post = post_announcement`` (in trading sessions), the three windows use
**adjacent closes** (contiguous, non-overlapping):

  - ``pre_announcement``:  (c[i-1] - c[i-pre]) / c[i-pre]   -- the run-up baseline
  - ``cpi_day``:           (c[i]   - c[i-1])  / c[i-1]      -- the release-day move
  - ``post_announcement``: (c[i+post] - c[i]) / c[i]        -- the follow-through

Pending discipline is **per window**: an event contributes to a window only
when every close that window needs exists. The earliest events lack enough
prior sessions for the pre window; the most recent events lack enough following
sessions for the post window (their post return is still unresolved). Each
window therefore reports its own sample size ``N``.

A window observation is **green** when its return is >= 0, otherwise **red**.

The CPI release dates are not derived from the OHLCV data; they are read from an
external economic calendar CSV (see ``load_cpi_release_dates``) and injected into
the stat, so the stat stays a pure function of (candles, release dates).
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
  StatResultRow,
  write_results,
)
from stats.config import InstrumentConfig, load_config, minute_of_day
from stats.utils.daily_candles import build_resolved_days

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="CPI Performance",
  fr="Performance CPI",
)
_DEFINITION = I18nString(
  en=(
    "What is the average close-to-close percent return in the trading sessions "
    "before a CPI release, on the release day itself, and in the sessions after?"
  ),
  fr=(
    "Quel est le rendement moyen en pourcentage de clôture à clôture lors des "
    "séances précédant une publication du CPI, le jour de la publication, et "
    "lors des séances suivantes ?"
  ),
)
_LABELS = Labels(
  conditions={
    "pre_announcement": I18nString(
      en="Pre-announcement window", fr="Fenêtre pré-annonce"
    ),
    "cpi_day": I18nString(en="CPI release day", fr="Jour de publication du CPI"),
    "post_announcement": I18nString(
      en="Post-announcement window", fr="Fenêtre post-annonce"
    ),
  },
  outcomes={
    "mean_return": I18nString(en="Average return", fr="Rendement moyen"),
    "green": I18nString(en="Green (up)", fr="Vert (hausse)"),
    "red": I18nString(en="Red (down)", fr="Rouge (baisse)"),
  },
)

# Window condition -> day-table column holding that window's signed return.
_WINDOWS: tuple[tuple[str, str], ...] = (
  ("pre_announcement", "pre_return"),
  ("cpi_day", "cpi_return"),
  ("post_announcement", "post_return"),
)
_RETURN_COLUMNS = [col for _, col in _WINDOWS]

# Calendar row identifying a CPI release. The forex-factory calendar carries
# several CPI lines per release (headline / core, m/m / y/y); the headline
# monthly print uniquely identifies each release date.
_CPI_EVENT = "cpi m/m"
_CPI_CURRENCY = "usd"

_DEFAULT_CALENDAR_PATH = Path("data/forex_factory_calendar.csv")


def load_cpi_release_dates(calendar_path: str | Path) -> set[pd.Timestamp]:
  """Read distinct CPI release dates from an economic calendar CSV.

  Keeps rows whose ``event`` is the headline monthly US CPI print and returns
  the distinct release dates as normalized (midnight, tz-naive) Timestamps.
  """
  cal = pd.read_csv(calendar_path, usecols=["date", "currency", "event"])
  event = cal["event"].str.strip().str.lower()
  currency = cal["currency"].str.strip().str.lower()
  mask = (event == _CPI_EVENT) & (currency == _CPI_CURRENCY)
  dates = pd.to_datetime(cal.loc[mask, "date"]).dt.normalize()
  return set(dates.unique())


class CPIPerformance(BaseStat):
  """Close-to-close performance across the three CPI-release windows."""

  stat_name = "cpi_performance"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = ()  # Events are scattered across the calendar; per-day slicing is meaningless.

  def __init__(
    self,
    instrument: str,
    config: InstrumentConfig,
    cpi_dates: Iterable[pd.Timestamp],
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
    self.cpi_dates: set[pd.Timestamp] = {
      pd.Timestamp(d).normalize().tz_localize(None) for d in cpi_dates
    }

    rth = config.sessions["rth"]
    self.rth_start_min: int = minute_of_day(rth.start)
    self.rth_end_min: int = minute_of_day(rth.end)

  # -------------------------------------------------------------------------
  # Day table construction
  # -------------------------------------------------------------------------
  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-event table of the three window returns.

    Each row is one CPI release that maps to a resolved trading session, indexed
    by the release date. Columns ``pre_return``, ``cpi_return``, ``post_return``
    hold the signed decimal return for each window, or ``NaN`` when that window
    runs past the available resolved sessions (pending). Rows whose three
    windows are all pending are dropped.
    """
    columns = _RETURN_COLUMNS
    empty = pd.DataFrame(columns=columns)

    if not self.cpi_dates:
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
    for d in sorted(self.cpi_dates):
      i = pos_by_date.get(d)
      if i is None:
        continue  # release fell on a non-resolved session (holiday / outside data)

      pre_ret = (closes[i - 1] - closes[i - pre]) / closes[i - pre] if i >= pre else np.nan
      cpi_ret = (closes[i] - closes[i - 1]) / closes[i - 1] if i >= 1 else np.nan
      post_ret = (closes[i + post] - closes[i]) / closes[i] if i + post <= n - 1 else np.nan

      if np.isnan(pre_ret) and np.isnan(cpi_ret) and np.isnan(post_ret):
        continue
      records.append((pre_ret, cpi_ret, post_ret))
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
    for condition, column in _WINDOWS:
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
    for column in _RETURN_COLUMNS:
      if column not in tmp.columns:
        continue
      values = pd.to_numeric(tmp[column], errors="coerce").to_numpy(dtype=float).copy()
      mask = ~np.isnan(values)
      if mask.any():
        signs = rng.integers(0, 2, size=int(mask.sum())) * 2 - 1  # ±1
        values[mask] = signs * np.abs(values[mask])
      tmp[column] = values
    return self.compute_rows(tmp, baseline_rows=None)


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
  calendar_path: str | Path = _DEFAULT_CALENDAR_PATH,
  pre_announcement: int = 5,
  post_announcement: int = 5,
) -> Path:
  """Load data and the CPI calendar, compute CPI Performance, write the result."""
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)
  cpi_dates = load_cpi_release_dates(calendar_path)

  stat = CPIPerformance(
    instrument=instrument,
    config=config,
    cpi_dates=cpi_dates,
    pre_announcement=pre_announcement,
    post_announcement=post_announcement,
  )
  result = stat.compute(candles_df)
  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute CPI Performance stat")
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

  output_path = run(
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

  instrument_data = data["instruments"][args.instrument]
  for tf, tf_data in instrument_data.items():
    print(f"\n  {tf}: {tf_data['total_samples']} CPI events | {tf_data['data_range']}")
    by_window: dict[str, dict] = {}
    for row in tf_data["results"]:
      by_window.setdefault(row["condition"], {})[row["outcome"]] = row
    for condition, _ in _WINDOWS:
      rows = by_window.get(condition, {})
      mean_ret = rows.get("mean_return", {}).get("value", 0.0)
      green = rows.get("green", {})
      print(
        f"    {condition}: avg return={mean_ret:+.4%} "
        f"| P(green)={green.get('probability', 0.0):.3f} (N={green.get('total', 0)})"
      )
