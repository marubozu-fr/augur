"""Market Session Correlation stat — standard variant.

Measures: how often does a SECOND market session close green (or red) given the
color a FIRST market session closed, within the same trading cycle? Intraday
color follow-through between two named sessions, reported as a 2x2 conditional
matrix:
  - P(session 2 green | session 1 green), P(session 2 red | session 1 green)
  - P(session 2 green | session 1 red),   P(session 2 red | session 1 red)

Both sessions are named windows read from the instrument config (the single
source of truth for their bounds) — e.g. ``london`` (session 1) → ``ny``
(session 2), the default pair. Any ordered pair of configured sessions works; the
pair must be ordered within a cycle (session 1 earlier than session 2). This is
the within-cycle, two-session companion to ``prev_session_correlation`` (which
correlates a session with the chronologically PREVIOUS session of the same kind).

Direction is controlled by ``performance``:
  - ``close_to_close`` (default): a session is green when its close is at or above
    the PREVIOUS cycle's close for that same session. The first cycle has no prior
    close for either session and is excluded (pending-sample discipline).
  - ``open_to_close``: a session is green when its close is at or above its open.

A **cycle** attributes bars to the RTH session date they belong to (a
cross-midnight session maps its evening bars to the next day's cycle). Session 1
and session 2 are inner-joined on this shared cycle date: a cycle is countable
only when BOTH sessions are resolved for it. ``total_samples`` and each row's
``total`` count countable cycles only; the four matrix outcomes per condition
partition the cycles carrying that session 1 color.
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
from stats.utils.session_candles import session_bars

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="Market Session Correlation",
  fr="Corrélation de session de marché",
)
_DEFINITION = I18nString(
  en="How often does a second market session close green or red given the color a first market session closed, within the same trading cycle?",
  fr="À quelle fréquence une deuxième session de marché clôture-t-elle en vert ou en rouge selon la couleur de clôture d'une première session, au cours du même cycle de trading ?",
)
_LABELS = Labels(
  conditions={
    "s1_green": I18nString(en="Session 1 green", fr="Session 1 verte"),
    "s1_red": I18nString(en="Session 1 red", fr="Session 1 rouge"),
  },
  outcomes={
    "green": I18nString(en="Session 2 green (up)", fr="Session 2 verte (hausse)"),
    "red": I18nString(en="Session 2 red (down)", fr="Session 2 rouge (baisse)"),
  },
)

# Performance modes: how a session's direction (green/red) is determined.
_PERFORMANCE_MODES = ("close_to_close", "open_to_close")

# Condition / outcome enumeration: session 1 color -> session 2 color.
_CONDITIONS = (("s1_green", True), ("s1_red", False))
_OUTCOMES = (("green", True), ("red", False))

_MINUTES_PER_DAY = 24 * 60


class MarketSessionCorrelation(BaseStat):
  """Conditional probability of session 2's color given session 1's color."""

  stat_name = "market_session_correlation"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = (
    "weekday",
    SizeBucket(column="s1_size", preset="quartiles", name="size"),
  )

  def __init__(
    self,
    instrument: str,
    config: InstrumentConfig,
    session1: str = "london",
    session2: str = "ny",
    performance: str = "close_to_close",
    close_tolerance_min: int = 15,
  ) -> None:
    if performance not in _PERFORMANCE_MODES:
      raise ValueError(
        f"Unsupported performance '{performance}'. Choose from {list(_PERFORMANCE_MODES)}"
      )
    if session1 == session2:
      raise ValueError("session1 and session2 must be different sessions")
    for name in (session1, session2):
      if name not in config.sessions:
        raise ValueError(
          f"Unknown session '{name}'. Configured sessions: {list(config.sessions)}"
        )

    self.instrument = instrument
    self.timeframe = "daily"
    self.config = config
    self.session1 = session1
    self.session2 = session2
    self.performance = performance
    self.close_tolerance_min = close_tolerance_min

    s1 = config.sessions[session1]
    self.s1_start_min: int = minute_of_day(s1.start)
    self.s1_end_min: int = minute_of_day(s1.end)
    s2 = config.sessions[session2]
    self.s2_start_min: int = minute_of_day(s2.start)
    self.s2_end_min: int = minute_of_day(s2.end)

  # -------------------------------------------------------------------------
  # Day table construction
  # -------------------------------------------------------------------------
  def _session_table(
    self, df: pd.DataFrame, start_min: int, end_min: int
  ) -> pd.DataFrame:
    """Per-cycle session candle, indexed by the cycle date.

    Columns: ``open`` (first bar's open), ``close`` (last bar's close), ``high``,
    ``low``. A cycle is **resolved** when the session has a clean open bar (offset
    0 from ``start``) AND end coverage (last bar at or after
    ``duration - close_tolerance``). Only resolved cycles are returned.
    """
    bars = session_bars(df, start_min, end_min)
    if bars.empty:
      return pd.DataFrame()

    duration = (end_min - start_min) % _MINUTES_PER_DAY or _MINUTES_PER_DAY
    bars["_offset"] = (bars["_mod"] - start_min) % _MINUTES_PER_DAY

    grouped = bars.groupby("_cycle")
    # idxmin/idxmax return the first matching label; bars are chronologically
    # sorted upstream, so the min-offset bar is the open and the max-offset bar
    # is the close.
    open_bar = bars.loc[grouped["_offset"].idxmin()].set_index("_cycle")["open"]
    close_bar = bars.loc[grouped["_offset"].idxmax()].set_index("_cycle")["close"]
    table = pd.DataFrame(
      {
        "open": open_bar,
        "close": close_bar,
        "high": grouped["high"].max(),
        "low": grouped["low"].min(),
        "_has_open": grouped["_offset"].min() == 0,
        "_last_offset": grouped["_offset"].max(),
      }
    )
    resolved = table["_has_open"] & (
      table["_last_offset"] >= duration - self.close_tolerance_min
    )
    table = table[resolved]
    return table.drop(columns=["_has_open", "_last_offset"])

  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-cycle table joining session 1 and session 2 colors.

    Columns returned:
      ``s1_green`` — session 1 close color (bool).
      ``s2_green`` — session 2 close color (bool).
      ``s1_size``  — session 1 range size (``s1_high - s1_low``), read by the size
                     slicer.

    A cycle is countable only when BOTH sessions are resolved for it; the inner
    join drops any cycle missing either (pending-sample discipline). In
    ``close_to_close`` mode the first joined cycle has no prior close for either
    session and is dropped here. The DatetimeIndex (cycle date) is required by the
    ``weekday`` slicer.
    """
    columns = ["s1_green", "s2_green", "s1_size"]
    empty = pd.DataFrame(columns=columns)

    if candles_df.empty:
      return empty

    df = candles_df.sort_values("timestamp").reset_index(drop=True)
    df["_mod"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["_date"] = df["timestamp"].dt.normalize()

    s1 = self._session_table(df, self.s1_start_min, self.s1_end_min)
    s2 = self._session_table(df, self.s2_start_min, self.s2_end_min)
    if s1.empty or s2.empty:
      return empty

    day = pd.DataFrame(
      {
        "s1_open": s1["open"],
        "s1_close": s1["close"],
        "s1_high": s1["high"],
        "s1_low": s1["low"],
      }
    ).join(
      pd.DataFrame({"s2_open": s2["open"], "s2_close": s2["close"]}),
      how="inner",
    )
    if day.empty:
      return empty

    day = day.sort_index()
    day["s1_size"] = day["s1_high"] - day["s1_low"]

    if self.performance == "open_to_close":
      day["s1_green"] = day["s1_close"] >= day["s1_open"]
      day["s2_green"] = day["s2_close"] >= day["s2_open"]
    else:  # close_to_close
      s1_prev = day["s1_close"].shift(1)
      s2_prev = day["s2_close"].shift(1)
      # First joined cycle has no prior close: excluded (pending discipline).
      day = day[s1_prev.notna() & s2_prev.notna()].copy()
      day["s1_green"] = day["s1_close"] >= s1_prev.loc[day.index]
      day["s2_green"] = day["s2_close"] >= s2_prev.loc[day.index]

    return day[columns]

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the four matrix rows (session 1 green/red x session 2 green/red).

    Only countable cycles (both colors present) count. The two outcomes per
    condition partition the cycles carrying that session 1 color. If
    ``baseline_rows`` is provided, merges its ``probability`` / ``total`` into each
    row's ``baseline_prob`` / ``baseline_n``.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    if len(day_table) > 0:
      s1_green = day_table["s1_green"]
      s2_green = day_table["s2_green"].astype(bool)
      countable = s1_green.notna() & day_table["s2_green"].notna()
      s1_is_green = s1_green.fillna(False).astype(bool)
    else:
      s2_green = pd.Series([], dtype=bool)
      countable = pd.Series([], dtype=bool)
      s1_is_green = pd.Series([], dtype=bool)

    rows: list[StatResultRow] = []
    for cond_key, cond_is_green in _CONDITIONS:
      cond_mask = countable & (s1_is_green == cond_is_green)
      total = int(cond_mask.sum())
      for out_key, out_is_green in _OUTCOMES:
        out_match = s2_green if out_is_green else ~s2_green
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
    """Random baseline: independent green/red colors (p=0.5) destroy correlation.

    Each cycle's session 1 and session 2 are recolored independently at random, so
    session 2's color is independent of session 1's. Expected baseline_prob ~ 0.5
    for every row. Deterministic for a fixed seed (``np.random.default_rng``).
    """
    rng = np.random.default_rng(seed)
    n = len(day_table)

    tmp = day_table.copy()
    tmp["s1_green"] = rng.integers(0, 2, size=n).astype(bool)
    tmp["s2_green"] = rng.integers(0, 2, size=n).astype(bool)
    return self.compute_rows(tmp, baseline_rows=None)


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
  session1: str = "london",
  session2: str = "ny",
  performance: str = "close_to_close",
) -> Path:
  """Load data and compute Market Session Correlation for the daily timeframe.

  Writes the result JSON to results/. The file is keyed on the stat family name
  only (``market_session_correlation.json``), so re-running with a different
  session pair or performance basis OVERWRITES the previous result for the
  instrument.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  stat = MarketSessionCorrelation(
    instrument=instrument,
    config=config,
    session1=session1,
    session2=session2,
    performance=performance,
  )
  result = stat.compute(candles_df)

  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Market Session Correlation stat")
  parser.add_argument("--instrument", default="NQ", help="Instrument name (default: NQ)")
  parser.add_argument("--data-path", default=None, help="Override parquet file path")
  parser.add_argument("--config-dir", default="config", help="Config directory (default: config)")
  parser.add_argument("--session1", default="london", help="First session name (default: london)")
  parser.add_argument("--session2", default="ny", help="Second session name (default: ny)")
  parser.add_argument(
    "--performance",
    default="close_to_close",
    choices=list(_PERFORMANCE_MODES),
    help="Session direction basis (default: close_to_close)",
  )
  args = parser.parse_args()

  output_path = run(
    instrument=args.instrument,
    config_dir=args.config_dir,
    data_path=args.data_path,
    session1=args.session1,
    session2=args.session2,
    performance=args.performance,
  )

  print(f"Written: {output_path}")

  with open(output_path, encoding="utf-8") as f:
    data = json.load(f)

  instrument_data = data["instruments"][args.instrument]
  for tf, tf_data in instrument_data.items():
    print(f"\n  {tf}: {tf_data['total_samples']} countable cycles | {tf_data['data_range']}")
    for row in tf_data["results"]:
      print(
        f"    {row['condition']} -> {row['outcome']}: "
        f"{row['probability']:.3f} (N={row['total']}, baseline={row['baseline_prob']:.3f})"
      )
