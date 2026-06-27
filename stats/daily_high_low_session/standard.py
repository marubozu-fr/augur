"""Daily High / Low Session stat — standard variant.

Measures which intraday market session produced the day's HIGH and which
produced the day's LOW within each 24-hour trading cycle.  The configured
sessions (default: ``asia``, ``london``, ``ny``) partition the 24-hour cycle;
together they cover the complete day with no gap between them (the 16:15–18:00
CME break carries no NQ trading and lies outside every session window).  For
each countable cycle, the stat attributes each daily extreme to exactly one
session, so the probability distribution over sessions sums to 1 within each
condition.

Two conditions are reported:
  - ``daily_high``: which session contained the cycle's max high.  When two
    sessions share an identical extreme value, the session listed first in
    ``self.sessions`` (chronologically earliest) is credited.
  - ``daily_low``:  which session contained the cycle's min low (same
    tie-break rule).

**Cycle attribution**: every bar is tagged to the RTH session date it belongs
to.  A cross-midnight session (``start >= end``, e.g. ``asia`` 18:00→03:00)
attributes its evening bars to the NEXT calendar day's cycle and its early bars
to the SAME day's cycle.  An intraday session (``start < end``) maps to its
own calendar date.  All configured sessions are joined on this shared cycle
date.

A cycle is **countable** only when ALL configured sessions are resolved (a
clean open bar AND end-of-session coverage within ``close_tolerance_min``
minutes) for that cycle.  If any session is missing or unresolved the entire
cycle is dropped from the denominator (pending-sample discipline).

Declared slices re-run the whole computation per subset:
  - ``weekday`` — the "by weekday" breakdown (shared ``Weekday`` slicer).
  - ``candle``  — split by daily candle color (green = ``cycle_close >=
    cycle_open``, red otherwise).  Implemented via the module-local
    ``_DailyCandle`` slicer, a lightweight subclass of the public ``Close``
    slicer from ``stats/base.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stats.base import (
  BaseStat,
  Close,
  I18nString,
  Labels,
  StatResultRow,
  write_results,
)
from stats.config import InstrumentConfig, load_config, minute_of_day
from stats.utils.session_candles import session_bars

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="Daily High / Low Session",
  fr="Session du plus haut / plus bas journalier",
)
_DEFINITION = I18nString(
  en=(
    "For each trading cycle, which intraday market session produced the day's"
    " highest high and which produced the day's lowest low? Reported as a"
    " probability distribution over the configured sessions; probabilities sum"
    " to 1 within each condition."
  ),
  fr=(
    "Pour chaque cycle de trading, quelle session de marché intrajournalière a"
    " produit le plus haut du jour et laquelle a produit le plus bas du jour ?"
    " Présenté comme une distribution de probabilité sur les sessions configurées ;"
    " les probabilités somment à 1 par condition."
  ),
)

# Known session display names.  Unknown session keys fall back to the raw key
# for both ``en`` and ``fr`` (see __init__ label construction below).
_SESSION_LABELS: dict[str, I18nString] = {
  "asia": I18nString(en="Asia", fr="Asie"),
  "london": I18nString(en="London", fr="Londres"),
  "ny": I18nString(en="New York", fr="New York"),
}

_MINUTES_PER_DAY = 24 * 60


# ---------------------------------------------------------------------------
# Local slicer
# ---------------------------------------------------------------------------
class _DailyCandle(Close):
  """Split the day table by daily candle color (green / red).

  Reads the ``day_green`` boolean column added by ``build_day_table``, where
  ``True`` means the cycle's overall close is at or above its overall open
  (open of the earliest bar across all sessions, close of the latest bar).
  """

  def __init__(self) -> None:
    super().__init__(column="day_green", name="candle")

  def dimension_label(self) -> I18nString:
    return I18nString(
      en="Daily candle color",
      fr="Couleur de la bougie journalière",
    )


# ---------------------------------------------------------------------------
# Stat class
# ---------------------------------------------------------------------------
class DailyHighLowSession(BaseStat):
  """Distribution of the sessions that produced the daily high and low."""

  stat_name = "daily_high_low_session"
  title = _TITLE
  definition = _DEFINITION
  # ``labels`` is overridden per-instance in ``__init__`` because the
  # ``outcomes`` dict depends on the configured session list.
  slices = ("weekday", _DailyCandle())

  def __init__(
    self,
    instrument: str,
    config: InstrumentConfig,
    sessions: tuple[str, ...] | list[str] = ("asia", "london", "ny"),
    close_tolerance_min: int = 15,
  ) -> None:
    """
    Parameters
    ----------
    instrument:
        Instrument name (e.g. ``"NQ"``).
    config:
        Loaded instrument configuration (from ``load_config``).
    sessions:
        Ordered sequence of session names to consider.  Must contain at
        least two distinct names all present in ``config.sessions``.
        **Chronological order matters**: on exact ties (two sessions share
        the same extreme value) the first session in this sequence is
        credited.  The default order ``("asia", "london", "ny")`` is
        already chronological within a 24-hour cycle.
    close_tolerance_min:
        A session is considered resolved when its last bar falls within
        this many minutes of the session's scheduled end (default 15).
    """
    if len(sessions) < 2:
      raise ValueError(
        f"At least 2 sessions are required, got {len(sessions)}: {list(sessions)}"
      )
    seen: set[str] = set()
    for name in sessions:
      if name in seen:
        raise ValueError(f"Duplicate session name: '{name}'")
      seen.add(name)
      if name not in config.sessions:
        raise ValueError(
          f"Unknown session '{name}'. Configured sessions: {list(config.sessions)}"
        )

    self.instrument = instrument
    self.timeframe = "daily"
    self.config = config
    self.sessions: tuple[str, ...] = tuple(sessions)
    self.close_tolerance_min = close_tolerance_min

    # Precompute session start / end in minutes since midnight.
    self.session_start_mins: dict[str, int] = {}
    self.session_end_mins: dict[str, int] = {}
    for name in self.sessions:
      s = config.sessions[name]
      self.session_start_mins[name] = minute_of_day(s.start)
      self.session_end_mins[name] = minute_of_day(s.end)

    # Build instance-level labels; outcomes depend on the session list.
    outcomes: dict[str, I18nString] = {
      s: _SESSION_LABELS.get(s, I18nString(en=s, fr=s)) for s in self.sessions
    }
    self.labels = Labels(
      conditions={
        "daily_high": I18nString(en="Day's high", fr="Plus haut du jour"),
        "daily_low": I18nString(en="Day's low", fr="Plus bas du jour"),
      },
      outcomes=outcomes,
    )

  # -------------------------------------------------------------------------
  # Day table construction
  # -------------------------------------------------------------------------
  def _session_table(
    self,
    df: pd.DataFrame,
    start_min: int,
    end_min: int,
  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-cycle aggregates for one session, indexed by the cycle date.

    Returns ``(table, resolved_bars)`` where:
      - ``table`` has columns ``high`` (max), ``low`` (min), ``n`` (bar
        count) for resolved cycles only.
      - ``resolved_bars`` is the subset of session bars belonging to those
        resolved cycles (used to compute the cycle-level open / close in
        ``build_day_table``).

    A cycle is **resolved** when the session has a clean open bar (offset
    0 from ``start_min``) AND its last bar falls within
    ``close_tolerance_min`` minutes of the session's scheduled end.
    Returns a pair of empty DataFrames when no bars or no resolved cycles
    exist.
    """
    bars = session_bars(df, start_min, end_min)
    if bars.empty:
      return pd.DataFrame(), pd.DataFrame()

    duration = (end_min - start_min) % _MINUTES_PER_DAY or _MINUTES_PER_DAY
    bars["_offset"] = (bars["_mod"] - start_min) % _MINUTES_PER_DAY

    grouped = bars.groupby("_cycle")
    table = pd.DataFrame(
      {
        "high": grouped["high"].max(),
        "low": grouped["low"].min(),
        "n": grouped["_offset"].count(),
        "_has_open": grouped["_offset"].min() == 0,
        "_last_offset": grouped["_offset"].max(),
      }
    )
    resolved = table["_has_open"] & (
      table["_last_offset"] >= duration - self.close_tolerance_min
    )
    table = table[resolved].drop(columns=["_has_open", "_last_offset"])
    if table.empty:
      return table, pd.DataFrame()

    resolved_bars = bars[bars["_cycle"].isin(table.index)]
    return table, resolved_bars

  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-cycle table attributing the daily high / low to sessions.

    Columns returned (in fixed order):
      ``{session}_high``  — per-session max high (one column per session).
      ``{session}_low``   — per-session min low (one column per session).
      ``high_session``    — name of the session that produced the cycle's
                            highest high; first session in ``self.sessions``
                            order wins on exact ties.
      ``low_session``     — name of the session that produced the cycle's
                            lowest low; first session wins on ties.
      ``n_{session}``     — bar count for that session in the cycle; used by
                            the duration-weighted baseline.
      ``day_green``       — bool; ``True`` when the cycle's overall close
                            is at or above its overall open (open = first bar
                            across all sessions; close = last bar).

    A cycle is countable only when EVERY session is resolved for it; the
    inner join across all session tables drops any cycle missing at least one
    resolved session (pending-sample discipline).  The DatetimeIndex (cycle
    date) is required by the ``weekday`` slicer.
    """
    columns: list[str] = (
      [f"{s}_high" for s in self.sessions]
      + [f"{s}_low" for s in self.sessions]
      + ["high_session", "low_session"]
      + [f"n_{s}" for s in self.sessions]
      + ["day_green"]
    )
    empty = pd.DataFrame(columns=columns)

    if candles_df.empty:
      return empty

    # reset_index guarantees unique positional labels so idxmin / idxmax
    # lookups in _session_table remain correct even if the source parquet
    # carried a non-unique index.
    df = candles_df.sort_values("timestamp").reset_index(drop=True)
    df["_mod"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["_date"] = df["timestamp"].dt.normalize()

    # Build per-session resolved tables and collect their resolved bars.
    session_tbls: dict[str, pd.DataFrame] = {}
    session_res_bars: dict[str, pd.DataFrame] = {}
    for sess in self.sessions:
      tbl, res_bars = self._session_table(
        df,
        self.session_start_mins[sess],
        self.session_end_mins[sess],
      )
      if tbl.empty:
        return empty
      session_tbls[sess] = tbl
      session_res_bars[sess] = res_bars

    # Inner-join all sessions: a cycle is countable only when every session
    # is resolved for it.
    first, rest = self.sessions[0], self.sessions[1:]
    day = session_tbls[first].rename(
      columns={
        "high": f"{first}_high",
        "low": f"{first}_low",
        "n": f"n_{first}",
      }
    )
    for sess in rest:
      tbl = session_tbls[sess].rename(
        columns={
          "high": f"{sess}_high",
          "low": f"{sess}_low",
          "n": f"n_{sess}",
        }
      )
      day = day.join(tbl, how="inner")

    if day.empty:
      return empty

    day = day.sort_index()

    # Attribute the cycle's high / low to a session.  np.argmax / np.argmin
    # return the FIRST occurrence of the extreme, so exact ties resolve in
    # favour of the earliest-listed (chronologically first) session.
    high_cols = [f"{s}_high" for s in self.sessions]
    low_cols = [f"{s}_low" for s in self.sessions]
    sessions_arr = np.array(self.sessions)
    day["high_session"] = sessions_arr[day[high_cols].to_numpy().argmax(axis=1)]
    day["low_session"] = sessions_arr[day[low_cols].to_numpy().argmin(axis=1)]

    # Cycle open / close: first and last bars across all sessions in the
    # countable cycles.  Sessions do not overlap, so concat + sort by
    # timestamp yields a monotone price path; idxmin / idxmax pick the
    # opening and closing bars of the full cycle.
    countable = set(day.index)
    all_bars = pd.concat(
      [b[b["_cycle"].isin(countable)] for b in session_res_bars.values()]
    ).sort_values("timestamp")
    grp = all_bars.groupby("_cycle")
    cycle_open = (
      all_bars.loc[grp["timestamp"].idxmin()].set_index("_cycle")["open"]
    )
    cycle_close = (
      all_bars.loc[grp["timestamp"].idxmax()].set_index("_cycle")["close"]
    )
    day["day_green"] = (cycle_close >= cycle_open).reindex(day.index)

    return day[columns]

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the high-session and low-session distribution rows.

    Two conditions (``daily_high``, ``daily_low``) each yield one row per
    configured session.  Within each condition the counts partition all
    countable cycles (they sum to ``total``), so probabilities sum to 1.
    If ``baseline_rows`` is provided, merges its ``probability`` / ``total``
    into each row's ``baseline_prob`` / ``baseline_n``.

    Row order: all ``daily_high`` outcomes in ``self.sessions`` order, then
    all ``daily_low`` outcomes in the same order.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    def _make(condition: str, outcome: str, count: int, total: int) -> StatResultRow:
      bl = baseline_map.get((condition, outcome))
      return StatResultRow(
        condition=condition,
        outcome=outcome,
        count=count,
        total=total,
        probability=count / total if total > 0 else 0.0,
        baseline_prob=bl.probability if bl else 0.0,
        baseline_n=bl.total if bl else 0,
      )

    if day_table.empty:
      rows: list[StatResultRow] = []
      for sess in self.sessions:
        rows.append(_make("daily_high", sess, 0, 0))
      for sess in self.sessions:
        rows.append(_make("daily_low", sess, 0, 0))
      return rows

    total = len(day_table)
    high_col = day_table["high_session"]
    low_col = day_table["low_session"]

    rows = []
    for sess in self.sessions:
      rows.append(_make("daily_high", sess, int((high_col == sess).sum()), total))
    for sess in self.sessions:
      rows.append(_make("daily_low", sess, int((low_col == sess).sum()), total))
    return rows

  # -------------------------------------------------------------------------
  # Baseline
  # -------------------------------------------------------------------------
  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Duration-weighted random-session baseline.  Deterministic for a fixed seed.

    For each countable cycle, the day's high (and independently the day's
    low) is assigned to a randomly drawn session with probability proportional
    to that session's bar count ``n_{session}`` in the cycle.  Rationale:
    under a structureless random walk, a price extreme is equally likely at
    any minute, so a session's expected share of the extreme equals its share
    of total bars (time).  The stat's edge is whether a session captures the
    high or low MORE than its duration share predicts.

    Implementation: build a per-row normalized weight matrix from the
    ``n_{session}`` columns, compute cumulative sums, then draw
    ``rng.random(n)`` for the high and again for the low.  The first session
    whose cumulative weight exceeds the draw is selected
    (``(u[:, None] < cum).argmax(axis=1)``).  A tmp copy of ``day_table``
    with the replaced ``high_session`` / ``low_session`` columns is passed to
    ``compute_rows``.  Uses ``np.random.default_rng(seed)``.
    """
    n = len(day_table)
    if n == 0:
      return self.compute_rows(day_table, baseline_rows=None)

    rng = np.random.default_rng(seed)

    n_cols = [f"n_{s}" for s in self.sessions]
    weights = day_table[n_cols].to_numpy().astype(float)
    row_sums = weights.sum(axis=1, keepdims=True)
    # Guard against all-zero rows (should not occur in practice): fall back to
    # uniform weights so we never divide by zero.
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    weights /= row_sums
    cum = weights.cumsum(axis=1)

    sessions_arr = np.array(self.sessions)

    # Independent draws for high and low; same RNG state → jointly deterministic.
    u_high = rng.random(n)
    high_idx = (u_high[:, None] < cum).argmax(axis=1)

    u_low = rng.random(n)
    low_idx = (u_low[:, None] < cum).argmax(axis=1)

    tmp = day_table.copy()
    tmp["high_session"] = sessions_arr[high_idx]
    tmp["low_session"] = sessions_arr[low_idx]

    return self.compute_rows(tmp, baseline_rows=None)


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
  sessions: tuple[str, ...] = ("asia", "london", "ny"),
) -> Path:
  """Load data and compute Daily High / Low Session for the daily timeframe.

  Writes the result JSON to ``results/``.  The file is keyed on the stat
  family name only (``daily_high_low_session.json``), so re-running with a
  different session list OVERWRITES the previous result for the instrument.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  stat = DailyHighLowSession(
    instrument=instrument,
    config=config,
    sessions=sessions,
  )
  result = stat.compute(candles_df)

  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Daily High / Low Session stat")
  parser.add_argument("--instrument", default="NQ", help="Instrument name (default: NQ)")
  parser.add_argument("--data-path", default=None, help="Override parquet file path")
  parser.add_argument(
    "--config-dir", default="config", help="Config directory (default: config)"
  )
  parser.add_argument(
    "--sessions",
    nargs="+",
    default=["asia", "london", "ny"],
    help="Ordered session names to attribute (default: asia london ny)",
  )
  args = parser.parse_args()

  output_path = run(
    instrument=args.instrument,
    config_dir=args.config_dir,
    data_path=args.data_path,
    sessions=tuple(args.sessions),
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
