"""Market Session Breakout stat — standard variant.

Measures how a SECOND market session breaks the high/low range formed by a FIRST
market session: does session 2 break above session 1's high only, below its low
only, both sides, or neither?

Both sessions are named windows read from the instrument config (the single
source of truth for their bounds) — e.g. ``london`` → ``ny`` (the default). This
is the generic, configurable sibling of ``overnight_range_breakout`` (which
hardwires overnight → RTH); here any ordered pair of configured sessions works.

Definitions:
  - **Cycle**: bars are attributed to the RTH session date they belong to. A
    session that crosses midnight (``start >= end``, e.g. ``asia`` 18:00→03:00)
    has its evening bars (at or after ``start``) attributed to the NEXT calendar
    day's cycle and its early bars (before ``end``) to the same day's cycle. An
    intraday session (``start < end``) is attributed to its own calendar date.
    Session 1 and session 2 are joined on this shared cycle date, so the chosen
    pair must be ordered within a cycle (session 1 earlier than session 2).
  - **Session 1 range**: ``s1_high`` / ``s1_low`` are the max high / min low over
    session 1's bars; ``s1_size`` is their difference.
  - **Break**: how session 2's price relates to the session 1 range. With
    ``breakout_criteria = "wick"`` (default) a break uses session 2's intraday
    extremes (a wick beyond the level counts); with ``"close"`` a break requires a
    bar to CLOSE beyond the level (session 2's extreme close).

Four MUTUALLY EXCLUSIVE outcomes partition every countable cycle exhaustively
(strict inequality defines a break — touching the level exactly is NOT a break,
mirroring the sibling ``overnight_range_breakout`` convention):
  - ``broke_high`` — broke above ``s1_high`` only (high break, low held)
  - ``broke_low``  — broke below ``s1_low`` only (low break, high held)
  - ``broke_both`` — broke both sides during session 2
  - ``neither``    — stayed entirely within the session 1 range

Reported under a single ``session1_range`` condition; the four outcomes sum to the
countable cycle count, so their probabilities sum to 1. ``total_samples`` counts
ALL countable cycles; a cycle is countable only when BOTH sessions are resolved
(a clean open bar and enough end-of-session coverage) for that cycle.

Declared slices re-run the whole computation per subset:
  - ``weekday``    — the "by weekday" breakdown.
  - ``size``       — quartile buckets of the session 1 range size.
  - ``levels``     — extension past the broken level in multiples of ``s1_size``.
  - ``rejection``  — split by which extreme of session 1 formed first (high/low).
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
  Levels,
  Rejection,
  SizeBucket,
  StatResultRow,
  write_results,
)
from stats.config import InstrumentConfig, load_config, minute_of_day

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="Market Session Breakout",
  fr="Cassure de session de marché",
)
_DEFINITION = I18nString(
  en="Comparing a second market session's high and low to the range a first market session formed, how often does price break above the first session's high only, below its low only, both sides, or neither during the second session?",
  fr="En comparant le plus haut et le plus bas d'une deuxième session de marché au range formé par une première session, à quelle fréquence le prix casse-t-il au-dessus du plus haut de la première session uniquement, en-dessous de son plus bas uniquement, des deux côtés, ou ni l'un ni l'autre pendant la deuxième session ?",
)
_LABELS = Labels(
  conditions={
    "session1_range": I18nString(en="Session 1 range", fr="Range de la session 1"),
  },
  outcomes={
    "broke_high": I18nString(
      en="Broke above session 1 high only",
      fr="Cassure au-dessus du plus haut de la session 1 uniquement",
    ),
    "broke_low": I18nString(
      en="Broke below session 1 low only",
      fr="Cassure en-dessous du plus bas de la session 1 uniquement",
    ),
    "broke_both": I18nString(
      en="Broke both sides",
      fr="Cassure des deux côtés",
    ),
    "neither": I18nString(
      en="Stayed within the session 1 range",
      fr="Resté dans le range de la session 1",
    ),
  },
)

# Outcome enumeration, in display order. All four partition the countable cycles.
_OUTCOMES: tuple[str, ...] = ("broke_high", "broke_low", "broke_both", "neither")

_BREAKOUT_CRITERIA: tuple[str, ...] = ("wick", "close")

_MINUTES_PER_DAY = 24 * 60


class MarketSessionBreakout(BaseStat):
  """Breakout-direction distribution of session 2 relative to session 1's range."""

  stat_name = "market_session_breakout"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = (
    "weekday",
    SizeBucket(column="s1_size", preset="quartiles", name="size"),
    Levels(ref="s1_size", ext="extension", name="levels"),
    Rejection(column="high_first", name="rejection"),
  )

  def __init__(
    self,
    instrument: str,
    config: InstrumentConfig,
    session1: str = "london",
    session2: str = "ny",
    breakout_criteria: str = "wick",
    close_tolerance_min: int = 15,
  ) -> None:
    if breakout_criteria not in _BREAKOUT_CRITERIA:
      raise ValueError(
        f"Unsupported breakout_criteria '{breakout_criteria}'. "
        f"Choose from {list(_BREAKOUT_CRITERIA)}"
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
    self.breakout_criteria = breakout_criteria
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
  @staticmethod
  def _session_bars(df: pd.DataFrame, start_min: int, end_min: int) -> pd.DataFrame:
    """Return the session's bars tagged with their cycle date in ``_cycle``.

    Intraday sessions (``start < end``) keep ``[start, end)`` on each calendar
    date. Cross-midnight sessions (``start >= end``) take evening bars (at or
    after ``start``) attributed to the NEXT day's cycle and early bars (before
    ``end``) to the same day's cycle.
    """
    mod = df["_mod"]
    if start_min < end_min:
      bars = df[(mod >= start_min) & (mod < end_min)].copy()
      bars["_cycle"] = bars["_date"]
    else:
      bars = df[(mod >= start_min) | (mod < end_min)].copy()
      evening = bars["_mod"] >= start_min
      bars["_cycle"] = bars["_date"].where(
        ~evening, bars["_date"] + pd.Timedelta(days=1)
      )
    return bars

  def _session_table(
    self, df: pd.DataFrame, start_min: int, end_min: int, ordering: bool
  ) -> pd.DataFrame:
    """Per-cycle aggregates for one session, indexed by the cycle date.

    Columns: ``high``, ``low``, ``close_high``, ``close_low`` (extreme closes),
    plus ``resolved`` (clean open bar AND end-of-session coverage). When
    ``ordering`` is set, also adds ``high_first`` (1.0 if the high was reached
    before the low, 0.0 otherwise, NaN if a single bar held both — undetermined).
    Only resolved cycles are returned.
    """
    bars = self._session_bars(df, start_min, end_min)
    if bars.empty:
      return pd.DataFrame()

    duration = (end_min - start_min) % _MINUTES_PER_DAY or _MINUTES_PER_DAY
    bars["_offset"] = (bars["_mod"] - start_min) % _MINUTES_PER_DAY

    grouped = bars.groupby("_cycle")
    table = pd.DataFrame(
      {
        "high": grouped["high"].max(),
        "low": grouped["low"].min(),
        "close_high": grouped["close"].max(),
        "close_low": grouped["close"].min(),
        "_has_open": grouped["_offset"].min() == 0,
        "_last_offset": grouped["_offset"].max(),
      }
    )
    resolved = table["_has_open"] & (
      table["_last_offset"] >= duration - self.close_tolerance_min
    )
    table = table[resolved]
    if table.empty:
      return table

    if ordering:
      # First occurrence of the per-cycle max high / min low (idxmax/idxmin return
      # the first matching label, and bars are chronologically sorted upstream).
      hi_ts = (
        bars.loc[grouped["high"].idxmax()].set_index("_cycle")["timestamp"]
      )
      lo_ts = (
        bars.loc[grouped["low"].idxmin()].set_index("_cycle")["timestamp"]
      )
      high_first = (hi_ts < lo_ts).astype(float)
      high_first[hi_ts == lo_ts] = np.nan
      table["high_first"] = high_first.reindex(table.index)

    return table.drop(columns=["_has_open", "_last_offset"])

  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-cycle table joining session 1's range and session 2's extremes.

    Columns returned:
      ``s1_high`` / ``s1_low``           — session 1 range extremes.
      ``s1_size``                        — ``s1_high - s1_low`` (read by the size slicer).
      ``s2_high`` / ``s2_low``           — session 2 intraday extremes (wick).
      ``s2_close_high`` / ``s2_close_low`` — session 2 extreme CLOSES (close criteria).
      ``extension``                      — furthest session 2 break past the session 1
                                           range in either direction (levels slicer).
      ``high_first``                     — which session 1 extreme formed first
                                           (rejection slicer); NaN when undetermined.

    A cycle is countable only when BOTH sessions are resolved for that cycle; the
    inner join drops any cycle missing either (pending-sample discipline). The
    DatetimeIndex (cycle date) is required by the ``weekday`` slicer.
    """
    columns = [
      "s1_high",
      "s1_low",
      "s1_size",
      "s2_high",
      "s2_low",
      "s2_close_high",
      "s2_close_low",
      "extension",
      "high_first",
    ]
    empty = pd.DataFrame(columns=columns)

    if candles_df.empty:
      return empty

    # reset_index guarantees unique, positional labels so the idxmax/idxmin +
    # .loc first-touch lookup in _session_table stays correct even if the source
    # parquet carried a non-unique index.
    df = candles_df.sort_values("timestamp").reset_index(drop=True)
    df["_mod"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["_date"] = df["timestamp"].dt.normalize()

    s1 = self._session_table(df, self.s1_start_min, self.s1_end_min, ordering=True)
    s2 = self._session_table(df, self.s2_start_min, self.s2_end_min, ordering=False)
    if s1.empty or s2.empty:
      return empty

    day = pd.DataFrame(
      {
        "s1_high": s1["high"],
        "s1_low": s1["low"],
        "high_first": s1["high_first"],
      }
    ).join(
      pd.DataFrame(
        {
          "s2_high": s2["high"],
          "s2_low": s2["low"],
          "s2_close_high": s2["close_high"],
          "s2_close_low": s2["close_low"],
        }
      ),
      how="inner",
    )
    if day.empty:
      return empty

    day = day.sort_index()
    day["s1_size"] = day["s1_high"] - day["s1_low"]

    # Furthest session 2 break past the session 1 range in either direction, using
    # the same criteria-selected extremes as the break classification.
    high, low = self._break_extremes(day)
    ext_up = (high - day["s1_high"]).clip(lower=0.0)
    ext_down = (day["s1_low"] - low).clip(lower=0.0)
    day["extension"] = pd.concat([ext_up, ext_down], axis=1).max(axis=1)

    return day[columns]

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def _break_extremes(self, day_table: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return the (high, low) session 2 extremes selected by ``breakout_criteria``."""
    if self.breakout_criteria == "close":
      return day_table["s2_close_high"], day_table["s2_close_low"]
    return day_table["s2_high"], day_table["s2_low"]

  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the four breakout-direction rows over the (sliced) day table.

    The four outcomes are mutually exclusive and partition the countable cycles,
    so their counts sum to ``total``. If ``baseline_rows`` is provided, merges its
    ``probability`` / ``total`` into each row's ``baseline_prob`` / ``baseline_n``.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    def _make(outcome: str, count: int, total: int) -> StatResultRow:
      bl = baseline_map.get(("session1_range", outcome))
      return StatResultRow(
        condition="session1_range",
        outcome=outcome,
        count=count,
        total=total,
        probability=count / total if total > 0 else 0.0,
        baseline_prob=bl.probability if bl else 0.0,
        baseline_n=bl.total if bl else 0,
      )

    if day_table.empty:
      return [_make(out, 0, 0) for out in _OUTCOMES]

    s1_high = day_table["s1_high"]
    s1_low = day_table["s1_low"]
    high, low = self._break_extremes(day_table)

    countable = s1_high.notna() & s1_low.notna() & high.notna() & low.notna()
    total = int(countable.sum())

    # Strict inequality defines a break (touching the level exactly is NOT a break).
    above = countable & (high > s1_high)
    below = countable & (low < s1_low)

    counts = {
      "broke_high": int((above & ~below).sum()),
      "broke_low": int((below & ~above).sum()),
      "broke_both": int((above & below).sum()),
      "neither": int((countable & ~above & ~below).sum()),
    }

    return [_make(out, counts[out], total) for out in _OUTCOMES]

  # -------------------------------------------------------------------------
  # Baseline
  # -------------------------------------------------------------------------
  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random directional baseline. Deterministic for a fixed seed.

    Each countable cycle's session 2 move is REFLECTED around the session 1 range
    midpoint with probability 0.5 (a price ``p`` maps to ``2*mid - p``, which swaps
    the high and low extremes). Reflection turns a ``broke_high`` cycle into a
    ``broke_low`` cycle and vice versa, while ``broke_both`` and ``neither`` are
    direction-symmetric and unchanged. The null therefore has NO directional bias:
    ``broke_high`` and ``broke_low`` converge to their shared mean, so the
    comparison reveals whether session 2 breaks UP more often than DOWN beyond a
    coin flip. Uses ``np.random.default_rng(seed)``.
    """
    n = len(day_table)
    if n == 0:
      return self.compute_rows(day_table, baseline_rows=None)

    rng = np.random.default_rng(seed)
    flip = rng.integers(0, 2, size=n).astype(bool)

    mid = (day_table["s1_high"].to_numpy() + day_table["s1_low"].to_numpy()) / 2.0
    tmp = day_table.copy()

    # Reflect every session 2 column pair (high <-> low) around the session 1
    # midpoint, then keep the reflected value only on flipped cycles. Both the wick
    # and close extremes are reflected so the chosen breakout_criteria is consistent.
    for hi_col, lo_col in (("s2_high", "s2_low"), ("s2_close_high", "s2_close_low")):
      hi = day_table[hi_col].to_numpy()
      lo = day_table[lo_col].to_numpy()
      refl_hi = 2.0 * mid - lo
      refl_lo = 2.0 * mid - hi
      tmp[hi_col] = np.where(flip, refl_hi, hi)
      tmp[lo_col] = np.where(flip, refl_lo, lo)

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
  breakout_criteria: str = "wick",
) -> Path:
  """Load data and compute Market Session Breakout for the daily timeframe.

  Writes the result JSON to results/. The file is keyed on the stat family name
  only (``market_session_breakout.json``), so re-running with a different session
  pair OVERWRITES the previous result — same convention as ``breakout_criteria``.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  stat = MarketSessionBreakout(
    instrument=instrument,
    config=config,
    session1=session1,
    session2=session2,
    breakout_criteria=breakout_criteria,
  )
  result = stat.compute(candles_df)

  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Market Session Breakout stat")
  parser.add_argument("--instrument", default="NQ", help="Instrument name (default: NQ)")
  parser.add_argument("--data-path", default=None, help="Override parquet file path")
  parser.add_argument("--config-dir", default="config", help="Config directory (default: config)")
  parser.add_argument("--session1", default="london", help="First session name (default: london)")
  parser.add_argument("--session2", default="ny", help="Second session name (default: ny)")
  parser.add_argument(
    "--breakout-criteria",
    default="wick",
    choices=list(_BREAKOUT_CRITERIA),
    help="Break detection: 'wick' (intraday extreme) or 'close' (bar close) (default: wick)",
  )
  args = parser.parse_args()

  output_path = run(
    instrument=args.instrument,
    config_dir=args.config_dir,
    data_path=args.data_path,
    session1=args.session1,
    session2=args.session2,
    breakout_criteria=args.breakout_criteria,
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
