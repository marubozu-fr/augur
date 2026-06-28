"""Asian Range Breakout stat — standard variant.

Measures how each RTH session's market-hours high/low relate to the PRIOR
Asian (Tokyo) session's high/low: does price break above the Asian high only,
below the Asian low only, both sides, or neither during the regular session?

Definitions:
  - **Asian range**: the high-low range of the Asian session that precedes a
    given RTH session, ``[asia_start (prev day), asia_end)``
    (18:00 ET → 03:00 ET for NQ). ``ar_high`` is the max high and ``ar_low``
    the min low over that window; ``ar_size`` is ``ar_high - ar_low``. The
    Asian session crosses midnight, so each bar is attributed to the RTH session
    date it PRECEDES via ``session_bars`` from
    ``stats/utils/session_candles.py``: evening bars (at or after
    ``asia_start``) are tagged to the NEXT calendar day's cycle and early bars
    (before ``asia_end``) to the same calendar day's cycle. Grouping the tagged
    bars by ``_cycle`` and taking the max high / min low gives the per-RTH-day
    Asian range.
  - **Breakout window**: the whole RTH session, ``[rth_start, rth_end)``.
  - **Break**: how the RTH session's price relates to the Asian range. With
    ``breakout_criteria = "wick"`` (default) a break uses the session's intraday
    extremes (a wick beyond the level counts); with ``"close"`` a break requires
    a bar to CLOSE beyond the level (the session's extreme close).

Four MUTUALLY EXCLUSIVE outcomes partition every countable day exhaustively
(strict inequality defines a break — touching the level exactly is NOT a break,
mirroring the sibling ``overnight_range_breakout`` / ``opening_range_breakout``
convention):
  - ``broke_high`` — broke above ``ar_high`` only (high break, low held)
  - ``broke_low``  — broke below ``ar_low`` only (low break, high held)
  - ``broke_both`` — broke both sides during the session
  - ``neither``    — stayed entirely within the Asian range all session

Reported under a single ``asian_range`` condition; the four outcomes sum to the
countable day count, so their probabilities sum to 1. ``total_samples`` counts
ALL resolved days; each row's ``total`` is the countable days (a resolved RTH
day that also has a prior Asian range).

Declared slices re-run the whole computation per subset:
  - ``weekday``     — the "by weekday" breakdown.
  - ``close``       — split by session close color (green/red).
  - ``prev_candle`` — split by the prior session's color (green/red).
  - ``size``        — quartile buckets of the Asian-range size.
  - ``levels``      — extension past the broken level in multiples of ``ar_size``.
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
  SizeBucket,
  StatResultRow,
  write_results,
)
from stats.config import InstrumentConfig, load_config, minute_of_day
from stats.utils.daily_candles import build_resolved_days
from stats.utils.session_candles import session_bars

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="Asian Range Breakout",
  fr="Cassure du range asiatique",
)
_DEFINITION = I18nString(
  en="Comparing each session's market-hours high and low to the prior Asian (Tokyo) session's high and low, how often does price break above the Asian high only, below the Asian low only, both sides, or neither during the RTH session?",
  fr="En comparant le plus haut et le plus bas de la séance régulière au plus haut et au plus bas de la session asiatique (Tokyo) précédente, à quelle fréquence le prix casse-t-il au-dessus du plus haut asiatique uniquement, en-dessous du plus bas uniquement, des deux côtés, ou ni l'un ni l'autre pendant la séance RTH ?",
)
_LABELS = Labels(
  conditions={
    "asian_range": I18nString(en="Asian range", fr="Range asiatique"),
  },
  outcomes={
    "broke_high": I18nString(
      en="Broke above Asian high only",
      fr="Cassure au-dessus du plus haut asiatique uniquement",
    ),
    "broke_low": I18nString(
      en="Broke below Asian low only",
      fr="Cassure en-dessous du plus bas asiatique uniquement",
    ),
    "broke_both": I18nString(
      en="Broke both sides",
      fr="Cassure des deux côtés",
    ),
    "neither": I18nString(
      en="Stayed within the Asian range",
      fr="Resté dans le range asiatique",
    ),
  },
)

# Outcome enumeration, in display order. All four partition the countable days.
_OUTCOMES: tuple[str, ...] = ("broke_high", "broke_low", "broke_both", "neither")

_BREAKOUT_CRITERIA: tuple[str, ...] = ("wick", "close")


class AsianRangeBreakout(BaseStat):
  """Breakout-direction distribution of the RTH session relative to the prior
  Asian (Tokyo) session range."""

  stat_name = "asian_range_breakout"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = (
    "weekday",
    "close",
    "prev_candle",
    SizeBucket(column="ar_size", preset="quartiles", name="size"),
    Levels(ref="ar_size", ext="extension", name="levels"),
  )

  def __init__(
    self,
    instrument: str,
    config: InstrumentConfig,
    breakout_criteria: str = "wick",
    close_tolerance_min: int = 15,
  ) -> None:
    if breakout_criteria not in _BREAKOUT_CRITERIA:
      raise ValueError(
        f"Unsupported breakout_criteria '{breakout_criteria}'. "
        f"Choose from {list(_BREAKOUT_CRITERIA)}"
      )
    self.instrument = instrument
    self.timeframe = "daily"
    self.config = config
    self.breakout_criteria = breakout_criteria
    self.close_tolerance_min = close_tolerance_min

    rth = config.sessions["rth"]
    self.rth_start_min: int = minute_of_day(rth.start)
    self.rth_end_min: int = minute_of_day(rth.end)

    asia = config.sessions["asia"]
    self.asia_start_min: int = minute_of_day(asia.start)
    self.asia_end_min: int = minute_of_day(asia.end)

  # -------------------------------------------------------------------------
  # Day table construction
  # -------------------------------------------------------------------------
  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-session table with the Asian range and RTH extremes.

    Columns returned:
      ``ar_high`` / ``ar_low``               — prior Asian session extremes.
      ``ar_size``                            — ``ar_high - ar_low`` (size slicer).
      ``day_high`` / ``day_low``             — RTH session intraday extremes (wick).
      ``day_close_high`` / ``day_close_low`` — RTH session extreme CLOSES (close
                                               criteria); max / min of RTH bar closes.
      ``extension``                          — furthest breakout past the Asian range
                                               in either direction (levels slicer).
      ``session_green``                      — session color (``session_close >= session_open``),
                                               read by the ``close`` slicer.
      ``prev_session_green``                 — prior session color (NaN for the first resolved
                                               day), read by the ``prev_candle`` slicer.

    The resolved-days core (RTH filter, session open/close, resolution filter,
    chronological sort) is shared via ``build_resolved_days``; the Asian-range
    and RTH-extreme aggregates are joined per day. A day is countable only when
    it is a resolved RTH day AND has a prior Asian range; days missing either
    drop out via the inner join (pending-sample discipline). The DatetimeIndex
    (normalized session date) is required by the ``weekday`` slicer.

    ``session_bars`` handles the cross-midnight Asian session attribution: evening
    bars (at or after ``asia_start``) receive ``_cycle = _date + 1 day`` (the
    next RTH session date) and early bars (before ``asia_end``) receive
    ``_cycle = _date`` (the same calendar day's RTH session date). Grouping by
    ``_cycle`` then yields the Asian range for each RTH session.
    """
    columns = [
      "ar_high",
      "ar_low",
      "ar_size",
      "day_high",
      "day_low",
      "day_close_high",
      "day_close_low",
      "extension",
      "session_green",
      "prev_session_green",
    ]
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

    # Asian range aggregates via session_bars, which handles the cross-midnight
    # attribution: evening bars (>= asia_start) are tagged _cycle = _date + 1 day;
    # early bars (< asia_end) are tagged _cycle = _date. The _cycle column is the
    # RTH session date the Asian bars precede.
    asia = session_bars(df, self.asia_start_min, self.asia_end_min)
    ar_high = asia.groupby("_cycle")["high"].max().rename("ar_high")
    ar_low = asia.groupby("_cycle")["low"].min().rename("ar_low")

    # RTH session extremes from the same RTH bar filter the resolution uses.
    rth_mask = (df["_mod"] >= self.rth_start_min) & (df["_mod"] < self.rth_end_min)
    rth = df[rth_mask]
    day_high = rth.groupby("_date")["high"].max().rename("day_high")
    day_low = rth.groupby("_date")["low"].min().rename("day_low")
    day_close_high = rth.groupby("_date")["close"].max().rename("day_close_high")
    day_close_low = rth.groupby("_date")["close"].min().rename("day_close_low")

    # Inner join onto the resolved index: a day survives only with an RTH session
    # AND a prior Asian range.
    day = (
      resolved.join(ar_high, how="inner")
      .join(ar_low, how="inner")
      .join(day_high, how="inner")
      .join(day_low, how="inner")
      .join(day_close_high, how="inner")
      .join(day_close_low, how="inner")
    )
    if day.empty:
      return empty

    day["ar_size"] = day["ar_high"] - day["ar_low"]
    day["session_green"] = day["session_close"] >= day["session_open"]
    # Prior RESOLVED session's color (shift over the resolved-only, sorted index,
    # so it skips any excluded/early-close day). NaN for the first resolved day.
    day["prev_session_green"] = day["session_green"].shift(1)

    # Furthest breakout past the Asian range in either direction, using the
    # same criteria-selected extremes as the break classification (wick or close).
    high, low = self._break_extremes(day)
    ext_up = (high - day["ar_high"]).clip(lower=0.0)
    ext_down = (day["ar_low"] - low).clip(lower=0.0)
    day["extension"] = pd.concat([ext_up, ext_down], axis=1).max(axis=1)

    return day[columns]

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def _break_extremes(self, day_table: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return the (high, low) RTH extremes selected by ``breakout_criteria``."""
    if self.breakout_criteria == "close":
      return day_table["day_close_high"], day_table["day_close_low"]
    return day_table["day_high"], day_table["day_low"]

  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the four breakout-direction rows over the (sliced) day table.

    Only days with a valid Asian range (``ar_high`` not NaN) are countable. The
    four outcomes are mutually exclusive and partition the countable days, so
    their counts sum to ``total``. If ``baseline_rows`` is provided, merges its
    ``probability`` / ``total`` into each row's ``baseline_prob`` / ``baseline_n``.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    def _make(outcome: str, count: int, total: int) -> StatResultRow:
      bl = baseline_map.get(("asian_range", outcome))
      return StatResultRow(
        condition="asian_range",
        outcome=outcome,
        count=count,
        total=total,
        probability=count / total if total > 0 else 0.0,
        baseline_prob=bl.probability if bl else 0.0,
        baseline_n=bl.total if bl else 0,
      )

    if day_table.empty:
      return [_make(out, 0, 0) for out in _OUTCOMES]

    ar_high = day_table["ar_high"]
    ar_low = day_table["ar_low"]
    high, low = self._break_extremes(day_table)

    countable = ar_high.notna() & ar_low.notna() & high.notna() & low.notna()
    total = int(countable.sum())

    # Strict inequality defines a break (touching the level exactly is NOT a break).
    above = countable & (high > ar_high)
    below = countable & (low < ar_low)

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

    Each countable day's RTH move is REFLECTED around its Asian-range midpoint
    with probability 0.5 (a price ``p`` maps to ``2*mid - p``, which swaps the
    high and low extremes). Reflection turns a ``broke_high`` day into a
    ``broke_low`` day and vice versa, while ``broke_both`` and ``neither`` are
    direction-symmetric and unchanged. The null therefore has NO directional
    bias: ``broke_high`` and ``broke_low`` converge to their shared mean, so the
    comparison reveals whether the Asian range breaks UP more often than DOWN
    beyond a coin flip. Uses ``np.random.default_rng(seed)``.
    """
    n = len(day_table)
    if n == 0:
      return self.compute_rows(day_table, baseline_rows=None)

    rng = np.random.default_rng(seed)
    flip = rng.integers(0, 2, size=n).astype(bool)

    mid = (day_table["ar_high"].to_numpy() + day_table["ar_low"].to_numpy()) / 2.0
    tmp = day_table.copy()

    # Reflect every column pair (high <-> low) around the Asian-range midpoint,
    # then keep the reflected value only on flipped days. Both the wick and close
    # extremes are reflected so the chosen breakout_criteria sees a consistent move.
    for hi_col, lo_col in (("day_high", "day_low"), ("day_close_high", "day_close_low")):
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
  breakout_criteria: str = "wick",
) -> Path:
  """Load data and compute Asian Range Breakout for the daily timeframe.

  Writes the result JSON to results/.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  stat = AsianRangeBreakout(
    instrument=instrument,
    config=config,
    breakout_criteria=breakout_criteria,
  )
  result = stat.compute(candles_df)

  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Asian Range Breakout stat")
  parser.add_argument("--instrument", default="NQ", help="Instrument name (default: NQ)")
  parser.add_argument("--data-path", default=None, help="Override parquet file path")
  parser.add_argument("--config-dir", default="config", help="Config directory (default: config)")
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
    breakout_criteria=args.breakout_criteria,
  )

  print(f"Written: {output_path}")

  with open(output_path, encoding="utf-8") as f:
    data = json.load(f)

  instrument_data = data["instruments"][args.instrument]
  for tf, tf_data in instrument_data.items():
    print(f"\n  {tf}: {tf_data['total_samples']} resolved sessions | {tf_data['data_range']}")
    for row in tf_data["results"]:
      print(
        f"    {row['condition']} -> {row['outcome']}: "
        f"{row['probability']:.3f} (N={row['total']}, baseline={row['baseline_prob']:.3f})"
      )
