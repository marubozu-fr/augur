"""Opening Range Breakout (ORB) stat — standard variant.

Measures the opening range from the first ``orb_period`` minutes of the RTH
session and classifies which direction price breaks that range during the REST
of the session.

Definitions:
  - **Opening range**: the high-low range of the bars in the ORB window
    ``[rth_start, rth_start + orb_period)``. ``orb_high`` is the max high and
    ``orb_low`` the min low over that window; ``orb_size`` is ``orb_high - orb_low``.
  - **Breakout window**: the rest of the session, ``[rth_start + orb_period,
    rth_end)``. The opening range itself is excluded so the range can never break
    its own bars.
  - **Break**: how the breakout window's price relates to the opening range.
    With ``breakout_criteria = "wick"`` (default) a break uses the window's
    intraday extremes (a wick beyond the level counts); with ``"close"`` a break
    requires a bar to CLOSE beyond the level (the window's extreme close).

Four MUTUALLY EXCLUSIVE outcomes partition every countable day exhaustively
(strict inequality defines a break — touching the level exactly is NOT a break,
mirroring the sibling ``inside_bars`` / ``outside_days`` convention):
  - ``broke_high`` — broke above ``orb_high`` only (high break, low held)
  - ``broke_low``  — broke below ``orb_low`` only (low break, high held)
  - ``broke_both`` — broke both sides during the breakout window
  - ``neither``    — stayed entirely within the opening range all session

Reported under a single ``orb`` condition; the four outcomes sum to the countable
day count, so their probabilities sum to 1. ``total_samples`` counts ALL resolved
days; each row's ``total`` is the countable days (a clean opening range AND a
non-empty breakout window).

Results are computed for three opening-range lengths, each emitted as its own
timeframe entry: 15min (09:30–09:45), 30min (09:30–10:00), 1h (09:30–10:30).

Declared slices re-run the whole computation per subset:
  - ``weekday``     — the "by weekday" breakdown.
  - ``close``       — split by session close color (green/red).
  - ``prev_candle`` — split by the prior session's color (green/red).
  - ``size``        — quartile buckets of the opening-range size.
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
  StatRunResult,
  TimeframeResult,
  write_results,
)
from stats.config import InstrumentConfig, load_config, minute_of_day
from stats.utils.daily_candles import build_resolved_days

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="Opening Range Breakout",
  fr="Cassure du range d'ouverture",
)
_DEFINITION = I18nString(
  en="After the opening range forms from the first N minutes of the session, how often does price break above the range high, below the range low, both sides, or neither during the rest of the session?",
  fr="Après la formation du range d'ouverture sur les N premières minutes de la session, à quelle fréquence le prix casse-t-il au-dessus du haut du range, en-dessous du bas du range, des deux côtés, ou ni l'un ni l'autre pendant le reste de la séance ?",
)
_LABELS = Labels(
  conditions={
    "orb": I18nString(en="Opening range", fr="Range d'ouverture"),
  },
  outcomes={
    "broke_high": I18nString(
      en="Broke above range high only",
      fr="Cassure au-dessus du haut du range uniquement",
    ),
    "broke_low": I18nString(
      en="Broke below range low only",
      fr="Cassure en-dessous du bas du range uniquement",
    ),
    "broke_both": I18nString(
      en="Broke both sides",
      fr="Cassure des deux côtés",
    ),
    "neither": I18nString(
      en="Stayed within the range",
      fr="Resté dans le range",
    ),
  },
)

# Outcome enumeration, in display order. All four partition the countable days.
_OUTCOMES: tuple[str, ...] = ("broke_high", "broke_low", "broke_both", "neither")

# ---------------------------------------------------------------------------
# Timeframe string → ORB window length in minutes
# ---------------------------------------------------------------------------
_TF_MINUTES: dict[str, int] = {
  "15min": 15,
  "30min": 30,
  "1h": 60,
}

_BREAKOUT_CRITERIA: tuple[str, ...] = ("wick", "close")


class OpeningRangeBreakout(BaseStat):
  """Breakout-direction distribution of price relative to the opening range."""

  stat_name = "opening_range_breakout"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = (
    "weekday",
    "close",
    "prev_candle",
    SizeBucket(column="orb_size", preset="quartiles", name="size"),
  )

  def __init__(
    self,
    instrument: str,
    timeframe: str,
    config: InstrumentConfig,
    breakout_criteria: str = "wick",
    close_tolerance_min: int = 15,
  ) -> None:
    if timeframe not in _TF_MINUTES:
      raise ValueError(f"Unsupported timeframe '{timeframe}'. Choose from {list(_TF_MINUTES)}")
    if breakout_criteria not in _BREAKOUT_CRITERIA:
      raise ValueError(
        f"Unsupported breakout_criteria '{breakout_criteria}'. "
        f"Choose from {list(_BREAKOUT_CRITERIA)}"
      )
    self.instrument = instrument
    self.timeframe = timeframe
    self.config = config
    self.orb_minutes = _TF_MINUTES[timeframe]
    self.breakout_criteria = breakout_criteria
    self.close_tolerance_min = close_tolerance_min

    rth = config.sessions["rth"]
    self.rth_start_min: int = minute_of_day(rth.start)
    self.rth_end_min: int = minute_of_day(rth.end)
    # Opening range window: [rth_start, orb_end). The breakout window is the rest
    # of the session, [orb_end, rth_end).
    self.orb_end_min: int = self.rth_start_min + self.orb_minutes

  # -------------------------------------------------------------------------
  # Day table construction
  # -------------------------------------------------------------------------
  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-session table with the opening range and breakout extremes.

    Columns returned:
      ``orb_high`` / ``orb_low``         — opening-range extremes from the ORB window.
      ``orb_size``                       — ``orb_high - orb_low`` (read by the size slicer).
      ``post_high`` / ``post_low``       — breakout-window intraday extremes (wick).
      ``post_close_high`` / ``post_close_low`` — breakout-window extreme CLOSES (close
                                           criteria); max / min of the window's bar closes.
      ``session_green``                  — session color (``session_close >= session_open``),
                                           read by the ``close`` slicer.
      ``prev_session_green``             — prior session color (NaN for the first resolved
                                           day), read by the ``prev_candle`` slicer.

    The resolved-days core (RTH filter, session open/close, resolution filter,
    chronological sort) is shared via ``build_resolved_days``; the ORB and
    breakout-window aggregates are joined on per day. A day is countable only when
    it has a clean opening range AND a non-empty breakout window; days missing
    either drop out via the inner join (pending-sample discipline). The
    DatetimeIndex (normalized session date) is required by the ``weekday`` slicer.
    """
    columns = [
      "orb_high",
      "orb_low",
      "orb_size",
      "post_high",
      "post_low",
      "post_close_high",
      "post_close_low",
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

    # Opening-range window aggregates: [rth_start, orb_end).
    orb_mask = (df["_mod"] >= self.rth_start_min) & (df["_mod"] < self.orb_end_min)
    orb = df[orb_mask]
    orb_high = orb.groupby("_date")["high"].max().rename("orb_high")
    orb_low = orb.groupby("_date")["low"].min().rename("orb_low")

    # Breakout-window aggregates: [orb_end, rth_end).
    post_mask = (df["_mod"] >= self.orb_end_min) & (df["_mod"] < self.rth_end_min)
    post = df[post_mask]
    post_high = post.groupby("_date")["high"].max().rename("post_high")
    post_low = post.groupby("_date")["low"].min().rename("post_low")
    post_close_high = post.groupby("_date")["close"].max().rename("post_close_high")
    post_close_low = post.groupby("_date")["close"].min().rename("post_close_low")

    # Inner join onto the resolved index: a day survives only with a full opening
    # range AND a non-empty breakout window.
    day = (
      resolved.join(orb_high, how="inner")
      .join(orb_low, how="inner")
      .join(post_high, how="inner")
      .join(post_low, how="inner")
      .join(post_close_high, how="inner")
      .join(post_close_low, how="inner")
    )
    if day.empty:
      return empty

    day["orb_size"] = day["orb_high"] - day["orb_low"]
    day["session_green"] = day["session_close"] >= day["session_open"]
    # Prior RESOLVED session's color (shift over the resolved-only, sorted index,
    # so it skips any excluded/early-close day). NaN for the first resolved day.
    day["prev_session_green"] = day["session_green"].shift(1)

    return day[columns]

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def _break_extremes(self, day_table: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return the (high, low) breakout extremes selected by ``breakout_criteria``."""
    if self.breakout_criteria == "close":
      return day_table["post_close_high"], day_table["post_close_low"]
    return day_table["post_high"], day_table["post_low"]

  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the four breakout-direction rows over the (sliced) day table.

    Only days with a valid opening range (``orb_high`` not NaN) are countable. The
    four outcomes are mutually exclusive and partition the countable days, so their
    counts sum to ``total``. If ``baseline_rows`` is provided, merges its
    ``probability`` / ``total`` into each row's ``baseline_prob`` / ``baseline_n``.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    def _make(outcome: str, count: int, total: int) -> StatResultRow:
      bl = baseline_map.get(("orb", outcome))
      return StatResultRow(
        condition="orb",
        outcome=outcome,
        count=count,
        total=total,
        probability=count / total if total > 0 else 0.0,
        baseline_prob=bl.probability if bl else 0.0,
        baseline_n=bl.total if bl else 0,
      )

    if day_table.empty:
      return [_make(out, 0, 0) for out in _OUTCOMES]

    orb_high = day_table["orb_high"]
    orb_low = day_table["orb_low"]
    high, low = self._break_extremes(day_table)

    countable = orb_high.notna() & orb_low.notna() & high.notna() & low.notna()
    total = int(countable.sum())

    # Strict inequality defines a break (touching the level exactly is NOT a break).
    above = countable & (high > orb_high)
    below = countable & (low < orb_low)

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

    Each countable day's breakout move is REFLECTED around its opening-range
    midpoint with probability 0.5 (a price ``p`` maps to ``2*mid - p``, which swaps
    the high and low extremes). Reflection turns a ``broke_high`` day into a
    ``broke_low`` day and vice versa, while ``broke_both`` and ``neither`` are
    direction-symmetric and unchanged. The null therefore has NO directional bias:
    ``broke_high`` and ``broke_low`` converge to their shared mean, so the
    comparison reveals whether the opening range breaks UP more often than DOWN
    beyond a coin flip. Uses ``np.random.default_rng(seed)``.
    """
    n = len(day_table)
    if n == 0:
      return self.compute_rows(day_table, baseline_rows=None)

    rng = np.random.default_rng(seed)
    flip = rng.integers(0, 2, size=n).astype(bool)

    mid = (day_table["orb_high"].to_numpy() + day_table["orb_low"].to_numpy()) / 2.0
    tmp = day_table.copy()

    # Reflect every column pair (high <-> low) around the midpoint, then keep the
    # reflected value only on flipped days. Both the wick and close extremes are
    # reflected so the chosen breakout_criteria sees a consistent move.
    for hi_col, lo_col in (("post_high", "post_low"), ("post_close_high", "post_close_low")):
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
  """Load data and compute ORB standard for all three opening-range lengths.

  Merges the per-timeframe TimeframeResults into one StatRunResult and writes the
  consolidated JSON to results/.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  timeframes = ["15min", "30min", "1h"]
  merged_tf: dict[str, TimeframeResult] = {}
  labels = _LABELS.model_copy(deep=True)

  for tf in timeframes:
    stat = OpeningRangeBreakout(
      instrument=instrument,
      timeframe=tf,
      config=config,
      breakout_criteria=breakout_criteria,
    )
    result = stat.compute(candles_df)
    merged_tf[tf] = result.instruments[instrument][tf]
    # Merge the framework-enriched slice dimension labels so no timeframe's
    # dimensions overwrite an earlier one's.
    labels.dimensions.update(result.labels.dimensions)

  final_result = StatRunResult(
    stat_name="opening_range_breakout",
    title=_TITLE,
    definition=_DEFINITION,
    labels=labels,
    instruments={instrument: merged_tf},
  )

  return write_results(final_result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Opening Range Breakout stat")
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
