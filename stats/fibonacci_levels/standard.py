"""Fibonacci Retracement Levels stat — standard variant.

Measures (a) which of 8 Fibonacci zone the current session opens in, and
(b) which Fibonacci levels the session touches during RTH, relative to the
PRIOR session's range anchored by the prior candle's direction.

Two tiers of rows are reported:

  1. Level touches (condition ``fib_levels``): over all countable days, the
     fraction of sessions that touch each Fibonacci level (0%, 23.6%, 38.2%,
     50%, 61.8%, 78.6%, 100%) during the RTH session. A level is "touched"
     when the session's RTH high >= level >= session's RTH low. These outcomes
     are INDEPENDENT — a single session can touch multiple levels — so they do
     NOT partition.

  2. Opening zone (condition ``opening_zone``): over all countable days, which
     of 8 Fibonacci zones the session open falls in. Zones tile the prior range
     plus two "outside" regions beyond the extremes. These outcomes DO partition
     the countable set (each session open lands in exactly one zone).

Fibonacci levels are anchored by the PRIOR session's direction so that f=0
always marks the starting extreme of the prior move and f=1 the opposite:
  - Prior session GREEN (low → high move): retracement goes DOWN from the top.
    ``level(f) = prev_high - f * (prev_high - prev_low)``.
    f=0 → prev_high, f=1 → prev_low.
  - Prior session RED (high → low move): retracement goes UP from the bottom.
    ``level(f) = prev_low + f * (prev_high - prev_low)``.
    f=0 → prev_low, f=1 → prev_high.

The opening zone fraction is computed as the open's position on this same [0,1]
scale: ``open_frac = (prev_high - session_open) / rng_`` for green prior and
``(session_open - prev_low) / rng_`` for red prior. Values outside [0, 1] land
in ``below_0`` (open_frac < 0, beyond the anchor) or ``above_100`` (open_frac
> 1, beyond the far extreme).

Countable days require a prior resolved day with a strictly positive prior
range (``prev_high > prev_low``). The first resolved day has NaN prior values
and is always excluded (pending-sample discipline).

Declared slices re-run the whole computation per subset:
  - ``weekday``     — the "by weekday" breakdown.
  - ``prev_candle`` — the "by prior close" breakdown (prior session green/red).
    This slice naturally isolates the fib-level behavior that each anchor
    orientation produces.
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
  StatResultRow,
  write_results,
)
from stats.config import InstrumentConfig, load_config, minute_of_day
from stats.utils.daily_candles import build_day_table_with_prior_range

# ---------------------------------------------------------------------------
# Fibonacci constants
# ---------------------------------------------------------------------------
# Fractions at which standard Fibonacci retracement levels are drawn.
_FIB_FRACS: tuple[float, ...] = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)

# Outcome keys for the fib-level touch tier (one per fraction, in order).
_TOUCH_KEYS: tuple[str, ...] = (
  "touch_0",
  "touch_236",
  "touch_382",
  "touch_500",
  "touch_618",
  "touch_786",
  "touch_100",
)

# Outcome keys for the opening-zone tier. Zones are ordered from the anchor
# side (f=0) outward toward the far extreme (f=1) and beyond.
_ZONE_KEYS: tuple[str, ...] = (
  "below_0",
  "0_236",
  "236_382",
  "382_500",
  "500_618",
  "618_786",
  "786_100",
  "above_100",
)

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="Fibonacci Retracement Levels",
  fr="Niveaux de retracement de Fibonacci",
)
_DEFINITION = I18nString(
  en=(
    "Which Fibonacci retracement levels (0%, 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100%) "
    "of the prior RTH session's range does the current session touch? "
    "And which Fibonacci zone does the session open in? "
    "Levels are anchored by the prior candle's direction."
  ),
  fr=(
    "Quels niveaux de retracement de Fibonacci (0 %, 23,6 %, 38,2 %, 50 %, 61,8 %, "
    "78,6 %, 100 %) du range de la session RTH précédente la session actuelle touche-t-elle ? "
    "Et dans quelle zone de Fibonacci la session ouvre-t-elle ? "
    "Les niveaux sont ancrés par la direction de la bougie précédente."
  ),
)
_LABELS = Labels(
  conditions={
    "fib_levels": I18nString(
      en="Fibonacci level touched",
      fr="Niveau de Fibonacci touché",
    ),
    "opening_zone": I18nString(
      en="Opening Fibonacci zone",
      fr="Zone de Fibonacci d'ouverture",
    ),
  },
  outcomes={
    # Touch outcomes
    "touch_0": I18nString(en="Touch 0% (anchor extreme)", fr="Touche 0 % (extrême ancre)"),
    "touch_236": I18nString(en="Touch 23.6%", fr="Touche 23,6 %"),
    "touch_382": I18nString(en="Touch 38.2%", fr="Touche 38,2 %"),
    "touch_500": I18nString(en="Touch 50%", fr="Touche 50 %"),
    "touch_618": I18nString(en="Touch 61.8%", fr="Touche 61,8 %"),
    "touch_786": I18nString(en="Touch 78.6%", fr="Touche 78,6 %"),
    "touch_100": I18nString(
      en="Touch 100% (far extreme)", fr="Touche 100 % (extrême opposé)"
    ),
    # Opening-zone outcomes (partition)
    "below_0": I18nString(
      en="Below 0% (beyond anchor)", fr="Sous 0 % (au-delà de l'ancre)"
    ),
    "0_236": I18nString(en="0%–23.6%", fr="0 %–23,6 %"),
    "236_382": I18nString(en="23.6%–38.2%", fr="23,6 %–38,2 %"),
    "382_500": I18nString(en="38.2%–50%", fr="38,2 %–50 %"),
    "500_618": I18nString(en="50%–61.8%", fr="50 %–61,8 %"),
    "618_786": I18nString(en="61.8%–78.6%", fr="61,8 %–78,6 %"),
    "786_100": I18nString(en="78.6%–100%", fr="78,6 %–100 %"),
    "above_100": I18nString(
      en="Above 100% (beyond far extreme)", fr="Au-delà de 100 % (au-delà de l'extrême opposé)"
    ),
  },
)


class FibonacciLevels(BaseStat):
  """Fibonacci retracement level touch rates and opening zone distribution.

  Computes, for each resolved trading day, Fibonacci retracement levels of
  the prior RTH session's range (anchored by the prior candle's direction),
  then measures (a) which levels the current session touches and (b) which
  zone the session opens in.
  """

  stat_name = "fibonacci_levels"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = ("weekday", "prev_candle")

  def __init__(
    self,
    instrument: str,
    config: InstrumentConfig,
    close_tolerance_min: int = 15,
  ) -> None:
    self.instrument = instrument
    self.timeframe = "daily"
    self.config = config
    self.close_tolerance_min = close_tolerance_min

    rth = config.sessions["rth"]
    self.rth_start_min: int = minute_of_day(rth.start)
    self.rth_end_min: int = minute_of_day(rth.end)

  # -------------------------------------------------------------------------
  # Day table construction
  # -------------------------------------------------------------------------
  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-session table with prior-range and direction columns.

    Delegates to ``build_day_table_with_prior_range``, which delivers:
      ``session_open``, ``session_close`` (resolved RTH open/close),
      ``day_high``, ``day_low`` (RTH intraday extremes), ``prev_high``,
      ``prev_low`` (prior RESOLVED session's RTH extremes; NaN for the first
      resolved day), and ``prev_session_green`` (prior session's color; NaN
      for the first day).

    The first resolved day always has NaN prior values and is excluded from
    every denominator downstream (pending-sample discipline). The prior values
    are shifted over the resolved-only, sorted index, so they skip over any
    excluded/early-close day.
    """
    return build_day_table_with_prior_range(
      candles_df, self.rth_start_min, self.rth_end_min, self.close_tolerance_min
    )

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute fib-touch and opening-zone rows from a (possibly sliced) day table.

    Only days with a prior resolved day AND a strictly positive prior range
    (``prev_high > prev_low``) are countable.

    Tier 1 — ``fib_levels`` (7 rows): for each Fibonacci fraction in
    {0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0}, compute the level price and
    count sessions whose RTH high/low bracket it. Outcomes are independent —
    do NOT partition.

    Tier 2 — ``opening_zone`` (8 rows): classify ``session_open`` relative to
    the 7 levels into one of 8 contiguous zones. Zones partition the countable
    set (every session open lands in exactly one).

    If ``baseline_rows`` is provided, merges ``baseline_prob`` / ``baseline_n``.
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

    def _empty_rows() -> list[StatResultRow]:
      rows: list[StatResultRow] = []
      for key in _TOUCH_KEYS:
        rows.append(_make("fib_levels", key, 0, 0))
      for key in _ZONE_KEYS:
        rows.append(_make("opening_zone", key, 0, 0))
      return rows

    if day_table.empty:
      return _empty_rows()

    # ----- Countable mask: requires a prior resolved day with rng_ > 0 -----
    has_prior = day_table["prev_high"].notna() & day_table["prev_low"].notna()
    prior_rng_series = (day_table["prev_high"] - day_table["prev_low"]).where(has_prior)
    countable = has_prior & (prior_rng_series > 0)

    ct = day_table[countable]
    countable_n = int(countable.sum())

    if countable_n == 0:
      return _empty_rows()

    # ----- Extract numpy arrays for vectorized computation -----
    prev_high = ct["prev_high"].to_numpy(dtype=float)
    prev_low = ct["prev_low"].to_numpy(dtype=float)
    rng_ = prev_high - prev_low  # shape (countable_n,)

    # prev_session_green: NaN can only appear on the first resolved row, which
    # already has NaN prev_high so it never reaches here. Guard defensively.
    green = ct["prev_session_green"].fillna(False).to_numpy(dtype=bool)

    day_high = ct["day_high"].to_numpy(dtype=float)
    day_low = ct["day_low"].to_numpy(dtype=float)
    session_open = ct["session_open"].to_numpy(dtype=float)

    rows: list[StatResultRow] = []

    # ----- Tier 1: level touches -----
    # level(f) = prev_high - f * rng_   (green prior — retrace down from top)
    # level(f) = prev_low  + f * rng_   (red prior   — retrace up from bottom)
    # Touched when day_low <= level <= day_high.
    for frac, key in zip(_FIB_FRACS, _TOUCH_KEYS):
      level = np.where(green, prev_high - frac * rng_, prev_low + frac * rng_)
      touched = (day_low <= level) & (day_high >= level)
      rows.append(_make("fib_levels", key, int(touched.sum()), countable_n))

    # ----- Tier 2: opening zone -----
    # open_frac measures where session_open sits on the anchored [0, 1] scale.
    # green: open_frac = (prev_high - session_open) / rng_
    # red:   open_frac = (session_open - prev_low)  / rng_
    # open_frac < 0  → beyond the anchor extreme  (below_0)
    # open_frac > 1  → beyond the far extreme      (above_100)
    open_frac = np.where(
      green,
      (prev_high - session_open) / rng_,
      (session_open - prev_low) / rng_,
    )

    # Zones partition the real line; boundaries are half-open except that
    # open_frac == 1.0 (exactly at the far extreme) belongs to ``786_100``.
    zone_masks: list[tuple[str, np.ndarray]] = [
      ("below_0",   open_frac < 0.0),
      ("0_236",     (open_frac >= 0.0) & (open_frac < 0.236)),
      ("236_382",   (open_frac >= 0.236) & (open_frac < 0.382)),
      ("382_500",   (open_frac >= 0.382) & (open_frac < 0.5)),
      ("500_618",   (open_frac >= 0.5) & (open_frac < 0.618)),
      ("618_786",   (open_frac >= 0.618) & (open_frac < 0.786)),
      ("786_100",   (open_frac >= 0.786) & (open_frac <= 1.0)),
      ("above_100", open_frac > 1.0),
    ]
    for key, mask in zone_masks:
      rows.append(_make("opening_zone", key, int(mask.sum()), countable_n))

    return rows

  # -------------------------------------------------------------------------
  # Baseline
  # -------------------------------------------------------------------------
  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random null baseline. Deterministic for a fixed seed.

    The prior-range triple (``prev_high``, ``prev_low``, ``prev_session_green``)
    is permuted together across days using a single permutation, so each
    session is compared against an UNRELATED day's Fibonacci levels. This is
    the null hypothesis: temporal adjacency of the prior range and anchor
    direction carries no information about which levels will be touched or
    where the session will open.

    All three columns are shuffled with the SAME permutation index so that
    each shuffled prior range stays internally consistent (``prev_low <=
    prev_high``) and each anchor direction remains aligned with its range.
    The lone NaN prior triple (the first resolved day) moves to a random row,
    preserving the countable count exactly.
    """
    n = len(day_table)
    if n == 0:
      return self.compute_rows(day_table, baseline_rows=None)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    tmp = day_table.copy()
    tmp["prev_high"] = day_table["prev_high"].to_numpy()[perm]
    tmp["prev_low"] = day_table["prev_low"].to_numpy()[perm]
    tmp["prev_session_green"] = day_table["prev_session_green"].to_numpy()[perm]

    return self.compute_rows(tmp, baseline_rows=None)


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
) -> Path:
  """Load data and compute Fibonacci Retracement Levels for the daily timeframe.

  Writes the result JSON to results/.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  stat = FibonacciLevels(instrument=instrument, config=config)
  result = stat.compute(candles_df)

  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Fibonacci Retracement Levels stat")
  parser.add_argument("--instrument", default="NQ", help="Instrument name (default: NQ)")
  parser.add_argument("--data-path", default=None, help="Override parquet file path")
  parser.add_argument(
    "--config-dir", default="config", help="Config directory (default: config)"
  )
  args = parser.parse_args()

  output_path = run(
    instrument=args.instrument,
    config_dir=args.config_dir,
    data_path=args.data_path,
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
