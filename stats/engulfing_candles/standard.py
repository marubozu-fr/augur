"""Engulfing Candles stat (standard variant).

Identifies bullish and bearish engulfing patterns on the RTH daily candle and
measures how far price *continues* in the pattern's direction, from the engulfing
candle's CLOSE until the pattern is INVALIDATED (price reclaims the engulfing
candle's OPEN).

Pattern (day T compared against the prior RESOLVED day T-1, using candle BODIES —
the ``[min(open, close), max(open, close)]`` interval):

  - **Bullish engulfing**: day T is green (``close >= open``), day T-1 is red, and
    day T's body fully engulfs day T-1's body
    (``max(o, c) >= prev_top`` AND ``min(o, c) <= prev_bot``).
  - **Bearish engulfing**: day T is red (``close < open``), day T-1 is green, and
    day T's body fully engulfs day T-1's body.

  Body engulfment uses inclusive inequalities (an exactly-equal edge still
  engulfs). The opposite-prior-color requirement is the canonical reversal-pattern
  definition: a green candle that swallows a prior red one (and vice versa).

Continuation (the magnitude measured):

  Starting from the engulfing candle's CLOSE, price is tracked forward over
  subsequent daily candles until the pattern INVALIDATES — defined as price
  reclaiming the engulfing candle's OPEN:
    - bullish: invalidated on the first later day whose ``day_low <= open_T``;
    - bearish: invalidated on the first later day whose ``day_high >= open_T``.
  The continuation is the **maximum favorable excursion** reached over that window
  (the highest high for bullish, the lowest low for bearish — the invalidation
  day's extreme is included, since intraday the favorable move can precede the
  reclaim), expressed as a **percent of the engulfing close** and floored at zero.

  A pattern that never invalidates before the end of the data is PENDING and is
  excluded from the continuation magnitude (pending-sample discipline). Its
  occurrence is still observable, so it is counted in the frequency tier.

  Continuation depends only on a candle's OWN body and the forward price path — not
  on the prior candle — so it is precomputed per day in ``build_day_table``. The
  classification (which days are engulfing) is recomputed in ``compute_rows`` from
  the body columns, so slices and the random baseline reuse the same precomputed
  continuation correctly.

Two tiers of rows are reported:

  1. Engulfing frequency (condition ``engulfing``): over all countable days, how
     often is the day a ``bullish`` or a ``bearish`` engulfing? The two outcomes
     are mutually exclusive but do not partition (most days are neither).
  2. Continuation magnitude (conditions ``bullish`` / ``bearish``): the
     ``avg_continuation`` and ``max_continuation`` percent over the RESOLVED
     engulfing patterns of that direction. Magnitude rows carry the metric in the
     ``value`` channel (random baseline in ``value_baseline``); ``probability`` is
     left at ``0.0``.

The "prior day" is the chronologically previous RESOLVED day, so it skips any
excluded/early-close day. The FIRST resolved day has no prior day and is excluded
from every denominator. ``total_samples`` counts ALL resolved days; each row's
``total`` counts only the relevant countable/resolved days.

Declared slices re-run the whole computation per subset:
  - ``weekday`` — the "by weekday" breakdown.
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
from stats.utils.daily_candles import build_day_table_with_prior_range

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="Engulfing Candles",
  fr="Bougies englobantes",
)
_DEFINITION = I18nString(
  en="How often is the RTH daily candle a bullish or bearish engulfing pattern (its body fully engulfs the prior day's body, opposite color), and given such a pattern, how far does price continue from the engulfing close until it reclaims the engulfing open?",
  fr="À quelle fréquence la bougie journalière RTH est-elle un pattern englobant haussier ou baissier (son corps englobe entièrement celui du jour précédent, couleur opposée), et après un tel pattern, jusqu'où le prix poursuit-il depuis la clôture englobante avant de reprendre l'ouverture englobante ?",
)
_LABELS = Labels(
  conditions={
    "engulfing": I18nString(en="Engulfing candle", fr="Bougie englobante"),
    "bullish": I18nString(
      en="Bullish engulfing",
      fr="Englobante haussière",
    ),
    "bearish": I18nString(
      en="Bearish engulfing",
      fr="Englobante baissière",
    ),
  },
  outcomes={
    "bullish": I18nString(
      en="Bullish engulfing",
      fr="Englobante haussière",
    ),
    "bearish": I18nString(
      en="Bearish engulfing",
      fr="Englobante baissière",
    ),
    "avg_continuation": I18nString(
      en="Average continuation (% of close)",
      fr="Continuation moyenne (% de la clôture)",
    ),
    "max_continuation": I18nString(
      en="Maximum continuation (% of close)",
      fr="Continuation maximale (% de la clôture)",
    ),
  },
)


def _continuation(
  opens: np.ndarray,
  closes: np.ndarray,
  highs: np.ndarray,
  lows: np.ndarray,
  is_green: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Per-day continuation (% of close) and resolved flag via a forward walk.

  For each day ``i`` the continuation is measured in its natural direction (up for
  a green candle, down for a red one): from the close, track the maximum favorable
  excursion over the following days until price reclaims the open (bullish: a later
  ``low <= open``; bearish: a later ``high >= open``). The invalidation day's
  extreme is included. The result is the excursion as a percent of the close,
  floored at zero. Days that never invalidate before the end of the data are
  PENDING: their continuation is NaN and ``resolved`` is False.
  """
  n = len(opens)
  cont = np.full(n, np.nan)
  resolved = np.zeros(n, dtype=bool)
  for i in range(n):
    ref = closes[i]
    inval = opens[i]
    if not np.isfinite(ref) or not np.isfinite(inval) or ref == 0.0:
      continue
    if is_green[i]:
      run = -np.inf
      for j in range(i + 1, n):
        if highs[j] > run:
          run = highs[j]
        if lows[j] <= inval:
          resolved[i] = True
          cont[i] = max(run - ref, 0.0) / ref * 100.0
          break
    else:
      run = np.inf
      for j in range(i + 1, n):
        if lows[j] < run:
          run = lows[j]
        if highs[j] >= inval:
          resolved[i] = True
          cont[i] = max(ref - run, 0.0) / ref * 100.0
          break
  return cont, resolved


class EngulfingCandles(BaseStat):
  """Engulfing-pattern frequency plus continuation magnitude until invalidation."""

  stat_name = "engulfing_candles"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = ("weekday",)

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
    """Build the per-session table with bodies, prior body and continuation.

    Columns:
      session_open, session_close (the RTH daily candle body), prev_open,
      prev_close (the prior RESOLVED day's body; NaN for the first resolved day),
      cont_pct (the day's own forward continuation as a percent of close; NaN if
      pending or undefined) and cont_resolved (whether that continuation
      invalidated before the end of the data).

    ``prev_open`` / ``prev_close`` are NaN for the first resolved day (no prior
    day), so it is excluded from every denominator downstream. Continuation is
    computed once here over the full chronological table, so slices and the
    random baseline reuse it without re-walking the forward path.
    """
    columns = [
      "session_open",
      "session_close",
      "prev_open",
      "prev_close",
      "cont_pct",
      "cont_resolved",
    ]
    daily = build_day_table_with_prior_range(
      candles_df, self.rth_start_min, self.rth_end_min, self.close_tolerance_min
    )
    if daily.empty:
      return pd.DataFrame(columns=columns)

    daily = daily.copy()
    daily["prev_open"] = daily["session_open"].shift(1)
    daily["prev_close"] = daily["session_close"].shift(1)

    opens = daily["session_open"].to_numpy(dtype=float)
    closes = daily["session_close"].to_numpy(dtype=float)
    highs = daily["day_high"].to_numpy(dtype=float)
    lows = daily["day_low"].to_numpy(dtype=float)
    is_green = closes >= opens

    cont, resolved = _continuation(opens, closes, highs, lows, is_green)
    daily["cont_pct"] = cont
    daily["cont_resolved"] = resolved

    return daily[columns]

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the engulfing-frequency and continuation-magnitude rows.

    Only days with a prior resolved day (``prev_open`` not NaN) are countable. The
    classification is recomputed here from the body columns so slices and the
    baseline reuse the precomputed ``cont_pct``. Magnitude rows aggregate the
    precomputed continuation over the RESOLVED engulfing patterns of each
    direction (pending patterns excluded). If ``baseline_rows`` is provided, the
    frequency rows merge ``baseline_prob`` / ``baseline_n`` and the magnitude rows
    merge ``value_baseline``.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    def prob_row(condition: str, outcome: str, count: int, total: int) -> StatResultRow:
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

    def mag_row(condition: str, outcome: str, value: float | None, n: int) -> StatResultRow:
      bl = baseline_map.get((condition, outcome))
      return StatResultRow(
        condition=condition,
        outcome=outcome,
        count=n,
        total=n,
        probability=0.0,
        baseline_prob=0.0,
        baseline_n=bl.total if bl else 0,
        value=value,
        value_baseline=bl.value if bl else None,
      )

    if day_table.empty:
      return [
        prob_row("engulfing", "bullish", 0, 0),
        prob_row("engulfing", "bearish", 0, 0),
        mag_row("bullish", "avg_continuation", None, 0),
        mag_row("bullish", "max_continuation", None, 0),
        mag_row("bearish", "avg_continuation", None, 0),
        mag_row("bearish", "max_continuation", None, 0),
      ]

    open_ = day_table["session_open"].to_numpy(dtype=float)
    close = day_table["session_close"].to_numpy(dtype=float)
    prev_open = day_table["prev_open"].to_numpy(dtype=float)
    prev_close = day_table["prev_close"].to_numpy(dtype=float)
    cont = day_table["cont_pct"].to_numpy(dtype=float)
    resolved = day_table["cont_resolved"].to_numpy(dtype=bool)

    countable = np.isfinite(prev_open) & np.isfinite(prev_close)
    is_green = close >= open_
    prev_green = prev_close >= prev_open

    curr_top = np.maximum(open_, close)
    curr_bot = np.minimum(open_, close)
    prev_top = np.maximum(prev_open, prev_close)
    prev_bot = np.minimum(prev_open, prev_close)
    # Inclusive body engulfment: current body fully encompasses the prior body.
    engulfs = countable & (curr_top >= prev_top) & (curr_bot <= prev_bot)

    # Canonical reversal pattern: opposite prior color.
    bullish = engulfs & is_green & ~prev_green
    bearish = engulfs & ~is_green & prev_green

    countable_n = int(countable.sum())
    bullish_n = int(bullish.sum())
    bearish_n = int(bearish.sum())

    # Continuation magnitude over RESOLVED patterns only (pending excluded).
    bull_res = bullish & resolved
    bear_res = bearish & resolved
    bull_vals = cont[bull_res]
    bear_vals = cont[bear_res]
    bull_res_n = int(bull_res.sum())
    bear_res_n = int(bear_res.sum())

    bull_avg = float(np.mean(bull_vals)) if bull_res_n > 0 else None
    bull_max = float(np.max(bull_vals)) if bull_res_n > 0 else None
    bear_avg = float(np.mean(bear_vals)) if bear_res_n > 0 else None
    bear_max = float(np.max(bear_vals)) if bear_res_n > 0 else None

    return [
      # Tier 1 — engulfing frequency over all countable days.
      prob_row("engulfing", "bullish", bullish_n, countable_n),
      prob_row("engulfing", "bearish", bearish_n, countable_n),
      # Tier 2 — continuation magnitude given an engulfing pattern.
      mag_row("bullish", "avg_continuation", bull_avg, bull_res_n),
      mag_row("bullish", "max_continuation", bull_max, bull_res_n),
      mag_row("bearish", "avg_continuation", bear_avg, bear_res_n),
      mag_row("bearish", "max_continuation", bear_max, bear_res_n),
    ]

  def classify_samples(self, day_table: pd.DataFrame) -> list[SampleRow]:
    """One SampleRow per countable day (tier 1) plus magnitude samples (tier 2).

    Tier 1 — for every countable day (a prior resolved day exists), a sample
    with condition ``engulfing`` and outcome ``bullish`` / ``bearish`` /
    ``neither``, mirroring the frequency rows.

    Tier 2 — for every RESOLVED bullish/bearish pattern day, two samples
    (``avg_continuation`` and ``max_continuation``) carrying ``cont_pct`` as
    ``value``, so their mean/max reproduce the magnitude rows.
    """
    if day_table.empty:
      return []

    open_ = day_table["session_open"].to_numpy(dtype=float)
    close = day_table["session_close"].to_numpy(dtype=float)
    prev_open = day_table["prev_open"].to_numpy(dtype=float)
    prev_close = day_table["prev_close"].to_numpy(dtype=float)
    cont = day_table["cont_pct"].to_numpy(dtype=float)
    resolved = day_table["cont_resolved"].to_numpy(dtype=bool)

    countable = np.isfinite(prev_open) & np.isfinite(prev_close)
    is_green = close >= open_
    prev_green = prev_close >= prev_open

    curr_top = np.maximum(open_, close)
    curr_bot = np.minimum(open_, close)
    prev_top = np.maximum(prev_open, prev_close)
    prev_bot = np.minimum(prev_open, prev_close)
    engulfs = countable & (curr_top >= prev_top) & (curr_bot <= prev_bot)

    bullish = engulfs & is_green & ~prev_green
    bearish = engulfs & ~is_green & prev_green

    samples: list[SampleRow] = []
    for pos, ts in enumerate(day_table.index):
      date = ts.strftime("%Y-%m-%d")

      # Tier 1 — engulfing frequency.
      if countable[pos]:
        if bullish[pos]:
          outcome = "bullish"
        elif bearish[pos]:
          outcome = "bearish"
        else:
          outcome = "neither"
        samples.append(SampleRow(date=date, condition="engulfing", outcome=outcome))

      # Tier 2 — continuation magnitude over resolved patterns only.
      if (bullish[pos] or bearish[pos]) and resolved[pos]:
        direction = "bullish" if bullish[pos] else "bearish"
        value = float(cont[pos])
        samples.append(
          SampleRow(date=date, condition=direction, outcome="avg_continuation", value=value)
        )
        samples.append(
          SampleRow(date=date, condition=direction, outcome="max_continuation", value=value)
        )

    return samples

  # -------------------------------------------------------------------------
  # Baseline
  # -------------------------------------------------------------------------
  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random null baseline. Deterministic for a fixed seed.

    The prior-body pair (``prev_open``, ``prev_close``) is **permuted together**
    across days, so each day's body is compared against an UNRELATED day's prior
    body. This randomizes which days qualify as engulfing — the joint null for both
    tiers at once. The frequency baseline answers "how often would a body engulf an
    *arbitrary* prior body?"; the magnitude baseline answers "what continuation do
    *randomly selected* days carry?" — since each day's own precomputed
    continuation is left untouched, this isolates whether the engulfing criterion
    selects days with abnormal forward continuation. The lone NaN pair (first day)
    simply moves to a random day, preserving the countable count.
    """
    n = len(day_table)
    if n == 0:
      return self.compute_rows(day_table, baseline_rows=None)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    tmp = day_table.copy()
    tmp["prev_open"] = day_table["prev_open"].to_numpy()[perm]
    tmp["prev_close"] = day_table["prev_close"].to_numpy()[perm]

    return self.compute_rows(tmp, baseline_rows=None)


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
) -> Path:
  """Load data and compute Engulfing Candles for the daily timeframe.

  Writes the result JSON to results/.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  stat = EngulfingCandles(instrument=instrument, config=config)
  result = stat.compute(candles_df)

  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Engulfing Candles stat")
  parser.add_argument("--instrument", default="NQ", help="Instrument name (default: NQ)")
  parser.add_argument("--data-path", default=None, help="Override parquet file path")
  parser.add_argument("--config-dir", default="config", help="Config directory (default: config)")
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
      if row["value"] is not None:
        print(
          f"    {row['condition']} -> {row['outcome']}: "
          f"{row['value']:.3f}% (N={row['total']}, baseline={row['value_baseline']})"
        )
      else:
        print(
          f"    {row['condition']} -> {row['outcome']}: "
          f"{row['probability']:.3f} (N={row['total']}, baseline={row['baseline_prob']:.3f})"
        )
