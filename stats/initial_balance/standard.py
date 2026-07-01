"""Initial Balance Breakout (IB) stat — standard variant.

Measures the **initial balance** — the high-low range of the first ``ib_period``
minutes of the RTH session — and classifies which direction price breaks that
range during the REST of the session.

Definitions:
  - **Initial balance**: the high-low range of the bars in the IB window
    ``[rth_start, rth_start + ib_period)``. ``ib_high`` is the max high and
    ``ib_low`` the min low over that window; ``ib_size`` is ``ib_high - ib_low``.
  - **Breakout window**: the rest of the session, ``[rth_start + ib_period,
    rth_end)``. The initial balance itself is excluded so the range can never
    break its own bars.
  - **Break**: how the breakout window's price relates to the initial balance.
    With ``breakout_criteria = "wick"`` (default) a break uses the window's
    intraday extremes (a wick beyond the level counts); with ``"close"`` a break
    requires a bar to CLOSE beyond the level (the window's extreme close).

Four MUTUALLY EXCLUSIVE outcomes partition every countable day exhaustively
(strict inequality defines a break — touching the level exactly is NOT a break,
mirroring the sibling ``opening_range_breakout`` / ``inside_bars`` convention):
  - ``broke_high`` — broke above ``ib_high`` only (high break, low held)
  - ``broke_low``  — broke below ``ib_low`` only (low break, high held)
  - ``broke_both`` — broke both sides during the breakout window
  - ``neither``    — stayed entirely within the initial balance all session

Reported under a single ``ib`` condition; the four outcomes sum to the countable
day count, so their probabilities sum to 1. ``total_samples`` counts ALL resolved
days; each row's ``total`` is the countable days (a clean initial balance AND a
non-empty breakout window).

The initial balance is conventionally the first 30 to 60 minutes of the session,
so results are computed for two IB lengths, each emitted as its own timeframe
entry: 30min (09:30–10:00) and 1h (09:30–10:30).

Declared slices re-run the whole computation per subset:
  - ``weekday``     — the "by weekday" breakdown.
  - ``close``       — split by session close color (green/red).
  - ``prev_candle`` — split by the prior session's color (green/red).
  - ``size``        — quartile buckets of the initial-balance size (absolute).
  - ``size_pct``    — preset buckets of the initial-balance size as % of price.
  - ``overnight``   — split by overnight gap (RTH open above/below prior close).
  - ``levels``      — bands of how far the breakout extended past the balance,
                      measured in multiples of the initial-balance size.
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
  SampleRow,
  SizeBucket,
  StatResultRow,
  StatRunResult,
  TimeframeResult,
  write_results,
)
from stats.config import InstrumentConfig, load_config, minute_of_day
from stats.initial_balance.common import build_ib_day_base

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="Initial Balance Breakout",
  fr="Cassure de l'initial balance",
)
_DEFINITION = I18nString(
  en="After the initial balance forms from the first N minutes of the session, how often does price break above the balance high, below the balance low, both sides, or neither during the rest of the session?",
  fr="Après la formation de l'initial balance sur les N premières minutes de la session, à quelle fréquence le prix casse-t-il au-dessus du haut de l'initial balance, en-dessous du bas, des deux côtés, ou ni l'un ni l'autre pendant le reste de la séance ?",
)
_LABELS = Labels(
  conditions={
    "ib": I18nString(en="Initial balance", fr="Initial balance"),
  },
  outcomes={
    "broke_high": I18nString(
      en="Broke above balance high only",
      fr="Cassure au-dessus du haut de l'initial balance uniquement",
    ),
    "broke_low": I18nString(
      en="Broke below balance low only",
      fr="Cassure en-dessous du bas de l'initial balance uniquement",
    ),
    "broke_both": I18nString(
      en="Broke both sides",
      fr="Cassure des deux côtés",
    ),
    "neither": I18nString(
      en="Stayed within the balance",
      fr="Resté dans l'initial balance",
    ),
  },
)

# Outcome enumeration, in display order. All four partition the countable days.
_OUTCOMES: tuple[str, ...] = ("broke_high", "broke_low", "broke_both", "neither")

# ---------------------------------------------------------------------------
# Timeframe string → IB window length in minutes
# ---------------------------------------------------------------------------
_TF_MINUTES: dict[str, int] = {
  "30min": 30,
  "1h": 60,
}

_BREAKOUT_CRITERIA: tuple[str, ...] = ("wick", "close")


class InitialBalanceBreakout(BaseStat):
  """Breakout-direction distribution of price relative to the initial balance."""

  stat_name = "initial_balance"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = (
    "weekday",
    "close",
    "prev_candle",
    "overnight",
    SizeBucket(column="ib_size", preset="quartiles", name="size"),
    # 0.6–0.9%, >0.9%); upper edge is open (inf).
    SizeBucket(
      column="ib_size_pct",
      buckets=[0.0, 0.2, 0.4, 0.6, 0.9, float("inf")],
      name="size_pct",
    ),
    # How far the breakout extended past the balance, in multiples of ib_size.
    Levels(ref="ib_size", ext="extension", name="levels"),
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
    self.ib_minutes = _TF_MINUTES[timeframe]
    self.breakout_criteria = breakout_criteria
    self.close_tolerance_min = close_tolerance_min

    rth = config.sessions["rth"]
    self.rth_start_min: int = minute_of_day(rth.start)
    self.rth_end_min: int = minute_of_day(rth.end)
    # Initial-balance window: [rth_start, ib_end). The breakout window is the rest
    # of the session, [ib_end, rth_end).
    self.ib_end_min: int = self.rth_start_min + self.ib_minutes

  # -------------------------------------------------------------------------
  # Day table construction
  # -------------------------------------------------------------------------
  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-session table with the initial balance and breakout extremes.

    Columns returned:
      ``ib_high`` / ``ib_low``           — initial-balance extremes from the IB window.
      ``ib_size``                        — ``ib_high - ib_low`` (read by the size slicer).
      ``ib_size_pct``                    — ``ib_size`` as % of the session open price
                                           (read by the ``size_pct`` slicer).
      ``extension``                      — furthest breakout past the balance in either
                                           direction (``>= 0``, criteria-aware); read by
                                           the ``levels`` slicer as a multiple of ``ib_size``.
      ``post_high`` / ``post_low``       — breakout-window intraday extremes (wick).
      ``post_close_high`` / ``post_close_low`` — breakout-window extreme CLOSES (close
                                           criteria); max / min of the window's bar closes.
      ``session_green``                  — session color (``session_close >= session_open``),
                                           read by the ``close`` slicer.
      ``prev_session_green``             — prior session color (NaN for the first resolved
                                           day), read by the ``prev_candle`` slicer.
      ``overnight_green``                — overnight gap: session open above the prior
                                           session's close (NaN for the first resolved day),
                                           read by the ``overnight`` slicer.

    The resolved-days core (RTH filter, session open/close, resolution filter,
    chronological sort) is shared via ``build_resolved_days``; the IB and
    breakout-window aggregates are joined on per day. A day is countable only when
    it has a clean initial balance AND a non-empty breakout window; days missing
    either drop out via the inner join (pending-sample discipline). The
    DatetimeIndex (normalized session date) is required by the ``weekday`` slicer.
    """
    columns = [
      "ib_high",
      "ib_low",
      "ib_size",
      "ib_size_pct",
      "extension",
      "post_high",
      "post_low",
      "post_close_high",
      "post_close_low",
      "session_green",
      "prev_session_green",
      "overnight_green",
    ]
    empty = pd.DataFrame(columns=columns)

    if candles_df.empty:
      return empty

    base = build_ib_day_base(
      candles_df,
      self.rth_start_min,
      self.ib_end_min,
      self.rth_end_min,
      self.close_tolerance_min,
    )
    if base.day.empty:
      return empty

    post = base.post
    # Breakout-window extremes (this stat's last step): wick extremes plus extreme
    # closes, joined onto the presence-guarded base. Every surviving day has a
    # non-empty breakout window, so all four are present for every row.
    post_high = post.groupby("_date")["high"].max().rename("post_high")
    post_low = post.groupby("_date")["low"].min().rename("post_low")
    post_close_high = post.groupby("_date")["close"].max().rename("post_close_high")
    post_close_low = post.groupby("_date")["close"].min().rename("post_close_low")
    day = (
      base.day.join(post_high)
      .join(post_low)
      .join(post_close_high)
      .join(post_close_low)
    )

    # Furthest breakout past the balance in either direction, using the same
    # criteria-selected extremes as the break classification (wick or close).
    high, low = self._break_extremes(day)
    ext_up = (high - day["ib_high"]).clip(lower=0.0)
    ext_down = (day["ib_low"] - low).clip(lower=0.0)
    day["extension"] = pd.concat([ext_up, ext_down], axis=1).max(axis=1)

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

    Only days with a valid initial balance (``ib_high`` not NaN) are countable. The
    four outcomes are mutually exclusive and partition the countable days, so their
    counts sum to ``total``. If ``baseline_rows`` is provided, merges its
    ``probability`` / ``total`` into each row's ``baseline_prob`` / ``baseline_n``.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    def _make(outcome: str, count: int, total: int) -> StatResultRow:
      bl = baseline_map.get(("ib", outcome))
      return StatResultRow(
        condition="ib",
        outcome=outcome,
        count=count,
        total=total,
        probability=count / total if total > 0 else 0.0,
        baseline_prob=bl.probability if bl else 0.0,
        baseline_n=bl.total if bl else 0,
      )

    if day_table.empty:
      return [_make(out, 0, 0) for out in _OUTCOMES]

    ib_high = day_table["ib_high"]
    ib_low = day_table["ib_low"]
    high, low = self._break_extremes(day_table)

    countable = ib_high.notna() & ib_low.notna() & high.notna() & low.notna()
    total = int(countable.sum())

    # Strict inequality defines a break (touching the level exactly is NOT a break).
    above = countable & (high > ib_high)
    below = countable & (low < ib_low)

    counts = {
      "broke_high": int((above & ~below).sum()),
      "broke_low": int((below & ~above).sum()),
      "broke_both": int((above & below).sum()),
      "neither": int((countable & ~above & ~below).sum()),
    }

    return [_make(out, counts[out], total) for out in _OUTCOMES]

  def classify_samples(self, day_table: pd.DataFrame) -> list[SampleRow]:
    """One SampleRow per countable day, mirroring compute_rows' four-way breakout
    partition. Days missing an IB or a breakout-window extreme are excluded
    (not countable), matching the ``countable`` mask used there.
    """
    if day_table.empty:
      return []

    ib_high = day_table["ib_high"]
    ib_low = day_table["ib_low"]
    high, low = self._break_extremes(day_table)

    countable = ib_high.notna() & ib_low.notna() & high.notna() & low.notna()
    if not bool(countable.any()):
      return []

    above = (high > ib_high).to_numpy()
    below = (low < ib_low).to_numpy()
    outcome = np.select(
      [above & below, above & ~below, below & ~above],
      ["broke_both", "broke_high", "broke_low"],
      default="neither",
    )

    dates = np.array(day_table.index.strftime("%Y-%m-%d"))
    idx = np.flatnonzero(countable.to_numpy())
    return [SampleRow(date=dates[i], condition="ib", outcome=outcome[i]) for i in idx]

  # -------------------------------------------------------------------------
  # Baseline
  # -------------------------------------------------------------------------
  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random directional baseline. Deterministic for a fixed seed.

    Each countable day's breakout move is REFLECTED around its initial-balance
    midpoint with probability 0.5 (a price ``p`` maps to ``2*mid - p``, which swaps
    the high and low extremes). Reflection turns a ``broke_high`` day into a
    ``broke_low`` day and vice versa, while ``broke_both`` and ``neither`` are
    direction-symmetric and unchanged. The null therefore has NO directional bias:
    ``broke_high`` and ``broke_low`` converge to their shared mean, so the
    comparison reveals whether the initial balance breaks UP more often than DOWN
    beyond a coin flip. Uses ``np.random.default_rng(seed)``.
    """
    n = len(day_table)
    if n == 0:
      return self.compute_rows(day_table, baseline_rows=None)

    rng = np.random.default_rng(seed)
    flip = rng.integers(0, 2, size=n).astype(bool)

    mid = (day_table["ib_high"].to_numpy() + day_table["ib_low"].to_numpy()) / 2.0
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
  """Load data and compute IB standard for both initial-balance lengths.

  Merges the per-timeframe TimeframeResults into one StatRunResult and writes the
  consolidated JSON to results/.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  timeframes = ["30min", "1h"]
  merged_tf: dict[str, TimeframeResult] = {}
  labels = _LABELS.model_copy(deep=True)

  for tf in timeframes:
    stat = InitialBalanceBreakout(
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
    stat_name="initial_balance",
    title=_TITLE,
    definition=_DEFINITION,
    labels=labels,
    instruments={instrument: merged_tf},
  )

  return write_results(final_result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Initial Balance Breakout stat")
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
