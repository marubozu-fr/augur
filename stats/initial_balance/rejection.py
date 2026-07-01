"""Initial Balance Breakout — by Rejection stat.

Measures whether the ORDER in which the initial balance (IB) edges formed predicts
the direction of the first breakout. It tracks two sequential events and reports
them as a contingency table:

1. **Which IB edge formed first** during the IB window — the high or the low.
2. **Which IB edge broke first** after the IB window ends — the high, the low,
   or neither.

Definitions
-----------
- **Initial balance**: the high-low range of the bars in the IB window
  ``[rth_start, rth_start + ib_period)``. ``ib_high`` is the max high and
  ``ib_low`` the min low over that window; ``ib_size`` is ``ib_high - ib_low``.
- **Formed first**: the IB edge whose extreme is reached by the chronologically
  earliest bar. ``ib_high`` is "formed first" when the first bar reaching the IB
  high precedes the first bar reaching the IB low. When a single bar holds BOTH
  extremes the order is undetermined and the day is excluded (pending-sample
  discipline), mirroring the shared ``Rejection`` slicer.
- **Breakout window**: the rest of the session ``[rth_start + ib_period,
  rth_end)``. The IB itself is excluded so the range can never break its own bars.
- **Broke first**: the IB edge broken by the chronologically earliest breakout-
  window bar. With ``breakout_criteria = "wick"`` (default) a break uses the
  intraday extreme (``high > ib_high`` or ``low < ib_low``); with ``"close"`` it
  requires a bar to CLOSE beyond a level. Strict inequality — touching a level
  exactly is NOT a break, mirroring the sibling ``initial_balance`` convention.
  When the SAME bar breaks both sides first, the order is undetermined and the
  day is excluded; when neither side ever breaks the day's break outcome is
  ``neither``.

Contingency structure
----------------------
A 2x3 cross-tabulation, expressed in the condition/outcome row model: each of the
two conditions holds the same three outcomes, and an outcome's counts sum to its
condition's countable day count (``total``):

- Condition ``formed_high`` — the IB high formed first.
- Condition ``formed_low``  — the IB low formed first.
- Outcomes ``broke_high`` / ``broke_low`` / ``neither`` — which IB edge broke
  first afterward (or neither broke).

A day is countable for a condition only when its IB-formation order is determined,
its breakout window is non-empty, and its break order is determined (not a single
bar breaking both sides at once). Comparing ``P(broke_high | formed_high)`` against
``P(broke_high | formed_low)`` reveals whether early formation order predicts
breakout direction.

Results are computed for two IB lengths (30 min and 1 h) emitted as separate
timeframe entries, mirroring the ``initial_balance`` standard stat.

Declared slices
---------------
- ``weekday``     — the "by weekday" breakdown.
- ``close``       — split by session close color (green / red).
- ``prev_candle`` — split by the prior session's color.
- ``overnight``   — split by overnight gap direction.
- ``size``        — quartile buckets of the IB size (absolute points).
- ``size_pct``    — preset buckets of the IB size as a percent of price.
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
  en="Initial Balance Breakout — by Rejection",
  fr="Cassure de l'initial balance — par rejet",
)
_DEFINITION = I18nString(
  en=(
    "Does the order in which the initial balance edges form predict the first "
    "breakout? Cross-tabulates which edge (high or low) formed first during the "
    "initial balance against which edge broke first afterward (high, low, or "
    "neither)."
  ),
  fr=(
    "L'ordre de formation des bornes de l'initial balance prédit-il la première "
    "cassure ? Croise la borne (haut ou bas) formée en premier pendant l'initial "
    "balance avec la borne cassée en premier ensuite (haut, bas, ou ni l'un ni "
    "l'autre)."
  ),
)

# Two conditions (which edge formed first), each holding the same three outcomes
# (which edge broke first). The three outcomes partition each condition's
# countable days, so they sum to that condition's `total`.
_CONDITION_FORMED_HIGH = "formed_high"
_CONDITION_FORMED_LOW = "formed_low"

_LABELS = Labels(
  conditions={
    _CONDITION_FORMED_HIGH: I18nString(
      en="IB high formed first",
      fr="Haut de l'IB formé en premier",
    ),
    _CONDITION_FORMED_LOW: I18nString(
      en="IB low formed first",
      fr="Bas de l'IB formé en premier",
    ),
  },
  outcomes={
    "broke_high": I18nString(
      en="Broke the balance high first",
      fr="Cassure du haut de l'IB en premier",
    ),
    "broke_low": I18nString(
      en="Broke the balance low first",
      fr="Cassure du bas de l'IB en premier",
    ),
    "neither": I18nString(
      en="Broke neither side",
      fr="Aucune cassure",
    ),
  },
)

# Outcome enumeration, in display order. All three partition each condition's days.
_OUTCOMES: tuple[str, ...] = ("broke_high", "broke_low", "neither")
_CONDITIONS: tuple[str, ...] = (_CONDITION_FORMED_HIGH, _CONDITION_FORMED_LOW)

# Timeframe string → IB window length in minutes (mirrors standard.py exactly).
_TF_MINUTES: dict[str, int] = {
  "30min": 30,
  "1h": 60,
}

_BREAKOUT_CRITERIA: tuple[str, ...] = ("wick", "close")


class InitialBalanceRejection(BaseStat):
  """Contingency of which IB edge formed first vs. which broke first."""

  stat_name = "initial_balance_rejection"
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
    # Initial-balance window: [rth_start, ib_end). Breakout window: [ib_end, rth_end).
    self.ib_end_min: int = self.rth_start_min + self.ib_minutes

  # -------------------------------------------------------------------------
  # Day table construction
  # -------------------------------------------------------------------------
  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-session table with IB formation order and break order.

    Columns returned:
      ``ib_high`` / ``ib_low``       — IB extremes from the IB window.
      ``ib_size``                    — ``ib_high - ib_low`` (read by the ``size`` slicer).
      ``ib_size_pct``                — ``ib_size`` as % of the session open, in percent
                                       (e.g. 0.3 = 0.3%), read by the ``size_pct`` slicer.
      ``formed_high_first``          — nullable bool; True when the IB high was reached
                                       before the IB low. NaN when a single bar held both
                                       extremes (order undetermined → excluded).
      ``first_break_high_mod``       — minute-of-day of the first breakout-window bar that
                                       breaks the IB high; NaN if the high never breaks.
      ``first_break_low_mod``        — minute-of-day of the first breakout-window bar that
                                       breaks the IB low; NaN if the low never breaks.
      ``session_green``              — session color (``session_close >= session_open``),
                                       read by the ``close`` slicer.
      ``prev_session_green``         — prior session's color (NaN for the first resolved
                                       day), read by the ``prev_candle`` slicer.
      ``overnight_green``            — overnight gap direction: open above prior close
                                       (NaN for the first resolved day), read by the
                                       ``overnight`` slicer.

    The resolved-days core is shared via ``build_resolved_days``. A day is included
    only when it has both a clean IB window AND a non-empty breakout window (inner
    join). Days where price never breaks either side keep both break-minute columns
    NaN and are classified ``neither``. The DatetimeIndex (normalised session date)
    is required by the ``weekday`` slicer.
    """
    columns = [
      "ib_high",
      "ib_low",
      "ib_size",
      "ib_size_pct",
      "formed_high_first",
      "first_break_high_mod",
      "first_break_low_mod",
      "session_green",
      "prev_session_green",
      "overnight_green",
    ]
    empty = pd.DataFrame(columns=columns)

    if candles_df.empty:
      return empty

    # The shared base sorts by timestamp and resets the index, so the idxmax/idxmin
    # + .loc first-touch lookup in _formation_order over ``ib`` stays correct even
    # if the source parquet carried a non-unique index.
    base = build_ib_day_base(
      candles_df,
      self.rth_start_min,
      self.ib_end_min,
      self.rth_end_min,
      self.close_tolerance_min,
    )
    if base.day.empty:
      return empty

    day = base.day
    ib = base.ib
    post = base.post

    # IB formation order: which extreme was reached by the earliest bar.
    day["formed_high_first"] = self._formation_order(ib).reindex(day.index)

    # First-break minute per side over the breakout window (criteria-aware).
    first_high, first_low = self._break_orders(post, day["ib_high"], day["ib_low"])
    day["first_break_high_mod"] = first_high
    day["first_break_low_mod"] = first_low

    return day[columns]

  def _formation_order(self, ib: pd.DataFrame) -> pd.Series:
    """Per-date bool Series: True when the IB high was reached before the IB low.

    Uses ``idxmax`` / ``idxmin`` (first matching label, bars chronologically
    sorted upstream) to find the earliest bar reaching each extreme, then compares
    their minute-of-day. When the SAME bar holds both extremes the order is
    undetermined and the value is NaN (excluded downstream).
    """
    if ib.empty:
      return pd.Series(dtype=float, name="formed_high_first")

    grouped = ib.groupby("_date")
    hi_mod = ib.loc[grouped["high"].idxmax()].set_index("_date")["_mod"]
    lo_mod = ib.loc[grouped["low"].idxmin()].set_index("_date")["_mod"]
    formed_high_first = (hi_mod < lo_mod).astype(float)
    formed_high_first[hi_mod == lo_mod] = np.nan
    return formed_high_first.rename("formed_high_first")

  def _break_orders(
    self, post: pd.DataFrame, ib_high: pd.Series, ib_low: pd.Series
  ) -> tuple[pd.Series, pd.Series]:
    """First high-break and low-break minute per session, aligned to ``ib_high.index``.

    ``post`` is the full set of breakout-window bars (all dates). ``ib_high`` /
    ``ib_low`` are indexed by the surviving session dates. A break uses strict
    inequality; the ``breakout_criteria`` selects wick (intraday extreme) or close.
    Sessions with no breaking bar on a side are left NaN for that side.
    """
    nan_high = pd.Series(np.nan, index=ib_high.index, name="first_break_high_mod")
    nan_low = pd.Series(np.nan, index=ib_high.index, name="first_break_low_mod")
    if post.empty:
      return nan_high, nan_low

    p = post[["_date", "_mod", "high", "low", "close"]].copy()
    p["_ib_high"] = p["_date"].map(ib_high)
    p["_ib_low"] = p["_date"].map(ib_low)
    # Only bars on surviving sessions (both IB extremes present) can break.
    p = p[p["_ib_high"].notna() & p["_ib_low"].notna()]
    if p.empty:
      return nan_high, nan_low

    if self.breakout_criteria == "close":
      high_break = p["close"] > p["_ib_high"]
      low_break = p["close"] < p["_ib_low"]
    else:  # wick
      high_break = p["high"] > p["_ib_high"]
      low_break = p["low"] < p["_ib_low"]

    first_high = (
      p[high_break].groupby("_date")["_mod"].min().reindex(ib_high.index)
    ).rename("first_break_high_mod")
    first_low = (
      p[low_break].groupby("_date")["_mod"].min().reindex(ib_high.index)
    ).rename("first_break_low_mod")
    return first_high, first_low

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def _classify(self, day_table: pd.DataFrame) -> tuple[pd.Series, dict[str, pd.Series]]:
    """Return the countable mask and the three break-outcome masks.

    Break order is decided by the earlier first-break minute. A day is countable
    only when its formation order is determined AND its break order is determined
    (not a single bar breaking both sides at the same minute). ``neither`` is the
    legitimate outcome for a non-empty breakout window where price never breaks.
    """
    th = day_table["first_break_high_mod"]
    tl = day_table["first_break_low_mod"]
    formed = day_table["formed_high_first"]

    both = th.notna() & tl.notna()
    tie = both & (th == tl)
    countable = formed.notna() & ~tie

    broke_high = countable & th.notna() & (tl.isna() | (th < tl))
    broke_low = countable & tl.notna() & (th.isna() | (tl < th))
    neither = countable & th.isna() & tl.isna()
    return countable, {"broke_high": broke_high, "broke_low": broke_low, "neither": neither}

  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the 2x3 contingency rows over the (sliced) day table.

    Each condition (which edge formed first) holds the three break outcomes; an
    outcome's counts sum to its condition's countable-day ``total``. If
    ``baseline_rows`` is provided, merges its ``probability`` / ``total`` into each
    row's ``baseline_prob`` / ``baseline_n``.
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
      return [_make(cond, out, 0, 0) for cond in _CONDITIONS for out in _OUTCOMES]

    countable, outcome_masks = self._classify(day_table)
    formed_high = countable & day_table["formed_high_first"].fillna(False).astype(bool)
    formed_low = countable & ~day_table["formed_high_first"].fillna(True).astype(bool)
    cond_masks = {_CONDITION_FORMED_HIGH: formed_high, _CONDITION_FORMED_LOW: formed_low}

    rows: list[StatResultRow] = []
    for cond in _CONDITIONS:
      cond_mask = cond_masks[cond]
      total = int(cond_mask.sum())
      for out in _OUTCOMES:
        count = int((cond_mask & outcome_masks[out]).sum())
        rows.append(_make(cond, out, count, total))
    return rows

  def classify_samples(self, day_table: pd.DataFrame) -> list[SampleRow]:
    """One SampleRow per countable day, mirroring compute_rows' 2x3 contingency:
    the condition is which IB edge formed first (``formed_high`` / ``formed_low``)
    and the outcome is which edge broke first afterward (``broke_high`` /
    ``broke_low`` / ``neither``). A day is countable only when both the formation
    order and the break order are determined (see ``_classify``).
    """
    if day_table.empty:
      return []

    countable, outcome_masks = self._classify(day_table)
    if not bool(countable.any()):
      return []

    formed_high = day_table["formed_high_first"].fillna(False).astype(bool).to_numpy()
    cond = np.where(formed_high, _CONDITION_FORMED_HIGH, _CONDITION_FORMED_LOW)
    outcome = np.select(
      [
        outcome_masks["broke_high"].to_numpy(),
        outcome_masks["broke_low"].to_numpy(),
        outcome_masks["neither"].to_numpy(),
      ],
      ["broke_high", "broke_low", "neither"],
      default="neither",
    )

    dates = np.array(day_table.index.strftime("%Y-%m-%d"))
    idx = np.flatnonzero(countable.to_numpy())
    return [
      SampleRow(date=dates[i], condition=cond[i], outcome=outcome[i]) for i in idx
    ]

  # -------------------------------------------------------------------------
  # Baseline
  # -------------------------------------------------------------------------
  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random null for the formation-order / break-direction association.

    The null hypothesis is that formation order carries NO information about which
    edge breaks first. The ``formed_high_first`` label is randomly permuted ONLY
    among the countable days (formation order determined, break order determined),
    leaving every excluded day — undetermined formation (NaN) or a same-bar break
    tie — exactly where it is. Permuting within the countable set preserves each
    condition's countable total exactly (``baseline_n == total``) and keeps the
    break outcomes in place, so the break-outcome marginal is preserved too; only
    the formation/break pairing is randomised. Under this null both conditions
    converge to the overall break-outcome distribution, so the comparison reveals
    whether formation order genuinely shifts the breakout direction. Uses
    ``np.random.default_rng(seed)``.
    """
    n = len(day_table)
    if n == 0:
      return self.compute_rows(day_table, baseline_rows=None)

    countable, _ = self._classify(day_table)
    rng = np.random.default_rng(seed)
    tmp = day_table.copy()
    # Permute the formation label only across countable days; excluded days (NaN
    # formation or a break tie) keep their value so the countable set — and thus
    # each condition's total — is invariant under the null.
    arr = day_table["formed_high_first"].to_numpy().copy()
    idx = np.flatnonzero(countable.to_numpy())
    arr[idx] = arr[idx][rng.permutation(len(idx))]
    tmp["formed_high_first"] = arr

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
  """Load data and compute IB-by-rejection for both initial-balance lengths.

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
    stat = InitialBalanceRejection(
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
    stat_name="initial_balance_rejection",
    title=_TITLE,
    definition=_DEFINITION,
    labels=labels,
    instruments={instrument: merged_tf},
  )

  return write_results(final_result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Compute Initial Balance Breakout — by Rejection stat"
  )
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
