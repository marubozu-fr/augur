"""Initial Balance Breakout — Retracement stat.

On **single-break days only** — sessions where price breaks exactly one side of
the initial balance (IB) during the breakout window — measures **how deep price
pulls back into the IB range** after the break, expressed as a fraction of IB
size, and reports hit counts at configurable retracement thresholds (default
0.25 / 0.50 / 0.75 of IB size).

Definitions
-----------
- **Initial balance**: the high-low range of the bars in the IB window
  ``[rth_start, rth_start + ib_period)``. ``ib_high`` is the max high and
  ``ib_low`` the min low over that window; ``ib_size = ib_high - ib_low``.
- **Breakout window**: the rest of the session ``[rth_start + ib_period,
  rth_end)``. The IB itself is excluded so the range can never break its own bars.
- **Break**: strict inequality — touching the level exactly is NOT a break,
  mirroring the sibling ``initial_balance`` / ``opening_range_breakout``
  convention. With ``breakout_criteria = "wick"`` (default) a break uses the
  breakout window's intraday extremes; with ``"close"`` a break requires a bar to
  CLOSE beyond the level.
- **Single-break day**: exactly one side of the IB breaks during the breakout
  window (``broke_high`` XOR ``broke_low``). Days that break both sides, or
  neither, are NOT single-break days and are excluded from every denominator.
- **Retracement**: after the break, how deep price pulls back into the IB. From
  the *chronologically first* break bar onward, the deepest intraday penetration
  back toward the opposite side, measured from the broken edge:
    - up-break (``broke_high``): ``ib_high - min(low)`` over the bars at and after
      the first up-break bar (the deepest pull-back DOWN into the range).
    - down-break (``broke_low``): ``max(high) - ib_low`` over the bars at and after
      the first down-break bar (the deepest pull-back UP into the range).
  The pull-back is always measured by the intraday extreme (wick) — "how deep
  price pulls back" is a physical penetration regardless of how the break itself
  is detected. ``depth_frac = retracement_points / ib_size``, clipped at 0 (a
  price that never returns toward the range retraces 0). On a single-break day the
  held side guarantees ``depth_frac <= 1.0``, so ``depth_frac`` lies in ``[0, 1]``.

Three threshold outcomes are reported under a single ``"single_break"`` condition.
Each outcome is the probability that ``depth_frac`` reaches at least that fraction
of IB size, so the outcomes are NESTED (a day hitting 0.75 also hits 0.50 and
0.25) rather than a partition. ``total_samples`` counts ALL resolved sessions;
each row's ``total`` is the single-break sessions (the denominator). Non-single-break
sessions stay in ``total_samples`` but are excluded from every denominator.

Results are computed for two IB lengths (30 min and 1 h) emitted as separate
timeframe entries, exactly mirroring the ``initial_balance`` standard stat.

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
  en="Initial Balance Breakout — Retracement",
  fr="Cassure de l'initial balance — Repli",
)
_DEFINITION = I18nString(
  en=(
    "On single-break days, after price breaks one side of the initial balance, "
    "how deep does it pull back into the balance range? Reported as the share of "
    "single-break sessions whose retracement reaches at least 25%, 50%, or 75% of "
    "the initial-balance size."
  ),
  fr=(
    "Les jours à cassure unique, après que le prix casse un côté de l'initial "
    "balance, jusqu'où revient-il dans le range de la balance ? Exprimé comme la "
    "proportion des séances à cassure unique dont le repli atteint au moins 25 %, "
    "50 % ou 75 % de la taille de l'initial balance."
  ),
)

# Single condition: every single-break session belongs to it.
_CONDITION = "single_break"

# Default retracement thresholds (fraction of IB size). Configurable per run.
_DEFAULT_THRESHOLDS: tuple[float, ...] = (0.25, 0.50, 0.75)


def _outcome_key(threshold: float) -> str:
  """Outcome key for a threshold, e.g. 0.25 -> 'retrace_25', 0.5 -> 'retrace_50'."""
  return f"retrace_{int(round(threshold * 100))}"


def _build_labels(thresholds: tuple[float, ...]) -> Labels:
  """Build the i18n Labels for a given set of retracement thresholds."""
  outcomes: dict[str, I18nString] = {}
  for t in thresholds:
    pct = int(round(t * 100))
    outcomes[_outcome_key(t)] = I18nString(
      en=f"Retraced ≥{pct}% of IB",
      fr=f"Repli ≥{pct}% de l'IB",
    )
  return Labels(
    conditions={
      "single_break": I18nString(en="Single-break day", fr="Jour à cassure unique"),
    },
    outcomes=outcomes,
  )


# Labels for the default thresholds (used standalone; run() rebuilds per thresholds).
_LABELS = _build_labels(_DEFAULT_THRESHOLDS)

# Timeframe string → IB window length in minutes (mirrors standard.py exactly).
_TF_MINUTES: dict[str, int] = {
  "30min": 30,
  "1h": 60,
}

_BREAKOUT_CRITERIA: tuple[str, ...] = ("wick", "close")


class InitialBalanceRetracement(BaseStat):
  """Retracement-depth hit rates after a single-side initial-balance breakout."""

  stat_name = "initial_balance_retracement"
  title = _TITLE
  definition = _DEFINITION
  slices = (
    "weekday",
    "close",
    "prev_candle",
    "overnight",
    SizeBucket(column="ib_size", preset="quartiles", name="size"),
    # IB size as % of price, in preset bands (<0.2%, 0.2–0.4%, 0.4–0.6%,
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
    thresholds: tuple[float, ...] = _DEFAULT_THRESHOLDS,
    close_tolerance_min: int = 15,
  ) -> None:
    if timeframe not in _TF_MINUTES:
      raise ValueError(
        f"Unsupported timeframe '{timeframe}'. Choose from {list(_TF_MINUTES)}"
      )
    if breakout_criteria not in _BREAKOUT_CRITERIA:
      raise ValueError(
        f"Unsupported breakout_criteria '{breakout_criteria}'. "
        f"Choose from {list(_BREAKOUT_CRITERIA)}"
      )
    if not thresholds:
      raise ValueError("thresholds must be a non-empty sequence")
    if any(t <= 0.0 for t in thresholds):
      raise ValueError("every threshold must be > 0 (a fraction of IB size)")

    self.instrument = instrument
    self.timeframe = timeframe
    self.config = config
    self.ib_minutes = _TF_MINUTES[timeframe]
    self.breakout_criteria = breakout_criteria
    self.thresholds = tuple(thresholds)
    self.close_tolerance_min = close_tolerance_min
    # Per-instance labels so configurable thresholds stay self-documenting.
    self.labels = _build_labels(self.thresholds)

    rth = config.sessions["rth"]
    self.rth_start_min: int = minute_of_day(rth.start)
    self.rth_end_min: int = minute_of_day(rth.end)
    # Initial-balance window: [rth_start, ib_end). Breakout window: [ib_end, rth_end).
    self.ib_end_min: int = self.rth_start_min + self.ib_minutes

  # -------------------------------------------------------------------------
  # Session-level retracement helper
  # -------------------------------------------------------------------------
  def _compute_retracement(
    self,
    bars: pd.DataFrame,
    ib_high: float,
    ib_low: float,
    break_up: bool,
  ) -> float:
    """Deepest retracement into the IB after the first break, in points (>= 0).

    ``bars`` is the chronologically sorted breakout-window slice for one
    single-break session. ``break_up`` is True for an up-break (``broke_high``),
    False for a down-break (``broke_low``). The first break bar is located with the
    same criteria used for the directional classification; the pull-back itself is
    always measured from the intraday extreme (wick).

      - up-break: first bar with ``high > ib_high`` (wick) or ``close > ib_high``
        (close); from there, ``ib_high - min(low)`` over the remaining bars.
      - down-break: first bar with ``low < ib_low`` (wick) or ``close < ib_low``
        (close); from there, ``max(high) - ib_low`` over the remaining bars.

    Returns ``0.0`` when price never pulls back toward the range (negative raw
    penetration is clipped to 0). The caller divides by ``ib_size`` to get the
    retracement fraction.
    """
    n = len(bars)
    if n == 0:
      return 0.0

    highs = bars["high"].to_numpy(dtype=float)
    lows = bars["low"].to_numpy(dtype=float)
    closes = bars["close"].to_numpy(dtype=float)

    if break_up:
      if self.breakout_criteria == "close":
        first = next((i for i in range(n) if closes[i] > ib_high), None)
      else:
        first = next((i for i in range(n) if highs[i] > ib_high), None)
      if first is None:
        return 0.0
      # Deepest pull-back DOWN into the range from the first up-break bar onward.
      retrace_low = float(lows[first:].min())
      return max(0.0, ib_high - retrace_low)

    # Down-break.
    if self.breakout_criteria == "close":
      first = next((i for i in range(n) if closes[i] < ib_low), None)
    else:
      first = next((i for i in range(n) if lows[i] < ib_low), None)
    if first is None:
      return 0.0
    # Deepest pull-back UP into the range from the first down-break bar onward.
    retrace_high = float(highs[first:].max())
    return max(0.0, retrace_high - ib_low)

  # -------------------------------------------------------------------------
  # Day table construction
  # -------------------------------------------------------------------------
  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-session table with IB extremes and single-break retracement depth.

    Columns returned:
      ``ib_high`` / ``ib_low``   — IB extremes from the IB window.
      ``ib_size``                — ``ib_high - ib_low`` (read by the ``size`` slicer).
      ``ib_size_pct``            — ``ib_size`` as % of the session open, in percent
                                   (e.g. 0.3 = 0.3%), read by the ``size_pct`` slicer.
      ``break_up``               — True for ``broke_high``, False for ``broke_low``,
                                   NaN for both/neither (not a single-break day).
      ``depth_frac``             — retracement depth as a fraction of IB size on
                                   single-break days; NaN on non-single-break days
                                   (excluded from every denominator).
      ``session_green``          — session color (``session_close >= session_open``),
                                   read by the ``close`` slicer.
      ``prev_session_green``     — prior session's color (NaN for the first resolved
                                   day), read by the ``prev_candle`` slicer.
      ``overnight_green``        — overnight gap direction: open above prior close
                                   (NaN for the first resolved day), read by the
                                   ``overnight`` slicer.

    The resolved-days core is shared via ``build_resolved_days``. A day is included
    only when it has both a clean IB window AND a non-empty breakout window (inner
    join). Non-single-break days (both sides broke, or neither) remain in the table
    with ``depth_frac = NaN`` and are excluded from every denominator. The
    chronological retracement is computed per single-break session by
    ``_compute_retracement`` and joined back. The DatetimeIndex (normalised session
    date) is required by the ``weekday`` slicer.
    """
    columns = [
      "ib_high",
      "ib_low",
      "ib_size",
      "ib_size_pct",
      "break_up",
      "depth_frac",
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
    # Breakout-window extremes used to classify the break direction (the criteria-
    # selected extremes; close extremes used only with the close criteria). Joined
    # onto the presence-guarded base, so every surviving day has them.
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

    # Break-direction classification (strict inequality; criteria-selected extremes).
    if self.breakout_criteria == "close":
      high_extreme, low_extreme = day["post_close_high"], day["post_close_low"]
    else:
      high_extreme, low_extreme = day["post_high"], day["post_low"]
    above = high_extreme > day["ib_high"]
    below = low_extreme < day["ib_low"]
    broke_high = above & ~below
    broke_low = below & ~above

    # break_up: True (broke_high), False (broke_low), NaN otherwise (not single break).
    day["break_up"] = pd.Series(np.nan, index=day.index, dtype=object)
    day.loc[broke_high, "break_up"] = True
    day.loc[broke_low, "break_up"] = False

    # Per-session chronological retracement, computed only for single-break days.
    post_sorted = post.sort_values("timestamp")
    ib_high_s = day["ib_high"]
    ib_low_s = day["ib_low"]
    single_break = broke_high | broke_low
    single_dates = set(day.index[single_break])
    up_dates = set(day.index[broke_high])

    records: list[dict] = []
    for date, grp in post_sorted.groupby("_date"):
      if date not in single_dates:
        continue
      points = self._compute_retracement(
        grp,
        float(ib_high_s.at[date]),
        float(ib_low_s.at[date]),
        break_up=date in up_dates,
      )
      ib_sz = float(day["ib_size"].at[date])
      frac = points / ib_sz if ib_sz > 0 else np.nan
      records.append({"_date": date, "depth_frac": frac})

    if records:
      depth_df = pd.DataFrame(records).set_index("_date")
      day = day.join(depth_df, how="left")
    else:
      day["depth_frac"] = np.nan

    return day[columns]

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the per-threshold retracement hit rows over the (sliced) day table.

    Countable sessions are the single-break days (``depth_frac`` not NaN);
    ``total`` always equals that count. For each threshold ``t``, ``count`` is the
    number of countable sessions whose ``depth_frac >= t`` (the retracement reached
    at least ``t`` of IB size). Outcomes are nested, not a partition. If
    ``baseline_rows`` is provided, the matching row's ``probability`` / ``total``
    are merged into ``baseline_prob`` / ``baseline_n``.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    if day_table.empty:
      depth = pd.Series([], dtype=float)
    else:
      depth = pd.to_numeric(day_table["depth_frac"], errors="coerce").dropna()
    total = int(len(depth))

    rows: list[StatResultRow] = []
    for t in self.thresholds:
      outcome = _outcome_key(t)
      count = int((depth >= t).sum()) if total > 0 else 0
      bl = baseline_map.get((_CONDITION, outcome))
      rows.append(
        StatResultRow(
          condition=_CONDITION,
          outcome=outcome,
          count=count,
          total=total,
          probability=count / total if total > 0 else 0.0,
          baseline_prob=bl.probability if bl else 0.0,
          baseline_n=bl.total if bl else 0,
        )
      )
    return rows

  def classify_samples(self, day_table: pd.DataFrame) -> list[SampleRow]:
    """One SampleRow per (countable day, threshold reached), mirroring compute_rows'
    nested threshold hit-rate outcomes. Countable days are single-break sessions
    (``depth_frac`` not NaN). Outcomes are NESTED, not a partition, so a day
    contributes zero, one, or several samples depending on how many thresholds its
    ``depth_frac`` reaches — never a ``value`` (this is a probability channel, not
    a magnitude one).
    """
    if day_table.empty:
      return []

    depth = pd.to_numeric(day_table["depth_frac"], errors="coerce")
    countable = depth.notna()
    if not bool(countable.any()):
      return []

    dates = np.array(day_table.index[countable].strftime("%Y-%m-%d"))
    depth_ct = depth[countable].to_numpy()

    samples: list[SampleRow] = []
    for t in self.thresholds:
      outcome = _outcome_key(t)
      for date in dates[depth_ct >= t]:
        samples.append(SampleRow(date=date, condition=_CONDITION, outcome=outcome))
    return samples

  # -------------------------------------------------------------------------
  # Baseline
  # -------------------------------------------------------------------------
  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random null for retracement depth. Deterministic for a fixed seed.

    The null hypothesis is that the deepest pull-back lands UNIFORMLY at random
    within the IB range — i.e. the breakout carries no information about how far
    price retraces. Each single-break session is assigned a random
    ``depth_frac ~ Uniform(0, 1)``; non-single-break sessions keep their NaN and
    stay excluded (pending-sample discipline). The same ``compute_rows`` path then
    counts threshold hits, so ``P(hit >= t)`` converges to ``1 - t``. Comparing the
    actual hit rate to this null reveals whether real retracements are deeper
    (above baseline) or shallower (below baseline) than a random pull-back. Uses
    ``np.random.default_rng(seed)``.
    """
    n = len(day_table)
    if n == 0:
      return self.compute_rows(day_table, baseline_rows=None)

    rng = np.random.default_rng(seed)
    draws = rng.random(n)
    countable = pd.to_numeric(day_table["depth_frac"], errors="coerce").notna().to_numpy()
    depth_bl = np.where(countable, draws, np.nan)

    tmp = day_table.copy()
    tmp["depth_frac"] = depth_bl
    return self.compute_rows(tmp, baseline_rows=None)


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
  breakout_criteria: str = "wick",
  thresholds: tuple[float, ...] = _DEFAULT_THRESHOLDS,
) -> Path:
  """Load data and compute IB retracement for both initial-balance lengths.

  Merges the per-timeframe TimeframeResults into one StatRunResult and writes the
  consolidated JSON to results/. Slice dimension labels are merged across
  timeframes so no timeframe's dimensions overwrite another's.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  timeframes = ["30min", "1h"]
  merged_tf: dict[str, TimeframeResult] = {}
  labels = _build_labels(tuple(thresholds))

  for tf in timeframes:
    stat = InitialBalanceRetracement(
      instrument=instrument,
      timeframe=tf,
      config=config,
      breakout_criteria=breakout_criteria,
      thresholds=thresholds,
    )
    result = stat.compute(candles_df)
    merged_tf[tf] = result.instruments[instrument][tf]
    # Merge framework-enriched slice dimension labels so each timeframe contributes
    # without overwriting earlier entries.
    labels.dimensions.update(result.labels.dimensions)

  final_result = StatRunResult(
    stat_name="initial_balance_retracement",
    title=_TITLE,
    definition=_DEFINITION,
    labels=labels,
    instruments={instrument: merged_tf},
  )

  return write_results(final_result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Compute Initial Balance Breakout — Retracement stat"
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
  parser.add_argument(
    "--thresholds",
    default=None,
    help="Comma-separated retracement thresholds as fractions of IB size "
    "(default: 0.25,0.5,0.75)",
  )
  args = parser.parse_args()

  thresholds_arg = _DEFAULT_THRESHOLDS
  if args.thresholds:
    thresholds_arg = tuple(float(x) for x in args.thresholds.split(","))

  output_path = run(
    instrument=args.instrument,
    config_dir=args.config_dir,
    data_path=args.data_path,
    breakout_criteria=args.breakout_criteria,
    thresholds=thresholds_arg,
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
