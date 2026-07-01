"""Initial Balance Breakout — Performance stat.

Measures the **magnitude of the first breakout** from the initial balance (IB):
after the IB forms during the first ``ib_period`` minutes of the RTH session,
this stat tracks how far price extends past the IB high or low on its
*chronologically first* excursion, before trading back into the balance range.

Definitions
-----------
- **Initial balance**: the high-low range of the bars in the IB window
  ``[rth_start, rth_start + ib_period)``. ``ib_high`` is the max high and
  ``ib_low`` the min low over that window.
- **Breakout window**: the rest of the session ``[rth_start + ib_period,
  rth_end)``. The IB itself is excluded so the range can never break its own bars.
- **Break**: strict inequality — touching the level exactly is NOT a break,
  mirroring the sibling ``initial_balance`` / ``opening_range_breakout``
  convention. With ``breakout_criteria = "wick"`` (default) a break uses the
  intraday extreme (any wick beyond the level counts); with ``"close"`` a break
  requires a bar to CLOSE beyond the level.
- **First up excursion**: the run of consecutive bars starting at the first
  up-break bar (``high > ib_high`` or ``close > ib_high``) that remain above
  ``ib_high``, until a bar re-enters the IB range. ``up_ext`` is the maximum
  extension past ``ib_high`` during that excursion (points, always > 0).
- **First down excursion**: the symmetric mirror past ``ib_low``.
- **First breakout direction**: the side whose first-break bar occurs earliest
  in the session. If both sides first break on the same bar (possible with the
  wick criteria only), the larger extension wins as a deterministic tiebreak.
- **extension**: the first-breakout extension in points on the winning side.
  Sessions where price never breaks either side are excluded from every row's
  denominator.
- **extension_pct**: ``extension / session_open`` (a decimal, e.g. 0.012 = 1.2%),
  following the same convention as ``session_reversal_range``.

``up_ext`` and ``down_ext`` are each computed from their OWN first-break bar,
independently of which side broke first overall, so the random baseline can pick
a side at random without losing information.

Four magnitude outcomes are reported — mean / max in points and in percent — under
a single ``"ib"`` condition. Sessions that never break the IB are excluded from
every denominator; ``count`` / ``total`` always equals the number of breaking
sessions.

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
  en="Initial Balance Breakout — Performance",
  fr="Cassure de l'initial balance — Performance",
)
_DEFINITION = I18nString(
  en=(
    "After the initial balance forms, how far does price extend past the balance "
    "high or low on its FIRST breakout, before breaking back into the range? "
    "Reported as the average and maximum first-breakout extension, in points and "
    "as a percentage of the session open."
  ),
  fr=(
    "Après la formation de l'initial balance, de combien le prix s'étend-il "
    "au-delà du haut ou du bas de la balance lors de sa PREMIÈRE cassure, avant "
    "de revenir dans le range ? Exprimé en tant qu'extension moyenne et maximale "
    "de la première cassure, en points et en pourcentage de l'ouverture de session."
  ),
)
_LABELS = Labels(
  conditions={
    "ib": I18nString(en="Initial balance", fr="Initial balance"),
  },
  outcomes={
    "mean_extension": I18nString(
      en="Average first-breakout extension (points)",
      fr="Extension moyenne de la première cassure (points)",
    ),
    "mean_extension_pct": I18nString(
      en="Average first-breakout extension (%)",
      fr="Extension moyenne de la première cassure (%)",
    ),
    "max_extension": I18nString(
      en="Maximum first-breakout extension (points)",
      fr="Extension maximale de la première cassure (points)",
    ),
    "max_extension_pct": I18nString(
      en="Maximum first-breakout extension (%)",
      fr="Extension maximale de la première cassure (%)",
    ),
  },
)

# Single condition: every countable session (at least one side broke) belongs to it.
_CONDITION = "ib"

# Outcome key, day-table column, aggregation — in display order.
_OUTCOMES: list[tuple[str, str, str]] = [
  ("mean_extension", "extension", "mean"),
  ("mean_extension_pct", "extension_pct", "mean"),
  ("max_extension", "extension", "max"),
  ("max_extension_pct", "extension_pct", "max"),
]

# Timeframe string → IB window length in minutes (mirrors standard.py exactly).
_TF_MINUTES: dict[str, int] = {
  "30min": 30,
  "1h": 60,
}

_BREAKOUT_CRITERIA: tuple[str, ...] = ("wick", "close")


class InitialBalancePerformance(BaseStat):
  """Average and maximum first-breakout extension from the initial balance."""

  stat_name = "initial_balance_performance"
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
      raise ValueError(
        f"Unsupported timeframe '{timeframe}'. Choose from {list(_TF_MINUTES)}"
      )
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
  # Session-level extension helper
  # -------------------------------------------------------------------------
  def _compute_session_extensions(
    self,
    bars: pd.DataFrame,
    ib_high: float,
    ib_low: float,
  ) -> tuple[float, float, bool | float]:
    """Compute (up_ext, down_ext, first_dir_up) for one session's breakout window.

    ``bars`` is the slice of breakout-window bars for a single session, already
    sorted chronologically. ``ib_high`` / ``ib_low`` are that session's
    initial-balance extremes.

    ``up_ext`` and ``down_ext`` are each computed from their OWN first-break bar,
    independently of which side broke first overall (so the baseline can pick a
    side at random without losing the magnitude information of the unbroken side).

    Break detection uses strict inequality (touching the level is NOT a break):

      - wick:  up-break when ``high > ib_high``; down-break when ``low < ib_low``.
      - close: up-break when ``close > ib_high``; down-break when ``close < ib_low``.

    Excursion extension (the "before breaking back into the range" part):

      - wick up:   from the first up-break bar, for each bar update
                   ``max(high) - ib_high`` FIRST, then stop if ``low < ib_high``
                   (bar traded back into the range from above; its high is still
                   counted). The start bar always has high > ib_high so up_ext > 0;
                   a one-bar excursion ends immediately if that bar also re-enters.
      - close up:  from the first up-break bar, walk while ``close > ib_high``,
                   tracking ``max(close) - ib_high``; stop at the FIRST bar with
                   ``close <= ib_high`` without counting it. up_ext always > 0
                   because the start bar has close > ib_high.
      - wick / close down: symmetric mirrors (``ib_low - low`` / ``ib_low - close``;
                   re-entry condition ``high > ib_low`` / ``close >= ib_low``).

    ``first_dir_up``:
      - Neither side broke        → ``np.nan`` (session excluded from the stat).
      - Only up broke             → ``True``.
      - Only down broke           → ``False``.
      - Both broke, up bar first  → ``True``; down bar first → ``False``.
      - Same first-break bar (wick only): tiebreak ``bool(up_ext >= down_ext)``
        (larger excursion wins; deterministic).

    Returns ``0.0`` for the extension of a side that never broke.
    """
    n = len(bars)
    if n == 0:
      return 0.0, 0.0, np.nan

    # Extract raw arrays for fast indexed access (avoids repeated Series overhead).
    highs = bars["high"].to_numpy(dtype=float)
    lows = bars["low"].to_numpy(dtype=float)
    closes = bars["close"].to_numpy(dtype=float)

    # Locate the first break bar for each side (positional, 0-based within bars).
    if self.breakout_criteria == "wick":
      up_pos: int | None = next(
        (i for i in range(n) if highs[i] > ib_high), None
      )
      down_pos: int | None = next(
        (i for i in range(n) if lows[i] < ib_low), None
      )
    else:  # close
      up_pos = next((i for i in range(n) if closes[i] > ib_high), None)
      down_pos = next((i for i in range(n) if closes[i] < ib_low), None)

    has_up = up_pos is not None
    has_down = down_pos is not None

    if not has_up and not has_down:
      return 0.0, 0.0, np.nan

    # --- Compute up_ext from the first up-break bar onward ---
    up_ext = 0.0
    if has_up:
      max_ext = 0.0
      if self.breakout_criteria == "close":
        for i in range(up_pos, n):  # type: ignore[arg-type]
          if closes[i] > ib_high:
            cand = closes[i] - ib_high
            if cand > max_ext:
              max_ext = cand
          else:
            break  # re-entered; do not count this bar
      else:  # wick: update first, THEN check re-entry
        for i in range(up_pos, n):  # type: ignore[arg-type]
          cand = highs[i] - ib_high
          if cand > max_ext:
            max_ext = cand
          if lows[i] < ib_high:
            break  # bar traded back into range; high already counted above
      up_ext = max_ext

    # --- Compute down_ext from the first down-break bar onward ---
    down_ext = 0.0
    if has_down:
      max_ext = 0.0
      if self.breakout_criteria == "close":
        for i in range(down_pos, n):  # type: ignore[arg-type]
          if closes[i] < ib_low:
            cand = ib_low - closes[i]
            if cand > max_ext:
              max_ext = cand
          else:
            break  # re-entered; do not count this bar
      else:  # wick: update first, THEN check re-entry
        for i in range(down_pos, n):  # type: ignore[arg-type]
          cand = ib_low - lows[i]
          if cand > max_ext:
            max_ext = cand
          if highs[i] > ib_low:
            break  # bar traded back into range; low already counted above
      down_ext = max_ext

    # --- Determine first_dir_up ---
    if has_up and not has_down:
      return up_ext, down_ext, True
    if has_down and not has_up:
      return up_ext, down_ext, False
    # Both sides broke; compare chronological positions.
    if up_pos < down_pos:  # type: ignore[operator]
      return up_ext, down_ext, True
    if down_pos < up_pos:  # type: ignore[operator]
      return up_ext, down_ext, False
    # Same bar (wick only): tiebreak by larger excursion (deterministic).
    return up_ext, down_ext, bool(up_ext >= down_ext)

  # -------------------------------------------------------------------------
  # Day table construction
  # -------------------------------------------------------------------------
  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-session table with IB extremes and first-breakout extensions.

    Columns returned:
      ``ib_high`` / ``ib_low``   — IB extremes from the IB window.
      ``ib_size``                — ``ib_high - ib_low`` (read by the ``size`` slicer).
      ``ib_size_pct``            — ``ib_size`` as % of the session open, in percent
                                   (e.g. 0.3 = 0.3%), read by the ``size_pct`` slicer.
      ``session_open``           — open of the RTH session-open bar; used as the
                                   denominator for ``extension_pct``.
      ``up_ext``                 — maximum extension past ``ib_high`` during the
                                   first up excursion (0.0 if no up break); retained
                                   for the baseline so it can pick a side at random.
      ``down_ext``               — symmetric extension past ``ib_low`` (0.0 if no
                                   down break); retained for the baseline.
      ``first_dir_up``           — True / False / NaN; retained for the baseline to
                                   exclude non-breaking sessions.
      ``extension``              — first-breakout extension in points: ``up_ext``
                                   where ``first_dir_up`` is True, ``down_ext``
                                   where False, NaN where neither side broke.
      ``extension_pct``          — ``extension / session_open`` (decimal, e.g.
                                   0.0005 = 0.05%).
      ``session_green``          — session color (``session_close >= session_open``),
                                   read by the ``close`` slicer.
      ``prev_session_green``     — prior session's color (NaN for the first resolved
                                   day), read by the ``prev_candle`` slicer.
      ``overnight_green``        — overnight gap direction: open above prior close
                                   (NaN for the first resolved day), read by the
                                   ``overnight`` slicer.

    The resolved-days core is shared via ``build_resolved_days``. A day is
    included in the table only when it has both a clean IB window AND a non-empty
    breakout window (inner join). Days where price never breaks either side during
    the breakout window remain in the table but have ``extension = NaN`` and are
    therefore excluded from every denominator.

    The chronological first-excursion computation is inherently sequential and is
    performed by ``_compute_session_extensions`` per session (small Python loop).
    The data volume (~5 000 sessions at most for a 20-year history) makes this
    practical; results are vectorised back via a join. The DatetimeIndex
    (normalised session date) is required by the ``weekday`` slicer.
    """
    columns = [
      "ib_high",
      "ib_low",
      "ib_size",
      "ib_size_pct",
      "session_open",
      "up_ext",
      "down_ext",
      "first_dir_up",
      "extension",
      "extension_pct",
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

    day = base.day
    post = base.post

    # Per-session chronological first-excursion computation.
    # Sort once so groupby preserves chronological order within each group.
    post_sorted = post.sort_values("timestamp")
    ib_high_s = day["ib_high"]
    ib_low_s = day["ib_low"]
    resolved_dates = set(day.index)

    records: list[dict] = []
    for date, grp in post_sorted.groupby("_date"):
      if date not in resolved_dates:
        continue
      up_e, dn_e, fdu = self._compute_session_extensions(
        grp,
        float(ib_high_s.at[date]),
        float(ib_low_s.at[date]),
      )
      records.append(
        {
          "_date": date,
          "up_ext": up_e,
          "down_ext": dn_e,
          # True / False / np.nan: stored as object dtype so pd.isna() detects NaN.
          "first_dir_up": fdu,
        }
      )

    if records:
      ext_df = pd.DataFrame(records).set_index("_date")
      day = day.join(ext_df, how="left")
    else:
      day["up_ext"] = np.nan
      day["down_ext"] = np.nan
      day["first_dir_up"] = np.nan

    # extension = up_ext where first_dir_up is True,
    #             down_ext where first_dir_up is False,
    #             NaN where neither side broke (first_dir_up is NaN).
    # Use .loc assignment to avoid the FutureWarning on chained indexing.
    day["extension"] = pd.Series(np.nan, index=day.index, dtype=float)
    up_mask = day["first_dir_up"] == True  # noqa: E712
    dn_mask = day["first_dir_up"] == False  # noqa: E712
    day.loc[up_mask, "extension"] = day.loc[up_mask, "up_ext"]
    day.loc[dn_mask, "extension"] = day.loc[dn_mask, "down_ext"]
    day["extension_pct"] = day["extension"] / day["session_open"]

    return day[columns]

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the four first-breakout magnitude rows over the (sliced) day table.

    Countable sessions are those where price broke at least one side of the IB
    (``extension`` is not NaN). ``count`` / ``total`` always equals that number.
    Slices re-run this method over a subset of the day table; the ``extension``
    NaN guard applies per-subset, so a slice group's countable count may differ
    from the overall total.

    Each row carries its aggregate metric in ``value`` (mean or max of
    ``extension`` / ``extension_pct`` over the countable subset). The
    ``probability`` channel is left at ``0.0`` (this is a magnitude stat). If
    ``baseline_rows`` is provided, the matching row's ``value`` is merged into
    ``value_baseline`` and its ``total`` into ``baseline_n``.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    if day_table.empty:
      countable_total = 0
    else:
      countable_total = int(day_table["extension"].notna().sum())

    countable_mask = day_table["extension"].notna() if not day_table.empty else pd.Series(
      [], dtype=bool
    )

    rows: list[StatResultRow] = []
    for out_key, column, agg in _OUTCOMES:
      if countable_total > 0:
        series = day_table.loc[countable_mask, column].astype(float)
        value = float(series.mean() if agg == "mean" else series.max())
      else:
        value = 0.0
      bl = baseline_map.get((_CONDITION, out_key))
      rows.append(
        StatResultRow(
          condition=_CONDITION,
          outcome=out_key,
          count=countable_total,
          total=countable_total,
          probability=0.0,
          baseline_prob=0.0,
          baseline_n=bl.total if bl else 0,
          value=value,
          value_baseline=bl.value if bl else None,
        )
      )
    return rows

  def classify_samples(self, day_table: pd.DataFrame) -> list[SampleRow]:
    """Four SampleRows per countable day (one per magnitude outcome), mirroring
    compute_rows' four columns. Countable sessions are those where price broke at
    least one side of the IB (``extension`` not NaN). ``mean_extension`` and
    ``max_extension`` both carry the day's raw ``extension`` (points), while
    ``mean_extension_pct`` / ``max_extension_pct`` carry ``extension_pct``, so the
    API can recompute either aggregate (mean or max) over a date-filtered subset.
    """
    if day_table.empty:
      return []

    countable = day_table["extension"].notna()
    if not bool(countable.any()):
      return []

    ct = day_table[countable]
    dates = np.array(ct.index.strftime("%Y-%m-%d"))

    samples: list[SampleRow] = []
    for out_key, column, _agg in _OUTCOMES:
      values = ct[column].to_numpy(dtype=float)
      for date, value in zip(dates, values):
        samples.append(
          SampleRow(date=date, condition=_CONDITION, outcome=out_key, value=float(value))
        )
    return samples

  # -------------------------------------------------------------------------
  # Baseline
  # -------------------------------------------------------------------------
  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random null baseline for first-breakout extension. Deterministic for a fixed seed.

    The null hypothesis is that the chronological direction of the first breakout
    carries no information about the magnitude of the extension beyond a randomly
    chosen side. For each session a fair coin picks whether to use ``up_ext`` or
    ``down_ext`` as the baseline extension:

      ext_bl = up_ext   if coin is heads,
               down_ext  if coin is tails.

    Sessions with no breakout on either side (``first_dir_up`` is NaN) keep their
    NaN ``extension`` in the temporary table so they remain excluded from the
    countable denominator (pending-sample discipline). For a session that broke
    only one side, the unbroken side contributes a 0.0 extension, which the coin
    may randomly select; those zero values pull the baseline mean down relative to
    the actual (which always picks the breaking side).

    A temporary copy of the day table is built with the randomised ``extension``
    and ``extension_pct``, then the same ``compute_rows`` path is reused to
    compute means and maxima. Uses ``np.random.default_rng(seed)``.
    """
    n = len(day_table)
    if n == 0:
      return self.compute_rows(day_table, baseline_rows=None)

    rng = np.random.default_rng(seed)
    pick_up = rng.random(n) < 0.5

    up_ext_arr = day_table["up_ext"].to_numpy(dtype=float)
    down_ext_arr = day_table["down_ext"].to_numpy(dtype=float)
    ext_bl = np.where(pick_up, up_ext_arr, down_ext_arr).astype(float)

    # Restore NaN for sessions that never broke either side; their first_dir_up
    # is NaN, and they must remain excluded from every denominator.
    no_break = day_table["first_dir_up"].isna().to_numpy()
    ext_bl[no_break] = np.nan

    session_open = day_table["session_open"].to_numpy(dtype=float)
    # NaN propagates through division wherever ext_bl is NaN.
    ext_pct_bl = (ext_bl / session_open).astype(float)

    tmp = day_table.copy()
    tmp["extension"] = ext_bl
    tmp["extension_pct"] = ext_pct_bl

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
  """Load data and compute IB performance for both initial-balance lengths.

  Merges the per-timeframe TimeframeResults into one StatRunResult and writes the
  consolidated JSON to results/. Slice dimension labels are merged across
  timeframes so no timeframe's dimensions overwrite another's.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  timeframes = ["30min", "1h"]
  merged_tf: dict[str, TimeframeResult] = {}
  labels = _LABELS.model_copy(deep=True)

  for tf in timeframes:
    stat = InitialBalancePerformance(
      instrument=instrument,
      timeframe=tf,
      config=config,
      breakout_criteria=breakout_criteria,
    )
    result = stat.compute(candles_df)
    merged_tf[tf] = result.instruments[instrument][tf]
    # Merge framework-enriched slice dimension labels so each timeframe contributes
    # without overwriting earlier entries.
    labels.dimensions.update(result.labels.dimensions)

  final_result = StatRunResult(
    stat_name="initial_balance_performance",
    title=_TITLE,
    definition=_DEFINITION,
    labels=labels,
    instruments={instrument: merged_tf},
  )

  return write_results(final_result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Compute Initial Balance Breakout — Performance stat"
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
    total = tf_data["total_samples"]
    dr = tf_data["data_range"]
    print(f"\n  {tf}: {total} resolved sessions | {dr}")
    rows_map = {r["outcome"]: r for r in tf_data["results"]}
    for out_key, _, agg in _OUTCOMES:
      row = rows_map.get(out_key, {})
      val = row.get("value") or 0.0
      bl = row.get("value_baseline")
      bl_str = f"{bl:.4f}" if bl is not None else "N/A"
      print(f"    {out_key}: {val:.4f} (baseline={bl_str})")
