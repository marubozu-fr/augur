"""Fair Value Gaps stat — standard variant.

Detects intraday 3-candle **Fair Value Gap** (FVG) patterns on 15-minute RTH
candles and measures how often each gap is **filled** (mitigated) within the same
RTH session.

Definitions:
  - **15-min candle**: built by aggregating 1-minute OHLCV bars within the RTH
    window ``[rth_start_min, rth_end_min)`` into 15-minute buckets aligned to
    ``rth_start_min``.  Bucket index ``k = (minute_of_day - rth_start_min) // 15``.

  - **FVG detection** — slide a consecutive 3-candle window (c1, c2, c3) across
    each session's 15-min candle sequence, **overlapping** windows (every valid
    triple is counted):
      Bullish FVG: ``c1.high < c3.low``.
        Gap zone ``(c1.high, c3.low)``, gap points ``gap_pts = c3.low - c1.high``.
      Bearish FVG: ``c1.low > c3.high``.
        Gap zone ``(c3.high, c1.low)``, gap points ``gap_pts = c1.low - c3.high``.

  - **Gap size**: ``gap_size_pct = 100 * gap_pts / c3.close`` — the gap expressed
    as a percent of price at formation.

  - **Fill** (mitigation): assessed over the 15-min candles AFTER c3 within the
    same session (up to session end). The fill target is governed by
    ``fill_threshold_pct`` (default 100 = full fill / full mitigation):
      Bullish target ``= c3.low - (pct/100) * gap_pts``.
        At 100 % the target collapses to ``c1.high`` — price must trade back down
        into the bottom of the gap.  Filled when ``min(subsequent lows) <= target``.
      Bearish target ``= c3.high + (pct/100) * gap_pts``.
        At 100 % the target collapses to ``c1.low``.  Filled when
        ``max(subsequent highs) >= target``.

  - **Pending discipline**: an FVG where c3 is the last 15-min candle of its
    session has NO subsequent candle with which to resolve the fill → excluded from
    BOTH numerator and denominator.  Every other FVG resolves by session end as
    ``filled`` or ``not_filled``.

  - **Resolved sessions only**: only RTH sessions that pass ``build_resolved_days``
    (clean 09:30 open + last bar at or after ``session_end - close_tolerance``)
    contribute FVG events.

Reported as a 2x2 conditional probability matrix:
  P(filled | bullish FVG), P(not filled | bullish FVG),
  P(filled | bearish FVG), P(not filled | bearish FVG).
The two outcomes partition each direction, so they sum to 1.

**Event-table design**: unlike daily stats whose day table has one row per session,
this stat's ``build_day_table`` returns an *event table* — one row per FVG event,
indexed by the (possibly duplicate) session date.  ``total_samples`` therefore
reports the total count of resolved FVG events, not sessions.  The ``weekday``
slicer reads ``index.dayofweek`` (same date → same weekday for all events in a
session), and the ``SizeBucket`` slicer reads the ``gap_size_pct`` column.
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
  write_results,
)
from stats.config import InstrumentConfig, load_config, minute_of_day
from stats.utils.daily_candles import build_resolved_days

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_BUCKET_MIN: int = 15  # 15-minute candles

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="Fair Value Gaps",
  fr="Imbalances / Gaps de juste valeur",
)
_DEFINITION = I18nString(
  en="On 15-minute RTH candles, a Fair Value Gap (FVG) forms when candle 1 and candle 3 leave a price imbalance (bullish: c1.high < c3.low; bearish: c1.low > c3.high). How often is that gap filled within the same session?",
  fr="Sur les bougies RTH de 15 minutes, un gap de juste valeur (FVG) se forme lorsque la bougie 1 et la bougie 3 laissent un déséquilibre de prix (haussier : haut c1 < bas c3 ; baissier : bas c1 > haut c3). À quelle fréquence ce gap est-il comblé au cours de la même session ?",
)
_LABELS = Labels(
  conditions={
    "bullish": I18nString(en="Bullish FVG", fr="FVG haussier"),
    "bearish": I18nString(en="Bearish FVG", fr="FVG baissier"),
  },
  outcomes={
    "filled": I18nString(en="Filled", fr="Comblé"),
    "not_filled": I18nString(en="Not filled", fr="Non comblé"),
  },
)

# Condition/outcome enumeration. The condition IS the FVG direction.
_CONDITION_DIRECTIONS: tuple[str, ...] = ("bullish", "bearish")
_OUTCOMES: tuple[tuple[str, bool], ...] = (("filled", True), ("not_filled", False))


class FairValueGaps(BaseStat):
  """Conditional probability that an intraday FVG is filled within the same session."""

  stat_name = "fair_value_gaps"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = (
    "weekday",
    SizeBucket(
      column="gap_size_pct",
      buckets=[0.0, 0.024, 0.049, 0.089, 0.149, 0.25, float("inf")],
      name="size",
    ),
  )

  def __init__(
    self,
    instrument: str,
    config: InstrumentConfig,
    fill_threshold_pct: float = 100.0,
    close_tolerance_min: int = 15,
  ) -> None:
    self.instrument = instrument
    self.timeframe = "15min"
    self.config = config
    self.fill_threshold_pct = fill_threshold_pct
    self.close_tolerance_min = close_tolerance_min

    rth = config.sessions["rth"]
    self.rth_start_min: int = minute_of_day(rth.start)
    self.rth_end_min: int = minute_of_day(rth.end)

  # -------------------------------------------------------------------------
  # Event table construction
  # -------------------------------------------------------------------------
  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build the FVG event table from 1-min OHLCV data.

    Returns an **event table**: one row per resolved FVG event, indexed by the
    (possibly duplicate) normalized session date (``DatetimeIndex``).  Events
    whose c3 is the last 15-min candle of its session are excluded (pending-
    sample discipline: no subsequent candle → outcome unresolvable).

    Columns:
      ``direction``    — ``"bullish"`` or ``"bearish"``.
      ``gap_size_pct`` — ``100 * gap_pts / c3.close``, percent of price at
                         formation.  Read by the ``size`` ``SizeBucket`` slicer.
      ``filled``       — ``bool``; whether the gap filled to
                         ``fill_threshold_pct`` within the same session.

    Steps:
    1. Build the resolved-sessions index via ``build_resolved_days``; only those
       dates contribute events (pending/early-close sessions are excluded).
    2. Filter 1-min bars to RTH ``[rth_start_min, rth_end_min)`` for resolved
       dates only, then aggregate to 15-min candles by bucket index.
    3. Within each session, slide a 3-candle window: use within-group
       ``shift(2)`` / ``shift(1)`` to attach c1 / c2 to each c3 row.
    4. Detect FVG conditions (bullish: ``c1_high < c3_low``; bearish:
       ``c1_low  > c3_high``).
    5. Compute suffix-min-low and suffix-max-high for candles *after* c3 in
       the same session using a reversed cumulative min/max on the shifted
       series.  A NaN suffix indicates c3 is the last candle → pending, excluded.
    6. Evaluate fill against the ``fill_threshold_pct`` target.
    7. Build the result DataFrame indexed by session date.

    Returns an empty DataFrame with the correct columns when no data is
    available.
    """
    columns = ["direction", "gap_size_pct", "filled"]
    empty = pd.DataFrame(columns=columns)

    if candles_df.empty:
      return empty

    # Step 1: resolved-session dates.
    resolved = build_resolved_days(
      candles_df, self.rth_start_min, self.rth_end_min, self.close_tolerance_min
    )
    if resolved.empty:
      return empty

    resolved_dates = set(resolved.index)

    # Step 2: filter to RTH bars of resolved sessions, assign 15-min bucket.
    df = candles_df.copy()
    df["_mod"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["_date"] = df["timestamp"].dt.normalize()

    rth_mask = (df["_mod"] >= self.rth_start_min) & (df["_mod"] < self.rth_end_min)
    date_mask = df["_date"].isin(resolved_dates)
    rth = df[rth_mask & date_mask].copy()
    if rth.empty:
      return empty

    # Sort chronologically so first/last aggregations below are correct.
    rth = rth.sort_values(["_date", "_mod"])
    rth["_bucket"] = (rth["_mod"] - self.rth_start_min) // _BUCKET_MIN

    # Aggregate 1-min bars → 15-min OHLC candles.
    agg = rth.groupby(["_date", "_bucket"]).agg(
      open=("open", "first"),
      high=("high", "max"),
      low=("low", "min"),
      close=("close", "last"),
    )

    if len(agg) < 3:
      return empty

    # Reset to a flat DataFrame sorted by (_date, _bucket) for shift operations.
    c = agg.reset_index().sort_values(["_date", "_bucket"]).reset_index(drop=True)

    # Step 3: within-session shifts to attach c1 info to each c3 row.
    # shift(2) within session: c["c1_high"][i] = high of candle i-2 in same session.
    grp = c.groupby("_date", sort=False)
    c["c1_high"] = grp["high"].shift(2)
    c["c1_low"] = grp["low"].shift(2)

    # Step 5: suffix min low / max high for candles AFTER the current row in
    # the same session.  Using shift(-1) to start from i+1, then reversed
    # cumulative min/max, then reverse back.
    #
    # For a within-session series [v0, v1, ..., vn]:
    #   shift(-1)        = [v1, v2, ..., vn, NaN]
    #   reversed         = [NaN, vn, ..., v2, v1]
    #   cummin/cummax    = running extreme from NaN onward (NaN stays NaN)
    #   reversed back    = [min(v1..vn), min(v2..vn), ..., vn, NaN]
    # The last row (last candle) gets NaN → pending; all others get the true
    # suffix extreme.
    def _suffix_min(s: pd.Series) -> pd.Series:
      shifted = s.shift(-1)
      return shifted.iloc[::-1].cummin().iloc[::-1]

    def _suffix_max(s: pd.Series) -> pd.Series:
      shifted = s.shift(-1)
      return shifted.iloc[::-1].cummax().iloc[::-1]

    c["sfx_min_low"] = grp["low"].transform(_suffix_min)
    c["sfx_max_high"] = grp["high"].transform(_suffix_max)

    # Step 4: detect FVG conditions.  Both conditions are mutually exclusive:
    # bullish requires c3 entirely above c1 (gap_pts > 0 upward),
    # bearish requires c3 entirely below c1 (gap_pts > 0 downward).
    has_c1 = c["c1_high"].notna()  # True from the 3rd candle in each session onward.
    bullish_mask = has_c1 & (c["c1_high"] < c["low"])
    bearish_mask = has_c1 & (c["c1_low"] > c["high"])
    fvg_mask = bullish_mask | bearish_mask

    # Pending discipline: exclude events where c3 is the last session candle.
    has_subsequent = c["sfx_min_low"].notna()
    fvg_mask = fvg_mask & has_subsequent

    if not fvg_mask.any():
      return empty

    events = c[fvg_mask].copy()
    is_bullish = bullish_mask[events.index]

    # Step 6: gap size and fill detection.
    gap_pts = np.where(
      is_bullish,
      events["low"].to_numpy() - events["c1_high"].to_numpy(),   # c3.low - c1.high
      events["c1_low"].to_numpy() - events["high"].to_numpy(),   # c1.low - c3.high
    )
    gap_size_pct = 100.0 * gap_pts / events["close"].to_numpy()

    frac = self.fill_threshold_pct / 100.0
    # Bullish target = c3.low - frac*gap_pts.  At 100 % → c1.high.
    # Filled when suffix min low <= target (price traded back into the gap).
    bull_target = events["low"].to_numpy() - frac * gap_pts
    bull_filled = events["sfx_min_low"].to_numpy() <= bull_target
    # Bearish target = c3.high + frac*gap_pts.  At 100 % → c1.low.
    # Filled when suffix max high >= target.
    bear_target = events["high"].to_numpy() + frac * gap_pts
    bear_filled = events["sfx_max_high"].to_numpy() >= bear_target

    filled = np.where(is_bullish.to_numpy(), bull_filled, bear_filled).astype(bool)
    direction = np.where(is_bullish.to_numpy(), "bullish", "bearish")

    # Step 7: build event table indexed by session date.
    result = pd.DataFrame(
      {
        "direction": direction,
        "gap_size_pct": gap_size_pct,
        "filled": filled,
      },
      index=pd.DatetimeIndex(events["_date"].to_numpy()),
    )
    return result[columns]

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the four matrix rows (bullish/bearish × filled/not filled).

    All rows in the event table are countable (pending events are already
    excluded by ``build_day_table``).  If ``baseline_rows`` is provided,
    merges its ``probability`` / ``total`` into each row's ``baseline_prob``
    / ``baseline_n``.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    if day_table.empty:
      return [
        StatResultRow(
          condition=cond_key,
          outcome=out_key,
          count=0,
          total=0,
          probability=0.0,
          baseline_prob=0.0,
          baseline_n=0,
        )
        for cond_key in _CONDITION_DIRECTIONS
        for out_key, _ in _OUTCOMES
      ]

    direction = day_table["direction"].astype(str)
    filled = day_table["filled"].astype(bool)

    rows: list[StatResultRow] = []
    for cond_dir in _CONDITION_DIRECTIONS:
      cond_mask = direction == cond_dir
      total = int(cond_mask.sum())
      for out_key, out_is_filled in _OUTCOMES:
        out_match = filled if out_is_filled else ~filled
        count = int((cond_mask & out_match).sum())
        probability = count / total if total > 0 else 0.0
        bl = baseline_map.get((cond_dir, out_key))
        rows.append(
          StatResultRow(
            condition=cond_dir,
            outcome=out_key,
            count=count,
            total=total,
            probability=probability,
            baseline_prob=bl.probability if bl else 0.0,
            baseline_n=bl.total if bl else 0,
          )
        )
    return rows

  def classify_samples(self, day_table: pd.DataFrame) -> list[SampleRow]:
    """One SampleRow per FVG event, mirroring ``compute_rows``.

    Every row in the event table is countable (pending events are already
    excluded by ``build_day_table``); a session with multiple FVG events
    contributes multiple SampleRows sharing the same date.
    """
    if day_table.empty:
      return []

    samples: list[SampleRow] = []
    for ts, direction, filled in zip(
      day_table.index, day_table["direction"], day_table["filled"]
    ):
      outcome = "filled" if bool(filled) else "not_filled"
      samples.append(SampleRow(date=ts.strftime("%Y-%m-%d"), condition=str(direction), outcome=outcome))
    return samples

  # -------------------------------------------------------------------------
  # Baseline
  # -------------------------------------------------------------------------
  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random null baseline. Deterministic for a fixed seed.

    The ``filled`` outcome is **permuted** across all FVG events, so the
    pooled fill rate is preserved but its association with the gap direction
    (bullish vs bearish) is destroyed.  Each condition's ``baseline_prob``
    therefore converges to the pooled fill rate — the null hypothesis that
    the gap's direction carries no information about whether it will fill.
    A coin-flip baseline (50/50) would be wrong here because FVGs typically
    fill well above 50 % of the time; the permutation baseline is the correct
    direction-agnostic null.  Uses ``np.random.default_rng(seed)``.
    """
    n = len(day_table)
    if n == 0:
      return self.compute_rows(day_table, baseline_rows=None)

    rng = np.random.default_rng(seed)
    tmp = day_table.copy()
    tmp["filled"] = tmp["filled"].to_numpy()[rng.permutation(n)]

    return self.compute_rows(tmp, baseline_rows=None)


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
  fill_threshold_pct: float = 100.0,
) -> Path:
  """Load data and compute Fair Value Gaps for the 15-min timeframe.

  Writes the result JSON to results/ and returns its path.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  stat = FairValueGaps(
    instrument=instrument,
    config=config,
    fill_threshold_pct=fill_threshold_pct,
  )
  result = stat.compute(candles_df)

  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Fair Value Gaps stat")
  parser.add_argument("--instrument", default="NQ", help="Instrument name (default: NQ)")
  parser.add_argument("--data-path", default=None, help="Override parquet file path")
  parser.add_argument("--config-dir", default="config", help="Config directory (default: config)")
  parser.add_argument(
    "--fill-threshold-pct",
    type=float,
    default=100.0,
    help="Fill threshold as a percent of the gap (default: 100 = full mitigation)",
  )
  args = parser.parse_args()

  output_path = run(
    instrument=args.instrument,
    config_dir=args.config_dir,
    data_path=args.data_path,
    fill_threshold_pct=args.fill_threshold_pct,
  )

  print(f"Written: {output_path}")

  with open(output_path, encoding="utf-8") as f:
    data = json.load(f)

  instrument_data = data["instruments"][args.instrument]
  for tf, tf_data in instrument_data.items():
    print(
      f"\n  {tf}: {tf_data['total_samples']} resolved FVG events | {tf_data['data_range']}"
    )
    for row in tf_data["results"]:
      print(
        f"    {row['condition']} -> {row['outcome']}: "
        f"{row['probability']:.3f} (N={row['total']}, baseline={row['baseline_prob']:.3f})"
      )
