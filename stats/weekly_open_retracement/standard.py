"""Weekly Open Retracement stat — standard variant.

Measures how often weeks that moved above or below the weekly opening price
retrace back to touch it.

Definitions:
  - **Weekly opening price L**: the ``open`` of the FIRST RTH bar of the ISO
    week (the 09:30 RTH session-open bar of the week's first trading day). RTH
    is the config session "rth"; bars are filtered by minute-of-day
    ``[rth_start_min, rth_end_min)``.
  - **Week key**: the Monday of each bar's ISO week
    (``date - to_timedelta(date.dayofweek, 'D')``), normalized tz-aware.
  - **Pending discipline**: the last week present in the data is always dropped
    (it may still be in progress).
  - **Direction** (from the FIRST RTH bar of the week):
      opened_above: ``close > L`` (direction_up = True)
      opened_below: ``close < L`` (direction_up = False)
      doji:         ``close == L`` → excluded from every denominator
                    (direction_up = pd.NA, stored as nullable boolean column).
  - **Countable**: direction_up is not NA.
  - **Retracement** (scanned from RTH bars AFTER the first bar of the week):
      opened_above → retraced when any later bar has ``low <= L``.
      opened_below → retraced when any later bar has ``high >= L``.
      Touching exactly counts.
  - **retrace_weekday**: python weekday int (0=Mon .. 4=Fri) of the FIRST
    retracing bar; NA when not retraced.
  - **spike_pts**: maximum favorable excursion from L in points, measured over
    the RTH bars from the week's first bar through the FIRST retracing bar
    inclusive (or through the last bar of the week if never retraced):
      opened_above → max(high over window) - L
      opened_below → L - min(low over window)
    Defined for all countable weeks; NA for doji (non-countable) weeks.
  - **spike_pct**: ``100 * spike_pts / L`` (label-only percent size metric).

Two tiers of results are reported:
  1. 2x2 conditional matrix — condition opened_above / opened_below;
     outcomes retraced / not_retraced. The two outcomes partition each
     direction so their counts sum to the condition total.
  2. Retracement-weekday distribution — condition retraced; outcomes
     monday..friday (weekday of the first retracing bar among all retraced
     countable weeks, regardless of direction).

Slices: spike_pts quartiles; spike_pct quartiles.
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

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="Weekly Open Retracement",
  fr="Retracement de l'ouverture hebdomadaire",
)
_DEFINITION = I18nString(
  en=(
    "When the first RTH bar of the week closes above or below the weekly opening "
    "price, how often does intraday price retrace back to touch that level before "
    "the week ends?"
  ),
  fr=(
    "Lorsque la première bougie RTH de la semaine clôture au-dessus ou en dessous "
    "du prix d'ouverture hebdomadaire, à quelle fréquence le prix intraday revient-il "
    "toucher ce niveau avant la fin de la semaine ?"
  ),
)
_LABELS = Labels(
  conditions={
    "opened_above": I18nString(en="Opened above", fr="Ouverture au-dessus"),
    "opened_below": I18nString(en="Opened below", fr="Ouverture en dessous"),
    "retraced": I18nString(en="Retraced", fr="Retracé"),
  },
  outcomes={
    "retraced": I18nString(en="Retraced", fr="Retracé"),
    "not_retraced": I18nString(en="Not retraced", fr="Non retracé"),
    "monday": I18nString(en="Monday", fr="Lundi"),
    "tuesday": I18nString(en="Tuesday", fr="Mardi"),
    "wednesday": I18nString(en="Wednesday", fr="Mercredi"),
    "thursday": I18nString(en="Thursday", fr="Jeudi"),
    "friday": I18nString(en="Friday", fr="Vendredi"),
  },
)

# Enumeration constants used in compute_rows / baseline_rows.
_CONDITIONS: tuple[tuple[str, bool], ...] = (
  ("opened_above", True),
  ("opened_below", False),
)
_OUTCOMES: tuple[tuple[str, bool], ...] = (
  ("retraced", True),
  ("not_retraced", False),
)
_WEEKDAY_OUTCOMES: tuple[tuple[str, int], ...] = (
  ("monday", 0),
  ("tuesday", 1),
  ("wednesday", 2),
  ("thursday", 3),
  ("friday", 4),
)

# Columns produced by build_week_table.
_COLUMNS = [
  "weekly_open",
  "direction_up",
  "retraced",
  "retrace_weekday",
  "spike_pts",
  "spike_pct",
]


class WeeklyOpenRetracement(BaseStat):
  """Conditional probability that weekly price retraces to the weekly opening
  level, conditioned on the direction of the first RTH bar of the week."""

  stat_name = "weekly_open_retracement"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = (
    SizeBucket(column="spike_pts", preset="quartiles", name="spike_pts"),
    SizeBucket(column="spike_pct", preset="quartiles", name="spike_pct"),
  )

  def __init__(
    self,
    instrument: str,
    config: InstrumentConfig,
  ) -> None:
    self.instrument = instrument
    self.timeframe = "weekly"
    self.config = config

    rth = config.sessions["rth"]
    self.rth_start_min: int = minute_of_day(rth.start)
    self.rth_end_min: int = minute_of_day(rth.end)

  # -------------------------------------------------------------------------
  # Week table construction
  # -------------------------------------------------------------------------
  def build_week_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-week summary table from raw 1-min OHLCV data.

    Indexed by the week's Monday date (normalized, tz-aware). Columns:
      ``weekly_open``     — open of the first RTH bar of the week.
      ``direction_up``    — nullable bool; True = opened_above,
                            False = opened_below, NA = doji first bar
                            (excluded from denominators).
      ``retraced``        — bool; True when price touched the weekly open
                            again after the first RTH bar.
      ``retrace_weekday`` — nullable Int64; python weekday (0=Mon..4=Fri) of
                            the chronologically first retracing bar; NA when
                            the week did not retrace.
      ``spike_pts``       — float; maximum favorable excursion from the weekly
                            open in points over the window [first bar .. first
                            retracing bar] (or [first bar .. last bar] when no
                            retracement); NA for doji weeks.
      ``spike_pct``       — float; ``100 * spike_pts / weekly_open``; NA for
                            doji weeks.

    The last ISO week in the data is always dropped (pending discipline — it
    may still be in progress). Returns an empty DataFrame with the correct
    column schema when the input has no RTH bars or at most one ISO week.
    """
    empty = pd.DataFrame(columns=_COLUMNS)

    if candles_df.empty:
      return empty

    df = candles_df.copy()
    df["_mod"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    rth_mask = (df["_mod"] >= self.rth_start_min) & (df["_mod"] < self.rth_end_min)
    rth = df[rth_mask].copy()
    if rth.empty:
      return empty

    # Week key = Monday of the bar's ISO week (normalized, tz-aware).
    date = rth["timestamp"].dt.normalize()
    rth["_week"] = date - pd.to_timedelta(date.dt.dayofweek, unit="D")
    rth["_date"] = date

    all_weeks = sorted(rth["_week"].unique())

    # Need at least 2 weeks to have 1 resolved (non-pending) week.
    if len(all_weeks) <= 1:
      return empty
    resolved_weeks = all_weeks[:-1]

    week_groups = {wk: grp for wk, grp in rth.groupby("_week")}

    rows: list[dict] = []
    for wk in resolved_weeks:
      bars = week_groups[wk].sort_values("timestamp")

      # First RTH bar of the week defines the weekly opening price L.
      first_bar = bars.iloc[0]
      weekly_open: float = float(first_bar["open"])
      first_close: float = float(first_bar["close"])

      # Direction from the first bar's close vs its open (= L).
      if first_close > weekly_open:
        direction_up: bool | type[pd.NA] = True
      elif first_close < weekly_open:
        direction_up = False
      else:
        # Doji first bar — no direction, not countable.
        rows.append({
          "week": wk,
          "weekly_open": weekly_open,
          "direction_up": pd.NA,
          "retraced": False,
          "retrace_weekday": pd.NA,
          "spike_pts": pd.NA,
          "spike_pct": pd.NA,
        })
        continue

      # All RTH bars after the first bar of the week.
      post_bars = bars.iloc[1:]

      # Retracement: any later bar that touches or crosses the weekly open.
      if direction_up:
        retrace_mask = post_bars["low"] <= weekly_open
      else:
        retrace_mask = post_bars["high"] >= weekly_open

      retraced: bool = bool(retrace_mask.any())

      if retraced:
        first_retrace_bar = post_bars[retrace_mask].iloc[0]
        retrace_weekday: int | type[pd.NA] = int(
          first_retrace_bar["timestamp"].dayofweek
        )
        # Spike window: first bar through the first retracing bar inclusive.
        window = bars[bars["timestamp"] <= first_retrace_bar["timestamp"]]
      else:
        retrace_weekday = pd.NA
        # Spike window: all bars of the week.
        window = bars

      # Maximum favorable excursion from the weekly open over the window.
      if direction_up:
        spike_pts: float = float(window["high"].max()) - weekly_open
      else:
        spike_pts = weekly_open - float(window["low"].min())

      spike_pct: float | type[pd.NA] = (
        100.0 * spike_pts / weekly_open if weekly_open != 0 else pd.NA
      )

      rows.append({
        "week": wk,
        "weekly_open": weekly_open,
        "direction_up": direction_up,
        "retraced": retraced,
        "retrace_weekday": retrace_weekday,
        "spike_pts": spike_pts,
        "spike_pct": spike_pct,
      })

    if not rows:
      return empty

    week_table = pd.DataFrame(rows).set_index("week").sort_index()
    # Ensure typed nullable columns.
    week_table["direction_up"] = week_table["direction_up"].astype("boolean")
    week_table["retrace_weekday"] = week_table["retrace_weekday"].astype("Int64")
    return week_table[_COLUMNS]

  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Delegate to build_week_table (required by BaseStat)."""
    return self.build_week_table(candles_df)

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the 2x2 matrix and weekday-distribution rows.

    Tier 1 (4 rows): condition opened_above / opened_below x outcome
    retraced / not_retraced. Only countable weeks (direction_up not NA)
    enter the denominators.

    Tier 2 (5 rows): condition retraced x outcome monday..friday. The
    denominator is the total number of retraced countable weeks; the count
    for each weekday is how many first retracements fell on that day.

    When ``baseline_rows`` is provided, merges baseline_prob / baseline_n
    from the matching (condition, outcome) key into each returned row.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    def _make(
      condition: str, outcome: str, count: int, total: int
    ) -> StatResultRow:
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

    def _zero() -> list[StatResultRow]:
      result: list[StatResultRow] = []
      for cond_key, _ in _CONDITIONS:
        for out_key, _ in _OUTCOMES:
          result.append(_make(cond_key, out_key, 0, 0))
      for out_key, _ in _WEEKDAY_OUTCOMES:
        result.append(_make("retraced", out_key, 0, 0))
      return result

    if day_table.empty:
      return _zero()

    direction_up = day_table["direction_up"]
    countable = direction_up.notna()
    # Convert to regular bool for mask arithmetic (NA → False, excluded by countable).
    dir_bool = direction_up.fillna(False).astype(bool)
    retraced = day_table["retraced"].astype(bool)
    retrace_wd = pd.to_numeric(day_table["retrace_weekday"], errors="coerce")

    rows: list[StatResultRow] = []

    # Tier 1: 2x2 conditional matrix.
    for cond_key, cond_is_up in _CONDITIONS:
      cond_mask = countable & (dir_bool == cond_is_up)
      total = int(cond_mask.sum())
      for out_key, out_is_retraced in _OUTCOMES:
        out_match = retraced if out_is_retraced else ~retraced
        count = int((cond_mask & out_match).sum())
        rows.append(_make(cond_key, out_key, count, total))

    # Tier 2: weekday distribution among retraced countable weeks.
    retraced_countable = countable & retraced
    retraced_total = int(retraced_countable.sum())
    for out_key, day_num in _WEEKDAY_OUTCOMES:
      count = int((retraced_countable & (retrace_wd == day_num)).sum())
      rows.append(_make("retraced", out_key, count, retraced_total))

    return rows

  # -------------------------------------------------------------------------
  # Per-week sample classification
  # -------------------------------------------------------------------------
  def classify_samples(self, day_table: pd.DataFrame) -> list[SampleRow]:
    """One or two SampleRows per countable week, mirroring ``compute_rows``.

    Every countable week (``direction_up`` not NA) emits a Tier-1 SampleRow
    whose condition is ``opened_above`` or ``opened_below`` and whose outcome
    is ``retraced`` or ``not_retraced``. Retraced countable weeks additionally
    emit a Tier-2 SampleRow with condition ``retraced`` and the weekday
    (monday..friday) of the first retracing bar. Doji (non-countable) weeks
    contribute nothing, matching the ``countable`` mask in ``compute_rows``.
    This stat is purely probability-based, so ``value`` is left None on every
    row (spike_pts/spike_pct are slice-only columns, never compute_rows
    outcomes).
    """
    if day_table.empty:
      return []

    countable = day_table["direction_up"].notna()
    if not bool(countable.any()):
      return []

    weekday_names = {day_num: key for key, day_num in _WEEKDAY_OUTCOMES}

    samples: list[SampleRow] = []
    for ts, row in day_table[countable].iterrows():
      date_str = ts.strftime("%Y-%m-%d")
      cond_key = "opened_above" if bool(row["direction_up"]) else "opened_below"
      retraced = bool(row["retraced"])
      outcome = "retraced" if retraced else "not_retraced"
      samples.append(SampleRow(date=date_str, condition=cond_key, outcome=outcome))

      if retraced:
        day_num = int(row["retrace_weekday"])
        samples.append(
          SampleRow(date=date_str, condition="retraced", outcome=weekday_names[day_num])
        )

    return samples

  # -------------------------------------------------------------------------
  # Baseline
  # -------------------------------------------------------------------------
  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random null baseline. Deterministic for a fixed seed.

    Two surrogate tables are computed and stitched:

      Tier 1 (condition opened_above / opened_below): the ``retraced`` column
      is PERMUTED across countable weeks. This preserves the pooled retracement
      rate while destroying its association with direction. Each condition's
      baseline_prob therefore converges to the pooled rate.

      Tier 2 (condition retraced): each retraced countable week's
      ``retrace_weekday`` is replaced by a uniform random draw over 0..4.
      The null is that no weekday is special for the first retracement
      (baseline ~ 0.2 each). The ``retraced`` column is kept as-is from the
      original table so the retraced-total denominator stays at full size.

    The two surrogate tables are stitched: rows whose condition is
    opened_above / opened_below come from tier-1, rows whose condition is
    retraced come from tier-2. Uses ``np.random.default_rng(seed)``.
    """
    if len(day_table) == 0:
      return self.compute_rows(day_table, baseline_rows=None)

    rng = np.random.default_rng(seed)
    direction_up = day_table["direction_up"]
    countable = direction_up.notna().to_numpy()

    # Tier 1: permute retraced across countable weeks.
    tier1_table = day_table.copy()
    if countable.any():
      idx = day_table.index[countable]
      vals = day_table.loc[idx, "retraced"].to_numpy()
      tier1_table.loc[idx, "retraced"] = vals[rng.permutation(len(vals))]
    tier1_rows = self.compute_rows(tier1_table, baseline_rows=None)

    # Tier 2: randomize retrace_weekday for retraced countable weeks.
    retraced = day_table["retraced"].astype(bool).to_numpy()
    retraced_countable = countable & retraced
    tier2_table = day_table.copy()
    if retraced_countable.any():
      idx2 = day_table.index[retraced_countable]
      n_retraced = int(retraced_countable.sum())
      tier2_table.loc[idx2, "retrace_weekday"] = rng.integers(0, 5, size=n_retraced)
    tier2_rows = self.compute_rows(tier2_table, baseline_rows=None)

    # Stitch: tier-1 rows for opened_above / opened_below; tier-2 rows for
    # the retraced condition. Final order matches compute_rows (tier1 first).
    by_key: dict[tuple[str, str], StatResultRow] = {
      (r.condition, r.outcome): r for r in tier1_rows
    }
    by_key.update({
      (r.condition, r.outcome): r
      for r in tier2_rows
      if r.condition == "retraced"
    })
    return [by_key[(r.condition, r.outcome)] for r in tier1_rows]


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
) -> Path:
  """Load data and compute Weekly Open Retracement for the weekly timeframe.

  Writes the result JSON to results/ and returns its path.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  stat = WeeklyOpenRetracement(instrument=instrument, config=config)
  result = stat.compute(candles_df)

  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Weekly Open Retracement stat")
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
    print(f"\n  {tf}: {tf_data['total_samples']} resolved weeks | {tf_data['data_range']}")
    for row in tf_data["results"]:
      print(
        f"    {row['condition']} -> {row['outcome']}: "
        f"{row['probability']:.3f} (N={row['total']}, baseline={row['baseline_prob']:.3f})"
      )
