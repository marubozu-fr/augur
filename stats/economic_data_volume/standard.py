"""Economic Data Volume stat.

Compares the average total RTH daily traded volume on economic-event days against
non-event days. Five event conditions are reported — CPI, FOMC, NFP and GDP
release days individually, plus an ``any_event`` union — alongside a
``non_event`` condition (sessions carrying none of the four releases).

This is a **magnitude** stat: the single ``mean_volume`` outcome carries its
metric in ``StatResultRow.value`` (and its random baseline in
``value_baseline``); the ordinary ``probability`` channel is left at ``0.0`` for
every row. Each condition reports its own sample size (``count`` / ``total``).

Methodology
-----------
A "day" is the RTH daily candle. Only **resolved** sessions are counted (clean
session open and a last RTH bar at or after ``session_end - close_tolerance_min``);
early-close days and the final incomplete day are excluded (pending discipline).
Each resolved day's volume is the sum of its RTH bar volumes.

Release dates are not derived from the OHLCV data; they are read from an external
economic calendar CSV (the headline US print for each event type) and injected
into the stat, so the computation stays a pure function of (candles, event
dates). A resolved session is flagged for an event type when its date matches one
of that type's release dates; ``any_event`` is the union of the four flags and
``non_event`` is their complement.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
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
from stats.cpi_performance.standard import load_cpi_release_dates
from stats.event_performance_base import _DEFAULT_CALENDAR_PATH
from stats.fomc_performance.standard import load_fomc_release_dates
from stats.nfp_performance.standard import load_nfp_release_dates
from stats.utils.daily_candles import build_resolved_days

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="Economic Data Volume",
  fr="Volume des données économiques",
)
_DEFINITION = I18nString(
  en=(
    "How does the average total RTH daily volume on economic-event days "
    "(CPI, FOMC, NFP, GDP) compare to non-event days?"
  ),
  fr=(
    "Comment le volume quotidien RTH total moyen des jours d'annonces "
    "économiques (CPI, FOMC, NFP, GDP) se compare-t-il aux jours sans annonce ?"
  ),
)
_LABELS = Labels(
  conditions={
    "cpi_day": I18nString(en="CPI release day", fr="Jour de publication du CPI"),
    "fomc_day": I18nString(en="FOMC decision day", fr="Jour de décision du FOMC"),
    "nfp_day": I18nString(en="NFP release day", fr="Jour de publication du NFP"),
    "gdp_day": I18nString(en="GDP release day", fr="Jour de publication du PIB"),
    "any_event_day": I18nString(
      en="Any event day", fr="Jour avec annonce"
    ),
    "non_event_day": I18nString(
      en="Non-event day", fr="Jour sans annonce"
    ),
  },
  outcomes={
    "mean_volume": I18nString(
      en="Average RTH volume", fr="Volume RTH moyen"
    ),
  },
)

# Single magnitude outcome carried in the `value` channel.
_OUTCOME = "mean_volume"

# Conditions in display order, each mapped to its boolean day-table flag column.
_CONDITIONS: list[tuple[str, str]] = [
  ("cpi_day", "is_cpi"),
  ("fomc_day", "is_fomc"),
  ("nfp_day", "is_nfp"),
  ("gdp_day", "is_gdp"),
  ("any_event_day", "is_any_event"),
  ("non_event_day", "is_non_event"),
]

# Event-type key -> flag column, for the four individual releases.
_EVENT_FLAGS: list[tuple[str, str]] = [
  ("cpi", "is_cpi"),
  ("fomc", "is_fomc"),
  ("nfp", "is_nfp"),
  ("gdp", "is_gdp"),
]

# Calendar rows identifying a GDP release. The forex-factory calendar publishes
# three real-GDP vintages per quarter (Advance / Prelim / Final, q/q); each is a
# distinct release day. The co-occurring "GDP Price Index" deflator lines are
# deliberately not matched.
_GDP_EVENTS = frozenset({"advance gdp q/q", "prelim gdp q/q", "final gdp q/q"})
_GDP_CURRENCY = "usd"


def load_gdp_release_dates(calendar_path: str | Path) -> set[pd.Timestamp]:
  """Read distinct GDP release dates from an economic calendar CSV.

  Keeps rows whose ``event`` is one of the three real-GDP q/q vintages
  (Advance / Prelim / Final) and returns the distinct release dates as
  normalized (midnight, tz-naive) Timestamps.
  """
  cal = pd.read_csv(calendar_path, usecols=["date", "currency", "event"])
  event = cal["event"].str.strip().str.lower()
  currency = cal["currency"].str.strip().str.lower()
  mask = event.isin(_GDP_EVENTS) & (currency == _GDP_CURRENCY)
  dates = pd.to_datetime(cal.loc[mask, "date"]).dt.normalize()
  return set(dates.unique())


class EconomicDataVolume(BaseStat):
  """Average RTH daily volume on each event-type day vs non-event days."""

  stat_name = "economic_data_volume"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = ()  # Events are scattered across the calendar; per-day slicing is meaningless.

  def __init__(
    self,
    instrument: str,
    config: InstrumentConfig,
    event_dates: dict[str, Iterable[pd.Timestamp]],
    close_tolerance_min: int = 15,
  ) -> None:
    self.instrument = instrument
    self.timeframe = "daily"
    self.config = config
    self.close_tolerance_min = close_tolerance_min

    # Normalize each event type's dates to a set of midnight, tz-naive Timestamps.
    self.event_dates: dict[str, set[pd.Timestamp]] = {
      etype: {pd.Timestamp(d).normalize().tz_localize(None) for d in dates}
      for etype, dates in event_dates.items()
    }

    rth = config.sessions["rth"]
    self.rth_start_min: int = minute_of_day(rth.start)
    self.rth_end_min: int = minute_of_day(rth.end)

  # -------------------------------------------------------------------------
  # Day table construction
  # -------------------------------------------------------------------------
  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-resolved-day table with RTH volume and event flags.

    Index = normalized session date (tz-naive). Columns: ``volume`` (sum of the
    day's RTH bar volumes) and the boolean flags ``is_cpi``, ``is_fomc``,
    ``is_nfp``, ``is_gdp``, ``is_any_event``, ``is_non_event``.

    Only resolved sessions appear (pending discipline via ``build_resolved_days``).
    Returns an empty DataFrame with the expected columns when there are no
    resolved days.
    """
    columns = [
      "volume",
      "is_cpi",
      "is_fomc",
      "is_nfp",
      "is_gdp",
      "is_any_event",
      "is_non_event",
    ]
    empty = pd.DataFrame(columns=columns)

    if candles_df.empty:
      return empty

    resolved = build_resolved_days(
      candles_df, self.rth_start_min, self.rth_end_min, self.close_tolerance_min
    )
    if resolved.empty:
      return empty

    resolved = resolved.sort_index()
    if resolved.index.tz is not None:
      resolved.index = resolved.index.tz_localize(None)

    # Per-day RTH volume from the same RTH bar filter the resolution uses.
    df = candles_df.copy()
    df["_mod"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    ts_norm = df["timestamp"].dt.normalize()
    df["_date"] = ts_norm.dt.tz_localize(None) if ts_norm.dt.tz is not None else ts_norm
    rth_mask = (df["_mod"] >= self.rth_start_min) & (df["_mod"] < self.rth_end_min)
    day_volume = df[rth_mask].groupby("_date")["volume"].sum().rename("volume")

    # Inner join keeps only resolved dates; result stays chronologically sorted.
    table = resolved.join(day_volume, how="inner")[["volume"]]
    if table.empty:
      return empty

    table["volume"] = table["volume"].astype(float)

    for etype, col in _EVENT_FLAGS:
      dates = self.event_dates.get(etype, set())
      event_index = pd.DatetimeIndex(sorted(dates))
      table[col] = table.index.isin(event_index)

    table["is_any_event"] = table[[col for _, col in _EVENT_FLAGS]].any(axis=1)
    table["is_non_event"] = ~table["is_any_event"]

    return table[columns]

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the average RTH volume for each condition.

    One ``mean_volume`` row per condition, carrying the average in ``value``
    (``probability`` left at ``0.0``) and the condition's day count in
    ``count`` / ``total``. All six rows are always emitted (even when a condition
    has no days) for a deterministic result shape. If ``baseline_rows`` is
    provided, the matching row's ``value`` / ``total`` are merged into
    ``value_baseline`` / ``baseline_n``.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    rows: list[StatResultRow] = []
    for cond_key, col in _CONDITIONS:
      if len(day_table) > 0 and col in day_table.columns:
        sub = day_table[day_table[col].astype(bool)]
      else:
        sub = day_table.iloc[0:0]
      n = len(sub)
      mean_value = float(sub["volume"].astype(float).mean()) if n > 0 else 0.0
      bl = baseline_map.get((cond_key, _OUTCOME))
      rows.append(
        StatResultRow(
          condition=cond_key,
          outcome=_OUTCOME,
          count=n,
          total=n,
          probability=0.0,
          baseline_prob=0.0,
          baseline_n=bl.total if bl else 0,
          value=mean_value,
          value_baseline=bl.value if bl else None,
          agg="mean",
        )
      )
    return rows

  # -------------------------------------------------------------------------
  # Baseline
  # -------------------------------------------------------------------------
  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random baseline: per condition, a random sample of N days from all days.

    For each condition holding ``N`` days, draw ``N`` days uniformly at random
    (without replacement) from the **whole** resolved-day table and average their
    volume. This is the null hypothesis that the event flag carries no
    information about volume: every condition's expected baseline equals the grand
    mean across all sessions, so an actual event-day mean well above it indicates
    a genuine volume lift around the release. Draws are sequential over the fixed
    condition order with ``np.random.default_rng(seed)`` for deterministic output.
    """
    pop_n = len(day_table)
    rng = np.random.default_rng(seed)

    rows: list[StatResultRow] = []
    if pop_n == 0:
      vols = np.empty(0, dtype=float)
    else:
      vols = day_table["volume"].to_numpy(dtype=float)

    for cond_key, col in _CONDITIONS:
      n_cond = int(day_table[col].astype(bool).sum()) if pop_n > 0 else 0
      sample_size = min(n_cond, pop_n)
      if sample_size > 0:
        idx = rng.choice(pop_n, size=sample_size, replace=False)
        mean_value = float(vols[idx].mean())
      else:
        mean_value = 0.0
      rows.append(
        StatResultRow(
          condition=cond_key,
          outcome=_OUTCOME,
          count=sample_size,
          total=sample_size,
          probability=0.0,
          baseline_prob=0.0,
          baseline_n=0,
          value=mean_value,
          value_baseline=None,
        )
      )
    return rows

  def classify_samples(self, day_table: pd.DataFrame) -> list[SampleRow]:
    """One SampleRow per (condition, day) where the day matches that condition.

    A day matching several conditions (e.g. a CPI day is also an any_event day)
    yields one sample per matching condition, mirroring how ``compute_rows``
    counts it under each.
    """
    if day_table.empty:
      return []

    samples: list[SampleRow] = []
    for cond_key, col in _CONDITIONS:
      if col not in day_table.columns:
        continue
      sub = day_table[day_table[col].astype(bool)]
      for ts, row in sub.iterrows():
        samples.append(
          SampleRow(
            date=ts.strftime("%Y-%m-%d"),
            condition=cond_key,
            outcome=_OUTCOME,
            value=float(row["volume"]),
          )
        )
    return samples


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
  calendar_path: str | Path = _DEFAULT_CALENDAR_PATH,
) -> Path:
  """Load data and the economic calendar, compute Economic Data Volume, write it."""
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)
  event_dates = {
    "cpi": load_cpi_release_dates(calendar_path),
    "fomc": load_fomc_release_dates(calendar_path),
    "nfp": load_nfp_release_dates(calendar_path),
    "gdp": load_gdp_release_dates(calendar_path),
  }

  stat = EconomicDataVolume(
    instrument=instrument,
    config=config,
    event_dates=event_dates,
  )
  result = stat.compute(candles_df)
  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Economic Data Volume stat")
  parser.add_argument("--instrument", default="NQ", help="Instrument name (default: NQ)")
  parser.add_argument("--data-path", default=None, help="Override parquet file path")
  parser.add_argument("--config-dir", default="config", help="Config directory (default: config)")
  parser.add_argument(
    "--calendar-path",
    default=str(_DEFAULT_CALENDAR_PATH),
    help=f"Economic calendar CSV path (default: {_DEFAULT_CALENDAR_PATH})",
  )
  args = parser.parse_args()

  output_path = run(
    instrument=args.instrument,
    config_dir=args.config_dir,
    data_path=args.data_path,
    calendar_path=args.calendar_path,
  )

  print(f"Written: {output_path}")

  with open(output_path, encoding="utf-8") as f:
    data = json.load(f)

  instrument_data = data["instruments"][args.instrument]
  for tf, tf_data in instrument_data.items():
    print(f"\n  {tf}: {tf_data['total_samples']} resolved days | {tf_data['data_range']}")
    by_cond = {r["condition"]: r for r in tf_data["results"]}
    for cond_key, _ in _CONDITIONS:
      row = by_cond.get(cond_key, {})
      print(
        f"    {cond_key}: avg volume={row.get('value', 0.0):,.0f}"
        f" (N={row.get('total', 0)}, baseline={row.get('value_baseline') or 0.0:,.0f})"
      )
