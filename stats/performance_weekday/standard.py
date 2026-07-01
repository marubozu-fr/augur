"""Performance by Weekday stat.

Measures, for each weekday: the average percent return, the green/red day counts,
and the average green-day and red-day move sizes.

The overall result aggregates all resolved days; the per-weekday breakdown is
produced by the framework's declared ``weekday`` slice.

Unlike the probability-only stats, this family reports **continuous magnitudes**
(average returns) alongside the green/red counts. Magnitude rows carry their
metric in ``StatResultRow.value`` (a signed decimal return, e.g. ``0.012`` = +1.2%)
and its random baseline in ``value_baseline``; the green/red count rows use the
ordinary ``probability`` channel.

A "day" is the RTH daily candle (session open to session close). The percent
return basis is controlled by ``performance``:
  - ``close_to_close`` (default): (session_close - PREVIOUS resolved day's
    session_close) / previous close. The first resolved day has no prior close
    and is excluded (pending-sample discipline).
  - ``open_to_close``: (session_close - session_open) / session_open.

A day is **green** when its return is >= 0, otherwise **red**.
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
from stats.utils.daily_candles import build_resolved_days

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="Performance by Weekday",
  fr="Performance par jour de la semaine",
)
_DEFINITION = I18nString(
  en="What is the average percent return, the green/red day counts, and the average green/red move size for each weekday?",
  fr="Quels sont le rendement moyen en pourcentage, le nombre de jours verts/rouges et la taille moyenne des mouvements verts/rouges pour chaque jour de la semaine ?",
)
_LABELS = Labels(
  conditions={
    "any_day": I18nString(en="All trading days", fr="Tous les jours de bourse"),
  },
  outcomes={
    "mean_return": I18nString(en="Average return", fr="Rendement moyen"),
    "green_day": I18nString(en="Green day (up)", fr="Jour vert (hausse)"),
    "red_day": I18nString(en="Red day (down)", fr="Jour rouge (baisse)"),
    "mean_green_move": I18nString(en="Average green move", fr="Mouvement vert moyen"),
    "mean_red_move": I18nString(en="Average red move", fr="Mouvement rouge moyen"),
  },
)

# Performance modes: how a day's percent return is computed.
_PERFORMANCE_MODES = ("close_to_close", "open_to_close")

# Single condition: every resolved day belongs to it; the weekday slice does the
# per-day breakdown.
_CONDITION = "any_day"

# Outcomes whose payload is a continuous metric in `value` (not a probability).
_MAGNITUDE_OUTCOMES = frozenset({"mean_return", "mean_green_move", "mean_red_move"})


class PerformanceByWeekday(BaseStat):
  """Average return, green/red counts, and average move sizes, sliced by weekday."""

  stat_name = "performance_weekday"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = ("weekday",)

  def __init__(
    self,
    instrument: str,
    config: InstrumentConfig,
    performance: str = "close_to_close",
    close_tolerance_min: int = 15,
  ) -> None:
    if performance not in _PERFORMANCE_MODES:
      raise ValueError(
        f"Unsupported performance '{performance}'. Choose from {list(_PERFORMANCE_MODES)}"
      )
    self.instrument = instrument
    self.timeframe = "daily"
    self.config = config
    self.performance = performance
    self.close_tolerance_min = close_tolerance_min

    rth = config.sessions["rth"]
    self.rth_start_min: int = minute_of_day(rth.start)
    self.rth_end_min: int = minute_of_day(rth.end)

  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-day RTH summary table from 1-min OHLCV data.

    Each row is one resolved trading day, indexed by the normalized session date,
    with columns:
      session_open, session_close, prev_session_close, return_pct, day_green

    ``return_pct`` is the signed decimal return per the ``performance`` mode;
    ``day_green`` is ``return_pct >= 0``.

    The resolved-days core (RTH filter, session open/close extraction, resolution
    filter, chronological sort) is shared via ``build_resolved_days``.

    In ``close_to_close`` mode the first resolved day has no previous session
    close and is excluded (pending-sample discipline).
    """
    columns = ["session_open", "session_close", "prev_session_close", "return_pct", "day_green"]
    day = build_resolved_days(
      candles_df, self.rth_start_min, self.rth_end_min, self.close_tolerance_min
    )
    if day.empty:
      return pd.DataFrame(columns=columns)

    # build_resolved_days returns rows in chronological order, so prev_session_close
    # references the previous resolved day.
    day["prev_session_close"] = day["session_close"].shift(1)

    if self.performance == "open_to_close":
      day["return_pct"] = (day["session_close"] - day["session_open"]) / day["session_open"]
    else:  # close_to_close
      # First resolved day has no prior close: excluded (pending discipline).
      day = day[day["prev_session_close"].notna()].copy()
      day["return_pct"] = (
        (day["session_close"] - day["prev_session_close"]) / day["prev_session_close"]
      )

    day["day_green"] = day["return_pct"] >= 0
    return day[columns]

  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the five outcome rows for one (possibly sliced) day subset.

    Magnitude rows (``mean_return``, ``mean_green_move``, ``mean_red_move``) carry
    their metric in ``value``; the count rows (``green_day``, ``red_day``) use
    ``probability``. Every row reports its sample size in ``count`` / ``total``.

    If ``baseline_rows`` is provided, merges each row's baseline into the matching
    ``baseline_prob`` / ``baseline_n`` (probability rows) and ``value_baseline``
    (magnitude rows).
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    total = len(day_table)
    if total > 0:
      returns = day_table["return_pct"].astype(float)
      green_mask = day_table["day_green"].astype(bool)
    else:
      returns = pd.Series([], dtype=float)
      green_mask = pd.Series([], dtype=bool)

    green_count = int(green_mask.sum())
    red_count = total - green_count

    mean_return = float(returns.mean()) if total > 0 else 0.0
    mean_green = float(returns[green_mask].mean()) if green_count > 0 else 0.0
    mean_red = float(returns[~green_mask].mean()) if red_count > 0 else 0.0

    # (outcome, count, probability, value, agg) — value/agg are None for
    # probability rows.
    specs: list[tuple[str, int, float, float | None, str | None]] = [
      ("mean_return", total, 0.0, mean_return, "mean"),
      ("green_day", green_count, green_count / total if total > 0 else 0.0, None, None),
      ("red_day", red_count, red_count / total if total > 0 else 0.0, None, None),
      ("mean_green_move", green_count, 0.0, mean_green, "mean"),
      ("mean_red_move", red_count, 0.0, mean_red, "mean"),
    ]

    rows: list[StatResultRow] = []
    for out_key, count, probability, value, agg in specs:
      bl = baseline_map.get((_CONDITION, out_key))
      rows.append(
        StatResultRow(
          condition=_CONDITION,
          outcome=out_key,
          count=count,
          total=total,
          probability=probability,
          baseline_prob=bl.probability if bl else 0.0,
          baseline_n=bl.total if bl else 0,
          value=value,
          value_baseline=(bl.value if bl else None) if value is not None else None,
          agg=agg,
        )
      )
    return rows

  def classify_samples(self, day_table: pd.DataFrame) -> list[SampleRow]:
    """Per-day SampleRows mirroring the five outcomes ``compute_rows`` aggregates.

    Every resolved day (all rows are already countable; pending-sample discipline
    is enforced in ``build_day_table``) yields:
      - one ``mean_return`` sample carrying that day's ``return_pct`` as ``value``
        (every day counts toward this outcome);
      - one ``green_day`` OR ``red_day`` sample (mutually exclusive, no ``value``);
      - one ``mean_green_move`` sample (only green days) OR ``mean_red_move``
        sample (only red days), again carrying ``return_pct`` as ``value``.
    All samples share the single condition ``any_day``.
    """
    if day_table.empty:
      return []

    samples: list[SampleRow] = []
    for ts, return_pct, day_green in zip(
      day_table.index, day_table["return_pct"], day_table["day_green"]
    ):
      date_str = ts.strftime("%Y-%m-%d")
      value = float(return_pct)
      samples.append(
        SampleRow(date=date_str, condition=_CONDITION, outcome="mean_return", value=value)
      )
      if day_green:
        samples.append(SampleRow(date=date_str, condition=_CONDITION, outcome="green_day"))
        samples.append(
          SampleRow(date=date_str, condition=_CONDITION, outcome="mean_green_move", value=value)
        )
      else:
        samples.append(SampleRow(date=date_str, condition=_CONDITION, outcome="red_day"))
        samples.append(
          SampleRow(date=date_str, condition=_CONDITION, outcome="mean_red_move", value=value)
        )
    return samples

  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random baseline: each day's return direction is a coin flip.

    Magnitudes are held fixed; only the sign of each day's return is randomized
    (p=0.5). Expected average return ≈ 0; expected green/red day counts ≈ 50/50.
    Uses a fixed seed for deterministic output. Operates directly on the day table
    so the framework can compute a baseline per slice group as well as overall.
    """
    rng = np.random.default_rng(seed)
    n = len(day_table)
    tmp = day_table.copy()
    if n > 0:
      sign = rng.integers(0, 2, size=n) * 2 - 1  # ±1
      random_return = sign * tmp["return_pct"].abs().to_numpy()
      tmp["return_pct"] = random_return
      tmp["day_green"] = tmp["return_pct"] >= 0
    return self.compute_rows(tmp, baseline_rows=None)


def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
  performance: str = "close_to_close",
) -> Path:
  """Load data and compute Performance by Weekday for the daily timeframe.

  Writes the result JSON to results/.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  stat = PerformanceByWeekday(instrument=instrument, config=config, performance=performance)
  result = stat.compute(candles_df)

  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Performance by Weekday stat")
  parser.add_argument("--instrument", default="NQ", help="Instrument name (default: NQ)")
  parser.add_argument("--data-path", default=None, help="Override parquet file path")
  parser.add_argument("--config-dir", default="config", help="Config directory (default: config)")
  parser.add_argument(
    "--performance",
    default="close_to_close",
    choices=list(_PERFORMANCE_MODES),
    help="Return basis (default: close_to_close)",
  )
  args = parser.parse_args()

  output_path = run(
    instrument=args.instrument,
    config_dir=args.config_dir,
    data_path=args.data_path,
    performance=args.performance,
  )

  print(f"Written: {output_path}")

  # Print a brief summary of results
  with open(output_path, encoding="utf-8") as f:
    data = json.load(f)

  instrument_data = data["instruments"][args.instrument]
  for tf, tf_data in instrument_data.items():
    print(f"\n  {tf}: {tf_data['total_samples']} resolved days | {tf_data['data_range']}")
    weekday = tf_data["slices"].get("weekday", {}).get("groups", {})
    for day_key, group in weekday.items():
      rows = {r["outcome"]: r for r in group["results"]}
      mean_ret = rows["mean_return"]["value"]
      green = rows["green_day"]
      print(
        f"    {day_key}: avg return={mean_ret:+.4%} "
        f"| P(green)={green['probability']:.3f} (N={green['total']})"
      )
