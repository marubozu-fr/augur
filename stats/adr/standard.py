"""Average Daily Range (ADR) stat — standard variant.

Measures how often a session's RTH high-to-low range exceeds or respects the
prior-session period-N ADR (Average Daily Range).

A single condition ``adr`` is reported with two outcomes that partition every
countable day:

  - ``exceeded``:  ``day_range > adr``  (strict; touching the ADR is *respected*).
  - ``respected``: ``day_range <= adr``.

ADR construction
----------------
``adr`` for day *d* is the rolling mean of the PREVIOUS ``period`` daily ranges,
i.e. it is built as::

    day_range.rolling(period).mean().shift(1)

The ``shift(1)`` guarantees strict prior-session status — no lookahead.  The
first ``period`` resolved sessions therefore have ``NaN`` ADR (the window is not
yet full) and are EXCLUDED from every denominator (pending-sample discipline).
``adr > 0`` is asserted defensively before counting a day as countable.

Declared slices:

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
  StatResultRow,
  write_results,
)
from stats.config import InstrumentConfig, load_config, minute_of_day
from stats.utils.daily_candles import build_day_table_with_prior_range

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="Average Daily Range (ADR)",
  fr="Range journalier moyen (ADR)",
)
_DEFINITION = I18nString(
  en="How often does a session's high-to-low range exceed or respect the prior session's period-N average daily range (ADR)?",
  fr="À quelle fréquence le range (plus haut – plus bas) d'une session dépasse-t-il ou respecte-t-il le range journalier moyen (ADR) sur N périodes de la session précédente ?",
)
_LABELS = Labels(
  conditions={
    "adr": I18nString(en="Daily range vs ADR", fr="Range journalier vs ADR"),
  },
  outcomes={
    "exceeded": I18nString(en="Exceeded ADR", fr="A dépassé l'ADR"),
    "respected": I18nString(en="Respected ADR", fr="A respecté l'ADR"),
  },
)


class AverageDailyRange(BaseStat):
  """How often the session range exceeds or respects the prior-session ADR."""

  stat_name = "adr"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = ("weekday",)

  def __init__(
    self,
    instrument: str,
    config: InstrumentConfig,
    period: int = 14,
    close_tolerance_min: int = 15,
  ) -> None:
    self.instrument = instrument
    self.timeframe = "daily"
    self.config = config
    self.period = period
    self.close_tolerance_min = close_tolerance_min

    rth = config.sessions["rth"]
    self.rth_start_min: int = minute_of_day(rth.start)
    self.rth_end_min: int = minute_of_day(rth.end)

  # -------------------------------------------------------------------------
  # Day table construction
  # -------------------------------------------------------------------------
  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-session table with daily range and prior-session ADR columns.

    Columns returned:
      ``day_range``        — RTH intraday range (``day_high - day_low``) for the
                             current session.
      ``adr``              — Rolling mean of the PREVIOUS ``self.period`` daily
                             ranges, computed as::

                                 day_range.rolling(period).mean().shift(1)

                             The ``shift(1)`` ensures no lookahead: each day's ADR
                             is the average of the ``period`` sessions STRICTLY
                             BEFORE it (ending with the prior session's range).
                             The first ``period`` resolved sessions therefore have
                             NaN ``adr`` and are excluded from every denominator
                             downstream (pending-sample discipline).
      ``prev_session_green`` — Prior session color (bool; NaN for the first
                               resolved day). Available for the ``prev_candle``
                               slicer if needed, and harmless when unused.

    The DatetimeIndex (normalized session date) is required by the ``weekday``
    slicer.
    """
    columns = ["day_range", "adr", "prev_session_green"]

    daily = build_day_table_with_prior_range(
      candles_df, self.rth_start_min, self.rth_end_min, self.close_tolerance_min
    )
    if daily.empty:
      return pd.DataFrame(columns=columns)

    daily = daily.copy()
    daily["day_range"] = daily["day_high"] - daily["day_low"]

    # Rolling ADR: mean of the prior `period` daily ranges (no lookahead).
    daily["adr"] = daily["day_range"].rolling(self.period).mean().shift(1)

    return daily[columns]

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute exceeded / respected rows against the prior-session ADR.

    Only days with a valid (non-NaN and positive) ADR are countable.  The two
    outcomes partition the countable set exactly.  If ``baseline_rows`` is
    provided, merges ``baseline_prob`` / ``baseline_n`` into the rows.
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
      return [
        _make("adr", "exceeded", 0, 0),
        _make("adr", "respected", 0, 0),
      ]

    countable = day_table["adr"].notna() & (day_table["adr"] > 0)
    countable_n = int(countable.sum())

    exceeded = countable & (day_table["day_range"] > day_table["adr"])
    exceeded_n = int(exceeded.sum())
    respected_n = countable_n - exceeded_n

    return [
      _make("adr", "exceeded", exceeded_n, countable_n),
      _make("adr", "respected", respected_n, countable_n),
    ]

  # -------------------------------------------------------------------------
  # Baseline
  # -------------------------------------------------------------------------
  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random null baseline.  Deterministic for a fixed seed.

    The ``adr`` reference column is **permuted** across all rows while
    ``day_range`` is held fixed.  This pairs each session's actual range with
    an unrelated session's ADR, destroying the temporal / regime link
    (e.g. a volatile regime's large ADR is randomly matched against a quiet
    day's small range).  The permutation also moves any NaN ``adr`` values to
    random rows, preserving the countable count exactly.

    Expected baseline: near 50% exceeded / 50% respected (since the two
    distributions are drawn from the same historical range population, a
    random ADR is roughly as likely to be above as below any given range
    value).  Fixed seed for reproducibility.
    """
    n = len(day_table)
    if n == 0:
      return self.compute_rows(day_table, baseline_rows=None)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    tmp = day_table.copy()
    tmp["adr"] = day_table["adr"].to_numpy()[perm]

    return self.compute_rows(tmp, baseline_rows=None)


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
  period: int = 14,
) -> Path:
  """Load data and compute Average Daily Range for the daily timeframe.

  Writes the result JSON to results/.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  stat = AverageDailyRange(instrument=instrument, config=config, period=period)
  result = stat.compute(candles_df)

  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Average Daily Range (ADR) stat")
  parser.add_argument("--instrument", default="NQ", help="Instrument name (default: NQ)")
  parser.add_argument("--data-path", default=None, help="Override parquet file path")
  parser.add_argument("--config-dir", default="config", help="Config directory (default: config)")
  parser.add_argument("--period", type=int, default=14, help="ADR rolling period (default: 14)")
  args = parser.parse_args()

  output_path = run(
    instrument=args.instrument,
    config_dir=args.config_dir,
    data_path=args.data_path,
    period=args.period,
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
