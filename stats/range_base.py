"""Shared base for range-exceedance stats (ADR, ATR).

Both ADR and ATR ask the same question — how often does a session's range metric
**exceed** or **respect** a rolling prior-session average of that metric? — and
the only things that differ between them are:

  - ``range_col`` — the range column measured each session. ADR uses the plain
    high-to-low ``day_range``; ATR uses the gap-aware ``true_range``.
  - ``ref_col`` — the rolling prior-session reference column (``adr`` vs ``atr``).
  - ``condition_key`` — the condition string reported (``"adr"`` vs ``"atr"``).
  - ``build_day_table`` — how the day table (and thus ``range_col`` / ``ref_col``)
    is constructed. This stays abstract here; each subclass implements it.
  - the i18n ``title`` / ``definition`` / ``labels`` metadata.

This base holds the shared ``compute_rows`` and ``baseline_rows`` (parameterized
by the three column / condition names), plus the shared ``run`` entry point and
CLI ``main`` parameterized by the stat class.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stats.base import BaseStat, StatResultRow, write_results
from stats.config import InstrumentConfig, load_config, minute_of_day


class RangeExceedanceStat(BaseStat):
  """How often a session's range metric exceeds or respects its prior-session
  rolling average.

  Subclasses set ``range_col``, ``ref_col``, ``condition_key``, the i18n metadata
  (``stat_name``, ``title``, ``definition``, ``labels``) and the declarative
  ``slices`` tuple, and implement ``build_day_table`` — which must produce both
  the ``range_col`` and ``ref_col`` columns (the latter being the rolling mean of
  the prior ``period`` range values, shifted by one to avoid lookahead).

  A single condition (``condition_key``) is reported with two outcomes that
  partition every countable session:

    - ``exceeded``:  ``range_col > ref_col``  (strict; touching the reference is
                     *respected*).
    - ``respected``: ``range_col <= ref_col``.
  """

  # Column / condition names — set by each subclass.
  range_col: str
  ref_col: str
  condition_key: str

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
  # Core computation
  # -------------------------------------------------------------------------
  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute exceeded / respected rows against the prior-session reference.

    Only days with a valid (non-NaN and positive) ``ref_col`` are countable. The
    two outcomes partition the countable set exactly. If ``baseline_rows`` is
    provided, merges ``baseline_prob`` / ``baseline_n`` into the rows.
    """
    cond = self.condition_key
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
        _make(cond, "exceeded", 0, 0),
        _make(cond, "respected", 0, 0),
      ]

    ref = day_table[self.ref_col]
    rng = day_table[self.range_col]

    countable = ref.notna() & (ref > 0)
    countable_n = int(countable.sum())

    exceeded = countable & (rng > ref)
    exceeded_n = int(exceeded.sum())
    respected_n = countable_n - exceeded_n

    return [
      _make(cond, "exceeded", exceeded_n, countable_n),
      _make(cond, "respected", respected_n, countable_n),
    ]

  # -------------------------------------------------------------------------
  # Baseline
  # -------------------------------------------------------------------------
  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random null baseline. Deterministic for a fixed seed.

    The ``ref_col`` reference column is **permuted** across all rows while
    ``range_col`` is held fixed. This pairs each session's actual range with an
    unrelated session's reference, destroying the temporal / regime link (e.g. a
    volatile regime's large reference is randomly matched against a quiet day's
    small range). The permutation also moves any NaN reference values to random
    rows, preserving the countable count exactly.

    Expected baseline: near 50% exceeded / 50% respected (since the range and the
    reference are drawn from the same historical population, a random reference is
    roughly as likely to be above as below any given range value). Fixed seed for
    reproducibility.
    """
    n = len(day_table)
    if n == 0:
      return self.compute_rows(day_table, baseline_rows=None)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    tmp = day_table.copy()
    tmp[self.ref_col] = day_table[self.ref_col].to_numpy()[perm]

    return self.compute_rows(tmp, baseline_rows=None)


# ---------------------------------------------------------------------------
# Shared run() entry point and CLI
# ---------------------------------------------------------------------------
def run_range_stat(
  stat_cls: type[RangeExceedanceStat],
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
  period: int = 14,
) -> Path:
  """Load data and compute a range-exceedance stat for the daily timeframe.

  Writes the result JSON to results/ and returns its path.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  stat = stat_cls(instrument=instrument, config=config, period=period)
  result = stat.compute(candles_df)

  return write_results(result)


def main(stat_cls: type[RangeExceedanceStat], description: str) -> None:
  """Shared CLI: parse args, run ``stat_cls``, and print a short summary."""
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument("--instrument", default="NQ", help="Instrument name (default: NQ)")
  parser.add_argument("--data-path", default=None, help="Override parquet file path")
  parser.add_argument("--config-dir", default="config", help="Config directory (default: config)")
  parser.add_argument("--period", type=int, default=14, help="Rolling period (default: 14)")
  args = parser.parse_args()

  output_path = run_range_stat(
    stat_cls,
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
