"""Average True Range (ATR) stat — standard variant.

Measures how often a session's True Range exceeds or respects the prior-session
period-N ATR (Average True Range). Unlike ADR (which uses the simple high-to-low
range), the True Range accounts for overnight gaps by including the prior
session's close in the range computation.

A single condition ``atr`` is reported with two outcomes that partition every
countable day:

  - ``exceeded``:  ``true_range > atr``  (strict; touching the ATR is *respected*).
  - ``respected``: ``true_range <= atr``.

The shared exceeded / respected computation, baseline, ``run`` entry point and
CLI live in :mod:`stats.range_base`; this module only defines the day table (the
gap-aware ``true_range`` and its rolling ``atr``) and the i18n metadata.

True Range construction
-----------------------
For session *d* the True Range is::

    true_range = max(
      day_high - day_low,
      abs(day_high - prev_close),
      abs(day_low  - prev_close),
    )

where ``prev_close`` is the chronologically PREVIOUS resolved session's close.
The first resolved session has no prior close (``prev_close`` is NaN), so the two
gap terms drop out and its True Range falls back to ``day_high - day_low`` — the
standard first-bar ATR convention (Wilder). It therefore remains countable and
contributes to the rolling window like any other session.

ATR construction
----------------
``atr`` for day *d* is the rolling mean of the PREVIOUS ``period`` true ranges,
i.e. it is built as::

    true_range.rolling(period).mean().shift(1)

The ``shift(1)`` guarantees strict prior-session status — no lookahead. The first
``period`` resolved sessions therefore have NaN ``atr`` (the window is not yet
full) and are EXCLUDED from every denominator (pending-sample discipline).

Declared slices:

  - ``weekday`` — the "by weekday" breakdown.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from stats.base import I18nString, Labels
from stats.range_base import RangeExceedanceStat, main, run_range_stat
from stats.utils.daily_candles import build_day_table_with_prior_range

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="Average True Range (ATR)",
  fr="Average True Range (ATR)",
)
_DEFINITION = I18nString(
  en="How often does a session's true range (high-low, accounting for the prior close gap) exceed or respect the prior session's period-N average true range (ATR)?",
  fr="À quelle fréquence le true range d'une session (plus haut – plus bas, en tenant compte du gap avec la clôture précédente) dépasse-t-il ou respecte-t-il l'average true range (ATR) sur N périodes de la session précédente ?",
)
_LABELS = Labels(
  conditions={
    "atr": I18nString(en="True range vs ATR", fr="True range vs ATR"),
  },
  outcomes={
    "exceeded": I18nString(en="Exceeded ATR", fr="A dépassé l'ATR"),
    "respected": I18nString(en="Respected ATR", fr="A respecté l'ATR"),
  },
)


class AverageTrueRange(RangeExceedanceStat):
  """How often the session true range exceeds or respects the prior-session ATR."""

  stat_name = "atr"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = ("weekday",)

  range_col = "true_range"
  ref_col = "atr"
  condition_key = "atr"

  # -------------------------------------------------------------------------
  # Day table construction
  # -------------------------------------------------------------------------
  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-session table with True Range and prior-session ATR columns.

    Columns returned:
      ``true_range``       — RTH True Range for the current session, accounting
                             for the overnight gap::

                                 max(day_high - day_low,
                                     abs(day_high - prev_close),
                                     abs(day_low  - prev_close))

                             ``prev_close`` is the prior resolved session's close
                             (NaN for the first resolved day, whose True Range then
                             falls back to ``day_high - day_low`` — the standard
                             first-bar ATR convention).
      ``atr``              — Rolling mean of the PREVIOUS ``self.period`` true
                             ranges, computed as::

                                 true_range.rolling(period).mean().shift(1)

                             The ``shift(1)`` ensures no lookahead: each day's ATR
                             is the average of the ``period`` sessions STRICTLY
                             BEFORE it. The first ``period`` resolved sessions
                             therefore have NaN ``atr`` and are excluded from every
                             denominator downstream (pending-sample discipline).
      ``prev_session_green`` — Prior session color (bool; NaN for the first
                               resolved day). Available for the ``prev_candle``
                               slicer if needed, and harmless when unused.

    The DatetimeIndex (normalized session date) is required by the ``weekday``
    slicer.
    """
    columns = ["true_range", "atr", "prev_session_green"]

    daily = build_day_table_with_prior_range(
      candles_df, self.rth_start_min, self.rth_end_min, self.close_tolerance_min
    )
    if daily.empty:
      return pd.DataFrame(columns=columns)

    daily = daily.copy()

    # Prior resolved session's close (the True Range gap reference). The shift is
    # over the resolved-only, sorted index, so it skips over any excluded day.
    prev_close = daily["session_close"].shift(1)

    high_low = daily["day_high"] - daily["day_low"]
    high_close = (daily["day_high"] - prev_close).abs()
    low_close = (daily["day_low"] - prev_close).abs()
    # True Range: max of the three components. For the first resolved day
    # prev_close is NaN, so the two gap terms are NaN and skipna leaves
    # true_range = day_high - day_low (the standard first-bar ATR convention).
    daily["true_range"] = pd.concat(
      [high_low, high_close, low_close], axis=1
    ).max(axis=1)

    # Rolling ATR: mean of the prior `period` true ranges (no lookahead).
    daily["atr"] = daily["true_range"].rolling(self.period).mean().shift(1)

    return daily[columns]


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
  period: int = 14,
) -> Path:
  """Load data and compute Average True Range for the daily timeframe."""
  return run_range_stat(
    AverageTrueRange,
    instrument=instrument,
    config_dir=config_dir,
    data_path=data_path,
    period=period,
  )


if __name__ == "__main__":
  main(AverageTrueRange, "Compute Average True Range (ATR) stat")
