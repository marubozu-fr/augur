"""Average Daily Range (ADR) stat — standard variant.

Measures how often a session's RTH high-to-low range exceeds or respects the
prior-session period-N ADR (Average Daily Range).

A single condition ``adr`` is reported with two outcomes that partition every
countable day:

  - ``exceeded``:  ``day_range > adr``  (strict; touching the ADR is *respected*).
  - ``respected``: ``day_range <= adr``.

The shared exceeded / respected computation, baseline, ``run`` entry point and
CLI live in :mod:`stats.range_base`; this module only defines the day table (the
plain ``day_range`` and its rolling ``adr``) and the i18n metadata.

ADR construction
----------------
``adr`` for day *d* is the rolling mean of the PREVIOUS ``period`` daily ranges,
i.e. it is built as::

    day_range.rolling(period).mean().shift(1)

The ``shift(1)`` guarantees strict prior-session status — no lookahead. The
first ``period`` resolved sessions therefore have ``NaN`` ADR (the window is not
yet full) and are EXCLUDED from every denominator (pending-sample discipline).

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


class AverageDailyRange(RangeExceedanceStat):
  """How often the session range exceeds or respects the prior-session ADR."""

  stat_name = "adr"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = ("weekday",)

  range_col = "day_range"
  ref_col = "adr"
  condition_key = "adr"

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


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
  period: int = 14,
) -> Path:
  """Load data and compute Average Daily Range for the daily timeframe."""
  return run_range_stat(
    AverageDailyRange,
    instrument=instrument,
    config_dir=config_dir,
    data_path=data_path,
    period=period,
  )


if __name__ == "__main__":
  main(AverageDailyRange, "Compute Average Daily Range (ADR) stat")
