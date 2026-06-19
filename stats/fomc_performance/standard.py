"""FOMC Performance stat.

Measures close-to-close price performance across three windows around each FOMC
rate decision: a pre-announcement window, the decision day itself, and a
post-announcement window.

All computation lives in :mod:`stats.event_performance_base`; this module only
sets the FOMC-specific names, i18n content and the FOMC calendar loader. See the
base module docstring for the methodology, pending discipline and baseline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from stats.base import I18nString, Labels
from stats.event_performance_base import (
  EventPerformanceStat,
  main,
  run_event_performance_stat,
)
from stats.event_performance_base import _DEFAULT_CALENDAR_PATH as _DEFAULT_CALENDAR_PATH

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="FOMC Performance",
  fr="Performance FOMC",
)
_DEFINITION = I18nString(
  en=(
    "What is the average close-to-close percent return in the trading sessions "
    "before an FOMC rate decision, on the decision day itself, and in the "
    "sessions after?"
  ),
  fr=(
    "Quel est le rendement moyen en pourcentage de clôture à clôture lors des "
    "séances précédant une décision de taux du FOMC, le jour de la décision, et "
    "lors des séances suivantes ?"
  ),
)
_LABELS = Labels(
  conditions={
    "pre_announcement": I18nString(
      en="Pre-announcement window", fr="Fenêtre pré-annonce"
    ),
    "fomc_day": I18nString(en="FOMC decision day", fr="Jour de décision du FOMC"),
    "post_announcement": I18nString(
      en="Post-announcement window", fr="Fenêtre post-annonce"
    ),
  },
  outcomes={
    "mean_return": I18nString(en="Average return", fr="Rendement moyen"),
    "green": I18nString(en="Green (up)", fr="Vert (hausse)"),
    "red": I18nString(en="Red (down)", fr="Rouge (baisse)"),
  },
)

# Calendar row identifying an FOMC rate decision. The forex-factory calendar
# publishes the headline rate decision as "Federal Funds Rate" (high impact,
# carrying the rate value); it uniquely identifies each decision date. The
# co-occurring "FOMC Statement" / "FOMC Press Conference" lines on the same date
# are deliberately not matched.
_FOMC_EVENT = "federal funds rate"
_FOMC_CURRENCY = "usd"


def load_fomc_release_dates(calendar_path: str | Path) -> set[pd.Timestamp]:
  """Read distinct FOMC rate-decision dates from an economic calendar CSV.

  Keeps rows whose ``event`` is the headline US rate decision and returns the
  distinct decision dates as normalized (midnight, tz-naive) Timestamps.
  """
  cal = pd.read_csv(calendar_path, usecols=["date", "currency", "event"])
  event = cal["event"].str.strip().str.lower()
  currency = cal["currency"].str.strip().str.lower()
  mask = (event == _FOMC_EVENT) & (currency == _FOMC_CURRENCY)
  dates = pd.to_datetime(cal.loc[mask, "date"]).dt.normalize()
  return set(dates.unique())


class FOMCPerformance(EventPerformanceStat):
  """Close-to-close performance across the three FOMC-decision windows."""

  stat_name = "fomc_performance"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = ()  # Events are scattered across the calendar; per-day slicing is meaningless.

  event_condition_key = "fomc_day"
  event_return_col = "fomc_return"


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
  calendar_path: str | Path = _DEFAULT_CALENDAR_PATH,
  pre_announcement: int = 5,
  post_announcement: int = 5,
) -> Path:
  """Load data and the FOMC calendar, compute FOMC Performance, write the result."""
  return run_event_performance_stat(
    FOMCPerformance,
    load_fomc_release_dates,
    instrument=instrument,
    config_dir=config_dir,
    data_path=data_path,
    calendar_path=calendar_path,
    pre_announcement=pre_announcement,
    post_announcement=post_announcement,
  )


if __name__ == "__main__":
  main(FOMCPerformance, load_fomc_release_dates, "Compute FOMC Performance stat", "FOMC")
