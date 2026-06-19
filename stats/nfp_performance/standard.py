"""NFP Performance stat.

Measures close-to-close price performance across three windows around each
Non-Farm Payrolls (NFP) release: a pre-announcement window, the NFP release day
itself, and a post-announcement window.

All computation lives in :mod:`stats.event_performance_base`; this module only
sets the NFP-specific names, i18n content and the NFP calendar loader. See the
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
  en="NFP Performance",
  fr="Performance NFP",
)
_DEFINITION = I18nString(
  en=(
    "What is the average close-to-close percent return in the trading sessions "
    "before an NFP release, on the release day itself, and in the sessions after?"
  ),
  fr=(
    "Quel est le rendement moyen en pourcentage de clôture à clôture lors des "
    "séances précédant une publication du NFP, le jour de la publication, et "
    "lors des séances suivantes ?"
  ),
)
_LABELS = Labels(
  conditions={
    "pre_announcement": I18nString(
      en="Pre-announcement window", fr="Fenêtre pré-annonce"
    ),
    "nfp_day": I18nString(en="NFP release day", fr="Jour de publication du NFP"),
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

# Calendar row identifying an NFP release. The forex-factory calendar labels the
# headline monthly US payrolls print as "Non-Farm Employment Change"; it
# uniquely identifies each release date.
_NFP_EVENT = "non-farm employment change"
_NFP_CURRENCY = "usd"


def load_nfp_release_dates(calendar_path: str | Path) -> set[pd.Timestamp]:
  """Read distinct NFP release dates from an economic calendar CSV.

  Keeps rows whose ``event`` is the headline monthly US payrolls print and
  returns the distinct release dates as normalized (midnight, tz-naive)
  Timestamps.
  """
  cal = pd.read_csv(calendar_path, usecols=["date", "currency", "event"])
  event = cal["event"].str.strip().str.lower()
  currency = cal["currency"].str.strip().str.lower()
  mask = (event == _NFP_EVENT) & (currency == _NFP_CURRENCY)
  dates = pd.to_datetime(cal.loc[mask, "date"]).dt.normalize()
  return set(dates.unique())


class NFPPerformance(EventPerformanceStat):
  """Close-to-close performance across the three NFP-release windows."""

  stat_name = "nfp_performance"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = ()  # Events are scattered across the calendar; per-day slicing is meaningless.

  event_condition_key = "nfp_day"
  event_return_col = "nfp_return"


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
  """Load data and the NFP calendar, compute NFP Performance, write the result."""
  return run_event_performance_stat(
    NFPPerformance,
    load_nfp_release_dates,
    instrument=instrument,
    config_dir=config_dir,
    data_path=data_path,
    calendar_path=calendar_path,
    pre_announcement=pre_announcement,
    post_announcement=post_announcement,
  )


if __name__ == "__main__":
  main(NFPPerformance, load_nfp_release_dates, "Compute NFP Performance stat", "NFP")
