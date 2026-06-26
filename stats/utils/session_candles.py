"""Shared session-bar attribution for multi-session stats.

Stats that compare two named market sessions within a cycle (e.g.
``market_session_breakout``, ``market_session_correlation``) all start by
slicing the 1-min bars into a single session window and tagging each bar with
the cycle it belongs to. This module holds that common core so the families do
not each re-implement it.
"""

from __future__ import annotations

import pandas as pd


def session_bars(df: pd.DataFrame, start_min: int, end_min: int) -> pd.DataFrame:
  """Return a session's bars tagged with their cycle date in ``_cycle``.

  ``df`` must already carry the helper columns ``_mod`` (minute of day) and
  ``_date`` (normalized calendar date), which callers compute once for the whole
  frame and reuse across both sessions.

  Intraday sessions (``start < end``) keep ``[start, end)`` on each calendar
  date, attributed to that date's cycle. Cross-midnight sessions
  (``start >= end``, e.g. ``asia`` 18:00→03:00) take evening bars (at or after
  ``start``) attributed to the NEXT day's cycle and early bars (before ``end``)
  to the same day's cycle.
  """
  mod = df["_mod"]
  if start_min < end_min:
    bars = df[(mod >= start_min) & (mod < end_min)].copy()
    bars["_cycle"] = bars["_date"]
  else:
    bars = df[(mod >= start_min) | (mod < end_min)].copy()
    evening = bars["_mod"] >= start_min
    bars["_cycle"] = bars["_date"].where(
      ~evening, bars["_date"] + pd.Timedelta(days=1)
    )
  return bars
