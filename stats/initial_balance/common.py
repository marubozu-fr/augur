"""Shared day-table base for the ``initial_balance`` stat family.

Every ``initial_balance`` consumer (``standard``, ``performance``,
``retracement``, ``time``, ``rejection``) starts from the same per-session
prefix: resolved RTH days, the IB-window extremes, the IB size (absolute and as
% of the session open), the session/prior-session/overnight colors, and the
inner-join discipline that drops any day lacking a clean initial balance OR a
non-empty breakout window. Only the LAST step differs per stat (post extremes,
first-break minute, retracement depth, first-excursion extension, formation
order). This module holds that common prefix so the consumers do not each
re-implement it, mirroring ``daily_candles.build_day_table_with_prior_range``
and ``session_candles.session_bars``.
"""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd

from stats.utils.daily_candles import build_resolved_days

# Base columns the helper computes; each consuming stat appends its own final
# column(s) and selects its own subset before returning from build_day_table.
BASE_COLUMNS: tuple[str, ...] = (
  "session_open",
  "session_close",
  "ib_high",
  "ib_low",
  "ib_size",
  "ib_size_pct",
  "session_green",
  "prev_session_green",
  "overnight_green",
)


class IbDayBase(NamedTuple):
  """The shared IB day-table prefix plus the window bars each stat consumes.

  ``day``  — per-session base table indexed by the normalized session date,
             carrying ``BASE_COLUMNS``, already filtered by the breakout-window
             presence guard. Empty (no rows) when no day survives; consumers
             check ``day.empty`` and return their own typed empty frame.
  ``ib``   — IB-window bars ``[rth_start, ib_end)``, chronologically sorted and
             tagged with ``_mod`` (minute of day) / ``_date`` (session date).
  ``post`` — breakout-window bars ``[ib_end, rth_end)``, same tagging; the
             stat-specific final column(s) are computed from these.
  """

  day: pd.DataFrame
  ib: pd.DataFrame
  post: pd.DataFrame


def build_ib_day_base(
  candles_df: pd.DataFrame,
  rth_start_min: int,
  ib_end_min: int,
  rth_end_min: int,
  close_tolerance_min: int,
) -> IbDayBase:
  """Build the shared IB day-table base and return it with the window bars.

  The ``day`` table carries ``BASE_COLUMNS``:
    ``session_open`` / ``session_close`` — the resolved-day open/close.
    ``ib_high`` / ``ib_low``             — IB-window extremes ``[rth_start, ib_end)``.
    ``ib_size``                          — ``ib_high - ib_low`` (read by ``size``).
    ``ib_size_pct``                      — ``ib_size`` as % of the session open
                                           (read by ``size_pct``).
    ``session_green``                    — session color (``session_close >= session_open``).
    ``prev_session_green``               — prior RESOLVED session's color (NaN for the
                                           first resolved day).
    ``overnight_green``                  — overnight gap: open above the prior RESOLVED
                                           session's close (NaN for the first resolved day).

  The resolved-days core (RTH filter, session open/close, resolution filter,
  chronological sort) is shared via ``build_resolved_days``. A day survives only
  when it has a clean initial balance AND a non-empty breakout window: the IB
  extremes and a breakout-window presence guard are joined onto the resolved
  index with an inner join, so days missing either drop out (pending-sample
  discipline). The DatetimeIndex (normalized session date) is required by the
  ``weekday`` slicer.

  The frame is sorted by ``timestamp`` with a fresh positional index so any
  first-touch lookup (e.g. ``rejection``'s formation order) over ``ib`` stays
  correct regardless of the source parquet's row order; the order-independent
  aggregates used by the other consumers are unaffected.
  """
  empty_day = pd.DataFrame(columns=list(BASE_COLUMNS))
  empty_bars = pd.DataFrame()

  if candles_df.empty:
    return IbDayBase(empty_day, empty_bars, empty_bars)

  resolved = build_resolved_days(
    candles_df, rth_start_min, rth_end_min, close_tolerance_min
  )
  if resolved.empty:
    return IbDayBase(empty_day, empty_bars, empty_bars)

  df = candles_df.sort_values("timestamp").reset_index(drop=True)
  df["_mod"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
  df["_date"] = df["timestamp"].dt.normalize()

  # Initial-balance window aggregates: [rth_start, ib_end).
  ib = df[(df["_mod"] >= rth_start_min) & (df["_mod"] < ib_end_min)]
  ib_high = ib.groupby("_date")["high"].max().rename("ib_high")
  ib_low = ib.groupby("_date")["low"].min().rename("ib_low")

  # Breakout-window bars: [ib_end, rth_end). ``post_present`` merely confirms the
  # window is non-empty so the inner join drops days without one; its value is
  # unused (each consumer derives its own final column(s) from ``post``).
  post = df[(df["_mod"] >= ib_end_min) & (df["_mod"] < rth_end_min)]
  post_present = post.groupby("_date")["high"].max().rename("_post_present")

  # Inner join onto resolved: a day survives only with a clean IB AND a non-empty
  # breakout window.
  day = (
    resolved.join(ib_high, how="inner")
    .join(ib_low, how="inner")
    .join(post_present, how="inner")
  )
  if day.empty:
    return IbDayBase(empty_day, ib, post)

  day = day.drop(columns=["_post_present"])

  day["ib_size"] = day["ib_high"] - day["ib_low"]
  # IB size as % of the session open price (price-relative size bucketing).
  day["ib_size_pct"] = (day["ib_size"] / day["session_open"]) * 100.0
  day["session_green"] = day["session_close"] >= day["session_open"]
  # Prior RESOLVED session's color (shift over the resolved-only, sorted index,
  # so it skips any excluded/early-close day). NaN for the first resolved day.
  day["prev_session_green"] = day["session_green"].shift(1)
  # Overnight gap: session open above the prior RESOLVED session's close. NaN for
  # the first resolved day (no prior close), so it is excluded from the slice.
  prev_close = day["session_close"].shift(1)
  day["overnight_green"] = (day["session_open"] > prev_close).where(prev_close.notna())

  return IbDayBase(day[list(BASE_COLUMNS)].copy(), ib, post)
