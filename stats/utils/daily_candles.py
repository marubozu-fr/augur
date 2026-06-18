"""Shared RTH daily-candle construction.

Every daily stat family starts from the same per-day table: keep RTH bars, take
the open of the session-open bar and the close of the last RTH bar, keep only
resolved days, sorted chronologically. This module holds that common core so the
families do not each re-implement it.
"""

from __future__ import annotations

import pandas as pd


def build_resolved_days(
  candles_df: pd.DataFrame,
  rth_start_min: int,
  rth_end_min: int,
  close_tolerance_min: int,
) -> pd.DataFrame:
  """Build the per-day RTH summary of resolved trading days (vectorized).

  Returns a DataFrame indexed by the normalized session date with columns
  ``session_open`` (open of the bar at exactly ``rth_start_min``) and
  ``session_close`` (close of the last RTH bar), sorted chronologically.

  A day is "resolved" (confirmed) when:
    - it has a bar exactly at ``rth_start_min`` (clean session open), and
    - its last RTH bar is at or after ``rth_end_min - close_tolerance_min``.

  Early-close days and the final incomplete day in the data are excluded
  (pending-sample discipline).
  """
  columns = ["session_open", "session_close"]
  if candles_df.empty:
    return pd.DataFrame(columns=columns)

  df = candles_df.copy()
  df["mod"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
  df["date"] = df["timestamp"].dt.normalize()

  # Keep only RTH bars: [rth_start_min, rth_end_min)
  rth_mask = (df["mod"] >= rth_start_min) & (df["mod"] < rth_end_min)
  rth = df[rth_mask].copy()
  if rth.empty:
    return pd.DataFrame(columns=columns)

  # session_open = open of the bar at exactly rth_start_min.
  open_bars = (
    rth[rth["mod"] == rth_start_min].set_index("date")["open"].rename("session_open")
  )
  # Defensive: guard against duplicate bars at rth_start_min for the same day.
  open_bars = open_bars[~open_bars.index.duplicated(keep="first")]
  # session_close = close of the last RTH bar for that day.
  last_bars = (
    rth.loc[rth.groupby("date")["mod"].idxmax(), ["date", "close", "mod"]]
    .set_index("date")
    .rename(columns={"close": "session_close", "mod": "last_minute"})
  )

  day = pd.concat([open_bars, last_bars], axis=1, sort=False)

  # Resolution filter: clean session open AND sufficient close coverage.
  resolved_min = rth_end_min - close_tolerance_min
  day = day[day["session_open"].notna() & (day["last_minute"] >= resolved_min)].copy()

  return day.sort_index()[columns]


def build_day_table_with_prior_range(
  candles_df: pd.DataFrame,
  rth_start_min: int,
  rth_end_min: int,
  close_tolerance_min: int = 15,
) -> pd.DataFrame:
  """Build the per-session table with RTH extremes and the prior day's range.

  Returns a DataFrame indexed by the normalized session date (the
  ``build_resolved_days`` index) with columns:
    ``session_open``, ``session_close`` (the resolved-day open/close),
    ``day_high``, ``day_low`` (RTH intraday extremes), ``prev_high``,
    ``prev_low`` (the prior RESOLVED day's extremes), and ``prev_session_green``
    (the prior session's color, ``session_close >= session_open``).

  ``prev_high`` / ``prev_low`` / ``prev_session_green`` are NaN for the first
  resolved day (no prior day), so callers exclude it from every denominator
  (pending-sample discipline). The prior values are shifted over the
  resolved-only, sorted index, so they skip over any excluded/early-close day.

  This is the common day-table core shared by the daily prior-range stats
  (``prev_days_range``, ``outside_days``, ``inside_bars``). Callers add any
  stat-specific columns (e.g. their own day-color flag) on top of this table.
  """
  columns = [
    "session_open",
    "session_close",
    "day_high",
    "day_low",
    "prev_high",
    "prev_low",
    "prev_session_green",
  ]
  empty = pd.DataFrame(columns=columns)

  if candles_df.empty:
    return empty

  resolved = build_resolved_days(
    candles_df, rth_start_min, rth_end_min, close_tolerance_min
  )
  if resolved.empty:
    return empty

  # Per-day RTH high/low from the same RTH bar filter the resolution uses.
  df = candles_df.copy()
  df["_mod"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
  df["_date"] = df["timestamp"].dt.normalize()
  rth_mask = (df["_mod"] >= rth_start_min) & (df["_mod"] < rth_end_min)
  rth = df[rth_mask]

  day_high = rth.groupby("_date")["high"].max().rename("day_high")
  day_low = rth.groupby("_date")["low"].min().rename("day_low")

  # Inner join keeps only resolved dates; result stays chronologically sorted.
  daily = resolved.join(day_high, how="inner").join(day_low, how="inner")
  if daily.empty:
    return empty

  day_green = daily["session_close"] >= daily["session_open"]
  # Prior RESOLVED day's values (shift over the resolved-only, sorted index).
  daily["prev_high"] = daily["day_high"].shift(1)
  daily["prev_low"] = daily["day_low"].shift(1)
  daily["prev_session_green"] = day_green.shift(1)

  return daily[columns]
