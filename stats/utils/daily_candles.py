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
