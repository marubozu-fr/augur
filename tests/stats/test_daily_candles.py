"""Tests for stats.utils.daily_candles.build_resolved_days.

All data is synthetic. Expected values are hand-calculated before each assertion.
"""

import pandas as pd
import pytest

from stats.utils.daily_candles import build_resolved_days

_NY = "America/New_York"

# RTH minutes: 09:30 = 570, 16:15 = 975. Resolved needs last bar >= 975 - 15 = 960.
_RTH_START = 570
_RTH_END = 975
_TOL = 15
_RTH_LAST = 974  # 16:14, last bar before 16:15


def _make_day(date: str, session_open: float, session_close: float) -> pd.DataFrame:
  """One trading day of 1-min RTH bars (09:30–16:14)."""
  base = pd.Timestamp(date, tz=_NY)
  records = []
  for mod in range(_RTH_START, _RTH_LAST + 1):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    o = session_open if mod == _RTH_START else 100.0
    c = session_close if mod == _RTH_LAST else 100.0
    records.append({
      "timestamp": ts, "open": o, "high": max(o, c) + 0.25,
      "low": min(o, c) - 0.25, "close": c, "volume": 1000,
    })
  return pd.DataFrame(records)


def _make_truncated_day(date: str) -> pd.DataFrame:
  """A day whose last RTH bar is 09:50 (mod 590 < 960) — unresolved, excluded."""
  base = pd.Timestamp(date, tz=_NY)
  records = []
  for mod in range(_RTH_START, 591):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    records.append({
      "timestamp": ts, "open": 100.0, "high": 100.25, "low": 99.75,
      "close": 100.0, "volume": 500,
    })
  return pd.DataFrame(records)


def _build(df: pd.DataFrame) -> pd.DataFrame:
  return build_resolved_days(df, _RTH_START, _RTH_END, _TOL)


# ===========================================================================
# Nominal case
# ===========================================================================

def test_nominal_open_close_and_columns() -> None:
  """Two resolved days → index = dates, session_open/close from the right bars."""
  df = pd.concat(
    [
      _make_day("2024-01-02", session_open=100.0, session_close=120.0),
      _make_day("2024-01-03", session_open=100.0, session_close=80.0),
    ],
    ignore_index=True,
  )
  out = _build(df)

  assert list(out.columns) == ["session_open", "session_close"]
  assert len(out) == 2

  d2 = pd.Timestamp("2024-01-02", tz=_NY).normalize()
  d3 = pd.Timestamp("2024-01-03", tz=_NY).normalize()
  assert out.loc[d2, "session_open"] == pytest.approx(100.0)
  assert out.loc[d2, "session_close"] == pytest.approx(120.0)
  assert out.loc[d3, "session_close"] == pytest.approx(80.0)


def test_nominal_sorted_chronologically() -> None:
  """Out-of-order input is returned sorted by date."""
  df = pd.concat(
    [
      _make_day("2024-01-05", session_open=100.0, session_close=110.0),
      _make_day("2024-01-02", session_open=100.0, session_close=120.0),
    ],
    ignore_index=True,
  )
  # Shuffle rows so the input is not pre-sorted.
  df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
  out = _build(df)
  assert list(out.index) == sorted(out.index)


# ===========================================================================
# Truncated (unresolved) day excluded
# ===========================================================================

def test_truncated_day_excluded() -> None:
  """A day whose last bar is 09:50 is not resolved and must be dropped."""
  df = pd.concat(
    [
      _make_day("2024-01-02", session_open=100.0, session_close=120.0),
      _make_truncated_day("2024-01-03"),
    ],
    ignore_index=True,
  )
  out = _build(df)

  assert len(out) == 1
  truncated = pd.Timestamp("2024-01-03", tz=_NY).normalize()
  assert truncated not in out.index


def test_missing_session_open_excluded() -> None:
  """A day with no 09:30 bar (open NaN) is excluded even if it closes late."""
  full = _make_day("2024-01-02", session_open=100.0, session_close=120.0)
  no_open = _make_day("2024-01-03", session_open=100.0, session_close=120.0)
  no_open = no_open[no_open["timestamp"].dt.hour * 60 + no_open["timestamp"].dt.minute != _RTH_START]
  df = pd.concat([full, no_open], ignore_index=True)
  out = _build(df)

  assert len(out) == 1
  assert pd.Timestamp("2024-01-03", tz=_NY).normalize() not in out.index


# ===========================================================================
# Empty input
# ===========================================================================

def test_empty_dataframe() -> None:
  """Empty input returns an empty frame with the expected columns."""
  df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
  df["timestamp"] = pd.array([], dtype="datetime64[ns, America/New_York]")
  out = _build(df)

  assert out.empty
  assert list(out.columns) == ["session_open", "session_close"]


def test_no_rth_bars_returns_empty() -> None:
  """Bars entirely outside RTH (e.g. overnight) yield an empty resolved table."""
  base = pd.Timestamp("2024-01-02", tz=_NY)
  records = []
  for mod in range(0, 300):  # 00:00–04:59, all before 09:30
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    records.append({
      "timestamp": ts, "open": 100.0, "high": 100.25, "low": 99.75,
      "close": 100.0, "volume": 100,
    })
  out = _build(pd.DataFrame(records))

  assert out.empty
  assert list(out.columns) == ["session_open", "session_close"]


# ===========================================================================
# Duplicate bars at the session open
# ===========================================================================

def test_dedup_duplicate_open_bars_keeps_first() -> None:
  """Two bars at exactly 09:30 for the same day → keep the first bar's open."""
  day = _make_day("2024-01-02", session_open=100.0, session_close=120.0)
  # Duplicate the 09:30 bar with a different open; it must be ignored.
  dup = day[day["timestamp"].dt.hour * 60 + day["timestamp"].dt.minute == _RTH_START].copy()
  dup["open"] = 999.0
  df = pd.concat([day, dup], ignore_index=True)
  out = _build(df)

  d2 = pd.Timestamp("2024-01-02", tz=_NY).normalize()
  assert len(out) == 1
  assert out.loc[d2, "session_open"] == pytest.approx(100.0)
