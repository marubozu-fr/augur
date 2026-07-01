"""Tests for stats.intraday_timing.standard.

All data is synthetic — no real market files required.
Expected values are hand-calculated before each assertion.

Bucket arithmetic reference (RTH start = 09:30 = 570 min):
  15min: bucket_idx = (mod - 570) // 15
    bucket 0  = 09:30 key "0930"  (mod 570-584)
    bucket 1  = 09:45 key "0945"  (mod 585-599)  e.g. 09:50 mod=590 -> (590-570)//15=1
    bucket 2  = 10:00 key "1000"  (mod 600-614)
    bucket 5  = 10:45 key "1045"  (mod 645-659)  e.g. 10:50 mod=650 -> (650-570)//15=5
    bucket 26 = 16:00 key "1600"  (mod 960-974)  partial final bucket
    n_buckets = ceil(405/15) = 27

  30min: bucket_idx = (mod - 570) // 30
    bucket 0 = 09:30 key "0930"  (mod 570-599)  e.g. 09:50 mod=590 -> (590-570)//30=0
    bucket 2 = 10:30 key "1030"  (mod 630-659)  e.g. 10:50 mod=650 -> (650-570)//30=2
    bucket 13= 16:00 key "1600"  (mod 960-974)  partial final bucket
    n_buckets = ceil(405/30) = 14

  1h: bucket_idx = (mod - 570) // 60
    bucket 0 = 09:30 key "0930"  (mod 570-629)  e.g. 09:50 mod=590 -> (590-570)//60=0
    bucket 6 = 15:30 key "1530"  (mod 930-974)  partial final bucket (~45 min)
    n_buckets = ceil(405/60) = 7
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, TimeframeResult, write_results
from stats.config import InstrumentConfig, Session
from stats.intraday_timing.standard import IntradayTiming

# ---------------------------------------------------------------------------
# Minimal InstrumentConfig — does NOT depend on NQ.yaml
# ---------------------------------------------------------------------------
_NY = "America/New_York"
_RTH_SESSION = Session(start="09:30", end="16:15")
_TEST_CONFIG = InstrumentConfig(
  instrument="NQ",
  working_timezone=_NY,
  sessions={"rth": _RTH_SESSION},
  timeframes=["15min"],
  parquet_path=Path("data/NQ_1min.parquet"),
)

# RTH minutes: 09:30 = 570, 16:15 = 975
# Resolved when last bar mod >= 975 - 15 = 960 (i.e. >= 16:00)
_RTH_START = 570   # 09:30
_RTH_END = 975     # 16:15 (exclusive in RTH filter: mod < 975)
_RTH_LAST = 974    # 16:14 (last valid 1-min bar)
_CLOSE_THRESH = 960  # 16:00 — last bar mod must be >= this to be resolved


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

def _make_day(
  date: str,
  session_open: float = 100.0,
  session_close: float = 105.0,
  high_minute: int = 571,
  high_val: float = 200.0,
  low_minute: int = 572,
  low_val: float = 50.0,
  last_mod: int | None = None,
) -> pd.DataFrame:
  """Build one trading day of 1-min OHLCV bars from 09:30 to 16:14 (or last_mod).

  Arguments:
    date         : YYYY-MM-DD string
    session_open : open price of the 09:30 bar
    session_close: close price of the last RTH bar
    high_minute  : minute-of-day (mod) of the bar that carries high_val
    high_val     : high value placed on the high_minute bar (must exceed all others)
    low_minute   : minute-of-day (mod) of the bar that carries low_val
    low_val      : low value placed on the low_minute bar (must be below all others)
    last_mod     : last bar's mod; defaults to _RTH_LAST (16:14). Pass <960 for
                   an early-close / unresolved day.

  All bars default to open=100, high=100.25, low=99.75, close=100 except:
    - The 09:30 bar uses session_open as its open.
    - The last bar uses session_close as its close.
    - The high_minute bar gets bar_high=high_val (all other highs <= 100.25).
    - The low_minute bar gets bar_low=low_val (all other lows >= 99.75).
  """
  if last_mod is None:
    last_mod = _RTH_LAST

  base = pd.Timestamp(date, tz=_NY)
  records = []
  for mod in range(_RTH_START, last_mod + 1):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)

    o = session_open if mod == _RTH_START else 100.0
    c = session_close if mod == last_mod else 100.0
    bar_high = high_val if mod == high_minute else 100.25
    bar_low = low_val if mod == low_minute else 99.75
    # Keep each bar OHLC-valid: high >= max(open, close), low <= min(open, close).
    bar_high = max(o, c, bar_high)
    bar_low = min(o, c, bar_low)

    records.append({
      "timestamp": ts,
      "open": o,
      "high": bar_high,
      "low": bar_low,
      "close": c,
      "volume": 1000,
    })

  return pd.DataFrame(records)


def _concat_days(frames: list[pd.DataFrame]) -> pd.DataFrame:
  """Concatenate per-day DataFrames into a single sorted 1-min DataFrame."""
  df = pd.concat(frames, ignore_index=True)
  return df.sort_values("timestamp").reset_index(drop=True)


def _empty_df() -> pd.DataFrame:
  df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
  df["timestamp"] = pd.array([], dtype="datetime64[ns, America/New_York]")
  return df


def _stat(timeframe: str = "15min") -> IntradayTiming:
  return IntradayTiming(instrument="NQ", config=_TEST_CONFIG, timeframe=timeframe)


def _tf(result: StatRunResult, timeframe: str = "15min") -> TimeframeResult:
  return result.instruments["NQ"][timeframe]


def _row(rows: list[StatResultRow], condition: str, outcome: str) -> StatResultRow:
  for r in rows:
    if r.condition == condition and r.outcome == outcome:
      return r
  raise KeyError(f"({condition}, {outcome})")


# ---------------------------------------------------------------------------
# Multi-day synthetic dataset
#
# We use 5 resolved sessions (Mon–Fri 2024-01-08..12) + 1 pending partial day.
#
# Session layout (15min buckets, bucket_idx = (mod-570)//15):
#   Day 1 (2024-01-08 Mon): high at mod=590 (09:50) -> bucket 1 "0945"
#                           low  at mod=600 (10:00) -> bucket 2 "1000"
#                           session_open=100, session_close=110 -> green
#   Day 2 (2024-01-09 Tue): high at mod=650 (10:50) -> bucket 5 "1045"
#                            low  at mod=590 (09:50) -> bucket 1 "0945"
#                            session_open=110, session_close=100 -> red
#   Day 3 (2024-01-10 Wed): high at mod=960 (16:00) -> bucket 26 "1600" (partial)
#                            low  at mod=600 (10:00) -> bucket 2 "1000"
#                            session_open=100, session_close=110 -> green
#   Day 4 (2024-01-11 Thu): high at mod=590 (09:50) -> bucket 1 "0945"
#                            low  at mod=960 (16:00) -> bucket 26 "1600" (partial)
#                            session_open=110, session_close=100 -> red
#   Day 5 (2024-01-12 Fri): high at mod=600 (10:00) -> bucket 2 "1000"
#                            low  at mod=590 (09:50) -> bucket 1 "0945"
#                            session_open=100, session_close=110 -> green
#   Day 6 (2024-01-15 Mon): UNRESOLVED — last bar at mod=590 (09:50 < 960)
#
# Bucket arithmetic verification:
#   mod=590: (590-570)//15 = 20//15 = 1 -> bucket 1 = "0945"
#   mod=600: (600-570)//15 = 30//15 = 2 -> bucket 2 = "1000"
#   mod=650: (650-570)//15 = 80//15 = 5 -> bucket 5 = "1045"
#   mod=960: (960-570)//15 = 390//15 = 26 -> bucket 26 = "1600"
#
# intraday_high across 5 days: bucket1="0945"->2, bucket2="1000"->1, bucket5="1045"->1, bucket26="1600"->1
# intraday_low  across 5 days: bucket1="0945"->2, bucket2="1000"->2, bucket26="1600"->1
#
# session_green: Day1=True, Day2=False, Day3=True, Day4=False, Day5=True
# green days (1,3,5): high_bucket=[1,26,2], low_bucket=[2,2,1]
#   intraday_high green: "0945"->1, "1000"->1, "1600"->1
#   intraday_low  green: "1000"->2, "0945"->1
# red days (2,4): high_bucket=[5,1], low_bucket=[1,26]
#   intraday_high red: "1045"->1, "0945"->1
#   intraday_low  red: "0945"->1, "1600"->1
#
# weekday groups (by session date):
#   monday    (0): days 1 only   -> high "0945"->1,  low "1000"->1
#   tuesday   (1): days 2 only   -> high "1045"->1,  low "0945"->1
#   wednesday (2): days 3 only   -> high "1600"->1,  low "1000"->1
#   thursday  (3): days 4 only   -> high "0945"->1,  low "1600"->1
#   friday    (4): days 5 only   -> high "1000"->1,  low "0945"->1
# ---------------------------------------------------------------------------

_DAY_SPECS: list[dict] = [
  {"date": "2024-01-08", "open": 100.0, "close": 110.0, "hmod": 590, "lmod": 600},  # Mon, green
  {"date": "2024-01-09", "open": 110.0, "close": 100.0, "hmod": 650, "lmod": 590},  # Tue, red
  {"date": "2024-01-10", "open": 100.0, "close": 110.0, "hmod": 960, "lmod": 600},  # Wed, green
  {"date": "2024-01-11", "open": 110.0, "close": 100.0, "hmod": 590, "lmod": 960},  # Thu, red
  {"date": "2024-01-12", "open": 100.0, "close": 110.0, "hmod": 600, "lmod": 590},  # Fri, green
]


def _make_five_days() -> pd.DataFrame:
  """Five fully resolved RTH sessions with known high/low bucket placement."""
  frames = []
  for s in _DAY_SPECS:
    frames.append(_make_day(
      s["date"],
      session_open=s["open"],
      session_close=s["close"],
      high_minute=s["hmod"],
      high_val=500.0,
      low_minute=s["lmod"],
      low_val=1.0,
    ))
  return _concat_days(frames)


def _make_five_days_with_unresolved() -> pd.DataFrame:
  """Five resolved days + one unresolved (last bar at mod=590 < 960)."""
  frames = []
  for s in _DAY_SPECS:
    frames.append(_make_day(
      s["date"],
      session_open=s["open"],
      session_close=s["close"],
      high_minute=s["hmod"],
      high_val=500.0,
      low_minute=s["lmod"],
      low_val=1.0,
    ))
  # Unresolved early-close day: last bar at mod=590 (09:50) — excluded
  frames.append(_make_day(
    "2024-01-15",
    session_open=100.0,
    session_close=100.0,
    high_minute=571,
    high_val=500.0,
    low_minute=572,
    low_val=1.0,
    last_mod=590,
  ))
  return _concat_days(frames)


# ===========================================================================
# 1. Empty DataFrame
# ===========================================================================

def test_empty_df_build_day_table_empty() -> None:
  """Empty input -> build_day_table returns an empty DataFrame."""
  day_table = _stat().build_day_table(_empty_df())
  assert len(day_table) == 0


def test_empty_df_compute_returns_empty_results() -> None:
  """Empty input -> compute returns total_samples=0 and empty results list."""
  result = _stat().compute(_empty_df())
  tf = _tf(result)
  assert tf.total_samples == 0
  assert tf.results == []


def test_empty_df_compute_data_range_empty() -> None:
  """Empty input -> data_range is []."""
  result = _stat().compute(_empty_df())
  assert _tf(result).data_range == []


def test_empty_df_no_slice_groups() -> None:
  """Empty input -> both slicers produce empty groups."""
  result = _stat().compute(_empty_df())
  tf = _tf(result)
  assert tf.slices["session_color"].groups == {}
  assert tf.slices["weekday"].groups == {}


# ===========================================================================
# 2. Known high/low minute placement -> correct bucket assignment
# ===========================================================================

def test_build_day_table_high_bucket_15min() -> None:
  """15min: high at mod=590 (09:50) -> (590-570)//15=1 -> bucket 1 -> "0945"."""
  df = _make_five_days()
  day_table = _stat("15min").build_day_table(df)
  assert len(day_table) == 5

  # Day 1: high at mod=590 -> bucket 1
  d1 = pd.Timestamp("2024-01-08", tz=_NY).normalize()
  assert int(day_table.loc[d1, "high_bucket"]) == 1


def test_build_day_table_low_bucket_15min() -> None:
  """15min: low at mod=600 (10:00) -> (600-570)//15=2 -> bucket 2 -> "1000"."""
  df = _make_five_days()
  day_table = _stat("15min").build_day_table(df)
  d1 = pd.Timestamp("2024-01-08", tz=_NY).normalize()
  assert int(day_table.loc[d1, "low_bucket"]) == 2


def test_build_day_table_high_bucket_mid_session_15min() -> None:
  """15min: high at mod=650 (10:50) -> (650-570)//15=5 -> bucket 5 -> "1045"."""
  df = _make_five_days()
  day_table = _stat("15min").build_day_table(df)
  d2 = pd.Timestamp("2024-01-09", tz=_NY).normalize()
  assert int(day_table.loc[d2, "high_bucket"]) == 5


def test_build_day_table_partial_final_bucket_15min() -> None:
  """15min: high at mod=960 (16:00) -> (960-570)//15=26 -> bucket 26 -> "1600" (partial)."""
  df = _make_five_days()
  day_table = _stat("15min").build_day_table(df)
  d3 = pd.Timestamp("2024-01-10", tz=_NY).normalize()
  assert int(day_table.loc[d3, "high_bucket"]) == 26


def test_build_day_table_high_bucket_30min() -> None:
  """30min: high at mod=590 (09:50) -> (590-570)//30=0 -> bucket 0 -> "0930"."""
  df = _make_five_days()
  day_table = _stat("30min").build_day_table(df)
  # Day 1: high at mod=590, 30min bucket = (590-570)//30 = 0
  d1 = pd.Timestamp("2024-01-08", tz=_NY).normalize()
  assert int(day_table.loc[d1, "high_bucket"]) == 0


def test_build_day_table_high_bucket_mid_session_30min() -> None:
  """30min: high at mod=650 (10:50) -> (650-570)//30=2 -> bucket 2 -> "1030"."""
  df = _make_five_days()
  day_table = _stat("30min").build_day_table(df)
  # Day 2: high at mod=650
  d2 = pd.Timestamp("2024-01-09", tz=_NY).normalize()
  assert int(day_table.loc[d2, "high_bucket"]) == 2


def test_build_day_table_partial_bucket_1h() -> None:
  """1h: high placed at mod=960 (16:00) -> (960-570)//60=6 -> bucket 6 -> "1530" (partial, 45min)."""
  # Build a single day with high at mod=960 (partial 1h bucket)
  df = _make_day(
    "2024-01-08",
    session_open=100.0,
    session_close=110.0,
    high_minute=960,
    high_val=500.0,
    low_minute=571,
    low_val=1.0,
  )
  day_table = _stat("1h").build_day_table(df)
  d = pd.Timestamp("2024-01-08", tz=_NY).normalize()
  # bucket index: (960-570)//60 = 390//60 = 6 -> starts at 570+6*60=930 -> 15:30
  assert int(day_table.loc[d, "high_bucket"]) == 6


def test_bucket_key_for_partial_1h_bucket() -> None:
  """1h stat has 7 buckets; bucket 6 key is "1530" (15:30)."""
  stat = _stat("1h")
  # n_buckets = ceil(405/60) = 7; bucket 6 starts at 570+360=930 -> 15:30
  assert stat.n_buckets == 7
  bucket_keys = [key for _, key in stat._bucket_order]
  assert bucket_keys[6] == "1530"


def test_bucket_key_for_partial_15min_bucket() -> None:
  """15min stat has 27 buckets; bucket 26 key is "1600"."""
  stat = _stat("15min")
  # n_buckets = ceil(405/15) = 27; bucket 26 starts at 570+390=960 -> 16:00
  assert stat.n_buckets == 27
  bucket_keys = [key for _, key in stat._bucket_order]
  assert bucket_keys[26] == "1600"


def test_bucket_key_for_partial_30min_bucket() -> None:
  """30min stat has 14 buckets; bucket 13 key is "1600"."""
  stat = _stat("30min")
  # n_buckets = ceil(405/30) = 14; bucket 13 starts at 570+390=960 -> 16:00
  assert stat.n_buckets == 14
  bucket_keys = [key for _, key in stat._bucket_order]
  assert bucket_keys[13] == "1600"


# ===========================================================================
# 3. Probabilities sum to ~1.0 per condition
# ===========================================================================

def test_probabilities_sum_to_one_intraday_high() -> None:
  """intraday_high probabilities across all present buckets sum to 1.0."""
  result = _stat().compute(_make_five_days())
  rows = _tf(result).results
  high_rows = [r for r in rows if r.condition == "intraday_high"]
  assert len(high_rows) > 0
  total = sum(r.probability for r in high_rows)
  assert total == pytest.approx(1.0)


def test_probabilities_sum_to_one_intraday_low() -> None:
  """intraday_low probabilities across all present buckets sum to 1.0."""
  result = _stat().compute(_make_five_days())
  rows = _tf(result).results
  low_rows = [r for r in rows if r.condition == "intraday_low"]
  assert len(low_rows) > 0
  total = sum(r.probability for r in low_rows)
  assert total == pytest.approx(1.0)


# ===========================================================================
# 4. Correct counts for known placement
# ===========================================================================

def test_intraday_high_counts_15min() -> None:
  """15min intraday_high: "0945"->2/5, "1000"->1/5, "1045"->1/5, "1600"->1/5.

  Days 1,4 have high in bucket 1 "0945"; day 5 in bucket 2 "1000";
  day 2 in bucket 5 "1045"; day 3 in bucket 26 "1600".
  """
  result = _stat("15min").compute(_make_five_days())
  rows = _tf(result, "15min").results

  # "0945" = bucket 1: days 1 and 4 -> count=2, P=2/5=0.4
  r0945 = _row(rows, "intraday_high", "0945")
  assert r0945.count == 2
  assert r0945.total == 5
  assert r0945.probability == pytest.approx(2 / 5)

  # "1000" = bucket 2: day 5 -> count=1, P=1/5
  r1000 = _row(rows, "intraday_high", "1000")
  assert r1000.count == 1
  assert r1000.probability == pytest.approx(1 / 5)

  # "1045" = bucket 5: day 2 -> count=1, P=1/5
  r1045 = _row(rows, "intraday_high", "1045")
  assert r1045.count == 1
  assert r1045.probability == pytest.approx(1 / 5)

  # "1600" = bucket 26: day 3 -> count=1, P=1/5
  r1600 = _row(rows, "intraday_high", "1600")
  assert r1600.count == 1
  assert r1600.probability == pytest.approx(1 / 5)


def test_intraday_low_counts_15min() -> None:
  """15min intraday_low: "0945"->2/5, "1000"->2/5, "1600"->1/5.

  Low minutes: day 1 lmod=600, day 2 lmod=590, day 3 lmod=600, day 4 lmod=960,
  day 5 lmod=590. So:
    "0945" (bucket 1) -> days 2,5 = 2
    "1000" (bucket 2) -> days 1,3 = 2
    "1600" (bucket 26) -> day 4 = 1
  Total = 5.
  """
  result = _stat("15min").compute(_make_five_days())
  rows = _tf(result, "15min").results

  # "0945" -> days 2,5: count=2, P=2/5
  r0945 = _row(rows, "intraday_low", "0945")
  assert r0945.count == 2
  assert r0945.total == 5
  assert r0945.probability == pytest.approx(2 / 5)

  # "1000" -> days 1,3: count=2, P=2/5
  r1000 = _row(rows, "intraday_low", "1000")
  assert r1000.count == 2
  assert r1000.probability == pytest.approx(2 / 5)

  # "1600" -> day 4: count=1, P=1/5
  r1600 = _row(rows, "intraday_low", "1600")
  assert r1600.count == 1
  assert r1600.probability == pytest.approx(1 / 5)


# ===========================================================================
# 5. session_green flag correctness
# ===========================================================================

def test_session_green_flag_up_day() -> None:
  """Day with close > open -> session_green=True."""
  df = _make_day(
    "2024-01-08",
    session_open=100.0,
    session_close=110.0,
    high_minute=571,
    high_val=200.0,
    low_minute=572,
    low_val=50.0,
  )
  day_table = _stat().build_day_table(df)
  d = pd.Timestamp("2024-01-08", tz=_NY).normalize()
  assert bool(day_table.loc[d, "session_green"]) is True


def test_session_green_flag_down_day() -> None:
  """Day with close < open -> session_green=False."""
  df = _make_day(
    "2024-01-08",
    session_open=110.0,
    session_close=100.0,
    high_minute=571,
    high_val=200.0,
    low_minute=572,
    low_val=50.0,
  )
  day_table = _stat().build_day_table(df)
  d = pd.Timestamp("2024-01-08", tz=_NY).normalize()
  assert bool(day_table.loc[d, "session_green"]) is False


def test_session_green_flag_doji_is_green() -> None:
  """Day with close == open -> session_green=True (close >= open)."""
  df = _make_day(
    "2024-01-08",
    session_open=100.0,
    session_close=100.0,
    high_minute=571,
    high_val=200.0,
    low_minute=572,
    low_val=50.0,
  )
  day_table = _stat().build_day_table(df)
  d = pd.Timestamp("2024-01-08", tz=_NY).normalize()
  assert bool(day_table.loc[d, "session_green"]) is True


# ===========================================================================
# 6. Early-close day excluded; clean day kept
# ===========================================================================

def test_early_close_day_excluded() -> None:
  """Day whose last bar is at mod=590 (09:50 < 960) is excluded as unresolved."""
  # One clean day + one early-close day
  clean = _make_day(
    "2024-01-08",
    high_minute=571, high_val=200.0,
    low_minute=572, low_val=50.0,
  )
  early_close = _make_day(
    "2024-01-09",
    high_minute=571, high_val=200.0,
    low_minute=572, low_val=50.0,
    last_mod=590,  # last bar at 09:50 — not resolved
  )
  df = _concat_days([clean, early_close])
  day_table = _stat().build_day_table(df)
  assert len(day_table) == 1
  d = pd.Timestamp("2024-01-08", tz=_NY).normalize()
  assert d in day_table.index


def test_clean_day_kept() -> None:
  """Day whose last bar is at mod=974 (16:14 >= 960) is kept as resolved."""
  df = _make_day(
    "2024-01-08",
    high_minute=571, high_val=200.0,
    low_minute=572, low_val=50.0,
    last_mod=_RTH_LAST,
  )
  day_table = _stat().build_day_table(df)
  assert len(day_table) == 1


def test_unresolved_in_five_days_excluded() -> None:
  """Five resolved days + one unresolved -> only 5 rows in day table."""
  df = _make_five_days_with_unresolved()
  day_table = _stat().build_day_table(df)
  assert len(day_table) == 5


def test_total_samples_five_resolved() -> None:
  """Five resolved sessions -> total_samples == 5."""
  result = _stat().compute(_make_five_days())
  assert _tf(result).total_samples == 5


# ===========================================================================
# 7. Slices: weekday groups
# ===========================================================================

def test_weekday_slice_groups_present() -> None:
  """Five Mon–Fri days -> weekday slicer has exactly 5 groups."""
  result = _stat().compute(_make_five_days())
  groups = _tf(result).slices["weekday"].groups
  assert set(groups.keys()) == {"monday", "tuesday", "wednesday", "thursday", "friday"}


def test_weekday_slice_each_has_one_sample() -> None:
  """Each weekday group has exactly 1 sample (one day per weekday)."""
  result = _stat().compute(_make_five_days())
  for key, grp in _tf(result).slices["weekday"].groups.items():
    assert grp.total_samples == 1, f"{key} should have 1 sample"


def test_weekday_monday_high_bucket() -> None:
  """Monday (day 1): high at mod=590 -> bucket 1 "0945" -> P=1.0."""
  result = _stat("15min").compute(_make_five_days())
  mon_rows = _tf(result, "15min").slices["weekday"].groups["monday"].results
  r = _row(mon_rows, "intraday_high", "0945")
  assert r.count == 1
  assert r.probability == pytest.approx(1.0)


def test_weekday_tuesday_high_bucket() -> None:
  """Tuesday (day 2): high at mod=650 -> bucket 5 "1045" -> P=1.0."""
  result = _stat("15min").compute(_make_five_days())
  tue_rows = _tf(result, "15min").slices["weekday"].groups["tuesday"].results
  r = _row(tue_rows, "intraday_high", "1045")
  assert r.count == 1
  assert r.probability == pytest.approx(1.0)


def test_weekday_wednesday_high_bucket() -> None:
  """Wednesday (day 3): high at mod=960 -> bucket 26 "1600" -> P=1.0."""
  result = _stat("15min").compute(_make_five_days())
  wed_rows = _tf(result, "15min").slices["weekday"].groups["wednesday"].results
  r = _row(wed_rows, "intraday_high", "1600")
  assert r.count == 1
  assert r.probability == pytest.approx(1.0)


def test_weekday_probs_sum_to_one() -> None:
  """Per-weekday-group probabilities for each condition sum to 1.0."""
  result = _stat("15min").compute(_make_five_days())
  slc = _tf(result, "15min").slices["weekday"]
  for grp_key, grp in slc.groups.items():
    for cond in ("intraday_high", "intraday_low"):
      cond_rows = [r for r in grp.results if r.condition == cond]
      total = sum(r.probability for r in cond_rows)
      assert total == pytest.approx(1.0), f"weekday={grp_key} cond={cond}: sum={total}"


# ===========================================================================
# 8. Slices: session_color (green/red)
# ===========================================================================

def test_session_color_groups_present() -> None:
  """3 green + 2 red days -> session_color slicer has green and red groups."""
  result = _stat().compute(_make_five_days())
  groups = _tf(result).slices["session_color"].groups
  assert set(groups.keys()) == {"green", "red"}


def test_session_color_green_sample_count() -> None:
  """Green group: days 1, 3, 5 -> 3 samples."""
  result = _stat().compute(_make_five_days())
  green = _tf(result).slices["session_color"].groups["green"]
  assert green.total_samples == 3


def test_session_color_red_sample_count() -> None:
  """Red group: days 2, 4 -> 2 samples."""
  result = _stat().compute(_make_five_days())
  red = _tf(result).slices["session_color"].groups["red"]
  assert red.total_samples == 2


def test_session_color_green_high_counts() -> None:
  """Green days (1,3,5): high buckets are 1("0945"), 26("1600"), 2("1000") -> each count=1, P=1/3.

  Day 1 high_bucket=1 "0945", day 3 high_bucket=26 "1600", day 5 high_bucket=2 "1000".
  """
  result = _stat("15min").compute(_make_five_days())
  green_rows = _tf(result, "15min").slices["session_color"].groups["green"].results

  r0945 = _row(green_rows, "intraday_high", "0945")
  assert r0945.count == 1
  assert r0945.total == 3
  assert r0945.probability == pytest.approx(1 / 3)

  r1000 = _row(green_rows, "intraday_high", "1000")
  assert r1000.count == 1
  assert r1000.probability == pytest.approx(1 / 3)

  r1600 = _row(green_rows, "intraday_high", "1600")
  assert r1600.count == 1
  assert r1600.probability == pytest.approx(1 / 3)


def test_session_color_red_high_counts() -> None:
  """Red days (2,4): high buckets are 5("1045"), 1("0945") -> each count=1, P=1/2.

  Day 2 high_bucket=5 "1045", day 4 high_bucket=1 "0945".
  """
  result = _stat("15min").compute(_make_five_days())
  red_rows = _tf(result, "15min").slices["session_color"].groups["red"].results

  r0945 = _row(red_rows, "intraday_high", "0945")
  assert r0945.count == 1
  assert r0945.total == 2
  assert r0945.probability == pytest.approx(1 / 2)

  r1045 = _row(red_rows, "intraday_high", "1045")
  assert r1045.count == 1
  assert r1045.probability == pytest.approx(1 / 2)


def test_session_color_green_low_counts() -> None:
  """Green days (1,3,5): low buckets 2("1000"), 2("1000"), 1("0945") -> "1000"=2/3, "0945"=1/3.

  Day 1 lmod=600->bucket2, day 3 lmod=600->bucket2, day 5 lmod=590->bucket1.
  """
  result = _stat("15min").compute(_make_five_days())
  green_rows = _tf(result, "15min").slices["session_color"].groups["green"].results

  r1000 = _row(green_rows, "intraday_low", "1000")
  assert r1000.count == 2
  assert r1000.total == 3
  assert r1000.probability == pytest.approx(2 / 3)

  r0945 = _row(green_rows, "intraday_low", "0945")
  assert r0945.count == 1
  assert r0945.probability == pytest.approx(1 / 3)


def test_session_color_red_low_counts() -> None:
  """Red days (2,4): low buckets 1("0945"), 26("1600") -> each count=1, P=1/2.

  Day 2 lmod=590->bucket1, day 4 lmod=960->bucket26.
  """
  result = _stat("15min").compute(_make_five_days())
  red_rows = _tf(result, "15min").slices["session_color"].groups["red"].results

  r0945 = _row(red_rows, "intraday_low", "0945")
  assert r0945.count == 1
  assert r0945.total == 2
  assert r0945.probability == pytest.approx(1 / 2)

  r1600 = _row(red_rows, "intraday_low", "1600")
  assert r1600.count == 1
  assert r1600.probability == pytest.approx(1 / 2)


def test_session_color_green_probs_sum_to_one() -> None:
  """Green subset: probabilities for each condition sum to 1.0."""
  result = _stat("15min").compute(_make_five_days())
  green_rows = _tf(result, "15min").slices["session_color"].groups["green"].results
  for cond in ("intraday_high", "intraday_low"):
    cond_rows = [r for r in green_rows if r.condition == cond]
    total = sum(r.probability for r in cond_rows)
    assert total == pytest.approx(1.0), f"green cond={cond}: sum={total}"


def test_session_color_red_probs_sum_to_one() -> None:
  """Red subset: probabilities for each condition sum to 1.0."""
  result = _stat("15min").compute(_make_five_days())
  red_rows = _tf(result, "15min").slices["session_color"].groups["red"].results
  for cond in ("intraday_high", "intraday_low"):
    cond_rows = [r for r in red_rows if r.condition == cond]
    total = sum(r.probability for r in cond_rows)
    assert total == pytest.approx(1.0), f"red cond={cond}: sum={total}"


def test_green_only_dataset_no_red_slice() -> None:
  """Dataset with only green days -> no red group in session_color slice."""
  frames = []
  dates = ["2024-01-08", "2024-01-09", "2024-01-10"]
  for date in dates:
    frames.append(_make_day(
      date, session_open=100.0, session_close=110.0,
      high_minute=571, high_val=200.0,
      low_minute=572, low_val=50.0,
    ))
  df = _concat_days(frames)
  result = _stat().compute(df)
  groups = _tf(result).slices["session_color"].groups
  assert "red" not in groups
  assert "green" in groups


def test_red_only_dataset_no_green_slice() -> None:
  """Dataset with only red days -> no green group in session_color slice."""
  frames = []
  dates = ["2024-01-08", "2024-01-09", "2024-01-10"]
  for date in dates:
    frames.append(_make_day(
      date, session_open=110.0, session_close=100.0,
      high_minute=571, high_val=200.0,
      low_minute=572, low_val=50.0,
    ))
  df = _concat_days(frames)
  result = _stat().compute(df)
  groups = _tf(result).slices["session_color"].groups
  assert "green" not in groups
  assert "red" in groups


def test_green_only_probs_sum_to_one() -> None:
  """Green-only subset: probabilities still sum to 1.0 per condition."""
  frames = []
  dates = ["2024-01-08", "2024-01-09", "2024-01-10"]
  high_mods = [571, 600, 650]
  low_mods = [572, 601, 651]
  for i, date in enumerate(dates):
    frames.append(_make_day(
      date, session_open=100.0, session_close=110.0,
      high_minute=high_mods[i], high_val=200.0,
      low_minute=low_mods[i], low_val=50.0,
    ))
  df = _concat_days(frames)
  result = _stat("15min").compute(df)
  green_rows = _tf(result, "15min").slices["session_color"].groups["green"].results
  for cond in ("intraday_high", "intraday_low"):
    cond_rows = [r for r in green_rows if r.condition == cond]
    total = sum(r.probability for r in cond_rows)
    assert total == pytest.approx(1.0), f"green-only cond={cond}: sum={total}"


# ===========================================================================
# 9. Baseline determinism
# ===========================================================================

def test_baseline_rows_same_seed_deterministic() -> None:
  """Two baseline_rows calls with the same seed -> identical output."""
  stat = _stat("15min")
  df = _make_five_days()
  day_table = stat.build_day_table(df)
  rows_a = stat.baseline_rows(day_table, seed=42)
  rows_b = stat.baseline_rows(day_table, seed=42)
  assert len(rows_a) == len(rows_b)
  for a, b in zip(rows_a, rows_b):
    assert a.condition == b.condition
    assert a.outcome == b.outcome
    assert a.count == b.count
    assert a.probability == pytest.approx(b.probability)


def test_baseline_rows_probs_sum_to_one() -> None:
  """Baseline probabilities for each condition sum to ~1.0."""
  stat = _stat("15min")
  day_table = stat.build_day_table(_make_five_days())
  bl_rows = stat.baseline_rows(day_table, seed=42)
  for cond in ("intraday_high", "intraday_low"):
    cond_rows = [r for r in bl_rows if r.condition == cond]
    total = sum(r.probability for r in cond_rows)
    assert total == pytest.approx(1.0), f"baseline {cond}: sum={total}"


def test_baseline_totals_match_day_table_size() -> None:
  """Baseline row totals equal the number of resolved days."""
  stat = _stat("15min")
  day_table = stat.build_day_table(_make_five_days())
  bl_rows = stat.baseline_rows(day_table, seed=42)
  for r in bl_rows:
    assert r.total == 5, f"baseline total mismatch for ({r.condition},{r.outcome})"


def test_baseline_embedded_in_compute() -> None:
  """compute() embeds baseline: every result row has baseline_n > 0."""
  result = _stat("15min").compute(_make_five_days())
  for row in _tf(result, "15min").results:
    assert row.baseline_n > 0, f"({row.condition},{row.outcome}) baseline_n=0"
    assert isinstance(row.baseline_prob, float)


# ===========================================================================
# 10. Full reproducibility
# ===========================================================================

def test_compute_twice_identical_model_dump() -> None:
  """compute() twice on the same input -> identical model_dump()."""
  stat = _stat("15min")
  df = _make_five_days()
  result_a = stat.compute(df)
  result_b = stat.compute(df)
  assert result_a.model_dump() == result_b.model_dump()


def test_compute_twice_identical_row_probabilities() -> None:
  """compute() rows are identical on two runs (deterministic seed)."""
  stat = _stat("15min")
  df = _make_five_days()
  rows_a = _tf(stat.compute(df), "15min").results
  rows_b = _tf(stat.compute(df), "15min").results
  assert len(rows_a) == len(rows_b)
  for a, b in zip(rows_a, rows_b):
    assert a.condition == b.condition
    assert a.outcome == b.outcome
    assert a.probability == pytest.approx(b.probability)
    assert a.baseline_prob == pytest.approx(b.baseline_prob)


# ===========================================================================
# 11. Outcome labels (i18n)
# ===========================================================================

def test_outcome_label_0930_en_fr() -> None:
  """Bucket key "0930" has label "09:30" in both en and fr."""
  stat = _stat("15min")
  label = stat.labels.outcomes["0930"]
  assert label.en == "09:30"
  assert label.fr == "09:30"


def test_outcome_label_0945_en_fr() -> None:
  """Bucket key "0945" has label "09:45" in both en and fr."""
  stat = _stat("15min")
  label = stat.labels.outcomes["0945"]
  assert label.en == "09:45"
  assert label.fr == "09:45"


def test_outcome_label_1600_present_15min() -> None:
  """15min: partial final bucket "1600" has label "16:00"."""
  stat = _stat("15min")
  label = stat.labels.outcomes["1600"]
  assert label.en == "16:00"
  assert label.fr == "16:00"


def test_outcome_labels_all_buckets_have_labels_15min() -> None:
  """All 27 bucket keys for 15min timeframe have non-empty en/fr labels."""
  stat = _stat("15min")
  assert len(stat.labels.outcomes) == 27
  for key, label in stat.labels.outcomes.items():
    assert label.en != "", f"bucket {key} has empty en label"
    assert label.fr != "", f"bucket {key} has empty fr label"


def test_condition_labels_present() -> None:
  """Both condition keys have non-empty en and fr labels."""
  stat = _stat("15min")
  for key in ("intraday_high", "intraday_low"):
    assert key in stat.labels.conditions
    label = stat.labels.conditions[key]
    assert label.en != "", f"{key}.en is empty"
    assert label.fr != "", f"{key}.fr is empty"


def test_i18n_title_and_definition() -> None:
  """Result carries non-empty title and definition in en and fr."""
  result = _stat().compute(_empty_df())
  assert result.title.en != ""
  assert result.title.fr != ""
  assert result.definition.en != ""
  assert result.definition.fr != ""


def test_i18n_dimension_labels_in_result() -> None:
  """compute() surfaces weekday and session_color dimension labels."""
  result = _stat().compute(_make_five_days())
  dims = result.labels.dimensions
  assert "weekday" in dims
  assert "session_color" in dims
  assert dims["weekday"].en != ""
  assert dims["session_color"].en != ""


# ===========================================================================
# 12. Single day edge case
# ===========================================================================

def test_single_day_one_bucket_per_condition() -> None:
  """Single resolved day -> exactly one bucket per condition has P=1.0; all others P=0.0.

  compute_rows emits a row for every bucket in the union of present_buckets (all
  27 for a full RTH day), but only the bucket containing the actual extreme has
  count=1 and P=1.0. The rest have count=0 and P=0.0.

  high at mod=590 -> bucket 1 "0945" has P=1.0.
  low  at mod=600 -> bucket 2 "1000" has P=1.0.
  """
  df = _make_day(
    "2024-01-08",
    session_open=100.0,
    session_close=110.0,
    high_minute=590,
    high_val=200.0,
    low_minute=600,
    low_val=50.0,
  )
  result = _stat("15min").compute(df)
  tf = _tf(result, "15min")
  assert tf.total_samples == 1

  # Only one bucket carries P=1.0 per condition; verify it is the correct bucket.
  high_rows = [r for r in tf.results if r.condition == "intraday_high"]
  assert sum(r.probability for r in high_rows) == pytest.approx(1.0)
  high_peak = max(high_rows, key=lambda r: r.probability)
  assert high_peak.outcome == "0945"
  assert high_peak.probability == pytest.approx(1.0)

  low_rows = [r for r in tf.results if r.condition == "intraday_low"]
  assert sum(r.probability for r in low_rows) == pytest.approx(1.0)
  low_peak = max(low_rows, key=lambda r: r.probability)
  assert low_peak.outcome == "1000"
  assert low_peak.probability == pytest.approx(1.0)


def test_single_day_data_range() -> None:
  """Single resolved day -> data_range is [date, date]."""
  df = _make_day(
    "2024-01-08",
    session_open=100.0,
    session_close=110.0,
    high_minute=571,
    high_val=200.0,
    low_minute=572,
    low_val=50.0,
  )
  result = _stat().compute(df)
  dr = _tf(result).data_range
  assert dr == ["2024-01-08", "2024-01-08"]


# ===========================================================================
# 13. data_range reflects resolved day table
# ===========================================================================

def test_data_range_five_days() -> None:
  """Five days Mon-Fri -> data_range spans 2024-01-08 to 2024-01-12."""
  result = _stat().compute(_make_five_days())
  dr = _tf(result).data_range
  assert dr[0] == "2024-01-08"
  assert dr[1] == "2024-01-12"


# ===========================================================================
# 14. write_results round-trip
# ===========================================================================

def test_write_results_round_trip(tmp_path: Path) -> None:
  """write_results serialises the result; model_validate round-trips correctly."""
  result = _stat("15min").compute(_make_five_days())
  written = write_results(result, results_dir=tmp_path)

  assert written.name == "intraday_timing.json"
  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)

  tf = validated.instruments["NQ"]["15min"]
  assert tf.total_samples == 5

  # Spot-check: intraday_high "0945" count=2 after round-trip
  rows = tf.results
  r = _row(rows, "intraday_high", "0945")
  assert r.count == 2
  assert r.probability == pytest.approx(2 / 5)


def test_write_results_stat_name(tmp_path: Path) -> None:
  """Serialised file name matches stat_name "intraday_timing"."""
  result = _stat().compute(_make_five_days())
  written = write_results(result, results_dir=tmp_path)
  assert written.stem == "intraday_timing"
  raw = json.loads(written.read_text(encoding="utf-8"))
  assert raw["stat_name"] == "intraday_timing"


# ===========================================================================
# 15. present_buckets column
# ===========================================================================

def test_present_buckets_column_is_sorted_tuple() -> None:
  """present_buckets for each day is a sorted tuple of int bucket indices."""
  df = _make_five_days()
  day_table = _stat("15min").build_day_table(df)
  for date, row in day_table.iterrows():
    pb = row["present_buckets"]
    assert isinstance(pb, tuple), f"{date}: present_buckets is not a tuple"
    assert list(pb) == sorted(pb), f"{date}: present_buckets not sorted"


def test_present_buckets_contains_high_and_low_buckets() -> None:
  """present_buckets for each day includes both the high_bucket and low_bucket."""
  df = _make_five_days()
  day_table = _stat("15min").build_day_table(df)
  for date, row in day_table.iterrows():
    pb = row["present_buckets"]
    assert int(row["high_bucket"]) in pb, f"{date}: high_bucket not in present_buckets"
    assert int(row["low_bucket"]) in pb, f"{date}: low_bucket not in present_buckets"


def test_present_buckets_spans_full_rth() -> None:
  """A full RTH day (09:30-16:14) has present_buckets covering all 27 buckets (15min)."""
  df = _make_day(
    "2024-01-08",
    high_minute=571, high_val=200.0,
    low_minute=572, low_val=50.0,
  )
  day_table = _stat("15min").build_day_table(df)
  d = pd.Timestamp("2024-01-08", tz=_NY).normalize()
  pb = day_table.loc[d, "present_buckets"]
  # Full RTH: 09:30-16:14 spans mods 570-974 = 405 bars across 27 buckets
  assert len(pb) == 27
  assert pb[0] == 0
  assert pb[-1] == 26


# ===========================================================================
# 16. Invalid timeframe raises ValueError
# ===========================================================================

def test_invalid_timeframe_raises() -> None:
  """IntradayTiming with unknown timeframe raises ValueError."""
  with pytest.raises(ValueError, match="Unknown timeframe"):
    IntradayTiming(instrument="NQ", config=_TEST_CONFIG, timeframe="2h")


# ===========================================================================
# 17. classify_samples
#
# Reusing _DAY_SPECS (see the multi-day dataset comment above):
#   2024-01-08: high_bucket=1 "0945", low_bucket=2 "1000"
#   2024-01-09: high_bucket=5 "1045", low_bucket=1 "0945"
#   2024-01-10: high_bucket=26 "1600", low_bucket=2 "1000"
#   2024-01-11: high_bucket=1 "0945", low_bucket=26 "1600"
#   2024-01-12: high_bucket=2 "1000", low_bucket=1 "0945"
# Each resolved day emits TWO SampleRows (intraday_high, intraday_low).
# ===========================================================================

def test_classify_samples_exact_list() -> None:
  """classify_samples emits two SampleRows per resolved day, matching the hand-calc."""
  stat = _stat("15min")
  table = stat.build_day_table(_make_five_days())
  samples = stat.classify_samples(table)
  assert [(s.date, s.condition, s.outcome) for s in samples] == [
    ("2024-01-08", "intraday_high", "0945"),
    ("2024-01-08", "intraday_low", "1000"),
    ("2024-01-09", "intraday_high", "1045"),
    ("2024-01-09", "intraday_low", "0945"),
    ("2024-01-10", "intraday_high", "1600"),
    ("2024-01-10", "intraday_low", "1000"),
    ("2024-01-11", "intraday_high", "0945"),
    ("2024-01-11", "intraday_low", "1600"),
    ("2024-01-12", "intraday_high", "1000"),
    ("2024-01-12", "intraday_low", "0945"),
  ]
  assert all(s.value is None for s in samples)


def test_classify_samples_matches_compute_rows_counts() -> None:
  """For every StatResultRow, the matching SampleRow count equals r.count."""
  stat = _stat("15min")
  table = stat.build_day_table(_make_five_days())
  samples = stat.classify_samples(table)
  rows = stat.compute_rows(table)
  for r in rows:
    n = sum(1 for s in samples if s.condition == r.condition and s.outcome == r.outcome)
    assert n == r.count


def test_classify_samples_empty_day_table() -> None:
  """Empty day_table -> classify_samples returns []."""
  stat = _stat("15min")
  assert stat.classify_samples(pd.DataFrame(columns=["high_bucket", "low_bucket"])) == []


def test_classify_samples_empty_df_via_build_day_table() -> None:
  """Empty raw input -> build_day_table + classify_samples returns []."""
  stat = _stat("15min")
  table = stat.build_day_table(_empty_df())
  assert stat.classify_samples(table) == []


def test_classify_samples_excludes_unresolved_day() -> None:
  """Unresolved (early-close) day is dropped by build_day_table, so it produces no SampleRow."""
  stat = _stat("15min")
  table = stat.build_day_table(_make_five_days_with_unresolved())
  samples = stat.classify_samples(table)
  sample_dates = {s.date for s in samples}
  assert "2024-01-15" not in sample_dates
  assert len(sample_dates) == 5
  assert len(samples) == 10
