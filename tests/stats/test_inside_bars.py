"""Tests for stats.inside_bars.standard.

All data is synthetic — no real market files required. Expected values are
hand-calculated before each assertion.

Framing:
  - Tier 1 (condition ``inside_open``): how often the session opens inside the
    prior resolved day's range (inclusive: open >= prev_low AND open <= prev_high).
    Opening exactly at prev_high or prev_low IS inside. total = countable days
    (those with a prior resolved day).
  - Tier 2 (condition ``inside``): breakout direction given an inside open.
    Four mutually exclusive outcomes exhaust the inside set:
      - ``broke_high``:  day_high > prev_high AND day_low >= prev_low
      - ``broke_low``:   day_low  < prev_low  AND day_high <= prev_high
      - ``broke_both``:  day_high > prev_high AND day_low  < prev_low
      - ``contained``:   day_high <= prev_high AND day_low >= prev_low

The first resolved day has no prior day and is excluded from every denominator.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.inside_bars.standard import InsideBars

# ---------------------------------------------------------------------------
# Minimal InstrumentConfig (does not depend on NQ.yaml)
# ---------------------------------------------------------------------------
_NY = "America/New_York"
_RTH_SESSION = Session(start="09:30", end="16:15")
_TEST_CONFIG = InstrumentConfig(
  instrument="NQ",
  working_timezone=_NY,
  sessions={"rth": _RTH_SESSION},
  timeframes=["daily"],
  parquet_path=Path("data/NQ_1min.parquet"),
)

_RTH_START = 570   # 09:30
_RTH_LAST = 974    # 16:14 (last bar before 16:15); resolved needs last mod >= 960


def _make_day(
  date: str,
  session_open: float,
  session_close: float,
  day_high: float,
  day_low: float,
) -> pd.DataFrame:
  """One trading day of 1-min RTH bars (09:30–16:14).

  The opening bar carries ``session_open`` as its open; the last bar carries
  ``session_close`` as its close. The day's RTH extremes ``day_high`` /
  ``day_low`` are placed on a neutral mid-session bar so they are independent of
  the open/close prices. All other bars sit between the extremes.
  """
  base = pd.Timestamp(date, tz=_NY)
  mid = (_RTH_START + _RTH_LAST) // 2
  records = []
  for mod in range(_RTH_START, _RTH_LAST + 1):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    o = session_open if mod == _RTH_START else session_close
    c = session_close
    hi = day_high if mod == mid else max(o, c)
    lo = day_low if mod == mid else min(o, c)
    records.append({
      "timestamp": ts,
      "open": o,
      "high": hi,
      "low": lo,
      "close": c,
      "volume": 1000,
    })
  return pd.DataFrame(records)


def make_candles(days: list[dict]) -> pd.DataFrame:
  """Concatenate per-day specs into a single sorted 1-min OHLCV DataFrame."""
  frames = [
    _make_day(d["date"], d["open"], d["close"], d["high"], d["low"]) for d in days
  ]
  df = pd.concat(frames, ignore_index=True)
  return df.sort_values("timestamp").reset_index(drop=True)


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


def _stat() -> InsideBars:
  return InsideBars(instrument="NQ", config=_TEST_CONFIG)


def _row(result: StatRunResult, condition: str, outcome: str) -> StatResultRow:
  for r in result.instruments["NQ"]["daily"].results:
    if r.condition == condition and r.outcome == outcome:
      return r
  raise KeyError((condition, outcome))


# ===========================================================================
# Core derivation (7 days)
#
#  idx  date        open   close  high   low  | prev_high prev_low | classification
#   0   2024-01-02  100.0  100.0  110.0  90.0 |   —         —     | excluded (no prior)
#   1   2024-01-03  105.0  108.0  115.0  95.0 |  110.0     90.0   | inside, broke_high
#                                                                       (105>=90, 105<=110)
#                                                                       (115>110, 95>=90)
#   2   2024-01-04  100.0  96.0   112.0  80.0 |  115.0     95.0   | inside, broke_low
#                                                                       (100>=95, 100<=115)
#                                                                       (112<=115, 80<95)
#   3   2024-01-05  100.0  102.0  111.0  81.0 |  112.0     80.0   | inside, broke_both
#                                                                       (100>=80, 100<=112)
#                                                                       (111<112? NO: 111<=112)
#                                                                       Wait: broke_both needs
#                                                                       day_high>prev_high AND
#                                                                       day_low<prev_low.
#                                                                       prev_high=112, day_high=113
#                                                                       prev_low=80, day_low=79
#                                                                       Revise: high=113, low=79
#   3   2024-01-05  100.0  102.0  113.0  79.0 |  112.0     80.0   | inside, broke_both
#                                                                       (100>=80, 100<=112)
#                                                                       (113>112, 79<80) ✓
#   4   2024-01-08  105.0  103.0  111.0  80.0 |  113.0     79.0   | inside, contained
#                                                                       (105>=79, 105<=113)
#                                                                       (111<=113, 80>=79) ✓
#   5   2024-01-09  120.0  118.0  125.0  115.0|  111.0     80.0   | NOT inside (outside)
#                                                                       (120>111 → outside high)
#   6   2024-01-10  100.0  102.0  108.0  95.0 |  125.0     115.0  | inside, contained
#                                                                       (100>=115? NO: 100<115)
#                                                                       Revise: open must be
#                                                                       between 115 and 125.
#                                                                       open=118, high=122, low=117
#   6   2024-01-10  118.0  120.0  122.0  117.0|  125.0     115.0  | inside, contained
#                                                                       (118>=115, 118<=125) ✓
#                                                                       (122<=125, 117>=115) ✓
#
# Summary (countable = idx 1..6, n=6):
#   inside opens: idx 1,2,3,4,6 → inside_n=5
#   not inside:   idx 5          → 1 day opened outside (above)
#   Tier-1: inside_open/inside → 5/6
#
# Tier-2 (total = inside_n = 5):
#   broke_high:  idx 1  → count=1
#   broke_low:   idx 2  → count=1
#   broke_both:  idx 3  → count=1
#   contained:   idx 4, 6 → count=2
#
# total_samples = 7 (all resolved days incl. idx 0)
# ===========================================================================

_SEQ = [
  # idx 0 — anchor day, no prior day
  {"date": "2024-01-02", "open": 100.0, "close": 100.0, "high": 110.0, "low": 90.0},
  # idx 1 — inside open (105 in [90,110]); broke_high: high=115>110, low=95>=90
  {"date": "2024-01-03", "open": 105.0, "close": 108.0, "high": 115.0, "low": 95.0},
  # idx 2 — inside open (100 in [95,115]); broke_low: low=80<95, high=112<=115
  {"date": "2024-01-04", "open": 100.0, "close": 96.0, "high": 112.0, "low": 80.0},
  # idx 3 — inside open (100 in [80,112]); broke_both: high=113>112, low=79<80
  {"date": "2024-01-05", "open": 100.0, "close": 102.0, "high": 113.0, "low": 79.0},
  # idx 4 — inside open (105 in [79,113]); contained: high=111<=113, low=80>=79
  {"date": "2024-01-08", "open": 105.0, "close": 103.0, "high": 111.0, "low": 80.0},
  # idx 5 — NOT inside (120>111 → open above prior high)
  {"date": "2024-01-09", "open": 120.0, "close": 118.0, "high": 125.0, "low": 115.0},
  # idx 6 — inside open (118 in [115,125]); contained: high=122<=125, low=117>=115
  {"date": "2024-01-10", "open": 118.0, "close": 120.0, "high": 122.0, "low": 117.0},
]


def test_total_samples_counts_all_resolved_days() -> None:
  """total_samples = 7 resolved sessions (the first one is still resolved)."""
  result = _stat().compute(make_candles(_SEQ))
  assert result.instruments["NQ"]["daily"].total_samples == 7


def test_inside_open_frequency() -> None:
  """5 of 6 countable days open inside the prior range → P=5/6."""
  # idx 1,2,3,4,6 inside; idx 5 opens above → 5/6
  row = _row(_stat().compute(make_candles(_SEQ)), "inside_open", "inside")
  assert row.total == 6
  assert row.count == 5
  assert row.probability == pytest.approx(5 / 6)


def test_broke_high_count() -> None:
  """idx 1: high=115>110, low=95>=90 → broke_high. Count=1 of 5."""
  row = _row(_stat().compute(make_candles(_SEQ)), "inside", "broke_high")
  assert row.total == 5
  assert row.count == 1
  assert row.probability == pytest.approx(1 / 5)


def test_broke_low_count() -> None:
  """idx 2: low=80<95, high=112<=115 → broke_low. Count=1 of 5."""
  row = _row(_stat().compute(make_candles(_SEQ)), "inside", "broke_low")
  assert row.total == 5
  assert row.count == 1
  assert row.probability == pytest.approx(1 / 5)


def test_broke_both_count() -> None:
  """idx 3: high=113>112, low=79<80 → broke_both. Count=1 of 5."""
  row = _row(_stat().compute(make_candles(_SEQ)), "inside", "broke_both")
  assert row.total == 5
  assert row.count == 1
  assert row.probability == pytest.approx(1 / 5)


def test_contained_count() -> None:
  """idx 4,6: fully within prior range → contained. Count=2 of 5."""
  row = _row(_stat().compute(make_candles(_SEQ)), "inside", "contained")
  assert row.total == 5
  assert row.count == 2
  assert row.probability == pytest.approx(2 / 5)


def test_tier2_partition_sums_to_inside_n() -> None:
  """The four tier-2 counts sum to inside_n (= each row's total)."""
  result = _stat().compute(make_candles(_SEQ))
  bh = _row(result, "inside", "broke_high")
  bl = _row(result, "inside", "broke_low")
  bb = _row(result, "inside", "broke_both")
  ct = _row(result, "inside", "contained")
  # All totals equal inside_n=5
  assert bh.total == bl.total == bb.total == ct.total == 5
  # Four counts partition the inside_n
  assert bh.count + bl.count + bb.count + ct.count == bh.total


def test_first_day_excluded_from_denominators() -> None:
  """Tier-1 total = 6 = n_resolved - 1 (first day has no prior day)."""
  result = _stat().compute(make_candles(_SEQ))
  assert _row(result, "inside_open", "inside").total == 7 - 1


def test_five_rows_only() -> None:
  """Exactly the five expected (condition, outcome) row pairs."""
  rows = _stat().compute(make_candles(_SEQ)).instruments["NQ"]["daily"].results
  assert {(r.condition, r.outcome) for r in rows} == {
    ("inside_open", "inside"),
    ("inside", "broke_high"),
    ("inside", "broke_low"),
    ("inside", "broke_both"),
    ("inside", "contained"),
  }


def test_data_range() -> None:
  """data_range spans from the first to the last resolved session date."""
  result = _stat().compute(make_candles(_SEQ))
  assert result.instruments["NQ"]["daily"].data_range == ["2024-01-02", "2024-01-10"]


# ===========================================================================
# Inside-open boundary: INCLUSIVE (touching the level is inside)
# ===========================================================================

def test_open_exactly_at_prev_high_is_inside() -> None:
  """open == prev_high → inside (inclusive inequality: open <= prev_high)."""
  days = [
    {"date": "2024-01-02", "open": 100.0, "close": 100.0, "high": 110.0, "low": 90.0},
    # Day 1 opens exactly at prior high (110): 110 <= 110 AND 110 >= 90 → inside
    {"date": "2024-01-03", "open": 110.0, "close": 112.0, "high": 115.0, "low": 105.0},
  ]
  result = _stat().compute(make_candles(days))
  row = _row(result, "inside_open", "inside")
  assert row.total == 1
  assert row.count == 1


def test_open_exactly_at_prev_low_is_inside() -> None:
  """open == prev_low → inside (inclusive inequality: open >= prev_low)."""
  days = [
    {"date": "2024-01-02", "open": 100.0, "close": 100.0, "high": 110.0, "low": 90.0},
    # Day 1 opens exactly at prior low (90): 90 >= 90 AND 90 <= 110 → inside
    {"date": "2024-01-03", "open": 90.0, "close": 88.0, "high": 95.0, "low": 85.0},
  ]
  result = _stat().compute(make_candles(days))
  row = _row(result, "inside_open", "inside")
  assert row.total == 1
  assert row.count == 1


def test_open_strictly_above_prev_high_is_not_inside() -> None:
  """open > prev_high → NOT inside (opens outside the prior range above)."""
  days = [
    {"date": "2024-01-02", "open": 100.0, "close": 100.0, "high": 110.0, "low": 90.0},
    # Day 1 opens at 111 > 110 → strictly outside, not inside
    {"date": "2024-01-03", "open": 111.0, "close": 113.0, "high": 116.0, "low": 108.0},
  ]
  result = _stat().compute(make_candles(days))
  row = _row(result, "inside_open", "inside")
  assert row.total == 1
  assert row.count == 0


def test_open_strictly_below_prev_low_is_not_inside() -> None:
  """open < prev_low → NOT inside (opens outside the prior range below)."""
  days = [
    {"date": "2024-01-02", "open": 100.0, "close": 100.0, "high": 110.0, "low": 90.0},
    # Day 1 opens at 89 < 90 → strictly outside, not inside
    {"date": "2024-01-03", "open": 89.0, "close": 87.0, "high": 92.0, "low": 84.0},
  ]
  result = _stat().compute(make_candles(days))
  row = _row(result, "inside_open", "inside")
  assert row.total == 1
  assert row.count == 0


# ===========================================================================
# Breakout boundary: STRICT inequality (touching the level is NOT a break)
# ===========================================================================

def test_day_high_exactly_at_prev_high_is_contained() -> None:
  """day_high == prev_high (touch only, no break) and day_low >= prev_low → contained."""
  days = [
    {"date": "2024-01-02", "open": 100.0, "close": 100.0, "high": 110.0, "low": 90.0},
    # Inside open (105 in [90,110]); day_high touches 110 exactly: 110 NOT > 110 → contained
    {"date": "2024-01-03", "open": 105.0, "close": 104.0, "high": 110.0, "low": 92.0},
  ]
  result = _stat().compute(make_candles(days))
  ct = _row(result, "inside", "contained")
  bh = _row(result, "inside", "broke_high")
  assert ct.count == 1
  assert bh.count == 0


def test_day_low_exactly_at_prev_low_is_contained() -> None:
  """day_low == prev_low (touch only, no break) and day_high <= prev_high → contained."""
  days = [
    {"date": "2024-01-02", "open": 100.0, "close": 100.0, "high": 110.0, "low": 90.0},
    # Inside open (105 in [90,110]); day_low touches 90 exactly: 90 NOT < 90 → contained
    {"date": "2024-01-03", "open": 105.0, "close": 104.0, "high": 108.0, "low": 90.0},
  ]
  result = _stat().compute(make_candles(days))
  ct = _row(result, "inside", "contained")
  bl = _row(result, "inside", "broke_low")
  assert ct.count == 1
  assert bl.count == 0


# ===========================================================================
# Slices: weekday and prev_candle
# ===========================================================================

def test_declared_slices_present() -> None:
  """The standard module declares the weekday and prev_candle slices."""
  slices = _stat().compute(make_candles(_SEQ)).instruments["NQ"]["daily"].slices
  assert set(slices) == {"weekday", "prev_candle"}


def test_weekday_slice_groups_sum_to_countable() -> None:
  """Across weekday groups, inside-open counts sum to the overall inside count."""
  result = _stat().compute(make_candles(_SEQ))
  wk = result.instruments["NQ"]["daily"].slices["weekday"]
  total_inside = 0
  for grp in wk.groups.values():
    for r in grp.results:
      if r.condition == "inside_open" and r.outcome == "inside":
        total_inside += r.count
  overall = _row(result, "inside_open", "inside").count
  assert total_inside == overall


def test_prev_candle_slice_splits_green_red() -> None:
  """prev_candle slice exposes green/red groups for the prior session color."""
  result = _stat().compute(make_candles(_SEQ))
  pc = result.instruments["NQ"]["daily"].slices["prev_candle"]
  assert set(pc.groups).issubset({"green", "red"})
  assert len(pc.groups) >= 1


def test_prev_candle_green_red_counts() -> None:
  """prev_candle slice: green group counts sessions after a green prior day.

  _SEQ prior sessions (preceding each countable day):
    idx 0 → close=100, open=100 → close >= open → green (used for idx 1)
    idx 1 → close=108, open=105 → green (used for idx 2)
    idx 2 → close=96,  open=100 → red   (used for idx 3)
    idx 3 → close=102, open=100 → green (used for idx 4)
    idx 4 → close=103, open=105 → red   (used for idx 5)
    idx 5 → close=118, open=120 → red   (used for idx 6)

  countable days and their prev_candle color:
    idx 1: green  → inside (105 in [90,110])   → inside group
    idx 2: green  → inside (100 in [95,115])   → inside group
    idx 3: red    → inside (100 in [80,112])   → inside group
    idx 4: green  → inside (105 in [79,113])   → inside group
    idx 5: red    → NOT inside (120 > 111)
    idx 6: red    → inside (118 in [115,125])  → inside group

  green group: idx 1,2,4 → 3 countable, 3 inside
  red group:   idx 3,5,6 → 3 countable, 2 inside (idx 5 not inside)
  """
  result = _stat().compute(make_candles(_SEQ))
  pc = result.instruments["NQ"]["daily"].slices["prev_candle"]

  green_grp = pc.groups["green"]
  red_grp = pc.groups["red"]

  # Green group: 3 countable days (prev is green for idx 1,2,4)
  # Each day appears once in the slice group — total_samples = 3
  assert green_grp.total_samples == 3
  # Find the inside_open/inside row in the green group
  green_inside_row = next(
    r for r in green_grp.results if r.condition == "inside_open" and r.outcome == "inside"
  )
  # All 3 days with green prior open inside → count=3, total=3
  assert green_inside_row.count == 3
  assert green_inside_row.total == 3

  # Red group: 3 countable days (prev is red for idx 3,5,6)
  assert red_grp.total_samples == 3
  red_inside_row = next(
    r for r in red_grp.results if r.condition == "inside_open" and r.outcome == "inside"
  )
  # idx 3 inside, idx 5 NOT inside, idx 6 inside → count=2, total=3
  assert red_inside_row.count == 2
  assert red_inside_row.total == 3


# ===========================================================================
# Baseline deterministic and embedded
# ===========================================================================

def _long_seq() -> pd.DataFrame:
  """~250 weekdays with frequent inside opens (small drift, wide prior ranges).

  Each day opens within a narrow band while the prior day had a wide range,
  ensuring many inside opens so the baseline has meaningful sample sizes.
  """
  dates: list[str] = []
  d = pd.Timestamp("2020-01-01", tz=_NY)
  while len(dates) < 250:
    if d.weekday() < 5:
      dates.append(d.strftime("%Y-%m-%d"))
    d += pd.Timedelta(days=1)

  days = []
  base = 100.0
  for i, date in enumerate(dates):
    # Wide range so the next day's open (small drift) is likely inside
    hi, lo = base + 15.0, base - 15.0
    # Small drift: open and close stay near base
    open_ = base + (1.0 if i % 2 == 0 else -1.0)
    close = base + (0.5 if i % 2 == 0 else -0.5)
    days.append({"date": date, "open": open_, "close": close, "high": hi, "low": lo})
    base += 0.5  # small drift so next open stays inside prior wide range

  return make_candles(days)


def test_baseline_deterministic() -> None:
  """Same seed produces identical baseline results across two calls."""
  stat = _stat()
  df = _long_seq()
  rows_a = stat.baseline(df, seed=7)
  rows_b = stat.baseline(df, seed=7)
  for a, b in zip(rows_a, rows_b):
    assert (a.condition, a.outcome) == (b.condition, b.outcome)
    assert a.probability == pytest.approx(b.probability)


def test_baseline_tier2_partitions() -> None:
  """Baseline tier-2 broke_high/broke_low/broke_both/contained partition inside_n."""
  rows = _stat().baseline(_long_seq(), seed=11)
  by_key = {(r.condition, r.outcome): r for r in rows}
  bh = by_key[("inside", "broke_high")]
  bl = by_key[("inside", "broke_low")]
  bb = by_key[("inside", "broke_both")]
  ct = by_key[("inside", "contained")]
  assert bh.total == bl.total == bb.total == ct.total
  assert bh.count + bl.count + bb.count + ct.count == bh.total


def test_baseline_embedded_in_compute() -> None:
  """After compute(), every countable row carries a positive baseline_n."""
  result = _stat().compute(_long_seq())
  for row in result.instruments["NQ"]["daily"].results:
    if row.total > 0:
      assert row.baseline_n > 0


# ===========================================================================
# Pending exclusion
# ===========================================================================

def test_pending_day_excluded_from_total_samples() -> None:
  """A truncated/early-close day does not appear in total_samples."""
  stat = _stat()
  base_df = make_candles(_SEQ)
  total_base = stat.compute(base_df).instruments["NQ"]["daily"].total_samples
  truncated = _make_truncated_day("2024-01-11")
  combined = (
    pd.concat([base_df, truncated], ignore_index=True)
    .sort_values("timestamp")
    .reset_index(drop=True)
  )
  total_with = stat.compute(combined).instruments["NQ"]["daily"].total_samples
  assert total_with == total_base == 7


def test_pending_day_absent_from_day_table() -> None:
  """build_day_table excludes the pending (truncated) day entirely."""
  day_table = _stat().build_day_table(_make_truncated_day("2024-01-11"))
  pending = pd.Timestamp("2024-01-11", tz=_NY).normalize()
  assert pending not in day_table.index


# ===========================================================================
# classify_samples()
# ===========================================================================


def test_classify_samples_exact_rows() -> None:
  """Exact (date, condition, outcome) samples for the 7-day ``_SEQ`` fixture.

  idx 0 (2024-01-02) has no prior day -> not countable -> no samples.
  idx 5 (2024-01-09) opens outside -> tier-1 ``outside`` sample only, no tier-2.
  All other countable days yield one tier-1 ``inside`` sample plus one tier-2
  breakout-direction sample.
  """
  stat = _stat()
  day_table = stat.build_day_table(make_candles(_SEQ))
  samples = stat.classify_samples(day_table)

  tier1 = {(s.date, s.outcome) for s in samples if s.condition == "inside_open"}
  tier2 = {(s.date, s.outcome) for s in samples if s.condition == "inside"}

  assert tier1 == {
    ("2024-01-03", "inside"),
    ("2024-01-04", "inside"),
    ("2024-01-05", "inside"),
    ("2024-01-08", "inside"),
    ("2024-01-09", "outside"),
    ("2024-01-10", "inside"),
  }
  assert tier2 == {
    ("2024-01-03", "broke_high"),
    ("2024-01-04", "broke_low"),
    ("2024-01-05", "broke_both"),
    ("2024-01-08", "contained"),
    ("2024-01-10", "contained"),
  }
  assert all(s.value is None for s in samples)
  assert not any(s.date == "2024-01-02" for s in samples)


def test_classify_samples_matches_compute_rows_counts() -> None:
  """For every row from ``compute_rows``, matching samples reproduce count/total.

  Tier-1's ``total`` is reproduced by counting all ``inside_open`` samples
  (both ``inside`` and its ``outside`` complement); tier-2 rows have no
  complement outcome, so their ``total`` is reproduced by counting all
  ``inside`` samples of any of the four breakout outcomes.
  """
  stat = _stat()
  day_table = stat.build_day_table(make_candles(_SEQ))
  samples = stat.classify_samples(day_table)
  rows = stat.compute_rows(day_table)

  for row in rows:
    matching = [s for s in samples if s.condition == row.condition and s.outcome == row.outcome]
    assert len(matching) == row.count, (row.condition, row.outcome, len(matching), row.count)

  inside_open_samples = [s for s in samples if s.condition == "inside_open"]
  inside_open_row = _row(stat.compute(make_candles(_SEQ)), "inside_open", "inside")
  assert len(inside_open_samples) == inside_open_row.total

  inside_samples = [s for s in samples if s.condition == "inside"]
  inside_row = _row(stat.compute(make_candles(_SEQ)), "inside", "contained")
  assert len(inside_samples) == inside_row.total


def test_classify_samples_empty_day_table_returns_empty_list() -> None:
  """Empty day_table -> classify_samples returns []."""
  stat = _stat()
  day_table = stat.build_day_table(_empty_df())
  assert stat.classify_samples(day_table) == []


def test_classify_samples_pending_day_excluded() -> None:
  """A truncated (unresolved) day never appears in classify_samples output."""
  stat = _stat()
  base_df = make_candles(_SEQ)
  truncated = _make_truncated_day("2024-01-11")
  combined = (
    pd.concat([base_df, truncated], ignore_index=True)
    .sort_values("timestamp")
    .reset_index(drop=True)
  )
  day_table = stat.build_day_table(combined)
  samples = stat.classify_samples(day_table)
  assert all(s.date != "2024-01-11" for s in samples)


def test_classify_samples_first_day_excluded() -> None:
  """The anchor day (no prior resolved day) contributes no samples at all."""
  stat = _stat()
  day_table = stat.build_day_table(make_candles(_SEQ))
  samples = stat.classify_samples(day_table)
  assert not any(s.date == "2024-01-02" for s in samples)


# ===========================================================================
# Edge cases & validation
# ===========================================================================

def _empty_df() -> pd.DataFrame:
  df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
  df["timestamp"] = pd.array([], dtype="datetime64[ns, America/New_York]")
  return df


def test_empty_dataframe_no_crash() -> None:
  """Empty input → zero samples, empty data_range, all rows zeroed."""
  result = _stat().compute(_empty_df())
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 0
  assert tf.data_range == []
  for row in tf.results:
    assert row.count == 0
    assert row.total == 0
    assert row.probability == pytest.approx(0.0)


def test_single_resolved_day_no_countable_rows() -> None:
  """One resolved session → no prior day → all totals zero, no crash."""
  days = [{"date": "2024-03-01", "open": 100.0, "close": 110.0, "high": 115.0, "low": 95.0}]
  result = _stat().compute(make_candles(days))
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 1
  for row in tf.results:
    assert row.total == 0


# ===========================================================================
# i18n
# ===========================================================================

def test_i18n_title_and_definition() -> None:
  """title and definition both have non-empty en and fr strings."""
  result = _stat().compute(_empty_df())
  assert result.title.en != "" and result.title.fr != ""
  assert result.definition.en != "" and result.definition.fr != ""


def test_i18n_labels_have_en_and_fr() -> None:
  """Every condition and outcome label has non-empty en and fr; correct keys."""
  result = _stat().compute(_empty_df())
  for mapping in (result.labels.conditions, result.labels.outcomes):
    for key, i18n in mapping.items():
      assert i18n.en != "", f"{key}.en empty"
      assert i18n.fr != "", f"{key}.fr empty"
  assert set(result.labels.conditions) == {"inside_open", "inside"}
  assert set(result.labels.outcomes) == {
    "inside", "broke_high", "broke_low", "broke_both", "contained"
  }


# ===========================================================================
# stat_name and write_results round-trip
# ===========================================================================

def test_stat_name() -> None:
  assert _stat().compute(_empty_df()).stat_name == "inside_bars"


def test_write_results_round_trip(tmp_path: Path) -> None:
  """write_results produces inside_bars.json that re-validates correctly."""
  result = _stat().compute(make_candles(_SEQ))
  written = write_results(result, results_dir=tmp_path)
  assert written.name == "inside_bars.json"

  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)
  assert validated.instruments["NQ"]["daily"].total_samples == 7


def test_write_results_french_accent_literal(tmp_path: Path) -> None:
  """French text is stored as literal UTF-8 characters, not escaped unicode."""
  result = _stat().compute(_empty_df())
  written = write_results(result, results_dir=tmp_path)
  raw = written.read_text(encoding="utf-8")
  # The French definition contains "précédent" (with é) or "fréquence" (with é).
  assert "é" in raw
  assert "\\u00e9" not in raw
