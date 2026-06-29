"""Tests for stats.ict_opening_retracement.standard.

All data is synthetic — no real market files required.
Expected values are hand-calculated before each assertion.

Matrix framing:
  condition = session open direction relative to the midnight reference level
              (opened_above: session_open > reference_level;
               opened_below: session_open < reference_level)
  outcome   = whether intraday RTH price retraced to the reference level
              (opened_above → day_low <= reference_level counts as retraced;
               opened_below → day_high >= reference_level counts as retraced;
               touching exactly counts as retraced)

Days opening exactly at the reference level are excluded from every denominator
(pending-sample discipline) but are still counted in total_samples.
Days with no midnight reference bar get reference_level=NaN and are also
excluded from directional totals but counted in total_samples.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.ict_opening_retracement.standard import IctOpeningRetracement

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

_RTH_START = 570  # 09:30
_RTH_LAST = 974   # 16:14 (last bar before 16:15); resolved needs last_mod >= 960


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

def _make_rth_bars(
  date: str,
  session_open: float,
  session_close: float,
  day_high: float | None = None,
  day_low: float | None = None,
) -> pd.DataFrame:
  """Build one full RTH trading day of 1-min OHLCV bars (09:30–16:14).

  The 09:30 bar sets session_open as its open price and records day_high as
  that bar's high (so day_high is always the maximum of all bar highs).
  The 16:14 bar carries session_close as its close and records day_low as its
  low (so day_low is always the minimum of all bar lows).
  All intermediate bars sit inside (day_low, day_high) using a neutral mid-
  price so they never violate the intended extremes.

  When day_high / day_low are omitted, they default to:
    day_high = max(session_open, session_close) + 0.25
    day_low  = min(session_open, session_close) - 0.25
  """
  if day_high is None:
    day_high = max(session_open, session_close) + 0.25
  if day_low is None:
    day_low = min(session_open, session_close) - 0.25

  assert day_high >= max(session_open, session_close), (
    f"day_high ({day_high}) < max(open, close) ({max(session_open, session_close)})"
  )
  assert day_low <= min(session_open, session_close), (
    f"day_low ({day_low}) > min(open, close) ({min(session_open, session_close)})"
  )

  base = pd.Timestamp(date, tz=_NY)
  mid = (day_high + day_low) / 2.0
  neutral_high = mid + 0.10
  neutral_low = mid - 0.10

  records = []
  for mod in range(_RTH_START, _RTH_LAST + 1):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    is_first = mod == _RTH_START
    is_last = mod == _RTH_LAST
    o = session_open if is_first else mid
    c = session_close if is_last else mid
    bar_high = day_high if is_first else (max(neutral_high, c) if is_last else neutral_high)
    bar_low = (min(neutral_low, o) if is_first else (day_low if is_last else neutral_low))
    records.append({
      "timestamp": ts,
      "open": o,
      "high": bar_high,
      "low": bar_low,
      "close": c,
      "volume": 1000,
    })
  return pd.DataFrame(records)


def _make_day(
  date: str,
  reference_level: float,
  session_open: float,
  session_close: float,
  day_high: float | None = None,
  day_low: float | None = None,
  ref_high: float | None = None,
) -> pd.DataFrame:
  """Build one full day: a 00:00 midnight reference bar plus RTH bars (09:30–16:14).

  The midnight bar's open equals reference_level (the default ICT reference price
  when reference_price='open'). ref_high overrides the midnight bar's high, which
  is needed when testing reference_price='high'.

  Both the midnight bar and all RTH bars share the same calendar date, so
  timestamp.normalize() aligns them to the same index key in build_day_table.
  """
  ts_midnight = pd.Timestamp(date, tz=_NY).normalize()
  midnight_high = ref_high if ref_high is not None else reference_level + 0.25
  midnight = pd.DataFrame([{
    "timestamp": ts_midnight,
    "open": reference_level,
    "high": midnight_high,
    "low": reference_level - 0.25,
    "close": reference_level,
    "volume": 100,
  }])
  rth = _make_rth_bars(date, session_open, session_close, day_high, day_low)
  return pd.concat([midnight, rth], ignore_index=True)


def _make_day_no_ref(
  date: str,
  session_open: float,
  session_close: float,
  day_high: float | None = None,
  day_low: float | None = None,
) -> pd.DataFrame:
  """Build a resolved RTH day with NO midnight reference bar.

  The day is fully resolved (09:30–16:14 bars present) but has no 00:00 bar,
  so build_day_table will left-join reference_level=NaN for this date.
  """
  return _make_rth_bars(date, session_open, session_close, day_high, day_low)


def make_candles(days: list[dict]) -> pd.DataFrame:
  """Concatenate per-day specs into a single sorted 1-min OHLCV DataFrame.

  Each dict must contain 'date', 'session_open', 'session_close'. Additionally:
    'reference_level': midnight bar's open (required unless no_ref=True)
    'no_ref': if True, omit the midnight bar entirely
    'day_high', 'day_low': RTH extremes (optional)
    'ref_high': midnight bar's high override (for reference_price='high' tests)
  """
  frames = []
  for d in days:
    if d.get("no_ref"):
      frames.append(_make_day_no_ref(
        d["date"],
        d["session_open"],
        d["session_close"],
        d.get("day_high"),
        d.get("day_low"),
      ))
    else:
      frames.append(_make_day(
        d["date"],
        d["reference_level"],
        d["session_open"],
        d["session_close"],
        d.get("day_high"),
        d.get("day_low"),
        d.get("ref_high"),
      ))
  df = pd.concat(frames, ignore_index=True)
  return df.sort_values("timestamp").reset_index(drop=True)


def _empty_df() -> pd.DataFrame:
  df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
  df["timestamp"] = pd.array([], dtype="datetime64[ns, America/New_York]")
  return df


def _stat(reference_price: str = "open") -> IctOpeningRetracement:
  return IctOpeningRetracement(
    instrument="NQ",
    config=_TEST_CONFIG,
    reference_price=reference_price,
  )


def _row(result: StatRunResult, condition: str, outcome: str) -> StatResultRow:
  for r in result.instruments["NQ"]["daily"].results:
    if r.condition == condition and r.outcome == outcome:
      return r
  raise KeyError((condition, outcome))


# ===========================================================================
# 1. Core matrix
#
# Day-by-day derivation (ref_level = 100 for all days):
#
#   idx  date        session_open  direction     day_low  day_high  retraced?
#    0   2024-01-02  105.00        above (+5)    102.00   112.00    102 <= 100? NO
#    1   2024-01-03  108.00        above (+8)     99.00   110.00     99 <= 100? YES
#    2   2024-01-04   95.00        below (-5)     93.00    99.00     99 >= 100? NO
#    3   2024-01-05   92.00        below (-8)     90.00   101.00    101 >= 100? YES
#    4   2024-01-08  105.00        above (+5)    100.00   107.00    100 <= 100? YES (exact touch)
#    5   2024-01-09   92.00        below (-8)     90.00   100.00    100 >= 100? YES (exact touch)
#    6   2024-01-10  103.00        above (+3)    101.00   105.00    101 <= 100? NO
#    7   2024-01-11  100.00        no dir (=0)    99.00   106.00    — excluded from denominators
#
# Matrix:
#   opened_above days: 0, 1, 4, 6  → total = 4
#     retraced:     1, 4            → count = 2, P = 2/4 = 0.5
#     not_retraced: 0, 6            → count = 2, P = 2/4 = 0.5
#
#   opened_below days: 2, 3, 5  → total = 3
#     retraced:     3, 5            → count = 2, P = 2/3 ≈ 0.6667
#     not_retraced: 2               → count = 1, P = 1/3 ≈ 0.3333
#
# total_samples = 8 (all 8 resolved days, including idx 7 which has zero direction)
# ===========================================================================

_SEQ = [
  # idx 0: opened_above (+5), day_low=102 > 100 → NOT retraced
  {
    "date": "2024-01-02",
    "reference_level": 100.00,
    "session_open": 105.00,
    "session_close": 110.00,
    "day_high": 112.00,
    "day_low": 102.00,
  },
  # idx 1: opened_above (+8), day_low=99 <= 100 → retraced
  {
    "date": "2024-01-03",
    "reference_level": 100.00,
    "session_open": 108.00,
    "session_close": 105.00,
    "day_high": 110.00,
    "day_low": 99.00,
  },
  # idx 2: opened_below (-5), day_high=99 < 100 → NOT retraced
  {
    "date": "2024-01-04",
    "reference_level": 100.00,
    "session_open": 95.00,
    "session_close": 98.00,
    "day_high": 99.00,
    "day_low": 93.00,
  },
  # idx 3: opened_below (-8), day_high=101 >= 100 → retraced
  {
    "date": "2024-01-05",
    "reference_level": 100.00,
    "session_open": 92.00,
    "session_close": 95.00,
    "day_high": 101.00,
    "day_low": 90.00,
  },
  # idx 4: opened_above (+5), day_low=100 <= 100 → retraced (exact touch)
  {
    "date": "2024-01-08",
    "reference_level": 100.00,
    "session_open": 105.00,
    "session_close": 102.00,
    "day_high": 107.00,
    "day_low": 100.00,
  },
  # idx 5: opened_below (-8), day_high=100 >= 100 → retraced (exact touch)
  {
    "date": "2024-01-09",
    "reference_level": 100.00,
    "session_open": 92.00,
    "session_close": 95.00,
    "day_high": 100.00,
    "day_low": 90.00,
  },
  # idx 6: opened_above (+3), day_low=101 > 100 → NOT retraced
  {
    "date": "2024-01-10",
    "reference_level": 100.00,
    "session_open": 103.00,
    "session_close": 102.00,
    "day_high": 105.00,
    "day_low": 101.00,
  },
  # idx 7: session_open == ref_level → gap_pts=0 → no direction, excluded from denominators
  {
    "date": "2024-01-11",
    "reference_level": 100.00,
    "session_open": 100.00,
    "session_close": 105.00,
    "day_high": 106.00,
    "day_low": 99.00,
  },
]


def test_total_samples_counts_all_resolved_days() -> None:
  """total_samples = 8: all 8 resolved sessions including the zero-direction day."""
  result = _stat().compute(make_candles(_SEQ))
  assert result.instruments["NQ"]["daily"].total_samples == 8


def test_opened_above_retraced_count_and_probability() -> None:
  """opened_above retraced: 2 of 4 above days → P=2/4=0.5."""
  # above days: idx 0(not), 1(retraced), 4(retraced), 6(not) → total=4, retraced=2
  result = _stat().compute(make_candles(_SEQ))
  r = _row(result, "opened_above", "retraced")
  assert r.total == 4
  assert r.count == 2
  assert r.probability == pytest.approx(2 / 4)


def test_opened_above_not_retraced_count_and_probability() -> None:
  """opened_above not_retraced: 2 of 4 above days → P=2/4=0.5."""
  result = _stat().compute(make_candles(_SEQ))
  r = _row(result, "opened_above", "not_retraced")
  assert r.total == 4
  assert r.count == 2
  assert r.probability == pytest.approx(2 / 4)


def test_opened_below_retraced_count_and_probability() -> None:
  """opened_below retraced: 2 of 3 below days → P=2/3≈0.6667."""
  # below days: idx 2(not), 3(retraced), 5(retraced) → total=3, retraced=2
  result = _stat().compute(make_candles(_SEQ))
  r = _row(result, "opened_below", "retraced")
  assert r.total == 3
  assert r.count == 2
  assert r.probability == pytest.approx(2 / 3)


def test_opened_below_not_retraced_count_and_probability() -> None:
  """opened_below not_retraced: 1 of 3 below days → P=1/3≈0.3333."""
  result = _stat().compute(make_candles(_SEQ))
  r = _row(result, "opened_below", "not_retraced")
  assert r.total == 3
  assert r.count == 1
  assert r.probability == pytest.approx(1 / 3)


def test_outcomes_partition_each_condition() -> None:
  """retraced.count + not_retraced.count == total for each direction."""
  result = _stat().compute(make_candles(_SEQ))
  for cond in ("opened_above", "opened_below"):
    retraced = _row(result, cond, "retraced")
    not_retraced = _row(result, cond, "not_retraced")
    assert retraced.count + not_retraced.count == retraced.total == not_retraced.total


def test_probabilities_sum_to_one_per_condition() -> None:
  """P(retraced) + P(not_retraced) == 1.0 for each direction with data."""
  result = _stat().compute(make_candles(_SEQ))
  for cond in ("opened_above", "opened_below"):
    p_ret = _row(result, cond, "retraced").probability
    p_not = _row(result, cond, "not_retraced").probability
    assert p_ret + p_not == pytest.approx(1.0)


def test_four_rows_only() -> None:
  """Exactly four rows: the 2x2 (direction × retraced/not_retraced) matrix."""
  result = _stat().compute(make_candles(_SEQ))
  rows = result.instruments["NQ"]["daily"].results
  assert {(r.condition, r.outcome) for r in rows} == {
    ("opened_above", "retraced"),
    ("opened_above", "not_retraced"),
    ("opened_below", "retraced"),
    ("opened_below", "not_retraced"),
  }


def test_data_range_spans_first_to_last_resolved_day() -> None:
  """data_range reflects the first and last resolved session dates."""
  result = _stat().compute(make_candles(_SEQ))
  assert result.instruments["NQ"]["daily"].data_range == ["2024-01-02", "2024-01-11"]


def test_condition_totals_equal_countable_days() -> None:
  """Sum of all condition totals = 7 = total_samples - 1 (zero-direction day excluded)."""
  # 8 resolved days; idx 7 has zero direction → 7 countable (4 above + 3 below)
  result = _stat().compute(make_candles(_SEQ))
  above_total = _row(result, "opened_above", "retraced").total
  below_total = _row(result, "opened_below", "retraced").total
  assert above_total + below_total == 8 - 1


# ===========================================================================
# 2. Exact-touch retracement semantics
#
# Opened above → retraced uses <=  (day_low touching level counts as retraced)
# Opened below → retraced uses >=  (day_high touching level counts as retraced)
#
# Minimal 1-day sequences isolate each exact-touch case:
#
#   above exact touch: ref_level=100, session_open=104, day_low=100  → 100<=100 YES
#   above clear miss:  ref_level=100, session_open=104, day_low=100.25 → 100.25>100 NO
#   below exact touch: ref_level=100, session_open=96,  day_high=100 → 100>=100 YES
#   below clear miss:  ref_level=100, session_open=96,  day_high=99.75 → 99.75<100 NO
# ===========================================================================

_ABOVE_EXACT_TOUCH = [
  {
    "date": "2024-02-01",
    "reference_level": 100.00,
    "session_open": 104.00,
    "session_close": 102.00,
    "day_high": 105.00,
    "day_low": 100.00,   # touches ref_level exactly
  },
]

_ABOVE_CLEAR_MISS = [
  {
    "date": "2024-02-01",
    "reference_level": 100.00,
    "session_open": 104.00,
    "session_close": 102.00,
    "day_high": 105.00,
    "day_low": 100.25,   # strictly above ref_level → not retraced
  },
]

_BELOW_EXACT_TOUCH = [
  {
    "date": "2024-02-01",
    "reference_level": 100.00,
    "session_open": 96.00,
    "session_close": 98.00,
    "day_high": 100.00,  # touches ref_level exactly
    "day_low": 95.00,
  },
]

_BELOW_CLEAR_MISS = [
  {
    "date": "2024-02-01",
    "reference_level": 100.00,
    "session_open": 96.00,
    "session_close": 98.00,
    "day_high": 99.75,   # strictly below ref_level → not retraced
    "day_low": 95.00,
  },
]


def test_exact_touch_above_counts_as_retraced() -> None:
  """day_low == reference_level for an above-open day → retraced (<=, touch counts)."""
  # ref_level=100, session_open=104 → opened_above; day_low=100 → 100<=100 → retraced
  result = _stat().compute(make_candles(_ABOVE_EXACT_TOUCH))
  r = _row(result, "opened_above", "retraced")
  assert r.count == 1
  assert r.total == 1
  assert r.probability == pytest.approx(1.0)


def test_one_tick_above_ref_level_is_not_retraced() -> None:
  """day_low > reference_level by 0.25 → not retraced (strictly above)."""
  # ref_level=100, session_open=104 → opened_above; day_low=100.25 → 100.25>100 → no
  result = _stat().compute(make_candles(_ABOVE_CLEAR_MISS))
  r = _row(result, "opened_above", "retraced")
  assert r.count == 0
  assert r.probability == pytest.approx(0.0)
  assert _row(result, "opened_above", "not_retraced").count == 1


def test_exact_touch_below_counts_as_retraced() -> None:
  """day_high == reference_level for a below-open day → retraced (>=, touch counts)."""
  # ref_level=100, session_open=96 → opened_below; day_high=100 → 100>=100 → retraced
  result = _stat().compute(make_candles(_BELOW_EXACT_TOUCH))
  r = _row(result, "opened_below", "retraced")
  assert r.count == 1
  assert r.total == 1
  assert r.probability == pytest.approx(1.0)


def test_one_tick_below_ref_level_is_not_retraced() -> None:
  """day_high < reference_level by 0.25 → not retraced (strictly below)."""
  # ref_level=100, session_open=96 → opened_below; day_high=99.75 → 99.75<100 → no
  result = _stat().compute(make_candles(_BELOW_CLEAR_MISS))
  r = _row(result, "opened_below", "retraced")
  assert r.count == 0
  assert _row(result, "opened_below", "not_retraced").count == 1


# ===========================================================================
# 3. Pending discipline / countability
#
# Two exclusion mechanisms:
#   A. Zero direction (session_open == reference_level): gap_size_pts=NaN →
#      excluded from every condition denominator, but counted in total_samples.
#   B. No reference bar for the calendar date: reference_level=NaN →
#      gap_size_pts=NaN → excluded from every condition denominator, but
#      counted in total_samples (resolved RTH day present).
#
# Scenario for B:
#   Day A: ref_level=100, session_open=105, day_high=110, day_low=102  → above, not retraced
#   Day B: NO midnight bar  → ref_level=NaN → not countable, in total_samples
#   Day C: ref_level=100, session_open=108, day_high=110, day_low=99   → above, retraced
#
#   total_samples=3; opened_above total=2 (only A, C); opened_below total=0
#   opened_above retraced=1 (C), not_retraced=1 (A), P=0.5
# ===========================================================================

_NO_REF_DAYS = [
  # Day A: has midnight bar, opened_above (+5), day_low=102>100 → NOT retraced
  {
    "date": "2024-02-01",
    "reference_level": 100.00,
    "session_open": 105.00,
    "session_close": 108.00,
    "day_high": 110.00,
    "day_low": 102.00,
  },
  # Day B: NO midnight bar → ref_level=NaN → excluded from directional totals
  {
    "date": "2024-02-02",
    "no_ref": True,
    "session_open": 98.00,
    "session_close": 102.00,
    "day_high": 103.00,
    "day_low": 97.00,
  },
  # Day C: has midnight bar, opened_above (+8), day_low=99<=100 → retraced
  {
    "date": "2024-02-05",
    "reference_level": 100.00,
    "session_open": 108.00,
    "session_close": 105.00,
    "day_high": 110.00,
    "day_low": 99.00,
  },
]


def test_zero_direction_excluded_from_denominators() -> None:
  """session_open == reference_level → zero direction, excluded from all condition totals."""
  # idx 7 in _SEQ: session_open=100 == ref_level=100 → not countable
  # opened_above total=4, opened_below total=3, sum=7 = total_samples(8) - 1
  result = _stat().compute(make_candles(_SEQ))
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 8
  above_total = _row(result, "opened_above", "retraced").total
  below_total = _row(result, "opened_below", "retraced").total
  assert above_total + below_total == 7  # 8 resolved days - 1 zero-direction day


def test_no_reference_bar_excluded_from_directional_totals() -> None:
  """A resolved day with no 00:00 bar → reference_level=NaN → excluded from conditions."""
  # Day B has no midnight bar → not countable
  # opened_above: only A (not retraced) and C (retraced) → total=2
  # opened_below: 0 (Day B would be opened_below if it had a ref, but it doesn't)
  result = _stat().compute(make_candles(_NO_REF_DAYS))
  assert _row(result, "opened_above", "retraced").total == 2
  assert _row(result, "opened_below", "retraced").total == 0


def test_total_samples_includes_no_reference_bar_day() -> None:
  """total_samples counts Day B (no midnight bar) — resolved RTH day regardless."""
  # 3 fully resolved RTH sessions → total_samples=3 even though Day B has no ref
  result = _stat().compute(make_candles(_NO_REF_DAYS))
  assert result.instruments["NQ"]["daily"].total_samples == 3


def test_total_samples_includes_zero_direction_day() -> None:
  """total_samples counts idx 7 (zero direction) — resolved RTH day regardless."""
  # _SEQ has 8 resolved days including idx 7 (session_open == ref_level)
  result = _stat().compute(make_candles(_SEQ))
  assert result.instruments["NQ"]["daily"].total_samples == 8


def test_no_ref_bar_day_table_reference_level_is_nan() -> None:
  """build_day_table: the no-ref-bar day's reference_level column is NaN."""
  stat = _stat()
  day_table = stat.build_day_table(make_candles(_NO_REF_DAYS))
  date_b = pd.Timestamp("2024-02-02", tz=_NY).normalize()
  assert date_b in day_table.index
  assert pd.isna(day_table.loc[date_b, "reference_level"])


def test_no_ref_bar_day_table_gap_size_pts_is_nan() -> None:
  """build_day_table: the no-ref-bar day has gap_size_pts=NaN (not countable)."""
  stat = _stat()
  day_table = stat.build_day_table(make_candles(_NO_REF_DAYS))
  date_b = pd.Timestamp("2024-02-02", tz=_NY).normalize()
  assert pd.isna(day_table.loc[date_b, "gap_size_pts"])


def test_no_ref_day_correct_counts_above() -> None:
  """opened_above: only Day A (not retraced) and Day C (retraced) → total=2, P=0.5."""
  # A: day_low=102 > ref_level=100 → not retraced
  # C: day_low=99  <= ref_level=100 → retraced
  result = _stat().compute(make_candles(_NO_REF_DAYS))
  r_ret = _row(result, "opened_above", "retraced")
  r_not = _row(result, "opened_above", "not_retraced")
  assert r_ret.count == 1
  assert r_not.count == 1
  assert r_ret.probability == pytest.approx(0.5)


# ===========================================================================
# 4. reference_price parameter
#
# Scenario: midnight bar has open=100, high=110. RTH: session_open=105,
# day_high=115, day_low=103.
#
# With reference_price='open':
#   ref_level = 100 (midnight bar open)
#   session_open=105 > 100  → opened_above
#   day_low=103 <= 100?    → NO  → not_retraced
#
# With reference_price='high':
#   ref_level = 110 (midnight bar high)
#   session_open=105 < 110  → opened_below
#   day_high=115 >= 110?   → YES → retraced
#
# OHLC check: max(105,108)=108<=115 ✓, min(105,108)=105>=103 ✓
# ===========================================================================

_REF_PRICE_DAY = [
  {
    "date": "2024-03-01",
    "reference_level": 100.00,  # midnight bar open
    "ref_high": 110.00,          # midnight bar high (differs from open)
    "session_open": 105.00,
    "session_close": 108.00,
    "day_high": 115.00,
    "day_low": 103.00,
  },
]


def test_reference_price_open_direction_and_outcome() -> None:
  """reference_price='open': ref_level=100, session_open=105 → above, not retraced."""
  # ref_level=100 (midnight open), session_open=105>100 → opened_above
  # day_low=103 > 100 → NOT retraced
  result = _stat(reference_price="open").compute(make_candles(_REF_PRICE_DAY))
  r_above_ret = _row(result, "opened_above", "retraced")
  r_above_not = _row(result, "opened_above", "not_retraced")
  assert r_above_ret.count == 0
  assert r_above_not.count == 1
  # opened_below has no days when using open as reference
  assert _row(result, "opened_below", "retraced").total == 0


def test_reference_price_high_changes_direction_and_outcome() -> None:
  """reference_price='high': ref_level=110, session_open=105 → below, retraced."""
  # ref_level=110 (midnight high), session_open=105<110 → opened_below
  # day_high=115 >= 110 → retraced
  result = _stat(reference_price="high").compute(make_candles(_REF_PRICE_DAY))
  r_below_ret = _row(result, "opened_below", "retraced")
  assert r_below_ret.count == 1
  assert r_below_ret.total == 1
  assert r_below_ret.probability == pytest.approx(1.0)
  # opened_above has no days when using high as reference
  assert _row(result, "opened_above", "retraced").total == 0


# ===========================================================================
# 5. Baseline: determinism, pooled-count preservation, baseline_n
#
# _SEQ: 7 countable days (idx 0–6; idx 7 is zero-direction → not countable).
# Retraced in countable days: idx 1(YES), idx 3(YES), idx 4(YES), idx 5(YES) → 4 of 7.
# Baseline permutes the 'retraced' column over those 7 days, preserving count=4.
# ===========================================================================

def test_baseline_deterministic_same_seed() -> None:
  """Two calls with the same seed return identical baseline row probabilities."""
  stat = _stat()
  df = make_candles(_SEQ)
  rows_a = stat.baseline(df, seed=7)
  rows_b = stat.baseline(df, seed=7)
  assert len(rows_a) == len(rows_b)
  for a, b in zip(rows_a, rows_b):
    assert (a.condition, a.outcome) == (b.condition, b.outcome)
    assert a.probability == pytest.approx(b.probability)


def test_baseline_different_seeds_differ() -> None:
  """Two calls with different seeds produce different baseline probabilities."""
  stat = _stat()
  df = make_candles(_SEQ)
  rows_42 = stat.baseline(df, seed=42)
  rows_0 = stat.baseline(df, seed=0)
  all_same = all(
    abs(a.probability - b.probability) < 1e-9
    for a, b in zip(rows_42, rows_0)
  )
  assert not all_same


def test_baseline_preserves_pooled_retracement_count() -> None:
  """The permutation baseline preserves the total retraced count across all countable days.

  Real retraced: idx 1, 3, 4, 5 → 4 of 7 countable days.
  Baseline permutes 'retraced' across those 7 days →
  sum(retraced_above.count + retraced_below.count) == 4 for every seed.
  """
  stat = _stat()
  df = make_candles(_SEQ)
  day_table = stat.build_day_table(df)

  # Verify the real pooled count first
  real_ret_above = _row(_stat().compute(df), "opened_above", "retraced").count
  real_ret_below = _row(_stat().compute(df), "opened_below", "retraced").count
  assert real_ret_above + real_ret_below == 4  # idx 1, 3, 4, 5

  for seed in (0, 1, 42, 123):
    bl_rows = stat.baseline_rows(day_table, seed=seed)
    bl_map = {(r.condition, r.outcome): r for r in bl_rows}
    bl_total = (
      bl_map[("opened_above", "retraced")].count
      + bl_map[("opened_below", "retraced")].count
    )
    assert bl_total == 4, f"seed={seed}: baseline retraced={bl_total}, expected 4"


def test_baseline_n_equals_condition_total() -> None:
  """After compute(), each row's baseline_n equals the condition's real total."""
  result = _stat().compute(make_candles(_SEQ))
  for cond in ("opened_above", "opened_below"):
    retraced = _row(result, cond, "retraced")
    not_retraced = _row(result, cond, "not_retraced")
    assert retraced.baseline_n == retraced.total
    assert not_retraced.baseline_n == not_retraced.total


def test_baseline_embedded_in_compute() -> None:
  """After compute(), every row with data has a positive baseline_n."""
  result = _stat().compute(make_candles(_SEQ))
  for row in result.instruments["NQ"]["daily"].results:
    if row.total > 0:
      assert row.baseline_n > 0


def test_compute_twice_same_data_identical_results() -> None:
  """Running compute twice on the same data yields identical results (determinism)."""
  stat = _stat()
  df = make_candles(_SEQ)
  res_a = stat.compute(df, seed=42)
  res_b = stat.compute(df, seed=42)
  rows_a = res_a.instruments["NQ"]["daily"].results
  rows_b = res_b.instruments["NQ"]["daily"].results
  for a, b in zip(rows_a, rows_b):
    assert a.condition == b.condition
    assert a.outcome == b.outcome
    assert a.count == b.count
    assert a.probability == pytest.approx(b.probability)
    assert a.baseline_prob == pytest.approx(b.baseline_prob)


# ===========================================================================
# 6. Slices present and correct
#
# IctOpeningRetracement declares slices: weekday, close, prev_candle,
# size_pts (SizeBucket on gap_size_pts), size_pct (SizeBucket on gap_size_pct).
#
# Close slicer groups by session_green:
#   session_green = (session_close >= session_open)
#   idx 0: close=110 >= open=105 → green
#   idx 1: close=105 <  open=108 → red
#   idx 2: close=98  >= open=95  → green
#   idx 3: close=95  >= open=92  → green
#   idx 4: close=102 <  open=105 → red
#   idx 5: close=95  >= open=92  → green
#   idx 6: close=102 <  open=103 → red
#   idx 7: close=105 >= open=100 → green
#
#   Green total_samples = 5 (idx 0, 2, 3, 5, 7)
#   Red   total_samples = 3 (idx 1, 4, 6)
#
#   Green countable (gap_size_pts not NaN; idx 7 excluded — zero direction):
#     idx 0: opened_above, not_retraced
#     idx 2: opened_below, not_retraced
#     idx 3: opened_below, retraced
#     idx 5: opened_below, retraced
#     → green opened_above: total=1, retraced=0, P=0.0
#     → green opened_below: total=3, retraced=2, P=2/3
#
#   Red countable:
#     idx 1: opened_above, retraced
#     idx 4: opened_above, retraced (exact touch)
#     idx 6: opened_above, not_retraced
#     → red opened_above: total=3, retraced=2, P=2/3
#     → red opened_below: total=0
# ===========================================================================

def test_slices_keys_present() -> None:
  """IctOpeningRetracement result must carry all five declared slice dimensions."""
  result = _stat().compute(make_candles(_SEQ))
  slices = result.instruments["NQ"]["daily"].slices
  assert set(slices.keys()) == {"weekday", "close", "prev_candle", "size_pts", "size_pct"}


def test_close_slice_has_green_and_red_groups() -> None:
  """The close slice produces green and red groups."""
  result = _stat().compute(make_candles(_SEQ))
  groups = result.instruments["NQ"]["daily"].slices["close"].groups
  assert "green" in groups
  assert "red" in groups


def test_close_slice_group_total_samples() -> None:
  """Green close group: 5 days (idx 0,2,3,5,7); red close group: 3 days (idx 1,4,6)."""
  result = _stat().compute(make_candles(_SEQ))
  groups = result.instruments["NQ"]["daily"].slices["close"].groups
  assert groups["green"].total_samples == 5
  assert groups["red"].total_samples == 3


def test_close_slice_green_opened_above_counts() -> None:
  """Green close group — opened_above: only idx 0 (not retraced) → total=1, P=0.0."""
  # Green countable above days: idx 0 only (day_low=102>100, not retraced)
  result = _stat().compute(make_candles(_SEQ))
  groups = result.instruments["NQ"]["daily"].slices["close"].groups
  green_results = groups["green"].results
  g_above_ret = next(
    r for r in green_results if r.condition == "opened_above" and r.outcome == "retraced"
  )
  g_above_not = next(
    r for r in green_results if r.condition == "opened_above" and r.outcome == "not_retraced"
  )
  assert g_above_ret.total == 1
  assert g_above_ret.count == 0
  assert g_above_ret.probability == pytest.approx(0.0)
  assert g_above_not.count == 1


def test_close_slice_green_opened_below_counts() -> None:
  """Green close group — opened_below: idx 2(not), 3(ret), 5(ret) → total=3, P=2/3."""
  result = _stat().compute(make_candles(_SEQ))
  groups = result.instruments["NQ"]["daily"].slices["close"].groups
  green_results = groups["green"].results
  g_below_ret = next(
    r for r in green_results if r.condition == "opened_below" and r.outcome == "retraced"
  )
  assert g_below_ret.total == 3
  assert g_below_ret.count == 2
  assert g_below_ret.probability == pytest.approx(2 / 3)


def test_close_slice_red_opened_above_counts() -> None:
  """Red close group — opened_above: idx 1(ret), 4(ret), 6(not) → total=3, P=2/3."""
  result = _stat().compute(make_candles(_SEQ))
  groups = result.instruments["NQ"]["daily"].slices["close"].groups
  red_results = groups["red"].results
  r_above_ret = next(
    r for r in red_results if r.condition == "opened_above" and r.outcome == "retraced"
  )
  assert r_above_ret.total == 3
  assert r_above_ret.count == 2
  assert r_above_ret.probability == pytest.approx(2 / 3)


def test_close_slice_red_opened_below_empty() -> None:
  """Red close group — opened_below: no red days opened below → total=0."""
  # All 3 red days (idx 1, 4, 6) opened above → opened_below total=0 in red group
  result = _stat().compute(make_candles(_SEQ))
  groups = result.instruments["NQ"]["daily"].slices["close"].groups
  red_results = groups["red"].results
  r_below_ret = next(
    r for r in red_results if r.condition == "opened_below" and r.outcome == "retraced"
  )
  assert r_below_ret.total == 0
  assert r_below_ret.probability == pytest.approx(0.0)


def test_weekday_slice_present() -> None:
  """The weekday slice produces at least one group."""
  result = _stat().compute(make_candles(_SEQ))
  groups = result.instruments["NQ"]["daily"].slices["weekday"].groups
  assert len(groups) > 0


def test_prev_candle_slice_present() -> None:
  """The prev_candle slice produces at least one group."""
  result = _stat().compute(make_candles(_SEQ))
  groups = result.instruments["NQ"]["daily"].slices["prev_candle"].groups
  assert len(groups) > 0


def test_size_pts_and_size_pct_slices_present() -> None:
  """Both SizeBucket slices (size_pts, size_pct) produce at least one group."""
  result = _stat().compute(make_candles(_SEQ))
  slices = result.instruments["NQ"]["daily"].slices
  assert len(slices["size_pts"].groups) >= 1
  assert len(slices["size_pct"].groups) >= 1


def test_slice_rows_partition_condition_totals() -> None:
  """For the close slicer, green + red group totals sum to the overall condition total."""
  # opened_above total overall = 4
  # green opened_above total = 1 (idx 0 only)
  # red   opened_above total = 3 (idx 1, 4, 6)
  # 1 + 3 = 4 ✓ (idx 7 has zero direction and is green but not countable)
  result = _stat().compute(make_candles(_SEQ))
  groups = result.instruments["NQ"]["daily"].slices["close"].groups
  green_results = groups["green"].results
  red_results = groups["red"].results

  g_above = next(r for r in green_results if r.condition == "opened_above" and r.outcome == "retraced")
  r_above = next(r for r in red_results if r.condition == "opened_above" and r.outcome == "retraced")
  overall_above = _row(result, "opened_above", "retraced")

  assert g_above.total + r_above.total == overall_above.total


# ===========================================================================
# 7. Empty input and edge cases
# ===========================================================================

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


def test_empty_dataframe_four_rows_present() -> None:
  """Even with no data, all four matrix rows are emitted."""
  result = _stat().compute(_empty_df())
  outcomes = {(r.condition, r.outcome) for r in result.instruments["NQ"]["daily"].results}
  assert outcomes == {
    ("opened_above", "retraced"),
    ("opened_above", "not_retraced"),
    ("opened_below", "retraced"),
    ("opened_below", "not_retraced"),
  }


def test_single_resolved_day_zero_direction_no_countable_rows() -> None:
  """One resolved session where session_open == reference_level → no countable rows."""
  # session_open == ref_level=100 → gap_size_pts=NaN → not countable
  days = [{
    "date": "2024-03-01",
    "reference_level": 100.0,
    "session_open": 100.0,
    "session_close": 105.0,
  }]
  result = _stat().compute(make_candles(days))
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 1
  for row in tf.results:
    assert row.total == 0
    assert row.probability == pytest.approx(0.0)


def test_all_opened_above_no_opened_below_condition() -> None:
  """All days open above ref level → opened_below condition is empty (total=0, P=0.0)."""
  days = [
    {"date": "2024-01-02", "reference_level": 100.0, "session_open": 105.0, "session_close": 108.0,
     "day_high": 110.0, "day_low": 104.0},
    {"date": "2024-01-03", "reference_level": 100.0, "session_open": 106.0, "session_close": 109.0,
     "day_high": 111.0, "day_low": 99.0},
    {"date": "2024-01-04", "reference_level": 100.0, "session_open": 107.0, "session_close": 110.0,
     "day_high": 112.0, "day_low": 105.0},
  ]
  result = _stat().compute(make_candles(days))
  r_below = _row(result, "opened_below", "retraced")
  assert r_below.total == 0
  assert r_below.probability == pytest.approx(0.0)
  assert _row(result, "opened_above", "retraced").total == 3


def test_all_opened_below_no_opened_above_condition() -> None:
  """All days open below ref level → opened_above condition is empty (total=0, P=0.0)."""
  days = [
    {"date": "2024-01-02", "reference_level": 100.0, "session_open": 95.0, "session_close": 92.0,
     "day_high": 99.0, "day_low": 91.0},
    {"date": "2024-01-03", "reference_level": 100.0, "session_open": 94.0, "session_close": 91.0,
     "day_high": 98.0, "day_low": 90.0},
  ]
  result = _stat().compute(make_candles(days))
  r_above = _row(result, "opened_above", "retraced")
  assert r_above.total == 0
  assert r_above.probability == pytest.approx(0.0)
  assert _row(result, "opened_below", "retraced").total == 2


# ===========================================================================
# 8. write_results round-trip + Pydantic validation
# ===========================================================================

def test_write_results_round_trip(tmp_path: Path) -> None:
  """write_results produces ict_opening_retracement.json that re-validates as StatRunResult."""
  result = _stat().compute(make_candles(_SEQ))
  written = write_results(result, results_dir=tmp_path)
  assert written.name == "ict_opening_retracement.json"

  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)
  tf = validated.instruments["NQ"]["daily"]
  assert tf.total_samples == 8

  # Spot-check the opened_above retraced row
  oa_ret = next(
    r for r in tf.results
    if r.condition == "opened_above" and r.outcome == "retraced"
  )
  assert oa_ret.count == 2
  assert oa_ret.total == 4
  assert oa_ret.probability == pytest.approx(2 / 4)


def test_write_results_has_all_slices(tmp_path: Path) -> None:
  """Serialised JSON contains all five declared slice dimensions."""
  result = _stat().compute(make_candles(_SEQ))
  written = write_results(result, results_dir=tmp_path)
  raw = json.loads(written.read_text(encoding="utf-8"))
  slices = raw["instruments"]["NQ"]["daily"]["slices"]
  assert set(slices.keys()) == {"weekday", "close", "prev_candle", "size_pts", "size_pct"}


def test_write_results_utf8_literals(tmp_path: Path) -> None:
  """French text is stored as literal UTF-8 characters, not escaped unicode."""
  result = _stat().compute(_empty_df())
  written = write_results(result, results_dir=tmp_path)
  raw = written.read_text(encoding="utf-8")
  # French definition contains accented characters (e.g. "fréquence")
  assert "é" in raw
  assert "\\u00e9" not in raw


def test_stat_name() -> None:
  """stat_name attribute must equal 'ict_opening_retracement'."""
  result = _stat().compute(_empty_df())
  assert result.stat_name == "ict_opening_retracement"


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
  assert set(result.labels.conditions) == {"opened_above", "opened_below"}
  assert set(result.labels.outcomes) == {"retraced", "not_retraced"}


def test_i18n_labels_dimensions_contain_declared_slicers() -> None:
  """Labels.dimensions carries entries for all five declared slice dimensions."""
  result = _stat().compute(make_candles(_SEQ))
  assert set(result.labels.dimensions.keys()) >= {
    "weekday", "close", "prev_candle", "size_pts", "size_pct"
  }
  for key, i18n in result.labels.dimensions.items():
    assert i18n.en != "", f"dimensions[{key}].en empty"
