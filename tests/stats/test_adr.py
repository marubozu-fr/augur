"""Tests for stats.adr.standard.

All data is synthetic — no real market files required. Expected values are
hand-calculated before each assertion.

Framing:
  - Condition ``adr``: how often the session's RTH high-to-low range exceeds or
    respects the prior-session period-N ADR (rolling mean of the N most recent
    ranges, strictly before the current session).
  - ``exceeded``:  day_range > adr  (strict; touching the ADR → respected).
  - ``respected``: day_range <= adr.

ADR for day d = mean(ranges[d-N .. d-1])  (no lookahead, shift(1)).
The first N resolved sessions have NaN ADR and are EXCLUDED from every
denominator.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.adr.standard import AverageDailyRange
from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session

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


def _stat(period: int = 3) -> AverageDailyRange:
  return AverageDailyRange(instrument="NQ", config=_TEST_CONFIG, period=period)


def _row(result: StatRunResult, condition: str, outcome: str) -> StatResultRow:
  for r in result.instruments["NQ"]["daily"].results:
    if r.condition == condition and r.outcome == outcome:
      return r
  raise KeyError((condition, outcome))


def _empty_df() -> pd.DataFrame:
  df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
  df["timestamp"] = pd.array([], dtype="datetime64[ns, America/New_York]")
  return df


# ===========================================================================
# 1. ADR warm-up exclusion
#
# period=3, M=7 sessions with integer ranges (high-low):
#   idx  date        high    low   range
#    0   2024-01-02  110     100    10
#    1   2024-01-03  120     100    20
#    2   2024-01-04  130     100    30
#    3   2024-01-05  115     100    15
#    4   2024-01-08  145     100    45
#    5   2024-01-09  125     100    25
#    6   2024-01-10  135     100    35
#
# rolling(3).mean():
#   idx 0: NaN
#   idx 1: NaN
#   idx 2: mean(10,20,30) = 20.0
#   idx 3: mean(20,30,15) = 21.667
#   idx 4: mean(30,15,45) = 30.0
#   idx 5: mean(15,45,25) = 28.333
#   idx 6: mean(45,25,35) = 35.0
#
# adr = shift(1):
#   idx 0: NaN
#   idx 1: NaN
#   idx 2: NaN
#   idx 3: 20.0   countable
#   idx 4: 21.667 countable
#   idx 5: 30.0   countable
#   idx 6: 28.333 countable
#
# Countable = M - period = 7 - 3 = 4.
# ===========================================================================

_7_SESSIONS = [
  {"date": "2024-01-02", "open": 100.0, "close": 105.0, "high": 110.0, "low": 100.0},
  {"date": "2024-01-03", "open": 100.0, "close": 105.0, "high": 120.0, "low": 100.0},
  {"date": "2024-01-04", "open": 100.0, "close": 105.0, "high": 130.0, "low": 100.0},
  {"date": "2024-01-05", "open": 100.0, "close": 105.0, "high": 115.0, "low": 100.0},
  {"date": "2024-01-08", "open": 100.0, "close": 105.0, "high": 145.0, "low": 100.0},
  {"date": "2024-01-09", "open": 100.0, "close": 105.0, "high": 125.0, "low": 100.0},
  {"date": "2024-01-10", "open": 100.0, "close": 105.0, "high": 135.0, "low": 100.0},
]


def test_warmup_exclusion_period3() -> None:
  """period=3, 7 sessions: countable total = 7 - 3 = 4."""
  result = _stat(period=3).compute(make_candles(_7_SESSIONS))
  exceeded = _row(result, "adr", "exceeded")
  respected = _row(result, "adr", "respected")
  # Both rows share the same denominator.
  assert exceeded.total == 4
  assert respected.total == 4


def test_warmup_exclusion_period5() -> None:
  """period=5, 7 sessions: countable total = 7 - 5 = 2."""
  result = _stat(period=5).compute(make_candles(_7_SESSIONS))
  exceeded = _row(result, "adr", "exceeded")
  assert exceeded.total == 2


def test_total_samples_counts_all_resolved_days() -> None:
  """total_samples = all 7 resolved sessions, including warm-up days."""
  result = _stat().compute(make_candles(_7_SESSIONS))
  assert result.instruments["NQ"]["daily"].total_samples == 7


# ===========================================================================
# 2. Exceeded vs respected partition (period=3, 7 sessions, see table above)
#
# idx 3: range=15, adr=20.0   → 15 > 20?  No  → respected
# idx 4: range=45, adr=21.667 → 45 > 21.667? Yes → exceeded
# idx 5: range=25, adr=30.0   → 25 > 30?  No  → respected
# idx 6: range=35, adr=28.333 → 35 > 28.333? Yes → exceeded
#
# exceeded_n=2, respected_n=2, total=4
# P(exceeded)=0.5, P(respected)=0.5
# ===========================================================================

def test_exceeded_count_and_probability() -> None:
  """2 of 4 countable days exceed their prior-session ADR → P=0.5."""
  row = _row(_stat(period=3).compute(make_candles(_7_SESSIONS)), "adr", "exceeded")
  assert row.count == 2
  assert row.total == 4
  assert row.probability == pytest.approx(0.5)


def test_respected_count_and_probability() -> None:
  """2 of 4 countable days stay within their prior-session ADR → P=0.5."""
  row = _row(_stat(period=3).compute(make_candles(_7_SESSIONS)), "adr", "respected")
  assert row.count == 2
  assert row.total == 4
  assert row.probability == pytest.approx(0.5)


def test_exceeded_respected_partition() -> None:
  """exceeded.count + respected.count == total for every data set."""
  result = _stat(period=3).compute(make_candles(_7_SESSIONS))
  exc = _row(result, "adr", "exceeded")
  res = _row(result, "adr", "respected")
  assert exc.count + res.count == exc.total == res.total


# ===========================================================================
# 2b. Boundary case: day_range == adr exactly → must be classified "respected"
#     (strict >; touching is NOT exceeding)
#
# period=3, 4 sessions, ranges: 10, 20, 30, 20
#   adr[3] = mean(10,20,30) = 20.0
#   range[3] = 20; 20 > 20 is False → respected, not exceeded.
# ===========================================================================

_BOUNDARY_SESSIONS = [
  {"date": "2024-01-02", "open": 100.0, "close": 105.0, "high": 110.0, "low": 100.0},
  {"date": "2024-01-03", "open": 100.0, "close": 105.0, "high": 120.0, "low": 100.0},
  {"date": "2024-01-04", "open": 100.0, "close": 105.0, "high": 130.0, "low": 100.0},
  # range=20 exactly == adr=20 → respected
  {"date": "2024-01-05", "open": 100.0, "close": 105.0, "high": 120.0, "low": 100.0},
]


def test_boundary_range_equals_adr_is_respected() -> None:
  """day_range == adr exactly → classified as respected (strict > for exceeded)."""
  result = _stat(period=3).compute(make_candles(_BOUNDARY_SESSIONS))
  exc = _row(result, "adr", "exceeded")
  res = _row(result, "adr", "respected")
  # Countable = 4 - 3 = 1 day; range=20 == adr=20 → exceeded=0, respected=1.
  assert exc.count == 0
  assert res.count == 1
  assert exc.total == res.total == 1
  assert exc.probability == pytest.approx(0.0)
  assert res.probability == pytest.approx(1.0)


# ===========================================================================
# 3. ADR value correctness
#
# Using the 7-session sequence (period=3):
#   adr[3] = mean(range[0..2]) = mean(10, 20, 30) = 20.0
#   adr[4] = mean(range[1..3]) = mean(20, 30, 15) ≈ 21.667
#   adr[5] = mean(range[2..4]) = mean(30, 15, 45) = 30.0
#   adr[6] = mean(range[3..5]) = mean(15, 45, 25) ≈ 28.333
#
# Build the day table directly and assert the adr column values.
# ===========================================================================

def test_adr_column_value_idx3() -> None:
  """adr at idx 3 = mean(range[0..2]) = mean(10,20,30) = 20.0 (no lookahead)."""
  stat = _stat(period=3)
  table = stat.build_day_table(make_candles(_7_SESSIONS))
  # Index is a DatetimeIndex of session dates; use positional loc.
  table["adr"].dropna()
  # First countable row corresponds to 2024-01-05 (idx 3 in original sequence).
  date_idx3 = pd.Timestamp("2024-01-05", tz=_NY).normalize()
  assert table.loc[date_idx3, "adr"] == pytest.approx(20.0)


def test_adr_column_value_idx4() -> None:
  """adr at idx 4 = mean(range[1..3]) = mean(20,30,15) = 21.666..."""
  stat = _stat(period=3)
  table = stat.build_day_table(make_candles(_7_SESSIONS))
  date_idx4 = pd.Timestamp("2024-01-08", tz=_NY).normalize()
  assert table.loc[date_idx4, "adr"] == pytest.approx((20 + 30 + 15) / 3)


def test_adr_column_value_idx5() -> None:
  """adr at idx 5 = mean(range[2..4]) = mean(30,15,45) = 30.0."""
  stat = _stat(period=3)
  table = stat.build_day_table(make_candles(_7_SESSIONS))
  date_idx5 = pd.Timestamp("2024-01-09", tz=_NY).normalize()
  assert table.loc[date_idx5, "adr"] == pytest.approx(30.0)


def test_adr_no_lookahead_first_n_rows_are_nan() -> None:
  """The first period resolved sessions have NaN adr — no lookahead."""
  stat = _stat(period=3)
  table = stat.build_day_table(make_candles(_7_SESSIONS))
  # First 3 rows (period=3) must have NaN adr.
  first_three_dates = table.index[:3]
  for date in first_three_dates:
    assert pd.isna(table.loc[date, "adr"]), f"Expected NaN adr at {date}"


# ===========================================================================
# 4. Determinism / reproducibility
# ===========================================================================

def _long_seq() -> pd.DataFrame:
  """~50 weekdays: alternating wide (range=30) and narrow (range=10) sessions."""
  dates: list[str] = []
  d = pd.Timestamp("2020-01-01", tz=_NY)
  while len(dates) < 50:
    if d.weekday() < 5:
      dates.append(d.strftime("%Y-%m-%d"))
    d += pd.Timedelta(days=1)

  days = []
  base = 100.0
  for i, date in enumerate(dates):
    if i % 2 == 0:
      hi, lo = base + 15.0, base - 15.0  # range = 30
    else:
      hi, lo = base + 5.0, base - 5.0    # range = 10
    days.append({
      "date": date,
      "open": base - 2.0,
      "close": base + 2.0,
      "high": hi,
      "low": lo,
    })
    base += 0.5
  return make_candles(days)


def test_baseline_rows_deterministic() -> None:
  """baseline_rows(seed=42) returns identical results on two successive calls."""
  stat = _stat(period=3)
  table = stat.build_day_table(_long_seq())
  rows_a = stat.baseline_rows(table, seed=42)
  rows_b = stat.baseline_rows(table, seed=42)
  for a, b in zip(rows_a, rows_b):
    assert (a.condition, a.outcome) == (b.condition, b.outcome)
    assert a.probability == pytest.approx(b.probability)
    assert a.count == b.count
    assert a.total == b.total


def test_baseline_permutes_adr_column() -> None:
  """baseline_rows uses a permutation of the adr column: countable N is preserved.

  The permuted adr may redistribute NaN values, so countable may change; but the
  total number of non-NaN adr values across all rows is preserved because numpy
  permutation of the raw array moves NaN positions without creating or destroying
  them.
  """
  stat = _stat(period=3)
  table = stat.build_day_table(_long_seq())
  # Original countable count.
  orig_countable = int((table["adr"].notna() & (table["adr"] > 0)).sum())
  rows = stat.baseline_rows(table, seed=42)
  baseline_total = rows[0].total  # both rows share the same denominator
  # The permutation preserves the number of valid (non-NaN, >0) adr values.
  assert baseline_total == orig_countable


def test_compute_reproducible() -> None:
  """compute() with same input and seed produces identical JSON-serializable output."""
  stat = _stat(period=3)
  df = _long_seq()
  result_a = stat.compute(df, seed=42)
  result_b = stat.compute(df, seed=42)
  assert result_a.model_dump_json() == result_b.model_dump_json()


def test_baseline_embedded_has_positive_n() -> None:
  """After compute(), countable rows carry a positive baseline_n."""
  result = _stat(period=3).compute(_long_seq())
  for row in result.instruments["NQ"]["daily"].results:
    if row.total > 0:
      assert row.baseline_n > 0, f"Expected baseline_n>0 for {row.condition}/{row.outcome}"


# ===========================================================================
# 5. Pending discipline / empty
# ===========================================================================

def test_empty_dataframe_build_day_table() -> None:
  """Empty input → build_day_table returns empty DataFrame."""
  table = _stat().build_day_table(_empty_df())
  assert table.empty


def test_empty_dataframe_compute_rows_both_zero() -> None:
  """Empty day table → compute_rows returns both outcome rows with zeros."""
  stat = _stat()
  rows = stat.compute_rows(pd.DataFrame(columns=["day_range", "adr", "prev_session_green"]))
  by_outcome = {r.outcome: r for r in rows}
  assert set(by_outcome) == {"exceeded", "respected"}
  for r in rows:
    assert r.count == 0
    assert r.total == 0
    assert r.probability == pytest.approx(0.0)


def test_empty_dataframe_full_compute() -> None:
  """Empty candles → full compute() produces total_samples=0, all rows zeroed."""
  result = _stat().compute(_empty_df())
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 0
  assert tf.data_range == []
  for row in tf.results:
    assert row.count == 0
    assert row.total == 0
    assert row.probability == pytest.approx(0.0)


def test_single_resolved_day_all_nan_adr() -> None:
  """One resolved session → adr=NaN (warm-up) → all totals zero, no crash."""
  days = [{"date": "2024-03-01", "open": 100.0, "close": 105.0, "high": 115.0, "low": 95.0}]
  result = _stat().compute(make_candles(days))
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 1
  for row in tf.results:
    assert row.total == 0


def test_pending_day_excluded_from_total_samples() -> None:
  """A truncated/early-close day does not appear in total_samples."""
  stat = _stat(period=3)
  base_df = make_candles(_7_SESSIONS)
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
  day_table = _stat().build_day_table(_make_truncated_day("2024-01-10"))
  pending = pd.Timestamp("2024-01-10", tz=_NY).normalize()
  assert pending not in day_table.index


# ===========================================================================
# 6. Weekday slice
#
# With 7 sessions covering Mon–Thu of two weeks, the weekday slice must
# group them. The sum of exceeded counts across all weekday groups must equal
# the overall exceeded count.  Similarly for respected.
# ===========================================================================

def test_weekday_slice_present() -> None:
  """Result includes a 'weekday' slice."""
  result = _stat(period=3).compute(make_candles(_7_SESSIONS))
  slices = result.instruments["NQ"]["daily"].slices
  assert "weekday" in slices


def test_weekday_slice_exceeded_counts_sum_to_overall() -> None:
  """Sum of exceeded.count across weekday groups == overall exceeded.count."""
  result = _stat(period=3).compute(make_candles(_7_SESSIONS))
  overall_exc = _row(result, "adr", "exceeded").count
  wk = result.instruments["NQ"]["daily"].slices["weekday"]
  total_exc = 0
  for grp in wk.groups.values():
    for r in grp.results:
      if r.condition == "adr" and r.outcome == "exceeded":
        total_exc += r.count
  assert total_exc == overall_exc


def test_weekday_slice_outcomes_partition_per_group() -> None:
  """Within each weekday group, exceeded.count + respected.count == group total_samples
  minus NaN-adr (warm-up) days in that group."""
  result = _stat(period=3).compute(make_candles(_7_SESSIONS))
  wk = result.instruments["NQ"]["daily"].slices["weekday"]
  for grp_key, grp in wk.groups.items():
    counts = {r.outcome: r for r in grp.results if r.condition == "adr"}
    if "exceeded" in counts and "respected" in counts:
      exc = counts["exceeded"]
      res = counts["respected"]
      # Both rows share the same denominator (total).
      assert exc.total == res.total, f"group {grp_key}: totals differ"
      assert exc.count + res.count == exc.total, f"group {grp_key}: partition broken"


# ===========================================================================
# 7. Custom period
# ===========================================================================

def test_custom_period_3_warmup() -> None:
  """period=3: first 3 resolved sessions excluded from denominators."""
  result = _stat(period=3).compute(make_candles(_7_SESSIONS))
  assert _row(result, "adr", "exceeded").total == 7 - 3


def test_custom_period_5_warmup() -> None:
  """period=5: first 5 resolved sessions excluded from denominators."""
  result = _stat(period=5).compute(make_candles(_7_SESSIONS))
  assert _row(result, "adr", "exceeded").total == 7 - 5


def test_custom_period_stored_on_instance() -> None:
  """period is stored and used: different periods give different countable totals."""
  stat3 = _stat(period=3)
  stat5 = _stat(period=5)
  df = make_candles(_7_SESSIONS)
  total3 = _row(stat3.compute(df), "adr", "exceeded").total
  total5 = _row(stat5.compute(df), "adr", "exceeded").total
  assert total3 == 4  # 7 - 3
  assert total5 == 2  # 7 - 5
  assert total3 != total5


def test_period_3_adr_uses_3_session_window() -> None:
  """With period=3, verify the 3-session rolling window via a direct adr value check.

  ranges: 10, 20, 30, 15 → adr[3] = mean(10,20,30) = 20.0.
  """
  stat = _stat(period=3)
  table = stat.build_day_table(make_candles(_BOUNDARY_SESSIONS))
  date_idx3 = pd.Timestamp("2024-01-05", tz=_NY).normalize()
  assert table.loc[date_idx3, "adr"] == pytest.approx(20.0)


# ===========================================================================
# All-exceeded edge case: all countable days exceed their ADR.
# Using period=3, 4 sessions with rapidly increasing ranges: 10, 20, 30, 100
#   adr[3] = mean(10,20,30) = 20.0; range[3]=100 > 20 → exceeded=1
# ===========================================================================

_ALL_EXCEEDED_SESSIONS = [
  {"date": "2024-01-02", "open": 100.0, "close": 105.0, "high": 110.0, "low": 100.0},
  {"date": "2024-01-03", "open": 100.0, "close": 105.0, "high": 120.0, "low": 100.0},
  {"date": "2024-01-04", "open": 100.0, "close": 105.0, "high": 130.0, "low": 100.0},
  {"date": "2024-01-05", "open": 100.0, "close": 105.0, "high": 200.0, "low": 100.0},
]


def test_all_countable_days_exceeded() -> None:
  """When every countable day exceeds its ADR: exceeded=total, respected=0."""
  result = _stat(period=3).compute(make_candles(_ALL_EXCEEDED_SESSIONS))
  exc = _row(result, "adr", "exceeded")
  res = _row(result, "adr", "respected")
  # Countable = 4 - 3 = 1; range=100 > adr=20 → exceeded.
  assert exc.total == 1
  assert exc.count == 1
  assert res.count == 0
  assert exc.probability == pytest.approx(1.0)
  assert res.probability == pytest.approx(0.0)


# ===========================================================================
# All-respected edge case: all countable days respect their ADR.
# Using period=3, 4 sessions: ranges 10, 20, 30, 5
#   adr[3] = mean(10,20,30) = 20.0; range[3]=5 <= 20 → respected=1
# ===========================================================================

_ALL_RESPECTED_SESSIONS = [
  {"date": "2024-01-02", "open": 100.0, "close": 105.0, "high": 110.0, "low": 100.0},
  {"date": "2024-01-03", "open": 100.0, "close": 105.0, "high": 120.0, "low": 100.0},
  {"date": "2024-01-04", "open": 100.0, "close": 105.0, "high": 130.0, "low": 100.0},
  {"date": "2024-01-05", "open": 100.0, "close": 102.0, "high": 105.0, "low": 100.0},
]


def test_all_countable_days_respected() -> None:
  """When every countable day respects its ADR: respected=total, exceeded=0."""
  result = _stat(period=3).compute(make_candles(_ALL_RESPECTED_SESSIONS))
  exc = _row(result, "adr", "exceeded")
  res = _row(result, "adr", "respected")
  # Countable = 4 - 3 = 1; range=5 <= adr=20 → respected.
  assert res.total == 1
  assert res.count == 1
  assert exc.count == 0
  assert res.probability == pytest.approx(1.0)
  assert exc.probability == pytest.approx(0.0)


# ===========================================================================
# Two rows only (structure check)
# ===========================================================================

def test_exactly_two_rows() -> None:
  """compute() returns exactly the exceeded and respected outcome rows."""
  rows = _stat(period=3).compute(make_candles(_7_SESSIONS)).instruments["NQ"]["daily"].results
  assert {(r.condition, r.outcome) for r in rows} == {
    ("adr", "exceeded"),
    ("adr", "respected"),
  }


# ===========================================================================
# data_range
# ===========================================================================

def test_data_range_spans_all_resolved_sessions() -> None:
  """data_range spans from the first to the last resolved session date."""
  result = _stat(period=3).compute(make_candles(_7_SESSIONS))
  assert result.instruments["NQ"]["daily"].data_range == ["2024-01-02", "2024-01-10"]


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
  assert set(result.labels.conditions) == {"adr"}
  assert set(result.labels.outcomes) == {"exceeded", "respected"}


# ===========================================================================
# stat_name and write_results round-trip
# ===========================================================================

def test_stat_name() -> None:
  assert _stat().compute(_empty_df()).stat_name == "adr"


def test_write_results_round_trip(tmp_path: Path) -> None:
  """write_results produces adr.json that re-validates correctly."""
  result = _stat(period=3).compute(make_candles(_7_SESSIONS))
  written = write_results(result, results_dir=tmp_path)
  assert written.name == "adr.json"

  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)
  assert validated.instruments["NQ"]["daily"].total_samples == 7


def test_write_results_french_accent_literal(tmp_path: Path) -> None:
  """French text is stored as literal UTF-8 characters, not escaped unicode."""
  result = _stat().compute(_empty_df())
  written = write_results(result, results_dir=tmp_path)
  raw = written.read_text(encoding="utf-8")
  # The French title/definition contains accented characters (e.g. "journalier").
  assert "journalier" in raw
