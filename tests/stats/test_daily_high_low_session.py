"""Tests for stats.daily_high_low_session.standard.

All data is synthetic — no real market files required. Expected values are
hand-calculated before each assertion.

Framing: three sessions (asia 18:00–03:00 cross-midnight, london 03:00–09:30,
ny 09:30–16:15) partition each 24-hour trading cycle. For each countable cycle
(all three sessions resolved), the stat attributes the daily high to the session
containing the maximum high across all its bars and the daily low to the session
containing the minimum low. Two conditions (``daily_high``, ``daily_low``) each
form a probability distribution over the three sessions that sums to 1.

Synthetic cycles are built with controlled per-session OHLCV values so the
correct high_session / low_session is known before each assertion.

Session resolution thresholds (close_tolerance_min=15, default):
  - Asia   : start 18:00 (1080 min), end 03:00 (180 min), dur 540 min,
             last bar must reach offset >= 525.  02:59 → offset 539 ✓
             Truncated at 01:00 → offset 420 ✗ (not resolved)
  - London : start 03:00 (180 min), end 09:30 (570 min), dur 390 min,
             last bar must reach offset >= 375.  09:29 → offset 389 ✓
  - NY     : start 09:30 (570 min), end 16:15 (975 min), dur 405 min,
             last bar must reach offset >= 390.  16:14 → offset 404 ✓

Dates used and their weekdays (Eastern time):
  2024-01-02  Tuesday
  2024-01-03  Wednesday
  2024-01-04  Thursday
  2024-01-08  Monday
  2024-01-09  Tuesday
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.daily_high_low_session.standard import DailyHighLowSession

# ---------------------------------------------------------------------------
# Minimal InstrumentConfig (does not depend on NQ.yaml)
# ---------------------------------------------------------------------------
_NY = "America/New_York"
_TEST_CONFIG = InstrumentConfig(
  instrument="NQ",
  working_timezone=_NY,
  sessions={
    "asia": Session(start="18:00", end="03:00"),
    "london": Session(start="03:00", end="09:30"),
    "ny": Session(start="09:30", end="16:15"),
  },
  timeframes=["daily"],
  parquet_path=Path("data/NQ_1min.parquet"),
)


# ---------------------------------------------------------------------------
# Bar and session builders
# ---------------------------------------------------------------------------
def _bar(ts: pd.Timestamp, o: float, h: float, lo: float, c: float) -> dict:
  return {"timestamp": ts, "open": o, "high": h, "low": lo, "close": c, "volume": 100}


def _asia_bars(date: str, high: float, low: float) -> list[dict]:
  """Asia session bars for cycle ``date``: 18:00 prev evening → 02:59.

  Cross-midnight logic: bars at 18:00 and 20:00 fall on the prev calendar day
  but ``session_bars`` attributes them to ``date``'s cycle (next-day attribution
  for evening bars). The 01:00 and 02:59 bars land on ``date`` itself.

  Session high/low:
    - high = max over all bars' ``high`` column = high parameter (bar at 20:00)
    - low  = min over all bars' ``low`` column  = low parameter (bar at 01:00)
  """
  base = pd.Timestamp(date, tz=_NY)
  prev = base - pd.Timedelta(days=1)
  mid = (high + low) / 2.0
  return [
    _bar(prev.replace(hour=18, minute=0), mid, mid, mid, mid),    # clean open (offset 0)
    _bar(prev.replace(hour=20, minute=0), mid, high, mid, mid),   # session high bar
    _bar(base.replace(hour=1, minute=0), mid, mid, low, mid),     # session low bar
    _bar(base.replace(hour=2, minute=59), mid, mid, mid, mid),    # end-coverage (offset 539)
  ]


def _london_bars(date: str, high: float, low: float) -> list[dict]:
  """London session bars: 03:00 → 09:29.

  Session high = high parameter (bar at 05:00).
  Session low  = low parameter  (bar at 07:00).
  """
  base = pd.Timestamp(date, tz=_NY)
  mid = (high + low) / 2.0
  return [
    _bar(base.replace(hour=3, minute=0), mid, mid, mid, mid),     # clean open (offset 0)
    _bar(base.replace(hour=5, minute=0), mid, high, mid, mid),    # session high bar
    _bar(base.replace(hour=7, minute=0), mid, mid, low, mid),     # session low bar
    _bar(base.replace(hour=9, minute=29), mid, mid, mid, mid),    # end-coverage (offset 389)
  ]


def _ny_bars(
  date: str,
  high: float,
  low: float,
  open_: float | None = None,
  close: float | None = None,
) -> list[dict]:
  """NY session bars: 09:30 → 16:14.

  Session high = high parameter (bar at 09:30).
  Session low  = low parameter  (bar at 12:00).

  ``close`` sets the 16:14 bar's close price, used to control ``day_green``
  (cycle_close). The bar's high/low stretch around ``mid`` and ``close`` so the
  bar stays OHLC-valid (low <= open, close <= high) for any ``close``; as long
  as ``close`` stays within the session range it does not disturb the session
  extremes (set in the 09:30 and 12:00 bars).
  """
  base = pd.Timestamp(date, tz=_NY)
  mid = (high + low) / 2.0
  o = open_ if open_ is not None else mid
  c = close if close is not None else mid
  last_h = max(mid, c)
  last_l = min(mid, c)
  return [
    _bar(base.replace(hour=9, minute=30), o, high, mid, mid),     # clean open, session high
    _bar(base.replace(hour=12, minute=0), mid, mid, low, mid),    # session low bar
    _bar(base.replace(hour=16, minute=14), mid, last_h, last_l, c),  # end-coverage (offset 404) + cycle close
  ]


def _full_cycle(
  date: str,
  asia_h: float,
  asia_l: float,
  lon_h: float,
  lon_l: float,
  ny_h: float,
  ny_l: float,
  ny_close: float | None = None,
) -> list[dict]:
  """One complete 3-session cycle with known per-session highs and lows.

  cycle_open  = asia 18:00 bar open = (asia_h + asia_l) / 2
  cycle_close = ny 16:14 bar close  = ny_close (defaults to ny mid)
  day_green   = cycle_close >= cycle_open
  """
  return (
    _asia_bars(date, asia_h, asia_l)
    + _london_bars(date, lon_h, lon_l)
    + _ny_bars(date, ny_h, ny_l, close=ny_close)
  )


def make_candles(cycles: list[list[dict]]) -> pd.DataFrame:
  records = [r for cycle in cycles for r in cycle]
  df = pd.DataFrame(records)
  return df.sort_values("timestamp").reset_index(drop=True)


def _stat(
  sessions: tuple[str, ...] = ("asia", "london", "ny"),
) -> DailyHighLowSession:
  return DailyHighLowSession(instrument="NQ", config=_TEST_CONFIG, sessions=sessions)


def _row(result: StatRunResult, condition: str, outcome: str) -> StatResultRow:
  for r in result.instruments["NQ"]["daily"].results:
    if r.condition == condition and r.outcome == outcome:
      return r
  raise KeyError(f"{condition}/{outcome}")


# ===========================================================================
# Test 1: High / Low attribution — 3 cycles, unambiguous extremes
#
# Cycle  date       asia_h  lon_h  ny_h   → high_session   asia_l  lon_l  ny_l  → low_session
# ──────────────────────────────────────────────────────────────────────────────────────────────
# 1    2024-01-02   120     105    110     asia (120)       95      90     80     ny  (80)
# 2    2024-01-03   100     125    110     london (125)     90      75     85     london (75)
# 3    2024-01-04   100     110    130     ny (130)         95      85     70     ny (70)
#
# daily_high counts: asia=1, london=1, ny=1  → prob each = 1/3
# daily_low  counts: asia=0, london=1, ny=2  → prob asia=0, london=1/3, ny=2/3
# total = 3; both conditions' counts sum to 3; both probability lists sum to 1.
# ===========================================================================
_THREE_CYCLES = [
  _full_cycle("2024-01-02", asia_h=120, asia_l=95, lon_h=105, lon_l=90, ny_h=110, ny_l=80),
  _full_cycle("2024-01-03", asia_h=100, asia_l=90, lon_h=125, lon_l=75, ny_h=110, ny_l=85),
  _full_cycle("2024-01-04", asia_h=100, asia_l=95, lon_h=110, lon_l=85, ny_h=130, ny_l=70),
]


def test_high_attribution_counts():
  # daily_high: asia=1, london=1, ny=1
  result = _stat().compute(make_candles(_THREE_CYCLES))
  assert result.instruments["NQ"]["daily"].total_samples == 3
  assert _row(result, "daily_high", "asia").count == 1
  assert _row(result, "daily_high", "london").count == 1
  assert _row(result, "daily_high", "ny").count == 1
  # Counts partition the total
  assert sum(
    _row(result, "daily_high", s).count for s in ("asia", "london", "ny")
  ) == 3


def test_low_attribution_counts():
  # daily_low: asia=0, london=1, ny=2
  result = _stat().compute(make_candles(_THREE_CYCLES))
  assert _row(result, "daily_low", "asia").count == 0
  assert _row(result, "daily_low", "london").count == 1
  assert _row(result, "daily_low", "ny").count == 2
  assert sum(
    _row(result, "daily_low", s).count for s in ("asia", "london", "ny")
  ) == 3


def test_probabilities_sum_to_one():
  result = _stat().compute(make_candles(_THREE_CYCLES))
  high_probs = [_row(result, "daily_high", s).probability for s in ("asia", "london", "ny")]
  low_probs = [_row(result, "daily_low", s).probability for s in ("asia", "london", "ny")]
  assert sum(high_probs) == pytest.approx(1.0)
  assert sum(low_probs) == pytest.approx(1.0)


def test_known_probabilities():
  result = _stat().compute(make_candles(_THREE_CYCLES))
  # daily_high: each session = 1/3
  for sess in ("asia", "london", "ny"):
    assert _row(result, "daily_high", sess).probability == pytest.approx(1 / 3)
  # daily_low: asia=0, london=1/3, ny=2/3
  assert _row(result, "daily_low", "asia").probability == pytest.approx(0.0)
  assert _row(result, "daily_low", "london").probability == pytest.approx(1 / 3)
  assert _row(result, "daily_low", "ny").probability == pytest.approx(2 / 3)


def test_build_day_table_high_low_session_columns():
  """build_day_table columns high_session / low_session match hand-computed values."""
  stat = _stat()
  day_table = stat.build_day_table(make_candles(_THREE_CYCLES))
  assert len(day_table) == 3

  def _ts(d: str) -> pd.Timestamp:
    return pd.Timestamp(d, tz=_NY)

  # Cycle 2024-01-02: asia_high=120 largest → asia; ny_low=80 smallest → ny
  assert day_table.loc[_ts("2024-01-02"), "high_session"] == "asia"
  assert day_table.loc[_ts("2024-01-02"), "low_session"] == "ny"

  # Cycle 2024-01-03: london_high=125 largest → london; london_low=75 smallest → london
  assert day_table.loc[_ts("2024-01-03"), "high_session"] == "london"
  assert day_table.loc[_ts("2024-01-03"), "low_session"] == "london"

  # Cycle 2024-01-04: ny_high=130 largest → ny; ny_low=70 smallest → ny
  assert day_table.loc[_ts("2024-01-04"), "high_session"] == "ny"
  assert day_table.loc[_ts("2024-01-04"), "low_session"] == "ny"


def test_build_day_table_required_columns_present():
  stat = _stat()
  day_table = stat.build_day_table(make_candles(_THREE_CYCLES))
  expected = {
    "asia_high", "london_high", "ny_high",
    "asia_low", "london_low", "ny_low",
    "high_session", "low_session",
    "n_asia", "n_london", "n_ny",
    "day_green",
  }
  assert expected.issubset(set(day_table.columns))


# ===========================================================================
# Test 2: Cross-midnight (asia) attribution
#
# Asia evening bars live on prev calendar day but session_bars assigns them to
# the next day's cycle. If this attribution were broken, 2024-01-02's asia
# extreme would not appear and the high/low result would be wrong.
#
# 2024-01-02: asia_h=150 >> lon_h=110 > ny_h=120 → high_session must be asia
# 2024-01-02: asia_l=50  << lon_l=90  < ny_l=85  → low_session must be asia
# ===========================================================================
def test_cross_midnight_asia_owns_high():
  cycles = [
    _full_cycle("2024-01-02", asia_h=150, asia_l=95, lon_h=110, lon_l=85, ny_h=120, ny_l=80),
  ]
  result = _stat().compute(make_candles(cycles))
  assert result.instruments["NQ"]["daily"].total_samples == 1
  assert _row(result, "daily_high", "asia").count == 1
  assert _row(result, "daily_high", "london").count == 0
  assert _row(result, "daily_high", "ny").count == 0


def test_cross_midnight_asia_owns_low():
  cycles = [
    _full_cycle("2024-01-02", asia_h=110, asia_l=50, lon_h=115, lon_l=90, ny_h=120, ny_l=85),
  ]
  result = _stat().compute(make_candles(cycles))
  assert _row(result, "daily_low", "asia").count == 1
  assert _row(result, "daily_low", "london").count == 0
  assert _row(result, "daily_low", "ny").count == 0


# ===========================================================================
# Test 3: Tie-break — earliest configured session wins on exact equal extreme
#
# HIGH tie: asia_h == ny_h = 120 > london_h = 100
#   high_cols row = [120, 100, 120] → np.argmax = 0 → sessions[0] = "asia"
#   Expected: high_session = "asia" (not "ny")
#
# LOW tie: london_l == ny_l = 70 < asia_l = 80
#   low_cols row = [80, 70, 70] → np.argmin = 1 → sessions[1] = "london"
#   Expected: low_session = "london" (not "ny")
# ===========================================================================
def test_tiebreak_high_credits_earliest_session():
  # asia_h == ny_h = 120; london_h = 100 → tie between asia and ny → asia wins
  cycles = [
    _full_cycle("2024-01-02", asia_h=120, asia_l=90, lon_h=100, lon_l=85, ny_h=120, ny_l=80),
  ]
  stat = _stat()
  day_table = stat.build_day_table(make_candles(cycles))
  assert day_table.iloc[0]["high_session"] == "asia"

  result = stat.compute(make_candles(cycles))
  assert _row(result, "daily_high", "asia").count == 1
  assert _row(result, "daily_high", "ny").count == 0


def test_tiebreak_low_credits_earliest_session():
  # london_l == ny_l = 70 < asia_l = 80 → tie between london and ny → london wins
  cycles = [
    _full_cycle("2024-01-02", asia_h=100, asia_l=80, lon_h=110, lon_l=70, ny_h=120, ny_l=70),
  ]
  stat = _stat()
  day_table = stat.build_day_table(make_candles(cycles))
  assert day_table.iloc[0]["low_session"] == "london"

  result = stat.compute(make_candles(cycles))
  assert _row(result, "daily_low", "london").count == 1
  assert _row(result, "daily_low", "ny").count == 0


# ===========================================================================
# Test 4: Pending-sample discipline
#
# Case A — missing session entirely:
#   2024-01-02 has all three sessions (countable).
#   2024-01-03 has only london + ny bars; no asia bars provided → asia table
#   does not contain 2024-01-03 → inner join drops that cycle.
#   total_samples = 1.
#
# Case B — unresolved session (no end coverage):
#   Asia for 2024-01-03 stops at 01:00 (offset 420 < threshold 525) → not
#   resolved → 2024-01-03 excluded from asia's resolved set → inner join drops.
#   total_samples = 1.
#
# Case C — single session only, no join possible:
#   Only NY bars for 2024-01-02 → asia and london tables both empty → stat
#   returns early with empty day table → total_samples = 0.
# ===========================================================================
def test_missing_session_excluded():
  # 2024-01-03 has no asia bars → inner join drops it
  cycles = [
    _full_cycle("2024-01-02", asia_h=110, asia_l=90, lon_h=105, lon_l=85, ny_h=115, ny_l=80),
    _london_bars("2024-01-03", 105, 85) + _ny_bars("2024-01-03", 110, 80),
  ]
  result = _stat().compute(make_candles(cycles))
  assert result.instruments["NQ"]["daily"].total_samples == 1


def test_unresolved_session_excluded():
  # Asia for 2024-01-03: last bar at 01:00 → offset 420 < 525 → not resolved
  base = pd.Timestamp("2024-01-03", tz=_NY)
  prev = base - pd.Timedelta(days=1)
  mid = 100.0
  truncated_asia = [
    _bar(prev.replace(hour=18, minute=0), mid, mid, mid, mid),    # clean open
    _bar(base.replace(hour=1, minute=0), mid, 110.0, 90.0, mid),  # last bar too early
  ]
  cycles = [
    _full_cycle("2024-01-02", asia_h=110, asia_l=90, lon_h=105, lon_l=85, ny_h=115, ny_l=80),
    truncated_asia + _london_bars("2024-01-03", 105, 85) + _ny_bars("2024-01-03", 110, 80),
  ]
  result = _stat().compute(make_candles(cycles))
  assert result.instruments["NQ"]["daily"].total_samples == 1


def test_single_session_only_yields_zero():
  # Only NY bars — asia and london tables are empty → early return
  only_ny = _ny_bars("2024-01-02", 120, 80)
  result = _stat().compute(make_candles([only_ny]))
  assert result.instruments["NQ"]["daily"].total_samples == 0


# ===========================================================================
# Test 5: Baseline determinism and reproducibility
#
# baseline_rows(day_table, seed=42) is deterministic: two calls with the same
# seed return identical probabilities.  Different seeds almost certainly differ
# for N >= 3 cycles (probability of identical outcome < (1/3)^6 ≈ 0.001).
#
# Per-condition baseline probabilities must sum to 1.
# Each baseline row's ``total`` must equal the number of countable cycles.
# ===========================================================================
def test_baseline_is_reproducible():
  stat = _stat()
  day_table = stat.build_day_table(make_candles(_THREE_CYCLES))
  a = stat.baseline_rows(day_table, seed=42)
  b = stat.baseline_rows(day_table, seed=42)
  for ra, rb in zip(a, b):
    assert ra.probability == pytest.approx(rb.probability)
    assert ra.count == rb.count


def test_baseline_different_seeds_differ():
  stat = _stat()
  day_table = stat.build_day_table(make_candles(_THREE_CYCLES))
  a = stat.baseline_rows(day_table, seed=42)
  b = stat.baseline_rows(day_table, seed=99)
  probs_a = [r.probability for r in a]
  probs_b = [r.probability for r in b]
  assert probs_a != probs_b


def test_baseline_probabilities_sum_to_one():
  stat = _stat()
  day_table = stat.build_day_table(make_candles(_THREE_CYCLES))
  baseline = stat.baseline_rows(day_table, seed=42)
  for condition in ("daily_high", "daily_low"):
    cond_rows = [r for r in baseline if r.condition == condition]
    assert sum(r.probability for r in cond_rows) == pytest.approx(1.0)


def test_baseline_total_equals_sample_count():
  stat = _stat()
  day_table = stat.build_day_table(make_candles(_THREE_CYCLES))
  baseline = stat.baseline_rows(day_table, seed=42)
  for row in baseline:
    assert row.total == 3


# ===========================================================================
# Test 6: Slices (weekday and candle)
#
# 4 cycles on distinct weekdays with controlled day_green.
#
# cycle_open  = asia 18:00 bar open = asia_mid = (asia_h + asia_l) / 2
# cycle_close = ny 16:14 bar close  = ny_close
# day_green   = (cycle_close >= cycle_open)
#
#  Date        Weekday   asia_mid  ny_close  day_green  high_session
#  2024-01-02  Tuesday   100       110       True       london (lon_h=120)
#  2024-01-03  Wednesday 100       85        False      london (lon_h=120)
#  2024-01-08  Monday    110       120       True       asia   (asia_h=130)
#  2024-01-09  Tuesday   100       85        False      ny     (ny_h=125)
#
# Weekday groups: monday=1, tuesday=2, wednesday=1  (total = 4)
# Candle groups : green=2 (01-02, 01-08), red=2 (01-03, 01-09)
#
# Within each group, daily_high counts must sum to that group's total and
# probabilities must sum to 1.
# ===========================================================================
_SLICE_CYCLES = [
  # 2024-01-02 Tue: asia_mid=(110+90)/2=100; ny_close=110 > 100 → green
  #   lon_h=120 > ny_h=115 > asia_h=110 → high_session=london
  _full_cycle(
    "2024-01-02",
    asia_h=110, asia_l=90, lon_h=120, lon_l=85, ny_h=115, ny_l=80, ny_close=110,
  ),
  # 2024-01-03 Wed: asia_mid=100; ny_close=85 < 100 → red
  #   lon_h=120 > ny_h=115 > asia_h=110 → high_session=london
  _full_cycle(
    "2024-01-03",
    asia_h=110, asia_l=90, lon_h=120, lon_l=85, ny_h=115, ny_l=80, ny_close=85,
  ),
  # 2024-01-08 Mon: asia_mid=(130+90)/2=110; ny_close=120 > 110 → green
  #   asia_h=130 > ny_h=120 > lon_h=110 → high_session=asia
  _full_cycle(
    "2024-01-08",
    asia_h=130, asia_l=90, lon_h=110, lon_l=85, ny_h=120, ny_l=80, ny_close=120,
  ),
  # 2024-01-09 Tue: asia_mid=100; ny_close=85 < 100 → red
  #   ny_h=125 > lon_h=110 = asia_h=110 → high_session=ny (argmax picks first max)
  _full_cycle(
    "2024-01-09",
    asia_h=110, asia_l=90, lon_h=110, lon_l=85, ny_h=125, ny_l=80, ny_close=85,
  ),
]


def test_weekday_slice_present():
  result = _stat().compute(make_candles(_SLICE_CYCLES))
  assert "weekday" in result.instruments["NQ"]["daily"].slices


def test_weekday_slice_sample_counts():
  result = _stat().compute(make_candles(_SLICE_CYCLES))
  wd = result.instruments["NQ"]["daily"].slices["weekday"].groups
  # monday=1, tuesday=2, wednesday=1
  assert wd["monday"].total_samples == 1
  assert wd["tuesday"].total_samples == 2
  assert wd["wednesday"].total_samples == 1
  assert sum(g.total_samples for g in wd.values()) == 4


def test_weekday_slice_outcomes_partition_subset():
  result = _stat().compute(make_candles(_SLICE_CYCLES))
  wd = result.instruments["NQ"]["daily"].slices["weekday"].groups
  for key, group in wd.items():
    high_rows = [r for r in group.results if r.condition == "daily_high"]
    # Counts sum to the group's total
    assert sum(r.count for r in high_rows) == group.total_samples, key
    # Probabilities sum to 1
    assert sum(r.probability for r in high_rows) == pytest.approx(1.0), key


def test_candle_slice_present():
  result = _stat().compute(make_candles(_SLICE_CYCLES))
  assert "candle" in result.instruments["NQ"]["daily"].slices


def test_candle_slice_green_red_counts():
  result = _stat().compute(make_candles(_SLICE_CYCLES))
  candle = result.instruments["NQ"]["daily"].slices["candle"].groups
  assert "green" in candle
  assert "red" in candle
  # 2 green cycles (2024-01-02, 2024-01-08), 2 red (2024-01-03, 2024-01-09)
  assert candle["green"].total_samples == 2
  assert candle["red"].total_samples == 2


def test_candle_slice_outcomes_partition_subset():
  result = _stat().compute(make_candles(_SLICE_CYCLES))
  candle = result.instruments["NQ"]["daily"].slices["candle"].groups
  for key, group in candle.items():
    high_rows = [r for r in group.results if r.condition == "daily_high"]
    assert sum(r.count for r in high_rows) == group.total_samples, key
    assert sum(r.probability for r in high_rows) == pytest.approx(1.0), key


def test_candle_slice_covers_all_cycles():
  result = _stat().compute(make_candles(_SLICE_CYCLES))
  candle = result.instruments["NQ"]["daily"].slices["candle"].groups
  assert sum(g.total_samples for g in candle.values()) == 4


# ===========================================================================
# Test 7: day_green correctness
#
# cycle_open  = asia 18:00 bar open  = asia_mid = (asia_h + asia_l) / 2
# cycle_close = ny 16:14 bar close   = ny_close parameter
#
# True  case: asia_mid=(110+90)/2=100, ny_close=115 → 115 >= 100 → True
# False case: asia_mid=100,            ny_close=85  → 85  <  100 → False
#
# ny_close values (115, 85) are within [ny_l=80, ny_h=120] so they do not
# inflate or deflate the NY session high/low captured in earlier bars.
# ===========================================================================
def test_day_green_true_when_cycle_close_above_open():
  # asia_mid = 100; ny_close = 115 > 100 → day_green = True
  cycles = [
    _full_cycle(
      "2024-01-02",
      asia_h=110, asia_l=90, lon_h=105, lon_l=85, ny_h=120, ny_l=80, ny_close=115,
    ),
  ]
  stat = _stat()
  day_table = stat.build_day_table(make_candles(cycles))
  assert bool(day_table.iloc[0]["day_green"]) is True


def test_day_green_false_when_cycle_close_below_open():
  # asia_mid = 100; ny_close = 85 < 100 → day_green = False
  cycles = [
    _full_cycle(
      "2024-01-02",
      asia_h=110, asia_l=90, lon_h=105, lon_l=85, ny_h=120, ny_l=80, ny_close=85,
    ),
  ]
  stat = _stat()
  day_table = stat.build_day_table(make_candles(cycles))
  assert bool(day_table.iloc[0]["day_green"]) is False


def test_day_green_true_when_cycle_close_equals_open():
  # cycle_close == cycle_open → day_green = True (>= semantics)
  # asia_mid = (110+90)/2 = 100; ny_close = 100
  cycles = [
    _full_cycle(
      "2024-01-02",
      asia_h=110, asia_l=90, lon_h=105, lon_l=85, ny_h=120, ny_l=80, ny_close=100,
    ),
  ]
  stat = _stat()
  day_table = stat.build_day_table(make_candles(cycles))
  assert bool(day_table.iloc[0]["day_green"]) is True


# ===========================================================================
# Test 8: write_results round-trip
#
# compute → write_results → reload JSON → StatRunResult.model_validate_json
# Verified: stat_name, title.en, conditions, outcomes, and a known probability.
# ===========================================================================
def test_write_results_round_trip(tmp_path: Path):
  result = _stat().compute(make_candles(_THREE_CYCLES))
  path = write_results(result, results_dir=tmp_path)

  with open(path, encoding="utf-8") as f:
    data = json.load(f)

  assert data["stat_name"] == "daily_high_low_session"
  assert data["title"]["en"] == "Daily High / Low Session"

  # Both conditions present in labels
  assert "daily_high" in data["labels"]["conditions"]
  assert "daily_low" in data["labels"]["conditions"]

  # All three session outcomes present
  for sess in ("asia", "london", "ny"):
    assert sess in data["labels"]["outcomes"]

  # Round-trip through Pydantic validates the full schema
  validated = StatRunResult.model_validate_json(path.read_text(encoding="utf-8"))
  assert validated.stat_name == "daily_high_low_session"
  assert validated.instruments["NQ"]["daily"].total_samples == 3

  # Known probability survives: daily_high asia = 1/3
  rows = validated.instruments["NQ"]["daily"].results
  high_asia = next(
    r for r in rows if r.condition == "daily_high" and r.outcome == "asia"
  )
  assert high_asia.probability == pytest.approx(1 / 3)
  assert high_asia.total == 3


# ===========================================================================
# Test 9: Validation errors
# ===========================================================================
def test_rejects_fewer_than_two_sessions():
  with pytest.raises(ValueError, match="2 sessions"):
    DailyHighLowSession(instrument="NQ", config=_TEST_CONFIG, sessions=("asia",))


def test_rejects_duplicate_session_names():
  with pytest.raises(ValueError, match="Duplicate"):
    DailyHighLowSession(
      instrument="NQ", config=_TEST_CONFIG, sessions=("asia", "asia")
    )


def test_rejects_unknown_session_name():
  with pytest.raises(ValueError, match="Unknown"):
    DailyHighLowSession(
      instrument="NQ", config=_TEST_CONFIG, sessions=("asia", "tokyo")
    )


def test_rejects_empty_session_list():
  with pytest.raises(ValueError, match="2 sessions"):
    DailyHighLowSession(instrument="NQ", config=_TEST_CONFIG, sessions=())


# ===========================================================================
# Test 10: Empty input
#
# Empty DataFrame → build_day_table returns empty DataFrame.
# compute → 6 rows (2 conditions × 3 sessions) with count=0, total=0,
# probability=0.0.  No crash, no KeyError.
# ===========================================================================
def test_empty_input_yields_empty_day_table():
  empty = pd.DataFrame(
    columns=["timestamp", "open", "high", "low", "close", "volume"]
  )
  stat = _stat()
  day_table = stat.build_day_table(empty)
  assert day_table.empty


def test_empty_input_yields_zero_count_rows():
  empty = pd.DataFrame(
    columns=["timestamp", "open", "high", "low", "close", "volume"]
  )
  result = _stat().compute(empty)
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 0
  for condition in ("daily_high", "daily_low"):
    for sess in ("asia", "london", "ny"):
      r = _row(result, condition, sess)
      assert r.count == 0
      assert r.total == 0
      assert r.probability == pytest.approx(0.0)


def test_empty_input_no_crash_on_slices():
  empty = pd.DataFrame(
    columns=["timestamp", "open", "high", "low", "close", "volume"]
  )
  result = _stat().compute(empty)
  # Weekday and candle slices should be present but with no groups
  slices = result.instruments["NQ"]["daily"].slices
  assert "weekday" in slices
  assert "candle" in slices
  assert len(slices["weekday"].groups) == 0
  assert len(slices["candle"].groups) == 0


# ===========================================================================
# classify_samples
#
# Reusing _THREE_CYCLES (see the attribution table above):
#   2024-01-02: high_session=asia,   low_session=ny
#   2024-01-03: high_session=london, low_session=london
#   2024-01-04: high_session=ny,     low_session=ny
# Each countable cycle emits TWO SampleRows (daily_high, daily_low).
# ===========================================================================

def test_classify_samples_exact_list():
  """classify_samples emits two SampleRows per countable cycle, matching the hand-calc."""
  stat = _stat()
  table = stat.build_day_table(make_candles(_THREE_CYCLES))
  samples = stat.classify_samples(table)
  assert [(s.date, s.condition, s.outcome) for s in samples] == [
    ("2024-01-02", "daily_high", "asia"),
    ("2024-01-02", "daily_low", "ny"),
    ("2024-01-03", "daily_high", "london"),
    ("2024-01-03", "daily_low", "london"),
    ("2024-01-04", "daily_high", "ny"),
    ("2024-01-04", "daily_low", "ny"),
  ]
  assert all(s.value is None for s in samples)


def test_classify_samples_matches_compute_rows_counts():
  """For every StatResultRow, the matching SampleRow count equals r.count."""
  stat = _stat()
  table = stat.build_day_table(make_candles(_THREE_CYCLES))
  samples = stat.classify_samples(table)
  rows = stat.compute_rows(table)
  for r in rows:
    n = sum(1 for s in samples if s.condition == r.condition and s.outcome == r.outcome)
    assert n == r.count


def test_classify_samples_empty_day_table():
  """Empty day_table -> classify_samples returns []."""
  stat = _stat()
  assert stat.classify_samples(pd.DataFrame(columns=["high_session", "low_session"])) == []


def test_classify_samples_excludes_noncountable_cycle():
  """A cycle missing a session (dropped by the inner join) produces no SampleRow."""
  cycles = [
    _full_cycle("2024-01-02", asia_h=110, asia_l=90, lon_h=105, lon_l=85, ny_h=115, ny_l=80),
    _london_bars("2024-01-03", 105, 85) + _ny_bars("2024-01-03", 110, 80),
  ]
  stat = _stat()
  table = stat.build_day_table(make_candles(cycles))
  samples = stat.classify_samples(table)
  sample_dates = {s.date for s in samples}
  assert "2024-01-03" not in sample_dates
  assert "2024-01-02" in sample_dates
