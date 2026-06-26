"""Tests for stats.market_session_correlation.standard.

All data is synthetic — no real market files required. Expected values are
hand-calculated before each assertion.

Matrix framing: condition = session 1's close color, outcome = session 2's close
color, both within the same cycle. The default pair is ``london`` (session 1) →
``ny`` (session 2). A cycle is countable only when BOTH sessions are resolved; in
``close_to_close`` mode the first joined cycle has no prior close and is excluded
(pending discipline).

Each synthetic cycle carries london bars (03:00–09:29, controlling session 1's
open/close and its high/low range) plus ny bars (09:30–16:14, controlling session
2's open/close). A flag instead builds session 1 as the cross-midnight ``asia``
session (18:00 prior evening → 02:59) to exercise cycle attribution.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.market_session_correlation.standard import MarketSessionCorrelation

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


def _bar(ts: pd.Timestamp, o: float, h: float, lo: float, c: float) -> dict:
  return {"timestamp": ts, "open": o, "high": h, "low": lo, "close": c, "volume": 100}


def _london_bars(
  date: str, open_: float, close_: float, high: float | None = None, low: float | None = None
) -> list[dict]:
  """London session 1 bars (03:00–09:29). Open bar at 03:00, close bar at 09:29.

  ``high`` / ``low`` (defaulting to the open/close envelope) set the session range
  read by the size slicer. Must enclose ``open_`` and ``close_``.
  """
  base = pd.Timestamp(date, tz=_NY)
  if high is None:
    high = max(open_, close_)
  if low is None:
    low = min(open_, close_)
  return [
    _bar(base.replace(hour=3, minute=0), open_, open_, open_, open_),  # clean open
    _bar(base.replace(hour=5, minute=0), open_, high, low, open_),  # range
    _bar(base.replace(hour=9, minute=29), close_, close_, close_, close_),  # close + coverage
  ]


def _asia_bars(date: str, open_: float, close_: float) -> list[dict]:
  """Asia session 1 bars for cycle ``date``: 18:00 prior evening → 02:59.

  Exercises cross-midnight attribution: the 18:00 open bar (prior day) and the
  02:59 close bar (cycle day) both belong to ``date``'s cycle.
  """
  base = pd.Timestamp(date, tz=_NY)
  prev = base - pd.Timedelta(days=1)
  return [
    _bar(prev.replace(hour=18, minute=0), open_, open_, open_, open_),  # clean open
    _bar(base.replace(hour=2, minute=59), close_, close_, close_, close_),  # close + coverage
  ]


def _ny_bars(date: str, open_: float, close_: float) -> list[dict]:
  """NY session 2 bars (09:30–16:14). Open bar at 09:30, close bar at 16:14."""
  base = pd.Timestamp(date, tz=_NY)
  hi, lo = max(open_, close_), min(open_, close_)
  return [
    _bar(base.replace(hour=9, minute=30), open_, hi, lo, open_),  # clean open
    _bar(base.replace(hour=16, minute=14), close_, hi, lo, close_),  # close + coverage
  ]


def _cycle(
  date: str,
  s1_open: float,
  s1_close: float,
  s2_open: float,
  s2_close: float,
  *,
  s1_high: float | None = None,
  s1_low: float | None = None,
  asia: bool = False,
) -> list[dict]:
  s1 = (
    _asia_bars(date, s1_open, s1_close)
    if asia
    else _london_bars(date, s1_open, s1_close, high=s1_high, low=s1_low)
  )
  return s1 + _ny_bars(date, s2_open, s2_close)


def make_candles(cycles: list[list[dict]]) -> pd.DataFrame:
  records = [r for cycle in cycles for r in cycle]
  df = pd.DataFrame(records)
  return df.sort_values("timestamp").reset_index(drop=True)


# Green = close above open; red = close below open (open_to_close basis).
_G_OPEN, _G_CLOSE = 100.0, 120.0
_R_OPEN, _R_CLOSE = 100.0, 80.0


def _stat(
  session1: str = "london", session2: str = "ny", performance: str = "open_to_close"
) -> MarketSessionCorrelation:
  return MarketSessionCorrelation(
    instrument="NQ",
    config=_TEST_CONFIG,
    session1=session1,
    session2=session2,
    performance=performance,
  )


def _row(result: StatRunResult, condition: str, outcome: str) -> StatResultRow:
  for r in result.instruments["NQ"]["daily"].results:
    if r.condition == condition and r.outcome == outcome:
      return r
  raise KeyError((condition, outcome))


# ===========================================================================
# Core 2x2 matrix (open_to_close)
#
# Eight cycles with (session1 color, session2 color) and distinct session 1
# range sizes (20..90) for the size slice:
#   s1G,s2G x3 | s1G,s2R x1 | s1R,s2G x2 | s1R,s2R x2
#   s1_green: total 4, s2 green 3 → P=3/4 ; s1_red: total 4, s2 green 2 → P=2/4
# ===========================================================================
def _green_s1(date: str, size: float, s2_green: bool) -> list[dict]:
  s2o, s2c = (_G_OPEN, _G_CLOSE) if s2_green else (_R_OPEN, _R_CLOSE)
  return _cycle(date, _G_OPEN, _G_CLOSE, s2o, s2c, s1_high=120.0, s1_low=120.0 - size)


def _red_s1(date: str, size: float, s2_green: bool) -> list[dict]:
  s2o, s2c = (_G_OPEN, _G_CLOSE) if s2_green else (_R_OPEN, _R_CLOSE)
  return _cycle(date, _R_OPEN, _R_CLOSE, s2o, s2c, s1_high=100.0, s1_low=100.0 - size)


_SEQ = [
  _green_s1("2024-01-01", 20, s2_green=True),
  _green_s1("2024-01-02", 30, s2_green=True),
  _green_s1("2024-01-03", 40, s2_green=True),
  _green_s1("2024-01-04", 50, s2_green=False),
  _red_s1("2024-01-05", 60, s2_green=True),
  _red_s1("2024-01-08", 70, s2_green=True),
  _red_s1("2024-01-09", 80, s2_green=False),
  _red_s1("2024-01-10", 90, s2_green=False),
]


def test_total_samples_counts_countable_cycles() -> None:
  result = _stat().compute(make_candles(_SEQ))
  assert result.instruments["NQ"]["daily"].total_samples == 8


def test_s1_green_conditional() -> None:
  """Given session 1 green: P(s2 green)=3/4, P(s2 red)=1/4 over 4 cycles."""
  result = _stat().compute(make_candles(_SEQ))
  green = _row(result, "s1_green", "green")
  red = _row(result, "s1_green", "red")
  assert green.total == red.total == 4
  assert green.count == 3
  assert green.probability == pytest.approx(0.75)
  assert red.count == 1
  assert red.probability == pytest.approx(0.25)


def test_s1_red_conditional() -> None:
  """Given session 1 red: P(s2 green)=2/4, P(s2 red)=2/4 over 4 cycles."""
  result = _stat().compute(make_candles(_SEQ))
  green = _row(result, "s1_red", "green")
  red = _row(result, "s1_red", "red")
  assert green.total == red.total == 4
  assert green.count == 2
  assert green.probability == pytest.approx(0.5)
  assert red.count == 2
  assert red.probability == pytest.approx(0.5)


def test_outcomes_partition_each_condition() -> None:
  result = _stat().compute(make_candles(_SEQ))
  for cond in ("s1_green", "s1_red"):
    green = _row(result, cond, "green")
    red = _row(result, cond, "red")
    assert green.count + red.count == green.total == red.total


def test_four_rows_only() -> None:
  result = _stat().compute(make_candles(_SEQ))
  rows = result.instruments["NQ"]["daily"].results
  assert {(r.condition, r.outcome) for r in rows} == {
    ("s1_green", "green"), ("s1_green", "red"),
    ("s1_red", "green"), ("s1_red", "red"),
  }


def test_data_range() -> None:
  result = _stat().compute(make_candles(_SEQ))
  assert result.instruments["NQ"]["daily"].data_range == ["2024-01-01", "2024-01-10"]


# ===========================================================================
# close_to_close direction basis
#
# Per-cycle closes (s1_close, s2_close):
#   c0: (100, 200)  dropped (no prior close)
#   c1: (110, 190)  s1 110>=100 G ; s2 190>=200 R
#   c2: (105, 195)  s1 105>=110 R ; s2 195>=190 G
#   c3: (105, 195)  s1 105>=105 G(equal) ; s2 195>=195 G(equal)
#   countable: c1(s1G,s2R) c2(s1R,s2G) c3(s1G,s2G)
#     s1_green: c1,c3 → s2 green 1/2 ; s1_red: c2 → s2 green 1/1
# ===========================================================================
_CTC = [
  _cycle("2024-01-01", 100, 100, 200, 200),
  _cycle("2024-01-02", 110, 110, 190, 190),
  _cycle("2024-01-03", 105, 105, 195, 195),
  _cycle("2024-01-04", 105, 105, 195, 195),
]


def test_close_to_close_drops_first_cycle() -> None:
  stat = _stat(performance="close_to_close")
  day_table = stat.build_day_table(make_candles(_CTC))
  assert len(day_table) == 3
  first = pd.Timestamp("2024-01-01", tz=_NY).normalize()
  assert first not in day_table.index


def test_close_to_close_equal_close_is_green() -> None:
  stat = _stat(performance="close_to_close")
  day_table = stat.build_day_table(make_candles(_CTC)).sort_index()
  c3 = pd.Timestamp("2024-01-04", tz=_NY).normalize()  # s1 105 vs prior 105
  assert bool(day_table.loc[c3, "s1_green"])
  assert bool(day_table.loc[c3, "s2_green"])


def test_close_to_close_prior_close_skips_gap_cycle() -> None:
  """A cycle with session 1 resolved but session 2 missing is dropped by the inner
  join; the next cycle's prior close is then taken from the last JOINED cycle,
  skipping the gap (shift(1) runs over the joined table, not the raw calendar)."""
  cycles = [
    _cycle("2024-01-02", 100, 100, 100, 100),  # both resolved
    _london_bars("2024-01-03", 100, 80),       # session 1 only → dropped by join
    _cycle("2024-01-04", 100, 110, 100, 90),   # both resolved
  ]
  stat = _stat(performance="close_to_close")
  day_table = stat.build_day_table(make_candles(cycles)).sort_index()
  # 2024-01-02 dropped (no prior close); 2024-01-03 dropped (no session 2).
  assert len(day_table) == 1
  c2 = pd.Timestamp("2024-01-04", tz=_NY).normalize()
  assert c2 in day_table.index
  # Prior close comes from 2024-01-02 (the gap cycle is skipped):
  assert bool(day_table.loc[c2, "s1_green"])      # s1 110 >= 100
  assert not bool(day_table.loc[c2, "s2_green"])  # s2 90 < 100


def test_close_to_close_conditionals() -> None:
  stat = _stat(performance="close_to_close")
  result = stat.compute(make_candles(_CTC))
  assert result.instruments["NQ"]["daily"].total_samples == 3
  s1g_green = _row(result, "s1_green", "green")
  assert s1g_green.total == 2  # c1, c3
  assert s1g_green.count == 1  # c3
  assert s1g_green.probability == pytest.approx(0.5)
  s1r_green = _row(result, "s1_red", "green")
  assert s1r_green.total == 1  # c2
  assert s1r_green.count == 1
  assert s1r_green.probability == pytest.approx(1.0)


# ===========================================================================
# Slices
# ===========================================================================
def test_declared_slices_present() -> None:
  result = _stat().compute(make_candles(_SEQ))
  slices = result.instruments["NQ"]["daily"].slices
  assert set(slices) == {"weekday", "size"}


def test_weekday_slice_partitions_cycles() -> None:
  result = _stat().compute(make_candles(_SEQ))
  weekday = result.instruments["NQ"]["daily"].slices["weekday"].groups
  assert sum(g.total_samples for g in weekday.values()) == 8


def test_size_slice_partitions_cycles() -> None:
  # Sizes 20..90 → four quartile buckets of two cycles each; every cycle lands in
  # exactly one bucket.
  result = _stat().compute(make_candles(_SEQ))
  size = result.instruments["NQ"]["daily"].slices["size"].groups
  assert sum(g.total_samples for g in size.values()) == 8
  assert len(size) == 4


# ===========================================================================
# Cross-midnight session 1 (asia → ny)
# ===========================================================================
def test_asia_session_cross_midnight_attribution() -> None:
  cycles = [
    _cycle("2024-01-02", 100, 120, 100, 120, asia=True),  # s1G, s2G
    _cycle("2024-01-03", 100, 80, 100, 120, asia=True),   # s1R, s2G
  ]
  result = _stat(session1="asia").compute(make_candles(cycles))
  assert result.instruments["NQ"]["daily"].total_samples == 2
  assert _row(result, "s1_green", "green").count == 1
  assert _row(result, "s1_red", "green").count == 1


# ===========================================================================
# Baseline ≈ 50% and deterministic
# ===========================================================================
def _long_seq() -> pd.DataFrame:
  """250 weekday cycles where session 2 perfectly follows session 1's color."""
  dates: list[str] = []
  d = pd.Timestamp("2020-01-01", tz=_NY)
  while len(dates) < 250:
    if d.weekday() < 5:
      dates.append(d.strftime("%Y-%m-%d"))
    d += pd.Timedelta(days=1)
  cycles = []
  for i, date in enumerate(dates):
    s1_green = (i % 5) < 3  # streaky pattern
    s1o, s1c = (_G_OPEN, _G_CLOSE) if s1_green else (_R_OPEN, _R_CLOSE)
    cycles.append(_cycle(date, s1o, s1c, s1o, s1c))  # session 2 mirrors session 1
  return make_candles(cycles)


def test_baseline_approx_fifty_percent() -> None:
  rows = _stat().baseline(_long_seq(), seed=42)
  for row in rows:
    assert row.probability == pytest.approx(0.5, abs=0.12)


def test_baseline_deterministic() -> None:
  df = _long_seq()
  rows_a = _stat().baseline(df, seed=7)
  rows_b = _stat().baseline(df, seed=7)
  for a, b in zip(rows_a, rows_b):
    assert (a.condition, a.outcome) == (b.condition, b.outcome)
    assert a.probability == pytest.approx(b.probability)


def test_baseline_embedded_in_compute() -> None:
  result = _stat().compute(_long_seq())
  for row in result.instruments["NQ"]["daily"].results:
    assert row.baseline_n > 0


def test_real_correlation_beats_baseline() -> None:
  """Perfect follow-through (s2 == s1) far exceeds the random baseline."""
  result = _stat().compute(_long_seq())
  s1g_green = _row(result, "s1_green", "green")
  assert s1g_green.probability == pytest.approx(1.0)
  assert s1g_green.probability > s1g_green.baseline_prob


# ===========================================================================
# Pending-sample discipline
# ===========================================================================
def test_unresolved_session2_excluded() -> None:
  # A cycle whose ny session stops at 09:50 (no end coverage) is dropped.
  base = pd.Timestamp("2024-01-03", tz=_NY)
  truncated = _london_bars("2024-01-03", 100, 120) + [
    _bar(base.replace(hour=9, minute=30), 100, 100, 100, 100),
    _bar(base.replace(hour=9, minute=50), 100, 100, 100, 100),
  ]
  cycles = [_cycle("2024-01-02", 100, 120, 100, 120), truncated]
  result = _stat().compute(make_candles(cycles))
  assert result.instruments["NQ"]["daily"].total_samples == 1


def test_missing_session1_excluded() -> None:
  # A cycle with only ny bars (no london range) is not countable.
  cycles = [
    _cycle("2024-01-02", 100, 120, 100, 120),
    _ny_bars("2024-01-03", 100, 120),
  ]
  result = _stat().compute(make_candles(cycles))
  assert result.instruments["NQ"]["daily"].total_samples == 1


# ===========================================================================
# Edge cases & validation
# ===========================================================================
def _empty_df() -> pd.DataFrame:
  df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
  df["timestamp"] = pd.array([], dtype="datetime64[ns, America/New_York]")
  return df


def test_empty_dataframe_no_crash() -> None:
  result = _stat().compute(_empty_df())
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 0
  assert tf.data_range == []
  for row in tf.results:
    assert row.count == 0
    assert row.total == 0
    assert row.probability == pytest.approx(0.0)


def test_invalid_performance_raises() -> None:
  with pytest.raises(ValueError, match="Unsupported performance"):
    MarketSessionCorrelation(
      instrument="NQ", config=_TEST_CONFIG, performance="tick_to_tick"
    )


def test_rejects_unknown_session() -> None:
  with pytest.raises(ValueError, match="Unknown session"):
    _stat(session1="tokyo")


def test_rejects_identical_sessions() -> None:
  with pytest.raises(ValueError, match="different"):
    _stat(session1="ny", session2="ny")


def test_defaults() -> None:
  stat = MarketSessionCorrelation(instrument="NQ", config=_TEST_CONFIG)
  assert stat.performance == "close_to_close"
  assert stat.session1 == "london"
  assert stat.session2 == "ny"
  assert stat.timeframe == "daily"


# ===========================================================================
# i18n
# ===========================================================================
def test_i18n_title_and_definition() -> None:
  result = _stat().compute(_empty_df())
  assert result.title.en != "" and result.title.fr != ""
  assert result.definition.en != "" and result.definition.fr != ""


def test_i18n_labels_have_en_and_fr() -> None:
  result = _stat().compute(_empty_df())
  for mapping in (result.labels.conditions, result.labels.outcomes):
    for key, i18n in mapping.items():
      assert i18n.en != "", f"{key}.en empty"
      assert i18n.fr != "", f"{key}.fr empty"
  assert set(result.labels.conditions) == {"s1_green", "s1_red"}
  assert set(result.labels.outcomes) == {"green", "red"}


# ===========================================================================
# write_results round-trip
# ===========================================================================
def test_stat_name() -> None:
  assert _stat().compute(_empty_df()).stat_name == "market_session_correlation"


def test_write_results_round_trip(tmp_path: Path) -> None:
  result = _stat().compute(make_candles(_SEQ))
  written = write_results(result, results_dir=tmp_path)
  assert written.name == "market_session_correlation.json"

  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)
  assert validated.instruments["NQ"]["daily"].total_samples == 8


def test_write_results_french_accent_literal(tmp_path: Path) -> None:
  result = _stat().compute(_empty_df())
  written = write_results(result, results_dir=tmp_path)
  raw = written.read_text(encoding="utf-8")
  assert "é" in raw  # "marché"
  assert "\\u00e9" not in raw
