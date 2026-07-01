"""Tests for stats.outside_days.standard.

All data is synthetic — no real market files required. Expected values are
hand-calculated before each assertion.

Framing:
  - Tier 1 (condition ``outside_open``): how often the session opens outside the
    prior resolved day's range. ``bullish`` = open > prev_high; ``bearish`` =
    open < prev_low. total = countable days (those with a prior resolved day).
  - Tier 2 (conditions ``bullish`` / ``bearish``): continuation vs reversal of
    the outside move. For a bullish outside, reversal means price re-entered the
    range (day_low < prev_high) and continuation means it held above
    (day_low >= prev_high); symmetric for bearish on day_high vs prev_low.

The first resolved day has no prior day and is excluded from every denominator.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.outside_days.standard import OutsideDays

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


def _stat() -> OutsideDays:
  return OutsideDays(instrument="NQ", config=_TEST_CONFIG)


def _row(result: StatRunResult, condition: str, outcome: str) -> StatResultRow:
  for r in result.instruments["NQ"]["daily"].results:
    if r.condition == condition and r.outcome == outcome:
      return r
  raise KeyError((condition, outcome))


# ===========================================================================
# Core derivation (6 days)
#
#  idx date        open close high  low  | prev_high prev_low | classification
#   0  2024-01-02  100  100   110   90   |    —        —      | excluded (no prior)
#   1  2024-01-03  112  118   120   111  |   110       90     | bullish, continuation
#   2  2024-01-04  122  110   125   108  |   120       111    | bullish, reversal
#   3  2024-01-05  105  102   107   100  |   125       108    | bearish, continuation
#   4  2024-01-08   98  104   106    95  |   107       100    | bearish, reversal
#   5  2024-01-09  100  103   105    96  |   106        95     | inside (neither)
#
# Countable = idx 1..5 (5 days; idx 0 has no prior day).
#
# bullish open days: 1 (112>110), 2 (122>120)         → count=2
# bearish open days: 3 (105<108), 4 (98<100)          → count=2
# inside:            5 (95 <= 100 <= 106)
#
# Tier-1 (outside_open), total=5:
#   bullish: 2/5 = 0.4   bearish: 2/5 = 0.4
#
# Tier-2 continuation/reversal:
#   bullish {1: day_low 111 >= 110 → continuation, 2: day_low 108 < 120 → reversal}
#     continuation=1, reversal=1 (total 2)
#   bearish {3: day_high 107 <= 108 → continuation, 4: day_high 106 > 100 → reversal}
#     continuation=1, reversal=1 (total 2)
#
# total_samples = 6 (all resolved days, incl. idx 0)
# ===========================================================================

_SEQ = [
  {"date": "2024-01-02", "open": 100.0, "close": 100.0, "high": 110.0, "low": 90.0},
  {"date": "2024-01-03", "open": 112.0, "close": 118.0, "high": 120.0, "low": 111.0},
  {"date": "2024-01-04", "open": 122.0, "close": 110.0, "high": 125.0, "low": 108.0},
  {"date": "2024-01-05", "open": 105.0, "close": 102.0, "high": 107.0, "low": 100.0},
  {"date": "2024-01-08", "open": 98.0, "close": 104.0, "high": 106.0, "low": 95.0},
  {"date": "2024-01-09", "open": 100.0, "close": 103.0, "high": 105.0, "low": 96.0},
]


def test_total_samples_counts_all_resolved_days() -> None:
  """total_samples = 6 resolved sessions (the first one is still resolved)."""
  result = _stat().compute(make_candles(_SEQ))
  assert result.instruments["NQ"]["daily"].total_samples == 6


def test_bullish_open_frequency() -> None:
  """2 of 5 countable days open above the prior high → P=0.4."""
  row = _row(_stat().compute(make_candles(_SEQ)), "outside_open", "bullish")
  assert row.total == 5
  assert row.count == 2
  assert row.probability == pytest.approx(0.4)


def test_bearish_open_frequency() -> None:
  """2 of 5 countable days open below the prior low → P=0.4."""
  row = _row(_stat().compute(make_candles(_SEQ)), "outside_open", "bearish")
  assert row.total == 5
  assert row.count == 2
  assert row.probability == pytest.approx(0.4)


def test_bullish_continuation_vs_reversal() -> None:
  """Of 2 bullish-outside days, 1 held above (continuation), 1 re-entered (reversal)."""
  result = _stat().compute(make_candles(_SEQ))
  cont = _row(result, "bullish", "continuation")
  rev = _row(result, "bullish", "reversal")
  assert cont.total == rev.total == 2
  assert cont.count == 1
  assert rev.count == 1
  assert cont.probability == pytest.approx(0.5)


def test_bearish_continuation_vs_reversal() -> None:
  """Of 2 bearish-outside days, 1 held below (continuation), 1 re-entered (reversal)."""
  result = _stat().compute(make_candles(_SEQ))
  cont = _row(result, "bearish", "continuation")
  rev = _row(result, "bearish", "reversal")
  assert cont.total == rev.total == 2
  assert cont.count == 1
  assert rev.count == 1
  assert rev.probability == pytest.approx(0.5)


def test_continuation_reversal_partitions_each_outside_condition() -> None:
  """continuation.count + reversal.count == total for each outside condition."""
  result = _stat().compute(make_candles(_SEQ))
  for cond in ("bullish", "bearish"):
    cont = _row(result, cond, "continuation")
    rev = _row(result, cond, "reversal")
    assert cont.count + rev.count == cont.total == rev.total


def test_first_day_excluded_from_denominators() -> None:
  """Tier-1 total = 5 = n_resolved - 1 (first day has no prior day)."""
  result = _stat().compute(make_candles(_SEQ))
  assert _row(result, "outside_open", "bullish").total == 6 - 1


def test_six_rows_only() -> None:
  """Exactly the two outside-open rows plus the four continuation/reversal rows."""
  rows = _stat().compute(make_candles(_SEQ)).instruments["NQ"]["daily"].results
  assert {(r.condition, r.outcome) for r in rows} == {
    ("outside_open", "bullish"), ("outside_open", "bearish"),
    ("bullish", "continuation"), ("bullish", "reversal"),
    ("bearish", "continuation"), ("bearish", "reversal"),
  }


def test_data_range() -> None:
  """data_range spans from the first to the last resolved session date."""
  result = _stat().compute(make_candles(_SEQ))
  assert result.instruments["NQ"]["daily"].data_range == ["2024-01-02", "2024-01-09"]


# ===========================================================================
# Outside-open definition: strict inequality (opening at the level is not outside)
# ===========================================================================

def test_opening_at_prior_level_is_not_outside() -> None:
  """open == prev_high (and inside on the low side) → neither bullish nor bearish."""
  days = [
    {"date": "2024-01-02", "open": 100.0, "close": 100.0, "high": 110.0, "low": 90.0},
    # Day 1 opens exactly at the prior high: 110 not > 110, and 110 not < 90.
    {"date": "2024-01-03", "open": 110.0, "close": 110.0, "high": 115.0, "low": 105.0},
  ]
  result = _stat().compute(make_candles(days))
  assert _row(result, "outside_open", "bullish").count == 0
  assert _row(result, "outside_open", "bearish").count == 0
  assert _row(result, "outside_open", "bullish").total == 1


def test_reentry_uses_strict_inequality() -> None:
  """day_low == prev_high is NOT a re-entry → counted as continuation (held outside)."""
  days = [
    {"date": "2024-01-02", "open": 100.0, "close": 100.0, "high": 110.0, "low": 90.0},
    # Bullish outside (open 115 > 110); day_low touches the prior high exactly.
    {"date": "2024-01-03", "open": 115.0, "close": 118.0, "high": 120.0, "low": 110.0},
  ]
  result = _stat().compute(make_candles(days))
  cont = _row(result, "bullish", "continuation")
  rev = _row(result, "bullish", "reversal")
  assert cont.total == 1
  assert cont.count == 1
  assert rev.count == 0


# ===========================================================================
# Slices: weekday and prev_candle
# ===========================================================================

def test_declared_slices_present() -> None:
  """The standard module declares the weekday and prev_candle slices."""
  slices = _stat().compute(make_candles(_SEQ)).instruments["NQ"]["daily"].slices
  assert set(slices) == {"weekday", "prev_candle"}


def test_weekday_slice_groups_sum_to_countable() -> None:
  """Across weekday groups, bullish-open counts sum to the overall bullish count."""
  result = _stat().compute(make_candles(_SEQ))
  wk = result.instruments["NQ"]["daily"].slices["weekday"]
  total_bull = 0
  for grp in wk.groups.values():
    for r in grp.results:
      if r.condition == "outside_open" and r.outcome == "bullish":
        total_bull += r.count
  overall = _row(result, "outside_open", "bullish").count
  assert total_bull == overall


def test_prev_candle_slice_splits_green_red() -> None:
  """prev_candle slice exposes green/red groups for the prior session color."""
  result = _stat().compute(make_candles(_SEQ))
  pc = result.instruments["NQ"]["daily"].slices["prev_candle"]
  assert set(pc.groups).issubset({"green", "red"})
  assert len(pc.groups) >= 1


# ===========================================================================
# Baseline deterministic and embedded
# ===========================================================================

def _long_seq() -> pd.DataFrame:
  """~250 weekdays with frequent outside opens (alternating gap up / gap down)."""
  dates: list[str] = []
  d = pd.Timestamp("2020-01-01", tz=_NY)
  while len(dates) < 250:
    if d.weekday() < 5:
      dates.append(d.strftime("%Y-%m-%d"))
    d += pd.Timedelta(days=1)

  days = []
  base = 100.0
  for i, date in enumerate(dates):
    hi, lo = base + 10.0, base - 10.0
    if i % 2 == 0:
      # Gap well above the prior day's high (prior high ≈ base + 9).
      open_, close = base + 15.0, base + 5.0
    else:
      # Gap well below the prior day's low (prior low ≈ base - 11).
      open_, close = base - 15.0, base - 5.0
    days.append({"date": date, "open": open_, "close": close, "high": hi, "low": lo})
    base += 1.0

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


def test_baseline_partitions_continuation_reversal() -> None:
  """Baseline continuation/reversal still partition each outside condition."""
  rows = _stat().baseline(_long_seq(), seed=11)
  by_key = {(r.condition, r.outcome): r for r in rows}
  for cond in ("bullish", "bearish"):
    cont = by_key[(cond, "continuation")]
    rev = by_key[(cond, "reversal")]
    assert cont.count + rev.count == cont.total == rev.total


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
  truncated = _make_truncated_day("2024-01-10")
  combined = (
    pd.concat([base_df, truncated], ignore_index=True)
    .sort_values("timestamp")
    .reset_index(drop=True)
  )
  total_with = stat.compute(combined).instruments["NQ"]["daily"].total_samples
  assert total_with == total_base == 6


def test_pending_day_absent_from_day_table() -> None:
  """build_day_table excludes the pending (truncated) day entirely."""
  day_table = _stat().build_day_table(_make_truncated_day("2024-01-10"))
  pending = pd.Timestamp("2024-01-10", tz=_NY).normalize()
  assert pending not in day_table.index


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
  assert set(result.labels.conditions) == {"outside_open", "bullish", "bearish"}
  assert set(result.labels.outcomes) == {"bullish", "bearish", "continuation", "reversal"}


# ===========================================================================
# stat_name and write_results round-trip
# ===========================================================================

def test_stat_name() -> None:
  assert _stat().compute(_empty_df()).stat_name == "outside_days"


def test_write_results_round_trip(tmp_path: Path) -> None:
  """write_results produces outside_days.json that re-validates correctly."""
  result = _stat().compute(make_candles(_SEQ))
  written = write_results(result, results_dir=tmp_path)
  assert written.name == "outside_days.json"

  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)
  assert validated.instruments["NQ"]["daily"].total_samples == 6


def test_write_results_french_accent_literal(tmp_path: Path) -> None:
  """French text is stored as literal UTF-8 characters, not escaped unicode."""
  result = _stat().compute(_empty_df())
  written = write_results(result, results_dir=tmp_path)
  raw = written.read_text(encoding="utf-8")
  # The French definition contains "précédent" (with é).
  assert "é" in raw
  assert "\\u00e9" not in raw


# ===========================================================================
# classify_samples
#
# Reusing _SEQ (6 resolved days; idx 0 is the anchor, no prior day):
#   idx 1 (2024-01-03): bullish, continuation
#   idx 2 (2024-01-04): bullish, reversal
#   idx 3 (2024-01-05): bearish, continuation
#   idx 4 (2024-01-08): bearish, reversal
#   idx 5 (2024-01-09): neither (inside)
# ===========================================================================

def test_classify_samples_exact_list() -> None:
  """classify_samples emits one outside_open sample per countable day, plus a
  tier-2 sample for each bullish/bearish day."""
  stat = _stat()
  day_table = stat.build_day_table(make_candles(_SEQ))
  samples = stat.classify_samples(day_table)
  assert [(s.date, s.condition, s.outcome) for s in samples] == [
    ("2024-01-03", "outside_open", "bullish"),
    ("2024-01-03", "bullish", "continuation"),
    ("2024-01-04", "outside_open", "bullish"),
    ("2024-01-04", "bullish", "reversal"),
    ("2024-01-05", "outside_open", "bearish"),
    ("2024-01-05", "bearish", "continuation"),
    ("2024-01-08", "outside_open", "bearish"),
    ("2024-01-08", "bearish", "reversal"),
    ("2024-01-09", "outside_open", "neither"),
  ]
  assert all(s.value is None for s in samples)


def test_classify_samples_matches_compute_rows_counts() -> None:
  """For every StatResultRow, the matching SampleRow count equals r.count; the
  outside_open condition's total sample count equals its r.total (all three
  outcomes, including the synthetic 'neither', are emitted for every countable
  day)."""
  stat = _stat()
  day_table = stat.build_day_table(make_candles(_SEQ))
  samples = stat.classify_samples(day_table)
  rows = stat.compute_rows(day_table)
  for r in rows:
    n = sum(1 for s in samples if s.condition == r.condition and s.outcome == r.outcome)
    assert n == r.count
  outside_open_total = sum(1 for s in samples if s.condition == "outside_open")
  assert outside_open_total == 5


def test_classify_samples_empty_day_table() -> None:
  """Empty day_table -> classify_samples returns []."""
  stat = _stat()
  assert stat.classify_samples(stat.build_day_table(_empty_df())) == []


def test_classify_samples_excludes_first_day_and_pending_day() -> None:
  """The anchor day (no prior day) and a trailing pending day yield no samples."""
  base_df = make_candles(_SEQ)
  truncated = _make_truncated_day("2024-01-10")
  combined = (
    pd.concat([base_df, truncated], ignore_index=True)
    .sort_values("timestamp")
    .reset_index(drop=True)
  )
  stat = _stat()
  day_table = stat.build_day_table(combined)
  samples = stat.classify_samples(day_table)
  sample_dates = {s.date for s in samples}
  assert "2024-01-02" not in sample_dates  # anchor (no prior day)
  assert "2024-01-10" not in sample_dates  # trailing pending day
  assert sample_dates == {"2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"}
