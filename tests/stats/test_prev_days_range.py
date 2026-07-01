"""Tests for stats.prev_days_range.standard.

All data is synthetic — no real market files required. Expected values are
hand-calculated before each assertion.

Framing:
  - Tier 1 (condition ``prior_range``): break frequency. ``break_high`` counts
    days whose RTH high exceeds the prior resolved day's high; ``break_low``
    counts days whose RTH low falls below the prior day's low. total = countable
    days (those with a prior resolved day).
  - Tier 2 (conditions ``break_high`` / ``break_low``): directional
    follow-through (green = close >= open, red otherwise) over the break days.

The first resolved day has no prior day and is excluded from every denominator.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.prev_days_range.standard import PrevDaysRange

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


def _stat() -> PrevDaysRange:
  return PrevDaysRange(instrument="NQ", config=_TEST_CONFIG)


def _row(result: StatRunResult, condition: str, outcome: str) -> StatResultRow:
  for r in result.instruments["NQ"]["daily"].results:
    if r.condition == condition and r.outcome == outcome:
      return r
  raise KeyError((condition, outcome))


# ===========================================================================
# Core derivation (6 days)
#
#   idx  date        open  close  high   low   | prev_high prev_low | breaks   dir
#    0   2024-01-02  100   105    110    95     |   —         —      | excluded  G
#    1   2024-01-03  104   108    115    100    |  110       95      | high      G
#    2   2024-01-04  108   102    112     90    |  115      100      | low       R
#    3   2024-01-05  101   106    118     85    |  112       90      | high+low  G
#    4   2024-01-08  106    99    109     95    |  118       85      | none      R
#    5   2024-01-09   99   103    120     80    |  109       95      | high+low  G
#
# Countable = idx 1..5 (5 days; idx 0 has no prior day).
#
# break_high days: 1(115>110), 3(118>112), 5(120>109)          → count=3
# break_low  days: 2(90<100), 3(85<90), 5(80<95)               → count=3
#
# Tier-1 (prior_range), total=5:
#   break_high: 3/5 = 0.6
#   break_low:  3/5 = 0.6
#
# Tier-2 follow-through:
#   break_high days {1:G, 3:G, 5:G} → green=3, red=0  (total 3)
#   break_low  days {2:R, 3:G, 5:G} → green=2, red=1  (total 3)
#
# total_samples = 6 (all resolved days, incl. idx 0)
# ===========================================================================

_SEQ = [
  {"date": "2024-01-02", "open": 100.0, "close": 105.0, "high": 110.0, "low": 95.0},
  {"date": "2024-01-03", "open": 104.0, "close": 108.0, "high": 115.0, "low": 100.0},
  {"date": "2024-01-04", "open": 108.0, "close": 102.0, "high": 112.0, "low": 90.0},
  {"date": "2024-01-05", "open": 101.0, "close": 106.0, "high": 118.0, "low": 85.0},
  {"date": "2024-01-08", "open": 106.0, "close": 99.0, "high": 109.0, "low": 95.0},
  {"date": "2024-01-09", "open": 99.0, "close": 103.0, "high": 120.0, "low": 80.0},
]


def test_total_samples_counts_all_resolved_days() -> None:
  """total_samples = 6 resolved sessions (the first one is still resolved)."""
  result = _stat().compute(make_candles(_SEQ))
  assert result.instruments["NQ"]["daily"].total_samples == 6


def test_break_high_frequency() -> None:
  """3 of 5 countable days break the prior high → P=0.6."""
  row = _row(_stat().compute(make_candles(_SEQ)), "prior_range", "break_high")
  assert row.total == 5
  assert row.count == 3
  assert row.probability == pytest.approx(0.6)


def test_break_low_frequency() -> None:
  """3 of 5 countable days break the prior low → P=0.6."""
  row = _row(_stat().compute(make_candles(_SEQ)), "prior_range", "break_low")
  assert row.total == 5
  assert row.count == 3
  assert row.probability == pytest.approx(0.6)


def test_high_follow_through() -> None:
  """All 3 high-break days closed green → green=3/3, red=0/3."""
  result = _stat().compute(make_candles(_SEQ))
  green = _row(result, "break_high", "green")
  red = _row(result, "break_high", "red")
  assert green.total == red.total == 3
  assert green.count == 3
  assert green.probability == pytest.approx(1.0)
  assert red.count == 0
  assert red.probability == pytest.approx(0.0)


def test_low_follow_through() -> None:
  """Of 3 low-break days, 2 green and 1 red."""
  result = _stat().compute(make_candles(_SEQ))
  green = _row(result, "break_low", "green")
  red = _row(result, "break_low", "red")
  assert green.total == red.total == 3
  assert green.count == 2
  assert green.probability == pytest.approx(2 / 3)
  assert red.count == 1
  assert red.probability == pytest.approx(1 / 3)


def test_follow_through_partitions_each_break_condition() -> None:
  """green.count + red.count == total for each break condition."""
  result = _stat().compute(make_candles(_SEQ))
  for cond in ("break_high", "break_low"):
    green = _row(result, cond, "green")
    red = _row(result, cond, "red")
    assert green.count + red.count == green.total == red.total


def test_first_day_excluded_from_denominators() -> None:
  """Break-frequency total = 5 = n_resolved - 1 (first day has no prior day)."""
  result = _stat().compute(make_candles(_SEQ))
  assert _row(result, "prior_range", "break_high").total == 6 - 1


def test_six_rows_only() -> None:
  """Exactly the two break-frequency rows plus the four follow-through rows."""
  rows = _stat().compute(make_candles(_SEQ)).instruments["NQ"]["daily"].results
  assert {(r.condition, r.outcome) for r in rows} == {
    ("prior_range", "break_high"), ("prior_range", "break_low"),
    ("break_high", "green"), ("break_high", "red"),
    ("break_low", "green"), ("break_low", "red"),
  }


def test_data_range() -> None:
  """data_range spans from the first to the last resolved session date."""
  result = _stat().compute(make_candles(_SEQ))
  assert result.instruments["NQ"]["daily"].data_range == ["2024-01-02", "2024-01-09"]


# ===========================================================================
# Break definition: strict inequality (touching the level is not a break)
# ===========================================================================

def test_touching_prior_level_is_not_a_break() -> None:
  """day_high == prev_high and day_low == prev_low → neither is a break."""
  days = [
    {"date": "2024-01-02", "open": 100.0, "close": 100.0, "high": 110.0, "low": 90.0},
    # Day 1 exactly retests the prior extremes: 110 not > 110, 90 not < 90.
    {"date": "2024-01-03", "open": 100.0, "close": 100.0, "high": 110.0, "low": 90.0},
  ]
  result = _stat().compute(make_candles(days))
  assert _row(result, "prior_range", "break_high").count == 0
  assert _row(result, "prior_range", "break_low").count == 0
  assert _row(result, "prior_range", "break_high").total == 1


def test_break_both_counts_in_both_rows() -> None:
  """A day that breaks both high and low is counted in each break row."""
  days = [
    {"date": "2024-01-02", "open": 100.0, "close": 100.0, "high": 105.0, "low": 95.0},
    # Day 1 engulfs the prior range entirely.
    {"date": "2024-01-03", "open": 100.0, "close": 102.0, "high": 110.0, "low": 90.0},
  ]
  result = _stat().compute(make_candles(days))
  assert _row(result, "prior_range", "break_high").count == 1
  assert _row(result, "prior_range", "break_low").count == 1
  # Single countable day, closed green → high follow-through green.
  assert _row(result, "break_high", "green").count == 1
  assert _row(result, "break_low", "green").count == 1


def test_flat_session_classified_green() -> None:
  """session_close == session_open → green day (>= is green)."""
  days = [
    {"date": "2024-01-02", "open": 100.0, "close": 100.0, "high": 105.0, "low": 95.0},
    # Flat close, breaks the high only.
    {"date": "2024-01-03", "open": 100.0, "close": 100.0, "high": 110.0, "low": 96.0},
  ]
  result = _stat().compute(make_candles(days))
  green = _row(result, "break_high", "green")
  assert green.total == 1
  assert green.count == 1


# ===========================================================================
# Slices: weekday and prev_candle
# ===========================================================================

def test_declared_slices_present() -> None:
  """The standard module declares the weekday and prev_candle slices."""
  slices = _stat().compute(make_candles(_SEQ)).instruments["NQ"]["daily"].slices
  assert set(slices) == {"weekday", "prev_candle"}


def test_weekday_slice_groups_sum_to_countable() -> None:
  """Across weekday groups, break_high counts sum to the overall break_high count."""
  result = _stat().compute(make_candles(_SEQ))
  wk = result.instruments["NQ"]["daily"].slices["weekday"]
  total_high = 0
  for grp in wk.groups.values():
    for r in grp.results:
      if r.condition == "prior_range" and r.outcome == "break_high":
        total_high += r.count
  overall = _row(result, "prior_range", "break_high").count
  assert total_high == overall


def test_prev_candle_slice_splits_green_red() -> None:
  """prev_candle slice exposes green/red groups for the prior session color."""
  result = _stat().compute(make_candles(_SEQ))
  pc = result.instruments["NQ"]["daily"].slices["prev_candle"]
  assert set(pc.groups).issubset({"green", "red"})
  assert len(pc.groups) >= 1


# ===========================================================================
# Baseline ≈ neutral and deterministic
# ===========================================================================

def _long_seq() -> pd.DataFrame:
  """~250 weekdays with a strong, predictable expanding/contracting pattern."""
  dates: list[str] = []
  d = pd.Timestamp("2020-01-01", tz=_NY)
  while len(dates) < 250:
    if d.weekday() < 5:
      dates.append(d.strftime("%Y-%m-%d"))
    d += pd.Timedelta(days=1)

  days = []
  base = 100.0
  for i, date in enumerate(dates):
    # Alternate wide and narrow days so breaks of the prior range are frequent
    # and follow-through direction is strongly tied to the break.
    if i % 2 == 0:
      hi, lo = base + 20.0, base - 20.0
      open_, close = base - 5.0, base + 15.0  # green
    else:
      hi, lo = base + 5.0, base - 5.0
      open_, close = base + 2.0, base - 2.0   # red
    days.append({"date": date, "open": open_, "close": close, "high": hi, "low": lo})
    base += 1.0

  return make_candles(days)


def test_baseline_follow_through_near_fifty_percent() -> None:
  """Follow-through baseline (randomized day color) is near 0.5."""
  rows = _stat().baseline(_long_seq(), seed=42)
  for r in rows:
    if r.condition in ("break_high", "break_low") and r.total > 0:
      assert r.probability == pytest.approx(0.5, abs=0.15)


def test_baseline_deterministic() -> None:
  """Same seed produces identical baseline results across two calls."""
  stat = _stat()
  df = _long_seq()
  rows_a = stat.baseline(df, seed=7)
  rows_b = stat.baseline(df, seed=7)
  for a, b in zip(rows_a, rows_b):
    assert (a.condition, a.outcome) == (b.condition, b.outcome)
    assert a.probability == pytest.approx(b.probability)


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
  assert set(result.labels.conditions) == {"prior_range", "break_high", "break_low"}
  assert set(result.labels.outcomes) == {"break_high", "break_low", "green", "red"}


# ===========================================================================
# stat_name and write_results round-trip
# ===========================================================================

def test_stat_name() -> None:
  assert _stat().compute(_empty_df()).stat_name == "prev_days_range"


def test_write_results_round_trip(tmp_path: Path) -> None:
  """write_results produces prev_days_range.json that re-validates correctly."""
  result = _stat().compute(make_candles(_SEQ))
  written = write_results(result, results_dir=tmp_path)
  assert written.name == "prev_days_range.json"

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
# ===========================================================================

def test_classify_samples_exact_list() -> None:
  """Exact per-day SampleRows for the 6-day _SEQ fixture (see derivation above).

  idx0 (2024-01-02) is excluded (no prior day). idx1 breaks the high only
  (green). idx2 breaks the low only (red). idx3 and idx5 break both (green).
  idx4 breaks neither and contributes nothing.
  """
  stat = _stat()
  day_table = stat.build_day_table(make_candles(_SEQ))
  samples = stat.classify_samples(day_table)
  assert [(s.date, s.condition, s.outcome) for s in samples] == [
    ("2024-01-03", "prior_range", "break_high"),
    ("2024-01-03", "break_high", "green"),
    ("2024-01-04", "prior_range", "break_low"),
    ("2024-01-04", "break_low", "red"),
    ("2024-01-05", "prior_range", "break_high"),
    ("2024-01-05", "break_high", "green"),
    ("2024-01-05", "prior_range", "break_low"),
    ("2024-01-05", "break_low", "green"),
    ("2024-01-09", "prior_range", "break_high"),
    ("2024-01-09", "break_high", "green"),
    ("2024-01-09", "prior_range", "break_low"),
    ("2024-01-09", "break_low", "green"),
  ]
  assert all(s.value is None for s in samples)


def test_classify_samples_excludes_first_day() -> None:
  """The first resolved day (no prior day) yields no samples."""
  stat = _stat()
  day_table = stat.build_day_table(make_candles(_SEQ))
  samples = stat.classify_samples(day_table)
  sample_dates = {s.date for s in samples}
  assert "2024-01-02" not in sample_dates


def test_classify_samples_no_break_day_contributes_nothing() -> None:
  """idx4 (2024-01-08), which breaks neither, produces zero samples."""
  stat = _stat()
  day_table = stat.build_day_table(make_candles(_SEQ))
  samples = stat.classify_samples(day_table)
  assert not any(s.date == "2024-01-08" for s in samples)


def test_classify_samples_empty_day_table() -> None:
  """Empty day_table -> classify_samples returns []."""
  stat = _stat()
  assert stat.classify_samples(stat.build_day_table(_empty_df())) == []


def test_classify_samples_matches_compute_rows_counts() -> None:
  """For every (condition, outcome), SampleRow count == compute_rows count.

  This is the core invariant: covers tier 1 (independent break rates) and
  tier 2 (follow-through partitions) over a long, varied synthetic sequence.
  """
  stat = _stat()
  day_table = stat.build_day_table(_long_seq())
  samples = stat.classify_samples(day_table)
  rows = stat.compute_rows(day_table)
  for row in rows:
    n_samples = len(
      [s for s in samples if s.condition == row.condition and s.outcome == row.outcome]
    )
    assert n_samples == row.count, (
      f"{row.condition}/{row.outcome}: samples={n_samples}, expected count={row.count}"
    )


def test_classify_samples_tier1_denominator_matches_countable_n() -> None:
  """Days producing a tier-1 break_high or break_low sample never exceed
  the countable day count (prior_range's shared total in compute_rows)."""
  stat = _stat()
  day_table = stat.build_day_table(_long_seq())
  samples = stat.classify_samples(day_table)
  rows = stat.compute_rows(day_table)
  countable_n = next(
    r.total for r in rows if r.condition == "prior_range" and r.outcome == "break_high"
  )
  tier1_dates = {s.date for s in samples if s.condition == "prior_range"}
  assert len(tier1_dates) <= countable_n
