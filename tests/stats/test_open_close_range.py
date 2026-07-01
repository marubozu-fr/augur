"""Tests for stats.open_close_range.standard.

All data is synthetic — no real market files required. Expected values are
hand-calculated before each assertion.

Framing:
  - Condition ``open_close_range``: how often a session closes within a given
    percentage range of its open.
  - For each resolved session::
        oc_move_pct = abs(session_close - session_open) / session_open * 100
  - ``within``:  oc_move_pct <= range_percentage  (boundary == threshold → within).
  - ``outside``: oc_move_pct > range_percentage.

Every resolved session is countable — there is no warm-up window — so each row's
``total`` equals ``total_samples``.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatRunResult, write_results
from stats.open_close_range.standard import OpenCloseRange
from tests.stats.range_helpers import (
  _NY,
  _TEST_CONFIG,
  _empty_df,
  _make_truncated_day,
  make_candles,
)


def _stat(range_percentage: float = 1.0) -> OpenCloseRange:
  return OpenCloseRange(
    instrument="NQ", config=_TEST_CONFIG, range_percentage=range_percentage
  )


def _row(result: StatRunResult, outcome: str) -> "pd.Series | object":
  for r in result.instruments["NQ"]["daily"].results:
    if r.condition == "open_close_range" and r.outcome == outcome:
      return r
  raise KeyError(outcome)


def _spec(date: str, open_: float, close: float) -> dict:
  """A day spec; high/low straddle the open/close (irrelevant to this stat)."""
  return {
    "date": date,
    "open": open_,
    "close": close,
    "high": max(open_, close) + 1.0,
    "low": min(open_, close) - 1.0,
  }


# ===========================================================================
# Main fixture (range_percentage default = 1.0)
#
#  idx  date        open   close   move%   within(<=1)?  color (close>=open)
#   0   2024-01-02  100    100.5    0.5       within        green
#   1   2024-01-03  100    101.0    1.0       within        green   (boundary)
#   2   2024-01-04  100    102.0    2.0       outside       green
#   3   2024-01-05  100     99.0    1.0       within        red     (boundary)
#   4   2024-01-08  100     97.0    3.0       outside       red
#   5   2024-01-09  100    100.0    0.0       within        green   (==open → green)
#
#  within = {0,1,3,5} = 4 ; outside = {2,4} = 2 ; total = 6
#  green  = {0,1,2,5} = 4 (within 3: 0,1,5 / outside 1: 2)
#  red    = {3,4}     = 2 (within 1: 3   / outside 1: 4)
# ===========================================================================

_SESSIONS = [
  _spec("2024-01-02", 100.0, 100.5),
  _spec("2024-01-03", 100.0, 101.0),
  _spec("2024-01-04", 100.0, 102.0),
  _spec("2024-01-05", 100.0, 99.0),
  _spec("2024-01-08", 100.0, 97.0),
  _spec("2024-01-09", 100.0, 100.0),
]


# ===========================================================================
# 1. Counts and probability
# ===========================================================================

def test_within_count_and_probability() -> None:
  """4 of 6 sessions close within 1% of their open → P=4/6."""
  row = _row(_stat().compute(make_candles(_SESSIONS)), "within")
  assert row.count == 4
  assert row.total == 6
  assert row.probability == pytest.approx(4 / 6)


def test_outside_count_and_probability() -> None:
  """2 of 6 sessions close outside the 1% band → P=2/6."""
  row = _row(_stat().compute(make_candles(_SESSIONS)), "outside")
  assert row.count == 2
  assert row.total == 6
  assert row.probability == pytest.approx(2 / 6)


def test_within_outside_partition() -> None:
  """within.count + outside.count == total for every data set."""
  result = _stat().compute(make_candles(_SESSIONS))
  win = _row(result, "within")
  out = _row(result, "outside")
  assert win.count + out.count == win.total == out.total == 6


def test_total_samples_equals_row_total() -> None:
  """No warm-up: total_samples == each row's total (all sessions countable)."""
  result = _stat().compute(make_candles(_SESSIONS))
  assert result.instruments["NQ"]["daily"].total_samples == 6
  assert _row(result, "within").total == 6


# ===========================================================================
# 2. Boundary: move exactly == range_percentage → within (<= is inclusive)
# ===========================================================================

def test_boundary_move_equals_threshold_is_within() -> None:
  """A move exactly equal to range_percentage is classified within (inclusive)."""
  # Two sessions, both move exactly 1.0% (one up, one down) → both within.
  sessions = [
    _spec("2024-01-02", 100.0, 101.0),  # +1.0%
    _spec("2024-01-03", 100.0, 99.0),   # -1.0%
  ]
  result = _stat(range_percentage=1.0).compute(make_candles(sessions))
  assert _row(result, "within").count == 2
  assert _row(result, "outside").count == 0


# ===========================================================================
# 3. oc_move_pct column correctness (uses the open, not the close, as base)
# ===========================================================================

def test_oc_move_pct_column_values() -> None:
  """oc_move_pct = abs(close - open) / open * 100 for each session."""
  table = _stat().build_day_table(make_candles(_SESSIONS))
  expected = {
    "2024-01-02": 0.5,
    "2024-01-03": 1.0,
    "2024-01-04": 2.0,
    "2024-01-05": 1.0,
    "2024-01-08": 3.0,
    "2024-01-09": 0.0,
  }
  for date, exp in expected.items():
    idx = pd.Timestamp(date, tz=_NY).normalize()
    assert table.loc[idx, "oc_move_pct"] == pytest.approx(exp)


def test_session_green_column() -> None:
  """session_green = (close >= open); a flat session (close==open) is green."""
  table = _stat().build_day_table(make_candles(_SESSIONS))
  assert bool(table.loc[pd.Timestamp("2024-01-09", tz=_NY).normalize(), "session_green"]) is True
  assert bool(table.loc[pd.Timestamp("2024-01-05", tz=_NY).normalize(), "session_green"]) is False


# ===========================================================================
# 4. All-within / all-outside edge cases
# ===========================================================================

def test_all_within() -> None:
  """Every session inside the band → within=total, outside=0."""
  sessions = [
    _spec("2024-01-02", 100.0, 100.2),
    _spec("2024-01-03", 100.0, 99.8),
  ]
  result = _stat(range_percentage=1.0).compute(make_candles(sessions))
  assert _row(result, "within").count == 2
  assert _row(result, "within").probability == pytest.approx(1.0)
  assert _row(result, "outside").count == 0


def test_all_outside() -> None:
  """Every session beyond the band → outside=total, within=0."""
  sessions = [
    _spec("2024-01-02", 100.0, 105.0),
    _spec("2024-01-03", 100.0, 95.0),
  ]
  result = _stat(range_percentage=1.0).compute(make_candles(sessions))
  assert _row(result, "outside").count == 2
  assert _row(result, "outside").probability == pytest.approx(1.0)
  assert _row(result, "within").count == 0


# ===========================================================================
# 5. Custom range_percentage
# ===========================================================================

def test_custom_range_percentage_tight() -> None:
  """range_percentage=0.5: only moves <=0.5% are within → {0.5, 0.0} = 2."""
  result = _stat(range_percentage=0.5).compute(make_candles(_SESSIONS))
  assert _row(result, "within").count == 2  # idx 0 (0.5) and idx 5 (0.0)
  assert _row(result, "outside").count == 4


def test_custom_range_percentage_wide() -> None:
  """range_percentage=2.0: moves <=2.0% are within → all but the 3.0% move = 5."""
  result = _stat(range_percentage=2.0).compute(make_candles(_SESSIONS))
  assert _row(result, "within").count == 5  # all except idx 4 (3.0%)
  assert _row(result, "outside").count == 1


def test_range_percentage_stored_on_instance() -> None:
  """Different thresholds give different within counts on the same data."""
  df = make_candles(_SESSIONS)
  tight = _row(_stat(range_percentage=0.5).compute(df), "within").count
  wide = _row(_stat(range_percentage=2.0).compute(df), "within").count
  assert tight == 2
  assert wide == 5
  assert tight != wide


def test_non_positive_range_percentage_rejected() -> None:
  """range_percentage must be strictly positive."""
  with pytest.raises(ValueError):
    OpenCloseRange(instrument="NQ", config=_TEST_CONFIG, range_percentage=0.0)


# ===========================================================================
# 6. Weekday slice
# ===========================================================================

def test_weekday_slice_present() -> None:
  result = _stat().compute(make_candles(_SESSIONS))
  assert "weekday" in result.instruments["NQ"]["daily"].slices


def test_weekday_slice_within_counts_sum_to_overall() -> None:
  """Sum of within.count across weekday groups == overall within.count."""
  result = _stat().compute(make_candles(_SESSIONS))
  overall = _row(result, "within").count
  wk = result.instruments["NQ"]["daily"].slices["weekday"]
  total = sum(
    r.count
    for grp in wk.groups.values()
    for r in grp.results
    if r.outcome == "within"
  )
  assert total == overall == 4


def test_weekday_slice_partitions_per_group() -> None:
  """Within each weekday group, within + outside == group total."""
  result = _stat().compute(make_candles(_SESSIONS))
  wk = result.instruments["NQ"]["daily"].slices["weekday"]
  for key, grp in wk.groups.items():
    by_outcome = {r.outcome: r for r in grp.results}
    win, out = by_outcome["within"], by_outcome["outside"]
    assert win.total == out.total, f"group {key}: totals differ"
    assert win.count + out.count == win.total, f"group {key}: partition broken"


# ===========================================================================
# 7. Close (session color) slice — the day_type dimension
# ===========================================================================

def test_close_slice_present() -> None:
  result = _stat().compute(make_candles(_SESSIONS))
  assert "close" in result.instruments["NQ"]["daily"].slices


def test_close_slice_green_red_groups() -> None:
  """Green / red groups carry the expected within counts.

  green = {0,1,2,5}: within {0,1,5} = 3, outside {2} = 1, total 4.
  red   = {3,4}:     within {3}     = 1, outside {4} = 1, total 2.
  """
  result = _stat().compute(make_candles(_SESSIONS))
  groups = result.instruments["NQ"]["daily"].slices["close"].groups
  assert set(groups) == {"green", "red"}

  green = {r.outcome: r for r in groups["green"].results}
  assert green["within"].count == 3
  assert green["outside"].count == 1
  assert green["within"].total == 4

  red = {r.outcome: r for r in groups["red"].results}
  assert red["within"].count == 1
  assert red["outside"].count == 1
  assert red["within"].total == 2


def test_close_slice_within_counts_sum_to_overall() -> None:
  """Green within + red within == overall within."""
  result = _stat().compute(make_candles(_SESSIONS))
  overall = _row(result, "within").count
  groups = result.instruments["NQ"]["daily"].slices["close"].groups
  total = sum(
    r.count for grp in groups.values() for r in grp.results if r.outcome == "within"
  )
  assert total == overall == 4


# ===========================================================================
# 8. Baseline — determinism, count preservation, non-degeneracy
#
# A "trending" sequence with rising opens and small same-day moves: same-session
# moves are tiny (within), but the baseline pairs each open with an unrelated
# (far-away, differently-priced) close, producing large moves (outside). This
# exercises the close-permutation baseline and its documented behaviour.
# ===========================================================================

def _trending_seq() -> pd.DataFrame:
  """~40 weekdays: open trends 100 -> 139, close = open + 0.1 (0.1% move/day)."""
  dates: list[str] = []
  d = pd.Timestamp("2020-01-01", tz=_NY)
  while len(dates) < 40:
    if d.weekday() < 5:
      dates.append(d.strftime("%Y-%m-%d"))
    d += pd.Timedelta(days=1)
  return make_candles(
    [_spec(date, 100.0 + i, 100.0 + i + 0.1) for i, date in enumerate(dates)]
  )


def test_baseline_deterministic() -> None:
  """baseline_rows(seed=42) is identical on two successive calls."""
  stat = _stat()
  table = stat.build_day_table(_trending_seq())
  a = stat.baseline_rows(table, seed=42)
  b = stat.baseline_rows(table, seed=42)
  for ra, rb in zip(a, b):
    assert (ra.condition, ra.outcome) == (rb.condition, rb.outcome)
    assert ra.count == rb.count
    assert ra.total == rb.total
    assert ra.probability == pytest.approx(rb.probability)


def test_baseline_preserves_total() -> None:
  """The close permutation preserves the countable total exactly."""
  stat = _stat()
  table = stat.build_day_table(_trending_seq())
  rows = stat.baseline_rows(table, seed=42)
  assert rows[0].total == len(table) == 40


def test_baseline_breaks_same_session_link() -> None:
  """On a trending series, the actual within-rate far exceeds the baseline.

  Same-session moves are ~0.1% (all within the 1% band) → actual within ≈ 1.0.
  Random open/close pairings span the whole price range → mostly outside, so the
  baseline within-rate is much lower.
  """
  stat = _stat(range_percentage=1.0)
  df = _trending_seq()
  result = stat.compute(df)
  actual_within = _row(result, "within").probability
  baseline_within = _row(result, "within").baseline_prob
  assert actual_within == pytest.approx(1.0)
  assert baseline_within < actual_within


def test_compute_reproducible() -> None:
  """compute() with same input and seed → identical serialized output."""
  stat = _stat()
  df = _trending_seq()
  assert stat.compute(df, seed=42).model_dump_json() == stat.compute(df, seed=42).model_dump_json()


def test_baseline_embedded_has_positive_n() -> None:
  """After compute(), rows carry a positive baseline_n when there are samples."""
  result = _stat().compute(make_candles(_SESSIONS))
  for row in result.instruments["NQ"]["daily"].results:
    assert row.baseline_n == 6


# ===========================================================================
# 9. Pending discipline / empty input
# ===========================================================================

def test_empty_dataframe_build_day_table() -> None:
  assert _stat().build_day_table(_empty_df()).empty


def test_empty_dataframe_compute_rows_both_zero() -> None:
  rows = _stat().compute_rows(
    pd.DataFrame(columns=["session_open", "session_close", "oc_move_pct", "session_green"])
  )
  by_outcome = {r.outcome: r for r in rows}
  assert set(by_outcome) == {"within", "outside"}
  for r in rows:
    assert r.count == 0
    assert r.total == 0
    assert r.probability == pytest.approx(0.0)


def test_empty_dataframe_full_compute() -> None:
  result = _stat().compute(_empty_df())
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 0
  assert tf.data_range == []
  for row in tf.results:
    assert row.count == 0
    assert row.total == 0


def test_pending_day_excluded() -> None:
  """A truncated/early-close day is excluded from total_samples and the table."""
  stat = _stat()
  base_df = make_candles(_SESSIONS)
  base_total = stat.compute(base_df).instruments["NQ"]["daily"].total_samples
  truncated = _make_truncated_day("2024-01-10")
  combined = (
    pd.concat([base_df, truncated], ignore_index=True)
    .sort_values("timestamp")
    .reset_index(drop=True)
  )
  assert stat.compute(combined).instruments["NQ"]["daily"].total_samples == base_total == 6
  pending = pd.Timestamp("2024-01-10", tz=_NY).normalize()
  assert pending not in stat.build_day_table(combined).index


# ===========================================================================
# 10. Structure / data_range / i18n / write round-trip
# ===========================================================================

def test_exactly_two_rows() -> None:
  rows = _stat().compute(make_candles(_SESSIONS)).instruments["NQ"]["daily"].results
  assert {(r.condition, r.outcome) for r in rows} == {
    ("open_close_range", "within"),
    ("open_close_range", "outside"),
  }


def test_data_range_spans_all_resolved_sessions() -> None:
  result = _stat().compute(make_candles(_SESSIONS))
  assert result.instruments["NQ"]["daily"].data_range == ["2024-01-02", "2024-01-09"]


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
  assert set(result.labels.conditions) == {"open_close_range"}
  assert set(result.labels.outcomes) == {"within", "outside"}


def test_stat_name() -> None:
  assert _stat().compute(_empty_df()).stat_name == "open_close_range"


def test_write_results_round_trip(tmp_path: Path) -> None:
  result = _stat().compute(make_candles(_SESSIONS))
  written = write_results(result, results_dir=tmp_path)
  assert written.name == "open_close_range.json"
  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)
  assert validated.instruments["NQ"]["daily"].total_samples == 6


def test_write_results_french_accent_literal(tmp_path: Path) -> None:
  """French text is stored as literal UTF-8 characters, not escaped unicode."""
  result = _stat().compute(_empty_df())
  raw = write_results(result, results_dir=tmp_path).read_text(encoding="utf-8")
  assert "clôture" in raw


# ===========================================================================
# classify_samples
# ===========================================================================

def test_classify_samples_exact_list() -> None:
  """One SampleRow per session, matching the hand-computed within/outside split."""
  stat = _stat()
  table = stat.build_day_table(make_candles(_SESSIONS))
  samples = stat.classify_samples(table)
  assert [(s.date, s.condition, s.outcome) for s in samples] == [
    ("2024-01-02", "open_close_range", "within"),
    ("2024-01-03", "open_close_range", "within"),
    ("2024-01-04", "open_close_range", "outside"),
    ("2024-01-05", "open_close_range", "within"),
    ("2024-01-08", "open_close_range", "outside"),
    ("2024-01-09", "open_close_range", "within"),
  ]


def test_classify_samples_matches_compute_rows_counts() -> None:
  """The invariant: sample counts per (condition, outcome) equal compute_rows' count/total."""
  stat = _stat()
  table = stat.build_day_table(make_candles(_SESSIONS))
  samples = stat.classify_samples(table)
  rows = stat.compute_rows(table)
  for r in rows:
    matching = [s for s in samples if s.condition == r.condition and s.outcome == r.outcome]
    assert len(matching) == r.count
  total_all = len({s.date for s in samples})
  assert total_all == rows[0].total == 6


def test_classify_samples_empty_day_table() -> None:
  assert _stat().classify_samples(pd.DataFrame(
    columns=["session_open", "session_close", "oc_move_pct", "session_green"]
  )) == []


def test_classify_samples_excludes_pending_day() -> None:
  """A truncated/unresolved day never enters the day_table, so it has no sample."""
  stat = _stat()
  base_df = make_candles(_SESSIONS)
  truncated = _make_truncated_day("2024-01-10")
  combined = (
    pd.concat([base_df, truncated], ignore_index=True)
    .sort_values("timestamp")
    .reset_index(drop=True)
  )
  table = stat.build_day_table(combined)
  samples = stat.classify_samples(table)
  assert "2024-01-10" not in {s.date for s in samples}
  assert len(samples) == 6
