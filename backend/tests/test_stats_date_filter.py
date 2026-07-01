"""Unit tests for backend.app.services.stats_date_filter (issue #179).

All test data is synthetic and hand-calculated in comments before each
assertion, per project convention. These are pure-function tests: no
FastAPI app, no filesystem, no StatsLoader involved.
"""

import pytest

from backend.app.services.stats_date_filter import (
  build_filtered_timeframe_result,
  filter_samples,
  has_undeclared_agg_metadata,
  reaggregate,
  validate_date_params,
)
from stats.base import SampleRow, StatResultRow, TimeframeResult


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _month_samples() -> list[SampleRow]:
  """10 samples, one per month Jan-Oct 2023, all condition 'cond_a'.

  Outcome alternates: odd months (1,3,5,7,9) -> 'green', even months
  (2,4,6,8,10) -> 'red'. Dates are the first of each month so month-aligned
  bounds are easy to reason about.
  """
  return [
    SampleRow(
      date=f"2023-{month:02d}-01",
      condition="cond_a",
      outcome="green" if month % 2 else "red",
    )
    for month in range(1, 11)
  ]


def _month_result_rows() -> list[StatResultRow]:
  """Original (condition, outcome) pairs matching _month_samples()."""
  return [
    StatResultRow(
      condition="cond_a", outcome="green", count=5, total=10,
      probability=0.5, baseline_prob=0.5, baseline_n=10,
    ),
    StatResultRow(
      condition="cond_a", outcome="red", count=5, total=10,
      probability=0.5, baseline_prob=0.5, baseline_n=10,
    ),
  ]


# ---------------------------------------------------------------------------
# filter_samples
# ---------------------------------------------------------------------------

def test_filter_samples_basic_range() -> None:
  # 10 samples Jan-Oct; filter to Mar-Jun inclusive -> Mar, Apr, May, Jun = 4
  samples = _month_samples()
  filtered = filter_samples(samples, start="2023-03-01", end="2023-06-01")

  assert [s.date for s in filtered] == [
    "2023-03-01", "2023-04-01", "2023-05-01", "2023-06-01",
  ]


def test_filter_samples_start_only() -> None:
  # start=Jun -> Jun, Jul, Aug, Sep, Oct = 5 samples
  samples = _month_samples()
  filtered = filter_samples(samples, start="2023-06-01", end=None)

  assert len(filtered) == 5
  assert filtered[0].date == "2023-06-01"
  assert filtered[-1].date == "2023-10-01"


def test_filter_samples_end_only() -> None:
  # end=Mar -> Jan, Feb, Mar = 3 samples
  samples = _month_samples()
  filtered = filter_samples(samples, start=None, end="2023-03-01")

  assert len(filtered) == 3
  assert filtered[0].date == "2023-01-01"
  assert filtered[-1].date == "2023-03-01"


def test_filter_samples_no_bounds_returns_all() -> None:
  samples = _month_samples()
  filtered = filter_samples(samples, start=None, end=None)

  assert filtered == samples


def test_filter_samples_empty_range_yields_no_samples() -> None:
  # Range entirely outside the data -> 0 samples
  samples = _month_samples()
  filtered = filter_samples(samples, start="2024-01-01", end="2024-12-31")

  assert filtered == []


# ---------------------------------------------------------------------------
# reaggregate — probability rows
# ---------------------------------------------------------------------------

def test_reaggregate_counts_and_probability() -> None:
  # Mar-Jun subset: green on Mar(3), May(5) -> count=2; red on Apr(4), Jun(6) -> count=2
  # total (both outcomes share condition 'cond_a') = 4
  # P(green) = 2/4 = 0.5, P(red) = 2/4 = 0.5
  samples = _month_samples()
  filtered = filter_samples(samples, start="2023-03-01", end="2023-06-01")
  rows = reaggregate(filtered, _month_result_rows())

  green = next(r for r in rows if r.outcome == "green")
  red = next(r for r in rows if r.outcome == "red")

  assert green.count == 2
  assert green.total == 4
  assert green.probability == pytest.approx(0.5)
  assert red.count == 2
  assert red.total == 4
  assert red.probability == pytest.approx(0.5)
  # baseline cannot be recomputed from samples alone
  assert green.baseline_prob == 0.0
  assert green.baseline_n == 0


def test_reaggregate_empty_subset_yields_zero_probability() -> None:
  # Range outside the data -> every row keeps its (condition, outcome) pair
  # with count=0, total=0, probability=0.0 (guarded division by zero)
  samples = _month_samples()
  filtered = filter_samples(samples, start="2024-01-01", end="2024-12-31")
  rows = reaggregate(filtered, _month_result_rows())

  assert len(rows) == 2
  for row in rows:
    assert row.count == 0
    assert row.total == 0
    assert row.probability == 0.0


def test_reaggregate_preserves_row_order() -> None:
  samples = _month_samples()
  filtered = filter_samples(samples, start="2023-03-01", end="2023-06-01")
  originals = _month_result_rows()
  rows = reaggregate(filtered, originals)

  assert [(r.condition, r.outcome) for r in rows] == [
    (o.condition, o.outcome) for o in originals
  ]


# ---------------------------------------------------------------------------
# reaggregate — magnitude rows (value/agg reaggregation)
# ---------------------------------------------------------------------------

def _magnitude_samples() -> list[SampleRow]:
  """5 samples Jan-May 2023, same (condition, outcome), values 10..50."""
  return [
    SampleRow(date=f"2023-{month:02d}-01", condition="cond_a", outcome="out_x", value=float(value))
    for month, value in zip(range(1, 6), [10.0, 20.0, 30.0, 40.0, 50.0])
  ]


def _magnitude_row(agg: str | None) -> StatResultRow:
  return StatResultRow(
    condition="cond_a", outcome="out_x", count=5, total=5,
    probability=0.0, baseline_prob=0.0, baseline_n=0,
    value=999.0, value_baseline=None, agg=agg,
  )


def test_reaggregate_magnitude_mean() -> None:
  # Filter to Feb-Apr -> values [20, 30, 40] -> mean = 30.0
  samples = _magnitude_samples()
  filtered = filter_samples(samples, start="2023-02-01", end="2023-04-01")
  rows = reaggregate(filtered, [_magnitude_row(agg="mean")])

  assert rows[0].value == pytest.approx(30.0)
  assert rows[0].agg == "mean"


def test_reaggregate_magnitude_max() -> None:
  # Filter to Feb-Apr -> values [20, 30, 40] -> max = 40.0
  samples = _magnitude_samples()
  filtered = filter_samples(samples, start="2023-02-01", end="2023-04-01")
  rows = reaggregate(filtered, [_magnitude_row(agg="max")])

  assert rows[0].value == pytest.approx(40.0)


def test_reaggregate_magnitude_min() -> None:
  # Filter to Feb-Apr -> values [20, 30, 40] -> min = 20.0
  samples = _magnitude_samples()
  filtered = filter_samples(samples, start="2023-02-01", end="2023-04-01")
  rows = reaggregate(filtered, [_magnitude_row(agg="min")])

  assert rows[0].value == pytest.approx(20.0)


def test_reaggregate_magnitude_median() -> None:
  # Filter to full range -> values [10, 20, 30, 40, 50] -> median = 30.0
  samples = _magnitude_samples()
  filtered = filter_samples(samples, start=None, end=None)
  rows = reaggregate(filtered, [_magnitude_row(agg="median")])

  assert rows[0].value == pytest.approx(30.0)


def test_reaggregate_non_decomposable_magnitude_yields_none() -> None:
  # agg=None (e.g. Pearson r) -> value stays None even though the row's
  # original value was set and matching samples carry no per-sample value.
  samples = [
    SampleRow(date="2023-01-01", condition="cond_a", outcome="out_x", value=None),
    SampleRow(date="2023-02-01", condition="cond_a", outcome="out_x", value=None),
  ]
  rows = reaggregate(samples, [_magnitude_row(agg=None)])

  assert rows[0].value is None
  assert rows[0].agg is None


# ---------------------------------------------------------------------------
# build_filtered_timeframe_result
# ---------------------------------------------------------------------------

def test_build_filtered_timeframe_result_basic_range() -> None:
  tf = TimeframeResult(
    data_range=["2023-01-01", "2023-10-01"],
    total_samples=10,
    results=_month_result_rows(),
    samples=_month_samples(),
  )
  result = build_filtered_timeframe_result(tf, start="2023-03-01", end="2023-06-01")

  assert result.data_range == ["2023-03-01", "2023-06-01"]
  assert result.total_samples == 4
  assert result.slices == {}
  assert len(result.samples) == 4


def test_build_filtered_timeframe_result_empty_range() -> None:
  # Range entirely outside the data -> 0 samples, empty data_range, 0.0 probabilities
  tf = TimeframeResult(
    data_range=["2023-01-01", "2023-10-01"],
    total_samples=10,
    results=_month_result_rows(),
    samples=_month_samples(),
  )
  result = build_filtered_timeframe_result(tf, start="2024-01-01", end="2024-12-31")

  assert result.data_range == []
  assert result.total_samples == 0
  for row in result.results:
    assert row.probability == 0.0
    assert row.count == 0


# ---------------------------------------------------------------------------
# validate_date_params
# ---------------------------------------------------------------------------

def test_validate_date_params_malformed_date_returns_message() -> None:
  message = validate_date_params("not-a-date", None)

  assert message is not None
  assert "Invalid" in message


def test_validate_date_params_start_after_end_returns_message() -> None:
  message = validate_date_params("2023-06-01", "2023-01-01")

  assert message is not None
  assert "must not be after" in message


def test_validate_date_params_valid_returns_none() -> None:
  assert validate_date_params("2023-01-01", "2023-06-01") is None
  assert validate_date_params(None, None) is None
  assert validate_date_params("2023-01-01", None) is None
  assert validate_date_params(None, "2023-06-01") is None


# ---------------------------------------------------------------------------
# has_undeclared_agg_metadata
# ---------------------------------------------------------------------------

def test_has_undeclared_agg_metadata_true_when_samples_carry_values() -> None:
  # value != None, agg == None, but a matching sample DOES carry a value ->
  # this is a decomposable stat family missing its agg metadata.
  samples = [
    SampleRow(date="2023-01-01", condition="cond_a", outcome="out_x", value=1.5),
  ]
  results = [_magnitude_row(agg=None)]

  assert has_undeclared_agg_metadata(results, samples) is True


def test_has_undeclared_agg_metadata_false_for_non_decomposable_case() -> None:
  # value != None, agg == None, and no matching sample carries a value ->
  # this is the non-decomposable case (e.g. Pearson r), not missing metadata.
  samples = [
    SampleRow(date="2023-01-01", condition="cond_a", outcome="out_x", value=None),
  ]
  results = [_magnitude_row(agg=None)]

  assert has_undeclared_agg_metadata(results, samples) is False


def test_has_undeclared_agg_metadata_false_when_agg_declared() -> None:
  # agg is set -> row is skipped entirely regardless of sample values.
  samples = [
    SampleRow(date="2023-01-01", condition="cond_a", outcome="out_x", value=1.5),
  ]
  results = [_magnitude_row(agg="mean")]

  assert has_undeclared_agg_metadata(results, samples) is False
