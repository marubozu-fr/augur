"""Pure-function helpers for date-range filtering of a TimeframeResult.

Business logic for narrowing a TimeframeResult's samples to a [start, end]
date range and reaggregating StatResultRow objects from that subset. These
functions operate on already-loaded objects from the StatsLoader cache and
never touch the filesystem or FastAPI.
"""

import statistics
from collections import defaultdict
from datetime import date

from stats.base import SampleRow, StatResultRow, TimeframeResult

_AGG_FUNCS = {
  "mean": statistics.mean,
  "max": max,
  "min": min,
  "median": statistics.median,
}


def filter_samples(
  samples: list[SampleRow],
  start: str | None,
  end: str | None,
) -> list[SampleRow]:
  """Return samples whose date falls within [start, end], inclusive.

  Both bounds are optional; None means unbounded on that side. Dates are
  zero-padded ISO strings (YYYY-MM-DD) so plain string comparison is safe.
  """
  filtered = samples
  if start is not None:
    filtered = [s for s in filtered if s.date >= start]
  if end is not None:
    filtered = [s for s in filtered if s.date <= end]
  return filtered


def _aggregate_values(values: list[float], agg: str) -> float | None:
  """Apply the named aggregation function, or None if there are no values."""
  if not values:
    return None
  return _AGG_FUNCS[agg](values)


def reaggregate(
  filtered_samples: list[SampleRow],
  original_results: list[StatResultRow],
) -> list[StatResultRow]:
  """Rebuild StatResultRow objects from a filtered sample subset.

  Preserves the exact (condition, outcome) pairs and order of
  ``original_results`` so the schema stays stable even when the filtered range
  has zero samples for some pairs. ``count``/``total``/``probability`` are
  recomputed directly from the filtered samples. ``baseline_prob``/
  ``baseline_n`` cannot be recomputed from samples alone and are reset to
  0.0/0.

  Magnitude rows (``original.value is not None``) reaggregate
  ``SampleRow.value`` over the matching samples using the original row's
  ``agg`` function, ignoring samples whose ``value`` is None. Rows without an
  ``agg`` (non-decomposable metrics, e.g. Pearson r) yield ``value=None``.
  Probability rows (``original.value is None``) leave ``value`` and
  ``value_baseline`` as None. ``agg`` is always preserved from the original
  row.
  """
  totals_by_condition: dict[str, int] = defaultdict(int)
  by_pair: dict[tuple[str, str], list[SampleRow]] = defaultdict(list)
  for s in filtered_samples:
    totals_by_condition[s.condition] += 1
    by_pair[(s.condition, s.outcome)].append(s)

  rows: list[StatResultRow] = []
  for original in original_results:
    pair_samples = by_pair.get((original.condition, original.outcome), [])
    count = len(pair_samples)
    total = totals_by_condition.get(original.condition, 0)
    probability = count / total if total > 0 else 0.0

    value: float | None = None
    if original.value is not None and original.agg is not None:
      values = [s.value for s in pair_samples if s.value is not None]
      value = _aggregate_values(values, original.agg)

    rows.append(
      StatResultRow(
        condition=original.condition,
        outcome=original.outcome,
        count=count,
        total=total,
        probability=probability,
        baseline_prob=0.0,
        baseline_n=0,
        value=value,
        value_baseline=None,
        agg=original.agg,
      )
    )
  return rows


def build_filtered_timeframe_result(
  tf: TimeframeResult,
  start: str | None,
  end: str | None,
) -> TimeframeResult:
  """Return a new TimeframeResult filtered to samples within [start, end].

  Recomputes ``data_range``, ``total_samples``, and ``results`` from the
  filtered sample set. ``slices`` are not re-aggregated in this MVP, so the
  returned result always has an empty ``slices`` dict.
  """
  filtered = filter_samples(tf.samples, start, end)
  results = reaggregate(filtered, tf.results)

  if filtered:
    dates = sorted(s.date for s in filtered)
    data_range = [dates[0], dates[-1]]
  else:
    data_range = []

  return TimeframeResult(
    data_range=data_range,
    total_samples=len({s.date for s in filtered}),
    results=results,
    slices={},
    samples=filtered,
  )


def validate_date_params(start: str | None, end: str | None) -> str | None:
  """Validate ISO date query params. Returns an error message, or None if valid.

  Checks that ``start`` and ``end`` (when provided) parse as YYYY-MM-DD dates
  and that ``start`` is not after ``end`` when both are present. Kept free of
  FastAPI imports so it stays a plain, testable function; the router turns a
  non-None return value into an HTTP 400 response.
  """
  for label, value in (("start", start), ("end", end)):
    if value is None:
      continue
    try:
      date.fromisoformat(value)
    except ValueError:
      return f"Invalid '{label}' date '{value}'. Expected format YYYY-MM-DD."

  if start is not None and end is not None and start > end:
    return f"'start' ({start}) must not be after 'end' ({end})."

  return None


def has_undeclared_agg_metadata(
  original_results: list[StatResultRow],
  samples: list[SampleRow],
) -> bool:
  """Detect a decomposable magnitude row that is missing its ``agg`` metadata.

  A row with ``value is not None and agg is None`` is ambiguous on its own: it
  could be a genuinely non-decomposable metric (e.g. Pearson r, which has no
  per-sample value to reaggregate) or a magnitude stat family whose result
  file predates the ``agg`` field and needs to be regenerated. The two cases
  are told apart by the samples themselves: if any sample for that
  (condition, outcome) pair carries a non-None ``value``, the family *is*
  decomposable and is simply missing its ``agg`` field, so the caller should
  reject the request. If no sample carries a value, the row is the
  non-decomposable case, which ``reaggregate()`` already handles by yielding
  ``value=None`` without needing a 400.
  """
  by_pair: dict[tuple[str, str], list[SampleRow]] = defaultdict(list)
  for s in samples:
    by_pair[(s.condition, s.outcome)].append(s)

  for original in original_results:
    if original.value is None or original.agg is not None:
      continue
    pair_samples = by_pair.get((original.condition, original.outcome), [])
    if any(s.value is not None for s in pair_samples):
      return True
  return False
