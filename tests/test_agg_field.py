"""Tests for the `agg` field on StatResultRow (issue #185).

All data is synthetic — no real market files required.

Covers:
  - StatResultRow.agg round-trips through write_results() for all four
    aggregation values ("mean", "max", "min", "median").
  - Backward compatibility: a StatRunResult/JSON built WITHOUT the `agg`
    field still validates, and `agg` defaults to None.
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from stats.base import (
  I18nString,
  Labels,
  StatResultRow,
  StatRunResult,
  TimeframeResult,
  write_results,
)

_LABELS = Labels(
  conditions={"window": I18nString(en="Window", fr="Fenêtre")},
  outcomes={
    "range_avg": I18nString(en="Average range", fr="Amplitude moyenne"),
    "range_max": I18nString(en="Maximum range", fr="Amplitude maximale"),
    "range_min": I18nString(en="Minimum range", fr="Amplitude minimale"),
    "range_median": I18nString(en="Median range", fr="Amplitude médiane"),
    "green_close": I18nString(en="Green close", fr="Clôture verte"),
  },
)

# The four decomposable aggregation values, plus None for probability rows and
# non-decomposable magnitude rows (Pearson r).
_AGG_VALUES: list[str | None] = ["mean", "max", "min", "median", None]


def _make_result_with_agg_values() -> StatRunResult:
  """Build a StatRunResult with one row per `agg` value (all four plus None)."""
  rows = [
    StatResultRow(
      condition="window",
      outcome=f"range_{agg or 'none'}",
      count=10,
      total=10,
      probability=0.0 if agg is not None else 0.6,
      baseline_prob=0.0,
      baseline_n=10,
      value=1.0 if agg is not None else None,
      value_baseline=0.5 if agg is not None else None,
      agg=agg,
    )
    for agg in _AGG_VALUES
  ]
  tf_result = TimeframeResult(
    data_range=["2024-01-01", "2024-01-10"],
    total_samples=10,
    results=rows,
  )
  return StatRunResult(
    stat_name="agg_field_stat",
    title=I18nString(en="Agg field stat", fr="Stat agg"),
    definition=I18nString(en="A stat", fr="Une stat"),
    labels=_LABELS,
    instruments={"NQ": {"1h": tf_result}},
  )


# ===========================================================================
# 1. Round-trip: all four agg values (+ None) survive write_results()
# ===========================================================================


def test_agg_field_round_trip_all_values(tmp_path: Path) -> None:
  """A StatRunResult carrying rows with agg set to each of the four values
  (plus None) survives JSON write -> read -> validate intact."""
  written_path = write_results(_make_result_with_agg_values(), results_dir=tmp_path)

  raw = json.loads(written_path.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)

  rows = validated.instruments["NQ"]["1h"].results
  by_outcome = {r.outcome: r for r in rows}
  assert by_outcome["range_mean"].agg == "mean"
  assert by_outcome["range_max"].agg == "max"
  assert by_outcome["range_min"].agg == "min"
  assert by_outcome["range_median"].agg == "median"
  assert by_outcome["range_none"].agg is None


def test_agg_field_defaults_to_none_when_omitted() -> None:
  """StatResultRow.agg defaults to None when not passed to the constructor."""
  row = StatResultRow(
    condition="green_open",
    outcome="green_close",
    count=6,
    total=10,
    probability=0.6,
    baseline_prob=0.5,
    baseline_n=10,
  )
  assert row.agg is None


# ===========================================================================
# 2. Backward compatibility: JSON built without `agg` still validates
# ===========================================================================


def test_result_row_without_agg_field_validates() -> None:
  """A dict lacking the `agg` key entirely still validates, defaulting to None."""
  raw = {
    "condition": "green_open",
    "outcome": "green_close",
    "count": 6,
    "total": 10,
    "probability": 0.6,
    "baseline_prob": 0.5,
    "baseline_n": 10,
  }
  row = StatResultRow.model_validate(raw)
  assert row.agg is None


def test_full_result_without_agg_field_round_trips() -> None:
  """A full StatRunResult JSON blob predating #185 (no `agg` key anywhere)
  still validates, and every row's `agg` defaults to None."""
  raw = {
    "stat_name": "legacy_agg_stat",
    "title": {"en": "Legacy", "fr": "Ancien"},
    "definition": {"en": "A stat", "fr": "Une stat"},
    "labels": {
      "conditions": {"green_open": {"en": "Green open", "fr": "Ouverture verte"}},
      "outcomes": {"green_close": {"en": "Green close", "fr": "Clôture verte"}},
    },
    "instruments": {
      "NQ": {
        "15min": {
          "data_range": ["2024-01-01", "2024-01-10"],
          "total_samples": 10,
          "results": [
            {
              "condition": "green_open",
              "outcome": "green_close",
              "count": 6,
              "total": 10,
              "probability": 0.6,
              "baseline_prob": 0.5,
              "baseline_n": 10,
              # No "value", "value_baseline" or "agg" key at all — the shape
              # produced before either #174 (value channel) or #185 (agg).
            }
          ],
        }
      }
    },
  }
  validated = StatRunResult.model_validate(raw)
  row = validated.instruments["NQ"]["15min"].results[0]
  assert row.agg is None
  assert row.value is None


# ===========================================================================
# 3. Invalid agg values are rejected at validation (Literal constraint)
# ===========================================================================


def test_invalid_agg_value_is_rejected() -> None:
  """An unknown aggregation name (e.g. a typo) fails validation instead of
  silently loading — the field is a Literal, not a free-form str."""
  raw = {
    "condition": "window",
    "outcome": "range_avg",
    "count": 10,
    "total": 10,
    "probability": 0.0,
    "baseline_prob": 0.0,
    "baseline_n": 10,
    "value": 1.0,
    "agg": "meen",  # typo — not one of mean/max/min/median
  }
  with pytest.raises(ValidationError):
    StatResultRow.model_validate(raw)
