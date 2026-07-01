"""Tests for the per-day samples framework in stats/base.py (issue #174).

All data is synthetic — no real market files required. A minimal concrete
``BaseStat`` subclass exercises the ``classify_samples`` hook and its
integration in ``compute()``.

Covers:
  - SampleRow round-trip through write_results()
  - TimeframeResult.samples default keeps sample-less JSON valid (backward compat)
  - BaseStat.classify_samples default returns []
  - compute() stores classified samples, sorted by date ascending
  - value channel for magnitude stats survives the round-trip
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import (
  BaseStat,
  I18nString,
  Labels,
  SampleRow,
  StatResultRow,
  StatRunResult,
  TimeframeResult,
  write_results,
)

_NY = "America/New_York"


def _make_idx(dates: list[str]) -> pd.DatetimeIndex:
  """Build a normalized, tz-aware DatetimeIndex from a list of YYYY-MM-DD strings."""
  return pd.to_datetime(dates).tz_localize(_NY).normalize()


# ---------------------------------------------------------------------------
# Minimal concrete stats
# ---------------------------------------------------------------------------

_LABELS = Labels(
  conditions={"green_open": I18nString(en="Green open", fr="Ouverture verte")},
  outcomes={"green_close": I18nString(en="Green close", fr="Clôture verte")},
)


class _NoSamplesStat(BaseStat):
  """A stat that never overrides classify_samples (framework default path)."""

  stat_name = "no_samples_stat"
  title = I18nString(en="No samples", fr="Sans échantillons")
  definition = I18nString(en="A stat", fr="Une stat")
  labels = _LABELS

  def __init__(self, instrument: str = "NQ", timeframe: str = "15min") -> None:
    self.instrument = instrument
    self.timeframe = timeframe

  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    return candles_df

  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    return []

  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    return []


class _SamplesStat(_NoSamplesStat):
  """A stat that classifies one SampleRow per row of its day table.

  The day table carries a boolean ``green_open`` column; the outcome is a
  ``green_close`` boolean. ``classify_samples`` deliberately returns rows in
  reverse-date order so the framework's sort is exercised.
  """

  stat_name = "samples_stat"

  def classify_samples(self, day_table: pd.DataFrame) -> list[SampleRow]:
    rows: list[SampleRow] = []
    for ts, r in day_table.iterrows():
      rows.append(
        SampleRow(
          date=ts.strftime("%Y-%m-%d"),
          condition="green_open" if bool(r["green_open"]) else "red_open",
          outcome="green_close" if bool(r["green_close"]) else "red_close",
        )
      )
    # Return in reverse-date order to prove compute() sorts ascending.
    return list(reversed(rows))


class _MagnitudeStat(_NoSamplesStat):
  """A magnitude stat: classify_samples populates the value channel."""

  stat_name = "magnitude_stat"

  def classify_samples(self, day_table: pd.DataFrame) -> list[SampleRow]:
    return [
      SampleRow(
        date=ts.strftime("%Y-%m-%d"),
        condition="green_open",
        outcome="return",
        value=float(r["ret"]),
      )
      for ts, r in day_table.iterrows()
    ]


def _day_table(dates: list[str], **columns: list) -> pd.DataFrame:
  return pd.DataFrame(columns, index=_make_idx(dates))


# ===========================================================================
# 1. SampleRow model + round-trip
# ===========================================================================


def _make_result_with_samples() -> StatRunResult:
  """Build a minimal StatRunResult carrying two samples."""
  tf_result = TimeframeResult(
    data_range=["2024-01-01", "2024-01-02"],
    total_samples=2,
    results=[],
    samples=[
      SampleRow(date="2024-01-01", condition="green_open", outcome="green_close"),
      SampleRow(date="2024-01-02", condition="red_open", outcome="red_close", value=1.5),
    ],
  )
  return StatRunResult(
    stat_name="samples_stat",
    title=I18nString(en="Samples", fr="Échantillons"),
    definition=I18nString(en="A stat", fr="Une stat"),
    labels=_LABELS,
    instruments={"NQ": {"15min": tf_result}},
  )


def test_samples_round_trip(tmp_path: Path) -> None:
  """A StatRunResult with samples must survive JSON write → read → validate."""
  written_path = write_results(_make_result_with_samples(), results_dir=tmp_path)

  raw = json.loads(written_path.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)

  samples = validated.instruments["NQ"]["15min"].samples
  assert len(samples) == 2
  assert samples[0] == SampleRow(
    date="2024-01-01", condition="green_open", outcome="green_close"
  )
  assert samples[1].value == pytest.approx(1.5)


def test_sample_value_defaults_to_none() -> None:
  """SampleRow.value defaults to None for pure probability stats."""
  row = SampleRow(date="2024-01-01", condition="green_open", outcome="green_close")
  assert row.value is None


# ===========================================================================
# 2. Backward compatibility: JSON without a samples field still validates
# ===========================================================================


def test_timeframe_result_without_samples_validates() -> None:
  """A JSON blob lacking the samples field validates with samples == []."""
  raw = {
    "data_range": ["2024-01-01", "2024-01-02"],
    "total_samples": 2,
    "results": [],
  }
  tf = TimeframeResult.model_validate(raw)
  assert tf.samples == []


def test_full_result_without_samples_round_trips(tmp_path: Path) -> None:
  """An on-disk result written without any samples reloads with empty samples."""
  tf_result = TimeframeResult(data_range=[], total_samples=0, results=[])
  run_result = StatRunResult(
    stat_name="legacy_stat",
    title=I18nString(en="Legacy", fr="Ancien"),
    definition=I18nString(en="A stat", fr="Une stat"),
    labels=_LABELS,
    instruments={"NQ": {"15min": tf_result}},
  )
  written_path = write_results(run_result, results_dir=tmp_path)

  # The serialized field defaults to [] and the reload must succeed.
  validated = StatRunResult.model_validate_json(written_path.read_text(encoding="utf-8"))
  assert validated.instruments["NQ"]["15min"].samples == []


# ===========================================================================
# 3. BaseStat.classify_samples default
# ===========================================================================


def test_default_classify_samples_returns_empty() -> None:
  """The framework default classify_samples returns no rows."""
  stat = _NoSamplesStat()
  dt = _day_table(["2024-01-01", "2024-01-02"], green_open=[True, False])
  assert stat.classify_samples(dt) == []


def test_compute_without_override_has_empty_samples() -> None:
  """compute() on a stat that does not override classify_samples yields []."""
  stat = _NoSamplesStat()
  dt = _day_table(["2024-01-01", "2024-01-02"], green_open=[True, False])
  result = stat.compute(dt)
  assert result.instruments["NQ"]["15min"].samples == []


# ===========================================================================
# 4. compute() integration: samples stored and sorted ascending
# ===========================================================================


def test_compute_stores_classified_samples() -> None:
  """compute() populates TimeframeResult.samples from classify_samples.

  Hand-calculation for the three rows:
    2024-01-02: green_open=True,  green_close=True  → green_open / green_close
    2024-01-01: green_open=False, green_close=True  → red_open   / green_close
    2024-01-03: green_open=True,  green_close=False → green_open / red_close
  classify_samples returns these reversed; compute() must sort by date ascending.
  """
  stat = _SamplesStat()
  dt = _day_table(
    ["2024-01-02", "2024-01-01", "2024-01-03"],
    green_open=[True, False, True],
    green_close=[True, True, False],
  )
  result = stat.compute(dt)
  samples = result.instruments["NQ"]["15min"].samples

  assert [s.date for s in samples] == ["2024-01-01", "2024-01-02", "2024-01-03"]
  assert samples[0].condition == "red_open"
  assert samples[0].outcome == "green_close"
  assert samples[1].condition == "green_open"
  assert samples[2].outcome == "red_close"


def test_compute_samples_sorted_ascending() -> None:
  """Samples must be sorted by date ascending regardless of input order."""
  stat = _SamplesStat()
  dt = _day_table(
    ["2024-03-05", "2024-01-01", "2024-02-02"],
    green_open=[True, True, True],
    green_close=[True, True, True],
  )
  result = stat.compute(dt)
  dates = [s.date for s in result.instruments["NQ"]["15min"].samples]
  assert dates == sorted(dates)


def test_compute_magnitude_samples_carry_value() -> None:
  """A magnitude stat's samples carry the per-day value through compute()."""
  stat = _MagnitudeStat()
  dt = _day_table(["2024-01-01", "2024-01-02"], ret=[0.5, -0.25])
  result = stat.compute(dt)
  samples = result.instruments["NQ"]["15min"].samples

  by_date = {s.date: s for s in samples}
  assert by_date["2024-01-01"].value == pytest.approx(0.5)
  assert by_date["2024-01-02"].value == pytest.approx(-0.25)


def test_compute_empty_day_table_has_empty_samples() -> None:
  """An empty day table yields an empty samples list, not an error."""
  stat = _SamplesStat()
  dt = _day_table([], green_open=[], green_close=[])
  result = stat.compute(dt)
  assert result.instruments["NQ"]["15min"].samples == []
