"""Shared pytest fixtures and stat-result factories for the backend tests."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app
from stats.base import (
  I18nString,
  Labels,
  StatResultRow,
  StatRunResult,
  TimeframeResult,
)


@pytest.fixture()
def client() -> TestClient:
  """Return a synchronous TestClient bound to a fresh FastAPI app."""
  return TestClient(create_app())


# ---------------------------------------------------------------------------
# Stat-result factories (synthetic data for tests — no real results/ needed)
# ---------------------------------------------------------------------------

def make_result_row(condition: str = "cond_a", outcome: str = "out_x") -> StatResultRow:
  """Build a minimal valid StatResultRow."""
  return StatResultRow(
    condition=condition,
    outcome=outcome,
    count=40,
    total=100,
    probability=0.4,
    baseline_prob=0.38,
    baseline_n=100,
  )


def make_stat_run_result(
  stat_name: str,
  instrument: str = "NQ",
  timeframe: str = "1h",
  total_samples: int = 100,
  data_range: list[str] | None = None,
) -> StatRunResult:
  """Build a minimal valid StatRunResult via Pydantic models."""
  if data_range is None:
    data_range = ["2023-01-02", "2023-12-29"]
  tf_result = TimeframeResult(
    data_range=data_range,
    total_samples=total_samples,
    results=[make_result_row()],
  )
  return StatRunResult(
    stat_name=stat_name,
    title=I18nString(en=f"{stat_name} title", fr=f"{stat_name} titre"),
    definition=I18nString(en=f"{stat_name} def", fr=f"{stat_name} déf"),
    labels=Labels(
      conditions={"cond_a": I18nString(en="Condition A", fr="Condition A")},
      outcomes={"out_x": I18nString(en="Outcome X", fr="Résultat X")},
    ),
    instruments={instrument: {timeframe: tf_result}},
  )


def write_stat_result(result: StatRunResult, directory: Path) -> Path:
  """Serialize a StatRunResult to <directory>/<stat_name>.json."""
  path = directory / f"{result.stat_name}.json"
  path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
  return path
