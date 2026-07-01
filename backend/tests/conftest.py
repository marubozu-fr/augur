"""Shared pytest fixtures and stat-result factories for the backend tests."""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.core.config import settings
from backend.app.core.db import init_db
from backend.app.core.dependencies import get_stats_loader
from backend.app.core.stats_loader import StatsLoader
from backend.app.main import create_app
from backend.app.repositories import users as users_repo
from backend.app.services.auth import hash_password
from stats.base import (
  I18nString,
  Labels,
  SampleRow,
  SliceGroupResult,
  SliceResult,
  StatResultRow,
  StatRunResult,
  TimeframeResult,
)


@pytest.fixture()
def client() -> TestClient:
  """Return a synchronous TestClient bound to a fresh FastAPI app."""
  return TestClient(create_app())


def make_test_app(tmp_db: Path, tmp_results: Path) -> tuple[FastAPI, TestClient]:
  """Build a FastAPI app + TestClient with isolated DB and results dir.

  Creates an admin_user/admin_pass and a reader_user/reader_pass directly
  in the temp database, writes one synthetic stat family into tmp_results,
  and overrides get_stats_loader so tests don't need the real results/ dir.
  Also redirects settings.db_path to tmp_db — callers should restore it.
  """
  init_db(tmp_db)
  users_repo.create_user("admin_user", hash_password("admin_pass"), "admin", tmp_db)
  users_repo.create_user("reader_user", hash_password("reader_pass"), "reader", tmp_db)

  write_stat_result(make_stat_run_result("test_stat"), tmp_results)
  loader = StatsLoader(results_dir=tmp_results)
  loader.load_all()

  app = create_app()
  app.dependency_overrides[get_stats_loader] = lambda: loader
  settings.db_path = tmp_db

  client = TestClient(app, raise_server_exceptions=True)
  return app, client


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


def make_sliced_stat_run_result(
  stat_name: str,
  instrument: str = "NQ",
  timeframe: str = "1h",
  total_samples: int = 100,
  data_range: list[str] | None = None,
) -> StatRunResult:
  """Build a StatRunResult that declares a 'weekday' slice dimension.

  The TimeframeResult includes a 'weekday' SliceResult with a single
  'monday' SliceGroupResult, so slice-filtering tests have real data to narrow.
  """
  if data_range is None:
    data_range = ["2023-01-02", "2023-12-29"]
  monday_group = SliceGroupResult(
    label=I18nString(en="Monday", fr="Lundi"),
    total_samples=20,
    results=[make_result_row()],
  )
  weekday_slice = SliceResult(
    dimension="weekday",
    groups={"monday": monday_group},
  )
  tf_result = TimeframeResult(
    data_range=data_range,
    total_samples=total_samples,
    results=[make_result_row()],
    slices={"weekday": weekday_slice},
  )
  return StatRunResult(
    stat_name=stat_name,
    title=I18nString(en=f"{stat_name} title", fr=f"{stat_name} titre"),
    definition=I18nString(en=f"{stat_name} def", fr=f"{stat_name} déf"),
    labels=Labels(
      conditions={"cond_a": I18nString(en="Condition A", fr="Condition A")},
      outcomes={"out_x": I18nString(en="Outcome X", fr="Résultat X")},
      dimensions={"weekday": I18nString(en="Day of week", fr="Jour de la semaine")},
    ),
    instruments={instrument: {timeframe: tf_result}},
  )


def make_sampled_stat_run_result(
  stat_name: str,
  instrument: str = "NQ",
  timeframe: str = "1h",
  samples: list[SampleRow] | None = None,
  results: list[StatResultRow] | None = None,
  data_range: list[str] | None = None,
) -> StatRunResult:
  """Build a StatRunResult whose TimeframeResult carries per-day samples.

  Unlike ``make_stat_run_result``, ``TimeframeResult.samples`` is populated so
  date-range filtering (issue #179) has real data to filter and reaggregate.
  ``total_samples`` is derived from the number of distinct sample dates.
  """
  if samples is None:
    samples = []
  if results is None:
    results = [make_result_row()]
  if data_range is None:
    data_range = ["2023-01-02", "2023-12-29"]
  tf_result = TimeframeResult(
    data_range=data_range,
    total_samples=len({s.date for s in samples}),
    results=results,
    samples=samples,
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
