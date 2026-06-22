"""Tests for the /admin/status endpoint (issue #114).

The endpoint exposes loaded stats files, available data files, runtime
versions, and uptime. It requires the admin role.
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import settings
from backend.app.core.db import init_db
from backend.app.core.dependencies import get_stats_loader
from backend.app.core.stats_loader import StatsLoader
from backend.app.main import create_app
from backend.app.repositories import users as users_repo
from backend.app.services.auth import hash_password
from backend.tests.conftest import make_stat_run_result, write_stat_result


_REAL_DB = Path(__file__).resolve().parents[3] / "backend" / "db" / "augur.db"


def _write_test_parquet(directory: Path, filename: str = "TEST_1min.parquet") -> Path:
  """Write a tiny tz-aware OHLCV Parquet so the status endpoint can describe it."""
  ts = pd.to_datetime(
    ["2024-01-02 09:30", "2024-01-03 09:30", "2024-01-04 09:30"]
  ).tz_localize("America/New_York")
  df = pd.DataFrame(
    {
      "timestamp": ts,
      "open": [1.0, 2.0, 3.0],
      "high": [1.5, 2.5, 3.5],
      "low": [0.5, 1.5, 2.5],
      "close": [1.2, 2.2, 3.2],
      "volume": [100, 200, 300],
    }
  )
  path = directory / filename
  df.to_parquet(path)
  return path


@pytest.fixture()
def status_env(tmp_path: Path):
  """TestClient wired to tmp results, tmp data dir, and a seeded admin+reader."""
  results_dir = tmp_path / "results"
  results_dir.mkdir()
  write_stat_result(make_stat_run_result("alpha_stat"), results_dir)

  data_dir = tmp_path / "data"
  # Mirror prod layout — Parquet files live under data/processed/, so the
  # fixture must exercise the recursive glob, not the flat one.
  processed_dir = data_dir / "processed"
  processed_dir.mkdir(parents=True)
  _write_test_parquet(processed_dir)

  loader = StatsLoader(results_dir=results_dir)
  loader.load_all()

  tmp_db = tmp_path / "test.db"
  init_db(tmp_db)
  users_repo.create_user("admin_user", hash_password("admin_pass"), "admin", tmp_db)
  users_repo.create_user("reader_user", hash_password("reader_pass"), "reader", tmp_db)
  settings.db_path = tmp_db

  original_data_dir = settings.data_dir
  settings.data_dir = data_dir

  app = create_app()
  app.dependency_overrides[get_stats_loader] = lambda: loader
  # TestClient does not trigger the lifespan, so set the started_at that
  # the /admin/status endpoint reads from app.state.
  app.state.started_at = datetime.now(timezone.utc)
  client = TestClient(app)

  yield client, results_dir, data_dir

  settings.db_path = _REAL_DB
  settings.data_dir = original_data_dir


def _login(client: TestClient, username: str, password: str) -> None:
  resp = client.post("/auth/login", json={"username": username, "password": password})
  assert resp.status_code == 200, f"Login failed: {resp.json()}"


def test_status_requires_auth(status_env) -> None:
  """GET /admin/status without a session returns 401."""
  client, _, _ = status_env
  assert client.get("/admin/status").status_code == 401


def test_status_requires_admin_role(status_env) -> None:
  """A reader-role session is rejected with 403."""
  client, _, _ = status_env
  _login(client, "reader_user", "reader_pass")
  assert client.get("/admin/status").status_code == 403


def test_status_envelope_shape(status_env) -> None:
  """The response uses the standard { data, error } envelope."""
  client, _, _ = status_env
  _login(client, "admin_user", "admin_pass")

  body = client.get("/admin/status").json()
  assert body["error"] is None
  assert isinstance(body["data"], dict)


def test_status_reports_versions_and_uptime(status_env) -> None:
  """Versions and uptime are populated and consistent."""
  client, _, _ = status_env
  _login(client, "admin_user", "admin_pass")

  data = client.get("/admin/status").json()["data"]

  assert data["versions"]["python"].count(".") == 2  # major.minor.patch
  assert isinstance(data["versions"]["fastapi"], str)
  assert data["versions"]["fastapi"]

  assert data["uptime_seconds"] >= 0.0
  # started_at parses as an ISO 8601 datetime
  datetime.fromisoformat(data["started_at"])


def test_status_lists_loaded_stats_files(status_env) -> None:
  """stats_files reports the loaded family with size and mtime."""
  client, _, _ = status_env
  _login(client, "admin_user", "admin_pass")

  data = client.get("/admin/status").json()["data"]

  assert len(data["stats_files"]) == 1
  entry = data["stats_files"][0]
  assert entry["family"] == "alpha_stat"
  assert entry["filename"] == "alpha_stat.json"
  assert entry["size_bytes"] > 0
  datetime.fromisoformat(entry["modified_at"])


def test_status_lists_data_files_with_metadata(status_env) -> None:
  """data_files reports each Parquet file with size, row count, and date range.

  The fixture writes the file under data/processed/ so this also covers the
  recursive glob.
  """
  client, _, _ = status_env
  _login(client, "admin_user", "admin_pass")

  data = client.get("/admin/status").json()["data"]

  assert len(data["data_files"]) == 1
  entry = data["data_files"][0]
  assert entry["filename"] == str(Path("processed") / "TEST_1min.parquet")
  assert entry["size_bytes"] > 0
  assert entry["num_rows"] == 3
  assert entry["data_range"] == {"start": "2024-01-02", "end": "2024-01-04"}


def test_status_data_files_empty_when_dir_missing(status_env, tmp_path: Path) -> None:
  """When data_dir does not exist, data_files is an empty list (not an error)."""
  client, _, _ = status_env
  _login(client, "admin_user", "admin_pass")

  settings.data_dir = tmp_path / "no_such_dir"
  body = client.get("/admin/status").json()
  assert body["error"] is None
  assert body["data"]["data_files"] == []
