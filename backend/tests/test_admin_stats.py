"""Tests for the /admin/stats endpoints.

The StatsLoader dependency is overridden with a loader bound to a synthetic
tmp_path, so these tests never depend on the real results/ directory.

Auth is required since issue #110: all tests authenticate as admin before
hitting the protected routes.
"""

from pathlib import Path

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


@pytest.fixture()
def stats_client(tmp_path: Path) -> tuple[TestClient, StatsLoader, Path]:
  """TestClient with isolated DB + get_stats_loader bound to tmp_path.

  The client is authenticated as an admin user so it can access all
  protected endpoints.
  """
  results_dir = tmp_path / "results"
  results_dir.mkdir()
  write_stat_result(make_stat_run_result("alpha_stat"), results_dir)

  loader = StatsLoader(results_dir=results_dir)
  loader.load_all()

  # Initialise a temp database and seed an admin user.
  tmp_db = tmp_path / "test.db"
  init_db(tmp_db)
  users_repo.create_user("admin_user", hash_password("admin_pass"), "admin", tmp_db)
  settings.db_path = tmp_db

  app = create_app()
  app.dependency_overrides[get_stats_loader] = lambda: loader

  client = TestClient(app)

  # Log in so all subsequent requests carry the session cookie.
  resp = client.post("/auth/login", json={"username": "admin_user", "password": "admin_pass"})
  assert resp.status_code == 200, f"Login failed: {resp.json()}"

  yield client, loader, results_dir

  # Restore real db_path after the test.
  settings.db_path = Path(__file__).resolve().parents[3] / "backend" / "db" / "augur.db"


def test_list_stats_returns_200_and_envelope(
  stats_client: tuple[TestClient, StatsLoader, Path],
) -> None:
  """GET /admin/stats returns 200 with the standard { data, error } envelope."""
  client, _, _ = stats_client
  response = client.get("/admin/stats")

  assert response.status_code == 200
  body = response.json()
  assert body["error"] is None
  assert isinstance(body["data"], list)


def test_list_stats_returns_loaded_family_metadata(
  stats_client: tuple[TestClient, StatsLoader, Path],
) -> None:
  """GET /admin/stats reports each loaded family with its timeframe metadata."""
  client, _, _ = stats_client
  body = client.get("/admin/stats").json()

  assert len(body["data"]) == 1
  family = body["data"][0]
  assert family["family"] == "alpha_stat"
  assert family["title"] == {"en": "alpha_stat title", "fr": "alpha_stat titre"}
  assert family["timeframes"] == [
    {
      "instrument": "NQ",
      "timeframe": "1h",
      "data_range": ["2023-01-02", "2023-12-29"],
      "total_samples": 100,
    }
  ]


def test_reload_stats_picks_up_new_file(
  stats_client: tuple[TestClient, StatsLoader, Path],
) -> None:
  """POST /admin/stats/reload reflects a file added after the initial load."""
  client, _, results_dir = stats_client

  assert len(client.get("/admin/stats").json()["data"]) == 1

  write_stat_result(make_stat_run_result("beta_stat"), results_dir)
  reload_body = client.post("/admin/stats/reload").json()

  assert reload_body["error"] is None
  families = {f["family"] for f in reload_body["data"]}
  assert families == {"alpha_stat", "beta_stat"}


def test_get_stat_returns_full_result(
  stats_client: tuple[TestClient, StatsLoader, Path],
) -> None:
  """GET /admin/stats/{family} returns 200 with the full StatRunResult payload."""
  client, _, _ = stats_client
  response = client.get("/admin/stats/alpha_stat")

  assert response.status_code == 200
  body = response.json()
  assert body["error"] is None

  data = body["data"]
  assert data["result"]["stat_name"] == "alpha_stat"
  assert "NQ" in data["result"]["instruments"]
  assert "1h" in data["result"]["instruments"]["NQ"]
  tf = data["result"]["instruments"]["NQ"]["1h"]
  assert tf["total_samples"] == 100
  assert len(tf["results"]) == 1
  assert isinstance(data["computed_at"], str)
  assert len(data["computed_at"]) > 0


def test_get_stat_returns_404_for_unknown_family(
  stats_client: tuple[TestClient, StatsLoader, Path],
) -> None:
  """GET /admin/stats/{family} returns 404 with an error message when family is absent."""
  client, _, _ = stats_client
  response = client.get("/admin/stats/does_not_exist")

  assert response.status_code == 404
  body = response.json()
  assert body["data"] is None
  assert "not found" in body["error"].lower()
