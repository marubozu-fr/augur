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
from backend.tests.conftest import (
  make_sampled_stat_run_result,
  make_stat_run_result,
  write_stat_result,
)
from stats.base import SampleRow, StatResultRow


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


# ---------------------------------------------------------------------------
# GET /admin/stats/{family}/{instrument}/{timeframe} — date-range filter
# The backoffice period filter cannot send an X-API-Key, so it uses this
# session-authenticated counterpart to the /api/v1 timeframe endpoint.
# ---------------------------------------------------------------------------


@pytest.fixture()
def sampled_stats_client(tmp_path: Path) -> tuple[TestClient, Path]:
  """Authenticated client backed by a family carrying per-day samples.

  10 samples, one per month Jan-Oct 2023, all condition 'cond_a'. Outcome
  alternates green (odd months) / red (even months) so date filtering has
  real (condition, outcome) pairs to reaggregate.
  """
  results_dir = tmp_path / "results"
  results_dir.mkdir()
  samples = [
    SampleRow(
      date=f"2023-{month:02d}-01",
      condition="cond_a",
      outcome="green" if month % 2 else "red",
    )
    for month in range(1, 11)
  ]
  results = [
    StatResultRow(
      condition="cond_a", outcome="green", count=5, total=10,
      probability=0.5, baseline_prob=0.5, baseline_n=10,
    ),
    StatResultRow(
      condition="cond_a", outcome="red", count=5, total=10,
      probability=0.5, baseline_prob=0.5, baseline_n=10,
    ),
  ]
  write_stat_result(
    make_sampled_stat_run_result(
      "sampled_stat",
      samples=samples,
      results=results,
      data_range=["2023-01-01", "2023-10-01"],
    ),
    results_dir,
  )
  loader = StatsLoader(results_dir=results_dir)
  loader.load_all()

  tmp_db = tmp_path / "test.db"
  init_db(tmp_db)
  users_repo.create_user("admin_user", hash_password("admin_pass"), "admin", tmp_db)
  settings.db_path = tmp_db

  app = create_app()
  app.dependency_overrides[get_stats_loader] = lambda: loader
  client = TestClient(app)
  resp = client.post("/auth/login", json={"username": "admin_user", "password": "admin_pass"})
  assert resp.status_code == 200, f"Login failed: {resp.json()}"

  yield client, results_dir

  settings.db_path = Path(__file__).resolve().parents[3] / "backend" / "db" / "augur.db"


def test_get_timeframe_without_filter_returns_full_result(
  sampled_stats_client: tuple[TestClient, Path],
) -> None:
  """Without ?start/?end the stored TimeframeResult is returned unchanged."""
  client, _ = sampled_stats_client
  response = client.get("/admin/stats/sampled_stat/NQ/1h")

  assert response.status_code == 200
  body = response.json()
  assert body["error"] is None
  assert body["data"]["total_samples"] == 10
  assert body["data"]["data_range"] == ["2023-01-01", "2023-10-01"]


def test_get_timeframe_date_filter_reaggregates(
  sampled_stats_client: tuple[TestClient, Path],
) -> None:
  """?start/?end narrows the sample window and reaggregates totals."""
  client, _ = sampled_stats_client
  # Jan-Mar 2023: 3 samples (green, red, green).
  response = client.get(
    "/admin/stats/sampled_stat/NQ/1h?start=2023-01-01&end=2023-03-31"
  )

  assert response.status_code == 200
  body = response.json()
  assert body["error"] is None
  assert body["data"]["total_samples"] == 3
  # Date filtering drops slices in this MVP.
  assert body["data"]["slices"] == {}


def test_get_timeframe_malformed_start_returns_400(
  sampled_stats_client: tuple[TestClient, Path],
) -> None:
  """A non-ISO ?start value is rejected with 400."""
  client, _ = sampled_stats_client
  response = client.get("/admin/stats/sampled_stat/NQ/1h?start=not-a-date")

  assert response.status_code == 400
  assert response.json()["data"] is None


def test_get_timeframe_start_after_end_returns_400(
  sampled_stats_client: tuple[TestClient, Path],
) -> None:
  """start later than end is rejected with 400."""
  client, _ = sampled_stats_client
  response = client.get(
    "/admin/stats/sampled_stat/NQ/1h?start=2023-06-01&end=2023-01-01"
  )

  assert response.status_code == 400
  assert response.json()["data"] is None


def test_get_timeframe_date_filter_without_samples_returns_400(
  stats_client: tuple[TestClient, StatsLoader, Path],
) -> None:
  """Filtering a family that carries no samples returns a 400 with guidance."""
  client, _, _ = stats_client
  response = client.get("/admin/stats/alpha_stat/NQ/1h?start=2023-01-01")

  assert response.status_code == 400
  assert "regenerated" in response.json()["error"].lower()


def test_get_timeframe_unknown_instrument_returns_404(
  sampled_stats_client: tuple[TestClient, Path],
) -> None:
  """An instrument absent from the family yields 404."""
  client, _ = sampled_stats_client
  response = client.get("/admin/stats/sampled_stat/ES/1h")

  assert response.status_code == 404
  assert response.json()["data"] is None


def test_get_timeframe_unknown_timeframe_returns_404(
  sampled_stats_client: tuple[TestClient, Path],
) -> None:
  """A timeframe absent for the instrument yields 404."""
  client, _ = sampled_stats_client
  response = client.get("/admin/stats/sampled_stat/NQ/5m")

  assert response.status_code == 404
  assert response.json()["data"] is None


def test_get_timeframe_requires_authentication(tmp_path: Path) -> None:
  """Without a session cookie the endpoint returns 401."""
  results_dir = tmp_path / "results"
  results_dir.mkdir()
  write_stat_result(make_stat_run_result("alpha_stat"), results_dir)
  loader = StatsLoader(results_dir=results_dir)
  loader.load_all()

  app = create_app()
  app.dependency_overrides[get_stats_loader] = lambda: loader
  client = TestClient(app)
  response = client.get("/admin/stats/alpha_stat/NQ/1h")

  assert response.status_code == 401
