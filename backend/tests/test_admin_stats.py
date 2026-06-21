"""Tests for the /admin/stats endpoints.

The StatsLoader dependency is overridden with a loader bound to a synthetic
tmp_path, so these tests never depend on the real results/ directory.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.core.dependencies import get_stats_loader
from backend.app.core.stats_loader import StatsLoader
from backend.app.main import create_app
from backend.tests.conftest import make_stat_run_result, write_stat_result


@pytest.fixture()
def stats_client(tmp_path: Path) -> tuple[TestClient, StatsLoader, Path]:
  """TestClient whose get_stats_loader dependency is bound to tmp_path."""
  write_stat_result(make_stat_run_result("alpha_stat"), tmp_path)

  loader = StatsLoader(results_dir=tmp_path)
  loader.load_all()

  app = create_app()
  app.dependency_overrides[get_stats_loader] = lambda: loader
  return TestClient(app), loader, tmp_path


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
