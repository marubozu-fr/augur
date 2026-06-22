"""Tests for the public /api/v1/stats endpoints.

These endpoints require an X-API-Key header (issue #113). The require_api_key
dependency is bypassed via dependency_overrides in each fixture so tests remain
focused on stat routing rather than key management (covered in test_api_keys.py).
The StatsLoader dependency is overridden with a loader bound to a synthetic
tmp_path, so tests never depend on the real results/ directory.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.core.dependencies import get_stats_loader, require_api_key
from backend.app.core.stats_loader import StatsLoader
from backend.app.main import create_app
from backend.tests.conftest import (
  make_sliced_stat_run_result,
  make_stat_run_result,
  write_stat_result,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def api_client(tmp_path: Path) -> TestClient:
  """TestClient with get_stats_loader bound to a tmp_path holding alpha_stat.

  require_api_key is bypassed so tests focus on stat routing only.
  """
  results_dir = tmp_path / "results"
  results_dir.mkdir()
  write_stat_result(make_stat_run_result("alpha_stat"), results_dir)

  loader = StatsLoader(results_dir=results_dir)
  loader.load_all()

  app = create_app()
  app.dependency_overrides[get_stats_loader] = lambda: loader
  app.dependency_overrides[require_api_key] = lambda: None
  return TestClient(app)


@pytest.fixture()
def sliced_api_client(tmp_path: Path) -> TestClient:
  """TestClient with a family that declares a 'weekday' slice dimension.

  require_api_key is bypassed so tests focus on slice filtering only.
  """
  results_dir = tmp_path / "results"
  results_dir.mkdir()
  write_stat_result(make_sliced_stat_run_result("sliced_stat"), results_dir)

  loader = StatsLoader(results_dir=results_dir)
  loader.load_all()

  app = create_app()
  app.dependency_overrides[get_stats_loader] = lambda: loader
  app.dependency_overrides[require_api_key] = lambda: None
  return TestClient(app)


# ---------------------------------------------------------------------------
# GET /api/v1/stats — list all families
# ---------------------------------------------------------------------------

def test_list_stats_returns_200_and_envelope(api_client: TestClient) -> None:
  """GET /api/v1/stats returns 200 with the standard {data, error} envelope."""
  response = api_client.get("/api/v1/stats")

  assert response.status_code == 200
  body = response.json()
  assert body["error"] is None
  assert isinstance(body["data"], list)


def test_list_stats_returns_loaded_family_metadata(api_client: TestClient) -> None:
  """GET /api/v1/stats lists each loaded family with its title and timeframe metadata."""
  body = api_client.get("/api/v1/stats").json()

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


# ---------------------------------------------------------------------------
# GET /api/v1/stats/{family} — full family result
# ---------------------------------------------------------------------------

def test_get_stat_returns_full_result(api_client: TestClient) -> None:
  """GET /api/v1/stats/{family} returns 200 with the full StatRunResult payload."""
  response = api_client.get("/api/v1/stats/alpha_stat")

  assert response.status_code == 200
  body = response.json()
  assert body["error"] is None

  data = body["data"]
  assert data["stat_name"] == "alpha_stat"
  assert "NQ" in data["instruments"]
  assert "1h" in data["instruments"]["NQ"]
  tf = data["instruments"]["NQ"]["1h"]
  assert tf["total_samples"] == 100
  assert len(tf["results"]) == 1


def test_get_stat_unknown_family_returns_404(api_client: TestClient) -> None:
  """GET /api/v1/stats/{family} returns 404 with an error when the family is absent."""
  response = api_client.get("/api/v1/stats/does_not_exist")

  assert response.status_code == 404
  body = response.json()
  assert body["data"] is None
  assert "not found" in body["error"].lower()


# ---------------------------------------------------------------------------
# GET /api/v1/stats/{family}/{instrument} — narrowed to one instrument
# ---------------------------------------------------------------------------

def test_get_stat_instrument_returns_narrowed_result(api_client: TestClient) -> None:
  """GET /api/v1/stats/{family}/{instrument} returns only the requested instrument."""
  response = api_client.get("/api/v1/stats/alpha_stat/NQ")

  assert response.status_code == 200
  body = response.json()
  assert body["error"] is None

  data = body["data"]
  assert list(data["instruments"].keys()) == ["NQ"]


def test_get_stat_instrument_unknown_instrument_returns_404(api_client: TestClient) -> None:
  """GET /api/v1/stats/{family}/{instrument} returns 404 when the instrument is absent."""
  response = api_client.get("/api/v1/stats/alpha_stat/ES")

  assert response.status_code == 404
  body = response.json()
  assert body["data"] is None
  assert "ES" in body["error"]


def test_get_stat_instrument_unknown_family_returns_404(api_client: TestClient) -> None:
  """GET /api/v1/stats/{family}/{instrument} returns 404 when the family itself is absent."""
  response = api_client.get("/api/v1/stats/unknown_family/NQ")

  assert response.status_code == 404
  body = response.json()
  assert body["data"] is None
  assert "not found" in body["error"].lower()


# ---------------------------------------------------------------------------
# GET /api/v1/stats/{family}/{instrument}/{timeframe} — single timeframe
# ---------------------------------------------------------------------------

def test_get_stat_timeframe_returns_timeframe_result(api_client: TestClient) -> None:
  """GET /api/v1/stats/{family}/{instrument}/{timeframe} returns the TimeframeResult."""
  response = api_client.get("/api/v1/stats/alpha_stat/NQ/1h")

  assert response.status_code == 200
  body = response.json()
  assert body["error"] is None

  data = body["data"]
  assert data["total_samples"] == 100
  assert len(data["results"]) == 1
  assert data["data_range"] == ["2023-01-02", "2023-12-29"]


def test_get_stat_timeframe_unknown_timeframe_returns_404(api_client: TestClient) -> None:
  """GET /api/v1/stats/{family}/{instrument}/{timeframe} returns 404 for an absent timeframe."""
  response = api_client.get("/api/v1/stats/alpha_stat/NQ/5m")

  assert response.status_code == 404
  body = response.json()
  assert body["data"] is None
  assert "5m" in body["error"]


def test_get_stat_timeframe_unknown_instrument_returns_404(api_client: TestClient) -> None:
  """GET /api/v1/stats/{family}/{instrument}/{timeframe} returns 404 for an absent instrument."""
  response = api_client.get("/api/v1/stats/alpha_stat/ES/1h")

  assert response.status_code == 404
  body = response.json()
  assert body["data"] is None
  assert "ES" in body["error"]


# ---------------------------------------------------------------------------
# ?slice= query parameter — requires a family with declared dimensions
# ---------------------------------------------------------------------------

def test_slice_on_family_endpoint_narrows_slices(sliced_api_client: TestClient) -> None:
  """?slice=weekday on GET /{family} keeps only the weekday key in every timeframe's slices."""
  response = sliced_api_client.get("/api/v1/stats/sliced_stat?slice=weekday")

  assert response.status_code == 200
  body = response.json()
  assert body["error"] is None

  data = body["data"]
  for tf in data["instruments"]["NQ"].values():
    assert list(tf["slices"].keys()) == ["weekday"]


def test_slice_unknown_dimension_returns_400(sliced_api_client: TestClient) -> None:
  """?slice=nonexistent returns 400 with an error mentioning 'Unknown slice'."""
  response = sliced_api_client.get("/api/v1/stats/sliced_stat?slice=nonexistent")

  assert response.status_code == 400
  body = response.json()
  assert body["data"] is None
  assert "Unknown slice" in body["error"]


def test_slice_on_timeframe_endpoint_narrows_slices(sliced_api_client: TestClient) -> None:
  """?slice=weekday on GET /{family}/{instrument}/{timeframe} keeps only weekday in slices."""
  response = sliced_api_client.get("/api/v1/stats/sliced_stat/NQ/1h?slice=weekday")

  assert response.status_code == 200
  body = response.json()
  assert body["error"] is None

  data = body["data"]
  assert list(data["slices"].keys()) == ["weekday"]
