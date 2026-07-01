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
  make_sampled_stat_run_result,
  make_sliced_stat_run_result,
  make_stat_run_result,
  write_stat_result,
)
from stats.base import SampleRow, StatResultRow


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


@pytest.fixture()
def sampled_api_client(tmp_path: Path) -> TestClient:
  """TestClient with a family carrying per-day samples (issue #179).

  10 samples, one per month Jan-Oct 2023, condition 'cond_a'. Outcome
  alternates: odd months -> 'green', even months -> 'red'. Two result rows
  (green/red) give date filtering real (condition, outcome) pairs to
  reaggregate. require_api_key is bypassed as in the other fixtures.
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

  app = create_app()
  app.dependency_overrides[get_stats_loader] = lambda: loader
  app.dependency_overrides[require_api_key] = lambda: None
  return TestClient(app)


@pytest.fixture()
def undeclared_agg_api_client(tmp_path: Path) -> TestClient:
  """TestClient with a decomposable magnitude row missing its `agg` field.

  Samples for the row's (condition, outcome) pair carry non-None values, so
  has_undeclared_agg_metadata() should flag this as a family that needs
  regeneration rather than the non-decomposable Pearson-r case.
  """
  results_dir = tmp_path / "results"
  results_dir.mkdir()
  samples = [
    SampleRow(date=f"2023-{month:02d}-01", condition="cond_a", outcome="out_x", value=float(month))
    for month in range(1, 11)
  ]
  results = [
    StatResultRow(
      condition="cond_a", outcome="out_x", count=10, total=10,
      probability=0.0, baseline_prob=0.0, baseline_n=0,
      value=5.5, value_baseline=None, agg=None,
    ),
  ]
  write_stat_result(
    make_sampled_stat_run_result("undeclared_agg_stat", samples=samples, results=results),
    results_dir,
  )

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


# ---------------------------------------------------------------------------
# ?start= / ?end= date-range filter (issue #179)
# ---------------------------------------------------------------------------

def test_date_filter_malformed_start_returns_400(api_client: TestClient) -> None:
  """?start=not-a-date returns 400 with the validate_date_params message."""
  response = api_client.get("/api/v1/stats/alpha_stat/NQ/1h?start=not-a-date")

  assert response.status_code == 400
  body = response.json()
  assert body["data"] is None
  assert "Invalid" in body["error"]


def test_date_filter_start_after_end_returns_400(api_client: TestClient) -> None:
  """?start after ?end returns 400 with the validate_date_params message."""
  response = api_client.get(
    "/api/v1/stats/alpha_stat/NQ/1h?start=2023-06-01&end=2023-01-01"
  )

  assert response.status_code == 400
  body = response.json()
  assert body["data"] is None
  assert "must not be after" in body["error"]


def test_date_filter_no_samples_returns_400(api_client: TestClient) -> None:
  """A date param on a family with no samples returns the exact 'not available' message."""
  response = api_client.get(
    "/api/v1/stats/alpha_stat/NQ/1h?start=2023-01-01&end=2023-06-01"
  )

  assert response.status_code == 400
  body = response.json()
  assert body["data"] is None
  assert body["error"] == (
    "Date filtering is not available for this stat family. "
    "Results must be regenerated with samples support."
  )


def test_date_filter_missing_agg_metadata_returns_400(
  undeclared_agg_api_client: TestClient,
) -> None:
  """A decomposable magnitude row missing `agg` returns the exact regeneration message."""
  response = undeclared_agg_api_client.get(
    "/api/v1/stats/undeclared_agg_stat/NQ/1h?start=2023-01-01&end=2023-12-31"
  )

  assert response.status_code == 400
  body = response.json()
  assert body["data"] is None
  assert body["error"] == (
    "Date filtering requires result metadata (agg field). Results must be regenerated."
  )


def test_date_filter_absent_params_no_regression(sampled_api_client: TestClient) -> None:
  """Without start/end, a samples-backed family behaves exactly as before (unfiltered)."""
  response = sampled_api_client.get("/api/v1/stats/sampled_stat/NQ/1h")

  assert response.status_code == 200
  body = response.json()
  assert body["error"] is None

  data = body["data"]
  assert data["total_samples"] == 10
  assert data["data_range"] == ["2023-01-01", "2023-10-01"]
  assert len(data["results"]) == 2
  for row in data["results"]:
    assert row["count"] == 5
    assert row["total"] == 10
    assert row["probability"] == pytest.approx(0.5)


def test_date_filter_happy_path_filters_and_reaggregates(
  sampled_api_client: TestClient,
) -> None:
  """?start=2023-03-01&end=2023-06-01 filters to Mar-Jun and reaggregates counts.

  Sample outcomes: Mar (odd month) -> green, Apr (even) -> red,
  May (odd) -> green, Jun (even) -> red. So green count=2, red count=2,
  total=4 for each outcome, P(green)=P(red)=2/4=0.5.
  """
  response = sampled_api_client.get(
    "/api/v1/stats/sampled_stat/NQ/1h?start=2023-03-01&end=2023-06-01"
  )

  assert response.status_code == 200
  body = response.json()
  assert body["error"] is None

  data = body["data"]
  assert data["total_samples"] == 4
  assert data["data_range"] == ["2023-03-01", "2023-06-01"]

  green_row = next(r for r in data["results"] if r["outcome"] == "green")
  red_row = next(r for r in data["results"] if r["outcome"] == "red")
  assert green_row["count"] == 2
  assert green_row["total"] == 4
  assert green_row["probability"] == pytest.approx(0.5)
  assert red_row["count"] == 2
  assert red_row["total"] == 4
  assert red_row["probability"] == pytest.approx(0.5)


def test_date_filter_combined_with_slice_returns_400(
  sampled_api_client: TestClient,
) -> None:
  """Combining ?slice= with date filtering returns 400 (unsupported in MVP).

  Date filtering drops slices, so the combination cannot yield a sliced view;
  the endpoint rejects it explicitly instead of returning empty slices.
  """
  response = sampled_api_client.get(
    "/api/v1/stats/sampled_stat/NQ/1h?start=2023-03-01&slice=weekday"
  )

  assert response.status_code == 400
  body = response.json()
  assert body["data"] is None
  assert body["error"] == "Slice filtering is not available when using date range filtering."
