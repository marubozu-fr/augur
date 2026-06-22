"""Tests verifying that every HTTPException is wrapped in the ApiResponse envelope.

Each test asserts:
  - The expected HTTP status code.
  - body["data"] is None.
  - body["error"] is a non-empty string.
  - "detail" is absent from the body (old FastAPI default field).
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
from backend.app.services.api_keys import create_key, revoke_key
from backend.app.services.auth import hash_password
from backend.tests.conftest import make_stat_run_result, make_test_app, write_stat_result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def envelope_client(tmp_path: Path):
  """Return (app, TestClient) with isolated DB pre-populated with admin+reader."""
  tmp_db = tmp_path / "test.db"
  tmp_results = tmp_path / "results"
  tmp_results.mkdir()
  app, client = make_test_app(tmp_db, tmp_results)
  yield app, client
  settings.db_path = Path(__file__).resolve().parents[3] / "backend" / "db" / "augur.db"


@pytest.fixture()
def logged_in_reader(envelope_client):
  """Return a TestClient already logged in as the reader user."""
  app, client = envelope_client
  resp = client.post("/auth/login", json={"username": "reader_user", "password": "reader_pass"})
  assert resp.status_code == 200, resp.json()
  return client


@pytest.fixture()
def v1_envelope_client(tmp_path: Path):
  """Return (TestClient, tmp_db) with isolated DB + stats loader for /api/v1 tests."""
  tmp_db = tmp_path / "test.db"
  tmp_results = tmp_path / "results"
  tmp_results.mkdir()

  init_db(tmp_db)
  users_repo.create_user("admin_user", hash_password("admin_pass"), "admin", tmp_db)

  write_stat_result(make_stat_run_result("test_stat"), tmp_results)
  loader = StatsLoader(results_dir=tmp_results)
  loader.load_all()

  app = create_app()
  app.dependency_overrides[get_stats_loader] = lambda: loader
  settings.db_path = tmp_db

  client = TestClient(app, raise_server_exceptions=True)

  yield client, tmp_db

  settings.db_path = Path(__file__).resolve().parents[3] / "backend" / "db" / "augur.db"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _assert_error_envelope(body: dict, expected_status: int, actual_status: int) -> None:
  """Assert the standard error envelope shape."""
  assert actual_status == expected_status
  assert body["data"] is None
  assert isinstance(body["error"], str) and body["error"]
  assert "detail" not in body


# ---------------------------------------------------------------------------
# Dependency-raised 401 — unauthenticated /auth/me
# ---------------------------------------------------------------------------

def test_unauthenticated_me_returns_envelope_401(envelope_client) -> None:
  """GET /auth/me with no cookie returns an ApiResponse envelope, not bare detail."""
  _, client = envelope_client
  resp = client.get("/auth/me")
  _assert_error_envelope(resp.json(), 401, resp.status_code)


# ---------------------------------------------------------------------------
# Dependency-raised 401 — invalid session cookie on /auth/me
# ---------------------------------------------------------------------------

def test_invalid_cookie_me_returns_envelope_401(envelope_client) -> None:
  """GET /auth/me with a bogus session cookie returns an ApiResponse envelope."""
  _, client = envelope_client
  client.cookies.set(settings.session_cookie_name, "totally_bogus_session_token")
  resp = client.get("/auth/me")
  _assert_error_envelope(resp.json(), 401, resp.status_code)


# ---------------------------------------------------------------------------
# Dependency-raised 403 — reader hitting an admin-only route
# ---------------------------------------------------------------------------

def test_reader_reload_returns_envelope_403(logged_in_reader: TestClient) -> None:
  """POST /admin/stats/reload by a reader-role session returns an ApiResponse envelope."""
  resp = logged_in_reader.post("/admin/stats/reload")
  _assert_error_envelope(resp.json(), 403, resp.status_code)


# ---------------------------------------------------------------------------
# Dependency-raised 401 — missing X-API-Key on /api/v1/stats
# ---------------------------------------------------------------------------

def test_missing_api_key_returns_envelope_401(v1_envelope_client) -> None:
  """GET /api/v1/stats with no X-API-Key header returns an ApiResponse envelope."""
  client, _ = v1_envelope_client
  resp = client.get("/api/v1/stats")
  _assert_error_envelope(resp.json(), 401, resp.status_code)


# ---------------------------------------------------------------------------
# Dependency-raised 403 — revoked X-API-Key on /api/v1/stats
# ---------------------------------------------------------------------------

def test_revoked_api_key_returns_envelope_403(v1_envelope_client) -> None:
  """GET /api/v1/stats with a revoked key returns an ApiResponse envelope."""
  client, tmp_db = v1_envelope_client
  plaintext, created = create_key("revoke-envelope", tmp_db)
  revoke_key(created.id, tmp_db)
  resp = client.get("/api/v1/stats", headers={"X-API-Key": plaintext})
  _assert_error_envelope(resp.json(), 403, resp.status_code)
