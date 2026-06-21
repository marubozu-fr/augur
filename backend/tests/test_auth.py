"""Tests for the auth layer: login, logout, /auth/me, and admin-protected routes.

All tests use a temporary SQLite database isolated from the real augur.db.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import settings
from backend.app.core.db import init_db
from backend.app.core.stats_loader import StatsLoader
from backend.app.main import create_app
from backend.app.repositories import users as users_repo
from backend.app.services.auth import hash_password
from backend.tests.conftest import make_stat_run_result, write_stat_result


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_app(tmp_db: Path, tmp_results: Path) -> tuple:
  """Build a FastAPI app + TestClient with isolated DB and results dir."""
  init_db(tmp_db)

  # Insert a known admin and a known reader directly.
  users_repo.create_user("admin_user", hash_password("admin_pass"), "admin", tmp_db)
  users_repo.create_user("reader_user", hash_password("reader_pass"), "reader", tmp_db)

  # Provide at least one stat family so list_stats returns a real result.
  write_stat_result(make_stat_run_result("test_stat"), tmp_results)
  loader = StatsLoader(results_dir=tmp_results)
  loader.load_all()

  app = create_app()

  # Override stats loader so tests don't need the real results/ directory.
  from backend.app.core.dependencies import get_stats_loader  # noqa: PLC0415
  app.dependency_overrides[get_stats_loader] = lambda: loader

  # Patch settings.db_path so auth services route to the temp database.
  settings.db_path = tmp_db

  client = TestClient(app, raise_server_exceptions=True)
  return app, client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def auth_client(tmp_path: Path):
  """Return (app, TestClient) with isolated DB pre-populated with admin+reader."""
  tmp_db = tmp_path / "test.db"
  tmp_results = tmp_path / "results"
  tmp_results.mkdir()
  app, client = _make_app(tmp_db, tmp_results)
  yield app, client
  # Reset settings.db_path to avoid leaking into other tests.
  settings.db_path = Path(__file__).resolve().parents[3] / "backend" / "db" / "augur.db"


@pytest.fixture()
def logged_in_admin(auth_client):
  """Return a TestClient that is already logged in as the admin user."""
  app, client = auth_client
  resp = client.post("/auth/login", json={"username": "admin_user", "password": "admin_pass"})
  assert resp.status_code == 200, resp.json()
  return client


@pytest.fixture()
def logged_in_reader(auth_client):
  """Return a TestClient that is already logged in as the reader user."""
  app, client = auth_client
  resp = client.post("/auth/login", json={"username": "reader_user", "password": "reader_pass"})
  assert resp.status_code == 200, resp.json()
  return client


# ---------------------------------------------------------------------------
# Login tests
# ---------------------------------------------------------------------------

def test_login_success_returns_200_and_user(auth_client) -> None:
  """POST /auth/login with correct credentials returns 200 and the user payload."""
  _, client = auth_client
  resp = client.post("/auth/login", json={"username": "admin_user", "password": "admin_pass"})

  assert resp.status_code == 200
  body = resp.json()
  assert body["error"] is None
  assert body["data"]["username"] == "admin_user"
  assert body["data"]["role"] == "admin"
  assert "password_hash" not in body["data"]


def test_login_success_sets_session_cookie(auth_client) -> None:
  """POST /auth/login sets an HttpOnly session cookie on success."""
  _, client = auth_client
  resp = client.post("/auth/login", json={"username": "admin_user", "password": "admin_pass"})

  assert resp.status_code == 200
  cookie = resp.cookies.get(settings.session_cookie_name)
  assert cookie is not None
  assert len(cookie) > 0


def test_login_wrong_password_returns_401(auth_client) -> None:
  """POST /auth/login with wrong password returns 401 and error envelope."""
  _, client = auth_client
  resp = client.post("/auth/login", json={"username": "admin_user", "password": "wrong"})

  assert resp.status_code == 401
  body = resp.json()
  assert body["data"] is None
  assert body["error"] is not None


def test_login_unknown_user_returns_401(auth_client) -> None:
  """POST /auth/login with unknown username returns 401."""
  _, client = auth_client
  resp = client.post("/auth/login", json={"username": "nobody", "password": "pass"})

  assert resp.status_code == 401
  body = resp.json()
  assert body["data"] is None
  assert body["error"] is not None


# ---------------------------------------------------------------------------
# Logout tests
# ---------------------------------------------------------------------------

def test_logout_clears_session_and_cookie(logged_in_admin: TestClient) -> None:
  """POST /auth/logout removes the session from the DB and clears the cookie."""
  client = logged_in_admin

  # Should be authenticated before logout.
  assert client.get("/auth/me").status_code == 200

  resp = client.post("/auth/logout")
  assert resp.status_code == 200

  # After logout the session is gone — /auth/me must reject.
  me_resp = client.get("/auth/me")
  assert me_resp.status_code == 401


# ---------------------------------------------------------------------------
# /auth/me tests
# ---------------------------------------------------------------------------

def test_me_returns_current_user(logged_in_admin: TestClient) -> None:
  """GET /auth/me returns the authenticated user's profile."""
  resp = logged_in_admin.get("/auth/me")

  assert resp.status_code == 200
  body = resp.json()
  assert body["data"]["username"] == "admin_user"
  assert body["data"]["role"] == "admin"


def test_me_returns_401_when_unauthenticated(auth_client) -> None:
  """GET /auth/me with no cookie returns 401."""
  _, client = auth_client
  resp = client.get("/auth/me")
  assert resp.status_code == 401


def test_me_returns_401_for_invalid_token(auth_client) -> None:
  """GET /auth/me with a bogus session token returns 401."""
  _, client = auth_client
  client.cookies.set(settings.session_cookie_name, "bogus_token_that_does_not_exist")
  resp = client.get("/auth/me")
  assert resp.status_code == 401


# ---------------------------------------------------------------------------
# /admin/stats access control
# ---------------------------------------------------------------------------

def test_admin_stats_requires_auth(auth_client) -> None:
  """GET /admin/stats returns 401 when unauthenticated."""
  _, client = auth_client
  resp = client.get("/admin/stats")
  assert resp.status_code == 401


def test_admin_stats_accessible_to_reader(logged_in_reader: TestClient) -> None:
  """GET /admin/stats returns 200 for a reader-role user."""
  resp = logged_in_reader.get("/admin/stats")
  assert resp.status_code == 200
  body = resp.json()
  assert body["error"] is None
  assert isinstance(body["data"], list)


def test_admin_stats_accessible_to_admin(logged_in_admin: TestClient) -> None:
  """GET /admin/stats returns 200 for an admin-role user."""
  resp = logged_in_admin.get("/admin/stats")
  assert resp.status_code == 200


def test_admin_stats_family_requires_auth(auth_client) -> None:
  """GET /admin/stats/{family} returns 401 when unauthenticated."""
  _, client = auth_client
  resp = client.get("/admin/stats/test_stat")
  assert resp.status_code == 401


def test_admin_stats_family_accessible_to_reader(logged_in_reader: TestClient) -> None:
  """GET /admin/stats/{family} is accessible to reader role."""
  resp = logged_in_reader.get("/admin/stats/test_stat")
  assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /admin/stats/reload — admin only
# ---------------------------------------------------------------------------

def test_reload_requires_auth(auth_client) -> None:
  """POST /admin/stats/reload returns 401 when unauthenticated."""
  _, client = auth_client
  resp = client.post("/admin/stats/reload")
  assert resp.status_code == 401


def test_reload_forbidden_for_reader(logged_in_reader: TestClient) -> None:
  """POST /admin/stats/reload returns 403 for a reader-role user."""
  resp = logged_in_reader.post("/admin/stats/reload")
  assert resp.status_code == 403


def test_reload_allowed_for_admin(logged_in_admin: TestClient) -> None:
  """POST /admin/stats/reload returns 200 for an admin-role user."""
  resp = logged_in_admin.post("/admin/stats/reload")
  assert resp.status_code == 200
  body = resp.json()
  assert body["error"] is None


# ---------------------------------------------------------------------------
# Expired / invalid session
# ---------------------------------------------------------------------------

def test_expired_session_returns_401(tmp_path: Path) -> None:
  """A session with an expires_at in the past is rejected with 401."""
  tmp_db = tmp_path / "test.db"
  tmp_results = tmp_path / "results"
  tmp_results.mkdir()

  init_db(tmp_db)
  users_repo.create_user("u", hash_password("p"), "admin", tmp_db)

  # Insert an already-expired session directly.
  from backend.app.repositories import sessions as sessions_repo  # noqa: PLC0415
  sessions_repo.create_session("expired_token", 1, "2000-01-01 00:00:00", tmp_db)

  settings.db_path = tmp_db

  write_stat_result(make_stat_run_result("x"), tmp_results)
  loader = StatsLoader(results_dir=tmp_results)
  loader.load_all()

  app = create_app()
  from backend.app.core.dependencies import get_stats_loader  # noqa: PLC0415
  app.dependency_overrides[get_stats_loader] = lambda: loader

  client = TestClient(app)
  client.cookies.set(settings.session_cookie_name, "expired_token")
  resp = client.get("/auth/me")
  assert resp.status_code == 401

  # Reset
  settings.db_path = Path(__file__).resolve().parents[3] / "backend" / "db" / "augur.db"
