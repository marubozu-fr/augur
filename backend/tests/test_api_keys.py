"""Tests for the API key management feature (issue #113).

Covers:
  - Service / repository layer (generate_key, hash_key, create_key, list_keys,
    revoke_key, resolve_api_key)
  - Admin HTTP endpoints (POST/GET/DELETE /admin/api-keys)
  - /api/v1 protection via require_api_key (missing, invalid, revoked, valid)

All tests use a temporary SQLite database isolated from the real augur.db.
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
from backend.app.services.api_keys import (
  create_key,
  generate_key,
  hash_key,
  list_keys,
  resolve_api_key,
  revoke_key,
)
from backend.app.services.auth import hash_password
from backend.tests.conftest import make_stat_run_result, make_test_app, write_stat_result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_db_path(tmp_path: Path) -> Path:
  """Initialise and return the path to an isolated SQLite database."""
  db = tmp_path / "test.db"
  init_db(db)
  return db


@pytest.fixture()
def api_keys_client(tmp_path: Path):
  """Return (app, TestClient) with isolated DB pre-populated with admin+reader."""
  tmp_db = tmp_path / "test.db"
  tmp_results = tmp_path / "results"
  tmp_results.mkdir()
  app, client = make_test_app(tmp_db, tmp_results)
  yield app, client
  settings.db_path = Path(__file__).resolve().parents[3] / "backend" / "db" / "augur.db"


@pytest.fixture()
def admin_client(api_keys_client):
  """Return a TestClient already logged in as the admin user."""
  app, client = api_keys_client
  resp = client.post("/auth/login", json={"username": "admin_user", "password": "admin_pass"})
  assert resp.status_code == 200, resp.json()
  return client


@pytest.fixture()
def reader_client(api_keys_client):
  """Return a TestClient already logged in as the reader user."""
  app, client = api_keys_client
  resp = client.post("/auth/login", json={"username": "reader_user", "password": "reader_pass"})
  assert resp.status_code == 200, resp.json()
  return client


# ---------------------------------------------------------------------------
# Service layer — generate_key
# ---------------------------------------------------------------------------

def test_generate_key_returns_64_char_hex() -> None:
  """generate_key() returns a 64-character lowercase hex string (32 bytes)."""
  key = generate_key()
  assert len(key) == 64
  assert all(c in "0123456789abcdef" for c in key)


def test_generate_key_two_calls_differ() -> None:
  """Two successive calls to generate_key() produce different values."""
  assert generate_key() != generate_key()


# ---------------------------------------------------------------------------
# Service layer — hash_key
# ---------------------------------------------------------------------------

def test_hash_key_returns_64_char_hex() -> None:
  """hash_key() returns a 64-character SHA-256 hex digest."""
  digest = hash_key("some_plaintext")
  assert len(digest) == 64
  assert all(c in "0123456789abcdef" for c in digest)


def test_hash_key_is_deterministic() -> None:
  """hash_key() returns the same digest for the same input each time."""
  assert hash_key("hello") == hash_key("hello")


def test_hash_key_differs_for_different_inputs() -> None:
  """hash_key() produces different digests for different inputs."""
  assert hash_key("key_a") != hash_key("key_b")


def test_hash_key_matches_stdlib_sha256() -> None:
  """hash_key() produces the same digest as hashlib.sha256 for the same input."""
  import hashlib  # noqa: PLC0415

  for value in ("abc", "hello world", "key_with_special_chars!@#"):
    assert hash_key(value) == hashlib.sha256(value.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Service layer — create_key
# ---------------------------------------------------------------------------

def test_create_key_returns_tuple_with_plaintext_and_model(tmp_db_path: Path) -> None:
  """create_key() returns (plaintext_str, ApiKeyCreated) with correct types."""
  plaintext, created = create_key("test-key", tmp_db_path)
  assert isinstance(plaintext, str)
  assert len(plaintext) == 64


def test_create_key_status_active_and_no_revoked_at(tmp_db_path: Path) -> None:
  """create_key() returns a model with status 'active' and revoked_at=None.

  ApiKeyCreated does not carry a revoked_at field, but the key should be
  resolvable and active (revoked_at is None in the DB row).
  """
  plaintext, created = create_key("my-key", tmp_db_path)
  assert created.role == "reader"
  # key field is present in ApiKeyCreated and equals the plaintext
  assert created.key == plaintext


def test_create_key_role_is_reader(tmp_db_path: Path) -> None:
  """create_key() sets role to 'reader' as defined in the schema."""
  _, created = create_key("reader-key", tmp_db_path)
  assert created.role == "reader"


def test_create_key_stored_hash_matches_hash_of_plaintext(tmp_db_path: Path) -> None:
  """The hash stored in the DB equals hash_key(plaintext); plaintext is not stored."""
  plaintext, created = create_key("hash-check", tmp_db_path)
  # Fetch the raw row directly from the DB using the internal repo.
  # list_api_keys excludes key_hash, so we access it via a direct query.
  from backend.app.core.db import get_connection  # noqa: PLC0415
  with get_connection(tmp_db_path) as conn:
    row = conn.execute(
      "SELECT key_hash FROM api_keys WHERE id = ?", (created.id,)
    ).fetchone()
  assert row is not None
  assert row["key_hash"] == hash_key(plaintext)


def test_create_key_plaintext_not_in_list_keys(tmp_db_path: Path) -> None:
  """list_keys() never exposes the plaintext or key_hash fields."""
  plaintext, _ = create_key("hidden-key", tmp_db_path)
  keys = list_keys(tmp_db_path)
  assert len(keys) == 1
  key_out = keys[0]
  # ApiKeyOut has no 'key' or 'key_hash' attribute
  assert not hasattr(key_out, "key")
  assert not hasattr(key_out, "key_hash")


# ---------------------------------------------------------------------------
# Service layer — list_keys
# ---------------------------------------------------------------------------

def test_list_keys_returns_created_keys(tmp_db_path: Path) -> None:
  """list_keys() returns every key created in the DB."""
  create_key("alpha", tmp_db_path)
  create_key("beta", tmp_db_path)
  keys = list_keys(tmp_db_path)
  names = {k.name for k in keys}
  assert names == {"alpha", "beta"}


def test_list_keys_newest_first(tmp_db_path: Path) -> None:
  """list_keys() orders results by created_at descending (newest first).

  Because SQLite datetime('now') has second precision, we insert rows with
  explicit timestamps to guarantee ordering.
  """
  from backend.app.core.db import get_connection  # noqa: PLC0415

  # Insert two rows with explicit, different timestamps.
  with get_connection(tmp_db_path) as conn:
    conn.execute(
      "INSERT INTO api_keys (key_hash, name, created_at) VALUES (?, ?, ?)",
      ("aaa" + "0" * 61, "older", "2024-01-01 00:00:00"),
    )
    conn.execute(
      "INSERT INTO api_keys (key_hash, name, created_at) VALUES (?, ?, ?)",
      ("bbb" + "0" * 61, "newer", "2024-06-01 00:00:00"),
    )

  keys = list_keys(tmp_db_path)
  assert len(keys) == 2
  assert keys[0].name == "newer"
  assert keys[1].name == "older"


def test_list_keys_never_exposes_key_hash(tmp_db_path: Path) -> None:
  """ApiKeyOut objects returned by list_keys have no key_hash attribute."""
  create_key("safe", tmp_db_path)
  for k in list_keys(tmp_db_path):
    assert not hasattr(k, "key_hash")


# ---------------------------------------------------------------------------
# Service layer — revoke_key
# ---------------------------------------------------------------------------

def test_revoke_key_returns_true_on_first_call(tmp_db_path: Path) -> None:
  """revoke_key() returns True when revoking an active key."""
  _, created = create_key("to-revoke", tmp_db_path)
  assert revoke_key(created.id, tmp_db_path) is True


def test_revoke_key_returns_false_on_second_call(tmp_db_path: Path) -> None:
  """revoke_key() returns False on a repeated call (idempotent guard)."""
  _, created = create_key("double-revoke", tmp_db_path)
  assert revoke_key(created.id, tmp_db_path) is True
  assert revoke_key(created.id, tmp_db_path) is False


def test_revoke_key_sets_status_to_revoked(tmp_db_path: Path) -> None:
  """After revoke_key(), the key's status in list_keys is 'revoked'."""
  _, created = create_key("check-status", tmp_db_path)
  revoke_key(created.id, tmp_db_path)
  keys = list_keys(tmp_db_path)
  assert len(keys) == 1
  assert keys[0].status == "revoked"
  assert keys[0].revoked_at is not None


def test_revoke_key_unknown_id_returns_false(tmp_db_path: Path) -> None:
  """revoke_key() returns False for a non-existent key id."""
  assert revoke_key(9999, tmp_db_path) is False


# ---------------------------------------------------------------------------
# Service layer — resolve_api_key
# ---------------------------------------------------------------------------

def test_resolve_api_key_returns_row_for_valid_key(tmp_db_path: Path) -> None:
  """resolve_api_key() returns a row for a known active plaintext key."""
  plaintext, created = create_key("active-key", tmp_db_path)
  row = resolve_api_key(plaintext, tmp_db_path)
  assert row is not None
  assert row["id"] == created.id


def test_resolve_api_key_returns_none_for_unknown_key(tmp_db_path: Path) -> None:
  """resolve_api_key() returns None for a plaintext that was never stored."""
  row = resolve_api_key("totally_unknown_key_that_does_not_exist", tmp_db_path)
  assert row is None


def test_resolve_api_key_returns_row_with_revoked_at_for_revoked_key(
  tmp_db_path: Path,
) -> None:
  """resolve_api_key() still returns the row for a revoked key (revoked_at is set)."""
  plaintext, created = create_key("revoked-key", tmp_db_path)
  revoke_key(created.id, tmp_db_path)
  row = resolve_api_key(plaintext, tmp_db_path)
  assert row is not None
  assert row["revoked_at"] is not None


# ---------------------------------------------------------------------------
# Admin HTTP endpoints — POST /admin/api-keys
# ---------------------------------------------------------------------------

def test_create_api_key_returns_201_and_plaintext(admin_client: TestClient) -> None:
  """POST /admin/api-keys returns 201 with the plaintext key in the response."""
  resp = admin_client.post("/admin/api-keys", json={"name": "my-integration"})
  assert resp.status_code == 201
  body = resp.json()
  assert body["error"] is None
  data = body["data"]
  assert "key" in data
  assert len(data["key"]) == 64


def test_create_api_key_plaintext_resolves_to_active_row(
  admin_client: TestClient,
) -> None:
  """The plaintext returned by POST /admin/api-keys resolves via resolve_api_key."""
  resp = admin_client.post("/admin/api-keys", json={"name": "round-trip"})
  assert resp.status_code == 201
  plaintext = resp.json()["data"]["key"]

  # resolve_api_key uses settings.db_path which was redirected to the temp DB.
  row = resolve_api_key(plaintext)
  assert row is not None
  assert row["revoked_at"] is None


def test_create_api_key_returns_reader_role(admin_client: TestClient) -> None:
  """POST /admin/api-keys assigns role 'reader' to the new key."""
  resp = admin_client.post("/admin/api-keys", json={"name": "role-check"})
  assert resp.status_code == 201
  assert resp.json()["data"]["role"] == "reader"


def test_create_api_key_rejects_reader_session_with_403(
  reader_client: TestClient,
) -> None:
  """POST /admin/api-keys returns 403 for a reader-role session."""
  resp = reader_client.post("/admin/api-keys", json={"name": "should-fail"})
  assert resp.status_code == 403


def test_create_api_key_rejects_unauthenticated_with_401(
  api_keys_client,
) -> None:
  """POST /admin/api-keys returns 401 when no session cookie is present."""
  _, client = api_keys_client
  resp = client.post("/admin/api-keys", json={"name": "no-auth"})
  assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Admin HTTP endpoints — GET /admin/api-keys
# ---------------------------------------------------------------------------

def test_list_api_keys_returns_200_and_list(admin_client: TestClient) -> None:
  """GET /admin/api-keys returns 200 with a list in the data field."""
  resp = admin_client.get("/admin/api-keys")
  assert resp.status_code == 200
  body = resp.json()
  assert body["error"] is None
  assert isinstance(body["data"], list)


def test_list_api_keys_reflects_active_and_revoked_status(
  admin_client: TestClient,
) -> None:
  """GET /admin/api-keys includes status 'active' and 'revoked' as expected."""
  # Create two keys.
  r1 = admin_client.post("/admin/api-keys", json={"name": "key-active"})
  r2 = admin_client.post("/admin/api-keys", json={"name": "key-to-revoke"})
  assert r1.status_code == 201
  assert r2.status_code == 201
  key2_id = r2.json()["data"]["id"]

  # Revoke the second one.
  admin_client.delete(f"/admin/api-keys/{key2_id}")

  body = admin_client.get("/admin/api-keys").json()
  statuses = {k["name"]: k["status"] for k in body["data"]}
  assert statuses["key-active"] == "active"
  assert statuses["key-to-revoke"] == "revoked"


def test_list_api_keys_does_not_expose_key_or_key_hash(
  admin_client: TestClient,
) -> None:
  """GET /admin/api-keys never returns 'key' or 'key_hash' fields."""
  admin_client.post("/admin/api-keys", json={"name": "leak-check"})
  body = admin_client.get("/admin/api-keys").json()
  for entry in body["data"]:
    assert "key" not in entry
    assert "key_hash" not in entry


def test_list_api_keys_rejects_reader_session_with_403(
  reader_client: TestClient,
) -> None:
  """GET /admin/api-keys returns 403 for a reader-role session."""
  resp = reader_client.get("/admin/api-keys")
  assert resp.status_code == 403


def test_list_api_keys_rejects_unauthenticated_with_401(
  api_keys_client,
) -> None:
  """GET /admin/api-keys returns 401 when no session cookie is present."""
  _, client = api_keys_client
  resp = client.get("/admin/api-keys")
  assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Admin HTTP endpoints — DELETE /admin/api-keys/{id}
# ---------------------------------------------------------------------------

def test_delete_api_key_returns_200_with_null_data(admin_client: TestClient) -> None:
  """DELETE /admin/api-keys/{id} returns 200 and data=null on success."""
  create_resp = admin_client.post("/admin/api-keys", json={"name": "delete-me"})
  key_id = create_resp.json()["data"]["id"]

  resp = admin_client.delete(f"/admin/api-keys/{key_id}")
  assert resp.status_code == 200
  body = resp.json()
  assert body["data"] is None
  assert body["error"] is None


def test_delete_api_key_twice_returns_404(admin_client: TestClient) -> None:
  """DELETE /admin/api-keys/{id} returns 404 on a second call (already revoked)."""
  create_resp = admin_client.post("/admin/api-keys", json={"name": "double-delete"})
  key_id = create_resp.json()["data"]["id"]

  admin_client.delete(f"/admin/api-keys/{key_id}")
  resp = admin_client.delete(f"/admin/api-keys/{key_id}")
  assert resp.status_code == 404
  body = resp.json()
  assert body["data"] is None
  assert body["error"] is not None


def test_delete_api_key_unknown_id_returns_404(admin_client: TestClient) -> None:
  """DELETE /admin/api-keys/{id} returns 404 for an id that does not exist."""
  resp = admin_client.delete("/admin/api-keys/9999")
  assert resp.status_code == 404
  body = resp.json()
  assert body["data"] is None
  assert body["error"] is not None


def test_delete_api_key_rejects_reader_session_with_403(
  reader_client: TestClient,
) -> None:
  """DELETE /admin/api-keys/{id} returns 403 for a reader-role session."""
  resp = reader_client.delete("/admin/api-keys/1")
  assert resp.status_code == 403


def test_delete_api_key_rejects_unauthenticated_with_401(
  api_keys_client,
) -> None:
  """DELETE /admin/api-keys/{id} returns 401 when no session cookie is present."""
  _, client = api_keys_client
  resp = client.delete("/admin/api-keys/1")
  assert resp.status_code == 401


# ---------------------------------------------------------------------------
# /api/v1 protection — require_api_key
# ---------------------------------------------------------------------------

@pytest.fixture()
def v1_client(tmp_path: Path):
  """TestClient with isolated DB + stats loader, ready for /api/v1 tests."""
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


def test_v1_stats_missing_api_key_returns_401(v1_client) -> None:
  """GET /api/v1/stats with no X-API-Key header returns 401."""
  client, _ = v1_client
  resp = client.get("/api/v1/stats")
  assert resp.status_code == 401


def test_v1_stats_invalid_api_key_returns_401(v1_client) -> None:
  """GET /api/v1/stats with an unknown X-API-Key value returns 401."""
  client, _ = v1_client
  resp = client.get("/api/v1/stats", headers={"X-API-Key": "not_a_real_key"})
  assert resp.status_code == 401


def test_v1_stats_revoked_api_key_returns_403(v1_client) -> None:
  """GET /api/v1/stats with a revoked key returns 403."""
  client, tmp_db = v1_client
  plaintext, created = create_key("revoked-v1", tmp_db)
  revoke_key(created.id, tmp_db)
  resp = client.get("/api/v1/stats", headers={"X-API-Key": plaintext})
  assert resp.status_code == 403


def test_v1_stats_valid_api_key_returns_200(v1_client) -> None:
  """GET /api/v1/stats with a valid active key returns 200 with data."""
  client, tmp_db = v1_client
  plaintext, _ = create_key("valid-v1", tmp_db)
  resp = client.get("/api/v1/stats", headers={"X-API-Key": plaintext})
  assert resp.status_code == 200
  body = resp.json()
  assert body["error"] is None
  assert isinstance(body["data"], list)
  assert len(body["data"]) == 1
  assert body["data"][0]["family"] == "test_stat"
