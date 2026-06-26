"""Tests for seed_default_api_key() (issue #150).

Covers:
  1. No-op when settings.default_api_key is empty — zero keys inserted.
  2. Seeds one key with the correct name, role='reader', and active status.
  3. Idempotent — calling seed_default_api_key() twice yields exactly one key.
  4. End-to-end — the seeded plaintext authenticates /api/v1/stats (200);
     a wrong key and a missing key both return 401.

All tests run on the SQLite path (no DATABASE_URL set) against a temp database
isolated via tmp_path.  Settings mutations are always restored in teardown so
they do not leak across tests.
"""

from pathlib import Path

import pytest

from backend.app.core.config import settings
from backend.app.core.db import init_db
from backend.app.services.api_keys import list_keys, seed_default_api_key
from backend.tests.conftest import make_test_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fresh_db(tmp_path: Path) -> Path:
  """Return the path to an initialised, isolated SQLite database.

  No users or API keys are pre-created — the DB is a blank slate on which
  seed_default_api_key() can be exercised directly.
  """
  db = tmp_path / "seed_test.db"
  init_db(db)
  return db


@pytest.fixture()
def patched_default_key():
  """Set settings.default_api_key/name to known test values; restore after the test.

  Yields (plaintext, name) so tests can assert against the exact values used.
  """
  original_key = settings.default_api_key
  original_name = settings.default_api_key_name
  settings.default_api_key = "test-seed-plaintext-key-abc123xyz000"
  settings.default_api_key_name = "seeder-test-key"
  yield settings.default_api_key, settings.default_api_key_name
  settings.default_api_key = original_key
  settings.default_api_key_name = original_name


@pytest.fixture()
def seeded_v1_client(tmp_path: Path):
  """TestClient with isolated DB pre-seeded via seed_default_api_key().

  make_test_app() is used so the app gets a real DB path override and a
  synthetic stats loader, matching the pattern in test_api_keys.py.
  Yields (client, plaintext) — the plaintext is the seeded key value.
  All settings mutations are restored in teardown.
  """
  tmp_db = tmp_path / "e2e_test.db"
  tmp_results = tmp_path / "results"
  tmp_results.mkdir()

  original_db_path = settings.db_path
  original_key = settings.default_api_key
  original_name = settings.default_api_key_name

  plaintext = "e2e-seed-plaintext-api-key-00000001"
  settings.default_api_key = plaintext
  settings.default_api_key_name = "e2e-seeded-key"

  # make_test_app calls init_db(tmp_db) and sets settings.db_path = tmp_db.
  _, client = make_test_app(tmp_db, tmp_results)
  seed_default_api_key(tmp_db)

  yield client, plaintext

  settings.db_path = original_db_path
  settings.default_api_key = original_key
  settings.default_api_key_name = original_name


# ---------------------------------------------------------------------------
# Test 1: No-op when default_api_key is empty
# ---------------------------------------------------------------------------

def test_seed_noop_when_default_api_key_empty(fresh_db: Path) -> None:
  """seed_default_api_key() inserts nothing when settings.default_api_key is ''.

  Expected: list_keys() returns an empty list — zero rows in api_keys.
  """
  original = settings.default_api_key
  try:
    settings.default_api_key = ""
    seed_default_api_key(fresh_db)
    # Hand-calculation: seeder is a no-op → 0 keys.
    assert list_keys(fresh_db) == []
  finally:
    settings.default_api_key = original


# ---------------------------------------------------------------------------
# Test 2: Seeds a key with the correct name, role, and active status
# ---------------------------------------------------------------------------

def test_seed_creates_key_with_correct_name_role_and_status(
  fresh_db: Path,
  patched_default_key: tuple[str, str],
) -> None:
  """seed_default_api_key() inserts exactly one active reader key.

  Expected: 1 key, name matches settings.default_api_key_name,
  role == 'reader', status == 'active', revoked_at is None.
  """
  plaintext, name = patched_default_key
  seed_default_api_key(fresh_db)

  keys = list_keys(fresh_db)
  # Hand-calculation: one call with non-empty key → exactly 1 row.
  assert len(keys) == 1
  key = keys[0]
  assert key.name == name
  assert key.role == "reader"
  assert key.status == "active"
  assert key.revoked_at is None


# ---------------------------------------------------------------------------
# Test 3: Idempotent — two calls yield exactly one key
# ---------------------------------------------------------------------------

def test_seed_is_idempotent(
  fresh_db: Path,
  patched_default_key: tuple[str, str],
) -> None:
  """seed_default_api_key() called twice inserts exactly one row, not two.

  Expected: after two calls the same hash already exists on the second call,
  so the seeder skips the insert → still 1 key.
  """
  seed_default_api_key(fresh_db)
  seed_default_api_key(fresh_db)
  # Hand-calculation: second call finds existing hash → skips → 1 row total.
  assert len(list_keys(fresh_db)) == 1


# ---------------------------------------------------------------------------
# Test 4: End-to-end — seeded plaintext authenticates /api/v1/stats
# ---------------------------------------------------------------------------

def test_e2e_seeded_key_returns_200_on_v1_stats(seeded_v1_client: tuple) -> None:
  """The plaintext seeded by seed_default_api_key() grants 200 on /api/v1/stats."""
  client, plaintext = seeded_v1_client
  resp = client.get("/api/v1/stats", headers={"X-API-Key": plaintext})
  assert resp.status_code == 200
  body = resp.json()
  assert body["error"] is None
  assert isinstance(body["data"], list)


def test_e2e_wrong_key_returns_401_on_v1_stats(seeded_v1_client: tuple) -> None:
  """A wrong X-API-Key header returns 401 on /api/v1/stats (key not in DB)."""
  client, _ = seeded_v1_client
  resp = client.get("/api/v1/stats", headers={"X-API-Key": "wrong-key-not-seeded"})
  assert resp.status_code == 401


def test_e2e_missing_key_returns_401_on_v1_stats(seeded_v1_client: tuple) -> None:
  """A missing X-API-Key header returns 401 on /api/v1/stats."""
  client, _ = seeded_v1_client
  resp = client.get("/api/v1/stats")
  assert resp.status_code == 401
