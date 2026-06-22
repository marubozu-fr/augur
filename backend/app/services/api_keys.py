"""API key service — generation, hashing, persistence, and resolution."""

import hashlib
import secrets
import sqlite3
from pathlib import Path

from backend.app.auth.models import ApiKeyCreated, ApiKeyOut
from backend.app.repositories import api_keys as api_keys_repo


def generate_key() -> str:
  """Return a cryptographically random 64-character hex key.

  Generates 32 random bytes encoded as hex (256 bits of entropy).
  """
  return secrets.token_hex(32)


def hash_key(plaintext: str) -> str:
  """Return the SHA-256 hex digest of the plaintext key.

  SHA-256 is appropriate here because keys are cryptographically random
  (high entropy), unlike passwords which need a slow KDF.
  """
  return hashlib.sha256(plaintext.encode()).hexdigest()


def create_key(
  name: str,
  db_path: Path | None = None,
) -> tuple[str, ApiKeyCreated]:
  """Generate a new API key, persist its hash, and return it once.

  Args:
    name: Human-readable label for this key.
    db_path: Optional override for the database path.

  Returns:
    A tuple of (plaintext_key, ApiKeyCreated). The plaintext is returned
    exactly once and is not stored — subsequent calls cannot retrieve it.
  """
  plaintext = generate_key()
  key_hash = hash_key(plaintext)
  row_id = api_keys_repo.create_api_key(key_hash, name, db_path)
  row = api_keys_repo.get_api_key_by_id(row_id, db_path)
  # row is guaranteed to exist: we just inserted it in the same db_path.
  assert row is not None
  created = ApiKeyCreated(
    id=row["id"],
    name=row["name"],
    role=row["role"],
    created_at=row["created_at"],
    key=plaintext,
  )
  return plaintext, created


def list_keys(
  db_path: Path | None = None,
) -> list[ApiKeyOut]:
  """Return all API keys as ApiKeyOut instances (no hashes or plaintext).

  Args:
    db_path: Optional override for the database path.
  """
  rows = api_keys_repo.list_api_keys(db_path)
  return [_row_to_api_key_out(row) for row in rows]


def revoke_key(
  key_id: int,
  db_path: Path | None = None,
) -> bool:
  """Revoke an API key by primary key.

  Args:
    key_id: Primary key of the api_keys row to revoke.
    db_path: Optional override for the database path.

  Returns:
    True if the key was revoked, False if not found or already revoked.
  """
  rowcount = api_keys_repo.revoke_api_key(key_id, db_path)
  return rowcount > 0


def resolve_api_key(
  plaintext: str,
  db_path: Path | None = None,
) -> sqlite3.Row | None:
  """Look up an API key by its plaintext value.

  Hashes the plaintext and queries the database. Returns the raw row
  (including revoked_at) so the caller can distinguish 401 from 403.

  Args:
    plaintext: The raw key string from the X-API-Key header.
    db_path: Optional override for the database path.

  Returns:
    A sqlite3.Row if a matching key exists (active or revoked), else None.
  """
  key_hash = hash_key(plaintext)
  return api_keys_repo.get_api_key_by_hash(key_hash, db_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _row_to_api_key_out(row: sqlite3.Row) -> ApiKeyOut:
  """Map a sqlite3.Row from api_keys to an ApiKeyOut model."""
  revoked_at: str | None = row["revoked_at"]
  return ApiKeyOut(
    id=row["id"],
    name=row["name"],
    role=row["role"],
    created_at=row["created_at"],
    revoked_at=revoked_at,
    status="revoked" if revoked_at is not None else "active",
  )
