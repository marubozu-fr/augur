"""API key repository — parameterized queries for the api_keys table."""

from pathlib import Path

from backend.app.core.db import RowMapping, get_connection


def create_api_key(
  key_hash: str,
  name: str,
  db_path: Path | None = None,
) -> int:
  """Insert a new API key row and return the new row id.

  The role defaults to 'reader' as defined in the schema.

  Args:
    key_hash: SHA-256 hex digest of the plaintext key.
    name: Human-readable label for this key.
    db_path: Optional override for the database path.

  Returns:
    The integer primary key of the inserted row.
  """
  with get_connection(db_path) as conn:
    return conn.insert(
      "INSERT INTO api_keys (key_hash, name) VALUES (?, ?)",
      (key_hash, name),
    )


def list_api_keys(
  db_path: Path | None = None,
) -> list[RowMapping]:
  """Return all API key rows ordered by creation date descending.

  key_hash is never selected — callers receive only the safe fields.

  Returns:
    A list of row mappings with columns: id, name, role, created_at, revoked_at.
  """
  with get_connection(db_path) as conn:
    return conn.execute(
      """
      SELECT id, name, role, created_at, revoked_at
      FROM   api_keys
      ORDER  BY created_at DESC
      """
    ).fetchall()


def get_api_key_by_hash(
  key_hash: str,
  db_path: Path | None = None,
) -> RowMapping | None:
  """Return an API key row by its SHA-256 hash, or None if not found.

  Returns revoked keys as well — callers must inspect revoked_at to decide
  whether to return 401 (not found) or 403 (revoked).

  Args:
    key_hash: SHA-256 hex digest of the plaintext key to look up.
    db_path: Optional override for the database path.

  Returns:
    Row mapping with columns id, name, role, created_at, revoked_at, or None.
  """
  with get_connection(db_path) as conn:
    return conn.execute(
      """
      SELECT id, name, role, created_at, revoked_at
      FROM   api_keys
      WHERE  key_hash = ?
      """,
      (key_hash,),
    ).fetchone()


def get_api_key_by_id(
  key_id: int,
  db_path: Path | None = None,
) -> RowMapping | None:
  """Return an API key row by primary key, or None if not found.

  Args:
    key_id: Primary key of the api_keys row.
    db_path: Optional override for the database path.

  Returns:
    Row mapping with columns id, name, role, created_at, revoked_at, or None.
  """
  with get_connection(db_path) as conn:
    return conn.execute(
      """
      SELECT id, name, role, created_at, revoked_at
      FROM   api_keys
      WHERE  id = ?
      """,
      (key_id,),
    ).fetchone()


def revoke_api_key(
  key_id: int,
  db_path: Path | None = None,
) -> int:
  """Set revoked_at to now for a key that is currently active.

  The WHERE clause guards against double-revocation: only rows where
  revoked_at IS NULL are updated.

  Args:
    key_id: Primary key of the api_keys row to revoke.
    db_path: Optional override for the database path.

  Returns:
    The number of rows updated (1 if revoked, 0 if not found or already revoked).
  """
  with get_connection(db_path) as conn:
    cursor = conn.execute(
      """
      UPDATE api_keys
      SET    revoked_at = CURRENT_TIMESTAMP
      WHERE  id = ?
        AND  revoked_at IS NULL
      """,
      (key_id,),
    )
    return cursor.rowcount
