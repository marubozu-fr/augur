"""User repository — parameterized queries for the users table."""

from pathlib import Path

from backend.app.core.db import RowMapping, get_connection


def get_user_by_username(
  username: str,
  db_path: Path | None = None,
) -> RowMapping | None:
  """Return a user row by username, or None if not found."""
  with get_connection(db_path) as conn:
    return conn.execute(
      "SELECT id, username, password_hash, role, created_at FROM users WHERE username = ?",
      (username,),
    ).fetchone()


def get_user_by_id(
  user_id: int,
  db_path: Path | None = None,
) -> RowMapping | None:
  """Return a user row by primary key, or None if not found."""
  with get_connection(db_path) as conn:
    return conn.execute(
      "SELECT id, username, password_hash, role, created_at FROM users WHERE id = ?",
      (user_id,),
    ).fetchone()


def create_user(
  username: str,
  password_hash: str,
  role: str,
  db_path: Path | None = None,
) -> int:
  """Insert a new user and return the new row id.

  Args:
    username: Unique login name.
    password_hash: bcrypt hash string.
    role: 'admin' or 'reader'.
    db_path: Optional override for the database path.

  Returns:
    The integer primary key of the inserted row.
  """
  with get_connection(db_path) as conn:
    return conn.insert(
      "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
      (username, password_hash, role),
    )
