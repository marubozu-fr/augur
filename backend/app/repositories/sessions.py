"""Session repository — parameterized SQLite queries for the sessions table."""

import sqlite3
from pathlib import Path

from backend.app.core.db import get_connection


def create_session(
  token: str,
  user_id: int,
  expires_at: str,
  db_path: Path | None = None,
) -> None:
  """Insert a new session row.

  Args:
    token: Cryptographically random URL-safe token (the session identifier).
    user_id: FK to users.id.
    expires_at: ISO-8601 datetime string for expiry (UTC).
    db_path: Optional override for the database path.
  """
  with get_connection(db_path) as conn:
    conn.execute(
      "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
      (token, user_id, expires_at),
    )


def get_session(
  token: str,
  db_path: Path | None = None,
) -> sqlite3.Row | None:
  """Return a non-expired session joined to its user, or None.

  The query filters out rows whose expires_at is in the past so callers
  never need to check expiry themselves.
  """
  with get_connection(db_path) as conn:
    return conn.execute(
      """
      SELECT s.token, s.expires_at,
             u.id AS user_id, u.username, u.role
      FROM   sessions s
      JOIN   users u ON u.id = s.user_id
      WHERE  s.token = ?
        AND  s.expires_at > datetime('now')
      """,
      (token,),
    ).fetchone()


def delete_session(
  token: str,
  db_path: Path | None = None,
) -> None:
  """Delete a session by token (logout)."""
  with get_connection(db_path) as conn:
    conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


def prune_expired_sessions(db_path: Path | None = None) -> int:
  """Delete all expired sessions and return the number of rows removed."""
  with get_connection(db_path) as conn:
    cursor = conn.execute(
      "DELETE FROM sessions WHERE expires_at <= datetime('now')"
    )
    return cursor.rowcount
