"""SQLite connection helper and schema initialisation."""

import sqlite3
from pathlib import Path

from backend.app.core.config import settings


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
  """Open a SQLite connection with Row factory and foreign-key enforcement.

  Args:
    db_path: Override the database path (useful in tests). Defaults to
      settings.db_path.

  Returns:
    A configured sqlite3.Connection.
  """
  path = db_path or settings.db_path
  conn = sqlite3.connect(str(path), detect_types=sqlite3.PARSE_DECLTYPES)
  conn.row_factory = sqlite3.Row
  conn.execute("PRAGMA foreign_keys = ON")
  return conn


def init_db(db_path: Path | None = None) -> None:
  """Create the auth tables if they do not already exist (idempotent).

  Tables created:
    - users: id, username (unique), password_hash, role, created_at
    - sessions: token (PK), user_id (FK → users.id CASCADE), created_at, expires_at
    - api_keys: id, key_hash (unique), name, role, created_at, revoked_at
  """
  path = db_path or settings.db_path
  path.parent.mkdir(parents=True, exist_ok=True)
  with get_connection(db_path) as conn:
    conn.executescript("""
      CREATE TABLE IF NOT EXISTS users (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        username    TEXT    NOT NULL UNIQUE,
        password_hash TEXT  NOT NULL,
        role        TEXT    NOT NULL CHECK(role IN ('admin', 'reader')),
        created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
      );

      CREATE TABLE IF NOT EXISTS sessions (
        token       TEXT    PRIMARY KEY,
        user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
        expires_at  TEXT    NOT NULL
      );

      CREATE TABLE IF NOT EXISTS api_keys (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        key_hash    TEXT    NOT NULL UNIQUE,
        name        TEXT    NOT NULL,
        role        TEXT    NOT NULL DEFAULT 'reader' CHECK(role IN ('reader')),
        created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
        revoked_at  TEXT
      );
    """)
