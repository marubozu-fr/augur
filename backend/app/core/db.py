"""Dual-backend database connection helper (PostgreSQL or SQLite).

When settings.database_url is set, all connections go to Postgres via
psycopg2.  Otherwise the app falls back to SQLite — the default for local
development and the test suite.

Public API
----------
get_connection(db_path)  -- returns a _DBConnection context manager
init_db(db_path)         -- create tables idempotently
DBError                  -- tuple of exception classes for both backends
RowMapping               -- type alias for dict-like row objects
"""

import sqlite3
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras

from backend.app.core.config import settings

# ---------------------------------------------------------------------------
# Public type aliases
# ---------------------------------------------------------------------------

RowMapping = Mapping[str, Any]

# Catch-all error tuple — use `except DBError` in services / repos.
DBError = (sqlite3.Error, psycopg2.Error)


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _is_postgres() -> bool:
  """Return True when a Postgres DATABASE_URL is configured."""
  return bool(settings.database_url)


# ---------------------------------------------------------------------------
# Row normalisation for Postgres
# ---------------------------------------------------------------------------

def _normalize_pg_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
  """Convert datetime/date values to str in a Postgres result row.

  psycopg2 returns Python datetime objects for TIMESTAMPTZ columns.
  Pydantic models and repo callers expect plain strings — this brings
  Postgres rows in line with the string-based SQLite output.
  """
  if row is None:
    return None
  return {
    k: str(v) if isinstance(v, (datetime, date)) else v
    for k, v in row.items()
  }


# ---------------------------------------------------------------------------
# Cursor-result wrapper (Postgres only)
# ---------------------------------------------------------------------------

class _CursorResult:
  """Wraps a psycopg2 cursor so callers get the same interface as sqlite3.Cursor.

  Postgres-only: normalises datetime values to str on fetch.
  """

  def __init__(self, cursor: Any) -> None:
    self._cursor = cursor
    # Cache rowcount immediately — it stays valid after fetch calls.
    self._rowcount: int = cursor.rowcount

  @property
  def rowcount(self) -> int:
    """Number of rows affected by the last statement."""
    return self._rowcount

  def fetchone(self) -> dict[str, Any] | None:
    """Return the next row as a normalised dict, or None."""
    return _normalize_pg_row(self._cursor.fetchone())

  def fetchall(self) -> list[dict[str, Any]]:
    """Return all remaining rows as a list of normalised dicts."""
    return [
      {k: str(v) if isinstance(v, (datetime, date)) else v for k, v in row.items()}
      for row in self._cursor.fetchall()
    ]


# ---------------------------------------------------------------------------
# Connection wrapper
# ---------------------------------------------------------------------------

class _DBConnection:
  """Context-manager wrapper providing a backend-agnostic DB interface.

  Usage (same as before):
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT ...", (val,)).fetchone()
        new_id = conn.insert("INSERT INTO t (col) VALUES (?)", (val,))

  On clean exit the transaction is committed; on exception it is rolled back.
  The underlying connection is always closed on exit.
  """

  def __init__(self, conn: Any, is_postgres: bool) -> None:
    self._conn = conn
    self._is_postgres = is_postgres

  # -- Query helpers --------------------------------------------------------

  def execute(self, sql: str, params: tuple[Any, ...] = ()) -> Any:
    """Execute a SELECT / UPDATE / DELETE and return a cursor-like object.

    For Postgres, translates ``?`` placeholders to ``%s`` and returns a
    _CursorResult.  For SQLite, delegates directly to the connection and
    returns the native sqlite3.Cursor.
    """
    if self._is_postgres:
      cur = self._conn.cursor()
      cur.execute(sql.replace("?", "%s"), params)
      return _CursorResult(cur)
    return self._conn.execute(sql, params)

  def insert(self, sql: str, params: tuple[Any, ...] = ()) -> int:
    """Execute an INSERT and return the new row's integer primary key.

    The ``sql`` argument must NOT already contain ``RETURNING``.

    For Postgres, ``RETURNING id`` is appended automatically.
    For SQLite, ``cursor.lastrowid`` is used.
    """
    if self._is_postgres:
      cur = self._conn.cursor()
      cur.execute(sql.replace("?", "%s") + " RETURNING id", params)
      row = cur.fetchone()
      return int(row["id"])
    cursor = self._conn.execute(sql, params)
    return int(cursor.lastrowid)  # type: ignore[arg-type]

  # -- Context manager ------------------------------------------------------

  def __enter__(self) -> "_DBConnection":
    return self

  def __exit__(
    self,
    exc_type: type[BaseException] | None,
    exc_val: BaseException | None,
    exc_tb: Any,
  ) -> None:
    try:
      if exc_type is None:
        self._conn.commit()
      else:
        self._conn.rollback()
    finally:
      self._conn.close()


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def get_connection(db_path: Path | None = None) -> _DBConnection:
  """Open a database connection and return a _DBConnection context manager.

  When settings.database_url is set, connects to Postgres (ignores db_path).
  Otherwise opens a SQLite file at db_path or settings.db_path.

  Args:
    db_path: SQLite file path override (used in tests). Ignored for Postgres.

  Returns:
    A _DBConnection ready to be used as a context manager.
  """
  if _is_postgres():
    conn = psycopg2.connect(
      settings.database_url,
      cursor_factory=psycopg2.extras.RealDictCursor,
    )
    return _DBConnection(conn, is_postgres=True)

  path = db_path or settings.db_path
  conn = sqlite3.connect(str(path))
  conn.row_factory = sqlite3.Row
  conn.execute("PRAGMA foreign_keys = ON")
  return _DBConnection(conn, is_postgres=False)


# ---------------------------------------------------------------------------
# Schema DDL (two variants)
# ---------------------------------------------------------------------------

_SQLITE_DDL = [
  """
  CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL UNIQUE,
    password_hash TEXT    NOT NULL,
    role          TEXT    NOT NULL CHECK(role IN ('admin', 'reader')),
    created_at    TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
  )
  """,
  """
  CREATE TABLE IF NOT EXISTS sessions (
    token       TEXT    PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at  TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at  TEXT    NOT NULL
  )
  """,
  """
  CREATE TABLE IF NOT EXISTS api_keys (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    key_hash    TEXT    NOT NULL UNIQUE,
    name        TEXT    NOT NULL,
    role        TEXT    NOT NULL DEFAULT 'reader' CHECK(role IN ('reader')),
    created_at  TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    revoked_at  TEXT
  )
  """,
]

_POSTGRES_DDL = [
  """
  CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    username      TEXT   NOT NULL UNIQUE,
    password_hash TEXT   NOT NULL,
    role          TEXT   NOT NULL CHECK(role IN ('admin', 'reader')),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
  )
  """,
  """
  CREATE TABLE IF NOT EXISTS sessions (
    token       TEXT    PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at  TIMESTAMPTZ NOT NULL
  )
  """,
  """
  CREATE TABLE IF NOT EXISTS api_keys (
    id          SERIAL  PRIMARY KEY,
    key_hash    TEXT    NOT NULL UNIQUE,
    name        TEXT    NOT NULL,
    role        TEXT    NOT NULL DEFAULT 'reader' CHECK(role IN ('reader')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    revoked_at  TIMESTAMPTZ
  )
  """,
]


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

def init_db(db_path: Path | None = None) -> None:
  """Create the auth tables if they do not already exist (idempotent).

  Tables created:
    - users: id, username (unique), password_hash, role, created_at
    - sessions: token (PK), user_id (FK → users.id CASCADE), created_at, expires_at
    - api_keys: id, key_hash (unique), name, role, created_at, revoked_at

  For SQLite, the parent directory is created automatically.
  For Postgres, the database must already exist; only tables are created here.

  Args:
    db_path: SQLite file path override. Ignored when Postgres is active.
  """
  if _is_postgres():
    ddl_statements = _POSTGRES_DDL
  else:
    path = db_path or settings.db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    ddl_statements = _SQLITE_DDL

  with get_connection(db_path) as conn:
    for stmt in ddl_statements:
      conn.execute(stmt)
