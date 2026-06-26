"""Auth service — password hashing, session lifecycle, admin seeding."""

import logging
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path

import bcrypt

from backend.app.auth.models import UserOut
from backend.app.core.config import settings
from backend.app.core.db import DBError
from backend.app.repositories import sessions as sessions_repo
from backend.app.repositories import users as users_repo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Password helpers
# ---------------------------------------------------------------------------

def hash_password(plaintext: str) -> str:
  """Return a bcrypt hash of the plaintext password."""
  return bcrypt.hashpw(plaintext.encode(), bcrypt.gensalt()).decode()


def verify_password(plaintext: str, hashed: str) -> bool:
  """Return True if plaintext matches the stored bcrypt hash."""
  return bcrypt.checkpw(plaintext.encode(), hashed.encode())


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def authenticate(
  username: str,
  password: str,
  db_path: Path | None = None,
) -> UserOut | None:
  """Verify credentials and return a UserOut if valid, else None."""
  row = users_repo.get_user_by_username(username, db_path)
  if row is None:
    return None
  if not verify_password(password, row["password_hash"]):
    return None
  return UserOut(id=row["id"], username=row["username"], role=row["role"])


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

def create_session(user_id: int, db_path: Path | None = None) -> str:
  """Generate a secure token, persist the session, and return the token."""
  token = secrets.token_urlsafe(32)
  expires_at = (
    datetime.now(tz=timezone.utc) + timedelta(hours=settings.session_ttl_hours)
  ).strftime("%Y-%m-%d %H:%M:%S")
  sessions_repo.create_session(token, user_id, expires_at, db_path)
  return token


def resolve_session(
  token: str,
  db_path: Path | None = None,
) -> UserOut | None:
  """Return the UserOut for a valid, non-expired session token, or None."""
  row = sessions_repo.get_session(token, db_path)
  if row is None:
    return None
  return UserOut(id=row["user_id"], username=row["username"], role=row["role"])


def logout(token: str, db_path: Path | None = None) -> None:
  """Delete the session identified by token."""
  sessions_repo.delete_session(token, db_path)


# ---------------------------------------------------------------------------
# Admin seeding
# ---------------------------------------------------------------------------

def seed_admin(db_path: Path | None = None) -> None:
  """Create the admin user from settings if credentials are set and the user
  does not already exist.  No-op if AUGUR_ADMIN_USERNAME / AUGUR_ADMIN_PASSWORD
  are empty strings.
  """
  if not settings.admin_username or not settings.admin_password:
    return

  try:
    existing = users_repo.get_user_by_username(settings.admin_username, db_path)
    if existing is not None:
      return
    users_repo.create_user(
      settings.admin_username,
      hash_password(settings.admin_password),
      "admin",
      db_path,
    )
  except DBError as exc:
    # Non-fatal: log and continue so a seeding failure does not crash startup,
    # but the operator can see why login is impossible.
    logger.warning("Failed to seed admin user: %s", exc)
