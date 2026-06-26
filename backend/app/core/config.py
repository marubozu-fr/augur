"""Application settings loaded from environment variables."""

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root is three levels up from this file: backend/app/core/config.py
_REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
  """Environment-overridable application settings."""

  model_config = SettingsConfigDict(
    env_prefix="AUGUR_",
    env_file=".env",
    env_file_encoding="utf-8",
    case_sensitive=False,
  )

  # Paths
  results_dir: Path = _REPO_ROOT / "results"
  data_dir: Path = _REPO_ROOT / "data"
  db_path: Path = _REPO_ROOT / "backend" / "db" / "augur.db"
  frontend_dist_dir: Path = _REPO_ROOT / "frontend" / "dist"

  # Database — bare DATABASE_URL (Railway) or AUGUR_DATABASE_URL both work.
  # Empty string means use SQLite (local dev / test suite).
  database_url: str = Field(
    default="",
    validation_alias=AliasChoices("DATABASE_URL", "AUGUR_DATABASE_URL"),
  )

  # CORS
  cors_origins: list[str] = ["http://localhost:5173"]

  # Auth — seed admin credentials (empty = skip seeding)
  admin_username: str = ""
  admin_password: str = ""

  # Default API key seeding (empty = skip seeding)
  default_api_key: str = ""
  default_api_key_name: str = "default"

  # Session cookie
  session_cookie_name: str = "augur_session"
  session_ttl_hours: int = 24
  session_cookie_secure: bool = False  # Set True in production (HTTPS only)


settings = Settings()
