"""Application settings loaded from environment variables."""

from pathlib import Path

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
  db_path: Path = _REPO_ROOT / "backend" / "db" / "augur.db"
  frontend_dist_dir: Path = _REPO_ROOT / "frontend" / "dist"

  # CORS
  cors_origins: list[str] = ["http://localhost:5173"]

  # Auth — seed admin credentials (empty = skip seeding)
  admin_username: str = ""
  admin_password: str = ""

  # Session cookie
  session_cookie_name: str = "augur_session"
  session_ttl_hours: int = 24
  session_cookie_secure: bool = False  # Set True in production (HTTPS only)


settings = Settings()
