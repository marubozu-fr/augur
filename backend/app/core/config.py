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

  # CORS
  cors_origins: list[str] = ["http://localhost:5173"]


settings = Settings()
