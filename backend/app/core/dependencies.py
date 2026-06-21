"""FastAPI dependency functions for injecting core services."""

from fastapi import Request

from backend.app.core.stats_loader import StatsLoader


def get_stats_loader(request: Request) -> StatsLoader:
  """Inject the StatsLoader instance stored on app.state."""
  return request.app.state.stats_loader
