"""FastAPI dependency functions for injecting core services and auth."""

from fastapi import Depends, HTTPException, Request

from backend.app.core.config import settings
from backend.app.core.stats_loader import StatsLoader


def get_stats_loader(request: Request) -> StatsLoader:
  """Inject the StatsLoader instance stored on app.state."""
  return request.app.state.stats_loader


def get_current_user(request: Request):  # type: ignore[return]
  """Resolve the session cookie to the authenticated user.

  Returns:
    A UserOut instance for the session owner.

  Raises:
    HTTPException(401): if the cookie is missing, invalid, or expired.
  """
  # Import here to avoid circular dependency at module load time.
  from backend.app.auth.models import UserOut  # noqa: PLC0415
  from backend.app.services import auth as auth_service  # noqa: PLC0415

  token = request.cookies.get(settings.session_cookie_name)
  if not token:
    raise HTTPException(status_code=401, detail="Not authenticated")

  user: UserOut | None = auth_service.resolve_session(token)
  if user is None:
    raise HTTPException(status_code=401, detail="Session invalid or expired")

  return user


def require_admin(
  current_user=Depends(get_current_user),
) -> None:
  """Dependency that restricts access to admin-role users only.

  Raises:
    HTTPException(403): if the authenticated user is not an admin.
  """
  if current_user.role != "admin":
    raise HTTPException(status_code=403, detail="Admin role required")
