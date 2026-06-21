"""Auth endpoints: login, logout, and current-user."""

from fastapi import APIRouter, Depends, Request, Response

from backend.app.auth.models import LoginRequest, UserOut
from backend.app.core.config import settings
from backend.app.core.dependencies import get_current_user
from backend.app.core.models import ApiResponse
from backend.app.services import auth as auth_service

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login")
def login(
  body: LoginRequest,
  response: Response,
) -> ApiResponse[UserOut]:
  """Validate credentials, create a server-side session, and set the session cookie."""
  user = auth_service.authenticate(body.username, body.password)
  if user is None:
    response.status_code = 401
    return ApiResponse(error="Invalid username or password")

  token = auth_service.create_session(user.id)
  response.set_cookie(
    key=settings.session_cookie_name,
    value=token,
    httponly=True,
    samesite="lax",
    secure=settings.session_cookie_secure,
    max_age=settings.session_ttl_hours * 3600,
  )
  return ApiResponse(data=user)


@router.post("/logout")
def logout(
  request: Request,
  response: Response,
  current_user: UserOut = Depends(get_current_user),
) -> ApiResponse[None]:
  """Delete the current session from the DB and clear the session cookie."""
  _ = current_user  # dependency enforces authentication
  token = request.cookies.get(settings.session_cookie_name, "")
  if token:
    auth_service.logout(token)
  response.delete_cookie(
    key=settings.session_cookie_name,
    httponly=True,
    samesite="lax",
  )
  return ApiResponse(data=None)


@router.get("/me")
def me(
  current_user: UserOut = Depends(get_current_user),
) -> ApiResponse[UserOut]:
  """Return the authenticated user's public profile."""
  return ApiResponse(data=current_user)
