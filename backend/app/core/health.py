"""Health check endpoint."""

from fastapi import APIRouter

from backend.app.core.models import ApiResponse

router = APIRouter()


@router.get("/health")
def health_check() -> ApiResponse[dict[str, str]]:
  """Return application health status."""
  return ApiResponse(data={"status": "ok"})
