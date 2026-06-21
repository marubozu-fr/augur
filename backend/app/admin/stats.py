"""Admin endpoints for browsing loaded stat families.

GET  /admin/stats           — any authenticated user (admin or reader)
GET  /admin/stats/{family}  — any authenticated user (admin or reader)
POST /admin/stats/reload    — admin role required
"""

from fastapi import APIRouter, Depends, Response

from backend.app.core.dependencies import get_current_user, get_stats_loader, require_admin
from backend.app.core.models import ApiResponse
from backend.app.core.stats_loader import StatFamilyDetail, StatFamilyMeta, StatsLoader

router = APIRouter(prefix="/admin")


@router.get("/stats", dependencies=[Depends(get_current_user)])
def list_stats(
  loader: StatsLoader = Depends(get_stats_loader),
) -> ApiResponse[list[StatFamilyMeta]]:
  """Return lightweight metadata for every loaded stat family."""
  return ApiResponse(data=loader.list_families())


@router.post("/stats/reload", dependencies=[Depends(require_admin)])
def reload_stats(
  loader: StatsLoader = Depends(get_stats_loader),
) -> ApiResponse[list[StatFamilyMeta]]:
  """Re-scan results/ and return the refreshed stat family index."""
  loader.reload()
  return ApiResponse(data=loader.list_families())


@router.get("/stats/{family}", dependencies=[Depends(get_current_user)])
def get_stat(
  family: str,
  response: Response,
  loader: StatsLoader = Depends(get_stats_loader),
) -> ApiResponse[StatFamilyDetail]:
  """Return the full result payload for a single stat family."""
  detail = loader.get_detail(family)
  if detail is None:
    response.status_code = 404
    return ApiResponse(error=f"Stat family '{family}' not found")
  return ApiResponse(data=detail)
