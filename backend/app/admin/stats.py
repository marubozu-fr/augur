"""Admin endpoints for browsing loaded stat families.

Session cookie auth will be added with the auth layer (#110); for now these
endpoints are open so the backoffice shell can list stat families.
"""

from fastapi import APIRouter, Depends

from backend.app.core.dependencies import get_stats_loader
from backend.app.core.models import ApiResponse
from backend.app.core.stats_loader import StatFamilyMeta, StatsLoader

router = APIRouter(prefix="/admin")


@router.get("/stats")
def list_stats(
  loader: StatsLoader = Depends(get_stats_loader),
) -> ApiResponse[list[StatFamilyMeta]]:
  """Return lightweight metadata for every loaded stat family."""
  return ApiResponse(data=loader.list_families())


@router.post("/stats/reload")
def reload_stats(
  loader: StatsLoader = Depends(get_stats_loader),
) -> ApiResponse[list[StatFamilyMeta]]:
  """Re-scan results/ and return the refreshed stat family index."""
  loader.reload()
  return ApiResponse(data=loader.list_families())
