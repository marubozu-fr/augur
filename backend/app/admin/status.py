"""Admin endpoint exposing system, data, and stats-loader state.

GET /admin/status — admin role required
"""

from fastapi import APIRouter, Depends, Request

from backend.app.core.config import settings
from backend.app.core.dependencies import get_stats_loader, require_admin
from backend.app.core.models import ApiResponse
from backend.app.core.stats_loader import StatsLoader
from backend.app.services.status import SystemStatus, build_system_status

router = APIRouter(prefix="/admin")


@router.get("/status", dependencies=[Depends(require_admin)])
def get_status(
  request: Request,
  loader: StatsLoader = Depends(get_stats_loader),
) -> ApiResponse[SystemStatus]:
  """Return loaded stats files, data files, versions, and uptime."""
  payload = build_system_status(
    loader=loader,
    data_dir=settings.data_dir,
    started_at=request.app.state.started_at,
  )
  return ApiResponse(data=payload)
