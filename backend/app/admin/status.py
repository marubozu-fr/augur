"""Admin endpoint exposing system, data, and stats-loader state.

GET /admin/status — any authenticated user (admin or reader). The reload action
that mutates the stats cache lives on POST /admin/stats/reload and remains
admin-only; readers can observe state, only admins can change it.
"""

from fastapi import APIRouter, Depends, Request

from backend.app.core.config import settings
from backend.app.core.dependencies import get_current_user, get_stats_loader
from backend.app.core.models import ApiResponse
from backend.app.core.stats_loader import StatsLoader
from backend.app.services.status import SystemStatus, build_system_status

router = APIRouter(prefix="/admin")


@router.get("/status", dependencies=[Depends(get_current_user)])
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
