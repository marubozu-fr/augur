"""Admin endpoints for browsing loaded stat families.

GET  /admin/stats                                    — any authenticated user (admin or reader)
GET  /admin/stats/{family}                           — any authenticated user (admin or reader)
GET  /admin/stats/{family}/{instrument}/{timeframe}  — any authenticated user (admin or reader)
POST /admin/stats/reload                             — admin role required
"""

from fastapi import APIRouter, Depends, Query, Response

from backend.app.core.dependencies import get_current_user, get_stats_loader, require_admin
from backend.app.core.models import ApiResponse
from backend.app.core.stats_loader import StatFamilyDetail, StatFamilyMeta, StatsLoader
from backend.app.services.stats_api import narrow_to_timeframe
from backend.app.services.stats_date_filter import (
  build_filtered_timeframe_result,
  has_undeclared_agg_metadata,
  validate_date_params,
)
from stats.base import TimeframeResult

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


@router.get(
  "/stats/{family}/{instrument}/{timeframe}",
  dependencies=[Depends(get_current_user)],
)
def get_stat_timeframe(
  family: str,
  instrument: str,
  timeframe: str,
  response: Response,
  start: str | None = Query(default=None, description="Filter samples from this ISO date (YYYY-MM-DD), inclusive"),
  end: str | None = Query(default=None, description="Filter samples up to this ISO date (YYYY-MM-DD), inclusive"),
  loader: StatsLoader = Depends(get_stats_loader),
) -> ApiResponse[TimeframeResult]:
  """Return one family/instrument/timeframe result, optionally date-filtered.

  This is the session-authenticated counterpart to
  GET /api/v1/stats/{family}/{instrument}/{timeframe}: it powers the backoffice
  period filter, which cannot send an X-API-Key. When ?start= and/or ?end= are
  provided the result is reaggregated from the family's per-day samples; absent
  both, the stored TimeframeResult is returned unchanged (slices included).

  Returns 404 if the family, instrument, or timeframe is not found.
  Returns 400 if start/end are not valid ISO dates, or start is after end.
  Returns 400 if date filtering is requested but this family carries no samples,
  or has magnitude rows missing their `agg` metadata.
  """
  full = loader.get(family)
  if full is None:
    response.status_code = 404
    return ApiResponse(error=f"Stat family '{family}' not found")

  if instrument not in full.instruments:
    response.status_code = 404
    return ApiResponse(
      error=f"Instrument '{instrument}' not found in stat family '{family}'"
    )

  tf_result = narrow_to_timeframe(full, instrument, timeframe)
  if tf_result is None:
    response.status_code = 404
    return ApiResponse(
      error=f"Timeframe '{timeframe}' not found for instrument '{instrument}' in stat family '{family}'"
    )

  if start is not None or end is not None:
    date_error = validate_date_params(start, end)
    if date_error is not None:
      response.status_code = 400
      return ApiResponse(error=date_error)

    if not tf_result.samples:
      response.status_code = 400
      return ApiResponse(
        error=(
          "Date filtering is not available for this stat family. "
          "Results must be regenerated with samples support."
        )
      )

    if has_undeclared_agg_metadata(tf_result.results, tf_result.samples):
      response.status_code = 400
      return ApiResponse(
        error="Date filtering requires result metadata (agg field). Results must be regenerated."
      )

    tf_result = build_filtered_timeframe_result(tf_result, start, end)

  return ApiResponse(data=tf_result)
