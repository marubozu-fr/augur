"""Public REST endpoints for browsing stat results.

Note: these endpoints will be protected by X-API-Key auth once issue #113 lands.

GET /api/v1/stats                           — list all stat families (metadata only)
GET /api/v1/stats/{family}                  — full result for one stat family
GET /api/v1/stats/{family}/{instrument}     — result narrowed to one instrument
GET /api/v1/stats/{family}/{instrument}/{timeframe} — single timeframe result

All endpoints accept an optional ?slice=<dimension> query parameter that narrows
every returned TimeframeResult.slices to the requested dimension only.
"""

from fastapi import APIRouter, Depends, Query, Response

from backend.app.core.dependencies import get_stats_loader
from backend.app.core.models import ApiResponse
from backend.app.core.stats_loader import StatFamilyMeta, StatsLoader
from backend.app.services.stats_api import (
  apply_slice,
  apply_slice_tf,
  is_declared_slice,
  narrow_to_instrument,
  narrow_to_timeframe,
)
from stats.base import StatRunResult, TimeframeResult

router = APIRouter(prefix="/api/v1")


def _available_slices(result: StatRunResult) -> list[str]:
  """Return the sorted list of slice dimension names declared by this family."""
  return sorted(result.labels.dimensions.keys())


@router.get("/stats", summary="List stat families")
def list_stats(
  loader: StatsLoader = Depends(get_stats_loader),
) -> ApiResponse[list[StatFamilyMeta]]:
  """Return lightweight metadata for every loaded stat family.

  No results arrays are included — only the family name, title, definition,
  and the list of available instrument/timeframe combinations.
  """
  return ApiResponse(data=loader.list_families())


@router.get("/stats/{family}", summary="Get stat family result")
def get_stat(
  family: str,
  response: Response,
  slice: str | None = Query(default=None, description="Narrow results to a single slice dimension"),
  loader: StatsLoader = Depends(get_stats_loader),
) -> ApiResponse[StatRunResult]:
  """Return the full result payload for a single stat family.

  Includes all instruments, all timeframes, and all slice breakdowns unless
  the optional ?slice= query parameter is provided, in which case every
  TimeframeResult.slices dict is narrowed to that dimension only.

  Returns 404 if the family is not found.
  Returns 400 if the requested slice dimension is not declared by this family.
  """
  result = loader.get(family)
  if result is None:
    response.status_code = 404
    return ApiResponse(error=f"Stat family '{family}' not found")

  if slice is not None:
    if not is_declared_slice(result, slice):
      response.status_code = 400
      available = _available_slices(result)
      return ApiResponse(
        error=f"Unknown slice '{slice}' for family '{family}'. Available: {available}"
      )
    result = apply_slice(result, slice)

  return ApiResponse(data=result)


@router.get("/stats/{family}/{instrument}", summary="Get stat family result for one instrument")
def get_stat_instrument(
  family: str,
  instrument: str,
  response: Response,
  slice: str | None = Query(default=None, description="Narrow results to a single slice dimension"),
  loader: StatsLoader = Depends(get_stats_loader),
) -> ApiResponse[StatRunResult]:
  """Return the result for a single stat family narrowed to one instrument.

  The returned StatRunResult.instruments dict contains only the requested
  instrument key. Slice filtering applies as in GET /api/v1/stats/{family}.

  Returns 404 if the family or instrument is not found.
  Returns 400 if the requested slice dimension is not declared by this family.
  """
  full = loader.get(family)
  if full is None:
    response.status_code = 404
    return ApiResponse(error=f"Stat family '{family}' not found")

  if slice is not None:
    if not is_declared_slice(full, slice):
      response.status_code = 400
      available = _available_slices(full)
      return ApiResponse(
        error=f"Unknown slice '{slice}' for family '{family}'. Available: {available}"
      )

  result = narrow_to_instrument(full, instrument)
  if result is None:
    response.status_code = 404
    return ApiResponse(
      error=f"Instrument '{instrument}' not found in stat family '{family}'"
    )

  if slice is not None:
    result = apply_slice(result, slice)

  return ApiResponse(data=result)


@router.get(
  "/stats/{family}/{instrument}/{timeframe}",
  summary="Get a single timeframe result",
)
def get_stat_timeframe(
  family: str,
  instrument: str,
  timeframe: str,
  response: Response,
  slice: str | None = Query(default=None, description="Narrow results to a single slice dimension"),
  loader: StatsLoader = Depends(get_stats_loader),
) -> ApiResponse[TimeframeResult]:
  """Return the TimeframeResult for a specific family/instrument/timeframe combination.

  Slice filtering applies: when ?slice= is provided, the returned
  TimeframeResult.slices dict contains only the requested dimension.

  Returns 404 if the family, instrument, or timeframe is not found
  (the error message identifies which level was missing).
  Returns 400 if the requested slice dimension is not declared by this family.
  """
  full = loader.get(family)
  if full is None:
    response.status_code = 404
    return ApiResponse(error=f"Stat family '{family}' not found")

  if slice is not None:
    if not is_declared_slice(full, slice):
      response.status_code = 400
      available = _available_slices(full)
      return ApiResponse(
        error=f"Unknown slice '{slice}' for family '{family}'. Available: {available}"
      )

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

  if slice is not None:
    tf_result = apply_slice_tf(tf_result, slice)

  return ApiResponse(data=tf_result)
