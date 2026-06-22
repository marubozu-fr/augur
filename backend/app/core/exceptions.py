"""Global exception handlers that normalize error responses to ApiResponse envelope."""

from fastapi import Request
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse

from backend.app.core.models import ApiResponse


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
  """Wrap any HTTPException in the standard ApiResponse envelope.

  Ensures clients always receive {"data": null, "error": "..."} regardless of
  whether the error was raised from a dependency, a router, or middleware.
  """
  return JSONResponse(
    status_code=exc.status_code,
    content=ApiResponse[None](error=str(exc.detail)).model_dump(),
    headers=getattr(exc, "headers", None),
  )
