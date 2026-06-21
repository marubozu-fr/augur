"""Optional static file serving for the built React frontend.

In production, Vite builds the frontend into frontend/dist/. This module
mounts those assets onto the FastAPI app so one process serves everything.

In dev (or CI before a build), frontend/dist/ does not exist — the app
starts normally and this module does nothing.

API route prefixes (/api, /auth, /admin, /health) always take priority
because routers are registered before this module is called. The catch-all
GET route returns 404 for paths that start with those prefixes rather than
serving index.html, so unknown API paths stay as proper 404 API errors.
"""

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

# Route prefixes that belong to the API — the SPA catch-all must not shadow them.
# NOTE: this list must stay in sync with the router prefixes registered in
# backend/app/main.py. When a new API router prefix is added there (e.g. /ws),
# add it here too, otherwise the SPA fallback would shadow its unknown paths.
_API_PREFIXES = ("/api", "/auth", "/admin", "/health")


def mount_frontend(app: FastAPI, dist_dir: Path) -> None:
  """Mount the built frontend onto app if dist_dir exists.

  Registers two things:
  - StaticFiles mount at /assets for the Vite-generated asset bundle.
  - A catch-all GET /{full_path:path} route that returns index.html, enabling
    client-side routing. Requests under API prefixes return 404 instead.

  This function is a no-op when dist_dir does not exist, which keeps dev and
  CI environments unaffected.
  """
  if not dist_dir.is_dir():
    logger.debug("frontend dist dir not found (%s) — skipping static mount", dist_dir)
    return

  assets_dir = dist_dir / "assets"
  index_html = dist_dir / "index.html"

  if assets_dir.is_dir():
    app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")

  dist_root = dist_dir.resolve()

  @app.get("/{full_path:path}", include_in_schema=False)
  def spa_fallback(full_path: str, request: Request) -> Response:
    """Serve a root-level static file, else index.html (SPA client-side routing)."""
    path = request.url.path
    if any(path == prefix or path.startswith(prefix + "/") for prefix in _API_PREFIXES):
      return JSONResponse({"detail": "Not found"}, status_code=404)
    # Serve root-level static files (favicon, robots.txt, manifest, ...) that Vite
    # places at the dist root. The resolve() + prefix check block path traversal.
    if full_path:
      candidate = (dist_dir / full_path).resolve()
      if candidate.is_file() and str(candidate).startswith(str(dist_root)):
        return FileResponse(candidate)
    if index_html.is_file():
      return FileResponse(index_html)
    return Response(status_code=404)

  logger.info("Frontend static files mounted from %s", dist_dir)
