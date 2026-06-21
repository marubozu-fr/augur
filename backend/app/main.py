"""Augur FastAPI application entry point."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.admin import stats as admin_stats
from backend.app.auth import routes as auth_routes
from backend.app.core import health
from backend.app.core.config import settings
from backend.app.core.db import init_db
from backend.app.core.stats_loader import StatsLoader
from backend.app.services.auth import seed_admin


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
  """Application lifespan handler.

  Runs startup logic before yielding, and teardown logic after.
  """
  # Startup
  init_db()
  seed_admin()

  loader = StatsLoader()
  loader.load_all()
  app.state.stats_loader = loader
  yield
  # Teardown


def create_app() -> FastAPI:
  """Construct and configure the FastAPI application."""
  app = FastAPI(
    title="Augur API",
    description="Trading probability terminal — backoffice API",
    version="0.1.0",
    lifespan=lifespan,
  )

  app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
  )

  # Routers
  app.include_router(health.router)
  app.include_router(auth_routes.router)
  app.include_router(admin_stats.router)

  return app


app = create_app()
