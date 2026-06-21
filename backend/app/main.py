"""Augur FastAPI application entry point."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import settings
from backend.app.core import health


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
  """Application lifespan handler.

  Runs startup logic before yielding, and teardown logic after.
  Future: initialise StatsLoader and DB connection pool here.
  """
  # Startup
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

  return app


app = create_app()
