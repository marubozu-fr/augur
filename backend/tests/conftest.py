"""Shared pytest fixtures for the backend test suite."""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app


@pytest.fixture()
def client() -> TestClient:
  """Return a synchronous TestClient bound to a fresh FastAPI app."""
  return TestClient(create_app())
