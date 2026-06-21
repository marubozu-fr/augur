"""Tests for the /health endpoint."""

from fastapi.testclient import TestClient


def test_health_returns_200(client: TestClient) -> None:
  """GET /health must return HTTP 200."""
  response = client.get("/health")
  assert response.status_code == 200


def test_health_envelope_shape(client: TestClient) -> None:
  """GET /health must return the standard { data, error } envelope."""
  response = client.get("/health")
  body = response.json()
  assert "data" in body
  assert "error" in body
  assert body["error"] is None


def test_health_data_payload(client: TestClient) -> None:
  """GET /health data payload must report status ok."""
  response = client.get("/health")
  body = response.json()
  assert body["data"] == {"status": "ok"}
