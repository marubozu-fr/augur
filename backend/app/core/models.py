"""Shared response models used across all endpoints."""

from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
  """Standard API response envelope.

  All endpoints return either:
    { "data": <payload>, "error": null }
    { "data": null, "error": "<message>" }
  """

  data: T | None = None
  error: str | None = None
