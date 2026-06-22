"""Pydantic models for auth request bodies and response payloads."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class LoginRequest(BaseModel):
  """Body for POST /auth/login."""

  username: str
  password: str


class UserOut(BaseModel):
  """Public user representation — never exposes password_hash."""

  id: int
  username: str
  role: str


class ApiKeyCreateRequest(BaseModel):
  """Body for POST /admin/api-keys."""

  name: str = Field(min_length=1, max_length=100)

  @field_validator("name")
  @classmethod
  def _strip_and_require_non_empty(cls, value: str) -> str:
    stripped = value.strip()
    if not stripped:
      raise ValueError("name must not be blank")
    return stripped


class ApiKeyOut(BaseModel):
  """Public API key representation — never exposes key_hash or plaintext.

  status is 'active' when revoked_at is None, 'revoked' otherwise.
  """

  id: int
  name: str
  role: str
  created_at: str
  revoked_at: str | None
  status: Literal["active", "revoked"]


class ApiKeyCreated(BaseModel):
  """Response for POST /admin/api-keys — carries the plaintext key exactly once.

  The key field will never be retrievable again after this response.
  """

  id: int
  name: str
  role: str
  created_at: str
  key: str
