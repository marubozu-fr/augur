"""Pydantic models for auth request bodies and response payloads."""

from pydantic import BaseModel


class LoginRequest(BaseModel):
  """Body for POST /auth/login."""

  username: str
  password: str


class UserOut(BaseModel):
  """Public user representation — never exposes password_hash."""

  id: int
  username: str
  role: str
