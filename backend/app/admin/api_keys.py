"""Admin endpoints for managing API keys.

POST   /admin/api-keys          — create a new key (plaintext shown once)
GET    /admin/api-keys          — list all keys (no hashes or plaintext)
DELETE /admin/api-keys/{key_id} — revoke a key by id
"""

from fastapi import APIRouter, Depends, Response

from backend.app.auth.models import ApiKeyCreateRequest, ApiKeyCreated, ApiKeyOut
from backend.app.core.dependencies import require_admin
from backend.app.core.models import ApiResponse
from backend.app.services import api_keys as api_keys_service

router = APIRouter(prefix="/admin")


@router.post("/api-keys", status_code=201, dependencies=[Depends(require_admin)])
def create_api_key(
  body: ApiKeyCreateRequest,
) -> ApiResponse[ApiKeyCreated]:
  """Create a new API key and return its plaintext value exactly once.

  The plaintext key is never stored — it cannot be retrieved after this
  response. The caller must copy it immediately.
  """
  _plaintext, created = api_keys_service.create_key(body.name)
  return ApiResponse(data=created)


@router.get("/api-keys", dependencies=[Depends(require_admin)])
def list_api_keys() -> ApiResponse[list[ApiKeyOut]]:
  """Return all API keys with their status (active or revoked).

  key_hash and plaintext are never included in the response.
  """
  return ApiResponse(data=api_keys_service.list_keys())


@router.delete("/api-keys/{key_id}", dependencies=[Depends(require_admin)])
def revoke_api_key(
  key_id: int,
  response: Response,
) -> ApiResponse[None]:
  """Revoke an API key by its numeric id.

  Returns 404 if the key does not exist or is already revoked.
  Returns 200 with data=null on success.
  """
  revoked = api_keys_service.revoke_key(key_id)
  if not revoked:
    response.status_code = 404
    return ApiResponse(error="API key not found or already revoked")
  return ApiResponse(data=None)
