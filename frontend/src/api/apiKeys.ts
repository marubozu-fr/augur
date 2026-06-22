import { apiFetch } from './client'
import type { ApiKey, ApiKeyCreated } from '../types/apiKeys'

/**
 * GET /admin/api-keys
 * Returns all API keys (never includes the plaintext key).
 */
export function listApiKeys(): Promise<ApiKey[]> {
  return apiFetch<ApiKey[]>('/admin/api-keys')
}

/**
 * POST /admin/api-keys
 * Creates a new API key. The plaintext key is returned once in the response.
 */
export function createApiKey(name: string): Promise<ApiKeyCreated> {
  return apiFetch<ApiKeyCreated>('/admin/api-keys', {
    method: 'POST',
    body: JSON.stringify({ name }),
  })
}

/**
 * DELETE /admin/api-keys/{id}
 * Revokes the API key with the given id. Returns void on success.
 */
export function revokeApiKey(id: number): Promise<void> {
  return apiFetch<null>(`/admin/api-keys/${id}`, { method: 'DELETE' }).then(
    () => undefined,
  )
}
