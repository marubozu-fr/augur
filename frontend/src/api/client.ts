/**
 * Thin fetch wrapper that:
 * - Targets relative paths (Vite proxies /admin, /api, /auth to :8000).
 * - Unwraps the backend { data, error } response envelope.
 * - Throws an ApiError (with HTTP status) when the response is not ok
 *   or when the envelope carries a non-null error field.
 * - Dispatches a global `auth:unauthorized` window event when a 401 is
 *   received for any path other than /auth/login, so the AuthContext can
 *   clear the current user and the router can redirect to /login.
 */

interface ApiEnvelope<T> {
  data: T | null
  error: string | null
}

export class ApiError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

export const AUTH_UNAUTHORIZED_EVENT = 'auth:unauthorized'

export async function apiFetch<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const { headers: initHeaders, ...restInit } = init ?? {}
  const response = await fetch(path, {
    headers: {
      'Content-Type': 'application/json',
      ...(initHeaders as Record<string, string>),
    },
    ...restInit,
  })

  if (!response.ok) {
    // Try to extract a backend error message from the envelope.
    let message = `HTTP ${response.status} ${response.statusText}`
    try {
      const envelope = (await response.json()) as ApiEnvelope<unknown>
      if (envelope.error) message = envelope.error
    } catch {
      // Ignore parse errors — use the HTTP status message.
    }
    // A 401 on any non-login endpoint means the session is missing or expired.
    // The login endpoint also returns 401 on bad credentials, but that case
    // must be handled inline by the login form, not as a global session loss.
    if (response.status === 401 && path !== '/auth/login') {
      window.dispatchEvent(new CustomEvent(AUTH_UNAUTHORIZED_EVENT))
    }
    throw new ApiError(message, response.status)
  }

  const envelope = (await response.json()) as ApiEnvelope<T>

  if (envelope.error !== null && envelope.error !== undefined) {
    throw new ApiError(envelope.error, response.status)
  }

  // data is guaranteed non-null when error is null per backend contract.
  return envelope.data as T
}
