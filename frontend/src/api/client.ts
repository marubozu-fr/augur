/**
 * Thin fetch wrapper that:
 * - Targets relative paths (Vite proxies /admin, /api, /auth to :8000).
 * - Unwraps the backend { data, error } response envelope.
 * - Throws an Error with the error message when the HTTP status is not ok
 *   or when the envelope carries a non-null error field.
 */

interface ApiEnvelope<T> {
  data: T | null
  error: string | null
}

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
    throw new Error(message)
  }

  const envelope = (await response.json()) as ApiEnvelope<T>

  if (envelope.error !== null && envelope.error !== undefined) {
    throw new Error(envelope.error)
  }

  // data is guaranteed non-null when error is null per backend contract.
  return envelope.data as T
}
