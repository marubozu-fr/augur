import { apiFetch } from './client'
import type { User } from '../types/auth'

/**
 * POST /auth/login
 * Returns the authenticated user. Sets the session cookie as a side effect.
 * Throws ApiError(401) on invalid credentials.
 */
export function login(username: string, password: string): Promise<User> {
  return apiFetch<User>('/auth/login', {
    method: 'POST',
    body: JSON.stringify({ username, password }),
  })
}

/**
 * POST /auth/logout
 * Clears the server-side session and the session cookie.
 */
export async function logout(): Promise<void> {
  await apiFetch<null>('/auth/logout', { method: 'POST' })
}

/**
 * GET /auth/me
 * Returns the currently authenticated user.
 * Throws ApiError(401) when there is no valid session.
 */
export function getCurrentUser(): Promise<User> {
  return apiFetch<User>('/auth/me')
}
