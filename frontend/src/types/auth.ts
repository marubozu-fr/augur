/**
 * Public user shape returned by GET /auth/me and POST /auth/login.
 * Mirrors backend `UserOut` (snake_case is preserved verbatim).
 */

export type UserRole = 'admin' | 'reader'

export interface User {
  id: number
  username: string
  role: UserRole
}
