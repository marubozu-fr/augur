import { createContext } from 'react'
import type { User } from '../types/auth'

export interface AuthContextValue {
  /** Current user, or null when unauthenticated. */
  user: User | null
  /** True while the initial /auth/me probe is in flight. */
  loading: boolean
  /** Authenticate with username + password. Throws on failure. */
  login: (username: string, password: string) => Promise<void>
  /** Clear the server-side session and the local user state. */
  logout: () => Promise<void>
}

export const AuthContext = createContext<AuthContextValue | null>(null)
