import { useCallback, useEffect, useState, type ReactNode } from 'react'
import { ApiError, AUTH_UNAUTHORIZED_EVENT } from '../api/client'
import * as authApi from '../api/auth'
import type { User } from '../types/auth'
import { AuthContext } from './authContext'

interface AuthProviderProps {
  children: ReactNode
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)

  // Initial session probe: ask the backend whether the cookie is valid.
  useEffect(() => {
    let cancelled = false
    authApi
      .getCurrentUser()
      .then((current) => {
        if (!cancelled) setUser(current)
      })
      .catch((err: unknown) => {
        // 401 simply means "no session" — that is the default state and
        // not an error worth surfacing. Any other failure is unexpected
        // but should not block the UI; we still treat the user as anonymous.
        if (!(err instanceof ApiError) || err.status !== 401) {
          console.warn('Failed to probe current user:', err)
        }
        if (!cancelled) setUser(null)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  // Global handler: any 401 from a non-login endpoint clears the user so
  // the route guard can redirect to /login (session expired flow).
  useEffect(() => {
    const handler = () => setUser(null)
    window.addEventListener(AUTH_UNAUTHORIZED_EVENT, handler)
    return () => {
      window.removeEventListener(AUTH_UNAUTHORIZED_EVENT, handler)
    }
  }, [])

  const login = useCallback(async (username: string, password: string) => {
    const authed = await authApi.login(username, password)
    setUser(authed)
  }, [])

  const logout = useCallback(async () => {
    try {
      await authApi.logout()
    } finally {
      // Even if the backend call fails, drop the local user — the cookie
      // is no longer trustworthy from the UI's point of view.
      setUser(null)
    }
  }, [])

  return (
    <AuthContext.Provider value={{ user, loading, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}
