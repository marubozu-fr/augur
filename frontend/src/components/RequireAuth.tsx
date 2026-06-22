import { Center, Loader } from '@mantine/core'
import { Navigate, Outlet, useLocation } from 'react-router-dom'
import { useAuth } from '../hooks/useAuth'

/**
 * Route guard. Renders nested routes when authenticated, otherwise redirects
 * to /login and remembers the original location so the user can be sent back
 * after a successful sign-in.
 */
export function RequireAuth() {
  const { user, loading } = useAuth()
  const location = useLocation()

  if (loading) {
    return (
      <Center mih="100vh">
        <Loader size="sm" />
      </Center>
    )
  }

  if (user === null) {
    return <Navigate to="/login" replace state={{ from: location }} />
  }

  return <Outlet />
}
