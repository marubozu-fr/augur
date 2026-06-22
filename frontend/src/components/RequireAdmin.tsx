import { Center, Loader } from '@mantine/core'
import { Navigate, Outlet } from 'react-router-dom'
import { useAuth } from '../hooks/useAuth'

export function RequireAdmin() {
  const { user, loading } = useAuth()

  if (loading) {
    return (
      <Center mih="100vh">
        <Loader size="sm" />
      </Center>
    )
  }

  if (user?.role !== 'admin') {
    return <Navigate to="/" replace />
  }

  return <Outlet />
}
