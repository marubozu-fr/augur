import '@mantine/core/styles.css'
import '@mantine/notifications/styles.css'
import { MantineProvider } from '@mantine/core'
import { Notifications } from '@mantine/notifications'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { theme } from './theme/theme'
import { ShellPage } from './pages/ShellPage'
import { DashboardPage } from './pages/DashboardPage'
import { StatDetailPage } from './pages/StatDetailPage'
import { ApiKeysPage } from './pages/ApiKeysPage'
import { SystemStatusPage } from './pages/SystemStatusPage'
import { LoginPage } from './pages/LoginPage'
import { RequireAuth } from './components/RequireAuth'
import { RequireAdmin } from './components/RequireAdmin'
import { AuthProvider } from './hooks/AuthProvider'
import { useShellContext } from './hooks/useShellContext'
import { pickDefaultInstrument } from './utils/statHelpers'

/**
 * Redirects "/" (and catch-all) to "/<defaultInstrument>/stats".
 * Waits for shell data to load before redirecting so the instrument list is
 * known. Returns null while loading — ShellPage's layout (header/nav) remains
 * visible.
 */
function DefaultRedirect() {
  const { instruments, loading } = useShellContext()
  if (loading) return null
  const target = instruments.length > 0
    ? `/${pickDefaultInstrument(instruments)}/stats`
    : null
  if (target === null) return null
  return <Navigate to={target} replace />
}

export function App() {
  return (
    <MantineProvider theme={theme} forceColorScheme="dark">
      <Notifications />
      <BrowserRouter>
        <AuthProvider>
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route element={<RequireAuth />}>
              <Route element={<ShellPage />}>
                {/* "/" — wait for instruments then redirect */}
                <Route index element={<DefaultRedirect />} />
                {/* Instrument-agnostic pages — outrank :instrument in RR ranking */}
                <Route path="system-status" element={<SystemStatusPage />} />
                <Route element={<RequireAdmin />}>
                  <Route path="api-keys" element={<ApiKeysPage />} />
                </Route>
                {/* Instrument-scoped pages */}
                <Route path=":instrument/stats" element={<DashboardPage />} />
                <Route path=":instrument/stats/:family" element={<StatDetailPage />} />
                {/* Bare instrument segment — redirect to its stats page */}
                <Route path=":instrument" element={<Navigate to="stats" replace />} />
                {/* Catch-all */}
                <Route path="*" element={<DefaultRedirect />} />
              </Route>
            </Route>
          </Routes>
        </AuthProvider>
      </BrowserRouter>
    </MantineProvider>
  )
}
