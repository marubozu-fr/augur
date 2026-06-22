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
import { LoginPage } from './pages/LoginPage'
import { RequireAuth } from './components/RequireAuth'
import { RequireAdmin } from './components/RequireAdmin'
import { AuthProvider } from './hooks/AuthProvider'

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
                <Route index element={<DashboardPage />} />
                <Route path="stats/:family" element={<StatDetailPage />} />
                <Route element={<RequireAdmin />}>
                  <Route path="api-keys" element={<ApiKeysPage />} />
                </Route>
                <Route path="*" element={<Navigate to="/" replace />} />
              </Route>
            </Route>
          </Routes>
        </AuthProvider>
      </BrowserRouter>
    </MantineProvider>
  )
}
