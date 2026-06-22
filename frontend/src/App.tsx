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

export function App() {
  return (
    <MantineProvider theme={theme} forceColorScheme="dark">
      <Notifications />
      <BrowserRouter>
        <Routes>
          <Route element={<ShellPage />}>
            <Route index element={<DashboardPage />} />
            <Route path="stats/:family" element={<StatDetailPage />} />
            <Route path="api-keys" element={<ApiKeysPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </MantineProvider>
  )
}
