import '@mantine/core/styles.css'
import '@mantine/notifications/styles.css'
import { MantineProvider } from '@mantine/core'
import { Notifications } from '@mantine/notifications'
import { theme } from './theme/theme'
import { ShellPage } from './pages/ShellPage'

export function App() {
  return (
    <MantineProvider theme={theme} forceColorScheme="dark">
      <Notifications />
      <ShellPage />
    </MantineProvider>
  )
}
