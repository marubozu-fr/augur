import { useState } from 'react'
import {
  Alert,
  Button,
  Center,
  Loader,
  PasswordInput,
  TextInput,
} from '@mantine/core'
import { IconAlertCircle, IconLogin } from '@tabler/icons-react'
import { Navigate, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../hooks/useAuth'
import { ApiError } from '../api/client'
import styles from './LoginPage.module.css'

interface LocationState {
  from?: { pathname: string }
}

export function LoginPage() {
  const { user, loading, login } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()

  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  // Wait for the /auth/me probe before rendering: an already-authenticated
  // user landing on /login would otherwise see the form flash before the
  // redirect below kicks in.
  if (loading) {
    return (
      <Center mih="100vh">
        <Loader size="sm" />
      </Center>
    )
  }

  // Already logged in — bounce to the requested page or the dashboard.
  if (user !== null) {
    const from = (location.state as LocationState | null)?.from?.pathname ?? '/'
    return <Navigate to={from} replace />
  }

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault()
    if (submitting) return
    setError(null)

    const trimmedUsername = username.trim()
    if (trimmedUsername.length === 0 || password.length === 0) {
      setError('Username and password are required')
      return
    }

    setSubmitting(true)
    try {
      await login(trimmedUsername, password)
      const from =
        (location.state as LocationState | null)?.from?.pathname ?? '/'
      navigate(from, { replace: true })
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        setError('Invalid username or password')
      } else {
        setError(err instanceof Error ? err.message : 'Failed to sign in')
      }
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className={styles.page}>
      <div className={styles.card}>
        <div className={styles.brand}>
          <div className={styles.brandMark}>A</div>
          <span className={styles.brandText}>Augur</span>
        </div>
        <h1 className={styles.title}>Sign in</h1>
        <p className={styles.subtitle}>
          Admin backoffice — credentials required.
        </p>

        {error !== null && (
          <Alert
            icon={<IconAlertCircle size={16} />}
            color="red"
            variant="light"
            mb="md"
          >
            {error}
          </Alert>
        )}

        <form onSubmit={(e) => void handleSubmit(e)} className={styles.form}>
          <TextInput
            label="Username"
            placeholder="admin"
            value={username}
            onChange={(e) => setUsername(e.currentTarget.value)}
            disabled={submitting}
            autoComplete="username"
            autoFocus
            required
          />
          <PasswordInput
            label="Password"
            placeholder="••••••••"
            value={password}
            onChange={(e) => setPassword(e.currentTarget.value)}
            disabled={submitting}
            autoComplete="current-password"
            required
          />
          <Button
            type="submit"
            loading={submitting}
            leftSection={<IconLogin size={14} />}
            fullWidth
            mt="sm"
          >
            Sign in
          </Button>
        </form>
      </div>
    </div>
  )
}
