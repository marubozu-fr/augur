import { useMemo } from 'react'
import {
  AppShell,
  Burger,
  ActionIcon,
  Select,
  Skeleton,
  Alert,
  Tooltip,
} from '@mantine/core'
import { useDisclosure } from '@mantine/hooks'
import { notifications } from '@mantine/notifications'
import {
  IconRefresh,
  IconLayoutDashboard,
  IconChartBar,
  IconAlertCircle,
  IconKey,
  IconLogout,
  IconActivity,
} from '@tabler/icons-react'
import {
  NavLink,
  Navigate,
  Outlet,
  useLocation,
  useNavigate,
  useParams,
} from 'react-router-dom'
import { useStatFamilies } from '../hooks/useStatFamilies'
import type { ShellContext } from '../hooks/useShellContext'
import { useAuth } from '../hooks/useAuth'
import { getAllInstruments, pickDefaultInstrument } from '../utils/statHelpers'
import styles from './ShellPage.module.css'

export function ShellPage() {
  const [opened, { toggle }] = useDisclosure()
  const statFamilies = useStatFamilies()
  const { families, loading, error, reload } = statFamilies
  const { user, logout } = useAuth()
  const location = useLocation()
  const navigate = useNavigate()

  // Instrument comes from the URL when a :instrument/* route is matched.
  // In a layout route, useParams sees params from descendant matches too.
  const { instrument: urlInstrument } = useParams<{ instrument?: string }>()

  const instruments = useMemo(() => getAllInstruments(families), [families])
  const defaultInstrument = useMemo(
    () => pickDefaultInstrument(instruments),
    [instruments],
  )

  // Derive the effective instrument: URL param when valid, else the default.
  const selectedInstrument =
    urlInstrument && instruments.includes(urlInstrument)
      ? urlInstrument
      : defaultInstrument

  // Invalid-instrument redirect: after data loads, if the URL carries an
  // instrument segment that is not in the list, swap it for the default and
  // preserve the rest of the path + search.
  if (!loading && !error && instruments.length > 0 && urlInstrument && !instruments.includes(urlInstrument)) {
    const rest = location.pathname.slice(('/' + urlInstrument).length)
    return (
      <Navigate
        replace
        to={`/${defaultInstrument}${rest}${location.search}`}
      />
    )
  }

  function handleInstrumentChange(next: string | null) {
    if (next === null) return
    if (urlInstrument) {
      // On an instrument-scoped page: swap the instrument prefix, keep the rest.
      const rest = location.pathname.slice(('/' + urlInstrument).length)
      navigate(`/${next}${rest}${location.search}`)
    } else {
      // On an instrument-agnostic page (system-status, api-keys): go to stats.
      navigate(`/${next}/stats`)
    }
  }

  async function handleLogout() {
    try {
      await logout()
      // No navigate() here: AuthProvider clears the user, RequireAuth handles
      // the redirect to /login. Keeping routing the guard's responsibility.
    } catch (err) {
      notifications.show({
        title: 'Logout failed',
        message: err instanceof Error ? err.message : 'Unknown error',
        color: 'red',
      })
    }
  }

  // Dashboard nav link target — instrument-scoped when we have one.
  const dashboardTo = selectedInstrument ? `/${selectedInstrument}/stats` : '/'

  const shellContext: ShellContext = {
    ...statFamilies,
    instruments,
    selectedInstrument,
  }

  return (
    <AppShell
      header={{ height: 56 }}
      navbar={{ width: 264, breakpoint: 'sm', collapsed: { mobile: !opened } }}
      padding={0}
    >
      {/* ===== Header ===== */}
      <AppShell.Header>
        <div className={styles.header}>
          <Burger opened={opened} onClick={toggle} hiddenFrom="sm" size="sm" />
          <div className={styles.brand}>
            <div className={styles.brandMark}>A</div>
            <span className={styles.brandText}>Augur</span>
          </div>
          <div className={styles.headerSpacer} />
          {instruments.length > 0 && (
            <div className={styles.instrumentSelector}>
              <span className={styles.instrumentLabel}>Instrument</span>
              <Select
                size="xs"
                data={instruments}
                value={selectedInstrument || null}
                onChange={handleInstrumentChange}
                aria-label="Select instrument"
                disabled={instruments.length === 1}
                searchable={false}
                clearable={false}
                className={styles.instrumentSelect}
              />
            </div>
          )}
          <div className={styles.headerStatus}>
            <div
              className={`${styles.statusDot} ${
                loading
                  ? styles.statusDotLoading
                  : error
                    ? styles.statusDotError
                    : ''
              }`}
            />
            <span>
              {loading
                ? 'Loading stats…'
                : error
                  ? 'Failed to load stats'
                  : 'Stats loaded'}
            </span>
          </div>
          <ActionIcon
            variant="subtle"
            color="gray"
            size="md"
            aria-label="Reload stat families"
            loading={loading}
            onClick={() => void reload()}
            className={styles.headerAction}
          >
            <IconRefresh size={16} />
          </ActionIcon>
          {user !== null && (
            <div className={styles.user}>
              <span className={styles.userName}>{user.username}</span>
              <span
                className={`${styles.userRole} ${
                  user.role === 'admin'
                    ? styles.userRoleAdmin
                    : styles.userRoleReader
                }`}
              >
                {user.role}
              </span>
            </div>
          )}
          <Tooltip label="Sign out" withArrow>
            <ActionIcon
              variant="subtle"
              color="gray"
              size="md"
              aria-label="Sign out"
              onClick={() => void handleLogout()}
              className={styles.headerAction}
            >
              <IconLogout size={16} />
            </ActionIcon>
          </Tooltip>
        </div>
      </AppShell.Header>

      {/* ===== Sidebar ===== */}
      <AppShell.Navbar>
        <div className={styles.navbar}>
          {/* Dashboard link */}
          <p className={styles.sectionTitle}>Dashboard</p>
          <NavLink
            to={dashboardTo}
            end
            className={({ isActive }) =>
              `${styles.navItem} ${isActive ? styles.navItemActive : ''}`
            }
          >
            <IconLayoutDashboard size={16} />
            <span className={styles.navLabel}>All stat families</span>
          </NavLink>

          {/* Stat families list */}
          <p className={styles.sectionTitle}>Stat families</p>

          {loading && (
            <div className={styles.skeletonList}>
              {[1, 2, 3, 4].map((n) => (
                <Skeleton key={n} height={32} radius="md" />
              ))}
            </div>
          )}

          {!loading && error && (
            <div className={styles.errorBox}>
              <Alert
                icon={<IconAlertCircle size={14} />}
                color="red"
                variant="light"
                p="xs"
              >
                <span className={styles.errorMessage}>{error}</span>
                <button
                  type="button"
                  className={styles.retryButton}
                  onClick={() => void reload()}
                >
                  Retry
                </button>
              </Alert>
            </div>
          )}

          {!loading && !error && families.length === 0 && (
            <p className={styles.emptyText}>No stat families found.</p>
          )}

          {!loading &&
            !error &&
            families.map((family) => (
              <NavLink
                key={family.family}
                to={`/${selectedInstrument}/stats/${family.family}`}
                className={({ isActive }) =>
                  `${styles.navItem} ${isActive ? styles.navItemActive : ''}`
                }
              >
                <IconChartBar size={16} className={styles.navIcon} />
                <span className={styles.navLabel}>{family.title.en}</span>
              </NavLink>
            ))}

          {/* System section — visible to all authenticated users */}
          <p className={styles.sectionTitle}>System</p>
          <NavLink
            to="/system-status"
            className={({ isActive }) =>
              `${styles.navItem} ${isActive ? styles.navItemActive : ''}`
            }
          >
            <IconActivity size={16} className={styles.navIcon} />
            <span className={styles.navLabel}>System status</span>
          </NavLink>

          {/* Admin section — admin-only entries */}
          {user?.role === 'admin' && (
            <>
              <p className={styles.sectionTitle}>Admin</p>
              <NavLink
                to="/api-keys"
                className={({ isActive }) =>
                  `${styles.navItem} ${isActive ? styles.navItemActive : ''}`
                }
              >
                <IconKey size={16} className={styles.navIcon} />
                <span className={styles.navLabel}>API keys</span>
              </NavLink>
            </>
          )}
        </div>
      </AppShell.Navbar>

      {/* ===== Main content (router outlet) ===== */}
      <AppShell.Main>
        <Outlet context={shellContext satisfies ShellContext} />
      </AppShell.Main>
    </AppShell>
  )
}
