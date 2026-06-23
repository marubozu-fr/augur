import { useMemo, useState } from 'react'
import {
  AppShell,
  Burger,
  ActionIcon,
  SegmentedControl,
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
import { NavLink, Outlet } from 'react-router-dom'
import { useStatFamilies } from '../hooks/useStatFamilies'
import type { ShellContext } from '../hooks/useShellContext'
import { useAuth } from '../hooks/useAuth'
import { getAllInstruments } from '../utils/statHelpers'
import styles from './ShellPage.module.css'

export function ShellPage() {
  const [opened, { toggle }] = useDisclosure()
  const statFamilies = useStatFamilies()
  const { families, loading, error, reload } = statFamilies
  const { user, logout } = useAuth()

  // Global instrument selection — derived dynamically from loaded families,
  // never hardcoded. State lives here so it survives child-route navigation
  // (ShellPage stays mounted for the whole session).
  const instruments = useMemo(() => getAllInstruments(families), [families])
  const [selectedInstrument, setSelectedInstrument] = useState<string>('')
  // Fall back to the first instrument when nothing is chosen yet or the prior
  // choice is no longer present (e.g. after a reload), so the context value is
  // always a valid instrument.
  const effectiveInstrument =
    selectedInstrument && instruments.includes(selectedInstrument)
      ? selectedInstrument
      : (instruments[0] ?? '')

  const shellContext: ShellContext = {
    ...statFamilies,
    instruments,
    selectedInstrument: effectiveInstrument,
    setSelectedInstrument,
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
              <SegmentedControl
                size="xs"
                value={effectiveInstrument}
                onChange={setSelectedInstrument}
                data={instruments}
                aria-label="Select instrument"
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
            to="/"
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
                to={`/stats/${family.family}`}
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
