import { AppShell, Burger, ActionIcon, Skeleton, Alert } from '@mantine/core'
import { useDisclosure } from '@mantine/hooks'
import {
  IconRefresh,
  IconLayoutDashboard,
  IconChartBar,
  IconAlertCircle,
  IconKey,
} from '@tabler/icons-react'
import { NavLink, Outlet } from 'react-router-dom'
import {
  useStatFamilies,
  type UseStatFamiliesResult,
} from '../hooks/useStatFamilies'
import { getInstruments } from '../utils/statHelpers'
import styles from './ShellPage.module.css'

export function ShellPage() {
  const [opened, { toggle }] = useDisclosure()
  const statFamilies = useStatFamilies()
  const { families, loading, error, reload } = statFamilies

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
            className={styles.reloadButton}
          >
            <IconRefresh size={16} />
          </ActionIcon>
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
            families.map((family) => {
              const instruments = getInstruments(family)
              return (
                <NavLink
                  key={family.family}
                  to={`/stats/${family.family}`}
                  className={({ isActive }) =>
                    `${styles.navItem} ${isActive ? styles.navItemActive : ''}`
                  }
                >
                  <IconChartBar size={16} className={styles.navIcon} />
                  <span className={styles.navLabel}>{family.title.en}</span>
                  <div className={styles.badges}>
                    {instruments.map((inst) => (
                      <span key={inst} className={styles.badgeInstrument}>
                        {inst}
                      </span>
                    ))}
                  </div>
                </NavLink>
              )
            })}

          {/* Admin section */}
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
        </div>
      </AppShell.Navbar>

      {/* ===== Main content (router outlet) ===== */}
      <AppShell.Main>
        <Outlet context={statFamilies satisfies UseStatFamiliesResult} />
      </AppShell.Main>
    </AppShell>
  )
}
