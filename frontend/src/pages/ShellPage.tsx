import { AppShell, Burger } from '@mantine/core'
import { useDisclosure } from '@mantine/hooks'
import { IconChartBar, IconKey, IconLayoutDashboard } from '@tabler/icons-react'
import styles from './ShellPage.module.css'

const NAV_ITEMS = [
  { icon: IconLayoutDashboard, label: 'Dashboard', href: '/' },
  { icon: IconChartBar, label: 'Stats', href: '/stats' },
  { icon: IconKey, label: 'API Keys', href: '/api-keys' },
] as const

export function ShellPage() {
  const [opened, { toggle }] = useDisclosure()

  return (
    <AppShell
      header={{ height: 56 }}
      navbar={{ width: 264, breakpoint: 'sm', collapsed: { mobile: !opened } }}
      padding={0}
    >
      <AppShell.Header>
        <div className={styles.header}>
          <Burger opened={opened} onClick={toggle} hiddenFrom="sm" size="sm" />
          <div className={styles.brand}>
            <div className={styles.brandMark}>A</div>
            <span className={styles.brandText}>Augur</span>
          </div>
          <div className={styles.headerSpacer} />
          <div className={styles.headerStatus}>
            <div className={styles.statusDot} />
            <span>Stats loaded</span>
          </div>
        </div>
      </AppShell.Header>

      <AppShell.Navbar>
        <div className={styles.navbar}>
          <p className={styles.sectionTitle}>Navigation</p>
          {NAV_ITEMS.map(({ icon: Icon, label, href }) => (
            <a key={href} href={href} className={styles.navItem}>
              <Icon size={16} />
              <span>{label}</span>
            </a>
          ))}
        </div>
      </AppShell.Navbar>

      <AppShell.Main>
        <div className={styles.main}>
          <div className={styles.placeholder}>
            <p className={styles.placeholderTitle}>Augur Backoffice</p>
            <p className={styles.placeholderText}>
              Scaffold placeholder — dashboard and stat-detail pages will be
              implemented in subsequent issues.
            </p>
          </div>
        </div>
      </AppShell.Main>
    </AppShell>
  )
}
