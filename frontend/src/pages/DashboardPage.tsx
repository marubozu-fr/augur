import { useMemo, useState } from 'react'
import { Link, useOutletContext } from 'react-router-dom'
import { Alert, Skeleton } from '@mantine/core'
import {
  IconSearch,
  IconArrowRight,
  IconAlertCircle,
  IconLayoutDashboard,
  IconRefresh,
  IconX,
} from '@tabler/icons-react'
import type { UseStatFamiliesResult } from '../hooks/useStatFamilies'
import type { StatFamilyMeta } from '../types/stats'
import {
  getInstruments,
  getTimeframeLabels,
  getTotalSamples,
  getDateRange,
} from '../utils/statHelpers'
import styles from './DashboardPage.module.css'

const numberFormat = new Intl.NumberFormat('en-US')

interface DashboardMetrics {
  families: number
  timeframes: number
  instruments: number
  coverage: string
}

/** Computes the summary metric strip values from the loaded families. */
function computeMetrics(families: StatFamilyMeta[]): DashboardMetrics {
  const instruments = new Set<string>()
  let timeframes = 0
  let minDate = ''
  let maxDate = ''
  for (const family of families) {
    timeframes += family.timeframes.length
    for (const t of family.timeframes) {
      instruments.add(t.instrument)
      const [start, end] = t.data_range
      if (start && (minDate === '' || start < minDate)) minDate = start
      if (end && (maxDate === '' || end > maxDate)) maxDate = end
    }
  }
  const coverage =
    minDate && maxDate ? `${minDate.slice(0, 4)}–${maxDate.slice(0, 4)}` : '—'
  return {
    families: families.length,
    timeframes,
    instruments: instruments.size,
    coverage,
  }
}

export function DashboardPage() {
  const { families, loading, error, reload } =
    useOutletContext<UseStatFamiliesResult>()
  const [query, setQuery] = useState('')

  const metrics = useMemo(() => computeMetrics(families), [families])

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (q === '') return families
    return families.filter((f) => f.title.en.toLowerCase().includes(q))
  }, [families, query])

  return (
    <div className={styles.page}>
      {/* ===== Page head ===== */}
      <div className={styles.pageHead}>
        <div>
          <h1 className={styles.pageTitle}>Stat Families</h1>
          <p className={styles.pageSubtitle}>
            Conditional market probabilities computed from historical OHLCV.
          </p>
        </div>
        {!loading && !error && families.length > 0 && (
          <div className={styles.metricStrip}>
            <div className={styles.metric}>
              <span className={styles.metricValue}>{metrics.families}</span>
              <span className={styles.metricLabel}>Families</span>
            </div>
            <div className={styles.metric}>
              <span className={styles.metricValue}>{metrics.timeframes}</span>
              <span className={styles.metricLabel}>Timeframes</span>
            </div>
            <div className={styles.metric}>
              <span className={styles.metricValue}>{metrics.instruments}</span>
              <span className={styles.metricLabel}>Instruments</span>
            </div>
            <div className={styles.metric}>
              <span className={styles.metricValue}>{metrics.coverage}</span>
              <span className={styles.metricLabel}>Coverage</span>
            </div>
          </div>
        )}
      </div>

      {/* ===== Error ===== */}
      {!loading && error && (
        <Alert
          icon={<IconAlertCircle size={16} />}
          color="red"
          variant="light"
          title="Failed to load stat families"
        >
          <p className={styles.alertMessage}>{error}</p>
          <button
            type="button"
            className={styles.alertRetry}
            onClick={() => void reload()}
          >
            <IconRefresh size={14} />
            <span>Retry</span>
          </button>
        </Alert>
      )}

      {/* ===== Loading skeleton ===== */}
      {loading && (
        <div className={styles.grid}>
          {[1, 2, 3, 4, 5, 6].map((n) => (
            <article key={n} className={styles.card}>
              <div className={styles.cardHead}>
                <Skeleton height={16} width="60%" radius="sm" />
                <Skeleton height={18} width={32} radius="sm" />
              </div>
              <div className={styles.cardTags}>
                <Skeleton height={16} width={44} radius="sm" />
                <Skeleton height={16} width={44} radius="sm" />
              </div>
              <div className={styles.cardStats}>
                <Skeleton height={34} radius="sm" />
                <Skeleton height={34} radius="sm" />
              </div>
              <Skeleton height={12} width="80%" radius="sm" />
            </article>
          ))}
        </div>
      )}

      {/* ===== Empty: no results at all ===== */}
      {!loading && !error && families.length === 0 && (
        <div className={styles.empty}>
          <span className={styles.emptyIcon}>
            <IconLayoutDashboard size={26} />
          </span>
          <h2 className={styles.emptyTitle}>No stat results found</h2>
          <p className={styles.emptyText}>
            The results directory is empty or no JSON files matched. Run a stat
            module to compute results, then reload to populate the dashboard.
          </p>
          <button
            type="button"
            className={styles.emptyAction}
            onClick={() => void reload()}
          >
            <IconRefresh size={14} />
            <span>Reload results</span>
          </button>
        </div>
      )}

      {/* ===== Populated ===== */}
      {!loading && !error && families.length > 0 && (
        <>
          <div className={styles.toolbar}>
            <div className={styles.search}>
              <IconSearch size={14} className={styles.searchIcon} />
              <input
                className={styles.searchInput}
                type="text"
                placeholder="Search stat families…"
                value={query}
                onChange={(e) => setQuery(e.currentTarget.value)}
                aria-label="Search stat families"
              />
            </div>
            <div className={styles.toolbarSpacer} />
            <span className={styles.resultCount}>
              {query.trim() === ''
                ? `${families.length} families`
                : `${filtered.length} of ${families.length} families`}
            </span>
          </div>

          {filtered.length === 0 ? (
            <div className={styles.empty}>
              <span className={styles.emptyIcon}>
                <IconSearch size={26} />
              </span>
              <h2 className={styles.emptyTitle}>
                No families match “{query.trim()}”
              </h2>
              <p className={styles.emptyText}>
                Try a different search term or clear the filter to see all{' '}
                {families.length} stat families.
              </p>
              <button
                type="button"
                className={styles.emptyAction}
                onClick={() => setQuery('')}
              >
                <IconX size={14} />
                <span>Clear search</span>
              </button>
            </div>
          ) : (
            <div className={styles.grid}>
              {filtered.map((family) => (
                <StatFamilyCard key={family.family} family={family} />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  )
}

interface StatFamilyCardProps {
  family: StatFamilyMeta
}

function StatFamilyCard({ family }: StatFamilyCardProps) {
  const instruments = getInstruments(family)
  const timeframeLabels = getTimeframeLabels(family)
  const totalSamples = getTotalSamples(family)
  const dateRange = getDateRange(family)

  return (
    <Link to={`/stats/${family.family}`} className={styles.card}>
      <div className={styles.cardHead}>
        <h2 className={styles.cardTitle}>{family.title.en}</h2>
        <div className={styles.cardInstruments}>
          {instruments.map((inst) => (
            <span key={inst} className={styles.badgeInstrument}>
              {inst}
            </span>
          ))}
        </div>
      </div>

      <div className={styles.cardTags}>
        {timeframeLabels.map((tf) => (
          <span key={tf} className={styles.badgeTf}>
            {tf}
          </span>
        ))}
      </div>

      <div className={styles.cardStats}>
        <div className={styles.cardStat}>
          <span className={styles.cardStatLabel}>Samples</span>
          <span className={styles.cardStatValue}>
            {numberFormat.format(totalSamples)}
          </span>
        </div>
        <div className={styles.cardStat}>
          <span className={styles.cardStatLabel}>Timeframes</span>
          <span className={styles.cardStatValue}>
            {family.timeframes.length}
          </span>
        </div>
      </div>

      <div className={styles.cardFoot}>
        <span className={styles.cardRange}>
          {dateRange ? `${dateRange[0]} → ${dateRange[1]}` : '—'}
        </span>
        <span className={styles.cardLink}>
          View <IconArrowRight size={13} />
        </span>
      </div>
    </Link>
  )
}
