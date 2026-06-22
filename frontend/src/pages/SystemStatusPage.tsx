import { useState } from 'react'
import { Alert, Button, Skeleton } from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { IconAlertCircle, IconRefresh } from '@tabler/icons-react'
import { useAuth } from '../hooks/useAuth'
import { useSystemStatus } from '../hooks/useSystemStatus'
import type {
  DataFileInfo,
  LoadedStatFile,
} from '../types/systemStatus'
import styles from './SystemStatusPage.module.css'

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

const dateFormat = new Intl.DateTimeFormat('en-US', {
  year: 'numeric',
  month: 'short',
  day: 'numeric',
  hour: '2-digit',
  minute: '2-digit',
})

function fmtDateTime(iso: string): string {
  return dateFormat.format(new Date(iso))
}

function fmtBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`
}

function fmtRows(rows: number | null): string {
  if (rows === null) return '—'
  return rows.toLocaleString('en-US')
}

function fmtUptime(seconds: number): string {
  const total = Math.floor(seconds)
  const days = Math.floor(total / 86400)
  const hours = Math.floor((total % 86400) / 3600)
  const minutes = Math.floor((total % 3600) / 60)
  const secs = total % 60
  if (days > 0) return `${days}d ${hours}h ${minutes}m`
  if (hours > 0) return `${hours}h ${minutes}m ${secs}s`
  if (minutes > 0) return `${minutes}m ${secs}s`
  return `${secs}s`
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export function SystemStatusPage() {
  const { user } = useAuth()
  const { status, loading, error, refresh, reloadStats } = useSystemStatus()
  const [reloading, setReloading] = useState(false)

  const isAdmin = user?.role === 'admin'

  async function handleReload() {
    setReloading(true)
    try {
      await reloadStats()
      notifications.show({
        title: 'Stats reloaded',
        message: 'Stat results have been re-scanned from disk.',
        color: 'green',
      })
    } catch (err) {
      notifications.show({
        title: 'Failed to reload stats',
        message: err instanceof Error ? err.message : 'Unknown error',
        color: 'red',
      })
    } finally {
      setReloading(false)
    }
  }

  return (
    <div className={styles.page}>
      {/* ===== Page head ===== */}
      <div className={styles.pageHead}>
        <div>
          <h1 className={styles.pageTitle}>System status</h1>
          <p className={styles.pageSubtitle}>
            Loaded stat files, OHLCV data, runtime versions, and uptime.
          </p>
        </div>
        {isAdmin && (
          <div className={styles.pageActions}>
            <Button
              leftSection={<IconRefresh size={14} />}
              variant="light"
              loading={reloading}
              onClick={() => void handleReload()}
            >
              Reload stats
            </Button>
          </div>
        )}
      </div>

      {/* ===== Error ===== */}
      {!loading && error && (
        <Alert
          icon={<IconAlertCircle size={16} />}
          color="red"
          variant="light"
          title="Failed to load system status"
          mb="lg"
        >
          <p className={styles.alertMessage}>{error}</p>
          <button
            type="button"
            className={styles.alertRetry}
            onClick={() => void refresh()}
          >
            <IconRefresh size={14} />
            <span>Retry</span>
          </button>
        </Alert>
      )}

      {/* ===== Overview ===== */}
      <OverviewSection status={status} loading={loading} />

      {/* ===== Loaded stat files ===== */}
      <StatsFilesSection
        files={status?.stats_files ?? []}
        loading={loading}
      />

      {/* ===== Data files ===== */}
      <DataFilesSection
        files={status?.data_files ?? []}
        loading={loading}
      />
    </div>
  )
}

// ---------------------------------------------------------------------------
// Overview section
// ---------------------------------------------------------------------------

interface OverviewSectionProps {
  status: ReturnType<typeof useSystemStatus>['status']
  loading: boolean
}

function OverviewSection({ status, loading }: OverviewSectionProps) {
  return (
    <div className={styles.section}>
      <div className={styles.sectionHead}>
        <h2 className={styles.sectionTitle}>Runtime</h2>
      </div>
      {loading && status === null ? (
        <div className={styles.skeletonTable}>
          <Skeleton height={48} radius="md" />
        </div>
      ) : (
        <div className={styles.overviewGrid}>
          <div className={styles.overviewCell}>
            <span className={styles.overviewLabel}>Uptime</span>
            <span className={styles.overviewValue}>
              {status ? fmtUptime(status.uptime_seconds) : '—'}
            </span>
          </div>
          <div className={styles.overviewCell}>
            <span className={styles.overviewLabel}>Started at</span>
            <span className={styles.overviewValue}>
              {status ? fmtDateTime(status.started_at) : '—'}
            </span>
          </div>
          <div className={styles.overviewCell}>
            <span className={styles.overviewLabel}>Python</span>
            <span className={styles.overviewValue}>
              {status ? status.versions.python : '—'}
            </span>
          </div>
          <div className={styles.overviewCell}>
            <span className={styles.overviewLabel}>FastAPI</span>
            <span className={styles.overviewValue}>
              {status ? status.versions.fastapi : '—'}
            </span>
          </div>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Stats files section
// ---------------------------------------------------------------------------

interface StatsFilesSectionProps {
  files: LoadedStatFile[]
  loading: boolean
}

function StatsFilesSection({ files, loading }: StatsFilesSectionProps) {
  // Show skeletons only on the initial load (no data yet). During a reload,
  // keep the previous rows visible so the page doesn't flash blank.
  const showSkeleton = loading && files.length === 0
  return (
    <div className={styles.section}>
      <div className={styles.sectionHead}>
        <h2 className={styles.sectionTitle}>Loaded stat files</h2>
        {!showSkeleton && (
          <span className={styles.sectionCount}>
            {files.length} file{files.length !== 1 ? 's' : ''}
          </span>
        )}
      </div>
      {showSkeleton ? (
        <div className={styles.skeletonTable}>
          {[1, 2, 3].map((n) => (
            <Skeleton key={n} height={32} radius="md" />
          ))}
        </div>
      ) : (
        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Family</th>
                <th>File</th>
                <th>Size</th>
                <th>Last modified</th>
              </tr>
            </thead>
            <tbody>
              {files.length === 0 ? (
                <tr className={styles.emptyRow}>
                  <td colSpan={4}>No stat result files loaded.</td>
                </tr>
              ) : (
                files.map((f) => (
                  <tr key={f.family}>
                    <td className={styles.cellName}>{f.family}</td>
                    <td className={styles.cellMono}>{f.filename}</td>
                    <td className={styles.cellNumeric}>
                      {fmtBytes(f.size_bytes)}
                    </td>
                    <td className={styles.cellMono}>
                      {fmtDateTime(f.modified_at)}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Data files section
// ---------------------------------------------------------------------------

interface DataFilesSectionProps {
  files: DataFileInfo[]
  loading: boolean
}

function DataFilesSection({ files, loading }: DataFilesSectionProps) {
  // Same stale-data-friendly pattern as StatsFilesSection above.
  const showSkeleton = loading && files.length === 0
  return (
    <div className={styles.section}>
      <div className={styles.sectionHead}>
        <h2 className={styles.sectionTitle}>OHLCV data files</h2>
        {!showSkeleton && (
          <span className={styles.sectionCount}>
            {files.length} file{files.length !== 1 ? 's' : ''}
          </span>
        )}
      </div>
      {showSkeleton ? (
        <div className={styles.skeletonTable}>
          {[1, 2, 3].map((n) => (
            <Skeleton key={n} height={32} radius="md" />
          ))}
        </div>
      ) : (
        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>File</th>
                <th>Size</th>
                <th>Rows</th>
                <th>Date range</th>
              </tr>
            </thead>
            <tbody>
              {files.length === 0 ? (
                <tr className={styles.emptyRow}>
                  <td colSpan={4}>No Parquet data files found.</td>
                </tr>
              ) : (
                files.map((f) => (
                  <tr key={f.filename}>
                    <td className={styles.cellMono}>{f.filename}</td>
                    <td className={styles.cellNumeric}>
                      {fmtBytes(f.size_bytes)}
                    </td>
                    <td className={styles.cellNumeric}>{fmtRows(f.num_rows)}</td>
                    <td className={styles.cellMono}>
                      {f.data_range
                        ? `${f.data_range.start} → ${f.data_range.end}`
                        : '—'}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
