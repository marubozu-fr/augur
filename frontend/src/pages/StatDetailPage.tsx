import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import {
  Accordion,
  Alert,
  SegmentedControl,
  Skeleton,
} from '@mantine/core'
import { IconAlertCircle, IconArrowLeft, IconRefresh } from '@tabler/icons-react'
import { useStatDetail } from '../hooks/useStatDetail'
import type {
  Labels,
  SliceResult,
  StatResultRow,
  TimeframeResult,
} from '../types/stats'
import styles from './StatDetailPage.module.css'

// ---------------------------------------------------------------------------
// Number formatting helpers
// ---------------------------------------------------------------------------

const pctFormat = new Intl.NumberFormat('en-US', {
  style: 'percent',
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
})

const intFormat = new Intl.NumberFormat('en-US')

function fmtPct(v: number): string {
  return pctFormat.format(v)
}

function fmtNum(v: number): string {
  return v.toFixed(2)
}

function fmtEdgePct(v: number): string {
  const sign = v > 0 ? '+' : ''
  return `${sign}${(v * 100).toFixed(1)}pp`
}

function fmtEdgeNum(v: number): string {
  const sign = v > 0 ? '+' : ''
  return `${sign}${v.toFixed(2)}`
}

// ---------------------------------------------------------------------------
// Edge class helper — neutral band ±0.1pp
// ---------------------------------------------------------------------------

function edgeClass(edge: number): string | undefined {
  if (Math.abs(edge) < 0.001) return styles.edgeNeutral
  return edge > 0 ? styles.edgePositive : styles.edgeNegative
}

// ---------------------------------------------------------------------------
// ResultRowsTable — renders probability OR magnitude rows.
// Used at both the top-level and inside each slice group.
// ---------------------------------------------------------------------------

interface ResultRowsTableProps {
  rows: StatResultRow[]
  labels: Labels
}

function ResultRowsTable({ rows, labels }: ResultRowsTableProps) {
  // Magnitude rows carry a continuous `value`; probability rows leave it null.
  // The discriminant is value presence, NOT probability > 0 — a probability of
  // exactly 0 (0 occurrences over N) is a valid probability result, common on
  // small-N slices, and must stay in the probability table.
  const probRows = rows.filter((r) => r.value === null || r.value === undefined)
  const magRows = rows.filter((r) => r.value !== null && r.value !== undefined)

  return (
    <>
      {probRows.length > 0 && (
        <ProbabilityTable rows={probRows} labels={labels} />
      )}
      {magRows.length > 0 && (
        <MagnitudeGrid rows={magRows} labels={labels} />
      )}
    </>
  )
}

// ---------------------------------------------------------------------------
// ProbabilityTable
// ---------------------------------------------------------------------------

interface ProbabilityTableProps {
  rows: StatResultRow[]
  labels: Labels
}

function ProbabilityTable({ rows, labels }: ProbabilityTableProps) {
  return (
    <div className={styles.tableWrap}>
      <table className={styles.dataTable}>
        <thead>
          <tr>
            <th>Condition</th>
            <th>Outcome</th>
            <th className={styles.colNum}>Probability</th>
            <th className={styles.colNum}>Baseline</th>
            <th className={styles.colNum}>N</th>
            <th className={styles.colNum}>Edge</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => {
            const condLabel =
              labels.conditions[row.condition]?.en ?? row.condition
            const outcomeLabel =
              labels.outcomes[row.outcome]?.en ?? row.outcome
            const edge = row.probability - row.baseline_prob
            const aboveBaseline = edge > 0
            const barFillClass = Math.abs(edge) < 0.001
              ? styles.probBarFillNeutral
              : aboveBaseline
                ? styles.probBarFillPositive
                : styles.probBarFillNegative

            return (
              <tr key={i}>
                <td className={styles.cellCondition}>{condLabel}</td>
                <td>{outcomeLabel}</td>
                <td className={styles.cellNum}>
                  <div className={styles.probCell}>
                    <span className={styles.probBar}>
                      <span
                        className={barFillClass}
                        style={{ width: `${(row.probability * 100).toFixed(1)}%` }}
                      />
                    </span>
                    <span className={styles.probValue}>
                      {fmtPct(row.probability)}
                    </span>
                  </div>
                </td>
                <td className={`${styles.cellNum} ${styles.cellBaseline}`}>
                  {fmtPct(row.baseline_prob)}
                </td>
                <td className={styles.cellNum}>{intFormat.format(row.total)}</td>
                <td className={`${styles.cellNum} ${edgeClass(edge)}`}>
                  {fmtEdgePct(edge)}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

// ---------------------------------------------------------------------------
// MagnitudeGrid
// ---------------------------------------------------------------------------

interface MagnitudeGridProps {
  rows: StatResultRow[]
  labels: Labels
}

function MagnitudeGrid({ rows, labels }: MagnitudeGridProps) {
  return (
    <div className={styles.magGrid}>
      {rows.map((row, i) => {
        const outcomeLabel = labels.outcomes[row.outcome]?.en ?? row.outcome
        const val = row.value as number
        const base = row.value_baseline
        const delta = base !== null ? val - base : null

        return (
          <div key={i} className={styles.magBlock}>
            <span className={styles.magLabel}>{outcomeLabel}</span>
            <span className={styles.magValue}>{fmtNum(val)}</span>
            <div className={styles.magFoot}>
              <span className={styles.magBaseline}>
                {base !== null ? `Baseline ${fmtNum(base)}` : 'No baseline'}
              </span>
              {delta !== null && (
                <span className={edgeClass(delta)}>
                  {fmtEdgeNum(delta)}
                </span>
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}

// ---------------------------------------------------------------------------
// SlicesSection — Accordion with one item per slice dimension
// ---------------------------------------------------------------------------

interface SlicesSectionProps {
  slices: Record<string, SliceResult>
  labels: Labels
}

function SlicesSection({ slices, labels }: SlicesSectionProps) {
  const entries = Object.entries(slices)
  if (entries.length === 0) return null

  return (
    <div className={styles.panel}>
      <div className={styles.panelHead}>
        <h2 className={styles.panelTitle}>Slices</h2>
      </div>
      <div className={styles.slicesWrapper}>
        <Accordion variant="separated" radius="md">
          {entries.map(([dimKey, sliceResult]) => {
            const dimLabel =
              labels.dimensions[dimKey]?.en ?? dimKey
            const groups = Object.entries(sliceResult.groups)

            return (
              <Accordion.Item key={dimKey} value={dimKey}>
                <Accordion.Control>
                  <span>{dimLabel}</span>
                  <span className={styles.sliceCount}>
                    {groups.length} group{groups.length !== 1 ? 's' : ''}
                  </span>
                </Accordion.Control>
                <Accordion.Panel>
                  <div className={styles.sliceContent}>
                    {groups.map(([groupKey, group]) => (
                      <div key={groupKey} className={styles.sliceGroup}>
                        <div className={styles.sliceGroupHead}>
                          <span className={styles.sliceGroupLabel}>
                            {group.label.en}
                          </span>
                          <span className={styles.sliceGroupCount}>
                            {intFormat.format(group.total_samples)} samples
                          </span>
                        </div>
                        <ResultRowsTable
                          rows={group.results}
                          labels={labels}
                        />
                      </div>
                    ))}
                  </div>
                </Accordion.Panel>
              </Accordion.Item>
            )
          })}
        </Accordion>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// ResultsSection — top-level results panel
// ---------------------------------------------------------------------------

interface ResultsSectionProps {
  tfResult: TimeframeResult
  labels: Labels
}

function ResultsSection({ tfResult, labels }: ResultsSectionProps) {
  // Discriminate by `value` presence, consistent with ResultRowsTable — a
  // probability of exactly 0 is a valid probability row, not a magnitude row.
  const hasProbRows = tfResult.results.some(
    (r) => r.value === null || r.value === undefined,
  )
  const hasMagRows = tfResult.results.some(
    (r) => r.value !== null && r.value !== undefined,
  )

  const typeTag = hasProbRows ? (
    <span className={styles.statTypeProbability}>Probability</span>
  ) : hasMagRows ? (
    <span className={styles.statTypeMagnitude}>Magnitude</span>
  ) : null

  return (
    <div className={styles.panel}>
      <div className={styles.panelHead}>
        {typeTag}
        <h2 className={styles.panelTitle}>Results</h2>
        <div className={styles.panelSpacer} />
        <span className={styles.panelNote}>
          Probability vs random baseline · edge = Δ
        </span>
      </div>
      <ResultRowsTable rows={tfResult.results} labels={labels} />
    </div>
  )
}

// ---------------------------------------------------------------------------
// StatDetailPage — main page component
// ---------------------------------------------------------------------------

export function StatDetailPage() {
  const { family } = useParams<{ family: string }>()
  const { detail, loading, error, reload } = useStatDetail(family)

  // Instrument + timeframe selection state
  const [selectedInstrument, setSelectedInstrument] = useState<string>('')
  const [selectedTimeframe, setSelectedTimeframe] = useState<string>('')

  // Derive selector options once detail is loaded
  const instruments = detail
    ? Object.keys(detail.result.instruments)
    : []

  const effectiveInstrument =
    selectedInstrument && instruments.includes(selectedInstrument)
      ? selectedInstrument
      : instruments[0] ?? ''

  const timeframes =
    detail && effectiveInstrument
      ? Object.keys(detail.result.instruments[effectiveInstrument] ?? {})
      : []

  const effectiveTimeframe =
    selectedTimeframe && timeframes.includes(selectedTimeframe)
      ? selectedTimeframe
      : timeframes[0] ?? ''

  const tfResult: TimeframeResult | null =
    detail && effectiveInstrument && effectiveTimeframe
      ? (detail.result.instruments[effectiveInstrument]?.[effectiveTimeframe] ??
        null)
      : null

  // When instrument changes, reset timeframe so first TF is auto-selected
  function handleInstrumentChange(val: string) {
    setSelectedInstrument(val)
    setSelectedTimeframe('')
  }

  // ---------------------------------------------------------------------------
  // Loading state
  // ---------------------------------------------------------------------------

  if (loading) {
    return (
      <div className={styles.page}>
        <div className={styles.skeletonStack}>
          <Skeleton height={28} width="40%" radius="sm" />
          <Skeleton height={14} width="60%" radius="sm" />
          <Skeleton height={120} radius="md" />
          <Skeleton height={200} radius="md" />
          <Skeleton height={160} radius="md" />
        </div>
      </div>
    )
  }

  // ---------------------------------------------------------------------------
  // Error state
  // ---------------------------------------------------------------------------

  if (error) {
    return (
      <div className={styles.page}>
        <Alert
          icon={<IconAlertCircle size={16} />}
          color="red"
          variant="light"
          title="Failed to load stat detail"
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
      </div>
    )
  }

  if (!detail) return null

  const { result } = detail

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className={styles.page}>
      {/* ===== Page head ===== */}
      <div>
        <Link to="/" className={styles.backLink}>
          <IconArrowLeft size={12} />
          <span>Stat Families</span>
        </Link>
        <div className={styles.titleRow}>
          <h1 className={styles.pageTitle}>{result.title.en}</h1>
          {instruments.map((inst) => (
            <span key={inst} className={styles.badgeInstrument}>
              {inst}
            </span>
          ))}
        </div>
      </div>

      {/* ===== Metadata metric strip ===== */}
      {tfResult && (
        <div className={styles.metricStrip}>
          <div className={styles.metric}>
            <span className={styles.metricValue}>
              {tfResult.data_range[0] ?? '—'} → {tfResult.data_range[1] ?? '—'}
            </span>
            <span className={styles.metricLabel}>Date range</span>
          </div>
          <div className={styles.metric}>
            <span className={styles.metricValue}>
              {intFormat.format(tfResult.total_samples)}
            </span>
            <span className={styles.metricLabel}>Total samples</span>
          </div>
          <div className={styles.metric}>
            <span className={styles.metricValue}>
              {detail.computed_at ?? '—'}
            </span>
            <span className={styles.metricLabel}>Last computed</span>
          </div>
        </div>
      )}

      {/* ===== Instrument + timeframe selector ===== */}
      {instruments.length > 0 && (
        <div className={styles.selectorBar}>
          {instruments.length > 1 && (
            <>
              <span className={styles.selectorLabel}>Instrument</span>
              <SegmentedControl
                size="xs"
                value={effectiveInstrument}
                onChange={handleInstrumentChange}
                data={instruments}
              />
              <span className={styles.selectorDivider} />
            </>
          )}
          <span className={styles.selectorLabel}>Timeframe</span>
          <SegmentedControl
            size="xs"
            value={effectiveTimeframe}
            onChange={setSelectedTimeframe}
            data={timeframes}
          />
        </div>
      )}

      {/* ===== Documentation panel ===== */}
      <div className={styles.panel}>
        <div className={styles.panelHead}>
          <h2 className={styles.panelTitle}>Definition</h2>
        </div>
        <p className={styles.docDefinition}>{result.definition.en}</p>
        <div className={styles.paramRow}>
          {instruments.map((inst) => (
            <span key={inst} className={styles.param}>
              <span className={styles.paramKey}>Instrument</span>
              <span className={styles.paramVal}>{inst}</span>
            </span>
          ))}
          {timeframes.map((tf) => (
            <span key={tf} className={styles.param}>
              <span className={styles.paramKey}>Timeframe</span>
              <span className={styles.paramVal}>{tf}</span>
            </span>
          ))}
        </div>
      </div>

      {/* ===== Results panel ===== */}
      {tfResult && (
        <ResultsSection
          tfResult={tfResult}
          labels={result.labels}
        />
      )}

      {/* ===== Slices ===== */}
      {tfResult && Object.keys(tfResult.slices).length > 0 && (
        <SlicesSection
          slices={tfResult.slices}
          labels={result.labels}
        />
      )}
    </div>
  )
}
