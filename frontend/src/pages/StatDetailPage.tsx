import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { Accordion, Alert, Box, LoadingOverlay, Skeleton } from '@mantine/core'
import { IconAlertCircle, IconArrowLeft, IconChartBar, IconRefresh } from '@tabler/icons-react'
import { useStatDetail } from '../hooks/useStatDetail'
import { useTimeframeResult } from '../hooks/useTimeframeResult'
import { InstrumentSelect } from '../components/InstrumentSelect'
import { DateRangePopover } from '../components/DateRangePopover'
import { MagnitudeBarChart } from '../components/charts/MagnitudeBarChart'
import { ProbabilityBarChart } from '../components/charts/ProbabilityBarChart'
import { SliceGroupedBarChart } from '../components/charts/SliceGroupedBarChart'
import {
  computePresetRange,
  DURATION_PRESETS,
  PRESET_LABELS,
  type PeriodPreset,
} from '../utils/periodPresets'
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
  showCharts: boolean
}

function SlicesSection({ slices, labels, showCharts }: SlicesSectionProps) {
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

            // Determine if this slice dimension contains probability rows
            const hasProbRows = Object.values(sliceResult.groups).some((g) =>
              g.results.some((r) => r.value === null || r.value === undefined),
            )

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
                    {/* Grouped bar chart across all groups for this dimension */}
                    {showCharts && (
                      <SliceGroupedBarChart
                        groups={sliceResult.groups}
                        labels={labels}
                        preferProbability={hasProbRows}
                      />
                    )}

                    {/* Per-group detail tables */}
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
  showCharts: boolean
}

function ResultsSection({ tfResult, labels, showCharts }: ResultsSectionProps) {
  // Discriminate by `value` presence, consistent with ResultRowsTable — a
  // probability of exactly 0 is a valid probability row, not a magnitude row.
  const probRows = tfResult.results.filter(
    (r) => r.value === null || r.value === undefined,
  )
  const magRows = tfResult.results.filter(
    (r) => r.value !== null && r.value !== undefined,
  )
  const hasProbRows = probRows.length > 0
  const hasMagRows = magRows.length > 0

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

      {/* Side-by-side: table on the left, chart on the right */}
      <div className={showCharts ? styles.resultsLayout : undefined}>
        <div className={styles.resultsTable}>
          <ResultRowsTable rows={tfResult.results} labels={labels} />
        </div>
        {showCharts && (
          <div className={styles.resultsChart}>
            {hasProbRows && (
              <ProbabilityBarChart rows={probRows} labels={labels} />
            )}
            {hasMagRows && (
              <MagnitudeBarChart rows={magRows} labels={labels} />
            )}
          </div>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Period filter bar — global row (instrument + period) + stat-specific row
// (timeframe + context + charts toggle). Mirrors docs/mockups/stat-detail.html.
// ---------------------------------------------------------------------------

/** Parse an ISO "YYYY-MM-DD" string to a local Date, or undefined if absent. */
function parseISODate(iso: string | undefined): Date | undefined {
  if (!iso) return undefined
  const [y, m, d] = iso.split('-').map(Number)
  if (!y || !m || !d) return undefined
  return new Date(y, m - 1, d)
}

interface FilterBarProps {
  instruments: string[]
  timeframes: string[]
  effectiveInstrument: string
  effectiveTimeframe: string
  onTimeframeChange: (tf: string) => void
  period: PeriodPreset
  onPeriodChange: (p: PeriodPreset) => void
  customRange: [string | null, string | null]
  onCustomApply: (start: string, end: string) => void
  dataRange: string[] | undefined
  contextSamples: number | null
  showCharts: boolean
  onShowChartsChange: (v: boolean) => void
}

function FilterBar({
  instruments,
  timeframes,
  effectiveInstrument,
  effectiveTimeframe,
  onTimeframeChange,
  period,
  onPeriodChange,
  customRange,
  onCustomApply,
  dataRange,
  contextSamples,
  showCharts,
  onShowChartsChange,
}: FilterBarProps) {
  const chipClass = (active: boolean) =>
    `${styles.filterChip} ${active ? styles.filterChipActive : ''}`

  const contextLabel = [
    effectiveInstrument,
    effectiveTimeframe,
    contextSamples !== null
      ? `${intFormat.format(contextSamples)} samples`
      : null,
  ]
    .filter(Boolean)
    .join(' · ')

  return (
    <div className={styles.filterBar}>
      {/* Row 1 — global filters: instrument + period presets + custom range */}
      <div className={styles.filterBarRow}>
        {instruments.length > 0 && (
          <div className={styles.filterBarGroup}>
            <span className={styles.filterBarLabel}>Instrument</span>
            <InstrumentSelect />
          </div>
        )}

        <span className={styles.filterBarSep} aria-hidden="true" />

        <span className={styles.filterBarLabel}>Period</span>
        {/* "All" (no filter, full dataset) sits alone before the durations */}
        <div className={styles.filterGroupCompact}>
          <button
            type="button"
            className={chipClass(period === 'all')}
            onClick={() => onPeriodChange('all')}
          >
            {PRESET_LABELS.all}
          </button>
        </div>

        <span className={styles.filterBarSep} aria-hidden="true" />

        <div className={styles.filterGroupCompact}>
          {DURATION_PRESETS.map((preset) => (
            <button
              key={preset}
              type="button"
              className={chipClass(period === preset)}
              onClick={() => onPeriodChange(preset)}
            >
              {PRESET_LABELS[preset]}
            </button>
          ))}
        </div>

        <span className={styles.filterBarSep} aria-hidden="true" />

        <DateRangePopover
          active={period === 'custom'}
          value={customRange}
          minDate={parseISODate(dataRange?.[0])}
          maxDate={parseISODate(dataRange?.[1])}
          onApply={onCustomApply}
        />
      </div>

      {/* Row 2 — stat-specific filters: timeframe + context + charts toggle */}
      <div className={`${styles.filterBarRow} ${styles.filterBarRowStat}`}>
        <div className={styles.filterBarGroup}>
          <span className={styles.filterBarLabel}>Timeframe</span>
          <div className={styles.filterGroup}>
            {timeframes.map((tf) => (
              <button
                key={tf}
                type="button"
                className={chipClass(tf === effectiveTimeframe)}
                onClick={() => onTimeframeChange(tf)}
              >
                {tf}
              </button>
            ))}
          </div>
        </div>

        <span className={styles.filterBarSpacer} />

        <span className={styles.resultCount}>{contextLabel}</span>

        <span className={styles.filterBarSep} aria-hidden="true" />

        <label className={styles.chartsToggle}>
          <span className={styles.chartsToggleLabel}>
            <IconChartBar size={12} />
            Charts
          </span>
          <input
            type="checkbox"
            checked={showCharts}
            onChange={(e) => onShowChartsChange(e.currentTarget.checked)}
          />
          <span className={styles.chartsSwitch} aria-hidden="true" />
        </label>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// StatDetailPage — main page component
// ---------------------------------------------------------------------------

export function StatDetailPage() {
  const { instrument, family } = useParams<{ instrument: string; family: string }>()
  const { detail, loading, error, reload } = useStatDetail(family)

  // Instrument comes from the URL — the filter-bar Select navigates to a new
  // prefix. Timeframe selection is local to this page.
  const [selectedTimeframe, setSelectedTimeframe] = useState<string>('')

  // Chart visibility toggle — charts visible by default
  const [showCharts, setShowCharts] = useState<boolean>(true)

  // Period filter — local state, resets on navigation (instrument/timeframe).
  const [period, setPeriod] = useState<PeriodPreset>('all')
  const [customRange, setCustomRange] = useState<[string | null, string | null]>([
    null,
    null,
  ])

  // Derive selector options once detail is loaded
  const instruments = detail
    ? Object.keys(detail.result.instruments)
    : []

  // Honour the URL instrument when this family has data for it; otherwise
  // fall back to the family's first instrument so the page still renders.
  const effectiveInstrument =
    instrument && instruments.includes(instrument)
      ? instrument
      : instruments[0] ?? ''

  const timeframes =
    detail && effectiveInstrument
      ? Object.keys(detail.result.instruments[effectiveInstrument] ?? {})
      : []

  const effectiveTimeframe =
    selectedTimeframe && timeframes.includes(selectedTimeframe)
      ? selectedTimeframe
      : timeframes[0] ?? ''

  // The full, pre-computed result for the current instrument/timeframe. Serves
  // the "All" period directly and anchors period-preset date math.
  const baseTf: TimeframeResult | null =
    detail && effectiveInstrument && effectiveTimeframe
      ? (detail.result.instruments[effectiveInstrument]?.[effectiveTimeframe] ??
        null)
      : null

  // Period persists across instrument and timeframe changes (it only resets on
  // navigation, when the component unmounts/remounts). Duration presets
  // recompute their dates against the new data_range[1] below, preserving the
  // user's intent ("last 3 months"); a Custom absolute range carries over as-is
  // and the empty state covers a context with no data in that window.
  const filterActive = period !== 'all'
  const dataEnd = baseTf?.data_range[1]

  let start: string | null = null
  let end: string | null = null
  if (period === 'custom') {
    ;[start, end] = customRange
  } else if (filterActive) {
    const range = computePresetRange(period, dataEnd)
    if (range) {
      start = range.start
      end = range.end
    }
  }

  const {
    result: filteredTf,
    loading: filterLoading,
    error: filterError,
  } = useTimeframeResult(family, effectiveInstrument, effectiveTimeframe, start, end)

  // The result feeding the tables/charts/strip: full when "All", filtered
  // otherwise (null while a filtered fetch is in flight).
  const tfResult: TimeframeResult | null = filterActive ? filteredTf : baseTf
  // Keep the metric strip populated during a filtered refetch.
  const displayTf = tfResult ?? baseTf

  function handleCustomApply(rangeStart: string, rangeEnd: string) {
    setCustomRange([rangeStart, rangeEnd])
    setPeriod('custom')
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
        <Link to={`/${instrument}/stats`} className={styles.backLink}>
          <IconArrowLeft size={12} />
          <span>Stat Families</span>
        </Link>
        <div className={styles.titleRow}>
          <h1 className={styles.pageTitle}>{result.title.en}</h1>
          {effectiveInstrument && (
            <span className={styles.badgeInstrument}>{effectiveInstrument}</span>
          )}
        </div>
      </div>

      {/* ===== Metadata metric strip ===== */}
      {displayTf && (
        <div className={styles.metricStrip}>
          <div className={styles.metric}>
            <span className={styles.metricValue}>
              {displayTf.data_range[0] ?? '—'} → {displayTf.data_range[1] ?? '—'}
            </span>
            <span className={styles.metricLabel}>Date range</span>
          </div>
          <div className={styles.metric}>
            <span className={styles.metricValue}>
              {intFormat.format(displayTf.total_samples)}
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

      {/* ===== Unified filter bar (instrument · period | timeframe · charts) ===== */}
      {instruments.length > 0 && (
        <FilterBar
          instruments={instruments}
          timeframes={timeframes}
          effectiveInstrument={effectiveInstrument}
          effectiveTimeframe={effectiveTimeframe}
          onTimeframeChange={setSelectedTimeframe}
          period={period}
          onPeriodChange={setPeriod}
          customRange={customRange}
          onCustomApply={handleCustomApply}
          dataRange={baseTf?.data_range}
          contextSamples={displayTf?.total_samples ?? null}
          showCharts={showCharts}
          onShowChartsChange={setShowCharts}
        />
      )}

      {/* ===== Documentation panel ===== */}
      <div className={styles.panel}>
        <div className={styles.panelHead}>
          <h2 className={styles.panelTitle}>Definition</h2>
        </div>
        <p className={styles.docDefinition}>{result.definition.en}</p>
        <div className={styles.paramRow}>
          {effectiveInstrument && (
            <span className={styles.param}>
              <span className={styles.paramKey}>Instrument</span>
              <span className={styles.paramVal}>{effectiveInstrument}</span>
            </span>
          )}
          {timeframes.map((tf) => (
            <span key={tf} className={styles.param}>
              <span className={styles.paramKey}>Timeframe</span>
              <span className={styles.paramVal}>{tf}</span>
            </span>
          ))}
        </div>
      </div>

      {/* ===== Results panel ===== */}
      {/* A filtered fetch may be in flight (overlay), have failed (alert), or
          have returned zero samples for the chosen window (empty state). */}
      <Box pos="relative" mih={filterActive ? 120 : undefined}>
        <LoadingOverlay
          visible={filterActive && filterLoading}
          overlayProps={{ blur: 1 }}
        />
        {filterActive && filterError ? (
          <Alert
            icon={<IconAlertCircle size={16} />}
            color="red"
            variant="light"
            title="Failed to load filtered result"
          >
            <p className={styles.alertMessage}>{filterError}</p>
          </Alert>
        ) : tfResult && tfResult.total_samples === 0 ? (
          <div className={styles.panel}>
            <p className={styles.emptyPeriod}>
              No data available for this period.
            </p>
          </div>
        ) : tfResult ? (
          <ResultsSection
            tfResult={tfResult}
            labels={result.labels}
            showCharts={showCharts}
          />
        ) : null}
      </Box>

      {/* ===== Slices ===== */}
      {/* Hidden while a date filter is active: the backend returns no slices
          for reaggregated results, and combining slices with a range is
          unsupported. */}
      {!filterActive &&
        tfResult &&
        Object.keys(tfResult.slices).length > 0 && (
          <SlicesSection
            slices={tfResult.slices}
            labels={result.labels}
            showCharts={showCharts}
          />
        )}
    </div>
  )
}
