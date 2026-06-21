import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { SliceGroupResult, Labels } from '../../types/stats'
import {
  CHART_ACCENT,
  CHART_FONT_FAMILY,
  CHART_FONT_SIZE,
  CHART_GRID_COLOR,
  CHART_POSITIVE,
  CHART_NEGATIVE,
  CHART_TEXT_DIM,
  CHART_TEXT_MUTED,
} from './chartTheme'
import { ChartTooltip } from './ChartTooltip'
import styles from './SliceGroupedBarChart.module.css'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fmtPct(v: number): string {
  return `${(v * 100).toFixed(1)}%`
}

function fmtNum(v: number): string {
  return v.toFixed(2)
}

/**
 * Pick a deterministic color for the n-th outcome series. We use a fixed
 * palette to keep colors consistent across re-renders.
 */
const SERIES_COLORS: [string, ...string[]] = [
  CHART_ACCENT,
  CHART_POSITIVE,
  CHART_NEGATIVE,
  '#c9a227', // --color-instrument
  '#a78bfa', // purple
  '#f59e0b', // amber
]

function seriesColor(index: number): string {
  return SERIES_COLORS[index % SERIES_COLORS.length] ?? CHART_ACCENT
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** One row in the BarChart data array — keys are outcome slugs, value is probability */
type ChartDataPoint = Record<string, string | number>

interface SliceGroupedBarChartProps {
  /** groups keyed by group slug, in iteration order */
  groups: Record<string, SliceGroupResult>
  labels: Labels
  /** Optional: show only probability rows (default true — magnitude slices fall back to value) */
  preferProbability?: boolean
}

// ---------------------------------------------------------------------------
// SliceGroupedBarChart
// ---------------------------------------------------------------------------

export function SliceGroupedBarChart({
  groups,
  labels,
  preferProbability = true,
}: SliceGroupedBarChartProps) {
  const groupEntries = Object.entries(groups)
  if (groupEntries.length === 0) return null

  // Collect all unique outcome keys across all groups (preserve first-seen order)
  const outcomeKeys: string[] = []
  for (const [, group] of groupEntries) {
    for (const row of group.results) {
      const isProbRow = row.value === null || row.value === undefined
      if (preferProbability && !isProbRow) continue
      if (!preferProbability && isProbRow) continue
      if (!outcomeKeys.includes(row.outcome)) {
        outcomeKeys.push(row.outcome)
      }
    }
  }

  // Fallback: if preferred mode yielded nothing, try the other mode
  if (outcomeKeys.length === 0) {
    for (const [, group] of groupEntries) {
      for (const row of group.results) {
        if (!outcomeKeys.includes(row.outcome)) {
          outcomeKeys.push(row.outcome)
        }
      }
    }
  }

  // Build one data point per group
  const data: ChartDataPoint[] = groupEntries.map(([, group]) => {
    const point: ChartDataPoint = { _label: group.label.en }
    for (const outcomeKey of outcomeKeys) {
      const matchingRow = group.results.find((r) => r.outcome === outcomeKey)
      if (matchingRow) {
        const isProbRow =
          matchingRow.value === null || matchingRow.value === undefined
        point[outcomeKey] = isProbRow
          ? matchingRow.probability
          : (matchingRow.value as number)
      }
    }
    return point
  })

  const isMagnitude = !preferProbability

  return (
    <div className={styles.chartWrap}>
      <ResponsiveContainer
        width="100%"
        height={Math.max(200, groupEntries.length * 28 + 100)}
      >
        <BarChart
          data={data}
          margin={{ top: 8, right: 16, bottom: 8, left: 8 }}
          barCategoryGap="20%"
          barGap={2}
        >
          <CartesianGrid
            vertical={false}
            strokeDasharray="3 3"
            stroke={CHART_GRID_COLOR}
          />
          <XAxis
            dataKey="_label"
            tick={{
              fill: CHART_TEXT_MUTED,
              fontSize: CHART_FONT_SIZE,
              fontFamily: CHART_FONT_FAMILY,
            }}
            tickLine={false}
            axisLine={{ stroke: CHART_GRID_COLOR }}
          />
          <YAxis
            tickFormatter={isMagnitude ? (v: number) => v.toFixed(2) : fmtPct}
            tick={{
              fill: CHART_TEXT_DIM,
              fontSize: CHART_FONT_SIZE,
              fontFamily: CHART_FONT_FAMILY,
            }}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            content={<ChartTooltip formatter={isMagnitude ? fmtNum : fmtPct} />}
            cursor={{ fill: 'rgba(255,255,255,0.04)' }}
          />
          {outcomeKeys.length > 1 && (
            <Legend
              wrapperStyle={{
                fontSize: CHART_FONT_SIZE,
                fontFamily: CHART_FONT_FAMILY,
                color: CHART_TEXT_MUTED,
                paddingTop: 8,
              }}
            />
          )}
          {outcomeKeys.map((outcomeKey, i) => (
            <Bar
              key={outcomeKey}
              dataKey={outcomeKey}
              name={labels.outcomes[outcomeKey]?.en ?? outcomeKey}
              fill={seriesColor(i)}
              radius={[2, 2, 0, 0]}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
