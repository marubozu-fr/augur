import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { StatResultRow, Labels } from '../../types/stats'
import {
  CHART_FONT_FAMILY,
  CHART_FONT_SIZE,
  CHART_GRID_COLOR,
  CHART_NEGATIVE,
  CHART_NEUTRAL,
  CHART_POSITIVE,
  CHART_TEXT_DIM,
  CHART_TEXT_MUTED,
} from './chartTheme'
import { ChartTooltip } from './ChartTooltip'
import styles from './MagnitudeBarChart.module.css'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fmtNum(v: number): string {
  return v.toFixed(2)
}

function barColor(value: number, baseline: number | null): string {
  if (baseline === null) return CHART_NEUTRAL
  const delta = value - baseline
  if (Math.abs(delta) < 0.0001) return CHART_NEUTRAL
  return delta > 0 ? CHART_POSITIVE : CHART_NEGATIVE
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ChartDataPoint {
  label: string
  value: number
  baseline: number | null
  fill: string
}

interface MagnitudeBarChartProps {
  rows: StatResultRow[]
  labels: Labels
}

// ---------------------------------------------------------------------------
// MagnitudeBarChart
// ---------------------------------------------------------------------------

export function MagnitudeBarChart({ rows, labels }: MagnitudeBarChartProps) {
  const data: ChartDataPoint[] = rows.map((row) => {
    // This component only receives magnitude rows, so `value` is present; the
    // guard makes the intent explicit and avoids a non-null cast.
    const value = row.value != null ? row.value : 0
    return {
      label: labels.outcomes[row.outcome]?.en ?? row.outcome,
      value,
      baseline: row.value_baseline,
      fill: barColor(value, row.value_baseline),
    }
  })

  return (
    <div className={styles.chartWrap}>
      <ResponsiveContainer width="100%" height={Math.max(180, data.length * 68 + 60)}>
        <BarChart
          data={data}
          margin={{ top: 16, right: 48, bottom: 16, left: 8 }}
          barCategoryGap="30%"
          barGap={4}
        >
          <CartesianGrid
            vertical={false}
            strokeDasharray="3 3"
            stroke={CHART_GRID_COLOR}
          />
          <XAxis
            dataKey="label"
            tick={{ fill: CHART_TEXT_MUTED, fontSize: CHART_FONT_SIZE, fontFamily: CHART_FONT_FAMILY }}
            tickLine={false}
            axisLine={{ stroke: CHART_GRID_COLOR }}
          />
          <YAxis
            tickFormatter={fmtNum}
            tick={{ fill: CHART_TEXT_DIM, fontSize: CHART_FONT_SIZE, fontFamily: CHART_FONT_FAMILY }}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            content={<ChartTooltip formatter={fmtNum} />}
            cursor={{ fill: 'rgba(255,255,255,0.04)' }}
          />
          <Legend
            wrapperStyle={{
              fontSize: CHART_FONT_SIZE,
              fontFamily: CHART_FONT_FAMILY,
              color: CHART_TEXT_MUTED,
              paddingTop: 8,
            }}
          />
          <Bar dataKey="value" name="Value" radius={[2, 2, 0, 0]}>
            <LabelList
              dataKey="value"
              position="top"
              formatter={(v: string | number | boolean | null | undefined) =>
                typeof v === 'number' ? fmtNum(v) : ''
              }
              style={{ fill: CHART_TEXT_MUTED, fontSize: CHART_FONT_SIZE, fontFamily: CHART_FONT_FAMILY }}
            />
          </Bar>
          <Bar
            dataKey="baseline"
            name="Baseline"
            fill={CHART_NEUTRAL}
            fillOpacity={0.35}
            radius={[2, 2, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
