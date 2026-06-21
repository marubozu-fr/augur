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
import styles from './ProbabilityBarChart.module.css'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fmtPct(v: number): string {
  return `${(v * 100).toFixed(1)}%`
}

function barColor(probability: number, baselineProb: number): string {
  const edge = probability - baselineProb
  if (Math.abs(edge) < 0.001) return CHART_NEUTRAL
  return edge > 0 ? CHART_POSITIVE : CHART_NEGATIVE
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ChartDataPoint {
  label: string
  probability: number
  baseline: number
  edge: number
  fill: string
}

interface ProbabilityBarChartProps {
  rows: StatResultRow[]
  labels: Labels
}

// ---------------------------------------------------------------------------
// ProbabilityBarChart
// ---------------------------------------------------------------------------

export function ProbabilityBarChart({ rows, labels }: ProbabilityBarChartProps) {
  const data: ChartDataPoint[] = rows.map((row) => {
    const condLabel = labels.conditions[row.condition]?.en ?? row.condition
    const outcomeLabel = labels.outcomes[row.outcome]?.en ?? row.outcome
    const label =
      condLabel === outcomeLabel ? condLabel : `${condLabel} / ${outcomeLabel}`
    return {
      label,
      probability: row.probability,
      baseline: row.baseline_prob,
      edge: row.probability - row.baseline_prob,
      fill: barColor(row.probability, row.baseline_prob),
    }
  })

  // Compute axis domain: 0 to max(probability, baseline) * 1.15, rounded up
  const maxVal = data.reduce(
    (m, d) => Math.max(m, d.probability, d.baseline),
    0,
  )
  const axisDomain: [number, number] = [0, Math.min(1, maxVal * 1.2)]

  return (
    <div className={styles.chartWrap}>
      <ResponsiveContainer width="100%" height={Math.max(160, data.length * 52 + 40)}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 48, bottom: 4, left: 8 }}
          barCategoryGap="28%"
          barGap={3}
        >
          <CartesianGrid
            horizontal={false}
            strokeDasharray="3 3"
            stroke={CHART_GRID_COLOR}
          />
          <XAxis
            type="number"
            domain={axisDomain}
            tickFormatter={fmtPct}
            tick={{ fill: CHART_TEXT_DIM, fontSize: CHART_FONT_SIZE, fontFamily: CHART_FONT_FAMILY }}
            tickLine={false}
            axisLine={{ stroke: CHART_GRID_COLOR }}
          />
          <YAxis
            type="category"
            dataKey="label"
            width={160}
            tick={{ fill: CHART_TEXT_MUTED, fontSize: CHART_FONT_SIZE, fontFamily: CHART_FONT_FAMILY }}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            content={<ChartTooltip formatter={fmtPct} />}
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
          <Bar dataKey="probability" name="Probability" radius={[0, 2, 2, 0]}>
            <LabelList
              dataKey="probability"
              position="right"
              formatter={(v: string | number | boolean | null | undefined) =>
                typeof v === 'number' ? fmtPct(v) : ''
              }
              style={{ fill: CHART_TEXT_MUTED, fontSize: CHART_FONT_SIZE, fontFamily: CHART_FONT_FAMILY }}
            />
          </Bar>
          <Bar
            dataKey="baseline"
            name="Baseline"
            fill={CHART_NEUTRAL}
            fillOpacity={0.35}
            radius={[0, 2, 2, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
