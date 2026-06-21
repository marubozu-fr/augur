import styles from './ChartTooltip.module.css'

// ---------------------------------------------------------------------------
// Shared chart tooltip
//
// Recharts injects `active`, `payload`, and `label` when it renders the
// element passed to a <Tooltip content={...} />. The only per-chart variation
// is the value `formatter` (percentage vs raw number), which callers supply.
// ---------------------------------------------------------------------------

export interface ChartTooltipPayloadEntry {
  name?: string
  value?: number | null
  color?: string
  dataKey?: string | number
}

export interface ChartTooltipProps {
  active?: boolean
  payload?: ChartTooltipPayloadEntry[]
  label?: string
  /** Formats a numeric series value for display (e.g. fmtPct, fmtNum). */
  formatter: (v: number) => string
}

export function ChartTooltip({
  active,
  payload,
  label,
  formatter,
}: ChartTooltipProps) {
  if (!active || !payload || payload.length === 0) return null

  return (
    <div className={styles.tooltip}>
      <div className={styles.tooltipLabel}>{label}</div>
      {payload.map((entry, i) => (
        <div key={entry.dataKey ?? entry.name ?? i} className={styles.tooltipRow}>
          <span
            className={styles.tooltipDot}
            style={{ backgroundColor: entry.color }}
          />
          <span className={styles.tooltipName}>{entry.name}</span>
          <span className={styles.tooltipValue}>
            {entry.value !== null && entry.value !== undefined
              ? formatter(entry.value)
              : '—'}
          </span>
        </div>
      ))}
    </div>
  )
}
