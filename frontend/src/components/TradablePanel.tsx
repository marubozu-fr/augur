import type {
  ConditionalMae,
  Labels,
  TpLadderLevel,
  TradableResult,
} from '../types/stats'
import styles from './TradablePanel.module.css'

// ---------------------------------------------------------------------------
// Number formatting — mirrors the spec in issue #197.
// ---------------------------------------------------------------------------

const pct1 = new Intl.NumberFormat('en-US', {
  style: 'percent',
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
})

const pct2 = new Intl.NumberFormat('en-US', {
  style: 'percent',
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
})

const intFmt = new Intl.NumberFormat('en-US')

/** Probability / win rate / hit rate → 1 decimal percent (61.8%). */
function fmtProb(v: number): string {
  return pct1.format(v)
}

/** Fraction of anchor → 2 decimal percent (0.30%). */
function fmtAnchorPct(v: number): string {
  return pct2.format(v)
}

/** Signed fraction of anchor → 2 decimal percent (+0.25%). */
function fmtSignedAnchorPct(v: number): string {
  const sign = v > 0 ? '+' : ''
  return `${sign}${pct2.format(v)}`
}

/** Points: 0 decimals at |v| >= 10, 1 decimal below (103 pt, 5.7 pt). */
function fmtPts(v: number): string {
  const decimals = Math.abs(v) >= 10 ? 0 : 1
  return `${v.toFixed(decimals)} pt`
}

/** Signed points (+42.5 pt, -12.3 pt). */
function fmtSignedPts(v: number): string {
  const sign = v > 0 ? '+' : ''
  return `${sign}${fmtPts(v)}`
}

/** Sample size → comma-separated (2,287). */
function fmtN(v: number): string {
  return intFmt.format(v)
}

/** Multiplier → "0.25×". */
function fmtMult(v: number): string {
  return `${v}×`
}

// ---------------------------------------------------------------------------
// Edge tint for probability cells — green > 0.55, red < 0.45, neutral between.
// ---------------------------------------------------------------------------

function probTintClass(prob: number): string {
  if (prob > 0.55) return styles.tintPositive ?? ''
  if (prob < 0.45) return styles.tintNegative ?? ''
  return styles.tintNeutral ?? ''
}

/** Edge tint for a signed value — by sign, with a small neutral band. */
function signTintClass(v: number): string {
  if (Math.abs(v) < 1e-9) return styles.tintNeutral ?? ''
  return (v > 0 ? styles.tintPositive : styles.tintNegative) ?? ''
}

// ---------------------------------------------------------------------------
// Condition tone — maps a condition key to an up/down/neutral color.
// Derived from the semantic key (green/red, up/down, bull/bear), not from any
// hardcoded instrument or ordering.
// ---------------------------------------------------------------------------

type Tone = 'up' | 'down' | 'neutral'

function conditionTone(key: string): Tone {
  const k = key.toLowerCase()
  if (k.includes('green') || k.includes('up') || k.includes('bull')) return 'up'
  if (k.includes('red') || k.includes('down') || k.includes('bear'))
    return 'down'
  return 'neutral'
}

function toneClass(tone: Tone): string {
  if (tone === 'up') return styles.toneUp ?? ''
  if (tone === 'down') return styles.toneDown ?? ''
  return styles.toneNeutral ?? ''
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface TradablePanelProps {
  tradable: Record<string, TradableResult>
  labels: Labels
}

interface ConditionColumn {
  key: string
  label: string
  tone: Tone
  data: TradableResult
}

export function TradablePanel({ tradable, labels }: TradablePanelProps) {
  const columns: ConditionColumn[] = Object.entries(tradable).map(
    ([key, data]) => ({
      key,
      label: labels.conditions[key]?.en ?? key,
      tone: conditionTone(key),
      data,
    }),
  )

  const first = columns[0]
  if (!first) return null

  // The meta block is shared across conditions (same anchor / window). Use the
  // first condition's meta for the subtitle.
  const meta = first.data.meta
  const subtitle = [
    meta.outcome_window,
    meta.overlap_free ? 'overlap-free' : null,
    `${meta.excluded_days.length} excluded days`,
  ]
    .filter(Boolean)
    .join(' · ')

  // Column grid: one column per condition. Single condition centers itself.
  const gridStyle = { gridTemplateColumns: `repeat(${columns.length}, 1fr)` }
  const single = columns.length === 1

  return (
    <div className={styles.panel}>
      <div className={styles.panelHead}>
        <span className={styles.statTypeTradable}>Tradable</span>
        <h2 className={styles.panelTitle}>Tradable Layer</h2>
      </div>
      <p className={styles.subtitle}>{subtitle}</p>

      {/* ===== Residual ===== */}
      <section className={styles.block}>
        <div className={styles.blockHead}>Residual</div>
        <div
          className={`${styles.condGrid} ${single ? styles.condGridSingle : ''}`}
          style={gridStyle}
        >
          {columns.map((col) => (
            <div key={col.key} className={styles.condCol}>
              <div className={`${styles.condHeader} ${toneClass(col.tone)}`}>
                {col.label}
              </div>
              <dl className={styles.statList}>
                <div className={styles.statRow}>
                  <dt>Win rate</dt>
                  <dd
                    className={`${styles.mono} ${probTintClass(col.data.win_rate)}`}
                  >
                    {fmtProb(col.data.win_rate)}
                  </dd>
                </div>
                <div className={styles.statRow}>
                  <dt>Mean</dt>
                  <dd
                    className={`${styles.mono} ${signTintClass(col.data.remaining_mean_pts)}`}
                  >
                    {fmtSignedPts(col.data.remaining_mean_pts)}
                  </dd>
                </div>
                <div className={styles.statRow}>
                  <dt>Median</dt>
                  <dd
                    className={`${styles.mono} ${signTintClass(col.data.remaining_median_pts)}`}
                  >
                    {fmtSignedPts(col.data.remaining_median_pts)}
                  </dd>
                </div>
                <div className={styles.statRow}>
                  <dt>Mean %</dt>
                  <dd
                    className={`${styles.mono} ${signTintClass(col.data.remaining_mean_pct)}`}
                  >
                    {fmtSignedAnchorPct(col.data.remaining_mean_pct)}
                  </dd>
                </div>
                <div className={styles.statRow}>
                  <dt>N</dt>
                  <dd className={styles.mono}>{fmtN(col.data.n)}</dd>
                </div>
              </dl>
            </div>
          ))}
        </div>
      </section>

      {/* ===== TP Ladder (MFE) ===== */}
      <TpLadderTable columns={columns} />

      {/* ===== Drawdown (MAE) ===== */}
      <section className={styles.block}>
        <div className={styles.blockHead}>Drawdown (MAE)</div>
        <div
          className={`${styles.condGrid} ${single ? styles.condGridSingle : ''}`}
          style={gridStyle}
        >
          {columns.map((col) => (
            <div key={col.key} className={styles.condCol}>
              <div className={`${styles.condHeader} ${toneClass(col.tone)}`}>
                {col.label}
              </div>
              <dl className={styles.statList}>
                <div className={styles.statRow}>
                  <dt>p50</dt>
                  <dd className={styles.mono}>{fmtPts(col.data.mae.p50)}</dd>
                </div>
                <div className={styles.statRow}>
                  <dt>p75</dt>
                  <dd className={styles.mono}>{fmtPts(col.data.mae.p75)}</dd>
                </div>
                <div className={styles.statRow}>
                  <dt>p90</dt>
                  <dd className={styles.mono}>{fmtPts(col.data.mae.p90)}</dd>
                </div>
              </dl>
            </div>
          ))}
        </div>
        <ConditionalMaeTable columns={columns} />
      </section>
    </div>
  )
}

// ---------------------------------------------------------------------------
// TP Ladder — one row per ladder level, hit rate per condition side by side.
// Rows are aligned by ladder index (shared level_x_range ordering). The PTS
// column shows the reference (first) condition's level; per-condition points
// differ marginally since each ladder is scaled to its own median range.
// ---------------------------------------------------------------------------

function TpLadderTable({ columns }: { columns: ConditionColumn[] }) {
  const rowCount = Math.max(...columns.map((c) => c.data.tp_ladder.length))
  const ref = columns[0]?.data.tp_ladder ?? []

  const levelAt = (i: number): TpLadderLevel | undefined => {
    for (const col of columns) {
      const lvl = col.data.tp_ladder[i]
      if (lvl) return lvl
    }
    return undefined
  }

  return (
    <section className={styles.block}>
      <div className={styles.blockHead}>TP Ladder (MFE)</div>
      <div className={styles.tableWrap}>
        <table className={styles.dataTable}>
          <thead>
            <tr>
              <th>Level</th>
              <th className={styles.colNum}>Pts</th>
              {columns.map((col) => (
                <th
                  key={col.key}
                  className={`${styles.colNum} ${toneClass(col.tone)}`}
                >
                  {col.label} hit rate
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Array.from({ length: rowCount }, (_, i) => {
              const level = levelAt(i)
              const pts = ref[i]?.level_pts ?? level?.level_pts
              return (
                <tr key={i}>
                  <td className={styles.mono}>
                    {level ? fmtMult(level.level_x_range) : '—'}
                  </td>
                  <td className={`${styles.cellNum}`}>
                    {pts !== undefined ? fmtPts(pts) : '—'}
                  </td>
                  {columns.map((col) => {
                    const lvl = col.data.tp_ladder[i]
                    return (
                      <td
                        key={col.key}
                        className={`${styles.cellNum} ${lvl ? probTintClass(lvl.prob) : ''}`}
                      >
                        {lvl ? fmtProb(lvl.prob) : '—'}
                      </td>
                    )
                  })}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </section>
  )
}

// ---------------------------------------------------------------------------
// Conditional MAE — drawdown before first TP touch, % of anchor. p50/p75 per
// condition side by side, aligned by index. Values are fractions of the anchor
// price, formatted as percentages with 2 decimals.
// ---------------------------------------------------------------------------

function ConditionalMaeTable({ columns }: { columns: ConditionColumn[] }) {
  const rowCount = Math.max(
    ...columns.map((c) => c.data.mae.conditional.length),
  )
  const ref = columns[0]?.data.mae.conditional ?? []

  const entryAt = (i: number): ConditionalMae | undefined => {
    for (const col of columns) {
      const e = col.data.mae.conditional[i]
      if (e) return e
    }
    return undefined
  }

  if (rowCount === 0) return null

  return (
    <div className={styles.tableWrap}>
      <div className={styles.subBlockHead}>
        Conditional MAE (before first TP touch)
      </div>
      <table className={styles.dataTable}>
        <thead>
          <tr>
            <th>TP level</th>
            {columns.map((col) => (
              <th
                key={col.key}
                className={`${styles.colNum} ${toneClass(col.tone)}`}
              >
                {col.label} p50 / p75
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Array.from({ length: rowCount }, (_, i) => {
            const pts = ref[i]?.level_pts ?? entryAt(i)?.level_pts
            return (
              <tr key={i}>
                <td className={styles.cellNum}>
                  {pts !== undefined ? fmtPts(pts) : '—'}
                </td>
                {columns.map((col) => {
                  const e = col.data.mae.conditional[i]
                  return (
                    <td key={col.key} className={styles.cellNum}>
                      {e
                        ? `${fmtAnchorPct(e.mae_pct_p50)} / ${fmtAnchorPct(e.mae_pct_p75)}`
                        : '—'}
                    </td>
                  )
                })}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
