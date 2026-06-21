import type { StatFamilyMeta } from '../types/stats'

/** Distinct instrument codes across a family's timeframes, in first-seen order. */
export function getInstruments(family: StatFamilyMeta): string[] {
  return [...new Set(family.timeframes.map((t) => t.instrument))]
}

/** Distinct timeframe codes across a family, in first-seen order. */
export function getTimeframeLabels(family: StatFamilyMeta): string[] {
  return [...new Set(family.timeframes.map((t) => t.timeframe))]
}

/** Sum of confirmed samples across a family's timeframes. */
export function getTotalSamples(family: StatFamilyMeta): number {
  return family.timeframes.reduce((sum, t) => sum + t.total_samples, 0)
}

/** Overall [min, max] date across a family, or null when no ranges exist. */
export function getDateRange(family: StatFamilyMeta): [string, string] | null {
  const starts: string[] = []
  const ends: string[] = []
  for (const t of family.timeframes) {
    const [start, end] = t.data_range
    if (start) starts.push(start)
    if (end) ends.push(end)
  }
  if (starts.length === 0 || ends.length === 0) return null
  // ISO "YYYY-MM-DD" strings sort lexicographically by date.
  return [
    starts.reduce((a, b) => (a < b ? a : b)),
    ends.reduce((a, b) => (a > b ? a : b)),
  ]
}
