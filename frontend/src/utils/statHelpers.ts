import type { StatFamilyMeta } from '../types/stats'

/**
 * Distinct instrument codes across all families, in first-seen order.
 * Drives the global instrument selector — no hardcoded list.
 */
export function getAllInstruments(families: StatFamilyMeta[]): string[] {
  const seen = new Set<string>()
  const out: string[] = []
  for (const family of families) {
    for (const t of family.timeframes) {
      if (!seen.has(t.instrument)) {
        seen.add(t.instrument)
        out.push(t.instrument)
      }
    }
  }
  return out
}

/** True when the family has at least one timeframe for the given instrument. */
export function familyHasInstrument(
  family: StatFamilyMeta,
  instrument: string,
): boolean {
  return family.timeframes.some((t) => t.instrument === instrument)
}

/**
 * Distinct timeframe codes across a family, in first-seen order.
 * When `instrument` is given, only that instrument's timeframes are counted.
 */
export function getTimeframeLabels(
  family: StatFamilyMeta,
  instrument?: string,
): string[] {
  const timeframes = instrument
    ? family.timeframes.filter((t) => t.instrument === instrument)
    : family.timeframes
  return [...new Set(timeframes.map((t) => t.timeframe))]
}

/**
 * Sum of confirmed samples across a family's timeframes.
 * When `instrument` is given, only that instrument's timeframes are summed.
 */
export function getTotalSamples(
  family: StatFamilyMeta,
  instrument?: string,
): number {
  return family.timeframes
    .filter((t) => !instrument || t.instrument === instrument)
    .reduce((sum, t) => sum + t.total_samples, 0)
}

/**
 * Overall [min, max] date across a family, or null when no ranges exist.
 * When `instrument` is given, only that instrument's timeframes are considered.
 */
export function getDateRange(
  family: StatFamilyMeta,
  instrument?: string,
): [string, string] | null {
  const starts: string[] = []
  const ends: string[] = []
  for (const t of family.timeframes) {
    if (instrument && t.instrument !== instrument) continue
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
