import dayjs from 'dayjs'

/**
 * Period filter presets for the stat detail filter bar.
 *
 * `all` means "no date filter" — the full pre-computed result is used.
 * `custom` means a user-selected range from the date popover.
 * Every other preset is a duration measured back from the last date with
 * actual data (`data_range[1]` of the current TimeframeResult), NOT from today,
 * so the window always lands on real samples.
 */
export type PeriodPreset =
  | 'all'
  | '1m'
  | '3m'
  | '6m'
  | 'ytd'
  | '1yr'
  | '2yr'
  | '3yr'
  | '5yr'
  | '10yr'
  | 'custom'

export interface DateRange {
  start: string // inclusive ISO date "YYYY-MM-DD"
  end: string // inclusive ISO date "YYYY-MM-DD"
}

/** Ordered duration presets rendered as chips (excludes `all` and `custom`). */
export const DURATION_PRESETS: readonly PeriodPreset[] = [
  '1m',
  '3m',
  '6m',
  'ytd',
  '1yr',
  '2yr',
  '3yr',
  '5yr',
  '10yr',
] as const

/** Human-readable chip label for each preset. */
export const PRESET_LABELS: Record<PeriodPreset, string> = {
  all: 'All',
  '1m': '1m',
  '3m': '3m',
  '6m': '6m',
  ytd: 'YTD',
  '1yr': '1yr',
  '2yr': '2yr',
  '3yr': '3yr',
  '5yr': '5yr',
  '10yr': '10yr',
  custom: 'Custom',
}

/**
 * Compute the inclusive [start, end] range for a duration preset, anchored to
 * `dataEnd` (the last date with data). Returns null for `all`, `custom`, or
 * when `dataEnd` is missing — those cases don't derive a range from a preset.
 *
 * The computed start may fall before the dataset's first date; the caller sends
 * it anyway, and the backend clamps to the samples actually available.
 */
export function computePresetRange(
  preset: PeriodPreset,
  dataEnd: string | undefined,
): DateRange | null {
  if (!dataEnd || preset === 'all' || preset === 'custom') return null
  const end = dayjs(dataEnd)
  if (!end.isValid()) return null

  let start: dayjs.Dayjs
  switch (preset) {
    case '1m':
      start = end.subtract(1, 'month')
      break
    case '3m':
      start = end.subtract(3, 'month')
      break
    case '6m':
      start = end.subtract(6, 'month')
      break
    case 'ytd':
      start = end.startOf('year')
      break
    case '1yr':
      start = end.subtract(1, 'year')
      break
    case '2yr':
      start = end.subtract(2, 'year')
      break
    case '3yr':
      start = end.subtract(3, 'year')
      break
    case '5yr':
      start = end.subtract(5, 'year')
      break
    case '10yr':
      start = end.subtract(10, 'year')
      break
    default:
      return null
  }

  return { start: start.format('YYYY-MM-DD'), end: end.format('YYYY-MM-DD') }
}
