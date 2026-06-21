export interface I18nString {
  en: string
  fr: string
}

export interface TimeframeMeta {
  instrument: string
  timeframe: string
  data_range: string[] // [minDate, maxDate] as "YYYY-MM-DD"
  total_samples: number
}

export interface StatFamilyMeta {
  family: string // slug — used as route param and React key
  title: I18nString
  definition: I18nString
  timeframes: TimeframeMeta[]
}

// ---------------------------------------------------------------------------
// Stat detail — mirrors stats/base.py Pydantic models
// ---------------------------------------------------------------------------

export interface StatResultRow {
  condition: string
  outcome: string
  count: number
  total: number
  probability: number
  baseline_prob: number
  baseline_n: number
  value: number | null
  value_baseline: number | null
}

export interface SliceGroupResult {
  label: I18nString
  total_samples: number
  results: StatResultRow[]
}

export interface SliceResult {
  dimension: string
  groups: Record<string, SliceGroupResult>
}

export interface TimeframeResult {
  data_range: string[]
  total_samples: number
  results: StatResultRow[]
  slices: Record<string, SliceResult>
}

export interface Labels {
  conditions: Record<string, I18nString>
  outcomes: Record<string, I18nString>
  dimensions: Record<string, I18nString>
}

export interface StatRunResult {
  stat_name: string
  title: I18nString
  definition: I18nString
  labels: Labels
  instruments: Record<string, Record<string, TimeframeResult>>
}

export interface StatFamilyDetail {
  computed_at: string | null
  result: StatRunResult
}
