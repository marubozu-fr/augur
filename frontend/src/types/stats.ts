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

// ---------------------------------------------------------------------------
// Tradable layer — mirrors stats/base.py TradableResult (v2+ stats).
// Wire format is snake_case; do NOT camelCase-transform these fields.
// ---------------------------------------------------------------------------

export interface TpLadderLevel {
  level_x_range: number // multiplier (0.25, 0.5, ...)
  level_pts: number // absolute level in points
  prob: number // P(MFE >= level)
  n: number
}

export interface ConditionalMae {
  level_pts: number
  mae_pct_p50: number // fraction of anchor
  mae_pct_p75: number
}

export interface MaeResult {
  p50: number // points
  p75: number
  p90: number
  conditional: ConditionalMae[]
}

export interface TradableMeta {
  anchor: string
  outcome_window: string
  overlap_free: boolean
  excluded_days: string[]
}

export interface TradableResult {
  win_rate: number
  remaining_mean_pts: number
  remaining_median_pts: number
  remaining_mean_pct: number
  n: number
  tp_ladder: TpLadderLevel[]
  mae: MaeResult
  meta: TradableMeta
}

export interface TimeframeResult {
  data_range: string[]
  total_samples: number
  results: StatResultRow[]
  slices: Record<string, SliceResult>
  // Tradable layer keyed by condition (v2+ stats). Null/absent for
  // probability-only families and for period-filtered results (backend
  // reaggregation of the tradable layer is not implemented yet).
  tradable?: Record<string, TradableResult> | null
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
