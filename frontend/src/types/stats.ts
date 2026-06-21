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
