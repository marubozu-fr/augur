/**
 * TypeScript interfaces mirroring the backend SystemStatus payload
 * returned by GET /admin/status.
 */

export interface LoadedStatFile {
  family: string
  filename: string
  size_bytes: number
  modified_at: string
}

export interface DataRange {
  start: string
  end: string
}

export interface DataFileInfo {
  filename: string
  size_bytes: number
  num_rows: number | null
  data_range: DataRange | null
}

export interface Versions {
  python: string
  fastapi: string
}

export interface SystemStatus {
  started_at: string
  uptime_seconds: number
  versions: Versions
  stats_files: LoadedStatFile[]
  data_files: DataFileInfo[]
}
