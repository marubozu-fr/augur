import { apiFetch } from './client'
import type {
  StatFamilyDetail,
  StatFamilyMeta,
  TimeframeResult,
} from '../types/stats'

/**
 * GET /admin/stats
 * Returns all loaded stat families.
 */
export function listFamilies(): Promise<StatFamilyMeta[]> {
  return apiFetch<StatFamilyMeta[]>('/admin/stats')
}

/**
 * POST /admin/stats/reload
 * Reloads stat results from disk and returns the updated family list.
 */
export function reloadFamilies(): Promise<StatFamilyMeta[]> {
  return apiFetch<StatFamilyMeta[]>('/admin/stats/reload', { method: 'POST' })
}

/**
 * GET /admin/stats/:family
 * Returns the full stat run result for one family.
 * Throws on 404 (unknown family) with the envelope error message.
 */
export function getFamilyDetail(family: string): Promise<StatFamilyDetail> {
  return apiFetch<StatFamilyDetail>(`/admin/stats/${encodeURIComponent(family)}`)
}

/**
 * GET /admin/stats/:family/:instrument/:timeframe?start=&end=
 * Returns a single TimeframeResult, optionally reaggregated over a date range.
 *
 * This is the session-authenticated counterpart to the API-key-protected
 * /api/v1 timeframe endpoint — the backoffice cannot send an X-API-Key, so the
 * period filter fetches date-narrowed results here. `start`/`end` are inclusive
 * ISO dates (YYYY-MM-DD); omit both to receive the full stored result.
 * Throws on 400 (bad range / unfilterable family) or 404 with the envelope error.
 */
export function getTimeframeResult(
  family: string,
  instrument: string,
  timeframe: string,
  range: { start?: string; end?: string } = {},
): Promise<TimeframeResult> {
  const params = new URLSearchParams()
  if (range.start) params.set('start', range.start)
  if (range.end) params.set('end', range.end)
  const query = params.toString()
  const base = `/admin/stats/${encodeURIComponent(family)}/${encodeURIComponent(
    instrument,
  )}/${encodeURIComponent(timeframe)}`
  return apiFetch<TimeframeResult>(query ? `${base}?${query}` : base)
}
