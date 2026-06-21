import { apiFetch } from './client'
import type { StatFamilyDetail, StatFamilyMeta } from '../types/stats'

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
