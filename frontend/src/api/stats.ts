import { apiFetch } from './client'
import type { StatFamilyMeta } from '../types/stats'

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
