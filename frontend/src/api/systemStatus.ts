import { apiFetch } from './client'
import type { SystemStatus } from '../types/systemStatus'

/**
 * GET /admin/status
 * Returns loaded stats files, data files, versions, and uptime.
 */
export function getSystemStatus(): Promise<SystemStatus> {
  return apiFetch<SystemStatus>('/admin/status')
}
