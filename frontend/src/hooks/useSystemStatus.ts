import { useCallback, useEffect, useState } from 'react'
import { getSystemStatus } from '../api/systemStatus'
import { reloadFamilies } from '../api/stats'
import type { SystemStatus } from '../types/systemStatus'

export interface UseSystemStatusResult {
  status: SystemStatus | null
  loading: boolean
  error: string | null
  /** Re-fetch the status payload without reloading the stats cache. */
  refresh: () => Promise<void>
  /** Re-scan results/ via POST /admin/stats/reload, then refresh status. Admin-only. */
  reloadStats: () => Promise<void>
}

export function useSystemStatus(): UseSystemStatusResult {
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    getSystemStatus()
      .then((data) => {
        if (!cancelled) {
          setStatus(data)
          setError(null)
          setLoading(false)
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(
            err instanceof Error ? err.message : 'Failed to load system status',
          )
          setLoading(false)
        }
      })

    return () => {
      cancelled = true
    }
  }, [])

  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await getSystemStatus()
      setStatus(data)
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Failed to refresh system status',
      )
    } finally {
      setLoading(false)
    }
  }, [])

  const reloadStats = useCallback(async () => {
    // Reload errors propagate to the caller so the page can show a
    // notification. They do NOT touch the hook's error state — that stays
    // reserved for initial-load / refresh failures so the inline alert
    // ("Failed to load system status") doesn't mislabel a reload failure.
    setLoading(true)
    try {
      await reloadFamilies()
      const data = await getSystemStatus()
      setStatus(data)
    } finally {
      setLoading(false)
    }
  }, [])

  return { status, loading, error, refresh, reloadStats }
}
