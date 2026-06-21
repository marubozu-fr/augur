import { useCallback, useEffect, useState } from 'react'
import { getFamilyDetail } from '../api/stats'
import type { StatFamilyDetail } from '../types/stats'

export interface UseStatDetailResult {
  detail: StatFamilyDetail | null
  loading: boolean
  error: string | null
  reload: () => Promise<void>
}

export function useStatDetail(family: string | undefined): UseStatDetailResult {
  const [detail, setDetail] = useState<StatFamilyDetail | null>(null)
  const [loading, setLoading] = useState(family !== undefined)
  const [error, setError] = useState<string | null>(
    family === undefined ? 'No family specified.' : null,
  )
  // The family the current detail/error/loading belongs to. When `family`
  // changes we reset synchronously *during render* — the React-recommended way
  // to adjust state on a prop change — so no stale detail frame is ever
  // committed, and the effect itself never sets state synchronously.
  const [loadedFamily, setLoadedFamily] = useState(family)
  if (family !== loadedFamily) {
    setLoadedFamily(family)
    setDetail(null)
    setError(family === undefined ? 'No family specified.' : null)
    setLoading(family !== undefined)
  }

  useEffect(() => {
    if (!family) return

    let cancelled = false

    getFamilyDetail(family)
      .then((data) => {
        if (!cancelled) setDetail(data)
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(
            err instanceof Error ? err.message : 'Failed to load stat detail',
          )
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [family])

  // Manual retry — runs from an event handler, so setting state here is fine.
  const reload = useCallback(async () => {
    if (!family) return
    setLoading(true)
    setError(null)
    setDetail(null)
    try {
      const data = await getFamilyDetail(family)
      setDetail(data)
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Failed to load stat detail',
      )
    } finally {
      setLoading(false)
    }
  }, [family])

  return { detail, loading, error, reload }
}
