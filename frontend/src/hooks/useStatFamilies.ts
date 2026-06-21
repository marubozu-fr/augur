import { useCallback, useEffect, useState } from 'react'
import { listFamilies, reloadFamilies } from '../api/stats'
import type { StatFamilyMeta } from '../types/stats'

interface UseStatFamiliesResult {
  families: StatFamilyMeta[]
  loading: boolean
  error: string | null
  reload: () => Promise<void>
}

export function useStatFamilies(): UseStatFamiliesResult {
  const [families, setFamilies] = useState<StatFamilyMeta[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    listFamilies()
      .then((data) => {
        if (!cancelled) {
          setFamilies(data)
          setLoading(false)
          setError(null)
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(
            err instanceof Error ? err.message : 'Failed to load stat families',
          )
          setLoading(false)
        }
      })

    return () => {
      cancelled = true
    }
  }, [])

  const reload = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await reloadFamilies()
      setFamilies(data)
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Failed to reload stat families',
      )
    } finally {
      setLoading(false)
    }
  }, [])

  return { families, loading, error, reload }
}
