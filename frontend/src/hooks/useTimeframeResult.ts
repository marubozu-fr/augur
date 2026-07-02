import { useEffect, useState } from 'react'
import { getTimeframeResult } from '../api/stats'
import type { TimeframeResult } from '../types/stats'

export interface UseTimeframeResult {
  result: TimeframeResult | null
  loading: boolean
  error: string | null
}

/**
 * Fetch a date-filtered TimeframeResult for the given family/instrument/timeframe.
 *
 * The hook is idle (no request, no loading) while `start` and `end` are both
 * null — in that "All" case the caller renders the pre-loaded full result. Once
 * a range is set, it fetches from the session-authenticated admin endpoint and
 * re-fetches whenever any argument changes. Stale responses are discarded.
 */
export function useTimeframeResult(
  family: string | undefined,
  instrument: string,
  timeframe: string,
  start: string | null,
  end: string | null,
): UseTimeframeResult {
  const canFetch = Boolean(
    family && instrument && timeframe && (start !== null || end !== null),
  )
  // Identifies the current request; null when idle. Distinct keys => refetch.
  const key = canFetch
    ? `${family}|${instrument}|${timeframe}|${start}|${end}`
    : null

  const [result, setResult] = useState<TimeframeResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  // Adjust state synchronously during render when the target changes — the
  // React-recommended alternative to a reset effect (mirrors useStatDetail),
  // so no stale frame is committed and the effect only sets state in callbacks.
  const [loadedKey, setLoadedKey] = useState(key)
  if (key !== loadedKey) {
    setLoadedKey(key)
    setResult(null)
    setError(null)
    setLoading(key !== null)
  }

  useEffect(() => {
    if (!canFetch || !family) return

    let cancelled = false

    getTimeframeResult(family, instrument, timeframe, {
      start: start ?? undefined,
      end: end ?? undefined,
    })
      .then((data) => {
        if (!cancelled) setResult(data)
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(
            err instanceof Error ? err.message : 'Failed to load filtered result',
          )
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [family, instrument, timeframe, start, end, canFetch])

  return { result, loading, error }
}
