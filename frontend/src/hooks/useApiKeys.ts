import { useCallback, useEffect, useState } from 'react'
import { listApiKeys, createApiKey, revokeApiKey } from '../api/apiKeys'
import type { ApiKey, ApiKeyCreated } from '../types/apiKeys'

export interface UseApiKeysResult {
  keys: ApiKey[]
  loading: boolean
  error: string | null
  reload: () => Promise<void>
  create: (name: string) => Promise<ApiKeyCreated>
  revoke: (id: number) => Promise<void>
}

export function useApiKeys(): UseApiKeysResult {
  const [keys, setKeys] = useState<ApiKey[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    listApiKeys()
      .then((data) => {
        if (!cancelled) {
          setKeys(data)
          setLoading(false)
          setError(null)
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(
            err instanceof Error ? err.message : 'Failed to load API keys',
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
      const data = await listApiKeys()
      setKeys(data)
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Failed to reload API keys',
      )
    } finally {
      setLoading(false)
    }
  }, [])

  const create = useCallback(async (name: string): Promise<ApiKeyCreated> => {
    const created = await createApiKey(name)
    // Refresh the list so the new key appears in the table.
    await reload()
    return created
  }, [reload])

  const revoke = useCallback(
    async (id: number): Promise<void> => {
      await revokeApiKey(id)
      // Refresh the list to reflect the revoked status.
      await reload()
    },
    [reload],
  )

  return { keys, loading, error, reload, create, revoke }
}
