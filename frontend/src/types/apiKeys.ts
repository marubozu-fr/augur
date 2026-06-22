/**
 * TypeScript interfaces for API key management.
 * Field names match the backend snake_case responses directly — no mapping layer.
 */

export interface ApiKey {
  id: number
  name: string
  role: string
  created_at: string
  revoked_at: string | null
  status: 'active' | 'revoked'
}

/** Returned only on creation — contains the plaintext key (shown once). */
export interface ApiKeyCreated {
  id: number
  name: string
  role: string
  created_at: string
  key: string
}
