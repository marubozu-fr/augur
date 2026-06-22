import { useState } from 'react'
import {
  Alert,
  Button,
  CopyButton,
  Modal,
  Skeleton,
  TextInput,
} from '@mantine/core'
import { useDisclosure } from '@mantine/hooks'
import { notifications } from '@mantine/notifications'
import {
  IconAlertCircle,
  IconCheck,
  IconCopy,
  IconKey,
  IconRefresh,
} from '@tabler/icons-react'
import { useApiKeys } from '../hooks/useApiKeys'
import type { ApiKey, ApiKeyCreated } from '../types/apiKeys'
import styles from './ApiKeysPage.module.css'

// ---------------------------------------------------------------------------
// Date formatting helper
// ---------------------------------------------------------------------------

const dateFormat = new Intl.DateTimeFormat('en-US', {
  year: 'numeric',
  month: 'short',
  day: 'numeric',
  hour: '2-digit',
  minute: '2-digit',
})

function fmtDate(iso: string): string {
  return dateFormat.format(new Date(iso))
}

// ---------------------------------------------------------------------------
// Revoke confirmation modal
// ---------------------------------------------------------------------------

interface RevokeModalProps {
  target: ApiKey | null
  opened: boolean
  onClose: () => void
  onConfirm: () => Promise<void>
}

function RevokeModal({ target, opened, onClose, onConfirm }: RevokeModalProps) {
  const [loading, setLoading] = useState(false)

  async function handleConfirm() {
    setLoading(true)
    try {
      await onConfirm()
      onClose()
    } finally {
      setLoading(false)
    }
  }

  return (
    <Modal
      opened={opened}
      onClose={onClose}
      title="Revoke API key"
      size="sm"
      centered
    >
      <p className={styles.modalBody}>
        Are you sure you want to revoke{' '}
        <strong className={styles.modalBodyStrong}>{target?.name}</strong>?
        This cannot be undone. Any service using this key will lose access
        immediately.
      </p>
      <div className={styles.modalActions}>
        <Button variant="subtle" color="gray" onClick={onClose} disabled={loading}>
          Cancel
        </Button>
        <Button color="red" loading={loading} onClick={() => void handleConfirm()}>
          Revoke key
        </Button>
      </div>
    </Modal>
  )
}

// ---------------------------------------------------------------------------
// New key callout (shown once after creation)
// ---------------------------------------------------------------------------

interface NewKeyCalloutProps {
  created: ApiKeyCreated
  onDismiss: () => void
}

function NewKeyCallout({ created, onDismiss }: NewKeyCalloutProps) {
  return (
    <div className={styles.keyCallout}>
      <p className={styles.keyCalloutTitle}>
        API key created — copy it now
      </p>
      <p className={styles.keyCalloutWarning}>
        This is the only time the key will be shown. Store it in a secure
        location. It cannot be retrieved later.
      </p>
      <div className={styles.keyDisplay}>
        <span className={styles.keyValue}>{created.key}</span>
        <CopyButton value={created.key} timeout={2000}>
          {({ copied, copy }) => (
            <Button
              size="xs"
              variant={copied ? 'filled' : 'light'}
              color={copied ? 'green' : 'blue'}
              leftSection={
                copied ? <IconCheck size={13} /> : <IconCopy size={13} />
              }
              onClick={copy}
            >
              {copied ? 'Copied' : 'Copy'}
            </Button>
          )}
        </CopyButton>
      </div>
      <div className={styles.keyCalloutDismiss}>
        <Button size="xs" variant="subtle" color="gray" onClick={onDismiss}>
          Dismiss
        </Button>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export function ApiKeysPage() {
  const { keys, loading, error, reload, create, revoke } = useApiKeys()

  // Newly created key to display once
  const [newKey, setNewKey] = useState<ApiKeyCreated | null>(null)

  // Revoke confirmation modal
  const [revokeOpened, { open: openRevoke, close: closeRevoke }] =
    useDisclosure(false)
  const [revokeTarget, setRevokeTarget] = useState<ApiKey | null>(null)

  // Creation form
  const [name, setName] = useState('')
  const [nameError, setNameError] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)

  async function handleCreate(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault()
    const trimmed = name.trim()
    if (trimmed.length === 0) {
      setNameError('Key name is required')
      return
    }
    setNameError(null)
    setCreating(true)
    try {
      const created = await create(trimmed)
      setNewKey(created)
      setName('')
      notifications.show({
        title: 'API key created',
        message: `"${created.name}" is now active.`,
        color: 'green',
      })
    } catch (err) {
      notifications.show({
        title: 'Failed to create API key',
        message: err instanceof Error ? err.message : 'Unknown error',
        color: 'red',
      })
    } finally {
      setCreating(false)
    }
  }

  function handleRevokeClick(key: ApiKey) {
    setRevokeTarget(key)
    openRevoke()
  }

  async function handleRevokeConfirm() {
    if (revokeTarget === null) return
    try {
      await revoke(revokeTarget.id)
      notifications.show({
        title: 'API key revoked',
        message: `"${revokeTarget.name}" has been revoked.`,
        color: 'blue',
      })
    } catch (err) {
      notifications.show({
        title: 'Failed to revoke API key',
        message: err instanceof Error ? err.message : 'Unknown error',
        color: 'red',
      })
      throw err // Let the modal handle the loading state
    }
  }

  return (
    <div className={styles.page}>
      {/* ===== Page head ===== */}
      <div className={styles.pageHead}>
        <div>
          <h1 className={styles.pageTitle}>API Keys</h1>
          <p className={styles.pageSubtitle}>
            Manage API keys for programmatic access to stat results.
          </p>
        </div>
      </div>

      {/* ===== Error ===== */}
      {!loading && error && (
        <Alert
          icon={<IconAlertCircle size={16} />}
          color="red"
          variant="light"
          title="Failed to load API keys"
          mb="lg"
        >
          <p className={styles.alertMessage}>{error}</p>
          <button
            type="button"
            className={styles.alertRetry}
            onClick={() => void reload()}
          >
            <IconRefresh size={14} />
            <span>Retry</span>
          </button>
        </Alert>
      )}

      {/* ===== Create form ===== */}
      <form
        className={styles.createForm}
        onSubmit={(e) => void handleCreate(e)}
      >
        <TextInput
          className={styles.createFormInput}
          label="Key name"
          placeholder="e.g. pine-script-indicator"
          description="A descriptive name to identify this key"
          value={name}
          onChange={(e) => setName(e.currentTarget.value)}
          error={nameError}
          disabled={creating}
        />
        <Button
          type="submit"
          loading={creating}
          leftSection={<IconKey size={14} />}
        >
          Create key
        </Button>
      </form>

      {/* ===== New key callout (shown once) ===== */}
      {newKey !== null && (
        <NewKeyCallout created={newKey} onDismiss={() => setNewKey(null)} />
      )}

      {/* ===== Keys table ===== */}
      <div className={styles.tableSection}>
        <div className={styles.tableSectionHead}>
          <h2 className={styles.tableSectionTitle}>
            {loading
              ? 'API keys'
              : `${keys.length} API key${keys.length !== 1 ? 's' : ''}`}
          </h2>
        </div>

        {/* Loading skeletons */}
        {loading && (
          <div className={styles.skeletonTable}>
            {[1, 2, 3].map((n) => (
              <Skeleton key={n} height={40} radius="md" />
            ))}
          </div>
        )}

        {/* Table */}
        {!loading && (
          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Role</th>
                  <th>Status</th>
                  <th>Created</th>
                  <th>Revoked</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {keys.length === 0 ? (
                  <tr className={styles.emptyRow}>
                    <td colSpan={6}>No API keys yet. Create one above.</td>
                  </tr>
                ) : (
                  keys.map((key) => (
                    <tr key={key.id}>
                      <td className={styles.cellName}>{key.name}</td>
                      <td className={styles.cellMono}>{key.role}</td>
                      <td>
                        <span
                          className={`${styles.statusBadge} ${
                            key.status === 'active'
                              ? styles.statusActive
                              : styles.statusRevoked
                          }`}
                        >
                          {key.status === 'active' ? 'Active' : 'Revoked'}
                        </span>
                      </td>
                      <td className={styles.cellMono}>
                        {fmtDate(key.created_at)}
                      </td>
                      <td className={styles.cellMono}>
                        {key.revoked_at ? fmtDate(key.revoked_at) : '—'}
                      </td>
                      <td>
                        {key.status === 'active' && (
                          <Button
                            size="xs"
                            variant="subtle"
                            color="red"
                            onClick={() => handleRevokeClick(key)}
                          >
                            Revoke
                          </Button>
                        )}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* ===== Revoke confirmation modal ===== */}
      <RevokeModal
        target={revokeTarget}
        opened={revokeOpened}
        onClose={closeRevoke}
        onConfirm={handleRevokeConfirm}
      />
    </div>
  )
}
