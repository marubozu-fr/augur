import { useLocation, useNavigate, useParams } from 'react-router-dom'
import { useShellContext } from '../hooks/useShellContext'
import styles from './InstrumentSelect.module.css'

/**
 * Gold-toned instrument selector shared by the dashboard toolbar and the stat
 * detail filter bar. The active instrument is URL-driven: changing it navigates
 * to the same path under the new instrument prefix, keeping the rest intact.
 *
 * This centralizes the prefix-swap navigation that previously lived in the app
 * shell header, so both pages stay in sync with a single source of truth.
 */
export function InstrumentSelect() {
  const { instruments, selectedInstrument } = useShellContext()
  const location = useLocation()
  const navigate = useNavigate()
  const { instrument: urlInstrument } = useParams<{ instrument?: string }>()

  if (instruments.length === 0) return null

  function handleChange(next: string) {
    if (next === selectedInstrument) return
    if (urlInstrument) {
      // Instrument-scoped page: swap the prefix, keep the rest of the path.
      const rest = location.pathname.slice(('/' + urlInstrument).length)
      navigate(`/${next}${rest}${location.search}`)
    } else {
      // Instrument-agnostic page: land on the new instrument's stats page.
      navigate(`/${next}/stats`)
    }
  }

  return (
    <span className={styles.instrumentSelect}>
      <span className={styles.icon} aria-hidden="true">
        &#9670;
      </span>
      <select
        aria-label="Select instrument"
        value={selectedInstrument}
        onChange={(e) => handleChange(e.currentTarget.value)}
        disabled={instruments.length === 1}
      >
        {instruments.map((code) => (
          <option key={code} value={code}>
            {code}
          </option>
        ))}
      </select>
      <span className={styles.caret} aria-hidden="true">
        &#9662;
      </span>
    </span>
  )
}
