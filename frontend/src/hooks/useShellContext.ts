import { useOutletContext } from 'react-router-dom'
import type { UseStatFamiliesResult } from './useStatFamilies'

/**
 * Context shared by the app shell with every routed page. Carries the loaded
 * stat families plus the globally-selected instrument, so the dashboard and
 * stat detail pages stay in sync with the header instrument selector.
 */
export interface ShellContext extends UseStatFamiliesResult {
  /** Distinct instruments across all loaded families, in first-seen order. */
  instruments: string[]
  /** Currently active instrument (always one of `instruments`, or '' when none). */
  selectedInstrument: string
  /** Switch the active instrument. */
  setSelectedInstrument: (instrument: string) => void
}

export function useShellContext(): ShellContext {
  return useOutletContext<ShellContext>()
}
