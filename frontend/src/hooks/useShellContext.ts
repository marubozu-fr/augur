import { useOutletContext } from 'react-router-dom'
import type { UseStatFamiliesResult } from './useStatFamilies'

/**
 * Context shared by the app shell with every routed page. Carries the loaded
 * stat families plus the instrument derived from the current URL, so dashboard
 * and stat detail pages stay in sync with the header instrument selector.
 *
 * The selected instrument is URL-driven: the Select dropdown in ShellPage
 * navigates to the new instrument prefix rather than updating local state.
 */
export interface ShellContext extends UseStatFamiliesResult {
  /** Distinct instruments across all loaded families, in first-seen order. */
  instruments: string[]
  /** Currently active instrument (always one of `instruments`, or '' when none). */
  selectedInstrument: string
}

export function useShellContext(): ShellContext {
  return useOutletContext<ShellContext>()
}
