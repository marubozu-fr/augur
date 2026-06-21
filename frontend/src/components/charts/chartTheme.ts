/**
 * Recharts color/dimension constants.
 *
 * Recharts requires concrete hex values — it cannot read CSS custom properties.
 * Every value here maps 1-to-1 to a token in index.css so if the palette ever
 * changes this is the single place to update.
 */

// Surfaces — maps to --color-surface, --color-surface-2, --color-surface-3
export const CHART_BG = '#111722'
export const CHART_TOOLTIP_BG = '#161d2b'

// Borders — maps to --color-border, --color-border-strong
export const CHART_GRID_COLOR = '#232c3d'
export const CHART_TOOLTIP_BORDER = '#324054'

// Text — maps to --color-text, --color-text-muted, --color-text-dim
export const CHART_TEXT = '#e6ebf2'
export const CHART_TEXT_MUTED = '#9aa7bd'
export const CHART_TEXT_DIM = '#66738c'

// Semantics
export const CHART_POSITIVE = '#2ecc8f'  // --color-positive
export const CHART_NEGATIVE = '#ff5c6c'  // --color-negative
export const CHART_NEUTRAL = '#9aa7bd'   // --color-neutral (baseline series)
export const CHART_ACCENT = '#4c8dff'    // --color-accent

// Recharts shared defaults
export const CHART_FONT_SIZE = 11
export const CHART_FONT_FAMILY =
  '"JetBrains Mono", "SF Mono", "Roboto Mono", Menlo, Consolas, monospace'
