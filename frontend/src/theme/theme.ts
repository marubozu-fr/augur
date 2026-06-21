import { createTheme, rem } from '@mantine/core'

// Design tokens sourced from docs/mockups/styles.css
// Dark theme first — this is a quant/trading tool.

export const theme = createTheme({
  // Typography
  fontFamily:
    'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
  fontFamilyMonospace:
    '"JetBrains Mono", "SF Mono", "Roboto Mono", Menlo, Consolas, monospace',

  // Font sizes mapped from --fs-* tokens (base 4px scale)
  fontSizes: {
    xs: rem(11), // --fs-xs
    sm: rem(12), // --fs-sm
    md: rem(13), // --fs-md
    lg: rem(15), // --fs-lg
    xl: rem(18), // --fs-xl
  },

  // Border radii mapped from --radius-* tokens
  radius: {
    xs: rem(4), // --radius-sm
    sm: rem(4), // --radius-sm
    md: rem(6), // --radius-md
    lg: rem(10), // --radius-lg
    xl: rem(10), // --radius-lg
  },

  // Spacing (4px base) mapped from --space-* tokens
  spacing: {
    xs: rem(4), // --space-1
    sm: rem(8), // --space-2
    md: rem(12), // --space-3
    lg: rem(16), // --space-4
    xl: rem(24), // --space-6
  },

  // Accent color mapped to Mantine's blue palette slot
  primaryColor: 'blue',
  primaryShade: { dark: 5 },

  colors: {
    // Custom blue mapped from --color-accent: #4c8dff
    blue: [
      '#e8f0ff', // 0
      '#c5d8ff', // 1
      '#99baff', // 2
      '#6aa0ff', // 3 --color-accent-hover
      '#4c8dff', // 4 --color-accent (primary)
      '#4c8dff', // 5 primary shade
      '#2e6bd6', // 6
      '#1a55c0', // 7
      '#0d3fa8', // 8
      '#052e8f', // 9
    ],
    // Positive / green  --color-positive: #2ecc8f
    green: [
      '#e9faf3', // 0
      '#c4f0dc', // 1
      '#97e4c3', // 2
      '#63d8a8', // 3
      '#2ecc8f', // 4 --color-positive
      '#2ecc8f', // 5
      '#1fa870', // 6
      '#128551', // 7
      '#086234', // 8
      '#02401c', // 9
    ],
    // Negative / red  --color-negative: #ff5c6c
    red: [
      '#fff0f1', // 0
      '#ffd6d9', // 1
      '#ffb3b8', // 2
      '#ff8a93', // 3
      '#ff5c6c', // 4 --color-negative
      '#ff5c6c', // 5
      '#e03a4a', // 6
      '#c01e2e', // 7
      '#9e0a19', // 8
      '#7a0009', // 9
    ],
    // Dark palette for surfaces
    dark: [
      '#e6ebf2', // 0  --color-text
      '#9aa7bd', // 1  --color-text-muted
      '#66738c', // 2  --color-text-dim
      '#324054', // 3  --color-border-strong
      '#232c3d', // 4  --color-border
      '#1d2738', // 5  --color-surface-3
      '#161d2b', // 6  --color-surface-2
      '#111722', // 7  --color-surface
      '#0d1119', // 8  --color-sidebar / --color-header
      '#0b0e14', // 9  --color-bg (darkest)
    ],
  },

  // Component overrides to match the design system
  components: {
    AppShell: {
      styles: {
        navbar: {
          backgroundColor: '#0d1119', // --color-sidebar
          borderRight: '1px solid #232c3d', // --color-border
        },
        header: {
          backgroundColor: '#0d1119', // --color-header
          borderBottom: '1px solid #232c3d', // --color-border
        },
        main: {
          backgroundColor: '#0b0e14', // --color-bg
        },
      },
    },
  },

  // Disable focus ring on mouse click — keep it only for keyboard nav
  focusRing: 'auto',
})
