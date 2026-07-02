# Augur Backoffice — Mockups (Issues #116 dashboard, #117 stat detail)

Static HTML/CSS visual references for the future React frontend
(React 18 + Mantine v7 + CSS Modules + Recharts). These are **design
references only** — no JS frameworks, no production code. The frontend agent
should map the CSS custom properties in `styles.css` to the Mantine theme and
rebuild these layouts as React components + CSS Modules.

This is the first backoffice step; there is no `docs/DESIGN_SYSTEM.md` yet.
The token system established here (in `styles.css`) is the starting point for it.

## Files

| File                     | What it shows |
| ------------------------ | ------------- |
| `index.html`             | The dashboard in the app shell (sidebar + header + main), **populated state**: metric strip, a search / category-filter / **instrument-selector** toolbar, and a 16-card grid (one card per real stat family). |
| `dashboard-states.html`  | The dashboard **loading** state (skeleton cards + spinner) and two **empty** states (no results at all, and no search match), as clearly labeled sections on one page. |
| `stat-detail.html`       | The **stat detail page** (Issue #117): page header + metadata strip, a unified **filter bar** (instrument selector · timeframe · period presets · custom date-range popover · charts toggle), documentation panel, a **probability** results table (Opening Candle Continuation, NQ 15min) and a visually distinct **magnitude** results block (Opening Range Indicator, NQ 15min), plus collapsible `<details>` slice sections by weekday / close color / size bucket. |
| `stat-detail.css`        | Page-specific styles for `stat-detail.html`. Extends `styles.css` (reuses every token); adds the panel surface, data tables, probability bars, magnitude value blocks, and collapsible slice groups. |
| `styles.css`             | Shared stylesheet. All design tokens are CSS custom properties (`:root`) — colors, typography, spacing scale, radii, elevation, layout dimensions. |
| `README.md`              | This file. |

## How to view

Open any `*.html` file directly in a browser. `index.html` and
`dashboard-states.html` link to `styles.css`; `stat-detail.html` links to both
`styles.css` and `stat-detail.css`. No build step.

## Interactions demonstrated without JS

- **Sidebar collapse**: checkbox hack (`#sidebar-toggle` + `☰` label in the
  header) collapses the sidebar to an icon rail. Also collapses automatically
  below 900px via media query.
- **Collapsible slices**: the stat detail page uses native `<details>` /
  `<summary>` elements for slice dimensions — expand/collapse with zero JS.
- **Filter bar & date picker**: the stat detail filter bar has an instrument
  `<select>`, timeframe/period segmented chips (shared `.filter-group` /
  `.filter-chip`), a **Charts** switch, and a **custom date-range popover**
  built with `<details>` / `<summary>` — two side-by-side month calendars with a
  highlighted start→end range, shown open so the calendar design is reviewable.
  The **instrument selector** is a shared component reused on the dashboard
  toolbar and the stat-detail filter bar (styles in `styles.css`).
- **Hover states**: nav items, cards, buttons, table rows, and inputs (focus
  ring) all have hover/focus styling for the frontend to replicate.

## Data

All card content uses **real stat families** and **real date ranges / sample
counts** read from `results/*.json` (e.g. Initial Balance Breakout, NQ,
30min + 1h, 2008-12-11 → 2026-05-05, 4,332 samples). The "Top edge" pills on
the dashboard are representative illustrative values to demonstrate the
green/red semantic. The stat detail page uses **real result values**:
probabilities and edges from `opening_candle_continuation.json` and
value/baseline/slice numbers from `opening_range_indicator.json` (both NQ,
15min).

## Conventions baked in

- **Dark theme first** — the only theme designed here.
- **Color semantics** — green = above baseline / positive edge, red = below
  baseline / negative edge, neutral grey = near baseline. See the `.edge-*`
  classes.
- **Numeric data** — mono font + `tabular-nums`, percentages at 1 decimal,
  sample sizes with comma separators (`.num` class).
- **Stat-type distinction** — probability families render as a dense table with
  inline probability bars; magnitude families render as labelled value blocks
  with value/baseline/Δ, so the two shapes are never confused.
- **Instrument badge** — distinct gold-toned `NQ` badge so the instrument is
  scannable everywhere.
- English only. 2-space indentation throughout.
