# Augur Backoffice — Dashboard Mockups (Issue #116)

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
| `index.html`             | The dashboard in the app shell (sidebar + header + main), **populated state**: metric strip, search/filter toolbar, and a 16-card grid (one card per real stat family). |
| `dashboard-states.html`  | The dashboard **loading** state (skeleton cards + spinner) and two **empty** states (no results at all, and no search match), as clearly labeled sections on one page. |
| `styles.css`             | Shared stylesheet. All design tokens are CSS custom properties (`:root`) — colors, typography, spacing scale, radii, elevation, layout dimensions. |
| `README.md`              | This file. |

## How to view

Open `index.html` or `dashboard-states.html` directly in a browser. Both link
to `styles.css` relatively. No build step.

## Interactions demonstrated without JS

- **Sidebar collapse**: checkbox hack (`#sidebar-toggle` + `☰` label in the
  header) collapses the sidebar to an icon rail. Also collapses automatically
  below 900px via media query.
- **Hover states**: nav items, cards, buttons, and inputs (focus ring) all have
  hover/focus styling for the frontend to replicate.

## Data

All card content uses **real stat families** and **real date ranges / sample
counts** read from `results/*.json` (e.g. Initial Balance Breakout, NQ,
30min + 1h, 2008-12-11 → 2026-05-05, 4,332 samples). The "Top edge" pills are
representative illustrative values to demonstrate the green/red semantic.

## Conventions baked in

- **Dark theme first** — the only theme designed here.
- **Color semantics** — green = above baseline / positive edge, red = below
  baseline / negative edge, neutral grey = near baseline. See the `.edge-*`
  classes.
- **Numeric data** — mono font + `tabular-nums`, percentages at 1 decimal,
  sample sizes with comma separators (`.num` class).
- **Instrument badge** — distinct gold-toned `NQ` badge so the instrument is
  scannable everywhere.
- English only. 2-space indentation throughout.
