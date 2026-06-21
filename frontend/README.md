# Augur Frontend

Admin backoffice for **Augur — Trading Probability Terminal**. A React + TypeScript
single-page app that displays statistical market probabilities loaded from the
FastAPI backend.

Stack: **Vite** · **React** · **TypeScript** (strict) · **Mantine v7** · **CSS Modules**.
Dark-theme first. English only (no i18n).

## Prerequisites

- Node 20+
- [pnpm](https://pnpm.io/) (project package manager)

## Commands

```bash
pnpm install        # install dependencies
pnpm dev            # dev server on :5173 (proxies /api, /auth, /admin → :8000)
pnpm build          # type-check (tsc -b) + production build to dist/
pnpm preview        # preview the production build locally
pnpm lint           # ESLint
pnpm format         # Prettier — write
pnpm format:check   # Prettier — check only
pnpm tsc --noEmit   # standalone type check
```

The dev server proxies API calls to the FastAPI backend on `http://localhost:8000`
(see `vite.config.ts`). Start the backend separately:

```bash
cd ../backend && uv run uvicorn app.main:app --reload --port 8000
```

In production, Vite builds static assets that FastAPI serves — one process, one port.

## Structure

```
src/
├── pages/        # Page-level route components (e.g. ShellPage)
├── components/   # Reusable UI components
├── hooks/        # Custom React hooks
├── services/     # API client functions
├── types/        # TypeScript interfaces
├── theme/        # Mantine theme overrides (theme.ts)
├── App.tsx       # MantineProvider + app root
├── main.tsx      # React entry point
└── index.css     # Global design tokens (CSS custom properties)
```

## Conventions

- **2-space indentation** everywhere (TS, JSON, CSS). Enforced by Prettier (`.prettierrc`).
- **TypeScript strict** mode — no `any`. Prefer `interface` over `type` for objects.
- **CSS Modules only** — no inline styles, no Tailwind, no styled-components.
- **Design tokens** live in `src/index.css` (`:root`) and are mapped into the Mantine
  theme in `src/theme/theme.ts`. Source of truth: `docs/mockups/styles.css`.
- **Color semantics**: green = above baseline / positive edge, red = below baseline /
  negative edge, neutral grey = near baseline.

See the root `CLAUDE.md` and `docs/mockups/` for the full design reference.
