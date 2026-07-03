# Augur — Trading Probability Terminal

## Project Overview
Augur is a statistical analysis engine that computes conditional market probabilities
from historical OHLCV data and delivers them as a real-time TradingView indicator.
Repository: `marubozu-fr/augur` — License: AGPL-3.0

## Tech Stack
- **Stats engine**: Python 3.12+, pandas, pyarrow (Parquet)
- **Backend**: FastAPI, dual-backend auth DB (PostgreSQL in production, SQLite fallback), Pydantic v2
- **Frontend (backoffice)**: React 18+, TypeScript, Mantine v7, CSS Modules, Recharts, Vite 6+
- **Output**: Pine Script v6 indicator
- **Package managers**: `uv` (Python), `pnpm` (Node)

## Code Style — STRICT
- **Language**: All code, comments, commit messages, and documentation in English.
- **Conversations** (chat, prompts): French.
- **Indentation**: 2 spaces everywhere (Python, TypeScript, JSON, YAML, CSS).
- **Python**: PEP 8 except 2-space indentation. Type hints on all functions.
- **TypeScript**: Strict mode. No `any` types. Prefer `interface` over `type` for objects.
- **CSS**: CSS Modules only. No inline styles. No Tailwind. No styled-components.

## Data Conventions
- **Input data**: Parquet files in `data/` with timezone-aware timestamps in `America/New_York`.
- **Data preparation is external**: OHLCV Parquet files are prepared outside this repo and placed in `data/`. No pipeline code is committed.
- **Session definition**: Regular Trading Hours = 09:30–16:15 ET for NQ.
- **Overnight session**: 18:00 ET (previous day) → 09:30 ET.
- **Daily candle**: RTH-based (session open to session close), not midnight-based.

## Statistics Conventions
- Every stat module MUST include a **random baseline** comparison.
- Every probability MUST report its **sample size** (N).
- **Pending samples** (unresolved at end of data) are ALWAYS excluded from every statistic.
- Denominators = confirmed outcomes only. Never count pending in numerator or denominator.
- Results MUST be **reproducible**: fixed seed for any randomized baseline.
- No stat is declared significant without sufficient sample size (N > 100 minimum).

## Storage Format
Stat results are stored as **one JSON file per stat family** in `results/`:
```
results/
└── opening_candle_continuation.json
```

Each file is **self-documenting** with i18n-ready fields:
- `title` and `definition`: `{"en": "...", "fr": "..."}` — what the stat measures
- `labels`: human-readable condition and outcome names in en/fr
- `instruments.{INSTRUMENT}.{TIMEFRAME}`: the actual results

**`results/*.json` IS committed to git** — it is the published stat output the
backend serves, not a throwaway artifact. The `results/` line in `.gitignore` is
commented out (`#results/`). Every stat PR MUST include its result JSON file
(e.g. `results/<stat_family>.json`). This is unlike `data/`, `output/`, and
`frontend/dist/`, which ARE gitignored and never committed.

Results are validated by **Pydantic models** before writing. Atomic write (temp file + rename).
No historical runs are stored — re-run the stat module to recompute. The source Parquet data
is the audit trail.

## Backoffice Architecture

### Backend (FastAPI)
- **Architecture**: routers → services → repositories → models (Kiroku pattern)
- **Stats loader** (`backend/app/core/stats_loader.py`): reads `results/*.json`, caches in memory, reloadable on demand. This is a core service, not a repository — it reads files, not a database.
- **Auth**: dual-backend DB for users (admin/reader roles) and API keys only. PostgreSQL (Supabase/Railway) when `DATABASE_URL` is set, otherwise SQLite (`backend/db/augur.db`) for local dev and tests. The backend is selected at runtime in `backend/app/core/db.py`, which centralizes dialect differences (schema DDL, inserted-id retrieval) so repository SQL stays portable. Parameterized queries only.
- **Endpoints**: `/auth/` (public), `/admin/` (session cookie auth), `/api/v1/` (API key auth via `X-API-Key` header)
- **Response envelope**: `{ "data": ..., "error": null }` or `{ "data": null, "error": "message" }`
- The backend NEVER writes to `results/` or `data/` — those are managed by the stats engine.

### Frontend (React — backoffice admin)
- **Architecture**: pages → components → hooks → services → types
- Mantine v7 for UI components, CSS Modules for custom styling
- Recharts for stat visualizations (probability bars, grouped comparisons)
- Dark theme first — this is a quant/trading tool
- English only — no i18n
- Design mockups produced by **ui-designer** agent before frontend implementation
- Color semantics: green = above baseline / positive edge, red = below baseline / negative edge

### Production
- Vite builds static assets, FastAPI serves them. One process, one port.
- Dev mode: Vite on :5173 (with proxy to backend), uvicorn on :8000.

## Project Structure
```
augur/
├── .claude/
│   ├── agents/           # Claude Code specialized agents
│   │   ├── stats-dev.md      # Stat module implementation
│   │   ├── backend-dev.md    # FastAPI backend
│   │   ├── frontend-dev.md   # React admin frontend
│   │   ├── ui-designer.md    # HTML/CSS mockups (no production code)
│   │   ├── pinescript-dev.md # Pine Script indicator
│   │   ├── code-reviewer.md
│   │   └── test-writer.md
│   └── skills/           # Claude Code automation skills
│       ├── fix-issue/    # Issue → agent routing
│       ├── pr-ready/     # Quality gate before PR
│       └── create-stat/  # Scaffold new stat module
├── stats/                # Statistical modules
│   ├── base.py           # BaseStat ABC + Pydantic models + write_results()
│   └── opening_candle/   # First stat family
├── backend/              # FastAPI backoffice + API
│   ├── app/
│   │   ├── main.py
│   │   ├── routers/      # HTTP route handlers
│   │   ├── models/       # Pydantic models
│   │   ├── services/     # Business logic
│   │   ├── repositories/ # Auth DB queries (Postgres/SQLite, auth only)
│   │   └── core/         # Config, stats loader, dependencies
│   ├── db/               # Local SQLite auth database (fallback)
│   └── tests/
├── frontend/             # React admin dashboard
│   ├── src/
│   │   ├── pages/        # Page-level route components
│   │   ├── components/   # Reusable UI components
│   │   ├── hooks/        # Custom React hooks
│   │   ├── services/     # API client functions
│   │   ├── types/        # TypeScript interfaces
│   │   └── theme/        # Mantine theme overrides
│   ├── package.json
│   └── vite.config.ts
├── pinescript/           # Pine Script generator
│   ├── generator.py
│   └── templates/
├── config/
│   └── NQ.yaml           # Instrument sessions and timeframes
├── data/                 # OHLCV Parquet files (gitignored, prepared externally)
├── results/              # Stat result JSON files (COMMITTED — published output)
├── output/               # Generated Pine Script files (gitignored)
├── tests/                # Stats engine tests
├── docs/
│   ├── STATS_CATALOG.md
│   └── DESIGN_SYSTEM.md  # Backoffice visual reference
├── CLAUDE.md
└── pyproject.toml
```

## Git Workflow
- Branch naming: `feature/<issue-number>-short-description`, `fix/<issue-number>-short-description`
- Commit messages: `feat(scope): description`, `fix(scope): description`, `docs(scope): description`
- Every change goes through a PR linked to a GitHub issue.
- Never commit directly to `main`.
- Issue labels determine agent routing: `agent-stats`, `agent-pinescript`, `agent-backend`, `agent-frontend`, `agent-designer`.

## Commands
```bash
# Virtual environment
cd augur && source .venv/bin/activate

# Run a specific stat
python -m stats.opening_candle.standard --instrument NQ

# Generate Pine Script
python -m pinescript.generator --instrument NQ --output output/augur_nq.pine

# Stats tests
pytest tests/ -v

# Stats lint
ruff check .

# Backend dev server
cd backend && uv run uvicorn app.main:app --reload --port 8000

# Backend tests
cd backend && uv run pytest

# Backend lint
cd backend && uv run ruff check .

# Frontend dev server
cd frontend && pnpm dev

# Frontend lint
cd frontend && pnpm lint

# Frontend type check
cd frontend && pnpm tsc --noEmit

# Production build
cd frontend && pnpm build
```

## Behavioral Guidelines

### Think Before Coding
- State your assumptions explicitly before implementing. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.

### Simplicity First
- Minimum code that solves the problem. Nothing speculative.
- No features beyond what was asked. No abstractions for single-use code.
- If you write 200 lines and it could be 50, rewrite it.

### Surgical Changes
- Touch only what you must. Don't "improve" adjacent code.
- If you notice unrelated issues, mention them — don't fix them silently.
- The test: every changed line should trace directly to the request.

### Goal-Driven Execution
- Transform tasks into verifiable goals with success criteria.
- For multi-step tasks, state a brief plan with verification at each step.

## Project Rules
- NEVER hardcode instrument names, session times, or timezone strings. Use config.
- NEVER publish a stat without its random baseline comparison.
- NEVER count pending/unresolved samples in any statistic.
- ALWAYS write tests with known synthetic data for every new stat module.
- ALWAYS verify stat results are reproducible (deterministic output for same input).
- NEVER put business logic in routers — it belongs in services.
- NEVER use SQL string concatenation — parameterized queries only.
- NEVER use `any` type in TypeScript — define proper interfaces.
- ALWAYS check `docs/DESIGN_SYSTEM.md` before creating UI components.
- `data/`, `output/`, and `frontend/dist/` directories are gitignored — never commit data or generated artifacts. NOTE: `results/` is NOT gitignored — its stat JSON files ARE committed (published output the backend serves); always include the result JSON in a stat PR.