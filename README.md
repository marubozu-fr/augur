# Augur — Trading Probability Terminal

License: MIT — Repository: `marubozu-fr/augur`

## What Augur is

Augur is a conditional-probability engine for NQ and ES futures. It computes
statistics from historical 1-minute OHLCV data — e.g. "given the opening
candle closed green, how often does the session also close green?" — and
publishes the results through a JSON API, an admin backoffice, and a
generated TradingView (Pine Script) indicator.

Augur is **non-prescriptive**: it characterizes historical market behavior. It
does not issue trade signals, entries, or recommendations.

## Status

This project is archived and not maintained. There is no live deployment of
the API; the instructions below are for running it locally.

Its member-facing front end is published separately as an open-source demo:
[marubozu-fr/augur-saas](https://github.com/marubozu-fr/augur-saas), live at
<https://augur-saas.marubozu.ovh>. The demo serves a frozen snapshot of this
repository's aggregated `results/` and does not call a live API.

## Statistical methodology

- Every stat includes a **random baseline** comparison, so a probability can
  be judged against chance rather than in isolation.
- Every probability reports its **sample size** (N).
- **Pending samples** (unresolved at the end of the data) are always excluded
  from every statistic — denominators are confirmed outcomes only.
- Randomized baselines use a **fixed seed**, so results are reproducible for
  the same input data.
- No stat is treated as significant unless **N > 100**.

## Architecture

- **Stats engine** (`stats/`): Python modules built on a `BaseStat` base
  class and a declarative slice architecture (`stats/base.py`). A stat
  computes a per-day table once, then the framework re-runs the core
  computation over named subsets (weekday, size quartile, prior-candle
  color, ...) without each stat reimplementing the slicing.
- **Results** (`results/`): one self-documenting JSON file per stat family,
  written atomically and validated against Pydantic models before being
  saved. Each file carries i18n (`en`/`fr`) titles and labels alongside the
  numbers.
- **Backend** (`backend/`): FastAPI. On startup it loads every
  `results/*.json` into an in-memory stats loader
  (`backend/app/core/stats_loader.py`) and never writes to `results/` or
  `data/`. It exposes:
  - session-cookie admin endpoints under `/admin/` (API key management,
    stats reload),
  - API-key endpoints under `/api/v1/` (`X-API-Key` header), including
    date-range re-aggregation of a stat's results from its per-day samples,
  - public `/auth/` endpoints (login/logout) and `/health`.

  Layering follows routers → services → repositories → models. Auth data
  (users, sessions, API keys) lives in PostgreSQL or SQLite, selected at
  runtime (`backend/app/core/db.py`).
- **Frontend** (`frontend/`): React 18 + TypeScript + Mantine v7, built with
  Vite. In production the static Vite build (`frontend/dist/`) is served by
  the same FastAPI process — one process, one port.
- **Pine Script** (`pinescript/`): generator that turns stat results into a
  TradingView v6 indicator.

## How it was built

Augur was developed with Claude Code sub-agents (`.claude/agents/`,
`.claude/skills/`, `CLAUDE.md`) through an issue → PR → diff-review workflow:
each GitHub issue is routed to a specialized agent (stats, backend, frontend,
Pine Script, UI design) based on its label, and changes land through
reviewed pull requests.

## Running locally

Market data is **not included** in this repository — OHLCV data is licensed
and prepared externally. To run the stats engine yourself you need a Parquet
file per instrument at `data/{INSTRUMENT}_1min.parquet` (e.g.
`data/NQ_1min.parquet`, overridable via `data.parquet_1min` in
`config/{INSTRUMENT}.yaml`), with a `timestamp` column and OHLCV columns
(`open`, `high`, `low`, `close`, `volume`). The `timestamp` column must be
tz-aware in `America/New_York`: the engine reads hour and minute directly
from it with no timezone conversion (`stats/utils/daily_candles.py`), so
timestamps in any other timezone silently break session detection. Session
boundaries (RTH, overnight, ETH, and the geographic sessions) are read from
`config/{INSTRUMENT}.yaml`, in `America/New_York`. Configs are committed for
`NQ` and `ES`.

The published `results/*.json` files contain aggregated results only:
per-day samples are stripped because they are derived from licensed market
data. Date-range re-aggregation in the backend therefore requires recomputing
the stats against your own Parquet data — see `python -m
stats.<family>.<variant> --instrument NQ` below.

```bash
# Python environment (creates .venv with runtime + dev dependencies)
cd augur
uv sync --extra dev
source .venv/bin/activate

# Build the frontend (installs deps + Vite build into frontend/dist/)
make build

# Run the backend on :8000 (serves the built frontend if present)
make serve

# Or run the frontend dev server on :5173 (proxies API calls to :8000)
cd frontend && pnpm dev
```

Without `DATABASE_URL`, local runs use the SQLite auth DB at
`backend/db/augur.db`, created automatically on first startup.

## Auth database (dual-backend)

The auth layer runs on PostgreSQL or SQLite, selected at runtime:

- **PostgreSQL** when `DATABASE_URL` is set.
- **SQLite** otherwise — the fallback for local development and the test
  suite (`backend/db/augur.db`).

The backend selection lives in `backend/app/core/db.py`, which centralizes
the dialect differences (schema DDL and inserted-id retrieval) so the
repository SQL stays portable.

There is **no migration tool**. `init_db()` runs at application startup and
creates the three tables (`users`, `sessions`, `api_keys`) idempotently
(`CREATE TABLE IF NOT EXISTS`). The Postgres database itself must already
exist; only the tables are created here.

## Auth model

- **Admin sessions** (backoffice): cookie-based. `POST /auth/login` validates
  credentials, creates a server-side session row, and sets an httponly
  session cookie. The admin user is seeded from environment variables at
  startup (see below) — there is no self-service user-registration endpoint.
- **API access** (`/api/v1/`): API keys via the `X-API-Key` header. Keys are
  created and revoked by an admin; the plaintext is returned exactly once at
  creation and only its SHA-256 hash is stored.

Admin operations (managing API keys) are done with `curl`, not through the
backoffice UI. The backoffice is a read-oriented stats viewer.

## Environment variables

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | Postgres connection string. When set, the auth DB uses Postgres; otherwise SQLite. |
| `AUGUR_ADMIN_USERNAME` | Admin username seeded at startup (skipped if empty). |
| `AUGUR_ADMIN_PASSWORD` | Admin password seeded at startup (skipped if empty). |
| `AUGUR_DEFAULT_API_KEY` | If set, a reader API key with this plaintext is seeded at startup (idempotent). |
| `AUGUR_DEFAULT_API_KEY_NAME` | Name/label for the seeded default API key (default: `default`). |
| `AUGUR_SESSION_COOKIE_SECURE` | Set to `true` in production so the session cookie is only sent over HTTPS. |

Seeding is idempotent and non-fatal: an existing admin/API key is left
untouched, and a seeding failure logs a warning instead of crashing startup.

## Admin operations (curl)

All examples target a local run (`http://localhost:8000`).

```bash
# Log in — saves the session cookie to cookies.txt
curl -c cookies.txt -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "ADMIN_USERNAME", "password": "ADMIN_PASSWORD"}'

# List API keys (sends the saved cookie)
curl -b cookies.txt http://localhost:8000/admin/api-keys

# Create an API key — the plaintext key is returned once, copy it now
curl -b cookies.txt -X POST http://localhost:8000/admin/api-keys \
  -H "Content-Type: application/json" \
  -d '{"name": "indicator"}'

# Revoke an API key by its numeric id
curl -b cookies.txt -X DELETE http://localhost:8000/admin/api-keys/1
```

`cookies.txt` may contain a live session token — it is gitignored; delete it
when done.

Calling the public API with an API key:

```bash
curl http://localhost:8000/api/v1/stats \
  -H "X-API-Key: YOUR_PLAINTEXT_KEY"
```

## Stats engine

Run a stat module as `python -m stats.<family>.<variant> --instrument NQ`.
Most families expose a `standard` variant; some have named variants. For
example:

```bash
python -m stats.opening_candle.standard --instrument NQ
python -m stats.fair_value_gaps.standard --instrument NQ
```

Each run validates its output with Pydantic models and writes (atomically)
one JSON file per family to `results/<family>.json`. No historical runs are
kept — re-run the module to recompute; the source Parquet in `data/` is the
audit trail.

The backend does not watch `results/`. New or updated files are picked up
when the stats loader runs at startup, or on demand via the admin reload
endpoint (`POST /admin/stats/reload`).

Generate the Pine Script indicator from the current results:

```bash
python -m pinescript.generator --instrument NQ --output output/augur_nq.pine
```

## Testing

```bash
# Stats engine + backend tests (SQLite path)
uv run --extra dev pytest

# Backend only
cd backend && uv run --extra dev pytest

# Lint (Python)
uv run --extra dev ruff check .

# Frontend lint + type check
cd frontend && pnpm lint && pnpm tsc --noEmit
```
