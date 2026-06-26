# Augur — Trading Probability Terminal

Augur is a statistical analysis engine that computes conditional market
probabilities from historical OHLCV data and exposes them through a JSON API,
an admin backoffice, and a generated TradingView (Pine Script) indicator.

License: AGPL-3.0 — Repository: `marubozu-fr/augur`

## Architecture

- **Stats engine** (`stats/`): Python modules that read Parquet OHLCV data from
  `data/` and write one self-documenting JSON file per stat family to
  `results/`. Each result file carries i18n titles/labels and the actual
  per-instrument, per-timeframe numbers.
- **Backend** (`backend/`): FastAPI. On startup it loads every `results/*.json`
  into an in-memory stats loader (`backend/app/core/stats_loader.py`). The
  backend never writes to `results/` or `data/`. Layering follows
  routers → services → repositories → models.
- **Frontend** (`frontend/`): React 18 + TypeScript + Mantine v7, built with
  Vite. In production the static Vite build (`frontend/dist/`) is served by the
  same FastAPI process; there is one process and one port.
- **Pine Script** (`pinescript/`): generator that turns stat results into a
  TradingView v6 indicator.

## Infrastructure

- **Hosting**: Railway deploys the backend. Build and start are defined in
  `railpack.json` (and `nixpacks.toml`): install deps with `uv sync`, build the
  frontend with `pnpm build`, then run
  `uvicorn backend.app.main:app` on `$PORT`.
- **Database**: Supabase PostgreSQL holds the auth data (users, sessions, API
  keys). The connection string is provided to Railway as `DATABASE_URL`.
- **Domain**: `augur.marubozu.ovh`, an OVH CNAME pointing at the Railway
  service.

## Auth database (dual-backend)

The auth layer runs on PostgreSQL or SQLite, selected at runtime:

- **PostgreSQL** when `DATABASE_URL` is set (production / Railway, backed by
  Supabase).
- **SQLite** otherwise — the fallback for local development and the test suite
  (`backend/db/augur.db`).

The backend selection lives in `backend/app/core/db.py`, which centralizes the
dialect differences (schema DDL and inserted-id retrieval) so the repository
SQL stays portable.

There is **no migration tool**. `init_db()` runs at application startup and
creates the three tables (`users`, `sessions`, `api_keys`) idempotently
(`CREATE TABLE IF NOT EXISTS`). The Postgres database itself must already exist;
only the tables are created here.

## Auth model

- **Admin sessions** (backoffice): cookie-based. `POST /auth/login` validates
  credentials, creates a server-side session row, and sets an httponly session
  cookie. The admin user is seeded from environment variables at startup (see
  below) — there is no self-service user-registration endpoint.
- **API access** (`/api/v1/`): API keys via the `X-API-Key` header. Keys are
  created and revoked by an admin; the plaintext is returned exactly once at
  creation and only its SHA-256 hash is stored.

Admin operations (managing API keys) are done with `curl`, not through the
backoffice UI. The backoffice is a read-oriented stats viewer.

## Environment variables (Railway)

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | Supabase Postgres connection string. When set, the auth DB uses Postgres; otherwise SQLite. |
| `AUGUR_ADMIN_USERNAME` | Admin username seeded at startup (skipped if empty). |
| `AUGUR_ADMIN_PASSWORD` | Admin password seeded at startup (skipped if empty). |
| `AUGUR_DEFAULT_API_KEY` | If set, a reader API key with this plaintext is seeded at startup (idempotent). |
| `AUGUR_DEFAULT_API_KEY_NAME` | Name/label for the seeded default API key (default: `default`). |
| `AUGUR_SESSION_COOKIE_SECURE` | Set to `true` in production so the session cookie is only sent over HTTPS. |

Seeding is idempotent and non-fatal: an existing admin/API key is left
untouched, and a seeding failure logs a warning instead of crashing startup.

## Local development

```bash
# Python environment
cd augur && source .venv/bin/activate

# Build the frontend (installs deps + Vite build into frontend/dist/)
make build

# Run the backend on :8000 (serves the built frontend if present)
make serve

# Or run the frontend dev server on :5173 (proxies API calls to :8000)
cd frontend && pnpm dev
```

Without `DATABASE_URL`, local runs use the SQLite auth DB at
`backend/db/augur.db`, created automatically on first startup.

## Admin operations (curl)

Examples target production (`https://augur.marubozu.ovh`). For local dev use
`http://localhost:8000`.

```bash
# Log in — saves the session cookie to cookies.txt
curl -c cookies.txt -X POST https://augur.marubozu.ovh/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "ADMIN_USERNAME", "password": "ADMIN_PASSWORD"}'

# List API keys (sends the saved cookie)
curl -b cookies.txt https://augur.marubozu.ovh/admin/api-keys

# Create an API key — the plaintext key is returned once, copy it now
curl -b cookies.txt -X POST https://augur.marubozu.ovh/admin/api-keys \
  -H "Content-Type: application/json" \
  -d '{"name": "indicator"}'

# Revoke an API key by its numeric id
curl -b cookies.txt -X DELETE https://augur.marubozu.ovh/admin/api-keys/1
```

`cookies.txt` may contain a live session token — it is gitignored; delete it
when done.

Calling the public API with an API key:

```bash
curl https://augur.marubozu.ovh/api/v1/stats \
  -H "X-API-Key: YOUR_PLAINTEXT_KEY"
```

## Stats engine

Run a stat module as `python -m stats.<family>.<variant> --instrument NQ`. Most
families expose a `standard` variant; some have named variants. For example:

```bash
python -m stats.opening_candle.continuation --instrument NQ
python -m stats.fair_value_gaps.standard --instrument NQ
```

Each run validates its output with Pydantic models and writes (atomically) one
JSON file per family to `results/<family>.json`. No historical runs are kept —
re-run the module to recompute; the source Parquet in `data/` is the audit
trail.

The backend does not watch `results/`. New or updated files are picked up when
the stats loader runs at startup, or on demand via the admin reload endpoint
(`POST /admin/stats/reload`).

Generate the Pine Script indicator from the current results:

```bash
python -m pinescript.generator --instrument NQ --output output/augur_nq.pine
```

## Testing

```bash
# Stats engine + backend tests (SQLite path)
uv run pytest

# Backend only
cd backend && uv run pytest

# Lint (Python)
uv run ruff check .

# Frontend lint + type check
cd frontend && pnpm lint && pnpm tsc --noEmit
```
