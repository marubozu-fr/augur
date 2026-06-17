# Augur — Trading Probability Terminal

## Project Overview
Augur is a statistical analysis engine that computes conditional market probabilities
from historical OHLCV data and delivers them as a real-time TradingView indicator.
Repository: `marubozu-fr/augur` — License: AGPL-3.0

## Tech Stack
- **Stats engine**: Python 3.12+, pandas, pyarrow (Parquet)
- **API** (Phase 2): FastAPI, SQLite
- **Frontend** (Phase 2): React 18+, TypeScript, Mantine UI, TradingView Lightweight Charts
- **Output**: Pine Script v5 indicator (Phase 1 deliverable)
- **Package manager**: `uv` (Python), `pnpm` (Node — Phase 2)

## Code Style — STRICT
- **Language**: All code, comments, commit messages, and documentation in English.
- **Conversations** (chat, prompts): French.
- **Indentation**: 2 spaces everywhere (Python, TypeScript, JSON, YAML, CSS).
- **Python**: PEP 8 except 2-space indentation. Type hints on all functions.
- **TypeScript**: Strict mode. No `any` types. Prefer `interface` over `type` for objects.

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

Results are validated by **Pydantic models** before writing. Atomic write (temp file + rename).
No historical runs are stored — re-run the stat module to recompute. The source Parquet data
is the audit trail.

## Project Structure
```
augur/
├── .claude/
│   ├── agents/           # Claude Code specialized agents
│   │   ├── stats-dev.md  # Stat module implementation
│   │   ├── code-reviewer.md
│   │   └── test-writer.md
│   └── skills/           # Claude Code automation skills
│       ├── fix-issue.md  # Issue → agent routing
│       ├── pr-ready.md   # Quality gate before PR
│       └── create-stat.md # Scaffold new stat module
├── stats/                # Statistical modules
│   ├── base.py           # BaseStat ABC + Pydantic models + write_results()
│   └── opening_candle/   # First stat family
├── pinescript/           # Pine Script generator
│   ├── generator.py
│   └── templates/
├── config/
│   └── NQ.yaml           # Instrument sessions and timeframes
├── data/                 # OHLCV Parquet files (gitignored, prepared externally)
├── results/              # Stat result JSON files (gitignored)
├── output/               # Generated Pine Script files (gitignored)
├── tests/
├── docs/
│   └── STATS_CATALOG.md
├── CLAUDE.md
├── STATUS.md
└── pyproject.toml
```

## Git Workflow
- Branch naming: `feature/<issue-number>-short-description`, `fix/<issue-number>-short-description`
- Commit messages: `feat(scope): description`, `fix(scope): description`, `docs(scope): description`
- Every change goes through a PR linked to a GitHub issue.
- Never commit directly to `main`.
- Issue labels determine agent routing: `agent-stats`, `agent-frontend`.

## Commands
```bash
# Virtual environment
cd augur && source .venv/bin/activate

# Run a specific stat
python -m stats.opening_candle.continuation --instrument NQ

# Generate Pine Script
python -m pinescript.generator --instrument NQ --output output/augur_nq.pine

# Tests
pytest tests/ -v

# Lint
ruff check .
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
- `data/`, `results/`, and `output/` directories are gitignored. Never commit data or generated files.
