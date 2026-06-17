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
- **Source data**: NQ 1-minute OHLCV CSV files in `America/Chicago` timezone.
- **Working timezone**: `America/New_York` (ET). All stats are computed in ET.
- **Conversion**: Chicago → New York directly. No UTC intermediate step.
- **Storage**: Parquet files with timezone-aware timestamps (`America/New_York`).
- **Session definition**: Regular Trading Hours = 09:30–16:15 ET for NQ.
- **Overnight session**: 18:00 ET (previous day) → 09:30 ET.

## Statistics Conventions
- Every stat module MUST include a **random baseline** comparison.
- Every probability MUST report its **sample size** (N).
- **Pending samples** (unresolved at end of data) are ALWAYS excluded from every statistic.
- Denominators = confirmed outcomes only. Never count pending in numerator or denominator.
- Results MUST be **reproducible**: fixed seed for any randomized baseline.
- No stat is declared significant without sufficient sample size (N > 100 minimum).

## Project Structure
```
augur/
├── .claude/
│   ├── agents/           # Claude Code specialized agents
│   └── skills/           # Claude Code automation skills
├── pipeline/             # Data ingestion and OHLCV construction
│   ├── convert_tz.py     # Chicago → New York conversion
│   └── build_candles.py  # 1min → 5min/15min/30min/1h/daily
├── stats/                # Statistical modules (1 directory per report family)
│   ├── base.py           # BaseStat ABC — interface all stats implement
│   └── opening_candle/   # Opening candle continuation stats
├── pinescript/           # Pine Script generator
│   ├── generator.py      # Reads stat results → produces .pine files
│   └── templates/        # Pine Script template fragments
├── data/                 # Parquet files (gitignored)
│   ├── raw/              # Original CSV files
│   └── processed/        # Timezone-converted Parquet
├── output/               # Generated Pine Script files (gitignored)
├── tests/
├── docs/
│   └── STATS_CATALOG.md  # Definition and methodology of every stat
├── CLAUDE.md
├── STATUS.md
└── pyproject.toml
```

## Git Workflow
- Branch naming: `feature/<issue-number>-short-description`, `fix/<issue-number>-short-description`
- Commit messages: `feat(scope): description`, `fix(scope): description`, `docs(scope): description`
- Every change goes through a PR linked to a GitHub issue.
- Never commit directly to `main`.
- Issue labels determine agent routing: `agent-stats`, `agent-pipeline`, `agent-frontend`.

## Commands
```bash
# Virtual environment
cd augur && source .venv/bin/activate

# Run pipeline (convert + build candles)
python -m pipeline.convert_tz
python -m pipeline.build_candles

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
- `data/` and `output/` directories are gitignored. Never commit data or generated files.
