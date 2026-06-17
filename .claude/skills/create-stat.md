---
name: create-stat
description: Scaffold a new statistical analysis module with implementation, baseline, tests, and catalog entry.
disable-model-invocation: true
---
Create a new stat module for: $ARGUMENTS

## 1. Read the spec

Check `docs/STATS_CATALOG.md` for the stat definition. If it doesn't exist yet, ask the user for:
- What condition is being measured (e.g., "opening candle direction")
- What outcome is being tracked (e.g., "end-of-day direction")
- What timeframes apply (e.g., "15min, 30min, 1h")
- What session window applies (e.g., "RTH only")

## 2. Create the module directory

```bash
mkdir -p stats/<stat_family>/
```

## 3. Implement the stat module

Using the **stats-dev** agent, create `stats/<stat_family>/<stat_name>.py`:
- Implement `compute()` following the BaseStat interface
- Implement `baseline()` with random entry comparison
- Include sample sizes in all results
- Exclude pending/unresolved samples

## 4. Write tests

Using the **test-writer** agent, create `tests/stats/test_<stat_name>.py`:
- Build synthetic data with known outcomes
- Hand-calculate expected probabilities
- Test compute, baseline, and edge cases

## 5. Update the catalog

Add the new stat to `docs/STATS_CATALOG.md` with:
- Name and description
- Methodology (how it's computed)
- Parameters (timeframe, session window, etc.)
- Baseline method description

## 6. Run and verify

```bash
pytest tests/stats/test_<stat_name>.py -v
ruff check stats/<stat_family>/
```

## 7. Test on real data (if available)

```bash
python -m stats.<stat_family>.<stat_name> --instrument NQ
```

Review output: are probabilities in a reasonable range? Is sample size adequate?
Does the baseline show the expected ~50% rate?
