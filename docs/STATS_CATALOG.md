# Augur — Stats Catalog

Reference document for all implemented statistical modules.
Each stat has a methodology section precise enough to reproduce the computation.

---

## Slice Architecture

Most stat families share the same secondary breakdowns ("how does this probability
differ by weekday / by gap size / by prior-candle color?"). Rather than re-implement
those in every module, a family **declares** which slicers apply and the framework
re-runs the family's core row computation over each named subset of days.

A family implements only:
- `build_day_table(candles_df) -> pd.DataFrame` — one row per resolved day, indexed by
  the normalized session date, with every column the core computation and the declared
  slicers read. Pending days are excluded here.
- `compute_rows(day_table, baseline_rows=None) -> list[StatResultRow]` — the core
  condition/outcome computation over a (possibly sliced) subset.
- `baseline_rows(day_table, seed) -> list[StatResultRow]` — the random baseline.

The framework's `BaseStat.compute()` builds the day table once, computes the overall
result, then for each declared slice splits the day table into groups and re-runs
`compute_rows` (with a per-group baseline) over each. Each group also carries its own
i18n `label`, because some labels are data-dependent (quantile bucket edges).

Families declare slices on the class. Parameterless slicers may use a bare-string
shorthand; parameterized slicers are configured instances:

```python
class GapFill(BaseStat):
  slices = [
    'weekday',                                   # by day of week
    'close',                                     # by session close color
    'prev_candle',                               # by prior session candle color
    SizeBucket(column='gap_size', preset='quartiles'),
    Levels(ref='orb_range', ext='extension', multiples=(0.5, 1.0, 1.5, 2.0)),
  ]
```

### Reused slicers (shared in `stats/base.py`)

| Slicer | Shorthand | Parameters | Groups |
|--------|-----------|------------|--------|
| `Weekday` | `'weekday'` | — | one per weekday present (Mon→Sun) |
| `Close` | `'close'` | `column='session_green'` | green / red |
| `PrevCandle` | `'prev_candle'` | `column='prev_session_green'` | green / red |
| `SizeBucket` | — | `column`, `preset` (median/terciles/quartiles/quintiles) **or** explicit `buckets` | `q1…qn` (equal-frequency quantile bins, or fixed edges) |
| `Levels` | — | `ref`, `ext`, `multiples` | extension bands in multiples of a reference range |

`SizeBucket` and `Levels` require parameters, so they have no bare-string shorthand —
declare them as instances. Slicers omit empty groups and exclude rows with missing
values in their required columns.

### JSON output

Each `TimeframeResult` keeps its overall `results` and gains a `slices` map:

```jsonc
"slices": {
  "weekday": {
    "dimension": "weekday",
    "groups": {
      "monday": { "label": {"en": "Monday", "fr": "Lundi"}, "total_samples": 42, "results": [ /* StatResultRow[] */ ] }
    }
  }
}
```

The static dimension names live in `labels.dimensions` (`{"weekday": {"en": "Day of week", ...}}`);
the per-group labels (including data-dependent bucket ranges) live inside each group.

---

## 1. Opening Candle Continuation

**Family**: `opening_candle`
**Module**: `stats/opening_candle/continuation.py`
**Result file**: `results/opening_candle_continuation.json`
**Status**: To implement

### What it measures
After the first N-minute candle of the RTH session, how often does the session
close in the same direction as that candle?

### Methodology
1. For each trading day, identify the opening candle: the timeframe candle that
   contains the RTH open, aligned to the clock grid as `floor(rth_start / tf) * tf`.
   For a 09:30 ET open this is 09:30 for 15min/30min, but 09:00 for 1h (the 1h
   candle 09:00–10:00 is the one that contains 09:30, since clock-aligned 1h bars
   fall on the hour).
2. Classify the candle as **green** (close >= open) or **red** (close < open).
3. Classify the daily RTH session as **green** (session close >= session open) or **red**.
4. Count the four combinations: green->green, green->red, red->green, red->red.
5. Compute conditional probabilities:
   - P(session green | candle green) = green->green / total green candles
   - P(session red | candle red) = red->red / total red candles

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| timeframe | 15min | Opening candle duration: 15min, 30min, or 1h |
| session_start | 09:30 | RTH start time (ET) |
| session_end | 16:15 | RTH end time (ET) |

### Timeframes computed
- 15min (first 15-minute candle: 09:30-09:45)
- 30min (first 30-minute candle: 09:30-10:00)
- 1h (first 1-hour candle: 09:00-10:00)

### Baseline
Random direction assignment (50/50 green/red) for each day.
Expected baseline: ~50% for all conditions. Fixed seed for reproducibility.

### i18n
- **title.en**: "Opening Candle Continuation"
- **title.fr**: "Continuation de la bougie d'ouverture"
- **definition.en**: "After the first N-minute candle of the NY session, how often does the session close in the same direction?"
- **definition.fr**: "Après la première bougie de N minutes de la session NY, à quelle fréquence la session clôture-t-elle dans la même direction ?"

### Slices
- `weekday` — implemented (declared via `slices = ("weekday",)`)

### Future variants (not in MVP)
- `by_size` — sliced by candle body size buckets (would add an opening-body-size
  column to the day table and declare `SizeBucket(...)`)
- `by_close` — a distinct *metric*, not the generic `Close` slicer (which merely
  splits days into session-green / session-red). `by_close` measures where the
  session **closes relative to the opening candle**: e.g. inside the opening
  candle's range, beyond its close, or back through its open — quantifying how
  far the session travels from the opening candle, not just its color.

---

## 2. Green & Red Days by Weekday

**Family**: `green_red_days`
**Module**: `stats/green_red_days/by_weekday.py`
**Result file**: `results/green_red_days_by_weekday.json`
**Status**: Implemented

### What it measures
How often does each weekday close green (up) versus red (down)? The overall
result is the marginal probability across all resolved days; the per-weekday
breakdown is produced by the declared `weekday` slice.

### Methodology
1. Build the RTH daily candle per day: `session_open` = open of the 09:30 bar,
   `session_close` = close of the last RTH bar.
2. A day is **resolved** if it has a clean session-open bar and its last RTH bar
   is at or after `session_end - close_tolerance` (default 16:00). Unresolved
   days (early closes, the final incomplete day) are excluded.
3. Classify each resolved day as **green** or **red** per the `performance` mode:
   - `close_to_close` (default): green if `session_close >= previous resolved
     day's session_close`. The first resolved day has no prior close and is
     excluded (pending-sample discipline). The reference is the previous
     **resolved** session, so it skips over any excluded/early-close day.
   - `open_to_close`: green if `session_close >= session_open`.
4. Count green vs red over all resolved days, then re-run per weekday via the
   `weekday` slice.

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| performance | close_to_close | Day-direction basis: `close_to_close` or `open_to_close` |
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `daily` (the RTH daily candle; a single entry).

### Baseline
Random direction assignment (50/50 green/red) per resolved day.
Expected baseline: ~50% for both green and red. Fixed seed for reproducibility.

### i18n
- **title.en**: "Green & Red Days by Weekday"
- **title.fr**: "Jours verts et rouges par jour de la semaine"
- **definition.en**: "How often does each weekday close green (up) versus red (down)?"
- **definition.fr**: "À quelle fréquence chaque jour de la semaine clôture-t-il en vert (hausse) plutôt qu'en rouge (baisse) ?"

### Slices
- `weekday` — implemented (declared via `slices = ("weekday",)`)

