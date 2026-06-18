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


---

## 3. Green & Red Streaks

**Family**: `green_red_streaks`
**Module**: `stats/green_red_streaks/standard.py`
**Result file**: `results/green_red_streaks.json`
**Status**: Implemented

### What it measures
Once a period is green (or red), how likely is the streak to continue for one more
period of the same color? Computed at **daily**, **weekly**, and **monthly**
granularity.

The issue frames this as "consecutive green/red periods" with average and max
streak lengths. We express it as a **continuation probability** so it fits the
probability-row framework natively (each row carries its sample size N and a random
baseline). The **average streak length** is recoverable by consumers as
`1 / (1 - P(continue))`. Max streak length is not represented in this framing.

### Methodology
1. Build the RTH daily candle per resolved day (`session_open`, `session_close`),
   using the same resolution rule as the other daily stats (clean 09:30 open and a
   last RTH bar at or after `session_end - close_tolerance`).
2. Aggregate resolved days into **periods** at the granularity:
   - `daily`: each day is one period.
   - `weekly`: days grouped by ISO week; `period_open` = first day's `session_open`,
     `period_close` = last day's `session_close`.
   - `monthly`: days grouped by calendar month (analogous).
3. Classify each period as **green** or **red** per the `performance` mode:
   - `close_to_close` (default): green if `period_close >= previous resolved
     period's close`. The first period has no prior close and is excluded.
   - `open_to_close`: green if `period_close >= period_open`.
4. For each period, look at the chronologically **next** resolved period. A period
   of a given color "continues" if the next period shares that color, otherwise it
   "breaks". The final period has no next period, so its continuation outcome is
   unresolved and it is excluded from every denominator (pending discipline).
5. Report, per granularity, the four rows: green→continue, green→break,
   red→continue, red→break.

`total_samples` counts all resolved periods; each row's `total` counts only the
**countable** periods of that color (those with a following period).

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| performance | close_to_close | Period-direction basis: `close_to_close` or `open_to_close` |
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `daily`, `weekly`, `monthly` (merged into one result file under
  `instruments.{INSTRUMENT}.{granularity}`).

### Baseline
Each period is recolored at random (50/50 green/red) and the next-period color is
recomputed from the randomized sequence, so continuation is independent of the
current color. Expected baseline: ~50% for every row. Fixed seed for reproducibility.

### i18n
- **title.en**: "Green & Red Streaks"
- **title.fr**: "Séries vertes et rouges"
- **definition.en**: "Once a period is green (or red), how likely is the streak to continue for one more period of the same color?"
- **definition.fr**: "Une fois qu'une période est verte (ou rouge), quelle est la probabilité que la série se poursuive d'une période supplémentaire de la même couleur ?"

### Slices
- None. The streaks stat declares `slices = ()`.


---

## 4. Performance by Weekday

**Family**: `performance_weekday`
**Module**: `stats/performance_weekday/standard.py`
**Result file**: `results/performance_weekday.json`
**Status**: Implemented

### What it measures
For each weekday: the **average percent return**, the **green/red day counts**, and
the **average green-day and red-day move sizes**. The overall result aggregates all
resolved days; the per-weekday breakdown is produced by the declared `weekday` slice.

This is the first **magnitude** stat. Most families report probabilities; here three
of the five rows carry a continuous metric instead. To support this, `StatResultRow`
gained two optional fields:
- `value` — the continuous metric (a signed decimal return, e.g. `0.012` = +1.2%).
- `value_baseline` — its random-baseline counterpart.

Probability rows (`green_day`, `red_day`) leave `value`/`value_baseline` as `null` and
use the ordinary `probability` / `baseline_prob` channel; magnitude rows (`mean_return`,
`mean_green_move`, `mean_red_move`) leave `probability` at `0.0` and populate `value`.
Every row reports its sample size in `count` / `total`.

### Methodology
1. Build the RTH daily candle per day (`session_open` = open of the 09:30 bar,
   `session_close` = close of the last RTH bar), keeping only **resolved** days
   (clean session open and a last RTH bar at or after `session_end - close_tolerance`).
2. Compute each resolved day's signed percent return per the `performance` mode:
   - `close_to_close` (default): `(session_close - previous resolved day's
     session_close) / previous close`. The first resolved day has no prior close and
     is excluded (pending-sample discipline). The reference is the previous **resolved**
     session, so it skips over any excluded/early-close day.
   - `open_to_close`: `(session_close - session_open) / session_open`.
3. A day is **green** when its return is `>= 0`, otherwise **red**.
4. Over all resolved days, report five rows under the single `any_day` condition:
   - `mean_return` — average return across all days (`value`).
   - `green_day` / `red_day` — green and red day counts (`probability`).
   - `mean_green_move` — average return over green days (`value`, `>= 0`).
   - `mean_red_move` — average return over red days (`value`, `<= 0`).
5. Re-run the same five rows per weekday via the `weekday` slice. When a group has no
   green (or red) days, the corresponding mean falls back to `0.0`.

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| performance | close_to_close | Return basis: `close_to_close` or `open_to_close` |
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `daily` (the RTH daily candle; a single entry).

### Baseline
Each day's return **direction is a coin flip**: magnitudes are held fixed and the sign
of each day's return is randomized (p=0.5). Expected average return ≈ 0 and expected
green/red day counts ≈ 50/50. The baseline is computed per slice group as well as
overall, with a fixed seed for reproducibility.

### i18n
- **title.en**: "Performance by Weekday"
- **title.fr**: "Performance par jour de la semaine"
- **definition.en**: "What is the average percent return, the green/red day counts, and the average green/red move size for each weekday?"
- **definition.fr**: "Quels sont le rendement moyen en pourcentage, le nombre de jours verts/rouges et la taille moyenne des mouvements verts/rouges pour chaque jour de la semaine ?"

### Slices
- `weekday` — implemented (declared via `slices = ("weekday",)`)


---

## 5. Previous Session Correlation

**Family**: `prev_session_correlation`
**Module**: `stats/prev_session_correlation/standard.py`
**Result file**: `results/prev_session_correlation.json`
**Status**: Implemented

### What it measures
How often does the current session close green (or red) given the prior session's
color? Day-over-day color follow-through, reported as a 2x2 conditional matrix:
P(green | prior green), P(red | prior green), P(green | prior red), P(red | prior red).

This is the explicit-matrix companion to **Green & Red Streaks** (section 3): both
read the same daily color sequence, but streaks collapses it into one continue/break
outcome, while this stat keeps both outcomes per prior color so the asymmetry
between green- and red-follow-through is visible directly.

### Methodology
1. Build the RTH daily candle per resolved session (`session_open`, `session_close`),
   using the same resolution rule as the other daily stats (clean 09:30 open and a
   last RTH bar at or after `session_end - close_tolerance`).
2. Classify each resolved session as **green** or **red** per the `performance` mode:
   - `close_to_close` (default): green if `session_close >= previous resolved
     session's close`. The first resolved session has no prior close and is excluded
     (pending-sample discipline).
   - `open_to_close`: green if `session_close >= session_open`.
3. Attach each session's **prior** color: the color of the chronologically previous
   resolved session. The first usable session has no prior session, so its condition
   is undefined and it is excluded from every denominator.
4. Report the four rows: prior-green→green, prior-green→red, prior-red→green,
   prior-red→red.

`total_samples` counts all resolved sessions; each row's `total` counts only the
**countable** sessions (those with a prior session) carrying that prior color.

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| performance | close_to_close | Session-direction basis: `close_to_close` or `open_to_close` |
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `daily` (the RTH daily candle; a single entry).

### Baseline
Each session is recolored at random (50/50 green/red) and the prior-session color is
recomputed from the randomized sequence, so the current color is independent of the
prior one. Expected baseline: ~50% for every row. Fixed seed for reproducibility.

### i18n
- **title.en**: "Previous Session Correlation"
- **title.fr**: "Corrélation de la session précédente"
- **definition.en**: "How often does the current session close green or red given the prior session's color?"
- **definition.fr**: "À quelle fréquence la session actuelle clôture-t-elle en vert ou en rouge selon la couleur de la session précédente ?"

### Slices
- None. The prev-session-correlation stat declares `slices = ()`.


---

## 6. Overnight Continuation

**Family**: `overnight_continuation`
**Module**: `stats/overnight_continuation/standard.py`
**Result file**: `results/overnight_continuation.json`
**Status**: Implemented

### What it measures
Given the overnight gap's direction (today's open vs the prior session's close),
how often does the day continue in that direction intraday (close vs open)?
Reported as a 2x2 conditional matrix: P(green day | green gap), P(red day | green
gap), P(green day | red gap), P(red day | red gap).

"Continuation" is the diagonal: a green gap that closes green, or a red gap that
closes red.

### Methodology
1. Build the RTH daily candle per resolved session (`session_open`, `session_close`),
   using the same resolution rule as the other daily stats (clean 09:30 open and a
   last RTH bar at or after `session_end - close_tolerance`).
2. Classify each resolved day's **overnight gap** from `session_open` vs the
   PREVIOUS resolved session's `session_close`:
   - `gap_green`: `session_open >= prev_session_close`
   - `gap_red`: `session_open < prev_session_close`
   The first resolved session has no prior close, so its gap is undefined and it is
   excluded from every denominator (pending-sample discipline).
3. Classify each day's **intraday color** from `session_close` vs `session_open`:
   green if `session_close >= session_open`, otherwise red. This is always
   close-vs-open (intraday) — there is no `close_to_close` mode, because the gap
   already encodes the prior-close reference.
4. Report the four rows: gap-green→green, gap-green→red, gap-red→green, gap-red→red.

`total_samples` counts all resolved sessions; each row's `total` counts only the
**countable** sessions (those with a prior session close) carrying that gap color.

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

Unlike the day-color stats, this family has no `performance` parameter: the gap is
always open-vs-prior-close and the outcome is always intraday close-vs-open.

### Timeframes computed
- `daily` (the RTH daily candle; a single entry).

### Baseline
Each day's intraday color is randomized (50/50 green/red) independently of its gap
color, so continuation is destroyed. Expected baseline: ~0.5 for every row. Fixed
seed for reproducibility.

### i18n
- **title.en**: "Overnight Continuation"
- **title.fr**: "Continuation overnight"
- **definition.en**: "Given the overnight gap's direction (today's open vs the prior session's close), how often does the day close in the same direction (intraday)?"
- **definition.fr**: "Selon la direction du gap overnight (ouverture du jour vs clôture de la session précédente), à quelle fréquence le jour clôture-t-il dans la même direction (intraday) ?"

### Slices
- None. The overnight-continuation stat declares `slices = ()`.

### Future variants (not in MVP)
- `by_weekday` — the gap→intraday matrix sliced per weekday (would declare
  `slices = ("weekday",)`).


---

## 7. Seasonality

**Family**: `seasonality`
**Module**: `stats/seasonality/standard.py`
**Result file**: `results/seasonality.json`
**Status**: Implemented

### What it measures
Month-of-year and week-of-year average performance patterns from monthly and
weekly returns. For each calendar month (January…December) and each ISO week
(1…53), it reports the **average percent return**, the **green/red period split**,
and the **average green/red move size**, averaged across all years in the data.

Like **Performance by Weekday** (section 4), this is a **magnitude** stat: three
of the five rows carry a continuous metric in `value` (a signed decimal return)
with its random baseline in `value_baseline`; the green/red count rows
(`green_period`, `red_period`) use the ordinary `probability` channel.

### Methodology
1. Build the RTH daily candle per resolved day (`session_open`, `session_close`),
   using the same resolution rule as the other daily stats (clean 09:30 open and
   a last RTH bar at or after `session_end - close_tolerance`).
2. Aggregate resolved days into **periods** at the granularity:
   - `monthly`: days grouped by calendar month; `period_open` = first day's
     `session_open`, `period_close` = last day's `session_close`.
   - `weekly`: days grouped by ISO week (analogous).
   Each period is indexed by its first session date, so its calendar position
   (month / ISO week) is read directly by the slice.
3. Compute each period's signed percent return per the `performance` mode:
   - `close_to_close` (default): `(period_close - previous resolved period's
     close) / previous close`. The first resolved period has no prior close and
     is excluded (pending-sample discipline).
   - `open_to_close`: `(period_close - period_open) / period_open`.
4. A period is **green** when its return is `>= 0`, otherwise **red**.
5. Over all resolved periods, report five rows under the single `any_period`
   condition: `mean_return`, `green_period`, `red_period`, `mean_green_move`,
   `mean_red_move`.
6. Re-run the same five rows per calendar position via the slice:
   - `monthly` → `month_of_year` slice (one group per month present, Jan→Dec).
   - `weekly` → `week_of_year` slice (one group per ISO week present, 1→53).

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| performance | close_to_close | Return basis: `close_to_close` or `open_to_close` |
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `monthly`, `weekly` (merged into one result file under
  `instruments.{INSTRUMENT}.{granularity}`).

### Baseline
Each period's return **direction is a coin flip**: magnitudes are held fixed and
the sign of each period's return is randomized (p=0.5). Expected average return
≈ 0 and expected green/red period counts ≈ 50/50. The baseline is computed per
slice group as well as overall, with a fixed seed for reproducibility.

### i18n
- **title.en**: "Seasonality"
- **title.fr**: "Saisonnalité"
- **definition.en**: "What is the average percent return and green/red period split for each month of the year and each week of the year?"
- **definition.fr**: "Quels sont le rendement moyen en pourcentage et la répartition des périodes vertes/rouges pour chaque mois de l'année et chaque semaine de l'année ?"

### Slices
- `month_of_year` — implemented (declared on the monthly granularity).
- `week_of_year` — implemented (declared on the weekly granularity).

Both slicers are module-local (`stats/seasonality/standard.py`); they read the
period table's DatetimeIndex and are not shared in `stats/base.py`.


---

## 8. High & Low by Weekday

**Family**: `high_low_weekday`
**Module**: `stats/high_low_weekday/standard.py`
**Result file**: `results/high_low_weekday.json`
**Status**: Implemented

### What it measures
For each ISO week, which weekday produced the weekly high (RTH intraday high),
the weekly low (RTH intraday low), the weekly high close (highest session close),
and the weekly low close (lowest session close)? Aggregated across weeks as the
fraction of weeks each weekday claims each extreme. For a given condition,
probabilities sum to ~100% across weekdays (each week contributes exactly one
weekday per condition).

### Methodology
1. Build the RTH daily candle per resolved day (`session_open`, `session_close`)
   using the same resolution rule as the other daily stats (clean 09:30 open and
   a last RTH bar at or after `session_end - close_tolerance`).
2. Compute per-day RTH extremes from the same RTH bar filter
   (`hour*60+minute >= rth_start_min` and `< rth_end_min`):
   - `day_high` = max of `high` column over RTH bars for that date.
   - `day_low`  = min of `low` column over RTH bars for that date.
   Join these onto the resolved-days index (inner join — only resolved dates
   are retained).
3. Group resolved days by ISO (year, week). For each week compute:
   - `high_weekday`: `dayofweek` (0=Mon…4=Fri) of the day with max `day_high`
     (first occurrence on ties via `idxmax`).
   - `low_weekday`: `dayofweek` of the day with min `day_low`.
   - `high_close_weekday`: `dayofweek` of the day with max `session_close`.
   - `low_close_weekday`: `dayofweek` of the day with min `session_close`.
   - `weekly_green`: `bool` — `True` when the weekly close (last day's
     `session_close`) is `>=` the weekly open (first day's `session_open`).
   - `present_weekdays`: sorted tuple of distinct `dayofweek` ints present in
     the week (used by the baseline to draw uniformly from actual trading days).
   Each week is indexed by its first session date for chronological sorting.
4. **Pending discipline**: the final ISO week present in the data is always
   excluded — it may be in progress and its extremes are not final. If 0 or 1
   weeks remain after exclusion, an empty table is returned.

### Conditions
| Condition key | Source column | Description |
|---|---|---|
| `weekly_high` | `high_weekday` | Weekday with highest RTH intraday high |
| `weekly_low` | `low_weekday` | Weekday with lowest RTH intraday low |
| `weekly_high_close` | `high_close_weekday` | Weekday with highest RTH session close |
| `weekly_low_close` | `low_close_weekday` | Weekday with lowest RTH session close |

### Outcomes
One per weekday present in the dataset: `monday` … `friday` (and `saturday`,
`sunday` if ever present). Labels are the canonical English/French names from
`_WEEKDAYS` in `stats/base.py`. Weekdays with no occurrences in a subset are
still emitted (count=0) if they appear in any week's `present_weekdays`.

### Parameters
| Parameter | Default | Description |
|---|---|---|
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `weekly` (one entry per ISO week; single entry under
  `instruments.{INSTRUMENT}.weekly`).

### Baseline
Random null: for each week, four independent uniform draws are made from that
week's `present_weekdays` (one per condition), representing the null hypothesis
that each extreme falls on a uniformly random trading day of the week. A temp
copy of the week table is built with the four weekday columns replaced by these
random draws, then `compute_rows` is called on it. Uses
`np.random.default_rng(seed)` for deterministic output. The baseline is
computed per slice group as well as overall.

### i18n
- **title.en**: "High & Low by Weekday"
- **title.fr**: "Plus haut & plus bas par jour de la semaine"
- **definition.en**: "For each ISO week, which weekday produced the weekly high, weekly low, weekly high close, and weekly low close? Aggregated as the fraction of weeks each weekday claims each extreme."
- **definition.fr**: "Pour chaque semaine ISO, quel jour de la semaine a produit le plus haut hebdomadaire, le plus bas hebdomadaire, le plus haut de clôture hebdomadaire et le plus bas de clôture hebdomadaire ? Agrégé sous forme de fraction des semaines où chaque jour revendique chaque extrême."

### Slices
- `weekly_candle` — custom `WeeklyCandle` slicer (subclass of `_ColorSlicer`
  from `stats/base.py`) that splits the week table by `weekly_green` into green
  weeks and red weeks. Delivers the "by weekly candle color" breakdown for free
  via the framework.


---

## 9. Volume & Range by Weekday

**Family**: `volume_range_weekday`
**Module**: `stats/volume_range_weekday/standard.py`
**Result file**: `results/volume_range_weekday.json`
**Status**: Implemented

### What it measures
For each weekday: the **average RTH session volume**, the **average price range**
(`high - low`, in points), and the **average percentage range**
(`(high - low) / session_open`). The overall result aggregates all resolved days;
the per-weekday breakdown is produced by the declared `weekday` slice.

This is a **magnitude** stat: all three rows carry a continuous metric in `value`
(and its random-baseline counterpart in `value_baseline`). The `probability` /
`baseline_prob` channel is left at `0.0` for every row. Every row reports its
sample size in `count` / `total`.

### Methodology
1. Build the RTH daily candle per resolved day (`session_open` = open of the
   09:30 bar) using the shared resolution rule (clean session open and a last RTH
   bar at or after `session_end - close_tolerance`).
2. Compute per-day metrics from the same RTH bar filter
   (`hour*60+minute >= rth_start_min` and `< rth_end_min`):
   - `day_high` = max of `high` over the day's RTH bars.
   - `day_low`  = min of `low` over the day's RTH bars.
   - `volume`   = sum of `volume` over the day's RTH bars.
   Join these onto the resolved-days index (inner join — only resolved dates
   are retained).
3. Derive the two range columns:
   - `range`     = `day_high - day_low` (points).
   - `range_pct` = `range / session_open` (a decimal, e.g. `0.012` = 1.2%).
4. Over all resolved days, report three rows under the single `any_day` condition:
   - `mean_volume`    — average summed RTH volume (`value`).
   - `mean_range`     — average range in points (`value`).
   - `mean_range_pct` — average percentage range (`value`).
5. Re-run the same three rows per weekday via the `weekday` slice.

### Conditions
| Condition key | Description |
|---|---|
| `any_day` | All resolved trading days (the `weekday` slice does the per-day breakdown) |

### Outcomes
| Outcome key | Channel | Description |
|---|---|---|
| `mean_volume` | `value` | Average summed RTH session volume |
| `mean_range` | `value` | Average price range (`high - low`), in points |
| `mean_range_pct` | `value` | Average percentage range (`range / session_open`), decimal |

### Parameters
| Parameter | Default | Description |
|---|---|---|
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `daily` (the RTH daily candle; a single entry).

### Baseline
Random null: for a subset of `N` days (one weekday, or the whole table), draw `N`
days uniformly at random **without replacement** from the full resolved-day table
and compute the same three averages on that random sample. This represents the
null hypothesis that the weekday carries no information — its expected metrics
equal the grand mean across all days. Uses `np.random.default_rng(seed)` for
deterministic output. The baseline is computed per slice group as well as overall;
for the overall result the sample is a permutation of the full table, so its
`value_baseline` equals the overall `value`.

### i18n
- **title.en**: "Volume & Range by Weekday"
- **title.fr**: "Volume & amplitude par jour de la semaine"
- **definition.en**: "What is the average volume, the average price range, and the average percentage range for each weekday?"
- **definition.fr**: "Quels sont le volume moyen, l'amplitude moyenne en points et l'amplitude moyenne en pourcentage pour chaque jour de la semaine ?"

### Slices
- `weekday` — implemented (declared via `slices = ("weekday",)`)


---

## 10. Volume Trends

**Family**: `volume_trends`
**Module**: `stats/volume_trends/standard.py`
**Result file**: `results/volume_trends.json`
**Status**: Implemented

### What it measures
Volume seasonality: the **average traded volume** for each calendar month
(January…December) and each ISO week of the year (1…53). Resolved RTH days are
aggregated into monthly and weekly periods; each period's volume is the **sum** of
its days' RTH volume (the period's total traded volume), and the slice averages
those period totals across all years in the data.

Like **Volume & Range by Weekday** (section 9), this is a **magnitude** stat: the
single `mean_volume` row carries its metric in `value` (and its random-baseline
counterpart in `value_baseline`). The `probability` / `baseline_prob` channel is
left at `0.0`. Every row reports its sample size (number of periods) in
`count` / `total`.

### Methodology
1. Build the RTH daily candle per resolved day and sum each day's RTH volume,
   using the same RTH bar filter (`hour*60+minute >= rth_start_min` and
   `< rth_end_min`) and resolution rule (clean 09:30 open and a last RTH bar at or
   after `session_end - close_tolerance`) as the other daily stats.
2. Aggregate resolved days into **periods** at the granularity:
   - `monthly`: days grouped by calendar month; `volume` = sum of the month's
     daily RTH volume.
   - `weekly`: days grouped by ISO week (analogous).
   Each period is indexed by its first session date, so its calendar position
   (month / ISO week) is read directly by the slice.
3. Drop the **most recent period** (pending-sample discipline): data usually ends
   mid-period, so its total is incomplete and would bias the average downward.
4. Over all resolved periods, report one row under the single `any_period`
   condition: `mean_volume` — the average period volume (`value`).
5. Re-run the same row per calendar position via the slice:
   - `monthly` → `month_of_year` slice (one group per month present, Jan→Dec).
   - `weekly` → `week_of_year` slice (one group per ISO week present, 1→53).

### Conditions
| Condition key | Description |
|---|---|
| `any_period` | All resolved periods (the month/week-of-year slice does the seasonal breakdown) |

### Outcomes
| Outcome key | Channel | Description |
|---|---|---|
| `mean_volume` | `value` | Average summed RTH volume per period |

### Parameters
| Parameter | Default | Description |
|---|---|---|
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `monthly`, `weekly` (merged into one result file under
  `instruments.{INSTRUMENT}.{granularity}`).

### Baseline
Random null: for a subset of `N` periods (one month/week of year, or the whole
table), draw `N` periods uniformly at random **without replacement** from the full
period table and compute the same average on that random sample. This represents
the null hypothesis that the calendar position carries no information — its
expected volume equals the grand mean across all periods. Uses
`np.random.default_rng(seed)` for deterministic output. The baseline is computed
per slice group as well as overall; for the overall result the sample is a
permutation of the full table, so its `value_baseline` equals the overall `value`.

### i18n
- **title.en**: "Volume Trends"
- **title.fr**: "Tendances de volume"
- **definition.en**: "What is the average traded volume for each calendar month and each ISO week of the year?"
- **definition.fr**: "Quel est le volume échangé moyen pour chaque mois civil et chaque semaine ISO de l'année ?"

### Slices
- `month_of_year` — implemented (declared on the monthly granularity).
- `week_of_year` — implemented (declared on the weekly granularity).

Both slicers are reused from the Seasonality family
(`stats/seasonality/standard.py`); they read the period table's DatetimeIndex.

