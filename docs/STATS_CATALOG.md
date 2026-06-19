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


---

## 11. Previous Day's Range

**Family**: `prev_days_range`
**Module**: `stats/prev_days_range/standard.py`
**Result file**: `results/prev_days_range.json`
**Status**: Implemented

### What it measures
How often does intraday price break the prior trading day's range (its RTH high
or low)? And, given a break, does the session follow through by closing in the
break's direction? Two tiers of rows are reported: the **break frequency** over
all countable days, and the **directional follow-through** among the days that
broke each level.

### Methodology
1. Build the RTH daily candle per resolved day (`session_open`, `session_close`)
   using the same resolution rule as the other daily stats (clean 09:30 open and
   a last RTH bar at or after `session_end - close_tolerance`).
2. Compute per-day RTH extremes from the same RTH bar filter
   (`hour*60+minute >= rth_start_min` and `< rth_end_min`):
   - `day_high` = max of `high` over the day's RTH bars.
   - `day_low`  = min of `low` over the day's RTH bars.
   Join these onto the resolved-days index (inner join — only resolved dates).
3. Attach each day's prior-range reference from the chronologically **previous
   resolved** day (so it skips over any excluded/early-close day):
   - `prev_high` = previous resolved day's `day_high`.
   - `prev_low`  = previous resolved day's `day_low`.
   The first resolved day has no prior day, so `prev_high` / `prev_low` are NaN
   and it is excluded from every denominator (pending-sample discipline).
4. Classify each countable day:
   - **break_high**: `day_high > prev_high` (strict — touching is not a break).
   - **break_low**:  `day_low  < prev_low`.
   A single day may break both, one, or neither.
5. Classify each day's direction: **green** if `session_close >= session_open`,
   otherwise **red**.

### Conditions & outcomes
| Condition | Outcomes | Partition? | Description |
|---|---|---|---|
| `prior_range` | `break_high`, `break_low` | No (independent rates) | Break frequency over all countable days; `total` = countable days |
| `break_high` | `green`, `red` | Yes | Follow-through given a high break; `total` = high-break days. `green` = continuation (up) |
| `break_low` | `green`, `red` | Yes | Follow-through given a low break; `total` = low-break days. `red` = continuation (down) |

Follow-through here is the **session direction** (close vs open). The distinct
"closed outside vs back inside the prior range" classification is a separate
variant (see below), not computed by this standard module.

`total_samples` counts all resolved days; each row's `total` counts only the
relevant countable days (those with a prior resolved day).

### Parameters
| Parameter | Default | Description |
|---|---|---|
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `daily` (the RTH daily candle; a single entry).

### Baseline
Random null with two independent randomizations (fixed seed, deterministic):
- The prior-range pair (`prev_high`, `prev_low`) is **permuted together** across
  days, so each day is compared against an unrelated day's range — the null for
  break frequency (temporal adjacency carries no information). Permuting the pair
  jointly keeps each prior range internally consistent (`low <= high`); the lone
  NaN pair moves to a random day, preserving the countable count.
- `day_green` is reassigned by a fair coin (p=0.5), so direction is independent
  of any break — the null for follow-through (~0.5 for every break row).

### Slices
- `weekday` — the "by weekday" breakdown (declared via the shared `Weekday`
  slicer).
- `prev_candle` — the "by prev close" breakdown: prior session green/red
  (declared via the shared `PrevCandle` slicer reading `prev_session_green`).

### Future variants (not in MVP)
- `by_levels` — break extension measured in multiples of the prior day's range
  (would add an extension column and declare `Levels(ref='prev_range', ...)`).
- `by_outside_close` — a distinct *follow-through metric*: whether the session
  **closed outside** the broken level (close beyond prior high/low) versus **back
  inside** the prior range, rather than the green/red session direction used here.


---

## 12. Previous Week's Range

**Family**: `prev_weeks_range`
**Module**: `stats/prev_weeks_range/standard.py`
**Result file**: `results/prev_weeks_range.json`
**Status**: Implemented

### What it measures
How does a trading week resolve against the **prior week's** RTH range? Each
countable week falls into exactly one of four buckets — it breaks the prior
high only, the prior low only, **both**, or neither (**stays inside**). And when
both levels are taken, which one is reached **first** during the week?

### Methodology
1. Keep RTH bars only (`hour*60+minute >= rth_start_min` and `< rth_end_min`).
2. Assign each bar to its **ISO week**, keyed by that week's Monday date.
3. Build the per-week range from **every** trading day in the week (early-close
   days still contribute valid highs/lows):
   - `week_high` = max of `high` over the week's RTH bars.
   - `week_low`  = min of `low`  over the week's RTH bars.
4. Drop the **last** week present in the data: it may still be in progress, so
   its range is not final (pending-sample discipline). All earlier weeks are
   resolved.
5. Attach each week's prior-range reference from the chronologically **previous
   resolved** week (skips over any gap in the data):
   - `prev_high` = previous resolved week's `week_high`.
   - `prev_low`  = previous resolved week's `week_low`.
   The first resolved week has no prior week, so `prev_high` / `prev_low` are NaN
   and it is excluded from every denominator.
6. Classify each countable week against the prior range (strict inequality —
   touching the level is not a break):
   - **break_high_only**: `week_high > prev_high` and `week_low >= prev_low`.
   - **break_low_only**:  `week_low < prev_low` and `week_high <= prev_high`.
   - **break_both**:      both `week_high > prev_high` and `week_low < prev_low`.
   - **inside**:          neither level is taken.
7. For each both-break week, determine the **break sequence** from intra-week
   1-minute bars: the first bar (by timestamp) whose `high > prev_high` versus
   the first whose `low < prev_low`. The earlier one is taken first. If a single
   bar is the first to break **both**, the order is inferred from that bar's
   candle path — a bearish bar (`close < open`) prints its high first
   (`high_first`), a bullish bar its low first (`low_first`).

### Conditions & outcomes
| Condition | Outcomes | Partition? | Description |
|---|---|---|---|
| `prior_range` | `break_high_only`, `break_low_only`, `break_both`, `inside` | Yes | Four-way outcome partition over all countable weeks; `total` = countable weeks |
| `break_both` | `high_first`, `low_first` | Yes | Which prior level is taken first, given both broke; `total` = both-break weeks |

`total_samples` counts all resolved weeks; each row's `total` counts only the
relevant countable weeks (those with a prior resolved week).

### Parameters
None (the RTH session bounds come from the instrument config).

### Timeframes computed
- `weekly` (the RTH week, keyed by its Monday; a single entry).

### Baseline
Random null, computed per tier (fixed seed, deterministic):
- **Outcome partition**: the prior-range pair (`prev_high`, `prev_low`) is
  **permuted together** across weeks, so each week is compared against an
  unrelated week's range — the null for the partition (temporal adjacency carries
  no information). Permuting the pair jointly keeps each prior range internally
  consistent (`low <= high`).
- **Double-break sequence**: the real prior ranges are kept (so the both-break
  `total` stays full size) and only the break order is reassigned by a fair coin
  (p=0.5) — the null for which level is taken first (~0.5). Permuting prior ranges
  here would be wrong: over a long, trending history it dissolves almost every
  both-break, leaving a meaninglessly small baseline N.

### Slices
None. The standard report has no secondary breakdowns.

### Future variants (not in MVP)
- `by_open` — split weeks by their open above/below the prior week's midpoint.
- `by_outside_close` — split by whether the week **closed outside** the broken
  level versus back inside the prior range.


---

## 13. Outside Days

**Family**: `outside_days`
**Module**: `stats/outside_days/standard.py`
**Result file**: `results/outside_days.json`
**Status**: Implemented

### What it measures
How often does a session **open outside** the prior trading day's range — above
its RTH high (**bullish** outside) or below its RTH low (**bearish** outside)?
And, given such an open, does price **continue** away from the range (hold
outside all session) or **reverse** back into it during the session? Two tiers of
rows are reported: the **outside-open frequency** over all countable days, and
the **continuation vs reversal** split among the days that opened outside.

### Methodology
1. Build the RTH daily candle per resolved day (`session_open`, `session_close`)
   using the same resolution rule as the other daily stats (clean 09:30 open and
   a last RTH bar at or after `session_end - close_tolerance`).
2. Compute per-day RTH extremes from the same RTH bar filter
   (`hour*60+minute >= rth_start_min` and `< rth_end_min`):
   - `day_high` = max of `high` over the day's RTH bars.
   - `day_low`  = min of `low` over the day's RTH bars.
   Join these onto the resolved-days index (inner join — only resolved dates).
3. Attach each day's prior-range reference from the chronologically **previous
   resolved** day (so it skips over any excluded/early-close day):
   - `prev_high` = previous resolved day's `day_high`.
   - `prev_low`  = previous resolved day's `day_low`.
   The first resolved day has no prior day, so `prev_high` / `prev_low` are NaN
   and it is excluded from every denominator (pending-sample discipline).
4. Classify the outside open of each countable day (strict — opening exactly at
   the level is not outside; the two are mutually exclusive):
   - **bullish**: `session_open > prev_high`.
   - **bearish**: `session_open < prev_low`.
   Most days open inside the prior range and fall into neither.
5. For an outside open, classify continuation vs reversal (re-entry uses strict
   inequality, mirroring the strict-break convention of `prev_days_range`):
   - bullish: **reversal** if `day_low  <  prev_high` (price re-entered the
     range); **continuation** if `day_low  >= prev_high` (held above all session).
   - bearish: **reversal** if `day_high >  prev_low`; **continuation** if
     `day_high <= prev_low`.

### Conditions & outcomes
| Condition | Outcomes | Partition? | Description |
|---|---|---|---|
| `outside_open` | `bullish`, `bearish` | No (mutually exclusive, most days open inside) | Outside-open frequency over all countable days; `total` = countable days |
| `bullish` | `continuation`, `reversal` | Yes | Continuation vs reversal given a bullish outside open; `total` = bullish-open days |
| `bearish` | `continuation`, `reversal` | Yes | Continuation vs reversal given a bearish outside open; `total` = bearish-open days |

Continuation/reversal here is **intraday range re-entry** (did price trade back
into the prior range?). The distinct "where the close lands relative to the prior
high/low" classification is a separate variant (see below), not computed by this
standard module.

`total_samples` counts all resolved days; each row's `total` counts only the
relevant countable days (those with a prior resolved day).

### Parameters
| Parameter | Default | Description |
|---|---|---|
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `daily` (the RTH daily candle; a single entry).

### Baseline
Random null (fixed seed, deterministic): the prior-range pair (`prev_high`,
`prev_low`) is **permuted together** across days, so each day's open and intraday
extremes are compared against an **unrelated** day's range. This is the joint null
for both tiers at once — it destroys the temporal adjacency the stat measures
(does opening outside *yesterday's* range carry information?) while keeping each
prior range internally consistent (`low <= high`); the lone NaN pair moves to a
random day, preserving the countable count. Against a long, trending history the
baseline reversal rate is near zero (a random distant range rarely sits where
price can re-enter it), which makes the strong real-data reversal rate the signal.

### Slices
- `weekday` — the "by weekday" breakdown (declared via the shared `Weekday`
  slicer).
- `prev_candle` — split by the prior session's color, green/red (declared via the
  shared `PrevCandle` slicer reading `prev_session_green`).

### Future variants (not in MVP)
- `by_close` — where the close lands relative to the prior high/low.
- `by_size` — bucket outside days by gap size (open distance beyond the level).
- `by_spike` — maximum extension reached before a reversal.
- `by_time` — whether the reversal occurred before/after an intraday cutoff.


## 14. Inside Bars

**Family**: `inside_bars`
**Module**: `stats/inside_bars/standard.py`
**Result file**: `results/inside_bars.json`
**Status**: Implemented

### What it measures
How often does a session **open inside** the prior trading day's range — within its
RTH high-low range (inclusive)? And, given such an open, which direction does price
**break** during the session: above the prior high only, below the prior low only,
both sides, or does it stay **contained** within the prior range all session? Two
tiers of rows are reported: the **inside-open frequency** over all countable days,
and the **breakout-direction** breakdown among the days that opened inside. Inside
Bars is the conceptual inverse of `outside_days`.

### Methodology
1. Build the RTH daily candle per resolved day (`session_open`, `session_close`)
   using the same resolution rule as the other daily stats (clean 09:30 open and
   a last RTH bar at or after `session_end - close_tolerance`).
2. Compute per-day RTH extremes from the same RTH bar filter
   (`hour*60+minute >= rth_start_min` and `< rth_end_min`):
   - `day_high` = max of `high` over the day's RTH bars.
   - `day_low`  = min of `low` over the day's RTH bars.
   Join these onto the resolved-days index (inner join — only resolved dates).
3. Attach each day's prior-range reference from the chronologically **previous
   resolved** day (so it skips over any excluded/early-close day):
   - `prev_high` = previous resolved day's `day_high`.
   - `prev_low`  = previous resolved day's `day_low`.
   The first resolved day has no prior day, so `prev_high` / `prev_low` are NaN
   and it is excluded from every denominator (pending-sample discipline).
4. Classify the inside open of each countable day (inclusive — opening exactly at
   `prev_high` or `prev_low` counts as inside, the exact complement of the
   `outside_days` strict-outside definition):
   - **inside**: `prev_low <= session_open <= prev_high`.
5. For an inside open, classify the breakout direction (strict — touching a level
   exactly is not a break, mirroring the strict re-entry convention of
   `outside_days`). The four outcomes partition the inside set exhaustively:
   - **broke_high**: `day_high >  prev_high` AND `day_low  >= prev_low`.
   - **broke_low**:  `day_low  <  prev_low`  AND `day_high <= prev_high`.
   - **broke_both**: `day_high >  prev_high` AND `day_low  <  prev_low`.
   - **contained**:  `day_high <= prev_high` AND `day_low  >= prev_low`.

### Conditions & outcomes
| Condition | Outcomes | Partition? | Description |
|---|---|---|---|
| `inside_open` | `inside` | No (frequency of one event vs all countable days) | Inside-open frequency over all countable days; `total` = countable days |
| `inside` | `broke_high`, `broke_low`, `broke_both`, `contained` | Yes | Breakout direction given an inside open; `total` = inside-open days |

`total_samples` counts all resolved days; each row's `total` counts only the
relevant countable days (those with a prior resolved day).

### Parameters
| Parameter | Default | Description |
|---|---|---|
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `daily` (the RTH daily candle; a single entry).

### Baseline
Random null (fixed seed, deterministic): the prior-range pair (`prev_high`,
`prev_low`) is **permuted together** across days, so each day's open and intraday
extremes are compared against an **unrelated** day's range. This is the joint null
for both tiers at once — it destroys the temporal adjacency the stat measures (does
opening inside *yesterday's* range carry information?) while keeping each prior range
internally consistent (`low <= high`); the lone NaN pair moves to a random day,
preserving the countable count. Against a long, trending history a random distant
range rarely brackets the open, so the baseline inside-open rate collapses toward
zero — which makes the strong real-data inside-open rate the signal.

### Slices
- `weekday` — the "by weekday" breakdown (declared via the shared `Weekday`
  slicer).
- `prev_candle` — split by the prior session's color, green/red (declared via the
  shared `PrevCandle` slicer reading `prev_session_green`).

### Future variants (not in MVP)
- `by_breakout` — outcome classification refinements on the breakout move.
- `by_open` — split by open above/below the prior midpoint.
- `by_prev_day_size` — bucket inside days by the prior range relative to ADR.


---

## 15. Engulfing Candles

**Family**: `engulfing_candles`
**Module**: `stats/engulfing_candles/standard.py`
**Result file**: `results/engulfing_candles.json`
**Status**: Implemented (standard variant)

### What it measures
How often is the RTH daily candle a **bullish** or **bearish engulfing** pattern —
its **body** fully engulfs the prior resolved day's body, in the opposite color —
and, given such a pattern, how far does price **continue** in the pattern's
direction? Continuation is measured from the engulfing candle's **close** until the
pattern **invalidates** (price reclaims the engulfing candle's **open**), and is
reported as the **average** and **maximum** favorable excursion in percent. Two
tiers of rows are reported: the **engulfing frequency** over all countable days, and
the **continuation magnitude** among the resolved patterns of each direction.

### Methodology
1. Build the RTH daily candle per resolved day (`session_open`, `session_close`,
   plus `day_high` / `day_low`) using the same resolution rule as the other daily
   stats (clean 09:30 open and a last RTH bar at or after
   `session_end - close_tolerance`).
2. Attach each day's prior **body** from the chronologically **previous resolved**
   day (so it skips over any excluded/early-close day): `prev_open`, `prev_close`.
   The first resolved day has no prior day, so these are NaN and it is excluded from
   every denominator (pending-sample discipline).
3. Classify each countable day using candle **bodies** (`[min(o,c), max(o,c)]`,
   inclusive engulfment — an exactly-equal edge still engulfs):
   - **bullish**: current green (`close >= open`) AND prior red AND
     `max(o,c) >= prev_top` AND `min(o,c) <= prev_bot`.
   - **bearish**: current red (`close < open`) AND prior green AND the same body
     engulfment.
   The opposite-prior-color requirement is the canonical reversal-pattern
   definition.
4. Precompute each day's **continuation** over the full chronological table in its
   natural direction (up for a green candle, down for a red one): from the close,
   track the **maximum favorable excursion** forward until price reclaims the open
   (bullish: a later `day_low <= open`; bearish: a later `day_high >= open`), with
   the invalidation day's extreme included. Report it as a percent of the close,
   floored at zero. A pattern that never invalidates before the end of the data is
   **pending** and is excluded from the continuation magnitude. Continuation depends
   only on a candle's own body and forward path, so precomputing it makes slices and
   the baseline reuse it correctly.

### Conditions & outcomes
| Condition | Outcomes | Partition? | Description |
|---|---|---|---|
| `engulfing` | `bullish`, `bearish` | No (two mutually exclusive events vs all countable days) | Engulfing frequency over all countable days; `total` = countable days |
| `bullish` | `avg_continuation`, `max_continuation` | — (magnitude rows) | Continuation % over resolved bullish patterns; `total` = resolved bullish patterns |
| `bearish` | `avg_continuation`, `max_continuation` | — (magnitude rows) | Continuation % over resolved bearish patterns; `total` = resolved bearish patterns |

The magnitude rows carry their metric in `value` (random baseline in
`value_baseline`); the `probability` channel is left at `0.0`. `total_samples`
counts all resolved days; each row's `total` counts only the relevant
countable/resolved days.

### Parameters
| Parameter | Default | Description |
|---|---|---|
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `daily` (the RTH daily candle; a single entry).

### Baseline
Random null (fixed seed, deterministic): the prior-body pair (`prev_open`,
`prev_close`) is **permuted together** across days, so each day's body is compared
against an **unrelated** day's prior body. This is the joint null for both tiers at
once — it randomizes which days qualify as engulfing while leaving each day's own
body and precomputed continuation intact, isolating whether the engulfing criterion
selects days with abnormal forward continuation. Against a long, trending history a
random distant body rarely fully engulfs the current one, so the baseline engulfing
rate collapses toward zero — which makes the real-data engulfing rate the signal.

### Slices
- `weekday` — the "by weekday" breakdown (declared via the shared `Weekday`
  slicer).

### Future variants (not in MVP)
- `by_daily_candle` — 3-day sequences (engulf day color → next day color).
- `by_rr` — risk-reward follow-through buckets.
- `by_size` — body size as a percent of open, bucketed.


---

## 16. Average Daily Range (ADR)

**Family**: `adr`
**Module**: `stats/adr/standard.py`
**Result file**: `results/adr.json`
**Status**: Implemented (standard variant)

### What it measures
How often does a session's RTH high-to-low range exceed or respect the prior
session's period-N Average Daily Range (ADR)? Reported as two outcomes that
partition every countable session: **exceeded** (range > ADR) and **respected**
(range <= ADR).

### Methodology
1. Build the RTH daily candle per resolved session (`session_open`, `session_close`,
   `day_high`, `day_low`) using the standard resolution rule (clean 09:30 open and
   a last RTH bar at or after `session_end - close_tolerance`).
2. Compute the per-session RTH range: `day_range = day_high - day_low`.
3. Compute the prior-session ADR using a rolling mean with no lookahead:
   ```
   adr = day_range.rolling(period).mean().shift(1)
   ```
   The `shift(1)` guarantees that each session's ADR is the mean of the
   `period` sessions STRICTLY BEFORE it (ending with the prior session's range).
   The first `period` resolved sessions therefore have NaN `adr` and are
   excluded from every denominator (pending-sample discipline). `adr > 0` is
   also asserted defensively.
4. For each countable session classify the outcome:
   - **exceeded**: `day_range > adr` (strict — touching the ADR is respected).
   - **respected**: `day_range <= adr`.
   The two outcomes are mutually exclusive and exhaustive for all countable days.

`total_samples` counts all resolved sessions; each row's `total` counts only the
**countable** sessions (those with a valid prior-session ADR, i.e., at least
`period` sessions of history).

### Parameters
| Parameter | Default | Description |
|---|---|---|
| period | 14 | Number of prior sessions in the ADR rolling window |
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `daily` (the RTH daily candle; a single entry).

### Baseline
Random null (fixed seed, deterministic): the `adr` column is **permuted** across
all rows while `day_range` is held fixed. This pairs each session's actual range
with an unrelated session's ADR, destroying the temporal and regime link (e.g. a
volatile-regime ADR is randomly matched against a quiet day's range). The
permutation also moves any NaN `adr` values to random rows, preserving the
countable count exactly. Expected baseline: near 50% exceeded / 50% respected,
since a random ADR drawn from the same historical distribution is roughly as
likely to be above as below any given range value.

### i18n
- **title.en**: "Average Daily Range (ADR)"
- **title.fr**: "Range journalier moyen (ADR)"
- **definition.en**: "How often does a session's high-to-low range exceed or respect the prior session's period-N average daily range (ADR)?"
- **definition.fr**: "À quelle fréquence le range (plus haut – plus bas) d'une session dépasse-t-il ou respecte-t-il le range journalier moyen (ADR) sur N périodes de la session précédente ?"

### Slices
- `weekday` — the "by weekday" breakdown (declared via `slices = ("weekday",)`).

### Future variants (not in MVP)
- `by_extension` — how far above the ADR the session extended (in multiples of
  the ADR), bucketed into levels.
- `by_range_to_adr` — range / ADR ratio bucketed into quartiles.
- `by_weekday` — the weekday breakdown is already available via the declared
  `weekday` slice, but a dedicated variant could expose additional weekday-level
  metrics (e.g. average ADR utilisation per weekday).
- `by_streak` — how often the exceeded / respected outcome repeats on consecutive
  sessions.


---

## 17. Average True Range (ATR)

**Family**: `atr`
**Module**: `stats/atr/standard.py`
**Result file**: `results/atr.json`
**Status**: Implemented (standard variant)

### What it measures
How often does a session's **True Range** exceed or respect the prior session's
period-N Average True Range (ATR)? This is the gap-aware companion to **ADR**
(section 16): where ADR uses the simple high-to-low range, ATR's True Range also
accounts for the overnight gap by including the prior session's close. Reported as
two outcomes that partition every countable session: **exceeded** (true range >
ATR) and **respected** (true range <= ATR).

### Methodology
1. Build the RTH daily candle per resolved session (`session_open`, `session_close`,
   `day_high`, `day_low`) using the standard resolution rule (clean 09:30 open and
   a last RTH bar at or after `session_end - close_tolerance`).
2. Take each session's prior-session reference: `prev_close` = the chronologically
   **previous resolved** session's `session_close` (shifted over the resolved-only,
   sorted index, so it skips any excluded/early-close day).
3. Compute the per-session True Range:
   ```
   true_range = max(
     day_high - day_low,
     abs(day_high - prev_close),
     abs(day_low  - prev_close),
   )
   ```
   The first resolved session has no prior close, so its two gap terms drop out and
   its True Range falls back to `day_high - day_low` — the standard first-bar ATR
   convention (Wilder). It remains countable.
4. Compute the prior-session ATR using a rolling mean with no lookahead:
   ```
   atr = true_range.rolling(period).mean().shift(1)
   ```
   The `shift(1)` guarantees that each session's ATR is the mean of the `period`
   sessions STRICTLY BEFORE it. The first `period` resolved sessions therefore have
   NaN `atr` and are excluded from every denominator (pending-sample discipline).
   `atr > 0` is also asserted defensively.
5. For each countable session classify the outcome:
   - **exceeded**: `true_range > atr` (strict — touching the ATR is respected).
   - **respected**: `true_range <= atr`.
   The two outcomes are mutually exclusive and exhaustive for all countable days.

`total_samples` counts all resolved sessions; each row's `total` counts only the
**countable** sessions (those with a valid prior-session ATR, i.e., at least
`period` sessions of history).

### Parameters
| Parameter | Default | Description |
|---|---|---|
| period | 14 | Number of prior sessions in the ATR rolling window |
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `daily` (the RTH daily candle; a single entry).

### Baseline
Random null (fixed seed, deterministic): the `atr` column is **permuted** across
all rows while `true_range` is held fixed. This pairs each session's actual true
range with an unrelated session's ATR, destroying the temporal and regime link
(e.g. a volatile-regime ATR is randomly matched against a quiet day's true range).
The permutation also moves any NaN `atr` values to random rows, preserving the
countable count exactly. Expected baseline: near 50% exceeded / 50% respected,
since a random ATR drawn from the same historical distribution is roughly as
likely to be above as below any given true-range value.

### i18n
- **title.en**: "Average True Range (ATR)"
- **title.fr**: "Average True Range (ATR)"
- **definition.en**: "How often does a session's true range (high-low, accounting for the prior close gap) exceed or respect the prior session's period-N average true range (ATR)?"
- **definition.fr**: "À quelle fréquence le true range d'une session (plus haut – plus bas, en tenant compte du gap avec la clôture précédente) dépasse-t-il ou respecte-t-il l'average true range (ATR) sur N périodes de la session précédente ?"

### Slices
- `weekday` — the "by weekday" breakdown (declared via `slices = ("weekday",)`).

### Future variants (not in MVP)
- `by_extension` — how far above the ATR the session extended (in multiples of
  the ATR), bucketed into levels.
- `by_range_to_atr_by_weekday` — range / ATR ratio bucketed, split per weekday.
- `by_streak` — how often the exceeded / respected outcome repeats on consecutive
  sessions.
- `by_weekday` — a dedicated weekday variant (the weekday breakdown is already
  available via the declared `weekday` slice).


---

## 18. Open to Close Range

**Family**: `open_close_range`
**Module**: `stats/open_close_range/standard.py`
**Result file**: `results/open_close_range.json`
**Status**: Implemented (standard variant)

### What it measures
How often does a session close **within** a given percentage range of its open,
versus closing **outside** that range? Reported as two outcomes that partition
every resolved session: **within** (`oc_move_pct <= range_percentage`) and
**outside** (`oc_move_pct > range_percentage`).

### Methodology
1. Build the RTH daily candle per resolved session (`session_open`,
   `session_close`) using the standard resolution rule (clean 09:30 open and a
   last RTH bar at or after `session_end - close_tolerance`).
2. Measure the open-to-close move as a percent of the open:
   ```
   oc_move_pct = abs(session_close - session_open) / session_open * 100
   ```
3. Classify each session against the `range_percentage` threshold:
   - **within**: `oc_move_pct <= range_percentage` (boundary — a move exactly
     equal to the threshold counts as within).
   - **outside**: `oc_move_pct > range_percentage`.
   The two outcomes are mutually exclusive and exhaustive for all resolved days.

Every resolved session is countable — there is no warm-up window — so each row's
`total` equals `total_samples`.

### Parameters
| Parameter | Default | Description |
|---|---|---|
| range_percentage | 1.0 | Open-to-close band as a percent of the open |
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `daily` (the RTH daily candle; a single entry).

### Baseline
Random null (fixed seed, deterministic): the `session_close` column is
**permuted** across all rows while `session_open` is held fixed, and the
open-to-close move is recomputed against each session's own open. This pairs
every session's open with an unrelated session's close, destroying the
same-session open-to-close link. The permutation preserves the countable total
exactly. Expected baseline: a random historical close lands within the
(typically tight) `range_percentage` band of a given session's open far less
often than the session's own close does, so the within-rate is much lower than
the actual — confirming the same-session move is genuinely contained rather than
an artifact of the band width.

### i18n
- **title.en**: "Open to Close Range"
- **title.fr**: "Range ouverture-clôture"
- **definition.en**: "How often does a session close within a given percentage range of its open?"
- **definition.fr**: "À quelle fréquence une session clôture-t-elle dans une plage de pourcentage donnée par rapport à son ouverture ?"

### Slices
- `weekday` — the "by weekday" breakdown.
- `close` — the "by session color" breakdown (green / red), which exposes the
  `day_type` dimension (declared via `slices = ("weekday", "close")`).

### Future variants (not in MVP)
- `by_day_type` — the green / red breakdown is already available via the declared
  `close` slice, but a dedicated variant could expose additional per-color
  metrics.
- `by_streak` — how often the within / outside outcome repeats on consecutive
  sessions.


---

## 19. Session Reversal Range

**Family**: `session_reversal_range`
**Module**: `stats/session_reversal_range/standard.py`
**Result file**: `results/session_reversal_range.json`
**Status**: Implemented (standard variant)

### What it measures
How far does a session move **against its eventual direction** before it closes —
the adverse intraday excursion from the session open? For a **green** session
(closes at or above its open) this is the open-to-low move; for a **red** session
(closes below its open) it is the open-to-high move. This is a **magnitude** stat:
each outcome carries its metric in `value` (random baseline in `value_baseline`)
and the `probability` channel is left at `0.0`.

### Methodology
1. Build the RTH daily candle per resolved session (`session_open`,
   `session_close`) using the standard resolution rule (clean 09:30 open and a
   last RTH bar at or after `session_end - close_tolerance`), and join the RTH
   intraday extremes `day_high` / `day_low`.
2. Compute the two excursions from the open (both non-negative, since the open
   always lies within `[day_low, day_high]`):
   ```
   down_excursion = session_open - day_low
   up_excursion   = day_high - session_open
   ```
3. Classify the session color (`session_green = session_close >= session_open`)
   and select the adverse side:
   ```
   reversal     = down_excursion if session_green else up_excursion
   reversal_pct = reversal / session_open        # a decimal of the open
   ```
4. Report four outcomes over the (possibly sliced) day set:
   - **mean_reversal** — average reversal range in points.
   - **mean_reversal_pct** — average reversal range as a decimal of the open.
   - **max_reversal** — maximum reversal range in points.
   - **max_reversal_pct** — maximum reversal range as a decimal of the open.

Every resolved session is countable — there is no warm-up window — so each row's
`count` / `total` equals `total_samples`.

### Parameters
| Parameter | Default | Description |
|---|---|---|
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `daily` (the RTH daily candle; a single entry).

### Baseline
Random null (fixed seed, deterministic): for each session the adverse side is
chosen by a **fair coin** — the open-to-low or the open-to-high excursion,
independent of the actual close direction — and the same averages / maxima are
computed on that random-side reversal. This is the null hypothesis that the
session's direction carries no information about which side is adverse. Expected
baseline: the true adverse excursion (the side opposite the close) tends to be
smaller than a randomly chosen side, so the actual mean reversal is **below** the
baseline — confirming the directional containment is real rather than an artifact
of intraday range. The coin flip randomizes within the overall table and within
every slice (including the green-only / red-only `close` groups).

### i18n
- **title.en**: "Session Reversal Range"
- **title.fr**: "Amplitude de retournement de session"
- **definition.en**: "How far does a session move against its eventual direction (open-to-low for green sessions, open-to-high for red sessions) before it closes?"
- **definition.fr**: "De combien une session évolue-t-elle à contre-sens de sa direction finale (ouverture-bas pour les sessions vertes, ouverture-haut pour les rouges) avant de clôturer ?"

### Slices
- `weekday` — the "by weekday" breakdown.
- `close` — the "by session color" breakdown (green / red), the central
  green-vs-red split of the report (declared via `slices = ("weekday", "close")`).

### Future variants (not in MVP)
- `by_weekday` — the per-weekday reversal magnitude is already available via the
  declared `weekday` slice; a dedicated variant could expose additional per-day
  detail.


---

## 20. SMA Performance

**Family**: `sma_performance`
**Module**: `stats/sma_performance/standard.py`
**Result file**: `results/sma_performance.json`
**Status**: Implemented (standard variant)

### What it measures
When price crosses above or below the N-period simple moving average (SMA) of
session closes, **how long does the move last** (duration, in bars) and **how far
does price travel** from the SMA before crossing back (travel, in points)? This is
a **magnitude** stat: each outcome carries its metric in `value` (random baseline
in `value_baseline`) and the `probability` channel is left at `0.0`. Reported per
direction — **cross_up** (close above the SMA) and **cross_down** (close on or
below the SMA).

### Methodology
1. Build the RTH daily candle per resolved session (`session_close`) using the
   standard resolution rule (clean 09:30 open and a last RTH bar at or after
   `session_end - close_tolerance`).
2. Compute the prior-window SMA with no lookahead and the signed distance from it:
   ```
   sma  = session_close.rolling(period).mean().shift(1)
   dist = session_close - sma
   ```
   The `shift(1)` makes each session's SMA the mean of the `period` sessions
   STRICTLY BEFORE it (identical to the ATR convention). The first `period`
   resolved sessions have NaN `sma` and are excluded from the dist sequence
   (pending-sample discipline).
3. A **move** is a maximal run of consecutive sessions with the same sign of
   `dist`: `up` when `dist > 0`, `down` when `dist <= 0` (a close exactly on the
   SMA goes to `down`, for determinism).
4. A run is a **confirmed cross event** only when it is both preceded and followed
   by an opposite-regime run (it has genuinely crossed in *and* back out). The
   first run (no prior regime → not a cross) and the last run (has not yet crossed
   back → pending) are excluded.
5. Among confirmed runs, keep only those with `duration >= min_duration` sessions,
   then for each run compute:
   - `duration` = number of bars (sessions) in the run.
   - `travel` = peak favorable excursion in points: `max(dist)` over the run for
     an up-move, `max(-dist)` for a down-move — i.e. the furthest the close got
     from the SMA in the move's direction.
6. Report four outcomes per direction over the qualifying runs:
   - **avg_duration** — average move length in bars.
   - **max_duration** — longest move length in bars.
   - **avg_travel** — average peak excursion in points.
   - **max_travel** — maximum peak excursion in points.

All 8 rows (2 conditions × 4 outcomes) are always emitted; when a direction has no
qualifying run its rows carry `count = total = 0` and `value = 0.0`. Each row's
`count` / `total` equals the number of qualifying confirmed runs of that direction
(note: SMA crosses are rare events — a long-period SMA over years of data yields
only a few dozen crosses, so N is naturally small and far below the usual N > 100
significance bar).

### Parameters
| Parameter | Default | Description |
|---|---|---|
| period | 200 | Number of prior sessions in the SMA rolling window |
| min_duration | 5 | Minimum run length (bars) to count as a qualifying cross event |
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `daily` (the RTH daily candle; a single entry).

### Baseline
Random null (fixed seed, deterministic): the `dist` series is **permuted** across
all rows, then the same run extraction, `min_duration` filter and aggregation are
applied to the shuffled sequence. Permutation destroys the serial autocorrelation
that produces persistent trends, so the baseline reflects the durations and travel
one would see if daily distances from the SMA were i.i.d. (no memory). Actual
move durations well above the baseline confirm genuine trend persistence on the
side of the SMA rather than an artifact of the distance distribution.

### i18n
- **title.en**: "SMA Performance"
- **title.fr**: "Performance SMA"
- **definition.en**: "When price crosses above or below the N-period simple moving average, how long does the move last (in bars) and how far does price travel from the SMA before crossing back?"
- **definition.fr**: "Lorsque le prix croise au-dessus ou en dessous de la moyenne mobile simple sur N périodes, combien de temps dure le mouvement (en barres) et jusqu'où le prix s'éloigne-t-il de la SMA avant de retraverser ?"

### Slices
- None. A move spans multiple sessions of variable length, so the per-day slicers
  (weekday, close color, …) do not apply; `slices = ()`.

### Future variants (not in MVP)
- `by_weekday` — bucketing cross events by the weekday on which the cross occurred.
- Percent-based travel (excursion as a fraction of price) alongside the point
  metric, mirroring the dual point/percent reporting of other magnitude stats.


---

## 21. CPI Performance

**Family**: `cpi_performance`
**Module**: `stats/cpi_performance/standard.py`
**Result file**: `results/cpi_performance.json`
**Status**: Implemented (standard variant)

### What it measures
Average close-to-close percent return across three windows around each CPI
release: the run-up before the release, the release day itself, and the
follow-through after.

### Methodology
1. Build the RTH resolved-day table (shared `build_resolved_days`): one row per
   resolved session, sorted chronologically, with `session_close`.
2. CPI release dates come from an **external economic calendar CSV** (not from
   the OHLCV data). `load_cpi_release_dates` keeps the headline monthly US print
   (`event == "CPI m/m"`, `currency == "USD"`) and returns the distinct release
   dates. They are injected into the stat, so the computation stays a pure
   function of (candles, release dates).
3. Each release date `D` is mapped to its trading session in the resolved-day
   sequence. Releases on a non-resolved session (holiday / early close / outside
   the data range) are dropped. Window offsets are positional over resolved
   days, so "N trading days before/after" naturally skips non-trading days.
4. With `c[i]` the session close at resolved position `i` (the release sits at
   position `i`), `pre = pre_announcement`, `post = post_announcement`, the three
   windows use **adjacent closes** (contiguous, non-overlapping):
   - **pre_announcement**: `(c[i-1] - c[i-pre]) / c[i-pre]` — the run-up baseline.
   - **cpi_day**: `(c[i] - c[i-1]) / c[i-1]` — the release-day move.
   - **post_announcement**: `(c[i+post] - c[i]) / c[i]` — the follow-through.
5. For each window, report three outcomes over its observations:
   - **mean_return** — average signed decimal return (magnitude in `value`).
   - **green** — share of observations with return `>= 0` (probability channel).
   - **red** — share with return `< 0`.

A window observation is **green** when its return is `>= 0`, else **red**.
Pending discipline is **per window**: an observation counts only when every
close that window needs exists, so the earliest releases drop out of the pre
window and the most recent releases drop out of the (still-unresolved) post
window. Each window therefore reports its own `count` / `total`. All 9 rows
(3 windows × 3 outcomes) are always emitted; an empty window carries
`count = total = 0` and `value = 0.0`.

### Parameters
| Parameter | Default | Description |
|---|---|---|
| pre_announcement | 5 | Trading sessions before the release in the pre window |
| post_announcement | 5 | Trading sessions after the release in the post window |
| calendar_path | `data/forex_factory_calendar.csv` | Economic calendar CSV with CPI release dates |
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `daily` (the RTH daily candle; a single entry).

### Baseline
Random null (fixed seed, deterministic): per window, each observation's return
magnitude is held fixed and only its sign is randomized (p=0.5). The expected
average return is ~0 and the expected green/red split ~50/50, so an actual mean
return or green share well away from the baseline indicates a genuine directional
tendency around CPI rather than an artifact of the return distribution.

### i18n
- **title.en**: "CPI Performance"
- **title.fr**: "Performance CPI"
- **definition.en**: "What is the average close-to-close percent return in the trading sessions before a CPI release, on the release day itself, and in the sessions after?"
- **definition.fr**: "Quel est le rendement moyen en pourcentage de clôture à clôture lors des séances précédant une publication du CPI, le jour de la publication, et lors des séances suivantes ?"

### Slices
- None. CPI releases are scattered across the calendar and each window spans
  several sessions, so the per-day slicers (weekday, close color, …) do not
  apply; `slices = ()`.


---

## 22. NFP Performance

**Family**: `nfp_performance`
**Module**: `stats/nfp_performance/standard.py`
**Result file**: `results/nfp_performance.json`
**Status**: Implemented (standard variant)

### What it measures
Average close-to-close percent return across three windows around each Non-Farm
Payrolls (NFP) release: the run-up before the release, the release day itself,
and the follow-through after.

### Methodology
1. Build the RTH resolved-day table (shared `build_resolved_days`): one row per
   resolved session, sorted chronologically, with `session_close`.
2. NFP release dates come from an **external economic calendar CSV** (not from
   the OHLCV data). `load_nfp_release_dates` keeps the headline monthly US
   payrolls print (`event == "Non-Farm Employment Change"`, `currency == "USD"`)
   and returns the distinct release dates. They are injected into the stat, so
   the computation stays a pure function of (candles, release dates).
3. Each release date `D` is mapped to its trading session in the resolved-day
   sequence. Releases on a non-resolved session (holiday / early close / outside
   the data range) are dropped. Window offsets are positional over resolved
   days, so "N trading days before/after" naturally skips non-trading days.
4. With `c[i]` the session close at resolved position `i` (the release sits at
   position `i`), `pre = pre_announcement`, `post = post_announcement`, the three
   windows use **adjacent closes** (contiguous, non-overlapping):
   - **pre_announcement**: `(c[i-1] - c[i-pre]) / c[i-pre]` — the run-up baseline.
   - **nfp_day**: `(c[i] - c[i-1]) / c[i-1]` — the release-day move.
   - **post_announcement**: `(c[i+post] - c[i]) / c[i]` — the follow-through.
5. For each window, report three outcomes over its observations:
   - **mean_return** — average signed decimal return (magnitude in `value`).
   - **green** — share of observations with return `>= 0` (probability channel).
   - **red** — share with return `< 0`.

A window observation is **green** when its return is `>= 0`, else **red**.
Pending discipline is **per window**: an observation counts only when every
close that window needs exists, so the earliest releases drop out of the pre
window and the most recent releases drop out of the (still-unresolved) post
window. Each window therefore reports its own `count` / `total`. All 9 rows
(3 windows × 3 outcomes) are always emitted; an empty window carries
`count = total = 0` and `value = 0.0`.

### Parameters
| Parameter | Default | Description |
|---|---|---|
| pre_announcement | 5 | Trading sessions before the release in the pre window |
| post_announcement | 5 | Trading sessions after the release in the post window |
| calendar_path | `data/forex_factory_calendar.csv` | Economic calendar CSV with NFP release dates |
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `daily` (the RTH daily candle; a single entry).

### Baseline
Random null (fixed seed, deterministic): per window, each observation's return
magnitude is held fixed and only its sign is randomized (p=0.5). The expected
average return is ~0 and the expected green/red split ~50/50, so an actual mean
return or green share well away from the baseline indicates a genuine directional
tendency around NFP rather than an artifact of the return distribution.

### i18n
- **title.en**: "NFP Performance"
- **title.fr**: "Performance NFP"
- **definition.en**: "What is the average close-to-close percent return in the trading sessions before an NFP release, on the release day itself, and in the sessions after?"
- **definition.fr**: "Quel est le rendement moyen en pourcentage de clôture à clôture lors des séances précédant une publication du NFP, le jour de la publication, et lors des séances suivantes ?"

### Slices
- None. NFP releases are scattered across the calendar and each window spans
  several sessions, so the per-day slicers (weekday, close color, …) do not
  apply; `slices = ()`.

