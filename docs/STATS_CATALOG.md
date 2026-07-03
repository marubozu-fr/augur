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
| `Overnight` | `'overnight'` | `column='overnight_green'` | green / red (open above / below prior close) |
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
**Module**: `stats/opening_candle/standard.py`
**Result file**: `results/opening_candle_continuation.json`
**Status**: Implemented

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
- `weekday` — day-of-week breakdown (declared via the `"weekday"` shorthand).
- `close` — where the session **closes relative to the opening candle range**
  (wick extremes): `above` (close > opening high), `inside` (close within the
  opening high/low), or `below` (close < opening low). This is a distinct metric,
  **not** the generic `Close` slicer (which merely splits days into session-green
  / session-red); it quantifies how far the session travels from the opening
  candle. Driven by the `close_location` column and the stat-local
  `_CloseLocation` slicer.
- `size` — opening-candle **body size** buckets, `|opening_close - opening_open|`,
  split into equal-frequency quartiles via
  `SizeBucket(column="opening_body", preset="quartiles", name="size")`. Small vs
  large opening moves are compared for differing continuation behaviour. (We use
  data-driven quartiles rather than hardcoded percentage thresholds, per the
  no-hardcoded-instrument-values rule.)


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



## 23. FOMC Performance

**Family**: `fomc_performance`
**Module**: `stats/fomc_performance/standard.py`
**Result file**: `results/fomc_performance.json`
**Status**: Implemented (standard variant)

### What it measures
Average close-to-close percent return across three windows around each FOMC rate
decision: the run-up before the decision, the decision day itself, and the
follow-through after.

### Methodology
1. Build the RTH resolved-day table (shared `build_resolved_days`): one row per
   resolved session, sorted chronologically, with `session_close`.
2. FOMC decision dates come from an **external economic calendar CSV** (not from
   the OHLCV data). `load_fomc_release_dates` keeps the headline US rate decision
   (`event == "Federal Funds Rate"`, `currency == "USD"`) and returns the distinct
   decision dates. The co-occurring "FOMC Statement" / "FOMC Press Conference"
   lines on the same date are deliberately not matched. They are injected into the
   stat, so the computation stays a pure function of (candles, decision dates).
3. Each decision date `D` is mapped to its trading session in the resolved-day
   sequence. Decisions on a non-resolved session (holiday / early close / outside
   the data range) are dropped. Window offsets are positional over resolved
   days, so "N trading days before/after" naturally skips non-trading days.
4. With `c[i]` the session close at resolved position `i` (the decision sits at
   position `i`), `pre = pre_announcement`, `post = post_announcement`, the three
   windows use **adjacent closes** (contiguous, non-overlapping):
   - **pre_announcement**: `(c[i-1] - c[i-pre]) / c[i-pre]` — the run-up baseline.
   - **fomc_day**: `(c[i] - c[i-1]) / c[i-1]` — the decision-day move.
   - **post_announcement**: `(c[i+post] - c[i]) / c[i]` — the follow-through.
5. For each window, report three outcomes over its observations:
   - **mean_return** — average signed decimal return (magnitude in `value`).
   - **green** — share of observations with return `>= 0` (probability channel).
   - **red** — share with return `< 0`.

A window observation is **green** when its return is `>= 0`, else **red**.
Pending discipline is **per window**: an observation counts only when every
close that window needs exists, so the earliest decisions drop out of the pre
window and the most recent decisions drop out of the (still-unresolved) post
window. Each window therefore reports its own `count` / `total`. All 9 rows
(3 windows × 3 outcomes) are always emitted; an empty window carries
`count = total = 0` and `value = 0.0`.

### Parameters
| Parameter | Default | Description |
|---|---|---|
| pre_announcement | 5 | Trading sessions before the decision in the pre window |
| post_announcement | 5 | Trading sessions after the decision in the post window |
| calendar_path | `data/forex_factory_calendar.csv` | Economic calendar CSV with FOMC decision dates |
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `daily` (the RTH daily candle; a single entry).

### Baseline
Random null (fixed seed, deterministic): per window, each observation's return
magnitude is held fixed and only its sign is randomized (p=0.5). The expected
average return is ~0 and the expected green/red split ~50/50, so an actual mean
return or green share well away from the baseline indicates a genuine directional
tendency around the FOMC decision rather than an artifact of the return
distribution.

### i18n
- **title.en**: "FOMC Performance"
- **title.fr**: "Performance FOMC"
- **definition.en**: "What is the average close-to-close percent return in the trading sessions before an FOMC rate decision, on the decision day itself, and in the sessions after?"
- **definition.fr**: "Quel est le rendement moyen en pourcentage de clôture à clôture lors des séances précédant une décision de taux du FOMC, le jour de la décision, et lors des séances suivantes ?"

### Slices
- None. FOMC decisions are scattered across the calendar and each window spans
  several sessions, so the per-day slicers (weekday, close color, …) do not
  apply; `slices = ()`.


---

## 24. Gap Fill

**Family**: `gap_fill`
**Module**: `stats/gap_fill/standard.py`
**Result file**: `results/gap_fill.json`
**Status**: Implemented (standard variant)

### What it measures
When a session opens with a gap up or down from the prior session's close, how
often does intraday RTH price retrace back through a configurable percentage of
that gap and **fill** it? Reported as a 2x2 conditional matrix: P(fill | gap up),
P(no fill | gap up), P(fill | gap down), P(no fill | gap down). The two outcomes
partition each gap direction, so they sum to 1.

### Methodology
1. Build the RTH daily candle per resolved session (`session_open`,
   `session_close`) using the same resolution rule as the other daily stats (clean
   09:30 open and a last RTH bar at or after `session_end - close_tolerance`).
2. Compute per-day RTH extremes from the same RTH bar filter
   (`hour*60+minute >= rth_start_min` and `< rth_end_min`):
   - `day_high` = max of `high` over the day's RTH bars.
   - `day_low`  = min of `low` over the day's RTH bars.
   Join these onto the resolved-days index (inner join — only resolved dates).
3. Take each session's `prev_session_close` = the chronologically **previous
   resolved** session's `session_close` (shifted over the resolved-only, sorted
   index, so it skips any excluded/early-close day). The first resolved session has
   no prior close, so its gap is undefined and it is excluded from every
   denominator (pending-sample discipline).
4. Classify the **gap** from `session_open` vs `prev_session_close` (strict — an
   open exactly at the prior close is **no gap** and is excluded):
   - **gap_up**:   `session_open > prev_session_close`.
   - **gap_down**: `session_open < prev_session_close`.
   The gap distance is `gap_size = abs(session_open - prev_session_close)`.
5. Classify the **fill**: intraday price retraces from the open back toward the
   prior close by at least `fill_threshold_pct` percent of the gap. The target is
   measured from the open toward the prior close:
   - gap up:   `target = session_open - (pct/100) * gap_size`; **filled** when
     `day_low <= target`.
   - gap down: `target = session_open + (pct/100) * gap_size`; **filled** when
     `day_high >= target`.
   At the default `pct = 100` a full fill means price trades all the way back to the
   prior close (gap up → `day_low <= prev_session_close`; gap down →
   `day_high >= prev_session_close`).

`total_samples` counts all resolved sessions; each row's `total` counts only the
**countable** sessions (those with a prior close and a non-zero gap) carrying that
gap direction.

### Conditions & outcomes
| Condition | Outcomes | Partition? | Description |
|---|---|---|---|
| `gap_up` | `filled`, `not_filled` | Yes | Fill rate given an up gap; `total` = up-gap days |
| `gap_down` | `filled`, `not_filled` | Yes | Fill rate given a down gap; `total` = down-gap days |

### Parameters
| Parameter | Default | Description |
|---|---|---|
| fill_threshold_pct | 100.0 | Fraction of the gap that must be retraced to count as filled (100 = full fill) |
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `daily` (the RTH daily candle; a single entry).

### Baseline
Random null (fixed seed, deterministic): the `filled` outcome is **permuted**
across countable days, so the overall fill rate is preserved but its association
with the gap direction is destroyed. Each condition's `baseline_prob` therefore
converges to the pooled fill rate, and the comparison reveals whether gap up and
gap down fill at *different* rates than the market does on average. A fair-coin
baseline would instead anchor to 0.5, which is not the relevant null for an event
that fills well above half the time. Uses `np.random.default_rng(seed)`.

### i18n
- **title.en**: "Gap Fill"
- **title.fr**: "Comblement de gap"
- **definition.en**: "When a session opens with a gap up or down from the prior session's close, how often does intraday price retrace back through the gap and fill it?"
- **definition.fr**: "Lorsqu'une session ouvre avec un gap à la hausse ou à la baisse par rapport à la clôture de la session précédente, à quelle fréquence le prix intraday revient-il combler le gap ?"

### Slices
- `weekday` — the "by weekday" breakdown (declared via the shared `Weekday` slicer).
- `close` — the "by close" breakdown: session close color, green/red (declared via
  the shared `Close` slicer reading `session_green`).
- `prev_candle` — the "by prev candle" breakdown: prior session color, green/red
  (declared via the shared `PrevCandle` slicer reading `prev_session_green`).
- `size_pts` — the "by size" breakdown in points: gap size quartiles (declared via
  `SizeBucket(column="gap_size_pts")`).
- `size_pct` — the "by size" breakdown in percent of the prior close: gap size
  quartiles (declared via `SizeBucket(column="gap_size_pct")`). `gap_size_pct` is
  stored in percent units (not a decimal) so the bucket-edge labels are legible.

### Future variants (not in MVP)
- `by_fill_time` — split filled days by whether the fill occurred before/after an
  intraday cutoff minute (needs the per-day fill timestamp from the intraday path).
- `by_spike` — bucket days by the maximum spike **against** the gap direction
  before the fill (needs the intraday path, not just the day extremes).


---

## 25. Opening Range Breakout

**Family**: `opening_range_breakout`
**Module**: `stats/opening_range_breakout/standard.py`
**Result file**: `results/opening_range_breakout.json`
**Status**: Implemented (standard variant)

### What it measures
The opening range is the high-low range of the first N minutes of the RTH session.
Given that range, which direction does price break during the **rest** of the
session? Reported under a single `orb` condition with four mutually exclusive
outcomes that partition every countable day: `broke_high`, `broke_low`,
`broke_both`, `neither`. Their probabilities sum to 1.

### Methodology
1. Build the RTH daily candle per resolved session (`session_open`,
   `session_close`) using the same resolution rule as the other daily stats (clean
   09:30 open and a last RTH bar at or after `session_end - close_tolerance`).
2. Compute the **opening range** from the bars in the ORB window
   `[rth_start, rth_start + orb_period)`:
   - `orb_high` = max of `high`, `orb_low` = min of `low` over the window.
   - `orb_size` = `orb_high - orb_low`.
3. Compute the **breakout window** extremes from the rest of the session,
   `[rth_start + orb_period, rth_end)` (the opening range itself is excluded so it
   can never break its own bars):
   - wick criteria (default): `post_high` = max `high`, `post_low` = min `low`.
   - close criteria: `post_close_high` = max `close`, `post_close_low` = min `close`.
   Join these onto the resolved-days index (inner join — a day survives only with a
   full opening range AND a non-empty breakout window).
4. Classify the breakout direction with **strict** inequalities (touching a level
   exactly is NOT a break), using the extremes selected by `breakout_criteria`:
   - `broke_high`: `high > orb_high` AND `low >= orb_low` (broke up only).
   - `broke_low`:  `low < orb_low`  AND `high <= orb_high` (broke down only).
   - `broke_both`: `high > orb_high` AND `low < orb_low` (broke both sides).
   - `neither`:    `high <= orb_high` AND `low >= orb_low` (held inside all session).

`total_samples` counts all resolved sessions; each row's `total` is the
**countable** days (a clean opening range and a non-empty breakout window). The
four outcomes are mutually exclusive and exhaustive, so their counts sum to `total`.

### Conditions & outcomes
| Condition | Outcomes | Partition? | Description |
|---|---|---|---|
| `orb` | `broke_high`, `broke_low`, `broke_both`, `neither` | Yes | Breakout direction relative to the opening range; `total` = countable days |

### Parameters
| Parameter | Default | Description |
|---|---|---|
| timeframe | 15min | Opening-range length: 15min, 30min, or 1h (emitted as separate timeframe entries) |
| breakout_criteria | wick | Break detection: `wick` (intraday extreme) or `close` (bar close beyond the level) |
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `15min` (opening range 09:30–09:45).
- `30min` (opening range 09:30–10:00).
- `1h` (opening range 09:30–10:30).

### Baseline
Random directional null (fixed seed, deterministic): each countable day's breakout
move is **reflected** around its opening-range midpoint with probability 0.5 (a
price `p` maps to `2*mid - p`, which swaps the high and low extremes). Reflection
turns a `broke_high` day into a `broke_low` day and vice versa, while `broke_both`
and `neither` are direction-symmetric and unchanged. The null therefore carries NO
directional bias, so `broke_high` and `broke_low` converge to their shared mean and
the comparison reveals whether the opening range breaks **up** more often than
**down** beyond a coin flip. Uses `np.random.default_rng(seed)`.

### i18n
- **title.en**: "Opening Range Breakout"
- **title.fr**: "Cassure du range d'ouverture"
- **definition.en**: "After the opening range forms from the first N minutes of the session, how often does price break above the range high, below the range low, both sides, or neither during the rest of the session?"
- **definition.fr**: "Après la formation du range d'ouverture sur les N premières minutes de la session, à quelle fréquence le prix casse-t-il au-dessus du haut du range, en-dessous du bas du range, des deux côtés, ou ni l'un ni l'autre pendant le reste de la séance ?"

### Slices
- `weekday` — the "by weekday" breakdown (declared via the shared `Weekday` slicer).
- `close` — the "by close" breakdown: session close color, green/red (declared via
  the shared `Close` slicer reading `session_green`).
- `prev_candle` — the "by prev candle" breakdown: prior session color, green/red
  (declared via the shared `PrevCandle` slicer reading `prev_session_green`).
- `size` — the "by size" breakdown: opening-range size quartiles (declared via
  `SizeBucket(column="orb_size")`).

### Future variants (not in MVP)
- `by_levels` — extension in multiples of the ORB size beyond the broken level.
- `by_performance` — average/max extension before a breakback.
- `by_rejection` — tags the ORB high/low then closes back inside.
- `by_retracement` — pullback depth after the break.
- `by_time` — breakout before/after an intraday threshold.


---

## 26. Opening Range Indicator

**Family**: `opening_range_indicator`
**Module**: `stats/opening_range_indicator/standard.py`
**Result file**: `results/opening_range_indicator.json`
**Status**: Implemented (standard variant)

### What it measures
How big is the opening range (the high-low of the first N minutes of the RTH
session) relative to the range of the **rest** of the session, and how strongly are
the two correlated? A **magnitude** stat: a single `opening_range` condition reports
four value-channel outcomes (the `probability` channel is left at `0.0`):
`mean_opening_range`, `mean_remaining_range`, `opening_range_share`, and
`correlation`.

### Methodology
1. Build the RTH daily candle per resolved session (`session_open`,
   `session_close`) using the same resolution rule as the other daily stats (clean
   09:30 open and a last RTH bar at or after `session_end - close_tolerance`).
2. Compute the **opening range** from the bars in the opening window
   `[rth_start, rth_start + orb_period)`:
   `opening_range = max(high) - min(low)` over the window.
3. Compute the **remaining range** from the rest of the session,
   `[rth_start + orb_period, rth_end)` (the opening window is excluded so the two
   ranges never share bars): `remaining_range = max(high) - min(low)`.
   Join both onto the resolved-days index (inner join — a day survives only with a
   full opening range AND a non-empty remaining window).
4. The opening and remaining windows exactly partition the RTH session, so the
   full **session range** is `max(orb_high, rest_high) - min(orb_low, rest_low)`.
5. Reduce the (sliced) day table to the four metrics:
   - `mean_opening_range` = average `opening_range` in points.
   - `mean_remaining_range` = average `remaining_range` in points.
   - `opening_range_share` = average per-day `opening_range / session_range`
     (a decimal in [0, 1], over days with a positive session range).
   - `correlation` = Pearson r between `opening_range` and `remaining_range` across
     days (in [-1, 1]; reported as `0.0` when undefined — fewer than two days or
     zero variance).

`total_samples` counts all resolved sessions; each row's `count` / `total` equals
the countable-day count (a clean opening range and a non-empty remaining window).

### Conditions & outcomes
| Condition | Outcomes | Channel | Description |
|---|---|---|---|
| `opening_range` | `mean_opening_range`, `mean_remaining_range`, `opening_range_share`, `correlation` | `value` | Size of the opening range vs the remaining-session range, and their correlation |

### Parameters
| Parameter | Default | Description |
|---|---|---|
| timeframe | 15min | Opening-range length: 15min, 30min, or 1h (emitted as separate timeframe entries) |
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `15min` (opening range 09:30–09:45).
- `30min` (opening range 09:30–10:00).
- `1h` (opening range 09:30–10:30).

### Baseline
Random permutation null (fixed seed, deterministic): the `remaining_range` column
is permuted across days while `opening_range` and `session_range` are held fixed,
pairing each day's opening range with an unrelated day's remaining range. This
destroys the same-day link that `correlation` measures, so the baseline correlation
collapses toward 0 — the headline test is whether the actual correlation sits well
above its permuted null. `mean_opening_range`, `mean_remaining_range` and
`opening_range_share` are invariant under a permutation of `remaining_range`, so
their baselines equal their actual values by construction (descriptive context, not
hypothesis-tested). Uses `np.random.default_rng(seed)`.

### i18n
- **title.en**: "Opening Range Indicator"
- **title.fr**: "Indicateur du range d'ouverture"
- **definition.en**: "How big is the opening range (the first N minutes of the session) relative to the rest-of-session range, and how strongly are the two correlated?"
- **definition.fr**: "Quelle est la taille du range d'ouverture (les N premières minutes de la séance) par rapport au range du reste de la séance, et à quel point les deux sont-ils corrélés ?"

### Slices
- `weekday` — the "by weekday" breakdown (declared via the shared `Weekday` slicer).
- `close` — the "by close" breakdown: session close color, green/red (declared via
  the shared `Close` slicer reading `session_green`).
- `size` — the "by size" breakdown: opening-range size quartiles (declared via
  `SizeBucket(column="opening_range")`); each bucket's `mean_remaining_range` makes
  the relationship visible directly.


---

## 27. Opening Stats

**Family**: `opening_stats`
**Module**: `stats/opening_stats/standard.py`
**Result file**: `results/opening_stats.json`
**Status**: Implemented (standard variant)

### What it measures
Where does the daily RTH session open land relative to the **prior resolved
session's** RTH intraday high and low? The three outcomes form a mutually
exclusive and exhaustive partition whose probabilities sum to 1 — a marginal
3-way distribution:

| Outcome | Condition |
|---|---|
| `above_high` | `session_open > prev_high` (strict) |
| `inside_range` | `prev_low <= session_open <= prev_high` (both boundaries inclusive) |
| `below_low` | `session_open < prev_low` (strict) |

Opening exactly on the prior high or prior low counts as `inside_range` — the
boundaries are inclusive on the inside, which is the natural complement of the
strict-outside convention used by `outside_days`.

### Methodology
1. Build the RTH daily candle per resolved session (`session_open`,
   `session_close`) using the standard resolution rule (clean 09:30 open and a
   last RTH bar at or after `session_end - close_tolerance`).
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
4. Classify `open_location` for each countable day:
   - `above_high` when `session_open > prev_high`.
   - `below_low` when `session_open < prev_low`.
   - `inside_range` otherwise (ties on either boundary are inside).
5. Report one row per outcome under the single `open` condition.

`total_samples` counts all resolved sessions; each row's `total` counts only the
**countable** sessions (those with a prior resolved day).

### Conditions & outcomes
| Condition | Outcomes | Partition? | Description |
|---|---|---|---|
| `open` | `above_high`, `inside_range`, `below_low` | Yes | 3-way location of the session open relative to the prior session's range; `total` = countable days |

### Parameters
| Parameter | Default | Description |
|---|---|---|
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `daily` (the RTH daily candle; a single entry).

### Baseline
Random null (fixed seed, deterministic): each day's `open_location` is replaced
by a uniform draw from `{above_high, inside_range, below_low}` using
`np.random.default_rng(seed)`. Expected `baseline_prob` ≈ 1/3 per outcome — the
null hypothesis is that the daily open is equally likely to land above, inside,
or below the prior range. The baseline is computed per slice group as well as
overall.

### i18n
- **title.en**: "Opening Stats"
- **title.fr**: "Statistiques d'ouverture"
- **definition.en**: "Where does the daily session open land relative to the prior session's high and low — above the prior high, inside the prior range, or below the prior low?"
- **definition.fr**: "Où se situe l'ouverture quotidienne par rapport au plus haut et au plus bas de la session précédente — au-dessus du plus haut, dans la fourchette ou en dessous du plus bas ?"

### Slices
- `weekday` — the "by weekday" breakdown (declared via the shared `Weekday`
  slicer). This is the issue's "by weekday" variant — it is embedded in the
  standard module rather than a separate file, mirroring the pattern used by
  `gap_fill` and `green_red_days`.
- `prev_candle` — split by the prior session's color, green/red (declared via the
  shared `PrevCandle` slicer reading `prev_session_green`).
- `close` — split by the current session's color, green/red (declared via the
  shared `Close` slicer reading `session_green`).

### Future variants (not in MVP)
- `by_levels` — extension above/below the prior range measured in multiples of
  the prior range, for days that open outside it.
- `by_size` — bucket open-distance-from-range for above/below days.


---

## 28. Opening Week Range

**Family**: `opening_week_range`
**Module**: `stats/opening_week_range/standard.py`
**Result file**: `results/opening_week_range.json`
**Status**: Implemented (standard variant)

### What it measures
The **opening week range** is the high-low RTH range formed by the first
`open_days` trading sessions of each ISO week. Given that range, how does the
**rest** of the same week resolve against it — break the opening high only, the
opening low only, **both**, or stay **inside** (neither)? And when both levels
are taken, which one is reached **first**? This is the weekly analogue of
`opening_range_breakout` (opening period → breakout window).

### Methodology
1. Keep RTH bars only (`hour*60+minute >= rth_start_min` and `< rth_end_min`).
2. Assign each bar to its **ISO week**, keyed by that week's Monday date.
3. Drop the **last** week present in the data: it may still be in progress, so
   its range is not final (pending-sample discipline). All earlier weeks are
   resolved.
4. For each resolved week, order its unique trading dates. The first `open_days`
   dates form the **opening window**, the remaining dates the **breakout window**:
   - `opening_high` = max `high`, `opening_low` = min `low` over the opening
     sessions; `opening_size = opening_high - opening_low`.
   - `post_high` = max `high`, `post_low` = min `low` over the breakout sessions.
   A week with no breakout window (`<= open_days` trading sessions) has
   `post_high` / `post_low` as NaN and is excluded from every denominator.
5. Classify each countable week against the opening range (strict inequality —
   touching the level is not a break):
   - **break_high_only**: `post_high > opening_high` and `post_low >= opening_low`.
   - **break_low_only**:  `post_low < opening_low` and `post_high <= opening_high`.
   - **break_both**:      both `post_high > opening_high` and `post_low < opening_low`.
   - **inside**:          neither level is taken.
6. For each both-break week, determine the **break sequence** from intra-week
   1-minute breakout-window bars: the first bar (by timestamp) whose
   `high > opening_high` versus the first whose `low < opening_low`. The earlier
   one is taken first. If a single bar is first to break **both**, the order is
   inferred from that bar's candle path — a bearish bar (`close < open`) prints
   its high first (`high_first`), a bullish bar its low first (`low_first`).

### Conditions & outcomes
| Condition | Outcomes | Partition? | Description |
|---|---|---|---|
| `opening_range` | `break_high_only`, `break_low_only`, `break_both`, `inside` | Yes | Four-way outcome partition over all countable weeks; `total` = countable weeks |
| `break_both` | `high_first`, `low_first` | Yes | Which opening level is taken first, given both broke; `total` = both-break weeks |

`total_samples` counts all resolved weeks; each row's `total` counts only the
relevant countable weeks (those with a non-empty breakout window).

### Parameters
None (the RTH session bounds come from the instrument config). The opening length
is fixed by each timeframe entry.

### Timeframes computed
- `1d` (opening window = first 1 session of the week).

### Baseline
Random null, computed per tier (fixed seed, deterministic):
- **Outcome partition**: each week's breakout extremes are **reflected** around
  the opening-range midpoint with probability 0.5 (`p -> 2*mid - p`, swapping
  `post_high` and `post_low`). Reflection turns a `break_high_only` week into a
  `break_low_only` week and vice versa, while `break_both` and `inside` are
  direction-symmetric and unchanged — the null has no directional bias, so the
  comparison reveals whether the week breaks **up** more often than **down**
  beyond a coin flip.
- **Double-break sequence**: the real opening ranges are kept (so the both-break
  `total` stays full size) and only the break order is reassigned by a fair coin
  (p=0.5) — the null for which level is taken first (~0.5).

### Slices
- `size` — the "by size" breakdown: opening-week-range size quartiles (declared
  `week_range_size` parameter.

### Future variants (not in MVP)
- `by_levels` — extension in multiples of the opening-week range beyond the
  broken level.
- `by_retracement` — pullback depth after the break.


---

## 29. Initial Balance Breakout

**Family**: `initial_balance`
**Module**: `stats/initial_balance/standard.py`
**Result file**: `results/initial_balance.json`
**Status**: Implemented (standard variant)

### What it measures
The **initial balance** is the high-low range of the first N minutes of the RTH
session (conventionally the first 30 to 60 minutes). Given that range, which
direction does price break during the **rest** of the session? Reported under a
single `ib` condition with four mutually exclusive outcomes that partition every
countable day: `broke_high`, `broke_low`, `broke_both`, `neither`. Their
probabilities sum to 1. This is the intraday sibling of `opening_range_breakout`
(opening period → breakout window); the two differ only by convention — the
initial balance is a longer window (>= 30min), so the 15min length is not offered.

### Methodology
1. Build the RTH daily candle per resolved session (`session_open`,
   `session_close`) using the same resolution rule as the other daily stats (clean
   09:30 open and a last RTH bar at or after `session_end - close_tolerance`).
2. Compute the **initial balance** from the bars in the IB window
   `[rth_start, rth_start + ib_period)`:
   - `ib_high` = max of `high`, `ib_low` = min of `low` over the window.
   - `ib_size` = `ib_high - ib_low`.
3. Compute the **breakout window** extremes from the rest of the session,
   `[rth_start + ib_period, rth_end)` (the initial balance itself is excluded so it
   can never break its own bars):
   - wick criteria (default): `post_high` = max `high`, `post_low` = min `low`.
   - close criteria: `post_close_high` = max `close`, `post_close_low` = min `close`.
   Join these onto the resolved-days index (inner join — a day survives only with a
   full initial balance AND a non-empty breakout window).
4. Classify the breakout direction with **strict** inequalities (touching a level
   exactly is NOT a break), using the extremes selected by `breakout_criteria`:
   - `broke_high`: `high > ib_high` AND `low >= ib_low` (broke up only).
   - `broke_low`:  `low < ib_low`  AND `high <= ib_high` (broke down only).
   - `broke_both`: `high > ib_high` AND `low < ib_low` (broke both sides).
   - `neither`:    `high <= ib_high` AND `low >= ib_low` (held inside all session).

`total_samples` counts all resolved sessions; each row's `total` is the
**countable** days (a clean initial balance and a non-empty breakout window). The
four outcomes are mutually exclusive and exhaustive, so their counts sum to `total`.

### Conditions & outcomes
| Condition | Outcomes | Partition? | Description |
|---|---|---|---|
| `ib` | `broke_high`, `broke_low`, `broke_both`, `neither` | Yes | Breakout direction relative to the initial balance; `total` = countable days |

### Parameters
| Parameter | Default | Description |
|---|---|---|
| timeframe | 30min | Initial-balance length: 30min or 1h (emitted as separate timeframe entries) |
| breakout_criteria | wick | Break detection: `wick` (intraday extreme) or `close` (bar close beyond the level) |
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `30min` (initial balance 09:30–10:00).
- `1h` (initial balance 09:30–10:30, the canonical initial balance).

### Baseline
Random directional null (fixed seed, deterministic): each countable day's breakout
move is **reflected** around its initial-balance midpoint with probability 0.5 (a
price `p` maps to `2*mid - p`, which swaps the high and low extremes). Reflection
turns a `broke_high` day into a `broke_low` day and vice versa, while `broke_both`
and `neither` are direction-symmetric and unchanged. The null therefore carries NO
directional bias, so `broke_high` and `broke_low` converge to their shared mean and
the comparison reveals whether the initial balance breaks **up** more often than
**down** beyond a coin flip. Uses `np.random.default_rng(seed)`.

### i18n
- **title.en**: "Initial Balance Breakout"
- **title.fr**: "Cassure de l'initial balance"
- **definition.en**: "After the initial balance forms from the first N minutes of the session, how often does price break above the balance high, below the balance low, both sides, or neither during the rest of the session?"
- **definition.fr**: "Après la formation de l'initial balance sur les N premières minutes de la session, à quelle fréquence le prix casse-t-il au-dessus du haut de l'initial balance, en-dessous du bas, des deux côtés, ou ni l'un ni l'autre pendant le reste de la séance ?"

### Slices
- `weekday` — the "by weekday" breakdown (declared via the shared `Weekday` slicer).
- `close` — the "by close" breakdown: session close color, green/red (declared via
  the shared `Close` slicer reading `session_green`).
- `prev_candle` — the "by color" breakdown: prior session color, green/red
  (declared via the shared `PrevCandle` slicer reading `prev_session_green`).
- `size` — the "by size" breakdown: initial-balance size quartiles, absolute
  (declared via `SizeBucket(column="ib_size")`).
- `size_pct` — the "by size (% of price)" breakdown: initial-balance size as a
  percentage of the session open, in preset bands (<0.2%, 0.2–0.4%, 0.4–0.6%,
  `SizeBucket(column="ib_size_pct", buckets=[…])`).
- `overnight` — the "by overnight session" breakdown: overnight gap direction, the
  session open above (green) / below (red) the prior session's close (declared via
  the shared `Overnight` slicer reading `overnight_green`).
- `levels` — the "by levels" breakdown: how far the breakout extended past the
  balance, in multiples of `ib_size` (<0.5x, 0.5–1x, 1–1.5x, 1.5–2x, >=2x; declared
  via `Levels(ref="ib_size", ext="extension")`).

### Extension variants (slices added by the IB extensions issue #36)
The slice-friendly extension variants are implemented above as `size_pct`,
outcome/metric computations (they do not fit the four-outcome breakout partition)
and are deferred to follow-up issues:
- `by_performance` — average / maximum extension before price breaks back into the
  balance (a magnitude metric via the `value` channel). Implemented as section 53
  (`initial_balance_performance`).
- `by_retracement` — pullback depth into the balance after the break, on single-break
  days only, bucketed at 0.25 / 0.5 / 0.75 of `ib_size`. Implemented as section 54
  (`initial_balance_retracement`).
- `by_time` — distribution of the first-breakout time (early vs. late). Implemented
  as section 55 (`initial_balance_time`).
- `by_rejection` — contingency of which balance edge formed first vs. which broke
  first. Implemented as section 56 (`initial_balance_rejection`).


## 30. Power Hour Breakout

**Family**: `power_hour_breakout`
**Module**: `stats/power_hour_breakout/standard.py`
**Result file**: `results/power_hour_breakout.json`
**Status**: Implemented (standard variant)

### What it measures
The **power hour** is the last N minutes of the RTH session (conventionally the
final hour). Going into the power hour, the session has already established a
high-low range (the "high/low of day so far"). Does the power hour make a **new
high of day**, a **new low of day**, both, or neither relative to that prior
range? Reported under a single `power_hour` condition with four mutually
exclusive outcomes that partition every countable day: `made_high`, `made_low`,
`made_both`, `neither`. Their probabilities sum to 1. This is the end-of-session
mirror of `initial_balance` (early window sets a range, a later window may break
it) — here the measured window sits at the END and is compared against everything
before it. The marginal new-high / new-low rates are recoverable as
`made_high + made_both` and `made_low + made_both`.

### Methodology
1. Build the RTH daily candle per resolved session (`session_open`,
   `session_close`) using the same resolution rule as the other daily stats (clean
   09:30 open and a last RTH bar at or after `session_end - close_tolerance`).
2. Compute the **pre-power-hour** extremes from the bars before the power hour,
   `[rth_start, ph_start)` where `ph_start = rth_end - ph_period`:
   - `pre_high` = max of `high`, `pre_low` = min of `low` over the window.
3. Compute the **power-hour** extremes from the last `ph_period` minutes,
   `[ph_start, rth_end)`:
   - `ph_high` = max `high`, `ph_low` = min `low` (wick: a wick beyond the level
     counts as a new extreme).
   - `ph_open` = open of the bar at exactly `ph_start`.
   Join these onto the resolved-days index (inner join — a day survives only with a
   non-empty pre-power-hour window, a clean power-hour open bar, AND a non-empty
   power-hour window).
4. Classify the new-extreme direction with **strict** inequalities (touching a
   level exactly is NOT a new extreme):
   - `made_high`: `ph_high > pre_high` AND `ph_low >= pre_low` (new high only).
   - `made_low`:  `ph_low < pre_low`  AND `ph_high <= pre_high` (new low only).
   - `made_both`: `ph_high > pre_high` AND `ph_low < pre_low` (both extremes).
   - `neither`:   `ph_high <= pre_high` AND `ph_low >= pre_low` (held inside).

`total_samples` counts all resolved sessions; each row's `total` is the
**countable** days (a non-empty pre-power-hour window and a non-empty power hour).
The four outcomes are mutually exclusive and exhaustive, so their counts sum to
`total`.

### Conditions & outcomes
| Condition | Outcomes | Partition? | Description |
|---|---|---|---|
| `power_hour` | `made_high`, `made_low`, `made_both`, `neither` | Yes | New-extreme direction of the power hour vs. the prior range; `total` = countable days |

### Parameters
| Parameter | Default | Description |
|---|---|---|
| timeframe | 1h | Power-hour length: `1h` (the conventional final hour) |
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `1h` (power hour 15:15–16:15 for NQ, the canonical final hour).

### Baseline
Random directional null (fixed seed, deterministic): each countable day's
power-hour move is **reflected** around its pre-power-hour range midpoint with
probability 0.5 (a price `p` maps to `2*mid - p`, which swaps the high and low
extremes). Reflection turns a `made_high` day into a `made_low` day and vice
versa, while `made_both` and `neither` are direction-symmetric and unchanged. The
null therefore carries NO directional bias, so `made_high` and `made_low`
converge to their shared mean and the comparison reveals whether the power hour
makes a new high **up** more often than **down** beyond a coin flip. Uses
`np.random.default_rng(seed)`.

### i18n
- **title.en**: "Power Hour Breakout"
- **title.fr**: "Cassure du power hour"
- **definition.en**: "During the last hour of the session (the power hour), how often does price make a new high of day, a new low of day, both, or neither relative to the range established earlier in the session?"
- **definition.fr**: "Pendant la dernière heure de la séance (le power hour), à quelle fréquence le prix inscrit-il un nouveau plus haut du jour, un nouveau plus bas du jour, les deux, ou ni l'un ni l'autre par rapport à l'amplitude établie plus tôt dans la séance ?"

### Slices
- `weekday` — the "by weekday" breakdown (declared via the shared `Weekday` slicer).
- `open` — the "by open" breakdown: where the power-hour open falls relative to the
  pre-power-hour range midpoint, the upper half (`above`) or lower half (`below`)
  (declared via the `PowerHourOpen` slicer reading `ph_open_above`).


## 31. Power Hour Continuation

**Family**: `power_hour_continuation`
**Module**: `stats/power_hour_continuation/standard.py`
**Result file**: `results/power_hour_continuation.json`
**Status**: Implemented (standard variant)

### What it measures
The **power hour** is the last N minutes of the RTH session (conventionally the
final hour). This stat cross-tabulates the **pre-power-hour move** — the session's
direction going INTO the power hour — against the **power-hour candle direction**.
Does the power hour continue the earlier move or reverse it? A 2x2 conditional
matrix (the same shape as `overnight_continuation`, applied to the power-hour
window instead of the overnight gap): condition = pre-power-hour color
(`pre_green`/`pre_red`), outcome = power-hour color (`green`/`red`). "Continuation"
is the diagonal (pre green → power hour green, pre red → power hour red); all four
cells are reported so the asymmetry is directly visible.

### Methodology
1. Build the RTH daily candle per resolved session (`session_open`,
   `session_close`) via the shared resolution rule (clean 09:30 open and a last RTH
   bar at or after `session_end - close_tolerance`). `session_open` is the
   **pre-power-hour candle open**; `session_close` is the **power-hour candle
   close** (the last RTH bar sits inside the power hour).
2. The power hour is `[ph_start, rth_end)` where `ph_start = rth_end - ph_period`;
   the pre-power-hour window is `[rth_start, ph_start)`.
3. Join two per-day fields onto the resolved index (inner join — a day survives
   only with a non-empty pre-power-hour window AND a clean power-hour open bar):
   - `pre_close` = close of the last bar before `ph_start`.
   - `ph_open` = open of the bar at exactly `ph_start` (the **reference open** for
     the power-hour candle color).
   `pre_high` / `pre_low` (pre-window extremes) are also joined for the `open`
   slice midpoint.
4. Classify both legs (ties count as green, `>=`):
   - `pre_green` = `pre_close >= session_open` (pre-power-hour candle color).
   - `ph_green` = `session_close >= ph_open` (power-hour candle color).

`total_samples` counts the countable sessions; each row's `total` counts the
countable days in that condition (`pre_green` or `pre_red`), so the two
conditions' totals sum to `total_samples`. Within each condition the two outcomes
partition the days, so `green + red` counts equal that condition's `total`.

### Conditions & outcomes
| Condition | Outcomes | Partition? | Description |
|---|---|---|---|
| `pre_green` | `green`, `red` | Yes | Power-hour color when the pre-power-hour move was up; `total` = countable green-pre days |
| `pre_red` | `green`, `red` | Yes | Power-hour color when the pre-power-hour move was down; `total` = countable red-pre days |

### Parameters
| Parameter | Default | Description |
|---|---|---|
| timeframe | 1h | Power-hour length: `1h` (the conventional final hour) |
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `1h` (power hour 15:15–16:15 for NQ, the canonical final hour).

### Baseline
Random null (fixed seed, deterministic): the power-hour color (`ph_green`) is
replaced with an independent fair-coin sequence while the pre-power-hour condition
stays as-is from real data. Every cell's `baseline_prob` converges to ~0.5, so the
comparison reveals whether the pre-power-hour move predicts the power-hour
direction beyond a coin flip. Uses `np.random.default_rng(seed)`.

### i18n
- **title.en**: "Power Hour Continuation"
- **title.fr**: "Continuation du power hour"
- **definition.en**: "Given the session's direction going into the power hour (the pre-power-hour move, green or red), how often does the power hour itself close in the same direction (continuation) versus reverse?"
- **definition.fr**: "Selon la direction de la séance à l'entrée du power hour (le mouvement pré-power-hour, vert ou rouge), à quelle fréquence le power hour clôture-t-il dans la même direction (continuation) plutôt que de se retourner ?"

### Slices
- `weekday` — the "by weekday" breakdown (declared via the shared `Weekday` slicer).
- `open` — the "by open" breakdown: where the power-hour open falls relative to the
  pre-power-hour range midpoint, the upper half (`above`) or lower half (`below`)
  (declared via the `PowerHourOpen` slicer reading `ph_open_above`).


---

## 32. Intraday Timing

**Family**: `intraday_timing`
**Module**: `stats/intraday_timing/standard.py`
**Result file**: `results/intraday_timing.json`
**Status**: Implemented

### What it measures
For each resolved RTH session, which intraday time bucket produced the day's
high and which produced the day's low? Aggregated across days as the fraction of
days each bucket claims each extreme. For a given condition, probabilities sum to
~100% across buckets (each day contributes exactly one bucket per condition).

### Methodology
1. Build the RTH daily candle per resolved day (`session_open`, `session_close`)
   via `build_resolved_days` (clean 09:30 open and a last RTH bar at or after
   `session_end - close_tolerance`).
2. Over the same RTH bar filter (`hour*60+minute >= rth_start_min` and
   `< rth_end_min`), assign each bar a bucket index
   `(minute_of_day - rth_start_min) // bucket_min`. For each date find:
   - `high_bucket`: bucket of the bar with max `high` (first occurrence on ties
     via `idxmax`).
   - `low_bucket`: bucket of the bar with min `low` (via `idxmin`).
   - `present_buckets`: sorted tuple of distinct bucket ints present that day
     (used by the baseline to draw uniformly from actually-traded buckets).
   Join onto the resolved-days index (inner join — only resolved dates retained)
   and add `session_green` (`session_close >= session_open`).
3. **Pending discipline**: only resolved days participate; early-close days and
   the final incomplete day are excluded by `build_resolved_days`.

### Conditions
| Condition key | Source column | Description |
|---|---|---|
| `intraday_high` | `high_bucket` | Time bucket containing the RTH session high |
| `intraday_low` | `low_bucket` | Time bucket containing the RTH session low |

### Outcomes
One per time bucket in the session grid present in the dataset. Keys are the
zero-padded `HHMM` of the bucket start (e.g. `0930`, `0945`, `1000`); labels are
`HH:MM` (en/fr identical). The grid spans `rth_start`→`rth_end` in `bucket_min`
steps; the final bucket may be partial when the session length is not a whole
multiple of the bucket size (e.g. NQ 09:30–16:15 with 30-min buckets ends with a
15-min `1600` bucket). Buckets with no occurrences in a subset are still emitted
(count=0) when they appear in any day's `present_buckets`.

### Parameters
| Parameter | Default | Description |
|---|---|---|
| timeframe | 15min | Bucket granularity: `1min`, `5min`, `15min`, `30min`, `1h` |
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- One bucket granularity per run, stored under
  `instruments.{INSTRUMENT}.{timeframe}` (default `15min`).

### Baseline
Random null: for each day, two independent uniform draws are made from that day's
`present_buckets` (one for the high, one for the low), representing the null
hypothesis that each extreme falls in a uniformly random traded bucket. A temp
copy of the day table is built with the two bucket columns replaced by these
draws, then `compute_rows` is called on it. Uses `np.random.default_rng(seed)`
for deterministic output. The baseline is computed per slice group as well as
overall.

### i18n
- **title.en**: "Intraday Timing"
- **title.fr**: "Timing intraday"
- **definition.en**: "For each RTH session, which intraday time bucket produced the day's high and which produced the day's low? Aggregated as the fraction of days each bucket claims each extreme."
- **definition.fr**: "Pour chaque séance RTH, quelle tranche horaire intraday a produit le plus haut et le plus bas du jour ? Agrégé sous forme de fraction des jours où chaque tranche revendique chaque extrême."

### Slices
- `weekday` — the "by weekday" breakdown (declared via the shared `Weekday`
  slicer).
- `session_color` — the `session_color` parameter (green / red days), expressed
  as a custom `SessionColor` slicer (subclass of `_ColorSlicer`) splitting on
  `session_green`. A single run yields the overall (`all`) result plus the green
  and red breakdowns.


---

## 33. Intraday Volume & Range

**Family**: `intraday_volume_range`
**Module**: `stats/intraday_volume_range/standard.py`
**Result file**: `results/intraday_volume_range.json`
**Status**: Implemented

### What it measures
For each intraday time bucket of the RTH session: the **average traded volume**,
the **average price range** (`high - low`, in points), and the **average
percentage range** (`(high - low) / bucket_open`). Shows when volume and
volatility build or fade through the session (the classic open/close peaks with a
midday lull). The overall result aggregates all resolved days; the per-weekday
breakdown is produced by the declared `weekday` slice.

This is a **magnitude** stat: all three rows carry a continuous metric in `value`
(and its random-baseline counterpart in `value_baseline`). The `probability` /
`baseline_prob` channel is left at `0.0` for every row. Every row reports its
sample size in `count` / `total` (the number of resolved days that had bars in
that bucket).

Each intraday bucket is a **condition** (key `HHMM` of the bucket start, e.g.
`0930`); the three metrics are the **outcomes**. The bucket conditions are
timeframe-dependent and built per instance.

### Methodology
1. Build the resolved-days table via `build_resolved_days` (clean session open at
   exactly `rth_start` and a last RTH bar at or after `rth_end - close_tolerance`).
   Early-close days and the final incomplete day are excluded (pending discipline).
2. Over the same RTH bar filter (`hour*60+minute` in `[rth_start_min, rth_end_min)`),
   assign each bar a bucket index `(minute_of_day - rth_start_min) // bucket_min`.
   The final bucket may be partial when the session length is not a whole multiple
   of `bucket_min` (NQ 09:30–16:15 with 15-minute buckets ends with the 16:00–16:15
   bucket).
3. Per `(day, bucket)` compute: `vol` = summed bar volume, `rng` = `max(high) -
   min(low)`, `pct` = `rng / bucket_open`, where `bucket_open` is the open of the
   bucket's earliest bar that day. Pivot to one row per resolved day with three
   columns per bucket (`vol_{b}`, `rng_{b}`, `pct_{b}`); absent buckets are NaN.
4. For each bucket present in the subset, report three rows whose `value` is the
   NaN-skipping mean across the contributing days:
   - `mean_volume`    — average bucket volume.
   - `mean_range`     — average bucket range in points.
   - `mean_range_pct` — average bucket percentage range (decimal).
5. Re-run the same rows per weekday via the `weekday` slice.

### Conditions
| Condition key | Description |
|---|---|
| `HHMM` (e.g. `0930`, `0945`, …) | One per intraday time bucket present, keyed by the bucket's start time |

### Outcomes
| Outcome key | Channel | Description |
|---|---|---|
| `mean_volume` | `value` | Average summed volume of the bucket's bars |
| `mean_range` | `value` | Average bucket range (`high - low`), in points |
| `mean_range_pct` | `value` | Average bucket percentage range (`range / bucket_open`), decimal |

### Parameters
| Parameter | Default | Description |
|---|---|---|
| timeframe | `15min` | Bucket granularity (`1min`, `5min`, `15min`, `30min`, `1h`) |
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- One per run, keyed by the chosen bucket granularity (default `15min`).

### Baseline
Random null: for each metric, all per-`(day, bucket)` observations in the subset
are pooled. For a bucket contributing `n` days, `n` observations are drawn
uniformly at random **without replacement** from the pool and their mean is the
bucket's baseline. This represents the null hypothesis that the intraday time
bucket carries no information — its expected metric equals the grand mean across
all buckets. Uses `np.random.default_rng(seed)` for deterministic output. The
baseline is computed per slice group as well as overall.

### i18n
- **title.en**: "Intraday Volume & Range"
- **title.fr**: "Volume & amplitude intraday"
- **definition.en**: "For each intraday time bucket: the average traded volume, the average price range, and the average percentage range. Shows when volume and volatility build or fade through the session."
- **definition.fr**: "Pour chaque tranche horaire intraday : le volume échangé moyen, l'amplitude moyenne en points et l'amplitude moyenne en pourcentage. Montre quand le volume et la volatilité montent ou retombent au fil de la séance."

### Slices
- `weekday` — the "by weekday" breakdown (declared via `slices = ("weekday",)`).


---

## 34. Intraday Range Window

**Family**: `intraday_range_window`
**Module**: `stats/intraday_range_window/standard.py`
**Result file**: `results/intraday_range_window.json`
**Status**: Implemented

### What it measures
The high-low range over a single, **configurable intraday window** (e.g.
09:30–10:30) measured each trading day, then summarized across days. For both the
**absolute** range (`high - low`, in points) and the **percentage** range
(`(high - low) / window_open`) it reports four aggregates: average, maximum,
minimum, and median. It quantifies how far a given time-of-day window tends to
travel and how that varies day to day. The overall result aggregates all resolved
days; the per-weekday breakdown is produced by the declared `weekday` slice.

This is a **magnitude** stat: all eight rows carry a continuous metric in `value`
(and its random-baseline counterpart in `value_baseline`). The `probability` /
`baseline_prob` channel is left at `0.0` for every row. Every row reports its
sample size in `count` / `total` (the number of resolved days whose window is
valid).

There is a single **condition**: the measured window, keyed `HHMM-HHMM` (e.g.
`0930-1030`). The eight aggregates are the **outcomes**.

### Methodology
1. Build the resolved-days index via `build_resolved_days` (clean session open at
   exactly `rth_start` and a last RTH bar at or after `rth_end - close_tolerance`).
   Early-close days and the final incomplete day are excluded (pending discipline).
2. Pivot RTH bars onto a per-day `(date × minute)` grid for high / low / open.
3. Slide a window of `win_len` bars across the grid (`win_len = last_bar -
   start_min + 1`, where `last_bar = min(end_min, rth_end_min - 1)`). Each slide
   position's range is `max(high) - min(low)` and its percentage range is
   `range / open` at the window's first bar. The **actual** window is the slide
   position starting at `start_min`; all positions form the baseline candidate
   pool. A day counts only when both its actual range and percentage range are
   finite, so all eight rows share the same `count` / `total`.
4. Report eight rows whose `value` is the aggregate over the contributing days:
   `range_avg` / `range_max` / `range_min` / `range_median` (points) and
   `range_pct_avg` / `range_pct_max` / `range_pct_min` / `range_pct_median`.
5. Re-run the same rows per weekday via the `weekday` slice.

### Conditions
| Condition key | Description |
|---|---|
| `HHMM-HHMM` (e.g. `0930-1030`) | The single measured window (start–end) |

### Outcomes
| Outcome key | Channel | Description |
|---|---|---|
| `range_avg` | `value` | Average daily range (`high - low`), in points |
| `range_max` | `value` | Maximum daily range, in points |
| `range_min` | `value` | Minimum daily range, in points |
| `range_median` | `value` | Median daily range, in points |
| `range_pct_avg` | `value` | Average daily percentage range (`range / window_open`), decimal |
| `range_pct_max` | `value` | Maximum daily percentage range, decimal |
| `range_pct_min` | `value` | Minimum daily percentage range, decimal |
| `range_pct_median` | `value` | Median daily percentage range, decimal |

### Parameters
| Parameter | Default | Description |
|---|---|---|
| start_window | `09:30` | Window start `HH:MM` (must be `>= rth_start`) |
| end_window | `10:30` | Window end `HH:MM` (must be `> start` and `<= rth_end`) |
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- One per run, keyed by the chosen window (`HHMM-HHMM`, default `0930-1030`).

### Baseline
Random null: for each day a single same-length window is drawn uniformly at
random from all windows that fit the RTH session (range and percentage kept
paired), and the eight aggregates are recomputed over those random draws. This is
the null hypothesis that the chosen time-of-day window carries no information
beyond a random window of the same length — a window that genuinely concentrates
volatility (e.g. the opening hour) shows aggregates well above the baseline. Uses
`np.random.default_rng(seed)` for deterministic output; computed per slice group
as well as overall.

### i18n
- **title.en**: "Intraday Range Window"
- **title.fr**: "Amplitude d'une fenêtre intraday"
- **definition.en**: "The high-low range over a configurable intraday window measured each day, summarized across days. Reports the average, maximum, minimum, and median of both the absolute range (points) and the percentage range."
- **definition.fr**: "L'amplitude haut-bas sur une fenêtre intraday configurable mesurée chaque jour, résumée sur l'ensemble des jours. Donne la moyenne, le maximum, le minimum et la médiane de l'amplitude en points et de l'amplitude en pourcentage."

### Slices
- `weekday` — the "by weekday" breakdown (declared via `slices = ("weekday",)`).


---

## 35. Market Open Volume

**Family**: `market_open_volume`
**Module**: `stats/market_open_volume/standard.py`
**Result file**: `results/market_open_volume.json`
**Status**: Implemented

### What it measures
The **Pearson correlation** between the opening bar's traded volume and the
rest-of-session volume, measured across resolved RTH trading days. It answers:
does a high-volume open predict a high-volume session? A correlation near `+1`
means opening volume tracks session activity; a correlation near `0` means the
open carries no information about the rest of the day.

This is a **magnitude** stat: the single row carries the correlation coefficient
in `value` (and its random-baseline counterpart in `value_baseline`). The
`probability` / `baseline_prob` channel is left at `0.0`. The row reports its
sample size in `count` / `total` (the number of resolved days with both volumes
present).

There is a single **condition** (`any_day`) and a single **outcome**
(`correlation`). The opening bar's duration is governed by the chosen timeframe;
the three timeframes are merged into one result file.

### Methodology
1. Build the resolved-days index via `build_resolved_days` (clean session open at
   exactly `rth_start` and a last RTH bar at or after `rth_end - close_tolerance`).
   Early-close days and the final incomplete day are excluded (pending discipline).
2. For each resolved day, sum 1-min bar volume over two minute-of-day windows:
   `open_volume` over `[rth_start, rth_start + open_bar_min)` and `rest_volume`
   over `[rth_start + open_bar_min, rth_end)`, where `open_bar_min` is the
   timeframe's minute length (15 / 30 / 60).
3. Compute the Pearson `r` between `open_volume` and `rest_volume` across all days
   where both are non-NaN. When fewer than two such days exist, or either series
   has zero variance, `r = 0.0`.
4. Repeat for timeframes `15min`, `30min`, `1h`; merge into one result file keyed
   by timeframe.

### Conditions
| Condition key | Description |
|---|---|
| `any_day` | All resolved days (the single population) |

### Outcomes
| Outcome key | Channel | Description |
|---|---|---|
| `correlation` | `value` | Pearson `r` between opening-bar volume and rest-of-session volume |

### Parameters
| Parameter | Default | Description |
|---|---|---|
| timeframe | `1h` | Opening-bar duration; one of `15min` / `30min` / `1h` (all three computed by `run()`) |
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `15min`, `30min`, `1h` — merged into a single result file under
  `instruments.{INSTRUMENT}.{timeframe}`.

### Baseline
Random null: hold `open_volume` fixed and randomly **permute** `rest_volume`
across days, then recompute Pearson `r`. Shuffling breaks any open→rest
relationship, so the expected baseline `r ≈ 0`. A genuine relationship shows an
actual `r` well above the baseline. Uses `np.random.default_rng(seed)` for
deterministic output.

### i18n
- **title.en**: "Market Open Volume"
- **title.fr**: "Volume à l'ouverture"
- **definition.en**: "Pearson correlation between the opening bar's volume and the rest-of-session volume: does a high-volume open predict a high-volume session?"
- **definition.fr**: "Corrélation de Pearson entre le volume de la bougie d'ouverture et le volume du reste de la séance : un volume élevé à l'ouverture annonce-t-il une séance active ?"

### Slices
- None (`slices = ()`).


---

## 36. Overnight Range Breakout

**Family**: `overnight_range_breakout`
**Module**: `stats/overnight_range_breakout/standard.py`
**Result file**: `results/overnight_range_breakout.json`
**Status**: Implemented (standard variant)

### What it measures
The **overnight range** is the high-low range of the overnight session that
precedes a given RTH session (18:00 ET previous day → 09:30 ET). Given that range,
which direction does price break during the **RTH session**? Reported under a
single `overnight_range` condition with four mutually exclusive outcomes that
partition every countable day: `broke_high`, `broke_low`, `broke_both`, `neither`.
Their probabilities sum to 1. The marginal high-break / low-break rates are
recoverable as `broke_high + broke_both` and `broke_low + broke_both`.

### Methodology
1. Build the RTH daily candle per resolved session (`session_open`,
   `session_close`) using the same resolution rule as the other daily stats (clean
   09:30 open and a last RTH bar at or after `session_end - close_tolerance`).
2. Compute the **overnight range** from the overnight session bars. The session
   crosses midnight, so each bar is attributed to the RTH session date it
   **precedes**: evening bars (`hour*60+minute >= overnight_start`, i.e. ≥ 18:00)
   belong to the *next* calendar day's overnight; early bars
   (`hour*60+minute < overnight_end`, i.e. < 09:30) belong to the *same* calendar
   day's overnight. Over that window:
   - `on_high` = max of `high`, `on_low` = min of `low`.
   - `on_size` = `on_high - on_low`.
3. Compute the **RTH session** extremes over `[rth_start, rth_end)`:
   - wick criteria (default): `day_high` = max `high`, `day_low` = min `low`.
   - close criteria: `day_close_high` = max `close`, `day_close_low` = min `close`.
   Join the overnight range and the RTH extremes onto the resolved-days index
   (inner join — a day survives only with a resolved RTH session AND a prior
   overnight range).
4. Classify the breakout direction with **strict** inequalities (touching a level
   exactly is NOT a break), using the extremes selected by `breakout_criteria`:
   - `broke_high`: `high > on_high` AND `low >= on_low` (broke up only).
   - `broke_low`:  `low < on_low`  AND `high <= on_high` (broke down only).
   - `broke_both`: `high > on_high` AND `low < on_low` (broke both sides).
   - `neither`:    `high <= on_high` AND `low >= on_low` (held inside all session).

`total_samples` counts all resolved sessions; each row's `total` is the
**countable** days (a resolved RTH session and a prior overnight range). The four
outcomes are mutually exclusive and exhaustive, so their counts sum to `total`.

### Conditions & outcomes
| Condition | Outcomes | Partition? | Description |
|---|---|---|---|
| `overnight_range` | `broke_high`, `broke_low`, `broke_both`, `neither` | Yes | Breakout direction of the RTH session relative to the prior overnight range; `total` = countable days |

### Parameters
| Parameter | Default | Description |
|---|---|---|
| breakout_criteria | wick | Break detection: `wick` (intraday extreme) or `close` (bar close beyond the level) |
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `daily` (the RTH daily candle; a single entry).

### Baseline
Random directional null (fixed seed, deterministic): each countable day's RTH move
is **reflected** around its overnight-range midpoint with probability 0.5 (a price
`p` maps to `2*mid - p`, which swaps the high and low extremes). Reflection turns a
`broke_high` day into a `broke_low` day and vice versa, while `broke_both` and
`neither` are direction-symmetric and unchanged. The null therefore carries NO
directional bias, so `broke_high` and `broke_low` converge to their shared mean and
the comparison reveals whether the overnight range breaks **up** more often than
**down** beyond a coin flip. Uses `np.random.default_rng(seed)`.

### i18n
- **title.en**: "Overnight Range Breakout"
- **title.fr**: "Cassure du range overnight"
- **definition.en**: "Comparing each session's market-hours high and low to the prior overnight session's high and low, how often does price break above the overnight high only, below the overnight low only, both sides, or neither during the RTH session?"
- **definition.fr**: "En comparant le plus haut et le plus bas de la séance régulière au plus haut et au plus bas de la session overnight précédente, à quelle fréquence le prix casse-t-il au-dessus du plus haut overnight uniquement, en-dessous du plus bas uniquement, des deux côtés, ou ni l'un ni l'autre pendant la séance RTH ?"

### Slices
- `weekday` — the "by weekday" breakdown (declared via the shared `Weekday` slicer).
- `close` — the "by close" breakdown: session close color, green/red (declared via
  the shared `Close` slicer reading `session_green`).
- `prev_candle` — the "by prev candle" breakdown: prior session color, green/red
  (declared via the shared `PrevCandle` slicer reading `prev_session_green`).
- `size` — the "by size" breakdown: overnight-range size quartiles (declared via
  `SizeBucket(column="on_size")`).
- `levels` — the "by levels" breakdown: how far the breakout extended past the
  overnight range, in multiples of `on_size` (<0.5x, 0.5–1x, 1–1.5x, 1.5–2x, >=2x;
  declared via `Levels(ref="on_size", ext="extension")`).


## 37. Market Session Breakout

**Family**: `market_session_breakout`
**Module**: `stats/market_session_breakout/standard.py`
**Result file**: `results/market_session_breakout.json`
**Status**: Implemented (standard variant)

### What it measures
Given the high-low range a **first** market session forms, which direction does
price break during a **second** session? Both sessions are named windows read from
the instrument config — the single source of truth for their bounds — so any
ordered pair works; the default is `london` (session 1) → `ny` (session 2).
Reported under a single `session1_range` condition with four mutually exclusive
outcomes that partition every countable cycle: `broke_high`, `broke_low`,
`broke_both`, `neither`. Their probabilities sum to 1.

This is the generic, configurable sibling of `overnight_range_breakout` (which
hardwires overnight → RTH). For NQ the geographic sessions are carved from the 24h
cycle on the 09:30 RTH boundary so they never overlap: `asia` 18:00→03:00,
`london` 03:00→09:30, `ny` 09:30→16:15 (= RTH).

### Methodology
1. Attribute every bar to a **cycle** (the RTH session date). An intraday session
   (`start < end`) maps to its own calendar date; a cross-midnight session
   (`start >= end`, e.g. `asia`) maps its evening bars to the *next* day's cycle
   and its early bars to the *same* day's cycle. Session 1 and session 2 are joined
   on this shared cycle date, so the pair must be ordered within a cycle.
2. A session is **resolved** for a cycle when it has a clean open bar (offset 0
   from `start`) AND end coverage (last bar at or after `duration - close_tolerance`,
   with `duration = (end - start) mod 1440`). This generalizes the daily resolution
   rule to an arbitrary session window.
3. Compute the **session 1 range**: `s1_high` = max `high`, `s1_low` = min `low`,
   `s1_size` = `s1_high - s1_low`. Also record `high_first` — whether the session's
   high was reached before its low (NaN when a single bar held both, so the order is
   undetermined).
4. Compute the **session 2** extremes: wick criteria (default) uses `s2_high` /
   `s2_low`; close criteria uses `s2_close_high` / `s2_close_low`. Inner-join both
   sessions on the cycle date — a cycle survives only when BOTH are resolved.
5. Classify with **strict** inequalities (touching a level exactly is NOT a break),
   using the extremes selected by `breakout_criteria`:
   - `broke_high`: `high > s1_high` AND `low >= s1_low` (broke up only).
   - `broke_low`:  `low < s1_low`  AND `high <= s1_high` (broke down only).
   - `broke_both`: `high > s1_high` AND `low < s1_low` (broke both sides).
   - `neither`:    `high <= s1_high` AND `low >= s1_low` (held inside).

`total_samples` and each row's `total` are the **countable** cycles (both sessions
resolved). The four outcomes are mutually exclusive and exhaustive, so their counts
sum to `total`.

### Conditions & outcomes
| Condition | Outcomes | Partition? | Description |
|---|---|---|---|
| `session1_range` | `broke_high`, `broke_low`, `broke_both`, `neither` | Yes | Breakout direction of session 2 relative to the session 1 range; `total` = countable cycles |

### Parameters
| Parameter | Default | Description |
|---|---|---|
| session1 | london | First session (range former); must be a configured session name |
| session2 | ny | Second session (breakout window); must differ from `session1` |
| breakout_criteria | wick | Break detection: `wick` (intraday extreme) or `close` (bar close beyond the level) |
| close_tolerance_min | 15 | Minutes before a session's end still considered full coverage |

### Timeframes computed
- `daily` (one entry per cycle).

### Baseline
Random directional null (fixed seed, deterministic): each countable cycle's
session 2 move is **reflected** around the session 1 range midpoint with
probability 0.5 (a price `p` maps to `2*mid - p`, which swaps the high and low
extremes). Reflection turns a `broke_high` cycle into a `broke_low` cycle and vice
versa, while `broke_both` and `neither` are direction-symmetric and unchanged. The
null therefore carries NO directional bias, so `broke_high` and `broke_low`
converge to their shared mean and the comparison reveals whether session 2 breaks
**up** more often than **down** beyond a coin flip. Uses
`np.random.default_rng(seed)`.

### i18n
- **title.en**: "Market Session Breakout"
- **title.fr**: "Cassure de session de marché"
- **definition.en**: "Comparing a second market session's high and low to the range a first market session formed, how often does price break above the first session's high only, below its low only, both sides, or neither during the second session?"
- **definition.fr**: "En comparant le plus haut et le plus bas d'une deuxième session de marché au range formé par une première session, à quelle fréquence le prix casse-t-il au-dessus du plus haut de la première session uniquement, en-dessous de son plus bas uniquement, des deux côtés, ou ni l'un ni l'autre pendant la deuxième session ?"

### Slices
- `weekday` — the "by weekday" breakdown (declared via the shared `Weekday` slicer).
- `size` — the "by size" breakdown: session-1-range size quartiles (declared via
  `SizeBucket(column="s1_size")`).
- `levels` — the "by levels" breakdown: how far session 2 extended past the session
  1 range, in multiples of `s1_size` (<0.5x, 0.5–1x, 1–1.5x, 1.5–2x, >=2x; declared
  via `Levels(ref="s1_size", ext="extension")`).
- `rejection` — the "by rejection" breakdown: which session 1 extreme formed first,
  `high_first` / `low_first` (declared via the shared `Rejection` slicer reading
  `high_first`); cycles with an undetermined order are excluded.


---

## 38. Fair Value Gaps

**Family**: `fair_value_gaps`
**Module**: `stats/fair_value_gaps/standard.py`
**Result file**: `results/fair_value_gaps.json`
**Status**: Implemented (standard variant)

### What it measures
On 15-minute RTH candles, a **Fair Value Gap (FVG)** forms when three consecutive
candles leave a price imbalance: candle 1 and candle 3 have no price overlap
(bullish: `c1.high < c3.low`; bearish: `c1.low > c3.high`), with candle 2 as the
impulse body inside the gap. This stat measures **how often each FVG is filled
(mitigated) within the same RTH session**. Reported as a 2×2 conditional
probability matrix: P(filled | bullish FVG), P(not filled | bullish FVG),
P(filled | bearish FVG), P(not filled | bearish FVG). The two outcomes partition
each direction, so they sum to 1.

### Methodology
1. Build the resolved-sessions index via `build_resolved_days` (clean RTH open at
   `rth_start` and a last RTH bar at or after `rth_end - close_tolerance`).
   Early-close days and the final incomplete day are excluded.
2. Aggregate 1-minute OHLCV bars within the RTH window
   `[rth_start_min, rth_end_min)` into **15-minute candles** using bucket index
   `k = (minute_of_day - rth_start_min) // 15`. For each bucket: `open` = first
   bar's open, `high` = max high, `low` = min low, `close` = last bar's close.
3. Slide a **consecutive 3-candle window** (c1, c2, c3) across each session's
   15-min candles with overlapping steps (every qualifying triple is counted):
   - **Bullish FVG**: `c1.high < c3.low`.
     Gap zone `(c1.high, c3.low)`, `gap_pts = c3.low - c1.high`.
   - **Bearish FVG**: `c1.low > c3.high`.
     Gap zone `(c3.high, c1.low)`, `gap_pts = c1.low - c3.high`.
   The two FVG types are mutually exclusive for any given triple.
4. Compute the **gap size**: `gap_size_pct = 100 * gap_pts / c3.close` (percent
   of price at formation). This is the column read by the `size` slicer.
5. **Pending discipline**: an FVG whose c3 is the **last 15-min candle** of its
   session has no subsequent candle to resolve the fill outcome → excluded from
   BOTH numerator and denominator. Every other FVG resolves by session end.
6. **Fill detection**: for each non-pending FVG, evaluate the `fill_threshold_pct`
   target over all 15-min candles after c3 within the same session.  The
   suffix-min-low and suffix-max-high for those candles are computed vectorially
   via a reversed cumulative min/max applied to the within-session shifted series:
   - Bullish target `= c3.low - (pct/100) * gap_pts`.
     At 100 % the target collapses to `c1.high` (full mitigation — price must
     trade back through the gap to the candle that originally bounded it).
     **Filled** when `min(subsequent lows) <= target`.
   - Bearish target `= c3.high + (pct/100) * gap_pts`.
     At 100 % the target collapses to `c1.low`.
     **Filled** when `max(subsequent highs) >= target`.
7. Build the **event table**: one row per resolved FVG event, indexed by the
   normalized session date (duplicate dates expected — multiple FVGs per session
   are all recorded). Columns: `direction` (`"bullish"` / `"bearish"`),
   `gap_size_pct`, `filled` (bool).

`total_samples` = total number of resolved FVG events (not sessions). Each row's
`total` = FVG events of that direction.

### Conditions & outcomes
| Condition | Outcomes | Partition? | Description |
|---|---|---|---|
| `bullish` | `filled`, `not_filled` | Yes | Fill rate among bullish FVGs; `total` = bullish events |
| `bearish` | `filled`, `not_filled` | Yes | Fill rate among bearish FVGs; `total` = bearish events |

### Parameters
| Parameter | Default | Description |
|---|---|---|
| fill_threshold_pct | 100.0 | Fill target as a percent of the gap (100 = full mitigation, 50 = halfway into the gap) |
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `15min` (single entry under `instruments.{INSTRUMENT}.15min`).

### Baseline
Random null (fixed seed, deterministic): the `filled` column is **permuted**
across all FVG events. This preserves the pooled fill rate but destroys its
association with the gap direction — the null hypothesis that bullish and bearish
FVGs fill at the same rate as the market-wide average. Each condition's
`baseline_prob` therefore converges to the pooled fill rate. A fair-coin baseline
(50 %) would be inappropriate here because FVGs fill well above 50 % of the time;
the permutation baseline is the correct direction-agnostic null. Uses
`np.random.default_rng(seed)`.

### Slices
- `weekday` — the "by weekday" breakdown (declared via the shared `Weekday`
  slicer; reads `index.dayofweek` on the event table — same session date → same
  weekday for all FVGs within a session).
  `gap_size_pct` (`SizeBucket(column="gap_size_pct", buckets=[0.0, 0.024, 0.049,
  0.089, 0.149, 0.25, inf], name="size")`). The six buckets are: 0–0.024 %,
  0.025–0.049 %, 0.05–0.089 %, 0.09–0.149 %, 0.15–0.25 %, >0.25 %.

### Future variants (not in MVP)
- `by_time` — which intraday time bucket the FVG forms in (first hour, mid-session,
  power hour), using a `SizeBucket` on the c3 bucket index.
- `by_direction` — whether the FVG is with or against the prior daily trend.
- `by_partial_fill` — varying `fill_threshold_pct` (e.g. 25 %, 50 %, 75 %, 100 %)
  to show how fill rate degrades as the threshold tightens.
- `by_gap_age` — how many candles after formation the gap is finally filled (a
  magnitude stat using the bucket offset to fill).


---

## 39. Market Session Correlation

**Family**: `market_session_correlation`
**Module**: `stats/market_session_correlation/standard.py`
**Result file**: `results/market_session_correlation.json`
**Status**: Implemented (standard variant)

### What it measures
How often does a **second** market session close green (or red) given the color a
**first** market session closed, within the same trading cycle? Intraday color
follow-through between two named sessions, reported as a 2x2 conditional matrix:
P(s2 green | s1 green), P(s2 red | s1 green), P(s2 green | s1 red), P(s2 red | s1 red).

Both sessions are named windows read from the instrument config — the single source
of truth for their bounds — so any ordered pair works; the default is `london`
(session 1) → `ny` (session 2). This is the within-cycle, two-session companion to
`prev_session_correlation` (section 5), which instead correlates a session with the
chronologically PREVIOUS session of the same kind.

### Methodology
1. Attribute every bar to a **cycle** (the RTH session date). An intraday session
   (`start < end`) maps to its own calendar date; a cross-midnight session
   (`start >= end`, e.g. `asia`) maps its evening bars to the *next* day's cycle and
   its early bars to the *same* day's cycle. Session 1 and session 2 are joined on
   this shared cycle date, so the pair must be ordered within a cycle.
2. Build each session's per-cycle candle (`open` = first bar's open, `close` = last
   bar's close, plus `high` / `low`). A session is **resolved** for a cycle when it
   has a clean open bar (offset 0 from `start`) AND end coverage (last bar at or
   after `duration - close_tolerance`, with `duration = (end - start) mod 1440`).
3. Inner-join both sessions on the cycle date — a cycle survives only when BOTH are
   resolved (pending-sample discipline).
4. Classify each session as **green** or **red** per the `performance` mode:
   - `close_to_close` (default): green if `close >= the previous cycle's close for
     that same session`. The first joined cycle has no prior close for either
     session and is excluded.
   - `open_to_close`: green if `close >= open`.
5. Report the four rows: s1-green→s2-green, s1-green→s2-red, s1-red→s2-green,
   s1-red→s2-red.

`total_samples` and each row's `total` count the **countable** cycles (both sessions
resolved, prior close present in `close_to_close`). The two outcomes per condition
partition the cycles carrying that session 1 color, so their counts sum to `total`.

### Conditions & outcomes
| Condition | Outcomes | Partition? | Description |
|---|---|---|---|
| `s1_green` | `green`, `red` | Yes | Session 2 color given session 1 closed green |
| `s1_red` | `green`, `red` | Yes | Session 2 color given session 1 closed red |

### Parameters
| Parameter | Default | Description |
|---|---|---|
| session1 | london | First session (the condition); must be a configured session name |
| session2 | ny | Second session (the outcome); must differ from `session1` |
| performance | close_to_close | Session-direction basis: `close_to_close` or `open_to_close` |
| close_tolerance_min | 15 | Minutes before a session's end still considered full coverage |

### Timeframes computed
- `daily` (one entry per cycle).

### Baseline
Each cycle's session 1 and session 2 are recolored independently at random (50/50
green/red), so session 2's color is independent of session 1's. Expected baseline:
~50 % for every row. Fixed seed for reproducibility (`np.random.default_rng(seed)`).

### i18n
- **title.en**: "Market Session Correlation"
- **title.fr**: "Corrélation de session de marché"
- **definition.en**: "How often does a second market session close green or red given the color a first market session closed, within the same trading cycle?"
- **definition.fr**: "À quelle fréquence une deuxième session de marché clôture-t-elle en vert ou en rouge selon la couleur de clôture d'une première session, au cours du même cycle de trading ?"

### Slices
- `weekday` — the "by weekday" breakdown (declared via the shared `Weekday` slicer).
- `size` — the "by size" breakdown: session-1-range size quartiles (declared via
  `SizeBucket(column="s1_size")`).


---

## 40. Daily High / Low Session

**Family**: `daily_high_low_session`
**Module**: `stats/daily_high_low_session/standard.py`
**Result file**: `results/daily_high_low_session.json`
**Status**: Implemented (standard variant)

### What it measures
For each 24-hour trading cycle, which intraday market session produced the
cycle's **highest high** and which produced the **lowest low**? The configured
sessions (default: `asia`, `london`, `ny`) partition the full cycle with no
overlap; together they cover the complete 24-hour period. The stat reports the
fraction of cycles where each session "owns" each daily extreme. Probabilities
sum to 1 within each condition (every cycle's extreme belongs to exactly one
session).

### Methodology
1. **Cycle attribution**: bars are tagged to the RTH session date they belong
   to. A cross-midnight session (`start >= end`, e.g. `asia` 18:00→03:00)
   maps its evening bars to the NEXT calendar day's cycle and its early bars
   to the SAME day's cycle. An intraday session (`start < end`) maps to its
   own calendar date. All configured sessions are joined on this shared cycle
   date, so the session list must be ordered chronologically within a cycle.
2. **Session resolution**: for each cycle, a session is **resolved** when (a)
   it has a clean open bar (offset 0 from the session's `start`) AND (b) its
   last bar falls within `close_tolerance_min` minutes of the session's
   scheduled end. `duration = (end - start) mod 1440` (1440 if equal).
3. **Countable cycles**: a cycle is countable only when ALL configured sessions
   are resolved. Any missing or unresolved session drops the entire cycle from
   the denominator (pending-sample discipline). The cycle tables are inner-joined
   on the cycle date.
4. **Extreme attribution**: for each countable cycle, `high_session` is the
   session whose `max(high)` equals the overall cycle max. `low_session` is the
   session whose `min(low)` equals the overall cycle min. On exact ties (two
   sessions share the same extreme value), the first session in the configured
   order is credited — `np.argmax` / `np.argmin` return the first occurrence.
5. **Daily candle**: `day_green = True` when the cycle's overall close (last
   bar across all sessions) is `>=` the cycle's overall open (first bar across
   all sessions). Used by the `candle` slice.

### Conditions & outcomes
| Condition | Outcomes | Partition? | Description |
|---|---|---|---|
| `daily_high` | one per session | Yes | Which session contained the cycle's max high; `total` = countable cycles |
| `daily_low` | one per session | Yes | Which session contained the cycle's min low; `total` = countable cycles |

Both conditions have the same `total` (all countable cycles). Counts for a
condition sum to `total`, so probabilities sum to 1.

### Parameters
| Parameter | Default | Description |
|---|---|---|
| sessions | `("asia", "london", "ny")` | Ordered session names; must be at least 2, distinct, and all present in the instrument config. Chronological order matters: first session wins on exact ties. |
| close_tolerance_min | 15 | Minutes before a session's scheduled end still considered full coverage |

### Timeframes computed
- `daily` (one entry per countable cycle).

### Baseline
Duration-weighted random null (fixed seed, deterministic). For each countable
cycle, the day's high (and independently the day's low) is assigned to a
randomly drawn session with probability proportional to that session's bar
count `n_{session}` in the cycle. Rationale: under a structureless random
walk a price extreme is equally likely at any minute, so the probability a
session captures an extreme should equal its share of the cycle's total bar
count (time). The stat's edge is whether a session captures the high or low
MORE than its duration share predicts.

Implementation: build a per-row normalized weight matrix from the `n_{session}`
columns, compute cumulative sums, draw `rng.random(n)` for high and again for
low, pick the first session whose cumulative weight exceeds the draw
(`(u[:, None] < cum).argmax(axis=1)`). A tmp copy of the day table with
replaced `high_session` / `low_session` columns is passed to `compute_rows`.
Uses `np.random.default_rng(seed)`.

### i18n
- **title.en**: "Daily High / Low Session"
- **title.fr**: "Session du plus haut / plus bas journalier"
- **definition.en**: "For each trading cycle, which intraday market session produced the day's highest high and which produced the day's lowest low? Reported as a probability distribution over the configured sessions; probabilities sum to 1 within each condition."
- **definition.fr**: "Pour chaque cycle de trading, quelle session de marché intrajournalière a produit le plus haut du jour et laquelle a produit le plus bas du jour ? Présenté comme une distribution de probabilité sur les sessions configurées ; les probabilités somment à 1 par condition."

### Slices
- `weekday` — the "by weekday" breakdown (declared via the shared `Weekday`
  slicer; reads `index.dayofweek` on the day table).
- `candle` — split by daily candle color: green (`cycle_close >= cycle_open`)
  or red. Declared via the module-local `_DailyCandle` slicer, a subclass of
  the public `Close` slicer reading the `day_green` boolean column.


---

## 41. Session Range by Weekday

**Family**: `session_range_weekday`
**Module**: `stats/session_range_weekday/standard.py`
**Result file**: `results/session_range_weekday.json`
**Status**: Implemented

### What it measures
For each weekday: the **average price range** (`high − low`, in points) of the
Asia, London, and NY geographic market sessions. The overall result aggregates
all countable cycles; the per-weekday breakdown is produced by the declared
`weekday` slice.

This is a **magnitude** stat: all three rows carry a continuous metric in
`value` (and its random-baseline counterpart in `value_baseline`). The
`probability` / `baseline_prob` channel is left at `0.0` for every row. Every
row reports its sample size in `count` / `total`.

### Methodology
1. **Cycle attribution**: bars are tagged to the RTH session date they belong
   to via `session_bars` (cross-midnight sessions such as `asia` 18:00→03:00
   attribute their evening bars to the NEXT calendar day's cycle).
2. **Session resolution**: for each cycle, a session is **resolved** when (a)
   it has a clean open bar (offset 0 from the session's `start`) AND (b) its
   last bar falls within `close_tolerance_min` minutes of the session's
   scheduled end. `duration = (end − start) mod 1440` (1440 if equal).
3. **Countable cycles**: a cycle is countable only when ALL three sessions are
   resolved. The three session tables are inner-joined on the cycle date —
   any missing session drops the entire cycle (pending-sample discipline). This
   keeps N uniform across all three outcomes.
4. **Per-session range**: for each countable cycle,
   - `asia_range`   = `asia_high   − asia_low`
   - `london_range` = `london_high − london_low`
   - `ny_range`     = `ny_high     − ny_low`
5. Over all countable cycles, report three rows under the single `any_day`
   condition: `mean_asia_range`, `mean_london_range`, `mean_ny_range`.
6. Re-run the same three rows per weekday via the `weekday` slice.

### Conditions
| Condition key | Description |
|---|---|
| `any_day` | All countable cycles (the `weekday` slice does the per-day breakdown) |

### Outcomes
| Outcome key | Channel | Description |
|---|---|---|
| `mean_asia_range`   | `value` | Average Asia session range (`high − low`), in points |
| `mean_london_range` | `value` | Average London session range (`high − low`), in points |
| `mean_ny_range`     | `value` | Average NY session range (`high − low`), in points |

### Parameters
| Parameter | Default | Description |
|---|---|---|
| `asia` | `"asia"` | Config session name for the Asia window |
| `london` | `"london"` | Config session name for the London window |
| `ny` | `"ny"` | Config session name for the NY window |
| `close_tolerance_min` | 15 | Minutes before a session's scheduled end still considered full coverage |

### Timeframes computed
- `daily` (one entry per countable cycle).

### Baseline
Random null: for a subset of `N` cycles (one weekday, or the whole table),
draw `N` cycles uniformly at random **without replacement** from the full
countable-cycle table and compute the same three averages on that random
sample. This represents the null hypothesis that the weekday carries no
information — its expected session ranges equal the grand mean across all
cycles. Uses `np.random.default_rng(seed)` for deterministic output. The
baseline is computed per slice group as well as overall; for the overall result
the sample is a permutation of the full table, so its `value_baseline` equals
the overall `value`.

### i18n
- **title.en**: "Session Range by Weekday"
- **title.fr**: "Amplitude de session par jour de la semaine"
- **definition.en**: "What is the average price range (high − low, in points) of the Asia, London, and NY sessions for each weekday?"
- **definition.fr**: "Quelle est l'amplitude moyenne (plus haut − plus bas, en points) des sessions asiatique, de Londres et de New York pour chaque jour de la semaine ?"

### Slices
- `weekday` — implemented (declared via `slices = ("weekday",)`)


---

## 42. Session Volume by Weekday

**Family**: `session_volume_weekday`
**Module**: `stats/session_volume_weekday/standard.py`
**Result file**: `results/session_volume_weekday.json`
**Status**: Implemented

### What it measures
For each weekday: the **average total traded volume** of the Asia, London, and
NY geographic market sessions. The overall result aggregates all countable
cycles; the per-weekday breakdown is produced by the declared `weekday` slice.

This is a **magnitude** stat: all three rows carry a continuous metric in
`value` (and its random-baseline counterpart in `value_baseline`). The
`probability` / `baseline_prob` channel is left at `0.0` for every row. Every
row reports its sample size in `count` / `total`.

### Methodology
1. **Cycle attribution**: bars are tagged to the RTH session date they belong
   to via `session_bars` (cross-midnight sessions such as `asia` 18:00→03:00
   attribute their evening bars to the NEXT calendar day's cycle).
2. **Session resolution**: for each cycle, a session is **resolved** when (a)
   it has a clean open bar (offset 0 from the session's `start`) AND (b) its
   last bar falls within `close_tolerance_min` minutes of the session's
   scheduled end. `duration = (end − start) mod 1440` (1440 if equal).
3. **Countable cycles**: a cycle is countable only when ALL three sessions are
   resolved. The three session tables are inner-joined on the cycle date —
   any missing session drops the entire cycle (pending-sample discipline). This
   keeps N uniform across all three outcomes.
4. **Per-session volume**: for each countable cycle,
   - `asia_volume`   = sum of the Asia session bars' `volume`
   - `london_volume` = sum of the London session bars' `volume`
   - `ny_volume`     = sum of the NY session bars' `volume`
5. Over all countable cycles, report three rows under the single `any_day`
   condition: `mean_asia_volume`, `mean_london_volume`, `mean_ny_volume`.
6. Re-run the same three rows per weekday via the `weekday` slice.

### Conditions
| Condition key | Description |
|---|---|
| `any_day` | All countable cycles (the `weekday` slice does the per-day breakdown) |

### Outcomes
| Outcome key | Channel | Description |
|---|---|---|
| `mean_asia_volume`   | `value` | Average total traded volume of the Asia session |
| `mean_london_volume` | `value` | Average total traded volume of the London session |
| `mean_ny_volume`     | `value` | Average total traded volume of the NY session |

### Parameters
| Parameter | Default | Description |
|---|---|---|
| `asia` | `"asia"` | Config session name for the Asia window |
| `london` | `"london"` | Config session name for the London window |
| `ny` | `"ny"` | Config session name for the NY window |
| `close_tolerance_min` | 15 | Minutes before a session's scheduled end still considered full coverage |

### Timeframes computed
- `daily` (one entry per countable cycle).

### Baseline
Random null: for a subset of `N` cycles (one weekday, or the whole table),
draw `N` cycles uniformly at random **without replacement** from the full
countable-cycle table and compute the same three averages on that random
sample. This represents the null hypothesis that the weekday carries no
information — its expected session volumes equal the grand mean across all
cycles. Uses `np.random.default_rng(seed)` for deterministic output. The
baseline is computed per slice group as well as overall; for the overall result
the sample is a permutation of the full table, so its `value_baseline` equals
the overall `value`.

### i18n
- **title.en**: "Session Volume by Weekday"
- **title.fr**: "Volume de session par jour de la semaine"
- **definition.en**: "What is the average total traded volume of the Asia, London, and NY sessions for each weekday?"
- **definition.fr**: "Quel est le volume total moyen échangé des sessions asiatique, de Londres et de New York pour chaque jour de la semaine ?"

### Slices
- `weekday` — implemented (declared via `slices = ("weekday",)`)


---

## 43. CPI Reaction

**Family**: `cpi_reaction`
**Module**: `stats/cpi_reaction/standard.py`
**Result file**: `results/cpi_reaction.json`
**Status**: Implemented

### What it measures
On CPI release dates, given the direction of the initial reaction candle
(the 08:30–09:30 ET pre-RTH window that captures the immediate post-release
move), how often does the RTH day close in the same direction? Reported as a
2×2 conditional matrix: P(green day | green reaction), P(red day | green
reaction), P(green day | red reaction), P(red day | red reaction).

### Methodology
1. Build RTH daily candles via `build_resolved_days` (`session_open`,
   `session_close`), using the standard resolution rule (clean session-open
   bar and a last RTH bar at or after `session_end - close_tolerance_min`).
   **Day color**: `day_green = session_close >= session_open`.
2. **Reaction candle** = the intraday window `[reaction_start_min, reaction_end_min)`
   in ET on the release date, computed from the 1-min `candles_df`:
   - `reaction_open` = `open` of the bar at EXACTLY `reaction_start_min`.
     If that bar is missing for a date, the reaction is undefined → date dropped.
   - `reaction_close` = `close` of the LAST bar in the window for that date.
   - `reaction_green = reaction_close >= reaction_open`.
   Duplicate bars at the exact start minute are de-duplicated (`keep="first"`).
3. **CPI release dates** loaded from the external economic calendar via
   `load_cpi_release_dates(calendar_path)` (reused from `cpi_performance`).
   Injected into `__init__` as `event_dates`; normalized to tz-naive midnight
   Timestamps for date-level alignment.
4. **day_table**: one row per CPI release date that has BOTH (a) a resolved
   RTH session AND (b) a complete reaction candle. Columns: `reaction_green`
   (bool), `day_green` (bool). Dates failing either requirement are excluded
   from all denominators (pending-sample discipline).
5. **compute_rows**: 2×2 over `_CONDITIONS = (("reaction_green", True),
   ("reaction_red", False))` and `_OUTCOMES = (("green", True), ("red", False))`.
   All four rows are always emitted even when the table is empty.

### Parameters
| Parameter | Default | Description |
|---|---|---|
| `reaction_start` | `"08:30"` | Reaction window start time (HH:MM ET) |
| `reaction_end` | `"09:30"` | Reaction window end time HH:MM ET; must be after `reaction_start` |
| `close_tolerance_min` | `15` | Minutes before RTH end still considered a full close |

### Timeframes computed
- `daily` (one entry; CPI events are date-keyed, not intraday-timeframe-keyed).

### Baseline
Each qualifying day's RTH color (`day_green`) is replaced by an independent
fair-coin flip (`np.random.default_rng(seed)`, p=0.5); the reaction condition
stays from real data. Expected baseline_prob ≈ 0.5 for every row. Deterministic
for a fixed seed.

### i18n
- **title.en**: "CPI Reaction"
- **title.fr**: "Réaction CPI"
- **definition.en**: "On CPI release dates, given the initial reaction candle's direction (08:30–09:30 ET move after the release), how often does the RTH day close in the same direction?"
- **definition.fr**: "Les jours de publication du CPI, selon la direction de la bougie de réaction initiale (mouvement 08:30–09:30 ET après la publication), à quelle fréquence la séance RTH clôture-t-elle dans la même direction ?"

### Slices
None. The CPI Reaction stat declares `slices = ()`. Events are scattered across
the calendar; per-day slicing is meaningless.

---

## 44. FOMC Intraday

**Family**: `fomc_intraday`
**Module**: `stats/fomc_intraday/standard.py`
**Result file**: `results/fomc_intraday.json`
**Status**: Implemented

### What it measures
On FOMC decision days, how does price perform within each 15-minute RTH
interval? Days are split into positive vs negative reaction groups based on the
direction of the 2pm ET (14:00) interval — the window that captures the
immediate market response to the Fed's rate decision. Each interval reports its
average % change, average $ change, and average volume, both overall (all FOMC
days) and per reaction group.


### Methodology
1. **Interval grid**: the RTH session `[rth_start_min, rth_end_min)` is split
   into consecutive 15-minute intervals (`range(rth_start, rth_end, 15)`). For
   NQ (09:30–16:15) this is 27 intervals, keyed `i0930`, `i0945`, …, `i1600`.
   No times are hardcoded — the grid and its labels are derived from config.
2. **Per-interval metrics** for interval `[t, t+15)` on a given day, from the
   1-min `candles_df`:
   - `interval_open` = `open` of the bar at EXACTLY minute `t`. If that bar is
     missing, the interval is undefined for that day → its metrics are NaN
     (pending-sample discipline applied at the **interval** level, not the day).
   - `interval_close` = `close` of the LAST bar in `[t, t+15)`.
   - `pct_{key}` = `(interval_close - interval_open) / interval_open`
   - `dollar_{key}` = `interval_close - interval_open`
   - `vol_{key}` = sum of 1-min bar volumes in `[t, t+15)` (NaN when the open
     bar is missing, for consistent pending discipline).
3. **Day qualification**: a date is included iff it is an RTH-resolved session
   (via `build_resolved_days`: clean session-open bar AND a last RTH bar at or
   after `session_end - close_tolerance_min`) AND an FOMC decision date.
4. **FOMC decision dates** loaded from the external economic calendar via
   `load_fomc_release_dates(calendar_path)` (reused from `fomc_performance`).
   Injected into `__init__` as `event_dates`; normalized to tz-naive midnight
   Timestamps for date-level alignment.
5. **Reaction flag**: `reaction_positive` = direction of the 14:00 interval —
   `1.0` when strictly up (`close > open`), `0.0` when flat or down, `NaN` when
   the 14:00 bar (or its window close) is missing. Days with `NaN` stay in the
   overall results but are excluded from both reaction groups.
6. **day_table**: one row per qualifying FOMC date, columns `pct_{key}`,
   `dollar_{key}`, `vol_{key}` for every interval plus `reaction_positive`.
7. **compute_rows**: for each interval (`condition`) × metric (`outcome` ∈
   `pct_change`, `dollar_change`, `volume`) emit one row with `value` = mean of
   non-NaN observations, `total` = N non-NaN days, `count` = up-count (`val >= 0`)
   for pct/dollar or `= total` for volume, `probability = count / total`. All
   rows are always emitted (27 × 3 = 81 for NQ) for a deterministic shape.

### Parameters
| Parameter | Default | Description |
|---|---|---|
| `close_tolerance_min` | `15` | Minutes before RTH end still considered a full close |

The reaction window is fixed at 14:00 ET (the conventional FOMC announcement
time). The interval size is 15 minutes.

### Timeframes computed
- `daily` (one entry; FOMC events are date-keyed, not intraday-timeframe-keyed).

### Baseline
The sign of each non-NaN `pct_change` / `dollar_change` observation is
independently randomized (`np.random.default_rng(seed)`, ±1 with p=0.5), holding
magnitudes fixed; expected mean ≈ 0 and expected up-fraction ≈ 0.5 per interval.
Volume has no directional baseline, so `value_baseline` is `None` for every
volume row. Deterministic for a fixed seed.

### i18n
- **title.en**: "FOMC Intraday"
- **title.fr**: "FOMC Intrajournalier"
- **definition.en**: "On FOMC decision days, how does price perform within each 15-minute RTH interval? Results are split by the direction of the 2pm ET reaction window (the bar that captures the immediate response to the Fed's rate decision)."
- **definition.fr**: "Les jours de décision du FOMC, comment le prix évolue-t-il dans chaque intervalle de 15 minutes de la session RTH ? Les résultats sont divisés selon la direction de la fenêtre de réaction de 14h ET (la bougie capturant la réponse immédiate à la décision de taux de la Fed)."

Interval condition labels are generated dynamically from the grid (e.g. `i0930`
→ "09:30–09:45"). Outcome labels: `pct_change` → "Average % change", `dollar_change`
→ "Average $ change", `volume` → "Average volume".

### Slices
- `reaction` (dimension "2pm ET reaction"): splits qualifying days into
  `positive` (14:00 interval closed strictly up) and `negative` (flat or down).
  Days missing the 14:00 bar are excluded from both groups. This is the core
  reaction split; the top-level results cover all FOMC days aggregated.

### Future variants (not in MVP)
- `by_individual_days` — per-date breakdown of each FOMC day's intraday profile

---

## 45. Economic Data Volume

**Family**: `economic_data_volume`
**Module**: `stats/economic_data_volume/standard.py`
**Result file**: `results/economic_data_volume.json`
**Status**: Implemented (standard variant)

### What it measures
The average total RTH daily traded volume on economic-event days compared to
non-event days. Five event conditions are reported — CPI, FOMC, NFP and GDP
release days individually, plus an `any_event` union — alongside a `non_event`
condition (sessions carrying none of the four releases).

### Methodology
1. Build the RTH resolved-day table (shared `build_resolved_days`): one row per
   resolved session (clean 09:30 open bar AND a last RTH bar at or after
   `session_end - close_tolerance_min`), sorted chronologically. Early-close days
   and the final incomplete day are excluded (pending discipline).
2. Each resolved day's `volume` is the sum of its RTH bar volumes.
3. Release dates come from an **external economic calendar CSV** (not from the
   OHLCV data). Per type the headline US print is matched and the distinct
   release dates injected:
   - CPI — `load_cpi_release_dates` (`event == "CPI m/m"`, `currency == "USD"`).
   - FOMC — `load_fomc_release_dates` (`event == "Federal Funds Rate"`).
   - NFP — `load_nfp_release_dates` (`event == "Non-Farm Employment Change"`).
   - GDP — `load_gdp_release_dates` (`event in {Advance, Prelim, Final} GDP q/q`;
     the "GDP Price Index" deflator lines are deliberately not matched).
4. A resolved session is flagged for an event type when its date matches one of
   that type's release dates. `is_any_event` is the union of the four flags;
   `is_non_event` is its complement (a day with no matched release).
5. For each of the six conditions, report a single magnitude outcome:
   - **mean_volume** — average summed RTH volume over that condition's days
     (carried in `value`; the `probability` channel is left at `0.0`).

Each condition reports its own sample size in `count` / `total`. All 6 rows are
always emitted; a condition with no days carries `count = total = 0` and
`value = 0.0`. `total_samples` is the number of resolved days (the whole
population), not the per-condition count.

### Parameters
| Parameter | Default | Description |
|---|---|---|
| calendar_path | `data/forex_factory_calendar.csv` | Economic calendar CSV with the release dates |
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `daily` (the RTH daily candle; a single entry).

### Baseline
Random null (fixed seed, deterministic): per condition holding `N` days, draw `N`
days uniformly at random (without replacement) from the whole resolved-day table
and average their volume. Every condition's expected baseline equals the grand
mean across all sessions, so an actual event-day mean well above its baseline
indicates a genuine volume lift around the release rather than an artifact of the
day count. Draws are sequential over the fixed condition order with
`np.random.default_rng(seed)`.

### i18n
- **title.en**: "Economic Data Volume"
- **title.fr**: "Volume des données économiques"
- **definition.en**: "How does the average total RTH daily volume on economic-event days (CPI, FOMC, NFP, GDP) compare to non-event days?"
- **definition.fr**: "Comment le volume quotidien RTH total moyen des jours d'annonces économiques (CPI, FOMC, NFP, GDP) se compare-t-il aux jours sans annonce ?"

### Slices
- None. Releases are scattered across the calendar, so the per-day slicers
  (weekday, close color, …) do not apply; `slices = ()`.


## 46. Asian Range Breakout

**Family**: `asian_range_breakout`
**Module**: `stats/asian_range_breakout/standard.py`
**Result file**: `results/asian_range_breakout.json`
**Status**: Implemented (standard variant)

### What it measures
The **Asian range** is the high-low range of the Asian (Tokyo) session that
precedes a given RTH session (NQ: 18:00 ET previous day → 03:00 ET, read from the
config `asia` session window). Given that range, which direction does price break
during the **RTH session** (09:30–16:15)? Reported under a single `asian_range`
condition with four mutually exclusive outcomes that partition every countable
day: `broke_high`, `broke_low`, `broke_both`, `neither`. Their probabilities sum
to 1. The marginal high-break / low-break rates are recoverable as
`broke_high + broke_both` and `broke_low + broke_both`. This is the Asian-session
sibling of `overnight_range_breakout` (which uses the wider 18:00→09:30 overnight
window).

### Methodology
1. Build the RTH daily candle per resolved session (`session_open`,
   `session_close`) via `build_resolved_days` — the same resolution rule as the
   other daily stats (clean 09:30 open and a last RTH bar at or after
   `session_end - close_tolerance`).
2. Compute the **Asian range** from the `asia` session bars, attributed to the RTH
   session date they precede via `session_bars` (`stats/utils/session_candles.py`).
   The session crosses midnight, so evening bars (`hour*60+minute >= asia_start`,
   i.e. ≥ 18:00) are tagged to the *next* calendar day's cycle and early bars
   (`hour*60+minute < asia_end`, i.e. < 03:00) to the *same* calendar day's cycle.
   Grouping the tagged bars by `_cycle`:
   - `ar_high` = max of `high`, `ar_low` = min of `low`.
   - `ar_size` = `ar_high - ar_low`.
3. Compute the **RTH session** extremes over `[rth_start, rth_end)`:
   - wick criteria (default): `day_high` = max `high`, `day_low` = min `low`.
   - close criteria: `day_close_high` = max `close`, `day_close_low` = min `close`.
   Join the Asian range and the RTH extremes onto the resolved-days index (inner
   join — a day survives only with a resolved RTH session AND a prior Asian range).
4. Classify the breakout direction with **strict** inequalities (touching a level
   exactly is NOT a break), using the extremes selected by `breakout_criteria`:
   - `broke_high`: `high > ar_high` AND `low >= ar_low` (broke up only).
   - `broke_low`:  `low < ar_low`  AND `high <= ar_high` (broke down only).
   - `broke_both`: `high > ar_high` AND `low < ar_low` (broke both sides).
   - `neither`:    `high <= ar_high` AND `low >= ar_low` (held inside all session).

`total_samples` counts all resolved sessions; each row's `total` is the
**countable** days (a resolved RTH session and a prior Asian range). The four
outcomes are mutually exclusive and exhaustive, so their counts sum to `total`.

### Conditions & outcomes
| Condition | Outcomes | Partition? | Description |
|---|---|---|---|
| `asian_range` | `broke_high`, `broke_low`, `broke_both`, `neither` | Yes | Breakout direction of the RTH session relative to the prior Asian range; `total` = countable days |

### Parameters
| Parameter | Default | Description |
|---|---|---|
| breakout_criteria | wick | Break detection: `wick` (intraday extreme) or `close` (bar close beyond the level) |
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `daily` (the RTH daily candle; a single entry).

### Baseline
Random directional null (fixed seed, deterministic): each countable day's RTH move
is **reflected** around its Asian-range midpoint with probability 0.5 (a price `p`
maps to `2*mid - p`, which swaps the high and low extremes). Reflection turns a
`broke_high` day into a `broke_low` day and vice versa, while `broke_both` and
`neither` are direction-symmetric and unchanged. The null therefore carries NO
directional bias, so `broke_high` and `broke_low` converge to their shared mean and
the comparison reveals whether the Asian range breaks **up** more often than
**down** beyond a coin flip. Uses `np.random.default_rng(seed)`.

### i18n
- **title.en**: "Asian Range Breakout"
- **title.fr**: "Cassure du range asiatique"
- **definition.en**: "Comparing each session's market-hours high and low to the prior Asian (Tokyo) session's high and low, how often does price break above the Asian high only, below the Asian low only, both sides, or neither during the RTH session?"
- **definition.fr**: "En comparant le plus haut et le plus bas de la séance régulière au plus haut et au plus bas de la session asiatique (Tokyo) précédente, à quelle fréquence le prix casse-t-il au-dessus du plus haut asiatique uniquement, en-dessous du plus bas uniquement, des deux côtés, ou ni l'un ni l'autre pendant la séance RTH ?"

### Slices
- `weekday` — the "by weekday" breakdown (declared via the shared `Weekday` slicer).
- `close` — the "by close" breakdown: session close color, green/red (declared via
  the shared `Close` slicer reading `session_green`).
- `prev_candle` — the "by prev candle" breakdown: prior session color, green/red
  (declared via the shared `PrevCandle` slicer reading `prev_session_green`).
- `size` — the "by size" breakdown: Asian-range size quartiles (declared via
  `SizeBucket(column="ar_size")`).
- `levels` — the "by levels" breakdown: how far the breakout extended past the
  Asian range, in multiples of `ar_size` (<0.5x, 0.5–1x, 1–1.5x, 1.5–2x, >=2x;
  declared via `Levels(ref="ar_size", ext="extension")`).


---

## 47. Candle Body Ratio

**Family**: `candle_body_ratio`
**Module**: `stats/candle_body_ratio/standard.py`
**Result file**: `results/candle_body_ratio.json`
**Status**: Implemented (standard variant)

### What it measures
For each intraday time bucket of the RTH session: the fraction of resolved days
whose bucket candle has a **body** (`|close - open|`) covering at least
`candle_size` percent of its full **range** (`high - low`). A high body ratio
marks a decisive, trend-like candle; a low ratio an indecisive, wick-heavy one.
Bucketed by time slot, the stat shows when in the session decisive candles are
most (and least) likely to print.

This is a **probability** stat. Each intraday bucket is a **condition** (key
`HHMM`, e.g. `0930`); the single **outcome** `large_body` carries the fraction of
that bucket's days whose candle had a large body. The `value` channel is left
`None` on every row.

### Methodology
1. Build the resolved-days table via `build_resolved_days` (clean session open at
   exactly `rth_start` and a last RTH bar at or after `rth_end - close_tolerance`).
   Early-close days and the final incomplete day are excluded (pending discipline).
2. Over the same RTH bar filter (`hour*60+minute` in `[rth_start_min, rth_end_min)`),
   assign each bar a bucket index `(minute_of_day - rth_start_min) // bucket_min`.
   The final bucket may be partial when the session length is not a whole multiple
   of `bucket_min` (NQ 09:30–16:15 with 15-minute buckets ends with the 16:00–16:15
   bucket).
3. Per `(day, bucket)` build the candle: `open` = open of the bucket's earliest
   bar, `close` = close of its latest bar, `high` = max `high`, `low` = min `low`.
   Compute `ratio = |close - open| / (high - low)`, defined as `0.0` for a flat
   candle (`high == low`). The candle is a **large body** when
   `ratio >= candle_size / 100`. Pivot to one row per resolved day with one
   `body_{b}` column per bucket holding `1.0` (large), `0.0` (not large), or `NaN`
   (bucket absent that day, excluded from the denominator).
4. For each bucket present in the subset, report one `large_body` row whose
   `probability` is the large-body rate over the contributing days. `count` =
   large-body days; `total` = contributing days.
5. Re-run the same row per weekday via the `weekday` slice.

### Conditions
| Condition key | Description |
|---|---|
| `HHMM` (e.g. `0930`, `0945`, …) | One per intraday time bucket present, keyed by the bucket's start time |

### Outcomes
| Outcome key | Channel | Description |
|---|---|---|
| `large_body` | `probability` | Fraction of the bucket's days whose candle body is ≥ `candle_size`% of its range |

### Parameters
| Parameter | Default | Description |
|---|---|---|
| timeframe | `15min` | Bucket granularity (`1min`, `5min`, `15min`, `30min`, `1h`) |
| candle_size | 50 | Minimum body-to-range ratio percentage (must be in `[0, 100]`) |
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- One per run, keyed by the chosen bucket granularity (default `15min`).

### Baseline
Random null: all per-`(day, bucket)` large-body flags in the subset are pooled.
For a bucket contributing `n` days, `n` flags are drawn uniformly at random —
without replacement — from the pool and their mean is the bucket's baseline rate.
This represents the null hypothesis that the intraday time bucket carries no
information — its expected large-body rate equals the grand mean across all
buckets. Uses `np.random.default_rng(seed)` for deterministic output. The baseline
is computed per slice group as well as overall.

### i18n
- **title.en**: "Candle Body Ratio"
- **title.fr**: "Ratio de corps de bougie"
- **definition.en**: "For each intraday time bucket: the fraction of days whose candle body covers at least the configured percentage of its high-to-low range. Shows when in the session decisive, trend-like candles are most likely to print."
- **definition.fr**: "Pour chaque tranche horaire intraday : la fraction des jours dont le corps de la bougie couvre au moins le pourcentage configuré de son amplitude haut-bas. Montre à quel moment de la séance les bougies décisives, de tendance, sont les plus probables."

### Slices
- `weekday` — the "by weekday" breakdown (declared via the shared `Weekday` slicer).


---

## 48. Average Consecutive Bars

**Family**: `avg_consecutive_bars`
**Module**: `stats/avg_consecutive_bars/standard.py`
**Result file**: `results/avg_consecutive_bars.json`
**Status**: Implemented (standard variant)

### What it measures
For each resolved trading day, on the intraday chart at a given `timeframe`,
find the **longest run of consecutive green bars** and the **longest run of
consecutive red bars** within the RTH session. Then average those per-day
maximums across all resolved days. The output is, per color, the mean of the
daily-max streak lengths.

This is a **magnitude** stat: each row's `value` holds the mean of daily
maximum streak lengths; `value_baseline` holds the permutation-null baseline.
The `probability` channel is always `0.0` (unused).

### Methodology
1. Build the resolved-days table via `build_resolved_days` (clean session open
   at exactly `rth_start` and a last RTH bar at or after
   `rth_end - close_tolerance`). Early-close days and the final incomplete day
   are excluded (pending discipline).
2. Over the same RTH bar filter (`hour*60+minute` in `[rth_start_min, rth_end_min)`),
   assign each 1-min bar a bucket index
   `(minute_of_day - rth_start_min) // bucket_min`.
3. Per `(day, bucket)` build the candle: `open` = open of the bucket's earliest
   bar, `close` = close of its latest bar. A bar is **green** when
   `close >= open`, **red** otherwise.
4. For each resolved day, order the bucket bars chronologically (ascending
   bucket index) and compute:
   - `max_green_streak`: length of the longest consecutive green run that day.
   - `max_red_streak`: length of the longest consecutive red run that day.
   Days without any RTH bucket bars are excluded.
5. Report the mean of `max_green_streak` across all contributing days as the
   `green` / `max_streak` value, and similarly for `max_red_streak` → `red`.
   `count` = `total` = number of contributing days.

### Conditions
| Condition key | Description |
|---|---|
| `green` | Green bars (bucket close ≥ open) |
| `red` | Red bars (bucket close < open) |

### Outcomes
| Outcome key | Channel | Description |
|---|---|---|
| `max_streak` | `value` | Avg of the daily-max consecutive-bar run length |

### Parameters
| Parameter | Default | Description |
|---|---|---|
| timeframe | `5min` | Bucket granularity (`1min`, `5min`, `15min`, `30min`, `1h`) |
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- One per run, keyed by the chosen bucket granularity (default `5min`).

### Baseline
Permutation null (fixed seed, deterministic): for each resolved day, the
`bar_colors` array (a bool array preserving the day's actual green/red counts)
is randomly shuffled using `np.random.default_rng(seed)`. The max green and
max red streak lengths are recomputed on the shuffled array. Averaging these
shuffled maxima across all days gives the null: what streak length would occur
if bar colors within each day were independently ordered (no clustering or
momentum). Days are iterated in stable chronological order for reproducibility.

### i18n
- **title.en**: "Average Consecutive Bars"
- **title.fr**: "Bougies consécutives moyennes"
- **definition.en**: "For each trading day, the longest run of consecutive green and of consecutive red intraday bars, averaged across days. Shows the typical length of intraday color streaks."
- **definition.fr**: "Pour chaque jour de bourse, la plus longue série de bougies intraday vertes consécutives et de rouges consécutives, moyennée sur l'ensemble des jours. Montre la longueur typique des séries de couleur intraday."

### Slices
None.


---

## 49. Fibonacci Retracement Levels

**Family**: `fibonacci_levels`
**Module**: `stats/fibonacci_levels/standard.py`
**Result file**: `results/fibonacci_levels.json`
**Status**: Implemented (standard variant)

### What it measures
For each resolved trading day, compute the seven standard Fibonacci retracement
levels (0%, 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100%) of the **prior RTH session's
range**, anchored by the prior candle's direction. Then measure two things about
the **current session**:
- **Which levels the session touches** during RTH (independent rates per level).
- **Which Fibonacci zone the session opens in** (one of 8 mutually exclusive zones).

The prior candle's direction determines the anchor orientation:
- **Green prior session** (low → high move): the retracement starts from the top.
  f=0 → prev_high, f=1 → prev_low.
- **Red prior session** (high → low move): the retracement starts from the bottom.
  f=0 → prev_low, f=1 → prev_high.

The `prev_candle` slice naturally exposes the green/red asymmetry in fib-level

### Methodology
1. Build the RTH daily candle per resolved day (`session_open`, `session_close`)
   and per-day RTH extremes (`day_high`, `day_low`) using
   `build_day_table_with_prior_range`. This also provides the prior RESOLVED
   day's extremes (`prev_high`, `prev_low`) and color (`prev_session_green`).
   The first resolved day has NaN prior values and is excluded from every
   denominator (pending-sample discipline).
2. Countable days require `prev_high` and `prev_low` not NaN **and** a strictly
   positive prior range (`prev_high > prev_low`). `total_samples` counts all
   resolved days; each row's `total` = countable days.
3. Compute the 7 Fibonacci level prices per countable day:
   - Green prior: `level(f) = prev_high - f * rng_` where `rng_ = prev_high - prev_low`.
   - Red prior: `level(f) = prev_low + f * rng_`.
4. **Tier 1 — level touches**: a level `L` is touched when
   `day_low <= L <= day_high`. Each of the 7 levels is tested independently;
   outcomes do NOT partition (a single session can touch multiple levels).
5. **Tier 2 — opening zone**: compute the open's position on the anchored scale:
   - Green prior: `open_frac = (prev_high - session_open) / rng_`.
   - Red prior:   `open_frac = (session_open - prev_low) / rng_`.
   Then classify into one of 8 zones. Zones partition the countable set.

### Conditions & outcomes

#### Tier 1 — `fib_levels` (independent rates, do NOT partition)
| Outcome key | Description |
|---|---|
| `touch_0` | Session touched the 0% level (anchor extreme) |
| `touch_236` | Session touched the 23.6% level |
| `touch_382` | Session touched the 38.2% level |
| `touch_500` | Session touched the 50% level |
| `touch_618` | Session touched the 61.8% level |
| `touch_786` | Session touched the 78.6% level |
| `touch_100` | Session touched the 100% level (far extreme) |

#### Tier 2 — `opening_zone` (partition; each session in exactly one zone)
| Outcome key | open_frac range | Description |
|---|---|---|
| `below_0` | open_frac < 0 | Open beyond the anchor extreme |
| `0_236` | 0 ≤ open_frac < 0.236 | Open in 0%–23.6% zone |
| `236_382` | 0.236 ≤ open_frac < 0.382 | Open in 23.6%–38.2% zone |
| `382_500` | 0.382 ≤ open_frac < 0.5 | Open in 38.2%–50% zone |
| `500_618` | 0.5 ≤ open_frac < 0.618 | Open in 50%–61.8% zone |
| `618_786` | 0.618 ≤ open_frac < 0.786 | Open in 61.8%–78.6% zone |
| `786_100` | 0.786 ≤ open_frac ≤ 1.0 | Open in 78.6%–100% zone |
| `above_100` | open_frac > 1.0 | Open beyond the far extreme |

### Parameters
| Parameter | Default | Description |
|---|---|---|
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `daily` (the RTH daily candle; a single entry under
  `instruments.{INSTRUMENT}.daily`).

### Baseline
Random null (fixed seed, deterministic): the prior-range triple (`prev_high`,
`prev_low`, `prev_session_green`) is **permuted together** across days using the
same permutation index. Each session is thus compared against an UNRELATED day's
Fibonacci levels — the null that temporal adjacency of the prior range and anchor
direction carries no information. Permuting all three columns jointly preserves
the internal consistency of each prior range (`prev_low <= prev_high`) and keeps
each anchor direction aligned with its range. The lone NaN prior triple (the
first resolved day) moves to a random row, preserving the countable count exactly.

### i18n
- **title.en**: "Fibonacci Retracement Levels"
- **title.fr**: "Niveaux de retracement de Fibonacci"
- **definition.en**: "Which Fibonacci retracement levels (0%, 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100%) of the prior RTH session's range does the current session touch? And which Fibonacci zone does the session open in? Levels are anchored by the prior candle's direction."
- **definition.fr**: "Quels niveaux de retracement de Fibonacci (0 %, 23,6 %, 38,2 %, 50 %, 61,8 %, 78,6 %, 100 %) du range de la session RTH précédente la session actuelle touche-t-elle ? Et dans quelle zone de Fibonacci la session ouvre-t-elle ? Les niveaux sont ancrés par la direction de la bougie précédente."

### Slices
- `weekday` — the "by weekday" breakdown (declared via the shared `Weekday`
  slicer).
- `prev_candle` — the "by prior close" breakdown: prior session green/red
  (declared via the shared `PrevCandle` slicer reading `prev_session_green`).
  asymmetry in fib-level touch rates and opening zones.


---

## 50. ICT Opening Retracement

**Family**: `ict_opening_retracement`
**Module**: `stats/ict_opening_retracement/standard.py`
**Result file**: `results/ict_opening_retracement.json`
**Status**: Implemented (standard variant)

### What it measures
There is an **ICT reference candle level**: the price of a reference candle at a
configurable reference time (default `00:00` ET — the midnight candle), using one
of its OHLC prices (default `open`). For each resolved RTH session, classify
whether the session **opened above or below** that reference level, and whether
intraday RTH price **retraced back to touch** the level before the session ended.
Reported as a 2x2 conditional matrix: P(retraced | opened above),
P(not retraced | opened above), P(retraced | opened below),
P(not retraced | opened below). The two outcomes partition each direction, so they
sum to 1.

### Methodology
1. Build the RTH daily candle per resolved session (`session_open`,
   `session_close`) using the same resolution rule as the other daily stats (clean
   09:30 open and a last RTH bar at or after `session_end - close_tolerance`).
2. Compute per-day RTH extremes from the same RTH bar filter
   (`hour*60+minute >= rth_start_min` and `< rth_end_min`):
   - `day_high` = max of `high` over the day's RTH bars.
   - `day_low`  = min of `low` over the day's RTH bars.
   Join these onto the resolved-days index (inner join — only resolved dates).
3. **Reference level**: from the FULL candle set (the reference time may sit
   outside RTH), select bars at exactly `reference_time` (`hour*60+minute ==
   reference_time_min`), index by normalized calendar date, and take the
   `reference_price` column (`open`/`high`/`low`/`close`). The midnight (`00:00`)
   bar on calendar date D normalizes to date D, which is the same normalized date
   as the RTH session that opens at 09:30 on D, so they align by a left-join on the
   resolved index. A resolved day with no reference bar that date has a NaN
   reference level and is **not countable** (excluded from denominators, still
   counted in `total_samples`).
4. Classify the **direction** from `session_open` vs `reference_level` (strict — an
   open exactly at the level has no direction and is excluded):
   - **opened_above**: `session_open > reference_level`.
   - **opened_below**: `session_open < reference_level`.
   A day is **countable** only with a reference level present AND a non-zero
   distance. The distance is `gap_size = abs(session_open - reference_level)`.
5. Classify the **retracement**: intraday RTH price returns to touch the reference
   level (touching exactly counts):
   - opened above: **retraced** when `day_low <= reference_level`.
   - opened below: **retraced** when `day_high >= reference_level`.

`total_samples` counts all resolved sessions; each row's `total` counts only the
**countable** sessions (those with a reference level and a non-zero distance)
carrying that opening direction.

### Conditions & outcomes
| Condition | Outcomes | Partition? | Description |
|---|---|---|---|
| `opened_above` | `retraced`, `not_retraced` | Yes | Retrace rate given the open was above the level; `total` = opened-above days |
| `opened_below` | `retraced`, `not_retraced` | Yes | Retrace rate given the open was below the level; `total` = opened-below days |

### Parameters
| Parameter | Default | Description |
|---|---|---|
| reference_time | "00:00" | `HH:MM` (ET) of the reference candle whose level the open is measured against |
| reference_price | "open" | Which OHLC price of the reference candle is the level (`open`/`high`/`low`/`close`) |
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `daily` (the RTH daily candle; a single entry).

### Baseline
Random null (fixed seed, deterministic): the `retraced` outcome is **permuted**
across countable days, so the overall retrace rate is preserved but its
association with the opening direction is destroyed. Each condition's
`baseline_prob` therefore converges to the pooled retrace rate, and the comparison
reveals whether opening above and opening below retrace at *different* rates than
the market does on average. Uses `np.random.default_rng(seed)`.

### i18n
- **title.en**: "ICT Opening Retracement"
- **title.fr**: "Retracement d'ouverture ICT"
- **definition.en**: "When a session opens above or below the ICT reference candle level (default the midnight open), how often does intraday price retrace back to that level?"
- **definition.fr**: "Lorsqu'une session ouvre au-dessus ou en dessous du niveau de la bougie de référence ICT (par défaut l'ouverture de minuit), à quelle fréquence le prix intraday revient-il à ce niveau ?"

### Slices
- `weekday` — the "by weekday" breakdown (declared via the shared `Weekday` slicer).
- `close` — the "by close" breakdown: session close color, green/red (declared via
  the shared `Close` slicer reading `session_green`).
- `prev_candle` — the "by prev candle" breakdown: prior session color, green/red
  (declared via the shared `PrevCandle` slicer reading `prev_session_green`).
- `size_pts` — the "by size" breakdown in points: opening-distance quartiles
  (declared via `SizeBucket(column="gap_size_pts")`).
- `size_pct` — the "by size" breakdown in percent of the reference level:
  opening-distance quartiles (declared via `SizeBucket(column="gap_size_pct")`).
  `gap_size_pct` is stored in percent units (not a decimal) so the bucket-edge
  labels are legible.

### Future variants (not in MVP)
- `by_fill_time` — split retraced days by whether the retrace occurred before/after
  an intraday cutoff minute (needs the per-day retrace timestamp from the intraday
  path).
- `by_spike` — bucket days by the maximum spike **away** from the level before the
  retrace (needs the intraday path, not just the day extremes).


---

## 51. Pivot Points

**Family**: `pivot_points`
**Module**: `stats/pivot_points/standard.py`
**Result file**: `results/pivot_points.json`
**Status**: Implemented (standard variant — traditional pivots)

### What it measures

For each resolved trading day, compute classic pivot levels from the PRIOR RTH
session's High (H), Low (L), and Close (C), then measure three things about the
CURRENT session:

- **Which levels the session touches** during RTH (independent rates per level).
- **Which zone the session opens in** (one of the mutually exclusive bands
  between consecutive levels, plus the two outer regions).
- **Which zone the session closes in** (same zone structure as the opening).

Two pivot formulas are supported; the standard committed output uses traditional.

### Methodology

1. Build the RTH daily candle per resolved day (`session_open`, `session_close`)
   and per-day RTH extremes (`day_high`, `day_low`) using
   `build_day_table_with_prior_range`. Add `prev_close = session_close.shift(1)`
   (shifted over the resolved-only sorted index, skipping any excluded day).
   The first resolved day has NaN prior values and is excluded from every
   denominator (pending-sample discipline).
2. Countable days require `prev_high`, `prev_low`, and `prev_close` all non-NaN
   AND a strictly positive prior range (`prev_high > prev_low`).
   `total_samples` counts all resolved days; each row's `total` = countable days.
3. **Traditional (floor) pivot formulas** (from prior H, L, C; `rng = H - L`):
   - `PP = (H + L + C) / 3`
   - `R1 = 2*PP - L` ; `S1 = 2*PP - H`
   - `R2 = PP + rng` ; `S2 = PP - rng`
   - `R3 = H + 2*(PP - L)` ; `S3 = L - 2*(H - PP)`
   Levels ordered low → high: S3, S2, S1, PP, R1, R2, R3 (7 levels, 8 zones).
4. **Camarilla pivot formulas** (from prior C and rng; `factor = rng * 1.1`):
   - `Rn = C + factor / divisor` ; `Sn = C - factor / divisor`
   - divisors: R1/S1=12, R2/S2=6, R3/S3=4, R4/S4=2
   Levels ordered low → high: S4, S3, S2, S1, R1, R2, R3, R4 (8 levels, 9 zones).
5. **Tier 1 — `pivot_levels`**: a level L is touched when
   `day_low <= L <= day_high`. Each level is tested independently; outcomes do
   NOT partition (a single session can touch multiple levels).
6. **Tier 2 — `opening_zone`**: classify `session_open` into the zone it falls
   in. Zone boundaries are half-open `[lo, hi)` ascending; the topmost finite
   zone is closed at its upper bound (`<= R3` for traditional, `<= R4` for
   camarilla). Zones partition the countable set.
7. **Tier 3 — `close_zone`**: same zone classification for `session_close`.

### Conditions & outcomes

#### Traditional — `pivot_levels` (independent rates, do NOT partition)

| Outcome key | Description |
|---|---|
| `s3` | Session touched S3 (Support 3) |
| `s2` | Session touched S2 (Support 2) |
| `s1` | Session touched S1 (Support 1) |
| `pp` | Session touched PP (Pivot Point) |
| `r1` | Session touched R1 (Resistance 1) |
| `r2` | Session touched R2 (Resistance 2) |
| `r3` | Session touched R3 (Resistance 3) |

#### Traditional — `opening_zone` and `close_zone` (partition; sum to `total`)

| Outcome key | Price range | Description |
|---|---|---|
| `below_s3` | price < S3 | Below Support 3 |
| `s3_s2` | S3 ≤ price < S2 | S3–S2 band |
| `s2_s1` | S2 ≤ price < S1 | S2–S1 band |
| `s1_pp` | S1 ≤ price < PP | S1–PP band |
| `pp_r1` | PP ≤ price < R1 | PP–R1 band |
| `r1_r2` | R1 ≤ price < R2 | R1–R2 band |
| `r2_r3` | R2 ≤ price ≤ R3 | R2–R3 band (closed at top) |
| `above_r3` | price > R3 | Above Resistance 3 |

Camarilla keys follow the same structure with `cam_` prefix (e.g. `cam_s4`,
`cam_r4`, `cam_s1_r1`, `cam_above_r4`). See label definitions in the module.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| pp_type | `traditional` | Pivot formula: `traditional` or `camarilla` |
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed

- `daily` (the RTH daily candle; a single entry under
  `instruments.{INSTRUMENT}.daily`).

### Baseline

Random null (fixed seed, deterministic): the prior-day quad (`prev_high`,
`prev_low`, `prev_close`, `prev_session_green`) is **permuted together** across
days using a single permutation index. Each session is thus scored against an
UNRELATED day's pivot levels — the null that temporal adjacency of the prior
session carries no information. Permuting all four columns jointly preserves the
internal consistency of each prior day (`prev_low <= prev_high`, `prev_close`
paired with its own H/L). The lone NaN prior quad (the first resolved day)
moves to a random row, preserving the countable count exactly.

### i18n

- **title.en**: "Pivot Points"
- **title.fr**: "Points pivots"
- **definition.en**: "Classic floor (traditional) and camarilla pivot levels computed from the prior RTH session's High, Low, and Close. Three measurement tiers: (1) pivot_levels — fraction of sessions whose RTH range touches each pivot level (independent rates; a session can touch multiple levels); (2) opening_zone — which band between consecutive levels the session open falls in (mutually exclusive zones that partition every countable day); (3) close_zone — which zone the session close falls in (same partition structure). The standard variant uses traditional (floor) pivots."
- **definition.fr**: "Niveaux pivots classiques (traditionnels et camarilla) calculés à partir du Plus Haut, Plus Bas et Clôture de la session RTH précédente. Trois niveaux de mesure : (1) pivot_levels — fraction des sessions dont le range RTH touche chaque niveau pivot (taux indépendants ; une session peut toucher plusieurs niveaux) ; (2) opening_zone — dans quelle zone entre deux niveaux consécutifs l'ouverture de session se situe (zones mutuellement exclusives couvrant tous les jours comptables) ; (3) close_zone — dans quelle zone se situe la clôture de session (même structure). La variante standard utilise les pivots traditionnels (floor)."

### Slices

- `weekday` — the "by weekday" breakdown (declared via the shared `Weekday`
  slicer).
- `prev_candle` — the "by prior close" breakdown: prior session green/red
  (declared via the shared `PrevCandle` slicer reading `prev_session_green`).



---

## 52. Weekly Open Retracement

**Family**: `weekly_open_retracement`
**Module**: `stats/weekly_open_retracement/standard.py`
**Result file**: `results/weekly_open_retracement.json`
**Status**: Implemented (standard variant)

### What it measures
There is a **weekly opening price** `L`: the `open` of the FIRST RTH bar (09:30)
of the ISO week. For each resolved week, classify whether the week first moved
**above or below** `L` (from the first RTH bar's direction), and whether intraday
RTH price later **retraced back to touch** `L` before the week ended. Reported as
a 2x2 conditional matrix — P(retraced | opened above), P(not retraced | opened
above), P(retraced | opened below), P(not retraced | opened below) — plus a
weekday distribution of *when* the first retracement occurred. The two retrace
outcomes partition each direction, so they sum to 1.

### Methodology
1. Filter to RTH bars (`hour*60+minute >= rth_start_min` and `< rth_end_min`).
   Key each bar to the **Monday** of its ISO week
   (`date - to_timedelta(date.dayofweek, 'D')`, normalized tz-aware) and index the
   week table by that Monday date.
2. **Pending discipline**: the LAST ISO week present is always dropped (it may
   still be in progress).
3. **Weekly opening price** `L` = `open` of the chronologically first RTH bar of
   the week.
4. Classify the **direction** from that first bar's `close` vs `L` (strict — a
   doji first bar with `close == L` has no direction and is excluded from every
   denominator, though still counted in `total_samples`):
   - **opened_above**: `close > L` (week first moved up).
   - **opened_below**: `close < L` (week first moved down).
   A week is **countable** only when it has a direction.
5. Classify the **retracement** over the RTH bars AFTER the first bar (touching
   exactly counts):
   - opened above: **retraced** when a later bar's `low <= L`.
   - opened below: **retraced** when a later bar's `high >= L`.
6. **retrace_weekday** = python weekday (0=Mon..4=Fri) of the chronologically
   first retracing bar (NA when not retraced) — drives the weekday distribution.
7. **spike_pts** = maximum favorable excursion from `L` (in points) over the
   window from the first bar through the first retracing bar inclusive (or the
   whole week when never retraced): opened above → `max(high) - L`; opened below →
   `L - min(low)`. **spike_pct** = `100 * spike_pts / L`. Both are defined for all
   countable weeks (NA for doji) and back the size slices.

`total_samples` counts all resolved weeks; each tier-1 row's `total` counts only
the **countable** weeks carrying that direction.

> **Modeling note**: direction is taken from the first RTH bar's close-vs-open
> rather than a multi-bar move, so the classification is deterministic and the
> retracement can occur on the opening Monday itself. The result confirms this:
> on NQ, retraced weeks revisit `L` predominantly on the same Monday (~91%).

### Conditions & outcomes
| Condition | Outcomes | Partition? | Description |
|---|---|---|---|
| `opened_above` | `retraced`, `not_retraced` | Yes | Retrace rate given the week first moved up; `total` = opened-above weeks |
| `opened_below` | `retraced`, `not_retraced` | Yes | Retrace rate given the week first moved down; `total` = opened-below weeks |
| `retraced` | `monday`..`friday` | Yes | Weekday of the first retracement among all retraced weeks; `total` = retraced weeks |

### Timeframes computed
- `weekly` (one entry under `instruments.{INSTRUMENT}.weekly`).

### Baseline
Random null (fixed seed, deterministic; `np.random.default_rng(seed)`). Two
surrogate tables are stitched:
- **Tier 1** (`opened_above` / `opened_below`): the `retraced` outcome is
  **permuted** across countable weeks, preserving the pooled retracement rate
  while destroying its association with direction — each direction's
  `baseline_prob` converges to the pooled rate, so the comparison reveals whether
  up-weeks and down-weeks retrace at *different* rates.
- **Tier 2** (`retraced`): each retraced week's `retrace_weekday` is replaced by a
  uniform random draw over Mon..Fri (`baseline_prob ~ 0.2` each) — the null that
  no weekday is special for the first retracement.

### i18n
- **title.en**: "Weekly Open Retracement"
- **title.fr**: "Retracement de l'ouverture hebdomadaire"
- **definition.en**: "When the first RTH bar of the week closes above or below the weekly opening price, how often does intraday price retrace back to touch that level before the week ends?"
- **definition.fr**: "Lorsque la première bougie RTH de la semaine clôture au-dessus ou en dessous du prix d'ouverture hebdomadaire, à quelle fréquence le prix intraday revient-il toucher ce niveau avant la fin de la semaine ?"

### Slices
- `spike_pts` — the "by spike" breakdown in points: quartiles of the maximum
  excursion away from `L` before the retracement (declared via
  `SizeBucket(column="spike_pts")`). Reveals P(retrace | spike size).
- `spike_pct` — the same in percent of the weekly open (declared via
  `SizeBucket(column="spike_pct")`); `spike_pct` is stored in percent units so the
  bucket-edge labels are legible.

The "by weekday" variant is handled by the tier-2 `retraced` rows in
`compute_rows` (the weekday slicer would split by the week's Monday index, which
is constant, so it is not declared).


---

## 53. Initial Balance Breakout — Performance

**Family**: `initial_balance`
**Module**: `stats/initial_balance/performance.py`
**Result file**: `results/initial_balance_performance.json`
**Status**: Implemented

### What it measures
The magnitude of the **first breakout** from the initial balance (IB). After the
IB forms from the first N minutes of the RTH session, this stat measures how far
price extends past the IB high or low on its *chronologically first* excursion,
before trading back into the balance range. This is the **by-performance** sibling
of section 29 (`initial_balance`, which measures breakout *direction*); it does not
fit the four-outcome partition and instead uses the magnitude (`value` /
`value_baseline`) channel. Reported as the average and maximum first-breakout
extension, in points and as a percent of the session open, under a single `ib`
condition.

### Methodology
1. Build the RTH daily candle per resolved session (`session_open`,
   `session_close`) using the shared resolution rule (clean 09:30 open and a last
   RTH bar within `close_tolerance` of session end).
2. Compute the **initial balance** from the IB window `[rth_start, rth_start +
   ib_period)`: `ib_high` = max `high`, `ib_low` = min `low`, `ib_size = ib_high -
   ib_low`. Inner-join onto resolved days (a day needs a clean IB **and** a
   non-empty breakout window).
3. In the **breakout window** `[rth_start + ib_period, rth_end)`, locate the first
   bar that breaks each side, with **strict** inequalities (touching a level is NOT
   a break), using the chosen criteria:
   - wick (default): up-break `high > ib_high`; down-break `low < ib_low`.
   - close: up-break `close > ib_high`; down-break `close < ib_low`.
4. From each side's first-break bar, walk forward and measure the **maximum
   extension** past that level during the first excursion, stopping when price
   trades back into the IB range (re-entry: wick — opposite extreme crosses back
   inside; close — a bar closes back inside). `up_ext` and `down_ext` are computed
   independently from their own first-break bar (0.0 if that side never breaks).
5. The **first breakout direction** is the side whose first-break bar occurs
   earliest; if both first break on the same bar (wick only), the larger extension
   wins as a deterministic tiebreak. `extension` is the chosen side's extension;
   `extension_pct = extension / session_open` (a decimal).

`total_samples` counts all resolved sessions; each row's `total` is the
**countable** sessions — those that broke at least one side of the IB. Sessions
that never break the IB stay in `total_samples` but are excluded from every
denominator.

### Conditions & outcomes
| Condition | Outcomes | Channel | Description |
|---|---|---|---|
| `ib` | `mean_extension`, `mean_extension_pct`, `max_extension`, `max_extension_pct` | `value` | Average / maximum first-breakout extension, in points and as a decimal of the session open; `total` = countable sessions |

### Parameters
| Parameter | Default | Description |
|---|---|---|
| timeframe | 30min | Initial-balance length: 30min or 1h (emitted as separate timeframe entries) |
| breakout_criteria | wick | Break / break-back detection: `wick` (intraday extreme) or `close` (bar close) |
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `30min` (initial balance 09:30–10:00).
- `1h` (initial balance 09:30–10:30, the canonical initial balance).

### Baseline
Random null (fixed seed, deterministic): the null hypothesis is that the
chronological direction of the first breakout carries no information about the
extension magnitude beyond a randomly chosen side. For each session a fair coin
selects `up_ext` or `down_ext` as the baseline extension; sessions that never broke
keep their `NaN` extension and stay excluded. For a session that broke only one
side, the unbroken side contributes a `0.0` extension, so the coin sometimes picks
it — pulling the baseline mean below the actual (which always picks the breaking
side). Uses `np.random.default_rng(seed)`.

### i18n
- **title.en**: "Initial Balance Breakout — Performance"
- **title.fr**: "Cassure de l'initial balance — Performance"
- **definition.en**: "After the initial balance forms, how far does price extend past the balance high or low on its FIRST breakout, before breaking back into the range? Reported as the average and maximum first-breakout extension, in points and as a percentage of the session open."
- **definition.fr**: "Après la formation de l'initial balance, de combien le prix s'étend-il au-delà du haut ou du bas de la balance lors de sa PREMIÈRE cassure, avant de revenir dans le range ? Exprimé en tant qu'extension moyenne et maximale de la première cassure, en points et en pourcentage de l'ouverture de session."

### Slices
- `weekday` — the "by weekday" breakdown (shared `Weekday` slicer).
- `close` — by session close color, green/red (shared `Close` slicer).
- `prev_candle` — by prior session color, green/red (shared `PrevCandle` slicer).
- `overnight` — by overnight gap direction, open above/below prior close (shared
  `Overnight` slicer).
- `size` — by IB size quartiles, absolute points (`SizeBucket(column="ib_size")`).
- `size_pct` — by IB size as a percent of price, preset bands (<0.2%, 0.2–0.4%,
  0.4–0.6%, 0.6–0.9%, >0.9%) (`SizeBucket(column="ib_size_pct", buckets=[…])`).


---

## 54. Initial Balance Breakout — Retracement

**Family**: `initial_balance`
**Module**: `stats/initial_balance/retracement.py`
**Result file**: `results/initial_balance_retracement.json`
**Status**: Implemented

### What it measures
On **single-break days only** — sessions where price breaks exactly one side of the
initial balance (IB) during the breakout window — how deep price **pulls back into
the IB range** after the break, expressed as a fraction of IB size. Reported as the
share of single-break sessions whose retracement reaches at least each configured
threshold (default 0.25 / 0.50 / 0.75 of IB size). This is the **by-retracement**
sibling of section 29 (`initial_balance`, breakout *direction*); it has a different
outcome set and its own retracement computation. Days that break both sides, or
neither, are excluded from every denominator.

### Methodology
1. Build the RTH daily candle per resolved session (`session_open`,
   `session_close`) using the shared resolution rule (clean 09:30 open and a last
   RTH bar within `close_tolerance` of session end).
2. Compute the **initial balance** from the IB window `[rth_start, rth_start +
   ib_period)`: `ib_high` = max `high`, `ib_low` = min `low`, `ib_size = ib_high -
   ib_low`. Inner-join onto resolved days (a day needs a clean IB **and** a
   non-empty breakout window).
3. Classify the breakout direction over the **breakout window** `[rth_start +
   ib_period, rth_end)` with **strict** inequalities (touching a level is NOT a
   break), using the chosen criteria (wick: `high > ib_high` / `low < ib_low`;
   close: `close > ib_high` / `close < ib_low`). A **single-break day** is one that
   breaks exactly one side (`broke_high` XOR `broke_low`).
4. On a single-break day, measure the **retracement** from the *chronologically
   first* break bar onward — the deepest intraday penetration back toward the
   opposite side, from the broken edge:
   - up-break: `ib_high - min(low)` over the bars at/after the first up-break bar.
   - down-break: `max(high) - ib_low` over the bars at/after the first down-break bar.
   The pull-back is always measured by the intraday extreme (wick), regardless of
   the break-detection criteria. `depth_frac = retracement / ib_size`, clipped at 0;
   the held side guarantees `depth_frac <= 1.0`.

`total_samples` counts all resolved sessions; each row's `total` is the
**single-break** sessions (the denominator). Non-single-break sessions stay in
`total_samples` but are excluded from every denominator.

### Conditions & outcomes
| Condition | Outcomes | Channel | Description |
|---|---|---|---|
| `single_break` | `retrace_25`, `retrace_50`, `retrace_75` | `probability` | Share of single-break sessions whose retracement reaches ≥25% / 50% / 75% of IB size; outcomes are **nested** (a day hitting 0.75 also hits 0.50 and 0.25), not a partition; `total` = single-break sessions |

The outcome keys track the configured thresholds (`retrace_{int(threshold*100)}`).

### Parameters
| Parameter | Default | Description |
|---|---|---|
| timeframe | 30min | Initial-balance length: 30min or 1h (emitted as separate timeframe entries) |
| breakout_criteria | wick | Break detection: `wick` (intraday extreme) or `close` (bar close) |
| thresholds | (0.25, 0.50, 0.75) | Retracement thresholds as fractions of IB size |
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `30min` (initial balance 09:30–10:00).
- `1h` (initial balance 09:30–10:30, the canonical initial balance).

### Baseline
Random null (fixed seed, deterministic): the null hypothesis is that the deepest
pull-back lands **uniformly at random** within the IB range — the breakout carries
no information about how far price retraces. Each single-break session is assigned a
random `depth_frac ~ Uniform(0, 1)`; non-single-break sessions keep their `NaN` and
stay excluded. The same hit-counting path then yields `P(hit ≥ t) → 1 - t`.
Comparing the actual hit rate to this null reveals whether real retracements are
deeper (above baseline) or shallower (below baseline) than a random pull-back. Uses
`np.random.default_rng(seed)`.

### i18n
- **title.en**: "Initial Balance Breakout — Retracement"
- **title.fr**: "Cassure de l'initial balance — Repli"
- **definition.en**: "On single-break days, after price breaks one side of the initial balance, how deep does it pull back into the balance range? Reported as the share of single-break sessions whose retracement reaches at least 25%, 50%, or 75% of the initial-balance size."
- **definition.fr**: "Les jours à cassure unique, après que le prix casse un côté de l'initial balance, jusqu'où revient-il dans le range de la balance ? Exprimé comme la proportion des séances à cassure unique dont le repli atteint au moins 25 %, 50 % ou 75 % de la taille de l'initial balance."

### Slices
- `weekday` — the "by weekday" breakdown (shared `Weekday` slicer).
- `close` — by session close color, green/red (shared `Close` slicer).
- `prev_candle` — by prior session color, green/red (shared `PrevCandle` slicer).
- `overnight` — by overnight gap direction, open above/below prior close (shared
  `Overnight` slicer).
- `size` — by IB size quartiles, absolute points (`SizeBucket(column="ib_size")`).
- `size_pct` — by IB size as a percent of price, preset bands (<0.2%, 0.2–0.4%,
  0.4–0.6%, 0.6–0.9%, >0.9%) (`SizeBucket(column="ib_size_pct", buckets=[…])`).


## 55. Initial Balance Breakout — by Time

**Family**: `initial_balance`
**Module**: `stats/initial_balance/time.py`
**Result file**: `results/initial_balance_time.json`
**Status**: Implemented

### What it measures
*When* the first breakout of the initial balance (IB) happens within the RTH
session — the distribution of the **first-breakout time** across the session,
summarised both as a coarse early/late split around a configurable threshold and as
a fine-grained time histogram. This is the **by-time** sibling of section 29
(`initial_balance`, breakout *direction*); it has a different outcome set and its
own first-breakout-time computation. Sessions that never break either side during
the breakout window are excluded from every denominator.

### Methodology
1. Build the RTH daily candle per resolved session (`session_open`,
   `session_close`) using the shared resolution rule (clean 09:30 open and a last
   RTH bar within `close_tolerance` of session end).
2. Compute the **initial balance** from the IB window `[rth_start, rth_start +
   ib_period)`: `ib_high` = max `high`, `ib_low` = min `low`, `ib_size = ib_high -
   ib_low`. Inner-join onto resolved days (a day needs a clean IB **and** a
   non-empty breakout window).
3. Over the **breakout window** `[rth_start + ib_period, rth_end)`, find the
   *chronologically first* bar that breaks **either** side of the IB, irrespective
   of direction, with **strict** inequalities (touching a level is NOT a break),
   using the chosen criteria (wick: `high > ib_high` / `low < ib_low`; close:
   `close > ib_high` / `close < ib_low`). Its minute-of-day is `first_break_mod`.
4. Classify that time two ways over the **same** countable denominator (sessions
   that broke at least once): an early/late split at the `early_threshold_min`-th
   minute from the open (default 150 → 12:00 ET), and a `bin_minutes`-wide histogram
   spanning `[rth_start, rth_end)`.

`total_samples` counts all resolved sessions; each row's `total` is the **breaking**
sessions (the denominator). Non-breaking sessions stay in `total_samples` but are
excluded from every denominator.

### Conditions & outcomes
| Condition | Outcomes | Channel | Description |
|---|---|---|---|
| `timing` | `early`, `late` | `probability` | Share of breaking sessions whose first breakout is before (`early`) vs at/after (`late`) the threshold; a partition of the denominator |
| `bucket` | `bucket_0`, `bucket_1`, … | `probability` | First-breakout-time histogram: one `bin_minutes`-wide bucket per session window, also a partition of the denominator |

Histogram buckets are aligned to the session open; buckets entirely inside the IB
window are unreachable and always read zero (kept so the outcome set is identical
across timeframes). At the default 30-min bin width the threshold is a bucket
boundary, so `early` / `late` are exactly the sums of the buckets on each side.
Bucket outcome labels are clock ranges (e.g. `10:00–10:30`) injected at `run()`
time from the instrument's session times.

### Parameters
| Parameter | Default | Description |
|---|---|---|
| timeframe | 30min | Initial-balance length: 30min or 1h (emitted as separate timeframe entries) |
| breakout_criteria | wick | Break detection: `wick` (intraday extreme) or `close` (bar close) |
| early_threshold_min | 150 | Minutes from session open separating early from late breakouts |
| bin_minutes | 30 | Histogram bucket width in minutes |
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `30min` (initial balance 09:30–10:00).
- `1h` (initial balance 09:30–10:30, the canonical initial balance).

### Baseline
Random null (fixed seed, deterministic): the null hypothesis is that the first
breakout is equally likely at any minute of the breakout window — its timing carries
no information. Each breaking session is assigned a uniform random minute-of-day in
`[ib_end, rth_end)`; non-breaking sessions keep their `NaN` and stay excluded. The
same counting path then yields the early/late split and histogram a purely uniform
breakout time would produce. Comparing the actual distribution to this null reveals
whether breakouts cluster earlier (above baseline early) or later than chance. Uses
`np.random.default_rng(seed)`.

### i18n
- **title.en**: "Initial Balance Breakout — by Time"
- **title.fr**: "Cassure de l'initial balance — par heure"
- **definition.en**: "After the initial balance forms, WHEN does price first break above its high or below its low? Reported as the share of sessions breaking early vs late around a configurable threshold, plus the full first-breakout-time histogram."
- **definition.fr**: "Après la formation de l'initial balance, QUAND le prix casse-t-il pour la première fois au-dessus de son haut ou en-dessous de son bas ? Exprimé comme la proportion des séances cassant tôt ou tard autour d'un seuil configurable, ainsi que l'histogramme complet de l'heure de première cassure."

### Slices
- `weekday` — the "by weekday" breakdown (shared `Weekday` slicer).
- `close` — by session close color, green/red (shared `Close` slicer).
- `prev_candle` — by prior session color, green/red (shared `PrevCandle` slicer).
- `overnight` — by overnight gap direction, open above/below prior close (shared
  `Overnight` slicer).
- `size` — by IB size quartiles, absolute points (`SizeBucket(column="ib_size")`).
- `size_pct` — by IB size as a percent of price, preset bands (<0.2%, 0.2–0.4%,
  0.4–0.6%, 0.6–0.9%, >0.9%) (`SizeBucket(column="ib_size_pct", buckets=[…])`).


## 56. Initial Balance Breakout — by Rejection

**Family**: `initial_balance`
**Module**: `stats/initial_balance/rejection.py`
**Result file**: `results/initial_balance_rejection.json`
**Status**: Implemented

### What it measures
Whether the ORDER in which the initial balance (IB) edges formed predicts the
direction of the first breakout. It cross-tabulates two sequential events: which IB
edge (high or low) **formed first** during the IB window, against which IB edge
**broke first** afterward (high, low, or neither). This is the **by-rejection**
sibling of section 29 (`initial_balance`, breakout *direction*); it uses a 2×3
contingency structure rather than the four-outcome partition, with its own
formation-order and break-order computations.

### Methodology
1. Build the RTH daily candle per resolved session (`session_open`,
   `session_close`) using the shared resolution rule (clean 09:30 open and a last
   RTH bar within `close_tolerance` of session end).
2. Compute the **initial balance** from the IB window `[rth_start, rth_start +
   ib_period)`: `ib_high` = max `high`, `ib_low` = min `low`, `ib_size = ib_high -
   ib_low`. Inner-join onto resolved days (a day needs a clean IB **and** a
   non-empty breakout window).
3. Determine the **formation order**: via `idxmax` / `idxmin` (first matching bar,
   chronologically sorted), find the earliest bar reaching `ib_high` and the
   earliest reaching `ib_low`. `formed_high_first` is True when the high's bar
   precedes the low's. When a single bar holds **both** extremes the order is
   undetermined (`NaN`) and the day is excluded.
4. Determine the **break order** over the breakout window `[rth_start + ib_period,
   rth_end)`: the minute of the first bar breaking the high and the first breaking
   the low, with **strict** inequalities (touching a level is NOT a break), using
   the chosen criteria (wick: `high > ib_high` / `low < ib_low`; close: `close >
   ib_high` / `close < ib_low`). The earlier minute decides `broke_high` vs
   `broke_low`; if neither side ever breaks the outcome is `neither`; if the **same**
   bar first-breaks both sides at once the order is undetermined and the day is
   excluded.

`total_samples` counts all resolved sessions; each row's `total` is the **countable**
days for its condition (formation order determined, non-empty breakout window, break
order determined). The three outcomes partition each condition's countable days, so
their counts sum to that condition's `total`.

### Conditions & outcomes
| Condition | Outcomes | Partition? | Description |
|---|---|---|---|
| `formed_high` | `broke_high`, `broke_low`, `neither` | Yes | Days where the IB high formed first; which edge broke first afterward; `total` = countable days |
| `formed_low` | `broke_high`, `broke_low`, `neither` | Yes | Days where the IB low formed first; which edge broke first afterward; `total` = countable days |

### Parameters
| Parameter | Default | Description |
|---|---|---|
| timeframe | 30min | Initial-balance length: 30min or 1h (emitted as separate timeframe entries) |
| breakout_criteria | wick | Break detection: `wick` (intraday extreme) or `close` (bar close) |
| close_tolerance_min | 15 | Minutes before session end still considered a full close |

### Timeframes computed
- `30min` (initial balance 09:30–10:00).
- `1h` (initial balance 09:30–10:30, the canonical initial balance).

### Baseline
Random null (fixed seed, deterministic): the null hypothesis is that formation order
carries NO information about which edge breaks first. The `formed_high_first` column
is randomly **permuted** across days, which destroys any association with the break
outcome while preserving the formation-order marginal exactly; the break outcomes are
untouched, so their marginal is preserved too. Under this null both conditions
converge to the overall break-outcome distribution, so the comparison reveals whether
formation order genuinely shifts the breakout direction. Uses
`np.random.default_rng(seed)`.

### i18n
- **title.en**: "Initial Balance Breakout — by Rejection"
- **title.fr**: "Cassure de l'initial balance — par rejet"
- **definition.en**: "Does the order in which the initial balance edges form predict the first breakout? Cross-tabulates which edge (high or low) formed first during the initial balance against which edge broke first afterward (high, low, or neither)."
- **definition.fr**: "L'ordre de formation des bornes de l'initial balance prédit-il la première cassure ? Croise la borne (haut ou bas) formée en premier pendant l'initial balance avec la borne cassée en premier ensuite (haut, bas, ou ni l'un ni l'autre)."

### Slices
- `weekday` — the "by weekday" breakdown (shared `Weekday` slicer).
- `close` — by session close color, green/red (shared `Close` slicer).
- `prev_candle` — by prior session color, green/red (shared `PrevCandle` slicer).
- `overnight` — by overnight gap direction, open above/below prior close (shared
  `Overnight` slicer).
- `size` — by IB size quartiles, absolute points (`SizeBucket(column="ib_size")`).
- `size_pct` — by IB size as a percent of price, preset bands (<0.2%, 0.2–0.4%,
  0.4–0.6%, 0.6–0.9%, >0.9%) (`SizeBucket(column="ib_size_pct", buckets=[…])`).

