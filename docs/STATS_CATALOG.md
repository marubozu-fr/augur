# Augur — Stats Catalog

Reference document for all implemented statistical modules.
Each stat has a methodology section precise enough to reproduce the computation.

---

## 1. Opening Candle Continuation

**Family**: `opening_candle`
**Module**: `stats/opening_candle/continuation.py`
**Status**: To implement

### What it measures
After the first N-minute candle of the RTH session, how often does the session
close in the same direction as that candle?

### Methodology
1. For each trading day, identify the first candle of RTH (opens at 09:30 ET).
2. Classify the candle as **green** (close ≥ open) or **red** (close < open).
3. Classify the daily RTH session as **green** (session close ≥ session open) or **red**.
4. Count the four combinations: green→green, green→red, red→green, red→red.
5. Compute conditional probabilities:
   - P(session green | candle green) = green→green / (green→green + green→red)
   - P(session red | candle red) = red→red / (red→red + red→green)

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| timeframe | 15min | Opening candle duration: 15min, 30min, or 1h |
| session_start | 09:30 | RTH start time (ET) |
| session_end | 16:15 | RTH end time (ET) |

### Baseline
Random baseline: for each day, assign a random candle direction (50/50) and compute
the same conditional probabilities. Expected baseline: ~50% for all conditions.

- `opening_candle_continuation_standard` — aggregate across all days
- `opening_candle_continuation_by_weekday` — sliced by day of week
- `opening_candle_continuation_by_size` — sliced by candle body size buckets
- `opening_candle_continuation_by_close` — compare session close zone vs opening candle
