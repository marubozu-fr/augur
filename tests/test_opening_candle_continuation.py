"""Tests for stats.opening_candle.continuation.

All data is synthetic — no real market files required.
Expected values are hand-calculated before each assertion.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session, load_config, minute_of_day
from stats.opening_candle.continuation import OpeningCandleContinuation

# ---------------------------------------------------------------------------
# Minimal InstrumentConfig used by all tests (does not depend on NQ.yaml)
# ---------------------------------------------------------------------------
_RTH_SESSION = Session(start="09:30", end="16:15")
_TEST_CONFIG = InstrumentConfig(
  instrument="NQ",
  working_timezone="America/New_York",
  sessions={"rth": _RTH_SESSION},
  timeframes=["15min", "30min", "1h"],
  parquet_path=Path("data/NQ_1min.parquet"),
)

# RTH minutes: 09:30 = 570, 16:15 = 975
# Resolved if last bar mod >= 975 - 15 = 960  (i.e., >= 16:00)
# RTH bars: mod in [570, 975)  =>  09:30 … 16:14 inclusive
_NY = "America/New_York"
_RTH_START = 570   # 09:30
_RTH_LAST = 974    # 16:14  (last bar before 16:15)
_RESOLVED_MIN = 960  # 16:00 (975 - 15 close_tolerance)


# ---------------------------------------------------------------------------
# Synthetic data builder
# ---------------------------------------------------------------------------

def _make_day(
  date: str,
  session_open: float,
  session_close: float,
  opening_close: float,
  tf_minutes: int,
) -> pd.DataFrame:
  """Build one trading day of 1-min OHLCV bars covering the opening candle + RTH.

  The opening candle is grid-aligned: it is the timeframe candle that contains
  the RTH open, at `floor(rth_start / tf) * tf`. For 15min/30min that is 09:30,
  but for 1h it is 09:00, so pre-market bars 09:00–09:29 are generated too.

  Args:
    date: "YYYY-MM-DD" string.
    session_open: open price of the 09:30 bar (= session open) and of the
      grid-aligned opening-candle open bar.
    session_close: close price of the 16:14 bar (= session close).
    opening_close: close price of the last bar of the opening candle window.
      - 15min: bar at 09:44  (mod = 584)
      - 30min: bar at 09:59  (mod = 599)
      - 1h:    bar at 09:59  (mod = 599, window 09:00–10:00)
    tf_minutes: 15, 30, or 60.

  Intermediate bars are flat at 100.0 to avoid NaN.
  """
  tz = _NY
  base = pd.Timestamp(date, tz=tz)

  # Opening candle aligned to the timeframe grid (floor of the RTH start).
  candle_open_mod = (_RTH_START // tf_minutes) * tf_minutes
  oc_last_mod = candle_open_mod + tf_minutes - 1  # last minute of opening candle
  assert oc_last_mod != _RTH_LAST, (
    f"tf_minutes={tf_minutes} makes the opening candle end on the session "
    f"close bar ({_RTH_LAST}); opening_close and session_close would collide"
  )

  # Build one minute bar from the opening-candle open through the session close.
  start_mod = min(candle_open_mod, _RTH_START)
  records = []
  for mod in range(start_mod, _RTH_LAST + 1):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)

    # Default flat bar; the four control points below override open/close.
    o = 100.0
    c = 100.0
    if mod == candle_open_mod:
      o = session_open  # opening-candle open
    if mod == _RTH_START:
      o = session_open  # session open
    if mod == oc_last_mod:
      c = opening_close  # opening-candle close (controls opening direction)
    if mod == _RTH_LAST:
      c = session_close  # session close

    records.append({
      "timestamp": ts,
      "open": o,
      "high": max(o, c) + 0.25,
      "low": min(o, c) - 0.25,
      "close": c,
      "volume": 1000,
    })

  return pd.DataFrame(records)


def make_candles(days: list[dict]) -> pd.DataFrame:
  """Build a multi-day 1-min OHLCV DataFrame from a list of day specs.

  Each spec dict must have:
    date (str), session_open (float), session_close (float),
    opening_close (float), tf_minutes (int).

  Returns a DataFrame with tz-aware timestamps in America/New_York, sorted
  by timestamp, with columns [timestamp, open, high, low, close, volume].
  """
  frames = [_make_day(**d) for d in days]
  df = pd.concat(frames, ignore_index=True)
  df = df.sort_values("timestamp").reset_index(drop=True)
  return df


# ---------------------------------------------------------------------------
# Shared day specs used across all three timeframe tests
#
# Design: 10 days, controlled per-timeframe by tf_minutes.
#
# Pattern (same for all tf):
#   GG x3: green opening + green session close
#   GR x2: green opening + red session close
#   RG x4: red opening + green session close
#   RR x1: red opening + red session close
#
# green_open total = 5:
#   P(green_close | green_open) = 3/5 = 0.6
#   P(red_close   | green_open) = 2/5 = 0.4
# red_open total = 5:
#   P(green_close | red_open) = 4/5 = 0.8
#   P(red_close   | red_open) = 1/5 = 0.2
# ---------------------------------------------------------------------------

_DATES = [
  "2024-01-02",
  "2024-01-03",
  "2024-01-04",
  "2024-01-05",
  "2024-01-08",
  "2024-01-09",
  "2024-01-10",
  "2024-01-11",
  "2024-01-12",
  "2024-01-16",
]

# (session_open, opening_close_offset, session_close_offset, label)
# offset is relative to session_open=100.
# green opening:  opening_close > session_open  → use 110
# red opening:    opening_close < session_open  → use 90
# green session:  session_close >= session_open → use 120
# red session:    session_close < session_open  → use 80
_DAY_PATTERNS = [
  # GG x3
  (100.0, 110.0, 120.0),  # 2024-01-02
  (100.0, 110.0, 120.0),  # 2024-01-03
  (100.0, 110.0, 120.0),  # 2024-01-04
  # GR x2
  (100.0, 110.0, 80.0),   # 2024-01-05
  (100.0, 110.0, 80.0),   # 2024-01-08
  # RG x4
  (100.0, 90.0, 120.0),   # 2024-01-09
  (100.0, 90.0, 120.0),   # 2024-01-10
  (100.0, 90.0, 120.0),   # 2024-01-11
  (100.0, 90.0, 120.0),   # 2024-01-12
  # RR x1
  (100.0, 90.0, 80.0),    # 2024-01-16
]


def _build_spec(tf_minutes: int) -> list[dict]:
  return [
    {
      "date": _DATES[i],
      "session_open": so,
      "opening_close": oc,
      "session_close": sc,
      "tf_minutes": tf_minutes,
    }
    for i, (so, oc, sc) in enumerate(_DAY_PATTERNS)
  ]


# ---------------------------------------------------------------------------
# Helper: find a row by condition+outcome
# ---------------------------------------------------------------------------

def _get_row(result: StatRunResult, instrument: str, tf: str, condition: str, outcome: str) -> StatResultRow:
  tf_result = result.instruments[instrument][tf]
  for row in tf_result.results:
    if row.condition == condition and row.outcome == outcome:
      return row
  raise KeyError(f"No row for {condition}/{outcome}")


# ===========================================================================
# 1. All 3 timeframes: exact counts, totals, probabilities
# ===========================================================================

@pytest.mark.parametrize("tf,tf_minutes", [
  ("15min", 15),
  ("30min", 30),
  ("1h", 60),
])
def test_compute_exact_counts(tf: str, tf_minutes: int) -> None:
  """Hand-calculated:
    green_open total=5, green_close=3, P=3/5=0.6
    green_open total=5, red_close=2,   P=2/5=0.4
    red_open   total=5, green_close=4, P=4/5=0.8
    red_open   total=5, red_close=1,   P=1/5=0.2
  """
  stat = OpeningCandleContinuation(instrument="NQ", timeframe=tf, config=_TEST_CONFIG)
  df = make_candles(_build_spec(tf_minutes))
  result = stat.compute(df)

  tf_result = result.instruments["NQ"][tf]
  assert tf_result.total_samples == 10

  # green_open -> green_close: count=3, total=5, prob=0.6
  gg = _get_row(result, "NQ", tf, "green_open", "green_close")
  assert gg.count == 3
  assert gg.total == 5
  assert gg.probability == pytest.approx(3 / 5)

  # green_open -> red_close: count=2, total=5, prob=0.4
  gr = _get_row(result, "NQ", tf, "green_open", "red_close")
  assert gr.count == 2
  assert gr.total == 5
  assert gr.probability == pytest.approx(2 / 5)

  # red_open -> green_close: count=4, total=5, prob=0.8
  rg = _get_row(result, "NQ", tf, "red_open", "green_close")
  assert rg.count == 4
  assert rg.total == 5
  assert rg.probability == pytest.approx(4 / 5)

  # red_open -> red_close: count=1, total=5, prob=0.2
  rr = _get_row(result, "NQ", tf, "red_open", "red_close")
  assert rr.count == 1
  assert rr.total == 5
  assert rr.probability == pytest.approx(1 / 5)

  # Exhaustiveness: counts within each condition must sum to total
  assert gg.count + gr.count == gg.total
  assert rg.count + rr.count == rg.total


@pytest.mark.parametrize("tf,tf_minutes", [
  ("15min", 15),
  ("30min", 30),
  ("1h", 60),
])
def test_data_range_populated(tf: str, tf_minutes: int) -> None:
  """data_range must be [first_date, last_date] as YYYY-MM-DD."""
  stat = OpeningCandleContinuation(instrument="NQ", timeframe=tf, config=_TEST_CONFIG)
  df = make_candles(_build_spec(tf_minutes))
  result = stat.compute(df)

  tf_result = result.instruments["NQ"][tf]
  assert tf_result.data_range == ["2024-01-02", "2024-01-16"]


# ===========================================================================
# 2. Baseline ≈ 50% and deterministic
# ===========================================================================

def _make_200_day_spec(tf_minutes: int) -> list[dict]:
  """200 alternating GG/RR days to provide varied opening directions."""
  dates: list[str] = []
  d = pd.Timestamp("2020-01-02", tz=_NY)
  while len(dates) < 200:
    if d.weekday() < 5:  # Mon–Fri
      dates.append(d.strftime("%Y-%m-%d"))
    d += pd.Timedelta(days=1)

  specs = []
  for i, date in enumerate(dates):
    if i % 2 == 0:
      oc, sc = 110.0, 120.0  # green open, green close
    else:
      oc, sc = 90.0, 80.0   # red open, red close
    specs.append({
      "date": date,
      "session_open": 100.0,
      "opening_close": oc,
      "session_close": sc,
      "tf_minutes": tf_minutes,
    })
  return specs


def test_baseline_approx_fifty_percent() -> None:
  """With seed=42 over 200 days, all baseline_prob values should be near 0.5."""
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  df = make_candles(_make_200_day_spec(15))
  baseline_rows = stat.baseline(df, seed=42)

  for row in baseline_rows:
    assert row.probability == pytest.approx(0.5, abs=0.12), (
      f"{row.condition}/{row.outcome}: baseline_prob={row.probability:.4f} not near 0.5"
    )


def test_baseline_deterministic() -> None:
  """Same seed => identical baseline_prob values on two calls."""
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  df = make_candles(_make_200_day_spec(15))

  rows_a = stat.baseline(df, seed=7)
  rows_b = stat.baseline(df, seed=7)

  assert len(rows_a) == len(rows_b)
  for a, b in zip(rows_a, rows_b):
    assert a.condition == b.condition
    assert a.outcome == b.outcome
    assert a.probability == pytest.approx(b.probability)


def test_baseline_different_seeds_differ() -> None:
  """Different seeds should (in practice) produce different results."""
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  df = make_candles(_make_200_day_spec(15))

  rows_42 = stat.baseline(df, seed=42)
  rows_99 = stat.baseline(df, seed=99)

  probs_42 = [r.probability for r in rows_42]
  probs_99 = [r.probability for r in rows_99]
  assert probs_42 != probs_99


def test_baseline_embedded_in_compute() -> None:
  """compute() must embed baseline_prob into result rows (not 0.0)."""
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  df = make_candles(_make_200_day_spec(15))
  result = stat.compute(df)

  tf_result = result.instruments["NQ"]["15min"]
  for row in tf_result.results:
    # With 200 days and seed=42, baseline_prob should not be exactly 0
    assert row.baseline_prob != 0.0, (
      f"{row.condition}/{row.outcome} has baseline_prob==0.0, baseline not embedded"
    )
    assert row.baseline_n > 0


# ===========================================================================
# 3. Pending exclusion — day with early RTH end is excluded
# ===========================================================================

def _make_truncated_day(date: str, tf_minutes: int) -> pd.DataFrame:
  """Build a day whose last RTH bar is at 09:50 (mod=590), well below the
  resolved threshold of 960 (16:00). This day must be excluded.
  """
  tz = _NY
  base = pd.Timestamp(date, tz=tz)
  records = []
  # Only include bars 09:30..09:50 (mod 570..590)
  for mod in range(_RTH_START, 591):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    records.append({
      "timestamp": ts,
      "open": 100.0,
      "high": 100.25,
      "low": 99.75,
      "close": 100.0,
      "volume": 500,
    })
  return pd.DataFrame(records)


def test_pending_day_excluded() -> None:
  """A day whose last bar is at 09:50 (mod 590 < 960) must not appear
  in build_day_table and must not change total_samples.
  """
  tf = "15min"
  tf_minutes = 15
  stat = OpeningCandleContinuation(instrument="NQ", timeframe=tf, config=_TEST_CONFIG)

  base_specs = _build_spec(tf_minutes)
  df_base = make_candles(base_specs)
  result_base = stat.compute(df_base)
  total_base = result_base.instruments["NQ"][tf].total_samples

  # Append a truncated day (2024-01-17, not already in base_specs)
  truncated = _make_truncated_day("2024-01-17", tf_minutes)
  df_with_pending = pd.concat([df_base, truncated], ignore_index=True)
  df_with_pending = df_with_pending.sort_values("timestamp").reset_index(drop=True)

  result_with = stat.compute(df_with_pending)
  total_with = result_with.instruments["NQ"][tf].total_samples

  assert total_with == total_base, (
    f"Truncated day was counted: total went from {total_base} to {total_with}"
  )


def test_pending_day_absent_from_day_table() -> None:
  """The truncated day's date must not appear in build_day_table's index."""
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  truncated = _make_truncated_day("2024-01-17", 15)
  day_table = stat.build_day_table(truncated)

  pending_date = pd.Timestamp("2024-01-17", tz=_NY).normalize()
  assert pending_date not in day_table.index


# ===========================================================================
# 4. Edge cases
# ===========================================================================

def _empty_df() -> pd.DataFrame:
  """Return a 0-row DataFrame whose columns match the real data contract.

  The timestamp column is explicitly cast to datetime64[ns, America/New_York]
  so that .dt accessors work on an empty frame (pandas requires a concrete
  datetime dtype — an object-typed empty column raises AttributeError on .dt).
  """
  df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
  df["timestamp"] = pd.array([], dtype="datetime64[ns, America/New_York]")
  return df


def test_empty_dataframe_no_crash() -> None:
  """Empty input must not crash and must return 0 total_samples."""
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  result = stat.compute(_empty_df())

  tf_result = result.instruments["NQ"]["15min"]
  assert tf_result.total_samples == 0
  assert tf_result.data_range == []


def test_empty_dataframe_probabilities_zero() -> None:
  """Empty input: all probability values must be 0.0, no ZeroDivisionError."""
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  result = stat.compute(_empty_df())

  tf_result = result.instruments["NQ"]["15min"]
  for row in tf_result.results:
    assert row.probability == pytest.approx(0.0)
    assert row.count == 0
    assert row.total == 0


def test_single_day_green_open_green_close() -> None:
  """Single GG day: total_samples=1, green_open row total=1, count=1, prob=1.0."""
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  df = make_candles([{
    "date": "2024-03-01",
    "session_open": 100.0,
    "opening_close": 110.0,  # green opening
    "session_close": 120.0,  # green session
    "tf_minutes": 15,
  }])
  result = stat.compute(df)

  tf_result = result.instruments["NQ"]["15min"]
  assert tf_result.total_samples == 1

  gg = _get_row(result, "NQ", "15min", "green_open", "green_close")
  assert gg.count == 1
  assert gg.total == 1
  assert gg.probability == pytest.approx(1.0)

  # red_open must have total=0 and prob=0.0
  rg = _get_row(result, "NQ", "15min", "red_open", "green_close")
  assert rg.total == 0
  assert rg.probability == pytest.approx(0.0)


def test_all_green_days_no_zerodivision() -> None:
  """All days have green openings: red_open total=0, prob=0.0 (no ZeroDivisionError)."""
  tf, tf_minutes = "15min", 15
  stat = OpeningCandleContinuation(instrument="NQ", timeframe=tf, config=_TEST_CONFIG)

  specs = [
    {
      "date": _DATES[i],
      "session_open": 100.0,
      "opening_close": 110.0,  # always green opening
      "session_close": 120.0 if i % 2 == 0 else 80.0,  # alternating
      "tf_minutes": tf_minutes,
    }
    for i in range(len(_DATES))
  ]
  df = make_candles(specs)
  result = stat.compute(df)

  tf_result = result.instruments["NQ"][tf]
  assert tf_result.total_samples == len(_DATES)

  # red_open rows must have total=0 and prob=0.0
  for out_key in ("green_close", "red_close"):
    row = _get_row(result, "NQ", tf, "red_open", out_key)
    assert row.total == 0
    assert row.probability == pytest.approx(0.0), (
      f"red_open/{out_key} prob should be 0.0 when total=0, got {row.probability}"
    )


def test_all_red_days_no_zerodivision() -> None:
  """All days have red openings: green_open total=0, prob=0.0 (no ZeroDivisionError)."""
  tf, tf_minutes = "30min", 30
  stat = OpeningCandleContinuation(instrument="NQ", timeframe=tf, config=_TEST_CONFIG)

  specs = [
    {
      "date": _DATES[i],
      "session_open": 100.0,
      "opening_close": 90.0,   # always red opening
      "session_close": 120.0 if i % 2 == 0 else 80.0,
      "tf_minutes": tf_minutes,
    }
    for i in range(len(_DATES))
  ]
  df = make_candles(specs)
  result = stat.compute(df)

  for out_key in ("green_close", "red_close"):
    row = _get_row(result, "NQ", tf, "green_open", out_key)
    assert row.total == 0
    assert row.probability == pytest.approx(0.0)


def test_green_equals_open_counts_as_green() -> None:
  """Green = close >= open. When opening_close == session_open it is green."""
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  # opening_close == session_open (100.0 == 100.0) → opening_green=True
  # session_close == session_open (100.0 == 100.0) → session_green=True
  df = make_candles([{
    "date": "2024-05-01",
    "session_open": 100.0,
    "opening_close": 100.0,  # equal → green
    "session_close": 100.0,  # equal → green
    "tf_minutes": 15,
  }])
  result = stat.compute(df)

  gg = _get_row(result, "NQ", "15min", "green_open", "green_close")
  assert gg.count == 1
  assert gg.total == 1

  ro_total = _get_row(result, "NQ", "15min", "red_open", "green_close").total
  assert ro_total == 0


# ===========================================================================
# 5. i18n — title, definition, labels
# ===========================================================================

def test_i18n_title_and_definition() -> None:
  """StatRunResult must carry non-empty en and fr title + definition."""
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  result = stat.compute(_empty_df())

  assert result.title.en != ""
  assert result.title.fr != ""
  assert result.definition.en != ""
  assert result.definition.fr != ""


def test_i18n_french_accented_chars_in_title() -> None:
  """French title must contain the accented word 'ouverture' or similar accent."""
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  result = stat.compute(_empty_df())

  # Title: "Continuation de la bougie d'ouverture"
  assert "ouverture" in result.title.fr


def test_i18n_french_accented_chars_in_definition() -> None:
  """French definition must contain 'clôture' (circumflex accent)."""
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  result = stat.compute(_empty_df())

  assert "clôture" in result.definition.fr


def test_i18n_labels_conditions_have_en_and_fr() -> None:
  """labels.conditions must define green_open and red_open with en+fr."""
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  result = stat.compute(_empty_df())

  conds = result.labels.conditions
  assert "green_open" in conds
  assert "red_open" in conds
  for key, i18n in conds.items():
    assert i18n.en != "", f"labels.conditions[{key}].en is empty"
    assert i18n.fr != "", f"labels.conditions[{key}].fr is empty"


def test_i18n_labels_outcomes_have_en_and_fr() -> None:
  """labels.outcomes must define green_close and red_close with en+fr."""
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  result = stat.compute(_empty_df())

  outs = result.labels.outcomes
  assert "green_close" in outs
  assert "red_close" in outs
  for key, i18n in outs.items():
    assert i18n.en != "", f"labels.outcomes[{key}].en is empty"
    assert i18n.fr != "", f"labels.outcomes[{key}].fr is empty"


def test_i18n_outcome_labels_contain_french_accent() -> None:
  """French outcome labels must include 'clôture' (accented)."""
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  result = stat.compute(_empty_df())

  fr_labels = " ".join(
    i18n.fr for i18n in result.labels.outcomes.values()
  )
  assert "clôture" in fr_labels, (
    f"Expected 'clôture' in French outcome labels, got: {fr_labels!r}"
  )


# ===========================================================================
# 6. write_results round-trip
# ===========================================================================

def test_write_results_creates_json(tmp_path: Path) -> None:
  """write_results() creates a JSON file named after stat_name."""
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  df = make_candles(_build_spec(15))
  result = stat.compute(df)

  written_path = write_results(result, results_dir=tmp_path)
  assert written_path.exists()
  assert written_path.suffix == ".json"
  assert written_path.name == "opening_candle_continuation.json"


def test_write_results_json_valid_structure(tmp_path: Path) -> None:
  """Written JSON must re-load and validate back into StatRunResult."""
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  df = make_candles(_build_spec(15))
  result = stat.compute(df)
  written_path = write_results(result, results_dir=tmp_path)

  raw = json.loads(written_path.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)

  assert validated.stat_name == "opening_candle_continuation"
  assert validated.instruments["NQ"]["15min"].total_samples == 10


def test_write_results_french_accents_literal(tmp_path: Path) -> None:
  """JSON must use literal UTF-8 accented chars, not \\u escapes (ensure_ascii=False)."""
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  result = stat.compute(_empty_df())
  written_path = write_results(result, results_dir=tmp_path)

  raw_bytes = written_path.read_text(encoding="utf-8")
  # Must contain literal ô, not ô
  assert "clôture" in raw_bytes, (
    "Expected literal 'clôture' in JSON, got \\u escapes instead."
  )
  # If ensure_ascii were True, the circumflex ô would appear as ô
  assert "\\u00f4" not in raw_bytes


def test_write_results_no_leftover_tmp_files(tmp_path: Path) -> None:
  """Atomic write must leave no .tmp_* files in the results dir."""
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  result = stat.compute(_empty_df())
  write_results(result, results_dir=tmp_path)

  tmp_files = list(tmp_path.glob(".tmp_*"))
  assert tmp_files == [], f"Leftover temp files: {tmp_files}"


def test_write_results_roundtrip_probabilities(tmp_path: Path) -> None:
  """Probabilities survive the JSON round-trip without precision loss."""
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  df = make_candles(_build_spec(15))
  result = stat.compute(df)
  written_path = write_results(result, results_dir=tmp_path)

  raw = json.loads(written_path.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)

  orig_rows = result.instruments["NQ"]["15min"].results
  reloaded_rows = validated.instruments["NQ"]["15min"].results

  for orig, reloaded in zip(orig_rows, reloaded_rows):
    assert orig.probability == pytest.approx(reloaded.probability)
    assert orig.count == reloaded.count
    assert orig.total == reloaded.total


# ===========================================================================
# 7. Config loader
# ===========================================================================

def test_minute_of_day_930() -> None:
  """09:30 must convert to 570 minutes since midnight."""
  assert minute_of_day("09:30") == 570


def test_minute_of_day_1615() -> None:
  """16:15 must convert to 975 minutes since midnight."""
  assert minute_of_day("16:15") == 975


def test_minute_of_day_midnight() -> None:
  """00:00 must convert to 0."""
  assert minute_of_day("00:00") == 0


def test_minute_of_day_1200() -> None:
  """12:00 must convert to 720."""
  assert minute_of_day("12:00") == 720


_AUGUR_ROOT = Path(__file__).parent.parent


def test_load_config_nq_rth_session() -> None:
  """load_config('NQ') must return rth session with start=09:30 and end=16:15."""
  config = load_config("NQ", config_dir=_AUGUR_ROOT / "config")
  rth = config.sessions["rth"]
  assert rth.start == "09:30"
  assert rth.end == "16:15"


def test_load_config_nq_instrument_name() -> None:
  """load_config('NQ') must return instrument='NQ' and working_timezone='America/New_York'."""
  config = load_config("NQ", config_dir=_AUGUR_ROOT / "config")
  assert config.instrument == "NQ"
  assert config.working_timezone == "America/New_York"


def test_load_config_nq_parquet_path_default() -> None:
  """When NQ.yaml has no data.parquet_1min, parquet_path defaults to data/NQ_1min.parquet."""
  config = load_config("NQ", config_dir=_AUGUR_ROOT / "config")
  assert config.parquet_path == Path("data/NQ_1min.parquet")


def test_load_config_nq_has_timeframes() -> None:
  """NQ config must include at least 15min, 30min, 1h timeframes."""
  config = load_config("NQ", config_dir=_AUGUR_ROOT / "config")
  for tf in ("15min", "30min", "1h"):
    assert tf in config.timeframes, f"Missing timeframe {tf} in NQ config"


def test_load_config_sessions_are_frozen() -> None:
  """Session and InstrumentConfig must be frozen dataclasses (immutable)."""
  config = load_config("NQ", config_dir=_AUGUR_ROOT / "config")
  with pytest.raises((AttributeError, TypeError)):
    config.instrument = "ES"  # type: ignore[misc]


# ===========================================================================
# 8. stat_name and result structure
# ===========================================================================

def test_stat_name() -> None:
  """StatRunResult.stat_name must be 'opening_candle_continuation'."""
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  result = stat.compute(_empty_df())
  assert result.stat_name == "opening_candle_continuation"


def test_result_has_four_rows() -> None:
  """TimeframeResult must always have exactly 4 rows (2 conditions x 2 outcomes)."""
  for tf in ("15min", "30min", "1h"):
    stat = OpeningCandleContinuation(instrument="NQ", timeframe=tf, config=_TEST_CONFIG)
    result = stat.compute(_empty_df())
    tf_result = result.instruments["NQ"][tf]
    assert len(tf_result.results) == 4, f"Expected 4 rows for {tf}, got {len(tf_result.results)}"


def test_result_condition_outcome_keys() -> None:
  """All four condition/outcome key combos must be present."""
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  result = stat.compute(_empty_df())
  tf_result = result.instruments["NQ"]["15min"]

  expected = {
    ("green_open", "green_close"),
    ("green_open", "red_close"),
    ("red_open", "green_close"),
    ("red_open", "red_close"),
  }
  actual = {(r.condition, r.outcome) for r in tf_result.results}
  assert actual == expected


# ===========================================================================
# 9. build_day_table internals
# ===========================================================================

def test_build_day_table_columns() -> None:
  """build_day_table must return a DataFrame with the required columns."""
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  df = make_candles(_build_spec(15))
  day_table = stat.build_day_table(df)

  required = {"session_open", "session_close", "last_minute", "opening_open", "opening_close",
              "opening_green", "session_green"}
  assert required.issubset(set(day_table.columns))


def test_build_day_table_index_is_dates() -> None:
  """build_day_table index must contain normalized pd.Timestamp dates."""
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  df = make_candles(_build_spec(15))
  day_table = stat.build_day_table(df)

  assert len(day_table) == 10
  # Each index value should be a normalized Timestamp (time component = 0)
  for idx in day_table.index:
    assert isinstance(idx, pd.Timestamp)
    assert idx.hour == 0
    assert idx.minute == 0


def test_build_day_table_opening_green_flags() -> None:
  """opening_green must match the hand-designed patterns (first 5 green, last 5 red)."""
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  df = make_candles(_build_spec(15))
  day_table = stat.build_day_table(df)

  # Days 0-4 (index 0-4) are green opening (oc=110 > so=100)
  # Days 5-9 (index 5-9) are red opening (oc=90 < so=100)
  sorted_table = day_table.sort_index()
  for i, (_, row) in enumerate(sorted_table.iterrows()):
    if i < 5:
      assert row["opening_green"], f"Day {i} should be green opening"
    else:
      assert not row["opening_green"], f"Day {i} should be red opening"


def test_build_day_table_session_green_flags() -> None:
  """session_green must match the hand-designed close pattern."""
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  df = make_candles(_build_spec(15))
  day_table = stat.build_day_table(df)
  sorted_table = day_table.sort_index()

  # Pattern: GG GG GG GR GR RG RG RG RG RR
  # session_green:  T  T  T  F  F  T  T  T  T  F
  expected_session_green = [True, True, True, False, False, True, True, True, True, False]
  for i, (_, row) in enumerate(sorted_table.iterrows()):
    assert bool(row["session_green"]) == expected_session_green[i], (
      f"Day {i} session_green mismatch: expected {expected_session_green[i]}"
    )


# ===========================================================================
# 9b. Slice-metric columns (opening_body, close_location, opening wick extremes)
# ===========================================================================

def test_build_day_table_has_slice_columns() -> None:
  """build_day_table must expose the columns the size/close slices read."""
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  day_table = stat.build_day_table(make_candles(_build_spec(15)))

  required = {"opening_high", "opening_low", "opening_body", "close_location"}
  assert required.issubset(set(day_table.columns))


def test_opening_body_is_absolute_candle_body() -> None:
  """opening_body = |opening_close - opening_open|, regardless of candle color."""
  days = [
    {"date": "2024-01-02", "session_open": 100.0, "opening_close": 110.0,
     "session_close": 120.0, "tf_minutes": 15},  # green, body 10
    {"date": "2024-01-03", "session_open": 100.0, "opening_close": 92.0,
     "session_close": 80.0, "tf_minutes": 15},   # red, body 8
  ]
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  day_table = stat.build_day_table(make_candles(days)).sort_index()

  bodies = list(day_table["opening_body"])
  assert bodies[0] == pytest.approx(10.0)
  assert bodies[1] == pytest.approx(8.0)


def test_close_location_above_inside_below() -> None:
  """close_location classifies the session close vs the opening candle range.

  The range uses wick extremes. With the synthetic builder a green opening
  (open=100, close=110) spans high=110.25, low=99.75; a session close above
  110.25 is 'above', within [99.75, 110.25] is 'inside', below 99.75 is 'below'.
  """
  days = [
    {"date": "2024-01-02", "session_open": 100.0, "opening_close": 110.0,
     "session_close": 120.0, "tf_minutes": 15},  # above the opening high
    {"date": "2024-01-03", "session_open": 100.0, "opening_close": 110.0,
     "session_close": 105.0, "tf_minutes": 15},  # inside the opening range
    {"date": "2024-01-04", "session_open": 100.0, "opening_close": 110.0,
     "session_close": 80.0, "tf_minutes": 15},   # below the opening low
  ]
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  day_table = stat.build_day_table(make_candles(days)).sort_index()

  assert list(day_table["close_location"]) == ["above", "inside", "below"]


# ===========================================================================
# 10. Instrument and timeframe attributes
# ===========================================================================

def test_stat_attributes() -> None:
  """OpeningCandleContinuation must expose instrument, timeframe, tf_minutes."""
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="1h", config=_TEST_CONFIG)
  assert stat.instrument == "NQ"
  assert stat.timeframe == "1h"
  assert stat.tf_minutes == 60
  assert stat.rth_start_min == 570
  assert stat.rth_end_min == 975
  # 1h candle that contains 09:30 is the 09:00–10:00 bar: floor(570/60)*60 = 540.
  assert stat.candle_open_min == 540


@pytest.mark.parametrize("tf,expected_open_min", [
  ("15min", 570),  # 09:30 — grid-aligned, equals the RTH open
  ("30min", 570),  # 09:30 — grid-aligned, equals the RTH open
  ("1h", 540),     # 09:00 — the 1h candle that contains the 09:30 open
])
def test_candle_open_min_grid_aligned(tf: str, expected_open_min: int) -> None:
  """The opening candle is aligned to floor(rth_start / tf) * tf, so 1h captures
  the 09:00–10:00 candle while 15min/30min still start at 09:30."""
  stat = OpeningCandleContinuation(instrument="NQ", timeframe=tf, config=_TEST_CONFIG)
  assert stat.candle_open_min == expected_open_min


def test_stat_invalid_timeframe_raises() -> None:
  """Unsupported timeframe string must raise ValueError."""
  with pytest.raises(ValueError, match="Unsupported timeframe"):
    OpeningCandleContinuation(instrument="NQ", timeframe="5min", config=_TEST_CONFIG)
