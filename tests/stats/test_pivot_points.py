"""Tests for stats.pivot_points.standard.

All data is synthetic — no real market files required. Expected values are
hand-calculated before each assertion.

Canonical prior session (used for both traditional and camarilla tests):
  Prior day: open=100, close=120, high=150, low=60
  → prev_high=150, prev_low=60, prev_close=120, rng=90

  Traditional levels:
    PP  = (150 + 60 + 120) / 3 = 110.0
    R1  = 2*110 - 60            = 160.0
    S1  = 2*110 - 150           = 70.0
    R2  = 110 + 90              = 200.0
    S2  = 110 - 90              = 20.0
    R3  = 150 + 2*(110 - 60)   = 250.0
    S3  = 60  - 2*(150 - 110)  = -20.0   ← S3 < L=60 ✓

  Ordered S3→R3: -20, 20, 70, 110, 160, 200, 250
  8 zones: below_s3 | s3_s2 | s2_s1 | s1_pp | pp_r1 | r1_r2 | r2_r3 | above_r3

  Camarilla levels (same prior, C=120, rng=90, factor=99.0):
    R1=128.25  S1=111.75   (factor/12 = 8.25)
    R2=136.5   S2=103.5    (factor/6  = 16.5)
    R3=144.75  S3=95.25    (factor/4  = 24.75)
    R4=169.5   S4=70.5     (factor/2  = 49.5)

  Ordered S4→R4: 70.5, 95.25, 103.5, 111.75, 128.25, 136.5, 144.75, 169.5
  9 zones: cam_below_s4 | cam_s4_s3 | cam_s3_s2 | cam_s2_s1 | cam_s1_r1
           | cam_r1_r2 | cam_r2_r3 | cam_r3_r4 | cam_above_r4

Touch rule: day_low <= level <= day_high.
Level touches are INDEPENDENT (a session can touch multiple levels).
Zone outcomes PARTITION the countable set (exactly one zone per session).
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import StatResultRow, StatRunResult, write_results
from stats.config import InstrumentConfig, Session
from stats.pivot_points.standard import PivotPoints

# ---------------------------------------------------------------------------
# Minimal InstrumentConfig (does not depend on NQ.yaml)
# ---------------------------------------------------------------------------
_NY = "America/New_York"
_RTH_SESSION = Session(start="09:30", end="16:15")
_TEST_CONFIG = InstrumentConfig(
  instrument="NQ",
  working_timezone=_NY,
  sessions={"rth": _RTH_SESSION},
  timeframes=["daily"],
  parquet_path=Path("data/NQ_1min.parquet"),
)

_RTH_START = 570   # 09:30
_RTH_LAST = 974    # 16:14; resolved requires last bar mod >= 960


# ---------------------------------------------------------------------------
# Synthetic day builder
# ---------------------------------------------------------------------------

def _make_day(
  date: str,
  session_open: float,
  session_close: float,
  day_high: float,
  day_low: float,
) -> pd.DataFrame:
  """One trading day of 1-min RTH bars (09:30–16:14).

  Opening bar carries ``session_open``; all bars close at ``session_close``.
  The RTH extremes are placed on a mid-session bar so they are independent of
  open/close. Callers must ensure day_high >= max(session_open, session_close)
  and day_low <= min(session_open, session_close).
  """
  base = pd.Timestamp(date, tz=_NY)
  mid = (_RTH_START + _RTH_LAST) // 2
  records = []
  for mod in range(_RTH_START, _RTH_LAST + 1):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    o = session_open if mod == _RTH_START else session_close
    c = session_close
    hi = day_high if mod == mid else max(o, c)
    lo = day_low if mod == mid else min(o, c)
    records.append({
      "timestamp": ts,
      "open": o,
      "high": hi,
      "low": lo,
      "close": c,
      "volume": 1000,
    })
  return pd.DataFrame(records)


def make_candles(days: list[dict]) -> pd.DataFrame:
  """Concatenate per-day specs into a single sorted 1-min OHLCV DataFrame."""
  frames = [
    _make_day(d["date"], d["open"], d["close"], d["high"], d["low"])
    for d in days
  ]
  df = pd.concat(frames, ignore_index=True)
  return df.sort_values("timestamp").reset_index(drop=True)


def _make_truncated_day(date: str) -> pd.DataFrame:
  """A day whose last RTH bar is 09:50 (mod 590 < 960) — unresolved, excluded."""
  base = pd.Timestamp(date, tz=_NY)
  records = []
  for mod in range(_RTH_START, 591):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    records.append({
      "timestamp": ts, "open": 100.0, "high": 100.25, "low": 99.75,
      "close": 100.0, "volume": 500,
    })
  return pd.DataFrame(records)


def _empty_df() -> pd.DataFrame:
  df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
  df["timestamp"] = pd.array([], dtype="datetime64[ns, America/New_York]")
  return df


# ---------------------------------------------------------------------------
# Stat factory and result helpers
# ---------------------------------------------------------------------------

def _stat(pp_type: str = "traditional") -> PivotPoints:
  return PivotPoints(instrument="NQ", config=_TEST_CONFIG, pp_type=pp_type)


def _row(result: StatRunResult, condition: str, outcome: str) -> StatResultRow:
  for r in result.instruments["NQ"]["daily"].results:
    if r.condition == condition and r.outcome == outcome:
      return r
  raise KeyError((condition, outcome))


# ---------------------------------------------------------------------------
# Canonical prior session
#
# Prior: open=100, close=120, high=150, low=60
# day_high=150 >= max(100,120)=120 ✓   day_low=60 <= min(100,120)=100 ✓
# prev_high=150, prev_low=60, prev_close=120, rng=90
# ---------------------------------------------------------------------------
_PRIOR_DATE = "2024-01-02"
_PRIOR = {
  "date": _PRIOR_DATE,
  "open": 100.0, "close": 120.0, "high": 150.0, "low": 60.0,
}
_CURRENT_DATE = "2024-01-03"

_ALL_TRAD_TOUCH = ("s3", "s2", "s1", "pp", "r1", "r2", "r3")
_ALL_TRAD_ZONES = (
  "below_s3", "s3_s2", "s2_s1", "s1_pp", "pp_r1", "r1_r2", "r2_r3", "above_r3",
)
_ALL_CAM_TOUCH = (
  "cam_s4", "cam_s3", "cam_s2", "cam_s1", "cam_r1", "cam_r2", "cam_r3", "cam_r4",
)
_ALL_CAM_ZONES = (
  "cam_below_s4", "cam_s4_s3", "cam_s3_s2", "cam_s2_s1", "cam_s1_r1",
  "cam_r1_r2", "cam_r2_r3", "cam_r3_r4", "cam_above_r4",
)


def _two_days(
  session_open: float,
  session_close: float,
  day_high: float,
  day_low: float,
  pp_type: str = "traditional",
) -> StatRunResult:
  """Build [_PRIOR, current] candles and run compute."""
  days = [
    _PRIOR,
    {
      "date": _CURRENT_DATE,
      "open": session_open,
      "close": session_close,
      "high": day_high,
      "low": day_low,
    },
  ]
  return _stat(pp_type).compute(make_candles(days))


# ===========================================================================
# 1. Traditional pivot math correctness
#
# PP=110, R1=160, S1=70, R2=200, S2=20, R3=250, S3=-20
# ===========================================================================

def test_traditional_touch_pp_110() -> None:
  """PP=110 touched when day range [109, 111] brackets it.

  PP=110 in [109, 111] ✓. S1=70 < 109, R1=160 > 111 — no other level in range.
  """
  result = _two_days(109.5, 110.5, 111.0, 109.0)
  assert _row(result, "pivot_levels", "pp").count == 1
  for key in ("s3", "s2", "s1", "r1", "r2", "r3"):
    assert _row(result, "pivot_levels", key).count == 0, f"{key} should be 0"


def test_traditional_touch_r1_160() -> None:
  """R1=160 touched when day range [159, 161] brackets it.

  R1=160 in [159, 161] ✓. PP=110 < 159, R2=200 > 161.
  """
  result = _two_days(160.0, 160.5, 161.0, 159.0)
  assert _row(result, "pivot_levels", "r1").count == 1
  for key in ("s3", "s2", "s1", "pp", "r2", "r3"):
    assert _row(result, "pivot_levels", key).count == 0, f"{key} should be 0"


def test_traditional_touch_s1_70() -> None:
  """S1=70 touched when day range [69, 71] brackets it.

  S1=70 in [69, 71] ✓. S2=20 < 69, PP=110 > 71.
  """
  result = _two_days(70.0, 70.5, 71.0, 69.0)
  assert _row(result, "pivot_levels", "s1").count == 1
  for key in ("s3", "s2", "pp", "r1", "r2", "r3"):
    assert _row(result, "pivot_levels", key).count == 0, f"{key} should be 0"


def test_traditional_touch_r2_200() -> None:
  """R2=200 touched when day range [199, 201] brackets it.

  R2=200 in [199, 201] ✓. R1=160 < 199, R3=250 > 201.
  """
  result = _two_days(200.0, 200.5, 201.0, 199.0)
  assert _row(result, "pivot_levels", "r2").count == 1
  for key in ("s3", "s2", "s1", "pp", "r1", "r3"):
    assert _row(result, "pivot_levels", key).count == 0, f"{key} should be 0"


def test_traditional_touch_s2_20() -> None:
  """S2=20 touched when day range [19, 21] brackets it.

  S2=20 in [19, 21] ✓. S3=-20 < 19, S1=70 > 21.
  """
  result = _two_days(20.0, 20.5, 21.0, 19.0)
  assert _row(result, "pivot_levels", "s2").count == 1
  for key in ("s3", "s1", "pp", "r1", "r2", "r3"):
    assert _row(result, "pivot_levels", key).count == 0, f"{key} should be 0"


def test_traditional_touch_r3_250() -> None:
  """R3=250 touched when day range [249, 251] brackets it.

  R3=250 in [249, 251] ✓. R2=200 < 249.
  """
  result = _two_days(250.0, 250.5, 251.0, 249.0)
  assert _row(result, "pivot_levels", "r3").count == 1
  for key in ("s3", "s2", "s1", "pp", "r1", "r2"):
    assert _row(result, "pivot_levels", key).count == 0, f"{key} should be 0"


def test_traditional_touch_s3_negative_20() -> None:
  """S3=-20 touched when day range [-25, -10] contains -20.

  S3=-20 in [-25, -10] ✓. S2=20 > -10 — not touched. Confirms S3 < L=60.
  _make_day(-20, -15, -10, -25):
    first bar: o=-20 c=-15 → hi=-15, lo=-20
    mid bar:   hi=-10, lo=-25
    day_high=max(-15,-10)=-10, day_low=min(-20,-25)=-25 ✓
  """
  result = _two_days(-20.0, -15.0, -10.0, -25.0)
  assert _row(result, "pivot_levels", "s3").count == 1
  assert _row(result, "pivot_levels", "s2").count == 0


def test_traditional_s3_formula_places_level_below_prior_low() -> None:
  """S3 < L verifies L - 2*(H - PP) formula.

  PP=110, H=150, L=60: S3 = 60 - 2*(150-110) = 60-80 = -20 < 60 = L ✓.
  Day range [50, 65] lies between S2=20 and S1=70 — touches NO level.
  """
  # S3=-20 < 50, S2=20 < 50, S1=70 > 65, PP=110 > 65 → nothing touched.
  # day_high=65 >= max(55,58)=58 ✓, day_low=50 <= min(55,58)=55 ✓
  result = _two_days(55.0, 58.0, 65.0, 50.0)
  for key in _ALL_TRAD_TOUCH:
    assert _row(result, "pivot_levels", key).count == 0, f"{key} should be 0"


# ===========================================================================
# 2. Camarilla pivot math correctness
#
# C=120, rng=90, factor=99.0
# S4=70.5, S3=95.25, S2=103.5, S1=111.75
# R1=128.25, R2=136.5, R3=144.75, R4=169.5
# ===========================================================================

def test_camarilla_touch_r1_128_25() -> None:
  """Cam R1=128.25 touched when day range [128, 129] contains 128.25.

  S1=111.75 < 128 — not touched. R2=136.5 > 129 — not touched.
  """
  result = _two_days(128.5, 128.8, 129.0, 128.0, pp_type="camarilla")
  assert _row(result, "pivot_levels", "cam_r1").count == 1
  for key in ("cam_s4", "cam_s3", "cam_s2", "cam_s1", "cam_r2", "cam_r3", "cam_r4"):
    assert _row(result, "pivot_levels", key).count == 0, f"{key} should be 0"


def test_camarilla_touch_s4_70_5() -> None:
  """Cam S4=70.5 touched when day range [70, 71] contains 70.5.

  S3=95.25 > 71 — not touched. All levels above are even higher.
  """
  result = _two_days(70.8, 70.6, 71.0, 70.0, pp_type="camarilla")
  assert _row(result, "pivot_levels", "cam_s4").count == 1
  for key in ("cam_s3", "cam_s2", "cam_s1", "cam_r1", "cam_r2", "cam_r3", "cam_r4"):
    assert _row(result, "pivot_levels", key).count == 0, f"{key} should be 0"


def test_camarilla_touch_r4_169_5() -> None:
  """Cam R4=169.5 touched when day range [169, 170] contains 169.5.

  R3=144.75 < 169 — not touched. Only R4.
  """
  result = _two_days(169.8, 169.6, 170.0, 169.0, pp_type="camarilla")
  assert _row(result, "pivot_levels", "cam_r4").count == 1
  for key in ("cam_s4", "cam_s3", "cam_s2", "cam_s1", "cam_r1", "cam_r2", "cam_r3"):
    assert _row(result, "pivot_levels", key).count == 0, f"{key} should be 0"


def test_camarilla_touch_s3_95_25() -> None:
  """Cam S3=95.25 touched when day range [95, 96] contains 95.25.

  S4=70.5 < 95 — not touched. S2=103.5 > 96 — not touched.
  """
  result = _two_days(95.5, 95.8, 96.0, 95.0, pp_type="camarilla")
  assert _row(result, "pivot_levels", "cam_s3").count == 1
  for key in ("cam_s4", "cam_s2", "cam_s1", "cam_r1", "cam_r2", "cam_r3", "cam_r4"):
    assert _row(result, "pivot_levels", key).count == 0, f"{key} should be 0"


# ===========================================================================
# 3. Opening zone — specific value tests (traditional)
#
# PP=110, R1=160, S1=70, R2=200, S2=20, R3=250, S3=-20
# Zone boundaries (half-open [lo, hi), topmost finite uses <=):
#   below_s3 : open < -20
#   s3_s2    : -20 <= open < 20
#   s2_s1    : 20  <= open < 70
#   s1_pp    : 70  <= open < 110
#   pp_r1    : 110 <= open < 160
#   r1_r2    : 160 <= open < 200
#   r2_r3    : 200 <= open <= 250
#   above_r3 : open > 250
# ===========================================================================

def _assert_only_opening_zone(result: StatRunResult, expected: str) -> None:
  for key in _ALL_TRAD_ZONES:
    row = _row(result, "opening_zone", key)
    if key == expected:
      assert row.count == 1, f"expected opening_zone/{key}=1, got {row.count}"
    else:
      assert row.count == 0, f"expected opening_zone/{key}=0 (expected={expected}), got {row.count}"


def test_traditional_opening_zone_below_s3() -> None:
  """session_open=-25 < S3=-20 → below_s3.

  day_high=-10 >= max(-25,-22)=-22 ✓, day_low=-28 <= min(-25,-22)=-25 ✓.
  """
  result = _two_days(-25.0, -22.0, -10.0, -28.0)
  _assert_only_opening_zone(result, "below_s3")


def test_traditional_opening_zone_s3_s2_boundary_at_s3() -> None:
  """session_open=-20 (exactly at S3) lands in s3_s2, not below_s3.

  Zone s3_s2 uses price >= S3: -20 >= -20 ✓ and -20 < S2=20 ✓.
  Zone below_s3 uses price < S3: -20 < -20 = False ✓.
  """
  # day_high=-10 >= max(-20,-18)=-18 ✓, day_low=-22 <= min(-20,-18)=-20 ✓
  result = _two_days(-20.0, -18.0, -10.0, -22.0)
  assert _row(result, "opening_zone", "s3_s2").count == 1
  assert _row(result, "opening_zone", "below_s3").count == 0


def test_traditional_opening_zone_s1_pp() -> None:
  """session_open=90 in [S1=70, PP=110) → s1_pp.

  70 <= 90 < 110 ✓.
  """
  result = _two_days(90.0, 92.0, 100.0, 88.0)
  _assert_only_opening_zone(result, "s1_pp")


def test_traditional_opening_zone_pp_r1_boundary_at_pp() -> None:
  """session_open exactly at PP=110 lands in pp_r1 (not s1_pp).

  Zone pp_r1 uses price >= PP: 110 >= 110 ✓ and 110 < R1=160 ✓.
  """
  # open=110, close=115. day_high=120 >= 115 ✓, day_low=108 <= 110 ✓
  result = _two_days(110.0, 115.0, 120.0, 108.0)
  assert _row(result, "opening_zone", "pp_r1").count == 1
  assert _row(result, "opening_zone", "s1_pp").count == 0


def test_traditional_opening_zone_r2_r3_boundary_at_r3() -> None:
  """session_open exactly at R3=250 lands in r2_r3 (topmost finite zone, uses <=).

  Zone r2_r3 uses price <= R3: 250 <= 250 ✓. Zone above_r3 uses price > R3: False ✓.
  """
  # open=250, close=252. day_high=255 >= 252 ✓, day_low=248 <= 250 ✓
  result = _two_days(250.0, 252.0, 255.0, 248.0)
  assert _row(result, "opening_zone", "r2_r3").count == 1
  assert _row(result, "opening_zone", "above_r3").count == 0


def test_traditional_opening_zone_above_r3_just_above_r3() -> None:
  """session_open=250.01 (just above R3=250) lands in above_r3, not r2_r3.

  250.01 > 250 → above_r3 ✓.
  """
  # open=250.01, close=252. day_high=255 >= 252 ✓, day_low=248 <= 250.01 ✓
  result = _two_days(250.01, 252.0, 255.0, 248.0)
  assert _row(result, "opening_zone", "above_r3").count == 1
  assert _row(result, "opening_zone", "r2_r3").count == 0


# ===========================================================================
# 4. Close zone — specific value tests (traditional)
# ===========================================================================

def test_traditional_close_zone_s1_pp() -> None:
  """session_close=90 in [S1=70, PP=110) → s1_pp close_zone.

  70 <= 90 < 110 ✓.
  """
  # open=90, close=90. high=95 >= 90 ✓, low=85 <= 90 ✓
  result = _two_days(90.0, 90.0, 95.0, 85.0)
  assert _row(result, "close_zone", "s1_pp").count == 1
  for key in _ALL_TRAD_ZONES:
    if key != "s1_pp":
      assert _row(result, "close_zone", key).count == 0, f"{key} should be 0"


def test_traditional_close_zone_r2_r3() -> None:
  """session_close=225 in [R2=200, R3=250] → r2_r3 close_zone.

  200 <= 225 <= 250 ✓.
  """
  # open=220, close=225. high=230 >= 225 ✓, low=218 <= 220 ✓
  result = _two_days(220.0, 225.0, 230.0, 218.0)
  assert _row(result, "close_zone", "r2_r3").count == 1
  for key in _ALL_TRAD_ZONES:
    if key != "r2_r3":
      assert _row(result, "close_zone", key).count == 0, f"{key} should be 0"


# ===========================================================================
# 5. Zone partitioning — sums to countable_n
# ===========================================================================

def _build_8_session_days() -> list[dict]:
  """8 current sessions designed to cover all 8 traditional zones against _PRIOR.

  Each open targets a different zone relative to _PRIOR's levels:
    below_s3: open=-30 (< -20)
    s3_s2:    open=0   (-20 <= 0 < 20)
    s2_s1:    open=45  (20 <= 45 < 70)
    s1_pp:    open=90  (70 <= 90 < 110)
    pp_r1:    open=130 (110 <= 130 < 160)
    r1_r2:    open=180 (160 <= 180 < 200)
    r2_r3:    open=220 (200 <= 220 <= 250)
    above_r3: open=260 (> 250)
  Note: subsequent sessions use the preceding day as their prior, so zones
  shift. The partition property (sum == countable_n) must hold regardless.
  """
  return [
    _PRIOR,
    {"date": "2024-01-03", "open": -30.0, "close": -28.0, "high": -20.0, "low": -35.0},
    {"date": "2024-01-04", "open": 0.0,   "close": 2.0,   "high": 10.0,  "low": -5.0},
    {"date": "2024-01-05", "open": 45.0,  "close": 47.0,  "high": 55.0,  "low": 40.0},
    {"date": "2024-01-08", "open": 90.0,  "close": 92.0,  "high": 100.0, "low": 85.0},
    {"date": "2024-01-09", "open": 130.0, "close": 132.0, "high": 140.0, "low": 125.0},
    {"date": "2024-01-10", "open": 180.0, "close": 182.0, "high": 190.0, "low": 175.0},
    {"date": "2024-01-11", "open": 220.0, "close": 222.0, "high": 230.0, "low": 215.0},
    {"date": "2024-01-12", "open": 260.0, "close": 262.0, "high": 270.0, "low": 255.0},
  ]


def test_traditional_opening_zone_partition_sums_to_countable_n() -> None:
  """Sum of all opening_zone counts == countable_n for traditional type.

  8 sessions → countable_n=8. Every price lands in exactly one zone.
  """
  result = _stat("traditional").compute(make_candles(_build_8_session_days()))
  tf = result.instruments["NQ"]["daily"]
  countable_n = _row(result, "pivot_levels", "pp").total
  zone_sum = sum(r.count for r in tf.results if r.condition == "opening_zone")
  assert zone_sum == countable_n, f"opening_zone sum={zone_sum} != countable_n={countable_n}"
  assert countable_n == 8


def test_traditional_close_zone_partition_sums_to_countable_n() -> None:
  """Sum of all close_zone counts == countable_n for traditional type."""
  result = _stat("traditional").compute(make_candles(_build_8_session_days()))
  tf = result.instruments["NQ"]["daily"]
  countable_n = _row(result, "pivot_levels", "pp").total
  zone_sum = sum(r.count for r in tf.results if r.condition == "close_zone")
  assert zone_sum == countable_n, f"close_zone sum={zone_sum} != countable_n={countable_n}"


def _build_9_camarilla_days() -> list[dict]:
  """9 current sessions with opens designed to cover all 9 camarilla zones against _PRIOR.

  Cam levels: S4=70.5, S3=95.25, S2=103.5, S1=111.75, R1=128.25,
              R2=136.5, R3=144.75, R4=169.5
    cam_below_s4: open=60   (< 70.5)
    cam_s4_s3:    open=80   (70.5 <= 80 < 95.25)
    cam_s3_s2:    open=100  (95.25 <= 100 < 103.5)
    cam_s2_s1:    open=108  (103.5 <= 108 < 111.75)
    cam_s1_r1:    open=120  (111.75 <= 120 < 128.25)
    cam_r1_r2:    open=132  (128.25 <= 132 < 136.5)
    cam_r2_r3:    open=140  (136.5 <= 140 < 144.75)
    cam_r3_r4:    open=150  (144.75 <= 150 <= 169.5)
    cam_above_r4: open=175  (> 169.5)
  Each subsequent day's prior changes, but partition property always holds.
  """
  return [
    _PRIOR,
    {"date": "2024-01-03", "open": 60.0,  "close": 62.0,  "high": 70.0,  "low": 58.0},
    {"date": "2024-01-04", "open": 80.0,  "close": 82.0,  "high": 90.0,  "low": 78.0},
    {"date": "2024-01-05", "open": 100.0, "close": 102.0, "high": 103.0, "low": 99.0},
    {"date": "2024-01-08", "open": 108.0, "close": 109.0, "high": 111.0, "low": 107.0},
    {"date": "2024-01-09", "open": 120.0, "close": 122.0, "high": 125.0, "low": 118.0},
    {"date": "2024-01-10", "open": 132.0, "close": 133.0, "high": 135.0, "low": 131.0},
    {"date": "2024-01-11", "open": 140.0, "close": 141.0, "high": 143.0, "low": 139.0},
    {"date": "2024-01-12", "open": 150.0, "close": 151.0, "high": 155.0, "low": 148.0},
    {"date": "2024-01-15", "open": 175.0, "close": 176.0, "high": 180.0, "low": 173.0},
  ]


def test_camarilla_opening_zone_partition_sums_to_countable_n() -> None:
  """Sum of all opening_zone counts == countable_n for camarilla type.

  9 sessions → countable_n=9 (all have valid prior from the chain).
  """
  result = _stat("camarilla").compute(make_candles(_build_9_camarilla_days()))
  tf = result.instruments["NQ"]["daily"]
  countable_n = _row(result, "pivot_levels", "cam_r1").total
  zone_sum = sum(r.count for r in tf.results if r.condition == "opening_zone")
  assert zone_sum == countable_n, f"opening_zone sum={zone_sum} != countable_n={countable_n}"
  assert countable_n == 9


def test_camarilla_close_zone_partition_sums_to_countable_n() -> None:
  """Sum of all close_zone counts == countable_n for camarilla type."""
  result = _stat("camarilla").compute(make_candles(_build_9_camarilla_days()))
  tf = result.instruments["NQ"]["daily"]
  countable_n = _row(result, "pivot_levels", "cam_r1").total
  zone_sum = sum(r.count for r in tf.results if r.condition == "close_zone")
  assert zone_sum == countable_n, f"close_zone sum={zone_sum} != countable_n={countable_n}"


# ===========================================================================
# 6. Independent touch tier — pivot_levels do NOT partition
# ===========================================================================

def test_traditional_touch_outcomes_are_independent() -> None:
  """A session touching PP=110 AND R1=160 is counted in both rows.

  Day range [109, 161]: PP=110 ✓, R1=160 ✓. S1=70 < 109 ✗, R2=200 > 161 ✗.
  countable_n=1. Sum of touch counts = 2 > 1 = countable_n — NOT a partition.
  """
  # open=120, close=130. high=161 >= 130 ✓, low=109 <= 120 ✓
  result = _two_days(120.0, 130.0, 161.0, 109.0)
  assert _row(result, "pivot_levels", "pp").count == 1
  assert _row(result, "pivot_levels", "r1").count == 1
  assert _row(result, "pivot_levels", "pp").total == 1
  assert _row(result, "pivot_levels", "r1").total == 1
  tf = result.instruments["NQ"]["daily"]
  touch_sum = sum(r.count for r in tf.results if r.condition == "pivot_levels")
  countable_n = _row(result, "pivot_levels", "pp").total
  assert touch_sum > countable_n, "touch_sum should exceed countable_n (not a partition)"


def test_all_touch_rows_share_same_total() -> None:
  """Every pivot_levels row has the same total (countable_n), regardless of touch count.

  With 2 countable sessions (3 days total), every touch row must report total=2.
  """
  days = [
    _PRIOR,
    # Day 1: touches nothing (open/close/range in s2_s1 zone gap)
    {"date": "2024-01-03", "open": 40.0, "close": 45.0, "high": 55.0, "low": 35.0},
    # Day 2: prior=Day1 (H=55,L=35,C=45,rng=20). Touches all levels of Day1's pivots
    {"date": "2024-01-04", "open": 40.0, "close": 45.0, "high": 200.0, "low": 1.0},
  ]
  result = _stat().compute(make_candles(days))
  for key in _ALL_TRAD_TOUCH:
    assert _row(result, "pivot_levels", key).total == 2, f"{key}.total should be 2"


# ===========================================================================
# 7. Pending-sample discipline
# ===========================================================================

def test_first_day_excluded_from_every_denominator() -> None:
  """First resolved day (NaN prior) is never counted in any row's total.

  3 total days → total_samples=3, countable_n=2.
  Every pivot_levels, opening_zone, and close_zone row must have total=2.
  """
  days = [
    _PRIOR,
    {"date": "2024-01-03", "open": 110.0, "close": 115.0, "high": 120.0, "low": 108.0},
    {"date": "2024-01-04", "open": 130.0, "close": 135.0, "high": 140.0, "low": 128.0},
  ]
  result = _stat().compute(make_candles(days))
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 3
  for key in _ALL_TRAD_TOUCH:
    assert _row(result, "pivot_levels", key).total == 2, f"pivot_levels/{key}.total != 2"
  for key in _ALL_TRAD_ZONES:
    assert _row(result, "opening_zone", key).total == 2, f"opening_zone/{key}.total != 2"
    assert _row(result, "close_zone", key).total == 2, f"close_zone/{key}.total != 2"


def test_zero_range_prior_excluded_from_countable() -> None:
  """A prior day with prev_high == prev_low (rng=0) is not countable.

  Day 0: high=low=100 → zero range.
  Day 1: prior=Day0 → rng=0 → NOT countable.
  Day 2: prior=Day1 (H=130,L=90,rng=40>0) → countable.
  total_samples=3, countable_n=1.
  """
  days = [
    {"date": "2024-01-02", "open": 100.0, "close": 100.0, "high": 100.0, "low": 100.0},
    {"date": "2024-01-03", "open": 110.0, "close": 120.0, "high": 130.0, "low": 90.0},
    {"date": "2024-01-04", "open": 115.0, "close": 118.0, "high": 125.0, "low": 112.0},
  ]
  result = _stat().compute(make_candles(days))
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 3
  # Only Day 2 is countable (Day 1's prior has rng=0)
  assert _row(result, "pivot_levels", "pp").total == 1


def test_nan_prev_close_coincides_with_nan_prior_on_first_day() -> None:
  """The first resolved day has NaN for prev_high, prev_low, AND prev_close.

  All three NaN guards in compute_rows are triggered together for the first row.
  With 2 sessions total, only the second is countable → total=1.
  """
  days = [
    _PRIOR,
    {"date": _CURRENT_DATE, "open": 110.0, "close": 115.0, "high": 120.0, "low": 108.0},
  ]
  result = _stat().compute(make_candles(days))
  assert result.instruments["NQ"]["daily"].total_samples == 2
  assert _row(result, "pivot_levels", "pp").total == 1


def test_truncated_day_excluded_from_total_samples() -> None:
  """An unresolved (truncated) day is never counted in total_samples."""
  stat = _stat()
  base_days = [
    _PRIOR,
    {"date": "2024-01-03", "open": 110.0, "close": 115.0, "high": 120.0, "low": 108.0},
  ]
  base_df = make_candles(base_days)
  base_total = stat.compute(base_df).instruments["NQ"]["daily"].total_samples

  truncated = _make_truncated_day("2024-01-04")
  combined = (
    pd.concat([base_df, truncated], ignore_index=True)
    .sort_values("timestamp")
    .reset_index(drop=True)
  )
  combined_total = stat.compute(combined).instruments["NQ"]["daily"].total_samples
  assert combined_total == base_total == 2


# ===========================================================================
# 8. Baseline — determinism and reproducibility
# ===========================================================================

def _long_seq() -> pd.DataFrame:
  """~200 weekdays with alternating wide/narrow sessions (non-trivial patterns).

  Even-indexed days: green (open=base-10, close=base+20, hi=base+30, lo=base-30).
  Odd-indexed days:  red   (open=base+3, close=base-3, hi=base+8, lo=base-8).
  base starts at 100 and increases by 0.5 per day, always keeping lo > 0.
  """
  dates: list[str] = []
  d = pd.Timestamp("2020-01-01", tz=_NY)
  while len(dates) < 200:
    if d.weekday() < 5:
      dates.append(d.strftime("%Y-%m-%d"))
    d += pd.Timedelta(days=1)

  days = []
  base = 100.0
  for i, date in enumerate(dates):
    if i % 2 == 0:
      hi, lo = base + 30.0, base - 30.0
      open_, close = base - 10.0, base + 20.0   # green
    else:
      hi, lo = base + 8.0, base - 8.0
      open_, close = base + 3.0, base - 3.0     # red
    days.append({"date": date, "open": open_, "close": close, "high": hi, "low": lo})
    base += 0.5

  return make_candles(days)


def test_baseline_deterministic_same_seed() -> None:
  """Same seed produces identical baseline rows across two independent calls."""
  stat = _stat()
  df = _long_seq()
  rows_a = stat.baseline(df, seed=42)
  rows_b = stat.baseline(df, seed=42)
  assert len(rows_a) == len(rows_b)
  for a, b in zip(rows_a, rows_b):
    assert (a.condition, a.outcome) == (b.condition, b.outcome)
    assert a.probability == pytest.approx(b.probability)
    assert a.count == b.count
    assert a.total == b.total


def test_baseline_different_seeds_produce_different_results() -> None:
  """Different seeds produce at least some different probabilities."""
  stat = _stat()
  df = _long_seq()
  rows_42 = stat.baseline(df, seed=42)
  rows_99 = stat.baseline(df, seed=99)
  probs_42 = [r.probability for r in rows_42]
  probs_99 = [r.probability for r in rows_99]
  assert probs_42 != probs_99, "Two different seeds should yield different permutations"


def test_baseline_embedded_in_compute_has_positive_baseline_n() -> None:
  """After compute(), every row with total > 0 carries a positive baseline_n."""
  result = _stat().compute(_long_seq())
  for row in result.instruments["NQ"]["daily"].results:
    if row.total > 0:
      assert row.baseline_n > 0, f"baseline_n=0 for {row.condition}/{row.outcome}"


def test_baseline_permutes_prior_quad_preserving_countable_n() -> None:
  """Shuffling (prev_high, prev_low, prev_close, prev_session_green) together
  preserves the count of NaN rows, so baseline countable_n == real countable_n.
  """
  stat = _stat()
  df = _long_seq()
  day_table = stat.build_day_table(df)
  real_rows = stat.compute_rows(day_table)
  bl_rows = stat.baseline_rows(day_table, seed=7)

  real_total = next(
    r.total for r in real_rows if r.condition == "pivot_levels" and r.outcome == "pp"
  )
  bl_total = next(
    r.total for r in bl_rows if r.condition == "pivot_levels" and r.outcome == "pp"
  )
  assert real_total == bl_total


# ===========================================================================
# 9. Slices
# ===========================================================================

def test_declared_slices_present() -> None:
  """The stat declares weekday and prev_candle slices."""
  result = _stat().compute(_long_seq())
  slices = result.instruments["NQ"]["daily"].slices
  assert set(slices) == {"weekday", "prev_candle"}


def test_weekday_slice_touch_counts_sum_to_overall() -> None:
  """Sum of pp touch counts across weekday groups == the overall pp count."""
  result = _stat().compute(_long_seq())
  wk = result.instruments["NQ"]["daily"].slices["weekday"]
  slice_total = sum(
    r.count
    for grp in wk.groups.values()
    for r in grp.results
    if r.condition == "pivot_levels" and r.outcome == "pp"
  )
  overall = _row(result, "pivot_levels", "pp").count
  assert slice_total == overall


def test_weekday_slice_opening_zone_partition_per_group() -> None:
  """Within each weekday group, opening_zone counts sum to that group's countable_n."""
  result = _stat().compute(_long_seq())
  wk = result.instruments["NQ"]["daily"].slices["weekday"]
  for day_key, grp in wk.groups.items():
    countable_n = next(
      (r.total for r in grp.results if r.condition == "pivot_levels"), 0
    )
    zone_sum = sum(r.count for r in grp.results if r.condition == "opening_zone")
    assert zone_sum == countable_n, (
      f"Weekday group '{day_key}': opening_zone sum={zone_sum} != countable_n={countable_n}"
    )


def test_prev_candle_slice_has_green_and_red_groups() -> None:
  """prev_candle slice exposes 'green' and/or 'red' groups."""
  result = _stat().compute(_long_seq())
  pc = result.instruments["NQ"]["daily"].slices["prev_candle"]
  assert set(pc.groups).issubset({"green", "red"})
  assert len(pc.groups) >= 1


def test_prev_candle_slice_touch_counts_sum_to_overall() -> None:
  """Sum of pp touch counts across prev_candle groups == the overall pp count."""
  result = _stat().compute(_long_seq())
  pc = result.instruments["NQ"]["daily"].slices["prev_candle"]
  slice_total = sum(
    r.count
    for grp in pc.groups.values()
    for r in grp.results
    if r.condition == "pivot_levels" and r.outcome == "pp"
  )
  overall = _row(result, "pivot_levels", "pp").count
  assert slice_total == overall


def test_prev_candle_green_group_from_green_prior() -> None:
  """prev_candle 'green' group captures only days with a green prior.

  1 prior (green) + 1 current day → only 'green' group present; 1 countable day.
  """
  days = [
    # GREEN prior: close=120 > open=100
    _PRIOR,
    {"date": _CURRENT_DATE, "open": 110.0, "close": 115.0, "high": 120.0, "low": 108.0},
  ]
  result = _stat().compute(make_candles(days))
  pc = result.instruments["NQ"]["daily"].slices["prev_candle"]
  assert "green" in pc.groups
  # total_samples is the raw group size (1 row in the slice sub-table)
  assert pc.groups["green"].total_samples == 1
  assert "red" not in pc.groups


def test_weekday_slice_close_zone_partition_per_group() -> None:
  """Within each weekday group, close_zone counts also sum to countable_n."""
  result = _stat().compute(_long_seq())
  wk = result.instruments["NQ"]["daily"].slices["weekday"]
  for day_key, grp in wk.groups.items():
    countable_n = next(
      (r.total for r in grp.results if r.condition == "pivot_levels"), 0
    )
    zone_sum = sum(r.count for r in grp.results if r.condition == "close_zone")
    assert zone_sum == countable_n, (
      f"Weekday group '{day_key}': close_zone sum={zone_sum} != countable_n={countable_n}"
    )


# ===========================================================================
# 10. Edge cases and degenerate input
# ===========================================================================

def test_empty_dataframe_no_crash() -> None:
  """Empty input → zero samples, empty data_range, all rows zeroed, no crash."""
  result = _stat().compute(_empty_df())
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 0
  assert tf.data_range == []
  for row in tf.results:
    assert row.count == 0
    assert row.total == 0
    assert row.probability == pytest.approx(0.0)


def test_single_resolved_day_no_countable_rows() -> None:
  """One resolved session → no prior → countable_n=0, all totals zero, no crash."""
  days = [{"date": "2024-03-01", "open": 100.0, "close": 120.0, "high": 150.0, "low": 60.0}]
  result = _stat().compute(make_candles(days))
  tf = result.instruments["NQ"]["daily"]
  assert tf.total_samples == 1
  for row in tf.results:
    assert row.total == 0
    assert row.count == 0


def test_invalid_pp_type_raises_value_error() -> None:
  """Constructor raises ValueError for an unknown pp_type."""
  with pytest.raises(ValueError, match="Unknown pp_type"):
    PivotPoints(instrument="NQ", config=_TEST_CONFIG, pp_type="fibonacci")


# ===========================================================================
# 11. Result structure — row counts and key completeness
# ===========================================================================

def test_traditional_result_row_count() -> None:
  """Traditional stat produces exactly 7 touch + 8 opening_zone + 8 close_zone = 23 rows."""
  result = _stat("traditional").compute(_empty_df())
  tf = result.instruments["NQ"]["daily"]
  assert len(tf.results) == 23
  assert len([r for r in tf.results if r.condition == "pivot_levels"]) == 7
  assert len([r for r in tf.results if r.condition == "opening_zone"]) == 8
  assert len([r for r in tf.results if r.condition == "close_zone"]) == 8


def test_camarilla_result_row_count() -> None:
  """Camarilla stat produces exactly 8 touch + 9 opening_zone + 9 close_zone = 26 rows."""
  result = _stat("camarilla").compute(_empty_df())
  tf = result.instruments["NQ"]["daily"]
  assert len(tf.results) == 26
  assert len([r for r in tf.results if r.condition == "pivot_levels"]) == 8
  assert len([r for r in tf.results if r.condition == "opening_zone"]) == 9
  assert len([r for r in tf.results if r.condition == "close_zone"]) == 9


def test_traditional_result_key_completeness() -> None:
  """All expected traditional condition/outcome combinations are present."""
  result = _stat("traditional").compute(_empty_df())
  pairs = {(r.condition, r.outcome) for r in result.instruments["NQ"]["daily"].results}
  for key in _ALL_TRAD_TOUCH:
    assert ("pivot_levels", key) in pairs, f"missing pivot_levels/{key}"
  for key in _ALL_TRAD_ZONES:
    assert ("opening_zone", key) in pairs, f"missing opening_zone/{key}"
    assert ("close_zone",   key) in pairs, f"missing close_zone/{key}"


def test_camarilla_result_key_completeness() -> None:
  """All expected camarilla condition/outcome combinations are present."""
  result = _stat("camarilla").compute(_empty_df())
  pairs = {(r.condition, r.outcome) for r in result.instruments["NQ"]["daily"].results}
  for key in _ALL_CAM_TOUCH:
    assert ("pivot_levels", key) in pairs, f"missing pivot_levels/{key}"
  for key in _ALL_CAM_ZONES:
    assert ("opening_zone", key) in pairs, f"missing opening_zone/{key}"
    assert ("close_zone",   key) in pairs, f"missing close_zone/{key}"


def test_data_range_spans_first_to_last_resolved_session() -> None:
  """data_range spans from the first to the last resolved session date."""
  days = [
    _PRIOR,
    {"date": "2024-01-03", "open": 110.0, "close": 115.0, "high": 120.0, "low": 108.0},
    {"date": "2024-01-04", "open": 130.0, "close": 135.0, "high": 140.0, "low": 128.0},
  ]
  result = _stat().compute(make_candles(days))
  assert result.instruments["NQ"]["daily"].data_range == ["2024-01-02", "2024-01-04"]


# ===========================================================================
# 12. i18n metadata
# ===========================================================================

def test_stat_name() -> None:
  """stat_name is 'pivot_points'."""
  assert _stat().compute(_empty_df()).stat_name == "pivot_points"


def test_i18n_title_and_definition() -> None:
  """title and definition both have non-empty en and fr strings."""
  result = _stat().compute(_empty_df())
  assert result.title.en != "" and result.title.fr != ""
  assert result.definition.en != "" and result.definition.fr != ""


def test_i18n_labels_have_en_and_fr_for_all_keys() -> None:
  """Every condition and outcome label has non-empty en and fr strings."""
  result = _stat("traditional").compute(_empty_df())
  for mapping in (result.labels.conditions, result.labels.outcomes):
    for key, i18n in mapping.items():
      assert i18n.en != "", f"{key}.en is empty"
      assert i18n.fr != "", f"{key}.fr is empty"
  assert set(result.labels.conditions) == {"pivot_levels", "opening_zone", "close_zone"}


# ===========================================================================
# 13. write_results round-trip
# ===========================================================================

def test_write_results_round_trip(tmp_path: Path) -> None:
  """write_results produces pivot_points.json that re-validates correctly."""
  days = [
    _PRIOR,
    {"date": _CURRENT_DATE, "open": 110.0, "close": 115.0, "high": 120.0, "low": 108.0},
  ]
  result = _stat().compute(make_candles(days))
  written = write_results(result, results_dir=tmp_path)
  assert written.name == "pivot_points.json"

  raw = json.loads(written.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)
  assert validated.instruments["NQ"]["daily"].total_samples == 2


def test_write_results_french_accent_literal(tmp_path: Path) -> None:
  """French text is stored as literal UTF-8, not unicode-escaped."""
  result = _stat().compute(_empty_df())
  written = write_results(result, results_dir=tmp_path)
  raw = written.read_text(encoding="utf-8")
  # The French definition contains "calculés" (with é).
  assert "é" in raw
  assert "\\u00e9" not in raw
