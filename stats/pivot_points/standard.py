"""Pivot Points stat — standard variant (traditional and camarilla).

Computes classic pivot levels from the PRIOR RTH session's High (H), Low (L),
and Close (C), then measures three things about the CURRENT session:

  1. Level touches (condition ``pivot_levels``): over all countable days, the
     fraction of sessions whose RTH range (``day_low <= level <= day_high``)
     touches each pivot level. Outcomes are INDEPENDENT — a single session can
     touch multiple levels — so they do NOT partition.

  2. Opening zone (condition ``opening_zone``): which band between consecutive
     pivot levels the ``session_open`` falls in. These outcomes DO partition
     the countable set (each session open lands in exactly one zone).

  3. Close zone (condition ``close_zone``): which band the ``session_close``
     falls in. Same partition structure as the opening zone.

Two pivot types are supported via the ``pp_type`` constructor parameter:

Traditional (floor) pivots — default, ``pp_type="traditional"``:
  PP = (H + L + C) / 3
  R1 = 2*PP - L    ;  S1 = 2*PP - H
  R2 = PP + rng    ;  S2 = PP - rng
  R3 = H + 2*(PP - L)  ;  S3 = L - 2*(H - PP)
  where rng = H - L.  Produces 7 levels and 8 zones.

Camarilla pivots — ``pp_type="camarilla"``:
  Rn = C + rng * 1.1 / divisor  ;  Sn = C - rng * 1.1 / divisor
  divisors: R1/S1=12, R2/S2=6, R3/S3=4, R4/S4=2.
  Produces 8 levels (no PP) and 9 zones.

Zone-boundary convention: half-open ``[lo, hi)`` ascending, with the topmost
finite zone closed at the top (mirrors fibonacci's ``786_100`` using ``<=``).

The standard ``run()`` function computes and writes the traditional variant only.
Camarilla is accessible via ``--pp-type camarilla`` for exploratory use but is
not the committed default output.

Countable days require a prior resolved day with strictly positive prior range
(``prev_high > prev_low``) AND a non-NaN prior close (``prev_close``). The
first resolved day always has NaN prior values and is excluded from every
denominator (pending-sample discipline).

Declared slices re-run the whole computation per subset:
  - ``weekday``     — the "by weekday" breakdown.
  - ``prev_candle`` — the "by prior close" breakdown (prior session green/red).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stats.base import (
  BaseStat,
  I18nString,
  Labels,
  StatResultRow,
  write_results,
)
from stats.config import InstrumentConfig, load_config, minute_of_day
from stats.utils.daily_candles import build_day_table_with_prior_range

# ---------------------------------------------------------------------------
# Traditional pivot level and zone keys
# ---------------------------------------------------------------------------
# Level keys ordered low → high: S3, S2, S1, PP, R1, R2, R3.
_TRAD_TOUCH_KEYS: tuple[str, ...] = ("s3", "s2", "s1", "pp", "r1", "r2", "r3")

# Zone keys ordered low → high (8 zones between / around 7 levels).
_TRAD_ZONE_KEYS: tuple[str, ...] = (
  "below_s3",
  "s3_s2",
  "s2_s1",
  "s1_pp",
  "pp_r1",
  "r1_r2",
  "r2_r3",
  "above_r3",
)

# ---------------------------------------------------------------------------
# Camarilla pivot level and zone keys
# ---------------------------------------------------------------------------
# Level keys ordered low → high: S4, S3, S2, S1, R1, R2, R3, R4.
_CAM_TOUCH_KEYS: tuple[str, ...] = (
  "cam_s4", "cam_s3", "cam_s2", "cam_s1",
  "cam_r1", "cam_r2", "cam_r3", "cam_r4",
)

# Zone keys ordered low → high (9 zones between / around 8 levels).
_CAM_ZONE_KEYS: tuple[str, ...] = (
  "cam_below_s4",
  "cam_s4_s3",
  "cam_s3_s2",
  "cam_s2_s1",
  "cam_s1_r1",
  "cam_r1_r2",
  "cam_r2_r3",
  "cam_r3_r4",
  "cam_above_r4",
)

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="Pivot Points",
  fr="Points pivots",
)
_DEFINITION = I18nString(
  en=(
    "Classic floor (traditional) and camarilla pivot levels computed from the "
    "prior RTH session's High, Low, and Close. Three measurement tiers: "
    "(1) pivot_levels — fraction of sessions whose RTH range touches each pivot level "
    "(independent rates; a session can touch multiple levels); "
    "(2) opening_zone — which band between consecutive levels the session open falls in "
    "(mutually exclusive zones that partition every countable day); "
    "(3) close_zone — which zone the session close falls in (same partition structure). "
    "The standard variant uses traditional (floor) pivots."
  ),
  fr=(
    "Niveaux pivots classiques (traditionnels et camarilla) calculés à partir du Plus Haut, "
    "Plus Bas et Clôture de la session RTH précédente. Trois niveaux de mesure : "
    "(1) pivot_levels — fraction des sessions dont le range RTH touche chaque niveau pivot "
    "(taux indépendants ; une session peut toucher plusieurs niveaux) ; "
    "(2) opening_zone — dans quelle zone entre deux niveaux consécutifs l'ouverture de session "
    "se situe (zones mutuellement exclusives couvrant tous les jours comptables) ; "
    "(3) close_zone — dans quelle zone se situe la clôture de session (même structure). "
    "La variante standard utilise les pivots traditionnels (floor)."
  ),
)
_LABELS = Labels(
  conditions={
    "pivot_levels": I18nString(
      en="Pivot level touched",
      fr="Niveau pivot touché",
    ),
    "opening_zone": I18nString(
      en="Opening zone",
      fr="Zone d'ouverture",
    ),
    "close_zone": I18nString(
      en="Close zone",
      fr="Zone de clôture",
    ),
  },
  outcomes={
    # --- Traditional level touch outcomes ---
    "pp": I18nString(en="PP — Pivot Point",  fr="PP — Point pivot"),
    "r1": I18nString(en="R1 — Resistance 1", fr="R1 — Résistance 1"),
    "r2": I18nString(en="R2 — Resistance 2", fr="R2 — Résistance 2"),
    "r3": I18nString(en="R3 — Resistance 3", fr="R3 — Résistance 3"),
    "s1": I18nString(en="S1 — Support 1",    fr="S1 — Support 1"),
    "s2": I18nString(en="S2 — Support 2",    fr="S2 — Support 2"),
    "s3": I18nString(en="S3 — Support 3",    fr="S3 — Support 3"),
    # --- Traditional zone outcomes ---
    "below_s3": I18nString(en="Below S3",   fr="Sous S3"),
    "s3_s2":    I18nString(en="S3–S2",      fr="S3–S2"),
    "s2_s1":    I18nString(en="S2–S1",      fr="S2–S1"),
    "s1_pp":    I18nString(en="S1–PP",      fr="S1–PP"),
    "pp_r1":    I18nString(en="PP–R1",      fr="PP–R1"),
    "r1_r2":    I18nString(en="R1–R2",      fr="R1–R2"),
    "r2_r3":    I18nString(en="R2–R3",      fr="R2–R3"),
    "above_r3": I18nString(en="Above R3",   fr="Au-dessus de R3"),
    # --- Camarilla level touch outcomes ---
    "cam_r1": I18nString(
      en="Cam R1 — Camarilla Resistance 1", fr="Cam R1 — Résistance camarilla 1"
    ),
    "cam_r2": I18nString(
      en="Cam R2 — Camarilla Resistance 2", fr="Cam R2 — Résistance camarilla 2"
    ),
    "cam_r3": I18nString(
      en="Cam R3 — Camarilla Resistance 3", fr="Cam R3 — Résistance camarilla 3"
    ),
    "cam_r4": I18nString(
      en="Cam R4 — Camarilla Resistance 4", fr="Cam R4 — Résistance camarilla 4"
    ),
    "cam_s1": I18nString(
      en="Cam S1 — Camarilla Support 1", fr="Cam S1 — Support camarilla 1"
    ),
    "cam_s2": I18nString(
      en="Cam S2 — Camarilla Support 2", fr="Cam S2 — Support camarilla 2"
    ),
    "cam_s3": I18nString(
      en="Cam S3 — Camarilla Support 3", fr="Cam S3 — Support camarilla 3"
    ),
    "cam_s4": I18nString(
      en="Cam S4 — Camarilla Support 4", fr="Cam S4 — Support camarilla 4"
    ),
    # --- Camarilla zone outcomes ---
    "cam_below_s4": I18nString(en="Below Cam S4",   fr="Sous Cam S4"),
    "cam_s4_s3":    I18nString(en="Cam S4–S3",      fr="Cam S4–S3"),
    "cam_s3_s2":    I18nString(en="Cam S3–S2",      fr="Cam S3–S2"),
    "cam_s2_s1":    I18nString(en="Cam S2–S1",      fr="Cam S2–S1"),
    "cam_s1_r1":    I18nString(en="Cam S1–R1",      fr="Cam S1–R1"),
    "cam_r1_r2":    I18nString(en="Cam R1–R2",      fr="Cam R1–R2"),
    "cam_r2_r3":    I18nString(en="Cam R2–R3",      fr="Cam R2–R3"),
    "cam_r3_r4":    I18nString(en="Cam R3–R4",      fr="Cam R3–R4"),
    "cam_above_r4": I18nString(en="Above Cam R4",   fr="Au-dessus de Cam R4"),
  },
)


class PivotPoints(BaseStat):
  """Pivot-point level touch rates and zone distribution (opening and close).

  Computes, for each resolved trading day, pivot levels from the prior RTH
  session's High, Low, and Close, then measures (a) which levels the current
  session touches, (b) which zone the session opens in, and (c) which zone the
  session closes in.
  """

  stat_name = "pivot_points"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = ("weekday", "prev_candle")

  def __init__(
    self,
    instrument: str,
    config: InstrumentConfig,
    pp_type: str = "traditional",
    close_tolerance_min: int = 15,
  ) -> None:
    if pp_type not in ("traditional", "camarilla"):
      raise ValueError(
        f"Unknown pp_type '{pp_type}'. Use 'traditional' or 'camarilla'."
      )
    self.instrument = instrument
    self.timeframe = "daily"
    self.config = config
    self.pp_type = pp_type
    self.close_tolerance_min = close_tolerance_min

    rth = config.sessions["rth"]
    self.rth_start_min: int = minute_of_day(rth.start)
    self.rth_end_min: int = minute_of_day(rth.end)

  # -------------------------------------------------------------------------
  # Day table construction
  # -------------------------------------------------------------------------
  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-session table with prior-range, prior-color, and prior-close.

    Delegates to ``build_day_table_with_prior_range``, which provides the
    standard resolved-day columns (``session_open``, ``session_close``,
    ``day_high``, ``day_low``, ``prev_high``, ``prev_low``,
    ``prev_session_green``), then adds ``prev_close = session_close.shift(1)``
    (shifted over the resolved-only sorted index, so it skips any excluded or
    early-close day — identical to how ``prev_high`` / ``prev_low`` are
    produced).

    ``prev_close`` is required by both pivot formulas: traditional uses it
    directly in PP = (H + L + C) / 3 and camarilla anchors all levels to C.

    The first resolved day always has NaN prior values (``prev_high``,
    ``prev_low``, ``prev_close``) and is excluded from every denominator
    downstream (pending-sample discipline).
    """
    day = build_day_table_with_prior_range(
      candles_df, self.rth_start_min, self.rth_end_min, self.close_tolerance_min
    )
    day = day.copy()
    if not day.empty:
      day["prev_close"] = day["session_close"].shift(1)
    else:
      day["prev_close"] = pd.Series(dtype=float)
    return day

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute pivot-touch, opening-zone, and close-zone rows from a day table.

    Only days with a prior resolved day AND ``prev_high > prev_low`` AND a
    non-NaN ``prev_close`` are countable.

    Tier 1 — ``pivot_levels``: for each pivot level, the fraction of countable
    days whose RTH range (``day_low <= level <= day_high``) touches it.
    Outcomes are INDEPENDENT — do NOT partition.

    Tier 2 — ``opening_zone``: classify ``session_open`` into one of the zones
    delimited by consecutive pivot levels (half-open ``[lo, hi)`` ascending,
    topmost finite zone closed at top). Zones partition the countable set.

    Tier 3 — ``close_zone``: same zone classification for ``session_close``.

    If ``baseline_rows`` is provided, merges ``baseline_prob`` / ``baseline_n``.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    touch_keys = _TRAD_TOUCH_KEYS if self.pp_type == "traditional" else _CAM_TOUCH_KEYS
    zone_keys = _TRAD_ZONE_KEYS if self.pp_type == "traditional" else _CAM_ZONE_KEYS

    def _make(condition: str, outcome: str, count: int, total: int) -> StatResultRow:
      bl = baseline_map.get((condition, outcome))
      return StatResultRow(
        condition=condition,
        outcome=outcome,
        count=count,
        total=total,
        probability=count / total if total > 0 else 0.0,
        baseline_prob=bl.probability if bl else 0.0,
        baseline_n=bl.total if bl else 0,
      )

    def _empty_rows() -> list[StatResultRow]:
      rows: list[StatResultRow] = []
      for key in touch_keys:
        rows.append(_make("pivot_levels", key, 0, 0))
      for key in zone_keys:
        rows.append(_make("opening_zone", key, 0, 0))
      for key in zone_keys:
        rows.append(_make("close_zone", key, 0, 0))
      return rows

    if day_table.empty:
      return _empty_rows()

    # ----- Countable mask: prior day with rng > 0 AND non-NaN prev_close -----
    has_prior = (
      day_table["prev_high"].notna()
      & day_table["prev_low"].notna()
      & day_table["prev_close"].notna()
    )
    prior_rng_series = (day_table["prev_high"] - day_table["prev_low"]).where(has_prior)
    countable = has_prior & (prior_rng_series > 0)

    ct = day_table[countable]
    countable_n = int(countable.sum())

    if countable_n == 0:
      return _empty_rows()

    # ----- Extract numpy arrays for vectorized computation -----
    H = ct["prev_high"].to_numpy(dtype=float)
    L = ct["prev_low"].to_numpy(dtype=float)
    C = ct["prev_close"].to_numpy(dtype=float)
    rng = H - L  # strictly positive by countable mask

    day_high = ct["day_high"].to_numpy(dtype=float)
    day_low = ct["day_low"].to_numpy(dtype=float)
    session_open = ct["session_open"].to_numpy(dtype=float)
    session_close = ct["session_close"].to_numpy(dtype=float)

    rows: list[StatResultRow] = []

    if self.pp_type == "traditional":
      # ----- Traditional pivot levels -----
      PP = (H + L + C) / 3
      R1 = 2 * PP - L
      S1 = 2 * PP - H
      R2 = PP + rng
      S2 = PP - rng
      R3 = H + 2 * (PP - L)
      # S3 = L - 2*(H - PP): correct floor-pivot formula (H - PP > 0 so S3 < L)
      S3 = L - 2 * (H - PP)

      # Tier 1: level touches — ordered low→high (s3, s2, s1, pp, r1, r2, r3)
      level_arrays = (S3, S2, S1, PP, R1, R2, R3)
      for level_arr, key in zip(level_arrays, _TRAD_TOUCH_KEYS):
        touched = (day_low <= level_arr) & (day_high >= level_arr)
        rows.append(_make("pivot_levels", key, int(touched.sum()), countable_n))

      # Tier 2 + 3: zone classification (opening and close)
      def _trad_zone_masks(
        price: np.ndarray,
      ) -> list[tuple[str, np.ndarray]]:
        """Half-open [lo, hi) zones; topmost finite zone uses <=."""
        return [
          ("below_s3", price < S3),
          ("s3_s2",    (price >= S3) & (price < S2)),
          ("s2_s1",    (price >= S2) & (price < S1)),
          ("s1_pp",    (price >= S1) & (price < PP)),
          ("pp_r1",    (price >= PP) & (price < R1)),
          ("r1_r2",    (price >= R1) & (price < R2)),
          ("r2_r3",    (price >= R2) & (price <= R3)),
          ("above_r3", price > R3),
        ]

      for price, condition in (
        (session_open,  "opening_zone"),
        (session_close, "close_zone"),
      ):
        for key, mask in _trad_zone_masks(price):
          rows.append(_make(condition, key, int(mask.sum()), countable_n))

    else:
      # ----- Camarilla pivot levels -----
      factor = rng * 1.1
      R1 = C + factor / 12
      S1 = C - factor / 12
      R2 = C + factor / 6
      S2 = C - factor / 6
      R3 = C + factor / 4
      S3 = C - factor / 4
      R4 = C + factor / 2
      S4 = C - factor / 2

      # Tier 1: level touches — ordered low→high (cam_s4, …, cam_r4)
      level_arrays_cam = (S4, S3, S2, S1, R1, R2, R3, R4)
      for level_arr, key in zip(level_arrays_cam, _CAM_TOUCH_KEYS):
        touched = (day_low <= level_arr) & (day_high >= level_arr)
        rows.append(_make("pivot_levels", key, int(touched.sum()), countable_n))

      # Tier 2 + 3: zone classification (opening and close)
      def _cam_zone_masks(
        price: np.ndarray,
      ) -> list[tuple[str, np.ndarray]]:
        """Half-open [lo, hi) zones; topmost finite zone uses <=."""
        return [
          ("cam_below_s4", price < S4),
          ("cam_s4_s3",    (price >= S4) & (price < S3)),
          ("cam_s3_s2",    (price >= S3) & (price < S2)),
          ("cam_s2_s1",    (price >= S2) & (price < S1)),
          ("cam_s1_r1",    (price >= S1) & (price < R1)),
          ("cam_r1_r2",    (price >= R1) & (price < R2)),
          ("cam_r2_r3",    (price >= R2) & (price < R3)),
          ("cam_r3_r4",    (price >= R3) & (price <= R4)),
          ("cam_above_r4", price > R4),
        ]

      for price, condition in (
        (session_open,  "opening_zone"),
        (session_close, "close_zone"),
      ):
        for key, mask in _cam_zone_masks(price):
          rows.append(_make(condition, key, int(mask.sum()), countable_n))

    return rows

  # -------------------------------------------------------------------------
  # Baseline
  # -------------------------------------------------------------------------
  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random null baseline. Deterministic for a fixed seed.

    The prior-day quad (``prev_high``, ``prev_low``, ``prev_close``,
    ``prev_session_green``) is permuted together across days using a single
    permutation, so each session is scored against an UNRELATED day's pivot
    levels. This is the null hypothesis: temporal adjacency of the prior session
    carries no information about which levels will be touched or where the
    session will open / close.

    All four columns are shuffled with the SAME permutation index so that each
    shuffled prior day stays internally consistent (``prev_low <= prev_high``,
    ``prev_close`` paired with its own H/L, anchor direction aligned). The lone
    NaN prior quad (the first resolved day) moves to a random position,
    preserving the countable count exactly.
    """
    n = len(day_table)
    if n == 0:
      return self.compute_rows(day_table, baseline_rows=None)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    tmp = day_table.copy()
    tmp["prev_high"] = day_table["prev_high"].to_numpy()[perm]
    tmp["prev_low"] = day_table["prev_low"].to_numpy()[perm]
    tmp["prev_close"] = day_table["prev_close"].to_numpy()[perm]
    tmp["prev_session_green"] = day_table["prev_session_green"].to_numpy()[perm]

    return self.compute_rows(tmp, baseline_rows=None)


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
  pp_type: str = "traditional",
) -> Path:
  """Load data and compute Pivot Points for the daily timeframe.

  The default ``pp_type="traditional"`` is the committed production output.
  Writes the result JSON to ``results/``.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  stat = PivotPoints(instrument=instrument, config=config, pp_type=pp_type)
  result = stat.compute(candles_df)

  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Pivot Points stat")
  parser.add_argument("--instrument", default="NQ", help="Instrument name (default: NQ)")
  parser.add_argument("--data-path", default=None, help="Override parquet file path")
  parser.add_argument(
    "--config-dir", default="config", help="Config directory (default: config)"
  )
  parser.add_argument(
    "--pp-type",
    default="traditional",
    choices=["traditional", "camarilla"],
    help="Pivot type (default: traditional)",
  )
  args = parser.parse_args()

  output_path = run(
    instrument=args.instrument,
    config_dir=args.config_dir,
    data_path=args.data_path,
    pp_type=args.pp_type,
  )

  print(f"Written: {output_path}")

  with open(output_path, encoding="utf-8") as f:
    data = json.load(f)

  instrument_data = data["instruments"][args.instrument]
  for tf, tf_data in instrument_data.items():
    print(f"\n  {tf}: {tf_data['total_samples']} resolved sessions | {tf_data['data_range']}")
    for row in tf_data["results"]:
      print(
        f"    {row['condition']} -> {row['outcome']}: "
        f"{row['probability']:.3f} (N={row['total']}, baseline={row['baseline_prob']:.3f})"
      )
