"""Overnight Range Breakout stat — standard variant.

Measures how each RTH session's market-hours high/low relate to the PRIOR
overnight session's high/low: does price break above the overnight high only,
below the overnight low only, both sides, or neither during the regular session?

Definitions:
  - **Overnight range**: the high-low range of the overnight session that
    precedes a given RTH session, ``[overnight_start (prev day), overnight_end)``
    (18:00 ET → 09:30 ET for NQ). ``on_high`` is the max high and ``on_low`` the
    min low over that window; ``on_size`` is ``on_high - on_low``. The overnight
    session crosses midnight, so each bar is attributed to the RTH session date it
    PRECEDES: evening bars (at or after ``overnight_start``) belong to the next
    calendar day's overnight; early bars (before ``overnight_end``) belong to the
    same calendar day's overnight.
  - **Breakout window**: the whole RTH session, ``[rth_start, rth_end)``.
  - **Break**: how the RTH session's price relates to the overnight range. With
    ``breakout_criteria = "wick"`` (default) a break uses the session's intraday
    extremes (a wick beyond the level counts); with ``"close"`` a break requires a
    bar to CLOSE beyond the level (the session's extreme close).

Four MUTUALLY EXCLUSIVE outcomes partition every countable day exhaustively
(strict inequality defines a break — touching the level exactly is NOT a break,
mirroring the sibling ``opening_range_breakout`` / ``initial_balance`` convention):
  - ``broke_high`` — broke above ``on_high`` only (high break, low held)
  - ``broke_low``  — broke below ``on_low`` only (low break, high held)
  - ``broke_both`` — broke both sides during the session
  - ``neither``    — stayed entirely within the overnight range all session

Reported under a single ``overnight_range`` condition; the four outcomes sum to
the countable day count, so their probabilities sum to 1. ``total_samples``
counts ALL resolved days; each row's ``total`` is the countable days (a resolved
RTH day that also has a prior overnight range).

Declared slices re-run the whole computation per subset:
  - ``weekday``     — the "by weekday" breakdown.
  - ``close``       — split by session close color (green/red).
  - ``prev_candle`` — split by the prior session's color (green/red).
  - ``size``        — quartile buckets of the overnight-range size.
  - ``levels``      — extension past the broken level in multiples of ``on_size``.
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
  Levels,
  SampleRow,
  SizeBucket,
  StatResultRow,
  write_results,
)
from stats.config import InstrumentConfig, load_config, minute_of_day
from stats.utils.daily_candles import build_resolved_days

# ---------------------------------------------------------------------------
# i18n content
# ---------------------------------------------------------------------------
_TITLE = I18nString(
  en="Overnight Range Breakout",
  fr="Cassure du range overnight",
)
_DEFINITION = I18nString(
  en="Comparing each session's market-hours high and low to the prior overnight session's high and low, how often does price break above the overnight high only, below the overnight low only, both sides, or neither during the RTH session?",
  fr="En comparant le plus haut et le plus bas de la séance régulière au plus haut et au plus bas de la session overnight précédente, à quelle fréquence le prix casse-t-il au-dessus du plus haut overnight uniquement, en-dessous du plus bas uniquement, des deux côtés, ou ni l'un ni l'autre pendant la séance RTH ?",
)
_LABELS = Labels(
  conditions={
    "overnight_range": I18nString(en="Overnight range", fr="Range overnight"),
  },
  outcomes={
    "broke_high": I18nString(
      en="Broke above overnight high only",
      fr="Cassure au-dessus du plus haut overnight uniquement",
    ),
    "broke_low": I18nString(
      en="Broke below overnight low only",
      fr="Cassure en-dessous du plus bas overnight uniquement",
    ),
    "broke_both": I18nString(
      en="Broke both sides",
      fr="Cassure des deux côtés",
    ),
    "neither": I18nString(
      en="Stayed within the overnight range",
      fr="Resté dans le range overnight",
    ),
  },
)

# Outcome enumeration, in display order. All four partition the countable days.
_OUTCOMES: tuple[str, ...] = ("broke_high", "broke_low", "broke_both", "neither")

_BREAKOUT_CRITERIA: tuple[str, ...] = ("wick", "close")


class OvernightRangeBreakout(BaseStat):
  """Breakout-direction distribution of the RTH session relative to the prior
  overnight range."""

  stat_name = "overnight_range_breakout"
  title = _TITLE
  definition = _DEFINITION
  labels = _LABELS
  slices = (
    "weekday",
    "close",
    "prev_candle",
    SizeBucket(column="on_size", preset="quartiles", name="size"),
    Levels(ref="on_size", ext="extension", name="levels"),
  )

  def __init__(
    self,
    instrument: str,
    config: InstrumentConfig,
    breakout_criteria: str = "wick",
    close_tolerance_min: int = 15,
  ) -> None:
    if breakout_criteria not in _BREAKOUT_CRITERIA:
      raise ValueError(
        f"Unsupported breakout_criteria '{breakout_criteria}'. "
        f"Choose from {list(_BREAKOUT_CRITERIA)}"
      )
    self.instrument = instrument
    self.timeframe = "daily"
    self.config = config
    self.breakout_criteria = breakout_criteria
    self.close_tolerance_min = close_tolerance_min

    rth = config.sessions["rth"]
    self.rth_start_min: int = minute_of_day(rth.start)
    self.rth_end_min: int = minute_of_day(rth.end)

    overnight = config.sessions["overnight"]
    self.on_start_min: int = minute_of_day(overnight.start)
    self.on_end_min: int = minute_of_day(overnight.end)

  # -------------------------------------------------------------------------
  # Day table construction
  # -------------------------------------------------------------------------
  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-session table with the overnight range and RTH extremes.

    Columns returned:
      ``on_high`` / ``on_low``           — prior overnight session extremes.
      ``on_size``                        — ``on_high - on_low`` (read by the size slicer).
      ``day_high`` / ``day_low``         — RTH session intraday extremes (wick).
      ``day_close_high`` / ``day_close_low`` — RTH session extreme CLOSES (close
                                           criteria); max / min of the RTH bar closes.
      ``extension``                      — furthest breakout past the overnight range
                                           in either direction (read by the levels slicer).
      ``session_green``                  — session color (``session_close >= session_open``),
                                           read by the ``close`` slicer.
      ``prev_session_green``             — prior session color (NaN for the first resolved
                                           day), read by the ``prev_candle`` slicer.

    The resolved-days core (RTH filter, session open/close, resolution filter,
    chronological sort) is shared via ``build_resolved_days``; the overnight-range
    and RTH-extreme aggregates are joined on per day. A day is countable only when
    it is a resolved RTH day AND has a prior overnight range; days missing either
    drop out via the inner join (pending-sample discipline). The DatetimeIndex
    (normalized session date) is required by the ``weekday`` slicer.
    """
    columns = [
      "on_high",
      "on_low",
      "on_size",
      "day_high",
      "day_low",
      "day_close_high",
      "day_close_low",
      "extension",
      "session_green",
      "prev_session_green",
    ]
    empty = pd.DataFrame(columns=columns)

    if candles_df.empty:
      return empty

    resolved = build_resolved_days(
      candles_df, self.rth_start_min, self.rth_end_min, self.close_tolerance_min
    )
    if resolved.empty:
      return empty

    df = candles_df.copy()
    df["_mod"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["_date"] = df["timestamp"].dt.normalize()

    # Overnight range aggregates. The overnight session crosses midnight, so each
    # bar is attributed to the RTH session date it PRECEDES: evening bars (>=
    # on_start) belong to the next calendar day's overnight; early bars (< on_end)
    # belong to the same calendar day's overnight.
    is_evening = df["_mod"] >= self.on_start_min
    is_early = df["_mod"] < self.on_end_min
    on_bars = df[is_evening | is_early].copy()
    evening = on_bars["_mod"] >= self.on_start_min
    on_bars["_osd"] = on_bars["_date"].where(
      ~evening, on_bars["_date"] + pd.Timedelta(days=1)
    )
    on_high = on_bars.groupby("_osd")["high"].max().rename("on_high")
    on_low = on_bars.groupby("_osd")["low"].min().rename("on_low")

    # RTH session extremes from the same RTH bar filter the resolution uses.
    rth_mask = (df["_mod"] >= self.rth_start_min) & (df["_mod"] < self.rth_end_min)
    rth = df[rth_mask]
    day_high = rth.groupby("_date")["high"].max().rename("day_high")
    day_low = rth.groupby("_date")["low"].min().rename("day_low")
    day_close_high = rth.groupby("_date")["close"].max().rename("day_close_high")
    day_close_low = rth.groupby("_date")["close"].min().rename("day_close_low")

    # Inner join onto the resolved index: a day survives only with an RTH session
    # AND a prior overnight range.
    day = (
      resolved.join(on_high, how="inner")
      .join(on_low, how="inner")
      .join(day_high, how="inner")
      .join(day_low, how="inner")
      .join(day_close_high, how="inner")
      .join(day_close_low, how="inner")
    )
    if day.empty:
      return empty

    day["on_size"] = day["on_high"] - day["on_low"]
    day["session_green"] = day["session_close"] >= day["session_open"]
    # Prior RESOLVED session's color (shift over the resolved-only, sorted index,
    # so it skips any excluded/early-close day). NaN for the first resolved day.
    day["prev_session_green"] = day["session_green"].shift(1)

    # Furthest breakout past the overnight range in either direction, using the
    # same criteria-selected extremes as the break classification (wick or close).
    high, low = self._break_extremes(day)
    ext_up = (high - day["on_high"]).clip(lower=0.0)
    ext_down = (day["on_low"] - low).clip(lower=0.0)
    day["extension"] = pd.concat([ext_up, ext_down], axis=1).max(axis=1)

    return day[columns]

  # -------------------------------------------------------------------------
  # Core computation
  # -------------------------------------------------------------------------
  def _break_extremes(self, day_table: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return the (high, low) RTH extremes selected by ``breakout_criteria``."""
    if self.breakout_criteria == "close":
      return day_table["day_close_high"], day_table["day_close_low"]
    return day_table["day_high"], day_table["day_low"]

  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute the four breakout-direction rows over the (sliced) day table.

    Only days with a valid overnight range (``on_high`` not NaN) are countable. The
    four outcomes are mutually exclusive and partition the countable days, so their
    counts sum to ``total``. If ``baseline_rows`` is provided, merges its
    ``probability`` / ``total`` into each row's ``baseline_prob`` / ``baseline_n``.
    """
    baseline_map: dict[tuple[str, str], StatResultRow] = {}
    if baseline_rows:
      baseline_map = {(r.condition, r.outcome): r for r in baseline_rows}

    def _make(outcome: str, count: int, total: int) -> StatResultRow:
      bl = baseline_map.get(("overnight_range", outcome))
      return StatResultRow(
        condition="overnight_range",
        outcome=outcome,
        count=count,
        total=total,
        probability=count / total if total > 0 else 0.0,
        baseline_prob=bl.probability if bl else 0.0,
        baseline_n=bl.total if bl else 0,
      )

    if day_table.empty:
      return [_make(out, 0, 0) for out in _OUTCOMES]

    on_high = day_table["on_high"]
    on_low = day_table["on_low"]
    high, low = self._break_extremes(day_table)

    countable = on_high.notna() & on_low.notna() & high.notna() & low.notna()
    total = int(countable.sum())

    # Strict inequality defines a break (touching the level exactly is NOT a break).
    above = countable & (high > on_high)
    below = countable & (low < on_low)

    counts = {
      "broke_high": int((above & ~below).sum()),
      "broke_low": int((below & ~above).sum()),
      "broke_both": int((above & below).sum()),
      "neither": int((countable & ~above & ~below).sum()),
    }

    return [_make(out, counts[out], total) for out in _OUTCOMES]

  def classify_samples(self, day_table: pd.DataFrame) -> list[SampleRow]:
    """One SampleRow per countable day, mirroring ``compute_rows``.

    Single condition ("overnight_range"); each countable day (valid overnight
    range AND valid RTH extremes under the selected ``breakout_criteria``)
    contributes exactly one of the four mutually exclusive outcomes
    (``broke_high``, ``broke_low``, ``broke_both``, ``neither``), reproducing
    each row's ``count`` and the table's ``total`` directly. This is a
    probability channel (the four-way partition), not a magnitude one, so
    ``value`` stays ``None``.
    """
    if day_table.empty:
      return []

    on_high = day_table["on_high"]
    on_low = day_table["on_low"]
    high, low = self._break_extremes(day_table)

    countable = on_high.notna() & on_low.notna() & high.notna() & low.notna()
    if not bool(countable.any()):
      return []

    sub = day_table[countable]
    above = high[countable] > on_high[countable]
    below = low[countable] < on_low[countable]

    samples: list[SampleRow] = []
    for ts, is_above, is_below in zip(sub.index, above, below):
      if is_above and is_below:
        outcome = "broke_both"
      elif is_above:
        outcome = "broke_high"
      elif is_below:
        outcome = "broke_low"
      else:
        outcome = "neither"
      samples.append(
        SampleRow(date=ts.strftime("%Y-%m-%d"), condition="overnight_range", outcome=outcome)
      )
    return samples

  # -------------------------------------------------------------------------
  # Baseline
  # -------------------------------------------------------------------------
  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random directional baseline. Deterministic for a fixed seed.

    Each countable day's RTH move is REFLECTED around its overnight-range midpoint
    with probability 0.5 (a price ``p`` maps to ``2*mid - p``, which swaps the high
    and low extremes). Reflection turns a ``broke_high`` day into a ``broke_low``
    day and vice versa, while ``broke_both`` and ``neither`` are direction-symmetric
    and unchanged. The null therefore has NO directional bias: ``broke_high`` and
    ``broke_low`` converge to their shared mean, so the comparison reveals whether
    the overnight range breaks UP more often than DOWN beyond a coin flip. Uses
    ``np.random.default_rng(seed)``.
    """
    n = len(day_table)
    if n == 0:
      return self.compute_rows(day_table, baseline_rows=None)

    rng = np.random.default_rng(seed)
    flip = rng.integers(0, 2, size=n).astype(bool)

    mid = (day_table["on_high"].to_numpy() + day_table["on_low"].to_numpy()) / 2.0
    tmp = day_table.copy()

    # Reflect every column pair (high <-> low) around the midpoint, then keep the
    # reflected value only on flipped days. Both the wick and close extremes are
    # reflected so the chosen breakout_criteria sees a consistent move.
    for hi_col, lo_col in (("day_high", "day_low"), ("day_close_high", "day_close_low")):
      hi = day_table[hi_col].to_numpy()
      lo = day_table[lo_col].to_numpy()
      refl_hi = 2.0 * mid - lo
      refl_lo = 2.0 * mid - hi
      tmp[hi_col] = np.where(flip, refl_hi, hi)
      tmp[lo_col] = np.where(flip, refl_lo, lo)

    return self.compute_rows(tmp, baseline_rows=None)


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------
def run(
  instrument: str = "NQ",
  config_dir: str = "config",
  data_path: str | None = None,
  breakout_criteria: str = "wick",
) -> Path:
  """Load data and compute Overnight Range Breakout for the daily timeframe.

  Writes the result JSON to results/.
  """
  config = load_config(instrument, config_dir=config_dir)
  parquet_path = Path(data_path) if data_path else config.parquet_path

  candles_df = pd.read_parquet(parquet_path)

  stat = OvernightRangeBreakout(
    instrument=instrument,
    config=config,
    breakout_criteria=breakout_criteria,
  )
  result = stat.compute(candles_df)

  return write_results(result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compute Overnight Range Breakout stat")
  parser.add_argument("--instrument", default="NQ", help="Instrument name (default: NQ)")
  parser.add_argument("--data-path", default=None, help="Override parquet file path")
  parser.add_argument("--config-dir", default="config", help="Config directory (default: config)")
  parser.add_argument(
    "--breakout-criteria",
    default="wick",
    choices=list(_BREAKOUT_CRITERIA),
    help="Break detection: 'wick' (intraday extreme) or 'close' (bar close) (default: wick)",
  )
  args = parser.parse_args()

  output_path = run(
    instrument=args.instrument,
    config_dir=args.config_dir,
    data_path=args.data_path,
    breakout_criteria=args.breakout_criteria,
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
