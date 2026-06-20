"""Base classes, Pydantic models, and the declarative slice architecture.

A stat family computes a per-day table once, then the framework re-runs the
family's core row computation over named subsets of those days. Each subset is
defined by a *slicer* (weekday, size bucket, prior-candle color, ...). Stats
declare which slicers apply via the class-level ``slices`` attribute; they never
implement the slicing themselves.
"""

import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pydantic import BaseModel


class I18nString(BaseModel):
  en: str
  fr: str


class StatResultRow(BaseModel):
  condition: str
  outcome: str
  count: int
  total: int
  probability: float
  baseline_prob: float
  baseline_n: int
  # Continuous-metric channel for magnitude stats (e.g. average percent return).
  # Probability rows leave these None; magnitude rows populate `value` with the
  # metric and `value_baseline` with its random-baseline counterpart.
  value: float | None = None
  value_baseline: float | None = None


class Labels(BaseModel):
  conditions: dict[str, I18nString]
  outcomes: dict[str, I18nString]
  # Static, human-readable names of the slice dimensions a stat exposes
  # (e.g. "weekday" -> {"en": "Day of week", "fr": "Jour de la semaine"}).
  # Group labels live inside SliceGroupResult because some are data-dependent
  # (quantile bucket edges differ per instrument/timeframe).
  dimensions: dict[str, I18nString] = {}


class SliceGroupResult(BaseModel):
  """Results for one group within a slice (e.g. Mondays, or size quartile Q1)."""

  label: I18nString
  total_samples: int
  results: list[StatResultRow]


class SliceResult(BaseModel):
  """All groups produced by a single slicer for one timeframe."""

  dimension: str  # slicer name, e.g. "weekday"
  groups: dict[str, SliceGroupResult]


class TimeframeResult(BaseModel):
  data_range: list[str]  # [min_date, max_date] as "YYYY-MM-DD", empty if no samples
  total_samples: int
  results: list[StatResultRow]
  # Sliced breakdowns, keyed by slicer name. Empty when a stat declares no slices.
  slices: dict[str, SliceResult] = {}


class StatRunResult(BaseModel):
  stat_name: str
  title: I18nString
  definition: I18nString
  labels: Labels
  instruments: dict[str, dict[str, TimeframeResult]]


# ===========================================================================
# Slice architecture
# ===========================================================================
@dataclass
class SliceGroup:
  """One named subset of the per-day table.

  ``mask`` is a boolean Series aligned to the day-table index selecting the
  rows that belong to this group. ``label`` travels with the group so that
  data-dependent labels (quantile edges, level bands) can be expressed.
  """

  key: str
  label: I18nString
  mask: pd.Series


class Slicer(ABC):
  """Partitions a per-day table into named groups.

  A slicer is pure: it reads columns from the day table and returns groups.
  It never computes statistics — the framework runs the stat's ``compute_rows``
  over each group's subset.
  """

  name: str

  @abstractmethod
  def dimension_label(self) -> I18nString:
    """Human-readable name of the dimension this slicer splits on."""
    ...

  @abstractmethod
  def split(self, day_table: pd.DataFrame) -> list[SliceGroup]:
    """Return the groups for this day table, in display order.

    Empty groups are omitted. Rows with missing values in the required
    columns are excluded from every group (pending-sample discipline).
    """
    ...


_WEEKDAYS: list[tuple[int, str, str, str]] = [
  (0, "monday", "Monday", "Lundi"),
  (1, "tuesday", "Tuesday", "Mardi"),
  (2, "wednesday", "Wednesday", "Mercredi"),
  (3, "thursday", "Thursday", "Jeudi"),
  (4, "friday", "Friday", "Vendredi"),
  (5, "saturday", "Saturday", "Samedi"),
  (6, "sunday", "Sunday", "Dimanche"),
]


class Weekday(Slicer):
  """Group days by day of week. Reads the DatetimeIndex of the day table."""

  name = "weekday"

  def dimension_label(self) -> I18nString:
    return I18nString(en="Day of week", fr="Jour de la semaine")

  def split(self, day_table: pd.DataFrame) -> list[SliceGroup]:
    if len(day_table) == 0:
      return []
    dow = day_table.index.dayofweek
    groups: list[SliceGroup] = []
    for num, key, en, fr in _WEEKDAYS:
      mask = pd.Series(dow == num, index=day_table.index)
      if bool(mask.any()):
        groups.append(SliceGroup(key=key, label=I18nString(en=en, fr=fr), mask=mask))
    return groups


class _ColorSlicer(Slicer):
  """Shared logic for slicers that split a boolean column into green/red."""

  def __init__(self, column: str, name: str) -> None:
    self.column = column
    self.name = name

  def split(self, day_table: pd.DataFrame) -> list[SliceGroup]:
    if len(day_table) == 0 or self.column not in day_table.columns:
      return []
    col = day_table[self.column]
    present = col.notna()
    groups: list[SliceGroup] = []
    specs = [
      ("green", I18nString(en="Green", fr="Verte"), present & col.fillna(False).astype(bool)),
      ("red", I18nString(en="Red", fr="Rouge"), present & ~col.fillna(True).astype(bool)),
    ]
    for key, label, mask in specs:
      if bool(mask.any()):
        groups.append(SliceGroup(key=key, label=label, mask=mask))
    return groups


class Close(_ColorSlicer):
  """Split by session close color after the measured event."""

  def __init__(self, column: str = "session_green", name: str = "close") -> None:
    super().__init__(column=column, name=name)

  def dimension_label(self) -> I18nString:
    return I18nString(en="Session close color", fr="Couleur de clôture de session")


class PrevCandle(_ColorSlicer):
  """Split by the prior session candle color."""

  def __init__(self, column: str = "prev_session_green", name: str = "prev_candle") -> None:
    super().__init__(column=column, name=name)

  def dimension_label(self) -> I18nString:
    return I18nString(en="Prior session candle", fr="Bougie de la session précédente")


class Overnight(_ColorSlicer):
  """Split by overnight gap direction: RTH open above (green) / below (red) prior close.

  ``green`` collects sessions opening above the previous session's close (gap up);
  ``red`` collects sessions opening below it (gap down). The first resolved day has
  no prior close and is excluded (pending-sample discipline).
  """

  def __init__(self, column: str = "overnight_green", name: str = "overnight") -> None:
    super().__init__(column=column, name=name)

  def dimension_label(self) -> I18nString:
    return I18nString(en="Overnight gap", fr="Gap overnight")


class Rejection(Slicer):
  """Split by which extreme of a range formed first: high vs low.

  Reads a boolean column where ``True`` means the range's high was reached before
  its low (high formed first) and ``False`` the opposite. Rows whose value is
  missing — e.g. a single bar held both the high and the low so the order is
  undetermined — are excluded from every group (pending-sample discipline).
  """

  def __init__(self, column: str = "high_first", name: str = "rejection") -> None:
    self.column = column
    self.name = name

  def dimension_label(self) -> I18nString:
    return I18nString(en="First extreme of the range", fr="Premier extrême du range")

  def split(self, day_table: pd.DataFrame) -> list[SliceGroup]:
    if len(day_table) == 0 or self.column not in day_table.columns:
      return []
    col = day_table[self.column]
    present = col.notna()
    groups: list[SliceGroup] = []
    specs = [
      (
        "high_first",
        I18nString(en="High formed first", fr="Plus haut formé en premier"),
        present & col.fillna(False).astype(bool),
      ),
      (
        "low_first",
        I18nString(en="Low formed first", fr="Plus bas formé en premier"),
        present & ~col.fillna(True).astype(bool),
      ),
    ]
    for key, label, mask in specs:
      if bool(mask.any()):
        groups.append(SliceGroup(key=key, label=label, mask=mask))
    return groups


_PRESET_BINS: dict[str, int] = {
  "median": 2,
  "terciles": 3,
  "quartiles": 4,
  "quintiles": 5,
}


class SizeBucket(Slicer):
  """Bucket days by a configurable size metric.

  Either a ``preset`` (equal-frequency quantile bins via the column's own
  distribution) or explicit ``buckets`` (fixed numeric edges) defines the bins.
  Bucket labels carry the numeric range and are therefore data-dependent.
  """

  def __init__(
    self,
    column: str,
    preset: str = "quartiles",
    buckets: list[float] | None = None,
    name: str = "size_bucket",
  ) -> None:
    if buckets is None and preset not in _PRESET_BINS:
      raise ValueError(
        f"Unknown preset '{preset}'. Choose from {list(_PRESET_BINS)} or pass explicit buckets."
      )
    self.column = column
    self.preset = preset
    self.buckets = buckets
    self.name = name

  def dimension_label(self) -> I18nString:
    return I18nString(en="Size bucket", fr="Tranche de taille")

  def split(self, day_table: pd.DataFrame) -> list[SliceGroup]:
    if len(day_table) == 0 or self.column not in day_table.columns:
      return []
    values = pd.to_numeric(day_table[self.column], errors="coerce")
    valid = values.notna()
    if not bool(valid.any()):
      return []

    if self.buckets is not None:
      binned = pd.cut(values, bins=self.buckets, include_lowest=True)
    else:
      # Equal-frequency quantile bins; duplicates='drop' collapses ties so a
      # near-constant column does not raise.
      binned = pd.qcut(values, q=_PRESET_BINS[self.preset], duplicates="drop")

    groups: list[SliceGroup] = []
    categories = list(binned.cat.categories)
    for i, interval in enumerate(categories, start=1):
      mask = pd.Series(binned == interval, index=day_table.index).fillna(False)
      if not bool(mask.any()):
        continue
      lo, hi = float(interval.left), float(interval.right)
      groups.append(
        SliceGroup(
          key=f"q{i}",
          label=I18nString(
            en=f"{self.column} bin {i} ({lo:.2f}–{hi:.2f}]",
            fr=f"{self.column} tranche {i} ({lo:.2f}–{hi:.2f}]",
          ),
          mask=mask,
        )
      )
    return groups


class Levels(Slicer):
  """Bucket days by extension measured in multiples of a reference range.

  ``ext / ref`` is binned into bands delimited by ``multiples`` (e.g. <0.5x,
  0.5–1x, 1–1.5x, 1.5–2x, >=2x). Rows with a non-positive or missing reference
  range are excluded.
  """

  def __init__(
    self,
    ref: str,
    ext: str,
    multiples: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0),
    name: str = "levels",
  ) -> None:
    self.ref = ref
    self.ext = ext
    self.multiples = sorted(float(m) for m in multiples)
    self.name = name

  def dimension_label(self) -> I18nString:
    return I18nString(en="Extension level", fr="Niveau d'extension")

  def split(self, day_table: pd.DataFrame) -> list[SliceGroup]:
    if len(day_table) == 0:
      return []
    if self.ref not in day_table.columns or self.ext not in day_table.columns:
      return []
    ref = pd.to_numeric(day_table[self.ref], errors="coerce")
    ext = pd.to_numeric(day_table[self.ext], errors="coerce")
    valid = ref.notna() & ext.notna() & (ref > 0)
    if not bool(valid.any()):
      return []
    ratio = ext.where(valid) / ref.where(valid)

    edges = [0.0, *self.multiples, float("inf")]
    groups: list[SliceGroup] = []
    for i in range(len(edges) - 1):
      lo, hi = edges[i], edges[i + 1]
      band = valid & (ratio >= lo) & (ratio < hi)
      if not bool(band.any()):
        continue
      if hi == float("inf"):
        en, fr = f">={lo:g}x", f">={lo:g}x"
        key = f"ge_{_mult_key(lo)}x"
      else:
        en, fr = f"{lo:g}–{hi:g}x", f"{lo:g}–{hi:g}x"
        key = f"{_mult_key(lo)}_{_mult_key(hi)}x"
      groups.append(
        SliceGroup(key=key, label=I18nString(en=en, fr=fr), mask=band.fillna(False))
      )
    return groups


def _mult_key(m: float) -> str:
  """Stable key fragment for a multiple, e.g. 0.5 -> '0_5', 2.0 -> '2'."""
  return f"{m:g}".replace(".", "_")


# Bare-string shorthand is only allowed for parameterless slicers. Parameterized
# slicers (SizeBucket, Levels) must be declared as configured instances.
_SLICER_SHORTHAND: dict[str, type[Slicer]] = {
  "weekday": Weekday,
  "close": Close,
  "prev_candle": PrevCandle,
  "overnight": Overnight,
}


def resolve_slicer(spec: "str | Slicer") -> Slicer:
  """Turn a declarative slice spec into a Slicer instance."""
  if isinstance(spec, Slicer):
    return spec
  if spec in _SLICER_SHORTHAND:
    return _SLICER_SHORTHAND[spec]()
  raise ValueError(
    f"Unknown slice shorthand '{spec}'. Known: {list(_SLICER_SHORTHAND)}. "
    "Parameterized slicers (size_bucket, levels) must be declared as instances, "
    "e.g. SizeBucket(column='gap_size') or Levels(ref='orb_range', ext='extension')."
  )


# ===========================================================================
# BaseStat
# ===========================================================================
class BaseStat(ABC):
  """A stat family. Implements the core per-day computation; the framework
  orchestrates the overall result and every declared slice.

  Subclasses set the metadata class attributes (``stat_name``, ``title``,
  ``definition``, ``labels``), the declarative ``slices`` tuple, the instance
  attributes ``instrument`` and ``timeframe``, and implement the three hooks
  below.
  """

  stat_name: str
  title: I18nString
  definition: I18nString
  labels: Labels
  slices: tuple[str | Slicer, ...] = ()

  instrument: str
  timeframe: str

  @abstractmethod
  def build_day_table(self, candles_df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-day table from raw 1-min OHLCV data.

    The index must be the normalized session date. Columns must include every
    field the core computation and the declared slicers read. Only resolved
    (confirmed) days appear; pending days are excluded here.
    """
    ...

  @abstractmethod
  def compute_rows(
    self,
    day_table: pd.DataFrame,
    baseline_rows: list[StatResultRow] | None = None,
  ) -> list[StatResultRow]:
    """Compute condition/outcome rows over a (possibly sliced) day table.

    When ``baseline_rows`` is provided, merge their rates into the rows'
    ``baseline_prob`` / ``baseline_n`` fields.
    """
    ...

  @abstractmethod
  def baseline_rows(self, day_table: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Random baseline over a day table. Deterministic for a fixed seed."""
    ...

  def baseline(self, candles_df: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Convenience: build the day table and compute the random baseline."""
    return self.baseline_rows(self.build_day_table(candles_df), seed)

  def _slice_results(
    self, day_table: pd.DataFrame, seed: int
  ) -> tuple[dict[str, SliceResult], dict[str, I18nString]]:
    """Compute every declared slice and collect their dimension labels."""
    slice_results: dict[str, SliceResult] = {}
    dimension_labels: dict[str, I18nString] = {}
    for spec in self.slices:
      slicer = resolve_slicer(spec)
      dimension_labels[slicer.name] = slicer.dimension_label()
      groups: dict[str, SliceGroupResult] = {}
      for group in slicer.split(day_table):
        sub = day_table[group.mask]
        bl = self.baseline_rows(sub, seed)
        groups[group.key] = SliceGroupResult(
          label=group.label,
          total_samples=len(sub),
          results=self.compute_rows(sub, baseline_rows=bl),
        )
      slice_results[slicer.name] = SliceResult(dimension=slicer.name, groups=groups)
    return slice_results, dimension_labels

  def compute(self, candles_df: pd.DataFrame, seed: int = 42) -> StatRunResult:
    """Compute the overall result plus every declared slice for one timeframe."""
    day_table = self.build_day_table(candles_df)
    baseline = self.baseline_rows(day_table, seed)
    rows = self.compute_rows(day_table, baseline_rows=baseline)
    slice_results, dimension_labels = self._slice_results(day_table, seed)

    if len(day_table) > 0:
      dates = day_table.index
      data_range = [dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")]
    else:
      data_range = []

    tf_result = TimeframeResult(
      data_range=data_range,
      total_samples=len(day_table),
      results=rows,
      slices=slice_results,
    )

    labels = self.labels.model_copy(deep=True)
    labels.dimensions = {**labels.dimensions, **dimension_labels}

    return StatRunResult(
      stat_name=self.stat_name,
      title=self.title,
      definition=self.definition,
      labels=labels,
      instruments={self.instrument: {self.timeframe: tf_result}},
    )


def write_results(
  result: StatRunResult,
  results_dir: str | Path = "results",
) -> Path:
  """Validate and atomically write a StatRunResult to a JSON file.

  The file is written to a temp file first, then renamed to ensure
  atomicity (no partial reads on failure).

  Returns the path of the written file.
  """
  results_dir = Path(results_dir)
  results_dir.mkdir(parents=True, exist_ok=True)

  # Defensive re-validation: re-instantiate from dict to catch any drift
  validated = StatRunResult.model_validate(result.model_dump())

  final_path = results_dir / f"{validated.stat_name}.json"

  # Atomic write: write to temp in same directory, then rename
  fd, tmp_path = tempfile.mkstemp(dir=results_dir, prefix=".tmp_", suffix=".json")
  try:
    with os.fdopen(fd, "w", encoding="utf-8") as f:
      f.write(validated.model_dump_json(indent=2))
      f.write("\n")
    os.replace(tmp_path, final_path)
  except Exception:
    # Clean up temp file if something goes wrong
    try:
      os.unlink(tmp_path)
    except OSError:
      pass
    raise

  return final_path
