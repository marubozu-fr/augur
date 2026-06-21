"""In-memory cache for stat result JSON files from results/.

Reads and validates StatRunResult files produced by the stats engine.
Never writes to the results directory.
"""

import logging
from pathlib import Path

from pydantic import BaseModel, ValidationError

from backend.app.core.config import settings
from stats.base import I18nString, StatRunResult

logger = logging.getLogger(__name__)


class TimeframeMeta(BaseModel):
  """Lightweight summary of one instrument/timeframe entry."""

  instrument: str
  timeframe: str
  data_range: list[str]
  total_samples: int


class StatFamilyMeta(BaseModel):
  """Index-level metadata for a stat family — no results arrays."""

  family: str
  title: I18nString
  definition: I18nString
  timeframes: list[TimeframeMeta]


class StatsLoader:
  """Read-only in-memory cache of validated StatRunResult objects.

  Args:
    results_dir: Directory containing stat result JSON files.
      Defaults to settings.results_dir. Override for tests.
  """

  def __init__(self, results_dir: Path | None = None) -> None:
    self._results_dir: Path = results_dir if results_dir is not None else settings.results_dir
    self._cache: dict[str, StatRunResult] = {}

  def scan(self) -> list[Path]:
    """Discover all stat JSON files, excluding atomic temp files.

    Returns:
      Sorted list of .json file paths (excludes .tmp_*.json).
    """
    if not self._results_dir.exists():
      return []
    return sorted(
      p for p in self._results_dir.glob("*.json")
      if not p.name.startswith(".tmp_")
    )

  def load_all(self) -> None:
    """Parse, validate, and cache all stat files found by scan().

    Builds a fresh cache and swaps it in atomically, so a concurrent get()
    never observes a partially-loaded cache. Corrupt or missing files are
    logged as warnings and skipped.
    """
    new_cache: dict[str, StatRunResult] = {}
    for path in self.scan():
      family = path.stem
      try:
        raw = path.read_text(encoding="utf-8")
        result = StatRunResult.model_validate_json(raw)
        new_cache[family] = result
      except (OSError, ValueError, ValidationError) as exc:
        logger.warning("Skipping %s — failed to load: %s", path.name, exc)
    self._cache = new_cache
    logger.info("Loaded %d stat families from %s", len(self._cache), self._results_dir)

  def get(self, family: str) -> StatRunResult | None:
    """Return the cached StatRunResult for a family, or None if absent.

    Args:
      family: Stat family name (equals the JSON filename stem).
    """
    return self._cache.get(family)

  def list_families(self) -> list[StatFamilyMeta]:
    """Return lightweight metadata for every loaded stat family.

    Returns a directory/index view — results arrays are not included.
    """
    metas: list[StatFamilyMeta] = []
    for family, result in sorted(self._cache.items()):
      timeframes: list[TimeframeMeta] = []
      for instrument, tf_map in result.instruments.items():
        for timeframe, tf_result in tf_map.items():
          timeframes.append(
            TimeframeMeta(
              instrument=instrument,
              timeframe=timeframe,
              data_range=tf_result.data_range,
              total_samples=tf_result.total_samples,
            )
          )
      metas.append(
        StatFamilyMeta(
          family=family,
          title=result.title,
          definition=result.definition,
          timeframes=timeframes,
        )
      )
    return metas

  def reload(self) -> None:
    """Re-scan and re-load all stat files. Idempotent.

    Delegates to load_all(), which rebuilds the cache off to the side and
    swaps it in atomically — there is no window where get() sees an empty
    cache, so this is safe to expose via an endpoint later.
    """
    self.load_all()
