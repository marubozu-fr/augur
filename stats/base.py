"""Base classes and Pydantic models for Augur stat modules."""

import os
import tempfile
from abc import ABC, abstractmethod
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


class Labels(BaseModel):
  conditions: dict[str, I18nString]
  outcomes: dict[str, I18nString]


class TimeframeResult(BaseModel):
  data_range: list[str]  # [min_date, max_date] as "YYYY-MM-DD", empty if no samples
  total_samples: int
  results: list[StatResultRow]


class StatRunResult(BaseModel):
  stat_name: str
  title: I18nString
  definition: I18nString
  labels: Labels
  instruments: dict[str, dict[str, TimeframeResult]]


class BaseStat(ABC):
  @abstractmethod
  def compute(self, candles_df: pd.DataFrame) -> StatRunResult:
    """Compute the statistic from raw 1-min OHLCV data."""
    ...

  @abstractmethod
  def baseline(self, candles_df: pd.DataFrame, seed: int) -> list[StatResultRow]:
    """Compute a random baseline for comparison. Must be deterministic for fixed seed."""
    ...


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
