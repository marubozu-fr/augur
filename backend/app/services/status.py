"""Business logic for /admin/status — gathers loaded stats, data, versions, uptime."""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import fastapi
import pyarrow.parquet as pq
from pydantic import BaseModel

from backend.app.core.stats_loader import StatsLoader

logger = logging.getLogger(__name__)


class LoadedStatFile(BaseModel):
  """One result JSON file currently held in the StatsLoader cache."""

  family: str
  filename: str
  size_bytes: int
  modified_at: str  # ISO 8601 UTC datetime


class DataRange(BaseModel):
  """Inclusive [start, end] date range covered by a Parquet file."""

  start: str  # ISO 8601 date
  end: str  # ISO 8601 date


class DataFileInfo(BaseModel):
  """One Parquet file discovered in the data directory."""

  filename: str  # relative to data_dir, so subdirectories disambiguate basenames
  size_bytes: int
  num_rows: int | None
  data_range: DataRange | None


class Versions(BaseModel):
  """Runtime versions reported by the running process."""

  python: str
  fastapi: str


class SystemStatus(BaseModel):
  """Aggregate system status payload returned by GET /admin/status."""

  started_at: str  # ISO 8601 UTC datetime
  uptime_seconds: float
  versions: Versions
  stats_files: list[LoadedStatFile]
  data_files: list[DataFileInfo]


def collect_loaded_stat_files(loader: StatsLoader) -> list[LoadedStatFile]:
  """Build the list of loaded stat files with their on-disk size and mtime."""
  files: list[LoadedStatFile] = []
  for meta in loader.list_families():
    path = loader.results_dir / f"{meta.family}.json"
    if not path.exists():
      continue
    st = path.stat()
    files.append(
      LoadedStatFile(
        family=meta.family,
        filename=path.name,
        size_bytes=st.st_size,
        modified_at=datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
      )
    )
  return files


def collect_data_files(data_dir: Path) -> list[DataFileInfo]:
  """List Parquet files in data_dir (recursively) with size, row count, and date range.

  Recursion matters: Augur's prepared OHLCV files live under data/processed/ per
  config (processed_dir), not at the top level.
  """
  if not data_dir.exists():
    return []
  return [_describe_parquet(p, data_dir) for p in sorted(data_dir.rglob("*.parquet"))]


def _describe_parquet(path: Path, data_dir: Path) -> DataFileInfo:
  size_bytes = path.stat().st_size
  num_rows: int | None = None
  data_range: DataRange | None = None
  # The data directory is prepared externally — treat it as an untrusted boundary.
  try:
    pf = pq.ParquetFile(path)
    md = pf.metadata
    num_rows = md.num_rows
    data_range = _extract_timestamp_range(md)
  except Exception as exc:  # noqa: BLE001 — external file boundary
    logger.warning("Failed to read Parquet metadata for %s: %s", path.name, exc)
  return DataFileInfo(
    filename=str(path.relative_to(data_dir)),
    size_bytes=size_bytes,
    num_rows=num_rows,
    data_range=data_range,
  )


def _extract_timestamp_range(md: pq.FileMetaData) -> DataRange | None:
  """Return the [min, max] date range from column 0 row-group stats, or None.

  Augur Parquet files put a tz-aware timestamp column first; we aggregate min/max
  across all row groups so the answer is exact when stats are present, without
  reading any actual data pages.
  """
  if md.num_row_groups == 0 or md.row_group(0).num_columns == 0:
    return None
  mins: list[datetime] = []
  maxs: list[datetime] = []
  for rg in range(md.num_row_groups):
    col = md.row_group(rg).column(0)
    if not col.is_stats_set:
      continue
    cmin, cmax = col.statistics.min, col.statistics.max
    if isinstance(cmin, datetime) and isinstance(cmax, datetime):
      mins.append(cmin)
      maxs.append(cmax)
  if not mins or not maxs:
    return None
  return DataRange(start=min(mins).date().isoformat(), end=max(maxs).date().isoformat())


def get_versions() -> Versions:
  """Return the Python and FastAPI version strings of the running process."""
  py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
  return Versions(python=py, fastapi=fastapi.__version__)


def build_system_status(
  loader: StatsLoader,
  data_dir: Path,
  started_at: datetime,
) -> SystemStatus:
  """Compose the full SystemStatus payload."""
  now = datetime.now(timezone.utc)
  uptime = (now - started_at).total_seconds()
  return SystemStatus(
    started_at=started_at.isoformat(),
    uptime_seconds=uptime,
    versions=get_versions(),
    stats_files=collect_loaded_stat_files(loader),
    data_files=collect_data_files(data_dir),
  )
