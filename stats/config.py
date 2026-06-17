"""Config loader for Augur instrument configuration files."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Session:
  start: str  # "HH:MM" in working_timezone
  end: str    # "HH:MM" in working_timezone


@dataclass(frozen=True)
class InstrumentConfig:
  instrument: str
  working_timezone: str
  sessions: dict[str, Session]
  timeframes: list[str]
  parquet_path: Path


def minute_of_day(hhmm: str) -> int:
  """Convert "HH:MM" time string to minutes since midnight.

  Example: "09:30" -> 570, "16:15" -> 975.
  """
  h, m = hhmm.split(":")
  return int(h) * 60 + int(m)


def load_config(instrument: str, config_dir: str | Path = "config") -> InstrumentConfig:
  """Load instrument configuration from a YAML file.

  Reads `config/{instrument}.yaml`. The parquet_path is resolved from
  `data.parquet_1min` if present in the YAML, otherwise defaults to
  `data/{instrument}_1min.parquet`.
  """
  config_path = Path(config_dir) / f"{instrument}.yaml"
  with open(config_path, "r", encoding="utf-8") as f:
    raw = yaml.safe_load(f)

  sessions: dict[str, Session] = {}
  for name, times in raw.get("sessions", {}).items():
    sessions[name] = Session(start=times["start"], end=times["end"])

  data_section = raw.get("data", {})
  if "parquet_1min" in data_section:
    parquet_path = Path(data_section["parquet_1min"])
  else:
    parquet_path = Path("data") / f"{instrument}_1min.parquet"

  return InstrumentConfig(
    instrument=raw["instrument"],
    working_timezone=raw["working_timezone"],
    sessions=sessions,
    timeframes=raw.get("timeframes", []),
    parquet_path=parquet_path,
  )
