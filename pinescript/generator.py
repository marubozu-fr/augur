"""Generate a self-contained TradingView Pine Script v6 indicator from stat results.

Reads `results/opening_candle_continuation.json` and the instrument config, then
renders a chart-timeframe-agnostic Pine Script overlay indicator via Jinja2.
All probabilities are embedded as static `var` constants — no runtime data fetches.

CLI:
  python -m pinescript.generator --instrument NQ --output output/augur_NQ.pine
"""

import argparse
import os
import tempfile
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from stats.base import StatResultRow, StatRunResult, TimeframeResult
from stats.config import load_config, minute_of_day

_STAT_FILENAME = "opening_candle_continuation.json"
_TEMPLATE_NAME = "overlay_table.pine.j2"
_TEMPLATES_DIR = Path(__file__).parent / "templates"
_TABLE_POSITION = "position.top_right"

# Bilingual label for the RTH session-open row. The config currently carries no
# session display name, so these defaults are used.
_OPEN_LABEL_EN = "New York Open"
_OPEN_LABEL_FR = "Ouverture New York"


def _timeframe_minutes(timeframe: str) -> int:
  """Parse a timeframe label into minutes since the candle opens.

  Examples: "15min" -> 15, "30min" -> 30, "1h" -> 60.
  """
  if timeframe.endswith("min"):
    return int(timeframe[:-3])
  if timeframe.endswith("h"):
    return int(timeframe[:-1]) * 60
  raise ValueError(f"Unsupported timeframe label '{timeframe}'")


def _format_hhmm(minutes: int) -> str:
  """Format minutes-since-midnight as a "HH:MM" 24h string. 585 -> "09:45"."""
  h, m = divmod(minutes, 60)
  return f"{h:02d}:{m:02d}"


def _find_row(rows: list[StatResultRow], condition: str, outcome: str) -> StatResultRow | None:
  """Return the row matching condition+outcome, or None if absent."""
  for row in rows:
    if row.condition == condition and row.outcome == outcome:
      return row
  return None


def _build_timeframes(
  instrument_data: dict[str, TimeframeResult],
  rth_start_min: int,
) -> list[dict[str, object]]:
  """Build the per-timeframe render context, sorted by opening-candle close time.

  Each entry carries all four conditional outcome probabilities (so the indicator
  can show the most-likely session direction, continuation or reversal), the green
  and red opening sample sizes, the Pine timeframe string, and the close-detection
  minute (rth_start + timeframe duration).

  Timeframes missing any of the four outcome rows are skipped (issue edge case).
  """
  entries: list[dict[str, object]] = []
  for tf_key, tf_result in instrument_data.items():
    # All four conditional outcomes; prob_gg + prob_gr == 1 and prob_rg + prob_rr == 1.
    gg = _find_row(tf_result.results, "green_open", "green_close")
    gr = _find_row(tf_result.results, "green_open", "red_close")
    rg = _find_row(tf_result.results, "red_open", "green_close")
    rr = _find_row(tf_result.results, "red_open", "red_close")
    if gg is None or gr is None or rg is None or rr is None:
      continue

    tf_minutes = _timeframe_minutes(tf_key)
    close_min = rth_start_min + tf_minutes
    entries.append({
      "label": tf_key,
      "pine_tf": str(tf_minutes),
      "var": f"tf{tf_minutes}",
      "close_mod": close_min,
      "close_str": _format_hhmm(close_min),
      "prob_gg": gg.probability,
      "prob_gr": gr.probability,
      "prob_rg": rg.probability,
      "prob_rr": rr.probability,
      "n_green": gg.total,
      "n_red": rr.total,
    })

  entries.sort(key=lambda e: e["close_mod"])
  for row_index, entry in enumerate(entries):
    entry["row"] = row_index
  return entries


def _atomic_write(path: Path, content: str) -> None:
  """Write content to path atomically (temp file in same dir, then rename)."""
  path.parent.mkdir(parents=True, exist_ok=True)
  fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".pine")
  try:
    with os.fdopen(fd, "w", encoding="utf-8") as f:
      f.write(content)
    os.replace(tmp_path, path)
  except Exception:
    try:
      os.unlink(tmp_path)
    except OSError:
      pass
    raise


def render_indicator(
  instrument: str,
  results_dir: str | Path = "results",
  config_dir: str | Path = "config",
  templates_dir: str | Path = _TEMPLATES_DIR,
) -> str:
  """Render the Pine Script source for an instrument and return it as a string.

  All session times and the timezone come from `config/{instrument}.yaml`; nothing
  is hardcoded. Raises if the stat file, instrument, or RTH session is missing, or
  if no timeframe yields renderable results.
  """
  config = load_config(instrument, config_dir=config_dir)
  rth = config.sessions.get("rth")
  if rth is None:
    raise ValueError(f"Config for '{instrument}' has no 'rth' session")
  rth_start_min = minute_of_day(rth.start)
  rth_end_min = minute_of_day(rth.end)

  stat_path = Path(results_dir) / _STAT_FILENAME
  if not stat_path.exists():
    raise FileNotFoundError(f"Stat result file not found: {stat_path}")
  result = StatRunResult.model_validate_json(stat_path.read_text(encoding="utf-8"))

  if instrument not in result.instruments:
    raise KeyError(f"Instrument '{instrument}' not found in {stat_path}")
  timeframes = _build_timeframes(result.instruments[instrument], rth_start_min)
  if not timeframes:
    raise ValueError(f"No renderable timeframes for '{instrument}' in {stat_path}")

  env = Environment(
    loader=FileSystemLoader(str(templates_dir)),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
  )
  template = env.get_template(_TEMPLATE_NAME)
  return template.render(
    instrument=instrument,
    # syminfo.root resolves to the instrument symbol (e.g. "NQ" for NQ futures);
    # the indicator refuses to display stats on any other asset.
    expected_root=instrument,
    timezone=config.working_timezone,
    rth_start_mod=rth_start_min,
    rth_end_mod=rth_end_min,
    rth_start_str=_format_hhmm(rth_start_min),
    open_label_en=_OPEN_LABEL_EN,
    open_label_fr=_OPEN_LABEL_FR,
    table_position=_TABLE_POSITION,
    n_rows=len(timeframes),
    timeframes=timeframes,
  )


def generate(
  instrument: str,
  output_path: str | Path | None = None,
  results_dir: str | Path = "results",
  config_dir: str | Path = "config",
  templates_dir: str | Path = _TEMPLATES_DIR,
) -> Path:
  """Render and atomically write the Pine Script indicator for an instrument.

  Returns the path of the written file. Defaults output to
  `output/augur_{instrument}.pine`.
  """
  rendered = render_indicator(
    instrument,
    results_dir=results_dir,
    config_dir=config_dir,
    templates_dir=templates_dir,
  )
  if output_path is None:
    output_path = Path("output") / f"augur_{instrument}.pine"
  output_path = Path(output_path)
  _atomic_write(output_path, rendered)
  return output_path


def main() -> None:
  parser = argparse.ArgumentParser(description="Generate a Pine Script v6 indicator from stat results.")
  parser.add_argument("--instrument", required=True, help="Instrument symbol (e.g. NQ).")
  parser.add_argument(
    "--output",
    default=None,
    help="Output .pine path (default: output/augur_{instrument}.pine).",
  )
  parser.add_argument("--results-dir", default="results", help="Directory holding stat result JSON files.")
  parser.add_argument("--config-dir", default="config", help="Directory holding instrument config YAML files.")
  args = parser.parse_args()

  written = generate(
    args.instrument,
    output_path=args.output,
    results_dir=args.results_dir,
    config_dir=args.config_dir,
  )
  print(f"Generated {written}")


if __name__ == "__main__":
  main()
