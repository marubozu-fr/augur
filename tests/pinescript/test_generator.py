"""Tests for pinescript.generator.

All inputs are synthetic — no real market data or committed config required.
Each test writes a self-contained results JSON and instrument config YAML into a
tmp directory, so config-derived values (session times, timezone, instrument
name) can be asserted independently of the repo's NQ.yaml.
"""

import json
from pathlib import Path

import pytest

from pinescript.generator import (
  _format_hhmm,
  _timeframe_minutes,
  generate,
  render_indicator,
)


# ---------------------------------------------------------------------------
# Synthetic input builders
# ---------------------------------------------------------------------------

def _tf_block(prob_green: float, n_green: int, prob_red: float, n_red: int) -> dict:
  """Build one TimeframeResult block with the four condition/outcome rows.

  Only green_open/green_close and red_open/red_close are read by the generator;
  the complementary rows are filled with plausible values for completeness.
  """
  return {
    "data_range": ["2020-01-01", "2024-01-01"],
    "total_samples": n_green + n_red,
    "results": [
      {
        "condition": "green_open", "outcome": "green_close",
        "count": round(prob_green * n_green), "total": n_green,
        "probability": prob_green, "baseline_prob": 0.5, "baseline_n": n_green,
      },
      {
        "condition": "green_open", "outcome": "red_close",
        "count": n_green - round(prob_green * n_green), "total": n_green,
        "probability": 1 - prob_green, "baseline_prob": 0.5, "baseline_n": n_green,
      },
      {
        "condition": "red_open", "outcome": "green_close",
        "count": n_red - round(prob_red * n_red), "total": n_red,
        "probability": 1 - prob_red, "baseline_prob": 0.5, "baseline_n": n_red,
      },
      {
        "condition": "red_open", "outcome": "red_close",
        "count": round(prob_red * n_red), "total": n_red,
        "probability": prob_red, "baseline_prob": 0.5, "baseline_n": n_red,
      },
    ],
  }


def _write_results(results_dir: Path, instrument: str, timeframes: dict) -> Path:
  """Write a synthetic opening_candle_continuation.json into results_dir."""
  doc = {
    "stat_name": "opening_candle_continuation",
    "title": {"en": "Opening Candle Continuation", "fr": "Continuation de la bougie d'ouverture"},
    "definition": {"en": "x", "fr": "y"},
    "labels": {
      "conditions": {
        "green_open": {"en": "Green opening candle", "fr": "Bougie d'ouverture verte"},
        "red_open": {"en": "Red opening candle", "fr": "Bougie d'ouverture rouge"},
      },
      "outcomes": {
        "green_close": {"en": "Session closes green", "fr": "Session clôture en vert"},
        "red_close": {"en": "Session closes red", "fr": "Session clôture en rouge"},
      },
    },
    "instruments": {instrument: timeframes},
  }
  results_dir.mkdir(parents=True, exist_ok=True)
  path = results_dir / "opening_candle_continuation.json"
  path.write_text(json.dumps(doc), encoding="utf-8")
  return path


def _write_config(
  config_dir: Path,
  instrument: str,
  rth_start: str = "09:30",
  rth_end: str = "16:15",
  timezone: str = "America/New_York",
) -> Path:
  """Write a minimal instrument config YAML into config_dir."""
  config_dir.mkdir(parents=True, exist_ok=True)
  path = config_dir / f"{instrument}.yaml"
  path.write_text(
    f"instrument: {instrument}\n"
    f"working_timezone: {timezone}\n"
    "sessions:\n"
    "  rth:\n"
    f'    start: "{rth_start}"\n'
    f'    end: "{rth_end}"\n'
    "timeframes:\n"
    "  - 15min\n  - 30min\n  - 1h\n",
    encoding="utf-8",
  )
  return path


def _setup(tmp_path: Path, instrument: str = "NQ", rth_start: str = "09:30",
           rth_end: str = "16:15", timezone: str = "America/New_York",
           timeframes: dict | None = None):
  """Create results + config dirs for a default 3-timeframe NQ setup."""
  if timeframes is None:
    timeframes = {
      "15min": _tf_block(0.75, 2000, 0.60, 1500),
      "30min": _tf_block(0.80, 2100, 0.65, 1400),
      "1h": _tf_block(0.70, 2200, 0.55, 1300),
    }
  results_dir = tmp_path / "results"
  config_dir = tmp_path / "config"
  _write_results(results_dir, instrument, timeframes)
  _write_config(config_dir, instrument, rth_start=rth_start, rth_end=rth_end, timezone=timezone)
  return results_dir, config_dir


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,minutes", [("15min", 15), ("30min", 30), ("1h", 60)])
def test_timeframe_minutes(label: str, minutes: int) -> None:
  assert _timeframe_minutes(label) == minutes


def test_timeframe_minutes_invalid_raises() -> None:
  with pytest.raises(ValueError, match="Unsupported timeframe"):
    _timeframe_minutes("daily")


@pytest.mark.parametrize("minutes,expected", [(585, "09:45"), (600, "10:00"), (630, "10:30"), (0, "00:00")])
def test_format_hhmm(minutes: int, expected: str) -> None:
  assert _format_hhmm(minutes) == expected


# ---------------------------------------------------------------------------
# generate() writes a file
# ---------------------------------------------------------------------------

def test_generate_creates_file(tmp_path: Path) -> None:
  results_dir, config_dir = _setup(tmp_path)
  out = tmp_path / "out" / "augur_NQ.pine"
  written = generate("NQ", output_path=out, results_dir=results_dir, config_dir=config_dir)
  assert written == out
  assert out.exists()
  assert out.read_text(encoding="utf-8").startswith("//@version=6")


def test_generate_default_output_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  results_dir, config_dir = _setup(tmp_path)
  monkeypatch.chdir(tmp_path)
  written = generate("NQ", results_dir=results_dir, config_dir=config_dir)
  assert written == Path("output") / "augur_NQ.pine"
  assert written.exists()


def test_generate_no_leftover_tmp_files(tmp_path: Path) -> None:
  results_dir, config_dir = _setup(tmp_path)
  out_dir = tmp_path / "out"
  generate("NQ", output_path=out_dir / "augur_NQ.pine", results_dir=results_dir, config_dir=config_dir)
  assert list(out_dir.glob(".tmp_*")) == []


# ---------------------------------------------------------------------------
# Embedded lookup values
# ---------------------------------------------------------------------------

def test_embedded_probabilities(tmp_path: Path) -> None:
  results_dir, config_dir = _setup(tmp_path)
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  assert "var float prob_tf15_green = 0.75" in pine
  assert "var float prob_tf15_red   = 0.6" in pine
  assert "var float prob_tf30_green = 0.8" in pine
  assert "var float prob_tf60_green = 0.7" in pine


def test_embedded_sample_sizes(tmp_path: Path) -> None:
  results_dir, config_dir = _setup(tmp_path)
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  assert "var int   n_tf15_green    = 2000" in pine
  assert "var int   n_tf15_red      = 1500" in pine
  assert "var int   n_tf30_green    = 2100" in pine
  assert "var int   n_tf60_red      = 1300" in pine


# ---------------------------------------------------------------------------
# Pine structure & chart-TF agnosticism
# ---------------------------------------------------------------------------

def test_version_and_indicator_header(tmp_path: Path) -> None:
  """The indicator title is generic ("Augur Terminal"), not instrument-specific."""
  results_dir, config_dir = _setup(tmp_path)
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  assert "//@version=6" in pine
  assert 'indicator("Augur Terminal", overlay = true)' in pine
  assert 'indicator("Augur — NQ"' not in pine


def test_request_security_timeframes(tmp_path: Path) -> None:
  """Each opening candle is requested on its own timeframe via the capture
  function (not a direct [open, close] read, which returns the wrong candle on
  charts coarser than the requested timeframe — see issue #4)."""
  results_dir, config_dir = _setup(tmp_path)
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  assert 'request.security(syminfo.tickerid, "15", openingCandle()' in pine
  assert 'request.security(syminfo.tickerid, "30", openingCandle()' in pine
  assert 'request.security(syminfo.tickerid, "60", openingCandle()' in pine
  # The naive direct read is the bug we fixed; it must not appear.
  assert "[open, close]" not in pine


def test_opening_candle_capture_logic(tmp_path: Path) -> None:
  """The template must latch the RTH opening candle into var state inside the
  requested-timeframe context, rather than reading request.security directly.

  This is the fix for issue #4: a direct read returns the last closed candle of
  the requested timeframe, which on a chart coarser than that timeframe is not
  the opening candle. The capture function holds the first candle of the session.
  """
  results_dir, config_dir = _setup(tmp_path)
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  # Capture function defined.
  assert "openingCandle() =>" in pine
  # Persistent var storage for the latched open/close.
  assert "var float capturedOpen = na" in pine
  assert "var float capturedClose = na" in pine
  # The opening candle is the one whose open lands on the RTH session start.
  assert "candleMod == rthStartMod" in pine
  # Latching assignments, held for the whole session.
  assert "capturedOpen := open" in pine
  assert "capturedClose := close" in pine
  # The capture is wired into request.security for every timeframe.
  assert pine.count("openingCandle()") == 1 + 3  # 1 definition call + 3 security calls


def test_direction_from_security_not_chart(tmp_path: Path) -> None:
  """Direction must compare request.security open/close, not chart bars."""
  results_dir, config_dir = _setup(tmp_path)
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  assert "dir_tf15 = c_tf15 >= o_tf15" in pine
  assert "dir_tf30 = c_tf30 >= o_tf30" in pine
  assert "dir_tf60 = c_tf60 >= o_tf60" in pine


# ---------------------------------------------------------------------------
# Config-derived detection times & timezone (nothing hardcoded)
# ---------------------------------------------------------------------------

def test_detection_times_from_rth_start_930(tmp_path: Path) -> None:
  """rth start 09:30 (570) -> closes at 585/600/630, labels 09:45/10:00/10:30."""
  results_dir, config_dir = _setup(tmp_path, rth_start="09:30")
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  assert "rthStartMod  = 570" in pine
  assert "is_tf15_close = nyMod >= 585 and nyModPrev < 585" in pine
  assert "is_tf30_close = nyMod >= 600 and nyModPrev < 600" in pine
  assert "is_tf60_close = nyMod >= 630 and nyModPrev < 630" in pine
  assert '"09:45"' in pine and '"10:00"' in pine and '"10:30"' in pine


def test_detection_times_shift_with_config(tmp_path: Path) -> None:
  """Changing RTH start to 08:00 (480) must shift every detection time."""
  results_dir, config_dir = _setup(tmp_path, rth_start="08:00")
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  assert "rthStartMod  = 480" in pine
  assert "is_tf15_close = nyMod >= 495 and nyModPrev < 495" in pine  # 08:15
  assert '"08:15"' in pine and '"08:30"' in pine and '"09:00"' in pine
  # The 09:30-derived times must NOT appear (proves not hardcoded)
  assert "585" not in pine


def test_timezone_from_config(tmp_path: Path) -> None:
  results_dir, config_dir = _setup(tmp_path, timezone="America/Chicago")
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  assert 'TZ = "America/Chicago"' in pine


def test_instrument_name_from_arg(tmp_path: Path) -> None:
  """The instrument identity (comment + guard) follows the CLI arg, not the title."""
  timeframes = {"15min": _tf_block(0.75, 2000, 0.60, 1500)}
  results_dir, config_dir = _setup(tmp_path, instrument="ES", timeframes=timeframes)
  pine = render_indicator("ES", results_dir=results_dir, config_dir=config_dir)
  assert 'indicator("Augur Terminal", overlay = true)' in pine
  assert "// Instrument: ES" in pine


# ---------------------------------------------------------------------------
# Table sizing & row ordering
# ---------------------------------------------------------------------------

def test_table_rows_match_timeframe_count(tmp_path: Path) -> None:
  """3 columns; rows = header + session-open + one per timeframe (3 tf -> 5)."""
  results_dir, config_dir = _setup(tmp_path)
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  assert "table.new(position.top_right, 3, 5, bgcolor = colBg" in pine
  # Clears start at row 1, never the header (row 0).
  assert "table.clear(augurTable, 0, 1, 2, 4)" in pine


def test_rows_ordered_by_close_time(tmp_path: Path) -> None:
  """Even if JSON lists timeframes out of order, rows are sorted by close time."""
  timeframes = {
    "1h": _tf_block(0.70, 2200, 0.55, 1300),
    "15min": _tf_block(0.75, 2000, 0.60, 1500),
    "30min": _tf_block(0.80, 2100, 0.65, 1400),
  }
  results_dir, config_dir = _setup(tmp_path, timeframes=timeframes)
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  # Row 0 = header, row 1 = session open, so 15min owns row 2 and 1h owns row 4.
  assert 'table.cell(augurTable, 0, 2, "09:45"' in pine
  assert 'table.cell(augurTable, 0, 4, "10:30"' in pine


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_missing_timeframe_skipped(tmp_path: Path) -> None:
  """A single-timeframe JSON yields a one-row table and no other tf vars."""
  timeframes = {"15min": _tf_block(0.75, 2000, 0.60, 1500)}
  results_dir, config_dir = _setup(tmp_path, timeframes=timeframes)
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  assert "prob_tf15_green" in pine
  assert "prob_tf30_green" not in pine
  assert "prob_tf60_green" not in pine
  # One timeframe row + the header and session-open rows.
  assert "table.new(position.top_right, 3, 3, bgcolor = colBg" in pine


def test_deterministic_output(tmp_path: Path) -> None:
  results_dir, config_dir = _setup(tmp_path)
  a = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  b = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  assert a == b


def test_missing_results_file_raises(tmp_path: Path) -> None:
  config_dir = tmp_path / "config"
  _write_config(config_dir, "NQ")
  with pytest.raises(FileNotFoundError):
    render_indicator("NQ", results_dir=tmp_path / "results", config_dir=config_dir)


def test_instrument_not_in_json_raises(tmp_path: Path) -> None:
  results_dir, config_dir = _setup(tmp_path, instrument="NQ")
  _write_config(config_dir, "ES")
  with pytest.raises(KeyError):
    render_indicator("ES", results_dir=results_dir, config_dir=config_dir)


def test_french_accents_literal_in_output(tmp_path: Path) -> None:
  """Generated Pine must carry literal accented French, not escapes."""
  results_dir, config_dir = _setup(tmp_path)
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  assert "haussière" in pine
  assert "baissière" in pine
  assert "\\u" not in pine


# ---------------------------------------------------------------------------
# Instrument guard (issue #4 follow-up — refuse to run on the wrong asset)
# ---------------------------------------------------------------------------

def test_instrument_guard_present(tmp_path: Path) -> None:
  """Indicator compares syminfo.root with the generator-embedded expected root."""
  results_dir, config_dir = _setup(tmp_path)
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  assert 'expectedRoot = "NQ"' in pine
  assert "instrumentOk = syminfo.root == expectedRoot" in pine
  # Normal rendering is gated on a match; the warning on a mismatch.
  assert "if instrumentOk" in pine
  assert "if not instrumentOk" in pine


def test_instrument_guard_uses_arg_symbol(tmp_path: Path) -> None:
  """The expected root is the CLI instrument, not hardcoded as NQ."""
  timeframes = {"15min": _tf_block(0.75, 2000, 0.60, 1500)}
  results_dir, config_dir = _setup(tmp_path, instrument="ES", timeframes=timeframes)
  pine = render_indicator("ES", results_dir=results_dir, config_dir=config_dir)
  assert 'expectedRoot = "ES"' in pine
  assert '"NQ"' not in pine


def test_instrument_mismatch_warning_bilingual(tmp_path: Path) -> None:
  """Both EN and FR mismatch messages are embedded behind the language switch."""
  results_dir, config_dir = _setup(tmp_path)
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  assert "No statistics available for this asset" in pine
  assert "Pas de statistiques disponibles pour cet asset" in pine


# ---------------------------------------------------------------------------
# Session-open line
# ---------------------------------------------------------------------------

def test_session_open_line(tmp_path: Path) -> None:
  """Row 1 (below the header) shows the RTH open time and bilingual session label."""
  results_dir, config_dir = _setup(tmp_path)  # rth start 09:30
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  assert 'table.cell(augurTable, 0, 1, "09:30"' in pine
  assert 'lang == "FR" ? "Ouverture New York" : "New York Open"' in pine
  # The open line is written under the existing isNewSession branch.
  assert "if isNewSession" in pine


def test_session_open_time_from_config(tmp_path: Path) -> None:
  """Changing rth.start shifts the open-line time — it is not hardcoded."""
  results_dir, config_dir = _setup(tmp_path, rth_start="08:00")
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  assert 'table.cell(augurTable, 0, 1, "08:00"' in pine


def test_probability_rows_shift_below_header_and_open(tmp_path: Path) -> None:
  """With the header at row 0 and open line at row 1, 15min lands on row 2."""
  results_dir, config_dir = _setup(tmp_path)
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  assert 'table.cell(augurTable, 0, 2, "09:45"' in pine


# ---------------------------------------------------------------------------
# Full i18n (language input drives every displayed label)
# ---------------------------------------------------------------------------

def test_language_input_present(tmp_path: Path) -> None:
  results_dir, config_dir = _setup(tmp_path)
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  assert 'lang = input.string("EN", "Language", options = ["EN", "FR"])' in pine


def test_direction_labels_bilingual(tmp_path: Path) -> None:
  results_dir, config_dir = _setup(tmp_path)
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  assert 'lang == "FR" ? "haussière" : "bullish"' in pine
  assert 'lang == "FR" ? "baissière" : "bearish"' in pine


def test_candle_label_bilingual(tmp_path: Path) -> None:
  results_dir, config_dir = _setup(tmp_path)
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  assert 'lang == "FR" ? "Bougie 15min" : "15min candle"' in pine
  assert 'lang == "FR" ? "Bougie 1h" : "1h candle"' in pine


def test_session_word_is_language_neutral(tmp_path: Path) -> None:
  """'Session' is identical in both languages, so it stays a plain literal."""
  results_dir, config_dir = _setup(tmp_path)
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  assert '"→ Session " + sessionWord' in pine


# ---------------------------------------------------------------------------
# Dark terminal theme
# ---------------------------------------------------------------------------

def test_dark_theme_palette(tmp_path: Path) -> None:
  """The GitHub-Dark palette is embedded and the table uses the dark background."""
  results_dir, config_dir = _setup(tmp_path)
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  assert "colBg = #0D1117" in pine
  assert "colHeaderBg = #161B22" in pine
  assert "colMuted = #8B949E" in pine
  assert "colAccent = #58A6FF" in pine
  assert "colBull = #3FB950" in pine
  assert "colBear = #F85149" in pine
  assert "colWarn = #D29922" in pine
  assert "bgcolor = colBg" in pine


def test_table_has_no_frame_or_border(tmp_path: Path) -> None:
  """No frame/border on the table, so an empty (cleared) table is invisible."""
  results_dir, config_dir = _setup(tmp_path)
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  assert "frame_width" not in pine
  assert "frame_color" not in pine
  assert "border_width" not in pine
  assert "border_color" not in pine


def test_data_cells_carry_dark_background(tmp_path: Path) -> None:
  """Every non-header cell sets bgcolor = colBg for a uniform dark look."""
  results_dir, config_dir = _setup(tmp_path)
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  assert "text_color = colMuted, bgcolor = colBg" in pine   # time cells
  assert "text_color = colAccent, bgcolor = colBg" in pine  # label cells
  assert "text_color = predColor, bgcolor = colBg" in pine  # prediction cells
  # Header keeps its own darker background.
  assert "text_color = colMuted, bgcolor = colHeaderBg" in pine


def test_header_row(tmp_path: Path) -> None:
  """A muted 'AUGUR TERMINAL' header sits on its own dark-grey background."""
  results_dir, config_dir = _setup(tmp_path)
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  assert 'table.cell(augurTable, 0, 0, "AUGUR TERMINAL", text_color = colMuted, bgcolor = colHeaderBg' in pine


def test_prediction_cell_colored_by_direction(tmp_path: Path) -> None:
  """The prediction cell (col 2) is green/red; the candle label (col 1) is blue."""
  results_dir, config_dir = _setup(tmp_path)
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  assert "predColor   = sessionBull ? colBull : colBear" in pine
  # Time in muted grey, factual candle label in accent blue, prediction colored.
  assert 'candleLbl + " " + candleWord, text_color = colAccent' in pine
  assert "text_color = predColor" in pine


# ---------------------------------------------------------------------------
# Progressive display: live RTH session only (cleared outside RTH)
# ---------------------------------------------------------------------------

def test_in_rth_helper_present(tmp_path: Path) -> None:
  results_dir, config_dir = _setup(tmp_path)
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  assert "inRTH = nyMod >= rthStartMod and nyMod < rthEndMod" in pine


def test_rth_end_mod_from_config(tmp_path: Path) -> None:
  """rthEndMod is derived from rth.end; 16:15 -> 975, 15:00 -> 900."""
  results_dir, config_dir = _setup(tmp_path)  # default end 16:15
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  assert "rthEndMod    = 975" in pine

  results_dir2, config_dir2 = _setup(tmp_path, rth_end="15:00")
  pine2 = render_indicator("NQ", results_dir=results_dir2, config_dir=config_dir2)
  assert "rthEndMod    = 900" in pine2


def test_cleared_outside_rth(tmp_path: Path) -> None:
  """Outside RTH the data rows are cleared (header at row 0 is preserved)."""
  results_dir, config_dir = _setup(tmp_path)
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  assert "if not inRTH" in pine
  assert "table.clear(augurTable, 0, 1, 2, 4)" in pine


def test_waiting_message_outside_rth(tmp_path: Path) -> None:
  """Outside RTH the terminal shows a bilingual waiting message under the header."""
  results_dir, config_dir = _setup(tmp_path)
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  assert "⏳ Waiting for New York Open" in pine
  assert "⏳ En attente de l'ouverture de New York" in pine
  # Written in row 1, muted grey, on the dark background, inside the not-inRTH branch.
  assert 'table.cell(augurTable, 0, 1, lang == "FR" ? "⏳' in pine


def test_warning_is_styled(tmp_path: Path) -> None:
  """Instrument-mismatch warning is an orange ⚠ line in row 1 (header sits above)."""
  results_dir, config_dir = _setup(tmp_path)
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  assert "⚠ No statistics available for this asset" in pine
  assert "⚠ Pas de statistiques disponibles pour cet asset" in pine
  assert 'table.cell(augurTable, 0, 1, lang == "FR" ? "⚠' in pine
  assert "text_color = colWarn" in pine


def test_header_written_unconditionally(tmp_path: Path) -> None:
  """The header is printed before the instrument/RTH branches, so it is always
  visible (RTH, waiting, and warning states) and never cleared."""
  results_dir, config_dir = _setup(tmp_path)
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  header_idx = pine.index('"AUGUR TERMINAL"')
  guard_idx = pine.index("if not instrumentOk")
  assert header_idx < guard_idx


def test_progressive_guard_and_flags(tmp_path: Path) -> None:
  """Each row prints once per session, guarded by a var bool reset at open."""
  results_dir, config_dir = _setup(tmp_path)
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  assert "var bool shown_tf15 = false" in pine
  assert "if is_tf15_close and not shown_tf15 and inRTH" in pine
  assert "shown_tf15 := true" in pine
  # Flags are reset to false at the start of each RTH session.
  assert "shown_tf15 := false" in pine


# ---------------------------------------------------------------------------
# Most-likely outcome (continuation vs reversal) and dropped N column
# ---------------------------------------------------------------------------

def test_all_four_outcome_probabilities_embedded(tmp_path: Path) -> None:
  """All four conditional probabilities are embedded, not just the two of
  continuation — needed to pick the most likely session direction."""
  results_dir, config_dir = _setup(tmp_path)
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  # _tf_block(0.75, .., 0.60, ..) -> gg=0.75, gr=0.25, rg=0.40, rr=0.60.
  assert "var float prob_tf15_green = 0.75" in pine        # gg (continuation, green)
  assert "var float prob_tf15_green_rev = 0.25" in pine    # gr (reversal, green)
  assert "var float prob_tf15_red_rev   = 0.4" in pine     # rg (reversal, red)
  assert "var float prob_tf15_red   = 0.6" in pine         # rr (continuation, red)


def test_most_likely_outcome_logic(tmp_path: Path) -> None:
  """The displayed prediction is the higher of bullish/bearish probability."""
  results_dir, config_dir = _setup(tmp_path)
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  assert "pBull       = dir ? prob_tf15_green : prob_tf15_red_rev" in pine
  assert "pBear       = dir ? prob_tf15_green_rev : prob_tf15_red" in pine
  assert "sessionBull = pBull >= pBear" in pine
  assert "predProb    = sessionBull ? pBull : pBear" in pine


def test_sample_size_column_removed(tmp_path: Path) -> None:
  """N is no longer displayed (no (N=...) cell), though it stays embedded."""
  results_dir, config_dir = _setup(tmp_path)
  pine = render_indicator("NQ", results_dir=results_dir, config_dir=config_dir)
  assert "(N=" not in pine
  assert "str.tostring(n)" not in pine
  # The sample sizes remain embedded as constants for future use.
  assert "var int   n_tf15_green    = 2000" in pine
