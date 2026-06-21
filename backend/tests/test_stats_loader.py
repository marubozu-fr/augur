"""Tests for StatsLoader — in-memory cache of stat result JSON files.

All tests use synthetic JSON files written to pytest's tmp_path.
No real results/ directory is required.
"""

import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.core.stats_loader import StatFamilyMeta, StatsLoader
from backend.app.main import create_app
from stats.base import (
  I18nString,
  Labels,
  StatResultRow,
  StatRunResult,
  TimeframeResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result_row(condition: str = "cond_a", outcome: str = "out_x") -> StatResultRow:
  """Build a minimal valid StatResultRow."""
  return StatResultRow(
    condition=condition,
    outcome=outcome,
    count=40,
    total=100,
    probability=0.4,
    baseline_prob=0.38,
    baseline_n=100,
  )


def _make_stat_run_result(
  stat_name: str,
  instrument: str = "NQ",
  timeframe: str = "1h",
  total_samples: int = 100,
  data_range: list[str] | None = None,
) -> StatRunResult:
  """Build a minimal valid StatRunResult via Pydantic models."""
  if data_range is None:
    data_range = ["2023-01-02", "2023-12-29"]
  tf_result = TimeframeResult(
    data_range=data_range,
    total_samples=total_samples,
    results=[_make_result_row()],
  )
  return StatRunResult(
    stat_name=stat_name,
    title=I18nString(en=f"{stat_name} title", fr=f"{stat_name} titre"),
    definition=I18nString(en=f"{stat_name} def", fr=f"{stat_name} déf"),
    labels=Labels(
      conditions={"cond_a": I18nString(en="Condition A", fr="Condition A")},
      outcomes={"out_x": I18nString(en="Outcome X", fr="Résultat X")},
    ),
    instruments={instrument: {timeframe: tf_result}},
  )


def _write_result(result: StatRunResult, directory: Path) -> Path:
  """Serialize a StatRunResult to <directory>/<stat_name>.json."""
  path = directory / f"{result.stat_name}.json"
  path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
  return path


# ---------------------------------------------------------------------------
# scan()
# ---------------------------------------------------------------------------

def test_scan_returns_sorted_json_files(tmp_path: Path) -> None:
  """scan() returns all .json files sorted, in a non-empty directory."""
  (tmp_path / "beta.json").write_text("{}", encoding="utf-8")
  (tmp_path / "alpha.json").write_text("{}", encoding="utf-8")
  (tmp_path / "gamma.json").write_text("{}", encoding="utf-8")

  loader = StatsLoader(results_dir=tmp_path)
  paths = loader.scan()

  assert [p.name for p in paths] == ["alpha.json", "beta.json", "gamma.json"]


def test_scan_excludes_tmp_prefix_files(tmp_path: Path) -> None:
  """scan() excludes files whose name starts with .tmp_."""
  (tmp_path / "real.json").write_text("{}", encoding="utf-8")
  (tmp_path / ".tmp_real.json").write_text("{}", encoding="utf-8")
  (tmp_path / ".tmp_other.json").write_text("{}", encoding="utf-8")

  loader = StatsLoader(results_dir=tmp_path)
  paths = loader.scan()

  names = [p.name for p in paths]
  assert names == ["real.json"]
  assert ".tmp_real.json" not in names
  assert ".tmp_other.json" not in names


def test_scan_returns_empty_list_when_dir_missing(tmp_path: Path) -> None:
  """scan() returns [] when the results directory does not exist."""
  missing = tmp_path / "nonexistent"
  loader = StatsLoader(results_dir=missing)

  assert loader.scan() == []


def test_scan_returns_empty_list_for_empty_dir(tmp_path: Path) -> None:
  """scan() returns [] when the directory exists but has no JSON files."""
  loader = StatsLoader(results_dir=tmp_path)

  assert loader.scan() == []


# ---------------------------------------------------------------------------
# load_all() + get()
# ---------------------------------------------------------------------------

def test_load_all_and_get_returns_validated_result(tmp_path: Path) -> None:
  """load_all() + get(family) returns the StatRunResult for a valid file."""
  result = _make_stat_run_result("opening_candle_continuation")
  _write_result(result, tmp_path)

  loader = StatsLoader(results_dir=tmp_path)
  loader.load_all()

  loaded = loader.get("opening_candle_continuation")
  assert loaded is not None
  assert isinstance(loaded, StatRunResult)
  assert loaded.stat_name == "opening_candle_continuation"


def test_family_key_equals_filename_stem(tmp_path: Path) -> None:
  """The family key in the cache equals the JSON filename stem."""
  result = _make_stat_run_result("my_stat")
  _write_result(result, tmp_path)

  loader = StatsLoader(results_dir=tmp_path)
  loader.load_all()

  # Key must be the stem, not the full filename or the stat_name field.
  assert loader.get("my_stat") is not None
  assert loader.get("my_stat.json") is None


def test_get_returns_none_for_unknown_family(tmp_path: Path) -> None:
  """get() returns None when the family has not been loaded."""
  loader = StatsLoader(results_dir=tmp_path)
  loader.load_all()

  assert loader.get("does_not_exist") is None


def test_get_returns_none_before_load_all(tmp_path: Path) -> None:
  """get() returns None when load_all() has not been called yet."""
  result = _make_stat_run_result("some_stat")
  _write_result(result, tmp_path)

  loader = StatsLoader(results_dir=tmp_path)
  # Deliberately NOT calling load_all()
  assert loader.get("some_stat") is None


# ---------------------------------------------------------------------------
# Corrupt / invalid file handling
# ---------------------------------------------------------------------------

def test_load_all_skips_malformed_json_file(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
  """load_all() skips a file with malformed JSON and logs a warning."""
  good = _make_stat_run_result("good_stat")
  _write_result(good, tmp_path)
  (tmp_path / "bad_json.json").write_text("{ not valid json !!!", encoding="utf-8")

  loader = StatsLoader(results_dir=tmp_path)
  with caplog.at_level(logging.WARNING, logger="backend.app.core.stats_loader"):
    loader.load_all()

  # Good file is loaded; bad file is skipped without raising.
  assert loader.get("good_stat") is not None
  assert loader.get("bad_json") is None
  assert any("bad_json.json" in record.message for record in caplog.records)


def test_load_all_skips_invalid_schema_file(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
  """load_all() skips a valid-JSON file that fails StatRunResult validation."""
  good = _make_stat_run_result("good_stat")
  _write_result(good, tmp_path)
  # Valid JSON, but missing every required StatRunResult field.
  (tmp_path / "invalid_schema.json").write_text(
    '{"stat_name": "x"}',
    encoding="utf-8",
  )

  loader = StatsLoader(results_dir=tmp_path)
  with caplog.at_level(logging.WARNING, logger="backend.app.core.stats_loader"):
    loader.load_all()

  assert loader.get("good_stat") is not None
  assert loader.get("invalid_schema") is None
  assert any("invalid_schema.json" in record.message for record in caplog.records)


def test_load_all_does_not_raise_on_all_bad_files(tmp_path: Path) -> None:
  """load_all() does not raise even when every file in the dir is corrupt."""
  (tmp_path / "corrupt1.json").write_text("NOPE", encoding="utf-8")
  (tmp_path / "corrupt2.json").write_text("[]", encoding="utf-8")

  loader = StatsLoader(results_dir=tmp_path)
  loader.load_all()  # Must not raise

  assert loader.get("corrupt1") is None
  assert loader.get("corrupt2") is None


# ---------------------------------------------------------------------------
# list_families()
# ---------------------------------------------------------------------------

def test_list_families_returns_stat_family_meta_instances(tmp_path: Path) -> None:
  """list_families() returns a list of StatFamilyMeta objects."""
  result = _make_stat_run_result("alpha_stat")
  _write_result(result, tmp_path)

  loader = StatsLoader(results_dir=tmp_path)
  loader.load_all()
  metas = loader.list_families()

  assert len(metas) == 1
  assert isinstance(metas[0], StatFamilyMeta)


def test_list_families_correct_family_and_title(tmp_path: Path) -> None:
  """list_families() reports the correct family name and title."""
  # Expected title: I18nString(en="alpha_stat title", fr="alpha_stat titre")
  result = _make_stat_run_result("alpha_stat")
  _write_result(result, tmp_path)

  loader = StatsLoader(results_dir=tmp_path)
  loader.load_all()
  meta = loader.list_families()[0]

  assert meta.family == "alpha_stat"
  assert meta.title.en == "alpha_stat title"
  assert meta.title.fr == "alpha_stat titre"
  assert meta.definition.en == "alpha_stat def"
  assert meta.definition.fr == "alpha_stat déf"


def test_list_families_correct_timeframe_metadata(tmp_path: Path) -> None:
  """list_families() reports instrument, timeframe, data_range, total_samples per entry."""
  # Synthetic: NQ / 15m, 250 samples, data_range 2023-01-02 to 2023-12-29
  result = _make_stat_run_result(
    "my_stat",
    instrument="NQ",
    timeframe="15m",
    total_samples=250,
    data_range=["2023-01-02", "2023-12-29"],
  )
  _write_result(result, tmp_path)

  loader = StatsLoader(results_dir=tmp_path)
  loader.load_all()
  meta = loader.list_families()[0]

  assert len(meta.timeframes) == 1
  tf = meta.timeframes[0]
  assert tf.instrument == "NQ"
  assert tf.timeframe == "15m"
  assert tf.data_range == ["2023-01-02", "2023-12-29"]
  assert tf.total_samples == 250


def test_list_families_no_results_arrays(tmp_path: Path) -> None:
  """list_families() metadata has no results arrays (StatFamilyMeta has no such field)."""
  result = _make_stat_run_result("stat_x")
  _write_result(result, tmp_path)

  loader = StatsLoader(results_dir=tmp_path)
  loader.load_all()
  meta = loader.list_families()[0]

  assert not hasattr(meta, "results")
  assert not hasattr(meta.timeframes[0], "results")


def test_list_families_sorted_by_family_name(tmp_path: Path) -> None:
  """list_families() returns families sorted alphabetically by family name."""
  for name in ("zebra_stat", "alpha_stat", "middle_stat"):
    _write_result(_make_stat_run_result(name), tmp_path)

  loader = StatsLoader(results_dir=tmp_path)
  loader.load_all()
  names = [m.family for m in loader.list_families()]

  assert names == sorted(names)


def test_list_families_returns_empty_before_load(tmp_path: Path) -> None:
  """list_families() returns [] when load_all() has not been called."""
  _write_result(_make_stat_run_result("stat_a"), tmp_path)

  loader = StatsLoader(results_dir=tmp_path)
  # Deliberately NOT calling load_all()
  assert loader.list_families() == []


def test_list_families_multiple_timeframes_per_instrument(tmp_path: Path) -> None:
  """list_families() flattens multiple instrument/timeframe entries correctly."""
  # Build a result with NQ/1h and NQ/15m inside a single StatRunResult.
  tf_1h = TimeframeResult(
    data_range=["2023-01-02", "2023-12-29"],
    total_samples=120,
    results=[_make_result_row()],
  )
  tf_15m = TimeframeResult(
    data_range=["2023-01-02", "2023-12-29"],
    total_samples=600,
    results=[_make_result_row()],
  )
  result = StatRunResult(
    stat_name="multi_tf_stat",
    title=I18nString(en="Multi TF", fr="Multi TF"),
    definition=I18nString(en="def", fr="déf"),
    labels=Labels(
      conditions={"cond_a": I18nString(en="C", fr="C")},
      outcomes={"out_x": I18nString(en="O", fr="O")},
    ),
    instruments={"NQ": {"1h": tf_1h, "15m": tf_15m}},
  )
  _write_result(result, tmp_path)

  loader = StatsLoader(results_dir=tmp_path)
  loader.load_all()
  meta = loader.list_families()[0]

  assert len(meta.timeframes) == 2
  tf_keys = {(t.instrument, t.timeframe) for t in meta.timeframes}
  assert tf_keys == {("NQ", "1h"), ("NQ", "15m")}


# ---------------------------------------------------------------------------
# reload()
# ---------------------------------------------------------------------------

def test_reload_picks_up_new_file(tmp_path: Path) -> None:
  """reload() reflects a new file added after the first load_all()."""
  _write_result(_make_stat_run_result("first_stat"), tmp_path)

  loader = StatsLoader(results_dir=tmp_path)
  loader.load_all()
  assert loader.get("first_stat") is not None
  assert loader.get("second_stat") is None

  # Add a second file and reload.
  _write_result(_make_stat_run_result("second_stat"), tmp_path)
  loader.reload()

  assert loader.get("first_stat") is not None
  assert loader.get("second_stat") is not None


def test_reload_is_idempotent(tmp_path: Path) -> None:
  """Calling reload() twice yields the same family set."""
  _write_result(_make_stat_run_result("stat_a"), tmp_path)
  _write_result(_make_stat_run_result("stat_b"), tmp_path)

  loader = StatsLoader(results_dir=tmp_path)
  loader.reload()
  families_after_first = {m.family for m in loader.list_families()}

  loader.reload()
  families_after_second = {m.family for m in loader.list_families()}

  assert families_after_first == families_after_second


def test_reload_removes_deleted_file(tmp_path: Path) -> None:
  """reload() drops a family whose file was removed from the directory."""
  path_a = _write_result(_make_stat_run_result("stat_a"), tmp_path)
  _write_result(_make_stat_run_result("stat_b"), tmp_path)

  loader = StatsLoader(results_dir=tmp_path)
  loader.load_all()
  assert loader.get("stat_a") is not None

  path_a.unlink()
  loader.reload()

  assert loader.get("stat_a") is None
  assert loader.get("stat_b") is not None


# ---------------------------------------------------------------------------
# Lifespan wiring
# ---------------------------------------------------------------------------

def test_lifespan_populates_app_state_stats_loader() -> None:
  """Startup lifespan sets app.state.stats_loader to a StatsLoader instance."""
  with TestClient(create_app()) as client:
    loader = client.app.state.stats_loader
    assert isinstance(loader, StatsLoader)


def test_lifespan_stats_loader_families_is_list() -> None:
  """After startup, list_families() returns a list (may be empty if results/ absent)."""
  with TestClient(create_app()) as client:
    loader = client.app.state.stats_loader
    families = loader.list_families()
    assert isinstance(families, list)
    # Each element, if any, must be a StatFamilyMeta.
    for meta in families:
      assert isinstance(meta, StatFamilyMeta)
