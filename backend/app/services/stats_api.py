"""Pure-function helpers for the public stats API.

Business logic for narrowing and filtering StatRunResult objects.
These functions operate on already-loaded objects from the StatsLoader cache
and never touch the filesystem.

Note: these endpoints will be protected by X-API-Key auth once issue #113 lands.
"""

from stats.base import StatRunResult, TimeframeResult


def is_declared_slice(result: StatRunResult, slice_name: str) -> bool:
  """Return True if slice_name is declared in the family's labels.dimensions."""
  return slice_name in result.labels.dimensions


def narrow_to_instrument(result: StatRunResult, instrument: str) -> StatRunResult | None:
  """Return a copy of result containing only the given instrument, or None if absent.

  Args:
    result: The full StatRunResult from the loader cache.
    instrument: Instrument key to keep (e.g. "NQ").

  Returns:
    A copy of result with instruments narrowed to the requested key, or None
    if the instrument is not present in this family. Only the kept instrument's
    timeframes are deep-copied; the dropped instruments are never copied.
  """
  tf_map = result.instruments.get(instrument)
  if tf_map is None:
    return None
  copied_tf_map = {tf_key: tf.model_copy(deep=True) for tf_key, tf in tf_map.items()}
  return result.model_copy(update={"instruments": {instrument: copied_tf_map}})


def narrow_to_timeframe(
  result: StatRunResult,
  instrument: str,
  timeframe: str,
) -> TimeframeResult | None:
  """Return a copy of the TimeframeResult for a specific instrument/timeframe, or None.

  Args:
    result: The full StatRunResult from the loader cache.
    instrument: Instrument key (e.g. "NQ").
    timeframe: Timeframe key (e.g. "5m").

  Returns:
    A deep copy of the matching TimeframeResult, or None if either the
    instrument or the timeframe is not found.
  """
  tf_map = result.instruments.get(instrument)
  if tf_map is None:
    return None
  tf = tf_map.get(timeframe)
  if tf is None:
    return None
  return tf.model_copy(deep=True)


def apply_slice_tf(tf: TimeframeResult, slice_name: str) -> TimeframeResult:
  """Return a copy of tf with slices narrowed to slice_name only.

  If tf has no entry for slice_name, the returned copy has slices = {}.
  The top-level results list is kept intact.

  Args:
    tf: A TimeframeResult (may be a deep copy or the cached original).
    slice_name: The single slice dimension to keep.

  Returns:
    A deep copy of tf with .slices containing at most one key.
  """
  copy = tf.model_copy(deep=True)
  if slice_name in copy.slices:
    copy.slices = {slice_name: copy.slices[slice_name]}
  else:
    copy.slices = {}
  return copy


def apply_slice(result: StatRunResult, slice_name: str) -> StatRunResult:
  """Return a copy of result with every TimeframeResult narrowed to slice_name only.

  The top-level results lists inside each TimeframeResult are kept intact.
  If a particular timeframe has no data for slice_name, its slices become {}.

  Args:
    result: The StatRunResult to filter (a deep copy is made internally).
    slice_name: The single slice dimension to keep across all timeframes.

  Returns:
    A copy of result with all TimeframeResult.slices narrowed. Each timeframe is
    deep-copied exactly once (by apply_slice_tf); no intermediate tree copy.
  """
  new_instruments: dict[str, dict[str, TimeframeResult]] = {
    instr: {tf_key: apply_slice_tf(tf, slice_name) for tf_key, tf in tf_map.items()}
    for instr, tf_map in result.instruments.items()
  }
  return result.model_copy(update={"instruments": new_instruments})
