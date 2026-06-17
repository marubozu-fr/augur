"""Tests for the declarative slice architecture in stats/base.py.

All data is synthetic — no real market files required.
Day-tables are constructed directly as pandas DataFrames with a
tz-aware DatetimeIndex (America/New_York).
Expected values are hand-calculated before each assertion.

Covers:
  - Weekday slicer
  - Close / PrevCandle slicers (_ColorSlicer subclasses)
  - SizeBucket slicer (preset and explicit buckets)
  - Levels slicer
  - resolve_slicer()
  - Pydantic model round-trip for slices via write_results()
  - Framework integration: OpeningCandleContinuation weekday slices
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from stats.base import (
  Close,
  I18nString,
  Labels,
  Levels,
  PrevCandle,
  SliceGroup,
  SliceGroupResult,
  SliceResult,
  SizeBucket,
  StatResultRow,
  StatRunResult,
  TimeframeResult,
  Weekday,
  resolve_slicer,
  write_results,
)
from stats.config import InstrumentConfig, Session
from stats.opening_candle.continuation import OpeningCandleContinuation

# ---------------------------------------------------------------------------
# Shared test infrastructure
# ---------------------------------------------------------------------------

_NY = "America/New_York"


def _make_idx(dates: list[str]) -> pd.DatetimeIndex:
  """Build a normalized, tz-aware DatetimeIndex from a list of YYYY-MM-DD strings."""
  return pd.to_datetime(dates).tz_localize(_NY).normalize()


def _make_day_table(dates: list[str], **columns: list) -> pd.DataFrame:
  """Build a minimal day-table DataFrame.

  Args:
    dates: list of YYYY-MM-DD strings used as the index.
    **columns: keyword args whose values are parallel lists of column values.

  Returns a DataFrame with a tz-aware normalized DatetimeIndex.
  """
  idx = _make_idx(dates)
  return pd.DataFrame(columns, index=idx)


def _groups_by_key(groups: list[SliceGroup]) -> dict[str, SliceGroup]:
  return {g.key: g for g in groups}


# ---------------------------------------------------------------------------
# Minimal InstrumentConfig (mirrors pattern in test_opening_candle_continuation)
# ---------------------------------------------------------------------------

_RTH_SESSION = Session(start="09:30", end="16:15")
_TEST_CONFIG = InstrumentConfig(
  instrument="NQ",
  working_timezone=_NY,
  sessions={"rth": _RTH_SESSION},
  timeframes=["15min", "30min", "1h"],
  parquet_path=Path("data/NQ_1min.parquet"),
)


# ===========================================================================
# 1. Weekday slicer
# ===========================================================================

class TestWeekday:
  """Tests for stats.base.Weekday."""

  def test_name(self) -> None:
    """Weekday.name must be 'weekday'."""
    assert Weekday.name == "weekday"

  def test_dimension_label_en_fr(self) -> None:
    """dimension_label() must return an I18nString with non-empty en and fr."""
    label = Weekday().dimension_label()
    assert label.en != ""
    assert label.fr != ""

  def test_empty_day_table_returns_empty(self) -> None:
    """Empty day-table must produce no groups."""
    dt = pd.DataFrame(index=pd.DatetimeIndex([], tz=_NY))
    result = Weekday().split(dt)
    assert result == []

  def test_five_days_one_per_weekday(self) -> None:
    """Mon–Fri week produces exactly 5 groups in Monday→Friday order.

    Hand-calculation:
      2024-01-01 Monday    dayofweek=0 → key='monday'
      2024-01-02 Tuesday   dayofweek=1 → key='tuesday'
      2024-01-03 Wednesday dayofweek=2 → key='wednesday'
      2024-01-04 Thursday  dayofweek=3 → key='thursday'
      2024-01-05 Friday    dayofweek=4 → key='friday'
    Each group mask selects exactly 1 row.
    """
    dt = _make_day_table(
      ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
      value=[1, 2, 3, 4, 5],
    )
    groups = Weekday().split(dt)

    assert len(groups) == 5
    expected_keys = ["monday", "tuesday", "wednesday", "thursday", "friday"]
    assert [g.key for g in groups] == expected_keys

    # Each group mask must select exactly 1 row
    for g in groups:
      assert int(g.mask.sum()) == 1, f"Group '{g.key}' should have mask.sum()==1"

  def test_mask_alignment_to_index(self) -> None:
    """The mask for each weekday selects only rows on that weekday.

    Hand-calculation:
      2024-01-01 Monday   → monday mask  = [True, False, False, False, False]
      2024-01-02 Tuesday  → tuesday mask = [False, True, False, False, False]
      etc.
    """
    dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    dt = _make_day_table(dates, value=[10, 20, 30, 40, 50])
    groups = Weekday().split(dt)
    by_key = _groups_by_key(groups)

    # Monday mask: only 2024-01-01 is True
    assert list(by_key["monday"].mask) == [True, False, False, False, False]
    # Friday mask: only 2024-01-05 is True
    assert list(by_key["friday"].mask) == [False, False, False, False, True]

  def test_absent_weekdays_produce_no_group(self) -> None:
    """Weekdays with no data produce no group; Sat/Sun never appear.

    Hand-calculation: only Tuesday and Thursday → 2 groups, no Mon/Wed/Fri/Sat/Sun.
    """
    dt = _make_day_table(
      ["2024-01-02", "2024-01-04"],  # Tuesday=1, Thursday=3
      value=[1, 2],
    )
    groups = Weekday().split(dt)

    assert len(groups) == 2
    assert [g.key for g in groups] == ["tuesday", "thursday"]

  def test_monday_appears_first_even_if_added_last(self) -> None:
    """Groups are always in Monday→Friday order regardless of date input order.

    Hand-calculation: Friday=4 data first, Monday=0 data second → groups ordered [monday, friday].
    """
    dt = _make_day_table(
      ["2024-01-05", "2024-01-08"],  # Friday, then Monday
      value=[1, 2],
    )
    groups = Weekday().split(dt)

    assert [g.key for g in groups] == ["monday", "friday"]

  def test_multiple_weeks_group_counts(self) -> None:
    """7 days spanning two weeks → correct per-weekday counts.

    Hand-calculation for dates:
      2024-01-01 Mon, 2024-01-02 Tue, 2024-01-03 Wed, 2024-01-04 Thu, 2024-01-05 Fri,
      2024-01-08 Mon, 2024-01-09 Tue
    monday=2, tuesday=2, wednesday=1, thursday=1, friday=1
    """
    dates = [
      "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
      "2024-01-08", "2024-01-09",
    ]
    dt = _make_day_table(dates, value=list(range(7)))
    groups = Weekday().split(dt)

    by_key = _groups_by_key(groups)
    assert int(by_key["monday"].mask.sum()) == 2
    assert int(by_key["tuesday"].mask.sum()) == 2
    assert int(by_key["wednesday"].mask.sum()) == 1
    assert int(by_key["thursday"].mask.sum()) == 1
    assert int(by_key["friday"].mask.sum()) == 1

  def test_weekday_labels_have_en_and_fr(self) -> None:
    """Each group label must have non-empty en and fr strings."""
    dt = _make_day_table(["2024-01-01", "2024-01-02"], value=[1, 2])  # Mon, Tue
    groups = Weekday().split(dt)

    for g in groups:
      assert g.label.en != "", f"Group {g.key} has empty en label"
      assert g.label.fr != "", f"Group {g.key} has empty fr label"

  def test_saturday_never_appears(self) -> None:
    """Saturday (dayofweek=5) never appears even if a date falls on a Saturday.

    This is a guard test: the slicer only exposes Mon-Sun from _WEEKDAYS but Saturday
    is included in that table, so we verify no group has key='saturday' for a typical
    Mon-Fri dataset.
    """
    dt = _make_day_table(
      ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
      value=[1, 2, 3, 4, 5],
    )
    groups = Weekday().split(dt)
    keys = [g.key for g in groups]
    assert "saturday" not in keys
    assert "sunday" not in keys


# ===========================================================================
# 2. Close slicer (_ColorSlicer subclass)
# ===========================================================================

class TestClose:
  """Tests for stats.base.Close."""

  def test_name(self) -> None:
    """Close.name must be 'close'."""
    assert Close().name == "close"

  def test_dimension_label_en_fr(self) -> None:
    """dimension_label() must return an I18nString with non-empty en and fr."""
    label = Close().dimension_label()
    assert label.en != ""
    assert label.fr != ""

  def test_empty_day_table_returns_empty(self) -> None:
    """Empty day-table must produce no groups."""
    dt = pd.DataFrame({"session_green": pd.Series([], dtype=bool)},
                      index=pd.DatetimeIndex([], tz=_NY))
    result = Close().split(dt)
    assert result == []

  def test_missing_column_returns_empty(self) -> None:
    """If the target column is absent, split() must return []."""
    dt = _make_day_table(["2024-01-02"], other_col=[True])
    result = Close().split(dt)
    assert result == []

  def test_two_groups_green_and_red(self) -> None:
    """3 green + 2 red → two groups with correct member counts.

    Hand-calculation:
      green group mask selects rows where session_green=True  → 3 rows
      red   group mask selects rows where session_green=False → 2 rows
    """
    dt = _make_day_table(
      ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
      session_green=[True, True, True, False, False],
    )
    groups = Close().split(dt)
    by_key = _groups_by_key(groups)

    assert set(by_key.keys()) == {"green", "red"}
    assert int(by_key["green"].mask.sum()) == 3
    assert int(by_key["red"].mask.sum()) == 2

  def test_nan_excluded_from_both_groups(self) -> None:
    """NaN values must not appear in either group.

    Hand-calculation:
      rows: [True, False, NaN, True, False]
      green mask = rows 0,3 (True only) → 2
      red   mask = rows 1,4 (False only) → 2
      row 2 (NaN) appears in neither
    """
    dt = _make_day_table(
      ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
      session_green=[True, False, None, True, False],
    )
    groups = Close().split(dt)
    by_key = _groups_by_key(groups)

    assert int(by_key["green"].mask.sum()) == 2
    assert int(by_key["red"].mask.sum()) == 2
    # Total covered rows must not exceed 4 (NaN row excluded)
    total_covered = int(by_key["green"].mask.sum()) + int(by_key["red"].mask.sum())
    assert total_covered == 4

  def test_all_green_omits_red_group(self) -> None:
    """When all values are True, no red group is emitted.

    Hand-calculation: 4 green rows → only 'green' group, no 'red' group.
    """
    dt = _make_day_table(
      ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
      session_green=[True, True, True, True],
    )
    groups = Close().split(dt)

    assert len(groups) == 1
    assert groups[0].key == "green"
    assert int(groups[0].mask.sum()) == 4

  def test_all_red_omits_green_group(self) -> None:
    """When all values are False, no green group is emitted.

    Hand-calculation: 3 red rows → only 'red' group.
    """
    dt = _make_day_table(
      ["2024-01-01", "2024-01-02", "2024-01-03"],
      session_green=[False, False, False],
    )
    groups = Close().split(dt)

    assert len(groups) == 1
    assert groups[0].key == "red"

  def test_custom_column_name(self) -> None:
    """Close(column='my_flag') must use the specified column."""
    dt = _make_day_table(
      ["2024-01-01", "2024-01-02"],
      my_flag=[True, False],
    )
    groups = Close(column="my_flag").split(dt)
    assert len(groups) == 2

  def test_green_label_en_and_fr(self) -> None:
    """The 'green' group label must have non-empty en and fr."""
    dt = _make_day_table(["2024-01-01"], session_green=[True])
    groups = Close().split(dt)
    assert groups[0].label.en != ""
    assert groups[0].label.fr != ""

  def test_mask_selects_correct_rows(self) -> None:
    """Green mask must be True only at positions where session_green=True.

    Hand-calculation for [True, False, True]:
      green mask = [True, False, True]
      red   mask = [False, True, False]
    """
    dt = _make_day_table(
      ["2024-01-01", "2024-01-02", "2024-01-03"],
      session_green=[True, False, True],
    )
    groups = Close().split(dt)
    by_key = _groups_by_key(groups)

    assert list(by_key["green"].mask) == [True, False, True]
    assert list(by_key["red"].mask) == [False, True, False]


# ===========================================================================
# 3. PrevCandle slicer (_ColorSlicer subclass)
# ===========================================================================

class TestPrevCandle:
  """Tests for stats.base.PrevCandle.

  PrevCandle shares all _ColorSlicer logic with Close, so tests focus on
  name, dimension_label, and default column differences.
  """

  def test_name(self) -> None:
    """PrevCandle.name must be 'prev_candle'."""
    assert PrevCandle().name == "prev_candle"

  def test_dimension_label_different_from_close(self) -> None:
    """PrevCandle.dimension_label() must differ from Close.dimension_label()."""
    pc_label = PrevCandle().dimension_label()
    close_label = Close().dimension_label()
    assert pc_label.en != close_label.en

  def test_default_column_prev_session_green(self) -> None:
    """PrevCandle default column is 'prev_session_green'.

    Hand-calculation: 2 rows, both green → 1 group 'green' with 2 members.
    """
    dt = _make_day_table(
      ["2024-01-02", "2024-01-03"],
      prev_session_green=[True, True],
    )
    groups = PrevCandle().split(dt)
    assert len(groups) == 1
    assert groups[0].key == "green"

  def test_missing_default_column_returns_empty(self) -> None:
    """Missing 'prev_session_green' column must return []."""
    dt = _make_day_table(["2024-01-02"], session_green=[True])
    result = PrevCandle().split(dt)
    assert result == []

  def test_nan_excluded(self) -> None:
    """NaN in prev_session_green excluded from both groups.

    Hand-calculation: [True, NaN, False] → green=1, red=1, NaN excluded.
    """
    dt = _make_day_table(
      ["2024-01-01", "2024-01-02", "2024-01-03"],
      prev_session_green=[True, None, False],
    )
    groups = PrevCandle().split(dt)
    by_key = _groups_by_key(groups)

    assert int(by_key["green"].mask.sum()) == 1
    assert int(by_key["red"].mask.sum()) == 1


# ===========================================================================
# 4. SizeBucket slicer
# ===========================================================================

class TestSizeBucket:
  """Tests for stats.base.SizeBucket."""

  def test_name(self) -> None:
    """SizeBucket.name defaults to 'size_bucket'."""
    assert SizeBucket(column="gap_size").name == "size_bucket"

  def test_dimension_label_en_fr(self) -> None:
    """dimension_label() must return non-empty en and fr."""
    label = SizeBucket(column="x").dimension_label()
    assert label.en != ""
    assert label.fr != ""

  def test_unknown_preset_raises_at_construction(self) -> None:
    """An unknown preset with no explicit buckets must raise ValueError at construction."""
    with pytest.raises(ValueError, match="Unknown preset"):
      SizeBucket(column="x", preset="deciles")

  def test_empty_day_table_returns_empty(self) -> None:
    """Empty day-table must return []."""
    dt = pd.DataFrame({"gap_size": pd.Series([], dtype=float)},
                      index=pd.DatetimeIndex([], tz=_NY))
    result = SizeBucket(column="gap_size").split(dt)
    assert result == []

  def test_missing_column_returns_empty(self) -> None:
    """Column absent from the day-table must return []."""
    dt = _make_day_table(["2024-01-02"], other=[1.0])
    result = SizeBucket(column="gap_size").split(dt)
    assert result == []

  def test_all_nan_returns_empty(self) -> None:
    """All-NaN column must return []."""
    dt = _make_day_table(
      ["2024-01-01", "2024-01-02"],
      gap_size=[float("nan"), float("nan")],
    )
    result = SizeBucket(column="gap_size").split(dt)
    assert result == []

  def test_quartiles_eight_distinct_values(self) -> None:
    """preset='quartiles' on 8 distinct values → 4 groups of 2 each.

    Hand-calculation with pd.qcut(q=4, duplicates='drop') on [1,2,3,4,5,6,7,8]:
      q=4 quantile edges ≈ [0.999, 2.75, 4.5, 6.25, 8.0]
      q1: values 1,2  (≤2.75)
      q2: values 3,4  (2.75–4.5]
      q3: values 5,6  (4.5–6.25]
      q4: values 7,8  (6.25–8.0]
    Keys are 'q1','q2','q3','q4' in ascending order.
    """
    dates = [
      "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04",
      "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10",
    ]
    dt = _make_day_table(dates, gap_size=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    groups = SizeBucket(column="gap_size", preset="quartiles").split(dt)

    assert len(groups) == 4
    assert [g.key for g in groups] == ["q1", "q2", "q3", "q4"]

    # Each group must have exactly 2 members
    for g in groups:
      assert int(g.mask.sum()) == 2, f"Group {g.key} expected 2 members, got {int(g.mask.sum())}"

  def test_quartiles_q1_selects_lowest_two_rows(self) -> None:
    """q1 mask must select the two rows with the smallest gap_size values.

    Hand-calculation: gap_size=[1,2,3,4,5,6,7,8], q1 covers values 1 and 2.
    Dates are in ascending order, so the first two rows match.
    """
    dates = [
      "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04",
      "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10",
    ]
    dt = _make_day_table(dates, gap_size=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    groups = SizeBucket(column="gap_size", preset="quartiles").split(dt)
    by_key = _groups_by_key(groups)

    q1_mask = by_key["q1"].mask
    # Rows 0 and 1 (gap_size 1.0 and 2.0) must be True; all others False
    assert list(q1_mask) == [True, True, False, False, False, False, False, False]

  def test_quartiles_q4_selects_highest_two_rows(self) -> None:
    """q4 mask must select the two rows with the largest gap_size values.

    Hand-calculation: gap_size=[1,2,3,4,5,6,7,8], q4 covers values 7 and 8.
    Rows 6 and 7 must be True.
    """
    dates = [
      "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04",
      "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10",
    ]
    dt = _make_day_table(dates, gap_size=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    groups = SizeBucket(column="gap_size", preset="quartiles").split(dt)
    by_key = _groups_by_key(groups)

    q4_mask = by_key["q4"].mask
    assert list(q4_mask) == [False, False, False, False, False, False, True, True]

  def test_explicit_buckets_three_bins(self) -> None:
    """buckets=[0,10,20,30] on values [5,15,25,5,15,25] → 3 groups, 2 members each.

    Hand-calculation with pd.cut(bins=[0,10,20,30], include_lowest=True):
      q1 (≤10):  values 5,5  → rows 0,3
      q2 (≤20):  values 15,15 → rows 1,4
      q3 (≤30):  values 25,25 → rows 2,5
    """
    dates = [
      "2024-01-01", "2024-01-02", "2024-01-03",
      "2024-01-04", "2024-01-05", "2024-01-08",
    ]
    dt = _make_day_table(dates, gap_size=[5.0, 15.0, 25.0, 5.0, 15.0, 25.0])
    groups = SizeBucket(column="gap_size", buckets=[0, 10, 20, 30]).split(dt)

    assert len(groups) == 3
    assert [g.key for g in groups] == ["q1", "q2", "q3"]

    by_key = _groups_by_key(groups)
    assert int(by_key["q1"].mask.sum()) == 2
    assert int(by_key["q2"].mask.sum()) == 2
    assert int(by_key["q3"].mask.sum()) == 2

  def test_explicit_buckets_q1_selects_correct_rows(self) -> None:
    """q1 mask (values ≤10) must select rows 0 and 3 (gap_size=5.0).

    Hand-calculation: [5,15,25,5,15,25] with bins=[0,10,20,30]
      row 0: 5 → bin (−0.001,10] → q1=True
      row 1: 15 → bin (10,20]   → q1=False
      row 2: 25 → bin (20,30]   → q1=False
      row 3: 5 → q1=True
      row 4: 15 → q1=False
      row 5: 25 → q1=False
    """
    dates = [
      "2024-01-01", "2024-01-02", "2024-01-03",
      "2024-01-04", "2024-01-05", "2024-01-08",
    ]
    dt = _make_day_table(dates, gap_size=[5.0, 15.0, 25.0, 5.0, 15.0, 25.0])
    groups = SizeBucket(column="gap_size", buckets=[0, 10, 20, 30]).split(dt)
    by_key = _groups_by_key(groups)

    assert list(by_key["q1"].mask) == [True, False, False, True, False, False]

  def test_all_identical_values_does_not_raise(self) -> None:
    """All-identical column must not raise; duplicates='drop' collapses bins → [].

    pd.qcut on [5,5,5,5] with q=4 and duplicates='drop' produces 0 categories
    because all quantile edges are identical. The slicer returns [] (no groups).
    """
    dt = _make_day_table(
      ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
      gap_size=[5.0, 5.0, 5.0, 5.0],
    )
    # Must not raise
    result = SizeBucket(column="gap_size", preset="quartiles").split(dt)
    # All categories collapse → empty or single group depending on pandas version
    # Either way: no exception and result is a list
    assert isinstance(result, list)

  def test_nan_rows_excluded_from_all_groups(self) -> None:
    """NaN values are excluded from all groups; numeric rows still bucketed.

    Hand-calculation: [1.0, NaN, 3.0, NaN, 5.0, NaN, 7.0, NaN] with quartiles
      Valid values: [1, 3, 5, 7] → 4 rows, quartile-split into 4 groups of 1 each.
      NaN rows (indices 1,3,5,7) must not appear in any group mask.
    """
    dates = [
      "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04",
      "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10",
    ]
    dt = _make_day_table(
      dates,
      gap_size=[1.0, float("nan"), 3.0, float("nan"), 5.0, float("nan"), 7.0, float("nan")],
    )
    groups = SizeBucket(column="gap_size", preset="quartiles").split(dt)

    # NaN rows must never appear as True in any group mask
    for g in groups:
      # Odd-indexed rows (NaN) must all be False in every group
      for i in [1, 3, 5, 7]:
        assert not g.mask.iloc[i], (
          f"Group '{g.key}' has NaN row {i} in its mask"
        )

  def test_group_labels_contain_range(self) -> None:
    """Group labels must contain the numeric range (e.g. '1.00–2.75').

    The SizeBucket label format is:
      en: '{column} bin {i} ({lo:.2f}–{hi:.2f}]'
    """
    dates = [
      "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04",
      "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10",
    ]
    dt = _make_day_table(dates, gap_size=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    groups = SizeBucket(column="gap_size", preset="quartiles").split(dt)

    for g in groups:
      # Each label must mention the column name
      assert "gap_size" in g.label.en, f"Label missing column name: {g.label.en!r}"
      # Each label must contain the '–' range separator
      assert "–" in g.label.en, f"Label missing range separator: {g.label.en!r}"

  def test_median_preset_two_groups(self) -> None:
    """preset='median' (2 bins) on 4 values → 2 groups of 2.

    Hand-calculation: [1,2,3,4] with q=2 → [1,2] in q1, [3,4] in q2.
    """
    dt = _make_day_table(
      ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
      gap_size=[1.0, 2.0, 3.0, 4.0],
    )
    groups = SizeBucket(column="gap_size", preset="median").split(dt)

    assert len(groups) == 2
    assert groups[0].key == "q1"
    assert groups[1].key == "q2"


# ===========================================================================
# 5. Levels slicer
# ===========================================================================

class TestLevels:
  """Tests for stats.base.Levels."""

  def test_name(self) -> None:
    """Levels.name defaults to 'levels'."""
    assert Levels(ref="orb", ext="ext").name == "levels"

  def test_dimension_label_en_fr(self) -> None:
    """dimension_label() must return non-empty en and fr."""
    label = Levels(ref="orb", ext="ext").dimension_label()
    assert label.en != ""
    assert label.fr != ""

  def test_empty_day_table_returns_empty(self) -> None:
    """Empty day-table returns []."""
    dt = pd.DataFrame(
      {"orb": pd.Series([], dtype=float), "ext": pd.Series([], dtype=float)},
      index=pd.DatetimeIndex([], tz=_NY),
    )
    result = Levels(ref="orb", ext="ext").split(dt)
    assert result == []

  def test_missing_ref_column_returns_empty(self) -> None:
    """Missing ref column must return []."""
    dt = _make_day_table(["2024-01-02"], ext=[10.0])
    result = Levels(ref="orb", ext="ext").split(dt)
    assert result == []

  def test_missing_ext_column_returns_empty(self) -> None:
    """Missing ext column must return []."""
    dt = _make_day_table(["2024-01-02"], orb=[10.0])
    result = Levels(ref="orb", ext="ext").split(dt)
    assert result == []

  def test_five_ratios_in_five_bands(self) -> None:
    """Five rows with ratios landing in each of the 5 default bands.

    Hand-calculation with ref=10.0 and multiples=(0.5,1.0,1.5,2.0):
      Band edges: [0, 0.5, 1.0, 1.5, 2.0, inf)
      ratio=0.3  (ext=3.0):  [0,0.5)   → key '0_0_5x'
      ratio=0.7  (ext=7.0):  [0.5,1.0) → key '0_5_1x'
      ratio=1.2  (ext=12.0): [1.0,1.5) → key '1_1_5x'
      ratio=1.7  (ext=17.0): [1.5,2.0) → key '1_5_2x'
      ratio=2.5  (ext=25.0): [2.0,inf) → key 'ge_2x'
    Each band has exactly 1 member.
    """
    dates = [
      "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
    ]
    dt = _make_day_table(
      dates,
      orb=[10.0, 10.0, 10.0, 10.0, 10.0],
      ext=[3.0,  7.0,  12.0, 17.0, 25.0],
    )
    groups = Levels(ref="orb", ext="ext").split(dt)
    by_key = _groups_by_key(groups)

    assert set(by_key.keys()) == {"0_0_5x", "0_5_1x", "1_1_5x", "1_5_2x", "ge_2x"}
    for key in ("0_0_5x", "0_5_1x", "1_1_5x", "1_5_2x", "ge_2x"):
      assert int(by_key[key].mask.sum()) == 1, f"Band {key} should have 1 member"

  def test_band_order_ascending(self) -> None:
    """Groups must be in ascending band order (smallest ratio first).

    Hand-calculation: ratios 0.3, 0.7, 1.2, 1.7, 2.5 → bands in order
    [0_0_5x, 0_5_1x, 1_1_5x, 1_5_2x, ge_2x].
    """
    dates = [
      "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
    ]
    dt = _make_day_table(
      dates,
      orb=[10.0, 10.0, 10.0, 10.0, 10.0],
      ext=[3.0,  7.0,  12.0, 17.0, 25.0],
    )
    groups = Levels(ref="orb", ext="ext").split(dt)

    assert [g.key for g in groups] == ["0_0_5x", "0_5_1x", "1_1_5x", "1_5_2x", "ge_2x"]

  def test_top_band_key_starts_with_ge(self) -> None:
    """The top (unbounded) band key must start with 'ge_'.

    Hand-calculation: ratio=2.5 → >=2x band → key='ge_2x'.
    """
    dt = _make_day_table(["2024-01-01"], orb=[10.0], ext=[25.0])  # ratio=2.5
    groups = Levels(ref="orb", ext="ext").split(dt)

    assert len(groups) == 1
    assert groups[0].key.startswith("ge_")
    assert groups[0].key == "ge_2x"

  def test_ref_zero_excluded(self) -> None:
    """Rows with ref=0 must be excluded (division by zero guard).

    Hand-calculation: row0 ref=0 (excluded), row1 ref=10 ext=5 → ratio=0.5 → '0_5_1x' band.
    Total included: 1 row.
    """
    dt = _make_day_table(
      ["2024-01-01", "2024-01-02"],
      orb=[0.0, 10.0],
      ext=[5.0, 5.0],
    )
    groups = Levels(ref="orb", ext="ext").split(dt)

    # Only 1 valid row (ref=10.0 row)
    total_in_groups = sum(int(g.mask.sum()) for g in groups)
    assert total_in_groups == 1

  def test_ref_negative_excluded(self) -> None:
    """Rows with ref<0 must be excluded (ref must be positive).

    Hand-calculation: row0 ref=-5 (excluded), row1 ref=10 ext=3 → ratio=0.3 → '0_0_5x'.
    """
    dt = _make_day_table(
      ["2024-01-01", "2024-01-02"],
      orb=[-5.0, 10.0],
      ext=[3.0, 3.0],
    )
    groups = Levels(ref="orb", ext="ext").split(dt)

    total_in_groups = sum(int(g.mask.sum()) for g in groups)
    assert total_in_groups == 1

  def test_ref_nan_excluded(self) -> None:
    """Rows with NaN ref must be excluded.

    Hand-calculation: row0 ref=NaN (excluded), row1 ref=10 ext=12 → ratio=1.2 → '1_1_5x'.
    """
    dt = _make_day_table(
      ["2024-01-01", "2024-01-02"],
      orb=[float("nan"), 10.0],
      ext=[5.0, 12.0],
    )
    groups = Levels(ref="orb", ext="ext").split(dt)

    total_in_groups = sum(int(g.mask.sum()) for g in groups)
    assert total_in_groups == 1

  def test_ext_nan_excluded(self) -> None:
    """Rows with NaN ext must be excluded.

    Hand-calculation: row0 ext=NaN (excluded), row1 ref=10 ext=7 → ratio=0.7 → '0_5_1x'.
    """
    dt = _make_day_table(
      ["2024-01-01", "2024-01-02"],
      orb=[10.0, 10.0],
      ext=[float("nan"), 7.0],
    )
    groups = Levels(ref="orb", ext="ext").split(dt)

    total_in_groups = sum(int(g.mask.sum()) for g in groups)
    assert total_in_groups == 1

  def test_empty_bands_omitted(self) -> None:
    """Bands with no members are not emitted.

    Hand-calculation: all rows have ratio=1.2 → only '1_1_5x' band.
    """
    dt = _make_day_table(
      ["2024-01-01", "2024-01-02", "2024-01-03"],
      orb=[10.0, 10.0, 10.0],
      ext=[12.0, 12.0, 12.0],
    )
    groups = Levels(ref="orb", ext="ext").split(dt)

    assert len(groups) == 1
    assert groups[0].key == "1_1_5x"
    assert int(groups[0].mask.sum()) == 3

  def test_boundary_at_0_5_goes_to_second_band(self) -> None:
    """Ratio exactly equal to 0.5 lands in [0.5,1.0) band, not [0,0.5).

    Band assignment: (ratio >= lo) & (ratio < hi), so 0.5 >= 0.5 → second band '0_5_1x'.
    """
    dt = _make_day_table(["2024-01-01"], orb=[10.0], ext=[5.0])  # ratio=0.5
    groups = Levels(ref="orb", ext="ext").split(dt)

    assert len(groups) == 1
    assert groups[0].key == "0_5_1x"

  def test_boundary_at_2_goes_to_top_band(self) -> None:
    """Ratio exactly equal to 2.0 lands in the top [2.0,inf) band.

    Band assignment: ratio=2.0 >= 2.0 and 2.0 < inf → 'ge_2x'.
    """
    dt = _make_day_table(["2024-01-01"], orb=[10.0], ext=[20.0])  # ratio=2.0
    groups = Levels(ref="orb", ext="ext").split(dt)

    assert len(groups) == 1
    assert groups[0].key == "ge_2x"

  def test_custom_multiples(self) -> None:
    """Custom multiples=(1.0, 2.0) produce 3 bands: [0,1), [1,2), [2,inf).

    Hand-calculation: ratios 0.5, 1.5, 2.5 → each in one of 3 bands.
    """
    dt = _make_day_table(
      ["2024-01-01", "2024-01-02", "2024-01-03"],
      orb=[10.0, 10.0, 10.0],
      ext=[5.0, 15.0, 25.0],
    )
    groups = Levels(ref="orb", ext="ext", multiples=(1.0, 2.0)).split(dt)

    assert len(groups) == 3
    # Keys: 0_1x, 1_2x, ge_2x
    assert groups[0].key == "0_1x"
    assert groups[1].key == "1_2x"
    assert groups[2].key == "ge_2x"

  def test_all_ref_nonpositive_returns_empty(self) -> None:
    """If every ref value is <=0, no valid rows exist → []."""
    dt = _make_day_table(
      ["2024-01-01", "2024-01-02"],
      orb=[0.0, -1.0],
      ext=[5.0, 5.0],
    )
    result = Levels(ref="orb", ext="ext").split(dt)
    assert result == []


# ===========================================================================
# 6. resolve_slicer
# ===========================================================================

class TestResolveSlicer:
  """Tests for stats.base.resolve_slicer()."""

  def test_weekday_string_returns_weekday_instance(self) -> None:
    """'weekday' string → Weekday instance."""
    slicer = resolve_slicer("weekday")
    assert isinstance(slicer, Weekday)

  def test_close_string_returns_close_instance(self) -> None:
    """'close' string → Close instance."""
    slicer = resolve_slicer("close")
    assert isinstance(slicer, Close)

  def test_prev_candle_string_returns_prev_candle_instance(self) -> None:
    """'prev_candle' string → PrevCandle instance."""
    slicer = resolve_slicer("prev_candle")
    assert isinstance(slicer, PrevCandle)

  def test_slicer_instance_passed_through(self) -> None:
    """A Slicer instance must be returned as-is."""
    slicer = SizeBucket(column="gap_size")
    result = resolve_slicer(slicer)
    assert result is slicer

  def test_levels_instance_passed_through(self) -> None:
    """A Levels instance must be returned as-is."""
    slicer = Levels(ref="orb", ext="ext")
    result = resolve_slicer(slicer)
    assert result is slicer

  def test_bogus_string_raises_value_error(self) -> None:
    """Unknown string shorthand must raise ValueError."""
    with pytest.raises(ValueError, match="Unknown slice shorthand"):
      resolve_slicer("bogus")

  def test_size_bucket_string_raises_value_error(self) -> None:
    """'size_bucket' string (parameterized) is not a valid shorthand → ValueError.

    Parameterized slicers must be passed as configured instances.
    """
    with pytest.raises(ValueError, match="Unknown slice shorthand"):
      resolve_slicer("size_bucket")

  def test_levels_string_raises_value_error(self) -> None:
    """'levels' string (parameterized) is not a valid shorthand → ValueError."""
    with pytest.raises(ValueError, match="Unknown slice shorthand"):
      resolve_slicer("levels")


# ===========================================================================
# 7. Pydantic model round-trip with slices via write_results
# ===========================================================================

def _make_stat_run_result_with_slices() -> StatRunResult:
  """Build a minimal StatRunResult that contains slice data."""
  slice_group = SliceGroupResult(
    label=I18nString(en="Monday", fr="Lundi"),
    total_samples=5,
    results=[
      StatResultRow(
        condition="green_open",
        outcome="green_close",
        count=3,
        total=5,
        probability=0.6,
        baseline_prob=0.5,
        baseline_n=5,
      )
    ],
  )
  slice_result = SliceResult(
    dimension="weekday",
    groups={"monday": slice_group},
  )
  tf_result = TimeframeResult(
    data_range=["2024-01-01", "2024-01-05"],
    total_samples=5,
    results=[
      StatResultRow(
        condition="green_open",
        outcome="green_close",
        count=3,
        total=5,
        probability=0.6,
        baseline_prob=0.5,
        baseline_n=5,
      )
    ],
    slices={"weekday": slice_result},
  )
  return StatRunResult(
    stat_name="test_stat_with_slices",
    title=I18nString(en="Test Stat", fr="Stat de test"),
    definition=I18nString(en="A test stat", fr="Une stat de test"),
    labels=Labels(
      conditions={"green_open": I18nString(en="Green open", fr="Ouverture verte")},
      outcomes={"green_close": I18nString(en="Green close", fr="Clôture verte")},
      dimensions={"weekday": I18nString(en="Day of week", fr="Jour de la semaine")},
    ),
    instruments={"NQ": {"15min": tf_result}},
  )


def test_write_results_with_slices_creates_json(tmp_path: Path) -> None:
  """write_results() with slice data must create a valid JSON file."""
  result = _make_stat_run_result_with_slices()
  written_path = write_results(result, results_dir=tmp_path)

  assert written_path.exists()
  assert written_path.name == "test_stat_with_slices.json"


def test_write_results_slices_round_trip(tmp_path: Path) -> None:
  """Slice data must survive JSON write → read → validate round-trip.

  After round-trip:
    - slices['weekday'].dimension == 'weekday'
    - slices['weekday'].groups['monday'].total_samples == 5
    - slices['weekday'].groups['monday'].results[0].probability == 0.6
  """
  result = _make_stat_run_result_with_slices()
  written_path = write_results(result, results_dir=tmp_path)

  raw = json.loads(written_path.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)

  tf = validated.instruments["NQ"]["15min"]
  assert "weekday" in tf.slices

  weekday_slice = tf.slices["weekday"]
  assert weekday_slice.dimension == "weekday"
  assert "monday" in weekday_slice.groups

  monday_group = weekday_slice.groups["monday"]
  assert monday_group.total_samples == 5
  assert monday_group.label.en == "Monday"
  assert monday_group.label.fr == "Lundi"

  assert len(monday_group.results) == 1
  assert monday_group.results[0].probability == pytest.approx(0.6)


def test_write_results_dimensions_round_trip(tmp_path: Path) -> None:
  """labels.dimensions must survive JSON round-trip.

  After round-trip:
    validated.labels.dimensions['weekday'].en == 'Day of week'
  """
  result = _make_stat_run_result_with_slices()
  written_path = write_results(result, results_dir=tmp_path)

  raw = json.loads(written_path.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)

  assert "weekday" in validated.labels.dimensions
  assert validated.labels.dimensions["weekday"].en == "Day of week"
  assert validated.labels.dimensions["weekday"].fr == "Jour de la semaine"


def test_write_results_french_accent_in_slices(tmp_path: Path) -> None:
  """Slice labels with accented French characters must be stored as literal UTF-8.

  'Clôture verte' (outcome label) contains ô; must not appear as \\u00f4.
  """
  result = _make_stat_run_result_with_slices()
  written_path = write_results(result, results_dir=tmp_path)

  raw_text = written_path.read_text(encoding="utf-8")
  assert "Clôture verte" in raw_text, "Expected literal accented text in JSON"
  assert "\\u00f4" not in raw_text, "JSON must not use unicode escapes for accents"


def test_write_results_empty_slices_dict_round_trips(tmp_path: Path) -> None:
  """A TimeframeResult with no slices (slices={}) must round-trip cleanly."""
  tf_result = TimeframeResult(
    data_range=[],
    total_samples=0,
    results=[],
    slices={},  # explicitly empty
  )
  run_result = StatRunResult(
    stat_name="no_slices_stat",
    title=I18nString(en="No slices", fr="Sans tranches"),
    definition=I18nString(en="A stat", fr="Une stat"),
    labels=Labels(conditions={}, outcomes={}),
    instruments={"NQ": {"15min": tf_result}},
  )
  written_path = write_results(run_result, results_dir=tmp_path)

  raw = json.loads(written_path.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)

  assert validated.instruments["NQ"]["15min"].slices == {}


# ===========================================================================
# 8. Framework integration: OpeningCandleContinuation weekday slices
# ===========================================================================

# Reuse the same synthetic candle builder and day patterns from the continuation tests.

_DATES_10 = [
  "2024-01-02",  # Tuesday
  "2024-01-03",  # Wednesday
  "2024-01-04",  # Thursday
  "2024-01-05",  # Friday
  "2024-01-08",  # Monday
  "2024-01-09",  # Tuesday
  "2024-01-10",  # Wednesday
  "2024-01-11",  # Thursday
  "2024-01-12",  # Friday
  "2024-01-16",  # Tuesday (MLK day observed on 15th, so 16 is next trading day)
]

# Pattern (session_open, opening_close, session_close):
#   GG x3, GR x2, RG x4, RR x1
# green_open=True when opening_close > session_open
# session_green=True when session_close >= session_open
_DAY_PATTERNS_10 = [
  (100.0, 110.0, 120.0),  # Tue  GG
  (100.0, 110.0, 120.0),  # Wed  GG
  (100.0, 110.0, 120.0),  # Thu  GG
  (100.0, 110.0, 80.0),   # Fri  GR
  (100.0, 110.0, 80.0),   # Mon  GR
  (100.0, 90.0, 120.0),   # Tue  RG
  (100.0, 90.0, 120.0),   # Wed  RG
  (100.0, 90.0, 120.0),   # Thu  RG
  (100.0, 90.0, 120.0),   # Fri  RG
  (100.0, 90.0, 80.0),    # Tue  RR
]

# Weekday breakdown (hand-calculated):
#   Monday   (1 day):  GR       → green_open=1, red_open=0
#   Tuesday  (3 days): GG,RG,RR → green_open=1, red_open=2
#   Wednesday(2 days): GG,RG    → green_open=1, red_open=1
#   Thursday (2 days): GG,RG    → green_open=1, red_open=1
#   Friday   (2 days): GR,RG    → green_open=1, red_open=1
#
# Sum check: green_open totals per weekday = 1+1+1+1+1 = 5 ✓ (matches overall)
#            red_open totals per weekday   = 0+2+1+1+1 = 5 ✓ (matches overall)

_NY = "America/New_York"
_RTH_START_MOD = 570   # 09:30
_RTH_LAST_MOD = 974    # 16:14


def _make_day_for_integration(
  date: str,
  session_open: float,
  opening_close: float,
  session_close: float,
  tf_minutes: int,
) -> pd.DataFrame:
  """Build one trading day of 1-min OHLCV bars for integration tests."""
  base = pd.Timestamp(date, tz=_NY)
  candle_open_mod = (_RTH_START_MOD // tf_minutes) * tf_minutes
  oc_last_mod = candle_open_mod + tf_minutes - 1
  start_mod = min(candle_open_mod, _RTH_START_MOD)

  records = []
  for mod in range(start_mod, _RTH_LAST_MOD + 1):
    h, m = divmod(mod, 60)
    ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
    o = 100.0
    c = 100.0
    if mod == candle_open_mod:
      o = session_open
    if mod == _RTH_START_MOD:
      o = session_open
    if mod == oc_last_mod:
      c = opening_close
    if mod == _RTH_LAST_MOD:
      c = session_close
    records.append({
      "timestamp": ts,
      "open": o,
      "high": max(o, c) + 0.25,
      "low": min(o, c) - 0.25,
      "close": c,
      "volume": 1000,
    })
  return pd.DataFrame(records)


def _make_integration_candles(tf_minutes: int) -> pd.DataFrame:
  """Build the 10-day synthetic dataset for integration tests."""
  frames = []
  for i, (so, oc, sc) in enumerate(_DAY_PATTERNS_10):
    frames.append(
      _make_day_for_integration(
        date=_DATES_10[i],
        session_open=so,
        opening_close=oc,
        session_close=sc,
        tf_minutes=tf_minutes,
      )
    )
  df = pd.concat(frames, ignore_index=True)
  return df.sort_values("timestamp").reset_index(drop=True)


def test_compute_produces_weekday_slice(tmp_path: Path) -> None:
  """After compute(), TimeframeResult.slices must contain 'weekday'.

  The stat declares slices=['weekday'], so the framework must run the
  weekday slicer and populate slices['weekday'] in the result.
  """
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  df = _make_integration_candles(15)
  result = stat.compute(df)

  tf_result = result.instruments["NQ"]["15min"]
  assert "weekday" in tf_result.slices
  assert tf_result.slices["weekday"].dimension == "weekday"


def test_weekday_slice_has_five_groups(tmp_path: Path) -> None:
  """The weekday slice must have exactly 5 groups (Mon–Fri all present in data).

  Hand-calculation: _DATES_10 spans Mon,Tue,Wed,Thu,Fri → 5 weekday keys.
  """
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  df = _make_integration_candles(15)
  result = stat.compute(df)

  weekday_slice = result.instruments["NQ"]["15min"].slices["weekday"]
  assert len(weekday_slice.groups) == 5
  assert set(weekday_slice.groups.keys()) == {
    "monday", "tuesday", "wednesday", "thursday", "friday"
  }


def test_weekday_slice_total_samples(tmp_path: Path) -> None:
  """Each weekday group's total_samples must match the day count for that weekday.

  Hand-calculation:
    Monday=1, Tuesday=3, Wednesday=2, Thursday=2, Friday=2
  """
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  df = _make_integration_candles(15)
  result = stat.compute(df)

  groups = result.instruments["NQ"]["15min"].slices["weekday"].groups

  assert groups["monday"].total_samples == 1
  assert groups["tuesday"].total_samples == 3
  assert groups["wednesday"].total_samples == 2
  assert groups["thursday"].total_samples == 2
  assert groups["friday"].total_samples == 2


def test_weekday_slice_totals_sum_to_overall(tmp_path: Path) -> None:
  """Sum of per-weekday group totals for each condition must equal the overall total.

  Hand-calculation:
    green_open condition overall total=5
    Per weekday green_open totals: Mon=1, Tue=1, Wed=1, Thu=1, Fri=1 → sum=5 ✓

    red_open condition overall total=5
    Per weekday red_open totals: Mon=0, Tue=2, Wed=1, Thu=1, Fri=1 → sum=5 ✓
  """
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  df = _make_integration_candles(15)
  result = stat.compute(df)

  tf_result = result.instruments["NQ"]["15min"]
  groups = tf_result.slices["weekday"].groups

  # Find overall totals
  def _overall_total(condition: str) -> int:
    for row in tf_result.results:
      if row.condition == condition and row.outcome == "green_close":
        return row.total
    raise KeyError(condition)

  overall_green_open_total = _overall_total("green_open")  # == 5
  overall_red_open_total = _overall_total("red_open")      # == 5

  # Sum per-weekday group totals across all groups
  def _sum_condition_total(condition: str) -> int:
    total = 0
    for g in groups.values():
      for row in g.results:
        if row.condition == condition and row.outcome == "green_close":
          total += row.total
    return total

  sum_green_open = _sum_condition_total("green_open")
  sum_red_open = _sum_condition_total("red_open")

  assert sum_green_open == overall_green_open_total, (
    f"Sum of weekday green_open totals={sum_green_open} != overall {overall_green_open_total}"
  )
  assert sum_red_open == overall_red_open_total, (
    f"Sum of weekday red_open totals={sum_red_open} != overall {overall_red_open_total}"
  )


def test_weekday_slice_monday_exact_counts(tmp_path: Path) -> None:
  """Monday has exactly 1 day (GR pattern) → green_open=1, red_open=0.

  Hand-calculation: 2024-01-08 is Monday with pattern GR.
    green_open → green_close: count=0, total=1, prob=0.0
    green_open → red_close:   count=1, total=1, prob=1.0
    red_open   → *:           total=0 (no red open days on Monday)
  """
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  df = _make_integration_candles(15)
  result = stat.compute(df)

  monday = result.instruments["NQ"]["15min"].slices["weekday"].groups["monday"]
  rows_by_key = {(r.condition, r.outcome): r for r in monday.results}

  gg_mon = rows_by_key[("green_open", "green_close")]
  gr_mon = rows_by_key[("green_open", "red_close")]
  rg_mon = rows_by_key[("red_open", "green_close")]

  # Monday: 1 day, green opening, red session close
  assert gg_mon.total == 1
  assert gg_mon.count == 0
  assert gg_mon.probability == pytest.approx(0.0)

  assert gr_mon.total == 1
  assert gr_mon.count == 1
  assert gr_mon.probability == pytest.approx(1.0)

  assert rg_mon.total == 0
  assert rg_mon.probability == pytest.approx(0.0)


def test_weekday_slice_tuesday_exact_counts(tmp_path: Path) -> None:
  """Tuesday has 3 days (GG, RG, RR) → correct per-condition totals.

  Hand-calculation:
    Tue days: 2024-01-02 (GG), 2024-01-09 (RG), 2024-01-16 (RR)
    green_open total=1 (only 2024-01-02)
      green_close: count=1, prob=1.0
      red_close:   count=0, prob=0.0
    red_open total=2 (2024-01-09 and 2024-01-16)
      green_close: count=1 (2024-01-09), prob=0.5
      red_close:   count=1 (2024-01-16), prob=0.5
  """
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  df = _make_integration_candles(15)
  result = stat.compute(df)

  tuesday = result.instruments["NQ"]["15min"].slices["weekday"].groups["tuesday"]
  rows_by_key = {(r.condition, r.outcome): r for r in tuesday.results}

  # green_open: 1 day on Tuesday (GG)
  assert rows_by_key[("green_open", "green_close")].total == 1
  assert rows_by_key[("green_open", "green_close")].count == 1
  assert rows_by_key[("green_open", "green_close")].probability == pytest.approx(1.0)

  assert rows_by_key[("green_open", "red_close")].total == 1
  assert rows_by_key[("green_open", "red_close")].count == 0
  assert rows_by_key[("green_open", "red_close")].probability == pytest.approx(0.0)

  # red_open: 2 days on Tuesday (RG, RR)
  assert rows_by_key[("red_open", "green_close")].total == 2
  assert rows_by_key[("red_open", "green_close")].count == 1
  assert rows_by_key[("red_open", "green_close")].probability == pytest.approx(0.5)

  assert rows_by_key[("red_open", "red_close")].total == 2
  assert rows_by_key[("red_open", "red_close")].count == 1
  assert rows_by_key[("red_open", "red_close")].probability == pytest.approx(0.5)


def test_weekday_slice_groups_have_baseline_n(tmp_path: Path) -> None:
  """Every result row in every weekday group must have baseline_n > 0.

  The framework calls baseline_rows() on each slice group's sub-table.
  For non-empty groups, baseline_n must reflect the group's sample size.
  """
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  df = _make_integration_candles(15)
  result = stat.compute(df)

  groups = result.instruments["NQ"]["15min"].slices["weekday"].groups

  for day_key, group in groups.items():
    for row in group.results:
      if row.total > 0:
        assert row.baseline_n > 0, (
          f"Weekday '{day_key}', condition='{row.condition}', outcome='{row.outcome}': "
          f"baseline_n={row.baseline_n} but total={row.total}"
        )


def test_labels_dimensions_contains_weekday(tmp_path: Path) -> None:
  """result.labels.dimensions must contain 'weekday' with non-empty en/fr.

  The framework merges dimension labels from all slicers into labels.dimensions.
  """
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  df = _make_integration_candles(15)
  result = stat.compute(df)

  assert "weekday" in result.labels.dimensions
  dim_label = result.labels.dimensions["weekday"]
  assert dim_label.en != ""
  assert dim_label.fr != ""


def test_weekday_slice_write_results_round_trip(tmp_path: Path) -> None:
  """Weekday slice data from compute() must survive a JSON write/read round-trip.

  Verifies that the full pipeline (compute → write_results → JSON → validate)
  preserves the slice structure, including group counts and probabilities.
  """
  stat = OpeningCandleContinuation(instrument="NQ", timeframe="15min", config=_TEST_CONFIG)
  df = _make_integration_candles(15)
  result = stat.compute(df)
  written_path = write_results(result, results_dir=tmp_path)

  raw = json.loads(written_path.read_text(encoding="utf-8"))
  validated = StatRunResult.model_validate(raw)

  tf = validated.instruments["NQ"]["15min"]
  assert "weekday" in tf.slices

  # Verify Monday group survives round-trip
  monday = tf.slices["weekday"].groups["monday"]
  assert monday.total_samples == 1

  rows_by_key = {(r.condition, r.outcome): r for r in monday.results}
  gr_mon = rows_by_key[("green_open", "red_close")]
  assert gr_mon.count == 1
  assert gr_mon.probability == pytest.approx(1.0)
