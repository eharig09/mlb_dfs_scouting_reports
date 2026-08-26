"""The editable pool: the merge that protects hand markup, and the parsers around it."""

import numpy as np
import pandas as pd
import pytest

from nfl import pool


def _slate(rows):
    return pd.DataFrame(rows, columns=["Name", "DK Pos", "Team", "Salary", "Proj"])


TWO = _slate([["Josh Allen", "QB", "BUF", 8000, 22.0],
              ["Tyreek Hill", "WR", "MIA", 7000, 18.0]])


class TestTextGuard:
    def test_a_missing_cell_reads_as_empty_not_as_the_word_nan(self):
        """`str(value or "")` returns the four-character string "nan" for a blank CSV cell,
        because NaN is truthy. Every guard downstream then sees a non-empty value, and a
        freshly written pool reports all of its rows as hand-edited."""
        assert pool._text(np.nan) == ""
        assert pool._text(None) == ""
        assert pool._text(pd.NA) == ""
        assert pool._text("  y ") == "y"


class TestIsEdited:
    def test_a_freshly_written_row_is_not_an_edit(self):
        assert not pool._is_edited(dict(pool.DEFAULT_EDITS))

    def test_a_blank_row_read_back_from_csv_is_not_an_edit(self):
        untouched = {"Lock": np.nan, "Exclude": np.nan, "Boost": "1.0",
                     "Min%": np.nan, "Max%": np.nan}
        assert not pool._is_edited(untouched)

    def test_any_real_mark_counts(self):
        for column, value in (("Lock", "y"), ("Exclude", "x"),
                              ("Boost", "1.2"), ("Max%", "25%")):
            values = dict(pool.DEFAULT_EDITS, **{column: value})
            assert pool._is_edited(values), column


class TestParsers:
    def test_a_single_boost_and_a_range(self):
        assert pool.parse_boost("1.2") == (1.2, 1.2)
        assert pool.parse_boost("0.9-1.3") == (0.9, 1.3)

    def test_a_reversed_range_is_ordered_rather_than_rejected(self):
        assert pool.parse_boost("1.3-0.9") == (0.9, 1.3)

    def test_blank_and_junk_are_none(self):
        for value in ("", None, np.nan, "abc"):
            assert pool.parse_boost(value) is None

    def test_percent_accepts_both_spellings(self):
        """A spreadsheet will happily turn one into the other, and a Max% read as 2500%
        is not a constraint at all."""
        assert pool._percent("25") == 0.25
        assert pool._percent("25%") == 0.25
        assert pool._percent("0.25") == 0.25

    def test_percent_rejects_blank_and_negative(self):
        assert pool._percent("") is None
        assert pool._percent(np.nan) is None
        assert pool._percent("-5") is None


class TestMergeProtectsMarkup:
    def _write(self, frame, tmp_path, **kwargs):
        return pool.write_pool(frame, "2026-09-14", slate="main", root=str(tmp_path),
                               **kwargs)

    def test_a_fresh_pool_carries_nothing(self, tmp_path):
        _, _, report = self._write(TWO, tmp_path)
        assert report["carried"] == 0

    def test_hand_marks_survive_a_rebuild(self, tmp_path):
        """The whole reason the pool merges. A rebuild that discarded markup would lose
        days of decisions, and you would only notice after the lineups were wrong."""
        path, _, _ = self._write(TWO, tmp_path)
        frame = pd.read_csv(path, dtype=str)
        frame.loc[0, "Lock"] = "y"
        frame.to_csv(path, index=False)

        path, _, report = self._write(TWO, tmp_path, overwrite=True)
        assert report["carried"] == 1
        locks, _, _, _ = pool.read_pool(path)
        assert locks == ["Josh Allen"]

    def test_merge_false_is_a_deliberate_clean_slate(self, tmp_path):
        path, _, _ = self._write(TWO, tmp_path)
        frame = pd.read_csv(path, dtype=str)
        frame.loc[0, "Lock"] = "y"
        frame.to_csv(path, index=False)

        path, _, report = self._write(TWO, tmp_path, overwrite=True, merge=False)
        assert report["carried"] == 0
        assert pool.read_pool(path)[0] == []

    def test_a_lock_never_migrates_to_a_same_named_player(self, tmp_path):
        """`normalize_name` strips generational suffixes, so two players separated only by
        a 'Jr.' fold to one key and the team code is all that holds them apart.

        Before the ambiguity guard, marking one of them locked both.
        """
        both = _slate([["Odell Beckham Jr.", "WR", "BUF", 5000, 10.0],
                       ["Odell Beckham", "WR", "MIA", 4000, 8.0]])
        path, _, _ = self._write(both, tmp_path)
        frame = pd.read_csv(path, dtype=str)
        frame.loc[frame["Team"] == "BUF", "Lock"] = "y"
        frame.to_csv(path, index=False)

        path, _, report = self._write(both, tmp_path, overwrite=True)
        out = pd.read_csv(path, dtype=str)
        locked = out[out["Lock"].fillna("") == "y"]
        assert list(locked["Team"]) == ["BUF"]
        assert report["carried"] == 1

    def test_an_edit_survives_the_player_changing_teams(self, tmp_path):
        """The name-only fallback exists for this, and football churns rosters weekly."""
        path, _, _ = self._write(TWO, tmp_path)
        frame = pd.read_csv(path, dtype=str)
        frame.loc[frame["Name"] == "Tyreek Hill", "Lock"] = "y"
        frame.to_csv(path, index=False)

        moved = _slate([["Josh Allen", "QB", "BUF", 8000, 22.0],
                        ["Tyreek Hill", "WR", "KAN", 7000, 18.0]])
        path, _, report = self._write(moved, tmp_path, overwrite=True)
        assert report["carried"] == 1
        assert pool.read_pool(path)[0] == ["Tyreek Hill"]

    def test_an_edit_outliving_its_player_is_kept_not_deleted(self, tmp_path):
        """A marked-up player who leaves the slate keeps his row, blank, so the decision
        is visible rather than silently dropped by a rebuild."""
        path, _, _ = self._write(TWO, tmp_path)
        frame = pd.read_csv(path, dtype=str)
        frame.loc[frame["Name"] == "Tyreek Hill", "Exclude"] = "x"
        frame.to_csv(path, index=False)

        smaller = _slate([["Josh Allen", "QB", "BUF", 8000, 22.0]])
        path, _, report = self._write(smaller, tmp_path, overwrite=True)
        assert report["orphaned"] == ["Tyreek Hill"]
        assert "Tyreek Hill" in set(pd.read_csv(path, dtype=str)["Name"])


class TestReadPool:
    def test_an_excluded_players_other_settings_are_moot(self, tmp_path):
        path = tmp_path / "pool.csv"
        pd.DataFrame([{"Name": "A", "Team": "BUF", "Lock": "y", "Exclude": "y",
                       "Boost": "1.5", "Min%": "", "Max%": ""}]).to_csv(path, index=False)
        locks, excludes, boosts, _ = pool.read_pool(str(path))
        assert excludes == ["A"] and locks == [] and boosts == {}

    def test_a_file_that_is_not_a_pool_fails_loudly(self, tmp_path):
        path = tmp_path / "other.csv"
        pd.DataFrame([{"Player": "A"}]).to_csv(path, index=False)
        with pytest.raises(ValueError):
            pool.read_pool(str(path))


class TestResolveExposure:
    def test_fractions_become_lineup_counts(self):
        out = pool.resolve_exposure(TWO, {"josh allen": (0.25, 0.5)}, 20)
        assert out["Josh Allen"] == (5, 10)

    def test_a_player_no_longer_on_the_slate_produces_no_constraint(self):
        """Keyed off the current slate on purpose. A Min% on someone who is no longer
        priced would make the whole set infeasible with nothing on screen to say why."""
        seen = []
        out = pool.resolve_exposure(TWO, {"nobody at all": (0.5, None)}, 20,
                                    report=seen.append)
        assert out == {}
        assert seen == [["nobody at all"]]

    def test_an_inverted_range_does_not_become_unsatisfiable(self):
        out = pool.resolve_exposure(TWO, {"josh allen": (0.9, 0.1)}, 10)
        low, high = out["Josh Allen"]
        assert low <= high
