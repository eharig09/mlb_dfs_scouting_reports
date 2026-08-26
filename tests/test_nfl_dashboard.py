"""The NFL dashboard page and the readers behind it.

Lives at `tests/` root beside the other dashboard tests rather than in `tests/nfl/`, which
ships no conftest on purpose. See `tests/nfl/nfl_fixtures.py` for why.
"""

import os

import pandas as pd
import pytest

from dashboards import nfl_boards


def _write(root, date, slate, kind, frame):
    directory = os.path.join(root, date)
    os.makedirs(directory, exist_ok=True)
    extension = {"pool": ".csv", "lineups": ".csv", "exposure": ".csv"}[kind]
    path = os.path.join(directory, f"{kind}_{slate}{extension}")
    frame.to_csv(path, index=False)
    return path


LINEUPS = pd.DataFrame([
    {"Lineup": 1, "Slot": "QB", "Name": "A", "Pos": "QB", "Team": "BUF",
     "Lineup Salary": 49900, "Lineup Proj": 130.0, "Lineup Ceiling": 240.0,
     "Shape": "3-1", "Anchor": "BUF"},
    {"Lineup": 1, "Slot": "WR", "Name": "B", "Pos": "WR", "Team": "BUF",
     "Lineup Salary": 49900, "Lineup Proj": 130.0, "Lineup Ceiling": 240.0,
     "Shape": "3-1", "Anchor": "BUF"},
    {"Lineup": 2, "Slot": "QB", "Name": "C", "Pos": "QB", "Team": "KAN",
     "Lineup Salary": 49000, "Lineup Proj": 125.0, "Lineup Ceiling": 250.0,
     "Shape": "3", "Anchor": "KAN"},
])


class TestListing:
    def test_dates_come_back_newest_first(self, tmp_path):
        root = str(tmp_path)
        for date in ("2026-09-07", "2026-09-14"):
            _write(root, date, "main", "pool", pd.DataFrame([{"Name": "A"}]))
        assert nfl_boards.list_dates(root=root) == ["2026-09-14", "2026-09-07"]

    def test_a_missing_root_is_empty_not_an_error(self, tmp_path):
        """A first run has no `nfl_boards/` at all, and the page must say so rather than
        raise on a directory that has simply not been created yet."""
        assert nfl_boards.list_dates(root=str(tmp_path / "nope")) == []

    def test_slates_are_the_union_across_kinds(self, tmp_path):
        """A slate whose pool exists but whose lineups have not been built yet still
        exists. Listing only one kind would make it look untouched."""
        root = str(tmp_path)
        _write(root, "2026-09-14", "main", "pool", pd.DataFrame([{"Name": "A"}]))
        _write(root, "2026-09-14", "primetime", "lineups", LINEUPS)
        assert nfl_boards.list_slates("2026-09-14", root=root) == ["main", "primetime"]

    def test_a_step_that_has_not_run_reads_as_none(self, tmp_path):
        root = str(tmp_path)
        _write(root, "2026-09-14", "main", "pool", pd.DataFrame([{"Name": "A"}]))
        assert nfl_boards.read("lineups", "2026-09-14", "main", root=root) is None
        assert nfl_boards.read("pool", "2026-09-14", "main", root=root) is not None


class TestLineupSummary:
    def test_one_row_per_lineup_best_ceiling_first(self):
        summary = nfl_boards.lineup_summary(LINEUPS)
        assert list(summary["Lineup"]) == [2, 1]
        assert summary["Ceiling"].iloc[0] == 250.0

    def test_the_stack_is_read_off_the_passing_game(self):
        """QB/WR/TE only -- a running back never counts toward his own team's stack."""
        summary = nfl_boards.lineup_summary(LINEUPS)
        first = summary[summary["Lineup"] == 1].iloc[0]
        assert "BUF x2" in first["Stack"]

    def test_nothing_built_is_an_empty_frame_not_a_crash(self):
        assert nfl_boards.lineup_summary(None).empty
        assert nfl_boards.lineup_summary(pd.DataFrame()).empty


class TestSplitExposure:
    def test_player_and_team_rows_separate(self):
        exposure = pd.DataFrame([
            {"Level": "player", "Name": "A", "Lineups": 3, "Exposure%": 100.0},
            {"Level": "team", "Team": "BUF", "Lineups": 2, "Exposure%": 66.7},
        ])
        players, teams = nfl_boards.split_exposure(exposure)
        assert list(players["Name"]) == ["A"]
        assert list(teams["Team"]) == ["BUF"]

    def test_no_exposure_is_two_empty_frames(self):
        players, teams = nfl_boards.split_exposure(None)
        assert players.empty and teams.empty


class TestArrowSafe:
    def test_a_numeric_column_carrying_blanks_becomes_nullable(self):
        """Arrow refuses an object column holding numbers and '' together -- and Streamlit
        then renders **nothing at all** where the table should be, with no error on screen.
        The pool writes exactly that shape, because its edit columns ship blank.
        """
        frame = pd.DataFrame({"Proj": ["10.5", "", "12.0"]})
        out = nfl_boards.arrow_safe(frame)
        assert pd.api.types.is_numeric_dtype(out["Proj"])
        assert out["Proj"].isna().sum() == 1

    def test_a_genuinely_mixed_column_is_left_alone(self):
        frame = pd.DataFrame({"Basis": ["grade", "", "draft capital"]})
        out = nfl_boards.arrow_safe(frame)
        assert list(out["Basis"]) == ["grade", "", "draft capital"]

    def test_values_are_never_reformatted_only_retyped(self):
        frame = pd.DataFrame({"Salary": ["5500", "6200"]})
        out = nfl_boards.arrow_safe(frame)
        assert list(out["Salary"]) == [5500, 6200]


class TestThePageRuns:
    """Executes the page script itself, which is the only way to catch a page-level error.

    An HTTP health check on the running server does not run the page, so a page that raises
    on every render still answers 200.
    """

    def _apptest(self):
        pytest.importorskip("streamlit.testing.v1")
        from streamlit.testing.v1 import AppTest
        # Absolute: AppTest resolves a relative path against the *calling file*, not the
        # working directory, so "dashboards/..." from here looks under tests/.
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        page = os.path.join(root, "dashboards", "app_pages", "nfl_slate.py")
        return AppTest.from_file(page, default_timeout=90)

    def test_it_renders_without_raising(self):
        app = self._apptest().run()
        assert not app.exception
        assert [h.value for h in app.header] == ["NFL slate"]

    def test_it_tells_you_what_to_run_when_there_is_nothing_yet(self, monkeypatch):
        """The state a first-time user is in. An empty page with no explanation reads as a
        broken dashboard rather than as an empty one."""
        monkeypatch.setattr(nfl_boards, "list_dates", lambda root=None: [])
        app = self._apptest().run()
        assert not app.exception
        assert any("nfl.optimize" in i.value for i in app.info)
