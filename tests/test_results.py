"""Tying a contest export to the slate it was run on.

Standings files are named by contest id, so the slate is inferred. The original inference --
the share of the *contest's* players appearing on the slate -- rewards small files: an
80-player export is a near-subset of almost any slate, and with 19 files on disk five slates
resolved to another night's contest and were reviewed against the wrong ownership.
"""

import pandas as pd
import pytest
import zipfile

from dfs.results import contest_night, list_contests, match_contest, result_files


def contest(names, entries=1000, path="contest-standings-1.csv"):
    return {"path": path, "entries": entries,
            "ownership": {n.lower(): 10.0 for n in names},
            "fpts": {}, "scores": pd.Series(dtype=float), "players": len(names)}


def slate(names):
    return pd.DataFrame({"Name": list(names)})


def index_entry(date, label, names):
    return {"date": date, "slate": label, "names": {n.lower() for n in names},
            "size": len(names)}


ALPHA = [f"player {i}" for i in range(40)]        # on both nights
BETA = [f"beta {i}" for i in range(40)]           # 08-07 only
GAMMA = [f"gamma {i}" for i in range(40)]         # 08-08 only


class TestContestNight:
    def test_the_tightest_containing_export_wins(self):
        """Containment ties across days; the smallest export is the slate it ran on."""
        index = [index_entry("2026-08-06", "main", ALPHA),
                 index_entry("2026-08-04", "main", ALPHA + BETA + GAMMA)]
        assert contest_night(contest(ALPHA), index=index) == ("2026-08-06", "main")

    def test_an_export_that_does_not_contain_the_players_is_not_it(self):
        index = [index_entry("2026-08-08", "turbo", GAMMA)]
        assert contest_night(contest(ALPHA), index=index) == (None, None)

    def test_consecutive_days_are_separated_by_what_they_do_not_share(self):
        index = [index_entry("2026-08-07", "turbo", ALPHA + BETA),
                 index_entry("2026-08-08", "turbo", ALPHA + GAMMA)]
        assert contest_night(contest(ALPHA + BETA), index=index)[0] == "2026-08-07"
        assert contest_night(contest(ALPHA + GAMMA), index=index)[0] == "2026-08-08"

    def test_an_empty_contest_is_undatable(self):
        assert contest_night(contest([]), index=[index_entry("2026-08-07", "main", ALPHA)]) \
            == (None, None)

    def test_archive_chronology_prevents_a_later_subset_collision(self):
        export = contest(ALPHA)
        export["observed_at"] = "2026-08-26T02:59:40"
        index = [index_entry("2026-08-25", "turbo", ALPHA + ["extra"]),
                 index_entry("2026-08-29", "early", ALPHA)]

        assert contest_night(export, index=index) == ("2026-08-25", "turbo")


class TestMatchContest:
    def test_a_small_file_no_longer_beats_the_night_s_own(self):
        """The bug: one-way overlap made an 80-player subset outscore the real file."""
        here = slate(ALPHA + BETA)
        right = contest(ALPHA + BETA, entries=951, path="right.csv")
        subset = contest(ALPHA[:20], entries=235, path="subset.csv")
        best, _ = match_contest(here, contests=[subset, right])
        assert best["path"] == "right.csv"

    def test_the_largest_field_supplies_the_ownership(self):
        """Once dated, overlap differences are contest size -- %Drafted over 5,945 wins."""
        here = slate(ALPHA)
        small = contest(ALPHA, entries=235, path="small.csv")
        big = contest(ALPHA, entries=5945, path="big.csv")
        best, _ = match_contest(here, contests=[small, big])
        assert best["path"] == "big.csv"

    def test_an_unrelated_contest_is_refused(self):
        best, overlap = match_contest(slate(ALPHA), contests=[contest(GAMMA)])
        assert best is None and overlap < 0.45

    def test_an_empty_slate_is_refused(self):
        best, overlap = match_contest(pd.DataFrame({"Name": []}), contests=[contest(ALPHA)])
        assert best is None and overlap == 0.0

    def test_no_contests_at_all_is_not_an_error(self):
        assert match_contest(slate(ALPHA), contests=[]) == (None, 0.0)


def _standings_frame(lineup):
    return pd.DataFrame({
        "Rank": [1], "Points": [100], "Lineup": [lineup],
        "Player": ["Player One"], "Roster Position": ["OF"],
        "%Drafted": ["25%"], "FPTS": [10],
    })


def _write_zip(path, frame):
    info = zipfile.ZipInfo(path.with_suffix(".csv").name, (2026, 8, 26, 2, 59, 40))
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(info, frame.to_csv(index=False).encode("utf-8-sig"))


def test_zip_is_a_native_input_and_is_not_double_counted(tmp_path):
    classic = "P One P Two C Three 1B Four 2B Five 3B Six SS Seven OF Eight OF Nine OF Ten"
    frame = _standings_frame(classic)
    archive = tmp_path / "contest-standings-123.zip"
    _write_zip(archive, frame)

    contests = list_contests(str(tmp_path))
    assert len(contests) == 1
    assert contests[0]["format"] == "classic"
    assert contests[0]["observed_at"] == "2026-08-26T02:59:40"

    csv_path = archive.with_suffix(".csv")
    frame.to_csv(csv_path, index=False)
    assert result_files(str(tmp_path)) == [str(csv_path)]
    assert len(list_contests(str(tmp_path))) == 1


def test_showdown_is_excluded_from_classic_results(tmp_path):
    frame = _standings_frame("CPT Captain One UTIL Player Two UTIL Player Three")
    archive = tmp_path / "contest-standings-showdown.zip"
    _write_zip(archive, frame)

    assert list_contests(str(tmp_path)) == []
    assert len(list_contests(str(tmp_path), include_showdown=True)) == 1
