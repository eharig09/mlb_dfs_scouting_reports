"""Tying a contest export to the slate it was run on.

Standings files are named by contest id, so the slate is inferred. The original inference --
the share of the *contest's* players appearing on the slate -- rewards small files: an
80-player export is a near-subset of almost any slate, and with 19 files on disk five slates
resolved to another night's contest and were reviewed against the wrong ownership.
"""

import pandas as pd
import pytest

from dfs.results import contest_night, match_contest


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
