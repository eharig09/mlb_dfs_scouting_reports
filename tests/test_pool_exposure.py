"""A Min% the slate cannot honour must not disappear quietly.

Three ways a marked-up player stops being rosterable between the pool file being written and
the optimizer running, and every one of them used to end the same way: the run reported
success and the player was simply absent from lineups he was supposed to be in — or, worse,
forced into all of them after being scratched.
"""

import numpy as np
import pandas as pd
import pytest

from dfs.pool import resolve_exposure
from dfs.salaries import normalize_name


def slate(**overrides):
    frame = pd.DataFrame({
        "Name": ["Josh Naylor", "Cal Raleigh", "Julio Rodriguez"],
        "Team": ["SEA"] * 3,
        "Type": ["H"] * 3,
        "Salary": [3000.0, 5200.0, 4300.0],
        "Lineup": ["Confirmed"] * 3,
    })
    for column, value in overrides.items():
        frame[column] = value
    return frame


def minimum(name, share=0.4):
    return {normalize_name(name): (share, None)}


class TestHonoured:
    def test_a_playing_priced_hitter_gets_his_floor(self):
        report = []
        limits = resolve_exposure(slate(), minimum("Josh Naylor"), 20, report=report)
        assert limits["Josh Naylor"] == (8, 20)
        assert report == []

    def test_a_maximum_alone_is_not_treated_as_a_floor(self):
        limits = resolve_exposure(slate(), {normalize_name("Cal Raleigh"): (None, 0.5)}, 20)
        assert limits["Cal Raleigh"] == (0, 10)

    def test_the_floor_rounds_up_so_a_small_share_still_appears(self):
        limits = resolve_exposure(slate(), minimum("Cal Raleigh", 0.01), 20)
        assert limits["Cal Raleigh"][0] == 1


class TestDropped:
    def test_an_unpriced_player_loses_his_floor_and_says_so(self):
        """DraftKings drops IL/OUT/NA/SUSP rows, so an unpriced row is usually DK saying he
        is not playing — and `optimize` filters those out of the roster pool entirely, so a
        minimum on one can never be met however hard the solver tries."""
        frame = slate()
        frame.loc[frame["Name"] == "Josh Naylor", "Salary"] = np.nan
        report = []
        limits = resolve_exposure(frame, minimum("Josh Naylor"), 20, report=report)
        assert "Josh Naylor" not in limits
        assert len(report) == 1
        assert "no DK price" in report[0][1]

    def test_a_player_no_longer_on_the_slate_says_so(self):
        """What a scratch looks like once the payload has been refreshed: no lineup row."""
        frame = slate()
        frame = frame[frame["Name"] != "Josh Naylor"]
        report = []
        limits = resolve_exposure(frame, minimum("Josh Naylor"), 20, report=report)
        assert limits == {} or "Josh Naylor" not in limits
        assert any("not on the slate" in why for _, why in report)

    def test_an_absent_player_with_only_a_maximum_is_not_reported(self):
        """A cap on someone who is not playing is satisfied by doing nothing — there is no
        instruction being silently discarded, so there is nothing to warn about."""
        frame = slate()
        frame = frame[frame["Name"] != "Josh Naylor"]
        report = []
        resolve_exposure(frame, {normalize_name("Josh Naylor"): (None, 0.5)}, 20,
                         report=report)
        assert report == []

    def test_the_report_is_optional(self):
        frame = slate()
        frame.loc[frame["Name"] == "Josh Naylor", "Salary"] = np.nan
        assert "Josh Naylor" not in resolve_exposure(frame, minimum("Josh Naylor"), 20)


class TestUnconfirmed:
    def test_a_floor_on_a_merely_projected_bat_is_kept_but_flagged(self):
        """The case that bites: the projector still has him batting sixth, DraftKings still
        has him priced, and nothing in the solve knows he was scratched. The minimum is
        honoured — refusing it would break every morning run, before any card is posted —
        but the run has to say what it is about to force."""
        frame = slate(Lineup=["Projected (8/9)", "Confirmed", "Confirmed"])
        report = []
        limits = resolve_exposure(frame, minimum("Josh Naylor"), 20, report=report)
        assert limits["Josh Naylor"] == (8, 20), "still honoured"
        assert any("not confirmed" in why for _, why in report)

    def test_a_confirmed_bat_draws_no_warning(self):
        report = []
        resolve_exposure(slate(), minimum("Josh Naylor"), 20, report=report)
        assert report == []

    def test_a_blank_lineup_status_is_not_treated_as_unconfirmed(self):
        """An older payload carries no status at all; inventing a warning for it would cry
        wolf on every run against a cached night."""
        report = []
        resolve_exposure(slate(Lineup=""), minimum("Josh Naylor"), 20, report=report)
        assert report == []


class TestOptimizerGate:
    def test_the_roster_pool_really_does_exclude_unpriced_players(self):
        """The reason an unpriced minimum is unmeetable rather than merely unlikely."""
        import inspect

        from dfs import optimizer

        source = inspect.getsource(optimizer)
        assert 'players["Salary"].notna()' in source
