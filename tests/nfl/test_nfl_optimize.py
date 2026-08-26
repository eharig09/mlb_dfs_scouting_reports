"""The lineup CLI: shape parsing, bands, the board, and the reports it writes."""

import numpy as np
import pandas as pd
import pytest

from nfl import optimize
from nfl import optimizer as opt


class TestModuleIdentity:
    def test_nfl_optimize_is_the_pipeline_module_not_the_solver_function(self):
        """`nfl/__init__.py` used to re-export the solver as `optimize`, which shadowed this
        module: `from nfl import optimize` handed back a function and every attribute lookup
        on it failed. `python -m nfl.optimize` still worked, because runpy loads the file
        directly -- so the collision was invisible until something imported it.

        The solver is `nfl.optimizer.optimize`; the pipeline is `nfl.optimize`.
        """
        import types

        import nfl
        assert isinstance(optimize, types.ModuleType)
        assert not isinstance(getattr(nfl, "optimize", None), types.FunctionType)
        assert callable(opt.optimize)


class TestParseShape:
    def test_a_bare_number_is_a_stack_with_no_bringback(self):
        assert optimize.parse_shape("3") == [(3, 0, 0)]

    def test_a_hyphen_adds_the_bringback(self):
        assert optimize.parse_shape("3-1") == [(3, 1, 0)]

    def test_a_plus_adds_a_secondary_stack_from_another_game(self):
        """The third number is diversification: a second, smaller correlated block so one
        game going quiet does not take the whole lineup with it."""
        assert optimize.parse_shape("3-1+2") == [(3, 1, 2)]

    def test_several_shapes_are_explored_in_turn(self):
        assert optimize.parse_shape("4-1,3-1,3") == [(4, 1, 0), (3, 1, 0), (3, 0, 0)]

    def test_a_one_man_secondary_stack_is_refused(self):
        """One player is not a stack — he is a player, and asking for `+1` almost always
        means the shape was mistyped."""
        with pytest.raises(SystemExit):
            optimize.parse_shape("3-1+1")

    def test_a_shape_that_cannot_fit_the_roster_is_refused(self):
        """Better to say so than to hand the solver contradictory constraints and report
        'no lineup satisfies them at all'."""
        with pytest.raises(SystemExit):
            optimize.parse_shape("6-3+2")

    def test_junk_fails_loudly_rather_than_silently_building_nothing(self):
        for bad in ("x", "3-x", "0"):
            with pytest.raises(SystemExit):
                optimize.parse_shape(bad)


class TestParseStacks:
    def test_it_reads_team_and_count(self):
        assert optimize.parse_stacks("BUF:3,kan:2") == {"BUF": 3, "KAN": 2}

    def test_a_missing_count_fails_loudly(self):
        with pytest.raises(SystemExit):
            optimize.parse_stacks("BUF")


class TestAttachBands:
    def _frame(self, position, proj):
        return pd.DataFrame([{"Name": "A", "Pos": position, "Proj": proj}])

    def test_bands_come_from_the_fitted_table(self):
        from nfl.projections import BAND_FIT
        out = optimize.attach_bands(self._frame("WR", 10.0))
        slope, intercept = BAND_FIT["WR"]["ceiling"]
        assert out["Ceiling"].iloc[0] == pytest.approx(slope * 10.0 + intercept, abs=0.01)

    def test_a_defense_does_not_borrow_the_receiver_band(self):
        """`BAND_FIT` is fitted on skill positions only. Falling through to WR would claim
        a fit that was never measured on a defense."""
        dst = optimize.attach_bands(self._frame("DST", 8.0))
        wr = optimize.attach_bands(self._frame("WR", 8.0))
        assert dst["Ceiling"].iloc[0] != wr["Ceiling"].iloc[0]

    def test_a_supplied_band_is_never_overwritten(self):
        frame = self._frame("WR", 10.0)
        frame["Ceiling"] = 99.0
        assert optimize.attach_bands(frame)["Ceiling"].iloc[0] == 99.0

    def test_a_floor_never_goes_negative(self):
        """The fitted floors have negative intercepts, so a low projection crosses zero.
        A negative floor is not a score anyone can post."""
        out = optimize.attach_bands(self._frame("WR", 0.5))
        assert out["Floor"].iloc[0] >= 0.0

    def test_no_projection_leaves_the_bands_blank(self):
        """Blank means "not projected", which is a different claim from "projected to
        score nothing" -- the same rule the report layer follows."""
        out = optimize.attach_bands(self._frame("WR", np.nan))
        assert pd.isna(out["Ceiling"].iloc[0]) and pd.isna(out["Floor"].iloc[0])


class TestGenerateDefaults:
    def test_an_unset_flag_defers_to_the_optimizer_default(self):
        """Passing None *overrides* the solver's default rather than deferring to it, and
        `randomness` is then compared against a number -- so the flag nobody set is the one
        that crashes the run. It did.
        """
        import sys
        sys.path.insert(0, "tests/nfl")
        from nfl_fixtures import make_slate

        slate = make_slate(n_games=3, seed=4)
        lineups, _, _ = optimize.generate(slate, n_lineups=1, randomness=None,
                                          max_overlap=None, seed=3)
        assert len(lineups) == 1


class TestStackShapesHold:
    def _slate(self):
        import sys
        sys.path.insert(0, "tests/nfl")
        from nfl_fixtures import make_slate
        return make_slate(n_games=4, seed=11)

    def test_the_anchor_quarterback_is_the_one_pinned(self):
        slate = self._slate()
        lineups, _, _ = optimize.generate(slate, n_lineups=3, shapes=[(3, 1, 0)], seed=5)
        assert lineups
        for lineup in lineups:
            players = lineup["players"]
            quarterback = players[players["Pos"] == "QB"].iloc[0]
            assert quarterback["Team"] == lineup["anchor"]

    def test_the_anchor_supplies_the_requested_pass_game_count(self):
        slate = self._slate()
        lineups, _, _ = optimize.generate(slate, n_lineups=3, shapes=[(3, 1, 0)], seed=5)
        for lineup in lineups:
            players = lineup["players"]
            pass_game = players[players["Pos"].isin(opt.PASS_GAME_POSITIONS)]
            assert (pass_game["Team"] == lineup["anchor"]).sum() >= 3

    def test_a_bare_shape_permits_a_bringback_rather_than_forbidding_one(self):
        """'3' asks for three from the anchor. It does not say the opposing side is banned,
        and reading it that way would quietly reject legal lineups."""
        shapes = optimize.parse_shape("3")
        assert shapes == [(3, 0, 0)]

    def test_lineups_in_a_set_are_distinct(self):
        slate = self._slate()
        lineups, _, _ = optimize.generate(slate, n_lineups=4, shapes=[(3, 1, 0)], seed=9)
        seen = {frozenset(l["players"]["Name"]) for l in lineups}
        assert len(seen) == len(lineups)


class TestReports:
    def _lineups(self):
        import sys
        sys.path.insert(0, "tests/nfl")
        from nfl_fixtures import make_slate
        slate = make_slate(n_games=4, seed=2)
        lineups, _, _ = optimize.generate(slate, n_lineups=3, seed=1)
        return lineups

    def test_lineup_rows_are_one_row_per_player_per_lineup(self):
        lineups = self._lineups()
        rows = optimize.lineup_rows(lineups)
        assert len(rows) == len(lineups) * opt.ROSTER_SIZE
        assert set(rows["Lineup"]) == set(range(1, len(lineups) + 1))

    def test_exposure_is_written_every_run_not_on_request(self):
        """The MLB twin caught 54% exposure on a 1.6%-owned punt. Nobody goes looking for
        an exposure they do not already suspect, so it is produced unconditionally."""
        lineups = self._lineups()
        players, teams = optimize.exposure_rows(lineups)
        assert not players.empty and not teams.empty
        assert players["Exposure%"].max() <= 100.0
        assert players["Lineups"].max() <= len(lineups)

    def test_exposure_percentages_are_out_of_the_lineup_count(self):
        lineups = self._lineups()
        players, _ = optimize.exposure_rows(lineups)
        row = players.iloc[0]
        assert row["Exposure%"] == pytest.approx(row["Lineups"] / len(lineups) * 100, abs=0.1)


class TestExposure:
    """Exposure is a property of the SET of lineups, not of any one lineup."""

    def _slate(self):
        import sys
        sys.path.insert(0, "tests/nfl")
        from nfl_fixtures import make_slate
        return make_slate(n_games=5, seed=7)

    def test_a_cap_is_held(self):
        slate = self._slate()
        name = slate[slate["Pos"] == "WR"]["Name"].iloc[0]
        lineups, _, _ = opt.optimize(slate, n_lineups=20, seed=3,
                                     exposure={name: (0, 3)})
        used = sum(name in set(l["players"]["Name"]) for l in lineups)
        assert used <= 3

    def test_a_minimum_is_met(self):
        slate = self._slate()
        name = slate[slate["Pos"] == "WR"]["Name"].iloc[1]
        lineups, _, missing = opt.optimize(slate, n_lineups=20, seed=3,
                                           exposure={name: (10, 20)})
        used = sum(name in set(l["players"]["Name"]) for l in lineups)
        assert used >= 10
        assert not missing["exposure_unmet"]

    def test_a_minimum_is_paced_not_front_loaded(self):
        """Ceil makes every minimum due on the first lineup, so a long Min% list fills the
        roster in file order and leaves the solver nothing to decide. The pace rounds to
        nearest, which leaves slack at both ends of the set."""
        slate = self._slate()
        name = slate[slate["Pos"] == "WR"]["Name"].iloc[0]
        lineups, _, _ = opt.optimize(slate, n_lineups=10, seed=5,
                                     exposure={name: (5, 10)})
        first_half = sum(name in set(l["players"]["Name"]) for l in lineups[:5])
        assert 1 <= first_half <= 5      # spread, not all at one end

    def test_team_stack_exposure_is_counted_the_way_the_report_counts_it(self):
        slate = self._slate()
        team = sorted(slate["Team"].unique())[0]
        lineups, _, missing = opt.optimize(slate, n_lineups=12, seed=5,
                                           team_exposure={team: (6, 12)})
        stacked = 0
        for lineup in lineups:
            players = lineup["players"]
            pass_game = players[players["Roster"].isin(("QB", "WR", "TE", "FLEX"))]
            if (pass_game["Team"] == team).sum() >= opt.STACK_EXPOSURE_AT:
                stacked += 1
        assert stacked >= 6
        # what the solver counted and what the frame shows must agree
        assert missing["team_stacks"][team] == stacked

    def test_an_exposure_on_someone_not_on_the_slate_is_ignored_not_fatal(self):
        slate = self._slate()
        lineups, _, _ = opt.optimize(slate, n_lineups=3, seed=1,
                                     exposure={"Nobody At All": (3, 3)})
        assert len(lineups) == 3


class TestParseExposure:
    def test_percentages_become_lineup_counts(self):
        assert optimize.parse_exposure("BUF:20-60%", 20) == {"BUF": (4, 12)}

    def test_a_bare_value_is_both_ends(self):
        assert optimize.parse_exposure("KAN:25%", 20) == {"KAN": (5, 5)}

    def test_junk_fails_loudly(self):
        with pytest.raises(SystemExit):
            optimize.parse_exposure("BUF", 20)
