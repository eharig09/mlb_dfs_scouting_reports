"""Cold-start priors: weeks 1-3, when nobody has played yet.

Named `test_nfl_*` and carrying no conftest, per the collision documented in
tests/nfl/nfl_fixtures.py.

Everything here is synthetic. What is under test is the *selection* logic — who gets which
prior, and what each prior is composed of — not the fitted constants, which come from
`fit_draft_curves` against real history.
"""

import numpy as np
import pandas as pd
import pytest

from nfl import coldstart as cs


def _roster(rows):
    return pd.DataFrame([{
        "gsis_id": r.get("id"), "full_name": r.get("name"), "position": r.get("pos"),
        "team": r.get("team"), "status": r.get("status", "ACT"),
        "years_exp": r.get("exp"), "draft_number": r.get("pick"), "pff_id": None,
    } for r in rows])


def _depth(rows, dt="2026-08-19T07:43:54Z"):
    return pd.DataFrame([{
        "dt": r.get("dt", dt), "team": r.get("team"), "gsis_id": r.get("id"),
        "pos_abb": r.get("pos"), "pos_rank": r.get("rank"), "pos_slot": 1,
    } for r in rows])


def _history(player_id, games, carries=0.0, targets=0.0, attempts=0.0, season=2025):
    return pd.DataFrame([{
        "player_id": player_id, "season": season, "week": w + 1,
        "attempts": attempts, "carries": carries, "targets": targets,
    } for w in range(games)])


class TestDraftCurves:
    def test_earlier_picks_get_more_of_everything(self):
        """The whole point of draft capital as a prior — it must be monotone."""
        for position, metric in (("RB", "carries"), ("WR", "targets"),
                                 ("TE", "targets"), ("QB", "attempts")):
            early = cs.rookie_usage(position, 5)[metric]
            mid = cs.rookie_usage(position, 60)[metric]
            late = cs.rookie_usage(position, 220)[metric]
            assert early > mid > late, (position, metric, early, mid, late)
            assert cs.expected_games(position, 5) >= cs.expected_games(position, 220)

    def test_usage_is_per_game_played_not_diluted_by_availability(self):
        """A pick-5 back must read as a starter's workload, not a season average.

        Fitting usage as season-total/17 folds missed games into the rate and reads a
        first-round RB at ~11.8 carries instead of ~17.5. Availability is a separate term
        precisely so that "probably starts, 17 carries" and "17 carries if he ever plays"
        stay distinguishable.
        """
        assert cs.rookie_usage("RB", 5)["carries"] > 15.0
        assert cs.rookie_usage("QB", 5)["attempts"] > 25.0

    def test_availability_is_bounded_by_the_season(self):
        for position in cs.POSITIONS:
            for pick in (1, 5, 50, 257, 400):
                assert 0.0 <= cs.expected_games(position, pick) <= cs.SEASON_GAMES

    def test_an_undrafted_rookie_sits_past_the_last_pick(self):
        prior = cs.rookie_prior("WR", None)
        assert prior["pick"] == cs.UNDRAFTED_PICK
        assert prior["targets"] < cs.rookie_prior("WR", 200)["targets"]

    def test_a_position_that_does_not_accumulate_a_metric_gets_zero(self):
        """RBs do not throw. An all-zero column is not something to fit a line to."""
        assert cs.rookie_usage("RB", 5)["attempts"] == 0.0
        assert cs.rookie_usage("TE", 5)["carries"] == 0.0


class TestRookieGate:
    """The bug that made this a gate rather than a lookup."""

    def test_a_veteran_is_not_priced_off_the_pick_he_was_drafted_at(self):
        """`draft_number` stays on the roster row for a player's whole career.

        452 of the 915 rostered 2026 skill players are veterans carrying one. Reading it
        without an experience check turned Phillip Dorsett — eleven years in, taken 29th in
        2015 — into a rookie WR1 on 5.0 targets a game.
        """
        universe = cs.roster_universe(_roster([
            {"id": "VET", "name": "Phillip Dorsett", "pos": "WR", "team": "LV",
             "exp": 11, "pick": 29},
        ]))
        priors = cs.build_priors(universe, pd.DataFrame())
        row = priors.iloc[0]
        assert row["source"] == "replacement"
        assert row["targets"] == cs.REPLACEMENT_USAGE["targets"]

    def test_a_true_rookie_does_get_the_draft_prior(self):
        universe = cs.roster_universe(_roster([
            {"id": "RK", "name": "Rook", "pos": "WR", "team": "SF", "exp": 0, "pick": 33},
        ]))
        row = cs.build_priors(universe, pd.DataFrame()).iloc[0]
        assert row["source"] == "draft"
        assert row["pick"] == 33
        assert row["targets"] > cs.REPLACEMENT_USAGE["targets"]

    def test_a_second_year_player_who_never_played_drops_to_replacement(self):
        """Failing to earn snaps is evidence against the draft slot, not absent evidence."""
        universe = cs.roster_universe(_roster([
            {"id": "Y2", "name": "Bust", "pos": "QB", "team": "HOU", "exp": 1, "pick": 197},
        ]))
        assert cs.build_priors(universe, pd.DataFrame()).iloc[0]["source"] == "replacement"


class TestUniverse:
    def test_team_comes_from_the_roster_not_the_last_box_score(self):
        """24% of skill players who played in 2025 are on a different team in 2026.

        Reading team off the most recent game puts them on the old club — wrong implied
        total, wrong opponent, and on a two-game slate the wrong game entirely.
        """
        universe = cs.roster_universe(_roster([
            {"id": "MOVED", "name": "Kirk Cousins", "pos": "QB", "team": "LV",
             "exp": 14, "pick": 102},
        ]))
        history = _history("MOVED", 12, attempts=30.0)     # all of it for the old club
        priors = cs.build_priors(universe, history)
        assert priors.iloc[0]["team"] == "LV"

    def test_a_player_with_no_game_history_still_appears(self):
        """376 of 915 rostered skill players have no 2025 game. They cannot be invisible."""
        universe = cs.roster_universe(_roster([
            {"id": "NEW", "name": "Rook", "pos": "RB", "team": "SF", "exp": 0, "pick": 90},
        ]))
        priors = cs.build_priors(universe, pd.DataFrame())
        assert len(priors) == 1
        assert priors.iloc[0]["carries"] > 0

    def test_inactive_players_are_left_off(self):
        roster = _roster([
            {"id": "A", "name": "Active", "pos": "RB", "team": "SF", "exp": 3, "pick": 40},
            {"id": "R", "name": "Reserve", "pos": "RB", "team": "SF", "exp": 3, "pick": 41},
        ])
        roster.loc[roster.gsis_id == "R", "status"] = "RES"
        assert set(cs.roster_universe(roster)["player_id"]) == {"A"}

    def test_only_the_latest_depth_chart_snapshot_is_used(self):
        """The file is a time series — 449k rows for 2026 across March-to-August snapshots.

        Reading it whole mixes a player's camp position with his current one.
        """
        depth = _depth([
            {"id": "P", "team": "SF", "pos": "RB", "rank": 4, "dt": "2026-03-22T06:38:42Z"},
            {"id": "P", "team": "SF", "pos": "RB", "rank": 1, "dt": "2026-08-19T07:43:54Z"},
        ])
        universe = cs.roster_universe(
            _roster([{"id": "P", "name": "Riser", "pos": "RB", "team": "SF",
                      "exp": 2, "pick": 88}]), depth)
        assert universe.iloc[0]["depth_rank"] == 1


class TestDepthRank:
    def test_a_demoted_veteran_is_scaled_down(self):
        """Aidan O'Connell threw 27.6 a game in the starts he made and is QB3 now."""
        universe = cs.roster_universe(
            _roster([{"id": "QB3", "name": "Backup", "pos": "QB", "team": "LV",
                      "exp": 4, "pick": 135}]),
            _depth([{"id": "QB3", "team": "LV", "pos": "QB", "rank": 3}]))
        priors = cs.build_priors(universe, _history("QB3", 10, attempts=27.6))
        row = priors.iloc[0]
        assert row["source"] == "prior_season"
        assert row["attempts"] == pytest.approx(27.6 * cs.DEPTH_MULTIPLIER["QB"][3], rel=1e-6)

    def test_a_rank_one_veteran_is_untouched(self):
        universe = cs.roster_universe(
            _roster([{"id": "S", "name": "Starter", "pos": "RB", "team": "SF",
                      "exp": 6, "pick": 20}]),
            _depth([{"id": "S", "team": "SF", "pos": "RB", "rank": 1}]))
        priors = cs.build_priors(universe, _history("S", 15, carries=17.8))
        assert priors.iloc[0]["carries"] == pytest.approx(17.8, rel=1e-6)

    def test_a_rookie_is_deliberately_not_depth_adjusted(self):
        """The draft curve already averages across roles at that pick.

        Multiplying it by a rank factor would count the same information twice. Composing
        them honestly needs an expected-rank-given-pick term that is not fitted.
        """
        universe = cs.roster_universe(
            _roster([{"id": "RK", "name": "Rook", "pos": "RB", "team": "SF",
                      "exp": 0, "pick": 90}]),
            _depth([{"id": "RK", "team": "SF", "pos": "RB", "rank": 3}]))
        row = cs.build_priors(universe, pd.DataFrame()).iloc[0]
        assert row["carries"] == pytest.approx(cs.rookie_usage("RB", 90)["carries"], rel=1e-6)

    def test_a_missing_rank_is_neutral(self):
        assert cs.depth_multiplier("WR", None) == 1.0
        assert cs.depth_multiplier("WR", np.nan) == 1.0
        assert cs.depth_multiplier("XX", 2) == 1.0

    def test_rank_four_and_beyond_share_the_rank_three_bucket(self):
        for position in cs.DEPTH_MULTIPLIER:
            assert cs.depth_multiplier(position, 9) == cs.DEPTH_MULTIPLIER[position][3]


class TestExpectedOpportunity:
    def test_availability_is_folded_back_in_for_ranking(self):
        """Conditional usage alone ranks a fourth-string QB among the starters.

        He would throw if he played; the whole question is whether he plays.
        """
        universe = cs.roster_universe(_roster([
            {"id": "QB4", "name": "Fourth", "pos": "QB", "team": "LV", "exp": 0, "pick": None},
            {"id": "RB1", "name": "Back", "pos": "RB", "team": "LV", "exp": 0, "pick": 90},
        ]))
        priors = cs.build_priors(universe, pd.DataFrame()).set_index("player_name")
        # The QB's raw attempts exceed the back's carries...
        assert priors.loc["Fourth", "attempts"] > priors.loc["Back", "carries"]
        # ...but almost none of it survives availability.
        assert priors.loc["Fourth", "expected_opportunity"] < priors.loc["Back", "expected_opportunity"]

    def test_expected_is_usage_times_availability(self):
        universe = cs.roster_universe(_roster([
            {"id": "RK", "name": "Rook", "pos": "WR", "team": "SF", "exp": 0, "pick": 33},
        ]))
        row = cs.build_priors(universe, pd.DataFrame()).iloc[0]
        assert row["expected_targets"] == pytest.approx(row["targets"] * row["availability"])

    def test_a_veterans_availability_is_not_his_game_count(self):
        """Game count is mostly a statement about role, and role is already in the depth
        factor. Deriving availability from it too demotes the same player twice."""
        universe = cs.roster_universe(
            _roster([{"id": "B", "name": "Backup", "pos": "QB", "team": "HOU",
                      "exp": 5, "pick": 67}]),
            _depth([{"id": "B", "team": "HOU", "pos": "QB", "rank": 2}]))
        row = cs.build_priors(universe, _history("B", 10, attempts=21.9)).iloc[0]
        assert row["availability"] == 1.0
        assert row["attempts"] == pytest.approx(21.9 * cs.DEPTH_MULTIPLIER["QB"][2], rel=1e-6)


class TestDegenerateInput:
    def test_an_empty_roster_is_an_empty_frame(self):
        assert cs.roster_universe(pd.DataFrame()).empty
        assert cs.build_priors(pd.DataFrame(), pd.DataFrame()).empty

    def test_no_depth_chart_at_all_is_fine(self):
        universe = cs.roster_universe(
            _roster([{"id": "A", "name": "A", "pos": "RB", "team": "SF", "exp": 2, "pick": 50}]),
            None)
        assert len(universe) == 1
        assert cs.build_priors(universe, pd.DataFrame()).iloc[0]["source"] == "replacement"

    def test_non_skill_positions_are_dropped(self):
        universe = cs.roster_universe(_roster([
            {"id": "K", "name": "Kicker", "pos": "K", "team": "SF", "exp": 4, "pick": 150},
        ]))
        assert universe.empty


class TestProjectionPriors:
    """The bridge from cold start into the projection model.

    `build_priors` always produced a per-player opportunity prior and `project_week` always
    shrank toward a per-*position* one. Both were called "priors", both were passed as
    `priors=`, and they were never connected -- so the model had no cold-start handling at
    all. Measured on held-out 2025 it beat a season average by 0.434 MAE at 16+ games of
    history and LOST to it below that, across a quarter of all rows.
    """

    def _universe(self):
        import pandas as pd
        return pd.DataFrame([
            {"player_id": "A", "player_name": "Vet", "position": "WR", "team": "BUF",
             "depth_rank": 1},
            {"player_id": "B", "player_name": "Rook", "position": "RB", "team": "KAN",
             "depth_rank": 2},
        ])

    def _history(self):
        import pandas as pd
        return pd.DataFrame([
            {"player_id": "A", "season": 2024, "week": w, "attempts": 0,
             "carries": 0, "targets": 8} for w in range(1, 15)])

    def test_it_keys_by_player_and_carries_the_three_volume_terms(self):
        out = cs.projection_priors(self._universe(), self._history())
        assert set(out) == {"A", "B"}
        for prior in out.values():
            assert set(prior) == {"attempts", "carries", "targets"}
            assert all(isinstance(v, float) for v in prior.values())

    def test_it_uses_conditional_usage_not_expected(self):
        """The model shrinks per-game-*played* rates, so the prior must be on the same
        footing. `expected_*` folds availability in and belongs on a board that ranks
        players, not in a shrinkage target."""
        universe, history = self._universe(), self._history()
        full = cs.build_priors(universe, history)
        bridged = cs.projection_priors(universe, history)
        row = full[full["player_id"] == "A"].iloc[0]
        assert bridged["A"]["targets"] == pytest.approx(float(row["targets"]))
        # and that it is NOT the availability-weighted number, when they differ
        if float(row["availability"]) < 1.0:
            assert bridged["A"]["targets"] != pytest.approx(float(row["expected_targets"]))

    def test_an_empty_universe_is_an_empty_map_not_a_crash(self):
        import pandas as pd
        assert cs.projection_priors(pd.DataFrame(), pd.DataFrame()) == {}


class TestProjectWeekHonoursPlayerPriors:
    def test_a_personal_volume_prior_overrides_the_positional_one(self):
        """Only the three volume terms. Cold start projects opportunity, not efficiency,
        and the rate priors stay positional because efficiency does not persist well enough
        to personalise -- YPC self-correlates at +0.27, TD rates at 0.03-0.24."""
        from nfl import projections

        positional = {"WR": {"attempts": 0.0, "carries": 0.5, "targets": 5.0,
                             "rec_ypt": 8.0, "catch_rate": 0.65}}
        personal = {"P1": {"attempts": 0.0, "carries": 0.0, "targets": 11.0}}
        prior = dict(positional["WR"])
        for metric in ("attempts", "carries", "targets"):
            value = personal["P1"].get(metric)
            if value is not None:
                prior[metric] = float(value)
        assert prior["targets"] == 11.0
        assert prior["rec_ypt"] == 8.0        # efficiency untouched


class TestActiveOnly:
    """`active_only` is a question about when you are standing, not a quality filter."""

    def _roster(self):
        import pandas as pd
        return pd.DataFrame([
            {"gsis_id": "A", "full_name": "Starter", "position": "WR", "team": "BUF",
             "status": "ACT", "years_exp": 4, "draft_number": 40},
            {"gsis_id": "B", "full_name": "Cut in December", "position": "WR",
             "team": "KAN", "status": "CUT", "years_exp": 3, "draft_number": 120},
            {"gsis_id": "C", "full_name": "Practice squad", "position": "RB",
             "team": "MIA", "status": "DEV", "years_exp": 1, "draft_number": None},
        ])

    def test_a_live_board_keeps_only_active_players(self):
        """A cut player is not a DFS option today, and that is the default."""
        out = cs.roster_universe(self._roster())
        assert list(out["player_id"]) == ["A"]

    def test_history_keeps_everyone_the_roster_knows(self):
        """A roster file carries one status per player, and for a completed season that is
        a season-*end* snapshot -- a player who started eight games and was cut in December
        reads CUT. Measured on 2025: the active filter drops 197 players who actually
        played that season, a third of everyone who took a snap.
        """
        out = cs.roster_universe(self._roster(), active_only=False)
        assert set(out["player_id"]) == {"A", "B", "C"}

    def test_a_roster_without_a_status_column_is_not_filtered_away(self):
        import pandas as pd
        roster = self._roster().drop(columns=["status"])
        assert len(cs.roster_universe(roster)) == 3
