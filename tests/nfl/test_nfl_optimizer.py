"""NFL Classic optimizer.

The class taxonomy is ported straight from `tests/test_optimizer.py` -- roster validity,
locks and excludes, stacks, overlap, guards -- because none of those categories were about
baseball. What changes is the content of the assertions, and one class is genuinely new:
`TestFlex` exists because FLEX is the constraint MLB has no equivalent of.
"""

import pandas as pd
import pytest

# Imported rather than supplied by a conftest -- see nfl_fixtures for why there is none.
from nfl_fixtures import make_slate, slate, small_slate           # noqa: F401

from nfl.optimizer import (DST_CONFLICT_MAX, MIN_GAMES_REPRESENTED, OBJECTIVES, ROSTER,
                           ROSTER_SIZE, OptimizerError, _fits_seats, build_pool,
                           eligible_positions, lineup_conflicts, optimize)
from nfl.scoring import DK_SALARY_CAP, FLEX_POSITIONS


def roster_counts(lineup):
    return lineup["players"]["Roster"].value_counts().to_dict()


def assert_legal(lineup):
    """Every rule DK will reject an entry for, checked in one place."""
    players = lineup["players"]
    assert len(players) == ROSTER_SIZE
    assert roster_counts(lineup) == ROSTER
    assert lineup["salary"] <= DK_SALARY_CAP
    # No player twice, and every player actually eligible at the seat he was given.
    assert players["Name"].nunique() == ROSTER_SIZE
    for _, row in players.iterrows():
        assert row["Roster"] in eligible_positions(row), f"{row['Name']} at {row['Roster']}"
    assert players["Game"].nunique() >= MIN_GAMES_REPRESENTED


class TestRosterValidity:
    def test_single_lineup_is_legal(self, slate):
        lineups, _, _ = optimize(slate, n_lineups=1, seed=1)
        assert len(lineups) == 1
        assert_legal(lineups[0])

    def test_every_lineup_in_a_set_is_legal(self, slate):
        lineups, _, _ = optimize(slate, n_lineups=10, seed=2)
        assert len(lineups) == 10
        for lineup in lineups:
            assert_legal(lineup)

    @pytest.mark.parametrize("objective", sorted(OBJECTIVES))
    def test_all_objectives_produce_legal_lineups(self, slate, objective):
        if objective == "leverage":
            slate = slate.assign(**{"Lev Score": slate["Ceiling"] - 0.35 * slate["Own%"]})
        lineups, _, _ = optimize(slate, n_lineups=2, objective=objective, seed=3)
        for lineup in lineups:
            assert_legal(lineup)

    def test_unknown_objective_is_rejected(self, slate):
        with pytest.raises(OptimizerError, match="objective must be one of"):
            optimize(slate, objective="vibes")

    def test_a_missing_objective_column_is_reported_not_crashed(self, slate):
        with pytest.raises(OptimizerError, match="Lev Score"):
            optimize(slate, objective="leverage")

    def test_the_cap_actually_binds(self, slate):
        lineups, _, _ = optimize(slate, n_lineups=5, seed=4, randomness=0)
        for lineup in lineups:
            assert lineup["salary"] <= DK_SALARY_CAP

    def test_two_games_are_represented(self, slate):
        lineups, _, _ = optimize(slate, n_lineups=5, seed=5)
        for lineup in lineups:
            assert lineup["players"]["Game"].nunique() >= 2


class TestFlex:
    """FLEX is the whole reason the assignment model had to come across."""

    def test_flex_is_filled_by_a_running_back_receiver_or_tight_end(self, slate):
        lineups, _, _ = optimize(slate, n_lineups=8, seed=6)
        for lineup in lineups:
            flex = lineup["players"][lineup["players"]["Roster"] == "FLEX"]
            assert len(flex) == 1
            assert flex.iloc[0]["Pos"] in FLEX_POSITIONS

    def test_a_quarterback_is_never_eligible_at_flex(self):
        assert eligible_positions({"DK Pos": "QB"}) == {"QB"}

    def test_a_defense_is_never_eligible_at_flex(self):
        assert eligible_positions({"DK Pos": "DST"}) == {"DST"}
        assert eligible_positions({"DK Pos": "D/ST"}) == {"DST"}

    def test_skill_positions_carry_both_seats(self):
        for position in ("RB", "WR", "TE"):
            assert eligible_positions({"DK Pos": position}) == {position, "FLEX"}

    def test_a_flex_eligible_player_fills_only_one_slot(self, slate):
        """The failure the assignment model exists to prevent: one receiver satisfying both
        a WR seat and the FLEX seat, which frees a roster spot that does not exist."""
        lineups, _, _ = optimize(slate, n_lineups=6, seed=7)
        for lineup in lineups:
            players = lineup["players"]
            assert players["Name"].nunique() == len(players)
            # Exactly four of the nine are receivers-by-seat only if FLEX took a WR; the
            # invariant that always holds is that seat counts match the roster exactly.
            assert roster_counts(lineup) == ROSTER

    def test_position_counts_never_undershoot_the_dedicated_seats(self, slate):
        """A lineup must hold at least 2 RB / 3 WR / 1 TE by POSITION, whatever FLEX did."""
        lineups, _, _ = optimize(slate, n_lineups=8, seed=8)
        for lineup in lineups:
            by_position = lineup["players"]["Pos"].value_counts()
            assert by_position.get("QB", 0) == 1
            assert by_position.get("DST", 0) == 1
            assert by_position.get("RB", 0) >= 2
            assert by_position.get("WR", 0) >= 3
            assert by_position.get("TE", 0) >= 1
            assert by_position.get("RB", 0) + by_position.get("WR", 0) + \
                by_position.get("TE", 0) == 7

    def test_seat_matching_rejects_an_unseatable_group(self):
        # Four tight ends cannot be rostered: one TE seat plus one FLEX is two.
        assert _fits_seats([{"TE", "FLEX"}] * 2)
        assert not _fits_seats([{"TE", "FLEX"}] * 3)

    def test_seat_matching_accepts_a_full_legal_roster(self):
        groups = ([{"QB"}] + [{"RB", "FLEX"}] * 2 + [{"WR", "FLEX"}] * 3
                  + [{"TE", "FLEX"}] + [{"WR", "FLEX"}] + [{"DST"}])
        assert _fits_seats(groups)


class TestLocksAndExcludes:
    def test_lock_appears_in_every_lineup(self, slate):
        name = slate[slate["Pos"] == "WR"].iloc[3]["Name"]
        lineups, _, _ = optimize(slate, n_lineups=5, locks=[name], seed=9)
        for lineup in lineups:
            assert name in set(lineup["players"]["Name"])
            assert_legal(lineup)

    def test_exclude_never_appears(self, slate):
        top = slate.sort_values("Ceiling", ascending=False).iloc[0]["Name"]
        lineups, _, _ = optimize(slate, n_lineups=5, excludes=[top], seed=10)
        for lineup in lineups:
            assert top not in set(lineup["players"]["Name"])

    def test_a_lock_survives_a_threshold_that_would_drop_him(self, slate):
        """Checked on the pool rather than end to end. A min_proj high enough to threaten
        the weakest player also wipes out every defense on the board -- DST projections top
        out around 10 while a quarterback clears 25 -- so an end-to-end version of this
        test fails on roster feasibility rather than on the rule it names."""
        weak = slate.sort_values("Proj").iloc[0]["Name"]
        pool, lock_idx, _ = build_pool(slate, locks=[weak], min_proj=12.0)
        assert weak in set(pool["Name"])
        assert lock_idx and pool.loc[lock_idx[0], "Name"] == weak

    def test_a_threshold_drops_everyone_else_below_it(self, slate):
        pool, _, _ = build_pool(slate, min_proj=12.0)
        assert (pd.to_numeric(pool["Proj"]) >= 12.0).all()

    def test_missing_names_are_reported_not_raised(self, slate):
        _, _, missing = optimize(slate, n_lineups=1, locks=["Nobody At All"], seed=12)
        assert missing["locks"] == ["Nobody At All"]

    def test_too_many_locks_is_an_error(self, slate):
        names = slate["Name"].head(ROSTER_SIZE + 1).tolist()
        with pytest.raises(OptimizerError, match="exceed"):
            optimize(slate, locks=names)

    def test_unseatable_locks_are_reported_before_the_solver(self, slate):
        """Three quarterbacks are three players and still cannot share one seat. Without
        the seat check this reaches the solver and comes back as a bare infeasibility."""
        quarterbacks = slate[slate["Pos"] == "QB"]["Name"].head(3).tolist()
        with pytest.raises(OptimizerError, match="seated"):
            optimize(slate, locks=quarterbacks)

    def test_a_pin_holds_a_player_to_one_exact_seat(self, slate):
        name = slate[slate["Pos"] == "WR"].iloc[0]["Name"]
        lineups, _, _ = optimize(slate, n_lineups=3, pins={name: "FLEX"}, seed=13)
        for lineup in lineups:
            players = lineup["players"]
            assert players[players["Name"] == name].iloc[0]["Roster"] == "FLEX"
            assert_legal(lineup)

    def test_a_pin_to_an_ineligible_seat_is_rejected(self, slate):
        quarterback = slate[slate["Pos"] == "QB"].iloc[0]["Name"]
        with pytest.raises(OptimizerError, match="not eligible"):
            optimize(slate, pins={quarterback: "FLEX"})

    def test_a_pin_to_an_unknown_seat_is_rejected(self, slate):
        name = slate.iloc[0]["Name"]
        with pytest.raises(OptimizerError, match="unknown roster slot"):
            optimize(slate, pins={name: "K"})


class TestStackConstraints:
    def test_explicit_stack_is_honoured(self, slate):
        team = slate.iloc[0]["Team"]
        lineups, _, _ = optimize(slate, n_lineups=3, stacks={team: 3}, seed=14)
        for lineup in lineups:
            players = lineup["players"]
            pass_game = players[(players["Team"] == team)
                                & players["Pos"].isin({"QB", "WR", "TE"})]
            assert len(pass_game) >= 3
            assert_legal(lineup)

    def test_a_running_back_does_not_count_toward_his_own_stack(self, slate):
        """PASS_GAME_POSITIONS excludes RB on purpose -- a rushing touchdown is a drive
        that did not end in a passing touchdown."""
        team = slate.iloc[0]["Team"]
        lineups, _, _ = optimize(slate, n_lineups=2, stacks={team: 3}, seed=15)
        for lineup in lineups:
            players = lineup["players"]
            qb_wr_te = players[(players["Team"] == team)
                               & players["Pos"].isin({"QB", "WR", "TE"})]
            assert len(qb_wr_te) >= 3

    def test_a_stack_bigger_than_the_pool_says_how_many_there_are(self, slate):
        team = slate.iloc[0]["Team"]
        with pytest.raises(OptimizerError, match="only .* pass-game players"):
            optimize(slate, stacks={team: 20})

    def test_team_aliases_fold(self, slate):
        """A typed 'JAX' must find the players the slate files under 'JAC'."""
        renamed = slate.copy()
        renamed.loc[renamed["Team"] == renamed.iloc[0]["Team"], "Team"] = "JAC"
        lineups, _, _ = optimize(renamed, n_lineups=1, stacks={"JAX": 2}, seed=16)
        players = lineups[0]["players"]
        assert len(players[(players["Team"] == "JAC")
                           & players["Pos"].isin({"QB", "WR", "TE"})]) >= 2

    def test_the_per_team_cap_binds_when_tightened(self, slate):
        lineups, _, _ = optimize(slate, n_lineups=5, max_players_per_team=2, seed=17)
        for lineup in lineups:
            players = lineup["players"]
            skill = players[players["Roster"] != "DST"]
            assert skill["Team"].value_counts().max() <= 2
            assert_legal(lineup)


class TestDefenseConflict:
    def test_a_defense_is_not_stacked_against_by_default(self, slate):
        lineups, _, _ = optimize(slate, n_lineups=10, seed=18)
        for lineup in lineups:
            assert lineup_conflicts(lineup["players"], DST_CONFLICT_MAX) == []

    def test_tightening_the_cap_forbids_any_pairing(self, slate):
        lineups, _, _ = optimize(slate, n_lineups=5, dst_conflict_max=0, seed=19)
        for lineup in lineups:
            assert lineup_conflicts(lineup["players"], 0) == []
            assert_legal(lineup)

    def test_turning_it_off_still_produces_legal_lineups(self, slate):
        lineups, _, _ = optimize(slate, n_lineups=5, dst_conflict_max=None, seed=20)
        for lineup in lineups:
            assert_legal(lineup)


class TestOverlapConstraints:
    def test_overlap_cap_is_respected_pairwise(self, slate):
        lineups, _, _ = optimize(slate, n_lineups=8, max_overlap=4, seed=21)
        sets = [set(l["players"]["Name"]) for l in lineups]
        for i, first in enumerate(sets):
            for second in sets[i + 1:]:
                assert len(first & second) <= 4

    def test_the_dk_minimum_still_blocks_exact_duplicates(self, slate):
        lineups, _, _ = optimize(slate, n_lineups=6, max_overlap=ROSTER_SIZE - 1,
                                 randomness=0, seed=22)
        sets = [frozenset(l["players"]["Name"]) for l in lineups]
        assert len(set(sets)) == len(sets)

    def test_tighter_overlap_gives_more_distinct_players(self, slate):
        """Measured with the jitter OFF, because that is the only condition under which
        the overlap cap is the mechanism being tested. 19 / 22 / 27 distinct players at
        caps of 8 / 5 / 3 on this fixture."""
        distinct = lambda ls: len({n for l in ls for n in l["players"]["Name"]})
        loose, _, _ = optimize(slate, n_lineups=6, max_overlap=8, randomness=0, seed=23)
        tight, _, _ = optimize(slate, n_lineups=6, max_overlap=3, randomness=0, seed=23)
        assert distinct(tight) > distinct(loose)

    def test_jitter_alone_already_diversifies_a_small_set(self, slate):
        """Worth pinning down, because it says the overlap cap is doing much less than it
        looks like at these set sizes. With randomness at its default the six lineups are
        already more varied than a cap of 3 produces without it, so a test of the cap that
        leaves jitter on is measuring the jitter."""
        jittered, _, _ = optimize(slate, n_lineups=6, max_overlap=8, seed=23)
        strict, _, _ = optimize(slate, n_lineups=6, max_overlap=8, randomness=0, seed=23)
        distinct = lambda ls: len({n for l in ls for n in l["players"]["Name"]})
        assert distinct(jittered) > distinct(strict)

    def test_a_set_that_runs_out_returns_what_it_built(self, small_slate):
        lineups, _, _ = optimize(small_slate, n_lineups=200, max_overlap=2, seed=24)
        assert 0 < len(lineups) < 200
        for lineup in lineups:
            assert_legal(lineup)


class TestDeterminism:
    def test_the_same_seed_gives_the_same_set(self, slate):
        first, _, _ = optimize(slate, n_lineups=5, seed=99)
        second, _, _ = optimize(slate, n_lineups=5, seed=99)
        assert ([set(l["players"]["Name"]) for l in first]
                == [set(l["players"]["Name"]) for l in second])

    def test_different_seeds_differ(self, slate):
        first, _, _ = optimize(slate, n_lineups=5, seed=1)
        second, _, _ = optimize(slate, n_lineups=5, seed=2)
        assert ([set(l["players"]["Name"]) for l in first]
                != [set(l["players"]["Name"]) for l in second])

    def test_zero_randomness_is_the_single_best_lineup(self, slate):
        lineups, _, _ = optimize(slate, n_lineups=1, randomness=0, objective="proj", seed=1)
        best = lineups[0]
        again, _, _ = optimize(slate, n_lineups=1, randomness=0, objective="proj", seed=77)
        # With the jitter off the seed cannot matter: there is one optimum and both solves
        # must find it.
        assert best["proj"] == again[0]["proj"]
        assert set(best["players"]["Name"]) == set(again[0]["players"]["Name"])


class TestGuards:
    def test_a_pool_too_small_to_roster_is_refused(self, slate):
        with pytest.raises(OptimizerError, match="need"):
            optimize(slate.head(5))

    def test_an_empty_frame_is_refused(self, slate):
        with pytest.raises(OptimizerError):
            optimize(slate.iloc[0:0])

    def test_players_without_a_salary_are_dropped(self, slate):
        priced = slate.copy()
        priced.loc[priced["Pos"] == "QB", "Salary"] = pd.NA
        with pytest.raises(OptimizerError):
            optimize(priced)

    def test_contradictory_constraints_say_so(self, slate):
        team = slate.iloc[0]["Team"]
        with pytest.raises(OptimizerError, match="contradictory|only"):
            optimize(slate, stacks={team: 5}, max_players_per_team=1)

    def test_an_unrosterable_position_string_is_ignored(self, slate):
        odd = slate.copy()
        odd.loc[odd.index[0], "DK Pos"] = "K"
        lineups, pool, _ = optimize(odd, n_lineups=1, seed=25)
        assert odd.iloc[0]["Name"] not in set(pool["Name"])
        assert_legal(lineups[0])
