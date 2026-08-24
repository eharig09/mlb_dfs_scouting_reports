"""Roster validity and every constraint the optimizer claims to enforce.

These are the tests that stop a silent illegal lineup reaching DK. An invalid entry is
rejected at upload with no useful message, and the failure surfaces hours later.
"""

import collections

import numpy as np
import pandas as pd
import pytest

from dfs.optimizer import (MAX_HITTERS_PER_TEAM, ROSTER, ROSTER_SIZE, OptimizerError,
                           eligible_positions, lineup_conflicts, optimize, stack_shapes)
from dfs.scoring import DK_SALARY_CAP


def assert_legal(lineup):
    """Every DK Classic rule, checked on a finished lineup."""
    frame = lineup["players"]
    assert len(frame) == ROSTER_SIZE

    counts = frame["Roster"].value_counts().to_dict()
    for slot, required in ROSTER.items():
        assert counts.get(slot, 0) == required, f"{slot}: {counts.get(slot, 0)} != {required}"

    # Every player must actually be eligible at the slot they were given.
    for _, row in frame.iterrows():
        assert row["Roster"] in eligible_positions(row), \
            f"{row['Name']} is not eligible at {row['Roster']}"

    assert frame["Name"].nunique() == ROSTER_SIZE, "a player appears twice"
    assert pd.to_numeric(frame["Salary"]).sum() <= DK_SALARY_CAP

    hitters = frame[frame["Type"] == "H"]
    assert hitters["Team"].value_counts().max() <= MAX_HITTERS_PER_TEAM
    assert frame["Game"].nunique() >= 2, "DK requires two games"


class TestRosterValidity:
    def test_single_lineup_is_legal(self, slate):
        lineups, _, _ = optimize(slate, n_lineups=1, randomness=0.0, seed=1)
        assert len(lineups) == 1
        assert_legal(lineups[0])

    def test_every_lineup_in_a_set_is_legal(self, slate):
        lineups, _, _ = optimize(slate, n_lineups=10, seed=3)
        assert len(lineups) == 10
        for lineup in lineups:
            assert_legal(lineup)

    @pytest.mark.parametrize("objective", ["ceiling", "proj", "floor", "leverage"])
    def test_all_objectives_produce_legal_lineups(self, slate, objective):
        lineups, _, _ = optimize(slate, n_lineups=2, objective=objective, seed=4)
        assert lineups
        for lineup in lineups:
            assert_legal(lineup)

    def test_unknown_objective_is_rejected(self, slate):
        with pytest.raises(OptimizerError):
            optimize(slate, objective="vibes")

    def test_a_multi_position_player_fills_only_one_slot(self, slate):
        """The reason the model assigns players to slots rather than just selecting them."""
        frame = slate.copy()
        mask = frame["Name"] == "NYY Bat2"
        frame.loc[mask, "DK Pos"] = "2B/3B/SS"
        frame.loc[mask, "Proj"] = 40.0
        frame.loc[mask, "Ceiling"] = 80.0
        frame.loc[mask, "Salary"] = 2000
        lineups, _, _ = optimize(frame, n_lineups=1, randomness=0.0, seed=1)
        assert_legal(lineups[0])
        assert (lineups[0]["players"]["Name"] == "NYY Bat2").sum() == 1


class TestLocksAndExcludes:
    def test_lock_appears_in_every_lineup(self, slate):
        lineups, _, _ = optimize(slate, n_lineups=5, locks=["LAD Bat3"], seed=2)
        assert lineups
        for lineup in lineups:
            assert "LAD Bat3" in set(lineup["players"]["Name"])
            assert_legal(lineup)

    def test_exclude_never_appears(self, slate):
        lineups, _, _ = optimize(slate, n_lineups=8, excludes=["NYY Arm"], seed=2)
        for lineup in lineups:
            assert "NYY Arm" not in set(lineup["players"]["Name"])

    def test_a_lock_survives_a_threshold_that_would_drop_him(self, slate):
        """An explicit lock outranks a filter -- otherwise the filter silently wins."""
        worst = slate.loc[slate["Proj"].idxmin()]
        threshold = float(slate["Proj"].median())
        assert worst["Proj"] < threshold
        lineups, _, _ = optimize(slate, n_lineups=1, locks=[worst["Name"]],
                                 min_proj=threshold, randomness=0.0, seed=1)
        assert worst["Name"] in set(lineups[0]["players"]["Name"])

    def test_missing_names_are_reported_not_raised(self, slate):
        _, _, missing = optimize(slate, n_lineups=1, locks=["Nobody At All"], seed=1)
        assert "Nobody At All" in missing["locks"]

    def test_too_many_locks_is_an_error(self, slate):
        names = slate["Name"].head(11).tolist()
        with pytest.raises(OptimizerError):
            optimize(slate, locks=names)

    def test_slot_group_requires_one_candidate_in_that_exact_slot(self, slate):
        frame = slate.copy()
        candidates = frame[(frame["Type"] == "H") & (frame["DK Pos"] == "SS")]
        names = candidates.nsmallest(2, "Salary")["Name"].tolist()
        lineups, _, _ = optimize(
            frame, objective="proj", n_lineups=1, randomness=0.0,
            stack_bonus=False, slot_groups={"SS": names})
        lineup = lineups[0]["players"]
        chosen = lineup.loc[lineup["Roster"] == "SS", "Name"].item()
        assert chosen in names

    def test_slot_group_rejects_candidates_ineligible_for_the_slot(self, slate):
        pitcher = slate[slate["Type"] == "P"].iloc[0]["Name"]
        with pytest.raises(OptimizerError, match="constrained slot SS"):
            optimize(slate, slot_groups={"SS": [pitcher]})


class TestStackConstraints:
    def test_explicit_stack_is_honoured_exactly(self, slate):
        lineups, _, _ = optimize(slate, n_lineups=4, stacks={"LAD": 4}, seed=6)
        assert lineups
        for lineup in lineups:
            hitters = lineup["players"]
            assert (hitters[hitters["Type"] == "H"]["Team"] == "LAD").sum() >= 4
            assert_legal(lineup)

    def test_two_team_stack(self, slate):
        lineups, _, _ = optimize(slate, n_lineups=3, stacks={"LAD": 4, "NYY": 3}, seed=6)
        for lineup in lineups:
            hitters = lineup["players"][lineup["players"]["Type"] == "H"]
            assert (hitters["Team"] == "LAD").sum() >= 4
            assert (hitters["Team"] == "NYY").sum() >= 3

    def test_a_stack_bigger_than_the_pool_says_how_many_there_are(self, slate):
        thin = slate[~((slate["Team"] == "LAD") & (slate["Slot"] > 3))]
        with pytest.raises(OptimizerError, match="only 3 LAD hitters"):
            optimize(thin, stacks={"LAD": 5})

    def test_a_stack_bigger_than_the_roster_is_infeasible(self, slate):
        """Nine hitters from one team cannot fit eight hitter slots."""
        with pytest.raises(OptimizerError):
            optimize(slate, stacks={"LAD": 9})

    def test_team_aliases_fold(self, slate):
        """A typed CHW must find the CWS hitters, not silently match nothing."""
        frame = slate.copy()
        frame.loc[frame["Team"] == "CHC", "Team"] = "CWS"
        frame.loc[frame["Opp"] == "CHC", "Opp"] = "CWS"
        lineups, _, _ = optimize(frame, n_lineups=1, stacks={"CHW": 4}, randomness=0.0, seed=1)
        hitters = lineups[0]["players"]
        assert (hitters[hitters["Type"] == "H"]["Team"] == "CWS").sum() >= 4

    def test_max_hitters_per_team_is_respected(self, slate):
        lineups, _, _ = optimize(slate, n_lineups=5, max_hitters_per_team=3, seed=8)
        for lineup in lineups:
            hitters = lineup["players"]
            assert hitters[hitters["Type"] == "H"]["Team"].value_counts().max() <= 3

    def test_stack_shape_expands_to_team_pairs(self, slate):
        combos = stack_shapes(slate, "4-3", top_teams=4)
        assert combos
        for combo in combos:
            assert sorted(combo.values()) == [3, 4]
            assert len(combo) == 2

    def test_a_single_stack_shape_leaves_the_team_to_the_solver(self, slate):
        """'5' is one stack with the team unnamed -- the least restrictive way to require one.

        Three of the eight hitter slots and both pitchers stay unclaimed, which is the whole
        point: docs/benchmarks.md §13 found the ceiling comes from stacking *someone*, not
        from stacking a particular someone.
        """
        combos = stack_shapes(slate, "5", top_teams=None)
        teams = {team for combo in combos for team in combo}
        assert len(combos) == len(teams) > 4
        assert all(list(combo.values()) == [5] for combo in combos)

    def test_lifting_the_team_cap_widens_the_search(self, slate):
        capped = stack_shapes(slate, "5", top_teams=3)
        uncapped = stack_shapes(slate, "5", top_teams=None)
        assert len(capped) == 3
        assert len(uncapped) > len(capped)
        # The cap takes the best teams, so it must be a subset -- widening adds, never swaps.
        assert {t for c in capped for t in c} <= {t for c in uncapped for t in c}

    def test_several_shapes_are_spread_evenly_not_concatenated(self, slate):
        """A cycling caller takes a prefix, so the mix must be right at every length.

        '5' yields one combination per team while '5-3' yields a permutation per pair, so
        concatenating would bury the one-stack shape entirely.
        """
        combos = stack_shapes(slate, "5-3,5", top_teams=None)

        def shape(combo):
            return "-".join(str(v) for v in sorted(combo.values(), reverse=True))

        for prefix in (10, 30, 100):
            mix = collections.Counter(shape(c) for c in combos[:prefix])
            assert mix["5-3"] == mix["5"] == prefix // 2

    def test_a_comma_list_is_not_silently_reread_as_one_shape(self, slate):
        """The old parser split on '-', dropped the non-numeric part, and returned 5-2."""
        combos = stack_shapes(slate, "5-3,5-2", top_teams=None)
        assert {tuple(sorted(c.values(), reverse=True)) for c in combos} == {(5, 3), (5, 2)}

    def test_a_malformed_shape_raises_rather_than_dropping_the_bad_part(self, slate):
        for bad in ("5-x", "5--3", "", "  "):
            with pytest.raises(OptimizerError):
                stack_shapes(slate, bad)

    def test_a_single_shape_is_unchanged(self, slate):
        """Backward compatibility: the comma path must not perturb the common case."""
        combos = stack_shapes(slate, "4-3", top_teams=4)
        assert len(combos) == 12          # permutations of 4 teams taken 2 at a time
        assert all(sorted(c.values()) == [3, 4] for c in combos)

    def test_shape_bigger_than_the_roster_is_rejected(self, slate):
        with pytest.raises(OptimizerError):
            stack_shapes(slate, "5-5")

    def test_shape_over_the_team_cap_is_rejected(self, slate):
        with pytest.raises(OptimizerError):
            stack_shapes(slate, "6-2")


class TestOverlapConstraints:
    def test_overlap_cap_is_respected_pairwise(self, slate):
        cap = 4
        lineups, _, _ = optimize(slate, n_lineups=8, max_overlap=cap, seed=11)
        sets = [set(l["players"]["Name"]) for l in lineups]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                assert len(sets[i] & sets[j]) <= cap

    def test_dk_minimum_overlap_still_blocks_exact_duplicates(self, slate):
        lineups, _, _ = optimize(slate, n_lineups=5, max_overlap=ROSTER_SIZE - 1,
                                 randomness=0.0, seed=1)
        sets = [frozenset(l["players"]["Name"]) for l in lineups]
        assert len(set(sets)) == len(sets)

    def test_tighter_overlap_gives_more_distinct_players(self, slate):
        # Randomness off, so the overlap cap is the only thing creating diversity. With the
        # default jitter both settings already saturate the pool and the comparison says
        # nothing about overlap at all.
        loose, _, _ = optimize(slate, n_lineups=8, max_overlap=9, randomness=0.0, seed=5)
        tight, _, _ = optimize(slate, n_lineups=8, max_overlap=3, randomness=0.0, seed=5)
        distinct = lambda ls: len({n for l in ls for n in l["players"]["Name"]})
        assert distinct(tight) > distinct(loose)


class TestExposureConstraints:
    def test_maximum_exposure_is_never_exceeded(self, slate):
        n = 10
        exposure = {"LAD Bat1": (0, 3)}
        lineups, _, _ = optimize(slate, n_lineups=n, exposure=exposure, seed=13)
        used = sum("LAD Bat1" in set(l["players"]["Name"]) for l in lineups)
        assert used <= 3

    def test_zero_maximum_bars_a_player_outright(self, slate):
        lineups, _, _ = optimize(slate, n_lineups=6, exposure={"NYY Arm": (0, 0)}, seed=13)
        for lineup in lineups:
            assert "NYY Arm" not in set(lineup["players"]["Name"])

    def test_minimum_exposure_is_met(self, slate):
        n = 10
        lineups, _, _ = optimize(slate, n_lineups=n, exposure={"SF Bat4": (6, n)}, seed=17)
        used = sum("SF Bat4" in set(l["players"]["Name"]) for l in lineups)
        assert used >= 6

    def test_an_oversubscribed_minimum_reports_rather_than_failing(self, slate):
        """148 good lineups must not die for a shortfall of 55. Report it, keep building."""
        exposure = {name: (10, 10) for name in slate["Name"].head(30)}
        lineups, _, missing = optimize(slate, n_lineups=10, exposure=exposure, seed=19)
        assert lineups, "an impossible minimum must not destroy the whole set"
        for lineup in lineups:
            assert_legal(lineup)
        assert "exposure_minimums" in missing


class TestConflictPenalty:
    def test_default_penalty_avoids_pitcher_against_own_stack(self, slate):
        lineups, _, _ = optimize(slate, n_lineups=6, stacks={"LAD": 4}, seed=21)
        for lineup in lineups:
            for _pitcher, team, count in lineup["conflicts"]:
                assert not (team == "LAD" and count >= 3), \
                    "a taxed pairing was chosen despite alternatives"

    def test_zero_penalty_allows_the_pairing(self, slate):
        """A tax, not a ban -- with the tax off the pairing must be reachable."""
        lineups, _, _ = optimize(slate, n_lineups=1, stacks={"SF": 4},
                                 locks=["LAD Arm"], conflict_penalty=0.0,
                                 randomness=0.0, seed=1)
        assert "LAD Arm" in set(lineups[0]["players"]["Name"])

    def test_conflicts_are_reported_off_the_finished_lineup(self, slate):
        lineups, _, _ = optimize(slate, n_lineups=1, stacks={"SF": 4}, locks=["LAD Arm"],
                                 conflict_penalty=0.0, randomness=0.0, seed=1)
        found = lineup_conflicts(lineups[0]["players"], min_hitters=1)
        assert any(team == "SF" for _name, team, _count in found)


class TestOwnershipCap:
    def test_total_ownership_cap_binds(self, slate):
        cap = 60.0
        lineups, _, _ = optimize(slate, n_lineups=3, max_ownership=cap, seed=23)
        assert lineups
        for lineup in lineups:
            total = pd.to_numeric(lineup["players"]["Own%"], errors="coerce").sum()
            assert total <= cap + 1e-6

    def test_capping_ownership_lowers_it(self, slate):
        free, _, _ = optimize(slate, n_lineups=3, seed=23)
        capped, _, _ = optimize(slate, n_lineups=3, max_ownership=60.0, seed=23)
        owned = lambda ls: np.mean([pd.to_numeric(l["players"]["Own%"]).sum() for l in ls])
        assert owned(capped) < owned(free)


class TestDeterminism:
    def test_same_seed_reproduces_the_set(self, slate):
        a, _, _ = optimize(slate, n_lineups=5, seed=99)
        b, _, _ = optimize(slate, n_lineups=5, seed=99)
        assert [sorted(l["players"]["Name"]) for l in a] == \
               [sorted(l["players"]["Name"]) for l in b]

    def test_different_seeds_differ_when_jitter_is_on(self, slate):
        """The seed only has something to act on when randomness is non-zero.

        This used to call `optimize` with no randomness argument and rely on the default
        being 0.20. That default is now 0.0 -- jitter measured as a net loss across four
        snapshot-scored nights -- so the test was pinning a property the shipped settings no
        longer have. The invariant worth keeping is the conditional one.
        """
        a, _, _ = optimize(slate, n_lineups=5, randomness=0.20, seed=99)
        b, _, _ = optimize(slate, n_lineups=5, randomness=0.20, seed=100)
        assert [sorted(l["players"]["Name"]) for l in a] !=                [sorted(l["players"]["Name"]) for l in b]

    def test_the_shipped_defaults_are_seed_independent(self, slate):
        """With jitter off the seed cannot change the set, so a board reproduces without
        anyone having recorded which seed produced it."""
        a, _, _ = optimize(slate, n_lineups=3, seed=99)
        b, _, _ = optimize(slate, n_lineups=3, seed=100)
        assert [sorted(l["players"]["Name"]) for l in a] ==                [sorted(l["players"]["Name"]) for l in b]

    def test_zero_randomness_is_the_single_best_lineup(self, slate):
        a, _, _ = optimize(slate, n_lineups=1, randomness=0.0, seed=1)
        b, _, _ = optimize(slate, n_lineups=1, randomness=0.0, seed=2)
        assert sorted(a[0]["players"]["Name"]) == sorted(b[0]["players"]["Name"])


class TestPins:
    def test_a_pinned_player_holds_his_exact_column(self, slate):
        """Late swap depends on this: DK diffs an edited entry slot by slot."""
        lineups, _, _ = optimize(slate, n_lineups=1, pins={"LAD Bat6": "OF"},
                                 randomness=0.0, seed=1)
        frame = lineups[0]["players"]
        row = frame[frame["Name"] == "LAD Bat6"]
        assert len(row) == 1
        assert row.iloc[0]["Roster"] == "OF"

    def test_pinning_to_an_ineligible_slot_is_an_error(self, slate):
        with pytest.raises(OptimizerError, match="not eligible"):
            optimize(slate, pins={"LAD Bat1": "SS"})       # Bat1 is the catcher
