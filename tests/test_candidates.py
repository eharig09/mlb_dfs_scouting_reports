"""Candidate lineup generation.

Three properties, in order of how badly they fail:

1. Every candidate is a legal DK lineup. A pool with illegal entries poisons everything
   downstream and only surfaces at upload.
2. The pool is *diverse*. Portfolio selection can decline a stack it was offered; it cannot
   select one that was never generated.
3. Pruning does not delete reachable good lineups. This is the quiet one -- an
   over-pruned pool still produces legal, diverse, high-projection candidates while having
   removed the lineup that would have won.
"""

import numpy as np
import pandas as pd
import pytest

from dfs.candidates import (CandidateError, CandidatePool, LineupModel, dominated_players,
                            generate, stack_cores)
from dfs.optimizer import MAX_HITTERS_PER_TEAM, ROSTER, ROSTER_SIZE, eligible_positions
from dfs.salaries import canon_team
from dfs.scoring import DK_SALARY_CAP


def assert_pool_legal(pool):
    for _, row in pool.lineups.iterrows():
        indices = list(row["players"])
        assert len(indices) == ROSTER_SIZE
        assert len(set(indices)) == ROSTER_SIZE, "a player appears twice"

        frame = pool.pool.loc[indices]
        counts = pd.Series(list(row["roster"])).value_counts().to_dict()
        for slot, required in ROSTER.items():
            assert counts.get(slot, 0) == required

        for i, slot in zip(indices, row["roster"]):
            assert slot in eligible_positions(pool.pool.loc[i])

        assert pd.to_numeric(frame["Salary"]).sum() <= DK_SALARY_CAP
        hitters = frame[frame["Type"] == "H"]
        assert hitters["Team"].map(canon_team).value_counts().max() <= MAX_HITTERS_PER_TEAM
        assert frame["Game"].nunique() >= 2


class TestGeneration:
    def test_every_candidate_is_legal(self, slate):
        pool = generate(slate, n_candidates=60, seed=1)
        assert len(pool) == 60
        assert_pool_legal(pool)

    def test_candidates_are_distinct(self, slate):
        pool = generate(slate, n_candidates=80, seed=1)
        keys = {frozenset(p) for p in pool.lineups["players"]}
        assert len(keys) == len(pool)

    def test_reproducible_for_a_seed(self, slate):
        a = generate(slate, n_candidates=40, seed=5)
        b = generate(slate, n_candidates=40, seed=5)
        assert [sorted(p) for p in a.lineups["players"]] == \
               [sorted(p) for p in b.lineups["players"]]

    def test_different_seeds_differ(self, slate):
        a = generate(slate, n_candidates=40, seed=5)
        b = generate(slate, n_candidates=40, seed=6)
        assert [sorted(p) for p in a.lineups["players"]] != \
               [sorted(p) for p in b.lineups["players"]]

    def test_every_candidate_carries_a_stack(self, slate):
        """Generation is core-driven, so a candidate with no stack is a bug."""
        pool = generate(slate, n_candidates=50, seed=2)
        assert (pool.lineups["primary_size"] >= 3).all()

    def test_metrics_are_attached(self, slate):
        pool = generate(slate, n_candidates=20, seed=2)
        for column in ("salary", "proj", "ceiling", "floor", "own_sum", "stack_shape",
                       "primary_stack", "pitcher_pair", "n_games"):
            assert column in pool.lineups.columns
        assert pool.lineups["proj"].notna().all()

    def test_recorded_metrics_match_the_players(self, slate):
        pool = generate(slate, n_candidates=15, seed=3)
        for _, row in pool.lineups.iterrows():
            frame = pool.pool.loc[list(row["players"])]
            assert row["salary"] == pytest.approx(pd.to_numeric(frame["Salary"]).sum())
            assert row["proj"] == pytest.approx(
                pd.to_numeric(frame["Proj"]).sum(), abs=0.01)


class TestDiversity:
    def test_many_primary_teams_are_covered(self, slate):
        """Score-ordered cores would spend the whole budget on the best offense."""
        pool = generate(slate, n_candidates=120, seed=1)
        teams = pool.lineups["primary_stack"].nunique()
        available = slate[slate["Type"] == "H"]["Team"].nunique()
        assert teams >= min(available, 8), f"only {teams} primary teams of {available}"

    def test_several_pitcher_pairs(self, slate):
        pool = generate(slate, n_candidates=120, seed=1)
        assert pool.lineups["pitcher_pair"].nunique() >= 5

    def test_interleaving_beats_pure_score_order(self, slate):
        """Directly tests _interleave_by_team's reason for existing."""
        from dfs.optimizer import build_pool
        base, _, _ = build_pool(slate)
        interleaved = stack_cores(base)
        by_score = sorted(interleaved, key=lambda c: -c["score"])
        head = 20
        assert len({c["team"] for c in interleaved[:head]}) > \
               len({c["team"] for c in by_score[:head]})


class TestStackCores:
    def test_cores_respect_requested_sizes(self, slate):
        from dfs.optimizer import build_pool
        base, _, _ = build_pool(slate)
        cores = stack_cores(base, sizes=(4,))
        assert cores
        assert {c["size"] for c in cores} == {4}

    def test_contiguous_runs_include_a_wraparound(self, slate):
        from dfs.optimizer import build_pool
        base, _, _ = build_pool(slate)
        cores = stack_cores(base, sizes=(4,), best_of=False)
        # 8-9-1-2 is a real stack; the order turns over.
        assert any(set(c["slots"]) == {8, 9, 1, 2} for c in cores)

    def test_focus_teams_restrict_the_cores(self, slate):
        from dfs.optimizer import build_pool
        base, _, _ = build_pool(slate)
        cores = stack_cores(base, focus_teams=["LAD", "NYY"])
        assert {c["team"] for c in cores} == {"LAD", "NYY"}

    def test_unknown_focus_team_is_an_error(self, slate):
        from dfs.optimizer import build_pool
        base, _, _ = build_pool(slate)
        with pytest.raises(CandidateError, match="no hitters"):
            stack_cores(base, focus_teams=["ZZZ"])


class TestPruning:
    def test_pruning_never_strands_a_team_below_stack_size(self, slate):
        """The measured failure: unguarded pruning left 7 of 20 teams unstackable."""
        from dfs.optimizer import build_pool
        base, _, _ = build_pool(slate)
        before = base[base["Type"] == "H"]["Team"].map(canon_team).value_counts()
        dropped = dominated_players(base, keep_per_position=8)
        after = (base.drop(index=dropped).pipe(lambda f: f[f["Type"] == "H"])
                 ["Team"].map(canon_team).value_counts())
        assert set(before.index) == set(after.index), "a team was pruned out of existence"
        assert (after >= min(MAX_HITTERS_PER_TEAM, before.min())).all()

    def test_pruning_keeps_every_position_fillable(self, slate):
        from dfs.optimizer import build_pool
        base, _, _ = build_pool(slate)
        kept = base.drop(index=dominated_players(base, keep_per_position=8))
        for slot, required in ROSTER.items():
            n = sum(slot in eligible_positions(kept.iloc[i]) for i in range(len(kept)))
            assert n >= required

    def test_a_cheap_contrarian_is_never_dominated_by_chalk(self, slate):
        """Low ownership is a payoff term, not a defect."""
        frame = slate.copy()
        chalk = frame.index[(frame["Type"] == "H") & (frame["Pos"] == "SS")][0]
        contrarian = frame.index[(frame["Type"] == "H") & (frame["Pos"] == "SS")][1]
        frame.loc[chalk, ["Proj", "Ceiling", "Salary", "Own%"]] = [12.0, 26.0, 4000, 45.0]
        frame.loc[contrarian, ["Proj", "Ceiling", "Salary", "Own%"]] = [9.0, 22.0, 4000, 1.5]
        from dfs.optimizer import build_pool
        base, _, _ = build_pool(frame)
        target = base.index[base["Name"] == frame.loc[contrarian, "Name"]][0]
        assert target not in set(dominated_players(base, keep_per_position=0,
                                                   keep_per_team=0))

    def test_pruning_is_off_by_default(self, slate):
        pool = generate(slate, n_candidates=10, seed=1)
        assert pool.meta["pruned_players"] == 0

    def test_pruned_pool_is_still_legal_and_diverse(self, slate):
        pool = generate(slate, n_candidates=60, seed=1, prune=True)
        assert_pool_legal(pool)
        assert pool.lineups["primary_stack"].nunique() >= 6


class TestConstraints:
    def test_locks_appear_everywhere(self, slate):
        pool = generate(slate, n_candidates=25, seed=4, locks=["LAD Bat3"])
        for names in pool.lineups["names"]:
            assert "LAD Bat3" in list(names)

    def test_excludes_appear_nowhere(self, slate):
        pool = generate(slate, n_candidates=25, seed=4, excludes=["NYY Arm"])
        for names in pool.lineups["names"]:
            assert "NYY Arm" not in list(names)

    def test_focus_teams_confine_the_primary_stack(self, slate):
        pool = generate(slate, n_candidates=30, seed=4, focus_teams=["LAD", "SF"])
        assert set(pool.lineups["primary_stack"]) <= {"LAD", "SF"}

    def test_ownership_cap_binds(self, slate):
        pool = generate(slate, n_candidates=25, seed=4, max_ownership=70.0)
        assert (pool.lineups["own_sum"] <= 70.0 + 1e-6).all()

    def test_minimum_salary_binds(self, slate):
        pool = generate(slate, n_candidates=25, seed=4, min_salary=48000)
        assert (pool.lineups["salary"] >= 48000).all()

    def test_max_hitters_per_team_is_respected(self, slate):
        pool = generate(slate, n_candidates=30, seed=4, max_hitters_per_team=3,
                        sizes=(3,))
        assert (pool.lineups["primary_size"] <= 3).all()


class TestMatrixAndScoring:
    def test_matrix_shape_and_column_sums(self, slate):
        pool = generate(slate, n_candidates=30, seed=1)
        matrix = pool.matrix()
        assert matrix.shape == (len(pool.pool), len(pool))
        assert np.allclose(np.asarray(matrix.sum(axis=0)).ravel(), ROSTER_SIZE)

    def test_scoring_a_simulation_matches_a_manual_sum(self, slate):
        from dfs.simulate import simulate_slate
        pool = generate(slate, n_candidates=20, seed=1)
        sims = simulate_slate(pool.pool, n_sims=100, seed=3)
        scores = pool.score(sims)
        assert scores.shape == (100, len(pool))
        first = list(pool.lineups["players"].iloc[0])
        assert np.allclose(scores[:, 0], sims[:, first].sum(axis=1), atol=1e-3)

    def test_simulation_reranks_candidates(self, slate):
        """The whole thesis: a correlated simulation does not agree with summed ceiling."""
        from dfs.simulate import simulate_slate
        pool = generate(slate, n_candidates=120, seed=1)
        sims = simulate_slate(pool.pool, n_sims=3000, seed=3)
        scores = pool.score(sims)
        p99 = np.percentile(scores, 99, axis=0)
        ceiling = pool.lineups["ceiling"].to_numpy()
        disagreement = np.abs(pd.Series(-ceiling).rank() - pd.Series(-p99).rank()).mean()
        assert disagreement > 3, "simulation and summed ceiling rank identically"


class TestPersistence:
    def test_save_and_load_round_trip(self, slate, tmp_path):
        pool = generate(slate, n_candidates=25, seed=1)
        directory = str(tmp_path / "candidates")
        pool.save(directory)
        loaded = CandidatePool.load(directory)
        assert len(loaded) == len(pool)
        assert [list(p) for p in loaded.lineups["players"]] == \
               [list(p) for p in pool.lineups["players"]]
        assert loaded.meta["seed"] == 1
        pd.testing.assert_frame_equal(loaded.pool.reset_index(drop=True),
                                      pool.pool.reset_index(drop=True))

    def test_a_loaded_pool_can_still_be_scored(self, slate, tmp_path):
        from dfs.simulate import simulate_slate
        pool = generate(slate, n_candidates=15, seed=1)
        directory = str(tmp_path / "candidates")
        pool.save(directory)
        loaded = CandidatePool.load(directory)
        sims = simulate_slate(loaded.pool, n_sims=50, seed=1)
        assert loaded.score(sims).shape == (50, len(loaded))


class TestModel:
    def test_sparse_encoding_is_smaller_than_dense(self, slate):
        from dfs.optimizer import build_pool
        base, _, _ = build_pool(slate)
        model = LineupModel(base)
        assert model.width < len(base) * len(ROSTER)

    def test_an_empty_pool_is_refused(self, slate):
        with pytest.raises(Exception):
            generate(slate.head(3), n_candidates=5, seed=1)
