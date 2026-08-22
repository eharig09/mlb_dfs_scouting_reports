"""Exposure and ownership caps in the portfolio path.

Both caps were reachable from the command line and neither reached the code that enforces
them: `--max-player-exposure` and `--max-stack-exposure` were parsed and then dropped by
`compare`, and `--max-ownership` was applied only when candidates were generated, never
when they were read back from the optimizer -- which is the documented production path.
A flag that silently does nothing is worse than a missing one, so the tests here assert
that each cap actually binds, on the path a real run takes.

The locked-player case is the awkward one and gets the most attention. A lock puts a player
in 100% of the candidate pool, so any cap below 100% is unsatisfiable for him. Enforced
naively that blocks every candidate and the selector returns a short set; the contract is
that locks are exempt and everything else is still capped.
"""

import numpy as np
import pandas as pd
import pytest

from dfs.portfolio import PortfolioWeights, _universal_players, compare, select


POPULAR = 200


def make_candidates(n_lineups=40, roster=6, forced=None, seed=5, popular_top=None):
    """A candidate pool with known player membership, plus matching sim scores.

    `forced` players are placed in every lineup, which is exactly what an upstream lock
    does to a pool by the time the portfolio sees it.

    `popular_top` puts one extra player into the highest-value lineups only. Without him
    the pool is too diffuse for any realistic cap to bind -- the free players spread over
    enough lineups that nobody approaches 50% -- and a cap test on such a pool passes
    whether or not the cap is implemented at all.
    """
    rng = np.random.default_rng(seed)
    forced = list(forced or [])
    popular_top = n_lineups if popular_top is None else popular_top
    rows = []
    for i in range(n_lineups):
        players = list(forced)
        if i < popular_top:
            players.append(POPULAR)
        free_slots = roster - len(players)
        # A deliberately small free universe, so lineups overlap the way real ones do.
        others = rng.choice(np.arange(100, 118), size=free_slots, replace=False)
        players = players + [int(p) for p in others]
        rows.append({
            "lineup": i,
            "players": players,
            "primary_stack": f"T{i % 4}",
            "pitcher_pair": f"P{i % 5}",
            "stack_shape": f"{4 - i % 2}-2",
            "games": [f"G{i % 3}"],
            "salary": 49000 + (i % 5) * 100,
            "proj": float(120 - i * 0.5),
            "ceiling": float(150 - i),      # a strict, known preference order
            "exp_payout": float(150 - i),
            "exp_dupes": 0.0,
        })
    candidates = pd.DataFrame(rows)
    scores = rng.normal(120, 25, (400, n_lineups)).astype(np.float32)
    # Make the objective agree with `ceiling` so the greedy order is predictable.
    scores += np.linspace(20, 0, n_lineups, dtype=np.float32)
    return candidates, scores


def exposure_counts(candidates, chosen):
    counts = {}
    for index in chosen:
        for player in candidates.iloc[index]["players"]:
            counts[player] = counts.get(player, 0) + 1
    return counts


class TestUniversalPlayers:
    def test_finds_a_player_in_every_lineup(self):
        candidates, _ = make_candidates(forced=[7, 8], popular_top=0)
        assert _universal_players(candidates) == {7, 8}

    def test_finds_nothing_when_nobody_is_forced(self):
        candidates, _ = make_candidates(forced=[], popular_top=0)
        assert _universal_players(candidates) == set()

    def test_a_merely_popular_player_is_not_treated_as_locked(self):
        """Exemption is for players in *every* candidate, not just most of them."""
        candidates, _ = make_candidates(n_lineups=60, forced=[], popular_top=59)
        assert _universal_players(candidates) == set()

    def test_is_empty_on_an_empty_pool(self):
        assert _universal_players(pd.DataFrame({"players": []})) == set()


class TestPlayerExposureCap:
    def test_the_cap_binds(self):
        candidates, scores = make_candidates(n_lineups=60, forced=[], popular_top=40)
        chosen, _, _ = select(None, scores, candidates, n_lineups=20,
                              weights=PortfolioWeights(), seed=1,
                              max_player_exposure=0.5, swap_passes=1)
        counts = exposure_counts(candidates, chosen)
        assert max(counts.values()) <= 0.5 * 20 + 1e-9

    def test_without_the_cap_someone_exceeds_it(self):
        """Guards against a cap that 'passes' only because nothing was near it anyway."""
        candidates, scores = make_candidates(n_lineups=60, forced=[], popular_top=40)
        chosen, _, _ = select(None, scores, candidates, n_lineups=20,
                              weights=PortfolioWeights(), seed=1, swap_passes=1)
        counts = exposure_counts(candidates, chosen)
        assert max(counts.values()) > 0.5 * 20

    def test_a_locked_player_does_not_truncate_the_set(self):
        """The reported failure: two locked players and a 50% cap must still fill 20."""
        candidates, scores = make_candidates(n_lineups=60, forced=[7, 8], popular_top=40)
        chosen, _, _ = select(None, scores, candidates, n_lineups=20,
                              weights=PortfolioWeights(), seed=1,
                              max_player_exposure=0.5, swap_passes=1)
        assert len(chosen) == 20
        counts = exposure_counts(candidates, chosen)
        assert counts[7] == 20 and counts[8] == 20      # locks stay at 100%

    def test_everyone_else_is_still_capped_around_a_lock(self):
        candidates, scores = make_candidates(n_lineups=60, forced=[7, 8], popular_top=40)
        chosen, _, _ = select(None, scores, candidates, n_lineups=20,
                              weights=PortfolioWeights(), seed=1,
                              max_player_exposure=0.5, swap_passes=1)
        counts = exposure_counts(candidates, chosen)
        free = {p: c for p, c in counts.items() if p not in (7, 8)}
        assert max(free.values()) <= 0.5 * 20 + 1e-9

    def test_an_explicit_exemption_is_honoured(self):
        candidates, scores = make_candidates(n_lineups=60, forced=[], popular_top=40)
        chosen, _, _ = select(None, scores, candidates, n_lineups=20,
                              weights=PortfolioWeights(), seed=1,
                              max_player_exposure=0.25, swap_passes=1,
                              exempt_players={POPULAR})
        counts = exposure_counts(candidates, chosen)
        assert counts[POPULAR] > 0.25 * 20          # the exemption actually mattered
        capped = {p: c for p, c in counts.items() if p != POPULAR}
        assert not capped or max(capped.values()) <= 0.25 * 20 + 1e-9

    def test_the_swap_pass_cannot_undo_the_cap(self):
        """Greedy honoured the caps and the improvement pass used to ignore them."""
        candidates, scores = make_candidates(n_lineups=80, forced=[], popular_top=55)
        chosen, _, _ = select(None, scores, candidates, n_lineups=24,
                              weights=PortfolioWeights(), seed=4,
                              max_player_exposure=0.4, swap_passes=3)
        counts = exposure_counts(candidates, chosen)
        assert max(counts.values()) <= 0.4 * 24 + 1e-9


class TestStackExposureCap:
    def test_the_cap_binds(self):
        candidates, scores = make_candidates(n_lineups=60, forced=[], popular_top=40)
        chosen, _, _ = select(None, scores, candidates, n_lineups=20,
                              weights=PortfolioWeights(), seed=1,
                              max_stack_exposure=0.3, swap_passes=2)
        stacks = candidates.iloc[chosen]["primary_stack"].value_counts()
        assert stacks.max() <= 0.3 * 20 + 1e-9


class TestComparePlumbing:
    """`compare` is the only path `main` takes; a cap that stops there never happens."""

    def test_compare_forwards_the_player_cap(self):
        candidates, scores = make_candidates(n_lineups=60, forced=[], popular_top=40)
        chosen, _ = compare(None, scores, candidates, n_lineups=20,
                            weights=PortfolioWeights(), seed=1, swap_passes=1,
                            max_player_exposure=0.5)
        counts = exposure_counts(candidates, chosen)
        assert max(counts.values()) <= 0.5 * 20 + 1e-9

    def test_compare_forwards_the_stack_cap(self):
        candidates, scores = make_candidates(n_lineups=60, forced=[], popular_top=40)
        chosen, _ = compare(None, scores, candidates, n_lineups=20,
                            weights=PortfolioWeights(), seed=1, swap_passes=1,
                            max_stack_exposure=0.3)
        stacks = candidates.iloc[chosen]["primary_stack"].value_counts()
        assert stacks.max() <= 0.3 * 20 + 1e-9


class TestOwnershipCapOnOptimizerLineups:
    def make_pool(self, own_values):
        from dfs.candidates import CandidatePool
        lineups = pd.DataFrame({
            "lineup": range(len(own_values)),
            "players": [[1, 2, 3] for _ in own_values],
            "own_sum": own_values,
            "proj": [100.0] * len(own_values),
            "ceiling": [150.0] * len(own_values),
            "stack_shape": ["4-2"] * len(own_values),
            "primary_stack": ["T1"] * len(own_values),
            "pitcher_pair": ["P1"] * len(own_values),
        })
        pool = pd.DataFrame({"Name": ["A", "B", "C"], "Own%": [10.0, 20.0, 30.0]})
        return CandidatePool(lineups, pool, {"source_path": "x.csv"})

    def test_drops_lineups_over_the_cap(self):
        from dfs.contest import _cap_pool_ownership
        capped = _cap_pool_ownership(self.make_pool([80.0, 120.0, 150.0, 90.0]), 100.0,
                                     quiet=True)
        assert sorted(capped.lineups["own_sum"]) == [80.0, 90.0]

    def test_keeps_everything_when_the_cap_would_empty_the_pool(self):
        """Selecting from nothing fails several stages later, pointing somewhere else."""
        from dfs.contest import _cap_pool_ownership
        original = self.make_pool([180.0, 200.0])
        capped = _cap_pool_ownership(original, 50.0, quiet=True)
        assert len(capped.lineups) == 2

    def test_tolerates_lineups_with_no_ownership_column(self):
        from dfs.candidates import CandidatePool
        from dfs.contest import _cap_pool_ownership
        pool = self.make_pool([80.0, 120.0])
        stripped = CandidatePool(pool.lineups.drop(columns=["own_sum"]), pool.pool, {})
        assert len(_cap_pool_ownership(stripped, 100.0, quiet=True).lineups) == 2
