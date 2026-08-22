"""The incremental swap-pass objective.

Recomputing the portfolio objective in full costs 93 ms at n=150 with 20,000 simulations,
and the swap pass wants ~75,000 evaluations -- 1.9 hours. `_SwapState` maintains the
correlation as a running column sum of standardised scores, E[max] via the top two values
per simulation, and coverage as counter deltas.

`_objective` stays as the reference implementation and these pin the fast path against it.
The one that matters most is `test_the_fast_pass_picks_what_the_slow_one_would`: agreeing
on a number is necessary, agreeing on the chosen portfolio is the point.
"""

import numpy as np
import pytest

from dfs.portfolio import (PortfolioWeights, _objective, _SwapState, _standardise,
                           pairwise_correlation, select)


@pytest.fixture(scope="module")
def setup():
    from conftest import make_slate
    from dfs.candidates import generate
    from dfs.contest import evaluate, gpp_payout
    from dfs.field import ContestConfig, simulate_field
    from dfs.portfolio import lineup_scores
    from dfs.simulate import simulate_slate

    players = make_slate(n_games=6, seed=1)
    pool = generate(players, n_candidates=100, seed=3)
    sims = simulate_slate(pool.pool, n_sims=1200, seed=3)
    field, _ = simulate_field(
        pool.pool, config=ContestConfig("t", entries=600, max_entries_per_user=20), seed=3)
    evaluated = evaluate(pool, sims, field, gpp_payout(600, 5.0))
    return pool, lineup_scores(pool, sims), evaluated


def _parts(evaluated):
    weights = PortfolioWeights()
    payoffs = evaluated["exp_payout"].to_numpy(float)
    dupes = evaluated["exp_dupes"].to_numpy(float)
    return weights, payoffs, dupes, float(np.nanmax(np.abs(payoffs)))


class TestStandardise:
    def test_a_correlation_becomes_a_dot_product(self):
        """The identity the whole optimisation rests on."""
        rng = np.random.default_rng(0)
        scores = rng.normal(100, 25, (4000, 6))
        standard = _standardise(scores)
        for i in range(6):
            for j in range(6):
                assert (standard[:, i] @ standard[:, j]) / 4000 == pytest.approx(
                    np.corrcoef(scores[:, i], scores[:, j])[0, 1], abs=1e-9)

    def test_the_column_sum_identity_gives_mean_pairwise_correlation(self):
        rng = np.random.default_rng(1)
        scores = rng.normal(100, 25, (4000, 8))
        selection = [0, 2, 3, 6]
        standard = _standardise(scores)
        column_sum = standard[:, selection].sum(axis=1)
        k = len(selection)
        implied = ((column_sum @ column_sum) / 4000 - k) / (k * (k - 1))
        assert implied == pytest.approx(pairwise_correlation(scores, selection), abs=1e-9)

    def test_a_constant_column_is_uncorrelated_rather_than_nan(self):
        scores = np.column_stack([np.random.default_rng(2).normal(0, 1, 500),
                                  np.full(500, 7.0)])
        standard = _standardise(scores)
        assert np.all(standard[:, 1] == 0.0)
        assert np.isfinite(standard).all()


class TestAgainstTheReference:
    def test_base_selection(self, setup):
        _pool, scores, evaluated = setup
        weights, payoffs, dupes, unit = _parts(evaluated)
        selection = list(range(20))
        state = _SwapState(scores, evaluated, selection, weights, payoffs, dupes, unit)
        assert state.value() == pytest.approx(
            _objective(scores, evaluated, selection, weights, payoffs, dupes, unit),
            abs=1e-2)

    def test_trial_swaps(self, setup):
        _pool, scores, evaluated = setup
        weights, payoffs, dupes, unit = _parts(evaluated)
        rng = np.random.default_rng(4)
        selection = [int(i) for i in rng.choice(len(evaluated), 20, replace=False)]
        state = _SwapState(scores, evaluated, selection, weights, payoffs, dupes, unit)
        outside = [i for i in range(len(evaluated)) if i not in selection]
        for _ in range(40):
            position = int(rng.integers(0, len(selection)))
            replacement = int(rng.choice(outside))
            trial = list(selection)
            trial[position] = replacement
            assert state.value(position, replacement) == pytest.approx(
                _objective(scores, evaluated, trial, weights, payoffs, dupes, unit),
                abs=1e-2)

    def test_stays_correct_after_applying(self, setup):
        """`apply` updates every running quantity; drift would compound silently."""
        _pool, scores, evaluated = setup
        weights, payoffs, dupes, unit = _parts(evaluated)
        rng = np.random.default_rng(5)
        selection = [int(i) for i in rng.choice(len(evaluated), 20, replace=False)]
        state = _SwapState(scores, evaluated, selection, weights, payoffs, dupes, unit)
        for _ in range(15):
            outside = [i for i in range(len(evaluated)) if i not in state.selection]
            state.apply(int(rng.integers(0, len(state.selection))), int(rng.choice(outside)))
            assert state.value() == pytest.approx(
                _objective(scores, evaluated, state.selection, weights, payoffs, dupes,
                           unit), abs=1e-2)

    def test_coverage_delta_matches_a_recount(self, setup):
        """The counter arithmetic replaced three dict copies per trial swap."""
        _pool, scores, evaluated = setup
        weights, payoffs, dupes, unit = _parts(evaluated)
        rng = np.random.default_rng(6)
        selection = [int(i) for i in rng.choice(len(evaluated), 18, replace=False)]
        state = _SwapState(scores, evaluated, selection, weights, payoffs, dupes, unit)
        outside = [i for i in range(len(evaluated)) if i not in selection]
        for _ in range(30):
            position = int(rng.integers(0, len(selection)))
            replacement = int(rng.choice(outside))
            trial = list(selection)
            trial[position] = replacement
            rows = evaluated.iloc[trial]
            expected = (weights.stack_coverage * rows["primary_stack"].nunique()
                        + weights.pitcher_coverage * rows["pitcher_pair"].nunique()
                        + weights.game_coverage
                        * len({g for games in rows["games"] for g in games}))
            got = (state.coverage_base
                   + weights.stack_coverage * state._distinct_delta(
                       state.stack_count, (state.stack_code[selection[position]],),
                       (state.stack_code[replacement],))
                   + weights.pitcher_coverage * state._distinct_delta(
                       state.pair_count, (state.pair_code[selection[position]],),
                       (state.pair_code[replacement],))
                   + weights.game_coverage * state._distinct_delta(
                       state.game_count, state.game_lists[selection[position]],
                       state.game_lists[replacement]))
            assert got == pytest.approx(expected, abs=1e-9)


class TestSelectionEquivalence:
    def test_the_fast_pass_picks_what_the_slow_one_would(self, setup):
        """Agreeing on a number is necessary; agreeing on the portfolio is the point."""
        pool, scores, evaluated = setup
        weights, payoffs, dupes, unit = _parts(evaluated)

        greedy, _m, _t = select(pool, scores, evaluated, n_lineups=15, weights=weights,
                                seed=1, swap_passes=0)
        fast, _m, _t = select(pool, scores, evaluated, n_lineups=15, weights=weights,
                              seed=1, swap_passes=2)

        selected = list(greedy)
        allowed = np.ones(len(evaluated), dtype=bool)
        allowed[selected] = False
        for _ in range(2):
            improved = False
            baseline = _objective(scores, evaluated, selected, weights, payoffs, dupes, unit)
            for position in range(len(selected)):
                current = selected[position]
                best_value, best_index = baseline, current
                for index in np.flatnonzero(allowed):
                    trial = list(selected)
                    trial[position] = int(index)
                    value = _objective(scores, evaluated, trial, weights, payoffs, dupes,
                                       unit)
                    if value > best_value + 1e-9:
                        best_value, best_index = value, int(index)
                if best_index != current:
                    allowed[current] = True
                    allowed[best_index] = False
                    selected[position] = best_index
                    baseline = best_value
                    improved = True
            if not improved:
                break
        assert sorted(fast) == sorted(selected)

    def test_the_swap_pass_does_not_worsen_the_objective(self, setup):
        pool, scores, evaluated = setup
        weights, payoffs, dupes, unit = _parts(evaluated)
        greedy, _m, _t = select(pool, scores, evaluated, n_lineups=15, weights=weights,
                                seed=1, swap_passes=0)
        swapped, _m, _t = select(pool, scores, evaluated, n_lineups=15, weights=weights,
                                 seed=1, swap_passes=2)
        before = _objective(scores, evaluated, greedy, weights, payoffs, dupes, unit)
        after = _objective(scores, evaluated, swapped, weights, payoffs, dupes, unit)
        assert after >= before - 1e-6

    def test_swap_passes_zero_is_greedy_only(self, setup):
        pool, scores, evaluated = setup
        a, _m, _t = select(pool, scores, evaluated, n_lineups=15, seed=1, swap_passes=0)
        b, _m, _t = select(pool, scores, evaluated, n_lineups=15, seed=1, swap_passes=0)
        assert a == b

    def test_locks_survive_the_fast_pass(self, setup):
        pool, scores, evaluated = setup
        chosen, _m, _t = select(pool, scores, evaluated, n_lineups=12, seed=1,
                                locks=[2, 9], swap_passes=2)
        assert 2 in chosen and 9 in chosen

    def test_exposure_caps_survive_the_fast_pass(self, setup):
        pool, scores, evaluated = setup
        n = 15
        chosen, _m, _t = select(pool, scores, evaluated, n_lineups=n, seed=1,
                                max_player_exposure=0.4, swap_passes=2)
        counts = {}
        for index in chosen:
            for player in evaluated.iloc[index]["players"]:
                counts[player] = counts.get(player, 0) + 1
        assert max(counts.values()) <= 0.4 * n + 1
