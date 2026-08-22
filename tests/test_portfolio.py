"""Joint portfolio selection.

The property that matters is *diversification measured on outcomes*. A set of twenty
lineups that all fail in the same simulated worlds is worth two or three bets, not twenty,
and no count of shared players reveals that. `effective_lineups` is the number under test.
"""

import numpy as np
import pytest

from dfs.portfolio import (PRESETS, PortfolioError, PortfolioWeights, baselines, compare,
                           effective_lineups, lineup_scores, pairwise_correlation,
                           portfolio_metrics, select)


@pytest.fixture(scope="module")
def setup():
    from conftest import make_slate
    from dfs.candidates import generate
    from dfs.contest import evaluate, gpp_payout
    from dfs.field import ContestConfig, simulate_field
    from dfs.simulate import simulate_slate

    players = make_slate(n_games=6, seed=1)
    pool = generate(players, n_candidates=120, seed=3)
    sims = simulate_slate(pool.pool, n_sims=1500, seed=3)
    field, _ = simulate_field(
        pool.pool, config=ContestConfig("t", entries=600, max_entries_per_user=20), seed=3)
    payout = gpp_payout(600, 5.0)
    evaluated = evaluate(pool, sims, field, payout)
    scores = lineup_scores(pool, sims)
    return pool, sims, scores, evaluated, payout


class TestCorrelationMetrics:
    def test_identical_lineups_correlate_perfectly(self):
        rng = np.random.default_rng(1)
        column = rng.normal(100, 30, 2000)
        scores = np.column_stack([column, column, column])
        assert pairwise_correlation(scores, [0, 1, 2]) == pytest.approx(1.0)
        assert effective_lineups(scores, [0, 1, 2]) == pytest.approx(1.0)

    def test_independent_lineups_are_worth_their_count(self):
        rng = np.random.default_rng(1)
        scores = rng.normal(100, 30, (20000, 4))
        assert abs(pairwise_correlation(scores, [0, 1, 2, 3])) < 0.05
        assert effective_lineups(scores, [0, 1, 2, 3]) == pytest.approx(4.0, rel=0.15)

    def test_a_single_lineup_is_worth_one(self):
        scores = np.random.default_rng(1).normal(100, 30, (500, 3))
        assert effective_lineups(scores, [0]) == 1.0
        assert pairwise_correlation(scores, [0]) == 0.0

    def test_correlated_lineups_are_worth_less_than_their_count(self):
        rng = np.random.default_rng(2)
        shared = rng.normal(0, 1, (5000, 1))
        scores = shared + rng.normal(0, 0.4, (5000, 5))
        assert effective_lineups(scores, list(range(5))) < 2.0


class TestSelection:
    def test_returns_the_requested_number(self, setup):
        pool, _sims, scores, evaluated, _payout = setup
        chosen, metrics, _trace = select(pool, scores, evaluated, n_lineups=15, seed=1)
        assert len(chosen) == 15
        assert len(set(chosen)) == 15
        assert metrics["n"] == 15

    def test_never_repeats_a_candidate(self, setup):
        pool, _sims, scores, evaluated, _payout = setup
        chosen, _m, _t = select(pool, scores, evaluated, n_lineups=30, seed=1)
        assert len(set(chosen)) == len(chosen)

    def test_reproducible_for_a_seed(self, setup):
        pool, _sims, scores, evaluated, _payout = setup
        a, _m, _t = select(pool, scores, evaluated, n_lineups=12, seed=9)
        b, _m, _t = select(pool, scores, evaluated, n_lineups=12, seed=9)
        assert a == b

    def test_beats_the_ceiling_baseline_on_diversification(self, setup):
        """The whole justification for the module."""
        pool, _sims, scores, evaluated, _payout = setup
        chosen, _m, _t = select(pool, scores, evaluated, n_lineups=20, seed=1)
        by_ceiling = list(evaluated.sort_values("ceiling", ascending=False).index[:20])
        assert effective_lineups(scores, chosen) > effective_lineups(scores, by_ceiling)
        assert pairwise_correlation(scores, chosen) < pairwise_correlation(scores, by_ceiling)

    def test_beats_the_ceiling_baseline_on_best_of_set(self, setup):
        pool, _sims, scores, evaluated, _payout = setup
        chosen, _m, _t = select(pool, scores, evaluated, n_lineups=20, seed=1)
        by_ceiling = list(evaluated.sort_values("ceiling", ascending=False).index[:20])
        mine = scores[:, chosen].max(axis=1).mean()
        theirs = scores[:, by_ceiling].max(axis=1).mean()
        assert mine > theirs

    def test_covers_more_stacks_than_the_naive_pick(self, setup):
        pool, _sims, scores, evaluated, _payout = setup
        chosen, _m, _t = select(pool, scores, evaluated, n_lineups=20, seed=1)
        by_ceiling = list(evaluated.sort_values("ceiling", ascending=False).index[:20])
        assert evaluated.iloc[chosen]["primary_stack"].nunique() > \
            evaluated.iloc[by_ceiling]["primary_stack"].nunique()

    def test_asking_for_more_than_exists_is_capped(self, setup):
        pool, _sims, scores, evaluated, _payout = setup
        chosen, _m, _t = select(pool, scores, evaluated, n_lineups=10_000, seed=1)
        assert len(chosen) == len(evaluated)

    def test_an_empty_candidate_set_is_an_error(self, setup):
        pool, _sims, scores, evaluated, _payout = setup
        with pytest.raises(PortfolioError):
            select(pool, scores[:, :0], evaluated.head(0), n_lineups=5)


class TestUserOverrides:
    def test_locks_are_always_selected(self, setup):
        pool, _sims, scores, evaluated, _payout = setup
        chosen, _m, _t = select(pool, scores, evaluated, n_lineups=10, seed=1, locks=[3, 7])
        assert 3 in chosen and 7 in chosen

    def test_excludes_are_never_selected(self, setup):
        pool, _sims, scores, evaluated, _payout = setup
        banned = list(range(0, 40))
        chosen, _m, _t = select(pool, scores, evaluated, n_lineups=10, seed=1,
                                excludes=banned)
        assert not set(chosen) & set(banned)

    def test_player_exposure_cap_binds(self, setup):
        pool, _sims, scores, evaluated, _payout = setup
        n = 20
        cap = 0.30
        chosen, _m, _t = select(pool, scores, evaluated, n_lineups=n, seed=1,
                                max_player_exposure=cap)
        counts = {}
        for index in chosen:
            for player in evaluated.iloc[index]["players"]:
                counts[player] = counts.get(player, 0) + 1
        # The cap gates the *next* pick, so a player can reach it but not exceed it by more
        # than the one lineup that was in flight when it bound.
        assert max(counts.values()) <= cap * n + 1

    def test_stack_exposure_cap_binds(self, setup):
        pool, _sims, scores, evaluated, _payout = setup
        n = 20
        chosen, _m, _t = select(pool, scores, evaluated, n_lineups=n, seed=1,
                                max_stack_exposure=0.25)
        counts = evaluated.iloc[chosen]["primary_stack"].value_counts()
        assert counts.max() <= 0.25 * n + 1


class TestWeights:
    def test_zero_correlation_penalty_concentrates_the_set(self, setup):
        pool, _sims, scores, evaluated, _payout = setup
        spread = select(pool, scores, evaluated, n_lineups=15, seed=1,
                        weights=PortfolioWeights(correlation=1.5))[0]
        tight = select(pool, scores, evaluated, n_lineups=15, seed=1,
                       weights=PortfolioWeights(correlation=0.0, stack_coverage=0.0,
                                                pitcher_coverage=0.0, game_coverage=0.0))[0]
        assert pairwise_correlation(scores, spread) < pairwise_correlation(scores, tight)

    def test_coverage_weight_increases_distinct_stacks(self, setup):
        pool, _sims, scores, evaluated, _payout = setup
        none = select(pool, scores, evaluated, n_lineups=15, seed=1,
                      weights=PortfolioWeights(stack_coverage=0.0, correlation=0.0,
                                               pitcher_coverage=0.0, game_coverage=0.0))[0]
        lots = select(pool, scores, evaluated, n_lineups=15, seed=1,
                      weights=PortfolioWeights(stack_coverage=1.5, correlation=0.0,
                                               pitcher_coverage=0.0, game_coverage=0.0))[0]
        assert evaluated.iloc[lots]["primary_stack"].nunique() >= \
            evaluated.iloc[none]["primary_stack"].nunique()

    def test_every_preset_produces_a_legal_selection(self, setup):
        pool, _sims, scores, evaluated, _payout = setup
        for name, weights in PRESETS.items():
            chosen, metrics, _t = select(pool, scores, evaluated, n_lineups=10,
                                         weights=weights, seed=1)
            assert len(chosen) == 10, name
            assert metrics["n"] == 10, name


class TestMetricsAndComparison:
    def test_metrics_are_complete(self, setup):
        pool, _sims, scores, evaluated, payout = setup
        chosen, _m, _t = select(pool, scores, evaluated, n_lineups=12, seed=1)
        metrics = portfolio_metrics(scores, chosen, evaluated, payout=payout)
        for key in ("n", "best_of_set_mean", "best_of_set_p99", "effective_lineups",
                    "mean_pairwise_correlation", "distinct_primary_stacks",
                    "distinct_pitcher_pairs", "distinct_players", "portfolio_roi"):
            assert key in metrics, key

    def test_best_of_set_beats_the_mean(self, setup):
        pool, _sims, scores, evaluated, _payout = setup
        chosen, metrics, _t = select(pool, scores, evaluated, n_lineups=12, seed=1)
        assert metrics["best_of_set_mean"] > metrics["mean_score"]

    def test_distinct_players_is_bounded_by_the_roster(self, setup):
        pool, _sims, scores, evaluated, _payout = setup
        chosen, metrics, _t = select(pool, scores, evaluated, n_lineups=12, seed=1)
        assert 10 <= metrics["distinct_players"] <= 120

    def test_baselines_are_all_the_requested_size(self, setup):
        pool, _sims, scores, evaluated, _payout = setup
        for name, selection in baselines(pool, scores, evaluated, n_lineups=15).items():
            assert len(selection) == 15, name

    def test_compare_returns_a_row_per_method(self, setup):
        pool, _sims, scores, evaluated, payout = setup
        chosen, table = compare(pool, scores, evaluated, n_lineups=15, payout=payout)
        assert len(chosen) == 15
        assert "joint portfolio" in set(table["method"])
        assert len(table) >= 5
        assert table["effective_lineups"].notna().all()

    def test_the_joint_portfolio_leads_the_comparison_on_best_of_set(self, setup):
        pool, _sims, scores, evaluated, payout = setup
        _chosen, table = compare(pool, scores, evaluated, n_lineups=20, payout=payout)
        row = table[table["method"] == "joint portfolio"].iloc[0]
        assert row["best_of_set_mean"] >= table["best_of_set_mean"].max() - 1e-6
