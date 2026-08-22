"""Correlated slate simulation.

Two things have to hold and they pull against each other. The **marginals** must match what
the projection claims, or every downstream EV number is biased. The **correlations** must
match baseball, or the marginals being right is worthless -- a simulator with perfect
per-player distributions and independent draws will confidently recommend a four-man stack
whose joint outcome cannot happen.

The correlation tests are the ones that matter most, because a marginal error is visible in
a summary table and a correlation error is not.
"""

import numpy as np
import pandas as pd
import pytest

from dfs.simulate import (SimConfig, SimulationError, correlation_report,
                          player_summary, simulate_slate, validate)


@pytest.fixture(scope="module")
def sims(request):
    from conftest import make_slate
    frame = make_slate(n_games=6, seed=1)
    return frame, simulate_slate(frame, n_sims=4000, seed=7)


class TestReproducibility:
    def test_same_seed_is_bit_identical(self, slate):
        a = simulate_slate(slate, n_sims=300, seed=42)
        b = simulate_slate(slate, n_sims=300, seed=42)
        assert np.array_equal(a, b)

    def test_different_seeds_differ(self, slate):
        a = simulate_slate(slate, n_sims=300, seed=42)
        b = simulate_slate(slate, n_sims=300, seed=43)
        assert not np.array_equal(a, b)

    def test_shape_and_dtype(self, slate):
        out = simulate_slate(slate, n_sims=250, seed=1)
        assert out.shape == (250, len(slate))
        assert out.dtype == np.float32

    def test_row_order_matches_the_input_frame(self, slate):
        """Callers index straight into the frame they passed, so alignment is load-bearing."""
        shuffled = slate.sample(frac=1.0, random_state=3).reset_index(drop=True)
        out = simulate_slate(shuffled, n_sims=2000, seed=5)
        means = out.mean(axis=0)
        projected = pd.to_numeric(shuffled["Proj"]).to_numpy()
        # Correlation near 1 only holds if row i of the output is row i of the input.
        assert np.corrcoef(means, projected)[0, 1] > 0.95


class TestMarginals:
    def test_simulated_mean_tracks_the_projection(self, sims):
        frame, out = sims
        summary = player_summary(frame, out)
        for kind, tolerance in (("H", 0.8), ("P", 1.5)):
            group = summary[summary["Type"] == kind]
            gap = (group["Sim Mean"] - group["Proj"]).mean()
            assert abs(gap) < tolerance, f"{kind}: mean gap {gap:+.2f}"

    def test_no_systematic_inflation_per_player(self, sims):
        """Guards the log-normal bias: exp(shock) averages above 1 unless de-biased."""
        frame, out = sims
        summary = player_summary(frame, out)
        hitters = summary[summary["Type"] == "H"]
        ratio = (hitters["Sim Mean"] / hitters["Proj"].clip(lower=0.5)).mean()
        assert 0.9 < ratio < 1.15, f"hitters simulate {ratio:.2f}x their projection"

    def test_hitters_blank_at_a_realistic_rate(self, sims):
        """27% of real hitter-games score zero. A simulator that never blanks is useless."""
        frame, out = sims
        hitters = out[:, (frame["Type"] == "H").to_numpy()]
        p_zero = (hitters <= 0).mean()
        assert 0.08 < p_zero < 0.35, f"P(0) = {p_zero:.3f}"

    def test_pitchers_can_score_negative(self, sims):
        frame, out = sims
        pitchers = out[:, (frame["Type"] == "P").to_numpy()]
        assert (pitchers < 0).any(), "no simulated pitcher ever had a disaster"

    def test_distribution_is_right_skewed(self, sims):
        frame, out = sims
        hitters = out[:, (frame["Type"] == "H").to_numpy()].ravel()
        assert np.median(hitters) < hitters.mean(), "DK hitter scoring is right-skewed"

    def test_ceiling_is_near_the_simulated_ninetieth(self, sims):
        frame, out = sims
        summary = player_summary(frame, out)
        gap = (summary["Sim p90"] - summary["Ceiling"]).abs().mean()
        assert gap < 4.0, f"stated ceiling and simulated p90 differ by {gap:.1f} points"


class TestCorrelation:
    def test_signs_and_ordering(self, sims):
        frame, out = sims
        report = correlation_report(frame, out).set_index("relationship")["mean r"]

        teammates = report["teammate hitters"]
        opposing = report["pitcher vs the hitters he faces"]
        cross = report["hitters in different games"]

        assert teammates > 0.10, f"teammates barely correlated ({teammates:+.3f})"
        assert opposing < -0.05, f"pitcher not negatively correlated with his opponents ({opposing:+.3f})"
        assert abs(cross) < 0.06, f"different games should be near-independent ({cross:+.3f})"
        assert teammates > cross, "teammates must outcorrelate strangers"

    def test_adjacent_batters_correlate_at_least_as_much_as_distant_ones(self, sims):
        frame, out = sims
        report = correlation_report(frame, out).set_index("relationship")["mean r"]
        adjacent = report["  adjacent in the order (<=2 apart)"]
        distant = report["  distant in the order"]
        assert adjacent >= distant - 0.01, \
            f"adjacent {adjacent:+.3f} should not trail distant {distant:+.3f}"

    def test_a_stack_has_a_fatter_tail_than_independent_players_would(self, sims):
        """The entire reason to stack. Compare a real team's four to four strangers."""
        frame, out = sims
        hitters = frame[frame["Type"] == "H"]
        team = hitters["Team"].iloc[0]
        stack = hitters[hitters["Team"] == team].head(4).index.to_numpy()
        spread = (hitters[hitters["Team"] != team]
                  .drop_duplicates("Team").head(4).index.to_numpy())

        stack_scores = out[:, stack].sum(axis=1)
        spread_scores = out[:, spread].sum(axis=1)
        # Normalise for the different means: it is the shape of the tail being compared.
        stack_ratio = np.percentile(stack_scores, 99) / max(stack_scores.mean(), 1e-9)
        spread_ratio = np.percentile(spread_scores, 99) / max(spread_scores.mean(), 1e-9)
        assert stack_ratio > spread_ratio, \
            f"stack tail {stack_ratio:.2f}x mean vs spread {spread_ratio:.2f}x"

    def test_team_runs_are_shared_not_duplicated(self, slate):
        """Teammates draw R and RBI from one pool, so their sum is bounded by team runs."""
        out, parts = simulate_slate(slate, n_sims=500, seed=3, return_components=True)
        runs = parts["hitter_runs"].sum(axis=1)
        team_total = parts["team_runs"].sum(axis=1)
        assert (runs <= team_total + 1e-9).all(), "hitters scored more runs than their teams"


class TestConfiguration:
    def test_more_team_dispersion_means_more_teammate_correlation(self, slate):
        low = simulate_slate(slate, n_sims=2500, seed=9,
                             config=SimConfig(team_sigma=0.05, run_dispersion=1.5))
        high = simulate_slate(slate, n_sims=2500, seed=9,
                              config=SimConfig(team_sigma=0.55, run_dispersion=1.5))
        pick = lambda out: correlation_report(slate, out).set_index(
            "relationship")["mean r"]["teammate hitters"]
        assert pick(high) > pick(low)

    def test_zero_shocks_still_produce_variance(self, slate):
        """Event sampling is variance in its own right, even with every shock switched off."""
        out = simulate_slate(slate, n_sims=500, seed=2,
                             config=SimConfig(slate_sigma=0, game_sigma=0, team_sigma=0,
                                              order_sigma=0, player_sigma=0))
        assert out.std() > 1.0

    @pytest.mark.parametrize("n", [100, 1000])
    def test_sim_counts_are_configurable(self, slate, n):
        assert simulate_slate(slate, n_sims=n, seed=1).shape[0] == n


class TestGuards:
    def test_a_frame_without_event_columns_is_refused(self, slate):
        stripped = slate.drop(columns=[c for c in slate.columns if c.startswith("E_")])
        with pytest.raises(SimulationError, match="expected-event"):
            simulate_slate(stripped, n_sims=10, seed=1)

    def test_empty_frame_is_refused(self, slate):
        with pytest.raises(SimulationError):
            simulate_slate(slate.head(0), n_sims=10, seed=1)


class TestValidateReport:
    def test_report_is_complete(self, sims):
        frame, out = sims
        report = validate(frame, out)
        assert report["n_sims"] == out.shape[0]
        assert set(report["marginals"]) == {"hitters", "pitchers"}
        assert len(report["correlations"]) == 6
