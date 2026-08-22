"""Payout structures, tie splitting, and contest EV.

The tie-splitting rule is the one to get right. Duplicate lineups score identically, so they
tie, so DK divides the prizes for the positions they occupy between them. Treating a tie as
"everyone gets the rank-r prize" values a 20-times-duplicated winning lineup at twenty times
what it actually pays -- which is precisely the error that makes duplication invisible.
"""

import json

import numpy as np
import pandas as pd
import pytest

from dfs.contest import (ContestError, Payout, _shared_prize, double_up, duplicate_counts,
                         evaluate, gpp_payout)


class TestPayoutStructure:
    def test_prizes_expand_from_tiers(self):
        payout = Payout("t", entry_fee=1.0, field_size=10,
                        tiers=((1, 100.0), (3, 20.0), (5, 5.0)))
        prizes = payout.prizes()
        assert prizes.tolist() == [100, 20, 20, 5, 5, 0, 0, 0, 0, 0]

    def test_paid_places_and_total(self):
        payout = Payout("t", 1.0, 10, ((1, 100.0), (5, 10.0)))
        assert payout.paid_places == 5
        assert payout.total_prizes == pytest.approx(140.0)

    def test_rake_is_computed(self):
        payout = Payout("t", entry_fee=10.0, field_size=10, tiers=((10, 9.0),))
        assert payout.rake == pytest.approx(0.10)

    def test_tiers_beyond_the_field_are_clipped(self):
        payout = Payout("t", 1.0, 3, ((1, 50.0), (100, 5.0)))
        assert len(payout.prizes()) == 3
        assert payout.prizes().tolist() == [50, 5, 5]

    def test_json_round_trip(self, tmp_path):
        path = tmp_path / "payout.json"
        path.write_text(json.dumps({"name": "real", "entry_fee": 3.0, "field_size": 100,
                                    "tiers": [[1, 500], [10, 20]]}), encoding="utf-8")
        payout = Payout.from_json(str(path))
        assert payout.name == "real"
        assert payout.entry_fee == 3.0
        assert payout.prizes()[0] == 500


class TestGeneratedPayouts:
    def test_gpp_respects_the_rake(self):
        payout = gpp_payout(6000, 5.0, rake=0.15)
        gross = 6000 * 5.0
        # Rounding to printable prizes moves the total a little.
        assert payout.total_prizes == pytest.approx(gross * 0.85, rel=0.05)

    def test_gpp_is_top_heavy(self):
        payout = gpp_payout(6000, 5.0)
        prizes = payout.prizes()
        assert prizes[0] > 20 * prizes[100], "first place should dwarf 100th"
        assert prizes[0] / payout.total_prizes > 0.10

    def test_gpp_pays_roughly_a_quarter_of_the_field(self):
        payout = gpp_payout(6000, 5.0)
        assert 0.15 < payout.paid_places / 6000 < 0.35

    def test_gpp_prizes_never_increase_with_rank(self):
        prizes = gpp_payout(3000, 10.0).prizes()
        assert np.all(np.diff(prizes) <= 0)

    def test_double_up_is_flat_to_half_the_field(self):
        payout = double_up(1000, 10.0, rake=0.10)
        prizes = payout.prizes()
        assert payout.paid_places == 500
        assert len(set(prizes[:500])) == 1
        assert prizes[500] == 0
        assert prizes[0] == pytest.approx(18.0, rel=0.02)


class TestTieSplitting:
    def _prizes(self):
        # 1st 100, 2nd 50, 3rd 10, rest nothing.
        return np.array([100.0, 50.0, 10.0, 0.0, 0.0])

    def test_an_outright_win_takes_first(self):
        got = _shared_prize(self._prizes(), np.array([1.0]), np.array([1.0]))
        assert got[0] == pytest.approx(100.0)

    def test_two_tied_for_first_split_first_and_second(self):
        got = _shared_prize(self._prizes(), np.array([1.0]), np.array([2.0]))
        assert got[0] == pytest.approx(75.0)

    def test_three_tied_for_first_split_the_top_three(self):
        got = _shared_prize(self._prizes(), np.array([1.0]), np.array([3.0]))
        assert got[0] == pytest.approx((100 + 50 + 10) / 3)

    def test_a_tie_straddling_the_cash_line_includes_the_zeros(self):
        """Otherwise duplication past the last paying place looks free."""
        got = _shared_prize(self._prizes(), np.array([3.0]), np.array([2.0]))
        assert got[0] == pytest.approx(5.0)      # (10 + 0) / 2

    def test_a_tie_outside_the_money_pays_nothing(self):
        got = _shared_prize(self._prizes(), np.array([4.0]), np.array([2.0]))
        assert got[0] == pytest.approx(0.0)

    def test_duplication_strictly_reduces_the_prize(self):
        prizes = self._prizes()
        values = [_shared_prize(prizes, np.array([1.0]), np.array([float(k)]))[0]
                  for k in (1, 2, 3, 4)]
        assert values == sorted(values, reverse=True)
        assert values[0] > values[-1]

    def test_vectorises_over_candidates(self):
        got = _shared_prize(self._prizes(), np.array([1.0, 2.0, 3.0]),
                            np.array([1.0, 1.0, 1.0]))
        assert got.tolist() == pytest.approx([100.0, 50.0, 10.0])


class TestDuplicateCounts:
    def test_counts_identical_field_entries(self):
        field = [[1, 2, 3], [1, 2, 3], [4, 5, 6]]
        counts = duplicate_counts([[3, 2, 1], [4, 5, 6], [7, 8, 9]], field)
        assert counts.tolist() == [2.0, 1.0, 0.0]

    def test_order_does_not_matter(self):
        assert duplicate_counts([[3, 1, 2]], [[1, 2, 3]])[0] == 1.0

    def test_scale_extrapolates_a_smaller_simulated_field(self):
        counts = duplicate_counts([[1, 2, 3]], [[1, 2, 3]], scale=16.7)
        assert counts[0] == pytest.approx(16.7)


@pytest.fixture(scope="module")
def scored():
    from conftest import make_slate
    from dfs.candidates import generate
    from dfs.field import ContestConfig, simulate_field
    from dfs.simulate import simulate_slate

    players = make_slate(n_games=6, seed=1)
    pool = generate(players, n_candidates=60, seed=3)
    sims = simulate_slate(pool.pool, n_sims=1200, seed=3)
    field, _ = simulate_field(
        pool.pool, config=ContestConfig("t", entries=600, max_entries_per_user=20), seed=3)
    payout = gpp_payout(600, 5.0)
    return pool, sims, field, payout, evaluate(pool, sims, field, payout)


class TestEvaluate:
    def test_every_candidate_gets_every_metric(self, scored):
        _pool, _sims, _field, _payout, result = scored
        for column in ("sim_mean", "sim_sd", "mean_rank", "p_win", "p_top01", "p_top1",
                       "p_cash", "exp_dupes", "p_unique", "exp_payout", "exp_profit", "roi"):
            assert column in result.columns
            assert result[column].notna().all(), column

    def test_probabilities_are_probabilities(self, scored):
        *_ , result = scored
        for column in ("p_win", "p_top01", "p_top1", "p_cash", "p_unique"):
            assert (result[column] >= 0).all() and (result[column] <= 1).all(), column

    def test_probability_ordering_is_nested(self, scored):
        """Winning implies top 0.1% implies top 1% implies cashing.

        Tolerance covers the rounding, not the logic: the columns are rounded to different
        decimal places for display (6 for p_win, 5 for p_top01, 4 for p_top1 and p_cash), so
        an exact comparison can fail on the rounding alone. It has to be half of the
        *coarsest* step -- 5e-5 for four decimals. At 1e-5 this test failed on a candidate
        that won exactly one simulation in 1,200: 0.000833 rounds to 0.00083 as p_top01 and
        to 0.0008 as p_top1, a 3.3e-5 gap with nothing wrong underneath it.
        """
        *_, result = scored
        tol = 5e-5
        assert (result["p_win"] <= result["p_top01"] + tol).all()
        assert (result["p_top01"] <= result["p_top1"] + tol).all()
        assert (result["p_top1"] <= result["p_cash"] + tol).all()

    def test_a_better_lineup_ranks_better(self, scored):
        *_, result = scored
        assert result["sim_mean"].corr(result["mean_rank"], method="spearman") < -0.5

    def test_a_higher_mean_ranks_better_against_the_field(self, scored):
        """The near-deterministic relationship: mean score and expected finishing rank.

        This is the one worth asserting tightly. Cash *probability* is a different and much
        looser question -- see below.
        """
        *_, result = scored
        assert result["sim_mean"].corr(result["mean_rank"], method="spearman") < -0.9

    def test_higher_mean_broadly_means_higher_cash_probability(self, scored):
        """Broadly, and deliberately not tightly.

        Cashing depends on the *shape* of a candidate's distribution, not only its mean, and
        the edge correction pushes the cash line further into the right tail where variance
        matters more. A high-variance candidate with a lower mean genuinely can out-cash a
        steadier one with a higher mean. On this fixture the candidates span only ~6 points
        of mean, so variance dominates and the rank correlation sits near 0.6 regardless of
        how many simulations are run (0.59 at 1,200, 0.56 at 3,000, 0.62 at 5,000).

        The threshold catches a broken monotonic relationship, which would land near zero,
        without claiming a precision the model does not have.
        """
        *_, result = scored
        assert result["sim_mean"].corr(result["p_cash"], method="spearman") > 0.4

    def test_profit_is_payout_minus_the_entry_fee(self, scored):
        *_, payout, result = scored
        assert np.allclose(result["exp_profit"],
                           result["exp_payout"] - payout.entry_fee, atol=1e-3)

    def test_roi_is_profit_over_the_entry_fee(self, scored):
        *_, payout, result = scored
        assert np.allclose(result["roi"], result["exp_profit"] / payout.entry_fee,
                           atol=1e-3)

    def test_mean_rank_is_inside_the_field(self, scored):
        *_, payout, result = scored
        assert (result["mean_rank"] >= 1).all()
        assert (result["mean_rank"] <= payout.field_size + 1).all()

    def test_sim_mean_matches_a_direct_computation(self, scored):
        pool, sims, _field, _payout, result = scored
        first = list(pool.lineups["players"].iloc[0])
        assert result["sim_mean"].iloc[0] == pytest.approx(
            float(sims[:, first].sum(axis=1).mean()), abs=0.05)

    def test_reproducible(self, scored):
        pool, sims, field, payout, result = scored
        again = evaluate(pool, sims, field, payout)
        pd.testing.assert_series_equal(result["exp_payout"], again["exp_payout"])

    def test_no_candidates_is_an_error(self, scored):
        pool, sims, field, payout, _result = scored
        empty = type(pool)(pool.lineups.head(0), pool.pool, pool.meta)
        with pytest.raises(ContestError):
            evaluate(empty, sims, field, payout)

    def test_no_field_is_an_error(self, scored):
        pool, sims, _field, payout, _result = scored
        with pytest.raises(ContestError, match="field"):
            evaluate(pool, sims, [], payout)


class TestPayoutSensitivity:
    def test_a_double_up_and_a_gpp_disagree_about_the_best_lineup(self):
        """They should: one pays for clearing a line, the other for winning outright."""
        from conftest import make_slate
        from dfs.candidates import generate
        from dfs.field import ContestConfig, simulate_field
        from dfs.simulate import simulate_slate

        players = make_slate(n_games=6, seed=1)
        pool = generate(players, n_candidates=80, seed=5)
        sims = simulate_slate(pool.pool, n_sims=1500, seed=5)
        field, _ = simulate_field(
            pool.pool, config=ContestConfig("t", entries=600, max_entries_per_user=20),
            seed=5)

        gpp = evaluate(pool, sims, field, gpp_payout(600, 5.0))
        cash = evaluate(pool, sims, field, double_up(600, 5.0))
        # Cash rewards floor, GPP rewards ceiling, so the rankings must not be identical.
        agreement = gpp["exp_payout"].corr(cash["exp_payout"], method="spearman")
        assert agreement < 0.995
        assert cash["p_cash"].max() > gpp["p_win"].max()
