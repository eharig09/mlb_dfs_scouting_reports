"""DK NFL Classic scoring.

Named `test_nfl_*` rather than `test_scoring.py` on purpose: pytest identifies test modules
by basename when the directory is not a package, so a second `test_scoring.py` alongside the
MLB one collides at import with a confusing "import file mismatch".
"""

import pytest

from nfl.scoring import (DK_CLASSIC_SLOTS, DK_SALARY_CAP, DST_POINTS_ALLOWED,
                         FLEX_POSITIONS, dst_points, expected_points_allowed_points,
                         offense_points, points_allowed_points)


class TestRosterShape:
    def test_classic_is_nine_players(self):
        assert len(DK_CLASSIC_SLOTS) == 9

    def test_the_cap_is_fifty_thousand(self):
        assert DK_SALARY_CAP == 50000

    def test_flex_takes_rb_wr_te_and_nothing_else(self):
        assert FLEX_POSITIONS == {"RB", "WR", "TE"}
        assert "QB" not in FLEX_POSITIONS and "DST" not in FLEX_POSITIONS


class TestOffense:
    def test_empty_line_scores_nothing(self):
        assert offense_points({}) == 0.0

    def test_passing_yards_pay_a_point_per_twenty_five(self):
        assert offense_points({"PASS_YD": 250}) == pytest.approx(10.0)

    def test_rushing_and_receiving_yards_pay_a_point_per_ten(self):
        assert offense_points({"RUSH_YD": 100}) == pytest.approx(10.0)
        assert offense_points({"REC_YD": 100}) == pytest.approx(10.0)

    def test_a_reception_is_worth_a_point(self):
        assert offense_points({"REC": 6}) == pytest.approx(6.0)

    def test_touchdowns_are_four_passing_and_six_everywhere_else(self):
        assert offense_points({"PASS_TD": 1}) == pytest.approx(4.0)
        assert offense_points({"RUSH_TD": 1}) == pytest.approx(6.0)
        assert offense_points({"REC_TD": 1}) == pytest.approx(6.0)

    def test_turnovers_are_negative(self):
        assert offense_points({"INT": 2}) == pytest.approx(-2.0)
        assert offense_points({"FUM_LOST": 1}) == pytest.approx(-1.0)

    def test_a_full_line_adds_up(self):
        line = {"PASS_YD": 300, "PASS_TD": 3, "INT": 1, "RUSH_YD": 20, "RUSH_TD": 1}
        # 12 + 12 - 1 + 2 + 6
        assert offense_points(line) == pytest.approx(31.0)


class TestYardageBonuses:
    """The step-function bonuses are the part a naive implementation gets wrong."""

    def test_a_bonus_is_not_inferred_from_the_yardage_line(self):
        # 100 rushing yards on its own pays 10, not 13. The bonus is a separate claim.
        assert offense_points({"RUSH_YD": 100}) == pytest.approx(10.0)

    def test_a_bonus_probability_is_paid_pro_rata(self):
        assert offense_points({"P_RUSH_100": 0.5}) == pytest.approx(1.5)
        assert offense_points({"P_REC_100": 0.25}) == pytest.approx(0.75)
        assert offense_points({"P_PASS_300": 1.0}) == pytest.approx(3.0)

    def test_certainty_is_the_completed_game_case(self):
        projected = offense_points({"RUSH_YD": 100, "P_RUSH_100": 0.4})
        realised = offense_points({"RUSH_YD": 100, "P_RUSH_100": 1.0})
        assert realised == pytest.approx(13.0)
        assert projected < realised


class TestDefense:
    def test_counting_stats_add_up(self):
        events = {"SACK": 3, "INT": 1, "FUM_REC": 1}
        assert dst_points(events, points_allowed=24) == pytest.approx(3 + 2 + 2 + 0)

    def test_every_points_allowed_bucket_is_reachable(self):
        expected = [(0, 10.0), (3, 7.0), (6, 7.0), (7, 4.0), (13, 4.0),
                    (14, 1.0), (20, 1.0), (21, 0.0), (27, 0.0),
                    (28, -1.0), (34, -1.0), (35, -4.0), (59, -4.0)]
        for allowed, points in expected:
            assert points_allowed_points(allowed) == points, allowed

    def test_a_shutout_is_the_best_bucket(self):
        assert points_allowed_points(0) == max(v for _, v in DST_POINTS_ALLOWED)

    def test_the_distribution_is_not_the_mean_s_bucket(self):
        """The reason the API takes a distribution at all.

        Points allowed enters through a step function, so E[f(X)] and f(E[X]) are simply
        different numbers -- and, because the ladder is neither convex nor concave across
        its whole range, the error has no fixed sign. Mass on a shutout makes the mean
        UNDERSTATE (the +10 bucket is a large jump above its neighbour); mass past 35 makes
        it OVERSTATE (the -4 bucket is an equally large drop). Either way a projection that
        hands over a mean is wrong, which is why the parameter is a distribution.
        """
        understated = {0: 0.2, 17: 0.6, 24: 0.2}
        mean = sum(k * v for k, v in understated.items())
        assert mean == pytest.approx(15.0)
        assert points_allowed_points(mean) == 1.0
        assert expected_points_allowed_points(understated) == pytest.approx(2.6)

        overstated = {20: 0.5, 45: 0.5}
        mean = sum(k * v for k, v in overstated.items())
        assert mean == pytest.approx(32.5)
        assert points_allowed_points(mean) == -1.0
        assert expected_points_allowed_points(overstated) == pytest.approx(-1.5)

    def test_a_symmetric_spread_can_agree_with_its_mean(self):
        """Not a contradiction of the above -- a reminder that the two coincide by accident
        rather than by rule, so agreeing on one slate proves nothing about the next."""
        spread = {7: 0.25, 14: 0.25, 28: 0.25, 35: 0.25}
        assert points_allowed_points(21.0) == 0.0
        assert expected_points_allowed_points(spread) == pytest.approx(0.0)

    def test_a_distribution_wins_over_a_scalar_when_both_are_given(self):
        spread = {0: 1.0}
        assert dst_points({}, points_allowed=35,
                          points_allowed_distribution=spread) == pytest.approx(10.0)

    def test_no_points_allowed_information_scores_only_the_counting_stats(self):
        assert dst_points({"SACK": 4}) == pytest.approx(4.0)
