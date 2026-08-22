"""DraftKings Classic scoring.

The one function everything else agrees through: projections build events and score them
here, the simulator draws events and scores them here, the review reads a box score and
scores it here. A silent change to this table would move every number in the project in the
same direction and look like a model improvement, so it is pinned against DK's published
rules rather than against the code's own output.
"""

import pytest

from dfs.scoring import (DK_CLASSIC_SLOTS, DK_HITTER, DK_PITCHER, DK_SALARY_CAP,
                         hitter_points, pitcher_points)


class TestHitterScoring:
    def test_published_values(self):
        # DraftKings MLB Classic, as published.
        assert DK_HITTER == {"1B": 3.0, "2B": 5.0, "3B": 8.0, "HR": 10.0, "RBI": 2.0,
                             "R": 2.0, "BB": 2.0, "HBP": 2.0, "SB": 5.0}

    def test_empty_line_scores_zero(self):
        assert hitter_points({}) == 0.0
        assert hitter_points({"1B": 0, "HR": 0}) == 0.0

    def test_single_with_a_run(self):
        assert hitter_points({"1B": 1, "R": 1}) == 5.0

    def test_solo_home_run(self):
        # 10 for the homer, 2 for the run, 2 for the RBI.
        assert hitter_points({"HR": 1, "R": 1, "RBI": 1}) == 14.0

    def test_grand_slam(self):
        assert hitter_points({"HR": 1, "R": 1, "RBI": 4}) == 20.0

    def test_full_line(self):
        line = {"1B": 2, "2B": 1, "3B": 1, "HR": 1, "R": 3, "RBI": 4, "BB": 1,
                "HBP": 1, "SB": 2}
        expected = 6 + 5 + 8 + 10 + 6 + 8 + 2 + 2 + 10
        assert hitter_points(line) == pytest.approx(expected)

    def test_fractional_events_are_linear(self):
        """Projections pass expected counts, not integers, and must scale linearly."""
        assert hitter_points({"HR": 0.5}) == pytest.approx(5.0)
        assert hitter_points({"1B": 1.5, "BB": 0.25}) == pytest.approx(5.0)

    def test_unknown_keys_are_ignored(self):
        assert hitter_points({"1B": 1, "GIDP": 3, "CS": 2}) == 3.0


class TestPitcherScoring:
    def test_published_values(self):
        for key, value in {"IP": 2.25, "K": 2.0, "W": 4.0, "ER": -2.0, "H": -0.6,
                           "BB": -0.6, "HBP": -0.6}.items():
            assert DK_PITCHER[key] == value

    def test_innings_are_worth_three_quarters_per_out(self):
        assert pitcher_points({"IP": 1}) == pytest.approx(2.25)
        assert pitcher_points({"IP": 1 / 3}) == pytest.approx(0.75)

    def test_quality_start_with_a_win(self):
        # 6 IP, 7 K, 1 ER, 4 H, 2 BB, win.
        score = pitcher_points({"IP": 6, "K": 7, "W": 1, "ER": 1, "H": 4, "BB": 2})
        assert score == pytest.approx(6 * 2.25 + 14 + 4 - 2 - 2.4 - 1.2)

    def test_a_blowup_can_score_negative(self):
        score = pitcher_points({"IP": 1, "K": 0, "W": 0, "ER": 7, "H": 9, "BB": 3})
        assert score < 0

    def test_complete_game_bonuses_are_excluded(self):
        """Documented omission: at fractional probability they are worth <0.1 points."""
        assert pitcher_points({"IP": 9, "CG": 1, "CGSO": 1, "NH": 1}) == \
            pytest.approx(pitcher_points({"IP": 9}))

    def test_win_probability_is_linear(self):
        assert pitcher_points({"W": 0.44}) == pytest.approx(1.76)


class TestRosterConstants:
    def test_classic_roster_shape(self):
        assert DK_CLASSIC_SLOTS == ["P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"]
        assert len(DK_CLASSIC_SLOTS) == 10

    def test_salary_cap(self):
        assert DK_SALARY_CAP == 50000
