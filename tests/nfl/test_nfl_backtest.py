"""Scoring real box scores.

Unit tests run on a synthetic weekly frame and need no network. The reconciliation against
nflverse's own scoring is the one thing that cannot be faked -- it is only meaningful over
real data -- so it is marked `integration` and skips when nflverse is unreachable.
"""

import pandas as pd
import pytest

from nfl.backtest import (dst_actuals, offense_actuals, reconcile_with_nflverse)
from nfl.data import current_season


def weekly_frame(rows):
    """A weekly frame carrying only the columns a test names; the rest default to zero."""
    base = {"season": 2024, "week": 1, "player_id": "00-0000001", "position": "WR",
            "player_display_name": "Test Player", "team": "BUF", "opponent_team": "MIA",
            "game_id": "2024_01_MIA_BUF"}
    return pd.DataFrame([{**base, **row} for row in rows])


class TestOffenseActuals:
    def test_a_receiving_line_scores(self):
        out = offense_actuals(weekly_frame([
            {"receptions": 8, "receiving_yards": 90, "receiving_tds": 1}]))
        # 8 rec + 9.0 yards + 6 TD
        assert out.iloc[0]["dk_points"] == pytest.approx(23.0)

    def test_a_passing_line_scores(self):
        out = offense_actuals(weekly_frame([
            {"position": "QB", "passing_yards": 275, "passing_tds": 2,
             "passing_interceptions": 1, "rushing_yards": 30}]))
        # 11.0 + 8 - 1 + 3.0
        assert out.iloc[0]["dk_points"] == pytest.approx(21.0)

    def test_the_three_hundred_yard_bonus_fires_at_exactly_three_hundred(self):
        under = offense_actuals(weekly_frame([{"position": "QB", "passing_yards": 299}]))
        exact = offense_actuals(weekly_frame([{"position": "QB", "passing_yards": 300}]))
        assert under.iloc[0]["dk_points"] == pytest.approx(11.96)
        assert exact.iloc[0]["dk_points"] == pytest.approx(12.0 + 3.0)

    def test_the_hundred_yard_bonuses_fire_at_exactly_one_hundred(self):
        rush = offense_actuals(weekly_frame([{"position": "RB", "rushing_yards": 100}]))
        rec = offense_actuals(weekly_frame([{"receiving_yards": 100}]))
        assert rush.iloc[0]["dk_points"] == pytest.approx(13.0)
        assert rec.iloc[0]["dk_points"] == pytest.approx(13.0)

    def test_both_yardage_bonuses_can_land_on_one_player(self):
        out = offense_actuals(weekly_frame([
            {"position": "RB", "rushing_yards": 110, "receiving_yards": 105}]))
        # 11.0 + 10.5 + 3 + 3
        assert out.iloc[0]["dk_points"] == pytest.approx(27.5)

    def test_a_lost_fumble_costs_one_not_two(self):
        out = offense_actuals(weekly_frame([{"receptions": 1, "fumbles_lost_total": 1}]))
        assert out.iloc[0]["dk_points"] == pytest.approx(0.0)

    def test_two_point_conversions_fold_together(self):
        out = offense_actuals(weekly_frame([
            {"position": "QB", "passing_2pt_conversions": 1,
             "rushing_2pt_conversions": 1, "receiving_2pt_conversions": 1}]))
        assert out.iloc[0]["dk_points"] == pytest.approx(6.0)
        assert out.iloc[0]["E_TWO_PT"] == 3

    def test_return_and_recovery_touchdowns_pay_six(self):
        out = offense_actuals(weekly_frame([
            {"special_teams_tds": 1, "fumble_recovery_tds": 1}]))
        assert out.iloc[0]["dk_points"] == pytest.approx(12.0)

    def test_a_missing_column_is_treated_as_zero(self):
        """nflverse renamed fields across eras, so a backtest spanning both must not die on
        the boundary."""
        frame = weekly_frame([{"receptions": 3, "receiving_yards": 40}])
        out = offense_actuals(frame.drop(columns=["opponent_team"]))
        assert out.iloc[0]["dk_points"] == pytest.approx(7.0)

    def test_non_offensive_positions_are_excluded(self):
        frame = weekly_frame([{"position": "CB", "def_interceptions": 2},
                              {"position": "WR", "receptions": 4}])
        out = offense_actuals(frame)
        assert len(out) == 1 and out.iloc[0]["position"] == "WR"

    def test_an_empty_frame_gives_an_empty_result(self):
        assert offense_actuals(pd.DataFrame()).empty
        assert offense_actuals(None).empty

    def test_event_columns_are_carried_through(self):
        out = offense_actuals(weekly_frame([{"receiving_yards": 120, "receptions": 6}]))
        assert out.iloc[0]["E_REC_YD"] == 120
        assert out.iloc[0]["E_P_REC_100"] == 1.0


class TestDstActuals:
    def schedules(self, home="BUF", away="MIA", home_score=17, away_score=24):
        return pd.DataFrame([{"season": 2024, "week": 1, "game_id": "2024_01_MIA_BUF",
                              "home_team": home, "away_team": away,
                              "home_score": home_score, "away_score": away_score}])

    def test_a_defense_line_aggregates_its_players(self):
        frame = weekly_frame([
            {"position": "LB", "def_sacks": 2, "def_interceptions": 1},
            {"position": "CB", "def_sacks": 1, "fumble_recovery_opp": 1},
        ])
        out = dst_actuals(frame, self.schedules())
        row = out[out["team"] == "BUF"].iloc[0]
        assert row["SACK"] == 3 and row["INT"] == 1 and row["FUM_REC"] == 1
        # 3 sacks + 2 int + 2 fum, allowing 24 -> the 21-27 bucket pays 0
        assert row["points_allowed"] == 24
        assert row["dk_points"] == pytest.approx(7.0)

    def test_points_allowed_comes_from_the_scoreboard(self):
        frame = weekly_frame([{"position": "LB"}])
        out = dst_actuals(frame, self.schedules(home_score=0, away_score=0))
        assert out[out["team"] == "BUF"].iloc[0]["dk_points"] == pytest.approx(10.0)

    def test_every_kind_of_touchdown_counts(self):
        frame = weekly_frame([
            {"position": "CB", "def_tds": 1},
            {"position": "WR", "special_teams_tds": 1},
            {"position": "LB", "fumble_recovery_tds": 1},
        ])
        out = dst_actuals(frame, self.schedules())
        assert out[out["team"] == "BUF"].iloc[0]["TD"] == 3

    def test_blocks_are_summed_across_their_three_columns(self):
        frame = weekly_frame([{"position": "LB", "def_fg_blocks": 1,
                               "def_pat_blocks": 1, "def_punt_blocks": 1}])
        out = dst_actuals(frame, self.schedules())
        assert out[out["team"] == "BUF"].iloc[0]["BLK"] == 3

    def test_an_empty_input_gives_an_empty_result(self):
        assert dst_actuals(pd.DataFrame(), self.schedules()).empty
        assert dst_actuals(weekly_frame([{}]), pd.DataFrame()).empty


class TestReconciliation:
    """The differences between DK and nflverse's standard scoring, one at a time."""

    def frame(self, row, ppr):
        weekly = weekly_frame([row])
        weekly["fantasy_points_ppr"] = ppr
        return weekly

    def check(self, row, ppr):
        weekly = self.frame(row, ppr)
        result = reconcile_with_nflverse(offense_actuals(weekly), weekly)
        return result.iloc[0]

    def test_a_plain_line_has_no_difference(self):
        assert self.check({"receptions": 5, "receiving_yards": 60}, 11.0)["residual"] == 0.0

    def test_an_interception_differs_by_one(self):
        row = self.check({"position": "QB", "passing_yards": 200,
                          "passing_interceptions": 1}, 6.0)
        assert row["expected_delta"] == 1.0 and row["residual"] == 0.0

    def test_an_offensive_fumble_differs_by_one(self):
        # 2 receptions is 2 points either way; standard then charges -2 for the fumble and
        # DK charges -1, so nflverse lands on 0.0 and we land on 1.0.
        row = self.check({"receptions": 2, "rushing_fumbles_lost": 1,
                          "fumbles_lost_total": 1}, 0.0)
        assert row["dk_points"] == 1.0
        assert row["offensive_fumbles_lost"] == 1
        assert row["expected_delta"] == 1.0 and row["residual"] == 0.0

    def test_a_return_fumble_differs_the_other_way(self):
        """The case the reconciliation discovered: nflverse charges nothing for a fumble
        lost on a punt or kick return, DraftKings charges -1."""
        row = self.check({"receptions": 2, "fumbles_lost_total": 1}, 2.0)
        assert row["offensive_fumbles_lost"] == 0
        assert row["return_fumbles_lost"] == 1
        assert row["expected_delta"] == -1.0 and row["residual"] == 0.0

    def test_a_yardage_bonus_differs_by_three(self):
        row = self.check({"receptions": 5, "receiving_yards": 110}, 16.0)
        assert row["expected_delta"] == 3.0 and row["residual"] == 0.0

    def test_a_fumble_recovery_touchdown_differs_by_six(self):
        row = self.check({"receptions": 1, "fumble_recovery_tds": 1}, 1.0)
        assert row["expected_delta"] == 6.0 and row["residual"] == 0.0


@pytest.mark.integration
class TestAgainstRealSeasons:
    def test_a_full_season_reconciles_exactly(self):
        """23,737 player-weeks over 2021-24 reconcile with zero residual. One season here
        keeps the test quick; widen the range if a scoring change needs more confidence."""
        from nfl.data import load_weekly
        try:
            weekly = load_weekly([2024])
        except Exception as error:
            pytest.skip(f"nflverse unreachable: {error}")
        if weekly.empty:
            pytest.skip("no 2024 weekly data available")
        result = reconcile_with_nflverse(offense_actuals(weekly), weekly)
        assert len(result) > 5000
        assert (result["residual"].abs() > 0.01).sum() == 0


class TestSeasonBoundary:
    def test_january_belongs_to_the_previous_season(self):
        """A season is named for the year it starts in and runs into February; reading the
        calendar year would file the Super Bowl under a season that has not kicked off."""
        from datetime import datetime
        assert current_season(datetime(2025, 1, 12)) == 2024
        assert current_season(datetime(2025, 2, 9)) == 2024
        assert current_season(datetime(2025, 3, 1)) == 2025
        assert current_season(datetime(2025, 9, 7)) == 2025
        assert current_season(datetime(2025, 12, 25)) == 2025
