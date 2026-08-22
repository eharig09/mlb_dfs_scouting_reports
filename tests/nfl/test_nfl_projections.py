"""Projection mechanics.

Synthetic and offline. The things worth pinning down here are structural rather than
numerical: that the walk-forward cutoff cannot see the week it is projecting, that shrinkage
moves toward the prior in the right direction and amount, and that the implied-total sign
convention stays the way it was measured. The question of whether the model is any *good*
is not a unit test -- that is `walk_forward` over completed seasons.
"""

import numpy as np
import pandas as pd
import pytest

from nfl.projections import (BAND_FIT, EFFICIENCY_SHRINK, TD_SHRINK, VOLUME_SHRINK,
                             YARDAGE_CV, _before, _bonus_probability, _shrink,
                             _points_allowed_spread, _weighted_rates, implied_totals,
                             positional_priors)


def weekly_frame(rows):
    base = {"season": 2024, "week": 1, "player_id": "p1", "position": "WR",
            "player_display_name": "Test", "team": "BUF", "opponent_team": "MIA"}
    return pd.DataFrame([{**base, **r} for r in rows])


class TestWalkForwardCutoff:
    """The guarantee the whole evaluation rests on."""

    def frame(self):
        return weekly_frame([
            {"season": 2023, "week": 17, "targets": 5},
            {"season": 2024, "week": 1, "targets": 6},
            {"season": 2024, "week": 2, "targets": 7},
            {"season": 2024, "week": 3, "targets": 8},
        ])

    def test_the_target_week_is_excluded(self):
        out = _before(self.frame(), 2024, 3)
        # Checked within the target season: a week 17 from a PRIOR season is older than
        # week 3 of this one and must survive, so a bare max() over week is not the test.
        assert out[out["season"] == 2024]["week"].max() == 2
        assert 8 not in out["targets"].values
        assert len(out) == 3

    def test_earlier_seasons_are_included(self):
        out = _before(self.frame(), 2024, 1)
        assert len(out) == 1 and out.iloc[0]["season"] == 2023

    def test_week_one_of_the_first_season_leaves_nothing(self):
        assert _before(self.frame(), 2023, 1).empty


class TestShrinkage:
    def test_no_sample_returns_the_prior(self):
        assert _shrink(99.0, 5.0, 0, 10.0) == 5.0

    def test_a_nan_observation_returns_the_prior(self):
        assert _shrink(np.nan, 5.0, 100, 10.0) == 5.0

    def test_more_sample_moves_toward_the_observation(self):
        near = _shrink(10.0, 0.0, 1, 10.0)
        far = _shrink(10.0, 0.0, 100, 10.0)
        assert near < far < 10.0
        assert near == pytest.approx(10.0 * 1 / 11)

    def test_touchdown_rates_are_shrunk_far_harder_than_volume(self):
        """The measured week-to-week reliability of a touchdown rate is near 0.1 against
        0.68 for carries, so this ordering is the model's central claim."""
        assert TD_SHRINK > EFFICIENCY_SHRINK > max(VOLUME_SHRINK.values()) * 10

    def test_a_hot_streak_barely_moves_a_touchdown_rate(self):
        """40 targets with a 25% TD rate against a 5% prior should still land near the
        prior -- otherwise the model is projecting last month's luck forward."""
        shrunk = _shrink(0.25, 0.05, 40, TD_SHRINK)
        assert shrunk < 0.09


class TestWeightedRates:
    def test_recent_games_weigh_more(self):
        frame = weekly_frame([{"week": 1, "targets": 0}, {"week": 2, "targets": 10}])
        assert _weighted_rates(frame)["targets"] > 5.0

    def test_an_empty_frame_gives_nothing(self):
        assert _weighted_rates(weekly_frame([]).iloc[0:0]) is None

    def test_efficiency_divides_weighted_totals_not_per_game_ratios(self):
        """A one-target game must not count as much as a ten-target game."""
        frame = weekly_frame([
            {"week": 1, "targets": 10, "receiving_yards": 100},
            {"week": 2, "targets": 1, "receiving_yards": 0},
        ])
        form = _weighted_rates(frame)
        assert form["rec_yards"] / form["rec_targets"] > 0.0
        # Per-game ratios would average 10 and 0 to 5; weighted totals must not.
        assert form["rec_yards"] / form["rec_targets"] != pytest.approx(5.0)

    def test_lookback_caps_the_history(self):
        frame = weekly_frame([{"week": w, "targets": 1} for w in range(1, 40)])
        assert _weighted_rates(frame, lookback=10)["games"] == 10


class TestImpliedTotals:
    def schedule(self, spread, total=45.0):
        return pd.DataFrame([{"season": 2024, "week": 1, "game_id": "g",
                              "home_team": "BUF", "away_team": "MIA",
                              "spread_line": spread, "total_line": total}])

    def test_a_positive_spread_favours_the_home_team(self):
        """Measured, not assumed: corr(spread_line, home margin) is +0.45 over 2022-25.
        Reading it the other way inverts every implied total while still looking plausible."""
        out = implied_totals(self.schedule(spread=7.0))
        home = out[out["team"] == "BUF"].iloc[0]
        away = out[out["team"] == "MIA"].iloc[0]
        assert home["implied"] == pytest.approx(26.0)
        assert away["implied"] == pytest.approx(19.0)

    def test_the_two_sides_sum_to_the_total(self):
        out = implied_totals(self.schedule(spread=-3.5, total=48.0))
        assert out["implied"].sum() == pytest.approx(48.0)

    def test_each_side_sees_the_other_as_its_opponent_total(self):
        out = implied_totals(self.schedule(spread=6.0))
        home = out[out["team"] == "BUF"].iloc[0]
        away = out[out["team"] == "MIA"].iloc[0]
        assert home["opp_implied"] == pytest.approx(away["implied"])

    def test_games_without_a_line_are_dropped(self):
        frame = self.schedule(spread=3.0)
        frame.loc[0, "total_line"] = np.nan
        assert implied_totals(frame).empty


class TestBonusProbability:
    def test_zero_projection_gives_zero(self):
        assert _bonus_probability(0, 100, 0.7) == 0.0

    def test_it_rises_with_the_projection(self):
        low = _bonus_probability(50, 100, 0.7)
        high = _bonus_probability(95, 100, 0.7)
        assert 0 < low < high < 1

    def test_it_is_well_under_a_half_at_the_threshold(self):
        """A right-skewed distribution has its median below its mean, so a player projected
        for exactly 100 yards clears 100 less than half the time. A normal would say 50% and
        overpay the bonus on every mid-range back on the slate."""
        assert _bonus_probability(100, 100, 0.74) < 0.45

    def test_more_dispersion_fattens_the_tail(self):
        assert _bonus_probability(60, 100, 1.0) > _bonus_probability(60, 100, 0.4)

    def test_passing_is_fitted_tighter_than_receiving(self):
        """A quarterback's yardage is the sum of thirty attempts and concentrates; a
        receiver's is the sum of six targets and does not."""
        assert YARDAGE_CV["QB_pass"] < YARDAGE_CV["WR_rec"]


class TestPointsAllowedSpread:
    def test_it_is_a_distribution(self):
        spread = _points_allowed_spread(21.0)
        assert sum(spread.values()) == pytest.approx(1.0)
        assert all(p >= 0 for p in spread.values())

    def test_it_centres_on_the_implied_total(self):
        spread = _points_allowed_spread(24.0)
        mean = sum(total * p for total, p in spread.items())
        assert mean == pytest.approx(24.0, abs=0.5)

    def test_a_lower_implied_total_shifts_mass_down(self):
        low = _points_allowed_spread(13.0)
        high = _points_allowed_spread(30.0)
        assert sum(p for t, p in low.items() if t <= 13) > \
               sum(p for t, p in high.items() if t <= 13)


class TestPriorsAndBands:
    def test_priors_cover_every_position(self):
        history = pd.concat([
            weekly_frame([{"position": p, "targets": 5, "receptions": 3,
                           "receiving_yards": 40, "attempts": 30, "carries": 10,
                           "passing_yards": 220, "rushing_yards": 45}])
            for p in ("QB", "RB", "WR", "TE")])
        priors = positional_priors(history)
        assert set(priors) == {"QB", "RB", "WR", "TE"}
        assert priors["WR"]["catch_rate"] == pytest.approx(0.6)

    def test_bands_are_fitted_per_position(self):
        assert set(BAND_FIT) == {"QB", "RB", "WR", "TE"}
        for position, band in BAND_FIT.items():
            assert band["ceiling"][0] > band["floor"][0], position

    def test_the_quarterback_ceiling_is_flatter_than_the_skill_positions(self):
        """A QB accumulates from every drive, so his upside is far less projection-
        dependent than a receiver's who needs the ball to find him."""
        assert BAND_FIT["QB"]["ceiling"][0] < BAND_FIT["WR"]["ceiling"][0]
        assert BAND_FIT["QB"]["floor"][0] > BAND_FIT["WR"]["floor"][0]
