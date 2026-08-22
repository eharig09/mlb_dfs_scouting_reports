"""Red-zone touchdown equity, and positional strength of schedule.

Named `test_nfl_*` and carrying no conftest, per the collision documented in
tests/nfl/nfl_fixtures.py.

The fitted constants are checked for the properties they must hold (ordering, nesting,
direction) rather than for their exact values, which come from `fantasy-stats-*` and will
move when the files are re-exported.
"""

import numpy as np
import pandas as pd
import pytest

from nfl import redzone as rz
from nfl import sos


class TestTierRates:
    def test_closer_to_the_end_zone_is_worth_more(self):
        assert (rz.RECEIVING_TD_RATE["end_zone"]
                > rz.RECEIVING_TD_RATE["red_zone"]
                > rz.RECEIVING_TD_RATE["open_field"])
        assert (rz.RUSHING_TD_RATE["inside_five"]
                > rz.RUSHING_TD_RATE["red_zone"]
                > rz.RUSHING_TD_RATE["open_field"])

    def test_the_spread_is_the_whole_point(self):
        """An end-zone target is worth ~66 open-field ones.

        If this ratio ever collapses toward 1, tracking where a target came from has stopped
        buying anything and the module is dead weight.
        """
        ratio = rz.RECEIVING_TD_RATE["end_zone"] / rz.RECEIVING_TD_RATE["open_field"]
        assert ratio > 20


class TestNesting:
    """End zone sits inside red zone; inside-five sits inside red zone.

    Checked against all 559 rows of the 2025 file with zero violations. Converting each
    bucket and summing would count the same target two and three times.
    """

    def test_an_end_zone_target_is_not_also_charged_as_a_red_zone_target(self):
        only_end_zone = rz.expected_receiving_tds([1], [1], [1])[0]
        assert only_end_zone == pytest.approx(rz.RECEIVING_TD_RATE["end_zone"])

    def test_each_tier_prices_at_its_own_rate(self):
        assert rz.expected_receiving_tds([1], [1], [0])[0] == pytest.approx(
            rz.RECEIVING_TD_RATE["red_zone"])
        assert rz.expected_receiving_tds([1], [0], [0])[0] == pytest.approx(
            rz.RECEIVING_TD_RATE["open_field"])
        assert rz.expected_rushing_tds([1], [1], [1])[0] == pytest.approx(
            rz.RUSHING_TD_RATE["inside_five"])
        assert rz.expected_rushing_tds([1], [0], [0])[0] == pytest.approx(
            rz.RUSHING_TD_RATE["open_field"])

    def test_a_mixed_workload_adds_up_by_tier(self):
        # 10 targets: 2 end zone, 3 more inside the 20, 5 outside it.
        got = rz.expected_receiving_tds([10], [5], [2])[0]
        want = (2 * rz.RECEIVING_TD_RATE["end_zone"]
                + 3 * rz.RECEIVING_TD_RATE["red_zone"]
                + 5 * rz.RECEIVING_TD_RATE["open_field"])
        assert got == pytest.approx(want)

    def test_rounding_noise_cannot_produce_a_negative_touchdown(self):
        """Season aggregates are rounded independently, so inner can exceed outer."""
        assert rz.expected_receiving_tds([1], [1], [3])[0] >= 0
        assert rz.expected_receiving_tds([0], [5], [5])[0] >= 0
        assert rz.expected_rushing_tds([2], [9], [9])[0] >= 0

    def test_more_opportunity_never_scores_less(self):
        low = rz.expected_receiving_tds([10], [3], [1])[0]
        high = rz.expected_receiving_tds([10], [6], [1])[0]
        assert high > low


class TestVectorised:
    def test_it_takes_columns_not_just_scalars(self):
        frame = pd.DataFrame({"t": [10, 20, 0], "rz": [5, 2, 0], "ez": [2, 0, 0]})
        out = rz.expected_receiving_tds(frame.t, frame.rz, frame.ez)
        assert len(out) == 3
        assert out[2] == 0.0

    def test_a_player_with_no_work_scores_nothing(self):
        assert rz.expected_receiving_tds([0], [0], [0])[0] == 0.0
        assert rz.expected_rushing_tds([0], [0], [0])[0] == 0.0


class TestNameKey:
    def test_it_folds_suffixes_and_accents(self):
        assert rz.name_key("Deebo Samuel Sr.") == rz.name_key("Deebo Samuel")
        assert rz.name_key("Marvin Harrison Jr.") == rz.name_key("Marvin Harrison")
        assert rz.name_key("Amon-Ra St. Brown") == rz.name_key("AmonRa St Brown")


class TestSeasonMap:
    def test_the_newest_season_is_mapped_to_the_unnumbered_file(self):
        """Empirically identified, and the conventions disagree between families.

        `fantasy-stats-receiving` uses the unnumbered file for the newest season while
        `defense_coverage_scheme` uses it for the oldest, so this cannot be inferred from
        the filename and must stay pinned.
        """
        assert rz.PFF_FILES["fantasy_receiving"][2025] == "pff/fantasy-stats-receiving.csv"
        assert rz.PFF_FILES["fantasy_receiving"][2024] == "pff/fantasy-stats-receiving (1).csv"

    def test_an_unmapped_season_fails_loudly(self):
        with pytest.raises(KeyError):
            rz.load_fantasy_receiving(1999)


class TestSos:
    def test_higher_means_easier(self):
        """Verified at +0.506 against 2025 WR PPR allowed.

        Reading it backwards inverts every matchup adjustment built on it — the same class
        of error as `spread_line`'s sign.
        """
        easy = sos.sos_multiplier([10.0])[0]
        hard = sos.sos_multiplier([0.0])[0]
        neutral = sos.sos_multiplier([sos.SCALE_MID])[0]
        assert easy > neutral > hard
        assert neutral == pytest.approx(1.0)

    def test_the_multiplier_stays_gentle(self):
        """Not fitted. A schedule term big enough to reorder a board has to earn it."""
        for rating in (0.0, 5.0, 10.0, 50.0, -20.0):
            assert 0.85 <= sos.sos_multiplier([rating])[0] <= 1.15

    def test_a_missing_rating_is_neutral_not_zero(self):
        """Zero is a real rating here — the hardest matchup — so NaN must not become it."""
        assert sos.sos_multiplier([np.nan])[0] == pytest.approx(1.0)

    def test_pff_team_codes_fold_through_the_package_canonicaliser(self):
        """One canonicaliser per package.

        This module used to carry its own alias map that folded to nflverse spellings
        (SF, GB) while the package canon is PFR-style (SFO, GNB). A board joined through
        both matched nothing for seven clubs and looked like missing data, not an error.
        """
        from nfl.salaries import canon_team
        assert sos.canon_team("HST") == canon_team("HST") == "HOU"
        assert sos.canon_team("ARZ") == "ARI"
        assert sos.canon_team("SF") == "SFO"

    def test_dk_roster_slots_fold_to_positions(self):
        """DK writes "WR/FLEX", the SOS files write "WR". Nothing joins otherwise."""
        board = pd.DataFrame({"Position": ["WR/FLEX", "RB/FLEX", "QB"],
                              "TeamAbbrev": ["SF", "SF", "SF"]})
        out = sos.attach_sos(board, week=1)
        assert out["sos_rating"].notna().all()

    def test_an_empty_board_survives(self):
        assert sos.attach_sos(pd.DataFrame()).empty
