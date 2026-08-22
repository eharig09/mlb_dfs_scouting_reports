"""The hitter composite: splits enter by how much they are worth, not by clearing a gate.

`_best_hitter_signal` used to report the *maximum* OPS across five samples of wildly
different size, gated at 50 / 20 / 8 / 8 / 6 at-bats. The maximum of several noisy estimates
is systematically the luckiest small one, so the headline read on every hitter was inflated.
Measured over 3,539 rows from 203 cached reports: it reported a .921 OPS for hitters whose
season OPS was .730, and when it crowned a batter-vs-pitcher line -- median nine at-bats --
it reported 1.176. After shrinking each split toward the hitter's own season line the same
rows report .769 against a .731 season, and the headline names a 129-at-bat platoon split
far more often than an 8-at-bat one.

`Composite` had the additive form of the same bug: it added the raw split once it cleared
its gate, which gave an 8-at-bat arsenal sample a heavier weight (55) than a 20-at-bat
platoon split (45).
"""

import numpy as np
import pandas as pd
import pytest

from scouting_report import (HITTER_SPLIT_SHRINK_AB, HITTER_SPLIT_WEIGHTS,
                             _best_hitter_signal, _shrunk_split_ops,
                             build_hitter_composite_table)


def hitter(season_ops=0.750, season_ab=400, **splits):
    row = {"Season OPS": season_ops, "Season AB": season_ab, "Season HR": 15}
    for key, value in splits.items():
        row[key.replace("_", " ")] = value
    for column in ("Platoon AB", "Platoon OPS", "Platoon HR", "Arsenal AB", "Arsenal OPS",
                   "Arsenal HR", "Similar AB", "Similar OPS", "Similar HR",
                   "BvP AB", "BvP OPS", "BvP HR"):
        row.setdefault(column, 0 if column.endswith(("AB", "HR")) else None)
    return pd.Series(row)


class TestShrinkage:
    def test_a_tiny_sample_barely_moves_the_estimate(self):
        shrunk, deviation = _shrunk_split_ops(1.500, 8, 0.750)
        assert deviation == pytest.approx(8 / 308 * 0.750, abs=0.002)
        assert shrunk < 0.780                      # nowhere near the 1.500 observed

    def test_a_large_sample_moves_it_a_lot(self):
        _, small = _shrunk_split_ops(1.000, 10, 0.700)
        _, large = _shrunk_split_ops(1.000, 400, 0.700)
        assert large > small * 5

    def test_at_the_half_reliable_point_it_moves_halfway(self):
        _, deviation = _shrunk_split_ops(0.900, HITTER_SPLIT_SHRINK_AB, 0.700)
        assert deviation == pytest.approx(0.100, abs=0.001)

    def test_a_split_equal_to_the_season_line_deviates_by_nothing(self):
        _, deviation = _shrunk_split_ops(0.750, 200, 0.750)
        assert deviation == pytest.approx(0.0)

    def test_no_sample_is_none_not_zero(self):
        assert _shrunk_split_ops(1.200, 0, 0.750) == (None, None)
        assert _shrunk_split_ops(None, 50, 0.750) == (None, None)


class TestBestSignal:
    def test_a_lucky_tiny_split_no_longer_wins(self):
        """The reported failure: 9 at-bats at a 1.400 OPS was the headline."""
        row = hitter(season_ops=0.750, BvP_AB=9, BvP_OPS=1.400,
                     Platoon_AB=150, Platoon_OPS=0.850)
        assert _best_hitter_signal(row).startswith("Platoon")

    def test_it_reports_what_survives_regression_not_just_the_raw_line(self):
        row = hitter(season_ops=0.700, Arsenal_AB=50, Arsenal_OPS=1.500)
        signal = _best_hitter_signal(row)
        assert "1.500 OPS / 50 AB" in signal          # the raw line is still visible
        assert "vs season after regression" in signal  # and so is what it is worth

    def test_an_unscored_split_is_never_the_headline(self):
        """BvP is weighted zero, so it must not be named a hitter's best signal."""
        row = hitter(season_ops=0.700, BvP_AB=40, BvP_OPS=1.600)
        assert not _best_hitter_signal(row).startswith("BvP")

    def test_it_falls_back_to_the_season_line_when_nothing_stands_out(self):
        row = hitter(season_ops=0.750, Platoon_AB=200, Platoon_OPS=0.752)
        assert _best_hitter_signal(row).startswith("Season")

    def test_it_surfaces_a_bad_matchup_too(self):
        """Deviation is scored on magnitude, so a cold split is as findable as a hot one."""
        row = hitter(season_ops=0.850, Platoon_AB=250, Platoon_OPS=0.550)
        signal = _best_hitter_signal(row)
        assert signal.startswith("Platoon") and "-0." in signal

    def test_a_hitter_with_no_season_line_is_reported_as_thin(self):
        row = hitter(season_ops=None, season_ab=4)
        assert "Thin samples" in _best_hitter_signal(row)


class TestComposite:
    def lineup(self, **overrides):
        base = {"Name": ["A Hitter"], "Bats": ["R"], "AB": [400], "OPS": [0.750], "HR": [15]}
        base.update(overrides)
        return pd.DataFrame(base)

    def test_a_tiny_hot_split_cannot_move_a_hitter_two_tiers(self):
        arsenal = pd.DataFrame({"Name": ["A Hitter"], "AB": [8], "OPS": [1.600], "HR": [2]})
        with_split = build_hitter_composite_table("TB", self.lineup(), arsenal_df=arsenal)
        without = build_hitter_composite_table("TB", self.lineup())
        moved = float(with_split["Composite"].iloc[0]) - float(without["Composite"].iloc[0])
        assert moved < 10, f"an 8-AB sample moved Composite by {moved:.1f}"

    def test_a_real_platoon_split_still_counts(self):
        splits = pd.DataFrame({"Name": ["A Hitter"], "AB vs R": [250], "OPS vs R": [0.950],
                               "HR vs R": [12]})
        with_split = build_hitter_composite_table("TB", self.lineup(), splits_df=splits)
        without = build_hitter_composite_table("TB", self.lineup())
        moved = float(with_split["Composite"].iloc[0]) - float(without["Composite"].iloc[0])
        assert moved > 5, f"a 250-AB platoon split only moved Composite by {moved:.1f}"

    def test_bvp_does_not_move_the_score_at_all(self):
        bvp = pd.DataFrame({"Name": ["A Hitter"], "AB": [30], "OPS": [1.500], "HR": [4]})
        with_bvp = build_hitter_composite_table("TB", self.lineup(), bvp_df=bvp)
        without = build_hitter_composite_table("TB", self.lineup())
        assert float(with_bvp["Composite"].iloc[0]) == pytest.approx(
            float(without["Composite"].iloc[0]))

    def test_bvp_is_still_displayed(self):
        """Zero weight is a claim about signal, not a reason to hide the history."""
        bvp = pd.DataFrame({"Name": ["A Hitter"], "AB": [30], "OPS": [1.500], "HR": [4]})
        table = build_hitter_composite_table("TB", self.lineup(), bvp_df=bvp)
        assert "30 AB" in str(table["BvP"].iloc[0])

    def test_weights_rank_the_constructs_by_what_they_measured(self):
        assert HITTER_SPLIT_WEIGHTS["Platoon"] > HITTER_SPLIT_WEIGHTS["Arsenal"]
        assert HITTER_SPLIT_WEIGHTS["Arsenal"] > HITTER_SPLIT_WEIGHTS["Similar"]
        assert HITTER_SPLIT_WEIGHTS["BvP"] == 0.0
