"""The lineup arsenal K edge: the one arsenal number the pitcher projection reads.

Three things have to hold or the feature is worse than not having it. It must be a
*residual* -- a hitter who strikes out a lot against everything is not a matchup edge. It
must shrink toward that hitter's own rate, because the median arsenal sample is about
eighteen plate appearances. And it must fail closed: a cached game built before the column
existed has to produce no adjustment rather than a wrong one.

See docs/arsenal_study.md for where the constants come from.
"""

import numpy as np
import pandas as pd
import pytest

from dfs.projections import (ARSENAL_IP_CLIP, ARSENAL_IP_PASSTHROUGH,
                             ARSENAL_K_PASSTHROUGH, lineup_arsenal_k_edge)
from scouting_report import ARSENAL_K_SHRINK_PA, _arsenal_k_edge


def lineup(names, spots=None):
    return pd.DataFrame({"Name": names, "Spot": spots or list(range(1, len(names) + 1))})


def arsenal(names, edges):
    return pd.DataFrame({"Name": names, "K Edge": edges})


NINE = [f"Hitter {i}" for i in range(1, 10)]


class TestArsenalKEdge:
    def test_is_a_residual_not_a_level(self):
        """A high-strikeout hitter who strikes out at his usual rate has no edge."""
        assert _arsenal_k_edge(35.0, 200, 35.0) == 0.0
        assert _arsenal_k_edge(12.0, 200, 12.0) == 0.0

    def test_sign_is_the_pitcher_s(self):
        assert _arsenal_k_edge(35.0, 200, 25.0) > 0    # whiffs more than usual here
        assert _arsenal_k_edge(15.0, 200, 25.0) < 0    # makes more contact than usual

    def test_shrinks_toward_the_hitter_s_own_rate(self):
        """Same raw gap, less history -> less edge."""
        big = _arsenal_k_edge(40.0, 400, 20.0)
        small = _arsenal_k_edge(40.0, 5, 20.0)
        assert abs(small) < abs(big)
        # At exactly the prior weight the edge is half the raw gap.
        assert _arsenal_k_edge(40.0, ARSENAL_K_SHRINK_PA, 20.0) == pytest.approx(10.0)

    def test_no_history_means_no_edge(self):
        assert _arsenal_k_edge(40.0, 0, 20.0) == pytest.approx(0.0)

    def test_missing_baseline_is_none_not_zero(self):
        """Unknown and 'no edge' are different, and only one may reach the projection."""
        assert _arsenal_k_edge(40.0, 200, None) is None


class TestLineupAggregate:
    def test_averages_the_lineup(self):
        edge, count = lineup_arsenal_k_edge(arsenal(NINE, [2.0] * 9), lineup(NINE))
        assert count == 9
        assert edge == pytest.approx(0.02)   # percentage points -> fraction

    def test_weights_the_top_of_the_order_harder(self):
        """The leadoff hitter gets more plate appearances, so he gets more say."""
        leadoff = arsenal(NINE, [9.0] + [0.0] * 8)
        ninth = arsenal(NINE, [0.0] * 8 + [9.0])
        top, _ = lineup_arsenal_k_edge(leadoff, lineup(NINE))
        bottom, _ = lineup_arsenal_k_edge(ninth, lineup(NINE))
        assert top > bottom

    def test_respects_the_batting_order_column(self):
        """Slots come from `Spot`, not from row order."""
        reversed_order = lineup(NINE, spots=list(range(9, 0, -1)))
        table = arsenal(NINE, [9.0] + [0.0] * 8)
        weighted, _ = lineup_arsenal_k_edge(table, reversed_order)
        natural, _ = lineup_arsenal_k_edge(table, lineup(NINE))
        assert weighted < natural   # the same hitter now bats ninth

    def test_ignores_hitters_without_an_edge(self):
        table = arsenal(NINE, [4.0] * 5 + [None] * 4)
        edge, count = lineup_arsenal_k_edge(table, lineup(NINE))
        assert count == 5
        assert edge == pytest.approx(0.04)

    def test_fails_closed_when_too_few_hitters_have_history(self):
        table = arsenal(NINE, [4.0] * 4 + [None] * 5)
        edge, count = lineup_arsenal_k_edge(table, lineup(NINE))
        assert edge is None and count == 4

    def test_fails_closed_on_a_table_from_before_the_column_existed(self):
        """Cached payloads predate `K Edge`; they must produce no adjustment at all."""
        legacy = pd.DataFrame({"Name": NINE, "PA": [40] * 9, "OPS": [0.750] * 9,
                               "K%": [22.0] * 9})
        assert lineup_arsenal_k_edge(legacy, lineup(NINE)) == (None, 0)

    def test_fails_closed_on_missing_input(self):
        assert lineup_arsenal_k_edge(None, lineup(NINE)) == (None, 0)
        assert lineup_arsenal_k_edge(pd.DataFrame(), lineup(NINE)) == (None, 0)

    def test_survives_a_missing_lineup(self):
        """Without a batting order every hitter falls back to a middle slot."""
        edge, count = lineup_arsenal_k_edge(arsenal(NINE, [3.0] * 9), None)
        assert count == 9 and edge == pytest.approx(0.03)


class TestConstants:
    def test_pass_through_is_a_fraction_of_the_measured_edge(self):
        """Passing through more than the measured edge would amplify noise, not use it."""
        assert 0.0 < ARSENAL_K_PASSTHROUGH < 1.0

    def test_innings_effect_is_clipped_to_something_small(self):
        """At the widest edge the study saw (~7 K points), innings must stay bounded."""
        extreme = ARSENAL_IP_PASSTHROUGH * 0.075
        assert min(extreme, ARSENAL_IP_CLIP) <= ARSENAL_IP_CLIP
        assert ARSENAL_IP_CLIP <= 0.5
