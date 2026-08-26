"""Deployment profiles: the blend, the rate guards, and the units that carry the signal."""

import numpy as np
import pandas as pd
import pytest

from nfl import usage


class TestRateGuard:
    def test_a_zero_denominator_is_blank_not_infinite(self):
        """0/0 means "we cannot say", which the report layer prints as an empty cell.

        An inf sorts to the top of every leaderboard, which is the opposite of unknown.
        """
        out = usage._rate(pd.Series([3.0, 1.0]), pd.Series([0.0, 2.0]))
        assert np.isnan(out.iloc[0])
        assert out.iloc[1] == 0.5

    def test_a_missing_column_yields_a_real_series(self):
        """`frame.get(col)` returns a scalar NaN -- no index, no `.fillna` -- and crashes.

        The recurring failure in this codebase, which is why nothing here uses `.get`.
        """
        out = usage._num(pd.DataFrame({"a": [1, 2]}), "absent")
        assert isinstance(out, pd.Series)
        assert len(out) == 2 and out.isna().all()


class TestBlend:
    def _frame(self, index, value, weight_col="routes_pg"):
        return pd.DataFrame({weight_col: value, "player": ["P"] * len(index),
                             "team": ["BUF"] * len(index), "pos": ["WR"] * len(index)},
                            index=pd.Index(index, name="gsis_id"))

    def test_a_single_season_passes_through_untouched(self):
        frame = self._frame(["a"], [10.0])
        assert usage._blend([frame], [2025]) is frame

    def test_seasons_are_weighted_by_recency(self):
        newest = self._frame(["a"], [10.0])
        older = self._frame(["a"], [0.0])
        out = usage._blend([newest, older], [2025, 2024], weights=(1.0, 0.5))
        # 10*1.0 + 0*0.5, over 1.5 of weight
        assert out.loc["a", "routes_pg"] == pytest.approx(10 / 1.5)

    def test_a_player_with_one_season_is_not_shrunk_toward_zero(self):
        """The weights renormalise over the seasons that actually have him.

        Without that, a rookie carries two seasons of absence as two seasons of zero and
        comes out a third of his real usage -- which reads as a role he does not have.
        """
        newest = self._frame(["rookie", "vet"], [20.0, 20.0])
        older = self._frame(["vet"], [20.0])
        out = usage._blend([newest, older], [2025, 2024], weights=(1.0, 0.5))
        assert out.loc["rookie", "routes_pg"] == pytest.approx(20.0)
        assert out.loc["vet", "routes_pg"] == pytest.approx(20.0)

    def test_identity_comes_from_the_most_recent_season_that_has_him(self):
        """A player who moved clubs is on his new team, not the one he used to be on."""
        newest = pd.DataFrame({"routes_pg": [1.0], "player": ["P"], "team": ["NYJ"],
                               "pos": ["WR"]}, index=pd.Index(["a"], name="gsis_id"))
        older = pd.DataFrame({"routes_pg": [1.0], "player": ["P"], "team": ["BUF"],
                              "pos": ["WR"]}, index=pd.Index(["a"], name="gsis_id"))
        out = usage._blend([newest, older], [2025, 2024])
        assert out.loc["a", "team"] == "NYJ"


class TestSeasonArgument:
    def test_a_bare_int_is_accepted(self):
        assert usage._as_seasons(2025) == [2025]

    def test_an_iterable_is_normalised(self):
        assert usage._as_seasons((2025, 2024)) == [2025, 2024]
