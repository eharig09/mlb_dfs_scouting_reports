"""The football exploration pages and the readers behind them.

The page tests execute the page script, which is the only way to catch a page-level error:
an HTTP health check on the running server answers 200 for a page that raises on every
render, because it never runs the script.
"""

import os

import altair as alt
import numpy as np
import pandas as pd
import pytest

from dashboards import nfl_charts, nfl_pff

PAGES = ("nfl_receivers", "nfl_targets", "nfl_ground", "nfl_coverage", "nfl_slate")


def _page(name):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "dashboards", "app_pages", f"{name}.py")


class TestAlignment:
    def test_a_receiver_is_named_for_where_he_mostly_lines_up(self):
        frame = pd.DataFrame([{"slot_rate": 70.0, "wide_rate": 25.0, "inline_rate": 5.0},
                              {"slot_rate": 10.0, "wide_rate": 85.0, "inline_rate": 5.0},
                              {"slot_rate": 20.0, "wide_rate": 10.0, "inline_rate": 70.0}])
        assert list(nfl_pff._alignment(frame)) == ["Slot", "Wide", "Inline"]

    def test_a_receiver_with_no_rates_is_blank_not_guessed(self):
        frame = pd.DataFrame([{"slot_rate": np.nan, "wide_rate": np.nan,
                               "inline_rate": np.nan}])
        assert list(nfl_pff._alignment(frame)) == [""]

    def test_a_frame_without_the_columns_does_not_crash(self):
        assert list(nfl_pff._alignment(pd.DataFrame([{"x": 1}]))) == [""]


class TestRatio:
    def test_a_zero_denominator_is_blank_not_infinite(self):
        """A blank is "cannot say". An inf sorts to the top of every leaderboard, which is
        the opposite of unknown."""
        out = nfl_pff._ratio(pd.Series([5.0, 3.0]), pd.Series([0.0, 2.0]))
        assert np.isnan(out.iloc[0])
        assert out.iloc[1] == 1.5


class TestRule:
    def test_a_rule_without_a_domain_is_valid(self):
        """`alt.Scale(domain=None)` is not "no domain" -- Altair validates it and refuses.
        That surfaced as a redacted "this app has encountered an error" on four pages at
        once, naming none of them.
        """
        chart = nfl_charts._rule(10.0, "y")
        assert chart.to_dict()          # raises SchemaValidationError if malformed

    def test_a_rule_with_a_domain_keeps_it(self):
        chart = nfl_charts._rule(10.0, "y", domain=[0, 20])
        assert chart.to_dict()["encoding"]["y"]["scale"]["domain"] == [0, 20]


class TestUsageScatter:
    def _frame(self):
        return pd.DataFrame([
            {"Name": "A", "Team": "BUF", "Alignment": "Wide", "Routes/G": 30.0,
             "TPRR": 0.25, "Targets": 100},
            {"Name": "B", "Team": "KAN", "Alignment": "Slot", "Routes/G": 20.0,
             "TPRR": 0.20, "Targets": 60},
        ])

    def test_it_builds(self):
        chart = nfl_charts.usage_scatter(self._frame(), "Routes/G", "TPRR", size="Targets")
        assert chart is not None and chart.to_dict()

    def test_an_axis_with_no_data_returns_none_rather_than_an_empty_chart(self):
        frame = self._frame()
        frame["TPRR"] = np.nan
        assert nfl_charts.usage_scatter(frame, "Routes/G", "TPRR") is None

    def test_it_does_not_pin_axis_colours(self):
        """Streamlit themes Vega charts to the live app theme and knows which is active;
        hard-coded ink does not, and pinning a light palette makes dark mode unreadable."""
        spec = nfl_charts.usage_scatter(self._frame(), "Routes/G", "TPRR").to_dict()
        assert "config" not in spec or "axis" not in spec.get("config", {})


class TestReaders:
    """These read the real PFF exports, so they are skipped when none are on disk."""

    def _seasons(self):
        seasons = nfl_pff.available_seasons()
        if not seasons:
            pytest.skip("no PFF receiving exports on disk")
        return (seasons[0],)

    def test_receivers_come_back_keyed_and_labelled(self):
        frame = nfl_pff.receivers(self._seasons(), min_routes=150)
        assert not frame.empty
        for column in ("Name", "Team", "Pos", "Alignment", "Routes/G", "TPRR", "aDOT"):
            assert column in frame.columns
        assert set(frame["Alignment"]) <= {"Slot", "Wide", "Inline", ""}

    def test_defense_scheme_covers_every_club(self):
        seasons = nfl_pff.available_seasons("defense_coverage_scheme")
        if not seasons:
            pytest.skip("no coverage exports on disk")
        scheme = nfl_pff.defense_scheme(seasons[0])
        assert len(scheme) == 32
        assert scheme["Man%"].between(0, 100).all()
        # Man and zone are the two halves of one thing.
        assert np.allclose(scheme["Man%"] + scheme["Zone%"], 100.0)

    def test_target_shares_sum_to_one_within_a_club(self):
        """Shares are of the club's *qualifying* receivers, so they close at 1.0 -- that is
        what makes it a comparison between the players who actually play."""
        frame = nfl_pff.target_distribution(self._seasons(), min_routes=100)
        totals = frame.groupby("Team")["Target share"].sum()
        assert np.allclose(totals.values, 1.0)


class TestPagesRender:
    @pytest.mark.parametrize("page", PAGES)
    def test_it_renders_without_raising(self, page):
        pytest.importorskip("streamlit.testing.v1")
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(_page(page), default_timeout=240).run()
        assert not app.exception, [str(e.value)[:200] for e in app.exception]
        assert app.header

    def test_the_coverage_page_warns_about_allowed_rates(self):
        """The page shows what defences gave up, and those columns do not predict next
        season (man +0.05, zone +0.01, slot +0.10). Showing them without the warning would
        invite exactly the matchup call the measurement rules out."""
        pytest.importorskip("streamlit.testing.v1")
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(_page("nfl_coverage"), default_timeout=240).run()
        assert not app.exception
        assert any("+0.05" in w.value or "+0.10" in w.value for w in app.warning)

    def test_no_page_reuses_a_widget_key(self):
        """Two widgets sharing a key raise `StreamlitDuplicateElementKey` at render, which
        the Targets page did when a shared sidebar filter collided with its club selector."""
        pytest.importorskip("streamlit.testing.v1")
        from streamlit.testing.v1 import AppTest
        for page in PAGES:
            app = AppTest.from_file(_page(page), default_timeout=240).run()
            assert not any("DuplicateElementKey" in str(e.value) for e in app.exception), page


class TestMarkStyle:
    def test_marks_are_drawn_without_a_stroke(self):
        """A light outline separates overlapping points, but at these sizes it eats most of
        a small mark and the colour — the channel carrying alignment or position — stops
        being readable. Opacity does the overlap job instead."""
        assert "stroke" not in nfl_charts.MARK
        assert nfl_charts.MARK["filled"] is True

    def test_colouring_by_team_does_not_duplicate_the_column(self):
        """`usage_scatter` names Team in its own column list, so `color="Team"` selected it
        twice and Vega refused the frame — `DuplicateError: got 'Team' 2 times`, surfaced as
        a redacted Streamlit error naming neither the chart nor the column."""
        frame = pd.DataFrame([
            {"Name": "A", "Team": "BUF", "x": 1.0, "y": 2.0},
            {"Name": "B", "Team": "KAN", "x": 2.0, "y": 3.0}])
        chart = nfl_charts.usage_scatter(frame, "x", "y", color="Team")
        assert chart is not None and chart.to_dict()


class TestRounding:
    def test_a_rate_under_one_keeps_three_places(self):
        """TPRR runs 0.15 to 0.30 and rounds to a single value for half the league at two
        decimals, so magnitude decides the precision rather than a column list."""
        out = nfl_charts.round_display(pd.DataFrame({"TPRR": [0.23456, 0.18765]}))
        assert list(out["TPRR"]) == [0.235, 0.188]

    def test_a_larger_number_keeps_two(self):
        out = nfl_charts.round_display(pd.DataFrame({"Proj": [17.28456, 9.1234]}))
        assert list(out["Proj"]) == [17.28, 9.12]

    def test_integers_and_text_are_untouched(self):
        frame = pd.DataFrame({"Salary": [6600, 5100], "Name": ["A", "B"]})
        out = nfl_charts.round_display(frame)
        assert list(out["Salary"]) == [6600, 5100]
        assert list(out["Name"]) == ["A", "B"]

    def test_an_empty_frame_passes_through(self):
        assert nfl_charts.round_display(pd.DataFrame()).empty


class TestQuarterbackDepth:
    def test_the_depth_buckets_are_pffs_four(self):
        from dashboards import nfl_pff as module
        assert module.DEPTH_BUCKETS == ("behind_los", "short", "medium", "deep")
        assert set(module.DEPTH_LABELS) == set(module.DEPTH_BUCKETS)

    def test_the_buckets_do_not_cover_every_attempt(self):
        """PFF charts a depth for 86-95% of throws; the rest are throwaways, spikes and
        batted balls with no meaningful air yards.

        Asserted rather than normalised away, because a quarterback whose mix accounts for
        only 86% of his attempts is telling you how often he throws the ball away — and
        because a test expecting 100% would have quietly forced a wrong denominator.
        """
        from dashboards import nfl_pff as module
        seasons = module.available_seasons("passing_depth")
        if not seasons:
            pytest.skip("no passing-depth exports on disk")
        frame = module.quarterbacks(seasons[0], min_attempts=200)
        assert not frame.empty
        shares = frame[[f"{module.DEPTH_LABELS[b]} att%" for b in module.DEPTH_BUCKETS]]
        total = shares.sum(axis=1)
        assert total.between(80, 100).all()
        assert (total < 99).any()
        # and the frame says so rather than leaving it to be discovered
        assert frame["Charted%"].between(80, 100).all()

    def test_the_long_form_is_one_row_per_quarterback_per_bucket(self):
        from dashboards import nfl_pff as module
        seasons = module.available_seasons("passing_depth")
        if not seasons:
            pytest.skip("no passing-depth exports on disk")
        frame = module.quarterbacks(seasons[0], min_attempts=200)
        long = module.quarterback_depth_long(frame)
        assert len(long) == len(frame) * len(module.DEPTH_BUCKETS)


class TestAllStackCombinations:
    def test_top_per_team_keeps_every_club(self):
        """The shortlist exists so the priciest offences do not crowd out the slate."""
        from dashboards import nfl_stacking as stacking
        combos = pd.DataFrame([
            {"Team": t, "Size": 3, "Proj": p, "Partners": f"{t}{p}"}
            for t, projections in (("BUF", [40, 39, 38, 37]), ("KAN", [20, 19]))
            for p in projections])
        top = stacking.top_per_team(combos, per_team=2)
        assert set(top["Team"]) == {"BUF", "KAN"}
        assert len(top) == 4
        assert sorted(top[top["Team"] == "BUF"]["Proj"], reverse=True) == [40, 39]


class TestBarHeight:
    def test_every_row_gets_enough_space_for_its_label(self):
        """Below about 18px a row Vega starts dropping axis labels to fit, which is exactly
        what makes a ranked chart useless — you see the shape but cannot read off who."""
        assert nfl_charts.ROW_HEIGHT >= 18
        assert nfl_charts.bar_height(30) >= 30 * nfl_charts.ROW_HEIGHT

    def test_it_stays_within_sane_bounds(self):
        assert nfl_charts.bar_height(1) >= 240
        assert nfl_charts.bar_height(500) <= 1400


class TestExtraTooltip:
    def test_named_columns_reach_the_tooltip(self):
        """The value number has to be readable on hover, not only in the table."""
        frame = pd.DataFrame([
            {"Name": "A", "Team": "BUF", "Salary$": 6600.0, "Proj": 17.0, "Value": 2.58},
            {"Name": "B", "Team": "KAN", "Salary$": 5100.0, "Proj": 12.0, "Value": 2.35}])
        spec = nfl_charts.usage_scatter(frame, "Salary$", "Proj", color="Team",
                                        extra_tooltip=("Value",)).to_dict()
        fields = {t["field"] for layer in spec["layer"]
                  for t in layer.get("encoding", {}).get("tooltip", [])}
        assert "Value" in fields

    def test_an_absent_extra_column_is_skipped_quietly(self):
        frame = pd.DataFrame([{"Name": "A", "Team": "BUF", "x": 1.0, "y": 2.0}])
        assert nfl_charts.usage_scatter(frame, "x", "y", color="Team",
                                        extra_tooltip=("Nope",)) is not None
