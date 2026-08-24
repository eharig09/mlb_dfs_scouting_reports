"""The strikeout panels on the Pitching page.

Two of these guard silent-wrong-answer bugs rather than crashes: a K% stored as a fraction
plotted against one stored as a percentage draws flat against the axis without erroring, and
innings read in the wrong notation produce a K% that is wrong by a third on every fractional
start while looking entirely plausible.
"""

import pandas as pd
import pytest

from dashboards import charts, strikeouts as K


def _log(rows):
    return {"K%": 0.214, "Last Starts": rows}


START_ROWS = [
    {"Date": "2026-08-15", "Opponent": "vs TEX", "IP": 6.0, "SO": 7, "H": 5, "BB": 2},
    {"Date": "2026-08-09", "Opponent": "@ BOS", "IP": 6.333333, "SO": 2, "H": 4, "BB": 0},
    {"Date": "2026-08-04", "Opponent": "@ CIN", "IP": 5.666667, "SO": 5, "H": 6, "BB": 1},
]


class TestScaleNormalisation:
    @pytest.mark.parametrize("value,expected", [
        (0.189, 18.9),      # the starter-info dict stores a fraction
        (18.9, 18.9),       # every frame stores percentage points
        (0.0, 0.0),
        (1.0, 100.0),       # a 1.0 can only be a fraction; nobody strikes out 1% of the time
    ])
    def test_a_rate_lands_in_percentage_points_either_way(self, value, expected):
        assert K.as_pct(value) == pytest.approx(expected)

    def test_a_missing_rate_stays_missing(self):
        assert pd.isna(K.as_pct(None))
        assert pd.isna(K.as_pct("—"))


class TestBattersFaced:
    def test_innings_are_decimal_thirds_not_baseball_notation(self):
        """`Last Starts` stores 5.667 for five and two thirds, already converted upstream.

        Reading it as baseball notation instead turns 5.667 into 5*3 + round(6.67) = 22 outs
        against a true 17 -- a K% wrong by a third, on a table that still looks sensible.
        """
        assert K._tbf(5.666667, 0, 0) == 17
        assert K._tbf(6.333333, 0, 0) == 19
        assert K._tbf(6.0, 0, 0) == 18

    def test_hits_and_walks_are_added(self):
        assert K._tbf(6.0, 5, 2) == 25

    def test_a_scoreless_or_absent_start_is_not_a_zero(self):
        assert pd.isna(K._tbf(0, 3, 1))
        assert pd.isna(K._tbf(None, 3, 1))

    def test_missing_hit_and_walk_columns_do_not_poison_the_total(self):
        assert K._tbf(6.0, None, None) == 18


class TestRecentStarts:
    def test_it_derives_a_rate_per_start(self):
        out = K.recent_starts(_log(START_ROWS))
        assert list(out.columns) >= ["Date", "IP", "SO", "TBF", "K%", "K/9"]
        first = out.iloc[0]
        assert first["TBF"] == 25          # 18 outs + 5 H + 2 BB
        assert first["K%"] == pytest.approx(28.0, abs=0.1)
        assert first["K/9"] == pytest.approx(10.5, abs=0.01)

    def test_the_season_rate_rides_along_in_percentage_points(self):
        out = K.recent_starts(_log(START_ROWS))
        assert out["Season K%"].iloc[0] == pytest.approx(21.4)

    def test_no_game_log_is_an_empty_frame_not_an_error(self):
        assert K.recent_starts({}).empty
        assert K.recent_starts({"Last Starts": []}).empty

    def test_a_log_without_strikeouts_is_declined(self):
        assert K.recent_starts({"Last Starts": [{"Date": "x", "IP": 6.0}]}).empty


class TestHeadline:
    def test_the_edge_is_his_rate_minus_the_cards(self):
        lineup = pd.DataFrame({"Split": ["Overall", "vs LHP"], "K%": [20.4, 17.9]})
        out = K.headline({"K%": 0.214}, lineup, pd.DataFrame())
        assert out["his_k"] == pytest.approx(21.4)
        assert out["lineup_k"] == pytest.approx(20.4)
        assert out["edge"] == pytest.approx(1.0, abs=0.05)

    def test_it_reads_the_overall_row_not_whichever_came_first(self):
        lineup = pd.DataFrame({"Split": ["vs LHP", "Overall"], "K%": [17.9, 20.4]})
        assert K.headline({"K%": 21.4}, lineup, pd.DataFrame())["lineup_k"] == pytest.approx(20.4)

    def test_the_platoon_spread_is_the_gap_between_his_two_sides(self):
        splits = pd.DataFrame({"Batter Side": ["L", "R"], "K%": [19.5, 24.6]})
        assert K.headline({"K%": 21.4}, pd.DataFrame(), splits)["split_spread"] == pytest.approx(5.1)

    def test_a_missing_side_leaves_the_edge_missing_rather_than_zero(self):
        out = K.headline({}, pd.DataFrame(), pd.DataFrame())
        assert pd.isna(out["edge"]) and pd.isna(out["his_k"])


class TestCharts:
    def _recent(self):
        return K.recent_starts(_log(START_ROWS))

    def test_the_start_line_builds_with_all_three_rules(self):
        spec = charts.k_recent_starts(self._recent(), season_k=21.4, lineup_k=20.4).to_dict()
        marks = [l.get("mark") if isinstance(l.get("mark"), str)
                 else (l.get("mark") or {}).get("type") for l in spec["layer"]]
        assert marks[0] == "line"
        assert marks.count("rule") == 3

    def test_a_missing_reference_is_omitted_not_drawn_at_zero(self):
        spec = charts.k_recent_starts(self._recent(), season_k=None, lineup_k=float("nan")).to_dict()
        marks = [l.get("mark") if isinstance(l.get("mark"), str)
                 else (l.get("mark") or {}).get("type") for l in spec["layer"]]
        assert marks.count("rule") == 1, "only the league line should survive"

    def test_the_platoon_bars_label_the_bats_actually_faced(self):
        frame = pd.DataFrame({"Batter Side": ["L", "R"], "K%": [19.5, 24.6],
                              "PA": [354, 240], "Bats faced": [3, 6]})
        spec = charts.k_platoon_bars(frame).to_dict()
        text = [l for l in spec["layer"]
                if (l.get("mark") or {}).get("type") == "text"]
        assert text and text[0]["encoding"]["text"]["field"] == "Bats faced"

    def test_neither_bar_panel_double_encodes_its_own_length(self):
        """Bar length already is the value; a colour ramp on the same field says it twice."""
        frame = pd.DataFrame({"Batter Side": ["L", "R"], "K%": [19.5, 24.6],
                              "PA": [354, 240], "Bats faced": [3, 6]})
        platoon = charts.k_platoon_bars(frame).to_dict()
        bars = platoon["layer"][0]["encoding"]
        assert "color" not in bars

        lineup = pd.DataFrame({"Name": ["A", "B"], "K%": [30.0, 12.0]})
        spec = charts.k_lineup_bars(lineup).to_dict()
        assert "color" not in spec["layer"][0]["encoding"]

    def test_the_lineup_bars_rank_rather_than_keep_slot_order(self):
        frame = pd.DataFrame({"Name": ["Low", "High"], "K%": [12.0, 30.0]})
        spec = charts.k_lineup_bars(frame).to_dict()
        assert spec["layer"][0]["encoding"]["y"]["sort"] == ["High", "Low"]

    def test_an_empty_frame_declines_rather_than_drawing_an_axis(self):
        for builder in (charts.k_recent_starts, charts.k_platoon_bars, charts.k_lineup_bars):
            assert builder(pd.DataFrame()) is None

    def test_chart_chrome_is_still_left_to_the_theme(self):
        spec = charts.k_recent_starts(self._recent()).to_dict()
        config = spec.get("config", {})
        assert "axis" not in config and "legend" not in config
