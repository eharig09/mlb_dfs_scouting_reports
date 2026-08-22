"""The split panels' second view, the outlier labels, and the table tints.

Three things here are the kind that fail silently in a browser rather than raising: a
selection param that lands on the wrong layer (the drill-down opens on the wrong player), a
tint that reads on one theme and not the other, and a label shortener that returns the wrong
word entirely.
"""

import pandas as pd
import pytest

from dashboards import charts, salaries, tables


def _alpha(css):
    """The alpha out of a `background-color: rgba(r, g, b, a)` declaration."""
    return float(css.rstrip(")").split(",")[-1])


def _hitters(n=8):
    """A synthetic matchup frame: two clubs, a spread of splits, one clear outlier."""
    return pd.DataFrame({
        "Name": ["Aaron Judge", "Elly De La Cruz", "Ronald Acuna Jr.", "Juan Soto",
                 "Bobby Witt Jr.", "Corey Seager", "Kyle Tucker", "Matt Olson"][:n],
        "Team": (["NYY", "CIN"] * 4)[:n],
        "Bats": (["R", "S", "R", "L"] * 2)[:n],
        "Composite": [45.0, 30.0, 12.0, -5.0, 22.0, 8.0, 33.0, 17.0][:n],
        "Signal": (["Priority", "Watch", "Neutral", "Fade"] * 2)[:n],
        "Season OPS": [0.900, 0.860, 0.840, 0.910, 0.800, 0.780, 0.850, 0.820][:n],
        "Platoon OPS": [0.930, 0.600, 0.850, 0.905, 0.810, 0.770, 0.860, 0.830][:n],
        "Platoon AB": [120, 90, 150, 200, 60, 110, 130, 95][:n],
        "Arsenal OPS": [0.880, 0.700, 0.860, 0.900, 0.820, 0.760, 0.870, 0.810][:n],
        "Arsenal AB": [40, 30, 55, 60, 25, 35, 45, 38][:n],
        "Off L28": [130.0, 90.0, 105.0, 140.0, 95.0, 88.0, 120.0, 100.0][:n],
        "Season AB": [400, 380, 420, 450, 300, 390, 410, 405][:n],
    })


def _layer_marks(spec):
    return [layer.get("mark") if isinstance(layer.get("mark"), str)
            else (layer.get("mark") or {}).get("type")
            for layer in spec.get("layer", [])]


class TestShortName:
    @pytest.mark.parametrize("full,expected", [
        ("Aaron Judge", "Judge"),
        # A naive last-token split returns "Cruz", which is not what anyone calls him.
        ("Elly De La Cruz", "De La Cruz"),
        # And here it returns "Jr." — not a shortened name, a different word.
        ("Ronald Acuna Jr.", "Acuna"),
        ("Vladimir Guerrero Jr.", "Guerrero"),
        ("Ke'Bryan Hayes", "Hayes"),
        ("", ""),
    ])
    def test_the_surname_survives_particles_and_suffixes(self, full, expected):
        assert charts._short_name(full) == expected


class TestSplitViews:
    def test_the_context_view_plots_the_raw_split_against_the_composite(self):
        """The y axis is the number a decision is made on, not its distance from a baseline.
        A deviation answers "is this unusual for him", which is a later question."""
        spec = charts.platoon_scatter(_hitters(), view="context").to_dict()
        points = spec["layer"][1]
        assert points["encoding"]["x"]["field"] == "Composite"
        assert points["encoding"]["y"]["field"] == "Platoon OPS"

    def test_the_plotted_value_is_the_split_itself(self):
        frame = _hitters()
        spec = charts.platoon_scatter(frame, view="context").to_dict()
        rows = spec["datasets"][spec["layer"][1]["data"]["name"]]
        by_name = {r["Name"]: r["Platoon OPS"] for r in rows}
        assert by_name["Elly De La Cruz"] == pytest.approx(0.600)
        assert "Split delta" not in rows[0]

    def test_the_reference_rule_shares_the_points_axis(self):
        """A reference line off its own scale is decoration, not a rule."""
        spec = charts.platoon_scatter(_hitters(), view="context").to_dict()
        rule_y = spec["layer"][0]["encoding"]["y"]["scale"]["domain"]
        point_y = spec["layer"][1]["encoding"]["y"]["scale"]["domain"]
        assert rule_y == point_y

    def test_the_rule_sits_at_the_median_of_the_bats_drawn(self):
        """A raw axis needs an anchor or every dot is an absolute with nothing to sit
        against. It moves with the panel rather than pretending to be a league line."""
        frame = _hitters()
        spec = charts.platoon_scatter(frame, view="context").to_dict()
        rule_rows = spec["datasets"][spec["layer"][0]["data"]["name"]]
        assert rule_rows[0]["y"] == pytest.approx(frame["Platoon OPS"].median())

    def test_the_arsenal_panel_takes_the_same_view(self):
        spec = charts.arsenal_scatter(_hitters(), view="context").to_dict()
        assert spec["layer"][1]["encoding"]["x"]["field"] == "Composite"

    def test_no_composite_falls_back_to_parity_rather_than_drawing_nothing(self):
        frame = _hitters().drop(columns=["Composite"])
        spec = charts.platoon_scatter(frame, view="context").to_dict()
        assert spec["layer"][1]["encoding"]["x"]["field"] == "Season OPS"

    def test_both_views_are_offered(self):
        assert set(charts.SPLIT_VIEWS.values()) == {"context", "parity"}


class TestPriceShapes:
    def test_price_tier_is_the_shape_channel(self):
        frame = _hitters().assign(Salary=[2500, 3500, 4500, 5500, 2900, 6000, 4000, 3100])
        spec = charts.platoon_scatter(frame, view="context").to_dict()
        shape = spec["layer"][1]["encoding"]["shape"]
        assert shape["field"] == "Price"
        assert shape["scale"]["domain"] == salaries.TIER_LABELS
        assert shape["scale"]["range"] == [salaries.TIER_SHAPES[t]
                                           for t in salaries.TIER_LABELS]

    def test_shape_needs_a_point_mark_not_a_circle(self):
        """`mark_circle` silently has no shape channel."""
        frame = _hitters().assign(Salary=3000)
        spec = charts.platoon_scatter(frame, view="context").to_dict()
        assert spec["layer"][1]["mark"]["type"] == "point"

    def test_an_unpriced_hitter_gets_his_own_category(self):
        """A hitter the market never priced is a different thing from a cheap one — the same
        rule that keeps him out of the surplus calculation."""
        frame = _hitters().assign(Salary=[2500, None, 4500, 5500, None, 6000, 4000, 3100])
        tiered = salaries.price_tier(frame)
        assert list(tiered["Price"])[1] == salaries.UNPRICED
        spec = charts.platoon_scatter(frame, view="context").to_dict()
        assert salaries.UNPRICED in spec["layer"][1]["encoding"]["shape"]["scale"]["domain"]

    def test_a_frame_with_no_salary_at_all_still_draws(self):
        spec = charts.platoon_scatter(_hitters(), view="context").to_dict()
        assert spec["layer"][1]["encoding"]["shape"]["scale"]["domain"] == [salaries.UNPRICED]

    def test_the_size_floor_lifts_so_shapes_stay_apart(self):
        """A triangle and a diamond stop being distinguishable long before a circle stops
        being visible, so the shaped view cannot use the unshaped size floor."""
        spec = charts.platoon_scatter(_hitters(), view="context").to_dict()
        assert spec["layer"][1]["encoding"]["size"]["scale"]["range"][0] >= 90


class TestOutlierLabels:
    def test_only_the_extremes_are_named(self):
        spec = charts.platoon_scatter(_hitters(), view="context").to_dict()
        text = [l for l in spec["layer"] if _layer_marks(spec)[spec["layer"].index(l)] == "text"]
        assert len(text) == 1
        rows = spec["datasets"][text[0]["data"]["name"]]
        assert len(rows) == charts.LABEL_LIMIT < len(_hitters())
        assert "De La Cruz" in [r["_label"] for r in rows]

    def test_a_label_is_never_the_only_thing_carrying_identity(self):
        """Unlabelled points still resolve — the tooltip names every one of them."""
        spec = charts.platoon_scatter(_hitters(), view="context").to_dict()
        fields = [t["field"] for t in spec["layer"][1]["encoding"]["tooltip"]]
        assert "Name" in fields and "Team" in fields

    def test_labels_take_the_marks_colour_scale_not_their_own_ink(self):
        """Pinned label ink is the same bug that made legends vanish in dark mode."""
        spec = charts.platoon_scatter(_hitters(), view="context").to_dict()
        text = spec["layer"][2]
        assert text["encoding"]["color"]["scale"]["range"] == charts.TEAM_SLOTS[:2]
        assert "color" not in (text.get("mark") or {})

    def test_a_frame_smaller_than_the_limit_labels_what_it_has(self):
        chart = charts.platoon_scatter(_hitters(n=2), view="context")
        assert chart is not None
        assert chart.to_dict() is not None


class TestSelectionLayer:
    def test_the_click_target_is_the_circles_not_the_labels(self):
        """The param used to go on `layers[-1]` — "the points are the top layer". Adding
        direct labels put text there, and every click would have resolved against the four
        labelled names instead of the eighteen points."""
        chart = charts.selectable(charts.platoon_scatter(_hitters(), view="context"))
        spec = chart.to_dict()
        marks = _layer_marks(spec)
        named = [i for i, l in enumerate(spec["layer"]) if "name" in l]
        assert len(named) == 1
        assert marks[named[0]] in ("point", "circle")
        assert marks[-1] == "text", "labels should still draw on top of the points"

    def test_a_none_chart_stays_none(self):
        assert charts.selectable(None) is None


class TestTableTints:
    def test_a_tint_is_translucent_so_theme_ink_survives_it(self):
        """An opaque pastel is legible on exactly one of the two surfaces Streamlit ships."""
        for value in (tables.GOOD, tables.WARN, tables.BAD, tables.MUTED):
            assert value.startswith("background-color: rgba(")
            alpha = float(value.rstrip(")").split(",")[-1])
            assert 0 < alpha < 0.35

    def test_a_tint_is_a_full_css_declaration(self):
        """A bare colour raises "Styles supplied as string must follow CSS rule formats"."""
        frame = pd.DataFrame({"Edge": ["hitter", "pitcher"]})
        styled = tables.style(frame, [(tables.by_value, {"column": "Edge",
                                                         "tints": tables.EDGE_TINTS})])
        assert styled.to_html().count("background-color") == 2

    def test_an_unknown_category_is_left_alone(self):
        frame = pd.DataFrame({"Edge": ["hitter", "something new"]})
        styles = tables.by_value(frame, "Edge", tables.EDGE_TINTS)
        assert styles.loc[0, "Edge"] == tables.BAD
        assert styles.loc[1, "Edge"] == ""

    def test_signed_respects_the_threshold_it_is_given(self):
        """Inside the band the pipeline declines to rule, so neither should the tint."""
        frame = pd.DataFrame({"RV/100": [0.11, 0.9, -0.9, None]})
        styles = tables.signed(frame, "RV/100", good_when_negative=True, threshold=0.7)
        out = list(styles["RV/100"])
        assert out[0] == "" and out[3] == ""
        assert tables.BAD_RGB in out[1] and tables.GOOD_RGB in out[2]

    def test_signed_deepens_with_distance_past_the_threshold(self):
        frame = pd.DataFrame({"RV/100": [0.8, 2.4]})
        out = list(tables.signed(frame, "RV/100", threshold=0.7)["RV/100"])
        assert _alpha(out[0]) < _alpha(out[1])

    def test_ranked_is_a_ramp_not_two_flags(self):
        frame = pd.DataFrame({"ERA": [3.94, 3.26, 5.10, 4.00, 4.55]})
        out = list(tables.ranked(frame, "ERA", best="low")["ERA"])
        assert sum(1 for v in out if v) >= 4, "a ramp shades the middle, not just the ends"
        # Best gets the deepest good step, worst the deepest bad step.
        assert _alpha(out[1]) == pytest.approx(tables.RAMP_MAX)
        assert _alpha(out[2]) == pytest.approx(tables.RAMP_MAX)
        assert tables.GOOD_RGB in out[1] and tables.BAD_RGB in out[2]
        # ... and the ramp is monotone on each side of the midpoint.
        assert _alpha(out[0]) > _alpha(out[3])

    def test_ranked_reverses_when_high_is_good(self):
        frame = pd.DataFrame({"K%": [18.0, 31.0]})
        out = list(tables.ranked(frame, "K%", best="high")["K%"])
        assert tables.BAD_RGB in out[0] and tables.GOOD_RGB in out[1]

    def test_the_midpoint_of_a_ramp_is_the_bare_surface(self):
        """A diverging scale takes a neutral midpoint, not a third hue."""
        frame = pd.DataFrame({"ERA": [1.0, 2.0, 3.0]})
        middle = list(tables.ranked(frame, "ERA", best="low")["ERA"])[1]
        assert middle == ""

    def test_every_ramp_step_stays_inside_the_readable_alpha_band(self):
        frame = pd.DataFrame({"ERA": [1.0, 2.0, 3.0, 4.0, 9.0]})
        for css in tables.ranked(frame, "ERA")["ERA"]:
            if css:
                assert tables.RAMP_MIN <= _alpha(css) <= tables.RAMP_MAX

    def test_ranked_declines_on_a_single_distinct_value(self):
        frame = pd.DataFrame({"ERA": [3.0, 3.0]})
        assert set(tables.ranked(frame, "ERA")["ERA"]) == {""}

    def test_a_missing_column_returns_the_frame_untouched(self):
        frame = pd.DataFrame({"a": [1, 2]})
        assert isinstance(tables.style(frame, [(tables.by_value,
                                                {"column": "Edge", "tints": {}})]),
                          pd.DataFrame)

    def test_an_empty_frame_is_not_styled(self):
        assert isinstance(tables.style(pd.DataFrame(), [(tables.ranked, {"column": "ERA"})]),
                          pd.DataFrame)

    def test_availability_tints_match_the_bullpen_tiers(self):
        from dashboards import bullpen

        assert set(tables.AVAIL_TINTS) == set(bullpen.TIERS)
