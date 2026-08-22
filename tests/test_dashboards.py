"""Dashboard data and chart layer.

Streamlit's own runtime is not exercised here — these are the pure pieces underneath it:
filename parsing, typing of a cached payload, and the Altair specs. Chart builders are
checked by validating the Vega spec they produce, which catches composition mistakes that
only surface when the browser tries to render them.

`st.cache_data` wraps its target, so tests call `.__wrapped__` to reach the real function
rather than depending on a cache that has no runtime behind it.
"""

import numpy as np
import pandas as pd
import pytest

from dashboards import charts, data


def _hitters(n=6, teams=("ATH", "KC")):
    rows = []
    for i in range(n):
        rows.append({
            "Team": teams[i % len(teams)],
            "Name": f"Player {i}",
            "Bats": "LR"[i % 2],
            "Season OPS": 0.700 + i * 0.02,
            "Season AB": 200 + i * 20,
            "Platoon OPS": 0.680 + i * 0.015,
            "Platoon AB": 80 + i * 10,
            "Arsenal OPS": 0.650 + i * 0.03,
            "Arsenal AB": 20 + i,
            "Composite": 30 - i * 6,
            "Signal": ["Priority", "Watch", "Neutral", "Fade"][i % 4],
            "Off Szn": 105 + i,
            "Off L28": 95 + i * 3,
            "game": "ATH @ KC",
        })
    return pd.DataFrame(rows)


class TestFilenames:
    def test_a_normal_game_parses(self):
        parsed = data._parse(".cache/report_data/2026-08-20_WSH_TEX.pkl")
        assert parsed["date"] == "2026-08-20"
        assert (parsed["away"], parsed["home"]) == ("WSH", "TEX")
        assert parsed["game"] == 1
        assert parsed["label"] == "WSH @ TEX"

    def test_a_doubleheader_keeps_its_game_number(self):
        """Both games of a doubleheader are cached; without the suffix one hides the other."""
        parsed = data._parse(".cache/report_data/2026-08-20_WSH_TEX_g2.pkl")
        assert parsed["game"] == 2
        assert parsed["label"] == "WSH @ TEX (g2)"

    def test_an_unrecognised_name_is_skipped_not_guessed(self):
        assert data._parse(".cache/report_data/notes.pkl") is None
        assert data._parse(".cache/report_data/2026-08-20_WSH.pkl") is None


class TestHitters:
    def _payload(self, frame):
        return {"advanced_context": {"hitter_composite": frame}}

    def _meta(self):
        return {"date": "2026-08-20", "away": "ATH", "home": "KC", "game": 1,
                "label": "ATH @ KC", "path": "x"}

    def test_numeric_columns_are_coerced(self):
        """Numeric, not specifically float — whole-number strings coerce to int, which is
        correct; the chart layer only needs them to stop being objects."""
        frame = _hitters()
        frame["Composite"] = frame["Composite"].astype(str)
        frame["Season OPS"] = frame["Season OPS"].astype(str)
        out = data.hitters(self._payload(frame), self._meta())
        assert out["Composite"].dtype.kind in "if"
        assert out["Season OPS"].dtype.kind in "if"

    def test_unparseable_values_become_null_rather_than_raising(self):
        """A blank or dash in a cached frame must coerce to null, not blow up a page.

        The column is built as object here on purpose: pandas 3.0 refuses to assign a
        string into an int64 column, so writing the test the obvious way fails in its own
        setup rather than in the code under test.
        """
        frame = _hitters()
        frame["Off L28"] = frame["Off L28"].astype(object)
        frame.loc[0, "Off L28"] = "—"
        out = data.hitters(self._payload(frame), self._meta())
        assert pd.isna(out.loc[0, "Off L28"])
        assert out["Off L28"].dtype.kind in "if"

    def test_the_opponent_is_the_other_club_in_the_game(self):
        out = data.hitters(self._payload(_hitters()), self._meta())
        pairs = set(zip(out["Team"], out["opponent"]))
        assert pairs == {("ATH", "KC"), ("KC", "ATH")}

    def test_a_payload_with_no_composite_is_an_empty_frame(self):
        assert data.hitters({"advanced_context": {}}, self._meta()).empty
        assert data.hitters({}, self._meta()).empty

    def test_a_missing_numeric_column_does_not_crash(self):
        """`frame.get` returns a bare NaN for an absent column — the recurring trap."""
        frame = _hitters().drop(columns=["Arsenal OPS", "Off L28"])
        out = data.hitters(self._payload(frame), self._meta())
        assert out["Arsenal OPS"].isna().all()
        assert out["Off L28"].isna().all()


class TestCharts:
    @pytest.mark.parametrize("builder", [
        charts.platoon_scatter, charts.form_scatter,
        charts.arsenal_scatter, charts.slate_scatter,
    ])
    def test_each_builds_a_valid_vega_spec(self, builder):
        chart = builder(_hitters())
        assert chart is not None
        spec = chart.to_dict()          # raises if the composition is invalid
        assert "encoding" in spec or "layer" in spec

    @pytest.mark.parametrize("builder", [
        charts.platoon_scatter, charts.form_scatter,
        charts.arsenal_scatter, charts.slate_scatter,
    ])
    def test_an_empty_frame_returns_none_rather_than_an_empty_chart(self, builder):
        assert builder(_hitters().iloc[0:0]) is None

    def test_a_layered_chart_is_configured_not_rewrapped(self):
        """`alt.Chart(layer_chart)` treats the chart as *data* and fails validation.

        The error is "`Data` has no parameter named 'layer'", which says nothing useful, so
        this pins the composition rather than trusting it to be noticed in a browser.
        """
        chart = charts.platoon_scatter(_hitters(), view="parity")
        spec = chart.to_dict()
        # rule, points, outlier labels
        assert "layer" in spec and len(spec["layer"]) == 3

    def test_chart_chrome_is_left_to_the_theme(self):
        """Axis and legend ink must not be pinned.

        Hard-coding a light-surface ink rendered every legend label invisible the moment the
        app was viewed in dark mode. Streamlit themes Vega charts and knows which theme is
        live; a constant in this module does not.
        """
        spec = charts.form_scatter(_hitters()).to_dict()
        config = spec.get("config", {})
        assert "axis" not in config
        assert "legend" not in config

    def test_data_colours_are_ours_and_stay_put(self):
        """Team identity and the status scale carry meaning, so they are pinned."""
        spec = charts.platoon_scatter(_hitters(), view="parity").to_dict()
        colour = spec["layer"][1]["encoding"]["color"]
        assert colour["scale"]["range"] == charts.TEAM_SLOTS[:2]

    def test_signal_uses_the_reserved_status_palette_in_its_own_order(self):
        spec = charts.slate_scatter(_hitters()).to_dict()
        colour = spec["encoding"]["color"]
        assert colour["scale"]["domain"] == charts.SIGNAL_ORDER
        assert colour["scale"]["range"] == [charts.SIGNAL_COLORS[s]
                                            for s in charts.SIGNAL_ORDER]

    def test_team_colour_never_exceeds_the_all_pairs_cap(self):
        """Scatter is an all-pairs form; the validated palette carries three series at most.

        A matchup is two clubs. If a caller ever hands a team-coloured scatter more, the
        scale must not silently invent hues.
        """
        assert len(charts.TEAM_SLOTS) <= 3

    def test_the_parity_line_spans_the_same_domain_as_the_points(self):
        """A diagonal that does not share the axis domain is a decorative line, not a rule."""
        spec = charts.platoon_scatter(_hitters(), view="parity").to_dict()
        rule_x = spec["layer"][0]["encoding"]["x"]["scale"]["domain"]
        point_x = spec["layer"][1]["encoding"]["x"]["scale"]["domain"]
        point_y = spec["layer"][1]["encoding"]["y"]["scale"]["domain"]
        assert rule_x == point_x == point_y

    def test_arsenal_ignores_players_with_no_sample_against_the_arsenal(self):
        frame = _hitters()
        frame["Arsenal AB"] = 0
        assert charts.arsenal_scatter(frame) is None


class TestSalaryValue:
    """Composite measured against what the market charged for it."""

    def _priced(self):
        from dashboards import salaries
        # Wide enough that each price band clears MIN_BAND; a thin fixture exercises only
        # the fallback and never the band logic the chart is actually built on.
        frame = _hitters(n=30)
        base = [2000, 2200, 2400, 2500, 2700, 2900, 3000, 3100, 3300, 3400,
                3500, 3600, 3700, 3900, 4000, 4100, 4200, 4300, 4500, 4600,
                4700, 4800, 5000, 5100, 5300, 5500, 5700, 5900, 6100, 6400]
        frame["Salary"] = base
        frame["DK Pos"] = (["C", "1B", "2B", "3B", "SS", "OF"] * 5)
        return salaries.with_surplus(frame)

    def test_par_is_the_band_median_not_one_number_for_the_whole_slate(self):
        """Each hitter is measured against hitters priced like him.

        Monotonicity of the curve is a property of real boards, not of this function, so
        what is pinned here is the mechanism: every band gets its own par, and a hitter's
        par is his band's median rather than the slate's.
        """
        from dashboards import salaries
        priced = self._priced()
        curve = salaries.band_curve(priced)
        assert len(curve) >= 4
        assert curve["expected"].nunique() > 1          # not one number for everybody

        slate_median = priced["Composite"].median()
        for _, band in curve.iterrows():
            members = priced[priced["band"] == band["band"]]
            assert members["expected"].nunique() == 1
            assert members["expected"].iloc[0] == pytest.approx(band["expected"])
        assert (curve["expected"] != slate_median).any()

    def test_a_thin_band_falls_back_rather_than_letting_one_bat_set_par(self):
        from dashboards import salaries
        frame = _hitters(n=3)
        frame["Salary"] = [2000, 2100, 6000]        # the 6k band holds one player
        out = salaries.with_surplus(frame)
        top = out[out["Salary"] == 6000].iloc[0]
        assert top["expected"] == pytest.approx(out["Composite"].median())

    def test_surplus_is_composite_minus_par(self):
        priced = self._priced()
        row = priced.iloc[0]
        assert row["surplus"] == pytest.approx(row["Composite"] - row["expected"])

    def test_points_per_thousand_is_available_alongside(self):
        priced = self._priced()
        row = priced.iloc[0]
        assert row["per_1k"] == pytest.approx(row["Composite"] / (row["Salary"] / 1000))

    def test_an_unpriced_hitter_gets_no_surplus_rather_than_zero(self):
        from dashboards import salaries
        frame = _hitters(n=4)
        frame["Salary"] = [2500, np.nan, 3500, np.nan]
        out = salaries.with_surplus(frame)
        assert out["surplus"].isna().sum() == 2

    def test_name_folding_matches_across_sources(self):
        from dashboards import salaries
        assert salaries.name_key("Luis Garcia Jr.") == salaries.name_key("Luis Garcia")
        assert salaries.name_key("José Ramírez") == salaries.name_key("Jose Ramirez")

    def test_the_charts_build(self):
        priced = self._priced()
        assert charts.salary_scatter(priced).to_dict()
        assert charts.surplus_bars(priced).to_dict()
        assert charts.price_position_scatter(priced).to_dict()

    def test_surplus_is_not_double_encoded_as_colour(self):
        """Bar length already is the surplus; a ramp would restate it in hue.

        Colour on the bars is a two-state sign, and on the scatter it carries Signal —
        which position does not already show.
        """
        priced = self._priced()
        bars = charts.surplus_bars(priced).to_dict()
        assert bars["layer"][1]["encoding"]["color"]["scale"]["domain"] == [
            "above par", "below par"]
        scatter = charts.salary_scatter(priced).to_dict()
        # The par curve is layered in only when enough bands qualify to draw one.
        points = scatter["layer"][-1] if "layer" in scatter else scatter
        assert points["encoding"]["color"]["field"] == "Signal"

    def test_a_slate_with_no_salaries_degrades_to_none(self):
        from dashboards import salaries
        frame = _hitters(n=4)
        frame["Salary"] = np.nan
        out = salaries.with_surplus(frame)
        assert charts.salary_scatter(out) is None
