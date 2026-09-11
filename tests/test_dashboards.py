"""Dashboard data and chart layer.

Streamlit's own runtime is not exercised here — these are the pure pieces underneath it:
filename parsing, typing of a cached payload, and the Altair specs. Chart builders are
checked by validating the Vega spec they produce, which catches composition mistakes that
only surface when the browser tries to render them.

`st.cache_data` wraps its target, so tests call `.__wrapped__` to reach the real function
rather than depending on a cache that has no runtime behind it.
"""

import os
import pickle

import numpy as np
import pandas as pd
import pytest

from dashboards import charts, data, drill_ui, scales


def _marks(spec):
    """The layer that draws the marks, by mark type rather than by position.

    Every scatter here is layered and the text layers sit on top, so "the points are the
    last layer" has not been true since direct labels arrived — and indexing past them was
    how a handful of these tests started asserting things about a `text` mark. This is the
    same rule `charts._marks_layer` uses to decide where the click selection goes.
    """
    layers = spec.get("layer")
    if not layers:
        return spec
    for layer in reversed(layers):
        mark = layer.get("mark")
        kind = mark if isinstance(mark, str) else (mark or {}).get("type")
        if kind in ("circle", "point", "square", "bar"):
            return layer
    return layers[-1]


def _hitters(n=6, teams=("ATH", "KC")):
    rows = []
    for i in range(n):
        rows.append({
            "Team": teams[i % len(teams)],
            "Name": f"Player {i}",
            "Bats": "LR"[i % 2],
            "Season OPS": 0.700 + i * 0.02,
            "L28 OPS": 0.690 + i * 0.018,
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

    def test_a_new_file_appears_without_waiting_for_a_ttl(self, tmp_path):
        assert data.list_games(str(tmp_path)).empty
        path = tmp_path / "2026-08-20_ATH_KC.pkl"
        with path.open("wb") as handle:
            pickle.dump({"version": 1}, handle)
        games = data.list_games(str(tmp_path))
        assert list(games["label"]) == ["ATH @ KC"]

    def test_an_in_place_rewrite_invalidates_the_payload_cache(self, tmp_path):
        path = tmp_path / "2026-08-20_ATH_KC.pkl"
        with path.open("wb") as handle:
            pickle.dump({"version": 1}, handle)
        assert data.load_payload(str(path))["version"] == 1
        before = path.stat().st_mtime_ns
        with path.open("wb") as handle:
            pickle.dump({"version": 2}, handle)
        # Filesystems with coarse timestamps still get an unambiguous version change.
        os.utime(path, ns=(before + 2_000_000_000, before + 2_000_000_000))
        assert data.load_payload(str(path))["version"] == 2


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

    def test_ops_context_can_be_attached_to_another_player_frame(self):
        target = pd.DataFrame({"Name": ["Player 0"], "K%": [28.0]})
        out = data.attach_hitter_ops(target, self._payload(_hitters()), self._meta())
        assert out.loc[0, "Season OPS"] == pytest.approx(0.700)
        assert out.loc[0, "L28 OPS"] == pytest.approx(0.690)
        assert out.loc[0, "Platoon OPS"] == pytest.approx(0.680)
        assert out.loc[0, "Arsenal OPS"] == pytest.approx(0.650)

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
        charts.salary_scatter, charts.ceiling_scatter,
        charts.ceiling_value_scatter,
    ])
    def test_every_hitter_hover_carries_the_four_ops_contexts(self, builder):
        frame = _hitters().assign(Salary=4000, Proj=np.arange(6) + 6,
                                  Ceiling=np.arange(6) + 12, Floor=np.arange(6) + 2,
                                  surplus=np.linspace(-4, 4, 6), per_1k=np.arange(6) + 1)
        fields = {tip["field"] for tip in _marks(builder(frame).to_dict())
                  ["encoding"]["tooltip"]}
        assert {"Season OPS", "L28 OPS", "Platoon OPS", "Arsenal OPS"} <= fields

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
        # parity rule, points, the named extremes, and the tier that zooming reveals
        assert "layer" in spec and len(spec["layer"]) == 4
        assert _marks(spec)["mark"]["type"] == "circle"

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
        colour = _marks(spec)["encoding"]["color"]
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



class TestAnchoredAxes:
    """A filter must change which points are drawn and never where they land.

    Every panel used to size its axes to whatever survived the filters, which meant the two
    things a reader was comparing -- the view before the switch and the view after it -- both
    moved at once, and neither position meant anything. `scales.anchor` is the fix and these
    are its consequences.
    """

    @staticmethod
    def _domain(chart, channel, field):
        spec = chart.to_dict()
        for layer in spec.get("layer", [spec]):
            encoding = layer.get("encoding", {}).get(channel, {})
            if encoding.get("field") == field:
                return tuple((encoding.get("scale") or {}).get("domain") or ())
        return None

    def test_a_filter_does_not_move_the_axis(self):
        board = _hitters(n=12, teams=("ATH", "KC", "TEX"))
        share = [("Season OPS", "Platoon OPS", "Arsenal OPS")]
        full = charts.basis_scatter(scales.anchor(board, board, share=share), "Season OPS")
        for mask in (board["Signal"] == "Priority", board["Season AB"] > 250,
                     board["Team"] == "ATH"):
            narrowed = scales.anchor(board[mask], board, share=share)
            chart = charts.basis_scatter(narrowed, "Season OPS")
            assert self._domain(chart, "x", "Composite") == \
                self._domain(full, "x", "Composite")
            assert self._domain(chart, "y", "Season OPS") == \
                self._domain(full, "y", "Season OPS")

    def test_switching_the_measure_does_not_rescale_the_plane(self):
        """The split control changes the question, not the units.

        Three OPS readings on one y axis have to share a domain or flipping between them
        moves every point for a reason that has nothing to do with the data.
        """
        board = _hitters(n=12)
        anchored = scales.anchor(board, board,
                                 share=[("Season OPS", "Platoon OPS", "Arsenal OPS")])
        domains = {basis: self._domain(charts.basis_scatter(anchored, basis), "y", basis)
                   for basis in ("Season OPS", "Platoon OPS", "Arsenal OPS")}
        assert len(set(domains.values())) == 1

    def test_a_frame_nobody_anchored_keeps_the_old_behaviour(self):
        """Anchoring is the page's call, so an un-anchored builder must not invent one."""
        chart = charts.basis_scatter(_hitters(n=8), "Season OPS")
        spec = chart.to_dict()
        scale = next(l["encoding"]["y"]["scale"] for l in spec["layer"]
                     if l.get("encoding", {}).get("y", {}).get("field") == "Season OPS")
        assert scale == {"zero": False}

    def test_full_bounds_cover_every_point_in_the_reference(self):
        """Full is the explicit completeness view and must include the entire board."""
        board = _hitters(n=12, teams=("ATH", "KC", "TEX"))
        anchored = scales.anchor(board[board["Team"] == "ATH"], board, mode="full")
        low, high = scales.of(anchored, "Composite")
        assert low <= board["Composite"].min() and high >= board["Composite"].max()

    def test_a_round_step_absorbs_a_small_change_in_the_data(self):
        """Two games whose ranges differ slightly get one axis, so stepping between them is a
        like-for-like read rather than a re-scale."""
        assert scales.nice(0.412, 0.988) == scales.nice(0.437, 0.971)

    def test_a_column_that_cannot_carry_a_domain_declines(self):
        """A zero-width domain is not an axis, and neither is one built from text."""
        assert scales.nice(5, 5) is None
        assert scales.nice(float("nan"), 1.0) is None
        frame = pd.DataFrame({"flat": [2.0, 2.0], "word": ["a", "b"]})
        assert scales.measure(frame) == {}

    def test_focus_domains_do_not_change_between_refreshes(self):
        quiet = _hitters(n=8)
        noisy = _hitters(n=8)
        noisy.loc[0, "Composite"] = 240.0
        noisy.loc[1, "Season OPS"] = 3.2
        for frame in (quiet, noisy):
            anchored = scales.anchor(frame, frame, mode="focus")
            assert scales.of(anchored, "Composite") == scales.FOCUS_DOMAINS["Composite"]
            assert scales.of(anchored, "Season OPS") == scales.FOCUS_DOMAINS["Season OPS"]

    def test_full_mode_expands_to_show_every_value(self):
        board = _hitters(n=8)
        board.loc[0, "Composite"] = 240.0
        anchored = scales.anchor(board, board, mode="full")
        assert scales.of(anchored, "Composite")[1] >= 240.0
        assert scales.overflow(anchored, ["Composite"]) == {}

    def test_focus_clamps_and_discloses_an_extreme(self):
        board = _hitters(n=8)
        board.loc[0, "Composite"] = 240.0
        anchored = scales.anchor(board, board, mode="focus")
        spec = charts.basis_scatter(anchored, "Season OPS").to_dict()
        scale = next(layer["encoding"]["x"]["scale"] for layer in spec["layer"]
                     if layer.get("encoding", {}).get("x", {}).get("field") == "Composite")
        assert scale["clamp"] is True
        assert scales.overflow(anchored, ["Composite"]) == {"Composite": (0, 1)}
        assert "Composite: 1 above" in scales.overflow_note(anchored, ["Composite"])

    def test_reference_medians_and_size_extents_survive_filtering(self):
        board = _hitters(n=12, teams=("ATH", "KC", "TEX"))
        anchored = scales.anchor(board[board["Team"] == "ATH"], board)
        assert scales.reference(anchored, "Season OPS") == pytest.approx(
            board["Season OPS"].median())
        assert scales.extent(anchored, "Season AB") == (
            float(board["Season AB"].min()), float(board["Season AB"].max()))

    def test_one_remaining_point_keeps_the_reference_size_mapping(self):
        board = _hitters(n=12)
        full = scales.anchor(board, board)
        one = scales.anchor(board.iloc[[0]], board)
        full_size = _marks(charts.basis_scatter(full, "Season OPS").to_dict()) \
            ["encoding"]["size"]["scale"]["domain"]
        one_size = _marks(charts.basis_scatter(one, "Season OPS").to_dict()) \
            ["encoding"]["size"]["scale"]["domain"]
        assert one_size == full_size


class TestZoom:
    """Zoom is what makes a fixed axis liveable: it holds still, and the reader opens up a
    crowded corner when they want one."""

    def test_the_scatters_take_pan_and_zoom_and_the_bars_do_not(self):
        """Panning a ranked bar chart hides rows without saying so."""
        board = _hitters(n=8)
        priced = board.assign(Salary=range(3000, 3000 + 8 * 200, 200),
                              surplus=range(-4, 4), expected=50.0, per_1k=8.0)
        assert self._params(charts.form_scatter(board)) == [scales.ZOOM]
        assert self._params(charts.surplus_bars(priced)) == []

    def test_the_binding_sits_on_the_marks_layer(self):
        """On the reference-rule layer it would bind that rule's own scale, and the dashed
        line would slide about while the points sat still.

        Altair hoists a layer's params to the top of the spec and records which layer they
        belong to under `views`, so the layer is identified by the name that ends up there
        rather than by where the param is written.
        """
        spec = charts.form_scatter(_hitters(n=8)).to_dict()
        views = {view for param in spec["params"] for view in param.get("views", [])}
        owner = [layer for layer in spec["layer"] if layer.get("name") in views]
        assert len(owner) == 1
        assert owner[0]["mark"]["type"] == "circle"

    def test_zoom_and_the_click_selection_coexist(self):
        """The drill-down is the reason these charts exist; zoom must not cost it."""
        chart = charts.selectable(charts.form_scatter(_hitters(n=8)))
        assert set(self._params(chart)) == {scales.ZOOM, "point"}

    def test_only_the_point_selection_triggers_a_streamlit_rerun(self, monkeypatch):
        captured = {}

        def render(chart, **kwargs):
            captured.update(kwargs)
            return {"selection": {}}

        monkeypatch.setattr(drill_ui.st, "altair_chart", render)
        drill_ui.chart_with_drilldown(charts.form_scatter(_hitters(n=8)), _hitters(n=8),
                                     "2026-08-20", "chart")
        assert captured["on_select"] == "rerun"
        assert captured["selection_mode"] == "point"

    def test_the_predicate_names_the_x_field_and_is_false_before_any_interaction(self):
        """`zoom` resolves to an empty object until the reader touches the chart, so the
        `isValid` guard is both the "has anyone zoomed" test and the guard that keeps a chart
        whose x field never reached the binding quiet rather than broken."""
        test = scales.zoomed_in("Season OPS", 0.3)
        assert "zoom['Season OPS']" in test
        assert test.startswith("isValid(zoom) && isValid(zoom['Season OPS'])")
        assert test.endswith("< 0.3")

    @staticmethod
    def _params(chart):
        spec = chart.to_dict()
        found = []
        for layer in spec.get("layer", [spec]):
            for param in layer.get("params", []):
                found.append(param["name"])
        for param in spec.get("params", []):
            found.append(param["name"])
        return found


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
        # The par curve is layered in only when enough bands qualify to draw one, and the
        # name labels sit on top of both, so the points are found by mark type.
        assert _marks(scatter)["encoding"]["color"]["field"] == "Signal"

    def test_a_slate_with_no_salaries_degrades_to_none(self):
        from dashboards import salaries
        frame = _hitters(n=4)
        frame["Salary"] = np.nan
        out = salaries.with_surplus(frame)
        assert charts.salary_scatter(out) is None
