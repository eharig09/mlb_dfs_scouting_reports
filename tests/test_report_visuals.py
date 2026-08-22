"""Key-metric visuals.

These are decision aids printed at the top of the report, so the failure that matters is
not an ugly chart -- it is a chart that draws something *wrong* (the tonight split marked
on the wrong row, a club's colour changing between panels) or that takes the whole report
down when one table happens to be empty. Everything here is synthetic; nothing touches the
network or the cache.
"""

import os

import numpy as np
import pandas as pd
import pytest

import report_visuals as rv


def _splits(team, tonight="vs RHP"):
    rows = [("Overall", 0.714, 2900), ("vs LHP", 0.662, 800),
            ("vs RHP", 0.734, 2100), ("L28", 0.622, 700)]
    return pd.DataFrame([
        {"Team": team, "Split": f"{name} *" if name == tonight else name,
         "PA": pa, "OPS": ops}
        for name, ops, pa in rows
    ])


def _type_results(team, typed=0.842, baseline=0.766):
    return pd.DataFrame([{
        "Team": team, "Games": 12, "Record": "7-5", "Runs/G": 5.8, "PA": 280,
        "Type OPS": typed, "Baseline OPS": baseline, "OPS Diff": "+76 OPS pts",
        "Pitcher Type": "R / SI/FF/FC", "Comps": 6, "Confidence": "High",
    }])


def _rolling(team):
    return pd.DataFrame([
        {"Team": team, "Window": w, "Runs/G": r, "Allowed/G": a, "Run Diff/G": r - a}
        for w, r, a in (("7D", 5.29, 4.14), ("14D", 4.50, 4.07), ("30D", 4.46, 4.65))
    ])


def _context():
    return {
        "away_lineup_splits": _splits("NYY"), "home_lineup_splits": _splits("BAL"),
        "away_type_results": _type_results("NYY"),
        "home_type_results": _type_results("BAL", typed=0.718, baseline=0.735),
        "away_rolling_form": _rolling("NYY"), "home_rolling_form": _rolling("BAL"),
    }


class TestSplitParsing:
    def test_the_trailing_star_marks_tonights_split(self):
        """The '*' is the only carrier of 'this is the split that applies tonight'."""
        found = rv._split_lookup(_splits("NYY", tonight="vs LHP"))
        assert found["vs LHP"][2] is True
        assert found["vs RHP"][2] is False
        # And the star must not survive into the key.
        assert "vs LHP *" not in found

    def test_venue_rows_key_on_the_bare_word(self):
        frame = pd.DataFrame([{"Split": "Home (PF)", "OPS": 0.728, "PA": 1424}])
        assert "Home" in rv._split_lookup(frame)

    def test_a_frame_without_a_split_column_is_empty_not_an_error(self):
        assert rv._split_lookup(pd.DataFrame({"OPS": [0.7]})) == {}
        assert rv._split_lookup(pd.DataFrame()) == {}
        assert rv._split_lookup(None) == {}


class TestFigure:
    def test_it_renders_all_three_panels(self, tmp_path):
        out = tmp_path / "viz.png"
        assert rv.build_key_metrics_figure("NYY", "BAL", _context(), str(out)) == str(out)
        assert out.exists() and out.stat().st_size > 5000
        width, height = rv.figure_aspect(str(out))
        assert width > height          # landscape, to sit full-width on a landscape page

    def test_nothing_is_written_when_there_is_no_data(self, tmp_path):
        """An empty frame beats an empty chart -- the report omits the section."""
        out = tmp_path / "viz.png"
        assert rv.build_key_metrics_figure("NYY", "BAL", {}, str(out)) is None
        assert not out.exists()

    @pytest.mark.parametrize("drop", ["lineup_splits", "type_results", "rolling_form"])
    def test_one_missing_table_drops_its_panel_not_the_figure(self, tmp_path, drop):
        """A club with no comp bench must not cost the other two panels."""
        context = _context()
        for side in ("away", "home"):
            context[f"{side}_{drop}"] = pd.DataFrame()
        out = tmp_path / f"viz_{drop}.png"
        assert rv.build_key_metrics_figure("NYY", "BAL", context, str(out)) == str(out)
        assert out.exists()

    def test_a_one_sided_panel_still_draws(self, tmp_path):
        """Only one club having type results is normal, not a failure."""
        context = _context()
        context["home_type_results"] = pd.DataFrame()
        out = tmp_path / "viz.png"
        assert rv.build_key_metrics_figure("NYY", "BAL", context, str(out)) == str(out)

    def test_non_numeric_values_do_not_raise(self, tmp_path):
        context = _context()
        context["away_lineup_splits"] = pd.DataFrame([
            {"Team": "NYY", "Split": "Overall", "PA": "-", "OPS": "-"},
            {"Team": "NYY", "Split": "vs RHP *", "PA": 900, "OPS": 0.71},
        ])
        out = tmp_path / "viz.png"
        assert rv.build_key_metrics_figure("NYY", "BAL", context, str(out)) == str(out)


class TestColourContract:
    def test_the_two_team_colours_are_the_validated_pair(self):
        """Changing these means re-running scripts/validate_palette.js.

        The pair was checked on the light surface (worst CVD ΔE 24.7, normal-vision 33.6,
        both above 3:1 contrast). A hand-picked replacement is exactly the change that
        quietly breaks colour-vision-deficient and greyscale readers.
        """
        assert rv.AWAY_COLOR == "#2a78d6"
        assert rv.HOME_COLOR == "#eb6834"
        assert rv.AWAY_COLOR != rv.HOME_COLOR

    def test_colour_follows_the_side_even_when_the_home_club_hits_better(self):
        """Whoever hits better must not get a particular colour.

        Checked on the drawn bars rather than by grepping the source: panels legitimately
        sort rows for display order, so "does the file call sort_values" says nothing about
        whether colour tracks rank. Swapping which club is stronger must leave the hues
        exactly where they were.
        """
        strong, weak = _splits("AAA"), _splits("BBB")
        weak.loc[:, "OPS"] = weak["OPS"] * 0.5

        def away_and_home_colours(away_splits, home_splits):
            fig, ax = rv.plt.subplots()
            rv._panel_platoon(ax, "AAA", "BBB", away_splits, home_splits)
            bars = [p for p in ax.patches]
            half = len(bars) // 2
            colours = ({p.get_facecolor() for p in bars[:half]},
                       {p.get_facecolor() for p in bars[half:]})
            rv.plt.close(fig)
            return colours

        away_first, home_first = away_and_home_colours(strong, weak)
        swapped_away, swapped_home = away_and_home_colours(weak, strong)
        assert away_first == swapped_away
        assert home_first == swapped_home

        from matplotlib.colors import to_rgba
        assert away_first == {to_rgba(rv.AWAY_COLOR)}
        assert home_first == {to_rgba(rv.HOME_COLOR)}


class TestRegressions:
    """Two bugs the synthetic fixture missed and a real slate caught."""

    def test_each_club_gets_its_own_tonight_marker(self):
        """In a L-vs-R matchup both hand rows are live -- for different clubs.

        The first version OR-ed the two clubs' flags together, so an LHP-vs-RHP game
        marked *both* 'vs LHP' and 'vs RHP' as 'tonight', which reads as each team facing
        both hands. Caught on LAD@COL.
        """
        away = _splits("LAD", tonight="vs LHP")     # LAD faces the lefty
        home = _splits("COL", tonight="vs RHP")     # COL faces the righty
        fig, ax = rv.plt.subplots()
        assert rv._panel_platoon(ax, "LAD", "COL", away, home)
        labels = [t.get_text() for t in ax.get_yticklabels()]
        rv.plt.close(fig)

        lhp = next(l for l in labels if l.startswith("vs LHP"))
        rhp = next(l for l in labels if l.startswith("vs RHP"))
        assert "LAD" in lhp and "COL" not in lhp
        assert "COL" in rhp and "LAD" not in rhp
        # A row nobody faces tonight carries no marker at all.
        assert all("◄" not in l for l in labels if l.startswith(("Overall", "L28")))

    def test_a_verbose_pitcher_type_is_reduced_to_its_codes(self):
        """The field has two shapes; the long one ran across the neighbouring panel."""
        verbose = ("L / FF 36% (91.5 mph/2378 rpm)/FC 32% (87.7 mph/2373 rpm)"
                   "/KC 32% (82.5 mph/2363 rpm)")
        assert rv._short_pitcher_type(verbose) == "L / FF/FC/KC"
        # The terse form is already fine and must survive untouched.
        assert rv._short_pitcher_type("R / SI/FF/FC") == "R / SI/FF/FC"
        assert rv._short_pitcher_type("") == ""
        assert rv._short_pitcher_type(None) == ""

    def test_any_pitcher_type_fits_the_label_budget(self):
        """Whatever shape arrives, it cannot be long enough to collide."""
        for raw in ("L / FF 36% (91.5 mph/2378 rpm)/FC 32% (87.7 mph/2373 rpm)",
                    "R / " + "/".join(["SI"] * 40),
                    "a really long free-text description with no codes at all in it"):
            assert len(rv._short_pitcher_type(raw)) <= 30

    def test_a_blank_confidence_leaves_no_dangling_separator(self, tmp_path):
        context = _context()
        frame = _type_results("NYY")
        frame.loc[0, "Confidence"] = ""
        context["away_type_results"] = frame
        fig, ax = rv.plt.subplots()
        assert rv._panel_type(ax, "NYY", "BAL", frame, context["home_type_results"])
        texts = [t.get_text() for t in ax.texts]
        rv.plt.close(fig)
        assert not any(t.rstrip().endswith("·") for t in texts), texts


def _hand_splits(pitcher, ops_l=0.817, ops_r=0.682, tag_l="Vulnerable", tag_r="Neutral"):
    return pd.DataFrame([
        {"Pitcher": pitcher, "Batter Side": "L", "PA": 209, "OPS": ops_l,
         "SLG": 0.444, "xwOBA": 0.376, "K%": 13.9, "HardHit%": 22.7, "Split Tag": tag_l},
        {"Pitcher": pitcher, "Batter Side": "R", "PA": 142, "OPS": ops_r,
         "SLG": 0.344, "xwOBA": 0.294, "K%": 14.8, "HardHit%": 22.3, "Split Tag": tag_r},
    ])


def _arsenal(edges=("hitter", "pitcher", "even")):
    return pd.DataFrame([
        {"Pitch": p, "Usage%": u, "Pitches": 1000, "RV/100": rv, "Whiff%": 20.0,
         "Chase%": 25.0, "Contact%": 80.0, "xwOBA": 0.330, "Edge": e}
        for p, u, rv, e in zip(("FF", "SI", "SL"), (38.5, 25.6, 19.5),
                               (-0.64, 0.68, -0.23), edges)
    ])


def _pen(team_scale=1.0):
    names = ["Rico Garcia", "Josh Walker", "Cam Sanders", "Yennier Cano"]
    avail = ["Available", "Monitor", "Available", "Taxed"]
    return pd.DataFrame([
        {"Name": n, "T": "R", "App": 3, "IP": 3.0, "Pit": int(p * team_scale),
         "P/Out": 5.7, "ERA": 0.0, "K-BB": 1, "Avail": a}
        for n, p, a in zip(names, (51, 46, 45, 39), avail)
    ])


def _composite():
    rows = []
    for team, scores in (("NYY", (45.4, 21.0, -3.0)), ("BAL", (41.4, 7.7, -18.0))):
        for rank, score in enumerate(scores, start=1):
            rows.append({
                "Team": team, "Rank": rank, "Name": f"{team} Bat{rank}", "Bats": "L",
                "Composite": score,
                "Signal": "Priority" if score > 40 else "Fade" if score < -10 else "Neutral",
            })
    return pd.DataFrame(rows)


def _full_context():
    context = _context()
    context.update({
        "away_pitcher_hand_splits": _hand_splits("Will Warren", 0.784, 0.690, "Neutral"),
        "home_pitcher_hand_splits": _hand_splits("Chris Bassitt"),
        "away_arsenal_matchup": _arsenal(),
        "home_arsenal_matchup": _arsenal(),
        "away_bullpen_l5": (pd.DataFrame({"Date": ["08-18"]}), _pen()),
        "home_bullpen_l5": (pd.DataFrame({"Date": ["08-18"]}), _pen(team_scale=1.8)),
        "hitter_composite": _composite(),
    })
    return context


class TestStarterPanel:
    def test_it_names_the_side_that_beats_him(self):
        fig, ax = rv.plt.subplots()
        assert rv._panel_starter_platoon(ax, "NYY", "BAL",
                                         _hand_splits("Warren", 0.60, 0.90, "Neutral", "Vulnerable"),
                                         _hand_splits("Bassitt"))
        texts = " | ".join(t.get_text() for t in ax.texts)
        rv.plt.close(fig)
        # Warren is beaten by RHB here, Bassitt by LHB; each must be called out on its own row.
        assert "RHB 0.900" in texts and "Vulnerable" in texts
        assert "LHB 0.817" in texts

    def test_a_starter_with_only_one_side_is_skipped(self):
        one_side = _hand_splits("Warren").head(1)
        fig, ax = rv.plt.subplots()
        drawn = rv._panel_starter_platoon(ax, "NYY", "BAL", one_side, pd.DataFrame())
        rv.plt.close(fig)
        assert drawn is False


class TestArsenalPanel:
    def test_the_edge_word_rides_the_tick_label_not_the_bar(self):
        """A left-growing bar labels into the tick-label gutter; the third field collided."""
        fig, ax = rv.plt.subplots()
        assert rv._panel_arsenal_edge(ax, "NYY bats vs X", _arsenal(), rv.AWAY_COLOR)
        ticks = [t.get_text() for t in ax.get_yticklabels()]
        bar_labels = [t.get_text() for t in ax.texts]
        rv.plt.close(fig)
        assert any("hitter" in t for t in ticks)
        assert all("hitter" not in t and "pitcher" not in t for t in bar_labels)
        assert any("% use" in t for t in bar_labels)

    def test_it_is_drawn_off_a_zero_rule(self):
        """Sign is the polarity channel here, so zero has to be on the chart."""
        fig, ax = rv.plt.subplots()
        rv._panel_arsenal_edge(ax, "t", _arsenal(), rv.AWAY_COLOR)
        lo, hi = ax.get_xlim()
        rv.plt.close(fig)
        assert lo < 0 < hi


class TestSharedScales:
    def test_both_bullpens_share_one_x_scale(self, tmp_path):
        """Side-by-side panels invite a comparison independent axes cannot support."""
        context = _full_context()          # home pen throws 1.8x the away pen
        out = tmp_path / "pen.png"
        assert rv.build_bullpen_figure("NYY", "BAL", context, str(out)) == str(out)

        fig, axes = rv.plt.subplots(1, 2)
        frames = {s: rv._bullpen_usage_frame(context[f"{s}_bullpen_l5"]) for s in ("away", "home")}
        top = float(max(f["Pit"].max() for f in frames.values()))
        rv._panel_bullpen(axes[0], "NYY", frames["away"], rv.AWAY_COLOR, shared_top=top)
        rv._panel_bullpen(axes[1], "BAL", frames["home"], rv.HOME_COLOR, shared_top=top)
        assert axes[0].get_xlim() == axes[1].get_xlim()
        rv.plt.close(fig)

    def test_both_target_panels_share_one_x_scale(self):
        composite = _composite()
        span = float(composite["Composite"].abs().max())
        fig, axes = rv.plt.subplots(1, 2)
        rv._panel_hitter_targets(axes[0], "NYY", composite, rv.AWAY_COLOR, shared_span=span)
        rv._panel_hitter_targets(axes[1], "BAL", composite, rv.HOME_COLOR, shared_span=span)
        assert axes[0].get_xlim() == axes[1].get_xlim()
        rv.plt.close(fig)


class TestBullpenPayload:
    def test_the_usage_frame_is_found_by_shape_not_by_index(self):
        """`{side}_bullpen_l5` is a tuple whose order is not guaranteed anywhere."""
        game_log = pd.DataFrame({"Date": ["08-18"], "Opp": ["vs NYY"]})
        usage = _pen()
        assert rv._bullpen_usage_frame((game_log, usage)) is usage
        assert rv._bullpen_usage_frame((usage, game_log)) is usage
        assert rv._bullpen_usage_frame(None) is None
        assert rv._bullpen_usage_frame((game_log,)) is None

    def test_status_words_ride_alongside_the_status_colour(self):
        """Status colour must never be the only carrier of availability."""
        fig, ax = rv.plt.subplots()
        assert rv._panel_bullpen(ax, "BAL", _pen(), rv.HOME_COLOR)
        texts = " | ".join(t.get_text() for t in ax.texts)
        rv.plt.close(fig)
        for word in ("Available", "Monitor", "Taxed"):
            assert word in texts


class TestTargetsPanel:
    def test_the_report_signal_word_is_shown_not_re_derived(self):
        fig, ax = rv.plt.subplots()
        assert rv._panel_hitter_targets(ax, "BAL", _composite(), rv.HOME_COLOR)
        texts = " | ".join(t.get_text() for t in ax.texts)
        rv.plt.close(fig)
        assert "Priority" in texts or "Fade" in texts

    def test_a_team_with_no_rows_draws_nothing(self):
        fig, ax = rv.plt.subplots()
        drawn = rv._panel_hitter_targets(ax, "SEA", _composite(), rv.AWAY_COLOR)
        rv.plt.close(fig)
        assert drawn is False


class TestFigureSet:
    def test_all_four_figures_render_from_a_full_context(self, tmp_path):
        made = rv.build_report_figures("NYY", "BAL", _full_context(), str(tmp_path),
                                       game_date="2026-08-19")
        assert [caption for caption, _ in made] == [
            "Offense", "Starting pitching", "Bullpen availability", "Hitters to target"]
        for _, path in made:
            assert os.path.exists(path) and os.path.getsize(path) > 5000

    def test_an_empty_context_yields_no_figures(self, tmp_path):
        assert rv.build_report_figures("NYY", "BAL", {}, str(tmp_path)) == []

    def test_one_broken_section_does_not_stop_the_others(self, tmp_path):
        context = _full_context()
        context["hitter_composite"] = "not a frame at all"
        made = rv.build_report_figures("NYY", "BAL", context, str(tmp_path))
        assert [c for c, _ in made] == ["Offense", "Starting pitching", "Bullpen availability"]


class TestSmallSampleArsenal:
    """A pitch the report flagged as low-sample must not set the axis.

    Caught on LAD@COL: a 10%-usage forkball at +6.77 RV/100 -- tagged "small", meaning
    fewer than min_pitches seen, so the number is noise -- stretched the axis to +/-20 and
    squashed every trustworthy bar in the panel into a sliver. The biggest bar on the chart
    was its least reliable number.
    """

    def _with_outlier(self):
        frame = _arsenal()
        frame.loc[len(frame)] = {"Pitch": "FO", "Usage%": 10.0, "Pitches": 12,
                                 "RV/100": 6.77, "Whiff%": 0.0, "Chase%": 0.0,
                                 "Contact%": 0.0, "xwOBA": 0.5, "Edge": "small"}
        return frame

    def test_the_low_sample_pitch_is_left_off(self):
        fig, ax = rv.plt.subplots()
        assert rv._panel_arsenal_edge(ax, "t", self._with_outlier(), rv.AWAY_COLOR)
        ticks = [t.get_text() for t in ax.get_yticklabels()]
        rv.plt.close(fig)
        assert not any(t.startswith("FO") for t in ticks)

    def test_it_no_longer_sets_the_scale(self):
        fig, ax = rv.plt.subplots()
        rv._panel_arsenal_edge(ax, "t", self._with_outlier(), rv.AWAY_COLOR)
        lo, hi = ax.get_xlim()
        rv.plt.close(fig)
        # Real values top out at 0.68; the axis must reflect those, not the 6.77 outlier.
        assert hi < 4.0, (lo, hi)

    def test_what_was_hidden_is_disclosed(self):
        """Silently showing a partial arsenal would be its own kind of wrong."""
        fig, ax = rv.plt.subplots()
        rv._panel_arsenal_edge(ax, "t", self._with_outlier(), rv.AWAY_COLOR)
        label = ax.get_xlabel()
        rv.plt.close(fig)
        assert "1 low-sample pitch hidden" in label

    def test_an_all_small_arsenal_draws_nothing(self):
        frame = _arsenal()
        frame["Edge"] = "small"
        fig, ax = rv.plt.subplots()
        drawn = rv._panel_arsenal_edge(ax, "t", frame, rv.AWAY_COLOR)
        rv.plt.close(fig)
        assert drawn is False
