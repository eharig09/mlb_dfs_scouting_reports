"""Shared filters, conditions, and the surrendered-OPS baseline.

The common failure mode across all three is the same: a view that *renders* while quietly
describing a different subset than the reader believes. A position filter that drops
multi-slot players, a switch hitter matched to the wrong split, a weather term applied under
a closed roof — none of them raise, and all of them are wrong.
"""

import numpy as np
import pandas as pd
import pytest

from dashboards import charts, environment as env, filters


def _board(n=12):
    slots = ["OF", "1B/OF", "SS", "C", "2B/SS", "3B"] * 4
    return pd.DataFrame({
        "Name": [f"Player {i}" for i in range(n)],
        "Team": ["ATH", "KC", "TEX"] * (n // 3),
        "Bats": ["L", "R", "S"] * (n // 3),
        "Signal": ["Priority", "Watch", "Neutral", "Fade"] * (n // 4),
        "Composite": np.linspace(-10, 40, n),
        "Season AB": np.linspace(0, 500, n),
        "Season OPS": np.linspace(0.600, 0.900, n),
        "DK Pos": slots[:n],
        "slates": [["main"], ["main", "late"], ["early"]] * (n // 3),
    })


class TestPositionFilter:
    def test_a_multi_slot_player_matches_any_of_his_slots(self):
        """DK writes every slot a player qualifies for into one field. An equality test
        drops exactly the multi-position players a roster is built around."""
        view = filters.apply_filters(_board(), {"position": ["OF"],
                                                "_position_column": "DK Pos"})
        assert set(view["DK Pos"]) == {"OF", "1B/OF"}

    def test_matching_is_case_and_separator_tolerant(self):
        frame = _board().assign(**{"DK Pos": ["of", "1b,of", "SS", "C", "2B/SS", "3B"] * 2})
        view = filters.apply_filters(frame, {"position": ["OF"],
                                             "_position_column": "DK Pos"})
        assert len(view) == 4

    def test_no_selection_means_everything_not_nothing(self):
        board = _board()
        assert len(filters.apply_filters(board, {"position": [],
                                                 "_position_column": "DK Pos"})) == len(board)


class TestSlateFilter:
    def test_a_player_on_two_slates_survives_either(self):
        """A game appears on several DK slates. Membership is a set, and testing it by
        equality would show a main-slate reader only the main-slate-*exclusive* players."""
        board = _board()
        assert len(filters.apply_filters(board, {"slate": ["late"]})) == 4
        assert len(filters.apply_filters(board, {"slate": ["main"]})) == 8

    def test_selecting_several_slates_unions_them(self):
        board = _board()
        both = filters.apply_filters(board, {"slate": ["late", "early"]})
        assert len(both) == 8

    def test_a_frame_with_no_membership_is_not_silently_emptied(self):
        board = _board().drop(columns=["slates"])
        assert len(filters.apply_filters(board, {"slate": ["main"]})) == len(board)


class TestOtherFilters:
    def test_they_compose(self):
        view = filters.apply_filters(_board(), {
            "team": ["ATH"], "bats": "L", "ab": 0,
            "position": ["OF", "SS", "C", "1B", "2B", "3B"],
            "_position_column": "DK Pos"})
        assert set(view["Team"]) == {"ATH"}
        assert set(view["Bats"]) == {"L"}

    def test_bats_all_is_not_a_filter(self):
        board = _board()
        assert len(filters.apply_filters(board, {"bats": "All"})) == len(board)

    def test_an_empty_frame_survives(self):
        assert filters.apply_filters(pd.DataFrame(), {"team": ["ATH"]}).empty


class TestBases:
    def test_only_bases_with_enough_data_are_offered(self):
        """A basis in the menu that plots four points is worse than one that is absent."""
        board = _board().assign(xwOBA=[0.31] * 3 + [np.nan] * 9)
        available = filters.available_bases(board, minimum=20)
        assert "Season OPS" not in available          # 12 rows, under the floor
        assert "xwOBA" not in available
        assert filters.available_bases(board, minimum=5).get("Season OPS")

    def test_the_season_rate_line_is_offered(self):
        """ISO, AVG and OBP come from the **lineup card** in `report_args`, which carries a
        full season line per hitter (162 of 162 on a checked slate).

        An earlier version of this module asserted ISO was *not* derivable, on the reasoning
        that the only per-hitter batting averages were in the head-to-head tables. That was
        wrong — it missed the card entirely — and the test encoded the mistake.
        """
        for basis in ("Season ISO", "Season AVG", "Season OBP"):
            assert basis in filters.BASES

    def test_the_card_slg_does_not_overwrite_the_matchup_slg(self):
        """`SLG` means two different things here: season-long on the card, against tonight's
        pitch mix in the arsenal frame. Merging blind would silently replace one with the
        other under a name that reads correct either way."""
        from dashboards import data
        assert "SLG" in data.ARSENAL_MEASURES
        assert "Season SLG" not in data.ARSENAL_MEASURES
        assert "SLG" in data.CARD_MEASURES


class TestConditions:
    def test_a_closed_roof_pins_the_weather_term_to_neutral(self):
        """Still air and a controlled temperature: neither term should move. A hot day
        outside a closed roof must not become a home-run nudge."""
        payload = {"advanced_context": {"environment": {
            "park": {"HR": 1.22, "Runs": 1.05}, "park_name": "Tropicana Field",
            "forecast": {"roof": "dome", "temp_f": 95.0, "wind": "12 mph, Out To CF"}}}}
        conditions = env.hr_environment(payload)
        assert conditions["weather_hr"] == 1.0
        assert conditions["hr_env"] == pytest.approx(1.22)
        assert conditions["enclosed"]

    def test_an_open_park_lets_temperature_move_it(self):
        hot = {"advanced_context": {"environment": {
            "park": {"HR": 1.0}, "forecast": {"roof": "open", "temp_f": 95.0}}}}
        cold = {"advanced_context": {"environment": {
            "park": {"HR": 1.0}, "forecast": {"roof": "open", "temp_f": 50.0}}}}
        assert env.hr_environment(hot)["weather_hr"] > 1.0
        assert env.hr_environment(cold)["weather_hr"] < 1.0

    def test_a_missing_park_factor_is_neutral_rather_than_zero(self):
        """A NaN multiplier would wipe out the environment for that game entirely."""
        conditions = env.hr_environment({"advanced_context": {"environment": {}}})
        assert conditions["park_hr"] == 1.0
        assert conditions["hr_env"] == pytest.approx(conditions["weather_hr"])

    def test_direction_calls_a_small_wobble_nothing(self):
        assert env.direction(1.01) == "neutral"
        assert env.direction(0.99) == "neutral"
        assert env.direction(1.20) == "up"
        assert env.direction(0.80) == "down"
        assert env.direction(float("nan")) == "neutral"

    def test_exposure_scales_with_fly_ball_rate_and_vanishes_at_neutral(self):
        """The heuristic's two claims, and the only two it makes: a fly-ball arm is more
        exposed than a ground-ball arm in the same park, and nobody is exposed to a neutral
        environment."""
        flyball = env.hr_leverage(35.0, 1.20)
        groundball = env.hr_leverage(18.0, 1.20)
        assert flyball > groundball > 0
        assert env.hr_leverage(35.0, 1.0) == pytest.approx(0.0)
        assert env.hr_leverage(35.0, 0.85) < 0

    def test_exposure_is_undefined_rather_than_zero_when_unknown(self):
        """Zero would sort into the middle of the ranking as though it were measured."""
        assert pd.isna(env.hr_leverage(None, 1.2))
        assert pd.isna(env.hr_leverage(30.0, float("nan")))


def _staffs():
    return pd.DataFrame({
        "name": ["A", "B", "C", "D"], "team": ["ATH", "KC", "TEX", "WSH"],
        "game": ["ATH @ KC"] * 2 + ["TEX @ WSH"] * 2,
        "unit": ["Starter", "Bullpen"] * 2,
        "park": ["Coliseum", "Coliseum", "Globe Life", "Globe Life"],
        "FB%": [30.0, 22.0, 27.0, 19.0], "GB%": [40.0, 48.0, 43.0, 52.0],
        "LD%": [22.0, 24.0, 23.0, 21.0], "GB+LD%": [62.0, 72.0, 66.0, 73.0],
        "BIP": [120, 480, 150, 500],
        "park_hr": [0.92, 0.92, 1.10, 1.10], "weather_hr": [1.01, 1.01, 0.99, 0.99],
        "hr_env": [0.93, 0.93, 1.09, 1.09],
        "temp": [68.0] * 4, "wind": ["5 mph, In From LF"] * 4,
        "impact": ["down", "down", "up", "up"],
        "hr_leverage": [-8.4, -6.2, 9.8, 6.8],
        "park_2b": [96.0, 96.0, 104.0, 104.0], "park_r": [94.0, 94.0, 102.0, 102.0],
        "hits_saved": [0.12, 0.12, -0.08, -0.08],
        "defense_grade": ["Plus", "Plus", "Neutral", "Neutral"],
    })


class TestConditionCharts:
    def test_each_builds_a_valid_spec(self):
        for chart in (charts.hr_exposure_scatter(_staffs()),
                      charts.hr_impact_arrows(_staffs()),
                      charts.defense_park_scatter(_staffs()),
                      charts.defense_park_scatter(_staffs(), x="GB%",
                                                  park_column="park_r")):
            assert chart is not None and chart.to_dict()

    def test_an_empty_frame_degrades_to_none(self):
        empty = _staffs().iloc[0:0]
        assert charts.hr_exposure_scatter(empty) is None
        assert charts.hr_impact_arrows(empty) is None
        assert charts.defense_park_scatter(empty) is None

    def test_direction_is_shape_not_colour(self):
        """The status palette is reserved, and "up" is good for hitters and bad for
        pitchers depending on who is reading — a red/green ramp would take a side the chart
        has no business taking."""
        spec = charts.hr_exposure_scatter(_staffs()).to_dict()
        points = spec["layer"][-1]["encoding"]
        assert points["shape"]["field"] == "impact"
        assert points["color"]["field"] == "unit"

    def test_the_exposure_ranking_marks_zero(self):
        spec = charts.hr_impact_arrows(_staffs()).to_dict()
        assert any(layer["mark"]["type"] == "rule" for layer in spec["layer"])

    def test_hits_saved_is_not_encoded_as_a_size(self):
        """It swings both ways around zero and a negative radius is meaningless."""
        spec = charts.defense_park_scatter(_staffs()).to_dict()
        points = spec["layer"][-1]["encoding"]
        assert "size" not in points
        assert points["color"]["field"] == "defence"


def _matchups():
    return pd.DataFrame({
        "Name": [f"Hitter {i}" for i in range(6)],
        "Team": ["ATH"] * 3 + ["KC"] * 3,
        "Bats": ["L", "R", "S", "L", "R", "S"],
        "Effective Side": ["L", "R", "R", "L", "R", "L"],
        "Faces Hand": ["R", "R", "R", "L", "L", "R"],
        "Opp SP": ["Arm A"] * 3 + ["Arm B"] * 3,
        "Signal": ["Priority", "Watch", "Neutral", "Fade", "Watch", "Priority"],
        "Arsenal OPS": [0.900, 0.700, 0.820, 0.640, 0.980, 0.750],
        "Platoon OPS": [0.850, 0.720, 0.800, 0.660, 0.900, 0.770],
        "Season OPS": [0.800, 0.740, 0.780, 0.700, 0.860, 0.760],
        "Allowed OPS": [0.700, 0.700, 0.700, 0.780, 0.780, 0.780],
        "Allowed PA": [296, 296, 296, 211, 211, 211],
        "Arsenal AB": [40, 25, 8, 30, 22, 12],
        "Split Tag": ["Suppresses"] * 6,
    }).assign(**{"Arsenal Edge": lambda d: d["Arsenal OPS"] - d["Allowed OPS"],
                 "Platoon Edge": lambda d: d["Platoon OPS"] - d["Allowed OPS"],
                 "Season Edge": lambda d: d["Season OPS"] - d["Allowed OPS"]})


class TestAllowedBaseline:
    def test_the_chart_shares_one_domain_across_both_axes(self):
        """The diagonal is the entire chart. Two different domains make it decorative."""
        spec = charts.arsenal_vs_allowed_scatter(_matchups()).to_dict()
        points = spec["layer"][-1]["encoding"]
        assert points["x"]["scale"]["domain"] == points["y"]["scale"]["domain"]
        rule = spec["layer"][0]["encoding"]
        assert rule["x"]["scale"]["domain"] == points["x"]["scale"]["domain"]

    def test_thin_samples_are_shown_hollow_rather_than_dropped(self):
        """A slate view that silently drops everyone with few at-bats against the mix is
        hiding the most common case, not cleaning it up."""
        spec = charts.arsenal_vs_allowed_scatter(_matchups()).to_dict()
        fill = spec["layer"][-1]["encoding"]["fill"]
        assert "transparent" in fill["scale"]["range"]

    def test_every_baseline_builds(self):
        for measure in ("Arsenal OPS", "Platoon OPS", "Season OPS"):
            assert charts.arsenal_vs_allowed_scatter(_matchups(), measure).to_dict()

    def test_the_edge_ranking_marks_zero_and_respects_the_sample_floor(self):
        chart = charts.edge_bars(_matchups(), min_ab=20)
        spec = chart.to_dict()
        assert any(layer["mark"]["type"] == "rule" for layer in spec["layer"])
        assert charts.edge_bars(_matchups(), min_ab=999) is None

    def test_a_frame_without_the_baseline_declines_rather_than_crashing(self):
        assert charts.arsenal_vs_allowed_scatter(
            _matchups().drop(columns=["Allowed OPS"])) is None


class TestEffectiveSide:
    """Which side a hitter bats from decides which of a pitcher's split rows applies.

    On a typical slate ten percent of the board switch-hits, and `Bats == "S"` matches
    neither row. Getting this wrong does not raise — it hands a hitter the opposing split,
    which reads exactly as plausible as the right one.
    """

    def test_a_fixed_hitter_keeps_his_own_side(self):
        from dashboards.data import effective_side
        assert effective_side("L", "R") == "L"
        assert effective_side("R", "L") == "R"
        assert effective_side("R", "R") == "R"

    def test_a_switch_hitter_bats_opposite_the_arm(self):
        from dashboards.data import effective_side
        assert effective_side("S", "R") == "L"
        assert effective_side("S", "L") == "R"

    def test_an_unknown_arm_leaves_a_switch_hitter_unresolved(self):
        """Half of them would get the wrong split, and a wrong split is invisible."""
        from dashboards.data import effective_side
        assert effective_side("S", "") == ""
        assert effective_side("S", None) == ""

    def test_it_tolerates_the_shapes_a_cached_frame_actually_holds(self):
        from dashboards.data import effective_side
        assert effective_side(" s ", " r ") == "L"
        assert effective_side("l", "R") == "L"
        assert effective_side(None, "R") == "L"


class TestDoubleheaderHandLookup:
    """Both games of a doubleheader share a club and a date.

    The pitcher's throwing hand is what picks a hitter's split row, so keying that lookup by
    club would hand one of the two games the *other* game's hand — and every hitter in it the
    wrong split, silently. Cached case: on 2026-07-29 Atlanta faced Sean Manaea (L) in one
    game and Christian Scott (R) in the other.
    """

    def test_the_lookup_key_is_the_pitcher_not_the_club(self):
        import inspect
        from dashboards import data
        source = inspect.getsource(data.hitters_vs_allowed)
        assert "name_key(pitcher)" in source
        assert '(meta["date"], team)' not in source

    def test_two_arms_on_one_date_keep_separate_hands(self):
        from dashboards import salaries
        hands = {}
        for name, hand in [("Sean Manaea", "L"), ("Christian Scott", "R")]:
            hands.setdefault(salaries.name_key(name), hand)
        assert hands[salaries.name_key("Sean Manaea")] == "L"
        assert hands[salaries.name_key("Christian Scott")] == "R"


def _vs_lineup():
    return pd.DataFrame({
        "pitcher": ["Lefty A", "Righty B", "Lefty C"],
        "throws": ["L", "R", "L"],
        "team": ["WSH", "TEX", "ATH"], "faces": ["TEX", "WSH", "KC"],
        "game": ["WSH @ TEX", "WSH @ TEX", "ATH @ KC"],
        "Allowed OPS": [0.644, 0.673, 0.700],
        "Lineup OPS": [0.723, 0.640, 0.760],
        "Allowed vs L": [0.654, 0.636, 0.820],
        "Allowed vs R": [0.639, 0.721, 0.640],
        "Lineup vs L": [0.709, 0.660, 0.780],
        "Lineup vs R": [0.732, 0.630, 0.750],
        "Bats vs L": [3, 6, 2], "Bats vs R": [6, 3, 7],
        "L bats": [3, 6, 2], "R bats": [6, 3, 7], "switch bats": [0, 0, 0],
        "lineup_bats": [9, 9, 9],
        "Allowed PA vs L": [97, 296, 120], "Allowed PA vs R": [246, 211, 300],
    })


class TestPitcherVsLineup:
    """Two OPS figures only mean something on one scale, and the platoon rows have teeth."""

    def test_both_axes_share_a_domain_so_the_diagonal_is_a_rule(self):
        spec = charts.pitcher_vs_lineup_scatter(_vs_lineup()).to_dict()
        points = spec["layer"][-1]["encoding"]
        assert points["x"]["scale"]["domain"] == points["y"]["scale"]["domain"]
        assert spec["layer"][0]["mark"]["type"] == "line"

    def test_every_split_builds(self):
        for view in charts.VS_LINEUP_VIEWS:
            assert charts.pitcher_vs_lineup_scatter(_vs_lineup(), view).to_dict()

    def test_size_is_the_bats_the_split_applies_to(self):
        """A favourable platoon number over two bats is not a matchup, and the only way to
        see that on the chart is for the point to be small."""
        spec = charts.pitcher_vs_lineup_scatter(_vs_lineup(),
                                                "vs left-handed bats").to_dict()
        assert spec["layer"][-1]["encoding"]["size"]["field"] == "Bats vs L"

    def test_a_split_with_no_bats_is_dropped_rather_than_drawn_at_zero(self):
        frame = _vs_lineup()
        frame["Bats vs L"] = 0
        assert charts.pitcher_vs_lineup_scatter(frame, "vs left-handed bats") is None

    def test_exposure_measures_the_side_he_handles_worse(self):
        """Worked through row by row, because the sign is the entire message:

        * Lefty A allows .654 to L and .639 to R, so **L** is his weaker side. He draws 3 L
          against 6 R — sheltered, so -3.
        * Righty B allows .636 to L and .721 to R, so **R** is weaker. He draws 3 R against
          6 L — sheltered, so -3.
        * Lefty C allows .820 to L and .640 to R, so **L** is weaker, and badly. He draws
          only 2 L against 7 R — the most sheltered of the three, so -5.

        All three cards lean away from the weaker split, which is what managers do; a
        positive bar is the case worth hunting for.
        """
        chart = charts.platoon_exposure_bars(_vs_lineup())
        assert chart is not None and chart.to_dict()
        frame = _vs_lineup()
        worse_left = frame["Allowed vs L"] > frame["Allowed vs R"]
        exposed = np.where(worse_left, frame["Bats vs L"], frame["Bats vs R"])
        sheltered = np.where(worse_left, frame["Bats vs R"], frame["Bats vs L"])
        assert (exposed - sheltered).tolist() == [-3, -3, -5]

    def test_a_card_stacked_against_the_weaker_split_reads_positive(self):
        """The inverse case, which is the one a reader is looking for."""
        frame = _vs_lineup()
        frame.loc[0, "Bats vs L"], frame.loc[0, "Bats vs R"] = 7, 2   # weaker side is L
        worse_left = frame["Allowed vs L"] > frame["Allowed vs R"]
        exposed = np.where(worse_left, frame["Bats vs L"], frame["Bats vs R"])
        sheltered = np.where(worse_left, frame["Bats vs R"], frame["Bats vs L"])
        assert (exposed - sheltered)[0] == 5

    def test_the_exposure_chart_marks_zero(self):
        spec = charts.platoon_exposure_bars(_vs_lineup()).to_dict()
        assert any(layer["mark"]["type"] == "rule" for layer in spec["layer"])

    def test_empty_input_degrades_to_none(self):
        empty = _vs_lineup().iloc[0:0]
        assert charts.pitcher_vs_lineup_scatter(empty) is None
        assert charts.platoon_exposure_bars(empty) is None


class TestTeamSignalEncoding:
    def test_colour_is_the_club_and_shape_is_the_signal(self):
        board = _board()
        spec = charts.team_signal_scatter(board, "Composite", "Season OPS").to_dict()
        assert spec["encoding"]["color"]["field"] == "Team"
        assert spec["encoding"]["shape"]["field"] == "Signal"

    def test_filtering_does_not_repaint_the_survivors(self):
        """Colour follows the club, never its rank. A domain taken from the filtered frame
        would hand two remaining clubs somebody else's hues."""
        board = _board()
        domain = charts.team_domain(board)
        narrowed = board[board["Team"] == "ATH"]
        spec = charts.team_signal_scatter(narrowed, "Composite", "Season OPS",
                                          domain=domain).to_dict()
        assert spec["encoding"]["color"]["scale"]["domain"] == domain
        assert len(domain) == 3

    def test_the_shape_scale_covers_every_signal_in_order(self):
        assert list(charts.SIGNAL_SHAPES) == charts.SIGNAL_ORDER

    def test_the_identity_cap_is_stated_rather_than_assumed(self):
        """Above it colour can only carry clustering, and the page says so instead of
        implying thirty hues are separable."""
        assert charts.TEAM_IDENTITY_CAP >= 3


class TestOnSlate:
    """Cached games are not the DraftKings slate, and the gap is not small.

    The dashboard reads every game the report ran; DK puts up a subset. Measured on the
    cache: 2026-08-21 had 13 cached games but Atlanta and Milwaukee on no slate at all (18
    hitters), and 2026-08-19 had four such clubs (37 hitters). Every one of them was being
    ranked, coloured, and counted into the slate analysis while being impossible to roster.
    """

    def _files(self, tmp_path, date, rows):
        for slate, teams in rows.items():
            frame = pd.DataFrame({
                "Name": [f"P{i}" for i in range(len(teams))],
                "TeamAbbrev": teams,
                "Salary": [3000] * len(teams),
                "Roster Position": ["OF"] * len(teams),
                "AvgPointsPerGame": [7.0] * len(teams),
            })
            frame.to_csv(tmp_path / f"DKSalaries_{date}_{slate}.csv", index=False)
        return str(tmp_path)

    def test_a_club_on_no_slate_is_dropped(self, tmp_path):
        salary_dir = self._files(tmp_path, "2026-08-21", {"main": ["ATH", "KC", "TEX"]})
        frame = pd.DataFrame({"Team": ["ATH", "KC", "TEX", "ATL", "MIL"],
                              "Name": list("abcde")})
        out = filters.on_slate(frame, "2026-08-21", salary_dir=salary_dir)
        assert sorted(out["Team"]) == ["ATH", "KC", "TEX"]

    def test_the_dropped_clubs_are_reportable(self, tmp_path):
        """A silent exclusion is its own bug — the page has to be able to name them."""
        salary_dir = self._files(tmp_path, "2026-08-21", {"main": ["ATH", "KC"]})
        frame = pd.DataFrame({"Team": ["ATH", "KC", "ATL", "MIL"], "Name": list("abcd")})
        assert filters.off_slate_teams(frame, "2026-08-21",
                                       salary_dir=salary_dir) == ["ATL", "MIL"]

    def test_membership_unions_every_slate_file(self, tmp_path):
        """A club on the late slate only is still rosterable that day."""
        salary_dir = self._files(tmp_path, "2026-08-21",
                                 {"main": ["ATH", "KC"], "late": ["SF", "LAD"]})
        assert filters.priced_teams("2026-08-21", salary_dir) == {"ATH", "KC", "SF", "LAD"}

    def test_no_salary_file_leaves_the_frame_alone(self, tmp_path):
        """With nothing to compare against, dropping everything is worse than a superset."""
        frame = pd.DataFrame({"Team": ["ATH", "KC"], "Name": ["a", "b"]})
        out = filters.on_slate(frame, "1999-01-01", salary_dir=str(tmp_path))
        assert len(out) == 2
        assert filters.off_slate_teams(frame, "1999-01-01", salary_dir=str(tmp_path)) == []

    def test_it_works_on_a_staff_frame_too(self, tmp_path):
        """Conditions and the vs-lineup panel key on `team`, not `Team`."""
        salary_dir = self._files(tmp_path, "2026-08-21", {"main": ["ATH"]})
        frame = pd.DataFrame({"team": ["ATH", "MIL"], "name": ["a", "b"]})
        out = filters.on_slate(frame, "2026-08-21", salary_dir=salary_dir,
                               team_column="team")
        assert list(out["team"]) == ["ATH"]


class TestTeamColours:
    def test_every_club_has_its_own_colour(self):
        assert len(charts.TEAM_COLORS) == 30

    def test_no_two_clubs_share_a_hex(self):
        """Detroit and the Yankees both wear #0C2340; Detroit takes its orange instead, or
        two clubs are indistinguishable on a chart whose whole point is identity."""
        assert len(set(charts.TEAM_COLORS.values())) == len(charts.TEAM_COLORS)

    def test_an_unmapped_club_is_grey_not_a_borrowed_identity(self):
        assert charts.team_colors(["XYZ"]) == [charts.UNKNOWN_TEAM_COLOR]

    def test_the_range_follows_the_domain_order(self):
        domain = ["TEX", "ATH", "NYY"]
        assert charts.team_colors(domain) == [charts.TEAM_COLORS[t] for t in domain]

    def test_the_axis_domain_and_the_team_domain_do_not_collide(self):
        """`arsenal_vs_allowed_scatter` binds `domain` locally to the shared axis bounds.

        A team parameter named `domain` was silently overwritten by that two-element list,
        so `_team_color` was handed axis numbers as club names and the chart claimed team
        colour under every setting. The parameter is `teams` for that reason.
        """
        import inspect
        signature = inspect.signature(charts.arsenal_vs_allowed_scatter)
        assert "teams" in signature.parameters
        assert "domain" not in signature.parameters
