"""Reliever availability and the pitching views.

The availability grading is the part with a claim behind it, so most of this file pins that
claim's consequences: back-to-back detection, the graded thresholds, and — the important
one — that combining this rule with the report's own can only ever tighten a board.

Measured basis (2025 regular season, Statcast, 281 relievers, 20,868 appearances):
next-day appearance is 26.0% on rest and 5.5% after back-to-back days.
"""

import numpy as np
import pandas as pd
import pytest

from dashboards import bullpen, charts, data


def _usage(rows, days=("08-19", "08-18", "08-17", "08-16", "08-15")):
    """A bullpen usage grid shaped like the report's, newest date first."""
    frame = pd.DataFrame([{"Name": name, "T": "R", **dict(zip(days, counts)),
                           "Avail": avail}
                          for name, counts, avail in rows])
    return frame.reindex(columns=["Name", "T", *days, "Avail"])


class TestPitchCounts:
    @pytest.mark.parametrize("value,expected", [
        ("21", 21), ("51+", 51), ("-", 0), ("—", 0), ("", 0), (None, 0), (np.nan, 0),
    ])
    def test_a_days_pitch_count_is_read_or_treated_as_no_appearance(self, value, expected):
        assert bullpen._pitches(value) == expected

    def test_date_columns_keep_the_files_order(self):
        """Newest first, as written. Sorting them as strings breaks across a month end —
        '09-01' would sort before '08-31'."""
        frame = _usage([("A", (1, 2, 3, 4, 5), "Available")],
                       days=("09-01", "08-31", "08-30", "08-29", "08-28"))
        assert bullpen.date_columns(frame) == ["09-01", "08-31", "08-30", "08-29", "08-28"]


class TestBackToBack:
    def test_two_consecutive_appearances_are_flagged(self):
        graded = bullpen.availability(_usage([("A", (10, 11, "-", "-", "-"), "Available")]))
        row = graded.iloc[0]
        assert row["b2b"] is np.True_ or row["b2b"] is True
        assert row["two_day"] == 21
        assert row["Status"] == "Doubtful"

    def test_a_gap_day_is_not_back_to_back(self):
        graded = bullpen.availability(_usage([("A", (10, "-", 11, "-", "-"), "Available")]))
        assert not graded.iloc[0]["b2b"]
        assert graded.iloc[0]["Status"] == "Available"

    def test_a_heavy_back_to_back_is_out_not_merely_doubtful(self):
        """35+ pitches across the two days: those arms appeared 1.4% of the time."""
        graded = bullpen.availability(_usage([("A", (20, 18, "-", "-", "-"), "Available")]))
        assert graded.iloc[0]["two_day"] == 38
        assert graded.iloc[0]["Status"] == "Out"

    def test_a_light_back_to_back_is_still_doubtful(self):
        graded = bullpen.availability(_usage([("A", (5, 6, "-", "-", "-"), "Available")]))
        assert graded.iloc[0]["Status"] == "Doubtful"


class TestCombiningGrades:
    """The final grade is the stricter of the measurement and the report's own."""

    def test_the_reports_taxed_survives_when_the_measurement_is_softer(self):
        """A 51-pitch outing two days ago is not back-to-back.

        The measurement alone grades him Monitor; the report calls him Taxed, and a
        51-pitch appearance is a multi-inning one. Taking only the measurement would have
        quietly loosened the board.
        """
        graded = bullpen.availability(_usage([("A", ("-", "51+", "-", "-", "-"), "Taxed")]))
        row = graded.iloc[0]
        assert not row["b2b"]
        assert row["measured"] == "Monitor"
        assert row["Status"] == "Doubtful"          # the report's Taxed, on this ladder

    def test_the_measurement_survives_when_the_report_is_softer(self):
        graded = bullpen.availability(_usage([("A", (10, 11, "-", "-", "-"), "Available")]))
        assert graded.iloc[0]["measured"] == "Doubtful"
        assert graded.iloc[0]["Status"] == "Doubtful"

    def test_combining_never_loosens(self):
        rows = [("A", (10, 11, "-", "-", "-"), "Available"),
                ("B", ("-", "51+", "-", "-", "-"), "Taxed"),
                ("C", ("-", "-", "-", "-", "-"), "Available"),
                ("D", (20, 18, "-", "-", "-"), "Monitor")]
        graded = bullpen.availability(_usage(rows))
        for _, row in graded.iterrows():
            report = bullpen.REPORT_TIER.get(str(row["Avail"]).strip(), "Available")
            assert (bullpen.SEVERITY[row["Status"]]
                    >= max(bullpen.SEVERITY[row["measured"]], bullpen.SEVERITY[report]))

    def test_disagreements_are_only_where_this_rule_is_harder(self):
        rows = [("Caught", (10, 11, "-", "-", "-"), "Available"),
                ("Agreed", ("-", "-", "-", "-", "-"), "Available")]
        graded = bullpen.availability(_usage(rows))
        assert list(bullpen.disagreements(graded)["Name"]) == ["Caught"]


class TestDegenerate:
    def test_an_empty_usage_frame_is_empty(self):
        assert bullpen.availability(pd.DataFrame()).empty
        assert bullpen.availability(None).empty

    def test_a_frame_with_no_date_columns_falls_back_to_the_report(self):
        frame = pd.DataFrame([{"Name": "A", "Avail": "Monitor"}])
        graded = bullpen.availability(frame)
        assert graded.iloc[0]["Status"] == "Monitor"

    def test_summarise_reports_every_tier_even_at_zero(self):
        graded = bullpen.availability(_usage([("A", (1, "-", "-", "-", "-"), "Available")]))
        assert set(bullpen.summarise(graded)) == set(bullpen.TIERS)
        assert bullpen.summarise(pd.DataFrame()) == {t: 0 for t in bullpen.TIERS}


class TestPitchingCharts:
    def _batted(self):
        return pd.DataFrame([
            {"Name": "A", "BIP": 150, "GB%": 55.0, "FB%": 18.0, "LD%": 27.0,
             "HH%": 30.0, "Brl%": 5.0, "Status": "Available"},
            {"Name": "B", "BIP": 120, "GB%": 38.0, "FB%": 33.0, "LD%": 29.0,
             "HH%": 40.0, "Brl%": 9.0, "Status": "Doubtful"},
        ])

    def _slate(self):
        return pd.DataFrame([
            {"pitcher": "A", "team": "ATH", "faces": "KC", "game": "ATH @ KC",
             "park": "Kauffman", "park_runs": 1.01, "hits_saved": -0.54,
             "fb_rate": 27.1, "gb_rate": 41.1, "fip": 3.80, "k_bb": 14.4,
             "opp_ops": 0.712, "score": 20.0},
            {"pitcher": "B", "team": "KC", "faces": "ATH", "game": "ATH @ KC",
             "park": "Kauffman", "park_runs": 1.01, "hits_saved": 0.12,
             "fb_rate": 18.8, "gb_rate": 55.0, "fip": 3.84, "k_bb": 5.1,
             "opp_ops": 0.701, "score": 12.0},
        ])

    def test_batted_ball_builds_and_carries_the_league_cross(self):
        spec = charts.batted_ball_scatter(self._batted()).to_dict()
        assert "layer" in spec and len(spec["layer"]) == 3   # two rules plus the points

    def test_availability_colour_uses_its_own_reserved_scale(self):
        spec = charts.batted_ball_scatter(self._batted()).to_dict()
        colour = spec["layer"][2]["encoding"]["color"]
        assert colour["scale"]["domain"] == charts.AVAIL_ORDER

    def test_batted_ball_without_a_status_column_still_builds(self):
        """Starters have no availability grade; the panel is shared with relievers."""
        frame = self._batted().drop(columns=["Status"])
        assert charts.batted_ball_scatter(frame) is not None

    def test_workload_and_park_and_slate_pitcher_all_build(self):
        graded = bullpen.availability(_usage([("A", (10, 11, "-", "-", "-"), "Available"),
                                              ("B", ("-", "-", 9, "-", "-"), "Available")]))
        assert charts.bullpen_workload_scatter(graded).to_dict()
        assert charts.park_defense_scatter(self._slate()).to_dict()
        assert charts.slate_pitcher_scatter(self._slate()).to_dict()

    def test_fip_is_drawn_so_better_arms_sit_higher(self):
        """FIP is lower-is-better; an un-reversed axis reads backwards beside every other
        chart in the app."""
        spec = charts.slate_pitcher_scatter(self._slate()).to_dict()
        assert spec["encoding"]["y"]["scale"]["reverse"] is True

    @pytest.mark.parametrize("builder", [
        charts.batted_ball_scatter, charts.bullpen_workload_scatter,
        charts.park_defense_scatter, charts.slate_pitcher_scatter,
    ])
    def test_empty_input_returns_none(self, builder):
        assert builder(pd.DataFrame()) is None


class TestStarters:
    def _payload(self):
        return {
            "report_args": (),
            "advanced_context": {
                "away_pitcher_hand_splits": pd.DataFrame([{"Pitcher": "Away Arm"}]),
                "home_pitcher_hand_splits": pd.DataFrame([{"Pitcher": "Home Arm"}]),
                "away_team_defense": pd.DataFrame([{"Hits Saved/G": -0.5, "Grade": "Poor"}]),
                "home_team_defense": pd.DataFrame([{"Hits Saved/G": 0.2, "Grade": "Plus"}]),
                "away_sp_profile": {"season_batted": {"GB%": 41.0, "FB%": 27.0, "BIP": 300}},
                "home_sp_profile": {"season_batted": {"GB%": 55.0, "FB%": 19.0, "BIP": 280}},
                "environment": {"park_name": "Kauffman", "park": {"Runs": 1.01, "HR": 0.88}},
                "pitcher_watchlist": pd.DataFrame([
                    {"Team": "ATH", "FIP": 3.80, "K-BB": 14.4, "Score": 20.0, "Why": "x"},
                    {"Team": "KC", "FIP": 3.84, "K-BB": 5.1, "Score": 12.0, "Why": "y"}]),
            },
        }

    def _meta(self):
        return {"date": "2026-08-20", "away": "ATH", "home": "KC", "game": 1,
                "label": "ATH @ KC", "path": "x"}

    def test_a_starter_faces_the_other_club_not_his_own(self):
        """The side naming is the easy thing to invert here.

        `home_pitcher_hand_splits` is the home *starter*, while `home_opp_pitching` is what
        the home club faces — so team and faces must always be opposites.
        """
        frame = data.starters(self._payload(), self._meta())
        pairs = set(zip(frame["team"], frame["faces"]))
        assert pairs == {("ATH", "KC"), ("KC", "ATH")}

    def test_the_defence_attached_is_the_one_behind_him(self):
        frame = data.starters(self._payload(), self._meta()).set_index("team")
        assert frame.loc["ATH", "hits_saved"] == pytest.approx(-0.5)
        assert frame.loc["KC", "hits_saved"] == pytest.approx(0.2)

    def test_batted_ball_and_park_come_through(self):
        frame = data.starters(self._payload(), self._meta()).set_index("team")
        assert frame.loc["KC", "gb_rate"] == 55.0
        assert frame.loc["ATH", "park_runs"] == 1.01

    def test_a_game_with_no_starter_named_yields_nothing(self):
        payload = self._payload()
        payload["advanced_context"]["away_pitcher_hand_splits"] = pd.DataFrame()
        payload["advanced_context"]["home_pitcher_hand_splits"] = pd.DataFrame()
        assert data.starters(payload, self._meta()).empty
