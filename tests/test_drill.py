"""Drill-down: showing a number's provenance.

The risk here is not a crash. It is a dialog that renders confidently with the *wrong club's*
evidence, or a table that silently disappears because Arrow refused it. Both look fine from
the outside, so both are pinned.
"""

import pandas as pd
import pytest

from dashboards import data, drill


def _payload():
    """A payload shaped like the pipeline's, including the parts that trip callers up:
    tuple-wrapped context entries and workbook blanks inside numeric columns."""
    return {
        "advanced_context": {
            "hitter_composite": pd.DataFrame({
                "Name": ["CJ Abrams", "José Ramírez"], "Team": ["WSH", "TEX"],
                "Composite": [30.0, 12.0], "Season OPS": [0.780, 0.810]}),
            "away_batter_arsenal": pd.DataFrame({
                "Name": ["CJ Abrams"], "OPS": [0.929], "Fit": ["Attack"]}),
            "home_batter_arsenal": pd.DataFrame({
                "Name": ["José Ramírez"], "OPS": [0.700], "Fit": ["Neutral"]}),
            "away_bvp": pd.DataFrame({"Name": ["CJ Abrams"], "PA": [3], "OPS": [0.0]}),
            "away_relevant_games": pd.DataFrame({
                "Date": ["08-19", "08-18"], "Opp": ["@ TEX", "@ TEX"],
                "FIP": [4.47, ""], "Rest": ["1", ""]}),
            "home_relevant_games": pd.DataFrame({
                "Date": ["08-19"], "Opp": ["WSH"], "FIP": [3.2], "Rest": ["1"]}),
            "away_comparable_games": (["a note"],
                                      pd.DataFrame({"#": [1], "Date": ["2026-08-14"]}),
                                      "a caption"),
            "away_pitcher_hand_splits": pd.DataFrame({
                "Pitcher": ["Andrew Alvarez", "Andrew Alvarez"], "Hand": ["L", "R"]}),
            "home_pitcher_hand_splits": pd.DataFrame({
                "Pitcher": ["Jacob deGrom", "Jacob deGrom"], "Hand": ["L", "R"]}),
            "home_pitcher_similar": pd.DataFrame({
                "Name": ["Tim Herrin"], "Similarity": [70]}),
            "home_sp_rest_splits": pd.DataFrame({
                "Rest": ["5d", "6d+"], "GS": [2, 7], "ERA": [2.79, ""]}),
            "home_sp_profile": {
                "starts": [{"date": "2026-08-13", "opponent": "LAA", "pitches": 53},
                           {"date": "2026-08-08", "opponent": "BAL", "pitches": 88},
                           {"date": "2026-08-01", "opponent": "HOU", "pitches": 91}],
                "mix": pd.DataFrame({"Date": ["SEASON", "08-13", "08-08", "08-01"],
                                     "P": [2067, 53, 88, 91],
                                     "FF%": [41.9, 43.4, 48.9, 36.3]}),
                "batted": pd.DataFrame({"Date": ["SEASON", "08-13", "08-08", "08-01"],
                                        "GB%": [35.0, 25.0, 25.0, 42.9],
                                        "HH%": [25.5, 25.0, 25.0, 13.3]}),
            },
        }
    }


def _meta():
    return {"date": "2026-08-20", "away": "WSH", "home": "TEX", "game": 1,
            "label": "WSH @ TEX", "path": "x"}


class TestSides:
    def test_a_club_maps_to_its_own_half_of_the_payload(self):
        meta = _meta()
        assert drill.side_for_team(meta, "WSH") == "away"
        assert drill.side_for_team(meta, "TEX") == "home"

    def test_a_club_not_in_this_game_maps_to_nothing(self):
        """Better to show no evidence than one of the two clubs at random."""
        assert drill.side_for_team(_meta(), "LAD") is None
        assert drill.hitter_evidence(_payload(), _meta(), "Anyone", "LAD") == {}

    def test_a_hitter_is_shown_the_arm_he_faces_not_his_own(self):
        """The inversion is the whole point of the panel and the easy bug: a hitter's
        matchup evidence lives under the *opponent's* prefix. Getting it backwards shows
        the away lineup its own starter's splits, which reads entirely plausible."""
        payload, meta = _payload(), _meta()
        assert drill.opposing_starter(payload, meta, "WSH")[0] == "Jacob deGrom"
        assert drill.opposing_starter(payload, meta, "TEX")[0] == "Andrew Alvarez"

    def test_team_context_is_that_club_s_own_games(self):
        payload, meta = _payload(), _meta()
        away = drill.team_context(payload, meta, "WSH")["relevant games"]
        home = drill.team_context(payload, meta, "TEX")["relevant games"]
        assert len(away) == 2 and len(home) == 1


class TestEvidence:
    def test_a_tuple_wrapped_frame_is_still_found(self):
        """Several context entries are `(notes, frame, caption)`, and a plain `.get`
        returns the tuple — the panel then renders nothing with no error."""
        context = drill.team_context(_payload(), _meta(), "WSH")
        assert isinstance(context["comparable games"], pd.DataFrame)

    def test_names_fold_the_way_the_rest_of_the_app_folds_them(self):
        evidence = drill.hitter_evidence(_payload(), _meta(), "Jose Ramirez", "TEX")
        assert "Arsenal fit" in evidence

    def test_an_absent_section_is_omitted_rather_than_shown_empty(self):
        """'Head-to-head' with no rows reads as 'no history'; the truth is usually 'these
        two have never met', and those are different claims."""
        evidence = drill.hitter_evidence(_payload(), _meta(), "José Ramírez", "TEX")
        assert "Head-to-head" not in evidence
        assert "Head-to-head" in drill.hitter_evidence(_payload(), _meta(),
                                                       "CJ Abrams", "WSH")

    def test_a_player_in_no_frame_yields_nothing_rather_than_a_wrong_row(self):
        assert drill.hitter_evidence(_payload(), _meta(), "Nobody At All", "WSH") == {}


class TestStarterLog:
    def test_rest_is_the_gap_to_the_previous_start(self):
        log = drill.starter_game_log(_payload(), "home")
        assert list(log["Date"]) == ["2026-08-13", "2026-08-08", "2026-08-01"]
        assert list(log["Rest"].dropna()) == [5.0, 7.0]

    def test_the_oldest_start_gets_no_rest_rather_than_a_guess(self):
        """Its predecessor is outside the cached window. A zero or a filled-forward value
        would land in `rest_summary` as a real start on real rest."""
        log = drill.starter_game_log(_payload(), "home")
        assert pd.isna(log["Rest"].iloc[-1])

    def test_the_season_row_is_not_counted_as_a_game(self):
        """`mix` and `batted` both carry a SEASON total. Joined in, it becomes a fourth
        start, and every per-start mean is then wrong."""
        log = drill.starter_game_log(_payload(), "home")
        assert len(log) == 3
        assert "SEASON" not in log["Date"].astype(str).to_numpy()

    def test_the_pitch_and_batted_logs_join_onto_the_right_start(self):
        log = drill.starter_game_log(_payload(), "home").set_index("Date")
        assert log.loc["2026-08-13", "FF%"] == pytest.approx(43.4)
        assert log.loc["2026-08-01", "GB%"] == pytest.approx(42.9)

    def test_a_side_with_no_profile_is_empty_not_an_exception(self):
        assert drill.starter_game_log(_payload(), "away").empty
        assert drill.rest_summary(pd.DataFrame()).empty

    def test_rest_summary_groups_the_starts_it_has(self):
        summary = drill.rest_summary(drill.starter_game_log(_payload(), "home"))
        assert list(summary["Rest"]) == [5, 7]
        assert list(summary["Starts"]) == [1, 1]
        assert summary["Pitches"].tolist() == [53.0, 88.0]

    def test_the_report_s_own_rest_split_is_left_alone(self):
        """The workbook's `sp_rest_splits` is computed from a different source. It is shown
        beside this, not replaced by it — a disagreement between them is information."""
        sections = drill.pitcher_evidence(_payload(), _meta(), "home")
        assert list(sections["By days rest"]["Rest"]) == ["5d", "6d+"]


class TestArrowSafety:
    """A frame Arrow refuses does not render *at all* — no error in the page, just a gap."""

    def test_a_numeric_column_with_workbook_blanks_becomes_nullable_numeric(self):
        frame = pd.DataFrame({"ERA": [2.79, "", 4.10]})
        out = data.arrow_safe(frame)
        assert out["ERA"].dtype.kind == "f"
        assert out["ERA"].tolist()[0] == pytest.approx(2.79)
        assert pd.isna(out["ERA"].iloc[1])

    def test_values_that_were_present_are_never_altered(self):
        frame = pd.DataFrame({"FIP": [4.47, "", 3.25], "K%": [20.6, 16.7, ""]},
                             dtype=object)
        out = data.arrow_safe(frame)
        assert out["FIP"].dropna().tolist() == [4.47, 3.25]
        assert out["K%"].dropna().tolist() == [20.6, 16.7]

    def test_an_all_string_column_is_left_alone_even_when_it_looks_numeric(self):
        """The contract is renderability, and a pure-string column already renders.

        Widening it to "anything that parses as a number" was measured against the cache
        and would have been actively harmful: the *only* fully-numeric string columns in
        2,126 frames are `bvp.Years`, whose normal value is `2022,2025`. Coercing it would
        turn a single-season row into a year-shaped number while its multi-season
        neighbours stayed text.
        """
        frame = pd.DataFrame({"Years": ["2022,2025", "", "2025"]})
        out = data.arrow_safe(frame)
        assert out["Years"].tolist() == ["2022,2025", "", "2025"]

    def test_a_mixed_column_keeps_every_value(self):
        frame = pd.DataFrame({"Note": ["12", "opener", ""]})
        out = data.arrow_safe(frame)
        assert out["Note"].tolist() == ["12", "opener", ""]

    def test_an_all_blank_column_is_not_silently_emptied(self):
        frame = pd.DataFrame({"Unused": ["", "", ""]})
        assert data.arrow_safe(frame)["Unused"].tolist() == ["", "", ""]

    def test_context_frame_applies_it_so_pages_do_not_have_to(self):
        out = data.context_frame(_payload(), "home_sp_rest_splits")
        assert out["ERA"].dtype.kind == "f"

    def test_every_rendered_frame_in_the_fixture_converts(self):
        pa = pytest.importorskip("pyarrow")
        payload, meta = _payload(), _meta()
        frames = list(drill.team_context(payload, meta, "WSH").values())
        frames += list(drill.pitcher_evidence(payload, meta, "home").values())
        frames.append(drill.starter_game_log(payload, "home"))
        for frame in frames:
            pa.Table.from_pandas(data.arrow_safe(frame))
