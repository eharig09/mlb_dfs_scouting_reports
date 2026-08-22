"""Rebuilding the plate appearances behind an arsenal number.

This is the highest-risk code in the dashboard, because a wrong answer here is *evidence*.
A drill-down that lists the wrong at-bats under a published figure is worse than no
drill-down: it makes an unverified number look verified.

The reproduction took five separate corrections to reach, and each one produced a plausible
result that was wrong. Every one of them is pinned below, because none would have been
caught by a smoke test — they all rendered a full table of real at-bats.
"""

import pandas as pd
import pytest

from dashboards import atbats


def _arsenal():
    """A starter arsenal shaped like `report_args[4]` — with pitch shapes."""
    return pd.DataFrame({
        "Pitch": ["FF", "SL", "CH", "CU"],
        "Usage %": [46.0, 40.0, 14.0, 6.0],
        "release_speed": [97.3, 91.3, 89.8, 80.7],
        "release_spin_rate": [2383.0, 2559.0, 1930.0, 2527.0],
    })


def _matchup_frame():
    """The `*_arsenal_matchup` context frame — usage and results, but **no shapes**."""
    return pd.DataFrame({
        "Pitch": ["FF", "SL", "CH"],
        "Usage%": [46.0, 40.0, 14.0],
        "Pitches": [3570, 1719, 900],
        "xwOBA": [0.363, 0.298, 0.310],
    })


def _payload():
    card_home = pd.DataFrame({"Name": ["Home One", "Home Two"], "ID": [1, 2],
                              "Bats": ["R", "L"], "Spot": [1, 2], "ISO": [0.180, 0.140]})
    card_away = pd.DataFrame({"Name": ["Away One", "Away Two"], "ID": [11, 12],
                              "Bats": ["L", "S"], "Spot": [1, 2], "ISO": [0.200, 0.090]})
    args = [None] * 14
    args[0] = card_home
    args[7] = card_away
    args[4] = _arsenal()
    args[11] = _arsenal()
    return {"report_args": tuple(args),
            "advanced_context": {"home_arsenal_matchup": _matchup_frame(),
                                 "away_arsenal_matchup": _matchup_frame()}}


def _meta():
    return {"date": "2026-08-20", "away": "WSH", "home": "TEX", "game": 1,
            "label": "WSH @ TEX", "path": "x"}


class TestTheFiveCorrections:
    """Each of these produced a full, plausible, wrong table before it was fixed."""

    def test_the_arsenal_frame_used_is_the_one_carrying_pitch_shapes(self):
        """`*_arsenal_matchup` looks like the right input and is not.

        It has usage and results but no `release_speed`/`release_spin_rate`, so
        `_arsenal_shape_traits` yields traits with no bands, the velocity and spin mask
        never applies, and the filter degenerates to bare pitch types — about 2.7x too many
        at-bats, every one of them real, none of them the ones counted.
        """
        arsenal = atbats._starter_arsenal(_payload(), "home")
        assert arsenal is not None
        assert {"release_speed", "release_spin_rate"} <= set(arsenal.columns)
        assert "Usage %" in arsenal.columns

        matchup = _matchup_frame()
        assert not {"release_speed", "release_spin_rate"} & set(matchup.columns)

    def test_the_lineup_card_is_the_one_the_hitter_is_actually_on(self):
        """`report_args` holds both clubs' cards, and their order is not away-first.

        Resolving the id from one card while taking the lineup from another measured a
        Washington hitter against Texas's sample floor.
        """
        payload = _payload()
        card, row = atbats._card_for(payload, "Away Two")
        assert row["ID"] == 12
        assert set(atbats._card_ids(card)) == {11, 12}

        card, row = atbats._card_for(payload, "Home One")
        assert set(atbats._card_ids(card)) == {1, 2}

    def test_an_unknown_hitter_resolves_to_no_card_rather_than_the_first(self):
        card, row = atbats._card_for(_payload(), "Nobody At All")
        assert card is None and row is None

    def test_the_arsenal_slots_are_pinned(self):
        """Positional indices into `report_args` are fragile by nature, so the mapping is
        stated once and tested rather than repeated at each call site."""
        assert atbats.ARSENAL_SLOT == {"home": 4, "away": 11}

    def test_only_pa_ending_rows_count(self):
        """`load_statcast_detail` keeps rows where `events` is set. The arsenal line is
        built on completed plate appearances, not on every pitch seen."""
        assert atbats.PA_MARKER == "events"


class TestSummaries:
    def _at_bats(self):
        return pd.DataFrame({
            "game_date": ["2026-08-01", "2026-08-05", "2026-08-09", "2026-08-11"],
            "pitch_type": ["FF", "FF", "SL", "CH"],
            "events": ["single", "strikeout", "home_run", "field_out"],
            "release_speed": [97.1, 96.8, 91.0, 89.5],
            "launch_speed": [102.0, float("nan"), 108.0, 78.0],
            "estimated_woba_using_speedangle": [0.61, float("nan"), 1.85, 0.09],
        })

    def test_outcomes_are_counted_not_rescaled(self):
        table = atbats.outcome_summary(self._at_bats())
        assert table["Count"].sum() == 4
        assert table["Share %"].sum() == pytest.approx(100.0)

    def test_the_pitch_split_shows_which_pitch_carries_the_line(self):
        """The arsenal number is a weighted blend, so a reader has to be able to see
        whether one pitch is doing all the work."""
        split = atbats.by_pitch_type(self._at_bats()).set_index("pitch_type")
        assert split.loc["FF", "PA"] == 2
        assert split.loc["FF", "Hits"] == 1
        assert split.loc["FF", "K"] == 1
        assert split.loc["SL", "Hit %"] == pytest.approx(100.0)

    def test_a_home_run_counts_as_a_hit_and_a_strikeout_does_not(self):
        split = atbats.by_pitch_type(self._at_bats()).set_index("pitch_type")
        assert split.loc["SL", "Hits"] == 1
        assert split.loc["CH", "Hits"] == 0

    def test_empty_input_gives_empty_output_rather_than_raising(self):
        empty = pd.DataFrame()
        assert atbats.outcome_summary(empty).empty
        assert atbats.by_pitch_type(empty).empty
        assert atbats.outcome_summary(None).empty


class TestGuards:
    def test_a_club_not_in_the_game_yields_nothing(self):
        frame, context = atbats.arsenal_at_bats(_payload(), _meta(), "Away One", "LAD")
        assert frame.empty and context == {}

    def test_a_hitter_not_on_either_card_yields_nothing(self):
        frame, context = atbats.arsenal_at_bats(_payload(), _meta(), "Nobody", "WSH")
        assert frame.empty and context == {}

    def test_a_payload_with_no_arsenal_yields_nothing(self):
        payload = _payload()
        args = list(payload["report_args"])
        args[4] = None
        payload["report_args"] = tuple(args)
        frame, context = atbats.arsenal_at_bats(payload, _meta(), "Away One", "WSH")
        assert frame.empty and context == {}
