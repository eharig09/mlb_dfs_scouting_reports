"""File naming and slate identification.

The MLB twin's tests cover the same mechanism; what is tested here is the football
vocabulary and the one collision football has that baseball does not.
"""

import os

import pytest

from nfl import naming


SUNDAY, THURSDAY, MONDAY = 6, 3, 0


def _info(name, games, first, last, players=300, weekday=SUNDAY, matchups=None):
    return {"path": name, "name": name, "players": players, "weekday": weekday,
            "games": matchups or [f"A{i}@B{i}" for i in range(games)],
            "first": first, "last": last}


class TestSlugFromFilename:
    def test_a_human_added_label_survives_the_structure(self):
        assert naming.slug_from_filename("DKSalaries_2026-09-14_main.csv") == "main"

    def test_a_browser_duplicate_marker_is_not_a_label(self):
        """'DKSalaries (39).csv' carries no slate information at all."""
        assert naming.slug_from_filename("DKSalaries (39).csv") == ""
        assert naming.slug_from_filename("DKSalaries.csv") == ""


class TestSlugFromContents:
    def test_a_single_game_is_a_showdown(self):
        info = _info("x.csv", 1, 20 * 60 + 20, 20 * 60 + 20, matchups=["KC@BUF"])
        assert naming.slug_from_contents(info) == "showdown-kcbuf"

    def test_weeknights_are_named_for_their_night(self):
        """Football spreads across the week; baseball does not. This is the whole
        difference between the two vocabularies."""
        assert naming.slug_from_contents(
            _info("x.csv", 2, 20 * 60, 20 * 60, weekday=THURSDAY)) == "thursday"
        assert naming.slug_from_contents(
            _info("x.csv", 2, 20 * 60, 20 * 60, weekday=MONDAY)) == "monday"

    def test_sunday_splits_by_kickoff_window(self):
        assert naming.slug_from_contents(_info("x.csv", 13, 13 * 60, 16 * 60 + 25)) == "main"
        assert naming.slug_from_contents(
            _info("x.csv", 4, 16 * 60 + 25, 16 * 60 + 25)) == "afternoon"
        assert naming.slug_from_contents(
            _info("x.csv", 2, 20 * 60 + 20, 20 * 60 + 20)) == "primetime"

    def test_nothing_to_go_on_is_admitted_rather_than_guessed(self):
        info = _info("x.csv", 3, None, None, weekday=None)
        assert naming.slug_from_contents(info) == naming.UNKNOWN_SLATE


class TestLabelSlates:
    """The collision football has that baseball does not."""

    def test_main_and_early_only_share_a_kickoff_and_are_separated_by_size(self):
        """Both open at 1:00PM ET, so no single-file rule can tell them apart.

        The early-only block is a strict subset of the main slate; the only thing
        distinguishing them is that one is a fragment of the other, which is visible only
        when the week's exports are labelled together.
        """
        main = _info("DKSalaries.csv", 13, 13 * 60, 16 * 60 + 25, players=400)
        early = _info("DKSalaries (1).csv", 9, 13 * 60, 13 * 60, players=280)
        labels = naming.label_slates([main, early])
        assert labels["DKSalaries.csv"] == "main"
        assert labels["DKSalaries (1).csv"] == "early"

    def test_a_block_starting_after_the_main_slate_is_later_not_a_fragment(self):
        """Being small is not what makes a slate a fragment. A block whose games all start
        after every main-slate game has kicked off is a separate, later slate."""
        main = _info("DKSalaries.csv", 13, 13 * 60, 16 * 60 + 25, players=400)
        late = _info("DKSalaries (2).csv", 4, 16 * 60 + 25, 16 * 60 + 25, players=130)
        labels = naming.label_slates([main, late])
        assert labels["DKSalaries (2).csv"] == "afternoon"

    def test_a_whole_sunday_resolves(self):
        exports = [
            _info("DKSalaries.csv", 13, 13 * 60, 16 * 60 + 25, players=400),
            _info("DKSalaries (1).csv", 9, 13 * 60, 13 * 60, players=280),
            _info("DKSalaries (2).csv", 4, 16 * 60 + 25, 16 * 60 + 25, players=130),
            _info("DKSalaries (3).csv", 1, 20 * 60 + 20, 20 * 60 + 20, players=60,
                  matchups=["KC@BUF"]),
        ]
        assert set(naming.label_slates(exports).values()) == {
            "main", "early", "afternoon", "showdown-kcbuf"}

    def test_a_named_export_is_never_relabelled(self):
        """Renaming a file is an explicit decision; this function only resolves guesses."""
        main = _info("DKSalaries.csv", 13, 13 * 60, 16 * 60 + 25, players=400)
        named = _info("DKSalaries_2026-09-14_flex.csv", 9, 13 * 60, 13 * 60, players=280)
        labels = naming.label_slates([main, named])
        assert labels["DKSalaries_2026-09-14_flex.csv"] == "flex"


class TestSlateFor:
    def test_an_explicit_request_wins(self):
        assert naming.slate_for(requested="Showdown") == "showdown"

    def test_a_request_naming_the_export_selects_it_rather_than_renaming_it(self):
        """`--slate DKSalaries_2026-09-14_main.csv` picks a file. Treating it as a label
        would file the week under 'dksalaries-2026-09-14-main-csv', a name that resolves
        back to nothing and strands every downstream output."""
        info = _info("DKSalaries_2026-09-14_main.csv", 13, 13 * 60, 16 * 60 + 25)
        assert naming.slate_for(info, requested="DKSalaries_2026-09-14_main.csv") == "main"


class TestPaths:
    def test_an_unknown_kind_fails_loudly(self):
        with pytest.raises(ValueError):
            naming.output_path("nonsense", "2026-09-14", "main")

    def test_every_kind_has_a_described_extension(self):
        for kind, (extension, description) in naming.KINDS.items():
            assert extension.startswith(".") and description

    def test_versioning_never_overwrites(self, tmp_path):
        root = str(tmp_path)
        first, note = naming.resolve("pool", "2026-09-14", "main", root)
        assert note is None
        open(first, "w").close()
        second, note = naming.resolve("pool", "2026-09-14", "main", root)
        assert second != first and second.endswith(".r2.csv")
        assert note and "r2" in note

    def test_overwrite_replaces_in_place(self, tmp_path):
        root = str(tmp_path)
        first, _ = naming.resolve("pool", "2026-09-14", "main", root)
        open(first, "w").close()
        again, note = naming.resolve("pool", "2026-09-14", "main", root, overwrite=True)
        assert again == first and note is None

    def test_latest_goes_by_mtime_not_by_version_number(self, tmp_path):
        """An --overwrite run rewrites the base name after .r2 exists. A reader trusting
        the highest version number would then hand back the older file."""
        import time
        root = str(tmp_path)
        base = naming.output_path("pool", "2026-09-14", "main", root)
        os.makedirs(os.path.dirname(base), exist_ok=True)
        open(base, "w").close()
        second = naming.next_version(base)
        open(second, "w").close()
        time.sleep(0.01)
        open(base, "w").close()                    # the overwrite run
        assert naming.latest("pool", "2026-09-14", "main", root) == base


class TestGameInfo:
    def test_it_parses_dks_format(self):
        game, kickoff = naming.parse_game_info("LV@HOU 08/20/2026 08:00PM ET")
        assert game == "LV@HOU"
        assert (kickoff.year, kickoff.month, kickoff.day) == (2026, 8, 20)
        assert (kickoff.hour, kickoff.minute) == (20, 0)

    def test_noon_and_midnight_do_not_wrap(self):
        _, noon = naming.parse_game_info("KC@BUF 09/14/2026 12:00PM ET")
        _, midnight = naming.parse_game_info("KC@BUF 09/14/2026 12:00AM ET")
        assert noon.hour == 12 and midnight.hour == 0

    def test_junk_is_rejected_rather_than_half_parsed(self):
        assert naming.parse_game_info("not a game") == (None, None)
        assert naming.parse_game_info(None) == (None, None)

    def test_a_one_character_code_is_not_a_team(self):
        """No NFL club has a one-letter code, so half-parsing 'A@B' into a real matchup
        would invent a game rather than admit the cell was not a Game Info string."""
        assert naming.parse_game_info("A@B 09/14/2026 01:00PM ET") == (None, None)
