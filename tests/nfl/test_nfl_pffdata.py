"""The PFF data layer: season identification, the id crosswalk, and the guards.

These tests are deliberately offline. `identify_season` takes its roster fingerprints as an
argument for exactly this reason -- the method is what needs testing, not nflverse's uptime.
"""

import pandas as pd
import pytest

from nfl import pffdata
from nfl.salaries import canon_position


def _export(rows):
    """A minimal player-level PFF export."""
    return pd.DataFrame(rows, columns=["player", "player_id", "position", "team_name",
                                       "player_game_count"])


class TestFamilyParsing:
    def test_the_download_number_is_not_part_of_the_family(self):
        assert pffdata._family("Receiving Grades/receiving_summary (15).csv") == \
            "receiving_summary"
        assert pffdata._family("pff/receiving_summary.csv") == "receiving_summary"

    def test_a_hyphenated_family_survives(self):
        assert pffdata._family("pff/fantasy-stats-passing (3).csv") == "fantasy-stats-passing"


class TestSeasonIdentification:
    """The filenames carry no season, so it is read off the roster the file describes."""

    def _fingerprints(self):
        # Two seasons. Ash moved clubs between them, which is the entire signal: everyone
        # else is on the same team both years and therefore separates nothing.
        return {
            2024: ({(1, "BUF"), (2, "MIA"), (3, "BUF")}, {("ash", "BUF"), ("bo", "MIA")}),
            2025: ({(1, "BUF"), (2, "MIA"), (3, "NYJ")}, {("ash", "NYJ"), ("bo", "MIA")}),
        }

    def test_it_picks_the_season_whose_roster_matches(self):
        frame = _export([["A", 1, "WR", "BUF", 17],
                         ["B", 2, "WR", "MIA", 17],
                         ["C", 3, "WR", "NYJ", 17]])
        season, confidence, margin, _ = pffdata.identify_season(frame, self._fingerprints())
        assert season == 2025
        assert confidence == 1.0
        assert margin > pffdata.MIN_MARGIN

    def test_it_refuses_to_guess_when_the_seasons_are_not_separable(self):
        """A file of players who never moved matches every season equally.

        Returning the winner anyway would be a coin flip presented as a fact, and the whole
        point of identifying rather than assuming is not to do that.
        """
        frame = _export([["A", 1, "WR", "BUF", 17], ["B", 2, "WR", "MIA", 17]])
        season, confidence, margin, _ = pffdata.identify_season(frame, self._fingerprints())
        assert season is None
        assert confidence == 1.0          # it matched perfectly...
        assert margin == 0.0              # ...against both, which is the problem

    def test_it_falls_back_to_names_when_the_export_has_no_ids(self):
        """PFF's fantasy-stats endpoint publishes no player id at all."""
        frame = _export([["Ash", None, "WR", "NYJ", 17], ["Bo", None, "WR", "MIA", 17]])
        season, _, _, _ = pffdata.identify_season(frame, self._fingerprints())
        assert season == 2025

    def test_team_codes_are_folded_before_matching(self):
        """PFF writes ARZ, BLT, CLV, HST. Unfolded, those clubs match nothing."""
        prints = {2025: ({(1, "ARI")}, {("ash", "ARI")})}
        frame = _export([["Ash", 1, "WR", "ARZ", 17]])
        season, confidence, _, _ = pffdata.identify_season(frame, prints)
        assert season == 2025 and confidence == 1.0


class TestSchemaAliases:
    def test_the_fantasy_endpoint_columns_are_renamed_to_the_canonical_ones(self):
        frame = pd.DataFrame({"player": ["A"], "team": ["BUF"], "games": [17]})
        out = pffdata._canonical_columns(frame, "fantasy-stats-receiving")
        assert "team_name" in out.columns and "player_game_count" in out.columns

    def test_a_family_with_no_aliases_is_untouched(self):
        frame = pd.DataFrame({"player": ["A"], "team_name": ["BUF"]})
        assert pffdata._canonical_columns(frame, "receiving_summary") is frame


class TestPositionFolding:
    def test_pff_has_no_rb(self):
        """PFF writes HB. A filter on "RB" silently emptied a whole position league-wide."""
        assert canon_position("HB") == "RB"
        assert canon_position("FB") == "RB"

    def test_line_positions_keep_their_identity(self):
        """Where a lineman plays is the point of the line work, so T/G/C do not collapse."""
        assert canon_position("T") == "T"
        assert canon_position("G") == "G"
        assert canon_position("C") == "C"

    def test_an_unknown_label_passes_through(self):
        assert canon_position("LS") == "LS"
        assert canon_position(None) == ""


class TestCacheKeys:
    """The catalog is cached per path, so the path spelling has to be stable."""

    def test_equivalent_spellings_produce_one_key(self):
        """A caller passing `root="."` globs `./nfl/pff/x.csv` for the same file the default
        root finds as `nfl/pff/x.csv`. Unnormalised those are two keys, so every such call
        missed the cache and re-fingerprinted all ~90 exports -- minutes per lookup, and it
        hung the test suite until it was found.
        """
        assert pffdata._key("./nfl/pff/x.csv") == pffdata._key("nfl/pff/x.csv")
        assert pffdata._key("nfl\\pff\\x.csv") == pffdata._key("nfl/pff/x.csv")
        assert pffdata._key("nfl/pff/../pff/x.csv") == pffdata._key("nfl/pff/x.csv")


class TestLatest:
    """Projections have no season to identify -- a refresh supersedes its predecessor."""

    def test_it_returns_the_newest_projection_export(self):
        """`slate.py` pinned `projections (1).csv` by name, so a fresher pull landing beside
        it was never read. The two on disk differ on 374 of 533 players.
        """
        newest = pffdata.latest("projections")
        assert newest.endswith(".csv")
        others = [p for p in pffdata.catalog()
                  .query("family == 'projections' and usable").path if p != newest]
        import os
        for other in others:
            assert os.stat(newest).st_mtime >= os.stat(other).st_mtime

    def test_an_absent_family_fails_loudly(self):
        with pytest.raises(pffdata.PffDataError):
            pffdata.latest("no_such_projection_family")


class TestResolveFailsLoudly:
    def test_an_unknown_family_names_what_is_available(self):
        with pytest.raises(pffdata.PffDataError) as excinfo:
            pffdata.resolve("no_such_report", 2025)
        assert "no_such_report" in str(excinfo.value)

    def test_a_missing_season_says_which_seasons_exist(self):
        with pytest.raises(pffdata.PffDataError) as excinfo:
            pffdata.resolve("receiving_summary", 1999)
        assert "identified" in str(excinfo.value)
