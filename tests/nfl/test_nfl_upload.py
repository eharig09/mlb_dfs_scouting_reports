"""The DK upload writer: cell format, slot ordering, and the guards that refuse a bad file."""

import csv

import pandas as pd
import pytest

from nfl import optimizer as opt
from nfl import upload


def _lineup(slots=None, team="BUF"):
    slots = slots or list(upload.SLOT_ORDER)
    return pd.DataFrame([{"Lineup": 1, "Slot": slot, "Name": f"P{i}", "Pos": slot,
                          "Team": team, "DK ID": 1000 + i}
                         for i, slot in enumerate(slots)])


class TestRosterAgreesWithTheOptimizer:
    def test_the_upload_columns_are_the_roster_the_solver_fills(self):
        """Two independent spellings of the same roster. If they drift, every upload is
        silently written into the wrong columns."""
        from collections import Counter
        assert Counter(upload.SLOT_ORDER) == Counter(opt.SEATS)
        assert len(upload.SLOT_ORDER) == opt.ROSTER_SIZE


class TestParseCell:
    def test_the_dk_format_is_name_then_id(self):
        assert upload.parse_cell("Josh Allen (43837774)") == ("43837774", "Josh Allen")

    def test_a_bare_id_is_accepted(self):
        assert upload.parse_cell("43837774") == ("43837774", "")

    def test_a_float_round_tripped_id_is_repaired(self):
        """pandas turns an integer id column into floats given half a chance."""
        assert upload.parse_cell("43837774.0") == ("43837774", "")

    def test_a_name_with_no_id_is_not_an_entry(self):
        """DK silently ignores a cell holding only a name, so it must not read as resolved."""
        assert upload.parse_cell("Josh Allen") == ("", "Josh Allen")

    def test_blank_is_blank(self):
        assert upload.parse_cell("") == ("", "")
        assert upload.parse_cell(None) == ("", "")


class TestSlotStart:
    def test_it_finds_the_nfl_roster_run(self):
        header = ["Entry ID", "Contest Name", "Contest ID", "Entry Fee"] + list(upload.SLOT_ORDER)
        assert upload._slot_start(header) == 4

    def test_a_baseball_template_is_not_an_nfl_one(self):
        """The MLB and NFL templates live in different folders for this reason, but a file
        put in the wrong one must still be refused rather than half-filled."""
        header = ["Entry ID", "Contest Name", "P", "P", "C", "1B", "2B", "3B", "SS", "OF"]
        assert upload._slot_start(header) is None


class TestParseSelection:
    def test_ranges_and_singles(self):
        assert upload.parse_selection("1,3,5-8", range(1, 11)) == [1, 3, 5, 6, 7, 8]

    def test_blank_or_all_takes_everything(self):
        assert upload.parse_selection("", [1, 2]) == [1, 2]
        assert upload.parse_selection("all", [1, 2]) == [1, 2]

    def test_duplicates_collapse(self):
        assert upload.parse_selection("2,2,3", [1, 2, 3]) == [2, 3]

    def test_a_lineup_that_does_not_exist_is_refused(self):
        with pytest.raises(upload.UploadError):
            upload.parse_selection("9", [1, 2])

    def test_junk_is_refused(self):
        with pytest.raises(upload.UploadError):
            upload.parse_selection("x", [1, 2])


class TestPlayerIndex:
    def _rows(self, players):
        header = [""] * 4 + list(upload.LIST_COLUMNS)
        rows = [["Entry ID"] + [""] * 12, [], header]
        for position, name, player_id, team in players:
            rows.append([""] * 4 + [position, f"{name} ({player_id})", name, player_id,
                                    position, "5000", "", team, "10"])
        return rows

    def test_it_reads_name_and_team_to_id(self):
        by_pair, by_name = upload.player_index(
            self._rows([("QB", "Josh Allen", "111", "BUF")]))
        assert by_pair[("josh allen", "BUF")] == "111"
        assert by_name["josh allen"] == "111"

    def test_an_ambiguous_name_is_dropped_from_the_name_only_map(self):
        """Two players sharing a name must be matched on team or not at all -- writing the
        wrong id is worse than refusing to write one."""
        by_pair, by_name = upload.player_index(self._rows([
            ("WR", "Mike Williams", "111", "NYJ"),
            ("WR", "Mike Williams", "222", "LAC")]))
        assert by_pair[("mike williams", "NYJ")] == "111"
        assert by_pair[("mike williams", "LAC")] == "222"
        assert "mike williams" not in by_name


class TestOrderPlayers:
    def test_players_come_back_in_dk_column_order(self):
        ordered = upload.order_players(_lineup(), 1)
        assert [p.Slot for p in ordered] == list(upload.SLOT_ORDER)

    def test_a_lineup_shuffled_out_of_order_still_orders(self):
        shuffled = _lineup(list(reversed(upload.SLOT_ORDER)))
        ordered = upload.order_players(shuffled, 1)
        assert [p.Slot for p in ordered] == list(upload.SLOT_ORDER)

    def test_a_missing_slot_is_refused_with_what_it_had(self):
        broken = _lineup()
        broken.loc[broken.index[0], "Slot"] = "WR"
        with pytest.raises(upload.UploadError) as excinfo:
            upload.order_players(broken, 1)
        assert "QB" in str(excinfo.value)

    def test_an_extra_player_is_refused(self):
        over = pd.concat([_lineup(), _lineup(["WR"])], ignore_index=True)
        with pytest.raises(upload.UploadError):
            upload.order_players(over, 1)


class TestTemplateDates:
    def test_it_reads_the_slate_date_out_of_game_info(self):
        header = [""] * 4 + list(upload.LIST_COLUMNS)
        rows = [["Entry ID"], [], header,
                [""] * 4 + ["QB", "A (1)", "A", "1", "QB", "5000",
                            "LV@HOU 08/20/2026 08:00PM ET", "HOU", "10"]]
        assert upload.template_dates(rows) == {"2026-08-20"}


class TestBuildUpload:
    def _template(self):
        header = (["Entry ID", "Contest Name", "Contest ID", "Entry Fee"]
                  + list(upload.SLOT_ORDER) + ["", "Instructions"])
        rows = [header, ["9001", "NFL $10K", "77", "$5"] + [""] * len(upload.SLOT_ORDER)
                + ["", ""]]
        rows += [[], [""] * 4 + list(upload.LIST_COLUMNS)]
        for i, slot in enumerate(upload.SLOT_ORDER):
            rows.append([""] * 4 + [slot, f"P{i} ({1000 + i})", f"P{i}", str(1000 + i),
                                    slot, "5000", "LV@HOU 09/14/2026 01:00PM ET", "BUF", "1"])
        return rows

    def test_it_fills_the_roster_cells_in_place(self):
        rows = self._template()
        _, _, slot_start, index = (rows, "entries", 4, upload.player_index(rows))
        out, _ = upload.build_upload(rows, "entries", slot_start, index,
                                     {1: _lineup()}, [1], date="2026-09-14")
        cells = out[1][slot_start:slot_start + len(upload.SLOT_ORDER)]
        assert all(cell.endswith(")") for cell in cells)
        assert out[1][0] == "9001"           # the entry id is preserved

    def test_a_template_from_another_week_is_refused_not_warned_about(self):
        """Names resolve against whatever template is handed over, so a stale file produces
        a full sheet of valid-looking ids for a contest that has already finished."""
        rows = self._template()
        index = upload.player_index(rows)
        with pytest.raises(upload.UploadError) as excinfo:
            upload.build_upload(rows, "entries", 4, index, {1: _lineup()}, [1],
                                date="2026-10-05")
        assert "2026-09-14" in str(excinfo.value)

    def test_an_unresolvable_player_is_refused(self):
        rows = self._template()
        index = ({}, {})                      # nothing resolves
        lineup = _lineup()
        lineup["DK ID"] = ""
        with pytest.raises(upload.UploadError) as excinfo:
            upload.build_upload(rows, "entries", 4, index, {1: lineup}, [1])
        assert "no DK id" in str(excinfo.value)

    def test_more_lineups_than_entries_fills_what_it_can_and_says_so(self):
        rows = self._template()
        index = upload.player_index(rows)
        out, notes = upload.build_upload(rows, "entries", 4, index,
                                         {1: _lineup(), 2: _lineup()}, [1, 2],
                                         date="2026-09-14")
        assert notes and "only 1 entries" in notes[0]
