"""StreamFinder priority file generation.

Synthetic throughout: no cached payload, no DK export, no network. The upload files are
written to tmp_path in both DK layouts, because the layout is the part most likely to move.
"""

import json

import pandas as pd
import pytest

from dfs.streamfinder import (DEFAULTS, build_document, build_priority,
                              exposure_from_upload, _slate_index)

# Ten roster columns in DK's order. The entries layout puts four metadata columns first.
SLOTS = ["P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"]
ENTRY_META = ["Entry ID", "Contest Name", "Contest ID", "Entry Fee"]


def write_csv(path, rows):
    path.write_text("\n".join(",".join(str(c) for c in row) for row in rows),
                    encoding="utf-8")
    return str(path)


def entries_upload(tmp_path, lineups, name="upload.csv"):
    rows = [ENTRY_META + SLOTS]
    for n, lineup in enumerate(lineups, start=1):
        rows.append([f"52217664{n:02d}", "MLB $6K Solo Shot", "193891712", "$1", *lineup])
    return write_csv(tmp_path / name, rows)


def bulk_upload(tmp_path, lineups, name="bulk.csv"):
    return write_csv(tmp_path / name, [SLOTS] + [list(l) for l in lineups])


def slate_frame(rows):
    """rows: (dk_id, mlbam, name, type, proj)."""
    return pd.DataFrame(
        [{"DK ID": d, "MLBAM": m, "Name": n, "Type": t, "Proj": p} for d, m, n, t, p in rows])


class TestExposureFromUpload:
    def test_counts_every_appearance(self, tmp_path):
        path = entries_upload(tmp_path, [
            ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            ["1", "2", "3", "4", "5", "6", "7", "8", "9", "11"],
        ])
        counts = {dk: n for dk, n, _, _ in exposure_from_upload(path)}
        assert counts["1"] == 2 and counts["9"] == 2
        assert counts["10"] == 1 and counts["11"] == 1

    def test_is_ordered_most_used_first(self, tmp_path):
        path = entries_upload(tmp_path, [
            ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            ["1", "2", "3", "4", "5", "6", "7", "8", "9", "11"],
            ["1", "2", "3", "4", "5", "6", "7", "8", "9", "12"],
        ])
        counts = [n for _, n, _, _ in exposure_from_upload(path)]
        assert counts == sorted(counts, reverse=True)

    def test_the_first_two_columns_are_pitchers(self, tmp_path):
        ids = [str(100 + i) for i in range(10)]
        path = entries_upload(tmp_path, [ids])
        roles = {dk: is_pit for dk, _, is_pit, _ in exposure_from_upload(path)}
        assert roles["100"] is True and roles["101"] is True
        assert not any(roles[i] for i in ids[2:])

    def test_the_bulk_layout_works_too(self, tmp_path):
        first = [str(100 + i) for i in range(10)]
        second = first[:9] + ["999"]
        path = bulk_upload(tmp_path, [first, second])
        counts = {dk: n for dk, n, _, _ in exposure_from_upload(path)}
        assert counts["100"] == 2 and counts["999"] == 1

    def test_a_non_numeric_cell_is_not_taken_for_an_id(self, tmp_path):
        """DK ids are numeric. A bare name in a roster cell means the file was hand-edited
        or exported wrong, and inventing an id from it would put a garbage entry in the
        priority list rather than surfacing the problem."""
        path = entries_upload(tmp_path, [["Blake Snell"] + [str(100 + i) for i in range(9)]])
        found = {dk for dk, _, _, _ in exposure_from_upload(path)}
        assert "Blake Snell" not in found
        assert len(found) == 9

    def test_name_with_id_cells_are_parsed(self, tmp_path):
        cells = [f"Player {c} ({i})" for i, c in enumerate("ABCDEFGHIJ", start=100)]
        path = entries_upload(tmp_path, [cells])
        found = {dk: display for dk, _, _, display in exposure_from_upload(path)}
        assert found["100"] == "Player A"
        assert found["109"] == "Player J"

    def test_blank_and_trailing_rows_are_skipped(self, tmp_path):
        rows = [ENTRY_META + SLOTS,
                ["1", "c", "2", "$1", *[str(100 + i) for i in range(10)]],
                [],
                ["", "", "", "", *[""] * 10],
                ["Instructions: do not edit the header row"]]
        path = write_csv(tmp_path / "messy.csv", rows)
        exposure = exposure_from_upload(path)
        assert len(exposure) == 10
        assert all(n == 1 for _, n, _, _ in exposure)

    def test_a_file_without_the_slot_run_is_refused(self, tmp_path):
        path = write_csv(tmp_path / "not_an_upload.csv",
                         [["Name", "Team", "Salary"], ["Someone", "LAD", "5000"]])
        with pytest.raises(ValueError, match="DK upload file"):
            exposure_from_upload(path)

    def test_an_empty_file_is_refused(self, tmp_path):
        path = write_csv(tmp_path / "empty.csv", [])
        with pytest.raises(ValueError, match="empty"):
            exposure_from_upload(path)


class TestSlateIndex:
    def test_float_round_tripped_ids_still_match(self):
        """pandas reads an all-numeric id column as float, so the slate can carry
        '43854270.0' for a cell the upload spells '43854270'."""
        index = _slate_index(slate_frame([("43854270.0", 605483, "Blake Snell", "P", 19.4)]))
        assert "43854270" in index

    def test_players_without_an_mlbam_id_are_left_out(self):
        index = _slate_index(slate_frame([("1", float("nan"), "Nobody", "H", 5.0)]))
        assert index == {}

    def test_an_empty_frame_is_handled(self):
        assert _slate_index(pd.DataFrame()) == {}


class TestPriority:
    def build(self, exposure, rows, top=None):
        return build_priority(exposure, _slate_index(slate_frame(rows)), top)

    def test_exposure_orders_the_list(self):
        exposure = [("1", 9, True, ""), ("2", 4, False, ""), ("3", 6, False, "")]
        rows = [("1", 111, "Ace", "P", 20.0), ("2", 222, "Bat", "H", 9.0),
                ("3", 333, "Cat", "H", 8.0)]
        entries, resolved, _ = self.build(exposure, rows)
        assert [e["data"] for e in entries] == ["111", "333", "222"]
        assert [e["priority"] for e in entries] == [1, 2, 3]
        assert [r["count"] for r in resolved] == [9, 6, 4]

    def test_ties_break_on_projection(self):
        exposure = [("1", 5, False, ""), ("2", 5, False, "")]
        rows = [("1", 111, "Low", "H", 6.0), ("2", 222, "High", "H", 14.0)]
        entries, _, _ = self.build(exposure, rows)
        assert [e["data"] for e in entries] == ["222", "111"]

    def test_the_role_comes_from_the_roster_column(self):
        exposure = [("1", 3, True, "")]
        entries, _, _ = self.build(exposure, [("1", 111, "Two Way", "H", 12.0)])
        assert entries[0]["type"] == "pit"

    def test_top_truncates_after_sorting_not_before(self):
        exposure = [("1", 1, False, ""), ("2", 9, False, "")]
        rows = [("1", 111, "Rare", "H", 5.0), ("2", 222, "Chalk", "H", 5.0)]
        entries, _, _ = self.build(exposure, rows, top=1)
        assert len(entries) == 1 and entries[0]["data"] == "222"

    def test_unmatched_players_are_reported_not_dropped_silently(self):
        exposure = [("1", 5, False, "Known"), ("999", 4, False, "Stranger")]
        entries, resolved, unresolved = self.build(
            exposure, [("1", 111, "Known", "H", 7.0)])
        assert len(entries) == 1
        assert unresolved == [("999", "Stranger", 4)]

    def test_a_missing_projection_does_not_crash_the_sort(self):
        exposure = [("1", 5, False, ""), ("2", 5, False, "")]
        rows = [("1", 111, "NoProj", "H", None), ("2", 222, "HasProj", "H", 3.0)]
        entries, _, _ = self.build(exposure, rows)
        assert len(entries) == 2


class TestDocument:
    def test_it_matches_the_template_shape(self):
        document = build_document([{"type": "bat", "data": "1", "immediate": "", "priority": 1}],
                                  ignore=[116])
        assert list(document) == ["on_deck", "include_CLI", "delay", "ignore", "priority"]
        assert list(document["priority"][0]) == ["type", "data", "immediate", "priority"]

    def test_the_numeric_looking_fields_stay_strings(self):
        """StreamFinder writes them quoted; a real int in `delay` is not read back the
        same way."""
        document = build_document([], ignore=[116, "113"])
        assert isinstance(document["delay"], str)
        assert document["ignore"] == ["116", "113"]

    def test_defaults_match_a_file_the_app_wrote(self):
        assert DEFAULTS == {"on_deck": "N", "include_CLI": "Y", "delay": "5000"}

    def test_overrides_apply(self):
        document = build_document([], ignore=[], on_deck="Y", delay="2500")
        assert document["on_deck"] == "Y" and document["delay"] == "2500"

    def test_none_overrides_are_ignored(self):
        """The CLI passes every flag through whether or not it was given."""
        document = build_document([], ignore=[], on_deck=None, delay=None)
        assert document["on_deck"] == "N" and document["delay"] == "5000"

    def test_it_round_trips_as_json(self):
        document = build_document(
            [{"type": "pit", "data": "605483", "immediate": "", "priority": 1}], ignore=[116])
        assert json.loads(json.dumps(document)) == document


class TestEndToEnd:
    def test_an_upload_and_a_slate_produce_a_usable_file(self, tmp_path):
        path = entries_upload(tmp_path, [
            ["10", "11", "12", "13", "14", "15", "16", "17", "18", "19"],
            ["10", "11", "12", "13", "14", "15", "16", "17", "18", "20"],
        ])
        rows = [(str(i), 600000 + i, f"Player{i}", "P" if i < 12 else "H", 20.0 - i * 0.1)
                for i in range(10, 21)]
        entries, resolved, unresolved = build_priority(
            exposure_from_upload(path), _slate_index(slate_frame(rows)))
        document = build_document(entries, ignore=[116])
        assert not unresolved
        assert len(document["priority"]) == 11
        # The nine shared players outrank the two that appear once.
        assert [r["count"] for r in resolved][:9] == [2] * 9
        assert [r["count"] for r in resolved][9:] == [1, 1]
        assert [e["type"] for e in document["priority"][:2]] == ["pit", "pit"]
