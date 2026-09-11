import pandas as pd
import pytest

from dashboards import changes


def _board():
    return pd.DataFrame([
        {"DK ID": 1, "MLBAM": 101, "Name": "One", "Team": "ATH", "Type": "H",
         "Game": "ATH@KC", "Proj": 7.0, "Ceiling": 15.0, "Lineup": "Projected",
         "Slot": 2},
        {"DK ID": 2, "MLBAM": 102, "Name": "Two", "Team": "KC", "Type": "H",
         "Game": "ATH@KC", "Proj": 8.0, "Ceiling": 17.0, "Lineup": "Projected",
         "Slot": 4},
    ])


def test_snapshot_diff_tracks_numbers_and_lineup_details():
    before = _board()
    after = _board()
    after.loc[0, ["Proj", "Lineup", "Slot"]] = [9.25, "Confirmed", 1]
    diff = changes.compare(before, after)
    one = diff[diff["Name"] == "One"].iloc[0]
    assert one["Δ Proj"] == pytest.approx(2.25)
    assert one["Status"] == "Changed"
    assert "Lineup: Projected → Confirmed" in one["Details"]
    assert "Slot: 2 → 1" in one["Details"]
    assert changes.confirmed_between(diff) == 1


def test_snapshot_diff_keeps_additions_and_removals_visible():
    before = _board()
    after = _board().iloc[[0]].copy()
    after.loc[len(after)] = {"DK ID": 3, "MLBAM": 103, "Name": "Three", "Team": "KC",
                             "Type": "H", "Game": "ATH@KC", "Proj": 6.0,
                             "Ceiling": 14.0, "Lineup": "Confirmed", "Slot": 8}
    statuses = dict(zip(changes.compare(before, after)["Name"],
                        changes.compare(before, after)["Status"]))
    assert statuses["Two"] == "Removed"
    assert statuses["Three"] == "Added"


def test_contest_id_survives_a_name_correction():
    before = _board()
    after = _board()
    after.loc[0, "Name"] = "One Jr."
    diff = changes.compare(before, after)
    assert len(diff) == 2
    assert set(diff["Status"]) <= {"Changed", "Unchanged"}
