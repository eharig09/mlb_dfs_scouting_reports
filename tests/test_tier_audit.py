import pandas as pd
from unittest.mock import patch

from dfs.tier_audit import entry_constructions, read_ranked_entries


LINEUP = (
    "P Pitcher One P Pitcher Two C Catcher One 1B First One 2B Second One "
    "3B Third One SS Short One OF Out One OF Out Two OF Out Three"
)


def test_rank_cohorts_include_ties_and_use_all_scored_entries():
    frame = pd.DataFrame({
        "Rank": [1, 1, 3, 4],
        "EntryId": [10, 11, 12, 13],
        "Points": [200, 200, 180, 170],
        "Lineup": [LINEUP] * 4,
    })

    with patch("dfs.tier_audit.read_result_frame", return_value=frame):
        entries = read_ranked_entries("contest.csv")

    assert entries.attrs["contest_entries"] == 4
    assert entries["Winner"].sum() == 2
    assert entries["Top 1%"].sum() == 2
    assert entries["Top 20%"].sum() == 2


def test_construction_premium_means_learned_top_tier_not_lineup_highest():
    rows = []
    slots = ["P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"]
    tiers = [2, 2, 5, 3, 3, 3, 3, 3, 3, 3]
    for index, (slot, tier) in enumerate(zip(slots, tiers)):
        rows.append({
            "Date": "2026-08-01", "Slate": "main", "Contest": "contest-1",
            "Entry Key": "1", "Entry Name": "winner", "Rank": 1, "Points": 200,
            "Top 20%": True, "Top 1%": True, "Winner": True,
            "Position": slot, "Type": "P" if slot == "P" else "H",
            "Tier": tier, "Max Type Tier": 2 if slot == "P" else 5,
            "Name": f"Player {index}",
        })

    entry = entry_constructions(pd.DataFrame(rows)).iloc[0]

    assert entry["Premium Position Pattern"] == "Px2+C"
    assert entry["Premium Pitcher Slots"] == 2
    assert entry["Premium Hitter Slots"] == 1
