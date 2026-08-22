import pandas as pd

from dfs.position_strategy import (
    depth_tier_decisions,
    pair_comparison,
    paired_lineup_backtest,
    replacement_scarcity,
    tier_strategy_card,
    walkforward_tier_audit,
)
from dfs.salaries import normalize_name


def _dated_slate(slate, night="2026-08-01", actual=True):
    frame = slate.copy()
    frame["Date"] = night
    frame["Slate"] = "main"
    frame["PlayerKey"] = frame["Name"].map(normalize_name)
    frame["Actual"] = frame["Proj"] if actual else None
    frame["Source"] = "board"
    frame["OutcomeSource"] = "synthetic.csv" if actual else ""
    frame["BoardPath"] = "synthetic.csv"
    frame["SlatePlayers"] = len(frame)
    return frame


def test_replacement_scarcity_is_walk_forward(slate):
    frames = []
    for day in range(1, 7):
        frame = _dated_slate(slate, f"2026-08-{day:02d}")
        # Make the last day unusually shallow at SS without touching earlier history.
        if day == 6:
            mask = (frame["Type"] == "H") & frame["DK Pos"].astype(str).str.contains("SS")
            keep = frame[mask].sort_values("Proj", ascending=False).index[:1]
            frame.loc[mask & ~frame.index.isin(keep), "Proj"] -= 4
        frames.append(frame)

    result = replacement_scarcity(pd.concat(frames, ignore_index=True), min_history=4)
    last_ss = result[(result["Date"] == "2026-08-06") & (result["Position"] == "SS")].iloc[0]

    assert last_ss["Scarcity"] in {"One standout", "Weak/empty"}
    assert last_ss["Viable N"] == 1
    assert result[result["Date"] == "2026-08-01"]["Scarcity"].eq("Warmup").all()


def test_walkforward_tiers_never_use_target_date_results(slate):
    first = _dated_slate(slate, "2026-08-01")
    second = _dated_slate(slate, "2026-08-02")
    second["Actual"] = 999  # Must not enter its own training sample.
    frame = pd.concat([first, second], ignore_index=True)

    result = walkforward_tier_audit(frame, min_train_hitters=10, min_train_pitchers=2)

    assert set(result["Date"]) == {"2026-08-02"}
    assert result[result["Type"] == "H"]["Train N"].eq((first["Type"] == "H").sum()).all()
    assert result[result["Type"] == "P"]["Train N"].eq((first["Type"] == "P").sum()).all()


def test_paired_lineups_reoptimize_the_complete_roster(small_slate):
    frame = _dated_slate(small_slate)

    lineups = paired_lineup_backtest(frame, objectives=("proj",))
    comparison = pair_comparison(lineups)

    assert set(comparison["Position"]) == {"C", "1B", "2B", "3B", "SS", "OF"}
    assert len(lineups) == 12
    assert (comparison["Position Extra $"] > 0).all()
    # The complete lineup delta decomposes exactly into the target slot and other nine.
    total = comparison["Position Proj Lift"] + comparison["Other Proj Change"]
    assert (total.round(6) == comparison["Lineup Proj Lift"].round(6)).all()
    assert comparison["Complete Actual Pair"].all()
    assert comparison["Comparable Actual Pair"].all()


def test_tier_strategy_keeps_candidate_sets_instead_of_player_proxies():
    common = {
        "Date": "2026-08-08", "Slate": "early", "Position": "1B",
        "Objective": "ceiling", "Lineups": 5, "Scarcity": "Deep", "Viable N": 5,
        "Position Pool N": 7, "Historical Avg Pts": 7.0, "Historical Pts/$1K": 1.5,
        "Historical Elite%": 8.0, "Historical Solid+%": 30.0,
        "Avg Lineup Proj": 100.0, "Best Lineup Proj": 102.0,
        "Best Lineup Ceiling": 212.0, "Avg Complete Lineup Actual": 95.0,
        "Complete Lineups": 5, "Tier Actual Lineups": 5,
    }
    summary = pd.DataFrame([
        {**common, "Tier": 4, "Learned Price Band": "$3,900-$4,900",
         "Tier Candidate N": 2, "Tier Candidates": "A | B", "Exposure": "A 60% | B 40%",
         "Tier Pool Share%": 28.6, "Avg Tier Salary": 4400,
         "Avg Position Salary": 4400, "Avg Lineup Salary": 49800,
         "Avg Lineup Ceiling": 208.2, "Avg Tier Actual": 8.0},
        {**common, "Tier": 5, "Learned Price Band": "$5,000-$6,900",
         "Tier Candidate N": 3, "Tier Candidates": "Olson | Contreras | Rice",
         "Exposure": "Rice 40% | Olson 40% | Contreras 20%", "Tier Pool Share%": 42.9,
         "Avg Tier Salary": 5500, "Avg Position Salary": 5500,
         "Avg Lineup Salary": 50000, "Avg Lineup Ceiling": 208.5,
         "Avg Tier Actual": 9.0},
    ])

    card = tier_strategy_card(summary, "2026-08-08")
    decision = depth_tier_decisions(summary).iloc[0]

    premium = card[card["Tier"].eq(5)].iloc[0]
    assert premium["Candidates"] == "Olson | Contreras | Rice"
    assert premium["Decision"] == "Flexible leader"
    assert decision["Near-best Tiers"] == 2
    assert decision["Near-best Candidates"] == 5
    assert decision["Strategy Read"] == "Flexible across near-best tiers"
