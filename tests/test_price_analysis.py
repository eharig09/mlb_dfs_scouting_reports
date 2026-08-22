import numpy as np
import pandas as pd

from dfs.price_analysis import (
    canonical_observations,
    change_summary,
    performance_origins,
    player_history,
    position_depth,
    position_payup,
    position_days,
    price_tiers,
)


def _observations(rows):
    defaults = {
        "Slate": "main", "Team": "AAA", "Type": "H", "DK Pos": "2B",
        "DK ID": np.nan, "Source": "board", "OutcomeSource": "result.csv",
        "BoardPath": "board.csv", "SlatePlayers": 100,
    }
    records = []
    for row in rows:
        item = defaults | row
        item["PlayerKey"] = item["Name"].lower()
        records.append(item)
    return pd.DataFrame(records)


def test_canonical_observations_does_not_double_count_overlapping_slates():
    frame = _observations([
        {"Date": "2026-08-01", "Name": "Same Player", "Salary": 5000,
         "Actual": 12, "Slate": "turbo", "SlatePlayers": 50},
        {"Date": "2026-08-01", "Name": "Same Player", "Salary": 5000,
         "Actual": 12, "Slate": "main", "SlatePlayers": 180},
        {"Date": "2026-08-02", "Name": "Same Player", "Salary": 5200,
         "Actual": 8, "SlatePlayers": 140},
    ])

    result = canonical_observations(frame)

    assert len(result) == 2
    assert result.loc[result["Date"] == "2026-08-01", "Slate"].item() == "main"


def test_price_tiers_discover_a_score_break_and_ignore_missing_results():
    rows = []
    for index in range(20):
        rows.append({"Date": f"2026-08-{index % 9 + 1:02d}", "Name": f"Low {index}",
                     "Salary": 5000 + index * 10, "Actual": 5, "Type": "P", "DK Pos": "P"})
    for index in range(20):
        rows.append({"Date": f"2026-08-{index % 9 + 1:02d}", "Name": f"High {index}",
                     "Salary": 9000 + index * 10, "Actual": 25, "Type": "P", "DK Pos": "P"})
    rows.append({"Date": "2026-08-01", "Name": "Unknown", "Salary": 12000,
                 "Actual": np.nan, "Type": "P", "DK Pos": "P"})
    frame = _observations(rows)

    result = price_tiers(frame)

    assert len(result) == 2
    assert result["Scored N"].sum() == 40
    assert result.iloc[0]["Max Salary"] < result.iloc[1]["Min Salary"]
    assert result.iloc[0]["Avg Pts"] == 5
    assert result.iloc[1]["Avg Pts"] == 25


def test_performance_origins_reports_salary_distribution_by_outcome():
    frame = _observations([
        {"Date": "2026-08-01", "Name": "Elite", "Salary": 6000, "Actual": 22},
        {"Date": "2026-08-01", "Name": "Solid", "Salary": 4500, "Actual": 12},
        {"Date": "2026-08-01", "Name": "Useful", "Salary": 3500, "Actual": 7},
        {"Date": "2026-08-01", "Name": "Low", "Salary": 2500, "Actual": 2},
    ])

    result = performance_origins(frame).set_index("Performance Tier")

    assert result.loc["Elite", "Median Salary"] == 6000
    assert result.loc["Solid", "Median Salary"] == 4500


def test_position_payup_reports_absolute_and_marginal_value():
    rows = []
    for index, (salary, points) in enumerate(
            ((3000, 4), (3200, 5), (3400, 6), (5000, 10)), start=1):
        rows.append({"Date": f"2026-08-0{index}", "Name": f"SS {index}",
                     "Salary": salary, "Actual": points, "DK Pos": "SS"})

    result = position_payup(_observations(rows)).iloc[0]

    assert result["Position"] == "SS"
    assert result["Extra $"] > 0
    assert result["Extra Pts"] > 0
    assert result["Marginal Pts/$1K"] > 0


def test_position_depth_counts_empirical_upper_tier_options():
    rows = []
    salaries_by_date = {
        "2026-08-01": (2500, 2700, 2900, 4000),
        "2026-08-02": (2500, 4000, 4200, 5000),
        "2026-08-03": (4000, 4200, 5000, 5200),
    }
    for night, salaries in salaries_by_date.items():
        for index, salary in enumerate(salaries):
            rows.append({"Date": night, "Name": f"{night}-{index}", "Salary": salary,
                         "Actual": index * 3, "DK Pos": "SS"})
    empirical = pd.DataFrame([
        {"Type": "H", "Tier": 1, "Min Salary": 2000, "Max Salary": 3000},
        {"Type": "H", "Tier": 2, "Min Salary": 3100, "Max Salary": 3800},
        {"Type": "H", "Tier": 3, "Min Salary": 3900, "Max Salary": 4900},
        {"Type": "H", "Tier": 4, "Min Salary": 5000, "Max Salary": 6000},
    ])

    days, _ = position_depth(_observations(rows), empirical)
    counts = days.set_index("Date")["Upper Tier N"]

    assert counts["2026-08-01"] == 1
    assert counts["2026-08-03"] == 4
    assert days.set_index("Date").loc["2026-08-03", "Depth"] == "Deep"


def test_player_history_calculates_up_and_down_from_prior_appearance():
    frame = _observations([
        {"Date": "2026-08-01", "Name": "Mover", "Salary": 4000, "Actual": 5},
        {"Date": "2026-08-03", "Name": "Mover", "Salary": 4400, "Actual": 12},
        {"Date": "2026-08-07", "Name": "Mover", "Salary": 4100, "Actual": 3},
    ])

    summary, detail = player_history(frame)

    assert summary.loc[0, "Ups"] == 1
    assert summary.loc[0, "Downs"] == 1
    assert summary.loc[0, "Pts After Up"] == 12
    assert summary.loc[0, "Pts After Down"] == 3
    assert detail["Price Change"].tolist()[1:] == [400, -300]


def test_change_summary_uses_different_bust_threshold_for_pitchers():
    detail = pd.DataFrame([
        {"Type": "H", "Direction": "Up", "Price Change": 100, "Actual": 5},
        {"Type": "P", "Direction": "Up", "Price Change": 100, "Actual": 5},
    ])

    result = change_summary(detail).set_index("Direction")

    assert result.loc["Up", "Bust%"] == 50


def test_position_days_flags_bottom_quartile_salary_pool():
    rows = []
    for index, salary in enumerate((4000, 4200, 4400, 2000), start=1):
        for player in range(4):
            rows.append({
                "Date": f"2026-08-0{index}", "Name": f"P{index}-{player}",
                "Salary": salary, "Actual": player * 2, "DK Pos": "2B",
            })
    days, _ = position_days(_observations(rows))

    weakest = days.loc[days["Date"] == "2026-08-04"].iloc[0]
    assert bool(weakest["Weak"])
    assert weakest["Salary Index"] < 50
