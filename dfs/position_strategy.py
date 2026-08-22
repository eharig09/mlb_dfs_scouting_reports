"""Position scarcity and paired full-lineup strategy backtests.

The salary analysis describes individual players.  This module measures the decision that
actually matters: if one roster slot is filled from the pay-up pool instead of the value
pool, what happens to the other nine slots and to the complete lineup?

All strategy inputs (salary, projection, ceiling, lineup status) come from the archived
pregame board.  Actual points are evaluation only.  Scarcity labels and learned salary
tiers are walk-forward: only earlier dates define a target date's historical context.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .optimizer import OptimizerError, eligible_positions, optimize
from .price_analysis import (
    PERFORMANCE_LEVELS,
    POSITION_ORDER,
    canonical_observations,
    explode_positions,
    price_tiers,
)


HITTER_POSITIONS = POSITION_ORDER[1:]
OBJECTIVE_COLUMNS = {"proj": "Proj", "ceiling": "Ceiling", "floor": "Floor"}


def _percentile_against(value, history):
    values = pd.to_numeric(history, errors="coerce").dropna()
    if not len(values) or pd.isna(value):
        return np.nan
    return round(float((values <= value).mean() * 100), 1)


def replacement_scarcity(observations, min_history=4):
    """One pre-lock scarcity row per slate and hitter position.

    Replacement is the third projected option at a one-seat position and the sixth at OF,
    where three starters plus three alternatives represent a genuinely flexible pool.
    ``Viable N`` counts players no more than 1.5 projected points behind the best while
    still clearing replacement level.
    """
    frame = explode_positions(observations)
    if frame.empty:
        return pd.DataFrame()
    frame = frame[frame["Position"].isin(HITTER_POSITIONS)].copy()
    frame["Proj"] = pd.to_numeric(frame.get("Proj"), errors="coerce")
    frame["Ceiling"] = pd.to_numeric(frame.get("Ceiling"), errors="coerce")
    frame = frame[frame["Proj"].notna()]
    rows = []
    for (night, slate, position), group in frame.groupby(["Date", "Slate", "Position"]):
        ordered = group.sort_values(["Proj", "Ceiling"], ascending=False)
        replacement_rank = 6 if position == "OF" else 3
        index = min(replacement_rank, len(ordered)) - 1
        replacement = float(ordered.iloc[index]["Proj"])
        top = float(ordered.iloc[0]["Proj"])
        second = float(ordered.iloc[min(1, len(ordered) - 1)]["Proj"])
        viable_floor = max(replacement, top - 1.5)
        viable = group[group["Proj"] >= viable_floor]
        cheapest = viable.loc[viable["Salary"].idxmin()]
        top_player = ordered.iloc[0]
        pay_cut = float(group["Salary"].quantile(0.75))
        pay = group[group["Salary"] >= pay_cut]
        rows.append({
            "Date": night, "Slate": slate, "Position": position, "Pool N": len(group),
            "Top Player": top_player["Name"], "Top Proj": round(top, 2),
            "Top Ceiling": round(float(top_player["Ceiling"]), 2),
            "Top Salary": round(float(top_player["Salary"])),
            "Second Proj": round(second, 2), "Replacement Rank": replacement_rank,
            "Replacement Proj": round(replacement, 2),
            "Top-to-Replacement": round(top - replacement, 2),
            "Top-to-Second": round(top - second, 2), "Viable N": len(viable),
            "Cheapest Viable": cheapest["Name"],
            "Cheapest Viable $": round(float(cheapest["Salary"])),
            "Cheapest Viable Proj": round(float(cheapest["Proj"]), 2),
            "Pay-up Cut": round(pay_cut / 100) * 100, "Pay-up N": len(pay),
            "Confirmed%": round(group["Lineup"].astype(str).str.lower().eq("confirmed").mean() * 100, 1),
        })
    result = pd.DataFrame(rows).sort_values(["Date", "Slate", "Position"]).reset_index(drop=True)
    result["Viable Percentile"] = np.nan
    result["Top Quality Percentile"] = np.nan
    result["Scarcity"] = "Warmup"

    # Strictly earlier dates establish what is shallow/deep and strong/weak. Current-slate
    # salary and projections are known at decision time; future results are never used.
    for index, row in result.iterrows():
        prior = result[(result["Position"] == row["Position"]) & (result["Date"] < row["Date"])]
        if len(prior) < min_history:
            continue
        viable_pct = _percentile_against(row["Viable N"], prior["Viable N"])
        quality_pct = _percentile_against(row["Top Proj"], prior["Top Proj"])
        result.loc[index, "Viable Percentile"] = viable_pct
        result.loc[index, "Top Quality Percentile"] = quality_pct
        if viable_pct <= 33.4:
            label = "One standout" if quality_pct >= 60 else "Weak/empty"
        elif viable_pct >= 66.7:
            label = "Deep"
        else:
            label = "Typical"
        result.loc[index, "Scarcity"] = label
    return result


def walkforward_tier_audit(observations, min_train_hitters=100, min_train_pitchers=30):
    """Assign each player a salary tier learned strictly from earlier scored dates."""
    canonical = canonical_observations(observations).copy()
    dates = sorted(canonical["Date"].unique())
    rows = []
    for night in dates:
        train = canonical[(canonical["Date"] < night) & canonical["Actual"].notna()]
        target = canonical[canonical["Date"] == night]
        for kind, minimum in (("H", min_train_hitters), ("P", min_train_pitchers)):
            kind_train = train[train["Type"] == kind]
            kind_target = target[target["Type"] == kind]
            if len(kind_train) < minimum or kind_target.empty:
                continue
            learned = price_tiers(kind_train)
            learned = learned[learned["Type"] == kind].sort_values("Tier")
            if learned.empty:
                continue
            ceilings = list(learned["Max Salary"])

            def assign(salary):
                for tier, ceiling in enumerate(ceilings, start=1):
                    if salary <= ceiling:
                        return tier
                return len(ceilings)

            for _, player in kind_target.iterrows():
                tier = assign(float(player["Salary"]))
                definition = learned.iloc[tier - 1]
                rows.append({
                    "Date": night, "Slate": player["Slate"], "Name": player["Name"],
                    "Type": kind, "Salary": player["Salary"], "Walk-forward Tier": tier,
                    "Learned Min $": definition["Min Salary"],
                    "Learned Max $": definition["Max Salary"],
                    "Train N": len(kind_train), "Actual": player["Actual"],
                    "Performance Tier": (
                        "" if pd.isna(player["Actual"])
                        else next((label for label, floor in PERFORMANCE_LEVELS[kind]
                                   if player["Actual"] >= floor), "Low")
                    ),
                })
    return pd.DataFrame(rows)


def _prepare_optimizer_pool(group):
    pool = group.copy().reset_index(drop=True)
    for column in ("Proj", "Ceiling", "Floor"):
        pool[column] = pd.to_numeric(pool.get(column), errors="coerce")
    pool = pool[pool["Proj"].notna() & pool["Salary"].notna()].copy()
    pool["Ceiling"] = pool["Ceiling"].fillna(pool["Proj"])
    pool["Floor"] = pool["Floor"].fillna(0)
    pool["Game"] = pool["Game"].replace("@", np.nan)
    pool["Game"] = pool["Game"].fillna(
        pool.apply(lambda row: "@".join(sorted((str(row["Team"]), str(row["Opp"])))), axis=1))
    return pool


def _lineup_record(lineup, position, strategy, objective, pay_cut):
    players = lineup["players"].copy()
    players["Actual"] = pd.to_numeric(players.get("Actual"), errors="coerce")
    position_players = players[players["Roster"] == position]
    position_scored = int(position_players["Actual"].notna().sum())
    position_actual = (float(position_players["Actual"].sum())
                       if position_scored == len(position_players) else np.nan)
    scored = int(players["Actual"].notna().sum())
    actual = float(players["Actual"].sum()) if scored == len(players) else np.nan
    other = players[players["Roster"] != position]
    other_scored = int(other["Actual"].notna().sum())
    other_actual = float(other["Actual"].sum()) if other_scored == len(other) else np.nan
    return {
        "Strategy": strategy, "Objective": objective, "Pay-up Cut": pay_cut,
        "Position Player": " | ".join(position_players["Name"]),
        "Position Salary": float(position_players["Salary"].sum()),
        "Position Proj": float(position_players["Proj"].sum()),
        "Position Ceiling": float(position_players["Ceiling"].sum()),
        "Position Actual": position_actual,
        "Lineup Salary": lineup["salary"], "Lineup Proj": lineup["proj"],
        "Lineup Ceiling": lineup["ceiling"], "Lineup Floor": lineup["floor"],
        "Lineup Actual": actual, "Lineup Scored": scored,
        "Other Salary": float(other["Salary"].sum()),
        "Other Proj": float(other["Proj"].sum()),
        "Other Ceiling": float(other["Ceiling"].sum()),
        "Other Actual": other_actual, "Other Scored": other_scored,
        "Lineup Names": " | ".join(players["Name"]),
        "Actual Map": dict(zip(players["Name"], players["Actual"])),
    }


def paired_lineup_backtest(observations, objectives=("proj", "ceiling")):
    """Best full lineup with the target slot constrained to pay-up versus value."""
    rows = []
    for (night, slate), raw in observations.groupby(["Date", "Slate"]):
        pool = _prepare_optimizer_pool(raw)
        # A raw salary/results fallback has no pre-lock projections and cannot support a
        # decision backtest. It remains in the descriptive price history only.
        if len(pool) < 10:
            continue
        for position in HITTER_POSITIONS:
            eligible = pool[pool.apply(lambda row: position in eligible_positions(row), axis=1)]
            if len(eligible) < 2:
                continue
            pay_cut = round(float(eligible["Salary"].quantile(0.75)) / 100) * 100
            value_names = eligible[eligible["Salary"] < pay_cut]["Name"].tolist()
            pay_names = eligible[eligible["Salary"] >= pay_cut]["Name"].tolist()
            if not value_names or not pay_names:
                continue
            for objective in objectives:
                if OBJECTIVE_COLUMNS.get(objective) not in pool.columns:
                    continue
                records = []
                strategies = {
                    "Value": {"names": pay_names, "min": 0, "max": 0},
                    "Pay-up": {"names": pay_names, "min": 1, "max": 3 if position == "OF" else 1},
                }
                for strategy, group_constraint in strategies.items():
                    try:
                        lineups, _, _ = optimize(
                            pool, n_lineups=1, objective=objective, randomness=0.0,
                            max_overlap=9, stack_bonus=False, conflict_penalty=0,
                            slot_groups={position: group_constraint})
                    except OptimizerError:
                        lineups = []
                    if not lineups:
                        continue
                    records.append(_lineup_record(
                        lineups[0], position, strategy, objective, pay_cut))
                if len(records) != 2:
                    continue
                rows.extend({"Date": night, "Slate": slate, "Position": position, **record}
                            for record in records)
    return pd.DataFrame(rows)


def pair_comparison(lineups, scarcity=None):
    """Wide pay-up-minus-value deltas, including what happened to the other nine slots."""
    if lineups.empty:
        return pd.DataFrame()
    records = []
    keys = ["Date", "Slate", "Position", "Objective"]
    for key, group in lineups.groupby(keys):
        indexed = group.set_index("Strategy")
        if not {"Pay-up", "Value"} <= set(indexed.index):
            continue
        pay, value = indexed.loc["Pay-up"], indexed.loc["Value"]
        extra = pay["Position Salary"] - value["Position Salary"]
        pay_actuals = pay["Actual Map"]
        value_actuals = value["Actual Map"]
        pay_only = set(pay_actuals) - set(value_actuals)
        value_only = set(value_actuals) - set(pay_actuals)
        changed_known = all(pd.notna(pay_actuals[name]) for name in pay_only) and \
            all(pd.notna(value_actuals[name]) for name in value_only)
        changed_actual_lift = (
            sum(float(pay_actuals[name]) for name in pay_only)
            - sum(float(value_actuals[name]) for name in value_only)
            if changed_known else np.nan)
        complete_pair = pd.notna(pay["Lineup Actual"]) and pd.notna(value["Lineup Actual"])
        row = dict(zip(keys, key))
        row.update({
            "Pay-up Player": pay["Position Player"], "Value Player": value["Position Player"],
            "Pay-up Position $": pay["Position Salary"], "Value Position $": value["Position Salary"],
            "Position Extra $": extra,
            "Position Proj Lift": pay["Position Proj"] - value["Position Proj"],
            "Position Ceiling Lift": pay["Position Ceiling"] - value["Position Ceiling"],
            "Position Actual Lift": pay["Position Actual"] - value["Position Actual"]
                                    if pd.notna(pay["Position Actual"]) and pd.notna(value["Position Actual"])
                                    else np.nan,
            "Other Salary Change": pay["Other Salary"] - value["Other Salary"],
            "Other Proj Change": pay["Other Proj"] - value["Other Proj"],
            "Other Ceiling Change": pay["Other Ceiling"] - value["Other Ceiling"],
            "Other Actual Change": pay["Other Actual"] - value["Other Actual"]
                                   if pd.notna(pay["Other Actual"]) and pd.notna(value["Other Actual"])
                                   else np.nan,
            "Lineup Salary Change": pay["Lineup Salary"] - value["Lineup Salary"],
            "Lineup Proj Lift": pay["Lineup Proj"] - value["Lineup Proj"],
            "Lineup Ceiling Lift": pay["Lineup Ceiling"] - value["Lineup Ceiling"],
            "Lineup Actual Lift": changed_actual_lift,
            "Pay-up Actual": pay["Lineup Actual"], "Value Actual": value["Lineup Actual"],
            "Complete Actual Pair": complete_pair,
            "Comparable Actual Pair": changed_known,
            "Actual Coverage": ("Complete" if complete_pair else
                                "Changed players known" if changed_known else "Missing changed player"),
            "Changed Players": " | ".join(sorted(pay_only | value_only)),
            "Net Proj/$1K": (pay["Lineup Proj"] - value["Lineup Proj"]) / (extra / 1000)
                            if extra else np.nan,
            "Net Ceiling/$1K": (pay["Lineup Ceiling"] - value["Lineup Ceiling"]) / (extra / 1000)
                               if extra else np.nan,
            "Net Actual/$1K": changed_actual_lift / (extra / 1000)
                              if extra and pd.notna(changed_actual_lift) else np.nan,
        })
        records.append(row)
    result = pd.DataFrame(records)
    if scarcity is not None and not scarcity.empty and not result.empty:
        result = result.merge(
            scarcity[["Date", "Slate", "Position", "Scarcity", "Viable N",
                      "Top-to-Replacement", "Cheapest Viable $"]],
            on=["Date", "Slate", "Position"], how="left")
    return result


def strategy_summary(comparisons):
    if comparisons.empty:
        return pd.DataFrame()
    rows = []
    for (position, objective, scarcity), group in comparisons.groupby(
            ["Position", "Objective", "Scarcity"], dropna=False):
        comparable = group[group["Comparable Actual Pair"]]
        complete = group[group["Complete Actual Pair"]]
        rows.append({
            "Position": position, "Objective": objective, "Scarcity": scarcity,
            "Pairs": len(group), "Comparable Actual Pairs": len(comparable),
            "Complete Actual Pairs": len(complete),
            "Avg Extra $": round(group["Position Extra $"].mean()),
            "Avg Lineup Proj Lift": round(group["Lineup Proj Lift"].mean(), 2),
            "Avg Lineup Ceiling Lift": round(group["Lineup Ceiling Lift"].mean(), 2),
            "Avg Other Proj Cost": round(group["Other Proj Change"].mean(), 2),
            "Avg Actual Lift": round(comparable["Lineup Actual Lift"].mean(), 2)
                               if len(comparable) else np.nan,
            "Pay-up Win%": round((comparable["Lineup Actual Lift"] > 0).mean() * 100, 1)
                            if len(comparable) else np.nan,
            "Net Actual/$1K": round(comparable["Net Actual/$1K"].mean(), 2)
                              if len(comparable) else np.nan,
        })
    return pd.DataFrame(rows)


def strategy_card(observations, scarcity, comparisons, target_date=None, objective="ceiling"):
    """Current slate decision card using current projections and prior results only."""
    if observations.empty or scarcity.empty or comparisons.empty:
        return pd.DataFrame()
    target_date = str(target_date or observations["Date"].max())
    current = comparisons[(comparisons["Date"] == target_date) &
                          (comparisons["Objective"] == objective)].copy()
    if current.empty:
        return pd.DataFrame()
    history = comparisons[(comparisons["Date"] < target_date) &
                          (comparisons["Objective"] == objective) &
                          comparisons["Comparable Actual Pair"]]
    rows = []
    for _, row in current.iterrows():
        matched = history[(history["Position"] == row["Position"]) &
                          (history["Scarcity"] == row.get("Scarcity"))]
        context = "Matched scarcity"
        if len(matched) < 2:
            matched = history[history["Position"] == row["Position"]]
            context = "All prior slates"
        historical_lift = matched["Lineup Actual Lift"].mean() if len(matched) else np.nan
        win_rate = (matched["Lineup Actual Lift"] > 0).mean() * 100 if len(matched) else np.nan
        current_edge = row["Lineup Ceiling Lift"] if objective == "ceiling" else row["Lineup Proj Lift"]
        if current_edge > 1 and (pd.isna(historical_lift) or historical_lift > 0):
            lean = "Pay up"
        elif current_edge < -0.5:
            lean = "Use value"
        else:
            lean = "Flexible"
        rows.append({
            "Date": target_date, "Slate": row["Slate"], "Position": row["Position"],
            "Scarcity": row.get("Scarcity", ""), "Viable N": row.get("Viable N"),
            "Top-to-Replacement": round(float(row.get("Top-to-Replacement")), 2),
            "Value Choice": row["Value Player"], "Value $": round(row["Value Position $"]),
            "Pay-up Choice": row["Pay-up Player"], "Pay-up $": round(row["Pay-up Position $"]),
            "Extra $": round(row["Position Extra $"]),
            "Net Proj Lift": round(row["Lineup Proj Lift"], 2),
            "Net Ceiling Lift": round(row["Lineup Ceiling Lift"], 2),
            "Other Proj Cost": round(row["Other Proj Change"], 2),
            "Historical Pairs": len(matched),
            "Historical Context": context,
            "Historical Actual Lift": round(historical_lift, 2) if pd.notna(historical_lift) else np.nan,
            "Historical Pay-up Win%": round(win_rate, 1) if pd.notna(win_rate) else np.nan,
            "Strategy": lean,
        })
    order = {position: index for index, position in enumerate(HITTER_POSITIONS)}
    result = pd.DataFrame(rows)
    result["_order"] = result["Position"].map(order)
    return result.sort_values(["Slate", "_order"]).drop(columns="_order").reset_index(drop=True)


def _salary_tier_definitions(observations, night, min_training_hitters=50):
    """Outcome-derived hitter salary bands learned strictly before ``night``."""
    history = canonical_observations(observations)
    history = history[(history["Date"] < night) & history["Actual"].notna() &
                      history["Type"].eq("H")]
    if len(history) < min_training_hitters:
        return pd.DataFrame()
    tiers = price_tiers(history)
    return tiers[tiers["Type"].eq("H")].sort_values("Tier").reset_index(drop=True)


def _assign_salary_tier(salary, tiers):
    for _, tier in tiers.iterrows():
        if float(salary) <= float(tier["Max Salary"]):
            return int(tier["Tier"])
    return int(tiers.iloc[-1]["Tier"])


def _tier_lineup_record(lineup, position, tier, tier_names, definition, objective):
    players = lineup["players"].copy()
    players["Actual"] = pd.to_numeric(players.get("Actual"), errors="coerce")
    position_players = players[players["Roster"].eq(position)]
    chosen = position_players[position_players["Name"].isin(tier_names)]
    scored = int(players["Actual"].notna().sum())
    chosen_scored = int(chosen["Actual"].notna().sum())
    return {
        "Tier": int(tier),
        "Learned Price Band": definition["Observed Price Band"],
        "Historical Avg Pts": definition["Avg Pts"],
        "Historical Pts/$1K": definition["Pts/$1K"],
        "Historical Elite%": definition["Elite%"],
        "Historical Solid+%": definition["Solid+%"],
        "Objective": objective,
        "Tier Player": " | ".join(chosen["Name"]),
        "Tier Player Count": len(chosen),
        "Tier Salary": float(chosen["Salary"].sum()),
        "Tier Proj": float(chosen["Proj"].sum()),
        "Tier Ceiling": float(chosen["Ceiling"].sum()),
        "Tier Actual": (float(chosen["Actual"].sum())
                        if len(chosen) and chosen_scored == len(chosen) else np.nan),
        "Position Players": " | ".join(position_players["Name"]),
        "Position Salary": float(position_players["Salary"].sum()),
        "Position Proj": float(position_players["Proj"].sum()),
        "Position Ceiling": float(position_players["Ceiling"].sum()),
        "Lineup Salary": lineup["salary"],
        "Lineup Proj": lineup["proj"],
        "Lineup Ceiling": lineup["ceiling"],
        "Lineup Floor": lineup["floor"],
        "Lineup Actual": (float(players["Actual"].sum()) if scored == len(players) else np.nan),
        "Lineup Scored": scored,
        "Lineup Names": " | ".join(players["Name"]),
    }


def tier_lineup_backtest(observations, objectives=("proj", "ceiling"), n_lineups=5,
                         min_training_hitters=50, target_dates=None):
    """Build a diversified lineup set for every empirical salary tier at each position.

    This intentionally never selects one value player and one pay-up player as proxies for
    their groups.  Each solve requires at least one target-slot player from the tier, then
    generates several complete lineups so the comparison is between opportunity sets.
    """
    rows = []
    target_dates = {str(value) for value in target_dates} if target_dates is not None else None
    for (night, slate), raw in observations.groupby(["Date", "Slate"]):
        if target_dates is not None and str(night) not in target_dates:
            continue
        tiers = _salary_tier_definitions(observations, night, min_training_hitters)
        if len(tiers) < 2:
            continue
        pool = _prepare_optimizer_pool(raw)
        if len(pool) < 10:
            continue
        for position in HITTER_POSITIONS:
            eligible = pool[pool.apply(lambda row: position in eligible_positions(row), axis=1)].copy()
            if len(eligible) < 2:
                continue
            eligible["_tier"] = eligible["Salary"].map(lambda value: _assign_salary_tier(value, tiers))
            for tier, candidates in eligible.groupby("_tier", sort=True):
                definition = tiers[tiers["Tier"].eq(int(tier))].iloc[0]
                names = candidates["Name"].drop_duplicates().tolist()
                for objective in objectives:
                    if OBJECTIVE_COLUMNS.get(objective) not in pool.columns:
                        continue
                    seed_text = f"{night}|{slate}|{position}|{tier}|{objective}"
                    seed = sum((index + 1) * ord(char) for index, char in enumerate(seed_text))
                    try:
                        lineups, _, _ = optimize(
                            pool, n_lineups=n_lineups, objective=objective,
                            randomness=0.08, seed=seed, max_overlap=8,
                            stack_bonus=False, conflict_penalty=0,
                            slot_groups={position: {
                                "names": names, "min": 1,
                                "max": 3 if position == "OF" else 1,
                            }})
                    except OptimizerError:
                        lineups = []
                    for lineup_number, lineup in enumerate(lineups, start=1):
                        record = _tier_lineup_record(
                            lineup, position, tier, names, definition, objective)
                        rows.append({
                            "Date": night, "Slate": slate, "Position": position,
                            "Lineup #": lineup_number,
                            "Tier Candidate N": len(names),
                            "Tier Candidates": " | ".join(names),
                            **record,
                        })
    return pd.DataFrame(rows)


def tier_strategy_summary(lineups, observations=None, scarcity=None):
    """One row per tier opportunity set, including exposures and score distributions."""
    if lineups.empty:
        return pd.DataFrame()
    rows = []
    keys = ["Date", "Slate", "Position", "Objective", "Tier"]
    for key, group in lineups.groupby(keys):
        exposures = group["Tier Player"].str.split(" | ", regex=False).explode()
        exposures = exposures[exposures.astype(str).str.len().gt(0)].value_counts()
        complete = group["Lineup Actual"].dropna()
        tier_actual = group["Tier Actual"].dropna()
        first = group.iloc[0]
        row = dict(zip(keys, key))
        row.update({
            "Learned Price Band": first["Learned Price Band"],
            "Tier Candidate N": int(first["Tier Candidate N"]),
            "Tier Candidates": first["Tier Candidates"],
            "Lineups": len(group),
            "Exposure": " | ".join(
                f"{name} {count / len(group) * 100:.0f}%" for name, count in exposures.items()),
            "Avg Tier Salary": round(group["Tier Salary"].mean()),
            "Avg Position Salary": round(group["Position Salary"].mean()),
            "Avg Lineup Salary": round(group["Lineup Salary"].mean()),
            "Avg Lineup Proj": round(group["Lineup Proj"].mean(), 2),
            "Best Lineup Proj": round(group["Lineup Proj"].max(), 2),
            "Avg Lineup Ceiling": round(group["Lineup Ceiling"].mean(), 2),
            "Best Lineup Ceiling": round(group["Lineup Ceiling"].max(), 2),
            "Avg Tier Actual": round(tier_actual.mean(), 2) if len(tier_actual) else np.nan,
            "Tier Actual Lineups": len(tier_actual),
            "Avg Complete Lineup Actual": round(complete.mean(), 2) if len(complete) else np.nan,
            "Complete Lineups": len(complete),
            "Historical Avg Pts": first["Historical Avg Pts"],
            "Historical Pts/$1K": first["Historical Pts/$1K"],
            "Historical Elite%": first["Historical Elite%"],
            "Historical Solid+%": first["Historical Solid+%"],
        })
        rows.append(row)
    result = pd.DataFrame(rows)
    if observations is not None and not result.empty:
        exploded = explode_positions(observations)
        exploded = exploded[exploded["Position"].isin(HITTER_POSITIONS)].copy()
        for index, row in result.iterrows():
            slate_pool = exploded[(exploded["Date"].eq(row["Date"])) &
                                  (exploded["Slate"].eq(row["Slate"])) &
                                  (exploded["Position"].eq(row["Position"]))]
            result.loc[index, "Position Pool N"] = slate_pool["Name"].nunique()
            result.loc[index, "Tier Pool Share%"] = round(
                row["Tier Candidate N"] / max(1, slate_pool["Name"].nunique()) * 100, 1)
    if scarcity is not None and not scarcity.empty and not result.empty:
        result = result.merge(
            scarcity[["Date", "Slate", "Position", "Scarcity", "Viable N",
                      "Top Player", "Top-to-Replacement"]],
            on=["Date", "Slate", "Position"], how="left")
    return result.sort_values(keys).reset_index(drop=True)


def tier_tradeoffs(summary):
    """Compare each tier's lineup distribution with the cheapest available tier."""
    if summary.empty:
        return pd.DataFrame()
    rows = []
    keys = ["Date", "Slate", "Position", "Objective"]
    for key, group in summary.groupby(keys):
        ordered = group.sort_values("Tier")
        baseline = ordered.iloc[0]
        for _, tier in ordered.iloc[1:].iterrows():
            row = dict(zip(keys, key))
            row.update({
                "Baseline Tier": int(baseline["Tier"]),
                "Compared Tier": int(tier["Tier"]),
                "Baseline Band": baseline["Learned Price Band"],
                "Compared Band": tier["Learned Price Band"],
                "Baseline Candidates": baseline["Tier Candidates"],
                "Compared Candidates": tier["Tier Candidates"],
                "Extra Avg Tier $": tier["Avg Tier Salary"] - baseline["Avg Tier Salary"],
                "Avg Lineup Proj Lift": tier["Avg Lineup Proj"] - baseline["Avg Lineup Proj"],
                "Avg Lineup Ceiling Lift": tier["Avg Lineup Ceiling"] - baseline["Avg Lineup Ceiling"],
                "Best Ceiling Lift": tier["Best Lineup Ceiling"] - baseline["Best Lineup Ceiling"],
                "Tier Actual Lift": (tier["Avg Tier Actual"] - baseline["Avg Tier Actual"]
                                     if pd.notna(tier["Avg Tier Actual"]) and
                                     pd.notna(baseline["Avg Tier Actual"]) else np.nan),
                "Complete Lineup Actual Lift": (
                    tier["Avg Complete Lineup Actual"] - baseline["Avg Complete Lineup Actual"]
                    if pd.notna(tier["Avg Complete Lineup Actual"]) and
                    pd.notna(baseline["Avg Complete Lineup Actual"]) else np.nan),
                "Baseline Candidate N": baseline["Tier Candidate N"],
                "Compared Candidate N": tier["Tier Candidate N"],
                "Scarcity": tier.get("Scarcity", ""),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def tier_strategy_card(summary, target_date=None, objective="ceiling"):
    """Rank all currently available tiers; retain near-equal tiers as flexible options."""
    if summary.empty:
        return pd.DataFrame()
    target_date = str(target_date or summary["Date"].max())
    current = summary[(summary["Date"].eq(target_date)) &
                      (summary["Objective"].eq(objective))].copy()
    if current.empty:
        return pd.DataFrame()
    metric = "Avg Lineup Ceiling" if objective == "ceiling" else "Avg Lineup Proj"
    rows = []
    for (slate, position), group in current.groupby(["Slate", "Position"]):
        group = group.sort_values(metric, ascending=False)
        best = float(group.iloc[0][metric])
        second = float(group.iloc[1][metric]) if len(group) > 1 else best
        for rank, (_, tier) in enumerate(group.iterrows(), start=1):
            gap = best - float(tier[metric])
            if rank == 1 and best - second < 0.5:
                decision = "Flexible leader"
            elif rank == 1 and tier.get("Scarcity") in ("One standout", "Weak/empty"):
                decision = "Priority tier"
            elif rank == 1:
                decision = "Best tier set"
            elif gap <= 0.75:
                decision = "Viable alternative"
            else:
                decision = "Needs savings case"
            rows.append({
                "Date": target_date, "Slate": slate, "Position": position,
                "Rank": rank, "Tier": int(tier["Tier"]),
                "Learned Price Band": tier["Learned Price Band"],
                "Tier Candidate N": tier["Tier Candidate N"],
                "Tier Pool Share%": tier.get("Tier Pool Share%", np.nan),
                "Candidates": tier["Tier Candidates"], "Exposure": tier["Exposure"],
                "Avg Tier $": tier["Avg Tier Salary"],
                "Avg Lineup Proj": tier["Avg Lineup Proj"],
                "Avg Lineup Ceiling": tier["Avg Lineup Ceiling"],
                "Edge From Best": round(float(tier[metric]) - best, 2),
                "Historical Avg Pts": tier["Historical Avg Pts"],
                "Historical Pts/$1K": tier["Historical Pts/$1K"],
                "Historical Elite%": tier["Historical Elite%"],
                "Historical Solid+%": tier["Historical Solid+%"],
                "Scarcity": tier.get("Scarcity", ""), "Viable N": tier.get("Viable N"),
                "Decision": decision,
            })
    order = {position: index for index, position in enumerate(HITTER_POSITIONS)}
    result = pd.DataFrame(rows)
    result["_order"] = result["Position"].map(order)
    return result.sort_values(["Slate", "_order", "Rank"]).drop(columns="_order").reset_index(drop=True)


def depth_tier_decisions(summary):
    """Slate-level answer to whether depth changes the preferred salary tier."""
    if summary.empty:
        return pd.DataFrame()
    rows = []
    keys = ["Date", "Slate", "Position", "Objective"]
    for key, group in summary.groupby(keys):
        metric = "Avg Lineup Ceiling" if key[-1] == "ceiling" else "Avg Lineup Proj"
        group = group.sort_values("Tier")
        cheapest, premium = group.iloc[0], group.iloc[-1]
        best = group.loc[group[metric].idxmax()]
        best_score = float(best[metric])
        near = group[group[metric].ge(best_score - 0.75)]
        premium_lift = float(premium[metric]) - float(cheapest[metric])
        actual_lift = (
            premium["Avg Tier Actual"] - cheapest["Avg Tier Actual"]
            if pd.notna(premium["Avg Tier Actual"]) and pd.notna(cheapest["Avg Tier Actual"])
            else np.nan)
        scarcity_label = best.get("Scarcity", "")
        if scarcity_label == "One standout" and int(best["Tier"]) == int(premium["Tier"]):
            strategy = "Prioritize the premium tier"
        elif scarcity_label == "Weak/empty" and premium_lift < 0.75:
            strategy = "Spend elsewhere"
        elif scarcity_label == "Deep" or len(near) >= 2:
            strategy = "Flexible across near-best tiers"
        elif int(best["Tier"]) == int(premium["Tier"]) and premium_lift >= 0.75:
            strategy = "Premium tier earns the spend"
        else:
            strategy = "Use the best mid/value tier"
        row = dict(zip(keys, key))
        row.update({
            "Scarcity": scarcity_label,
            "Viable N": best.get("Viable N"),
            "Available Tiers": len(group),
            "Near-best Tiers": len(near),
            "Near-best Candidates": int(near["Tier Candidate N"].sum()),
            "Best Tier": int(best["Tier"]),
            "Best Band": best["Learned Price Band"],
            "Best Tier Candidates": best["Tier Candidates"],
            "Best Tier Candidate N": int(best["Tier Candidate N"]),
            "Best Tier Pool Share%": best.get("Tier Pool Share%", np.nan),
            "Cheapest Tier": int(cheapest["Tier"]),
            "Premium Tier": int(premium["Tier"]),
            "Premium Candidate N": int(premium["Tier Candidate N"]),
            "Premium Pool Share%": premium.get("Tier Pool Share%", np.nan),
            "Premium Extra Avg Tier $": premium["Avg Tier Salary"] - cheapest["Avg Tier Salary"],
            "Premium Lineup Edge": round(premium_lift, 2),
            "Premium Tier Actual Lift": round(actual_lift, 2) if pd.notna(actual_lift) else np.nan,
            "Comparable Tier Actual": pd.notna(actual_lift),
            "Strategy Read": strategy,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def depth_tier_summary(decisions):
    """Historical rollup of premium-versus-flexible behavior by scarcity state."""
    if decisions.empty:
        return pd.DataFrame()
    rows = []
    for (position, objective, scarcity), group in decisions.groupby(
            ["Position", "Objective", "Scarcity"], dropna=False):
        comparable = group[group["Comparable Tier Actual"]]
        rows.append({
            "Position": position, "Objective": objective, "Scarcity": scarcity,
            "Slates": len(group),
            "Avg Viable N": round(group["Viable N"].mean(), 1),
            "Avg Available Tiers": round(group["Available Tiers"].mean(), 1),
            "Avg Near-best Tiers": round(group["Near-best Tiers"].mean(), 1),
            "Avg Near-best Candidates": round(group["Near-best Candidates"].mean(), 1),
            "Premium Was Best%": round(
                group["Best Tier"].eq(group["Premium Tier"]).mean() * 100, 1),
            "Avg Premium Candidate N": round(group["Premium Candidate N"].mean(), 1),
            "Avg Premium Pool Share%": round(group["Premium Pool Share%"].mean(), 1),
            "Avg Premium Extra $": round(group["Premium Extra Avg Tier $"].mean()),
            "Avg Premium Lineup Edge": round(group["Premium Lineup Edge"].mean(), 2),
            "Comparable Actual Slates": len(comparable),
            "Avg Premium Tier Actual Lift": (
                round(comparable["Premium Tier Actual Lift"].mean(), 2)
                if len(comparable) else np.nan),
            "Premium Actual Win%": (
                round(comparable["Premium Tier Actual Lift"].gt(0).mean() * 100, 1)
                if len(comparable) else np.nan),
        })
    return pd.DataFrame(rows)


def _markdown_table(frame):
    if frame.empty:
        return "_No rows._"
    view = frame.copy().replace({np.nan: ""})
    headers = list(view.columns)
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in view.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def write_strategy_outputs(observations, directory, target_date=None,
                           objectives=("proj", "ceiling"), card_objective="ceiling"):
    scarcity = replacement_scarcity(observations)
    walkforward = walkforward_tier_audit(observations)
    lineups = tier_lineup_backtest(observations, objectives=objectives)
    summary = tier_strategy_summary(
        lineups, observations=observations, scarcity=scarcity)
    comparisons = tier_tradeoffs(summary)
    card = tier_strategy_card(
        summary, target_date=target_date, objective=card_objective)
    depth_decisions = depth_tier_decisions(summary)
    depth_summary = depth_tier_summary(depth_decisions)
    outputs = {
        "replacement_scarcity.csv": scarcity,
        "walkforward_tiers.csv": walkforward,
        "tier_lineups.csv": lineups,
        "tier_strategy_summary.csv": summary,
        "tier_tradeoffs.csv": comparisons,
        "slate_depth_tier_decisions.csv": depth_decisions,
        "depth_tier_strategy.csv": depth_summary,
        "daily_strategy_card.csv": card,
    }
    os.makedirs(directory, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(os.path.join(directory, name), index=False)
    target = str(target_date or observations["Date"].max())
    report = [f"# Position strategy card: {target}", "", _markdown_table(card), "",
              "Each row is a salary tier, not a one-player proxy. Every tier is tested through "
              "a diversified set of complete DK lineups under the same constraints. The learned "
              "bands use only earlier scored slates; candidate count and pool share show whether "
              "the tier is unusually crowded or thin. Missing outcomes are never treated as zero.", ""]
    path = os.path.join(directory, "strategy_card.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report))
    return outputs, path


def main(argv=None):
    import argparse
    from .price_analysis import ANALYSIS_ROOT, load_observations

    parser = argparse.ArgumentParser(
        description="Build position scarcity and tier-level lineup strategy.")
    parser.add_argument("--from", dest="start", help="First archived date (YYYY-MM-DD).")
    parser.add_argument("--to", dest="end", help="Last/current slate date (YYYY-MM-DD).")
    parser.add_argument("--slate", help="Only one slate label, e.g. main.")
    parser.add_argument("--objective", choices=sorted(OBJECTIVE_COLUMNS), default="ceiling",
                        help="Objective used for the daily strategy card (default ceiling).")
    parser.add_argument("--output-root", default=ANALYSIS_ROOT)
    args = parser.parse_args(argv)

    observations, _ = load_observations(args.start, args.end, slate=args.slate)
    if observations.empty:
        parser.error("no archived boards matched that date/slate range")
    start = args.start or str(observations["Date"].min())
    end = args.end or str(observations["Date"].max())
    directory = os.path.join(args.output_root, f"price_{start}_{end}")
    objectives = tuple(dict.fromkeys(("proj", args.objective)))
    outputs, report = write_strategy_outputs(
        observations, directory, target_date=end, objectives=objectives,
        card_objective=args.objective)
    print(f"Strategy card: {report}")
    print(f"Tables: {directory} ({len(outputs)} strategy CSV files)")


if __name__ == "__main__":
    main()
