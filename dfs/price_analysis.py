"""Longitudinal DraftKings salary analysis.

This module answers questions that one nightly board cannot:

* how a player's price changes from appearance to appearance;
* whether price and price changes are associated with actual DK points;
* where hitter and pitcher salary tiers separate (or fail to separate);
* what each roster position costs and whether paying up has historically helped;
* which slates offered an unusually cheap/weak position pool; and
* the cheapest pitcher bands that have supplied useful scores.

The input is deliberately the archived ``slate_<slate>.csv`` board output rather than the
raw DK export.  A raw export includes hundreds of bench players and relievers who were
never viable for Classic contests.  Actual points come from the largest matching
``contest-standings`` export for the same date and slate.  A missing outcome is unknown,
never zero.

Run it with::

    python -m dfs.price_analysis --from 2026-07-27 --to 2026-08-08
"""

from __future__ import annotations

import argparse
import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from .naming import OUTPUT_ROOT, label_slates, latest
from .results import RESULTS_DIR, contest_night, list_contests
from .salaries import SALARY_DIRS, list_salary_files, load_salaries, normalize_name


ANALYSIS_ROOT = "dfs_analysis"
POSITION_ORDER = ("P", "C", "1B", "2B", "3B", "SS", "OF")
PERFORMANCE_LEVELS = {
    "H": (("Elite", 20), ("Solid", 10), ("Useful", 5)),
    "P": (("Elite", 25), ("Solid", 15), ("Useful", 10)),
}


def _between(value, start=None, end=None):
    text = str(value)
    return (not start or text >= str(start)) and (not end or text <= str(end))


def _best_contests(directory=RESULTS_DIR):
    """Largest field export keyed by (date, slate).

    Several contests can share one slate.  FPTS is the same in each, but the largest field
    lists the most players and therefore leaves the fewest outcomes unknown.
    """
    best = {}
    for contest in list_contests(directory):
        night, slate = contest_night(contest)
        if not night or not slate:
            continue
        key = (str(night), str(slate))
        if key not in best or contest.get("entries", 0) > best[key].get("entries", 0):
            best[key] = contest
    return best


def _read_board(path):
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [str(column).strip() for column in frame.columns]
    required = {"Name", "Salary"}
    if not required <= set(frame.columns):
        raise ValueError(f"{path} is missing {', '.join(sorted(required - set(frame.columns)))}")
    return frame


def _fallback_from_salary(path, contest):
    """Use priced players seen in results when an old date has no archived board.

    This is intentionally unavailable without a contest.  Keeping the entire raw salary
    export would add every bench hitter and relief pitcher and corrupt every position
    average.  The Source column makes this narrower fallback visible in the coverage table.
    """
    frame = load_salaries(path)
    if frame is None or not contest:
        return pd.DataFrame()
    keys = set(contest.get("fpts") or ())
    frame = frame[frame["DK Name"].map(normalize_name).isin(keys)].copy()
    if frame.empty:
        return frame
    frame = frame.rename(columns={"DK Name": "Name", "DK Team": "Team"})
    frame["Type"] = np.where(
        frame["DK Pos"].astype(str).str.contains(r"(?:^|/)P(?:$|/)", regex=True), "P", "H")
    return frame


def load_observations(start=None, end=None, slate=None, boards_root=OUTPUT_ROOT,
                      results_dir=RESULTS_DIR, salary_dirs=None):
    """Return (player-slate observations, coverage diagnostics).

    One row is one player on one archived DFS board.  The same player may occur twice on a
    date when main and turbo overlap; ``canonical_observations`` removes that duplication
    for longitudinal work while position-day analysis intentionally retains it.
    """
    infos = [info for info in list_salary_files(salary_dirs=salary_dirs or SALARY_DIRS)
             if info.get("date") and _between(info["date"], start, end)]
    by_date = defaultdict(list)
    for info in infos:
        by_date[str(info["date"])].append(info)

    contests = _best_contests(results_dir)
    rows, coverage = [], []
    wanted_slate = str(slate).lower() if slate else None

    for night in sorted(by_date):
        labels = label_slates(by_date[night], night)
        for info in by_date[night]:
            label = labels.get(info["path"], "unknown")
            if wanted_slate and label != wanted_slate:
                continue
            contest = contests.get((night, label))
            board_path = latest("slate", night, label, root=boards_root)
            source = "board"
            if board_path:
                try:
                    frame = _read_board(board_path)
                except Exception as error:
                    coverage.append({"Date": night, "Slate": label, "Status": "error",
                                     "Detail": str(error)})
                    continue
            else:
                frame = _fallback_from_salary(info["path"], contest)
                source = "salary+results"

            if frame.empty:
                coverage.append({"Date": night, "Slate": label, "Status": "skipped",
                                 "Detail": "no archived board and no matching results"})
                continue

            out = pd.DataFrame(index=frame.index)
            out["Date"] = night
            out["Slate"] = label
            out["Name"] = frame.get("Name", "").astype(str).str.strip()
            out["PlayerKey"] = out["Name"].map(normalize_name)
            out["Team"] = frame.get("Team", pd.Series("", index=frame.index)).astype(str)
            out["Type"] = frame.get("Type", pd.Series("", index=frame.index)).astype(str)
            out["DK Pos"] = frame.get("DK Pos", pd.Series("", index=frame.index)).astype(str)
            out["Salary"] = pd.to_numeric(frame["Salary"], errors="coerce")
            out["DK ID"] = frame.get("DK ID", pd.Series(np.nan, index=frame.index))
            # Preserve every signal that was available on the archived board. These are
            # the inputs a pre-lock scarcity or paired-lineup decision is allowed to use;
            # actual points are attached only afterward as the evaluation target.
            for column in ("Proj", "Ceiling", "Floor", "Bust%", "Value", "Ceil Value",
                           "GPP", "CASH", "Own%", "Leverage", "Lev Score", "Slot", "PA"):
                out[column] = pd.to_numeric(
                    frame.get(column, pd.Series(np.nan, index=frame.index)), errors="coerce")
            out["Lineup"] = frame.get(
                "Lineup", pd.Series("", index=frame.index)).astype(str)
            out["Opp"] = frame.get("Opp", pd.Series("", index=frame.index)).astype(str)
            # Old board exports did not persist Game, but the unordered team pairing is
            # enough for DK's two-game rule and pitcher/hitter conflict accounting.
            out["Game"] = ["@".join(sorted((str(team), str(opp))))
                           for team, opp in zip(out["Team"], out["Opp"])]
            out["Source"] = source
            out["BoardPath"] = board_path or info["path"]
            out["SlatePlayers"] = len(frame)
            fpts = (contest or {}).get("fpts") or {}
            out["Actual"] = out["PlayerKey"].map(fpts)
            out["OutcomeSource"] = os.path.basename(contest["path"]) if contest else ""
            out = out[(out["Name"] != "") & out["Salary"].notna() & (out["Salary"] > 0)]
            # Old/raw fallback type detection; boards already carry the authoritative type.
            blank_type = ~out["Type"].isin(["H", "P"])
            out.loc[blank_type, "Type"] = np.where(
                out.loc[blank_type, "DK Pos"].str.contains(r"(?:^|/)P(?:$|/)", regex=True),
                "P", "H")
            rows.append(out)
            coverage.append({
                "Date": night, "Slate": label, "Status": "loaded", "Detail": source,
                "Players": len(out), "Scored": int(out["Actual"].notna().sum()),
                "ContestEntries": int((contest or {}).get("entries", 0)),
            })

    columns = ["Date", "Slate", "Name", "PlayerKey", "Team", "Opp", "Game", "Type",
               "DK Pos", "Salary", "DK ID", "Proj", "Ceiling", "Floor", "Bust%", "Value",
               "Ceil Value", "GPP", "CASH", "Own%", "Leverage", "Lev Score", "Slot", "PA",
               "Lineup", "Actual", "Source", "OutcomeSource", "BoardPath", "SlatePlayers"]
    observations = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)
    return observations, pd.DataFrame(coverage)


def canonical_observations(observations):
    """One observation per player-day, preferring the largest available slate.

    Actual points describe a game, not a contest.  Counting a player once on main and once
    on turbo would give overlapping slates twice the influence on every correlation.
    """
    if observations.empty:
        return observations.copy()
    frame = observations.copy()
    frame["_scored"] = frame["Actual"].notna().astype(int)
    frame = frame.sort_values(
        ["Date", "PlayerKey", "SlatePlayers", "_scored", "Slate"],
        ascending=[True, True, False, False, True],
    )
    return frame.drop_duplicates(["Date", "PlayerKey"], keep="first").drop(columns="_scored")


def _positions(value, player_type=None):
    if str(player_type) == "P":
        return ["P"]
    found = []
    for raw in str(value or "").upper().split("/"):
        position = raw.strip()
        if position in {"LF", "CF", "RF"}:
            position = "OF"
        if position in POSITION_ORDER and position != "P" and position not in found:
            found.append(position)
    return found


def explode_positions(observations):
    rows = []
    for _, row in observations.iterrows():
        for position in _positions(row.get("DK Pos"), row.get("Type")):
            item = row.to_dict()
            item["Position"] = position
            rows.append(item)
    return pd.DataFrame(rows)


def _ci95(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 2:
        return np.nan
    return 1.96 * values.std(ddof=1) / math.sqrt(len(values))


def _spearman(left, right):
    """Rank correlation without scipy's constant-input warning."""
    pair = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(pair) < 3 or pair["left"].nunique() < 2 or pair["right"].nunique() < 2:
        return np.nan
    return pair["left"].corr(pair["right"], method="spearman")


def _performance_level(actual, kind):
    if pd.isna(actual):
        return ""
    for label, floor in PERFORMANCE_LEVELS[kind]:
        if float(actual) >= floor:
            return label
    return "Low"


def _points_range(kind, level):
    levels = PERFORMANCE_LEVELS[kind]
    if level == "Elite":
        return f"{levels[0][1]}+"
    if level == "Solid":
        return f"{levels[1][1]}-{levels[0][1] - 0.01:g}"
    if level == "Useful":
        return f"{levels[2][1]}-{levels[1][1] - 0.01:g}"
    return f"Under {levels[2][1]}"


def performance_origins(observations):
    """Salary distribution conditional on the result being elite/solid/useful/low.

    This reverses the usual salary-bucket question.  Instead of choosing an arbitrary
    price band and asking how it scored, it starts with the DFS outcome and shows where in
    the salary pool those performances actually came from.
    """
    frame = canonical_observations(observations)
    frame = frame[frame["Actual"].notna()].copy()
    frame["Performance Tier"] = [
        _performance_level(actual, kind) for actual, kind in zip(frame["Actual"], frame["Type"])
    ]
    rows = []
    order = ("Elite", "Solid", "Useful", "Low")
    for kind in ("H", "P"):
        type_frame = frame[frame["Type"] == kind]
        for level in order:
            group = type_frame[type_frame["Performance Tier"] == level]
            if group.empty:
                continue
            salary = group["Salary"]
            rows.append({
                "Type": kind, "Performance Tier": level,
                "Points Range": _points_range(kind, level), "N": len(group),
                "Outcome Share%": round(len(group) / len(type_frame) * 100, 1),
                "Avg Pts": round(group["Actual"].mean(), 2),
                "Avg Salary": round(salary.mean()),
                "Salary P10": round(salary.quantile(0.10) / 100) * 100,
                "Salary P25": round(salary.quantile(0.25) / 100) * 100,
                "Median Salary": round(salary.median() / 100) * 100,
                "Salary P75": round(salary.quantile(0.75) / 100) * 100,
                "Salary P90": round(salary.quantile(0.90) / 100) * 100,
                "Avg Pts/$1K": round((group["Actual"] / (salary / 1000)).mean(), 2),
            })
    return pd.DataFrame(rows)


def price_tiers(observations, max_tiers=5, min_leaf_share=0.12, min_leaf=10):
    """Empirical salary bands whose cut points best separate actual DK scores.

    A shallow one-dimensional regression tree discovers the breakpoints.  The minimum
    leaf size prevents a handful of outliers from becoming their own "tier".  Reported
    bounds are real observed DK salaries, not rounded thousand-dollar intervals.
    """
    frame = canonical_observations(observations)
    frame = frame[frame["Actual"].notna()].copy()
    rows = []
    for kind in ("H", "P"):
        group = frame[frame["Type"] == kind].copy()
        if group.empty:
            continue
        leaf_size = min(len(group), max(min_leaf, int(math.ceil(len(group) * min_leaf_share))))
        model = DecisionTreeRegressor(
            max_leaf_nodes=max_tiers, min_samples_leaf=leaf_size, random_state=0)
        model.fit(group[["Salary"]], group["Actual"])
        thresholds = sorted(float(value) for value in model.tree_.threshold if value != -2)
        group["_band"] = pd.cut(
            group["Salary"], [-np.inf, *thresholds, np.inf], labels=False, include_lowest=True)
        for band, band_frame in group.groupby("_band", sort=True):
            actual = band_frame["Actual"]
            salary = band_frame["Salary"]
            elite_floor = PERFORMANCE_LEVELS[kind][0][1]
            solid_floor = PERFORMANCE_LEVELS[kind][1][1]
            bust = PERFORMANCE_LEVELS[kind][2][1]
            low, high = int(salary.min()), int(salary.max())
            rows.append({
                "Type": kind, "Tier": int(band) + 1,
                "Observed Price Band": f"${low:,}-${high:,}",
                "Min Salary": low, "Max Salary": high, "Scored N": len(band_frame),
                "Avg Salary": round(salary.mean()), "Avg Pts": round(actual.mean(), 2),
                "Median Pts": round(actual.median(), 2), "Mean 95% +/-": round(_ci95(actual), 2),
                "Pts/$1K": round((actual / (salary / 1000)).mean(), 2),
                "Elite%": round((actual >= elite_floor).mean() * 100, 1),
                "Solid+%": round((actual >= solid_floor).mean() * 100, 1),
                "Low%": round((actual < bust).mean() * 100, 1),
            })
    return pd.DataFrame(rows).sort_values(["Type", "Min Salary"]).reset_index(drop=True)


def player_history(observations):
    frame = canonical_observations(observations).sort_values(["PlayerKey", "Date"])
    frame["Previous Salary"] = frame.groupby("PlayerKey")["Salary"].shift()
    frame["Price Change"] = frame["Salary"] - frame["Previous Salary"]
    frame["Direction"] = np.select(
        [frame["Price Change"] > 0, frame["Price Change"] < 0], ["Up", "Down"], "Flat")
    frame.loc[frame["Previous Salary"].isna(), "Direction"] = "First"

    rows = []
    for key, group in frame.groupby("PlayerKey", sort=False):
        scored = group[group["Actual"].notna()]
        direction = group[group["Direction"].isin(["Up", "Down", "Flat"])]
        current = group.iloc[-1]
        corr = _spearman(scored["Salary"], scored["Actual"])
        rows.append({
            "Name": current["Name"], "Type": current["Type"], "Team": current["Team"],
            "Observations": len(group), "Scored N": len(scored),
            "Avg Salary": round(group["Salary"].mean()), "Min Salary": round(group["Salary"].min()),
            "Max Salary": round(group["Salary"].max()), "Latest Salary": round(current["Salary"]),
            "Net Change": round(current["Salary"] - group.iloc[0]["Salary"]),
            "Ups": int((direction["Direction"] == "Up").sum()),
            "Downs": int((direction["Direction"] == "Down").sum()),
            "Flats": int((direction["Direction"] == "Flat").sum()),
            "Avg Pts": round(scored["Actual"].mean(), 2) if len(scored) else np.nan,
            "Pts After Up": round(group.loc[(group["Direction"] == "Up") & group["Actual"].notna(), "Actual"].mean(), 2),
            "Pts After Down": round(group.loc[(group["Direction"] == "Down") & group["Actual"].notna(), "Actual"].mean(), 2),
            "Salary-Pts Spearman": round(corr, 3) if pd.notna(corr) else np.nan,
        })
    detail = frame.drop(columns="PlayerKey").reset_index(drop=True)
    summary = pd.DataFrame(rows).sort_values(["Observations", "Avg Salary"], ascending=False)
    return summary.reset_index(drop=True), detail


def change_summary(history_detail, player_type=None):
    frame = history_detail.copy()
    if player_type:
        frame = frame[frame["Type"] == player_type]
    frame = frame[frame["Direction"].isin(["Up", "Down", "Flat"])]
    rows = []
    for direction in ("Up", "Down", "Flat"):
        group = frame[frame["Direction"] == direction]
        scored = group[group["Actual"].notna()]
        thresholds = np.where(scored["Type"].eq("P"), 10, 3)
        rows.append({
            "Direction": direction, "Changes": len(group), "Scored N": len(scored),
            "Avg $ Change": round(group["Price Change"].mean()) if len(group) else np.nan,
            "Avg Pts": round(scored["Actual"].mean(), 2) if len(scored) else np.nan,
            "Median Pts": round(scored["Actual"].median(), 2) if len(scored) else np.nan,
            "Bust%": round((scored["Actual"].to_numpy() <= thresholds).mean() * 100, 1)
                     if len(scored) else np.nan,
        })
    return pd.DataFrame(rows)


def correlations(observations, history_detail):
    rows = []
    canonical = canonical_observations(observations)
    for kind in ("H", "P"):
        scored = canonical[(canonical["Type"] == kind) & canonical["Actual"].notna()]
        rows.append({
            "Group": "Hitters" if kind == "H" else "Pitchers", "Relationship": "Salary vs points",
            "N": len(scored),
            "Spearman": round(_spearman(scored["Salary"], scored["Actual"]), 3),
        })
        changes = history_detail[(history_detail["Type"] == kind) &
                                 history_detail["Price Change"].notna() &
                                 history_detail["Actual"].notna()]
        rows.append({
            "Group": "Hitters" if kind == "H" else "Pitchers", "Relationship": "Price change vs points",
            "N": len(changes),
            "Spearman": round(_spearman(changes["Price Change"], changes["Actual"]), 3),
        })

    pitchers = history_detail[history_detail["Type"] == "P"].sort_values(["Name", "Date"]).copy()
    pitchers["Previous Pts"] = pitchers.groupby("Name")["Actual"].shift()
    repriced = pitchers[pitchers["Previous Pts"].notna() & pitchers["Price Change"].notna()]
    rows.append({
        "Group": "Pitchers", "Relationship": "Previous points vs next price change",
        "N": len(repriced),
        "Spearman": round(_spearman(repriced["Previous Pts"], repriced["Price Change"]), 3),
    })
    return pd.DataFrame(rows), pitchers


def position_summary(observations):
    frame = explode_positions(canonical_observations(observations))
    if frame.empty:
        return pd.DataFrame()
    frame = frame[frame["Position"] != "P"].copy()
    rows = []
    for position, group in frame.groupby("Position"):
        scored = group[group["Actual"].notna()].copy()
        threshold = group["Salary"].quantile(0.75)
        premium = scored[scored["Salary"] >= threshold]
        rest = scored[scored["Salary"] < threshold]
        productive = scored[scored["Actual"] >= 10]
        lift = premium["Actual"].mean() - rest["Actual"].mean() if len(premium) and len(rest) else np.nan
        rows.append({
            "Position": position, "Price N": len(group), "Scored N": len(scored),
            "Avg Price": round(group["Salary"].mean()), "Median Price": round(group["Salary"].median()),
            "Avg Pts": round(scored["Actual"].mean(), 2) if len(scored) else np.nan,
            "Median Pts": round(scored["Actual"].median(), 2) if len(scored) else np.nan,
            "10+%": round((scored["Actual"] >= 10).mean() * 100, 1) if len(scored) else np.nan,
            "Bust%": round((scored["Actual"] <= 3).mean() * 100, 1) if len(scored) else np.nan,
            "Pay-up Cut": round(threshold / 100) * 100,
            "Pay-up Pts": round(premium["Actual"].mean(), 2) if len(premium) else np.nan,
            "Pay-up Lift": round(lift, 2) if pd.notna(lift) else np.nan,
            "Productive Median $": round(productive["Salary"].median()) if len(productive) else np.nan,
        })
    order = {position: index for index, position in enumerate(POSITION_ORDER)}
    result = pd.DataFrame(rows)
    result["_order"] = result["Position"].map(order)
    return result.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def position_payup(observations):
    """Cost and return of the top salary quartile at each hitter position.

    ``Marginal Pts/$1K`` is the direct answer to what the extra salary bought: point lift
    divided by the additional average salary. ``Value Retained%`` compares absolute
    points-per-dollar in the pay-up pool with the rest; below 100 means raw points improved
    but efficiency fell.
    """
    frame = explode_positions(canonical_observations(observations))
    if frame.empty:
        return pd.DataFrame()
    frame = frame[frame["Position"] != "P"].copy()
    rows = []
    for position, pool in frame.groupby("Position"):
        cut = pool["Salary"].quantile(0.75)
        scored = pool[pool["Actual"].notna()]
        pay = scored[scored["Salary"] >= cut]
        base = scored[scored["Salary"] < cut]
        if pay.empty or base.empty:
            continue
        pay_salary, base_salary = pay["Salary"].mean(), base["Salary"].mean()
        pay_points, base_points = pay["Actual"].mean(), base["Actual"].mean()
        extra_salary = pay_salary - base_salary
        extra_points = pay_points - base_points
        pay_eff = (pay["Actual"] / (pay["Salary"] / 1000)).mean()
        base_eff = (base["Actual"] / (base["Salary"] / 1000)).mean()
        rows.append({
            "Position": position, "Pay-up Cut": round(cut / 100) * 100,
            "Base N": len(base), "Pay-up N": len(pay),
            "Base Avg $": round(base_salary), "Pay-up Avg $": round(pay_salary),
            "Extra $": round(extra_salary),
            "Base Avg Pts": round(base_points, 2), "Pay-up Avg Pts": round(pay_points, 2),
            "Extra Pts": round(extra_points, 2),
            "Base Pts/$1K": round(base_eff, 2), "Pay-up Pts/$1K": round(pay_eff, 2),
            "Marginal Pts/$1K": round(extra_points / (extra_salary / 1000), 2)
                                if extra_salary else np.nan,
            "Value Retained%": round(pay_eff / base_eff * 100, 1) if base_eff else np.nan,
            "Base Solid+%": round((base["Actual"] >= 10).mean() * 100, 1),
            "Pay-up Solid+%": round((pay["Actual"] >= 10).mean() * 100, 1),
            "Base Elite%": round((base["Actual"] >= 20).mean() * 100, 1),
            "Pay-up Elite%": round((pay["Actual"] >= 20).mean() * 100, 1),
            "Base Bust%": round((base["Actual"] <= 3).mean() * 100, 1),
            "Pay-up Bust%": round((pay["Actual"] <= 3).mean() * 100, 1),
        })
    order = {position: index for index, position in enumerate(POSITION_ORDER)}
    result = pd.DataFrame(rows)
    result["_order"] = result["Position"].map(order)
    return result.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def position_depth(observations, empirical_tiers):
    """Measure slate-level positional depth and how it changed pay-up economics.

    Depth is based on the upper two hitter salary tiers discovered from actual scoring,
    not a hard-coded dollar amount. Counts and shares are combined so a large slate is not
    automatically called deep merely because it has more games.
    """
    frame = explode_positions(observations)
    frame = frame[frame["Position"].isin(POSITION_ORDER[1:])].copy()
    hitter_tiers = empirical_tiers[empirical_tiers["Type"] == "H"].sort_values("Min Salary")
    if frame.empty or hitter_tiers.empty:
        return pd.DataFrame(), pd.DataFrame()

    max_tier = int(hitter_tiers["Tier"].max())
    upper_floor = max(1, max_tier - 1)
    ceilings = list(hitter_tiers.sort_values("Tier")["Max Salary"])

    def tier_for(salary):
        for number, ceiling in enumerate(ceilings, start=1):
            if salary <= ceiling:
                return number
        return max_tier

    frame["Empirical Tier"] = frame["Salary"].map(tier_for)
    canonical = explode_positions(canonical_observations(observations))
    canonical = canonical[canonical["Position"].isin(POSITION_ORDER[1:])]
    pay_cuts = canonical.groupby("Position")["Salary"].quantile(0.75).to_dict()

    rows = []
    for (night, slate, position), group in frame.groupby(["Date", "Slate", "Position"]):
        cut = pay_cuts[position]
        scored = group[group["Actual"].notna()]
        pay = scored[scored["Salary"] >= cut]
        base = scored[scored["Salary"] < cut]
        upper = group[group["Empirical Tier"] >= upper_floor]
        top = group[group["Empirical Tier"] == max_tier]
        ordered_salary = group["Salary"].sort_values(ascending=False).reset_index(drop=True)
        pay_eff = (pay["Actual"] / (pay["Salary"] / 1000)).mean() if len(pay) else np.nan
        base_eff = (base["Actual"] / (base["Salary"] / 1000)).mean() if len(base) else np.nan
        lift = pay["Actual"].mean() - base["Actual"].mean() if len(pay) and len(base) else np.nan
        extra_salary = pay["Salary"].mean() - base["Salary"].mean() if len(pay) and len(base) else np.nan
        rows.append({
            "Date": night, "Slate": slate, "Position": position, "Pool N": len(group),
            "Upper Tier N": len(upper), "Upper Tier %": round(len(upper) / len(group) * 100, 1),
            "Top Tier N": len(top), "Top Tier %": round(len(top) / len(group) * 100, 1),
            "Highest $": int(ordered_salary.iloc[0]),
            "Top-to-4th $ Drop": int(ordered_salary.iloc[0] - ordered_salary.iloc[3])
                                  if len(ordered_salary) >= 4 else np.nan,
            "Scored N": len(scored), "Pay-up N": len(pay),
            "Pay-up Lift": round(lift, 2) if pd.notna(lift) else np.nan,
            "Base Pts/$1K": round(base_eff, 2) if pd.notna(base_eff) else np.nan,
            "Pay-up Pts/$1K": round(pay_eff, 2) if pd.notna(pay_eff) else np.nan,
            "Marginal Pts/$1K": round(lift / (extra_salary / 1000), 2)
                                if pd.notna(lift) and extra_salary else np.nan,
            "Base Elite%": round((base["Actual"] >= 20).mean() * 100, 1) if len(base) else np.nan,
            "Pay-up Elite%": round((pay["Actual"] >= 20).mean() * 100, 1) if len(pay) else np.nan,
        })
    days = pd.DataFrame(rows)
    count_rank = days.groupby("Position")["Upper Tier N"].rank(pct=True, method="average")
    share_rank = days.groupby("Position")["Upper Tier %"].rank(pct=True, method="average")
    days["Depth Percentile"] = ((count_rank + share_rank) / 2 * 100).round(1)
    days["Depth"] = np.select(
        [days["Depth Percentile"] <= 33.4, days["Depth Percentile"] >= 66.7],
        ["Shallow", "Deep"], "Typical")

    strategy_rows = []
    for (position, depth), group in days.groupby(["Position", "Depth"]):
        scored = group[group["Pay-up Lift"].notna()]
        pay_eff = scored["Pay-up Pts/$1K"].mean()
        base_eff = scored["Base Pts/$1K"].mean()
        marginal = scored["Marginal Pts/$1K"].mean()
        lift = scored["Pay-up Lift"].mean()
        if len(scored) < 2:
            lean = "Insufficient scored slates"
        elif lift >= 2 and marginal >= 1:
            lean = "Pay up; alternatives showed meaningful opportunity cost"
        elif pay_eff >= base_eff * 0.9:
            lean = "Flexible; pay-up retained most of its points-per-dollar"
        else:
            lean = "Prefer value; pay-up gained points inefficiently"
        strategy_rows.append({
            "Position": position, "Depth": depth, "Slates": len(group),
            "Scored Slates": len(scored), "Avg Upper Tier N": round(group["Upper Tier N"].mean(), 1),
            "Avg Upper Tier %": round(group["Upper Tier %"].mean(), 1),
            "Pay-up Lift": round(lift, 2) if len(scored) else np.nan,
            "Base Pts/$1K": round(base_eff, 2) if len(scored) else np.nan,
            "Pay-up Pts/$1K": round(pay_eff, 2) if len(scored) else np.nan,
            "Marginal Pts/$1K": round(marginal, 2) if len(scored) else np.nan,
            "Strategy Lean": lean,
        })
    strategy = pd.DataFrame(strategy_rows)
    return days.sort_values(["Date", "Slate", "Position"]), strategy


def position_days(observations, min_pool=3):
    """Slate-position salary index and weak-day pay-up evidence."""
    frame = explode_positions(observations)
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    frame = frame[frame["Position"] != "P"].copy()
    frame["DayKey"] = frame["Date"].astype(str) + "|" + frame["Slate"].astype(str)
    rows = []
    for (night, slate, position), group in frame.groupby(["Date", "Slate", "Position"]):
        scored = group[group["Actual"].notna()]
        cutoff = group["Salary"].quantile(0.75)
        premium = scored[scored["Salary"] >= cutoff]
        rest = scored[scored["Salary"] < cutoff]
        rows.append({
            "Date": night, "Slate": slate, "Position": position, "Players": len(group),
            "Avg Price": round(group["Salary"].mean()), "Median Price": round(group["Salary"].median()),
            "Scored N": len(scored), "Avg Pts": round(scored["Actual"].mean(), 2) if len(scored) else np.nan,
            "Pay-up Lift": round(premium["Actual"].mean() - rest["Actual"].mean(), 2)
                           if len(premium) and len(rest) else np.nan,
        })
    days = pd.DataFrame(rows)
    days = days[days["Players"] >= min_pool].copy()
    if days.empty:
        return days, pd.DataFrame()
    days["Historical Median $"] = days.groupby("Position")["Median Price"].transform("median")
    days["Salary Index"] = (days["Median Price"] / days["Historical Median $"] * 100).round(1)
    days["Salary Percentile"] = (days.groupby("Position")["Median Price"].rank(pct=True, method="average") * 100).round(1)
    days["Weak"] = days["Salary Percentile"] <= 25

    strategy = []
    for position, group in days.groupby("Position"):
        weak = group[group["Weak"] & group["Avg Pts"].notna()]
        normal = group[~group["Weak"] & group["Avg Pts"].notna()]
        pay_lift = weak["Pay-up Lift"].mean() if len(weak) else np.nan
        if len(weak) >= 2 and pd.notna(pay_lift) and pay_lift >= 2:
            action = "Pay up: the top price quartile separated on weak slates"
        elif len(weak) >= 2:
            action = "Take the savings: pay-up bats have not separated on weak slates"
        else:
            action = "Insufficient weak-slate outcomes"
        strategy.append({
            "Position": position, "Weak Slates": len(weak),
            "Weak Avg Pts": round(weak["Avg Pts"].mean(), 2) if len(weak) else np.nan,
            "Normal Avg Pts": round(normal["Avg Pts"].mean(), 2) if len(normal) else np.nan,
            "Weak Pay-up Lift": round(pay_lift, 2) if pd.notna(pay_lift) else np.nan,
            "Evidence-based Lean": action,
        })
    return days.sort_values(["Date", "Slate", "Position"]), pd.DataFrame(strategy)


def _markdown_table(frame, columns=None, limit=None):
    if frame is None or frame.empty:
        return "_No observations._"
    view = frame.loc[:, columns] if columns else frame
    if limit:
        view = view.head(limit)
    view = view.copy().replace({np.nan: ""})
    headers = [str(column) for column in view.columns]

    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in view.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def _finding_lines(tiers, origins, payup, correlations_table, pitcher_changes,
                   weak_strategy, depth_strategy, min_n=10):
    lines = []
    for _, row in correlations_table.iterrows():
        if row["N"] >= min_n and pd.notna(row["Spearman"]):
            lines.append(f"- {row['Group']} — {row['Relationship']}: Spearman "
                         f"{row['Spearman']:+.3f} over {int(row['N'])} observations.")

    for kind, label in (("H", "Hitter"), ("P", "Pitcher")):
        elite = origins[(origins["Type"] == kind) & (origins["Performance Tier"] == "Elite")]
        solid = origins[(origins["Type"] == kind) & (origins["Performance Tier"] == "Solid")]
        if not elite.empty:
            row = elite.iloc[0]
            lines.append(f"- {label} elite outcomes ({row['Points Range']} DK points) had a "
                         f"median salary of ${int(row['Median Salary']):,}; the middle 50% came "
                         f"from ${int(row['Salary P25']):,}-${int(row['Salary P75']):,} "
                         f"(n={int(row['N'])}).")
        if not solid.empty:
            row = solid.iloc[0]
            lines.append(f"- {label} solid outcomes ({row['Points Range']} DK points) had a "
                         f"median salary of ${int(row['Median Salary']):,} (n={int(row['N'])}).")
        group = tiers[(tiers["Type"] == kind) & (tiers["Scored N"] >= min_n)]
        if not group.empty:
            raw = group.loc[group["Avg Pts"].idxmax()]
            value = group.loc[group["Pts/$1K"].idxmax()]
            lines.append(f"- The empirical {label.lower()} salary bands put the strongest raw "
                         f"scoring at {raw['Observed Price Band']} ({raw['Avg Pts']:.2f} points) "
                         f"and best efficiency at {value['Observed Price Band']} "
                         f"({value['Pts/$1K']:.2f} points/$1K).")

    for position in ("SS", "OF"):
        match = payup[payup["Position"] == position]
        if not match.empty:
            row = match.iloc[0]
            lines.append(f"- {position} pay-up: an extra ${int(row['Extra $']):,} bought "
                         f"{row['Extra Pts']:+.2f} points ({row['Marginal Pts/$1K']:.2f} "
                         f"marginal points/$1K); absolute efficiency moved from "
                         f"{row['Base Pts/$1K']:.2f} to {row['Pay-up Pts/$1K']:.2f} points/$1K.")

    pitchers = pitcher_changes.set_index("Direction") if not pitcher_changes.empty else pd.DataFrame()
    if not pitchers.empty and {"Up", "Down"} <= set(pitchers.index):
        up, down = pitchers.loc["Up"], pitchers.loc["Down"]
        if up["Scored N"] >= min_n or down["Scored N"] >= min_n:
            lines.append(f"- Pitchers averaged {up['Avg Pts']} after a price increase "
                         f"(n={int(up['Scored N'])}) versus {down['Avg Pts']} after a decrease "
                         f"(n={int(down['Scored N'])}).")

    actionable = weak_strategy[weak_strategy["Evidence-based Lean"].str.startswith(("Pay", "Take"))]
    for _, row in actionable.iterrows():
        lines.append(f"- Weak {row['Position']} slates: {row['Evidence-based Lean'].lower()} "
                     f"(pay-up lift {row['Weak Pay-up Lift']:+.2f} across "
                     f"{int(row['Weak Slates'])} scored weak slates).")

    for position in ("SS", "OF"):
        match = depth_strategy[(depth_strategy["Position"] == position) &
                               depth_strategy["Depth"].isin(["Shallow", "Deep"]) &
                               (depth_strategy["Scored Slates"] >= 2)]
        for _, row in match.iterrows():
            lines.append(f"- {position} when {row['Depth'].lower()}: {row['Strategy Lean'].lower()} "
                         f"(pay-up lift {row['Pay-up Lift']:+.2f}, marginal value "
                         f"{row['Marginal Pts/$1K']:.2f} points/$1K across "
                         f"{int(row['Scored Slates'])} scored slates).")
    return lines or ["- There are not yet enough scored observations for a stable finding."]


def render_report(start, end, observations, coverage, tiers, origins, positions, payup,
                  depth_days, depth_strategy, correlations_table, change_all,
                  pitcher_changes, weak_days, weak_strategy, min_n=10):
    scored = int(observations["Actual"].notna().sum()) if not observations.empty else 0
    canonical = canonical_observations(observations)
    scored_days = sorted(observations.loc[observations["Actual"].notna(), "Date"].unique())
    missing = coverage[(coverage.get("Status") == "loaded") & (coverage.get("Scored", 0) == 0)] \
        if not coverage.empty else pd.DataFrame()
    weak = weak_days[weak_days["Weak"]].copy() if not weak_days.empty else weak_days

    lines = [
        f"# DFS price analysis: {start} through {end}", "",
        "## Coverage", "",
        f"The archive contains **{len(observations):,} player-slate price rows** and "
        f"**{len(canonical):,} unique player-days**. Actual DK points are available for "
        f"**{scored:,} player-slate rows** across **{len(scored_days)} dates**.", "",
        "Missing result exports are left blank, not scored as zero. Results-backed fallback "
        "rows are used only when a date has no archived board. Multi-position hitters count "
        "in every DK slot they were eligible to fill. This is descriptive evidence from a "
        "short window, not a causal rule or a projection.", "",
    ]
    if not missing.empty:
        labels = ", ".join(f"{r.Date} {r.Slate}" for r in missing.itertuples())
        lines += [f"No outcome export was available for: **{labels}**.", ""]

    lines += ["## What the window says", ""]
    lines += _finding_lines(tiers, origins, payup, correlations_table, pitcher_changes,
                            weak_strategy, depth_strategy, min_n=min_n)
    lines += ["", "Treat differences smaller than the reported 95% margin as noise. A tier "
              "with fewer than the minimum sample is shown for completeness but is not used "
              "for the headline findings.", ""]

    lines += ["## Where performances came from", "",
              _markdown_table(origins), "",
              "Hitter tiers are Elite 20+, Solid 10-19.99, Useful 5-9.99, and Low under 5. "
              "Pitcher tiers are Elite 25+, Solid 15-24.99, Useful 10-14.99, and Low under 10. "
              "The salary percentiles answer where each kind of result typically originated.", "",
              "## Empirically discovered salary tiers", "",
              _markdown_table(tiers), "",
              "These cut points are selected by a shallow regression tree to separate actual "
              "scores, with at least 12% of that player type in every tier. They are observed DK "
              "salary bounds—not arbitrary $1,000 buckets. `Elite%` and `Solid+%` use the outcome "
              "definitions above; `Low%` is under 5 for hitters and under 10 for pitchers. "
              "The breakpoints are fit and described on this same short sample, so treat them "
              "as exploratory until they persist in a longer or walk-forward window.", ""]

    lines += ["## What to pay by hitter position", "",
              _markdown_table(positions), "",
              "`Pay-up Cut` is that position's 75th-percentile salary. `Pay-up Lift` compares "
              "actual points above that cut with the rest of the position. `Productive Median $` "
              "is the median salary among 10+ point outcomes; it is a descriptive target, not an "
              "optimizer constraint.", "",
              "## What paying up actually bought", "", _markdown_table(payup), "",
              "`Marginal Pts/$1K` divides the point lift by the extra average salary. `Value "
              "Retained%` compares pay-up points/$1K with the cheaper pool: 100% means paying up "
              "added points without sacrificing efficiency.", "",
              "## Position depth and flexibility", "", _markdown_table(depth_strategy), "",
              "Depth counts players in the upper two empirically discovered hitter salary tiers. "
              "It combines the count and share within each position so large slates are not "
              "automatically called deep. The slate-level audit is in `position_depth.csv`.", ""]

    lines += ["## Weak-position slates", "",
              _markdown_table(weak, ["Date", "Slate", "Position", "Players", "Median Price",
                                     "Salary Index", "Salary Percentile", "Avg Pts", "Pay-up Lift"]),
              "", _markdown_table(weak_strategy), "",
              "A position is flagged weak when its slate median price is in the bottom quartile "
              "of that position's archived slates. `Salary Index` is 100 at the historical median.", ""]

    lines += ["## Price movement", "", "### All players", "", _markdown_table(change_all), "",
              "### Pitchers, start to start", "", _markdown_table(pitcher_changes), "",
              "The detailed start log—including previous salary and price change—is in "
              "`pitcher_changes.csv`. The repricing correlation compares a pitcher's prior "
              "actual score with the salary change on his next archived start.", ""]

    lines += ["## Correlations", "", _markdown_table(correlations_table), "",
              "Spearman correlation measures rank association and is less sensitive to one "
              "40-point game than Pearson correlation. Correlation does not show that a price "
              "change caused the next result.", ""]
    return "\n".join(lines)


def write_analysis(start, end, observations, coverage, output_root=ANALYSIS_ROOT, min_n=10):
    summary, detail = player_history(observations)
    tiers = price_tiers(observations)
    origins = performance_origins(observations)
    positions = position_summary(observations)
    payup = position_payup(observations)
    depth_days, depth_strategy = position_depth(observations, tiers)
    weak_days, weak_strategy = position_days(observations)
    all_changes = change_summary(detail)
    pitcher_changes = change_summary(detail, player_type="P")
    corr, pitcher_detail = correlations(observations, detail)

    directory = os.path.join(output_root, f"price_{start}_{end}")
    os.makedirs(directory, exist_ok=True)
    outputs = {
        "observations.csv": observations,
        "coverage.csv": coverage,
        "player_history.csv": summary,
        "player_changes.csv": detail,
        "price_tiers.csv": tiers,
        "performance_origins.csv": origins,
        "positions.csv": positions,
        "position_payup.csv": payup,
        "position_depth.csv": depth_days,
        "position_depth_strategy.csv": depth_strategy,
        "position_days.csv": weak_days,
        "weak_position_strategy.csv": weak_strategy,
        "pitcher_changes.csv": pitcher_detail,
        "correlations.csv": corr,
    }
    for name, frame in outputs.items():
        frame.to_csv(os.path.join(directory, name), index=False)

    # Full-lineup opportunity-cost analysis is kept in its own module because it reuses
    # the MILP optimizer, but lands beside the price tables so one command refreshes the
    # complete decision system.
    from .position_strategy import write_strategy_outputs
    strategy_outputs, strategy_report = write_strategy_outputs(
        observations, directory, target_date=end)
    outputs.update(strategy_outputs)

    report = render_report(start, end, observations, coverage, tiers, origins, positions,
                           payup, depth_days, depth_strategy, corr, all_changes,
                           pitcher_changes, weak_days, weak_strategy, min_n=min_n)
    card = strategy_outputs.get("daily_strategy_card.csv", pd.DataFrame())
    paired = strategy_outputs.get("paired_strategy_summary.csv", pd.DataFrame())
    report += ("\n## Full-lineup opportunity cost\n\n" + _markdown_table(card) +
               "\n\nEach pay-up/value pair re-optimizes the other nine roster spots. The detailed "
               "lineups and salary redeployment are in `paired_strategy_comparison.csv`; "
               "walk-forward salary assignments are in `walkforward_tiers.csv`.\n\n" +
               "### Historical paired results\n\n" + _markdown_table(paired) + "\n")
    report_path = os.path.join(directory, "report.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report + "\n")
    return directory, outputs, report_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Analyze longitudinal DraftKings salary, position and actual-points history.")
    parser.add_argument("--from", dest="start", help="First slate date (YYYY-MM-DD).")
    parser.add_argument("--to", dest="end", help="Last slate date (YYYY-MM-DD).")
    parser.add_argument("--slate", help="Only one slate label, e.g. main or early.")
    parser.add_argument("--min-sample", type=int, default=10, metavar="N",
                        help="Minimum scored tier sample used in written findings (default 10).")
    parser.add_argument("--output-root", default=ANALYSIS_ROOT)
    args = parser.parse_args(argv)

    observations, coverage = load_observations(args.start, args.end, slate=args.slate)
    if observations.empty:
        parser.error("no archived boards matched that date/slate range")
    start = args.start or str(observations["Date"].min())
    end = args.end or str(observations["Date"].max())
    directory, outputs, report_path = write_analysis(
        start, end, observations, coverage, output_root=args.output_root,
        min_n=args.min_sample)
    scored = int(observations["Actual"].notna().sum())
    print(f"Analyzed {len(observations):,} player-slate rows ({scored:,} with actual points).")
    print(f"Report: {report_path}")
    print(f"Tables: {directory} ({len(outputs)} CSV files)")


if __name__ == "__main__":
    main()
