"""Audit salary-tier usage in real DraftKings contest lineups.

The unit of comparison is an opportunity set: every player in a learned salary tier,
then every real lineup using that tier.  Bands are learned strictly from scored dates
before the contest, so winner and top-1% results never define their own salary tier.

Run with::

    python -m dfs.tier_audit --from 2026-07-27 --to 2026-08-08
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd

from .field import parse_contest_lineup
from .price_analysis import (
    ANALYSIS_ROOT,
    POSITION_ORDER,
    canonical_observations,
    explode_positions,
    load_observations,
    price_tiers,
)
from .results import RESULTS_DIR, contest_night, list_contests
from .salaries import normalize_name


COHORTS = (
    ("Field", "Field"),
    ("Top 20%", "Top 20%"),
    ("Top 1%", "Top 1%"),
    ("Winner", "Winner"),
)
SLOTS_PER_LINEUP = {"P": 2, "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3}


def primary_contests(start=None, end=None, directory=RESULTS_DIR):
    """Largest-field standings export for every dated slate."""
    best = {}
    for contest in list_contests(directory):
        night, slate = contest_night(contest)
        if not night or not slate:
            continue
        if start and str(night) < str(start):
            continue
        if end and str(night) > str(end):
            continue
        key = (str(night), str(slate))
        if key not in best or contest["entries"] > best[key]["entries"]:
            best[key] = contest
    return best


def walkforward_definitions(observations, night, min_hitters=50, min_pitchers=10):
    """Empirical hitter and pitcher bands from strictly earlier scored slates."""
    frame = canonical_observations(observations)
    prior = frame[(frame["Date"] < str(night)) & frame["Actual"].notna()]
    learned = {}
    for kind, minimum in (("H", min_hitters), ("P", min_pitchers)):
        group = prior[prior["Type"].eq(kind)]
        if len(group) < minimum:
            learned[kind] = pd.DataFrame()
            continue
        tiers = price_tiers(group)
        learned[kind] = tiers[tiers["Type"].eq(kind)].sort_values("Tier").reset_index(drop=True)
    return learned


def assign_tier(salary, definitions):
    if definitions is None or definitions.empty or pd.isna(salary):
        return None
    for _, tier in definitions.iterrows():
        if float(salary) <= float(tier["Max Salary"]):
            return int(tier["Tier"])
    return int(definitions.iloc[-1]["Tier"])


def _tier_metadata(tier, definitions):
    if tier is None or definitions is None or definitions.empty:
        return {"Tier": np.nan, "Tier Label": "Untrained", "Learned Price Band": ""}
    row = definitions[definitions["Tier"].eq(int(tier))].iloc[0]
    return {
        "Tier": int(tier),
        "Tier Label": f"T{int(tier)} {row['Observed Price Band']}",
        "Learned Price Band": row["Observed Price Band"],
        "Historical Avg Pts": row["Avg Pts"],
        "Historical Pts/$1K": row["Pts/$1K"],
        "Historical Elite%": row["Elite%"],
        "Historical Solid+%": row["Solid+%"],
    }


def read_ranked_entries(path):
    """Complete ten-player contest entries with inclusive rank-based cohort flags."""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [str(column).strip() for column in frame.columns]
    required = {"Rank", "Points", "Lineup"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    frame["Rank"] = pd.to_numeric(frame["Rank"], errors="coerce")
    frame["Points"] = pd.to_numeric(frame["Points"], errors="coerce")
    frame = frame[frame["Rank"].notna() & frame["Points"].notna()].copy()
    contest_entries = len(frame)
    frame["Parsed"] = frame["Lineup"].map(parse_contest_lineup)
    frame = frame[frame["Parsed"].map(len).eq(10)].copy().reset_index(drop=True)
    if frame.empty:
        return frame
    entries = contest_entries
    if "EntryId" in frame.columns:
        frame["Entry Key"] = frame["EntryId"].astype(str)
    else:
        frame["Entry Key"] = pd.Series(range(len(frame)), index=frame.index).astype(str)
    frame["Field"] = True
    frame["Top 20%"] = frame["Rank"].le(max(1, math.ceil(entries * 0.20)))
    frame["Top 1%"] = frame["Rank"].le(max(1, math.ceil(entries * 0.01)))
    frame["Winner"] = frame["Rank"].eq(frame["Rank"].min())
    frame.attrs["contest_entries"] = contest_entries
    return frame


def _board_for_slate(observations, night, slate, definitions):
    board = observations[(observations["Date"].eq(str(night))) &
                         observations["Slate"].eq(str(slate))].copy()
    if board.empty:
        return board, pd.DataFrame()
    board["PlayerKey"] = board["Name"].map(normalize_name)
    board["Tier"] = board.apply(
        lambda row: assign_tier(row["Salary"], definitions.get(row["Type"])), axis=1)
    position_board = explode_positions(board)
    position_board = position_board[position_board["Position"].isin(POSITION_ORDER)].copy()
    return board.drop_duplicates("PlayerKey"), position_board


def contest_slot_rows(observations, contests=None, start=None, end=None,
                      min_hitters=50, min_pitchers=10):
    """One row per real lineup slot, enriched with salary and walk-forward tier."""
    contests = contests or primary_contests(start, end)
    rows, coverage = [], []
    for (night, slate), contest in sorted(contests.items()):
        definitions = walkforward_definitions(
            observations, night, min_hitters=min_hitters, min_pitchers=min_pitchers)
        board, _ = _board_for_slate(observations, night, slate, definitions)
        entries = read_ranked_entries(contest["path"])
        if entries.empty or board.empty:
            continue
        lookup = board.set_index("PlayerKey").to_dict("index")
        contest_id = os.path.splitext(os.path.basename(contest["path"]))[0]
        contest_entries = int(entries.attrs.get("contest_entries", len(entries)))
        matched = 0
        for _, entry in entries.iterrows():
            for slot, raw_name in entry["Parsed"]:
                key = normalize_name(raw_name)
                player = lookup.get(key, {})
                kind = "P" if slot == "P" else "H"
                tier = player.get("Tier")
                metadata = _tier_metadata(tier, definitions.get(kind))
                # A single learned band has no meaningful "premium" distinction. Keep its
                # tier label for exposure, but do not call every player in it premium.
                max_type_tier = (int(definitions[kind]["Tier"].max())
                                 if len(definitions[kind]) >= 2 else np.nan)
                is_matched = bool(player)
                matched += int(is_matched)
                rows.append({
                    "Date": night, "Slate": slate, "Contest": contest_id,
                    "Contest Entries": contest_entries, "Entry Key": entry["Entry Key"],
                    "Entry Name": entry.get("EntryName", ""),
                    "Rank": int(entry["Rank"]), "Points": float(entry["Points"]),
                    "Field": True, "Top 20%": bool(entry["Top 20%"]),
                    "Top 1%": bool(entry["Top 1%"]), "Winner": bool(entry["Winner"]),
                    "Position": slot, "Name": player.get("Name", raw_name),
                    "PlayerKey": key, "Type": kind, "Matched": is_matched,
                    "Salary": player.get("Salary", np.nan),
                    "Actual": player.get("Actual", np.nan),
                    "Max Type Tier": max_type_tier,
                    **metadata,
                })
        total_slots = len(entries) * 10
        coverage.append({
            "Date": night, "Slate": slate, "Contest": contest_id,
            "Entries": contest_entries, "Complete Lineups": len(entries),
            "Slots": total_slots, "Matched Slots": matched,
            "Match%": round(matched / total_slots * 100, 2) if total_slots else np.nan,
            "Hitter Tiers": len(definitions["H"]),
            "Pitcher Tiers": len(definitions["P"]),
        })
    return pd.DataFrame(rows), pd.DataFrame(coverage)


def _cohort_metrics(group, position, tier, cohort, entries_total):
    cohort_slots = group[group[cohort]]
    cohort_entries = cohort_slots["Entry Key"].nunique()
    tier_slots = cohort_slots[cohort_slots["Tier"].eq(tier)]
    used_entries = tier_slots["Entry Key"].nunique()
    seats = cohort_entries * SLOTS_PER_LINEUP[position]
    return {
        f"{cohort} Entries": cohort_entries,
        f"{cohort} Tier Slots": len(tier_slots),
        f"{cohort} Seat Share%": round(len(tier_slots) / seats * 100, 2) if seats else np.nan,
        f"{cohort} Lineup Exposure%": round(used_entries / cohort_entries * 100, 2)
                                      if cohort_entries else np.nan,
    }


def tier_cohort_summary(slot_rows, observations):
    """Tier availability versus field, top-20%, top-1%, and winner usage."""
    if slot_rows.empty:
        return pd.DataFrame()
    rows = []
    slate_cache = {}
    for (night, slate, contest, position), group in slot_rows.groupby(
            ["Date", "Slate", "Contest", "Position"]):
        slate_key = (night, slate)
        if slate_key not in slate_cache:
            definitions = walkforward_definitions(observations, night)
            _, available = _board_for_slate(observations, night, slate, definitions)
            slate_cache[slate_key] = (definitions, available)
        definitions, available = slate_cache[slate_key]
        available = available[available["Position"].eq(position)]
        tiers = sorted(set(group["Tier"].dropna().astype(int)) |
                       set(available["Tier"].dropna().astype(int)))
        for tier in tiers:
            definition = definitions["P" if position == "P" else "H"]
            metadata = _tier_metadata(tier, definition)
            availability_n = available[available["Tier"].eq(tier)]["Name"].nunique()
            pool_n = available["Name"].nunique()
            row = {
                "Date": night, "Slate": slate, "Contest": contest,
                "Contest Entries": group["Entry Key"].nunique(), "Position": position,
                **metadata,
                "Available N": availability_n, "Position Pool N": pool_n,
                "Availability Share%": round(availability_n / pool_n * 100, 2)
                                       if pool_n else np.nan,
            }
            for cohort, _ in COHORTS:
                row.update(_cohort_metrics(
                    group, position, tier, cohort, group["Entry Key"].nunique()))
            field = row["Field Lineup Exposure%"]
            top = row["Top 1% Lineup Exposure%"]
            availability = row["Availability Share%"]
            row["Top1 Leverage pp"] = round(top - field, 2) if pd.notna(top) else np.nan
            row["Top1 vs Field Index"] = round(top / field, 2) if field else np.nan
            row["Field Selection Index"] = round(
                row["Field Seat Share%"] / availability, 2) if availability else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def player_cohort_summary(slot_rows):
    """Player-level result so the members within a tier remain visible."""
    if slot_rows.empty:
        return pd.DataFrame()
    rows = []
    keys = ["Date", "Slate", "Contest", "Position", "Tier", "Tier Label", "Name", "Salary"]
    matched = slot_rows[slot_rows["Matched"] & slot_rows["Tier"].notna()].copy()
    contest_keys = ["Date", "Slate", "Contest"]
    cohort_sizes = {
        cohort: matched[matched[cohort]].groupby(contest_keys)["Entry Key"].nunique().to_dict()
        for cohort, _ in COHORTS
    }
    for key, player in matched.groupby(keys, dropna=False):
        contest_key = key[:3]
        row = dict(zip(keys, key))
        row["Contest Entries"] = cohort_sizes["Field"].get(contest_key, 0)
        for cohort, _ in COHORTS:
            cohort_entries = cohort_sizes[cohort].get(contest_key, 0)
            used = player[player[cohort]]["Entry Key"].nunique()
            row[f"{cohort} Exposure%"] = round(used / cohort_entries * 100, 2) \
                if cohort_entries else np.nan
        row["Top1 Leverage pp"] = round(
            row["Top 1% Exposure%"] - row["Field Exposure%"], 2)
        row["Top1 vs Field Index"] = round(
            row["Top 1% Exposure%"] / row["Field Exposure%"], 2) \
            if row["Field Exposure%"] else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def entry_constructions(slot_rows):
    """One tier signature and premium-position pattern per real entry."""
    if slot_rows.empty:
        return pd.DataFrame()
    keys = ["Date", "Slate", "Contest", "Entry Key"]
    work = slot_rows.copy()
    numeric_tier = pd.to_numeric(work["Tier"], errors="coerce")
    work["_tier_order"] = numeric_tier.fillna(999)
    work["_tier_token"] = numeric_tier.map(
        lambda value: f"T{int(value)}" if pd.notna(value) else "U")
    work["_premium"] = numeric_tier.notna() & numeric_tier.eq(
        pd.to_numeric(work["Max Type Tier"], errors="coerce"))
    work = work.sort_values(keys + ["Position", "_tier_order"])

    tokens = work.groupby(keys + ["Position"], sort=False)["_tier_token"].agg("+".join)
    tokens = tokens.unstack("Position").reindex(columns=POSITION_ORDER).fillna("U")
    premium = work[work["_premium"]].groupby(keys + ["Position"]).size().unstack(
        "Position", fill_value=0).reindex(columns=POSITION_ORDER, fill_value=0)
    base = work.groupby(keys, sort=False).agg({
        "Entry Name": "first", "Rank": "first", "Points": "first",
        "Top 20%": "first", "Top 1%": "first", "Winner": "first",
    })
    result = base.join(tokens, how="left")
    premium = premium.reindex(result.index, fill_value=0)

    signature = pd.Series("", index=result.index, dtype=object)
    pattern = pd.Series("", index=result.index, dtype=object)
    for position in POSITION_ORDER:
        separator = "" if position == "P" else " | "
        signature = signature + separator + position + ":" + result[position].astype(str)
        count = premium[position].astype(int)
        label = np.where(count.gt(1), position + "x" + count.astype(str), position)
        piece = np.where(count.gt(0), label, "")
        pattern = pattern + np.where((pattern.str.len().gt(0)) & (piece != ""), "+", "") + piece

    result["Tier Signature"] = signature
    result["Premium Position Pattern"] = pattern.replace("", "None")
    result["Premium Pitcher Slots"] = premium["P"].astype(int)
    result["Premium Hitter Slots"] = premium.drop(columns="P").sum(axis=1).astype(int)
    return result.drop(columns=list(POSITION_ORDER)).reset_index()


def construction_summary(entries):
    if entries.empty:
        return pd.DataFrame()
    rows = []
    keys = ["Date", "Slate", "Contest", "Premium Position Pattern",
            "Premium Hitter Slots", "Premium Pitcher Slots"]
    contest_keys = ["Date", "Slate", "Contest"]
    cohort_sizes = {"Field": entries.groupby(contest_keys).size().to_dict()}
    cohort_sizes.update({
        cohort: entries[entries[cohort]].groupby(contest_keys).size().to_dict()
        for cohort in ("Top 20%", "Top 1%", "Winner")
    })
    for key, pattern in entries.groupby(keys):
        contest_key = key[:3]
        row = dict(zip(keys, key))
        row["Avg Points"] = round(pattern["Points"].mean(), 2)
        for cohort in ("Field", "Top 20%", "Top 1%", "Winner"):
            count = len(pattern) if cohort == "Field" else int(pattern[cohort].sum())
            row[f"{cohort} N"] = count
            cohort_n = cohort_sizes[cohort].get(contest_key, 0)
            row[f"{cohort} Share%"] = round(count / cohort_n * 100, 2) \
                if cohort_n else np.nan
        row["Top1 Leverage pp"] = round(
            row["Top 1% Share%"] - row["Field Share%"], 2)
        row["Top1 vs Field Index"] = round(
            row["Top 1% Share%"] / row["Field Share%"], 2) if row["Field Share%"] else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def historical_tier_summary(cohorts):
    """Slate-weighted historical tier usage; one contest cannot dominate by field size."""
    if cohorts.empty:
        return pd.DataFrame()
    rows = []
    for (position, tier), group in cohorts.groupby(["Position", "Tier"]):
        rows.append({
            "Position": position, "Tier": int(tier), "Slates": len(group),
            "Avg Availability N": round(group["Available N"].mean(), 1),
            "Avg Availability Share%": round(group["Availability Share%"].mean(), 1),
            "Avg Field Lineup Exposure%": round(group["Field Lineup Exposure%"].mean(), 1),
            "Avg Top1 Lineup Exposure%": round(group["Top 1% Lineup Exposure%"].mean(), 1),
            "Avg Top1 Leverage pp": round(group["Top1 Leverage pp"].mean(), 1),
            "Avg Top1 vs Field Index": round(group["Top1 vs Field Index"].mean(), 2),
            "Winner Used% of Slates": round(group["Winner Lineup Exposure%"].gt(0).mean() * 100, 1),
            "Avg Historical Pts": round(group["Historical Avg Pts"].mean(), 2),
            "Avg Historical Pts/$1K": round(group["Historical Pts/$1K"].mean(), 2),
        })
    return pd.DataFrame(rows)


def premium_count_summary(entries):
    """Pooled and slate-average results by number of premium hitter/pitcher slots."""
    if entries.empty:
        return pd.DataFrame()
    contest_keys = ["Date", "Slate", "Contest"]
    contests = entries[contest_keys].drop_duplicates()
    total_contests = len(contests)
    rows = []
    combinations = entries[["Premium Hitter Slots", "Premium Pitcher Slots"]].drop_duplicates()
    for _, combination in combinations.sort_values(
            ["Premium Hitter Slots", "Premium Pitcher Slots"]).iterrows():
        hitters = int(combination["Premium Hitter Slots"])
        pitchers = int(combination["Premium Pitcher Slots"])
        slate_shares = []
        top_shares = []
        field_n = top_n = winner_n = 0
        points = []
        appeared = winner_slates = 0
        for contest_key in contests.itertuples(index=False, name=None):
            contest = entries[
                entries["Date"].eq(contest_key[0]) &
                entries["Slate"].eq(contest_key[1]) &
                entries["Contest"].eq(contest_key[2])]
            selected = contest[
                contest["Premium Hitter Slots"].eq(hitters) &
                contest["Premium Pitcher Slots"].eq(pitchers)]
            top = contest[contest["Top 1%"]]
            slate_shares.append(len(selected) / len(contest) * 100 if len(contest) else 0)
            top_selected = selected[selected["Top 1%"]]
            top_shares.append(len(top_selected) / len(top) * 100 if len(top) else 0)
            field_n += len(selected)
            top_n += len(top_selected)
            winner_n += int(selected["Winner"].sum())
            appeared += int(len(selected) > 0)
            winner_slates += int(selected["Winner"].any())
            points.extend(selected["Points"].tolist())
        total_field = len(entries)
        total_top = int(entries["Top 1%"].sum())
        field_share = field_n / total_field * 100 if total_field else np.nan
        top_share = top_n / total_top * 100 if total_top else np.nan
        rows.append({
            "Premium Hitter Slots": hitters, "Premium Pitcher Slots": pitchers,
            "Construction": f"{hitters} premium hitters + {pitchers} premium pitchers",
            "Contests": total_contests, "Contests Used": appeared,
            "Field N": field_n, "Pooled Field Share%": round(field_share, 2),
            "Pooled Top1 N": top_n, "Pooled Top1 Share%": round(top_share, 2),
            "Pooled Top1 Leverage pp": round(top_share - field_share, 2),
            "Top1 vs Field Index": round(top_share / field_share, 2) if field_share else np.nan,
            "Avg Slate Field Share%": round(float(np.mean(slate_shares)), 2),
            "Avg Slate Top1 Share%": round(float(np.mean(top_shares)), 2),
            "Winner Lineups": winner_n,
            "Winner Used% of Slates": round(winner_slates / total_contests * 100, 1),
            "Avg Points": round(float(np.mean(points)), 2) if points else np.nan,
        })
    return pd.DataFrame(rows)


def premium_pattern_summary(entries):
    """Historical leverage for the exact positions filled from premium tiers."""
    if entries.empty:
        return pd.DataFrame()
    entries = entries.copy()
    entries["Premium Position Pattern"] = entries["Premium Position Pattern"].fillna("None")
    total_field = len(entries)
    total_top = int(entries["Top 1%"].sum())
    total_contests = entries[["Date", "Slate", "Contest"]].drop_duplicates().shape[0]
    rows = []
    for pattern, group in entries.groupby("Premium Position Pattern"):
        top_n = int(group["Top 1%"].sum())
        field_share = len(group) / total_field * 100
        top_share = top_n / total_top * 100 if total_top else np.nan
        winner_slates = group[group["Winner"]][["Date", "Slate", "Contest"]].drop_duplicates().shape[0]
        rows.append({
            "Premium Position Pattern": pattern,
            "Contests Used": group[["Date", "Slate", "Contest"]].drop_duplicates().shape[0],
            "Field N": len(group), "Pooled Field Share%": round(field_share, 2),
            "Top1 N": top_n, "Pooled Top1 Share%": round(top_share, 2),
            "Top1 Leverage pp": round(top_share - field_share, 2),
            "Top1 vs Field Index": round(top_share / field_share, 2) if field_share else np.nan,
            "Winner Lineups": int(group["Winner"].sum()),
            "Winner Used% of Slates": round(winner_slates / total_contests * 100, 1),
            "Avg Points": round(group["Points"].mean(), 2),
        })
    return pd.DataFrame(rows).sort_values(
        ["Top1 N", "Top1 Leverage pp"], ascending=False).reset_index(drop=True)


def _markdown_table(frame, columns=None, limit=30):
    if frame.empty:
        return "_No rows._"
    view = frame[list(columns) if columns else list(frame.columns)].head(limit).replace({np.nan: ""})
    headers = list(view.columns)
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in view.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def write_tier_audit_report(directory, start, end, coverage, winners, historical,
                            premium_counts, premium_patterns, minimum_match_pct=90.0):
    winner_view = winners[["Date", "Slate", "Position", "Name", "Salary", "Tier Label",
                           "Points"]].sort_values(["Date", "Slate", "Position"])
    history_view = historical.sort_values(
        ["Position", "Avg Top1 Leverage pp"], ascending=[True, False])
    count_view = premium_counts[premium_counts["Field N"].ge(100)].sort_values(
        "Pooled Top1 Leverage pp", ascending=False)
    pattern_view = premium_patterns[premium_patterns["Field N"].ge(150)].sort_values(
        "Top1 Leverage pp", ascending=False)
    report = [
        f"# Contest tier audit: {start} to {end}", "",
        "Largest-field contest per slate. Top 20% is a cash proxy because standings exports "
        "do not contain the payout table. Salary bands are learned only from earlier scored "
        f"slates. Strategic rates require at least {minimum_match_pct:.0f}% of real roster slots "
        "to match the archived board; excluded contests remain in the coverage and raw-slot files. "
        "Exact construction tables additionally require all ten lineup slots to have known tiers.",
        "", "## Coverage", "",
        _markdown_table(coverage), "", "## Winning lineup tiers", "",
        _markdown_table(winner_view, limit=100), "", "## Historical tier leverage", "",
        _markdown_table(history_view, columns=[
            "Position", "Tier", "Slates", "Avg Field Lineup Exposure%",
            "Avg Top1 Lineup Exposure%", "Avg Top1 Leverage pp",
            "Winner Used% of Slates", "Avg Historical Pts/$1K"], limit=50), "",
        "## Premium-slot construction", "",
        _markdown_table(count_view, columns=[
            "Construction", "Field N", "Pooled Field Share%", "Pooled Top1 N",
            "Pooled Top1 Share%", "Pooled Top1 Leverage pp", "Top1 vs Field Index",
            "Winner Used% of Slates"], limit=30), "",
        "## Premium-position patterns", "",
        _markdown_table(pattern_view, columns=[
            "Premium Position Pattern", "Contests Used", "Field N", "Pooled Field Share%",
            "Top1 N", "Pooled Top1 Share%", "Top1 Leverage pp",
            "Top1 vs Field Index"], limit=30), "",
    ]
    report_path = os.path.join(directory, "tier_audit.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report))
    return report_path


def write_tier_audit(observations, directory, start=None, end=None,
                     results_directory=RESULTS_DIR, minimum_match_pct=90.0):
    contests = primary_contests(start, end, results_directory)
    slots, coverage = contest_slot_rows(
        observations, contests=contests, start=start, end=end)
    coverage["Audited"] = (coverage["Match%"].ge(minimum_match_pct) &
                           coverage["Hitter Tiers"].ge(2))
    coverage["Audit Status"] = np.select(
        [coverage["Hitter Tiers"].lt(2), coverage["Match%"].lt(minimum_match_pct)],
        ["Warmup: no prior tiers", f"Excluded: match < {minimum_match_pct:.0f}%"],
        default="Included")
    valid = coverage[coverage["Audited"]][["Date", "Slate", "Contest"]].copy()
    valid["Audited"] = True
    slots = slots.merge(valid, on=["Date", "Slate", "Contest"], how="left")
    slots["Audited"] = slots["Audited"].fillna(False).astype(bool)
    audited_slots = slots[slots["Audited"]].copy()
    cohorts = tier_cohort_summary(audited_slots, observations)
    players = player_cohort_summary(audited_slots)
    entry_keys = ["Date", "Slate", "Contest", "Entry Key"]
    tier_complete = audited_slots.groupby(entry_keys).filter(
        lambda lineup: lineup["Matched"].all() and lineup["Tier"].notna().all())
    entries = entry_constructions(tier_complete)
    complete_counts = entries.groupby(["Date", "Slate", "Contest"]).size().rename(
        "Tier-complete Lineups").reset_index()
    coverage = coverage.merge(
        complete_counts, on=["Date", "Slate", "Contest"], how="left")
    coverage["Tier-complete Lineups"] = coverage["Tier-complete Lineups"].fillna(0).astype(int)
    patterns = construction_summary(entries)
    premium_counts = premium_count_summary(entries)
    premium_patterns = premium_pattern_summary(entries)
    historical = historical_tier_summary(cohorts)
    winners = audited_slots[audited_slots["Winner"]].copy()
    outputs = {
        "tier_audit_coverage.csv": coverage,
        "contest_lineup_tiers.csv": slots,
        "tier_cohort_summary.csv": cohorts,
        "player_cohort_summary.csv": players,
        "winning_lineup_tiers.csv": winners,
        "entry_tier_constructions.csv": entries,
        "construction_patterns.csv": patterns,
        "premium_count_summary.csv": premium_counts,
        "premium_pattern_summary.csv": premium_patterns,
        "tier_historical_summary.csv": historical,
    }
    os.makedirs(directory, exist_ok=True)
    for filename, frame in outputs.items():
        frame.to_csv(os.path.join(directory, filename), index=False)

    report_path = write_tier_audit_report(
        directory, start or observations["Date"].min(), end or observations["Date"].max(),
        coverage, winners, historical, premium_counts, premium_patterns,
        minimum_match_pct=minimum_match_pct)
    return outputs, report_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Audit real DK field, top-20%, top-1%, and winning lineups by salary tier.")
    parser.add_argument("--from", dest="start")
    parser.add_argument("--to", dest="end")
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--output-root", default=ANALYSIS_ROOT)
    parser.add_argument("--min-match", type=float, default=90.0,
                        help="Minimum matched roster-slot percentage for strategic rates.")
    args = parser.parse_args(argv)

    # Earlier scored boards are needed to define the first requested day's tiers.
    observations, _ = load_observations(None, args.end)
    if observations.empty:
        parser.error("no archived boards matched the requested date range")
    start = args.start or str(observations["Date"].min())
    end = args.end or str(observations["Date"].max())
    directory = os.path.join(args.output_root, f"price_{start}_{end}")
    outputs, report = write_tier_audit(
        observations, directory, start=start, end=end, results_directory=args.results_dir,
        minimum_match_pct=args.min_match)
    print(f"Tier audit: {report}")
    for filename, frame in outputs.items():
        print(f"  {filename}: {len(frame):,} rows")


if __name__ == "__main__":
    main()
