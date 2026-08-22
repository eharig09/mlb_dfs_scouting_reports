"""Is opener detection actually finding openers, and is the follower worth rostering?

Two questions, and they are separate. The first is whether `build_opener_index` identifies
the right pitchers -- a detection bug is silent, because a wrong opener call just produces a
slightly odd-looking report. The second is the DFS question: the arm behind an opener is
priced as a reliever and rostered by nobody, so if he reliably throws four innings and can
win the game, he is a structural edge rather than a curiosity.

Uses the same local extract as the arsenal study: statcast for *who pitched in what order*
and the box scores for *how long they actually went*, which is exact rather than inferred
from which innings a pitcher appears in.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OPENER_MAX_INNINGS = 2.0    # matches scouting_report
BULK_MIN_INNINGS = 3.0


def load(data_dir, seasons=(2024, 2025, 2026)):
    pitches = pd.concat(
        [pd.read_parquet(f"{data_dir}/pitches_{s}.parquet",
                         columns=["game_pk", "game_date", "inning", "inning_topbot",
                                  "at_bat_number", "pitch_number", "pitcher",
                                  "home_team", "away_team"])
         for s in seasons], ignore_index=True)
    box = pd.concat([pd.read_parquet(f"{data_dir}/box_{s}.parquet") for s in seasons],
                    ignore_index=True)
    return pitches, box


def appearance_order(pitches):
    """Every pitcher's appearance in every half-game, in the order he actually appeared.

    Ordering by inning alone is not enough: two pitchers can work the same inning, and the
    tie is then broken by whatever order the rows happen to sit in. `at_bat_number` is
    monotonic across the game, so the first plate appearance a pitcher faces orders them
    exactly.
    """
    # A pitcher is "fielding" for the half he throws in; inning_topbot names the batting side.
    first_pa = (pitches.groupby(["game_pk", "inning_topbot", "pitcher"], observed=True)
                ["at_bat_number"].min().reset_index(name="first_ab"))
    first_pa = first_pa.sort_values(["game_pk", "inning_topbot", "first_ab"])
    first_pa["slot"] = first_pa.groupby(["game_pk", "inning_topbot"], observed=True).cumcount()
    return first_pa


def opener_pairs(pitches, box):
    """One row per half-game: who started, who followed, and how long each went."""
    order = appearance_order(pitches)
    innings = (box[box["role"] == "P"][["game_pk", "player_id", "IP", "dk", "K", "ER", "W"]]
               .rename(columns={"player_id": "pitcher"}))
    order = order.merge(innings, on=["game_pk", "pitcher"], how="left")

    first = order[order["slot"] == 0].rename(columns={
        "pitcher": "first", "IP": "first_ip", "dk": "first_dk"})
    second = order[order["slot"] == 1].rename(columns={
        "pitcher": "bulk", "IP": "bulk_ip", "dk": "bulk_dk", "K": "bulk_k",
        "ER": "bulk_er", "W": "bulk_w"})
    pairs = first[["game_pk", "inning_topbot", "first", "first_ip", "first_dk"]].merge(
        second[["game_pk", "inning_topbot", "bulk", "bulk_ip", "bulk_dk", "bulk_k",
                "bulk_er", "bulk_w"]],
        on=["game_pk", "inning_topbot"], how="left")
    pairs["is_opener"] = (pairs["first_ip"].le(OPENER_MAX_INNINGS)
                          & pairs["bulk_ip"].ge(BULK_MIN_INNINGS))
    dates = pitches.groupby("game_pk", observed=True)["game_date"].first()
    pairs["game_date"] = pairs["game_pk"].map(dates)
    pairs["season"] = pd.to_datetime(pairs["game_date"]).dt.year
    return pairs


def shipped_index(pitches):
    """The shipped logic, reproduced: innings by `nunique`, ordering by inning only.

    Kept here so the two can be compared on the same games rather than argued about.
    """
    rows = []
    slim = pitches[["game_pk", "inning", "pitcher", "inning_topbot"]].dropna()
    for (game_pk, half), group in slim.groupby(["game_pk", "inning_topbot"], sort=False):
        group = group.sort_values("inning")          # not stable, and no tiebreak
        first = int(group.iloc[0]["pitcher"])
        first_inn = int(group.loc[group["pitcher"] == first, "inning"].nunique())
        others = group[group["pitcher"] != first]
        bulk, bulk_inn = None, 0
        if not others.empty:
            bulk = int(others.sort_values("inning").iloc[0]["pitcher"])
            bulk_inn = int(others.loc[others["pitcher"] == bulk, "inning"].nunique())
        rows.append({"game_pk": int(game_pk), "inning_topbot": str(half), "first": first,
                     "first_inn": first_inn, "bulk": bulk, "bulk_inn": bulk_inn,
                     "is_opener": bool(first_inn <= 2 and bulk_inn >= 3)})
    return pd.DataFrame(rows)


def main():
    data = sys.argv[1] if len(sys.argv) > 1 else "data"
    pitches, box = load(data)
    print(f"{len(pitches):,} pitches, {pitches['game_pk'].nunique():,} games\n")

    pairs = opener_pairs(pitches, box)
    truth = pairs.dropna(subset=["first_ip"])
    print("=== how many openers, by season (box-score innings) ===")
    for season, group in truth.groupby("season"):
        openers = int(group["is_opener"].sum())
        print(f"  {season}: {openers:4d} opener halves of {len(group):5d} "
              f"({openers / len(group) * 100:.1f}%)")

    print("\n=== the shipped detector vs the same games ===")
    shipped = shipped_index(pitches)
    merged = truth.merge(shipped[["game_pk", "inning_topbot", "first", "bulk", "is_opener"]],
                         on=["game_pk", "inning_topbot"], how="inner",
                         suffixes=("", "_shipped"))
    same_first = (merged["first"] == merged["first_shipped"]).mean() * 100
    both = merged.dropna(subset=["bulk", "bulk_shipped"])
    same_bulk = (both["bulk"] == both["bulk_shipped"]).mean() * 100
    agree = (merged["is_opener"] == merged["is_opener_shipped"]).mean() * 100
    print(f"  identifies the same starting pitcher: {same_first:.2f}%")
    print(f"  identifies the same follower:         {same_bulk:.2f}%")
    print(f"  agrees on is_opener:                  {agree:.2f}%")
    false_pos = int((merged["is_opener_shipped"] & ~merged["is_opener"]).sum())
    false_neg = int((~merged["is_opener_shipped"] & merged["is_opener"]).sum())
    print(f"  shipped says opener, box score says no: {false_pos}")
    print(f"  shipped misses a real opener:           {false_neg}")

    print("\n=== what the follower actually scores ===")
    followers = truth[truth["is_opener"]].dropna(subset=["bulk_dk"])
    regular = truth[~truth["is_opener"] & truth["first_ip"].ge(4)].dropna(subset=["first_dk"])
    for label, series, ip in (("follower behind an opener", followers["bulk_dk"], followers["bulk_ip"]),
                              ("conventional starter", regular["first_dk"], regular["first_ip"])):
        print(f"  {label:<26} n={len(series):5d}  mean {series.mean():5.2f}  "
              f"median {series.median():5.2f}  p90 {series.quantile(0.90):5.2f}  "
              f"20+ {(series >= 20).mean() * 100:4.1f}%   mean IP {ip.mean():.2f}")
    print(f"  follower win rate: {followers['bulk_w'].mean() * 100:.1f}%  "
          f"(conventional starter {regular.merge(box, left_on=['game_pk', 'first'], right_on=['game_pk', 'player_id'])['W'].mean() * 100:.1f}%)")

    print("\n=== can you know the follower in advance? ===")
    # For each opener appearance, was the follower the arm who most often followed that
    # opener *before* this game? Anything else is unrosterable.
    ordered = truth[truth["is_opener"]].dropna(subset=["bulk"]).sort_values("game_date")
    hits = total = 0
    history = {}
    for _, row in ordered.iterrows():
        key = int(row["first"])
        seen = history.get(key, {})
        if seen:
            predicted = max(seen, key=seen.get)
            hits += int(predicted == int(row["bulk"]))
            total += 1
        seen[int(row["bulk"])] = seen.get(int(row["bulk"]), 0) + 1
        history[key] = seen
    print(f"  most-frequent prior follower repeats: {hits}/{total} "
          f"({hits / total * 100:.1f}%)" if total else "  not enough history")


if __name__ == "__main__":
    main()
