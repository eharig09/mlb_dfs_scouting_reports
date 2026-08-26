"""Enumerating the stacks available inside one club, and pricing each of them.

A club is not one stack. A quarterback with four rosterable pass-catchers offers fourteen
distinct two- and three-man combinations, and they differ in price by thousands and in
captured target share by tens of points. "Stack the Eagles" is a decision not yet made.

## What the measurement says to build, and not to

**Rank inside the passing game is not the lever.** Over 2023-25 a club's WR2 correlates with
its quarterback as strongly as its WR1 — +0.359 against +0.367, and higher on the median —
while carrying about a quarter less target share. So nothing here privileges the WR1, and
the cheaper combination carrying the same correlation is the point of the page.

**Target concentration is not the mechanism either.** corr(team target HHI, QB-WR1 weekly
correlation) is +0.13 and not even monotonic by tercile. So captured target share is reported
as *what the stack owns*, which is a fact, and never as a quality score.

What is left is price and captured share, which is why those are the two axes.
"""

from itertools import combinations

import numpy as np
import pandas as pd
import streamlit as st

# Positions that share the quarterback's events. A running back is excluded on purpose: a
# rushing touchdown is a drive that did *not* end in a passing touchdown, so a back is at
# best uncorrelated with his own quarterback and on the goal line actively negative.
PASS_GAME = ("WR", "TE")

MAX_PARTNERS = 3


@st.cache_data(ttl="10m", max_entries=16, show_spinner=False)
def stack_combinations(board, team, sizes=(1, 2, 3), min_projection=0.0):
    """Every QB + n pass-catcher combination for one club, priced and scored.

    `board` must carry Name, Pos, Team, Salary and Proj. Returns one row per combination:
    who is in it, what it costs, what it projects, and the share of the club's projected
    targets it captures.

    The quarterback is whichever of the club's QBs projects highest — a slate rarely prices
    two real starters for one team, and enumerating both would double every row for a
    backup nobody will roster.
    """
    club = board[(board["Team"] == team)].copy()
    if club.empty:
        return pd.DataFrame()

    quarterbacks = club[club["Pos"] == "QB"].dropna(subset=["Proj"])
    if quarterbacks.empty:
        return pd.DataFrame()
    quarterback = quarterbacks.nlargest(1, "Proj").iloc[0]

    pass_game = club[club["Pos"].isin(PASS_GAME)].dropna(subset=["Proj", "Salary"])

    # **The denominator is the club's whole pass game, not the filtered shortlist.** Taking
    # it from the survivors of `min_projection` makes the share relative to a set the slider
    # moves, so raising the floor inflates every share and a stack of all three qualifiers
    # reads as capturing 100% of an offence that also throws to four other men. The share
    # has to mean the same thing at every filter setting or it means nothing.
    team_targets = pd.to_numeric(pass_game.reindex(columns=["Targets"])["Targets"],
                                 errors="coerce").sum()

    catchers = pass_game[pass_game["Proj"] >= min_projection]
    if catchers.empty:
        return pd.DataFrame()

    rows = []
    for size in sizes:
        if size > len(catchers):
            continue
        for chosen in combinations(catchers.itertuples(index=False), size):
            names = [p.Name for p in chosen]
            salary = float(quarterback["Salary"]) + sum(float(p.Salary) for p in chosen)
            projection = float(quarterback["Proj"]) + sum(float(p.Proj) for p in chosen)
            targets = sum(_number(getattr(p, "Targets", np.nan)) for p in chosen)
            rows.append({
                "Team": team,
                "QB": quarterback["Name"],
                "Partners": ", ".join(names),
                "Size": size + 1,
                "Salary": salary,
                "Proj": projection,
                "Value": projection / (salary / 1000.0) if salary else np.nan,
                "Captured targets": targets,
                "Captured share": (targets / team_targets) if team_targets else np.nan,
                "Positions": "/".join(sorted({p.Pos for p in chosen})),
                "_names": tuple(names),
            })
    frame = pd.DataFrame(rows)
    return frame.sort_values("Proj", ascending=False).reset_index(drop=True)


def _number(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if np.isnan(out) else out


def bring_back_candidates(board, team, opponent, limit=8):
    """The other side of the game, ranked — a stack's correlated hedge.

    Correlation with the *opposing* quarterback is what a bring-back buys: the shootout that
    lifts your stack lifts him too. Ranked on projection rather than on any correlation
    number, because the measured spread between a club's own receivers is small enough that
    price and volume decide it.
    """
    if not opponent:
        return pd.DataFrame()
    other = board[(board["Team"] == opponent) & board["Pos"].isin(PASS_GAME)]
    other = other.dropna(subset=["Proj"])
    columns = [c for c in ("Name", "Pos", "Team", "Salary", "Proj", "Value", "Targets")
               if c in other.columns]
    return other.nlargest(min(limit, len(other)), "Proj")[columns]


def team_stack_summary(board, min_projection=0.0):
    """One row per club: the best stack available at each size, for comparing clubs.

    Deliberately the *best available*, not an average: a stack is a thing you pick, so the
    number that matters about a club is the ceiling of what it offers, not its mean.
    """
    rows = []
    for team in sorted(board["Team"].dropna().unique()):
        combos = stack_combinations(board, team, min_projection=min_projection)
        if combos.empty:
            continue
        record = {"Team": team, "QB": combos.iloc[0]["QB"]}
        for size in (2, 3, 4):
            part = combos[combos["Size"] == size]
            if part.empty:
                continue
            best = part.nlargest(1, "Proj").iloc[0]
            record[f"Best {size}-man"] = best["Proj"]
            record[f"{size}-man cost"] = best["Salary"]
        opponents = board.loc[board["Team"] == team, "Opp"].dropna()
        record["Opp"] = opponents.iloc[0] if not opponents.empty else ""
        rows.append(record)
    frame = pd.DataFrame(rows)
    if not frame.empty and "Best 3-man" in frame.columns:
        frame = frame.sort_values("Best 3-man", ascending=False)
    return frame.reset_index(drop=True)


@st.cache_data(ttl="10m", max_entries=8, show_spinner="Enumerating every stack…")
def all_stack_combinations(board, sizes=(1, 2, 3), min_projection=0.0):
    """Every club's combinations in one frame, for comparing across the slate.

    The per-club view answers "which stack of theirs", this one answers "whose stack" -- and
    they are different questions with different answers, because a club's best combination
    can be the slate's fifth-best at twice the price.
    """
    frames = []
    for team in sorted(board["Team"].dropna().unique()):
        combos = stack_combinations(board, team, sizes=sizes,
                                    min_projection=min_projection)
        if not combos.empty:
            frames.append(combos)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    opponents = (board.dropna(subset=["Team", "Opp"])
                 .drop_duplicates("Team").set_index("Team")["Opp"])
    out["Opp"] = out["Team"].map(opponents)
    out["Rank"] = out.groupby("Size")["Proj"].rank(ascending=False, method="min")
    return out.sort_values("Proj", ascending=False).reset_index(drop=True)


def top_per_team(combos, per_team=3, sizes=None):
    """The best `per_team` combinations for each club, across the whole slate.

    This is the shortlist view: every club represented, nobody drowned out by whichever
    offence happens to be priciest. Ranked within the shortlist so the slate-wide order is
    still visible alongside the per-club one.
    """
    if combos.empty:
        return combos
    frame = combos[combos["Size"].isin(sizes)] if sizes else combos
    if frame.empty:
        return frame
    best = (frame.sort_values("Proj", ascending=False)
            .groupby("Team", group_keys=False).head(per_team))
    return best.sort_values("Proj", ascending=False).reset_index(drop=True)
