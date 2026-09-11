"""Clubs as stack units rather than as bags of individual hitters.

A tournament roster is not assembled one bat at a time: five hitters from one club is the
unit, because their outcomes are correlated — the innings that produce runs produce them for
everybody in the order at once. So the useful aggregate is not "this club's mean composite"
but "what the top five of this card project for, and what they cost".

Reusing the pipeline's definition
---------------------------------
`dfs.slate.build_stacks` already defines a stack, and it is reused here rather than
reimplemented. Two of its choices matter and would be easy to get wrong independently:

* the five are taken **by batting order**, not by projection — because slots 1-5 are the
  highest-scoring slots and they guarantee the plate appearances, *not* because adjacency
  creates the correlation. That second reason used to be written here and it is wrong; see
  below.
* clubs with fewer than four priced hitters are dropped, because a "stack" of two carries a
  meaningless salary and would outrank real stacks on value.

A second implementation would drift from the optimiser's own view of a stack, and then the
dashboard and the lineups it is meant to inform would disagree about what a stack even is.

Contiguity is not the mechanism
-------------------------------
Measured over 704 team-games (2026-07-22 .. 08-22, actual boxscore batting order against
actual DK points), **adjacency does nothing**:

* mean pairwise correlation between two slots, by the gap between them: 1 -> 0.128,
  2 -> 0.121, 3 -> 0.119, 4 -> 0.110, 5 -> 0.109, 6 -> 0.131, 8 -> 0.131. Hitters eight slots
  apart correlate as much as neighbours.
* across all 126 five-slot combinations, contiguous blocks average p90 58.9 against 58.0 for
  the rest. The best p90 of any combination is 1-2-3-4-**6** (65.0), a shade above
  1-2-3-4-5 (64.7), which still wins on mean and p95.

The correlation is a **team-level** effect -- the whole card scores when the club scores --
so it does not care which slots are chosen. Slots 1-5 remain the right five (mean 36.9, p90
64.7, clearing 60 points 14.9% of the time, against 29.2 / 53.0 / 5.3% for slots 5-9), but a
skipped slot costs almost nothing: a salary-driven 1-2-3-4-6 is a legitimate construction
rather than a compromise.
"""

import numpy as np
import pandas as pd
import streamlit as st

from dashboards import data

#: Columns the aggregate view adds on top of the pipeline's stack row — the context that
#: decides whether a good-looking stack is actually a good spot.
CONTEXT_COLUMNS = ["Opp SP Hand", "Allowed OPS", "Lineup OPS", "park", "hr_env",
                   "park_hr", "weather_hr", "temp", "wind"]


@st.cache_data(max_entries=8, show_spinner="Building stacks…")
def _team_stacks(date, cache_dir, fingerprint):
    """One row per club: its top-five unit, its price, and the conditions it plays in.

    Everything here is the pipeline's own — the stack definition from `dfs.slate`, the
    projections from `dfs.projections`, the surrendered/lineup OPS pair and the park and
    weather terms from the dashboard's own cached readers.
    """
    from dfs import slate as slate_module

    projected = data.projections_for_date(date, cache_dir)
    if projected.empty:
        return pd.DataFrame()

    frame = _with_salary(projected, date)
    has_salary = frame["Salary"].notna().any()
    try:
        stacks = slate_module.build_stacks(frame, has_salary)
    except Exception:
        return pd.DataFrame()
    if stacks.empty:
        return stacks

    stacks = stacks.merge(_matchup_context(date, cache_dir), on="Team", how="left")
    stacks = stacks.merge(_conditions_context(date, cache_dir), on="Team", how="left")
    return stacks


def team_stacks(date, cache_dir=data.CACHE_DIR):
    """File-aware wrapper around the cached `build_stacks` / `_with_salary` path."""
    return _team_stacks(str(date), cache_dir, data._fingerprint(cache_dir, date))


def _with_salary(projected, date):
    """Attach DK price to a projection frame.

    `project_game` carries no price, so anything built straight off it silently drops the
    entire "what does this cost" half of the question — `build_stacks` returns a stack with
    a null salary and a null value, and the page shows blanks where the ranking should be.
    Both the stack table and the member drill-down go through here for that reason.
    """
    from dashboards import salaries as salaries_module

    frame = projected.copy()
    prices = salaries_module.salaries_for_date(date)
    if not prices.empty and "Name" in frame.columns:
        frame["_key"] = frame["Name"].map(salaries_module.name_key)
        frame = frame.merge(prices[["key", "Salary"]].rename(columns={"key": "_key"}),
                            on="_key", how="left").drop(columns="_key")
    if "Salary" not in frame.columns:
        frame["Salary"] = np.nan
    return frame


def _matchup_context(date, cache_dir):
    """What each club's opponent surrenders, against what the club hits."""
    matchups = data.pitcher_vs_lineup(date, cache_dir)
    if matchups.empty:
        return pd.DataFrame(columns=["Team"])
    # `faces` is the club batting against this starter, so it is the stack's own club.
    out = matchups.rename(columns={"faces": "Team", "throws": "Opp SP Hand"})
    keep = ["Team", "Opp SP Hand", "Allowed OPS", "Lineup OPS", "Allowed vs L",
            "Allowed vs R", "L bats", "R bats"]
    return out.reindex(columns=[c for c in keep if c in out.columns]).drop_duplicates("Team")


def _conditions_context(date, cache_dir):
    """Park and weather for the game each club is in."""
    from dashboards import environment as env

    staffs = env.staff_environment(date, cache_dir)
    if staffs.empty:
        return pd.DataFrame(columns=["Team"])
    # A staff row's conditions belong to its game, so the club *batting* against it takes
    # the same park and air — read off the opponent column rather than the staff's own club.
    out = staffs.rename(columns={"opponent": "Team"})
    keep = ["Team", "park", "hr_env", "park_hr", "weather_hr", "temp", "wind", "enclosed"]
    return (out.reindex(columns=[c for c in keep if c in out.columns])
            .drop_duplicates("Team"))


def stack_members(date, team, cache_dir=data.CACHE_DIR):
    """The five hitters a club's stack is actually made of, in batting order.

    The drill-down for a stack row: a stack score is an aggregate of five specific bats, and
    which five is not guessable from the score.
    """
    projected = data.projections_for_date(date, cache_dir)
    if projected.empty or "Team" not in projected.columns:
        return pd.DataFrame()
    projected = _with_salary(projected, date)
    hitters = projected[(projected.get("Type", "") == "H")
                        & (projected["Team"].astype(str) == str(team))]
    if hitters.empty:
        return pd.DataFrame()
    ordered = hitters.sort_values("Slot")
    columns = ["Slot", "Name", "Pos", "Bats", "Salary", "Proj", "Ceiling", "Floor",
               "PA", "Opp SP", "Opp SP Hand", "Matchup"]
    top = ordered.head(5).reindex(columns=[c for c in columns if c in ordered.columns])
    rest = ordered.iloc[5:].reindex(columns=[c for c in columns if c in ordered.columns])
    top = top.assign(**{"In stack": True})
    rest = rest.assign(**{"In stack": False})
    return pd.concat([top, rest], ignore_index=True)


def add_composite(stacks, board):
    """Attach the report's own composite, aggregated over the same five bats.

    Kept separate from `team_stacks` because the composite is a *scouting* number and the
    stack row is a roster-construction one; a reader comparing them is comparing two
    different opinions, which is the point.
    """
    if stacks is None or stacks.empty or board is None or board.empty:
        return stacks
    if not {"Team", "Composite"}.issubset(board.columns):
        return stacks
    grouped = (board.groupby("Team", as_index=False)
               .agg(**{"Mean composite": ("Composite", "mean"),
                       "Priority bats": ("Signal",
                                         lambda s: int((s == "Priority").sum()))}))
    return stacks.merge(grouped, on="Team", how="left")
