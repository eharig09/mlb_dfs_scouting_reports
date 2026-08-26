"""Team-level football: what defences allow, what offences produce, and how games went.

Reads nflverse through `nfl.data` and scores with `nfl.backtest`, so the DK points here are
the same numbers the backtest reconciles against nflverse — 23,737 player-weeks over
2021-24 with zero residuals.

## The measured warning that shapes this whole module

**DK points allowed by position barely predicts itself.** Measured over 2021-25, mean
year-over-year correlation per defence:

    RB  +0.30      TE  +0.19      WR  +0.11      QB  +0.09

and within a single season (2025 weeks 1-9 against 10-18): QB +0.36, WR +0.32, TE +0.23,
RB +0.02.

The *spread* is nonetheless real — WR allowed runs 26.8 to 37.2 DK points a game between the
10th and 90th percentile. So the honest presentation is both numbers: what a defence actually
allowed, and that figure **regressed toward the league mean by its own measured
reliability**. A defence 5 points above average on WRs projects about half a point above
average, not five. `projected_allowed` does exactly that arithmetic and nothing cleverer.
"""

import numpy as np
import pandas as pd
import streamlit as st

from nfl import backtest
from nfl import data as nfl_data
from nfl import snapshot
from nfl.salaries import canon_team

POSITIONS = ("QB", "RB", "WR", "TE")

# Mean year-over-year self-correlation of DK points allowed per game, 2021-25. Used as the
# regression-to-the-mean coefficient, which is exactly what a correlation is: a defence's
# distance from average carries forward by this much and no more.
RELIABILITY = {"QB": 0.09, "RB": 0.30, "WR": 0.11, "TE": 0.19}

# The same figure measured inside one season, 2025 weeks 1-9 against 10-18. Shown in the
# caption rather than used: one season is one observation, and swapping the coefficient for
# a number with n=1 behind it would be trading a weak estimate for a weaker one.
IN_SEASON_RELIABILITY = {"QB": 0.36, "RB": 0.02, "WR": 0.32, "TE": 0.23}


@st.cache_data(ttl="6h", max_entries=8, show_spinner="Scoring nflverse box scores…")
def _scored(seasons):
    """DK points per player-week, with canonical team codes on both sides."""
    weekly = nfl_data.load_weekly(list(seasons))
    frame = backtest.offense_actuals(weekly)
    frame = frame[frame["dk_points"].notna()].copy()
    frame["team"] = frame["team"].map(canon_team)
    frame["opponent"] = frame["opponent"].map(canon_team)
    return frame


@st.cache_data(ttl="6h", max_entries=8, show_spinner=False)
def defense_allowed(seasons, positions=POSITIONS):
    """DK points each defence allowed per game, by position, with league context.

    Columns: `Team, Pos, Allowed/G, Rank, vs League, Projected, Reliability`.

    `Rank` is 1 = allowed the most, so a high rank is a good matchup — stated on the page
    rather than left for the reader to infer from a number that could sort either way.
    """
    # **Snapshot first, and this is the one that matters for hosting.** These read
    # nflverse over the network; on an ephemeral container every cold start would
    # re-download five seasons of parquet before the first page rendered.
    cached = snapshot.read("defense_allowed")
    if cached is not None:
        return cached

    frame = _scored(tuple(seasons))
    frame = frame[frame["position"].isin(positions)]
    if frame.empty:
        return pd.DataFrame()

    per_game = (frame.groupby(["opponent", "position", "season", "week"], as_index=False)
                .agg(points=("dk_points", "sum")))
    allowed = (per_game.groupby(["opponent", "position"], as_index=False)
               .agg(**{"Allowed/G": ("points", "mean"), "Games": ("week", "size")}))
    allowed = allowed.rename(columns={"opponent": "Team", "position": "Pos"})

    league = allowed.groupby("Pos")["Allowed/G"].transform("mean")
    allowed["vs League"] = allowed["Allowed/G"] - league
    allowed["Reliability"] = allowed["Pos"].map(RELIABILITY)
    # Regression to the mean, with the correlation as the coefficient. Nothing cleverer.
    allowed["Projected"] = league + allowed["Reliability"] * allowed["vs League"]
    allowed["Rank"] = (allowed.groupby("Pos")["Allowed/G"]
                       .rank(ascending=False, method="min").astype(int))
    return allowed.sort_values(["Pos", "Rank"]).reset_index(drop=True)


@st.cache_data(ttl="6h", max_entries=8, show_spinner=False)
def team_offense(seasons, positions=POSITIONS):
    """DK points each offence produced per game, by position. The mirror of `defense_allowed`."""
    # **Snapshot first, and this is the one that matters for hosting.** These read
    # nflverse over the network; on an ephemeral container every cold start would
    # re-download five seasons of parquet before the first page rendered.
    cached = snapshot.read("team_offense")
    if cached is not None:
        return cached

    frame = _scored(tuple(seasons))
    frame = frame[frame["position"].isin(positions)]
    if frame.empty:
        return pd.DataFrame()
    per_game = (frame.groupby(["team", "position", "season", "week"], as_index=False)
                .agg(points=("dk_points", "sum")))
    produced = (per_game.groupby(["team", "position"], as_index=False)
                .agg(**{"Produced/G": ("points", "mean"), "Games": ("week", "size")}))
    produced = produced.rename(columns={"team": "Team", "position": "Pos"})
    produced["Rank"] = (produced.groupby("Pos")["Produced/G"]
                        .rank(ascending=False, method="min").astype(int))
    return produced.sort_values(["Pos", "Rank"]).reset_index(drop=True)


@st.cache_data(ttl="6h", max_entries=8, show_spinner="Reading schedules…")
def team_results(seasons):
    """One row per team: record, points for and against, and the totals market's read.

    Built from `schedules`, which carries the closing spread and total alongside the score,
    so what the market expected and what happened sit in the same row.
    """
    # **Snapshot first, and this is the one that matters for hosting.** These read
    # nflverse over the network; on an ephemeral container every cold start would
    # re-download five seasons of parquet before the first page rendered.
    cached = snapshot.read("team_results")
    if cached is not None:
        return cached

    schedules = nfl_data.load_schedules(list(seasons))
    frame = schedules[schedules.reindex(columns=["home_score"])["home_score"].notna()].copy()
    if frame.empty:
        return pd.DataFrame()

    rows = []
    for side, opposite in (("home", "away"), ("away", "home")):
        part = pd.DataFrame({
            "Season": frame["season"],
            "Week": frame["week"],
            "Team": frame[f"{side}_team"].map(canon_team),
            "Opp": frame[f"{opposite}_team"].map(canon_team),
            "For": pd.to_numeric(frame[f"{side}_score"], errors="coerce"),
            "Against": pd.to_numeric(frame[f"{opposite}_score"], errors="coerce"),
            "Total": pd.to_numeric(frame.reindex(columns=["total_line"])["total_line"],
                                   errors="coerce"),
            "Home": side == "home",
        })
        rows.append(part)
    games = pd.concat(rows, ignore_index=True)
    games["Margin"] = games["For"] - games["Against"]
    games["Combined"] = games["For"] + games["Against"]
    games["Result"] = np.where(games["Margin"] > 0, "W",
                               np.where(games["Margin"] < 0, "L", "T"))
    return games


@st.cache_data(ttl="6h", max_entries=8, show_spinner=False)
def team_summary(seasons):
    """`team_results` folded to one row per club."""
    # **Snapshot first, and this is the one that matters for hosting.** These read
    # nflverse over the network; on an ephemeral container every cold start would
    # re-download five seasons of parquet before the first page rendered.
    cached = snapshot.read("team_summary")
    if cached is not None:
        return cached

    games = team_results(tuple(seasons))
    if games.empty:
        return pd.DataFrame()
    summary = games.groupby("Team", as_index=False).agg(
        G=("Week", "size"),
        W=("Result", lambda s: int((s == "W").sum())),
        L=("Result", lambda s: int((s == "L").sum())),
        **{"PF/G": ("For", "mean"), "PA/G": ("Against", "mean"),
           "Margin/G": ("Margin", "mean"), "Combined/G": ("Combined", "mean"),
           "Total line": ("Total", "mean")})
    summary["vs Total"] = summary["Combined/G"] - summary["Total line"]
    return summary.sort_values("Margin/G", ascending=False).reset_index(drop=True)


def matchup_table(allowed, slate_players):
    """Join what each defence allows onto the offences facing it this slate.

    One row per (offence, position): the club, who it plays, and what that opponent has been
    giving up to the position. This is the shape the question is actually asked in --
    "who does my guy's position get to face" -- rather than a defence-first table the reader
    has to invert in their head.
    """
    if allowed.empty or slate_players is None or slate_players.empty:
        return pd.DataFrame()
    pairs = (slate_players.dropna(subset=["Team", "Opp"])
             .drop_duplicates(["Team", "Opp"])[["Team", "Opp", "Game"]])
    rows = []
    for position in sorted(allowed["Pos"].unique()):
        by_team = allowed[allowed["Pos"] == position].set_index("Team")
        for pair in pairs.itertuples(index=False):
            if pair.Opp not in by_team.index:
                continue
            defence = by_team.loc[pair.Opp]
            rows.append({
                "Team": pair.Team, "Opp": pair.Opp, "Game": pair.Game, "Pos": position,
                "Opp allows/G": defence["Allowed/G"],
                "Rank": int(defence["Rank"]),
                "vs League": defence["vs League"],
                "Projected": defence["Projected"],
            })
    return pd.DataFrame(rows).sort_values(["Pos", "Opp allows/G"], ascending=[True, False])


def allowed_against_stacks(allowed, stack_summary, position, size="Best 3-man"):
    """What a defence allows to a position, set against the stack facing it.

    One row per game side: the offence, the defence it draws, what that defence gives up to
    `position`, and what the offence's best stack of `size` costs and projects. The two
    halves of a stacking decision -- is the matchup soft, and is the stack affordable -- have
    to be on the same axes before either is worth acting on.

    **The allowed half is weak and the page says so.** It is here because the *spread* is
    real, not because the rate forecasts itself; see `RELIABILITY`.
    """
    if allowed.empty or stack_summary is None or stack_summary.empty:
        return pd.DataFrame()
    by_team = allowed[allowed["Pos"] == position].set_index("Team")
    cost = f"{str(size).split('-')[0].split()[-1]}-man cost"
    if size not in stack_summary.columns:
        return pd.DataFrame()

    # **`itertuples` is not usable here.** It renames any column that is not a valid Python
    # identifier to a positional `_5`, and every column this function needs -- "Best 3-man",
    # "3-man cost" -- is exactly that shape. Every lookup came back None and the frame came
    # back empty with no error. Indexing the row by its real column name is the fix.
    rows = []
    for _, row in stack_summary.iterrows():
        opponent = row.get("Opp")
        if not opponent or opponent not in by_team.index:
            continue
        defence = by_team.loc[opponent]
        projection = row.get(size)
        salary = row.get(cost)
        if projection is None or pd.isna(projection):
            continue
        rows.append({
            "Team": row["Team"], "Opp": opponent, "QB": row.get("QB", ""),
            "Opp allows/G": defence["Allowed/G"],
            "Opp allows rank": int(defence["Rank"]),
            "Projected allowed": defence["Projected"],
            "Stack proj": projection,
            "Stack cost": salary,
            "Stack value": (projection / (salary / 1000.0)
                            if salary and not pd.isna(salary) else float("nan")),
        })
    return pd.DataFrame(rows)
