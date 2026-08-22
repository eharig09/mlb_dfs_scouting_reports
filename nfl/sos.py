"""Positional strength of schedule.

One file per position (QB / RB / WR / TE / DST), 32 teams by 17 weeks, plus season and
playoff aggregates. A blank week is a bye.

**Higher means easier, and this was checked rather than assumed.** Each distinct rating in
the grid is a *defense*, repeated wherever that defense appears, so the ratings can be
joined back to opponents through the schedule and correlated against what those defenses
actually gave up. Against 2025 WR PPR allowed per game:

    corr(rating, WR PPR allowed) = +0.506

    highest rated: ARI 10.00, WAS 9.89, TEN 9.45  ->  allowed 30.2, 34.9, 35.6
    lowest rated:  NE 0.00, DEN 0.23, PHI 1.31    ->  allowed 29.2, 27.0, 26.8

Reading it the other way would invert every matchup adjustment built on it — the same class
of error as `spread_line`'s sign, which `nfl.projections` documents for the same reason.

The ratings are also **team codes in PFF's dialect** (`ARZ`, `BLT`, `CLV`, `HST`, `LA`), not
nflverse's. `nfl.salaries.canon_team` folds them; skipping that silently drops Houston.
"""

import os

import numpy as np
import pandas as pd

from nfl.salaries import canon_team

POSITIONS = ("QB", "RB", "WR", "TE", "DST")

SOS_FILES = {
    "QB": "SOS/qb-fantasy-sos (1).csv",
    "RB": "SOS/rb-fantasy-sos (1).csv",
    "WR": "SOS/wr-fantasy-sos (1).csv",
    "TE": "SOS/te-fantasy-sos (1).csv",
    "DST": "SOS/dst-fantasy-sos (1).csv",
}

# Team codes are folded by `nfl.salaries.canon_team`, which already knows PFF's dialect
# (`HST`, `ARZ`, `BLT`, `CLV`, `LA`, `SD`) alongside DK's and nflverse's. This module used
# to carry its own map, which was worse than redundant: it folded to nflverse spellings
# (SF, GB) while the package canon is PFR-style (SFO, GNB), so a board joined through both
# matched on nothing for seven clubs and reported it as missing data rather than an error.
# One canonicaliser per package, always.

# The published scale runs 0-10 across the league. Centring on the midpoint keeps a neutral
# schedule at 1.0 once converted to a multiplier.
SCALE_MID = 5.0
SCALE_HALF_RANGE = 5.0


def load_sos(position, root="."):
    """Long-form ratings: one row per (team, week). Byes dropped."""
    path = SOS_FILES.get(str(position).upper())
    if path is None:
        raise KeyError(f"no SOS file for position {position!r}")
    frame = pd.read_csv(os.path.join(root, path))
    week_columns = [c for c in frame.columns if str(c).strip().isdigit()]

    rows = frame.melt(id_vars=["Offense"], value_vars=week_columns,
                      var_name="week", value_name="rating")
    rows["team"] = rows["Offense"].map(canon_team)
    rows["week"] = pd.to_numeric(rows["week"], errors="coerce").astype("Int64")
    rows["rating"] = pd.to_numeric(rows["rating"], errors="coerce")
    rows["position"] = str(position).upper()
    # A blank cell is a bye, not a zero. Zero is a real and meaningful rating here -- it is
    # the hardest matchup on the board -- so the two must not be conflated.
    rows = rows.dropna(subset=["rating", "week"])
    return rows[["team", "position", "week", "rating"]].reset_index(drop=True)


def season_sos(position, root="."):
    """Season-level rating per team, straight from the file's own aggregate column."""
    path = SOS_FILES.get(str(position).upper())
    frame = pd.read_csv(os.path.join(root, path))
    out = pd.DataFrame({
        "team": frame["Offense"].map(canon_team),
        "position": str(position).upper(),
        "season_sos": pd.to_numeric(frame.get("Season SOS"), errors="coerce"),
        "games": pd.to_numeric(frame.get("Season #G"), errors="coerce"),
    })
    return out.dropna(subset=["season_sos"]).reset_index(drop=True)


def load_all(root="."):
    """Every position's weekly ratings in one long frame."""
    frames = []
    for position in POSITIONS:
        try:
            frames.append(load_sos(position, root=root))
        except (KeyError, FileNotFoundError):
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def sos_multiplier(rating, strength=0.06):
    """Turn a 0-10 rating into a gentle multiplier centred on 1.0.

    `strength` is the swing at the extremes: 0.06 moves the very easiest schedule +6% and
    the very hardest -6%. Deliberately small and **not fitted** — it is a display and
    tie-break weight until somebody scores a season with it on and off. A schedule term
    large enough to reorder a board should have to earn that with a held-out result.
    """
    value = pd.to_numeric(pd.Series(np.asarray(rating, dtype="float64")), errors="coerce")
    centred = (value - SCALE_MID) / SCALE_HALF_RANGE
    out = 1.0 + centred.clip(-1.0, 1.0) * float(strength)
    return out.fillna(1.0).to_numpy()


def attach_sos(board, position_column="Position", team_column="TeamAbbrev",
               week=None, root="."):
    """Add `sos_rating` and `sos_mult` to a board, matched on position and team."""
    if board is None or board.empty:
        return board
    ratings = load_all(root=root)
    if ratings.empty:
        return board
    if week is not None:
        ratings = ratings[ratings["week"] == int(week)]
    else:
        # No week given means "the schedule as a whole", which is the season average of the
        # weekly ratings rather than any one week.
        ratings = ratings.groupby(["team", "position"], as_index=False)["rating"].mean()

    out = board.copy()
    out["_team"] = out[team_column].map(canon_team)
    # DK writes roster slots, not positions: "WR/FLEX" has to fold to "WR" or nothing joins.
    out["_pos"] = (out[position_column].astype(str).str.upper()
                   .str.split("/").str[0].str.strip())
    ratings = ratings.rename(columns={"team": "_team", "position": "_pos",
                                      "rating": "sos_rating"})
    merged = out.merge(ratings[["_team", "_pos", "sos_rating"]], on=["_team", "_pos"],
                       how="left")
    merged["sos_mult"] = sos_multiplier(merged["sos_rating"])
    return merged.drop(columns=["_team", "_pos"])
