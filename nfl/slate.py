"""Assemble one DK slate into the frame every report tab reads from.

This is the join layer. It takes a DraftKings salary export and attaches, per player:

    PFF season projection      points, and the events behind them
    PFF 2025 grades            YPRR, route grade, elusive rating, yards after contact
    PFF red-zone splits        RZ / EZ / inside-5 workload and the touchdown equity it implies
    PFF coverage splits        man and zone YPRR, and what the opposing defense actually plays
    positional SOS             how hard this week's matchup is, for this position
    cold-start prior           role, availability and expected opportunity from our own model
    snap history               2025 per-game average

**Every source spells names and teams differently**, so both are folded through
`nfl.salaries` before anything is matched -- one canonicaliser for the package. Team codes
are the sharper trap: PFF writes `HST` for Houston, and an unfolded join drops the club
without raising anything.

**Nothing here invents a number.** Where a source has no row for a player the column stays
null, and `basis` records which sources did answer. A preseason slate is roughly half
third- and fourth-stringers who have no PFF projection at all; that has to read as absence
on the page rather than as a zero.
"""

import os
import re

import numpy as np
import pandas as pd

from nfl import pffdata
from nfl import redzone as rz
from nfl import sos as sos_module
from nfl.salaries import canon_team, normalize_name

SKILL_POSITIONS = ("QB", "RB", "WR", "TE")

# The DK export writes roster slots ("WR/FLEX"), the other sources write positions.
_SLOT = re.compile(r"/.*$")

# Not a fixed path. A projection export is superseded by its next pull rather than describing
# a distinct season, so the newest one wins -- see `nfl.pffdata.latest`. Pinning this to a
# filename meant a fresher pull could land beside it and never be read; the two on disk
# differed on 374 of 533 players.
PFF_PROJECTION_FAMILY = "projections"
PFF_RECEIVING_GRADES = {2025: "pff/receiving_summary (4).csv"}
PFF_RUSHING_GRADES = {2025: "pff/rushing_summary (3).csv"}
PFF_RECEIVING_SCHEME = {2025: "pff_coverage_data/receiving_scheme (3).csv"}
PFF_DEFENSE_SCHEME = {2025: "pff_coverage_data/defense_coverage_scheme (5).csv"}
SNAP_COUNTS = {2025: "snap_counts/FantasyPros_Fantasy_Football_2025_Offense_Snap_Counts.csv"}

# Minimum routes in *each* scheme before a man/zone split is shown. Without it the top of
# the coverage table is a receiver with nine man routes and a 9.50 YPRR.
MIN_SCHEME_ROUTES = 50


def _column(frame, name, default=np.nan):
    """A real Series for `name`, even when the column is absent.

    `frame.get(name)` returns **a scalar NaN** for a missing column, which has no index and
    no `.fillna` — the recurring crash in this codebase, documented three times over in
    `dfs.ownership`, `build_bullpen_batted` and `positional_priors`. Never reach for `.get`
    on a frame whose schema depends on which source files happened to be present.
    """
    if name in frame.columns:
        return frame[name]
    return pd.Series(default, index=frame.index)


def _key(series):
    return series.map(normalize_name)


def _read(path, root="."):
    full = os.path.join(root, path)
    return pd.read_csv(full) if os.path.exists(full) else None


def _read_projections(root="."):
    """The newest PFF projection export, or None when there is none on disk.

    Absence is not an error here: half a preseason board has no PFF projection anyway, and
    the `basis` column already carries that.
    """
    roots = [os.path.join(root, r) for r in pffdata.PFF_ROOTS]
    try:
        return pd.read_csv(pffdata.latest(PFF_PROJECTION_FAMILY, roots=roots))
    except pffdata.PffDataError:
        return None


def load_dk_slate(path):
    """The DK export, with position and team folded and a join key attached."""
    frame = pd.read_csv(path)
    frame["key"] = _key(frame["Name"])
    frame["team"] = frame["TeamAbbrev"].map(canon_team)
    frame["pos"] = frame["Position"].astype(str).str.upper().str.replace(_SLOT, "", regex=True)
    return frame


def slate_opponents(dk):
    """{team: opponent} parsed from DK's `Game Info` ("LV@HOU 08/20/2026 08:00PM ET")."""
    pairs = {}
    for info in dk.get("Game Info", pd.Series(dtype=str)).dropna().unique():
        matchup = str(info).split(" ")[0]
        if "@" not in matchup:
            continue
        away, home = (canon_team(t) for t in matchup.split("@", 1))
        pairs[away] = home
        pairs[home] = away
    return pairs


def defense_coverage_rates(season=2025, root="."):
    """Share of each defense's coverage snaps played in man, plus the league rate.

    Aggregated from the defenders a club actually fielded, because PFF publishes this per
    player rather than per team.
    """
    path = PFF_DEFENSE_SCHEME.get(int(season))
    frame = _read(path, root) if path else None
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["team", "man_rate"]), np.nan
    frame = frame.copy()
    frame["team"] = frame["team_name"].map(canon_team)
    man = pd.to_numeric(frame.get("man_snap_counts_coverage"), errors="coerce").fillna(0)
    zone = pd.to_numeric(frame.get("zone_snap_counts_coverage"), errors="coerce").fillna(0)
    frame["_man"], frame["_zone"] = man, zone
    grouped = frame.groupby("team", as_index=False)[["_man", "_zone"]].sum()
    total = grouped["_man"] + grouped["_zone"]
    grouped["man_rate"] = np.where(total > 0, grouped["_man"] / total * 100, np.nan)
    league = float(frame["_man"].sum() / max(frame["_man"].sum() + frame["_zone"].sum(), 1) * 100)
    return grouped[["team", "man_rate"]], league


def build_slate(dk_path, season=2025, week=None, priors=None, root="."):
    """One row per DK skill entry, with every source attached. DSTs are excluded.

    `priors` is an optional cold-start frame from `nfl.coldstart.build_priors`; passing it
    avoids re-downloading rosters when a caller already has one.
    """
    dk = load_dk_slate(dk_path)
    board = dk[dk["pos"].isin(SKILL_POSITIONS)].copy()
    if board.empty:
        return board

    opponents = slate_opponents(dk)
    board["opp"] = board["team"].map(opponents)

    # --- PFF season projection ---------------------------------------------------------
    projections = _read_projections(root)
    if projections is not None:
        projections = projections.copy()
        projections["key"] = _key(projections["playerName"])
        games = pd.to_numeric(projections["games"], errors="coerce").replace(0, np.nan)
        projections["pff_ppg"] = pd.to_numeric(
            projections["fantasyPoints"], errors="coerce") / games
        keep = ["key", "pff_ppg", "passAtt", "rushAtt", "recvTargets", "recvReceptions"]
        board = board.merge(projections.reindex(columns=keep).drop_duplicates("key"),
                            on="key", how="left")

    # --- PFF grades --------------------------------------------------------------------
    grades = _read(PFF_RECEIVING_GRADES.get(season, ""), root)
    if grades is not None:
        grades = grades.copy()
        grades["key"] = _key(grades["player"])
        keep = ["key", "yprr", "routes", "route_rate", "slot_rate", "grades_pass_route",
                "avg_depth_of_target", "contested_catch_rate"]
        board = board.merge(grades.reindex(columns=keep).drop_duplicates("key"),
                            on="key", how="left")

    rushing = _read(PFF_RUSHING_GRADES.get(season, ""), root)
    if rushing is not None:
        rushing = rushing.copy()
        rushing["key"] = _key(rushing["player"])
        keep = ["key", "elusive_rating", "yco_attempt", "breakaway_percent", "grades_run"]
        board = board.merge(rushing.reindex(columns=keep).drop_duplicates("key"),
                            on="key", how="left")

    # --- red zone ----------------------------------------------------------------------
    try:
        profile = rz.team_red_zone_share(rz.red_zone_profile(season, root=root))
    except (KeyError, FileNotFoundError):
        profile = None
    if profile is not None and not profile.empty:
        keep = ["key", "games", "rzRecTarg", "ezRecTarg", "rzRushCarries", "i5RushCarries",
                "xTD", "TD", "TD_oe", "xTD_pg", "rz_target_share", "rz_carry_share"]
        board = board.merge(profile.reindex(columns=keep).drop_duplicates("key"),
                            on="key", how="left")

    # --- coverage: the receiver's split, and what he will face --------------------------
    scheme = _read(PFF_RECEIVING_SCHEME.get(season, ""), root)
    if scheme is not None:
        scheme = scheme.copy()
        scheme["key"] = _key(scheme["player"])
        for source, name in (("man_yprr", "man_yprr"), ("zone_yprr", "zone_yprr"),
                             ("man_routes", "man_routes"), ("zone_routes", "zone_routes")):
            scheme[name] = pd.to_numeric(scheme.get(source), errors="coerce")
        keep = ["key", "man_yprr", "zone_yprr", "man_routes", "zone_routes"]
        board = board.merge(scheme.reindex(columns=keep).drop_duplicates("key"),
                            on="key", how="left")

    rates, league_man = defense_coverage_rates(season, root)
    board = board.merge(rates.rename(columns={"team": "opp", "man_rate": "opp_man_rate"}),
                        on="opp", how="left")
    board["league_man_rate"] = league_man

    enough = ((_column(board, "man_routes") >= MIN_SCHEME_ROUTES)
              & (_column(board, "zone_routes") >= MIN_SCHEME_ROUTES))
    zone_edge = _column(board, "zone_yprr") - _column(board, "man_yprr")
    board["zone_edge"] = np.where(enough, zone_edge, np.nan)
    # Positive when the receiver's strength matches what this defense actually plays.
    faces_zone = board["opp_man_rate"] < board["league_man_rate"]
    board["scheme_fit"] = np.where(enough,
                                   np.where(faces_zone, zone_edge, -zone_edge), np.nan)

    # --- schedule ----------------------------------------------------------------------
    board = sos_module.attach_sos(board, position_column="pos", team_column="team",
                                  week=week, root=root)

    # --- snaps -------------------------------------------------------------------------
    snaps = _read(SNAP_COUNTS.get(season, ""), root)
    if snaps is not None:
        snaps = snaps.copy()
        snaps["key"] = _key(snaps["PLAYER"])
        snaps["snap_avg"] = pd.to_numeric(snaps["AVG"], errors="coerce")
        board = board.merge(snaps.reindex(columns=["key", "snap_avg"]).drop_duplicates("key"),
                            on="key", how="left")

    # --- our own prior -----------------------------------------------------------------
    if priors is not None and not priors.empty:
        prior = priors.copy()
        prior["key"] = _key(prior["player_name"])
        keep = ["key", "source", "depth_rank", "availability", "expected_opportunity", "pick"]
        board = board.merge(prior.reindex(columns=keep).drop_duplicates("key"),
                            on="key", how="left")

    board["basis"] = _basis(board)
    board["proj"] = _blended_projection(board)
    return board.sort_values("proj", ascending=False, na_position="last").reset_index(drop=True)


def _basis(board):
    """Which sources actually answered for each player. Never inferred from a value."""
    has_pff = _column(board, "pff_ppg").notna()
    source = _column(board, "source", None)
    out = pd.Series("none", index=board.index, dtype=object)
    out[source == "replacement"] = "replacement"
    out[source == "draft"] = "draft capital"
    out[source == "prior_season"] = "prior season"
    out[has_pff] = "PFF projection"
    return out


def _blended_projection(board):
    """PFF's number where it exists, nudged by schedule; ours where it does not.

    Deliberately *not* an average of the two. PFF projects points and the cold-start model
    projects opportunity, so averaging them would be adding different units; and where both
    exist PFF is the better number. The prior earns the row only when PFF is silent, which
    on a preseason slate is half the board.
    """
    pff = pd.to_numeric(_column(board, "pff_ppg"), errors="coerce")
    multiplier = pd.to_numeric(_column(board, "sos_mult"), errors="coerce").fillna(1.0)
    opportunity = pd.to_numeric(_column(board, "expected_opportunity"), errors="coerce")
    # Opportunity is not points. This is a crude conversion held deliberately low so a
    # cold-start row never outranks a projected starter on an unfitted constant.
    fallback = opportunity * 0.35
    return (pff * multiplier).where(pff.notna(), fallback)
