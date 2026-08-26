"""Cached readers over the PFF profiles, for the football exploration pages.

The MLB pages read one night's cached payloads. These read **season profiles** —
`nfl.usage`, `nfl.oline`, `nfl.pffdata` — because that is the shape PFF publishes: a season
of deployment per player, not a game log. So the football question these pages answer is
"how is this player used, and by whom" rather than "is tonight a good spot".

That framing is not a compromise, it is what the data measured out to be worth. Deployment
self-correlates at 0.82–0.95 year over year while defensive allowed-rates sit under 0.11, so
a page built on *where a player lines up and how often he is thrown to* is standing on the
solid half. See `docs/nfl/pff_integration.md`.

Everything here is cached: a page switch or a filter change re-reads nothing.
"""

import numpy as np
import pandas as pd
import streamlit as st

from nfl import oline as oline_module
from nfl import pffdata
from nfl import snapshot
from nfl import usage

# Seasons the receiving/rushing summaries cover. The depth and concept exports only reach
# back to 2023, and a profile silently loses those columns before then rather than failing --
# `usage._depth_season` returns an empty frame, which is the honest answer for a season with
# no depth export.
DEEP_DATA_FROM = 2023


@st.cache_data(ttl="1h", show_spinner=False)
def available_seasons(family="receiving_summary"):
    """Seasons that family has a usable export for, newest first."""
    # Snapshot first. Building the catalog means fingerprinting the raw exports, and on a
    # host that has none it falls through to downloading nflverse rosters -- a network call
    # to answer "which seasons do I have files for", on a machine with no files.
    cached = snapshot.available_seasons(family)
    if cached is not None:
        return cached

    table = pffdata.catalog()
    rows = table[(table["family"] == family) & table["usable"] & table["season"].notna()]
    return sorted({int(s) for s in rows["season"]}, reverse=True)


@st.cache_data(ttl="1h", max_entries=16, show_spinner="Reading PFF receiving profiles…")
def receivers(seasons, min_routes=100):
    """Deployment profile per receiver, with a few derived reading aids attached."""

    # A single-season request is served from the committed snapshot when there
    # is one -- that is what makes the hosted app instant and network-free. A
    # multi-season blend still computes: the snapshot stores seasons apart, and
    # re-deriving a blend from them would duplicate `usage._blend` badly.
    if len(seasons) == 1:
        cached = snapshot.read("receivers", int(list(seasons)[0]))
        if cached is not None:
            return cached
    frame = usage.receiver_profile(list(seasons), min_routes=min_routes).reset_index()
    frame = frame[frame["pos"].isin(("WR", "TE", "RB"))]
    # Alignment as a single readable label. The three rates are the most stable numbers in
    # the data (0.82-0.95), so where a man lines up is the one thing worth naming outright.
    frame["Alignment"] = _alignment(frame)
    frame["Deep rate"] = frame.get("deep_tgt_rate")
    return frame.rename(columns={
        "player": "Name", "team": "Team", "pos": "Pos", "games": "G",
        "routes_pg": "Routes/G", "tprr": "TPRR", "slot_rate": "Slot%",
        "wide_rate": "Wide%", "inline_rate": "Inline%", "adot": "aDOT",
        "yprr": "YPRR", "grade_route": "Route grade", "targets": "Targets",
        "routes": "Routes", "yac_per_rec": "YAC/rec",
    })


def _alignment(frame):
    """Where a receiver mostly lines up: Slot, Wide or Inline."""
    columns = ["slot_rate", "wide_rate", "inline_rate"]
    present = [c for c in columns if c in frame.columns]
    if not present:
        return pd.Series("", index=frame.index)
    labels = {"slot_rate": "Slot", "wide_rate": "Wide", "inline_rate": "Inline"}
    values = frame.reindex(columns=present).astype(float)

    # **`idxmax` raises on an all-NaN row** -- `ValueError: Encountered all NA values` --
    # so the rows with nothing to compare have to be excluded *before* it runs, not masked
    # out afterwards. A blend can easily produce such a row: a player present in the rushing
    # export but not the receiving one carries no alignment at all, and the whole page died
    # on him rather than leaving one cell blank.
    known = values.notna().any(axis=1)
    out = pd.Series("", index=frame.index, dtype=object)
    if known.any():
        out.loc[known] = values.loc[known].idxmax(axis=1).map(labels)
    return out


@st.cache_data(ttl="1h", max_entries=16, show_spinner="Reading PFF rushing profiles…")
def rushers(seasons, min_carries=25):
    # A single-season request is served from the committed snapshot when there
    # is one -- that is what makes the hosted app instant and network-free. A
    # multi-season blend still computes: the snapshot stores seasons apart, and
    # re-deriving a blend from them would duplicate `usage._blend` badly.
    if len(seasons) == 1:
        cached = snapshot.read("rushers", int(list(seasons)[0]))
        if cached is not None:
            return cached
    frame = usage.rusher_profile(list(seasons), min_carries=min_carries).reset_index()
    return frame.rename(columns={
        "player": "Name", "team": "Team", "pos": "Pos", "games": "G",
        "carries_pg": "Carries/G", "targets_pg": "Targets/G", "routes_pg": "Routes/G",
        "gap_share": "Gap%", "elusive": "Elusive", "breakaway_pct": "Breakaway%",
        "grade_run": "Run grade", "ypc": "YPC", "yco_per_att": "YCO/att",
        "carries": "Carries", "targets": "Targets",
    })


@st.cache_data(ttl="1h", max_entries=16, show_spinner="Reading red-zone splits…")
def high_value(seasons, min_games=4):
    # A single-season request is served from the committed snapshot when there
    # is one -- that is what makes the hosted app instant and network-free. A
    # multi-season blend still computes: the snapshot stores seasons apart, and
    # re-deriving a blend from them would duplicate `usage._blend` badly.
    if len(seasons) == 1:
        cached = snapshot.read("high_value", int(list(seasons)[0]))
        if cached is not None:
            return cached
    frame = usage.high_value_touches(list(seasons), min_games=min_games).reset_index()
    return frame.rename(columns={
        "player": "Name", "team": "Team", "pos": "Pos", "games": "G",
        "rz_carries_pg": "RZ car/G", "i5_carries_pg": "i5 car/G",
        "rz_targets_pg": "RZ tgt/G", "ez_targets_pg": "EZ tgt/G",
        "hv_touches_pg": "HV touches/G", "carries_pg": "Carries/G",
        "targets_pg": "Targets/G", "rz_rush_td_pg": "RZ rush TD/G",
    })


@st.cache_data(ttl="1h", max_entries=16, show_spinner="Building target distribution…")
def target_distribution(seasons, min_routes=100):
    # A single-season request is served from the committed snapshot when there
    # is one -- that is what makes the hosted app instant and network-free. A
    # multi-season blend still computes: the snapshot stores seasons apart, and
    # re-deriving a blend from them would duplicate `usage._blend` badly.
    if len(seasons) == 1:
        cached = snapshot.read("target_distribution", int(list(seasons)[0]))
        if cached is not None:
            return cached
    frame = usage.team_target_distribution(list(seasons), min_routes=min_routes)
    return frame.rename(columns={
        "player": "Name", "team": "Team", "pos": "Pos",
        "target_share": "Target share", "route_share": "Route share",
        "wr_target_share": "WR share", "te_target_share": "TE share",
        "rb_target_share": "RB share", "targets": "Targets", "routes": "Routes",
        "adot": "aDOT", "slot_rate": "Slot%", "tprr": "TPRR",
    })


@st.cache_data(ttl="1h", max_entries=8, show_spinner="Reading coverage schemes…")
def defense_scheme(season):
    """Team man/zone **tendency**, plus what each scheme gave up.

    The allowed columns are carried so the page can show them, and the page says plainly
    that they do not persist: man YPT allowed self-correlates at +0.05 year over year, zone
    at +0.01. Tendency (man rate, +0.46) is the only half worth acting on.
    """
    cached = snapshot.read("defense_scheme", int(season))
    if cached is not None:
        return cached
    frame = pffdata.load("defense_coverage_scheme", int(season))
    number = lambda c: pd.to_numeric(frame.reindex(columns=[c])[c], errors="coerce").fillna(0)
    grouped = pd.DataFrame({
        "Team": frame["team"],
        "man_snaps": number("man_snap_counts_coverage"),
        "zone_snaps": number("zone_snap_counts_coverage"),
        "man_targets": number("man_targets"), "zone_targets": number("zone_targets"),
        "man_yards": number("man_yards"), "zone_yards": number("zone_yards"),
        "man_tds": number("man_touchdowns"), "zone_tds": number("zone_touchdowns"),
    }).groupby("Team", as_index=False).sum()

    snaps = grouped["man_snaps"] + grouped["zone_snaps"]
    grouped["Man%"] = np.where(snaps > 0, grouped["man_snaps"] / snaps * 100, np.nan)
    grouped["Zone%"] = 100 - grouped["Man%"]
    grouped["Man YPT"] = _ratio(grouped["man_yards"], grouped["man_targets"])
    grouped["Zone YPT"] = _ratio(grouped["zone_yards"], grouped["zone_targets"])
    grouped["Scheme gap"] = grouped["Man YPT"] - grouped["Zone YPT"]
    return grouped[["Team", "Man%", "Zone%", "Man YPT", "Zone YPT", "Scheme gap"]]


@st.cache_data(ttl="1h", max_entries=8, show_spinner="Reading slot coverage…")
def defense_slot(season):
    cached = snapshot.read("defense_slot", int(season))
    if cached is not None:
        return cached
    frame = pffdata.load("slot_coverage", int(season))
    number = lambda c: pd.to_numeric(frame.reindex(columns=[c])[c], errors="coerce").fillna(0)
    grouped = pd.DataFrame({
        "Team": frame["team"], "snaps": number("coverage_snaps"),
        "targets": number("targets"), "yards": number("yards"),
        "receptions": number("receptions"), "tds": number("touchdowns"),
    }).groupby("Team", as_index=False).sum()
    grouped["Slot YPT"] = _ratio(grouped["yards"], grouped["targets"])
    grouped["Slot Y/snap"] = _ratio(grouped["yards"], grouped["snaps"])
    grouped["Slot TD%"] = _ratio(grouped["tds"], grouped["targets"]) * 100
    return grouped[["Team", "Slot YPT", "Slot Y/snap", "Slot TD%", "targets", "snaps"]]


@st.cache_data(ttl="1h", max_entries=8, show_spinner="Reading receiver scheme splits…")
def receiver_scheme(season, min_routes=50):
    """Per-receiver man/zone splits, with the target-rate gap that actually persists."""


    cached = snapshot.read("receiver_scheme", int(season))
    if cached is not None:
        return cached
    frame = pffdata.load("receiving_scheme", int(season))
    number = lambda c: pd.to_numeric(frame.reindex(columns=[c])[c], errors="coerce")
    out = pd.DataFrame({
        "Name": frame["player"], "Team": frame["team"], "Pos": frame["pos"],
        "G": number("player_game_count"),
        "Man routes": number("man_routes"), "Zone routes": number("zone_routes"),
        "Man YPRR": number("man_yprr"), "Zone YPRR": number("zone_yprr"),
        "man_targets": number("man_targets"), "zone_targets": number("zone_targets"),
    })
    out = out[(out["Man routes"] >= min_routes) & (out["Zone routes"] >= min_routes)]
    out["YPRR gap"] = out["Man YPRR"] - out["Zone YPRR"]
    out["Man TPRR"] = _ratio(out["man_targets"], out["Man routes"])
    out["Zone TPRR"] = _ratio(out["zone_targets"], out["Zone routes"])
    out["TPRR gap"] = out["Man TPRR"] - out["Zone TPRR"]
    return out.drop(columns=["man_targets", "zone_targets"])


@st.cache_data(ttl="1h", max_entries=4, show_spinner="Projecting offensive lines…")
def lines(season):
    """Projected starting fives, plus the unit table. Needs nflverse depth charts."""

    # Needs nflverse depth charts, so it is the frame most worth having
    # precomputed: on an ephemeral container it is a network call per boot.
    players = snapshot.read("line_players")
    units = snapshot.read("line_units")
    if players is not None and units is not None:
        return players, units
    line = oline_module.projected_line(int(season))
    unit = oline_module.team_line_strength(line).reset_index()
    line = line.rename(columns={
        "team": "Team", "spot": "Spot", "player": "Name", "last_team": "Last team",
        "grade": "Grade", "proj_grade": "Proj grade", "basis": "Basis",
        "proj_pass_block": "Proj pass blk", "proj_run_block": "Proj run blk",
        "draft_number": "Pick", "moved": "Moved",
    })
    unit = unit.rename(columns={
        "team": "Team", "proj_grade": "Proj grade",
        "proj_pass_block": "Proj pass blk", "proj_run_block": "Proj run blk",
        "starters": "Starters", "continuity": "Returning", "newcomers": "New",
        "rookies": "Rookies", "unknown": "Unknown",
    })
    return line, unit


def _ratio(numerator, denominator):
    """Zero denominator -> NaN, never inf. A blank is "cannot say"; an inf sorts first."""
    return numerator / denominator.replace(0, np.nan)


def scope_sidebar(frame, key, positions=("WR", "TE", "RB"), team_column="Team"):
    """The shared filter bar. Returns the filtered frame.

    One bar for every page so a habit learned on one carries to the next, which is the
    whole reason the MLB side has exactly one.
    """
    out = frame
    with st.sidebar:
        available = [p for p in positions if p in set(frame.get("Pos", pd.Series()))]
        if available:
            chosen = st.pills("Position", available, selection_mode="multi",
                              default=available, key=f"{key}_pos", persist_state="session")
            if chosen:
                out = out[out["Pos"].isin(chosen)]
        teams = sorted(t for t in out.get(team_column, pd.Series()).dropna().unique())
        if teams:
            picked = st.multiselect("Team", teams, key=f"{key}_team", persist_state="session",
                                    placeholder="Every team")
            if picked:
                out = out[out[team_column].isin(picked)]
    return out


@st.cache_data(ttl="1h", max_entries=8, show_spinner="Building scheme matchups…")
def scheme_matchup(season, slate_path=None, min_routes=50):
    """Each receiver's man/zone splits set against the scheme his opponent actually plays.

    This is the join the coverage question needs. A receiver's man numbers matter in
    proportion to how much man he will see, and how much man he will see is a property of
    the defence opposite him -- so the two have to be in the same row before either means
    anything about Sunday.

    With no slate the opponent column is blank and the frame is a league-wide profile; with
    one, every receiver carries the man rate of the defence he draws.
    """
    receivers_frame = receiver_scheme(int(season), min_routes=min_routes)
    if receivers_frame.empty:
        return pd.DataFrame()
    defense = defense_scheme(int(season)).set_index("Team")

    out = receivers_frame.copy()
    out["Own man%"] = out["Team"].map(defense["Man%"])

    if slate_path:
        from dashboards import nfl_slates

        players, _ = nfl_slates.slate_players(slate_path)
        opponents = (players.dropna(subset=["Team", "Opp"])
                     .drop_duplicates("Team").set_index("Team")["Opp"])
        out["Opp"] = out["Team"].map(opponents)
        out = out[out["Opp"].notna()]
        out["Opp man%"] = out["Opp"].map(defense["Man%"])
        out["Opp zone%"] = 100.0 - out["Opp man%"]
        # Weighted by what he will actually face. This is the only number on the page that
        # combines the two halves, and it is a *description* of the matchup rather than a
        # projection: the receiver-side efficiency gap behind it self-correlates at +0.19.
        share = out["Opp man%"] / 100.0
        out["Scheme-weighted YPRR"] = out["Man YPRR"] * share + out["Zone YPRR"] * (1 - share)
        out["Scheme-weighted TPRR"] = out["Man TPRR"] * share + out["Zone TPRR"] * (1 - share)
    return out.reset_index(drop=True)


# Depth buckets PFF splits a quarterback's throws into, shallowest first. `behind_los` is
# screens and check-downs; `deep` is 20+ air yards.
DEPTH_BUCKETS = ("behind_los", "short", "medium", "deep")
DEPTH_LABELS = {"behind_los": "Behind LOS", "short": "Short", "medium": "Medium",
                "deep": "Deep"}


@st.cache_data(ttl="1h", max_entries=8, show_spinner="Reading QB depth profiles…")
def quarterbacks(season, min_attempts=100):
    """One row per quarterback: overall line plus a column per depth bucket.

    Built from `passing_depth`, which is 554 columns of depth-by-direction and was unused
    until now. Only the depth half is surfaced: the directional split multiplies the columns
    by four and divides the sample by the same, and a quarterback's left-versus-right numbers
    are mostly a fact about who he was throwing to.
    """

    cached = snapshot.read("quarterbacks", int(season))
    if cached is not None:
        return cached
    frame = pffdata.load("passing_depth", int(season))
    number = lambda c: pd.to_numeric(frame.reindex(columns=[c])[c], errors="coerce")

    out = pd.DataFrame({
        "Name": frame["player"], "Team": frame["team"], "Pos": frame["pos"],
        "G": number("player_game_count"),
        "Attempts": number("base_attempts"),
        "Dropbacks": number("base_dropbacks"),
    })
    for bucket in DEPTH_BUCKETS:
        label = DEPTH_LABELS[bucket]
        out[f"{label} att%"] = number(f"{bucket}_attempts_percent")
        out[f"{label} att"] = number(f"{bucket}_attempts")
        out[f"{label} comp%"] = number(f"{bucket}_completion_percent")
        out[f"{label} YPA"] = number(f"{bucket}_ypa")
        out[f"{label} grade"] = number(f"{bucket}_grades_pass")
        out[f"{label} TD"] = number(f"{bucket}_touchdowns")
        out[f"{label} BTT%"] = number(f"{bucket}_btt_rate")
        out[f"{label} TWP%"] = number(f"{bucket}_twp_rate")

    attempts = out[[f"{DEPTH_LABELS[b]} att" for b in DEPTH_BUCKETS]].sum(axis=1)
    # **The buckets do not cover every attempt.** PFF charts a depth for 86-95% of throws;
    # the rest are throwaways, spikes and batted balls with no meaningful air yards. Carried
    # as a column rather than quietly normalised away, because a quarterback whose mix only
    # accounts for 86% of his attempts is telling you something about how often he is
    # throwing the ball away.
    out["Charted%"] = _ratio(attempts, out["Attempts"]) * 100
    yards = sum(out[f"{DEPTH_LABELS[b]} att"].fillna(0)
                * out[f"{DEPTH_LABELS[b]} YPA"].fillna(0) for b in DEPTH_BUCKETS)
    # Over *charted* attempts, which is the only honest denominator: a throwaway has no
    # depth to attribute the zero yards to.
    out["YPA"] = _ratio(yards, attempts)
    out["Att/G"] = _ratio(out["Attempts"], out["G"])
    # Average depth of the throws he actually made, from the bucket mix. A single number
    # for "how far downfield does this offence look", which the mix carries but does not say.
    midpoints = {"Behind LOS": -1.0, "Short": 4.5, "Medium": 14.5, "Deep": 26.0}
    depth = sum(out[f"{label} att"].fillna(0) * value for label, value in midpoints.items())
    out["aDOT est"] = _ratio(depth, attempts)

    out = out[out["Attempts"] >= min_attempts]
    return out.sort_values("Attempts", ascending=False).reset_index(drop=True)


def quarterback_depth_long(frame):
    """`quarterbacks` reshaped to one row per (quarterback, depth bucket), for stacked bars."""
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for bucket in DEPTH_BUCKETS:
        label = DEPTH_LABELS[bucket]
        part = pd.DataFrame({
            "Name": frame["Name"], "Team": frame["Team"], "Depth": label,
            "Att%": frame[f"{label} att%"], "Attempts": frame[f"{label} att"],
            "YPA": frame[f"{label} YPA"], "Grade": frame[f"{label} grade"],
            "Comp%": frame[f"{label} comp%"], "TD": frame[f"{label} TD"],
        })
        rows.append(part)
    return pd.concat(rows, ignore_index=True)
