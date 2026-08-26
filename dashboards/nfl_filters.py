"""The one shared filter bar for the football pages.

One bar, used everywhere, so a habit learned on one page carries to the next — which is the
whole reason the MLB side has exactly one. Every page passes `show=` to say which controls
it wants; nothing here is page-specific.

**Slate scoping is the control that matters most.** A season profile covers every club, and
DraftKings puts up a subset. Without scoping, a club nobody can roster is ranked, coloured
and counted alongside the ones you can — the MLB side's `off_slate_teams` exists for exactly
this and it is the same mistake in football.
"""

import pandas as pd
import streamlit as st

from dashboards import nfl_slates

EVERY_TEAM = "Every team"
ALL_POSITIONS = ("QB", "RB", "WR", "TE", "DST")


def slate_picker(key, label="Slate", help=None):
    """Choose a DK export, or none. Returns (path, row) with (None, None) for no slate."""
    slates = nfl_slates.list_slates()
    if slates.empty:
        st.caption("No DK salary exports found under "
                   f"`{nfl_slates.SALARY_DIR}` — add one to price and scope these pages.")
        return None, None
    options = [EVERY_TEAM] + [
        f"{row.Label} · {row.Date} · {row.Games}g" for row in slates.itertuples(index=False)]
    chosen = st.selectbox(label, options, key=f"{key}_slate", persist_state="session", help=help)
    if chosen == EVERY_TEAM:
        return None, None
    row = slates.iloc[options.index(chosen) - 1]
    return row["Path"], row


def sidebar(frame, key, show=("slate", "position", "team"), positions=ALL_POSITIONS,
            team_column="Team", pos_column="Pos"):
    """Render the bar and return (filtered frame, slate path, slate row).

    The frame is filtered in the order a reader thinks: narrow to the slate first, because
    that decides *who is even available*, then to positions, then to clubs.
    """
    path, row = None, None
    out = frame

    with st.sidebar:
        st.subheader("Filters", anchor=False)

        if "slate" in show:
            path, row = slate_picker(key,
                                     help="Narrow to the clubs a DK slate can actually "
                                          "roster, and price everyone on it.")
            if path is not None:
                scoped = st.toggle("Only clubs on this slate", value=True,
                                   key=f"{key}_only", persist_state="session",
                                   help="Off, nothing is excluded and the slate is used "
                                        "only for prices.")
                if scoped and team_column in out.columns:
                    teams = set(nfl_slates.teams_on_slate(path))
                    out = out[out[team_column].isin(teams)]

        if "position" in show and pos_column in out.columns:
            available = [p for p in positions if p in set(out[pos_column].dropna())]
            if available:
                chosen = st.pills("Position", available, selection_mode="multi",
                                  default=available, key=f"{key}_pos", persist_state="session")
                if chosen:
                    out = out[out[pos_column].isin(chosen)]

        if "team" in show and team_column in out.columns:
            teams = sorted(t for t in out[team_column].dropna().unique())
            if teams:
                picked = st.multiselect("Team", teams, key=f"{key}_team", persist_state="session",
                                        placeholder="Every team")
                if picked:
                    out = out[out[team_column].isin(picked)]

        if "game" in show and "Game" in out.columns:
            games = sorted(g for g in out["Game"].dropna().unique() if g)
            if games:
                picked = st.multiselect("Game", games, key=f"{key}_game", persist_state="session",
                                        placeholder="Every game")
                if picked:
                    out = out[out["Game"].isin(picked)]

        if "band" in show and "Band" in out.columns:
            bands = [b for _, _, b in nfl_slates.PRICE_BANDS
                     if b in set(out["Band"].dropna())]
            if bands:
                picked = st.pills("Price band", bands, selection_mode="multi",
                                  default=bands, key=f"{key}_band", persist_state="session")
                if picked:
                    out = out[out["Band"].isin(picked)]

    return out, path, row


def season_picker(key, family="receiving_summary", default=1, label="Seasons"):
    """Multi-season selector. Several seasons blend on recency weights."""
    from dashboards import nfl_pff

    seasons = nfl_pff.available_seasons(family)
    if not seasons:
        return []
    with st.sidebar:
        return st.multiselect(label, seasons, default=seasons[:default],
                              key=f"{key}_seasons", persist_state="session",
                              help="Several seasons blend on recency weights "
                                   "1.0 / 0.45 / 0.20, renormalised per player.")


def priced(frame, path, projection=None):
    """Attach salary and value to a profile frame, when a slate is in scope."""
    if path is None:
        return frame
    out = nfl_slates.attach_salary(frame, path)
    if projection and projection in out.columns:
        out = nfl_slates.add_value(out, projection=projection)
    return out
