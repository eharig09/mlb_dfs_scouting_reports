"""One filter bar, used by every view.

Filters that differ between pages are worse than no filters: a reader who narrows the Slate
page to two clubs and a position, then clicks to Value and silently sees the whole board,
has been told something false about what they are looking at. So the controls, their keys
and their semantics live here, and each page asks for the subset it can support.

What each filter is actually doing
----------------------------------
* **Slate** is DraftKings' contest grouping (main, early, late, turbo) and is a property of
  the *salary file*, not of the game — one game appears on several slates. Membership comes
  from which `DKSalaries_<date>_<slate>.csv` a player is priced in.
* **Position** is the DK roster slot, which is not the defensive position: a player is
  listed at every slot he qualifies for (`OF`, `1B/OF`), so the filter matches *any* of a
  player's slots rather than the whole string.
* **Team** and **Signal** are the report's own.
"""

import glob
import os

import pandas as pd
import streamlit as st

from dashboards import data, salaries

#: The slates DraftKings runs, in the order a reader thinks about them.
SLATE_ORDER = ["main", "early", "afternoon", "turbo", "late", "express"]


@st.cache_data(ttl="30m", max_entries=8, show_spinner=False)
def slate_membership(date, salary_dir=salaries.SALARY_DIR):
    """`{folded name: [slate, ...]}` for a date.

    `salaries_for_date` unions the slates and keeps one row per player, which is right for
    pricing — prices agree across slates — but it throws away exactly the membership this
    filter needs, so the files are read again here rather than the union re-used.
    """
    found = {}
    for path in glob.glob(os.path.join(salary_dir, f"DKSalaries_{date}_*.csv")):
        slate = os.path.basename(path).rsplit("_", 1)[-1][:-4].lower()
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if "Name" not in frame.columns:
            continue
        for name in frame["Name"].dropna():
            found.setdefault(salaries.name_key(name), set()).add(slate)
    return {key: sorted(value) for key, value in found.items()}


def slates_for_date(date, salary_dir=salaries.SALARY_DIR):
    """Every slate DraftKings ran on a date, in reading order."""
    slates = set()
    for path in glob.glob(os.path.join(salary_dir, f"DKSalaries_{date}_*.csv")):
        slates.add(os.path.basename(path).rsplit("_", 1)[-1][:-4].lower())
    ordered = [s for s in SLATE_ORDER if s in slates]
    return ordered + sorted(slates - set(ordered))


def attach_slates(frame, date, salary_dir=salaries.SALARY_DIR):
    """Add a `slates` column listing every DK slate a player is priced on."""
    if frame is None or frame.empty:
        return frame
    membership = slate_membership(date, salary_dir)
    out = frame.copy()
    out["slates"] = [membership.get(salaries.name_key(name), [])
                     for name in out["Name"]]
    return out


def _slot_matches(value, wanted):
    """Whether a DK roster-position string covers any wanted slot.

    DK writes every slot a player qualifies for into one field (`1B/OF`), so an equality
    test drops the multi-position players — who are precisely the ones a roster is being
    built around.
    """
    if not wanted:
        return True
    slots = {part.strip().upper() for part in str(value or "").replace(",", "/").split("/")}
    return bool(slots & {w.upper() for w in wanted})


def sidebar(frame, prefix, date=None, show=("signal", "team", "position", "slate", "ab"),
            salary_dir=salaries.SALARY_DIR):
    """Draw the shared filters and return the narrowed frame.

    Only the controls a page can support are drawn — a position filter over a frame with no
    roster slot would be a control that does nothing, which is worse than its absence.
    """
    if frame is None or frame.empty:
        return frame

    working = frame
    if "slate" in show and date and "slates" not in working.columns:
        working = attach_slates(working, date, salary_dir)

    with st.sidebar:
        st.subheader("Filter", anchor=False)
        chosen = {}

        if "signal" in show and "Signal" in working.columns:
            chosen["signal"] = st.pills("Signal", data.SIGNAL_ORDER, selection_mode="multi",
                                        default=data.SIGNAL_ORDER, key=f"{prefix}_signal",
                                        persist_state="session")

        if "slate" in show and date:
            available = slates_for_date(date, salary_dir)
            if available:
                chosen["slate"] = st.pills("DK slate", available, selection_mode="multi",
                                           default=[], key=f"{prefix}_slate",
                                           help="Leave empty for every slate. A game "
                                                "appears on more than one.",
                                           persist_state="session")

        if "position" in show:
            column = ("DK Pos" if "DK Pos" in working.columns
                      else ("Roster Position" if "Roster Position" in working.columns
                            else None))
            if column:
                slots = sorted({part.strip().upper()
                                for value in working[column].dropna()
                                for part in str(value).replace(",", "/").split("/")
                                if part.strip()})
                if slots:
                    chosen["position"] = st.multiselect(
                        "Position", slots, default=[], key=f"{prefix}_position",
                        help="Matches any slot a player qualifies for.",
                        persist_state="session")
                    chosen["_position_column"] = column

        if "team" in show and "Team" in working.columns:
            chosen["team"] = st.multiselect("Team",
                                            sorted(working["Team"].dropna().unique()),
                                            default=[], key=f"{prefix}_team",
                                            help="Leave empty for every club.",
                                            persist_state="session")

        if "bats" in show and "Bats" in working.columns:
            chosen["bats"] = st.segmented_control("Bats", ["All", "L", "R", "S"],
                                                  default="All", key=f"{prefix}_bats",
                                                  persist_state="session")

        if "ab" in show and "Season AB" in working.columns:
            chosen["ab"] = st.slider("Minimum season at-bats", 0, 500, 0, step=25,
                                     key=f"{prefix}_ab",
                                     persist_state="session")

    return apply_filters(working, chosen)


def apply_filters(frame, chosen):
    """Narrow a frame by the selections `sidebar` gathered. Pure, so it is testable."""
    if frame is None or frame.empty:
        return frame
    view = frame

    signal = chosen.get("signal")
    if signal and "Signal" in view.columns:
        view = view[view["Signal"].isin(signal)]

    slate = chosen.get("slate")
    if slate and "slates" in view.columns:
        wanted = {s.lower() for s in slate}
        view = view[view["slates"].map(lambda s: bool(wanted & set(s or [])))]

    position = chosen.get("position")
    column = chosen.get("_position_column")
    if position and column and column in view.columns:
        view = view[view[column].map(lambda v: _slot_matches(v, position))]

    team = chosen.get("team")
    if team and "Team" in view.columns:
        view = view[view["Team"].isin(team)]

    bats = chosen.get("bats")
    if bats and bats != "All" and "Bats" in view.columns:
        view = view[view["Bats"].astype(str).str.upper() == bats]

    minimum = chosen.get("ab")
    if minimum and "Season AB" in view.columns:
        view = view[view["Season AB"].fillna(0) >= minimum]

    return view


#: What a matchup score can be read against. Each is a real cached column; the label says
#: what the reader is actually asking.
#:
BASES = {
    "Season OPS": "Season OPS — his overall level",
    "Off L28": "Recent offence (last 28 days)",
    "Off Szn": "Season offence index",
    "Platoon OPS": "OPS against this hand",
    "Arsenal OPS": "OPS against this arsenal",
    "xwOBA": "xwOBA against this arsenal",
    "SLG": "SLG against this arsenal",
    "HardHit%": "Hard-hit rate against this arsenal",
    "Season ISO": "Isolated power (season)",
    "Season AVG": "Batting average (season)",
    "Season OBP": "On-base percentage (season)",
}


def available_bases(frame, minimum=20):
    """The bases this frame can actually support, with enough rows to plot."""
    if frame is None or frame.empty:
        return {}
    return {key: label for key, label in BASES.items()
            if key in frame.columns and frame[key].notna().sum() >= minimum}


def priced_teams(date, salary_dir=salaries.SALARY_DIR):
    """Clubs with at least one priced player on a date — i.e. actually on a DK slate.

    Derived from the salary files rather than from the schedule, because those are the only
    statement of what DraftKings actually put up. A club can play and not be on any slate.
    """
    teams = set()
    for path in glob.glob(os.path.join(salary_dir, f"DKSalaries_{date}_*.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        column = "TeamAbbrev" if "TeamAbbrev" in frame.columns else None
        if column:
            teams.update(str(t).upper() for t in frame[column].dropna())
    return teams


def on_slate(frame, date, salary_dir=salaries.SALARY_DIR, team_column="Team"):
    """Drop players from clubs that are not on any DK slate that day.

    **Cached games are not the slate.** The dashboard reads every game the report ran, and
    DraftKings puts up a subset: on 2026-08-21 thirteen games were cached but Atlanta and
    Milwaukee were on no slate at all, so eighteen hitters were being ranked, coloured and
    fed into the analysis while being impossible to roster. On 2026-08-19 it was four clubs
    and thirty-seven hitters.

    Falls back to the frame unchanged when no salary file exists for the date — with nothing
    to compare against, dropping everything would be worse than showing a superset.
    """
    if frame is None or frame.empty or team_column not in frame.columns:
        return frame
    priced = priced_teams(date, salary_dir)
    if not priced:
        return frame
    keep = frame[team_column].astype(str).str.upper().isin(priced)
    return frame[keep]


def off_slate_teams(frame, date, salary_dir=salaries.SALARY_DIR, team_column="Team"):
    """Which clubs in a frame are not on any DK slate — so a page can say so."""
    if frame is None or frame.empty or team_column not in frame.columns:
        return []
    priced = priced_teams(date, salary_dir)
    if not priced:
        return []
    present = {str(t).upper() for t in frame[team_column].dropna()}
    return sorted(present - priced)
