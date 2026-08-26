"""DK slates: which exports are on disk, who is on them, and what everyone costs.

This is what turns the PFF pages from a scouting library into a slate tool. A season profile
says how a player is used; a slate says whether you can roster him this week and at what
price. Joining the two is the whole point of the filter bar.

**Cached profiles are not the DK slate**, and the MLB side learned that the hard way: without
scoping, a club nobody can roster is ranked, coloured and counted alongside the ones you can.
Every page that offers a slate filter routes through `teams_on_slate` here.

The salary join is **team-scoped first, name second**, for the reason `nfl.salaries` spells
out in its own docstring: `normalize_name` strips generational suffixes, so two players
separated only by a "Jr." fold to one key and the club is the only thing left holding them
apart.
"""

import os

import numpy as np
import pandas as pd
import streamlit as st

from nfl import naming
from nfl.salaries import canon_team, load_dk_export, normalize_name

SALARY_DIR = os.path.join("nfl", "nfl_dfs", "nfl_daily_files")
ENTRIES_DIR = os.path.join("nfl", "Lineups")

# Price bands, in DK dollars. Cut where the roster actually thinks: a punt is a salary you
# are spending to afford someone else, and the top band is what you are affording. Not
# fitted -- they are reading aids, and the page says so.
PRICE_BANDS = [(0, 4000, "Punt"), (4000, 5500, "Value"),
               (5500, 7000, "Mid"), (7000, 100000, "Stud")]

# The order bands are *shown* in, which is not the order they are defined in. Studs first
# and punts last, because a board is read from the money down: what am I paying up for, then
# what am I paying for it with.
BAND_DISPLAY_ORDER = ("Stud", "Value", "Mid", "Punt")


@st.cache_data(ttl="10m", show_spinner=False)
def list_slates(directory=SALARY_DIR):
    """Every DK salary export on disk, described. Newest first.

    Labelled as a set rather than one at a time: a Sunday `main` slate and a Sunday
    `early`-only block share a 1:00PM kickoff and both come back "main" in isolation. Only
    comparing a week's exports to each other separates them -- see `nfl.naming.label_slates`.
    """
    if not os.path.isdir(directory):
        return pd.DataFrame()
    paths = [os.path.join(directory, name) for name in os.listdir(directory)
             if name.lower().endswith(".csv")]
    if not paths:
        return pd.DataFrame()

    infos = []
    for path in sorted(paths):
        try:
            infos.append(naming.export_info(path))
        except Exception:
            continue
    if not infos:
        return pd.DataFrame()

    by_date = {}
    for info in infos:
        by_date.setdefault(info.get("date") or "", []).append(info)

    rows = []
    for date, group in by_date.items():
        labels = naming.label_slates(group, date)
        for info in group:
            rows.append({
                "Label": labels.get(info["path"], naming.UNKNOWN_SLATE),
                "Date": date,
                "Games": len(info["games"]),
                "Players": info["players"],
                "Kickoff": _clock(info.get("first")),
                "Path": info["path"],
                "File": info["name"],
            })
    frame = pd.DataFrame(rows)
    return frame.sort_values(["Date", "Games"], ascending=[False, False]).reset_index(drop=True)


def _clock(minutes):
    if minutes is None:
        return ""
    hour, minute = divmod(int(minutes), 60)
    suffix = "AM" if hour < 12 else "PM"
    display = hour % 12 or 12
    return f"{display}:{minute:02d}{suffix}"


@st.cache_data(ttl="10m", max_entries=16, show_spinner=False)
def slate_players(path, drop_out=True):
    """One DK export as a priced player frame, plus the report of who was dropped."""
    players, report = load_dk_export(path, drop_out=drop_out)
    players = players.copy()
    players["Value"] = np.nan
    players["Band"] = players["Salary"].map(price_band)
    players["key"] = players["Name"].map(normalize_name)
    return players, report


def price_band(salary):
    """A DK salary -> the band it sits in. Reading aid, not a fitted boundary."""
    if pd.isna(salary):
        return ""
    for low, high, label in PRICE_BANDS:
        if low <= float(salary) < high:
            return label
    return ""


def teams_on_slate(path):
    """Canonical team codes a slate can roster."""
    players, _ = slate_players(path)
    return sorted(players["Team"].dropna().unique())


def games_on_slate(path):
    players, _ = slate_players(path)
    return sorted(g for g in players["Game"].dropna().unique() if g)


def attach_salary(frame, path, name_column="Name", team_column="Team"):
    """Join DK salary and slate context onto a profile frame.

    Adds `Salary`, `DK Pos`, `Game`, `Opp`, `Band`, `Status` and `On slate`. A player the
    slate does not price keeps every profile column and is simply marked `On slate = False`
    -- **not dropped**, because "not on this slate" is a fact worth seeing next to a profile
    rather than a silent disappearance.
    """
    players, _ = slate_players(path)
    out = frame.copy()
    out["key"] = out[name_column].map(normalize_name)
    out["_team"] = out[team_column].map(canon_team) if team_column in out.columns else ""

    by_pair = {(row.key, row.Team): row for row in players.itertuples(index=False)}
    counts = players["key"].value_counts()
    by_name = {row.key: row for row in players.itertuples(index=False)
               if counts.get(row.key, 0) == 1}

    fields = ("Salary", "DK Pos", "Game", "Opp", "Band", "Status")
    collected = {field: [] for field in fields}
    on_slate = []
    for key, team in zip(out["key"], out["_team"]):
        match = by_pair.get((key, team)) or by_name.get(key)
        on_slate.append(match is not None)
        for field in fields:
            collected[field].append(getattr(match, field.replace(" ", "_"), None)
                                    if match is not None else None)
    for field in fields:
        out[field] = collected[field]
    out["On slate"] = on_slate
    return out.drop(columns=["_team"])


def add_value(frame, projection="Proj", salary="Salary"):
    """Points per $1,000 of salary, plus the value rank within the player's price band.

    Two numbers rather than one on purpose. Raw value/$ always favours the cheapest player
    on the board, so a punt with a replacement-level projection outranks every real starter;
    ranking **within a band** asks the question a roster actually asks, which is "who is the
    best use of this slot at this price".
    """
    out = frame.copy()
    points = pd.to_numeric(out.reindex(columns=[projection])[projection], errors="coerce")
    dollars = pd.to_numeric(out.reindex(columns=[salary])[salary], errors="coerce")
    out["Value"] = points / (dollars / 1000.0).replace(0, np.nan)
    if "Band" in out.columns:
        out["Band rank"] = (out.groupby("Band")["Value"]
                            .rank(ascending=False, method="min"))
    return out


@st.cache_data(ttl="10m", show_spinner=False)
def list_entry_files(directory=ENTRIES_DIR):
    """DK entry exports on disk -- the lineups actually submitted."""
    if not os.path.isdir(directory):
        return []
    return sorted((os.path.join(directory, name) for name in os.listdir(directory)
                   if name.lower().endswith(".csv")),
                  key=os.path.getmtime, reverse=True)


@st.cache_data(ttl="10m", max_entries=8, show_spinner="Reading entered lineups…")
def entered_lineups(path):
    """A filled DK entries export -> one row per player per entry.

    Reads the roster cells rather than the embedded player list: the cells are what was
    actually submitted. A blank entry -- a contest row with no lineup in it -- is dropped,
    because it is a slot you own rather than a lineup you entered.
    """
    from nfl import upload

    rows, kind, slot_start, index = upload.read_template(path)
    slots = list(upload.SLOT_ORDER)
    width = slot_start + len(slots)

    records = []
    for row in rows[1:]:
        if len(row) < 4 or not str(row[0]).strip().isdigit():
            continue
        padded = list(row) + [""] * (width - len(row))
        cells = padded[slot_start:width]
        if not any(str(c).strip() for c in cells):
            continue
        for slot, cell in zip(slots, cells):
            player_id, name = upload.parse_cell(cell)
            if not (player_id or name):
                continue
            records.append({
                "Entry": str(row[0]).strip(),
                "Contest": str(row[1]).strip() if len(row) > 1 else "",
                "Entry fee": str(row[3]).strip() if len(row) > 3 else "",
                "Slot": slot, "Name": name, "DK ID": player_id,
            })
    frame = pd.DataFrame(records)
    if not frame.empty:
        frame["key"] = frame["Name"].map(normalize_name)
    return frame


def entry_exposure(entries, players=None):
    """Per-player exposure across the entered set, priced when a slate is supplied."""
    if entries is None or entries.empty:
        return pd.DataFrame()
    total = entries["Entry"].nunique()
    grouped = (entries.groupby(["Name", "key", "Slot"], as_index=False)
               .agg(Entries=("Entry", "nunique")))
    grouped = (grouped.groupby(["Name", "key"], as_index=False)
               .agg(Entries=("Entries", "max"),
                    Slots=("Slot", lambda s: ", ".join(sorted(set(s))))))
    grouped["Exposure%"] = grouped["Entries"] / max(total, 1) * 100

    if players is not None and not players.empty:
        lookup = players.drop_duplicates("key").set_index("key")
        for column in ("Team", "Opp", "Salary", "DK Pos", "Band"):
            if column in lookup.columns:
                grouped[column] = grouped["key"].map(lookup[column])
    return grouped.sort_values("Entries", ascending=False).reset_index(drop=True)


@st.cache_data(ttl="10m", show_spinner=False)
def projections():
    """The newest PFF projection export, as points **per game**.

    Per game, not per season: a slate is one week, and a season total silently rewards
    whoever PFF expects to play seventeen games. `nfl.pffdata.latest` picks the file, so a
    fresher pull landing beside an older one is used without anything being renamed.
    """
    from nfl import pffdata, snapshot

    # Snapshot first: resolving the newest export goes through the catalog, which
    # fingerprints raw files and falls through to the network when there are none.
    cached = snapshot.read("projections")
    if cached is not None:
        return cached, "nfl/snapshot"

    try:
        path = pffdata.latest("projections")
    except pffdata.PffDataError:
        return pd.DataFrame(), None
    frame = pd.read_csv(path)
    games = pd.to_numeric(frame.reindex(columns=["games"])["games"],
                          errors="coerce").replace(0, np.nan)
    out = pd.DataFrame({
        "Name": frame.reindex(columns=["playerName"])["playerName"],
        "PFF team": frame.reindex(columns=["teamName"])["teamName"].map(canon_team),
        "Proj": pd.to_numeric(frame.reindex(columns=["fantasyPoints"])["fantasyPoints"],
                              errors="coerce") / games,
        "Proj season": pd.to_numeric(
            frame.reindex(columns=["fantasyPoints"])["fantasyPoints"], errors="coerce"),
        "Proj games": games,
    })
    for source, target in (("passAtt", "Pass att"), ("rushAtt", "Rush att"),
                           ("recvTargets", "Targets"), ("recvReceptions", "Rec"),
                           ("passYds", "Pass yds"), ("rushYds", "Rush yds"),
                           ("recvYds", "Rec yds"), ("passTd", "Pass TD"),
                           ("rushTd", "Rush TD"), ("recvTd", "Rec TD")):
        if source in frame.columns:
            out[target] = pd.to_numeric(frame[source], errors="coerce") / games
    out["key"] = out["Name"].map(normalize_name)
    return out.dropna(subset=["key"]).drop_duplicates("key"), path


def attach_projection(frame, name_column="Name"):
    """Join PFF per-game projections onto a priced board. Returns (frame, source path).

    Matched on name alone, because the projection export is the one source that does not
    carry a canonical team: it ships DK-style codes and a player traded since the pull would
    fail a team-scoped join that a name-only one gets right. Duplicated names are dropped
    from the index rather than resolved arbitrarily.
    """
    projected, path = projections()
    out = frame.copy()
    if projected.empty:
        out["Proj"] = np.nan
        return out, None
    keys = out[name_column].map(normalize_name)
    lookup = projected.set_index("key")
    for column in ("Proj", "Targets", "Rush att", "Pass att", "Rec", "Rec yds",
                   "Rush yds", "Pass yds", "Rec TD", "Rush TD", "Pass TD"):
        if column in lookup.columns:
            out[column] = keys.map(lookup[column])
    return out, path
