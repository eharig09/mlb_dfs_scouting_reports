"""Cached readers over what the NFL pipeline writes into `nfl_boards/`.

The MLB half of this dashboard reads `.cache/report_data/` and never re-runs a report or
touches the network. This is the same bargain for football: it reads the **outputs**
`nfl.optimize` and `nfl.upload` already produced -- the pool, the lineups, the exposure --
so the page opens instantly and shows exactly what was built rather than rebuilding it.

Paths are resolved through `nfl.naming` rather than globbed here. That module already knows
about `.rN` versioning and that the newest version is the one by mtime, not by the highest
number; a second implementation would eventually disagree with it about which file is
current, and would do so silently.
"""

import os

import pandas as pd
import streamlit as st

from nfl import naming

ROOT = naming.OUTPUT_ROOT

# What the page can show, and which pipeline step produces it. Ordered as they are produced,
# so a gap in the middle reads as a missing step rather than as an empty dashboard.
KINDS = {
    "pool": "the editable player pool -- `nfl.optimize --write-pool`",
    "lineups": "built lineups -- `nfl.optimize --n ...`",
    "exposure": "per-player and per-team exposure, written with every lineup set",
    "upload": "the filled DK entries file -- `nfl.upload`",
}


@st.cache_data(ttl="5m", show_spinner=False)
def list_dates(root=ROOT):
    """Dates with anything written, newest first."""
    if not os.path.isdir(root):
        return []
    dates = [name for name in os.listdir(root)
             if os.path.isdir(os.path.join(root, name)) and name[:2] == "20"]
    return sorted(dates, reverse=True)


@st.cache_data(ttl="5m", show_spinner=False)
def list_slates(date, root=ROOT):
    """Slate labels that have any output on a date.

    Union across kinds on purpose: a slate whose pool was written but whose lineups have not
    been built yet still exists, and hiding it would make the page look like the slate was
    never touched.
    """
    found = set()
    for kind in KINDS:
        found.update(naming.find_slates(date, kind, root))
    return sorted(found)


@st.cache_data(ttl="5m", max_entries=32, show_spinner=False)
def read(kind, date, slate, root=ROOT):
    """The newest version of one output, or None when that step has not run."""
    path = naming.latest(kind, date, slate, root)
    if not path or not os.path.exists(path):
        return None
    return pd.read_csv(path)


def path_of(kind, date, slate, root=ROOT):
    """Where `read` got its frame from, for showing the filename on the page."""
    return naming.latest(kind, date, slate, root)


def arrow_safe(frame):
    """Make a frame renderable without changing what it says.

    The same trap the MLB side documents: a column carrying numbers alongside empty strings
    is an object column Arrow refuses outright -- `Could not convert '' with type str` -- and
    Streamlit then renders **nothing at all** where the table should be, with no error on the
    page. The pool writes exactly that shape, because its edit columns ship blank.

    A column is converted only when *every* non-blank value in it parses as a number, so a
    blank becomes a real null and a genuinely mixed text column is left alone. Nothing is
    reformatted or rounded: this changes the type a value is carried in, never the value.
    """
    if frame is None or frame.empty:
        return frame
    out = frame.copy()
    for column in out.columns:
        # **Not `dtype != object`.** pandas 3.0 reads text columns back as its own `str`
        # dtype rather than as `object`, so an object-only guard skips every string column
        # and silently does nothing -- which looks exactly like the bug it exists to fix.
        # Asking whether the column is *already* usable covers both spellings.
        values = out[column]
        if (pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values)
                or pd.api.types.is_datetime64_any_dtype(values)):
            continue
        filled = values[values.astype(str).str.strip() != ""]
        if filled.empty:
            continue
        parsed = pd.to_numeric(filled, errors="coerce")
        if parsed.notna().all():
            out[column] = pd.to_numeric(values.replace("", None), errors="coerce")
    return out


def lineup_summary(lineups):
    """One row per lineup: salary, projection, ceiling, shape and the stack it carries."""
    if lineups is None or lineups.empty:
        return pd.DataFrame()
    rows = []
    for number, group in lineups.groupby("Lineup"):
        first = group.iloc[0]
        pass_game = group[group["Pos"].isin(("QB", "WR", "TE"))]["Team"].value_counts()
        stack = ", ".join(f"{team} x{count}" for team, count in pass_game.items()
                          if count >= 2)
        rows.append({
            "Lineup": int(number),
            "Salary": first.get("Lineup Salary"),
            "Proj": first.get("Lineup Proj"),
            "Ceiling": first.get("Lineup Ceiling"),
            "Shape": first.get("Shape", ""),
            "Anchor": first.get("Anchor", ""),
            "Stack": stack,
        })
    return pd.DataFrame(rows).sort_values("Ceiling", ascending=False)


def split_exposure(exposure):
    """(player rows, team rows) from the combined exposure file."""
    if exposure is None or exposure.empty:
        return pd.DataFrame(), pd.DataFrame()
    if "Level" not in exposure.columns:
        return exposure, pd.DataFrame()
    players = exposure[exposure["Level"] == "player"].drop(columns=["Level"])
    teams = exposure[exposure["Level"] == "team"].drop(columns=["Level"])
    return players.dropna(axis=1, how="all"), teams.dropna(axis=1, how="all")
