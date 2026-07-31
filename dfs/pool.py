"""Editable pool file: export the slate to CSV, mark it up in Excel, feed it back.

The columns you edit are Lock / Exclude / Boost / Min% / Max%. Everything else is there to
decide with and is ignored on read.

Re-exporting **merges** by default rather than overwriting. A slate is not finished when you
first look at it -- games get cached late, a postponement reshuffles the board, prices move
-- so the file you marked up at noon is usually not the last one you want. Rebuilding it
from scratch would silently discard an afternoon of decisions, which is the kind of loss you
only notice after the lineups are wrong. Pass merge=False for a deliberate clean slate.
"""

import math
import os

import pandas as pd

from .salaries import canon_team, normalize_name

POOL_DIR = "dfs_daily_files"

EDIT_COLUMNS = ["Lock", "Exclude", "Boost", "Min%", "Max%"]

# Pool files moved into the night folder alongside everything else for that slate. The old
# flat name is still read when the new one is absent, because the whole point of the merge
# is that an afternoon of hand-marked decisions survives a rebuild -- including the rebuild
# that introduced the new layout.
LEGACY_POOL_DIR = "dfs_daily_files"

# Written left-to-right: your edit columns first so they are visible without scrolling.
VIEW_COLUMNS = [
    "Type", "Name", "DK Pos", "Team", "Opp", "Opp SP", "Slot", "Salary",
    "Proj", "Ceiling", "Floor", "Bust%", "Value", "Ceil Value", "Floor Value",
    "GPP", "CASH", "Edge", "Tier", "Role", "Team Runs", "Lineup", "Why",
]

TRUTHY = {"1", "y", "yes", "true", "x", "lock", "t"}


def pool_path(date, directory=None, slate=None):
    """Where this night's pool file lives, newest version first.

    Falls back to the legacy flat `dfs_daily_files/pool_<date>.csv` so edits made under the
    old layout are still found and merged forward.
    """
    from .naming import OUTPUT_ROOT, latest, output_path

    root = directory or OUTPUT_ROOT
    found = latest("pool", date, slate, root)
    if found:
        return found
    legacy = os.path.join(LEGACY_POOL_DIR, f"pool_{date}.csv")
    if os.path.exists(legacy):
        return legacy
    return output_path("pool", date, slate, root)


DEFAULT_EDITS = {"Lock": "", "Exclude": "", "Boost": "1.0", "Min%": "", "Max%": ""}


def _is_edited(values):
    """True when a row carries a decision worth preserving, rather than export defaults."""
    for column, default in DEFAULT_EDITS.items():
        text = str(values.get(column) or "").strip()
        if text.lower() in {"", "nan", "none"}:
            continue
        if column == "Boost":
            parsed = parse_boost(text)
            if parsed is None or parsed == (1.0, 1.0):
                continue
        elif column in {"Lock", "Exclude"} and not _flag(text):
            continue
        return True
    return False


def _edit_index(path):
    """({(name, team): edits}, {name: edits}) from an existing pool file.

    Keyed on name plus team first: two players can share a name, and quietly moving one
    player's lock onto another's row is worse than losing it.
    """
    try:
        frame = pd.read_csv(path, dtype={c: str for c in EDIT_COLUMNS})
    except Exception:
        return {}, {}
    frame.columns = [str(c).strip() for c in frame.columns]
    if "Name" not in frame.columns:
        return {}, {}

    by_pair, seen = {}, {}
    for _, row in frame.iterrows():
        name = str(row.get("Name") or "").strip()
        if not name:
            continue
        edits = {c: ("" if pd.isna(row.get(c)) else str(row.get(c)).strip())
                 for c in EDIT_COLUMNS if c in frame.columns}
        if not _is_edited(edits):
            continue
        key = normalize_name(name)
        team = canon_team(row.get("Team")) if "Team" in frame.columns else ""
        record = {"name": name, "team": team, "edits": edits}
        by_pair[(key, team)] = record
        seen.setdefault(key, []).append(record)
    by_name = {key: rows[0] for key, rows in seen.items() if len(rows) == 1}
    return by_pair, by_name


def write_pool(players, date, directory=None, path=None, merge=True, slate=None,
               overwrite=False):
    """Export the slate as an editable pool file, sorted best-first within type.

    Returns (path, report). With merge=True an existing file's edits are carried onto the
    new export: matching players keep their settings, players new to the slate arrive with
    defaults, and edits for players who have since left the slate are kept as trailing rows
    rather than thrown away -- a player can drop out because a game is merely uncached, and
    that is not a good enough reason to delete a decision.

    Edits are read from wherever the pool currently lives -- including the legacy flat path
    -- but written to the current convention, so an old file is migrated by being merged
    forward rather than by being moved.
    """
    from .naming import OUTPUT_ROOT, resolve

    source = path or pool_path(date, directory, slate)
    if path:
        target, _ = path, None
    else:
        target, _ = resolve("pool", date, slate, directory or OUTPUT_ROOT, overwrite=True)
        # A pool file is the one output that is *meant* to be replaced in place: it carries
        # your edits forward by merging them, so versioning it would strand them in .r1
        # while the optimizer read an empty .r2.
    path = target
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    frame = players.copy()
    frame = frame.sort_values(["Type", "GPP"], ascending=[True, False])

    out = pd.DataFrame(index=frame.index)
    for column, default in DEFAULT_EDITS.items():
        out[column] = default
    for column in VIEW_COLUMNS:
        if column in frame.columns:
            out[column] = frame[column]

    report = {"carried": [], "new": 0, "orphaned": [], "merged": False, "from": None}
    if merge and os.path.exists(source):
        out, report = _merge_edits(out, source)
        # Named so a migration from the legacy flat path is visible rather than assumed.
        report["from"] = source if os.path.abspath(source) != os.path.abspath(path) else None

    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path, report


def _merge_edits(out, path):
    """Carry edits from the pool file at `path` onto a freshly built export."""
    by_pair, by_name = _edit_index(path)
    report = {"carried": [], "new": 0, "orphaned": [], "merged": True}
    if not by_pair and not by_name:
        return out, report

    used = set()
    for index, row in out.iterrows():
        key = normalize_name(row.get("Name"))
        team = canon_team(row.get("Team"))
        record = by_pair.get((key, team)) or by_name.get(key)
        if record is None:
            report["new"] += 1
            continue
        for column, value in record["edits"].items():
            if column in out.columns:
                out.at[index, column] = value
        used.add((normalize_name(record["name"]), record["team"]))
        report["carried"].append(str(row.get("Name")))

    # Edits whose player is no longer on the slate. Kept as rows with the slate columns
    # blank, so they are obvious in the spreadsheet and survive a re-export -- the
    # optimizer already reports a lock it cannot find and carries on.
    orphans = []
    for record in by_pair.values():
        if (normalize_name(record["name"]), record["team"]) in used:
            continue
        blank = {column: "" for column in out.columns}
        blank.update(record["edits"])
        blank["Name"] = record["name"]
        if "Team" in out.columns:
            blank["Team"] = record["team"]
        orphans.append(blank)
        report["orphaned"].append(record["name"])

    if orphans:
        out = pd.concat([out, pd.DataFrame(orphans, columns=out.columns)], ignore_index=True)
    return out, report


def _flag(value):
    return str(value).strip().lower() in TRUTHY


def parse_boost(value):
    """Read a Boost cell as (low, high).

    Accepts a single number ("1.25") or a range ("1.1-1.3", "1.1:1.3", "1.1 to 1.3").
    A range is sampled per lineup, which spreads a player across a set of lineups instead
    of pinning them into every one -- diversity you control per player.
    """
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    text = text.replace("to", "-").replace(":", "-")
    # Keep a leading minus from splitting a negative (which is invalid anyway, but the
    # split must not silently produce garbage).
    parts = [p.strip() for p in text.lstrip("-").split("-") if p.strip()]
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return None
    if not values or any(v <= 0 for v in values):
        return None
    return (min(values), max(values)) if len(values) > 1 else (values[0], values[0])


def _percent(value):
    """Read a Min%/Max% cell as a 0-1 fraction. Accepts '40', '40%' or '0.4'."""
    text = str(value or "").strip().rstrip("%")
    if not text or text.lower() in {"nan", "none"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if number < 0:
        return None
    fraction = number / 100.0 if number > 1 else number
    return min(1.0, fraction)


def read_pool(path):
    """Read an edited pool file -> (locks, excludes, boosts, exposure).

    boosts:   {name: (low, high)} — a single value gives low == high.
    exposure: {name: (min_fraction, max_fraction)} — either side may be None.
    Untouched rows are dropped, so a blank file behaves identically to no file at all.
    """
    # Flag columns read as strings: left empty they would otherwise come back as float
    # NaN, and a spreadsheet may write anything from "y" to "TRUE" to 1.
    frame = pd.read_csv(path, dtype={"Lock": str, "Exclude": str, "Boost": str,
                                     "Min%": str, "Max%": str})
    frame.columns = [str(c).strip() for c in frame.columns]
    if "Name" not in frame.columns:
        raise ValueError(f"{path} has no Name column — is it a pool file?")

    locks, excludes, boosts, exposure = [], [], {}, {}
    for _, row in frame.iterrows():
        name = str(row.get("Name") or "").strip()
        if not name:
            continue
        if "Exclude" in frame.columns and _flag(row.get("Exclude")):
            excludes.append(name)
            continue                      # an excluded player's other settings are moot
        key = normalize_name(name)
        if "Lock" in frame.columns and _flag(row.get("Lock")):
            locks.append(name)
        if "Boost" in frame.columns:
            parsed = parse_boost(row.get("Boost"))
            if parsed and parsed != (1.0, 1.0):
                boosts[key] = parsed
        low = _percent(row.get("Min%")) if "Min%" in frame.columns else None
        high = _percent(row.get("Max%")) if "Max%" in frame.columns else None
        if low is not None or high is not None:
            exposure[key] = (low, high)
    return locks, excludes, boosts, exposure


def apply_boosts(players, boosts, rng=None):
    """Attach a Boost column the optimizer multiplies the objective by.

    Ranged boosts are sampled once per call, so calling this before each lineup produces a
    different draw and therefore a different lineup.
    """
    frame = players.copy()
    if not boosts:
        frame["Boost"] = 1.0
        return frame

    def draw(name):
        span = boosts.get(normalize_name(name))
        if span is None:
            return 1.0
        low, high = span
        if high <= low or rng is None:
            return low
        return float(rng.uniform(low, high))

    frame["Boost"] = frame["Name"].map(draw)
    return frame


def resolve_exposure(players, exposure, n_lineups):
    """{player name: (min_lineups, max_lineups)} as integer counts."""
    limits = {}
    for _, row in players.iterrows():
        span = exposure.get(normalize_name(row["Name"]))
        if not span:
            continue
        low, high = span
        limits[row["Name"]] = (
            0 if low is None else int(math.ceil(low * n_lineups)),
            n_lineups if high is None else int(math.floor(high * n_lineups)),
        )
    return limits
