"""Editable pool file: export the slate to CSV, mark it up in Excel, feed it back.

The columns you edit are **Lock / Exclude / Boost / Min% / Max%**. Everything else is there
to decide with and is ignored on read.

Ported from `dfs.pool`, and the one behaviour worth restating is the important one:
re-exporting **merges** by default rather than overwriting. A slate is not finished when you
first look at it -- inactives land ninety minutes before kickoff, a Friday practice report
moves a starter, prices drift -- so the file you marked up on Thursday is usually not the
last one you want. Rebuilding it from scratch would silently discard days of decisions,
which is the kind of loss you only notice after the lineups are wrong. Pass `merge=False`
for a deliberate clean slate.

Football makes this sharper than baseball, not softer: an NFL slate is marked up over three
or four days rather than one afternoon, so there is simply more to lose.
"""

import math
import os

import pandas as pd

from nfl import naming
from nfl.salaries import canon_team, normalize_name

EDIT_COLUMNS = ["Lock", "Exclude", "Boost", "Min%", "Max%"]

# Written left-to-right: your edit columns first so they are visible without scrolling.
#
# `Game` earns its place next to Team/Opp because the NFL optimizer's two-game rule and the
# DST conflict are both *game*-level facts, and reading them off two separate columns is
# how you talk yourself into a lineup that breaks one. `Basis` comes straight off the board
# so a projection you are about to lock is never separated from what produced it.
VIEW_COLUMNS = [
    "Name", "DK Pos", "Team", "Opp", "Game", "Salary",
    "Proj", "Ceiling", "Floor", "Value", "Own%", "Basis",
]

TRUTHY = {"1", "y", "yes", "true", "x", "lock", "t"}

DEFAULT_EDITS = {"Lock": "", "Exclude": "", "Boost": "1.0", "Min%": "", "Max%": ""}


def pool_path(date, slate=None, root=naming.OUTPUT_ROOT):
    """Where this slate's pool lives. One canonical answer, versioned like everything else."""
    return naming.output_path("pool", date, slate, root)


def _text(value):
    """A cell as clean text, with pandas' missing values read as empty.

    **`str(value or "")` is not enough and the failure is silent.** An empty CSV cell comes
    back as float NaN; NaN is truthy, so `NaN or ""` returns NaN and `str()` makes it the
    four-character string `"nan"`. Every guard below then sees a non-empty value. That is
    what made `_is_edited` report all 66 rows of a fresh pool as hand-marked, which turns
    the one number telling you your edits survived into noise.
    """
    if value is None:
        return ""
    # NaN is the only value not equal to itself; this also catches pd.NA and pd.NaT.
    try:
        if value != value:
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>", "nat"} else text


def _flag(value):
    return _text(value).lower() in TRUTHY


def parse_boost(value):
    """'1.2' -> (1.2, 1.2);  '0.9-1.3' -> (0.9, 1.3);  blank/invalid -> None.

    A range is sampled per lineup, which is how you say "I like him, but not identically in
    every build" without hand-editing between runs.
    """
    text = _text(value)
    if not text:
        return None
    if "-" in text[1:]:
        low, _, high = text.partition("-") if not text.startswith("-") else (text, "", "")
        try:
            low, high = float(low), float(high)
        except ValueError:
            return None
        if math.isnan(low) or math.isnan(high):
            return None
        return (min(low, high), max(low, high))
    try:
        single = float(text)
    except ValueError:
        return None
    return None if math.isnan(single) else (single, single)


def _percent(value):
    """'25' or '25%' or '0.25' -> 0.25. Blank -> None.

    Both spellings are accepted because a spreadsheet will happily turn one into the other,
    and a Max% silently read as 2500% is not a constraint at all.
    """
    text = _text(value).rstrip("%")
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number) or number < 0:
        return None
    return number / 100.0 if number > 1 else number


def _is_edited(values):
    """Has this row been touched? Compared against the defaults, not against blank.

    `Boost` ships as "1.0", so a row that still says 1.0 is untouched -- treating it as an
    edit would make every row look hand-marked and defeat the merge.
    """
    for column, default in DEFAULT_EDITS.items():
        current = _text(values.get(column, ""))
        if current and current != default:
            return True
    return False


def _edit_index(path):
    """({(name, team): edits}, {name: edits}) from an existing pool file.

    **Keyed on name plus team first**, mirroring `dfs.pool`: two players can share a name,
    and quietly moving one player's lock onto another's row is worse than losing it. That
    matters more in football than in baseball, because `nfl.salaries.normalize_name` strips
    generational suffixes for matching -- so a father and son, or any two players separated
    only by a "Jr.", fold to the same key and the team code is the only thing left holding
    them apart. Its own docstring says a real matcher must be team-scoped; this is that.

    The name-only map is the fallback, and it deliberately holds **only unambiguous names**
    -- a name appearing on two rows is dropped from it rather than resolved arbitrarily.
    """
    if not path or not os.path.exists(path):
        return {}, {}
    try:
        frame = pd.read_csv(path, dtype={c: str for c in EDIT_COLUMNS})
    except Exception:
        return {}, {}
    frame.columns = [str(c).strip() for c in frame.columns]
    if "Name" not in frame.columns:
        return {}, {}

    by_pair, seen = {}, {}
    for _, row in frame.iterrows():
        name = _text(row.get("Name"))
        if not name:
            continue
        values = {c: _text(row.get(c, "")) for c in EDIT_COLUMNS if c in frame.columns}
        if not _is_edited(values):
            continue
        key = normalize_name(name)
        team = canon_team(row.get("Team")) if "Team" in frame.columns else ""
        record = {"name": name, "team": team, "edits": values}
        by_pair[(key, team)] = record
        seen.setdefault(key, []).append(record)
    by_name = {key: rows[0] for key, rows in seen.items() if len(rows) == 1}
    return by_pair, by_name


def _merge_edits(out, by_pair, by_name):
    """Carry saved edits onto a freshly built export. Returns (frame, carried, orphaned).

    Two rules, and the second one is where this departs from `dfs.pool`.

    **Team-scoped match first.** A lock belongs to a player on a club, not to a spelling.

    **The name-only fallback requires the name to be unambiguous on _both_ sides.** It
    exists so a player who changed teams since markup keeps his edit -- a real and common
    case in football's off-week churn. But keying on name alone re-opens exactly the hole
    the pair key closes: with two same-named players and only one of them marked, the
    unmarked one matched the fallback and silently inherited the other's lock. Verified on a
    two-team duplicate before the guard, both rows came back locked. So the fallback fires
    only when the name appears once in the saved edits *and* once on the new slate.
    """
    carried, orphaned = 0, []
    if not by_pair and not by_name:
        return out, carried, orphaned

    keys = out["Name"].map(normalize_name)
    teams = (out["Team"].map(canon_team) if "Team" in out.columns
             else pd.Series("", index=out.index))
    counts = keys.value_counts()

    used = set()
    for index, key in keys.items():
        record = by_pair.get((key, teams.at[index]))
        if record is None and counts.get(key, 0) == 1:
            record = by_name.get(key)
        if record is None:
            continue
        for column, value in record["edits"].items():
            if column in out.columns:
                out.at[index, column] = value
        used.add((normalize_name(record["name"]), record["team"]))
        carried += 1

    # Edits whose player is no longer on the slate. Kept as rows with the slate columns
    # blank, so they are obvious in the spreadsheet and survive a re-export rather than
    # being deleted by a rebuild -- the optimizer already reports a lock it cannot find
    # and carries on.
    extra = []
    for record in by_pair.values():
        if (normalize_name(record["name"]), record["team"]) in used:
            continue
        blank = {column: "" for column in out.columns}
        blank.update({c: v for c, v in record["edits"].items() if c in out.columns})
        blank["Name"] = record["name"]
        if "Team" in out.columns:
            blank["Team"] = record["team"]
        extra.append(blank)
        orphaned.append(record["name"])

    if extra:
        out = pd.concat([out, pd.DataFrame(extra, columns=out.columns)], ignore_index=True)
    return out, carried, orphaned


def write_pool(players, date, slate=None, path=None, merge=True, overwrite=False,
               root=naming.OUTPUT_ROOT):
    """Write the editable pool.

    Returns `(path, note, report)`. `report` is `{"carried": n, "orphaned": [names]}` --
    how many hand-marked rows survived the rebuild, and whose edits outlived their player's
    place on the slate. Both are worth printing: `carried` is the signal that days of markup
    were not just discarded.
    """
    target = path or pool_path(date, slate, root)
    by_pair, by_name = _edit_index(target) if merge else ({}, {})

    frame = players.copy()
    if "Name" not in frame.columns:
        raise ValueError("player frame has no Name column")
    if "Value" not in frame.columns and {"Proj", "Salary"} <= set(frame.columns):
        salary = pd.to_numeric(frame["Salary"], errors="coerce").replace(0, pd.NA)
        frame["Value"] = pd.to_numeric(frame["Proj"], errors="coerce") / (salary / 1000.0)

    for column, default in DEFAULT_EDITS.items():
        frame[column] = default

    columns = EDIT_COLUMNS + [c for c in VIEW_COLUMNS if c in frame.columns]
    out = frame.reindex(columns=columns)
    out, carried, orphaned = _merge_edits(out, by_pair, by_name)

    if path is None:
        target, note = naming.resolve("pool", date, slate, root, overwrite=overwrite)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(target)) or ".", exist_ok=True)
        note = None
    out.to_csv(target, index=False)
    return target, note, {"carried": carried, "orphaned": orphaned}


def read_pool(path):
    """Read an edited pool file -> (locks, excludes, boosts, exposure).

    `boosts`:   {name: (low, high)} -- a single value gives low == high.
    `exposure`: {name: (min_fraction, max_fraction)} -- either side may be None.

    Untouched rows are dropped, so a blank file behaves identically to no file at all.
    """
    # Flag columns read as strings: left empty they would come back as float NaN, and a
    # spreadsheet may write anything from "y" to "TRUE" to 1.
    frame = pd.read_csv(path, dtype={c: str for c in EDIT_COLUMNS})
    frame.columns = [str(c).strip() for c in frame.columns]
    if "Name" not in frame.columns:
        raise ValueError(f"{path} has no Name column -- is it a pool file?")

    locks, excludes, boosts, exposure = [], [], {}, {}
    for _, row in frame.iterrows():
        name = _text(row.get("Name"))
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
    """Attach a `Boost` column the optimizer multiplies the objective by.

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


def resolve_exposure(players, exposure, n_lineups, report=None):
    """{player name: (min_lineups, max_lineups)} as integer counts.

    Keyed off the *current* slate, so a marked-up player who has since left it produces no
    constraint rather than an unsatisfiable one. A `Min%` on someone who is no longer
    priced would otherwise make the whole set infeasible with nothing on screen to say why.
    """
    if not exposure or n_lineups <= 0:
        return {}
    present = {normalize_name(n): n for n in players["Name"]}
    resolved, dropped = {}, []
    for key, (low, high) in exposure.items():
        name = present.get(key)
        if name is None:
            dropped.append(key)
            continue
        floor = int(math.ceil(low * n_lineups)) if low is not None else 0
        ceiling = int(math.floor(high * n_lineups)) if high is not None else n_lineups
        floor = max(0, min(floor, n_lineups))
        ceiling = max(0, min(ceiling, n_lineups))
        if ceiling < floor:
            ceiling = floor
        resolved[name] = (floor, ceiling)
    if dropped and report is not None:
        report(dropped)
    return resolved
