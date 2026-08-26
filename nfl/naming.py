"""One place that decides what every NFL DFS file is called.

    nfl_boards/<date>/<kind>_<slate>.<ext>

The MLB twin is `dfs.naming` and this is a deliberate port rather than a shared module,
for the reason the whole `nfl/` package is a fork: MLB is live and being submitted against,
and a refactor that touches it to serve football is a bad trade while that is true. The
*mechanism* here is the proven one. What changes is the vocabulary, because a football week
is not a baseball night.

Two things drove the layout, both inherited. A slate is the unit you work in and archive, so
it gets a folder. And every name carries the **slate label**, because a date alone does not
identify a contest -- an NFL Sunday routinely ships a main slate, an early-only block and a
primetime showdown, and under date-only naming the second run silently overwrites the first.

Nothing is ever overwritten by default. A rerun writes `pool_main.r2.csv` beside
`pool_main.csv`, so two runs can be compared and an accidental rebuild cannot destroy the
file you uploaded from. Pass `overwrite=True` to replace in place.

## Where football differs from baseball

A baseball night is one evening and every slate sits inside it, so MLB identifies a slate by
first pitch alone. Football spreads across **Thursday, Sunday and Monday**, single-game
showdowns run most weeks, and the Sunday blocks overlap by design:

    main            the 1:00PM ET block, usually carrying the 4:05/4:25 games too
    early           1:00PM games only -- same first kickoff as main, a strict subset
    afternoon       the 4:00PM+ block
    primetime       Sunday night
    thursday        Thursday night
    monday          Monday night
    showdown-XXYYY  any single game

`early` is football's `turbo`: a fragment that **cannot be told from the main slate by
kickoff time**, because both start at 1:00PM. Only comparing a week's exports to each other
separates them -- see `label_slates`.
"""

import os
import re
from collections import defaultdict
from datetime import datetime

OUTPUT_ROOT = "nfl_boards"

# kind -> (extension, one-line description). The description is what `nfl.files` prints, so
# a folder full of csvs explains itself without anyone having to open one.
KINDS = {
    "board":    (".xlsx", "the slate report workbook"),
    "slate":    (".csv",  "every projected player, all columns"),
    "stacks":   (".csv",  "team stacks ranked"),
    "pool":     (".csv",  "editable pool: Lock / Exclude / Boost / Min% / Max%"),
    "lineups":  (".csv",  "optimizer output, one row per player per lineup"),
    "exposure": (".csv",  "per-player and per-team exposure across the lineup set"),
    "upload":   (".csv",  "DraftKings upload file, one row per lineup"),
    "swap":     (".csv",  "late-swap upload file"),
    "review":   (".md",   "post-game scoring vs actual results"),
}

# A slate that cannot be identified from a filename or its own contents. Deliberately not
# "main": a wrong-but-plausible label is harder to notice than an obviously unknown one.
UNKNOWN_SLATE = "unknown"

# The fragment that overlaps the main slate's opening window. See the module docstring.
FRAGMENT_SLATE = "early"

DATE_RE = re.compile(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})")
VERSION_RE = re.compile(r"^(?P<stem>.+?)\.r(?P<n>\d+)$")

# "LV@HOU 08/20/2026 08:00PM ET"
GAME_INFO_RE = re.compile(
    r"^\s*(?P<away>[A-Z]{2,4})@(?P<home>[A-Z]{2,4})\s+"
    r"(?P<month>\d{2})/(?P<day>\d{2})/(?P<year>\d{4})\s+"
    r"(?P<hour>\d{1,2}):(?P<minute>\d{2})(?P<meridiem>AM|PM)", re.IGNORECASE)

# Kickoff windows, in minutes past midnight ET. Football's blocks are scheduled rather than
# staggered the way baseball first pitches are, so these are real boundaries and not buckets
# chosen to spread a continuum.
_AFTERNOON_START = 15 * 60 + 30      # 3:30PM ET -- after the 1:00 block, before 4:05
_PRIMETIME_START = 18 * 60 + 30      # 6:30PM ET


def slate_slug(text):
    """Free text -> a short filename-safe slate label ('Main' -> 'main')."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(text or "").strip().lower()).strip("-")
    return slug or UNKNOWN_SLATE


def slug_from_filename(name, date=None):
    """Pull the slate label out of a DK export's filename.

    'DKSalaries_2026-09-14_main.csv' -> 'main'. Everything structural is stripped -- the
    DKSalaries prefix, the date, separators -- and whatever a human added is what remains.
    """
    stem = os.path.splitext(os.path.basename(str(name or "")))[0]
    stem = re.sub(r"(?i)^dk\s*salaries", "", stem)
    stem = DATE_RE.sub("", stem)
    if date:
        stem = stem.replace(str(date), "")
    # Browser duplicate markers: 'DKSalaries (39)' carries no slate information.
    stem = re.sub(r"\(\s*\d+\s*\)", "", stem)
    slug = slate_slug(stem)
    return slug if slug != UNKNOWN_SLATE else ""


def parse_game_info(text):
    """('LV@HOU', kickoff datetime) from a DK `Game Info` cell, or (None, None)."""
    match = GAME_INFO_RE.match(str(text or ""))
    if not match:
        return None, None
    hour = int(match.group("hour")) % 12
    if match.group("meridiem").upper() == "PM":
        hour += 12
    kickoff = datetime(int(match.group("year")), int(match.group("month")),
                       int(match.group("day")), hour, int(match.group("minute")))
    return f"{match.group('away').upper()}@{match.group('home').upper()}", kickoff


def export_info(path, frame=None):
    """Describe one DK export: its games, kickoff span, weekday and player count.

    This is the `info` dict every labelling function below reads. Building it here rather
    than in each caller means the `Game Info` format is parsed in exactly one place.
    """
    import pandas as pd

    if frame is None:
        frame = pd.read_csv(path)
    games, kickoffs = [], []
    column = frame.reindex(columns=["Game Info"])["Game Info"]
    for value in column.dropna().unique():
        game, kickoff = parse_game_info(value)
        if game and game not in games:
            games.append(game)
            kickoffs.append(kickoff)
    minutes = sorted(k.hour * 60 + k.minute for k in kickoffs if k is not None)
    return {
        "path": path,
        "name": os.path.basename(str(path)),
        "games": games,
        "players": int(len(frame)),
        "first": minutes[0] if minutes else None,
        "last": minutes[-1] if minutes else None,
        "weekday": kickoffs[0].weekday() if kickoffs else None,
        "date": kickoffs[0].strftime("%Y-%m-%d") if kickoffs else None,
    }


def slug_from_contents(info):
    """Infer a slate label from what an export actually prices.

    Used only when the filename says nothing. DK's own naming is what people expect, so it
    is reconstructed from the games rather than invented: a single game is a showdown, a
    weeknight block is named for its night, and Sunday is split by kickoff window.

    **A Sunday fragment cannot be seen from here.** `early` and `main` share a first
    kickoff, so separating them needs the week's other exports -- see `label_slates`.
    """
    games = info.get("games") or []
    if len(games) == 1:
        return slate_slug(f"showdown-{games[0].replace('@', '')}")

    weekday = info.get("weekday")
    if weekday == 3:
        return "thursday"
    if weekday == 0:
        return "monday"
    if weekday == 4:
        return "friday"
    if weekday == 5:
        return "saturday"

    first = info.get("first")
    if first is None:
        return UNKNOWN_SLATE
    if first >= _PRIMETIME_START:
        return "primetime"
    if first >= _AFTERNOON_START:
        return "afternoon"
    return "main"


def _key(info):
    """Identity of one export within a week's set."""
    return info.get("path") or info.get("name")


def _first(info):
    value = info.get("first")
    return value if value is not None else -1


def label_slates(infos, date=None):
    """{path: slate label} for one week's exports, resolved against each other.

    `slug_from_contents` sees one file at a time and so cannot see a collision. A Sunday
    main slate and a Sunday early-only block **both** open at 1:00PM ET and both come back
    "main"; the only thing separating them is that one is a strict subset of the other. So a
    week's exports are labelled together, and where two land on the same label the smaller
    block -- fewer games -- is the fragment.

    Afternoon blocks are decided before that, because a block whose games all start after
    every game on the bigger slate has kicked off is a *later* slate, not a fragment of this
    one, however few games it has.

    A label taken from a filename is never overridden. Renaming an export is an explicit
    decision and this function only ever resolves guesses.
    """
    labels, guessed = {}, {}
    for info in infos:
        named = slug_from_filename(info.get("name"), date)
        labels[_key(info)] = named or slug_from_contents(info)
        guessed[_key(info)] = not named

    by_key = {_key(info): info for info in infos}
    groups = defaultdict(list)
    for key, label in labels.items():
        groups[label].append(key)

    def claim(preferred):
        """`preferred` if it is free, else preferred-2, -3, ... Resolving one collision by
        creating another would defeat the point of labelling the week as a set."""
        taken = set(labels.values())
        if preferred not in taken:
            return preferred
        suffix = 2
        while f"{preferred}-{suffix}" in taken:
            suffix += 1
        return f"{preferred}-{suffix}"

    for keys in groups.values():
        if len(keys) < 2:
            continue
        # Most games wins the plain label; ties broken by player count, then name, so the
        # same set of exports always labels the same way.
        ordered = sorted(keys, reverse=True,
                         key=lambda k: (len(by_key[k].get("games") or []),
                                        by_key[k].get("players") or 0, str(k)))
        biggest = by_key[ordered[0]]
        # The last kickoff on the big block: a smaller slate starting at or after it is not
        # running alongside, it is running after.
        cutoff = biggest.get("last")
        for key in ordered[1:]:
            if not guessed[key]:
                continue
            info = by_key[key]
            later = cutoff is not None and _first(info) >= cutoff
            labels[key] = claim("afternoon" if later else FRAGMENT_SLATE)
    return labels


def slate_for(info=None, requested=None, date=None, peers=None):
    """The slate label to file a week's output under.

    An explicit `--slate` wins, then the filename the export was saved as, then the
    contents. `peers` is the week's other exports: pass them and a main/early collision is
    resolved rather than silently filing two contests under one name.
    """
    if requested:
        # A --slate that names the chosen export is selecting it, not renaming it. The flag
        # does double duty -- it picks the file and it labels the output -- so
        # `--slate DKSalaries_2026-09-14_main.csv` would otherwise file the whole week under
        # "dksalaries-2026-09-14-main-csv": a label that no longer resolves back to any
        # export, which strands the board, the upload and everything read off them.
        name = os.path.basename(str((info or {}).get("name") or (info or {}).get("path") or ""))
        given = os.path.basename(str(requested)).lower()
        if not (name and given in (name.lower(), os.path.splitext(name)[0].lower())):
            return slate_slug(requested)
    if info is None:
        return UNKNOWN_SLATE
    if peers:
        labelled = label_slates(peers, date)
        if _key(info) in labelled:
            return labelled[_key(info)]
    return slug_from_filename(info.get("name"), date) or slug_from_contents(info)


def slate_dir(date, root=OUTPUT_ROOT):
    return os.path.join(root, str(date))


def output_path(kind, date, slate=None, root=OUTPUT_ROOT):
    """The canonical path for one output, before any versioning."""
    if kind not in KINDS:
        raise ValueError(f"unknown output kind '{kind}' (have {', '.join(sorted(KINDS))})")
    extension, _ = KINDS[kind]
    return os.path.join(slate_dir(date, root), f"{kind}_{slate_slug(slate)}{extension}")


def _version_of(path):
    """(base stem, version number, extension) for a path that may carry an .rN marker."""
    stem, extension = os.path.splitext(path)
    match = VERSION_RE.match(stem)
    if match:
        return match.group("stem"), int(match.group("n")), extension
    return stem, 1, extension


def next_version(path):
    """The next free `.rN` sibling of `path`, skipping every version already on disk."""
    base, _, extension = _version_of(path)
    version = 2
    while os.path.exists(f"{base}.r{version}{extension}"):
        version += 1
    return f"{base}.r{version}{extension}"


def existing_versions(path):
    """Every version of an output already written, oldest naming first."""
    base, _, extension = _version_of(path)
    found = [base + extension] if os.path.exists(base + extension) else []
    version = 2
    while os.path.exists(f"{base}.r{version}{extension}"):
        found.append(f"{base}.r{version}{extension}")
        version += 1
    return found


def resolve(kind, date, slate=None, root=OUTPUT_ROOT, overwrite=False):
    """Where to write, plus a note when the name had to change. Creates the slate folder.

    Returns `(path, note)`. `note` is None when the canonical name was free, and otherwise
    explains what the file was called instead -- callers print it, so a versioned write is
    never silent.
    """
    path = output_path(kind, date, slate, root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if overwrite or not os.path.exists(path):
        return path, None
    versioned = next_version(path)
    return versioned, (f"{os.path.basename(path)} exists; wrote "
                       f"{os.path.basename(versioned)} instead")


def latest(kind, date, slate=None, root=OUTPUT_ROOT):
    """The most recently written version of an output, or None. Counterpart to `resolve`.

    Chosen by modification time, not by the highest `.rN`. Those usually agree, but an
    `--overwrite` run rewrites the base name after `.r2` already exists -- and a reader that
    trusted the version number would then hand back the older file.
    """
    versions = existing_versions(output_path(kind, date, slate, root))
    if not versions:
        return None
    return max(versions, key=os.path.getmtime)


def find_slates(date, kind="slate", root=OUTPUT_ROOT):
    """Slate labels that have a given output written for a date, for error messages."""
    directory = slate_dir(date, root)
    if not os.path.isdir(directory):
        return []
    extension, _ = KINDS[kind]
    found = set()
    for name in sorted(os.listdir(directory)):
        stem, ext = os.path.splitext(name)
        if ext != extension or not stem.startswith(f"{kind}_"):
            continue
        label = stem[len(kind) + 1:]
        match = VERSION_RE.match(label)
        found.add(match.group("stem") if match else label)
    return sorted(found)
