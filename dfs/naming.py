"""One place that decides what every DFS file is called.

    dfs_boards/<date>/<kind>_<slate>.<ext>

Two things drove the layout. A night is the unit you actually work in and archive, so it
gets a folder -- the same shape `scouting_reports/<date>/` already uses. And every name
carries the **slate**, because a date alone does not identify a contest: 2026-07-28 had a
Main and a late export, and under date-only naming the second run of a night silently
overwrote the first night's board, lineups and upload with a different contest's players.

Nothing is ever overwritten by default. A rerun writes `board_main.r2.md` beside
`board_main.md`, so two runs can be compared and an accidental rebuild cannot destroy the
file you uploaded from. Pass overwrite=True to replace in place.
"""

import os
import re

OUTPUT_ROOT = "dfs_boards"

# kind -> (extension, one-line description). The description is what `dfs.files` prints, so
# a folder full of csvs explains itself without anyone having to open one.
KINDS = {
    "board":   (".md",  "human-readable board: tiers, stacks, writeups"),
    "slate":   (".csv", "every projected player, all columns"),
    "stacks":  (".csv", "team stacks ranked"),
    "pool":    (".csv", "editable pool: Lock / Exclude / Boost / Min% / Max%"),
    "lineups": (".csv", "optimizer output, one row per player per lineup"),
    "upload":  (".csv", "DraftKings upload file, one row per lineup"),
    "swap":    (".csv", "late-swap upload file"),
    "review":  (".md",  "post-game scoring vs actual results"),
}

# A slate that cannot be identified from a filename or its own contents. Deliberately not
# "main": a wrong-but-plausible label is harder to notice than an obviously unknown one.
UNKNOWN_SLATE = "unknown"

DATE_RE = re.compile(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})")
VERSION_RE = re.compile(r"^(?P<stem>.+?)\.r(?P<n>\d+)$")


def slate_slug(text):
    """Free text -> a short filename-safe slate label ('Main' -> 'main')."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(text or "").strip().lower()).strip("-")
    return slug or UNKNOWN_SLATE


def slug_from_filename(name, date=None):
    """Pull the slate label out of a DK export's filename.

    'DKSalaries_2026-07-28_Main.csv' -> 'main'. Everything structural is stripped -- the
    DKSalaries prefix, the date, separators -- and whatever a human added is what remains.
    """
    stem = os.path.splitext(os.path.basename(str(name or "")))[0]
    stem = re.sub(r"(?i)^dk\s*salaries", "", stem)
    stem = DATE_RE.sub("", stem)
    if date:
        stem = stem.replace(str(date), "")
    # Browser duplicate markers: 'DKSalaries (2)' carries no slate information.
    stem = re.sub(r"\(\s*\d+\s*\)", "", stem)
    slug = slate_slug(stem)
    return slug if slug != UNKNOWN_SLATE else ""


def slug_from_contents(info):
    """Infer a slate label from what an export actually prices.

    Used only when the filename says nothing. DK's own naming is what people expect --
    a single game is a Showdown, the big evening block is the main slate, and the smaller
    blocks either side of it are early and late -- so it is reconstructed from first pitch
    and game count rather than invented.
    """
    games = info.get("games") or []
    if len(games) == 1:
        return slate_slug(f"showdown-{games[0].replace('@', '')}")
    first = info.get("first")
    if first is None:
        return UNKNOWN_SLATE
    if first < 17 * 60:          # before 5:00PM ET
        return "early"
    if first >= 21 * 60:         # 9:00PM ET or later
        return "late"
    return "main"


def slate_for(info=None, requested=None, date=None):
    """The slate label to file a night's output under.

    An explicit --slate wins, then the filename the export was saved as, then the contents.
    """
    if requested:
        return slate_slug(requested)
    if info:
        return slug_from_filename(info.get("name"), date) or slug_from_contents(info)
    return UNKNOWN_SLATE


def night_dir(date, root=OUTPUT_ROOT):
    return os.path.join(root, str(date))


def output_path(kind, date, slate=None, root=OUTPUT_ROOT):
    """The canonical path for one output, before any versioning."""
    if kind not in KINDS:
        raise ValueError(f"unknown output kind '{kind}' (have {', '.join(sorted(KINDS))})")
    extension, _ = KINDS[kind]
    return os.path.join(night_dir(date, root), f"{kind}_{slate_slug(slate)}{extension}")


def _version_of(path):
    """(base stem, version number) for a path that may already carry an .rN marker."""
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
    """Where to write, plus a note when the name had to change. Creates the night folder.

    Returns (path, note). `note` is None when the canonical name was free, and otherwise
    explains what the file was called instead -- callers print it so a versioned write is
    never silent.
    """
    path = output_path(kind, date, slate, root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if overwrite or not os.path.exists(path):
        return path, None
    versioned = next_version(path)
    return versioned, (f"{os.path.basename(path)} exists; wrote "
                       f"{os.path.basename(versioned)} (--overwrite to replace)")


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
    directory = night_dir(date, root)
    if not os.path.isdir(directory):
        return []
    extension, _ = KINDS[kind]
    found = set()
    for name in sorted(os.listdir(directory)):
        stem, ext = os.path.splitext(name)
        if ext != extension or not stem.startswith(f"{kind}_"):
            continue
        label = stem[len(kind) + 1:]
        found.add(VERSION_RE.match(label).group("stem") if VERSION_RE.match(label) else label)
    return sorted(found)
