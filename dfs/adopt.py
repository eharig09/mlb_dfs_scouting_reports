"""Take in DraftKings downloads that were never named properly, and file them.

A browser writes `DKSalaries (2).csv`; DK's own upload template also downloads as
`DKSalaries.csv` whatever it actually contains. Neither name says which date, which
contest, or even which kind of file it is -- and this repo already has both problems on
disk at once. The old behaviour was to either miss the file entirely or, worse, half-match
it and price a board off the wrong contest.

Detection is by contents, never by name: a salary export is recognised by its columns, an
upload template by its `P,P,C,1B,...` header row. Once a file is identified it is renamed
into the convention and used, so the next run finds it the ordinary way. Renaming rather
than copying is deliberate -- two files with the same contents under different names is the
ambiguity this is meant to remove.
"""

import csv
import os
import re
import shutil

from .naming import slate_slug, slug_from_contents, slug_from_filename
from .salaries import canon_team, describe_slate, is_salary_file, list_salary_files

# Searched automatically before a run. Project folders only: a file inside the repo is one
# you meant to use here, so renaming it is housekeeping.
SEARCH_DIRS = ["dfs_daily_files", "dk_lineups", "."]

# Searched only when asked for explicitly (`--downloads`). This is where DK actually puts
# things, but a routine board build has no business reaching into it -- and files found
# outside the project are copied rather than moved, so nothing vanishes from under you.
DOWNLOADS_DIR = os.path.join(os.path.expanduser("~"), "Downloads")

SALARY_HOME = "dfs_daily_files"
TEMPLATE_HOME = "dk_lineups"

# The ten roster columns a DK upload template leads with.
SLOT_HEADER = ("P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF")

# Written by us, not downloaded -- never adopt our own output back into an input.
GENERATED_PREFIXES = ("DKUpload_", "DKLateSwap_", "upload_", "swap_", "lineups_", "pool_")


class AdoptionError(Exception):
    """A stray file was found but could not be filed safely."""


def is_upload_template(path):
    """True when a CSV is a DK upload template, whatever it happens to be named."""
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            header = next(csv.reader(handle), [])
    except OSError:
        return False
    cells = [str(c).strip() for c in header]
    for i in range(len(cells) - len(SLOT_HEADER) + 1):
        if tuple(cells[i:i + len(SLOT_HEADER)]) == SLOT_HEADER:
            return True
    return False


def template_dates(path):
    """Slate dates carried by a template's embedded player list."""
    from .upload import template_dates as _dates
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            return _dates([list(row) for row in csv.reader(handle)])
    except OSError:
        return set()


def template_kind(path):
    """'entries' for a filled contest export, 'bulk' for the blank 500-lineup template.

    They are not interchangeable -- an entries export overwrites lineups you have already
    entered -- so they must never be filed under one name.
    """
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            header = next(csv.reader(handle), [])
    except OSError:
        return "bulk"
    return "entries" if str((header or [""])[0]).strip() == "Entry ID" else "bulk"


def _candidates(search_dirs=None):
    for directory in (search_dirs or SEARCH_DIRS):
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            if not name.lower().endswith(".csv"):
                continue
            if name.startswith(GENERATED_PREFIXES):
                continue
            yield os.path.join(directory, name)


def _inside_project(path):
    """True when a file already lives under the repo, and so can be moved rather than copied."""
    root = os.path.normcase(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    return os.path.normcase(os.path.abspath(path)).startswith(root + os.sep)


def _fingerprint(path):
    """Size plus first kilobyte -- enough to spot the same download sitting in two folders."""
    try:
        with open(path, "rb") as handle:
            return os.path.getsize(path), handle.read(1024)
    except OSError:
        return None


def _is_conventional(name, date):
    """True when a filename already states its kind, date and slate."""
    return bool(re.match(rf"(?i)^DKSalaries_{re.escape(str(date))}_[a-z0-9-]+\.csv$", name))


def find_strays(date, search_dirs=None):
    """Every unfiled DK download that belongs to `date`, identified by its contents.

    Returns a list of dicts with `path`, `kind` ('salary' or 'template'), the proposed new
    name, and enough description to print before anything is moved.
    """
    seen, prints, strays = set(), set(), []
    for path in _candidates(search_dirs):
        real = os.path.normcase(os.path.abspath(path))
        if real in seen:
            continue
        seen.add(real)
        # The same download commonly sits in Downloads and in the project at once. Filing
        # it twice would collide on the target name and report a spurious conflict.
        mark = _fingerprint(path)
        if mark is not None:
            if mark in prints:
                continue
            prints.add(mark)
        name = os.path.basename(path)

        if is_salary_file(path):
            info = describe_slate(path)
            if info.get("date") != str(date):
                continue
            if _is_conventional(name, date):
                continue        # already filed correctly
            label = slug_from_filename(name, date) or slug_from_contents(info)
            strays.append({
                "path": path,
                "kind": "salary",
                "target": os.path.join(SALARY_HOME, f"DKSalaries_{date}_{label}.csv"),
                "slate": label,
                "why": (f"{len(info['games'])} games, {info['players']} players"
                        if info.get("games") else f"{info['players']} players"),
            })
            continue

        if is_upload_template(path):
            dates = template_dates(path)
            # A template with no dated player list cannot be placed; leaving it alone is
            # safer than guessing, because uploading to the wrong slate is unrecoverable.
            if not dates or str(date) not in dates:
                continue
            layout = template_kind(path)
            label = template_slate(path, date)
            target_name = f"DKTemplate_{date}_{label}_{layout}.csv"
            if name == target_name:
                continue
            strays.append({
                "path": path,
                "kind": "template",
                "target": os.path.join(TEMPLATE_HOME, target_name),
                "slate": label,
                "why": f"DK upload template ({layout})",
            })
    return strays


def template_slate(path, date):
    """Which slate an upload template belongs to, from the clubs it prices.

    Matched against the DK salary exports for the date, which already carry slate labels --
    so a template and the board built from the same contest end up filed under the same
    name. A date alone cannot separate an early template from a main one, and filing both
    as `DKTemplate_<date>` means the second collides with the first and the wrong one is
    silently kept.
    """
    from .naming import UNKNOWN_SLATE, slate_for
    from .upload import template_teams_for

    priced = template_teams_for(path)
    if not priced:
        return UNKNOWN_SLATE

    best, best_overlap = None, 0
    for info in list_salary_files(date):
        teams = set()
        for game in info.get("games") or []:
            teams |= {canon_team(part) for part in str(game).split("@")}
        teams = {t for t in teams if t}
        if not teams:
            continue
        # Exact agreement is the answer; otherwise the closest export wins, so a template
        # downloaded before a postponement still lands under the right slate.
        if teams == priced:
            return slate_for(info, date=date)
        overlap = len(teams & priced)
        if overlap > best_overlap:
            best, best_overlap = info, overlap
    return slate_for(best, date=date) if best else UNKNOWN_SLATE


def adopt(stray, dry_run=False):
    """File one stray under the convention. Returns the path it now lives at.

    Inside the repo the file is moved -- leaving the old name behind would recreate the
    ambiguity. From anywhere else it is copied, because emptying someone's Downloads folder
    is not this tool's business.
    """
    target = stray["target"]
    source = stray["path"]
    if os.path.normcase(os.path.abspath(source)) == os.path.normcase(os.path.abspath(target)):
        return target
    if os.path.exists(target):
        raise AdoptionError(
            f"cannot file {os.path.basename(source)} as {os.path.basename(target)}: "
            f"that name is taken. Rename one of them by hand."
        )
    if dry_run:
        return target
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    if _inside_project(source):
        shutil.move(source, target)
    else:
        shutil.copy2(source, target)
    return target


def adopt_for_date(date, search_dirs=None, kinds=("salary", "template"), dry_run=False):
    """Find and file every stray belonging to `date`. Returns (adopted, problems).

    Called before a run resolves its inputs, so a download that was never renamed still
    gets used -- and gets a proper name in the process, which is what stops the same file
    from being re-adopted every night.
    """
    adopted, problems = [], []
    for stray in find_strays(date, search_dirs):
        if stray["kind"] not in kinds:
            continue
        try:
            target = adopt(stray, dry_run=dry_run)
        except AdoptionError as error:
            problems.append(str(error))
            continue
        adopted.append({**stray, "target": target, "copied": not _inside_project(stray["path"])})
    return adopted, problems


def describe(adopted):
    """Lines describing what was filed, for printing before a run continues."""
    lines = []
    for item in adopted:
        verb = "copied" if item.get("copied") else "renamed"
        lines.append(f"    {item['path']}  ->  {item['target']}  ({verb})")
        lines.append(f"        {item['why']}"
                     + (f", filed as slate '{item['slate']}'" if item.get("slate") else ""))
    return lines


def main():
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(
        description="Find DraftKings downloads that were never named properly and file them.")
    parser.add_argument("--date", default=datetime.today().strftime("%Y-%m-%d"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be renamed without moving anything.")
    parser.add_argument("--downloads", action="store_true",
                        help=f"Also search {DOWNLOADS_DIR}. Files there are copied, not moved.")
    parser.add_argument("--dir", action="append", dest="dirs",
                        help="Extra directory to search. Repeatable.")
    args = parser.parse_args()

    search = list(SEARCH_DIRS)
    if args.downloads:
        search.append(DOWNLOADS_DIR)
    if args.dirs:
        search.extend(args.dirs)
    search = search if search != SEARCH_DIRS else None
    adopted, problems = adopt_for_date(args.date, search_dirs=search, dry_run=args.dry_run)

    if not adopted and not problems:
        print(f"No unfiled DraftKings downloads for {args.date}.")
        return
    if adopted:
        print(f"{'Would file' if args.dry_run else 'Filed'} {len(adopted)} file(s) for {args.date}:")
        for line in describe(adopted):
            print(line)
    for problem in problems:
        print(f"[!] {problem}")


if __name__ == "__main__":
    main()
