"""What is in a night's folder, and what each file is for.

`python -m dfs.files --date 2026-07-28`

The naming convention is only useful if it is legible, and a folder holding four csvs
called slate / stacks / lineups / pool is legible only once you know which is which. This
prints that, plus the versions -- seeing `board_main.md` next to `board_main.r2.md` is how
you notice you have been reading an old board.
"""

import argparse
import os
from datetime import datetime

from .naming import KINDS, OUTPUT_ROOT, VERSION_RE, night_dir


# Pipeline order, not alphabetical: this is the order the files are produced in, so a gap in
# the middle is what a missing step looks like. Anything in KINDS but not named here is
# appended rather than dropped -- a hardcoded tuple silently hid the exposure report from
# this listing the day it was added, which is the one place you go to find out what exists.
_ORDER = ("board", "slate", "stacks", "pool", "lineups", "exposure", "upload", "swap", "review")
PIPELINE_ORDER = _ORDER + tuple(k for k in KINDS if k not in _ORDER)


def _describe(path):
    stat = os.stat(path)
    size = stat.st_size
    for unit in ("B", "KB", "MB"):
        if size < 1024 or unit == "MB":
            break
        size /= 1024.0
    written = datetime.fromtimestamp(stat.st_mtime).strftime("%H:%M")
    return f"{size:,.0f} {unit}", written


def scan(date, root=OUTPUT_ROOT):
    """{slate: {kind: [(path, version)]}} for one night, plus anything unrecognised."""
    directory = night_dir(date, root)
    if not os.path.isdir(directory):
        return {}, []

    found, other = {}, []
    for name in sorted(os.listdir(directory)):
        stem, extension = os.path.splitext(name)
        kind, _, label = stem.partition("_")
        if kind not in KINDS or not label or KINDS[kind][0] != extension:
            other.append(name)
            continue
        match = VERSION_RE.match(label)
        version = int(match.group("n")) if match else 1
        slate = match.group("stem") if match else label
        found.setdefault(slate, {}).setdefault(kind, []).append(
            (os.path.join(directory, name), version))
    return found, other


def main():
    parser = argparse.ArgumentParser(description="List a night's DFS files and what each one is.")
    parser.add_argument("--date", default=datetime.today().strftime("%Y-%m-%d"))
    parser.add_argument("--root", default=OUTPUT_ROOT)
    args = parser.parse_args()

    found, other = scan(args.date, args.root)
    directory = night_dir(args.date, args.root)
    if not found and not other:
        print(f"Nothing written for {args.date} yet ({directory}/ is empty or missing).")
        return

    print(f"{directory}/")
    for slate in sorted(found):
        print(f"\n  slate: {slate}")
        for kind in PIPELINE_ORDER:
            entries = found[slate].get(kind)
            if not entries:
                continue
            _, description = KINDS[kind]
            # By write time, matching naming.latest -- after an --overwrite the base name is
            # newer than the .r2 beside it, and showing .r2 as current would mislead.
            newest = max(entries, key=lambda e: os.path.getmtime(e[0]))
            size, written = _describe(newest[0])
            stale = f"   (+{len(entries) - 1} older)" if len(entries) > 1 else ""
            print(f"    {os.path.basename(newest[0]):<28} {size:>9}  {written}{stale}")
            print(f"      {description}")

    if other:
        print(f"\n  not part of the convention ({len(other)}):")
        for name in other:
            print(f"    {name}")


if __name__ == "__main__":
    main()
