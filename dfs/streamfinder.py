"""Turn a night's DK upload into a Baseball-Reference StreamFinder priority file.

StreamFinder watches a live feed and cuts to whichever of your priority players is up next.
The list it wants is small and ordered, so the question it really asks is "who is your night
riding on" -- and after a set of lineups is built, that is not a matter of opinion. It is
exposure. A player in 60% of your entries deserves the feed ahead of one in 8%, whatever
either was projected for.

**Read off the upload, not the lineups.** `lineups_<slate>.csv` is everything the optimizer
produced; `upload_<slate>.csv` is what was actually submitted, which after a `--select` or a
hand edit is a different set. Exposure computed from the wrong one describes a night you did
not enter.

**Ids come from the slate, never from a name lookup.** The old standalone script resolved
players through `pybaseball.playerid_lookup` on a split first/last name, which is slow, needs
the network, and silently fails on accents, suffixes and anyone with three names. The DK id
is already in the upload file and the slate frame already carries the MLBAM id beside it, so
this is a join.

    python -m dfs.streamfinder --date 2026-08-17
    python -m dfs.streamfinder --date 2026-08-17 --top 20 --ignore 116 --ignore 113
"""

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime

from .naming import OUTPUT_ROOT, find_slates, latest
from .upload import SLOT_ORDER, _slot_start, parse_cell

# StreamFinder's own defaults, matching the shape of a file it has written itself. The
# numeric-looking fields are strings on purpose -- that is how the app writes them, and a
# real integer in `delay` is not read back the same way.
DEFAULTS = {
    "on_deck": "N",
    "include_CLI": "Y",
    "delay": "5000",
}
DEFAULT_OUTPUT = "StreamFinder.txt"

# Roster columns that hold pitchers. Read off SLOT_ORDER rather than hardcoded, so a
# scoring-format change moves this with it.
PITCHER_SLOTS = {index for index, slot in enumerate(SLOT_ORDER) if slot == "P"}


def exposure_from_upload(path):
    """[(dk id, count, is_pitcher, display name)] from a DK upload file, most-used first.

    Handles both DK layouts. The bulk template's rows are lineups and the entries export's
    rows are existing entries, but in both the ten roster columns sit in a run that
    `_slot_start` finds by name, so neither the leading metadata columns nor a trailing
    instructions block can be mistaken for players.
    """
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"{path} is empty")

    start = None
    header_index = 0
    for index, row in enumerate(rows[:5]):
        start = _slot_start(row)
        if start is not None:
            header_index = index
            break
    if start is None:
        raise ValueError(
            f"{os.path.basename(path)} has no {'/'.join(SLOT_ORDER)} column run — "
            "is it a DK upload file?")

    counts, roles, names = Counter(), {}, {}
    for row in rows[header_index + 1:]:
        cells = row[start:start + len(SLOT_ORDER)]
        if len(cells) < len(SLOT_ORDER) or not any(str(c).strip() for c in cells):
            continue                       # blank row, or DK's trailing instructions block
        for offset, cell in enumerate(cells):
            dk_id, display = parse_cell(cell)
            if not dk_id:
                continue
            counts[dk_id] += 1
            # First appearance decides the role. A DK id belongs to exactly one roster
            # position -- a two-way player is listed twice under different ids -- so a
            # disagreement here means the file's columns are not what the header says.
            roles.setdefault(dk_id, offset in PITCHER_SLOTS)
            if display and dk_id not in names:
                names[dk_id] = display

    return [(dk_id, count, roles[dk_id], names.get(dk_id, ""))
            for dk_id, count in counts.most_common()]


def _slate_index(players):
    """{dk id: {"mlbam", "name", "type", "proj"}} from a slate frame."""
    index = {}
    if players is None or not len(players):
        return index
    for _, row in players.iterrows():
        dk_id = str(row.get("DK ID") or "").strip()
        if dk_id.endswith(".0"):
            dk_id = dk_id[:-2]
        mlbam = row.get("MLBAM")
        if not dk_id or mlbam is None or mlbam != mlbam:      # NaN check
            continue
        index[dk_id] = {
            "mlbam": str(int(mlbam)),
            "name": row.get("Name", ""),
            "type": row.get("Type", ""),
            "proj": row.get("Proj"),
        }
    return index


def build_priority(exposure, index, top=None):
    """[(entry dict, detail dict)] ready for the JSON, plus what could not be resolved.

    Ordering is exposure first and projection second. The tiebreak matters more than it
    looks: on a 20-lineup set half the roster lands on the same count, and leaving those
    ties to dictionary order would put the feed on whichever of them the optimizer happened
    to write first.
    """
    resolved, unresolved = [], []
    for dk_id, count, is_pitcher, display in exposure:
        found = index.get(dk_id)
        if not found:
            unresolved.append((dk_id, display or "unknown", count))
            continue
        resolved.append({
            "mlbam": found["mlbam"],
            # Role from the roster column it was played in, not from the slate's Type.
            # They agree in every ordinary case; when they do not, the column is what the
            # lineup actually did.
            "type": "pit" if is_pitcher else "bat",
            "name": found["name"] or display,
            "count": count,
            "proj": found["proj"],
        })

    def sort_key(item):
        proj = item["proj"]
        return (-item["count"], -(proj if isinstance(proj, (int, float)) else 0.0))

    resolved.sort(key=sort_key)
    if top:
        resolved = resolved[:top]
    entries = [{"type": r["type"], "data": r["mlbam"], "immediate": "", "priority": n}
               for n, r in enumerate(resolved, start=1)]
    return entries, resolved, unresolved


def build_document(entries, ignore=None, **overrides):
    """The full StreamFinder document. Key order matches a file the app has written."""
    document = dict(DEFAULTS)
    document.update({k: v for k, v in overrides.items() if v is not None})
    document["ignore"] = [str(t) for t in (ignore or [])]
    document["priority"] = entries
    return document


def generate(date, slate=None, root=OUTPUT_ROOT, upload_path=None, top=30, ignore=None,
             players=None, **overrides):
    """Build the document for one night. Returns (document, resolved, unresolved)."""
    if upload_path is None:
        # A date alone does not identify a contest -- a night can carry a main and a late
        # slate -- so an unspecified label is resolved against what is actually on disk
        # rather than defaulting to one. `naming.output_path` would otherwise quietly build
        # a path for the "unknown" slate and report the file as missing.
        if slate is None:
            found = find_slates(date, kind="upload", root=root)
            if not found:
                raise FileNotFoundError(
                    f"no upload file for {date} under {root}/ - "
                    "run the optimizer with --upload first")
            if len(found) > 1:
                raise ValueError(
                    f"{date} has uploads for {len(found)} slates ({', '.join(found)}); "
                    "pass --slate to choose one")
            slate = found[0]
        upload_path = latest("upload", date, slate, root)
    if not upload_path or not os.path.exists(upload_path):
        raise FileNotFoundError(
            f"no upload file for {date} slate '{slate}' under {root}/ - "
            "run the optimizer with --upload first")

    if players is None:
        from .slate import build_slate
        players, _stacks, _meta = build_slate(date, slate=slate)

    exposure = exposure_from_upload(upload_path)
    entries, resolved, unresolved = build_priority(exposure, _slate_index(players), top)
    return build_document(entries, ignore, **overrides), resolved, unresolved, upload_path


def main():
    parser = argparse.ArgumentParser(
        description="Write a StreamFinder priority file ordered by lineup exposure.")
    parser.add_argument("--date", default=datetime.today().strftime("%Y-%m-%d"))
    parser.add_argument("--slate", help="slate label, e.g. main. Defaults to the only one.")
    parser.add_argument("--upload", help="explicit upload csv, bypassing the date lookup")
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--top", type=int, default=30,
                        help="how many players to prioritise (default 30)")
    parser.add_argument("--ignore", action="append", default=[], metavar="TEAM_ID",
                        help="MLB team id to ignore globally; repeatable")
    parser.add_argument("--on-deck", choices=["Y", "N"])
    parser.add_argument("--include-cli", choices=["Y", "N"])
    parser.add_argument("--delay", help="milliseconds, e.g. 5000")
    parser.add_argument("--root", default=OUTPUT_ROOT)
    args = parser.parse_args()

    overrides = {"on_deck": args.on_deck, "include_CLI": args.include_cli,
                 "delay": args.delay}
    document, resolved, unresolved, upload_path = generate(
        args.date, args.slate, args.root, args.upload, args.top, args.ignore, **overrides)

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=4)

    total = sum(r["count"] for r in resolved)
    lineups = max((r["count"] for r in resolved), default=0)
    print(f"{os.path.basename(upload_path)} -> {args.out}")
    print(f"  {len(document['priority'])} players prioritised"
          + (f", ignoring team(s) {', '.join(document['ignore'])}" if document["ignore"] else ""))
    if resolved:
        print(f"  {'#':>3}  {'player':<24} {'':3} {'used':>5}  share")
        for entry, detail in zip(document["priority"], resolved):
            share = detail["count"] / lineups * 100 if lineups else 0
            print(f"  {entry['priority']:>3}  {detail['name']:<24} "
                  f"{detail['type']:<3} {detail['count']:>5}  {share:5.1f}%")
    if unresolved:
        print(f"\n  {len(unresolved)} not matched to an MLBAM id "
              "(not on this slate's board?):")
        for dk_id, display, count in unresolved[:10]:
            print(f"    {display or dk_id} (DK {dk_id}, {count} lineups)")
    if total and lineups:
        print(f"\n  read from {lineups} lineups")


if __name__ == "__main__":
    main()
