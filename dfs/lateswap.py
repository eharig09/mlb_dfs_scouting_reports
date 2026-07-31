"""Re-read a filled DK upload file and refill only the slots that have gone bad.

Between building a lineup and first pitch the slate moves: a game is postponed, a starter
is scratched, a hitter is out of the confirmed card. DK's own late swap lets you replace
those players right up until their game begins, but doing it by hand across twenty entries
is slow and error-prone at exactly the moment there is no time.

This reads a file that has already been filled -- `DKUpload_<date>.csv` or a DK entry
export -- works out which of its ten slots are no longer playable, and re-solves each
lineup with everything else pinned in place. The base you built stays exactly as it was;
only the broken slots move.

Two rules make this different from a normal optimizer run, and both come from DK:

  * A replacement has to be in a game that has not started. Anyone else is unrosterable
    however good the projection, so started games are excluded from the fill pool.
  * A kept player has to stay in the same roster column. DK compares an edited entry slot
    by slot, so a kept 2B who drifts into the 3B column reads as two edited slots -- one of
    them very possibly locked. Keepers are pinned, not merely locked.

One caveat worth knowing. The slate is rebuilt from the cached game reports, which hold
only the nine hitters on each lineup card -- there are no bench players in the pool. That
is what makes a scratch detectable at all: the replaced hitter simply stops existing. But
it also means a scratch is only visible once the cache has seen the new card. Refresh it
first (`--refresh-lineups`) if you are swapping on a scratch rather than a postponement.

Run it as often as you like: with nothing broken it reports "nothing to swap" and writes
no file.
"""

import argparse
import os
from datetime import datetime

from .optimizer import OptimizerError, optimize
from .salaries import canon_team, normalize_name
from .schedule import describe_postponed
from .slate import build_slate, started_players
from .upload import (
    SLOT_ORDER, SWAP_PREFIX, TEMPLATE_DIR, UploadError, _pad,
    id_index, parse_cell, read_template, resolve_ids,
)

# Why a slot needed refilling, worst first -- a postponed game is a fact, a missing name is
# a guess about a file, and the two deserve different levels of trust in the output.
REASONS = {
    "empty": "slot was empty",
    "postponed": "game called off",
    "elsewhere": "already pitching the doubleheader opener",
    "unknown": "off the slate (postponed, scratched or unpriced)",
}


class SwapError(Exception):
    pass


def find_filled(date, path=None, directory=TEMPLATE_DIR, slate=None):
    """Locate the already-filled upload file to re-read.

    Prefers the newest `upload_<slate>.csv` in the night folder, then a swap file already
    written for it -- a second swap of the evening must start from the first one's output,
    not from the original upload -- and finally the pre-dated-folder name.
    """
    from .naming import OUTPUT_ROOT, find_slates, latest

    if path:
        if not os.path.exists(path):
            raise SwapError(f"no upload file at {path}")
        return path

    labels = [slate] if slate else find_slates(date, "upload", OUTPUT_ROOT)
    if len(labels) > 1:
        raise SwapError(f"{date} has uploads for {len(labels)} slates "
                        f"({', '.join(labels)}) — pass --slate to choose")
    for label in labels:
        for kind in ("swap", "upload"):
            found = latest(kind, date, label, OUTPUT_ROOT)
            if found:
                return found

    legacy = os.path.join(directory, f"DKUpload_{date}.csv")
    if os.path.exists(legacy):
        return legacy
    raise SwapError(
        f"no filled upload file for {date} — build one first with "
        f"`python -m dfs.upload --date {date}`, or pass --file"
    )


def read_filled(rows, kind, slot_start):
    """[(row index, [ten cells], [ten ids])] for every filled lineup row in the file.

    The raw cells are carried alongside the ids because a kept slot is written back
    verbatim -- reformatting a slot DK did not ask us to touch is exactly the kind of
    difference that gets an edit rejected.
    """
    width = slot_start + len(SLOT_ORDER)
    found = []
    for i, row in enumerate(rows):
        if i == 0:
            continue
        padded = _pad(row, width)
        cells = [str(cell).strip() for cell in padded[slot_start:width]]
        if kind == "entries" and not str(padded[0]).strip():
            continue            # a player-list row, or an entry with no lineup yet
        if not any(cells):
            continue            # an untouched spare row, not a lineup
        found.append((i, cells, [parse_cell(cell)[0] for cell in cells]))
    return found


def cell_style(lineups):
    """True when the file writes roster cells as 'Name (ID)' rather than a bare id."""
    for _, cells, _ in lineups:
        for cell in cells:
            if cell and not cell.replace(".0", "").isdigit():
                return True
    return False


def _pool_lookups(pool):
    """(by id, by name+team, by name) indexes into the current playable pool."""
    by_id, by_pair, seen = {}, {}, {}
    for _, row in pool.iterrows():
        key = normalize_name(row.get("Name"))
        team = canon_team(row.get("Team"))
        stored = str(row.get("DK ID") or "").strip()
        if stored.endswith(".0"):
            stored = stored[:-2]
        if stored:
            by_id[stored] = row
        by_pair[(key, team)] = row
        seen.setdefault(key, []).append(row)
    by_name = {key: rows[0] for key, rows in seen.items() if len(rows) == 1}
    return by_id, by_pair, by_name


def classify(cells, ids, lookups, id_names, postponed=(), dead=()):
    """Per slot, the player to keep or the reason it needs refilling.

    Resolution runs id first, then the player list's name for that id, then the name
    written into the cell itself. The fallbacks matter: re-downloading a DK file after a
    postponement can renumber the draft group, and an id that no longer resolves is then a
    numbering change rather than a player who has gone away.
    """
    by_id, by_pair, by_name = lookups
    slots = []
    for slot, cell, player_id in zip(SLOT_ORDER, cells, ids):
        if not player_id and not cell:
            slots.append((slot, None, "empty"))
            continue
        row = by_id.get(player_id)
        named = id_names.get(player_id)
        if row is None and named:
            row = by_pair.get((named["key"], named["team"])) or by_name.get(named["key"])
        if row is None:
            # An entry export writes "Bailey Ober (38384954)", so even a file whose player
            # list is for a different draft group still names who was in the slot.
            in_cell = normalize_name(parse_cell(cell)[1])
            if in_cell:
                row = by_name.get(in_cell)
        if row is None:
            # Naming the postponement when the file knows the player's team turns a vague
            # "not on the slate" into the reason you are standing here at 6:55.
            reason = "postponed" if named and named["team"] in postponed else "unknown"
            slots.append((slot, None, reason))
            continue
        if normalize_name(row.get("Name")) in dead:
            slots.append((slot, None, "elsewhere"))
            continue
        slots.append((slot, row, None))
    return slots


def rebuild_row(slots, lineup, cells, ids, index, by_id, number, named=True):
    """The ten roster cells for the edited row, column by column.

    Built positionally rather than by re-deriving DK's slot order, because two of the ten
    columns are P and three are OF: a rebuild that only knows "this player is an OF" is
    free to shuffle the three OF columns among themselves, which reads to DK as editing
    three slots -- including any that are locked. A kept slot is copied across verbatim,
    so untouched columns are untouched byte for byte.

    `named` writes replacements as "Name (ID)" to match an entry export; a bulk template
    takes the bare id.
    """
    kept = {normalize_name(row["Name"]) for _, row, _ in slots if row is not None}
    added = {}
    for record in lineup["players"].to_dict("records"):
        if normalize_name(record["Name"]) in kept:
            continue
        added.setdefault(str(record.get("Roster") or "").strip().upper(), []).append(record)

    out = []
    for (slot, row, _), cell, original in zip(slots, cells, ids):
        if row is not None:
            # The cell is copied as-is unless its id is the thing that went stale, in which
            # case the player was matched by name and needs today's number.
            if by_id.get(original) is not None:
                out.append(cell)
            else:
                fresh = resolve_ids([row], index, number)[0]
                out.append(f"{row['Name']} ({fresh})" if named else fresh)
            continue
        bucket = added.get(slot)
        if not bucket:
            raise OptimizerError(f"the refill produced no {slot} for the empty {slot} slot")
        record = bucket.pop(0)
        fresh = resolve_ids([record], index, number)[0]
        out.append(f"{record['Name']} ({fresh})" if named else fresh)

    leftover = [record["Name"] for bucket in added.values() for record in bucket]
    if leftover:
        raise OptimizerError(f"refill produced unplaceable player(s): {', '.join(leftover)}")
    written = [parse_cell(value)[0] for value in out]
    if len(set(written)) != len(written):
        raise OptimizerError("refill would duplicate a player already in the lineup")
    return out


def swap_lineup(players, slots, locked, objective="ceiling", **kwargs):
    """Re-solve one lineup, pinning every keeper to the slot it already holds."""
    pins, keepers = {}, set()
    for slot, row, _ in slots:
        if row is None:
            continue
        pins[str(row["Name"])] = slot
        keepers.add(normalize_name(row["Name"]))

    # Everyone in a game already under way is unrosterable as a replacement. Keepers are
    # spared: they are staying put, and DK does not mind a locked player it is not asked
    # to change. Excludes beat locks in build_pool, so this filter has to be built by
    # subtraction rather than applied afterwards.
    excludes = [name for name in players[locked]["Name"]
                if normalize_name(name) not in keepers]

    try:
        lineups, _, missing = optimize(
            players, n_lineups=1, objective=objective, pins=pins, excludes=excludes,
            randomness=0, max_overlap=len(SLOT_ORDER), **kwargs,
        )
    except OptimizerError as error:
        # Late in the evening almost every failure here is the same one, and saying so
        # beats leaving you to work out why a 140-player slate has eight playable names.
        raise OptimizerError(f"{error} ({len(excludes)} already playing)") from error
    if not lineups:
        raise OptimizerError("no legal refill exists — every candidate is locked or priced out")
    if missing.get("pins"):
        raise OptimizerError(f"kept player(s) vanished from the pool: {', '.join(missing['pins'])}")
    return lineups[0]


def swap_file(date, path=None, objective="ceiling", slate=None, **kwargs):
    """Refill the broken slots of every lineup in a filled upload file.

    Returns (output rows, report). Output rows are None when nothing needed swapping.
    """
    filled_path = find_filled(date, path, slate=slate)
    rows, kind, slot_start, index = read_template(filled_path)
    lineups = read_filled(rows, kind, slot_start)
    if not lineups:
        raise SwapError(f"{filled_path} has no filled lineups to re-read")

    # The slate has to be named here too, not just when locating the file: on a night with
    # two exports an unqualified build prices nothing, and the swap fails claiming there
    # are no salaries when the real problem is that it did not know which contest this is.
    players, _, meta = build_slate(date, slate=slate)
    if players.empty:
        raise SwapError(f"no slate data for {date} — nothing to refill from")
    if not meta.get("has_salary"):
        detail = (f" — {date} has several DK exports; pass --slate"
                  if meta.get("ambiguous_slates") else "")
        raise SwapError(f"no salaries matched; the optimizer needs prices to refill a slot{detail}")

    # Per game, not per team: a doubleheader nightcap is fully swappable while the opener
    # is in progress, and a team-level check would freeze both.
    locked, elsewhere, started_error = started_players(players, meta, date)
    started = sorted(set(players[locked]["Game"].dropna())) if locked.any() else []
    # A starter already throwing the opener is the one player DK will let you move who is
    # guaranteed to score nothing, so the slot counts as broken rather than kept.
    dead = {normalize_name(name) for name in players[elsewhere]["Name"]}
    lookups = _pool_lookups(players)
    id_names = id_index(rows)
    postponed = set(meta.get("postponed") or {})

    report = {
        "path": filled_path, "kind": kind, "postponed": meta.get("postponed") or {},
        "started": started, "started_error": started_error,
        "changed": [], "unchanged": [], "failed": [], "swaps": [],
    }

    width = slot_start + len(SLOT_ORDER)
    out = [list(row) for row in rows]
    touched = False

    named = cell_style(lineups)
    for row_index, cells, ids in lineups:
        slots = classify(cells, ids, lookups, id_names, postponed, dead)
        broken = [(slot, reason) for slot, row, reason in slots if row is None]
        if not broken:
            report["unchanged"].append(row_index)
            continue
        try:
            lineup = swap_lineup(players, slots, locked | elsewhere,
                                 objective=objective, **kwargs)
            new_ids = rebuild_row(slots, lineup, cells, ids, index, lookups[0],
                                  row_index, named=named)
        except (OptimizerError, UploadError) as error:
            # The original row is left exactly as it was. A lineup that cannot be refilled
            # is still the lineup you entered, and half-swapping it would be worse.
            report["failed"].append((row_index, [s for s, _ in broken], str(error)))
            continue

        kept = {normalize_name(row["Name"]) for _, row, _ in slots if row is not None}
        added = [record for record in lineup["players"].to_dict("records")
                 if normalize_name(record["Name"]) not in kept]
        report["swaps"].append((row_index, broken, added))
        report["changed"].append(row_index)
        padded = _pad(out[row_index], width)
        padded[slot_start:width] = new_ids
        out[row_index] = padded
        touched = True

    return (out if touched else None), report


def format_report(report):
    lines = [f"re-read {report['path']} ({report['kind']})"]
    if report["postponed"]:
        lines.append(f"  postponed: {describe_postponed(report['postponed'])}")
    if report["started"]:
        lines.append(f"  locked (game started): {', '.join(report['started'])}")
    if report["started_error"]:
        lines.append(f"  [!] could not check start times: {report['started_error']}. "
                     f"A started game would not be excluded — verify before uploading.")

    for row_index, broken, added in report["swaps"]:
        detail = ", ".join(f"{slot} ({REASONS.get(reason, reason)})" for slot, reason in broken)
        names = ", ".join(f"{p['Name']} [{p.get('Roster')}]" for p in added)
        lines.append(f"  row {row_index + 1}: {detail} -> {names or '(none)'}")
    for row_index, broken, error in report["failed"]:
        lines.append(f"  [!] row {row_index + 1}: could not refill {', '.join(broken)} — {error}")

    lines.append(f"{len(report['changed'])} lineup(s) swapped, "
                 f"{len(report['unchanged'])} untouched, {len(report['failed'])} failed")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Refill the broken slots of an already-filled DK upload file.")
    parser.add_argument("--date", default=datetime.today().strftime("%Y-%m-%d"))
    parser.add_argument("--file", help="The filled upload file. Default: the newest "
                                       "upload_<slate>.csv (or swap, if one exists) in the "
                                       "night folder.")
    parser.add_argument("--slate", help="Which slate to swap, when a night has more than one.")
    parser.add_argument("--out", help="Where to write. Default: "
                                      "dfs_boards/<date>/swap_<slate>.csv.")
    parser.add_argument("--in-place", action="store_true",
                        help="Overwrite the file that was read instead of writing a copy.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing swap file instead of writing a new .rN version.")
    parser.add_argument("--objective", choices=["ceiling", "proj", "floor"], default="ceiling",
                        help="What to maximize when picking replacements.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing anything.")
    args = parser.parse_args()

    try:
        out_rows, report = swap_file(args.date, args.file, objective=args.objective,
                                     slate=args.slate)
    except (SwapError, UploadError, OptimizerError) as error:
        print(f"[!] {error}")
        raise SystemExit(1)

    print(format_report(report))

    if out_rows is None:
        print("nothing to swap — every slot is still playable.")
        return
    if args.dry_run:
        print("dry run: no file written.")
        return

    from .naming import OUTPUT_ROOT, find_slates, resolve
    from .upload import write_upload

    version_note = None
    if args.out:
        out_path = args.out
    elif args.in_place:
        out_path = report["path"]
    else:
        label = args.slate or (find_slates(args.date, "upload", OUTPUT_ROOT) or [None])[0]
        out_path, version_note = resolve("swap", args.date, label, OUTPUT_ROOT, args.overwrite)
    write_upload(out_rows, out_path)
    print(f"-> {out_path}")
    if version_note:
        print(f"  [i] {version_note}")


if __name__ == "__main__":
    main()
