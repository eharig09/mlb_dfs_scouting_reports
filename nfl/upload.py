"""Write generated lineups into a DraftKings NFL upload file.

    python -m nfl.upload --date 2026-09-14 --slate main
    python -m nfl.upload --date 2026-09-14 --slate main --lineups 1,3,5-8

The football twin of `dfs.upload`. It reads `lineups_<slate>.csv`, fills a DK entries
template, and writes `upload_<slate>.csv` beside it. **The template you downloaded is never
modified** -- the filled copy is a new file, so a bad run cannot cost you the original.

## The two traps this file exists around

**`Name (ID)` is the cell format, and the id is the part that matters.** DK accepts a bare
id, but writes `Name (ID)` itself; a cell holding only a name is silently not an entry. Ids
are resolved against the template's own embedded player list wherever there is one, because
an id is only valid inside the draftgroup it came from.

**A template from the wrong slate will accept your lineups.** Names resolve against whatever
template is handed over, so last week's file happily produces a full sheet of valid-looking
ids for a contest that has already finished. Only the date catches it, so the date is
checked and a mismatch is refused rather than warned about.
"""

import argparse
import csv
import os
import re
import sys
from datetime import datetime

import pandas as pd

from nfl import naming
from nfl.salaries import canon_team, normalize_name

# DK NFL Classic, in the order the upload file lays the columns out.
SLOT_ORDER = ("QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST")

# Where DK entry exports are kept. Separate from the MLB `dk_lineups/` so a football
# template and a baseball one can never be picked up for each other.
TEMPLATE_DIR = os.path.join("nfl", "nfl_dfs", "dk_lineups")

CELL_WITH_ID = re.compile(r"^\s*(?P<name>.*?)\s*\((?P<id>\d+)\)\s*$")
DATE_IN_GAME_INFO = re.compile(r"(?P<month>\d{2})/(?P<day>\d{2})/(?P<year>\d{4})")

# Columns of the player list DK embeds to the right of the roster grid.
LIST_COLUMNS = ("Position", "Name + ID", "Name", "ID", "Roster Position", "Salary",
                "Game Info", "TeamAbbrev", "AvgPointsPerGame")


class UploadError(Exception):
    """The template, the lineups, or the pairing of the two is unusable."""


def _pad(row, width):
    return list(row) + [""] * (width - len(row))


def parse_cell(cell):
    """A roster cell -> (id, display name). Handles 'Name (ID)' and a bare id alike."""
    text = str(cell or "").strip()
    if not text:
        return "", ""
    match = CELL_WITH_ID.match(text)
    if match:
        return match.group("id"), match.group("name")
    if text.endswith(".0"):                 # pandas round-tripped the id as a float
        text = text[:-2]
    return (text, "") if text.isdigit() else ("", text)


def _slot_start(header):
    """Index of the run of nine roster columns, or None if this is not an upload layout."""
    cells = [str(c).strip().upper() for c in header]
    for start in range(len(cells) - len(SLOT_ORDER) + 1):
        if tuple(cells[start:start + len(SLOT_ORDER)]) == SLOT_ORDER:
            return start
    return None


def _locate_player_list(rows):
    """(header row index, {column name: index}) for DK's embedded player list."""
    for index, row in enumerate(rows):
        cells = [str(c).strip() for c in row]
        if "Name" in cells and "ID" in cells and "TeamAbbrev" in cells:
            return index, {name: cells.index(name) for name in LIST_COLUMNS
                           if name in cells}
    return None, {}


def player_index(rows):
    """({(name, team): id}, {name: id}) from the player list embedded in a template.

    Bare-name keys are kept **only when unambiguous**: two players sharing a name must be
    matched on team or not at all, since writing the wrong id is worse than refusing.
    """
    header_at, columns = _locate_player_list(rows)
    if header_at is None:
        return {}, {}
    name_column, id_column = columns.get("Name"), columns.get("ID")
    if name_column is None or id_column is None:
        return {}, {}
    team_column = columns.get("TeamAbbrev")

    by_pair, seen = {}, {}
    for row in rows[header_at + 1:]:
        width = max(name_column, id_column, team_column or 0) + 1
        if len(row) < width:
            continue
        name = str(row[name_column]).strip()
        player_id = str(row[id_column]).strip()
        if not name or not player_id:
            continue
        key = normalize_name(name)
        team = canon_team(row[team_column]) if team_column is not None else ""
        by_pair[(key, team)] = player_id
        seen.setdefault(key, set()).add(player_id)
    by_name = {key: next(iter(ids)) for key, ids in seen.items() if len(ids) == 1}
    return by_pair, by_name


def template_dates(rows):
    """Slate dates found in the template's embedded player list, as YYYY-MM-DD."""
    header_at, columns = _locate_player_list(rows)
    if header_at is None or "Game Info" not in columns:
        return set()
    column = columns["Game Info"]
    found = set()
    for row in rows[header_at + 1:]:
        if len(row) <= column:
            continue
        match = DATE_IN_GAME_INFO.search(str(row[column]))
        if match:
            found.add(f"{match.group('year')}-{match.group('month')}-{match.group('day')}")
    return found


def read_template(path):
    """Parse a DK upload file into (rows, kind, slot_start, index)."""
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = [list(row) for row in csv.reader(handle)]
    if not rows:
        raise UploadError(f"{path} is empty")
    slot_start = _slot_start(rows[0])
    if slot_start is None:
        raise UploadError(
            f"{path} is not a DK NFL upload file -- its header has no "
            f"{','.join(SLOT_ORDER)} columns")
    kind = "entries" if str(rows[0][0]).strip() == "Entry ID" else "bulk"
    return rows, kind, slot_start, player_index(rows)


def find_template(path=None, directory=TEMPLATE_DIR, date=None):
    """Resolve `--template` to a real file, or find the newest one in `directory`."""
    if path and os.path.isfile(path):
        return path
    if path and not os.path.isdir(path):
        raise UploadError(f"no template at {path}")
    where = path if path and os.path.isdir(path) else directory
    if not os.path.isdir(where):
        raise UploadError(
            f"no template directory {where} -- download the entries file from the contest "
            f"and put it there, or pass --template")
    candidates = [os.path.join(where, name) for name in os.listdir(where)
                  if name.lower().endswith(".csv") and not name.startswith("upload_")]
    if not candidates:
        raise UploadError(f"no csv templates in {where}")
    return max(candidates, key=os.path.getmtime)


def parse_selection(text, available):
    """'1,3,5-8' -> [1, 3, 5, 6, 7, 8]. Blank or 'all' selects everything."""
    available = list(available)
    text = str(text or "").strip().lower()
    if not text or text == "all":
        return available
    chosen = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token[1:]:
            low, _, high = token.partition("-")
            try:
                span = range(int(low), int(high) + 1)
            except ValueError:
                raise UploadError(f"bad --lineups range '{token}'")
            chosen.extend(span)
        else:
            try:
                chosen.append(int(token))
            except ValueError:
                raise UploadError(f"bad --lineups value '{token}'")
    missing = [n for n in chosen if n not in available]
    if missing:
        raise UploadError(f"no lineup numbered {missing} -- have 1..{max(available)}")
    return list(dict.fromkeys(chosen))


def read_lineups(path):
    """{lineup number: frame} from a `lineups_<slate>.csv`."""
    frame = pd.read_csv(path)
    for column in ("Lineup", "Name", "Slot"):
        if column not in frame.columns:
            raise UploadError(f"{path} has no {column} column -- is it a lineups file?")
    return {int(number): group for number, group in frame.groupby("Lineup")}


def order_players(frame, number):
    """The nine players in DK's column order. Raises when the lineup does not fit."""
    remaining = list(frame.itertuples(index=False))
    ordered = []
    for slot in SLOT_ORDER:
        for i, player in enumerate(remaining):
            if str(getattr(player, "Slot", "")).strip().upper() == slot:
                ordered.append(remaining.pop(i))
                break
        else:
            raise UploadError(
                f"lineup {number} has no player in the {slot} slot -- it has "
                f"{sorted(str(getattr(p, 'Slot', '')) for p in frame.itertuples(index=False))}")
    if remaining:
        extra = ", ".join(str(getattr(p, "Name", "?")) for p in remaining)
        raise UploadError(f"lineup {number} has {len(remaining)} player(s) over: {extra}")
    return ordered


def resolve_ids(players, index, number):
    """Roster cells for one lineup, as 'Name (ID)'. Returns (cells, unresolved).

    The template's own player list wins over the id carried on the lineup row, because an
    id is only meaningful inside the draftgroup it came from. The lineup's `DK ID` is the
    fallback for a bulk template that embeds no list at all.
    """
    by_pair, by_name = index
    cells, unresolved = [], []
    for player in players:
        name = str(getattr(player, "Name", "")).strip()
        key = normalize_name(name)
        team = canon_team(getattr(player, "Team", ""))
        player_id = by_pair.get((key, team)) or by_name.get(key)
        if not player_id:
            fallback = getattr(player, "DK ID", None) or getattr(player, "_asdict", dict)()
            raw = fallback.get("DK ID") if isinstance(fallback, dict) else fallback
            player_id, _ = parse_cell(raw)
        if not player_id:
            unresolved.append(f"{name} ({team or '?'})")
            cells.append("")
            continue
        cells.append(f"{name} ({player_id})")
    return cells, unresolved


def build_upload(rows, kind, slot_start, index, lineups, selection, date=None,
                 contest=None):
    """Fill the template with the selected lineups. Returns (output rows, notes)."""
    width = slot_start + len(SLOT_ORDER)
    notes = []

    # Names resolve against whatever template is handed over, so last week's file would
    # happily produce a full sheet of valid-looking ids for a contest that has finished.
    # Only the date catches that, which is why it is refused rather than warned about.
    if date:
        dates = template_dates(rows)
        if dates and str(date) not in dates:
            raise UploadError(
                f"the template is for {', '.join(sorted(dates))} but these lineups are for "
                f"{date} -- download the entries file from this week's contest")

    out = [list(rows[0])]
    if kind == "entries":
        targets = []
        for row in rows[1:]:
            entry = str(row[0]).strip() if row else ""
            if not entry or not entry.isdigit():
                continue
            if contest and contest.lower() not in str(row[1]).lower():
                continue
            targets.append(row)
        if not targets:
            raise UploadError(
                "the template has no contest entries to fill"
                + (f" matching '{contest}'" if contest else ""))
        if len(selection) > len(targets):
            notes.append(f"{len(selection)} lineups but only {len(targets)} entries; "
                         f"filling the first {len(targets)}")
            selection = selection[:len(targets)]

        filled = {id(row): row for row in targets}
        for row, number in zip(targets, selection):
            padded = _pad(row, max(width, len(row)))
            cells, unresolved = resolve_ids(order_players(lineups[number], number),
                                            index, number)
            if unresolved:
                raise UploadError(
                    f"lineup {number}: no DK id for {', '.join(unresolved)} -- the template "
                    f"is probably for a different slate")
            padded[slot_start:slot_start + len(SLOT_ORDER)] = cells
            filled[id(row)] = padded
        for row in rows[1:]:
            out.append(filled.get(id(row), list(row)))
    else:
        for number in selection:
            cells, unresolved = resolve_ids(order_players(lineups[number], number),
                                            index, number)
            if unresolved:
                raise UploadError(
                    f"lineup {number}: no DK id for {', '.join(unresolved)}")
            row = [""] * width
            row[slot_start:slot_start + len(SLOT_ORDER)] = cells
            out.append(row)
    return out, notes


def write_upload(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)
    return path


def summarize(lineups):
    """One line per generated lineup, for picking with --lineups."""
    lines = []
    for number, frame in lineups.items():
        first = frame.iloc[0]
        pass_game = frame[frame["Pos"].isin(("QB", "WR", "TE"))]["Team"].value_counts()
        stack = ", ".join(f"{team} x{count}" for team, count in pass_game.items()
                          if count >= 2)
        lines.append(
            f"  {number:>3}  ${int(first.get('Lineup Salary', 0)):,}  "
            f"proj {float(first.get('Lineup Proj', 0)):6.2f}  "
            f"ceil {float(first.get('Lineup Ceiling', 0)):6.2f}  {stack}")
    return "\n".join(lines)


def _run(args):
    date = args.date or datetime.today().strftime("%Y-%m-%d")
    source = args.source or naming.latest("lineups", date, args.slate, args.root)
    if not source or not os.path.exists(source):
        slates = naming.find_slates(date, "lineups", args.root)
        raise UploadError(
            f"no lineups for {args.slate or '(no slate)'} on {date}"
            + (f" -- have {', '.join(slates)}" if slates else " -- run nfl.optimize first"))
    lineups = read_lineups(source)
    print(f"lineups: {os.path.basename(source)}  ({len(lineups)} built)")

    if args.list:
        print(summarize(lineups))
        return 0

    template = find_template(args.template, args.template_dir, date)
    rows, kind, slot_start, index = read_template(template)
    print(f"template: {os.path.basename(template)}  ({kind})")

    selection = parse_selection(args.lineups, sorted(lineups))
    out, notes = build_upload(rows, kind, slot_start, index, lineups, selection,
                              date=date, contest=args.contest)
    for note in notes:
        print(f"  [!] {note}")

    path, note = naming.resolve("upload", date, args.slate, args.root,
                                overwrite=args.overwrite)
    write_upload(out, path)
    if note:
        print(f"  {note}")
    print(f"  wrote {len(selection)} lineups -> {path}")
    print("  upload that file to DK; the downloaded template is untouched")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Write generated NFL lineups into a DraftKings upload file.")
    parser.add_argument("--date", default=None, help="Defaults to today.")
    parser.add_argument("--slate", default=None, help="Slate label the lineups were filed under.")
    parser.add_argument("--lineups", default=None, help="Which to upload: '1,3,5-8', or all.")
    parser.add_argument("--list", action="store_true", help="Show the lineups and stop.")
    parser.add_argument("--template", default=None, help="DK entries file to fill.")
    parser.add_argument("--template-dir", default=TEMPLATE_DIR)
    parser.add_argument("--contest", default=None,
                        help="Entries templates only: fill entries whose contest name "
                             "contains this text.")
    parser.add_argument("--source", default=None, help="Lineups CSV. Default: the newest.")
    parser.add_argument("--root", default=naming.OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except UploadError as error:
        raise SystemExit(f"error: {error}")


if __name__ == "__main__":
    sys.exit(main())
