"""Turn generated lineups into a DraftKings upload file.

`dfs_boards/dfs_lineups_<date>.csv` is written to be read -- one row per player, with
projections and tiers alongside. DK wants the opposite: one row per lineup, ten player
ids in roster order and nothing else. This selects from what the optimizer produced and
writes it into DK's own template.

Two DK layouts exist and both are handled, detected from the header row:

  bulk     `P,P,C,1B,2B,3B,SS,OF,OF,OF` -- the "create up to 500 lineups" template that
           comes with an embedded player list. Rows are new lineups.
  entries  `Entry ID,Contest Name,Contest ID,Entry Fee,P,P,...` -- the export of contests
           already entered. Each row is an existing entry whose lineup is overwritten, so
           the number of lineups is capped by the number of entries.

Ids are resolved against the template's own player list rather than the `DK ID` recorded
when the lineups were built. DK numbers players per draft group, so a template downloaded
for a different contest -- or re-downloaded after a postponement reshuffled the slate --
renumbers everyone. An upload built from stale ids is rejected as a whole, with nothing in
the error naming the player at fault, so it is worth catching here.
"""

import argparse
import csv
import filecmp
import glob
import os
import re
from datetime import datetime

import pandas as pd

from .salaries import canon_team, normalize_name

TEMPLATE_DIR = "dk_lineups"
LINEUP_DIR = "dfs_boards"

# Written files land beside the template and share its layout, so they are skipped when
# looking for one -- otherwise the second run of a night finds two candidates and stops.
OUTPUT_PREFIX = "DKUpload_"
SWAP_PREFIX = "DKLateSwap_"
GENERATED_PREFIXES = (OUTPUT_PREFIX, SWAP_PREFIX)

# DK Classic, in the column order the upload file expects.
SLOT_ORDER = ("P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF")

# Stated on the template itself.
MAX_LINEUPS = 500


class UploadError(Exception):
    """Anything that would produce a file DK rejects."""


def lineups_path(date, directory=LINEUP_DIR, slate=None):
    """Newest lineup file for a night, falling back to the pre-dated-folder name.

    Older runs wrote `dfs_boards/dfs_lineups_<date>.csv` with no slate in the name; those
    files are still readable so an upload built from last week's lineups keeps working.
    """
    from .naming import find_slates, latest

    if slate is None:
        labels = find_slates(date, "lineups", directory)
        # One slate is unambiguous. Several means the night had more than one contest and
        # picking silently would upload the wrong one.
        if len(labels) == 1:
            slate = labels[0]
        elif len(labels) > 1:
            raise UploadError(
                f"{date} has lineups for {len(labels)} slates ({', '.join(labels)}) — "
                f"pass --slate to choose"
            )
    if slate is not None:
        found = latest("lineups", date, slate, directory)
        if found:
            return found
    return os.path.join(directory, f"dfs_lineups_{date}.csv")


def read_lineups(date, path=None, directory=LINEUP_DIR, slate=None):
    """{lineup number: frame} from the optimizer's output, in generated order."""
    path = path or lineups_path(date, directory, slate)
    if not os.path.exists(path):
        raise UploadError(f"no generated lineups at {path} — run: python -m dfs.optimize --date {date}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "Lineup" not in frame.columns or "Roster" not in frame.columns:
        raise UploadError(f"{path} is not an optimizer lineup file (no Lineup/Roster columns)")
    return {int(n): group for n, group in frame.groupby("Lineup", sort=True)}


def parse_selection(text, available):
    """'1,3,5-8' -> [1, 3, 5, 6, 7, 8]. Empty or 'all' takes everything.

    Selection order is preserved: DK ignores it, but a file that reads back in the order
    asked for is easier to check against the terminal output.
    """
    available = list(available)
    if not available:
        raise UploadError("no lineups to select from")
    if not text or str(text).strip().lower() in {"all", "*"}:
        return available

    span = f"{min(available)}-{max(available)}"
    picked = []
    for part in str(text).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            low, _, high = part.partition("-")
            if not (low.isdigit() and high.isdigit()):
                raise UploadError(f"bad --lineups range '{part}' (expected N-M)")
            wanted = range(int(low), int(high) + 1)
        elif part.isdigit():
            wanted = [int(part)]
        else:
            raise UploadError(f"bad --lineups value '{part}'")
        for number in wanted:
            if number not in available:
                raise UploadError(f"lineup {number} was not generated (have {span})")
            if number not in picked:
                picked.append(number)
    if not picked:
        raise UploadError("--lineups selected nothing")
    return picked


def _pad(row, width):
    return list(row) + [""] * (width - len(row))


def _slot_start(header):
    """Index of the run of ten roster columns, or None if this isn't an upload layout."""
    cells = [str(c).strip() for c in header]
    for i in range(len(cells) - len(SLOT_ORDER) + 1):
        if tuple(cells[i:i + len(SLOT_ORDER)]) == SLOT_ORDER:
            return i
    return None


def template_dates(rows):
    """Slate dates found in the template's embedded player list, as YYYY-MM-DD.

    DK writes 'SEA@LAD 07/28/2026 10:10PM ET' in Game Info. Postponed games carry no date
    and are skipped.
    """
    header_at = None
    for i, row in enumerate(rows):
        cells = [str(c).strip() for c in row]
        if "Game Info" in cells and "Name + ID" in cells:
            header_at, column = i, cells.index("Game Info")
            break
    if header_at is None:
        return set()

    found = set()
    for row in rows[header_at + 1:]:
        if len(row) <= column:
            continue
        match = re.search(r"(\d{2})/(\d{2})/(\d{4})", str(row[column]))
        if match:
            month, day, year = match.groups()
            found.add(f"{year}-{month}-{day}")
    return found


def _locate_player_list(rows):
    """(row index, {column name: index}) for the player list embedded in a template.

    The list sits below the lineup rows, offset into the far columns, so it is located by
    its own header rather than by a fixed position.
    """
    for i, row in enumerate(rows):
        cells = [str(c).strip() for c in row]
        if "Name + ID" in cells and "ID" in cells and "Name" in cells:
            return i, {name: j for j, name in enumerate(cells)}
    return None, {}


# A filled entry export writes roster cells as "Bailey Ober (38384954)"; the bulk template
# writes the bare id. Both have to be read, and -- more importantly -- written back in
# whichever form the file already uses, so an edited entry stays the shape DK handed out.
CELL_WITH_ID = re.compile(r"^\s*(?P<name>.*?)\s*\((?P<id>\d+)\)\s*$")


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


def id_index(rows):
    """{id: {key, team, display}} -- the player list read backwards.

    Late swap starts from a file of ids and has to work out who they are, which is the
    opposite direction from building an upload. `display` is DK's own "Name + ID" text,
    kept so a replacement can be written in the same form as the cell it replaces.
    """
    header_at, columns = _locate_player_list(rows)
    if header_at is None:
        return {}
    name_col, id_col = columns["Name"], columns["ID"]
    team_col = columns.get("TeamAbbrev")
    display_col = columns.get("Name + ID")
    width = max(name_col, id_col, team_col or 0, display_col or 0) + 1

    found = {}
    for row in rows[header_at + 1:]:
        if len(row) < width:
            continue
        name = str(row[name_col]).strip()
        player_id = str(row[id_col]).strip()
        if not name or not player_id:
            continue
        found[player_id] = {
            "key": normalize_name(name),
            "team": canon_team(row[team_col]) if team_col is not None else "",
            "display": (str(row[display_col]).strip() if display_col is not None
                        else f"{name} ({player_id})"),
        }
    return found


def _player_index(rows):
    """{(name, team): id} and {name: id} from the player list embedded in a template.

    Bare-name keys are only kept when unambiguous -- two players sharing a name must be
    matched on team or not at all.
    """
    header_at, columns = _locate_player_list(rows)
    if header_at is None:
        return {}, {}, set()

    name_col, id_col = columns["Name"], columns["ID"]
    team_col = columns.get("TeamAbbrev")
    width = max(name_col, id_col, team_col or 0) + 1

    by_pair, seen_names, all_ids = {}, {}, set()
    for row in rows[header_at + 1:]:
        if len(row) < width:
            continue
        name = str(row[name_col]).strip()
        player_id = str(row[id_col]).strip()
        if not name or not player_id:
            continue
        team = canon_team(row[team_col]) if team_col is not None else ""
        key = normalize_name(name)
        by_pair[(key, team)] = player_id
        seen_names.setdefault(key, set()).add(player_id)
        all_ids.add(player_id)

    by_name = {key: next(iter(ids)) for key, ids in seen_names.items() if len(ids) == 1}
    return by_pair, by_name, all_ids


# The bulk template is a fixed frame around a player list: ten roster columns, five lines
# of instructions, then the draft group's players from column 11 on. Reproduced verbatim
# from DK's own download so a synthesized file is byte-comparable to a real one.
BULK_INSTRUCTIONS = (
    "1. Locate the player you want to select in the list below ",
    "2. Copy the ID of your player (you can use the Name + ID column or the ID column) ",
    "3. Paste the ID into the roster position desired ",
    "4. You must include an ID for each player; you cannot use just the player's name ",
    "5. You can create up to 500 lineups per file ",
)

# Columns 0-9 are the roster slots, 10 is a spacer, and the player list starts at 11.
LIST_COLUMN = len(SLOT_ORDER) + 1

# The player-list columns DK's template carries, in its order. A salary export is the same
# list plus whatever else DK (or this project) has appended -- 'Status' and 'Starting' are
# there now -- so the rows are projected onto these by name rather than copied across.
LIST_COLUMNS = ("Position", "Name + ID", "Name", "ID", "Roster Position", "Salary",
                "Game Info", "TeamAbbrev", "AvgPointsPerGame")


def template_from_salaries(salary_path):
    """Reconstruct a bulk upload template from the slate's DK salary export.

    DK's bulk template and DK's salary export carry the *same* player list -- Position,
    Name + ID, Name, ID, Roster Position, Salary, Game Info, TeamAbbrev -- because both are
    generated from the draft group being entered. Everything that makes a template
    slate-specific therefore already sits in the export, and the rest of the file is a
    constant frame. So a night whose template was never downloaded is not actually missing
    anything: the ids written are DK's own, for the right draft group, not invented.

    Bulk only, and deliberately. An entries file also carries Entry ID, Contest ID and Entry
    Fee, which DK issues per entry when you enter a contest -- nothing local can supply
    those, so editing already-submitted entries (late swap) still needs the real download.
    """
    try:
        with open(salary_path, newline="", encoding="utf-8-sig") as handle:
            rows = [list(row) for row in csv.reader(handle)]
    except OSError as error:
        raise UploadError(f"cannot read {salary_path}: {error}") from None

    rows = [row for row in rows if any(str(cell).strip() for cell in row)]
    if not rows:
        raise UploadError(f"{salary_path} is empty")
    header = [str(cell).strip() for cell in rows[0]]
    missing = [column for column in LIST_COLUMNS if column not in header]
    if missing:
        raise UploadError(
            f"{os.path.basename(salary_path)} is missing {', '.join(missing)}, so it cannot "
            f"stand in for an upload template"
        )

    # By name, not by position: the export carries columns the template does not, and
    # copying it across verbatim would shift Game Info and the ids out from under anything
    # reading the result by column.
    picks = [header.index(column) for column in LIST_COLUMNS]
    blanks = [""] * LIST_COLUMN

    def listed(row):
        return blanks + [row[i] if i < len(row) else "" for i in picks]

    out = [list(SLOT_ORDER) + ["", "Instructions"]]
    out.extend(blanks + [line] for line in BULK_INSTRUCTIONS)
    out.append([" "])                                   # DK's own spacer row
    out.extend(listed(row) for row in rows)
    return out


def parse_template(rows, source="template"):
    """Rows of a DK upload file -> (rows, kind, slot_start, player index).

    Split out from read_template so a synthesized template goes through exactly the same
    validation and indexing as a downloaded one, rather than a parallel path that could
    drift from it.
    """
    if not rows:
        raise UploadError(f"{source} is empty")

    slot_start = _slot_start(rows[0])
    if slot_start is None:
        raise UploadError(
            f"{source} is not a DK upload file — its header has no "
            f"{','.join(SLOT_ORDER)} columns"
        )
    kind = "entries" if str(rows[0][0]).strip() == "Entry ID" else "bulk"
    by_pair, by_name, all_ids = _player_index(rows)
    return rows, kind, slot_start, (by_pair, by_name, all_ids)


def read_template(path):
    """Parse a DK upload file into (rows, kind, slot_start, player index)."""
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = [list(row) for row in csv.reader(handle)]
    return parse_template(rows, source=path)


def resolve_template_path(path, directory=TEMPLATE_DIR, date=None, teams=None):
    """Turn whatever the user typed for --template into a real path.

    DK's own filenames contain parentheses ("DKSalaries (3).csv"), which PowerShell splits
    on unless they are quoted exactly right -- so an exact-path-only flag turns a one-line
    command into a quoting puzzle. Accepted, in order: the path as given, the same name
    inside the template folder, and finally a unique case-insensitive substring of a
    filename there. Substring matches are narrowed by the requested date and lineup teams
    before they are called ambiguous; a slate nickname such as ``turbo`` is commonly reused
    every day and is not unique by itself.
    """
    if os.path.exists(path):
        return path
    inside = os.path.join(directory, os.path.basename(path))
    if os.path.exists(inside):
        return inside

    needle = os.path.splitext(os.path.basename(path))[0].lower()
    matches = [f for f in sorted(glob.glob(os.path.join(directory, "*.csv")))
               if needle and needle in os.path.basename(f).lower()]

    if len(matches) > 1 and date:
        dated = [(candidate, template_dates_for(candidate)) for candidate in matches]
        matching = [candidate for candidate, dates in dated if str(date) in dates]
        unknown = [candidate for candidate, dates in dated if not dates]
        if matching:
            matches = matching
        elif unknown:
            # Some hand-built templates have no embedded player list. They remain eligible,
            # but a file that explicitly advertises another date never does.
            matches = unknown
        else:
            listed = "\n".join(
                f"    {candidate} ({', '.join(sorted(dates))})"
                for candidate, dates in dated
            )
            raise UploadError(
                f"--template '{path}' matches files, but none is for {date}:\n{listed}"
            )

    if len(matches) > 1 and teams:
        covering = [candidate for candidate in matches
                    if not _team_mismatch(candidate, teams)]
        if covering:
            matches = covering

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        listed = "\n".join(f"    {m}" for m in matches)
        raise UploadError(f"--template '{path}' matches several files:\n{listed}")
    raise UploadError(f"no template at {path} (also looked in {directory}/)")


def _team_mismatch(path, teams):
    """Message describing a template that cannot hold these lineups, or None if it can."""
    if not teams:
        return None
    priced = template_teams_for(path)
    if not priced or teams <= priced:
        return None
    return (f"{os.path.basename(path)} prices {', '.join(sorted(priced))} — "
            f"it does not have {', '.join(sorted(teams - priced))}")


def _dedupe_template_files(paths):
    """Collapse byte-identical downloads, preferring the conventionally filed name."""
    ordered = sorted(
        paths,
        key=lambda path: (
            0 if os.path.basename(path).lower().startswith("dktemplate_") else 1,
            os.path.basename(path).lower(),
        ),
    )
    unique = []
    for candidate in ordered:
        duplicate = False
        for kept in unique:
            try:
                if filecmp.cmp(candidate, kept, shallow=False):
                    duplicate = True
                    break
            except OSError:
                pass
        if not duplicate:
            unique.append(candidate)
    return unique


def _which_contest_hint(date, teams, directory, searched_downloads=False):
    """Name the contest whose template is missing, so it can be found on DK's site.

    "Download the right one" is not actionable when a date ran three contests. The salary
    exports already on disk say which slate these teams belong to, so the games and the
    slate label can both be named -- that is what identifies the contest page to open.
    """
    from .naming import slate_for
    from .salaries import list_salary_files

    lines = []
    exports = list_salary_files(date)
    for info in exports:
        priced = set()
        for game in info.get("games") or []:
            priced |= {canon_team(part) for part in str(game).split("@")}
        if teams <= priced:
            label = slate_for(info, date=date, peers=exports)
            lines.append(f"  You need the '{label}' slate's template "
                         f"({', '.join(info.get('games') or [])}).")
            break
    if not lines:
        lines.append(f"  {date} ran more than one contest.")
    lines.append(f"  Download that contest's entry file into {directory}/ — it can keep "
                 f"whatever name DK gives it.")
    if not searched_downloads:
        lines.append("  Downloads/ is not searched unless you pass --downloads.")
    return "\n".join(lines)


def find_template(path=None, directory=TEMPLATE_DIR, date=None, teams=None,
                  searched_downloads=False):
    """Locate the upload template for a slate. Several candidates is an error, not a coin flip.

    Filtered on both the date and the **teams** in the embedded player list. The date alone
    is not enough: 2026-07-30 ran an early and a main contest, both dated the same, and
    picking on date would hand back a template whose player list has none of your hitters --
    which surfaces much later as ten unresolvable ids and reads like a corrupted file.
    """
    if path:
        found = resolve_template_path(path, directory, date=date, teams=teams)
        found_dates = template_dates_for(found)
        if date and found_dates and str(date) not in found_dates:
            raise UploadError(
                f"{os.path.basename(found)} is for {', '.join(sorted(found_dates))}, "
                f"not the requested date {date}. Choose that date's template."
            )
        problem = _team_mismatch(found, teams)
        if problem:
            raise UploadError(
                f"{problem}.\nThat template is for a different contest than these lineups. "
                f"Download the upload file from the slate you are entering."
            )
        return found

    candidates, entries_only = [], []
    for found in sorted(glob.glob(os.path.join(directory, "*.csv"))):
        if os.path.basename(found).startswith(GENERATED_PREFIXES):
            continue
        try:
            with open(found, newline="", encoding="utf-8-sig") as handle:
                header = next(csv.reader(handle), [])
        except OSError:
            continue
        if _slot_start(header) is None:
            continue
        # Filling an entries export rewrites lineups already submitted, so a blank bulk
        # template wins when both are present -- but an entries file is still usable when
        # it is all there is.
        if str((header or [""])[0]).strip() == "Entry ID":
            entries_only.append(found)
        else:
            candidates.append(found)

    candidates = _dedupe_template_files(candidates or entries_only)
    if not candidates:
        raise UploadError(
            f"no DK upload template in {directory}/ — download one from the contest's "
            f"lineup page and drop it there"
        )

    if date:
        dated = [(c, template_dates_for(c)) for c in candidates]
        matching = [c for c, dates in dated if not dates or str(date) in dates]
        if not matching:
            found = sorted({d for _, dates in dated for d in dates})
            raise UploadError(
                f"no upload template for {date} in {directory}/ — the {len(candidates)} "
                f"there {'is' if len(candidates) == 1 else 'are'} for "
                f"{', '.join(found) or 'an unknown date'}. Download tonight's from the "
                f"contest's lineup page."
            )
        candidates = matching

    if teams:
        covering = [c for c in candidates if not _team_mismatch(c, teams)]
        if not covering:
            detail = "\n".join(f"    {_team_mismatch(c, teams)}" for c in candidates)
            raise UploadError(
                f"no upload template covers these lineups (need "
                f"{', '.join(sorted(teams))}):\n{detail}\n"
                f"{_which_contest_hint(date, teams, directory, searched_downloads)}"
            )
        candidates = covering

    if len(candidates) > 1:
        listed = "\n".join(f"    {c}" for c in candidates)
        raise UploadError(
            f"several upload templates for {date or 'this slate'} in {directory}/ — "
            f"pass --template to choose (a distinctive part of the name is enough):\n{listed}"
        )
    return candidates[0]


def template_dates_for(path):
    """Slate dates a template file carries, read from its embedded player list."""
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            return template_dates([list(row) for row in csv.reader(handle)])
    except OSError:
        return set()


def template_teams(rows):
    """Team abbreviations a template prices, from its embedded player list.

    A date is not enough to identify a template -- 2026-07-30 had an early and a main
    contest, both dated the same, pricing different clubs. The teams are what actually say
    which contest a template belongs to.
    """
    header_at, columns = _locate_player_list(rows)
    if header_at is None or "TeamAbbrev" not in columns:
        return set()
    column = columns["TeamAbbrev"]
    found = set()
    for row in rows[header_at + 1:]:
        if len(row) > column:
            team = canon_team(row[column])
            if team:
                found.add(team)
    return found


def template_teams_for(path):
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            return template_teams([list(row) for row in csv.reader(handle)])
    except OSError:
        return set()


def lineup_teams(lineups, selection=None):
    """Every team appearing in the lineups about to be uploaded."""
    wanted = selection if selection is not None else list(lineups)
    teams = set()
    for number in wanted:
        frame = lineups.get(number)
        if frame is None:
            continue
        teams |= {canon_team(t) for t in frame.get("Team", []) if str(t).strip()}
    return {t for t in teams if t}


def _order_players(frame, number):
    """The ten players of one lineup, in DK's column order.

    Doubles as a roster check: anything that isn't exactly DK Classic fails here rather
    than uploading a file DK silently drops.
    """
    by_slot = {}
    for row in frame.to_dict("records"):
        by_slot.setdefault(str(row.get("Roster") or "").strip().upper(), []).append(row)

    ordered = []
    for slot in SLOT_ORDER:
        bucket = by_slot.get(slot)
        if not bucket:
            raise UploadError(f"lineup {number} has no {slot} — not a DK Classic roster")
        ordered.append(bucket.pop(0))
    leftover = sorted(row["Name"] for bucket in by_slot.values() for row in bucket)
    if leftover:
        raise UploadError(f"lineup {number} has {len(leftover)} extra player(s): {', '.join(leftover)}")
    return ordered


def resolve_ids(players, index, number):
    """Player rows -> DK ids, preferring the template's numbering over the stored one."""
    by_pair, by_name, all_ids = index
    ids, problems = [], []
    for row in players:
        name = str(row.get("Name") or "")
        team = canon_team(row.get("Team"))
        key = normalize_name(name)
        stored = str(row.get("DK ID") or "").strip()
        if stored.endswith(".0"):                      # pandas read the id column as float
            stored = stored[:-2]

        found = by_pair.get((key, team)) or by_name.get(key)
        if not found and all_ids and stored in all_ids:
            # The template has no row under this spelling but does know the stored id, so
            # the id is valid for this draft group and only the name failed to match.
            found = stored
        if not found and not all_ids:
            found = stored                             # template carries no player list
        if not found:
            problems.append(f"{name} ({team})")
            continue
        ids.append(found)

    if problems:
        raise UploadError(
            f"lineup {number}: {len(problems)} player(s) not in the template's player list "
            f"— {', '.join(problems)}. The template is probably for a different slate than "
            f"the salary file the lineups were built from."
        )
    return ids


def build_upload(rows, kind, slot_start, index, lineups, selection, contest=None, date=None):
    """Fill the template with the selected lineups. Returns (output rows, notes)."""
    width = slot_start + len(SLOT_ORDER)
    notes = []

    # Names resolve against the template whatever slate it is for, so a template left over
    # from a previous night would happily accept today's lineups and produce a file of
    # valid-looking ids for the wrong contest. Only the date catches that.
    if date:
        dates = template_dates(rows)
        if dates and str(date) not in dates:
            raise UploadError(
                f"the template is for {', '.join(sorted(dates))} but these lineups are for "
                f"{date} — download the upload file from tonight's contest"
            )

    filled = []
    for number in selection:
        players = _order_players(lineups[number], number)
        filled.append((number, resolve_ids(players, index, number)))

    if len(filled) > MAX_LINEUPS:
        raise UploadError(f"{len(filled)} lineups exceeds DK's {MAX_LINEUPS} per file")

    if kind == "entries":
        # Each row is a contest entry, so lineups can only go where an entry exists. DK
        # asks that unchanged entries be left out, so only the filled rows are written.
        header = _pad(rows[0], width)
        entries = [
            _pad(row, width) for row in rows[1:]
            if row and str(row[0]).strip()
            and (not contest or contest.lower() in str(row[1] if len(row) > 1 else "").lower())
        ]
        if not entries:
            where = f" matching '{contest}'" if contest else ""
            raise UploadError(f"the template has no contest entries{where} to fill")
        if len(filled) > len(entries):
            raise UploadError(
                f"{len(filled)} lineups but only {len(entries)} entr"
                f"{'y' if len(entries) == 1 else 'ies'} in the template"
            )
        out = [header[:width]]
        for (number, ids), entry in zip(filled, entries):
            entry[slot_start:width] = ids
            out.append(entry[:width])
            notes.append(f"lineup {number} -> entry {entry[0]}")
        if len(entries) > len(filled):
            notes.append(f"{len(entries) - len(filled)} entr"
                         f"{'y' if len(entries) - len(filled) == 1 else 'ies'} left unchanged")
        return out, notes

    # Bulk template: keep it intact -- the instructions and player list live in the columns
    # to the right of the roster slots and DK's parser expects to see the file it handed
    # out. Only the ten roster columns are written.
    out = [list(row) for row in rows]
    needed = 1 + len(filled)
    while len(out) < needed:
        out.append([])
    for i, row in enumerate(out):
        if i == 0:
            continue
        padded = _pad(row, width)
        if i <= len(filled):
            number, ids = filled[i - 1]
            padded[slot_start:width] = ids
            notes.append(f"lineup {number} -> row {i + 1}")
        else:
            # Clear rather than leave: a template that was filled once already would
            # otherwise upload those stale lineups alongside the new ones.
            padded[slot_start:width] = [""] * len(SLOT_ORDER)
        out[i] = padded
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
        teams = frame[frame["Roster"] != "P"]["Team"].value_counts()
        stack = ", ".join(f"{team} x{count}" for team, count in teams.items() if count >= 2)
        lines.append(
            f"  {number:>3}  ${int(first.get('Lineup Salary', 0)):,}  "
            f"proj {float(first.get('Lineup Proj', 0)):6.2f}  "
            f"ceil {float(first.get('Lineup Ceiling', 0)):6.2f}  {stack}"
        )
    return "\n".join(lines)


def resolve_template(date, slate=None, template=None, teams=None, searched_downloads=False):
    """(rows, kind, slot_start, index, description) for tonight's upload template.

    Prefers a real DK download, and rebuilds one from the slate's salary export when there
    isn't one. A night whose template was never downloaded is not actually missing anything
    for a *bulk* entry: DK's bulk template is a fixed frame wrapped around the draft group's
    player list, and the salary export the board was priced from is that same player list --
    same ids, same slate, issued by DK. `dfs.optimize --upload` has always done this; doing
    it here too means the standalone path is not the one that dead-ends.

    Two things it deliberately will not do:

    * **Substitute for an explicit `--template`.** Naming a file that then failed is a
      mistake worth surfacing, not one to paper over.
    * **Stand in for an entries export.** Editing entries you have already submitted needs
      Entry ID, Contest ID and Entry Fee, which DK issues per entry. Nothing local can
      invent those, so late swap still needs the real download.
    """
    try:
        path = find_template(template, date=date, teams=teams,
                             searched_downloads=searched_downloads)
        rows, kind, slot_start, index = read_template(path)
        return rows, kind, slot_start, index, f"template {path} ({kind})"
    except UploadError as error:
        if template:
            raise

        from .salaries import AmbiguousSlate, find_salary_file
        try:
            salary_file = find_salary_file(date, slate=slate)
        except AmbiguousSlate as choice:
            names = ", ".join(candidate["name"] for candidate in choice.candidates)
            raise UploadError(
                f"{error}\n    Several DK exports match {date} ({names}), so a template "
                f"cannot be rebuilt without knowing which contest. Pass --slate."
            ) from None
        if not salary_file:
            raise

        rows, kind, slot_start, index = parse_template(
            template_from_salaries(salary_file), source=os.path.basename(salary_file))
        print(f"[i] no DK template for {date} — {error}")
        print(f"[i] rebuilding one from {os.path.basename(salary_file)}; its ids are DK's "
              f"own for this slate.")
        print("    Bulk entry only — editing entries you have already submitted (late "
              "swap) still needs the real entries export.")
        return (rows, kind, slot_start, index,
                f"rebuilt from {os.path.basename(salary_file)} ({kind})")


def main():
    parser = argparse.ArgumentParser(
        description="Write selected optimizer lineups into a DraftKings upload file.")
    parser.add_argument("--date", default=datetime.today().strftime("%Y-%m-%d"))
    parser.add_argument("--lineups", help="Which to upload: '1,3,5-8', or all (default).")
    parser.add_argument("--template", help=f"DK upload file. Default: the one in {TEMPLATE_DIR}/.")
    parser.add_argument("--contest", help="Entry-export templates only: fill entries whose "
                                          "contest name contains this text.")
    parser.add_argument("--source", help="Optimizer lineup CSV. Default: the newest "
                                         f"{LINEUP_DIR}/<date>/lineups_<slate>.csv.")
    parser.add_argument("--slate", help="Which slate's lineups to upload, when a night has "
                                        "more than one.")
    parser.add_argument("--out", help=f"Where to write. Default: {LINEUP_DIR}/<date>/upload_<slate>.csv.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing upload file instead of writing a new .rN version.")
    parser.add_argument("--no-adopt", action="store_true",
                        help="Do not rename misnamed DK downloads into the convention.")
    parser.add_argument("--downloads", action="store_true",
                        help="Also look in your Downloads folder for the template. Files "
                             "found there are copied into dk_lineups/, never moved.")
    parser.add_argument("--list", action="store_true",
                        help="Show the generated lineups and exit, without writing.")
    args = parser.parse_args()

    # A template DK saved as "DKSalaries (3).csv" is the normal case, not the exception.
    # Filing it here as well as in dfs.cli means the name never has to be typed -- and
    # typing it is the part that goes wrong, because the parentheses need quoting.
    if not args.no_adopt and not args.list:
        from .adopt import DOWNLOADS_DIR, SEARCH_DIRS, adopt_for_date, describe
        search = SEARCH_DIRS + ([DOWNLOADS_DIR] if args.downloads else [])
        adopted, problems = adopt_for_date(args.date, kinds=("template",), search_dirs=search)
        if adopted:
            print(f"Filed {len(adopted)} unnamed DraftKings download(s):")
            for line in describe(adopted):
                print(line)
            print()
        for problem in problems:
            print(f"[!] {problem}")

    try:
        lineups = read_lineups(args.date, args.source, slate=args.slate)
        if args.list:
            print(f"{len(lineups)} lineups for {args.date}:")
            print(summarize(lineups))
            return
        selection = parse_selection(args.lineups, lineups.keys())
        rows, kind, slot_start, index, source = resolve_template(
            args.date, slate=args.slate, template=args.template,
            teams=lineup_teams(lineups, selection), searched_downloads=args.downloads)
        out_rows, notes = build_upload(rows, kind, slot_start, index, lineups, selection,
                                       contest=args.contest, date=args.date)
    except UploadError as error:
        print(f"[!] {error}")
        raise SystemExit(1)

    from .naming import find_slates, resolve
    label = args.slate or (find_slates(args.date, "lineups", LINEUP_DIR) or [None])[0]
    if args.out:
        out_path, version_note = args.out, None
    else:
        out_path, version_note = resolve("upload", args.date, label, LINEUP_DIR, args.overwrite)
    write_upload(out_rows, out_path)

    print(source)
    for note in notes:
        print(f"  {note}")
    print(f"{len(selection)} lineup(s) -> {out_path}")
    if version_note:
        print(f"  [i] {version_note}")


if __name__ == "__main__":
    main()
