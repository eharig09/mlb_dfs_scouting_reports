"""Load a DraftKings salary export and match it to projected players.

DK has no public salary API, so the workflow is: download `DKSalaries.csv` from the
contest page and drop it in `dfs/salaries/` (named with the slate date, or passed
explicitly). The export also defines the slate -- if a game isn't in the file, it isn't
in the contest, so matching doubles as the slate filter.
"""

import glob
import os
import re
import unicodedata

import pandas as pd

# Searched in order; the daily drop folder wins over the dated archive.
SALARY_DIRS = ["dfs_daily_files", os.path.join("dfs", "salaries")]

# DK's abbreviations drift from the ones StatsAPI (and the scouting report) uses.
TEAM_ALIASES = {
    "AZ": "ARI",
    "WAS": "WSH", "WSN": "WSH",
    "CHW": "CWS", "CHA": "CWS",
    "CHN": "CHC",
    "OAK": "ATH",
    "SDP": "SD", "SFG": "SF", "TBR": "TB", "KCR": "KC",
    "NYA": "NYY", "NYN": "NYM", "LAN": "LAD", "ANA": "LAA",
}

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def canon_team(abbr):
    key = str(abbr or "").strip().upper()
    return TEAM_ALIASES.get(key, key)


def normalize_name(name):
    """Fold accents, punctuation, and suffixes so DK and StatsAPI spellings meet."""
    text = unicodedata.normalize("NFKD", str(name or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("'", "").replace("`", "")
    text = re.sub(r"[.\-]", " ", text)
    parts = [p for p in re.split(r"\s+", text) if p and p not in SUFFIXES]
    return " ".join(parts)


def _short_key(name):
    """First initial + last name, for spellings that differ on the given name."""
    parts = normalize_name(name).split()
    if len(parts) < 2:
        return normalize_name(name)
    return f"{parts[0][:1]} {parts[-1]}"


# A real DK export carries these. Checking them stops a look-alike CSV in the same folder
# -- an exported pool file, say -- from being read as a salary file: it has Name and
# Salary columns too, so nothing downstream would notice until positions came back empty.
DK_SIGNATURE_COLUMNS = {"TeamAbbrev", "Game Info", "Roster Position", "Name + ID"}


def is_salary_file(path):
    """True only if the CSV really looks like a DraftKings salary export."""
    try:
        header = pd.read_csv(path, nrows=0)
    except Exception:
        return False
    columns = {str(c).strip() for c in header.columns}
    return {"Name", "Salary"} <= columns and bool(DK_SIGNATURE_COLUMNS & columns)


def describe_slate(path):
    """Fingerprint a DK export from its own contents.

    Files can be named anything -- detection is content-based -- so the slate is described
    from the games it actually prices rather than from a filename convention.
    """
    info = {"path": path, "name": os.path.basename(path), "date": None,
            "games": [], "teams": 0, "first": None, "last": None, "players": 0}
    try:
        frame = pd.read_csv(path)
    except Exception:
        return info

    info["players"] = len(frame)
    if "TeamAbbrev" in frame.columns:
        info["teams"] = int(frame["TeamAbbrev"].nunique())

    times = []
    for value in frame.get("Game Info", pd.Series(dtype=str)).dropna().astype(str).unique():
        match = re.match(r"\s*(\S+@\S+)\s+(\d{2})/(\d{2})/(\d{4})\s+(\d{1,2}):(\d{2})(AM|PM)", value)
        if not match:
            continue
        game, month, day, year, hour, minute, meridiem = match.groups()
        info["games"].append(game)
        info["date"] = f"{year}-{month}-{day}"
        hour = int(hour) % 12 + (12 if meridiem == "PM" else 0)
        times.append(hour * 60 + int(minute))

    info["games"] = sorted(set(info["games"]))
    if times:
        info["first"], info["last"] = min(times), max(times)
    return info


def _clock(minutes):
    if minutes is None:
        return "?"
    hour, minute = divmod(minutes, 60)
    meridiem = "AM" if hour < 12 else "PM"
    return f"{hour % 12 or 12}:{minute:02d}{meridiem}"


def slate_label(info):
    """One-line human description used to tell two exports apart."""
    count = len(info["games"])
    if count == 1:
        return f"Showdown {info['games'][0]} ({info['players']} players)"
    window = f"{_clock(info['first'])}-{_clock(info['last'])} ET" if info["first"] is not None else "times unknown"
    return f"{count} games, {window} ({info['players']} players)"


def list_salary_files(date=None, salary_dirs=None):
    """Every valid DK export in the search path, described. Optionally filtered to a date."""
    seen, found = set(), []
    for directory in (salary_dirs or SALARY_DIRS):
        for path in sorted(glob.glob(os.path.join(directory, "*.csv"))):
            real = os.path.normcase(os.path.abspath(path))
            if real in seen or not is_salary_file(path):
                continue
            seen.add(real)
            info = describe_slate(path)
            if date and info["date"] and info["date"] != str(date):
                continue
            found.append(info)
    return found


class AmbiguousSlate(Exception):
    """Raised when several exports match and none was chosen."""

    def __init__(self, candidates):
        self.candidates = candidates
        super().__init__("multiple salary files match")


def find_salary_file(date, path=None, salary_dirs=None, slate=None):
    """Locate the DK export for a slate.

    path:  explicit file, wins outright.
    slate: case-insensitive substring matched against the filename, so renaming exports
           (DKSalaries_2026-07-27_main.csv, ..._early.csv) is all that is needed to pick
           between them.

    Raises AmbiguousSlate when several exports match and none was chosen -- silently
    picking one would price the board off the wrong contest.
    """
    if path:
        return path if os.path.exists(path) else None

    candidates = list_salary_files(date, salary_dirs)
    if not candidates:
        # Nothing carries this date; fall back to any valid export so a mislabelled or
        # undated file still works, with the date guard downstream catching a mismatch.
        candidates = list_salary_files(None, salary_dirs)
    if not candidates:
        return None

    if slate:
        needle = str(slate).lower()
        matched = [c for c in candidates if needle in c["name"].lower()]
        if not matched:
            raise AmbiguousSlate(candidates)
        candidates = matched

    if len(candidates) > 1:
        raise AmbiguousSlate(candidates)
    return candidates[0]["path"]


# DK prices injured players but they cannot be rostered, so they must never reach the board.
UNAVAILABLE_STATUS = {"IL", "OUT", "NA", "SUSP"}

# Once a game is called, DK replaces the "ATL@NYM 07/28/2026 07:10PM ET" in Game Info with
# this. The players stay in the file at full price, so nothing else marks them unplayable.
POSTPONED_GAME_INFO = re.compile(r"^\s*(postponed|cancell?ed)\s*$", re.IGNORECASE)


def postponed_in_export(path):
    """{team abbr} that the DK export itself marks postponed. Empty for a pre-postponement file."""
    try:
        frame = pd.read_csv(path)
    except Exception:
        return set()
    if "Game Info" not in frame.columns or "TeamAbbrev" not in frame.columns:
        return set()
    called = frame["Game Info"].astype(str).str.match(POSTPONED_GAME_INFO)
    return {canon_team(t) for t in frame.loc[called, "TeamAbbrev"].dropna().unique()}


def load_salaries(path):
    """Read a DK export into a normalized frame, or None if it isn't one."""
    frame = pd.read_csv(path)
    frame.columns = [str(c).strip() for c in frame.columns]
    if "Salary" not in frame.columns or "Name" not in frame.columns:
        return None

    if "Status" in frame.columns:
        status = frame["Status"].astype(str).str.strip().str.upper()
        frame = frame[~status.isin(UNAVAILABLE_STATUS)].copy()

    out = pd.DataFrame({
        "DK Name": frame["Name"].astype(str).str.strip(),
        "Salary": pd.to_numeric(frame["Salary"], errors="coerce"),
    })
    out["DK Team"] = (
        frame["TeamAbbrev"].astype(str).map(canon_team) if "TeamAbbrev" in frame.columns else ""
    )
    out["DK Pos"] = (
        frame["Roster Position"].astype(str) if "Roster Position" in frame.columns
        else frame.get("Position", pd.Series("", index=frame.index)).astype(str)
    )
    out["DK Avg"] = pd.to_numeric(frame.get("AvgPointsPerGame"), errors="coerce")
    out["DK ID"] = frame.get("ID")
    out["DK Status"] = frame["Status"].astype(str).str.strip() if "Status" in frame.columns else ""
    # DK flags confirmed starting pitchers with "P" once lineups are posted.
    out["DK Starting"] = (
        frame["Starting"].astype(str).str.strip().str.upper().eq("P")
        if "Starting" in frame.columns else False
    )
    out["_key"] = out["DK Name"].map(normalize_name)
    out["_short"] = out["DK Name"].map(_short_key)
    return out.dropna(subset=["Salary"])


def slate_game_times(path):
    """{"AWAY@HOME": set of start times as minutes past midnight ET} from a DK export.

    Needed to tell doubleheader games apart: both share a pairing, so the start time is
    the only thing in the salary file that distinguishes which one is on the slate.
    """
    try:
        frame = pd.read_csv(path)
    except Exception:
        return {}
    if "Game Info" not in frame.columns:
        return {}
    times = {}
    for value in frame["Game Info"].dropna().astype(str).unique():
        match = re.match(r"\s*(\S+@\S+)\s+\d{2}/\d{2}/\d{4}\s+(\d{1,2}):(\d{2})(AM|PM)", value)
        if not match:
            continue
        game, hour, minute, meridiem = match.groups()
        away, _, home = game.partition("@")
        key = f"{canon_team(away)}@{canon_team(home)}"
        hour = int(hour) % 12 + (12 if meridiem == "PM" else 0)
        times.setdefault(key, set()).add(hour * 60 + int(minute))
    return times


def slate_games(path):
    """Games on the contest slate, as {"AWAY@HOME"}, read from the DK export.

    The salary file is the authority on what the slate contains, so this is what the
    cached reports have to be checked against.
    """
    try:
        frame = pd.read_csv(path)
    except Exception:
        return set()
    if "Game Info" not in frame.columns:
        return set()
    games = set()
    for value in frame["Game Info"].dropna().astype(str):
        token = value.strip().split(" ")[0]
        if "@" in token:
            away, home = token.split("@", 1)
            games.add(f"{canon_team(away)}@{canon_team(home)}")
    return games


def slate_date(path):
    """The date the DK export covers, as YYYY-MM-DD, or None if it can't be read.

    Guards against the easy mistake of building a board from yesterday's salary file,
    which matches enough names to look plausible while being entirely wrong.
    """
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    if "Game Info" not in frame.columns:
        return None
    dates = []
    for value in frame["Game Info"].dropna().astype(str):
        match = re.search(r"(\d{2})/(\d{2})/(\d{4})", value)
        if match:
            month, day, year = match.groups()
            dates.append(f"{year}-{month}-{day}")
    if not dates:
        return None
    return max(set(dates), key=dates.count)


def attach_salaries(projections, salaries):
    """Join salaries onto projections and compute value. Unmatched rows keep a blank salary.

    Matching goes name+team, then name, then initial+surname+team -- team is checked first
    because DK slates routinely carry two players with the same common name.
    """
    frame = projections.copy()
    frame["_key"] = frame["Name"].map(normalize_name)
    frame["_short"] = frame["Name"].map(_short_key)
    frame["_team"] = frame["Team"].map(canon_team)

    columns = ("Salary", "DK Pos", "DK Avg", "DK ID", "DK Name", "DK Starting")

    if salaries is None or salaries.empty:
        for column in columns:
            frame[column] = pd.Series([None] * len(frame), index=frame.index, dtype=object)
        return frame.drop(columns=["_key", "_short", "_team"])

    by_name_team = {(r["_key"], r["DK Team"]): r for _, r in salaries.iterrows()}
    by_short_team = {(r["_short"], r["DK Team"]): r for _, r in salaries.iterrows()}
    name_counts = salaries["_key"].value_counts()
    by_name = {r["_key"]: r for _, r in salaries.iterrows() if name_counts.get(r["_key"], 0) == 1}

    # Collected into lists and assigned as whole columns. Pre-creating the columns with
    # `= None` made them float64, which silently swallowed every string written per-cell
    # -- salaries landed but DK Pos and DK Name came back empty for the entire slate.
    collected = {column: [] for column in columns}
    for _, row in frame.iterrows():
        # Explicit None checks: the candidates are pandas rows, and `or` on a Series
        # raises rather than falling through.
        match = by_name_team.get((row["_key"], row["_team"]))
        if match is None:
            match = by_name.get(row["_key"])
        if match is None:
            match = by_short_team.get((row["_short"], row["_team"]))

        if match is None:
            for column in columns:
                collected[column].append(None)
            continue
        collected["Salary"].append(float(match["Salary"]))
        collected["DK Pos"].append(str(match["DK Pos"]))
        collected["DK Avg"].append(match["DK Avg"])
        collected["DK ID"].append(match["DK ID"])
        collected["DK Name"].append(str(match["DK Name"]))
        collected["DK Starting"].append(bool(match.get("DK Starting", False)))

    for column in columns:
        frame[column] = pd.Series(collected[column], index=frame.index, dtype=object)
    frame["Salary"] = pd.to_numeric(frame["Salary"], errors="coerce")
    frame["DK Avg"] = pd.to_numeric(frame["DK Avg"], errors="coerce")

    return frame.drop(columns=["_key", "_short", "_team"])
