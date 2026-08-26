"""Team-code and player-name folding for NFL DK exports.

The MLB twin (`dfs.salaries`) also owns finding and parsing the DK salary file. That half
is not here yet -- this scaffold carries only the two helpers the optimizer needs, so the
solver can be exercised against a synthetic slate before any file format work exists.

Both functions are deliberately the same shape as the MLB ones. Every downstream module
that folds a team code or matches a typed name calls through here, which is what stops a
board from silently matching nothing when a source spells a club differently.
"""

import re
import unicodedata

# DK, nflverse, ESPN and the networks disagree on five clubs, and the disagreements are
# stable. Folding both directions is deliberate: a caller may hand over either spelling.
TEAM_ALIASES = {
    "JAX": "JAC",
    "WSH": "WAS", "WFT": "WAS",
    "LA": "LAR", "STL": "LAR",
    "SD": "LAC",
    "OAK": "LV", "LVR": "LV",
    "ARZ": "ARI",
    "BLT": "BAL", "CLV": "CLE", "HST": "HOU",
    "NO": "NOR", "NOS": "NOR",
    "TB": "TAM", "TBB": "TAM",
    "KC": "KAN", "KCC": "KAN",
    "SF": "SFO", "GB": "GNB", "NE": "NWE",
}

# The 32 canonical codes, after folding. Used by the fixtures and by any caller that wants
# to reject a typo rather than pass it through as a new "team".
TEAMS = (
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GNB", "HOU", "IND", "JAC", "KAN",
    "LAC", "LAR", "LV", "MIA", "MIN", "NOR", "NWE", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SFO", "TAM", "TEN", "WAS",
)

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# PFF writes its own position vocabulary, and it is not DK's. The one that bites is **`HB`**:
# PFF has no `RB` at all, so a filter on "RB" matches nothing and every running back silently
# leaves the frame -- a whole position's target share came back blank league-wide before this
# existed. `FB` folds to `RB` too, matching DK's roster eligibility.
#
# Only genuine spelling disagreements are folded. `T`/`G`/`C` stay distinct because where a
# lineman plays is the point of the line work, and `DI`/`ED` stay distinct for the same
# reason on the other side.
POSITION_ALIASES = {
    "HB": "RB", "FB": "RB", "RB": "RB",
    "WR": "WR", "TE": "TE", "QB": "QB",
}


def canon_position(position):
    """Fold a position label to the package's spelling. Unknown labels pass through
    unchanged, so a line or defensive position keeps its own identity."""
    key = str(position or "").strip().upper()
    return POSITION_ALIASES.get(key, key)


def canon_team(abbr):
    """Fold a team code to its canonical spelling. Unknown codes pass through unchanged,
    so a new or relocated club is never silently dropped."""
    key = str(abbr or "").strip().upper()
    return TEAM_ALIASES.get(key, key)


def normalize_name(name):
    """Fold accents, punctuation and generational suffixes so two sources' spellings meet.

    NFL rosters make this load-bearing in a way MLB's do not: a single team can carry two
    players who differ only by suffix, and the sources disagree about whether the suffix is
    part of the name at all. Suffixes are stripped here for matching, which means a caller
    holding both "Odell Beckham Jr." and "Odell Beckham" resolves them together -- correct
    for the overwhelmingly common case, and the reason any real matcher should fall back to
    a team-scoped comparison rather than trusting this alone.
    """
    text = unicodedata.normalize("NFKD", str(name or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("'", "").replace("`", "")
    text = re.sub(r"[.\-]", " ", text)
    parts = [p for p in re.split(r"\s+", text) if p and p not in SUFFIXES]
    return " ".join(parts)


# --- the DK export ---------------------------------------------------------------------------
#
# The other half this module's docstring said was missing. `dfs.salaries` owns finding and
# parsing the DK file on the MLB side; this is the football equivalent, and it is deliberately
# thin -- one function that turns a DK export into the frame `nfl.optimizer` already expects,
# so nothing downstream has to know DK's column names.

# DK marks availability in `Status`. OUT and IR are settled facts; Q and D are not, and a
# questionable player who plays is often the best value on the board -- so only the settled
# ones are excluded by default and the rest are surfaced for a human to decide.
OUT_STATUSES = frozenset({"OUT", "IR", "O", "NA", "SUSP", "PUP"})
DOUBTFUL_STATUSES = frozenset({"Q", "D", "QUESTIONABLE", "DOUBTFUL"})

DK_COLUMNS = ("Name", "ID", "Roster Position", "Position", "Salary", "TeamAbbrev",
              "Game Info")


def load_dk_export(path, drop_out=True):
    """A DK NFL salary export as the frame the optimizer takes.

    Returns `(players, report)`. `players` carries **Name, DK Pos, Pos, Team, Opp, Game,
    Salary, DK ID, Status, Type** -- `DK Pos` straight from DK's `Roster Position`, which
    already spells FLEX eligibility as "RB/FLEX" exactly the way `eligible_positions` reads
    it. `report` counts what was dropped and who is questionable.

    **`Opp` and `Game` are parsed, not assumed.** They come out of the `Game Info` cell,
    which is the only place the export states them; deriving an opponent any other way on a
    thirteen-game slate means guessing.
    """
    import pandas as pd

    from nfl.naming import parse_game_info

    frame = pd.read_csv(path)
    missing = [c for c in DK_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing DK columns {missing} -- is it a salary export?")

    games, opponents = [], []
    teams = frame["TeamAbbrev"].map(canon_team)
    for info, team in zip(frame["Game Info"], teams):
        matchup, _ = parse_game_info(info)
        if not matchup:
            games.append("")
            opponents.append("")
            continue
        away, _, home = matchup.partition("@")
        away, home = canon_team(away), canon_team(home)
        games.append(f"{away}@{home}")
        opponents.append(home if team == away else away)

    status = frame.reindex(columns=["Status"])["Status"].astype("string").fillna("")
    status = status.str.strip().str.upper()
    out = pd.DataFrame({
        "Name": frame["Name"].astype("string").str.strip(),
        "DK Pos": frame["Roster Position"].astype("string").str.strip(),
        "Pos": frame["Position"].astype("string").str.strip().map(canon_position),
        "Team": teams,
        "Opp": opponents,
        "Game": games,
        "Salary": pd.to_numeric(frame["Salary"], errors="coerce"),
        "DK ID": frame["ID"],
        "Status": status,
        "AvgPts": pd.to_numeric(frame.reindex(columns=["AvgPointsPerGame"])["AvgPointsPerGame"],
                                errors="coerce"),
    })
    out["Type"] = out["Pos"].map(lambda p: "DST" if p == "DST" else "OFF")

    ruled_out = out["Status"].isin(OUT_STATUSES)
    report = {
        "total": int(len(out)),
        "out": sorted(out.loc[ruled_out, "Name"].tolist()),
        "questionable": sorted(out.loc[out["Status"].isin(DOUBTFUL_STATUSES), "Name"].tolist()),
        "games": sorted({g for g in games if g}),
        "no_game_info": int(sum(1 for g in games if not g)),
    }
    if drop_out:
        out = out[~ruled_out].reset_index(drop=True)
    return out, report
