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
