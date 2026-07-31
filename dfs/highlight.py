"""Player-name highlighting for the scouting report, on two independent axes.

The report's own heat coloring runs red -> amber -> green on league percentiles. Both
highlight families deliberately avoid that range, so a tinted name reads as a different
kind of statement ("this is a DFS play") rather than as another percentile.

Two axes, because points-per-dollar alone is a cheap-player detector: on 2026-07-27 it
tinted names averaging $3,150 and missed 7 of the 10 best-projected hitters outright.

  Value (indigo) -- points per $1k. The efficiency play that frees up salary.
  Stud  (teal)   -- raw projection regardless of price. Expensive, but the bat you are
                    building around.
  Both  (violet) -- high projection AND priced well. The rare ones.

Only names meeting a threshold are tinted; colouring everyone would make it worthless.
"""

import re
import unicodedata

# Thresholds are deliberately high. With two axes the tinted set is their union, so a
# loose cut colours most of a lineup and stops meaning anything -- at a 55th-percentile
# value cut, 60% of a priced slate lit up. These land nearer a third.
#
# Indigo tints, light to strong. Distinct from the report's red/amber/green heat scale.
VALUE_TIERS = [
    ("Elite Value", 92.0, (126, 152, 224)),
    ("Value", 80.0, (178, 195, 238)),
]

# Teal tints for raw projection, keyed off Proj Pct rather than Value Pct.
STUD_TIERS = [
    ("Stud", 92.0, (104, 190, 186)),
    ("Strong Bat", 80.0, (168, 216, 214)),
]

# Both axes at their top tier -- worth its own colour, since these are the anchors.
BOTH_TIER = ("Stud + Value", (150, 124, 206))

# Columns whose cells hold a player name.
NAME_COLUMNS = {"Name", "Player", "Hitter", "Pitcher", "Catcher", "Starter", "Reliever"}

_INDEX = {}


def _key(name):
    text = unicodedata.normalize("NFKD", str(name or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("'", "").replace("`", "")
    text = re.sub(r"[.\-]", " ", text)
    parts = [p for p in re.split(r"\s+", text) if p and p not in {"jr", "sr", "ii", "iii", "iv", "v"}]
    return " ".join(parts)


def _pct(row, column):
    try:
        value = float(row.get(column))
    except (TypeError, ValueError):
        return None
    return None if value != value else value      # reject NaN


def build_index(players):
    """Map normalized player name -> (tier, rgb, note) from a slate frame.

    Percentiles are taken as computed on the slate, which ranks pitchers and hitters
    separately -- neither points-per-dollar nor raw projection compares across the two.
    """
    index = {}
    if players is None or players.empty:
        return index
    if "Value Pct" not in players.columns and "Proj Pct" not in players.columns:
        return index

    for _, row in players.iterrows():
        value_pct = _pct(row, "Value Pct")
        proj_pct = _pct(row, "Proj Pct")

        value_tier = next(((t, rgb) for t, threshold, rgb in VALUE_TIERS
                           if value_pct is not None and value_pct >= threshold), None)
        stud_tier = next(((t, rgb) for t, threshold, rgb in STUD_TIERS
                          if proj_pct is not None and proj_pct >= threshold), None)

        # A player who is both a top bat and priced well gets its own colour rather than
        # being filed under whichever axis happened to be checked first.
        if value_tier and stud_tier and value_tier[0] == "Elite Value" and stud_tier[0] == "Stud":
            tier, rgb = BOTH_TIER
        elif stud_tier:
            tier, rgb = stud_tier
        elif value_tier:
            tier, rgb = value_tier
        else:
            continue

        note = []
        if proj_pct is not None:
            note.append(f"{row.get('Proj')} proj")
        if row.get("Value") is not None and row.get("Value") == row.get("Value"):
            note.append(f"{row.get('Value')}/$1k")
        index[_key(row.get("Name"))] = (tier, rgb, ", ".join(note))
    return index


def resolve_slate(date, games=None, salary_path=None):
    """(slate label, explanation) for the export a report should be highlighted from.

    A date with two exports has no single answer, and the old behaviour -- build the slate
    unqualified, get nothing, tint nothing -- looked exactly like "these players aren't
    worth highlighting". When the report's own games appear in only one export, that export
    is the answer and no one should have to say so; otherwise the candidates are named.
    """
    if salary_path:
        return None, None
    try:
        from .naming import slate_for
        from .salaries import describe_slate, list_salary_files
    except Exception:
        return None, None

    candidates = list_salary_files(date)
    if len(candidates) <= 1:
        return None, None

    wanted = {str(g).strip().upper() for g in (games or []) if g}
    if wanted:
        matched = [c for c in candidates if wanted <= {g.upper() for g in (c.get("games") or [])}]
        if len(matched) == 1:
            label = slate_for(matched[0], date=date)
            return label, f"matched to the '{label}' export, the only one pricing {', '.join(sorted(wanted))}"

    names = ", ".join(c["name"] for c in candidates)
    return None, (f"{len(candidates)} DK exports match {date} ({names}); "
                  f"pass --dfs-slate to pick one")


def load_for_date(date, salary_path=None, slate=None, games=None):
    """Build the index for one slate date. Returns (index, note).

    `note` explains an empty index. Highlighting silently doing nothing is indistinguishable
    from highlighting deciding nothing qualifies, and that ambiguity cost a real debugging
    session -- so the reason always comes back, even though it is only decoration.
    """
    if slate is None and salary_path is None:
        slate, note = resolve_slate(date, games=games)
        if slate is None and note:
            return {}, note
    try:
        from .slate import build_slate
        players, _, meta = build_slate(date, salary_path=salary_path, slate=slate)
        if not meta.get("has_salary"):
            if meta.get("ambiguous_slates"):
                names = ", ".join(c["name"] for c in meta["ambiguous_slates"])
                return {}, f"several DK exports match {date} ({names}); pass --dfs-slate"
            return {}, f"no priced DK salary file found for {date}"
        index = build_index(players)
        if not index:
            return {}, f"the {slate or 'matched'} slate priced no player above the tint thresholds"
        return index, None
    except Exception as error:
        # Highlighting is decoration. It must never be able to break report generation --
        # but the reason it gave up is still worth surfacing.
        return {}, f"slate could not be built: {error}"


def activate(date, salary_path=None, slate=None, games=None):
    """Load the index into module state so report renderers can consult it.

    Returns (index, note); the note is non-empty only when nothing was loaded.
    """
    global _INDEX
    _INDEX, note = load_for_date(date, salary_path=salary_path, slate=slate, games=games)
    return _INDEX, note


def set_index(index):
    global _INDEX
    _INDEX = index or {}


def lookup(column, value):
    """(tier, rgb, value) for a name cell, or None.

    Returns None unless the column actually holds player names, so a stray string match
    in a notes column can't tint an unrelated cell.
    """
    if not _INDEX or column not in NAME_COLUMNS:
        return None
    return _INDEX.get(_key(value))


def active():
    return bool(_INDEX)


def tier_colors():
    """[(tier, rgb), ...] in legend order: studs, value, then the combined tier."""
    return ([BOTH_TIER]
            + [(t, rgb) for t, _, rgb in STUD_TIERS]
            + [(t, rgb) for t, _, rgb in VALUE_TIERS])


def legend():
    """Human-readable legend for whatever tiers are in play."""
    if not _INDEX:
        return ""
    counts = {}
    for tier, _, _ in _INDEX.values():
        counts[tier] = counts.get(tier, 0) + 1
    parts = [f"{tier} {counts[tier]}" for tier, _ in tier_colors() if tier in counts]
    return ("DFS: teal = raw projection, indigo = pts per $1k, violet = both; "
            "ranked within pitchers/hitters. " + ", ".join(parts))
