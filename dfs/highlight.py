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

import pandas as pd

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

# --------------------------------------------------------------------------------------
# Ownership, as a third axis that deliberately does NOT use colour.
#
# Two colour families already tint about a third of a priced slate between them, so a third
# one would turn the report into a paint chart -- and projected ownership is wanted for
# *every* player, not just the notable ones, which no colour scheme survives. So it rides on
# a separate channel: one trailing character on the name.
#
# Fill fraction, not bar height. The first attempt used the eighth-blocks (▁▂▃▄▅▆▇█) and they
# were unreadable at 9pt: eight levels differing only in height, all sitting on the baseline,
# so two adjacent players looked identical unless you compared them side by side. These change
# *shape* rather than height, which the eye resolves at a glance and out of order -- ◔ against
# ◕ is obvious in a way that ▄ against ▅ is not. Five levels instead of eight, because
# ownership bands are what get acted on and eight was false precision.
OWN_RAMP = [
    (5.0, "○"),      # empty circle      -- under 5%
    (12.0, "◔"),     # quarter filled    -- 5-12%
    (22.0, "◑"),     # half filled       -- 12-22%
    (35.0, "◕"),     # three-quarters    -- 22-35%
    (101.0, "●"),    # full circle       -- 35%+
]

# A player the field is ignoring is only interesting if he is also worth playing -- otherwise
# the mark lands on every min-priced scrub on the slate, which is most of the low-owned pool
# and none of the useful part. So the leverage mark needs BOTH: low ownership and a projection
# well up the board for his own player type.
#
# It replaces the ramp glyph rather than adding to it, because the whole point of a diamond in
# a column of bars is that it breaks the pattern.
#
# The projection gate is a top-quartile cut, not a median one, and that is load-bearing. The
# field only has 1,000 points of ownership to spend across a slate, so on a 120-player board
# the *average* player is around 8% owned and "under 10%" is very nearly the median -- a
# median-projection gate flagged 30% of the slate, which is not a highlight. At the top
# quartile it lands near a tenth, comparable to the two tint families.
#
# Two tiers, mirroring Elite Value / Value: filled for the genuinely ignored, hollow for the
# merely unpopular.
#
# Stars rather than diamonds so the leverage flag cannot be confused with the ownership ramp.
# Diamonds (◆ ◇) read as members of the same family as the circles at small sizes -- same
# weight, same footprint -- and the flag needs to be a different *kind* of mark, not a
# different value on the same scale.
LEVERAGE_TIERS = [
    ("Leverage", 5.0, "★"),        # filled star: under 5% owned
    ("Contrarian", 10.0, "☆"),     # hollow star: 5-10% owned
]
LEVERAGE_MIN_PROJ_PCT = 75.0
LEVERAGE_MARKS = {glyph for _label, _cut, glyph in LEVERAGE_TIERS}

# Columns whose cells hold a player name.
NAME_COLUMNS = {"Name", "Player", "Hitter", "Pitcher", "Catcher", "Starter", "Reliever"}

_INDEX = {}
# Ownership markers for EVERY priced player, keyed the same way as _INDEX. Kept separate
# because _INDEX only holds players that cleared a tint threshold, and this axis covers all.
_OWN = {}
# The priced slate itself, kept so a report can render a table off it rather than only tint
# names. The two index dicts are lookups by name and cannot answer "which players qualify".
_FRAME = None


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


def own_mark(own_pct, proj_pct):
    """The one-character ownership mark for a player, or "" when ownership is unknown.

    Returns a leverage diamond when the field is off him and the model is not, otherwise a
    block from the ramp. `proj_pct` gates the diamond so it cannot land on a cheap scrub who
    is unowned for the obvious reason.
    """
    if own_pct is None:
        return ""
    if proj_pct is not None and proj_pct >= LEVERAGE_MIN_PROJ_PCT:
        for _label, cut, glyph in LEVERAGE_TIERS:
            if own_pct <= cut:
                return glyph
    for ceiling, glyph in OWN_RAMP:
        if own_pct < ceiling:
            return glyph
    return OWN_RAMP[-1][1]


def build_own_index(players):
    """{normalized name: (mark, own_pct, is_leverage)} for every priced player on the slate."""
    out = {}
    if players is None or players.empty or "Own%" not in players.columns:
        return out
    for _, row in players.iterrows():
        own = _pct(row, "Own%")
        if own is None:
            continue
        proj_pct = _pct(row, "Proj Pct")
        mark = own_mark(own, proj_pct)
        if not mark:
            continue
        out[_key(row.get("Name"))] = (mark, own, mark in LEVERAGE_MARKS)
    return out


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
        own = _pct(row, "Own%")
        if own is not None:
            note.append(f"{own:.0f}% owned")
        index[_key(row.get("Name"))] = (tier, rgb, ", ".join(note))
    return index


def leverage_table(players, games=None, own_max=None, min_proj_pct=None, limit=10):
    """The sub-5%-owned players worth a second look, with the reason they qualify.

    Low ownership on its own is not a finding -- most of a slate's cheap end is unowned for
    the obvious reason -- so this is gated on projection percentile exactly as the star mark
    is, and sorted by it. The "why" is taken from the board's own `Why` / `Risks` text, which
    is already assembled from the named factors the projection used: park and wind, opposing
    starter FIP, platoon split, batting order, arsenal fit. Reusing it keeps one explanation
    of a player rather than inventing a second.

    `games` restricts to one matchup, since a report covers a game and the slate covers a
    night. Returns a DataFrame ready to render, or an empty one.
    """
    columns = ["Name", "Team", "Pos", "Salary", "Proj", "Ceiling", "Own%", "Why"]
    if players is None or players.empty or "Own%" not in players.columns:
        return pd.DataFrame(columns=columns)

    own_max = OWN_RAMP[0][0] if own_max is None else own_max
    min_proj_pct = LEVERAGE_MIN_PROJ_PCT if min_proj_pct is None else min_proj_pct

    frame = players.copy()
    for column in ("Own%", "Proj", "Ceiling", "Salary", "Proj Pct"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if games:
        wanted = {str(g).strip().upper() for g in games if g}
        if "Game" in frame.columns:
            frame = frame[frame["Game"].astype(str).str.upper().isin(wanted)]
        elif "Team" in frame.columns:
            # No Game column: fall back to the team codes named in the matchup strings.
            teams = {t for g in wanted for t in re.split(r"[@/]", g) if t}
            frame = frame[frame["Team"].astype(str).str.upper().isin(teams)]

    keep = frame["Own%"].notna() & (frame["Own%"] < own_max)
    if "Proj Pct" in frame.columns:
        keep &= frame["Proj Pct"].fillna(0) >= min_proj_pct
    frame = frame[keep]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    if "Why" not in frame.columns:
        frame["Why"] = ""
    if "Pos" not in frame.columns:
        frame["Pos"] = frame.get("DK Pos", "")
    frame = frame.sort_values(
        ["Proj Pct" if "Proj Pct" in frame.columns else "Proj"], ascending=False)
    return frame.reindex(columns=columns).head(limit).reset_index(drop=True)


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
            label = slate_for(matched[0], date=date, peers=candidates)
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
        own = build_own_index(players)
        if not index and not own:
            return {}, f"the {slate or 'matched'} slate priced no player above the tint thresholds"
        return {"tints": index, "own": own, "frame": players}, None
    except Exception as error:
        # Highlighting is decoration. It must never be able to break report generation --
        # but the reason it gave up is still worth surfacing.
        return {}, f"slate could not be built: {error}"


def activate(date, salary_path=None, slate=None, games=None):
    """Load the index into module state so report renderers can consult it.

    Returns (index, note); the note is non-empty only when nothing was loaded.
    """
    global _INDEX, _OWN, _FRAME
    loaded, note = load_for_date(date, salary_path=salary_path, slate=slate, games=games)
    # `load_for_date` used to return the tint index alone; it now returns every axis. Older
    # shapes are still accepted so a caller that stored one can hand it straight back.
    if isinstance(loaded, dict) and "tints" in loaded:
        _INDEX = loaded.get("tints") or {}
        _OWN = loaded.get("own") or {}
        _FRAME = loaded.get("frame")
    else:
        _INDEX, _OWN, _FRAME = loaded or {}, {}, None
    return _INDEX, note


def set_index(index, own=None, frame=None):
    global _INDEX, _OWN, _FRAME
    if isinstance(index, dict) and "tints" in index:
        _INDEX = index.get("tints") or {}
        _OWN = index.get("own") or {}
        _FRAME = index.get("frame")
        return
    _INDEX = index or {}
    _OWN = own if own is not None else {}
    _FRAME = frame


def slate_frame():
    """The priced slate the current highlight index was built from, or None."""
    return _FRAME


def lookup(column, value):
    """(tier, rgb, value) for a name cell, or None.

    Returns None unless the column actually holds player names, so a stray string match
    in a notes column can't tint an unrelated cell.
    """
    if not _INDEX or column not in NAME_COLUMNS:
        return None
    return _INDEX.get(_key(value))


def own_lookup(column, value):
    """(mark, own_pct, is_leverage) for a name cell, or None.

    Separate from `lookup` because this axis covers every priced player rather than only the
    ones worth tinting, and because it is consumed differently -- a character appended to the
    displayed text rather than a cell colour.
    """
    if not _OWN or column not in NAME_COLUMNS:
        return None
    return _OWN.get(_key(value))


def active():
    return bool(_INDEX) or bool(_OWN)


def tier_colors():
    """[(tier, rgb), ...] in legend order: studs, value, then the combined tier."""
    return ([BOTH_TIER]
            + [(t, rgb) for t, _, rgb in STUD_TIERS]
            + [(t, rgb) for t, _, rgb in VALUE_TIERS])


def legend():
    """Human-readable legend for whatever tiers and marks are in play."""
    parts = []
    if _INDEX:
        counts = {}
        for tier, _, _ in _INDEX.values():
            counts[tier] = counts.get(tier, 0) + 1
        named = ", ".join(f"{tier} {counts[tier]}" for tier, _ in tier_colors()
                          if tier in counts)
        parts.append("DFS: teal = raw projection, indigo = pts per $1k, violet = both; "
                     "ranked within pitchers/hitters. " + named)
    if _OWN:
        counts = {}
        for mark, _own, _is_lev in _OWN.values():
            counts[mark] = counts.get(mark, 0) + 1
        tiers = ", ".join(
            f"{glyph} {label} (<={cut:.0f}% owned) {counts.get(glyph, 0)}"
            for label, cut, glyph in LEVERAGE_TIERS)
        parts.append(
            f"Ownership: {OWN_RAMP[0][1]}..{OWN_RAMP[-1][1]} = fill fraction, projected "
            f"under {OWN_RAMP[0][0]:.0f}% .. {OWN_RAMP[-2][0]:.0f}%+ of the field. "
            f"Stars also need a top-{100 - LEVERAGE_MIN_PROJ_PCT:.0f}% projection: {tiers}.")
    return " ".join(parts)
