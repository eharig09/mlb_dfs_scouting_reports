"""PFF export access: which file is which season, and how a PFF row meets an nflverse one.

Two problems stand between a folder of PFF CSVs and a usable frame, and both of them fail
*silently* -- they produce a join that runs clean and matches almost nothing.

**A PFF filename carries no season.** The exports arrive as `receiving_summary (4).csv`,
and the numbering is just download order: it differs per family, it is not chronological,
and the unnumbered file is the newest in one family and the oldest in another. Guessing
from the number is how a board once got 2018 grades onto a 2025 slate and matched 15 of
126 players. So the season is **identified, never read** -- see `identify_season`.

**Names are the wrong join key.** nflverse rosters publish a `pff_id` column, which is the
same id space as the `player_id` in every PFF export, so the two sources can be joined
exactly. Coverage is 97-99% on skill positions; it is only ~66% on the offensive line,
where nflverse has not filled the column in. Name-and-team matching is therefore kept as a
*fallback*, not the primary path, and `key_source` records which one answered.

    from nfl.pffdata import catalog, load
    catalog()                        # every file, with its identified season
    load("receiving_summary", 2025)  # tidy, keyed, canonical team codes

Everything here is read-only and cached; nothing writes into the export folders.
"""

import glob
import json
import os

import numpy as np
import pandas as pd

from nfl import data as nfl_data
from nfl.salaries import canon_position, canon_team, normalize_name

# Where exports are dropped. The curated per-report folders under `nfl/pff/` are scanned
# first, ahead of the loose files at its root and the original `pff_coverage_data/` drop.
PFF_ROOTS = ("nfl/pff", "nfl/Projections", "pff_coverage_data")

CACHE_DIR = os.path.join(".cache", "pff")
CATALOG_CACHE = os.path.join(CACHE_DIR, "catalog.json")
CROSSWALK_CACHE = os.path.join(CACHE_DIR, "crosswalk.parquet")

# Seasons the fingerprint may return. Rosters exist earlier, but nothing in this project
# reaches back past 2017 and every extra season is a download.
FINGERPRINT_SEASONS = tuple(range(2017, 2026))

# A file whose best season scores below this is reported as unidentified rather than
# guessed at. True seasons land at 0.97-0.99; the runner-up sits near 0.60, because roughly
# a quarter of the league changes clubs every off-season and that is what separates them.
MIN_CONFIDENCE = 0.80
# ...and the lead over the runner-up must be this wide. Confidence alone is not enough: a
# file covering a handful of players could score 1.00 on two adjacent seasons by luck.
MIN_MARGIN = 0.15

# Families that are *not* per-player stat exports. They carry a different schema (no
# `player_id`, no `team_name`), so they are catalogued but never fingerprinted.
NON_PLAYER_FAMILIES = frozenset({"projections", "projection-set-preseason-all-2026"})

# The identity columns a player-level export opens with. PFF's "fantasy stats" reports come
# off a different endpoint and spell them differently -- and carry no player id at all, so
# they fingerprint and join by name, the fallback path.
IDENTITY = ("player", "player_id", "position", "team_name", "player_game_count")
SCHEMA_ALIASES = {
    "fantasy-stats-receiving": {"team": "team_name", "games": "player_game_count"},
    "fantasy-stats-passing": {"team": "team_name", "games": "player_game_count"},
}
# Of those, the ones a file must have before it can be identified or joined at all.
REQUIRED_COLUMNS = ("player", "team_name")


class PffDataError(Exception):
    """No file for that family and season, or the file present is unusable."""


# Built catalogs and roster fingerprints, per process. `resolve` is called once per family
# per report, and rebuilding the table each time re-stats ~90 files for no new information.
_CATALOG_MEMO = {}
_FINGERPRINT_MEMO = {}


def _family(path):
    """`Receiving Grades/receiving_summary (15).csv` -> `receiving_summary`."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.split(" (")[0].strip()


def _key(path):
    """The catalog's cache key for a path.

    **Normalised, and that is load-bearing.** A caller passing `root="."` builds the root as
    `./nfl/pff`, which globs to `./nfl/pff/x.csv` -- a different *string* from the cached
    `nfl/pff/x.csv` for the very same file. Unnormalised, every such call missed the cache
    and re-fingerprinted all ~90 exports, several of which are 500 columns wide. It turned a
    sub-second lookup into minutes and hung the test suite.
    """
    return os.path.normpath(path).replace("\\", "/")


def _discover(roots=PFF_ROOTS):
    """Every CSV under the export roots, de-duplicated by real path."""
    seen, out = set(), []
    for root in roots:
        pattern = os.path.join(root, "**", "*.csv")
        for path in sorted(glob.glob(pattern, recursive=True)):
            real = os.path.realpath(path)
            if real not in seen:
                seen.add(real)
                out.append(_key(path))
    return out


def _canonical_columns(frame, family):
    """Rename a family's identity columns to the canonical spellings, leaving the rest be."""
    aliases = SCHEMA_ALIASES.get(family)
    if not aliases:
        return frame
    rename = {src: dst for src, dst in aliases.items()
              if src in frame.columns and dst not in frame.columns}
    return frame.rename(columns=rename) if rename else frame


def _stamp(path):
    """File identity for the catalog cache: size and mtime, so re-downloading the same
    filename with different contents re-fingerprints instead of reusing a stale season."""
    stat = os.stat(path)
    return "%d:%d" % (stat.st_size, int(stat.st_mtime))


# --- the id crosswalk ---------------------------------------------------------------------


def crosswalk(force=False):
    """PFF `player_id` -> nflverse `gsis_id`, pooled over every season on file.

    Pooled deliberately: the column is populated unevenly year to year, and a player who
    carries the id on one roster row carries the same id forever. Pooling took skill-position
    coverage from 50% on a single season to 97%.
    """
    if not force and os.path.exists(CROSSWALK_CACHE):
        return pd.read_parquet(CROSSWALK_CACHE)
    rosters = nfl_data.load_rosters(FINGERPRINT_SEASONS)
    frame = pd.DataFrame({
        "pff_id": pd.to_numeric(rosters.reindex(columns=["pff_id"])["pff_id"], errors="coerce"),
        "gsis_id": rosters.reindex(columns=["gsis_id"])["gsis_id"],
        "roster_name": rosters.reindex(columns=["full_name"])["full_name"],
        "season": rosters.reindex(columns=["season"])["season"],
    })
    frame = frame[frame["pff_id"].notna() & frame["gsis_id"].notna()]
    # Newest roster row per id, so the name is the spelling nflverse uses now.
    frame = frame.sort_values("season").drop_duplicates("pff_id", keep="last")
    frame["pff_id"] = frame["pff_id"].astype("int64")
    frame = frame[["pff_id", "gsis_id", "roster_name"]].reset_index(drop=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    frame.to_parquet(CROSSWALK_CACHE, index=False)
    return frame


def _roster_fingerprints():
    """season -> the (pff_id, team) and (name_key, team) pairs that season's rosters show."""
    if _FINGERPRINT_MEMO:
        return _FINGERPRINT_MEMO
    rosters = nfl_data.load_rosters(FINGERPRINT_SEASONS)
    rosters = rosters.assign(
        _pff=pd.to_numeric(rosters.reindex(columns=["pff_id"])["pff_id"], errors="coerce"),
        _team=rosters.reindex(columns=["team"])["team"].map(canon_team),
        _name=rosters.reindex(columns=["full_name"])["full_name"].map(normalize_name),
    )
    out = {}
    for season, group in rosters.groupby("season"):
        with_id = group[group["_pff"].notna()]
        out[int(season)] = (
            set(zip(with_id["_pff"].astype("int64"), with_id["_team"])),
            set(zip(group["_name"], group["_team"])),
        )
    _FINGERPRINT_MEMO.update(out)
    return out


def identify_season(frame, fingerprints=None):
    """Which season a PFF export describes, decided by *who was on which team*.

    The discriminating signal is player movement: about a quarter of the league changes
    clubs every off-season, so the roster of the true season matches nearly every row while
    its neighbours match roughly 60%. That works for any export carrying a player and a
    team -- including the offensive-line and slot-coverage reports, whose seasons an earlier
    stat-error approach could not pin at all, because it needed a counting stat that maps
    onto nflverse and neither of those two has one.

    Returns `(season, confidence, margin, scores)`. `season` is None when the winner is weak
    or its lead is narrow, which is the honest answer for a file covering too few players to
    separate two adjacent years.
    """
    fingerprints = _roster_fingerprints() if fingerprints is None else fingerprints
    pff_id = pd.to_numeric(frame.reindex(columns=["player_id"])["player_id"], errors="coerce")
    team = frame.reindex(columns=["team_name"])["team_name"].map(canon_team)
    name = frame.reindex(columns=["player"])["player"].map(normalize_name)

    has_id = pff_id.notna()
    pairs_id = set(zip(pff_id[has_id].astype("int64"), team[has_id]))
    pairs_name = set(zip(name, team))

    scores = {}
    for season, (roster_ids, roster_names) in sorted(fingerprints.items()):
        by_id = len(pairs_id & roster_ids) / len(pairs_id) if pairs_id else 0.0
        by_name = len(pairs_name & roster_names) / len(pairs_name) if pairs_name else 0.0
        # Whichever key matched better. On the offensive line that is the name; everywhere
        # else the id, which cannot be confused by two players sharing a spelling.
        scores[season] = round(max(by_id, by_name), 4)

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best, confidence = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = confidence - runner_up
    if confidence < MIN_CONFIDENCE or margin < MIN_MARGIN:
        return None, confidence, round(margin, 4), scores
    return int(best), confidence, round(margin, 4), scores


# --- the catalog --------------------------------------------------------------------------


def catalog(roots=PFF_ROOTS, force=False):
    """Every export on disk, with the season it was found to describe.

    Columns: `family, path, season, confidence, margin, rows, usable, note`.

    Cached against each file's size and mtime, so dropping in a new export re-fingerprints
    that file alone. This is the table to read when a join comes back empty: it says what
    the loader believes, which is the first thing to check against what you assumed.
    """
    memo_key = tuple(roots)
    if not force and memo_key in _CATALOG_MEMO:
        return _CATALOG_MEMO[memo_key]

    cached = {}
    if not force and os.path.exists(CATALOG_CACHE):
        with open(CATALOG_CACHE, "r", encoding="utf-8") as handle:
            cached = json.load(handle)

    fingerprints = None
    rows, updated = [], {}
    for path in _discover(roots):
        stamp = _stamp(path)
        hit = cached.get(path)
        if hit and hit.get("stamp") == stamp:
            rows.append(dict(hit, path=path))
            updated[path] = hit
            continue

        family = _family(path)
        record = {"stamp": stamp, "family": family, "season": None, "confidence": None,
                  "margin": None, "rows": 0, "usable": False, "note": ""}
        try:
            frame = pd.read_csv(path, low_memory=False)
        except Exception as exc:            # a partial download, not a bug on this side
            record["note"] = "unreadable: %s" % exc.__class__.__name__
            rows.append(dict(record, path=path))
            updated[path] = record
            continue

        frame = _canonical_columns(frame, family)
        record["rows"] = int(len(frame))
        payload = [c for c in frame.columns if c not in IDENTITY and c != "franchise_id"]
        if family in NON_PLAYER_FAMILIES:
            record["usable"] = True
            record["note"] = "projection export; season comes from its own columns"
        elif not set(REQUIRED_COLUMNS) <= set(frame.columns):
            missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
            record["note"] = "truncated export, missing " + ", ".join(missing)
        elif not payload:
            # `receiving_depth (5).csv` arrived carrying only the identity columns. It has
            # the right shape and the right players, so it fingerprints perfectly and then
            # contributes no statistics at all -- the exact failure this catalog exists to
            # make visible rather than let a downstream join absorb.
            record["note"] = "identity columns only, no statistics"
        else:
            if fingerprints is None:
                fingerprints = _roster_fingerprints()
            season, confidence, margin, _ = identify_season(frame, fingerprints)
            record.update(season=season, confidence=round(float(confidence), 4),
                          margin=float(margin), usable=season is not None)
            if season is None:
                record["note"] = "season not separable from its neighbours"
        rows.append(dict(record, path=path))
        updated[path] = record

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CATALOG_CACHE, "w", encoding="utf-8") as handle:
        json.dump(updated, handle, indent=1, sort_keys=True)

    frame = pd.DataFrame(rows, columns=["family", "path", "season", "confidence", "margin",
                                        "rows", "usable", "note"])
    frame = frame.sort_values(["family", "season"], na_position="last").reset_index(drop=True)
    _CATALOG_MEMO[memo_key] = frame
    return frame


def resolve(family, season, roots=PFF_ROOTS):
    """The one file to read for a family and season.

    A family can hold two files for the same season -- the same PFF report pulled twice,
    sometimes under different snap filters. The wider file wins, since PFF's filters only
    ever *remove* players, and the tie is broken on path so the choice is stable across runs
    rather than depending on directory order.
    """
    table = catalog(roots)
    hits = table[(table["family"] == family) & (table["season"] == season) & table["usable"]]
    if hits.empty:
        known = table.loc[(table["family"] == family) & table["season"].notna(), "season"]
        seasons = sorted(int(s) for s in known.unique())
        if not seasons:
            families = sorted(table["family"].unique())
            raise PffDataError("no PFF family '%s'; have %s" % (family, families))
        raise PffDataError("no %s export for %s; identified %s" % (family, season, seasons))
    hits = hits.sort_values(["rows", "path"], ascending=[False, True])
    return hits.iloc[0]["path"]


def latest(family, roots=PFF_ROOTS):
    """The newest file in a family that has no season to identify.

    Forward-looking exports -- projections -- are not one-per-season. A refresh *supersedes*
    its predecessor rather than describing a different year, so "newest wins" is the right
    rule and file mtime is the only ordering available: the filenames carry no date either.

    This exists because a hardcoded `projections (1).csv` kept being read after a fresher
    pull landed beside it. The two differed on 374 of 533 players.
    """
    table = catalog(roots)
    hits = table[(table["family"] == family) & table["usable"]]
    if hits.empty:
        raise PffDataError("no usable '%s' export found under %s" % (family, list(roots)))
    newest = max(hits["path"], key=lambda p: os.stat(p).st_mtime)
    return newest


# --- loading ------------------------------------------------------------------------------


def attach_keys(frame, cross=None):
    """Add the join keys every downstream module reads: `gsis_id`, `name_key`, `team`.

    `key_source` says which one answered -- `pff_id` for an exact crosswalk hit, `none` for
    a player nflverse has never listed. Downstream code that treats a name match as equal to
    an id match is one duplicate spelling away from a wrong row, and this column is how that
    stays visible.
    """
    cross = crosswalk() if cross is None else cross
    out = frame.copy()
    out["pff_id"] = pd.to_numeric(out.reindex(columns=["player_id"])["player_id"],
                                  errors="coerce").astype("Int64")
    out["team"] = out.reindex(columns=["team_name"])["team_name"].map(canon_team)
    out["name_key"] = out.reindex(columns=["player"])["player"].map(normalize_name)
    # PFF has no `RB` -- backs are `HB`. `pos` is the folded label; `position` keeps PFF's
    # own, since the line work cares about the distinction between T, G and C.
    out["pos"] = out.reindex(columns=["position"])["position"].map(canon_position)

    by_id = dict(zip(cross["pff_id"], cross["gsis_id"]))
    out["gsis_id"] = [by_id.get(int(v)) if pd.notna(v) else None for v in out["pff_id"]]
    out["key_source"] = np.where(out["gsis_id"].notna(), "pff_id", "none")
    return out


def load(family, season, roots=PFF_ROOTS, keys=True):
    """One PFF export, canonicalised and keyed.

    Team codes are folded through `nfl.salaries.canon_team` on the way in. PFF writes `HST`
    for Houston, and ARZ/BLT/CLV/LA/SD elsewhere; unfolded, those clubs vanish from a join
    without raising anything at all.
    """
    path = resolve(family, season, roots)
    frame = _canonical_columns(pd.read_csv(path, low_memory=False), family)
    out = attach_keys(frame) if keys else frame
    out.attrs["pff_path"] = path
    out.attrs["pff_season"] = season
    return out
