"""Read DraftKings contest-standings exports: real ownership, and what it took to win.

A standings export is two tables glued side by side. The left is one row per entry
(Rank, EntryId, EntryName, Points, Lineup); the right is one row per player
(Player, Roster Position, %Drafted, FPTS), padded with blanks to the taller table's length.
Both are needed: `%Drafted` is the only *measured* ownership there is, and the score
distribution is what says whether a lineup would have cashed.

Files are named by contest id, which says nothing about the slate, so they are identified
the same way everything else here is -- by contents. Which night a file belongs to is
settled first, against the DK salary exports (`contest_night`); only then is it scored
against the slate in hand. Doing it in that order matters -- see `match_contest`.
"""

import functools
import glob
import os

import pandas as pd

from .naming import label_slates
from .salaries import SALARY_DIRS, list_salary_files, normalize_name

RESULTS_DIR = "dk_results"

# Below this share of a file's players appearing on a slate, the match is not trustworthy
# enough to attach ownership from. Overlap is rarely 100% even when correct -- a slate whose
# games were not all cached is missing real players -- but a wrong slate scores far lower.
MIN_OVERLAP = 0.45

# A contest's players must nearly all appear in an export for that export to be the slate it
# was run on. Not 1.0: DK occasionally lists a player in standings under a spelling the
# export does not use, and a late scratch can be removed from one and not the other.
MIN_CONTAINMENT = 0.95


def read_contest(path):
    """Parse one standings export into ownership, scores and contest shape."""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [str(c).strip() for c in frame.columns]

    result = {"path": path, "ownership": {}, "fpts": {}, "entries": 0,
              "scores": pd.Series(dtype=float), "players": 0}

    if {"Player", "%Drafted"} <= set(frame.columns):
        block = frame[["Player", "Roster Position", "%Drafted", "FPTS"]].copy()
        block = block[block["Player"].astype(str).str.strip().ne("")].dropna(subset=["Player"])
        pct = pd.to_numeric(block["%Drafted"].astype(str).str.rstrip("%").str.strip(),
                            errors="coerce")
        keys = block["Player"].map(normalize_name)
        result["ownership"] = {k: float(v) for k, v in zip(keys, pct) if pd.notna(v)}
        fpts = pd.to_numeric(block["FPTS"], errors="coerce")
        result["fpts"] = {k: float(v) for k, v in zip(keys, fpts) if pd.notna(v)}
        result["players"] = len(result["ownership"])

    if "Points" in frame.columns:
        scores = pd.to_numeric(frame["Points"], errors="coerce").dropna()
        result["scores"] = scores
        result["entries"] = int(len(scores))
    return result


def list_contests(directory=RESULTS_DIR):
    """Every parsed standings export on disk."""
    found = []
    for path in sorted(glob.glob(os.path.join(directory, "*.csv"))):
        try:
            contest = read_contest(path)
        except Exception:
            continue
        if contest["ownership"]:
            found.append(contest)
    return found


def _export_stamp():
    """Cheap fingerprint of the export folders, so the index below can expire itself.

    `dfs.autosnap` runs as a loop for a whole evening and downloads land in the middle of it,
    so a cache keyed on nothing would answer the rest of the night from a snapshot of the
    folder taken before tonight's export existed. Globbing names and mtimes costs nothing;
    it is parsing every file that is worth caching.
    """
    stamp = []
    for directory in SALARY_DIRS:
        for path in sorted(glob.glob(os.path.join(directory, "*.csv"))):
            try:
                stamp.append((path, os.path.getmtime(path)))
            except OSError:
                continue
    return tuple(stamp)


@functools.lru_cache(maxsize=2)
def _export_index_for(_stamp):
    """Every DK salary export as (date, slate label, player names, size), read once.

    The exports are the only thing on disk that states which games a slate ran, and they
    are cheap to read, so they -- not the projected boards -- are what a contest is dated
    against. Never call `build_slate` from here: `build_slate` is the caller.
    """
    from collections import defaultdict

    infos = list_salary_files()
    by_date = defaultdict(list)
    for info in infos:
        by_date[str(info.get("date"))].append(info)
    labels = {}
    for date, group in by_date.items():
        labels.update(label_slates(group, date))

    index = []
    for info in infos:
        try:
            frame = pd.read_csv(info["path"], usecols=["Name"])
        except Exception:
            continue
        names = set(frame["Name"].map(normalize_name))
        if not names or not info.get("date"):
            continue
        index.append({"date": str(info["date"]), "slate": labels.get(info["path"], ""),
                      "names": names, "size": len(names)})
    return tuple(index)


def _export_index():
    return _export_index_for(_export_stamp())


def contest_night(contest, index=None):
    """(date, slate label) the contest was run on, or (None, None).

    A contest's players are a subset of its own slate's export, and consecutive days share
    most of their rosters, so "which export contains these players" is satisfied by several
    days at once. The one that identifies the slate is the *tightest* -- the smallest export
    that still contains them. On 2026-08-06 the 4-game main slate (370 players) and the
    11-game 2026-08-04 export both contain the contest completely; only the smaller one is
    the slate it was actually run on.
    """
    keys = set(contest.get("ownership") or ())
    if not keys:
        return None, None
    best = None
    for entry in (index if index is not None else _export_index()):
        containment = len(keys & entry["names"]) / len(keys)
        if containment < MIN_CONTAINMENT:
            continue
        # Tightest containing export wins; a fuller one breaks a tie on the same size.
        rank = (containment, -entry["size"])
        if best is None or rank > best[0]:
            best = (rank, entry)
    if best is None:
        return None, None
    return best[1]["date"], best[1]["slate"]


def match_contest(players, directory=RESULTS_DIR, contests=None, date=None, slate=None):
    """The standings export for this slate, or None. Returns (contest, overlap).

    Overlap is scored **both ways** -- the share of the file's players on the slate and the
    share of the slate's players in the file, whichever is worse. One-way overlap rewards
    small files: a 235-entry export listing 80 players is a near-subset of almost any slate
    and so used to beat the night's own 951-entry file, attaching another night's ownership
    to a review that then read as a critique of the model.

    Two-way overlap fixes that but cannot separate consecutive days, which genuinely share
    players -- 2026-08-08's turbo scored 0.72 against 2026-08-07's turbo contest. So when
    the night is known, each candidate is dated first and anything from another night is
    dropped before scoring. Among what survives the largest field wins, because %Drafted
    measured over 5,945 entries is a better number than the same column over 235.
    """
    if players is None or players.empty:
        return None, 0.0
    names = set(players["Name"].map(normalize_name))
    if not names:
        return None, 0.0

    candidates = list(contests if contests is not None else list_contests(directory))
    filtered = False
    if date:
        wanted = str(date)
        dated = []
        for contest in candidates:
            contest_date, contest_slate = contest_night(contest)
            if contest_date != wanted:
                continue
            # An unlabelled slate cannot contradict anything, so it only filters on date.
            if slate and contest_slate and contest_slate != str(slate):
                continue
            dated.append(contest)
        candidates, filtered = dated, True

    scored = []
    for contest in candidates:
        keys = set(contest["ownership"])
        if not keys:
            continue
        overlap = min(len(keys & names) / len(keys), len(keys & names) / len(names))
        if overlap >= MIN_OVERLAP:
            scored.append((overlap, contest))
    if not scored:
        loose = max((min(len(set(c["ownership"]) & names) / max(len(c["ownership"]), 1),
                         len(set(c["ownership"]) & names) / len(names))
                     for c in candidates), default=0.0)
        return None, loose

    if filtered:
        # Everything here is already the same night and slate, so the differences in overlap
        # are differences in contest size, not in correctness -- a 235-entry export lists
        # fewer players than a 5,945-entry one and scores higher for it. Field size is what
        # decides, since that is what %Drafted is measured over.
        best = max((c for _, c in scored), key=lambda c: c["entries"])
    else:
        best_overlap = max(overlap for overlap, _ in scored)
        best = max((c for overlap, c in scored if overlap >= best_overlap - 0.10),
                   key=lambda c: c["entries"])
    return best, min(len(set(best["ownership"]) & names) / len(best["ownership"]),
                     len(set(best["ownership"]) & names) / len(names))


def contest_summary(contest):
    """One line describing what it took to win and to cash."""
    scores = contest["scores"]
    if scores.empty:
        return f"{contest['players']} players priced"
    return (f"{contest['entries']:,} entries — winner {scores.max():.1f}, "
            f"top 1% {scores.quantile(0.99):.1f}, top 20% {scores.quantile(0.80):.1f}, "
            f"median {scores.median():.1f}")


def cash_line(contest, fraction=0.20):
    """Score at the given finishing percentile, the bar a lineup actually had to clear."""
    scores = contest["scores"]
    if scores.empty:
        return None
    return float(scores.quantile(1.0 - fraction))


def main():
    import argparse
    from .slate import build_slate

    parser = argparse.ArgumentParser(
        description="Match DK contest-standings exports to slates and show real ownership.")
    parser.add_argument("--dir", default=RESULTS_DIR)
    parser.add_argument("--date")
    parser.add_argument("--slate")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    contests = list_contests(args.dir)
    if not contests:
        print(f"No contest-standings exports in {args.dir}/.")
        return

    if not args.date:
        print(f"{len(contests)} contest export(s) in {args.dir}/:")
        for contest in contests:
            print(f"  {os.path.basename(contest['path'])}: {contest_summary(contest)}")
        print("\nPass --date (and --slate) to match one to a slate.")
        return

    players, _, meta = build_slate(args.date, slate=args.slate)
    contest, overlap = match_contest(players, contests=contests)
    if contest is None:
        print(f"No contest export matches {args.date} {args.slate or ''} "
              f"(best overlap {overlap*100:.0f}%).")
        return

    print(f"{os.path.basename(contest['path'])} -> {args.date} {args.slate or ''} "
          f"({overlap*100:.0f}% overlap)")
    print(f"  {contest_summary(contest)}")
    owned = sorted(contest["ownership"].items(), key=lambda kv: -kv[1])[:args.top]
    print(f"\n  highest actual ownership:")
    for key, pct in owned:
        print(f"    {key:<26} {pct:5.1f}%   {contest['fpts'].get(key, float('nan')):6.1f} pts")


if __name__ == "__main__":
    main()
