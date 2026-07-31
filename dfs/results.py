"""Read DraftKings contest-standings exports: real ownership, and what it took to win.

A standings export is two tables glued side by side. The left is one row per entry
(Rank, EntryId, EntryName, Points, Lineup); the right is one row per player
(Player, Roster Position, %Drafted, FPTS), padded with blanks to the taller table's length.
Both are needed: `%Drafted` is the only *measured* ownership there is, and the score
distribution is what says whether a lineup would have cashed.

Files are named by contest id, which says nothing about the slate, so they are identified
the same way everything else here is -- by contents. The player set is matched against the
cached slates and the best overlap wins.
"""

import glob
import os

import pandas as pd

from .salaries import normalize_name

RESULTS_DIR = "dk_results"

# Below this share of a file's players appearing on a slate, the match is not trustworthy
# enough to attach ownership from. Overlap is rarely 100% even when correct -- a slate whose
# games were not all cached is missing real players -- but a wrong slate scores far lower.
MIN_OVERLAP = 0.45


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


def match_contest(players, directory=RESULTS_DIR, contests=None):
    """The standings export whose players best match this slate, or None.

    Returns (contest, overlap). Overlap is the share of the file's players that appear on
    the slate -- not the reverse, because a slate missing an uncached game is still the
    right slate, while a file from a different contest shares far fewer names either way.
    """
    if players is None or players.empty:
        return None, 0.0
    names = set(players["Name"].map(normalize_name))
    best, best_overlap = None, 0.0
    for contest in (contests if contests is not None else list_contests(directory)):
        keys = set(contest["ownership"])
        if not keys:
            continue
        overlap = len(keys & names) / len(keys)
        if overlap > best_overlap:
            best, best_overlap = contest, overlap
    if best is None or best_overlap < MIN_OVERLAP:
        return None, best_overlap
    return best, best_overlap


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
