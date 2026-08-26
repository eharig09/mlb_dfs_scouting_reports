"""Reading DK contest-standings exports — the ground truth ownership and scores are fitted to.

This exists **before** anything uses it, on purpose. `nfl.ownership` cannot be calibrated
and portfolio selection cannot be built until several weeks of real NFL contest results are
on disk, and results only accumulate if something is reading and checking them from the
first week. A reader written after the fact discovers in week five that weeks one through
four were saved in the wrong format.

    python -m nfl.results                    # what is on disk, and whether it is usable

Drop exports in `nfl/nfl_dfs/dk_results/`. DraftKings names them
`contest-standings-<id>.csv`; the name is kept as-is, because the contest id in it is the
only durable link back to which contest a set of standings belongs to.

## What each export carries, and which half matters

A standings file has two blocks side by side: one row per **entry** (rank, name, points,
the lineup) and one block of per-**player** aggregates (`%Drafted`, `FPTS`). Both are read.

- **`%Drafted` is the field's realised ownership** — the number `nfl.ownership` is trying to
  predict, and the only place it can be observed.
- **The score distribution is the contest shape** — what it took to cash, what it took to
  win. Portfolio selection is a bet against that distribution, so it is worth as much as
  the ownership.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

from nfl.salaries import normalize_name

RESULTS_DIR = os.path.join("nfl", "nfl_dfs", "dk_results")

# Below this many entries a standings export is a private contest or a scraped fragment, and
# its ownership is not the public field's. Kept and flagged rather than dropped -- a small
# contest you actually entered is still worth scoring against.
FIELD_MIN_ENTRIES = 500


def read_contest(path):
    """Parse one standings export into ownership, scores and contest shape."""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [str(c).strip() for c in frame.columns]

    result = {"path": path, "name": os.path.basename(path), "ownership": {}, "fpts": {},
              "entries": 0, "scores": pd.Series(dtype=float), "players": 0,
              "contest_id": _contest_id(path), "notes": []}

    if {"Player", "%Drafted"} <= set(frame.columns):
        columns = [c for c in ("Player", "Roster Position", "%Drafted", "FPTS")
                   if c in frame.columns]
        block = frame[columns].copy()
        block = block[block["Player"].astype(str).str.strip().ne("")]
        block = block.dropna(subset=["Player"])
        drafted = pd.to_numeric(
            block["%Drafted"].astype(str).str.rstrip("%").str.strip(), errors="coerce")
        keys = block["Player"].map(normalize_name)
        result["ownership"] = {k: float(v) for k, v in zip(keys, drafted) if pd.notna(v)}
        if "FPTS" in block.columns:
            points = pd.to_numeric(block["FPTS"], errors="coerce")
            result["fpts"] = {k: float(v) for k, v in zip(keys, points) if pd.notna(v)}
        result["players"] = len(result["ownership"])
    else:
        result["notes"].append("no %Drafted block — ownership cannot be read from this file")

    if "Points" in frame.columns:
        scores = pd.to_numeric(frame["Points"], errors="coerce").dropna()
        result["scores"] = scores
        result["entries"] = int(len(scores))
    else:
        result["notes"].append("no Points column — the contest shape cannot be read")

    if result["entries"] and result["entries"] < FIELD_MIN_ENTRIES:
        result["notes"].append(
            f"only {result['entries']} entries — usable for scoring your own lineups, but "
            f"its ownership is not the public field's")
    return result


def _contest_id(path):
    """The DK contest id out of `contest-standings-<id>.csv`, or ''."""
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    tail = stem.rsplit("-", 1)[-1]
    return tail if tail.isdigit() else ""


def list_contests(directory=RESULTS_DIR):
    """Every parsed standings export on disk, unreadable ones included and flagged."""
    found = []
    for path in sorted(glob.glob(os.path.join(directory, "*.csv"))):
        try:
            found.append(read_contest(path))
        except Exception as error:
            found.append({"path": path, "name": os.path.basename(path), "ownership": {},
                          "fpts": {}, "entries": 0, "scores": pd.Series(dtype=float),
                          "players": 0, "contest_id": _contest_id(path),
                          "notes": [f"unreadable: {error.__class__.__name__}"]})
    return found


def contest_summary(contest):
    """One row describing a contest: size, shape and what it took to do well."""
    scores = contest["scores"]
    return {
        "Contest": contest["contest_id"] or contest["name"],
        "Entries": contest["entries"],
        "Players": contest["players"],
        "Winning": float(scores.max()) if len(scores) else np.nan,
        "Cash line": cash_line(contest),
        "Median": float(scores.median()) if len(scores) else np.nan,
        "Top 1%": float(scores.quantile(0.99)) if len(scores) else np.nan,
        "Notes": "; ".join(contest["notes"]),
    }


def cash_line(contest, fraction=0.20):
    """The score at the paid line, as a share of the field.

    Defaults to the top 20%, which is roughly where a large GPP starts paying. This is the
    number a portfolio is actually optimising against — "how often do I clear the line" is a
    different and more answerable question than "how often do I win".
    """
    scores = contest["scores"]
    if not len(scores):
        return np.nan
    return float(scores.quantile(1.0 - fraction))


def observed_ownership(contests):
    """{player key: mean %Drafted} across contests big enough to be the public field."""
    pooled = {}
    for contest in contests:
        if contest["entries"] < FIELD_MIN_ENTRIES:
            continue
        for key, value in contest["ownership"].items():
            pooled.setdefault(key, []).append(value)
    return {key: float(np.mean(values)) for key, values in pooled.items()}


def readiness(directory=RESULTS_DIR):
    """Is there enough here to calibrate ownership yet? Returns a report, never an opinion.

    The bar is deliberately explicit rather than a feeling: `nfl.ownership.calibrate` needs
    several *distinct slates* worth of public-field contests, because fitting a field model
    to one week fits that week's chalk rather than the field's behaviour.
    """
    contests = list_contests(directory)
    usable = [c for c in contests if c["entries"] >= FIELD_MIN_ENTRIES and c["ownership"]]
    return {
        "directory": directory,
        "files": len(contests),
        "usable": len(usable),
        "unreadable": [c["name"] for c in contests if any("unreadable" in n
                                                          for n in c["notes"])],
        "entries_total": int(sum(c["entries"] for c in usable)),
        "ready": len(usable) >= 3,
        "needed": max(0, 3 - len(usable)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Show DK contest results on disk and whether they can calibrate yet.")
    parser.add_argument("--directory", default=RESULTS_DIR)
    args = parser.parse_args(argv)

    contests = list_contests(args.directory)
    if not contests:
        print(f"No contest-standings exports in {args.directory}.")
        print("Download them from DraftKings after a slate settles and drop them there — "
              "ownership cannot be calibrated, and portfolio selection cannot be built, "
              "without several weeks of them.")
        return 0

    frame = pd.DataFrame([contest_summary(c) for c in contests])
    print(frame.to_string(index=False))

    state = readiness(args.directory)
    print(f"\n{state['usable']} of {state['files']} usable as a public field "
          f"({state['entries_total']:,} entries).")
    if state["unreadable"]:
        print(f"unreadable: {', '.join(state['unreadable'])}")
    if state["ready"]:
        print("Enough to attempt `nfl.ownership.calibrate`.")
    else:
        print(f"Need {state['needed']} more distinct slate(s) before calibrating — fitting "
              f"a field model to one week fits that week's chalk, not the field.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
