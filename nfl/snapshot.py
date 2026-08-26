"""Precomputed frames for the hosted dashboard — the football twin of `.cache/report_data/`.

    python -m nfl.snapshot --build          # rebuild from local data
    python -m nfl.snapshot                  # what is in it, and how stale

The MLB dashboard is deployed by committing the payloads its pages read and pushing. The
football pages cannot use that pattern directly, for three reasons this module exists to
fix:

**The raw PFF drop is the wrong thing to commit.** It is 14 MB across ~90 exports, several
of them 500 columns wide, and the pages read a few dozen of those columns. A snapshot of
the *derived* frames is a fraction of the size, loads instantly, and cannot drift from what
you saw locally.

**Three pages reach the network.** Defence, Teams and Trenches call nflverse through
`nfl.data` — weekly stats, schedules, depth charts. Render's filesystem is ephemeral, so
every cold start would re-download five seasons of parquet before the first page rendered,
on a free-tier container, while someone waits. Precomputed, they are a few hundred rows.

**Pickle is version-sensitive.** `requirements-web.txt` pins pandas exactly because the
MLB payloads are pickles. These are **parquet**, which survives a pandas minor upgrade, so
the pin protects the MLB half without also constraining this one.

## What is in it, and what is not

Everything the dashboard computes from PFF exports or from nflverse. **Not** the DK salary
and entries exports: those are small, they change every week, and they are the one input you
want to be able to drop in and see immediately without a rebuild.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

SNAPSHOT_DIR = os.path.join("nfl", "snapshot")
MANIFEST = "manifest.json"

# Frames the dashboard reads, and how to build each one. Keyed by the name the reader asks
# for; the builder is resolved lazily so importing this module never pulls in the world.
#
# Season-scoped frames are stored one file per season, because a page asks for one season at
# a time and reading five to serve one is the kind of waste that only shows up on a
# free-tier container.
SEASON_FRAMES = ("receivers", "rushers", "high_value", "target_distribution",
                 "defense_scheme", "defense_slot", "receiver_scheme", "quarterbacks")
LEAGUE_FRAMES = ("defense_allowed", "team_offense", "team_results", "team_summary")
LINE_FRAMES = ("line_players", "line_units")


def available_seasons(family, root=SNAPSHOT_DIR):
    """Seasons the snapshot holds for a family, or None when there is no snapshot.

    None and `[]` mean different things and the caller depends on it: None is "no snapshot,
    go look at the raw exports", `[]` is "the snapshot was built and this family had
    nothing".
    """
    state = manifest(root)
    if not state:
        return None
    return state.get("families", {}).get(family, [])


def path_for(name, season=None, root=SNAPSHOT_DIR):
    stem = f"{name}_{season}" if season is not None else name
    return os.path.join(root, f"{stem}.parquet")


def read(name, season=None, root=SNAPSHOT_DIR):
    """One snapshot frame, or None when it has not been built.

    Returning None rather than raising is deliberate: every reader falls back to computing
    the frame live, so a missing snapshot degrades to "slower locally" instead of "broken in
    production". The manifest is where you find out the snapshot is stale.
    """
    path = path_for(name, season, root)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _write(frame, name, season=None, root=SNAPSHOT_DIR):
    if frame is None or (hasattr(frame, "empty") and frame.empty):
        return None
    os.makedirs(root, exist_ok=True)
    path = path_for(name, season, root)
    # Column names come from the pages and carry spaces, percent signs and slashes.
    # Parquet is fine with all of them; the index is not, so it is dropped rather than
    # written as a column nobody reads back.
    frame.reset_index(drop=True).to_parquet(path, index=False)
    return path


def manifest(root=SNAPSHOT_DIR):
    path = os.path.join(root, MANIFEST)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build(seasons=None, line_season=None, root=SNAPSHOT_DIR, report=print):
    """Rebuild every snapshot frame from the local data. Returns the manifest written."""
    from dashboards import nfl_league, nfl_pff
    from nfl import pffdata

    def _raw_seasons(family):
        """Seasons on disk, read from the **raw catalog**, never from the snapshot.

        `nfl_pff.available_seasons` prefers the snapshot, which is correct for the app and
        wrong here: the builder would read the snapshot it is about to replace, find the
        previous run's answer, and on a first build find nothing at all. That is exactly
        what happened -- "no PFF exports found" on a machine holding ninety of them.
        """
        table = pffdata.catalog()
        rows = table[(table["family"] == family) & table["usable"]
                     & table["season"].notna()]
        return sorted({int(y) for y in rows["season"]}, reverse=True)

    seasons = sorted(seasons or _raw_seasons("receiving_summary"), reverse=True)
    if not seasons:
        raise SystemExit("no PFF exports found — nothing to snapshot")

    written, rows = [], {}
    # **Which seasons exist per family has to be recorded, not rediscovered.** The pages ask
    # `available_seasons()`, which reads `pffdata.catalog()` -- and on a host with no raw
    # exports the catalog tries to re-fingerprint, which needs nflverse rosters, which needs
    # the network. Nine of thirteen pages died on that in a hosted dry run.
    families = {}
    for family in ("receiving_summary", "rushing_summary", "defense_coverage_scheme",
                   "slot_coverage", "receiving_scheme", "passing_depth",
                   "offense_blocking", "fantasy-stats-receiving"):
        try:
            families[family] = _raw_seasons(family)
        except Exception:
            families[family] = []

    def keep(frame, name, season=None):
        path = _write(frame, name, season, root)
        if path:
            written.append(os.path.relpath(path, root))
            rows[os.path.basename(path)] = int(len(frame))

    # **Every builder is guarded, not just some of them.** The PFF families cover different
    # spans -- `receiving_summary` reaches 2018, `fantasy-stats-receiving` only 2022,
    # `passing_depth` only 2023 -- so any season will legitimately miss several frames. A
    # build that aborts on the first gap produces a partial snapshot that looks complete.
    builders = (
        ("receivers", lambda y: nfl_pff.receivers((y,))),
        ("rushers", lambda y: nfl_pff.rushers((y,))),
        ("high_value", lambda y: nfl_pff.high_value((y,))),
        ("target_distribution", lambda y: nfl_pff.target_distribution((y,))),
        ("defense_scheme", nfl_pff.defense_scheme),
        ("defense_slot", nfl_pff.defense_slot),
        ("receiver_scheme", nfl_pff.receiver_scheme),
        ("quarterbacks", nfl_pff.quarterbacks),
    )
    for season in seasons:
        report(f"  {season}...")
        for name, builder in builders:
            try:
                keep(builder(season), name, season)
            except Exception as error:
                report(f"    [skip] {name} {season}: "
                       f"{error.__class__.__name__}: {str(error)[:70]}")

    report("  league (nflverse)...")
    league_seasons = tuple(s for s in seasons if s >= min(seasons))
    try:
        keep(nfl_league.defense_allowed(league_seasons), "defense_allowed")
        keep(nfl_league.team_offense(league_seasons), "team_offense")
        keep(nfl_league.team_results(league_seasons), "team_results")
        keep(nfl_league.team_summary(league_seasons), "team_summary")
    except Exception as error:
        report(f"    [skip] league frames: {error}")

    report("  projections...")
    try:
        from dashboards import nfl_slates
        projected, source = nfl_slates.projections()
        keep(projected, "projections")
    except Exception as error:
        report(f"    [skip] projections: {error.__class__.__name__}")

    line_season = int(line_season or max(seasons) + 1)
    report(f"  offensive lines for {line_season}...")
    try:
        players, units = nfl_pff.lines(line_season)
        keep(players, "line_players")
        keep(units, "line_units")
    except Exception as error:
        report(f"    [skip] lines {line_season}: {error}")

    written_manifest = {
        "built": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seasons": seasons,
        "families": families,
        "league_seasons": list(league_seasons),
        "line_season": line_season,
        "files": sorted(written),
        "rows": rows,
        "pandas": pd.__version__,
    }
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, MANIFEST), "w", encoding="utf-8") as handle:
        json.dump(written_manifest, handle, indent=1, sort_keys=True)
    return written_manifest


def describe(root=SNAPSHOT_DIR):
    """What is in the snapshot and how old it is, for the CLI and for the dashboard."""
    state = manifest(root)
    if not state:
        return {"present": False, "files": 0, "built": None, "age_days": None,
                "bytes": 0, "seasons": []}
    total = 0
    for name in state.get("files", []):
        path = os.path.join(root, name)
        if os.path.exists(path):
            total += os.path.getsize(path)
    age = None
    try:
        built = datetime.strptime(state["built"], "%Y-%m-%dT%H:%M:%SZ")
        age = (datetime.now(timezone.utc).replace(tzinfo=None) - built).days
    except Exception:
        pass
    return {"present": True, "files": len(state.get("files", [])),
            "built": state.get("built"), "age_days": age, "bytes": total,
            "seasons": state.get("seasons", []), "pandas": state.get("pandas")}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build or inspect the precomputed dashboard snapshot.")
    parser.add_argument("--build", action="store_true", help="Rebuild from local data.")
    parser.add_argument("--seasons", default=None,
                        help="Comma-separated, e.g. '2025,2024'. Default: every season "
                             "with a PFF export.")
    parser.add_argument("--line-season", type=int, default=None)
    parser.add_argument("--root", default=SNAPSHOT_DIR)
    args = parser.parse_args(argv)

    if args.build:
        seasons = ([int(s) for s in args.seasons.split(",")] if args.seasons else None)
        print(f"Building snapshot into {args.root}")
        state = build(seasons, line_season=args.line_season, root=args.root)
        print(f"  wrote {len(state['files'])} frames")

    state = describe(args.root)
    if not state["present"]:
        print(f"No snapshot in {args.root}. Run with --build.")
        return 0
    print(f"\n{state['files']} frames, {state['bytes'] / 1_000_000:.1f} MB, "
          f"built {state['built']}"
          + (f" ({state['age_days']}d ago)" if state["age_days"] is not None else ""))
    print(f"seasons: {', '.join(str(s) for s in state['seasons'])}")
    print("\nCommit `nfl/snapshot/` and push — that is the whole deploy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
