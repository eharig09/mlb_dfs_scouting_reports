"""Immutable slate snapshots: everything that was knowable at one moment, frozen.

**The problem this solves.** `.cache/report_data/*.pkl` is mutated in place --
`--refresh-lineups` overwrites the same path -- so re-reading a past night gives you the
payload *as it stands now*, not as it stood when the lineups were built. Confirmed lineups,
late scratches and corrected probables all leak backwards. Every review and backtest built
on `build_slate(date)` is therefore quietly scoring a model that had information it did not
have. On a night where a scratch arrived at 6:55, the "projection" being graded may never
have existed.

A snapshot fixes that by copying the state out of the mutable cache into a directory that
is never written twice. Re-taking a stage writes `<stage>.r2` beside `<stage>` and the
manifest records the chain; nothing is ever replaced.

**Stages.** A slate is not one state, it is four:

    morning     probables posted, no lineups
    t-2h        two hours out; some lineups, market has moved
    confirmed   lineup cards in
    final       last look before lock

They are separate snapshots because the interesting questions are about the differences --
how much a projection moved once the card posted, whether the extra information paid.

**Layout.**

    .cache/snapshots/<date>/<slate>/<stage>[.rN]/
        manifest.json     versions, hashes, cutoff, counts, optimizer settings
        players.parquet   the full slate frame, exactly as the optimizer saw it
        stacks.parquet    team stack table
        games.json        per game: pairing, gamePk, first pitch, park, weather, probables,
                          batting orders with their confirmation status
        salaries.csv      verbatim copy of the DK export

Parquet for tables, JSON for metadata, CSV only for the DK file (which has to stay
byte-comparable with what DK served). A 15-game snapshot is ~250 KB.

**Integrity.** Every cached payload is hashed into the manifest. A later reader can tell
whether the underlying report data has been mutated since -- which is the specific failure
this module exists to make visible rather than silent.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone

import pandas as pd

from .naming import slate_slug
from .ownership import OWNERSHIP_VERSION
from .projections import MODEL_VERSION
from .salaries import canon_team
from .slate import REPORT_DATA_DIR, build_slate, cached_games

SNAPSHOT_ROOT = os.path.join(".cache", "snapshots")

# The four moments a slate is worth freezing. Order matters: `latest_stage` walks it
# backwards to find the most informed snapshot on hand.
STAGES = ["morning", "t-2h", "confirmed", "final"]

# Snapshot format version. Bumped when the on-disk layout changes in a way a reader has to
# know about; readers refuse a major version they do not understand rather than guessing.
FORMAT_VERSION = 1

# Columns dropped before writing. Python lists survive Parquet fine, but these two are
# rebuilt from Supports/Cautions on read and storing all four doubles the file for nothing.
DERIVED_COLUMNS = ("Why", "Risks")


class SnapshotError(Exception):
    pass


# ---------------------------------------------------------------------------
# Versions and hashing
# ---------------------------------------------------------------------------

def code_version():
    """Short git revision, with a dirty marker. 'unknown' outside a checkout.

    Recorded rather than computed at read time because the whole point is to say which code
    produced the numbers, and by the time anyone reads a snapshot the tree has moved on.
    """
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if rev.returncode != 0:
            return "unknown"
        head = rev.stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10)
        return head + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def file_digest(path, chunk=1 << 20):
    """sha256 of a file, or None if it cannot be read."""
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            while True:
                block = handle.read(chunk)
                if not block:
                    break
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def frame_digest(frame, columns=None):
    """Stable sha256 of a DataFrame's contents.

    Column-ordered and index-free, so a reordered frame with the same content hashes the
    same -- the question being asked is "are these the same projections", not "is this the
    same object".
    """
    if frame is None or frame.empty:
        return hashlib.sha256(b"").hexdigest()
    use = [c for c in (columns or sorted(frame.columns)) if c in frame.columns]
    payload = frame[use].to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def snapshot_dir(date, slate, stage, revision=1, root=SNAPSHOT_ROOT):
    name = stage if revision <= 1 else f"{stage}.r{revision}"
    return os.path.join(root, str(date), slate_slug(slate or "unknown"), name)


def _revisions(date, slate, stage, root=SNAPSHOT_ROOT):
    """Existing revision numbers of one stage, ascending."""
    parent = os.path.join(root, str(date), slate_slug(slate or "unknown"))
    if not os.path.isdir(parent):
        return []
    found = []
    for entry in os.listdir(parent):
        if entry == stage:
            found.append(1)
        else:
            match = re.fullmatch(re.escape(stage) + r"\.r(\d+)", entry)
            if match:
                found.append(int(match.group(1)))
    return sorted(found)


def next_revision(date, slate, stage, root=SNAPSHOT_ROOT):
    existing = _revisions(date, slate, stage, root)
    return (existing[-1] + 1) if existing else 1


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

def _lineup_block(frame, team):
    """Batting order for one team, with each hitter's confirmation status."""
    hitters = frame[(frame["Team"].astype(str) == str(team)) & (frame["Type"] == "H")]
    hitters = hitters.sort_values("Slot")
    return [
        {
            "slot": int(row["Slot"]) if pd.notna(row.get("Slot")) else None,
            "name": row["Name"],
            "mlbam": None if pd.isna(row.get("MLBAM")) else int(row["MLBAM"]),
            "pos": row.get("Pos"),
            "bats": row.get("Bats"),
            "status": row.get("Lineup") or "",
        }
        for _, row in hitters.iterrows()
    ]


def _game_records(date, players, meta, data_dir):
    """Per-game context: identity, start time, park, weather, probables, batting orders.

    Read from the cached payloads rather than from the projection rows, because the point of
    a snapshot is to preserve the *inputs* -- a projection can be recomputed from them, but a
    weather forecast that has since been replaced by an observation cannot be recovered.
    """
    import pickle

    records = []
    for path, away, home in cached_games(date, data_dir):
        pairing = f"{away}@{home}"
        if meta.get("games") and pairing not in meta["games"]:
            continue                       # dropped as a doubleheader half, or postponed
        try:
            with open(path, "rb") as handle:
                payload = pickle.load(handle)
        except Exception as error:
            records.append({"game": pairing, "error": str(error), "payload": os.path.basename(path)})
            continue

        args = payload.get("report_args") or ()
        context = payload.get("advanced_context") or {}
        environment = context.get("environment") or {}
        forecast = environment.get("forecast") or {}

        def probable(starter):
            starter = starter or {}
            return {
                "name": starter.get("Name"),
                "throws": starter.get("Throws"),
                "fip": starter.get("FIP"),
                "era": starter.get("ERA"),
                "k_pct": starter.get("K%"),
                "bb_pct": starter.get("BB%"),
                "whip": starter.get("WHIP"),
                "gs": starter.get("GS"),
                "ip": starter.get("IP"),
            }

        records.append({
            "game": pairing,
            "payload": os.path.basename(path),
            # The hash is the integrity check: if the payload is later refreshed, a reader
            # can say so instead of quietly comparing against different inputs.
            "payload_sha256": file_digest(path),
            "payload_mtime_utc": datetime.fromtimestamp(
                os.path.getmtime(path), timezone.utc).isoformat(timespec="seconds"),
            "game_pk": environment.get("game_id"),
            "first_pitch_utc": environment.get("game_datetime"),
            "venue": environment.get("venue"),
            "park": environment.get("park"),
            "weather": {
                # Both sources, unmerged. Which one the model chose is a model decision and
                # may change; what was on hand is a fact and must not.
                "statsapi": environment.get("weather") or {},
                "forecast": {k: forecast.get(k) for k in
                             ("roof", "temp_f", "feels_f", "humidity_pct", "precip_pct",
                              "wind", "wind_gust_mph", "sky", "carry", "wind_estimated")},
            },
            "hp_umpire": environment.get("hp_umpire"),
            "probables": {
                # args[2] is the home starter, args[9] the away one -- the same indices
                # dfs.projections reads.
                "home": probable(args[2] if len(args) > 9 else None),
                "away": probable(args[9] if len(args) > 9 else None),
            },
            "market": context.get("scorecard").to_dict("records")
            if isinstance(context.get("scorecard"), pd.DataFrame) else None,
            "lineups": {
                canon_team(away): _lineup_block(players, away),
                canon_team(home): _lineup_block(players, home),
            } if not players.empty else {},
        })
    return records


def take_snapshot(date, slate=None, stage="final", players=None, stacks=None, meta=None,
                  settings=None, data_dir=REPORT_DATA_DIR, root=SNAPSHOT_ROOT,
                  salary_path=None, note=None):
    """Freeze the current state of a slate. Returns the written Snapshot.

    Pass `players`/`stacks`/`meta` from a build you already have; otherwise the slate is
    built here. Either way the *inputs* are captured from the payload cache, not from the
    projection rows, so the snapshot can outlive a model change.

    `settings` is whatever optimizer configuration is in force -- a dict, stored verbatim.
    """
    if stage not in STAGES:
        raise SnapshotError(f"stage must be one of {STAGES}, got {stage!r}")

    if players is None:
        players, stacks, meta = build_slate(date, salary_path=salary_path, slate=slate,
                                            data_dir=data_dir)
    meta = dict(meta or {})
    if players is None or players.empty:
        raise SnapshotError(f"no slate data for {date}; nothing to snapshot")

    label = slate or meta.get("slate_label") or "unknown"
    revision = next_revision(date, label, stage, root)
    directory = snapshot_dir(date, label, stage, revision, root)
    os.makedirs(directory, exist_ok=True)

    frame = players.drop(columns=[c for c in DERIVED_COLUMNS if c in players.columns])
    frame.to_parquet(os.path.join(directory, "players.parquet"), index=False)
    if stacks is not None and not stacks.empty:
        stacks.to_parquet(os.path.join(directory, "stacks.parquet"), index=False)

    games = _game_records(date, players, meta, data_dir)
    with open(os.path.join(directory, "games.json"), "w", encoding="utf-8") as handle:
        json.dump(games, handle, indent=2, default=str)

    salary_file = meta.get("salary_file") or salary_path
    salary_copy = None
    if salary_file and os.path.exists(salary_file):
        salary_copy = os.path.join(directory, "salaries.csv")
        shutil.copy2(salary_file, salary_copy)

    # The cutoff is the newest thing that went in. Anything timestamped after it was not
    # available, which is exactly the guarantee a walk-forward backtest needs.
    mtimes = [g.get("payload_mtime_utc") for g in games if g.get("payload_mtime_utc")]
    manifest = {
        "format_version": FORMAT_VERSION,
        "date": str(date),
        "slate": label,
        "stage": stage,
        "revision": revision,
        "taken_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_cutoff_utc": max(mtimes) if mtimes else None,
        "note": note,
        "versions": {
            "model": MODEL_VERSION,
            "ownership": OWNERSHIP_VERSION,
            "code": code_version(),
        },
        "counts": {
            "games": len(games),
            "players": int(len(players)),
            "hitters": int((players["Type"] == "H").sum()),
            "pitchers": int((players["Type"] == "P").sum()),
            "confirmed_hitters": int(
                ((players["Type"] == "H") & (players["Lineup"] == "Confirmed")).sum()),
        },
        "digests": {
            # Restricted to the columns a decision is actually made on, so a cosmetic column
            # addition does not read as "the projections changed".
            "projections": frame_digest(players, ["Name", "Team", "Type", "Slot", "Salary",
                                                  "Proj", "Ceiling", "Floor", "Bust%",
                                                  "Own%"]),
            "salary_file": file_digest(salary_file) if salary_file else None,
        },
        "salary_file": salary_file,
        "ownership_source": meta.get("ownership_source"),
        "has_salary": bool(meta.get("has_salary")),
        "missing_games": meta.get("missing_games") or [],
        "postponed": meta.get("postponed") or {},
        "doubleheaders": meta.get("doubleheaders") or [],
        "settings": settings or {},
        "previous_revisions": _revisions(date, label, stage, root)[:-1],
    }
    with open(os.path.join(directory, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)

    return Snapshot(directory, manifest, players, stacks, games)


# ---------------------------------------------------------------------------
# Read back
# ---------------------------------------------------------------------------

class Snapshot:
    """A frozen slate. Loaded lazily -- a listing should not read every Parquet file."""

    def __init__(self, directory, manifest, players=None, stacks=None, games=None):
        self.directory = directory
        self.manifest = manifest
        self._players = players
        self._stacks = stacks
        self._games = games

    def __repr__(self):
        m = self.manifest
        return (f"<Snapshot {m.get('date')} {m.get('slate')} {m.get('stage')}"
                f"{'' if m.get('revision', 1) == 1 else '.r%d' % m['revision']} "
                f"{m.get('counts', {}).get('players', '?')} players>")

    @property
    def date(self):
        return self.manifest.get("date")

    @property
    def slate(self):
        return self.manifest.get("slate")

    @property
    def stage(self):
        return self.manifest.get("stage")

    @property
    def players(self):
        if self._players is None:
            frame = pd.read_parquet(os.path.join(self.directory, "players.parquet"))
            # Rebuilt rather than stored: they are pure functions of Supports/Cautions and
            # every consumer of a slate frame expects them to exist.
            for column in ("Supports", "Cautions"):
                if column in frame.columns:
                    frame[column] = frame[column].apply(
                        lambda v: list(v) if v is not None and not isinstance(v, str) else [])
            if "Supports" in frame.columns:
                frame["Why"] = frame.apply(
                    lambda row: "; ".join(list(row["Supports"])[:3]
                                          + [f"[risk] {c}" for c in list(row["Cautions"])[:1]])
                    or "neutral matchup", axis=1)
                frame["Risks"] = frame["Cautions"].apply(lambda r: "; ".join(list(r)[:3]))
            self._players = frame
        return self._players

    @property
    def stacks(self):
        if self._stacks is None:
            path = os.path.join(self.directory, "stacks.parquet")
            self._stacks = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()
        return self._stacks

    @property
    def games(self):
        if self._games is None:
            path = os.path.join(self.directory, "games.json")
            with open(path, encoding="utf-8") as handle:
                self._games = json.load(handle)
        return self._games

    @property
    def salary_file(self):
        """The copy stored inside the snapshot, not the original path."""
        path = os.path.join(self.directory, "salaries.csv")
        return path if os.path.exists(path) else None

    def meta(self):
        """A `build_slate`-shaped meta dict, so snapshot consumers need no special case."""
        m = self.manifest
        return {
            "date": m.get("date"),
            "slate_label": m.get("slate"),
            "games": [g["game"] for g in self.games],
            "salary_file": self.salary_file or m.get("salary_file"),
            "has_salary": m.get("has_salary", False),
            "missing_games": m.get("missing_games") or [],
            "postponed": m.get("postponed") or {},
            "doubleheaders": m.get("doubleheaders") or [],
            "ownership_source": m.get("ownership_source"),
            "snapshot": {"stage": m.get("stage"), "revision": m.get("revision"),
                         "taken_utc": m.get("taken_utc"),
                         "data_cutoff_utc": m.get("data_cutoff_utc"),
                         "versions": m.get("versions", {})},
            "game_starts": {},
            "schedule_error": None,
            "environment_error": None,
            "skipped": [], "ambiguous_slates": [], "cached_but_unreadable": 0,
            "postponed_players": 0,
        }

    def stale_payloads(self, data_dir=REPORT_DATA_DIR):
        """Games whose cached payload has been rewritten since this snapshot was taken.

        The whole reason this module exists. A non-empty result means rebuilding this slate
        today would NOT reproduce the snapshot, so any comparison against a rebuild is
        measuring a refresh rather than a model.
        """
        changed = []
        for record in self.games:
            name, expected = record.get("payload"), record.get("payload_sha256")
            if not name or not expected:
                continue
            path = os.path.join(data_dir, name)
            if not os.path.exists(path):
                changed.append((record["game"], "payload deleted"))
            elif file_digest(path) != expected:
                changed.append((record["game"], "payload rewritten since snapshot"))
        return changed


def load_snapshot(date, slate=None, stage=None, revision=None, root=SNAPSHOT_ROOT):
    """Load one snapshot. Without `stage`, the most informed one on hand wins.

    "Most informed" is stage order, not write order: a `final` taken at 6 p.m. beats a
    `morning` re-taken at 6:30 by accident.
    """
    available = list_snapshots(date, slate, root=root)
    if not available:
        raise SnapshotError(f"no snapshots for {date}"
                            + (f" slate {slate}" if slate else "")
                            + f" under {root}/")
    if stage:
        available = [s for s in available if s.manifest.get("stage") == stage]
        if not available:
            raise SnapshotError(f"no '{stage}' snapshot for {date} {slate or ''}")
    if revision is not None:
        available = [s for s in available if s.manifest.get("revision") == revision]
        if not available:
            raise SnapshotError(f"no revision {revision} of {stage or 'any stage'}")

    def rank(snap):
        m = snap.manifest
        order = STAGES.index(m["stage"]) if m.get("stage") in STAGES else -1
        return (order, m.get("revision", 1), m.get("taken_utc") or "")

    return max(available, key=rank)


def list_snapshots(date=None, slate=None, root=SNAPSHOT_ROOT):
    """Every snapshot on disk, newest date first. Manifests only -- tables stay unread."""
    if not os.path.isdir(root):
        return []
    found = []
    dates = [date] if date else sorted(os.listdir(root), reverse=True)
    for day in dates:
        day_dir = os.path.join(root, str(day))
        if not os.path.isdir(day_dir):
            continue
        for label in sorted(os.listdir(day_dir)):
            if slate and slate_slug(slate) != label:
                continue
            slate_dir = os.path.join(day_dir, label)
            if not os.path.isdir(slate_dir):
                continue
            for entry in sorted(os.listdir(slate_dir)):
                directory = os.path.join(slate_dir, entry)
                manifest_path = os.path.join(directory, "manifest.json")
                if not os.path.exists(manifest_path):
                    continue
                try:
                    with open(manifest_path, encoding="utf-8") as handle:
                        manifest = json.load(handle)
                except (OSError, ValueError):
                    continue
                if manifest.get("format_version", 1) > FORMAT_VERSION:
                    continue          # written by a newer version; do not guess at it
                found.append(Snapshot(directory, manifest))
    return found


def load_or_build(date, slate=None, stage=None, root=SNAPSHOT_ROOT, **build_kwargs):
    """(players, stacks, meta, source) preferring a snapshot, falling back to a rebuild.

    The fallback is deliberately loud in the returned `source`: a review that silently
    rebuilt is a review of today's data, and the caller has to be able to say so.
    """
    try:
        snap = load_snapshot(date, slate, stage=stage, root=root)
    except SnapshotError:
        players, stacks, meta = build_slate(date, slate=slate, **build_kwargs)
        return players, stacks, meta, "rebuilt"
    return snap.players, snap.stacks, snap.meta(), snap


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Freeze a slate so a later review scores what was actually known.")
    parser.add_argument("--date", default=datetime.today().strftime("%Y-%m-%d"))
    parser.add_argument("--slate", help="Which DK export, when the date has several.")
    parser.add_argument("--stage", choices=STAGES, default="final",
                        help="Which moment this is. Default: final (last look before lock).")
    parser.add_argument("--salaries", help="Explicit DK export path.")
    parser.add_argument("--note", help="Free text stored in the manifest.")
    parser.add_argument("--list", action="store_true", help="Show snapshots on disk and stop.")
    parser.add_argument("--verify", action="store_true",
                        help="Check whether the payloads behind a snapshot have since changed.")
    args = parser.parse_args()

    if args.list:
        snaps = list_snapshots(args.date if args.date else None, args.slate)
        if not snaps:
            print(f"No snapshots under {SNAPSHOT_ROOT}/.")
            return
        print(f"{len(snaps)} snapshot(s):")
        for snap in snaps:
            m = snap.manifest
            counts = m.get("counts", {})
            rev = "" if m.get("revision", 1) == 1 else f".r{m['revision']}"
            print(f"  {m['date']}  {m['slate']:<12} {m['stage'] + rev:<12} "
                  f"{counts.get('players', 0):3d} players "
                  f"({counts.get('confirmed_hitters', 0)} confirmed) "
                  f"model v{m.get('versions', {}).get('model')} "
                  f"code {m.get('versions', {}).get('code')}  {m.get('taken_utc')}")
        return

    if args.verify:
        snap = load_snapshot(args.date, args.slate, stage=args.stage)
        stale = snap.stale_payloads()
        print(f"{snap}  taken {snap.manifest.get('taken_utc')}")
        if not stale:
            print("  every payload still matches; a rebuild would reproduce this snapshot.")
        else:
            print(f"  [!] {len(stale)} game(s) have changed since:")
            for game, why in stale:
                print(f"      {game}: {why}")
            print("      A rebuild would NOT reproduce this slate. Score from the snapshot.")
        return

    snap = take_snapshot(args.date, slate=args.slate, stage=args.stage,
                         salary_path=args.salaries, note=args.note)
    m = snap.manifest
    print(f"snapshot -> {snap.directory}")
    print(f"  {m['date']} {m['slate']} / {m['stage']}"
          f"{'' if m['revision'] == 1 else ' r%d' % m['revision']}")
    print(f"  {m['counts']['games']} games, {m['counts']['players']} players "
          f"({m['counts']['confirmed_hitters']} confirmed hitters)")
    print(f"  model v{m['versions']['model']}, ownership v{m['versions']['ownership']}, "
          f"code {m['versions']['code']}")
    print(f"  data cutoff {m['data_cutoff_utc']}")
    if m["revision"] > 1:
        print(f"  [i] earlier revisions kept: {m['previous_revisions']}")


if __name__ == "__main__":
    main()
