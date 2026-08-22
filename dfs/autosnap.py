"""Run the night's refreshes and snapshots on the slate's clock, not on yours.

`dfs.snapshot` freezes what was knowable at one moment, and the four stages it defines --
morning, t-2h, confirmed, final -- are only meaningful if they are actually taken at those
moments. Taken by hand they drift: the "confirmed" snapshot gets remembered at 6:58 for a
7:05 slate, or missed entirely, and the whole point of comparing stages is lost.

**Snapshots are scheduled against first pitch, not against the wall clock.** A 1:05 getaway
slate and a 10:10 west-coast slate need the same *relative* moments, and a fixed cron time
would take "t-2h" four hours early on one and after lock on the other. Lock comes from the
DK export itself -- it prices the games it prices, and its `Game Info` column carries each
first pitch -- so the schedule is derived from the same file the contest is built on.

**A snapshot without a refresh is a copy of the last one.** `take_snapshot` reads the
mutable `.cache/report_data` payloads, so snapshotting four times without re-pulling lineups
in between produces four identical morning states with four different labels -- which looks
like the feature working and is worse than not having it. Each stage therefore refreshes
first and snapshots second, and `--no-refresh` has to be asked for explicitly.

Usage::

    python -m dfs.autosnap --date 2026-08-08 --slate main --plan   # show the times, exit
    python -m dfs.autosnap --slate main                            # run until final
    python -m dfs.autosnap --slate main --stages confirmed,final   # only the late ones

`--plan` prints wall-clock times so they can be pasted into Task Scheduler or cron by anyone
who would rather not leave a process running.
"""

import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from .salaries import AmbiguousSlate, describe_slate, find_salary_file
from .snapshot import STAGES, take_snapshot

EASTERN = ZoneInfo("America/New_York")

# Minutes before the slate's first pitch that each stage fires.
#
# `confirmed` sits at 55 minutes rather than the two hours the stage name suggests: MLB
# lineup cards post on no fixed schedule, and while most are out three hours early, the ones
# that matter -- a star resting, a late scratch -- routinely land inside the last hour.
# `final` is close enough to lock to catch a scratch and far enough back to leave time to
# rebuild and upload.
STAGE_OFFSETS = {
    "morning": 360,      # six hours out: probables known, no cards
    "t-2h": 120,
    "confirmed": 55,
    "final": 12,
}

# The morning snapshot is capped to a civilised hour: for a 1:05 start, lock - 6h is 7:05am
# and there is nothing to see. Local time, because this one is about the user's day.
MORNING_NOT_BEFORE_HOUR = 9

# Re-pulls lineups into the report cache and re-renders. `--dfs-slate` matters: a date with
# two DK exports is ambiguous, and without it the run drops DFS highlighting and warns.
# The render is the slow part -- override with --refresh-cmd if only the cache is wanted.
DEFAULT_REFRESH = (
    "{python} scouting_report.py --date {date} --all-games --from-cache "
    "--refresh-lineups --format xlsx{slate_flag}"
)


class AutosnapError(Exception):
    pass


def slate_lock(date, salary_path=None, slate=None):
    """(lock datetime in ET, salary file, slate description) for the night's slate.

    Lock is the earliest first pitch the DK export prices, which is what DK itself locks on.
    """
    try:
        path = salary_path or find_salary_file(date, slate=slate)
    except AmbiguousSlate as choice:
        # Naming the exports beats "not found": the usual cause is a --slate that does not
        # match how the file is named, and the fix is visible from the list.
        names = ", ".join(c["name"] for c in choice.candidates)
        raise AutosnapError(
            f"no DK salary export matches slate '{slate}' for {date}. Available: {names}"
        ) from choice
    if not path:
        raise AutosnapError(
            f"no DK salary export found for {date}"
            + (f" slate {slate}" if slate else "")
            + " -- autosnap schedules off the export's own first-pitch times.")
    info = describe_slate(path)
    if info.get("first") is None:
        raise AutosnapError(f"{os.path.basename(path)} has no readable Game Info times.")
    day = datetime.strptime(str(date), "%Y-%m-%d")
    lock = datetime(day.year, day.month, day.day, tzinfo=EASTERN) + timedelta(
        minutes=int(info["first"]))
    return lock, path, info


def adopt_downloads(date):
    """File any stray DK download for `date` under the naming convention.

    A browser writes `DKSalaries (20).csv`, which matches no `--slate` and leaves an
    unattended run dead on arrival -- the failure this hit on its first real firing. Running
    adoption first means the task works on a day the downloads were never renamed by hand.
    Collisions are reported and not forced: two exports that both classify as the same slate
    is a genuine ambiguity, and picking one would price the board off the wrong contest.
    """
    from .adopt import adopt_for_date

    try:
        adopted, problems = adopt_for_date(date)
    except Exception as error:
        print(f"  [!] could not file stray downloads: {error}")
        return
    for stray in adopted or []:
        print(f"  filed {os.path.basename(stray['path'])} -> "
              f"{os.path.basename(stray['target'])}")
    for problem in problems or []:
        print(f"  [!] {problem}")


def wait_for_slate(date, salary_path=None, slate=None, timeout_minutes=240, poll_seconds=300):
    """Resolve the slate, waiting for the DK export to appear if it has not been downloaded.

    A single daily scheduled task fires at a fixed hour, but the salary file arrives whenever
    it arrives -- so failing immediately on "no DK export" would make the task useless on any
    day the download happened later. Polling instead means the morning entry can be set once
    and forgotten.

    An *ambiguous* slate is not waited on: another export appearing will not resolve it, and
    the error already names the alternatives.
    """
    deadline = datetime.now(EASTERN) + timedelta(minutes=timeout_minutes)
    announced = False
    while True:
        try:
            return slate_lock(date, salary_path, slate)
        except AutosnapError as error:
            if "Available:" in str(error) or datetime.now(EASTERN) >= deadline:
                raise
            if not announced:
                print(f"  waiting for the DK export for {date}"
                      + (f" (slate '{slate}')" if slate else "")
                      + f"; checking every {poll_seconds // 60}m until "
                      f"{_clock12(deadline)} ET...", flush=True)
                announced = True
            time.sleep(poll_seconds)


def stage_times(lock, stages=None):
    """[(stage, fire time in ET)] in chronological order."""
    wanted = [s for s in (stages or STAGES) if s in STAGE_OFFSETS]
    plan = []
    for stage in wanted:
        fire = lock - timedelta(minutes=STAGE_OFFSETS[stage])
        if stage == "morning":
            # Local-time floor, expressed back in ET so the whole plan stays on one clock.
            local_floor = datetime.now().astimezone().replace(
                hour=MORNING_NOT_BEFORE_HOUR, minute=0, second=0, microsecond=0)
            floor_et = local_floor.astimezone(EASTERN).replace(
                year=fire.year, month=fire.month, day=fire.day)
            fire = max(fire, floor_et)
        plan.append((stage, fire))
    return sorted(plan, key=lambda item: item[1])


def _run(command, dry_run=False):
    """Run a shell command, streaming its output. Returns True on success."""
    if dry_run:
        print(f"      would run: {command}")
        return True
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"      [!] command exited {result.returncode}")
    return result.returncode == 0


def run_stage(date, stage, slate=None, salary_path=None, refresh=True,
              refresh_cmd=None, dry_run=False):
    """Refresh the cache, then freeze it. Refresh failure does not skip the snapshot.

    A stage that cannot refresh is still worth capturing: a snapshot of stale data is a
    known quantity, and the manifest records the cutoff, whereas no snapshot at all leaves
    a hole in the night that cannot be reconstructed afterwards.
    """
    print(f"\n=== {stage} @ {datetime.now(EASTERN):%H:%M ET} ===", flush=True)
    if refresh:
        command = (refresh_cmd or DEFAULT_REFRESH).format(
            python=sys.executable, date=date, slate=slate or "",
            slate_flag=f" --dfs-slate {slate}" if slate else "")
        print(f"  refreshing lineups...", flush=True)
        _run(command, dry_run=dry_run)
    if dry_run:
        print(f"      would snapshot stage={stage}")
        return None
    snap = take_snapshot(date, slate=slate, stage=stage, salary_path=salary_path,
                         note=f"autosnap: {STAGE_OFFSETS[stage]}m before first pitch")
    counts = snap.manifest["counts"]
    print(f"  snapshot -> {snap.directory}")
    print(f"    {counts['games']} games, {counts['players']} players, "
          f"{counts['confirmed_hitters']} confirmed hitters", flush=True)
    return snap


class _Tee:
    """Write to the console and to a log file at once.

    Autosnap logs itself rather than being wrapped in `cmd /c ... >> file`, because that
    wrapper is a quoting minefield: `cmd /c` mangles a command line whose first character is
    a quote, which is exactly what an interpreter path containing spaces produces. The
    scheduled task therefore invokes python directly and passes --log.
    """

    def __init__(self, stream, path):
        self.stream = stream
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.handle = open(path, "a", encoding="utf-8", buffering=1)

    def write(self, text):
        self.stream.write(text)
        self.handle.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()
        self.handle.flush()


def _clock12(moment):
    """"7:05PM" without the leading zero. `%-I` is a POSIX extension and raises on Windows."""
    return f"{moment.hour % 12 or 12}:{moment.minute:02d}{'AM' if moment.hour < 12 else 'PM'}"


def _sleep_until(target, poll=20):
    """Sleep until `target` (ET-aware), waking periodically.

    Short naps rather than one long one so the process stays interruptible and so a laptop
    that suspended through a stage notices as soon as it wakes rather than at the end of a
    six-hour sleep.
    """
    while True:
        remaining = (target - datetime.now(EASTERN)).total_seconds()
        if remaining <= 0:
            return
        time.sleep(min(poll, remaining))


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Take the night's snapshots on a schedule derived from first pitch.")
    parser.add_argument("--date", default=datetime.now(EASTERN).strftime("%Y-%m-%d"))
    parser.add_argument("--slate", help="Which DK export, when the date has several.")
    parser.add_argument("--salaries", help="Explicit DK export path.")
    parser.add_argument("--stages", help=f"Comma-separated subset of {','.join(STAGES)}.")
    parser.add_argument("--plan", action="store_true",
                        help="Print the schedule and exit, without running anything.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Walk the schedule and print each action without doing it.")
    parser.add_argument("--no-refresh", action="store_true",
                        help="Snapshot without re-pulling lineups first. Rarely what you "
                             "want: consecutive stages then capture identical data.")
    parser.add_argument("--refresh-cmd",
                        help="Override the refresh command. {python} {date} {slate} expand.")
    parser.add_argument("--adopt", action="store_true",
                        help="File stray DK downloads under the naming convention first. "
                             "Needed for an unattended run: a browser writes "
                             "'DKSalaries (20).csv', which matches no --slate.")
    parser.add_argument("--log", metavar="PATH",
                        help="Append all output to this file as well as the console. Used by "
                             "the scheduled task, whose console output would otherwise be "
                             "discarded with no way to see why a night failed.")
    parser.add_argument("--wait-for-salaries", type=int, default=240, metavar="MINUTES",
                        help="If the DK export is not downloaded yet, poll for it this long "
                             "(default 240). A daily scheduled task fires at a fixed hour "
                             "but the export arrives when it arrives. 0 fails immediately.")
    parser.add_argument("--skip-past", action="store_true",
                        help="Skip stages whose time has already passed instead of running "
                             "them immediately on start-up.")
    args = parser.parse_args()

    if args.log:
        sys.stdout = _Tee(sys.stdout, args.log)
        sys.stderr = sys.stdout
        rule = "=" * 70
        started = datetime.now(EASTERN).strftime("%Y-%m-%d %H:%M:%S ET")
        print(f"\n{rule}\nautosnap {args.date} slate={args.slate or '(auto)'} "
              f"started {started}\n{rule}", flush=True)

    if args.adopt:
        adopt_downloads(args.date)

    stages = [s.strip() for s in args.stages.split(",")] if args.stages else None
    try:
        if args.plan or not args.wait_for_salaries:
            lock, salary_path, info = slate_lock(args.date, args.salaries, args.slate)
        else:
            lock, salary_path, info = wait_for_slate(
                args.date, args.salaries, args.slate, args.wait_for_salaries)
    except AutosnapError as error:
        print(f"[!] {error}")
        return 1

    plan = stage_times(lock, stages)
    local = datetime.now().astimezone().tzinfo
    print(f"{args.date}  {len(info['games'])} games from {os.path.basename(salary_path)}")
    print(f"  first pitch {_clock12(lock)} ET  ({_clock12(lock.astimezone(local))} local)")
    print(f"\n  stage      fires at (ET)   local           relative")
    now = datetime.now(EASTERN)
    for stage, fire in plan:
        delta = (fire - now).total_seconds() / 60
        when = "PAST" if delta < 0 else f"in {int(delta // 60)}h{int(delta % 60):02d}m"
        print(f"  {stage:<10} {fire:%H:%M}           "
              f"{fire.astimezone(local):%H:%M}           "
              f"{when:<10} (lock -{STAGE_OFFSETS[stage]}m)")
    if args.plan:
        return 0

    for stage, fire in plan:
        if fire <= datetime.now(EASTERN):
            if args.skip_past:
                print(f"\n=== {stage}: time already passed, skipping ===")
                continue
            print(f"\n=== {stage}: time already passed, running now ===")
        else:
            print(f"\nwaiting for {stage} at {fire:%H:%M} ET "
                  f"({fire.astimezone(local):%H:%M} local)...", flush=True)
            _sleep_until(fire)
        try:
            run_stage(args.date, stage, slate=args.slate, salary_path=args.salaries,
                      refresh=not args.no_refresh, refresh_cmd=args.refresh_cmd,
                      dry_run=args.dry_run)
        except Exception as error:
            # One failed stage must not take the rest of the night with it -- the later
            # snapshots are the ones that decide the lineups.
            print(f"  [!] {stage} failed: {error}")

    print(f"\ndone. Review the night with: python -m dfs.snapshot --date {args.date} --list")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
