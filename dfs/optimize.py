"""Lineup optimizer CLI: `python -m dfs.optimize --date 2026-07-27 --n 20`.

Settings can live in `dfs_daily_files/optimizer.json` so a day's locks, excludes, and
stack rules persist between runs. Command-line flags always override the file.
"""

import argparse
import json
import os
from datetime import datetime

import pandas as pd

from .optimizer import (
    CONFLICT_MIN_HITTERS, CONFLICT_PENALTY, CORRELATED_OBJECTIVES, DEFAULT_MAX_OVERLAP,
    DEFAULT_RANDOMNESS, MIN_GAMES_REPRESENTED, OptimizerError, ROSTER_SIZE, optimize,
    stack_shapes,
)
from .naming import OUTPUT_ROOT, latest, resolve
from .pool import pool_path, read_pool, resolve_exposure, write_pool
from .salaries import canon_team
from .schedule import describe_postponed
from .scoring import DK_SALARY_CAP as DK_CAP
from .slate import build_slate, started_players
from .upload import (
    OUTPUT_PREFIX, SWAP_PREFIX, TEMPLATE_DIR, UploadError, build_upload, find_template,
    lineup_teams, parse_selection, read_lineups, read_template, write_upload,
)

CONFIG_PATH = os.path.join("dfs_daily_files", "optimizer.json")
OUTPUT_DIR = "dfs_boards"


def _split(value):
    if not value:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_stacks(value):
    """'CWS:4,HOU:3' -> {'CWS': 4, 'HOU': 3}"""
    stacks = {}
    for part in _split(value):
        team, _, count = part.partition(":")
        if not count.strip().isdigit():
            raise OptimizerError(f"bad --stack entry '{part}' (expected TEAM:COUNT)")
        stacks[canon_team(team)] = int(count)
    return stacks


def load_config(path):
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as error:
        print(f"[!] could not read {path}: {error}")
        return {}


def format_lineup(lineup, index):
    frame = lineup["players"]
    lines = [
        f"--- Lineup {index}  |  {lineup['objective']}  "
        f"proj {lineup['proj']}  ceil {lineup['ceiling']}  floor {lineup['floor']}  "
        f"${lineup['salary']:,} (${DK_CAP - lineup['salary']:,} left)"
    ]
    stack = ", ".join(f"{team} x{count}" for team, count in
                      sorted(lineup["teams"].items(), key=lambda kv: -kv[1]) if count >= 2)
    if stack:
        lines.append(f"    stacks: {stack}")
    for pitcher, team, count in lineup.get("conflicts", []):
        lines.append(f"    [!] {pitcher} pitches against your own {team} x{count}")
    for _, row in frame.iterrows():
        lines.append(
            f"    {str(row['Roster']):<3} {str(row['Name'])[:24]:<24} {str(row['Team']):<4} "
            f"${int(row['Salary']):>6,}  proj {float(row['Proj']):5.1f}  "
            f"ceil {float(row['Ceiling']):5.1f}  {str(row.get('Tier') or '')}"
        )
    return "\n".join(lines)


def run_swap(date, objective, path=None, slate=None, overwrite=False):
    """Late-swap pass: refill dead slots in the filled upload file. Never raises."""
    from .lateswap import SwapError, format_report, swap_file

    try:
        out_rows, report = swap_file(date, path, objective=objective)
    except (SwapError, UploadError, OptimizerError) as error:
        print(f"[!] late swap: {error}")
        return None
    print(format_report(report))
    if out_rows is None:
        print("nothing to swap — every slot is still playable.")
        return None
    # Always a new file, never the one that was read. The file being re-read is the record
    # of what is actually entered at DK, and overwriting it would destroy the only copy of
    # that if a swap turns out to be wrong.
    out_path, note = resolve("swap", date, slate, OUTPUT_ROOT, overwrite)
    write_upload(out_rows, out_path)
    print(f"-> {out_path}")
    if note:
        print(f"  [i] {note}")
    return out_path


def drop_started(players, meta, date, allow=False):
    """Drop players whose game has already begun -- they cannot be drafted at all.

    The slate removes postponed games but knows nothing about start times, so a run late in
    the evening would otherwise build around a player who is already batting and hand back
    an entry DK will not take.

    Resolved per game, not per team: on a doubleheader the opener can be in progress while
    the nightcap -- the game the main slate is priced on -- is hours away and completely
    draftable.

    A slate where *every* game has started is a review or backtest, not a live build, so
    nothing is dropped there: filtering would empty the pool and fail with a message about
    the wrong thing. (`dfs.review` calls the optimizer directly and never comes through
    here, which is why that path is untouched by any of this.)
    """
    if allow:
        return players, []
    locked, elsewhere, error = started_players(players, meta, date)
    if error:
        return players, [f"[!] could not check start times: {error}. Players from games "
                         f"already under way may still be in the pool — check before entering."]
    if locked.all():
        return players, []              # nothing live at all: a review or backtest

    notes = []
    if elsewhere.any():
        # Worth naming individually. This one looks like a bargain rather than a mistake --
        # a top-salary arm with a full projection whose game, per DK, has not started.
        for name in players[elsewhere]["Name"]:
            notes.append(f"[!] {name} removed — already pitching the earlier game of a "
                         f"doubleheader. DK prices the whole staff against the nightcap, so "
                         f"he is listed as available and would score nothing.")
    if locked.any():
        gone = sorted(set(players[locked]["Game"].dropna()))
        notes.append(f"[!] {int(locked.sum())} player(s) removed — "
                     f"{', '.join(gone)} already under way and cannot be drafted.")
    drop = locked | elsewhere
    if not drop.any():
        return players, notes

    remaining = players[~drop]
    games = remaining["Game"].dropna().nunique() if "Game" in remaining.columns else 0
    if games < MIN_GAMES_REPRESENTED:
        notes.append(f"    Only {games} game(s) left on the board; DK needs "
                     f"{MIN_GAMES_REPRESENTED}. Use --allow-started to build anyway.")
    return remaining.copy(), notes


def main():
    parser = argparse.ArgumentParser(description="DraftKings Classic lineup optimizer.")
    parser.add_argument("--date", default=datetime.today().strftime("%Y-%m-%d"))
    parser.add_argument("--salaries", help="Path to DKSalaries.csv.")
    parser.add_argument("--slate", help="Pick between multiple DK exports by filename "
                                        "substring, e.g. 'main' or 'early'.")
    parser.add_argument("--config", default=CONFIG_PATH, help="JSON settings file.")
    parser.add_argument("--n", type=int, help="How many lineups to generate.")
    parser.add_argument("--objective", choices=["ceiling", "proj", "floor", "leverage"],
                        help="What to maximize. ceiling=GPP, floor=cash, proj=balanced, "
                             "leverage=ceiling discounted by projected ownership.")
    parser.add_argument("--max-ownership", type=float, metavar="PCT",
                        help="Cap the lineup's total projected ownership, e.g. 120. Ten "
                             "chalk plays sum near 300; a real field-beating lineup is "
                             "usually well under 150.")
    parser.add_argument("--lock", help="Comma-separated players who must appear.")
    parser.add_argument("--exclude", help="Comma-separated players to never use.")
    parser.add_argument("--stack", help="Explicit stacks, e.g. 'CWS:4,HOU:3'.")
    parser.add_argument("--stack-shape", help="Explore shapes instead, e.g. '4-3' or '5-3'.")
    parser.add_argument("--focus-teams", help="Build stacks only from these teams, e.g. "
                                              "'CIN,NYY,SD'. Other teams still fill the "
                                              "leftover slots and pitchers stay open.")
    parser.add_argument("--max-overlap", type=int,
                        help=f"Max shared players between lineups (default {DEFAULT_MAX_OVERLAP}; "
                             f"{ROSTER_SIZE - 1} blocks exact duplicates only).")
    parser.add_argument("--min-proj", type=float, help="Drop players below this projection.")
    parser.add_argument("--max-bust", type=float, help="Drop players above this bust%%.")
    parser.add_argument("--max-from-team", type=int, help="Max hitters from one team (DK allows 5).")
    parser.add_argument("--randomness", type=float,
                        help=f"Objective jitter (default {DEFAULT_RANDOMNESS}). Explores "
                             f"near-optimal lineups; 0 for the single best lineup.")
    parser.add_argument("--seed", type=int,
                        help="Seed for reproducible randomness. Without it each run differs.")
    parser.add_argument("--write-pool", action="store_true",
                        help="Export an editable pool CSV (Lock/Exclude/Boost) and stop. "
                             "Merges with the existing file, keeping your edits.")
    parser.add_argument("--reset-pool", action="store_true",
                        help="With --write-pool, rebuild the file from scratch and discard "
                             "every edit in it.")
    parser.add_argument("--pool", nargs="?", const="", metavar="PATH",
                        help="Read locks/excludes/boosts from a pool CSV. Bare flag uses "
                             "dfs_daily_files/pool_<date>.csv.")
    parser.add_argument("--conflict-penalty", type=float,
                        help=f"Tax on a pitcher rostered against your own stack, as a "
                             f"multiple of his projection (default {CONFLICT_PENALTY}). "
                             f"0 allows the pairing; higher makes it prohibitive.")
    parser.add_argument("--conflict-min-hitters", type=int,
                        help=f"Opposing hitters that make a stack 'prominent' enough to tax "
                             f"(default {CONFLICT_MIN_HITTERS}).")
    parser.add_argument("--no-stack-bonus", action="store_true",
                        help="Disable the stack correlation bonus (on by default for "
                             "ceiling/proj objectives).")
    parser.add_argument("--save-config", action="store_true",
                        help="Write the resolved settings back to the config file.")
    parser.add_argument("--upload", nargs="?", const="all", metavar="SELECTION",
                        help="Also write a DK upload file. Bare flag takes every lineup; "
                             "pass a selection like '1,3,5-8'.")
    parser.add_argument("--template", help="DK upload file to fill (see --upload).")
    parser.add_argument("--allow-started", action="store_true",
                        help="Keep players whose game has already begun. They cannot be "
                             "drafted, so this is for analysis and backtests only.")
    parser.add_argument("--swap", action="store_true",
                        help="Re-read the filled upload file and refill any dead slots "
                             "(postponed, scratched, empty). On its own it skips generation "
                             "entirely, so tonight's entries are never rebuilt; with "
                             "--upload it runs as a final step. Writes "
                             f"{OUTPUT_ROOT}/<date>/swap_<slate>.csv.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing files instead of writing a new .rN version.")
    args = parser.parse_args()

    # Swapping without uploading is the evening command: check what broke and refill it.
    # Generation is skipped outright rather than done and discarded -- re-running the
    # optimizer would overwrite the lineup file that records what is actually entered.
    if args.swap and args.upload is None:
        run_swap(args.date, args.objective or "ceiling", args.template,
                 slate=args.slate, overwrite=args.overwrite)
        return

    config = load_config(args.config)

    def setting(name, default):
        value = getattr(args, name, None)
        if value not in (None, ""):
            return value
        return config.get(name, default)

    n_lineups = int(setting("n", 5))
    objective = setting("objective", "ceiling")
    locks = _split(args.lock) or config.get("lock", [])
    excludes = _split(args.exclude) or config.get("exclude", [])
    max_overlap = setting("max_overlap", DEFAULT_MAX_OVERLAP)
    randomness = float(setting("randomness", DEFAULT_RANDOMNESS))
    max_from_team = int(setting("max_from_team", 5))
    conflict_penalty = float(setting("conflict_penalty", CONFLICT_PENALTY))
    conflict_min_hitters = int(setting("conflict_min_hitters", CONFLICT_MIN_HITTERS))

    players, _, meta = build_slate(args.date, salary_path=args.salaries, slate=args.slate)
    if meta.get("ambiguous_slates"):
        from .cli import print_slates
        print_slates(meta["ambiguous_slates"], args.date, ambiguous=True)
        return
    if players.empty:
        print(f"No slate data for {args.date}. Build the board first: "
              f"python -m dfs.cli --date {args.date}")
        return
    if not meta.get("has_salary"):
        print("[!] No salaries matched — the optimizer needs prices. "
              "Drop DKSalaries.csv in dfs_daily_files/.")
        return
    if meta.get("postponed"):
        print(f"[!] POSTPONED — {describe_postponed(meta['postponed'])}. "
              f"{meta['postponed_players']} player(s) removed from the pool and cannot be rostered.")
    if meta.get("schedule_error"):
        print(f"[!] Could not check for postponements: {meta['schedule_error']}. "
              f"A game called off after the salary file was downloaded would not be caught.")
    if meta.get("missing_games"):
        print(f"[!] INCOMPLETE SLATE — {', '.join(meta['missing_games'])} priced but not cached. "
              f"Those players cannot be optimized into a lineup.")

    players, started_notes = drop_started(players, meta, args.date, allow=args.allow_started)
    for note in started_notes:
        print(note)

    if args.write_pool:
        path, report = write_pool(players, args.date, merge=not args.reset_pool,
                                  slate=meta.get("slate_label"))
        print(f"pool -> {path}")
        if report["merged"]:
            source = f" ({report['from']})" if report.get("from") else ""
            print(f"  merged with the existing file{source}: {len(report['carried'])} edit(s) kept, "
                  f"{report['new']} player(s) new to the slate.")
            if report["orphaned"]:
                print(f"  [!] {len(report['orphaned'])} edited player(s) are no longer on "
                      f"the slate: {', '.join(report['orphaned'])}.")
                print("      Their rows are kept at the bottom with the slate columns blank, "
                      "so nothing is lost if the game is only waiting to be cached.")
        elif args.reset_pool:
            print("  existing edits discarded (--reset-pool).")
        print("  Edit Lock / Exclude / Boost / Min% / Max% in Excel, then re-run with --pool.")
        print("  Lock, Exclude : any of 1, y, yes, x, true")
        print("  Boost         : 1.25 favours, 0.8 fades, or a range like 1.1-1.3")
        print("                  (a range is redrawn per lineup, spreading the player out)")
        print("  Min%, Max%    : share of lineups the player may appear in, e.g. 30 and 60")
        return

    boosts, exposure = {}, {}
    if args.pool is not None:
        path = args.pool or pool_path(args.date, slate=meta.get("slate_label"))
        if not os.path.exists(path):
            print(f"[!] no pool file at {path} — create one with --write-pool.")
            return
        pool_locks, pool_excludes, boosts, exposure_pct = read_pool(path)
        # Pool file and CLI flags combine rather than compete.
        locks = list(dict.fromkeys(locks + pool_locks))
        excludes = list(dict.fromkeys(excludes + pool_excludes))
        exposure = resolve_exposure(players, exposure_pct, n_lineups)
        ranged = sum(1 for low, high in boosts.values() if high > low)
        print(f"pool {path}: {len(pool_locks)} locked, {len(pool_excludes)} excluded, "
              f"{len(boosts)} boosted ({ranged} ranged), {len(exposure)} exposure-capped")

    stacks = _parse_stacks(args.stack) if args.stack else config.get("stack") or None
    shape = args.stack_shape or config.get("stack_shape")
    focus_teams = _split(args.focus_teams) or config.get("focus_teams") or []
    if focus_teams:
        # Checked here rather than left to the solver: a mistyped code would otherwise just
        # quietly shrink the stack pool, and the run would look like it worked.
        focus_teams = list(dict.fromkeys(canon_team(team) for team in focus_teams))
        on_slate = {canon_team(team) for team
                    in players[players["Type"] == "H"]["Team"].dropna()}
        for team in focus_teams:
            if team not in on_slate:
                print(f"[!] focus team '{team}' has no hitters on this slate — ignored.")
        focus_teams = [team for team in focus_teams if team in on_slate]
        if not focus_teams:
            print("[!] no focus team is on this slate. Drop --focus-teams or fix the codes.")
            return

    # Naming teams or shapes is an unambiguous request to stack, so it turns the correlation
    # bonus on even under an objective that would not stack by itself. Without this,
    # `--objective floor --focus-teams LAD,WSH` silently produced no stacks at all: floor is
    # not a correlated objective, so the bonus defaulted off, and focus_teams only *confines*
    # a bonus that was never applied. The run looked like it worked.
    asked_to_stack = bool(focus_teams or stacks or shape)
    if args.no_stack_bonus:
        stack_bonus = False
    elif asked_to_stack:
        stack_bonus = True
    else:
        stack_bonus = None

    if focus_teams:
        if args.no_stack_bonus:
            print(f"[!] --focus-teams {', '.join(focus_teams)} ignored: --no-stack-bonus "
                  f"turns off the correlation bonus that builds stacks.")
        else:
            extra = ("" if args.objective in CORRELATED_OBJECTIVES
                     else f" (stack bonus forced on; '{args.objective}' does not stack by default)")
            print(f"stacking only from: {', '.join(focus_teams)}{extra}")

    try:
        if shape and not stacks:
            combos = stack_shapes(players, shape, focus_teams=focus_teams)
            lineups = []
            # Spread the requested lineups across shape combinations so the set explores
            # different stack pairings rather than re-solving the single best one.
            per_combo = max(1, n_lineups // max(1, min(len(combos), n_lineups)))
            for combo in combos:
                if len(lineups) >= n_lineups:
                    break
                try:
                    built, _, _ = optimize(
                        players, n_lineups=per_combo, objective=objective, locks=locks,
                        excludes=excludes, stacks=combo, max_overlap=max_overlap,
                        min_proj=args.min_proj, max_bust=args.max_bust,
                        max_hitters_per_team=max_from_team, randomness=randomness,
                        seed=args.seed, stack_bonus=stack_bonus,
                        exposure=exposure, boosts=boosts,
                        conflict_penalty=conflict_penalty,
                        conflict_min_hitters=conflict_min_hitters,
                        focus_teams=focus_teams,
                        max_ownership=args.max_ownership,
                    )
                    lineups.extend(built)
                except OptimizerError:
                    continue        # this pairing is infeasible; try the next
            lineups = lineups[:n_lineups]
            missing = {"locks": [], "excludes": []}
        else:
            lineups, _, missing = optimize(
                players, n_lineups=n_lineups, objective=objective, locks=locks,
                excludes=excludes, stacks=stacks, max_overlap=max_overlap,
                min_proj=args.min_proj, max_bust=args.max_bust,
                max_hitters_per_team=max_from_team, randomness=randomness,
                seed=args.seed, stack_bonus=stack_bonus,
                exposure=exposure, boosts=boosts,
                conflict_penalty=conflict_penalty,
                conflict_min_hitters=conflict_min_hitters,
                focus_teams=focus_teams,
                max_ownership=args.max_ownership,
            )
    except OptimizerError as error:
        print(f"[!] {error}")
        return

    for name in missing.get("locks", []):
        print(f"[!] lock '{name}' is not on this slate — ignored.")
    for name in missing.get("excludes", []):
        print(f"[!] exclude '{name}' is not on this slate — ignored.")

    if not lineups:
        print("[!] No feasible lineup. Loosen locks, stacks, or thresholds.")
        return

    # --focus-teams only *prefers* a stack: the correlation bonus scales with the objective's
    # own magnitude, so under a small-valued objective like floor it can be worth a fraction
    # of a point and lose to individual merit. Saying so beats handing back twenty lineups
    # that quietly ignored the flag.
    if focus_teams and not stacks:
        biggest = 0
        for lineup in lineups:
            hitters = lineup["players"]
            hitters = hitters[hitters["Roster"] != "P"]
            counts = hitters["Team"].map(canon_team).value_counts()
            biggest = max(biggest, max((counts.get(t, 0) for t in focus_teams), default=0))
        if biggest < 3:
            print(f"[!] --focus-teams {', '.join(focus_teams)} produced no stack of 3+ "
                  f"(largest was {biggest}).")
            print(f"    The correlation bonus scales with the objective's magnitude, and "
                  f"'{args.objective}' values are small enough that it loses to individual "
                  f"merit.")
            print(f"    For a guaranteed stack use the hard constraint instead:")
            print(f"      --stack \"{focus_teams[0]}:4"
                  + (f",{focus_teams[1]}:3" if len(focus_teams) > 1 else "") + "\"")

    print()
    for i, lineup in enumerate(lineups, start=1):
        print(format_lineup(lineup, i))
        print()


    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = []
    for i, lineup in enumerate(lineups, start=1):
        for _, row in lineup["players"].iterrows():
            rows.append({
                "Lineup": i, "Roster": row["Roster"], "Order": row.get("Slot"),
                "Name": row["Name"], "Team": row["Team"],
                "Opp": row.get("Opp"), "Salary": row["Salary"], "Proj": row["Proj"],
                "Ceiling": row["Ceiling"], "Floor": row.get("Floor"),
                "Tier": row.get("Tier"), "Role": row.get("Role"), "DK ID": row.get("DK ID"),
                "Lineup Salary": lineup["salary"], "Lineup Proj": lineup["proj"],
                "Lineup Ceiling": lineup["ceiling"],
            })
    label = meta.get("slate_label")
    path, note = resolve("lineups", args.date, label, OUTPUT_ROOT, args.overwrite)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"{len(lineups)} lineups -> {path}")
    if note:
        print(f"  [i] {note}")

    if args.upload is not None:
        # Re-read what was just written so the upload is built from the file on disk --
        # the same path `python -m dfs.upload` takes, rather than a second code path that
        # could drift from it.
        try:
            built = read_lineups(args.date, path)
            selection = parse_selection(args.upload, built.keys())
            template_path = find_template(args.template, date=args.date,
                                          teams=lineup_teams(built, selection))
            template, kind, slot_start, index = read_template(template_path)
            out_rows, _ = build_upload(template, kind, slot_start, index, built, selection,
                                       date=args.date)
            upload_path, upload_note = resolve("upload", args.date, label,
                                               OUTPUT_ROOT, args.overwrite)
            out_path = write_upload(out_rows, upload_path)
            print(f"{len(selection)} lineup(s) -> {out_path}  (template {os.path.basename(template_path)})")
            if upload_note:
                print(f"  [i] {upload_note}")
            if args.swap:
                print()
                run_swap(args.date, objective, out_path, slate=label, overwrite=args.overwrite)
        except UploadError as error:
            # The lineups themselves are already saved, so a template problem is worth
            # reporting without discarding the run.
            print(f"[!] upload file not written: {error}")

    if args.save_config:
        resolved = {
            "n": n_lineups, "objective": objective, "lock": locks, "exclude": excludes,
            "max_overlap": max_overlap, "randomness": randomness,
            "max_from_team": max_from_team, "conflict_penalty": conflict_penalty,
            "conflict_min_hitters": conflict_min_hitters,
        }
        if stacks:
            resolved["stack"] = stacks
        if shape:
            resolved["stack_shape"] = shape
        if focus_teams:
            resolved["focus_teams"] = focus_teams
        os.makedirs(os.path.dirname(args.config) or ".", exist_ok=True)
        with open(args.config, "w", encoding="utf-8") as handle:
            json.dump(resolved, handle, indent=2)
        print(f"settings -> {args.config}")


if __name__ == "__main__":
    main()
