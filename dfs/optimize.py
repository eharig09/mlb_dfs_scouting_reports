"""Lineup optimizer CLI: `python -m dfs.optimize --date 2026-07-27 --n 20`.

Settings can live in `dfs_daily_files/optimizer.json` so a day's locks, excludes, and
stack rules persist between runs. Command-line flags always override the file.
"""

import argparse
import json
import math
import os
from datetime import datetime

import pandas as pd

from .optimizer import (
    STACK_EXPOSURE_AT,
    CONFLICT_MIN_HITTERS, CONFLICT_PENALTY, CORRELATED_OBJECTIVES, DEFAULT_MAX_OVERLAP,
    DEFAULT_RANDOMNESS, MIN_GAMES_REPRESENTED, OptimizerError, ROSTER, ROSTER_SIZE, optimize,
    stack_shapes,
)
from .exposure import format_exposure, player_exposure, team_exposure
from .naming import OUTPUT_ROOT, latest, resolve
from .pool import pool_path, read_pool, resolve_exposure, write_pool
from .profiling import format_report, profiler
from .salaries import canon_team
from .schedule import describe_postponed
from .scoring import DK_SALARY_CAP as DK_CAP
# drop_started now lives in dfs.slate so the candidate, field, contest and
# portfolio paths get the same filtering. Re-exported here because it was part
# of this module's surface.
from .slate import build_slate, drop_started
from .upload import (
    OUTPUT_PREFIX, SWAP_PREFIX, TEMPLATE_DIR, UploadError, build_upload, find_template,
    lineup_teams, parse_selection, parse_template, read_lineups, read_template,
    template_from_salaries, write_upload,
)

CONFIG_PATH = os.path.join("dfs_daily_files", "optimizer.json")
OUTPUT_DIR = "dfs_boards"

# How many times --stack-shape may cycle its team pairings to fill the requested count. The
# loop stops early the moment a full sweep adds nothing new, so this only bounds the
# pathological case where every pairing keeps returning lineups already seen.
MAX_SHAPE_SWEEPS = 25

# How many teams --stack-shape considers stacking, best first. Six keeps a two-stack shape
# tractable (30 orderings), but a one-stack shape has only as many combinations as teams, so
# widening it there is nearly free -- and the measurement in docs/benchmarks.md §13 says the
# ceiling comes from *some* team being stacked, not from it being a pre-anointed one.
DEFAULT_STACK_TEAMS = 6


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


def _parse_team_exposure(value, n_lineups):
    """'NYY:0-40,TOR:20-60' -> {'NYY': (0, 8), 'TOR': (4, 12)} for a 20-lineup set.

    Bounds are percentages of the set, matching how player exposure is written in a pool
    file, and are converted to lineup counts here so the optimizer only ever deals in
    counts. Either side may be blank -- 'NYY:-40' caps without a floor, 'TOR:20-' floors
    without a cap -- because the common cases are one-sided.
    """
    limits = {}
    for part in _split(value):
        team, _, span = part.partition(":")
        if not span.strip():
            raise OptimizerError(
                f"bad --team-exposure entry '{part}' (expected TEAM:MIN-MAX as percentages)")
        low_text, _, high_text = span.partition("-")

        def pct(text, default):
            text = text.strip().rstrip("%")
            if not text:
                return default
            try:
                return float(text)
            except ValueError:
                raise OptimizerError(f"bad --team-exposure percentage '{text}' in '{part}'")

        low, high = pct(low_text, 0.0), pct(high_text, 100.0)
        if not 0 <= low <= 100 or not 0 <= high <= 100 or low > high:
            raise OptimizerError(
                f"--team-exposure '{part}' must be 0-100 with MIN <= MAX")
        # Ceil the floor and floor the cap, so a stated bound is never quietly exceeded in
        # either direction by rounding.
        limits[canon_team(team)] = (int(math.ceil(low / 100.0 * n_lineups)),
                                    int(math.floor(high / 100.0 * n_lineups)))
    return limits


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
        # `slate` has to reach swap_file, not just the output path below. Without it a
        # night with two DK exports resolved the *swap* against an ambiguous slate while
        # still naming the output after the one that was asked for -- so `--slate early
        # --swap` failed with "several DK exports match", which is exactly the thing the
        # flag was passed to prevent.
        out_rows, report = swap_file(date, path, objective=objective, slate=slate)
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
    parser.add_argument("--stack-shape", help="Explore shapes instead, e.g. '5' (one 5-stack, "
                                              "team free) or '4-3'. Comma-separate several to "
                                              "spread lineups evenly across them: '5-3,5-2,5'.")
    parser.add_argument("--stack-teams", type=int, default=DEFAULT_STACK_TEAMS,
                        help=f"How many teams --stack-shape may stack, best first "
                             f"(default {DEFAULT_STACK_TEAMS}; 0 = every team on the slate).")
    parser.add_argument("--team-exposure", metavar="SPEC",
                        help="Cap or floor how much of the SET stacks each team, as "
                             "percentages: 'NYY:0-40,TOR:20-60'. Either side may be blank "
                             "('NYY:-40' caps only). A team counts as stacked when it "
                             f"supplies {STACK_EXPOSURE_AT}+ hitters, the same threshold the "
                             "exposure report uses. Caps limit clustering, not the players: "
                             "individual hitters from a capped team stay available.")
    parser.add_argument("--stack-at", type=int, default=STACK_EXPOSURE_AT, metavar="N",
                        help=f"Hitters from one team that count as a stack for "
                             f"--team-exposure (default {STACK_EXPOSURE_AT}).")
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
    parser.add_argument("--exposure-top", type=int, default=20, metavar="N",
                        help="How many players the exposure summary prints (default 20; "
                             "0 prints every one). The csv always holds the full set.")
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
    parser.add_argument("--profile", action="store_true",
                        help="Time each pipeline stage and write docs/benchmarks/profile_*.json.")
    args = parser.parse_args()

    with profiler.session("optimize", date=args.date, slate=args.slate,
                          enabled=args.profile, n=args.n, objective=args.objective):
        _run(args)
    if args.profile:
        print("\n=== pipeline profile ===")
        print(format_report(profiler.last_report))
        print(f"  -> {profiler.last_report.get('path')}")


def _run(args):
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

    team_exposure_spec = args.team_exposure or config.get("team_exposure")
    team_limits = _parse_team_exposure(team_exposure_spec, n_lineups) \
        if team_exposure_spec else {}
    # Carried across the stack-shape loop's separate solves, same as player appearances.
    team_stacks = {team: 0 for team in team_limits}
    if team_limits:
        print("team stack exposure: " + ", ".join(
            f"{team} {low}-{high} of {n_lineups}" for team, (low, high) in
            sorted(team_limits.items())) + f"  (stack = {args.stack_at}+ hitters)")

    stacks = _parse_stacks(args.stack) if args.stack else config.get("stack") or None
    shape = args.stack_shape or config.get("stack_shape")
    stack_teams = args.stack_teams
    if stack_teams == DEFAULT_STACK_TEAMS and config.get("stack_teams") is not None:
        stack_teams = int(config["stack_teams"])
    # 0 means "every team on the slate"; stack_shapes takes that as an uncapped slice.
    stack_teams = None if stack_teams is not None and stack_teams <= 0 else stack_teams
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
            combos = stack_shapes(players, shape, top_teams=stack_teams,
                                  focus_teams=focus_teams)
            if not focus_teams:
                free = sorted({ROSTER_SIZE - ROSTER["P"] - sum(combo.values())
                               for combo in combos})
                slots = (f"{free[0]}" if len(free) == 1
                         else f"{free[0]}–{free[-1]}")
                print(f"stack shape '{shape}': {len(combos)} team "
                      f"{'combination' if len(combos) == 1 else 'combinations'} over "
                      f"{len({t for combo in combos for t in combo})} teams — "
                      f"the solver picks which, and the remaining {slots} hitter "
                      f"{'slot' if free[-1] == 1 else 'slots'} plus both pitchers stay open.")
            lineups = []
            # Spread the requested lineups across shape combinations so the set explores
            # different stack pairings rather than re-solving the single best one.
            per_combo = max(1, n_lineups // max(1, min(len(combos), n_lineups)))
            # Exposure is satisfied over the whole set, not within one pairing, so the
            # running counts travel between solves. Without this each solve believes it is
            # the last chance to meet every minimum and forces the entire exposure list into
            # one lineup -- which on a pool of five capped pitchers is simply impossible, and
            # failed thirty times in a row without a word.
            appearances = {name: 0 for name in exposure}
            combo_errors = {}
            seen = set()
            # Cycle the pairings rather than making a single pass. One pass gives exactly
            # one lineup per combination -- 30 lineups for a request of 50 -- and, worse,
            # ends before the exposure shortfall ever becomes urgent, so minimums are simply
            # never met. Repeating until the request is filled fixes both.
            for sweep in range(MAX_SHAPE_SWEEPS):
                if len(lineups) >= n_lineups:
                    break
                progressed = False
                for index, combo in enumerate(combos):
                    if len(lineups) >= n_lineups:
                        break
                    try:
                        built, _, _ = optimize(
                            players, n_lineups=min(per_combo, n_lineups - len(lineups)),
                            objective=objective, locks=locks,
                            excludes=excludes, stacks=combo, max_overlap=max_overlap,
                            min_proj=args.min_proj, max_bust=args.max_bust,
                            max_hitters_per_team=max_from_team, randomness=randomness,
                            # Re-solving the same pairing with the same seed returns the same
                            # lineup, so each sweep gets its own.
                            seed=(None if args.seed is None
                                  else args.seed + sweep * 1009 + index),
                            stack_bonus=stack_bonus,
                            exposure=exposure, boosts=boosts,
                            conflict_penalty=conflict_penalty,
                            conflict_min_hitters=conflict_min_hitters,
                            focus_teams=focus_teams,
                            max_ownership=args.max_ownership,
                            total_lineups=n_lineups, prior_appearances=appearances,
                            prior_lineups=len(lineups),
                            team_exposure=team_limits, stack_at=args.stack_at,
                            # Stack counts have to carry across pairings for the same reason
                            # player appearances do: each solve sees only its own slice, so
                            # without this every pairing starts from zero and a cap that
                            # should bind over the set never binds at all.
                            prior_team_stacks=team_stacks,
                        )
                    except OptimizerError as error:
                        combo_errors[str(error)] = combo_errors.get(str(error), 0) + 1
                        continue        # this pairing is infeasible; try the next
                    for lineup in built:
                        key = tuple(sorted(lineup["players"]["Name"]))
                        if key in seen:
                            continue        # same pairing re-solved to the same ten
                        seen.add(key)
                        lineups.append(lineup)
                        progressed = True
                        for name in lineup["players"]["Name"]:
                            if name in appearances:
                                appearances[name] += 1
                        roster = lineup["players"]
                        taken = roster[roster["Roster"] != "P"]["Team"] \
                            .map(canon_team).value_counts()
                        for team in team_stacks:
                            if int(taken.get(team, 0)) >= args.stack_at:
                                team_stacks[team] += 1
                if not progressed:
                    break               # nothing new is reachable; stop rather than spin
            lineups = lineups[:n_lineups]
            # Every pairing failing is a fact about the constraints, not about the shape.
            if not lineups and combo_errors:
                print(f"[!] all {len(combos)} stack pairings for '{shape}' were infeasible:")
                for message, count in sorted(combo_errors.items(), key=lambda kv: -kv[1])[:3]:
                    print(f"    {count}x  {message}")
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
                team_exposure=team_limits, stack_at=args.stack_at,
            )
    except OptimizerError as error:
        print(f"[!] {error}")
        return

    for name in missing.get("locks", []):
        print(f"[!] lock '{name}' is not on this slate — ignored.")
    for name in missing.get("excludes", []):
        print(f"[!] exclude '{name}' is not on this slate — ignored.")

    # Unmet Min% values used to be reported here. They now belong to the exposure report at
    # the end of the run, which says the same thing against the delivered share and the rest
    # of the set -- printing it twice, once before the lineups and once after, only taught
    # the reader to skip both.
    if exposure and lineups and len(lineups) < n_lineups:
        print(f"\n[!] the set stopped at {len(lineups)} of the {n_lineups} requested, so "
              f"Min% targets were scaled to a set that never finished.")
        print(f"    Ask for fewer lineups, or loosen what is limiting the pool "
              f"(stack shape, excludes, incomplete slate).")

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

    # Exposure is a property of the whole set, so it can only be read here -- no single
    # lineup, and nothing during the solve, can show which players the set actually
    # committed to or which Min%/Max% requests survived.
    player_table = player_exposure(lineups, exposure=exposure, players=players)
    team_table = team_exposure(lineups)
    if not player_table.empty:
        print()
        print(format_exposure(player_table, team_table, len(lineups),
                              top=args.exposure_top))
        exposure_path, exposure_note = resolve("exposure", args.date, label,
                                               OUTPUT_ROOT, args.overwrite)
        # Both tables in one file: they answer the same question at different grain, and a
        # second file per night is another thing to find. The blank line and second header
        # keep it readable, and pandas reads either half back with skiprows.
        with open(exposure_path, "w", newline="", encoding="utf-8-sig") as handle:
            player_table.to_csv(handle, index=False, lineterminator="\n")
            handle.write("\n")
            team_table.to_csv(handle, index=False, lineterminator="\n")
        print(f"\nexposure -> {exposure_path}")
        if exposure_note:
            print(f"  [i] {exposure_note}")

    if args.upload is not None:
        # Re-read what was just written so the upload is built from the file on disk --
        # the same path `python -m dfs.upload` takes, rather than a second code path that
        # could drift from it.
        try:
            built = read_lineups(args.date, path)
            selection = parse_selection(args.upload, built.keys())
            try:
                template_path = find_template(args.template, date=args.date,
                                              teams=lineup_teams(built, selection))
                template, kind, slot_start, index = read_template(template_path)
                source = f"template {os.path.basename(template_path)}"
            except UploadError as error:
                # No template for tonight is not a dead end. DK's bulk template is a fixed
                # frame around the draft group's player list, and the salary export this
                # board was priced from *is* that player list -- same ids, same slate. So
                # rebuild the frame rather than making the night's lineups unenterable.
                # An explicit --template that failed is a different matter: the user named
                # a file, and quietly substituting something else would hide the mistake.
                salary_file = meta.get("salary_file")
                if args.template or not salary_file:
                    raise
                template, kind, slot_start, index = parse_template(
                    template_from_salaries(salary_file),
                    source=os.path.basename(salary_file),
                )
                source = f"rebuilt from {os.path.basename(salary_file)}"
                print(f"  [i] no DK template for {args.date} — {error}")
                print(f"  [i] rebuilding one from {os.path.basename(salary_file)}; its ids "
                      f"are DK's own for this slate. Bulk entry only — late swap still "
                      f"needs the real entries export.")
            out_rows, _ = build_upload(template, kind, slot_start, index, built, selection,
                                       date=args.date)
            upload_path, upload_note = resolve("upload", args.date, label,
                                               OUTPUT_ROOT, args.overwrite)
            out_path = write_upload(out_rows, upload_path)
            print(f"{len(selection)} lineup(s) -> {out_path}  ({source})")
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
            resolved["stack_teams"] = 0 if stack_teams is None else stack_teams
        if focus_teams:
            resolved["focus_teams"] = focus_teams
        os.makedirs(os.path.dirname(args.config) or ".", exist_ok=True)
        with open(args.config, "w", encoding="utf-8") as handle:
            json.dump(resolved, handle, indent=2)
        print(f"settings -> {args.config}")


if __name__ == "__main__":
    main()
