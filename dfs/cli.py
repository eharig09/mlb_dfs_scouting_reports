"""CLI entry point: `python -m dfs.cli --date 2026-07-24`."""

import argparse
import os
from datetime import datetime

from .adopt import DOWNLOADS_DIR, SEARCH_DIRS, adopt_for_date, describe
from .board import render_board
from .naming import OUTPUT_ROOT, resolve
from .profiling import format_report, profiler
from .salaries import list_salary_files, slate_label
from .schedule import describe_postponed
from .slate import REPORT_DATA_DIR, build_slate


def print_slates(candidates, date, ambiguous=False):
    """Show the DK exports on hand so a specific one can be named."""
    if not candidates:
        print(f"No DraftKings salary exports found for {date}. "
              f"Drop one in dfs_daily_files/.")
        return
    # Plain ASCII: the Windows console is cp1252 and cannot encode em dashes.
    if ambiguous:
        print(f"[!] {len(candidates)} salary files match {date} - name one with --slate "
              f"so the board is not priced off the wrong contest:\n")
    else:
        print(f"DraftKings exports found for {date}:\n")
    for info in candidates:
        print(f"  {info['name']}")
        print(f"      {slate_label(info)}")
        if info["games"]:
            print(f"      {', '.join(info['games'])}")
    print("\n  Pick one:  --slate <part of the filename>")
    print("  Rename freely - files are identified by contents, not name.")
    print("  Suggested:  DKSalaries_<date>_<label>.csv  e.g. DKSalaries_2026-07-27_main.csv")

OUTPUT_DIR = "dfs_boards"

CSV_COLUMNS = [
    "Type", "Name", "Team", "Opp", "Opp SP", "Slot", "Pos", "DK Pos", "Salary",
    "Proj", "Ceiling", "Floor", "Bust%", "Value", "Ceil Value", "Floor Value",
    "GPP", "CASH", "Edge", "Tier", "Role",
    "PA", "IP", "K", "W%", "ER", "HR", "SB", "Team Runs", "Matchup",
    "Proj Pct", "Ceiling Pct", "Value Pct", "Floor Value Pct", "Field Pct",
    "Lineup", "DK Avg", "DK Starting", "DK ID", "Why", "Risks",
]


def main():
    parser = argparse.ArgumentParser(description="DraftKings DFS recommender for a cached slate.")
    parser.add_argument("--date", default=datetime.today().strftime("%Y-%m-%d"),
                        help="Slate date in YYYY-MM-DD format. Defaults to today.")
    parser.add_argument("--salaries",
                        help="Path to DKSalaries.csv. Defaults to a DK export found in "
                             "dfs_daily_files/, then dfs/salaries/.")
    parser.add_argument("--slate",
                        help="Pick between multiple DK exports by filename substring, "
                             "e.g. 'main' or 'early'.")
    parser.add_argument("--list-slates", action="store_true",
                        help="Show every DK export found for --date and stop.")
    parser.add_argument("--data-dir", default=REPORT_DATA_DIR, help="Cached report-data directory.")
    parser.add_argument("--output-dir", default=OUTPUT_ROOT,
                        help=f"Root for night folders. Default: {OUTPUT_ROOT}/<date>/.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing files instead of writing a new .rN version.")
    parser.add_argument("--downloads", action="store_true",
                        help=f"Also look in {DOWNLOADS_DIR} for an unfiled DK export.")
    parser.add_argument("--no-adopt", action="store_true",
                        help="Do not rename misnamed DK downloads into the convention.")
    parser.add_argument("--top-pitchers", type=int, default=14, help="Pitchers shown on the board.")
    parser.add_argument("--top-per-position", type=int, default=8, help="Hitters shown per position.")
    parser.add_argument("--print", dest="echo", action="store_true", help="Print the board to stdout.")
    parser.add_argument("--profile", action="store_true",
                        help="Time each pipeline stage and write docs/benchmarks/profile_*.json.")
    args = parser.parse_args()

    with profiler.session("board", date=args.date, slate=args.slate, enabled=args.profile):
        _run(args)
    if args.profile:
        print("\n=== pipeline profile ===")
        print(format_report(profiler.last_report))
        print(f"  -> {profiler.last_report.get('path')}")


def _run(args):

    # File any stray download before resolving inputs, so a DK export that was never
    # renamed still gets found -- and gets a proper name, which is what stops it from
    # being re-adopted every night.
    if not args.no_adopt:
        search = SEARCH_DIRS + ([DOWNLOADS_DIR] if args.downloads else [])
        adopted, problems = adopt_for_date(args.date, search_dirs=search)
        if adopted:
            print(f"Filed {len(adopted)} unnamed DraftKings download(s):")
            for line in describe(adopted):
                print(line)
            print()
        for problem in problems:
            print(f"[!] {problem}")

    if args.list_slates:
        print_slates(list_salary_files(args.date), args.date)
        return

    players, stacks, meta = build_slate(args.date, salary_path=args.salaries,
                                        data_dir=args.data_dir, slate=args.slate)
    if meta.get("ambiguous_slates"):
        print_slates(meta["ambiguous_slates"], args.date, ambiguous=True)
        return
    with profiler.stage("render", players=len(players)):
        board = render_board(players, stacks, meta,
                             top_pitchers=args.top_pitchers,
                             top_per_position=args.top_per_position)

    label = meta.get("slate_label")
    notes = []

    board_path, note = resolve("board", args.date, label, args.output_dir, args.overwrite)
    notes.append(note)
    with open(board_path, "w", encoding="utf-8") as handle:
        handle.write(board)

    outputs = [board_path]
    if not players.empty:
        csv_path, note = resolve("slate", args.date, label, args.output_dir, args.overwrite)
        notes.append(note)
        columns = [c for c in CSV_COLUMNS if c in players.columns]
        # utf-8-sig so Excel renders accented names correctly on open.
        players[columns].to_csv(csv_path, index=False, encoding="utf-8-sig")
        outputs.append(csv_path)

        if not stacks.empty:
            stacks_path, note = resolve("stacks", args.date, label, args.output_dir, args.overwrite)
            notes.append(note)
            stacks.to_csv(stacks_path, index=False, encoding="utf-8-sig")
            outputs.append(stacks_path)

    if args.echo:
        print(board)

    print(f"\n{len(players)} players across {len(meta['games'])} games"
          f"{' (salaries matched)' if meta.get('has_salary') else ' (no salaries)'}"
          f"  [slate: {label}]")
    for path in outputs:
        print(f"  -> {path}")
    # A versioned write means an earlier run of this same slate is still on disk. Say so:
    # silently writing next to it is how you end up uploading from the wrong file.
    for note in [n for n in notes if n]:
        print(f"  [i] {note}")

    # Plain ASCII throughout: the Windows console is cp1252 and cannot encode emoji.
    # Postponements print before the other warnings and outside the elif chain: a called
    # game removes real players from the pool, and that is worth seeing even alongside an
    # incomplete slate or a stale salary file.
    if meta.get("postponed"):
        print(f"\n[!] POSTPONED - {describe_postponed(meta['postponed'])}")
        print(f"    {meta['postponed_players']} player(s) removed from the pool.")
    if meta.get("schedule_error"):
        print(f"\n[!] Could not check for postponements: {meta['schedule_error']}")
        print("    A game called off after the salary file was downloaded would not be caught.")

    if meta.get("environment_error"):
        print(f"\n[!] ENVIRONMENT PROBLEM - not missing data. "
              f"{meta['cached_but_unreadable']} cached games are on disk but unreadable:")
        print(f"    {meta['environment_error']}")
        print("    Do NOT regenerate the reports. Use the project interpreter:")
        print("      .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt")
        print(f"      .\\.venv\\Scripts\\python.exe -m dfs.cli --date {args.date}")
    elif meta.get("stale_salaries"):
        print(f"\n[!] WRONG SALARY FILE - {meta['salary_file']} prices the "
              f"{meta['salary_date']} slate, not {args.date}.")
        print("    Salaries, value, and tiers on this board are meaningless.")
    elif meta.get("missing_games"):
        print(f"\n[!] INCOMPLETE SLATE - priced by DK but missing from the report cache: "
              f"{', '.join(meta['missing_games'])}")
        for game in meta["missing_games"]:
            away, _, home = game.partition("@")
            print(f"     python scouting_report.py --date {args.date} "
                  f"--away-team {away} --home-team {home} --format both")


if __name__ == "__main__":
    main()
