"""Build an NFL slate report.

    .venv/Scripts/python.exe -m nfl.cli --salaries "nfl/nfl_dfs/nfl_daily_files/DKSalaries (39).csv"

Writes a workbook next to the salary file's dated folder, with a Visuals tab and one tab
per read: board, red zone, matchup, cold start.
"""

import argparse
import io
import os
from datetime import datetime

import pandas as pd
import requests

NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"


def _parquet(tag, name, session):
    import pyarrow.parquet as pq
    response = session.get(f"{NFLVERSE}/{tag}/{name}", timeout=300)
    if not response.ok:
        return None
    return pq.read_table(io.BytesIO(response.content)).to_pandas()


def _cold_start_priors(season, session):
    """Roster universe + priors. Returns an empty frame if nflverse is unreachable.

    A slate report is still worth writing without it — the PFF half of the board does not
    depend on it — so this degrades rather than aborts.
    """
    from nfl import coldstart

    roster = _parquet("rosters", f"roster_{season}.parquet", session)
    if roster is None:
        print(f"[!] no roster published for {season}; cold-start priors skipped")
        return pd.DataFrame()
    depth = _parquet("depth_charts", f"depth_charts_{season}.parquet", session)

    frames = []
    for year in (season - 2, season - 1):
        weekly = _parquet("stats_player", f"stats_player_week_{year}.parquet", session)
        if weekly is not None:
            frames.append(weekly[weekly["season_type"] == "REG"])
    history = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return coldstart.build_priors(coldstart.roster_universe(roster, depth), history)


def build(salaries, season=None, stats_season=2025, week=None, output_dir=None,
          root=".", no_figures=False):
    from nfl import report, report_visuals, slate

    season = season or (datetime.today().year if datetime.today().month >= 3
                        else datetime.today().year - 1)
    session = requests.Session()
    print(f"Building slate report from {os.path.basename(salaries)}")

    priors = _cold_start_priors(season, session)
    if not priors.empty:
        print(f"  cold-start priors: {len(priors):,} players")

    board = slate.build_slate(salaries, season=stats_season, week=week,
                              priors=priors, root=root)
    if board.empty:
        raise SystemExit("no skill players found in the salary file")
    counts = board["basis"].value_counts().to_dict()
    print(f"  board: {len(board)} skill players  {counts}")

    teams = "-".join(sorted(board["team"].dropna().unique()))
    stamp = datetime.today().strftime("%Y-%m-%d")
    output_dir = output_dir or os.path.join("nfl", "nfl_reports", stamp)
    os.makedirs(output_dir, exist_ok=True)

    figures = []
    if not no_figures:
        figures = report_visuals.build_figures(
            board, os.path.join(output_dir, "figures"), label=teams)
        print(f"  figures: {len(figures)}")

    label = f"{teams}  ·  {stamp}" + (f"  ·  week {week}" if week else "")
    path = report.write_workbook(board, os.path.join(output_dir, f"slate_{stamp}_{teams}.xlsx"),
                                 slate_label=label, figures=figures)
    print(f"  -> {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Build an NFL DFS slate report.")
    parser.add_argument("--salaries", required=True, help="DKSalaries CSV for the slate.")
    parser.add_argument("--season", type=int, default=None,
                        help="Roster/depth-chart season. Defaults to the current one.")
    parser.add_argument("--stats-season", type=int, default=2025,
                        help="Season the PFF grade and red-zone files describe.")
    parser.add_argument("--week", type=int, default=None,
                        help="Week for strength of schedule. Omit for the season average.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    build(args.salaries, season=args.season, stats_season=args.stats_season,
          week=args.week, output_dir=args.output_dir, no_figures=args.no_figures)


if __name__ == "__main__":
    main()
