"""Actual DK points for every player in every regular-season game, 2024-2026.

The study needs an outcome column, and DK points are not derivable from statcast alone --
runs scored and stolen bases never appear in the pitch table. So the real box score is
pulled per game and reduced to the handful of counting stats DK scores, using the same
scoring tables the rest of the repo uses (dfs.scoring), so numbers here line up with
dfs.backtest.

Output: one row per (game_pk, player), with the DK line and enough raw stats to recompute
points under a different scoring table later.
"""

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

sys.path.insert(0, os.getcwd())
from dfs.scoring import hitter_points, pitcher_points  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "boxscores"
SEASONS = (2024, 2025, 2026)
WORKERS = 12

_local = threading.local()


def session():
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    return _local.session


def _ip_to_float(value):
    """MLB innings strings are base-3 after the decimal: '5.2' is five and two thirds."""
    try:
        whole, _, outs = str(value).partition(".")
        return int(whole or 0) + int(outs or 0) / 3.0
    except (TypeError, ValueError):
        return 0.0


def season_games(year):
    """Regular-season game ids that reached a final state."""
    response = requests.get(
        "https://statsapi.mlb.com/api/v1/schedule",
        params={"sportId": 1, "season": year, "gameType": "R",
                "startDate": f"{year}-03-01", "endDate": f"{year}-11-15"},
        timeout=60,
    )
    response.raise_for_status()
    games = []
    for date in response.json().get("dates", []):
        for game in date.get("games", []):
            if (game.get("status") or {}).get("codedGameState") == "F":
                games.append((int(game["gamePk"]), date["date"]))
    return games


def game_rows(game_pk, game_date):
    """Reduce one boxscore to DK lines. Retries once -- the API drops requests under load."""
    for attempt in range(3):
        try:
            response = session().get(
                f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore", timeout=45)
            response.raise_for_status()
            box = response.json()
            break
        except Exception:
            if attempt == 2:
                return []
    rows = []
    for side in ("away", "home"):
        team = (box.get("teams") or {}).get(side) or {}
        team_abbr = ((team.get("team") or {}).get("abbreviation")) or ""
        for entry in (team.get("players") or {}).values():
            player_id = ((entry.get("person") or {}).get("id"))
            if not player_id:
                continue
            stats = entry.get("stats") or {}
            order = entry.get("battingOrder")
            batting = stats.get("batting") or {}
            pitching = stats.get("pitching") or {}

            if batting.get("atBats") is not None or batting.get("baseOnBalls"):
                hits = batting.get("hits", 0) or 0
                doubles = batting.get("doubles", 0) or 0
                triples = batting.get("triples", 0) or 0
                homers = batting.get("homeRuns", 0) or 0
                walks = batting.get("baseOnBalls", 0) or 0
                hbp = batting.get("hitByPitch", 0) or 0
                runs = batting.get("runs", 0) or 0
                rbi = batting.get("rbi", 0) or 0
                steals = batting.get("stolenBases", 0) or 0
                rows.append({
                    "game_pk": game_pk, "game_date": game_date, "player_id": int(player_id),
                    "team": team_abbr, "role": "H",
                    # battingOrder is slot * 100, with 101/102 for in-game replacements.
                    "order_slot": int(order) // 100 if order else 0,
                    "started": bool(order) and int(order) % 100 == 0,
                    "PA": batting.get("plateAppearances", 0) or 0,
                    "AB": batting.get("atBats", 0) or 0,
                    "H": hits, "2B": doubles, "3B": triples, "HR": homers,
                    "R": runs, "RBI": rbi, "BB": walks, "HBP": hbp, "SB": steals,
                    "K": batting.get("strikeOuts", 0) or 0,
                    "IP": 0.0, "ER": 0, "W": 0,
                    "dk": hitter_points({
                        "1B": max(0, hits - doubles - triples - homers),
                        "2B": doubles, "3B": triples, "HR": homers,
                        "R": runs, "RBI": rbi, "BB": walks, "HBP": hbp, "SB": steals,
                    }),
                })

            if pitching.get("inningsPitched") is not None:
                innings = _ip_to_float(pitching.get("inningsPitched"))
                rows.append({
                    "game_pk": game_pk, "game_date": game_date, "player_id": int(player_id),
                    "team": team_abbr, "role": "P", "order_slot": 0,
                    "started": (pitching.get("gamesStarted", 0) or 0) > 0,
                    "PA": pitching.get("battersFaced", 0) or 0, "AB": 0,
                    "H": pitching.get("hits", 0) or 0, "2B": 0, "3B": 0,
                    "HR": pitching.get("homeRuns", 0) or 0, "R": pitching.get("runs", 0) or 0,
                    "RBI": 0, "BB": pitching.get("baseOnBalls", 0) or 0,
                    "HBP": pitching.get("hitByPitch", 0) or 0, "SB": 0,
                    "K": pitching.get("strikeOuts", 0) or 0,
                    "IP": round(innings, 3),
                    "ER": pitching.get("earnedRuns", 0) or 0,
                    "W": pitching.get("wins", 0) or 0,
                    "dk": pitcher_points({
                        "IP": innings,
                        "K": pitching.get("strikeOuts", 0) or 0,
                        "W": pitching.get("wins", 0) or 0,
                        "ER": pitching.get("earnedRuns", 0) or 0,
                        "H": pitching.get("hits", 0) or 0,
                        "BB": pitching.get("baseOnBalls", 0) or 0,
                        "HBP": pitching.get("hitByPitch", 0) or 0,
                    }),
                })
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    for year in SEASONS:
        target = os.path.join(OUT, f"box_{year}.parquet")
        if os.path.exists(target):
            print(f"  {year} already fetched, skipping", flush=True)
            continue
        games = season_games(year)
        print(f"  {year}: {len(games):,} final regular-season games", flush=True)
        rows, done = [], 0
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for result in pool.map(lambda g: game_rows(*g), games):
                rows.extend(result)
                done += 1
                if done % 500 == 0:
                    print(f"    {done:,}/{len(games):,} games", flush=True)
        frame = pd.DataFrame(rows)
        frame["game_date"] = pd.to_datetime(frame["game_date"])
        frame.to_parquet(target, index=False, compression="zstd")
        empty = frame.groupby("game_pk").size()
        print(f"    {len(frame):,} player-games across {frame['game_pk'].nunique():,} games "
              f"(min rows/game {empty.min()})", flush=True)


if __name__ == "__main__":
    main()
