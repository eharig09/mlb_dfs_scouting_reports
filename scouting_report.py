import argparse
import hashlib
import json
import math
import pickle
import html as html_lib
import os
import datetime
import sys
import re
from collections import deque

# Running under the wrong interpreter is the most common failure here: plain `python` can
# resolve to a system install ahead of .venv, and the only symptom is whichever dependency
# happens to be imported first going missing. Name the real problem instead.
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import LogisticRegression
except ImportError as _import_error:
    _venv = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Scripts", "python.exe")
    _in_venv = os.path.normcase(sys.executable).startswith(
        os.path.normcase(os.path.dirname(os.path.dirname(_venv)))
    )
    sys.stderr.write(
        f"\n[!] Missing dependency: {_import_error.name}\n"
        f"    Running under: {sys.executable}\n"
        + ("    This is not the project's .venv. Call it explicitly:\n"
           f"      {_venv} {' '.join(sys.argv)}\n"
           "    or activate it once: .\\.venv\\Scripts\\Activate.ps1\n"
           if not _in_venv else
           "    You are in .venv but it is incomplete. Reinstall:\n"
           f"      {_venv} -m pip install -r requirements.txt\n")
    )
    raise SystemExit(1)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_ROOT, ".matplotlib"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from tqdm import tqdm
from pybaseball import statcast_pitcher, pitching_stats, playerid_lookup, statcast_batter, batting_stats, statcast, playerid_reverse_lookup
from models.stuff_model import calculate_stuff_plus
from utils.cache import cached_dataframe_call, cached_json_request
from utils.statcast_cache import load_statcast_range
import statsapi
import requests

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import unicodedata
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import timedelta, datetime
from zoneinfo import ZoneInfo
from fpdf import FPDF
try:
    from fpdf.errors import FPDFException
except ImportError:
    class FPDFException(Exception):
        pass
import re
from pybaseball import fielding_stats
from functools import lru_cache

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

#%%

# In-process memo for recency-critical statsapi responses (fetched fresh, reused within a run).
_RECENCY_MEMO = {}


def get_team_performance(team_id, season=None, as_of_date=None):
    DIVISION_ID_MAP = {
        200: "AL West",
        201: "AL East",
        202: "AL Central",
        203: "NL West",
        204: "NL East",
        205: "NL Central"
        }

    current_date = as_of_date or datetime.today().strftime('%Y-%m-%d')
    season = season or int(current_date[:4])
    standings_url = f"https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season={season}&standingsTypes=regularSeason&date={current_date}"
    # Force-fresh (data advances during the day); memoize the league-wide response per date.
    memo_key = ("standings", current_date)
    if memo_key in _RECENCY_MEMO:
        standings_data = _RECENCY_MEMO[memo_key]
    else:
        standings_data = cached_json_request(standings_url, namespace="statsapi", force=True)
        _RECENCY_MEMO[memo_key] = standings_data

    record_data = None
    division_name = "Unknown Division"  # set a default

    for division in standings_data['records']:
        for team in division['teamRecords']:
            if team['team']['id'] == team_id:
                record_data = team

                division_info = division.get('division', {})
                division_id = division_info.get("id")
                division_name = DIVISION_ID_MAP.get(division_id, "Unknown Division")


                home_record = team['records']['splitRecords'][0]  # Home
                away_record = team['records']['splitRecords'][1]  # Away
                run_diff = team['runDifferential']
                break


    home_record = away_record = {'wins': 0, 'losses': 0}
    run_diff = 0
    day_record = night_record = {'wins': 0, 'losses': 0}

    for division in standings_data['records']:
        for team in division['teamRecords']:
            if team['team']['id'] == team_id:
                record_data = team
                division_info = division.get('division', {})
                division_id = division_info.get("id")
                division_name = DIVISION_ID_MAP.get(division_id, "Unknown Division")


                division_rank = team.get('divisionRank')
                games_back = team.get('gamesBack', "—")
                home_record = team['records']['splitRecords'][0]
                away_record = team['records']['splitRecords'][1]
                run_diff = team['runDifferential']
                break

    if not record_data:
        return {"Error": "Team ID not found in standings"}

    wins = record_data['wins']
    losses = record_data['losses']
    win_pct = record_data['winningPercentage']
    streak = record_data['streak']['streakCode']

    home_wins = home_record['wins']
    home_losses = home_record['losses']
    away_wins = away_record['wins']
    away_losses = away_record['losses']

    # Recent performance (wide enough window to compute L10 / L20 / L30 records)
    end_date = datetime.strptime(current_date, "%Y-%m-%d")
    start_date = end_date - timedelta(days=45)
    schedule_url = f"https://statsapi.mlb.com/api/v1/schedule?teamId={team_id}&startDate={start_date.strftime('%Y-%m-%d')}&endDate={end_date.strftime('%Y-%m-%d')}&sportId=1"
    schedule_data = cached_json_request(schedule_url, namespace="statsapi", force=True)
    games = schedule_data.get('dates', [])

    recent_results = []
    for date in reversed(games):
        for game in date['games']:
            if game['status']['detailedState'] == 'Final':
                home_team_data = game['teams']['home']
                away_team_data = game['teams']['away']

                if 'isWinner' not in home_team_data or 'isWinner' not in away_team_data:
                    continue

                is_winner = (
                    (home_team_data['team']['id'] == team_id and home_team_data['isWinner']) or
                    (away_team_data['team']['id'] == team_id and away_team_data['isWinner'])
                )
                recent_results.append('W' if is_winner else 'L')

                game_time = game['gameDate']
                game_hour = datetime.strptime(game_time, '%Y-%m-%dT%H:%M:%SZ').hour
                if 12 <= game_hour < 18:
                    if is_winner:
                        day_record['wins'] += 1
                    else:
                        day_record['losses'] += 1
                else:
                    if is_winner:
                        night_record['wins'] += 1
                    else:
                        night_record['losses'] += 1

    # Compute the current streak from the schedule so it stays consistent with the
    # displayed recent games / records (the standings streakCode can lag by a snapshot).
    if recent_results:
        first = recent_results[0]
        count = 0
        for result in recent_results:
            if result == first:
                count += 1
            else:
                break
        streak = f"{first}{count}"

    def _rec_pct(w, l):
        g = w + l
        return f"{w}-{l} ({('%.3f' % (w / g)).lstrip('0')})" if g else f"{w}-{l}"

    def _window_record(n):
        window = recent_results[:n]
        return _rec_pct(window.count('W'), window.count('L'))

    last_10_record = _window_record(10)
    last_20_record = _window_record(20)
    last_30_record = _window_record(30)

    runs_scored = record_data['runsScored']
    runs_allowed = record_data['runsAllowed']
    pythagorean_expectation = (runs_scored ** 2) / ((runs_scored ** 2) + (runs_allowed ** 2))

    return {
        'Division': division_name,
        'Division Rank': f"{division_rank}{ordinal_suffix(int(division_rank))}" if division_rank else "N/A",
        'Games Back': games_back,
        'Overall Record': f"{wins}-{losses} ({win_pct})",
        'Last 10 Games': last_10_record,
        'Last 20 Games': last_20_record,
        'Last 30 Games': last_30_record,
        'Current Streak': streak,
        'Home Record': _rec_pct(home_wins, home_losses),
        'Away Record': _rec_pct(away_wins, away_losses),
        'Run Differential': run_diff,
        'Pythagorean Expectation': f"{pythagorean_expectation:.3f}"
    }

def ordinal_suffix(n):
    return 'th' if 11 <= n % 100 <= 13 else {1:'st', 2:'nd', 3:'rd'}.get(n % 10, 'th')


def get_last_10_games_statsapi(team_id: int, as_of_date=None):
    today = datetime.strptime(as_of_date, "%Y-%m-%d") if as_of_date else datetime.now()
    start_date = (today - timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')

    url = (
        f"https://statsapi.mlb.com/api/v1/schedule?"
        f"sportId=1&teamId={team_id}&startDate={start_date}&endDate={end_date}&sort=desc&hydrate=team,linescore"
    )

    data = cached_json_request(url, namespace="statsapi", force=True)
    games = data.get('dates', [])

    rows = []
    for date_entry in reversed(games):  # reverse so we get most recent games first
        for game in date_entry.get('games', []):
            if game['status']['detailedState'] != 'Final':
                continue

            game_date = game['officialDate']
            is_home = game['teams']['home']['team']['id'] == team_id
            opponent = (
                game['teams']['away']['team']['name']
                if is_home else game['teams']['home']['team']['name']
            )
            team_score = game['teams']['home']['score'] if is_home else game['teams']['away']['score']
            opp_score = game['teams']['away']['score'] if is_home else game['teams']['home']['score']
            result = 'W' if team_score > opp_score else 'L'

            rows.append({
                'Date': game_date,
                'Opponent': opponent,
                'Score': f"{team_score}-{opp_score}",
                'Result': result,
                'Home/Away': 'Home' if is_home else 'Away'
            })

            if len(rows) == 10:
                return pd.DataFrame(rows)

    return pd.DataFrame(rows)





def get_standings_from_api(team_id, season=2025):
    url = f"https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season={season}&standingsTypes=regularSeason"
    response = requests.get(url)
    if response.status_code != 200:
        print("❌ Failed to fetch standings from MLB API")
        return None, None

    records = response.json().get("records", [])
    for division in records:
        division_info = division.get("division", {})
        division_name = (
            division_info.get("name") or 
            division_info.get("nameShort") or 
            division_info.get("abbreviation") or 
            "Unknown Division"
        )

        rows = []
        highlight_abbr = TEAM_ID_MAP.get(team_id)

        for team_record in division["teamRecords"]:
            team_info = team_record["team"]
            tid = team_info["id"]
            abbr = TEAM_ID_MAP.get(tid, team_info.get("name", "Unknown"))
            is_target = tid == team_id

            rows.append({
                "Tm": abbr,
                "W": team_record["wins"],
                "L": team_record["losses"],
                "W-L%": round(float(team_record["winningPercentage"]), 3),
                "GB": team_record["gamesBack"],
                "Strk": team_record["streak"]["streakCode"]
            })

        df = pd.DataFrame(rows)
        if highlight_abbr in df["Tm"].values:
            return df, division_name

    print("⚠️ Team ID not found in any division")
    return None, None




def remove_accents(input_str):
  nfkd_form = unicodedata.normalize('NFKD', input_str)
  return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

TEAM_NAME_MAP = {
    "ARI": "Arizona Diamondbacks",
    "ATL": "Atlanta Braves",
    "ATH": "Athletics",
    "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs",
    "CWS": "Chicago White Sox",
    "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies",
    "DET": "Detroit Tigers",
    "HOU": "Houston Astros",
    "KC": "Kansas City Royals",
    "LAA": "Los Angeles Angels",
    "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins",
    "NYM": "New York Mets",
    "NYY": "New York Yankees",
    "OAK": "Oakland Athletics",
    "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates",
    "SD": "San Diego Padres",
    "SEA": "Seattle Mariners",
    "SF": "San Francisco Giants",
    "STL": "St. Louis Cardinals",
    "TB": "Tampa Bay Rays",
    "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays",
    "WSH": "Washington Nationals"
}

TEAM_ID_MAP = {
    109: "ARI", 144: "ATL", 110: "BAL", 111: "BOS", 112: "CHC",
    145: "CWS", 113: "CIN", 114: "CLE", 115: "COL", 116: "DET",
    117: "HOU", 118: "KC", 108: "LAA", 119: "LAD", 146: "MIA",
    158: "MIL", 142: "MIN", 121: "NYM", 147: "NYY", 133: "ATH",
    143: "PHI", 134: "PIT", 135: "SD", 136: "SEA", 137: "SF",
    138: "STL", 139: "TB", 140: "TEX", 141: "TOR", 120: "WSH"
}

def get_pitchers_mlb(team_name):
    team_abbr = team_name.upper()
    normalized_team_name = TEAM_NAME_MAP.get(team_abbr, team_name).lower().strip()

    url = "https://www.mlb.com/probable-pitchers"
    response = requests.get(url)
    if response.status_code != 200:
        print("❌ Failed to load MLB probable pitchers page.")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    game_blocks = soup.select("div.probable-pitchers__pitchers")
    print(f"🧪 Found {len(game_blocks)} games on the page")

    for block in game_blocks:
        stats_div = block.find_next("div", class_="probable-pitchers__stats-header")
        if not stats_div:
            continue

        away_id = stats_div.get("data-player-id-away")
        home_id = stats_div.get("data-player-id-home")
        away_team_id = stats_div.get("data-team-id-away")
        home_team_id = stats_div.get("data-team-id-home")

        pitcher_links = block.select("a.probable-pitchers__pitcher-name-link")
        away_name = pitcher_links[0].text.strip() if len(pitcher_links) > 0 else ""
        home_name = pitcher_links[1].text.strip() if len(pitcher_links) > 1 else ""

        away_abbr = TEAM_ID_MAP.get(int(away_team_id)) if away_team_id else None
        home_abbr = TEAM_ID_MAP.get(int(home_team_id)) if home_team_id else None

        print(f"🔍 Comparing: '{team_abbr}' vs away='{away_abbr}' ({away_name}), home='{home_abbr}' ({home_name})")

        if team_abbr == home_abbr:
            print(f"✅ Match found with home team — {home_name}")
            return int(home_id) if home_id else lookup_player_id(remove_accents(home_name)) if home_name else None

        if team_abbr == away_abbr:
            print(f"✅ Match found with away team — {away_name}")
            return int(away_id) if away_id else lookup_player_id(remove_accents(away_name)) if away_name else None

    print(f"⚠️ No starter found for {team_name} today.")
    return None


def get_probable_pitchers_for_game(game_id):
    """
    Return MLBAM probable pitcher IDs for a specific game.
    Uses StatsAPI first so reports are tied to the selected date/game.
    """
    pitchers = {"home": None, "away": None}

    schedule_url = "https://statsapi.mlb.com/api/v1/schedule"
    schedule_params = {
        "sportId": 1,
        "gamePk": game_id,
        "hydrate": "probablePitcher",
    }

    try:
        data = cached_json_request(schedule_url, params=schedule_params, namespace="statsapi")
        games = data.get("dates", [{}])[0].get("games", [])
        if games:
            teams = games[0].get("teams", {})
            for side in ("home", "away"):
                probable = teams.get(side, {}).get("probablePitcher", {})
                if probable.get("id"):
                    pitchers[side] = int(probable["id"])
    except Exception as e:
        print(f"⚠️ Could not fetch schedule probable pitchers for game {game_id}: {e}")

    if pitchers["home"] and pitchers["away"]:
        return pitchers

    feed_url = f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
    try:
        data = cached_json_request(feed_url, namespace="statsapi")
        probable_pitchers = data.get("gameData", {}).get("probablePitchers", {})
        for side in ("home", "away"):
            if not pitchers[side]:
                probable = probable_pitchers.get(side, {})
                if probable.get("id"):
                    pitchers[side] = int(probable["id"])
    except Exception as e:
        print(f"⚠️ Could not fetch live-feed probable pitchers for game {game_id}: {e}")

    return pitchers


def resolve_probable_pitcher(selected_game, side, team_abbr):
    pitchers = get_probable_pitchers_for_game(selected_game["game_id"])
    pitcher_id = pitchers.get(side)
    if pitcher_id:
        if pitcher_belongs_to_team(pitcher_id, team_abbr):
            return pitcher_id
        print(f"Warning: ignoring StatsAPI probable pitcher {_player_name_from_id(pitcher_id)} for {team_abbr}; current team does not match.")


    probable_name = selected_game.get(f"{side}_probable_pitcher")
    if probable_name and probable_name.upper() != "TBD":
        pitcher_id = lookup_player_id(remove_accents(probable_name))
        if pitcher_id and pitcher_belongs_to_team(pitcher_id, team_abbr):
            return pitcher_id
        if pitcher_id:
            print(f"Warning: ignoring probable pitcher {probable_name} for {team_abbr}; current team does not match.")

    print(f"⚠️ Falling back to current probable-pitchers page for {team_abbr}.")
    return None

def _is_today(date):
    """True when `date` could be today, in local or Eastern time.

    Deliberately generous: the only caller uses it to *disable* a fallback, so the ambiguous
    case (a date that is today somewhere) keeps the fallback rather than silently dropping a
    starter the page really does know.
    """
    if not date:
        return True
    text = str(date)[:10]
    return text in {datetime.now().strftime("%Y-%m-%d"),
                    datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")}


def resolve_probable_pitcher_v2(selected_game, side, team_abbr, exclude_ids=None, date=None):
    """Resolve a probable starter for the selected game and side with validation.

    exclude_ids: starters already committed to this team's other game today. On a
    doubleheader MLB often lists game 2 as TBD, and the fallback below scrapes a page that
    shows one starter per team per day -- so without this it confidently returns game 1's
    pitcher for game 2, which is impossible.

    date: the game date. The page fallback has no date parameter -- it is *today's* board --
    so for any other date it answers a different question and is skipped. Without this a
    report two days out silently resolved every club to tonight's starter and reported it as
    an announced probable, which is worse than not building the report at all.
    """
    exclude_ids = {int(p) for p in (exclude_ids or []) if p}
    pitchers = get_probable_pitchers_for_game(selected_game["game_id"])
    pitcher_id = pitchers.get(side)
    if pitcher_id:
        return pitcher_id

    probable_name = selected_game.get(f"{side}_probable_pitcher")
    if probable_name and str(probable_name).upper() != "TBD":
        named_pitcher = lookup_player_id(remove_accents(probable_name))
        if named_pitcher:
            return named_pitcher

    if not _is_today(date):
        print(f"No probable posted for {team_abbr} on {date}; the probable-pitchers page "
              f"only covers today, so it is not consulted.")
        return None

    current_page_pitcher = get_pitchers_mlb(team_abbr)
    if current_page_pitcher and pitcher_belongs_to_team(current_page_pitcher, team_abbr):
        other_side = "away" if side == "home" else "home"
        other_pitcher = pitchers.get(other_side)
        if other_pitcher and int(other_pitcher) == int(current_page_pitcher):
            print(f"Warning: ignoring probable-pitchers fallback for {team_abbr}; it duplicates the {other_side} starter.")
        elif int(current_page_pitcher) in exclude_ids:
            print(f"Warning: ignoring probable-pitchers fallback {_player_name_from_id(current_page_pitcher)} "
                  f"for {team_abbr}; already starting their other game today.")
        else:
            return current_page_pitcher
    elif current_page_pitcher:
        print(f"Warning: ignoring probable-pitchers fallback {_player_name_from_id(current_page_pitcher)} for {team_abbr}; current team does not match.")

    print(f"No validated probable pitcher found for {team_abbr}.")
    return None


# ---------------------------------------------------------------------------------------
# Starter resolution ladder
#
# MLB posts a probable when it feels like it, and until it does the report used to refuse to
# build at all -- which is backwards, because the hours before a probable posts are exactly
# when a board is being put together. Two sources fill the gap, in this order:
#
#   the DK salary file   DK's `Starting` column reads "P" for the arm a club is starting and
#                        is usually populated well before MLB's feed. It is a statement about
#                        tonight, not a model.
#   the rotation model   `build_rotation` + `effective_starter`, already measured at 46.1%
#                        walk-forward over 7,084 team-games against 17.5% for naming the
#                        club's most frequent starter.
#
# Anything below `announced` is marked provisional and says so on the report. The point is
# never to substitute a pitcher silently: a labelled guess is useful, an unlabelled one is
# how a board becomes untrustworthy.
# ---------------------------------------------------------------------------------------

# What each `source` means in a sentence, for the line the report prints.
STARTER_SOURCE_LABEL = {
    "override": "manual override",
    "announced": "announced probable",
    "salary": "DraftKings salary file",
    "rotation": "rotation model",
    "bulk": "rotation model, bulk arm behind an opener",
}


def _roster_name_index(team_abbr, as_of_date=None):
    """{normalized name: id} over the club's 40-man, for turning a DK spelling into an id.

    Matching against the roster rather than a global name lookup does two jobs at once: it
    resolves the id and it proves the arm is on this club, so a DK row carrying a traded
    pitcher's old team cannot slip through.
    """
    index = {}
    try:
        entries = get_team_roster(team_abbr, "40Man", as_of_date=as_of_date) or []
    except Exception as error:
        print(f"⚠️ Roster unavailable for {team_abbr}: {error}")
        return index
    for entry in entries:
        person = entry.get("person") or {}
        pid, full = person.get("id"), person.get("fullName")
        if pid and full:
            index[_dk_normalize_name(full)] = int(pid)
    return index


def _dk_normalize_name(name):
    """Fold a name the way the DK matcher does, so both sides of the join agree."""
    try:
        from dfs.salaries import normalize_name
    except Exception:
        return remove_accents(str(name or "")).lower().strip()
    return normalize_name(name)


def starter_from_salary_file(date, team_abbr, exclude_ids=None):
    """The starter DraftKings has flagged for this club tonight, or None.

    DK's export carries a `Starting` column reading "P" for the arm a club is starting, and
    it is frequently populated hours before MLB posts a probable. Every slate for the date
    is unioned: DK puts up several a night and any one game appears on a subset of them, so
    reading only `main` misses most of the board (54 of 162 hitters on a checked date).

    The flag also fires on a pitcher listed at RP, which is exactly what is wanted -- that
    is an opener, and he is who the top of the order actually leads off against.
    """
    exclude_ids = {int(p) for p in (exclude_ids or []) if p}
    try:
        from dfs import salaries as dk_salaries
    except Exception as error:
        print(f"⚠️ DK salary lookup unavailable: {error}")
        return None

    try:
        files = dk_salaries.list_salary_files(date=str(date))
    except Exception as error:
        print(f"⚠️ Could not list DK salary files for {date}: {error}")
        return None
    if not files:
        return None

    flagged = []
    for info in files:
        try:
            frame = dk_salaries.load_salaries(info["path"])
        except Exception:
            continue
        if frame is None or frame.empty or "DK Starting" not in frame.columns:
            continue
        mine = frame[frame["DK Starting"].fillna(False).astype(bool)
                     & frame["DK Team"].eq(dk_salaries.canon_team(team_abbr))]
        flagged.extend(str(n).strip() for n in mine["DK Name"] if str(n).strip())
    if not flagged:
        return None

    # Prices are identical across a night's slates, but a club can be re-flagged between
    # exports (a late scratch). The most common spelling wins, ties to the last file read.
    names = {}
    for name in flagged:
        names[name] = names.get(name, 0) + 1
    best = max(names, key=lambda n: names[n])
    if len(names) > 1:
        print(f"⚠️ DK salary files disagree on {team_abbr}'s starter "
              f"({', '.join(sorted(names))}); taking {best}.")

    roster = _roster_name_index(team_abbr, as_of_date=str(date))
    pitcher_id = roster.get(_dk_normalize_name(best))
    if not pitcher_id:
        pitcher_id = lookup_player_id(remove_accents(best))
        if pitcher_id and not pitcher_belongs_to_team(pitcher_id, team_abbr):
            print(f"Warning: ignoring DK starter {best} for {team_abbr}; "
                  f"current team does not match.")
            return None
    if not pitcher_id:
        print(f"Warning: could not resolve DK starter '{best}' for {team_abbr} to a player id.")
        return None
    if int(pitcher_id) in exclude_ids:
        print(f"Warning: ignoring DK starter {best} for {team_abbr}; "
              f"already starting their other game today.")
        return None
    return int(pitcher_id)


def starter_from_rotation(team_abbr, season, context_end, as_of_date=None, exclude_ids=None):
    """`effective_starter` over this club's rotation -> (id, name, source, note) or Nones.

    Wired here rather than inside `build_rotation` on purpose: the model is checked against
    history on nights where a probable *was* posted, so it must never consult one.
    """
    exclude_ids = {int(p) for p in (exclude_ids or []) if p}
    try:
        rotation = build_rotation(team_abbr, season, context_end, as_of_date=as_of_date)
    except Exception as error:
        print(f"⚠️ Rotation model unavailable for {team_abbr}: {error}")
        return None, None, None, ""
    if exclude_ids:
        rotation = dict(rotation)
        rotation["candidates"] = [c for c in rotation.get("candidates") or []
                                  if int(c.get("id", 0)) not in exclude_ids]
        rotation["bulk_arms"] = [b for b in rotation.get("bulk_arms") or []
                                 if int(b.get("id", 0)) not in exclude_ids]
    pick = effective_starter(rotation)
    if not pick or not pick.get("id"):
        return None, None, None, (pick or {}).get("note", "")
    return int(pick["id"]), pick.get("name"), pick.get("source"), pick.get("note", "")


@lru_cache(maxsize=1024)
def pitcher_belongs_to_team(pitcher_id, team_abbr):
    try:
        data = statsapi.get("people", {"personIds": pitcher_id, "hydrate": "currentTeam"})
        people = data.get("people", [])
        if not people:
            return True
        current_team = people[0].get("currentTeam", {})
        current_id = current_team.get("id")
        if not current_id:
            return True
        return int(current_id) == int(get_team_id(team_abbr))
    except Exception:
        return True


def get_pitcher_handedness(player_id):
    """
    Fetches the handedness (throws) of a pitcher given their MLBAM player ID.

    Parameters:
        player_id (str): The MLBAM player ID.

    Returns:
        str: Pitcher's handedness ('R' for right-handed, 'L' for left-handed, 'Unknown' if not found).
    """
    # MLB Stats API URL to fetch player data
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}"

    # Send the request
    try:
        data = cached_json_request(url, namespace="statsapi")['people']
        # Try accessing the 'pitchHand' attribute (handedness)
        handedness = data[0].get('pitchHand', 'Unknown')
        return handedness['code'] if handedness != 'Unknown' else 'Unknown'
    except Exception:
        print("Error fetching player data.")
        return 'Unknown'




def generate_starter_arsenal(pitcher_id, start_date, end_date, save_dir='plots'):
    # Fetch the pitcher's data
    df = cached_dataframe_call("statcast_pitcher", statcast_pitcher, start_date, end_date, pitcher_id)
    df = df[df['pitch_type'].notnull()]
    df.dropna(subset=['release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z'], inplace=True)

    if df.empty:
        # If no data, fall back to 2024 data
        start_date = '2024-03-01'
        end_date = '2024-10-31'
        df = cached_dataframe_call("statcast_pitcher", statcast_pitcher, start_date, end_date, pitcher_id)
        df = df[df['pitch_type'].notnull()]
        df.dropna(subset=['release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z'], inplace=True)

    # Step 1: Weighted Stuff+ Target
    desc_weights = {
        'swinging_strike': 1.0,
        'swinging_strike_blocked': 1.0,
        'called_strike': 0.7,
        'foul': 0.3,
        'foul_tip': 0.3,
        'ball': 0.0,
        'ball_in_dirt': 0.0,
        'hit_by_pitch': -1.0,
        'blocked_ball': -0.2,
        'wild_pitch': -1.0,
        'in_play': 0.2,
    }
    df['stuff_score'] = df['description'].map(desc_weights).fillna(0.2)
    if 'launch_speed' in df.columns:
        weak_contact = (df['description'] == 'in_play') & (df['launch_speed'] < 85)
        df.loc[weak_contact, 'stuff_score'] = 0.4

    features = ['release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z']
    if len(df) >= 50:
        model = LinearRegression().fit(df[features], df['stuff_score'])
        df['stuff_score'] = model.predict(df[features])
        league_avg = df['stuff_score'].mean() or 1
        df['Stuff+'] = (df['stuff_score'] / league_avg) * 100
    else:
        df['Stuff+'] = 100.0

    # Step 2: Define CSW and Zone flags
    df['is_csw'] = df['description'].isin(['called_strike', 'swinging_strike', 'swinging_strike_blocked'])
    df['is_zone'] = df['zone'].isin(range(1, 10))

    # Step 3: Aggregate Metrics by Pitch Type
    summary = df.groupby('pitch_type').agg({
        'pitch_type': 'count',
        'release_speed': 'mean',
        'release_spin_rate': 'mean',
        'pfx_x': 'mean',
        'pfx_z': 'mean',
        'Stuff+': 'mean',
        'is_csw': 'sum',
        'is_zone': 'sum'
    }).rename(columns={'pitch_type': 'Total'})

    summary['Usage %'] = (summary['Total'] / summary['Total'].sum()) * 100
    summary['Zone%'] = (summary['is_zone'] / summary['Total']) * 100
    summary['CSW%'] = (summary['is_csw'] / summary['Total']) * 100
    summary['Horiz. Break'] = summary['pfx_x'] * 12
    summary['Vert. Break'] = summary['pfx_z'] * 12

    summary = summary.drop(columns=['is_csw', 'is_zone', 'pfx_x', 'pfx_z'])


    # Step 5: Format and Return
    summary.reset_index(inplace=True)  # Make 'pitch_type' a column
    summary.rename(columns={'pitch_type': 'Pitch'}, inplace=True)
    summary = summary.round(2)

    return summary


# %%

def playerid_lookup_list(name_list):
    results = []
    for name in name_list:
        try:
            last, first = name.split()[-1], " ".join(name.split()[:-1])
            pid = playerid_lookup(last, first)
            if not pid.empty:
                results.append(pid.iloc[0])
        except:
            continue
    return pd.DataFrame(results)


def lookup_player_id(name):
    try:
        first, last = name.split(" ", 1)
        result = playerid_lookup(last, first)
        if not result.empty:
            return int(result["key_mlbam"].iloc[0])
    except:
        pass
    return None

def get_matchup_summary(batter_name, pitcher, start_date="2022-01-01", end_date=None, min_pa=1):
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    # Lookup IDs
    batter_id = lookup_player_id(batter_name)
    try:
        pitcher_id = int(pitcher)
        pitcher_name = f"#{pitcher_id}"
    except (ValueError, TypeError):
        pitcher_id = lookup_player_id(pitcher)
        pitcher_name = pitcher

    if batter_id is None or pitcher_id is None:
        return f"{batter_name} vs {pitcher_name}: Player ID not found"

    try:
        df = cached_dataframe_call("statcast_pitcher", statcast_pitcher, start_date, end_date, pitcher_id)
        df = df[df["batter"] == batter_id]

        if df.empty or df["at_bat_number"].nunique() < min_pa:
            return f"{batter_name} vs {pitcher_name}: No recent data"

        # Only one row per AB (we group by game + AB number)
        ab_grouped = df.groupby(["game_pk", "at_bat_number"]).agg({
            "events": "first",  # Get result of the AB
            "launch_speed": "mean",
            "launch_angle": "mean",
            "estimated_ba_using_speedangle": "mean",
            "estimated_slg_using_speedangle": "mean"
        }).reset_index()

        ab = len(ab_grouped)
        hits = ab_grouped["events"].isin(["single", "double", "triple", "home_run"]).sum()
        hr = ab_grouped["events"].eq("home_run").sum()
        bb = ab_grouped["events"].eq("walk").sum()
        k = ab_grouped["events"].eq("strikeout").sum()
        avg = round(hits / ab, 3) if ab else 0.0

        ev = ab_grouped["launch_speed"].mean()
        la = ab_grouped["launch_angle"].mean()
        xba = ab_grouped["estimated_ba_using_speedangle"].mean()
        xslg = ab_grouped["estimated_slg_using_speedangle"].mean()

        summary = f"{batter_name} — {hits}-for-{ab} (.{str(avg)[2:]})"
        if hr: summary += f", {hr} HR"
        if bb: summary += f", {bb} BB"
        if k: summary += f", {k} K"

        batted = []
        if not pd.isna(ev): batted.append(f"EV: {round(ev,1)} mph")
        if not pd.isna(la): batted.append(f"LA: {round(la,1)}°")
        if not pd.isna(xba): batted.append(f"xBA: {round(xba,3)}")
        if not pd.isna(xslg): batted.append(f"xSLG: {round(xslg,3)}")

        if batted:
            summary += " | " + ", ".join(batted)

        return summary

    except Exception as e:
        return f"{batter_name} vs {pitcher_name}: Statcast error ({e})"


def _matchup_stats_row(batter_name, pitcher_id, start_date, end_date):
    """Structured career-vs-pitcher line for one batter, or None if no data."""
    batter_id = lookup_player_id(batter_name)
    if batter_id is None or pitcher_id is None:
        return None
    try:
        df = cached_dataframe_call("statcast_pitcher", statcast_pitcher, start_date, end_date, pitcher_id)
        df = df[df["batter"] == batter_id]
        if df.empty:
            return None
        ab_grouped = df.groupby(["game_pk", "at_bat_number"]).agg({
            "events": "first",
            "estimated_ba_using_speedangle": "mean",
            "estimated_slg_using_speedangle": "mean",
        }).reset_index()
        ab = len(ab_grouped)
        hits = int(ab_grouped["events"].isin(["single", "double", "triple", "home_run"]).sum())
        return {
            "Name": batter_name,
            "AB": ab,
            "H": hits,
            "HR": int(ab_grouped["events"].eq("home_run").sum()),
            "BB": int(ab_grouped["events"].eq("walk").sum()),
            "K": int(ab_grouped["events"].eq("strikeout").sum()),
            "AVG": round(hits / ab, 3) if ab else 0.0,
            "xBA": round(pd.to_numeric(ab_grouped["estimated_ba_using_speedangle"], errors="coerce").mean(), 3),
            "xSLG": round(pd.to_numeric(ab_grouped["estimated_slg_using_speedangle"], errors="coerce").mean(), 3),
        }
    except Exception:
        return None


def build_matchup_table(hitters_df, pitcher, min_pa=3, start_date="2022-01-01", end_date=None):
    """Career matchup lines for a lineup vs a given starter, as a table (most PA first)."""
    columns = ["Name", "AB", "H", "HR", "BB", "K", "AVG", "xBA", "xSLG"]
    if hitters_df is None or hitters_df.empty:
        return pd.DataFrame(columns=columns)
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    try:
        pitcher_id = int(pitcher)
    except (ValueError, TypeError):
        pitcher_id = lookup_player_id(pitcher)
    rows = []
    for name in hitters_df.get("Name", pd.Series(dtype=str)):
        row = _matchup_stats_row(name, pitcher_id, start_date, end_date)
        if row and row["AB"] >= min_pa:
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["AB", "AVG"], ascending=False).reset_index(drop=True)


def get_true_relievers(team_abbr, year=2025):
    data = cached_dataframe_call("pitching_stats", pitching_stats, year, qual=0)  # Include all pitchers
    team_pitchers = data[data['Team'] == team_abbr].copy()

    print(f"\n🔍 Full pitching staff for {team_abbr}:")
    print(team_pitchers[['Name', 'G', 'GS', 'IP']])

    # Reliever logic: either never started or low IP + appearances
    relievers = team_pitchers[
        (team_pitchers['GS'] < team_pitchers['G'])
    ].copy()

    if relievers.empty:
        print("⚠️ No relievers matched usage criteria.")
        return pd.DataFrame()

    ids = []
    for name in relievers['Name']:
        try:
            last, first = name.split()[-1], " ".join(name.split()[:-1])
            pid_df = playerid_lookup(last, first)
            if not pid_df.empty:
                ids.append({
                    'Name': name,
                    'player_id': int(pid_df.iloc[0]['key_mlbam'])
                })
        except:
            continue

    return pd.DataFrame(ids)




def get_gmLI_from_fangraphs(id, season=2025):
    fg_id = playerid_reverse_lookup([id])['key_fangraphs'].values[0]

    url = f"https://www.fangraphs.com/players/player-id/{fg_id}/stats?position=P#win-probability"

    # Setup headless Chrome
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")  # Chrome 109+
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)

    # Wait for dynamic content to load
    time.sleep(1)

    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    # Locate the correct table
    table = soup.find("div", {"id": "win-probability"})
    if not table:
        print("gmLI table not found.")
        return None

    # Parse it into a DataFrame
    html_table = str(table.find("table"))
    df = pd.read_html(html_table)[0]

    # Clean & filter by season
    df['Season'] = df['Season'].astype(str)
    season_row = df[df['Season'].str.contains(str(season)) & ~df['Season'].str.contains("Postseason")]
    
    if not season_row.empty:
        return float(season_row['gmLIgmLI - Average Leverage Index when entering the game'].values[0])
    
    return None


def calculate_entry_leverage_index(player_id, start_date, end_date):
    try:
        df = load_statcast_range(start_date, end_date)
        df = df[df['pitcher'] == player_id]
        if df.empty or 'home_win_exp' not in df.columns:
            return None

        df['pitching_team'] = df.apply(
            lambda row: row['home_team'] if row['inning_topbot'] == 'Top' else row['away_team'], axis=1
        )
        df['pitcher_is_home'] = df['pitching_team'] == df['home_team']
        df['pitcher_win_exp'] = df.apply(
            lambda row: row['home_win_exp'] if row['pitcher_is_home'] else 1 - row['home_win_exp'], axis=1
        )

        df = df.sort_values(['game_pk', 'inning', 'at_bat_number', 'pitch_number'])
        entry_df = df.groupby('game_pk').first().copy()

        lis = []
        for _, row in entry_df.iterrows():
            inning = row['inning']
            home_score = row['home_score']
            away_score = row['away_score']
            outs = row['outs_when_up']
            win_exp = row['pitcher_win_exp']

            # Base runners
            base_count = int(pd.notna(row['on_1b'])) + int(pd.notna(row['on_2b'])) + int(pd.notna(row['on_3b']))

            # Score leverage
            score_diff = abs(home_score - away_score)
            score_leverage = 1.5 if score_diff <= 1 else 1.2 if score_diff <= 3 else 0.8

            # Inning leverage
            inning_leverage = 1.5 if inning >= 8 else 1.2 if inning >= 6 else 1.0

            # Base-out leverage
            base_out_leverage = (1.0 + 0.2 * base_count) * (1.0 + 0.25 * (2 - outs))

            # Win expectancy leverage
            win_exp_leverage = (1 - abs(win_exp - 0.5)) * 2

            # Composite LI
            li = score_leverage * inning_leverage * base_out_leverage * win_exp_leverage
            lis.append(li)

        return round(np.mean(lis), 2) if lis else None

    except Exception as e:
        print(f"⚠️ Error calculating composite entry leverage index for {player_id}: {e}")
        return None



def get_team_roster(team_abbr, roster_type="active", as_of_date=None):
    team_id = get_team_id(team_abbr)
    url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster"
    params = {"rosterType": roster_type}
    if as_of_date:
        params["date"] = as_of_date
    data = cached_json_request(url, params=params, namespace="statsapi")
    return data.get("roster", [])


def get_player_season_stat(player_id, group, season):
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
    data = cached_json_request(
        url,
        params={"stats": "season", "group": group, "season": season},
        namespace="statsapi",
    )
    stats = data.get("stats", [])
    if not stats:
        return {}
    splits = stats[0].get("splits", [])
    return splits[0].get("stat", {}) if splits else {}


def _season_start_date(season):
    return f"{int(season)}-03-01"


def _regular_season(games):
    """Keep only regular-season games from a statsapi schedule pull.

    The season-long pulls start at March 1, which lands in spring training. Exhibition
    games were flowing into the schedule-context splits, the sweep record and the
    trip/homestand form -- and split-squad spring games register as doubleheaders, which
    is how this surfaced. statsapi marks regular season 'R'; 'S' is spring, 'E'/'A'
    exhibition, 'F'/'D'/'L'/'W' postseason rounds.
    """
    return [g for g in (games or []) if str(g.get("game_type", "R")).upper() == "R"]


def _pregame_end_date(game_date):
    if not game_date:
        return None
    return (datetime.strptime(game_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")


def get_player_stat(player_id, group, season, end_date=None):
    if not end_date:
        return get_player_season_stat(player_id, group, season)

    start_date = _season_start_date(season)
    if end_date < start_date:
        return {}

    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
    data = cached_json_request(
        url,
        params={
            "stats": "byDateRange",
            "group": group,
            "season": season,
            "startDate": start_date,
            "endDate": end_date,
        },
        namespace="statsapi",
    )
    stats = data.get("stats", [])
    if not stats:
        return {}
    splits = stats[0].get("splits", [])
    return splits[0].get("stat", {}) if splits else {}


def _stat_float(stat, key, default=0.0):
    value = stat.get(key, default)
    if isinstance(value, str):
        value = value.replace("%", "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _stat_int(stat, key, default=0):
    try:
        return int(float(stat.get(key, default)))
    except (TypeError, ValueError):
        return default


def _rate(numerator, denominator):
    return round(numerator / denominator, 3) if denominator else 0.0


def _pct(numerator, denominator):
    return round((numerator / denominator) * 100, 1) if denominator else 0.0


def _ip_to_float(ip_value):
    try:
        text = str(ip_value)
        whole, _, frac = text.partition(".")
        innings = int(whole or 0)
        if not frac:
            return float(innings)
        outs = int(frac[:1])
        if outs not in {0, 1, 2}:
            return float(ip_value)
        return innings + (outs / 3)
    except (TypeError, ValueError):
        return 0.0


def generate_bullpen_summary(team_abbr, year=2025):
    # Correct team abbreviation
    two_letter = {'TB': 'TBR', 'KC': 'KCR', 'SD': 'SDP', 'WSH': 'WSN',
                  'CWS': 'CHW', 'SF': 'SFG', 'OAK': 'ATH'}
    if team_abbr in two_letter:
        team_abbr = two_letter[team_abbr]

    # Load season-level pitching stats
    try:
        season_stats = cached_dataframe_call("pitching_stats", pitching_stats, year, qual=0)
    except Exception as e:
        print(f"⚠️ Bullpen leaderboard unavailable for {team_abbr}: {e}")
        return pd.DataFrame(columns=['Name', 'IP', 'ERA', 'FIP', 'K%', 'BB%', 'WHIP'])
    bullpen_stats = season_stats[
        (season_stats['Team'] == team_abbr) &
        (season_stats['G'] > season_stats['GS'])
    ].copy()

    if bullpen_stats.empty:
        print(f"\u26a0\ufe0f No bullpen stats found for {team_abbr}")
        return pd.DataFrame()

    # Select relevant columns
    summary = bullpen_stats[['Name', 'IP', 'ERA', 'FIP', 'K%', 'BB%', 'WHIP']].copy()

    # Total innings pitched
    total_ip = summary['IP'].sum()

    if total_ip > 0:
        # Weighted averages based on IP
        weighted_era = (summary['ERA'] * summary['IP']).sum() / total_ip
        weighted_fip = (summary['FIP'] * summary['IP']).sum() / total_ip
        weighted_kpct = (summary['K%'] * summary['IP']).sum() / total_ip
        weighted_bb_pct = (summary['BB%'] * summary['IP']).sum() / total_ip
        weighted_whip = (summary['WHIP'] * summary['IP']).sum() / total_ip
    else:
        weighted_era = weighted_fip = weighted_kpct = weighted_bb_pct = weighted_whip = 0

    # Create bullpen summary row
    bullpen_summary = pd.DataFrame([{
        'Name': 'BULLPEN TOTALS / WEIGHTED AVG',
        'IP': total_ip,
        'ERA': weighted_era,
        'FIP': weighted_fip,
        'K%': weighted_kpct,
        'BB%': weighted_bb_pct,
        'WHIP': weighted_whip
    }])

    print(f"🔍 Bullpen summary for {team_abbr}:")
    print(bullpen_summary)

    # Combine summary + details
    combined_df = pd.concat([
        bullpen_summary,
        summary
    ], ignore_index=True)

    # Return sorted detailed table with summary on top
    return combined_df.round(2).sort_values(by='IP', ascending=False)


# %%


def get_recent_team_hitters(team_abbr, as_of_date=None):
    roster_data = {"roster": get_team_roster(team_abbr, as_of_date=as_of_date)}
    hitters = []
    for player in roster_data['roster']:
        if player['position']['abbreviation'] != 'P':
            hitters.append({
                'name': player['person']['fullName'],
                'id': player['person']['id']
            })
    return hitters

# ------------- HOT/COLD SCOUTING -------------
HOT_COLD_TERMINAL_EVENTS = [
    'single', 'double', 'triple', 'home_run', 'strikeout', 'strikeout_double_play',
    'walk', 'intent_walk', 'hit_by_pitch', 'field_out', 'grounded_into_double_play',
    'double_play', 'sac_fly', 'sac_bunt', 'force_out', 'other_out', 'field_error',
    'fielders_choice', 'fielders_choice_out',
]

# `estimated_woba_using_speedangle` is populated on strikeouts and walks as well as on
# batted balls -- statcast fills in the wOBA value those outcomes are worth. So it is
# already a per-plate-appearance quantity and adding walk weights on top double-counts
# them. Verified on a full season: 479 of 493 plate appearances carry a value, including
# every strikeout and walk.


def _hot_cold_sorted(frame):
    """Ordered by departure from the player's own season, which is what the panel is for."""
    if frame is None or frame.empty:
        return frame
    key = ("ΔxwOBA" if "ΔxwOBA" in frame.columns and frame["ΔxwOBA"].notna().any()
           else ("xwOBA" if "xwOBA" in frame.columns else None))
    return frame.sort_values(key, ascending=False) if key else frame


def _hot_cold_metrics(df):
    """One hitter's line over whatever window `df` covers.

    Split out so the recent window and the season baseline are computed by identical code:
    a delta between two differently-computed numbers is not a delta.
    """
    if df is None or df.empty or 'events' not in df.columns:
        return None
    events = df['events'].astype('string')
    terminal = df[events.isin(HOT_COLD_TERMINAL_EVENTS)]
    pa = terminal[['game_pk', 'at_bat_number']].drop_duplicates().shape[0]
    if not pa:
        return None

    hit_events = ['single', 'double', 'triple', 'home_run']
    hits = int(events.isin(hit_events).sum())
    total_bases = int((events == 'single').sum() + (events == 'double').sum() * 2
                      + (events == 'triple').sum() * 3 + (events == 'home_run').sum() * 4)
    # An at-bat is any terminal PA that is not a walk, HBP or sacrifice. Reaching on an
    # error or a fielder's choice is an at-bat, and leaving those out inflated AVG.
    non_ab = ['walk', 'intent_walk', 'hit_by_pitch', 'sac_fly', 'sac_bunt']
    ab = int(events.isin(HOT_COLD_TERMINAL_EVENTS).sum() - events.isin(non_ab).sum())
    walks = int(events.isin(['walk', 'intent_walk']).sum())
    hbp = int((events == 'hit_by_pitch').sum())
    strikeouts = int(events.isin(['strikeout', 'strikeout_double_play']).sum())

    avg = hits / ab if ab else 0.0
    obp = (hits + walks + hbp) / pa if pa else 0.0
    slg = total_bases / ab if ab else 0.0

    launch = pd.to_numeric(df.get('launch_speed'), errors='coerce')
    batted = launch.notna()
    hard_hit_pct = float((launch[batted] >= 95).sum() / batted.sum() * 100) if batted.sum() else 0.0

    # Divided by plate appearances rather than by the count of priced rows, so the handful
    # of PAs statcast never priced (catcher interference, untracked contact) are not quietly
    # dropped from the denominator and inflating the rate.
    priced = pd.to_numeric(df.get('estimated_woba_using_speedangle'), errors='coerce')
    xwoba = float(priced.dropna().sum() / pa) if pa and priced.notna().any() else None

    # Swing decisions. These stabilise in far fewer plate appearances than any rate built on
    # batted-ball luck, so over a two-week window they are the part of the line that is
    # actually measuring the hitter -- and a rising chase rate leads a slump rather than
    # confirming it after the fact.
    description = df['description'].astype('string') if 'description' in df.columns else pd.Series(dtype='string')
    zone = pd.to_numeric(df.get('zone'), errors='coerce')
    swings = description.isin(SWING_DESCRIPTIONS)
    out_of_zone = zone >= 11
    in_zone = zone.between(1, 9)
    chases = int((swings & out_of_zone).sum())
    chase_pct = chases / int(out_of_zone.sum()) * 100 if int(out_of_zone.sum()) else 0.0
    zone_swings = int((swings & in_zone).sum())
    zone_contact = int((swings & in_zone & ~description.isin(WHIFF_DESCRIPTIONS)).sum())
    z_contact_pct = zone_contact / zone_swings * 100 if zone_swings else 0.0

    return {
        'PA': pa, 'AVG': avg, 'OBP': obp, 'SLG': slg, 'OPS': obp + slg,
        'xwOBA': xwoba, 'HardHit%': hard_hit_pct,
        'K%': strikeouts / pa * 100 if pa else 0.0,
        'BB%': walks / pa * 100 if pa else 0.0,
        'Chase%': chase_pct, 'Z-Con%': z_contact_pct,
    }


def generate_hot_cold_hitters(team_abbr, days=14, end_date=None):
    """Recent form, measured against each hitter's own season rather than against .900 OPS.

    **Why the baseline matters.** The old version called a hitter HOT at a .900 OPS over
    fourteen days. For a .950-OPS bat that is a slump and for a .650 bat it is the best
    fortnight of his career, so the flag was mostly re-reporting who the good hitters are --
    the same confound the arsenal study had to control for (docs/arsenal_study.md). Status
    now comes from the gap between the window and the player's own season to date.

    **Why this is not more expensive.** Each hitter's season is pulled once and the recent
    window is a slice of it, so the baseline is free and the whole function makes *fewer*
    API calls than the previous per-window pull.
    """
    today = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.today()
    start_date = (today - timedelta(days=days)).strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')
    season_start = _season_start_date(today.year)

    hitters = get_recent_team_hitters(team_abbr, as_of_date=end_date)
    data = []

    for hitter in tqdm(hitters, desc="Gathering Hot/Cold Hitters"):
        name = hitter['name']
        player_id = hitter['id']
        try:
            season = cached_dataframe_call("statcast_batter", statcast_batter,
                                           season_start, end_date, player_id)
            if season is None or season.empty:
                continue
            # The season pull starts March 1, which is spring training. Exhibition PAs in
            # the baseline would move the very number the status flag is measured against.
            if 'game_type' in season.columns:
                season = season[season['game_type'].astype('string').eq('R')]
            if season.empty:
                continue
            game_day = pd.to_datetime(season['game_date'], errors='coerce')
            window = season[game_day >= pd.Timestamp(start_date)]

            recent = _hot_cold_metrics(window)
            baseline = _hot_cold_metrics(season)
            if recent is None:
                continue

            delta = None
            if baseline and recent['xwOBA'] is not None and baseline['xwOBA'] is not None:
                delta = recent['xwOBA'] - baseline['xwOBA']

            # Thresholds are in xwOBA points against a player's own norm. 40 points is
            # roughly a standard deviation of a two-week window, so this flags a real
            # departure rather than a good week.
            if delta is None:
                status = ""
            elif delta >= 0.040:
                status = "HOT"
            elif delta <= -0.040:
                status = "COLD"
            else:
                status = ""

            data.append({
                'Name': name,
                'PA': recent['PA'],
                'AVG': round(recent['AVG'], 3),
                'OBP': round(recent['OBP'], 3),
                'SLG': round(recent['SLG'], 3),
                'OPS': round(recent['OPS'], 3),
                'xwOBA': round(recent['xwOBA'], 3) if recent['xwOBA'] is not None else None,
                'Szn xwOBA': round(baseline['xwOBA'], 3) if baseline and baseline['xwOBA'] is not None else None,
                'ΔxwOBA': round(delta, 3) if delta is not None else None,
                'HardHit%': round(recent['HardHit%'], 1),
                'Chase%': round(recent['Chase%'], 1),
                'Z-Con%': round(recent['Z-Con%'], 1),
                'K%': round(recent['K%'], 1),
                'BB%': round(recent['BB%'], 1),
                'Status': status,
            })
        except Exception as e:
            print(f"Failed for {name}: {e}")
            continue
    if not data:
        return pd.DataFrame(columns=['Name', 'PA', 'AVG', 'OBP', 'SLG', 'OPS', 'xwOBA',
                                     'Szn xwOBA', 'ΔxwOBA', 'HardHit%', 'Chase%',
                                     'Z-Con%', 'K%', 'BB%', 'Status'])
    df = pd.DataFrame(data)
    df = df[df['PA'] > 5]
    sort_col = 'ΔxwOBA' if df['ΔxwOBA'].notna().any() else 'OPS'
    return df.sort_values(sort_col, ascending=False)

# ------------- STARTING LINEUP + SEASON STATS -------------
def get_hand_by_id(mlbam_id):
    try:
        url = f"https://statsapi.mlb.com/api/v1/people/{mlbam_id}"
        data = cached_json_request(url, namespace="statsapi")
        return data['people'][0]['batSide']['code']  # 'L', 'R', or 'S'
    except Exception as e:
        print(f"❌ Failed to get handedness for player {mlbam_id}: {e}")
    return None


def get_today_lineup(team_abbr, game_date=None):
    team_id = get_team_id(team_abbr)
    target_date = game_date or datetime.today().date()
    schedule = statsapi.schedule(team=team_id, start_date=target_date, end_date=target_date)

    if not schedule:
        print("⚠️ No scheduled game found.")
        return []

    game = schedule[0]
    game_id = game['game_id']
    box = statsapi.boxscore_data(game_id)

    is_home = game['home_id'] == team_id
    side = 'home' if is_home else 'away'
    players = box[side]['players']

    lineup_with_details = []
    seen_spots = set()
    for player in players.values():
        name = remove_accents((player['person']['fullName']))
        batting_order = player.get('battingOrder')
        pos_info = player.get('position', {})
        player_id = player['person']['id']
        handed = get_hand_by_id(player_id)


        if batting_order and pos_info.get('code') != '1':
            try:
                order_num = int(batting_order)
                spot = order_num // 100
                if 100 <= order_num <= 999 and 1 <= spot <= 9 and spot not in seen_spots:
                    seen_spots.add(spot)
                    lineup_with_details.append({
                        'Name': name,
                        '#': order_num,
                        'Pos': pos_info.get('abbreviation'),
                        'Bats': handed,
                        'ID': player_id
                    })
            except:
                continue

    # Sort by batting order (101, 201, ..., 901)
    lineup_with_details.sort(key=lambda x: x['#'])

    return lineup_with_details


def get_latest_lineup(team_abbr, max_days_back=5, return_metadata=True, as_of_date=None):
    team_id = get_team_id(team_abbr)
    today = datetime.strptime(as_of_date, "%Y-%m-%d").date() if as_of_date else datetime.today().date()

    for i in range(max_days_back):
        date = today - timedelta(days=i)
        schedule = statsapi.schedule(team=team_id, start_date=date, end_date=date)
        if not schedule:
            continue

        game = schedule[0]
        game_id = game['game_id']
        box = statsapi.boxscore_data(game_id)

        side = 'home' if game['home_id'] == team_id else 'away'
        players = box[side]['players']

        seen_spots = set()
        lineup = []

        for player in players.values():
            name = remove_accents(player['person']['fullName'])
            batting_order = player.get('battingOrder')

            # Skip if no batting order
            if not batting_order:
                continue

            try:
                order_val = int(batting_order)
            except:
                continue

            spot = order_val // 100
            if 100 <= order_val <= 999 and 1 <= spot <= 9:
                pos_abbr = player.get('position', {}).get('abbreviation')
                player_id = player['person']['id']
                handed = get_hand_by_id(player_id)

                # Skip PH and players with no real position
                if pos_abbr == 'PH' or pos_abbr is None:
                    continue

                # Only keep the first player listed for each batting-order slot.
                if spot in seen_spots:
                    continue

                seen_spots.add(spot)
                lineup.append({
                    'Name': name,
                    '#': order_val,
                    'Pos': pos_abbr,
                    'Bats': handed,
                    'ID': player_id
                })

        if lineup:
            lineup.sort(key=lambda x: x['#'])
            return lineup if return_metadata else [p['Name'] for p in lineup]

    return []


# ------------- PROJECTED LINEUP (pre-posting) -------------
#
# Before MLB posts a lineup we used to replay the team's most recent one verbatim, which
# carries two errors that both feed straight into the projections: it ignores the platoon
# (yesterday's RHP lineup is the wrong nine against today's lefty) and it ignores
# availability (a player who hit the IL after that game still shows up in the report).
#
# The projector instead finds the configuration -- the ordered nine, not nine independent
# slot picks -- that the club uses most against the hand it is facing today, restricted to
# players on the active roster, then patches any holes like-for-like by position. Managers
# run a stable card and substitute into it rather than re-sorting the batting order, so
# keeping the configuration intact is both truer to the behaviour and self-protecting: a
# real historical lineup is guaranteed to field one catcher, one shortstop and so on, which
# a per-slot modal pick is not.

# Uses decay by how many same-hand games ago they were, so a card the team moved off a
# month ago loses to the one it has run all week without being discarded outright.
_LINEUP_RECENCY_DECAY = 0.93
_LINEUP_HISTORY_GAMES = 40


def get_available_player_ids(team_abbr, as_of_date=None):
    """MLBAM ids on the team's active roster as of `as_of_date`.

    The active roster is the availability signal: it already excludes every flavour of IL,
    optioned and released, and the endpoint is date-aware, so historical backfills see the
    roster as it stood that day rather than today's."""
    try:
        roster = get_team_roster(team_abbr, roster_type="active", as_of_date=as_of_date)
    except Exception as e:
        print(f"⚠️ Active roster unavailable for {team_abbr}: {e}")
        return set()
    ids = set()
    for entry in roster:
        pid = (entry.get("person") or {}).get("id")
        if pid:
            ids.add(int(pid))
    return ids


def _starting_pitcher_hand(box_team):
    """Handedness of the pitcher who started, from one side of a boxscore payload."""
    pitchers = box_team.get("pitchers") or []
    if not pitchers:
        return None
    hand = get_pitcher_handedness(int(pitchers[0]))
    return hand if hand in ("L", "R") else None


def _boxscore_lineup_rows(box_team):
    """(spot, id, name, pos) for the nine who started, from one side of a boxscore.

    MLB shares a batting slot between the starter and everyone who replaced them, encoding
    depth in the last two digits -- 100 is the starter at slot 1, 101 and 102 the men who
    followed. Taking the lowest value per slot picks the starter deterministically; taking
    whichever the payload happened to list first can hand back a pinch hitter."""
    best_by_spot = {}
    for player in (box_team.get("players") or {}).values():
        batting_order = player.get("battingOrder")
        if not batting_order:
            continue
        try:
            order_val = int(batting_order)
        except (TypeError, ValueError):
            continue
        spot = order_val // 100
        if not (100 <= order_val <= 999 and 1 <= spot <= 9):
            continue
        if spot in best_by_spot and best_by_spot[spot][0] <= order_val:
            continue
        # `position` is where the player FINISHED the game, so a defensive shuffle leaves
        # the card with two RF and no CF -- which then makes the like-for-like hole filling
        # match on a position the player never actually started at. allPositions is ordered
        # by when each was played, so its first entry is the true starting position.
        all_positions = player.get("allPositions") or []
        pos_abbr = (all_positions[0] or {}).get("abbreviation") if all_positions else None
        if not pos_abbr:
            pos_abbr = (player.get("position") or {}).get("abbreviation")
        # A pitcher batting is a non-DH NL-rules game; keep it, but drop pinch hitters and
        # anyone the API left unpositioned -- those are substitutes, not the starting card.
        if pos_abbr in (None, "PH", "PR"):
            continue
        person = player.get("person") or {}
        pid = person.get("id")
        if not pid:
            continue
        best_by_spot[spot] = (order_val, int(pid), remove_accents(person.get("fullName", "")), pos_abbr)
    return [(spot, pid, name, pos) for spot, (_ov, pid, name, pos) in sorted(best_by_spot.items())]


def get_lineup_history(team_abbr, as_of_date=None, games_back=_LINEUP_HISTORY_GAMES):
    """Newest-first starting lineups for a team, each tagged with the hand it faced.

    One cached boxscore call per game supplies both the lineup and the opposing starter, so
    the whole history costs `games_back` requests the first time and nothing after -- a
    completed boxscore never changes."""
    try:
        team_id = get_team_id(team_abbr)
    except Exception as e:
        print(f"⚠️ Could not resolve team {team_abbr} for lineup history: {e}")
        return []

    end = datetime.strptime(as_of_date, "%Y-%m-%d").date() if as_of_date else datetime.today().date()
    # ~1.6 days per game covers off-days with room to spare for the requested window.
    start = end - timedelta(days=int(games_back * 1.7) + 10)
    try:
        data = cached_json_request(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={"sportId": 1, "teamId": int(team_id), "gameType": "R",
                    "startDate": start.strftime("%Y-%m-%d"), "endDate": end.strftime("%Y-%m-%d")},
            namespace="statsapi",
            cache_key_extra=str(end),
        )
    except Exception as e:
        print(f"⚠️ Could not list games for {team_abbr} lineup history: {e}")
        return []

    metas = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            if game.get("gameType") != "R":
                continue
            if game.get("status", {}).get("detailedState") != "Final":
                continue
            if game.get("officialDate", "") >= str(end):
                continue          # today's game hasn't been played
            metas.append((game.get("officialDate", ""), game.get("gameNumber", 1), game.get("gamePk")))
    metas.sort(reverse=True)
    metas = metas[:games_back]

    history = []
    for game_date, _game_number, game_pk in metas:
        if not game_pk:
            continue
        try:
            box = cached_json_request(
                f"https://statsapi.mlb.com/api/v1/game/{int(game_pk)}/boxscore",
                namespace="statsapi",
            )
        except Exception:
            continue
        teams = box.get("teams") or {}
        home, away = teams.get("home") or {}, teams.get("away") or {}
        is_home = ((home.get("team") or {}).get("id")) == int(team_id)
        own, opp = (home, away) if is_home else (away, home)

        rows = _boxscore_lineup_rows(own)
        if len(rows) != 9:
            continue
        history.append({
            "date": game_date,
            "hand": _starting_pitcher_hand(opp),
            "lineup": rows,
        })
    return history


def _position_start_rates(history, hand=None):
    """{position: [(player_id, name, weighted starts), ...]} ranked, for hole filling."""
    by_pos = {}
    idx = 0
    for game in history:
        if hand and game.get("hand") and game["hand"] != hand:
            continue
        weight = _LINEUP_RECENCY_DECAY ** idx
        idx += 1
        for _spot, pid, name, pos in game["lineup"]:
            slot = by_pos.setdefault(pos, {})
            prev_name, prev_weight = slot.get(pid, (name, 0.0))
            slot[pid] = (prev_name, prev_weight + weight)
    return {
        pos: sorted(((pid, nm, w) for pid, (nm, w) in players.items()),
                    key=lambda t: t[2], reverse=True)
        for pos, players in by_pos.items()
    }


def _modal_configuration(history, hand, available_ids):
    """The team's most-used ordered nine versus `hand`, among available players.

    Configurations are scored by decayed usage and then discounted by how many of their nine
    are unavailable. The discount is multiplicative rather than a flat subtraction so it
    stays scale-free: a card the team has run four times with one man out still outranks one
    it used once with all nine on hand, which is the right call because the four-time card
    patched at one slot is closer to what the manager will actually write."""
    counts = {}
    idx = 0
    for game in history:
        if hand and game.get("hand") and game["hand"] != hand:
            continue
        key = tuple(pid for _spot, pid, _name, _pos in game["lineup"])
        weight = _LINEUP_RECENCY_DECAY ** idx
        idx += 1
        entry = counts.get(key)
        if entry:
            entry["weight"] += weight
            entry["uses"] += 1
        else:
            counts[key] = {"weight": weight, "uses": 1, "lineup": game["lineup"], "date": game["date"]}

    if not counts:
        return None

    best, best_score = None, -1.0
    for entry in counts.values():
        available_n = sum(1 for pid in (p for _s, p, _n, _po in entry["lineup"]) if pid in available_ids)
        score = entry["weight"] * (available_n / 9.0) ** 3
        if score > best_score or (score == best_score and best and entry["date"] > best["date"]):
            best, best_score = entry, score
    return best


def project_lineup(team_abbr, opposing_hand=None, as_of_date=None, history=None):
    """Best guess at today's card: modal same-hand configuration, availability-filtered.

    Returns the same row shape as `get_today_lineup` plus a `_meta` dict describing how many
    of the nine slots survived from the source configuration, so callers can grade their own
    confidence instead of treating every unposted lineup alike."""
    history = get_lineup_history(team_abbr, as_of_date=as_of_date) if history is None else history
    if not history:
        return [], {}

    available_ids = get_available_player_ids(team_abbr, as_of_date=as_of_date)
    if not available_ids:
        # No roster signal -- treat everyone who has started recently as available rather
        # than filtering the whole lineup away.
        available_ids = {pid for g in history for _s, pid, _n, _p in g["lineup"]}

    hand = opposing_hand if opposing_hand in ("L", "R") else None
    same_hand_games = sum(1 for g in history if g.get("hand") == hand) if hand else 0
    # Under about five same-hand looks the platoon signal is thinner than the noise it adds,
    # so fall back to the club's overall card rather than over-fitting three games.
    hand_used = hand if same_hand_games >= 5 else None

    best = _modal_configuration(history, hand_used, available_ids)
    if not best:
        best = _modal_configuration(history, None, available_ids)
        hand_used = None
    if not best:
        return [], {}

    rates = _position_start_rates(history, hand_used)
    fallback_rates = _position_start_rates(history, None)

    kept = [(spot, pid, name, pos) for spot, pid, name, pos in best["lineup"] if pid in available_ids]
    used_ids = {pid for _s, pid, _n, _p in kept}
    holes = [(spot, pid, name, pos) for spot, pid, name, pos in best["lineup"] if pid not in available_ids]

    filled = list(kept)
    substitutions = []
    for spot, _old_id, old_name, pos in holes:
        # Like-for-like: the most-used available player at that same position, which keeps
        # the defensive alignment of the source configuration legal.
        replacement = None
        for candidates in (rates.get(pos, []), fallback_rates.get(pos, [])):
            for pid, name, _weight in candidates:
                if pid in available_ids and pid not in used_ids:
                    replacement = (pid, name)
                    break
            if replacement:
                break
        if not replacement:
            # Nobody on the roster has started there recently -- fall back to the highest
            # weighted bat left over at any position.
            leftovers = sorted(
                ((pid, nm, w) for pos_players in fallback_rates.values() for pid, nm, w in pos_players
                 if pid in available_ids and pid not in used_ids),
                key=lambda t: t[2], reverse=True)
            if leftovers:
                replacement = (leftovers[0][0], leftovers[0][1])
        if not replacement:
            continue
        used_ids.add(replacement[0])
        substitutions.append(f"{old_name} -> {replacement[1]} ({pos})")
        filled.append((spot, replacement[0], replacement[1], pos))

    filled.sort()
    lineup_info = [{
        "Name": name,
        "#": spot * 100 + 1,
        "Pos": pos,
        "Bats": get_hand_by_id(pid),
        "ID": pid,
    } for spot, pid, name, pos in filled]

    meta = {
        "slots_kept": len(kept),
        "slots_filled": len(filled),
        "substitutions": substitutions,
        "hand": hand_used,
        "source_date": best.get("date"),
        "uses": best.get("uses"),
        "same_hand_games": same_hand_games,
    }
    return lineup_info, meta



def get_season_stats_for_players(player_names, year=None):
    if not year:
        year = datetime.today().year
    try:
        all_hitters = cached_dataframe_call("batting_stats", batting_stats, year, qual=0)
    except Exception as e:
        print(f"⚠️ Season hitter leaderboard unavailable: {e}")
        return pd.DataFrame([{
            'Name': name,
            'PA': 0,
            'AVG': 0.000,
            'OBP': 0.000,
            'SLG': 0.000,
            'OPS': 0.000,
            'ISO': 0.000,
            'HR': 0,
            'RBI': 0,
            'SB': 0,
            'WAR': 0.0
        } for name in player_names])
    all_hitters['Name'] = all_hitters['Name'].str.strip()

    matched = all_hitters[all_hitters['Name'].isin(player_names)].copy()
    missing_names = set(player_names) - set(matched['Name'])

    # Log missing
    print("\n🔍 Missing player stats for:")
    print(sorted(list(missing_names)))

    # Add fallbacks with zeros
    for name in missing_names:
        fallback_row = pd.Series({
            'Name': name,
            'PA': 0,
            'AVG': 0.000,
            'OBP': 0.000,
            'SLG': 0.000,
            'OPS': 0.000,
            'HR': 0,
            'RBI': 0
        })
        matched = pd.concat([matched, pd.DataFrame([fallback_row])], ignore_index=True)

    return matched[['Name', 'PA', 'AVG', 'OBP', 'SLG', 'OPS', 'ISO', 'HR', 'RBI', 'SB', 'WAR']]


# ------------- MASTER REPORT FUNCTION -------------


def  generate_team_hitter_report(team_abbr, pitcher, statcast_splits_df, game_date=None):
    opposing_hand = get_pitcher_handedness(pitcher)
    hotcold_end_date = _pregame_end_date(game_date) if game_date else None
    hotcold = generate_hot_cold_hitters(team_abbr, end_date=hotcold_end_date)
    lineup_info = get_today_lineup(team_abbr, game_date=game_date)
    lineup_source = "actual"
    lineup_confidence = "Confirmed"

    # A part-posted card is not a confirmed one -- MLB occasionally exposes a handful of
    # slots early, and replaying those alone would drop real bats from the report.
    if lineup_info and len(lineup_info) < 9:
        print(f"⚠️ {team_abbr} lineup only {len(lineup_info)}/9 posted. Projecting instead.")
        lineup_info = []

    if not lineup_info:
        projected, proj_meta = project_lineup(team_abbr, opposing_hand=opposing_hand, as_of_date=game_date)
        if projected:
            lineup_info = projected
            lineup_source = "projected"
            kept = proj_meta.get("slots_kept", 0)
            lineup_confidence = "Projected" if kept == 9 else f"Projected ({kept}/9)"
            hand_note = f"vs {proj_meta['hand']}HP" if proj_meta.get("hand") else "all hands"
            print(f"⚠️ No lineup posted for {team_abbr}. Projected from modal configuration "
                  f"({hand_note}, used {proj_meta.get('uses')}x, last {proj_meta.get('source_date')}).")
            for note in proj_meta.get("substitutions", []):
                print(f"   ↳ unavailable, substituted: {note}")

    if not lineup_info:
        print("⚠️ Could not project a lineup. Using most recent available lineup instead.")
        lineup_info = get_latest_lineup(team_abbr, return_metadata=True, as_of_date=game_date)
        lineup_source = "recent/fallback"
        lineup_confidence = "Fallback"

    if not lineup_info:
        print("❌ No lineup data available from recent games either.")
        return hotcold, pd.DataFrame(), pd.DataFrame()

    lineup = [player['Name'] for player in lineup_info]
    lineup_meta = pd.DataFrame(lineup_info)
    if not lineup_meta.empty and '#' in lineup_meta.columns:
        lineup_meta['Spot'] = (pd.to_numeric(lineup_meta['#'], errors='coerce') // 100).astype('Int64')
        lineup_meta = lineup_meta.drop(columns=['#'])
    if not lineup_meta.empty:
        lineup_meta["Lineup Source"] = lineup_source
        lineup_meta["Lineup Confidence"] = lineup_confidence

    player_ids = {player.get('Name'): player.get('ID') for player in lineup_info if player.get('ID')}
    report_year = int(game_date[:4]) if game_date else None
    season_stats = get_season_stats_for_players(lineup, year=report_year, player_ids=player_ids, as_of_date=game_date)
    if not lineup_meta.empty and not season_stats.empty:
        season_stats = lineup_meta.merge(season_stats, on='Name', how='left')
        numeric_fallbacks = {
            'AB': 0, 'PA': 0, 'AVG': 0.000, 'OBP': 0.000, 'SLG': 0.000, 'OPS': 0.000,
            'ISO': 0.000, 'HR': 0, 'RBI': 0, 'SB': 0, 'WAR': 0.0
        }
        for col, fallback in numeric_fallbacks.items():
            if col in season_stats.columns:
                season_stats[col] = season_stats[col].fillna(fallback)

    # Merge statcast splits
    id_map = {name: player_ids.get(name) or lookup_player_id(name) for name in lineup}
    split_rows = []

    for name in lineup:
        batter_id = id_map.get(name)
        if not batter_id:
            continue

        rows = statcast_splits_df[(statcast_splits_df['batter'] == batter_id) & 
                                  (statcast_splits_df['p_throws'] == opposing_hand)]
        if not rows.empty:
            row = rows.iloc[0]
            pa = row.get('PA', 0)
            split_ab = row.get('AB', 0)
            hits = row.get('Hits', 0)
            doubles = row.get('Doubles', 0)
            triples = row.get('Triples', 0)
            hr = row.get('HR', 0)
            bb = row.get('BB', 0)
            k = row.get('K', 0)
            singles = max(0, hits - doubles - triples - hr)
            total_bases = singles + (2 * doubles) + (3 * triples) + (4 * hr)

            avg = hits / pa if pa > 0 else 0
            obp = (hits + bb) / pa if pa > 0 else 0
            slg = total_bases / pa if pa > 0 else 0
            ops = obp + slg
            iso = slg - avg
            bb_pct = bb / pa * 100 if pa > 0 else 0
            k_pct = k / pa * 100 if pa > 0 else 0

            split_rows.append({
                'Name': name,
                f'AB vs {opposing_hand}': split_ab,
                f'PA vs {opposing_hand}': pa,
                f'AVG vs {opposing_hand}': round(avg, 3),
                f'OBP vs {opposing_hand}': round(obp, 3),
                f'SLG vs {opposing_hand}': round(slg, 3),
                f'OPS vs {opposing_hand}': round(ops, 3),
                f'ISO vs {opposing_hand}': round(iso, 3),
                f'HR vs {opposing_hand}': hr,
                f'BB% vs {opposing_hand}': round(bb_pct, 1),
                f'K% vs {opposing_hand}': round(k_pct, 1)
            })

    splits_df = pd.DataFrame(split_rows)

    return hotcold, season_stats, splits_df




#%%

def convert_ip(ip_raw):
    ip_int = int(ip_raw)
    ip_dec = round((ip_raw - ip_int) * 10)  # 0.1, 0.2, or 0.0
    if ip_dec > 2:
        print(f"⚠️ Invalid fractional IP: {ip_raw}")
        return ip_raw  # fallback
    return ip_int + (ip_dec / 3)


def get_league_pitching_stats(season=2025):
    url = f"https://statsapi.mlb.com/api/v1/teams/stats?stats=season&group=pitching&season={season}"
    try:
        data = cached_json_request(url, namespace="statsapi")

        total_er = 0.0
        total_ip = 0.0
        total_hr = 0.0
        total_bb = 0.0
        total_k = 0.0

        for team in data.get("stats", []):
            for split in team.get("splits", []):
                stat = split.get("stat", {})
                total_er += float(stat.get("earnedRuns", 0))
                total_ip += _ip_to_float(stat.get("inningsPitched", 0))
                total_hr += float(stat.get("homeRuns", 0))
                total_bb += float(stat.get("baseOnBalls", 0))
                total_k += float(stat.get("strikeOuts", 0))

        if total_ip == 0:
            return None, None

        league_era = (total_er * 9) / total_ip
        league_fip_raw = ((13 * total_hr) + (3 * total_bb) - (2 * total_k)) / total_ip
        fip_constant = league_era - league_fip_raw

        return round(league_era, 2), round(fip_constant, 2)
    except Exception as e:
        print(f"⚠️ Error fetching league stats: {e}")
        return None, None

def calculate_fip(hr, bb, k, ip, fip_constant=3.1):
    if ip == 0:
        return None
    return round(((13 * hr) + (3 * bb) - (2 * k)) / ip + fip_constant, 2)


def calculate_game_score(ip, h, r, er, bb, so):
    outs = int(round(ip * 3))
    innings = int(ip)  # only full innings count for bonus

    unearned = max(0, r - er)

    gs = 50
    gs += outs
    gs += 2 * max(0, innings - 4)
    gs += so
    gs -= 2 * h
    gs -= 4 * er
    gs -= 2 * unearned
    gs -= bb

    return gs



def get_last_n_starts_direct(pitcher_id, season="2025", last_n=5, end_date=None):
    url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats"
    params = {
        "stats": "gameLog",
        "group": "pitching",
        "season": season
    }
    data = cached_json_request(url, params=params, namespace="statsapi", cache_key_extra=end_date)

    game_logs = []
    logs = data.get("stats", [])
    if not logs:
        return []

    all_games = logs[0].get("splits", [])

    for g in reversed(all_games):  # ✅ newest games first
        if end_date and g.get("date") and g["date"] > end_date:
            continue
        s = g["stat"]
        if s.get("gamesStarted", "0") == "0":
            continue  # skip relief appearances

        ip = _ip_to_float(s.get("inningsPitched", 0))
        outs = round(ip * 3)
        h = int(s.get("hits", 0))
        r = int(s.get("runs", 0))
        er = int(s.get("earnedRuns", 0))
        bb = int(s.get("baseOnBalls", 0))
        so = int(s.get("strikeOuts", 0))
        hr = int(s.get("homeRuns", 0))
        pitches = int(s.get("numberOfPitches", 0))

        game_score = calculate_game_score(ip, h, r, er, bb, so)

        opp_team = TEAM_NAME_MAP.get(g['opponent']['name'], g['opponent']['name'])
        home_away = g['isHome']
        game_logs.append({
            "Date": g['date'],
            "Opponent": f"{'vs' if home_away else '@'} {opp_team}",
            "IP": ip,
            "H": h,
            "R": r,
            "ER": er,
            "BB": bb,
            "SO": so,
            "HR": hr,
            "Pitches": pitches,
            "Game Score": game_score
        })

        if len(game_logs) >= last_n:
            break

    return game_logs


def get_starts_vs_team(pitcher_id, opponent_abbr, season, end_date=None):
    """This-season starts by a pitcher against a specific opponent (abbr), newest first."""
    if not pitcher_id or not opponent_abbr:
        return []
    url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats"
    params = {"stats": "gameLog", "group": "pitching", "season": season}
    data = cached_json_request(url, params=params, namespace="statsapi", cache_key_extra=end_date)
    logs = data.get("stats", [])
    if not logs:
        return []
    game_logs = []
    for g in reversed(logs[0].get("splits", [])):
        if end_date and g.get("date") and g["date"] > end_date:
            continue
        s = g["stat"]
        if str(s.get("gamesStarted", "0")) == "0":
            continue
        opp_name = g.get("opponent", {}).get("name", "")
        if (get_team_abbreviation(opp_name) or "").upper() != str(opponent_abbr).upper():
            continue
        ip = _ip_to_float(s.get("inningsPitched", 0))
        h = int(s.get("hits", 0))
        r = int(s.get("runs", 0))
        er = int(s.get("earnedRuns", 0))
        bb = int(s.get("baseOnBalls", 0))
        so = int(s.get("strikeOuts", 0))
        hr = int(s.get("homeRuns", 0))
        game_logs.append({
            "Date": g["date"],
            "Opponent": f"{'vs' if g.get('isHome') else '@'} {opponent_abbr}",
            "IP": ip, "H": h, "R": r, "ER": er, "BB": bb, "SO": so, "HR": hr,
            "Pitches": int(s.get("numberOfPitches", 0)),
            "Game Score": calculate_game_score(ip, h, r, er, bb, so),
        })
    return game_logs


def build_rest_schedule(team_abbr, team_id, starter_id, season, game_date, bullpen_df=None):
    """Rest + schedule-spot summary: team days rest, SP days rest, schedule note, bullpen availability."""
    columns = ["Team", "Rest", "SP Rest", "Note", "Pen A/T"]
    try:
        gd = datetime.strptime(game_date, "%Y-%m-%d")
    except Exception:
        return pd.DataFrame(columns=columns)

    team_rest, schedule_note = "N/A", "normal"
    prior_hour = None
    try:
        sched = statsapi.schedule(
            start_date=(gd - timedelta(days=18)).strftime("%Y-%m-%d"),
            end_date=(gd - timedelta(days=1)).strftime("%Y-%m-%d"),
            team=team_id,
        )
        finals = [g for g in sched if g.get("status") == "Final" and g.get("game_date")]
        dset = sorted({g["game_date"] for g in finals})
        if dset:
            last = datetime.strptime(dset[-1], "%Y-%m-%d")
            rest_days = (gd - last).days
            team_rest = f"{rest_days}d"
            latest_game = max((g for g in finals if g["game_date"] == dset[-1]), key=lambda g: g.get("game_datetime", ""), default=None)
            prior_hour = _local_start(latest_game)[1] if latest_game else None
            # consecutive-game streak ending the day before
            streak, dd = 0, gd - timedelta(days=1)
            present = set(dset)
            while dd.strftime("%Y-%m-%d") in present:
                streak += 1
                dd -= timedelta(days=1)
            notes = []
            if rest_days >= 2:
                notes.append(f"{rest_days-1} off day(s) before")
            if streak >= 4:
                notes.append(f"{streak} straight games")
            schedule_note = "; ".join(notes) or "normal"
    except Exception:
        pass

    # Get-away day: night game yesterday, day game today -- in venue-local time. Comparing
    # UTC hours here called every Central/Mountain/Pacific night game a day game, so this
    # note contradicted the Schedule Spot row beside it.
    try:
        today = statsapi.schedule(start_date=game_date, end_date=game_date, team=team_id)
        today_hour = _local_start(today[0])[1] if today else None
        if (prior_hour is not None and today_hour is not None
                and prior_hour >= DAY_GAME_BEFORE_HOUR and today_hour < DAY_GAME_BEFORE_HOUR):
            schedule_note = (schedule_note + "; get-away day") if schedule_note != "normal" else "get-away day"
    except Exception:
        pass

    sp_rest = "N/A"
    try:
        logs = get_last_n_starts_direct(starter_id, str(season), last_n=1, end_date=(gd - timedelta(days=1)).strftime("%Y-%m-%d"))
        if logs:
            last_start = datetime.strptime(logs[0]["Date"], "%Y-%m-%d")
            sp_rest = f"{(gd - last_start).days}d"
    except Exception:
        pass

    pen = "N/A"
    if bullpen_df is not None and not bullpen_df.empty and "Availability" in bullpen_df.columns:
        avail = int((bullpen_df["Availability"] == "Available").sum())
        taxed = int((bullpen_df["Availability"] == "Taxed").sum())
        pen = f"{avail} / {taxed}"

    return pd.DataFrame([{
        "Team": team_abbr,
        "Rest": team_rest,
        "SP Rest": sp_rest,
        "Note": schedule_note,
        "Pen A/T": pen,
    }], columns=columns)


def build_starter_rest_splits(pitcher_id, season, end_date=None):
    """This starter's performance bucketed by days between starts: GS/IP/ERA/decisions."""
    columns = ["Rest", "GS", "IP", "ERA", "W-L"]
    if not pitcher_id:
        return pd.DataFrame(columns=columns)
    url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats"
    params = {"stats": "gameLog", "group": "pitching", "season": season}
    data = cached_json_request(url, params=params, namespace="statsapi", cache_key_extra=end_date)
    logs = data.get("stats", [])
    if not logs:
        return pd.DataFrame(columns=columns)
    starts = []
    for g in logs[0].get("splits", []):
        if end_date and g.get("date") and g["date"] > end_date:
            continue
        if str(g.get("stat", {}).get("gamesStarted", "0")) == "0":
            continue
        starts.append(g)
    starts.sort(key=lambda g: g.get("date", ""))
    buckets = {"<=4d": [], "5d": [], "6d+": []}
    prev = None
    for g in starts:
        try:
            d = datetime.strptime(g["date"], "%Y-%m-%d")
        except Exception:
            continue
        if prev is not None:
            rest = (d - prev).days
            label = "<=4d" if rest <= 4 else "5d" if rest == 5 else "6d+"
            s = g["stat"]
            buckets[label].append((
                _ip_to_float(s.get("inningsPitched", 0)), int(s.get("earnedRuns", 0)),
                _stat_int(s, "wins"), _stat_int(s, "losses"),
            ))
        prev = d
    rows = []
    for label in ["<=4d", "5d", "6d+"]:
        items = buckets[label]
        if not items:
            continue
        ip = sum(i[0] for i in items)
        er = sum(i[1] for i in items)
        w = sum(i[2] for i in items)
        l = sum(i[3] for i in items)
        rows.append({
            "Rest": label, "GS": len(items), "IP": round(ip, 1),
            "ERA": round(er * 9 / ip, 2) if ip else 0.0, "W-L": f"{w}-{l}",
        })
    return pd.DataFrame(rows, columns=columns)


_TRIP_SCHEDULE_MEMORY = {}

# IANA zones for every MLB and temporary/neutral venue seen in recent schedules.
# IANA names are essential here: fixed "ET/CT/PT" offsets mishandle daylight saving
# and Phoenix, which aligns with Pacific time for most of the baseball season.
_VENUE_IANA_TZ = {
    # Eastern
    2: "America/New_York", 3: "America/New_York", 5: "America/New_York",
    12: "America/New_York", 14: "America/Toronto", 31: "America/New_York",
    2394: "America/New_York", 2523: "America/New_York", 2536: "America/New_York",
    2602: "America/New_York", 2681: "America/New_York", 2735: "America/New_York",
    2756: "America/New_York", 3289: "America/New_York", 3309: "America/New_York",
    3313: "America/New_York", 4169: "America/New_York", 4705: "America/New_York",
    6130: "America/New_York",
    # Central
    4: "America/Chicago", 7: "America/Chicago", 17: "America/Chicago",
    32: "America/Chicago", 2392: "America/Chicago", 2889: "America/Chicago",
    3312: "America/Chicago", 3949: "America/Chicago", 5325: "America/Chicago",
    # Mountain / Arizona
    15: "America/Phoenix", 19: "America/Denver",
    # Pacific
    1: "America/Los_Angeles", 10: "America/Los_Angeles", 22: "America/Los_Angeles",
    680: "America/Los_Angeles", 2395: "America/Los_Angeles",
    2529: "America/Los_Angeles", 2680: "America/Los_Angeles",
    # International series
    2397: "Asia/Tokyo", 5150: "Asia/Seoul", 5340: "America/Mexico_City",
    5381: "Europe/London",
}

_TEAM_HOME_VENUE_ID = {
    108: 1, 109: 15, 110: 2, 111: 3, 112: 17, 113: 2602, 114: 5,
    115: 19, 116: 2394, 117: 2392, 118: 7, 119: 22, 120: 3309,
    121: 3289, 133: 2529, 134: 31, 135: 2680, 136: 680, 137: 2395,
    138: 2889, 139: 2523, 140: 5325, 141: 14, 142: 3312, 143: 2681,
    144: 4705, 145: 4, 146: 4169, 147: 3313, 158: 32,
}


def _venue_zone(venue_id, fallback_team_id=None):
    venue_id = _safe_number(venue_id, None)
    if venue_id is not None:
        zone = _VENUE_IANA_TZ.get(int(venue_id))
        if zone:
            return zone
    fallback_venue = _TEAM_HOME_VENUE_ID.get(int(fallback_team_id)) if fallback_team_id else None
    return _VENUE_IANA_TZ.get(fallback_venue)


def _zone_offset_hours(zone_name, when):
    if not zone_name:
        return None
    try:
        if isinstance(when, str) and "T" in when:
            utc_dt = datetime.fromisoformat(when.replace("Z", "+00:00"))
            return utc_dt.astimezone(ZoneInfo(zone_name)).utcoffset().total_seconds() / 3600
        date_text = str(when)[:10]
        local_noon = datetime.strptime(date_text, "%Y-%m-%d").replace(
            hour=12, tzinfo=ZoneInfo(zone_name)
        )
        return local_noon.utcoffset().total_seconds() / 3600
    except (TypeError, ValueError, KeyError):
        return None


def _zone_abbr(zone_name, when):
    if not zone_name:
        return "?"
    try:
        date_text = str(when)[:10]
        local_noon = datetime.strptime(date_text, "%Y-%m-%d").replace(
            hour=12, tzinfo=ZoneInfo(zone_name)
        )
        return local_noon.tzname() or "?"
    except (TypeError, ValueError, KeyError):
        return "?"


# Schedule-spot bucketing, in one place.
#
# These rules are all about *local* time -- was yesterday a night game, is today a day game,
# were there two games on one date -- and statsapi reports first pitch in UTC. A 7:10pm
# Pacific start is 02:10 UTC the *next day*, so reading `.hour` and `.date()` off the UTC
# stamp classified it as a day game on the wrong date. Measured over five dates of the 2026
# schedule: 28% of games came out on the wrong side of day/night and 26% landed on the wrong
# calendar date, which is why the note and the season table disagreed with each other and
# both disagreed with reality.
#
# The season table and tonight's note used to implement these rules separately. They now
# share `_schedule_spot`, because two copies of a six-branch precedence chain is the other
# half of how they came to disagree.
SCHEDULE_SPOTS = ["DH game 2", "Day after DH", "Extra rest (2+d)", "Get-away day",
                  "Day after night", "Normal"]

# Local first pitch before 5pm is a day game. MLB day games run 12:05-4:10 local and night
# games start 6:05 or later, so nothing real sits near the boundary.
DAY_GAME_BEFORE_HOUR = 17


def _local_start(game):
    """(local date, local hour) for one statsapi schedule row, or (None, None).

    The date comes from `game_date` -- MLB's official date for the game, which is what the
    standings and the doubleheader definition use -- and only the hour is derived from the
    UTC stamp. That way a late start cannot roll the game onto the following date.
    """
    stamp = str(game.get("game_datetime") or "").strip()
    official = str(game.get("game_date") or "")[:10]
    try:
        date = datetime.strptime(official, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None, None
    zone = _venue_zone(game.get("venue_id"), game.get("home_id"))
    try:
        utc = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=ZoneInfo("UTC"))
    except (TypeError, ValueError):
        return date, None
    if not zone:
        return date, None
    return date, utc.astimezone(ZoneInfo(zone)).hour


def _local_starts(games):
    """[(local date, local hour)] aligned with `games`, computed once per schedule pull."""
    return [_local_start(game) for game in games]


def _schedule_spot(games, starts, index, team_id, per_date):
    """Which schedule-spot bucket `games[index]` falls in.

    Doubleheader contexts are checked first: they are more specific than the rest spots
    below and would otherwise be swallowed by "Day after night" or "Normal".
    """
    date, hour = starts[index]
    if date is None:
        return None
    prev_date, prev_hour = starts[index - 1] if index > 0 else (None, None)

    def opponent(game):
        return game.get("away_id") if game.get("home_id") == team_id else game.get("home_id")

    gap = (date - prev_date).days if prev_date else None
    is_day = hour is not None and hour < DAY_GAME_BEFORE_HOUR
    prev_night = prev_hour is not None and prev_hour >= DAY_GAME_BEFORE_HOUR
    nxt = games[index + 1] if index + 1 < len(games) else None
    series_ends = nxt is not None and opponent(nxt) != opponent(games[index])

    if prev_date is not None and prev_date == date:
        return "DH game 2"
    if prev_date is not None and gap == 1 and per_date.get(prev_date, 0) >= 2:
        return "Day after DH"
    if gap is not None and gap >= 2:
        return "Extra rest (2+d)"
    if is_day and series_ends:
        return "Get-away day"
    if gap == 1 and prev_night and is_day:
        return "Day after night"
    return "Normal"


def _league_regular_schedule_games(season, end_date):
    """Return compact final-game rows from one league-wide schedule response."""
    season = int(season)
    end_date = min(str(end_date), f"{season}-10-10")
    key = (season, end_date)
    if key in _TRIP_SCHEDULE_MEMORY:
        return _TRIP_SCHEDULE_MEMORY[key]
    try:
        data = cached_json_request(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={
                "sportId": 1,
                "gameType": "R",
                "startDate": f"{season}-03-15",
                "endDate": end_date,
            },
            namespace="statsapi",
        )
    except Exception:
        _TRIP_SCHEDULE_MEMORY[key] = []
        return []

    rows, seen = [], set()
    final_states = {"Final", "Game Over", "Completed Early"}
    for date_block in data.get("dates", []):
        for game in date_block.get("games", []):
            game_pk = game.get("gamePk")
            if (
                game_pk in seen
                or game.get("gameType") != "R"
                or game.get("status", {}).get("detailedState") not in final_states
            ):
                continue
            home = game.get("teams", {}).get("home", {})
            away = game.get("teams", {}).get("away", {})
            if home.get("score") is None or away.get("score") is None:
                continue
            seen.add(game_pk)
            rows.append({
                "game_pk": game_pk,
                "date": game.get("officialDate") or date_block.get("date"),
                "game_datetime": game.get("gameDate", ""),
                "venue_id": game.get("venue", {}).get("id"),
                "venue_name": game.get("venue", {}).get("name", ""),
                "home_id": home.get("team", {}).get("id"),
                "away_id": away.get("team", {}).get("id"),
                "home_score": int(home["score"]),
                "away_score": int(away["score"]),
            })
    _TRIP_SCHEDULE_MEMORY[key] = rows
    return rows


def _team_trip_observations(team_id, season, end_date):
    """Label each team game by consecutive home/road position."""
    games = []
    for game in _league_regular_schedule_games(season, end_date):
        if team_id not in {game["home_id"], game["away_id"]}:
            continue
        is_home = game["home_id"] == team_id
        games.append({
            "date": game["date"],
            "game_datetime": game["game_datetime"],
            "game_pk": game["game_pk"],
            "is_home": is_home,
            "rf": game["home_score"] if is_home else game["away_score"],
            "ra": game["away_score"] if is_home else game["home_score"],
        })
    games.sort(key=lambda row: (row["date"], row["game_datetime"], row["game_pk"]))

    previous_site, previous_date = None, None
    position, segment = 0, 0
    observations = []
    for game in games:
        try:
            game_date = datetime.strptime(game["date"], "%Y-%m-%d").date()
        except (TypeError, ValueError):
            continue
        gap = (game_date - previous_date).days if previous_date else None
        if game["is_home"] != previous_site or (gap is not None and gap >= 4):
            position = 1
            segment += 1
        else:
            position += 1
        observations.append({
            **game,
            "season": int(season),
            "position": position,
            "segment": segment,
            "win": game["rf"] > game["ra"],
            "run_diff": game["rf"] - game["ra"],
        })
        previous_site, previous_date = game["is_home"], game_date
    return observations


def _trip_position_bucket(position):
    if position == 1:
        return "G1"
    if position <= 3:
        return "G2-3"
    if position <= 6:
        return "G4-6"
    return "G7+"


def trip_spot_bucket(current_label):
    """Map a current trip label ("Road G3 | trip 1-1, +2 RD") onto the matching
    Trip Spot row ("Road G2-3") so tonight's spot can be shaded in the season table."""
    match = re.match(r"\s*(Home|Road)\s+G(\d+)", str(current_label or ""))
    if not match:
        return ""
    return f"{match.group(1)} {_trip_position_bucket(int(match.group(2)))}"


def build_trip_homestand_form(team_id, season, game_date, current_is_home, lookback_seasons=3):
    """Three-year descriptive form by road-trip/homestand position.

    This intentionally does not create a predictive penalty. MLB-wide analysis shows
    trip position is a weak, unstable signal; team results and sample sizes are exposed
    so a scout can use the context without treating it as causal.
    """
    columns = ["Trip Spot", "G", "W%", "ΔW pp", "RD/G"]
    try:
        current_date = datetime.strptime(game_date, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return "", pd.DataFrame(columns=columns)
    context_end = current_date - timedelta(days=1)
    all_observations, current_observations = [], []
    first_season = max(2021, int(season) - int(lookback_seasons) + 1)
    for year in range(first_season, int(season) + 1):
        year_end = context_end.strftime("%Y-%m-%d") if year == int(season) else f"{year}-10-10"
        observations = _team_trip_observations(team_id, year, year_end)
        all_observations.extend(observations)
        if year == int(season):
            current_observations = observations

    current_position, current_segment = 1, []
    if current_observations:
        last = current_observations[-1]
        try:
            last_date = datetime.strptime(last["date"], "%Y-%m-%d").date()
            gap = (current_date - last_date).days
        except (TypeError, ValueError):
            gap = None
        if last["is_home"] == bool(current_is_home) and gap is not None and gap < 4:
            current_position = last["position"] + 1
            current_segment = [
                obs for obs in current_observations if obs["segment"] == last["segment"]
            ]

    site = "Home" if current_is_home else "Road"
    unit = "stand" if current_is_home else "trip"
    if not current_segment:
        current_label = f"{site} G1 | {unit} opener"
    else:
        wins = sum(int(obs["win"]) for obs in current_segment)
        losses = len(current_segment) - wins
        run_diff = sum(obs["run_diff"] for obs in current_segment)
        current_label = f"{site} G{current_position} | {unit} {wins}-{losses}, {run_diff:+d} RD"

    rows = []
    for is_home, site_label in [(False, "Road"), (True, "Home")]:
        site_games = [obs for obs in all_observations if obs["is_home"] == is_home]
        site_win_pct = (
            sum(int(obs["win"]) for obs in site_games) / len(site_games) if site_games else None
        )
        for bucket in ["G1", "G2-3", "G4-6", "G7+"]:
            games = [
                obs for obs in site_games
                if _trip_position_bucket(obs["position"]) == bucket
            ]
            if not games:
                continue
            win_pct = sum(int(obs["win"]) for obs in games) / len(games)
            rows.append({
                "Trip Spot": f"{site_label} {bucket}",
                "G": len(games),
                "W%": round(win_pct, 3),
                "ΔW pp": round((win_pct - site_win_pct) * 100, 1) if site_win_pct is not None else "",
                "RD/G": round(sum(obs["run_diff"] for obs in games) / len(games), 2),
            })
    return current_label, pd.DataFrame(rows, columns=columns)


def _move_clock_toward(body_offset, local_offset, adaptation_days):
    delta = local_offset - body_offset
    if not delta or adaptation_days <= 0:
        return body_offset
    adjustment = min(abs(delta), float(adaptation_days))
    return body_offset + adjustment * (1 if delta > 0 else -1)


def build_time_zone_context(
    team_id,
    season,
    game_date,
    current_venue_id=None,
    current_game_datetime=None,
    current_home_team_id=None,
):
    """Estimate residual circadian displacement from the team's actual venue path.

    The state follows the published MLB convention of roughly one hour of adaptation
    per available day. Consecutive-date travel is treated as zero full acclimation
    days. The residual displacement feeds the trusted eastward-travel projection
    baseline applied after the core game model.
    """
    empty = {
        "Summary": "Unavailable", "Compact": "Unavailable",
        "Route": "?", "Shift": "", "Residual": "",
        "Accl": "", "Level": "Unavailable", "Direction": "None",
        "Residual Hours": None, "Previous Venue": "", "Current Venue": "",
    }
    try:
        target_date = datetime.strptime(str(game_date), "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return empty

    history = []
    for game in _league_regular_schedule_games(season, (target_date - timedelta(days=1)).strftime("%Y-%m-%d")):
        if team_id not in {game["home_id"], game["away_id"]}:
            continue
        zone = _venue_zone(game.get("venue_id"), game.get("home_id"))
        offset = _zone_offset_hours(zone, game.get("game_datetime") or game.get("date"))
        if zone and offset is not None:
            history.append({**game, "zone": zone, "offset": offset})
    history.sort(key=lambda row: (row["date"], row["game_datetime"], row["game_pk"]))

    current_zone = _venue_zone(current_venue_id, current_home_team_id)
    current_when = current_game_datetime or str(game_date)
    current_offset = _zone_offset_hours(current_zone, current_when)
    if not history or not current_zone or current_offset is None:
        return empty

    body_offset = None
    previous = None
    for game in history:
        if body_offset is None:
            body_offset = game["offset"]
        else:
            try:
                gap = (
                    datetime.strptime(game["date"], "%Y-%m-%d").date()
                    - datetime.strptime(previous["date"], "%Y-%m-%d").date()
                ).days
            except (TypeError, ValueError):
                gap = 1
            changed_zone = game["zone"] != previous["zone"]
            adaptation_days = max(gap - 1, 0) if changed_zone else max(gap, 0)
            body_offset = _move_clock_toward(
                body_offset, game["offset"], adaptation_days
            )
        previous = game

    try:
        gap = (
            target_date - datetime.strptime(previous["date"], "%Y-%m-%d").date()
        ).days
    except (TypeError, ValueError):
        gap = 1
    changed_zone = current_zone != previous["zone"]
    adaptation_days = max(gap - 1, 0) if changed_zone else max(gap, 0)
    adjusted_body = _move_clock_toward(body_offset, current_offset, adaptation_days)
    residual = round(current_offset - adjusted_body, 1)
    direct_shift = round(current_offset - previous["offset"], 1)
    magnitude = abs(residual)
    direction = "East" if residual > 0 else "West" if residual < 0 else "None"
    is_long_haul = (
        abs(direct_shift) >= 4
        or current_zone.startswith(("Asia/", "Europe/"))
        or previous["zone"].startswith(("Asia/", "Europe/"))
    )
    if is_long_haul and magnitude >= 2:
        level = "Long haul"
    elif magnitude >= 2 and direction == "East":
        level = "Elevated"
    elif magnitude >= 2:
        level = "Monitor"
    elif magnitude >= 0.75:
        level = "Watch"
    else:
        level = "None"

    previous_abbr = _zone_abbr(previous["zone"], previous["date"])
    current_abbr = _zone_abbr(current_zone, game_date)
    route = f"{previous_abbr}→{current_abbr}"

    def signed_label(value):
        if abs(value) < 0.05:
            return "0h"
        return f"{'E' if value > 0 else 'W'}{abs(value):g}h"

    residual_label = signed_label(residual)
    summary = (
        "No residual circadian load"
        if level == "None"
        else f"{residual_label} | {adaptation_days}d accl | {level}"
    )
    compact = "None" if level == "None" else f"{residual_label} | {adaptation_days}d | {level}"
    return {
        "Summary": summary,
        "Compact": compact,
        "Route": route,
        "Shift": signed_label(direct_shift),
        "Residual": residual_label,
        "Accl": f"{adaptation_days}d",
        "Level": level,
        "Direction": direction,
        "Residual Hours": residual,
        "Previous Venue": previous.get("venue_name", ""),
        "Current Venue": (
            next(
                (
                    game.get("venue_name", "")
                    for game in reversed(history)
                    if game.get("venue_id") == current_venue_id
                ),
                "",
            )
            or str(current_venue_id or "")
        ),
    }


def _cached_team_schedule(start_date, end_date, team_id):
    """Disk-cached `statsapi.schedule` for a team over a date range.

    The library call is uncached network on every invocation; a full-season pull measured
    ~20s, and two of them ran per report. The range carries the end date, so a new day
    still busts the cache and picks up new results -- this only stops the same range being
    re-fetched on every re-run of the same night.
    """
    return cached_dataframe_call("statsapi_schedule", statsapi.schedule,
                                 start_date=start_date, end_date=end_date, team=team_id)


def build_schedule_context_performance(team_id, season, end_date):
    """Team offense (R/G) and pitching (RA/G) split by schedule spot: extra rest, get-away day,
    day-after-night, and normal. Frames hitter/pitcher output through the rest notes."""
    columns = ["Context", "G", "R/G", "RA/G", "W-L"]
    try:
        raw = _regular_season(
            _cached_team_schedule(_season_start_date(season), end_date, team_id))
    except Exception:
        return pd.DataFrame(columns=columns)
    games = sorted([g for g in raw if g.get("status") == "Final" and g.get("game_datetime")], key=lambda g: g["game_datetime"])

    def _dt(g):
        try:
            return datetime.strptime(g["game_datetime"], "%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            return None

    def _opp_id(g):
        return g.get("away_id") if g.get("home_id") == team_id else g.get("home_id")

    def _runs(g):
        if g.get("home_id") == team_id:
            return g.get("home_score"), g.get("away_score")
        return g.get("away_score"), g.get("home_score")

    # Games played per calendar date: two on a date is a doubleheader, whether it was
    # scheduled that way or a split admission. Both the nightcap itself and the following
    # day carry a real workload cost -- a short bullpen and a lineup that played 18 innings.
    from collections import Counter
    starts = _local_starts(games)
    per_date = Counter(date for date, _ in starts if date)

    buckets = {}
    for i, g in enumerate(games):
        rs, ra = _runs(g)
        if rs is None or ra is None:
            continue
        ctx = _schedule_spot(games, starts, i, team_id, per_date)
        if ctx is None:
            continue
        buckets.setdefault(ctx, []).append((rs, ra, rs > ra))

    rows = []
    for ctx in SCHEDULE_SPOTS:
        items = buckets.get(ctx, [])
        if not items:
            continue
        n = len(items)
        w = sum(1 for it in items if it[2])
        rows.append({
            "Context": ctx, "G": n,
            "R/G": round(sum(it[0] for it in items) / n, 2),
            "RA/G": round(sum(it[1] for it in items) / n, 2),
            "W-L": f"{w}-{n - w}",
        })
    return pd.DataFrame(rows, columns=columns)


def build_sweep_record(team_id, season, end_date):
    """Season series-sweep tally: series swept, series swept by the opponent, and how
    many of each were four-game "brooms".

    A series is a run of consecutive games against the same opponent at the same site.
    Only completed series of 3+ games count -- a two-game set isn't a sweep, and an
    in-progress series can still be split.
    """
    empty = {"Swept": 0, "Swept 4G": 0, "Been Swept": 0, "Been Swept 4G": 0,
             "Swept Line": "0", "Been Swept Line": "0"}
    # Look a few days past the cutoff so we can tell a finished series from one still in
    # progress -- without it, a 2-0 lead in a three-game set would book as a sweep.
    try:
        look_ahead = (datetime.strptime(str(end_date), "%Y-%m-%d") + timedelta(days=5)).strftime("%Y-%m-%d")
        raw = _regular_season(_cached_team_schedule(
            _season_start_date(season), look_ahead, team_id))
    except Exception:
        return empty
    played = sorted([g for g in raw if g.get("status") == "Final" and g.get("game_datetime")
                     and g.get("game_date", "") <= str(end_date)],
                    key=lambda g: g["game_datetime"])
    upcoming = sorted([g for g in raw if g.get("game_datetime") and g.get("game_date", "") > str(end_date)],
                      key=lambda g: g["game_datetime"])
    if not played:
        return empty

    def _series_key(g):
        is_home = g.get("home_id") == team_id
        return (g.get("away_id") if is_home else g.get("home_id")), is_home

    series, current, key = [], [], None
    for g in played:
        is_home = g.get("home_id") == team_id
        rs = g.get("home_score") if is_home else g.get("away_score")
        ra = g.get("away_score") if is_home else g.get("home_score")
        if rs is None or ra is None:
            continue
        this_key = _series_key(g)
        if this_key != key:
            if current:
                series.append(current)
            current, key = [], this_key
        current.append(rs > ra)
    if current:
        series.append(current)

    # Drop the trailing run only if the same matchup resumes after the cutoff.
    if series and upcoming and _series_key(upcoming[0]) == key:
        series = series[:-1]

    swept = [s for s in series if len(s) >= 3 and all(s)]
    been_swept = [s for s in series if len(s) >= 3 and not any(s)]
    swept_4 = sum(1 for s in swept if len(s) >= 4)
    been_swept_4 = sum(1 for s in been_swept if len(s) >= 4)

    def line(count, four_game):
        return f"{count}" + (f" ({four_game} 4G broom)" if four_game else "")

    return {
        "Swept": len(swept), "Swept 4G": swept_4,
        "Been Swept": len(been_swept), "Been Swept 4G": been_swept_4,
        "Swept Line": line(len(swept), swept_4),
        "Been Swept Line": line(len(been_swept), been_swept_4),
    }


def build_tonight_schedule_context(team_id, season, game_date, dh_game=None):
    """Which row of build_schedule_context_performance actually applies to tonight.

    Uses the same bucketing rules as the season table, so the highlighted row is
    genuinely the team's situation today rather than an approximation."""
    try:
        raw = _regular_season(statsapi.schedule(
            start_date=(datetime.strptime(game_date, "%Y-%m-%d") - timedelta(days=14)).strftime("%Y-%m-%d"),
            end_date=(datetime.strptime(game_date, "%Y-%m-%d") + timedelta(days=7)).strftime("%Y-%m-%d"),
            team=team_id,
        ))
    except Exception:
        return ""
    # Completed games plus tonight's, matching what the season table counts. A postponement
    # sitting in the window is not "the previous game" -- nobody played it -- and treating
    # it as one moved the rest gap by a day and produced a spot the table never assigns.
    games = sorted(
        [g for g in raw
         if g.get("game_datetime")
         and (g.get("status") == "Final" or str(g.get("game_date") or "")[:10] == game_date)],
        key=lambda g: g["game_datetime"])

    # On a doubleheader date there are two entries; dh_game picks which one is "tonight".
    todays = [i for i, g in enumerate(games) if g.get("game_date") == game_date]
    if not todays:
        return ""
    idx = todays[min(int(dh_game or 1), len(todays)) - 1]

    from collections import Counter
    starts = _local_starts(games)
    per_date = Counter(date for date, _ in starts if date)
    # The same function the season table buckets with, so the highlighted row is the row
    # this note names rather than a second opinion about the same night.
    return _schedule_spot(games, starts, idx, team_id, per_date) or ""


def _first_value(df, column, default=""):
    """First value in a column of a single-row summary frame, or `default`."""
    if df is None or df.empty or column not in df.columns:
        return default
    value = df[column].iloc[0]
    return default if value is None or (isinstance(value, float) and pd.isna(value)) else value


def sp_rest_bucket(sp_rest_text):
    """Map a "6d"-style rest string onto build_starter_rest_splits' bucket labels."""
    days = _safe_number(str(sp_rest_text or "").rstrip("dD"), None)
    if days is None:
        return ""
    return "<=4d" if days <= 4 else "5d" if days == 5 else "6d+"


def build_starting_pitcher_info(pitcher_id, team_abbr, report_date, last_n=5, season=None):
    player_info = statsapi.get("people", {"personIds": pitcher_id})
    player = player_info["people"][0]
    name = player["fullName"]
    throws = player.get("pitchHand", {}).get("description", "Unknown")

    report_season = int(season or report_date[:4])
    stat_end_date = _pregame_end_date(report_date)
    stat = None
    season_used = str(report_season)

    for stat_season in [str(report_season), str(report_season - 1)]:
        candidate_end_date = stat_end_date if int(stat_season) == report_season else None
        stat = get_player_stat(pitcher_id, "pitching", int(stat_season), end_date=candidate_end_date)
        if stat:
            season_used = stat_season
            break

    if stat is None:
        print(f"⚠️ No stats found for {name}")
        return {
            "Name": name,
            "Throws": throws,
            "ERA": None,
            "FIP": None,
            "K%": None,
            "BB%": None,
            "WHIP": None,
            "IP": None,
            "GS": None,
            "HR": None,
            "HR/9": None,
            "Last Starts": []
        }

    k = float(stat.get("strikeOuts", 0))
    ibb = float(stat.get("intentionalWalks", 0))
    bb = float(stat.get("baseOnBalls", 0)) - ibb
    hr = float(stat.get("homeRuns", 0))
    ip = _ip_to_float(stat.get("inningsPitched", 0.0))
    bf = float(stat.get("battersFaced", 1))
    k_pct = k / bf
    bb_pct = bb / bf
    era = float(stat.get("era", 0))
    whip = float(stat.get("whip", 0))
    gs = int(stat.get("gamesStarted", 0))

    league_era, fip_constant = get_league_pitching_stats(int(season_used))
    fip = calculate_fip(hr, bb, k, ip, fip_constant=fip_constant)

    return {
        "Name": name,
        "Throws": throws,
        "ERA": round(era, 2),
        "FIP": fip,
        "K%": round(k_pct, 3),
        "BB%": round(bb_pct, 3),
        "WHIP": round(whip, 2),
        "IP": round(ip, 1),
        "GS": gs,
        "HR": int(hr),
        "HR/9": round(hr * 9 / ip, 2) if ip > 0 else None,
        "Last Starts": get_last_n_starts_direct(
            pitcher_id,
            season_used,
            last_n=last_n,
            end_date=stat_end_date if int(season_used) == report_season else None,
        )
    }



def get_today_starting_pitcher(pitcher_id, team_abbr="CIN", date=None, last_n=5, season=None):
    from datetime import datetime
    if not date:
        today = datetime.today().strftime("%Y-%m-%d")
    else:
        today = date

    if pitcher_id is not None:
        return build_starting_pitcher_info(pitcher_id, team_abbr, today, last_n=last_n, season=season)

    team_id = get_team_id(team_abbr)
    schedule = statsapi.schedule(team=team_id, start_date=today, end_date=today)

    if not schedule:
        print(f"⚠️ No scheduled game found for {team_abbr} on {today}.")
        return None

    game = schedule[0]
    game_id = game['game_id']
    box = statsapi.boxscore_data(game_id)
    home_team = game['home_id'] == team_id
    side = 'home' if home_team else 'away'
    pitchers = box.get(f'{side}Pitchers', [])

    if not pitchers:
        print(f"⚠️ No pitchers found for {side} side.")
        return None

    if pitcher_id is None:
        print(f"⚠️ No probable pitcher found for {team_abbr} today.")
        return None

    # Get pitcher info
    player_info = statsapi.get("people", {"personIds": pitcher_id})
    player = player_info["people"][0]
    name = player["fullName"]
    throws = player.get("pitchHand", {}).get("description", "Unknown")

    # Get pitcher season stats
    for season in ["2025", "2024"]:
        url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats?stats=season&season={season}&group=pitching"
        res = cached_json_request(url, namespace="statsapi")
        stats = res.get("stats", [])
        if stats and stats[0].get("splits"):
            stat = stats[0]["splits"][0]["stat"]
            break
    else:
        print(f"⚠️ No stats found for {name}")
        return {
            "Name": name,
            "Throws": throws,
            "ERA": None,
            "FIP": None,
            "K%": None,
            "BB%": None,
            "WHIP": None,
            "IP": None,
            "GS": None,
            "HR": None,
            "HR/9": None,
            "Last Starts": []
        }

    # Pitcher stats
    k = float(stat.get("strikeOuts", 0))
    ibb = float(stat.get("intentionalWalks", 0))
    bb = float(stat.get("baseOnBalls", 0)) - ibb
    hr = float(stat.get("homeRuns", 0))
    ip = _ip_to_float(stat.get("inningsPitched", 0.0))
    bf = float(stat.get("battersFaced", 1))
    k_pct = k / bf
    bb_pct = bb / bf
    era = float(stat.get("era", 0))
    whip = float(stat.get("whip", 0))
    gs = int(stat.get("gamesStarted", 0))

    league_era, fip_constant = get_league_pitching_stats(season)
    fip = calculate_fip(hr, bb, k, ip, fip_constant=fip_constant)

    return {
        "Name": name,
        "Throws": throws,
        "ERA": round(era, 2),
        "FIP": fip,
        "K%": round(k_pct, 3),
        "BB%": round(bb_pct, 3),
        "WHIP": round(whip, 2),
        "IP": round(ip, 1),
        "GS": gs,
        "HR": int(hr),
        "HR/9": round(hr * 9 / ip, 2) if ip > 0 else None,
        "Last Starts": get_last_n_starts_direct(pitcher_id, season)
    }





# Function to generate the pitch heatmap
def plot_pitch_heatmap(df, title, filename, starter):
    # Ensure the static directory exists
    os.makedirs(os.path.join('plots'), exist_ok=True)


    # Drop rows missing location or pitch type
    df = df.dropna(subset=['plate_x', 'plate_z', 'pitch_type', 'player_name'])

    # Standardize name casing
    df['player_name'] = df['player_name'].str.title()
    starter = starter.title()

    # Filter for just the starter
    df = df[df['player_name'] == starter]

    if df.empty or df['plate_x'].nunique() < 2 or df['plate_z'].nunique() < 2:
        print(f"Not enough data to generate heatmap for: {starter} - {title}")
        return

    print(f"Average pitch locations for {starter}")
    avg_locations = df.groupby('pitch_type')[['plate_x', 'plate_z']].mean().reset_index()
    print(avg_locations)

    # Start the plot
    plt.figure(figsize=(5, 6))

    # KDE heatmap
    try:
        sns.kdeplot(
            data=df, x='plate_x', y='plate_z',
            fill=True, cmap="Reds", bw_adjust=0.5, levels=100, thresh=0.05
        )
    except Exception as e:
        print(f"Error while plotting KDE: {e}")
        return

    # Strike zone
    plt.axhline(1.5, color='black', linestyle='--', linewidth=1)
    plt.axhline(3.5, color='black', linestyle='--', linewidth=1)
    plt.axvline(-0.83, color='black', linestyle='--', linewidth=1)
    plt.axvline(0.83, color='black', linestyle='--', linewidth=1)

    # Label each pitch type
    for _, row in avg_locations.iterrows():
        pitch = row['pitch_type']
        x = row['plate_x']
        z = row['plate_z']
        print(f"Labeling {pitch} at ({round(x, 2)}, {round(z, 2)})")
        plt.scatter(x, z, s=100, edgecolors='black', facecolor='white', zorder=5)
        plt.text(x + 0.05, z + 0.05, pitch, fontsize=9, color='black', weight='bold', zorder=6)

    # Labels and limits
    plt.title(title)
    plt.xlabel("Horizontal Location (plate_x)")
    plt.ylabel("Vertical Location (plate_z)")
    plt.xlim(df['plate_x'].min() - 1, df['plate_x'].max() + 1)
    plt.ylim(df['plate_z'].min() - 1, df['plate_z'].max() + 1)
    plt.tight_layout()

    # Save the heatmap
    full_path = os.path.join(os.getcwd(), 'plots', filename)
    try:
        plt.savefig(full_path)
        print(f"Heatmap saved as {full_path}")
    except Exception as e:
        print(f"Failed to save heatmap: {e}")
    plt.close()

# Function to generate heatmaps for different types of pitches
def generate_heatmaps_for_pitches(df, starter):
    name = playerid_reverse_lookup([starter])
    starter_name = (name['name_last'] + ', ' + name['name_first'])[0].title()
    # Print the columns to ensure correct column name for description or event_type
    print(f"Columns in the DataFrame: {df.columns}")
    # Use the correct column name after inspecting the data
    if 'description' in df.columns:
        plot_pitch_heatmap(df, f'All Pitches Heatmap {starter_name}', f'{starter} heatmap_all.png', starter_name)

        # Filter data for called strikes
        called_strikes = df[df['description'] == 'called_strike']
        called_strikes = called_strikes.dropna(subset=['plate_x', 'plate_z'])

        plot_pitch_heatmap(called_strikes, f'Called Strikes Heatmap {starter_name}', f'{starter} heatmap_called_strikes.png', starter_name)

        # Filter data for whiffs
        whiffs = df[df['description'] == 'swinging_strike']
        whiffs = whiffs.dropna(subset=['plate_x', 'plate_z'])

        plot_pitch_heatmap(whiffs, f'Whiffs Heatmap {starter_name}', f'{starter} heatmap_whiffs.png', starter_name)
    else:
        print("Error: 'description' column not found in the data.")


# Function to fetch data, and fallback to 2024 if not available for 2025
def fetch_pitch_data_for_starter(starter, year=2025):
    try:
        # Attempt to fetch 2025 data
        print(f"Fetching data for {starter} ({year})")
        # Replace with actual data fetch logic for 2025
        df_2025 = cached_dataframe_call("statcast_pitcher", statcast_pitcher, start_dt=f'{year}-03-01', end_dt=f'{year}-10-31', player_id=starter)
        if df_2025.empty:
            print(f"No data for {starter} in {year}, falling back to 2024.")
            # If 2025 data is not available, fall back to 2024
            df_2024 = cached_dataframe_call("statcast_pitcher", statcast_pitcher, 2024, player_id=starter)
            return df_2024
        else:
            return df_2025
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()





# %%

def plot_dynamic_pitch_mix(df, starter):
    # Ensure the directory exists
    os.makedirs(os.path.join('plots'), exist_ok=True)

    # Filter data for just the starter
    df = df[df['player_name'] == starter]

    if df.empty or 'pitch_type' not in df.columns:
        print(f"Not enough data to generate pitch mix for: {starter}")
        return

    # Get the count of each pitch type
    pitch_counts = df['pitch_type'].value_counts()

    # Start the plot for pitch mix
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plotting the pitch mix
    ax.bar(pitch_counts.index, pitch_counts.values, color='skyblue')
    ax.set_title(f'Pitch Mix for {starter}', fontsize=14, weight='bold')
    ax.set_xlabel('Pitch Type')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)
    
    # Save the plot
    filename = f"{starter} pitch_mix.png"
    full_path = os.path.join(os.getcwd(), 'plots', filename)
    
    try:
        plt.tight_layout()
        plt.savefig(full_path)
        print(f"Pitch mix saved as {full_path}")
    except Exception as e:
        print(f"Failed to save pitch mix: {e}")
    plt.close()

    return filename  # Returning filename for later use


def generate_hot_cold_hitters_chart(hotcold_df, starter, output_file="hot_cold_hitters_chart.png"):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot OPS with color representing hot or cold status
    ax.barh(hotcold_df['Name'], hotcold_df['OPS'], color=hotcold_df['Status'].apply(lambda x: 'red' if 'HOT' in x else 'blue'))

    ax.set_xlabel('OPS')
    ax.set_title(f'Hot / Cold Hitters - {starter}', fontsize=14, weight='bold')
    ax.set_xlim(0, 1)  # Assuming OPS ranges from 0 to 1

    # Add status labels
    for i, row in hotcold_df.iterrows():
        ax.text(row['OPS'] + 0.02, i, row['Status'], va='center', fontsize=10, color='white')

    plt.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)

    print(f"Hot / Cold Hitters chart saved as {output_file}")

def generate_baserunning_leaderboard(team: str, year: int = 2024):
    two_letter = {'TB': 'TBR', 'KC':'KCR', 'SD': 'SDP', 'WSH':'WSN',
                  'CWS':'CHW', 'SF': 'SFG', 'OAK':'ATH'}
    if team in two_letter:
        team = two_letter[team]

    # Load season-long batting stats
    try:
        stats = cached_dataframe_call("batting_stats", batting_stats, year)
    except Exception as e:
        print(f"⚠️ Baserunning leaderboard unavailable for {team}: {e}")
        return pd.DataFrame(columns=['Name', 'Team', 'BsR', 'Spd', 'SB', 'CS', 'SB_Att', 'SB%'])

    # Filter by teams
    filtered_stats = stats[(stats['Team'] == team)]

    # Select relevant columns
    baserunning_df = filtered_stats[[
        'Name', 'Team', 'Spd', 'BsR', 'SB', 'CS' 
    ]].copy()

    # Compute SB attempts and SB success rate
    baserunning_df['SB_Att'] = baserunning_df['SB'] + baserunning_df['CS']
    baserunning_df['SB%'] = baserunning_df['SB'] / baserunning_df['SB_Att'].replace(0, 1)

    # Sort by stolen base attempts or speed
    baserunning_df = baserunning_df.sort_values(by='BsR', ascending=False).round(2)

    return baserunning_df




def generate_defensive_leaderboard_fangraphs(team: str, year: int = 2024):
    two_letter = {'TB': 'TBR', 'KC':'KCR', 'SD': 'SDP', 'WSH':'WSN',
                  'CWS':'CHW', 'SF':'SFG', 'OAK':'ATH'}
    if team in two_letter:
        team = two_letter[team]
    
    # Load all fielding stats from FanGraphs for the year
    try:
        df = cached_dataframe_call("fielding_stats", fielding_stats, year)
    except Exception as e:
        print(f"⚠️ Defensive leaderboard unavailable for {team}: {e}")
        return pd.DataFrame(columns=['Name', 'Team', 'Inn', 'DRS', 'UZR', 'ARM', 'Def', 'Pos', 'Fld'])

    # Show all columns (for debug)
    print("Available columns:", df.columns.tolist())

    # Standardize team names
    df['Team'] = df['Team'].str.upper()

    # Filter for selected teams
    filtered = df[df['Team'].isin([team.upper()])]

    # Safe column list (only select those that exist)
    possible_cols = ['Name', 'Team', 'Inn', 'DRS', 'UZR', 'ARM', 'Def', 'Pos', 'Fld']
    used_cols = [col for col in possible_cols if col in filtered.columns]

    # Reduce to available columns only
    filtered = filtered[used_cols]

    # Sort by a known defensive stat if available
    sort_col = 'DRS' if 'DRS' in filtered.columns else 'Def' if 'Def' in filtered.columns else None
    if sort_col:
        filtered = filtered.sort_values(by=sort_col, ascending=False)

    return filtered


def generate_bullpen_summary(team_abbr, year=2025, as_of_date=None):
    rows = []
    stat_end_date = _pregame_end_date(as_of_date) if as_of_date else None
    _, fip_constant = get_league_pitching_stats(year)
    fip_constant = fip_constant if fip_constant is not None else 3.1

    for player in get_team_roster(team_abbr, as_of_date=as_of_date):
        if player.get('position', {}).get('abbreviation') != 'P':
            continue

        person = player.get('person', {})
        player_id = person.get('id')
        if not player_id:
            continue

        stat = get_player_stat(player_id, "pitching", year, end_date=stat_end_date)
        if not stat:
            continue

        games = _stat_int(stat, "gamesPlayed")
        starts = _stat_int(stat, "gamesStarted")
        if games <= starts:
            continue
        if starts > 0 and (starts / games) >= 0.25:
            continue

        ip = _ip_to_float(stat.get("inningsPitched", 0))
        bf = _stat_float(stat, "battersFaced")
        k = _stat_float(stat, "strikeOuts")
        bb = _stat_float(stat, "baseOnBalls") - _stat_float(stat, "intentionalWalks")
        hr = _stat_float(stat, "homeRuns")
        saves = _stat_int(stat, "saves")
        save_opps = _stat_int(stat, "saveOpportunities")
        holds = _stat_int(stat, "holds")
        blown_saves = _stat_int(stat, "blownSaves")
        games_finished = _stat_int(stat, "gamesFinished")
        sv_pct = round(saves / save_opps, 3) if save_opps else 0.0

        rows.append({
            'Name': person.get('fullName', 'Unknown'),
            'Role': 'Swing' if starts else 'Relief',
            'G': games,
            'GS': starts,
            'IP': round(ip, 1),
            'ERA': _stat_float(stat, "era"),
            'FIP': calculate_fip(hr, bb, k, ip, fip_constant=fip_constant),
            'K%': _pct(k, bf),
            'BB%': _pct(bb, bf),
            'WHIP': _stat_float(stat, "whip"),
            'SV': saves,
            'SVO': save_opps,
            'HLD': holds,
            'BS': blown_saves,
            'GF': games_finished,
            'SV%': sv_pct,
        })

    columns = ['Name', 'Role', 'G', 'GS', 'IP', 'ERA', 'FIP', 'K%', 'BB%', 'WHIP', 'SV', 'SVO', 'HLD', 'BS', 'GF', 'SV%']
    return pd.DataFrame(rows, columns=columns).round(2).sort_values(by='IP', ascending=False) if rows else pd.DataFrame(columns=columns)


def get_season_stats_for_players(player_names, year=None, player_ids=None, as_of_date=None):
    year = year or datetime.today().year
    rows = []
    player_ids = player_ids or {}
    stat_end_date = _pregame_end_date(as_of_date) if as_of_date else None

    for name in player_names:
        player_id = player_ids.get(name) if isinstance(player_ids, dict) else None
        if not player_id:
            player_id = lookup_player_id(name)

        stat = get_player_stat(player_id, "hitting", year, end_date=stat_end_date) if player_id else {}

        ab = _stat_float(stat, "atBats")
        hits = _stat_float(stat, "hits")
        doubles = _stat_float(stat, "doubles")
        triples = _stat_float(stat, "triples")
        hr = _stat_float(stat, "homeRuns")
        singles = max(0, hits - doubles - triples - hr)
        total_bases = singles + (2 * doubles) + (3 * triples) + (4 * hr)
        slg = _rate(total_bases, ab)
        avg = _stat_float(stat, "avg", _rate(hits, ab))
        obp = _stat_float(stat, "obp", 0.0)
        ops = _stat_float(stat, "ops", obp + slg)

        rows.append({
            'Name': name,
            'AB': int(ab),
            'PA': _stat_int(stat, "plateAppearances"),
            'AVG': round(avg, 3),
            'OBP': round(obp, 3),
            'SLG': round(slg, 3),
            'OPS': round(ops, 3),
            'ISO': round(slg - avg, 3),
            'HR': int(hr),
            'RBI': _stat_int(stat, "rbi"),
            'SB': _stat_int(stat, "stolenBases")
        })

    return pd.DataFrame(rows)


def generate_baserunning_leaderboard(team: str, year: int = 2024, as_of_date=None):
    rows = []
    stat_end_date = _pregame_end_date(as_of_date) if as_of_date else None
    for player in get_team_roster(team, as_of_date=as_of_date):
        if player.get('position', {}).get('abbreviation') == 'P':
            continue

        person = player.get('person', {})
        player_id = person.get('id')
        stat = get_player_stat(player_id, "hitting", year, end_date=stat_end_date) if player_id else {}

        sb = _stat_int(stat, "stolenBases")
        cs = _stat_int(stat, "caughtStealing")
        attempts = sb + cs
        rows.append({
            'Name': person.get('fullName', 'Unknown'),
            'Team': team,
            'SB': sb,
            'CS': cs,
            'SB_Att': attempts,
            'SB%': round(sb / attempts, 3) if attempts else 0.0,
            'Runs': _stat_int(stat, "runs"),
            'Triples': _stat_int(stat, "triples")
        })

    columns = ['Name', 'Team', 'SB', 'CS', 'SB_Att', 'SB%', 'Runs', 'Triples']
    return pd.DataFrame(rows, columns=columns).sort_values(by=['SB_Att', 'SB'], ascending=False) if rows else pd.DataFrame(columns=columns)


def generate_defensive_leaderboard_fangraphs(team: str, year: int = 2024, as_of_date=None):
    rows = []
    stat_end_date = _pregame_end_date(as_of_date) if as_of_date else None
    for player in get_team_roster(team, as_of_date=as_of_date):
        position = player.get('position', {}).get('abbreviation')
        if position in {'P', 'DH'}:
            continue

        person = player.get('person', {})
        player_id = person.get('id')
        stat = get_player_stat(player_id, "fielding", year, end_date=stat_end_date) if player_id else {}
        innings = _stat_float(stat, "innings")
        chances = _stat_int(stat, "chances")
        if innings <= 0 and chances <= 0:
            continue

        rows.append({
            'Name': person.get('fullName', 'Unknown'),
            'Team': team,
            'Pos': position,
            'Inn': innings,
            'Chances': chances,
            'Errors': _stat_int(stat, "errors"),
            'Assists': _stat_int(stat, "assists"),
            'PutOuts': _stat_int(stat, "putOuts"),
            'Fielding%': _stat_float(stat, "fielding", 0.0),
            'CS': _stat_int(stat, "caughtStealing"),
            'SB Allowed': _stat_int(stat, "stolenBases"),
            'CS%': _stat_float(stat, "caughtStealingPercentage", 0.0),
            'PB': _stat_int(stat, "passedBall")
        })

    columns = ['Name', 'Team', 'Pos', 'Inn', 'Chances', 'Errors', 'Assists', 'PutOuts', 'Fielding%', 'CS', 'SB Allowed', 'CS%', 'PB']
    return pd.DataFrame(rows, columns=columns).sort_values(by='Inn', ascending=False) if rows else pd.DataFrame(columns=columns)


TERMINAL_PA_EVENTS = {
    'strikeout', 'strikeout_double_play', 'walk', 'intent_walk', 'hit_by_pitch',
    'single', 'double', 'triple', 'home_run', 'field_out', 'force_out',
    'grounded_into_double_play', 'double_play', 'sac_fly', 'sac_bunt',
    'fielders_choice', 'fielders_choice_out', 'other_out'
}

HIT_EVENTS = {'single', 'double', 'triple', 'home_run'}

# Statcast pitch-level `description` classifications for swing-decision metrics.
SWING_DESCRIPTIONS = {
    'swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip',
    'hit_into_play', 'missed_bunt', 'foul_bunt', 'bunt_foul_tip',
}
WHIFF_DESCRIPTIONS = {'swinging_strike', 'swinging_strike_blocked', 'missed_bunt'}
OUT_OF_ZONE_CODES = {11, 12, 13, 14}  # statcast `zone`: 1-9 in-zone, 11-14 out


def _numeric_col(df, col, default=0.0):
    if df is None:
        return pd.Series(dtype=float)
    if df.empty or col not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


PARK_CONTEXT = {
    "American Family Field": {"HR": 1.08, "Runs": 1.02, "profile": "roof/controlled; friendly to pulled air contact"},
    "Angel Stadium": {"HR": 0.96, "Runs": 0.98, "profile": "fair run environment; marine air can mute carry"},
    "Busch Stadium": {"HR": 0.92, "Runs": 0.96, "profile": "suppresses carry; outfield defense matters"},
    "Chase Field": {"HR": 1.05, "Runs": 1.04, "profile": "roof/controlled; gaps can reward hard contact"},
    "Citi Field": {"HR": 0.94, "Runs": 0.97, "profile": "bigger outfield; range and gap defense matter"},
    "Citizens Bank Park": {"HR": 1.12, "Runs": 1.03, "profile": "home-run friendly, especially pull-side air contact"},
    "Comerica Park": {"HR": 0.92, "Runs": 0.99, "profile": "large outfield; doubles/triples and OF range matter"},
    "Coors Field": {"HR": 1.18, "Runs": 1.28, "profile": "extreme run environment; outfield range and bullpen depth matter"},
    "Daikin Park": {"HR": 1.04, "Runs": 0.99, "profile": "short LF; pull power can play up"},
    "Dodger Stadium": {"HR": 1.02, "Runs": 0.99, "profile": "neutral-to-power friendly in warm air"},
    "Fenway Park": {"HR": 0.97, "Runs": 1.08, "profile": "doubles/gap park; LF/RF wall reads are important"},
    "George M. Steinbrenner Field": {"HR": 1.05, "Runs": 1.02, "profile": "warm-air run environment; weather can swing carry"},
    "Globe Life Field": {"HR": 1.00, "Runs": 0.98, "profile": "roof/controlled; plays close to neutral"},
    "Great American Ball Park": {"HR": 1.22, "Runs": 1.06, "profile": "very HR-friendly; fly-ball mistakes are punished"},
    "Guaranteed Rate Field": {"HR": 1.10, "Runs": 1.02, "profile": "power-friendly; wind can change HR risk quickly"},
    "Kauffman Stadium": {"HR": 0.88, "Runs": 1.01, "profile": "large gaps; OF range and relay defense matter"},
    "loanDepot park": {"HR": 0.92, "Runs": 0.95, "profile": "suppresses power; run prevention often comes from clean defense"},
    "Minute Maid Park": {"HR": 1.04, "Runs": 0.99, "profile": "short LF; pull power can play up"},
    "Nationals Park": {"HR": 1.00, "Runs": 1.00, "profile": "mostly neutral; weather drives daily run context"},
    "Oakland Coliseum": {"HR": 0.90, "Runs": 0.94, "profile": "foul ground and air suppress offense"},
    "Oracle Park": {"HR": 0.82, "Runs": 0.94, "profile": "suppresses HR; OF range and gap coverage matter"},
    "Oriole Park at Camden Yards": {"HR": 0.96, "Runs": 0.99, "profile": "deeper LF lowers easy pull HRs"},
    "Petco Park": {"HR": 0.91, "Runs": 0.95, "profile": "suppresses power; late movement/contact management matter"},
    "PNC Park": {"HR": 0.88, "Runs": 0.97, "profile": "large left-center; OF range matters"},
    "Progressive Field": {"HR": 0.97, "Runs": 0.98, "profile": "neutral-slight pitcher lean; weather matters"},
    "Rate Field": {"HR": 1.10, "Runs": 1.02, "profile": "power-friendly; wind can change HR risk quickly"},
    "Rogers Centre": {"HR": 1.04, "Runs": 1.02, "profile": "roof/controlled; rewards hard air contact"},
    "Sahlen Field": {"HR": 1.02, "Runs": 1.01, "profile": "minor-league venue context; treat as approximate"},
    "Sutter Health Park": {"HR": 1.13, "Runs": 1.08, "profile": "small warm-air park; carry, HR risk, and fast outfield reads matter"},
    "T-Mobile Park": {"HR": 0.90, "Runs": 0.93, "profile": "suppresses carry; favors run prevention and OF defense"},
    "Target Field": {"HR": 0.96, "Runs": 0.98, "profile": "weather-sensitive; cool air can suppress carry"},
    "Truist Park": {"HR": 1.04, "Runs": 1.01, "profile": "slightly power-friendly; pull-side contact can play"},
    "Wrigley Field": {"HR": 1.04, "Runs": 1.03, "profile": "wind-driven; weather can dominate park baseline"},
    "Yankee Stadium": {"HR": 1.14, "Runs": 1.01, "profile": "short RF; LHB pull power is a major stress point"}
}


PARK_ALIASES = {
    "Los Angeles Dodgers": "Dodger Stadium",
    "LA Dodgers": "Dodger Stadium",
    "Dodgers": "Dodger Stadium",
    "Houston Astros": "Daikin Park",
    "Minute Maid": "Daikin Park",
    "Raley Field": "Sutter Health Park",
    "Sacramento": "Sutter Health Park",
    "Chicago White Sox": "Rate Field",
    "Guaranteed Rate": "Rate Field",
}


def _normalize_park_name(value):
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _park_context_for_venue(venue_name):
    if venue_name in PARK_CONTEXT:
        return PARK_CONTEXT[venue_name], venue_name
    alias = PARK_ALIASES.get(venue_name)
    if alias and alias in PARK_CONTEXT:
        return PARK_CONTEXT[alias], alias

    normalized = _normalize_park_name(venue_name)
    for park_name, park in PARK_CONTEXT.items():
        park_norm = _normalize_park_name(park_name)
        if normalized == park_norm or (park_norm and park_norm in normalized):
            return park, park_name
    for alias_name, park_name in PARK_ALIASES.items():
        alias_norm = _normalize_park_name(alias_name)
        if alias_norm and alias_norm in normalized and park_name in PARK_CONTEXT:
            return PARK_CONTEXT[park_name], park_name
    return {"HR": 1.00, "Runs": 1.00, "profile": "park factor unavailable; treat as neutral"}, venue_name


CALIBRATION_PATH = os.path.join(PROJECT_ROOT, ".cache", "model_calibration.json")
CALIBRATION_DATA_PATH = os.path.join(PROJECT_ROOT, ".cache", "model_calibration_games.csv")
CALIBRATION_FEATURE_VERSION = 3
CALIBRATION_TEAM_PRIOR_GAMES = 20.0
CALIBRATION_RECENT_PRIOR_GAMES = 5.0
CALIBRATION_RUNS_PRIOR = 4.4
_STATCAST_DETAIL_MEMORY_CACHE = {}
_STATCAST_SPLITS_MEMORY_CACHE = {}


def report_output_dir(date, reports_root="scouting_reports", dated_output=True):
    return os.path.join(reports_root, date) if dated_output else reports_root


# --- assembled-report-data cache: lets layout be re-rendered without re-pulling data ---
REPORT_DATA_CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache", "report_data")


def _dh_suffix(dh_game):
    """Filename suffix so game 2 of a doubleheader does not overwrite game 1."""
    return f"_g{int(dh_game)}" if dh_game and int(dh_game) > 1 else ""


def _report_data_cache_path(date, away_team, home_team, dh_game=None):
    """Cache path for one game.

    Game 2 of a doubleheader gets a `_g2` suffix. Without it both games of a pairing map
    to the same file and the second silently overwrites the first.
    """
    stem = f"{date}_{away_team}_{home_team}"
    if dh_game and int(dh_game) > 1:
        stem += f"_g{int(dh_game)}"
    return os.path.join(REPORT_DATA_CACHE_DIR, f"{stem}.pkl")


def save_report_data_cache(date, away_team, home_team, report_args, advanced_context,
                           dh_game=None):
    os.makedirs(REPORT_DATA_CACHE_DIR, exist_ok=True)
    path = _report_data_cache_path(date, away_team, home_team, dh_game)
    try:
        summary = (advanced_context or {}).get("projection_summary", {})
        calibration = {
            "fingerprint": summary.get("Calibration Fingerprint", "unknown"),
            "feature_version": summary.get("Calibration Feature Version"),
            "created_at": summary.get("Calibration Created At"),
        }
        with open(path, "wb") as f:
            pickle.dump({
                "report_args": report_args,
                "advanced_context": advanced_context,
                "model_calibration": calibration,
            }, f)
    except Exception as e:
        print(f"⚠️ Could not cache report data: {e}")
    return path


def load_report_data_cache(date, away_team, home_team, dh_game=None):
    path = _report_data_cache_path(date, away_team, home_team, dh_game)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _lineup_signature(frame):
    """A batting order as sorted (spot, name) pairs -- what has to match to call it unchanged.

    None when the frame cannot describe a full order. That counts as *changed*, never as
    same: an unreadable cache is exactly the situation a refresh exists for.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    if not {"Name", "Spot"} <= set(frame.columns):
        return None
    spots = pd.to_numeric(frame["Spot"], errors="coerce")
    names = frame["Name"].astype(str).map(remove_accents).str.strip()
    pairs = [(int(spot), name) for spot, name in zip(spots, names)
             if pd.notna(spot) and name]
    return tuple(sorted(pairs)) if len(pairs) == 9 else None


def _posted_lineup_signature(team_abbr, game_date):
    """The same signature read off MLB's card, or None when no full card is posted."""
    try:
        posted = get_today_lineup(team_abbr, game_date=game_date)
    except Exception as error:
        print(f"⚠️ Could not read {team_abbr}'s lineup card ({error}); refreshing anyway.")
        return None
    pairs = [(int(player["#"]) // 100, remove_accents(str(player["Name"])).strip())
             for player in (posted or []) if player.get("#") and player.get("Name")]
    return tuple(sorted(pairs)) if len(pairs) == 9 else None


def _cached_lineup_confidence(frame):
    if not isinstance(frame, pd.DataFrame) or "Lineup Confidence" not in frame.columns:
        return ""
    values = frame["Lineup Confidence"].dropna().astype(str)
    return values.iloc[0] if len(values) else ""


def cached_lineups_are_current(payload, date, away_team, home_team, dh_game=None):
    """Is the cached lineup already what MLB is showing? -> (bool, one-line reason).

    Reading the two cards costs a schedule and a boxscore call per team. What it guards is
    the whole of `refresh_cached_lineups`: hot/cold game logs and season stats for eighteen
    hitters, the statcast splits and detail loads, and the composite plus offense-index
    rebuild. So the probe pays for itself even on the runs where it saves nothing, and the
    common case it is here for -- autosnap taking `confirmed` and then `final` an hour later
    off an unchanged card -- it skips outright.

    Every uncertain answer resolves to "not current". A refresh that was not needed costs
    time; a skip that was needed prints yesterday's bats.
    """
    if dh_game:
        # `get_today_lineup` reads schedule[0], so on a doubleheader it may answer about the
        # other half. Comparing the wrong game's card is how a stale lineup gets kept.
        return False, "doubleheader, so the card cannot be matched to the right game"

    args = payload.get("report_args") or ()
    if len(args) < 8:
        return False, "cached payload predates the current report_args layout"

    for team, frame in ((home_team, args[0]), (away_team, args[7])):
        cached = _lineup_signature(frame)
        confidence = _cached_lineup_confidence(frame)
        posted = _posted_lineup_signature(team, date)

        if posted is not None:
            if cached != posted:
                return False, f"{team}'s posted card differs from the cached lineup"
            if not confidence.startswith("Confirmed"):
                # Same nine names, but the cache recorded them as a guess. The refresh adds
                # no players -- it upgrades the label the report prints, which is the
                # difference between "this is the lineup" and "this is our guess".
                return False, f"{team}'s lineup is posted but cached as '{confidence}'"
            continue

        # Nothing posted for this team. A projection is deterministic for a given date, so
        # re-running it against the same history returns the same nine and there is nothing
        # to learn. A Fallback is a projection that failed, and is worth retrying.
        if confidence.startswith("Projected"):
            continue
        return False, (f"no card posted for {team} and the cached lineup is "
                       f"'{confidence or 'unreadable'}'")

    return True, "both cards already match the cached lineups"


def refresh_cached_lineups(payload, date, away_team, home_team, dh_game=None, force=False):
    """Re-pull both lineups into a cached payload and rebuild what depends on them.

    A plain cache re-render replays whatever lineup was cached, which is usually a
    "Fallback" guess made before lineups posted -- so it will not show confirmed lineups
    no matter how many times it is run. This re-pulls them.

    Refreshed: lineup, platoon splits, hot/cold, offense index, hitter composite.
    Carried over: arsenal / similar-pitcher / BvP columns, which are per-player and would
    need the similarity engine to rebuild. Players newly in the lineup simply have no
    values there, which is visible rather than wrong.

    Returns the payload untouched when both cards already match what is cached -- see
    `cached_lineups_are_current`. Pass force=True to rebuild regardless.
    """
    if not force:
        current, reason = cached_lineups_are_current(payload, date, away_team, home_team,
                                                     dh_game=dh_game)
        if current:
            print(f"🔄 Lineups unchanged — kept the cached report ({reason}).")
            return payload
        print(f"🔄 Re-pulling lineups: {reason}.")

    args = list(payload["report_args"])
    context = payload["advanced_context"]
    season = int(str(date)[:4])
    arsenal_end = _pregame_end_date(date) or date

    home_pitcher, away_pitcher = args[5], args[12]
    splits_source = load_statcast_splits(season=season, end_date=arsenal_end)
    detail = load_statcast_detail(season=season, end_date=arsenal_end)

    home_hotcold, home_lineup, home_splits = generate_team_hitter_report(
        home_team, away_pitcher, splits_source, game_date=date)
    away_hotcold, away_lineup, away_splits = generate_team_hitter_report(
        away_team, home_pitcher, splits_source, game_date=date)

    args[0], args[3], args[25] = home_lineup, home_hotcold, home_splits
    args[7], args[10], args[26] = away_lineup, away_hotcold, away_splits
    payload["report_args"] = tuple(args)

    league_context = generate_league_batter_context(detail)
    composite = pd.concat([
        build_hitter_composite_table(away_team, away_lineup, away_splits,
                                     context.get("away_similarity"), context.get("away_batter_arsenal"),
                                     context.get("away_bvp")),
        build_hitter_composite_table(home_team, home_lineup, home_splits,
                                     context.get("home_similarity"), context.get("home_batter_arsenal"),
                                     context.get("home_bvp")),
    ], ignore_index=True)
    context["hitter_composite"] = attach_offense_index(
        composite, [away_lineup, home_lineup], league_context,
        [away_hotcold, home_hotcold],
    )
    context["hitter_summary"] = summarize_hitter_composite(context["hitter_composite"])

    confidence = sorted(set(
        list(home_lineup.get("Lineup Confidence", pd.Series(dtype=str)).dropna().unique())
        + list(away_lineup.get("Lineup Confidence", pd.Series(dtype=str)).dropna().unique())
    ))
    print(f"🔄 Lineups re-pulled — confidence: {', '.join(confidence) or 'unknown'}")
    return payload


# How authoritative each starter source is. Only used to tell "firmed up" from "changed":
# a report built off the rotation model and later confirmed by DK is the same report with a
# better label, while a *different* arm is a different report entirely.
STARTER_SOURCE_RANK = {"unknown": 0, "rotation": 1, "bulk": 1, "salary": 2,
                       "announced": 3, "override": 4}


def recheck_cached_starters(payload, date, away_team, home_team, dh_game=None):
    """Re-run the starter ladder over a payload that was built on a provisional pick.

    Returns `(action, detail)`:

        "none"      nothing to do -- both starters were announced when the report was
                    built, or nothing has moved since
        "relabel"   the same arms, but a source firmed up (rotation model -> DK -> posted
                    probable). The provenance is updated in the payload and saved; no table
                    changes, only the line the report prints about where the name came from.
        "rebuild"   a different arm is starting. Nothing here can be patched: the arsenal,
                    the comparable-pitcher set, BvP, the platoon splits and the entire
                    hitter composite were all built against the old pitcher, so the only
                    honest answer is to generate the game again.

    A payload with no `starter_provenance` predates the ladder, which means its starters
    could only have come from an announced probable -- so it is left alone.
    """
    context = payload.get("advanced_context") or {}
    provenance = context.get("starter_provenance") or {}
    provisional = {side: pick for side, pick in provenance.items()
                   if isinstance(pick, dict) and pick.get("provisional")}
    if not provisional:
        return "none", ""

    args = payload.get("report_args") or ()
    if len(args) < 13:
        return "none", "cached payload predates the current report_args layout"
    cached_ids = {"home": args[5], "away": args[12]}

    try:
        selected_game = choose_game(date, away_team=away_team, home_team=home_team,
                                    dh_game=dh_game)
    except Exception as error:
        print(f"⚠️ Could not re-check starters for {away_team}@{home_team}: {error}")
        return "none", ""
    if not selected_game:
        return "none", ""

    season = int(str(date)[:4])
    context_end = _pregame_end_date(date) or date
    changed, firmed = [], []
    for side, team in (("home", home_team), ("away", away_team)):
        if side not in provisional:
            continue
        fresh = resolve_starter(selected_game, side, team, date,
                                season=season, context_end=context_end)
        if not fresh.get("id"):
            continue
        was = provenance.get(side) or {}
        if int(fresh["id"]) != int(cached_ids[side] or 0):
            changed.append(f"{team}: {was.get('name', 'unknown')} -> {fresh['name']} "
                           f"({STARTER_SOURCE_LABEL.get(fresh['source'], fresh['source'])})")
        elif STARTER_SOURCE_RANK.get(fresh["source"], 0) > STARTER_SOURCE_RANK.get(was.get("source"), 0):
            firmed.append(f"{team}: {fresh['name']} confirmed by "
                          f"{STARTER_SOURCE_LABEL.get(fresh['source'], fresh['source'])}")
        provenance[side] = fresh

    if changed:
        return "rebuild", "; ".join(changed)
    if firmed:
        context["starter_provenance"] = provenance
        save_report_data_cache(date, away_team, home_team, payload["report_args"],
                               context, dh_game=dh_game)
        return "relabel", "; ".join(firmed)
    return "none", ""


def render_report_from_cache(date, away_team, home_team, output_dir="scouting_reports",
                             dated_output=True, reports_root="scouting_reports", formats=("pdf",),
                             dfs_highlight=True, fast=False, refresh_lineups=False,
                             dh_game=None, dfs_slate=None, force_lineup_refresh=False):
    """Rebuild report(s) from cached data. Fast layout iteration.

    fast:            skip every API refresh below and render purely from cache (~1s).
                     Boxscores, sweeps, schedule spot and umpire stay as cached.
    refresh_lineups: re-pull lineups first, so a confirmed lineup actually appears. The
                     cards are checked before anything is rebuilt, so a re-run against an
                     unchanged lineup keeps the cached report instead of rebuilding it.
    force_lineup_refresh: rebuild even when the cards match.

    formats: any of "pdf", "xlsx". Returns list of output paths.
    """
    payload = load_report_data_cache(date, away_team, home_team, dh_game)
    if payload is None:
        print(f"⚠️ No cached report data for {date} {away_team}@{home_team}. Run a full generation first.")
        return []
    if refresh_lineups:
        action, detail = recheck_cached_starters(payload, date, away_team, home_team,
                                                 dh_game=dh_game)
        if action == "rebuild":
            # Re-rendering from cache here would print the old starter's arsenal under the
            # new starter's name, so the game is generated again instead. The first format
            # rewrites the payload; the rest then come off it for free.
            print(f"🔄 Starter changed ({detail}) — regenerating {away_team}@{home_team} "
                  f"from source rather than re-rendering the cache.")
            formats = tuple(formats)
            rebuilt = list(generate_scouting_report_for_game(
                date, away_team=away_team, home_team=home_team,
                output_format=formats[0], reports_root=reports_root,
                dated_output=dated_output, dfs_highlight=dfs_highlight,
                dh_game=dh_game, dfs_slate=dfs_slate) or [])
            for extra in formats[1:]:
                rebuilt.extend(render_report_from_cache(
                    date, away_team, home_team, reports_root=reports_root,
                    dated_output=dated_output, formats=(extra,),
                    dfs_highlight=dfs_highlight, fast=True, dh_game=dh_game,
                    dfs_slate=dfs_slate) or [])
            return [path for path in rebuilt if path]
        if action == "relabel":
            print(f"🔄 Starter provenance updated ({detail}).")
        payload = refresh_cached_lineups(payload, date, away_team, home_team,
                                         dh_game=dh_game, force=force_lineup_refresh)
    payload, calibration_changed, calibration_provenance = refresh_payload_model_calibration(payload)
    if calibration_changed:
        print(
            "Rebuilt cached scorecard for model calibration "
            f"{calibration_provenance['fingerprint']}."
        )
    cached_context = payload["advanced_context"]
    args = payload["report_args"]
    season = int(str(date)[:4])
    context_end = _pregame_end_date(date) or date

    # Local-only rebuilds: no network, so they run even in fast mode.
    cached_context["home_bullpen_form"] = summarize_bullpen_form(home_team, args[1])
    cached_context["away_bullpen_form"] = summarize_bullpen_form(away_team, args[8])

    # Everything below costs ~14s of API calls against <1s of actual rendering, which is
    # the entire reason a re-render feels slow. Skipping it changes only the boxscore,
    # sweep, schedule-spot and umpire blocks -- none of which move pregame.
    if not fast:
        cached_context = refresh_environment_umpire(cached_context, date, away_team, home_team)
        for side, team in (("away", away_team), ("home", home_team)):
            team_id = get_team_id(team)
            cached_context[f"{side}_recent_boxscores"] = build_recent_boxscores(team_id, team, date)
            # args[8]/args[1] are the away/home bullpen frames, same as the form tables above.
            cached_context[f"{side}_bullpen_l5"] = build_bullpen_l5(
                team_id, team, date, bullpen_df=args[8] if side == "away" else args[1])
            # Tonight's comparison is against the OPPOSING probable: args[2] is the home
            # starter (what the away lineup faces) and args[9] the away starter.
            conditions = _tonight_conditions(
                cached_context.get("environment"),
                args[2] if side == "away" else args[9],
                args[7] if side == "away" else args[0],
                is_home=(side == "home"),
                rest_schedule=cached_context.get(f"{side}_rest_schedule"))
            cached_context[f"{side}_relevant_games"] = build_relevant_games(
                team_id, team, date, conditions, season=season, context_end=context_end)
            cached_context[f"{side}_comparable_games"] = build_comparable_games(
                team_id, team, date, conditions, season=season, context_end=context_end)
            # Statcast is memory-cached per process and disk-cached across runs, so this is
            # a slice rather than a pull on any machine that has already built the night.
            try:
                cached_context[f"{side}_bullpen_batted"] = build_bullpen_batted(
                    args[8] if side == "away" else args[1],
                    load_statcast_detail(season=season, end_date=context_end),
                    team_abbr=team, as_of_date=context_end)
            except Exception as e:
                print(f"⚠️ Bullpen batted-ball profile unavailable for {team}: {e}")
            cached_context[f"{side}_sweeps"] = build_sweep_record(team_id, season, context_end)
            cached_context[f"{side}_context_tonight"] = build_tonight_schedule_context(team_id, season, date, dh_game)

    for side in ("away", "home"):
        cached_context[f"{side}_sp_rest_tonight"] = sp_rest_bucket(
            _first_value(cached_context.get(f"{side}_rest_schedule"), "SP Rest")
        )

    # Pitcher-type splits need the similarity engine (~30s/side), which is too slow to
    # redo on every layout tweak. Rebuild only when the cached table predates the current
    # comp-selection rules -- "Confidence" is the marker column. The refreshed context is
    # written back below, so this is a one-time upgrade per cached game. Bump the marker
    # if the type logic changes again.
    needs_type_rebuild = not fast and any(
        not isinstance(cached_context.get(f"{side}_type_results"), pd.DataFrame)
        or "Confidence" not in cached_context[f"{side}_type_results"].columns
        for side in ("away", "home")
    )
    if needs_type_rebuild:
        try:
            similarity_df = load_statcast_detail(season=season, end_date=context_end)
            home_pitcher, away_pitcher = args[5], args[12]
            for side, team, opposing_sp in (("away", away_team, home_pitcher),
                                            ("home", home_team, away_pitcher)):
                comps = generate_pitcher_similarity_report(opposing_sp, similarity_df, top_n=60)
                results = generate_team_pitcher_type_results(team, comps, similarity_df)
                if not results.empty:
                    cached_context[f"{side}_type_results"] = results
        except Exception as e:
            print(f"⚠️ Could not rebuild pitcher-type splits from cache: {e}")

    # The batter arsenal tables gained "K Edge", which is the only arsenal number the DFS
    # pitcher projection reads. A cached game without it does not fail -- the factor simply
    # never fires -- so it has to be rebuilt rather than tolerated. Same one-time-upgrade
    # pattern as the block above, keyed on the new column.
    def _stale(key, marker=None):
        frame = cached_context.get(key)
        if not isinstance(frame, pd.DataFrame):
            return True
        return marker is not None and marker not in frame.columns

    # Every artifact this block writes needs its own marker. Gating on one of them meant a
    # cache that had already gained "K Edge" skipped the whole block, so the watchlist kept
    # its old columns forever.
    needs_arsenal_rebuild = not fast and (
        any(_stale(f"{side}_batter_arsenal", "K Edge") for side in ("away", "home"))
        or any(_stale(f"{side}_team_defense") for side in ("away", "home"))
        or _stale("pitcher_watchlist", "K Edge")
    )

    # Opener profiles cached before the index was fixed carry innings counted as "distinct
    # innings appeared in" and, in ~3% of games, the wrong follower. `bulk_repeat_rate` is
    # the marker for a profile built by the corrected index.
    needs_opener_rebuild = not fast and any(
        not isinstance(cached_context.get(f"{side}_opener_profile"), dict)
        or "bulk_repeat_rate" not in cached_context[f"{side}_opener_profile"]
        for side in ("away", "home")
    )
    if needs_opener_rebuild:
        try:
            for side, pitcher_id in (("away", args[12]), ("home", args[5])):
                cached_context[f"{side}_opener_profile"] = build_opener_profile(
                    pitcher_id, season, context_end)
        except Exception as e:
            print(f"⚠️ Could not rebuild opener profiles from cache: {e}")

    if needs_arsenal_rebuild:
        try:
            detail_df = load_statcast_detail(season=season, end_date=context_end)

            def _arsenal_for(pitcher_id):
                pitcher_id = _safe_number(pitcher_id, None)
                if pitcher_id is None:
                    return pd.DataFrame()
                return generate_starter_arsenal(int(pitcher_id),
                                                _season_start_date(season), context_end)

            for side, team, lineup_df, opposing_arsenal, opposing_sp in (
                ("away", away_team, args[7], _arsenal_for(args[5]), args[2]),
                ("home", home_team, args[0], _arsenal_for(args[12]), args[9]),
            ):
                rebuilt = generate_batter_arsenal_matchups(
                    team, lineup_df, opposing_arsenal,
                    _pitcher_throw_code(opposing_sp), detail_df)
                if not rebuilt.empty:
                    cached_context[f"{side}_batter_arsenal"] = rebuilt
                # Same statcast frame is already in memory, so team defence rides along.
                cached_context[f"{side}_team_defense"] = build_team_defense(team, detail_df)
            refresh_baserunning_heat_anchors(season)
            # The watchlist is a cached artifact, so it keeps its old columns until it is
            # rebuilt -- and it is the table the K edge is most worth reading in. Pure
            # computation over frames already in hand, so it costs nothing to redo.
            cached_context["pitcher_watchlist"] = build_pitcher_watchlist(
                home_team, away_team, args[2], args[9],
                cached_context.get("away_vs_home_arsenal"),
                cached_context.get("home_vs_away_arsenal"),
                cached_context.get("environment"),
                away_batter_arsenal=cached_context.get("away_batter_arsenal"),
                home_batter_arsenal=cached_context.get("home_batter_arsenal"),
                away_lineup_df=args[7], home_lineup_df=args[0])
        except Exception as e:
            print(f"⚠️ Could not rebuild batter arsenal fits from cache: {e}")

    cached_context["scorecard"], cached_context["projection_summary"] = (
        apply_trusted_time_zone_baseline(
            cached_context.get("scorecard"),
            cached_context.get("projection_summary"),
            away_team,
            home_team,
            cached_context.get("away_time_zone", {}),
            cached_context.get("home_time_zone", {}),
        )
    )
    payload["advanced_context"] = cached_context
    save_report_data_cache(
        date, away_team, home_team, payload["report_args"], cached_context, dh_game=dh_game
    )
    activate_dfs_highlight(date, enabled=dfs_highlight, slate=dfs_slate,
                           games=[f"{away_team}@{home_team}"])
    out_dir = report_output_dir(date, reports_root, dated_output)
    outputs = []
    if "pdf" in formats:
        outputs.append(generate_comparison_report_with_fpdf(
            *payload["report_args"], output_dir=out_dir,
            advanced_context=payload["advanced_context"], primary=True, dh_game=dh_game))
    if "xlsx" in formats:
        outputs.append(generate_comparison_report_excel(
            *payload["report_args"], output_dir=out_dir,
            advanced_context=payload["advanced_context"], dh_game=dh_game))
    return outputs


def load_statcast_detail(season=None, end_date=None, lookback_seasons=1):
    end_date = end_date or datetime.today().strftime("%Y-%m-%d")
    season = season or int(end_date[:4])
    cache_key = (int(season), str(end_date), int(lookback_seasons))
    if cache_key in _STATCAST_DETAIL_MEMORY_CACHE:
        return _STATCAST_DETAIL_MEMORY_CACHE[cache_key].copy()
    start_season = max(2015, int(season) - int(lookback_seasons) + 1)
    frames = []
    for year in range(start_season, int(season) + 1):
        start_date = _season_start_date(year)
        year_end = end_date if year == int(season) else f"{year}-10-31"
        if year_end < start_date:
            continue
        df = load_statcast_range(start_date, year_end)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    out = df[df.get('events', pd.Series(index=df.index)).notna()].copy()
    _STATCAST_DETAIL_MEMORY_CACHE[cache_key] = out
    return out.copy()


_STATCAST_PITCHES_MEMORY_CACHE = {}


def load_statcast_pitches(season=None, end_date=None, lookback_seasons=1):
    """Full pitch-level statcast (unfiltered) for swing-decision / run-value work. Reuses the cached statcast pulls."""
    end_date = end_date or datetime.today().strftime("%Y-%m-%d")
    season = season or int(end_date[:4])
    cache_key = (int(season), str(end_date), int(lookback_seasons))
    if cache_key in _STATCAST_PITCHES_MEMORY_CACHE:
        return _STATCAST_PITCHES_MEMORY_CACHE[cache_key]
    start_season = max(2015, int(season) - int(lookback_seasons) + 1)
    frames = []
    for year in range(start_season, int(season) + 1):
        start_date = _season_start_date(year)
        year_end = end_date if year == int(season) else f"{year}-10-31"
        if year_end < start_date:
            continue
        df = load_statcast_range(start_date, year_end)
        if df is not None and not df.empty:
            frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    _STATCAST_PITCHES_MEMORY_CACHE[cache_key] = out
    return out


def _swing_metrics(pitches_df):
    """Swing/whiff/chase/contact counts from pitch-level statcast rows."""
    out = {"pitches": 0, "swings": 0, "whiffs": 0, "ooz": 0, "chases": 0, "contact": 0}
    if pitches_df is None or pitches_df.empty or "description" not in pitches_df.columns:
        return out
    desc = pitches_df["description"].astype(str)
    swings = desc.isin(SWING_DESCRIPTIONS)
    whiffs = desc.isin(WHIFF_DESCRIPTIONS)
    zone = pd.to_numeric(pitches_df.get("zone"), errors="coerce")
    ooz = zone.isin(OUT_OF_ZONE_CODES)
    out["pitches"] = int(len(pitches_df))
    out["swings"] = int(swings.sum())
    out["whiffs"] = int(whiffs.sum())
    out["ooz"] = int(ooz.sum())
    out["chases"] = int((swings & ooz).sum())
    out["contact"] = out["swings"] - out["whiffs"]
    return out


@lru_cache(maxsize=8192)
def _person_record(player_id):
    """One cached StatsAPI `people` record per id.

    **Why this exists.** `statsapi.get` goes straight to the network -- it does not touch
    the disk cache -- so the hand and name lookups re-fetched the same players on every
    run: 24.6s and 17.5s of a 588s report, for data (a name, a throwing hand) that cannot
    change mid-season. Routing through `cached_json_request` makes it permanent on disk,
    and sharing one record between the two lookups halves the requests as well.
    """
    try:
        data = cached_json_request("https://statsapi.mlb.com/api/v1/people",
                                   params={"personIds": player_id}, namespace="statsapi")
        people = data.get("people") or []
        return people[0] if people else {}
    except Exception:
        return {}


@lru_cache(maxsize=None)
def _throw_code_for_id(pitcher_id):
    return (_person_record(pitcher_id).get("pitchHand") or {}).get("code")


def _pitcher_throw_code(pitcher_id):
    """L/R for a pitcher id, or None.

    Memoized behind `_throw_code_for_id`: a throwing hand cannot change mid-season, and
    the relevant-games table asks for one opposing starter per game per club, which
    repeats heavily inside a division.

    The unhashable guard is load-bearing rather than defensive. `cmp_pitcher_type_view`
    calls this with a whole starter_info dict, which never resolved to a hand -- the old
    body swallowed it in its bare `except` and returned None, so that call site has always
    fallen through to its "same-handed" default. Caching turned that into a TypeError.
    Returning None keeps the existing output exactly as it was; fixing what that call site
    meant to pass is a separate change with a visible effect on the report text.
    """
    try:
        hash(pitcher_id)
    except TypeError:
        return None
    return _throw_code_for_id(pitcher_id)


def _player_name_from_id(player_id):
    return _person_record(player_id).get("fullName") or str(player_id)


_BATTING_LINE_BLANK = {"PA": 0, "AB": 0, "H": 0, "HR": 0, "BB": 0, "K": 0,
                       "AVG": 0.0, "SLG": 0.0, "OPS_proxy": 0.0}
_NON_AB_EVENTS = ("walk", "intent_walk", "hit_by_pitch", "sac_fly", "sac_bunt")


def _events_to_batting_line(events_df):
    """Batting line from statcast event rows.

    Counts come from a single `value_counts` over the one column that matters. The previous
    version copied the whole 119-column frame (which it never mutated) and then made eight
    separate passes over `events`; at 3,816 calls per report that was 49s of a 588s run.
    Verified against the previous implementation on 800 real batter/game slices.

    One deliberate difference: a frame with no `events` column returns a blank line, where
    the previous version raised KeyError. Blank is already this function's answer for "no
    data", so that path now degrades the way the empty-frame path always has.
    """
    if events_df is None or events_df.empty or "events" not in events_df.columns:
        return dict(_BATTING_LINE_BLANK)

    events = events_df["events"]
    counts = events[events.isin(TERMINAL_PA_EVENTS)].value_counts()
    pa = int(counts.sum())
    if not pa:
        return dict(_BATTING_LINE_BLANK)

    def total(*names):
        return int(sum(int(counts.get(name, 0)) for name in names))

    hits = total(*HIT_EVENTS)
    doubles, triples, homers = total("double"), total("triple"), total("home_run")
    singles = max(0, hits - doubles - triples - homers)
    total_bases = singles + (2 * doubles) + (3 * triples) + (4 * homers)
    walks = total("walk", "intent_walk")
    hbp = total("hit_by_pitch")
    strikeouts = total("strikeout", "strikeout_double_play")
    ab = pa - total(*_NON_AB_EVENTS)

    avg = hits / ab if ab else 0.0
    obp = (hits + walks + hbp) / pa if pa else 0.0
    slg = total_bases / ab if ab else 0.0
    return {
        "PA": pa,
        "AB": int(ab),
        "H": hits,
        "HR": homers,
        "BB": walks,
        "K": strikeouts,
        "AVG": round(avg, 3),
        "SLG": round(slg, 3),
        "OPS_proxy": round(obp + slg, 3),
    }


def _contact_quality(events_df):
    if events_df is None or events_df.empty:
        return {"xwOBA": 0.0, "HardHit%": 0.0, "Avg EV": 0.0, "Whiff%": 0.0}
    # `DataFrame.get` on a missing column returns None, and pd.to_numeric(None) is a scalar
    # nan -- which then fails on `.notna()`. Real statcast frames always carry these, but a
    # caller passing a trimmed frame got an AttributeError instead of zeros.
    def column(name):
        values = events_df[name] if name in events_df.columns else pd.Series(dtype=float,
                                                                             index=events_df.index)
        return pd.to_numeric(values, errors="coerce")

    xwoba = column("estimated_woba_using_speedangle").mean()
    batted = events_df[column("launch_speed").notna()].copy()
    hard_hit = (pd.to_numeric(batted.get("launch_speed"), errors="coerce") >= 95).mean() * 100 if not batted.empty else 0.0
    avg_ev = pd.to_numeric(batted.get("launch_speed"), errors="coerce").mean() if not batted.empty else 0.0
    descriptions = events_df.get("description", pd.Series(dtype=str)).astype(str)
    swings = descriptions.str.contains("swing|foul|hit_into_play", case=False, na=False)
    whiffs = descriptions.str.contains("swinging_strike|missed_bunt", case=False, na=False)
    whiff_rate = (whiffs.sum() / swings.sum()) * 100 if swings.sum() else 0.0
    return {
        "xwOBA": round(float(xwoba), 3) if not pd.isna(xwoba) else 0.0,
        "HardHit%": round(float(hard_hit), 1),
        "Avg EV": round(float(avg_ev), 1) if not pd.isna(avg_ev) else 0.0,
        "Whiff%": round(float(whiff_rate), 1),
    }


def _format_ops_points(delta):
    number = _safe_number(delta)
    if number is None:
        return ""
    points = int(round(number * 1000))
    return f"{points:+d} OPS pts"


def _arsenal_pitch_weights(starter_arsenal_df, top_n=3):
    if starter_arsenal_df is None or starter_arsenal_df.empty:
        return []
    arsenal = starter_arsenal_df.copy()
    arsenal["Usage %"] = _numeric_col(arsenal, "Usage %")
    arsenal = arsenal.sort_values("Usage %", ascending=False).head(top_n)
    total_usage = arsenal["Usage %"].sum()
    if total_usage <= 0:
        return []
    return [
        (str(row.get("Pitch")), float(row.get("Usage %")) / total_usage)
        for _, row in arsenal.iterrows()
        if str(row.get("Pitch", "")).strip()
    ]


def _arsenal_shape_traits(starter_arsenal_df, top_n=3):
    if starter_arsenal_df is None or starter_arsenal_df.empty:
        return []
    arsenal = starter_arsenal_df.copy()
    arsenal["Usage %"] = _numeric_col(arsenal, "Usage %")
    return [
        {
            "Pitch": str(row.get("Pitch")),
            "Velo": _safe_number(row.get("release_speed"), None),
            "Spin": _safe_number(row.get("release_spin_rate"), None),
            "Usage": _safe_number(row.get("Usage %"), 0),
        }
        for _, row in arsenal.sort_values("Usage %", ascending=False).head(top_n).iterrows()
        if str(row.get("Pitch", "")).strip()
    ]


def _format_pitch_weight_basis(starter_hand, pitch_weights, starter_arsenal_df=None):
    if not pitch_weights:
        return f"{starter_hand or '?'} / handedness"
    traits = {item["Pitch"]: item for item in _arsenal_shape_traits(starter_arsenal_df, top_n=len(pitch_weights) or 3)}
    parts = []
    for pitch, weight in pitch_weights:
        trait = traits.get(pitch, {})
        detail = f"{pitch} {weight * 100:.0f}%"
        velo = trait.get("Velo")
        spin = trait.get("Spin")
        if velo is not None and spin is not None:
            detail += f" ({velo:.1f} mph/{spin:.0f} rpm)"
        parts.append(detail)
    return f"{starter_hand or '?'} / " + "/".join(parts)


def _filter_to_arsenal_shape(df, starter_arsenal_df, top_pitches, min_pa=40):
    if df is None or df.empty or not top_pitches or "pitch_type" not in df.columns:
        return df
    pitch_df = df[df["pitch_type"].isin(top_pitches)].copy()
    traits = _arsenal_shape_traits(starter_arsenal_df, top_n=len(top_pitches))
    if pitch_df.empty or not traits:
        return pitch_df

    shape_mask = pd.Series(False, index=pitch_df.index)
    for trait in traits:
        pitch = trait.get("Pitch")
        mask = pitch_df["pitch_type"].astype(str).eq(str(pitch))
        velo = trait.get("Velo")
        spin = trait.get("Spin")
        if velo is not None and "release_speed" in pitch_df.columns:
            speed = pd.to_numeric(pitch_df["release_speed"], errors="coerce")
            mask &= speed.between(velo - 2.5, velo + 2.5)
        if spin is not None and "release_spin_rate" in pitch_df.columns:
            spin_series = pd.to_numeric(pitch_df["release_spin_rate"], errors="coerce")
            mask &= spin_series.between(spin - 300, spin + 300)
        shape_mask |= mask
    shaped = pitch_df[shape_mask].copy()
    return shaped if _events_to_batting_line(shaped)["PA"] >= min_pa else pitch_df


def _weighted_arsenal_metrics(events_df, pitch_weights):
    if events_df is None or events_df.empty or not pitch_weights:
        line = _events_to_batting_line(events_df)
        quality = _contact_quality(events_df)
        pa = line["PA"]
        return {
            "PA": pa,
            "AB": line["AB"],
            "OPS": line["OPS_proxy"],
            "xwOBA": quality["xwOBA"],
            "SLG": line["SLG"],
            "HR": line["HR"],
            "K%": round((line["K"] / pa) * 100, 1) if pa else 0.0,
            "HardHit%": quality["HardHit%"],
            "Whiff%": quality["Whiff%"],
        }

    weighted = {"OPS": 0.0, "xwOBA": 0.0, "SLG": 0.0, "K%": 0.0, "HardHit%": 0.0, "Whiff%": 0.0}
    observed_weight = 0.0
    total_pa = 0
    total_ab = 0
    total_hr = 0
    for pitch, weight in pitch_weights:
        pitch_df = events_df[events_df.get("pitch_type", pd.Series(dtype=str)).astype(str).eq(pitch)].copy()
        line = _events_to_batting_line(pitch_df)
        if line["PA"] <= 0:
            continue
        quality = _contact_quality(pitch_df)
        k_rate = (line["K"] / line["PA"]) * 100 if line["PA"] else 0.0
        weighted["OPS"] += weight * line["OPS_proxy"]
        weighted["xwOBA"] += weight * quality["xwOBA"]
        weighted["SLG"] += weight * line["SLG"]
        weighted["K%"] += weight * k_rate
        weighted["HardHit%"] += weight * quality["HardHit%"]
        weighted["Whiff%"] += weight * quality["Whiff%"]
        observed_weight += weight
        total_pa += line["PA"]
        total_ab += line["AB"]
        total_hr += line["HR"]

    if observed_weight <= 0:
        line = _events_to_batting_line(events_df)
        quality = _contact_quality(events_df)
        pa = line["PA"]
        return {
            "PA": pa,
            "AB": line["AB"],
            "OPS": line["OPS_proxy"],
            "xwOBA": quality["xwOBA"],
            "SLG": line["SLG"],
            "HR": line["HR"],
            "K%": round((line["K"] / pa) * 100, 1) if pa else 0.0,
            "HardHit%": quality["HardHit%"],
            "Whiff%": quality["Whiff%"],
        }

    return {
        "PA": int(total_pa),
        "AB": int(total_ab),
        "OPS": round(weighted["OPS"] / observed_weight, 3),
        "xwOBA": round(weighted["xwOBA"] / observed_weight, 3),
        "SLG": round(weighted["SLG"] / observed_weight, 3),
        "HR": int(total_hr),
        "K%": round(weighted["K%"] / observed_weight, 1),
        "HardHit%": round(weighted["HardHit%"] / observed_weight, 1),
        "Whiff%": round(weighted["Whiff%"] / observed_weight, 1),
    }


def generate_lineup_similarity_report(lineup_df, starter_arsenal_df, starter_hand, statcast_df):
    columns = ["Name", "PA", "AVG", "SLG", "OPS_proxy", "HR", "K%", "Match Basis"]
    if lineup_df is None or lineup_df.empty or statcast_df is None or statcast_df.empty:
        return pd.DataFrame(columns=columns)
    if "ID" not in lineup_df.columns:
        return pd.DataFrame(columns=columns)

    top_pitches = []
    primary_speed = None
    if starter_arsenal_df is not None and not starter_arsenal_df.empty:
        arsenal = starter_arsenal_df.copy()
        if "Usage %" in arsenal.columns:
            arsenal["Usage %"] = pd.to_numeric(arsenal["Usage %"], errors="coerce")
            arsenal = arsenal.sort_values("Usage %", ascending=False)
        top_pitches = arsenal.get("Pitch", pd.Series(dtype=str)).dropna().astype(str).head(3).tolist()
        if "release_speed" in arsenal.columns and not arsenal.empty:
            primary_speed = pd.to_numeric(arsenal.iloc[0:1]["release_speed"], errors="coerce").iloc[0]

    batter_ids = pd.to_numeric(lineup_df.get("ID", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist()
    if not batter_ids:
        return pd.DataFrame(columns=columns)

    df = statcast_df[statcast_df["batter"].isin(batter_ids)].copy()
    if starter_hand and "p_throws" in df.columns:
        df = df[df["p_throws"] == starter_hand]

    pitch_mask = df["pitch_type"].isin(top_pitches) if top_pitches and "pitch_type" in df.columns else pd.Series(False, index=df.index)
    if primary_speed is not None and not pd.isna(primary_speed) and "release_speed" in df.columns:
        speed = pd.to_numeric(df["release_speed"], errors="coerce")
        speed_mask = speed.between(primary_speed - 2.5, primary_speed + 2.5)
    else:
        speed_mask = pd.Series(False, index=df.index)

    filtered = df[pitch_mask | speed_mask].copy()
    rows = []
    id_to_name = dict(zip(pd.to_numeric(lineup_df["ID"], errors="coerce"), lineup_df.get("Name")))
    basis = []
    if top_pitches:
        basis.append("/".join(top_pitches))
    if primary_speed is not None and not pd.isna(primary_speed):
        basis.append(f"{primary_speed:.1f} mph band")
    basis_text = " + ".join(basis) if basis else "handedness only"

    for batter_id in batter_ids:
        batter_events = filtered[filtered["batter"] == batter_id]
        line = _events_to_batting_line(batter_events)
        pa = line["PA"]
        rows.append({
            "Name": id_to_name.get(float(batter_id), _player_name_from_id(batter_id)),
            "PA": pa,
            "AVG": line["AVG"],
            "SLG": line["SLG"],
            "OPS_proxy": line["OPS_proxy"],
            "HR": line["HR"],
            "K%": round((line["K"] / pa) * 100, 1) if pa else 0.0,
            "Match Basis": basis_text,
        })

    return pd.DataFrame(rows, columns=columns).sort_values(["OPS_proxy", "PA"], ascending=False)


# Plate appearances of prior weight pulling a hitter's arsenal K% back toward his own K%
# against that hand. A weighted arsenal sample is ~18 PA at the median, so the raw rate is
# mostly noise; 25 PA of shrinkage was the value that maximized the measured effect on
# starter scoring across 13,158 starts (2024-26). See docs/arsenal_study.md.
ARSENAL_K_SHRINK_PA = 25.0


def _arsenal_k_edge(arsenal_k_rate, arsenal_pa, baseline_k_rate):
    """How much more this hitter strikes out against this shape than he usually does.

    Positive is a pitcher's edge. The level alone is nearly useless -- a high-strikeout
    hitter posts a high K% against every arsenal -- so what is reported is the residual
    against his own rate versus the same hand, shrunk for sample size.
    """
    if baseline_k_rate is None or arsenal_pa is None:
        return None
    observed = float(arsenal_pa or 0.0)
    shrunk = ((float(arsenal_k_rate or 0.0) * observed
               + float(baseline_k_rate) * ARSENAL_K_SHRINK_PA)
              / (observed + ARSENAL_K_SHRINK_PA))
    return round(shrunk - float(baseline_k_rate), 2)


def generate_batter_arsenal_matchups(team_abbr, lineup_df, starter_arsenal_df, starter_hand, statcast_df):
    columns = ["Team", "Name", "PA", "AB", "OPS", "xwOBA", "SLG", "HR", "K%", "K% vs Hand",
               "K Edge", "HardHit%", "Whiff%", "Fit", "Arsenal Score", "Basis"]
    if (
        lineup_df is None or lineup_df.empty or "ID" not in lineup_df.columns
        or statcast_df is None or statcast_df.empty
        or starter_arsenal_df is None or starter_arsenal_df.empty
    ):
        return pd.DataFrame(columns=columns)

    pitch_weights = _arsenal_pitch_weights(starter_arsenal_df, top_n=3)
    top_pitches = [pitch for pitch, _ in pitch_weights]
    if not pitch_weights:
        return pd.DataFrame(columns=columns)

    batter_ids = pd.to_numeric(lineup_df["ID"], errors="coerce").dropna().astype(int).tolist()
    if not batter_ids:
        return pd.DataFrame(columns=columns)

    hand_df = statcast_df[statcast_df["batter"].isin(batter_ids)].copy()
    if starter_hand and "p_throws" in hand_df.columns:
        hand_df = hand_df[hand_df["p_throws"].astype(str).str.strip().eq(starter_hand)]
    # `hand_df` is everything the hitter sees from this hand; `df` narrows it to the
    # starter's pitch shapes. The pair is what makes an edge measurable -- the same
    # source, the same window, the same hand, differing only in the shape filter.
    df = _filter_to_arsenal_shape(hand_df, starter_arsenal_df, top_pitches, min_pa=max(30, len(batter_ids) * 4))

    id_to_name = dict(zip(pd.to_numeric(lineup_df["ID"], errors="coerce"), lineup_df.get("Name")))
    basis = _format_pitch_weight_basis(starter_hand, pitch_weights, starter_arsenal_df)
    rows = []
    for batter_id in batter_ids:
        batter_events = df[df["batter"] == batter_id]
        metrics = _weighted_arsenal_metrics(batter_events, pitch_weights)
        pa = metrics["PA"]
        k_rate = metrics["K%"]
        hand_line = _events_to_batting_line(hand_df[hand_df["batter"] == batter_id])
        baseline_k = (round((hand_line["K"] / hand_line["PA"]) * 100, 1)
                      if hand_line["PA"] >= 25 else None)
        k_edge = _arsenal_k_edge(k_rate, pa, baseline_k)
        ops = metrics["OPS"]
        xwoba = metrics["xwOBA"]
        if pa >= 10 and (ops >= 0.850 or xwoba >= 0.360):
            fit = "Attack"
        elif pa >= 10 and (ops <= 0.620 or xwoba <= 0.280 or k_rate >= 32):
            fit = "Caution"
        elif pa >= 6:
            fit = "Neutral"
        else:
            fit = "Small sample"
        sample_weight = min(pa / 20, 1.0)
        score = (((ops - 0.700) * 100) + ((xwoba - 0.320) * 160) + ((metrics["HardHit%"] - 38) * 0.7) - ((k_rate - 22) * 0.4)) * sample_weight
        rows.append({
            "Team": team_abbr,
            "Name": id_to_name.get(float(batter_id), _player_name_from_id(batter_id)),
            "PA": pa,
            "AB": metrics["AB"],
            "OPS": ops,
            "xwOBA": xwoba,
            "SLG": metrics["SLG"],
            "HR": metrics["HR"],
            "K%": k_rate,
            "K% vs Hand": baseline_k,
            "K Edge": k_edge,
            "HardHit%": metrics["HardHit%"],
            "Whiff%": metrics["Whiff%"],
            "Fit": fit,
            "Arsenal Score": round(score, 1),
            "Basis": basis,
        })

    out = pd.DataFrame(rows, columns=columns)
    if out.empty:
        return out
    return out.sort_values(["Arsenal Score", "OPS", "PA"], ascending=False)


PITCH_PROFILE_TYPES = ["FF", "SI", "FC", "SL", "ST", "CU", "KC", "CH", "FS", "SPL", "SV", "SW"]


_PITCHER_PROFILE_CACHE = {}
_PROFILE_NUMERIC = {"Velo": "release_speed", "Spin": "release_spin_rate",
                    "HB": "pfx_x", "VB": "pfx_z"}
# pfx_x / pfx_z are in feet; the report shows inches.
_PROFILE_SCALE = {"Velo": 1.0, "Spin": 1.0, "HB": 12.0, "VB": 12.0}
_PROFILE_ROUND = {"Velo": 1, "Spin": 0, "HB": 1, "VB": 1}


def _profile_frame_key(statcast_df, min_pitches):
    """Cheap fingerprint for a statcast frame, for memoising the profile table.

    The frame is a season (or three) of pitches and is rebuilt by `.copy()` on each
    `load_statcast_*` call, so identity is useless; length plus the first and last game
    separates the 1-season from the 3-season frame, which is all that varies here.
    """
    if statcast_df is None or statcast_df.empty:
        return ("empty", min_pitches)
    game = statcast_df["game_pk"] if "game_pk" in statcast_df.columns else statcast_df.index
    return (len(statcast_df), int(min_pitches),
            int(pd.to_numeric(game.iat[0], errors="coerce") or 0),
            int(pd.to_numeric(game.iat[-1], errors="coerce") or 0))


def _mode_by_pitcher(df, column):
    """Per-pitcher modal value, matching `groupby(...)[col].mode().iloc[0]`.

    `.mode()` returns its values sorted, so `.iloc[0]` breaks a tie on the lowest value;
    sorting by (count desc, value asc) and taking the first reproduces that exactly.
    """
    if column not in df.columns:
        return pd.Series(dtype=object)
    counts = df.dropna(subset=[column]).groupby(["pitcher", column], observed=True).size()
    if counts.empty:
        return pd.Series(dtype=object)
    counts = counts.reset_index(name="_n").sort_values(
        ["pitcher", "_n", column], ascending=[True, False, True], kind="mergesort")
    return counts.groupby("pitcher", observed=True)[column].first()


def _pitcher_profile_table(statcast_df, min_pitches=150):
    """Per-pitcher arsenal profile: overall shape plus one block per pitch type.

    **Why this is vectorised and memoised.** It used to walk ~800 pitchers in Python and,
    for each, mask the group once per pitch type and call `pd.to_numeric(...).mean()` on
    four columns -- roughly 48,000 pandas operations over a multi-season frame. Measured at
    81s per call, and `generate_pitcher_similarity_report` calls it four times per report
    with identical arguments, so it was 325s of a 588s report: 55% of the whole run. Two
    groupbys and a cache do the same work.
    """
    base_columns = ["PitcherID", "Name", "Throws", "Pitches", "Velo", "Spin", "HB", "VB"]
    for pitch in PITCH_PROFILE_TYPES:
        base_columns.extend([f"Mix_{pitch}", f"Velo_{pitch}", f"Spin_{pitch}", f"HB_{pitch}", f"VB_{pitch}"])
    if statcast_df is None or statcast_df.empty or "pitcher" not in statcast_df.columns:
        return pd.DataFrame(columns=base_columns)

    cache_key = _profile_frame_key(statcast_df, min_pitches)
    cached = _PITCHER_PROFILE_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()

    df = statcast_df.dropna(subset=["pitcher", "pitch_type"])
    if df.empty:
        return pd.DataFrame(columns=base_columns)
    keep = ["pitcher", "pitch_type"] + [c for c in ("p_throws", "player_name") if c in df.columns]
    keep += [c for c in _PROFILE_NUMERIC.values() if c in df.columns]
    df = df[keep].copy()
    df["pitcher"] = pd.to_numeric(df["pitcher"], errors="coerce")
    df = df.dropna(subset=["pitcher"])
    if df.empty:
        return pd.DataFrame(columns=base_columns)
    # Converted once for the whole frame rather than once per pitcher per pitch type.
    for source in _PROFILE_NUMERIC.values():
        if source in df.columns:
            df[source] = pd.to_numeric(df[source], errors="coerce")

    counts = df.groupby("pitcher", observed=True).size()
    qualified = counts[counts >= min_pitches]
    if qualified.empty:
        return pd.DataFrame(columns=base_columns)
    df = df[df["pitcher"].isin(qualified.index)]

    out = pd.DataFrame({"PitcherID": qualified.index.astype(int),
                        "Pitches": qualified.to_numpy()})
    overall = df.groupby("pitcher", observed=True)[
        [c for c in _PROFILE_NUMERIC.values() if c in df.columns]].mean()
    for label, source in _PROFILE_NUMERIC.items():
        values = overall[source].reindex(qualified.index) if source in overall.columns else np.nan
        out[label] = np.round(
            np.asarray(values, dtype=float) * _PROFILE_SCALE[label], _PROFILE_ROUND[label])

    throws = _mode_by_pitcher(df, "p_throws").reindex(qualified.index)
    out["Throws"] = throws.fillna("").to_numpy()

    names = _mode_by_pitcher(df, "player_name").reindex(qualified.index)
    resolved = []
    for pitcher_id, raw in zip(qualified.index, names.to_numpy()):
        if isinstance(raw, str) and raw:
            if "," in raw:
                last, first = [part.strip() for part in raw.split(",", 1)]
                resolved.append(f"{first} {last}".strip())
            else:
                resolved.append(raw)
        else:
            resolved.append(_player_name_from_id(int(pitcher_id)))
    out["Name"] = resolved

    per_pitch = df.groupby(["pitcher", "pitch_type"], observed=True).agg(
        _n=("pitch_type", "size"),
        **{label: (source, "mean") for label, source in _PROFILE_NUMERIC.items()
           if source in df.columns})
    mix = (per_pitch["_n"] / per_pitch["_n"].groupby(level=0).transform("sum"))

    for pitch in PITCH_PROFILE_TYPES:
        try:
            block = per_pitch.xs(pitch, level="pitch_type")
            block_mix = mix.xs(pitch, level="pitch_type")
        except KeyError:
            out[f"Mix_{pitch}"] = 0.0
            for label in _PROFILE_NUMERIC:
                out[f"{label}_{pitch}"] = 0.0
            continue
        out[f"Mix_{pitch}"] = np.round(
            np.nan_to_num(block_mix.reindex(qualified.index).to_numpy(dtype=float)), 3)
        for label in _PROFILE_NUMERIC:
            values = (block[label].reindex(qualified.index).to_numpy(dtype=float)
                      if label in block.columns
                      else np.full(len(qualified), np.nan))
            # A pitch a pitcher never threw reads 0.0, as it did before.
            values = np.where(np.isnan(values), 0.0, values * _PROFILE_SCALE[label])
            out[f"{label}_{pitch}"] = np.round(values, _PROFILE_ROUND[label])

    out = out.reindex(columns=base_columns).reset_index(drop=True)
    _PITCHER_PROFILE_CACHE[cache_key] = out
    return out.copy()


def generate_pitcher_similarity_report(starter_pitcher_id, statcast_df, min_pitches=80, top_n=14):
    columns = ["PitcherID", "Name", "Throws", "Similarity", "Pitches", "Velo", "HB", "VB", "Top Mix"]
    profiles = _pitcher_profile_table(statcast_df, min_pitches=min_pitches)
    if profiles.empty or not starter_pitcher_id:
        return pd.DataFrame(columns=columns)

    starter_rows = profiles[profiles["PitcherID"] == int(starter_pitcher_id)]
    if starter_rows.empty:
        return pd.DataFrame(columns=columns)

    starter = starter_rows.iloc[0]
    candidates = profiles[profiles["PitcherID"] != int(starter_pitcher_id)].copy()
    starter_throws = str(starter.get("Throws", "")).strip()
    if starter_throws:
        candidates = candidates[candidates["Throws"].astype(str).str.strip().eq(starter_throws)].copy()
    if candidates.empty:
        return pd.DataFrame(columns=columns)

    mix_cols = [f"Mix_{pitch}" for pitch in PITCH_PROFILE_TYPES]
    starter_mix = starter[mix_cols].astype(float).values

    distances = []
    for _, row in candidates.iterrows():
        mix_distance = float(np.linalg.norm(row[mix_cols].astype(float).values - starter_mix))
        shape_distance = 0.0
        shape_weight = 0.0
        for pitch in PITCH_PROFILE_TYPES:
            starter_share = _safe_number(starter.get(f"Mix_{pitch}"), 0)
            row_share = _safe_number(row.get(f"Mix_{pitch}"), 0)
            weight = max(starter_share, row_share)
            if weight < 0.05:
                continue
            shape_weight += weight
            shape_distance += weight * (
                abs(_safe_number(row.get(f"Velo_{pitch}"), 0) - _safe_number(starter.get(f"Velo_{pitch}"), 0)) / 5.0
                + abs(_safe_number(row.get(f"Spin_{pitch}"), 0) - _safe_number(starter.get(f"Spin_{pitch}"), 0)) / 600.0
                + abs(_safe_number(row.get(f"HB_{pitch}"), 0) - _safe_number(starter.get(f"HB_{pitch}"), 0)) / 18.0
                + abs(_safe_number(row.get(f"VB_{pitch}"), 0) - _safe_number(starter.get(f"VB_{pitch}"), 0)) / 18.0
            )
        shape_distance = shape_distance / shape_weight if shape_weight else 0.0
        velo_distance = abs(_safe_number(row.get("Velo"), 0) - _safe_number(starter.get("Velo"), 0)) / 7.0
        spin_distance = abs(_safe_number(row.get("Spin"), 0) - _safe_number(starter.get("Spin"), 0)) / 900.0
        distance = (mix_distance * 2.4) + (shape_distance * 1.0) + velo_distance + (spin_distance * 0.4)
        distances.append(distance)

    candidates["Distance"] = distances
    candidates["Similarity"] = (100 - (candidates["Distance"] * 18)).clip(lower=0, upper=100).round(0).astype(int)

    top = candidates.sort_values(["Distance", "Pitches"], ascending=[True, False]).head(top_n).copy()
    top_mix = []
    for _, row in top.iterrows():
        mixes = sorted(
            [(pitch.replace("Mix_", ""), _safe_number(row[pitch], 0)) for pitch in mix_cols],
            key=lambda item: item[1],
            reverse=True,
        )
        top_mix.append("/".join(f"{pitch} {share * 100:.0f}%" for pitch, share in mixes[:3] if share > 0))
    top["Top Mix"] = top_mix
    return top[columns]


def _pitcher_type_bucket(comp_df):
    if comp_df is None or comp_df.empty:
        return "", [], ""
    throws = ""
    if "Throws" in comp_df.columns:
        throws_values = comp_df["Throws"].dropna().astype(str).str.strip()
        throws_values = throws_values[throws_values.ne("")]
        if not throws_values.empty:
            throws = throws_values.mode().iloc[0]

    pitch_codes = []
    if "Top Mix" in comp_df.columns:
        for mix in comp_df["Top Mix"].dropna().astype(str):
            for part in mix.split("/"):
                code = part.strip().split(" ")[0].strip()
                if code and code.upper() not in {"N/A", "NONE"}:
                    pitch_codes.append(code)
    top_pitches = pd.Series(pitch_codes).value_counts().head(4).index.tolist() if pitch_codes else []
    label_parts = []
    if throws:
        label_parts.append(throws)
    if top_pitches:
        label_parts.append("/".join(top_pitches[:3]))
    return throws, top_pitches, " / ".join(label_parts)


TYPE_MIN_SIMILARITY = 45      # below this the "comp" no longer resembles the starter
TYPE_TARGET_PA = 250          # widen the comp set until the sample is this big
TYPE_MIN_PA = 80              # under this the split is noise, not a read


def _apply_pitcher_type_bucket(df, comp_df, target_pa=TYPE_TARGET_PA,
                               min_similarity=TYPE_MIN_SIMILARITY):
    """PAs against pitchers who actually resemble tonight's starter.

    The previous implementation filtered by handedness plus "pitch type appears in the
    comp set", which kept ~every same-handed pitch in the league (FF/SL/CU/CH covers the
    sport) and made the type sample indistinguishable from the season baseline. Instead,
    walk the similarity-ranked comp list and take whole pitchers until the sample is big
    enough -- so the split stays "vs this kind of arm" rather than "vs right-handers".
    """
    if df is None or df.empty or comp_df is None or comp_df.empty:
        return pd.DataFrame()
    if "pitcher" not in df.columns or "PitcherID" not in comp_df.columns:
        return pd.DataFrame()

    comps = comp_df.copy()
    comps["_sim"] = pd.to_numeric(comps.get("Similarity"), errors="coerce").fillna(0)
    comps = comps[comps["_sim"] >= min_similarity].sort_values("_sim", ascending=False)
    if comps.empty:
        return pd.DataFrame()

    chosen, rows_pa = [], 0
    for pitcher_id in pd.to_numeric(comps["PitcherID"], errors="coerce").dropna().astype(int):
        chosen.append(pitcher_id)
        rows_pa = _events_to_batting_line(df[df["pitcher"].isin(chosen)])["PA"]
        if rows_pa >= target_pa:
            break
    return df[df["pitcher"].isin(chosen)].copy()


def generate_lineup_vs_similar_pitchers(lineup_df, similar_pitchers_df, statcast_df):
    columns = ["Name", "PA", "AB", "AVG", "SLG", "OPS_proxy", "HR", "K%", "vs Similar"]
    if (
        lineup_df is None or lineup_df.empty or "ID" not in lineup_df.columns
        or similar_pitchers_df is None or similar_pitchers_df.empty
        or statcast_df is None or statcast_df.empty
    ):
        return pd.DataFrame(columns=columns)

    comp_df = similar_pitchers_df.copy()
    batter_ids = pd.to_numeric(lineup_df["ID"], errors="coerce").dropna().astype(int).tolist()
    if not batter_ids or "PitcherID" not in comp_df.columns:
        return pd.DataFrame(columns=columns)

    # Take whole comp pitchers until the lineup has a workable sample against them --
    # a per-batter table needs more PA than the team-level one, so aim higher.
    lineup_rows = statcast_df[statcast_df["batter"].isin(batter_ids)].copy()
    df = _apply_pitcher_type_bucket(lineup_rows, comp_df,
                                    target_pa=max(TYPE_TARGET_PA, len(batter_ids) * 40))
    if df.empty:
        return pd.DataFrame(columns=columns)

    id_to_name = dict(zip(pd.to_numeric(lineup_df["ID"], errors="coerce"), lineup_df.get("Name")))
    _, _, bucket_label = _pitcher_type_bucket(comp_df)
    comp_names = bucket_label or ", ".join(comp_df["Name"].astype(str).head(3).tolist())
    rows = []
    for batter_id in batter_ids:
        batter_events = df[df["batter"] == batter_id]
        line = _events_to_batting_line(batter_events)
        pa = line["PA"]
        rows.append({
            "Name": id_to_name.get(float(batter_id), _player_name_from_id(batter_id)),
            "PA": pa,
            "AB": line["AB"],
            "AVG": line["AVG"],
            "SLG": line["SLG"],
            "OPS_proxy": line["OPS_proxy"],
            "HR": line["HR"],
            "K%": round((line["K"] / pa) * 100, 1) if pa else 0.0,
            "vs Similar": comp_names,
        })

    return pd.DataFrame(rows, columns=columns).sort_values(["OPS_proxy", "PA"], ascending=False)


def _statcast_team_rows(statcast_df, team_abbr):
    if statcast_df is None or statcast_df.empty:
        return pd.DataFrame()
    if not {"home_team", "away_team", "inning_topbot"}.issubset(statcast_df.columns):
        return pd.DataFrame()
    statcast_alias = {"ARI": "AZ", "OAK": "ATH"}
    team = statcast_alias.get(team_abbr.upper(), team_abbr.upper())
    df = statcast_df.copy()
    batting_team = np.where(df["inning_topbot"].astype(str).str.lower().eq("bot"), df["home_team"], df["away_team"])
    df["batting_team"] = batting_team
    return df[df["batting_team"].astype(str).str.upper().eq(team)].copy()


def _game_result_for_team(game_pk, team_abbr):
    url = f"https://statsapi.mlb.com/api/v1.1/game/{int(game_pk)}/feed/live"
    data = cached_json_request(url, namespace="statsapi")
    game_data = data.get("gameData", {})
    live_data = data.get("liveData", {})
    teams = game_data.get("teams", {})
    linescore = live_data.get("linescore", {})
    home_abbr = get_team_abbreviation(teams.get("home", {}).get("name", "")) or teams.get("home", {}).get("abbreviation")
    away_abbr = get_team_abbreviation(teams.get("away", {}).get("name", "")) or teams.get("away", {}).get("abbreviation")
    home_runs = _stat_int(linescore.get("teams", {}).get("home", {}), "runs")
    away_runs = _stat_int(linescore.get("teams", {}).get("away", {}), "runs")
    if team_abbr.upper() == str(home_abbr).upper():
        return {"Runs": home_runs, "Allowed": away_runs, "Result": "W" if home_runs > away_runs else "L"}
    if team_abbr.upper() == str(away_abbr).upper():
        return {"Runs": away_runs, "Allowed": home_runs, "Result": "W" if away_runs > home_runs else "L"}
    return {"Runs": 0, "Allowed": 0, "Result": ""}


def generate_team_pitcher_type_results(team_abbr, similar_pitchers_df, statcast_df):
    columns = ["Team", "Games", "Record", "Runs/G", "PA", "Type OPS", "Baseline OPS", "OPS Diff", "Consistency", "Pitcher Type"]
    if similar_pitchers_df is None or similar_pitchers_df.empty or statcast_df is None or statcast_df.empty:
        return pd.DataFrame(columns=columns)
    comp_df = similar_pitchers_df.copy()
    team_df = _statcast_team_rows(statcast_df, team_abbr)
    if team_df.empty or "PitcherID" not in comp_df.columns:
        return pd.DataFrame(columns=columns)

    comp_rows = _apply_pitcher_type_bucket(team_df, comp_df)
    if comp_rows.empty:
        return pd.DataFrame(columns=columns)

    # Baseline is the same handedness, not the whole season: otherwise "OPS vs type"
    # mostly re-measures the platoon split the report already shows elsewhere. This way
    # OPS Diff means "beyond how they normally hit LHP/RHP".
    throws, _, _ = _pitcher_type_bucket(comp_df)
    baseline_df = team_df
    if throws and "p_throws" in team_df.columns:
        same_hand = team_df[team_df["p_throws"].astype(str).str.strip().eq(throws)]
        if _events_to_batting_line(same_hand)["PA"] >= 200:
            baseline_df = same_hand
    baseline_line = _events_to_batting_line(baseline_df)
    type_line = _events_to_batting_line(comp_rows)

    game_results = []
    eligible_games = []
    if "game_pk" in comp_rows.columns:
        for game_pk, game_df in comp_rows.groupby("game_pk"):
            if _events_to_batting_line(game_df)["PA"] >= 12:
                eligible_games.append(game_pk)
    for game_pk in sorted(eligible_games):
        try:
            game_results.append(_game_result_for_team(game_pk, team_abbr))
        except Exception:
            continue

    wins = sum(1 for game in game_results if game.get("Result") == "W")
    losses = sum(1 for game in game_results if game.get("Result") == "L")
    runs_pg = round(np.mean([game.get("Runs", 0) for game in game_results]), 1) if game_results else 0.0

    def game_ops_values(df):
        values = []
        if df.empty or "game_pk" not in df.columns:
            return values
        for _, game_df in df.groupby("game_pk"):
            if _events_to_batting_line(game_df)["PA"] >= 8:
                values.append(_events_to_batting_line(game_df)["OPS_proxy"])
        return values

    type_ops_games = game_ops_values(comp_rows)
    baseline_ops_games = game_ops_values(baseline_df)
    type_std = float(np.std(type_ops_games)) if len(type_ops_games) >= 2 else None
    baseline_std = float(np.std(baseline_ops_games)) if len(baseline_ops_games) >= 2 else None
    if type_std is None or baseline_std is None:
        consistency = "N/A"
    elif type_std <= baseline_std * 0.85:
        consistency = "More stable"
    elif type_std >= baseline_std * 1.15:
        consistency = "More volatile"
    else:
        consistency = "Typical"

    delta = type_line["OPS_proxy"] - baseline_line["OPS_proxy"]
    _, _, bucket_label = _pitcher_type_bucket(comp_df)
    comp_names = bucket_label or ", ".join(comp_df["Name"].astype(str).head(4).tolist())
    comp_used = int(comp_rows["pitcher"].nunique()) if "pitcher" in comp_rows.columns else 0
    return pd.DataFrame([{
        "Team": team_abbr,
        "Games": int(wins + losses),
        "Record": f"{wins}-{losses}" if wins + losses else "0-0",
        "Runs/G": runs_pg,
        "PA": type_line["PA"],
        "Type OPS": type_line["OPS_proxy"],
        "Baseline OPS": baseline_line["OPS_proxy"],
        "OPS Diff": _format_ops_points(delta),
        "Consistency": consistency,
        "Pitcher Type": comp_names,
        "Comps": comp_used,
        "Baseline Basis": f"vs {throws}HP" if baseline_df is not team_df else "all pitchers",
        # Sample size decides whether this split is a read or a coin flip.
        "Confidence": ("Low" if type_line["PA"] < TYPE_MIN_PA
                       else "Medium" if type_line["PA"] < TYPE_TARGET_PA else "High"),
    }], columns=columns + ["Comps", "Baseline Basis", "Confidence"])


def generate_team_arsenal_type_results(team_abbr, starter_arsenal_df, starter_hand, statcast_df):
    columns = ["Team", "Games", "Record", "Runs/G", "PA", "Type OPS", "Baseline OPS", "OPS Diff", "Consistency", "Pitcher Type"]
    if statcast_df is None or statcast_df.empty or starter_arsenal_df is None or starter_arsenal_df.empty:
        return pd.DataFrame(columns=columns)

    pitch_weights = _arsenal_pitch_weights(starter_arsenal_df, top_n=3)
    top_pitches = [pitch for pitch, _ in pitch_weights]
    team_df = _statcast_team_rows(statcast_df, team_abbr)
    if team_df.empty or not top_pitches:
        return pd.DataFrame(columns=columns)

    bucket_rows = team_df.copy()
    if starter_hand and "p_throws" in bucket_rows.columns:
        bucket_rows = bucket_rows[bucket_rows["p_throws"].astype(str).str.strip().eq(starter_hand)]
    if "pitch_type" in bucket_rows.columns:
        bucket_rows = bucket_rows[bucket_rows["pitch_type"].isin(top_pitches)]
    if bucket_rows.empty:
        return pd.DataFrame(columns=columns)

    baseline_line = _events_to_batting_line(team_df)
    type_metrics = _weighted_arsenal_metrics(bucket_rows, pitch_weights)

    eligible_games = []
    if "game_pk" in bucket_rows.columns:
        for game_pk, game_df in bucket_rows.groupby("game_pk"):
            if _events_to_batting_line(game_df)["PA"] >= 12:
                eligible_games.append(game_pk)

    game_results = []
    for game_pk in sorted(eligible_games):
        try:
            game_results.append(_game_result_for_team(game_pk, team_abbr))
        except Exception:
            continue

    wins = sum(1 for game in game_results if game.get("Result") == "W")
    losses = sum(1 for game in game_results if game.get("Result") == "L")
    runs_pg = round(np.mean([game.get("Runs", 0) for game in game_results]), 1) if game_results else 0.0

    type_ops_games = []
    baseline_ops_games = []
    if "game_pk" in bucket_rows.columns:
        for _, game_df in bucket_rows.groupby("game_pk"):
            if _events_to_batting_line(game_df)["PA"] >= 8:
                type_ops_games.append(_weighted_arsenal_metrics(game_df, pitch_weights)["OPS"])
    if "game_pk" in team_df.columns:
        for _, game_df in team_df.groupby("game_pk"):
            if _events_to_batting_line(game_df)["PA"] >= 8:
                baseline_ops_games.append(_events_to_batting_line(game_df)["OPS_proxy"])

    type_std = float(np.std(type_ops_games)) if len(type_ops_games) >= 2 else None
    baseline_std = float(np.std(baseline_ops_games)) if len(baseline_ops_games) >= 2 else None
    if type_std is None or baseline_std is None:
        consistency = "N/A"
    elif type_std <= baseline_std * 0.85:
        consistency = "More stable"
    elif type_std >= baseline_std * 1.15:
        consistency = "More volatile"
    else:
        consistency = "Typical"

    delta = type_metrics["OPS"] - baseline_line["OPS_proxy"]
    return pd.DataFrame([{
        "Team": team_abbr,
        "Games": int(wins + losses),
        "Record": f"{wins}-{losses}" if wins + losses else "0-0",
        "Runs/G": runs_pg,
        "PA": type_metrics["PA"],
        "Type OPS": type_metrics["OPS"],
        "Baseline OPS": baseline_line["OPS_proxy"],
        "OPS Diff": _format_ops_points(delta),
        "Consistency": consistency,
        "Pitcher Type": _format_pitch_weight_basis(starter_hand, pitch_weights, starter_arsenal_df),
    }], columns=columns)


def generate_batter_pitcher_matchups(lineup_df, pitcher_id, season, end_date, lookback_seasons=5):
    columns = ["Name", "Years", "PA", "AB", "H", "HR", "BB", "K", "AVG", "SLG", "OPS", "Avg EV", "xBA", "xSLG"]
    if lineup_df is None or lineup_df.empty or not pitcher_id:
        return pd.DataFrame(columns=columns)

    start_season = max(2015, int(season) - int(lookback_seasons) + 1)
    start_date = _season_start_date(start_season)
    if end_date < start_date:
        return pd.DataFrame(columns=columns)

    pitcher_df = cached_dataframe_call("statcast_pitcher", statcast_pitcher, start_date, end_date, pitcher_id)
    if pitcher_df is None or pitcher_df.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for _, player in lineup_df.iterrows():
        batter_id = player.get("ID")
        if pd.isna(batter_id):
            continue
        batter_df = pitcher_df[pitcher_df["batter"] == int(batter_id)]
        line = _events_to_batting_line(batter_df)
        if line["PA"] == 0:
            continue
        if "game_year" in batter_df.columns:
            years = sorted(pd.to_numeric(batter_df["game_year"], errors="coerce").dropna().astype(int).unique().tolist())
        elif "game_date" in batter_df.columns:
            years = sorted(pd.to_datetime(batter_df["game_date"], errors="coerce").dropna().dt.year.unique().tolist())
        else:
            years = []
        rows.append({
            "Name": player.get("Name"),
            "Years": ",".join(str(year) for year in years[-3:]) if years else f"{start_season}-{season}",
            **{key: line[key] for key in ["PA", "AB", "H", "HR", "BB", "K", "AVG", "SLG"]},
            "OPS": line["OPS_proxy"],
            "Avg EV": round(pd.to_numeric(batter_df.get("launch_speed"), errors="coerce").mean(), 1),
            "xBA": round(pd.to_numeric(batter_df.get("estimated_ba_using_speedangle"), errors="coerce").mean(), 3),
            "xSLG": round(pd.to_numeric(batter_df.get("estimated_slg_using_speedangle"), errors="coerce").mean(), 3),
        })

    return pd.DataFrame(rows, columns=columns).sort_values(["PA", "SLG"], ascending=False) if rows else pd.DataFrame(columns=columns)


def generate_batter_bullpen_matchups(lineup_df, bullpen_df, season, end_date,
                                     lookback_seasons=5, available_only=False):
    """Career line for each batter against the opposing bullpen *as a whole*.

    The starter matchup table asks "how has this hitter done against this arm", which is the
    question with the worst sample in the report -- a median BvP line is single digits of
    plate appearances. Pooling every reliever asks a coarser question with a much better
    sample: a hitter who has faced a division rival's pen for four years can have a hundred
    plate appearances here against nine for the starter.

    That extra sample buys less than it looks like. Facing "the bullpen" is not facing one
    skill -- it is an average over arms that come and go between seasons, so a big total is
    mostly a statement about how often the two clubs have met, and division opponents will
    dominate the column. It is worth showing next to the starter line and worth nothing as a
    projection input, which is the same verdict the report already reached for BvP itself
    (HITTER_SPLIT_WEIGHTS["BvP"] is 0.0, shown and not scored).

    `available_only` restricts the pool to arms the bullpen table thinks can actually pitch
    tonight, which is the more decision-relevant read on a night with a taxed pen.
    """
    columns = ["Name", "Arms", "Years", "PA", "AB", "H", "HR", "BB", "K", "AVG", "SLG", "OPS"]
    if lineup_df is None or lineup_df.empty:
        return pd.DataFrame(columns=columns)
    if bullpen_df is None or not isinstance(bullpen_df, pd.DataFrame) or bullpen_df.empty:
        return pd.DataFrame(columns=columns)

    pen = bullpen_df
    if available_only and "Availability" in pen.columns:
        pen = pen[pen["Availability"].astype(str).str.strip().str.lower() == "available"]
    names = [str(n).strip() for n in pen.get("Name", pd.Series(dtype=str)) if str(n).strip()]
    if not names:
        return pd.DataFrame(columns=columns)

    start_season = max(2015, int(season) - int(lookback_seasons) + 1)
    start_date = _season_start_date(start_season)
    if end_date < start_date:
        return pd.DataFrame(columns=columns)

    # One Statcast pull per reliever, cached per arm. Keyed that way on purpose: the same
    # bullpen recurs in every game its club plays, so the cache is reused across the slate
    # and across dates, where a per-(batter, pitcher) key would never hit twice.
    frames = []
    for name in names:
        pitcher_id = lookup_player_id(name)
        if pitcher_id is None:
            continue
        try:
            frame = cached_dataframe_call("statcast_pitcher", statcast_pitcher,
                                          start_date, end_date, pitcher_id)
        except Exception:
            continue
        if frame is None or frame.empty:
            continue
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=columns)
    pen_df = pd.concat(frames, ignore_index=True)

    rows = []
    for _, player in lineup_df.iterrows():
        batter_id = player.get("ID")
        if pd.isna(batter_id):
            continue
        faced = pen_df[pen_df["batter"] == int(batter_id)]
        line = _events_to_batting_line(faced)
        if line["PA"] == 0:
            continue
        # Distinct arms actually faced, not arms in the pen -- "0-for-6 against five
        # different relievers" and "0-for-6 against one" are not the same note.
        arms = int(pd.to_numeric(faced.get("pitcher"), errors="coerce").dropna().nunique())
        if "game_year" in faced.columns:
            years = sorted(pd.to_numeric(faced["game_year"], errors="coerce")
                           .dropna().astype(int).unique().tolist())
        elif "game_date" in faced.columns:
            years = sorted(pd.to_datetime(faced["game_date"], errors="coerce")
                           .dropna().dt.year.unique().tolist())
        else:
            years = []
        rows.append({
            "Name": player.get("Name"),
            "Arms": arms,
            "Years": ",".join(str(year) for year in years[-3:]) if years
            else f"{start_season}-{season}",
            **{key: line[key] for key in ["PA", "AB", "H", "HR", "BB", "K", "AVG", "SLG"]},
            "OPS": line["OPS_proxy"],
        })

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["PA", "SLG"], ascending=False)


def build_matchup_overview(starter_df, bullpen_df, starter_name=None):
    """One row per hitter: career vs tonight's starter beside career vs the whole pen.

    Merged outer, because the two tables rarely cover the same hitters -- a hitter with no
    history against a rookie starter can still have fifty plate appearances against the pen,
    and dropping him would hide the larger of the two samples.
    """
    columns = ["Name", "SP", "SP PA", "SP AVG", "SP OPS", "SP HR",
               "Pen Arms", "Pen PA", "Pen AVG", "Pen OPS", "Pen HR"]
    starter_df = starter_df if isinstance(starter_df, pd.DataFrame) else pd.DataFrame()
    bullpen_df = bullpen_df if isinstance(bullpen_df, pd.DataFrame) else pd.DataFrame()
    if starter_df.empty and bullpen_df.empty:
        return pd.DataFrame(columns=columns)

    left = starter_df.reindex(columns=["Name", "PA", "AVG", "OPS", "HR"]).rename(
        columns={"PA": "SP PA", "AVG": "SP AVG", "OPS": "SP OPS", "HR": "SP HR"})
    right = bullpen_df.reindex(columns=["Name", "Arms", "PA", "AVG", "OPS", "HR"]).rename(
        columns={"Arms": "Pen Arms", "PA": "Pen PA", "AVG": "Pen AVG",
                 "OPS": "Pen OPS", "HR": "Pen HR"})
    merged = left.merge(right, on="Name", how="outer") if not left.empty and not right.empty \
        else (left if right.empty else right)
    merged["SP"] = starter_name or ""
    for column in columns:
        if column not in merged.columns:
            merged[column] = pd.NA
    # Sorted by the bigger of the two samples, so the rows worth reading are on top.
    order = merged[["SP PA", "Pen PA"]].apply(pd.to_numeric, errors="coerce").max(axis=1)
    return merged.reindex(columns=columns).assign(_o=order) \
        .sort_values("_o", ascending=False).drop(columns="_o").reset_index(drop=True)


def _reliever_recent_workload(player_id, season, end_date):
    params = {"stats": "gameLog", "group": "pitching", "season": season}
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
    # Game logs change after every appearance. Reusing the season-long cache here
    # leaves the L7 strip frozen at the date of the first report run.
    data = cached_json_request(url, params=params, namespace="statsapi", force=True)
    splits = data.get("stats", [{}])[0].get("splits", []) if data.get("stats") else []
    cutoff = datetime.strptime(str(end_date)[:10], "%Y-%m-%d")
    last3_start = cutoff - timedelta(days=2)
    last5_start = cutoff - timedelta(days=4)
    l7_start = cutoff - timedelta(days=6)
    totals = {"Last3D_G": 0, "Last3D_Pitches": 0, "Last3D_IP": 0.0, "Last5D_Pitches": 0,
              "Last2D_Pitches": 0, "B2B": False,
              "Last Outing": "", "L7 Usage": "-------", "L7 Pitches": "- - - - - - -"}

    last2_start = cutoff - timedelta(days=1)
    pitched_on = {}    # date -> pitches, used to spot consecutive days
    day_codes = {}  # date -> single-char usage code within the last 7 days
    day_appearances = {}  # date -> compact code + pitch-count tokens
    for split in sorted(splits, key=lambda item: item.get("date", ""), reverse=True):
        game_date = split.get("date")
        if not game_date:
            continue
        try:
            dt = datetime.strptime(str(game_date)[:10], "%Y-%m-%d")
        except (TypeError, ValueError):
            continue
        if dt > cutoff:
            continue
        stat = split.get("stat", {})
        pitches = _stat_int(stat, "numberOfPitches")
        ip = _ip_to_float(stat.get("inningsPitched", 0))
        if dt >= last3_start:
            totals["Last3D_G"] += 1
            totals["Last3D_Pitches"] += pitches
            totals["Last3D_IP"] += ip
        if dt >= last5_start:
            totals["Last5D_Pitches"] += pitches
        if dt >= last2_start:
            totals["Last2D_Pitches"] += pitches
        pitched_on[dt.date()] = pitched_on.get(dt.date(), 0) + pitches
        if dt >= l7_start:
            if _stat_int(stat, "saves"):
                code = "s"
            elif _stat_int(stat, "blownSaves"):
                code = "b"
            elif _stat_int(stat, "holds"):
                code = "h"
            else:
                code = "a"
            prev = day_codes.get(dt.date())
            if prev in (None, "a"):  # prefer a decision code if multiple outings in a day
                day_codes[dt.date()] = code
            day_appearances.setdefault(dt.date(), []).append(f"{code}{pitches}")
        if not totals["Last Outing"]:
            totals["Last Outing"] = f"{game_date}: {round(ip, 1)} IP/{pitches} pit"

    # Back-to-back *into tonight*: he worked yesterday and the day before. Consecutive days
    # earlier in the week say nothing about whether he is available now, so only the two days
    # ending at the cutoff count.
    totals["B2B"] = bool(pitched_on.get(cutoff.date()) and
                         pitched_on.get((cutoff - timedelta(days=1)).date()))

    # 7 single-char slots, oldest -> newest: - off, a appeared, s save, b blown, h hold
    totals["L7 Usage"] = "".join(
        day_codes.get((l7_start + timedelta(days=offset)).date(), "-")
        for offset in range(7)
    )
    totals["L7 Pitches"] = " ".join(
        "/".join(reversed(day_appearances.get((l7_start + timedelta(days=offset)).date(), []))) or "-"
        for offset in range(7)
    )
    totals["Last3D_IP"] = round(totals["Last3D_IP"], 1)
    return totals


def enhance_bullpen_analysis(bullpen_df, team_abbr, season, end_date):
    if bullpen_df is None or bullpen_df.empty:
        return bullpen_df
    rows = []
    roster = get_team_roster(team_abbr, as_of_date=end_date)
    name_to_id = {remove_accents(p.get("person", {}).get("fullName", "")): p.get("person", {}).get("id") for p in roster}
    for _, row in bullpen_df.iterrows():
        player = row.to_dict()
        player_id = name_to_id.get(remove_accents(str(player.get("Name", ""))))
        throws = None
        if player_id:
            workload = _reliever_recent_workload(player_id, season, _pregame_end_date(end_date) or end_date)
            throws = _pitcher_throw_code(player_id)
        else:
            workload = {"Last3D_G": 0, "Last3D_Pitches": 0, "Last3D_IP": 0.0,
                        "Last5D_Pitches": 0, "Last2D_Pitches": 0, "B2B": False,
                        "Last Outing": "", "L7 Usage": "-------",
                        "L7 Pitches": "- - - - - - -"}
        player.update(workload)
        player["Throws"] = throws or ""
        # Kept on the frame so downstream blocks can slice statcast by pitcher without
        # re-resolving names against the roster.
        player["MLBAM"] = player_id
        k_minus_bb = _safe_number(player.get("K%"), 0) - _safe_number(player.get("BB%"), 0)
        fip = _safe_number(player.get("FIP"), 5.0)
        whip = _safe_number(player.get("WHIP"), 1.35)
        run_quality = max(0, 5.0 - fip) * 4
        ratio_quality = max(0, 1.45 - whip) * 8
        performance_score = k_minus_bb + run_quality + ratio_quality
        availability_penalty = 10 if player["Last3D_Pitches"] >= 45 else 5 if player["Last3D_Pitches"] >= 30 else 0
        availability_score = max(0, 100 - player["Last3D_Pitches"] - (player["Last3D_G"] * 8))
        player["Performance Score"] = round(performance_score, 1)
        player["Availability Score"] = round(availability_score, 0)
        player["Bullpen Score"] = round(performance_score - availability_penalty, 1)
        player["Leverage Score"] = player["Bullpen Score"]
        save_opps = _safe_number(player.get("SVO"), 0)
        saves = _safe_number(player.get("SV"), 0)
        holds = _safe_number(player.get("HLD"), 0)
        blown = _safe_number(player.get("BS"), 0)
        sv_pct = _safe_number(player.get("SV%"), 0)
        if save_opps:
            player["Save Efficiency"] = f"{sv_pct * 100:.0f}% ({int(saves)}/{int(save_opps)}, {int(blown)} BS)"
        elif holds:
            player["Save Efficiency"] = f"{int(holds)} holds"
        else:
            player["Save Efficiency"] = ""
        # **Back-to-back days count as taxed.** Measured over the 2025 regular season from
        # the Statcast pitch log -- 281 relievers, 20,868 appearances, restricted to days
        # the club played again the next day -- a reliever who worked on consecutive days
        # appeared the following day only **5.5%** of the time (n=2,032), against **26.0%**
        # on a day or more of rest. Above 35 pitches across the two days it falls to 1.4%.
        # A three-day pitch total misses this entirely: 15 pitches yesterday and 12 the day
        # before totals 27 and used to grade "Monitor", while the arm is effectively out.
        #
        # This deliberately reuses the existing "Taxed" wording rather than adding a tier.
        # `bullpen_deployment_notes`, `summarize_bullpen_form` and the available-only filter
        # all match on these exact strings, so a new label would be silently dropped by each
        # of them. `B2B` and `Last2D_Pitches` carry the evidence for anything that wants it.
        if (player.get("B2B") or player["Last3D_Pitches"] >= 45
                or player["Last3D_G"] >= 3):
            player["Availability"] = "Taxed"
        elif player["Last3D_Pitches"] >= 25:
            player["Availability"] = "Monitor"
        else:
            player["Availability"] = "Available"
        rows.append(player)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    max_svo = _numeric_col(out, "SVO").max()
    max_sv = _numeric_col(out, "SV").max()
    max_holds = _numeric_col(out, "HLD").max()

    roles = []
    for _, row in out.iterrows():
        svo = _safe_number(row.get("SVO"), 0)
        saves = _safe_number(row.get("SV"), 0)
        holds = _safe_number(row.get("HLD"), 0)
        gf = _safe_number(row.get("GF"), 0)
        if (svo >= 5 and svo == max_svo) or (saves >= 5 and saves == max_sv and gf >= 6):
            roles.append("Closer")
        elif (svo >= 3 and max_svo and svo >= max_svo - 2) or (saves >= 3 and max_sv and saves >= max_sv - 2):
            roles.append("Co-closer")
        elif holds >= 4 and holds >= max_holds - 1:
            roles.append("Setup/HL")
        else:
            roles.append(row.get("Role", "Relief"))
    out["Closer Role"] = roles
    return out.sort_values(["Leverage Score", "IP"], ascending=False)


def bullpen_deployment_notes(team, bullpen_df):
    if bullpen_df is None or bullpen_df.empty:
        return [f"{team}: bullpen data unavailable."]
    notes = []
    high_lev = _top_rows(bullpen_df, "Leverage Score", n=3)
    if not high_lev.empty:
        notes.append(f"{team} high-leverage candidates: {', '.join(high_lev['Name'].astype(str).tolist())}.")
    limited = pd.DataFrame()
    if {"Availability", "Leverage Score"}.issubset(bullpen_df.columns):
        limited = bullpen_df[
            bullpen_df["Availability"].astype(str).isin(["Taxed", "Monitor"])
            & (pd.to_numeric(bullpen_df["Leverage Score"], errors="coerce").fillna(0) >= 20)
        ].copy()
    if not limited.empty:
        limited_bits = []
        for _, row in limited.sort_values(["Leverage Score", "Last3D_Pitches"], ascending=False).head(4).iterrows():
            role = row.get("Closer Role") or row.get("Role", "Relief")
            limited_bits.append(f"{row.get('Name')} ({role}, {row.get('Availability')}, {row.get('Last3D_Pitches', 0)} pitches/3D)")
        notes.append(f"{team} important relievers possibly limited/unavailable: " + "; ".join(limited_bits) + ".")
    if "Closer Role" in bullpen_df.columns:
        closers = bullpen_df[bullpen_df["Closer Role"].astype(str).isin(["Closer", "Co-closer"])].copy()
        if not closers.empty:
            closer_bits = []
            for _, row in closers.sort_values(["Closer Role", "SVO", "Leverage Score"], ascending=[True, False, False]).head(3).iterrows():
                efficiency = row.get("Save Efficiency") or f"{int(_safe_number(row.get('SV'), 0))} SV"
                closer_bits.append(f"{row.get('Name')} {row.get('Closer Role')} ({efficiency}, {row.get('Availability', '')})")
            notes.append(f"{team} late-inning save structure: " + "; ".join(closer_bits) + ".")
    taxed = bullpen_df[bullpen_df["Availability"].astype(str).isin(["Taxed", "Monitor"])] if "Availability" in bullpen_df else pd.DataFrame()
    if not taxed.empty:
        notes.append(f"{team} workload flags: " + "; ".join(f"{r['Name']} {r['Availability']} ({r.get('Last3D_Pitches', 0)} pitches/3D)" for _, r in taxed.head(4).iterrows()) + ".")
    lefties = bullpen_df[bullpen_df.get("Throws", "").astype(str).eq("L")] if "Throws" in bullpen_df else pd.DataFrame()
    if not lefties.empty:
        names = ", ".join(lefties.sort_values("Leverage Score", ascending=False)["Name"].astype(str).head(3).tolist())
        notes.append(f"{team} left-handed/platoon looks: {names}.")
    return notes or [f"{team}: no obvious bullpen deployment flags."]


def lineup_composition_notes(team, lineup_df, splits_df=None):
    if lineup_df is None or lineup_df.empty:
        return [f"{team}: lineup unavailable."]
    bats = lineup_df.get("Bats", pd.Series(dtype=str)).fillna("?").astype(str)
    left = bats.isin(["L", "S"]).sum()
    right = bats.eq("R").sum()
    power = lineup_df[_numeric_col(lineup_df, "ISO") >= 0.180]
    contact = pd.DataFrame()
    if splits_df is not None and not splits_df.empty:
        k_cols = [col for col in splits_df.columns if col.startswith("K%")]
        if k_cols:
            contact = splits_df[pd.to_numeric(splits_df[k_cols[0]], errors="coerce").fillna(100) <= 18]
    notes = [f"{team} lineup shape: {left} L/S bats and {right} R bats."]
    if not power.empty:
        notes.append(f"Power-pressure bats by ISO: {', '.join(power['Name'].astype(str).head(4).tolist())}.")
    if not contact.empty:
        notes.append(f"Contact/put-in-play pressure: {', '.join(contact['Name'].astype(str).head(4).tolist())}.")
    return notes


def pitcher_vs_lineup_notes(team, starter_info, arsenal_df, opponent_lineup_df, pitcher_split_df=None, opponent_arsenal_df=None):
    notes = []
    starter = _first_name(starter_info)
    bats = opponent_lineup_df.get("Bats", pd.Series(dtype=str)).fillna("?").astype(str) if opponent_lineup_df is not None else pd.Series(dtype=str)
    left = bats.isin(["L", "S"]).sum()
    right = bats.eq("R").sum()
    split_text = ""
    if pitcher_split_df is not None and not pitcher_split_df.empty:
        split_bits = []
        for _, row in pitcher_split_df.iterrows():
            side = row.get("Batter Side", "?")
            split_bits.append(f"vs {side} {row.get('OPS')} OPS/{row.get('xwOBA')} xwOBA ({row.get('Split Tag')})")
        if split_bits:
            split_text = " Pitcher splits: " + "; ".join(split_bits) + "."
    arsenal_text = ""
    if opponent_arsenal_df is not None and not opponent_arsenal_df.empty:
        row = opponent_arsenal_df.iloc[0]
        arsenal_text = (
            f" Opponent vs main shapes: {row.get('Arsenal Tag', 'N/A')} "
            f"({row.get('OPS', 'N/A')} OPS, {row.get('xwOBA', 'N/A')} xwOBA, "
            f"{row.get('K%', 'N/A')} K%)."
        )
    if arsenal_df is not None and not arsenal_df.empty:
        arsenal = arsenal_df.copy()
        arsenal["Usage %"] = _numeric_col(arsenal, "Usage %")
        arsenal["CSW%"] = _numeric_col(arsenal, "CSW%")
        top_rows = arsenal.sort_values("Usage %", ascending=False).head(2)
        top = "/".join(
            f"{row.get('Pitch')} {row.get('release_speed', 'N/A')} mph/{row.get('release_spin_rate', 'N/A')} rpm"
            for _, row in top_rows.iterrows()
        )
        notes.append(f"{team} starter {starter} vs lineup mix ({left} L/S, {right} R): main shapes {top}.{split_text}{arsenal_text}")
    if opponent_lineup_df is not None and not opponent_lineup_df.empty:
        power_count = (_numeric_col(opponent_lineup_df, "ISO") >= 0.180).sum()
        if power_count >= 4:
            notes.append(f"{starter} faces {power_count} ISO-threat bats; avoid predictable in-zone fastballs after first look.")
        elif power_count <= 2:
            notes.append(f"{starter} can be more zone-forward if command is present; lineup has fewer obvious ISO bats.")
    return notes


def get_umpire_runs_tendency(ump_name, season, end_date):
    """Total runs/game in games this umpire worked home plate, vs league average. None if unknown."""
    if not ump_name or ump_name == "Not yet assigned":
        return None
    start = _season_start_date(season)
    if not end_date or end_date < start:
        return None
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {"sportId": 1, "startDate": start, "endDate": end_date, "hydrate": "officials,linescore"}
    try:
        data = cached_json_request(url, params=params, namespace="statsapi", cache_key_extra=end_date)
    except Exception:
        return None
    ump_runs, all_runs = [], []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            if game.get("status", {}).get("detailedState") != "Final":
                continue
            teams = game.get("teams", {})
            hr = teams.get("home", {}).get("score")
            ar = teams.get("away", {}).get("score")
            if hr is None or ar is None:
                continue
            total = hr + ar
            all_runs.append(total)
            hp = next((o.get("official", {}).get("fullName") for o in game.get("officials", []) if o.get("officialType") == "Home Plate"), None)
            if hp == ump_name:
                ump_runs.append(total)
    if not ump_runs or not all_runs:
        return None
    ump_rpg = sum(ump_runs) / len(ump_runs)
    lg_rpg = sum(all_runs) / len(all_runs)
    return {"Umpire": ump_name, "Games": len(ump_runs), "R/G": round(ump_rpg, 2), "Lg R/G": round(lg_rpg, 2), "vs Avg": round(ump_rpg - lg_rpg, 2)}


UMPIRE_TAG_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "Umpire Tags - Sheet1.csv")


def _umpire_name_key(name):
    """Punctuation-insensitive key so e.g. CB and C.B. Bucknor match."""
    plain = remove_accents(str(name or "")).casefold()
    return re.sub(r"[^a-z0-9]", "", plain)


@lru_cache(maxsize=4)
def load_umpire_tags(path=UMPIRE_TAG_PATH):
    """Load the hand-tagged umpire sheet, keyed by normalized full name."""
    if not os.path.exists(path):
        return {}
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:
        print(f"Could not read umpire tags: {exc}")
        return {}
    required = {"Umpire", "ERA", "Rating"}
    if not required.issubset(frame.columns):
        return {}
    tags = {}
    for _, row in frame.iterrows():
        key = _umpire_name_key(row.get("Umpire"))
        if not key:
            continue
        era = _safe_number(row.get("ERA"), None)
        tags[key] = {
            "Umpire": str(row.get("Umpire", "")).strip(),
            "ERA": round(float(era), 2) if era is not None else None,
            "Rating": str(row.get("Rating", "")).strip(),
        }
    return tags


def get_umpire_tag(ump_name, path=UMPIRE_TAG_PATH):
    if not ump_name or ump_name == "Not yet assigned":
        return None
    return load_umpire_tags(path).get(_umpire_name_key(ump_name))


# --- handedness park factors (park_factors/*.csv, 3-year Savant-style indices) ------
PARK_FACTOR_DIR = os.path.join(PROJECT_ROOT, "park_factors")
_HANDED_PARK_FACTOR_CACHE = {}


def load_handed_park_factors():
    """{normalized venue name: {"RHB": {...}, "LHB": {...}}} from park_factors/*.csv.

    Values are indices where 100 is league-neutral, so 122 HR means 22% more homers than
    a neutral park. Keyed on the normalized venue name so sponsor renames still match.
    """
    if _HANDED_PARK_FACTOR_CACHE:
        return _HANDED_PARK_FACTOR_CACHE
    for hand, filename in (("RHB", "park_factors_RHB.csv"), ("LHB", "park_factors_LHB.csv")):
        path = os.path.join(PARK_FACTOR_DIR, filename)
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"⚠️ Could not read {filename}: {e}")
            continue
        for _, row in df.iterrows():
            key = _normalize_park_name(row.get("Venue"))
            if not key:
                continue
            entry = _HANDED_PARK_FACTOR_CACHE.setdefault(key, {})
            entry[hand] = {col: row.get(col) for col in
                           ("Park Factor", "R", "HR", "H", "1B", "2B", "3B",
                            "OBP", "BB", "SO", "HardHit", "xwOBAcon", "Year", "PA")}
    return _HANDED_PARK_FACTOR_CACHE


def handed_park_factors_for_venue(venue_name):
    """Handedness splits for a venue, matching the same aliases PARK_CONTEXT uses."""
    table = load_handed_park_factors()
    if not table:
        return {}
    normalized = _normalize_park_name(venue_name)
    if normalized in table:
        return table[normalized]
    # Sponsor renames ("UNIQLO Field at Dodger Stadium") -- match on containment.
    for key, entry in table.items():
        if key and (key in normalized or normalized in key):
            return entry
    alias = PARK_ALIASES.get(venue_name)
    if alias:
        alias_norm = _normalize_park_name(alias)
        for key, entry in table.items():
            if key and (key in alias_norm or alias_norm in key):
                return entry
    return {}


# --- weather forecast -------------------------------------------------------
# Roof state per park. "fixed" = always indoors (weather is irrelevant); "retractable"
# = usually closed in heat/rain, so treat the wind read as unreliable rather than wrong.
PARK_ROOF = {
    "American Family Field": "retractable",
    "Chase Field": "retractable",
    "Daikin Park": "retractable",
    "Globe Life Field": "retractable",
    "loanDepot park": "retractable",
    "Minute Maid Park": "retractable",
    "Rogers Centre": "retractable",
    "T-Mobile Park": "cover",   # roof covers the seating bowl but the park is not sealed
    "Tropicana Field": "fixed",
}

# Compass bearing (deg from true north) from home plate out toward center field, used to
# turn a forecast wind direction into a park-relative read. Source values are coarse
# (ballparks.com diagrams, rounded to 15 deg), so this is deliberately an ESTIMATE and is
# always superseded by MLB's own wind string ("Out To CF" etc.) once that is posted.
# Parks with no trustworthy public bearing are intentionally absent -- they fall back to a
# plain compass read instead of an invented park-relative one.
PARK_CF_BEARING = {
    "American Family Field": 135,
    "Angel Stadium": 45,
    "Busch Stadium": 60,
    "Chase Field": 0,
    "Citi Field": 30,
    "Citizens Bank Park": 15,
    "Comerica Park": 150,
    "Coors Field": 0,
    "Daikin Park": 345,
    "Dodger Stadium": 30,
    "Fenway Park": 45,
    "Globe Life Field": 135,
    "Great American Ball Park": 120,
    "Kauffman Stadium": 45,
    "Minute Maid Park": 345,
    "Nationals Park": 30,
    "Oakland Coliseum": 60,
    "Oracle Park": 90,
    "Oriole Park at Camden Yards": 30,
    "Petco Park": 0,
    "PNC Park": 120,
    "Progressive Field": 0,
    "Rate Field": 135,
    "Rogers Centre": 0,
    "T-Mobile Park": 45,
    "Target Field": 90,
    "Tropicana Field": 45,
    "Wrigley Field": 30,
    "Yankee Stadium": 75,
}

_COMPASS_16 = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
               "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def _compass_point(degrees):
    return _COMPASS_16[int((float(degrees) % 360) / 22.5 + 0.5) % 16]


def _venue_coordinates(venue_id):
    """(lat, lon) for an MLB venue id, or None."""
    if not venue_id:
        return None
    try:
        data = cached_json_request(
            f"https://statsapi.mlb.com/api/v1/venues/{int(venue_id)}",
            params={"hydrate": "location"},
            namespace="statsapi",
        )
    except Exception:
        return None
    for venue in data.get("venues", []):
        coords = (venue.get("location", {}) or {}).get("defaultCoordinates", {}) or {}
        lat, lon = coords.get("latitude"), coords.get("longitude")
        if lat is not None and lon is not None:
            return float(lat), float(lon)
    return None


def _park_relative_wind(from_degrees, cf_bearing):
    """Turn a meteorological wind direction (the direction it blows FROM) into MLB's
    park-relative vocabulary: out to LF/CF/RF, in from LF/CF/RF, or an L-to-R crosswind.

    Facing the outfield from home plate, turning clockwise on the compass turns toward
    right field, so a positive offset from the center-field bearing is the RF side.
    Buckets are wide because PARK_CF_BEARING is rounded to 15 degrees -- a bearing that
    is off by a notch should never flip an "out" read into an "in" read.
    """
    if from_degrees is None or cf_bearing is None:
        return None
    toward = (float(from_degrees) + 180.0) % 360.0          # direction the wind blows toward
    offset = (toward - float(cf_bearing) + 180.0) % 360.0 - 180.0   # signed, -180..180
    magnitude = abs(offset)
    if magnitude <= 30:
        return "out to CF"
    if magnitude <= 75:
        return "out to RF" if offset > 0 else "out to LF"
    if magnitude < 105:
        # Pure crosswind, named by where it travels: toward RF means it runs L to R.
        return "L to R" if offset > 0 else "R to L"
    if magnitude < 150:
        # Blowing back toward home: it originates over the *opposite* corner from the
        # direction it travels, so the sign flips relative to the "out to" cases.
        return "in from LF" if offset > 0 else "in from RF"
    return "in from CF"


def _first_pitch_utc(game_datetime, game_date):
    """Parse the feed's ISO first-pitch stamp to a naive UTC datetime; fall back to
    a 7pm-ET-ish evening slot when the feed has no time yet."""
    text = str(game_datetime or "").strip()
    if text:
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(
                ZoneInfo("UTC")).replace(tzinfo=None)
        except (ValueError, TypeError):
            pass
    try:
        return datetime.strptime(str(game_date), "%Y-%m-%d").replace(hour=23)
    except (ValueError, TypeError):
        return None


def get_weather_forecast(venue_id, venue_name, game_datetime, game_date):
    """First-pitch-hour forecast at the venue from Open-Meteo (no API key).

    Uses the forecast endpoint for today/future dates and the archive endpoint for past
    dates, so historical re-runs report the weather the game was actually played in.
    Returns None if the venue can't be geocoded or the API is unreachable -- callers fall
    back to MLB's own feed weather.
    """
    coords = _venue_coordinates(venue_id)
    if coords is None:
        return None
    lat, lon = coords
    target = _first_pitch_utc(game_datetime, game_date)
    if target is None:
        return None

    hourly = ["temperature_2m", "relative_humidity_2m", "apparent_temperature",
              "precipitation_probability", "precipitation", "cloud_cover",
              "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"]
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(hourly),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "GMT",          # match on UTC so we don't juggle venue-local offsets
        "start_date": str(game_date),
        "end_date": str(game_date),
    }
    today = datetime.now().strftime("%Y-%m-%d")
    if str(game_date) < today:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params.pop("precipitation_probability", None)
        params["hourly"] = ",".join(h for h in hourly if h != "precipitation_probability")
    else:
        url = "https://api.open-meteo.com/v1/forecast"
    try:
        # Forecasts move during the day; key the cache by date so a morning pull is not
        # served back for an evening re-render.
        data = cached_json_request(url, params=params, namespace="weather",
                                   cache_key_extra=today)
    except Exception as e:
        print(f"⚠️ Weather forecast unavailable for {venue_name}: {e}")
        return None

    block = data.get("hourly", {}) or {}
    times = block.get("time", []) or []
    if not times:
        return None
    stamp = target.strftime("%Y-%m-%dT%H:00")
    try:
        idx = times.index(stamp)
    except ValueError:
        idx = min(range(len(times)),
                  key=lambda i: abs(datetime.fromisoformat(times[i]) - target))

    def at(key):
        values = block.get(key) or []
        return values[idx] if idx < len(values) else None

    return {
        "source": "Open-Meteo archive" if url.endswith("/archive") else "Open-Meteo forecast",
        "valid_utc": times[idx],
        "temp_f": at("temperature_2m"),
        "feels_f": at("apparent_temperature"),
        "humidity_pct": at("relative_humidity_2m"),
        "precip_pct": at("precipitation_probability"),
        "precip_in": at("precipitation"),
        "cloud_pct": at("cloud_cover"),
        "wind_mph": at("wind_speed_10m"),
        "wind_gust_mph": at("wind_gusts_10m"),
        "wind_from_deg": at("wind_direction_10m"),
    }


def _sky_label(cloud_pct, precip_pct, precip_in):
    precip_pct = _safe_number(precip_pct, None)
    precip_in = _safe_number(precip_in, 0) or 0
    cloud_pct = _safe_number(cloud_pct, None)
    if precip_in >= 0.05 or (precip_pct is not None and precip_pct >= 60):
        return "Rain likely"
    if precip_pct is not None and precip_pct >= 30:
        return "Shower risk"
    if cloud_pct is None:
        return "Unknown"
    if cloud_pct >= 80:
        return "Overcast"
    if cloud_pct >= 40:
        return "Partly cloudy"
    return "Clear"


def build_weather_block(venue_name, venue_id, game_datetime, game_date, feed_weather):
    """Merge the forecast with MLB's feed weather into one display-ready dict.

    MLB's feed is authoritative but only posts near first pitch, and its wind string is
    already park-relative. The forecast fills the pregame gap and adds precip/humidity,
    which the feed never provides.
    """
    feed_weather = feed_weather or {}
    roof = PARK_ROOF.get(venue_name)
    forecast = get_weather_forecast(venue_id, venue_name, game_datetime, game_date)

    feed_temp = _safe_number(str(feed_weather.get("temp", "")).split()[0]
                             if feed_weather.get("temp") else None, None)
    feed_wind = str(feed_weather.get("wind", "") or "").strip()
    feed_condition = str(feed_weather.get("condition", "") or "").strip()

    temp = feed_temp if feed_temp is not None else _safe_number(
        (forecast or {}).get("temp_f"), None)
    temp_source = "MLB feed" if feed_temp is not None else (
        (forecast or {}).get("source", "") if forecast else "")

    # Wind: MLB's string wins because it is already stated relative to the park.
    wind_text, wind_estimated = "", False
    if feed_wind and feed_wind.lower() not in {"0 mph", "none"}:
        wind_text = feed_wind
    elif forecast and forecast.get("wind_mph") is not None:
        speed = _safe_number(forecast.get("wind_mph"), 0) or 0
        from_deg = forecast.get("wind_from_deg")
        point = _compass_point(from_deg) if from_deg is not None else ""
        relative = _park_relative_wind(from_deg, PARK_CF_BEARING.get(venue_name))
        if speed < 4:
            wind_text = f"{speed:.0f} mph, calm"
        elif relative:
            wind_text = f"{speed:.0f} mph {relative} (from {point})"
            wind_estimated = True
        else:
            wind_text = f"{speed:.0f} mph from {point}"
            wind_estimated = True

    sky = feed_condition or (
        _sky_label(forecast.get("cloud_pct"), forecast.get("precip_pct"),
                   forecast.get("precip_in")) if forecast else "")

    # Carry read: how the air is likely to treat fly balls, before park factor.
    carry = []
    if roof == "fixed":
        carry.append("indoor park — weather neutral")
    else:
        if temp is not None:
            if temp >= 85:
                carry.append("hot air adds carry")
            elif temp >= 78:
                carry.append("warm air, mild carry boost")
            elif temp <= 50:
                carry.append("cold air kills carry")
            elif temp <= 60:
                carry.append("cool air mutes carry")
        low = wind_text.lower()
        speed = _safe_number((forecast or {}).get("wind_mph"), None)
        strong = speed is not None and speed >= 10
        if "out to" in low:
            carry.append("wind out — HR risk up" if strong else "wind slightly out")
        elif "in from" in low:
            carry.append("wind in — HR risk down" if strong else "wind slightly in")
        elif "l to r" in low or "r to l" in low:
            carry.append("crosswind")
        if roof == "retractable" and carry:
            carry.append("roof may close")

    return {
        "roof": roof or "open",
        "temp_f": temp,
        "temp_source": temp_source,
        "feels_f": (forecast or {}).get("feels_f"),
        "humidity_pct": (forecast or {}).get("humidity_pct"),
        "precip_pct": (forecast or {}).get("precip_pct"),
        "precip_in": (forecast or {}).get("precip_in"),
        "wind": wind_text,
        "wind_estimated": wind_estimated,
        "wind_gust_mph": (forecast or {}).get("wind_gust_mph"),
        "sky": sky,
        "carry": "; ".join(carry),
        "forecast": forecast,
        "has_forecast": bool(forecast),
    }


def get_game_environment(game_id, game_date):
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
    # Force-fresh: officials/weather are assigned close to game time and change during the day.
    data = cached_json_request(url, namespace="statsapi", force=True)
    game_data = data.get("gameData", {})
    venue = game_data.get("venue", {})
    weather = game_data.get("weather", {}) or {}
    venue_name = venue.get("name", "Unknown venue")
    venue_id = venue.get("id")
    game_datetime = game_data.get("datetime", {}).get("dateTime", "")
    park, matched_park = _park_context_for_venue(venue_name)
    venue_label = venue_name if matched_park == venue_name else f"{venue_name} (matched to {matched_park})"
    notes = [f"{venue_label}: HR factor {park['HR']:.2f}, run factor {park['Runs']:.2f}; {park['profile']}."]

    # matched_park is the canonical PARK_CONTEXT key, which is also how PARK_ROOF and
    # PARK_CF_BEARING are keyed -- so venue aliases resolve for the weather read too.
    forecast_block = build_weather_block(matched_park, venue_id, game_datetime, game_date, weather)
    if forecast_block.get("temp_f") is not None or forecast_block.get("wind"):
        notes.append(
            "Weather at first pitch: "
            f"{forecast_block.get('sky') or 'N/A'}, "
            f"{forecast_block['temp_f']:.0f}F, wind {forecast_block.get('wind') or 'N/A'}"
            f" ({forecast_block.get('temp_source') or 'unknown source'})."
            if forecast_block.get("temp_f") is not None else
            f"Weather at first pitch: wind {forecast_block.get('wind')}."
        )
        if forecast_block.get("carry"):
            notes.append(f"Air/carry read: {forecast_block['carry']}.")
    else:
        notes.append("Weather unavailable from both MLB feed and forecast; use park baseline.")

    hp_umpire = "Not yet assigned"
    officials = data.get("liveData", {}).get("boxscore", {}).get("officials", []) or game_data.get("officials", [])
    for off in officials:
        if str(off.get("officialType", "")).strip() == "Home Plate":
            hp_umpire = off.get("official", {}).get("fullName", "Not yet assigned")
            break

    ump_tendency = get_umpire_runs_tendency(hp_umpire, int(str(game_date)[:4]), _pregame_end_date(game_date) or game_date)
    umpire_tag = get_umpire_tag(hp_umpire)
    if ump_tendency:
        hp_umpire_line = f"{hp_umpire} ({ump_tendency['R/G']} R/G, {ump_tendency['vs Avg']:+.2f} vs lg avg, {ump_tendency['Games']}g)"
        notes.append(f"Home plate umpire: {hp_umpire} — {ump_tendency['R/G']} runs/game in his {ump_tendency['Games']} plate games ({ump_tendency['vs Avg']:+.2f} vs league {ump_tendency['Lg R/G']}).")
    else:
        hp_umpire_line = hp_umpire
        notes.append(f"Home plate umpire: {hp_umpire}.")
    if umpire_tag:
        hp_umpire_line += f" | {umpire_tag['Rating']} (tag ERA {umpire_tag['ERA']:.2f})"
        notes.append(
            f"Umpire tag: {umpire_tag['Rating']} with a {umpire_tag['ERA']:.2f} ERA benchmark."
        )
    return {
        "game_id": game_id,
        "venue": venue_name,
        "venue_id": venue_id,
        "game_datetime": game_datetime,
        "park": park,
        "park_name": matched_park,
        "park_handed": handed_park_factors_for_venue(venue_name) or handed_park_factors_for_venue(matched_park),
        "weather": weather,
        "forecast": forecast_block,
        "notes": notes,
        "hp_umpire": hp_umpire,
        "hp_umpire_line": hp_umpire_line,
        "ump_tendency": ump_tendency,
        "umpire_tag": umpire_tag,
    }


def _recent_final_games(team_id, as_of_date, count=3):
    """The team's `count` most recent completed games before `as_of_date`.

    Returns newest-first dicts of gamePk plus the scoreboard context needed to label a
    box score (date, both clubs, score, W/L from this team's side)."""
    try:
        start = (datetime.strptime(str(as_of_date), "%Y-%m-%d") - timedelta(days=21)).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return []
    try:
        data = cached_json_request(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={"sportId": 1, "teamId": int(team_id), "startDate": start,
                    "endDate": str(as_of_date), "hydrate": "team,linescore"},
            namespace="statsapi",
            cache_key_extra=str(as_of_date),
        )
    except Exception as e:
        print(f"⚠️ Could not list recent games for team {team_id}: {e}")
        return []

    games = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            if game.get("status", {}).get("detailedState") != "Final":
                continue
            if game.get("officialDate", "") >= str(as_of_date):
                continue          # today's game hasn't been played yet
            teams = game.get("teams", {})
            home, away = teams.get("home", {}), teams.get("away", {})
            is_home = home.get("team", {}).get("id") == int(team_id)
            own, opp = (home, away) if is_home else (away, home)
            games.append({
                "game_pk": game.get("gamePk"),
                "date": game.get("officialDate", ""),
                "game_number": game.get("gameNumber", 1),
                "home_id": home.get("team", {}).get("id"),
                "away_id": away.get("team", {}).get("id"),
                "home_name": home.get("team", {}).get("name", ""),
                "away_name": away.get("team", {}).get("name", ""),
                "home_abbr": TEAM_ID_MAP.get(home.get("team", {}).get("id"), ""),
                "away_abbr": TEAM_ID_MAP.get(away.get("team", {}).get("id"), ""),
                "home_score": home.get("score"),
                "away_score": away.get("score"),
                "own_score": own.get("score"),
                "opp_score": opp.get("score"),
                "opp_id": opp.get("team", {}).get("id"),
                "venue_name": (game.get("venue", {}) or {}).get("name", ""),
                "is_home": is_home,
                "result": "W" if (own.get("score") or 0) > (opp.get("score") or 0) else "L",
            })
    # Newest first, with game 2 of a doubleheader ahead of game 1.
    games.sort(key=lambda g: (g["date"], g["game_number"]), reverse=True)
    return games[:count]


_BOX_BATTING_COLS = ["Player", "Pos", "AB", "R", "H", "HR", "RBI", "BB", "K", "SB",
                     "AVG", "OPS", "EV", "xBA", "Brl"]
_BOX_PITCHING_COLS = ["Player", "Dec", "IP", "H", "R", "ER", "BB", "K", "HR", "P-S", "ERA"]


def _boxscore_statcast_index(game_dates, statcast_df=None):
    """Per-game batted-ball quality keyed by (game_date, batter id).

    EV is average exit velocity on batted balls, Brl is barrel count
    (launch_speed_angle == 6), and xBA charges strikeouts as .000 the way Savant does,
    so a 0-for-4 with three punchouts doesn't show a flattering xBA off one hard out.
    """
    dates = sorted({str(d) for d in game_dates if d})
    if not dates:
        return {}
    df = statcast_df
    if df is None or df.empty:
        try:
            df = load_statcast_range(dates[0], dates[-1])
        except Exception as e:
            print(f"⚠️ Statcast unavailable for box scores: {e}")
            return {}
    if df is None or df.empty or "batter" not in df.columns:
        return {}

    sub = df[df["game_date"].astype(str).isin(dates)].copy()
    if sub.empty:
        return {}
    sub = sub[sub.get("events").notna()]
    if sub.empty:
        return {}
    sub["_ev"] = pd.to_numeric(sub.get("launch_speed"), errors="coerce")
    sub["_xba"] = pd.to_numeric(sub.get("estimated_ba_using_speedangle"), errors="coerce")
    sub["_lsa"] = pd.to_numeric(sub.get("launch_speed_angle"), errors="coerce")
    sub["_k"] = sub["events"].astype(str).str.startswith("strikeout")

    index = {}
    for (date, batter), group in sub.groupby([sub["game_date"].astype(str), "batter"]):
        batted = group[group["_ev"].notna()]
        xba_values = list(group.loc[group["_xba"].notna(), "_xba"]) + [0.0] * int(group["_k"].sum())
        index[(date, int(batter))] = {
            "EV": round(float(batted["_ev"].mean()), 1) if not batted.empty else "",
            "xBA": round(sum(xba_values) / len(xba_values), 3) if xba_values else "",
            "Brl": int((group["_lsa"] == 6).sum()),
        }
    return index


def _box_side_tables(box_team, statcast_index=None, game_date=None):
    """Batting and pitching line tables for one club out of a statsapi boxscore team block.

    statcast_index (from _boxscore_statcast_index) adds that game's EV / xBA / barrels
    per hitter; without it those columns come back blank."""
    statcast_index = statcast_index or {}
    players = box_team.get("players", {}) or {}

    def ordered(id_list):
        return [players.get(f"ID{pid}", {}) for pid in id_list if f"ID{pid}" in players]

    batting = []
    for player in ordered(box_team.get("batters", [])):
        stats = (player.get("stats", {}) or {}).get("batting", {}) or {}
        season = (player.get("seasonStats", {}) or {}).get("batting", {}) or {}
        if not stats:
            continue
        # MLB encodes the lineup slot as "100"/"200"...; a substitute gets "101", "201",
        # etc. Indent those with a leading dash the way a printed box score does.
        order = str(player.get("battingOrder") or "")
        name = player.get("person", {}).get("fullName", "")
        quality = statcast_index.get((str(game_date), player.get("person", {}).get("id")), {})
        batting.append({
            "Player": name if (not order or order.endswith("00")) else f"  - {name}",
            "Pos": (player.get("position", {}) or {}).get("abbreviation", ""),
            "AB": stats.get("atBats", 0), "R": stats.get("runs", 0), "H": stats.get("hits", 0),
            "HR": stats.get("homeRuns", 0), "RBI": stats.get("rbi", 0),
            "BB": stats.get("baseOnBalls", 0), "K": stats.get("strikeOuts", 0),
            "SB": stats.get("stolenBases", 0),
            "AVG": season.get("avg", ""), "OPS": season.get("ops", ""),
            "EV": quality.get("EV", ""), "xBA": quality.get("xBA", ""),
            "Brl": quality.get("Brl", ""),
        })

    pitching = []
    for player in ordered(box_team.get("pitchers", [])):
        stats = (player.get("stats", {}) or {}).get("pitching", {}) or {}
        season = (player.get("seasonStats", {}) or {}).get("pitching", {}) or {}
        if not stats:
            continue
        decision = " ".join(
            tag for tag, present in (
                ("W", stats.get("wins")), ("L", stats.get("losses")),
                ("SV", stats.get("saves")), ("HLD", stats.get("holds")),
                ("BS", stats.get("blownSaves")),
            ) if present
        )
        pitches, strikes = stats.get("numberOfPitches"), stats.get("strikes")
        pitching.append({
            "Player": player.get("person", {}).get("fullName", ""),
            "Dec": decision,
            "IP": stats.get("inningsPitched", ""), "H": stats.get("hits", 0),
            "R": stats.get("runs", 0), "ER": stats.get("earnedRuns", 0),
            "BB": stats.get("baseOnBalls", 0), "K": stats.get("strikeOuts", 0),
            "HR": stats.get("homeRuns", 0),
            "P-S": f"{pitches}-{strikes}" if pitches is not None else "",
            "ERA": season.get("era", ""),
        })

    bat_df = pd.DataFrame(batting, columns=_BOX_BATTING_COLS)
    pit_df = pd.DataFrame(pitching, columns=_BOX_PITCHING_COLS)

    # Team totals, the way a printed box score closes each half.
    if not bat_df.empty:
        totals = {"Player": "TEAM TOTALS", "Pos": "", "AVG": "", "OPS": "", "xBA": ""}
        for col in ("AB", "R", "H", "HR", "RBI", "BB", "K", "SB", "Brl"):
            totals[col] = int(pd.to_numeric(bat_df[col], errors="coerce").fillna(0).sum())
        # Team EV is the average of the hitters who actually put a ball in play.
        team_ev = pd.to_numeric(bat_df["EV"], errors="coerce").dropna()
        totals["EV"] = round(float(team_ev.mean()), 1) if not team_ev.empty else ""
        bat_df = pd.concat([bat_df, pd.DataFrame([totals])], ignore_index=True)[_BOX_BATTING_COLS]
    if not pit_df.empty:
        totals = {"Player": "TEAM TOTALS", "Dec": "", "ERA": "", "P-S": ""}
        totals["IP"] = round(sum(_ip_to_float(v) for v in pit_df["IP"]), 1)
        for col in ("H", "R", "ER", "BB", "K", "HR"):
            totals[col] = int(pd.to_numeric(pit_df[col], errors="coerce").fillna(0).sum())
        pit_df = pd.concat([pit_df, pd.DataFrame([totals])], ignore_index=True)[_BOX_PITCHING_COLS]

    return bat_df, pit_df


def build_recent_boxscores(team_id, team_abbr, as_of_date, count=3, statcast_df=None):
    """Full box scores (both clubs) for a team's last `count` completed games.

    Each entry carries a header line, a line score, and away/home batting+pitching tables
    ready to stack vertically on a worksheet. Pass the pipeline's already-loaded
    statcast frame to avoid a second pull just for the batted-ball columns."""
    metas = _recent_final_games(team_id, as_of_date, count)
    statcast_index = _boxscore_statcast_index([m.get("date") for m in metas], statcast_df)
    built = [_build_one_boxscore(meta, statcast_index) for meta in metas]
    return [game for game in built if game is not None]


def _build_one_boxscore(meta, statcast_index=None):
    """One game's banner, line score and four stat tables, or None if it will not load.

    Split out of `build_recent_boxscores` so the comparable-games tabs can render an
    arbitrary set of games rather than a chronological window, off the same code and the
    same disk cache.
    """
    game_pk = meta.get("game_pk")
    if not game_pk:
        return None
    try:
        data = cached_json_request(
            f"https://statsapi.mlb.com/api/v1/game/{int(game_pk)}/boxscore",
            namespace="statsapi",
        )
        line = cached_json_request(
            f"https://statsapi.mlb.com/api/v1/game/{int(game_pk)}/linescore",
            namespace="statsapi",
        )
    except Exception as e:
        print(f"⚠️ Box score unavailable for game {game_pk}: {e}")
        return None
    teams = data.get("teams", {}) or {}
    away_bat, away_pit = _box_side_tables(teams.get("away", {}) or {}, statcast_index, meta["date"])
    home_bat, home_pit = _box_side_tables(teams.get("home", {}) or {}, statcast_index, meta["date"])

    innings = line.get("innings", []) or []
    header = ["Team"] + [str(i.get("num", n + 1)) for n, i in enumerate(innings)] + ["R", "H", "E"]
    line_rows = []
    for side, abbr in (("away", meta["away_abbr"]), ("home", meta["home_abbr"])):
        totals = (line.get("teams", {}) or {}).get(side, {}) or {}
        row = {"Team": abbr or meta[f"{side}_name"]}
        for n, inning in enumerate(innings):
            frame = inning.get(side, {}) or {}
            runs = frame.get("runs")
            # A home team that never bats in the 9th gets "-", not a 0.
            row[str(inning.get("num", n + 1))] = "-" if runs is None else runs
        row["R"] = totals.get("runs", "")
        row["H"] = totals.get("hits", "")
        row["E"] = totals.get("errors", "")
        line_rows.append(row)

    own_abbr = meta["home_abbr"] if meta["is_home"] else meta["away_abbr"]
    opp_abbr = meta["away_abbr"] if meta["is_home"] else meta["home_abbr"]
    versus = f"vs {opp_abbr}" if meta["is_home"] else f"@ {opp_abbr}"
    # Both halves of a doubleheader share a date, so tag the game number.
    dh = f" (G{meta['game_number']})" if _safe_number(meta.get("game_number"), 1) > 1 else ""
    return {
        "title": (f"{meta['date']}{dh}   {own_abbr} {versus}   "
                  f"{meta['away_abbr']} {meta['away_score']} - {meta['home_abbr']} {meta['home_score']}   "
                  f"({meta['result']})"),
        "result": meta["result"],
        "date": meta["date"],
        "linescore": pd.DataFrame(line_rows, columns=header),
        "away_label": f"{meta['away_abbr']} Batting",
        "home_label": f"{meta['home_abbr']} Batting",
        "away_pitch_label": f"{meta['away_abbr']} Pitching",
        "home_pitch_label": f"{meta['home_abbr']} Pitching",
        "away_batting": away_bat, "away_pitching": away_pit,
        "home_batting": home_bat, "home_pitching": home_pit,
    }


_PEN_L5_GROUP_COLS = ["Date", "Opp", "Res", "SP IP", "Pen IP", "Arms", "Pit", "P/Out",
                      "H", "R", "ER", "K", "BB", "HR", "IR"]
# A relief outing this long is a bulk arm or a mop-up, not a leverage appearance. Marked in
# the arm grid so five innings of long relief cannot be read as the pen being worked hard.
_BULK_RELIEF_IP = 3.0


def _relief_lines(box_team):
    """(starter, [relievers]) for one club's boxscore, each as (player, pitching stats).

    `pitchers` is in appearance order, so the first arm with a line is the starter and
    everything after it is the pen. On an opener night that makes the bulk arm count as
    relief, which is the right answer for the question this table asks: he threw those
    innings out of the pen and he is unavailable tomorrow because of them.
    """
    players = box_team.get("players", {}) or {}
    lines = []
    for pid in box_team.get("pitchers", []) or []:
        player = players.get(f"ID{pid}")
        if not player:
            continue
        stats = (player.get("stats", {}) or {}).get("pitching", {}) or {}
        if stats:
            lines.append((player, stats))
    return (lines[0] if lines else None), lines[1:]


def build_bullpen_l5(team_id, team_abbr, as_of_date, count=5, bullpen_df=None):
    """Relief workload over the team's last `count` completed games, two ways.

    Returns (group_df, arm_df). The group frame is one row per game plus a TOTAL row --
    how many innings the pen was asked for and what it did with them. The arm frame is one
    row per reliever carrying a pitch count per game, which is the half that says who can
    actually pitch tonight.

    Counted per GAME, not per calendar day. `L7 Usage` in the scouting table below is a
    seven-day strip, so an off day silently makes it span six games and a doubleheader
    eight. Availability is a function of games and pitches, not of dates.

    The boxscores come from the same disk cache `build_recent_boxscores` fills for the
    "Last 3" tabs, so on a normal run three of the five are already local.
    """
    group_cols = list(_PEN_L5_GROUP_COLS)
    metas = _recent_final_games(team_id, as_of_date, count)
    if not metas:
        return pd.DataFrame(columns=group_cols), pd.DataFrame(columns=["Name"])

    # Newest first, matching the starter's L5 log directly above this on the sheet. The
    # arm grid's game columns are built from this same list, so the two read the same way.
    labels, group_rows = [], []
    per_arm = {}
    for meta in metas:
        game_pk = meta.get("game_pk")
        if not game_pk:
            continue
        try:
            data = cached_json_request(
                f"https://statsapi.mlb.com/api/v1/game/{int(game_pk)}/boxscore",
                namespace="statsapi",
            )
        except Exception as e:
            print(f"⚠️ Box score unavailable for pen L5 game {game_pk}: {e}")
            continue
        side = "home" if meta["is_home"] else "away"
        starter, relievers = _relief_lines((data.get("teams", {}) or {}).get(side, {}) or {})
        if starter is None and not relievers:
            continue

        # Both halves of a doubleheader share a date, so the column has to say which.
        label = str(meta.get("date", ""))[5:]
        if _safe_number(meta.get("game_number"), 1) > 1:
            label += f" G{int(meta['game_number'])}"
        labels.append(label)

        # `battingOrder`, not `batters`: the latter appends every pitcher who appeared, so
        # testing against it flagged the entire bullpen as position players. Under the
        # universal DH a real reliever is never in the batting order, so anyone who is
        # there and also pitched is a position player mopping up (or a two-way starter).
        lineup = set((data.get("teams", {}) or {}).get(side, {}).get("battingOrder", []) or [])
        pen_ip = pen_pitches = 0.0
        totals = {key: 0 for key in ("H", "R", "ER", "K", "BB", "HR")}
        inherited = scored = 0
        for player, stats in relievers:
            ip = _ip_to_float(stats.get("inningsPitched", 0))
            pitches = _safe_number(stats.get("numberOfPitches"), 0) or 0
            pen_ip += ip
            pen_pitches += pitches
            totals["H"] += _stat_int(stats, "hits")
            totals["R"] += _stat_int(stats, "runs")
            totals["ER"] += _stat_int(stats, "earnedRuns")
            totals["K"] += _stat_int(stats, "strikeOuts")
            totals["BB"] += _stat_int(stats, "baseOnBalls")
            totals["HR"] += _stat_int(stats, "homeRuns")
            inherited += _stat_int(stats, "inheritedRunners")
            scored += _stat_int(stats, "inheritedRunnersScored")

            person = player.get("person", {}) or {}
            arm = per_arm.setdefault(remove_accents(str(person.get("fullName", ""))), {
                "Name": person.get("fullName", ""), "games": {}, "IP": 0.0, "Pit": 0,
                "ER": 0, "K": 0, "BB": 0, "App": 0,
                # A position player mopping up is a fact about the game, not about the pen.
                "position_player": person.get("id") in lineup,
            })
            arm["IP"] += ip
            arm["Pit"] += int(pitches)
            arm["ER"] += _stat_int(stats, "earnedRuns")
            arm["K"] += _stat_int(stats, "strikeOuts")
            arm["BB"] += _stat_int(stats, "baseOnBalls")
            arm["App"] += 1
            token = f"{int(pitches)}" if pitches else "0"
            if ip >= _BULK_RELIEF_IP:
                token += "+"
            # Two outings in one game (rare, but a suspended game does it) stack rather
            # than overwrite, so the pitch count in the cell is the day's real total.
            prior = arm["games"].get(label)
            arm["games"][label] = f"{prior}/{token}" if prior else token

        pen_outs = pen_ip * 3
        group_rows.append({
            "Date": label,
            "Opp": ("vs " if meta["is_home"] else "@ ") + (
                meta["away_abbr"] if meta["is_home"] else meta["home_abbr"]),
            "Res": meta.get("result", ""),
            "SP IP": _float_to_ip(_ip_to_float((starter[1] if starter else {}).get("inningsPitched", 0))),
            "Pen IP": _float_to_ip(pen_ip),
            "Arms": len(relievers),
            "Pit": int(pen_pitches),
            "P/Out": round(pen_pitches / pen_outs, 1) if pen_outs else "",
            "IR": f"{scored}/{inherited}" if inherited else "-",
            **totals,
        })

    if not group_rows:
        return pd.DataFrame(columns=group_cols), pd.DataFrame(columns=["Name"])

    group = pd.DataFrame(group_rows, columns=group_cols)
    total_ip = sum(_ip_to_float(v) for v in group["Pen IP"])
    total_outs = total_ip * 3
    total_pitches = int(pd.to_numeric(group["Pit"], errors="coerce").fillna(0).sum())
    inherited_total = sum(int(str(v).split("/")[1]) for v in group["IR"] if "/" in str(v))
    scored_total = sum(int(str(v).split("/")[0]) for v in group["IR"] if "/" in str(v))
    summary = {
        "Date": "TOTAL", "Opp": f"{len(group)} games", "Res": "",
        "SP IP": _float_to_ip(sum(_ip_to_float(v) for v in group["SP IP"])),
        "Pen IP": _float_to_ip(total_ip),
        # DISTINCT relievers, not the column sum -- "how deep into the pen did this week
        # go" is the question, and summing appearances answers a different one. Written
        # with the unit attached so it cannot be misread as a total of the rows above.
        "Arms": f"{len(per_arm)} arms",
        "Pit": total_pitches,
        "P/Out": round(total_pitches / total_outs, 1) if total_outs else "",
        "IR": f"{scored_total}/{inherited_total}" if inherited_total else "-",
    }
    for col in ("H", "R", "ER", "K", "BB", "HR"):
        summary[col] = int(pd.to_numeric(group[col], errors="coerce").fillna(0).sum())
    group = pd.concat([group, pd.DataFrame([summary])], ignore_index=True)[group_cols]

    # ---- per-arm grid -------------------------------------------------------------
    hand, avail = {}, {}
    if bullpen_df is not None and not bullpen_df.empty and "Name" in bullpen_df.columns:
        keys = bullpen_df["Name"].map(lambda n: remove_accents(str(n)))
        hand = dict(zip(keys, bullpen_df.get("Throws", pd.Series("", index=bullpen_df.index))))
        avail = dict(zip(keys, bullpen_df.get("Availability", pd.Series("", index=bullpen_df.index))))

    arm_rows = []
    for key, arm in per_arm.items():
        outs = arm["IP"] * 3
        arm_rows.append({
            "Name": arm["Name"] + (" *" if arm["position_player"] else ""),
            "T": _pen_hand(hand.get(key, "")),
            **{label: arm["games"].get(label, "-") for label in labels},
            "App": arm["App"],
            "IP": _float_to_ip(arm["IP"]),
            "Pit": arm["Pit"],
            "P/Out": round(arm["Pit"] / outs, 1) if outs else "",
            "ERA": round(arm["ER"] * 9 / arm["IP"], 2) if arm["IP"] else "",
            "K-BB": arm["K"] - arm["BB"],
            "Avail": avail.get(key, ""),
        })
    arm_cols = ["Name", "T"] + labels + ["App", "IP", "Pit", "P/Out", "ERA", "K-BB", "Avail"]
    arms = pd.DataFrame(arm_rows, columns=arm_cols)
    if not arms.empty:
        # Busiest first: the top of this table is the reason the pen is short tonight.
        arms = arms.sort_values(["Pit", "App"], ascending=False).reset_index(drop=True)
    return group, arms


_RELEVANT_GAME_COLS = ["Date", "Opp", "H/A", "Res", "R", "Park", "Opp SP", "T", "K%",
                       "FIP", "HR/9", "Rest", "LU", "Match", "Rk"]
# How many comparable games the per-team tabs carry. Five, because that is the count the
# eye can hold side by side and because the gate has to stay satisfiable -- asking for ten
# comparable games out of a fifteen-game pool means taking games that are not comparable.
_COMPARABLE_COUNT = 5
# Candidate pool for the ranking. Deliberately larger than the L10 table it sits under:
# ten games is not enough to find five genuinely similar ones, and every game beyond the
# tenth costs exactly one boxscore fetch, which the disk cache then keeps.
_COMPARABLE_POOL = 15
# What counts as "tonight-like" on each dimension. Deliberately loose enough that a
# ten-game window still produces matches and tight enough that the matched set is not
# simply the window: on a sample of slates these land 3-5 games out of 10. Both failure
# modes are visible in the MATCHED row -- 10 matches means the test is doing nothing, 0
# or 1 means it is too strict to average.
_RELEVANT_PARK_TOLERANCE = 0.04     # park Runs factor, absolute
_RELEVANT_LINEUP_MIN = 7            # of tonight's nine who started that game
_RELEVANT_MATCH_MIN = 3             # dimensions that must agree to call a game comparable


def _tonight_conditions(environment, opposing_starter_info, lineup_df, is_home,
                        rest_schedule=None):
    """What a past game is compared against: run environment, starter type, game context,
    and tonight's projected lineup.

    Note what is NOT here: the opposing bullpen's state. Reconstructing a pen's workload as
    of some date three weeks ago means pulling that club's preceding three boxscores for
    every candidate game -- roughly sixty extra requests per team -- and it is the one
    dimension whose historical cost exceeds its value. Tonight's pen state is covered
    directly by the Bullpen L5 block on the pitching sheet instead.
    """
    park = (environment or {}).get("park") or {}
    ids = []
    if lineup_df is not None and not lineup_df.empty and "ID" in lineup_df.columns:
        ids = pd.to_numeric(lineup_df["ID"], errors="coerce").dropna().astype(int).tolist()

    # The starter frame already carries K%, so tonight's side of the comparison costs no
    # API call. It arrives as either a fraction or a percentage depending on the source,
    # the same ambiguity `cmp_starter_row` handles.
    k_rate = _safe_number((opposing_starter_info or {}).get("K%"), None)
    if k_rate is not None and abs(k_rate) > 1:
        k_rate /= 100.0

    rest = None
    if rest_schedule is not None and not getattr(rest_schedule, "empty", True):
        rest = _safe_number(str(_first_value(rest_schedule, "Rest") or "").rstrip("dD"), None)

    return {
        "park_runs": _safe_number(park.get("Runs"), None),
        "opp_sp_hand": _pen_hand((opposing_starter_info or {}).get("Throws")),
        "opp_sp_k": k_rate,
        "is_home": is_home,
        "rest": rest,
        "lineup_ids": ids,
    }


def _relevant_agreement(flags):
    """How many dimensions a Match string agrees on. '·' is a tested-and-differed slot."""
    text = str(flags or "")
    return sum(1 for ch in text if ch in "PTHL")


def _starter_profile(pitcher_id, season, end_date):
    """A starter's season K rate, FIP and HR/9 as of `end_date`. Blanks when unavailable.

    Ex-ante only. It is tempting to use the line the pitcher actually threw in the game
    being scored -- it is already in the boxscore and costs nothing -- but selecting past
    games by how the opposing pitcher performed and then averaging the runs scored against
    him is circular: the selection has already read the answer. Season talent as of that
    point is the honest version of "what kind of arm was this".

    FIP rather than ERA because the whole point is the arm rather than the defense behind
    it, and HR/9 because the run environment this table is built around acts on the ball in
    the air more than on anything else.
    """
    blank = {"k_rate": None, "fip": None, "hr9": None}
    if not pitcher_id:
        return blank
    try:
        stat = get_player_stat(int(pitcher_id), "pitching", season, end_date) or {}
    except Exception:
        return blank
    batters = _safe_number(stat.get("battersFaced"), 0) or 0
    if batters < 100:                 # too little to characterise an arm
        return blank
    innings = _ip_to_float(stat.get("inningsPitched", 0))
    strikeouts = _safe_number(stat.get("strikeOuts"), 0) or 0
    intentional = _safe_number(stat.get("intentionalWalks"), 0) or 0
    walks = (_safe_number(stat.get("baseOnBalls"), 0) or 0) - intentional
    homers = _safe_number(stat.get("homeRuns"), 0) or 0
    try:
        _, fip_constant = get_league_pitching_stats(int(season))
    except Exception:
        fip_constant = 3.1
    return {
        "k_rate": strikeouts / batters,
        "fip": round(calculate_fip(homers, walks, strikeouts, innings,
                                   fip_constant=fip_constant or 3.1), 2) if innings else None,
        "hr9": round(homers * 9 / innings, 2) if innings else None,
    }


def _relevant_game_features(meta, tonight_lineup, season, context_end):
    """Everything one past game contributes, from a single boxscore fetch.

    Shared by the L10 context table and the comparable-game ranker so the two can never
    disagree about a game, and so the boxscore is pulled once.
    """
    # Through the alias resolver, not a raw dict lookup: neutral-site and renamed venues
    # appear in the schedule under names PARK_CONTEXT does not key on, and a miss here
    # reads as "no park data" for a game that has perfectly good park data.
    park_context, _ = _park_context_for_venue(meta.get("venue_name"))
    out = {
        "park": _safe_number(park_context.get("Runs"), None),
        "opp_sp_name": "", "opp_sp_hand": "", "opp_sp_k": None,
        "opp_sp_fip": None, "opp_sp_hr9": None, "lineup_hits": None,
        "is_home": bool(meta.get("is_home")),
    }
    game_pk = meta.get("game_pk")
    if not game_pk:
        return out
    try:
        data = cached_json_request(
            f"https://statsapi.mlb.com/api/v1/game/{int(game_pk)}/boxscore",
            namespace="statsapi",
        )
    except Exception:
        return out
    teams = data.get("teams", {}) or {}
    own_side = "home" if meta["is_home"] else "away"
    opp_starter, _ = _relief_lines(teams.get("away" if meta["is_home"] else "home", {}) or {})
    if opp_starter:
        person = opp_starter[0].get("person", {}) or {}
        out["opp_sp_name"] = person.get("fullName", "")
        out["opp_sp_hand"] = _pen_hand(_pitcher_throw_code(person.get("id")))
        profile = _starter_profile(person.get("id"), season, context_end)
        out["opp_sp_k"] = profile["k_rate"]
        out["opp_sp_fip"] = profile["fip"]
        out["opp_sp_hr9"] = profile["hr9"]
    order = (teams.get(own_side, {}) or {}).get("battingOrder") or []
    if order and tonight_lineup:
        out["lineup_hits"] = len(tonight_lineup.intersection(int(p) for p in order))
    return out


def _rest_days(metas):
    """Rest going INTO each game. The list runs newest first, so the preceding game is the
    NEXT element -- reading it the other way reports the gap that FOLLOWED each game and
    shifts the whole column by a row."""
    rests = []
    for index, meta in enumerate(metas):
        value = ""
        if index + 1 < len(metas):
            try:
                value = (datetime.strptime(meta["date"], "%Y-%m-%d")
                         - datetime.strptime(metas[index + 1]["date"], "%Y-%m-%d")).days
            except (TypeError, ValueError):
                value = ""
        rests.append(value)
    return rests


# Similarity weights. Numeric features are standardized by the spread of the candidate
# pool itself rather than by a league SD -- it needs no lookup table and it asks the right
# question ("unusual relative to how much this team's own schedule varies"), at the cost
# of not being comparable between teams. Binary features contribute their full weight when
# they differ. NONE of these are fitted; they are a stated prior, and the honest test of
# the whole idea is still the MATCHED-vs-TOTAL comparison on the Game Context sheet.
_SIMILARITY_WEIGHTS = {"park": 1.0, "hand": 1.0, "k_rate": 1.0,
                       "home": 0.75, "rest": 0.5}
# How hard to prefer recent games among equally similar ones. At 0.35 a game at the far
# end of the pool needs to be noticeably more similar to outrank a recent one, which is
# what "the 5 most relevant AND recent" asks for.
_RECENCY_WEIGHT = 0.35
# Floor on a pool's spread, so a feature that barely varies cannot divide by ~0 and
# dominate the distance. In the feature's own units.
_SIMILARITY_SPREAD_FLOOR = {"park": 0.02, "k_rate": 0.02, "rest": 0.5}


def rank_relevant_games(metas, features, rests, tonight):
    """Score every candidate game against tonight; return indices best-first.

    Returns (order, detail) where detail[i] carries the distance and the gate outcome for
    game i, so the report can show WHY a game was picked rather than just a rank.

    Lineup overlap is a gate, not a distance term: a game where four of tonight's nine
    played is not a game this team played, and no amount of park agreement fixes that.
    The gate relaxes step by step when it would leave too few games, and the relaxation is
    reported -- a silent gate that returns two games is worse than a loose one that says
    it loosened.
    """
    tonight_park = _safe_number((tonight or {}).get("park_runs"), None)
    tonight_hand = str((tonight or {}).get("opp_sp_hand", "") or "")[:1].upper()
    tonight_home = (tonight or {}).get("is_home")
    tonight_k = _safe_number((tonight or {}).get("opp_sp_k"), None)

    def spread(values, key):
        clean = [v for v in values if v is not None]
        if len(clean) < 2:
            return _SIMILARITY_SPREAD_FLOOR.get(key, 1.0)
        sd = float(pd.Series(clean, dtype=float).std(ddof=0))
        return max(sd, _SIMILARITY_SPREAD_FLOOR.get(key, 1.0))

    park_sd = spread([f["park"] for f in features], "park")
    k_sd = spread([f["opp_sp_k"] for f in features], "k_rate")
    rest_sd = spread([r if r != "" else None for r in rests], "rest")

    detail = []
    for index, feature in enumerate(features):
        terms, used = 0.0, 0.0

        def add(key, deviation):
            nonlocal terms, used
            weight = _SIMILARITY_WEIGHTS[key]
            terms += weight * deviation ** 2
            used += weight

        if tonight_park is not None and feature["park"] is not None:
            add("park", (feature["park"] - tonight_park) / park_sd)
        if tonight_hand and feature["opp_sp_hand"]:
            add("hand", 0.0 if feature["opp_sp_hand"] == tonight_hand else 1.0)
        if tonight_k is not None and feature["opp_sp_k"] is not None:
            add("k_rate", (feature["opp_sp_k"] - tonight_k) / k_sd)
        if tonight_home is not None:
            add("home", 0.0 if bool(feature["is_home"]) == bool(tonight_home) else 1.0)
        rest = rests[index]
        if rest != "" and (tonight or {}).get("rest") not in (None, ""):
            add("rest", (float(rest) - float(tonight["rest"])) / rest_sd)

        # Normalised by the weight actually used, so a game missing a feature is not
        # rewarded for having fewer terms to disagree on.
        distance = (terms / used) ** 0.5 if used else float("inf")
        detail.append({"distance": distance,
                       "recency": _RECENCY_WEIGHT * (index / max(len(features) - 1, 1)),
                       "lineup_hits": feature["lineup_hits"]})

    # Progressive gate: try the strict overlap first and loosen only as far as needed.
    gate = _RELEVANT_LINEUP_MIN
    wanted = min(_COMPARABLE_COUNT, len(features))
    while gate > 0:
        eligible = [i for i, d in enumerate(detail)
                    if d["lineup_hits"] is None or d["lineup_hits"] >= gate]
        if len(eligible) >= wanted:
            break
        gate -= 1
    else:
        eligible = list(range(len(features)))

    for index, d in enumerate(detail):
        d["gate"] = gate
        d["eligible"] = index in eligible
        d["score"] = d["distance"] + d["recency"]

    order = sorted(eligible, key=lambda i: detail[i]["score"])
    return order, detail


def build_relevant_games(team_id, team_abbr, as_of_date, tonight, count=10,
                         season=None, context_end=None):
    """The team's last `count` games with tonight's conditions attached to each one.

    This is the *diagnostic*, not the selector -- it stays in date order so the roster
    drift and the schedule are both visible. `Rk` marks where each game landed in the
    similarity ranking that drives the comparable-games tabs, so the two views agree.

    The acceptance test lives in the last two rows: if MATCHED's runs-per-game equals
    TOTAL's, conditioning bought nothing on this slate and the honest read is that it is a
    display feature rather than a model input.
    """
    # Ranked over the SAME pool the comparable-games tabs use, then truncated for display.
    # Ranking over just the ten shown would put a different number in `Rk` than the tab
    # puts in `#` for the same game, which is worse than having no rank at all.
    metas = _recent_final_games(team_id, as_of_date, max(count, _COMPARABLE_POOL))
    if not metas:
        return pd.DataFrame(columns=_RELEVANT_GAME_COLS)

    season = season or int(str(as_of_date)[:4])
    context_end = context_end or _pregame_end_date(as_of_date) or as_of_date
    tonight_park = _safe_number((tonight or {}).get("park_runs"), None)
    tonight_hand = str((tonight or {}).get("opp_sp_hand", "") or "")[:1].upper()
    tonight_home = (tonight or {}).get("is_home")
    tonight_lineup = {int(pid) for pid in (tonight or {}).get("lineup_ids", []) or []}

    features = [_relevant_game_features(m, tonight_lineup, season, context_end) for m in metas]
    rests = _rest_days(metas)
    order, detail = rank_relevant_games(metas, features, rests, tonight)
    rank_of = {index: position + 1 for position, index in enumerate(order)}
    metas = metas[:count]

    rows = []
    for index, meta in enumerate(metas):
        feature = features[index]
        park, lineup_hits = feature["park"], feature["lineup_hits"]

        # One letter per dimension that agrees with tonight, so the reason a game is
        # comparable is legible rather than hidden behind a score.
        flags = ""
        if tonight_park is not None and park is not None:
            flags += "P" if abs(park - tonight_park) <= _RELEVANT_PARK_TOLERANCE else "·"
        if tonight_hand and feature["opp_sp_hand"]:
            flags += "T" if feature["opp_sp_hand"] == tonight_hand else "·"
        if tonight_home is not None:
            flags += "H" if bool(meta["is_home"]) == bool(tonight_home) else "·"
        if lineup_hits is not None:
            flags += "L" if lineup_hits >= _RELEVANT_LINEUP_MIN else "·"

        rows.append({
            "Date": str(meta.get("date", ""))[5:],
            "Opp": ("vs " if meta["is_home"] else "@ ") + (
                meta["away_abbr"] if meta["is_home"] else meta["home_abbr"]),
            "H/A": "H" if meta["is_home"] else "A",
            "Res": meta.get("result", ""),
            "R": meta.get("own_score"),
            "Park": park if park is not None else "",
            "Opp SP": feature["opp_sp_name"],
            "T": feature["opp_sp_hand"],
            "K%": round(feature["opp_sp_k"] * 100, 1) if feature["opp_sp_k"] is not None else "",
            "FIP": feature["opp_sp_fip"] if feature["opp_sp_fip"] is not None else "",
            "HR/9": feature["opp_sp_hr9"] if feature["opp_sp_hr9"] is not None else "",
            "Rest": rests[index],
            "LU": f"{lineup_hits}/9" if lineup_hits is not None else "",
            "Match": flags,
            "Rk": rank_of.get(index, ""),
        })

    table = pd.DataFrame(rows, columns=_RELEVANT_GAME_COLS)
    # Agreed dimensions, not "no disagreements": a game whose flags are empty because
    # nothing could be computed has agreed on nothing and must not count as comparable.
    matched = table["Match"].map(_relevant_agreement) >= _RELEVANT_MATCH_MIN

    def _runs(frame):
        values = pd.to_numeric(frame["R"], errors="coerce").dropna()
        return round(float(values.mean()), 2) if not values.empty else ""

    summary = [{
        **{c: "" for c in _RELEVANT_GAME_COLS},
        "Date": "TOTAL / AVG", "Opp": f"{len(table)} games", "R": _runs(table),
        "Park": round(float(pd.to_numeric(table["Park"], errors="coerce").dropna().mean()), 3)
        if pd.to_numeric(table["Park"], errors="coerce").notna().any() else "",
    }, {
        **{c: "" for c in _RELEVANT_GAME_COLS},
        # The payoff line. If this is the same as the row above, conditioning bought
        # nothing on this slate and the honest read is that it is a display feature.
        "Date": "MATCHED", "Opp": f"{int(matched.sum())} of {len(table)}",
        "R": _runs(table[matched]),
        "Match": f"{_RELEVANT_MATCH_MIN}+ of P/T/H/L",
    }]
    return pd.concat([table, pd.DataFrame(summary)], ignore_index=True)[_RELEVANT_GAME_COLS]


# `Dist` is pure similarity and `Score` is what the ranking actually sorted on -- distance
# plus the recency preference. Both are shown because only Score is monotonic down the
# table, and a reader seeing a larger Dist above a smaller one deserves the explanation
# rather than a suspicion that the sort is broken.
_COMPARABLE_SUMMARY_COLS = ["#", "Date", "Opp", "Res", "R", "Park", "Opp SP", "T", "K%",
                            "FIP", "HR/9", "Rest", "LU", "Match", "Dist", "Score"]


def build_comparable_games(team_id, team_abbr, as_of_date, tonight, count=_COMPARABLE_COUNT,
                           pool=_COMPARABLE_POOL, season=None, context_end=None,
                           statcast_df=None):
    """The `count` games most like tonight, with full box scores.

    Returns (games, summary, note): `games` in the shape `_write_boxscore_sheet` consumes,
    `summary` a table of why each was chosen, and `note` a plain sentence describing the
    selection -- including whether the lineup gate had to be relaxed to fill the slots.

    This replaces "last 3 completed games" on the per-team tabs. Recency answers "what
    happened lately"; three chronological games spanning a different park, a different
    handedness and two roster moves answer nothing in particular. The pool is still recent
    (`pool` games) and the ranking still prefers recent, so what changes is *which* recent
    games get the space, not how far back the tab reaches.
    """
    season = season or int(str(as_of_date)[:4])
    context_end = context_end or _pregame_end_date(as_of_date) or as_of_date
    metas = _recent_final_games(team_id, as_of_date, pool)
    if not metas:
        return [], pd.DataFrame(columns=_COMPARABLE_SUMMARY_COLS), "No completed games found."

    tonight_lineup = {int(pid) for pid in (tonight or {}).get("lineup_ids", []) or []}
    features = [_relevant_game_features(m, tonight_lineup, season, context_end) for m in metas]
    rests = _rest_days(metas)
    order, detail = rank_relevant_games(metas, features, rests, tonight)
    chosen = order[:count]

    tonight_park = _safe_number((tonight or {}).get("park_runs"), None)
    tonight_hand = str((tonight or {}).get("opp_sp_hand", "") or "")[:1].upper()
    tonight_home = (tonight or {}).get("is_home")

    rows = []
    for position, index in enumerate(chosen, start=1):
        meta, feature = metas[index], features[index]
        flags = ""
        if tonight_park is not None and feature["park"] is not None:
            flags += "P" if abs(feature["park"] - tonight_park) <= _RELEVANT_PARK_TOLERANCE else "·"
        if tonight_hand and feature["opp_sp_hand"]:
            flags += "T" if feature["opp_sp_hand"] == tonight_hand else "·"
        if tonight_home is not None:
            flags += "H" if bool(meta["is_home"]) == bool(tonight_home) else "·"
        if feature["lineup_hits"] is not None:
            flags += "L" if feature["lineup_hits"] >= _RELEVANT_LINEUP_MIN else "·"
        rows.append({
            "#": position,
            "Date": meta.get("date", ""),
            "Opp": ("vs " if meta["is_home"] else "@ ") + (
                meta["away_abbr"] if meta["is_home"] else meta["home_abbr"]),
            "Res": meta.get("result", ""), "R": meta.get("own_score"),
            "Park": feature["park"] if feature["park"] is not None else "",
            "Opp SP": feature["opp_sp_name"], "T": feature["opp_sp_hand"],
            "K%": round(feature["opp_sp_k"] * 100, 1) if feature["opp_sp_k"] is not None else "",
            "FIP": feature["opp_sp_fip"] if feature["opp_sp_fip"] is not None else "",
            "HR/9": feature["opp_sp_hr9"] if feature["opp_sp_hr9"] is not None else "",
            "Rest": rests[index],
            "LU": f"{feature['lineup_hits']}/9" if feature["lineup_hits"] is not None else "",
            "Match": flags,
            "Dist": round(detail[index]["distance"], 3),
            "Score": round(detail[index]["score"], 3),
        })
    summary = pd.DataFrame(rows, columns=_COMPARABLE_SUMMARY_COLS)
    if not summary.empty:
        def _avg(column, digits=2):
            values = pd.to_numeric(summary[column], errors="coerce").dropna()
            return round(float(values.mean()), digits) if not values.empty else ""
        summary = pd.concat([summary, pd.DataFrame([{
            **{c: "" for c in _COMPARABLE_SUMMARY_COLS},
            "#": "AVG", "Opp": f"{len(rows)} games", "R": _avg("R"),
            # The average arm faced across the comparable set, which is the number that
            # says whether this stretch of runs came against pitching like tonight's.
            "K%": _avg("K%", 1), "FIP": _avg("FIP"), "HR/9": _avg("HR/9"),
        }])], ignore_index=True)[_COMPARABLE_SUMMARY_COLS]

    gate = detail[0]["gate"] if detail else _RELEVANT_LINEUP_MIN
    note = (f"{len(chosen)} of the last {len(metas)} games, ranked by similarity to tonight "
            f"(park run factor, opposing starter hand and K%, home/away, rest) with a mild "
            f"preference for recency. Lineup gate: {gate}+ of tonight's nine started. "
            f"FIP and HR/9 are shown for context and do NOT enter the ranking.")
    if gate < _RELEVANT_LINEUP_MIN:
        note += (f"  ⚠ Relaxed from {_RELEVANT_LINEUP_MIN}+ — the roster has turned over "
                 f"enough that a stricter gate could not fill {count} games.")

    # Box scores only for the games that made the cut, so the extra fetches are bounded by
    # `count` rather than by the pool.
    statcast_index = _boxscore_statcast_index([metas[i].get("date") for i in chosen], statcast_df)
    games = []
    for position, index in enumerate(chosen, start=1):
        built = _build_one_boxscore(metas[index], statcast_index)
        if built is None:
            continue
        row = rows[position - 1]
        built["title"] = (f"#{position}  ·  {built['title']}"
                          f"   |   park {row['Park']}, {row['Opp SP']} ({row['T']}"
                          + (f", {row['K%']}% K" if row["K%"] != "" else "")
                          + f"), rest {row['Rest']}, lineup {row['LU']}, match {row['Match']}")
        games.append(built)
    return games, summary, note


def cmp_comparable_summary_fills(summary):
    """Shade the match column the same way the L10 diagnostic does, so a tab that had to
    reach for its fifth game says so rather than presenting all five as equivalent."""
    if summary is None or summary.empty or "Match" not in summary.columns:
        return {}
    fills = []
    for _, row in summary.iterrows():
        if str(row.get("#", "")).strip().upper() == "AVG":
            fills.append(None)
            continue
        agreed = _relevant_agreement(row.get("Match"))
        if agreed >= _RELEVANT_MATCH_MIN:
            fills.append(_DELTA_UP_STRONG)
        elif agreed == _RELEVANT_MATCH_MIN - 1:
            fills.append(_DELTA_UP_MILD)
        else:
            fills.append(_DELTA_DOWN_MILD)
    return {"Match": fills, "Dist": list(fills), "Score": list(fills)}


def cmp_relevant_games_fills(table):
    """Tint the rows that resemble tonight, so the comparable stretch reads as a block."""
    if table is None or table.empty or "Match" not in table.columns:
        return {}
    fills = []
    for _, row in table.iterrows():
        # Summary rows are identified by their Date label, not by parsing the Match cell:
        # the MATCHED row puts a caption there, and pattern-matching a caption is how a
        # summary line ends up shaded as though it were a game.
        if str(row.get("Date", "")).strip().upper() in _PITCHING_SUMMARY_ROWS | {"MATCHED"}:
            fills.append(None)
            continue
        agreed = _relevant_agreement(row.get("Match"))
        if agreed >= _RELEVANT_MATCH_MIN:
            fills.append(_DELTA_UP_STRONG)
        elif agreed == _RELEVANT_MATCH_MIN - 1:
            fills.append(_DELTA_UP_MILD)
        else:
            fills.append(None)
    return {"Match": fills, "Date": list(fills), "R": list(fills)}


def _resolve_game_id(date, away_team, home_team):
    """Find the gamePk for date + away/home abbreviations. None if not found."""
    try:
        data = cached_json_request(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={"sportId": 1, "date": date},
            namespace="statsapi",
            force=True,
        )
    except Exception:
        return None
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            teams = game.get("teams", {})
            away_id = teams.get("away", {}).get("team", {}).get("id")
            home_id = teams.get("home", {}).get("team", {}).get("id")
            if (TEAM_ID_MAP.get(away_id) == away_team
                    and TEAM_ID_MAP.get(home_id) == home_team):
                return game.get("gamePk")
    return None


def refresh_environment_umpire(context, date, away_team, home_team):
    """Re-pull the time-sensitive parts of the game environment on a cached re-render.

    Two things go stale between a morning pipeline run and first pitch:
      * the HP umpire -- MLB only populates boxscore.officials once a game flips
        Scheduled -> Pre-Game, so an early run bakes in "Not yet assigned";
      * the weather -- a forecast pulled hours ahead is the whole point of refreshing.
    Everything else stays cached.
    """
    env = context.get("environment") if isinstance(context, dict) else None
    if not isinstance(env, dict):
        return context
    ump_pending = env.get("hp_umpire", "Not yet assigned") == "Not yet assigned"
    game_id = env.get("game_id") or _resolve_game_id(date, away_team, home_team)
    if not game_id:
        return context
    try:
        fresh = get_game_environment(game_id, date)
    except Exception as e:
        print(f"⚠️ Could not refresh game environment: {e}")
        return context
    env["game_id"] = game_id

    env.setdefault("park_name", fresh.get("park_name"))
    env["park_handed"] = fresh.get("park_handed") or env.get("park_handed")

    fresh_forecast = fresh.get("forecast") or {}
    if fresh_forecast.get("temp_f") is not None or fresh_forecast.get("wind"):
        env["forecast"] = fresh_forecast
        env["weather"] = fresh.get("weather", env.get("weather"))
        env["notes"] = [n for n in env.get("notes", [])
                        if not str(n).startswith(("Weather at first pitch:", "Weather from MLB feed:",
                                                  "Weather unavailable", "Air/carry read:"))]
        env["notes"].extend(n for n in fresh.get("notes", [])
                            if str(n).startswith(("Weather at first pitch:", "Air/carry read:")))
        print(f"🔄 Weather refreshed: {fresh_forecast.get('sky')} "
              f"{fresh_forecast.get('temp_f')}F, wind {fresh_forecast.get('wind')}")

    if ump_pending and fresh.get("hp_umpire", "Not yet assigned") != "Not yet assigned":
        for key in ("hp_umpire", "hp_umpire_line", "ump_tendency", "umpire_tag"):
            env[key] = fresh.get(key)
        notes = env.get("notes")
        if isinstance(notes, list):
            fresh_note = next((n for n in fresh.get("notes", []) if str(n).startswith("Home plate umpire:")), None)
            if fresh_note:
                env["notes"] = [fresh_note if str(n).startswith("Home plate umpire:") else n for n in notes]
        print(f"🔄 Umpire now assigned: {env['hp_umpire_line']}")
    return context


def get_notable_absences(team_abbr, as_of_date, days_back=14, lineup_df=None):
    transactions = get_recent_transactions(team_abbr, days_back=days_back, as_of_date=as_of_date)
    lineup_names = set()
    if lineup_df is not None and not lineup_df.empty and "Name" in lineup_df.columns:
        lineup_names = {remove_accents(str(name)).lower() for name in lineup_df["Name"].dropna()}

    seen = set()
    injury_lines = []
    roster_lines = []
    for tx in transactions:
        player = tx.get("player", "N/A")
        normalized_player = remove_accents(str(player)).lower()
        if normalized_player in lineup_names:
            continue

        date_text = tx.get("date", tx.get("transactionDate", ""))
        try:
            date_text = datetime.strptime(date_text, "%Y-%m-%d").strftime("%m/%d")
        except (TypeError, ValueError):
            pass
        description = _clean_markdown(tx.get("description", tx.get("type", ""))).strip()
        desc_lower = description.lower()
        if "activated" in desc_lower:
            continue
        line = f"{date_text} - {player}: {description}"
        if "injured list" in desc_lower and "placed" in desc_lower:
            key = f"{normalized_player}|{date_text}|injured-list"
        else:
            key = remove_accents(line).lower()
        if key in seen:
            continue
        seen.add(key)

        if ("injured list" in desc_lower and "placed" in desc_lower) or ("placed on" in desc_lower and "injured" in desc_lower):
            injury_lines.append(line)
        elif any(term in desc_lower for term in ["optioned", "designated", "recalled", "selected", "roster status"]):
            roster_lines.append(line)

    return (injury_lines or roster_lines)[:8]


def generate_defensive_stress_report(def_team, opponent_team, defense_df, opponent_lineup_df, opponent_baserunning_df, environment):
    rows = []
    if opponent_lineup_df is None or opponent_lineup_df.empty:
        return pd.DataFrame(columns=["Unit", "Stress", "Why", "Priority"])
    park = environment.get("park", {}) if isinstance(environment, dict) else {}
    hr_factor = park.get("HR", 1.0)
    iso = _numeric_col(opponent_lineup_df, "ISO")
    sb_att = _numeric_col(opponent_baserunning_df, "SB_Att") if opponent_baserunning_df is not None and not opponent_baserunning_df.empty else pd.Series(dtype=float)
    power_bats = opponent_lineup_df[iso >= 0.180]["Name"].astype(str).head(5).tolist()
    speed_count = int((sb_att >= 5).sum()) if not sb_att.empty else 0
    of_quality = ""
    if defense_df is not None and not defense_df.empty and "Pos" in defense_df.columns:
        of_df = defense_df[defense_df["Pos"].isin(["LF", "CF", "RF"])]
        if not of_df.empty:
            of_quality = ", ".join(of_df.sort_values("Inn", ascending=False)["Name"].astype(str).head(3).tolist())
    if power_bats:
        rows.append({
            "Unit": f"{def_team} OF",
            "Stress": "High" if hr_factor >= 1.05 or len(power_bats) >= 4 else "Medium",
            "Why": f"{opponent_team} power bats: {', '.join(power_bats)}; park HR factor {hr_factor:.2f}.",
            "Priority": f"Clean routes and wall/gap reads. Main OF coverage: {of_quality or 'confirm alignment'}."
        })
    if speed_count:
        rows.append({
            "Unit": f"{def_team} C/MI",
            "Stress": "High" if speed_count >= 3 else "Medium",
            "Why": f"{opponent_team} has {speed_count} runners with 5+ steal attempts.",
            "Priority": "Control run game, vary times, pre-plan pitchouts/slide steps in obvious run counts."
        })
    contact_bats = opponent_lineup_df[_numeric_col(opponent_lineup_df, "AVG") >= 0.270]["Name"].astype(str).head(5).tolist()
    if contact_bats:
        rows.append({
            "Unit": f"{def_team} IF",
            "Stress": "Medium",
            "Why": f"Contact bats who can force action: {', '.join(contact_bats)}.",
            "Priority": "Prioritize clean exchanges, shift/positioning discipline, and first-step reads."
        })
    return pd.DataFrame(rows, columns=["Unit", "Stress", "Why", "Priority"])


def _statcast_fielding_rows(statcast_df, team_abbr):
    """Pitches thrown while `team_abbr` was in the field -- the inverse of the batting view."""
    if statcast_df is None or statcast_df.empty:
        return pd.DataFrame()
    if not {"home_team", "away_team", "inning_topbot"}.issubset(statcast_df.columns):
        return pd.DataFrame()
    statcast_alias = {"ARI": "AZ", "OAK": "ATH"}
    team = statcast_alias.get(team_abbr.upper(), team_abbr.upper())
    df = statcast_df.copy()
    top = df["inning_topbot"].astype(str).str.lower().eq("top")
    # Top of the inning: the away team bats, so the home team is fielding.
    fielding_team = np.where(top, df["home_team"], df["away_team"])
    df["fielding_team"] = fielding_team
    return df[df["fielding_team"].astype(str).str.upper().eq(team)].copy()


# Roughly what one non-home-run hit is worth to the batting side in DK points once the
# single/double/triple mix and the runs and RBI it tends to produce are counted. Used only
# to translate a defensive rate into fantasy units; the pitcher side (-0.6 per hit) is exact.
DK_POINTS_PER_HIT_ALLOWED = 4.5


_LEAGUE_DEFENSE_CACHE = {}


def _league_defense_baseline(statcast_df):
    """League-average (xBA - hit) per batted ball, overall and by batted-ball type.

    Computed once per statcast frame: thirty teams asking for it individually is thirty
    passes over the same two million rows.
    """
    # Keyed on content, not on id(): a garbage-collected frame's id can be handed to a
    # different frame, and serving one season's baseline to another would be silent.
    dates = statcast_df.get("game_date")
    key = (len(statcast_df), len(statcast_df.columns),
           str(dates.min()) if dates is not None and len(statcast_df) else "",
           str(dates.max()) if dates is not None and len(statcast_df) else "")
    if key in _LEAGUE_DEFENSE_CACHE:
        return _LEAGUE_DEFENSE_CACHE[key]
    events = statcast_df["events"].astype("string")
    batted = statcast_df[
        events.notna() & events.ne("") & events.ne("home_run")
        & statcast_df["estimated_ba_using_speedangle"].notna()
    ]
    baseline = {"all": 0.0, "ground": 0.0, "air": 0.0, "frame": 0.0}
    if not batted.empty:
        xba = pd.to_numeric(batted["estimated_ba_using_speedangle"], errors="coerce").fillna(0.0)
        is_hit = batted["events"].astype("string").isin(["single", "double", "triple"])
        bb_type = batted.get("bb_type", pd.Series(index=batted.index, dtype="string")).astype("string")
        for name, mask in (("all", pd.Series(True, index=batted.index)),
                           ("ground", bb_type.eq("ground_ball")),
                           ("air", bb_type.isin(["fly_ball", "line_drive", "popup"]))):
            chances = int(mask.sum())
            if chances:
                baseline[name] = float((xba[mask].sum() - is_hit[mask].sum()) / chances)
    taken = statcast_df[statcast_df["description"].astype("string").isin(
        ["ball", "called_strike", "blocked_ball"])]
    if not taken.empty and "zone" in taken.columns:
        zone = pd.to_numeric(taken["zone"].astype("string"), errors="coerce")
        called = taken["description"].astype("string").eq("called_strike")
        baseline["frame"] = float((int((called & (zone >= 11)).sum())
                                   - int((~called & zone.between(1, 9)).sum())) / len(taken))
    _LEAGUE_DEFENSE_CACHE[key] = baseline
    return baseline


def build_team_defense(team_abbr, statcast_df, min_bip=200):
    """Team defense measured from what happened to batted balls, not from putout counts.

    **Why this replaces the fielding-stat view.** Putouts and assists per inning -- what
    `_defensive_range_unit` scored -- are set by the pitching staff, not the fielders. A
    ground-ball staff hands its infield more chances and a fly-ball staff hands its outfield
    more, so that number rates the rotation and calls it defense. Errors and fielding
    percentage only count plays a fielder already reached, which is the opposite of range,
    and for DFS specifically they are close to irrelevant: an error produces *unearned* runs
    and DK scores a pitcher on earned runs only.

    What does matter is whether balls in play become hits. Statcast prices every batted ball
    with an expected batting average from its exit velocity and launch angle, so

        hits saved = sum(xBA) - actual hits

    is the defense's contribution with the hitter's contact quality already divided out.
    Home runs are excluded: no defence has a say in those.

    Split by `bb_type` into an infield (ground balls) and an outfield (air) component,
    because they price differently -- an outfield that cannot cover gaps costs doubles.
    """
    columns = ["Team", "BIP", "xBA Allowed", "BA Allowed", "Hits Saved/G",
               "IF Saved/G", "OF Saved/G", "Frame +Str/G", "Grade", "DFS Read"]
    rows = _statcast_fielding_rows(statcast_df, team_abbr)
    if rows.empty:
        return pd.DataFrame(columns=columns)

    # xBA is what makes this a defensive measure rather than a pitching one; without it
    # there is nothing honest to report, so the table is omitted rather than approximated.
    if "estimated_ba_using_speedangle" not in rows.columns:
        return pd.DataFrame(columns=columns)
    events = rows["events"].astype("string")
    in_play = rows[events.notna() & events.ne("")].copy()
    if in_play.empty:
        return pd.DataFrame(columns=columns)

    # Balls the defence could actually field: tracked contact that stayed in the park.
    batted = in_play[
        in_play["estimated_ba_using_speedangle"].notna()
        & in_play["events"].astype("string").ne("home_run")
    ].copy()
    if len(batted) < min_bip:
        return pd.DataFrame(columns=columns)

    xba = pd.to_numeric(batted["estimated_ba_using_speedangle"], errors="coerce").fillna(0.0)
    is_hit = batted["events"].astype("string").isin(["single", "double", "triple"])
    games = max(int(batted["game_pk"].nunique()), 1)

    bb_type = batted.get("bb_type", pd.Series(index=batted.index, dtype="string")).astype("string")
    ground = bb_type.eq("ground_ball")
    air = bb_type.isin(["fly_ball", "line_drive", "popup"])

    # xBA runs above actual BA league-wide -- it is fitted on exit velocity and launch angle
    # alone and cannot see the fielders who are, on average, there. Uncentred, that bias put
    # eighteen of thirty defences above zero and made the grade meaningless. So the league's
    # own gap is the zero point and a team is measured against it.
    league = _league_defense_baseline(statcast_df)

    def saved(mask, key):
        chances = int(mask.sum())
        if not chances:
            return 0.0
        team_rate = float((xba[mask].sum() - is_hit[mask].sum()) / chances)
        return (team_rate - league.get(key, 0.0)) * chances / games

    everything = pd.Series(True, index=batted.index)
    hits_saved = saved(everything, "all")
    if_saved, of_saved = saved(ground, "ground"), saved(air, "air")

    # Framing: taken pitches that went the defence's way. Called strikes on pitches outside
    # the zone, minus in-zone takes that were called balls. Worth including because it moves
    # the starter's strikeout rate, which is the largest term in his DK line.
    taken = rows[rows["description"].astype("string").isin(
        ["ball", "called_strike", "blocked_ball"])].copy()
    frame_per_game = 0.0
    if not taken.empty and "zone" in taken.columns:
        zone = pd.to_numeric(taken["zone"].astype("string"), errors="coerce")
        called = taken["description"].astype("string").eq("called_strike")
        stolen = int((called & (zone >= 11)).sum())
        lost = int((~called & (zone.between(1, 9))).sum())
        frame_games = max(int(taken["game_pk"].nunique()), 1)
        # Centred the same way and for the same reason as the batted-ball figure: the raw
        # stolen-minus-lost count is positive for everyone, so only the gap to league reads.
        rate = (stolen - lost) / max(len(taken), 1)
        frame_per_game = (rate - league.get("frame", 0.0)) * len(taken) / frame_games

    grade = ("Plus" if hits_saved >= 0.35 else
             "Risk" if hits_saved <= -0.35 else "Neutral")
    pitcher_swing = hits_saved * 0.6
    hitter_swing = -hits_saved * DK_POINTS_PER_HIT_ALLOWED
    read = (f"~{hits_saved:+.2f} hits/g vs expectation -> "
            f"{pitcher_swing:+.1f} DK to their pitching, "
            f"{hitter_swing:+.1f} DK to opposing bats")

    return pd.DataFrame([{
        "Team": team_abbr,
        "BIP": int(len(batted)),
        "xBA Allowed": round(float(xba.mean()), 3),
        "BA Allowed": round(float(is_hit.mean()), 3),
        "Hits Saved/G": round(hits_saved, 2),
        "IF Saved/G": round(if_saved, 2),
        "OF Saved/G": round(of_saved, 2),
        "Frame +Str/G": round(frame_per_game, 1),
        "Grade": grade,
        "DFS Read": read,
    }], columns=columns)


def _defensive_range_unit(defense_df, positions):
    if defense_df is None or defense_df.empty or "Pos" not in defense_df.columns:
        return {"Label": "N/A", "Score": 50, "Ch/9": 0.0, "Range/9": 0.0, "Fld%": 0.0}
    unit = defense_df[defense_df["Pos"].isin(positions)].copy()
    if unit.empty:
        return {"Label": "N/A", "Score": 50, "Ch/9": 0.0, "Range/9": 0.0, "Fld%": 0.0}

    innings = max(_numeric_col(unit, "Inn").sum(), 1.0)
    chances = _numeric_col(unit, "Chances").sum()
    assists = _numeric_col(unit, "Assists").sum()
    putouts = _numeric_col(unit, "PutOuts").sum()
    fielding = _mean_numeric(unit, "Fielding%", default=0.980)
    chances_per_9 = (chances / innings) * 9
    range_events_per_9 = ((assists + putouts) / innings) * 9

    if set(positions).intersection({"LF", "CF", "RF"}):
        expected_chances = 2.2
        expected_range = 2.1
        fielding_target = 0.990
    else:
        expected_chances = 4.0
        expected_range = 3.7
        fielding_target = 0.975

    chance_ratio = max(0.65, min(1.35, chances_per_9 / expected_chances))
    range_ratio = max(0.65, min(1.35, range_events_per_9 / expected_range))
    fielding_adj = max(-10, min(8, (fielding - fielding_target) * 700))
    sample_adj = min(5, innings / 180)
    score = 50 + ((chance_ratio - 1) * 28) + ((range_ratio - 1) * 34) + fielding_adj + sample_adj
    score = max(25, min(85, score))
    label = "Plus" if score >= 64 else "Risk" if score < 45 else "Solid"
    return {
        "Label": label,
        "Score": round(score, 0),
        "Ch/9": round(chances_per_9, 2),
        "Range/9": round(range_events_per_9, 2),
        "Fld%": round(fielding, 3),
    }


def generate_park_defense_impact(def_team, opponent_team, defense_df, opponent_lineup_df, opponent_baserunning_df, environment):
    columns = ["Defense", "In-Play Importance", "Range Fit", "OF Range", "IF Range", "Range Stress", "Why"]
    park = environment.get("park", {}) if isinstance(environment, dict) else {}
    hr_factor = park.get("HR", 1.0)
    run_factor = park.get("Runs", 1.0)
    profile = park.get("profile", "")
    contact_count = int((_numeric_col(opponent_lineup_df, "AVG") >= 0.270).sum()) if opponent_lineup_df is not None and not opponent_lineup_df.empty else 0
    speed_count = int((_numeric_col(opponent_baserunning_df, "SB_Att") >= 8).sum()) if opponent_baserunning_df is not None and not opponent_baserunning_df.empty else 0
    if run_factor >= 1.06 or "large" in profile.lower() or "gap" in profile.lower() or contact_count >= 3 or speed_count >= 3:
        importance = "High"
    elif run_factor >= 1.0 or contact_count >= 2 or speed_count >= 2:
        importance = "Medium"
    else:
        importance = "Low"

    of_range = _defensive_range_unit(defense_df, ["LF", "CF", "RF"])
    if_range = _defensive_range_unit(defense_df, ["1B", "2B", "3B", "SS"])
    range_stress_score = 0
    range_stress_score += 25 if importance == "High" else 12 if importance == "Medium" else 3
    range_stress_score += max(0, contact_count - 1) * 6
    range_stress_score += speed_count * 4
    range_stress_score += 8 if hr_factor >= 1.08 else 0
    range_stress_score += 8 if of_range["Label"] == "Risk" else -6 if of_range["Label"] == "Plus" else 0
    range_stress_score += 6 if if_range["Label"] == "Risk" else -5 if if_range["Label"] == "Plus" else 0
    range_fit = "Stress" if range_stress_score >= 35 else "Advantage" if range_stress_score <= 12 else "Neutral"
    why = (
        f"{park.get('profile', 'Neutral park')}; opponent contact bats {contact_count}, run threats {speed_count}, "
        f"HR factor {hr_factor:.2f}, run factor {run_factor:.2f}; "
        f"OF ch/9 {of_range['Ch/9']}, IF ch/9 {if_range['Ch/9']}."
    )
    return pd.DataFrame([{
        "Defense": def_team,
        "In-Play Importance": importance,
        "Range Fit": range_fit,
        "OF Range": f"{of_range['Label']} {int(of_range['Score'])}",
        "IF Range": f"{if_range['Label']} {int(if_range['Score'])}",
        "Range Stress": int(round(range_stress_score)),
        "Why": why,
    }], columns=columns)


def generate_catcher_run_game_report(def_team, opponent_team, defense_df, opponent_baserunning_df):
    columns = ["Defense", "Catcher", "Inn", "Run Control", "CS%", "CS", "SB Allowed", "PB", "Opp SB_Att", "Opp SB%", "Aggression", "Priority"]
    if defense_df is None or defense_df.empty:
        return pd.DataFrame(columns=columns)

    catchers = defense_df[defense_df.get("Pos", pd.Series(dtype=str)).astype(str).eq("C")].copy()
    if catchers.empty:
        return pd.DataFrame(columns=columns)

    opp_attempts = int(_numeric_col(opponent_baserunning_df, "SB_Att").sum()) if opponent_baserunning_df is not None and not opponent_baserunning_df.empty else 0
    opp_sb = int(_numeric_col(opponent_baserunning_df, "SB").sum()) if opponent_baserunning_df is not None and not opponent_baserunning_df.empty else 0
    opp_cs = int(_numeric_col(opponent_baserunning_df, "CS").sum()) if opponent_baserunning_df is not None and not opponent_baserunning_df.empty else 0
    opp_success = round(opp_sb / opp_attempts, 3) if opp_attempts else 0.0
    threats = opponent_baserunning_df[_numeric_col(opponent_baserunning_df, "SB_Att") >= 8] if opponent_baserunning_df is not None and not opponent_baserunning_df.empty else pd.DataFrame()
    if opp_attempts >= 70 or len(threats) >= 5:
        aggression = "High"
    elif opp_attempts >= 35 or len(threats) >= 3 or (opp_attempts >= 20 and opp_success >= 0.78):
        aggression = "Medium"
    else:
        aggression = "Low"

    rows = []
    for _, catcher in catchers.sort_values("Inn", ascending=False).head(2).iterrows():
        cs_pct = _safe_number(catcher.get("CS%"), 0.0)
        catcher_cs = int(_safe_number(catcher.get("CS"), 0))
        catcher_sb_allowed = int(_safe_number(catcher.get("SB Allowed"), 0))
        catcher_attempts = catcher_cs + catcher_sb_allowed
        if cs_pct and cs_pct <= 1:
            cs_pct_display = round(cs_pct * 100, 1)
        else:
            cs_pct_display = round(cs_pct, 1)
        if catcher_attempts < 5:
            run_control = "Small sample"
            cs_pct_value = "Small sample"
        elif cs_pct_display >= 30:
            run_control = "Strong"
            cs_pct_value = cs_pct_display
        elif cs_pct_display < 20:
            run_control = "Vulnerable"
            cs_pct_value = cs_pct_display
        else:
            run_control = "Average"
            cs_pct_value = cs_pct_display
        if run_control == "Small sample":
            priority = "Catcher sample is thin; lean on pitcher times, holds, and matchup-specific runner tells."
        elif aggression == "High" and cs_pct_display < 25:
            priority = "Run pressure likely; vary times and pre-plan throw-through counts."
        elif aggression == "High":
            priority = "Opponent will test; catcher arm can be part of run prevention plan."
        elif aggression == "Medium" and opp_success >= 0.78:
            priority = "Selective aggression; prioritize first-move quality and pitch timing."
        elif aggression == "Medium":
            priority = "Pick spots; control obvious runners without disrupting pitch plan."
        else:
            priority = "Standard control; do not over-borrow from pitch-calling plan."
        rows.append({
            "Defense": def_team,
            "Catcher": catcher.get("Name"),
            "Inn": catcher.get("Inn"),
            "Run Control": run_control,
            "CS%": cs_pct_value,
            "CS": catcher_cs,
            "SB Allowed": catcher_sb_allowed,
            "PB": catcher.get("PB", 0),
            "Opp SB_Att": opp_attempts,
            "Opp SB%": opp_success,
            "Aggression": aggression,
            "Priority": priority,
        })
    return pd.DataFrame(rows, columns=columns)


def _opponent_quality_score(opp_w_pct=0.5, opp_rpg=4.4, opp_rapg=None, opp_ops=None):
    score = 50
    score += (_safe_number(opp_w_pct, 0.5) - 0.5) * 90
    score += (_safe_number(opp_rpg, 4.4) - 4.4) * 7
    if opp_rapg is not None:
        score += (4.4 - _safe_number(opp_rapg, 4.4)) * 5
    if opp_ops is not None:
        score += (_safe_number(opp_ops, 0.700) - 0.700) * 35
    return round(max(0, min(100, score)), 1)


def _opponent_quality_label(score):
    score = _safe_number(score, 50)
    if score >= 56:
        return "Hard"
    if score <= 44:
        return "Soft"
    return "Average"


def get_recent_form_opponent_context(team_id, team_abbr, season, as_of_date, max_games=10):
    columns = ["Team", "Recent", "Runs/G", "Allowed/G", "Opp Quality", "Opp Quality Score", "Opp Combined", "Avg Opp W%", "Avg Opp R/G", "Avg Opp RA/G", "Star SP Faced"]
    detail_columns = ["Date", "Opp", "Result", "Score", "Opp W%", "Opp R/G", "Opp RA/G", "Opp Quality Score", "SP Faced", "SP FIP"]
    end_date = datetime.strptime(as_of_date, "%Y-%m-%d")
    start_date = (end_date - timedelta(days=30)).strftime("%Y-%m-%d")
    url = (
        f"https://statsapi.mlb.com/api/v1/schedule?"
        f"sportId=1&teamId={team_id}&startDate={start_date}&endDate={as_of_date}&sort=desc&hydrate=team,linescore"
    )
    data = cached_json_request(url, namespace="statsapi")
    rows = []
    for date_entry in reversed(data.get("dates", [])):
        for game in date_entry.get("games", []):
            if game.get("status", {}).get("detailedState") != "Final":
                continue
            is_home = game["teams"]["home"]["team"]["id"] == team_id
            opp_side = "away" if is_home else "home"
            team_side = "home" if is_home else "away"
            opp_id = game["teams"][opp_side]["team"]["id"]
            opp_name = game["teams"][opp_side]["team"]["name"]
            opp_abbr = get_team_abbreviation(opp_name) or TEAM_ID_MAP.get(opp_id, opp_name)
            team_score = game["teams"][team_side].get("score", 0)
            opp_score = game["teams"][opp_side].get("score", 0)
            result = "W" if team_score > opp_score else "L"
            game_date = game.get("officialDate")
            context_day = _pregame_end_date(game_date) or as_of_date
            opp_perf = get_team_performance(opp_id, season=season, as_of_date=context_day)
            opp_rpg, opp_rapg = _team_runs_per_game(opp_id, season, context_day)
            opp_quality_score = _opponent_quality_score(_record_pct(opp_perf), opp_rpg, opp_rapg)

            sp_name = ""
            sp_fip = None
            try:
                box = statsapi.boxscore_data(game["gamePk"])
                pitchers = box.get(f"{opp_side}Pitchers", [])
                sp_id = None
                for pitcher in pitchers:
                    if isinstance(pitcher, dict):
                        candidate_id = pitcher.get("personId")
                        candidate_name = pitcher.get("name") or pitcher.get("namefield")
                    else:
                        candidate_id = pitcher
                        candidate_name = ""
                    try:
                        candidate_id = int(candidate_id)
                    except (TypeError, ValueError):
                        candidate_id = None
                    if candidate_id and candidate_id > 0:
                        sp_id = candidate_id
                        sp_name = candidate_name or _player_name_from_id(candidate_id)
                        break
                if sp_id:
                    stat = get_player_stat(sp_id, "pitching", season, end_date=context_day)
                    ip = _ip_to_float(stat.get("inningsPitched", 0))
                    k = _stat_float(stat, "strikeOuts")
                    bb = _stat_float(stat, "baseOnBalls") - _stat_float(stat, "intentionalWalks")
                    hr = _stat_float(stat, "homeRuns")
                    _, fip_constant = get_league_pitching_stats(season)
                    sp_fip = calculate_fip(hr, bb, k, ip, fip_constant=fip_constant or 3.1)
            except Exception:
                pass

            rows.append({
                "Date": game_date,
                "Opp": opp_abbr,
                "Result": result,
                "Score": f"{team_score}-{opp_score}",
                "Opp W%": round(_record_pct(opp_perf), 3),
                "Opp R/G": round(opp_rpg, 2),
                "Opp RA/G": round(opp_rapg, 2),
                "Opp Quality Score": opp_quality_score,
                "SP Faced": sp_name,
                "SP FIP": sp_fip,
            })
            if len(rows) >= max_games:
                break
        if len(rows) >= max_games:
            break

    detail = pd.DataFrame(rows, columns=detail_columns)
    if detail.empty:
        return pd.DataFrame(columns=columns), detail
    wins = (detail["Result"] == "W").sum()
    losses = (detail["Result"] == "L").sum()
    team_runs = pd.to_numeric(detail["Score"].astype(str).str.split("-", expand=True)[0], errors="coerce")
    opp_runs_allowed = pd.to_numeric(detail["Score"].astype(str).str.split("-", expand=True)[1], errors="coerce")
    opp_wins = opp_losses = 0
    for _, row in detail.iterrows():
        pct = _safe_number(row.get("Opp W%"), 0.5)
        opp_wins += pct
        opp_losses += (1 - pct)
    star_sp = detail[pd.to_numeric(detail["SP FIP"], errors="coerce").fillna(99) <= 3.75]
    avg_opp_w = pd.to_numeric(detail["Opp W%"], errors="coerce").mean()
    avg_opp_r = pd.to_numeric(detail["Opp R/G"], errors="coerce").mean()
    avg_opp_ra = pd.to_numeric(detail["Opp RA/G"], errors="coerce").mean()
    avg_quality_score = pd.to_numeric(detail["Opp Quality Score"], errors="coerce").mean()
    opp_quality = _opponent_quality_label(avg_quality_score)
    summary = pd.DataFrame([{
        "Team": team_abbr,
        "Recent": f"{wins}-{losses}",
        "Runs/G": round(team_runs.mean(), 2),
        "Allowed/G": round(opp_runs_allowed.mean(), 2),
        "Opp Quality": opp_quality,
        "Opp Quality Score": round(avg_quality_score, 1),
        "Opp Combined": f"{opp_wins:.1f}-{opp_losses:.1f}",
        "Avg Opp W%": round(avg_opp_w, 3),
        "Avg Opp R/G": round(avg_opp_r, 2),
        "Avg Opp RA/G": round(avg_opp_ra, 2),
        "Star SP Faced": int(len(star_sp)),
    }], columns=columns)
    return summary, detail


def get_team_rolling_form(team_id, team_abbr, season, as_of_date, windows=(7, 14, 30)):
    columns = ["Team", "Window", "Games", "Record", "Runs/G", "Allowed/G", "Run Diff/G", "Opp Quality", "Opp Quality Score", "Avg Opp W%"]
    end_date = datetime.strptime(as_of_date, "%Y-%m-%d")
    start_date = (end_date - timedelta(days=max(windows) + 2)).strftime("%Y-%m-%d")
    url = (
        f"https://statsapi.mlb.com/api/v1/schedule?"
        f"sportId=1&teamId={team_id}&startDate={start_date}&endDate={as_of_date}&sort=desc&hydrate=team,linescore"
    )
    data = cached_json_request(url, namespace="statsapi")
    games = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            if game.get("status", {}).get("detailedState") != "Final":
                continue
            game_date = datetime.strptime(game.get("officialDate"), "%Y-%m-%d")
            is_home = game["teams"]["home"]["team"]["id"] == team_id
            team_side = "home" if is_home else "away"
            opp_side = "away" if is_home else "home"
            team_score = game["teams"][team_side].get("score", 0)
            opp_score = game["teams"][opp_side].get("score", 0)
            opp_id = game["teams"][opp_side]["team"]["id"]
            context_day = _pregame_end_date(game.get("officialDate")) or as_of_date
            opp_perf = get_team_performance(opp_id, season=season, as_of_date=context_day)
            opp_rpg, opp_rapg = _team_runs_per_game(opp_id, season, context_day)
            opp_w_pct = _record_pct(opp_perf)
            games.append({
                "Date": game_date,
                "Runs": team_score,
                "Allowed": opp_score,
                "Win": int(team_score > opp_score),
                "Opp W%": opp_w_pct,
                "Opp Quality Score": _opponent_quality_score(opp_w_pct, opp_rpg, opp_rapg),
            })
    rows = []
    for window in windows:
        cutoff = end_date - timedelta(days=window)
        subset = [game for game in games if cutoff <= game["Date"] <= end_date]
        if not subset:
            rows.append({"Team": team_abbr, "Window": f"{window}D", "Games": 0, "Record": "0-0", "Runs/G": 0.0, "Allowed/G": 0.0, "Run Diff/G": 0.0, "Opp Quality": "", "Opp Quality Score": 0.0, "Avg Opp W%": 0.0})
            continue
        wins = sum(game["Win"] for game in subset)
        losses = len(subset) - wins
        runs = np.mean([game["Runs"] for game in subset])
        allowed = np.mean([game["Allowed"] for game in subset])
        quality_score = float(np.mean([game["Opp Quality Score"] for game in subset]))
        opp_w_pct = float(np.mean([game["Opp W%"] for game in subset]))
        rows.append({
            "Team": team_abbr,
            "Window": f"{window}D",
            "Games": len(subset),
            "Record": f"{wins}-{losses}",
            "Runs/G": round(runs, 2),
            "Allowed/G": round(allowed, 2),
            "Run Diff/G": round(runs - allowed, 2),
            "Opp Quality": _opponent_quality_label(quality_score),
            "Opp Quality Score": round(quality_score, 1),
            "Avg Opp W%": round(opp_w_pct, 3),
        })
    return pd.DataFrame(rows, columns=columns)


def summarize_bullpen_form(team, bullpen_df):
    columns = ["Team", "R/L", "Avail R/L", "Available", "Monitor", "Taxed",
               "Top3 Score", "Last3D Pitches", "Status"]
    if bullpen_df is None or bullpen_df.empty:
        return pd.DataFrame(columns=columns)
    availability = bullpen_df.get("Availability", pd.Series(dtype=str)).astype(str)
    available = int(availability.eq("Available").sum())
    monitor = int(availability.eq("Monitor").sum())
    taxed = int(availability.eq("Taxed").sum())
    # Handedness matters twice over: how the pen is built, and how much of it is usable
    # tonight -- a pen with two lefties is a different problem when both are taxed.
    hands = bullpen_df.get("Throws", pd.Series(dtype=str)).map(_pen_hand)
    ready = hands[availability.eq("Available").values] if len(hands) == len(availability) else hands
    hand_mix = f"{int((hands == 'R').sum())}R/{int((hands == 'L').sum())}L"
    ready_mix = f"{int((ready == 'R').sum())}R/{int((ready == 'L').sum())}L"
    top3_score = _top_mean_numeric(bullpen_df, "Bullpen Score", n=3, default=0.0)
    last3 = int(_numeric_col(bullpen_df, "Last3D_Pitches").sum())
    if taxed >= 2 or monitor >= 4:
        status = "Risk"
    elif top3_score >= 28 and available >= 3:
        status = "Advantage"
    else:
        status = "Neutral"
    return pd.DataFrame([{
        "Team": team,
        "R/L": hand_mix,
        "Avail R/L": ready_mix,
        "Available": available,
        "Monitor": monitor,
        "Taxed": taxed,
        "Top3 Score": round(top3_score, 1),
        "Last3D Pitches": last3,
        "Status": status,
    }], columns=columns)


def _lineup_confidence(lineup_df):
    if lineup_df is None or lineup_df.empty or "Lineup Confidence" not in lineup_df.columns:
        return "Unknown"
    values = lineup_df["Lineup Confidence"].dropna().astype(str)
    return values.mode().iloc[0] if not values.empty else "Unknown"


def generate_uncertainty_flags(scorecard, away_team, home_team, away_lineup_df, home_lineup_df, away_type_results, home_type_results, away_bullpen_df, home_bullpen_df, projection_summary):
    flags = []
    model_label = str((projection_summary or {}).get("Model", ""))
    if "low" in model_label:
        flags.append("Model calibration is low reliability; treat win lean as directional.")
    elif "medium" in model_label:
        flags.append("Model calibration is medium reliability; use win leans directionally and totals as ranges.")
    if "umpire experimental" in model_label:
        flags.append("Umpire tendency is included experimentally but has not shown holdout value by magnitude.")

    if scorecard is not None and not scorecard.empty and "Win Lean" in scorecard.columns:
        win_pcts = pd.to_numeric(scorecard["Win Lean"].astype(str).str.replace("%", "", regex=False), errors="coerce").dropna()
        if not win_pcts.empty and win_pcts.max() < 57:
            flags.append("Game edge is narrow; avoid over-reading the projected winner.")
        elif not win_pcts.empty and win_pcts.max() >= 58 and "win magnitude directional" in model_label:
            flags.append(
                "Large win-edge magnitude has not validated monotonically; treat the favorite as a lean, especially early season."
            )

    for team, lineup in [(away_team, away_lineup_df), (home_team, home_lineup_df)]:
        confidence = _lineup_confidence(lineup)
        if confidence != "Confirmed":
            flags.append(f"{team} lineup confidence is {confidence}; hitter-specific edges may move after confirmed lineups.")

    for team, type_df in [(away_team, away_type_results), (home_team, home_type_results)]:
        if type_df is None or type_df.empty:
            flags.append(f"{team} pitcher-type history unavailable or thin.")
            continue
        row = type_df.iloc[0]
        games = _safe_number(row.get("Games"), 0)
        consistency = str(row.get("Consistency", ""))
        if games < 8:
            flags.append(f"{team} pitcher-type sample is thin ({int(games)} games).")
        if "volatile" in consistency.lower():
            flags.append(f"{team} results vs this pitcher type have been volatile.")

    for team, bullpen in [(away_team, away_bullpen_df), (home_team, home_bullpen_df)]:
        if bullpen is None or bullpen.empty or "Availability" not in bullpen.columns:
            continue
        availability = bullpen["Availability"].astype(str)
        taxed_count = int(availability.isin(["Taxed", "Monitor"]).sum())
        if taxed_count >= 3:
            flags.append(f"{team} bullpen has {taxed_count} relievers to monitor for workload.")

    return flags[:6]


def get_pitcher_opponent_quality(pitcher_id, season, as_of_date, max_games=8):
    columns = ["Pitcher", "Games", "Opp Avg W%", "Opp Avg R/G", "Opp Avg OPS", "Opp Quality Score", "Quality Faced"]
    start_date = _season_start_date(season)
    pitcher_df = cached_dataframe_call("statcast_pitcher", statcast_pitcher, start_date, as_of_date, pitcher_id)
    if pitcher_df is None or pitcher_df.empty or "game_pk" not in pitcher_df.columns:
        return pd.DataFrame(columns=columns)
    rows = []
    sort_df = pitcher_df.copy()
    if "game_date" in sort_df.columns:
        sort_df["game_date_sort"] = pd.to_datetime(sort_df["game_date"], errors="coerce")
        game_order = sort_df.groupby("game_pk")["game_date_sort"].max().sort_values(ascending=False).index.tolist()
    else:
        game_order = list(sort_df["game_pk"].dropna().unique())
    for game_pk in game_order:
        if len(rows) >= max_games:
            break
        game_df = pitcher_df[pitcher_df["game_pk"] == game_pk]
        try:
            game_date = str(game_df["game_date"].dropna().iloc[0])[:10] if "game_date" in game_df.columns and not game_df["game_date"].dropna().empty else as_of_date
            context_day = _pregame_end_date(game_date) or as_of_date
            batting_team = ""
            if {"home_team", "away_team", "inning_topbot"}.issubset(game_df.columns):
                sample = game_df.iloc[0]
                batting_team = sample["away_team"] if str(sample.get("inning_topbot")).lower() == "top" else sample["home_team"]
            team_id = get_team_id(batting_team) if batting_team else None
            perf = get_team_performance(team_id, season=season, as_of_date=context_day) if team_id else {}
            rpg, _ = _team_runs_per_game(team_id, season, context_day) if team_id else (4.4, 4.4)
            line = _events_to_batting_line(game_df)
            opp_w = _record_pct(perf)
            rows.append({"Opp W%": opp_w, "Opp R/G": rpg, "Opp OPS": line["OPS_proxy"], "Opp Quality Score": _opponent_quality_score(opp_w, rpg, opp_ops=line["OPS_proxy"])})
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows)
    opp_w = _mean_numeric(df, "Opp W%", 0.5)
    opp_r = _mean_numeric(df, "Opp R/G", 4.4)
    opp_ops = _mean_numeric(df, "Opp OPS", 0.700)
    quality_score = _mean_numeric(df, "Opp Quality Score", 50)
    quality = _opponent_quality_label(quality_score)
    return pd.DataFrame([{
        "Pitcher": _player_name_from_id(pitcher_id),
        "Games": len(rows),
        "Opp Avg W%": round(opp_w, 3),
        "Opp Avg R/G": round(opp_r, 2),
        "Opp Avg OPS": round(opp_ops, 3),
        "Opp Quality Score": round(quality_score, 1),
        "Quality Faced": quality,
    }], columns=columns)


def generate_pitcher_batter_hand_splits(pitcher_id, season, as_of_date):
    columns = ["Pitcher", "Batter Side", "PA", "OPS", "SLG", "xwOBA", "K%", "HardHit%", "Split Tag"]
    if not pitcher_id:
        return pd.DataFrame(columns=columns)
    start_date = _season_start_date(season)
    if as_of_date < start_date:
        return pd.DataFrame(columns=columns)
    df = cached_dataframe_call("statcast_pitcher", statcast_pitcher, start_date, as_of_date, pitcher_id)
    if df is None or df.empty or "stand" not in df.columns:
        return pd.DataFrame(columns=columns)
    rows = []
    pitcher_name = _player_name_from_id(pitcher_id)
    for side, group in df.groupby("stand"):
        side = str(side).strip()
        if side not in {"L", "R"}:
            continue
        line = _events_to_batting_line(group)
        pa = line["PA"]
        quality = _contact_quality(group)
        k_rate = round((line["K"] / pa) * 100, 1) if pa else 0.0
        if pa >= 25 and (line["OPS_proxy"] >= 0.820 or quality["xwOBA"] >= 0.360):
            tag = "Vulnerable"
        elif pa >= 25 and (line["OPS_proxy"] <= 0.650 or quality["xwOBA"] <= 0.290 or k_rate >= 28):
            tag = "Suppresses"
        elif pa >= 15:
            tag = "Neutral"
        else:
            tag = "Small sample"
        rows.append({
            "Pitcher": pitcher_name,
            "Batter Side": side,
            "PA": pa,
            "OPS": line["OPS_proxy"],
            "SLG": line["SLG"],
            "xwOBA": quality["xwOBA"],
            "K%": k_rate,
            "HardHit%": quality["HardHit%"],
            "Split Tag": tag,
        })
    return pd.DataFrame(rows, columns=columns).sort_values(["Batter Side"]) if rows else pd.DataFrame(columns=columns)


def _edge_tag(rv100, xwoba, pitches, min_pitches):
    if pitches < min_pitches:
        return "small"
    if rv100 >= 0.7 or xwoba >= 0.350:
        return "hitter"
    if rv100 <= -0.7 or xwoba <= 0.290:
        return "pitcher"
    return "even"


def _pitch_rv_swing_metrics(sub):
    """Batter-perspective RV/100, whiff/chase/contact %, xwOBA for a set of pitches."""
    m = _swing_metrics(sub)
    # delta_run_exp is batter-positive (whiff < 0, HR > 0), so batter run value is +dRE.
    rv = pd.to_numeric(sub.get("delta_run_exp"), errors="coerce")
    rv100 = round(float(rv.mean()) * 100, 2) if rv.notna().any() else 0.0
    whiff = round(m["whiffs"] / m["swings"] * 100, 1) if m["swings"] else 0.0
    chase = round(m["chases"] / m["ooz"] * 100, 1) if m["ooz"] else 0.0
    contact = round(m["contact"] / m["swings"] * 100, 1) if m["swings"] else 0.0
    xw = pd.to_numeric(sub.get("estimated_woba_using_speedangle"), errors="coerce").dropna()
    xwoba = round(float(xw.mean()), 3) if len(xw) else 0.0
    return {"pitches": m["pitches"], "RV/100": rv100, "Whiff%": whiff, "Chase%": chase, "Contact%": contact, "xwOBA": xwoba}


def build_arsenal_pitch_matchup(team_abbr, lineup_df, starter_arsenal_df, statcast_pitches_df, top_n=5, min_pitches=60):
    """Per-pitch-type value + swing decisions for a lineup vs the starter's arsenal."""
    columns = ["Pitch", "Usage%", "Pitches", "RV/100", "Whiff%", "Chase%", "Contact%", "xwOBA", "Edge"]
    if (
        lineup_df is None or lineup_df.empty or "ID" not in lineup_df.columns
        or starter_arsenal_df is None or starter_arsenal_df.empty
        or statcast_pitches_df is None or statcast_pitches_df.empty or "batter" not in statcast_pitches_df.columns
    ):
        return pd.DataFrame(columns=columns)
    batter_ids = pd.to_numeric(lineup_df["ID"], errors="coerce").dropna().astype(int).tolist()
    if not batter_ids:
        return pd.DataFrame(columns=columns)
    lineup_pitches = statcast_pitches_df[statcast_pitches_df["batter"].isin(batter_ids)]
    if lineup_pitches.empty or "pitch_type" not in lineup_pitches.columns:
        return pd.DataFrame(columns=columns)
    ars = starter_arsenal_df.copy()
    ars["Usage %"] = _numeric_col(ars, "Usage %")
    ars = ars.sort_values("Usage %", ascending=False).head(top_n)
    rows = []
    for _, a in ars.iterrows():
        code = str(a.get("Pitch", "")).strip()
        if not code:
            continue
        sub = lineup_pitches[lineup_pitches["pitch_type"].astype(str) == code]
        metrics = _pitch_rv_swing_metrics(sub)
        rows.append({
            "Pitch": code,
            "Usage%": round(float(a.get("Usage %") or 0), 1),
            "Pitches": metrics["pitches"],
            "RV/100": metrics["RV/100"],
            "Whiff%": metrics["Whiff%"],
            "Chase%": metrics["Chase%"],
            "Contact%": metrics["Contact%"],
            "xwOBA": metrics["xwOBA"],
            "Edge": _edge_tag(metrics["RV/100"], metrics["xwOBA"], metrics["pitches"], min_pitches),
        })
    return pd.DataFrame(rows, columns=columns)


def build_batter_arsenal_swing_drilldown(team_abbr, lineup_df, starter_arsenal_df, statcast_pitches_df, top_n=5, min_type_pitches=5):
    """Per-hitter usage-weighted RV/whiff/chase/contact/xwOBA vs the starter's arsenal mix."""
    columns = ["Name", "Pitches", "RV/100", "Whiff%", "Chase%", "Contact%", "xwOBA", "Edge"]
    if (
        lineup_df is None or lineup_df.empty or {"ID", "Name"}.issubset(lineup_df.columns) is False
        or starter_arsenal_df is None or starter_arsenal_df.empty
        or statcast_pitches_df is None or statcast_pitches_df.empty or "batter" not in statcast_pitches_df.columns
    ):
        return pd.DataFrame(columns=columns)
    weights = _arsenal_pitch_weights(starter_arsenal_df, top_n=top_n)
    if not weights:
        return pd.DataFrame(columns=columns)
    codes = [c for c, _ in weights]
    team_pitches = statcast_pitches_df[statcast_pitches_df["pitch_type"].astype(str).isin(codes)]
    rows = []
    for _, lp in lineup_df.iterrows():
        bid = pd.to_numeric(lp.get("ID"), errors="coerce")
        if pd.isna(bid):
            continue
        bp = team_pitches[team_pitches["batter"] == int(bid)]
        num = {"RV/100": 0.0, "Whiff%": 0.0, "Chase%": 0.0, "Contact%": 0.0, "xwOBA": 0.0}
        wsum = {k: 0.0 for k in num}
        total_pitches = 0
        for code, w in weights:
            sub = bp[bp["pitch_type"].astype(str) == code]
            metrics = _pitch_rv_swing_metrics(sub)
            n = metrics["pitches"]
            if n < min_type_pitches:
                continue
            total_pitches += n
            sm = _swing_metrics(sub)
            if pd.to_numeric(sub.get("delta_run_exp"), errors="coerce").notna().any():
                num["RV/100"] += w * metrics["RV/100"]; wsum["RV/100"] += w
            if sm["swings"]:
                num["Whiff%"] += w * metrics["Whiff%"]; wsum["Whiff%"] += w
                num["Contact%"] += w * metrics["Contact%"]; wsum["Contact%"] += w
            if sm["ooz"]:
                num["Chase%"] += w * metrics["Chase%"]; wsum["Chase%"] += w
            if metrics["xwOBA"]:
                num["xwOBA"] += w * metrics["xwOBA"]; wsum["xwOBA"] += w
        rv100 = round(num["RV/100"] / wsum["RV/100"], 2) if wsum["RV/100"] else 0.0
        xwoba = round(num["xwOBA"] / wsum["xwOBA"], 3) if wsum["xwOBA"] else 0.0
        rows.append({
            "Name": lp.get("Name"),
            "Pitches": total_pitches,
            "RV/100": rv100,
            "Whiff%": round(num["Whiff%"] / wsum["Whiff%"], 1) if wsum["Whiff%"] else 0.0,
            "Chase%": round(num["Chase%"] / wsum["Chase%"], 1) if wsum["Chase%"] else 0.0,
            "Contact%": round(num["Contact%"] / wsum["Contact%"], 1) if wsum["Contact%"] else 0.0,
            "xwOBA": xwoba,
            "Edge": _edge_tag(rv100, xwoba, total_pitches, 30),
        })
    out = pd.DataFrame(rows, columns=columns)
    if out.empty:
        return out
    return out.sort_values("RV/100", ascending=False).reset_index(drop=True)


def opponent_vs_pitcher_arsenal(team_abbr, lineup_df, starter_arsenal_df, starter_hand, statcast_df):
    columns = ["Team", "Basis", "PA", "OPS", "xwOBA", "SLG", "HR", "K%", "HardHit%", "Whiff%", "League OPS Pctl", "Arsenal Tag"]
    if statcast_df is None or statcast_df.empty or starter_arsenal_df is None or starter_arsenal_df.empty:
        return pd.DataFrame(columns=columns)
    pitch_weights = _arsenal_pitch_weights(starter_arsenal_df, top_n=3)
    top_pitches = [pitch for pitch, _ in pitch_weights]
    if not pitch_weights:
        return pd.DataFrame(columns=columns)
    team_df = _statcast_team_rows(statcast_df, team_abbr)
    if team_df.empty:
        return pd.DataFrame(columns=columns)
    if starter_hand and "p_throws" in team_df.columns:
        team_df = team_df[team_df["p_throws"] == starter_hand]
    team_df = _filter_to_arsenal_shape(team_df, starter_arsenal_df, top_pitches, min_pa=75)
    metrics = _weighted_arsenal_metrics(team_df, pitch_weights)

    league_df = statcast_df.copy()
    if starter_hand and "p_throws" in league_df.columns:
        league_df = league_df[league_df["p_throws"] == starter_hand]
    league_df = _filter_to_arsenal_shape(league_df, starter_arsenal_df, top_pitches, min_pa=400)
    league_ops = []
    if not league_df.empty and {"home_team", "away_team", "inning_topbot"}.issubset(league_df.columns):
        batting_team = np.where(league_df["inning_topbot"].astype(str).str.lower().eq("bot"), league_df["home_team"], league_df["away_team"])
        league_df["batting_team"] = batting_team
        for _, group in league_df.groupby("batting_team"):
            group_metrics = _weighted_arsenal_metrics(group, pitch_weights)
            if group_metrics["PA"] >= 50:
                league_ops.append(group_metrics["OPS"])
    pct = _percentile(metrics["OPS"], pd.Series(league_ops)) if league_ops else None
    if (pct is not None and pct >= 70) or metrics["xwOBA"] >= 0.350:
        tag = "Advantage"
    elif (pct is not None and pct <= 30) or metrics["xwOBA"] <= 0.290 or metrics["K%"] >= 27:
        tag = "Risk"
    else:
        tag = "Neutral"
    return pd.DataFrame([{
        "Team": team_abbr,
        "Basis": _format_pitch_weight_basis(starter_hand, pitch_weights, starter_arsenal_df),
        "PA": metrics["PA"],
        "OPS": metrics["OPS"],
        "xwOBA": metrics["xwOBA"],
        "SLG": metrics["SLG"],
        "HR": metrics["HR"],
        "K%": metrics["K%"],
        "HardHit%": metrics["HardHit%"],
        "Whiff%": metrics["Whiff%"],
        "League OPS Pctl": _percentile_badge(pct),
        "Arsenal Tag": tag,
    }], columns=columns)


_TEAM_PARK_FACTOR_CACHE = {}


def _canon_team(abbr):
    """Normalize statcast/statsapi team abbreviations to a single canonical form."""
    aliases = {"AZ": "ARI", "ATH": "OAK"}
    value = str(abbr or "").upper().strip()
    return aliases.get(value, value)


def _team_home_park_run_factor(team_abbr):
    """Return a team's home-park run factor (1.0 = neutral) using PARK_CONTEXT."""
    if not team_abbr:
        return 1.0
    canon = _canon_team(team_abbr)
    if not _TEAM_PARK_FACTOR_CACHE:
        try:
            data = statsapi.get("teams", {"sportIds": 1})
            for team in data.get("teams", []):
                abbr = _canon_team(team.get("abbreviation", ""))
                venue_name = (team.get("venue", {}) or {}).get("name", "")
                if abbr and venue_name:
                    park, _ = _park_context_for_venue(venue_name)
                    _TEAM_PARK_FACTOR_CACHE[abbr] = float(park.get("Runs", 1.0) or 1.0)
        except Exception:
            _TEAM_PARK_FACTOR_CACHE["__failed__"] = 1.0
    return _TEAM_PARK_FACTOR_CACHE.get(canon, 1.0)


def _park_adjust(value, factor, strength=1.0):
    """Divide a rate stat by its park factor. strength<1 dampens the adjustment (used for xwOBA)."""
    val = _safe_number(value, 0.0) or 0.0
    fac = _safe_number(factor, 1.0) or 1.0
    eff = 1.0 + (fac - 1.0) * strength
    if eff <= 0:
        return round(val, 3)
    return round(val / eff, 3)


def _split_metrics(events_df):
    """One split's line: rate stats for a subset of statcast PA-end rows.

    Everything here already fell out of `_events_to_batting_line` and `_contact_quality`
    and was simply being discarded -- the two summary tables kept PA/OPS/xwOBA and threw
    the rest away.

    K% earns its place ahead of the others: it is the largest term in a starter's DK line
    (docs/arsenal_study.md) and it sets a hitter's floor, since a strikeout is the one
    outcome that can produce nothing downstream. ISO separates the lineups that get their
    OPS from power -- which is what pays on DK, at 10 points a home run -- from the ones
    that get it from singles and walks.

    Whiff% is deliberately *not* included. These frames are PA-ending pitches only, so a
    whiff rate computed over them would be whiffs per swing on final pitches, which is not
    the whiff rate anyone reads it as.
    """
    line = _events_to_batting_line(events_df)
    quality = _contact_quality(events_df)
    pa = line["PA"]
    return {
        "PA": pa,
        "OPS": line["OPS_proxy"],
        "xwOBA": quality["xwOBA"],
        "ISO": round(max(0.0, line["SLG"] - line["AVG"]), 3),
        "K%": round(line["K"] / pa * 100, 1) if pa else 0.0,
        "BB%": round(line["BB"] / pa * 100, 1) if pa else 0.0,
        "HardHit%": quality["HardHit%"],
        # A rate, not the count: these splits carry wildly different plate-appearance
        # totals, and "17 HR vs LHP" against "58 vs RHP" says more about who they faced
        # than about the lineup.
        "HR%": round(line["HR"] / pa * 100, 1) if pa else 0.0,
    }


# Columns both summary tables carry beyond the split label, in display order.
SPLIT_STAT_COLUMNS = ["PA", "OPS", "xwOBA", "ISO", "K%", "BB%", "HardHit%", "HR%"]


def _park_adjusted_split(events_df, factor):
    """A split with the park factor applied to the rate stats it should touch.

    Park adjustment belongs on run-scoring rates, not on strikeout and walk rates: a park
    moves how far the ball carries, not whether the hitter made contact. So OPS, xwOBA and
    ISO are adjusted and K%/BB%/HardHit% pass through untouched.
    """
    metrics = _split_metrics(events_df)
    metrics["OPS"] = _park_adjust(metrics["OPS"], factor)
    metrics["xwOBA"] = _park_adjust(metrics["xwOBA"], factor, 0.5)
    metrics["ISO"] = _park_adjust(metrics["ISO"], factor, 0.5)
    return metrics


def _keep_venue_split(df, keep_label):
    """Keep only the venue-relevant Home/Away split row (a team is one or the other this game)."""
    if df is None or df.empty or "Split" not in df.columns:
        return df
    drop = {"Home (PF)", "Away (PF)"} - {keep_label}
    return df[~df["Split"].isin(drop)].reset_index(drop=True)


def build_lineup_split_summary(team_abbr, lineup_df, statcast_detail_df, starter_hand, vs_type_row=None, context_end=None):
    """Aggregate lineup rate line sliced by Overall, Home/Away (park-adj), vs L/R, vs Type, L28."""
    columns = ["Team", "Split"] + SPLIT_STAT_COLUMNS
    if (
        lineup_df is None or lineup_df.empty or "ID" not in lineup_df.columns
        or statcast_detail_df is None or statcast_detail_df.empty
    ):
        return pd.DataFrame(columns=columns)
    batter_ids = pd.to_numeric(lineup_df["ID"], errors="coerce").dropna().astype(int).tolist()
    if not batter_ids:
        return pd.DataFrame(columns=columns)
    team_rows = _statcast_team_rows(statcast_detail_df, team_abbr)
    if team_rows.empty or "batter" not in team_rows.columns:
        return pd.DataFrame(columns=columns)
    df = team_rows[team_rows["batter"].isin(batter_ids)].copy()
    if df.empty:
        return pd.DataFrame(columns=columns)

    canon = _canon_team(team_abbr)
    rows = []

    rows.append({"Split": "Overall", **_split_metrics(df)})

    if "home_team" in df.columns:
        home_mask = df["home_team"].astype(str).map(_canon_team) == canon
        home_df = df[home_mask]
        away_df = df[~home_mask]
        home_factor = _team_home_park_run_factor(canon)
        rows.append({"Split": "Home (PF)", **_park_adjusted_split(home_df, home_factor)})
        away_factor = float(away_df["home_team"].astype(str).map(_team_home_park_run_factor).mean()) if not away_df.empty else 1.0
        rows.append({"Split": "Away (PF)", **_park_adjusted_split(away_df, away_factor)})

    if "p_throws" in df.columns:
        for hand, label in [("L", "vs LHP"), ("R", "vs RHP")]:
            sub = df[df["p_throws"].astype(str).str.strip() == hand]
            split_label = f"{label} *" if starter_hand and starter_hand == hand else label
            rows.append({"Split": split_label, **_split_metrics(sub)})

    if vs_type_row is not None and not vs_type_row.empty:
        # Carried across from the vs-arsenal table, which computes its own line. It reports
        # HR as a count, so the rate is derived here; ISO and BB% it does not carry at all
        # and are left blank rather than filled with a zero that would read as a real value.
        r = vs_type_row.iloc[0]
        type_pa = int(_safe_number(r.get("PA"), 0) or 0)
        type_hr = _safe_number(r.get("HR"), None)
        rows.append({
            "Split": "vs Type",
            "PA": type_pa,
            "OPS": round(_safe_number(r.get("OPS"), 0.0) or 0.0, 3),
            "xwOBA": round(_safe_number(r.get("xwOBA"), 0.0) or 0.0, 3),
            "K%": _safe_number(r.get("K%"), None),
            "HardHit%": _safe_number(r.get("HardHit%"), None),
            "HR%": round(type_hr / type_pa * 100, 1) if type_hr is not None and type_pa else None,
        })

    if "game_date" in df.columns and context_end:
        try:
            cutoff = (datetime.strptime(context_end, "%Y-%m-%d") - timedelta(days=28)).strftime("%Y-%m-%d")
            recent = df[df["game_date"].astype(str) >= cutoff]
            rows.append({"Split": "L28", **_split_metrics(recent)})
        except Exception:
            pass

    out = pd.DataFrame(rows)
    for column in SPLIT_STAT_COLUMNS:
        if column not in out.columns:
            out[column] = None
    out.insert(0, "Team", team_abbr)
    return out[columns]


_OPENER_INDEX_CACHE = {}
OPENER_MAX_INNINGS = 2.0      # an opener works the 1st and maybe the 2nd
BULK_MIN_INNINGS = 3.0        # the arm behind him has to actually carry the game

# Share of opener games in which the arm who most often followed before follows again.
# Measured over 371 opener halves in 2024-26: 28% overall, and still only ~34% restricted
# to openers with 2+ prior games or a 60%+ historical share. The follower is therefore a
# scouting note, not a projection -- see research/arsenal/opener_audit.py.
OPENER_BULK_REPEAT_RATE = 0.33

# Outs each terminal event records. Innings pitched used to be approximated by counting the
# distinct innings a pitcher appeared in, which is wrong in both directions: an arm who goes
# 2.1 shows up in three innings and stops looking like an opener, while one who records a
# single out either side of an inning break looks like he went two. Checked against 56,223
# box-score lines: distinct innings is exact 60% of the time and off by 0.24 IP on average;
# counting outs is exact 92% of the time and off by 0.03. On the "<= 2 IP" test that decides
# an opener, distinct innings misses 313 real short outings and outs-based misses 15.
_EVENT_OUTS = {
    "strikeout": 1, "field_out": 1, "force_out": 1, "sac_fly": 1, "sac_bunt": 1,
    "fielders_choice_out": 1, "other_out": 1, "caught_stealing_2b": 1,
    "caught_stealing_3b": 1, "caught_stealing_home": 1, "pickoff_1b": 1,
    "pickoff_2b": 1, "pickoff_3b": 1, "pickoff_caught_stealing_2b": 1,
    "pickoff_caught_stealing_3b": 1, "pickoff_caught_stealing_home": 1,
    "strikeout_double_play": 2, "grounded_into_double_play": 2, "double_play": 2,
    "sac_fly_double_play": 2, "sac_bunt_double_play": 2, "triple_play": 3,
}


def build_opener_index(season, context_end, statcast_pitches_df=None):
    """Every opener-shaped half-game this season, straight off the pitch-level data.

    A half-game's first pitcher who covers <= OPENER_MAX_INNINGS and hands off to someone
    who covers >= BULK_MIN_INNINGS is an opener/bulk pair. Derived from statcast rather
    than boxscore fetches, so scanning the whole league costs one already-cached pull.
    Returns a frame: game_pk, half, first, first_inn, bulk, bulk_inn, is_opener.

    Two things here are load-bearing and were previously got wrong:

    * **Order comes from `at_bat_number`, not from the inning.** Two pitchers can work the
      same inning, and sorting by inning alone left that tie to be broken by whatever order
      the rows happened to sit in -- with a non-stable sort. That named the wrong follower
      in 3.4% of half-games and the wrong starter in 0.3%.
    * **Exhibition games are excluded.** The season pull starts March 1, so without the
      filter roughly 7% of the index was spring training, where every arm is on a two-inning
      leash and nothing about the pattern is real.
    """
    key = (int(season), str(context_end))
    if key in _OPENER_INDEX_CACHE:
        return _OPENER_INDEX_CACHE[key]
    df = statcast_pitches_df
    if df is None or df.empty:
        try:
            df = load_statcast_pitches(season=season, end_date=context_end)
        except Exception as e:
            print(f"⚠️ Opener detection unavailable: {e}")
            return pd.DataFrame()
    needed = {"game_pk", "inning", "pitcher", "inning_topbot", "at_bat_number", "events"}
    if df is None or df.empty or not needed.issubset(df.columns):
        return pd.DataFrame()

    slim = df[list(needed) + (["game_type"] if "game_type" in df.columns else [])].copy()
    if "game_type" in slim.columns:
        slim = slim[slim["game_type"].astype("string").eq("R")]
    slim = slim.dropna(subset=["game_pk", "inning", "pitcher", "inning_topbot",
                               "at_bat_number"])
    if slim.empty:
        return pd.DataFrame()

    slim["outs"] = slim["events"].astype("string").map(_EVENT_OUTS).fillna(0.0)
    # One row per (half-game, pitcher): when he first appeared, and how many outs he got.
    per_pitcher = slim.groupby(["game_pk", "inning_topbot", "pitcher"], observed=True).agg(
        first_ab=("at_bat_number", "min"), outs=("outs", "sum")).reset_index()
    per_pitcher["ip"] = per_pitcher["outs"] / 3.0
    per_pitcher = per_pitcher.sort_values(["game_pk", "inning_topbot", "first_ab"],
                                          kind="mergesort")
    per_pitcher["slot"] = per_pitcher.groupby(["game_pk", "inning_topbot"],
                                              observed=True).cumcount()

    starters = per_pitcher[per_pitcher["slot"] == 0].rename(
        columns={"pitcher": "first", "ip": "first_ip"})
    followers = per_pitcher[per_pitcher["slot"] == 1].rename(
        columns={"pitcher": "bulk", "ip": "bulk_ip"})
    index = starters[["game_pk", "inning_topbot", "first", "first_ip"]].merge(
        followers[["game_pk", "inning_topbot", "bulk", "bulk_ip"]],
        on=["game_pk", "inning_topbot"], how="left")

    index["bulk_ip"] = index["bulk_ip"].fillna(0.0)
    index["is_opener"] = (index["first_ip"] <= OPENER_MAX_INNINGS) & (
        index["bulk_ip"] >= BULK_MIN_INNINGS)
    index = index.rename(columns={"inning_topbot": "half",
                                  "first_ip": "first_inn", "bulk_ip": "bulk_inn"})
    index["game_pk"] = index["game_pk"].astype(int)
    index["first"] = index["first"].astype(int)
    _OPENER_INDEX_CACHE[key] = index
    return index


def _pitcher_season_line(pitcher_id, season):
    """Light season pitching line for any arm (used for the bulk guy behind an opener)."""
    if not pitcher_id:
        return {}
    try:
        data = cached_json_request(
            f"https://statsapi.mlb.com/api/v1/people/{int(pitcher_id)}/stats",
            params={"stats": "season", "group": "pitching", "season": int(season)},
            namespace="statsapi",
        )
    except Exception:
        return {}
    splits = (data.get("stats", [{}])[0] or {}).get("splits", [])
    if not splits:
        return {}
    stat = splits[-1].get("stat", {}) or {}
    batters = _safe_number(stat.get("battersFaced"), 0)
    return {
        "G": stat.get("gamesPlayed", ""), "GS": stat.get("gamesStarted", ""),
        "IP": stat.get("inningsPitched", ""), "ERA": stat.get("era", ""),
        "WHIP": stat.get("whip", ""),
        "K%": round(_safe_number(stat.get("strikeOuts"), 0) / batters * 100, 1) if batters else "",
        "BB%": round(_safe_number(stat.get("baseOnBalls"), 0) / batters * 100, 1) if batters else "",
    }


def build_opener_profile(pitcher_id, season, context_end, statcast_pitches_df=None, min_starts=2):
    """Is tonight's listed starter actually an opener, and who really pitches the game?

    Looks at every game this pitcher opened, how deep he went, and which arm followed him.
    When the pattern holds, the returned `bulk` block is the pitcher the lineup will spend
    most of the night against -- which is the one worth scouting.
    """
    blank = {"role": "", "is_opener": False, "starts": 0, "opener_starts": 0,
             "avg_first_inn": None, "bulk": None, "candidates": []}
    if not pitcher_id:
        return blank
    index = build_opener_index(season, context_end, statcast_pitches_df)
    if index is None or index.empty:
        return blank

    mine = index[index["first"] == int(pitcher_id)]
    if mine.empty:
        return blank
    starts = len(mine)
    opener_starts = int(mine["is_opener"].sum())
    avg_first_inn = round(float(mine["first_inn"].mean()), 1)
    rate = opener_starts / starts if starts else 0.0

    if opener_starts >= min_starts and rate >= 0.5:
        role = "Opener"
    elif opener_starts >= min_starts and rate >= 0.25:
        role = "Sometimes an opener"
    elif avg_first_inn <= 3.5 and starts >= 3:
        role = "Short starter"
    else:
        role = "Traditional starter"

    candidates = []
    opener_games = mine[mine["is_opener"] & mine["bulk"].notna()]
    if not opener_games.empty:
        counts = opener_games["bulk"].value_counts()
        for bulk_id, times in counts.head(3).items():
            sub = opener_games[opener_games["bulk"] == bulk_id]
            candidates.append({
                "id": int(bulk_id),
                "name": _player_name_from_id(int(bulk_id)),
                "times": int(times),
                "avg_inn": round(float(sub["bulk_inn"].mean()), 1),
                "share": round(times / len(opener_games) * 100),
            })

    return {
        "role": role,
        "is_opener": role in {"Opener", "Sometimes an opener"},
        "starts": starts,
        "opener_starts": opener_starts,
        "opener_rate": round(rate * 100),
        "avg_first_inn": avg_first_inn,
        "bulk": candidates[0] if candidates else None,
        "candidates": candidates,
        # How often the most-frequent prior follower is the one who actually follows,
        # measured over 176 opener games in 2024-26. It sits at roughly a third even when
        # the historical share is 60%+ and even with four or more prior games to go on --
        # bullpen sequencing is a game-state decision, not a tendency. Carried on the
        # profile so the report can state the confidence instead of implying one.
        "bulk_repeat_rate": OPENER_BULK_REPEAT_RATE,
    }


# ---------------------------------------------------------------------------------------
# Team rotation model
#
# Replaces the old question "who follows tonight's opener" with a better-posed one: "who does
# this club start, and in what order". The first was abandoned at a 33% hit rate because
# bullpen sequencing is a game-state decision. Start order is not -- clubs cycle a rotation --
# and it measures far better. Walk-forward over 7,084 team-games of 2026:
#
#     most frequent starter (baseline)          17.5%
#     most days rest                             7.7%
#     most days rest, last 30 days              10.3%
#     who followed this starter before          46.1%     <- the model below
#
# Two facts shape the design. Rotations churn hard: **11.7 distinct starters per club** over a
# season, which is why "most rested" scores near nothing -- it keeps nominating arms who made
# one April start and were never seen again. Availability filtering is therefore not a polish
# step, it is the thing that makes the rest term usable at all. And opener-shaped starts are
# only 2.2% of team-games, so the everyday value here is the no-probable-announced case rather
# than the opener case.
# ---------------------------------------------------------------------------------------

_TEAM_START_LOG_CACHE = {}
_ROSTER_STATUS_CACHE = {}

# Starts inside this window make an arm a live rotation candidate. Long enough to survive a
# skipped turn or a short IL stint, short enough that April's call-ups age out.
ROTATION_RECENT_DAYS = 45
# How startable an arm is on N days of rest. Peaked, not monotone, and that matters: a curve
# that simply saturates above four days makes a pitcher who last started a month ago look as
# ready as one on a normal turn, which is how Logan Allen came back as Cleveland's second most
# likely starter on 29 days' rest off a single start. Four to six days is a rotation turn;
# past ten, the arm is not in the rotation right now whatever his history says.
ROTATION_REST_CURVE = [
    (0, 0.02), (1, 0.02), (2, 0.05), (3, 0.35),
    (4, 1.00), (5, 1.00), (6, 1.00),
    (7, 0.80), (9, 0.55), (12, 0.30), (16, 0.15),
]
ROTATION_REST_FAR = 0.08        # beyond the last knot
# Weight on the league opener rate when shrinking a pitcher's own. Openers are rare (2.2%), so
# an arm with one short outing in one start should not come back as a 100% opener.
OPENER_PRIOR_STARTS = 4.0


def build_team_start_log(season, context_end, statcast_pitches_df=None):
    """One row per (team, game): who started, how long, who followed, was it opener-shaped.

    Built from the same pitch-level data as `build_opener_index` and with the same two
    load-bearing details -- order from `at_bat_number` rather than inning, and exhibition
    games excluded -- but keyed by *team* and date so a rotation can be read off it.

    The pitching team comes from the half: the top of an inning is the away side batting, so
    the home club is on the mound.
    """
    key = (int(season), str(context_end))
    if key in _TEAM_START_LOG_CACHE:
        return _TEAM_START_LOG_CACHE[key]

    df = statcast_pitches_df
    if df is None or df.empty:
        try:
            df = load_statcast_pitches(season=season, end_date=context_end)
        except Exception as error:
            print(f"⚠️ Rotation model unavailable: {error}")
            return pd.DataFrame()
    needed = {"game_pk", "game_date", "home_team", "away_team", "inning_topbot",
              "pitcher", "at_bat_number", "events"}
    if df is None or df.empty or not needed.issubset(df.columns):
        return pd.DataFrame()

    columns = list(needed) + (["game_type"] if "game_type" in df.columns else [])
    slim = df[columns].copy()
    if "game_type" in slim.columns:
        slim = slim[slim["game_type"].astype("string").eq("R")]
    slim = slim.dropna(subset=["game_pk", "pitcher", "at_bat_number", "inning_topbot"])
    if slim.empty:
        return pd.DataFrame()

    slim["outs"] = slim["events"].astype("string").map(_EVENT_OUTS).fillna(0.0)
    per = slim.groupby(["game_pk", "inning_topbot", "pitcher"], observed=True).agg(
        first_ab=("at_bat_number", "min"), outs=("outs", "sum"),
        game_date=("game_date", "first"), home=("home_team", "first"),
        away=("away_team", "first")).reset_index()
    per["ip"] = per["outs"] / 3.0
    per = per.sort_values(["game_pk", "inning_topbot", "first_ab"], kind="mergesort")
    per["slot"] = per.groupby(["game_pk", "inning_topbot"], observed=True).cumcount()
    per["team"] = np.where(per["inning_topbot"].astype(str).str.startswith("Top"),
                           per["home"], per["away"])

    first = per[per["slot"] == 0].rename(columns={"pitcher": "starter", "ip": "starter_ip"})
    second = per[per["slot"] == 1][["game_pk", "inning_topbot", "pitcher", "ip"]].rename(
        columns={"pitcher": "bulk", "ip": "bulk_ip"})
    log = first[["game_pk", "inning_topbot", "game_date", "team",
                 "starter", "starter_ip"]].merge(
        second, on=["game_pk", "inning_topbot"], how="left")
    log["bulk_ip"] = log["bulk_ip"].fillna(0.0)
    log["is_opener"] = (log["starter_ip"] <= OPENER_MAX_INNINGS) & \
                       (log["bulk_ip"] >= BULK_MIN_INNINGS)
    log["starter"] = log["starter"].astype(int)
    log["game_date"] = pd.to_datetime(log["game_date"])
    log = log.sort_values(["team", "game_date", "game_pk"]).reset_index(drop=True)
    _TEAM_START_LOG_CACHE[key] = log
    return log


def roster_status(team_abbr, as_of_date=None):
    """{player_id: (state, note)} where state is 'active', 'injured' or 'gone'.

    Three lookups because the answer needs three: the active roster says who can pitch
    tonight, the 40-man adds the injured with the injury text attached, and anyone the start
    log knows about who appears on neither has been traded, released or outrighted. Without
    this the rotation nominates arms who are no longer on the club, which is most of why a
    naive rest-based prediction scores 7.7%.
    """
    key = (str(team_abbr), str(as_of_date))
    if key in _ROSTER_STATUS_CACHE:
        return _ROSTER_STATUS_CACHE[key]
    out = {}
    try:
        for entry in get_team_roster(team_abbr, "40Man", as_of_date=as_of_date) or []:
            pid = ((entry.get("person") or {}).get("id"))
            if pid is None:
                continue
            status = ((entry.get("status") or {}).get("description") or "").strip()
            note = (entry.get("note") or "").strip()
            state = "active" if status.lower() == "active" else "injured"
            out[int(pid)] = (state, note or status)
        for entry in get_team_roster(team_abbr, "active", as_of_date=as_of_date) or []:
            pid = ((entry.get("person") or {}).get("id"))
            if pid is not None:
                out[int(pid)] = ("active", "")
    except Exception as error:
        print(f"⚠️ Roster status unavailable for {team_abbr}: {error}")
        return {}
    _ROSTER_STATUS_CACHE[key] = out
    return out


def _rest_factor(days):
    """How startable an arm is on this much rest, from ROTATION_REST_CURVE."""
    if days is None:
        return 0.5
    for cut, value in ROTATION_REST_CURVE:
        if days <= cut:
            return value
    return ROTATION_REST_FAR


def _league_opener_rate(log):
    return float(log["is_opener"].mean()) if log is not None and not log.empty else 0.02


def build_rotation(team_abbr, season, context_end, as_of_date=None,
                   log=None, statcast_pitches_df=None):
    """This club's rotation as of a date: who the candidates are and who is likely to start.

    Returns a dict with `candidates` (each carrying a start probability and a roster state),
    the start-order `transitions` the probability is built from, and `bulk_arms` -- the arms
    who actually cover innings behind a short start, which is what an opener night needs.

    `p_start` is the sequence share for each candidate, gated by availability and rest and
    renormalised over whoever is left. Announced probables are not consulted here; that is
    `effective_starter`'s job, and keeping them apart means this can be checked against
    history on nights where a probable was in fact posted.
    """
    if log is None:
        log = build_team_start_log(season, context_end, statcast_pitches_df)
    blank = {"team": team_abbr, "as_of": as_of_date, "candidates": [],
             "transitions": {}, "bulk_arms": [], "last_starter": None,
             "league_opener_rate": 0.02, "note": ""}
    if log is None or log.empty:
        return {**blank, "note": "no start log available"}

    as_of = pd.to_datetime(as_of_date) if as_of_date else log["game_date"].max()
    mine = log[(log["team"] == team_abbr) & (log["game_date"] < as_of)]
    if mine.empty:
        return {**blank, "note": f"no starts on record for {team_abbr}"}
    mine = mine.sort_values(["game_date", "game_pk"])

    league_opener = _league_opener_rate(log)
    status = roster_status(team_abbr, as_of_date=as_of_date)

    # Start-order transitions: who has followed whom in the rotation.
    transitions = {}
    starters = mine["starter"].tolist()
    for previous, following in zip(starters, starters[1:]):
        transitions.setdefault(int(previous), {})
        transitions[int(previous)][int(following)] = \
            transitions[int(previous)].get(int(following), 0) + 1

    last_starter = int(starters[-1])
    following_counts = transitions.get(last_starter, {})
    following_total = sum(following_counts.values()) or 1

    recent_cut = as_of - pd.Timedelta(days=ROTATION_RECENT_DAYS)
    candidates = []
    for pid, group in mine.groupby("starter"):
        last_start = group["game_date"].max()
        starts = len(group)
        recent_starts = int((group["game_date"] >= recent_cut).sum())
        state, note = status.get(int(pid), ("gone", "not on the 40-man"))
        days_rest = int((as_of - last_start).days)
        opener_starts = int(group["is_opener"].sum())
        # Shrunk toward the league rate so one short outing is not a career.
        p_opener = (opener_starts + league_opener * OPENER_PRIOR_STARTS) / \
                   (starts + OPENER_PRIOR_STARTS)
        sequence_share = following_counts.get(int(pid), 0) / following_total
        available = state == "active" and recent_starts > 0
        candidates.append({
            "id": int(pid),
            "name": _player_name_from_id(int(pid)),
            "starts": starts,
            "recent_starts": recent_starts,
            "last_start": last_start.strftime("%Y-%m-%d"),
            "days_rest": days_rest,
            "avg_ip": round(float(group["starter_ip"].mean()), 2),
            "opener_starts": opener_starts,
            "p_opener": round(float(p_opener), 3),
            "state": state,
            "note": note,
            "sequence_share": round(float(sequence_share), 3),
            "available": available,
            "_score": sequence_share * _rest_factor(days_rest) if available else 0.0,
        })

    total = sum(c["_score"] for c in candidates)
    if total <= 0:
        # Nobody scored: fall back to rest order among available arms, which is the right
        # answer when the club has just turned its rotation over and no transition is known.
        pool = [c for c in candidates if c["available"]] or candidates
        for c in candidates:
            c["p_start"] = 0.0
        if pool:
            pool.sort(key=lambda c: -c["days_rest"])
            pool[0]["p_start"] = 1.0
    else:
        for c in candidates:
            c["p_start"] = round(c["_score"] / total, 3)
    for c in candidates:
        c.pop("_score", None)
    candidates.sort(key=lambda c: (-c["p_start"], -c["recent_starts"]))

    # Arms who actually cover innings behind a short start -- the opener-night substitute.
    bulk = mine[mine["bulk"].notna() & (mine["bulk_ip"] >= BULK_MIN_INNINGS)]
    bulk_arms = []
    if not bulk.empty:
        counts = bulk["bulk"].value_counts()
        for bid, times in counts.head(6).items():
            sub = bulk[bulk["bulk"] == bid]
            state, note = status.get(int(bid), ("gone", "not on the 40-man"))
            bulk_arms.append({
                "id": int(bid), "name": _player_name_from_id(int(bid)),
                "games": int(times), "avg_ip": round(float(sub["bulk_ip"].mean()), 2),
                "share": round(times / len(bulk) * 100),
                "last": sub["game_date"].max().strftime("%Y-%m-%d"),
                "state": state, "note": note,
            })

    return {"team": team_abbr, "as_of": as_of.strftime("%Y-%m-%d"),
            "candidates": candidates, "transitions": transitions,
            "bulk_arms": bulk_arms, "last_starter": last_starter,
            "league_opener_rate": round(league_opener, 4), "note": ""}


def effective_starter(rotation, announced_id=None, opener_profile=None,
                      opener_threshold=0.5):
    """Who to analyse as tonight's starter, and why.

    Three cases, in order:

      announced and not an opener   the listed probable, as before
      announced but an opener       the club's likeliest bulk arm, because the lineup will
                                    spend the night against him rather than the opener
      nothing announced             the rotation's most likely candidate

    Returns `{"id", "name", "source", "note", "displaced"}`. `source` says which branch
    fired so a report can state it rather than silently swapping a name -- substituting a
    pitcher without saying so is how a board becomes untrustworthy.
    """
    candidates = (rotation or {}).get("candidates") or []
    bulk_arms = [b for b in (rotation or {}).get("bulk_arms") or []
                 if b.get("state") == "active"]

    def bulk_pick(displaced, why):
        if not bulk_arms:
            return None
        best = max(bulk_arms, key=lambda b: (b["games"], b["avg_ip"]))
        return {"id": best["id"], "name": best["name"], "source": "bulk",
                "displaced": displaced,
                "note": (f"{why}; {best['name']} has covered bulk innings in "
                         f"{best['games']} of this club's short starts "
                         f"({best['share']}%, {best['avg_ip']} IP average)")}

    if announced_id:
        p_opener = None
        if isinstance(opener_profile, dict):
            p_opener = opener_profile.get("p_opener")
            if p_opener is None and opener_profile.get("is_opener"):
                p_opener = 1.0
        if p_opener is None:
            match = next((c for c in candidates if c["id"] == int(announced_id)), None)
            p_opener = (match or {}).get("p_opener", 0.0)
        if p_opener is not None and p_opener >= opener_threshold:
            swapped = bulk_pick(int(announced_id),
                                f"listed starter profiles as an opener (p={p_opener:.2f})")
            if swapped:
                return swapped
        name = next((c["name"] for c in candidates if c["id"] == int(announced_id)), None)
        return {"id": int(announced_id), "name": name or _player_name_from_id(int(announced_id)),
                "source": "announced", "displaced": None, "note": ""}

    startable = [c for c in candidates if c.get("p_start", 0) > 0]
    if startable:
        best = startable[0]
        if best.get("p_opener", 0.0) >= opener_threshold:
            swapped = bulk_pick(best["id"],
                                f"no probable announced and the likeliest starter "
                                f"{best['name']} profiles as an opener")
            if swapped:
                return swapped
        return {"id": best["id"], "name": best["name"], "source": "rotation",
                "displaced": None,
                "note": (f"no probable announced; rotation model gives {best['name']} "
                         f"{best['p_start']:.0%} on {best['days_rest']} days rest")}
    return bulk_pick(None, "no probable announced and no rotation candidate is startable") \
        or {"id": None, "name": "TBD", "source": "unknown", "displaced": None,
            "note": "no probable announced and no usable rotation history"}


_TEAM_HITTING_CONTEXT_CACHE = {}


def load_team_hitting_context(season):
    """{team_id: {Team, OPS, R/G, OPS Rank}} — season-to-date offense for every club.

    Used to grade the lineups a starter faced in his last few outings, so a strong line
    against a weak offense reads differently from the same line against a good one.
    """
    season = int(season)
    if season in _TEAM_HITTING_CONTEXT_CACHE:
        return _TEAM_HITTING_CONTEXT_CACHE[season]
    try:
        data = cached_json_request(
            "https://statsapi.mlb.com/api/v1/teams/stats",
            params={"season": season, "group": "hitting", "stats": "season", "sportIds": 1},
            namespace="statsapi",
            cache_key_extra=datetime.now().strftime("%Y-%m-%d"),
        )
    except Exception as e:
        print(f"⚠️ Team hitting context unavailable: {e}")
        return {}
    rows = []
    for split in (data.get("stats", [{}])[0] or {}).get("splits", []):
        stat, team = split.get("stat", {}) or {}, split.get("team", {}) or {}
        games = _safe_number(stat.get("gamesPlayed"), 0)
        if not team.get("id") or not games:
            continue
        rows.append({
            "team_id": int(team["id"]),
            "Team": TEAM_ID_MAP.get(int(team["id"]), team.get("name", "")),
            "Name": team.get("name", ""),
            "OPS": _safe_number(stat.get("ops"), 0.0),
            "R/G": round(_safe_number(stat.get("runs"), 0) / games, 2),
        })
    if not rows:
        return {}
    frame = pd.DataFrame(rows).sort_values("OPS", ascending=False).reset_index(drop=True)
    frame["OPS Rank"] = frame.index + 1
    table = {int(r["team_id"]): r.to_dict() for _, r in frame.iterrows()}
    _TEAM_HITTING_CONTEXT_CACHE[season] = table
    return table


def _team_hitting_lookup(season):
    """Hitting context keyed by every label a game log might use: abbreviation, full
    club name, and nickname ("Texas Rangers" / "TEX" / "Rangers" all resolve)."""
    table = {}
    for entry in load_team_hitting_context(season).values():
        full = str(entry.get("Name", "")).strip()
        for key in (entry.get("Team"), full, full.split()[-1] if full else ""):
            if key:
                table[str(key).lower()] = entry
    return table


_TEAM_ABBR_BY_NAME = {}


def _team_abbr_lookup():
    """Full club name / nickname -> abbreviation, for shortening game-log opponents."""
    if not _TEAM_ABBR_BY_NAME:
        try:
            for team in get_team_list():
                abbr, name = team.get("abbreviation"), team.get("name", "")
                for key in (name, name.split()[-1] if name else "", abbr):
                    if key and abbr:
                        _TEAM_ABBR_BY_NAME[str(key).lower()] = {"Team": abbr}
        except Exception as e:
            print(f"⚠️ Team abbreviation lookup unavailable: {e}")
    return _TEAM_ABBR_BY_NAME


def _resolve_opponent(label, lookup=None):
    """("vs Texas Rangers" | "@ HOU") -> (short display label, hitting-context entry).

    Game logs carry full club names with a vs/@ prefix; the report needs the abbreviation
    and the offense's season quality, so strip the prefix and match on any known alias.
    """
    lookup = lookup if lookup is not None else _team_abbr_lookup()
    text = str(label or "").strip()
    prefix = ""
    for marker in ("vs ", "@ ", "vs.", "at "):
        if text.lower().startswith(marker.strip().lower() + " ") or text.startswith(marker):
            prefix = "@" if marker.strip().lower() in {"@", "at"} else "vs"
            text = text[len(marker):].strip()
            break
    entry = lookup.get(text.lower())
    if entry is None and text:
        # Fall back to a nickname match ("Rangers" out of "Texas Rangers").
        entry = lookup.get(text.split()[-1].lower())
    short = str((entry or {}).get("Team") or text)
    return (f"{prefix} {short}".strip() if prefix else short), (entry or {})


def _float_to_ip(value):
    """6.3333 -> 6.1 : innings pitched read as whole innings plus outs, not decimals."""
    number = _safe_number(value, None)
    if number is None:
        return value if value not in (None, "") else ""
    whole = int(number)
    outs = int(round((number - whole) * 3))
    if outs >= 3:
        whole, outs = whole + 1, 0
    return float(f"{whole}.{outs}")


_MIX_PITCH_LIMIT = 5          # top-N pitches by season usage get their own columns
_BATTED_MIN = 5               # below this many batted balls a per-start rate is noise
                              # (an opener's 1-inning outing yields 1-3, i.e. "100% GB")


def _pitch_rate_block(frame):
    """Batted-ball profile for one slice of a pitcher's statcast rows."""
    batted = frame[frame["bb_type"].notna()] if "bb_type" in frame.columns else frame.iloc[0:0]
    ev = pd.to_numeric(frame.get("launch_speed"), errors="coerce").dropna()
    n = len(batted)
    lsa = pd.to_numeric(frame.get("launch_speed_angle"), errors="coerce")

    def share(kind):
        return round(float((batted["bb_type"] == kind).sum()) / n * 100, 1) if n else ""

    return {
        "BIP": n,
        "GB%": share("ground_ball"),
        "FB%": share("fly_ball"),
        "LD%": share("line_drive"),
        "HH%": round(float((ev >= 95).sum()) / len(ev) * 100, 1) if len(ev) else "",
        "Max EV": round(float(ev.max()), 1) if len(ev) else "",
        "Avg EV": round(float(ev.mean()), 1) if len(ev) else "",
        "Brl%": round(float((lsa == 6).sum()) / n * 100, 1) if n else "",
    }


def build_starter_pitch_profile(pitcher_id, season, context_end, last_n=5):
    """Per-start pitch mix, velocity and batted-ball profile against season baselines.

    Returns a dict of frames ready for the pitching tab, plus the season baseline rows the
    per-start cells are colored against. Reuses the same disk-cached statcast_pitcher pull
    the opposing-pitching table already makes, so this adds no new network cost.
    """
    empty = {"pitches": [], "mix": pd.DataFrame(), "batted": pd.DataFrame(),
             "season_mix": {}, "season_batted": {}, "starts": []}
    if not pitcher_id:
        return empty
    start_date = _season_start_date(season)
    try:
        df = cached_dataframe_call("statcast_pitcher", statcast_pitcher,
                                   start_date, context_end, int(pitcher_id))
    except Exception as e:
        print(f"⚠️ Statcast unavailable for pitcher {pitcher_id}: {e}")
        return empty
    if df is None or df.empty or "pitch_type" not in df.columns:
        return empty

    df = df[df["pitch_type"].notna()].copy()
    df["game_date"] = df["game_date"].astype(str)
    if df.empty:
        return empty

    total = len(df)
    usage = df["pitch_type"].value_counts()
    pitches = [p for p in usage.head(_MIX_PITCH_LIMIT).index]
    season_mix = {
        p: {"usage": round(usage[p] / total * 100, 1),
            "velo": round(float(pd.to_numeric(df.loc[df["pitch_type"] == p, "release_speed"],
                                              errors="coerce").mean()), 1)}
        for p in pitches
    }
    season_batted = _pitch_rate_block(df)

    # Opponent = whichever club was batting, which is always the side this pitcher faced.
    def opponent_for(frame):
        if not {"home_team", "away_team", "inning_topbot"}.issubset(frame.columns):
            return ""
        batting = np.where(frame["inning_topbot"].astype(str).str.lower().str.startswith("top"),
                           frame["away_team"], frame["home_team"])
        values = pd.Series(batting).dropna()
        return values.mode().iloc[0] if not values.empty else ""

    recent_dates = sorted(df["game_date"].unique())[-last_n:][::-1]
    mix_rows, batted_rows, starts = [], [], []
    for date in recent_dates:
        game = df[df["game_date"] == date]
        opponent = opponent_for(game)
        starts.append({"date": date, "opponent": opponent, "pitches": len(game)})
        counts = game["pitch_type"].value_counts()
        row = {"Date": date[5:], "Opp": opponent, "P": len(game)}
        for p in pitches:
            sub = game[game["pitch_type"] == p]
            row[f"{p}%"] = round(len(sub) / len(game) * 100, 1) if len(game) else 0.0
            velo = pd.to_numeric(sub.get("release_speed"), errors="coerce").mean()
            row[f"{p} mph"] = round(float(velo), 1) if pd.notna(velo) else ""
        mix_rows.append(row)
        batted_rows.append({"Date": date[5:], "Opp": opponent, **_pitch_rate_block(game)})

    season_row = {"Date": "SEASON", "Opp": "", "P": total}
    for p in pitches:
        season_row[f"{p}%"] = season_mix[p]["usage"]
        season_row[f"{p} mph"] = season_mix[p]["velo"]
    mix_cols = ["Date", "Opp", "P"] + [c for p in pitches for c in (f"{p}%", f"{p} mph")]
    batted_cols = ["Date", "Opp", "BIP", "GB%", "FB%", "LD%", "HH%", "Avg EV", "Max EV", "Brl%"]

    return {
        "pitches": pitches,
        "mix": pd.DataFrame([season_row] + mix_rows, columns=mix_cols),
        "batted": pd.DataFrame([{"Date": "SEASON", "Opp": "", **season_batted}] + batted_rows,
                               columns=batted_cols),
        "season_mix": season_mix,
        "season_batted": season_batted,
        "starts": starts,
    }


def _short_pitcher_name(name):
    """"Paul Skenes" -> "P. Skenes": fits a narrow Unit/label cell without losing identity."""
    parts = [p for p in str(name or "").strip().split() if p]
    if len(parts) < 2:
        return parts[0] if parts else ""
    return f"{parts[0][0]}. {' '.join(parts[1:])}"


def _starter_label(info, fallback="Starter"):
    """"P. Skenes (RHP)" for panel captions, from a starter-info dict."""
    name = _short_pitcher_name((info or {}).get("Name"))
    if not name:
        return fallback
    hand = str((info or {}).get("Throws", "")).strip()[:1].upper()
    return f"{name} ({hand}HP)" if hand in {"L", "R"} else name


def build_opp_pitching_allowed_summary(starter_id, pitching_team, bullpen_df, season, context_end,
                                       max_relievers=7, starter_name=None):
    """What the opposing starter and bullpen allow, sliced like the lineup summary.

    Same columns as the hitting summary and computed by the same helper, so a row here can
    be read directly against the row above it -- K% allowed against the lineup's own K%,
    ISO allowed against its ISO.

    The starter's rows are labeled with his actual name rather than "Starter" so the table
    reads without cross-referencing which pitcher the section belongs to."""
    columns = ["Team", "Unit", "Split"] + SPLIT_STAT_COLUMNS
    start_date = _season_start_date(season)
    if context_end and context_end < start_date:
        return pd.DataFrame(columns=columns)
    cutoff = None
    if context_end:
        try:
            cutoff = (datetime.strptime(context_end, "%Y-%m-%d") - timedelta(days=28)).strftime("%Y-%m-%d")
        except Exception:
            cutoff = None

    def pitcher_pa_rows(pid):
        if not pid:
            return None
        df = cached_dataframe_call("statcast_pitcher", statcast_pitcher, start_date, context_end, pid)
        if df is None or df.empty:
            return None
        if "events" in df.columns:
            return df[df["events"].notna()].copy()
        return df

    def add_common_splits(unit, frame, park_team=None):
        rows.append({"Unit": unit, "Split": "Overall", **_split_metrics(frame)})
        if park_team and "home_team" in frame.columns:
            canon = _canon_team(park_team)
            home_mask = frame["home_team"].astype(str).map(_canon_team) == canon
            hf = _team_home_park_run_factor(canon)
            rows.append({"Unit": unit, "Split": "Home (PF)",
                         **_park_adjusted_split(frame[home_mask], hf)})
            away_sub = frame[~home_mask]
            af = float(away_sub["home_team"].astype(str).map(_team_home_park_run_factor).mean()) if not away_sub.empty else 1.0
            rows.append({"Unit": unit, "Split": "Away (PF)",
                         **_park_adjusted_split(away_sub, af)})
        if "stand" in frame.columns:
            for hand, label in [("L", "vs LHB"), ("R", "vs RHB")]:
                sub = frame[frame["stand"].astype(str).str.strip() == hand]
                rows.append({"Unit": unit, "Split": label, **_split_metrics(sub)})
        if cutoff and "game_date" in frame.columns:
            sub = frame[frame["game_date"].astype(str) >= cutoff]
            rows.append({"Unit": unit, "Split": "L28", **_split_metrics(sub)})

    rows = []
    starter_df = pitcher_pa_rows(starter_id)
    if starter_df is not None and not starter_df.empty:
        add_common_splits(_short_pitcher_name(starter_name) or "Starter", starter_df,
                          park_team=pitching_team)

    reliever_ids = []
    if bullpen_df is not None and not bullpen_df.empty and "Name" in bullpen_df.columns:
        try:
            roster = get_team_roster(pitching_team, as_of_date=context_end)
            name_to_id = {remove_accents(p.get("person", {}).get("fullName", "")): p.get("person", {}).get("id") for p in roster}
        except Exception:
            name_to_id = {}
        pen = bullpen_df.sort_values("IP", ascending=False) if "IP" in bullpen_df.columns else bullpen_df
        for name in pen["Name"].head(max_relievers):
            pid = name_to_id.get(remove_accents(str(name)))
            if pid:
                reliever_ids.append(int(pid))
    pen_frames = [f for f in (pitcher_pa_rows(pid) for pid in reliever_ids) if f is not None and not f.empty]
    if pen_frames:
        add_common_splits("Bullpen", pd.concat(pen_frames, ignore_index=True))

    if not rows:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(rows)
    out.insert(0, "Team", pitching_team)
    return out[columns]


def compute_offense_index(ops, xwoba, lg_ops, lg_xwoba):
    """League-relative offense index (100 = league average) blending OPS and xwOBA. Proxy, not official wRC+."""
    o = _safe_number(ops, None)
    x = _safe_number(xwoba, None)
    lo = _safe_number(lg_ops, None)
    lx = _safe_number(lg_xwoba, None)
    parts, weights = [], []
    if o is not None and lo and lo > 0:
        parts.append(0.6 * (o / lo))
        weights.append(0.6)
    if x is not None and lx and lx > 0:
        parts.append(0.4 * (x / lx))
        weights.append(0.4)
    if not parts:
        return ""
    return int(round(100 * sum(parts) / sum(weights)))


def attach_offense_index(hitter_composite, lineup_frames, league_batter_context, hotcold_frames):
    """Add 'Off Szn' and 'Off L28' league-relative offense index columns to the hitter composite table."""
    if hitter_composite is None or hitter_composite.empty:
        return hitter_composite
    lg_ops = _mean_numeric(league_batter_context, "OPS_proxy", 0.0)
    lg_xw = _mean_numeric(league_batter_context, "xwOBA", 0.0)

    name_to_id = {}
    for frame in lineup_frames:
        if frame is not None and not frame.empty and {"Name", "ID"}.issubset(frame.columns):
            for _, r in frame.iterrows():
                name_to_id[str(r.get("Name"))] = pd.to_numeric(r.get("ID"), errors="coerce")

    ctx = pd.DataFrame()
    if league_batter_context is not None and not league_batter_context.empty and "BatterID" in league_batter_context.columns:
        ctx = league_batter_context.drop_duplicates("BatterID").set_index("BatterID")

    l28 = {}
    for frame in hotcold_frames:
        if frame is not None and not frame.empty and "Name" in frame.columns:
            for _, r in frame.iterrows():
                l28[str(r.get("Name"))] = (r.get("OPS"), r.get("xwOBA"))

    szn_vals, l28_vals = [], []
    for _, row in hitter_composite.iterrows():
        name = str(row.get("Name"))
        bid = name_to_id.get(name)
        szn = ""
        if bid is not None and not pd.isna(bid) and not ctx.empty and int(bid) in ctx.index:
            crow = ctx.loc[int(bid)]
            szn = compute_offense_index(crow.get("OPS_proxy"), crow.get("xwOBA"), lg_ops, lg_xw)
        szn_vals.append(szn)
        l28_val = ""
        if name in l28:
            ops_l, xw_l = l28[name]
            l28_val = compute_offense_index(ops_l, xw_l, lg_ops, lg_xw)
        l28_vals.append(l28_val)

    out = hitter_composite.copy()
    out["Off Szn"] = szn_vals
    out["Off L28"] = l28_vals
    return out


def advanced_catcher_label(def_team, catcher_run_game_df):
    if catcher_run_game_df is None or catcher_run_game_df.empty:
        return ""
    row = catcher_run_game_df.iloc[0]
    return f"{row.get('Aggression', 'N/A')} vs C {row.get('Run Control', row.get('CS%', 'N/A'))}"


def _percentile(value, population, higher_is_better=True):
    value = _safe_number(value)
    if value is None:
        return None
    series = pd.to_numeric(population, errors="coerce").dropna()
    if series.empty:
        return None
    pct = (series <= value).mean() * 100
    if not higher_is_better:
        pct = 100 - pct
    return int(round(pct))


def _percentile_badge(pct):
    if pct is None:
        return ""
    if pct >= 75:
        tier = "ELITE"
    elif pct >= 45:
        tier = "AVG"
    else:
        tier = "LOW"
    return f"{tier} P{pct}"


def generate_league_batter_context(statcast_df, min_pa=50):
    columns = ["BatterID", "PA", "OPS_proxy", "xwOBA", "HardHit%", "K%"]
    if statcast_df is None or statcast_df.empty or "batter" not in statcast_df.columns:
        return pd.DataFrame(columns=columns)
    df = statcast_df[statcast_df["events"].isin(TERMINAL_PA_EVENTS)].copy()
    if df.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for batter_id, group in df.groupby("batter"):
        line = _events_to_batting_line(group)
        if line["PA"] < min_pa:
            continue
        batted = group[pd.to_numeric(group.get("launch_speed"), errors="coerce").notna()]
        hard_hit = (pd.to_numeric(batted.get("launch_speed"), errors="coerce") >= 95).mean() * 100 if not batted.empty else 0.0
        rows.append({
            "BatterID": int(batter_id),
            "PA": line["PA"],
            "OPS_proxy": line["OPS_proxy"],
            "xwOBA": round(pd.to_numeric(group.get("estimated_woba_using_speedangle"), errors="coerce").mean(), 3),
            "HardHit%": round(hard_hit, 1),
            "K%": round((line["K"] / line["PA"]) * 100, 1) if line["PA"] else 0.0,
        })
    return pd.DataFrame(rows, columns=columns)


def add_lineup_percentile_context(lineup_df, league_batter_context):
    if lineup_df is None or lineup_df.empty or league_batter_context is None or league_batter_context.empty or "ID" not in lineup_df.columns:
        return lineup_df
    context = league_batter_context.copy()
    out = lineup_df.copy()
    out["BatterID"] = pd.to_numeric(out["ID"], errors="coerce")
    out = out.merge(context, on="BatterID", how="left", suffixes=("", "_lg"))
    out["OPS Pctl"] = out["OPS_proxy"].apply(lambda value: _percentile_badge(_percentile(value, context["OPS_proxy"])))
    out["xwOBA Pctl"] = out["xwOBA"].apply(lambda value: _percentile_badge(_percentile(value, context["xwOBA"])))
    out["HardHit Pctl"] = out["HardHit%"].apply(lambda value: _percentile_badge(_percentile(value, context["HardHit%"])))
    out["Contact Pctl"] = out["K%"].apply(lambda value: _percentile_badge(_percentile(value, context["K%"], higher_is_better=False)))
    return out


def build_hitter_watchlist(team, lineup_df, similar_df, league_batter_context, splits_df=None):
    columns = ["Team", "Name", "Priority", "Why", "OPS", "ISO", "Platoon", "Similar Sample", "OPS Pctl", "xwOBA Pctl", "HardHit Pctl", "Contact Pctl"]
    if lineup_df is None or lineup_df.empty:
        return pd.DataFrame(columns=columns)

    lineup = add_lineup_percentile_context(lineup_df, league_batter_context).copy()
    similar = similar_df[["Name", "OPS_proxy", "PA"]].rename(columns={"OPS_proxy": "Similar OPS", "PA": "Similar PA"}) if similar_df is not None and not similar_df.empty else pd.DataFrame(columns=["Name", "Similar OPS", "Similar PA"])
    lineup = lineup.merge(similar, on="Name", how="left")
    split_ops_col = split_pa_col = None
    if splits_df is not None and not splits_df.empty and "Name" in splits_df.columns:
        split_cols = ["Name"] + [col for col in splits_df.columns if col.startswith(("OPS vs", "PA vs", "ISO vs", "K% vs"))]
        lineup = lineup.merge(splits_df[split_cols], on="Name", how="left")
        split_ops_cols = [col for col in split_cols if col.startswith("OPS vs")]
        split_pa_cols = [col for col in split_cols if col.startswith("PA vs")]
        split_ops_col = split_ops_cols[0] if split_ops_cols else None
        split_pa_col = split_pa_cols[0] if split_pa_cols else None

    ops = _numeric_col(lineup, "OPS")
    iso = _numeric_col(lineup, "ISO")
    sim_ops = _numeric_col(lineup, "Similar OPS")
    sim_pa = _numeric_col(lineup, "Similar PA")
    sim_ops_for_score = sim_ops.where(sim_pa >= 10, 0.700)
    platoon_ops = _numeric_col(lineup, split_ops_col, 0.700) if split_ops_col else pd.Series([0.700] * len(lineup), index=lineup.index)
    platoon_pa = _numeric_col(lineup, split_pa_col, 0) if split_pa_col else pd.Series([0] * len(lineup), index=lineup.index)
    platoon_ops_for_score = platoon_ops.where(platoon_pa >= 25, 0.700)
    lineup["Watch Score"] = (
        (ops * 40)
        + (iso * 85)
        + (sim_ops_for_score * 16)
        + (platoon_ops_for_score * 18)
        + np.minimum(sim_pa, 40) * 0.01
        + np.minimum(platoon_pa, 80) * 0.015
    )

    rows = []
    for _, row in lineup.sort_values("Watch Score", ascending=False).head(5).iterrows():
        reasons = []
        if _safe_number(row.get("ISO"), 0) >= 0.180:
            reasons.append("power")
        if split_ops_col and _safe_number(row.get(split_ops_col), 0) >= 0.800 and _safe_number(row.get(split_pa_col), 0) >= 25:
            reasons.append("platoon fit")
        elif split_ops_col and _safe_number(row.get(split_ops_col), 0) <= 0.620 and _safe_number(row.get(split_pa_col), 0) >= 25:
            reasons.append("platoon concern")
        if _safe_number(row.get("Similar OPS"), 0) >= 0.800 and _safe_number(row.get("Similar PA"), 0) >= 10:
            reasons.append("similar-SP fit")
        if _safe_number(row.get("OPS"), 0) >= 0.800:
            reasons.append("season OPS")
        if not reasons:
            reasons.append("lineup value")
        platoon_label = "N/A"
        if split_ops_col and split_pa_col:
            pa = int(_safe_number(row.get(split_pa_col), 0))
            ops_value = _safe_number(row.get(split_ops_col), None)
            hand = split_ops_col.replace("OPS vs ", "")
            platoon_label = f"{ops_value:.3f} OPS vs {hand} / {pa} PA" if ops_value is not None and pa else f"small vs {hand}: {pa} PA"
        rows.append({
            "Team": team,
            "Name": row.get("Name"),
            "Priority": round(_safe_number(row.get("Watch Score"), 0), 1),
            "Why": ", ".join(reasons),
            "OPS": row.get("OPS"),
            "ISO": row.get("ISO"),
            "Platoon": platoon_label,
            "Similar Sample": (
                f"{_safe_number(row.get('Similar OPS'), 0):.3f} OPS / {int(_safe_number(row.get('Similar PA'), 0))} PA"
                if _safe_number(row.get("Similar PA"), 0) >= 10
                else f"small: {int(_safe_number(row.get('Similar PA'), 0))} PA"
            ),
            "OPS Pctl": row.get("OPS Pctl", ""),
            "xwOBA Pctl": row.get("xwOBA Pctl", ""),
            "HardHit Pctl": row.get("HardHit Pctl", ""),
            "Contact Pctl": row.get("Contact Pctl", ""),
        })
    return pd.DataFrame(rows, columns=columns)


def _format_ab_ops_hr(ab, ops, hr):
    ab = int(_safe_number(ab, 0))
    hr = int(_safe_number(hr, 0))
    ops_value = _safe_number(ops, None)
    if ab <= 0 or ops_value is None:
        return "0 AB"
    return f"{ab} AB / {ops_value:.3f} OPS / {hr} HR"


# At-bats at which a split OPS is about half-reliable, so a sample of that size is believed
# halfway and the rest is pulled back to the hitter's own season line. OBP stabilizes around
# 460 PA and SLG around 320 AB, so ~300 is the honest figure for the pair of them.
#
# This exists because picking the *maximum* OPS across five samples of wildly different size
# is a bias, not a read: the largest of several noisy estimates is systematically the
# luckiest small one. Measured over 3,539 hitter rows from 203 cached games, the old
# max-selection reported a .921 OPS for hitters whose season OPS was .730 -- 192 points of
# pure selection bias, rising to +439 when it crowned a BvP line on a median of 9 at-bats.
HITTER_SPLIT_SHRINK_AB = 300.0


def _shrunk_split_ops(ops, at_bats, season_ops):
    """A split OPS pulled toward the hitter's own season line by its sample size.

    Returns (shrunk OPS, deviation from season). An 8-at-bat split moves the estimate about
    2.5% of the way to what it observed; a 150-at-bat platoon split moves it a third.
    """
    ops = _safe_number(ops, None)
    season_ops = _safe_number(season_ops, None)
    at_bats = max(0.0, _safe_number(at_bats, 0) or 0.0)
    if ops is None or season_ops is None or not at_bats:
        return None, None
    shrunk = (ops * at_bats + season_ops * HITTER_SPLIT_SHRINK_AB) / (at_bats + HITTER_SPLIT_SHRINK_AB)
    return shrunk, shrunk - season_ops


# Per-source belief in the *construct*, once sample size is handled by shrinkage above.
# Set from what each one was measured to be worth, not from how interesting it looks:
# platoon splits are a real and large effect; the individual hitter's arsenal fit came back
# at t = 1.5 on DK points over 100,372 player-games; comp-pitcher "similar" ranks starters at
# a coin flip; and batter-vs-pitcher history is t = 0.66, turning negative where the sample
# reaches ten plate appearances. BvP is therefore shown and not scored.
# See docs/arsenal_study.md.
#
# Scale note: these multiply a deviation from the hitter's *own season line*, where the old
# weights multiplied a deviation from a flat .700. That is a far smaller quantity, so the
# numbers are larger to keep Composite on its original scale and the Priority/Watch/Fade
# thresholds meaningful. A 250-at-bat platoon split 200 points above a hitter's season now
# moves him ~14 points, close to the ~11 it moved before; an 8-at-bat arsenal split 850
# points above his season moves him ~1, where it used to move him ~41.
HITTER_SPLIT_WEIGHTS = {"Platoon": 150.0, "Arsenal": 50.0, "Similar": 20.0, "BvP": 0.0}

_HITTER_SPLIT_SPECS = [
    ("Platoon", "Platoon AB", "Platoon OPS", "Platoon HR"),
    ("Arsenal", "Arsenal AB", "Arsenal OPS", "Arsenal HR"),
    ("Similar", "Similar AB", "Similar OPS", "Similar HR"),
    ("BvP", "BvP AB", "BvP OPS", "BvP HR"),
]


def _best_hitter_signal(row):
    """The most notable split for this hitter, after shrinking each one for sample size.

    Reports the raw line the reader can check *and* how much of it survives regression, so
    a 1.200 OPS on nine at-bats is visible as the +.011 it is actually worth.
    """
    season_ops = _safe_number(row.get("Season OPS"), None)
    season_ab = int(_safe_number(row.get("Season AB"), 0))
    if season_ops is None:
        return f"Thin samples / {season_ab} season AB"

    candidates = []
    for label, ab_col, ops_col, hr_col in _HITTER_SPLIT_SPECS:
        # A split we do not score is a split we do not headline either. BvP is shown in its
        # own column and note, but naming it a hitter's best signal would assert something
        # the measurement says is not there.
        if not HITTER_SPLIT_WEIGHTS.get(label, 0.0):
            continue
        at_bats = int(_safe_number(row.get(ab_col), 0))
        _, deviation = _shrunk_split_ops(row.get(ops_col), at_bats, season_ops)
        if deviation is None:
            continue
        candidates.append((abs(deviation), deviation, label, at_bats,
                           _safe_number(row.get(ops_col), 0.0),
                           int(_safe_number(row.get(hr_col), 0))))
    if not candidates:
        return f"Season: {season_ops:.3f} OPS / {season_ab} AB"

    _, deviation, label, at_bats, raw_ops, hr = max(candidates, key=lambda item: item[0])
    # Below a point of OPS the split is not saying anything the season line does not.
    if abs(deviation) < 0.010:
        return f"Season: {season_ops:.3f} OPS / {season_ab} AB (no split stands out)"
    return (f"{label}: {raw_ops:.3f} OPS / {at_bats} AB / {hr} HR "
            f"({deviation:+.3f} vs season after regression)")


def build_hitter_composite_table(team, lineup_df, splits_df=None, similar_df=None, arsenal_df=None, bvp_df=None):
    columns = [
        "Team", "Rank", "Name", "Bats",
        "Season", "Platoon", "Arsenal", "Similar", "BvP", "Best Signal",
        "Season AB", "Season OPS", "Season HR",
        "Platoon AB", "Platoon OPS", "Platoon HR",
        "Arsenal AB", "Arsenal OPS", "Arsenal HR",
        "Similar AB", "Similar OPS", "Similar HR",
        "BvP AB", "BvP OPS", "BvP HR",
        "Composite", "Signal",
    ]
    if lineup_df is None or lineup_df.empty:
        return pd.DataFrame(columns=columns)

    base_cols = [col for col in ["Name", "Bats", "AB", "OPS", "HR"] if col in lineup_df.columns]
    out = lineup_df[base_cols].copy()
    if "AB" not in out.columns:
        out["AB"] = 0
    if "OPS" not in out.columns:
        out["OPS"] = 0.0
    if "HR" not in out.columns:
        out["HR"] = 0
    out = out.rename(columns={"AB": "Season AB", "OPS": "Season OPS", "HR": "Season HR"})

    if splits_df is not None and not splits_df.empty and "Name" in splits_df.columns:
        split_cols = ["Name"]
        for prefix in ["AB vs", "OPS vs", "HR vs"]:
            matches = [col for col in splits_df.columns if col.startswith(prefix)]
            if matches:
                split_cols.append(matches[0])
        split = splits_df[split_cols].copy()
        rename_map = {}
        for col in split.columns:
            if col.startswith("AB vs"):
                rename_map[col] = "Platoon AB"
            elif col.startswith("OPS vs"):
                rename_map[col] = "Platoon OPS"
            elif col.startswith("HR vs"):
                rename_map[col] = "Platoon HR"
        out = out.merge(split.rename(columns=rename_map), on="Name", how="left")

    if arsenal_df is not None and not arsenal_df.empty and "Name" in arsenal_df.columns:
        arsenal = arsenal_df[[col for col in ["Name", "AB", "OPS", "HR"] if col in arsenal_df.columns]].copy()
        out = out.merge(arsenal.rename(columns={"AB": "Arsenal AB", "OPS": "Arsenal OPS", "HR": "Arsenal HR"}), on="Name", how="left")

    if similar_df is not None and not similar_df.empty and "Name" in similar_df.columns:
        similar = similar_df[[col for col in ["Name", "AB", "OPS_proxy", "HR"] if col in similar_df.columns]].copy()
        out = out.merge(similar.rename(columns={"AB": "Similar AB", "OPS_proxy": "Similar OPS", "HR": "Similar HR"}), on="Name", how="left")

    if bvp_df is not None and not bvp_df.empty and "Name" in bvp_df.columns:
        bvp = bvp_df[[col for col in ["Name", "AB", "OPS", "HR"] if col in bvp_df.columns]].copy()
        out = out.merge(bvp.rename(columns={"AB": "BvP AB", "OPS": "BvP OPS", "HR": "BvP HR"}), on="Name", how="left")

    for col in columns:
        if col not in out.columns and col not in {"Team", "Composite", "Signal"}:
            out[col] = 0 if col.endswith(("AB", "HR")) else ""

    rows = []
    for _, row in out.iterrows():
        season_ops = _safe_number(row.get("Season OPS"), 0)
        platoon_ops = _safe_number(row.get("Platoon OPS"), None)
        arsenal_ops = _safe_number(row.get("Arsenal OPS"), None)
        similar_ops = _safe_number(row.get("Similar OPS"), None)
        bvp_ops = _safe_number(row.get("BvP OPS"), None)
        season_hr = _safe_number(row.get("Season HR"), 0)
        arsenal_ab = _safe_number(row.get("Arsenal AB"), 0)
        similar_ab = _safe_number(row.get("Similar AB"), 0)
        platoon_ab = _safe_number(row.get("Platoon AB"), 0)
        bvp_ab = _safe_number(row.get("BvP AB"), 0)

        # Each split enters as its *shrunk deviation from this hitter's own season line*,
        # scaled by how much the construct is worth. The old version added the raw split
        # once it cleared a minimum-at-bat gate, which gave an 8-at-bat arsenal sample a
        # heavier weight (55) than a 20-at-bat platoon split (45) and let a lucky fortnight
        # move a hitter two tiers.
        score = (season_ops - 0.700) * 95 + min(season_hr, 25) * 1.2
        for label, ab_col, ops_col, _hr_col in _HITTER_SPLIT_SPECS:
            weight = HITTER_SPLIT_WEIGHTS.get(label, 0.0)
            if not weight:
                continue
            _, deviation = _shrunk_split_ops(row.get(ops_col),
                                             _safe_number(row.get(ab_col), 0), season_ops)
            if deviation is not None:
                score += deviation * weight

        if score >= 45:
            signal = "Priority"
        elif score >= 20:
            signal = "Watch"
        elif score <= -15:
            signal = "Fade"
        else:
            signal = "Neutral"

        rows.append({
            "Team": team,
            "Name": row.get("Name"),
            "Bats": row.get("Bats", ""),
            "Season AB": int(_safe_number(row.get("Season AB"), 0)),
            "Season OPS": round(season_ops, 3),
            "Season HR": int(season_hr),
            "Platoon AB": int(platoon_ab),
            "Platoon OPS": round(platoon_ops, 3) if platoon_ops is not None else "",
            "Platoon HR": int(_safe_number(row.get("Platoon HR"), 0)),
            "Arsenal AB": int(arsenal_ab),
            "Arsenal OPS": round(arsenal_ops, 3) if arsenal_ops is not None else "",
            "Arsenal HR": int(_safe_number(row.get("Arsenal HR"), 0)),
            "Similar AB": int(similar_ab),
            "Similar OPS": round(similar_ops, 3) if similar_ops is not None else "",
            "Similar HR": int(_safe_number(row.get("Similar HR"), 0)),
            "BvP AB": int(bvp_ab),
            "BvP OPS": round(bvp_ops, 3) if bvp_ops is not None else "",
            "BvP HR": int(_safe_number(row.get("BvP HR"), 0)),
            "Composite": round(score, 1),
            "Signal": signal,
        })

    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(columns=columns)
    table = table.sort_values(["Composite", "Season OPS"], ascending=False).reset_index(drop=True)
    table["Rank"] = range(1, len(table) + 1)
    table["Season"] = table.apply(lambda row: _format_ab_ops_hr(row.get("Season AB"), row.get("Season OPS"), row.get("Season HR")), axis=1)
    table["Platoon"] = table.apply(lambda row: _format_ab_ops_hr(row.get("Platoon AB"), row.get("Platoon OPS"), row.get("Platoon HR")), axis=1)
    table["Arsenal"] = table.apply(lambda row: _format_ab_ops_hr(row.get("Arsenal AB"), row.get("Arsenal OPS"), row.get("Arsenal HR")), axis=1)
    table["Similar"] = table.apply(lambda row: _format_ab_ops_hr(row.get("Similar AB"), row.get("Similar OPS"), row.get("Similar HR")), axis=1)
    table["BvP"] = table.apply(lambda row: _format_ab_ops_hr(row.get("BvP AB"), row.get("BvP OPS"), row.get("BvP HR")), axis=1)
    table["Best Signal"] = table.apply(_best_hitter_signal, axis=1)
    return table[[col for col in columns if col in table.columns]]


def summarize_hitter_composite(hitter_composite):
    columns = ["Team", "Priority Bats", "Best Arsenal Fit", "Best Similar Fit", "BvP Note"]
    if hitter_composite is None or hitter_composite.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for team, team_df in hitter_composite.groupby("Team", sort=False):
        ranked = team_df.sort_values(["Composite", "Season OPS"], ascending=False)
        top_names = ranked.head(3)["Name"].dropna().astype(str).tolist()

        def best_by_shrunk(ab_column, ops_column, display_column, empty_text):
            """The most notable hitter on a split, after regressing it for sample size.

            Sorting on the raw split OPS picked whoever had the fewest at-bats and the most
            luck, which is how "Best Arsenal Fit" kept naming a hitter off eight at-bats.
            """
            pool = ranked.copy()
            deviations = []
            for _, entry in pool.iterrows():
                _, deviation = _shrunk_split_ops(entry.get(ops_column), entry.get(ab_column),
                                                 entry.get("Season OPS"))
                deviations.append(deviation if deviation is not None else np.nan)
            pool["_dev"] = deviations
            pool = pool[pool["_dev"].notna() & (pool["_dev"] > 0)]
            if pool.empty:
                return empty_text
            best = pool.sort_values("_dev", ascending=False).iloc[0]
            return (f"{best.get('Name')}: {best.get(display_column)} "
                    f"({best['_dev']:+.3f} vs season)")

        arsenal_note = best_by_shrunk("Arsenal AB", "Arsenal OPS", "Arsenal", "No strong sample")
        similar_note = best_by_shrunk("Similar AB", "Similar OPS", "Similar", "No strong sample")
        # Kept as a note and deliberately not scored anywhere: measured at t = 0.66 on DK
        # points, and negative where the sample reaches ten plate appearances.
        bvp_note = best_by_shrunk("BvP AB", "BvP OPS", "BvP", "No meaningful history")

        rows.append({
            "Team": team,
            "Priority Bats": ", ".join(top_names) if top_names else "N/A",
            "Best Arsenal Fit": arsenal_note,
            "Best Similar Fit": similar_note,
            "BvP Note": bvp_note,
        })
    return pd.DataFrame(rows, columns=columns)


def _mean_numeric(df, col, default=0.0):
    if df is None or df.empty or col not in df.columns:
        return default
    value = pd.to_numeric(df[col], errors="coerce").mean()
    return default if pd.isna(value) else float(value)


def _top_mean_numeric(df, col, n=3, default=0.0, ascending=False):
    top = _top_rows(df, col, n=n, ascending=ascending)
    return _mean_numeric(top, col, default=default)


def load_model_calibration(path=CALIBRATION_PATH):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def model_calibration_provenance(calibration=None, path=CALIBRATION_PATH):
    """Stable identity for the exact game-model artifact applied to a report cache."""
    calibration = calibration if calibration is not None else load_model_calibration(path)
    if not calibration:
        return {
            "fingerprint": "none",
            "feature_version": None,
            "created_at": None,
        }
    encoded = json.dumps(
        calibration, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return {
        "fingerprint": hashlib.sha256(encoded).hexdigest()[:16],
        "feature_version": calibration.get("feature_version"),
        "created_at": calibration.get("created_at"),
    }


def build_game_scorecard(
    home_team, away_team,
    home_starter_info, away_starter_info,
    home_lineup_df, away_lineup_df,
    home_bullpen_df, away_bullpen_df,
    home_similarity_df, away_similarity_df,
    environment,
):
    park_runs = environment.get("park", {}).get("Runs", 1.0) if isinstance(environment, dict) else 1.0

    def bullpen_strength(bullpen_df):
        score = _top_mean_numeric(bullpen_df, "Bullpen Score", n=3, default=15.0)
        workload = _mean_numeric(bullpen_df, "Last3D_Pitches", 0.0)
        prevention_edge = (score - 15.0) / 16.0 - max(0, workload - 22) / 35.0
        run_adj = -((score - 15.0) / 22.0) + max(0, workload - 22) / 45.0
        return score, workload, prevention_edge, run_adj

    def team_components(lineup_df, sim_df, opp_starter, opp_bullpen, own_bullpen):
        lineup_ops = _mean_numeric(lineup_df, "OPS", 0.700)
        top_iso = _top_mean_numeric(lineup_df, "ISO", n=4, default=0.140)
        sim_ops = _mean_numeric(sim_df[sim_df["PA"] >= 8] if sim_df is not None and not sim_df.empty and "PA" in sim_df.columns else sim_df, "OPS_proxy", lineup_ops)
        opp_fip = _safe_number(opp_starter.get("FIP") if isinstance(opp_starter, dict) else None, 4.25)
        _, _, bullpen_edge, _ = bullpen_strength(own_bullpen)
        _, _, _, opp_bullpen_run_adj = bullpen_strength(opp_bullpen)
        offense = ((lineup_ops - 0.700) * 7.0) + ((top_iso - 0.140) * 8.0) + ((sim_ops - 0.700) * 3.5)
        starter_edge = (opp_fip - 4.20) * 0.70
        park_multiplier = 0.65 + (0.35 * park_runs)
        expected_runs = (4.35 + offense + starter_edge + opp_bullpen_run_adj) * park_multiplier
        return {
            "Lineup OPS": lineup_ops,
            "Top ISO": top_iso,
            "Similar OPS": sim_ops,
            "Opp FIP": opp_fip,
            "Bullpen Edge": bullpen_edge,
            "Expected Runs": max(2.1, min(7.4, expected_runs)),
            "Edge Score": offense + starter_edge + bullpen_edge,
        }

    away = team_components(away_lineup_df, away_similarity_df, home_starter_info, home_bullpen_df, away_bullpen_df)
    home = team_components(home_lineup_df, home_similarity_df, away_starter_info, away_bullpen_df, home_bullpen_df)
    home["Edge Score"] += 0.18
    home["Expected Runs"] += 0.08

    diff = home["Edge Score"] - away["Edge Score"]
    home_win = 1 / (1 + np.exp(-diff / 1.65))
    away_win = 1 - home_win
    raw_total = away["Expected Runs"] + home["Expected Runs"]

    calibration = load_model_calibration()
    if calibration:
        win_model = calibration.get("win_model", {})
        total_model = calibration.get("total_model", {})
        if win_model.get("features"):
            feature_values = {
                "edge_diff": diff,
                "raw_total": raw_total,
                "home_lineup_ops": home["Lineup OPS"],
                "away_lineup_ops": away["Lineup OPS"],
                "home_bullpen_edge": home["Bullpen Edge"],
                "away_bullpen_edge": away["Bullpen Edge"],
            }
            z = win_model.get("intercept", 0.0)
            for feature, coef in zip(win_model.get("features", []), win_model.get("coef", [])):
                z += feature_values.get(feature, 0.0) * coef
            home_win = 1 / (1 + np.exp(-z))
            away_win = 1 - home_win
        if total_model.get("features"):
            feature_values = {
                "raw_total": raw_total,
                "park_runs": environment.get("park", {}).get("Runs", 1.0) if isinstance(environment, dict) else 1.0,
                "home_lineup_ops": home["Lineup OPS"],
                "away_lineup_ops": away["Lineup OPS"],
            }
            # Calibration is trained with these exact numeric transformations.  Without
            # this update, the report silently supplied zero for every fitted weather and
            # umpire coefficient even though those fields were present in the environment.
            feature_values.update(_weather_calibration_features(environment))
            model_total = total_model.get("intercept", raw_total)
            for feature, coef in zip(total_model.get("features", []), total_model.get("coef", [])):
                model_total += feature_values.get(feature, 0.0) * coef
            model_total = max(6.5, min(12.5, model_total))
            regressed_total = 8.8 + ((raw_total - 8.8) * 0.72)
            calibrated_total = (regressed_total * 0.75) + (model_total * 0.25)
            if raw_total >= 10.5 and calibrated_total > raw_total:
                calibrated_total = raw_total
            if raw_total > 0:
                scale = max(0.86, min(1.04, calibrated_total / raw_total))
                home["Expected Runs"] = max(2.1, min(7.4, home["Expected Runs"] * scale))
                away["Expected Runs"] = max(2.1, min(7.4, away["Expected Runs"] * scale))
                raw_total = home["Expected Runs"] + away["Expected Runs"]

    rows = []
    for team, comp, win_pct in [(away_team, away, away_win), (home_team, home, home_win)]:
        rows.append({
            "Team": team,
            "Win Lean": f"{win_pct * 100:.0f}%",
            "Exp Runs": round(comp["Expected Runs"], 1),
            "Edge": round(comp["Edge Score"], 2),
            "Lineup": _percentile_badge(70 if comp["Lineup OPS"] >= 0.760 else 55 if comp["Lineup OPS"] >= 0.710 else 35),
            "Similar SP": _percentile_badge(75 if comp["Similar OPS"] >= 0.820 else 55 if comp["Similar OPS"] >= 0.720 else 35),
            "Opp SP FIP": round(comp["Opp FIP"], 2),
            "Bullpen": _percentile_badge(75 if comp["Bullpen Edge"] >= 0.5 else 55 if comp["Bullpen Edge"] >= 0 else 35),
        })

    scorecard = pd.DataFrame(rows)
    total_runs = round(sum(row["Exp Runs"] for row in rows), 1)
    winner = home_team if home_win >= away_win else away_team
    if calibration:
        reliability = calibration.get("reliability", "unknown")
        win_signal = (
            calibration.get("validation", {})
            .get("signal_analysis", {})
            .get("win", {})
        )
        umpire_signal = (
            calibration.get("validation", {})
            .get("signal_analysis", {})
            .get("context", {})
            .get("components", {})
            .get("umpire_tendency", {})
        )
        magnitude_note = "win magnitude validated" if win_signal.get("magnitude_validated") else "win magnitude directional"
        umpire_note = "" if umpire_signal.get("magnitude_validated") else "; umpire experimental"
        model_note = f"calibrated/regressed ({reliability}; {magnitude_note}{umpire_note})"
    else:
        model_note = "heuristic"
    provenance = model_calibration_provenance(calibration)
    return scorecard, {
        "Projected Winner": winner,
        "Total Runs": total_runs,
        "Home Win%": round(home_win * 100, 1),
        "Away Win%": round(away_win * 100, 1),
        "Model": model_note,
        "Calibration Fingerprint": provenance["fingerprint"],
        "Calibration Feature Version": provenance["feature_version"],
        "Calibration Created At": provenance["created_at"],
    }


TRUSTED_TIME_ZONE_BASELINE = {
    "Name": "Song, Severini & Allada (PNAS 2017)",
    "Rule": "eastward residual >=2h",
    "Home Penalty pp": -3.5,
    "Away Penalty pp": -2.1,
}


def _trusted_time_zone_penalty_pp(context, team_role):
    """Return the accepted study penalty for an exposed team, in win-probability points."""
    context = context or {}
    residual = _safe_number(context.get("Residual Hours"), None)
    if (
        str(context.get("Direction", "")).lower() != "east"
        or residual is None
        or abs(residual) < 2.0
    ):
        return 0.0
    key = "Home Penalty pp" if team_role == "home" else "Away Penalty pp"
    return float(TRUSTED_TIME_ZONE_BASELINE[key])


def apply_trusted_time_zone_baseline(
    scorecard,
    projection_summary,
    away_team,
    home_team,
    away_time_zone,
    home_time_zone,
):
    """Apply the accepted study coefficients once, after the core projection.

    A home-team penalty directly lowers home win probability. An away-team
    penalty lowers away win probability and therefore raises home probability.
    The pre-adjustment probabilities are retained so cached reports can be
    upgraded idempotently and every output can show the model impact.
    """
    summary = dict(projection_summary or {})
    result = scorecard.copy() if isinstance(scorecard, pd.DataFrame) else pd.DataFrame()

    pre_home = _safe_number(
        summary.get("Pre-TZ Home Win%", summary.get("Home Win%")), None
    )
    pre_away = _safe_number(
        summary.get("Pre-TZ Away Win%", summary.get("Away Win%")), None
    )
    if pre_home is None:
        return result, summary
    if pre_away is None:
        pre_away = 100.0 - pre_home

    home_penalty = _trusted_time_zone_penalty_pp(home_time_zone, "home")
    away_penalty = _trusted_time_zone_penalty_pp(away_time_zone, "away")
    net_home_pp = home_penalty - away_penalty
    if abs(net_home_pp) < 0.05:
        net_home_pp = 0.0
    net_away_pp = -net_home_pp if net_home_pp else 0.0
    adjusted_home = max(5.0, min(95.0, pre_home + net_home_pp))
    adjusted_away = 100.0 - adjusted_home

    for context, role, penalty in [
        (away_time_zone, "Away", away_penalty),
        (home_time_zone, "Home", home_penalty),
    ]:
        if isinstance(context, dict):
            context["Study Role"] = role
            context["Study Penalty pp"] = round(penalty, 1)
            context["Study"] = f"{penalty:+.1f}pp" if penalty else "0.0pp"

    summary.update({
        "Pre-TZ Home Win%": round(pre_home, 1),
        "Pre-TZ Away Win%": round(pre_away, 1),
        "Home Win%": round(adjusted_home, 1),
        "Away Win%": round(adjusted_away, 1),
        "Projected Winner": home_team if adjusted_home >= adjusted_away else away_team,
        "TZ Home Penalty pp": round(home_penalty, 1),
        "TZ Away Penalty pp": round(away_penalty, 1),
        "TZ Net Home pp": round(net_home_pp, 1),
        "TZ Net Away pp": round(net_away_pp, 1),
        "Time-Zone Adjustment": f"{home_team} {net_home_pp:+.1f}pp / {away_team} {net_away_pp:+.1f}pp",
        "TZ Baseline": (
            f"{TRUSTED_TIME_ZONE_BASELINE['Name']}; "
            f"{TRUSTED_TIME_ZONE_BASELINE['Rule']}"
        ),
    })
    model_note = str(summary.get("Model", "heuristic"))
    if "east-TZ baseline" not in model_note:
        summary["Model"] = f"{model_note} + east-TZ baseline"

    if not result.empty and "Team" in result.columns:
        result["Win Lean"] = result["Team"].map({
            away_team: f"{adjusted_away:.0f}%",
            home_team: f"{adjusted_home:.0f}%",
        }).fillna(result.get("Win Lean", ""))
        result["TZ Study"] = result["Team"].map({
            away_team: f"{away_penalty:+.1f}pp" if away_penalty else "0.0pp",
            home_team: f"{home_penalty:+.1f}pp" if home_penalty else "0.0pp",
        }).fillna("")
    return result, summary


def refresh_payload_model_calibration(payload, force=False):
    """Rebuild calibration-dependent cache fields without any network requests.

    Returns ``(payload, changed, provenance)``.  DFS calls this before projecting a
    cached game, so a stale scorecard cannot silently feed old expected runs and win
    probabilities into the optimizer.
    """
    if not isinstance(payload, dict):
        raise ValueError("report cache payload is not a dictionary")
    args = payload.get("report_args")
    context = payload.get("advanced_context")
    if not isinstance(args, (tuple, list)) or len(args) <= 16:
        raise ValueError("report cache has no compatible report_args")
    if not isinstance(context, dict):
        raise ValueError("report cache has no advanced_context")

    calibration = load_model_calibration()
    if not calibration:
        raise RuntimeError(
            f"model calibration artifact is unavailable or invalid: {CALIBRATION_PATH}"
        )
    current = model_calibration_provenance(calibration)
    summary = context.get("projection_summary") or {}
    cached = payload.get("model_calibration") or {
        "fingerprint": summary.get("Calibration Fingerprint", "unknown"),
        "feature_version": summary.get("Calibration Feature Version"),
        "created_at": summary.get("Calibration Created At"),
    }
    if not force and cached.get("fingerprint") == current["fingerprint"]:
        payload["model_calibration"] = current
        return payload, False, current

    home_team, away_team = args[15], args[16]
    environment = context.get("environment") or {}
    fresh_scorecard, fresh_summary = build_game_scorecard(
        home_team, away_team,
        args[2], args[9],
        args[0], args[7],
        args[1], args[8],
        context.get("home_similarity"), context.get("away_similarity"),
        environment,
    )
    fresh_scorecard, fresh_summary = apply_trusted_time_zone_baseline(
        fresh_scorecard,
        fresh_summary,
        away_team,
        home_team,
        context.get("away_time_zone", {}),
        context.get("home_time_zone", {}),
    )

    existing = context.get("scorecard")
    if isinstance(existing, pd.DataFrame) and not existing.empty and "Team" in existing:
        refreshed = existing.copy()
        indexed = fresh_scorecard.set_index("Team") if not fresh_scorecard.empty else pd.DataFrame()
        for column in ("Win Lean", "Exp Runs", "Edge", "Lineup", "Similar SP",
                       "Opp SP FIP", "Bullpen", "TZ Study"):
            if not indexed.empty and column in indexed:
                refreshed[column] = refreshed["Team"].map(indexed[column]).fillna(
                    refreshed.get(column, "")
                )
    else:
        refreshed = fresh_scorecard

    merged_summary = dict(summary)
    merged_summary.update(fresh_summary)
    context["scorecard"] = refreshed
    context["projection_summary"] = merged_summary

    # Keep report diagnostics consistent with the scorecard the optimizer consumes.
    try:
        context["factor_matrix"] = build_factor_matrix(
            away_team, home_team, refreshed,
            args[9], args[2], args[8], args[1],
            context.get("away_vs_home_arsenal"), context.get("home_vs_away_arsenal"),
            context.get("away_recent_summary"), context.get("home_recent_summary"),
            context.get("away_park_defense"), context.get("home_park_defense"),
            environment,
        )
    except Exception:
        # The optimizer only consumes scorecard/environment; old report payloads may lack
        # a newer diagnostic table, which must not prevent a valid projection refresh.
        pass
    context["uncertainty_flags"] = generate_uncertainty_flags(
        refreshed, away_team, home_team,
        args[7], args[0],
        context.get("away_type_results"), context.get("home_type_results"),
        args[8], args[1], merged_summary,
    )
    payload["advanced_context"] = context
    payload["model_calibration"] = current
    return payload, True, current


def _lineup_k_edge_points(batter_arsenal_df, lineup_df):
    """Lineup-level arsenal K edge in percentage points, or None.

    Shares `dfs.projections.lineup_arsenal_k_edge` with the DFS projection rather than
    re-deriving it, so the number printed in the report is the number that moved the
    starter's projection. Imported lazily to keep the report importable without dfs.
    """
    try:
        from dfs.projections import lineup_arsenal_k_edge
    except Exception:
        return None
    edge, hitters = lineup_arsenal_k_edge(batter_arsenal_df, lineup_df)
    return None if edge is None else (edge * 100, hitters)


def build_pitcher_watchlist(home_team, away_team, home_starter_info, away_starter_info,
                            away_vs_home_arsenal=None, home_vs_away_arsenal=None,
                            environment=None, away_batter_arsenal=None,
                            home_batter_arsenal=None, away_lineup_df=None,
                            home_lineup_df=None):
    # "Opponent Arsenal" is the OPS-based read, which a three-season test found to be the
    # weak half of this measurement (docs/arsenal_study.md). "K Edge" is the half that
    # survived, so it is shown beside it rather than instead of it -- one is context, the
    # other is the number the projection actually uses.
    columns = ["Team", "Pitcher", "Role", "Why", "FIP", "K-BB", "Opponent Arsenal",
               "K Edge", "Score"]
    rows = []
    park_runs = environment.get("park", {}).get("Runs", 1.0) if isinstance(environment, dict) else 1.0
    for team, starter, opponent_arsenal, opp_batter_arsenal, opp_lineup in [
        (away_team, away_starter_info, home_vs_away_arsenal, home_batter_arsenal, home_lineup_df),
        (home_team, home_starter_info, away_vs_home_arsenal, away_batter_arsenal, away_lineup_df),
    ]:
        if isinstance(starter, dict):
            kbb = (_safe_number(starter.get("K%"), 0) * (100 if abs(_safe_number(starter.get("K%"), 0)) <= 1 else 1)) - (_safe_number(starter.get("BB%"), 0) * (100 if abs(_safe_number(starter.get("BB%"), 0)) <= 1 else 1))
            fip = _safe_number(starter.get("FIP"), 4.25)
            arsenal_row = opponent_arsenal.iloc[0].to_dict() if opponent_arsenal is not None and not opponent_arsenal.empty else {}
            opp_ops = _safe_number(arsenal_row.get("OPS"), 0.720)
            opp_tag = arsenal_row.get("Arsenal Tag", "N/A")
            measured = _lineup_k_edge_points(opp_batter_arsenal, opp_lineup)
            k_edge, k_edge_hitters = measured if measured else (None, 0)
            score = (18 - fip * 2.2) + (kbb * 0.7) + ((0.720 - opp_ops) * 45) + ((1.0 - park_runs) * 5)
            if k_edge is not None:
                # Scaled off the fitted K-rate pass-through: 0.45 of the edge reaches his
                # real K rate, at ~24 batters faced and 2 DK points a strikeout, so a point
                # of edge is worth about 0.2 DK. Kept small on purpose -- it is a tiebreak
                # between comparable arms, not a reason to start a bad one.
                score += k_edge * 0.22
            if k_edge is not None and k_edge >= 1.5:
                why = "lineup whiffs vs his shapes"
            elif fip <= 3.80 and opp_ops <= 0.720:
                why = "starter run-prevention fit"
            elif opp_ops <= 0.660:
                why = "opponent struggles vs arsenal"
            elif k_edge is not None and k_edge <= -1.5:
                why = "lineup makes contact vs his shapes"
            elif fip <= 3.50:
                why = "starter form"
            else:
                why = "monitor, not a clear positive edge"
            rows.append({
                "Team": team,
                "Pitcher": starter.get("Name"),
                "Role": "Starter",
                "Why": why,
                "FIP": round(fip, 2),
                "K-BB": round(kbb, 1),
                "Opponent Arsenal": f"{opp_tag} / {opp_ops:.3f} OPS",
                "K Edge": (f"{k_edge:+.1f} pts ({k_edge_hitters} bats)"
                           if k_edge is not None else "n/a"),
                "Score": round(score, 1),
            })
    return pd.DataFrame(rows, columns=columns).sort_values("Score", ascending=False)


def _scorecard_row(scorecard, team):
    if scorecard is None or scorecard.empty or "Team" not in scorecard.columns:
        return {}
    rows = scorecard[scorecard["Team"].astype(str).eq(team)]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def _lean_from_diff(away_team, home_team, away_value, home_value, higher_is_better=True, threshold=0.05):
    away_value = _safe_number(away_value, 0)
    home_value = _safe_number(home_value, 0)
    diff = home_value - away_value if higher_is_better else away_value - home_value
    if abs(diff) < threshold:
        return "Even", diff
    return (home_team if diff > 0 else away_team), diff


def build_factor_matrix(
    away_team, home_team, scorecard,
    away_starter_info, home_starter_info,
    away_bullpen_df, home_bullpen_df,
    away_vs_home_arsenal, home_vs_away_arsenal,
    away_recent_summary, home_recent_summary,
    away_park_defense, home_park_defense,
    environment,
):
    columns = ["Factor", "Edge", "Confidence", "Why"]
    away_row = _scorecard_row(scorecard, away_team)
    home_row = _scorecard_row(scorecard, home_team)
    rows = []

    away_win = _safe_number(str(away_row.get("Win Lean", "")).replace("%", ""), 50)
    home_win = _safe_number(str(home_row.get("Win Lean", "")).replace("%", ""), 50)
    projected_edge = home_team if home_win >= away_win else away_team
    rows.append({
        "Factor": "Overall projection",
        "Edge": projected_edge,
        "Confidence": "High" if max(away_win, home_win) >= 62 else "Medium" if max(away_win, home_win) >= 56 else "Low",
        "Why": f"{away_team} {away_row.get('Win Lean', 'N/A')} / {home_team} {home_row.get('Win Lean', 'N/A')}; expected runs {away_row.get('Exp Runs', 'N/A')}-{home_row.get('Exp Runs', 'N/A')}.",
    })

    away_opp_fip = _safe_number(away_row.get("Opp SP FIP"), 4.25)
    home_opp_fip = _safe_number(home_row.get("Opp SP FIP"), 4.25)
    starter_edge, starter_diff = _lean_from_diff(away_team, home_team, away_opp_fip, home_opp_fip, higher_is_better=True, threshold=0.25)
    rows.append({
        "Factor": "Starter matchup",
        "Edge": starter_edge,
        "Confidence": "High" if abs(starter_diff) >= 0.75 else "Medium" if abs(starter_diff) >= 0.35 else "Low",
        "Why": f"{away_team} faces {_first_name(home_starter_info)} FIP {away_opp_fip:.2f}; {home_team} faces {_first_name(away_starter_info)} FIP {home_opp_fip:.2f}.",
    })

    away_ars = away_vs_home_arsenal.iloc[0].to_dict() if away_vs_home_arsenal is not None and not away_vs_home_arsenal.empty else {}
    home_ars = home_vs_away_arsenal.iloc[0].to_dict() if home_vs_away_arsenal is not None and not home_vs_away_arsenal.empty else {}
    away_ops = _safe_number(away_ars.get("OPS"), 0.700)
    home_ops = _safe_number(home_ars.get("OPS"), 0.700)
    arsenal_edge, arsenal_diff = _lean_from_diff(away_team, home_team, away_ops, home_ops, higher_is_better=True, threshold=0.035)
    rows.append({
        "Factor": "Lineup vs arsenal",
        "Edge": arsenal_edge,
        "Confidence": "High" if abs(arsenal_diff) >= 0.090 else "Medium" if abs(arsenal_diff) >= 0.045 else "Low",
        "Why": f"{away_team}: {away_ars.get('Arsenal Tag', 'N/A')} ({away_ops:.3f} OPS); {home_team}: {home_ars.get('Arsenal Tag', 'N/A')} ({home_ops:.3f} OPS).",
    })

    away_pen = _top_mean_numeric(away_bullpen_df, "Bullpen Score", n=3, default=0.0)
    home_pen = _top_mean_numeric(home_bullpen_df, "Bullpen Score", n=3, default=0.0)
    pen_edge, pen_diff = _lean_from_diff(away_team, home_team, away_pen, home_pen, higher_is_better=True, threshold=3.0)
    away_taxed = int(away_bullpen_df.get("Availability", pd.Series(dtype=str)).astype(str).isin(["Monitor", "Taxed"]).sum()) if away_bullpen_df is not None and not away_bullpen_df.empty else 0
    home_taxed = int(home_bullpen_df.get("Availability", pd.Series(dtype=str)).astype(str).isin(["Monitor", "Taxed"]).sum()) if home_bullpen_df is not None and not home_bullpen_df.empty else 0
    rows.append({
        "Factor": "Bullpen leverage",
        "Edge": pen_edge,
        "Confidence": "High" if abs(pen_diff) >= 8 else "Medium" if abs(pen_diff) >= 4 else "Low",
        "Why": f"Top-3 bullpen score {away_team} {away_pen:.1f} vs {home_team} {home_pen:.1f}; workload flags {away_team} {away_taxed}, {home_team} {home_taxed}.",
    })

    away_recent = away_recent_summary.iloc[0].to_dict() if away_recent_summary is not None and not away_recent_summary.empty else {}
    home_recent = home_recent_summary.iloc[0].to_dict() if home_recent_summary is not None and not home_recent_summary.empty else {}
    away_run_diff = _safe_number(away_recent.get("Runs/G"), 0) - _safe_number(away_recent.get("Allowed/G"), 0)
    home_run_diff = _safe_number(home_recent.get("Runs/G"), 0) - _safe_number(home_recent.get("Allowed/G"), 0)
    recent_edge, recent_diff = _lean_from_diff(away_team, home_team, away_run_diff, home_run_diff, higher_is_better=True, threshold=0.30)
    rows.append({
        "Factor": "Recent form quality",
        "Edge": recent_edge,
        "Confidence": "High" if abs(recent_diff) >= 1.5 else "Medium" if abs(recent_diff) >= 0.7 else "Low",
        "Why": f"{away_team} {away_recent.get('Recent', 'N/A')} ({away_run_diff:+.1f} R/G, opp Q {away_recent.get('Opp Quality Score', 'N/A')}); {home_team} {home_recent.get('Recent', 'N/A')} ({home_run_diff:+.1f} R/G, opp Q {home_recent.get('Opp Quality Score', 'N/A')}).",
    })

    away_def = away_park_defense.iloc[0].to_dict() if away_park_defense is not None and not away_park_defense.empty else {}
    home_def = home_park_defense.iloc[0].to_dict() if home_park_defense is not None and not home_park_defense.empty else {}
    away_stress = _safe_number(away_def.get("Range Stress"), 50)
    home_stress = _safe_number(home_def.get("Range Stress"), 50)
    defense_edge, defense_diff = _lean_from_diff(away_team, home_team, away_stress, home_stress, higher_is_better=False, threshold=8)
    park = environment.get("park", {}) if isinstance(environment, dict) else {}
    rows.append({
        "Factor": "Park/defense fit",
        "Edge": defense_edge,
        "Confidence": "Medium" if abs(defense_diff) >= 12 else "Low",
        "Why": f"Run factor {park.get('Runs', 1.0):.2f}; range stress {away_team} {away_stress:.0f} vs {home_team} {home_stress:.0f}.",
    })

    return pd.DataFrame(rows, columns=columns)


def build_advanced_context(
    game_id, game_date, season, home_team, away_team,
    home_pitcher, away_pitcher,
    home_starter_info, away_starter_info,
    home_arsenal_df, away_arsenal_df,
    home_lineup_df, away_lineup_df,
    home_bullpen_df, away_bullpen_df,
    home_baserunning_df, away_baserunning_df,
    home_defense_df, away_defense_df,
    home_splits_df, away_splits_df,
    statcast_detail_df,
    statcast_similarity_df=None,
    dh_game=None,
    starter_provenance=None,
):
    context_end = _pregame_end_date(game_date) or game_date
    environment = get_game_environment(game_id, game_date)
    similarity_df = statcast_similarity_df if statcast_similarity_df is not None and not statcast_similarity_df.empty else statcast_detail_df
    league_batter_context = generate_league_batter_context(statcast_detail_df)
    home_pitcher_similar = generate_pitcher_similarity_report(home_pitcher, similarity_df)
    away_pitcher_similar = generate_pitcher_similarity_report(away_pitcher, similarity_df)
    # The type split needs a deeper comp bench than the 14 shown in the report: it takes
    # whole pitchers until the PA sample is usable, so it can run past the display list.
    home_pitcher_comps = generate_pitcher_similarity_report(home_pitcher, similarity_df, top_n=60)
    away_pitcher_comps = generate_pitcher_similarity_report(away_pitcher, similarity_df, top_n=60)
    away_similarity = generate_lineup_vs_similar_pitchers(away_lineup_df, home_pitcher_comps, similarity_df)
    home_similarity = generate_lineup_vs_similar_pitchers(home_lineup_df, away_pitcher_comps, similarity_df)
    away_type_results = generate_team_pitcher_type_results(away_team, home_pitcher_comps, similarity_df)
    home_type_results = generate_team_pitcher_type_results(home_team, away_pitcher_comps, similarity_df)
    if away_type_results.empty:
        away_type_results = generate_team_arsenal_type_results(away_team, home_arsenal_df, _pitcher_throw_code(home_pitcher), similarity_df)
    if home_type_results.empty:
        home_type_results = generate_team_arsenal_type_results(home_team, away_arsenal_df, _pitcher_throw_code(away_pitcher), similarity_df)
    away_recent_summary, away_recent_detail = get_recent_form_opponent_context(get_team_id(away_team), away_team, season, context_end)
    home_recent_summary, home_recent_detail = get_recent_form_opponent_context(get_team_id(home_team), home_team, season, context_end)
    away_rolling_form = get_team_rolling_form(get_team_id(away_team), away_team, season, context_end)
    home_rolling_form = get_team_rolling_form(get_team_id(home_team), home_team, season, context_end)
    away_catcher_run_game = generate_catcher_run_game_report(away_team, home_team, away_defense_df, home_baserunning_df)
    home_catcher_run_game = generate_catcher_run_game_report(home_team, away_team, home_defense_df, away_baserunning_df)
    # Rescale the baserunning heat to where the league actually sits today, before any
    # table is coloured. Cheap (one cached request) and silent if it fails.
    refresh_baserunning_heat_anchors(season)
    away_park_defense = generate_park_defense_impact(away_team, home_team, away_defense_df, home_lineup_df, home_baserunning_df, environment)
    home_park_defense = generate_park_defense_impact(home_team, away_team, home_defense_df, away_lineup_df, away_baserunning_df, environment)
    home_pitcher_quality = get_pitcher_opponent_quality(home_pitcher, season, context_end)
    away_pitcher_quality = get_pitcher_opponent_quality(away_pitcher, season, context_end)
    home_pitcher_hand_splits = generate_pitcher_batter_hand_splits(home_pitcher, season, context_end)
    away_pitcher_hand_splits = generate_pitcher_batter_hand_splits(away_pitcher, season, context_end)
    away_vs_home_arsenal = opponent_vs_pitcher_arsenal(away_team, away_lineup_df, home_arsenal_df, _pitcher_throw_code(home_pitcher), statcast_detail_df)
    home_vs_away_arsenal = opponent_vs_pitcher_arsenal(home_team, home_lineup_df, away_arsenal_df, _pitcher_throw_code(away_pitcher), statcast_detail_df)
    away_batter_arsenal = generate_batter_arsenal_matchups(away_team, away_lineup_df, home_arsenal_df, _pitcher_throw_code(home_pitcher), statcast_detail_df)
    home_batter_arsenal = generate_batter_arsenal_matchups(home_team, home_lineup_df, away_arsenal_df, _pitcher_throw_code(away_pitcher), statcast_detail_df)
    away_bvp = generate_batter_pitcher_matchups(away_lineup_df, home_pitcher, season, context_end)
    home_bvp = generate_batter_pitcher_matchups(home_lineup_df, away_pitcher, season, context_end)
    hitter_composite = pd.concat([
        build_hitter_composite_table(away_team, away_lineup_df, away_splits_df, away_similarity, away_batter_arsenal, away_bvp),
        build_hitter_composite_table(home_team, home_lineup_df, home_splits_df, home_similarity, home_batter_arsenal, home_bvp),
    ], ignore_index=True)
    hitter_summary = summarize_hitter_composite(hitter_composite)
    home_hand = _pitcher_throw_code(home_pitcher)
    away_hand = _pitcher_throw_code(away_pitcher)
    away_hotcold28 = generate_hot_cold_hitters(away_team, days=28, end_date=context_end)
    home_hotcold28 = generate_hot_cold_hitters(home_team, days=28, end_date=context_end)
    away_lineup_splits = _keep_venue_split(build_lineup_split_summary(away_team, away_lineup_df, statcast_detail_df, home_hand, away_vs_home_arsenal, context_end), "Away (PF)")
    home_lineup_splits = _keep_venue_split(build_lineup_split_summary(home_team, home_lineup_df, statcast_detail_df, away_hand, home_vs_away_arsenal, context_end), "Home (PF)")
    # The pitching each lineup faces plays this game at the home team's park.
    away_opp_pitching = _keep_venue_split(build_opp_pitching_allowed_summary(
        home_pitcher, home_team, home_bullpen_df, season, context_end,
        starter_name=home_starter_info.get("Name")), "Home (PF)")
    home_opp_pitching = _keep_venue_split(build_opp_pitching_allowed_summary(
        away_pitcher, away_team, away_bullpen_df, season, context_end,
        starter_name=away_starter_info.get("Name")), "Away (PF)")
    hitter_composite = attach_offense_index(
        hitter_composite, [away_lineup_df, home_lineup_df], league_batter_context, [away_hotcold28, home_hotcold28]
    )
    # Pitch-type run values + swing decisions (lineup vs the arsenal it faces).
    statcast_pitches_df = load_statcast_pitches(season, context_end)
    away_arsenal_matchup = build_arsenal_pitch_matchup(away_team, away_lineup_df, home_arsenal_df, statcast_pitches_df)
    home_arsenal_matchup = build_arsenal_pitch_matchup(home_team, home_lineup_df, away_arsenal_df, statcast_pitches_df)
    away_arsenal_drilldown = build_batter_arsenal_swing_drilldown(away_team, away_lineup_df, home_arsenal_df, statcast_pitches_df)
    home_arsenal_drilldown = build_batter_arsenal_swing_drilldown(home_team, home_lineup_df, away_arsenal_df, statcast_pitches_df)
    # Rest / schedule spot per team.
    away_rest_schedule = build_rest_schedule(away_team, get_team_id(away_team), away_pitcher, season, game_date, away_bullpen_df)
    home_rest_schedule = build_rest_schedule(home_team, get_team_id(home_team), home_pitcher, season, game_date, home_bullpen_df)
    away_sp_rest_splits = build_starter_rest_splits(away_pitcher, season, context_end)
    home_sp_rest_splits = build_starter_rest_splits(home_pitcher, season, context_end)
    away_schedule_context = build_schedule_context_performance(get_team_id(away_team), season, context_end)
    home_schedule_context = build_schedule_context_performance(get_team_id(home_team), season, context_end)
    away_trip_spot, away_trip_form = build_trip_homestand_form(
        get_team_id(away_team), season, game_date, current_is_home=False
    )
    home_trip_spot, home_trip_form = build_trip_homestand_form(
        get_team_id(home_team), season, game_date, current_is_home=True
    )
    current_venue_id = environment.get("venue_id") or _TEAM_HOME_VENUE_ID.get(
        get_team_id(home_team)
    )
    away_time_zone = build_time_zone_context(
        get_team_id(away_team), season, game_date,
        current_venue_id=current_venue_id,
        current_game_datetime=environment.get("game_datetime"),
        current_home_team_id=get_team_id(home_team),
    )
    home_time_zone = build_time_zone_context(
        get_team_id(home_team), season, game_date,
        current_venue_id=current_venue_id,
        current_game_datetime=environment.get("game_datetime"),
        current_home_team_id=get_team_id(home_team),
    )
    hitter_watchlist = pd.concat([
        build_hitter_watchlist(away_team, away_lineup_df, away_similarity, league_batter_context, away_splits_df),
        build_hitter_watchlist(home_team, home_lineup_df, home_similarity, league_batter_context, home_splits_df),
    ], ignore_index=True)
    if not hitter_watchlist.empty and "Priority" in hitter_watchlist.columns:
        hitter_watchlist = hitter_watchlist.sort_values("Priority", ascending=False)
    pitcher_watchlist = build_pitcher_watchlist(
        home_team, away_team, home_starter_info, away_starter_info,
        away_vs_home_arsenal, home_vs_away_arsenal, environment,
        away_batter_arsenal=away_batter_arsenal, home_batter_arsenal=home_batter_arsenal,
        away_lineup_df=away_lineup_df, home_lineup_df=home_lineup_df)
    away_bullpen_form = summarize_bullpen_form(away_team, away_bullpen_df)
    home_bullpen_form = summarize_bullpen_form(home_team, home_bullpen_df)
    scorecard, projection_summary = build_game_scorecard(
        home_team, away_team,
        home_starter_info, away_starter_info,
        home_lineup_df, away_lineup_df,
        home_bullpen_df, away_bullpen_df,
        home_similarity, away_similarity,
        environment,
    )
    if not scorecard.empty:
        recent_map = {
            away_team: away_recent_summary.iloc[0].to_dict() if not away_recent_summary.empty else {},
            home_team: home_recent_summary.iloc[0].to_dict() if not home_recent_summary.empty else {},
        }
        type_map = {
            away_team: away_type_results.iloc[0].to_dict() if not away_type_results.empty else {},
            home_team: home_type_results.iloc[0].to_dict() if not home_type_results.empty else {},
        }
        catcher_map = {
            away_team: advanced_catcher_label(home_team, home_catcher_run_game),
            home_team: advanced_catcher_label(away_team, away_catcher_run_game),
        }
        scorecard["Recent"] = scorecard["Team"].map(lambda team: recent_map.get(team, {}).get("Recent", ""))
        scorecard["Recent Opp"] = scorecard["Team"].map(
            lambda team: (
                f"{recent_map.get(team, {}).get('Opp Quality', '')} / "
                f"{recent_map.get(team, {}).get('Opp Quality Score', '')} Q / "
                f"{recent_map.get(team, {}).get('Avg Opp W%', '')} W% / "
                f"{recent_map.get(team, {}).get('Runs/G', '')}-{recent_map.get(team, {}).get('Allowed/G', '')} R"
            ).replace(" /  / ", " / ")
        )
        def type_summary(team):
            row = type_map.get(team, {})
            if not row:
                return "No type sample"
            games = row.get("Games", "")
            pa = row.get("PA", "")
            record = row.get("Record", "0-0")
            diff = row.get("OPS Diff", "")
            return f"{record} / {diff} / {games}g {pa} PA".strip(" /")

        scorecard["Pitcher Type"] = scorecard["Team"].map(type_summary)
        arsenal_map = {
            away_team: away_vs_home_arsenal.iloc[0].to_dict() if not away_vs_home_arsenal.empty else {},
            home_team: home_vs_away_arsenal.iloc[0].to_dict() if not home_vs_away_arsenal.empty else {},
        }
        scorecard["vs Arsenal"] = scorecard["Team"].map(
            lambda team: (
                f"{arsenal_map.get(team, {}).get('Arsenal Tag', '')}: "
                f"{arsenal_map.get(team, {}).get('League OPS Pctl', '')} "
                f"({_clean_markdown(arsenal_map.get(team, {}).get('OPS', ''))} OPS, "
                f"{_clean_markdown(arsenal_map.get(team, {}).get('xwOBA', ''))} xwOBA)"
            ).strip()
        )
        defense_map = {
            away_team: away_park_defense.iloc[0].to_dict() if not away_park_defense.empty else {},
            home_team: home_park_defense.iloc[0].to_dict() if not home_park_defense.empty else {},
        }
        scorecard["Defense Fit"] = scorecard["Team"].map(
            lambda team: (
                f"{defense_map.get(team, {}).get('Range Fit', '')} / "
                f"{defense_map.get(team, {}).get('In-Play Importance', '')}"
            ).strip(" /")
        )
        scorecard["Run Game"] = scorecard["Team"].map(lambda team: catcher_map.get(team, ""))
    scorecard, projection_summary = apply_trusted_time_zone_baseline(
        scorecard,
        projection_summary,
        away_team,
        home_team,
        away_time_zone,
        home_time_zone,
    )
    factor_matrix = build_factor_matrix(
        away_team, home_team, scorecard,
        away_starter_info, home_starter_info,
        away_bullpen_df, home_bullpen_df,
        away_vs_home_arsenal, home_vs_away_arsenal,
        away_recent_summary, home_recent_summary,
        away_park_defense, home_park_defense,
        environment,
    )
    uncertainty_flags = generate_uncertainty_flags(
        scorecard, away_team, home_team,
        away_lineup_df, home_lineup_df,
        away_type_results, home_type_results,
        away_bullpen_df, home_bullpen_df,
        projection_summary,
    )
    # Each club is compared against the arm it faces, so the starter info is crossed.
    _away_tonight = _tonight_conditions(environment, home_starter_info, away_lineup_df,
                                        is_home=False, rest_schedule=away_rest_schedule)
    _home_tonight = _tonight_conditions(environment, away_starter_info, home_lineup_df,
                                        is_home=True, rest_schedule=home_rest_schedule)
    return {
        "environment": environment,
        # Where each starter's name came from. Rides in the cached payload so a re-render
        # can label a provisional pick and `refresh_cached_lineups` can retire it.
        "starter_provenance": starter_provenance or {},
        "scorecard": scorecard,
        "factor_matrix": factor_matrix,
        "projection_summary": projection_summary,
        "hitter_composite": hitter_composite,
        "hitter_summary": hitter_summary,
        "hitter_watchlist": hitter_watchlist,
        "pitcher_watchlist": pitcher_watchlist,
        "home_pitcher_similar": home_pitcher_similar,
        "away_pitcher_similar": away_pitcher_similar,
        "bullpen_notes": bullpen_deployment_notes(away_team, away_bullpen_df) + bullpen_deployment_notes(home_team, home_bullpen_df),
        "lineup_notes": lineup_composition_notes(away_team, away_lineup_df, away_splits_df) + lineup_composition_notes(home_team, home_lineup_df, home_splits_df),
        "pitcher_lineup_notes": pitcher_vs_lineup_notes(home_team, home_starter_info, home_arsenal_df, away_lineup_df, home_pitcher_hand_splits, away_vs_home_arsenal) + pitcher_vs_lineup_notes(away_team, away_starter_info, away_arsenal_df, home_lineup_df, away_pitcher_hand_splits, home_vs_away_arsenal),
        "away_similarity": away_similarity,
        "home_similarity": home_similarity,
        "away_type_results": away_type_results,
        "home_type_results": home_type_results,
        "away_recent_summary": away_recent_summary,
        "home_recent_summary": home_recent_summary,
        "away_recent_detail": away_recent_detail,
        "home_recent_detail": home_recent_detail,
        "away_rolling_form": away_rolling_form,
        "home_rolling_form": home_rolling_form,
        "away_park_defense": away_park_defense,
        "home_park_defense": home_park_defense,
        "home_pitcher_quality": home_pitcher_quality,
        "away_pitcher_quality": away_pitcher_quality,
        "home_pitcher_hand_splits": home_pitcher_hand_splits,
        "away_pitcher_hand_splits": away_pitcher_hand_splits,
        "away_vs_home_arsenal": away_vs_home_arsenal,
        "home_vs_away_arsenal": home_vs_away_arsenal,
        "away_batter_arsenal": away_batter_arsenal,
        "home_batter_arsenal": home_batter_arsenal,
        "away_bullpen_form": away_bullpen_form,
        "home_bullpen_form": home_bullpen_form,
        "away_recent_boxscores": build_recent_boxscores(get_team_id(away_team), away_team, game_date,
                                                        statcast_df=statcast_detail_df),
        "home_recent_boxscores": build_recent_boxscores(get_team_id(home_team), home_team, game_date,
                                                        statcast_df=statcast_detail_df),
        # Relief workload over the last five GAMES. Shares the boxscore cache the tables
        # above just filled, so the two extra games cost one request each.
        "away_bullpen_l5": build_bullpen_l5(get_team_id(away_team), away_team, game_date,
                                            bullpen_df=away_bullpen_df),
        "home_bullpen_l5": build_bullpen_l5(get_team_id(home_team), home_team, game_date,
                                            bullpen_df=home_bullpen_df),
        # Batted-ball profile for each pen, sliced from the statcast frame already loaded.
        "away_bullpen_batted": build_bullpen_batted(away_bullpen_df, statcast_detail_df),
        "home_bullpen_batted": build_bullpen_batted(home_bullpen_df, statcast_detail_df),
        # Last ten games with tonight's conditions attached, and the ranked comparables the
        # per-team tabs render. The opposing starter is the OTHER club's probable, so each
        # side is compared against the matchup it is actually walking into.
        "away_relevant_games": build_relevant_games(
            get_team_id(away_team), away_team, game_date, _away_tonight,
            season=season, context_end=context_end),
        "home_relevant_games": build_relevant_games(
            get_team_id(home_team), home_team, game_date, _home_tonight,
            season=season, context_end=context_end),
        "away_comparable_games": build_comparable_games(
            get_team_id(away_team), away_team, game_date, _away_tonight,
            season=season, context_end=context_end, statcast_df=statcast_detail_df),
        "home_comparable_games": build_comparable_games(
            get_team_id(home_team), home_team, game_date, _home_tonight,
            season=season, context_end=context_end, statcast_df=statcast_detail_df),
        "uncertainty_flags": uncertainty_flags,
        "away_bvp": away_bvp,
        "home_bvp": home_bvp,
        "away_absences": get_notable_absences(away_team, context_end, lineup_df=away_lineup_df),
        "home_absences": get_notable_absences(home_team, context_end, lineup_df=home_lineup_df),
        # Persist structured roster moves with the assembled report. Layout-only
        # re-renders must not depend on a fresh transaction API call.
        "away_transactions": build_transaction_table(away_team, game_date),
        "home_transactions": build_transaction_table(home_team, game_date),
        "away_defense_stress": generate_defensive_stress_report(away_team, home_team, away_defense_df, home_lineup_df, home_baserunning_df, environment),
        "home_defense_stress": generate_defensive_stress_report(home_team, away_team, home_defense_df, away_lineup_df, away_baserunning_df, environment),
        # Measured from batted-ball outcomes rather than putout counts -- see
        # build_team_defense for why the fielding-stat view rates the pitching staff.
        "away_team_defense": build_team_defense(away_team, statcast_detail_df),
        "home_team_defense": build_team_defense(home_team, statcast_detail_df),
        "away_catcher_run_game": away_catcher_run_game,
        "home_catcher_run_game": home_catcher_run_game,
        "away_lineup_splits": away_lineup_splits,
        "home_lineup_splits": home_lineup_splits,
        "away_opp_pitching": away_opp_pitching,
        "home_opp_pitching": home_opp_pitching,
        "away_arsenal_matchup": away_arsenal_matchup,
        "home_arsenal_matchup": home_arsenal_matchup,
        "away_arsenal_drilldown": away_arsenal_drilldown,
        "home_arsenal_drilldown": home_arsenal_drilldown,
        "away_rest_schedule": away_rest_schedule,
        "home_rest_schedule": home_rest_schedule,
        "away_sp_rest_splits": away_sp_rest_splits,
        "home_sp_rest_splits": home_sp_rest_splits,
        "away_schedule_context": away_schedule_context,
        "home_schedule_context": home_schedule_context,
        # Per-start mix/velo/batted-ball profile behind the expanded pitching tab.
        "away_sp_profile": build_starter_pitch_profile(away_pitcher, season, context_end),
        "home_sp_profile": build_starter_pitch_profile(home_pitcher, season, context_end),
        # Opener/bulk detection reuses the pitch-level frame already loaded above.
        "away_opener_profile": build_opener_profile(away_pitcher, season, context_end, statcast_pitches_df),
        "home_opener_profile": build_opener_profile(home_pitcher, season, context_end, statcast_pitches_df),
        "away_sweeps": build_sweep_record(get_team_id(away_team), season, context_end),
        "home_sweeps": build_sweep_record(get_team_id(home_team), season, context_end),
        # Which row of the season splits above is tonight's actual situation.
        "away_context_tonight": build_tonight_schedule_context(get_team_id(away_team), season, game_date, dh_game),
        "home_context_tonight": build_tonight_schedule_context(get_team_id(home_team), season, game_date, dh_game),
        "away_sp_rest_tonight": sp_rest_bucket(_first_value(away_rest_schedule, "SP Rest")),
        "home_sp_rest_tonight": sp_rest_bucket(_first_value(home_rest_schedule, "SP Rest")),
        "away_trip_spot": away_trip_spot,
        "home_trip_spot": home_trip_spot,
        "away_trip_form": away_trip_form,
        "home_trip_form": home_trip_form,
        "away_time_zone": away_time_zone,
        "home_time_zone": home_time_zone,
    }


def _safe_number(value, default=None):
    if value is None:
        return default
    if isinstance(value, str):
        value = value.replace("%", "").replace(",", "").strip()
        if value in {"", "-", "--", "N/A", "nan"}:
            return default
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_pct(value):
    number = _safe_number(value)
    if number is None:
        return "N/A"
    if abs(number) <= 1:
        number *= 100
    return f"{number:.1f}%"


def _first_name(info):
    if not isinstance(info, dict):
        return "Unknown"
    return info.get("Name") or "Unknown"


def _top_rows(df, sort_col, n=3, ascending=False):
    if df is None or df.empty or sort_col not in df.columns:
        return pd.DataFrame()
    temp = df.copy()
    temp[sort_col] = pd.to_numeric(temp[sort_col], errors="coerce")
    return temp.dropna(subset=[sort_col]).sort_values(sort_col, ascending=ascending).head(n)


def _starter_points(team, starter_info, arsenal_df):
    name = _first_name(starter_info)
    points = []

    era = starter_info.get("ERA") if isinstance(starter_info, dict) else None
    fip = starter_info.get("FIP") if isinstance(starter_info, dict) else None
    k_pct = starter_info.get("K%") if isinstance(starter_info, dict) else None
    bb_pct = starter_info.get("BB%") if isinstance(starter_info, dict) else None
    whip = starter_info.get("WHIP") if isinstance(starter_info, dict) else None

    points.append(
        f"{team} starter {name}: ERA {era if era is not None else 'N/A'}, "
        f"FIP {fip if fip is not None else 'N/A'}, K% {_format_pct(k_pct)}, "
        f"BB% {_format_pct(bb_pct)}, WHIP {whip if whip is not None else 'N/A'}."
    )

    if arsenal_df is not None and not arsenal_df.empty:
        usage = _top_rows(arsenal_df, "Usage %", n=1)
        csw = _top_rows(arsenal_df, "CSW%", n=1)
        stuff = _top_rows(arsenal_df, "Stuff+", n=1)

        if not usage.empty:
            row = usage.iloc[0]
            points.append(f"Primary pitch is {row.get('Pitch')} at {row.get('Usage %')}% usage; build the plan around that pitch shape and location.")
        if not csw.empty:
            row = csw.iloc[0]
            points.append(f"Best bat-missing/called-strike weapon is {row.get('Pitch')} with {row.get('CSW%')}% CSW.")
        if not stuff.empty:
            row = stuff.iloc[0]
            points.append(f"Best raw pitch-quality signal is {row.get('Pitch')} with {row.get('Stuff+')} Stuff+.")

    return points


def _offense_points(team, opposing_pitcher_info, hotcold_df, splits_df):
    opposing_name = _first_name(opposing_pitcher_info)
    points = [f"{team} hitters vs {opposing_name}: prioritize the lineup pockets below before the full tables."]

    hot = pd.DataFrame()
    if hotcold_df is not None and not hotcold_df.empty and "Status" in hotcold_df.columns:
        hot = hotcold_df[hotcold_df["Status"].astype(str).str.contains("HOT", na=False)]
    if hot.empty:
        hot = _top_rows(hotcold_df, "xwOBA", n=3)
    else:
        hot = _top_rows(hot, "xwOBA", n=3)
    if not hot.empty:
        names = ", ".join(hot["Name"].astype(str).head(3).tolist())
        points.append(f"Current-form bats to manage carefully: {names}.")

    cold = pd.DataFrame()
    if hotcold_df is not None and not hotcold_df.empty and "Status" in hotcold_df.columns:
        cold = hotcold_df[hotcold_df["Status"].astype(str).str.contains("COLD", na=False)]
    cold = _top_rows(cold, "OPS", n=2, ascending=True)
    if not cold.empty:
        names = ", ".join(cold["Name"].astype(str).head(2).tolist())
        points.append(f"Potential attack lanes if ahead in count: {names}.")

    if splits_df is not None and not splits_df.empty:
        ops_cols = [col for col in splits_df.columns if col.startswith("OPS vs")]
        if ops_cols:
            best = _top_rows(splits_df, ops_cols[0], n=3)
            if not best.empty:
                names = ", ".join(best["Name"].astype(str).head(3).tolist())
                points.append(f"Best season platoon/split threats by OPS: {names}.")

    return points


def _bullpen_points(team, bullpen_df):
    if bullpen_df is None or bullpen_df.empty:
        return [f"{team} bullpen data unavailable; confirm availability manually."]

    high_ip = _top_rows(bullpen_df[bullpen_df["Name"].astype(str).str.upper().str.contains("BULLPEN") == False], "IP", n=3)
    names = ", ".join(high_ip["Name"].astype(str).head(3).tolist()) if not high_ip.empty else "N/A"
    era = _safe_number(bullpen_df["ERA"].mean()) if "ERA" in bullpen_df else None
    fip = _safe_number(bullpen_df["FIP"].mean()) if "FIP" in bullpen_df else None
    return [f"{team} bullpen: main workload arms by IP are {names}; group ERA {era:.2f} and FIP {fip:.2f}." if era is not None and fip is not None else f"{team} bullpen: main workload arms by IP are {names}."]


def _running_defense_points(team, baserun_df, defense_df):
    points = []
    runner_sort = "BsR" if baserun_df is not None and "BsR" in baserun_df.columns else "SB_Att"
    top_runner = _top_rows(baserun_df, runner_sort, n=1)
    if not top_runner.empty:
        row = top_runner.iloc[0]
        if "BsR" in row.index:
            points.append(f"{team} running game: top BsR threat is {row.get('Name')} with {row.get('BsR')} BsR and {row.get('SB_Att')} steal attempts.")
        else:
            points.append(f"{team} running game: top steal-pressure threat is {row.get('Name')} with {row.get('SB')} SB, {row.get('CS')} CS, and {row.get('SB_Att')} attempts.")

    if defense_df is not None and not defense_df.empty:
        sort_cols = [col for col in ["DRS", "Def", "Fld", "Inn", "Chances"] if col in defense_df.columns]
        if sort_cols:
            top_def = _top_rows(defense_df, sort_cols[0], n=1)
            if not top_def.empty:
                row = top_def.iloc[0]
                if sort_cols[0] in {"Inn", "Chances"}:
                    points.append(f"{team} defense: most-used defender by {sort_cols[0]} is {row.get('Name')} ({row.get('Pos', 'N/A')}) at {row.get(sort_cols[0])}.")
                else:
                    points.append(f"{team} defense: strongest available {sort_cols[0]} signal is {row.get('Name')} at {row.get(sort_cols[0])}.")

    return points or [f"{team} running/defense edge data unavailable."]


def build_actionable_game_plan(
    home_team, away_team,
    home_starter_info, away_starter_info,
    home_hotcold_df, away_hotcold_df,
    home_arsenal_df, away_arsenal_df,
    home_bullpen_df, away_bullpen_df,
    home_baserunning_df, away_baserunning_df,
    home_defense_df, away_defense_df,
    home_splits_df, away_splits_df,
):
    return [
        {"title": f"{away_team} Offense vs {_first_name(home_starter_info)}", "points": _offense_points(away_team, home_starter_info, away_hotcold_df, away_splits_df)},
        {"title": f"{home_team} Offense vs {_first_name(away_starter_info)}", "points": _offense_points(home_team, away_starter_info, home_hotcold_df, home_splits_df)},
        {"title": f"{away_team} Starter Plan", "points": _starter_points(away_team, away_starter_info, away_arsenal_df)},
        {"title": f"{home_team} Starter Plan", "points": _starter_points(home_team, home_starter_info, home_arsenal_df)},
        {"title": "Bullpen Leverage", "points": _bullpen_points(away_team, away_bullpen_df) + _bullpen_points(home_team, home_bullpen_df)},
        {"title": "Run Prevention Edges", "points": _running_defense_points(away_team, away_baserunning_df, away_defense_df) + _running_defense_points(home_team, home_baserunning_df, home_defense_df)},
    ]




# ------------- LEAGUE-PERCENTILE HEAT COLORING (shared by PDF + HTML) -------------
#
# Curated MLB reference anchors per metric: (raw_value, goodness_percentile) where
# goodness runs 0..1 (1 = elite). Anchors sit near the 5th / 50th / 95th percentile.
# Direction is baked into the anchors (dir "low" stats descend in raw as goodness rises),
# and context-specific metric ids keep the same column scored correctly in different tables
# (e.g. ops_off is high-good, ops_allowed is low-good).
_HEAT_METRICS = {
    # --- offense (higher is better unless noted) ---
    "ops_off":     [(0.600, 0.05), (0.720, 0.50), (0.870, 0.95)],
    "obp_off":     [(0.290, 0.05), (0.320, 0.50), (0.370, 0.95)],
    "slg_off":     [(0.350, 0.05), (0.410, 0.50), (0.510, 0.95)],
    "avg_off":     [(0.220, 0.05), (0.250, 0.50), (0.300, 0.95)],
    "iso":         [(0.090, 0.05), (0.160, 0.50), (0.250, 0.95)],
    "xwoba_off":   [(0.280, 0.05), (0.315, 0.50), (0.375, 0.95)],
    "hardhit":     [(28.0, 0.05), (38.0, 0.50), (50.0, 0.95)],
    "bb_pct_bat":  [(4.5, 0.05), (8.5, 0.50), (14.0, 0.95)],
    "k_pct_bat":   [(30.0, 0.05), (22.0, 0.50), (14.0, 0.95)],   # low is better
    "off_index":   [(80.0, 0.05), (100.0, 0.50), (140.0, 0.95)],
    # --- pitching / allowed (lower allowed is better) ---
    "ops_allowed":   [(0.850, 0.05), (0.710, 0.50), (0.560, 0.95)],
    "obp_allowed":   [(0.360, 0.05), (0.315, 0.50), (0.275, 0.95)],
    "slg_allowed":   [(0.480, 0.05), (0.400, 0.50), (0.320, 0.95)],
    "xwoba_allowed": [(0.370, 0.05), (0.315, 0.50), (0.270, 0.95)],
    "fip":           [(5.20, 0.05), (4.10, 0.50), (3.10, 0.95)],  # low is better
    "era":           [(5.40, 0.05), (4.10, 0.50), (2.90, 0.95)],
    "whip":          [(1.55, 0.05), (1.30, 0.50), (1.05, 0.95)],
    "k_pct_pitch":   [(15.0, 0.05), (22.0, 0.50), (30.0, 0.95)],  # high is better
    "bb_pct_pitch":  [(11.0, 0.05), (8.0, 0.50), (5.0, 0.95)],    # low is better
    "whiff":         [(18.0, 0.05), (25.0, 0.50), (33.0, 0.95)],
    "hardhit_allowed": [(45.0, 0.05), (38.0, 0.50), (30.0, 0.95)],
    "iso_allowed":     [(0.210, 0.05), (0.160, 0.50), (0.115, 0.95)],  # low is better
    "hr_pct_bat":      [(1.5, 0.05), (3.0, 0.50), (5.0, 0.95)],        # high is better
    "hr_pct_allowed":  [(4.5, 0.05), (3.0, 0.50), (1.8, 0.95)],        # low is better
    # --- arsenal ---
    "stuff_plus": [(85.0, 0.05), (100.0, 0.50), (120.0, 0.95)],
    "csw":        [(24.0, 0.05), (28.0, 0.50), (34.0, 0.95)],
    # --- pitch-type matchup (hitter/offense perspective) ---
    "rv100":       [(-1.5, 0.05), (0.0, 0.50), (1.5, 0.95)],   # batter run value per 100, high = good
    "whiff_off":   [(35.0, 0.05), (25.0, 0.50), (15.0, 0.95)], # low whiff = good for hitter
    "chase_off":   [(40.0, 0.05), (30.0, 0.50), (22.0, 0.95)], # low chase = good discipline
    "contact_off": [(68.0, 0.05), (78.0, 0.50), (86.0, 0.95)], # high contact = good
    # --- team run scoring / prevention (per game) ---
    "rpg_off": [(3.5, 0.05), (4.4, 0.50), (5.5, 0.95)],       # runs scored, high = good
    "rpg_allowed": [(5.5, 0.05), (4.4, 0.50), (3.3, 0.95)],   # runs allowed, low = good
    "win_pct": [(0.35, 0.05), (0.50, 0.50), (0.65, 0.95)],
    "run_diff_pg": [(-1.0, 0.05), (0.0, 0.50), (1.0, 0.95)],
    # --- baserunning (SB% stored as a fraction 0..1) ---
    "bsr": [(-4.0, 0.05), (0.0, 0.50), (6.0, 0.95)],
    "spd": [(3.0, 0.05), (5.0, 0.50), (7.5, 0.95)],
    "sb_pct": [(0.60, 0.05), (0.75, 0.50), (0.88, 0.95)],
    # --- defense / workload ---
    "caught_stealing": [(0.0, 0.05), (3.0, 0.50), (10.0, 0.95)],
    "sb_allowed": [(25.0, 0.05), (12.0, 0.50), (3.0, 0.95)],
    "caught_stealing_pct": [(0.10, 0.05), (0.25, 0.50), (0.40, 0.95)],
    "passed_balls": [(8.0, 0.05), (3.0, 0.50), (0.0, 0.95)],
    "passed_balls_rate": [(2.0, 0.05), (0.7, 0.50), (0.0, 0.95)],
    "appearances": [(5.0, 0.05), (25.0, 0.50), (50.0, 0.95)],
    "steal_attempts_allowed": [(70.0, 0.05), (35.0, 0.50), (10.0, 0.95)],
    "assists_plus": [(60.0, 0.05), (100.0, 0.50), (140.0, 0.95)],
    # Baserunning counting stats. Replaced at report time by the league's actual
    # season-to-date distribution; these are a full-season fallback for when that pull
    # fails, which is why they look high in April.
    "sb_count": [(0.0, 0.05), (5.0, 0.50), (25.0, 0.95)],
    "cs_count": [(7.0, 0.05), (2.0, 0.50), (0.0, 0.95)],
    "sb_attempts": [(0.0, 0.05), (7.0, 0.50), (31.0, 0.95)],
    "runs_scored": [(25.0, 0.05), (50.0, 0.50), (85.0, 0.95)],
    "triples": [(0.0, 0.05), (1.0, 0.50), (5.0, 0.95)],
    # Team defence, league-centred, so zero is exactly average by construction.
    "xwoba_delta": [(-0.060, 0.05), (0.0, 0.50), (0.060, 0.95)],
    "hits_saved": [(-0.45, 0.05), (0.0, 0.50), (0.45, 0.95)],
    "frame_strikes": [(-0.8, 0.05), (0.0, 0.50), (0.8, 0.95)],
    "xba_allowed": [(0.325, 0.05), (0.307, 0.50), (0.290, 0.95)],
    "ba_allowed": [(0.320, 0.05), (0.295, 0.50), (0.268, 0.95)],
}


def refresh_baserunning_heat_anchors(season, min_pa=200):
    """Recalibrate the baserunning heat scale to this season's actual distribution.

    Stolen bases, runs and triples are counting stats: a hitter with nine steals is in the
    top quartile in May and merely average in September. Fixed anchors would therefore mean
    something different every month, so the league's own season-to-date spread is pulled
    once per run and installed as the scale. One request covers every hitter.

    Caught stealing is anchored in reverse -- getting thrown out is the bad outcome -- and
    attempts are treated as upside, because on DK a steal is five points and being caught
    costs nothing.
    """
    try:
        data = cached_json_request(
            "https://statsapi.mlb.com/api/v1/stats",
            params={"stats": "season", "group": "hitting", "season": int(season),
                    "sportId": 1, "limit": 1500, "playerPool": "All", "gameType": "R"},
            namespace="statsapi", cache_key_extra=str(season),
        )
        splits = (data.get("stats") or [{}])[0].get("splits") or []
    except Exception:
        return False
    if not splits:
        return False

    frame = pd.DataFrame([s.get("stat", {}) for s in splits])
    if "plateAppearances" not in frame.columns:
        return False
    regulars = frame[pd.to_numeric(frame["plateAppearances"], errors="coerce") >= min_pa]
    if len(regulars) < 50:
        return False

    def anchors(series, inverted=False):
        values = pd.to_numeric(series, errors="coerce").dropna()
        if values.empty:
            return None
        low, mid, high = (float(values.quantile(q)) for q in (0.05, 0.50, 0.95))
        if low == high:
            return None
        return ([(high, 0.05), (mid, 0.50), (low, 0.95)] if inverted
                else [(low, 0.05), (mid, 0.50), (high, 0.95)])

    steals = pd.to_numeric(regulars.get("stolenBases"), errors="coerce").fillna(0)
    caught = pd.to_numeric(regulars.get("caughtStealing"), errors="coerce").fillna(0)
    updates = {
        "sb_count": anchors(steals),
        "cs_count": anchors(caught, inverted=True),
        "sb_attempts": anchors(steals + caught),
        "runs_scored": anchors(regulars.get("runs")),
        "triples": anchors(regulars.get("triples")),
    }
    for metric, scale in updates.items():
        if scale:
            _HEAT_METRICS[metric] = scale
    return True

# Column-header -> metric id, per table context.
_HEAT_CONTEXTS = {
    "offense": {
        "OPS": "ops_off", "OPS_proxy": "ops_off", "OBP": "obp_off", "SLG": "slg_off",
        "AVG": "avg_off", "ISO": "iso", "xwOBA": "xwoba_off", "HardHit%": "hardhit",
        "BB%": "bb_pct_bat", "K%": "k_pct_bat", "HR%": "hr_pct_bat",
        "Off Szn": "off_index", "Off L28": "off_index",
    },
    "pitching_allowed": {
        "OPS": "ops_allowed", "OPS_proxy": "ops_allowed", "OBP": "obp_allowed", "SLG": "slg_allowed",
        "xwOBA": "xwoba_allowed", "HardHit%": "hardhit_allowed", "Whiff%": "whiff",
        "K%": "k_pct_pitch", "BB%": "bb_pct_pitch", "ISO": "iso_allowed",
        "HR%": "hr_pct_allowed",
    },
    # Same "allowed" numbers, but read through the HITTING team's lens: a pitching staff
    # that has allowed a lot is a GOOD matchup for the batters, so high = green here.
    "pitching_allowed_batter_view": {
        "OPS": "ops_off", "OPS_proxy": "ops_off", "OBP": "obp_off", "SLG": "slg_off",
        "xwOBA": "xwoba_off", "HardHit%": "hardhit",
        # Read from the hitters' side: a staff that misses few bats and walks people is a
        # good matchup, so these point at the batter-direction anchors on purpose.
        "K%": "k_pct_bat", "BB%": "bb_pct_bat", "ISO": "iso", "HR%": "hr_pct_bat",
    },
    "pitching": {
        "FIP": "fip", "ERA": "era", "WHIP": "whip", "K%": "k_pct_pitch", "BB%": "bb_pct_pitch",
        "OPS": "ops_allowed", "xwOBA": "xwoba_allowed", "SLG": "slg_allowed", "HardHit%": "hardhit_allowed",
        "Whiff%": "whiff",
    },
    "arsenal": {
        "Stuff+": "stuff_plus", "CSW%": "csw",
    },
    "baserunning": {
        "BsR": "bsr", "Spd": "spd", "SB%": "sb_pct",
        # Counting stats, so their anchors are refreshed from the league's own
        # season-to-date distribution rather than hard-coded -- see
        # refresh_baserunning_heat_anchors. Hard-coded full-season anchors would paint
        # every hitter cold in April and warm in September.
        "SB": "sb_count", "CS": "cs_count", "SB_Att": "sb_attempts",
        "Runs": "runs_scored", "Triples": "triples",
    },
    "hot_cold": {
        "OPS": "ops_off", "xwOBA": "xwoba_off", "Szn xwOBA": "xwoba_off",
        "ΔxwOBA": "xwoba_delta", "HardHit%": "hardhit",
        "Chase%": "chase_off", "Z-Con%": "contact_off",
    },
    "team_defense": {
        "Hits Saved/G": "hits_saved", "IF Saved/G": "hits_saved",
        "OF Saved/G": "hits_saved", "Frame +Str/G": "frame_strikes",
        "xBA Allowed": "xba_allowed", "BA Allowed": "ba_allowed",
    },
    "last10": {
        "Result": "winloss",
    },
    "pitch_matchup": {
        "RV/100": "rv100", "Whiff%": "whiff_off", "Chase%": "chase_off",
        "Contact%": "contact_off", "xwOBA": "xwoba_off",
    },
}


def _stat_percentile(metric_id, value):
    """Return goodness in [0,1] for a raw stat value, or None if unknown/non-numeric."""
    if metric_id == "winloss":
        token = str(value).strip().upper()
        if token == "W":
            return 0.9
        if token == "L":
            return 0.1
        return None
    anchors = _HEAT_METRICS.get(metric_id)
    if not anchors:
        return None
    raw = _safe_number(value, None)
    if raw is None:
        return None
    percent_scale_metrics = {
        "hardhit", "bb_pct_bat", "k_pct_bat", "k_pct_pitch", "bb_pct_pitch",
        "whiff", "hardhit_allowed", "csw",
    }
    if metric_id in percent_scale_metrics and 0 < abs(raw) <= 1:
        raw *= 100.0
    if metric_id == "sb_pct" and raw > 1.5:  # tolerate percent-scale input
        raw = raw / 100.0
    pairs = sorted(anchors, key=lambda a: a[0])
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    goodness = float(np.interp(raw, xs, ys))
    return max(0.0, min(1.0, goodness))


def activate_dfs_highlight(date, enabled=True, salary_path=None, slate=None, games=None):
    """Turn on DFS value highlighting of player names for this report date.

    Reads the cached slate, so it only lights up names in games already generated. For a
    fully highlighted set, generate the slate first and then re-render from cache.

    `slate` picks between several DK exports for one date; without it, the report's own
    games are used to find the export that prices them. When nothing can be loaded the
    reason is printed -- an untinted report otherwise looks identical whether the slate was
    missing, ambiguous, or simply had no player worth tinting.
    """
    try:
        from dfs import highlight
    except Exception:
        return False
    if not enabled:
        highlight.set_index({})
        return False
    index, note = highlight.activate(date, salary_path=salary_path, slate=slate, games=games)
    if index:
        print(f"🎨 DFS value highlighting on for {len(index)} players — {highlight.legend()}")
    elif note:
        print(f"⚠️ DFS highlighting off: {note}")
    return bool(index)


def _dfs_highlight_legend():
    """[(tier, rgb), ...] when value highlighting is active, else []."""
    try:
        from dfs import highlight
        if not highlight.active():
            return []
        return highlight.tier_colors()
    except Exception:
        return []


def _dfs_highlight(column, value):
    """(tier, rgb, value) for a player-name cell, or None when highlighting is off.

    Imported lazily so the report never hard-depends on the dfs package, and failure-safe
    so a missing or broken slate can only cost the tint, never the report.
    """
    try:
        from dfs import highlight
        return highlight.lookup(column, value)
    except Exception:
        return None


def _dfs_own_mark(column, value):
    """Trailing ownership character for a player-name cell, or "" when unavailable.

    Appended to the *displayed* text only. The tint lookup keys off the underlying frame
    value, so decorating the cell cannot break name matching.
    """
    try:
        from dfs import highlight
        found = highlight.own_lookup(column, value)
        return f" {found[0]}" if found else ""
    except Exception:
        return ""


# Point size for the trailing ownership glyph. Two above the 9pt name it follows: the marks
# differ by fill *shape* (◔ against ◕) rather than by height, and at 9pt that distinction is
# the first thing lost when the sheet is zoomed to 85%.
_OWN_MARK_SIZE = 11

_HEAT_BAD_CUTOFF = 0.15
_HEAT_GOOD_CUTOFF = 0.85
_HEAT_BAD_RGB = (218, 86, 76)
_HEAT_MID_RGB = (255, 229, 145)
_HEAT_GOOD_RGB = (84, 169, 105)


def _heat_rgb(goodness):
    """Bounded red->amber->green gradient for goodness in [0,1].

    Values at or below the bad cutoff share one decisive red; values at or above the
    good cutoff share one decisive green.  The middle 70% receives the full gradient.
    This is intentionally winsorized at the color layer: an absurd outlier cannot make
    every merely good or bad value look neutral, while direction remains controlled by
    each metric's league-reference anchors in ``_HEAT_METRICS``.
    """
    if goodness is None:
        return None
    try:
        score = float(goodness)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    if score <= _HEAT_BAD_CUTOFF:
        return _HEAT_BAD_RGB
    if score >= _HEAT_GOOD_CUTOFF:
        return _HEAT_GOOD_RGB
    if score <= 0.5:
        t = (score - _HEAT_BAD_CUTOFF) / (0.5 - _HEAT_BAD_CUTOFF)
        a, b = _HEAT_BAD_RGB, _HEAT_MID_RGB
    else:
        t = (score - 0.5) / (_HEAT_GOOD_CUTOFF - 0.5)
        a, b = _HEAT_MID_RGB, _HEAT_GOOD_RGB
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def _resolve_heat_context(heading_text):
    """Infer a heat context name from a section heading (used by the HTML path)."""
    if not heading_text:
        return None
    text = str(heading_text).lower()
    if "pitch-type" in text or "per-batter" in text or "swing decision" in text:
        return "pitch_matchup"
    # Checked before the "allowed" rule: the team-defence heading also says "allowed",
    # and pitching_allowed would colour hits-saved as though it were a pitching stat.
    if "team defense" in text:
        return "team_defense"
    if "opposing pitching allowed" in text or "allowed" in text:
        return "pitching_allowed"
    if "offense" in text or "lineup" in text or "hot" in text or "batter arsenal" in text or "hitter" in text:
        return "offense"
    if "arsenal" in text:
        return "arsenal"
    if "bullpen" in text or "relief" in text or "starter" in text or "pitcher" in text:
        return "pitching"
    if "baserunning" in text:
        return "baserunning"
    return None


def _heat_metric_for(context, column):
    """Resolve a column header to a metric id given a context name or explicit map."""
    if not context:
        return None
    mapping = context if isinstance(context, dict) else _HEAT_CONTEXTS.get(context)
    if not mapping:
        return None
    return mapping.get(str(column).strip())


class ScoutingPDF(FPDF):
    def __init__(self, orientation="P"):
        super().__init__(orientation=orientation, format="A4")
        self.compact_tables = str(orientation).upper().startswith("L")
        self.add_font('DejaVu', '', 'dejavu-fonts/ttf/DejaVuSans.ttf', uni=True)
        self.add_font('DejaVu', 'B', 'dejavu-fonts/ttf/DejaVuSans-Bold.ttf', uni=True)
        self.add_font('DejaVu', 'I', 'dejavu-fonts/ttf/DejaVuSans-Oblique.ttf', uni=True)
        self.set_auto_page_break(auto=True, margin=12)
        self.set_font('DejaVu', '', 10)

    def header(self):
        if self.page_no() == 1:
            self.set_font('DejaVu', '', 10)
            self.cell(0, 10, 'Game Scouting Report', ln=True, align='C')


    def section_title(self, title):
        self.set_font('DejaVu', 'B', 10)
        self.set_fill_color(200, 220, 250)
        self.set_text_color(15, 25, 50)
        self.cell(0, 7, title, ln=True, fill=True)
        self.set_text_color(0, 0, 0)
        self.set_fill_color(255, 255, 255)
        self.ln(2)

    @staticmethod
    def _fmt_cell(value):
        """Render-friendly cell text: round floats, strip runaway/repeating decimals."""
        if isinstance(value, float):
            if pd.isna(value):
                return ""
            if value == int(value):
                return str(int(value))
            return f"{round(value, 3):g}"
        return str(value)

    def _draw_table(self, df, x0, avail_width, max_rows=20, heat=None, column_fill=None, precomputed_widths=None):
        """Draw one table at (current y, x0) within avail_width.

        column_fill: optional {column_name: [rgb_or_None per data row]} to force a cell's fill
        (e.g. color the bullpen L7 column by each reliever's availability).
        precomputed_widths: optional raw per-column widths so paired comparison tables align.
        """
        column_fill = column_fill or {}
        pad = 2.5 if self.compact_tables else 4
        data_font = 8 if self.compact_tables else 10
        header_font = 8 if self.compact_tables else 10
        row_h = 5.7 if self.compact_tables else 7.4

        cells = [[self._fmt_cell(v) for v in row] for row in df.itertuples(index=False)]

        self.set_font('DejaVu', '', data_font)
        if precomputed_widths is not None and len(precomputed_widths) == len(df.columns):
            col_widths = list(precomputed_widths)
        else:
            header_widths = [self.get_string_width(str(col)) + pad for col in df.columns]
            data_widths = [
                max([self.get_string_width(cells[r][c]) for r in range(len(cells))] + [0]) + pad
                for c in range(len(df.columns))
            ]
            col_widths = [max(header_widths[i], data_widths[i]) for i in range(len(df.columns))]

        total = sum(col_widths)
        if total > avail_width:
            scale = avail_width / total
            col_widths = [w * scale for w in col_widths]
            if scale < 0.68:
                data_font = header_font = 8

        metrics = [_heat_metric_for(heat, col) for col in df.columns]
        columns = list(df.columns)
        override_idx = {columns.index(c): fills for c, fills in column_fill.items() if c in columns}

        # Header row.
        self.set_x(x0)
        self.set_font('DejaVu', 'B', header_font)
        self.set_fill_color(210, 225, 245)
        self.set_text_color(20, 30, 55)
        for col, width in zip(df.columns, col_widths):
            self.cell(width, row_h, str(col), border=1, align='C', fill=True)
        self.ln()

        # Data rows.
        self.set_font('DejaVu', '', data_font)
        self.set_text_color(0, 0, 0)
        for i, row_cells in enumerate(cells, start=1):
            if i > max_rows:
                self.set_x(x0)
                self.cell(sum(col_widths), row_h, '... table truncated ...', ln=True)
                break
            self.set_x(x0)
            zebra = (250, 250, 250) if i % 2 else (255, 255, 255)
            for col_i, (width, metric, item) in enumerate(zip(col_widths, metrics, row_cells)):
                fill_rgb = None
                if col_i in override_idx:
                    fills = override_idx[col_i]
                    if i - 1 < len(fills):
                        fill_rgb = fills[i - 1]
                if fill_rgb is None and metric:
                    fill_rgb = _heat_rgb(_stat_percentile(metric, item))
                if fill_rgb is None:
                    fill_rgb = zebra
                self.set_fill_color(*fill_rgb)
                self.cell(width, row_h, item, border=1, align='C', fill=True)
            self.ln()
        self.set_fill_color(255, 255, 255)
        return sum(col_widths)

    def add_dataframe(self, df, max_rows=20, heat=None):
        if df is None or df.empty:
            self.set_font('DejaVu', '', 9)
            self.cell(0, 6, "No data available.", ln=True)
            return
        usable = self.w - self.l_margin - self.r_margin
        self._draw_table(df, self.l_margin, usable, max_rows=max_rows, heat=heat)
        self.ln(2)

    def add_dataframes_side_by_side(self, specs, gap=8):
        """Render several (title, df, heat[, column_fill]) tables in a row, each in its own column."""
        specs = [s for s in specs if s[1] is not None and not s[1].empty]
        if not specs:
            return
        usable = self.w - self.l_margin - self.r_margin
        col_w = (usable - gap * (len(specs) - 1)) / len(specs)

        # If the tables share identical columns (a comparison), give them the same column
        # widths so they line up cell-for-cell across panels.
        shared_widths = None
        col_sets = {tuple(s[1].columns) for s in specs}
        if len(col_sets) == 1:
            cols = list(specs[0][1].columns)
            pad = 2.5 if self.compact_tables else 4
            self.set_font('DejaVu', '', 8 if self.compact_tables else 10)
            shared_widths = []
            for c in cols:
                w = self.get_string_width(str(c)) + pad
                for spec in specs:
                    for v in spec[1][c]:
                        w = max(w, self.get_string_width(self._fmt_cell(v)) + pad)
                shared_widths.append(w)

        y_start = self.get_y()
        max_y = y_start
        for idx, spec in enumerate(specs):
            title, df, heat = spec[0], spec[1], spec[2]
            column_fill = spec[3] if len(spec) > 3 else None
            x0 = self.l_margin + idx * (col_w + gap)
            self.set_xy(x0, y_start)
            if title:
                self.set_font('DejaVu', 'B', 9)
                self.cell(col_w, 6, title, ln=True)
                self.set_x(x0)
            self._draw_table(df, x0, col_w, max_rows=20, heat=heat, column_fill=column_fill, precomputed_widths=shared_widths)
            max_y = max(max_y, self.get_y())
        self.set_y(max_y + 2)

    def add_actionable_game_plan(self, sections):
        self.section_title("Actionable Game Plan")
        usable_width = self.w - self.l_margin - self.r_margin

        for section in sections:
            self.set_font('DejaVu', 'B', 8)
            self.multi_cell(usable_width, 5, str(section.get("title", "Key Note")))
            self.set_font('DejaVu', '', 7)

            for point in section.get("points", []):
                safe_point = str(point).replace("–", "-").replace("—", "-").replace("\u00a0", " ")
                safe_point = safe_point.encode("ascii", "ignore").decode()
                self.multi_cell(usable_width, 4, f"- {safe_point}")

            self.ln(1)

            if self.get_y() > self.h - 30:
                self.add_page()

    def add_starter_summary(self, info):
        self.section_title("Starting Pitcher Overview")
        self.set_font('DejaVu', '', 8)

        def extract_val(val):
            if hasattr(val, "values"):
                return val.values[0]
            return val

        name = extract_val(info.get('Name', 'Unknown'))
        throws = extract_val(info.get('Throws', '?'))
        try:
            gs = extract_val(info.get('GS', 'N/A'))
            ip = round(extract_val(info.get('IP', 0.0)), 1)
            era = round(extract_val(info.get('ERA', 0.0)), 2)
            fip = round(extract_val(info.get('FIP', 0.0)), 2)
            k_pct = round(extract_val(info.get('K%', 0.0)), 3)
            bb_pct = round(extract_val(info.get('BB%', 0.0)), 3)
            whip = round(extract_val(info.get('WHIP', 0.0)), 2)
            hr = extract_val(info.get('HR', 'N/A'))
            hr9 = extract_val(info.get('HR/9', 'N/A'))
        except:
            ip = era = fip = k_pct = bb_pct = whip = 0
            hr = hr9 = 'N/A'

        line = (
            f"{name} ({throws})  |  GS: {gs}  |  IP: {ip}  |  ERA: {era}  |  "
            f"FIP: {fip}  |  K%: {k_pct}%  |  BB%: {bb_pct}%  |  WHIP: {whip}  |  "
            f"HR: {hr}  |  HR/9: {hr9}"
        )

        self.multi_cell(0, 5, line)
        self.ln(5)

    def add_images_side_by_side(self, image_paths, title=None, width=60, spacing=5):
        if title:
            self.section_title(title)

        max_width = self.w - self.l_margin - self.r_margin
        x_start = self.get_x()
        y = self.get_y()
        x = x_start
        max_row_height = 0

        for img_path in image_paths:
            img_height = width * 0.75  # Adjust if your images have a different aspect ratio

            # Wrap to next row if image won't fit horizontally
            if x + width > self.w - self.r_margin:
                x = x_start
                y += max_row_height + spacing
                max_row_height = 0

            # Add new page if image won't fit vertically
            if y + img_height > self.h - self.b_margin:
                self.add_page()
                if title:
                    self.section_title(title)
                x = x_start
                y = self.get_y()
                max_row_height = 0

            if not os.path.exists(img_path):
                self.set_xy(x, y)
                self.set_font('DejaVu', 'I', 7)
                self.cell(width, 5, f"Missing: {os.path.basename(img_path)}", ln=0)
            else:
                self.image(img_path, x=x, y=y, w=width)
                max_row_height = max(max_row_height, img_height)

            x += width + spacing

        # Move cursor below the last row of images
        self.set_y(y + max_row_height + 30)


    
    def _format_last10(self, recent_df):
        cols = ["Date", "Opp", "Site", "Score", "Result"]
        if recent_df is None or recent_df.empty:
            return pd.DataFrame(columns=cols)
        rows = []
        for _, row in recent_df.iterrows():
            opp = get_team_abbreviation(row.get("Opponent", "")) or row.get("Opponent", "")
            rows.append({
                "Date": str(row.get("Date", "")),
                "Opp": opp,
                "Site": "vs" if row.get("Home/Away") == "Home" else "@",
                "Score": str(row.get("Score", "")),
                "Result": str(row.get("Result", "")),
            })
        return pd.DataFrame(rows, columns=cols)

    def add_team_performance_and_last10(self, home_team, home_perf, home_recent, away_team, away_perf, away_recent):
        self.section_title("Team Summary")

        # Side-by-side comparison table so the two teams read against each other.
        keys = [k for k in home_perf.keys() if k != "Error"]
        for k in away_perf.keys():
            if k not in keys and k != "Error":
                keys.append(k)
        comp_rows = [{"Metric": k, away_team: str(away_perf.get(k, "")), home_team: str(home_perf.get(k, ""))} for k in keys]
        self.add_dataframe(pd.DataFrame(comp_rows, columns=["Metric", away_team, home_team]), max_rows=30)

        # Clean Last-10 tables, side by side, to read the two teams against each other.
        self.section_title("Last 10 Games")
        self.add_dataframes_side_by_side([
            (away_team, self._format_last10(away_recent), "last10"),
            (home_team, self._format_last10(home_recent), "last10"),
        ])


    

    def add_transactions_section(self, transactions):
        if not transactions:
            return

        self.section_title("Recent Team Transactions")
        self.set_font('DejaVu', '', 8)

        def sanitize_line(line):
            line = line.replace("–", "-").replace("—", "-").replace("\u00a0", " ")
            return line.encode("ascii", "ignore").decode()

        width = self.w - self.l_margin - self.r_margin

        for line in transactions:
            if not line.strip():
                self.ln(2)
            else:
                try:
                    safe_line = sanitize_line(line)
                    self.set_x(self.l_margin)
                    self.multi_cell(width, 5, safe_line)
                except FPDFException as e:
                    print(f"⚠️ STILL failed to render:\nRaw: {repr(line)}\nSanitized: {repr(safe_line)}\nError: {e}")
                    fallback = safe_line[:80] + "..." if len(safe_line) > 80 else safe_line
                    self.set_x(self.l_margin)
                    self.multi_cell(width, 5, fallback)

        self.ln(5)


    def add_matchup_summaries(self, df, pitcher, title="Matchup History vs Starter", min_pa=3):
        table = build_matchup_table(df, pitcher, min_pa=min_pa)
        if table is None or table.empty:
            return
        self.section_title(title)
        self.add_dataframe(table, heat="offense", max_rows=15)

    def add_division_standings_side_by_side(self, df1, title1, df2, title2, team1, team2):
        self.section_title("Division Standings")
        self.set_font('DejaVu', '', 8)
        
        col_width = (self.w - self.l_margin - self.r_margin) / 2 - 2

        def draw_table(df, title, team_to_bold):
            self.set_font('DejaVu', 'B', 9)
            self.cell(col_width, 5, title, ln=0)
            self.ln(5)

            for idx, row in df.iterrows():
                text = f"{row['Tm']}  {row['W']}-{row['L']}  GB: {row['GB']}"
                if row['Tm'].lower() == team_to_bold.lower():
                    self.set_font('DejaVu', 'B', 8)
                else:
                    self.set_font('DejaVu', '', 8)
                self.multi_cell(col_width, 5, text)

        x_start = self.get_x()
        y_start = self.get_y()

        self.set_xy(x_start, y_start)
        draw_table(df1, title1, team1)

        self.set_xy(x_start + col_width + 4, y_start)
        draw_table(df2, title2, team2)

        self.set_y(y_start + max(len(df1), len(df2)) * 5 + 10)

    def add_starter_gamelogs(self, logs, title="Last 5 Starts"):
        if not logs:
            return

        self.section_title(title)
        df = pd.DataFrame(logs)

        # Per-game ERA (replaces raw pitch count).
        if 'ER' in df.columns and 'IP' in df.columns:
            ip_true = [_ip_to_float(v) for v in df['IP']]
            er_vals = pd.to_numeric(df['ER'], errors='coerce').fillna(0)
            df['ERA'] = [round(er * 9 / ip, 2) if ip else 0.0 for er, ip in zip(er_vals, ip_true)]

        cols = [c for c in ['Date', 'Opponent', 'IP', 'H', 'R', 'ER', 'BB', 'SO', 'HR', 'ERA', 'Game Score'] if c in df.columns]
        df = df[cols]

        # TOTAL / AVG summary row: sums for counting stats, combined ERA, avg Game Score.
        sum_cols = [c for c in ['H', 'R', 'ER', 'BB', 'SO', 'HR'] if c in df.columns]
        total_ip = sum(_ip_to_float(v) for v in df['IP']) if 'IP' in df.columns else 0.0
        total_er = float(pd.to_numeric(df['ER'], errors='coerce').fillna(0).sum()) if 'ER' in df.columns else 0.0
        summary = {}
        for c in df.columns:
            if c == 'Date':
                summary[c] = 'TOTAL / AVG'
            elif c == 'Opponent':
                summary[c] = f'{len(df)} starts'
            elif c == 'IP':
                summary[c] = round(total_ip, 1)
            elif c == 'ERA':
                summary[c] = round(total_er * 9 / total_ip, 2) if total_ip else 0.0
            elif c == 'Game Score':
                summary[c] = round(pd.to_numeric(df[c], errors='coerce').mean(), 1)
            elif c in sum_cols:
                summary[c] = int(pd.to_numeric(df[c], errors='coerce').fillna(0).sum())
            else:
                summary[c] = ''
        df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
        self.add_dataframe(df)




def generate_report_with_fpdf(
    home_lineup_df, home_bullpen_df, home_starter_info, home_hotcold_df, home_arsenal_df, home_pitcher, home_team_performance,
    away_lineup_df, away_bullpen_df, away_starter_info, away_hotcold_df, away_arsenal_df, away_pitcher, away_team_performance,
    game_date, home_team, away_team, home_team_id, away_team_id,
    home_baserunning_df, away_baserunning_df,
    home_defense_df, away_defense_df, home_last_10, away_last_10,
    home_splits_df, away_splits_df,
    output_dir="scouting_reports",
    advanced_context=None,
):
    advanced_context = advanced_context or {}
    os.makedirs(output_dir, exist_ok=True)
    filename = f"scouting_report_{game_date}_{away_team}_{home_team}.pdf"
    output_path = os.path.join(output_dir, filename)

    pdf = ScoutingPDF(orientation="L")
    pdf.add_page()

    pdf.set_font('DejaVu', '', 10)
    pdf.cell(0, 10, 'Game Scouting Report', ln=True, align='C')

    pdf.set_font('DejaVu', '', 8)
    pdf.cell(0, 8, f'Teams: {away_team} vs {home_team}', ln=True)
    pdf.cell(0, 8, f'Date: {game_date}', ln=True)
    hp_umpire = advanced_context.get("environment", {}).get("hp_umpire", "Not yet assigned")
    pdf.cell(0, 8, f'Home Plate Umpire: {hp_umpire}', ln=True)
    pdf.ln(6)

    hitter_composite = advanced_context.get("hitter_composite", pd.DataFrame())
    lineup_splits_by_team = {
        away_team: advanced_context.get("away_lineup_splits", pd.DataFrame()),
        home_team: advanced_context.get("home_lineup_splits", pd.DataFrame()),
    }
    opp_pitching_by_team = {
        away_team: advanced_context.get("away_opp_pitching", pd.DataFrame()),
        home_team: advanced_context.get("home_opp_pitching", pd.DataFrame()),
    }
    arsenal_matchup_by_team = {
        away_team: advanced_context.get("away_arsenal_matchup", pd.DataFrame()),
        home_team: advanced_context.get("home_arsenal_matchup", pd.DataFrame()),
    }
    arsenal_drilldown_by_team = {
        away_team: advanced_context.get("away_arsenal_drilldown", pd.DataFrame()),
        home_team: advanced_context.get("home_arsenal_drilldown", pd.DataFrame()),
    }
    opp_sp_by_team = {away_team: _first_name(home_starter_info), home_team: _first_name(away_starter_info)}
    rest_schedule = pd.concat([
        advanced_context.get("away_rest_schedule", pd.DataFrame()),
        advanced_context.get("home_rest_schedule", pd.DataFrame()),
    ], ignore_index=True)

    pdf.section_title("Team Summary with Last 10 Games")
    pdf.add_team_performance_and_last10(
        home_team, home_team_performance, home_last_10,
        away_team, away_team_performance, away_last_10
    )

    if rest_schedule is not None and not rest_schedule.empty:
        pdf.section_title("Rest & Schedule Spot")
        pdf.add_dataframe(rest_schedule[["Team", "Rest", "SP Rest", "Note", "Pen A/T"]], max_rows=2)

    report_season = int(str(game_date)[:4])
    for team, lineup_df, splits_df, hotcold_df, starter_info, arsenal_df, pitcher, bullpen_df, baserun_df, defense_df, own_starter_id, opp_team in [
        (home_team, home_lineup_df, home_splits_df, home_hotcold_df, home_starter_info, home_arsenal_df, away_pitcher, home_bullpen_df, home_baserunning_df, home_defense_df, home_pitcher, away_team),
        (away_team, away_lineup_df, away_splits_df, away_hotcold_df, away_starter_info, away_arsenal_df, home_pitcher, away_bullpen_df, away_baserunning_df, away_defense_df, away_pitcher, home_team)
    ]:
        pdf.add_page()
        pitcher_hand = get_pitcher_handedness(pitcher) or 'R'

        pdf.section_title(f'{team} - Starting Lineup (Off Szn/L28: 100 = league avg)')
        lineup_with_off = _lineup_with_offense(lineup_df, hitter_composite)
        off_cols = [c for c in ["Spot", "Name", "Pos", "Bats", "OPS", "Off Szn", "Off L28", "ISO", "HR", "SB", "Lineup Confidence"] if lineup_with_off is not None and not lineup_with_off.empty and c in lineup_with_off.columns]
        pdf.add_dataframe(lineup_with_off[off_cols] if off_cols else lineup_df, heat="offense")

        team_splits = lineup_splits_by_team.get(team, pd.DataFrame())
        if team_splits is not None and not team_splits.empty:
            pdf.section_title(f'{team} - Lineup Offense Splits (Home/Away park-adj; * = opp SP hand)')
            pdf.add_dataframe(team_splits[["Split", "PA", "OPS", "xwOBA", "ISO", "K%", "BB%", "HardHit%", "HR%"]], max_rows=8, heat="offense")

        team_opp_pitching = opp_pitching_by_team.get(team, pd.DataFrame())
        if team_opp_pitching is not None and not team_opp_pitching.empty:
            pdf.section_title(f'{team} - Opposing Pitching Allowed (Starter + Bullpen)')
            pdf.add_dataframe(team_opp_pitching[["Unit", "Split", "PA", "OPS", "xwOBA", "ISO", "K%", "BB%", "HardHit%", "HR%"]], max_rows=14, heat="pitching_allowed")

        opp_sp_name = opp_sp_by_team.get(team, "SP")
        team_matchup = arsenal_matchup_by_team.get(team, pd.DataFrame())
        if team_matchup is not None and not team_matchup.empty:
            pdf.section_title(f'{team} - Pitch-Type Value & Swing Decisions vs {opp_sp_name} (RV/100 high = hitter)')
            pdf.add_dataframe(team_matchup[["Pitch", "Usage%", "Pitches", "RV/100", "Whiff%", "Chase%", "Contact%", "xwOBA", "Edge"]], max_rows=6, heat="pitch_matchup")
        team_drill = arsenal_drilldown_by_team.get(team, pd.DataFrame())
        if team_drill is not None and not team_drill.empty:
            pdf.section_title(f'{team} - Per-Batter vs {opp_sp_name} Arsenal')
            pdf.add_dataframe(team_drill[["Name", "Pitches", "RV/100", "Whiff%", "Chase%", "Contact%", "xwOBA", "Edge"]], max_rows=9, heat="pitch_matchup")

        pdf.section_title(f'{team} - Season Splits vs {pitcher_hand} Pitching')
        split_cols = [
            'Name',
            f'PA vs {pitcher_hand}',
            f'AVG vs {pitcher_hand}',
            f'OBP vs {pitcher_hand}',
            f'SLG vs {pitcher_hand}',
            f'OPS vs {pitcher_hand}',
            f'ISO vs {pitcher_hand}',
            f'HR vs {pitcher_hand}',
            f'BB% vs {pitcher_hand}',
            f'K% vs {pitcher_hand}'
        ]

        split_heat = {
            f'OPS vs {pitcher_hand}': 'ops_off',
            f'OBP vs {pitcher_hand}': 'obp_off',
            f'SLG vs {pitcher_hand}': 'slg_off',
            f'AVG vs {pitcher_hand}': 'avg_off',
            f'ISO vs {pitcher_hand}': 'iso',
            f'BB% vs {pitcher_hand}': 'bb_pct_bat',
            f'K% vs {pitcher_hand}': 'k_pct_bat',
        }
        if not splits_df.empty:
            try:
                pdf.add_dataframe(splits_df[split_cols].sort_values(f'PA vs {pitcher_hand}', ascending=False), heat=split_heat)
            except KeyError as e:
                pdf.cell(0, 8, f"⚠️ Split data columns missing: {e}", ln=True)
        else:
            pdf.cell(0, 8, "No split data available.", ln=True)

        pdf.section_title(f"{team} - Hot / Cold Hitters (Last 14 Days vs own season)")
        pdf.add_dataframe(
            _hot_cold_sorted(hotcold_df)[
                [c for c in ['Name', 'PA', 'OPS', 'xwOBA', 'Szn xwOBA', 'ΔxwOBA',
                             'HardHit%', 'Chase%', 'Z-Con%', 'Status']
                 if c in hotcold_df.columns]],
            heat="hot_cold")
        pdf.add_matchup_summaries(hotcold_df, pitcher)

        pdf.section_title(f'{team} - Starter Info')
        pdf.add_starter_summary(starter_info)
        pdf.add_starter_gamelogs(starter_info.get("Last Starts", []))

        vs_team_logs = get_starts_vs_team(own_starter_id, opp_team, report_season, _pregame_end_date(game_date) or game_date)
        if vs_team_logs:
            pdf.add_starter_gamelogs(vs_team_logs, title=f"{starter_info.get('Name', 'Starter')} vs {opp_team} This Season")

        pdf.section_title(f'{team} - Pitching Arsenal')
        pdf.add_dataframe(arsenal_df.sort_values('Usage %', ascending=False), heat="arsenal")

        pdf.add_images_side_by_side(
            image_paths=[
                f"plots/{pitcher} heatmap_all.png",
                f"plots/{pitcher} heatmap_called_strikes.png",
                f"plots/{pitcher} heatmap_whiffs.png"
            ],
            title="Visuals",
            width=60
        )

        pdf.section_title(f'{team} - Bullpen Scouting Report (L7: - off / a app / s save / b blown / h hold)')
        if bullpen_df is not None and not bullpen_df.empty:
            bp = bullpen_df.copy()
            ip = pd.to_numeric(bp.get('IP'), errors='coerce').fillna(0.0)
            ip_total = float(ip.sum())

            def _ip_weighted(col):
                vals = pd.to_numeric(bp.get(col), errors='coerce').fillna(0.0)
                return round(float((vals * ip).sum() / ip_total), 2) if ip_total else 0.0

            has_app = 'G' in bp.columns
            has_l7 = 'L7 Usage' in bp.columns
            summary_row = {
                'Name': 'BULLPEN (IP-wtd)',
                'App': int(pd.to_numeric(bp['G'], errors='coerce').fillna(0).sum()) if has_app else '',
                'IP': round(ip_total, 1),
                'ERA': _ip_weighted('ERA'),
                'FIP': _ip_weighted('FIP'),
                'K%': _ip_weighted('K%'),
                'BB%': _ip_weighted('BB%'),
                'WHIP': _ip_weighted('WHIP'),
                'L7': '',
            }
            display_cols = ['Name', 'App', 'IP', 'ERA', 'FIP', 'K%', 'BB%', 'WHIP', 'L7']
            reliever_view = pd.DataFrame({
                'Name': bp['Name'],
                'App': pd.to_numeric(bp['G'], errors='coerce').fillna(0).astype(int) if has_app else '',
                'IP': ip.round(1),
                'ERA': pd.to_numeric(bp['ERA'], errors='coerce').round(2),
                'FIP': pd.to_numeric(bp['FIP'], errors='coerce').round(2),
                'K%': pd.to_numeric(bp['K%'], errors='coerce').round(1),
                'BB%': pd.to_numeric(bp['BB%'], errors='coerce').round(1),
                'WHIP': pd.to_numeric(bp['WHIP'], errors='coerce').round(2),
                'L7': bp['L7 Usage'] if has_l7 else '',
            })
            combined_df = pd.concat([pd.DataFrame([summary_row]), reliever_view], ignore_index=True)[display_cols]
            pdf.add_dataframe(combined_df, heat="pitching")
        else:
            pdf.cell(0, 6, "Bullpen leaderboard unavailable.", ln=True)



        pdf.section_title(f'{team} - Baserunning Leaderboard')
        baserun_cols = ['Name', 'Team', 'BsR', 'Spd', 'SB', 'CS', 'SB_Att', 'SB%']
        if baserun_df is not None and not baserun_df.empty:
            pdf.add_dataframe(baserun_df[[col for col in baserun_cols if col in baserun_df.columns]], heat="baserunning")
        else:
            pdf.cell(0, 6, "Baserunning leaderboard unavailable.", ln=True)

        pdf.section_title(f'{team} - Defensive Leaderboard')
        pdf.add_dataframe(defense_df if defense_df is not None else pd.DataFrame())

        txns = get_recent_transactions(team, days_back=7, as_of_date=game_date)
        formatted_txns = format_transactions_for_report(txns)
        pdf.add_transactions_section(formatted_txns)

    try:
        pdf.output(output_path)
        return output_path
    except Exception as e:
        print(f"PDF Output Failed: {e}")
        print(f"Total pages: {len(pdf.pages)}")
        for i, page in enumerate(pdf.pages):
            print(f"Page {i+1} content: {page}")
        return None


# =====================================================================================
# Shared comparison-report table builders.
#
# Single source of truth for every derived table used by BOTH the comparison PDF and the
# Excel export. Keep table logic HERE (not inside a renderer) so a column added for one
# output — e.g. the defense A+ column, the Last-5 TOTAL/AVG row — can never silently
# disappear from the other. Renderers only choose columns/order and draw; they never
# recompute stats.
# =====================================================================================

_CMP_AVAIL_COLOR = {'Available': (150, 205, 150), 'Monitor': (250, 226, 150), 'Taxed': (233, 150, 140)}
_CMP_ASSIST_BASELINES = {'1B': 7.0, '2B': 26.0, '3B': 18.0, 'SS': 27.0}
_CMP_A_MIN_INN, _CMP_CS_MIN_ATT, _CMP_PB_MIN_INN = 100.0, 15, 150.0


def cmp_available(df, cols, n=None):
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    use = [c for c in cols if c in df.columns]
    out = df[use].copy() if use else pd.DataFrame(columns=cols)
    return out.head(n) if n else out


def cmp_enrich_pythag(performance):
    e = dict(performance or {})
    pct = _safe_number(e.get('Pythagorean Expectation'), None)
    m = re.search(r'(\d+)\s*-\s*(\d+)', str(e.get('Overall Record', '')))
    if pct is not None and m:
        g = int(m.group(1)) + int(m.group(2))
        ew = int(round(g * pct))
        e['Pythag Record'] = f"{ew}-{g - ew} ({('%.3f' % pct).lstrip('0')})"
    else:
        e['Pythag Record'] = e.get('Pythagorean Expectation', '')
    return e


def cmp_team_form_snapshot(away_performance, home_performance, advanced_context, away_team, home_team):
    """One shared Team Form transform for PDF and Excel."""
    advanced_context = advanced_context or {}
    keys = [
        'Overall Record', 'Current Streak', 'Pythag Record',
        'Run Differential', 'Division', 'Division Rank', 'Games Back',
        'Home Record', 'Away Record', 'Last 10 Games', 'Last 20 Games', 'Last 30 Games',
    ]
    rows = [
        {
            'Metric': key,
            away_team: away_performance.get(key, ''),
            home_team: home_performance.get(key, ''),
        }
        for key in keys
    ]
    away_sweeps = advanced_context.get('away_sweeps', {}) or {}
    home_sweeps = advanced_context.get('home_sweeps', {}) or {}
    if away_sweeps or home_sweeps:
        rows.append({
            'Metric': 'Series Swept',
            away_team: away_sweeps.get('Swept Line', ''),
            home_team: home_sweeps.get('Swept Line', ''),
        })
        rows.append({
            'Metric': 'Series Swept By',
            away_team: away_sweeps.get('Been Swept Line', ''),
            home_team: home_sweeps.get('Been Swept Line', ''),
        })
    away_spot = advanced_context.get('away_trip_spot', '')
    home_spot = advanced_context.get('home_trip_spot', '')
    if away_spot or home_spot:
        rows.append({
            'Metric': 'Trip/Stand Spot',
            away_team: away_spot,
            home_team: home_spot,
        })
    away_tz = advanced_context.get('away_time_zone', {}) or {}
    home_tz = advanced_context.get('home_time_zone', {}) or {}
    if away_tz or home_tz:
        rows.append({
            'Metric': 'Time-Zone Load',
            away_team: away_tz.get('Compact', away_tz.get('Summary', '')),
            home_team: home_tz.get('Compact', home_tz.get('Summary', '')),
        })
        projection = advanced_context.get('projection_summary', {}) or {}
        away_adj = _safe_number(projection.get('TZ Net Away pp'), 0.0)
        home_adj = _safe_number(projection.get('TZ Net Home pp'), 0.0)
        rows.append({
            'Metric': 'TZ Model Adj',
            away_team: f"{away_adj:+.1f}pp win",
            home_team: f"{home_adj:+.1f}pp win",
        })
    return pd.DataFrame(rows)


_ENV_GOOD = (176, 214, 168)     # run-scoring environment up
_ENV_MID = (250, 226, 150)
_ENV_BAD = (236, 176, 168)      # run-scoring environment down


def _env_factor_fill(value):
    """Color a park factor from the offense's side: >1 means more runs/HR, so green.
    Factors sit in a narrow band, so 0.99-1.01 stays uncolored rather than call noise."""
    value = _safe_number(value, None)
    if value is None:
        return None
    if value >= 1.05:
        return _ENV_GOOD
    if value <= 0.95:
        return _ENV_BAD
    if value >= 1.02 or value <= 0.98:
        return _ENV_MID
    return None


# The same rule colors two cells written in different vocabularies: Wind carries MLB's
# feed wording ("Out To LF") or our forecast wording, Carry Read carries our own summary.
_ENV_CARRY_UP = ("out to", "adds carry", "carry boost", "hr risk up", "wind out", "slightly out")
_ENV_CARRY_DOWN = ("in from", "kills carry", "mutes carry", "hr risk down", "wind in", "slightly in")


def _env_index_fill(*indices):
    """Color a handedness park-factor row (100 = neutral) by its most extreme index, so a
    neutral-overall park that plays 122 HR for lefties still reads as hitter-friendly."""
    values = [_safe_number(v, None) for v in indices]
    values = [v for v in values if v is not None]
    if not values:
        return None
    extreme = max(values, key=lambda v: abs(v - 100))
    if extreme >= 108:
        return _ENV_GOOD
    if extreme <= 92:
        return _ENV_BAD
    if extreme >= 103 or extreme <= 97:
        return _ENV_MID
    return None


def _env_carry_fill(text):
    low = str(text or "").lower()
    if "neutral" in low or "calm" in low:
        return None
    if any(k in low for k in _ENV_CARRY_UP):
        return _ENV_GOOD
    if any(k in low for k in _ENV_CARRY_DOWN):
        return _ENV_BAD
    return None


def cmp_game_environment(environment):
    """Weather forecast + park factor as one Metric/Value panel.

    Returns (display_df, {column: [rgb_or_None per row]}) like the other cmp_* views."""
    columns = ['Metric', 'Value']
    environment = environment or {}
    park = environment.get('park', {}) or {}
    fc = environment.get('forecast', {}) or {}
    if not environment:
        return pd.DataFrame(columns=columns), {'Value': []}

    rows, fills = [], []

    def add(metric, value, fill=None):
        if value in (None, ''):
            return
        rows.append({'Metric': metric, 'Value': value})
        fills.append(fill)

    venue = environment.get('venue', '')
    park_name = environment.get('park_name', venue)
    add('Venue', venue if venue == park_name else f"{venue}")

    # First pitch in venue-local time -- the hour the forecast below is valid for.
    stamp = _first_pitch_utc(environment.get('game_datetime'), None)
    zone = _VENUE_IANA_TZ.get(_safe_number(environment.get('venue_id'), 0) or 0)
    if stamp is not None and zone:
        local = stamp.replace(tzinfo=ZoneInfo('UTC')).astimezone(ZoneInfo(zone))
        add('First Pitch', local.strftime('%-I:%M %p local') if os.name != 'nt'
            else local.strftime('%#I:%M %p local'))

    roof = fc.get('roof', 'open')
    if roof and roof != 'open':
        add('Roof', {'fixed': 'Fixed (indoors)', 'retractable': 'Retractable',
                     'cover': 'Roof covers seats'}.get(roof, str(roof).title()))

    add('Sky', fc.get('sky'))
    temp = _safe_number(fc.get('temp_f'), None)
    if temp is not None:
        feels = _safe_number(fc.get('feels_f'), None)
        text = f"{temp:.0f}°F"
        if feels is not None and abs(feels - temp) >= 3:
            text += f" (feels {feels:.0f}°)"
        add('Temp', text, _ENV_GOOD if temp >= 80 else _ENV_BAD if temp <= 58 else None)

    wind = fc.get('wind')
    if wind:
        # Only append gusts when the wind reading itself came from the forecast -- pairing
        # an MLB-feed speed with a forecast gust would be mixing two sources in one cell.
        if fc.get('wind_estimated'):
            gust = _safe_number(fc.get('wind_gust_mph'), None)
            speed = _safe_number((fc.get('forecast') or {}).get('wind_mph'), None)
            if gust is not None and speed is not None and gust - speed >= 8:
                wind = f"{wind}, gusts {gust:.0f}"
            wind = f"{wind} ≈"     # park-relative read is estimated, not from MLB
        add('Wind', wind, _env_carry_fill(wind))

    precip = _safe_number(fc.get('precip_pct'), None)
    if precip is not None:
        add('Precip', f"{precip:.0f}% chance",
            _ENV_BAD if precip >= 50 else _ENV_MID if precip >= 30 else None)
    humidity = _safe_number(fc.get('humidity_pct'), None)
    if humidity is not None:
        add('Humidity', f"{humidity:.0f}%")

    add('Park HR Factor', round(_safe_number(park.get('HR'), 1.0), 2),
        _env_factor_fill(park.get('HR')))
    add('Park Run Factor', round(_safe_number(park.get('Runs'), 1.0), 2),
        _env_factor_fill(park.get('Runs')))

    umpire = environment.get('hp_umpire')
    if umpire:
        add('HP Umpire', umpire)
    tag = environment.get('umpire_tag') or get_umpire_tag(umpire)
    if tag:
        rating = tag.get('Rating', '')
        rating_low = str(rating).lower()
        fill = (_ENV_GOOD if 'hitter' in rating_low else
                _ENV_BAD if 'pitcher' in rating_low else None)
        add('Umpire Tag', rating, fill)
        tag_era = _safe_number(tag.get('ERA'), None)
        if tag_era is not None:
            add('Umpire Tag ERA', round(tag_era, 2), fill)
    tendency = environment.get('ump_tendency') or {}
    if tendency:
        add('Umpire Live R/G',
            f"{tendency.get('R/G')} ({_safe_number(tendency.get('vs Avg'), 0):+.2f} vs avg, "
            f"{tendency.get('Games')}g)")

    # 3-year handedness splits (100 = neutral). These are where a park actually bites:
    # Camden plays 122 HR for LHB and 92 for RHB off the same ~neutral overall factor.
    # Fall back to a direct lookup so contexts cached before park_handed existed still
    # show the splits -- the CSVs are memoized, so this costs nothing.
    handed = (environment.get('park_handed')
              or handed_park_factors_for_venue(park_name)
              or handed_park_factors_for_venue(venue)
              or {})
    for hand in ('RHB', 'LHB'):
        split = handed.get(hand)
        if not split:
            continue
        overall = _safe_number(split.get('Park Factor'), None)
        if overall is None:
            continue
        hr = _safe_number(split.get('HR'), 100)
        runs = _safe_number(split.get('R'), 100)
        add(f'PF vs {hand}',
            f"PF {overall:.0f} · HR {hr:.0f} · R {runs:.0f} · "
            f"2B {_safe_number(split.get('2B'), 100):.0f}",
            _env_index_fill(overall, hr, runs))

    add('Carry Read', fc.get('carry'), _env_carry_fill(fc.get('carry')))
    add('Park Profile', park.get('profile'))

    return pd.DataFrame(rows, columns=columns), {'Value': fills}


_PITCH_FAMILY = {
    "FF": "FB", "SI": "FB", "FT": "FB", "FC": "FB",
    "SL": "BB", "CU": "BB", "KC": "BB", "SV": "BB", "ST": "BB", "SC": "BB",
    "CH": "OS", "FS": "OS", "FO": "OS", "EP": "OS", "KN": "OS",
}


def _arsenal_archetype(arsenal_df):
    """Plain-language pitcher archetype from the arsenal mix: velocity band plus which
    family the pitcher actually leans on. Returns (archetype, mix_text, fb_velo)."""
    if arsenal_df is None or arsenal_df.empty or "Pitch" not in arsenal_df.columns:
        return "", "", None
    df = arsenal_df.copy()
    df["_usage"] = pd.to_numeric(df.get("Usage %"), errors="coerce").fillna(0.0)
    df["_velo"] = pd.to_numeric(df.get("release_speed"), errors="coerce")
    df = df.sort_values("_usage", ascending=False)
    total = float(df["_usage"].sum()) or 100.0

    family_usage = {"FB": 0.0, "BB": 0.0, "OS": 0.0}
    for _, row in df.iterrows():
        family = _PITCH_FAMILY.get(str(row["Pitch"]).strip().upper())
        if family:
            family_usage[family] += row["_usage"]
    family_usage = {k: v / total * 100 for k, v in family_usage.items()}

    fastballs = df[df["Pitch"].astype(str).str.upper().map(
        lambda p: _PITCH_FAMILY.get(p) == "FB")]
    fb_velo = float(fastballs["_velo"].iloc[0]) if not fastballs.empty and pd.notna(
        fastballs["_velo"].iloc[0]) else None

    if fb_velo is None:
        band = ""
    elif fb_velo >= 96:
        band = "Power"
    elif fb_velo >= 93:
        band = "Average-velo"
    else:
        band = "Soft-tossing"

    if family_usage["BB"] >= 40:
        lean = "breaking-ball heavy"
    elif family_usage["OS"] >= 25:
        lean = "changeup-forward"
    elif family_usage["FB"] >= 58:
        lean = "fastball-forward"
    else:
        lean = "balanced mix"

    top = df.head(3)
    mix_text = " / ".join(f"{r['Pitch']} {r['_usage']:.0f}%" for _, r in top.iterrows())
    archetype = " ".join(part for part in (band, lean) if part)
    return archetype, mix_text, fb_velo


def cmp_pitcher_type_view(starter_info, arsenal_df, type_results, vs_arsenal, opener=None,
                          batter_arsenal=None, lineup_df=None):
    """What kind of pitcher tonight's starter is, and how this lineup has hit that type.

    Sits next to the Team Hitting Summary so the splits above have a described opponent
    rather than just a handedness. When the listed starter is really an opener, that is
    said first -- every arsenal number below it describes an arm throwing ~1 inning."""
    columns = ['Metric', 'Value']
    starter_info = starter_info or {}
    opener = opener or {}
    rows = []

    def add(metric, value):
        if value not in (None, '') and str(value).strip():
            rows.append({'Metric': metric, 'Value': value})

    add('Pitcher', _starter_label(starter_info, fallback=''))
    if opener.get('is_opener'):
        bulk = opener.get('bulk') or {}
        add('⚠ Opener', f"{opener.get('role')} — {opener.get('opener_rate')}% of starts, "
                        f"avg {opener.get('avg_first_inn')} inn")
        if bulk:
            add('⚠ Real matchup', f"{bulk.get('name')} has followed {bulk.get('share')}% of "
                                  f"his opener games, avg {bulk.get('avg_inn')} inn "
                                  f"(~{round(OPENER_BULK_REPEAT_RATE * 100)}% repeat rate) "
                                  f"— see the Starting Pitchers tab")
    archetype, mix_text, fb_velo = _arsenal_archetype(arsenal_df)
    add('Type', archetype)
    add('Mix', mix_text)
    if fb_velo is not None:
        add('FB Velo', f"{fb_velo:.1f} mph")

    if arsenal_df is not None and not arsenal_df.empty and 'Stuff+' in arsenal_df.columns:
        graded = arsenal_df.copy()
        graded['_stuff'] = pd.to_numeric(graded['Stuff+'], errors='coerce')
        graded['_usage'] = pd.to_numeric(graded.get('Usage %'), errors='coerce')
        # Only pitches that actually carry a Stuff+ grade get weighted, otherwise a
        # missing grade would silently pull the average toward the 100 baseline.
        graded = graded[graded['_stuff'].notna() & graded['_usage'].notna()]
        if not graded.empty and graded['_usage'].sum():
            add('Stuff+ (wtd)',
                round(float((graded['_stuff'] * graded['_usage']).sum() / graded['_usage'].sum()), 1))
            # Stuff+ is scaled to 100, so the weighted mean is nearly always ~100; the
            # out pitch is the part that actually differentiates one starter from another.
            best = graded.loc[graded['_stuff'].idxmax()]
            csw = _safe_number(best.get('CSW%'), None)
            add('Out Pitch', f"{best['Pitch']} — {best['_stuff']:.0f} Stuff+"
                            + (f", {csw:.0f}% CSW" if csw is not None else ""))

    tr = type_results.iloc[0].to_dict() if type_results is not None and not type_results.empty else {}
    add('Comp Bucket', tr.get('Pitcher Type'))
    if tr.get('PA'):
        comps = tr.get('Comps')
        add('Sample', f"{tr.get('PA')} PA vs {comps} comp arms"
                      + (f" — {tr.get('Confidence')} confidence" if tr.get('Confidence') else ""))
        add('Lineup vs Type', f"{tr.get('Type OPS')} OPS "
                              f"(vs {tr.get('Baseline OPS')} {tr.get('Baseline Basis', 'baseline')}, "
                              f"{tr.get('OPS Diff')})")
        add('Record vs Type', f"{tr.get('Record')} ({tr.get('Runs/G')} R/G, {tr.get('Games')}g)")
        add('Volatility', tr.get('Consistency') if tr.get('Consistency') != 'N/A' else '')

    va = vs_arsenal.iloc[0].to_dict() if vs_arsenal is not None and not vs_arsenal.empty else {}
    if va.get('Arsenal Tag'):
        add('vs This Arsenal', f"{va.get('Arsenal Tag')} — {va.get('OPS')} OPS, "
                              f"{va.get('League OPS Pctl')}")

    # The one arsenal number that predicted anything over three seasons. Stated from the
    # hitters' side here, because this panel is read by someone choosing bats.
    measured = _lineup_k_edge_points(batter_arsenal, lineup_df)
    if measured:
        edge, hitters = measured
        direction = ("strikes out more" if edge > 0 else "strikes out less")
        # `Throws` is the long form ("Left"), which read as "LeftHP".
        hand = _pitcher_throw_code(starter_info) or "same-handed "
        add('K Edge vs Shapes',
            f"{edge:+.1f} pts — this lineup {direction} vs his pitch shapes than vs "
            f"{hand}HP generally ({hitters} bats)")

    return pd.DataFrame(rows, columns=columns)


# Over/under-season-baseline shading. Deliberately NOT a good/bad scale: green just means
# "more than his season norm", red "less". Legend on the sheet says so.
_DELTA_UP_STRONG = (150, 200, 145)
_DELTA_UP_MILD = (203, 228, 198)
_DELTA_DOWN_MILD = (247, 208, 201)
_DELTA_DOWN_STRONG = (235, 160, 148)


def _delta_fill(value, baseline, mild, strong):
    value, baseline = _safe_number(value, None), _safe_number(baseline, None)
    if value is None or baseline is None:
        return None
    delta = value - baseline
    if delta >= strong:
        return _DELTA_UP_STRONG
    if delta >= mild:
        return _DELTA_UP_MILD
    if delta <= -strong:
        return _DELTA_DOWN_STRONG
    if delta <= -mild:
        return _DELTA_DOWN_MILD
    return None


def cmp_starter_mix_fills(profile):
    """Per-cell fills for the mix table: usage vs season usage, velo vs season velo.

    Velocity thresholds are tight (0.7 / 1.5 mph) because a mile of fastball is a real
    signal; usage moves in much bigger increments so it needs a wider band.
    """
    mix, season = profile.get("mix"), profile.get("season_mix", {})
    if mix is None or mix.empty:
        return {}
    fills = {}
    for pitch, base in season.items():
        usage_col, velo_col = f"{pitch}%", f"{pitch} mph"
        if usage_col in mix.columns:
            fills[usage_col] = [None] + [_delta_fill(v, base["usage"], 4.0, 9.0)
                                         for v in mix[usage_col].iloc[1:]]
        if velo_col in mix.columns:
            fills[velo_col] = [None] + [_delta_fill(v, base["velo"], 0.7, 1.5)
                                        for v in mix[velo_col].iloc[1:]]
    return fills


def cmp_starter_batted_fills(profile):
    """Batted-ball cells shaded against the same pitcher's season rate."""
    batted, season = profile.get("batted"), profile.get("season_batted", {})
    if batted is None or batted.empty:
        return {}
    bands = {"GB%": (5.0, 12.0), "FB%": (5.0, 12.0), "LD%": (4.0, 9.0),
             "HH%": (5.0, 12.0), "Avg EV": (1.5, 3.5), "Brl%": (3.0, 7.0)}
    bip = pd.to_numeric(batted.get("BIP"), errors="coerce").fillna(0)
    fills = {}
    for col, (mild, strong) in bands.items():
        if col not in batted.columns:
            continue
        # A 2-ball sample produces 50% rates; leave those uncolored rather than shouting.
        fills[col] = [None] + [
            _delta_fill(v, season.get(col), mild, strong) if bip.iloc[i] >= _BATTED_MIN else None
            for i, v in enumerate(batted[col].iloc[1:], start=1)
        ]
    return fills


_PEN_BATTED_COLS = ['Name', 'T', 'BIP', 'GB%', 'FB%', 'LD%', 'HH%', 'Avg EV', 'Max EV', 'Brl%']
_PEN_BATTED_ARMS = 8          # arms shown under the group row, by batted balls allowed


def build_bullpen_batted(bullpen_df, statcast_df, team_abbr=None, as_of_date=None):
    """Batted-ball profile for the pen: the group, then arm by arm.

    The mirror of the starter's per-start block, with the one change the subject demands.
    A starter is one arm across several starts, so his block is sliced by start against his
    own season baseline. A bullpen is several arms in one night, so this is sliced by ARM
    against the pen's own aggregate -- which is the comparison that matters when the
    question is "who comes in if this game turns, and does he give up air".

    Costs nothing: it slices the league-wide statcast frame the pipeline already loaded
    rather than pulling per-pitcher, so the arms come free with the report.
    """
    if (bullpen_df is None or bullpen_df.empty or statcast_df is None
            or statcast_df.empty or "pitcher" not in statcast_df.columns):
        return pd.DataFrame(columns=_PEN_BATTED_COLS), {}

    # Reindexed rather than read with .get: a frame cached before MLBAM was carried has no
    # such column, .get returns None, and pd.to_numeric(None) is a scalar NaN with no
    # .dropna() -- which is a crash rather than the empty result it should be.
    keys = pd.to_numeric(bullpen_df.reindex(columns=["MLBAM"])["MLBAM"], errors="coerce")
    if keys.isna().all() and team_abbr:
        # Payloads cached before MLBAM was carried on the pen frame. Resolving off the
        # roster costs one cached request and is what lets the block appear on a re-render
        # instead of waiting for the next full generation of every game.
        try:
            roster = get_team_roster(team_abbr, as_of_date=as_of_date)
            by_name = {remove_accents(p.get("person", {}).get("fullName", "")):
                       p.get("person", {}).get("id") for p in roster}
            keys = bullpen_df.reindex(columns=["Name"])["Name"].map(
                lambda n: by_name.get(remove_accents(str(n))))
            keys = pd.to_numeric(keys, errors="coerce")
        except Exception:
            pass
    ids = keys.dropna().astype(int)
    if ids.empty:
        return pd.DataFrame(columns=_PEN_BATTED_COLS), {}
    pitcher_col = pd.to_numeric(statcast_df["pitcher"], errors="coerce")
    pen_rows = statcast_df[pitcher_col.isin(set(ids))]
    if pen_rows.empty:
        return pd.DataFrame(columns=_PEN_BATTED_COLS), {}

    group = _pitch_rate_block(pen_rows)
    hands = dict(zip(keys, bullpen_df.reindex(columns=["Throws"])["Throws"]))
    names = dict(zip(keys, bullpen_df.reindex(columns=["Name"])["Name"]))

    rows = [{"Name": "BULLPEN", "T": "", **group}]
    per_arm = []
    for pitcher_id in ids.unique():
        arm_rows = statcast_df[pitcher_col == pitcher_id]
        if arm_rows.empty:
            continue
        block = _pitch_rate_block(arm_rows)
        if not block["BIP"]:
            continue
        per_arm.append({"Name": names.get(pitcher_id, str(pitcher_id)),
                        "T": _pen_hand(hands.get(pitcher_id)), **block})
    # By sample, so the arms whose rates mean anything sit at the top.
    per_arm.sort(key=lambda r: -r["BIP"])
    rows.extend(per_arm[:_PEN_BATTED_ARMS])

    return pd.DataFrame(rows, columns=_PEN_BATTED_COLS), group


def cmp_bullpen_batted_fills(table, group):
    """Each arm's batted-ball rates shaded against the pen's own aggregate."""
    if table is None or table.empty or not group:
        return {}
    bands = {"GB%": (5.0, 12.0), "FB%": (5.0, 12.0), "LD%": (4.0, 9.0),
             "HH%": (5.0, 12.0), "Avg EV": (1.5, 3.5), "Brl%": (3.0, 7.0)}
    bip = pd.to_numeric(table.get("BIP"), errors="coerce").fillna(0)
    fills = {}
    for col, (mild, strong) in bands.items():
        if col not in table.columns:
            continue
        # Row 0 is the baseline itself; a thin sample produces 50% rates and is left plain.
        fills[col] = [None] + [
            _delta_fill(v, group.get(col), mild, strong) if bip.iloc[i] >= _BATTED_MIN else None
            for i, v in enumerate(table[col].iloc[1:], start=1)
        ]
    return fills


_OPENER_ALERT = (250, 214, 150)      # amber: the listed starter isn't the real matchup


def cmp_opener_view(profile, starter_info):
    """Whether tonight's listed starter is really an opener, and who follows him."""
    columns = ['Metric', 'Value']
    profile = profile or {}
    if not profile.get("role"):
        return pd.DataFrame(columns=columns), {}
    rows, fills = [], []

    def add(metric, value, fill=None):
        if value not in (None, '') and str(value).strip():
            rows.append({'Metric': metric, 'Value': value})
            fills.append(fill)

    add('Role', profile['role'], _OPENER_ALERT if profile.get('is_opener') else None)
    add('Games opened', f"{profile.get('starts', 0)} "
                        f"({profile.get('opener_starts', 0)} as opener, "
                        f"{profile.get('opener_rate', 0)}%)")
    if profile.get('avg_first_inn') is not None:
        add('Avg innings as starter', f"{profile['avg_first_inn']} innings")
    # "Likely" overstated it. Across 371 opener games the arm who most often followed
    # before follows again about a third of the time, and that does not improve with more
    # history -- so these are labelled as what they are, a record of who has followed.
    for i, cand in enumerate(profile.get('candidates', [])):
        label = 'Has followed' if i == 0 else f'Also followed {i}'
        add(label, f"{cand['name']} — {cand['times']}x ({cand['share']}% of his opener "
                   f"games), avg {cand['avg_inn']} inn",
            _OPENER_ALERT if i == 0 else None)
    if profile.get('is_opener'):
        repeat = round(profile.get('bulk_repeat_rate', OPENER_BULK_REPEAT_RATE) * 100)
        add('Read', 'Scout the bulk arm below — the lineup sees him most of the night.')
        add('Confidence', f"who follows repeats only ~{repeat}% of the time; treat the "
                          f"bulk arm as confirmed only if the team has announced it")
    return pd.DataFrame(rows, columns=columns), {'Value': fills}


def cmp_bulk_arm_view(bulk, season, context_end, statcast_pitches_df=None):
    """Season scouting line for the arm that actually pitches the bulk of the game."""
    columns = ['Metric', 'Value']
    if not bulk or not bulk.get('id'):
        return pd.DataFrame(columns=columns)
    rows = []

    def add(metric, value):
        if value not in (None, '') and str(value).strip():
            rows.append({'Metric': metric, 'Value': value})

    hand = _pitcher_throw_code(bulk['id']) or ''
    add('Pitcher', f"{bulk['name']}" + (f" ({hand}HP)" if hand else ''))
    add('Follows opener', f"{bulk['times']}x, avg {bulk['avg_inn']} innings")
    # The DFS case for this arm, and its condition. Min-priced relief innings are the
    # cheapest real innings on the board -- but only if he is actually the one who pitches.
    add('DFS angle', f"priced as a reliever (~DK minimum) for ~{bulk['avg_inn']} innings; "
                     f"followers average 11.9 DK pts, ~3.0 per $1k vs ~2.3 for a "
                     f"conventional starter — worth it only on a confirmed bulk assignment")

    line = _pitcher_season_line(bulk['id'], season)
    if line:
        add('Season', f"{line.get('G', '')}G/{line.get('GS', '')}GS · {line.get('IP', '')} IP · "
                      f"{line.get('ERA', '')} ERA · {line.get('WHIP', '')} WHIP")
        add('K% / BB%', f"{line.get('K%', '')}% / {line.get('BB%', '')}%")

    profile = build_starter_pitch_profile(bulk['id'], season, context_end)
    mix = profile.get('season_mix', {})
    if mix:
        add('Mix', " / ".join(f"{p} {v['usage']:.0f}%" for p, v in list(mix.items())[:4]))
        add('Velo', " / ".join(f"{p} {v['velo']:.1f}" for p, v in list(mix.items())[:4]))
        archetype, _, fb_velo = _arsenal_archetype(pd.DataFrame(
            [{'Pitch': p, 'Usage %': v['usage'], 'release_speed': v['velo']} for p, v in mix.items()]
        ))
        add('Type', archetype)
    batted = profile.get('season_batted', {})
    if batted.get('BIP'):
        add('Batted ball', f"GB {batted.get('GB%', '')}% · FB {batted.get('FB%', '')}% · "
                           f"LD {batted.get('LD%', '')}%")
        add('Contact quality', f"HH {batted.get('HH%', '')}% · avg EV {batted.get('Avg EV', '')} · "
                               f"max EV {batted.get('Max EV', '')} · Brl {batted.get('Brl%', '')}%")
    return pd.DataFrame(rows, columns=columns)


def cmp_starter_l5_log(info, season, last_n=5):
    """Last N starts, with the quality of the offense faced attached to each line.

    Closes with a TOTAL/AVG row. The two used to be separate tables on separate sheets --
    one carried the totals, the other the opponent context -- which is how a 2.40 ERA over
    five starts gets read without noticing all five came against bottom-eight offenses.
    The averaged Opp Rk on the TOTAL row is shaded by the same rule as the rows above, so
    the summary line says outright how hard the stretch was.

    ERA is blank per start on purpose: a one-inning three-run outing reads as 27.00 and
    tells you nothing the GSc column does not already say better. It is the combined
    figure on the TOTAL row that is worth having.
    """
    cols = ["Date", "Opp", "IP", "H", "R", "ER", "BB", "K", "HR", "P", "ERA", "GSc",
            "Opp OPS", "Opp R/G", "Opp Rk"]
    logs = pd.DataFrame((info or {}).get("Last Starts", []))
    if logs.empty:
        return pd.DataFrame(columns=cols)
    lookup = _team_hitting_lookup(season)
    rows = []
    for _, g in logs.head(last_n).iterrows():
        opponent, ctx = _resolve_opponent(g.get("Opponent"), lookup)
        rows.append({
            "Date": str(g.get("Date", ""))[5:], "Opp": opponent,
            "IP": _float_to_ip(g.get("IP", "")), "H": g.get("H", ""), "R": g.get("R", ""),
            "ER": g.get("ER", ""), "BB": g.get("BB", ""), "K": g.get("SO", ""),
            "HR": g.get("HR", ""), "P": g.get("Pitches", ""), "ERA": "",
            "GSc": g.get("Game Score", ""),
            "Opp OPS": ctx.get("OPS", ""), "Opp R/G": ctx.get("R/G", ""),
            "Opp Rk": ctx.get("OPS Rank", ""),
        })
    table = pd.DataFrame(rows, columns=cols)

    def _sum(column):
        return int(pd.to_numeric(table[column], errors="coerce").fillna(0).sum())

    def _mean(column, digits=3):
        values = pd.to_numeric(table[column], errors="coerce").dropna()
        return round(float(values.mean()), digits) if not values.empty else ""

    total_ip = sum(_ip_to_float(v) for v in table["IP"])
    total_er = _sum("ER")
    summary = {
        "Date": "TOTAL / AVG", "Opp": f"{len(table)} starts",
        "IP": _float_to_ip(total_ip), "H": _sum("H"), "R": _sum("R"), "ER": total_er,
        "BB": _sum("BB"), "K": _sum("K"), "HR": _sum("HR"), "P": _sum("P"),
        "ERA": round(total_er * 9 / total_ip, 2) if total_ip else "",
        "GSc": _mean("GSc", 1),
        "Opp OPS": _mean("Opp OPS"), "Opp R/G": _mean("Opp R/G", 2),
        "Opp Rk": _mean("Opp Rk", 1),
    }
    return pd.concat([table, pd.DataFrame([summary])], ignore_index=True)[cols]


def cmp_starter_l5_fills(table):
    """Shade the opponent-quality columns so a soft schedule is obvious.
    Green = faced a strong offense (the line is more impressive), red = weak."""
    if table is None or table.empty or "Opp Rk" not in table.columns:
        return {}
    ranks = pd.to_numeric(table["Opp Rk"], errors="coerce")
    fills = []
    for rank in ranks:
        if pd.isna(rank):
            fills.append(None)
        elif rank <= 8:
            fills.append(_DELTA_UP_STRONG)
        elif rank <= 13:
            fills.append(_DELTA_UP_MILD)
        elif rank >= 23:
            fills.append(_DELTA_DOWN_STRONG)
        elif rank >= 18:
            fills.append(_DELTA_DOWN_MILD)
        else:
            fills.append(None)
    return {"Opp OPS": fills, "Opp R/G": list(fills), "Opp Rk": list(fills)}


_CMP_TONIGHT = (168, 198, 240)   # "this is the row that applies to tonight"


def cmp_highlight_row(df, column, value):
    """Fills that mark the one row of a season-splits table matching tonight's situation.

    Only the label cell is filled so percentile heat on the stat cells still reads."""
    if df is None or df.empty or column not in df.columns or not value:
        return {}
    target = str(value).strip().lower()
    return {column: [_CMP_TONIGHT if str(v).strip().lower() == target else None
                     for v in df[column]]}


def cmp_time_zone_view(context):
    """Compact circadian/model-input table and shared Level-cell emphasis."""
    context = context or {}
    columns = ['Route', 'Shift', 'Residual', 'Accl', 'Level', 'Study']
    if not context or context.get('Level') == 'Unavailable':
        return pd.DataFrame(columns=columns), {'Level': []}
    table = pd.DataFrame([{
        'Route': context.get('Route', ''),
        'Shift': context.get('Shift', ''),
        'Residual': context.get('Residual', ''),
        'Accl': context.get('Accl', ''),
        'Level': context.get('Level', ''),
        'Study': context.get('Study', '0.0pp'),
    }], columns=columns)
    # Two axes are encoded here, and the colour has to keep them apart. Watch and Monitor
    # both mean "displaced, but the model applies no penalty" -- they differ only in size
    # and direction -- while Elevated and Long haul are the states that actually move the
    # projection. Watch and Monitor shared one amber until now, which made every westward
    # trip and every small hop read as the same cell and hid how rarely anything scores.
    color = {
        'None': (210, 225, 210),          # green: no residual displacement
        'Watch': (250, 226, 150),         # amber: under 2h, either direction, no penalty
        'Monitor': (198, 214, 240),       # blue: 2h+ but westward, deliberately no penalty
        'Elevated': (244, 188, 181),      # light red: 2h+ eastward, penalty applies
        'Long haul': (236, 150, 140),     # red: intercontinental
    }.get(context.get('Level'))
    return table, {'Level': [color]}


def cmp_starter_row(info):
    k = _safe_number(info.get('K%'), None)
    b = _safe_number(info.get('BB%'), None)
    if k is not None and abs(k) <= 1:
        k *= 100
    if b is not None and abs(b) <= 1:
        b *= 100
    return pd.DataFrame([{
        'Name': info.get('Name', 'Unknown'), 'Throws': info.get('Throws', ''),
        'GS': info.get('GS', ''), 'IP': info.get('IP', ''), 'ERA': info.get('ERA', ''),
        'FIP': info.get('FIP', ''), 'K%': round(k, 1) if k is not None else '',
        'BB%': round(b, 1) if b is not None else '', 'WHIP': info.get('WHIP', ''),
        'HR': info.get('HR', ''), 'HR/9': info.get('HR/9', ''),
    }])


def cmp_starter_provisional(advanced_context, side):
    """This side's starter pick if it is not an announced probable, else None.

    Reports built before MLB posts a probable are the reason this exists. Naming the source
    on the page is not decoration: a projected starter and a confirmed one carry the same
    tables, and only the label tells them apart.
    """
    pick = ((advanced_context or {}).get("starter_provenance") or {}).get(side) or {}
    return pick if pick.get("provisional") and pick.get("name") else None


def _starter_pick_reason(pick):
    """`salary` / `rotation, 44% on 6 days rest` -- the source plus what it is standing on.

    The rotation model's own note is carried through rather than reduced to "projected",
    because the confidence is renormalised over whoever survives availability gating: a club
    whose start-order history has only ever gone one way returns 100% for an arm on a month's
    rest. The number is doing real work on most clubs and is nonsense on a few, and the only
    way a reader can tell which is to see the rest alongside it.
    """
    source = STARTER_SOURCE_LABEL.get(pick.get("source"), pick.get("source"))
    if pick.get("source") not in ("rotation", "bulk"):
        # DK's flag is a statement, not an estimate -- naming the file says all of it.
        return source
    # The model writes its own preamble; the label already says no probable was posted.
    detail = str(pick.get("note") or "").split(";")[-1].strip()
    if detail.startswith("rotation model gives"):
        detail = detail.replace("rotation model gives ", "").replace(f"{pick.get('name')} ", "")
    return f"{source}, {detail}" if detail else source


def cmp_starter_provisional_tag(advanced_context, side):
    """The inline marker for a panel header, or '' when the probable is announced."""
    pick = cmp_starter_provisional(advanced_context, side)
    if not pick:
        return ""
    return f"   ⚠ PROJECTED STARTER — {_starter_pick_reason(pick)}"


def cmp_starter_provisional_note(advanced_context, away_team, home_team):
    """A title suffix naming every projected starter in the game, or ''."""
    parts = []
    for side, team in (("away", away_team), ("home", home_team)):
        pick = cmp_starter_provisional(advanced_context, side)
        if pick:
            parts.append(f"{team} {pick['name']} ({_starter_pick_reason(pick)})")
    if not parts:
        return ""
    return " | ⚠ PROJECTED, no probable announced: " + "; ".join(parts)


def cmp_recent_starts(info, last_n=5):
    """Last N starts with a TOTAL/AVG row: counting sums, combined ERA, avg Game Score."""
    cols = ['Date', 'Opponent', 'IP', 'H', 'R', 'ER', 'BB', 'SO', 'HR', 'Pit/ERA', 'GSc']
    logs = pd.DataFrame(info.get('Last Starts', []))
    if logs.empty:
        return pd.DataFrame(columns=cols)
    out = logs.copy()
    out['Pit/ERA'] = pd.to_numeric(out.get('Pitches'), errors='coerce').fillna(0).astype(int)
    out['GSc'] = out.get('Game Score', '')
    # Game logs carry full club names ("@ San Francisco Giants"); abbreviate so the
    # column stays narrow enough to read without clipping.
    if 'Opponent' in out.columns:
        out['Opponent'] = out['Opponent'].map(lambda v: _resolve_opponent(v)[0])
    body = cmp_available(out, cols, last_n).reset_index(drop=True)
    ip_true = [_ip_to_float(v) for v in body['IP']] if 'IP' in body.columns else []
    total_ip = sum(ip_true)
    total_er = float(pd.to_numeric(body.get('ER'), errors='coerce').fillna(0).sum())

    def _sum(col):
        return int(pd.to_numeric(body.get(col), errors='coerce').fillna(0).sum())

    summary = {
        'Date': 'TOTAL / AVG', 'Opponent': f'{len(body)} starts', 'IP': round(total_ip, 1),
        'H': _sum('H'), 'R': _sum('R'), 'ER': _sum('ER'), 'BB': _sum('BB'), 'SO': _sum('SO'), 'HR': _sum('HR'),
        'Pit/ERA': f"ERA {total_er * 9 / total_ip:.2f}" if total_ip else "ERA -",
        'GSc': round(pd.to_numeric(body.get('GSc'), errors='coerce').mean(), 1),
    }
    return pd.concat([body, pd.DataFrame([summary])], ignore_index=True)[cols]


def _cmp_pen_status_color(bp):
    taxed = int((bp.get('Availability') == 'Taxed').sum())
    monitor = int((bp.get('Availability') == 'Monitor').sum())
    if taxed >= 2 or (taxed >= 1 and monitor >= 2):
        return _CMP_AVAIL_COLOR['Taxed']
    if taxed >= 1 or monitor >= 2:
        return _CMP_AVAIL_COLOR['Monitor']
    return _CMP_AVAIL_COLOR['Available']


_CMP_HAND_COLOR = {'L': (206, 222, 245), 'R': None}   # tint lefties so they pop out of the list


def _pen_hand(value):
    """Normalize a throwing hand to a single L/R character."""
    text = str(value or '').strip().upper()
    return text[0] if text[:1] in {'L', 'R'} else ''


def cmp_bullpen_view(frame):
    """Returns (display_df, column_fills): totals + shared L7/App emphasis.
    T = throwing hand (lefties are tinted) so platoon matchups read off the same table."""
    cols = ['Name', 'T', 'App', 'IP', 'ERA', 'FIP', 'K%', 'BB%', 'WHIP', 'L7']
    if frame is None or frame.empty:
        return pd.DataFrame(columns=cols), {'L7': [], 'App': [], 'T': []}
    bp = frame.copy()
    bp['App'] = pd.to_numeric(bp.get('G'), errors='coerce').fillna(0).astype(int)
    bp['L7'] = bp.get('L7 Pitches', bp.get('L7 Usage', ''))
    bp['T'] = bp.get('Throws', pd.Series('', index=bp.index)).map(_pen_hand)
    ip = pd.to_numeric(bp.get('IP'), errors='coerce').fillna(0.0)
    ip_total = float(ip.sum())

    def weighted(col):
        vals = pd.to_numeric(bp.get(col), errors='coerce').fillna(0.0)
        return round(float((vals * ip).sum() / ip_total), 2) if ip_total else 0.0

    hands = bp['T'].value_counts()
    summary = pd.DataFrame([{
        'Name': 'BULLPEN (IP-wtd)',
        'T': f"{int(hands.get('R', 0))}R/{int(hands.get('L', 0))}L",
        'App': int(bp['App'].sum()),
        'IP': round(ip_total, 1), 'ERA': weighted('ERA'), 'FIP': weighted('FIP'),
        'K%': weighted('K%'), 'BB%': weighted('BB%'), 'WHIP': weighted('WHIP'), 'L7': '',
    }])
    shown = bp.head(10)
    l7_fills = [_cmp_pen_status_color(bp)] + [_CMP_AVAIL_COLOR.get(a) for a in shown.get('Availability', pd.Series(dtype=str))]
    hand_fills = [(232, 232, 232)] + [_CMP_HAND_COLOR.get(h) for h in shown['T']]
    app_fills = [(232, 232, 232)]
    for games in shown['App']:
        if games >= 40:
            app_fills.append((244, 188, 181))
        elif games >= 25:
            app_fills.append((250, 226, 150))
        else:
            app_fills.append(None)
    display = pd.concat([summary, shown[cols]], ignore_index=True)[cols]
    return display, {'L7': l7_fills, 'App': app_fills, 'T': hand_fills}


# Pen workload thresholds, per game. A pen asked for 4+ innings has spent a bullpen day;
# 3 is a normal modern night. P/Out above 6.0 is a pen that is not getting quick outs,
# which costs pitches now and arms tomorrow.
_PEN_HEAVY_IP, _PEN_BUSY_IP = 4.0, 3.0
_PEN_POOR_POUT, _PEN_SOFT_POUT = 6.5, 5.75


def cmp_pen_l5_fills(group):
    """Shade the pen's workload and efficiency columns on the L5 group table.

    Red is "this cost them", not "this was a loss": a pen can throw five clean innings and
    still be unavailable tomorrow, and that is exactly what this table is for.
    """
    if group is None or group.empty:
        return {}
    ip_fills, pout_fills = [], []
    for _, row in group.iterrows():
        innings = _ip_to_float(row.get("Pen IP", 0))
        if innings >= _PEN_HEAVY_IP:
            ip_fills.append(_DELTA_DOWN_STRONG)
        elif innings >= _PEN_BUSY_IP:
            ip_fills.append(_DELTA_DOWN_MILD)
        else:
            ip_fills.append(None)
        per_out = _safe_number(row.get("P/Out"), None)
        if per_out is None:
            pout_fills.append(None)
        elif per_out >= _PEN_POOR_POUT:
            pout_fills.append(_DELTA_DOWN_STRONG)
        elif per_out >= _PEN_SOFT_POUT:
            pout_fills.append(_DELTA_DOWN_MILD)
        else:
            pout_fills.append(_DELTA_UP_MILD)
    return {"Pen IP": ip_fills, "P/Out": pout_fills}


def cmp_pen_arms_fills(arms):
    """Shade the per-arm L5 grid: availability by status, pitch total by workload."""
    if arms is None or arms.empty:
        return {}
    avail_fills = [_CMP_AVAIL_COLOR.get(str(v)) for v in arms.get("Avail", [])]
    hand_fills = [_CMP_HAND_COLOR.get(str(v)) for v in arms.get("T", [])]
    pit_fills = []
    for total in pd.to_numeric(arms.get("Pit"), errors="coerce").fillna(0):
        # Over five games: 50+ pitches is a heavily used arm, 35+ is a working one.
        if total >= 50:
            pit_fills.append(_DELTA_DOWN_STRONG)
        elif total >= 35:
            pit_fills.append(_DELTA_DOWN_MILD)
        else:
            pit_fills.append(None)
    return {"Avail": avail_fills, "T": hand_fills, "Pit": pit_fills}


def cmp_defense_view(frame):
    """Returns (display_df, fills): includes position-adjusted A+ assists and catcher
    run-game columns, with rate-stat cells colored (sample-gated). fills is
    {column: [rgb_or_None per row]}."""
    base_cols = ['Name', 'Pos', 'Inn', 'Assists', 'Fielding%', 'CS', 'SB Allowed', 'CS%', 'PB']
    display_cols = ['Name', 'Pos', 'Inn', 'A', 'A+', 'Fld%', 'Att', 'CS', 'SBA', 'CS%', 'PB', 'PB/100']
    table = cmp_available(frame, base_cols, 12)
    if table.empty or 'Pos' not in table.columns:
        return pd.DataFrame(columns=display_cols), {}
    table = table.reset_index(drop=True)
    pos = table['Pos'].astype(str).str.upper()
    caught = pd.to_numeric(table.get('CS'), errors='coerce').fillna(0)
    allowed = pd.to_numeric(table.get('SB Allowed'), errors='coerce').fillna(0)
    attempts = caught + allowed
    innings = pd.to_numeric(table.get('Inn'), errors='coerce').fillna(0)
    assists = pd.to_numeric(table.get('Assists'), errors='coerce').fillna(0)
    passed = pd.to_numeric(table.get('PB'), errors='coerce').fillna(0)
    csp = pd.to_numeric(table.get('CS%'), errors='coerce')
    baseline = pos.map(_CMP_ASSIST_BASELINES)
    assist_rate = np.where(innings > 0, assists * 100 / innings, np.nan)
    aplus = np.where(baseline.notna() & (innings >= _CMP_A_MIN_INN), (assist_rate / baseline * 100).round(0), np.nan)
    pb100 = np.where(innings > 0, (passed * 100 / innings).round(2), np.nan)
    table['A+'] = aplus
    table['Att'] = attempts.astype(int)
    table['PB/100'] = pb100
    is_c = pos.eq('C')
    non_catchers = ~is_c
    for col in ['CS', 'SB Allowed', 'CS%', 'PB', 'PB/100']:
        if col in table.columns:
            table[col] = table[col].astype(object)
            table.loc[non_catchers, col] = ''
    table.loc[is_c & (attempts < _CMP_CS_MIN_ATT), 'CS%'] = ''
    table = table.rename(columns={'Assists': 'A', 'Fielding%': 'Fld%', 'SB Allowed': 'SBA'})
    disp = table[[c for c in display_cols if c in table.columns]].reset_index(drop=True)

    def hrgb(metric, val, ok):
        return _heat_rgb(_stat_percentile(metric, val)) if ok else None

    n = len(disp)
    fills = {
        'A+': [hrgb('assists_plus', aplus[i], not pd.isna(aplus[i])) for i in range(n)],
        'CS%': [hrgb('caught_stealing_pct', csp.iloc[i], bool(is_c.iloc[i] and attempts.iloc[i] >= _CMP_CS_MIN_ATT and not pd.isna(csp.iloc[i]))) for i in range(n)],
        'PB/100': [hrgb('passed_balls_rate', pb100[i], bool(is_c.iloc[i] and innings.iloc[i] >= _CMP_PB_MIN_INN and not pd.isna(pb100[i]))) for i in range(n)],
    }
    return disp, fills


def cmp_l10_table(recent, detail, include_site=True):
    """Compact last-10 with O W% column and L10 R/RA + L10 SoS summary rows."""
    src = cmp_available(recent, ['Date', 'Opponent', 'Home/Away', 'Score', 'Result'], 10)
    src = src.rename(columns={'Opponent': 'Opp', 'Home/Away': 'Site'}).reset_index(drop=True)
    if 'Opp' in src.columns:
        src['Opp'] = src['Opp'].map(lambda v: get_team_abbreviation(v) or v)
    if 'Site' in src.columns:
        src['Site'] = src['Site'].map(lambda v: 'vs' if str(v) == 'Home' else '@')
    if not include_site and 'Site' in src.columns:
        src = src.drop(columns=['Site'])
    base = list(src.columns)
    rf, ra = [], []
    for sc in src.get('Score', pd.Series(dtype=str)).astype(str):
        m = re.match(r'\s*(\d+)\s*-\s*(\d+)', sc)
        if m:
            rf.append(int(m.group(1)))
            ra.append(int(m.group(2)))
    rpg = f"{sum(rf) / len(rf):.1f}" if rf else ""
    rapg = f"{sum(ra) / len(ra):.1f}" if ra else ""
    srow = {c: '' for c in base}
    srow['Date'] = 'L10 R/RA'
    srow['Score'] = f"{rpg}/{rapg}"
    summ = [srow]
    if detail is not None and not detail.empty and 'Opp W%' in detail.columns:
        owp = dict(zip(detail['Date'].astype(str), pd.to_numeric(detail['Opp W%'], errors='coerce')))
        src['O W%'] = src['Date'].astype(str).map(owp).round(3)
        avg = pd.to_numeric(src['O W%'], errors='coerce').mean()
        srow['O W%'] = ''
        s2 = {c: '' for c in list(src.columns)}
        s2['Date'] = 'L10 SoS'
        s2['Opp'] = 'AVG'
        s2['O W%'] = round(float(avg), 3) if not pd.isna(avg) else ''
        summ.append(s2)
    return pd.concat([src, pd.DataFrame(summ)], ignore_index=True)


def _cmp_transactions_from_absences(lines, as_of_date, days_back=7):
    """Rebuild structured transactions from older report caches.

    Older assembled caches retained notable roster-move strings but not the transaction
    table itself. Keeping this adapter here prevents a layout-only re-render from
    silently losing the section when MLB's API is unavailable.
    """
    cols = ["Date", "Player", "Pos", "Move", "Impact"]
    if not lines:
        return pd.DataFrame(columns=cols)
    try:
        end = datetime.strptime(str(as_of_date), "%Y-%m-%d")
        start = end - timedelta(days=days_back)
    except (TypeError, ValueError):
        end = start = None
    rows = []
    for line in lines:
        match = re.match(r"^\s*(\d{1,2}/\d{1,2})\s*[-–—]\s*(.*?):\s*(.+?)\s*$", str(line))
        if not match:
            continue
        date_text, player, desc = match.groups()
        if end is not None:
            try:
                month, day = (int(part) for part in date_text.split("/"))
                tx_date = datetime(end.year, month, day)
                if tx_date > end + timedelta(days=1):
                    tx_date = tx_date.replace(year=end.year - 1)
                if tx_date < start or tx_date > end:
                    continue
            except (TypeError, ValueError):
                pass
        move, impact, roster_impact = _classify_transaction(desc, "")
        if not roster_impact:
            continue
        pos_match = _TXN_POS_RE.search(desc.split(player)[0] if player in desc else desc)
        rows.append({
            "Date": date_text,
            "Player": player.strip(),
            "Pos": pos_match.group(1) if pos_match else "",
            "Move": move,
            "Impact": impact,
        })
    return pd.DataFrame(rows, columns=cols)


def cmp_transaction_view(advanced_context, side, team, as_of_date, bullpen_df, days_back=7):
    """Return cached-safe transaction rows plus roster-impact alert fills."""
    advanced_context = advanced_context or {}
    cached = advanced_context.get(f"{side}_transactions")
    table = cached.copy() if isinstance(cached, pd.DataFrame) else pd.DataFrame(cached or [])
    if table.empty:
        fallback = _cmp_transactions_from_absences(
            advanced_context.get(f"{side}_absences", []), as_of_date, days_back=days_back
        )
        live = build_transaction_table(team, as_of_date, days_back=days_back)
        table = pd.concat([live, fallback], ignore_index=True)
        if not table.empty:
            table = table.drop_duplicates(subset=["Date", "Player", "Move"], keep="first")
            table = table.tail(12).reset_index(drop=True)
    cols = ["Date", "Player", "Pos", "Move", "Impact"]
    table = cmp_available(table, cols).reset_index(drop=True)

    appearances = {}
    if bullpen_df is not None and not bullpen_df.empty and {"Name", "G"}.issubset(bullpen_df.columns):
        appearances = {
            remove_accents(str(name)).casefold(): _safe_number(games, 0)
            for name, games in zip(bullpen_df["Name"], bullpen_df["G"])
        }
    player_fills, impact_fills = [], []
    for _, row in table.iterrows():
        games = appearances.get(remove_accents(str(row.get("Player", ""))).casefold(), 0)
        if games >= 40:
            player_fills.append((244, 188, 181))
        elif games >= 25:
            player_fills.append((250, 226, 150))
        else:
            player_fills.append(None)

        impact = str(row.get("Impact", "")).upper()
        if impact.startswith(("OUT", "OFF", "GONE")) or "IL" in impact:
            impact_fills.append((244, 188, 181))
        elif impact.startswith(("IN", "BACK")) or "RETURN" in impact:
            impact_fills.append((183, 225, 183))
        elif "SHORT-TERM" in impact or "STATUS" in impact:
            impact_fills.append((250, 226, 150))
        else:
            impact_fills.append(None)
    return table, {"Player": player_fills, "Impact": impact_fills}


def generate_comparison_report_with_fpdf(
    home_lineup_df, home_bullpen_df, home_starter_info, home_hotcold_df, home_arsenal_df, home_pitcher, home_team_performance,
    away_lineup_df, away_bullpen_df, away_starter_info, away_hotcold_df, away_arsenal_df, away_pitcher, away_team_performance,
    game_date, home_team, away_team, home_team_id, away_team_id,
    home_baserunning_df, away_baserunning_df,
    home_defense_df, away_defense_df, home_last_10, away_last_10,
    home_splits_df, away_splits_df,
    output_dir="scouting_reports",
    advanced_context=None,
    primary=False,
    dh_game=None,
):
    """Landscape, topic-by-topic comparison layout (the default report layout)."""
    advanced_context = advanced_context or {}
    os.makedirs(output_dir, exist_ok=True)
    basename = (
        f"scouting_report_{game_date}_{away_team}_{home_team}{_dh_suffix(dh_game)}.pdf" if primary
        else f"scouting_report_comparison_v7_{game_date}_{away_team}_{home_team}{_dh_suffix(dh_game)}.pdf"
    )
    output_path = os.path.join(output_dir, basename)

    env = advanced_context.get("environment", {})
    hp_umpire_line = env.get("hp_umpire_line", env.get("hp_umpire", "Not yet assigned"))

    pdf = ScoutingPDF(orientation="L")
    pdf.add_page()
    pdf.set_font('DejaVu', 'B', 15)
    pdf.cell(0, 9, f"{away_team} at {home_team} | Scouting Report", ln=True, align='C')
    pdf.set_font('DejaVu', '', 8)
    pdf.cell(0, 5, f"{game_date} | Away left, home right | HP Umpire: {hp_umpire_line}", ln=True, align='C')
    pdf.ln(3)

    # The same figure set the workbook's Visuals tab carries: offense, the arms, bullpen
    # availability, and which bats to back. Best-effort -- a missing table or a matplotlib
    # failure drops a figure, never the report.
    try:
        from report_visuals import build_report_figures, figure_aspect
        _usable = pdf.w - pdf.l_margin - pdf.r_margin
        for _caption, _path in build_report_figures(
                away_team, home_team, advanced_context,
                os.path.join(output_dir, "figures"), game_date=game_date,
                dh_suffix=_dh_suffix(dh_game)):
            _px_w, _px_h = figure_aspect(_path)
            _drawn_h = _usable * _px_h / _px_w
            # Break before a figure that would run off the page rather than letting it be
            # clipped -- an image, unlike a table, has no row to break on.
            if pdf.get_y() + _drawn_h + 10 > pdf.h - pdf.b_margin:
                pdf.add_page()
            pdf.set_font('DejaVu', 'B', 10)
            pdf.cell(0, 6, _caption, ln=True)
            pdf.image(_path, x=pdf.l_margin, y=pdf.get_y(), w=_usable)
            pdf.set_y(pdf.get_y() + _drawn_h + 4)
    except Exception as _viz_error:
        print(f"⚠️ Key-metric visuals unavailable: {_viz_error}")

    # Packing constants: compact landscape row height and per-section chrome (section
    # title + panel caption). Used to estimate a block's height so we only break to a
    # new page when the block would actually overflow -- no more forced half-empty pages.
    _RH = 5.7
    _CHROME = 15.0
    _page_bottom = pdf.h - pdf.b_margin

    def ensure_space(needed):
        """Start a new page only if `needed` mm won't fit in the remaining page."""
        if pdf.get_y() + needed > _page_bottom:
            pdf.add_page()

    def block_height(rows):
        return _CHROME + (rows + 1) * _RH + 5

    def phase_banner(text):
        """Full-width phase divider so the narrative sections read as clear groups."""
        flush()  # render anything queued from the previous phase first
        # Only break if the banner plus a little content wouldn't fit at all.
        if pdf.get_y() + 10 + block_height(3) > _page_bottom:
            pdf.add_page()
        pdf.ln(1)
        pdf.set_font('DejaVu', 'B', 11)
        pdf.set_fill_color(28, 42, 74)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 8, f"  {text}", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_fill_color(255, 255, 255)
        pdf.ln(1.5)

    def available(df, cols, max_rows=None):
        if df is None or df.empty:
            return pd.DataFrame(columns=cols)
        use = [col for col in cols if col in df.columns]
        out = df[use].copy() if use else pd.DataFrame()
        return out.head(max_rows) if max_rows else out

    # Sections are queued, then laid out by flush(): narrow comparison blocks pair
    # up two-per-row, wide ones take the full width. This is what fills the page's
    # negative space instead of spreading one small table across the whole width.
    _queued = []

    def compare(title, away_df, home_df, heat=None, col_fills=None):
        _queued.append({
            'solo': False,
            'title': title,
            'away': away_df if away_df is not None else pd.DataFrame(),
            'home': home_df if home_df is not None else pd.DataFrame(),
            'heat': heat,
            'col_fills': col_fills or {},
        })

    def solo(title, df, heat=None, caption='', max_rows=20, col_fills=None):
        """Queue a single wide full-team table (no away/home split) into the flow."""
        _queued.append({
            'solo': True,
            'title': title,
            'df': df if df is not None else pd.DataFrame(),
            'heat': heat,
            'caption': caption,
            'max_rows': max_rows,
            'col_fills': col_fills or {},
        })

    # ---------------------------------------------------------------------
    # Shared geometry + helpers used across the narrative phases.
    # ---------------------------------------------------------------------
    hitter_composite = advanced_context.get('hitter_composite', pd.DataFrame())
    usable_w = pdf.w - pdf.l_margin - pdf.r_margin
    panel_gap = 9
    panel_w = (usable_w - panel_gap) / 2
    left_x = pdf.l_margin
    right_x = left_x + panel_w + panel_gap

    def panel_title(title, x, y, width):
        pdf.set_xy(x, y)
        pdf.set_font('DejaVu', 'B', 9)
        pdf.set_fill_color(200, 220, 250)
        pdf.set_text_color(15, 25, 50)
        pdf.cell(width, 7, title, ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_fill_color(255, 255, 255)
        return pdf.get_y() + 2

    # ------- two-up section packer (fills horizontal + vertical whitespace) -------
    _sec_inner_gap = 6   # gap between the away/home tables inside one section
    _sec_row_gap = 10    # gap between two sections sharing a row

    def _measure_widths(cols, frames):
        pad = 2.5 if pdf.compact_tables else 4
        pdf.set_font('DejaVu', '', 8 if pdf.compact_tables else 10)
        widths = []
        for c in cols:
            w = pdf.get_string_width(str(c)) + pad
            for f in frames:
                if c in f.columns:
                    for v in f[c]:
                        w = max(w, pdf.get_string_width(pdf._fmt_cell(v)) + pad)
            widths.append(w)
        return widths

    def _section_geometry(sec):
        """Return (natural_width, shared_col_widths_or_None) for a queued section."""
        if sec.get('solo'):
            cols = list(sec['df'].columns)
            ws = _measure_widths(cols, [sec['df']]) if cols else []
            return sum(ws), ws
        away, home = sec['away'], sec['home']
        if list(away.columns) == list(home.columns) and len(away.columns):
            ws = _measure_widths(list(away.columns), [away, home])
            return sum(ws) * 2 + _sec_inner_gap, ws
        wa = sum(_measure_widths(list(away.columns), [away])) if len(away.columns) else 0
        wh = sum(_measure_widths(list(home.columns), [home])) if len(home.columns) else 0
        return wa + wh + _sec_inner_gap, None

    def _section_height(sec):
        rows = len(sec['df']) if sec.get('solo') else max(len(sec['away']), len(sec['home']))
        return 7 + 5 + (rows + 1) * _RH + 3   # title bar + caption + table + slack

    def render_section(x, y, region_w, sec, shared):
        # Section title bar spanning the section's region.
        pdf.set_xy(x, y)
        pdf.set_font('DejaVu', 'B', 9)
        pdf.set_fill_color(200, 220, 250)
        pdf.set_text_color(15, 25, 50)
        pdf.cell(region_w, 6, sec['title'], ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_fill_color(255, 255, 255)
        table_y = y + 6 + 1
        if sec.get('solo'):
            pdf.set_xy(x, table_y)
            pdf.set_font('DejaVu', 'B', 8)
            pdf.cell(region_w, 5, sec.get('caption', ''), ln=True)
            pdf.set_x(x)
            pdf._draw_table(sec['df'], x, region_w, heat=sec['heat'],
                            column_fill=sec.get('col_fills'),
                            precomputed_widths=shared, max_rows=sec.get('max_rows', 20))
            return pdf.get_y()
        sub_w = (region_w - _sec_inner_gap) / 2
        bottom = table_y
        panels = [(away_team, sec['away'], sec['col_fills'].get('away')),
                  (home_team, sec['home'], sec['col_fills'].get('home'))]
        for i, (team, frame, col_fill) in enumerate(panels):
            sx = x + i * (sub_w + _sec_inner_gap)
            pdf.set_xy(sx, table_y)
            pdf.set_font('DejaVu', 'B', 8)
            pdf.cell(sub_w, 5, team, ln=True)
            pdf.set_x(sx)
            pdf._draw_table(frame, sx, sub_w, heat=sec['heat'], column_fill=col_fill, precomputed_widths=shared)
            bottom = max(bottom, pdf.get_y())
        return bottom

    def flush():
        """Lay out and render every queued section, then clear the queue.

        Sections are drawn at their natural width and flowed left-to-right, so as
        many as fit share a row. That packs the wide landscape page in both
        dimensions instead of stretching a single small table across it.
        """
        pending = list(_queued)
        _queued.clear()
        for sec in pending:
            pw, shared = _section_geometry(sec)
            sec['_pw'] = min(pw, usable_w)   # cap; _draw_table scales an over-wide pair down
            sec['_shared'] = shared
            sec['_h'] = _section_height(sec)

        just_paged = False
        while pending:
            rem = _page_bottom - pdf.get_y()
            # Build a row from whatever fits the remaining vertical space and page width.
            # Scanning in order but skipping sections too tall for `rem` lets a short
            # later block fill the page bottom instead of leaving it white.
            row, row_w = [], 0.0
            for s in list(pending):
                add_w = s['_pw'] if not row else _sec_row_gap + s['_pw']
                if row and row_w + add_w > usable_w:
                    continue
                if s['_h'] > rem and not (just_paged and not row):
                    continue
                row.append(s)
                row_w += add_w
                pending.remove(s)
                if s['_pw'] >= usable_w - 1:   # a full-width section closes the row
                    break
            if not row:
                pdf.add_page()
                just_paged = True
                continue
            y = pdf.get_y()
            pdf.set_auto_page_break(False)
            x = left_x
            bottom = y
            for s in row:
                bottom = max(bottom, render_section(x, y, s['_pw'], s, s['_shared']))
                x += s['_pw'] + _sec_row_gap
            pdf.set_auto_page_break(True, margin=12)
            pdf.set_y(bottom + 3)
            just_paged = False

    # Table transforms are shared with the Excel export (see cmp_* helpers) so a column
    # can't get dropped from one output. These thin aliases keep the call sites readable.
    starter_row = cmp_starter_row
    recent_starts_table = cmp_recent_starts
    bullpen_view = cmp_bullpen_view
    defense_view = cmp_defense_view
    run_cols = ['Name', 'SB', 'CS', 'SB_Att', 'SB%', 'Runs', 'Triples']

    away_team_performance = cmp_enrich_pythag(away_team_performance)
    home_team_performance = cmp_enrich_pythag(home_team_performance)
    split_hand_away = get_pitcher_handedness(home_pitcher) or 'R'
    split_hand_home = get_pitcher_handedness(away_pitcher) or 'R'

    # =====================================================================
    # PHASE 1 — GAME CONTEXT
    # =====================================================================
    phase_banner('1  ·  GAME CONTEXT   |   Records, form, rest & schedule')

    # Dashboard: season/team form (left) and both L10 snapshots (right).
    perf = cmp_team_form_snapshot(
        away_team_performance, home_team_performance, advanced_context, away_team, home_team
    )
    panel_y = pdf.get_y()
    left_table_y = panel_title('Team Form Snapshot', left_x, panel_y, panel_w)
    pdf.set_xy(left_x, left_table_y)
    pdf._draw_table(perf, left_x, panel_w, max_rows=20)
    left_bottom = pdf.get_y()

    right_table_y = panel_title('L10 Snapshots', right_x, panel_y, panel_w)
    l10_gap = 4
    l10_w = (panel_w - l10_gap) / 2
    right_bottom = right_table_y
    recent_specs = [
        (away_team, away_last_10, advanced_context.get('away_recent_detail', pd.DataFrame())),
        (home_team, home_last_10, advanced_context.get('home_recent_detail', pd.DataFrame())),
    ]
    for idx, (team, recent, recent_detail) in enumerate(recent_specs):
        l10_x = right_x + idx * (l10_w + l10_gap)
        pdf.set_xy(l10_x, right_table_y)
        pdf.set_font('DejaVu', 'B', 8)
        pdf.cell(l10_w, 5, team, ln=True)
        compact_l10 = cmp_l10_table(recent, recent_detail, include_site=False)
        pdf.set_x(l10_x)
        pdf._draw_table(compact_l10, l10_x, l10_w, max_rows=13, heat='last10')
        right_bottom = max(right_bottom, pdf.get_y())
    pdf.set_y(max(left_bottom, right_bottom) + 2)

    # Weather forecast + park factor. The park-profile sentence rides in the caption so
    # the table itself stays narrow enough to share a row with the rest/schedule blocks.
    env_view, env_fills = cmp_game_environment(advanced_context.get('environment', {}))
    if not env_view.empty:
        profile_mask = env_view['Metric'].eq('Park Profile')
        env_caption = ' '.join(env_view.loc[profile_mask, 'Value'].astype(str))
        keep = ~profile_mask
        env_fills = {'Value': [f for f, k in zip(env_fills.get('Value', []), keep) if k]}
        solo('Game Environment | weather + park factor (≈ = park-relative wind is estimated)',
             env_view[keep].reset_index(drop=True), caption=env_caption, col_fills=env_fills)

    # Rest & schedule spot, starter rest splits, and schedule-context performance --
    # all narrow, so they flow together and pack the bottom of the context page.
    rest_cols = ['Rest', 'SP Rest', 'Note', 'Pen A/T']
    compare(
        'Rest & Schedule Spot',
        available(advanced_context.get('away_rest_schedule', pd.DataFrame()), rest_cols),
        available(advanced_context.get('home_rest_schedule', pd.DataFrame()), rest_cols),
    )
    away_tz_view, away_tz_fills = cmp_time_zone_view(
        advanced_context.get('away_time_zone', {})
    )
    home_tz_view, home_tz_fills = cmp_time_zone_view(
        advanced_context.get('home_time_zone', {})
    )
    compare(
        'Time-Zone Load | trusted PNAS eastbound baseline',
        away_tz_view,
        home_tz_view,
        col_fills={'away': away_tz_fills, 'home': home_tz_fills},
    )
    compare(
        'Trip / Homestand Form | 3Y descriptive; ΔW pp vs team site baseline',
        available(
            advanced_context.get('away_trip_form', pd.DataFrame()),
            ['Trip Spot', 'G', 'W%', 'ΔW pp', 'RD/G'],
        ),
        available(
            advanced_context.get('home_trip_form', pd.DataFrame()),
            ['Trip Spot', 'G', 'W%', 'ΔW pp', 'RD/G'],
        ),
        heat={'W%': 'win_pct', 'RD/G': 'run_diff_pg'},
    )
    compare(
        'Starter Perf by Days Rest',
        available(advanced_context.get('away_sp_rest_splits', pd.DataFrame()), ['Rest', 'GS', 'IP', 'ERA', 'W-L']),
        available(advanced_context.get('home_sp_rest_splits', pd.DataFrame()), ['Rest', 'GS', 'IP', 'ERA', 'W-L']),
        heat={'ERA': 'era'},
    )
    compare(
        'Team Perf by Schedule Spot | R/G off, RA/G pitch',
        available(advanced_context.get('away_schedule_context', pd.DataFrame()), ['Context', 'G', 'R/G', 'RA/G', 'W-L']),
        available(advanced_context.get('home_schedule_context', pd.DataFrame()), ['Context', 'G', 'R/G', 'RA/G', 'W-L']),
        heat={'R/G': 'rpg_off', 'RA/G': 'rpg_allowed'},
    )

    # =====================================================================
    # PHASE 2 — OFFENSE (both lineups, form, splits, and what they face)
    # =====================================================================
    phase_banner('2  ·  OFFENSE   |   Lineups, form, splits & matchup vs today\'s starter')

    away_lineup = _lineup_with_offense(away_lineup_df, hitter_composite)
    home_lineup = _lineup_with_offense(home_lineup_df, hitter_composite)
    lineup_cols = ['Spot', 'Name', 'Pos', 'Bats', 'OPS', 'Off Szn', 'Off L28', 'ISO', 'HR', 'SB']
    compare('Projected Lineups', available(away_lineup, lineup_cols, 9), available(home_lineup, lineup_cols, 9), heat='offense')

    hot_cols = ['Name', 'PA', 'OPS', 'xwOBA', 'Szn xwOBA', 'ΔxwOBA', 'HardHit%',
            'Chase%', 'Z-Con%', 'Status']
    compare(
        "Hot / Cold Hitters | Last 14 Days vs each hitter's own season "
        "(HOT/COLD = +/-40 pts of xwOBA against his own norm)",
        available(_hot_cold_sorted(away_hotcold_df), hot_cols, 9),
        available(_hot_cold_sorted(home_hotcold_df), hot_cols, 9),
        heat='hot_cold',
    )

    hitting_cols = ['Split', 'PA', 'OPS', 'xwOBA', 'ISO', 'K%', 'BB%', 'HardHit%', 'HR%']
    compare(
        'Team Hitting Summary | Home/Away and opposing starter hand',
        available(advanced_context.get('away_lineup_splits', pd.DataFrame()), hitting_cols, 8),
        available(advanced_context.get('home_lineup_splits', pd.DataFrame()), hitting_cols, 8),
        heat='offense',
    )

    # Wide per-hitter season splits vs the hand they'll face; each is one full-team table.
    for team, frame, hand in [(away_team, away_splits_df, split_hand_away), (home_team, home_splits_df, split_hand_home)]:
        split_cols = ['Name', f'PA vs {hand}', f'AVG vs {hand}', f'OBP vs {hand}', f'SLG vs {hand}', f'OPS vs {hand}', f'ISO vs {hand}', f'HR vs {hand}', f'BB% vs {hand}', f'K% vs {hand}']
        table = available(frame, split_cols, 9)
        if f'PA vs {hand}' in table.columns:
            table = table.sort_values(f'PA vs {hand}', ascending=False)
        split_heat = {
            f'OPS vs {hand}': 'ops_off', f'OBP vs {hand}': 'obp_off',
            f'SLG vs {hand}': 'slg_off', f'AVG vs {hand}': 'avg_off',
            f'ISO vs {hand}': 'iso', f'BB% vs {hand}': 'bb_pct_bat',
            f'K% vs {hand}': 'k_pct_bat',
        }
        solo(f'{team} Season Splits vs {hand} Pitching', table, heat=split_heat, max_rows=9)

    bvp_cols = ['Name', 'Years', 'PA', 'AB', 'H', 'HR', 'BB', 'K', 'AVG', 'SLG', 'OPS']
    compare(
        'Prior Matchup History vs Today\'s Starter',
        available(advanced_context.get('away_bvp', pd.DataFrame()), bvp_cols, 5),
        available(advanced_context.get('home_bvp', pd.DataFrame()), bvp_cols, 5),
        heat='offense',
    )

    pitching_cols = ['Unit', 'Split', 'PA', 'OPS', 'xwOBA', 'ISO', 'K%', 'BB%', 'HardHit%', 'HR%']
    compare(
        'Opposing Pitching Allowed Summary | Starter and bullpen (colored from hitters\' view: high allowed = green)',
        available(advanced_context.get('away_opp_pitching', pd.DataFrame()), pitching_cols, 14),
        available(advanced_context.get('home_opp_pitching', pd.DataFrame()), pitching_cols, 14),
        heat='pitching_allowed_batter_view',
    )

    matchup_cols = ['Pitch', 'Usage%', 'Pitches', 'RV/100', 'Whiff%', 'Chase%', 'Contact%', 'xwOBA', 'Edge']
    compare(
        'Pitch-Type Value & Swing Decisions | Lineup vs today\'s starter (RV/100 high = hitter)',
        available(advanced_context.get('away_arsenal_matchup', pd.DataFrame()), matchup_cols, 6),
        available(advanced_context.get('home_arsenal_matchup', pd.DataFrame()), matchup_cols, 6),
        heat='pitch_matchup',
    )
    drill_cols = ['Name', 'Pitches', 'RV/100', 'Whiff%', 'Chase%', 'Contact%', 'xwOBA', 'Edge']
    compare(
        'Per-Batter vs Starter Arsenal | Usage-weighted',
        available(advanced_context.get('away_arsenal_drilldown', pd.DataFrame()), drill_cols, 9),
        available(advanced_context.get('home_arsenal_drilldown', pd.DataFrame()), drill_cols, 9),
        heat='pitch_matchup',
    )

    # =====================================================================
    # PHASE 3 — STARTING PITCHING
    # =====================================================================
    phase_banner('3  ·  STARTING PITCHING   |   Probables, recent form & arsenals')

    compare('Probable Starter Comparison'
            + cmp_starter_provisional_note(advanced_context, away_team, home_team),
            starter_row(away_starter_info), starter_row(home_starter_info), heat='pitching')
    compare(
        'Last Five Starts',
        recent_starts_table(away_starter_info),
        recent_starts_table(home_starter_info),
        heat='pitching',
    )
    arsenal_cols = ['Pitch', 'Usage %', 'release_speed', 'Horiz. Break', 'Vert. Break', 'CSW%', 'Stuff+']
    away_arsenal = available(away_arsenal_df.sort_values('Usage %', ascending=False), arsenal_cols, 7)
    home_arsenal = available(home_arsenal_df.sort_values('Usage %', ascending=False), arsenal_cols, 7)
    compare('Condensed Starter Pitch Arsenal', away_arsenal, home_arsenal, heat='arsenal')

    # =====================================================================
    # PHASE 4 — BULLPEN
    # =====================================================================
    phase_banner('4  ·  BULLPEN   |   Quality, workload & availability')

    away_pen_form = advanced_context.get('away_bullpen_form', pd.DataFrame())
    home_pen_form = advanced_context.get('home_bullpen_form', pd.DataFrame())
    compare(
        'Bullpen Summary and Availability | R/L = full pen mix, Avail R/L = mix that is actually usable tonight',
        away_pen_form,
        home_pen_form,
        heat='pitching',
    )
    away_pen_view, away_pen_fills = bullpen_view(away_bullpen_df)
    home_pen_view, home_pen_fills = bullpen_view(home_bullpen_df)
    compare(
        'Bullpen Scouting | T = throwing hand (LHP tinted); L7 colored by availability (green avail / amber monitor / red taxed)',
        away_pen_view, home_pen_view, heat='pitching',
        col_fills={'away': away_pen_fills, 'home': home_pen_fills},
    )

    # =====================================================================
    # PHASE 5 — RUN GAME & DEFENSE
    # =====================================================================
    phase_banner('5  ·  RUN GAME & DEFENSE   |   Baserunning and fielding')

    away_def_disp, away_def_fill = defense_view(away_defense_df)
    home_def_disp, home_def_fill = defense_view(home_defense_df)
    compare(
        'Baserunning',
        available(away_baserunning_df, run_cols, 8),
        available(home_baserunning_df, run_cols, 8),
        heat='baserunning',
    )
    team_def_cols = ['Team', 'BIP', 'xBA Allowed', 'BA Allowed', 'Hits Saved/G',
                     'IF Saved/G', 'OF Saved/G', 'Frame +Str/G', 'Grade']
    compare(
        'Team Defense | expected minus actual hits on balls in play, vs league '
        '(errors excluded: DK charges earned runs only)',
        available(advanced_context.get('away_team_defense', pd.DataFrame()), team_def_cols, 1),
        available(advanced_context.get('home_team_defense', pd.DataFrame()), team_def_cols, 1),
        heat='team_defense',
    )
    compare(
        'Defense | fielding-stat view: chances and errors, not range',
        away_def_disp, home_def_disp,
        col_fills={'away': away_def_fill, 'home': home_def_fill},
    )

    # =====================================================================
    # PHASE 6 — ROSTER ALERTS
    # =====================================================================
    phase_banner('6  ·  ROSTER ALERTS   |   Recent transactions (last 7 days)')
    away_txn, away_txn_fills = cmp_transaction_view(
        advanced_context, 'away', away_team, game_date, away_bullpen_df
    )
    home_txn, home_txn_fills = cmp_transaction_view(
        advanced_context, 'home', home_team, game_date, home_bullpen_df
    )
    compare(
        'Recent Transactions | Move = what happened, Impact = roster effect',
        away_txn,
        home_txn,
        col_fills={'away': away_txn_fills, 'home': home_txn_fills},
    )
    flush()

    pdf.output(output_path)
    return output_path


def generate_comparison_report_excel(
    home_lineup_df, home_bullpen_df, home_starter_info, home_hotcold_df, home_arsenal_df, home_pitcher, home_team_performance,
    away_lineup_df, away_bullpen_df, away_starter_info, away_hotcold_df, away_arsenal_df, away_pitcher, away_team_performance,
    game_date, home_team, away_team, home_team_id, away_team_id,
    home_baserunning_df, away_baserunning_df,
    home_defense_df, away_defense_df, home_last_10, away_last_10,
    home_splits_df, away_splits_df,
    output_dir="scouting_reports",
    advanced_context=None,
    dh_game=None,
):
    """Single scrollable worksheet: every comparison table stacked top-to-bottom,
    away block on the left and home block on the right, with the same league-percentile
    heat coloring as the PDF applied via conditional cell fills. No page breaks, so
    there is no page-driven whitespace."""
    from openpyxl import Workbook
    from openpyxl.cell.rich_text import CellRichText, TextBlock
    from openpyxl.cell.text import InlineFont
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    advanced_context = advanced_context or {}
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"scouting_report_{game_date}_{away_team}_{home_team}{_dh_suffix(dh_game)}.xlsx")

    wb = Workbook()

    def _prepare_sheet(sheet):
        sheet.sheet_view.showGridLines = False
        sheet.sheet_format.defaultRowHeight = 12.75
        # Open compactly and print one page wide while remaining a normal scrollable sheet.
        # Every sheet opens at 85%: the column bounds above are tuned so a full panel pair
        # fits on a standard screen at that zoom without clipping any cell.
        sheet.sheet_view.zoomScale = 85
        sheet.sheet_view.zoomScaleNormal = 85
        sheet.page_setup.orientation = "landscape"
        sheet.page_setup.fitToWidth = 1
        sheet.page_setup.fitToHeight = 0
        sheet.sheet_properties.pageSetUpPr.fitToPage = True
        return sheet

    # One sheet per section group, each with its OWN column widths.
    #
    # This is the fix for the report's worst readability problem, and the cause is structural
    # rather than cosmetic: an Excel column width is a property of the sheet, not of a table.
    # Every section used to be laid into the same twelve-column grid, so column A's width was
    # the widest first cell of *any* table on the sheet. In practice a handful of long prose
    # cells set the grid for all ~250 rows -- a 116-character team-defense note forced column
    # A to 21.1 and a 101-character matchup note forced column B to 17.1, stretching the
    # lineup and pitching tables that had no long text in them at all. Observed table widths
    # ran from three to eleven columns sharing one grid, which no single set of widths can
    # serve.
    #
    # Splitting by section means only similarly-shaped tables share a grid, and the prose-heavy
    # sections (game context, roster alerts) can be as wide as they need without dragging the
    # stat tables with them.
    # Bands 3 (starting pitching) and 4 (bullpen) are absent on purpose: they are written
    # to their own "Pitching" sheet by `_write_pitching_sheet`, which is slotted into
    # position below. Everything else routes by its leading number.
    _SHEET_FOR_BAND = {
        "1": "Game Context",
        "2": "Offense",
        "5": "Run & Defense",
        "6": "Roster Alerts",
    }
    # Where the pitching sheet belongs once it exists, so the tab order still reads in
    # phase order rather than in the order the sheets happened to be created.
    _PITCHING_SHEET_AFTER = "Offense"
    sheets = {}
    ws = None
    col_widths = {}
    # Set once the title/legend logic below is in scope, so every new sheet gets the same
    # header. Late-bound because the header needs `env` and the DFS legend, which are
    # resolved further down than this.
    _header_writer = {"fn": None}

    def use_sheet(name):
        """Point the writers at `name`, creating it on first use. Row and width state
        follow the sheet, which is the whole point of the split."""
        nonlocal ws, col_widths
        if ws is not None:
            sheets[ws.title]["row"] = state["row"]
        fresh = name not in sheets
        if fresh:
            sheet = _prepare_sheet(wb.active if not sheets else wb.create_sheet(name))
            sheet.title = name
            sheets[name] = {"ws": sheet, "widths": {}, "row": 1}
        entry = sheets[name]
        ws = entry["ws"]
        col_widths = entry["widths"]
        state["row"] = entry["row"]
        if fresh and _header_writer["fn"]:
            # Repeated on every tab on purpose: a sheet that does not say which game it
            # belongs to is a trap once there are six of them.
            _header_writer["fn"]()

    thin = Side(style="thin", color="C8C8C8")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    hdr_fill = PatternFill("solid", fgColor="D2E1F5")
    band_fill = PatternFill("solid", fgColor="1C2A4A")
    subhdr_fill = PatternFill("solid", fgColor="C8DCF5")
    center = Alignment(horizontal="center", vertical="center")
    left = Alignment(horizontal="left", vertical="center")
    center_wrap = Alignment(horizontal="center", vertical="center", wrap_text=True)
    left_wrap = Alignment(horizontal="left", vertical="center", wrap_text=True)

    def av(df, cols, n=None):
        if df is None or df.empty:
            return pd.DataFrame(columns=cols)
        use = [c for c in cols if c in df.columns]
        out = df[use].copy() if use else pd.DataFrame(columns=cols)
        return out.head(n) if n else out

    def fmt(v):
        if isinstance(v, float):
            if pd.isna(v):
                return ""
            if v == int(v):
                return int(v)
            return round(v, 3)
        return v

    # Column-width accumulator (in Excel width units ~ chars).
    # (per-sheet; bound by use_sheet above -- do not rebind here or the widths
    #  stop following the sheet)

    _NARROW_FIELDS = {
        'Spot', 'Pos', 'Bats', 'Throws', 'T', 'GS', 'G', 'IP', 'H', 'R', 'ER', 'BB',
        'SO', 'HR', 'PA', 'AB', 'SB', 'CS', 'SBA', 'PB', 'A', 'A+', 'Inn',
        'Att', 'Triples', 'Result', 'Site', 'Role', 'Brl', 'EV', 'xBA', 'Dec',
    }
    _MEDIUM_FIELDS = {
        'ERA', 'FIP', 'K%', 'BB%', 'WHIP', 'OPS', 'xwOBA', 'HardHit%', 'CS%',
        'PB/100', 'Fld%', 'Pit/ERA', 'GSc', 'Score', 'O W%', 'Usage %', 'Usage%',
        'Pitches', 'RV/100', 'Whiff%', 'Chase%', 'Contact%', 'Stuff+', 'CSW%',
    }

    def field_width_bounds(field):
        field = str(field or "")
        if field in _NARROW_FIELDS:
            return 4.5, 6.5
        if field in _MEDIUM_FIELDS:
            return 5.5, 9.0
        if field == 'L7':         # 7-day usage strip: clipping it loses actual data
            return 10.0, 22.0
        if field == 'Date':
            return 7.0, 9.0
        if field in {'Name', 'Player'}:
            # Longest real names ("Vladimir Guerrero Jr.") plus the trailing ownership
            # glyph and its space. At the old cap of 22 the name exactly filled the column
            # and the mark -- the one character that is there on every priced player --
            # was the part that got clipped.
            return 9.0, 24.5
        if field in {'Opponent', 'Opp', 'Metric', 'Context', 'Split', 'Pitch', 'Unit', 'Trip Spot'}:
            return 7.0, 12.0
        if field in {'Status', 'Consistency', 'Edge', 'Result'}:
            return 5.5, 10.5      # "Advantage" / "More volatile" should not lose a letter
        if field == 'Value':      # wraps, so it buys height instead of width
            return 11.0, 14.0
        if field in {'Move', 'Impact', 'Note'}:
            return 9.0, 10.0      # also wrapped
        return 5.5, 9.0

    def note_width(col_idx, text, field=None, is_header=False):
        """Grow a column to fit its content, within that field's bounds.

        Headers get to set their own floor -- a clipped header makes the whole column
        unreadable, whereas a clipped data cell is usually just a long name. The starting
        floor is deliberately small so short stat columns stay narrow and the sheet still
        fits on screen at 85% zoom.
        """
        lo, hi = field_width_bounds(field)
        needed = len(str(text)) + 1.1
        if is_header:
            # Headers wrap (center_wrap), so a long one buys height instead of width --
            # widening every column to fit "Last3D Pitches" would blow the 85% budget.
            lo = hi = max(lo, min(needed, 9.5))
        col_widths[col_idx] = max(col_widths.get(col_idx, 4.5), min(max(needed, lo), hi))

    def heat_fill(heat, column, value):
        metric = _heat_metric_for(heat, column) if heat else None
        if not metric:
            return None
        rgb = _heat_rgb(_stat_percentile(metric, value))
        if rgb is None:
            return None
        return PatternFill("solid", fgColor="{:02X}{:02X}{:02X}".format(*rgb))

    def _fill_from_rgb(rgb):
        return PatternFill("solid", fgColor="{:02X}{:02X}{:02X}".format(*rgb))

    def write_table(df, top_row, left_col, heat=None, col_fills=None):
        """Write one table; return (rows_used, cols_used).
        col_fills: optional {column_name: [rgb_or_None per data row]} forced fills
        (e.g. bullpen L7 by availability, defense A+/CS%/PB) that override heat."""
        col_fills = col_fills or {}
        if df is None or df.empty:
            ws.cell(row=top_row, column=left_col, value="No data.").font = Font(italic=True, size=9)
            return 1, 1
        cols = list(df.columns)
        is_metric_table = bool(cols and cols[0] == 'Metric')
        for j, c in enumerate(cols):
            cell = ws.cell(row=top_row, column=left_col + j, value=str(c))
            cell.font = Font(bold=True, size=8.5, color="14243C")
            cell.fill = hdr_fill
            cell.alignment = center_wrap
            cell.border = border
            note_width(left_col + j, c, c, is_header=True)
            # Give the header row the height its longest wrapped label needs.
            header_lines = max(1, math.ceil((len(str(c)) + 1.1) / max(col_widths[left_col + j], 4.5)))
            if header_lines > 1:
                ws.row_dimensions[top_row].height = max(
                    ws.row_dimensions[top_row].height or 0, 11.5 * header_lines
                )
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            for j, c in enumerate(cols):
                val = fmt(row[c])
                # Projected ownership rides along as one trailing character on player names.
                # Every priced player gets it, which is why it is not a colour: two tint
                # families already cover a third of a slate and a third would be unreadable.
                own_mark = _dfs_own_mark(c, row[c])
                cell = ws.cell(row=top_row + i, column=left_col + j,
                               value=(f"{val}{own_mark}" if own_mark else val))
                cell.font = Font(size=9)
                name_bold = False
                if c in {'Move', 'Impact', 'Note'}:
                    cell.alignment = left_wrap
                    # These columns are narrow and wrapped, so a long transaction note
                    # needs its own line count -- a fixed 2-line height hid the tail.
                    lines = max(2, math.ceil(len(str(val)) / 9))
                    ws.row_dimensions[top_row + i].height = max(
                        ws.row_dimensions[top_row + i].height or 0, 12.5 * lines
                    )
                elif is_metric_table and c != 'Metric':
                    cell.alignment = center_wrap
                    # Value cells wrap instead of forcing a very wide column, so the row
                    # has to be tall enough for however many lines the text actually needs.
                    lines = max(1, math.ceil(len(str(val)) / 10))
                    if lines > 1:
                        ws.row_dimensions[top_row + i].height = max(
                            ws.row_dimensions[top_row + i].height or 0, 12.75 * lines
                        )
                else:
                    cell.alignment = (
                        left if c in {'Name', 'Player', 'Metric', 'Context'} else center
                    )
                cell.border = border
                f = None
                forced = col_fills.get(c)
                if forced is not None and (i - 1) < len(forced) and forced[i - 1] is not None:
                    f = _fill_from_rgb(forced[i - 1])
                if f is None:
                    f = heat_fill(heat, c, row[c])
                if f is None:
                    # DFS value tint on player-name cells. Lowest precedence, so it never
                    # displaces a percentile heat color, and a no-op when no priced slate
                    # has been loaded.
                    tint = _dfs_highlight(c, row[c])
                    if tint is not None:
                        f = _fill_from_rgb(tint[1])
                        cell.font = Font(size=9, bold=True)
                        name_bold = True
                if f is not None:
                    cell.fill = f
                # The ownership glyph is set one size larger than the name it trails, which
                # needs rich text: a cell-level font would drag the name up with it and the
                # names are already at the size the column widths were built for. Applied
                # after the tint branch so a highlighted name keeps its bold.
                if own_mark:
                    cell.value = CellRichText(
                        TextBlock(InlineFont(sz=9, b=name_bold), str(val)),
                        TextBlock(InlineFont(sz=_OWN_MARK_SIZE), str(own_mark)),
                    )
                # Width is measured against the DISPLAYED text. Measuring `val` alone left
                # no room for the trailing glyph, which is what clipped it on the longest
                # name in a column.
                note_width(left_col + j, f"{val}{own_mark}",
                           'Value' if (is_metric_table and c != 'Metric') else c)
        return len(df) + 1, len(cols)

    _PANEL_COLS = 12
    _PANEL_GAP = 2
    _HOME_LEFT = 1 + _PANEL_COLS + _PANEL_GAP
    _BAND_W = _PANEL_COLS * 2 + _PANEL_GAP

    state = {"row": 1}

    def band(text):
        # The leading number routes the section to its sheet. Sections that share a sheet
        # (bullpen joins starting pitching) still get their band as an in-sheet divider.
        target = _SHEET_FOR_BAND.get(str(text).strip()[:1])
        if target and (ws is None or ws.title != target):
            use_sheet(target)
        r = state["row"]
        cell = ws.cell(row=r, column=1, value=text)
        cell.font = Font(bold=True, size=12, color="FFFFFF")
        cell.fill = band_fill
        cell.alignment = left
        for c in range(2, _BAND_W + 1):
            ws.cell(row=r, column=c).fill = band_fill
        ws.row_dimensions[r].height = 18
        state["row"] = r + 1

    def _title_stripe(title, r):
        tcell = ws.cell(row=r, column=1, value=title)
        tcell.font = Font(bold=True, size=10, color="14243C")
        tcell.fill = subhdr_fill
        tcell.alignment = left
        for c in range(2, _BAND_W + 1):
            ws.cell(row=r, column=c).fill = subhdr_fill

    def pair(title, left_label, left_df, right_label, right_df,
             heat=None, left_heat=None, right_heat=None, col_fills=None):
        """Two independently-labeled tables side by side under one title stripe.
        Used both for away/home comparisons and for unrelated left/right panels."""
        col_fills = col_fills or {}
        r = state["row"]
        _title_stripe(title, r)
        r += 1
        ws.cell(row=r, column=1, value=left_label).font = Font(bold=True, size=9)
        for spacer_col in range(1 + _PANEL_COLS, _HOME_LEFT):
            ws.column_dimensions[get_column_letter(spacer_col)].width = 1.5
        ws.cell(row=r, column=_HOME_LEFT, value=right_label).font = Font(bold=True, size=9)
        r += 1
        rows_l, _ = write_table(left_df, r, 1, heat=left_heat or heat,
                                col_fills=col_fills.get('left'))
        rows_r, _ = write_table(right_df, r, _HOME_LEFT, heat=right_heat or heat,
                                col_fills=col_fills.get('right'))
        state["row"] = r + max(rows_l, rows_r) + 1

    def section(title, away_df, home_df, heat=None, gap=2, col_fills=None):
        col_fills = col_fills or {}
        pair(title, away_team, away_df, home_team, home_df, heat=heat,
             col_fills={'left': col_fills.get('away'), 'right': col_fills.get('home')})

    def solo(title, df, heat=None, col_fills=None):
        r = state["row"]
        _title_stripe(title, r)
        r += 1
        rows, _ = write_table(df, r, 1, heat=heat, col_fills=col_fills)
        state["row"] = r + rows + 1

    # ---- title ----
    env = advanced_context.get("environment", {})
    hp = env.get("hp_umpire_line", env.get("hp_umpire", "Not yet assigned"))

    def write_sheet_header():
        """Title row and legends, on whichever sheet is current."""
        t = ws.cell(row=1, column=1,
                    value=f"{away_team} at {home_team}  |  {game_date}  |  HP Umpire: {hp}")
        t.font = Font(bold=True, size=14)
        state["row"] = 3

        # The stat scale is always visible: its hard endpoints are part of the meaning of
        # the report, not just decoration. DFS tint follows it when that layer is on.
        col = 1
        label = ws.cell(row=2, column=col, value="Stat heat:")
        label.font = Font(bold=True, size=8.5)
        col += 1
        for heat_label, score in (("Very bad", 0.0), ("Average", 0.5), ("Very good", 1.0)):
            rgb = _heat_rgb(score)
            swatch = ws.cell(row=2, column=col, value=heat_label)
            swatch.font = Font(bold=True, size=8.5)
            swatch.fill = PatternFill("solid", fgColor="{:02X}{:02X}{:02X}".format(*rgb))
            swatch.alignment = center
            swatch.border = border
            col += 1

        # Legend for the DFS value tint, written only when highlighting is actually on --
        # an unexplained colored name is worse than an uncolored one.
        legend_tiers = _dfs_highlight_legend()
        if legend_tiers:
            col += 1
            label = ws.cell(row=2, column=col, value="DFS value:")
            label.font = Font(bold=True, size=8.5)
            col += 1
            for tier, rgb in legend_tiers:
                swatch = ws.cell(row=2, column=col, value=tier)
                swatch.font = Font(bold=True, size=8.5)
                swatch.fill = PatternFill("solid", fgColor="{:02X}{:02X}{:02X}".format(*rgb))
                swatch.alignment = center
                swatch.border = border
                col += 1
            note = ws.cell(row=2, column=col,
                           value="  teal = raw projection, indigo = points per $1k, "
                                 "violet = both  (ranked within pitchers / hitters)")
            note.font = Font(italic=True, size=8)

    _header_writer["fn"] = write_sheet_header

    # The visuals get their own tab and lead the workbook: they are the at-a-glance read
    # of platoon splits, the arms, bullpen availability and which bats to back, and they
    # were crowding the top of Game Context. Rendered once here and reused by the PDF.
    # Best-effort throughout -- a missing table or a matplotlib failure drops a figure,
    # never the report.
    figure_paths = []
    try:
        from report_visuals import build_report_figures
        figure_paths = build_report_figures(
            away_team, home_team, advanced_context,
            os.path.join(output_dir, "figures"), game_date=game_date,
            dh_suffix=_dh_suffix(dh_game))
    except Exception as _viz_error:
        print(f"⚠️ Key-metric visuals unavailable: {_viz_error}")

    if figure_paths:
        try:
            from openpyxl.drawing.image import Image as XLImage
            use_sheet("Visuals")
            for _caption, _path in figure_paths:
                _title = ws.cell(row=state["row"], column=1, value=_caption)
                _title.font = Font(bold=True, size=11, color="1C2A4A")
                state["row"] += 1
                _img = XLImage(_path)
                # Sized to the ~290-unit sheet budget at 85% zoom (see the width notes
                # above), so a figure lands in the same span the tables use rather than
                # forcing a horizontal scroll.
                _target_w = 1380
                _img.height = int(_img.height * _target_w / _img.width)
                _img.width = _target_w
                ws.add_image(_img, f"A{state['row']}")
                # An anchored image floats over cells rather than occupying them, so the
                # rows underneath must be reserved by hand or the next caption lands on
                # top of it. 17px is the default row height (12.75pt) at 96dpi.
                state["row"] += -(-_img.height // 17) + 2
        except Exception as _viz_error:
            print(f"⚠️ Could not place visuals tab: {_viz_error}")

    use_sheet("Game Context")

    # Table transforms come from the shared cmp_* helpers (same source as the PDF) so
    # columns like defense A+ and the Last-5 TOTAL/AVG row stay in lockstep across outputs.
    starter_row = cmp_starter_row

    ac = advanced_context
    hitter_composite = ac.get('hitter_composite', pd.DataFrame())
    away_perf, home_perf = cmp_enrich_pythag(away_team_performance), cmp_enrich_pythag(home_team_performance)
    # Panel captions name the pitcher instead of saying "Starter". `*_sp` is each club's
    # own starter; `*_sp_vs` is the one that club's hitters have to face.
    away_sp, home_sp = _starter_label(away_starter_info), _starter_label(home_starter_info)
    away_sp_vs, home_sp_vs = home_sp, away_sp
    # Opener detection is needed twice (the Type panel on this sheet and the pitching
    # tab), so resolve it once here. Cached contexts predating the feature build it now.
    season = int(str(game_date)[:4])
    _sp_context_end = _pregame_end_date(game_date) or game_date
    opener_profiles = {}
    for side, pitcher in (('away', away_pitcher), ('home', home_pitcher)):
        profile = ac.get(f'{side}_opener_profile')
        opener_profiles[side] = (profile if profile is not None
                                 else build_opener_profile(pitcher, season, _sp_context_end))

    # ===== PHASE 1: CONTEXT =====
    band("1  ·  GAME CONTEXT")
    perf = cmp_team_form_snapshot(away_perf, home_perf, ac, away_team, home_team)
    env_view, env_fills = cmp_game_environment(env)
    # Team form only fills the left panel, so the weather/park read rides in the
    # otherwise-empty right panel on the same rows.
    pair("Team Form Snapshot   ·   Game Environment (green = run-scoring up; wind marked ≈ is a park-relative estimate)",
         "Season & Recent Form", perf,
         "Weather & Park Factor", env_view,
         col_fills={'right': env_fills})

    section("Last 10 Games (L10 R/RA = runs for/against per game; L10 SoS = avg opponent win %)",
            cmp_l10_table(away_last_10, ac.get('away_recent_detail', pd.DataFrame())),
            cmp_l10_table(home_last_10, ac.get('home_recent_detail', pd.DataFrame())), heat='last10')
    away_relevant = ac.get('away_relevant_games', pd.DataFrame())
    home_relevant = ac.get('home_relevant_games', pd.DataFrame())
    if not (away_relevant.empty and home_relevant.empty):
        # Stacked full width, not paired side by side. At fifteen columns this table is
        # wider than a panel (_PANEL_COLS is 12), so laying it out as an away/home pair put
        # the away side's last column underneath the home side's first -- Rk was silently
        # overwritten by the home panel's Date. Each club gets the full width instead.
        _relevant_caption = ("  ·  Match flags P=park run factor, T=opposing starter hand, "
                             "H=home/away, L=7+ of tonight's nine started (· = tested and "
                             "differed). Rk is the similarity rank driving the Comps tab; "
                             "blank = failed the lineup gate. FIP/HR-9 are context, not "
                             "ranking inputs. If MATCHED's R equals TOTAL's, conditioning "
                             "is telling you nothing tonight")
        for _team, _frame in ((away_team, away_relevant), (home_team, home_relevant)):
            if _frame is not None and not _frame.empty:
                solo(f"{_team} — Last 10 vs Tonight's Conditions{_relevant_caption}",
                     _frame, col_fills=cmp_relevant_games_fills(_frame))
    section("Rest & Schedule Spot",
            av(ac.get('away_rest_schedule', pd.DataFrame()), ['Rest', 'SP Rest', 'Note', 'Pen A/T']),
            av(ac.get('home_rest_schedule', pd.DataFrame()), ['Rest', 'SP Rest', 'Note', 'Pen A/T']))
    away_tz_view, away_tz_fills = cmp_time_zone_view(ac.get('away_time_zone', {}))
    home_tz_view, home_tz_fills = cmp_time_zone_view(ac.get('home_time_zone', {}))
    section("Time-Zone Load (trusted PNAS eastbound baseline)",
            away_tz_view, home_tz_view,
            col_fills={'away': away_tz_fills, 'home': home_tz_fills})
    trip_cols = ['Trip Spot', 'G', 'W%', 'ΔW pp', 'RD/G']
    away_trip = av(ac.get('away_trip_form', pd.DataFrame()), trip_cols)
    home_trip = av(ac.get('home_trip_form', pd.DataFrame()), trip_cols)
    section("Trip / Homestand Form (3Y descriptive; ΔW pp vs team site baseline; shaded = tonight's spot)",
            away_trip, home_trip,
            heat={'W%': 'win_pct', 'RD/G': 'run_diff_pg'},
            col_fills={'away': cmp_highlight_row(away_trip, 'Trip Spot',
                                                 trip_spot_bucket(ac.get('away_trip_spot'))),
                       'home': cmp_highlight_row(home_trip, 'Trip Spot',
                                                 trip_spot_bucket(ac.get('home_trip_spot')))})
    # Each of these tables has exactly one row that describes tonight; shade it so the
    # relevant split doesn't have to be picked out of the season list by eye.
    away_sp_rest = av(ac.get('away_sp_rest_splits', pd.DataFrame()), ['Rest', 'GS', 'IP', 'ERA', 'W-L'])
    home_sp_rest = av(ac.get('home_sp_rest_splits', pd.DataFrame()), ['Rest', 'GS', 'IP', 'ERA', 'W-L'])
    pair("Starter Perf by Days Rest (shaded = tonight's rest)",
         f"{away_sp} — {away_team}", away_sp_rest,
         f"{home_sp} — {home_team}", home_sp_rest, heat={'ERA': 'era'},
         col_fills={'left': cmp_highlight_row(away_sp_rest, 'Rest', ac.get('away_sp_rest_tonight')),
                    'right': cmp_highlight_row(home_sp_rest, 'Rest', ac.get('home_sp_rest_tonight'))})
    away_ctx = av(ac.get('away_schedule_context', pd.DataFrame()), ['Context', 'G', 'R/G', 'RA/G', 'W-L'])
    home_ctx = av(ac.get('home_schedule_context', pd.DataFrame()), ['Context', 'G', 'R/G', 'RA/G', 'W-L'])
    section("Team Perf by Schedule Spot (R/G off, RA/G pitch; shaded = tonight's spot)",
            away_ctx, home_ctx,
            heat={'R/G': 'rpg_off', 'RA/G': 'rpg_allowed'},
            col_fills={'away': cmp_highlight_row(away_ctx, 'Context', ac.get('away_context_tonight')),
                       'home': cmp_highlight_row(home_ctx, 'Context', ac.get('home_context_tonight'))})

    # ===== PHASE 2: OFFENSE =====
    band("2  ·  OFFENSE")
    lineup_cols = ['Spot', 'Name', 'Pos', 'Bats', 'OPS', 'Off Szn', 'Off L28', 'ISO', 'HR', 'SB']
    section("Projected Lineups",
            av(_lineup_with_offense(away_lineup_df, hitter_composite), lineup_cols, 9),
            av(_lineup_with_offense(home_lineup_df, hitter_composite), lineup_cols, 9), heat='offense')
    hot_cols = ['Name', 'PA', 'OPS', 'xwOBA', 'Szn xwOBA', 'ΔxwOBA', 'HardHit%',
            'Chase%', 'Z-Con%', 'Status']
    section("Hot / Cold Hitters (Last 14 Days — vs each hitter's own season)",
            av(_hot_cold_sorted(away_hotcold_df), hot_cols, 9),
            av(_hot_cold_sorted(home_hotcold_df), hot_cols, 9), heat='hot_cold')
    section("Team Hitting Summary (Home/Away and opposing hand)",
            av(ac.get('away_lineup_splits', pd.DataFrame()), ['Split', 'PA', 'OPS', 'xwOBA', 'ISO', 'K%', 'BB%', 'HardHit%', 'HR%'], 8),
            av(ac.get('home_lineup_splits', pd.DataFrame()), ['Split', 'PA', 'OPS', 'xwOBA', 'ISO', 'K%', 'BB%', 'HardHit%', 'HR%'], 8), heat='offense')
    # Each lineup faces the *other* club's starter, so the away panel describes the home
    # starter. Sits right under the hitting splits to give those splits a named opponent.
    pair("Opposing Starter Type (what each lineup is actually facing tonight)",
         f"{away_team} faces {away_sp_vs}",
         cmp_pitcher_type_view(home_starter_info, home_arsenal_df,
                               ac.get('away_type_results'), ac.get('away_vs_home_arsenal'),
                               opener=opener_profiles['home'],
                               batter_arsenal=ac.get('away_batter_arsenal'),
                               lineup_df=away_lineup_df),
         f"{home_team} faces {home_sp_vs}",
         cmp_pitcher_type_view(away_starter_info, away_arsenal_df,
                               ac.get('home_type_results'), ac.get('home_vs_away_arsenal'),
                               opener=opener_profiles['away'],
                               batter_arsenal=ac.get('home_batter_arsenal'),
                               lineup_df=home_lineup_df))
    ha = get_pitcher_handedness(home_pitcher) or 'R'
    hh = get_pitcher_handedness(away_pitcher) or 'R'

    def season_splits(frame, hand):
        """Per-batter season line vs the hand they face tonight, plus its heat map."""
        cols = ['Name'] + [f'{stat} vs {hand}' for stat in
                           ('PA', 'AVG', 'OBP', 'SLG', 'OPS', 'ISO', 'HR', 'BB%', 'K%')]
        table = av(frame, cols, 9)
        if f'PA vs {hand}' in table.columns:
            table = table.sort_values(f'PA vs {hand}', ascending=False)
        heat = {f'OPS vs {hand}': 'ops_off', f'OBP vs {hand}': 'obp_off',
                f'SLG vs {hand}': 'slg_off', f'AVG vs {hand}': 'avg_off',
                f'ISO vs {hand}': 'iso', f'BB% vs {hand}': 'bb_pct_bat',
                f'K% vs {hand}': 'k_pct_bat'}
        return table, heat

    # Each club faces a different hand, so the two tables carry different column names --
    # pair() writes them independently, which lets them sit side by side instead of
    # stacked full-width.
    away_splits_tb, away_splits_heat = season_splits(away_splits_df, ha)
    home_splits_tb, home_splits_heat = season_splits(home_splits_df, hh)
    pair("Season Splits vs Tonight's Opposing Hand",
         f"{away_team} vs {ha}HP", away_splits_tb,
         f"{home_team} vs {hh}HP", home_splits_tb,
         left_heat=away_splits_heat, right_heat=home_splits_heat)
    bvp_cols = ['Name', 'Years', 'PA', 'AB', 'H', 'HR', 'BB', 'K', 'AVG', 'SLG', 'OPS']
    pair("Prior Matchup History vs Tonight's Starter",
         f"{away_team} vs {away_sp_vs}", av(ac.get('away_bvp', pd.DataFrame()), bvp_cols, 5),
         f"{home_team} vs {home_sp_vs}", av(ac.get('home_bvp', pd.DataFrame()), bvp_cols, 5),
         heat='offense')
    opp_cols = ['Unit', 'Split', 'PA', 'OPS', 'xwOBA', 'ISO', 'K%', 'BB%', 'HardHit%', 'HR%']

    def named_units(frame, starter_info):
        """Cached contexts may still carry the generic "Starter" unit label; swap in the
        pitcher's name so the table reads the same however it was built."""
        table = av(frame, opp_cols, 14)
        name = _short_pitcher_name((starter_info or {}).get('Name'))
        if not table.empty and name and 'Unit' in table.columns:
            table['Unit'] = table['Unit'].replace('Starter', name)
        return table

    pair("Opposing Pitching Allowed (starter + bullpen — colored from hitters' view: high allowed = green)",
         f"{away_team} vs {away_sp_vs} + pen", named_units(ac.get('away_opp_pitching'), home_starter_info),
         f"{home_team} vs {home_sp_vs} + pen", named_units(ac.get('home_opp_pitching'), away_starter_info),
         heat='pitching_allowed_batter_view')
    mc = ['Pitch', 'Usage%', 'Pitches', 'RV/100', 'Whiff%', 'Chase%', 'Contact%', 'xwOBA', 'Edge']
    pair("Pitch-Type Value & Swing Decisions (RV/100 high = hitter)",
         f"{away_team} vs {away_sp_vs}", av(ac.get('away_arsenal_matchup', pd.DataFrame()), mc, 6),
         f"{home_team} vs {home_sp_vs}", av(ac.get('home_arsenal_matchup', pd.DataFrame()), mc, 6),
         heat='pitch_matchup')
    dc = ['Name', 'Pitches', 'RV/100', 'Whiff%', 'Chase%', 'Contact%', 'xwOBA', 'Edge']
    pair("Per-Batter vs Opposing Starter's Arsenal (usage-weighted)",
         f"{away_team} vs {away_sp_vs}", av(ac.get('away_arsenal_drilldown', pd.DataFrame()), dc, 9),
         f"{home_team} vs {home_sp_vs}", av(ac.get('home_arsenal_drilldown', pd.DataFrame()), dc, 9),
         heat='pitch_matchup')

    # ===== PHASE 3: STARTING PITCHING =====
    # ===== PHASES 3 & 4: PITCHING =====
    # Both live on their own sheet, written by `_write_pitching_sheet` further down. They
    # were emitted here too until the duplication became the problem: the starter's season
    # line was written by the same `starter_row` call on both tabs, and the L5 totals and
    # the quality of the offenses faced ended up on opposite sheets.

    # ===== PHASE 5: RUN GAME & DEFENSE =====
    band("5  ·  RUN GAME & DEFENSE")
    section("Baserunning",
            av(away_baserunning_df, ['Name', 'SB', 'CS', 'SB_Att', 'SB%', 'Runs', 'Triples'], 8),
            av(home_baserunning_df, ['Name', 'SB', 'CS', 'SB_Att', 'SB%', 'Runs', 'Triples'], 8), heat='baserunning')
    team_def_cols = ['Team', 'BIP', 'xBA Allowed', 'BA Allowed', 'Hits Saved/G',
                     'IF Saved/G', 'OF Saved/G', 'Frame +Str/G', 'Grade']
    section("Team Defense (expected minus actual hits on balls in play, vs league; "
            "errors excluded — DK charges earned runs only)",
            av(ac.get('away_team_defense', pd.DataFrame()), team_def_cols, 1),
            av(ac.get('home_team_defense', pd.DataFrame()), team_def_cols, 1),
            heat='team_defense')
    away_def_disp, away_def_fills = cmp_defense_view(away_defense_df)
    home_def_disp, home_def_fills = cmp_defense_view(home_defense_df)
    section("Defense (fielding-stat view: chances and errors, not range)",
            away_def_disp, home_def_disp,
            col_fills={'away': away_def_fills, 'home': home_def_fills})

    # ===== PHASE 6: ROSTER ALERTS =====
    away_txn, away_txn_fills = cmp_transaction_view(
        ac, 'away', away_team, game_date, away_bullpen_df
    )
    home_txn, home_txn_fills = cmp_transaction_view(
        ac, 'home', home_team, game_date, home_bullpen_df
    )
    band("6  ·  ROSTER ALERTS")
    section("Recent Transactions (last 7 days) — Move = what happened, Impact = roster effect",
            away_txn, home_txn,
            col_fills={'away': away_txn_fills, 'home': home_txn_fills})

    # ---- DFS leverage: sub-5%-owned players the model still likes ----
    #
    # Low ownership alone is not a finding: most of a slate's cheap end is unowned for the
    # obvious reason. Gated on a top-quartile projection, the same cut the star mark uses, so
    # this table and the glyphs on the name cells can never disagree. The reason column is the
    # board's own `Why` text -- park and wind, opposing starter FIP, platoon split, batting
    # order, arsenal fit -- rather than a second explanation invented here.
    try:
        from dfs import highlight as _dfs_hl
        _lev = _dfs_hl.leverage_table(_dfs_hl.slate_frame(),
                                      games=[f"{away_team}@{home_team}"])
    except Exception:
        _lev = None
    if _lev is not None and not _lev.empty:
        use_sheet("DFS Leverage")
        band("7  ·  DFS LEVERAGE  ·  under 5% projected ownership, top-quartile projection")
        solo("Players the field is off that the model is not — Why = the factors behind the "
             "projection", _lev)

    for entry in sheets.values():
        sheet, widths = entry["ws"], entry["widths"]
        for idx, w in widths.items():
            sheet.column_dimensions[get_column_letter(idx)].width = w
        sheet.freeze_panes = "A3"

    # ---- merged pitching tab: both starters and both bullpens, side by side ----
    arsenal_cols = ['Pitch', 'Usage %', 'release_speed', 'Horiz. Break', 'Vert. Break',
                    'CSW%', 'Stuff+']
    sp_panels = []
    for side, team, pitcher, info, caption, arsenal_df, pen_df in (
        ('away', away_team, away_pitcher, away_starter_info, away_sp,
         away_arsenal_df, away_bullpen_df),
        ('home', home_team, home_pitcher, home_starter_info, home_sp,
         home_arsenal_df, home_bullpen_df),
    ):
        profile = ac.get(f'{side}_sp_profile')
        if profile is None:
            # Older cached contexts predate this tab; the statcast pull is disk-cached
            # from the opposing-pitching table, so building it here is cheap.
            profile = build_starter_pitch_profile(pitcher, season, _sp_context_end)
        l5 = cmp_starter_l5_log(info, season)
        opener = opener_profiles[side]
        opener_view, opener_fills = cmp_opener_view(opener, info)
        bulk_view = (cmp_bulk_arm_view(opener.get('bulk'), season, _sp_context_end)
                     if opener.get('is_opener') else pd.DataFrame())
        pen_view, pen_fills = cmp_bullpen_view(pen_df)
        # Older cached contexts predate the pen L5; rebuild on demand so a re-render from
        # cache still gets the block. Shares the boxscore cache, so it is cheap.
        pen_l5 = ac.get(f'{side}_bullpen_l5')
        if pen_l5 is None:
            pen_l5 = build_bullpen_l5(get_team_id(team), team, game_date, bullpen_df=pen_df)
        pen_group, pen_arms = pen_l5
        # Sliced off the league-wide statcast frame during context assembly, so it rides in
        # the cached payload. An older payload simply has no block, which the panel loop
        # skips rather than printing an empty table.
        pen_batted, pen_batted_base = ac.get(f'{side}_bullpen_batted') or (pd.DataFrame(), {})
        sp_panels.append({
            'title': f"{away_team} at {home_team} — {game_date} — pitching",
            'header': (f"{caption} — {team}"
                       + ("   ⚠ LIKELY OPENER" if opener.get('is_opener') else "")
                       + cmp_starter_provisional_tag(ac, side)),
            'season': starter_row(info),
            'opener': opener_view,
            'opener_fills': opener_fills,
            'bulk': bulk_view,
            'l5': l5,
            'l5_fills': cmp_starter_l5_fills(l5),
            'arsenal': av(arsenal_df.sort_values('Usage %', ascending=False),
                          arsenal_cols, 7),
            'mix': profile.get('mix', pd.DataFrame()),
            'mix_fills': cmp_starter_mix_fills(profile),
            'batted': profile.get('batted', pd.DataFrame()),
            'batted_fills': cmp_starter_batted_fills(profile),
            'pen_form': ac.get(f'{side}_bullpen_form', pd.DataFrame()),
            'pen_l5': pen_group,
            'pen_l5_fills': cmp_pen_l5_fills(pen_group),
            'pen_arms': pen_arms,
            'pen_arms_fills': cmp_pen_arms_fills(pen_arms),
            'pen': pen_view,
            'pen_fills': pen_fills,
            'pen_batted': pen_batted,
            'pen_batted_fills': cmp_bullpen_batted_fills(pen_batted, pen_batted_base),
        })
    pitching_ws = _write_pitching_sheet(wb, sp_panels, hdr_fill, band_fill, subhdr_fill,
                                        border, center, left, get_column_letter)
    # created last, so slide it back to where phases 3-4 sit in the narrative
    if pitching_ws is not None and _PITCHING_SHEET_AFTER in wb.sheetnames:
        target = wb.sheetnames.index(_PITCHING_SHEET_AFTER) + 1
        wb.move_sheet(pitching_ws, offset=target - wb.sheetnames.index(pitching_ws.title))

    # ---- one box-score tab per team: the games most like tonight, both clubs, stacked ----
    # Was "last 3 completed games". Three chronological games spanning a different park, a
    # different handedness and two roster moves are three box scores, not a comparison.
    for side, team, team_id in (('away', away_team, away_team_id), ('home', home_team, home_team_id)):
        bundle = ac.get(f'{side}_comparable_games')
        if bundle is None:
            # Older cached contexts predate this tab; fall back to the chronological
            # boxscores they do carry rather than dropping the sheet.
            legacy = ac.get(f'{side}_recent_boxscores') or build_recent_boxscores(
                team_id, team, game_date)
            if legacy:
                _write_boxscore_sheet(wb, f"{team} Last 3", legacy, hdr_fill, band_fill,
                                      subhdr_fill, border, center, left, get_column_letter)
            continue
        games, summary, note = bundle
        if games:
            _write_boxscore_sheet(wb, f"{team} Comps", games, hdr_fill, band_fill,
                                  subhdr_fill, border, center, left, get_column_letter,
                                  summary=summary,
                                  summary_fills=cmp_comparable_summary_fills(summary),
                                  note=note)

    wb.save(output_path)
    return output_path


# Blocks on the merged pitching sheet, in reading order: who starts, what he throws, how
# he has been going, then the pen behind him. (panel key, stripe label, fills key, heat).
#
# This used to be two sheets. "Starting Pitchers" held the season line, L5 log and
# per-start mix; the "Pitching" band held a second copy of the season line (the same
# `cmp_starter_row` call), a second L5 table off the same `Last Starts` source, and the
# season arsenal. Three of the six blocks were duplicates, and the two halves of the L5
# story -- the totals and the quality of the offenses faced -- were on different tabs.
_PITCHING_BLOCKS = (
    ("season", "Season Line", None, "pitching"),
    ("opener", "Opener Check  ·  is the listed starter actually the matchup?",
     "opener_fills", None),
    ("bulk", "Likely Bulk Arm  ·  who the lineup really faces when an opener starts",
     None, "pitching"),
    ("l5", "Last 5 Starts  ·  Opp OPS/R-G/Rk = quality of the offense faced "
           "(green = strong lineup); TOTAL row carries the combined ERA", "l5_fills", None),
    ("arsenal", "Pitch Arsenal  ·  season usage and shape", None, "arsenal"),
    ("mix", "Pitch Mix & Velocity by Start  ·  drift against the season baseline above",
     "mix_fills", None),
    ("batted", "Batted Ball by Start", "batted_fills", None),
    ("pen_form", "Bullpen Summary and Availability  ·  R/L = full pen mix, "
                 "Avail R/L = mix usable tonight", None, "pitching"),
    ("pen_l5", "Bullpen Last 5 Games  ·  what the pen was asked for "
               "(P/Out = pitches per out, IR = inherited runners scored/inherited)",
     "pen_l5_fills", None),
    ("pen_arms", "Bullpen Last 5 Games by Arm  ·  pitch count per game, newest first; "
                 "+ = outing of 3+ IP, * = position player", "pen_arms_fills", None),
    ("pen", "Bullpen Scouting  ·  T = throws, LHP tinted; IP-weighted totals + L7 usage",
     "pen_fills", "pitching"),
    ("pen_batted", "Bullpen Batted Ball  ·  bold BULLPEN row is the group baseline; "
                   "each arm shaded against it (direction, not good/bad)",
     "pen_batted_fills", None),
)

# First-cell labels that mark a summary line rather than a data row.
_PITCHING_SUMMARY_ROWS = {"SEASON", "TOTAL", "TOTAL / AVG", "BULLPEN"}


def _write_pitching_sheet(wb, panels, hdr_fill, band_fill, subhdr_fill,
                          border, center, left, get_column_letter):
    """The whole pitching picture for one game: both probables and both pens, side by side.

    panels is a list of two dicts (away first) holding the blocks named in
    `_PITCHING_BLOCKS` for one club, plus the pitcher's caption.
    """
    from openpyxl.styles import Font, PatternFill, Alignment

    ws = wb.create_sheet("Pitching")
    ws.sheet_view.showGridLines = False
    ws.sheet_format.defaultRowHeight = 12.75
    ws.sheet_view.zoomScale = 85
    ws.sheet_view.zoomScaleNormal = 85

    widths = {}
    gap = 2
    # Panels are as wide as their widest table; the mix table (Date/Opp/P + 2 cols per
    # pitch) sets the floor, so both starters get identical geometry.
    panel_w = max(
        [len(p[key].columns) for p in panels
         for key, *_ in _PITCHING_BLOCKS
         if isinstance(p.get(key), pd.DataFrame) and not p[key].empty] or [12]
    )
    lefts = [1, 1 + panel_w + gap]
    total_w = panel_w * 2 + gap

    def note_width(col_idx, text, field="", is_header=False):
        field = str(field or "")
        if field == "Metric":       # holds labels like "Avg innings as starter"
            cap = 24.0
        elif field == "Value":      # merged across the panel, so it needs no width here
            return
        elif field in {"Date", "Opp"} or field.endswith("mph"):
            cap = 9.5
        else:
            cap = 7.5
        needed = len(str(text)) + 1.1
        if is_header:
            cap = max(cap, min(needed, 24.0))
        widths[col_idx] = max(widths.get(col_idx, 4.5), min(max(needed, 4.5), cap))

    def band(row, text):
        cell = ws.cell(row=row, column=1, value=text)
        cell.font = Font(bold=True, size=11, color="FFFFFF")
        cell.fill = band_fill
        cell.alignment = left
        for c in range(2, total_w + 1):
            ws.cell(row=row, column=c).fill = band_fill
        ws.row_dimensions[row].height = 17

    def stripe(row, text):
        cell = ws.cell(row=row, column=1, value=text)
        cell.font = Font(bold=True, size=9.5, color="14243C")
        cell.fill = subhdr_fill
        cell.alignment = left
        for c in range(2, total_w + 1):
            ws.cell(row=row, column=c).fill = subhdr_fill

    def table(df, top_row, left_col, col_fills=None, heat=None):
        col_fills = col_fills or {}
        if df is None or df.empty:
            ws.cell(row=top_row, column=left_col, value="No data.").font = Font(italic=True, size=9)
            return 1
        cols = list(df.columns)
        # Metric/Value blocks carry sentences ("Miles Mikolas - followed 7x (64%)...").
        # Merging Value across the rest of the panel shows them in full without forcing
        # the shared stat columns underneath to become 40 wide.
        is_metric = cols == ["Metric", "Value"]
        value_end = left_col + panel_w - 1

        def merge_value(row_idx):
            if is_metric and value_end > left_col + 1:
                ws.merge_cells(start_row=row_idx, start_column=left_col + 1,
                               end_row=row_idx, end_column=value_end)

        for j, c in enumerate(cols):
            cell = ws.cell(row=top_row, column=left_col + j, value=str(c))
            cell.font = Font(bold=True, size=8.5, color="14243C")
            cell.fill = hdr_fill
            cell.alignment = center
            cell.border = border
            note_width(left_col + j, c, c, is_header=True)
        merge_value(top_row)
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            # SEASON is the starter's baseline; TOTAL / TOTAL AVG close the L5 and pen
            # tables. All three are summary lines and are bolded, but -- unlike before --
            # an explicit fill still wins over the grey, so the averaged Opp Rk on a
            # TOTAL row is shaded by the same rule as the starts above it.
            summary = str(row[cols[0]]).strip().upper() in _PITCHING_SUMMARY_ROWS
            for j, c in enumerate(cols):
                value = row[c]
                cell = ws.cell(row=top_row + i, column=left_col + j,
                               value="" if value is None or (isinstance(value, float) and pd.isna(value)) else value)
                cell.font = Font(size=9, bold=summary)
                cell.alignment = left if (j == 0 or (is_metric and j == 1)) else center
                cell.border = border
                fill = None
                forced = col_fills.get(c)
                if forced is not None and i < len(forced) + 1 and forced[i - 1] is not None:
                    fill = PatternFill("solid", fgColor="{:02X}{:02X}{:02X}".format(*forced[i - 1]))
                elif heat:
                    metric = _heat_metric_for(heat, c)
                    rgb = _heat_rgb(_stat_percentile(metric, value)) if metric else None
                    if rgb is not None:
                        fill = PatternFill("solid", fgColor="{:02X}{:02X}{:02X}".format(*rgb))
                if fill is None and summary:
                    fill = PatternFill("solid", fgColor="E8E8E8")
                if fill is not None:
                    cell.fill = fill
                note_width(left_col + j, value, c)
            merge_value(top_row + i)
        return len(df) + 1

    title = ws.cell(row=1, column=1, value=panels[0].get("title", "Pitching"))
    title.font = Font(bold=True, size=13)
    row = 2
    band(row, "PITCHING  ·  starters then bullpens, away left / home right.  "
              "In the per-start blocks, bold SEASON is the baseline and green = above his "
              "season norm, red = below (direction, not good/bad)")
    row += 1

    for idx, panel in enumerate(panels):
        cell = ws.cell(row=row, column=lefts[idx], value=panel["header"])
        cell.font = Font(bold=True, size=11, color="14243C")
        cell.alignment = left
    row += 1

    for key, label, fills_key, heat in _PITCHING_BLOCKS:
        frames = [panel.get(key) for panel in panels]
        # Blocks that only exist in some games -- the bulk arm on an opener night, the pen
        # L5 when the boxscores would not load -- are skipped entirely rather than printing
        # two "No data." cells on every normal game.
        if all(f is None or (isinstance(f, pd.DataFrame) and f.empty) for f in frames):
            continue
        stripe(row, label)
        row += 1
        used = 1
        for idx, panel in enumerate(panels):
            used = max(used, table(panel.get(key), row, lefts[idx],
                                   col_fills=panel.get(fills_key) if fills_key else None,
                                   heat=heat))
        row += used + 1

    for idx, w in widths.items():
        ws.column_dimensions[get_column_letter(idx)].width = w
    for spacer in range(1 + panel_w, lefts[1]):
        ws.column_dimensions[get_column_letter(spacer)].width = 1.5
    ws.freeze_panes = "A4"
    return ws


def _write_boxscore_sheet(wb, title, games, hdr_fill, band_fill, subhdr_fill,
                          border, center, left, get_column_letter,
                          summary=None, summary_fills=None, note=None):
    """One worksheet holding several full box scores stacked top to bottom.

    Each game is a dark banner, a line score, then away batting / away pitching /
    home batting / home pitching stacked vertically (not side by side) so every table
    shares the same narrow column set and the sheet reads as one column of blocks.

    `summary` and `note` head the sheet when the games were selected rather than merely
    taken in order -- the reader has to be able to see why these five and not the last
    five, and the note is where a relaxed lineup gate declares itself."""
    from openpyxl.styles import Font, PatternFill

    totals_fill = PatternFill("solid", fgColor="E8E8E8")
    ws = wb.create_sheet(title[:31])
    ws.sheet_view.showGridLines = False
    ws.sheet_format.defaultRowHeight = 12.75
    ws.sheet_view.zoomScale = 85
    ws.sheet_view.zoomScaleNormal = 85

    widths = {}
    state = {"row": 1}
    # Extra-inning line scores are wider than the batting tables, so size the banners to
    # the widest table on the sheet rather than a fixed guess.
    banner_width = max(
        [len(g.get("linescore").columns) for g in games if g.get("linescore") is not None]
        + [len(_BOX_BATTING_COLS), len(_BOX_PITCHING_COLS)]
    )

    def note_width(col_idx, text):
        # These sheets are only ~15 columns wide, so the name column can take whatever
        # the longest name needs without threatening the 85% fit.
        cap = 30 if col_idx == 1 else 8
        widths[col_idx] = max(widths.get(col_idx, 4.5),
                              min(max(len(str(text)) + 1.0, 4.5), cap))

    def banner(text, fill, size=11, height=17):
        r = state["row"]
        cell = ws.cell(row=r, column=1, value=text)
        cell.font = Font(bold=True, size=size,
                         color="FFFFFF" if fill is band_fill else "14243C")
        cell.fill = fill
        cell.alignment = left
        for c in range(2, banner_width + 1):
            ws.cell(row=r, column=c).fill = fill
        ws.row_dimensions[r].height = height
        state["row"] = r + 1

    def table(df, col_fills=None):
        col_fills = col_fills or {}
        if df is None or df.empty:
            ws.cell(row=state["row"], column=1, value="No data.").font = Font(italic=True, size=9)
            state["row"] += 2
            return
        cols = list(df.columns)
        r = state["row"]
        for j, c in enumerate(cols, start=1):
            cell = ws.cell(row=r, column=j, value=str(c))
            cell.font = Font(bold=True, size=8.5, color="14243C")
            cell.fill = hdr_fill
            cell.alignment = center
            cell.border = border
            note_width(j, c)
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            is_totals = str(row[cols[0]]).strip() in {"TEAM TOTALS", "AVG"}
            for j, c in enumerate(cols, start=1):
                value = row[c]
                cell = ws.cell(row=r + i, column=j,
                               value="" if value is None or (isinstance(value, float) and pd.isna(value)) else value)
                cell.font = Font(size=9, bold=is_totals)
                cell.alignment = left if j == 1 else center
                cell.border = border
                forced = col_fills.get(c)
                if forced is not None and i <= len(forced) and forced[i - 1] is not None:
                    cell.fill = PatternFill("solid", fgColor="{:02X}{:02X}{:02X}".format(*forced[i - 1]))
                elif is_totals:
                    cell.fill = totals_fill
                note_width(j, value)
        state["row"] = r + len(df) + 2

    if summary is not None and not summary.empty:
        banner("WHY THESE GAMES  ·  ranked by similarity to tonight, best first",
               band_fill)
        if note:
            cell = ws.cell(row=state["row"], column=1, value=note)
            cell.font = Font(size=9, italic=True)
            cell.alignment = left
            state["row"] += 2
        table(summary, col_fills=summary_fills)

    for game in games:
        banner(game.get("title", ""), band_fill)
        table(game.get("linescore"))
        for label_key, frame_key in (("away_label", "away_batting"),
                                     ("away_pitch_label", "away_pitching"),
                                     ("home_label", "home_batting"),
                                     ("home_pitch_label", "home_pitching")):
            banner(game.get(label_key, ""), subhdr_fill, size=9.5, height=14)
            table(game.get(frame_key))
        state["row"] += 1

    for idx, w in widths.items():
        ws.column_dimensions[get_column_letter(idx)].width = w
    return ws


def _clean_markdown(value):
    if value is None:
        return ""
    text = str(value)
    try:
        text = text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return text.replace("\n", " ").replace("\r", " ").replace("|", "/").strip()


def _markdown_table(df, columns=None, max_rows=12, sort_col=None, ascending=False):
    if df is None or df.empty:
        return "_No data available._"

    table = df.copy()
    if sort_col and sort_col in table.columns:
        table[sort_col] = pd.to_numeric(table[sort_col], errors="coerce")
        table = table.sort_values(sort_col, ascending=ascending)

    if columns:
        columns = [col for col in columns if col in table.columns]
        if not columns:
            return "_No matching columns available._"
        table = table[columns]

    table = table.head(max_rows).fillna("")
    headers = [_clean_markdown(col) for col in table.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(_clean_markdown(row[col]) for col in table.columns) + " |")
    return "\n".join(lines)


def _markdown_pitcher_similarity_table(df, max_rows=4):
    if df is None or df.empty:
        return "_Insufficient same-handed starter pitch volume for a reliable comp set; arsenal-bucket context is used below._"
    return _markdown_table(df, columns=["Name", "Throws", "Similarity", "Velo", "Top Mix"], sort_col="Similarity", max_rows=max_rows)


def _markdown_bullets(items):
    if not items:
        return "_No notable items._"
    return "\n".join(f"- {_clean_markdown(item)}" for item in items if str(item).strip()) or "_No notable items._"


def _markdown_last_10(df):
    if df is None or df.empty:
        return "_No recent game data available._"
    rows = []
    for _, row in df.head(10).iterrows():
        prefix = "vs" if row.get("Home/Away") == "Home" else "@"
        rows.append(f"- {row.get('Date')}: {prefix} {row.get('Opponent')} {row.get('Score')} {row.get('Result')}")
    return "\n".join(_clean_markdown(row) for row in rows)


def _markdown_team_performance(team, perf, last_10):
    lines = [f"### {team}"]
    if isinstance(perf, dict):
        for key in ["Overall Record", "Last 10 Games", "Last 20 Games", "Last 30 Games", "Current Streak", "Run Differential", "Pythagorean Expectation", "Division", "Division Rank", "Games Back"]:
            if key in perf:
                lines.append(f"- **{key}:** {_clean_markdown(perf[key])}")
    lines.extend(["", "**Recent Games**", _markdown_last_10(last_10)])
    return "\n".join(lines)


def _markdown_starter(info):
    if not isinstance(info, dict):
        return "_Starter data unavailable._"
    return (
        f"**{_clean_markdown(info.get('Name', 'Unknown'))}** ({_clean_markdown(info.get('Throws', '?'))})  \n"
        f"GS: {_clean_markdown(info.get('GS'))} | IP: {_clean_markdown(info.get('IP'))} | "
        f"ERA: {_clean_markdown(info.get('ERA'))} | FIP: {_clean_markdown(info.get('FIP'))} | "
        f"K%: {_format_pct(info.get('K%'))} | BB%: {_format_pct(info.get('BB%'))} | "
        f"WHIP: {_clean_markdown(info.get('WHIP'))}"
    )


def _lineup_markdown_columns(df):
    base = ["Spot", "Name", "Pos", "Bats"]
    stat_cols = ["PA", "AVG", "OBP", "SLG", "OPS", "ISO", "HR", "RBI", "SB"]
    if df is None or df.empty:
        return base
    if "PA" in df.columns and pd.to_numeric(df["PA"], errors="coerce").fillna(0).sum() > 0:
        return base + stat_cols
    return base


def _lineup_with_offense(lineup_df, hitter_composite):
    """Merge the per-batter offense index (Off Szn / Off L28) onto a lineup table by name."""
    if lineup_df is None or lineup_df.empty:
        return lineup_df
    out = lineup_df.copy()
    off_map = {}
    if hitter_composite is not None and not hitter_composite.empty and {"Name", "Off Szn", "Off L28"}.issubset(hitter_composite.columns):
        for _, r in hitter_composite.iterrows():
            off_map[str(r.get("Name"))] = (r.get("Off Szn"), r.get("Off L28"))
    out["Off Szn"] = out["Name"].map(lambda n: off_map.get(str(n), ("", ""))[0])
    out["Off L28"] = out["Name"].map(lambda n: off_map.get(str(n), ("", ""))[1])
    return out


def generate_report_markdown(
    home_lineup_df, home_bullpen_df, home_starter_info, home_hotcold_df, home_arsenal_df, home_pitcher, home_team_performance,
    away_lineup_df, away_bullpen_df, away_starter_info, away_hotcold_df, away_arsenal_df, away_pitcher, away_team_performance,
    game_date, home_team, away_team, home_team_id, away_team_id,
    home_baserunning_df, away_baserunning_df,
    home_defense_df, away_defense_df, home_last_10, away_last_10,
    home_splits_df, away_splits_df,
    output_dir="scouting_reports",
    advanced_context=None,
):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"scouting_report_{game_date}_{away_team}_{home_team}.md")

    lines = [
        f"# {away_team} at {home_team} Scouting Report",
        "",
        f"**Date:** {game_date}",
        f"**Probable Starters:** {away_team} {_clean_markdown(_first_name(away_starter_info))} vs {home_team} {_clean_markdown(_first_name(home_starter_info))}",
        "",
    ]

    advanced_context = advanced_context or {}
    if advanced_context:
        projection = advanced_context.get("projection_summary", {})
        projection_line = (
            f"Projected winner: **{_clean_markdown(projection.get('Projected Winner', 'N/A'))}** | "
            f"Total runs: **{_clean_markdown(projection.get('Total Runs', 'N/A'))}** | "
            f"Win: **{away_team} {_clean_markdown(projection.get('Away Win%', 'N/A'))}% / "
            f"{home_team} {_clean_markdown(projection.get('Home Win%', 'N/A'))}%** | "
            f"TZ: **{_clean_markdown(projection.get('Time-Zone Adjustment', '0.0pp'))}** | "
            f"Model: **{_clean_markdown(projection.get('Model', 'heuristic'))}**"
            if projection else ""
        )
        lines.extend([
            "## Quick Read",
            "",
            "### Game Edge",
            projection_line,
            "",
            _markdown_table(
                advanced_context.get("scorecard"),
                columns=["Team", "Win Lean", "TZ Study", "Exp Runs", "Recent", "Recent Opp", "Lineup", "Similar SP", "vs Arsenal", "Pitcher Type", "Opp SP FIP", "Bullpen", "Run Game", "Defense Fit"],
                max_rows=2,
            ),
            "",
            "### Why This Edge Exists",
            _markdown_table(
                advanced_context.get("factor_matrix"),
                columns=["Factor", "Edge", "Confidence", "Why"],
                max_rows=8,
            ),
            "",
            "### Risk Flags",
            _markdown_bullets(advanced_context.get("uncertainty_flags", [])),
            "",
            "### Hitter Signal Summary",
            _markdown_table(
                advanced_context.get("hitter_summary"),
                columns=["Team", "Priority Bats", "Best Arsenal Fit", "Best Similar Fit", "BvP Note"],
                max_rows=2,
            ),
            "",
            "### Hitter Composite",
            _markdown_table(
                advanced_context.get("hitter_composite"),
                columns=[
                    "Team", "Rank", "Name", "Bats", "Season", "Platoon",
                    "Arsenal", "Similar", "BvP", "Best Signal", "Signal",
                ],
                max_rows=18,
            ),
            "",
            "### Hitters To Watch",
            _markdown_table(advanced_context.get("hitter_watchlist"), columns=["Team", "Name", "Why", "OPS", "ISO", "Platoon", "Similar Sample", "OPS Pctl", "HardHit Pctl"], sort_col="Priority", max_rows=6),
            "",
            "### Pitchers To Watch",
            _markdown_table(advanced_context.get("pitcher_watchlist"), columns=["Team", "Pitcher", "Role", "Why", "FIP", "K-BB", "Opponent Arsenal", "K Edge"], sort_col="Score", max_rows=2),
            "",
        ])

    if advanced_context:
        hitter_composite = advanced_context.get("hitter_composite")
        lineup_offense_columns = ["Spot", "Name", "Pos", "Bats", "OPS", "Off Szn", "Off L28", "ISO", "HR", "SB", "Lineup Confidence"]
        for _team_offense_row in [
            (away_team, away_lineup_df, advanced_context.get("away_lineup_splits", pd.DataFrame()), advanced_context.get("away_opp_pitching", pd.DataFrame()),
             advanced_context.get("away_arsenal_matchup", pd.DataFrame()), advanced_context.get("away_arsenal_drilldown", pd.DataFrame()), _first_name(home_starter_info)),
            (home_team, home_lineup_df, advanced_context.get("home_lineup_splits", pd.DataFrame()), advanced_context.get("home_opp_pitching", pd.DataFrame()),
             advanced_context.get("home_arsenal_matchup", pd.DataFrame()), advanced_context.get("home_arsenal_drilldown", pd.DataFrame()), _first_name(away_starter_info)),
        ]:
            team, lineup_df, splits, opp_pitching, arsenal_matchup, arsenal_drilldown, opp_sp = _team_offense_row
            lines.extend([
                f"## {team} Offense",
                "_Off Szn / Off L28 = league-relative offense index (100 = league avg; proxy, not official wRC+)._",
                "",
                "### Lineup",
                _markdown_table(_lineup_with_offense(lineup_df, hitter_composite), columns=lineup_offense_columns, max_rows=9),
                "",
                "### Split Summary",
                "_Aggregate lineup OPS/xwOBA. Home/Away park-factor adjusted; `*` marks the opposing starter's handedness._",
                _markdown_table(splits, columns=["Split", "PA", "OPS", "xwOBA", "ISO", "K%", "BB%", "HardHit%", "HR%"], max_rows=8),
                "",
                f"### Pitch-Type Value & Swing Decisions vs {_clean_markdown(opp_sp)}",
                "_Lineup RV/100 (batter run value; higher = better), Whiff/Chase/Contact and xwOBA per pitch he throws._",
                _markdown_table(arsenal_matchup, columns=["Pitch", "Usage%", "Pitches", "RV/100", "Whiff%", "Chase%", "Contact%", "xwOBA", "Edge"], max_rows=6),
                "",
                f"### Per-Batter vs {_clean_markdown(opp_sp)} Arsenal",
                _markdown_table(arsenal_drilldown, columns=["Name", "Pitches", "RV/100", "Whiff%", "Chase%", "Contact%", "xwOBA", "Edge"], max_rows=9),
                "",
                "### Opposing Pitching Allowed (Starter + Bullpen)",
                "_OPS/xwOBA allowed by the pitching staff this lineup faces. Home/Away park-factor adjusted._",
                _markdown_table(opp_pitching, columns=["Unit", "Split", "PA", "OPS", "xwOBA", "ISO", "K%", "BB%", "HardHit%", "HR%"], max_rows=14),
                "",
            ])

    env = advanced_context.get("environment", {})
    if advanced_context:
        lines.extend([
            "## Matchup Context",
            "",
            "### Lineup Shape And Pitcher Plans",
            _markdown_bullets(advanced_context.get("lineup_notes", []) + advanced_context.get("pitcher_lineup_notes", [])),
            "",
            "### Starter Batter-Handedness Splits",
            _markdown_table(
                pd.concat([
                    advanced_context.get("home_pitcher_hand_splits", pd.DataFrame()),
                    advanced_context.get("away_pitcher_hand_splits", pd.DataFrame()),
                ], ignore_index=True),
                columns=["Pitcher", "Batter Side", "PA", "OPS", "SLG", "xwOBA", "K%", "HardHit%", "Split Tag"],
                max_rows=4,
            ),
            "",
            "### Team vs Starter Arsenal",
            _markdown_table(
                pd.concat([
                    advanced_context.get("away_vs_home_arsenal", pd.DataFrame()),
                    advanced_context.get("home_vs_away_arsenal", pd.DataFrame()),
                ], ignore_index=True),
                columns=["Team", "Basis", "PA", "OPS", "xwOBA", "SLG", "HR", "K%", "HardHit%", "Whiff%", "League OPS Pctl", "Arsenal Tag"],
                max_rows=2,
            ),
            "",
            f"### {away_team} Batter Arsenal Fits",
            _markdown_table(
                advanced_context.get("away_batter_arsenal"),
                columns=["Name", "PA", "OPS", "xwOBA", "SLG", "HR", "K%", "K Edge", "HardHit%", "Whiff%", "Fit"],
                sort_col="Arsenal Score",
                max_rows=5,
            ),
            "",
            f"### {home_team} Batter Arsenal Fits",
            _markdown_table(
                advanced_context.get("home_batter_arsenal"),
                columns=["Name", "PA", "OPS", "xwOBA", "SLG", "HR", "K%", "K Edge", "HardHit%", "Whiff%", "Fit"],
                sort_col="Arsenal Score",
                max_rows=5,
            ),
            "",
            f"### Similar Pitchers To {_clean_markdown(_first_name(home_starter_info))}",
            _markdown_pitcher_similarity_table(advanced_context.get("home_pitcher_similar"), max_rows=4),
            "",
            f"### {away_team} vs Similar {home_team} Pitchers",
            _markdown_table(advanced_context.get("away_similarity"), columns=["Name", "PA", "AVG", "SLG", "OPS_proxy", "HR", "K%"], sort_col="OPS_proxy", max_rows=5),
            "",
            f"### Similar Pitchers To {_clean_markdown(_first_name(away_starter_info))}",
            _markdown_pitcher_similarity_table(advanced_context.get("away_pitcher_similar"), max_rows=4),
            "",
            f"### {home_team} vs Similar {away_team} Pitchers",
            _markdown_table(advanced_context.get("home_similarity"), columns=["Name", "PA", "AVG", "SLG", "OPS_proxy", "HR", "K%"], sort_col="OPS_proxy", max_rows=5),
            "",
            "### Starter Opponent Quality Faced",
            _markdown_table(
                pd.concat([
                    advanced_context.get("away_pitcher_quality", pd.DataFrame()),
                    advanced_context.get("home_pitcher_quality", pd.DataFrame()),
                ], ignore_index=True),
                max_rows=2,
            ),
            "",
            "### Team Results Vs Pitcher Type",
            _markdown_table(
                pd.concat([
                    advanced_context.get("away_type_results", pd.DataFrame()),
                    advanced_context.get("home_type_results", pd.DataFrame()),
                ], ignore_index=True),
                columns=["Team", "Pitcher Type", "Games", "Record", "Runs/G", "PA", "Type OPS", "Baseline OPS", "OPS Diff", "Consistency"],
                max_rows=2,
            ),
            "",
            "### Batter vs Pitcher History",
            f"**{away_team} hitters vs {_clean_markdown(_first_name(home_starter_info))}**",
            _markdown_table(advanced_context.get("away_bvp"), columns=["Name", "Years", "AB", "H", "HR", "BB", "K", "AVG", "SLG", "OPS"], sort_col="PA", max_rows=3),
            "",
            f"**{home_team} hitters vs {_clean_markdown(_first_name(away_starter_info))}**",
            _markdown_table(advanced_context.get("home_bvp"), columns=["Name", "Years", "AB", "H", "HR", "BB", "K", "AVG", "SLG", "OPS"], sort_col="PA", max_rows=3),
            "",
            "## Bullpen Deployment",
            "",
            _markdown_bullets(advanced_context.get("bullpen_notes", [])),
            "",
            "### Bullpen Form Summary",
            _markdown_table(
                pd.concat([
                    advanced_context.get("away_bullpen_form", pd.DataFrame()),
                    advanced_context.get("home_bullpen_form", pd.DataFrame()),
                ], ignore_index=True),
                max_rows=2,
            ),
            "",
            f"### {away_team} Relief Availability",
            _markdown_table(away_bullpen_df, columns=["Name", "L7 Usage", "Closer Role", "Throws", "Availability", "Save Efficiency", "Bullpen Score", "Availability Score", "Last3D_Pitches", "Last Outing", "FIP", "K%"], sort_col="Bullpen Score", max_rows=5),
            "",
            f"### {home_team} Relief Availability",
            _markdown_table(home_bullpen_df, columns=["Name", "L7 Usage", "Closer Role", "Throws", "Availability", "Save Efficiency", "Bullpen Score", "Availability Score", "Last3D_Pitches", "Last Outing", "FIP", "K%"], sort_col="Bullpen Score", max_rows=5),
            "",
            "## Park, Weather & Defensive Stress",
            "",
            "### Park And Weather Impact",
            _markdown_bullets(env.get("notes", [])),
            "",
            "### Team Defense (batted-ball outcomes vs expectation, league-centred)",
            "_Hits Saved/G is (expected hits - actual hits) on balls in play, against league "
            "average. Positive means this defence turns more balls into outs than the "
            "contact quality it faced deserved. Errors are deliberately absent: they produce "
            "unearned runs, and DK charges a pitcher for earned runs only._",
            _markdown_table(
                pd.concat([
                    advanced_context.get("away_team_defense", pd.DataFrame()),
                    advanced_context.get("home_team_defense", pd.DataFrame()),
                ], ignore_index=True),
                columns=["Team", "BIP", "xBA Allowed", "BA Allowed", "Hits Saved/G",
                         "IF Saved/G", "OF Saved/G", "Frame +Str/G", "Grade", "DFS Read"],
                max_rows=2,
            ),
            "",
            "### Park/Defense Impact",
            _markdown_table(
                pd.concat([
                    advanced_context.get("away_park_defense", pd.DataFrame()),
                    advanced_context.get("home_park_defense", pd.DataFrame()),
                ], ignore_index=True),
                max_rows=2,
            ),
            "",
            f"### {away_team} Defensive Stress",
            _markdown_table(advanced_context.get("away_defense_stress"), max_rows=5),
            "",
            f"### {home_team} Defensive Stress",
            _markdown_table(advanced_context.get("home_defense_stress"), max_rows=5),
            "",
            "### Catcher Run Game",
            _markdown_table(
                pd.concat([
                    advanced_context.get("away_catcher_run_game", pd.DataFrame()),
                    advanced_context.get("home_catcher_run_game", pd.DataFrame()),
                ], ignore_index=True),
                columns=["Defense", "Catcher", "Inn", "Run Control", "CS%", "CS", "SB Allowed", "PB", "Opp SB_Att", "Opp SB%", "Aggression", "Priority"],
                max_rows=4,
            ),
            "",
            "## Notable Absences",
            "",
            f"### {away_team}",
            _markdown_bullets(advanced_context.get("away_absences", [])),
            "",
            f"### {home_team}",
            _markdown_bullets(advanced_context.get("home_absences", [])),
            "",
        ])

    lines.extend([
        "## Team Context",
        "",
        "### Rest & Schedule Spot",
        _markdown_table(
            pd.concat([
                advanced_context.get("away_rest_schedule", pd.DataFrame()),
                advanced_context.get("home_rest_schedule", pd.DataFrame()),
            ], ignore_index=True),
            columns=["Team", "Rest", "SP Rest", "Note", "Pen A/T"],
            max_rows=2,
        ),
        "",
        f"**{away_team} {_clean_markdown(_first_name(away_starter_info))} — performance by days rest**",
        _markdown_table(advanced_context.get("away_sp_rest_splits"), columns=["Rest", "GS", "IP", "ERA", "W-L"], max_rows=3),
        f"**{home_team} {_clean_markdown(_first_name(home_starter_info))} — performance by days rest**",
        _markdown_table(advanced_context.get("home_sp_rest_splits"), columns=["Rest", "GS", "IP", "ERA", "W-L"], max_rows=3),
        "",
        "### Time-Zone / Circadian Load",
        "_Venue-path estimate with one hour of acclimation per available day. The trusted PNAS baseline applies -3.5 pp to an exposed home team's win probability and -2.1 pp to an exposed away team's win probability for eastward residuals of at least two hours; westward travel remains informational._",
        f"**{away_team}:** {_clean_markdown((advanced_context.get('away_time_zone') or {}).get('Summary', ''))}",
        _markdown_table(
            cmp_time_zone_view(advanced_context.get("away_time_zone", {}))[0],
            columns=["Route", "Shift", "Residual", "Accl", "Level", "Study"],
            max_rows=1,
        ),
        f"**{home_team}:** {_clean_markdown((advanced_context.get('home_time_zone') or {}).get('Summary', ''))}",
        _markdown_table(
            cmp_time_zone_view(advanced_context.get("home_time_zone", {}))[0],
            columns=["Route", "Shift", "Residual", "Accl", "Level", "Study"],
            max_rows=1,
        ),
        "",
        "### Trip / Homestand Form",
        "_Three-year descriptive splits; ΔW pp is versus that team's own home/road baseline._",
        f"**{away_team}:** {_clean_markdown(advanced_context.get('away_trip_spot', ''))}",
        _markdown_table(
            advanced_context.get("away_trip_form"),
            columns=["Trip Spot", "G", "W%", "ΔW pp", "RD/G"],
            max_rows=8,
        ),
        f"**{home_team}:** {_clean_markdown(advanced_context.get('home_trip_spot', ''))}",
        _markdown_table(
            advanced_context.get("home_trip_form"),
            columns=["Trip Spot", "G", "W%", "ΔW pp", "RD/G"],
            max_rows=8,
        ),
        "",
        "### Team Performance by Schedule Spot",
        "_R/G = offense (hitters), RA/G = pitching. Frames output through the schedule/rest notes._",
        f"**{away_team}**",
        _markdown_table(advanced_context.get("away_schedule_context"), columns=["Context", "G", "R/G", "RA/G", "W-L"], max_rows=4),
        f"**{home_team}**",
        _markdown_table(advanced_context.get("home_schedule_context"), columns=["Context", "G", "R/G", "RA/G", "W-L"], max_rows=4),
        "",
        "### Recent Opponent Context",
        _markdown_table(
            pd.concat([
                advanced_context.get("away_recent_summary", pd.DataFrame()),
                advanced_context.get("home_recent_summary", pd.DataFrame()),
            ], ignore_index=True),
            columns=["Team", "Recent", "Runs/G", "Allowed/G", "Opp Quality", "Opp Quality Score", "Avg Opp W%", "Avg Opp R/G", "Star SP Faced"],
            max_rows=2,
        ),
        "",
        "### Rolling Team Form",
        _markdown_table(
            pd.concat([
                advanced_context.get("away_rolling_form", pd.DataFrame()),
                advanced_context.get("home_rolling_form", pd.DataFrame()),
            ], ignore_index=True),
            max_rows=6,
        ),
        "",
        f"### {away_team} Recent Opponents",
        _markdown_table(
            advanced_context.get("away_recent_detail"),
            columns=["Date", "Opp", "Result", "Score", "Opp W%", "Opp Quality Score", "SP Faced", "SP FIP"],
            max_rows=4,
        ),
        "",
        f"### {home_team} Recent Opponents",
        _markdown_table(
            advanced_context.get("home_recent_detail"),
            columns=["Date", "Opp", "Result", "Score", "Opp W%", "Opp Quality Score", "SP Faced", "SP FIP"],
            max_rows=4,
        ),
        "",
        "## Supporting Tables",
        "",
        "### Starter Arsenals",
        f"**{away_team} {_clean_markdown(_first_name(away_starter_info))}**",
        _markdown_table(away_arsenal_df, columns=["Pitch", "Usage %", "release_speed", "Stuff+", "Zone%", "CSW%"], sort_col="Usage %", max_rows=5),
        "",
        f"**{home_team} {_clean_markdown(_first_name(home_starter_info))}**",
        _markdown_table(home_arsenal_df, columns=["Pitch", "Usage %", "release_speed", "Stuff+", "Zone%", "CSW%"], sort_col="Usage %", max_rows=5),
        "",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_path


def _format_html_cell(value):
    text = html_lib.escape(str(value))
    if text.startswith("ELITE P"):
        return f'<span class="badge elite">{text}</span>'
    if text.startswith("AVG P"):
        return f'<span class="badge avg">{text}</span>'
    if text.startswith("LOW P"):
        return f'<span class="badge low">{text}</span>'
    if text.endswith("%"):
        try:
            pct = float(text.rstrip("%"))
            if pct >= 60:
                return f'<span class="badge elite">{text}</span>'
            if pct <= 40:
                return f'<span class="badge low">{text}</span>'
            return f'<span class="badge avg">{text}</span>'
        except ValueError:
            pass
    if "OPS pts" in text:
        try:
            points = int(text.split("OPS pts")[0].strip().split()[-1].replace("+", ""))
            if points >= 40:
                return f'<span class="badge elite">{text}</span>'
            if points <= -40:
                return f'<span class="badge low">{text}</span>'
            return f'<span class="badge avg">{text}</span>'
        except ValueError:
            pass
    if text.startswith("High") or text.startswith("Taxed") or text.startswith("Stress") or text.startswith("Risk") or text.startswith("Caution") or text.startswith("Soft") or text.startswith("Fallback") or text.startswith("Fade"):
        return f'<span class="badge low">{text}</span>'
    if text.startswith("Medium") or text.startswith("Monitor") or text.startswith("Typical") or text.startswith("Solid") or text.startswith("Neutral") or text.startswith("Average") or text.startswith("Small sample") or text.startswith("Watch"):
        return f'<span class="badge avg">{text}</span>'
    if text.startswith("Low") or text.startswith("Available") or text.startswith("More stable") or text.startswith("Advantage") or text.startswith("Plus") or text.startswith("Attack") or text.startswith("Hard") or text.startswith("Confirmed") or text.startswith("Priority"):
        return f'<span class="badge elite">{text}</span>'
    return text


def _markdown_to_html(markdown_text):
    lines = markdown_text.splitlines()
    html_lines = []
    current_h2 = ""
    current_h3 = ""
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("### "):
            current_h3 = line[4:]
            html_lines.append(f"<h3>{html_lib.escape(current_h3)}</h3>")
            i += 1
            continue
        if line.startswith("## "):
            current_h2 = line[3:]
            current_h3 = ""
            html_lines.append(f"<h2>{html_lib.escape(current_h2)}</h2>")
            i += 1
            continue
        if line.startswith("# "):
            html_lines.append(f"<h1>{html_lib.escape(line[2:])}</h1>")
            i += 1
            continue
        if line.startswith("| ") and i + 1 < len(lines) and lines[i + 1].startswith("| ---"):
            headers = [cell.strip() for cell in line.strip("|").split("|")]
            context = _resolve_heat_context(current_h3) or _resolve_heat_context(current_h2)
            col_metrics = [_heat_metric_for(context, h) for h in headers]
            html_lines.append("<table><thead><tr>" + "".join(f"<th>{html_lib.escape(h)}</th>" for h in headers) + "</tr></thead><tbody>")
            i += 2
            while i < len(lines) and lines[i].startswith("| "):
                cells = [cell.strip() for cell in lines[i].strip("|").split("|")]
                tds = []
                for j, cell in enumerate(cells):
                    metric = col_metrics[j] if j < len(col_metrics) else None
                    rgb = _heat_rgb(_stat_percentile(metric, cell)) if metric else None
                    if rgb is not None:
                        tds.append(f'<td style="background:rgb{rgb}">{html_lib.escape(cell)}</td>')
                    else:
                        tds.append(f"<td>{_format_html_cell(cell)}</td>")
                html_lines.append("<tr>" + "".join(tds) + "</tr>")
                i += 1
            html_lines.append("</tbody></table>")
            continue
        if line.startswith("- "):
            html_lines.append("<ul>")
            while i < len(lines) and lines[i].startswith("- "):
                html_lines.append(f"<li>{html_lib.escape(lines[i][2:])}</li>")
                i += 1
            html_lines.append("</ul>")
            continue
        html_lines.append(f"<p>{html_lib.escape(line)}</p>")
        i += 1
    return "\n".join(html_lines)


def generate_report_html(*args, output_dir="scouting_reports", advanced_context=None):
    markdown_path = generate_report_markdown(*args, output_dir=output_dir, advanced_context=advanced_context)
    markdown_text = open(markdown_path, "r", encoding="utf-8").read()
    body = _markdown_to_html(markdown_text)
    html_path = markdown_path.replace(".md", ".html")
    css = """
body { font-family: Segoe UI, Arial, sans-serif; margin: 28px; color: #1f2937; background: #fafafa; }
h1 { margin-bottom: 4px; }
h2 { margin-top: 28px; border-bottom: 2px solid #d1d5db; padding-bottom: 4px; }
h3 { margin-top: 18px; }
table { border-collapse: collapse; width: 100%; margin: 10px 0 18px; font-size: 13px; background: white; }
th, td { border: 1px solid #d1d5db; padding: 6px 8px; text-align: left; vertical-align: top; }
th { background: #f3f4f6; position: sticky; top: 0; }
tr:nth-child(even) { background: #fbfbfb; }
.badge { display: inline-block; padding: 2px 7px; border-radius: 4px; font-weight: 700; font-size: 12px; }
.elite { background: #d8f3dc; color: #14532d; }
.avg { background: #fff3bf; color: #713f12; }
.low { background: #ffd6d6; color: #7f1d1d; }
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(f"<!doctype html><html><head><meta charset='utf-8'><title>Scouting Report</title><style>{css}</style></head><body>{body}</body></html>")
    return html_path



#%%

# Function to get all games scheduled for a given date
def get_available_games(date):
    schedule = statsapi.schedule(start_date=date, end_date=date)
    
    if not schedule:
        print(f"⚠️ No games scheduled on {date}")
        return []

    games = []
    for idx, game in enumerate(schedule):
        try:
            # Use 'home_name' and 'away_name' for teams
            home_team = game['home_name']
            away_team = game['away_name']
            games.append({
                'game_number': idx + 1,
                'home_team': home_team,
                'away_team': away_team,
                'game_id': game['game_id'],
                'status': game.get('status', ''),
                # Doubleheaders put two games on the same date for the same pairing. Without
                # these, every lookup silently resolves to game 1 -- wrong starter, wrong
                # lineup, and a cache entry that overwrites the other game.
                'game_num': int(game.get('game_num') or 1),
                'doubleheader': game.get('doubleheader', 'N'),
                'game_datetime': game.get('game_datetime', ''),
                'home_probable_pitcher': game.get('home_probable_pitcher', ''),
                'away_probable_pitcher': game.get('away_probable_pitcher', '')
            })
        except KeyError as e:
            print(f"⚠️ Missing key {e} for game {idx + 1}. Skipping this game.")
            continue
    
    return games


def _extract_boxscore_lineup(box, side):
    players = box.get(side, {}).get("players", {}) if isinstance(box, dict) else {}
    lineup = []
    seen_spots = set()
    for player in players.values():
        batting_order = player.get("battingOrder")
        if not batting_order:
            continue
        try:
            order_num = int(batting_order)
        except (TypeError, ValueError):
            continue
        spot = order_num // 100
        if not (100 <= order_num <= 999 and 1 <= spot <= 9) or spot in seen_spots:
            continue
        pos_info = player.get("position", {}) or {}
        pos_abbr = pos_info.get("abbreviation")
        if pos_abbr == "PH":
            continue
        seen_spots.add(spot)
        person = player.get("person", {}) or {}
        lineup.append({
            "Spot": spot,
            "Name": remove_accents(person.get("fullName", "")),
            "Pos": pos_abbr or "",
            "ID": person.get("id", ""),
        })
    lineup.sort(key=lambda row: row["Spot"])
    return lineup


def _lineup_confirmed(lineup):
    spots = [row.get("Spot") for row in lineup]
    return len(lineup) == 9 and spots == list(range(1, 10))


def _lineup_names(lineup):
    if not lineup:
        return ""
    return "; ".join(
        f"{row.get('Spot')}. {row.get('Name')} ({row.get('Pos')})".strip()
        for row in lineup
    )


def check_confirmed_lineups_for_date(date, reports_root="scouting_reports", dated_output=True, save_csv=True):
    schedule = statsapi.schedule(start_date=date, end_date=date)
    output_dir = report_output_dir(date, reports_root, dated_output)
    rows = []
    failures = []

    for game in schedule:
        game_id = game.get("game_id")
        away_team = get_team_abbreviation(game.get("away_name", "")) or game.get("away_name", "")
        home_team = get_team_abbreviation(game.get("home_name", "")) or game.get("home_name", "")
        label = f"{away_team}@{home_team}"
        try:
            box = statsapi.boxscore_data(game_id)
            away_lineup = _extract_boxscore_lineup(box, "away")
            home_lineup = _extract_boxscore_lineup(box, "home")
            away_confirmed = _lineup_confirmed(away_lineup)
            home_confirmed = _lineup_confirmed(home_lineup)
            rows.append({
                "Game": label,
                "Game ID": game_id,
                "Status": game.get("status", ""),
                "First Pitch": game.get("game_datetime", game.get("game_date", "")),
                "Away": away_team,
                "Home": home_team,
                "Away Lineup": "Confirmed" if away_confirmed else f"{len(away_lineup)}/9",
                "Home Lineup": "Confirmed" if home_confirmed else f"{len(home_lineup)}/9",
                "Ready": "Yes" if away_confirmed and home_confirmed else "No",
                "Away Players": _lineup_names(away_lineup),
                "Home Players": _lineup_names(home_lineup),
            })
        except Exception as exc:
            failures.append({"Game": label, "Game ID": game_id, "Reason": str(exc)})
            rows.append({
                "Game": label,
                "Game ID": game_id,
                "Status": game.get("status", ""),
                "First Pitch": game.get("game_datetime", game.get("game_date", "")),
                "Away": away_team,
                "Home": home_team,
                "Away Lineup": "Error",
                "Home Lineup": "Error",
                "Ready": "No",
                "Away Players": "",
                "Home Players": "",
            })

    status_df = pd.DataFrame(rows)
    ready_df = status_df[status_df["Ready"] == "Yes"].copy() if not status_df.empty else pd.DataFrame(columns=status_df.columns)
    paths = {}
    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        status_path = os.path.join(output_dir, f"lineup_status_{date}.csv")
        ready_path = os.path.join(output_dir, f"confirmed_lineups_{date}.csv")
        status_df.to_csv(status_path, index=False)
        ready_df.to_csv(ready_path, index=False)
        paths = {"status": status_path, "ready": ready_path}
        if failures:
            failure_path = os.path.join(output_dir, f"lineup_check_failures_{date}.csv")
            pd.DataFrame(failures).to_csv(failure_path, index=False)
            paths["failures"] = failure_path
    return ready_df, status_df, paths


@lru_cache(maxsize=1)
def get_team_list():
    return cached_json_request("https://statsapi.mlb.com/api/v1/teams?sportId=1", namespace="statsapi")['teams']

def get_team_id(team_identifier):
    alias_map = {'ari': 'az', 'oak': 'ath'}
    team_identifier = alias_map.get(team_identifier.lower(), team_identifier.lower())

    for team in get_team_list():
        if team_identifier in (team['abbreviation'].lower(), team['name'].lower()):
            return team['id']

    raise ValueError(f"Team '{team_identifier}' not found.")


def get_recent_transactions(team_abbr, days_back=7, as_of_date=None):
    TEAM_ID_MAP = {
        'ARI': 109, 'ATL': 144, 'BAL': 110, 'BOS': 111, 'CHC': 112, 'CIN': 113, 'CLE': 114, 'COL': 115,
        'CWS': 145, 'DET': 116, 'HOU': 117, 'KC': 118, 'LAA': 108, 'LAD': 119, 'MIA': 146, 'MIL': 158,
        'MIN': 142, 'NYM': 121, 'NYY': 147, 'OAK': 133, 'ATH': 133, 'PHI': 143, 'PIT': 134, 'SD': 135, 'SEA': 136,
        'SF': 137, 'STL': 138, 'TB': 139, 'TEX': 140, 'TOR': 141, 'WSH': 120
    }

    if team_abbr not in TEAM_ID_MAP:
        raise ValueError(f"Unknown team abbreviation: {team_abbr}")

    team_id = TEAM_ID_MAP[team_abbr]

    end_date = datetime.strptime(as_of_date, "%Y-%m-%d") if as_of_date else datetime.today()
    start_date = end_date - timedelta(days=days_back)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    url = f"https://statsapi.mlb.com/api/v1/transactions?teamId={team_id}&startDate={start_str}&endDate={end_str}"
    data = cached_json_request(url, namespace="statsapi")

    seen_ids = set()
    transactions = []

    for tx in data.get("transactions", []):
        tx_id = tx.get("id")
        if tx_id in seen_ids:
            continue
        seen_ids.add(tx_id)

        date = tx.get("date", tx.get("transactionDate", ""))
        type_desc = tx.get("typeDesc", tx.get("typeCode", ""))
        desc = tx.get("description", "")
        player = tx.get("person", {}).get("fullName", "N/A")

        transactions.append({
            "date": date,
            "player": player,
            "type": type_desc,
            "description": desc
        })

    return transactions


def format_transactions_for_report(transactions):
    categorized = {
        "Injuries": [],
        "Call-ups": [],
        "Roster Moves": [],
        "Trades": []
    }

    for tx in transactions:
        date = datetime.strptime(tx["date"], "%Y-%m-%d").strftime("%m/%d")
        player = tx["player"]
        desc = tx["description"].lower()

        # Categorize
        if "injured list" in desc:
            retro = ""
            if "retroactive to" in desc:
                try:
                    retro_date = desc.split("retroactive to")[-1].strip().rstrip(".")
                    retro = f" (retroactive to {retro_date})"
                except:
                    pass
            il_type = "15-day" if "15-day" in desc else "10-day"
            categorized["Injuries"].append(f"{date} – {player} placed on {il_type} IL{retro}")

        elif "selected the contract" in desc:
            team = desc.split("from")[-1].strip().rstrip(".") if "from" in desc else "Triple-A"
            categorized["Call-ups"].append(f"{date} – {player} contract selected from {team}")

        elif "optioned" in desc:
            team = desc.split("to")[-1].strip().rstrip(".") if "to" in desc else "Triple-A"
            categorized["Roster Moves"].append(f"{date} – {player} optioned to {team}")

        elif "designated for assignment" in desc:
            categorized["Roster Moves"].append(f"{date} – {player} designated for assignment")

        elif "roster status changed" in desc:
            categorized["Roster Moves"].append(f"{date} – {player} roster status changed")

        elif "traded" in desc:
            if "for" in desc:
                traded_for = desc.split("for")[-1].strip().rstrip(".")
                categorized["Trades"].append(f"{date} – {player} traded for {traded_for}")
            else:
                categorized["Trades"].append(f"{date} – {player} traded")

        else:
            # Default fallback
            categorized["Roster Moves"].append(f"{date} – {player}: {tx['type']}")

    # Combine output with headers + icons
    output_lines = []
    icons = {
        "Injuries": "🚑",
        "Call-ups": "🔼",
        "Roster Moves": "🔽",
        "Trades": "🔁"
    }

    for section in ["Injuries", "Call-ups", "Roster Moves", "Trades"]:
        entries = categorized[section]
        if entries:
            output_lines.append(f"{icons[section]} {section}:")
            output_lines.extend(entries)
            output_lines.append("")  # blank line between sections

    return output_lines


_TXN_POS_RE = re.compile(r'\b(RHP|LHP|INF|OF|IF|SS|1B|2B|3B|LF|CF|RF|DH|C|SP|RP|P)\b')


def _txn_level(desc_lower, prep):
    """Pull a readable minor-league level out of an MLB transaction description."""
    parts = re.split(rf"\b{re.escape(prep)}\b", desc_lower, maxsplit=1)
    if len(parts) < 2:
        return "the minors"
    tail = parts[-1].strip().rstrip(".")
    for token, short in [("triple-a", "AAA"), ("double-a", "AA"), ("high-a", "High-A"),
                         ("single-a", "A"), ("class a", "A"), ("rookie", "Rookie")]:
        if token in tail:
            return short
    return " ".join(tail.split()[:3]).title() or "the minors"


def _classify_transaction(desc, type_desc):
    """Turn a raw MLB transaction into (Move, Impact, roster_impact) columns.

    roster_impact is False for org-only noise (minor-league / draft signings) so the
    report can focus on moves that actually change the active/40-man roster.
    """
    d = str(desc).lower()
    if "injured list" in d or "injured-list" in d:
        il = next((lbl for lbl in ("60-day", "15-day", "10-day", "7-day") if lbl in d), "IL")
        retro = ""
        if "retroactive to" in d:
            retro_match = re.search(
                r"retroactive to\s+([a-z]+\s+\d{1,2}(?:,\s+\d{4})?)", d, flags=re.IGNORECASE
            )
            if retro_match:
                retro = f" (retro {retro_match.group(1).title()})"
        min_days = il.replace("-day", "")
        return f"To {il} IL{retro}", (f"OUT — min {min_days} days" if min_days.isdigit() else "OUT — on IL"), True
    if "rehab" in d:
        return "Rehab assignment", "Nearing return from IL", True
    if "reinstated" in d or "activated" in d:
        return "Reinstated from IL", "BACK — active roster", True
    if "recalled" in d:
        return f"Recalled from {_txn_level(d, 'from')}", "IN — added to active roster", True
    if "selected the contract" in d or ("selected" in d and "contract" in d):
        return f"Contract selected from {_txn_level(d, 'from')}", "IN — call-up (added to 40-man)", True
    if "optioned" in d:
        return f"Optioned to {_txn_level(d, 'to')}", "OUT — sent down (off active roster)", True
    if "designated" in d and "assignment" in d:
        return "Designated for assignment", "OFF 40-man (DFA)", True
    if "outright" in d:
        return f"Outrighted to {_txn_level(d, 'to')}", "OFF 40-man (kept in org)", True
    if "released" in d:
        return "Released", "GONE — no longer with org", True
    if "claimed off waivers" in d:
        return "Claimed off waivers", "IN — added to 40-man", True
    if "traded" in d:
        who = ""
        if " for " in d:
            who = " for " + d.split(" for ")[-1].strip().rstrip(".")
        return f"Traded{who}", "Roster change via trade", True
    if "paternity" in d:
        return "Paternity list", "OUT — short-term (1-3 days)", True
    if "bereavement" in d:
        return "Bereavement list", "OUT — short-term (3-7 days)", True
    if "signed" in d:
        # Overwhelmingly minor-league / draft signings; not active-roster impact.
        return "Signed", "Org depth (minor-league/draft)", False
    if "roster status" in d or "status change" in str(type_desc).lower():
        return "Roster status change", "Active-roster status changed", True
    # Fall back to the raw description (trimmed) so nothing is opaque.
    clean = re.sub(r'^\s*\w[\w\s]*?\b(placed|recalled|signed|optioned|selected|traded|activated|reinstated|claimed|released|designated|outrighted)\b',
                   r'\1', str(desc), flags=re.IGNORECASE).strip()
    return (clean[:48] or str(type_desc) or "Transaction"), "", True


def build_transaction_table(team_abbr, as_of_date, days_back=7, include_org_noise=False, max_rows=12):
    """Structured, plain-language transactions for the report.

    Columns: Date, Player, Pos, Move, Impact. Covers ALL players (not just relievers) —
    the old version only annotated bullpen arms, which was misleading. Minor-league /
    draft signings are filtered out by default so only active/40-man roster moves show.
    """
    cols = ["Date", "Player", "Pos", "Move", "Impact"]
    try:
        txns = get_recent_transactions(team_abbr, days_back=days_back, as_of_date=as_of_date)
    except Exception:
        return pd.DataFrame(columns=cols)
    rows = []
    seen = set()
    for tx in txns:
        desc = str(tx.get("description", ""))
        player = str(tx.get("player", ""))
        move, impact, roster_impact = _classify_transaction(desc, tx.get("type", ""))
        if not roster_impact and not include_org_noise:
            continue
        pos = ""
        head = desc.split(player)[0] if player and player in desc else desc
        m = _TXN_POS_RE.search(head)
        if m:
            pos = m.group(1)
        raw = str(tx.get("date", ""))
        try:
            date = datetime.strptime(raw, "%Y-%m-%d").strftime("%m/%d")
        except Exception:
            date = raw[5:] if len(raw) >= 5 else raw
        key = (date, player, move)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"Date": date, "Player": player, "Pos": pos, "Move": move, "Impact": impact})
    df = pd.DataFrame(rows, columns=cols)
    if max_rows and len(df) > max_rows:
        df = df.tail(max_rows).reset_index(drop=True)   # keep the most recent moves
    return df




# Mapping of full team names to abbreviations
team_name_to_abbr = {
    'Chicago White Sox': 'CWS',
    'Minnesota Twins': 'MIN',
    'Kansas City Royals': 'KC',
    'Milwaukee Brewers': 'MIL',
    'Boston Red Sox': 'BOS',
    'Baltimore Orioles': 'BAL',
    'Colorado Rockies': 'COL',
    'Philadelphia Phillies': 'PHI',
    'Texas Rangers': 'TEX',
    'Cincinnati Reds': 'CIN',
    'New York Mets': 'NYM',
    'Miami Marlins': 'MIA',
    'Pittsburgh Pirates': 'PIT',
    'Tampa Bay Rays': 'TB',
    'Washington Nationals': 'WSH',
    'Toronto Blue Jays': 'TOR',
    'Los Angeles Angels': 'LAA',
    'St. Louis Cardinals': 'STL',
    'San Francisco Giants': 'SF',
    'Houston Astros': 'HOU',
    'Cleveland Guardians': 'CLE',
    'San Diego Padres': 'SD',
    'Detroit Tigers': 'DET',
    'Seattle Mariners': 'SEA',
    'Chicago Cubs': 'CHC',
    'Atlanta Braves': 'ATL',
    'Los Angeles Dodgers': 'LAD',
    'Athletics': 'ATH',
    'Oakland Athletics': 'ATH',
    'Arizona Diamondbacks': 'ARI',
    'New York Yankees': 'NYY',
}

# Function to convert full team name to abbreviation
def get_team_abbreviation(team_name):
    # Handle case where the team name might have extra spaces
    team_name = team_name.strip()  # Strip leading/trailing spaces
    
    # Look for the abbreviation
    abbr = team_name_to_abbr.get(team_name)
    
    
    return abbr

# Function to allow the user to select a game from the list
def choose_game(date, game_number=None, away_team=None, home_team=None, dh_game=None):
    available_games = get_available_games(date)

    if not available_games:
        print("⚠️ No games available.")
        return None

    if away_team and home_team:
        requested_away = away_team.upper()
        requested_home = home_team.upper()
        matches = [
            game for game in available_games
            if get_team_abbreviation(game['home_team']) == requested_home
            and get_team_abbreviation(game['away_team']) == requested_away
        ]
        if not matches:
            print(f"⚠️ No {requested_away} at {requested_home} game found on {date}.")
            return None

        if len(matches) > 1:
            matches.sort(key=lambda g: g.get('game_num', 1))
            if dh_game is None:
                # Never guess on a doubleheader: the two games have different starters and
                # different lineups, so picking one silently is a wrong report, not a
                # slightly stale one.
                print(f"⚠️ {requested_away} at {requested_home} is a doubleheader on {date}. "
                      f"Choose with --dh-game:")
                for game in matches:
                    print(f"     --dh-game {game.get('game_num', 1)}  "
                          f"{game.get('game_datetime', '')}  gamePk {game['game_id']}")
                return None
            for game in matches:
                if int(game.get('game_num', 1)) == int(dh_game):
                    return game
            print(f"⚠️ No game {dh_game} for {requested_away} at {requested_home} on {date}.")
            return None
        return matches[0]

    if game_number is not None:
        if game_number < 1 or game_number > len(available_games):
            print("⚠️ Invalid game number.")
            return None
        return available_games[game_number - 1]

    # Display list of available games
    print(f"Games scheduled on {date}:")
    for game in available_games:
        home_team_abbr = get_team_abbreviation(game['home_team'])
        away_team_abbr = get_team_abbreviation(game['away_team'])
        if home_team_abbr and away_team_abbr:
            print(f"{game['game_number']}. {game['away_team']} at {game['home_team']}")
        else:
            print(f"⚠️ Missing abbreviation for {game['home_team']} or {game['away_team']}. Skipping.")
    
    # Get user selection
    try:
        game_number = int(input("Enter the number of the game you want to choose: "))
        
        if game_number < 1 or game_number > len(available_games):
            print("⚠️ Invalid selection.")
            return None
        
        selected_game = available_games[game_number - 1]
        return selected_game
    except ValueError:
        print("⚠️ Invalid input. Please enter a valid number.")
        return None




# Function to get the game details for a selected date
def get_game_info(date):
    schedule = statsapi.schedule(start_date=date, end_date=date)
    
    if not schedule:
        print(f"⚠️ No game scheduled on {date}")
        return None
    
    # Get the first game scheduled for the selected date (could be multiple)
    game = schedule[0]
    home_team = get_team_abbreviation(game['home_name'])
    away_team = get_team_abbreviation(game['away_name'])
    game_id = game['game_id']
    
    # Get boxscore and probable pitcher info
    boxscore = statsapi.boxscore_data(game_id)
    home_pitcher_id = boxscore['home']['probablePitcher']
    away_pitcher_id = boxscore['away']['probablePitcher']
    
    # Fetch pitcher names and IDs
    home_pitcher_name = statsapi.get(f"people/{home_pitcher_id}")['people'][0]['fullName']
    away_pitcher_name = statsapi.get(f"people/{away_pitcher_id}")['people'][0]['fullName']
    
    return {
        'home_team': home_team,
        'away_team': away_team,
        'home_pitcher': home_pitcher_name,
        'away_pitcher': away_pitcher_name,
    }

def format_last_10_games_for_df(last_10_df, abbreviation_map=None):
    formatted = []
    for _, row in last_10_df.iterrows():
        opp = row['Opponent']
        opp_abbr = abbreviation_map.get(opp, opp) if abbreviation_map else opp
        prefix = "vs" if row["Home/Away"] == "Home" else "@"
        formatted.append(f"{prefix} {opp_abbr} {row['Score']} {row['Result']}")
    return formatted


stat_keys = {
    "Record": ["Record", "record"],
    "Streak": ["Streak", "streak"],
    "Pythag %": ["Pythag %", "pythag"],
    "Run Diff": ["Run Diff", "run_diff"],
    "Division": ["Division", "division"]
}

def lookup_key(perf_dict, keys):
    for key in keys:
        if key in perf_dict:
            return perf_dict[key]
    return ""


def build_team_summary_side_by_side(home_team, home_perf, home_l10_df,
                                     away_team, away_perf, away_l10_df,
                                     abbreviation_map=None):
    # Ensure perf data is a dict
    if isinstance(home_perf, list):
        home_perf = dict(home_perf)
    if isinstance(away_perf, list):
        away_perf = dict(away_perf)

    # Define the stat keys in order (display label : actual dict key)
    stat_keys = [
        ("Record", "Overall Record"),
        ("Streak", "Current Streak"),
        ("Pythag %", "Pythagorean Expectation"),
        ("Run Diff", "Run Differential"),
        ("Division", "Division"),
        ("Div. Rank", "Division Rank"),
        ("GB", "Games Back"),
        ("Home Rec.", "Home Record"),
        ("Away Rec.", "Away Record"),
        ("Last 10", "Last 10 Games")
    ]

    # Format last 10 games
    home_l10 = format_last_10_games_for_df(home_l10_df, abbreviation_map)
    away_l10 = format_last_10_games_for_df(away_l10_df, abbreviation_map)

    # Pad L10 lists if shorter than 10 (to align properly)
    while len(home_l10) < 10:
        home_l10.append("")
    while len(away_l10) < 10:
        away_l10.append("")

    # Build rows
    rows = []
    for i, (label, key) in enumerate(stat_keys):
        home_val = home_perf.get(key, "—")
        away_val = away_perf.get(key, "—")
        home_game = home_l10[i]
        away_game = away_l10[i]
        rows.append([
            label,
            home_val,
            home_game,
            away_val,
            away_game
        ])

    return pd.DataFrame(rows, columns=["Stat", "Home Value", f"{home_team} L10", "Away Value", f"{away_team} L10"])


from pybaseball import statcast
from datetime import datetime

def load_statcast_splits(season=None, end_date=None):
    end_date = end_date or datetime.today().strftime("%Y-%m-%d")
    season = season or int(end_date[:4])
    cache_key = (int(season), str(end_date))
    if cache_key in _STATCAST_SPLITS_MEMORY_CACHE:
        return _STATCAST_SPLITS_MEMORY_CACHE[cache_key].copy()
    start_date = f"{season}-03-01"
    df = load_statcast_range(start_date, end_date)

    df = df[df['events'].notna()]
    df = df[df['p_throws'].isin(['R', 'L'])]

    df['hit'] = df['events'].isin(['single', 'double', 'triple', 'home_run'])
    df['pa'] = df['events'].isin([
        'strikeout', 'walk', 'hit_by_pitch', 'single', 'double', 'triple', 'home_run',
        'field_out', 'force_out', 'grounded_into_double_play', 'sac_fly', 'sac_bunt'
    ])

    splits = df[df['pa']].groupby(['batter', 'p_throws']).agg(
        PA=('pa', 'sum'),
        AB=('events', lambda x: (~x.isin(['walk', 'intent_walk', 'hit_by_pitch', 'sac_fly', 'sac_bunt'])).sum()),
        Hits=('hit', 'sum'),
        Doubles=('events', lambda x: (x == 'double').sum()),
        Triples=('events', lambda x: (x == 'triple').sum()),
        HR=('events', lambda x: (x == 'home_run').sum()),
        BB=('events', lambda x: (x == 'walk').sum()),
        K=('events', lambda x: (x == 'strikeout').sum())
    ).reset_index()

    from pybaseball import playerid_reverse_lookup

    # Unique batter IDs from statcast
    batter_ids = splits['batter'].unique()
    name_lookup = playerid_reverse_lookup(batter_ids, key_type='mlbam')
    name_lookup = name_lookup[['key_mlbam', 'name_first', 'name_last']]
    name_lookup['Name'] = name_lookup['name_first'] + ' ' + name_lookup['name_last']

    # Merge names into splits
    splits = splits.merge(name_lookup[['key_mlbam', 'Name']], left_on='batter', right_on='key_mlbam', how='left')
    splits = splits.drop(columns=['key_mlbam'])  # optional


    _STATCAST_SPLITS_MEMORY_CACHE[cache_key] = splits
    return splits.copy()



def _date_range(start_date, end_date):
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    while current <= end:
        yield current.strftime("%Y-%m-%d")
        current += timedelta(days=1)


def _record_components(perf):
    if not isinstance(perf, dict):
        return 0, 0
    text = str(perf.get("Overall Record", "0-0")).split()[0]
    try:
        wins, losses = text.split("-")
        return int(wins), int(losses)
    except (ValueError, TypeError):
        return 0, 0


def _record_pct(perf):
    wins, losses = _record_components(perf)
    total = wins + losses
    return wins / total if total else 0.5


def _last10_pct(perf):
    if not isinstance(perf, dict):
        return 0.5
    try:
        wins, losses = str(perf.get("Last 10 Games", "5-5")).split("-")
        total = int(wins) + int(losses)
        return int(wins) / total if total else 0.5
    except (ValueError, TypeError):
        return 0.5


def _run_diff_per_game(perf):
    wins, losses = _record_components(perf)
    total = wins + losses
    run_diff = _safe_number(perf.get("Run Differential") if isinstance(perf, dict) else 0, 0)
    return run_diff / total if total else 0.0


def _team_runs_per_game(team_id, season, as_of_date):
    url = f"https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season={season}&standingsTypes=regularSeason&date={as_of_date}"
    data = cached_json_request(url, namespace="statsapi")
    for division in data.get("records", []):
        for team in division.get("teamRecords", []):
            if team.get("team", {}).get("id") == team_id:
                wins = _stat_int(team, "wins")
                losses = _stat_int(team, "losses")
                games = max(1, wins + losses)
                return _safe_number(team.get("runsScored"), 0) / games, _safe_number(team.get("runsAllowed"), 0) / games
    return 4.4, 4.4


def _weather_calibration_features(environment):
    """Numeric pregame weather/umpire inputs shared by fitting and live prediction.

    Historical schedule weather is the observed first-pitch reading.  Live reports use
    MLB's posted reading when available and the Open-Meteo forecast otherwise.  Keeping
    the transformation here ensures the fitted coefficients see the same units at report
    time.  Missing readings are explicit rather than silently looking like perfect 72F,
    still-air weather.
    """
    environment = environment if isinstance(environment, dict) else {}
    weather = environment.get("weather") or {}
    forecast = environment.get("forecast") or {}

    temperature = _safe_number(weather.get("temp"), None)
    if temperature is None:
        temperature = _safe_number(forecast.get("temp_f"), None)

    wind_text = str(weather.get("wind") or forecast.get("wind") or "").strip()
    speed_match = re.match(r"\s*(\d+(?:\.\d+)?)", wind_text)
    wind_speed = float(speed_match.group(1)) if speed_match else 0.0
    wind_direction = wind_text.casefold()
    signed_wind = 0.0
    if "out to" in wind_direction:
        signed_wind = min(wind_speed, 25.0)
    elif "in from" in wind_direction:
        signed_wind = -min(wind_speed, 25.0)

    roof = str(
        forecast.get("roof")
        or environment.get("roof")
        or PARK_ROOF.get(environment.get("park_name"), "")
        or ""
    ).strip().casefold()
    roof_closed = float(roof in {"cover", "fixed", "closed", "dome"})
    weather_available = float(temperature is not None or bool(wind_text))
    if roof_closed:
        # Outdoor conditions must not move a game played in controlled air.
        temperature_delta = 0.0
        signed_wind = 0.0
    else:
        temperature_delta = ((temperature - 72.0) / 10.0) if temperature is not None else 0.0

    umpire = environment.get("ump_tendency") or {}
    umpire_games = int(_safe_number(umpire.get("Games"), 0) or 0)
    umpire_delta = _safe_number(umpire.get("vs Avg"), 0.0) or 0.0
    return {
        "temperature_f": float(temperature) if temperature is not None else 72.0,
        "temperature_delta_10f": float(temperature_delta),
        "wind_out_mph": float(signed_wind),
        "roof_closed": roof_closed,
        "weather_available": weather_available,
        "umpire_run_delta": float(umpire_delta),
        "umpire_available": float(umpire_games > 0),
        "umpire_games": umpire_games,
    }


def _assemble_calibration_features(home, away, environment):
    """Build model inputs from two leakage-free, pregame team snapshots."""
    environment = environment if isinstance(environment, dict) else {}
    park_runs = _safe_number((environment.get("park") or {}).get("Runs"), 1.0) or 1.0
    edge_diff = (
        1.8 * (home["record_pct"] - away["record_pct"])
        + 0.7 * (home["last10_pct"] - away["last10_pct"])
        + 0.03 * (home["run_diff_per_game"] - away["run_diff_per_game"])
        + 0.18
    )
    raw_home_runs = ((home["rpg"] + away["rapg"]) / 2) * park_runs
    raw_away_runs = ((away["rpg"] + home["rapg"]) / 2) * park_runs
    features = {
        "edge_diff": edge_diff,
        "raw_total": raw_home_runs + raw_away_runs,
        "raw_home_runs": raw_home_runs,
        "raw_away_runs": raw_away_runs,
        "home_rpg": home["rpg"],
        "away_rpg": away["rpg"],
        "home_rapg": home["rapg"],
        "away_rapg": away["rapg"],
        "park_runs": park_runs,
    }
    features.update(_weather_calibration_features(environment))
    return features


def _shrunk_team_calibration_snapshot(wins, losses, runs, allowed, recent_results):
    """Pregame team rates with league priors that fade as the season grows.

    Unshrunk Opening Week records created the largest 2026 win edges (up to 27
    probability points) without better accuracy.  Twenty neutral team-games keeps a 1-0
    record from looking decisive while contributing less than one-sixth of a full-season
    rate.  L10 gets a lighter five-game prior because it is intentionally more reactive.
    """
    games = max(0, int(wins) + int(losses))
    recent = [int(bool(value)) for value in recent_results]
    team_k = CALIBRATION_TEAM_PRIOR_GAMES
    recent_k = CALIBRATION_RECENT_PRIOR_GAMES
    return {
        "record_pct": (float(wins) + 0.5 * team_k) / (games + team_k),
        "last10_pct": (sum(recent) + 0.5 * recent_k) / (len(recent) + recent_k),
        "run_diff_per_game": (float(runs) - float(allowed)) / (games + team_k),
        "rpg": (float(runs) + CALIBRATION_RUNS_PRIOR * team_k) / (games + team_k),
        "rapg": (float(allowed) + CALIBRATION_RUNS_PRIOR * team_k) / (games + team_k),
    }


def _performance_calibration_snapshot(perf, rpg, rapg):
    wins, losses = _record_components(perf)
    games = wins + losses
    run_diff = _safe_number(
        perf.get("Run Differential") if isinstance(perf, dict) else 0, 0
    ) or 0
    try:
        recent_wins, recent_losses = str(
            perf.get("Last 10 Games", "") if isinstance(perf, dict) else ""
        ).split("-")
        recent = [1] * int(recent_wins) + [0] * int(recent_losses)
    except (ValueError, TypeError):
        recent = []
    observed_runs = float(rpg) * games if games else 0.0
    observed_allowed = float(rapg) * games if games else 0.0
    # Preserve the standings run differential when the endpoint omits cumulative runs.
    if games and observed_runs == 0.0 and observed_allowed == 0.0 and run_diff:
        observed_runs = CALIBRATION_RUNS_PRIOR * games + run_diff / 2
        observed_allowed = CALIBRATION_RUNS_PRIOR * games - run_diff / 2
    return _shrunk_team_calibration_snapshot(
        wins, losses, observed_runs, observed_allowed, recent
    )


def _calibration_features(home_team, away_team, season, as_of_date, environment):
    """Live/report-time feature builder using data available before first pitch."""
    home_id = get_team_id(home_team)
    away_id = get_team_id(away_team)
    home_perf = get_team_performance(home_id, season=season, as_of_date=as_of_date)
    away_perf = get_team_performance(away_id, season=season, as_of_date=as_of_date)
    home_rpg, home_rapg = _team_runs_per_game(home_id, season, as_of_date)
    away_rpg, away_rapg = _team_runs_per_game(away_id, season, as_of_date)
    home = _performance_calibration_snapshot(home_perf, home_rpg, home_rapg)
    away = _performance_calibration_snapshot(away_perf, away_rpg, away_rapg)
    return _assemble_calibration_features(home, away, environment)


def _month_ranges(start_date, end_date):
    cursor = datetime.strptime(start_date, "%Y-%m-%d")
    finish = datetime.strptime(end_date, "%Y-%m-%d")
    while cursor <= finish:
        next_month = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
        chunk_end = min(finish, next_month - timedelta(days=1))
        yield cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        cursor = next_month


def _calibration_schedule(start_date, end_date):
    """Fetch regular-season finals in monthly, disk-cached batches.

    The hydrated schedule already contains venue, first-pitch weather, officials and the
    final line.  That replaces the old per-day schedule plus several per-game requests.
    """
    url = "https://statsapi.mlb.com/api/v1/schedule"
    games = []
    for chunk_start, chunk_end in _month_ranges(start_date, end_date):
        data = cached_json_request(
            url,
            params={
                "sportId": 1,
                "startDate": chunk_start,
                "endDate": chunk_end,
                "gameType": "R",
                "hydrate": "venue,linescore,weather,officials",
            },
            namespace="calibration_schedule",
        )
        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                if str(game.get("gameType", "")).upper() != "R":
                    continue
                status = game.get("status", {}) or {}
                if str(status.get("abstractGameState", "")).casefold() != "final" and \
                        str(status.get("detailedState", "")).casefold() not in {"final", "game over"}:
                    continue
                teams = game.get("teams", {}) or {}
                home = teams.get("home", {}) or {}
                away = teams.get("away", {}) or {}
                if home.get("score") is None or away.get("score") is None:
                    continue
                hp_umpire = next(
                    (
                        (official.get("official") or {}).get("fullName")
                        for official in game.get("officials", [])
                        if official.get("officialType") == "Home Plate"
                    ),
                    "",
                )
                home_team = home.get("team", {}) or {}
                away_team = away.get("team", {}) or {}
                venue = game.get("venue", {}) or {}
                games.append({
                    "game_id": int(game["gamePk"]),
                    "date": str(game.get("officialDate") or date_entry.get("date")),
                    "game_datetime": str(game.get("gameDate") or ""),
                    "game_number": int(game.get("gameNumber", 1) or 1),
                    "game_type": "R",
                    "home_team": TEAM_ID_MAP.get(home_team.get("id")) or get_team_abbreviation(home_team.get("name", "")),
                    "away_team": TEAM_ID_MAP.get(away_team.get("id")) or get_team_abbreviation(away_team.get("name", "")),
                    "home_score": int(home["score"]),
                    "away_score": int(away["score"]),
                    "venue": str(venue.get("name") or ""),
                    "venue_id": venue.get("id"),
                    "weather": game.get("weather", {}) or {},
                    "hp_umpire": str(hp_umpire or ""),
                })
    ordered = sorted(games, key=lambda game: (
        game["date"], game.get("game_datetime", ""), game.get("game_number", 1), game["game_id"]
    ))
    # Suspended/resumed games can appear in two monthly responses with the same gamePk.
    # Keep the later, completed listing and never let one result train or validate twice.
    unique = {game["game_id"]: game for game in ordered}
    return sorted(unique.values(), key=lambda game: (
        game["date"], game.get("game_datetime", ""), game.get("game_number", 1), game["game_id"]
    ))


def _empty_team_history():
    return {"games": 0, "wins": 0, "runs": 0, "allowed": 0, "recent": deque(maxlen=10)}


def _team_history_snapshot(state):
    losses = state["games"] - state["wins"]
    return _shrunk_team_calibration_snapshot(
        state["wins"], losses, state["runs"], state["allowed"], list(state["recent"])
    )


def _update_team_history(state, runs, allowed):
    state["games"] += 1
    state["wins"] += int(runs > allowed)
    state["runs"] += runs
    state["allowed"] += allowed
    state["recent"].append(int(runs > allowed))


def _umpire_history_snapshot(umpire_state, league_state, umpire_name):
    umpire = umpire_state.get(umpire_name) if umpire_name else None
    if not umpire or not league_state["games"]:
        return None
    umpire_average = umpire["runs"] / umpire["games"]
    league_average = league_state["runs"] / league_state["games"]
    # Early-season umpire samples are volatile.  Shrink the observed difference toward
    # neutral while retaining Games so the artifact records how much evidence backed it.
    weight = umpire["games"] / (umpire["games"] + 10.0)
    return {
        "Games": umpire["games"],
        "R/G": round(umpire_average, 3),
        "Lg R/G": round(league_average, 3),
        "vs Avg": round((umpire_average - league_average) * weight, 4),
    }


def _load_calibration_dataset(path):
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        print(f"Ignoring unreadable calibration dataset {path}: {exc}")
        return pd.DataFrame()
    if "game_id" in frame:
        frame["game_id"] = pd.to_numeric(frame["game_id"], errors="coerce").astype("Int64")
        frame = frame.dropna(subset=["game_id"]).copy()
        frame["game_id"] = frame["game_id"].astype(int)
    return frame


def _save_calibration_dataset(frame, path):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    frame.sort_values(["date", "game_datetime", "game_id"]).to_csv(path, index=False)


def collect_calibration_dataset(start_date, end_date, max_games=0,
                                data_path=CALIBRATION_DATA_PATH, checkpoint_every=100):
    """Materialize reusable, leakage-free historical game features.

    Schedule context starts on March 1 of the first requested season so a midseason run
    still reconstructs season-to-date records and umpire priors correctly.  `max_games`
    limits the newest eligible games; zero means the complete date range.
    """
    context_start = f"{int(start_date[:4])}-03-01"
    schedule = _calibration_schedule(context_start, end_date)
    eligible = [game for game in schedule if start_date <= game["date"] <= end_date]
    if max_games and max_games > 0:
        eligible = eligible[-max_games:]
    selected_ids = {game["game_id"] for game in eligible}

    cached = _load_calibration_dataset(data_path)
    reusable = {}
    if not cached.empty and "feature_version" in cached:
        for _, row in cached[cached["feature_version"] == CALIBRATION_FEATURE_VERSION].iterrows():
            reusable[int(row["game_id"])] = row.to_dict()

    team_history = {}
    umpire_history = {}
    league_history = {}
    requested_rows = []
    new_rows = []
    for game in schedule:
        season = int(game["date"][:4])
        home_team, away_team = game["home_team"], game["away_team"]
        if not home_team or not away_team:
            continue
        home_state = team_history.setdefault((season, home_team), _empty_team_history())
        away_state = team_history.setdefault((season, away_team), _empty_team_history())
        season_umpires = umpire_history.setdefault(season, {})
        league_state = league_history.setdefault(season, {"games": 0, "runs": 0})
        total_runs = game["home_score"] + game["away_score"]

        if game["game_id"] in selected_ids:
            cached_row = reusable.get(game["game_id"])
            if cached_row is not None:
                requested_rows.append(cached_row)
            else:
                park, matched_park = _park_context_for_venue(game["venue"])
                ump_tendency = _umpire_history_snapshot(
                    season_umpires, league_state, game["hp_umpire"]
                )
                environment = {
                    "park": park,
                    "park_name": matched_park,
                    "roof": PARK_ROOF.get(matched_park, ""),
                    "weather": game["weather"],
                    "ump_tendency": ump_tendency,
                }
                features = _assemble_calibration_features(
                    _team_history_snapshot(home_state),
                    _team_history_snapshot(away_state),
                    environment,
                )
                row = {
                    "feature_version": CALIBRATION_FEATURE_VERSION,
                    "game_id": game["game_id"],
                    "date": game["date"],
                    "game_datetime": game["game_datetime"],
                    "season": season,
                    "game": f"{away_team}@{home_team}",
                    "away_team": away_team,
                    "home_team": home_team,
                    "venue": game["venue"],
                    "hp_umpire": game["hp_umpire"],
                    "weather_text": json.dumps(game["weather"], sort_keys=True),
                    "home_win": int(game["home_score"] > game["away_score"]),
                    "home_runs": game["home_score"],
                    "away_runs": game["away_score"],
                    "total_runs": total_runs,
                    **features,
                }
                requested_rows.append(row)
                new_rows.append(row)
                if data_path and checkpoint_every and len(new_rows) % checkpoint_every == 0:
                    combined = pd.concat([cached, pd.DataFrame(new_rows)], ignore_index=True)
                    combined = combined.drop_duplicates(subset=["game_id"], keep="last")
                    _save_calibration_dataset(combined, data_path)

        # Update only after feature construction: no game can inform its own prediction.
        _update_team_history(home_state, game["home_score"], game["away_score"])
        _update_team_history(away_state, game["away_score"], game["home_score"])
        if game["hp_umpire"]:
            umpire = season_umpires.setdefault(game["hp_umpire"], {"games": 0, "runs": 0})
            umpire["games"] += 1
            umpire["runs"] += total_runs
        league_state["games"] += 1
        league_state["runs"] += total_runs

    if new_rows and data_path:
        combined = pd.concat([cached, pd.DataFrame(new_rows)], ignore_index=True)
        combined = combined.drop_duplicates(subset=["game_id"], keep="last")
        _save_calibration_dataset(combined, data_path)
    return pd.DataFrame(requested_rows).sort_values(["date", "game", "game_id"]).reset_index(drop=True)


def _frame_model_predictions(model, frame):
    values = np.full(len(frame), float(model.get("intercept", 0.0)), dtype=float)
    for feature, coef in zip(model.get("features", []), model.get("coef", [])):
        column = pd.to_numeric(frame.get(feature, 0.0), errors="coerce")
        if not isinstance(column, pd.Series):
            column = pd.Series(float(column), index=frame.index)
        values += column.fillna(0.0).to_numpy() * float(coef)
    return values


def _magnitude_bucket_summary(magnitude, correct, metrics, buckets=5):
    """Equal-count signal buckets plus a continuous rank-trend test."""
    from scipy.stats import norm, spearmanr

    work = pd.DataFrame({
        "magnitude": pd.to_numeric(magnitude, errors="coerce"),
        "correct": pd.to_numeric(correct, errors="coerce"),
        **{name: pd.to_numeric(values, errors="coerce") for name, values in metrics.items()},
    }).dropna(subset=["magnitude", "correct"])
    if work.empty:
        return {"games": 0, "buckets": []}

    count = min(int(buckets), len(work))
    # Ranking guarantees balanced groups even when neutral/missing signals create ties.
    work["bucket"] = pd.qcut(
        work["magnitude"].rank(method="first"), count, labels=False, duplicates="drop"
    )
    rows = []
    for bucket, group in work.groupby("bucket", sort=True):
        n = len(group)
        successes = int(group["correct"].sum())
        rate = successes / n
        z = 1.96
        denominator = 1 + z * z / n
        center = (rate + z * z / (2 * n)) / denominator
        radius = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
        row = {
            "bucket": int(bucket) + 1,
            "games": int(n),
            "magnitude_min": round(float(group["magnitude"].min()), 4),
            "magnitude_max": round(float(group["magnitude"].max()), 4),
            "magnitude_mean": round(float(group["magnitude"].mean()), 4),
            "direction_accuracy_pct": round(rate * 100, 2),
            "accuracy_ci95_pct": [round((center - radius) * 100, 2), round((center + radius) * 100, 2)],
        }
        for name in metrics:
            row[name] = round(float(group[name].mean()), 4)
        rows.append(row)

    if work["magnitude"].nunique() < 2 or work["correct"].nunique() < 2:
        rho, trend_p = float("nan"), float("nan")
    else:
        rho, trend_p = spearmanr(work["magnitude"], work["correct"])
    first, last = rows[0], rows[-1]
    first_group = work[work["bucket"] == 0]["correct"]
    last_group = work[work["bucket"] == work["bucket"].max()]["correct"]
    p1, p2 = float(first_group.mean()), float(last_group.mean())
    pooled = float((first_group.sum() + last_group.sum()) / (len(first_group) + len(last_group)))
    standard_error = math.sqrt(
        pooled * (1 - pooled) * (1 / len(first_group) + 1 / len(last_group))
    ) if 0 < pooled < 1 else 0.0
    z_score = (p2 - p1) / standard_error if standard_error else 0.0
    top_bottom_p = float(2 * norm.sf(abs(z_score)))
    accuracies = [row["direction_accuracy_pct"] for row in rows]
    return {
        "games": int(len(work)),
        "spearman_rho": round(float(rho), 4) if np.isfinite(rho) else None,
        "spearman_p": round(float(trend_p), 6) if np.isfinite(trend_p) else None,
        "top_minus_bottom_accuracy_pp": round((p2 - p1) * 100, 2),
        "top_vs_bottom_p": round(top_bottom_p, 6),
        "monotonic_accuracy": all(b >= a for a, b in zip(accuracies, accuracies[1:])),
        "magnitude_validated": bool(
            np.isfinite(trend_p)
            and trend_p < 0.05
            and top_bottom_p < 0.05
            and p2 > p1
        ),
        "buckets": rows,
    }


def _holdout_signal_analysis(win_model, total_model, holdout_df):
    """Out-of-time evidence that larger model signals actually earn more trust."""
    if holdout_df is None or holdout_df.empty:
        return {}
    frame = holdout_df.copy()

    logits = _frame_model_predictions({
        "intercept": float(win_model.intercept_[0]),
        "features": ["edge_diff"],
        "coef": [float(win_model.coef_[0][0])],
    }, frame)
    home_probability = 1 / (1 + np.exp(-logits))
    favorite_correct = np.where(
        home_probability >= 0.5,
        frame["home_win"].to_numpy(),
        1 - frame["home_win"].to_numpy(),
    )
    win_magnitude = np.abs(home_probability - 0.5) * 100
    win = _magnitude_bucket_summary(
        win_magnitude,
        favorite_correct,
        {
            "predicted_confidence_pct": np.maximum(home_probability, 1 - home_probability) * 100,
            "brier": (home_probability - frame["home_win"].to_numpy()) ** 2,
        },
    )
    win["magnitude_unit"] = "probability points from 50%"

    total_spec = {
        "intercept": float(total_model.intercept_),
        "features": list(total_model.feature_names_in_),
        "coef": [float(value) for value in total_model.coef_],
    }
    predicted_total = _frame_model_predictions(total_spec, frame)
    actual_total = frame["total_runs"].to_numpy(dtype=float)
    raw_total = frame["raw_total"].to_numpy(dtype=float)
    adjustment = predicted_total - raw_total
    raw_error = np.abs(raw_total - actual_total)
    model_error = np.abs(predicted_total - actual_total)
    total = _magnitude_bucket_summary(
        np.abs(adjustment),
        (adjustment * (actual_total - raw_total) > 0).astype(int),
        {
            "raw_mae": raw_error,
            "model_mae": model_error,
            "mae_gain": raw_error - model_error,
            "model_beats_raw_pct": (model_error < raw_error).astype(float) * 100,
        },
    )
    total["magnitude_unit"] = "runs from raw total"

    coefficients = dict(zip(total_spec["features"], total_spec["coef"]))
    neutral_total = np.full(len(frame), total_spec["intercept"], dtype=float)
    neutral_total += raw_total * coefficients.get("raw_total", 0.0)
    context_delta = predicted_total - neutral_total
    neutral_error = np.abs(neutral_total - actual_total)
    context_error = np.abs(predicted_total - actual_total)
    context = _magnitude_bucket_summary(
        np.abs(context_delta),
        (context_delta * (actual_total - neutral_total) > 0).astype(int),
        {
            "neutral_mae": neutral_error,
            "context_mae": context_error,
            "mae_gain": neutral_error - context_error,
        },
    )
    context["magnitude_unit"] = "runs from weather/roof/umpire context"

    components = {}
    component_terms = {
        "weather": ["temperature_delta_10f", "wind_out_mph", "roof_closed"],
        # Availability is bookkeeping; only the measured pregame tendency is evidence
        # that a larger umpire signal deserves more weight.
        "umpire_tendency": ["umpire_run_delta"],
    }
    for name, feature_names in component_terms.items():
        delta = np.zeros(len(frame), dtype=float)
        for feature in feature_names:
            delta += frame[feature].fillna(0).to_numpy(dtype=float) * coefficients.get(feature, 0.0)
        nonzero = np.abs(delta) > 1e-12
        if not nonzero.any():
            components[name] = {"games": 0, "buckets": []}
            continue
        component_base = predicted_total - delta
        components[name] = _magnitude_bucket_summary(
            np.abs(delta[nonzero]),
            (delta[nonzero] * (actual_total[nonzero] - component_base[nonzero]) > 0).astype(int),
            {
                "base_mae": np.abs(component_base[nonzero] - actual_total[nonzero]),
                "component_mae": np.abs(predicted_total[nonzero] - actual_total[nonzero]),
                "mae_gain": (
                    np.abs(component_base[nonzero] - actual_total[nonzero])
                    - np.abs(predicted_total[nonzero] - actual_total[nonzero])
                ),
            },
            buckets=4,
        )
        components[name]["magnitude_unit"] = "runs"
    context["components"] = components
    return {"win": win, "total_adjustment": total, "context": context}


def calibrate_model(start_date=None, end_date=None, days=30, max_games=0,
                    output_path=CALIBRATION_PATH, data_path=CALIBRATION_DATA_PATH):
    end_date = end_date or (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    if not start_date:
        start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=days)).strftime("%Y-%m-%d")

    df = collect_calibration_dataset(
        start_date=start_date,
        end_date=end_date,
        max_games=max_games,
        data_path=data_path,
    )
    if len(df) < 12:
        raise RuntimeError(f"Not enough completed games to calibrate ({len(df)} found).")

    win_features = ["edge_diff"]
    context_features = [
        "temperature_delta_10f",
        "wind_out_mph",
        "roof_closed",
        "weather_available",
        "umpire_run_delta",
        "umpire_available",
    ]
    total_features = ["raw_total", *context_features]
    home_run_features = ["raw_home_runs", *context_features]
    away_run_features = ["raw_away_runs", *context_features]
    model_features = set(win_features + total_features + home_run_features + away_run_features)
    for column in model_features:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    df = df.sort_values(["date", "game"]).reset_index(drop=True)

    # With multiple seasons, the newest season is a clean out-of-time test.  A short
    # single-season run falls back to the previous chronological 75/25 split.
    seasons = pd.to_numeric(df.get("season", df["date"].str[:4]), errors="coerce")
    newest_season = int(seasons.max())
    season_holdout = df[seasons == newest_season]
    prior_seasons = df[seasons < newest_season]
    if len(season_holdout) >= 20 and len(prior_seasons) >= 20:
        train_df = prior_seasons.copy()
        holdout_df = season_holdout.copy()
        validation_method = "latest_season_holdout"
        holdout_label = str(newest_season)
    else:
        split_idx = max(12, int(len(df) * 0.75))
        if len(df) - split_idx < 8:
            split_idx = len(df)
        train_df = df.iloc[:split_idx].copy()
        holdout_df = df.iloc[split_idx:].copy()
        validation_method = "chronological_holdout" if not holdout_df.empty else "in_sample_only"
        holdout_label = (
            f"{holdout_df['date'].min()}..{holdout_df['date'].max()}"
            if not holdout_df.empty else ""
        )

    win_model = LogisticRegression(max_iter=1000)
    win_model.fit(train_df[win_features], train_df["home_win"])
    total_model = LinearRegression()
    total_model.fit(train_df[total_features], train_df["total_runs"])
    home_run_model = LinearRegression()
    home_run_model.fit(train_df[home_run_features], train_df["home_runs"])
    away_run_model = LinearRegression()
    away_run_model.fit(train_df[away_run_features], train_df["away_runs"])

    train_probs = win_model.predict_proba(train_df[win_features])[:, 1]
    brier = float(np.mean((train_probs - train_df["home_win"]) ** 2))
    total_mae = float(np.mean(np.abs(total_model.predict(train_df[total_features]) - train_df["total_runs"])))
    raw_total_mae = float(np.mean(np.abs(train_df["raw_total"] - train_df["total_runs"])))
    home_runs_mae = float(np.mean(np.abs(home_run_model.predict(train_df[home_run_features]) - train_df["home_runs"])))
    away_runs_mae = float(np.mean(np.abs(away_run_model.predict(train_df[away_run_features]) - train_df["away_runs"])))
    raw_home_runs_mae = float(np.mean(np.abs(train_df["raw_home_runs"] - train_df["home_runs"])))
    raw_away_runs_mae = float(np.mean(np.abs(train_df["raw_away_runs"] - train_df["away_runs"])))
    baseline_home_rate = float(train_df["home_win"].mean())
    baseline_brier = float(np.mean((baseline_home_rate - train_df["home_win"]) ** 2))

    validation = {
        "method": validation_method,
        "holdout_period": holdout_label,
        "train_games": int(len(train_df)),
        "holdout_games": int(len(holdout_df)),
        "baseline_brier": round(baseline_brier, 4),
        "model_brier": round(brier, 4),
        "raw_total_mae": round(raw_total_mae, 3),
        "model_total_mae": round(total_mae, 3),
        "raw_home_runs_mae": round(raw_home_runs_mae, 3),
        "model_home_runs_mae": round(home_runs_mae, 3),
        "raw_away_runs_mae": round(raw_away_runs_mae, 3),
        "model_away_runs_mae": round(away_runs_mae, 3),
    }
    if not holdout_df.empty:
        holdout_probs = win_model.predict_proba(holdout_df[win_features])[:, 1]
        holdout_brier = float(np.mean((holdout_probs - holdout_df["home_win"]) ** 2))
        holdout_baseline_brier = float(np.mean((baseline_home_rate - holdout_df["home_win"]) ** 2))
        holdout_total_pred = total_model.predict(holdout_df[total_features])
        holdout_total_mae = float(np.mean(np.abs(holdout_total_pred - holdout_df["total_runs"])))
        holdout_raw_total_mae = float(np.mean(np.abs(holdout_df["raw_total"] - holdout_df["total_runs"])))
        holdout_home_runs_mae = float(np.mean(np.abs(home_run_model.predict(holdout_df[home_run_features]) - holdout_df["home_runs"])))
        holdout_away_runs_mae = float(np.mean(np.abs(away_run_model.predict(holdout_df[away_run_features]) - holdout_df["away_runs"])))
        holdout_raw_home_runs_mae = float(np.mean(np.abs(holdout_df["raw_home_runs"] - holdout_df["home_runs"])))
        holdout_raw_away_runs_mae = float(np.mean(np.abs(holdout_df["raw_away_runs"] - holdout_df["away_runs"])))
        validation.update({
            "holdout_baseline_brier": round(holdout_baseline_brier, 4),
            "holdout_model_brier": round(holdout_brier, 4),
            "holdout_raw_total_mae": round(holdout_raw_total_mae, 3),
            "holdout_model_total_mae": round(holdout_total_mae, 3),
            "holdout_raw_home_runs_mae": round(holdout_raw_home_runs_mae, 3),
            "holdout_model_home_runs_mae": round(holdout_home_runs_mae, 3),
            "holdout_raw_away_runs_mae": round(holdout_raw_away_runs_mae, 3),
            "holdout_model_away_runs_mae": round(holdout_away_runs_mae, 3),
        })
        validation["signal_analysis"] = _holdout_signal_analysis(
            win_model, total_model, holdout_df
        )

    if validation.get("holdout_games", 0) >= 20:
        better_win = validation.get("holdout_model_brier", 1) <= validation.get("holdout_baseline_brier", 0) + 0.02
        better_total = validation.get("holdout_model_total_mae", 99) <= validation.get("holdout_raw_total_mae", 99) + 0.15
        reliability = "medium" if better_win and better_total else "low"
    else:
        reliability = "low"

    final_win_model = LogisticRegression(max_iter=1000)
    final_win_model.fit(df[win_features], df["home_win"])
    final_total_model = LinearRegression()
    final_total_model.fit(df[total_features], df["total_runs"])
    final_home_run_model = LinearRegression()
    final_home_run_model.fit(df[home_run_features], df["home_runs"])
    final_away_run_model = LinearRegression()
    final_away_run_model.fit(df[away_run_features], df["away_runs"])

    final_probs = final_win_model.predict_proba(df[win_features])[:, 1]
    final_brier = float(np.mean((final_probs - df["home_win"]) ** 2))
    final_total_mae = float(np.mean(np.abs(final_total_model.predict(df[total_features]) - df["total_runs"])))

    payload = {
        "created_at": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
        "feature_version": CALIBRATION_FEATURE_VERSION,
        "data_path": os.path.abspath(data_path) if data_path else None,
        "start_date": start_date,
        "end_date": end_date,
        "games": int(len(df)),
        "brier": round(final_brier, 4),
        "total_mae": round(final_total_mae, 3),
        "reliability": reliability,
        "validation": validation,
        "win_model": {
            "features": win_features,
            "intercept": float(final_win_model.intercept_[0]),
            "coef": [float(x) for x in final_win_model.coef_[0]],
        },
        "total_model": {
            "features": total_features,
            "intercept": float(final_total_model.intercept_),
            "coef": [float(x) for x in final_total_model.coef_],
        },
        "home_run_model": {
            "features": home_run_features,
            "intercept": float(final_home_run_model.intercept_),
            "coef": [float(x) for x in final_home_run_model.coef_],
        },
        "away_run_model": {
            "features": away_run_features,
            "intercept": float(final_away_run_model.intercept_),
            "coef": [float(x) for x in final_away_run_model.coef_],
        },
    }
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return payload


def _linear_model_predict(model, feature_values, fallback=0.0):
    if not model or not model.get("features"):
        return fallback
    value = model.get("intercept", fallback)
    for feature, coef in zip(model.get("features", []), model.get("coef", [])):
        value += feature_values.get(feature, 0.0) * coef
    return float(value)


def _calibrated_prediction_from_features(features, calibration=None):
    calibration = calibration or load_model_calibration() or {}
    win_model = calibration.get("win_model", {})
    total_model = calibration.get("total_model", {})
    home_run_model = calibration.get("home_run_model", {})
    away_run_model = calibration.get("away_run_model", {})

    z = _linear_model_predict(win_model, features, fallback=features.get("edge_diff", 0.0))
    home_win = 1 / (1 + np.exp(-z))
    raw_home = features.get("raw_home_runs", features.get("raw_total", 8.8) / 2)
    raw_away = features.get("raw_away_runs", features.get("raw_total", 8.8) / 2)
    home_runs = _linear_model_predict(home_run_model, features, fallback=raw_home)
    away_runs = _linear_model_predict(away_run_model, features, fallback=raw_away)
    total_runs = _linear_model_predict(total_model, features, fallback=home_runs + away_runs)
    if home_runs + away_runs > 0 and total_model:
        scale = max(0.75, min(1.25, total_runs / (home_runs + away_runs)))
        home_runs *= scale
        away_runs *= scale
    return {
        "home_win_prob": round(home_win, 4),
        "away_win_prob": round(1 - home_win, 4),
        "home_runs_pred": round(max(1.5, min(8.5, home_runs)), 2),
        "away_runs_pred": round(max(1.5, min(8.5, away_runs)), 2),
        "total_runs_pred": round(max(4.0, min(16.0, home_runs + away_runs)), 2),
    }


def backtest_model(start_date=None, end_date=None, days=30, max_games=120,
                   output_path="scouting_reports/model_backtest.csv",
                   data_path=CALIBRATION_DATA_PATH):
    end_date = end_date or (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    if not start_date:
        start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=days)).strftime("%Y-%m-%d")
    calibration = load_model_calibration()
    source = collect_calibration_dataset(
        start_date=start_date,
        end_date=end_date,
        max_games=max_games,
        data_path=data_path,
    )
    rows = []
    target_columns = {
        "feature_version", "game_id", "date", "game_datetime", "season", "game",
        "away_team", "home_team", "venue", "hp_umpire", "weather_text", "home_win",
        "home_runs", "away_runs", "total_runs",
    }
    for _, game in source.iterrows():
        features = {
            key: float(value)
            for key, value in game.items()
            if key not in target_columns and pd.notna(value) and isinstance(value, (int, float, np.number))
        }
        pred = _calibrated_prediction_from_features(features, calibration)
        home_team, away_team = game["home_team"], game["away_team"]
        home_score, away_score = int(game["home_runs"]), int(game["away_runs"])
        pick = home_team if pred["home_win_prob"] >= 0.5 else away_team
        actual_winner = home_team if home_score > away_score else away_team
        rows.append({
            "date": game["date"],
            "game": game["game"],
            "away_team": away_team,
            "home_team": home_team,
            "pick": pick,
            "actual_winner": actual_winner,
            "pick_correct": int(pick == actual_winner),
            "home_win_prob": pred["home_win_prob"],
            "away_win_prob": pred["away_win_prob"],
            "away_runs_pred": pred["away_runs_pred"],
            "home_runs_pred": pred["home_runs_pred"],
            "total_runs_pred": pred["total_runs_pred"],
            "away_score": away_score,
            "home_score": home_score,
            "total_runs": away_score + home_score,
            "away_run_error": round(pred["away_runs_pred"] - away_score, 2),
            "home_run_error": round(pred["home_runs_pred"] - home_score, 2),
            "total_error": round(pred["total_runs_pred"] - (away_score + home_score), 2),
            **{key: round(value, 4) for key, value in features.items()},
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No completed games found for backtest.")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    summary = {
        "games": int(len(df)),
        "accuracy": round(float(df["pick_correct"].mean()), 3),
        "total_mae": round(float(np.mean(np.abs(df["total_error"]))), 3),
        "home_runs_mae": round(float(np.mean(np.abs(df["home_run_error"]))), 3),
        "away_runs_mae": round(float(np.mean(np.abs(df["away_run_error"]))), 3),
        "output": output_path,
    }
    return summary


# Function to fetch starting pitcher and generate scouting report for both teams
# Function to generate scouting report for the selected game
def _doubleheader_sibling_starters(date, selected_game):
    """Starter ids already assigned to the other game of this doubleheader.

    Returns an empty list when the pairing plays once, so the normal path is unaffected.
    """
    if str(selected_game.get("doubleheader", "N")).upper() == "N":
        return []
    try:
        others = [
            game for game in get_available_games(date)
            if game.get("game_id") != selected_game.get("game_id")
            and game.get("home_team") == selected_game.get("home_team")
            and game.get("away_team") == selected_game.get("away_team")
        ]
    except Exception:
        return []
    starters = []
    for game in others:
        found = get_probable_pitchers_for_game(game["game_id"])
        starters.extend(pid for pid in found.values() if pid)
    return starters


def _resolve_pitcher_override(value, team_abbr):
    """Turn a --away-pitcher/--home-pitcher value into an MLBAM id.

    Accepts a raw id or a name. The team check is a warning rather than a hard failure:
    the override exists precisely for cases where the feeds are behind reality.
    """
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None

    pitcher_id = int(text) if text.isdigit() else lookup_player_id(remove_accents(text))
    if not pitcher_id:
        print(f"⚠️ Could not resolve pitcher override '{value}' for {team_abbr}.")
        return None
    if not pitcher_belongs_to_team(pitcher_id, team_abbr):
        print(f"⚠️ Override {_player_name_from_id(pitcher_id)} does not appear to be on {team_abbr}; using it anyway.")
    print(f"ℹ️ Using pitcher override for {team_abbr}: {_player_name_from_id(pitcher_id)} ({pitcher_id}).")
    return pitcher_id


def resolve_starter(selected_game, side, team_abbr, date, override=None,
                    exclude_ids=None, season=None, context_end=None,
                    allow_projected=True):
    """Who to build this side of the report against, and where the name came from.

    Ladder, most authoritative first:

        override    --away-pitcher / --home-pitcher
        announced   StatsAPI probable, then the validated probable-pitchers page
        salary      the arm DraftKings flagged as starting tonight
        rotation    `effective_starter` over `build_rotation`

    Returns `{"id", "name", "source", "provisional", "note"}`, with `id` None only when even
    the rotation model has nothing to offer -- which is the one case that still skips.

    `provisional` is True for anything below `announced`. That flag is the whole point: the
    report gets built and *labelled* instead of skipped, and `refresh_cached_lineups` knows
    to rebuild it once MLB actually posts a probable.

    `allow_projected=False` stops after the announced branch, which is the old behaviour --
    worth having for a date far enough out that a rotation guess is not worth the eight
    minutes a game costs.
    """
    exclude_ids = {int(p) for p in (exclude_ids or []) if p}
    season = int(season or str(date)[:4])
    context_end = context_end or _pregame_end_date(date) or date

    def result(pitcher_id, source, name=None, note=""):
        pitcher_id = int(pitcher_id)
        return {"id": pitcher_id,
                "name": name or _player_name_from_id(pitcher_id),
                "source": source, "provisional": source not in ("announced", "override"),
                "note": note}

    forced = _resolve_pitcher_override(override, team_abbr)
    if forced:
        return result(forced, "override")

    announced = resolve_probable_pitcher_v2(selected_game, side, team_abbr, exclude_ids,
                                            date=date)
    if announced:
        return result(announced, "announced")

    if not allow_projected:
        return {"id": None, "name": "TBD", "source": "unknown", "provisional": True,
                "note": "no probable announced (projected starters disabled)"}

    from_dk = starter_from_salary_file(date, team_abbr, exclude_ids=exclude_ids)
    if from_dk:
        name = _player_name_from_id(from_dk)
        print(f"ℹ️ No probable posted for {team_abbr}; DraftKings has {name} starting.")
        return result(from_dk, "salary", name=name,
                      note="MLB has not posted a probable; DraftKings lists him as starting")

    pitcher_id, name, source, note = starter_from_rotation(
        team_abbr, season, context_end, as_of_date=date, exclude_ids=exclude_ids)
    if pitcher_id:
        print(f"ℹ️ No probable posted for {team_abbr}; rotation model projects {name}.")
        return result(pitcher_id, source or "rotation", name=name, note=note)

    return {"id": None, "name": "TBD", "source": "unknown", "provisional": True,
            "note": note or "no probable, no DK listing and no usable rotation history"}


def generate_scouting_report_for_game(
    date,
    game_number=None,
    away_team=None,
    home_team=None,
    output_format="markdown",
    include_plots=False,
    reports_root="scouting_reports",
    dated_output=True,
    similarity_lookback=3,
    away_pitcher_override=None,
    home_pitcher_override=None,
    dfs_highlight=True,
    dh_game=None,
    dfs_slate=None,
    allow_projected_starters=True,
):
    selected_game = choose_game(date, game_number=game_number, away_team=away_team,
                                home_team=home_team, dh_game=dh_game)
    
    if not selected_game:
        print("⚠️ No game selected. Exiting.")
        return
    
    home_team = get_team_abbreviation(selected_game['home_team'])
    away_team = get_team_abbreviation(selected_game['away_team'])
    game_id = selected_game['game_id']
    season = int(date[:4])
    context_date = _pregame_end_date(date)
    
    home_team_id = get_team_id(home_team)
    away_team_id = get_team_id(away_team)

    # Get starting pitcher info for the selected game. An override lets a report be built
    # when MLB has not published a probable yet but the starter is known from elsewhere
    # (a DFS salary file, for instance, often confirms starters hours earlier).
    # On a doubleheader, collect the starters already committed to the sibling game so the
    # page-scrape fallback cannot hand back the same arm for both.
    sibling_starters = _doubleheader_sibling_starters(date, selected_game)

    home_start = resolve_starter(selected_game, "home", home_team, date,
                                 override=home_pitcher_override,
                                 exclude_ids=sibling_starters,
                                 season=season, context_end=context_date,
                                 allow_projected=allow_projected_starters)
    # The home pick joins the exclusion set so a provisional source cannot hand the same arm
    # to both clubs; the announced branch already guards this for itself.
    away_start = resolve_starter(selected_game, "away", away_team, date,
                                 override=away_pitcher_override,
                                 exclude_ids=list(sibling_starters) + [home_start["id"]],
                                 season=season, context_end=context_date,
                                 allow_projected=allow_projected_starters)
    home_pitcher, away_pitcher = home_start["id"], away_start["id"]

    if not home_pitcher or not away_pitcher:
        missing = ", ".join(team for team, pick in ((home_team, home_start), (away_team, away_start))
                            if not pick["id"])
        print(f"⚠️ No starter could be resolved for {missing} in {away_team} at {home_team} "
              f"on {date} — not even from the DK salary file or the rotation model.")
        return
    if int(home_pitcher) == int(away_pitcher):
        print(f"⚠️ Probable pitcher conflict for {away_team} at {home_team} on {date}: both sides resolved to {_player_name_from_id(home_pitcher)}.")
        return

    starter_provenance = {"home": home_start, "away": away_start}
    for team, pick in ((home_team, home_start), (away_team, away_start)):
        if pick["provisional"]:
            print(f"⚠️ {team}'s starter is PROVISIONAL: {pick['name']} "
                  f"({STARTER_SOURCE_LABEL.get(pick['source'], pick['source'])}). "
                  f"Re-run with --refresh-lineups once the probable posts.")
    
    
    print(f"\nGenerating report for {home_team} vs {away_team} ({date})...\n")
    
    # Fetch stats for both home and away pitchers
    arsenal_start = _season_start_date(season)
    arsenal_end = context_date or date
    home_pitcher_arsenal = generate_starter_arsenal(pitcher_id=home_pitcher, start_date=arsenal_start, end_date=arsenal_end, save_dir="plots")
    away_pitcher_arsenal = generate_starter_arsenal(pitcher_id=away_pitcher, start_date=arsenal_start, end_date=arsenal_end, save_dir="plots")

    home_pitcher_stats = get_today_starting_pitcher(home_pitcher, home_team, date, season=season)
    away_pitcher_stats = get_today_starting_pitcher(away_pitcher, away_team, date, season=season)

    # Generate reports for both teams
    home_bullpen_df = generate_bullpen_summary(home_team, year=season, as_of_date=date)
    away_bullpen_df = generate_bullpen_summary(away_team, year=season, as_of_date=date)

    statcast_detail_df = load_statcast_detail(season=season, end_date=arsenal_end)
    statcast_similarity_df = load_statcast_detail(season=season, end_date=arsenal_end, lookback_seasons=similarity_lookback)
    statcast_splits_df = load_statcast_splits(season=season, end_date=arsenal_end)
    home_hotcold_df, home_lineup_df, home_splits_df = generate_team_hitter_report(home_team, away_pitcher, statcast_splits_df, game_date=date)
    away_hotcold_df, away_lineup_df, away_splits_df = generate_team_hitter_report(away_team, home_pitcher, statcast_splits_df, game_date=date)

    # Team performance
    home_team_performance = get_team_performance(home_team_id, season=season, as_of_date=arsenal_end)
    away_team_performance = get_team_performance(away_team_id, season=season, as_of_date=arsenal_end)

    home_last_10 = get_last_10_games_statsapi(home_team_id, as_of_date=arsenal_end)
    away_last_10 = get_last_10_games_statsapi(away_team_id, as_of_date=arsenal_end)


    if include_plots:
        home_pitches = fetch_pitch_data_for_starter(home_pitcher, year=season)
        generate_heatmaps_for_pitches(home_pitches, home_pitcher)

        away_pitches = fetch_pitch_data_for_starter(away_pitcher, year=season)
        generate_heatmaps_for_pitches(away_pitches, away_pitcher)



        # Generate baserunning and defensive leaderboards
    home_baserunning_df = generate_baserunning_leaderboard(home_team, year=season, as_of_date=date)
    away_baserunning_df = generate_baserunning_leaderboard(away_team, year=season, as_of_date=date)

    home_defense_df = generate_defensive_leaderboard_fangraphs(home_team, year=season, as_of_date=date)
    away_defense_df = generate_defensive_leaderboard_fangraphs(away_team, year=season, as_of_date=date)

    home_bullpen_df = enhance_bullpen_analysis(home_bullpen_df, home_team, season, date)
    away_bullpen_df = enhance_bullpen_analysis(away_bullpen_df, away_team, season, date)

    advanced_context = build_advanced_context(
        game_id, date, season, home_team, away_team,
        home_pitcher, away_pitcher,
        home_pitcher_stats, away_pitcher_stats,
        home_pitcher_arsenal, away_pitcher_arsenal,
        home_lineup_df, away_lineup_df,
        home_bullpen_df, away_bullpen_df,
        home_baserunning_df, away_baserunning_df,
        home_defense_df, away_defense_df,
        home_splits_df, away_splits_df,
        statcast_detail_df,
        statcast_similarity_df,
        dh_game=selected_game.get("game_num"),
        starter_provenance=starter_provenance,
    )

    outputs = []
    report_args = (
            home_lineup_df, home_bullpen_df, home_pitcher_stats, home_hotcold_df, home_pitcher_arsenal, home_pitcher, home_team_performance,
            away_lineup_df, away_bullpen_df, away_pitcher_stats, away_hotcold_df, away_pitcher_arsenal, away_pitcher, away_team_performance,
            date, home_team, away_team, home_team_id, away_team_id,
            home_baserunning_df, away_baserunning_df,
            home_defense_df, away_defense_df, home_last_10, away_last_10,
            home_splits_df, away_splits_df
        )

    # Snapshot the fully-assembled report inputs so the layout can be re-rendered
    # instantly (no re-pulling data) via render_report_from_cache / --from-cache.
    save_report_data_cache(date, away_team, home_team, report_args, advanced_context,
                           dh_game=selected_game.get("game_num"))

    if output_format in {"markdown", "both"}:
        outputs.append(generate_report_markdown(*report_args, output_dir=report_output_dir(date, reports_root, dated_output), advanced_context=advanced_context))

    if output_format in {"html"}:
        outputs.append(generate_report_html(*report_args, output_dir=report_output_dir(date, reports_root, dated_output), advanced_context=advanced_context))

    if output_format in {"pdf", "both"}:
        outputs.append(generate_comparison_report_with_fpdf(*report_args, output_dir=report_output_dir(date, reports_root, dated_output), advanced_context=advanced_context, primary=True))

    if output_format == "comparison-pdf":
        outputs.append(generate_comparison_report_with_fpdf(*report_args, output_dir=report_output_dir(date, reports_root, dated_output), advanced_context=advanced_context))

    if output_format == "xlsx":
        activate_dfs_highlight(date, enabled=dfs_highlight, slate=dfs_slate,
                               games=[f"{away_team}@{home_team}"])
        outputs.append(generate_comparison_report_excel(
            *report_args, output_dir=report_output_dir(date, reports_root, dated_output),
            advanced_context=advanced_context, dh_game=selected_game.get("game_num")))

    if output_format == "legacy-pdf":
        outputs.append(generate_report_with_fpdf(*report_args, output_dir=report_output_dir(date, reports_root, dated_output), advanced_context=advanced_context))

    for output in outputs:
        if output:
            print(f"Saved report: {output}")

    return outputs


def generate_all_scouting_reports_for_date(
    date,
    output_format="html",
    include_plots=False,
    skip_existing=False,
    reports_root="scouting_reports",
    dated_output=True,
    similarity_lookback=3,
    upcoming_only=False,
    dfs_slate=None,
    allow_projected_starters=True,
):
    games = get_available_games(date)
    if not games:
        print(f"No games available for {date}.")
        return []

    if upcoming_only:
        started = {"Final", "Game Over", "In Progress", "Completed Early", "Postponed", "Suspended", "Cancelled"}
        before = len(games)
        games = [g for g in games if str(g.get("status", "")).strip() not in started]
        print(f"--upcoming-only: {len(games)} of {before} games have not started yet.")
        if not games:
            print(f"No upcoming (not-yet-started) games remain for {date}.")
            return []

    output_dir = report_output_dir(date, reports_root, dated_output)
    os.makedirs(output_dir, exist_ok=True)
    season = int(date[:4])
    context_date = _pregame_end_date(date) or date
    print(f"Preparing Statcast context for {date} (similarity lookback: {similarity_lookback} season(s))...")
    load_statcast_detail(season=season, end_date=context_date)
    load_statcast_detail(season=season, end_date=context_date, lookback_seasons=similarity_lookback)
    load_statcast_splits(season=season, end_date=context_date)

    all_outputs = []
    failures = []
    print(f"Generating {len(games)} scheduled reports for {date}...")
    for game in games:
        away_abbr = get_team_abbreviation(game.get("away_team", ""))
        home_abbr = get_team_abbreviation(game.get("home_team", ""))
        label = f"{away_abbr or game.get('away_team')} at {home_abbr or game.get('home_team')}"
        if not away_abbr or not home_abbr:
            failures.append({"game": label, "reason": "team abbreviation unavailable"})
            print(f"Skipping {label}: team abbreviation unavailable.")
            continue
        # Both games of a doubleheader appear in `games`; each needs its own number, or
        # generation refuses (ambiguous) and the pairing is skipped entirely.
        dh_game = game.get("game_num", 1)
        if int(dh_game or 1) > 1:
            label += f" (game {dh_game})"
        if skip_existing:
            expected_exts = ["html"] if output_format == "html" else ["md"] if output_format == "markdown" else ["md", "html"] if output_format == "both" else ["xlsx"] if output_format == "xlsx" else ["pdf"]
            expected_paths = [
                os.path.join(output_dir,
                             f"scouting_report_{date}_{away_abbr}_{home_abbr}{_dh_suffix(dh_game)}.{ext}")
                for ext in expected_exts
            ]
            if all(os.path.exists(path) for path in expected_paths):
                print(f"Skipping {label}: existing report found.")
                continue
        try:
            outputs = generate_scouting_report_for_game(
                date,
                away_team=away_abbr,
                home_team=home_abbr,
                output_format=output_format,
                include_plots=include_plots,
                reports_root=reports_root,
                dated_output=dated_output,
                similarity_lookback=similarity_lookback,
                dh_game=dh_game,
                dfs_slate=dfs_slate,
                allow_projected_starters=allow_projected_starters,
            ) or []
            if outputs:
                all_outputs.extend(outputs)
            else:
                failures.append({"game": label, "reason": "report generation returned no output"})
        except Exception as exc:
            failures.append({"game": label, "reason": str(exc)})
            print(f"Skipping {label}: {exc}")

    if failures:
        failure_path = os.path.join(output_dir, f"scouting_report_failures_{date}.csv")
        os.makedirs(os.path.dirname(failure_path), exist_ok=True)
        pd.DataFrame(failures).to_csv(failure_path, index=False)
        print(f"Saved failures: {failure_path}")

    print(f"Generated {len(all_outputs)} files for {date}.")
    return all_outputs


def qc_generated_reports(date, reports_dir=None, reports_root="scouting_reports", dated_output=True):
    reports_dir = reports_dir or report_output_dir(date, reports_root, dated_output)
    if not os.path.isdir(reports_dir) and dated_output:
        legacy_dir = reports_root
        if os.path.isdir(legacy_dir):
            reports_dir = legacy_dir
    rows = []
    prefix = f"scouting_report_{date}_"
    for filename in sorted(os.listdir(reports_dir)) if os.path.isdir(reports_dir) else []:
        if not filename.startswith(prefix) or not filename.endswith(".md"):
            continue
        path = os.path.join(reports_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        lines = text.splitlines()

        probable_line = next((line for line in lines if line.startswith("**Probable Starters:**")), "")
        game_edge_rows = 0
        if "### Game Edge" in lines:
            idx = lines.index("### Game Edge")
            section = lines[idx: next((i for i in range(idx + 1, len(lines)) if lines[i].startswith("### ") or lines[i].startswith("## ")), len(lines))]
            game_edge_rows = sum(1 for line in section if line.startswith("| ") and not line.startswith("| ---") and "Team |" not in line)

        type_rows = 0
        if "### Team Results Vs Pitcher Type" in lines:
            idx = lines.index("### Team Results Vs Pitcher Type")
            section = lines[idx: next((i for i in range(idx + 1, len(lines)) if lines[i].startswith("### ") or lines[i].startswith("## ")), len(lines))]
            type_rows = sum(1 for line in section if line.startswith("| ") and not line.startswith("| ---") and "Team |" not in line)

        factor_rows = 0
        if "### Why This Edge Exists" in lines:
            idx = lines.index("### Why This Edge Exists")
            section = lines[idx: next((i for i in range(idx + 1, len(lines)) if lines[i].startswith("### ") or lines[i].startswith("## ")), len(lines))]
            factor_rows = sum(1 for line in section if line.startswith("| ") and not line.startswith("| ---") and "Factor |" not in line)

        bvp_no_data = 0
        if "### Batter vs Pitcher History" in lines:
            idx = lines.index("### Batter vs Pitcher History")
            section = lines[idx: next((i for i in range(idx + 1, len(lines)) if lines[i].startswith("## ")), len(lines))]
            bvp_no_data = sum(1 for line in section if "_No data available._" in line)

        catcher_missing = False
        if "### Catcher Run Game" in lines:
            idx = lines.index("### Catcher Run Game")
            section = lines[idx: next((i for i in range(idx + 1, len(lines)) if lines[i].startswith("### ") or lines[i].startswith("## ")), len(lines))]
            catcher_missing = any("_No data available._" in line for line in section)
        else:
            catcher_missing = True

        range_scores = [int(match) for match in re.findall(r"(?:OF Range|IF Range)?\s*(?:Plus|Solid|Risk) (\d+)", text)]

        starter_arsenal_present = "### Starter Arsenals" in text
        issues = []
        if not probable_line or "Unknown" in probable_line:
            issues.append("probable starter missing/unknown")
        if game_edge_rows < 2:
            issues.append("game edge missing team row")
        if factor_rows < 5:
            issues.append("factor matrix missing/thin")
        if type_rows < 2:
            issues.append("pitcher type missing team row")
        if not starter_arsenal_present:
            issues.append("starter arsenals missing")
        if "park factor unavailable" in text:
            issues.append("park factor unresolved")
        if catcher_missing:
            issues.append("catcher run game missing")
        if bvp_no_data >= 2:
            issues.append("both BvP tables empty")
        if range_scores and len(range_scores) >= 4 and sum(1 for score in range_scores if score >= 80) / len(range_scores) >= 0.75:
            issues.append("range scores suspiciously inflated")
        if "<span style" in text:
            issues.append("raw html style leak")
        if "namefield" in text:
            issues.append("statsapi dict leak")
        if "g  PA" in text:
            issues.append("empty pitcher-type summary")

        rows.append({
            "Report": filename,
            "Status": "PASS" if not issues else "REVIEW",
            "Issues": "; ".join(issues),
            "No Data Count": text.count("_No data available._"),
            "Comp Fallback Count": text.count("Insufficient same-handed starter pitch volume"),
            "Probable Starters": probable_line.replace("**Probable Starters:**", "").strip(),
            "Game Edge Rows": game_edge_rows,
            "Factor Rows": factor_rows,
            "Pitcher Type Rows": type_rows,
            "Lines": len(lines),
        })

    qc_df = pd.DataFrame(rows)
    output_path = os.path.join(reports_dir, f"scouting_report_qc_{date}.csv")
    os.makedirs(reports_dir, exist_ok=True)
    qc_df.to_csv(output_path, index=False)
    return qc_df, output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate an MLB game scouting report.")
    parser.add_argument(
        "--date",
        default=datetime.today().strftime("%Y-%m-%d"),
        help="Game date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--game-number",
        type=int,
        help="1-based game number from the schedule on the selected date.",
    )
    parser.add_argument("--away-team", help="Away team abbreviation, for example CIN.")
    parser.add_argument("--home-team", help="Home team abbreviation, for example CHC.")
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Re-render an already-generated report from cached data instead of pulling fresh.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="With --from-cache, skip all API refreshes and render purely from cache (~1s).",
    )
    parser.add_argument(
        "--refresh-lineups",
        action="store_true",
        help="With --from-cache, re-pull lineups first so confirmed lineups actually appear. "
             "Checks the cards first and keeps the cached report when nothing has changed.",
    )
    parser.add_argument(
        "--force-lineup-refresh",
        action="store_true",
        help="With --refresh-lineups, rebuild even when the posted cards match the cache.",
    )
    parser.add_argument(
        "--strict-probables",
        action="store_true",
        help="Skip a game unless MLB has posted a probable, instead of falling back to the "
             "DK salary file and then the rotation model. The old behaviour; worth using for "
             "a date far enough out that a projected starter is not worth generating.",
    )
    parser.add_argument(
        "--no-dfs-highlight",
        action="store_true",
        help="Disable DFS value highlighting of player names in the Excel report.",
    )
    parser.add_argument(
        "--dfs-slate",
        help="Which DK export to highlight from, by filename substring (e.g. 'early'). "
             "Only needed when a date has several exports and the report's own game is "
             "priced on more than one of them.",
    )
    parser.add_argument(
        "--dh-game", type=int, choices=[1, 2],
        help="Which game of a doubleheader. Required when a pairing plays twice on --date.",
    )
    parser.add_argument(
        "--away-pitcher",
        help="Force the away starter (MLBAM id or name) when MLB has not posted a probable yet.",
    )
    parser.add_argument(
        "--home-pitcher",
        help="Force the home starter (MLBAM id or name) when MLB has not posted a probable yet.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "html", "pdf", "comparison-pdf", "xlsx", "both"],
        default="xlsx",
        help="Report output format. Excel (xlsx) single-sheet comparison is the default.",
    )
    parser.add_argument(
        "--include-plots",
        action="store_true",
        help="Generate heatmap images. Slower and mainly useful for PDF/visual review.",
    )
    parser.add_argument(
        "--output-dir",
        default="scouting_reports",
        help="Base directory for reports. By default, reports are written under OUTPUT_DIR/YYYY-MM-DD/.",
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="Write reports directly to --output-dir instead of a date-specific subfolder.",
    )
    parser.add_argument(
        "--similarity-lookback",
        type=int,
        default=3,
        help="Number of seasons to use for similar-pitcher and pitcher-type samples.",
    )
    parser.add_argument(
        "--all-games",
        action="store_true",
        help="Generate reports for every scheduled game on --date.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="With --all-games, skip games whose expected report file already exists.",
    )
    parser.add_argument(
        "--upcoming-only",
        action="store_true",
        help="With --all-games, only generate games that have not started yet (skip In Progress/Final).",
    )
    parser.add_argument(
        "--confirmed-lineups",
        "--lineup-check",
        action="store_true",
        help="List games on --date with both actual 9-man lineups posted. Does not generate reports.",
    )
    parser.add_argument(
        "--qc-reports",
        action="store_true",
        help="Scan generated Markdown reports for --date and write a QC CSV.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Backtest recent completed games and save model calibration coefficients.",
    )
    parser.add_argument("--calibration-start", help="Calibration start date in YYYY-MM-DD format.")
    parser.add_argument("--calibration-end", help="Calibration end date in YYYY-MM-DD format. Defaults to yesterday.")
    parser.add_argument(
        "--calibration-days",
        type=int,
        default=30,
        help="Days to use when --calibration-start is omitted.",
    )
    parser.add_argument(
        "--calibration-max-games",
        type=int,
        default=0,
        help="Maximum completed games to include, keeping the newest games; 0 uses the full date range.",
    )
    parser.add_argument(
        "--calibration-data",
        default=CALIBRATION_DATA_PATH,
        help="Reusable per-game calibration feature dataset.",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Write a per-game prediction backtest CSV using the current calibration artifact.",
    )
    parser.add_argument("--backtest-start", help="Backtest start date in YYYY-MM-DD format.")
    parser.add_argument("--backtest-end", help="Backtest end date in YYYY-MM-DD format. Defaults to yesterday.")
    parser.add_argument(
        "--backtest-days",
        type=int,
        default=30,
        help="Days to use when --backtest-start is omitted.",
    )
    parser.add_argument(
        "--backtest-max-games",
        type=int,
        default=120,
        help="Maximum completed games to include in the backtest CSV.",
    )
    parser.add_argument(
        "--backtest-output",
        default="scouting_reports/model_backtest.csv",
        help="CSV path for --backtest output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.calibrate:
        payload = calibrate_model(
            start_date=args.calibration_start,
            end_date=args.calibration_end,
            days=args.calibration_days,
            max_games=args.calibration_max_games,
            data_path=args.calibration_data,
        )
        validation = payload.get("validation", {})
        holdout = ""
        if validation.get("holdout_games"):
            holdout = (
                f", holdout Brier {validation.get('holdout_model_brier')}"
                f", holdout total MAE {validation.get('holdout_model_total_mae')}"
            )
        print(
            f"Saved calibration: {CALIBRATION_PATH} "
            f"({payload['games']} games, in-sample Brier {payload['brier']}, "
            f"in-sample total MAE {payload['total_mae']}{holdout}, "
            f"reliability {payload.get('reliability')})"
        )
        return

    if args.backtest:
        summary = backtest_model(
            start_date=args.backtest_start,
            end_date=args.backtest_end,
            days=args.backtest_days,
            max_games=args.backtest_max_games,
            output_path=args.backtest_output,
        )
        print(
            f"Saved backtest: {summary['output']} "
            f"({summary['games']} games, accuracy {summary['accuracy']}, total MAE {summary['total_mae']})"
        )
        return

    if args.confirmed_lineups:
        ready_df, status_df, paths = check_confirmed_lineups_for_date(
            args.date,
            reports_root=args.output_dir,
            dated_output=not args.flat_output,
        )
        print(f"Checked lineups for {args.date}: {len(ready_df)} of {len(status_df)} games have both confirmed lineups.")
        if paths:
            print(f"Saved lineup status: {paths.get('status')}")
            print(f"Saved ready games: {paths.get('ready')}")
            if paths.get("failures"):
                print(f"Saved lineup check failures: {paths.get('failures')}")
        if not ready_df.empty:
            print(ready_df[["Game", "Status", "First Pitch", "Away Lineup", "Home Lineup"]].to_string(index=False))
        return

    if args.qc_reports:
        qc_df, output_path = qc_generated_reports(
            args.date,
            reports_root=args.output_dir,
            dated_output=not args.flat_output,
        )
        failed = int((qc_df.get("Status", pd.Series(dtype=str)) != "PASS").sum()) if not qc_df.empty else 0
        print(f"Saved QC: {output_path} ({len(qc_df)} reports, {failed} need review)")
        if failed:
            print(qc_df[qc_df["Status"] != "PASS"].to_string(index=False))
        return

    if (args.away_team and not args.home_team) or (args.home_team and not args.away_team):
        raise SystemExit("Provide both --away-team and --home-team, or neither.")

    # Checked before --all-games: re-rendering a whole slate from cache is a different
    # operation from generating it, and must not fall through to a full regeneration.
    if args.from_cache:
        formats = ("xlsx",) if args.format in {"xlsx", "markdown"} else (args.format,)
        if args.format == "both":
            formats = ("pdf", "xlsx")
        if args.all_games:
            games = [
                (a, h) for _, a, h in
                __import__("dfs.slate", fromlist=["cached_games"]).cached_games(args.date)
            ]
        elif args.away_team and args.home_team:
            games = [(args.away_team, args.home_team)]
        else:
            raise SystemExit("--from-cache needs --away-team/--home-team or --all-games.")
        blocked = []
        for away, home in games:
            try:
                for path in render_report_from_cache(
                    args.date, away, home,
                    reports_root=args.output_dir, dated_output=not args.flat_output,
                    formats=formats, dfs_highlight=not args.no_dfs_highlight,
                    fast=args.fast, refresh_lineups=args.refresh_lineups,
                    force_lineup_refresh=args.force_lineup_refresh,
                    dfs_slate=args.dfs_slate,
                ):
                    print(f"  -> {path}")
            except PermissionError:
                # Almost always the workbook being open in Excel. One locked file must not
                # abandon the rest of the slate.
                blocked.append(f"{away}@{home}")
        if blocked:
            print(f"\n[!] Could not write {len(blocked)} report(s) — file open in Excel? "
                  f"{', '.join(blocked)}")
            print("    Close the workbook(s) and re-run; everything else was written.")
        return

    if args.all_games:
        generate_all_scouting_reports_for_date(
            args.date,
            output_format=args.format,
            include_plots=args.include_plots,
            skip_existing=args.skip_existing,
            reports_root=args.output_dir,
            dated_output=not args.flat_output,
            similarity_lookback=args.similarity_lookback,
            upcoming_only=args.upcoming_only,
            dfs_slate=args.dfs_slate,
            allow_projected_starters=not args.strict_probables,
        )
        return

    generate_scouting_report_for_game(
        args.date,
        game_number=args.game_number,
        away_team=args.away_team,
        home_team=args.home_team,
        output_format=args.format,
        include_plots=args.include_plots,
        reports_root=args.output_dir,
        dated_output=not args.flat_output,
        similarity_lookback=args.similarity_lookback,
        away_pitcher_override=args.away_pitcher,
        home_pitcher_override=args.home_pitcher,
        dfs_highlight=not args.no_dfs_highlight,
        dh_game=args.dh_game,
        dfs_slate=args.dfs_slate,
        allow_projected_starters=not args.strict_probables,
    )


if __name__ == "__main__":
    main()
