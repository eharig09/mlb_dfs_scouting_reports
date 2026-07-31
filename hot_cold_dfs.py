#%%

from pybaseball import statcast, playerid_reverse_lookup
import pandas as pd
from datetime import datetime, timedelta
import requests

# -----------------------------
# Step 1: Get Statcast Data
# -----------------------------
# Define date range (last 14 days, excluding today for stability)
end_date = datetime.today() - timedelta(days=2)
start_date = end_date - timedelta(days=14)
start_str = start_date.strftime('%Y-%m-%d')
end_str = end_date.strftime('%Y-%m-%d')

# Pull Statcast data
df = statcast(start_str, end_str)

# Filter to only batted balls with valid data
df = df[(df['launch_speed'].notna()) & (df['launch_angle'].notna())]

# -----------------------------
# Step 2: Tag Barrels & Hard Hits
# -----------------------------
df['barrel'] = ((df['launch_speed'] >= 98) & df['launch_angle'].between(26, 30)).astype(int)
df['hard_hit'] = (df['launch_speed'] >= 95).astype(int)

# -----------------------------
# Step 3: Aggregate by Batter
# -----------------------------
batter_summary = df.groupby('batter').agg(
    batted_balls=('events', 'count'),
    barrels=('barrel', 'sum'),
    hard_hits=('hard_hit', 'sum'),
    avg_launch_speed=('launch_speed', 'mean'),
    avg_launch_angle=('launch_angle', 'mean'),
    xwoba=('estimated_woba_using_speedangle', 'mean'),
    xba=('estimated_ba_using_speedangle', 'mean'),
    xslg=('estimated_slg_using_speedangle', 'mean'),
    woba_value=('woba_value', 'sum'),
    woba_denom=('woba_denom', 'sum')
).reset_index()

# Calculate rates
batter_summary['barrel_rate'] = batter_summary['barrels'] / batter_summary['batted_balls']
batter_summary['hard_hit_rate'] = batter_summary['hard_hits'] / batter_summary['batted_balls']
batter_summary['woba'] = batter_summary['woba_value'] / batter_summary['woba_denom']

# Minimum sample size filter
batter_summary = batter_summary[batter_summary['batted_balls'] >= 10]

# -----------------------------
# Step 4: Get Batter Names
# -----------------------------
batter_ids = batter_summary['batter'].unique()
name_lookup = playerid_reverse_lookup(batter_ids, key_type='mlbam')
batter_summary = pd.merge(
    batter_summary,
    name_lookup[['key_mlbam', 'name_first', 'name_last']],
    left_on='batter',
    right_on='key_mlbam',
    how='left'
)
batter_summary['player_name'] = batter_summary['name_first'] + ' ' + batter_summary['name_last']

# -----------------------------
# Step 5: Get Team & Position via MLB StatsAPI
# -----------------------------

def get_mlb_team_rosters():
    team_rosters = {}
    
    # Get all MLB team IDs
    teams_url = "https://statsapi.mlb.com/api/v1/teams?sportId=1"  # sportId=1 is MLB
    teams = requests.get(teams_url).json()['teams']
    
    for team in teams:
        team_id = team['id']
        team_name = team['name']
        roster_url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster"
        roster_res = requests.get(roster_url).json()
        
        for player in roster_res.get('roster', []):
            player_id = player['person']['id']
            team_rosters[player_id] = {
                'team': team_name,
                'position': player['position']['abbreviation']
            }
    
    return team_rosters


def get_player_info_from_rosters(batter_ids, team_rosters):
    data = []
    for pid in batter_ids:
        info = team_rosters.get(pid, {})
        data.append({
            'batter': pid,
            'team': info.get('team', None),
            'position': info.get('position', None)
        })
    return pd.DataFrame(data)


team_rosters = get_mlb_team_rosters()
player_info = get_player_info_from_rosters(batter_ids, team_rosters)
batter_summary = pd.merge(batter_summary, player_info, on='batter', how='left')


# -----------------------------
# Step 6: Filter for Hot Hitters
# -----------------------------
filtered_hot = batter_summary[
    (batter_summary['barrel_rate'] >= 0.10) & 
    (batter_summary['woba'] >= 0.350)
]
filtered_cold = batter_summary[
    (batter_summary['barrel_rate'] <= 0.10) & 
    (batter_summary['woba'] <= 0.350)
]

hot_hitters = filtered_hot.sort_values('xwoba', ascending=False)
cold_hitters = filtered_cold.sort_values('xwoba', ascending=True)

# -----------------------------
# Step 7: Output Results
# -----------------------------
hot_hitters = hot_hitters[['player_name', 'team', 'position', 'batted_balls', 'barrel_rate', 'hard_hit_rate', 'woba', 'xwoba', 'xba', 'xslg']].head(20).round(3)
cold_hitters = cold_hitters[['player_name', 'team', 'position', 'batted_balls', 'barrel_rate', 'hard_hit_rate', 'woba', 'xwoba', 'xba', 'xslg']].head(20).round(3)



# %%

from pybaseball import statcast_pitcher
from datetime import date
import pandas as pd
from scipy.stats import zscore
from bs4 import BeautifulSoup
import requests

# --- Team Maps & Accent Fix ---
TEAM_NAME_MAP = {
    "ARI": "Arizona Diamondbacks",
    "ATL": "Atlanta Braves",
    "OAK": "Athletics",
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
    158: "MIL", 142: "MIN", 121: "NYM", 147: "NYY", 133: "OAK",
    143: "PHI", 134: "PIT", 135: "SD", 136: "SEA", 137: "SF",
    138: "STL", 139: "TB", 140: "TEX", 141: "TOR", 120: "WSH"
}


# --- Get Probable Pitcher by Team Abbreviation ---
def get_all_probable_pitchers():
    url = "https://www.mlb.com/probable-pitchers"
    response = requests.get(url)
    if response.status_code != 200:
        print("❌ Failed to load MLB probable pitchers page.")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    game_blocks = soup.select("div.probable-pitchers__pitchers")

    pitchers = []

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

        if away_id and away_team_id:
            pitchers.append({
                'name': away_name,
                'player_id': int(away_id),
                'team_id': int(away_team_id),
                'team_abbr': TEAM_ID_MAP.get(int(away_team_id), 'UNK'),
                'side': 'away'
            })

        if home_id and home_team_id:
            pitchers.append({
                'name': home_name,
                'player_id': int(home_id),
                'team_id': int(home_team_id),
                'team_abbr': TEAM_ID_MAP.get(int(home_team_id), 'UNK'),
                'side': 'home'
            })

    return pitchers

# Traditional StatsAPI metrics
def get_pitcher_stats_mlb_statsapi(player_id, season=2025):
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=season&group=pitching&season={season}"
    response = requests.get(url)
    data = response.json()
    try:
        stats = data["stats"][0]["splits"][0]["stat"]
        return {
            "IP": float(stats.get("inningsPitched", 0)),
            "K": int(stats.get("strikeOuts", 0)),
            "ERA": float(stats.get("era", 0)),
            "AVG": float(stats.get("avg", 0)),
            "OBP": float(stats.get("obp", 0)),
            "SLG": float(stats.get("slg", 0)),
            "OPS": float(stats.get("ops", 0)),
            "BF": int(stats.get("battersFaced", 0))
        }
    except:
        return {}

# Advanced Statcast metrics
def get_statcast_metrics(player_id):
    try:
        data = statcast_pitcher('2025-03-27', date.today().strftime('%Y-%m-%d'), player_id)
        if data.empty:
            return {}

        # Calculate Hard Hit percentage on a pitch-by-pitch basis
        data["HardHit"] = data["launch_speed"] > 95
        woba = data["estimated_woba_using_speedangle"].mean()
        hard_hit_rate = data["HardHit"].mean()

        # Group by at-bat to get the final pitch (the one that ended the at-bat)
        last_pitches = data.sort_values(["game_date", "at_bat_number", "pitch_number"]) \
                           .groupby(["game_date", "at_bat_number"]).tail(1)
        
        # Correctly calculate K% using the outcomes from the final pitch of each at-bat
        strikeouts = (last_pitches["events"] == "strikeout").sum()
        total_at_bats = len(last_pitches)
        k_rate = strikeouts / total_at_bats if total_at_bats > 0 else 0

        # Calculate innings pitched (IP) based on final pitch outcomes
        outs = last_pitches["events"].isin([
            "strikeout", "groundout", "flyout", "lineout", "pop out",
            "double play", "triple play", "fielders choice out", "sac fly", "sac bunt"
        ]).sum()
        ip = outs / 3

        return {
            "Statcast_wOBA": woba,
            "HardHit%": hard_hit_rate,
            "K%": k_rate,
            "Statcast_IP": ip
        }
    except Exception as e:
        print(f"Error: {e}")
        return {}


# Main runner
def build_pitcher_report():
    pitchers = get_all_probable_pitchers()
    report = []

    for p in pitchers:
        traditional = get_pitcher_stats_mlb_statsapi(p['player_id'])
        advanced = get_statcast_metrics(p['player_id'])
        if traditional or advanced:
            row = {
                "name": p['name'],
                "team": p['team_abbr'],
                **traditional,
                **advanced
            }
            report.append(row)

    df = pd.DataFrame(report)

    # Optional: score hot/cold
    for col in ["Statcast_wOBA", "HardHit%", "K%", "Statcast_IP"]:
        if col in df:
            df[col + "_z"] = zscore(df[col].fillna(df[col].mean()))

    if "Statcast_wOBA_z" in df:
        df["hot_score"] = -df["Statcast_wOBA_z"] - df["HardHit%_z"] + df["K%_z"] + df["Statcast_IP_z"]
        df = df.sort_values("hot_score", ascending=False)
        df["status"] = "neutral"
        df.loc[df.head(5).index, "status"] = "hot"
        df.loc[df.tail(5).index, "status"] = "cold"

    df.to_csv("pitcher_report.csv", index=False)
    print("✅ Report saved as pitcher_report.csv")
    return df

if __name__ == "__main__":
    pitcher_report = build_pitcher_report()
    hot_pitchers = pitcher_report[pitcher_report['status'] == 'hot']
    cold_pitchers = pitcher_report[pitcher_report['status'] == 'cold']



# %%

from datetime import datetime
import pandas as pd
import re
from datetime import datetime, timedelta
import statsapi
import unicodedata
from pybaseball import playerid_lookup, statcast_pitcher, statcast_batter
import requests
from functools import lru_cache
from bs4 import BeautifulSoup

@lru_cache(maxsize=None)
def get_team_id(team_abbr):
    team_map = {
        "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111,
        "CHC": 112, "CIN": 113, "CLE": 114, "COL": 115,
        "DET": 116, "HOU": 117, "KC": 118, "LAA": 108,
        "LAD": 119, "MIA": 146, "MIL": 158, "MIN": 142,
        "NYY": 147, "NYM": 121, "OAK": 133, "PHI": 143,
        "PIT": 134, "SD": 135, "SEA": 136, "SF": 137,
        "STL": 138, "TB": 139, "TEX": 140, "TOR": 141,
        "WSH": 120, "CHW": 145
    }
    team_abbr = team_abbr.upper()
    if team_abbr in team_map:
        return team_map[team_abbr]
    raise ValueError(f"Team abbreviation '{team_abbr}' not found.")


def lookup_player_id(name):
    try:
        first, last = name.split(" ", 1)
        result = playerid_lookup(last, first)
        if not result.empty:
            return int(result["key_mlbam"].iloc[0])
    except:
        pass
    return None

def remove_accents(input_str):
  nfkd_form = unicodedata.normalize('NFKD', input_str)
  return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def attach_opponents(pitchers):
    # Create dictionaries keyed by side
    home_pitchers = [p for p in pitchers if p['side'] == 'home']
    away_pitchers = [p for p in pitchers if p['side'] == 'away']

    # Assume games are in order: each away matches with next home
    matchups = []
    for away, home in zip(away_pitchers, home_pitchers):
        # Assign opponent team to each
        away['opponent_team_abbr'] = home['team_abbr']
        home['opponent_team_abbr'] = away['team_abbr']
        matchups.extend([away, home])

    return matchups

@lru_cache(maxsize=None)
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
        df = statcast_pitcher(start_date, end_date, pitcher_id)
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

@lru_cache(maxsize=None)
def get_latest_lineup_names(team_abbr, max_days_back=5):
    lineup_map = {'CWS': 'CHW'}
    if team_abbr in lineup_map:
        team_abbr = lineup_map[team_abbr]
    team_id = get_team_id(team_abbr)
    today = datetime.today().date()

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

        seen_positions = set()
        lineup = []

        for player in players.values():
            name = remove_accents(player['person']['fullName'])
            batting_order = player.get('battingOrder')

            if not batting_order:
                continue

            try:
                order_val = int(batting_order)
            except:
                continue

            if 100 <= order_val <= 900:
                pos_abbr = player.get('position', {}).get('abbreviation')
                if pos_abbr == 'PH' or pos_abbr is None:
                    continue
                if pos_abbr in seen_positions:
                    continue

                seen_positions.add(pos_abbr)
                lineup.append((order_val, name))

        if lineup:
            lineup.sort(key=lambda x: x[0])
            return [name for _, name in lineup]

    return []


def parse_matchup_summary(summary):
    try:
        avg_match = re.search(r"\(\.(\d+)\)", summary)
        ev_match = re.search(r"EV: ([\d\.]+)", summary)
        xba_match = re.search(r"xBA: ([\d\.]+)", summary)

        return {
            "avg": float("0." + avg_match.group(1)) if avg_match else 0,
            "ev": float(ev_match.group(1)) if ev_match else 0,
            "xba": float(xba_match.group(1)) if xba_match else 0,
            "summary": summary
        }
    except:
        return None

@lru_cache(maxsize=None)
def get_today_matchup_summaries(min_pa=5):
    probable_pitchers = attach_opponents(get_all_probable_pitchers())
    matchup_data = []

    for pitcher in probable_pitchers:
        pitcher_id = pitcher['player_id']
        pitcher_name = pitcher['name']
        team_abbr = pitcher['team_abbr']

        print(f"📋 Checking lineup for {team_abbr} starter {pitcher_name}")
        opponent_abbr = pitcher.get("opponent_team_abbr")
        lineup = get_latest_lineup_names(opponent_abbr)

        print(f"🧢 Lineup: {lineup}")

        if not lineup:
            print(f"⚠️ No recent lineup found for {team_abbr}")
            continue

        for batter in lineup:
            print(f"🔍 Analyzing {batter} vs {pitcher_name}")
            summary = get_matchup_summary(batter, pitcher_id, min_pa=min_pa)
            print(f"📊 Result: {summary}")

            if "No recent data" in summary or "Player ID not found" in summary or "error" in summary:
                continue

            parsed = parse_matchup_summary(summary)
            if parsed:
                matchup_data.append({
                    "batter": batter,
                    "pitcher": pitcher_name,
                    "team": team_abbr,
                    **parsed
                })

    df = pd.DataFrame(matchup_data)
    if df.empty:
        return "⚠️ No qualifying matchups found with at least {min_pa} PAs."

    # Score matchups by combining AVG, xBA and EV
    df["score"] = df["avg"] * 1000 + df["xba"] * 1000 + df["ev"]

    hot = df.sort_values("score", ascending=False).head(10)
    cold = df.sort_values("score").head(10)

    print("\n🔥 HOT MATCHUPS 🔥")
    for i, row in hot.iterrows():
        print(f"{row['batter']} vs {row['pitcher']} ({row['team']}): {row['summary']}")

    print("\n❄️ COLD MATCHUPS ❄️")
    for i, row in cold.iterrows():
        print(f"{row['batter']} vs {row['pitcher']} ({row['team']}): {row['summary']}")

    return hot, cold


if __name__ == "__main__":
    hot_matchups, cold_matchups = get_today_matchup_summaries()


# %%

from pybaseball import statcast_pitcher

def get_pitcher_vs_lineup_summary(pitcher_id, batter_names, start_date="2022-01-01", end_date=None):
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    batter_ids = [lookup_player_id(name) for name in batter_names if lookup_player_id(name)]

    if not batter_ids:
        return None

    try:
        df = statcast_pitcher(start_date, end_date, pitcher_id)
        df = df[df["batter"].isin(batter_ids)]

        if df.empty:
            return None

        # Final pitch per AB
        ab_df = df.sort_values(["game_date", "at_bat_number", "pitch_number"]) \
                  .groupby(["game_date", "at_bat_number"]).tail(1)

        ab_df["is_strikeout"] = ab_df["events"] == "strikeout"
        ab_df["woba"] = ab_df["estimated_woba_using_speedangle"]

        # Batters faced (plate appearances)
        pa = len(ab_df)

        # At-bats (excluding BB, IBB, HBP, sac bunt/fly)
        non_ab_events = [
            "walk", "intent_walk", "hit_by_pitch", 
            "sac_bunt", "sac_fly", "catcher_interf"
        ]
        ab = ab_df[~ab_df["events"].isin(non_ab_events)].shape[0]

        k = ab_df["is_strikeout"].sum()
        k_rate = round(k / pa, 3) if pa else 0.0

        # HardHit & EV (all pitches)
        df["is_hard_hit"] = df["launch_speed"] > 95
        hard_hit_rate = round(df["is_hard_hit"].mean(), 3)
        avg_ev = round(df["launch_speed"].mean(), 1)

        return {
            "K%": k_rate,
            "wOBA": round(ab_df["woba"].mean(), 3),
            "HardHit%": hard_hit_rate,
            "Avg EV": avg_ev,
            "PA": pa,
            "AB": ab
        }

    except Exception as e:
        return {"error": str(e)}


pitchers = attach_opponents(get_all_probable_pitchers())
results = []

for pitcher in pitchers:
    opponent_abbr = pitcher["opponent_team_abbr"]
    lineup = get_latest_lineup_names(opponent_abbr)
    if not lineup:
        continue

    stats = get_pitcher_vs_lineup_summary(pitcher["player_id"], lineup)
    if stats:
        results.append({
            "pitcher": pitcher["name"],
            "team": pitcher["team_abbr"],
            "vs_team": opponent_abbr,
            **stats
        })

pitcher_v_lineup = pd.DataFrame(results)

# %%
