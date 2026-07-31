# daily_run.py

import datetime
from pybaseball import statcast
from models.stuff_model import calculate_stuff_plus
from utils.highlights import get_game_highlight_plays
import requests

def get_yesterdays_date():
    return (datetime.date.today() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')

def generate_savant_url(player_name, player_id):
    """
    Returns a Baseball Savant player URL.
    """
    slug_name = player_name.lower().replace(" ", "-")
    return f"https://baseballsavant.mlb.com/savant-player/{slug_name}-{player_id}"

def main():
    date = get_yesterdays_date()
    print(f"📅 Pulling Statcast data for: {date}")

    try:
        df = statcast(start_dt=date, end_dt=date)
        if df.empty:
            print("⚠️ No data found for this date.")
            return
    except Exception as e:
        print(f"❌ Error fetching Statcast data: {e}")
        return

    print(f"✅ Pulled {len(df)} pitches. Running Stuff+ model...")

    try:
        df = calculate_stuff_plus(df)
    except Exception as e:
        print(f"❌ Error calculating Stuff+: {e}")
        return

    top_pitches = df.sort_values('Stuff+', ascending=False).head(5).copy()

    # Fetch highlights for the game(s) these pitches came from
    top_game_pks = top_pitches['game_pk'].unique()
    all_highlights = []

    for pk in top_game_pks:
        all_highlights.extend(fetch_game_highlights(pk))

    # Match highlights to pitches
    top_pitches['Highlight URL'] = top_pitches.apply(lambda row: match_highlight_to_pitch(row, all_highlights), axis=1)
    print("\n🔥 Top 5 Pitches by Stuff+:")
    for i, row in top_pitches.iterrows():
        print(f"\n#{i+1} - {row['player_name']} | {row['pitch_type']}")
        print(f"   Stuff+: {round(row['Stuff+'], 1)} | Velo: {round(row['release_speed'], 1)} | Spin: {int(row['release_spin_rate'])}")
        print(f"   🎥 Highlight: {row['Highlight URL'] if row['Highlight URL'] else 'Not available'}")


def fetch_game_highlights(game_pk):
    """
    Returns a list of all available highlight playIds and titles for a given MLB game.
    """
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Error fetching gamefeed for {game_pk}: {e}")
        return []

    highlights = []
    for play in data['liveData']['plays']['allPlays']:
        media = play.get('media', {})
        if 'epg' in media:
            for group in media['epg']:
                if group.get('title') == 'highlight':
                    for item in group.get('items', []):
                        play_id = item.get('playId')
                        title = item.get('title')
                        pitcher = item.get('player', {}).get('fullName', '')
                        highlights.append({
                            'title': title,
                            'player': pitcher,
                            'play_id': play_id,
                            'url': f"https://baseballsavant.mlb.com/sporty-videos?playId={play_id}"
                        })
    return highlights

def match_highlight_to_pitch(row, highlights):
    """
    Attempts to match a Stuff+ pitch row to a highlight.
    Match is fuzzy, based on pitcher name.
    """
    pitcher_name = row['player_name'].lower()
    pitch_type = row['pitch_type'].lower()

    for h in highlights:
        if pitcher_name.split()[0] in h['player'].lower():
            if pitch_type in h['title'].lower() or 'strikeout' in h['title'].lower() or 'swinging' in h['title'].lower():
                return h['url']
    return None




if __name__ == "__main__":
    main()
