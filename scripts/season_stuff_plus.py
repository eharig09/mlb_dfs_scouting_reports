import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import datetime
from tqdm import tqdm
from pybaseball import statcast
from models.stuff_model import calculate_stuff_plus
from daily_run import fetch_game_highlights, match_highlight_to_pitch

# Define the season start and today
start_date = datetime.date(2025, 3, 27)
end_date = datetime.date.today()

# Aggregate all data
all_stuff = []

print(f"Processing Stuff+ from {start_date} to {end_date}...\n")

current_date = start_date
while current_date <= end_date:
    try:
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"📅 {date_str}")
        df = statcast(start_dt=date_str, end_dt=date_str)

        if not df.empty and 'player_name' in df.columns:
            df = calculate_stuff_plus(df)
            df['game_date'] = date_str
            all_stuff.append(df)
    except Exception as e:
        print(f"⚠️ Error on {current_date}: {e}")
    current_date += datetime.timedelta(days=1)

# Combine all collected data
season_df = pd.concat(all_stuff, ignore_index=True)
season_df.dropna(subset=['Stuff+', 'pitch_type'], inplace=True)
season_df = season_df.sort_values('Stuff+', ascending=False)

top_by_type = (
    season_df
    .sort_values('Stuff+', ascending=False)
    .groupby('pitch_type')
    .head(5)
    .copy()
)


top_by_type['Highlight URL'] = None

for idx, row in tqdm(top_by_type.iterrows(), total=len(top_by_type)):
    game_pk = row.get('game_pk')
    if pd.isna(game_pk): continue

    try:
        highlights = fetch_game_highlights(int(game_pk))
        highlight_url = match_highlight_to_pitch(row, highlights)
        top_by_type.at[idx, 'Highlight URL'] = highlight_url
    except Exception as e:
        print(f"Error matching highlight for {row['player_name']} on {row['game_date']}: {e}")

top_by_type = top_by_type[[
    'game_date', 'player_name', 'pitch_type', 'release_speed',
    'release_spin_rate', 'Stuff+', 'Highlight URL'
]].sort_values(['pitch_type', 'Stuff+'], ascending=[True, False])

# Export
top_by_type.to_csv("top_5_stuff_plus_per_pitch_type.csv", index=False)
top_by_type.to_html("top_5_stuff_plus_per_pitch_type.html", index=False)

print("\n🎯 Done! Top pitches saved to 'top_stuff_plus_by_pitch_type.csv'")
