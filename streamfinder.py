import pandas as pd
import re
import json
from collections import Counter
from pybaseball import playerid_lookup

# === Load CSV File ===
df = pd.read_csv("dk_lineups.csv", engine="python", quotechar='"', on_bad_lines="skip")

# === Identify Pitcher Columns ===
pitcher_columns = ["P", "P.1"]

# === Extract Player Names and Count ===
player_counter = Counter()
player_roles = {}

for col in df.columns:
    is_pitcher = col in pitcher_columns
    for entry in df[col].dropna():
        entry = str(entry)
        match = re.match(r"(.+?) \((\d+)\)", entry)
        if match:
            name, _ = match.groups()
            name = name.strip()
            player_counter[name] += 1
            if name not in player_roles:
                player_roles[name] = "pit" if is_pitcher else "bat"

# === Get Top 10 Most Frequent Players ===
top_players = player_counter.most_common(30)  # grab more in case lookups fail
resolved = {}
priority_count = 0

for name, _ in top_players:
    if priority_count >= 30:
        break
    try:
        first, last = name.split(" ")[0], name.split(" ")[-1]
        lookup = playerid_lookup(last, first)
        if not lookup.empty:
            row = lookup.iloc[0]
            resolved[name] = {
                "mlbam_id": str(row["key_mlbam"]),
                "type": player_roles.get(name, "bat")
            }
            priority_count += 1
    except Exception as e:
        print(f"Lookup failed for {name}: {e}")

# === Build StreamFinder Data with Detroit Tigers Ignored Globally ===
streamfinder_data = {
    "on_deck": "N",
    "include_CLI": "N",
    "delay": 10,
    "ignore": ["116"],  # Detroit Tigers team ID
    "priority": []
}

priority_num = 1
for name in resolved:
    streamfinder_data["priority"].append({
        "type": resolved[name]["type"],
        "data": resolved[name]["mlbam_id"],
        "immediate": "",
        "priority": priority_num
    })
    priority_num += 1

# === Save Cleaned StreamFinder.txt ===
with open("StreamFinder.txt", "w") as f:
    json.dump(streamfinder_data, f, indent=4)

print("✅ Clean StreamFinder.txt generated — Tigers globally ignored.")
