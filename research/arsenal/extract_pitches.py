"""Turn the big cached statcast season pickles into one compact parquet.

The cached pulls are ~120 columns of mostly-unused statcast fields at 600-900 MB each,
and this box has under 3 GB free. So each season is loaded in its own process, cut to
the ~35 columns the arsenal study needs, downcast, and written out; nothing ever holds
two seasons at once.
"""

import gc
import os
import pickle
import sys

import pandas as pd

CACHE = os.path.join(".cache", "statcast")
OUT = sys.argv[1] if len(sys.argv) > 1 else "pitches"

# Files identified by re-deriving utils.cache._cache_key over (start, end) date pairs.
SEASONS = {
    2024: "f8d728fbd3509e41af2a6a882c4d544836a8bf983cbb8fabfb14dc4b2822ba04.pkl",
    2025: "b73c557e409f181c3b33f038ec2c737bb1d77dc08b9a2c8fdcf2aa2158c82255.pkl",
    2026: "79e709110d956d1dfbfa5038c87852aeb41596a756a3c4471d4e19e0af2c61d9.pkl",
}

KEEP = [
    # identity / ordering
    "game_date", "game_pk", "game_type", "inning", "inning_topbot",
    "at_bat_number", "pitch_number", "home_team", "away_team",
    "batter", "pitcher", "stand", "p_throws", "n_thruorder_pitcher",
    # pitch characteristics (what "arsenal" is measured from)
    "pitch_type", "release_speed", "release_spin_rate", "pfx_x", "pfx_z",
    "release_extension", "arm_angle", "plate_x", "plate_z", "sz_top", "sz_bot", "zone",
    # outcome of the pitch / PA
    "type", "description", "events",
    "launch_speed", "launch_angle",
    "estimated_woba_using_speedangle", "woba_value", "woba_denom",
    "delta_run_exp",
]

FLOAT32 = [
    "release_speed", "release_spin_rate", "pfx_x", "pfx_z", "release_extension",
    "arm_angle", "plate_x", "plate_z", "sz_top", "sz_bot",
    "launch_speed", "launch_angle", "estimated_woba_using_speedangle",
    "woba_value", "woba_denom", "delta_run_exp",
]
INT32 = ["game_pk", "batter", "pitcher", "at_bat_number", "pitch_number", "inning"]
CATEGORY = ["game_type", "inning_topbot", "home_team", "away_team", "stand", "p_throws",
            "pitch_type", "type", "description", "events", "zone", "n_thruorder_pitcher"]


def load_season(year, filename):
    path = os.path.join(CACHE, filename)
    print(f"  loading {year} from {filename[:12]}... ({os.path.getsize(path)/1e6:.0f} MB)",
          flush=True)
    with open(path, "rb") as handle:
        raw = pickle.load(handle)
    print(f"    {raw.shape[0]:,} pitches x {raw.shape[1]} cols", flush=True)

    missing = [c for c in KEEP if c not in raw.columns]
    if missing:
        print(f"    missing columns (filled null): {missing}", flush=True)
    frame = raw[[c for c in KEEP if c in raw.columns]].copy()
    del raw
    gc.collect()

    for column in missing:
        frame[column] = pd.NA
    frame["game_date"] = pd.to_datetime(frame["game_date"]).dt.date.astype("datetime64[ns]")
    frame["season"] = year
    for column in FLOAT32:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float32")
    for column in INT32:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("int32")
    for column in CATEGORY:
        frame[column] = frame[column].astype("string").astype("category")
    return frame


def main():
    os.makedirs(OUT, exist_ok=True)
    for year, filename in SEASONS.items():
        target = os.path.join(OUT, f"pitches_{year}.parquet")
        if os.path.exists(target):
            print(f"  {year} already extracted, skipping", flush=True)
            continue
        frame = load_season(year, filename)
        # Regular season only. The season pulls start March 1, which is spring training.
        regular = frame[frame["game_type"].astype("string").eq("R")]
        print(f"    regular season: {len(regular):,} pitches, "
              f"{regular['game_pk'].nunique():,} games, "
              f"{regular['game_date'].min().date()} to {regular['game_date'].max().date()}",
              flush=True)
        regular.to_parquet(target, index=False, compression="zstd")
        print(f"    wrote {target} ({os.path.getsize(target)/1e6:.0f} MB)", flush=True)
        del frame, regular
        gc.collect()


if __name__ == "__main__":
    main()
