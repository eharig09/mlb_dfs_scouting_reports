"""Assemble the study frame: who faced whom, what they did, and what was knowable first.

One row per (starting hitter, opposing starting pitcher) for every regular-season game in
2024-2026, plus one row per start. Everything on a row that is not an outcome is measured
as of the morning of the game.

The controls matter more than the feature here. Arsenal fit is heavily confounded with
"is this a good hitter" -- a good hitter hits everything -- so the raw correlation between
any arsenal score and fantasy points is guaranteed positive and says nothing. What the
analysis needs is the residual: does fit add anything once the hitter's own recent form,
his platoon baseline, his lineup slot, the park, and the opposing pitcher's quality are
already in the model. So those are all built here.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arsenal_engine import add_counters, load_pitches  # noqa: E402

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
SEASONS = (2024, 2025, 2026)


def load_box():
    frames = [pd.read_parquet(f"{DATA}/box_{season}.parquet") for season in SEASONS]
    box = pd.concat(frames, ignore_index=True)
    box["game_date"] = pd.to_datetime(box["game_date"])
    return box


def _as_of_form(events, keys, value_cols, min_prior=1):
    """Cumulative totals per key, as of the day before each event. Used for form controls.

    Returns the same frame with `<col>_prior` columns and a `prior_games` count.
    """
    daily = events.groupby(keys + ["game_date"], observed=True,
                           sort=True)[value_cols].sum().reset_index()
    daily["games"] = events.groupby(keys + ["game_date"], observed=True,
                                    sort=True).size().to_numpy()
    cols = value_cols + ["games"]
    daily[cols] = daily.groupby(keys, observed=True, sort=False)[cols].cumsum()
    daily = daily.sort_values("game_date", ignore_index=True)

    request = events[keys + ["game_date"]].copy()
    request["_row"] = np.arange(len(request))
    merged = pd.merge_asof(request.sort_values("game_date", kind="mergesort"),
                           daily, on="game_date", by=keys,
                           direction="backward", allow_exact_matches=False)
    merged = merged.sort_values("_row")
    out = pd.DataFrame(index=events.index)
    for column in value_cols:
        out[f"{column}_prior"] = merged[column].to_numpy()
    out["prior_games"] = merged["games"].to_numpy()
    out.loc[out["prior_games"].fillna(0) < min_prior, :] = np.nan
    return out


def main():
    print("loading pitches...", flush=True)
    pitches = load_pitches(DATA, SEASONS)
    box = load_box()
    print(f"  {len(pitches):,} pitches, {len(box):,} player-games", flush=True)

    # --- who started, and against whom ---------------------------------------------
    starters = box[(box["role"] == "P") & box["started"]].copy()
    # A handful of games record two "starters" for a side (suspended/resumed games).
    starters = starters.sort_values(["game_pk", "team", "IP"], ascending=[True, True, False])
    starters = starters.drop_duplicates(["game_pk", "team"], keep="first")

    hands = (pitches.groupby(["game_pk", "pitcher"], observed=True)["p_throws"]
             .first().reset_index().rename(columns={"pitcher": "player_id"}))
    hands["p_throws"] = hands["p_throws"].astype("string")
    starters = starters.merge(hands, on=["game_pk", "player_id"], how="left")

    # Batters faced by the starter, and the innings he actually got through, from statcast.
    # `n_thruorder_pitcher` is what separates a real start from an opener.
    faced = pitches[pitches["events"].notna()].copy()
    faced["is_starter_pa"] = False
    starter_key = set(zip(starters["game_pk"], starters["player_id"]))
    faced["is_starter_pa"] = [
        (g, p) in starter_key for g, p in zip(faced["game_pk"], faced["pitcher"])
    ]
    starter_pas = faced[faced["is_starter_pa"]]

    starts = starters.rename(columns={"player_id": "pitcher"})[
        ["game_pk", "game_date", "pitcher", "team", "p_throws",
         "dk", "IP", "K", "ER", "H", "BB", "HBP", "W", "PA"]]
    starts = starts.rename(columns={"team": "pitch_team", "dk": "pitcher_dk",
                                    "PA": "batters_faced"})
    starts["start_id"] = np.arange(len(starts))

    # --- hitters who started -------------------------------------------------------
    hitters = box[(box["role"] == "H") & box["started"]
                  & box["order_slot"].between(1, 9)].copy()
    hitters = hitters.drop_duplicates(["game_pk", "player_id"], keep="first")

    game_teams = starts[["game_pk", "pitch_team", "start_id", "pitcher", "p_throws",
                         "game_date"]]
    pairs = hitters.merge(game_teams, on="game_pk", how="inner")
    pairs = pairs[pairs["team"] != pairs["pitch_team"]].copy()
    pairs = pairs.rename(columns={"player_id": "batter", "dk": "hitter_dk",
                                  "game_date_x": "game_date"})
    pairs = pairs.drop(columns=[c for c in ("game_date_y",) if c in pairs.columns])
    pairs["pair_id"] = np.arange(len(pairs))
    print(f"  {len(starts):,} starts, {len(pairs):,} hitter-vs-starter pairs", flush=True)

    # Which side of the plate the hitter actually took against this starter (switch
    # hitters flip, so the roster hand would be wrong for a fifth of the sample).
    stands = (starter_pas.groupby(["game_pk", "batter"], observed=True)["stand"]
              .first().reset_index())
    stands["stand"] = stands["stand"].astype("string")
    pairs = pairs.merge(stands, on=["game_pk", "batter"], how="left")
    same_side = pairs["stand"].eq(pairs["p_throws"]).fillna(False).to_numpy()
    unknown = (pairs["stand"].isna() | pairs["p_throws"].isna()).to_numpy()
    pairs["platoon"] = np.where(unknown, "unknown", np.where(same_side, "same", "opp"))

    # --- outcome 2: what the hitter did against the starter specifically ------------
    counters = add_counters(starter_pas)
    counters["game_pk"] = starter_pas["game_pk"].to_numpy()
    counters["batter"] = starter_pas["batter"].to_numpy()
    vs_sp = counters.groupby(["game_pk", "batter"], observed=True).sum()
    vs_sp = vs_sp[["pa", "woba_num", "woba_den", "xwoba_sum", "xwoba_n",
                   "rv_sum", "rv_n", "k", "hr"]].reset_index()
    vs_sp = vs_sp.rename(columns={c: f"sp_{c}" for c in vs_sp.columns
                                  if c not in ("game_pk", "batter")})
    pairs = pairs.merge(vs_sp, on=["game_pk", "batter"], how="left")
    pairs["woba_vs_sp"] = pairs["sp_woba_num"] / pairs["sp_woba_den"].replace(0, np.nan)
    pairs["rv_vs_sp"] = pairs["sp_rv_sum"]

    # --- controls: hitter form -----------------------------------------------------
    hitter_events = (box[(box["role"] == "H") & box["started"]]
                     [["player_id", "game_date", "dk", "PA", "H", "HR", "BB", "K", "AB"]]
                     .sort_values("game_date", ignore_index=True))
    form = _as_of_form(hitter_events, ["player_id"], ["dk", "PA", "H", "HR", "BB", "K", "AB"],
                       min_prior=10)
    hitter_form = pd.concat([hitter_events[["player_id", "game_date"]], form], axis=1)
    hitter_form["hitter_dk_pg"] = hitter_form["dk_prior"] / hitter_form["prior_games"]
    hitter_form["hitter_pa_pg"] = hitter_form["PA_prior"] / hitter_form["prior_games"]
    hitter_form["hitter_k_rate"] = hitter_form["K_prior"] / hitter_form["PA_prior"]
    hitter_form["hitter_hr_rate"] = hitter_form["HR_prior"] / hitter_form["PA_prior"]
    hitter_form = hitter_form[["player_id", "game_date", "hitter_dk_pg", "hitter_pa_pg",
                               "hitter_k_rate", "hitter_hr_rate", "prior_games"]]
    hitter_form = hitter_form.rename(columns={"prior_games": "hitter_prior_games"})
    hitter_form = hitter_form.drop_duplicates(["player_id", "game_date"])
    pairs = pairs.merge(hitter_form.rename(columns={"player_id": "batter"}),
                        on=["batter", "game_date"], how="left")

    # --- controls: starter form ----------------------------------------------------
    starter_events = starts[["pitcher", "game_date", "pitcher_dk", "IP", "K", "ER",
                             "H", "BB", "batters_faced"]].sort_values("game_date",
                                                                      ignore_index=True)
    sform = _as_of_form(starter_events, ["pitcher"],
                        ["pitcher_dk", "IP", "K", "ER", "H", "BB", "batters_faced"],
                        min_prior=3)
    starter_form = pd.concat([starter_events[["pitcher", "game_date"]], sform], axis=1)
    starter_form["sp_dk_pg"] = starter_form["pitcher_dk_prior"] / starter_form["prior_games"]
    starter_form["sp_ip_pg"] = starter_form["IP_prior"] / starter_form["prior_games"]
    starter_form["sp_k_rate"] = starter_form["K_prior"] / starter_form["batters_faced_prior"]
    starter_form["sp_era"] = 9 * starter_form["ER_prior"] / starter_form["IP_prior"].replace(0, np.nan)
    starter_form["sp_whip"] = ((starter_form["H_prior"] + starter_form["BB_prior"])
                               / starter_form["IP_prior"].replace(0, np.nan))
    starter_form = starter_form[["pitcher", "game_date", "sp_dk_pg", "sp_ip_pg",
                                 "sp_k_rate", "sp_era", "sp_whip", "prior_games"]]
    starter_form = starter_form.rename(columns={"prior_games": "sp_prior_starts"})
    starter_form = starter_form.drop_duplicates(["pitcher", "game_date"])
    pairs = pairs.merge(starter_form, on=["pitcher", "game_date"], how="left")
    starts = starts.merge(starter_form, on=["pitcher", "game_date"], how="left")

    # --- game context --------------------------------------------------------------
    venue = (pitches.groupby("game_pk", observed=True)
             .agg(home_team=("home_team", "first"), away_team=("away_team", "first"))
             .reset_index())
    venue["home_team"] = venue["home_team"].astype("string")
    venue["away_team"] = venue["away_team"].astype("string")
    pairs = pairs.merge(venue, on="game_pk", how="left")
    starts = starts.merge(venue, on="game_pk", how="left")
    pairs["season"] = pairs["game_date"].dt.year
    starts["season"] = starts["game_date"].dt.year
    starts["is_home"] = (starts["pitch_team"] == starts["home_team"]).astype(int)

    # An "opener" is a start that was never meant to go deep. Its arsenal is real but the
    # lineup barely sees it, so these are tagged rather than dropped.
    starts["opener"] = (starts["IP"] <= 2.0).astype(int)
    pairs = pairs.merge(starts[["start_id", "opener", "pitcher_dk"]], on="start_id",
                        how="left")

    keep_pairs = ["pair_id", "start_id", "game_pk", "game_date", "season", "batter", "team",
                  "order_slot", "pitcher", "pitch_team", "p_throws", "stand", "platoon",
                  "home_team", "away_team", "opener",
                  "hitter_dk", "PA", "AB", "H", "HR", "BB", "K", "R", "RBI", "SB",
                  "woba_vs_sp", "rv_vs_sp", "sp_pa", "sp_k", "sp_hr",
                  "hitter_dk_pg", "hitter_pa_pg", "hitter_k_rate", "hitter_hr_rate",
                  "hitter_prior_games",
                  "sp_dk_pg", "sp_ip_pg", "sp_k_rate", "sp_era", "sp_whip",
                  "sp_prior_starts", "pitcher_dk"]
    pairs = pairs[[c for c in keep_pairs if c in pairs.columns]]

    os.makedirs(DATA, exist_ok=True)
    pairs.to_parquet(f"{DATA}/pairs.parquet", index=False, compression="zstd")
    starts.to_parquet(f"{DATA}/starts.parquet", index=False, compression="zstd")

    print(f"\nwrote pairs.parquet: {len(pairs):,} rows")
    print(f"  with hitter form:   {pairs['hitter_dk_pg'].notna().sum():,}")
    print(f"  with starter form:  {pairs['sp_dk_pg'].notna().sum():,}")
    print(f"  mean hitter DK:     {pairs['hitter_dk'].mean():.2f}")
    print(f"wrote starts.parquet: {len(starts):,} rows")
    print(f"  mean starter DK:    {starts['pitcher_dk'].mean():.2f}")
    print(f"  openers:            {int(starts['opener'].sum()):,}")
    print(pairs[["hitter_dk", "woba_vs_sp", "hitter_dk_pg", "sp_dk_pg"]].describe().round(3))


if __name__ == "__main__":
    main()
