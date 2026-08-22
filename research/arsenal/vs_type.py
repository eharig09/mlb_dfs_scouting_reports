"""The other way the report measures a matchup: how the lineup hits *pitchers like this one*.

`generate_team_pitcher_type_results` does not slice by pitch shape. It picks whole comp
pitchers off a similarity ranking and asks how the lineup has hit them. That is a different
bet from the arsenal K edge -- comps carry sequencing, command and pitch mix together
rather than one pitch type at a time, and they buy far more plate appearances -- so it
deserves the same test rather than an assumption.

Reproduced here closely enough to compare:

* A pitcher is a vector of usage, velocity and movement per pitch type, snapshotted on the
  first of each month from data before that date. A start in that month uses that month's
  snapshot, so nothing in the start can reach the vector.
* Comps are the nearest same-handed pitchers in that space, which is also the report's
  baseline choice -- it means "vs type" reads as an edge beyond the platoon split rather
  than re-measuring it.
* The lineup's line against those comps is residualized against the same nine hitters'
  line against that hand and shrunk, exactly as the K edge is, so the two features differ
  in what they slice on and nothing else.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arsenal_engine import COUNTERS, _as_of, add_counters, load_pitches  # noqa: E402

# The pitch types a similarity vector is built from. Anything rarer is noise in the
# distance and is folded into nothing -- it simply does not appear as a dimension.
TYPES = ["FF", "SI", "FC", "SL", "ST", "CU", "KC", "CH", "FS", "SV"]
SHAPE_COLUMNS = ["velo", "pfx_x", "pfx_z"]

MIN_COMP_PITCHES = 300     # a comp needs this much history at the snapshot to be usable
DEFAULT_COMPS = 40
SHRINK_PA = 25.0           # same prior weight the arsenal K edge uses


def monthly_snapshots(dates):
    """First of each month covered by the data, as snapshot boundaries."""
    start = pd.Timestamp(dates.min()).replace(day=1)
    end = pd.Timestamp(dates.max()).replace(day=1) + pd.offsets.MonthBegin(1)
    return pd.date_range(start, end, freq="MS")


def pitcher_vectors(pitches, snapshots):
    """One arsenal vector per (pitcher, snapshot), built only from earlier data."""
    frame = pitches[pitches["pitch_type"].isin(TYPES)].copy()
    daily = frame.groupby(["pitcher", "pitch_type", "game_date"], observed=True).agg(
        n=("release_speed", "size"),
        velo=("release_speed", "sum"),
        pfx_x=("pfx_x", "sum"),
        pfx_z=("pfx_z", "sum"),
    ).reset_index()
    columns = ["n"] + SHAPE_COLUMNS
    daily[columns] = daily.groupby(["pitcher", "pitch_type"], observed=True,
                                   sort=False)[columns].cumsum()
    daily = daily.sort_values("game_date", ignore_index=True)

    pitchers = frame["pitcher"].unique()
    grid = pd.MultiIndex.from_product([pitchers, TYPES, snapshots],
                                      names=["pitcher", "pitch_type", "game_date"])
    grid = grid.to_frame(index=False)
    filled = _as_of(grid, daily, by=["pitcher", "pitch_type"], value_cols=columns,
                    window_days=None)

    filled["usage_n"] = filled["n"]
    total = filled.groupby(["pitcher", "game_date"], observed=True)["usage_n"].transform("sum")
    filled["usage"] = np.where(total > 0, filled["usage_n"] / total.replace(0, np.nan), 0.0)
    for column in SHAPE_COLUMNS:
        filled[column] = np.where(filled["n"] > 0,
                                  filled[column] / filled["n"].replace(0, np.nan), np.nan)

    wide = filled.pivot_table(index=["pitcher", "game_date"], columns="pitch_type",
                              values=["usage"] + SHAPE_COLUMNS, observed=True)
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide["total_pitches"] = total.groupby(
        [filled["pitcher"], filled["game_date"]]).first().to_numpy()

    # A pitch a pitcher does not throw gets usage 0 and league-average shape, so the
    # distance between "no slider" and "a slider" is carried by usage rather than by a
    # meaningless velocity gap.
    for column in wide.columns:
        if column.startswith("usage_"):
            wide[column] = wide[column].fillna(0.0)
        elif any(column.startswith(f"{s}_") for s in SHAPE_COLUMNS):
            wide[column] = wide[column].fillna(wide[column].mean())
    return wide


def comp_sets(starts, vectors, hands, n_comps=DEFAULT_COMPS):
    """The nearest same-handed pitchers to each starter, as of that start's snapshot."""
    feature_columns = [c for c in vectors.columns
                       if c.startswith("usage_") or any(c.startswith(f"{s}_")
                                                        for s in SHAPE_COLUMNS)]
    table = vectors.merge(hands, on="pitcher", how="left")
    table = table[table["total_pitches"] >= MIN_COMP_PITCHES].copy()

    # Standardize once, globally, so a millimetre of movement and a percent of usage are
    # not silently weighted by their units.
    values = table[feature_columns].to_numpy(dtype="float32")
    values = (values - values.mean(axis=0)) / (values.std(axis=0) + 1e-9)
    table = table.reset_index(drop=True)

    starts = starts.copy()
    starts["snapshot"] = starts["game_date"].values.astype("datetime64[M]").astype(
        "datetime64[ns]")
    rows = []
    for (snapshot, hand), group in starts.groupby(["snapshot", "p_throws"], observed=True):
        pool = table.index[(table["game_date"] == snapshot)
                           & (table["p_throws"].astype(str) == str(hand))].to_numpy()
        if len(pool) <= n_comps + 1:
            continue
        pool_values = values[pool]
        pool_ids = table.loc[pool, "pitcher"].to_numpy()
        index = {int(p): i for i, p in enumerate(pool_ids)}

        wanted = group["pitcher"].map(lambda p: index.get(int(p), -1)).to_numpy()
        usable = wanted >= 0
        if not usable.any():
            continue
        query = pool_values[wanted[usable]]
        distances = ((query[:, None, :] - pool_values[None, :, :]) ** 2).sum(axis=2)
        # A pitcher is his own nearest neighbour; drop him before taking the top N.
        distances[np.arange(len(query)), wanted[usable]] = np.inf
        nearest = np.argpartition(distances, n_comps, axis=1)[:, :n_comps]

        start_ids = group.loc[usable, "start_id"].to_numpy()
        rows.append(pd.DataFrame({
            "start_id": np.repeat(start_ids, n_comps),
            "comp": pool_ids[nearest.ravel()],
            "distance": distances[np.repeat(np.arange(len(query)), n_comps),
                                  nearest.ravel()],
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def matchup_history(pitches, counters):
    """Daily cumulative counters per (batter, pitcher) -- the batter-vs-comp lookup."""
    keys = pd.DataFrame({
        "batter": pitches["batter"].to_numpy(),
        "pitcher": pitches["pitcher"].to_numpy(),
        "game_date": pitches["game_date"].to_numpy(),
    })
    frame = pd.concat([keys, counters.reset_index(drop=True)], axis=1)
    daily = frame.groupby(["batter", "pitcher", "game_date"],
                          observed=True, sort=True)[COUNTERS].sum().reset_index()
    daily[COUNTERS] = daily.groupby(["batter", "pitcher"], observed=True,
                                    sort=False)[COUNTERS].cumsum()
    return daily.sort_values("game_date", ignore_index=True)


def vs_type_edge(pairs, comps, history, baseline, metric="k_rate", chunk=400):
    """Lineup K rate against comp pitchers, minus its rate against that hand, shrunk.

    Chunked over starts: the request frame is nine hitters times forty comps per start,
    which is four million rows for a full run and does not want to exist all at once.
    """
    request_base = pairs[["pair_id", "start_id", "batter", "p_throws", "game_date"]]
    start_ids = comps["start_id"].unique()
    pieces = []
    for begin in range(0, len(start_ids), chunk):
        batch = set(start_ids[begin:begin + chunk])
        sub_pairs = request_base[request_base["start_id"].isin(batch)]
        sub_comps = comps[comps["start_id"].isin(batch)][["start_id", "comp"]]
        request = sub_pairs.merge(sub_comps, on="start_id", how="inner")
        request = request.rename(columns={"comp": "pitcher"})
        if request.empty:
            continue
        filled = _as_of(request, history, by=["batter", "pitcher"],
                        value_cols=COUNTERS, window_days=None)
        pieces.append(filled.groupby("start_id", observed=True)[COUNTERS].sum())
    if not pieces:
        return pd.DataFrame()
    totals = pd.concat(pieces).groupby(level=0).sum()

    # The same nine hitters' overall line against that hand, over the same window.
    base_request = pairs[["pair_id", "start_id", "batter", "p_throws", "game_date"]]
    base = _as_of(base_request, baseline, by=["batter", "p_throws"], value_cols=COUNTERS,
                  window_days=None)
    base_totals = base.groupby("start_id", observed=True)[COUNTERS].sum()

    joined = totals.join(base_totals, how="right", rsuffix="_base").fillna(0.0)
    out = pd.DataFrame({"start_id": joined.index,
                        "type_pa": joined["pa"].to_numpy()})

    # Both metrics are emitted so a null result cannot be blamed on the choice of one.
    # K rate is what the arsenal edge uses; wOBA is closer to the report's own OPS Diff.
    for name, numerator, denominator in (("k", "k", "pa"),
                                         ("woba", "woba_num", "woba_den")):
        observed = joined[denominator]
        base_observed = joined[f"{denominator}_base"]
        value = np.where(observed > 0,
                         joined[numerator] / observed.replace(0, np.nan), np.nan)
        base_value = np.where(base_observed > 0,
                              joined[f"{numerator}_base"]
                              / base_observed.replace(0, np.nan), np.nan)
        shrunk = ((np.nan_to_num(value) * observed + np.nan_to_num(base_value) * SHRINK_PA)
                  / (observed + SHRINK_PA))
        edge = np.where(np.isnan(base_value), np.nan, shrunk - base_value)
        # Sign convention matches the arsenal K edge: positive is the pitcher's edge.
        out[f"type_{name}_edge"] = edge if name == "k" else -edge
        out[f"type_base_{name}"] = base_value
    return out


def build(data_dir, seasons=(2024, 2025, 2026), n_comps=DEFAULT_COMPS):
    """End to end: pitch table in, one `type_k_edge` per start out."""
    from arsenal_engine import build_batter_baseline

    print("  loading pitches...", flush=True)
    pitches = load_pitches(data_dir, seasons)
    counters = add_counters(pitches)
    pairs = pd.read_parquet(f"{data_dir}/pairs.parquet")
    starts = pd.read_parquet(f"{data_dir}/starts.parquet")

    snapshots = monthly_snapshots(pitches["game_date"])
    print(f"  {len(snapshots)} monthly snapshots; building pitcher vectors...", flush=True)
    vectors = pitcher_vectors(pitches, snapshots)
    hands = (pitches.groupby("pitcher", observed=True)["p_throws"].first()
             .astype("string").reset_index())

    print("  finding comps...", flush=True)
    comps = comp_sets(starts[["start_id", "pitcher", "game_date", "p_throws"]],
                      vectors, hands, n_comps=n_comps)
    print(f"  {comps['start_id'].nunique():,} starts have comps "
          f"({len(comps):,} comp rows)", flush=True)

    print("  building batter-vs-pitcher history...", flush=True)
    history = matchup_history(pitches, counters)
    baseline = build_batter_baseline(pitches, counters)
    del pitches, counters

    print("  scoring...", flush=True)
    edge = vs_type_edge(pairs, comps, history, baseline)
    edge.to_parquet(f"{data_dir}/vs_type.parquet", index=False, compression="zstd")
    print(f"  wrote vs_type.parquet: {len(edge):,} starts, "
          f"median comp PA {edge['type_pa'].median():.0f}")
    return edge


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "data")
