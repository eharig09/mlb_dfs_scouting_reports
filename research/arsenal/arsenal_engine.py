"""A tunable, leak-free arsenal-fit engine, vectorized over three seasons at once.

The report today measures arsenal fit one game at a time: take the starter's top-3 pitches
by usage, filter a batter's history to pitches of that type within +/-2.5 mph and +/-300 rpm
of the starter's average, and usage-weight the resulting OPS/xwOBA. That is one point in a
large space of choices, and none of the choices have been tested against anything.

This module makes every choice a knob and makes evaluating a knob setting cheap, so the
space can actually be searched. Two ideas do the work:

**Everything is a sum.** Every metric in play -- OPS, xwOBA, RV/100, whiff%, K% -- is a
ratio of two sums over pitches. So the pitch table is collapsed once into per-day counter
rows keyed by (batter, pitcher hand, pitch type, velocity bin), and a cumulative sum over
date turns "this batter's line against this kind of pitch before date D" into a lookup
instead of a scan. Changing the metric is then free; only the final division changes.

**As-of by construction.** Features are read with `merge_asof(..., allow_exact_matches=False)`
against those cumulative rows, so a row for game date D can only ever see data from strictly
before D. A rolling window is the difference of two such reads. There is no code path in
which today's game can inform today's feature, which matters because the thing being tested
is exactly the kind of claim that looks true when it leaks.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

# Pitches that are not really pitches (position players, pitchouts, intentional-ball lobs)
# or are too rare to have a stable batter history. Left out of both arsenal and history.
JUNK_PITCHES = {"PO", "UN", "EP", "SC", "FO", "CS", "KN", "FA"}

# Contact-quality thresholds, matching the report's existing definitions.
HARD_HIT_MPH = 95.0

SWING_DESCRIPTIONS = {
    "foul", "hit_into_play", "swinging_strike", "swinging_strike_blocked",
    "foul_tip", "foul_bunt", "missed_bunt", "bunt_foul_tip",
}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}


@dataclass(frozen=True)
class ArsenalConfig:
    """One way of measuring "how does this hitter do against this pitcher's stuff"."""

    # --- which pitches count as the arsenal ---
    top_n: int = 3                  # keep this many pitches, by usage
    min_usage: float = 0.0          # ...and drop any below this share (0-1) of the arsenal
    weight_power: float = 1.0       # usage ** power, renormalized. 0 = equal weight

    # --- how tightly the batter's history must match the pitcher's shape ---
    velo_tol: float = 2.5           # mph; None/inf disables the velocity match
    match_by_hand: bool = True      # only count history vs same-handed pitchers

    # --- how much history ---
    lookback_days: int = 730        # batter history window ending the day before the game
    arsenal_lookback_days: int = 365  # window used to measure the pitcher's own mix

    # --- how the raw numbers become a score ---
    metric: str = "rv100"           # rv100 | xwoba | woba | ops | slg | whiff | k_rate
    shrink: float = 150.0           # denominator units of prior weight toward the baseline
    residualize: bool = True        # score = fit - the batter's own overall baseline
    min_denom: float = 0.0          # below this much observed history, the row is NaN


# Counter columns carried through every aggregation. Each metric is a ratio of two of these.
COUNTERS = [
    "pitches", "swings", "whiffs", "ooz", "chases",
    "pa", "ab", "h1", "h2", "h3", "hr", "bb", "k", "hbp", "sf",
    "woba_num", "woba_den", "xwoba_sum", "xwoba_n",
    "bip", "hardhit", "rv_sum", "rv_n",
]

# metric -> (numerator column, denominator column, higher is better for the hitter)
METRICS = {
    "rv100": ("rv_sum", "rv_n", True),
    "xwoba": ("xwoba_sum", "xwoba_n", True),
    "woba": ("woba_num", "woba_den", True),
    "slg": ("tb", "ab", True),
    "ops": ("ops_num", "ops_den", True),
    "whiff": ("whiffs", "swings", False),
    "k_rate": ("k", "pa", False),
    "hardhit": ("hardhit", "bip", True),
}


def load_pitches(data_dir, seasons=(2024, 2025, 2026)):
    """The compact pitch table, cleaned of pitches nothing can be learned from."""
    frames = [pd.read_parquet(f"{data_dir}/pitches_{season}.parquet") for season in seasons]
    pitches = pd.concat(frames, ignore_index=True)
    pitches["pitch_type"] = pitches["pitch_type"].astype("string")
    pitches = pitches[
        pitches["pitch_type"].notna()
        & ~pitches["pitch_type"].isin(JUNK_PITCHES)
        & pitches["release_speed"].notna()
    ].copy()
    return pitches.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"],
                               ignore_index=True)


def add_counters(pitches):
    """Add the per-pitch 0/1 columns that every metric is a sum of."""
    frame = pitches
    description = frame["description"].astype("string")
    events = frame["events"].astype("string")

    out = pd.DataFrame(index=frame.index)
    out["pitches"] = 1.0
    out["swings"] = description.isin(SWING_DESCRIPTIONS).astype("float32")
    out["whiffs"] = description.isin(WHIFF_DESCRIPTIONS).astype("float32")
    # Zone 11-14 are the four out-of-zone quadrants in statcast's zone map.
    zone = pd.to_numeric(frame["zone"].astype("string"), errors="coerce")
    out["ooz"] = (zone >= 11).astype("float32")
    out["chases"] = (out["ooz"].astype(bool) & out["swings"].astype(bool)).astype("float32")

    # PA-level counters live on the pitch that ended the PA, so they sum correctly.
    ends_pa = events.notna() & events.ne("")
    out["pa"] = ends_pa.astype("float32")
    single = events.eq("single")
    double = events.eq("double")
    triple = events.eq("triple")
    homer = events.eq("home_run")
    walk = events.isin(["walk", "intent_walk"])
    strikeout = events.isin(["strikeout", "strikeout_double_play"])
    hbp = events.eq("hit_by_pitch")
    sac_fly = events.isin(["sac_fly", "sac_fly_double_play"])
    out["h1"] = single.astype("float32")
    out["h2"] = double.astype("float32")
    out["h3"] = triple.astype("float32")
    out["hr"] = homer.astype("float32")
    out["bb"] = walk.astype("float32")
    out["k"] = strikeout.astype("float32")
    out["hbp"] = hbp.astype("float32")
    out["sf"] = sac_fly.astype("float32")
    out["ab"] = (ends_pa & ~walk & ~hbp & ~sac_fly
                 & ~events.isin(["sac_bunt", "sac_bunt_double_play", "catcher_interf"])
                 ).astype("float32")

    woba_den = pd.to_numeric(frame["woba_denom"], errors="coerce").fillna(0.0)
    out["woba_num"] = (pd.to_numeric(frame["woba_value"], errors="coerce").fillna(0.0)
                       ).astype("float32")
    out["woba_den"] = woba_den.astype("float32")

    xwoba = pd.to_numeric(frame["estimated_woba_using_speedangle"], errors="coerce")
    out["xwoba_sum"] = xwoba.fillna(0.0).astype("float32")
    out["xwoba_n"] = xwoba.notna().astype("float32")

    launch = pd.to_numeric(frame["launch_speed"], errors="coerce")
    out["bip"] = launch.notna().astype("float32")
    out["hardhit"] = (launch >= HARD_HIT_MPH).fillna(False).astype("float32")

    run_value = pd.to_numeric(frame["delta_run_exp"], errors="coerce")
    out["rv_sum"] = run_value.fillna(0.0).astype("float32")
    out["rv_n"] = run_value.notna().astype("float32")
    return out


def build_batter_history(pitches, counters, velo_bin=1.0):
    """Daily counter totals per (batter, pitcher hand, pitch type, velocity bin).

    The velocity bin is what makes a tunable shape match possible without re-scanning
    2M pitches: a +/-2.5 mph tolerance is a slice of bins, and a wider tolerance is a
    wider slice of the same table.
    """
    keys = pd.DataFrame({
        "batter": pitches["batter"].to_numpy(),
        "p_throws": pitches["p_throws"].astype("string").to_numpy(),
        "pitch_type": pitches["pitch_type"].astype("string").to_numpy(),
        "velo_bin": np.round(pitches["release_speed"].to_numpy() / velo_bin).astype("int16"),
        "game_date": pitches["game_date"].to_numpy(),
    })
    frame = pd.concat([keys, counters.reset_index(drop=True)], axis=1)
    daily = frame.groupby(["batter", "p_throws", "pitch_type", "velo_bin", "game_date"],
                          observed=True, sort=True)[COUNTERS].sum().reset_index()
    group = ["batter", "p_throws", "pitch_type", "velo_bin"]
    daily[COUNTERS] = daily.groupby(group, observed=True, sort=False)[COUNTERS].cumsum()
    return daily.sort_values("game_date", ignore_index=True)


def build_batter_baseline(pitches, counters):
    """Daily counter totals per (batter, pitcher hand) over every pitch type.

    This is the batter's overall line against that hand -- the thing an arsenal-fit number
    has to beat to be worth anything.
    """
    keys = pd.DataFrame({
        "batter": pitches["batter"].to_numpy(),
        "p_throws": pitches["p_throws"].astype("string").to_numpy(),
        "game_date": pitches["game_date"].to_numpy(),
    })
    frame = pd.concat([keys, counters.reset_index(drop=True)], axis=1)
    daily = frame.groupby(["batter", "p_throws", "game_date"],
                          observed=True, sort=True)[COUNTERS].sum().reset_index()
    daily[COUNTERS] = daily.groupby(["batter", "p_throws"], observed=True,
                                    sort=False)[COUNTERS].cumsum()
    return daily.sort_values("game_date", ignore_index=True)


def build_pitcher_mix(pitches, counters):
    """Daily cumulative pitch counts and velocity sums per (pitcher, pitch type).

    Usage share and average velocity as of a date come from here, so an arsenal is
    measured from what the pitcher had actually thrown before the start -- never from
    the start itself.
    """
    keys = pd.DataFrame({
        "pitcher": pitches["pitcher"].to_numpy(),
        "pitch_type": pitches["pitch_type"].astype("string").to_numpy(),
        "game_date": pitches["game_date"].to_numpy(),
        "n": 1.0,
        "velo_sum": pitches["release_speed"].to_numpy(),
        "spin_sum": pd.to_numeric(pitches["release_spin_rate"], errors="coerce")
                      .fillna(0.0).to_numpy(),
        "spin_n": pitches["release_spin_rate"].notna().astype(float).to_numpy(),
    })
    cols = ["n", "velo_sum", "spin_sum", "spin_n"]
    daily = keys.groupby(["pitcher", "pitch_type", "game_date"],
                         observed=True, sort=True)[cols].sum().reset_index()
    daily[cols] = daily.groupby(["pitcher", "pitch_type"], observed=True,
                                sort=False)[cols].cumsum()
    return daily.sort_values("game_date", ignore_index=True)


def _normalize_keys(frame, by):
    """Make join keys comparable across frames. merge_asof matches on dtype, not value,
    so a category here and a string there silently produces an all-null join."""
    out = frame.copy()
    # Parquet round-trips dates as datetime64[us] and statcast gives [ns]; merge_asof
    # refuses to join across the two.
    out["game_date"] = out["game_date"].astype("datetime64[ns]")
    for column in by:
        if column not in out.columns:
            continue
        values = out[column]
        if isinstance(values.dtype, pd.CategoricalDtype):
            values = values.astype("string")
        if pd.api.types.is_numeric_dtype(values):
            out[column] = pd.to_numeric(values, errors="coerce").fillna(-1).astype("int64")
        else:
            out[column] = values.astype("string").fillna("?").astype(object)
    return out


def _as_of(requests, history, by, value_cols, window_days):
    """Cumulative `value_cols` over a trailing window ending the day before each request.

    A window is the difference of two as-of reads of the same cumulative table: the total
    through yesterday minus the total through (yesterday - window). `allow_exact_matches`
    is False on the near edge so a game can never see itself.
    """
    requests = _normalize_keys(requests.reset_index(drop=True), by)
    history = _normalize_keys(history, by)
    requests["_order"] = np.arange(len(requests), dtype="int64")
    requests = requests.sort_values("game_date", kind="mergesort", ignore_index=True)

    near = pd.merge_asof(requests, history, on="game_date", by=by,
                         direction="backward", allow_exact_matches=False)
    values = near[value_cols].fillna(0.0).to_numpy()

    if window_days is not None and np.isfinite(window_days):
        far = requests.copy()
        far["game_date"] = far["game_date"] - pd.Timedelta(days=int(window_days))
        far = far.sort_values("game_date", kind="mergesort", ignore_index=True)
        far = pd.merge_asof(far, history, on="game_date", by=by,
                            direction="backward", allow_exact_matches=False)
        # Both reads are re-sorted independently, so realign on the request row id.
        far = far.set_index("_order").loc[near["_order"].to_numpy()]
        values = values - far[value_cols].fillna(0.0).to_numpy()

    out = near.drop(columns=value_cols)
    out[value_cols] = values
    return out.sort_values("_order", ignore_index=True).drop(columns="_order")


def pitcher_arsenals(starts, mix, config):
    """The weighted arsenal each starter brought to each start, as of that morning.

    `starts` needs columns (start_id, pitcher, game_date). Returns one row per
    (start_id, pitch_type) with a usage weight and the average velocity behind it.
    """
    pitch_types = mix["pitch_type"].unique()
    grid = starts.merge(pd.DataFrame({"pitch_type": pitch_types}), how="cross")
    grid = _as_of(grid, mix, by=["pitcher", "pitch_type"],
                  value_cols=["n", "velo_sum", "spin_sum", "spin_n"],
                  window_days=config.arsenal_lookback_days)
    grid = grid[grid["n"] > 0].copy()
    grid["velo"] = grid["velo_sum"] / grid["n"]
    grid["spin"] = np.where(grid["spin_n"] > 0, grid["spin_sum"] / grid["spin_n"].replace(0, np.nan), np.nan)

    total = grid.groupby("start_id", observed=True)["n"].transform("sum")
    grid["usage"] = grid["n"] / total
    grid = grid.sort_values(["start_id", "usage"], ascending=[True, False])
    grid["rank"] = grid.groupby("start_id", observed=True).cumcount()
    grid = grid[(grid["rank"] < config.top_n) & (grid["usage"] >= config.min_usage)].copy()

    if config.weight_power == 0:
        grid["raw_weight"] = 1.0
    else:
        grid["raw_weight"] = grid["usage"] ** config.weight_power
    grid["weight"] = grid["raw_weight"] / grid.groupby("start_id", observed=True)[
        "raw_weight"].transform("sum")
    return grid[["start_id", "pitcher", "game_date", "pitch_type", "usage", "weight",
                 "velo", "spin", "n"]]


def _ratio(frame, metric):
    """Turn counter sums into one rate, plus the denominator it rests on."""
    if metric == "ops":
        obp_den = frame["ab"] + frame["bb"] + frame["hbp"] + frame["sf"]
        hits = frame["h1"] + frame["h2"] + frame["h3"] + frame["hr"]
        obp = np.where(obp_den > 0, (hits + frame["bb"] + frame["hbp"]) / obp_den.replace(0, np.nan), np.nan)
        total_bases = frame["h1"] + 2 * frame["h2"] + 3 * frame["h3"] + 4 * frame["hr"]
        slg = np.where(frame["ab"] > 0, total_bases / frame["ab"].replace(0, np.nan), np.nan)
        return pd.Series(obp + slg, index=frame.index), obp_den
    if metric == "slg":
        total_bases = frame["h1"] + 2 * frame["h2"] + 3 * frame["h3"] + 4 * frame["hr"]
        return total_bases / frame["ab"].replace(0, np.nan), frame["ab"]
    numerator, denominator, _ = METRICS[metric]
    values = frame[numerator] / frame[denominator].replace(0, np.nan)
    if metric == "rv100":
        values = values * 100
    return values, frame[denominator]


def batter_fit(pairs, arsenals, history, baseline, config):
    """The arsenal-fit score for every (batter, start) pair in `pairs`.

    `pairs` needs (pair_id, start_id, batter, game_date). The returned frame carries the
    raw fit, the batter's own baseline against that hand, the residual between them, and
    the amount of history each rests on -- because a fit number without its denominator
    is not interpretable.
    """
    arsenal = arsenals[["start_id", "pitch_type", "weight", "velo"]]
    request = pairs[["pair_id", "start_id", "batter", "p_throws", "game_date"]].merge(
        arsenal, on="start_id", how="inner").reset_index(drop=True)

    # Expand each (pair, arsenal pitch) into the velocity bins the tolerance admits.
    if config.velo_tol is None or not np.isfinite(config.velo_tol):
        # No shape match: every bin of that pitch type counts.
        bins = (history[["pitch_type", "velo_bin"]].drop_duplicates()
                .astype({"pitch_type": "string"}))
        request["pitch_type"] = request["pitch_type"].astype("string")
        request = request.merge(bins, on="pitch_type", how="left")
    else:
        span = int(np.floor(config.velo_tol))
        offsets = np.arange(-span, span + 1)
        centers = np.round(request["velo"].to_numpy()).astype("int32")
        request = request.iloc[np.repeat(np.arange(len(request)), len(offsets))].copy()
        request["velo_bin"] = (np.repeat(centers, len(offsets))
                               + np.tile(offsets, len(centers))).astype("int32")
    request = request.reset_index(drop=True)

    if config.match_by_hand:
        by = ["batter", "p_throws", "pitch_type", "velo_bin"]
    else:
        by = ["batter", "pitch_type", "velo_bin"]
        history = history.groupby(["batter", "pitch_type", "velo_bin", "game_date"],
                                  observed=True, sort=False)[COUNTERS].sum().reset_index()
        history[COUNTERS] = history.groupby(["batter", "pitch_type", "velo_bin"],
                                            observed=True, sort=False)[COUNTERS].cumsum()
        history = history.sort_values("game_date", ignore_index=True)

    filled = _as_of(request, history, by=by, value_cols=COUNTERS,
                    window_days=config.lookback_days)

    # Weighted counter sums per pair: the weight multiplies both the numerator and the
    # denominator, so the result is a usage-weighted rate rather than a rate of a pooled
    # sample. A pitcher who throws 60% fastballs then gets 60% of the say.
    weights = filled["weight"].to_numpy()[:, None]
    weighted = pd.DataFrame(filled[COUNTERS].to_numpy() * weights,
                            columns=COUNTERS, index=filled.index)
    weighted["pair_id"] = filled["pair_id"].to_numpy()
    fit_sums = weighted.groupby("pair_id", observed=True, sort=False)[COUNTERS].sum()

    base_request = pairs[["pair_id", "batter", "p_throws", "game_date"]].copy()
    base_by = ["batter", "p_throws"] if config.match_by_hand else ["batter"]
    if not config.match_by_hand:
        baseline = baseline.groupby(["batter", "game_date"], observed=True,
                                    sort=False)[COUNTERS].sum().reset_index()
        baseline[COUNTERS] = baseline.groupby("batter", observed=True,
                                              sort=False)[COUNTERS].cumsum()
        baseline = baseline.sort_values("game_date", ignore_index=True)
    base_sums = _as_of(base_request, baseline, by=base_by, value_cols=COUNTERS,
                       window_days=config.lookback_days).set_index("pair_id")[COUNTERS]

    joined = fit_sums.join(base_sums, how="right", rsuffix="_base").fillna(0.0)
    fit_value, fit_denom = _ratio(joined[COUNTERS], config.metric)
    base_value, base_denom = _ratio(joined[[f"{c}_base" for c in COUNTERS]]
                                    .rename(columns=lambda c: c[:-5]), config.metric)

    # Shrink the fit toward the batter's own baseline rather than toward league average:
    # the question is whether *this hitter* does better against *this shape* than he
    # normally does, so his own line is the right thing to fall back to.
    weight_obs = fit_denom.fillna(0.0)
    shrunk = ((fit_value.fillna(0.0) * weight_obs + base_value.fillna(0.0) * config.shrink)
              / (weight_obs + config.shrink))
    shrunk = shrunk.where(base_value.notna())

    out = pd.DataFrame({
        "pair_id": joined.index,
        "fit_raw": fit_value.to_numpy(),
        "fit": shrunk.to_numpy(),
        "baseline": base_value.to_numpy(),
        "fit_denom": weight_obs.to_numpy(),
        "base_denom": base_denom.fillna(0.0).to_numpy(),
    })
    out["score"] = out["fit"] - out["baseline"] if config.residualize else out["fit"]
    higher_is_better = METRICS[config.metric][2] if config.metric in METRICS else True
    if not higher_is_better:
        out["score"] = -out["score"]
    if config.min_denom:
        out.loc[out["fit_denom"] < config.min_denom, "score"] = np.nan
    return out


def sweep(base_config, **overrides):
    """Every combination of the given knob values, as configs."""
    import itertools
    names = list(overrides)
    for values in itertools.product(*(overrides[name] for name in names)):
        yield replace(base_config, **dict(zip(names, values)))
