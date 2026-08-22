"""Walk-forward projection evaluation: is the model right, and does it know when it isn't?

`dfs.backtest` answers the first question. This module answers the second, which matters
more for a simulator: a projection that is 4 points off on average but *honest about being
4 points off* is usable, and one that is 3 points off while claiming a 1-point band is not.

**Why walk-forward.** Randomly splitting player-game rows leaks badly here. Two hitters in
the same lineup share a game state, a starter, a park and a weather reading, so a random
split puts near-copies of the test rows in the training set and every metric comes back
flattering. Worse, several of the model's constants (`FLOOR_FIT`, `BUST_FIT`, `CEILING_Z`)
were fitted on the very games any naive backtest scores, so the published calibration is
in-sample. Here, for target date D, calibration is refitted using **only dates strictly
before D**, and the result is scored on D. Nothing else is offered, because a random split
of this dataset is not a weaker measurement -- it is a wrong one.

**What comes out.** Point accuracy, ranking quality, distribution calibration and
probability calibration, each broken down by hitter/pitcher, salary tier, batting-order
slot, handedness matchup, slate size, and confirmed-vs-projected lineup status.

The distribution metrics need a predictive distribution, and the model publishes four
numbers rather than one: mean (`Proj`), 25th percentile (`Floor`), ~90th (`Ceiling`) and
P(<=3) (`Bust%`). `implied_quantiles` turns those into a monotone piecewise-linear quantile
function, which is enough for pinball loss, PIT and CRPS. That function is an
*interpretation* of the model's published bands, and it is stated here rather than buried,
because every distribution number below inherits its assumptions.
"""

import json
import os
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .backtest import actual_points
from .profiling import profiler
from .projections import BUST_FIT, FLOOR_FIT, MODEL_VERSION, project_game
from .slate import REPORT_DATA_DIR, cached_games

BENCHMARK_DIR = os.path.join("docs", "benchmarks")

# The quantile levels reported. 10/25/50/75/90 is what the plan asks for; the finer grid is
# for CRPS, which is the integral of pinball loss over all of [0, 1].
REPORT_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
CRPS_GRID = np.linspace(0.005, 0.995, 199)

# "Bust" and "ceiling hit" as binary events, for Brier scoring. Both are what the board
# actually claims: Bust% is P(<=3) by construction, and a 90th-percentile ceiling should be
# beaten about a tenth of the time.
BUST_THRESHOLD = 3.0
CEILING_TARGET = 0.10

# A DK pitcher can score below zero (ER -2, H -0.6). A hitter cannot.
LOWER_BOUND = {"H": 0.0, "P": -12.0}

# Share of the sub-3-point mass that is an exact zero.
#
# DK hitter scoring has a hard atom at zero and the model does not publish one: it gives
# P(<=3) as `Bust%` and nothing finer. Without an atom the implied distribution puts ~0
# probability on the single most common outcome in the dataset, every zero lands in the
# bottom PIT bin, and the histogram reports catastrophic miscalibration for a reason that is
# an artefact of the *reading*, not of the model.
#
# The split comes from the model's own documented base rates (dfs/projections.py: 27% of
# hitter-games score zero, 44% score <=3), so 27/44 = 0.61. This is an interpretation of
# numbers the model already states, not new information smuggled in. Pitchers have no atom:
# an inning pitched is worth 2.25 points on its own.
ZERO_SHARE_OF_BUST = {"H": 0.61, "P": 0.0}


# ---------------------------------------------------------------------------
# Predictive distribution
# ---------------------------------------------------------------------------

def implied_quantiles(proj, floor, ceiling, bust, player_type, levels):
    """Quantile function implied by the model's published bands, evaluated at `levels`.

    Knots, in the order the model states them:

        (0.001,             lower bound) hitters floor at 0, pitchers can go negative
        (0.61 * bust,       0.0)         the zero atom, for hitters (see ZERO_SHARE_OF_BUST)
        (bust,              3.0)         Bust% is defined as P(score <= 3)
        (0.25,              Floor)       Floor is the 25th percentile
        (0.90,              Ceiling)     Ceiling targets the 90th
        (0.999,             extrapolated) the ceiling band's slope, continued

    Between knots it interpolates linearly, and the whole thing is forced monotone with a
    running maximum. Monotonicity is not decoration: at high projections the fitted Bust%
    floor (0.32 for hitters) can put the 3-point knot above the 25th-percentile knot, and an
    unsorted quantile function would produce negative pinball losses.

    Vectorised over players: `proj` and friends are arrays, `levels` is a 1-D grid, and the
    result is (n_players, n_levels).
    """
    proj = np.asarray(proj, dtype=float)
    floor = np.asarray(floor, dtype=float)
    ceiling = np.asarray(ceiling, dtype=float)
    bust = np.clip(np.asarray(bust, dtype=float), 0.002, 0.95)
    levels = np.asarray(levels, dtype=float)
    low = LOWER_BOUND.get(player_type, 0.0)

    # Slope of the upper band, per unit of probability, used to extend past the ceiling.
    # Guarded below zero so a degenerate band cannot invert the tail.
    upper_slope = np.maximum(ceiling - proj, 0.1) / 0.40

    zero_share = ZERO_SHARE_OF_BUST.get(player_type, 0.0)
    knot_p = np.stack([
        np.full_like(proj, 0.001),
        np.maximum(bust * zero_share, 0.0015),
        bust,
        np.full_like(proj, 0.25),
        np.full_like(proj, 0.90),
        np.full_like(proj, 0.999),
    ], axis=1)
    knot_v = np.stack([
        np.full_like(proj, low),
        np.full_like(proj, 0.0 if zero_share else low),
        np.full_like(proj, BUST_THRESHOLD),
        floor,
        ceiling,
        ceiling + upper_slope * 0.10,
    ], axis=1)

    order = np.argsort(knot_p, axis=1, kind="stable")
    knot_p = np.take_along_axis(knot_p, order, axis=1)
    knot_v = np.maximum.accumulate(np.take_along_axis(knot_v, order, axis=1), axis=1)

    out = np.empty((len(proj), len(levels)), dtype=float)
    for i in range(len(proj)):
        out[i] = np.interp(levels, knot_p[i], knot_v[i])
    return out


def pit_values(actual, quantile_grid, levels, rng=None):
    """Randomized probability integral transform.

    F(y) for a well-calibrated model is Uniform(0, 1); a histogram that piles up at the ends
    means the bands are too narrow, and one that bulges in the middle means too wide.

    Randomized because DK scores have a real atom at zero -- 27% of hitter-games score
    exactly nothing -- and a deterministic PIT would stack all of those on one value and
    report catastrophic miscalibration for a model that is fine. For an outcome at an atom
    the transform draws uniformly across the probability mass the atom occupies, which is
    the standard treatment for discrete outcomes.
    """
    rng = rng or np.random.default_rng(0)
    actual = np.asarray(actual, dtype=float)
    values = np.empty(len(actual), dtype=float)
    for i, y in enumerate(actual):
        q = quantile_grid[i]
        # F(y) and F(y-) as the first and last grid level whose quantile equals y.
        upper = float(np.interp(y, q, levels, left=0.0, right=1.0))
        below = q < y - 1e-9
        lower = float(levels[below][-1]) if below.any() else 0.0
        values[i] = rng.uniform(min(lower, upper), upper) if upper > lower else upper
    return values


def pinball(actual, predicted_quantile, level):
    """Quantile (pinball) loss. Lower is better; scale is points."""
    diff = np.asarray(actual, dtype=float) - np.asarray(predicted_quantile, dtype=float)
    return np.where(diff >= 0, level * diff, (level - 1.0) * diff)


def crps(actual, quantile_grid, levels=CRPS_GRID):
    """Continuous ranked probability score by integrating pinball loss over [0, 1].

        CRPS(F, y) = 2 * integral_0^1 pinball_tau(y, F^-1(tau)) dtau

    Computed on the same quantile function the other distribution metrics use, so CRPS,
    pinball and PIT are all statements about one object rather than three approximations.
    """
    actual = np.asarray(actual, dtype=float)[:, None]
    losses = np.where(actual >= quantile_grid,
                      levels * (actual - quantile_grid),
                      (levels - 1.0) * (actual - quantile_grid))
    return 2.0 * losses.mean(axis=1)


def brier(probability, outcome):
    """Mean squared error of a probability forecast against a 0/1 outcome."""
    probability = np.asarray(probability, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    return float(np.mean((probability - outcome) ** 2))


def reliability(probability, outcome, bins=5):
    """Calibration curve: predicted probability vs observed frequency, per bin."""
    probability = np.asarray(probability, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    if len(probability) < bins * 2:
        return pd.DataFrame()
    try:
        cut = pd.qcut(probability, bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    frame = pd.DataFrame({"p": probability, "y": outcome, "bin": cut})
    table = frame.groupby("bin", observed=True).agg(
        n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
    table["gap"] = (table["observed"] - table["predicted"]).round(4)
    return table.round(4)


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def _salary_tier(frame):
    """Salary quintile within (date, type). Named so a report reads without a legend."""
    labels = ["min-priced", "cheap", "mid", "expensive", "premium"]
    out = pd.Series("unknown", index=frame.index, dtype=object)
    for _, group in frame.groupby(["Date", "Type"], sort=False):
        salary = pd.to_numeric(group.get("Salary"), errors="coerce")
        if salary.notna().sum() < 10:
            continue
        try:
            out.loc[group.index] = pd.qcut(salary, 5, labels=labels, duplicates="drop")
        except ValueError:
            continue
    return out


def _handedness(frame):
    """'RHB vs LHP' style label. Switch hitters are their own bucket, not folded in."""
    bats = frame.get("Bats", pd.Series("", index=frame.index)).fillna("").astype(str).str[:1]
    hand = frame.get("Opp SP Hand", pd.Series("", index=frame.index)).fillna("").astype(str).str[:1]
    label = pd.Series("unknown", index=frame.index, dtype=object)
    known = bats.isin(["L", "R", "S"]) & hand.isin(["L", "R"])
    label[known] = (bats[known].map({"L": "LHB", "R": "RHB", "S": "SWH"})
                    + " vs " + hand[known].map({"L": "LHP", "R": "RHP"}))
    label[frame["Type"] == "P"] = "pitcher"
    return label


def _salary_lookup(date):
    """({normalized name: DK row}, games priced) for a date, or ({}, 0).

    Rebuilt rows come straight from `project_game` and carry no price, so salary-tier and
    slate-size segments would be empty for every date without a snapshot. `build_slate`
    would supply them but also *drops* unpriced players, shrinking the very sample being
    measured -- so the export is read directly and joined on, keeping every projected row.

    When a date has several exports the largest wins. That is the main slate, and MLB
    Classic prices a player identically across the day's slates, so a player priced only on
    the early export still gets his real price.
    """
    from .salaries import list_salary_files, load_salaries, normalize_name, slate_games

    candidates = list_salary_files(date)
    if not candidates:
        return {}, 0
    best = max(candidates, key=lambda c: c.get("players") or 0)
    path = best.get("path")
    try:
        salaries = load_salaries(path)
        games = len(slate_games(path))
    except Exception:
        return {}, 0
    if salaries is None or salaries.empty:
        return {}, 0
    lookup = {}
    for _, row in salaries.iterrows():
        key = normalize_name(row.get("DK Name") or row.get("Name"))
        if key and key not in lookup:
            lookup[key] = row
    return lookup, games


def collect(dates, data_dir=REPORT_DATA_DIR, use_snapshots=True, quiet=False):
    """One row per projected player-game, with actuals and every segment column attached.

    Prefers a snapshot for each date so the projection being scored is the one that existed
    at lock. Falls back to re-projecting the cached payload, and records which happened in
    the `Source` column -- a metric computed over rebuilt rows is measuring a model against
    data it did not have, and the column is how a reader can tell.
    """
    from .snapshot import SnapshotError, list_snapshots, load_snapshot

    from .salaries import normalize_name

    rows = []
    priced_games = {}
    for date in dates:
        source = "rebuilt"
        snapshot_players = None
        if use_snapshots and list_snapshots(date):
            try:
                snap = load_snapshot(date)
                snapshot_players = snap.players
                source = f"snapshot:{snap.stage}"
            except SnapshotError:
                pass

        salary_lookup, game_count = ({}, 0) if snapshot_players is not None \
            else _salary_lookup(date)
        priced_games[str(date)] = game_count

        for path, away, home in cached_games(date, data_dir):
            pairing = f"{away}@{home}"
            try:
                with open(path, "rb") as handle:
                    payload = pickle.load(handle)
            except Exception as error:
                if not quiet:
                    print(f"  skip {date} {pairing}: {error}")
                continue

            game_id = (payload["advanced_context"].get("environment") or {}).get("game_id")
            if not game_id:
                continue
            outcomes = actual_points(int(game_id))
            if not outcomes:
                continue

            # {batting team: hand of the starter they face}, recovered from the payload.
            #
            # Snapshots are frozen dataframes, so one written before a column existed is
            # missing it forever. Exactly that happened: the 2026-08-01 export predates
            # `Opp SP Hand`, and its 169 hitters became an "unknown" handedness bucket that
            # was really a single date wearing a segment's clothes -- pooled against buckets
            # spanning the whole window, it looked like the largest handedness bias in the
            # report. Filling from the payload keeps a legacy snapshot from inventing a
            # segment; it changes no projection, only how the rows are labelled.
            arguments = payload["report_args"]
            starter_hand = {}
            for batting_team, starter in ((arguments[15], arguments[9]),
                                          (arguments[16], arguments[2])):
                throws = str((starter or {}).get("Throws") or "")[:1].upper()
                if throws in ("L", "R"):
                    starter_hand[str(batting_team)] = throws

            if snapshot_players is not None:
                projections = snapshot_players[snapshot_players["Game"] == pairing] \
                    .to_dict("records")
            else:
                with profiler.stage("project"):
                    projections = project_game(payload)
                for record in projections:
                    record["Game"] = pairing

            for projection in projections:
                mlbam = projection.get("MLBAM")
                if mlbam is None or (isinstance(mlbam, float) and pd.isna(mlbam)):
                    continue
                outcome = outcomes.get(int(mlbam))
                priced = salary_lookup.get(normalize_name(projection.get("Name")))
                hand = projection.get("Opp SP Hand")
                if hand is None or (isinstance(hand, float) and pd.isna(hand)) or not str(hand).strip():
                    hand = starter_hand.get(str(projection.get("Team")), "")
                rows.append({
                    "Date": str(date),
                    "Game": pairing,
                    "Source": source,
                    "Name": projection["Name"],
                    "Team": projection["Team"],
                    "MLBAM": int(mlbam),
                    "Type": projection["Type"],
                    "Slot": projection.get("Slot"),
                    "Bats": projection.get("Bats"),
                    "Opp SP Hand": hand,
                    "Lineup": projection.get("Lineup") or "",
                    "Salary": projection.get("Salary") if priced is None
                    else priced.get("Salary"),
                    "Own%": projection.get("Own%"),
                    "Proj": projection["Proj"],
                    "Ceiling": projection["Ceiling"],
                    "Floor": projection.get("Floor"),
                    "Bust%": projection.get("Bust%"),
                    "PA": projection.get("PA"),
                    "IP": projection.get("IP"),
                    "Actual PA": (outcome or {}).get("Actual PA"),
                    "Actual IP": (outcome or {}).get("Actual IP"),
                    "Actual": outcome["Actual"] if outcome else None,
                    "Played": bool(outcome),
                })

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    # Slate size is a property of the night, not the row, and it changes how much a
    # projection error costs -- a 2-game slate has no alternatives to be wrong about. The
    # DK export's game count is the real slate size; the cache's is only what got scouted,
    # so it is the fallback rather than the answer.
    cached_per_date = frame.groupby("Date")["Game"].nunique()
    frame["Slate Games"] = frame["Date"].map(
        lambda d: priced_games.get(str(d)) or cached_per_date.get(str(d), 0))
    frame["Slate Size"] = pd.cut(frame["Slate Games"], [0, 4, 9, 99],
                                 labels=["small (<=4)", "medium (5-9)", "large (10+)"])
    frame["Salary Tier"] = _salary_tier(frame)
    frame["Handedness"] = _handedness(frame)
    frame["Lineup Status"] = np.where(
        frame["Lineup"].astype(str).str.strip().eq("Confirmed"), "confirmed", "projected")
    frame["Order"] = pd.to_numeric(frame["Slot"], errors="coerce").fillna(0).astype(int)
    return frame


# ---------------------------------------------------------------------------
# Walk-forward calibration
# ---------------------------------------------------------------------------

def fit_bands(history):
    """Refit the floor / bust calibration on a set of finished player-games.

    Same functional forms the model ships (`FLOOR_FIT`, `BUST_FIT`) so the refit is a
    like-for-like replacement rather than a different model wearing the same name:

        floor(p)  = max(0, slope * p + intercept)     targeting the 25th percentile
        bust(p)   = slope * p + intercept             targeting P(actual <= 3)

    Floor is fitted by regressing the per-bucket 25th percentile of actuals on the bucket's
    mean projection, not by least squares on the raw pairs -- ordinary regression fits the
    mean, and the mean of a right-skewed distribution is nowhere near its 25th percentile.
    """
    fits = {}
    for player_type in ("H", "P"):
        group = history[(history["Type"] == player_type) & history["Played"]]
        if len(group) < 60:
            fits[player_type] = {"floor": dict(FLOOR_FIT[player_type]),
                                 "bust": dict(BUST_FIT[player_type]),
                                 "n": int(len(group)), "fitted": False}
            continue
        try:
            buckets = pd.qcut(group["Proj"], min(8, group["Proj"].nunique()), duplicates="drop")
        except ValueError:
            fits[player_type] = {"floor": dict(FLOOR_FIT[player_type]),
                                 "bust": dict(BUST_FIT[player_type]),
                                 "n": int(len(group)), "fitted": False}
            continue
        table = group.groupby(buckets, observed=True).agg(
            proj=("Proj", "mean"),
            q25=("Actual", lambda s: float(np.quantile(s, 0.25))),
            bust=("Actual", lambda s: float((s <= BUST_THRESHOLD).mean())),
            n=("Actual", "size"))
        table = table[table["n"] >= 5]
        if len(table) < 3:
            fits[player_type] = {"floor": dict(FLOOR_FIT[player_type]),
                                 "bust": dict(BUST_FIT[player_type]),
                                 "n": int(len(group)), "fitted": False}
            continue
        floor_slope, floor_intercept = np.polyfit(table["proj"], table["q25"], 1)
        bust_slope, bust_intercept = np.polyfit(table["proj"], table["bust"], 1)
        fits[player_type] = {
            "floor": {"slope": float(floor_slope), "intercept": float(floor_intercept)},
            "bust": {"slope": float(bust_slope), "intercept": float(bust_intercept)},
            "n": int(len(group)),
            "fitted": True,
        }
    return fits


def apply_bands(frame, fits):
    """Replace Floor and Bust% with values from a calibration fitted on earlier dates."""
    out = frame.copy()
    floor = np.full(len(out), np.nan)
    bust = np.full(len(out), np.nan)
    for player_type, fit in fits.items():
        mask = (out["Type"] == player_type).to_numpy()
        if not mask.any():
            continue
        proj = pd.to_numeric(out.loc[mask, "Proj"], errors="coerce").fillna(0).to_numpy()
        floor[mask] = np.maximum(0.0, fit["floor"]["slope"] * proj + fit["floor"]["intercept"])
        raw = fit["bust"]["slope"] * proj + fit["bust"]["intercept"]
        bust[mask] = np.clip(raw, 0.02, 0.95) * 100.0
    out["Floor"] = np.where(np.isnan(floor), out["Floor"], floor)
    out["Bust%"] = np.where(np.isnan(bust), out["Bust%"], bust)
    return out


def walk_forward(frame, min_train_rows=200, recalibrate=True):
    """Score each date using bands fitted only on strictly earlier dates.

    Dates before the training set is large enough are dropped rather than scored with the
    shipped constants, because those constants were fitted on this same period -- scoring
    them here would report in-sample calibration as if it were out-of-sample, which is the
    exact error this module exists to avoid.

    `recalibrate=False` keeps the shipped bands and evaluates them as-is. That number is
    still worth having (it is what the tool actually does tonight), but it is in-sample for
    any date inside the original fitting window and is labelled as such in the report.
    """
    scored, used = [], []
    dates = sorted(frame["Date"].unique())
    for index, date in enumerate(dates):
        target = frame[frame["Date"] == date]
        history = frame[frame["Date"] < date]
        if recalibrate:
            if len(history[history["Played"]]) < min_train_rows:
                continue
            fits = fit_bands(history)
            target = apply_bands(target, fits)
            used.append({"date": date, "train_dates": index,
                         "train_rows": int(len(history[history["Played"]])),
                         "fits": {k: {"fitted": v["fitted"], "n": v["n"]}
                                  for k, v in fits.items()}})
        scored.append(target)
    if not scored:
        return pd.DataFrame(), used
    return pd.concat(scored, ignore_index=True), used


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(group, rng=None):
    """Every metric for one segment. Returns a flat dict; NaN where n is too small."""
    played = group[group["Played"] & group["Actual"].notna()]
    n = len(played)
    result = {"n": n, "n_projected": len(group),
              "scratch_rate": round(1 - n / max(1, len(group)), 4)}
    if n < 5:
        return result

    actual = pd.to_numeric(played["Actual"], errors="coerce").to_numpy(dtype=float)
    proj = pd.to_numeric(played["Proj"], errors="coerce").to_numpy(dtype=float)
    ceiling = pd.to_numeric(played["Ceiling"], errors="coerce").to_numpy(dtype=float)
    floor = pd.to_numeric(played["Floor"], errors="coerce").fillna(0).to_numpy(dtype=float)
    bust = pd.to_numeric(played["Bust%"], errors="coerce").fillna(40).to_numpy(dtype=float) / 100

    error = actual - proj
    result.update({
        "bias": round(float(error.mean()), 3),
        "mae": round(float(np.abs(error).mean()), 3),
        "rmse": round(float(np.sqrt((error ** 2).mean())), 3),
        "proj_mean": round(float(proj.mean()), 3),
        "actual_mean": round(float(actual.mean()), 3),
        "actual_sd": round(float(actual.std(ddof=1)), 3) if n > 1 else None,
    })

    if n >= 3 and np.std(proj) > 0 and np.std(actual) > 0:
        result["pearson"] = round(float(np.corrcoef(proj, actual)[0, 1]), 3)
        result["spearman"] = round(float(pd.Series(proj).corr(pd.Series(actual),
                                                              method="spearman")), 3)

    # Top-decile recall: of the players who actually finished in the top 10% of the segment,
    # what share did we rank in our top 10%? This is the question a lineup asks -- being
    # roughly right about everyone is worth less than catching the players who paid.
    if n >= 20:
        size = max(1, int(round(n * 0.10)))
        predicted_top = set(np.argsort(-proj)[:size])
        actual_top = set(np.argsort(-actual)[:size])
        result["top_decile_recall"] = round(len(predicted_top & actual_top) / size, 3)
        order = np.argsort(-proj)
        fifth = max(1, n // 5)
        result["quintile_spread"] = round(
            float(actual[order[:fifth]].mean() - actual[order[-fifth:]].mean()), 3)

    # --- distribution ---
    player_type = played["Type"].iloc[0] if played["Type"].nunique() == 1 else "H"
    report_grid = implied_quantiles(proj, floor, ceiling, bust, player_type, REPORT_QUANTILES)
    for j, level in enumerate(REPORT_QUANTILES):
        result[f"pinball_{int(level * 100)}"] = round(
            float(pinball(actual, report_grid[:, j], level).mean()), 4)
        result[f"coverage_{int(level * 100)}"] = round(
            float((actual <= report_grid[:, j]).mean()), 4)

    fine_grid = implied_quantiles(proj, floor, ceiling, bust, player_type, CRPS_GRID)
    result["crps"] = round(float(crps(actual, fine_grid).mean()), 4)

    pit = pit_values(actual, fine_grid, CRPS_GRID, rng=rng or np.random.default_rng(11))
    result["pit_mean"] = round(float(pit.mean()), 4)           # 0.50 when calibrated
    result["pit_sd"] = round(float(pit.std(ddof=1)), 4)        # 0.289 when calibrated
    # KS distance to Uniform(0,1): one number for "is the whole distribution honest".
    ordered = np.sort(pit)
    ecdf = np.arange(1, n + 1) / n
    result["pit_ks"] = round(float(np.max(np.abs(ecdf - ordered))), 4)
    result["pit_hist"] = [int(c) for c in np.histogram(pit, bins=10, range=(0, 1))[0]]

    # --- probability calibration ---
    bust_outcome = (actual <= BUST_THRESHOLD).astype(float)
    result["bust_brier"] = round(brier(bust, bust_outcome), 4)
    result["bust_predicted"] = round(float(bust.mean()), 4)
    result["bust_observed"] = round(float(bust_outcome.mean()), 4)
    # A climatological forecast -- "everyone busts at the base rate" -- is the bar any
    # per-player bust model has to clear. Skill below zero means the model adds noise.
    base = bust_outcome.mean()
    reference = brier(np.full(n, base), bust_outcome)
    result["bust_brier_skill"] = round(
        float(1 - result["bust_brier"] / reference), 4) if reference > 0 else None

    ceiling_outcome = (actual > ceiling).astype(float)
    result["ceiling_brier"] = round(brier(np.full(n, CEILING_TARGET), ceiling_outcome), 4)
    result["ceiling_exceeded"] = round(float(ceiling_outcome.mean()), 4)
    return result


SEGMENTS = [
    ("type", "Type"),
    ("salary tier", "Salary Tier"),
    ("batting order", "Order"),
    ("handedness", "Handedness"),
    ("slate size", "Slate Size"),
    ("lineup status", "Lineup Status"),
]


def evaluate(frame, segments=SEGMENTS, min_segment=25, seed=11):
    """Overall metrics plus a table per segment. Segments are split within player type."""
    rng = np.random.default_rng(seed)
    report = {"overall": {}, "segments": {}}
    for player_type, label in (("H", "hitters"), ("P", "pitchers")):
        group = frame[frame["Type"] == player_type]
        if not group.empty:
            report["overall"][label] = metrics(group, rng=rng)

    for name, column in segments:
        if column not in frame.columns or name == "type":
            continue
        rows = []
        for player_type in ("H", "P"):
            typed = frame[frame["Type"] == player_type]
            if typed.empty:
                continue
            for value, group in typed.groupby(column, observed=True):
                if len(group) < min_segment:
                    continue
                row = {"type": player_type, "segment": str(value), "column": column}
                row.update(metrics(group, rng=rng))
                rows.append(row)
        if rows:
            report["segments"][name] = rows
    return report


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_HEADLINE = ["n", "bias", "mae", "rmse", "spearman", "top_decile_recall", "crps",
             "pit_mean", "pit_ks", "bust_brier_skill", "ceiling_exceeded"]


def _row_line(label, row, width=26):
    def cell(key, spec="{:>8}"):
        value = row.get(key)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return spec.format("-")
        return spec.format(f"{value:.3f}" if isinstance(value, float) else value)

    return (f"{str(label)[:width]:<{width}}" + "".join(cell(k) for k in _HEADLINE))


def format_report(report, show_pit=True):
    lines = []
    header = f"{'':<26}" + "".join(f"{k.replace('_', ' ')[:8]:>8}" for k in _HEADLINE)
    lines.append(header)
    lines.append("-" * len(header))
    for label, row in report["overall"].items():
        lines.append(_row_line(label.upper(), row))

    for name, rows in report.get("segments", {}).items():
        lines.append("")
        lines.append(f"-- by {name} " + "-" * max(0, len(header) - len(name) - 7))
        for row in rows:
            lines.append(_row_line(f"{row['type']}  {row['segment']}", row))

    if show_pit:
        lines.append("")
        lines.append("-- PIT histogram (10 bins, flat = calibrated) --")
        for label, row in report["overall"].items():
            hist = row.get("pit_hist")
            if not hist:
                continue
            total = sum(hist) or 1
            bars = " ".join(f"{c / total * 100:4.1f}" for c in hist)
            lines.append(f"{label:<12} {bars}   (10.0 each when calibrated)")

    lines.append("")
    lines.append(
        "Reading it: bias / mae / rmse are DK points. spearman is rank quality.\n"
        "top_decile_recall is the share of the actual top 10% we ranked in our top 10%.\n"
        "crps scores the whole predicted distribution (lower is better).\n"
        "pit_mean should be 0.50 and pit_ks near 0; a U-shaped PIT histogram means the\n"
        "bands are too narrow, a hump in the middle means too wide.\n"
        "bust_brier_skill > 0 means the per-player bust number beats quoting the base rate.\n"
        "ceiling_exceeded should sit near 0.10 -- that is what a 90th percentile claims."
    )
    return "\n".join(lines)


def save_report(report, path=None, label="walkforward", extra=None):
    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    path = path or os.path.join(BENCHMARK_DIR, f"evaluate_{label}.json")
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version": MODEL_VERSION,
        **(extra or {}),
        "report": report,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def available_dates(data_dir=REPORT_DATA_DIR):
    """Dates with cached report data, ascending."""
    import glob
    dates = set()
    for path in glob.glob(os.path.join(data_dir, "*.pkl")):
        stem = os.path.basename(path).split("_")[0]
        if len(stem) == 10 and stem[4] == "-":
            dates.add(stem)
    return sorted(dates)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Walk-forward projection evaluation with distribution calibration.")
    parser.add_argument("--dates", help="Comma-separated dates. Default: every cached date.")
    parser.add_argument("--from", dest="start", help="First date (inclusive).")
    parser.add_argument("--to", dest="end", help="Last date (inclusive).")
    parser.add_argument("--data-dir", default=REPORT_DATA_DIR)
    parser.add_argument("--no-snapshots", action="store_true",
                        help="Re-project from the payload cache instead of using snapshots. "
                             "Faster to set up, but scores information the model did not have.")
    parser.add_argument("--no-recalibrate", action="store_true",
                        help="Evaluate the shipped floor/bust constants as-is. In-sample for "
                             "any date inside their original fitting window.")
    parser.add_argument("--min-train-rows", type=int, default=200,
                        help="Player-games of history required before a date is scored.")
    parser.add_argument("--min-segment", type=int, default=25,
                        help="Smallest segment worth reporting.")
    parser.add_argument("--label", default="walkforward", help="Output filename suffix.")
    parser.add_argument("--csv", help="Also write the collected player-game rows here.")
    args = parser.parse_args()

    if args.dates:
        dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    else:
        dates = available_dates(args.data_dir)
        if args.start:
            dates = [d for d in dates if d >= args.start]
        if args.end:
            dates = [d for d in dates if d <= args.end]
    if not dates:
        print("No cached dates to evaluate.")
        return

    print(f"collecting {len(dates)} date(s): {dates[0]} .. {dates[-1]}")
    frame = collect(dates, data_dir=args.data_dir, use_snapshots=not args.no_snapshots)
    if frame.empty:
        print("No finished games found for those dates.")
        return

    sources = frame["Source"].value_counts().to_dict()
    print(f"  {len(frame)} player-games, {frame['Played'].sum()} played")
    print("  source: " + ", ".join(f"{k} {v}" for k, v in sources.items()))
    if all(k == "rebuilt" for k in sources):
        print("  [i] no snapshots on disk, so these projections were rebuilt from the "
              "payload cache.\n      Payloads are mutated by --refresh-lineups, so some of "
              "this is hindsight.\n      Take snapshots going forward: "
              "python -m dfs.snapshot --date <date> --stage final")

    scored, schedule = walk_forward(frame, min_train_rows=args.min_train_rows,
                                    recalibrate=not args.no_recalibrate)
    if scored.empty:
        print(f"\nNot enough history: no date has {args.min_train_rows}+ earlier player-games.")
        print("Re-run with --no-recalibrate to score the shipped bands (in-sample), or "
              "--min-train-rows lower.")
        return

    kept = sorted(scored["Date"].unique())
    mode = "shipped bands (in-sample)" if args.no_recalibrate else \
        "walk-forward, refit per date on prior dates only"
    print(f"\n=== {mode} ===")
    print(f"scored {len(kept)} date(s): {kept[0]} .. {kept[-1]}  ({len(scored)} player-games)")
    print()

    report = evaluate(scored, min_segment=args.min_segment)
    print(format_report(report))

    path = save_report(report, label=args.label, extra={
        "dates_collected": dates, "dates_scored": kept,
        "recalibrated": not args.no_recalibrate,
        "sources": sources, "calibration_schedule": schedule,
    })
    print(f"\n  -> {path}")
    if args.csv:
        scored.to_csv(args.csv, index=False, encoding="utf-8-sig")
        print(f"  -> {args.csv}")


if __name__ == "__main__":
    main()
