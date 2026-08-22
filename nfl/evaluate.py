"""Walk-forward projection evaluation: is the model right, and does it beat doing nothing?

Two questions, and the second is the one that decides whether any of this ships.

**Accuracy** is the easy half -- error against actual DK points, and whether the published
bands mean what they say. A projection that is 4 points off on average but honest about
being 4 points off is usable; one that is 3 off while claiming a 1-point band is not.

**Skill over a baseline** is the half that matters. Any projection will correlate with
outcomes, because good players score more than bad ones and everybody knows who the good
players are. The question is whether it beats the cheapest thing that also knows that:
`season_average`, a player's own mean DK points to date. If the model cannot beat that, the
model is an expensive way to compute an average and should not be trusted with money.

**Walk-forward, strictly.** For target week W, everything is refit using only games before
W. `nfl.projections.project_week` enforces the cutoff itself, so there is no path by which
this can leak -- which is what makes running it over completed seasons meaningful rather
than flattering. Random splits are not offered: two receivers in the same game share an
opponent, a game script and a total, so a random split puts near-copies of the test rows in
the training set and every metric comes back looking better than the model is.

    from nfl.evaluate import walk_forward, summarize
    result = walk_forward(2025, weekly, schedules)
    print(summarize(result))
"""

import numpy as np
import pandas as pd

from .backtest import offense_actuals
from .projections import (MODEL_VERSION, POSITIONS, opponent_factors, positional_priors,
                          project_week, _before)

# Weeks 1-2 are excluded by default. Every player's prior is a positional average that early,
# so the model is not being tested -- the baseline it is measured against has the same
# problem, and comparing two priors to each other says nothing about either.
DEFAULT_FIRST_WEEK = 3


def season_average_baseline(history):
    """Each player's mean DK points to date. The bar the model has to clear.

    Deliberately the *cheapest* honest baseline. It needs no matchup, no betting line and no
    opponent model, and it already encodes the single largest signal in fantasy football --
    that good players score more than bad ones.
    """
    scored = offense_actuals(history)
    if scored.empty:
        return {}
    return scored.groupby("player_id")["dk_points"].mean().to_dict()


def walk_forward(season, weekly, schedules, first_week=DEFAULT_FIRST_WEEK, last_week=None,
                 positions=POSITIONS):
    """Project every week of a season from its past, and join on what actually happened."""
    weeks = sorted(weekly[weekly["season"] == season]["week"].unique())
    weeks = [int(w) for w in weeks if w >= first_week
             and (last_week is None or w <= last_week)]

    frames = []
    for week in weeks:
        history = _before(weekly, season, week)
        history = history[history["position"].isin(positions)]
        if history.empty:
            continue
        # Priors and opponent factors are refit per week from the past only. Expensive, and
        # the expense is the point: reusing a single fit computed over the whole season is
        # exactly the leak this module exists to avoid.
        priors = positional_priors(history)
        defense = opponent_factors(history)
        projected = project_week(season, week, weekly, schedules,
                                 history=history, priors=priors, defense=defense)
        if projected.empty:
            continue

        baseline = season_average_baseline(history)
        projected["Baseline"] = projected["player_id"].map(baseline)
        projected["season"], projected["week"] = season, week
        frames.append(projected)

    if not frames:
        return pd.DataFrame()
    projections = pd.concat(frames, ignore_index=True)

    actual = offense_actuals(weekly[weekly["season"] == season])
    actual = actual[["season", "week", "player_id", "dk_points"]].rename(
        columns={"dk_points": "Actual"})
    merged = projections.merge(actual, on=["season", "week", "player_id"], how="inner")
    merged["model_error"] = merged["Proj"] - merged["Actual"]
    merged["baseline_error"] = merged["Baseline"] - merged["Actual"]
    merged["model_version"] = MODEL_VERSION
    return merged


def _errors(frame, column):
    error = frame[column].dropna()
    if error.empty:
        return {"mae": np.nan, "rmse": np.nan}
    return {"mae": float(error.abs().mean()),
            "rmse": float(np.sqrt((error ** 2).mean()))}


def _rank_quality(frame, predictor):
    """Spearman correlation with the outcome. The metric a lineup actually cares about.

    A DFS build never uses the projected number as a number -- it uses the ordering, and the
    value per dollar that ordering implies. A model can be biased by three points on every
    player and still build the same lineups.
    """
    rows = frame[[predictor, "Actual"]].dropna()
    if len(rows) < 30:
        return np.nan
    return float(rows[predictor].corr(rows["Actual"], method="spearman"))


def summarize(result, by=("Pos",)):
    """Accuracy, ranking quality and the margin over the baseline, overall and by segment."""
    if result is None or result.empty:
        return pd.DataFrame()

    def block(frame, label):
        model, baseline = _errors(frame, "model_error"), _errors(frame, "baseline_error")
        return {
            "segment": label, "n": len(frame),
            "model_mae": round(model["mae"], 3), "base_mae": round(baseline["mae"], 3),
            "mae_gain": round(baseline["mae"] - model["mae"], 3),
            "model_rmse": round(model["rmse"], 3), "base_rmse": round(baseline["rmse"], 3),
            "model_rho": round(_rank_quality(frame, "Proj"), 4),
            "base_rho": round(_rank_quality(frame, "Baseline"), 4),
            "bias": round(float(frame["model_error"].mean()), 3),
        }

    rows = [block(result, "ALL")]
    for column in by:
        if column in result.columns:
            for value, group in result.groupby(column):
                if len(group) >= 50:
                    rows.append(block(group, f"{column}={value}"))
    return pd.DataFrame(rows)


def band_calibration(result):
    """Do Floor and Ceiling mean what they claim?

    Floor is published as roughly a 25th-percentile outcome and Ceiling as roughly a 90th,
    so the honest check is simply how often the actual landed below each. A ceiling that is
    only exceeded 2% of the time is not a ceiling, it is a number nobody can use to build a
    tournament lineup.
    """
    if result is None or result.empty:
        return pd.DataFrame()
    rows = []
    for label, group in [("ALL", result)] + list(result.groupby("Pos")):
        clean = group.dropna(subset=["Actual", "Floor", "Ceiling"])
        if len(clean) < 50:
            continue
        rows.append({
            "segment": label if isinstance(label, str) else f"Pos={label}",
            "n": len(clean),
            "below_floor_pct": round(float((clean["Actual"] < clean["Floor"]).mean() * 100), 1),
            "target_below_floor": 25.0,
            "above_ceiling_pct": round(float((clean["Actual"] > clean["Ceiling"]).mean() * 100), 1),
            "target_above_ceiling": 10.0,
        })
    return pd.DataFrame(rows)


def bonus_calibration(result, weekly):
    """Do the projected 100/300-yard bonus probabilities happen at the projected rate?

    The gamma tail is the least-inspected assumption in the model and it is worth three
    points a time, so it gets its own check rather than being buried in the total.
    """
    actual = offense_actuals(weekly)
    keys = ["season", "week", "player_id"]
    merged = result.merge(
        actual[keys + ["E_P_RUSH_100", "E_P_REC_100", "E_P_PASS_300"]].rename(columns={
            "E_P_RUSH_100": "hit_rush", "E_P_REC_100": "hit_rec",
            "E_P_PASS_300": "hit_pass"}), on=keys, how="inner")
    rows = []
    for label, projected, realised in (("rush 100", "E_P_RUSH_100", "hit_rush"),
                                       ("rec 100", "E_P_REC_100", "hit_rec"),
                                       ("pass 300", "E_P_PASS_300", "hit_pass")):
        clean = merged[[projected, realised]].dropna()
        clean = clean[clean[projected] > 0.01]
        if len(clean) < 50:
            continue
        rows.append({"bonus": label, "n": len(clean),
                     "projected_rate": round(float(clean[projected].mean()) * 100, 2),
                     "actual_rate": round(float(clean[realised].mean()) * 100, 2)})
    return pd.DataFrame(rows)
