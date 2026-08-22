"""Does the feature order tonight's starters correctly?

Predicting DK points and ranking pitchers against each other are different bars, and the
second is the one a lineup builder actually has to clear. A feature can lift R-squared by
explaining variation that is mostly *between* slates -- good pitchers pitch on Tuesdays too
-- and still never change which arm gets picked on any given night. So everything here is
computed strictly within a slate.

Four readings, in increasing order of how much they resemble a real decision:

* **Pairwise accuracy.** Of all pairs of starters on the same slate, how often does the
  model put the higher scorer first? 50% is a coin flip. This is the cleanest single
  number for "predicts the order of finish".
* **Per-slate Spearman.** Rank correlation inside each slate, averaged over slates.
* **Top-pick score.** What the model's first, second and third choice actually scored.
  This is the money metric and also the noisiest, since it rests on one arm per slate.
* **Decision changes.** How often adding the feature moves the top pick at all, and what
  happens to the score when it does. A feature that never changes the pick cannot help,
  whatever its t-statistic says.

Models are fit walk-forward -- 2024 -> 2025, 2024-25 -> 2026 -- so no slate is ranked by a
model that has seen it.
"""

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import PITCHER_NUMERIC, pitcher_frame  # noqa: E402
from arsenal_engine import ArsenalConfig  # noqa: E402
from features import Study  # noqa: E402
from stats import design  # noqa: E402

# Season is dropped: a walk-forward model has never seen the test season's dummy, so it
# would contribute nothing and break the design matrix alignment.
RANK_CATEGORICAL = ["is_home", "home_team"]
MIN_SLATE = 6           # starts on a date before it counts as a slate worth ranking
FOLDS = [((2024,), 2025), ((2024, 2025), 2026)]


def walk_forward_predictions(frame, controls, categoricals, extra=None):
    """Out-of-sample predicted DK points for every start in a test season."""
    extra = extra or []
    pieces = []
    for train_seasons, test_season in FOLDS:
        subset = frame.dropna(subset=["pitcher_dk"] + list(controls) + list(extra)).copy()
        train = subset[subset["season"].isin(train_seasons)]
        test = subset[subset["season"] == test_season]
        if len(train) < 500 or len(test) < 200:
            continue
        combined = pd.concat([train, test], ignore_index=True)
        X, _ = design(combined, list(controls) + list(extra), categoricals)
        split = len(train)
        beta, *_ = np.linalg.lstsq(X[:split], train["pitcher_dk"].to_numpy(dtype=float),
                                   rcond=None)
        out = test[["game_date", "season", "pitcher", "pitcher_dk"]].copy()
        out["pred"] = X[split:] @ beta
        pieces.append(out)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def pairwise_accuracy(frame, score_col, outcome_col="pitcher_dk", group="game_date"):
    """Share of same-slate pairs the ranking gets right. Ties in the outcome are dropped."""
    right = total = 0
    for _, slate in frame.groupby(group, observed=True):
        if len(slate) < 2:
            continue
        scores = slate[score_col].to_numpy(dtype=float)
        actual = slate[outcome_col].to_numpy(dtype=float)
        # All pairs at once: sign agreement between the two difference matrices.
        ds = scores[:, None] - scores[None, :]
        da = actual[:, None] - actual[None, :]
        upper = np.triu(np.ones_like(ds, dtype=bool), k=1)
        keep = upper & (da != 0) & (ds != 0)
        right += int(((ds > 0) == (da > 0))[keep].sum())
        total += int(keep.sum())
    return (right / total if total else np.nan), total


def slate_spearman(frame, score_col, outcome_col="pitcher_dk", group="game_date"):
    values, sizes = [], []
    for _, slate in frame.groupby(group, observed=True):
        if len(slate) < MIN_SLATE:
            continue
        rho = slate[score_col].corr(slate[outcome_col], method="spearman")
        if pd.notna(rho):
            values.append(rho)
            sizes.append(len(slate))
    if not values:
        return np.nan, 0
    return float(np.average(values, weights=sizes)), len(values)


def top_pick_scores(frame, score_col, outcome_col="pitcher_dk", group="game_date", top=3):
    """Mean actual DK of the model's top choices, and the slate mean for reference."""
    picks = {k: [] for k in range(1, top + 1)}
    slate_means, slate_max = [], []
    for _, slate in frame.groupby(group, observed=True):
        if len(slate) < MIN_SLATE:
            continue
        ranked = slate.sort_values(score_col, ascending=False)
        for k in range(1, top + 1):
            picks[k].append(float(ranked[outcome_col].iloc[:k].mean()))
        slate_means.append(float(slate[outcome_col].mean()))
        slate_max.append(float(slate[outcome_col].max()))
    return ({k: float(np.mean(v)) for k, v in picks.items()},
            float(np.mean(slate_means)), float(np.mean(slate_max)), len(slate_means))


def decision_changes(frame, base_col, full_col, outcome_col="pitcher_dk",
                     group="game_date"):
    """How often the feature moves the top pick, and what it is worth when it does."""
    changed, deltas = 0, []
    slates = 0
    for _, slate in frame.groupby(group, observed=True):
        if len(slate) < MIN_SLATE:
            continue
        slates += 1
        base_pick = slate.loc[slate[base_col].idxmax()]
        full_pick = slate.loc[slate[full_col].idxmax()]
        if base_pick["pitcher"] != full_pick["pitcher"]:
            changed += 1
            deltas.append(float(full_pick[outcome_col] - base_pick[outcome_col]))
    return {
        "slates": slates,
        "changed": changed,
        "changed_pct": 100 * changed / slates if slates else np.nan,
        "mean_delta": float(np.mean(deltas)) if deltas else np.nan,
        "win_rate": 100 * float(np.mean([d > 0 for d in deltas])) if deltas else np.nan,
    }


def evaluate(frame, features, label, controls=None, categoricals=None):
    """Rank every test-season slate with and without `features`, and report the four reads."""
    controls = list(controls if controls is not None else PITCHER_NUMERIC)
    categoricals = list(categoricals if categoricals is not None else RANK_CATEGORICAL)

    base = walk_forward_predictions(frame, controls, categoricals)
    full = walk_forward_predictions(frame, controls, categoricals, extra=features)
    if base.empty or full.empty:
        print(f"  {label}: not enough data")
        return None
    keys = ["game_date", "pitcher"]
    merged = base.merge(full[keys + ["pred"]], on=keys, how="inner",
                        suffixes=("_base", "_full"))
    # Slates are ranked, so a date with one start tells us nothing.
    counts = merged.groupby("game_date", observed=True)["pitcher"].transform("size")
    merged = merged[counts >= MIN_SLATE]

    print(f"\n  --- {label} ---")
    print(f"  {len(merged):,} starts over "
          f"{merged['game_date'].nunique():,} slates (>= {MIN_SLATE} starts)")

    rows = []
    for name, column in (("baseline", "pred_base"), ("+ feature", "pred_full")):
        accuracy, pairs = pairwise_accuracy(merged, column)
        rho, slates = slate_spearman(merged, column)
        picks, slate_mean, slate_best, _ = top_pick_scores(merged, column)
        rows.append((name, accuracy, pairs, rho, picks))
        print(f"    {name:<10} pairwise {accuracy*100:5.2f}%   slate rho {rho:+.4f}   "
              f"top1 {picks[1]:5.2f}  top2 {picks[2]:5.2f}  top3 {picks[3]:5.2f}")
    print(f"    {'slate':<10} mean {slate_mean:5.2f}   best available {slate_best:5.2f}")
    print(f"    delta      pairwise {(rows[1][1]-rows[0][1])*100:+.2f} pts   "
          f"slate rho {rows[1][3]-rows[0][3]:+.4f}   "
          f"top1 {rows[1][4][1]-rows[0][4][1]:+.2f}  "
          f"top3 {rows[1][4][3]-rows[0][4][3]:+.2f}")

    change = decision_changes(merged, "pred_base", "pred_full")
    print(f"    top pick changed on {change['changed']}/{change['slates']} slates "
          f"({change['changed_pct']:.1f}%)", end="")
    if change["changed"]:
        print(f"; when it did, {change['mean_delta']:+.2f} DK "
              f"({change['win_rate']:.0f}% better)")
    else:
        print()
    return merged


def bootstrap_pairwise_delta(merged, draws=400, seed=11):
    """Slate-level bootstrap on the pairwise-accuracy gain.

    Pairs inside a slate are not independent -- each starter appears in every pair on his
    own slate -- so resampling pairs would understate the error badly. Slates are the
    independent unit, so slates are what gets resampled.
    """
    rng = np.random.default_rng(seed)
    slates = merged["game_date"].unique()
    lookup = {date: group for date, group in merged.groupby("game_date", observed=True)}
    deltas = []
    for _ in range(draws):
        picked = rng.choice(slates, size=len(slates), replace=True)
        sample = pd.concat([lookup[d] for d in picked], ignore_index=True)
        sample["_slate"] = np.repeat(np.arange(len(picked)),
                                     [len(lookup[d]) for d in picked])
        base, _ = pairwise_accuracy(sample, "pred_base", group="_slate")
        full, _ = pairwise_accuracy(sample, "pred_full", group="_slate")
        deltas.append(full - base)
    deltas = np.array(deltas)
    return float(deltas.mean() * 100), float(np.percentile(deltas, 2.5) * 100), \
        float(np.percentile(deltas, 97.5) * 100)


def raw_feature_read(frame, feature, label):
    """The feature on its own, against what the baseline model could not explain.

    A feature that ranks pitchers has to order the *residual*, not the pitchers -- ordering
    pitchers is what the baseline is for.
    """
    base = walk_forward_predictions(frame, PITCHER_NUMERIC, RANK_CATEGORICAL)
    if base.empty:
        return
    merged = base.merge(frame[["game_date", "pitcher", feature]], on=["game_date", "pitcher"],
                        how="inner").dropna(subset=[feature])
    counts = merged.groupby("game_date", observed=True)["pitcher"].transform("size")
    merged = merged[counts >= MIN_SLATE].copy()
    merged["residual"] = merged["pitcher_dk"] - merged["pred"]
    rho, slates = slate_spearman(merged, feature, outcome_col="residual")
    accuracy, pairs = pairwise_accuracy(merged, feature, outcome_col="residual")
    print(f"\n  {label} alone vs the baseline's residual:")
    print(f"    within-slate rho {rho:+.4f} over {slates:,} slates   "
          f"pairwise {accuracy*100:.2f}% of {pairs:,} pairs")


def main():
    data = sys.argv[1] if len(sys.argv) > 1 else "data"
    config_path = sys.argv[2] if len(sys.argv) > 2 else None
    config = ArsenalConfig(**json.load(open(config_path))) if config_path else \
        ArsenalConfig(top_n=3, min_usage=0.05, weight_power=1.5, velo_tol=2.5,
                      metric="k_rate", shrink=25.0, residualize=True)
    print(f"config: {config}")

    study = Study(data)
    fit, _ = study.score(config)
    frame = pitcher_frame(study, fit)
    frame["is_home"] = frame["is_home"].astype(str)
    frame["k_edge"] = -frame["lineup_fit"]

    type_path = os.path.join(data, "vs_type.parquet")
    if os.path.exists(type_path):
        vs_type = pd.read_parquet(type_path)
        frame = frame.merge(vs_type, on="start_id", how="left")
        frame["log_type_pa"] = np.log1p(frame["type_pa"].fillna(0))
        print(f"\nvs-type available on {frame['type_k_edge'].notna().sum():,} of "
              f"{len(frame):,} starts (median comp PA {frame['type_pa'].median():.0f})")
        print(f"correlation between the two features: "
              f"{frame['k_edge'].corr(frame['type_k_edge']):+.3f}")
    else:
        vs_type = None

    print("\n=== Does it reorder a slate? ===")
    candidates = [(["k_edge"], "lineup arsenal K edge")]
    if vs_type is not None:
        candidates += [
            (["type_k_edge", "log_type_pa"], "vs type, K rate"),
            (["type_woba_edge", "log_type_pa"], "vs type, wOBA (closest to the panel)"),
            (["type_k_edge", "type_woba_edge", "log_type_pa"], "vs type, both metrics"),
            (["k_edge", "type_k_edge", "type_woba_edge", "log_type_pa"], "everything"),
        ]
    for features, label in candidates:
        merged = evaluate(frame, features, label)
        if merged is not None:
            mean, low, high = bootstrap_pairwise_delta(merged)
            print(f"    bootstrap pairwise gain {mean:+.2f} pts "
                  f"[{low:+.2f}, {high:+.2f}] over 95% of slate resamples")

    raw_feature_read(frame, "k_edge", "K edge")
    if vs_type is not None:
        raw_feature_read(frame, "type_k_edge", "vs-type K edge")
        raw_feature_read(frame, "type_woba_edge", "vs-type wOBA edge")

        # Ranking is a harder bar than prediction, so a feature can fail above and still
        # carry signal. Checking that separately keeps "does not help pick pitchers" from
        # being reported as "measures nothing".
        from stats import incremental
        print("\n  effect on starter DK points (not ranking), full controls:")
        for column in ("k_edge", "type_k_edge", "type_woba_edge"):
            result = incremental(frame, "pitcher_dk", PITCHER_NUMERIC + ["log_type_pa"],
                                 RANK_CATEGORICAL + ["season"], column)
            if result:
                print(f"    {column:<16} n={result['n']:>6,}  coef={result['coef']:+.4f}"
                      f"  t={result['t']:+6.2f}  dR2={result['dr2']:+.5f}")


if __name__ == "__main__":
    main()
