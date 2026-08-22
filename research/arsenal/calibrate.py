"""Turn the tuned effect into constants a projection can actually use.

The sweep reports effects per standard deviation, which is the right unit for comparing
configurations and the wrong unit for shipping. `project_pitcher` needs to know how much
of a measured K-rate edge to pass through to its own K-rate estimate, in natural units,
so what is estimated here is a pass-through fraction: if the lineup strikes out 2 points
more often against this arsenal than it does against this hand generally, how much of
those 2 points show up in the starter's actual strikeout rate?

Everything is fit on 2024-2025 and checked once on 2026.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import PITCHER_CATEGORICAL, PITCHER_NUMERIC, pitcher_frame  # noqa: E402
from arsenal_engine import ArsenalConfig  # noqa: E402
from features import Study  # noqa: E402
from stats import design, incremental, ols  # noqa: E402

FIT_SEASONS = (2024, 2025)
HOLDOUT = 2026


def passthrough(frame, outcome, feature, controls, categoricals, label):
    """Coefficient in natural units, with cluster-robust error, on a given outcome."""
    frame = frame.dropna(subset=[outcome, feature] + list(controls)).copy()
    y = frame[outcome].to_numpy(dtype=float)
    X, names = design(frame, list(controls) + [feature], categoricals)
    result = ols(y, X, frame["game_pk"].to_numpy())
    index = names.index(feature)
    coefficient, se = result["beta"][index], result["se"][index]
    print(f"  {label:<26} {coefficient:+9.4f}  (se {se:.4f}, t {coefficient/se:+.2f})"
          f"  n={len(frame):,}")
    return float(coefficient)


def main():
    data = sys.argv[1] if len(sys.argv) > 1 else "data"
    config_path = sys.argv[2] if len(sys.argv) > 2 else None
    config = ArsenalConfig(**json.load(open(config_path))) if config_path else \
        ArsenalConfig(top_n=3, velo_tol=2.5, metric="k_rate", shrink=25.0, residualize=True)
    print(f"config: {config}\n")

    study = Study(data)
    fit, _ = study.score(config)
    frame = pitcher_frame(study, fit)
    frame["is_home"] = frame["is_home"].astype(str)
    # `score` is sign-flipped for K rate so that higher always means better for the hitter.
    # For a pitcher-facing constant the natural direction is the other way: a positive
    # k_edge should mean this lineup strikes out MORE against this arsenal than usual.
    frame["k_edge"] = -frame["lineup_fit"]
    frame["k_rate_actual"] = frame["K"] / frame["batters_faced"].replace(0, np.nan)

    print(f"k_edge distribution (fraction of PA):")
    print(frame["k_edge"].describe().round(4).to_string())

    fit_rows = frame[frame["season"].isin(FIT_SEASONS)]
    test_rows = frame[frame["season"] == HOLDOUT]

    print(f"\npass-through on {FIT_SEASONS}:")
    constants = {
        "k_rate": passthrough(fit_rows, "k_rate_actual", "k_edge", PITCHER_NUMERIC,
                              PITCHER_CATEGORICAL, "d(K rate)/d(edge)"),
        "ip": passthrough(fit_rows, "IP", "k_edge", PITCHER_NUMERIC,
                          PITCHER_CATEGORICAL, "d(IP)/d(edge)"),
        "er": passthrough(fit_rows, "ER", "k_edge", PITCHER_NUMERIC,
                          PITCHER_CATEGORICAL, "d(ER)/d(edge)"),
        "dk": passthrough(fit_rows, "pitcher_dk", "k_edge", PITCHER_NUMERIC,
                          PITCHER_CATEGORICAL, "d(DK)/d(edge)"),
    }

    print(f"\nsame coefficients re-estimated on {HOLDOUT} alone (never used for fitting):")
    for outcome, label in (("k_rate_actual", "d(K rate)/d(edge)"), ("IP", "d(IP)/d(edge)"),
                           ("ER", "d(ER)/d(edge)"), ("pitcher_dk", "d(DK)/d(edge)")):
        passthrough(test_rows, outcome, "k_edge", PITCHER_NUMERIC, PITCHER_CATEGORICAL,
                    label)

    print("\nwhat the constants imply, at the observed spread of k_edge:")
    spread = frame["k_edge"].std()
    for name, value in constants.items():
        print(f"  {name:<8} 1 SD ({spread:.4f}) of edge -> {value * spread:+.4f}")

    print("\nDK points by k_edge decile (controls partialled out):")
    from stats import decile_table
    table = decile_table(frame, "pitcher_dk", "k_edge", PITCHER_NUMERIC,
                         PITCHER_CATEGORICAL)
    for _, row in table.iterrows():
        print(f"  d{int(row['bucket'])+1:<2} n={int(row['n']):>5}  "
              f"edge {row['feature']:+.4f}  adjusted DK {row['outcome']:6.2f}")

    with open(os.path.join(data, "constants.json"), "w") as handle:
        json.dump({"config": {k: (None if v == float("inf") else v)
                              for k, v in config.__dict__.items()},
                   "passthrough": constants, "k_edge_sd": float(spread)}, handle, indent=2)
    print(f"\nwrote {data}/constants.json")


if __name__ == "__main__":
    main()
