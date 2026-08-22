"""The two questions, asked properly.

1. Does a hitter's arsenal fit predict his DK points, over and above who he is, where he
   is batting, where the game is, and who is pitching?
2. Does a lineup that fits an arsenal badly hand the starter more DK points, over and
   above how good that lineup and that starter already are?
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arsenal_engine import ArsenalConfig  # noqa: E402
from features import Study, lineup_fit  # noqa: E402
from stats import decile_table, incremental, walk_forward  # noqa: E402

# Controls for the hitter model. `baseline` is the hitter's own line against this hand over
# the same window the fit is measured on -- without it, an arsenal score is just a noisier
# restatement of "this guy can hit", and every result below would be an artifact.
HITTER_NUMERIC = ["hitter_dk_pg", "hitter_pa_pg", "hitter_k_rate", "hitter_hr_rate",
                  "sp_dk_pg", "sp_k_rate", "sp_era", "sp_whip",
                  "baseline", "log_base_denom", "log_fit_denom"]
HITTER_CATEGORICAL = ["order_slot", "platoon", "home_team", "season"]

PITCHER_NUMERIC = ["sp_dk_pg", "sp_ip_pg", "sp_k_rate", "sp_era", "sp_whip",
                   "lineup_baseline", "lineup_dk_pg", "lineup_denom"]
PITCHER_CATEGORICAL = ["is_home", "home_team", "season"]


def hitter_frame(study, fit):
    frame = study.pairs.merge(fit, on="pair_id", how="left")
    frame["log_fit_denom"] = np.log1p(frame["fit_denom"])
    frame["log_base_denom"] = np.log1p(frame["base_denom"])
    return frame


def pitcher_frame(study, fit):
    slim = study.pairs[["pair_id", "start_id", "order_slot", "hitter_dk_pg"]]
    aggregate = lineup_fit(slim, fit[["pair_id", "score", "baseline", "fit_denom"]])
    quality = (slim.groupby("start_id", observed=True)["hitter_dk_pg"]
               .mean().rename("lineup_dk_pg").reset_index())
    frame = study.starts.merge(aggregate, on="start_id", how="left")
    frame = frame.merge(quality, on="start_id", how="left")
    return frame


def report(title, result):
    if result is None:
        print(f"  {title}: not enough data")
        return
    stars = "***" if abs(result["t"]) > 2.58 else ("**" if abs(result["t"]) > 1.96 else "")
    print(f"  {title:<34} n={result['n']:>7,}  coef={result['coef']:+.4f}"
          f"  t={result['t']:+6.2f} {stars:<3}  dR2={result['dr2']:+.5f}")


def main():
    data = sys.argv[1] if len(sys.argv) > 1 else "data"
    study = Study(f"{data}")

    # The report's settings today, plus the residualized variant this study argues for.
    configs = {
        "as-shipped (top3, +/-2.5mph, OPS, raw)":
            ArsenalConfig(top_n=3, velo_tol=2.5, metric="ops", shrink=0.0,
                          residualize=False),
        "same, residualized vs own baseline":
            ArsenalConfig(top_n=3, velo_tol=2.5, metric="ops", shrink=0.0,
                          residualize=True),
        "residualized + shrunk (150 PA)":
            ArsenalConfig(top_n=3, velo_tol=2.5, metric="ops", shrink=150.0,
                          residualize=True),
        "RV/100, residualized + shrunk":
            ArsenalConfig(top_n=3, velo_tol=2.5, metric="rv100", shrink=400.0,
                          residualize=True),
        "xwOBA, residualized + shrunk":
            ArsenalConfig(top_n=3, velo_tol=2.5, metric="xwoba", shrink=60.0,
                          residualize=True),
    }

    for label, config in configs.items():
        fit, _ = study.score(config)
        hitters = hitter_frame(study, fit)
        print(f"\n=== {label} ===")
        print(f"  fit denom: median {hitters['fit_denom'].median():.0f} "
              f"| score sd {hitters['score'].std():.4f}")

        print("\n  HITTER DK points")
        report("raw (no controls)", incremental(
            hitters, "hitter_dk", ["hitter_dk_pg"], [], "score"))
        report("full controls", incremental(
            hitters, "hitter_dk", HITTER_NUMERIC, HITTER_CATEGORICAL, "score"))
        print("\n  HITTER wOBA vs the starter only")
        report("full controls", incremental(
            hitters, "woba_vs_sp", HITTER_NUMERIC, HITTER_CATEGORICAL, "score"))

        pitchers = pitcher_frame(study, fit)
        print("\n  STARTER DK points (vs lineup fit)")
        report("full controls", incremental(
            pitchers, "pitcher_dk", PITCHER_NUMERIC, PITCHER_CATEGORICAL,
            "lineup_fit", cluster_col="game_pk"))


if __name__ == "__main__":
    main()
