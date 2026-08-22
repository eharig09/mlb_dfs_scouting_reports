"""Search the space of ways to measure an arsenal, and pick one honestly.

Selection happens on 2024-2025 only. 2026 is never looked at while choosing, and is then
reported once for the winner. Without that split, a sweep over this many knobs would find
a "best" configuration whose margin is entirely the sweep itself.

The objective is the starter-side effect, because that is where the earlier pass found the
signal: the lineup-level aggregate averages nine noisy hitter numbers and what survives is
usable, while the individual hitter number is not. The hitter-side effect on wOBA against
the starter is carried along as a secondary read, to check that a configuration winning on
the pitcher side is measuring the matchup rather than exploiting some lineup-level artifact.
"""

import itertools
import os
import sys
import time
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import (HITTER_CATEGORICAL, HITTER_NUMERIC, PITCHER_CATEGORICAL,  # noqa: E402
                     PITCHER_NUMERIC, hitter_frame, pitcher_frame)
from arsenal_engine import ArsenalConfig  # noqa: E402
from features import Study  # noqa: E402
from stats import incremental  # noqa: E402

SELECT_SEASONS = (2024, 2025)
HOLDOUT_SEASON = 2026

# Shrinkage is in denominator units, and the denominator differs by metric: plate
# appearances for OPS/wOBA, batted balls for xwOBA, pitches for RV/100 and whiff rate.
# Each metric therefore gets its own ladder, scaled to what a median fit actually rests on.
SHRINK_LADDER = {
    "ops": [0, 25, 75, 150, 400],
    "woba": [0, 25, 75, 150, 400],
    "xwoba": [0, 10, 30, 60, 150],
    "rv100": [0, 100, 300, 800, 2000],
    "whiff": [0, 50, 150, 400, 1000],
    "k_rate": [0, 25, 75, 150, 400],
}

BASE = ArsenalConfig(top_n=3, velo_tol=2.5, weight_power=1.0, metric="ops",
                     shrink=150.0, residualize=True, lookback_days=730,
                     arsenal_lookback_days=365, match_by_hand=True)


def evaluate(study, config, seasons):
    """One config's effect sizes, restricted to the given seasons."""
    fit, arsenals = study.score(config)
    hitters = hitter_frame(study, fit)
    pitchers = pitcher_frame(study, fit)
    hitters = hitters[hitters["season"].isin(seasons)]
    pitchers = pitchers[pitchers["season"].isin(seasons)]

    starter = incremental(pitchers, "pitcher_dk", PITCHER_NUMERIC, PITCHER_CATEGORICAL,
                          "lineup_fit")
    matchup = incremental(hitters, "woba_vs_sp", HITTER_NUMERIC, HITTER_CATEGORICAL,
                          "score")
    hitter_dk = incremental(hitters, "hitter_dk", HITTER_NUMERIC, HITTER_CATEGORICAL,
                            "score")
    row = dict(asdict(config))
    row.update({
        "sp_t": starter["t"] if starter else np.nan,
        "sp_coef": starter["coef"] if starter else np.nan,
        "sp_dr2": starter["dr2"] if starter else np.nan,
        "sp_n": starter["n"] if starter else 0,
        "hit_woba_t": matchup["t"] if matchup else np.nan,
        "hit_dk_t": hitter_dk["t"] if hitter_dk else np.nan,
        "median_denom": float(hitters["fit_denom"].median()),
        "arsenal_pitches": round(len(arsenals) / max(1, study.starts.shape[0]), 2),
    })
    return row


def sweep(study, base, axis, values, results, seen):
    print(f"\n-- {axis} --", flush=True)
    rows = []
    for value in values:
        config = replace(base, **{axis: value})
        key = tuple(sorted(asdict(config).items()))
        if key in seen:
            row = seen[key]
        else:
            started = time.time()
            row = evaluate(study, config, SELECT_SEASONS)
            row["secs"] = round(time.time() - started, 1)
            seen[key] = row
            results.append(row)
        rows.append((value, row))
        print(f"   {axis}={str(value):<8} sp_t={row['sp_t']:+6.2f}  "
              f"sp_coef={row['sp_coef']:+.3f}  sp_dR2={row['sp_dr2']:+.5f}  "
              f"hit_wOBA_t={row['hit_woba_t']:+5.2f}  denom={row['median_denom']:.0f}",
              flush=True)
    # Bigger |t| on the starter side wins; the sign must be negative (a lineup that fits
    # the arsenal well should cost the pitcher points), so a positive t is a failed config.
    best = min(rows, key=lambda item: item[1]["sp_t"])
    print(f"   -> best {axis}={best[0]}")
    return best[0]


def main():
    data = sys.argv[1] if len(sys.argv) > 1 else "data"
    out = sys.argv[2] if len(sys.argv) > 2 else "tuning.csv"
    study = Study(data)
    results, seen = [], {}
    base = BASE

    print(f"\n=== coordinate sweep, selecting on {SELECT_SEASONS} ===", flush=True)
    base = replace(base, metric=sweep(study, base, "metric",
                                      list(SHRINK_LADDER), results, seen))
    base = replace(base, shrink=sweep(study, base, "shrink",
                                      SHRINK_LADDER[base.metric], results, seen))
    base = replace(base, top_n=sweep(study, base, "top_n", [2, 3, 4, 5, 6, 8],
                                     results, seen))
    base = replace(base, velo_tol=sweep(study, base, "velo_tol",
                                        [1.0, 2.5, 4.0, 6.0, float("inf")], results, seen))
    base = replace(base, weight_power=sweep(study, base, "weight_power",
                                            [0.0, 0.5, 1.0, 1.5, 2.0], results, seen))
    base = replace(base, lookback_days=sweep(study, base, "lookback_days",
                                             [365, 730, 1095, 10000], results, seen))
    base = replace(base, arsenal_lookback_days=sweep(
        study, base, "arsenal_lookback_days", [45, 90, 180, 365, 10000], results, seen))
    base = replace(base, match_by_hand=sweep(study, base, "match_by_hand",
                                             [True, False], results, seen))
    base = replace(base, min_usage=sweep(study, base, "min_usage",
                                         [0.0, 0.05, 0.10], results, seen))
    # Shrinkage interacts with everything upstream, so it gets a second pass once the rest
    # of the configuration has moved.
    base = replace(base, shrink=sweep(study, base, "shrink",
                                      SHRINK_LADDER[base.metric], results, seen))

    print(f"\n=== selected configuration ===")
    for key, value in asdict(base).items():
        print(f"  {key:<22} {value}")

    frame = pd.DataFrame(results)
    frame.to_csv(out, index=False)
    print(f"\nwrote {out} ({len(frame)} configs)")

    print(f"\n=== holdout confirmation on {HOLDOUT_SEASON} ===")
    for label, config in (("as-shipped", ArsenalConfig(top_n=3, velo_tol=2.5, metric="ops",
                                                       shrink=0.0, residualize=False)),
                          ("tuned", base)):
        row = evaluate(study, config, (HOLDOUT_SEASON,))
        print(f"  {label:<12} sp_t={row['sp_t']:+6.2f}  sp_coef={row['sp_coef']:+.3f}"
              f"  sp_dR2={row['sp_dr2']:+.5f}  n={row['sp_n']:,}"
              f"  hit_wOBA_t={row['hit_woba_t']:+5.2f}")


if __name__ == "__main__":
    main()
