"""Is the starter-side result real, or is it the pitcher's identity in disguise?

The worry: a pitcher with an unusual arsenal -- an extreme sinkerballer, a soft-tossing
junkballer -- will draw systematically low lineup-fit numbers every time out. If such
pitchers are also better than their ERA suggests, "lineups that fit badly let the starter
score" is really "good pitchers are good", and the feature is useless for choosing between
tonight's starters.

The test is a pitcher fixed effect. With one dummy per pitcher, the only variation left is
*within* a pitcher: across his own starts, does he do better on the nights the lineup fits
him worse? That is the question DFS actually asks.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arsenal_engine import ArsenalConfig  # noqa: E402
from analyze import (HITTER_CATEGORICAL, HITTER_NUMERIC, PITCHER_CATEGORICAL,  # noqa: E402
                     PITCHER_NUMERIC, hitter_frame, pitcher_frame)
from features import Study  # noqa: E402
from stats import decile_table, incremental, walk_forward  # noqa: E402


def line(label, result):
    if result is None:
        print(f"  {label:<40} --")
        return
    stars = "***" if abs(result["t"]) > 2.58 else ("**" if abs(result["t"]) > 1.96 else "")
    print(f"  {label:<40} n={result['n']:>6,}  coef={result['coef']:+.4f}"
          f"  t={result['t']:+6.2f} {stars:<3}  dR2={result['dr2']:+.5f}")


def main():
    data = sys.argv[1] if len(sys.argv) > 1 else "data"
    study = Study(data)
    config = ArsenalConfig(top_n=3, velo_tol=2.5, metric="ops", shrink=150.0,
                           residualize=True)
    fit, _ = study.score(config)
    hitters = hitter_frame(study, fit)
    pitchers = pitcher_frame(study, fit)
    pitchers["pitcher_id"] = pitchers["pitcher"].astype(str)
    pitchers["opp_team"] = np.where(pitchers["pitch_team"] == pitchers["home_team"],
                                    pitchers["away_team"], pitchers["home_team"])

    print("\n### STARTER SIDE: lineup arsenal fit -> starter DK points")
    line("baseline controls", incremental(
        pitchers, "pitcher_dk", PITCHER_NUMERIC, PITCHER_CATEGORICAL, "lineup_fit"))
    line("+ opposing-team fixed effects", incremental(
        pitchers, "pitcher_dk", PITCHER_NUMERIC,
        PITCHER_CATEGORICAL + ["opp_team"], "lineup_fit"))
    line("+ PITCHER fixed effects", incremental(
        pitchers, "pitcher_dk", PITCHER_NUMERIC,
        PITCHER_CATEGORICAL + ["pitcher_id"], "lineup_fit"))
    line("+ pitcher AND opposing-team FE", incremental(
        pitchers, "pitcher_dk", PITCHER_NUMERIC,
        PITCHER_CATEGORICAL + ["pitcher_id", "opp_team"], "lineup_fit"))
    real = pitchers[pitchers["opener"] == 0]
    line("real starts only (IP > 2)", incremental(
        real, "pitcher_dk", PITCHER_NUMERIC, PITCHER_CATEGORICAL, "lineup_fit"))

    print("\n  components of the starter's line (baseline controls):")
    for outcome in ("IP", "K", "ER", "H", "BB"):
        line(f"    {outcome}", incremental(
            pitchers, outcome, PITCHER_NUMERIC, PITCHER_CATEGORICAL, "lineup_fit"))

    print("\n  walk-forward: fit 2024-25, score 2026")
    forward = walk_forward(pitchers, "pitcher_dk", PITCHER_NUMERIC, PITCHER_CATEGORICAL,
                           "lineup_fit", (2024, 2025), 2026)
    if forward:
        print(f"    n_train={forward['n_train']:,}  n_test={forward['n_test']:,}")
        print(f"    RMSE   base {forward['rmse_base']:.4f} -> full {forward['rmse_full']:.4f}"
              f"  ({forward['drmse']:+.4f})")
        print(f"    rank r base {forward['spearman_base']:.4f} -> full "
              f"{forward['spearman_full']:.4f}  ({forward['dspearman']:+.4f})")

    print("\n  starter DK by decile of lineup fit (controls partialled out):")
    table = decile_table(pitchers, "pitcher_dk", "lineup_fit", PITCHER_NUMERIC,
                         PITCHER_CATEGORICAL)
    for _, row in table.iterrows():
        print(f"    d{int(row['bucket'])+1:<2} n={int(row['n']):>5}  "
              f"fit {row['feature']:+.4f}  adjusted DK {row['outcome']:6.2f}  "
              f"raw DK {row['raw_outcome']:6.2f}")

    print("\n### HITTER SIDE: own arsenal fit -> own DK points")
    line("baseline controls", incremental(
        hitters, "hitter_dk", HITTER_NUMERIC, HITTER_CATEGORICAL, "score"))
    line("+ batter fixed effects", incremental(
        hitters, "hitter_dk", HITTER_NUMERIC,
        HITTER_CATEGORICAL + ["batter"], "score"))
    line("wOBA vs starter, batter FE", incremental(
        hitters, "woba_vs_sp", HITTER_NUMERIC,
        HITTER_CATEGORICAL + ["batter"], "score"))
    enough = hitters[hitters["fit_denom"] >= 40]
    line("DK, only fits with >=40 PA history", incremental(
        enough, "hitter_dk", HITTER_NUMERIC, HITTER_CATEGORICAL, "score"))

    print("\n  hitter DK by decile of arsenal fit (controls partialled out):")
    table = decile_table(hitters, "hitter_dk", "score", HITTER_NUMERIC, HITTER_CATEGORICAL)
    for _, row in table.iterrows():
        print(f"    d{int(row['bucket'])+1:<2} n={int(row['n']):>6}  "
              f"fit {row['feature']:+.4f}  adjusted DK {row['outcome']:6.2f}  "
              f"raw DK {row['raw_outcome']:6.2f}")

    print("\n  walk-forward: fit 2024-25, score 2026")
    forward = walk_forward(hitters, "hitter_dk", HITTER_NUMERIC, HITTER_CATEGORICAL,
                           "score", (2024, 2025), 2026)
    if forward:
        print(f"    RMSE   base {forward['rmse_base']:.4f} -> full {forward['rmse_full']:.4f}"
              f"  ({forward['drmse']:+.4f})")
        print(f"    rank r base {forward['spearman_base']:.4f} -> full "
              f"{forward['spearman_full']:.4f}  ({forward['dspearman']:+.4f})")


if __name__ == "__main__":
    main()
