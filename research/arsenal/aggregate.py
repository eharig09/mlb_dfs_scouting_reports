"""How should nine hitter fits become one number about the lineup?

The mean is the obvious choice and probably not the right one. A starter's night is not
decided by the average hitter he faces -- it is decided by whether anyone in the lineup
can hurt him. So the candidates here include order-weighted and unweighted means, the
worst and best individual fits, the share of the lineup that fits well, and versions that
weight by how much history each hitter's number rests on (an unshrunk fit built on nine
plate appearances deserves less say than one built on ninety).
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import PITCHER_CATEGORICAL, PITCHER_NUMERIC  # noqa: E402
from arsenal_engine import ArsenalConfig  # noqa: E402
from features import Study  # noqa: E402
from stats import incremental  # noqa: E402

SLOT_PA = {1: 4.65, 2: 4.55, 3: 4.45, 4: 4.35, 5: 4.25,
           6: 4.15, 7: 4.05, 8: 3.95, 9: 3.85}


def aggregates(pairs, fit):
    """Every candidate lineup-level summary, one column each."""
    frame = pairs.merge(fit, on="pair_id", how="left")
    frame = frame[frame["score"].notna()].copy()
    frame["slot_w"] = frame["order_slot"].map(SLOT_PA).fillna(4.2)
    frame["denom_w"] = np.sqrt(frame["fit_denom"].clip(lower=0))
    grouped = frame.groupby("start_id", observed=True)

    def weighted(group, weight):
        w = group[weight].to_numpy()
        return float(np.average(group["score"], weights=w)) if w.sum() else np.nan

    out = pd.DataFrame({
        "agg_mean": grouped["score"].mean(),
        "agg_slot": grouped.apply(lambda g: weighted(g, "slot_w"), include_groups=False),
        "agg_denom": grouped.apply(lambda g: weighted(g, "denom_w"), include_groups=False),
        "agg_max": grouped["score"].max(),
        "agg_min": grouped["score"].min(),
        "agg_top3": grouped["score"].apply(lambda s: s.nlargest(3).mean()),
        "agg_median": grouped["score"].median(),
        "agg_share_pos": grouped["score"].apply(lambda s: float((s > 0).mean())),
        "agg_sd": grouped["score"].std(),
        "lineup_baseline": grouped.apply(lambda g: weighted(g, "slot_w") * 0
                                          + float(np.average(g["baseline"],
                                                             weights=g["slot_w"])),
                                          include_groups=False),
        "lineup_denom": grouped["fit_denom"].mean(),
        "lineup_n": grouped.size(),
    }).reset_index()
    return out


def main():
    data = sys.argv[1] if len(sys.argv) > 1 else "data"
    study = Study(data)
    config = ArsenalConfig(top_n=3, velo_tol=2.5, metric="ops", shrink=150.0,
                           residualize=True)
    if len(sys.argv) > 2:
        import json
        config = ArsenalConfig(**json.load(open(sys.argv[2])))
    print(f"config: {config}\n")

    fit, _ = study.score(config)
    slim = study.pairs[["pair_id", "start_id", "order_slot", "hitter_dk_pg"]]
    table = aggregates(slim, fit[["pair_id", "score", "baseline", "fit_denom"]])
    quality = (slim.groupby("start_id", observed=True)["hitter_dk_pg"]
               .mean().rename("lineup_dk_pg").reset_index())
    frame = study.starts.merge(table, on="start_id", how="left").merge(
        quality, on="start_id", how="left")
    frame["is_home"] = frame["is_home"].astype(str)

    candidates = [c for c in table.columns if c.startswith("agg_")]
    print("lineup summary -> starter DK points (all seasons, full controls)")
    rows = []
    for column in candidates:
        result = incremental(frame, "pitcher_dk", PITCHER_NUMERIC, PITCHER_CATEGORICAL,
                             column)
        if result is None:
            continue
        rows.append((column, result))
        stars = "***" if abs(result["t"]) > 2.58 else ("**" if abs(result["t"]) > 1.96 else "")
        print(f"  {column:<16} n={result['n']:>6,}  coef={result['coef']:+.4f}"
              f"  t={result['t']:+6.2f} {stars:<3}  dR2={result['dr2']:+.5f}")

    print("\nbest two together (does the second add to the first?)")
    ranked = sorted(rows, key=lambda item: -abs(item[1]["t"]))
    first, second = ranked[0][0], ranked[1][0]
    combined = incremental(frame, "pitcher_dk", PITCHER_NUMERIC + [first],
                           PITCHER_CATEGORICAL, second)
    if combined:
        print(f"  {second} on top of {first}: coef={combined['coef']:+.4f}"
              f"  t={combined['t']:+6.2f}  dR2={combined['dr2']:+.5f}")


if __name__ == "__main__":
    main()
