"""Shared loader: build the history tables once, score any config against them cheaply."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arsenal_engine import (  # noqa: E402
    ArsenalConfig, add_counters, batter_fit, build_batter_baseline, build_batter_history,
    build_pitcher_mix, load_pitches, pitcher_arsenals,
)


class Study:
    """Everything a config needs, loaded once."""

    def __init__(self, data_dir, seasons=(2024, 2025, 2026), verbose=True):
        self.data_dir = data_dir
        if verbose:
            print("  loading pitches...", flush=True)
        pitches = load_pitches(data_dir, seasons)
        counters = add_counters(pitches)
        if verbose:
            print("  building batter history...", flush=True)
        self.history = build_batter_history(pitches, counters)
        self.baseline = build_batter_baseline(pitches, counters)
        if verbose:
            print("  building pitcher mix...", flush=True)
        self.mix = build_pitcher_mix(pitches, counters)
        self.pairs = pd.read_parquet(f"{data_dir}/pairs.parquet")
        self.starts = pd.read_parquet(f"{data_dir}/starts.parquet")
        del pitches, counters
        if verbose:
            print(f"  history {len(self.history):,} rows | baseline {len(self.baseline):,}"
                  f" | mix {len(self.mix):,}", flush=True)

    def score(self, config):
        """Arsenal-fit columns for every pair, under one config."""
        starts = self.starts[["start_id", "pitcher", "game_date"]]
        arsenals = pitcher_arsenals(starts, self.mix, config)
        pairs = self.pairs[["pair_id", "start_id", "batter", "p_throws", "game_date"]]
        fit = batter_fit(pairs, arsenals, self.history, self.baseline, config)
        return fit, arsenals


def lineup_fit(pairs, fit, weight_by_slot=True):
    """Aggregate the nine hitters' fit into one number per start.

    Slots are not equal: the leadoff hitter gets roughly 4.6 plate appearances and the
    ninth gets 3.9, so an unweighted mean overstates the bottom of the order. The weights
    are the league-average PA share by slot.
    """
    slot_pa = {1: 4.65, 2: 4.55, 3: 4.45, 4: 4.35, 5: 4.25,
               6: 4.15, 7: 4.05, 8: 3.95, 9: 3.85}
    frame = pairs.merge(fit, on="pair_id", how="left")
    frame["w"] = frame["order_slot"].map(slot_pa).fillna(4.2) if weight_by_slot else 1.0
    frame = frame[frame["score"].notna()]
    grouped = frame.groupby("start_id", observed=True)
    out = pd.DataFrame({
        "lineup_fit": grouped.apply(
            lambda g: np.average(g["score"], weights=g["w"]), include_groups=False),
        "lineup_baseline": grouped.apply(
            lambda g: np.average(g["baseline"], weights=g["w"]), include_groups=False),
        "lineup_n": grouped.size(),
        "lineup_denom": grouped["fit_denom"].mean(),
    }).reset_index()
    return out
