"""Touchdown projection from where the opportunity happens, not from past touchdowns.

Why this is worth its own module
--------------------------------
Touchdowns are the least predictable part of a fantasy line — `nfl.projections` sets
`TD_SHRINK = 220`, roughly five times `EFFICIENCY_SHRINK`, precisely because a player's own
touchdown history barely carries forward. Red-zone *opportunity* does carry forward, and it
is the thing touchdowns are actually made of.

Measured on PFF fantasy splits, 2022-25 (Spearman, mean over three season pairs, ≥6 games):

    year-over-year stability          predicting NEXT year's receiving TDs
      recTarg        0.787              rzRecTarg     0.611   <- best
      rzRecTarg      0.734              recTarg       0.590
      ezRecTarg      0.668              ezRecTarg     0.557
      recTds         0.557              recTds        0.557   <- what you'd use naively
                                        rzRecTds      0.545

**Red-zone targets predict next season's receiving touchdowns better than touchdowns do.**
That is the case for this module.

It does *not* hold for rushing, and saying so matters more than the headline. Rushing
touchdowns are already stable because carry volume is stable, and every candidate lands in
the same place: rzRushTds 0.735, rushTds 0.731, i5RushCarries 0.726, rzRushCarries 0.722.
For backs this is a decomposition that makes a projection legible, not one that makes it
more accurate.

The tiers nest — do not add them
--------------------------------
Checked on all 559 rows of the 2025 file: end-zone targets are a subset of red-zone targets
(0 violations) and inside-five carries are a subset of red-zone carries (0 violations).
Converting each bucket and summing would count the same opportunity two and three times.
Each rate below therefore applies to its tier *minus the tier inside it*, and
`expected_receiving_tds` does that subtraction for you.

Red-zone work accounts for 91.7% of receiving touchdowns and 81.4% of rushing touchdowns.
The rest are long scores, which is why an open-field tier exists at all rather than the
non-red-zone opportunity being treated as scoreless.

The rates, fitted 2022-25
-------------------------
An end-zone target is worth 66 times an open-field one. That ratio is the entire argument
for tracking where a target came from:

    end zone target   0.4005 TD    (season spread 0.053)
    red zone target   0.1134       (0.018)
    open field target 0.0061       (0.002)
    inside-5 carry    0.3904       (0.026)
    red zone carry    0.0615       (0.020)
    open field carry  0.0057       (0.003)
"""

import os
import re
import unicodedata

import numpy as np
import pandas as pd

# Each rate applies to its tier with the inner tier removed. See the module docstring.
RECEIVING_TD_RATE = {
    "end_zone": 0.4005,
    "red_zone": 0.1134,
    "open_field": 0.0061,
}
RUSHING_TD_RATE = {
    "inside_five": 0.3904,
    "red_zone": 0.0615,
    "open_field": 0.0057,
}

# **The filenames carry no season and the numbering is not consistent between families.**
# Every one of these was identified by matching the file's own totals against nflverse
# season totals, not by reading the number: `receiving_summary` counts *up* with age while
# `fantasy-stats-receiving` uses the unnumbered file for the newest season and
# `defense_coverage_scheme` uses it for the oldest. Re-identify with
# `scripts/identify_pff_season` rather than guessing when new exports land.
PFF_FILES = {
    "fantasy_receiving": {
        2025: "pff/fantasy-stats-receiving.csv",
        2024: "pff/fantasy-stats-receiving (1).csv",
        2023: "pff/fantasy-stats-receiving (2).csv",
        2022: "pff/fantasy-stats-receiving (3).csv",
    },
    "fantasy_passing": {
        2025: "pff/fantasy-stats-passing.csv",
        2024: "pff/fantasy-stats-passing (1).csv",
        2023: "pff/fantasy-stats-passing (2).csv",
        2022: "pff/fantasy-stats-passing (3).csv",
    },
}

_RECEIVING_COLUMNS = [
    "player", "team", "position", "games",
    "recTarg", "recRec", "recTds", "recYds",
    "rzRecTarg", "rzRecRec", "rzRecTds",
    "ezRecTarg", "ezRecTds",
    "rushCarries", "rushYds", "rushTds",
    "rzRushCarries", "rzRushTds",
    "i5RushCarries", "i5RushTds",
]

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?")


def name_key(value):
    """Accent- and suffix-insensitive join key. PFF and DK disagree on both."""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = _SUFFIX.sub("", text.lower())
    return re.sub(r"[^a-z ]", "", text).strip()


def load_fantasy_receiving(season, root="."):
    """PFF receiving + rushing fantasy splits for one season, with a join key attached."""
    path = PFF_FILES["fantasy_receiving"].get(int(season))
    if path is None:
        raise KeyError(f"no PFF fantasy-receiving file mapped for {season}")
    frame = pd.read_csv(os.path.join(root, path))
    frame = frame.reindex(columns=[c for c in _RECEIVING_COLUMNS if c in frame.columns])
    for column in frame.columns:
        if column not in ("player", "team", "position"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["key"] = frame["player"].map(name_key)
    return frame


def _tier_split(total, outer, inner):
    """(inner, outer-without-inner, total-without-outer), each floored at zero.

    Floored because the inputs are independently rounded season aggregates and a player
    with one red-zone target and one end-zone target can otherwise produce a -0.0 middle
    tier, which then earns a small negative touchdown.
    """
    total = np.asarray(total, dtype=float)
    outer = np.asarray(outer, dtype=float)
    inner = np.asarray(inner, dtype=float)
    inner_only = np.clip(inner, 0.0, None)
    middle = np.clip(outer - inner, 0.0, None)
    open_field = np.clip(total - outer, 0.0, None)
    return inner_only, middle, open_field


def expected_receiving_tds(targets, red_zone_targets, end_zone_targets):
    """Expected receiving touchdowns from where the targets came from."""
    end_zone, red_zone, open_field = _tier_split(targets, red_zone_targets, end_zone_targets)
    return (end_zone * RECEIVING_TD_RATE["end_zone"]
            + red_zone * RECEIVING_TD_RATE["red_zone"]
            + open_field * RECEIVING_TD_RATE["open_field"])


def expected_rushing_tds(carries, red_zone_carries, inside_five_carries):
    """Expected rushing touchdowns from where the carries came from."""
    inside_five, red_zone, open_field = _tier_split(
        carries, red_zone_carries, inside_five_carries)
    return (inside_five * RUSHING_TD_RATE["inside_five"]
            + red_zone * RUSHING_TD_RATE["red_zone"]
            + open_field * RUSHING_TD_RATE["open_field"])


def red_zone_profile(season, min_games=4, root="."):
    """Per-game red-zone workload and the touchdown equity it implies, per player.

    `xTD` is what that workload is worth at league conversion rates; `TD oe` is how far the
    player's actual touchdowns ran above or below it. A large positive gap is the classic
    regression candidate — a player whose scoring outran the opportunities behind it.
    """
    frame = load_fantasy_receiving(season, root=root)
    frame = frame[frame["games"].fillna(0) >= min_games].copy()
    if frame.empty:
        return frame

    games = frame["games"].replace(0, np.nan)
    for column in ("rzRecTarg", "ezRecTarg", "rzRushCarries", "i5RushCarries",
                   "recTarg", "rushCarries"):
        if column in frame.columns:
            frame[f"{column}_pg"] = frame[column] / games

    frame["xRecTD"] = expected_receiving_tds(
        frame.get("recTarg", 0), frame.get("rzRecTarg", 0), frame.get("ezRecTarg", 0))
    frame["xRushTD"] = expected_rushing_tds(
        frame.get("rushCarries", 0), frame.get("rzRushCarries", 0),
        frame.get("i5RushCarries", 0))
    frame["xTD"] = frame["xRecTD"] + frame["xRushTD"]
    frame["TD"] = frame.get("recTds", 0).fillna(0) + frame.get("rushTds", 0).fillna(0)
    frame["TD_oe"] = frame["TD"] - frame["xTD"]
    frame["xTD_pg"] = frame["xTD"] / games
    return frame.sort_values("xTD", ascending=False).reset_index(drop=True)


def team_red_zone_share(profile, team_column="team"):
    """Each player's share of his own team's red-zone opportunity.

    Share rather than raw count, because the count is a joint statement about the player
    and about how often his offense reached the red zone at all. Share is the part that
    belongs to him, and it is what survives a change in team pace or quality.
    """
    if profile is None or profile.empty:
        return profile
    frame = profile.copy()
    for source, name in (("rzRecTarg", "rz_target_share"),
                         ("rzRushCarries", "rz_carry_share")):
        if source not in frame.columns:
            continue
        totals = frame.groupby(team_column)[source].transform("sum")
        frame[name] = np.where(totals > 0, frame[source] / totals, np.nan)
    return frame
