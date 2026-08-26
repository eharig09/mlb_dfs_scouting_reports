"""Projected ownership: what fraction of the field will roster each player.

**Nothing in this module is calibrated yet, and that is the point of it existing now.**
Portfolio selection needs ownership — you cannot choose which of 150 lineups to enter
without knowing which of them the field is also on — and ownership needs contest results to
fit against. This is the scaffold, built so that the fitting has somewhere to land once a
few weeks of NFL exports are on disk. Every constant below is a placeholder and says so.

Read `calibrate()` before trusting any number this produces.

## Why the structure is what it is

The shape is lifted from `dfs.ownership`: score each player on how attractive he looks to
the field, then distribute a fixed pool of ownership points over those scores with a
softmax. The softmax is what makes it a *field* model rather than a ranking — ownership is
zero-sum within a roster slot, so one player getting more necessarily means another gets
less, and a model that scores players independently cannot express that.

**The normalisation is genuinely different from baseball, and this is the part not to copy.**
DK NFL Classic rosters 1 QB, 2 RB, 3 WR, 1 TE, 1 FLEX and 1 DST. So across the whole field
every quarterback's ownership sums to about 100 points, every running back's to about 200
plus whatever share of FLEX goes to backs, and so on. Normalising to one pool the way a
ten-hitter baseball roster does would put quarterbacks and receivers on the same scale and
overstate every quarterback by a factor of three.

    from nfl.ownership import estimate_ownership
    board = estimate_ownership(board)      # adds Own%, Own Pct, Leverage
"""

import numpy as np
import pandas as pd

from nfl.optimizer import FLEX_POSITIONS, ROSTER

OWNERSHIP_VERSION = 0          # 0 = never calibrated. Bumped by the first real fit.

# How the FLEX slot's 100 points of ownership divide between the positions eligible for it.
# **A guess, not a measurement.** Roughly the split a typical field runs, but it should be
# the first thing `calibrate` replaces, because it moves every RB/WR/TE number.
FLEX_SPLIT = {"RB": 0.35, "WR": 0.50, "TE": 0.15}

# Softmax temperature, **in projected DK points**. A player projecting `TEMPERATURE` points
# more than another is about e times as popular. Higher is a flatter field.
#
# The unit matters and was arrived at the hard way. The first version ran the softmax over
# *percentile* appeal, copying the MLB shape -- but percentiles are uniform by construction,
# so with 88 priced quarterbacks sharing a 100-point pool the top one came out at 2.5%
# however the temperature was set, on a slate where real chalk is thirty to fifty. No
# temperature fixes that, because the input carries no scale. Projected points do: the gap
# between a starter and a backup is large in the exponent, which is what concentrates a
# field.
#
# **The value is still unfitted** -- chosen so the shape is plausible, not measured against
# a contest. `calibrate()` replaces it.
TEMPERATURE = 3.0

# No single player is rostered by more than this share of the field, whatever the softmax
# says. Chalk saturates: a player everyone likes still competes with the lineup-building
# constraint that you can only fit so many chalk pieces under the cap.
MAX_OWNERSHIP = 65.0

# Ownership floor used when computing leverage. Nobody in a live contest is rostered by
# 0.01% of the field -- there is always someone -- and without a floor the leverage board is
# nothing but the players the model is least sure about.
MIN_OWNERSHIP_FOR_LEVERAGE = 0.5

# What makes a player look attractive to the field, and how much each part counts.
# **Unfitted.** The MLB twin fitted these against real contest exports; these are the same
# features re-weighted by judgement for football, which is a starting point and not a model.
APPEAL_WEIGHTS = {
    "value": 0.35,        # points per $1,000 -- the number every tool on earth shows
    "projection": 0.30,   # raw projected points
    "salary": 0.20,       # price itself: the field pays up for names
    "dk_avg": 0.15,       # DK's own season average, shown on the DK player card
}


def _percentile(series):
    """0-100 rank within the column, with ties shared and blanks left out."""
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index)
    return values.rank(pct=True) * 100.0


def field_appeal(players):
    """0-100 score for how attractive each player looks to the field, before normalising.

    Deliberately built only from things **the field can see**: price, projection, value and
    DK's own average. Nothing from the PFF layer goes in here — deployment and scheme fit
    are why *you* might like a player, and a field model that knew them would be modelling
    your own opinion rather than the crowd's.
    """
    frame = players.copy()
    parts, weights = [], []
    salary = pd.to_numeric(frame.reindex(columns=["Salary"])["Salary"], errors="coerce")
    projection = pd.to_numeric(frame.reindex(columns=["Proj"])["Proj"], errors="coerce")

    available = {
        "value": projection / (salary / 1000.0).replace(0, np.nan),
        "projection": projection,
        "salary": salary,
        "dk_avg": pd.to_numeric(frame.reindex(columns=["AvgPts"])["AvgPts"],
                                errors="coerce"),
    }
    for name, series in available.items():
        scored = _percentile(series)
        if scored.notna().any():
            # **A missing value scores at the bottom, not at the median.** Filling with the
            # median told the model that a player nobody projects is averagely attractive,
            # which is the opposite of true -- half a DK slate is third and fourth stringers
            # the field will not roster at all. With the median fill, 683 players shared the
            # ownership pool almost evenly and no chalk existed: the top quarterback came
            # out at 2.5% on a slate where real chalk is thirty to fifty.
            parts.append(scored.fillna(0.0) * APPEAL_WEIGHTS[name])
            weights.append(APPEAL_WEIGHTS[name])
    if not parts:
        return pd.Series(np.nan, index=frame.index)

    appeal = sum(parts) / sum(weights)
    # A player with no projection at all is not competing for ownership. He keeps a score so
    # the frame stays rectangular, but it is the floor rather than a guess.
    unprojected = projection.isna()
    appeal[unprojected] = appeal[~unprojected].min() if (~unprojected).any() else 0.0
    return appeal


def _position_pool(position):
    """Ownership points the whole field spends on one position, per 100 lineups.

    A roster slot is 100 points; FLEX is split across the positions eligible for it. This is
    the arithmetic that makes football ownership add up, and the reason a one-pool
    normalisation would treble every quarterback.
    """
    base = float(ROSTER.get(position, 0)) * 100.0
    if position in FLEX_POSITIONS:
        base += ROSTER.get("FLEX", 0) * 100.0 * FLEX_SPLIT.get(position, 0.0)
    return base


def _softmax(scores, total, temperature=TEMPERATURE, cap=MAX_OWNERSHIP):
    """Spread `total` points of ownership over players by score, capped per player.

    `scores` are in projected DK points -- see `TEMPERATURE` for why the unit matters.
    Capping redistributes rather than truncating: points taken off a saturated favourite
    have to land on somebody, because the field still fields a full roster.
    """
    scores = pd.to_numeric(scores, errors="coerce")
    if scores.notna().sum() == 0 or total <= 0:
        return pd.Series(np.nan, index=scores.index)
    # A player with no projection is not competing: he sits at the floor of the exponent
    # rather than at its middle, which is what a blank actually means here.
    centred = scores.fillna(scores.min()) - scores.max()
    weights = np.exp(centred / max(temperature, 1e-6))
    share = weights / weights.sum()
    owned = share * total

    # Redistribute what the cap sheds, until nobody is over it or nobody is left under it.
    for _ in range(12):
        over = owned > cap
        if not over.any():
            break
        excess = float((owned[over] - cap).sum())
        owned[over] = cap
        room = ~over
        if not room.any():
            break
        headroom = (cap - owned[room]).clip(lower=0)
        if headroom.sum() <= 0:
            break
        owned[room] = owned[room] + excess * headroom / headroom.sum()
    return owned


def estimate_ownership(players, temperature=TEMPERATURE):
    """Add `Own%`, `Own Pct` and `Appeal` to a board. Returns a new frame.

    Ownership is estimated **within each position**, for the reason in the module docstring.
    A player with no projection gets no estimate rather than a zero: the field is not
    rostering him at 0%, we simply have no basis to say.
    """
    frame = players.copy()
    frame["Appeal"] = field_appeal(frame)
    frame["Own%"] = np.nan

    # The softmax runs on **projected points scaled by appeal**, not on appeal alone. Points
    # supply the scale that makes a field concentrate; appeal supplies the reason the field
    # prefers one similarly-projected player to another. Appeal is centred on 1.0 so it
    # tilts the ordering without inventing points.
    projection = pd.to_numeric(frame.reindex(columns=["Proj"])["Proj"], errors="coerce")
    average = pd.to_numeric(frame.reindex(columns=["AvgPts"])["AvgPts"], errors="coerce")
    appeal = pd.to_numeric(frame["Appeal"], errors="coerce")
    tilt = 1.0 + (appeal - appeal.median()) / 200.0 if appeal.notna().any() else 1.0

    positions = frame.reindex(columns=["Pos"])["Pos"].astype("string").fillna("")
    for position in positions.dropna().unique():
        if not position:
            continue
        mask = positions == position
        pool = _position_pool(position)
        if pool <= 0:
            continue
        # **A position nobody projects still gets rostered.** PFF's export carries no
        # defenses at all, so keying only on `Proj` silently dropped every DST from the
        # field -- a position the field spends a full 100 ownership points on. DK's own
        # season average stands in where the projection is absent for the whole group.
        points = projection[mask]
        if points.notna().sum() == 0:
            points = average[mask]
        frame.loc[mask, "Own%"] = _softmax(points * tilt[mask], pool,
                                           temperature=temperature)

    frame["Own Pct"] = _percentile(frame["Own%"])
    frame["Own Src"] = f"nfl.ownership v{OWNERSHIP_VERSION} (UNCALIBRATED)"
    return frame


def leverage(players, ceiling_column="Ceiling", own_column="Own%"):
    """Ceiling per point of projected ownership -- the shape a GPP actually rewards.

    Reported as a **ratio, not a fitted edge**. The MLB twin prices leverage against a
    measured ceiling-versus-ownership relationship; nothing here has been measured, so this
    is a sorting aid and must not be used as an objective until `calibrate` has run.
    """
    frame = players.copy()
    ceiling = pd.to_numeric(frame.reindex(columns=[ceiling_column])[ceiling_column],
                            errors="coerce")
    owned = pd.to_numeric(frame.reindex(columns=[own_column])[own_column], errors="coerce")

    # **Floored, and only for players with a real projection.** A raw ceiling-over-ownership
    # ratio is unbounded as ownership goes to zero, so the leverage board filled with backup
    # quarterbacks at 0.01% owned and two projected points -- scores of 280 on a scale where
    # a real play sits near 0.5. Nobody is 0.01% owned in a live contest either; the field
    # has a floor and so does this.
    floor = max(MIN_OWNERSHIP_FOR_LEVERAGE, 1e-6)
    frame["Lev Score"] = ceiling / owned.clip(lower=floor)
    frame.loc[owned.isna() | ceiling.isna(), "Lev Score"] = np.nan
    return frame


def calibrate(contests, boards):
    """Fit `TEMPERATURE`, `APPEAL_WEIGHTS` and `FLEX_SPLIT` against real contest exports.

    **Not implemented, deliberately.** It needs what does not exist yet: several weeks of
    DK NFL contest-standings exports read through `nfl.results`, paired with the board each
    was played on. Writing the fitting code before there is anything to fit would produce a
    function that has never been run against real data and looks as though it has.

    What it will do, once there is data:

    1. For each contest, read actual ownership per player from the standings export.
    2. Score every player's appeal on the board that was live at lock.
    3. Fit `TEMPERATURE` by minimising error between the softmax and observed ownership.
    4. Refit `APPEAL_WEIGHTS` on the residual, and `FLEX_SPLIT` from the realised split of
       FLEX slots by position -- that one is measurable directly and should go first.
    5. Bump `OWNERSHIP_VERSION` and print a paste-ready constants block.

    Hold out at least one week to score on. A field model that cannot beat "everyone is
    owned at the position average" is an expensive way to compute an average.
    """
    raise NotImplementedError(
        "nfl.ownership.calibrate needs DK contest-standings exports in "
        "nfl/nfl_dfs/dk_results/. See nfl.results.list_contests() for what is on disk, "
        "and the module docstring for the fitting order.")
