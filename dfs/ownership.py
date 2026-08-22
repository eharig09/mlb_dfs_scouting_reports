"""Estimate what fraction of the field will roster each player, and turn that into leverage.

The board already ranks players by how good they are. That is not enough to win a
tournament, because everyone else is ranking them the same way -- a lineup only pays when it
is both right and *different*. Measuring "different" needs a model of what the field does.

The field is not mysterious. It overwhelmingly rosters on **points per dollar**, nudged by
batting order and team total. So ownership is modelled as a softmax over a "field appeal"
score, normalised so that ownership across each roster group sums to the number of slots the
field has to fill -- eight hitters and two pitchers, each worth 100% of the field.

This is an estimate, not data. `Own Src` says which it is, so nothing downstream mistakes a
model for a measurement. Once real `%Drafted` from DK contest results is imported it replaces
the estimate outright, and the same columns keep working.

Deliberately NOT modelled on our own projection: if ownership came from the same numbers the
lineups come from, leverage would be structurally zero. The inputs are what the field can
see -- DK's salary, DK's season average, batting order, and a public consensus projection
from `dfs.ros`. That last one is third-party rather than ours, which is what keeps it on the
right side of the circularity line.
"""

import numpy as np
import pandas as pd

# Bumped whenever a change here would move an ownership estimate. Recorded in snapshots.
#
#   1  softmax over fitted field appeal (salary .50 / dk_avg .30 / value .15 / order .05)
#   2  adds a public consensus DK-points rate from dfs.ros as a fifth appeal term, which
#      largely displaces DK Avg (salary .35 / ros_fpts .30 / value .20 / dk_avg .10 / order .05)
OWNERSHIP_VERSION = 2

# DK Classic: two pitchers and eight hitters, so the field spends 200 and 800 points of
# ownership respectively. Anything that sums elsewhere is not an ownership distribution.
GROUP_SLOTS = {"P": 2, "H": 8}

# Softmax temperature on a 0-100 appeal scale. Lower concentrates ownership on the chalk;
# higher flattens it. Fitted, not guessed: 40 against five slates of measured %Drafted from
# dk_results/. The first guess of 14 was far too sharp -- real ownership has a longer, flatter
# tail than a confident softmax produces, and over-concentrating it inflates leverage for
# every mid-owned player.
#
# **Refit 2026-08-08 on 14 slates / 1,724 player-slates -- three times the original data --
# and 40 came back optimal.** Worth recording what that refit ruled out, because the residual
# error looks like a calibration bug and is not one:
#
#   T      10     18     26     34     40     44     50
#   MAE  7.42   5.70   4.96   4.72   4.67   4.67   4.70
#
# The model under-calls real chalk badly -- players the field rosters at 30%+ come back at
# ~24%, a +18 point gap -- and lowering the temperature is the obvious fix. It does not work.
# Sharpening makes MAE worse at every step while barely moving the gap, because sharpening
# concentrates ownership on whoever `field_appeal` ranks first, and that ranking is only
# right about half the time (Spearman 0.56; 4.1 of the field's ten most-owned appear in our
# top ten). At T=8 the model's most-owned player averages 64% against the field's real 43% --
# it is perfectly capable of calling someone chalk, just not the right someone.
#
# Also ruled out, same fit set: the four APPEAL_WEIGHTS (1,771-point simplex sweep x three
# temperatures; the shipped weights are within 0.02 MAE of the global best), and replacing
# the percentile transform in `field_appeal` with z-scores or min-max, on the theory that
# percentiles discard the magnitude of a price gap. Every variant lands on the same
# MAE-vs-chalk-bias frontier.
#
# So the gap is not a tuning error, it is the honest output of a model that does not know
# which player the field will pile onto. Spreading the mass is the right response to that
# uncertainty. Moving it needs a better *signal* -- Vegas implied runs, a public consensus
# projection, last night's measured ownership -- not a better constant.
TEMPERATURE = 40.0

# No single player is on more than this share of lineups in practice, even the night's
# obvious chalk. Without a cap the softmax can hand one player an implausible share and
# starve everyone else in the group.
#
# At T=40 it never binds -- sweeping the cap from 40 to 100 moves MAE by 0.05 and the top
# estimate never reaches 60. It is a guard against a future sharper fit, not a live constraint.
MAX_OWNERSHIP = 60.0

# Fitted against five slates of measured %Drafted (503 player-slates) in dk_results/, and
# re-checked 2026-08-08 against 14 slates (1,724 player-slates) -- see TEMPERATURE above for
# the sweep. Held up: MAE 4.67 and Spearman 0.61 out of sample against 5.23 / 0.63 in.
#
# The result overturned the assumption these started from. "The field chases points per
# dollar" is the standard story and it is wrong here: **price is the dominant term**, at
# roughly three times the weight of value. The field rosters expensive players because they
# are expensive -- name recognition and perceived safety -- and season form second. Fitting
# it moved MAE from 7.41 to 5.23 and rank correlation from 0.51 to 0.63.
#
# The error surface is flat between salary .45 and .60, so the exact split of the remainder
# between value and dk_avg is not well determined by five nights; salary being large is the
# robust part. A small order weight is kept deliberately rather than fitted to zero, so the
# model still says something sensible on a slate where DK Avg is missing.
# **Refit 2026-08-17 on 25 slates / 2,976 player-slates, adding a fifth term.** The 08-08
# refit concluded the residual error was a missing *signal*, not a bad constant, and named
# "a public consensus projection" as a candidate. The rest-of-season exports supply one:
# `dfs.ros` carries a consensus DK-points rate (FPTS/G for hitters, FPTS/IP for pitchers)
# averaged over Steamer, ZiPS and The Bat. It is a public number, so it does not make
# Leverage circular the way our own projection would.
#
# It works, and it largely *displaces* DK Avg -- the consensus rate is a better read on what
# the field is looking at than DK's own season average. Validated rather than fitted-and-shipped:
#
#   last 10 nights, never fitted:  MAE 4.686 -> 4.507,  spearman 0.518 -> 0.541
#   per night:                     better on 19 of 25, mean gain +0.101 MAE
#                                  CI [+0.024, +0.180], Wilcoxon p=0.022
#
# DK Avg is kept at a small weight rather than dropped: it is the only one of these that is
# always present on the export, so it carries the model on a slate the ROS files do not cover.
# Average draft position was also tested as a fame proxy and added nothing measurable (+0.102
# vs +0.105 without it), so it is not included.
#
# The chalk problem is NOT fixed by this -- 30%+ players still come back ~17 points under
# (see the band table in docs/benchmarks/ownership_refit.json). This buys general accuracy,
# not the ability to name the night's chalk.
APPEAL_WEIGHTS = {"value": 0.20, "dk_avg": 0.10, "salary": 0.35, "order": 0.05,
                  "ros_fpts": 0.30}


def _pct(series):
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() <= 1:
        return pd.Series(50.0, index=series.index)
    return values.rank(pct=True) * 100.0


def _order_appeal(slot):
    """Batting order as the field reads it: top of the order is a selling point."""
    value = pd.to_numeric(slot, errors="coerce")
    # 1st -> 100, 5th -> ~55, 9th -> ~10. Flat outside 1-9 so pitchers are unaffected.
    scaled = 100.0 - (value - 1.0) * 11.0
    return scaled.where(value.between(1, 9), 50.0).fillna(50.0)


def _consensus_rate(group):
    """Public consensus DK-points rate per player, as a percentile, or None.

    Read from the rest-of-season exports via `dfs.ros`. This is the one term that is not on
    the DK file, and it is included precisely because it is *public* -- it stands in for the
    number the field is reading, which is what ownership is a function of.
    """
    mlbam = pd.to_numeric(group.get("MLBAM"), errors="coerce") if "MLBAM" in group else None
    if mlbam is None or mlbam.notna().sum() < 2:
        return None
    try:
        from .ros import load as load_ros
        rates = load_ros()[0]
    except Exception:
        return None
    lookup = {}
    for side in ("H", "P"):
        for key, values in (rates.get(side) or {}).items():
            value = values.get("fpts")
            if value is not None:
                lookup[key] = value
    if not lookup:
        return None
    values = pd.Series([lookup.get(int(m)) if pd.notna(m) else None for m in mlbam],
                       index=group.index, dtype=float)
    if values.notna().sum() < 2:
        return None
    # Neutral 50 for anyone the exports do not cover: `_pct` leaves NaN for missing values,
    # and a single unrated player would otherwise void the whole weighted sum and take his
    # slate's softmax with him.
    return _pct(values).fillna(50.0)


def field_appeal(group):
    """0-100 score for how attractive a player looks to the field, before normalisation.

    Built from what the field can see -- DK's price, season average and batting order, plus a
    public consensus projection -- and never from our own projection. If ownership were a
    function of our number then Leverage (ceiling minus ownership) would be structurally near
    zero and the whole exercise would be circular. A third-party consensus is not that: the
    field really does read Steamer and friends, and our projection disagrees with them often
    enough that the two are not interchangeable.
    """
    salary = pd.to_numeric(group.get("Salary"), errors="coerce")
    dk_avg = pd.to_numeric(group.get("DK Avg"), errors="coerce") if "DK Avg" in group else None
    if dk_avg is None or dk_avg.notna().sum() < 2:
        # No season average on the file: fall back to price alone, which is still the field's
        # loudest signal, and flag nothing else as known.
        return _pct(salary)

    # The field's own value metric, not ours: DK's season average per $1k.
    field_value = dk_avg / (salary / 1000.0).replace(0, np.nan)
    order_pct = _order_appeal(group.get("Slot", pd.Series(index=group.index, dtype=float)))
    consensus = _consensus_rate(group)

    weights = dict(APPEAL_WEIGHTS)
    parts = {"value": _pct(field_value), "dk_avg": _pct(dk_avg),
             "salary": _pct(salary), "order": order_pct}
    if consensus is not None:
        parts["ros_fpts"] = consensus
    else:
        # No exports on disk: drop the term and renormalise rather than feeding the softmax a
        # constant 50, which would flatten every appeal score toward the middle.
        weights.pop("ros_fpts", None)
    # Divided by the weight total so the result stays on the 0-100 scale the softmax expects.
    # A no-op while the shipped weights sum to 1.0; it matters when a term was dropped above.
    total = sum(weights.values()) or 1.0
    return sum(weights[k] * parts[k] for k in weights) / total


def _softmax_ownership(appeal, total):
    """Distribute `total` points of ownership over players by appeal, capped per player.

    Capping redistributes rather than truncates: the mass a capped player cannot hold has to
    land on someone, or the group stops summing to the number of slots the field must fill.
    """
    scores = np.exp((appeal - appeal.max()) / TEMPERATURE)
    own = total * scores / scores.sum()

    # A couple of passes is enough to settle the cap in practice; the loop guards the rest.
    for _ in range(10):
        over = own > MAX_OWNERSHIP
        if not over.any():
            break
        spill = float((own[over] - MAX_OWNERSHIP).sum())
        own[over] = MAX_OWNERSHIP
        free = ~over
        if not free.any() or spill <= 0:
            break
        headroom = scores[free] / scores[free].sum()
        own[free] = own[free] + spill * headroom
    return own


def estimate_ownership(players):
    """Add Own%, Own Pct and Leverage to a slate frame. Returns a new frame.

    Leverage is the whole point: ceiling percentile minus ownership percentile. A player the
    field is ignoring and we are not is a positive number, and that is the only kind of edge
    a tournament actually pays for.
    """
    frame = players.copy()
    if frame.empty or "Salary" not in frame.columns:
        frame["Own%"] = None
        frame["Own Pct"] = None
        frame["Leverage"] = None
        frame["Own Src"] = ""
        return frame

    priced = pd.to_numeric(frame["Salary"], errors="coerce").notna()
    frame["Own%"] = np.nan
    for group_type, slots in GROUP_SLOTS.items():
        mask = priced & (frame["Type"] == group_type)
        if mask.sum() < 2:
            continue
        appeal = field_appeal(frame.loc[mask])
        frame.loc[mask, "Own%"] = _softmax_ownership(appeal.to_numpy(dtype=float),
                                                     slots * 100.0).round(1)

    # Ranked within pitchers and hitters separately, matching how every other percentile on
    # the board is computed -- the two groups' ownership scales are not comparable.
    frame["Own Pct"] = np.nan
    for group_type in GROUP_SLOTS:
        mask = frame["Type"] == group_type
        if mask.sum() > 1:
            frame.loc[mask, "Own Pct"] = _pct(frame.loc[mask, "Own%"]).round(1)

    frame = _leverage_columns(frame)
    frame["Own Src"] = np.where(frame["Own%"].notna(), "est", "")
    return frame


# Points of ceiling surrendered per point of ownership, for the `leverage` objective. At
# 0.12 a 50%-owned player must out-ceiling a 5%-owned one by ~5.4 points to be worth the
# duplication -- roughly the gap between a premium bat and a mid-priced one, which is the
# trade the objective is meant to express. Leverage stays a *percentile* difference for
# reading the board; this is the same idea on a points scale so the solver and the stack
# bonus, which scales with the objective's own magnitude, stay commensurate.
#
# Kept as the fallback for a group too small to measure a spread on; the live weight is
# derived per group by `_own_weight` below.
LEVERAGE_OWN_WEIGHT = 0.12

# What that trade costs, stated in standard deviations of the group's own ceiling spread.
#
# A flat points constant silently means different things to different groups, and drifts
# whenever the projection model is rescaled. Both happened. The single 0.12 was never one
# trade: against the model it was written for it cost a pitcher 0.93 ceiling sd and a hitter
# 2.54 -- the asymmetry was real but invisible. Then the MODEL_VERSION 4 sample-size
# regression compressed pitcher ceiling spread 26% (sd 5.82 -> 4.33), and the same 0.12 went
# from spanning 28% of the pitcher ceiling distribution to 38%, quietly fading chalk arms a
# third harder than intended while hitters (49% -> 49%) were untouched.
#
# Denominating in sd fixes both: the numbers below say outright what each group's trade is,
# and they hold whatever the projection scale does next. These preserve the behaviour the
# shipped 0.12 produced, so this is a change of parameterisation and not of strategy --
# retune the two numbers to change the strategy.
LEVERAGE_OWN_SD = {"P": 0.93, "H": 2.54}     # ceiling sd per 45 points of ownership
LEVERAGE_OWN_SPAN = 45.0                     # the 5%-owned to 50%-owned swing


def _own_weight(ceiling, player_type):
    """Ceiling points charged per point of ownership, for one player group."""
    target = LEVERAGE_OWN_SD.get(str(player_type))
    spread = pd.to_numeric(ceiling, errors="coerce").std()
    if target is None or not np.isfinite(spread) or spread <= 0:
        return LEVERAGE_OWN_WEIGHT
    return float(target * spread / LEVERAGE_OWN_SPAN)


# Below this, a player was not rostered by anybody -- which on a finished slate means he did
# not play, not that the field missed him. Measured against dk_results: Ohtani came back at
# 0.13% on 2026-07-30 with 0.0 points, having been rested. Left alone, those players carry a
# full projected ceiling against ~zero ownership and so score as the *most* leveraged plays
# on the board, and the leverage objective would chase scratches. The estimate never produces
# a number this low, so this only ever bites on real ownership -- which is exactly where it
# would do the damage, since that is what any calibration or review runs on.
MIN_MEANINGFUL_OWN = 0.5


def _leverage_columns(frame):
    ceiling_pct = pd.to_numeric(frame.get("Ceiling Pct"), errors="coerce")
    own_pct = pd.to_numeric(frame.get("Own Pct"), errors="coerce")
    own = pd.to_numeric(frame.get("Own%"), errors="coerce")

    frame["Leverage"] = (ceiling_pct - own_pct).round(1)
    # Reindexed rather than read straight off the frame: `frame.get` returns None for a board
    # without a Ceiling column, and pd.to_numeric(None) is a scalar NaN, which has no .loc for
    # the per-group weighting below to index. A board can legitimately arrive without it --
    # anything that re-estimates ownership from a reduced frame does.
    ceiling = pd.to_numeric(frame.reindex(columns=["Ceiling"])["Ceiling"], errors="coerce")
    # The ownership charge is set per group against that group's own ceiling spread, so a
    # pitcher and a hitter are each traded on their own scale. Falls back to the flat weight
    # when there is no Type column to group by.
    weight = pd.Series(LEVERAGE_OWN_WEIGHT, index=frame.index, dtype=float)
    if "Type" in frame.columns:
        for player_type, group in frame.groupby("Type", observed=True):
            weight.loc[group.index] = _own_weight(ceiling.loc[group.index], player_type)
    # Floored at zero: a negative objective value would make the solver prefer to leave a
    # roster slot underfilled rather than take the best remaining player.
    frame["Lev Score"] = (ceiling - weight * own.fillna(0)).clip(lower=0.0).round(2)

    unrostered = own.notna() & (own < MIN_MEANINGFUL_OWN)
    frame.loc[unrostered, "Leverage"] = np.nan
    frame.loc[unrostered, "Lev Score"] = 0.0
    frame["Unrostered"] = unrostered
    return frame


def attach_actual_ownership(players, drafted, source="DK"):
    """Replace the estimate with real %Drafted, keyed on DK ID then name.

    Real ownership is strictly better than the model and this is how it gets in: parse a DK
    contest export into {dk id or normalized name: pct} and pass it here.
    """
    from .salaries import normalize_name

    frame = players.copy()
    if not drafted:
        return frame
    by_key = {str(k).strip(): float(v) for k, v in drafted.items()}

    def lookup(row):
        dk_id = str(row.get("DK ID") or "").strip()
        if dk_id.endswith(".0"):
            dk_id = dk_id[:-2]
        if dk_id and dk_id in by_key:
            return by_key[dk_id]
        return by_key.get(normalize_name(row.get("Name")))

    found = frame.apply(lookup, axis=1)
    frame["Own%"] = found.where(found.notna(), frame.get("Own%"))
    frame["Own Src"] = np.where(found.notna(), source, frame.get("Own Src", ""))

    frame["Own Pct"] = np.nan
    for group_type in GROUP_SLOTS:
        mask = frame["Type"] == group_type
        if mask.sum() > 1:
            frame.loc[mask, "Own Pct"] = _pct(frame.loc[mask, "Own%"]).round(1)
    return _leverage_columns(frame)
