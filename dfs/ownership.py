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
lineups come from, leverage would be structurally zero. The field's inputs are DK's salary
and DK's season average, and those are what go in here.
"""

import numpy as np
import pandas as pd

# DK Classic: two pitchers and eight hitters, so the field spends 200 and 800 points of
# ownership respectively. Anything that sums elsewhere is not an ownership distribution.
GROUP_SLOTS = {"P": 2, "H": 8}

# Softmax temperature on a 0-100 appeal scale. Lower concentrates ownership on the chalk;
# higher flattens it. Fitted, not guessed: 40 against five slates of measured %Drafted from
# dk_results/. The first guess of 14 was far too sharp -- real ownership has a longer, flatter
# tail than a confident softmax produces, and over-concentrating it inflates leverage for
# every mid-owned player.
TEMPERATURE = 40.0

# No single player is on more than this share of lineups in practice, even the night's
# obvious chalk. Without a cap the softmax can hand one player an implausible share and
# starve everyone else in the group.
MAX_OWNERSHIP = 60.0

# Fitted against five slates of measured %Drafted (503 player-slates) in dk_results/.
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
APPEAL_WEIGHTS = {"value": 0.15, "dk_avg": 0.30, "salary": 0.50, "order": 0.05}


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


def field_appeal(group):
    """0-100 score for how attractive a player looks to the field, before normalisation.

    Built only from what is on the DK salary file -- season average, price, batting order --
    never from our own projection. Two reasons: the field does not have our numbers, and if
    ownership were a function of our projection then Leverage (ceiling minus ownership) would
    be structurally near zero and the whole exercise would be circular.
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

    return (APPEAL_WEIGHTS["value"] * _pct(field_value)
            + APPEAL_WEIGHTS["dk_avg"] * _pct(dk_avg)
            + APPEAL_WEIGHTS["salary"] * _pct(salary)
            + APPEAL_WEIGHTS["order"] * order_pct)


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
LEVERAGE_OWN_WEIGHT = 0.12


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
    ceiling = pd.to_numeric(frame.get("Ceiling"), errors="coerce")
    # Floored at zero: a negative objective value would make the solver prefer to leave a
    # roster slot underfilled rather than take the best remaining player.
    frame["Lev Score"] = (ceiling - LEVERAGE_OWN_WEIGHT * own.fillna(0)).clip(lower=0.0).round(2)

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
