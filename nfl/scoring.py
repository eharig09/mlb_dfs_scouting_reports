"""DraftKings NFL Classic scoring.

Kept in one place so the projection code stays in event units (yards, receptions,
touchdowns) and only converts to points at the very end -- the same rule `dfs.scoring`
follows, and for the same reason: a scoring change then lands here and nowhere else, and
the simulator and the post-game review can both call the function the projection used.

**The yardage bonuses are the trap.** DK pays a flat +3 for a 100-yard rushing game, a
100-yard receiving game, or a 300-yard passing game. Those are step functions of the
realised total, so they cannot be applied to an expectation: a back projected for exactly
100 yards does not earn 3 points, he earns 3 x P(>= 100), and that probability is nowhere
near 0.5 because the yardage distribution is right-skewed. `offense_points` therefore takes
the bonus terms as *probabilities* under separate keys and will not infer them from the
yardage line. Passing a 1.0 there means "this happened", which is what the review does when
scoring a completed game.

This is the direct analogue of the complete-game and no-hitter terms `dfs.scoring` leaves
out of `pitcher_points`: rare step-function bonuses that are wrong to average.
"""

# Per-unit values. Yardage is the fractional-point part; everything else is discrete.
DK_OFFENSE = {
    "PASS_YD": 0.04,        # 1 point per 25 yards
    "PASS_TD": 4.0,
    "INT": -1.0,
    "RUSH_YD": 0.1,
    "RUSH_TD": 6.0,
    "REC": 1.0,             # full PPR
    "REC_YD": 0.1,
    "REC_TD": 6.0,
    "FUM_LOST": -1.0,
    "TWO_PT": 2.0,          # passing, rushing or receiving conversion
    "RET_TD": 6.0,          # punt / kickoff / FG return
    "FUM_REC_TD": 6.0,      # offensive fumble recovery touchdown
}

# Step-function bonuses, keyed on the PROBABILITY of clearing the threshold.
DK_OFFENSE_BONUS = {
    "P_PASS_300": 3.0,
    "P_RUSH_100": 3.0,
    "P_REC_100": 3.0,
}

DK_DST = {
    "SACK": 1.0,
    "INT": 2.0,
    "FUM_REC": 2.0,
    "SAFETY": 2.0,
    "BLK": 2.0,             # blocked punt, FG or extra point
    "TD": 6.0,              # any defensive or special-teams touchdown
    "TWO_PT_RET": 2.0,
}

# Points allowed, as (inclusive upper bound, points). Ordered low to high; the first
# bucket a total fits into is the one that pays.
DST_POINTS_ALLOWED = (
    (0, 10.0),
    (6, 7.0),
    (13, 4.0),
    (20, 1.0),
    (27, 0.0),
    (34, -1.0),
    (float("inf"), -4.0),
)

# DK Classic roster: QB, 2 RB, 3 WR, TE, FLEX (RB/WR/TE), DST. $50,000 cap.
DK_CLASSIC_SLOTS = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]
DK_SALARY_CAP = 50000

# Positions the FLEX seat accepts. Not QB and not DST -- the single most common way a
# hand-built NFL optimizer produces illegal lineups.
FLEX_POSITIONS = frozenset({"RB", "WR", "TE"})


def offense_points(events):
    """DK points from a dict of projected offensive events.

    `events` is in event units. Bonus keys are probabilities in [0, 1]; omit them and the
    bonuses simply do not score, which is the right default for a projection that has not
    modelled them yet -- silently deriving them from mean yardage would be worse than
    leaving points on the table, because it would be wrong in a direction nobody checks.
    """
    total = sum(value * float(events.get(key, 0.0)) for key, value in DK_OFFENSE.items())
    total += sum(value * float(events.get(key, 0.0))
                 for key, value in DK_OFFENSE_BONUS.items())
    return total


def points_allowed_points(points_allowed):
    """The DST points-allowed bucket for a realised final score."""
    for ceiling, value in DST_POINTS_ALLOWED:
        if points_allowed <= ceiling:
            return value
    return DST_POINTS_ALLOWED[-1][1]


def expected_points_allowed_points(distribution):
    """Points-allowed value under a distribution over opponent scores.

    `distribution` maps a points-allowed total to its probability. A DST's single largest
    scoring term is a step function of the opponent's final score, so projecting it from
    the mean is the same error as the yardage bonuses above -- a mean of 21 pays 0, while a
    realistic spread around 21 pays meaningfully more, because the upside buckets are
    further from zero than the downside ones.
    """
    return sum(probability * points_allowed_points(total)
               for total, probability in (distribution or {}).items())


def dst_points(events, points_allowed=None, points_allowed_distribution=None):
    """DK points for a defense/special teams unit.

    Pass `points_allowed` to score a completed game, or `points_allowed_distribution` to
    project one. Passing both prefers the distribution, since only a projection has one.
    """
    total = sum(value * float(events.get(key, 0.0)) for key, value in DK_DST.items())
    if points_allowed_distribution:
        total += expected_points_allowed_points(points_allowed_distribution)
    elif points_allowed is not None:
        total += points_allowed_points(points_allowed)
    return total
