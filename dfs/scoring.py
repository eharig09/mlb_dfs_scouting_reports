"""DraftKings MLB Classic scoring.

Kept in one place so the projection code stays in event units (singles, walks, outs)
and only converts to points at the very end. A FanDuel scoring table can drop in here
later without touching the projection math.
"""

DK_HITTER = {
    "1B": 3.0,
    "2B": 5.0,
    "3B": 8.0,
    "HR": 10.0,
    "RBI": 2.0,
    "R": 2.0,
    "BB": 2.0,
    "HBP": 2.0,
    "SB": 5.0,
}

DK_PITCHER = {
    "IP": 2.25,      # 0.75 per out
    "K": 2.0,
    "W": 4.0,
    "ER": -2.0,
    "H": -0.6,
    "BB": -0.6,
    "HBP": -0.6,
    "CG": 2.5,
    "CGSO": 2.5,
    "NH": 5.0,
}

# DK Classic roster: 2 SP, C, 1B, 2B, 3B, SS, 3 OF, $50,000 cap.
DK_CLASSIC_SLOTS = ["P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"]
DK_SALARY_CAP = 50000


def hitter_points(events):
    """DK points from a dict of projected hitter events."""
    return sum(DK_HITTER[key] * float(events.get(key, 0.0)) for key in DK_HITTER)


def pitcher_points(events):
    """DK points from a dict of projected pitcher events.

    Complete games and no-hitters are left out: at fractional probabilities they add
    well under a tenth of a point and only blur the comparison between arms.
    """
    return sum(
        DK_PITCHER[key] * float(events.get(key, 0.0))
        for key in ("IP", "K", "W", "ER", "H", "BB", "HBP")
    )
