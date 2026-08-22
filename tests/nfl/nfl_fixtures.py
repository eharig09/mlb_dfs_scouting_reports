"""Shared fixtures for the NFL package.

**Why this is not a conftest.py.** Six files in the MLB suite reach for their fixture
builder with a bare `from conftest import make_slate`. That is an ordinary Python import,
resolved off sys.path, so a second module named `conftest` anywhere pytest has put on the
path is ambiguous -- and `tests/nfl/` sorts before `tests/test_*.py`, so the NFL one won.
The MLB tests then silently ran against an NFL slate: 77 errors, and every one of them
reported as a failure inside `dfs`, nowhere near the cause. Shipping no NFL conftest at all
removes the collision without touching the MLB suite. Test modules import the fixtures they
need from here by name, which pytest resolves normally.

Everything here is **synthetic**, the same rule the MLB suite follows: the tests must run
on a clean checkout with no cached data and no DK export.

Two properties are load-bearing and both were learned on the baseball side.

**Events first, then points, then bands.** A fixture that draws `Proj` from a uniform and
invents events to match will disagree with itself, and any test of a marginal will then be
failing against the fixture rather than the code. Here every player's projection is the
output of `nfl.scoring` applied to his own event line, so the frame satisfies the same
invariant a real projection would.

**Just barely legal, with room on the cap.** Every team fields a full complement, so roster
feasibility is guaranteed and any infeasibility a test sees came from the constraint under
test. Salaries span DK's real range and the cheapest legal lineup lands near $29k of a $50k
cap -- if the cap bound on every solve, tests of stacks and locks would be failing on
affordability instead of on the thing they name.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# One full offense per team plus a defense. Two RB and four WR give the FLEX seat real
# choices rather than a single forced answer.
TEAM_POSITIONS = (["QB"] + ["RB"] * 3 + ["WR"] * 4 + ["TE"] * 2 + ["DST"])
TEAMS = ["BUF", "MIA", "KAN", "LAC", "PHI", "DAL", "SFO", "SEA",
         "BAL", "CIN", "DET", "GNB", "HOU", "IND", "MIN", "NOR"]

# DK's real salary bands by position, low to high.
SALARY_RANGE = {"QB": (4800, 8600), "RB": (4000, 9400), "WR": (3200, 9000),
                "TE": (2600, 7400), "DST": (2000, 4200)}

# Rough per-position event shapes at skill 1.0. Not a projection model -- just enough
# structure that scoring, salary and position move together the way a real board does.
_SHAPES = {
    "QB": {"PASS_YD": 245.0, "PASS_TD": 1.55, "INT": 0.75, "RUSH_YD": 18.0,
           "RUSH_TD": 0.18, "P_PASS_300": 0.26},
    "RB": {"RUSH_YD": 58.0, "RUSH_TD": 0.42, "REC": 2.6, "REC_YD": 21.0,
           "REC_TD": 0.09, "FUM_LOST": 0.04, "P_RUSH_100": 0.17},
    "WR": {"REC": 4.1, "REC_YD": 52.0, "REC_TD": 0.36, "RUSH_YD": 1.5,
           "FUM_LOST": 0.02, "P_REC_100": 0.15},
    "TE": {"REC": 3.3, "REC_YD": 36.0, "REC_TD": 0.29, "FUM_LOST": 0.02,
           "P_REC_100": 0.07},
}

# DST events are counting stats plus a points-allowed distribution. The distribution is the
# point: a defense's largest single term is a step function of the opponent's score, so a
# fixture that hands it a mean would understate every defense on the board.
_DST_SHAPE = {"SACK": 2.4, "INT": 0.8, "FUM_REC": 0.6, "SAFETY": 0.04,
              "BLK": 0.05, "TD": 0.11}
_DST_POINTS_ALLOWED = {0: 0.02, 3: 0.06, 7: 0.13, 10: 0.16, 14: 0.17,
                       17: 0.15, 21: 0.12, 24: 0.09, 28: 0.06, 35: 0.04}

# Bands. Deliberately crude and deliberately labelled: these are placeholders standing in
# for a fitted floor/ceiling model, not estimates of anything. The optimizer only needs
# them to be present, ordered and consistent with Proj.
CEILING_MULTIPLIER = {"QB": 1.45, "RB": 1.85, "WR": 1.95, "TE": 1.90, "DST": 2.10}
FLOOR_MULTIPLIER = {"QB": 0.62, "RB": 0.34, "WR": 0.28, "TE": 0.30, "DST": 0.20}


SKILL_LOW, SKILL_HIGH = 0.55, 1.45


def _price(skill, low, high, rng, jitter=200):
    """Map skill onto the position's DK salary range, jittered to break collinearity.

    The share has to span the FULL skill range. Clamping skill at 1.0 first put every
    player in the top half of his salary band, which made the cheapest legal lineup cost
    $57k of a $50k cap -- the slate was infeasible before a single constraint was applied.
    """
    share = (skill - SKILL_LOW) / max(SKILL_HIGH - SKILL_LOW, 1e-9)
    salary = low + share * (high - low) + rng.integers(-jitter, jitter + 1)
    return int(round(max(low, min(high, salary)) / 100) * 100)


def _skill_player(name, team, opponent, game, position, salary, skill):
    """One offensive player, built the way a projection would build one: events, then
    `nfl.scoring`, then the bands."""
    from nfl.scoring import offense_points

    shape = _SHAPES[position]
    events = {}
    for key, value in shape.items():
        if key.startswith("P_"):
            # A bonus probability scales with skill but must stay a probability.
            events[key] = min(0.95, value * skill ** 1.6)
        elif key in {"INT", "FUM_LOST"}:
            events[key] = value / max(skill, 0.35)      # better players turn it over less
        else:
            events[key] = value * skill
    proj = offense_points(events)
    return {
        "Name": name, "Team": team, "Opp": opponent, "Game": game,
        "Pos": position, "DK Pos": position, "Type": "OFF",
        "Salary": salary, "DK ID": abs(hash(name)) % 90000,
        "Proj": round(proj, 2),
        "Ceiling": round(proj * CEILING_MULTIPLIER[position], 2),
        "Floor": round(proj * FLOOR_MULTIPLIER[position], 2),
        **{f"E_{k}": round(v, 4) for k, v in events.items()},
    }


def _defense(team, opponent, game, salary, skill):
    """One DST, same rule: events and a points-allowed distribution, then scoring."""
    from nfl.scoring import dst_points

    events = {k: v * skill for k, v in _DST_SHAPE.items()}
    # A better defense shifts probability toward the low buckets. Renormalised so it stays
    # a distribution whatever the tilt.
    tilt = {total: weight * (skill ** (1.0 - total / 17.0))
            for total, weight in _DST_POINTS_ALLOWED.items()}
    scale = sum(tilt.values())
    allowed = {total: weight / scale for total, weight in tilt.items()}
    proj = dst_points(events, points_allowed_distribution=allowed)
    return {
        "Name": f"{team} DST", "Team": team, "Opp": opponent, "Game": game,
        "Pos": "DST", "DK Pos": "DST", "Type": "DST",
        "Salary": salary, "DK ID": abs(hash(team + "dst")) % 90000,
        "Proj": round(proj, 2),
        "Ceiling": round(proj * CEILING_MULTIPLIER["DST"], 2),
        "Floor": round(proj * FLOOR_MULTIPLIER["DST"], 2),
        **{f"E_{k}": round(v, 4) for k, v in events.items()},
    }


def make_slate(n_games=6, seed=1, with_ownership=True):
    """A legal synthetic DK NFL Classic slate. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_games):
        away, home = TEAMS[2 * g], TEAMS[2 * g + 1]
        game = f"{away}@{home}"
        for team, opponent in ((away, home), (home, away)):
            counts = {}
            for position in TEAM_POSITIONS:
                counts[position] = counts.get(position, 0) + 1
                skill = float(rng.uniform(SKILL_LOW, SKILL_HIGH))
                low, high = SALARY_RANGE[position]
                salary = _price(skill, low, high, rng)
                if position == "DST":
                    rows.append(_defense(team, opponent, game, salary, skill))
                else:
                    rows.append(_skill_player(
                        f"{team} {position}{counts[position]}", team, opponent, game,
                        position, salary, skill))

    frame = pd.DataFrame(rows)
    if with_ownership:
        # No ownership model exists in `nfl` yet. A crude points-per-dollar softmax stands
        # in so the max-ownership constraint has something to bind on; it is explicitly not
        # a field model and nothing should be fitted against it.
        value = frame["Proj"] / (frame["Salary"] / 1000.0)
        appeal = np.exp((value - value.mean()) / max(value.std(), 1e-9))
        frame["Own%"] = (appeal / appeal.sum() * 100 * 9).round(2)
        frame["Own Src"] = "fixture"
    return frame


@pytest.fixture
def slate():
    return make_slate()


@pytest.fixture
def small_slate():
    return make_slate(n_games=3, seed=5)
