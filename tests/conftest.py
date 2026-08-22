"""Shared fixtures.

Everything here is **synthetic**. The tests must run on a clean checkout with an empty
`.cache/`, so nothing in the core suite may depend on a cached payload or a DK export.
Tests that genuinely need real data are marked `integration` and skip themselves when it
is absent.

The synthetic slate is built to be *just* legal: every team fields exactly one of each
infield position and three outfielders, so roster feasibility is guaranteed and any
infeasibility a test sees came from the constraint under test rather than from the fixture.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# One of each infield slot, three outfielders, and a ninth bat that also plays outfield.
HITTER_POSITIONS = ["C", "1B", "2B", "3B", "SS", "OF", "OF", "OF", "OF"]
TEAMS = ["NYY", "BOS", "LAD", "SF", "CHC", "STL", "HOU", "TEX", "ATL", "NYM",
         "SD", "COL", "CLE", "DET", "TB", "TOR", "SEA", "MIN", "PHI", "MIA"]


def _hitter(name, team, opponent, game, slot, position, salary, skill, rng):
    """One hitter, built the way `dfs.projections` builds one.

    Events first, then points, then the bands -- not points drawn at random with events
    bolted on. An earlier version of this fixture drew `Proj` from a uniform and derived
    the events from batting slot alone, so the two disagreed by several points and the
    simulator's marginal tests failed against the *fixture*, not the code. A test double
    that cannot satisfy the invariants of the thing it doubles is worse than no double.
    """
    from dfs.projections import BUST_FIT, CEILING_Z, FLOOR_FIT, HITTER_BASE_VAR, HR_POINT_VALUE
    from dfs.scoring import DK_HITTER, hitter_points

    pa = (4.46 - 0.17 * (slot - 1))
    events = {
        "1B": pa * 0.145 * skill,
        "2B": pa * 0.045 * skill,
        "3B": pa * 0.004 * skill,
        "HR": pa * 0.030 * skill ** 1.7,
        "BB": pa * 0.085 * skill ** 0.35,
        "HBP": pa * 0.012,
        "R": (0.58 - 0.02 * slot) * skill,
        "RBI": (0.55 - 0.02 * slot) * skill,
        "SB": 0.05 * skill,
    }
    proj = hitter_points(events)
    variance = (HITTER_BASE_VAR + events["HR"] * HR_POINT_VALUE ** 2
                + events["SB"] * DK_HITTER["SB"] ** 2)
    ceiling = proj + CEILING_Z * np.sqrt(variance)
    floor = max(0.0, FLOOR_FIT["H"]["slope"] * proj + FLOOR_FIT["H"]["intercept"])
    bust = min(0.95, max(0.32, BUST_FIT["H"]["slope"] * proj + BUST_FIT["H"]["intercept"]))
    return {
        "Name": name, "Team": team, "Opp": opponent, "Game": game,
        "Slot": slot, "Pos": position, "DK Pos": position,
        "Bats": "LR"[slot % 2], "MLBAM": abs(hash(name)) % 900000 + 100000,
        "Type": "H", "Opp SP": f"{opponent} SP", "Opp SP Hand": "R",
        "Lineup": "Confirmed", "PA": round(pa, 2),
        "Salary": salary, "DK Avg": round(proj * 0.95, 2), "DK ID": abs(hash(name)) % 90000,
        "Proj": round(proj, 2), "Ceiling": round(ceiling, 2),
        "Floor": round(floor, 2), "Bust%": round(bust * 100, 1),
        "Team Runs": 4.4, "Matchup": 1.0,
        "HR": round(events["HR"], 3), "SB": round(events["SB"], 3),
        "Supports": [], "Cautions": [],
        # Expected events, so the frame can be simulated from. Same numbers Proj was
        # computed from, which is the invariant the simulator's marginal tests rest on.
        "E_PA": round(pa, 4),
        **{f"E_{k}": round(v, 5) for k, v in events.items()},
    }


def _pitcher(name, team, opponent, game, salary, skill):
    """One starter, same rule as the hitters: events first, then points, then bands."""
    from dfs.projections import (BUST_FIT, CEILING_Z, ER_OVERDISPERSION, FLOOR_FIT,
                                 PITCHER_BASE_VAR)
    from dfs.scoring import DK_PITCHER, pitcher_points

    innings = 4.6 + 1.4 * (skill - 1.0)
    batters = innings * 4.2
    k_rate = 0.235 * skill
    events = {
        "IP": innings,
        "K": batters * k_rate,
        "W": 0.44,
        "ER": 4.15 / skill / 9.0 * innings,
        "H": batters * 0.225 / skill,
        "BB": batters * 0.080,
        "HBP": batters * 0.010,
    }
    proj = pitcher_points(events)
    variance = (PITCHER_BASE_VAR + events["K"] * DK_PITCHER["K"] ** 2
                + events["ER"] * DK_PITCHER["ER"] ** 2 * ER_OVERDISPERSION
                + DK_PITCHER["W"] ** 2 * events["W"] * (1 - events["W"]))
    ceiling = proj + CEILING_Z * np.sqrt(variance)
    floor = max(0.0, FLOOR_FIT["P"]["slope"] * proj + FLOOR_FIT["P"]["intercept"])
    bust = min(0.95, max(0.05, BUST_FIT["P"]["slope"] * proj + BUST_FIT["P"]["intercept"]))
    return {
        "Name": name, "Team": team, "Opp": opponent, "Game": game,
        "Slot": 0, "Pos": "P", "DK Pos": "SP", "Bats": "",
        "MLBAM": abs(hash(name)) % 900000 + 100000, "Type": "P",
        "Opp SP": "", "Opp SP Hand": "R", "Lineup": "Confirmed", "PA": batters,
        "Salary": salary, "DK Avg": round(proj * 0.95, 2), "DK ID": abs(hash(name)) % 90000,
        "Proj": round(proj, 2), "Ceiling": round(ceiling, 2),
        "Floor": round(floor, 2), "Bust%": round(bust * 100, 1),
        "Team Runs": 4.4, "Matchup": 1.0,
        "IP": round(innings, 2), "K": round(events["K"], 2), "W%": 44.0,
        "ER": round(events["ER"], 2),
        "Supports": [], "Cautions": [],
        "E_BF": round(batters, 4), "E_KRATE": round(k_rate, 4),
        **{f"E_{k}": round(v, 5) for k, v in events.items()},
    }


def _price(skill, low, high, cheapest, dearest, rng, jitter=250):
    """Map skill onto DK's real salary range, with enough jitter to break collinearity."""
    share = (skill - low) / max(high - low, 1e-9)
    salary = cheapest + share * (dearest - cheapest) + rng.integers(-jitter, jitter + 1)
    return int(round(max(cheapest, min(dearest, salary)) / 100) * 100)


def make_slate(n_games=6, seed=1, with_ownership=True):
    """A legal synthetic DK Classic slate. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_games):
        away, home = TEAMS[2 * g], TEAMS[2 * g + 1]
        game = f"{away}@{home}"
        for team, opponent in ((away, home), (home, away)):
            # Skill is a multiplier on league-average rates, so price and production move
            # together the way they do on a real board without being perfectly collinear.
            #
            # Salary spans DK's real range, which matters more than it looks: a fixture
            # whose cheapest legal lineup costs $47k of a $50k cap makes the cap bind on
            # every solve, and tests of stacks, locks and ownership caps then fail on
            # affordability rather than on the constraint they are supposed to exercise.
            arm_skill = float(rng.uniform(0.80, 1.30))
            rows.append(_pitcher(
                f"{team} Arm", team, opponent, game,
                salary=_price(arm_skill, 0.80, 1.30, 4000, 10500, rng),
                skill=arm_skill))
            for slot, position in enumerate(HITTER_POSITIONS, start=1):
                skill = float(rng.uniform(0.70, 1.40))
                rows.append(_hitter(
                    f"{team} Bat{slot}", team, opponent, game, slot, position,
                    salary=_price(skill, 0.70, 1.40, 2100, 6300, rng),
                    skill=skill, rng=rng))

    frame = pd.DataFrame(rows)
    if with_ownership:
        from dfs.ownership import estimate_ownership
        frame = estimate_ownership(frame)
    return frame


@pytest.fixture
def slate():
    return make_slate()


@pytest.fixture
def small_slate():
    return make_slate(n_games=3, seed=5)


@pytest.fixture
def cached_dates():
    """Dates with real cached report data, or an empty list."""
    from dfs.evaluate import available_dates
    try:
        return available_dates()
    except Exception:
        return []


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: needs real cached report data in .cache/report_data")
