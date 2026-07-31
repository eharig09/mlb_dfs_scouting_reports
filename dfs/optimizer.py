"""DraftKings Classic lineup optimizer.

Exact optimization via scipy's MILP solver -- with ~140 players and 10 slots the problem
is tiny, so there is no reason to settle for a greedy approximation. Multiple lineups come
from re-solving with a constraint that limits overlap with everything already produced,
which is what makes the set genuinely diverse rather than ten near-copies of the same core.
"""

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from .salaries import canon_team
from .scoring import DK_SALARY_CAP

# DK Classic: 2 P plus one each of C/1B/2B/3B/SS and three OF.
ROSTER = {"P": 2, "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3}
ROSTER_SIZE = sum(ROSTER.values())

# DK contest rules.
MAX_HITTERS_PER_TEAM = 5
MIN_GAMES_REPRESENTED = 2

# Lineup-set defaults, tuned against actual results rather than left at the DK legal
# minimum. An overlap cap of ROSTER_SIZE - 1 only forbids exact duplicates, so a set of
# twenty comes back as twenty near-clones sharing nine players -- structurally useless for
# a tournament whatever the slate. Objective jitter then explores near-optimal lineups the
# strict re-solve would never reach.
#
# On the 2026-07-27 review, randomness 0.20 with overlap 6 produced the best set of every
# configuration tested (best 126.5 / mean 89.3 against 93.0 / 58.1 for the old defaults),
# and did so while projecting FEWER points -- diversity was worth more than precision.
# That is a single slate: revisit as more reviews accumulate.
DEFAULT_MAX_OVERLAP = 6
DEFAULT_RANDOMNESS = 0.20

OBJECTIVES = {"ceiling": "Ceiling", "proj": "Proj", "floor": "Floor",
              "leverage": "Lev Score"}

# Stack correlation. Summing individual ceilings treats hitters as independent, which they
# are not: teammates score in the same innings off the same pitcher, so a stack's real
# ceiling exceeds the sum of its parts. Without this the optimizer has no reason to stack
# at all and will happily spread eight hitters across eight games.
#
# The bonus is per *additional* hitter beyond the first from a team, applied as a fraction
# of that team's mean hitter value, and only on upside objectives -- correlation cuts both
# ways, so it is no help to a floor build (a stack busts together too).
STACK_BONUS = {2: 0.06, 3: 0.16, 4: 0.30, 5: 0.44}
CORRELATED_OBJECTIVES = {"ceiling", "proj", "leverage"}

# With focus teams named, everyone else is held below stack size. Two is deliberate: it is
# the largest group the bonus table still calls incidental, so a non-focus pairing that
# happens to be good is allowed while a real stack cannot form outside the focus set.
OFF_FOCUS_MAX_HITTERS = 2

# The mirror image of the stack bonus: rostering a pitcher against your own stack is
# negative correlation. Those two blocks of the lineup cannot both hit -- every run the
# stack scores comes off that pitcher's line -- so summing their projections counts points
# that can only be collected once. A stray single opposing hitter is noise; a real stack
# against your starter is a lineup that has bet on both sides of the same game.
#
# CONFLICT_MIN_HITTERS is what counts as "prominent". At 3 a mini-stack of two is left
# alone, which is deliberate: two hitters is often just two good prices in a good park.
CONFLICT_MIN_HITTERS = 3
# Penalty as a multiple of the conflicted pitcher's own objective value. At 1.0 the pitcher
# contributes nothing, so any comparable arm wins the slot and the pairing effectively
# disappears from the set -- while still being reachable if the slate leaves no alternative.
CONFLICT_PENALTY = 1.0


class OptimizerError(Exception):
    pass


def eligible_positions(row):
    """Roster slots a player can fill, from DK's multi-position string ("2B/3B")."""
    raw = str(row.get("DK Pos") or row.get("Pos") or "").strip().upper()
    if row.get("Type") == "P" or raw in {"P", "SP", "RP"}:
        return {"P"}
    slots = set()
    for token in raw.replace(",", "/").split("/"):
        token = token.strip()
        if token in {"LF", "CF", "RF"}:
            slots.add("OF")
        elif token in ROSTER:
            slots.add(token)
    return slots


def _match(players, names):
    """Resolve user-supplied names to row indices, case- and accent-insensitively."""
    from .salaries import normalize_name
    if not names:
        return [], []
    wanted = {normalize_name(n): n for n in names if str(n).strip()}
    keys = players["Name"].map(normalize_name)
    found, missing = [], []
    for key, original in wanted.items():
        hits = players.index[keys == key].tolist()
        if hits:
            found.extend(hits)
        else:
            missing.append(original)
    return found, missing


def build_pool(players, locks=None, excludes=None, min_proj=None, max_bust=None):
    """Filter the slate to a playable pool and resolve locks/excludes to indices."""
    pool = players[players["Salary"].notna()].copy()
    pool = pool[pool.apply(lambda r: bool(eligible_positions(r)), axis=1)]
    pool = pool.reset_index(drop=True)

    lock_idx, missing_locks = _match(pool, locks)
    exclude_idx, missing_excludes = _match(pool, excludes)

    # Thresholds must never drop a locked player -- an explicit lock outranks a filter.
    keep = pd.Series(True, index=pool.index)
    if min_proj is not None:
        keep &= pd.to_numeric(pool["Proj"], errors="coerce").fillna(0) >= min_proj
    if max_bust is not None and "Bust%" in pool.columns:
        keep &= pd.to_numeric(pool["Bust%"], errors="coerce").fillna(100) <= max_bust
    keep.loc[lock_idx] = True
    keep.loc[exclude_idx] = False

    pool = pool[keep].reset_index(drop=True)
    lock_idx, _ = _match(pool, locks)
    return pool, lock_idx, {"locks": missing_locks, "excludes": missing_excludes}


SLOTS = list(ROSTER)


def _player_selector(pool):
    """Matrix mapping the flattened player-slot variables back to 'is player i used'.

    The model assigns players to specific roster slots (variable per player per slot),
    which is what makes multi-position eligibility correct: a 2B/3B player satisfies one
    slot, not both. Selecting on players alone with '>= required' counts let a single
    multi-eligible hitter cover two slots and freed room for an extra pitcher.
    """
    n, s = len(pool), len(SLOTS)
    selector = np.zeros((n, n * s))
    for i in range(n):
        selector[i, i * s:(i + 1) * s] = 1
    return selector


def _team_hitter_masks(pool):
    """{team: per-player 0/1 mask of that team's hitters}, keyed by canonical code.

    Canonical keys mean a typed 'CHW' or 'OAK' finds the same hitters the slate files
    under CWS and ATH, rather than silently matching nothing.
    """
    is_hitter = (pool["Type"] == "H").to_numpy().astype(float)
    teams = pool["Team"].map(_team_key)
    masks = {}
    for team in teams.dropna().unique():
        if not team:
            continue
        mask = ((teams == team).to_numpy().astype(float)) * is_hitter
        if mask.sum():
            masks[team] = mask
    return masks


def _team_key(value):
    """Fold a team code to the slate's spelling, so typed input matches the data."""
    return canon_team(value)


def resolve_teams(pool, teams):
    """Split requested team codes into ones the slate has hitters for, and ones it doesn't."""
    available = set(_team_hitter_masks(pool))
    found, missing = [], []
    for team in teams or []:
        key = _team_key(team)
        if not key:
            continue
        (found if key in available else missing).append(key)
    return list(dict.fromkeys(found)), missing


def _pitcher_conflicts(pool):
    """[(pitcher index, opposing-hitter 0/1 mask)] for every pitcher facing rosterable bats.

    A pitcher's opponents are the hitters whose Team matches the pitcher's Opp, so this
    stays correct on a slate with a doubleheader or two same-named teams in different
    games -- it never has to guess from the game string.
    """
    teams = pool["Team"].map(_team_key)
    is_hitter = (pool["Type"] == "H").to_numpy()
    conflicts = []
    for index, row in pool.iterrows():
        if row.get("Type") != "P":
            continue
        opponent = _team_key(row.get("Opp"))
        if not opponent:
            continue
        mask = ((teams == opponent).to_numpy() & is_hitter).astype(float)
        if mask.sum():
            conflicts.append((index, mask))
    return conflicts


def _constraints(pool, lock_idx, stacks, max_hitters_per_team, min_games, stack_teams,
                 conflicts=(), conflict_min=CONFLICT_MIN_HITTERS, pins=None,
                 focus=None, off_focus_max=None, max_ownership=None):
    """Constraint rows over [player-slot vars | stack-level vars].

    Returns (rows, selector) where each row is (coefficients, lb, ub) sized to the full
    variable vector, so the caller does not have to track padding.
    """
    n, s = len(pool), len(SLOTS)
    selector = _player_selector(pool)          # n x (n*s): rows pick out one player
    levels = sorted(STACK_BONUS)
    n_z = len(stack_teams) * len(levels)
    n_y = len(conflicts)
    width = n * s + n_z + n_y

    def pad(block):
        """Widen an x-only coefficient block to the full variable vector."""
        block = np.atleast_2d(block)
        return np.hstack([block, np.zeros((block.shape[0], n_z + n_y))])

    rows = []

    # Each player fills at most one slot.
    rows.append((pad(selector), 0, 1))

    # Each roster slot is filled exactly the required number of times.
    for j, slot in enumerate(SLOTS):
        row = np.zeros((1, n * s))
        row[0, j::s] = 1
        rows.append((pad(row), ROSTER[slot], ROSTER[slot]))

    salary = pd.to_numeric(pool["Salary"], errors="coerce").fillna(0).to_numpy()
    rows.append((pad(salary @ selector), 0, DK_SALARY_CAP))

    # Total projected ownership across the ten. The cleanest way to force differentiation
    # from the field: unlike banning players or forcing stacks, it constrains the *lineup's*
    # relationship to the field while leaving every individual choice on merit. A lineup of
    # ten 30%-owned players is the same lineup thousands of other entries will submit, and
    # no amount of being right about those players pays when the prize is shared.
    if max_ownership is not None and "Own%" in pool.columns:
        own = pd.to_numeric(pool["Own%"], errors="coerce").fillna(0).to_numpy()
        rows.append((pad(own @ selector), 0, float(max_ownership)))

    team_masks = _team_hitter_masks(pool)
    for team, mask in team_masks.items():
        # Confining the *bonus* to the focus teams is not enough to confine the stacks. A
        # non-focus team with individually strong players clusters to three or four on its
        # own merit and outscores a bonus that, under a small-magnitude objective like
        # floor, is worth a fraction of a point. Capping everyone else below stack size is
        # what actually delivers "stacks form only there".
        cap = max_hitters_per_team
        if focus and off_focus_max is not None and team not in focus:
            cap = min(cap, off_focus_max)
        rows.append((pad(mask @ selector), 0, cap))

    for team, count in (stacks or {}).items():
        mask = team_masks.get(team)
        if mask is None or mask.sum() < count:
            have = 0 if mask is None else int(mask.sum())
            raise OptimizerError(
                f"stack {team}:{count} is impossible — only {have} {team} hitters in the pool"
            )
        rows.append((pad(mask @ selector), count, np.inf))

    for index in lock_idx:
        mask = np.zeros(n)
        mask[index] = 1
        rows.append((pad(mask @ selector), 1, 1))

    # Pins fix a player to one specific roster slot, not merely to the lineup. Late swap
    # needs this: DK matches an edited entry slot by slot, so a kept player who drifts from
    # the 2B column to the 3B column reads as two edited slots, one of them locked.
    for index, slot in (pins or {}).items():
        row = np.zeros((1, n * s))
        row[0, index * s + SLOTS.index(slot)] = 1
        rows.append((pad(row), 1, 1))

    # DK requires players from at least two games. Capping any single game's share of the
    # roster keeps this linear and is equivalent for a 10-player lineup.
    if min_games >= 2 and "Game" in pool.columns:
        for game in pool["Game"].dropna().unique():
            mask = (pool["Game"] == game).to_numpy().astype(float)
            rows.append((pad(mask @ selector), 0, ROSTER_SIZE - 1))

    # Stack-level indicators: z[team, k] may only be 1 when the lineup actually holds k or
    # more hitters from that team, i.e. sum(team hitters) - k*z >= 0. The objective gives
    # z a positive weight, so the solver turns on every level it has genuinely earned.
    for t, team in enumerate(stack_teams):
        mask = team_masks[team]
        for l, level in enumerate(levels):
            row = np.zeros((1, width))
            row[0, :n * s] = mask @ selector
            row[0, n * s + t * len(levels) + l] = -level
            rows.append((row, 0, np.inf))

    # Conflict indicators: y[c] is forced to 1 exactly when pitcher c is rostered AND at
    # least `conflict_min` of the hitters he faces are too. The objective weights y
    # negatively, so the solver keeps it at 0 wherever the constraint allows.
    #
    #   opposing_hitters + M*pitcher - M*y <= (conflict_min - 1) + M
    #
    # With M = max_hitters - conflict_min + 1 the row is slack whenever the pitcher is off
    # the roster (worst case max_hitters <= conflict_min - 1 + M holds by construction),
    # and collapses to opposing_hitters <= conflict_min - 1 when he is on it and y is 0.
    big_m = max_hitters_per_team - conflict_min + 1
    for c, (pitcher, mask) in enumerate(conflicts):
        if big_m < 1:
            break                   # the roster cap already forbids a stack this large
        pitcher_mask = np.zeros(n)
        pitcher_mask[pitcher] = 1
        row = np.zeros((1, width))
        row[0, :n * s] = (mask + big_m * pitcher_mask) @ selector
        row[0, n * s + n_z + c] = -big_m
        rows.append((row, -np.inf, (conflict_min - 1) + big_m))

    return rows, selector


def _eligibility_bounds(pool):
    """Upper bound of 0 on every player-slot pair the player cannot actually fill."""
    n, s = len(pool), len(SLOTS)
    upper = np.zeros(n * s)
    for i, (_, row) in enumerate(pool.iterrows()):
        for slot in eligible_positions(row):
            upper[i * s + SLOTS.index(slot)] = 1
    return Bounds(np.zeros(n * s), upper)


def optimize(players, n_lineups=1, objective="ceiling", locks=None, excludes=None,
             stacks=None, max_overlap=None, min_proj=None, max_bust=None,
             max_hitters_per_team=MAX_HITTERS_PER_TEAM, min_games=MIN_GAMES_REPRESENTED,
             randomness=DEFAULT_RANDOMNESS, seed=None, stack_bonus=None,
             exposure=None, boosts=None, conflict_penalty=CONFLICT_PENALTY,
             conflict_min_hitters=CONFLICT_MIN_HITTERS, focus_teams=None, pins=None,
             focus_exclusive=True, max_ownership=None):
    """Generate up to `n_lineups` distinct DK Classic lineups.

    max_overlap caps how many players a lineup may share with any earlier one. randomness
    jitters the objective per lineup, exploring near-optimal alternatives a strict re-solve
    would never surface. Both default to values tuned for tournament play -- pass
    max_overlap=ROSTER_SIZE - 1 and randomness=0 for the strict single-best-lineup behaviour.

    conflict_penalty taxes a pitcher rostered against `conflict_min_hitters` or more of the
    lineup's own hitters; 0 turns the tax off and allows the pairing freely.

    focus_teams builds stacks only from a hand-picked set: the correlation bonus is confined
    to them, and every other team is capped at OFF_FOCUS_MAX_HITTERS so an unfocused club
    cannot cluster into a stack on individual merit alone. Individual hitters from any team
    still fill the leftover slots, and pitchers are untouched. Pass focus_exclusive=False to
    confine only the bonus and leave the caps open -- the old, softer behaviour.

    pins is {player name: roster slot} and holds a player to that exact slot, leaving the
    rest of the roster free -- the late-swap case, where most of a lineup must stay put.
    """
    if objective not in OBJECTIVES:
        raise OptimizerError(f"objective must be one of {sorted(OBJECTIVES)}")

    # A pinned player has to survive pool filtering to be pinnable, so pins imply a lock.
    # Leaving that to the caller would fail as an infeasible solve with nothing to read.
    if pins:
        locks = list(dict.fromkeys(list(locks or []) + list(pins)))

    pool, lock_idx, missing = build_pool(players, locks, excludes, min_proj, max_bust)
    if len(pool) < ROSTER_SIZE:
        raise OptimizerError(f"pool has only {len(pool)} playable players; need {ROSTER_SIZE}")
    if len(lock_idx) > ROSTER_SIZE:
        raise OptimizerError(f"{len(lock_idx)} locks exceed the {ROSTER_SIZE}-player roster")

    column = OBJECTIVES[objective]
    if column not in pool.columns:
        raise OptimizerError(f"slate has no '{column}' column — run the board with salaries first")
    base = pd.to_numeric(pool[column], errors="coerce").fillna(0).to_numpy()
    if "Boost" in pool.columns:
        # Per-player multiplier from an edited pool file: >1 favours, <1 fades.
        base = base * pd.to_numeric(pool["Boost"], errors="coerce").fillna(1.0).to_numpy()

    if stack_bonus is None:
        stack_bonus = objective in CORRELATED_OBJECTIVES
    team_masks = _team_hitter_masks(pool)

    # The correlation bonus is the only reason the solver ever stacks, so confining it to
    # the focus teams is what makes stacks form there and nowhere else -- without banning
    # anyone, since every hitter keeps their own standalone value.
    focus, missing_focus = resolve_teams(pool, focus_teams)
    missing["focus_teams"] = missing_focus
    if focus_teams and not focus:
        raise OptimizerError(
            f"none of the focus teams {sorted(set(missing_focus))} have hitters on this slate"
        )
    bonus_teams = focus if focus else sorted(team_masks)
    stack_teams = bonus_teams if stack_bonus else []
    levels = sorted(STACK_BONUS)

    conflict_min_hitters = max(1, int(conflict_min_hitters))
    conflicts = _pitcher_conflicts(pool) if conflict_penalty else []

    pin_idx, missing_pins = {}, []
    for name, slot in (pins or {}).items():
        slot = str(slot or "").strip().upper()
        if slot not in ROSTER:
            raise OptimizerError(f"cannot pin {name} to unknown roster slot '{slot}'")
        found, _ = _match(pool, [name])
        if not found:
            missing_pins.append(name)
            continue
        if slot not in eligible_positions(pool.loc[found[0]]):
            raise OptimizerError(f"{name} is not eligible at {slot}")
        pin_idx[found[0]] = slot
    missing["pins"] = missing_pins

    rows, selector = _constraints(pool, lock_idx, stacks, max_hitters_per_team,
                                  min_games, stack_teams, conflicts, conflict_min_hitters,
                                  pin_idx, focus=focus,
                                  off_focus_max=(OFF_FOCUS_MAX_HITTERS if focus and focus_exclusive
                                                 else None),
                                  max_ownership=max_ownership)
    n_x = len(pool) * len(SLOTS)
    n_z = len(stack_teams) * len(levels)
    width = n_x + n_z + len(conflicts)

    # Correlation bonus per stack level, scaled by that team's mean hitter value so it
    # stays proportional to what the stack is actually worth.
    z_weights = np.zeros(n_z)
    for t, team in enumerate(stack_teams):
        mask = team_masks[team]
        mean_value = float((base * mask).sum() / max(1.0, mask.sum()))
        for l, level in enumerate(levels):
            increment = STACK_BONUS[level] - STACK_BONUS.get(level - 1, 0.0)
            z_weights[t * len(levels) + l] = increment * mean_value

    player_bounds = _eligibility_bounds(pool)
    bounds = Bounds(np.concatenate([player_bounds.lb, np.zeros(width - n_x)]),
                    np.concatenate([player_bounds.ub, np.ones(width - n_x)]))
    integrality = np.ones(width)
    max_overlap = DEFAULT_MAX_OVERLAP if max_overlap is None else max_overlap

    rng = np.random.default_rng(seed)
    lineups, chosen_sets = [], []

    # Exposure is a property of the SET of lineups, not of any one lineup, so it is
    # enforced by counting appearances as we go and hard-excluding anyone who has hit
    # their cap. Minimums are handled at the end by forcing the shortfall in.
    exposure = exposure or {}
    name_to_index = {name: i for i, name in enumerate(pool["Name"])}
    appearances = {name: 0 for name in exposure}

    for lineup_number in range(n_lineups):
        if boosts:
            # Redraw ranged boosts per lineup: a range is a diversity control, so it has
            # to vary between solves or it collapses to a single fixed value.
            from .pool import apply_boosts
            drawn = apply_boosts(pool, boosts, rng=rng)
            base_now = pd.to_numeric(pool[column], errors="coerce").fillna(0).to_numpy() * \
                pd.to_numeric(drawn["Boost"], errors="coerce").fillna(1.0).to_numpy()
        else:
            base_now = base
        weights = base_now.copy()
        if randomness > 0:
            weights = weights * (1 + rng.normal(0, randomness, size=len(weights)))

        # Overlap limits accumulate: each new lineup must differ from every previous one.
        extra = []
        for previous in chosen_sets:
            mask = np.isin(np.arange(len(pool)), list(previous)).astype(float)
            row = np.zeros((1, width))
            row[0, :n_x] = mask @ selector
            extra.append((row, 0, max_overlap))

        remaining = n_lineups - lineup_number
        for name, (low, high) in exposure.items():
            index = name_to_index.get(name)
            if index is None:
                continue
            mask = np.zeros(len(pool))
            mask[index] = 1
            row = np.zeros((1, width))
            row[0, :n_x] = mask @ selector
            used = appearances[name]
            if used >= high:
                extra.append((row, 0, 0))                       # cap reached: bar them
            elif low - used >= remaining:
                extra.append((row, 1, 1))                       # must appear from here on


        # Priced off the UNjittered value: the tax is a structural rule, not another thing
        # for randomness to explore, and jitter would let a cheaply-drawn pitcher buy his
        # way past it in whichever lineups happened to roll low.
        y_weights = np.array([-conflict_penalty * base_now[pitcher]
                              for pitcher, _ in conflicts])
        objective_vector = np.concatenate([weights @ selector, z_weights, y_weights])
        constraints = [LinearConstraint(a, lb, ub) for a, lb, ub in rows + extra]
        result = milp(c=-objective_vector, constraints=constraints,
                      integrality=integrality, bounds=bounds)
        if not result.success or result.x is None:
            break

        assignment = _decode(result.x[:n_x], len(pool))
        if len(assignment) != ROSTER_SIZE:
            break
        chosen_sets.append(set(assignment))
        built = _assemble(pool, assignment, objective, conflict_min_hitters)
        for name in built["players"]["Name"]:
            if name in appearances:
                appearances[name] += 1
        lineups.append(built)

    return lineups, pool, missing


def _decode(solution, n_players):
    """Flattened solution -> {player index: roster slot}."""
    s = len(SLOTS)
    assignment = {}
    for k, value in enumerate(solution):
        if value > 0.5:
            assignment[k // s] = SLOTS[k % s]
    return assignment


def lineup_conflicts(frame, min_hitters=1):
    """[(pitcher, opposing team, hitters)] for pitchers this lineup rostered against itself.

    Read back off the finished lineup rather than off the solver's indicator variables, so
    it reports what is actually on the roster whatever the penalty was set to.
    """
    hitters = frame[frame["Type"] == "H"]["Team"].map(_team_key).value_counts()
    found = []
    for _, row in frame[frame["Type"] == "P"].iterrows():
        count = int(hitters.get(_team_key(row.get("Opp")), 0))
        if count >= min_hitters:
            found.append((row["Name"], _team_key(row.get("Opp")), count))
    return sorted(found, key=lambda item: -item[2])


def _assemble(pool, assignment, objective, conflict_min=CONFLICT_MIN_HITTERS):
    """Build the lineup record. Roster slots come straight from the solver."""
    order = {slot: i for i, slot in enumerate(SLOTS)}
    rows = sorted(assignment, key=lambda i: (order.get(assignment[i], 99), -pool.loc[i, "Proj"]))
    slots = assignment

    frame = pool.loc[rows].copy()
    # "Roster", not "Slot" -- the pool already uses Slot for batting order.
    frame.insert(0, "Roster", [slots[i] for i in rows])
    return {
        # Reported at the taxed threshold whatever the penalty, so a run with the tax off
        # still shows which lineups bet on both sides of a game.
        "conflicts": lineup_conflicts(frame, min_hitters=conflict_min),
        "players": frame,
        "salary": int(pd.to_numeric(frame["Salary"], errors="coerce").sum()),
        "proj": round(pd.to_numeric(frame["Proj"], errors="coerce").sum(), 2),
        "ceiling": round(pd.to_numeric(frame["Ceiling"], errors="coerce").sum(), 2),
        "floor": round(pd.to_numeric(frame["Floor"], errors="coerce").sum(), 2),
        "objective": objective,
        "teams": frame[frame["Type"] == "H"]["Team"].value_counts().to_dict(),
    }


def stack_shapes(players, shape, top_teams=6, focus_teams=None):
    """Expand a shape like '4-3' into concrete team stack combinations.

    Candidate teams are ordered by summed hitter ceiling, so the combinations explored are
    the ones actually worth stacking rather than every arrangement on the slate.

    focus_teams narrows the candidates to a hand-picked list. Every team named is used --
    the top_teams cap is a way to keep an unguided search tractable, and an explicit choice
    should not be silently trimmed by it.
    """
    counts = [int(part) for part in str(shape).split("-") if part.strip().isdigit()]
    if not counts:
        raise OptimizerError(f"could not read stack shape '{shape}' (expected e.g. 4-3)")
    hitter_slots = ROSTER_SIZE - ROSTER["P"]
    if sum(counts) > hitter_slots:
        raise OptimizerError(
            f"stack shape '{shape}' needs {sum(counts)} hitters but the roster holds {hitter_slots}"
        )
    if max(counts) > MAX_HITTERS_PER_TEAM:
        raise OptimizerError(
            f"stack shape '{shape}' exceeds DK's {MAX_HITTERS_PER_TEAM}-hitter-per-team limit"
        )

    hitters = players[(players["Type"] == "H") & players["Salary"].notna()].copy()
    hitters["Team"] = hitters["Team"].map(_team_key)
    ranked = (hitters.groupby("Team")["Ceiling"].sum().sort_values(ascending=False)
              .index.tolist())

    if focus_teams:
        wanted = {_team_key(team) for team in focus_teams}
        ranked = [team for team in ranked if team in wanted]
        if len(ranked) < len(counts):
            raise OptimizerError(
                f"stack shape '{shape}' needs {len(counts)} teams; the focus list has "
                f"{len(ranked)} on this slate"
            )
    else:
        ranked = ranked[:top_teams]

    from itertools import permutations
    return [dict(zip(teams, counts)) for teams in permutations(ranked, len(counts))]
