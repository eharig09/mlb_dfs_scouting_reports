"""DraftKings Classic lineup optimizer.

Exact optimization via scipy's MILP solver -- with ~140 players and 10 slots the problem
is tiny, so there is no reason to settle for a greedy approximation. Multiple lineups come
from re-solving with a constraint that limits overlap with everything already produced,
which is what makes the set genuinely diverse rather than ten near-copies of the same core.
"""

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from .profiling import profiler
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
# a tournament whatever the slate. **The overlap cap is what makes a lineup set a set.**
#
# Objective jitter used to ride along with it at 0.20. It was wrong, and the way it got
# there is worth keeping: the 2026-07-27 review compared `rand .20 + overlap 6` against
# `rand 0 + overlap 9` -- two changes at once -- and credited the win to the jitter. Holding
# overlap at 6 and varying only randomness, over four snapshot-scored nights (8/17, 8/19,
# 8/20, 8/21 main, 20 lineups each):
#
#     overlap 6, randomness 0     mean 115.7   best 160.5    beat the old default 4 of 4
#     overlap 6, randomness .20   mean  95.0   best 132.0    (the old default)
#     overlap 6, randomness .40   mean  91.4   best 134.7
#
# Every no-randomness configuration beat every randomness one. It is not a floor-for-tail
# trade either: the worst lineup of the set improves too (66.2 against 60.6), and rand 0
# cleared the night's real cash line 4 of 4 against 3 of 4.
#
# The reason is upstream. Hitter rank skill is spearman ~0.19 overall and **0.02-0.09 within
# a salary tier** -- jittering an objective that barely orders its own candidates destroys
# the little ordering it has. Randomness would earn its place only against a projection good
# enough that near-optimal lineups were genuinely interchangeable. Revisit it if that
# changes; diversity now comes from the overlap cap alone.
DEFAULT_MAX_OVERLAP = 6
DEFAULT_RANDOMNESS = 0.0

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

# Hitters from one team that make a lineup "stacked" for exposure purposes. Matches
# dfs.exposure.STACK_AT on purpose: a cap you set is checked against the same definition the
# post-run report prints, so asking for "CWS in at most 40%" and reading back "Stk3+ 40%"
# are the same statement rather than two thresholds that happen to share a name.
STACK_EXPOSURE_AT = 3

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
                 focus=None, off_focus_max=None, max_ownership=None, slot_groups=None):
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

    # Require that a roster slot be filled by somebody from a named candidate group. This
    # is the paired-strategy backtest primitive: e.g. SS must come from the pay-up group,
    # while every other slot remains free to re-optimize. A player-slot constraint matters
    # for multi-position hitters; a generic player lock could put the candidate at 2B and
    # leave SS filled by the opposite strategy group.
    for slot, (indices, lower, upper) in (slot_groups or {}).items():
        row = np.zeros((1, n * s))
        column = SLOTS.index(slot)
        for index in indices:
            row[0, index * s + column] = 1
        rows.append((pad(row), lower, upper))

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


# The ten individual seats, so eligibility can be matched against them one by one.
SEATS = [slot for slot, count in ROSTER.items() for _ in range(count)]


def _fits_seats(slot_sets, seats=SEATS):
    """True when every candidate can be given a distinct roster seat.

    Counting forced players as "so many pitchers and so many hitters" is not the question
    DK asks: three shortstops are three hitters and still cannot be rostered together. The
    exposure pace forces players in before the solver ever sees them, so it has to check
    eligibility as an actual matching or it will hand the solver an impossible set.
    """
    match = {}

    def assign(candidate, seen):
        for seat, slot in enumerate(seats):
            if seat in seen or slot not in slot_sets[candidate]:
                continue
            seen.add(seat)
            if seat not in match or assign(match[seat], seen):
                match[seat] = candidate
                return True
        return False

    return all(assign(i, set()) for i in range(len(slot_sets)))


def optimize(players, n_lineups=1, objective="ceiling", locks=None, excludes=None,
             stacks=None, max_overlap=None, min_proj=None, max_bust=None,
             max_hitters_per_team=MAX_HITTERS_PER_TEAM, min_games=MIN_GAMES_REPRESENTED,
             randomness=DEFAULT_RANDOMNESS, seed=None, stack_bonus=None,
             exposure=None, boosts=None, conflict_penalty=CONFLICT_PENALTY,
             conflict_min_hitters=CONFLICT_MIN_HITTERS, focus_teams=None, pins=None,
             focus_exclusive=True, max_ownership=None,
             total_lineups=None, prior_appearances=None, prior_lineups=0,
             slot_groups=None, team_exposure=None, stack_at=STACK_EXPOSURE_AT,
             prior_team_stacks=None, on_lineup=None):
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

    team_exposure is {team: (min_lineups, max_lineups)} and limits how many lineups in the
    SET may stack that team -- where "stack" means `stack_at` or more of its hitters, the
    same threshold `dfs.exposure` reports against. It is the team-level twin of `exposure`
    and is enforced the same way: a team at its cap is held below stack size in later solves,
    and a team behind its minimum's pace is forced to stack in the next one. Ordinary
    non-stack hitters from a capped team are never blocked -- the cap is on *clustering*, not
    on the players.

    on_lineup(done, target), when given, fires once a lineup is accepted into the set --
    not once per solve, so an infeasible attempt that gets re-tried with a nudge dropped
    does not report a phantom lineup. `done` counts from `prior_lineups`, so a caller
    re-invoking this per stack-shape combination still reports progress against the whole
    requested set rather than restarting at zero on every combination.
    """
    if objective not in OBJECTIVES:
        raise OptimizerError(f"objective must be one of {sorted(OBJECTIVES)}")

    # Fold stack team codes the same way focus teams and the hitter masks are folded, so a
    # typed 'CHW' finds the hitters the slate files under CWS. `dfs.optimize` already does
    # this when parsing --stack; doing it only there left the library entry point behaving
    # differently from the CLI, and an unfolded code reads as "only 0 CHW hitters" rather
    # than working.
    if stacks:
        stacks = {canon_team(team): count for team, count in stacks.items()}

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

    slot_group_idx = {}
    for slot, specification in (slot_groups or {}).items():
        slot = str(slot or "").strip().upper()
        if slot not in ROSTER:
            raise OptimizerError(f"cannot constrain unknown roster slot '{slot}'")
        if isinstance(specification, dict):
            names = specification.get("names") or []
            lower = float(specification.get("min", 1))
            upper = float(specification.get("max", np.inf))
        else:
            names, lower, upper = specification, 1.0, np.inf
        if lower < 0 or upper < lower or (np.isfinite(upper) and upper > ROSTER[slot]):
            raise OptimizerError(
                f"invalid {slot} candidate-group bounds {lower:g}-{upper:g}")
        found, missing_group = _match(pool, names)
        eligible = sorted({index for index in found if slot in eligible_positions(pool.loc[index])})
        if not eligible and lower > 0:
            detail = f"; missing: {', '.join(missing_group)}" if missing_group else ""
            raise OptimizerError(f"no supplied players can fill constrained slot {slot}{detail}")
        slot_group_idx[slot] = (eligible, lower, upper)

    rows, selector = _constraints(pool, lock_idx, stacks, max_hitters_per_team,
                                  min_games, stack_teams, conflicts, conflict_min_hitters,
                                  pin_idx, focus=focus,
                                  off_focus_max=(OFF_FOCUS_MAX_HITTERS if focus and focus_exclusive
                                                 else None),
                                  max_ownership=max_ownership, slot_groups=slot_group_idx)
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
    # Minimums the pace ran out of room for. Reported with the finished set rather than
    # raised, so an oversubscribed board still hands back the lineups it did build.
    unmet = set()

    # Exposure is a property of the SET of lineups, not of any one lineup, so it is
    # enforced by counting appearances as we go: anyone who has hit their cap is barred
    # from the next solve, and anyone behind their minimum's pace is forced into it.
    exposure = exposure or {}
    name_to_index = {name: i for i, name in enumerate(pool["Name"])}
    appearances = {name: 0 for name in exposure}
    # A caller that builds one set across several solves -- the stack-shape loop does, one
    # solve per team pairing -- has to say so. Minimums are satisfied over the whole set, so
    # judging "how many lineups are left" from this call alone makes every minimum look
    # urgent on the first lineup and forces the entire exposure list in at once.
    for name, used in (prior_appearances or {}).items():
        if name in appearances:
            appearances[name] = int(used)
    budget = int(total_lineups if total_lineups else n_lineups)

    # Team stack exposure. Folded to canonical codes like every other team input, and checked
    # against the pool so a cap on a team with no hitters is reported rather than ignored.
    stack_at = max(2, int(stack_at))
    team_limits, missing_team_exposure = {}, []
    for team, span in (team_exposure or {}).items():
        key = canon_team(team)
        if key not in team_masks:
            missing_team_exposure.append(str(team))
            continue
        low, high = span
        team_limits[key] = (0 if low is None else int(low),
                            budget if high is None else int(high))
    missing["team_exposure"] = missing_team_exposure
    stacked_counts = {team: 0 for team in team_limits}
    for team, used in (prior_team_stacks or {}).items():
        key = canon_team(team)
        if key in stacked_counts:
            stacked_counts[key] = int(used)
    unmet_teams = set()

    # Fixed properties of the pool, read once: the pace consults them on every lineup.
    types = pool["Type"].to_numpy()
    salaries = pd.to_numeric(pool["Salary"], errors="coerce").fillna(0).to_numpy()
    slot_sets = [eligible_positions(pool.iloc[i]) for i in range(len(pool))]
    cheapest = float(salaries.min()) if len(salaries) else 0.0

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

        remaining = budget - prior_lineups - lineup_number
        done = prior_lineups + lineup_number

        # Minimums are held to a PACE rather than deferred to the end. A player wanted in
        # 30 of 50 lineups should be in roughly 3 of the first 5; letting him drift and then
        # forcing the whole shortfall in at once is what made a set that stops early miss its
        # minimums entirely, and what forced five pitchers into two slots when the set was
        # split across stack pairings.
        def _row(index):
            mask = np.zeros(len(pool))
            mask[index] = 1
            row = np.zeros((1, width))
            row[0, :n_x] = mask @ selector
            return row

        def _team_row(mask):
            """Constraint row counting how many of one team's hitters the lineup takes."""
            row = np.zeros((1, width))
            row[0, :n_x] = mask @ selector
            return row

        hitter_slots = ROSTER_SIZE - ROSTER["P"]
        # Slots a forced stack has already spoken for. Under-counts when a forced player is
        # himself part of the stack, which is the safe direction to be wrong in.
        free_hitter_slots = max(0, hitter_slots - sum((stacks or {}).values()))

        # --- team stack exposure -------------------------------------------------------
        # Caps are hard constraints on this solve; minimums are paced the same way player
        # minimums are, so a team wanted in half the set gets stacked steadily rather than
        # all at once at the end. A forced stack is appended to `extra` rather than to
        # `forced`, because it constrains a COUNT and not a named player -- there is nothing
        # for the infeasibility fallback to pop, so it is kept satisfiable by construction:
        # only one team is ever forced per lineup, and only when it can still fit.
        forced_stack_team = None
        stack_behind = []
        for team, (low, high) in team_limits.items():
            used = stacked_counts[team]
            if used >= high:
                # Held below stack size. Individual hitters from the team stay available.
                extra.append((_team_row(team_masks[team]), 0, stack_at - 1))
                continue
            if low <= 0:
                continue
            target = (2 * low * (done + 1) + budget) // (2 * max(budget, 1))
            last_chance = (low - used) >= remaining
            if target - used > 0 or last_chance:
                stack_behind.append((last_chance, (low - used) / max(remaining, 1), team))
        stack_behind.sort(key=lambda item: (item[0], item[1], rng.random()), reverse=True)
        for last_chance, _urgency, team in stack_behind:
            # An explicit --stack already dictates this team's count, and the hitter slots a
            # forced stack needs have to exist alongside it.
            if (stacks or {}).get(team) or stack_at > free_hitter_slots:
                if last_chance:
                    unmet_teams.add(team)
                continue
            extra.append((_team_row(team_masks[team]), stack_at, hitter_slots))
            forced_stack_team = team
            break
        for last_chance, _urgency, team in stack_behind:
            if last_chance and team != forced_stack_team:
                unmet_teams.add(team)

        behind, short = [], []
        for name, (low, high) in exposure.items():
            index = name_to_index.get(name)
            if index is None:
                continue
            used = appearances[name]
            if used >= high:
                extra.append((_row(index), 0, 0))               # cap reached: bar them
                continue
            if low <= 0:
                continue
            # Everyone still owed appearances, ranked by the share of the lineups left that
            # they have to appear in. This is what the top-up below draws from.
            if used < low:
                short.append(((low - used) / max(remaining, 1), name, index))
            # Nearest, not ceil and not floor -- both ends of the set are a trap.
            # Ceil made every minimum due on the very first lineup (ceil(low * 1 / budget)
            # is 1 for any minimum whatsoever), so 55 Min% players filled the whole ten-man
            # roster in file order and left the solver nothing to decide. Floor has the
            # mirror fault: it only reaches `low` on the final lineup, so every player's
            # last owed appearance came due simultaneously at lineup 150 and the ten that
            # did not fit finished one short. Rounding to nearest puts a player due half a
            # lineup either side of his own rate, which leaves slack at both ends.
            target = (2 * low * (done + 1) + budget) // (2 * max(budget, 1))
            deficit = target - used
            # Still kept as a backstop: if the lineups left are fewer than the shortfall,
            # this is the last chance regardless of what the pace says.
            last_chance = (low - used) >= remaining
            if deficit > 0 or last_chance:
                # Ranked by urgency -- the share of the lineups left that this player still
                # has to appear in -- rather than by raw deficit. Deficit alone is biased
                # towards big minimums: a player owed 10 of 100 remaining lineups outranks
                # one owed 3 of the next 3, and it is the second who is about to become
                # impossible. The random tiebreak matters too, since players tie constantly
                # and a stable sort let whoever sat highest in the file win every race.
                behind.append((last_chance, (low - used) / max(remaining, 1), name, index))

        behind.sort(key=lambda item: (item[0], item[1], rng.random()), reverse=True)

        forced, deferred = [], []

        def _take(name, index, last_chance=False):
            """Force this player if the resulting set can still be rostered for real."""
            trial = [i for _, _, i in forced] + [index]
            hitters = sum(1 for i in trial if types[i] != "P")
            # Distinct seats for everyone, and enough cap left to fill the rest of the
            # roster at all. Forcing a set that cannot be seated is what turned a
            # satisfiable board into "constraints are contradictory".
            if (hitters > free_hitter_slots
                    or not _fits_seats([slot_sets[i] for i in trial])
                    or salaries[trial].sum() + cheapest * (ROSTER_SIZE - len(trial))
                    > DK_SALARY_CAP):
                deferred.append((last_chance, name))
                return False
            forced.append((last_chance, name, index))
            return True

        for last_chance, _deficit, name, index in behind:
            _take(name, index, last_chance)

        # Top up to the rate the remaining minimums actually require. Pace alone leaves
        # early lineups nearly unforced -- a player owed 5% of 150 is not "due" until lineup
        # 19 -- so on a board whose minimums claim most of the slots the set banks a debt it
        # cannot repay at the end, and 148 good lineups died for a residual shortfall of 55.
        # Forcing the required rate from the first lineup spreads that debt evenly. It is
        # self-limiting: when minimums are modest the rate is low and nothing extra is
        # forced, so ordinary boards keep picking on merit.
        owed = sum(max(0, low - appearances[name])
                   for name, (low, _high) in exposure.items() if name in name_to_index)
        # Aim to finish the minimums slightly before the set ends rather than exactly on the
        # last lineup. Pacing at the bare average leaves no margin for the forcings the
        # solver later refuses -- a forced set can collide with the overlap limit or the cap
        # and get relaxed -- so the debt is never made up and the set lands a few
        # appearances short. Reserving a tail absorbs that.
        reserve = max(1, budget // 20)
        rate = -(-owed // max(remaining - reserve, 1))           # ceil: appearances per lineup
        if len(forced) < rate:
            already = {index for _, _, index in forced}
            short.sort(key=lambda item: (item[0], rng.random()), reverse=True)
            for _urgency, name, index in short:
                if len(forced) >= rate:
                    break
                if index in already:
                    continue
                _take(name, index)

        # A player who cannot fit is a shortfall, not a contradiction. Raising here threw
        # away every lineup already built -- 148 of 150 on the 8/1 board -- over minimums
        # that were merely oversubscribed. The set is reported against the minimums when it
        # finishes, so record it and keep building.
        for last_chance, name in deferred:
            if last_chance:
                unmet.add(name)



        # Priced off the UNjittered value: the tax is a structural rule, not another thing
        # for randomness to explore, and jitter would let a cheaply-drawn pitcher buy his
        # way past it in whichever lineups happened to roll low.
        y_weights = np.array([-conflict_penalty * base_now[pitcher]
                              for pitcher, _ in conflicts])
        objective_vector = np.concatenate([weights @ selector, z_weights, y_weights])

        # Forcing is a prediction about what will fit. Seat matching and the cap floor catch
        # the clashes visible player-by-player, but a forced set can still collide with a
        # stack, an overlap limit or the cap in combination -- so an infeasible solve hands
        # back its least urgent pace nudge and tries again instead of condemning the whole
        # board. `forced` is urgency-ordered, so popping the tail always gives up the
        # cheapest nudge first and only touches a genuinely due minimum once nothing else
        # is left; that minimum is then reported as unmet rather than ending the set.
        attempt = list(forced)
        while True:
            forcing = [(_row(index), 1, 1) for _, _, index in attempt]
            constraints = [LinearConstraint(a, lb, ub)
                           for a, lb, ub in rows + extra + forcing]
            with profiler.stage("solve", vars=width, players=len(pool)):
                result = milp(c=-objective_vector, constraints=constraints,
                              integrality=integrality, bounds=bounds)
            if (result.success and result.x is not None) or not attempt:
                break
            last_chance, name, _index = attempt.pop()
            if last_chance:
                unmet.add(name)

        if not result.success or result.x is None:
            # Running out part-way through a set is normal -- overlap limits and exposure
            # caps legitimately exhaust the alternatives, and the lineups already built are
            # good. Failing on the *first* is not: nothing was asked of the solver except
            # the constraints themselves, so they are contradictory. Returning an empty list
            # there let the stack-shape loop discard thirty infeasible solves in silence and
            # report only "No feasible lineup".
            if lineup_number == 0 and not lineups:
                raise OptimizerError(
                    "constraints are contradictory — no lineup satisfies them at all"
                    + (f" (stacks {stacks})" if stacks else "")
                    + (f", {len(exposure)} exposure rule(s)" if exposure else "")
                    + (f", {len(lock_idx)} lock(s)" if lock_idx else "")
                )
            break

        assignment = _decode(result.x[:n_x], len(pool))
        if len(assignment) != ROSTER_SIZE:
            break
        chosen_sets.append(set(assignment))
        built = _assemble(pool, assignment, objective, conflict_min_hitters)
        for name in built["players"]["Name"]:
            if name in appearances:
                appearances[name] += 1
        if stacked_counts:
            # Counted off the built lineup rather than the solver's own team rows, so the
            # tally matches what dfs.exposure will report from the same finished set.
            roster = built["players"]
            taken = roster[roster["Roster"] != "P"]["Team"].map(canon_team).value_counts()
            for team in stacked_counts:
                if int(taken.get(team, 0)) >= stack_at:
                    stacked_counts[team] += 1
        lineups.append(built)
        if on_lineup is not None:
            on_lineup(prior_lineups + lineup_number + 1, budget)

    # Minimums the roster could not make room for. The CLI recomputes the real shortfall
    # from the finished set, but a library caller that does not gets the signal here.
    missing["exposure_minimums"] = sorted(unmet)
    missing["team_stack_minimums"] = sorted(unmet_teams)
    missing["team_stacks_built"] = dict(stacked_counts)
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


def _parse_shape(shape):
    """'4-3' -> [4, 3], rejecting anything that is not a run of dashed integers.

    This used to keep only the parts that passed `.isdigit()` and silently drop the rest,
    which meant '5-3,5-2' split to ['5', '3,5', '2'], lost the middle, and came back as the
    shape 5-2 without a word. Malformed input now raises instead of being quietly reinterpreted.
    """
    parts = [part.strip() for part in str(shape).split("-")]
    if not all(part.isdigit() for part in parts) or not parts:
        raise OptimizerError(
            f"could not read stack shape '{shape}' (expected e.g. '4-3'; separate several "
            f"shapes with commas, e.g. '5-3,5-2')"
        )
    counts = [int(part) for part in parts]
    hitter_slots = ROSTER_SIZE - ROSTER["P"]
    if sum(counts) > hitter_slots:
        raise OptimizerError(
            f"stack shape '{shape}' needs {sum(counts)} hitters but the roster holds {hitter_slots}"
        )
    if max(counts) > MAX_HITTERS_PER_TEAM:
        raise OptimizerError(
            f"stack shape '{shape}' exceeds DK's {MAX_HITTERS_PER_TEAM}-hitter-per-team limit"
        )
    return counts


def stack_shapes(players, shape, top_teams=6, focus_teams=None):
    """Expand a shape like '4-3' into concrete team stack combinations.

    Several shapes may be given comma-separated -- '5-3,5-2,5'. The results are round-robin
    interleaved rather than concatenated, because callers consume this list by cycling it and
    the shapes produce wildly different counts: on a 20-team slate '5-3' is 380 combinations
    and '5' is 20, so plain concatenation would give the one-stack shape 5% of the lineups
    instead of a third of them. Interleaving makes position in the list mean "how many of
    each shape so far", which is what the cycling caller actually wants.

    Candidate teams are ordered by summed hitter ceiling, so the combinations explored are
    the ones actually worth stacking rather than every arrangement on the slate.

    focus_teams narrows the candidates to a hand-picked list. Every team named is used --
    the top_teams cap is a way to keep an unguided search tractable, and an explicit choice
    should not be silently trimmed by it.

    top_teams=None lifts the cap entirely. It exists because the cost of widening depends on
    the shape: a two-stack shape grows as permutations (6 teams -> 30 combinations, 20 -> 380),
    but a one-stack shape like '5' grows linearly, so opening it to the whole slate is cheap
    and leaves the choice of team to the solver rather than to a ceiling ranking made before
    salary is considered.
    """
    shapes = [part for part in str(shape).split(",") if part.strip()]
    if not shapes:
        raise OptimizerError(f"could not read stack shape '{shape}' (expected e.g. 4-3)")

    hitters = players[(players["Type"] == "H") & players["Salary"].notna()].copy()
    hitters["Team"] = hitters["Team"].map(_team_key)
    ranked = (hitters.groupby("Team")["Ceiling"].sum().sort_values(ascending=False)
              .index.tolist())

    if focus_teams:
        wanted = {_team_key(team) for team in focus_teams}
        ranked = [team for team in ranked if team in wanted]
    else:
        ranked = ranked[:top_teams]

    from itertools import permutations, zip_longest
    per_shape = []
    for one in shapes:
        counts = _parse_shape(one)
        if len(ranked) < len(counts):
            raise OptimizerError(
                f"stack shape '{one.strip()}' needs {len(counts)} teams; "
                f"{'the focus list has' if focus_teams else 'the slate offers'} "
                f"{len(ranked)}"
            )
        per_shape.append([dict(zip(teams, counts))
                          for teams in permutations(ranked, len(counts))])

    if len(per_shape) == 1:
        return per_shape[0]

    # Shorter shapes are cycled rather than exhausted. '5' has one combination per team (20)
    # against 380 for '5-3', so dropping it once spent would leave a 150-lineup run at 13%
    # one-stack instead of the third that was asked for. Repeats are harmless: the caller
    # varies the seed per solve and skips lineups it has already seen, so the same team
    # constraint yields a different lineup each time it comes round.
    longest = max(len(group) for group in per_shape)
    combos, seen = [], set()
    for index in range(longest):
        for group in per_shape:
            combo = group[index % len(group)]
            # Only a repeated shape can collide -- '5-2' and '5' produce different dicts even
            # for the same team -- so this guards against '5-3,5-3' rather than a real overlap.
            key = (index // len(group), tuple(sorted(combo.items())))
            if key not in seen:
                seen.add(key)
                combos.append(combo)
    return combos
