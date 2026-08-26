"""DraftKings NFL Classic lineup optimizer.

Exact optimization via scipy's MILP solver. Structurally this is `dfs.optimizer` with the
baseball taken out: the assignment model, the overlap-limited re-solve and the objective
jitter all port unchanged, because none of them were ever about baseball.

**Why the assignment model is the whole point here.** The MLB optimizer models a variable
per player *per roster slot* rather than one variable per player, and the comment there
explains it was needed because a 2B/3B eligible hitter would otherwise satisfy two slots at
once. NFL's FLEX is that problem in its pure form: every RB, WR and TE is eligible at two
seats simultaneously, on every lineup, and a "select 9 players with >= 2 RB, >= 3 WR, >= 1
TE" formulation quietly produces lineups that cannot be entered. Because the model already
assigns to seats, FLEX costs exactly one line in `eligible_positions`.

**What is deliberately NOT ported.** `dfs.optimizer` carries a soft stack bonus that adds a
per-level reward to the objective for clustering hitters. Two things say to leave it out.
It is tuned on MLB data (`STACK_BONUS = {2: 0.06 ... 5: 0.44}`), and the MLB work that
followed found the soft bonus strictly dominated by asking for the stack shape outright.
So stacks here are hard constraints only. The exposure-pacing machinery is also absent --
it is genuinely sport-agnostic and should be ported verbatim later rather than rewritten.
"""

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from .profiling import profiler
from .salaries import canon_team, normalize_name
from .scoring import DK_SALARY_CAP, FLEX_POSITIONS

# DK Classic: QB, 2 RB, 3 WR, TE, FLEX, DST.
ROSTER = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1, "DST": 1}
ROSTER_SIZE = sum(ROSTER.values())
SLOTS = list(ROSTER)
# The nine individual seats, so eligibility can be matched against them one by one.
SEATS = [slot for slot, count in ROSTER.items() for _ in range(count)]

# DK contest rules. Unlike MLB Classic there is no per-team cap in NFL Classic, so the
# default is the roster itself -- a real limit, just not a binding one. The parameter stays
# because it is how a caller keeps a single game from swallowing a lineup.
MAX_PLAYERS_PER_TEAM = ROSTER_SIZE
MIN_GAMES_REPRESENTED = 2

# Lineup-set defaults. Carried over from MLB as a STARTING POINT ONLY and not yet fitted on
# anything: over there, randomness 0.20 with overlap 6 beat every other configuration on a
# ten-player roster. This roster is nine, so even the overlap number is not on the same
# scale, and both should be re-measured against real NFL results before being trusted.
DEFAULT_MAX_OVERLAP = 5
DEFAULT_RANDOMNESS = 0.20

OBJECTIVES = {"ceiling": "Ceiling", "proj": "Proj", "floor": "Floor",
              "leverage": "Lev Score"}

# Positions that make up the passing game. This is the NFL stacking unit: a QB and the
# players who catch his throws share the same events, exactly as MLB teammates share
# innings. Running backs are excluded on purpose -- a rushing touchdown is a drive that did
# NOT end in a passing touchdown, so a back is at best uncorrelated with his own quarterback
# and on the goal line is actively negative.
PASS_GAME_POSITIONS = frozenset({"QB", "WR", "TE"})

# A DST rostered against several of your own skill players is the mirror of MLB's pitcher
# against his own stack: the two halves of the lineup cannot both hit, because every point
# your stack scores is a point your defense allowed. MLB prices this as a penalty so the
# pairing stays reachable; here it is a hard cap, which is simpler and is the behaviour to
# revisit once there are results to price it against.
DST_CONFLICT_MAX = 2

# Pass-game players from one club that make a lineup "stacked" for exposure purposes.
# Three, not two: on a nine-man roster carrying three WR slots plus a FLEX, two from one
# club happens by accident often enough that counting it as a stack would fire a team cap
# on lineups nobody would call stacked. Whatever this is, the exposure report has to count
# a stack the same way, or a cap you set is checked against a different definition.
STACK_EXPOSURE_AT = 3


class OptimizerError(Exception):
    pass


def eligible_positions(row):
    """Roster slots a player can fill, from DK's position string.

    The FLEX line is the entire NFL adaptation. Everything downstream -- the eligibility
    bounds, the seat matching, the decode -- already handles a player who can sit in more
    than one seat, because multi-position eligibility was solved for baseball first.
    """
    raw = str(row.get("DK Pos") or row.get("Pos") or "").strip().upper()
    slots = set()
    for token in raw.replace(",", "/").split("/"):
        token = token.strip()
        if token in {"D", "DEF", "D/ST", "DST"}:
            slots.add("DST")
        elif token in ROSTER and token != "FLEX":
            slots.add(token)
    if slots & FLEX_POSITIONS:
        slots.add("FLEX")
    return slots


def _match(players, names):
    """Resolve user-supplied names to row indices, case- and accent-insensitively."""
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


def build_pool(players, locks=None, excludes=None, min_proj=None):
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
    keep.loc[lock_idx] = True
    keep.loc[exclude_idx] = False

    pool = pool[keep].reset_index(drop=True)
    lock_idx, _ = _match(pool, locks)
    return pool, lock_idx, {"locks": missing_locks, "excludes": missing_excludes}


def _player_selector(pool):
    """Matrix mapping the flattened player-slot variables back to 'is player i used'."""
    n, s = len(pool), len(SLOTS)
    selector = np.zeros((n, n * s))
    for i in range(n):
        selector[i, i * s:(i + 1) * s] = 1
    return selector


def _position_series(pool):
    return pool.apply(lambda r: eligible_positions(r), axis=1)


def _team_pass_masks(pool):
    """{team: per-player 0/1 mask of that team's pass-game players}, canonical keys."""
    positions = _position_series(pool)
    is_pass = positions.map(lambda s: bool(s & PASS_GAME_POSITIONS)).to_numpy().astype(float)
    teams = pool["Team"].map(canon_team)
    masks = {}
    for team in teams.dropna().unique():
        if not team:
            continue
        mask = ((teams == team).to_numpy().astype(float)) * is_pass
        if mask.sum():
            masks[team] = mask
    return masks


def _team_skill_masks(pool):
    """{team: mask of that team's non-DST players}. Used for the per-team cap, which is
    about roster concentration rather than about correlation."""
    positions = _position_series(pool)
    is_skill = positions.map(lambda s: "DST" not in s).to_numpy().astype(float)
    teams = pool["Team"].map(canon_team)
    masks = {}
    for team in teams.dropna().unique():
        if not team:
            continue
        mask = ((teams == team).to_numpy().astype(float)) * is_skill
        if mask.sum():
            masks[team] = mask
    return masks


def resolve_teams(pool, teams):
    """Split requested team codes into ones the slate has pass-game players for, and not."""
    available = set(_team_pass_masks(pool))
    found, missing = [], []
    for team in teams or []:
        key = canon_team(team)
        if not key:
            continue
        (found if key in available else missing).append(key)
    return list(dict.fromkeys(found)), missing


def _dst_conflicts(pool):
    """[(dst index, opposing skill-player mask)] for each DST facing rosterable players.

    Opponents are resolved off the DST's own `Opp`, never guessed from the game string, so
    this stays correct when two clubs share a slate week in any arrangement.
    """
    positions = _position_series(pool)
    teams = pool["Team"].map(canon_team)
    is_skill = positions.map(lambda s: "DST" not in s).to_numpy()
    conflicts = []
    for index, row in pool.iterrows():
        if "DST" not in positions.loc[index]:
            continue
        opponent = canon_team(row.get("Opp"))
        if not opponent:
            continue
        mask = ((teams == opponent).to_numpy() & is_skill).astype(float)
        if mask.sum():
            conflicts.append((index, mask))
    return conflicts


def _constraints(pool, lock_idx, stacks, max_players_per_team, min_games,
                 dst_conflict_max, pins=None, max_ownership=None, slot_groups=None):
    """Constraint rows over the flattened player-slot variables.

    Returns (rows, selector) where each row is (coefficients, lb, ub) sized to the full
    variable vector, so the caller does not have to track padding.
    """
    n, s = len(pool), len(SLOTS)
    selector = _player_selector(pool)
    rows = []

    # Each player fills at most one slot. This is what makes FLEX correct: a WR assigned to
    # the FLEX seat is not also available at WR.
    rows.append((selector, 0, 1))

    # Each roster slot is filled exactly the required number of times.
    for j, slot in enumerate(SLOTS):
        row = np.zeros((1, n * s))
        row[0, j::s] = 1
        rows.append((row, ROSTER[slot], ROSTER[slot]))

    salary = pd.to_numeric(pool["Salary"], errors="coerce").fillna(0).to_numpy()
    rows.append((np.atleast_2d(salary @ selector), 0, DK_SALARY_CAP))

    # Total projected ownership across the nine: the cleanest way to force differentiation
    # from the field, because it constrains the lineup's relationship to the field while
    # leaving every individual choice on merit.
    if max_ownership is not None and "Own%" in pool.columns:
        own = pd.to_numeric(pool["Own%"], errors="coerce").fillna(0).to_numpy()
        rows.append((np.atleast_2d(own @ selector), 0, float(max_ownership)))

    skill_masks = _team_skill_masks(pool)
    for _team, mask in skill_masks.items():
        rows.append((np.atleast_2d(mask @ selector), 0, max_players_per_team))

    pass_masks = _team_pass_masks(pool)
    for team, count in (stacks or {}).items():
        mask = pass_masks.get(team)
        if mask is None or mask.sum() < count:
            have = 0 if mask is None else int(mask.sum())
            raise OptimizerError(
                f"stack {team}:{count} is impossible — only {have} {team} pass-game "
                f"players in the pool"
            )
        rows.append((np.atleast_2d(mask @ selector), count, np.inf))

    for index in lock_idx:
        mask = np.zeros(n)
        mask[index] = 1
        rows.append((np.atleast_2d(mask @ selector), 1, 1))

    # Pins fix a player to one specific roster slot, not merely to the lineup. Late swap
    # needs this: DK matches an edited entry slot by slot, so a kept player who drifts from
    # the WR column to the FLEX column reads as two edited slots.
    for index, slot in (pins or {}).items():
        row = np.zeros((1, n * s))
        row[0, index * s + SLOTS.index(slot)] = 1
        rows.append((row, 1, 1))

    # Require that a roster slot be filled from a named candidate group. A player-slot
    # constraint rather than a lock, because a FLEX-eligible player locked generically could
    # satisfy the requirement from the wrong seat.
    for slot, (indices, lower, upper) in (slot_groups or {}).items():
        row = np.zeros((1, n * s))
        column = SLOTS.index(slot)
        for index in indices:
            row[0, index * s + column] = 1
        rows.append((row, lower, upper))

    # DK requires players from at least two games. Capping any single game's share of the
    # roster keeps this linear and is equivalent for a nine-player lineup.
    if min_games >= 2 and "Game" in pool.columns:
        for game in pool["Game"].dropna().unique():
            mask = (pool["Game"] == game).to_numpy().astype(float)
            rows.append((np.atleast_2d(mask @ selector), 0, ROSTER_SIZE - 1))

    # A DST may not be rostered alongside more than `dst_conflict_max` of the offense it
    # faces. Written as: opposing_players + M*dst <= max + M, which is slack whenever the
    # DST is off the roster and collapses to the cap when it is on.
    if dst_conflict_max is not None:
        big_m = ROSTER_SIZE
        for dst_index, mask in _dst_conflicts(pool):
            dst_mask = np.zeros(n)
            dst_mask[dst_index] = 1
            row = np.atleast_2d((mask + big_m * dst_mask) @ selector)
            rows.append((row, -np.inf, dst_conflict_max + big_m))

    return rows, selector


def _eligibility_bounds(pool):
    """Upper bound of 0 on every player-slot pair the player cannot actually fill."""
    n, s = len(pool), len(SLOTS)
    upper = np.zeros(n * s)
    for i in range(n):
        for slot in eligible_positions(pool.iloc[i]):
            upper[i * s + SLOTS.index(slot)] = 1
    return Bounds(np.zeros(n * s), upper)


def _fits_seats(slot_sets, seats=SEATS):
    """True when every candidate can be given a distinct roster seat.

    Counting forced players by position is not the question DK asks: four running backs are
    four skill players and still cannot be rostered together, because there are only three
    seats they can occupy. This is the check that a lock list is satisfiable at all.
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
             stacks=None, max_overlap=None, min_proj=None,
             max_players_per_team=MAX_PLAYERS_PER_TEAM, min_games=MIN_GAMES_REPRESENTED,
             randomness=DEFAULT_RANDOMNESS, seed=None, pins=None,
             max_ownership=None, slot_groups=None, dst_conflict_max=DST_CONFLICT_MAX,
             exposure=None, team_exposure=None, stack_at=STACK_EXPOSURE_AT,
             prior_appearances=None, prior_team_stacks=None, total_lineups=None,
             prior_lineups=0):
    """Generate up to `n_lineups` distinct DK NFL Classic lineups.

    max_overlap caps how many players a lineup may share with any earlier one; randomness
    jitters the objective per lineup so the set explores near-optimal alternatives a strict
    re-solve would never reach. Pass max_overlap=ROSTER_SIZE - 1 and randomness=0 for the
    single-best-lineup behaviour.

    stacks is {team: count} over PASS-GAME players (QB/WR/TE) -- see PASS_GAME_POSITIONS
    for why a running back does not count toward his own team's stack.
    """
    if objective not in OBJECTIVES:
        raise OptimizerError(f"objective must be one of {sorted(OBJECTIVES)}")

    if stacks:
        stacks = {canon_team(team): count for team, count in stacks.items()}
    # A pinned player has to survive pool filtering to be pinnable, so pins imply a lock.
    if pins:
        locks = list(dict.fromkeys(list(locks or []) + list(pins)))

    pool, lock_idx, missing = build_pool(players, locks, excludes, min_proj)
    if len(pool) < ROSTER_SIZE:
        raise OptimizerError(f"pool has only {len(pool)} playable players; need {ROSTER_SIZE}")
    if len(lock_idx) > ROSTER_SIZE:
        raise OptimizerError(f"{len(lock_idx)} locks exceed the {ROSTER_SIZE}-player roster")
    if lock_idx and not _fits_seats([eligible_positions(pool.iloc[i]) for i in lock_idx]):
        raise OptimizerError(
            "locked players cannot all be seated at once — check how many share a position")

    column = OBJECTIVES[objective]
    if column not in pool.columns:
        raise OptimizerError(f"slate has no '{column}' column — run the board with salaries first")
    base = pd.to_numeric(pool[column], errors="coerce").fillna(0).to_numpy()
    if "Boost" in pool.columns:
        base = base * pd.to_numeric(pool["Boost"], errors="coerce").fillna(1.0).to_numpy()

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
            raise OptimizerError(f"invalid {slot} candidate-group bounds {lower:g}-{upper:g}")
        found, missing_group = _match(pool, names)
        eligible = sorted({i for i in found if slot in eligible_positions(pool.loc[i])})
        if not eligible and lower > 0:
            detail = f"; missing: {', '.join(missing_group)}" if missing_group else ""
            raise OptimizerError(f"no supplied players can fill constrained slot {slot}{detail}")
        slot_group_idx[slot] = (eligible, lower, upper)

    rows, selector = _constraints(pool, lock_idx, stacks, max_players_per_team, min_games,
                                  dst_conflict_max, pin_idx, max_ownership, slot_group_idx)
    width = len(pool) * len(SLOTS)
    bounds = _eligibility_bounds(pool)
    integrality = np.ones(width)
    max_overlap = DEFAULT_MAX_OVERLAP if max_overlap is None else max_overlap

    rng = np.random.default_rng(seed)
    lineups, chosen_sets = [], []
    # Minimums the pace ran out of room for. Reported with the finished set rather than
    # raised, so an oversubscribed board still hands back the lineups it did build.
    unmet, unmet_teams = set(), set()

    # **Exposure is a property of the SET of lineups, not of any one lineup**, so it is
    # enforced by counting appearances as the set is built: anyone at their cap is barred
    # from the next solve, anyone behind their minimum's pace is forced into it. Lifted
    # from `dfs.optimizer`, which is where the pacing arithmetic below was learned.
    exposure = exposure or {}
    name_to_index = {name: i for i, name in enumerate(pool["Name"])}
    appearances = {name: 0 for name in exposure}
    # A caller building one set across several solves -- the stack-shape loop does exactly
    # that, one solve per anchor -- has to say so. Minimums are satisfied over the whole
    # set, so judging "how many lineups are left" from this call alone makes every minimum
    # look urgent on the first lineup and forces the entire exposure list in at once.
    for name, used in (prior_appearances or {}).items():
        if name in appearances:
            appearances[name] = int(used)
    budget = int(total_lineups if total_lineups else n_lineups)

    stack_at = max(2, int(stack_at))
    pass_masks = _team_pass_masks(pool)
    team_limits, missing_team_exposure = {}, []
    for team, span in (team_exposure or {}).items():
        key = canon_team(team)
        if key not in pass_masks:
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

    def _player_row(index):
        mask = np.zeros(len(pool))
        mask[index] = 1
        return np.atleast_2d(mask @ selector)

    pass_slots = ROSTER["QB"] + ROSTER["WR"] + ROSTER["TE"] + ROSTER["FLEX"]

    for lineup_number in range(n_lineups):
        weights = base.copy()
        if randomness > 0:
            weights = weights * (1 + rng.normal(0, randomness, size=len(weights)))

        # Overlap limits accumulate: each new lineup must differ from every previous one.
        extra = []
        for previous in chosen_sets:
            mask = np.isin(np.arange(len(pool)), list(previous)).astype(float)
            extra.append((np.atleast_2d(mask @ selector), 0, max_overlap))

        remaining = max(1, budget - prior_lineups - lineup_number)
        done = prior_lineups + lineup_number
        free_pass_slots = max(0, pass_slots - sum((stacks or {}).values()))

        # --- team stack exposure -------------------------------------------------------
        # Caps are hard constraints on this solve; minimums are paced the way player
        # minimums are, so a club wanted in half the set gets stacked steadily rather than
        # all at once at the end. A forced stack constrains a COUNT rather than a named
        # player, so there is nothing an infeasibility fallback could pop -- it is kept
        # satisfiable by construction: one club at most per lineup, only when it still fits.
        forced_stack_team = None
        stack_behind = []
        for team, (low, high) in team_limits.items():
            used = stacked_counts[team]
            if used >= high:
                # Held below stack size. Individual players from the club stay available.
                extra.append((np.atleast_2d(pass_masks[team] @ selector), 0, stack_at - 1))
                continue
            if low <= 0:
                continue
            target = (2 * low * (done + 1) + budget) // (2 * max(budget, 1))
            last_chance = (low - used) >= remaining
            if target - used > 0 or last_chance:
                stack_behind.append((last_chance, (low - used) / remaining, team))
        stack_behind.sort(key=lambda item: (item[0], item[1], rng.random()), reverse=True)
        for last_chance, _urgency, team in stack_behind:
            # An explicit --stack already dictates this club's count, and the pass-game
            # slots a forced stack needs have to exist alongside it.
            if (stacks or {}).get(team) or stack_at > free_pass_slots:
                if last_chance:
                    unmet_teams.add(team)
                continue
            extra.append((np.atleast_2d(pass_masks[team] @ selector), stack_at, pass_slots))
            forced_stack_team = team
            break
        for last_chance, _urgency, team in stack_behind:
            if last_chance and team != forced_stack_team:
                unmet_teams.add(team)

        # --- player exposure -----------------------------------------------------------
        behind = []
        for name, (low, high) in exposure.items():
            index = name_to_index.get(name)
            if index is None:
                continue
            used = appearances[name]
            if used >= high:
                extra.append((_player_row(index), 0, 0))        # cap reached: bar them
                continue
            if low <= 0:
                continue
            # **Nearest, not ceil and not floor** -- both ends of the set are a trap, and
            # `dfs.optimizer` documents both failures. Ceil makes every minimum due on the
            # very first lineup, so a long Min% list fills the whole roster in file order
            # and leaves the solver nothing to decide. Floor only reaches `low` on the final
            # lineup, so every owed appearance falls due at once and whatever does not fit
            # finishes short. Nearest puts a player due half a lineup either side of his own
            # rate, which leaves slack at both ends.
            target = (2 * low * (done + 1) + budget) // (2 * max(budget, 1))
            last_chance = (low - used) >= remaining
            if target - used > 0 or last_chance:
                behind.append((last_chance, (low - used) / remaining, name, index))
        behind.sort(key=lambda item: (item[0], item[1]), reverse=True)
        # Only as many as there are free seats. Forcing more players than the roster holds
        # is how a long minimum list makes a solve infeasible rather than merely tight.
        seats = max(0, ROSTER_SIZE - len(lock_idx))
        for _last, _urgency, name, index in behind[:seats]:
            extra.append((_player_row(index), 1, 1))
        for last_chance, _urgency, name, _index in behind[seats:]:
            if last_chance:
                unmet.add(name)

        constraints = [LinearConstraint(a, lb, ub) for a, lb, ub in rows + extra]
        with profiler.stage("solve", vars=width, players=len(pool)):
            result = milp(c=-(weights @ selector), constraints=constraints,
                          integrality=integrality, bounds=bounds)

        if not result.success or result.x is None:
            # Running out part-way through a set is normal -- overlap limits legitimately
            # exhaust the alternatives. Failing on the FIRST is not: nothing was asked of
            # the solver except the constraints themselves, so they are contradictory.
            if lineup_number == 0 and not lineups:
                raise OptimizerError(
                    "constraints are contradictory — no lineup satisfies them at all"
                    + (f" (stacks {stacks})" if stacks else "")
                    + (f", {len(lock_idx)} lock(s)" if lock_idx else "")
                )
            break

        assignment = _decode(result.x, len(pool))
        if len(assignment) != ROSTER_SIZE:
            break
        chosen_sets.append(set(assignment))
        record = _assemble(pool, assignment, objective)
        lineups.append(record)

        # Count what this lineup used, so the next solve paces against it. Counted from the
        # assembled lineup rather than from the solver vector, so it can never disagree with
        # what the exposure report will later read off the same frame.
        chosen = record["players"]
        chosen_names = set(chosen["Name"])
        for name in appearances:
            if name in chosen_names:
                appearances[name] += 1
        if stacked_counts:
            pass_game = chosen[chosen["Roster"].isin(("QB", "WR", "TE", "FLEX"))]
            by_team = pass_game["Team"].map(canon_team).value_counts()
            for team in stacked_counts:
                if int(by_team.get(team, 0)) >= stack_at:
                    stacked_counts[team] += 1

    missing["exposure_unmet"] = sorted(unmet)
    missing["team_exposure_unmet"] = sorted(unmet_teams)
    missing["appearances"] = dict(appearances)
    missing["team_stacks"] = dict(stacked_counts)
    return lineups, pool, missing


def _decode(solution, n_players):
    """Flattened solution -> {player index: roster slot}."""
    s = len(SLOTS)
    assignment = {}
    for k, value in enumerate(solution):
        if value > 0.5:
            assignment[k // s] = SLOTS[k % s]
    return assignment


def lineup_conflicts(frame, max_opposing=0):
    """[(dst, opposing team, players)] for defenses this lineup rostered against itself.

    Read back off the finished lineup rather than off the solver, so it reports what is
    actually on the roster whatever the constraint was set to.
    """
    skill = frame[frame["Roster"] != "DST"]["Team"].map(canon_team).value_counts()
    found = []
    for _, row in frame[frame["Roster"] == "DST"].iterrows():
        count = int(skill.get(canon_team(row.get("Opp")), 0))
        if count > max_opposing:
            found.append((row["Name"], canon_team(row.get("Opp")), count))
    return sorted(found, key=lambda item: -item[2])


def _assemble(pool, assignment, objective):
    """Build the lineup record. Roster slots come straight from the solver."""
    order = {slot: i for i, slot in enumerate(SLOTS)}
    rows = sorted(assignment, key=lambda i: (order.get(assignment[i], 99), -pool.loc[i, "Proj"]))
    frame = pool.loc[rows].copy()
    frame.insert(0, "Roster", [assignment[i] for i in rows])
    return {
        "players": frame,
        "salary": int(pd.to_numeric(frame["Salary"], errors="coerce").sum()),
        "proj": round(pd.to_numeric(frame["Proj"], errors="coerce").sum(), 2),
        "ceiling": round(pd.to_numeric(frame["Ceiling"], errors="coerce").sum(), 2),
        "floor": round(pd.to_numeric(frame["Floor"], errors="coerce").sum(), 2),
        "objective": objective,
        "conflicts": lineup_conflicts(frame),
        "teams": frame[frame["Roster"] != "DST"]["Team"].value_counts().to_dict(),
    }
