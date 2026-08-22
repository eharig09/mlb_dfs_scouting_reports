"""Candidate lineup generation, separated from final selection.

**Why this is a separate module.** Today `dfs.optimize` produces exactly the lineups you
will enter: twenty solves, twenty entries. That conflates two different jobs. Deciding
*which lineups are worth considering* is a search problem over a huge legal space; deciding
*which twenty to submit* is a portfolio problem that depends on the field, the payout curve
and how the twenty interact. Fusing them means the portfolio question is never asked, and
that re-running it under a different contest structure means re-solving every lineup.

So: generate hundreds or thousands of good legal lineups, write them to disk with their
metrics, and let selection be a cheap, re-runnable read.

**Speed.** Architecture §2.2 measured the stack *bonus* at 7.4x the solve cost -- 120
indicator variables whose only job is to switch on, which wrecks the LP relaxation. A
candidate core states the same intent exactly (`these five hitters, from this team`) as a
hard constraint, which is both cheaper and stronger. Combined with a sparse encoding over
only eligible (player, slot) pairs and a reused base model, generation runs at roughly
0.02-0.05s per lineup instead of 0.40s.

**Two stages.**

1. `stack_cores` enumerates candidate stack cores -- contiguous batting-order runs and
   best-by-ceiling groups -- and scores them.
2. `generate` completes each core with pitchers and the remaining hitters.

**Pruning is deliberately gentle.** A player is only dropped if some other player is at
least as good on *every* axis that matters downstream: projection, ceiling, salary AND
ownership. Low ownership is a payoff term, not a defect, so a 2%-owned bat is never
dominated by chalk at the same price. The failure this guards against is pruning away the
lineup that wins the tournament because it looked bad on a mean.

Nothing here changes `dfs.optimizer`. The existing CLI keeps its exact behaviour.
"""

import json
import os
from datetime import datetime, timezone
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix

from .optimizer import (MAX_HITTERS_PER_TEAM, MIN_GAMES_REPRESENTED, ROSTER, ROSTER_SIZE,
                        build_pool, eligible_positions)
from .profiling import profiler
from .salaries import canon_team
from .scoring import DK_SALARY_CAP

SLOTS = list(ROSTER)

# Stack sizes worth generating cores for. Two is not a stack -- it is two good prices that
# share a jersey -- and DK caps a team at five hitters.
CORE_SIZES = (3, 4, 5)

# An optimality gap the candidate search is happy to accept. These are candidates, not the
# answer; 0.5% of objective is far inside the noise of the projection itself, and it buys a
# measurable speedup (architecture §2.3).
CANDIDATE_GAP = 0.005

# Per-solve time limit, so one pathological core cannot stall a 2,000-lineup run.
SOLVE_TIME_LIMIT = 5.0


class CandidateError(Exception):
    pass


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

def dominated_players(pool, keep_per_position=None, keep_per_team=MAX_HITTERS_PER_TEAM):
    """Indices of players another player beats on every axis at once.

    Domination is checked *within a roster slot* -- a dominated shortstop is only dominated
    by another shortstop, because he is not competing with outfielders for his seat -- and
    requires the rival to be:

        >= projection,  >= ceiling,  <= salary,  <= ownership

    with at least one strict. All four, not a weighted score: dropping a player because his
    blend is lower is how a contrarian play disappears from the pool.

    Ownership participates with the sign *reversed* from intuition on purpose. Being lowly
    owned is a benefit in a tournament, so a cheap high-ceiling chalk play does not dominate
    an equally cheap high-ceiling contrarian one -- they are incomparable, and both stay.
    """
    proj = pd.to_numeric(pool["Proj"], errors="coerce").fillna(0).to_numpy()
    ceiling = pd.to_numeric(pool["Ceiling"], errors="coerce").fillna(0).to_numpy()
    salary = pd.to_numeric(pool["Salary"], errors="coerce").fillna(1e9).to_numpy()
    own = pd.to_numeric(pool.get("Own%"), errors="coerce").fillna(0).to_numpy() \
        if "Own%" in pool.columns else np.zeros(len(pool))

    slot_sets = [eligible_positions(pool.iloc[i]) for i in range(len(pool))]
    dropped = set()
    for slot in SLOTS:
        members = [i for i in range(len(pool)) if slot_sets[i] == {slot}]
        if len(members) < 2:
            continue                      # multi-eligible players are never pruned this way
        # Cheapest first, so the first rival found is the strongest dominator.
        members.sort(key=lambda i: (salary[i], -ceiling[i]))
        for position, i in enumerate(members):
            if i in dropped:
                continue
            for j in members[:position]:
                if j in dropped:
                    continue
                if (proj[j] >= proj[i] and ceiling[j] >= ceiling[i]
                        and salary[j] <= salary[i] and own[j] <= own[i]
                        and (proj[j] > proj[i] or ceiling[j] > ceiling[i]
                             or salary[j] < salary[i] or own[j] < own[i])):
                    dropped.add(i)
                    break

    def rescue(members, floor):
        """Put back the best dropped members until `floor` of them survive."""
        survivors = [i for i in members if i not in dropped]
        if len(survivors) >= floor:
            return
        for i in sorted((i for i in members if i in dropped), key=lambda i: -ceiling[i]):
            dropped.discard(i)
            survivors.append(i)
            if len(survivors) >= floor:
                return

    if keep_per_position:
        # Never prune a position below a floor, whatever domination says: an over-pruned
        # position turns a solvable slate into an infeasible one.
        for slot in SLOTS:
            rescue([i for i in range(len(pool)) if slot in slot_sets[i]], keep_per_position)

    if keep_per_team:
        # The floor that actually matters for a *tournament* pool. Domination is a
        # statement about one seat; a stack is a statement about a team, and pruning a
        # team below stack size deletes every lineup built on it. Measured on 2026-08-01
        # main: unguarded pruning left only 13 of 20 teams able to field three hitters and
        # removed one team outright, so no candidate could ever stack them -- which is the
        # exact "plausible low-owned outcome" a candidate pool exists to keep reachable.
        hitters = (pool["Type"] == "H").to_numpy()
        teams = pool["Team"].map(canon_team).to_numpy()
        for team in sorted(set(teams[hitters])):
            members = [i for i in range(len(pool)) if hitters[i] and teams[i] == team]
            rescue(members, min(keep_per_team, len(members)))

    return sorted(dropped)


# ---------------------------------------------------------------------------
# Stack cores
# ---------------------------------------------------------------------------

def stack_cores(pool, sizes=CORE_SIZES, top_teams=None, focus_teams=None,
                contiguous=True, best_of=True, top_window=6):
    """Candidate stack cores as (team, tuple of player indices, score) records.

    Two families, because real GPP stacks come from both:

    * **contiguous batting-order runs** (1-2-3-4, 5-6-7-8, and 8-9-1-2 wrapping) -- the
      shape that actually shares innings, which is the whole physical basis for correlation.
      Adjacent hitters bat in the same frame far more often than 1 and 8 do.
    * **best-of-window** -- the strongest N of the top `top_window` slots, ignoring
      adjacency. This catches the stack a projection likes that the order does not line up.

    Scored on summed ceiling plus a contiguity credit. Score orders the search; it does not
    filter, so a low-scoring core is still generated if the budget reaches it.
    """
    hitters = pool[pool["Type"] == "H"].copy()
    if hitters.empty:
        return []
    hitters["_team"] = hitters["Team"].map(canon_team)
    if focus_teams:
        wanted = {canon_team(t) for t in focus_teams}
        hitters = hitters[hitters["_team"].isin(wanted)]
        if hitters.empty:
            raise CandidateError(f"no hitters on this slate for focus teams {sorted(wanted)}")

    ceiling = pd.to_numeric(pool["Ceiling"], errors="coerce").fillna(0)

    ranked_teams = (hitters.groupby("_team")["Ceiling"].sum()
                    .sort_values(ascending=False).index.tolist())
    if top_teams:
        ranked_teams = ranked_teams[:top_teams]

    cores, seen = [], set()
    for team in ranked_teams:
        group = hitters[hitters["_team"] == team].sort_values("Slot")
        indices = group.index.tolist()
        slots = pd.to_numeric(group["Slot"], errors="coerce").fillna(9).astype(int).tolist()
        if len(indices) < min(sizes):
            continue

        def add(members, bonus):
            key = (team, tuple(sorted(members)))
            if key in seen or len(members) < min(sizes):
                return
            seen.add(key)
            cores.append({
                "team": team,
                "members": tuple(sorted(members)),
                "size": len(members),
                "score": float(ceiling.loc[list(members)].sum()) + bonus,
                "slots": tuple(sorted(slots[indices.index(m)] for m in members)),
            })

        for size in sizes:
            if len(indices) < size:
                continue
            if contiguous:
                # Wrap-around included: 8-9-1-2 is a legitimate stack, because the order
                # turns over and those four do bat together.
                for start in range(len(indices)):
                    window = [indices[(start + k) % len(indices)] for k in range(size)]
                    if len(set(window)) == size:
                        add(window, bonus=1.5 * size)
            if best_of:
                window = group.head(top_window).index.tolist()
                if len(window) >= size:
                    scored = sorted(window, key=lambda i: -ceiling.loc[i])[:size]
                    add(scored, bonus=0.0)
                    # A couple of near-best alternatives, so the search is not one deep.
                    for swap in combinations(window, size):
                        if len(cores) > 4000:
                            break
                        add(list(swap), bonus=0.0)

    cores.sort(key=lambda c: -c["score"])
    return _interleave_by_team(cores)


def _interleave_by_team(cores):
    """Round-robin the scored cores across teams, keeping each team's own order.

    Pure score order is the wrong search order for a *candidate pool*. The best-scoring
    cores all belong to the best-scoring team, so a budget of 200 solves spends itself
    entirely on that team and the pool comes back with one primary stack. Coverage is the
    point here -- portfolio selection can decide later that a team is not worth entering,
    but it cannot select a stack that was never generated.

    Within a team the score order is untouched, so the search still reaches each team's best
    cores first.
    """
    by_team = {}
    for core in cores:
        by_team.setdefault(core["team"], []).append(core)
    # Teams enter the rotation best-first, so a truncated budget still favours good offenses.
    order = sorted(by_team, key=lambda t: -by_team[t][0]["score"])
    out, depth = [], 0
    while len(out) < len(cores):
        added = False
        for team in order:
            queue = by_team[team]
            if depth < len(queue):
                out.append(queue[depth])
                added = True
        if not added:
            break
        depth += 1
    return out


# ---------------------------------------------------------------------------
# Reusable lineup model
# ---------------------------------------------------------------------------

class LineupModel:
    """A DK Classic model whose base constraints are built once and reused.

    Variables are one binary per **eligible** (player, slot) pair rather than per player per
    slot. On a 198-player pool that is 219 variables instead of 1,386. HiGHS presolves most
    of the difference away, so the measured win is ~1.5x rather than 6x -- but the base
    `LinearConstraint` objects are also built once here instead of per solve, which scipy
    otherwise spends 0.5s on across twenty lineups.
    """

    def __init__(self, pool, max_hitters_per_team=MAX_HITTERS_PER_TEAM,
                 min_games=MIN_GAMES_REPRESENTED, max_ownership=None, min_salary=None):
        self.pool = pool.reset_index(drop=True)
        n = len(self.pool)
        self.n = n
        self.slot_sets = [eligible_positions(self.pool.iloc[i]) for i in range(n)]
        self.pairs = [(i, slot) for i in range(n) for slot in sorted(self.slot_sets[i])]
        if not self.pairs:
            raise CandidateError("no player in the pool is eligible at any roster slot")
        self.width = len(self.pairs)

        # selector[i, k] == 1 when variable k puts player i somewhere.
        rows = np.array([i for i, _ in self.pairs])
        cols = np.arange(self.width)
        self.selector = csc_matrix((np.ones(self.width), (rows, cols)), shape=(n, self.width))

        self.salary = pd.to_numeric(self.pool["Salary"], errors="coerce").fillna(0).to_numpy()
        self.teams = self.pool["Team"].map(canon_team).to_numpy()
        self.is_hitter = (self.pool["Type"] == "H").to_numpy()
        self.games = self.pool.get("Game", pd.Series("", index=self.pool.index)).to_numpy()

        base = []
        # Each player fills at most one slot.
        base.append(LinearConstraint(self.selector, 0, 1))
        # Each roster slot filled exactly.
        for slot in SLOTS:
            row = np.array([1.0 if s == slot else 0.0 for _, s in self.pairs])
            base.append(LinearConstraint(row.reshape(1, -1), ROSTER[slot], ROSTER[slot]))
        base.append(LinearConstraint((self.salary @ self.selector).reshape(1, -1),
                                     float(min_salary or 0), DK_SALARY_CAP))
        for team in sorted(set(self.teams)):
            mask = ((self.teams == team) & self.is_hitter).astype(float)
            if mask.sum():
                base.append(LinearConstraint((mask @ self.selector).reshape(1, -1),
                                             0, max_hitters_per_team))
        if min_games >= 2:
            for game in sorted({g for g in self.games if g}):
                mask = (self.games == game).astype(float)
                base.append(LinearConstraint((mask @ self.selector).reshape(1, -1),
                                             0, ROSTER_SIZE - 1))
        if max_ownership is not None and "Own%" in self.pool.columns:
            own = pd.to_numeric(self.pool["Own%"], errors="coerce").fillna(0).to_numpy()
            base.append(LinearConstraint((own @ self.selector).reshape(1, -1),
                                         0, float(max_ownership)))
        self.base = base
        self.bounds = Bounds(np.zeros(self.width), np.ones(self.width))
        self.integrality = np.ones(self.width)

    def player_row(self, indices):
        """A 1 x width row selecting those players, ready for a LinearConstraint."""
        mask = np.zeros(self.n)
        mask[list(indices)] = 1.0
        return (mask @ self.selector).reshape(1, -1)

    def solve(self, weights, extra=(), gap=CANDIDATE_GAP, time_limit=SOLVE_TIME_LIMIT):
        """Maximize `weights` (per player) subject to base + extra. Returns indices or None."""
        objective = -(np.asarray(weights, dtype=float) @ self.selector)
        with profiler.stage("solve", vars=self.width):
            result = milp(c=np.asarray(objective).ravel(),
                          constraints=list(self.base) + list(extra),
                          integrality=self.integrality, bounds=self.bounds,
                          options={"mip_rel_gap": gap, "time_limit": time_limit})
        if not result.success or result.x is None:
            return None
        chosen = np.flatnonzero(np.asarray(result.x) > 0.5)
        picked = {}
        for k in chosen:
            i, slot = self.pairs[k]
            picked[i] = slot
        return picked if len(picked) == ROSTER_SIZE else None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _lineup_record(pool, assignment, number):
    """Everything selection will need, so it never has to touch the solver again."""
    indices = sorted(assignment, key=lambda i: (SLOTS.index(assignment[i]),
                                                -float(pool.loc[i, "Proj"])))
    frame = pool.loc[indices]
    hitters = frame[frame["Type"] == "H"]
    counts = hitters["Team"].map(canon_team).value_counts()
    stacks = {team: int(count) for team, count in counts.items() if count >= 2}
    ordered = sorted(stacks.items(), key=lambda kv: (-kv[1], kv[0]))
    pitchers = sorted(frame[frame["Type"] == "P"]["Name"].tolist())
    own = pd.to_numeric(frame.get("Own%"), errors="coerce")

    return {
        "lineup": number,
        "players": [int(i) for i in indices],
        "roster": [assignment[i] for i in indices],
        "names": frame["Name"].tolist(),
        "mlbam": [None if pd.isna(v) else int(v) for v in frame["MLBAM"]],
        "salary": int(pd.to_numeric(frame["Salary"], errors="coerce").sum()),
        "proj": round(float(pd.to_numeric(frame["Proj"], errors="coerce").sum()), 2),
        "ceiling": round(float(pd.to_numeric(frame["Ceiling"], errors="coerce").sum()), 2),
        "floor": round(float(pd.to_numeric(frame["Floor"], errors="coerce").sum()), 2),
        "own_sum": round(float(own.sum()), 1) if own.notna().any() else None,
        "own_min": round(float(own.min()), 2) if own.notna().any() else None,
        "sub5_own": int((own < 5).sum()) if own.notna().any() else None,
        # The construction fingerprint. Portfolio selection groups on these, and a review
        # asks whether a shape was ever generated at all.
        "stack_shape": "-".join(str(c) for _, c in ordered) or "none",
        "primary_stack": ordered[0][0] if ordered else None,
        "primary_size": ordered[0][1] if ordered else 0,
        "secondary_stack": ordered[1][0] if len(ordered) > 1 else None,
        "secondary_size": ordered[1][1] if len(ordered) > 1 else 0,
        "pitcher_pair": "|".join(pitchers),
        "games": sorted({str(g) for g in frame.get("Game", pd.Series(dtype=str)) if g}),
        "n_games": len({str(g) for g in frame.get("Game", pd.Series(dtype=str)) if g}),
    }


class CandidatePool:
    """Generated lineups plus the player pool they index into.

    `players` are positional indices into `pool`, which is why the pool travels with the
    lineups: a candidate file that stored only names could not be re-scored against a
    simulation matrix without a fragile re-join.
    """

    def __init__(self, lineups, pool, meta=None):
        self.lineups = lineups.reset_index(drop=True) if isinstance(lineups, pd.DataFrame) \
            else pd.DataFrame(lineups)
        self.pool = pool.reset_index(drop=True)
        self.meta = meta or {}

    def __len__(self):
        return len(self.lineups)

    def __repr__(self):
        return (f"<CandidatePool {len(self.lineups)} lineups over {len(self.pool)} players, "
                f"{self.lineups['stack_shape'].nunique() if len(self.lineups) else 0} shapes>")

    def matrix(self):
        """Sparse (n_players x n_candidates) 0/1 matrix, for `S @ L` lineup scoring."""
        rows, cols = [], []
        for column, players in enumerate(self.lineups["players"]):
            for i in players:
                rows.append(int(i))
                cols.append(column)
        return csc_matrix((np.ones(len(rows)), (rows, cols)),
                          shape=(len(self.pool), len(self.lineups)))

    def score(self, simulations):
        """(n_sims x n_candidates) lineup scores from an (n_sims x n_players) matrix."""
        return np.asarray(simulations) @ self.matrix().toarray()

    def summary(self):
        if self.lineups.empty:
            return "no candidates"
        frame = self.lineups
        return (
            f"{len(frame)} candidates | "
            f"proj {frame['proj'].min():.1f}-{frame['proj'].max():.1f} "
            f"(mean {frame['proj'].mean():.1f}) | "
            f"ceiling mean {frame['ceiling'].mean():.1f} | "
            f"{frame['stack_shape'].nunique()} stack shapes, "
            f"{frame['primary_stack'].nunique()} primary teams, "
            f"{frame['pitcher_pair'].nunique()} pitcher pairs"
        )

    # -- persistence ------------------------------------------------------
    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        self.lineups.to_parquet(os.path.join(directory, "lineups.parquet"), index=False)
        self.pool.to_parquet(os.path.join(directory, "pool.parquet"), index=False)
        with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as handle:
            json.dump(self.meta, handle, indent=2, default=str)
        return directory

    @classmethod
    def load(cls, directory):
        lineups = pd.read_parquet(os.path.join(directory, "lineups.parquet"))
        pool = pd.read_parquet(os.path.join(directory, "pool.parquet"))
        for column in ("players", "roster", "names", "mlbam", "games"):
            if column in lineups.columns:
                lineups[column] = lineups[column].apply(
                    lambda v: list(v) if v is not None else [])
        meta_path = os.path.join(directory, "meta.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as handle:
                meta = json.load(handle)
        return cls(lineups, pool, meta)


def generate(players, n_candidates=500, objective="Ceiling", seed=None, randomness=0.25,
             locks=None, excludes=None, focus_teams=None, sizes=CORE_SIZES,
             max_hitters_per_team=MAX_HITTERS_PER_TEAM, min_games=MIN_GAMES_REPRESENTED,
             max_ownership=None, min_salary=None, prune=False, top_teams=None,
             cores_per_pass=None, max_attempts_factor=4, quiet=True):
    """Generate up to `n_candidates` distinct legal lineups. Returns a CandidatePool.

    Each pass walks the scored stack cores, forcing one core in as a hard constraint and
    letting the solver fill the other five or six seats. Objective jitter differs per
    attempt so the same core yields different completions across passes; duplicates are
    dropped by player set, so the count is of genuinely distinct lineups.

    `randomness` here is doing a different job from `dfs.optimize`'s. There it explores
    near-optimal lineups to build a *diverse entry set*; here it is a sampler over the
    high-value region, and the diversity that matters is enforced structurally by the cores.
    """
    pool, lock_idx, missing = build_pool(players, locks, excludes)
    if len(pool) < ROSTER_SIZE:
        raise CandidateError(f"pool has only {len(pool)} playable players; need {ROSTER_SIZE}")

    pruned = []
    if prune:
        pruned = dominated_players(pool, keep_per_position=8)
        # A locked player is a decision, not a candidate for pruning.
        pruned = [i for i in pruned if i not in set(lock_idx)]
        if pruned:
            pool = pool.drop(index=pruned).reset_index(drop=True)
            lock_idx, _ = _rematch(pool, locks)

    if objective not in pool.columns:
        raise CandidateError(f"slate has no '{objective}' column")
    base = pd.to_numeric(pool[objective], errors="coerce").fillna(0).to_numpy()

    with profiler.stage("candidates", pool=len(pool), target=n_candidates):
        model = LineupModel(pool, max_hitters_per_team=max_hitters_per_team,
                            min_games=min_games, max_ownership=max_ownership,
                            min_salary=min_salary)
        cores = stack_cores(pool, sizes=sizes, top_teams=top_teams, focus_teams=focus_teams)
        if not cores:
            raise CandidateError("no team on this slate has enough hitters to stack")

        lock_rows = [LinearConstraint(model.player_row([i]), 1, 1) for i in lock_idx]

        rng = np.random.default_rng(seed)
        found, seen = [], set()
        attempts = 0
        limit = n_candidates * max_attempts_factor
        per_pass = cores_per_pass or len(cores)

        while len(found) < n_candidates and attempts < limit:
            progressed = False
            for core in cores[:per_pass]:
                if len(found) >= n_candidates or attempts >= limit:
                    break
                attempts += 1
                weights = base * (1 + rng.normal(0, randomness, size=len(base))) \
                    if randomness > 0 else base
                extra = lock_rows + [
                    LinearConstraint(model.player_row(core["members"]),
                                     len(core["members"]), len(core["members"]))
                ]
                assignment = model.solve(weights, extra)
                if assignment is None:
                    continue
                key = frozenset(assignment)
                if key in seen:
                    continue
                seen.add(key)
                found.append(_lineup_record(pool, assignment, len(found) + 1))
                progressed = True
            if not progressed:
                # A full sweep of every core produced nothing new. More attempts would only
                # re-derive the same lineups, so stop and report the honest count.
                break

    if not found:
        raise CandidateError("no legal lineup could be built from any stack core")

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "requested": n_candidates,
        "generated": len(found),
        "attempts": attempts,
        "objective": objective,
        "randomness": randomness,
        "seed": seed,
        "cores": len(cores),
        "pool_players": int(len(pool)),
        "pruned_players": len(pruned),
        "locks": list(locks or []),
        "excludes": list(excludes or []),
        "focus_teams": list(focus_teams or []),
        "max_ownership": max_ownership,
        "missing": missing,
    }
    if not quiet:
        print(f"  {len(found)} candidates from {attempts} solves over {len(cores)} cores")
    return CandidatePool(pd.DataFrame(found), pool, meta)


def _rematch(pool, names):
    from .optimizer import _match
    return _match(pool, names)


# ---------------------------------------------------------------------------
# Bridge: use dfs.optimize's output as the candidate pool
# ---------------------------------------------------------------------------

def from_optimizer(players, path=None, date=None, slate=None, root="dfs_boards"):
    """Read a `lineups_<slate>.csv` written by `dfs.optimize` into a CandidatePool.

    **Why this exists.** `dfs.optimize` and `dfs.candidates` both generate lineups, but they
    are good at different halves of the job. The optimizer carries every constraint the
    workflow depends on -- the editable pool file, exposure minimums, ranged boosts,
    explicit stacks, pins, and the started-game filter -- while `generate` here carries
    stack-core coverage and speed. Rather than reimplementing seven features, this lets the
    optimizer generate and the portfolio select:

        python -m dfs.optimize  --date D --n 100 --pool      # constrained generation
        python -m dfs.portfolio --date D --from-optimizer    # correlation-aware selection

    The returned pool keeps each lineup's original optimizer number in `source_lineup`, so
    the selected set can be written back out in the optimizer's own format and uploaded
    without any of the ids being re-derived.
    """
    from .naming import find_slates, latest
    from .optimizer import build_pool
    from .salaries import canon_team, normalize_name

    if path is None:
        if date is None:
            raise CandidateError("from_optimizer needs a path or a date")
        path = latest("lineups", date, slate, root)
        if not path:
            # Nothing under the label asked for. The lineups usually *do* exist under a
            # different one -- the optimizer files them by the DK export's slate label, and
            # a caller that omitted --slate looked for `lineups_unknown.csv`. Telling them
            # to re-run a six-minute solve they have already done is the wrong answer.
            available = find_slates(date, "lineups", root)
            if len(available) == 1:
                path = latest("lineups", date, available[0], root)
            elif len(available) > 1:
                raise CandidateError(
                    f"{date} has optimizer lineups for {len(available)} slates "
                    f"({', '.join(available)}); pass --slate to say which. Picking one "
                    f"silently would select entries for a contest you are not in.")
            if not path:
                raise CandidateError(
                    f"no optimizer lineups found for {date}"
                    + (f" slate {slate}" if slate else "")
                    + f" under {root}/. Run: python -m dfs.optimize --date {date} --n 100")

    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [str(c).strip() for c in frame.columns]
    for column in ("Lineup", "Roster", "Name"):
        if column not in frame.columns:
            raise CandidateError(
                f"{path} is not an optimizer lineup file (no '{column}' column)")

    pool, _lock_idx, _missing = build_pool(players)
    if pool.empty:
        raise CandidateError("the slate has no playable players to match lineups against")
    # Name plus team, so two players sharing a name never collide.
    index = {}
    for position, (name, team) in enumerate(zip(pool["Name"], pool["Team"])):
        index.setdefault((normalize_name(name), canon_team(team)), position)

    records, unmatched = [], set()
    for number, group in frame.groupby("Lineup", sort=True):
        rows = []
        for name, team in zip(group["Name"], group["Team"]):
            key = (normalize_name(name), canon_team(team))
            if key in index:
                rows.append(index[key])
            else:
                unmatched.add(str(name))
        if len(rows) != ROSTER_SIZE:
            continue
        assignment = dict(zip(rows, group["Roster"]))
        record = _lineup_record(pool, assignment, len(records) + 1)
        record["source_lineup"] = int(number)
        records.append(record)

    if not records:
        raise CandidateError(
            f"none of the lineups in {path} could be matched to the current slate."
            + (f" Unmatched players include: {', '.join(sorted(unmatched)[:5])}."
               if unmatched else "")
            + " The slate has probably been rebuilt since; re-run dfs.optimize.")

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "dfs.optimize", "source_path": path,
        "generated": len(records), "requested": len(records),
        "attempts": len(records), "objective": "from optimizer",
        "randomness": None, "seed": None, "cores": 0,
        "pool_players": int(len(pool)), "pruned_players": 0,
        "locks": [], "excludes": [], "focus_teams": [], "max_ownership": None,
        "missing": {"unmatched_players": sorted(unmatched)},
    }
    return CandidatePool(pd.DataFrame(records), pool, meta)


def to_optimizer_format(candidate_pool, selection, source_path):
    """The selected candidates, written back in the optimizer's own long CSV format.

    `dfs.upload` reads one row per player with `Lineup` and `Roster` columns and takes DK
    ids from the upload template, so round-tripping through the optimizer's original file
    keeps every column it expects and renumbers the survivors 1..N in selection order.
    """
    lineups = candidate_pool.lineups
    if "source_lineup" not in lineups.columns:
        # Checked before touching the file: a generated pool has no source to round-trip
        # through, and failing on a missing path would name the wrong problem.
        raise CandidateError(
            "this candidate pool did not come from dfs.optimize, so it cannot be written "
            "back in the optimizer's format")
    frame = pd.read_csv(source_path, encoding="utf-8-sig")
    frame.columns = [str(c).strip() for c in frame.columns]

    out = []
    for new_number, index in enumerate(selection, start=1):
        original = int(lineups.iloc[index]["source_lineup"])
        block = frame[frame["Lineup"] == original].copy()
        block["Lineup"] = new_number
        out.append(block)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def default_dir(date, slate, root=os.path.join("dfs_boards")):
    return os.path.join(root, str(date), f"candidates_{slate or 'main'}")


def main():
    import argparse
    import time

    from .naming import slate_slug
    from .slate import build_slate

    parser = argparse.ArgumentParser(
        description="Generate a pool of candidate lineups, separate from final selection.")
    parser.add_argument("--date", default=datetime.today().strftime("%Y-%m-%d"))
    parser.add_argument("--slate")
    parser.add_argument("--salaries")
    parser.add_argument("--snapshot", action="store_true",
                        help="Build from the date's slate snapshot instead of the live cache.")
    parser.add_argument("--stage", help="Which snapshot stage, with --snapshot.")
    parser.add_argument("--n", type=int, default=500, help="Candidates to generate.")
    parser.add_argument("--objective", default="Ceiling",
                        choices=["Ceiling", "Proj", "Floor", "Lev Score"])
    parser.add_argument("--randomness", type=float, default=0.25)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lock")
    parser.add_argument("--exclude")
    parser.add_argument("--focus-teams")
    parser.add_argument("--max-ownership", type=float)
    parser.add_argument("--min-salary", type=int,
                        help="Floor on lineup salary; DK leaves no prize for unspent cap.")
    parser.add_argument("--sizes", default="3,4,5", help="Stack core sizes to enumerate.")
    parser.add_argument("--prune", action="store_true",
                        help="Drop players another beats on projection, ceiling, salary AND "
                             "ownership at once. Off by default: it is a deterministic test "
                             "on a stochastic question, and it shrinks the reachable set.")
    parser.add_argument("--out", help="Directory to write the candidate pool to.")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    split = lambda v: [p.strip() for p in str(v or "").split(",") if p.strip()]

    with profiler.session("candidates", date=args.date, slate=args.slate,
                          enabled=args.profile, n=args.n):
        if args.snapshot:
            from .snapshot import load_snapshot
            snap = load_snapshot(args.date, args.slate, stage=args.stage)
            players, meta = snap.players, snap.meta()
            print(f"from {snap}")
        else:
            players, _, meta = build_slate(args.date, salary_path=args.salaries,
                                           slate=args.slate)
        if players is None or players.empty:
            print(f"No slate data for {args.date}.")
            return
        if not meta.get("has_salary"):
            print("[!] no salaries matched; candidate generation needs prices.")
            return

        started = time.perf_counter()
        pool = generate(
            players, n_candidates=args.n, objective=args.objective, seed=args.seed,
            randomness=args.randomness, locks=split(args.lock), excludes=split(args.exclude),
            focus_teams=split(args.focus_teams),
            sizes=tuple(int(s) for s in split(args.sizes)),
            max_ownership=args.max_ownership, min_salary=args.min_salary,
            prune=args.prune,
        )
        elapsed = time.perf_counter() - started

    label = slate_slug(args.slate or meta.get("slate_label") or "main")
    directory = args.out or default_dir(args.date, label)
    pool.meta["elapsed_s"] = round(elapsed, 2)
    pool.meta["date"] = str(args.date)
    pool.meta["slate"] = label
    pool.save(directory)

    print(f"\n{pool.summary()}")
    print(f"  {elapsed:.1f}s  ({elapsed / max(1, len(pool)) * 1000:.0f} ms/lineup, "
          f"{pool.meta['attempts']} solves)")
    if pool.meta["pruned_players"]:
        print(f"  {pool.meta['pruned_players']} dominated player(s) pruned before solving")
    print(f"  -> {directory}/")

    frame = pool.lineups
    top = frame.nlargest(min(5, len(frame)), "ceiling")
    print("\n  highest-ceiling candidates:")
    for _, row in top.iterrows():
        print(f"    {row['ceiling']:6.1f} ceil  {row['proj']:6.1f} proj  ${row['salary']:,}  "
              f"{row['stack_shape']:<8} {row['primary_stack'] or '-'}  "
              f"own {row['own_sum'] if row['own_sum'] is not None else '-'}")

    if args.profile:
        from .profiling import format_report
        print("\n=== pipeline profile ===")
        print(format_report(profiler.last_report))


if __name__ == "__main__":
    main()
