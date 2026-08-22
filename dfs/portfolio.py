"""Choose the whole set of lineups you enter, jointly.

**Why not just take the top N.** Ranking candidates by expected profit and taking the best
twenty gives you twenty lineups that are good for the *same reason* -- same stack, same
pitcher, same read on the slate. They win together and they lose together, and in a
tournament that is close to entering one lineup twenty times. What a multi-entry portfolio
wants is not twenty good lineups but a set whose *best member* is likely to be very good,
which is a different objective and has a different answer.

**The objective.** For a portfolio P over simulated nights:

    E[max payout]        the tournament is won by your best lineup, not your average one
    P(any lineup wins)   computed directly off the simulation matrix
    P(any in top 1%)
    coverage             distinct primary stacks, distinct games, distinct pitcher pairs
  - correlation penalty  mean pairwise correlation of simulated lineup *scores*
  - exposure penalties   player, stack and pitcher-pair concentration

**Correlation is measured, not counted.** `dfs.optimize`'s `--max-overlap` counts shared
players, which is a poor proxy: two lineups sharing three players can be near-identical in
outcome (same game stack, same pitcher) while two sharing six can diverge sharply. Here the
pairwise correlation comes from the simulated score vectors, so it measures what actually
matters -- do these two lineups fail in the same worlds.

**Why greedy.** The objective is submodular-ish in the selection (each added lineup helps
less than the last) and is emphatically not linear, so a MILP is the wrong tool. Greedy
forward selection carries the standard 1 - 1/e guarantee on the submodular part, runs in
milliseconds for the sizes involved, and is followed by a local swap pass that recovers most
of the rest.
"""

import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .profiling import profiler

BENCHMARK_DIR = os.path.join("docs", "benchmarks")
OUTPUT_ROOT = "dfs_boards"


class PortfolioError(Exception):
    pass


@dataclass
class PortfolioWeights:
    """What the portfolio objective rewards and punishes.

    Defaults lean tournament: the payout term dominates, coverage breaks ties, and the
    correlation penalty is what stops the set collapsing onto one construction. For a cash
    game set `payout` high, `best_of` to zero and `correlation` to zero -- there you want
    twenty copies of the highest-floor lineup, and the machinery should let you say so.
    """

    payout: float = 1.0            # summed expected payout across the set
    best_of: float = 1.5           # E[max payout] -- the tournament term
    win_any: float = 0.0           # weight on P(at least one lineup wins outright)
    top1_any: float = 0.0          # weight on P(at least one top-1% finish)
    correlation: float = 0.60      # penalty on mean pairwise simulated-score correlation
    stack_coverage: float = 0.25   # reward for distinct primary stacks
    game_coverage: float = 0.10    # reward for distinct games represented
    pitcher_coverage: float = 0.15  # reward for distinct pitcher pairs
    duplication: float = 0.20      # penalty on expected duplicates

    def to_dict(self):
        return asdict(self)


PRESETS = {
    # One entry: there is no portfolio, so everything except the lineup's own EV is noise.
    "single-entry": PortfolioWeights(payout=1.0, best_of=0.0, correlation=0.0,
                                     stack_coverage=0.0, game_coverage=0.0,
                                     pitcher_coverage=0.0, duplication=0.35),
    "three-max": PortfolioWeights(payout=1.0, best_of=1.0, correlation=0.45,
                                  stack_coverage=0.30, pitcher_coverage=0.20),
    "twenty-max": PortfolioWeights(),
    "mass-multi": PortfolioWeights(payout=0.7, best_of=2.0, correlation=0.75,
                                   stack_coverage=0.40, game_coverage=0.20,
                                   pitcher_coverage=0.25, duplication=0.30),
    "cash": PortfolioWeights(payout=1.0, best_of=0.0, correlation=0.0,
                             stack_coverage=0.0, game_coverage=0.0,
                             pitcher_coverage=0.0, duplication=0.0),
}


# ---------------------------------------------------------------------------
# Portfolio metrics
# ---------------------------------------------------------------------------

def lineup_scores(candidate_pool, simulations):
    """(n_sims x n_candidates) simulated scores. float32 to keep 50k x 5k affordable."""
    matrix = np.asarray(candidate_pool.matrix().todense(), dtype=np.float32)
    return np.asarray(simulations, dtype=np.float32) @ matrix


def pairwise_correlation(scores, selection):
    """Mean pairwise correlation of the selected lineups' simulated scores."""
    if len(selection) < 2:
        return 0.0
    block = scores[:, list(selection)]
    standard = block.std(axis=0)
    if np.all(standard <= 1e-9):
        return 1.0
    matrix = np.corrcoef(block, rowvar=False)
    n = len(selection)
    off_diagonal = (matrix.sum() - np.trace(matrix)) / (n * (n - 1))
    return float(np.nan_to_num(off_diagonal))


def effective_lineups(scores, selection):
    """How many independent lineups the set is really worth.

    With mean pairwise correlation r, a set of n behaves like n / (1 + (n-1) r) independent
    bets. Twenty lineups at r = 0.9 are worth about two. It is the single most honest
    summary of whether a portfolio is diversified, and it is the number `--max-overlap` was
    always groping towards.
    """
    n = len(selection)
    if n < 2:
        return float(n)
    r = max(pairwise_correlation(scores, selection), 0.0)
    return float(n / (1.0 + (n - 1) * r))


def _standardise(scores):
    """Z-scored columns, so a correlation becomes a dot product.

    corr(i, j) = (Z[:, i] . Z[:, j]) / n_sims. That identity is what makes the swap pass
    affordable: for a selection with column sum S = sum of its Z columns,

        sum over all pairs (i, j) in the selection of corr(i, j) = (S . S) / n_sims

    so the mean pairwise correlation is ((S.S)/n_sims - k) / (k(k-1)) -- O(n_sims) to
    evaluate and O(n_sims) to update after a swap, instead of rebuilding a k x k
    correlation matrix from 20,000 samples every time.

    Zero-variance columns become all-zero, which gives them correlation 0 with everything.
    That matches what `pairwise_correlation` does after its nan_to_num.
    """
    block = np.asarray(scores, dtype=np.float64)
    centred = block - block.mean(axis=0, keepdims=True)
    deviation = centred.std(axis=0, keepdims=True)
    safe = np.where(deviation > 1e-12, deviation, 1.0)
    standard = centred / safe
    standard[:, deviation.ravel() <= 1e-12] = 0.0
    return standard


class _SwapState:
    """The portfolio objective, maintained incrementally for the local swap pass.

    `_objective` recomputes everything from scratch: a k x k correlation matrix over every
    simulation, a full max-across-columns, and three pandas `nunique` calls. At n=150 with
    20,000 simulations that is 125 ms, and the swap pass wants 75,000 of them -- 2.6 hours.

    Everything here is a running quantity updated in O(n_sims) per trial swap:

    * correlation  -- the column sum S of standardised scores (see `_standardise`)
    * E[max]       -- the top *two* values per simulation, so the max excluding any one
                      position is a lookup rather than a recomputation
    * coverage     -- integer counters over stack / pitcher-pair / game codes
    * payout, duplication -- plain sums

    It computes the same number `_objective` does; `tests/test_portfolio.py` pins that.
    """

    def __init__(self, scores, candidates, selection, weights, payoffs, dupes, unit):
        self.scores = np.asarray(scores)
        self.weights = weights
        self.payoffs = payoffs
        self.dupes = dupes
        self.unit = unit
        self.n_sims = self.scores.shape[0]
        self.selection = list(selection)

        # Stored transposed and contiguous. `scores` is (n_sims, n_candidates) in C order,
        # so a column is a strided view -- correct, but arithmetic over it walks 20,000
        # cache lines. Measured at 20,000 simulations: the running-sum update costs 403 us
        # strided against 32 us contiguous, and the E[max] update 118 us against 36 us.
        # float32 for the standardised copy keeps the extra memory to n_sims x n_cand x 4.
        self.standard_t = np.ascontiguousarray(
            _standardise(self.scores).T, dtype=np.float32)
        self.scores_t = np.ascontiguousarray(self.scores.T)
        self.column_sum = self.standard_t[self.selection].sum(axis=0, dtype=np.float64)

        # Integer codes, so coverage is a counter rather than a pandas groupby.
        self.stack_code = pd.factorize(candidates["primary_stack"].astype(str))[0]
        self.pair_code = pd.factorize(candidates["pitcher_pair"].astype(str))[0]
        game_ids, self.game_lists = {}, []
        for games in candidates["games"]:
            codes = []
            for game in (games if games is not None else []):
                codes.append(game_ids.setdefault(str(game), len(game_ids)))
            self.game_lists.append(codes)

        self.stack_count = Counter(self.stack_code[i] for i in self.selection)
        self.pair_count = Counter(self.pair_code[i] for i in self.selection)
        self.game_count = Counter(g for i in self.selection for g in self.game_lists[i])
        self.payout_sum = float(self.payoffs[self.selection].sum())
        self.dupe_sum = float(self.dupes[self.selection].sum())
        self.coverage_base = self._coverage(self.stack_count, self.pair_count,
                                            self.game_count)
        self._refresh_top2()

    def _refresh_top2(self):
        """Top two simulated scores per night, and which selected position holds the best."""
        block = self.scores[:, self.selection]
        if block.shape[1] == 1:
            self.best1 = block[:, 0].astype(np.float64)
            self.arg1 = np.zeros(self.n_sims, dtype=np.int64)
            self.best2 = np.full(self.n_sims, -np.inf)
            return
        top = np.argpartition(-block, 1, axis=1)[:, :2]
        rows = np.arange(self.n_sims)
        first, second = top[:, 0], top[:, 1]
        swap = block[rows, second] > block[rows, first]
        first, second = np.where(swap, second, first), np.where(swap, first, second)
        self.arg1 = first
        self.best1 = block[rows, first].astype(np.float64)
        self.best2 = block[rows, second].astype(np.float64)

    @staticmethod
    def _distinct_delta(counter, outgoing_codes, incoming_codes):
        """Change in the number of distinct codes, without copying the counter.

        Copying three Counters per trial swap was most of the remaining cost -- 75,000
        swaps means 225,000 dict copies. The count only moves when a code drops to zero or
        rises from it, which is decidable by arithmetic on the codes involved.
        """
        if tuple(outgoing_codes) == tuple(incoming_codes):
            return 0
        touched = {}
        for code in outgoing_codes:
            touched[code] = touched.get(code, 0) - 1
        for code in incoming_codes:
            touched[code] = touched.get(code, 0) + 1
        delta = 0
        for code, change in touched.items():
            before = counter.get(code, 0)
            after = before + change
            delta += (after > 0) - (before > 0)
        return delta

    def _coverage(self, stack_count, pair_count, game_count):
        return (self.weights.stack_coverage * sum(1 for v in stack_count.values() if v)
                + self.weights.pitcher_coverage * sum(1 for v in pair_count.values() if v)
                + self.weights.game_coverage * sum(1 for v in game_count.values() if v))

    def _correlation(self, column_sum, k):
        if not self.weights.correlation or k < 2:
            return 0.0
        total = float(column_sum @ column_sum) / self.n_sims
        return (total - k) / (k * (k - 1))

    def value(self, position=None, replacement=None):
        """Objective with the selection as it stands, or with one position swapped out."""
        k = len(self.selection)
        if position is None:
            column_sum = self.column_sum
            payout_sum, dupe_sum = self.payout_sum, self.dupe_sum
            best = self.best1
            coverage = self.coverage_base
        else:
            outgoing = self.selection[position]
            column_sum = self.column_sum - self.standard_t[outgoing] \
                + self.standard_t[replacement]
            payout_sum = self.payout_sum - self.payoffs[outgoing] + self.payoffs[replacement]
            dupe_sum = self.dupe_sum - self.dupes[outgoing] + self.dupes[replacement]
            # Max excluding the position being replaced: the runner-up when that position
            # held the best, otherwise the best is untouched.
            without = np.where(self.arg1 == position, self.best2, self.best1)
            best = np.maximum(without, self.scores_t[replacement])
            # Coverage as a delta rather than three copied Counters. 75,000 trial swaps
            # meant 225,000 dict copies, which was most of what remained.
            coverage = (
                self.coverage_base
                + self.weights.stack_coverage * self._distinct_delta(
                    self.stack_count, (self.stack_code[outgoing],),
                    (self.stack_code[replacement],))
                + self.weights.pitcher_coverage * self._distinct_delta(
                    self.pair_count, (self.pair_code[outgoing],),
                    (self.pair_code[replacement],))
                + self.weights.game_coverage * self._distinct_delta(
                    self.game_count, self.game_lists[outgoing],
                    self.game_lists[replacement]))

        value = self.weights.payout * payout_sum
        value += self.weights.best_of * float(best.mean()) * self.unit * 0.15
        value -= self.weights.duplication * dupe_sum * self.unit * 0.05
        value -= (self.weights.correlation * self._correlation(column_sum, k)
                  * self.unit * k)
        value += coverage * self.unit
        return value

    def apply(self, position, replacement):
        outgoing = self.selection[position]
        self.column_sum = self.column_sum - self.standard_t[outgoing] \
            + self.standard_t[replacement]
        self.payout_sum += self.payoffs[replacement] - self.payoffs[outgoing]
        self.dupe_sum += self.dupes[replacement] - self.dupes[outgoing]
        self.stack_count[self.stack_code[outgoing]] -= 1
        self.stack_count[self.stack_code[replacement]] += 1
        self.pair_count[self.pair_code[outgoing]] -= 1
        self.pair_count[self.pair_code[replacement]] += 1
        for game in self.game_lists[outgoing]:
            self.game_count[game] -= 1
        for game in self.game_lists[replacement]:
            self.game_count[game] += 1
        self.selection[position] = replacement
        self.coverage_base = self._coverage(self.stack_count, self.pair_count,
                                            self.game_count)
        self._refresh_top2()


def portfolio_metrics(scores, selection, evaluated, payout=None, field_rank=None):
    """Everything worth reporting about a chosen set."""
    selection = list(selection)
    rows = evaluated.iloc[selection]
    block = scores[:, selection]

    metrics = {
        "n": len(selection),
        "mean_score": round(float(block.mean()), 2),
        "best_of_set_mean": round(float(block.max(axis=1).mean()), 2),
        "best_of_set_p99": round(float(np.percentile(block.max(axis=1), 99)), 1),
        "mean_pairwise_correlation": round(pairwise_correlation(scores, selection), 4),
        "effective_lineups": round(effective_lineups(scores, selection), 2),
        "distinct_primary_stacks": int(rows["primary_stack"].nunique()),
        "distinct_stack_shapes": int(rows["stack_shape"].nunique()),
        "distinct_pitcher_pairs": int(rows["pitcher_pair"].nunique()),
        "distinct_players": len({p for players in rows["players"] for p in players}),
    }
    for column, key in (("exp_payout", "total_expected_payout"),
                        ("exp_profit", "total_expected_profit"),
                        ("exp_dupes", "mean_expected_duplicates"),
                        ("own_sum", "mean_total_ownership")):
        if column in rows.columns:
            value = rows[column].sum() if key.startswith("total") else rows[column].mean()
            metrics[key] = round(float(value), 3)
    if "p_win" in rows.columns:
        # Independence is the wrong assumption for "any of mine wins" -- the lineups are
        # correlated -- so it is read off the simulation instead, where available.
        metrics["p_any_wins_independent"] = round(
            float(1 - np.prod(1 - rows["p_win"].to_numpy())), 5)
    if payout is not None and "roi" in rows.columns:
        spend = payout.entry_fee * len(selection)
        metrics["total_entry_fees"] = round(spend, 2)
        metrics["portfolio_roi"] = round(
            float(rows["exp_profit"].sum() / spend) if spend else 0.0, 4)
    return metrics


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def _coverage_gain(state, candidate, weights):
    """Reward for the new kinds of exposure this candidate adds."""
    gain = 0.0
    if weights.stack_coverage and candidate["primary_stack"] not in state["stacks"]:
        gain += weights.stack_coverage
    if weights.pitcher_coverage and candidate["pitcher_pair"] not in state["pairs"]:
        gain += weights.pitcher_coverage
    if weights.game_coverage:
        new_games = set(candidate["games"]) - state["games"]
        gain += weights.game_coverage * len(new_games) / max(len(candidate["games"]), 1)
    return gain


def _universal_players(candidates):
    """Players who appear in *every* candidate -- i.e. locked upstream.

    A player lock is applied when the lineups are generated, so by the time a pool reaches
    here a locked player is in 100% of it. That matters because an exposure cap below 100%
    is then unsatisfiable for him and, applied naively, blocks every remaining candidate and
    silently truncates the set. The `--from-optimizer` path cannot be told which players
    were locked -- it reads a CSV of finished lineups -- so this is inferred rather than
    passed, which also catches a player who is effectively forced for any other reason.
    """
    if not len(candidates):
        return set()
    counts = Counter(player for players in candidates["players"] for player in players)
    return {player for player, count in counts.items() if count >= len(candidates)}


def select(candidate_pool, scores, evaluated, n_lineups=20, weights=None, seed=None,
           max_player_exposure=None, max_stack_exposure=None, locks=None, excludes=None,
           swap_passes=2, quiet=True, exempt_players=None):
    """Choose `n_lineups` candidates jointly. Returns (indices, metrics, trace).

    Greedy forward selection on the portfolio objective, then a local swap pass. Both are
    deterministic given `seed`; the seed only breaks ties.

    `max_player_exposure` is a hard cap on the share of the set any one player may appear
    in. Players locked upstream are exempt from it -- see `_universal_players` -- because a
    lock is a decision and a cap is a preference, and the alternative is returning fewer
    lineups than asked for without saying so.
    """
    weights = weights or PortfolioWeights()
    rng = np.random.default_rng(seed)
    candidates = evaluated
    n_candidates = len(candidates)
    if n_candidates == 0:
        raise PortfolioError("no candidates to choose from")
    n_lineups = min(n_lineups, n_candidates)

    payoffs = (candidates["exp_payout"].to_numpy(dtype=float)
               if "exp_payout" in candidates.columns
               else candidates["ceiling"].to_numpy(dtype=float))
    dupes = (candidates["exp_dupes"].to_numpy(dtype=float)
             if "exp_dupes" in candidates.columns else np.zeros(n_candidates))
    # Scale the coverage and correlation terms to the payoff scale, so the weights mean the
    # same thing whether payouts are in dollars or the fallback is in DK points.
    unit = float(np.nanmax(np.abs(payoffs))) or 1.0

    allowed = np.ones(n_candidates, dtype=bool)
    if excludes:
        for index in excludes:
            allowed[int(index)] = False

    selected = [int(i) for i in (locks or [])]
    for index in selected:
        allowed[index] = False

    # Cap problems always speak up, whatever `quiet` says: they change how many lineups come
    # back, and a set that is quietly short is worse than one that is quietly uncapped.
    # Integer caps, floored, matching `pool.resolve_exposure`. A fractional cap compared
    # directly against a count lets a player reach ceil(): 40% of 24 is 9.6, nothing is
    # blocked at 9, and the tenth appearance takes him to 42%. Floor also cannot round a
    # cap down to zero here -- a cap no lineup can satisfy would empty the whole set.
    player_cap = (max(1, int(np.floor(max_player_exposure * n_lineups)))
                  if max_player_exposure is not None else None)
    stack_cap = (max(1, int(np.floor(max_stack_exposure * n_lineups)))
                 if max_stack_exposure is not None else None)

    exempt = set(exempt_players or ())
    if max_player_exposure is not None and max_player_exposure < 1.0:
        forced = _universal_players(candidates) - exempt
        if forced:
            exempt |= forced
            shown = ", ".join(sorted(str(p) for p in list(forced)[:6]))
            print(f"  [i] exposure cap does not apply to {len(forced)} locked player(s) "
                  f"({shown}{', ...' if len(forced) > 6 else ''}): they are in every "
                  f"candidate, so {max_player_exposure:.0%} is unreachable for them.")

    state = {"stacks": set(), "pairs": set(), "games": set(),
             "player_counts": {}, "stack_counts": {}}

    def absorb(index):
        row = candidates.iloc[index]
        state["stacks"].add(row["primary_stack"])
        state["pairs"].add(row["pitcher_pair"])
        state["games"].update(row["games"])
        for player in row["players"]:
            state["player_counts"][player] = state["player_counts"].get(player, 0) + 1
        state["stack_counts"][row["primary_stack"]] = \
            state["stack_counts"].get(row["primary_stack"], 0) + 1

    for index in selected:
        absorb(index)

    trace = []
    with profiler.stage("portfolio", candidates=n_candidates, target=n_lineups):
        # Running best-of-set score per simulation. Adding a lineup improves the portfolio
        # only where it beats everything already in it, which is exactly what makes this
        # objective submodular and greedy defensible.
        if selected:
            running_best = scores[:, selected].max(axis=1)
        else:
            running_best = np.full(scores.shape[0], -np.inf, dtype=np.float32)
        # Built on first use: a run with the correlation weight at zero never needs it.
        standard = None

        while len(selected) < n_lineups:
            # --- hard exposure caps ---
            usable = allowed.copy()
            if player_cap is not None and selected:
                blocked = {p for p, c in state["player_counts"].items()
                           if c >= player_cap and p not in exempt}
                if blocked:
                    for index in np.flatnonzero(usable):
                        if blocked & set(candidates.iloc[index]["players"]):
                            usable[index] = False
            if stack_cap is not None and selected:
                blocked = {s for s, c in state["stack_counts"].items() if c >= stack_cap}
                if blocked:
                    for index in np.flatnonzero(usable):
                        if candidates.iloc[index]["primary_stack"] in blocked:
                            usable[index] = False
            if not usable.any():
                # Always reported, and always with the binding cap named. Returning 11
                # lineups when 18 were asked for is a result the caller has to know about
                # before uploading, and "which cap do I raise" is the immediate next
                # question -- a pool with two primary stacks cannot fill 18 entries under a
                # 35% stack cap no matter how the players are capped.
                reasons = []
                if stack_cap is not None:
                    at_cap = [s for s, c in state["stack_counts"].items() if c >= stack_cap]
                    if at_cap:
                        reasons.append(
                            f"--max-stack-exposure ({stack_cap}/{n_lineups} each) with only "
                            f"{candidates['primary_stack'].nunique()} primary stack(s) in "
                            f"the pool -> at most "
                            f"{stack_cap * candidates['primary_stack'].nunique()} lineups")
                if player_cap is not None:
                    at_cap = [p for p, c in state["player_counts"].items()
                              if c >= player_cap and p not in exempt]
                    if at_cap:
                        reasons.append(f"--max-player-exposure ({player_cap}/{n_lineups}): "
                                       f"{len(at_cap)} player(s) maxed out")
                print(f"  [!] stopped at {len(selected)} of {n_lineups}: no legal candidate "
                      f"left.")
                for reason in reasons:
                    print(f"      binding: {reason}")
                print(f"      raise the cap, or widen the candidate pool "
                      f"({len(candidates)} candidates available).")
                break

            options = np.flatnonzero(usable)

            # --- the marginal value of each option ---
            # E[max] gain, vectorised over every option at once. This is the expensive term
            # and the reason the whole thing is fast: one (sims x options) maximum rather
            # than a loop over options.
            block = scores[:, options]
            gain_best = (np.maximum(block, running_best[:, None]).mean(axis=0)
                         - (running_best.mean() if np.isfinite(running_best).all() else 0.0))
            value = weights.payout * payoffs[options] + weights.best_of * gain_best * \
                (unit / max(float(np.nanmax(np.abs(gain_best))) or 1.0, 1e-9)) * 0.15
            value = value - weights.duplication * dupes[options] * unit * 0.05

            if selected and weights.correlation:
                # Correlation against what is already chosen, not against everything.
                # With standardised columns the mean correlation of option i against the
                # chosen set is (Z[:, i] . S) / (n_sims * k) for S the chosen column sum --
                # one matrix-vector product instead of a matmul that grows with k.
                if standard is None:
                    standard = _standardise(scores)
                chosen_sum = standard[:, selected].sum(axis=1)
                correlation = (standard[:, options].T @ chosen_sum) / (
                    scores.shape[0] * len(selected))
                value = value - weights.correlation * correlation * unit

            for position, index in enumerate(options):
                value[position] += _coverage_gain(
                    state, candidates.iloc[index], weights) * unit

            # Deterministic tie-break, seeded.
            value = value + rng.random(len(options)) * 1e-9
            pick = int(options[int(np.argmax(value))])

            selected.append(pick)
            allowed[pick] = False
            absorb(pick)
            running_best = np.maximum(running_best, scores[:, pick])
            trace.append({"step": len(selected), "lineup": int(candidates.iloc[pick]["lineup"]),
                          "value": float(value.max())})

        # --- local improvement ---
        # Locked lineups are held out of the swap pass. Without this the greedy phase
        # honours a lock and the improvement phase quietly trades it away, so `--lock`
        # would work or not depending on whether a better swap happened to exist.
        #
        # Evaluated incrementally (`_SwapState`). Recomputing the whole objective per trial
        # swap costs 125 ms at n=150 with 20,000 simulations, and the pass wants ~75,000 of
        # them -- 2.6 hours. The state below updates in O(n_sims) instead.
        locked_positions = set(range(len(locks or [])))

        # The greedy phase honours the caps and the swap phase used to ignore them, so a
        # capped player could be swapped straight back over the line. Checked incrementally:
        # a swap only changes the counts of the two lineups involved.
        def swap_keeps_caps(position, index):
            if max_player_exposure is None and max_stack_exposure is None:
                return True
            outgoing, incoming = candidates.iloc[selected[position]], candidates.iloc[index]
            if player_cap is not None:
                counts = dict(state_counts["players"])
                for player in outgoing["players"]:
                    counts[player] = counts.get(player, 0) - 1
                for player in incoming["players"]:
                    counts[player] = counts.get(player, 0) + 1
                    if player not in exempt and counts[player] > player_cap:
                        return False
            if stack_cap is not None:
                counts = dict(state_counts["stacks"])
                counts[outgoing["primary_stack"]] = \
                    counts.get(outgoing["primary_stack"], 0) - 1
                counts[incoming["primary_stack"]] = \
                    counts.get(incoming["primary_stack"], 0) + 1
                if counts[incoming["primary_stack"]] > stack_cap:
                    return False
            return True

        def apply_counts(position, index):
            outgoing, incoming = candidates.iloc[selected[position]], candidates.iloc[index]
            for player in outgoing["players"]:
                state_counts["players"][player] -= 1
            for player in incoming["players"]:
                state_counts["players"][player] = \
                    state_counts["players"].get(player, 0) + 1
            state_counts["stacks"][outgoing["primary_stack"]] -= 1
            state_counts["stacks"][incoming["primary_stack"]] = \
                state_counts["stacks"].get(incoming["primary_stack"], 0) + 1

        state_counts = {"players": dict(state["player_counts"]),
                        "stacks": dict(state["stack_counts"])}

        if swap_passes and len(selected) > 1:
            state = _SwapState(scores, candidates, selected, weights, payoffs, dupes, unit)
            for _ in range(swap_passes):
                improved = False
                baseline = state.value()
                for position in range(len(selected)):
                    if position in locked_positions:
                        continue
                    current = selected[position]
                    best_value, best_index = baseline, current
                    for index in np.flatnonzero(allowed):
                        score = state.value(position, int(index))
                        if score > best_value + 1e-9 and swap_keeps_caps(position, int(index)):
                            best_value, best_index = score, int(index)
                    if best_index != current:
                        allowed[current] = True
                        allowed[best_index] = False
                        apply_counts(position, best_index)
                        state.apply(position, best_index)
                        selected[position] = best_index
                        baseline = best_value
                        improved = True
                if not improved:
                    break

    metrics = portfolio_metrics(scores, selected, candidates)
    return selected, metrics, trace


def _objective(scores, candidates, selection, weights, payoffs, dupes, unit):
    """The portfolio objective, evaluated on a complete selection."""
    block = scores[:, selection]
    value = weights.payout * float(payoffs[selection].sum())
    value += weights.best_of * float(block.max(axis=1).mean()) * unit * 0.15
    value -= weights.duplication * float(dupes[selection].sum()) * unit * 0.05
    if weights.correlation and len(selection) > 1:
        value -= weights.correlation * pairwise_correlation(scores, selection) * unit * len(selection)
    rows = candidates.iloc[selection]
    value += weights.stack_coverage * rows["primary_stack"].nunique() * unit
    value += weights.pitcher_coverage * rows["pitcher_pair"].nunique() * unit
    value += weights.game_coverage * len({g for games in rows["games"] for g in games}) * unit
    return value


# ---------------------------------------------------------------------------
# Comparison baselines
# ---------------------------------------------------------------------------

def baselines(candidate_pool, scores, evaluated, n_lineups=20, seed=7):
    """Alternative ways of picking the same number of lineups, for comparison.

    A portfolio method is only worth its complexity if it beats "take the top twenty by
    ceiling", so that comparison ships alongside it rather than being left to the reader.
    """
    rng = np.random.default_rng(seed)
    out = {}
    frame = evaluated

    def top(column, ascending=False):
        ordered = frame.sort_values(column, ascending=ascending)
        return list(ordered.index[:n_lineups])

    if "ceiling" in frame.columns:
        out["highest ceiling"] = top("ceiling")
    if "proj" in frame.columns:
        out["highest projection"] = top("proj")
    if "own_sum" in frame.columns:
        out["lowest ownership"] = top("own_sum", ascending=True)
    if "exp_payout" in frame.columns:
        out["highest EV (independently)"] = top("exp_payout")
    if "p_win" in frame.columns:
        out["highest P(win)"] = top("p_win")
    out["random"] = list(rng.choice(len(frame), size=min(n_lineups, len(frame)),
                                    replace=False))
    return out


def compare(candidate_pool, scores, evaluated, n_lineups=20, weights=None, seed=7,
            payout=None, swap_passes=2, max_player_exposure=None, max_stack_exposure=None,
            exempt_players=None):
    """Selected portfolio against every baseline, on the metrics that decide the night.

    The exposure caps are forwarded rather than defaulted: this is the only path `main`
    takes, so a cap that stops here is a cap that never happens.
    """
    chosen, metrics, _ = select(candidate_pool, scores, evaluated,
                                n_lineups=n_lineups, weights=weights, seed=seed,
                                swap_passes=swap_passes,
                                max_player_exposure=max_player_exposure,
                                max_stack_exposure=max_stack_exposure,
                                exempt_players=exempt_players)
    rows = [{"method": "joint portfolio", **portfolio_metrics(scores, chosen, evaluated,
                                                              payout=payout)}]
    for name, selection in baselines(candidate_pool, scores, evaluated, n_lineups, seed).items():
        rows.append({"method": name,
                     **portfolio_metrics(scores, selection, evaluated, payout=payout)})
    columns = ["method", "best_of_set_mean", "best_of_set_p99", "effective_lineups",
               "mean_pairwise_correlation", "distinct_primary_stacks",
               "distinct_pitcher_pairs", "distinct_players"]
    for extra in ("total_expected_profit", "portfolio_roi", "mean_expected_duplicates"):
        if extra in rows[0]:
            columns.append(extra)
    return chosen, pd.DataFrame(rows)[columns]


def save_report(metrics, comparison, label, extra=None):
    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    path = os.path.join(BENCHMARK_DIR, f"portfolio_{label}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   **(extra or {}), "portfolio": metrics,
                   "comparison": comparison.to_dict("records")},
                  handle, indent=2, default=str)
    return path


def format_report(metrics, comparison, weights):
    lines = [
        f"portfolio of {metrics['n']} lineups",
        f"  best-of-set mean {metrics['best_of_set_mean']:.1f}, "
        f"p99 {metrics['best_of_set_p99']:.1f}",
        f"  mean pairwise correlation {metrics['mean_pairwise_correlation']:.3f} "
        f"-> {metrics['effective_lineups']:.1f} effective independent lineups",
        f"  {metrics['distinct_primary_stacks']} primary stacks, "
        f"{metrics['distinct_pitcher_pairs']} pitcher pairs, "
        f"{metrics['distinct_players']} distinct players",
    ]
    if "mean_expected_duplicates" in metrics:
        lines.append(f"  mean expected duplicates {metrics['mean_expected_duplicates']:.2f}")
    lines.append("")
    lines.append("against every other way of picking the same number of lineups:")
    lines.append(comparison.to_string(index=False))
    lines.append("")
    lines.append(
        "  'effective_lineups' is the number that matters: twenty lineups at 0.9 mean\n"
        "  correlation are worth about two independent bets. It is what --max-overlap was\n"
        "  always approximating by counting shared players.")
    return "\n".join(lines)


def main():
    import argparse
    import time
    from datetime import datetime as _dt

    from .contest import (CALIBRATION_WARNING, build_pipeline, evaluate)

    parser = argparse.ArgumentParser(
        description="Choose a whole set of lineups jointly, against a simulated field.")
    parser.add_argument("--date", default=_dt.today().strftime("%Y-%m-%d"))
    parser.add_argument("--slate")
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--stage")
    parser.add_argument("--n", type=int, default=20, help="Lineups to enter.")
    parser.add_argument("--contest", default="twenty-max",
                        choices=["single-entry", "three-max", "twenty-max", "milly-maker"])
    parser.add_argument("--weights", choices=sorted(PRESETS),
                        help="Portfolio objective preset. Defaults to match --contest.")
    parser.add_argument("--entry-fee", type=float)
    parser.add_argument("--field-size", type=int)
    parser.add_argument("--field-entries", type=int, default=3000)
    parser.add_argument("--payout-json")
    parser.add_argument("--candidates", type=int, default=800)
    parser.add_argument("--sims", type=int, default=8000)
    parser.add_argument("--max-ownership", type=float)
    parser.add_argument("--from-optimizer", nargs="?", const=True, metavar="PATH",
                        help="Select from the lineups dfs.optimize already generated rather "
                             "than generating new candidates. Bare flag finds the night's "
                             "newest lineups CSV. This is the recommended production path: "
                             "the optimizer carries the pool file, exposure minimums, "
                             "explicit stacks and the started-game filter, and this stage "
                             "adds correlation-aware selection on top.")
    parser.add_argument("--write-lineups", nargs="?", const=True, metavar="PATH",
                        help="Write the chosen set back in the optimizer's own CSV format so "
                             "dfs.upload can read it. Needs --from-optimizer.")
    parser.add_argument("--allow-started", action="store_true",
                        help="Keep players whose game has begun. Analysis only.")
    parser.add_argument("--max-player-exposure", type=float, metavar="FRACTION",
                        help="Hard cap on how many of the set any one player may appear in.")
    parser.add_argument("--max-stack-exposure", type=float, metavar="FRACTION")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--swap-passes", type=int, default=2,
                        help="Local improvement passes after greedy selection (default 2). "
                             "0 runs greedy only, which already carries the submodular "
                             "guarantee; the passes typically move a handful of lineups.")
    parser.add_argument("--out", help="Write the chosen lineups to a CSV.")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    weights = PRESETS[args.weights] if args.weights else PRESETS.get(
        args.contest, PortfolioWeights())

    with profiler.session("portfolio", date=args.date, slate=args.slate,
                          enabled=args.profile, n=args.n):
        started = time.perf_counter()
        parts = build_pipeline(
            args.date, slate=args.slate, snapshot=args.snapshot, stage=args.stage,
            n_candidates=args.candidates, sims=args.sims,
            field_entries=args.field_entries, contest=args.contest,
            entry_fee=args.entry_fee, field_size=args.field_size,
            payout_json=args.payout_json, seed=args.seed,
            max_ownership=args.max_ownership, from_optimizer=args.from_optimizer,
            allow_started=args.allow_started)
        evaluated = evaluate(parts["candidates"], parts["simulations"], parts["field"],
                             parts["payout"])
        scores = lineup_scores(parts["candidates"], parts["simulations"])
        chosen, comparison = compare(parts["candidates"], scores, evaluated,
                                     n_lineups=args.n, weights=weights, seed=args.seed,
                                     payout=parts["payout"],
                                     swap_passes=args.swap_passes,
                                     max_player_exposure=args.max_player_exposure,
                                     max_stack_exposure=args.max_stack_exposure)
        metrics = portfolio_metrics(scores, chosen, evaluated, payout=parts["payout"])
        elapsed = time.perf_counter() - started

    print()
    print(format_report(metrics, comparison, weights))
    print()
    print(CALIBRATION_WARNING)

    picked = evaluated.iloc[chosen]
    columns = [c for c in ("lineup", "salary", "proj", "ceiling", "own_sum", "stack_shape",
                           "primary_stack", "pitcher_pair", "sim_mean", "exp_dupes",
                           "p_top1", "exp_payout")
               if c in picked.columns]
    print(f"\n=== the {len(chosen)} lineups ===")
    print(picked[columns].to_string(index=False))
    print(f"\n  {elapsed:.1f}s total")

    # The label the slate resolved to, not the one typed. Writing
    # `portfolio_<date>_main_*.json` for an early-slate run mislabels the artifact and the
    # mistake only surfaces weeks later when the file is read back.
    label = args.slate or parts["meta"].get("slate_label") or "main"
    path = save_report(metrics, comparison,
                       f"{args.date}_{label}_{args.contest}",
                       extra={"date": args.date, "slate": args.slate, "seed": args.seed,
                              "n_lineups": args.n, "weights": weights.to_dict(),
                              "source": parts.get("source_path")})
    print(f"  -> {path}")
    if args.out:
        picked.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"  -> {args.out}")

    if args.write_lineups:
        from .candidates import to_optimizer_format
        from .naming import resolve
        source = parts.get("source_path")
        if not source:
            print("\n[!] --write-lineups needs --from-optimizer: only lineups that came "
                  "from the optimizer can be written back in its format.")
        else:
            long_format = to_optimizer_format(parts["candidates"], chosen, source)
            if isinstance(args.write_lineups, str):
                target, note = args.write_lineups, ""
            else:
                target, note = resolve("lineups", args.date,
                                       args.slate or parts["meta"].get("slate_label"),
                                       OUTPUT_ROOT, False)
            long_format.to_csv(target, index=False, encoding="utf-8-sig")
            print(f"\n  portfolio selection -> {target}")
            if note:
                print(f"  [i] {note}")
            print(f"  ready to upload: python -m dfs.upload --date {args.date}"
                  + (f" --slate {label}" if label else ""))

    if args.profile:
        from .profiling import format_report as fmt
        print("\n=== pipeline profile ===")
        print(fmt(profiler.last_report))


if __name__ == "__main__":
    main()
