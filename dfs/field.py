"""Simulate the opponent field: legal lineups built the way real entrants build them.

**Why a marginal ownership model is not enough.** `dfs.ownership` says what fraction of the
field rosters each player. That is the right answer to the question it asks, and it is
useless for duplication, because duplication depends on which players are rostered
*together*. Measured across the first 26,000 real field lineups in `dk_results/`:

    P(teammate B | A rostered) / P(B)  =  2.62x      (median 2.31, p90 4.59)
    P(B on another team | A) / P(B)    =  0.75x
    pitcher pairs, vs independence     =  1.01x      (median; 51% above, 49% below)

So teammates are 2.6x more likely to appear together than independence implies, players on
other teams slightly *less* likely (they compete for the same salary), and pitcher pairs are
almost exactly independent once you condition on their marginals. A model that draws ten
players independently reproduces the marginals perfectly and gets every one of those wrong.

**How the field actually builds.** Refreshed 2026-09-02 over 232,935 legal lineups in 91
dated Classic contest exports -- `calibrate()` produces exactly this block, so it is the
number the constants below are set from:

    primary stack size    5 hitters 42.2%   4: 26.1%   3: 17.3%   2: 12.2%   1: 2.3%
    commonest shapes      5-2 20.3%   5-3 12.1%   5 9.8%   4-2 9.6%   4-3 7.9%
    pitcher vs own stack  0.8% of matched lineups
    both P in one game    4.7% of matched lineups
    duplication           91.0% of entries unique overall, max 27x; 81-98% by contest

That is the generative process implemented here: draw a primary stack, draw a secondary,
fill the rest by marginal appeal, add two pitchers, then repair to legality.

**Calibration, not invention.** Every constant below was measured, and `calibrate()`
recomputes them from whatever is in `dk_results/`. `validate()` checks the simulated field
back against the real one on the three things that matter: per-player ownership, stack-shape
distribution, and duplication.
"""

import glob
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .optimizer import MAX_HITTERS_PER_TEAM, ROSTER, ROSTER_SIZE, eligible_positions
from .profiling import profiler
from .salaries import canon_team, normalize_name
from .scoring import DK_SALARY_CAP

BENCHMARK_DIR = os.path.join("docs", "benchmarks")
RESULTS_DIR = "dk_results"

# The ten seats, in the order a DK entry lists them.
SEATS = [slot for slot, count in ROSTER.items() for _ in range(count)]
HITTER_SEATS = [s for s in SEATS if s != "P"]

# ---------------------------------------------------------------------------
# Measured field behaviour. Provenance: 232,935 legal lineups across 91 dated Classic
# contest exports in dk_results/; 214,282 lineups matched both pitchers to dated opponents.
# Showdown, undated, sub-90-entry, and invalidly joined lineups are excluded. Recompute with
# `python -m dfs.field --calibrate`.
# ---------------------------------------------------------------------------

# How many hitters the field takes from its most-used team.
PRIMARY_SIZE_SHARE = {1: 0.0226, 2: 0.1218, 3: 0.1730, 4: 0.2605, 5: 0.4222}

# Secondary stack size, conditional on the primary. This is the conditional structure the
# independent model cannot express: 5-2 is the single commonest shape in the whole field at
# 20.3%, and it only exists because the second stack is chosen *knowing* the first.
SECONDARY_SIZE_SHARE = {
    1: {0: 1.0000},
    2: {0: 0.4102, 2: 0.5898},
    3: {0: 0.2121, 2: 0.5516, 3: 0.2364},
    4: {0: 0.2001, 2: 0.4142, 3: 0.3028, 4: 0.0828},
    5: {0: 0.2321, 2: 0.4815, 3: 0.2864},
}

# Share of field lineups whose pitcher faces 3+ of their own hitters. The field treats this
# as nearly a hard rule, so the simulator does too rather than letting it emerge from
# independent draws, which would produce it an order of magnitude too often.
CONFLICT_RATE = 0.0084

# Share with both pitchers drawn from the same game.
SAME_GAME_PITCHER_RATE = 0.0470

# How sharply the field concentrates on its favourites when choosing *within* a group.
# Ownership is already a probability; raising it to this power sharpens (>1) or flattens
# (<1) the choice. Above 1.0 because the field is more concentrated than its own marginals
# suggest: the marginal is an average over users with different processes, while any one
# user picks decisively.
APPEAL_POWER = 1.35

# Probability that a user takes the *canonical* version of a stack -- the team's top N by
# ownership -- rather than sampling which N. And, given that, the probability they fill the
# leftover seats with the obvious value plays too.
#
# This is where duplication comes from, and it is why an independent sampler cannot produce
# it. Thousands of entrants run similar tools over the same projections and arrive at the
# same "top 5 of the order" stack independently. With these switched off the simulated field
# came back 99.8% unique with a maximum duplication of 2, against a real 92.9% and 23.
#
# Both are FITTED, not guessed: a 4x4 sweep against the duplication measured across the
# first 10 exports in dk_results/ (see docs/benchmarks.md). At these values a 6,000-entry
# field simulates to 93.1% unique with a maximum of 24, against a measured 92.9% and 23.
#
# Re-checked 2026-08-08 against all 19 exports, which measure 91.0% unique / max 27 -- the
# fit still lands inside the per-contest spread and was left alone. It is the one pair of
# constants here that `calibrate()` does not recompute, so it needs a sweep, not a re-read.
CANONICAL_STACK_RATE = 0.25
CANONICAL_FILL_RATE = 0.08

# Real field lineups spend nearly the whole cap. Below this share the repair step keeps
# trading up, because a simulated field that leaves $4,000 on the table is not the field you
# are actually playing against and would understate duplication on expensive chalk.
MIN_SALARY_SHARE = 0.955

# How much a team that has already had its stack placed is damped when filling the leftover
# seats. A user who decided on a four-stack means four; they do not then accidentally add a
# fifth from the same team while shopping for value.
#
# Undamped, the fill step inflated five-stacks to 59% against a measured 54% and starved
# two-stacks (5% against 10%), because every four-stack had eight more chances to become a
# five. Not zero, because incidental teammate pairs are real -- 3-2-2 is 2.6% of the field.
FILL_STACK_DAMPING = 0.25


class FieldError(Exception):
    pass


@dataclass
class ContestConfig:
    """The shape of the contest being simulated.

    Field construction genuinely differs by shape: a single-entry field stacks more
    conservatively and duplicates less, while a 20-max field is dominated by a few hundred
    users submitting structured blocks of correlated lineups.
    """

    name: str = "GPP"
    entries: int = 10000
    entry_fee: float = 5.0
    max_entries_per_user: int = 20
    # Multi-entry users reuse a core across their block, which is a large part of why the
    # real field duplicates at all. 0 makes every entry independent.
    block_reuse: float = 0.55

    def to_dict(self):
        return asdict(self)


# Shapes seen in dk_results/, offered as presets.
PRESETS = {
    "single-entry": ContestConfig("single-entry", entries=300, max_entries_per_user=1,
                                  block_reuse=0.0),
    "three-max": ContestConfig("three-max", entries=1500, max_entries_per_user=3,
                               block_reuse=0.35),
    "twenty-max": ContestConfig("twenty-max", entries=6000, max_entries_per_user=20,
                                block_reuse=0.55),
    "milly-maker": ContestConfig("milly-maker", entries=100000, entry_fee=20.0,
                                 max_entries_per_user=150, block_reuse=0.65),
}


# ---------------------------------------------------------------------------
# Parsing real contest lineups
# ---------------------------------------------------------------------------

SLOT_PATTERN = re.compile(r"\b(P|C|1B|2B|3B|SS|OF)\s+")


def parse_contest_lineup(text):
    """DK's 'Lineup' cell -> [(slot, player name), ...].

    The cell is slot labels and names run together with no delimiter
    ('1B Michael Busch 2B Nico Hoerner ...'), so the slot tokens are the only structure
    available. Splitting on them is why this is a regex and not a `str.split`.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    parts = SLOT_PATTERN.split(text)
    return [(parts[i], parts[i + 1].strip()) for i in range(1, len(parts) - 1, 2)]


def read_contest_lineups(path):
    """(lineups, ownership, entries) from one standings export.

    Only complete ten-player lineups are returned; DK pads the export with reservation rows
    that have no entry yet, and counting those as lineups would deflate every rate here.
    """
    from .results import read_result_frame
    frame = read_result_frame(path)
    frame.columns = [str(c).strip() for c in frame.columns]
    lineups = []
    for text in frame.get("Lineup", pd.Series(dtype=str)).fillna(""):
        parsed = parse_contest_lineup(text)
        if len(parsed) == ROSTER_SIZE:
            lineups.append([(slot, normalize_name(name)) for slot, name in parsed])

    ownership = {}
    if "Player" in frame.columns and "%Drafted" in frame.columns:
        block = frame[["Player", "%Drafted"]].dropna(subset=["Player"])
        for _, row in block.iterrows():
            try:
                ownership[normalize_name(row["Player"])] = float(
                    str(row["%Drafted"]).rstrip("%").strip())
            except ValueError:
                continue

    entries = int(pd.to_numeric(frame.get("Points"), errors="coerce").notna().sum())
    return lineups, ownership, entries


def max_entries_of(path):
    """Max entries per user, read off DK's 'handle (3/20)' entry names."""
    from .results import read_result_frame
    frame = read_result_frame(
        path, usecols=lambda c: str(c).strip() == "EntryName")
    counts = Counter()
    for name in frame.iloc[:, 0].dropna():
        match = re.search(r"\((\d+)/(\d+)\)$", str(name))
        counts[int(match.group(2)) if match else 1] += 1
    return counts.most_common(1)[0][0] if counts else 1


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _weights(values, power=None):
    """Non-negative selection weights from ownership, normalised. Never all-zero.

    `power` is looked up at call time rather than bound as a default argument. Bound, the
    module constant became unreachable the moment the module was imported -- a calibration
    sweep over APPEAL_POWER silently produced identical results at every value, which is
    the worst way for a tunable to fail.
    """
    power = APPEAL_POWER if power is None else power
    array = np.asarray(values, dtype=float)
    array = np.where(np.isfinite(array), array, 0.0)
    array = np.clip(array, 0.0, None) ** power
    total = array.sum()
    if total <= 0:
        return np.full(len(array), 1.0 / max(len(array), 1))
    return array / total


# How much the field's choices are driven by points-per-dollar as well as by ownership.
#
# **Zero, and that is a measured result rather than a default.** The hypothesis was that
# ownership-only sampling would build a field that scores below the real one, because
# ownership says who the field rosters but not how well those players combine. Swept from
# 0.0 to 0.85 against the score distributions of the five 1,000+ entry contests in
# dk_results/:
#
#     value weight   winner    p99    p80  median   ownership MAE
#            0.00     184.7  153.5  108.4    86.0            2.17
#            0.55     186.8  154.6  109.2    87.0            3.10
#            0.85     184.8  152.0  108.9    87.5            3.89
#     real             180.4  150.0  112.1    90.1
#
# It buys 1.5 points of field median and costs 1.7 points of ownership MAE -- a bad trade,
# and the field was never far off to begin with. Kept as a knob because it is the obvious
# thing to try again on more contests, but off by default because on this evidence it does
# not pay.
FIELD_VALUE_WEIGHT = 0.0


def _draw(rng, options, weights, k=1, replace=False):
    if len(options) == 0:
        return []
    k = min(k, len(options))
    picked = rng.choice(len(options), size=k, replace=replace,
                        p=weights / weights.sum())
    return [options[i] for i in np.atleast_1d(picked)]


def seat_mask(slot_set, seats=None):
    """Bitmask over the eight hitter seats this player's eligibility can fill."""
    seats = HITTER_SEATS if seats is None else seats
    mask = 0
    for seat, slot in enumerate(seats):
        if slot in slot_set:
            mask |= 1 << seat
    return mask


# Feasibility is asked hundreds of times per drawn entry and the answer depends only on the
# *multiset of eligibility patterns*, not on who the players are. There are very few
# distinct patterns on a DK slate, so the same handful of questions recur across thousands
# of entries. Profiling put 10 of 15 seconds in this check before it was memoised.
_SEAT_CACHE = {}
_SEAT_CACHE_LIMIT = 200_000


def _seatable(masks):
    """True when every mask in `masks` can be given a distinct hitter seat.

    Counting "eight hitters" is not the question DK asks -- three shortstops are three
    hitters and still cannot be rostered together -- so this is a real bipartite matching,
    the same check `dfs.optimizer` runs before forcing players in. Kuhn's algorithm over
    eight seats, with the seats held as a bitmask instead of a set.
    """
    key = tuple(sorted(masks))
    cached = _SEAT_CACHE.get(key)
    if cached is not None:
        return cached

    n_seats = len(HITTER_SEATS)
    if len(key) > n_seats:
        result = False
    else:
        # match[seat] = index into `key` currently holding that seat, or -1.
        match = [-1] * n_seats

        def augment(mask, seen, claimant):
            """Seat `claimant`, displacing earlier patterns along an augmenting path."""
            for seat in range(n_seats):
                bit = 1 << seat
                if not (mask & bit) or (seen & bit):
                    continue
                held = match[seat]
                if held == -1 or augment(key[held], seen | bit, held):
                    match[seat] = claimant
                    return True
            return False

        result = all(augment(mask, 0, position) for position, mask in enumerate(key))

    if len(_SEAT_CACHE) < _SEAT_CACHE_LIMIT:
        _SEAT_CACHE[key] = result
    return result


def _fits_hitter_seats(slot_sets, seats=None):
    """Set-based wrapper, kept for callers that hold eligibility as sets of slot names."""
    return _seatable([seat_mask(s, seats) for s in slot_sets])


class FieldPool:
    """The slate, pre-indexed for fast repeated sampling.

    Building this once and reusing it is the difference between simulating a 100,000-entry
    field in seconds and in minutes: every lookup below is done once here rather than once
    per drawn lineup.
    """

    def __init__(self, players, ownership_column="Own%",
                 value_weight=FIELD_VALUE_WEIGHT):
        frame = players[players["Salary"].notna()].copy()
        frame = frame[frame.apply(lambda r: bool(eligible_positions(r)), axis=1)]
        self.players = frame.reset_index(drop=True)
        if self.players.empty:
            raise FieldError("no priced, roster-eligible players on this slate")

        n = len(self.players)
        self.n = n
        self.slot_sets = [eligible_positions(self.players.iloc[i]) for i in range(n)]
        # Seat eligibility as a bitmask, computed once. The feasibility check runs hundreds
        # of times per drawn entry and must not be rebuilding these.
        self.seat_masks = [seat_mask(s) for s in self.slot_sets]
        self.salary = pd.to_numeric(self.players["Salary"], errors="coerce").fillna(0).to_numpy()
        self.is_hitter = (self.players["Type"] == "H").to_numpy()
        self.teams = self.players["Team"].map(canon_team).to_numpy()
        self.opponents = self.players["Opp"].map(canon_team).to_numpy()
        self.games = self.players.get("Game", pd.Series("", index=self.players.index)) \
            .astype(str).to_numpy()
        self.names = self.players["Name"].to_numpy()
        self.keys = np.array([normalize_name(x) for x in self.names])

        # `.get` on a missing column returns None, and pd.to_numeric(None) hands back a bare
        # numpy nan with no .isna() -- so the column has to be checked before conversion or
        # the guard below raises AttributeError instead of the message it exists to give.
        if ownership_column not in self.players.columns:
            raise FieldError(
                f"the slate has no '{ownership_column}' column to sample the field from. "
                "Build the board first so ownership is estimated.")
        own = pd.to_numeric(self.players[ownership_column], errors="coerce")
        if own.isna().all():
            raise FieldError(
                f"the slate has no '{ownership_column}' column to sample the field from. "
                "Build the board first so ownership is estimated.")
        self.own = own.fillna(0.0).to_numpy()

        self.hitters = np.flatnonzero(self.is_hitter)
        self.pitchers = np.flatnonzero(~self.is_hitter)

        # What the field actually samples on: ownership, tilted towards points per dollar.
        # `own` stays the pure marginal, because that is what validation checks against and
        # what duplication is reported in terms of. Built after the group indices it needs.
        self.appeal = self._blend_appeal(value_weight)
        # Pre-sorted for the canonical (highest-owned-first) path, so the fill loop never
        # sorts inside the per-entry hot path.
        self.hitters_by_own = self.hitters[np.argsort(-self.appeal[self.hitters])]
        self.pitchers_by_own = self.pitchers[np.argsort(-self.appeal[self.pitchers])]
        if len(self.pitchers) < ROSTER["P"]:
            raise FieldError(f"only {len(self.pitchers)} pitchers priced; need {ROSTER['P']}")

        self.by_team = defaultdict(list)
        for i in self.hitters:
            self.by_team[self.teams[i]].append(int(i))
        # A team the field cannot legally stack is not a stacking candidate.
        self.stackable = {t: v for t, v in self.by_team.items() if len(v) >= 2}
        if not self.stackable:
            raise FieldError("no team has two priced hitters; cannot build a field")

        # Team-level stack appeal: how much total appeal the team's bats carry. The field
        # stacks the offenses it likes, and its own ownership is the cleanest statement of
        # which those are.
        self.team_appeal = {t: float(self.appeal[v].sum()) for t, v in self.stackable.items()}
        self.cheapest_hitter = float(self.salary[self.hitters].min())
        self.cheapest_pitcher = float(self.salary[self.pitchers].min())

    def _blend_appeal(self, value_weight):
        """Ownership tilted towards points per dollar, within each roster group.

        Both terms are converted to within-group percentiles first, because raw ownership
        (0-60) and raw value (0-4) are on scales that cannot be added. The result is
        rescaled back onto the ownership scale so the sampling weights keep their meaning.
        """
        if not value_weight:
            return self.own.copy()
        projected = pd.to_numeric(self.players.get("Proj"), errors="coerce")
        if projected is None or projected.isna().all():
            return self.own.copy()
        value = (projected.fillna(0.0).to_numpy()
                 / np.maximum(self.salary, 1.0) * 1000.0)

        blended = self.own.astype(float).copy()
        for members in (self.hitters, self.pitchers):
            if len(members) < 2:
                continue
            own_rank = pd.Series(self.own[members]).rank(pct=True).to_numpy()
            value_rank = pd.Series(value[members]).rank(pct=True).to_numpy()
            mixed = (1.0 - value_weight) * own_rank + value_weight * value_rank
            # Back onto the ownership scale, preserving the group's total.
            total = self.own[members].sum()
            blended[members] = mixed / max(mixed.sum(), 1e-9) * total
        return blended

    def team_order(self, team):
        """That team's hitters, best-owned first."""
        members = self.stackable.get(team, [])
        return sorted(members, key=lambda i: -self.appeal[i])


# ---------------------------------------------------------------------------
# Drawing one entry
# ---------------------------------------------------------------------------

def _pick_stack(rng, pool, team, size, taken, budget_left, seats_left, canonical=False):
    """Choose `size` hitters from one team that can actually be seated and afforded.

    `canonical` takes the team's top `size` by ownership instead of sampling. That is what a
    large slice of the field does -- "stack the top of the order" is one decision, not five
    -- and it is the mechanism that makes real fields duplicate.
    """
    members = [i for i in pool.stackable.get(team, []) if i not in taken]
    if len(members) < size:
        return None

    if canonical:
        order = list(np.argsort(-pool.appeal[members]))
    else:
        weights = _weights(pool.appeal[members])
        # Sampled without replacement, weighted by ownership: the field stacks a team's
        # *good* hitters, not a uniform sample of the nine.
        order = list(rng.choice(len(members), size=len(members), replace=False,
                                p=weights / weights.sum()))

    chosen, slot_sets = [], []
    for position in order:
        if len(chosen) == size:
            break
        index = members[position]
        trial = slot_sets + [pool.seat_masks[index]]
        # Always matched against the full set of hitter seats. Slicing the seat list to the
        # number chosen so far looks like a tightening but is simply wrong: HITTER_SEATS is
        # ordered C/1B/2B/3B/SS/OF/OF/OF, so a five-man stack of outfielders was tested
        # against seats that contained no outfield slot and could never be built. That one
        # line was most of why five-stacks came out at 45% against the 54% measured at the
        # time. (The 2026-08-08 recalibration puts the real figure at 45%; that is a change
        # in the field, not a retraction of this bug -- the seat-slicing was still wrong.)
        if not _seatable(trial):
            continue
        cost = pool.salary[index]
        remaining = max(0, seats_left - len(trial))
        if budget_left - cost < pool.cheapest_hitter * remaining:
            continue
        chosen.append(index)
        slot_sets = trial
        budget_left -= cost
    return chosen if len(chosen) == size else None


def _seat_lineup(pool, indices):
    """Assign chosen players to DK's ten seats, or None if they do not fit.

    Returned in DK's slot order so a simulated entry is directly comparable with a real one.
    """
    hitters = [i for i in indices if pool.is_hitter[i]]
    pitchers = [i for i in indices if not pool.is_hitter[i]]
    if len(pitchers) != ROSTER["P"] or len(hitters) != len(HITTER_SEATS):
        return None

    assignment = {}
    used = set()

    def place(position):
        if position == len(hitters):
            return True
        for seat, slot in enumerate(HITTER_SEATS):
            if seat in used or slot not in pool.slot_sets[hitters[position]]:
                continue
            used.add(seat)
            assignment[seat] = hitters[position]
            if place(position + 1):
                return True
            used.discard(seat)
            del assignment[seat]
        return False

    if not place(0):
        return None
    ordered = pitchers + [assignment[seat] for seat in range(len(HITTER_SEATS))]
    return ordered


def draw_entry(rng, pool, config, core=None, max_tries=6):
    """One field entry: a stack-first construction, repaired to legality.

    `core` is a partial lineup a multi-entry user is reusing across their block. Real
    20-max users do not submit twenty independent lineups -- they submit variations on a
    couple of cores, which is most of why the field duplicates at all.
    """
    for attempt in range(max_tries):
        # Decided once per entry, not per decision: a user who stacks the obvious way tends
        # to fill the obvious way too, and it is that correlation between decisions that
        # makes two strangers submit the same lineup.
        canonical_stack = rng.random() < CANONICAL_STACK_RATE
        canonical_fill = canonical_stack and rng.random() < (CANONICAL_FILL_RATE
                                                             / max(CANONICAL_STACK_RATE, 1e-9))
        taken = set(core or [])
        budget = DK_SALARY_CAP - float(pool.salary[list(taken)].sum()) if taken else DK_SALARY_CAP
        hitters_needed = len(HITTER_SEATS) - sum(1 for i in taken if pool.is_hitter[i])
        pitchers_needed = ROSTER["P"] - sum(1 for i in taken if not pool.is_hitter[i])
        if hitters_needed < 0 or pitchers_needed < 0:
            return None

        chosen = list(taken)
        stacked_teams = set()
        primary_team = primary_size = None

        # A reused core *is* this user's stack, not spare players to build a fresh stack
        # around. Drawing a new primary on top of the core gave every 1-, 2- and 3-stack
        # block a second and larger stack in most of its entries, which is where the
        # simulated field's 4-stacks came from: +8pp over the measured share, with 1- and
        # 2-stacks starved by the same amount. A 20-max user reusing a four-stack submits
        # twenty four-stacks, not twenty lineups that stack somebody else as well.
        core_counts = Counter(pool.teams[i] for i in taken if pool.is_hitter[i])
        core_counts.pop(None, None)
        if core_counts:
            inherited, size = max(core_counts.items(), key=lambda item: item[1])
            if size >= 2:
                primary_team, primary_size = inherited, size
                stacked_teams.add(inherited)

        # --- primary stack ---
        if primary_size is None and hitters_needed >= 2:
            sizes = [s for s in PRIMARY_SIZE_SHARE if s <= min(hitters_needed,
                                                               MAX_HITTERS_PER_TEAM)]
            probability = _weights([PRIMARY_SIZE_SHARE[s] for s in sizes])
            drawn_size = int(rng.choice(sizes, p=probability))

            teams = list(pool.stackable)
            appeal = _weights([pool.team_appeal[t] for t in teams])
            candidate = teams[int(rng.choice(len(teams), p=appeal))]
            stack = _pick_stack(rng, pool, candidate, drawn_size, set(chosen),
                                budget, hitters_needed, canonical=canonical_stack)
            if stack:
                chosen.extend(stack)
                stacked_teams.add(candidate)
                budget -= float(pool.salary[stack].sum())
                hitters_needed -= len(stack)
                primary_team, primary_size = candidate, len(stack)

        # --- secondary stack, conditional on the primary ---
        if primary_size is not None and hitters_needed >= 2:
            table = SECONDARY_SIZE_SHARE.get(primary_size, {0: 1.0})
            options = [s for s in table if s == 0 or s <= hitters_needed]
            if options:
                weights = _weights([table[s] for s in options])
                secondary_size = int(rng.choice(options, p=weights))
                if secondary_size >= 2:
                    others = [t for t in pool.stackable if t != primary_team]
                    if others:
                        appeal = _weights([pool.team_appeal[t] for t in others])
                        team = others[int(rng.choice(len(others), p=appeal))]
                        second = _pick_stack(rng, pool, team, secondary_size,
                                             set(chosen), budget, hitters_needed,
                                             canonical=canonical_stack)
                        if second:
                            chosen.extend(second)
                            stacked_teams.add(team)
                            budget -= float(pool.salary[second].sum())
                            hitters_needed -= len(second)

        # --- fill the remaining hitter seats by marginal appeal ---
        chosen = _fill_hitters(rng, pool, chosen, hitters_needed, budget,
                               canonical=canonical_fill, stacked_teams=stacked_teams)
        if chosen is None:
            continue
        budget = DK_SALARY_CAP - float(pool.salary[chosen].sum())

        # --- pitchers ---
        chosen = _add_pitchers(rng, pool, chosen, pitchers_needed, budget,
                               canonical=canonical_fill,
                               allow_conflict_fallback=(attempt == max_tries - 1))
        if chosen is None:
            continue

        seated = _seat_lineup(pool, chosen)
        if seated is None:
            continue
        if not _legal(pool, seated):
            continue
        # A canonical builder does not then go hunting for spare salary; that is a
        # different kind of entrant. Leaving them alone preserves the duplication their
        # decisiveness creates.
        if not canonical_fill:
            seated = _spend_up(rng, pool, seated)
        return seated
    return None


def _fill_hitters(rng, pool, chosen, needed, budget, canonical=False, samples=12,
                  stacked_teams=()):
    """Top the lineup up to eight hitters, weighted by ownership and affordability.

    Candidates are *sampled and then checked* rather than filtered and then sampled. The
    filtered version ran a bipartite seat matching against every one of ~180 hitters on
    every one of eight fills, which was 80% of the simulator's runtime; sampling first turns
    that into a handful of checks. It falls back to the exhaustive scan only when the
    samples all fail, which is where the pool really is tight enough to need it.
    """
    chosen = list(chosen)
    for _ in range(needed):
        current = [i for i in chosen if pool.is_hitter[i]]
        current_masks = [pool.seat_masks[j] for j in current]
        team_counts = Counter(pool.teams[i] for i in current)
        ceiling_cost = budget - pool.cheapest_hitter * max(0, needed - 1)

        def usable(i):
            return (i not in chosen
                    and team_counts[pool.teams[i]] < MAX_HITTERS_PER_TEAM
                    and pool.salary[i] <= ceiling_cost
                    and _seatable(current_masks + [pool.seat_masks[i]]))

        def damped(indices):
            """Ownership weights, with already-stacked teams pushed down."""
            base = pool.appeal[indices].astype(float)
            if stacked_teams:
                penalty = np.array([FILL_STACK_DAMPING
                                    if pool.teams[i] in stacked_teams else 1.0
                                    for i in indices])
                base = base * penalty
            return _weights(base)

        pick = None
        if canonical:
            for i in pool.hitters_by_own:
                if pool.teams[int(i)] in stacked_teams:
                    continue          # the canonical builder took their stack already
                if usable(int(i)):
                    pick = int(i)
                    break
        else:
            affordable = pool.hitters[pool.salary[pool.hitters] <= ceiling_cost]
            if len(affordable):
                weights = damped(affordable)
                for candidate in rng.choice(affordable, size=min(samples, len(affordable)),
                                            replace=True, p=weights / weights.sum()):
                    if usable(int(candidate)):
                        pick = int(candidate)
                        break
        if pick is None:
            # Exhaustive fallback: the sampled draws all clashed on seat or team.
            options = [int(i) for i in pool.hitters if usable(int(i))]
            if not options:
                return None
            pick = options[int(rng.choice(len(options), p=damped(options)))]

        chosen.append(pick)
        budget -= pool.salary[pick]
        needed -= 1
    return chosen


def _add_pitchers(rng, pool, chosen, needed, budget, canonical=False,
                  allow_conflict_fallback=True):
    """Two starters, avoiding the lineup's own stack the way the field does.

    Measured: only 0.5% of real field lineups roster a pitcher facing 3+ of their own
    hitters. Independent draws produce that pairing far more often, so it is suppressed
    explicitly rather than left to emerge -- and left reachable at the measured rate, since
    the field does occasionally do it.
    """
    chosen = list(chosen)
    hitter_teams = Counter(pool.teams[i] for i in chosen if pool.is_hitter[i])
    allow_conflict = rng.random() < CONFLICT_RATE

    for step in range(needed):
        left = needed - step - 1
        candidates = []
        for i in pool.pitchers:
            i = int(i)
            if i in chosen:
                continue
            if pool.salary[i] > budget - pool.cheapest_pitcher * left:
                continue
            if not allow_conflict and hitter_teams.get(pool.opponents[i], 0) >= 3:
                continue
            candidates.append(i)
        if not candidates:
            # No affordable pitcher who avoids this lineup's own stack. Prefer to abandon
            # the entry and let `draw_entry` redraw with a different stack, because the real
            # field almost never makes this construction -- 0.5% of entries. Relaxing here
            # unconditionally pushed a thin 14-pitcher slate to 4.2%, eight times the
            # measured rate, purely because cheap arms happened to face the chosen stack.
            #
            # The relaxation survives as a last resort on the final attempt, so a pool that
            # genuinely cannot do better still yields an entry rather than failing the field.
            if not allow_conflict_fallback:
                return None
            candidates = [int(i) for i in pool.pitchers
                          if i not in chosen
                          and pool.salary[i] <= budget - pool.cheapest_pitcher * left]
            if not candidates:
                return None
        if canonical:
            pick = max(candidates, key=lambda i: pool.appeal[i])
        else:
            weights = _weights(pool.appeal[candidates])
            pick = candidates[int(rng.choice(len(candidates), p=weights))]
        chosen.append(pick)
        budget -= pool.salary[pick]
    return chosen


def _legal(pool, indices):
    if len(set(indices)) != ROSTER_SIZE:
        return False
    if float(pool.salary[list(indices)].sum()) > DK_SALARY_CAP:
        return False
    hitters = [i for i in indices if pool.is_hitter[i]]
    if len(hitters) != len(HITTER_SEATS):
        return False
    if max(Counter(pool.teams[i] for i in hitters).values()) > MAX_HITTERS_PER_TEAM:
        return False
    return len({pool.games[i] for i in indices}) >= 2


def _spend_up(rng, pool, indices, tries=8):
    """Trade cheap players up until the lineup spends what a real entry spends.

    Real field entries use 96-99% of the cap. A simulator that leaves thousands unspent
    produces a field of cheap lineups, which understates duplication on exactly the
    expensive chalk that duplicates most.
    """
    indices = list(indices)
    target = DK_SALARY_CAP * MIN_SALARY_SHARE
    total = float(pool.salary[indices].sum())
    for _ in range(tries):
        if total >= target:
            break
        position = int(rng.integers(0, len(indices)))
        current = indices[position]
        headroom = DK_SALARY_CAP - total + pool.salary[current]

        # Spending up must not dismantle the stack the entry was built around. A user with
        # spare salary upgrades *within* their stack, or trades a one-off bat; they do not
        # sell the fifth man of their five-stack to buy somebody else's expensive hitter.
        # Unconstrained this broke a stack in 9.8% of entries -- and because a block's core
        # is read off its leader's finished lineup, each break propagated to nineteen more.
        teams = Counter(pool.teams[i] for i in indices if pool.is_hitter[i])
        stacked = None
        if pool.is_hitter[current] and teams.get(pool.teams[current], 0) >= 2:
            stacked = pool.teams[current]

        swaps = []
        for i in (pool.hitters if pool.is_hitter[current] else pool.pitchers):
            i = int(i)
            if i in indices or pool.salary[i] <= pool.salary[current]:
                continue
            if pool.salary[i] > headroom:
                continue
            if pool.slot_sets[current] - pool.slot_sets[i]:
                continue           # must be able to take the same seat
            if stacked is not None and pool.teams[i] != stacked:
                continue           # stay on the stack's team
            swaps.append(i)
        if not swaps:
            continue
        weights = _weights(pool.appeal[swaps])
        replacement = swaps[int(rng.choice(len(swaps), p=weights))]
        trial = list(indices)
        trial[position] = replacement
        if _legal(pool, trial) and _seat_lineup(pool, trial) is not None:
            indices = trial
            total = float(pool.salary[indices].sum())
    return indices


# ---------------------------------------------------------------------------
# The field
# ---------------------------------------------------------------------------

def simulate_field(players, config=None, seed=None, ownership_column="Own%",
                   value_weight=FIELD_VALUE_WEIGHT, quiet=True):
    """Generate a whole opponent field. Returns (entries, FieldPool).

    `entries` is a list of player-index lists, indexing into `pool.players`.

    Multi-entry users are modelled as blocks sharing a core, because that is what they are.
    A 20-max field is not 6,000 independent lineups; it is a few hundred users submitting
    variations on a couple of constructions each, and that is where duplication comes from.
    """
    config = config or ContestConfig()
    pool = FieldPool(players, ownership_column=ownership_column,
                     value_weight=value_weight)
    rng = np.random.default_rng(seed)

    entries = []
    failures = 0
    with profiler.stage("field", entries=config.entries, players=pool.n):
        while len(entries) < config.entries:
            block = 1
            if config.max_entries_per_user > 1:
                # Users cluster at 1 entry or at the maximum, with little in between --
                # visible in every 20-max export in dk_results/.
                block = (config.max_entries_per_user
                         if rng.random() < 0.55 else int(rng.integers(1, 4)))
                block = min(block, config.entries - len(entries))

            core = []
            first = draw_entry(rng, pool, config)
            if first is None:
                failures += 1
                if failures > 200 + config.entries * 0.1:
                    raise FieldError(
                        f"could not build a legal field entry after {failures} attempts; "
                        "the pool is probably too thin or too constrained")
                continue
            entries.append(first)
            if block > 1 and config.block_reuse > 0:
                # The reused core: this user's stack, kept across their block.
                hitters = [i for i in first if pool.is_hitter[i]]
                counts = Counter(pool.teams[i] for i in hitters)
                best_team = max(counts, key=lambda t: counts[t])
                core = [i for i in hitters if pool.teams[i] == best_team]
                if rng.random() > config.block_reuse:
                    core = []
            for _ in range(block - 1):
                if len(entries) >= config.entries:
                    break
                entry = draw_entry(rng, pool, config, core=core or None)
                if entry is not None:
                    entries.append(entry)

    if not quiet:
        print(f"  {len(entries):,} field entries, {failures} failed draws")
    return entries, pool


def field_ownership(entries, pool):
    """Per-player roster rate implied by a simulated field, as a percentage."""
    counts = np.zeros(pool.n)
    for entry in entries:
        counts[list(entry)] += 1
    return counts / max(len(entries), 1) * 100.0


def field_shapes(entries, pool):
    """Stack-shape distribution of a field, in the same form the real exports give."""
    shapes = Counter()
    primary = Counter()
    for entry in entries:
        counts = Counter(pool.teams[i] for i in entry if pool.is_hitter[i])
        sizes = sorted((c for c in counts.values() if c >= 2), reverse=True)
        shapes["-".join(map(str, sizes)) or "none"] += 1
        primary[max(sizes) if sizes else 1] += 1
    total = max(sum(shapes.values()), 1)
    return ({k: v / total for k, v in shapes.items()},
            {k: v / max(sum(primary.values()), 1) for k, v in primary.items()})


def duplication(entries):
    """{lineup key: how many field entries are exactly it}."""
    return Counter(frozenset(entry) for entry in entries)


# ---------------------------------------------------------------------------
# Calibration and validation
# ---------------------------------------------------------------------------

def calibrate(directory=RESULTS_DIR, team_lookup=None, opponent_lookup=None):
    """Recompute the field constants from every contest export on disk.

    Returns a dict shaped like the module constants, so a future season can replace them
    without editing code. A contest export names players but not their teams, so both
    lookups are rebuilt from the cached slates.
    """
    from .results import contest_night, list_contests

    lookup_cache = {}

    shapes, primary, secondary = Counter(), Counter(), defaultdict(Counter)
    conflicts = same_game = examined = 0
    contests = 0

    for contest in list_contests(directory):
        path = contest["path"]
        try:
            lineups, _own, _entries = read_contest_lineups(path)
        except Exception:
            continue
        if len(lineups) < 90:
            continue
        local_team, local_opponent = team_lookup, opponent_lookup
        if local_team is None or local_opponent is None:
            night, slate = contest_night(contest)
            if not night:
                # Without a date, a starter's opponent cannot be identified safely. Do not
                # let an ambiguous contest dilute pitcher-conflict rates toward zero.
                continue
            cache_key = (night, slate)
            if cache_key not in lookup_cache:
                lookup_cache[cache_key] = _slate_lookups(night, slate)
            dated_team, dated_opponent = lookup_cache.get(cache_key, ({}, {}))
            local_team = local_team or dated_team
            local_opponent = local_opponent or dated_opponent
        if not local_team or not local_opponent:
            continue
        contests += 1
        for lineup in lineups:
            hitters = [name for slot, name in lineup if slot != "P"]
            pitchers = [name for slot, name in lineup if slot == "P"]
            counts = Counter(local_team.get(h) for h in hitters)
            counts.pop(None, None)
            if counts and max(counts.values()) > MAX_HITTERS_PER_TEAM:
                # A resolved six-hitter team is a bad name/team join, not a legal DK lineup.
                continue
            sizes = sorted((c for c in counts.values() if c >= 2), reverse=True)
            shapes["-".join(map(str, sizes)) or "none"] += 1
            top = max(sizes) if sizes else 1
            primary[top] += 1
            secondary[top][sizes[1] if len(sizes) > 1 else 0] += 1

            opponent = {p: local_opponent.get(p) for p in pitchers}
            if all(opponent.get(p) for p in pitchers):
                examined += 1
                if any(counts.get(opponent[p], 0) >= 3 for p in pitchers):
                    conflicts += 1
                if len(pitchers) == 2 and local_team.get(pitchers[0]) == opponent[pitchers[1]]:
                    same_game += 1

    def share(counter):
        total = max(sum(counter.values()), 1)
        return {k: round(v / total, 4) for k, v in sorted(counter.items())}

    return {
        "contests": contests,
        "lineups": int(sum(primary.values())),
        "pitcher_match_lineups": examined,
        "primary_size_share": share(primary),
        "secondary_size_share": {k: share(v) for k, v in sorted(secondary.items())},
        "conflict_rate": round(conflicts / max(examined, 1), 4),
        "same_game_pitcher_rate": round(same_game / max(examined, 1), 4),
        "top_shapes": {k: round(v / max(sum(shapes.values()), 1), 4)
                       for k, v in shapes.most_common(15)},
    }


def _team_lookup():
    """{normalized player name: team} from every cached slate."""
    from .evaluate import available_dates
    from .slate import build_slate
    mapping = defaultdict(set)
    for date in available_dates():
        try:
            players, _, _ = build_slate(date, check_schedule=False)
        except Exception:
            continue
        if players is None or players.empty:
            continue
        for name, team in zip(players["Name"], players["Team"]):
            mapping[normalize_name(name)].add(canon_team(team))
    # Only unambiguous names; a player who changed teams mid-window would corrupt the count.
    return {k: sorted(v)[0] for k, v in mapping.items() if len(v) == 1}


def _slate_lookups(night, slate=None):
    """Team and opponent mappings for one dated slate.

    Pitchers face different opponents on different dates, so season-wide name mappings
    silently discard almost every regular starter as ambiguous. Calibration must join each
    contest to the board from the date on which it was played.
    """
    from .naming import latest

    players = pd.DataFrame()
    board_path = latest("slate", str(night), slate, root="dfs_boards")
    if board_path:
        try:
            players = pd.read_csv(
                board_path, encoding="utf-8-sig",
                usecols=lambda column: str(column).strip() in {"Name", "Team", "Opp"})
            players.columns = [str(column).strip() for column in players.columns]
        except Exception:
            players = pd.DataFrame()
    if players.empty or not {"Name", "Team", "Opp"} <= set(players.columns):
        from .naming import label_slates
        from .salaries import list_salary_files

        infos = list_salary_files(date=str(night))
        labels = label_slates(infos, str(night))
        info = next((item for item in infos if labels.get(item["path"]) == slate), None)
        if info is None:
            return {}, {}
        try:
            salary = pd.read_csv(
                info["path"], encoding="utf-8-sig",
                usecols=lambda column: str(column).strip() in
                {"Name", "TeamAbbrev", "Game Info"})
            salary.columns = [str(column).strip() for column in salary.columns]
        except Exception:
            return {}, {}
        if not {"Name", "TeamAbbrev", "Game Info"} <= set(salary.columns):
            return {}, {}
        salary["Team"] = salary["TeamAbbrev"].map(canon_team)

        def salary_opponent(row):
            matchup = str(row["Game Info"]).split(" ", 1)[0].split("@")
            if len(matchup) != 2:
                return ""
            away, home = map(canon_team, matchup)
            return home if row["Team"] == away else away

        salary["Opp"] = salary.apply(salary_opponent, axis=1)
        players = salary
    if players is None or players.empty or not {"Name", "Team", "Opp"} <= set(players.columns):
        return {}, {}
    teams = {normalize_name(name): canon_team(team)
             for name, team in zip(players["Name"], players["Team"])}
    opponents = {normalize_name(name): canon_team(opponent)
                 for name, opponent in zip(players["Name"], players["Opp"])}
    return teams, opponents


def validate(players, config=None, seed=7, directory=RESULTS_DIR, ownership_column="Own%"):
    """Simulate a field and check it against the real ones.

    Three tests, in the order they matter. Ownership is the easy one -- the simulator draws
    from it, so it had better reproduce it. Stack shapes are the real test of the
    construction model. Duplication is the test of the block-reuse model, and it is the one
    that decides whether contest EV means anything.
    """
    config = config or ContestConfig()
    entries, pool = simulate_field(players, config=config, seed=seed,
                                   ownership_column=ownership_column)
    simulated_own = field_ownership(entries, pool)
    shapes, primary = field_shapes(entries, pool)
    dupes = duplication(entries)

    report = {
        "config": config.to_dict(),
        "entries_built": len(entries),
        "ownership": {
            "target_mean": round(float(np.nanmean(pool.own)), 3),
            "simulated_mean": round(float(simulated_own.mean()), 3),
            "mae": round(float(np.nanmean(np.abs(simulated_own - pool.own))), 3),
            "correlation": round(float(np.corrcoef(simulated_own, pool.own)[0, 1]), 3),
            "hitter_total": round(float(simulated_own[pool.is_hitter].sum()), 1),
            "pitcher_total": round(float(simulated_own[~pool.is_hitter].sum()), 1),
        },
        "primary_size_share": {int(k): round(v, 4) for k, v in sorted(primary.items())},
        "primary_size_target": PRIMARY_SIZE_SHARE,
        "top_shapes": dict(sorted(shapes.items(), key=lambda kv: -kv[1])[:10]),
        "duplication": {
            "unique_share": round(sum(1 for v in dupes.values() if v == 1) / max(len(entries), 1), 4),
            "distinct_lineups": len(dupes),
            "max_duplicates": max(dupes.values()) if dupes else 0,
            "mean_duplicates": round(float(np.mean(list(dupes.values()))), 3) if dupes else 0.0,
        },
        "salary": {
            "mean": int(np.mean([pool.salary[list(e)].sum() for e in entries])),
            "share_of_cap": round(float(np.mean(
                [pool.salary[list(e)].sum() for e in entries]) / DK_SALARY_CAP), 4),
        },
    }
    return report, entries, pool


def format_validation(report, real=None):
    own = report["ownership"]
    lines = [
        f"{report['entries_built']:,} simulated entries "
        f"({report['config']['name']}, {report['config']['max_entries_per_user']}-max)",
        "",
        "ownership reproduction",
        f"  MAE {own['mae']:.2f} pts   correlation {own['correlation']:.3f}",
        f"  hitters sum to {own['hitter_total']:.0f} (DK: 800), "
        f"pitchers {own['pitcher_total']:.0f} (DK: 200)",
        "",
        "primary stack size          simulated   real field",
    ]
    for size in sorted(set(report["primary_size_share"]) | set(PRIMARY_SIZE_SHARE)):
        sim = report["primary_size_share"].get(size, 0.0)
        target = PRIMARY_SIZE_SHARE.get(size, 0.0)
        lines.append(f"  {size} hitters from one team    {sim * 100:6.1f}%      {target * 100:6.1f}%")
    lines.append("")
    lines.append("commonest shapes: " + ", ".join(
        f"{k} {v * 100:.1f}%" for k, v in list(report["top_shapes"].items())[:6]))
    dup = report["duplication"]
    lines.append("")
    lines.append(f"duplication: {dup['unique_share'] * 100:.1f}% of entries unique, "
                 f"most-duplicated appears {dup['max_duplicates']}x")
    if real:
        lines.append(f"  real contests in dk_results/: "
                     f"{real['unique_share'] * 100:.1f}% unique, max {real['max_duplicates']}x")
    lines.append(f"salary: mean ${report['salary']['mean']:,} "
                 f"({report['salary']['share_of_cap'] * 100:.1f}% of cap)")
    return "\n".join(lines)


def real_duplication(directory=RESULTS_DIR):
    """Duplication actually observed across the contest exports."""
    from .results import list_contests
    unique, total, biggest = 0, 0, 0
    for contest in list_contests(directory):
        path = contest["path"]
        try:
            lineups, _, _ = read_contest_lineups(path)
        except Exception:
            continue
        if len(lineups) < 90:
            continue
        counts = Counter(frozenset(name for _s, name in l) for l in lineups)
        unique += sum(1 for v in counts.values() if v == 1)
        total += len(lineups)
        biggest = max(biggest, max(counts.values()))
    if not total:
        return None
    return {"unique_share": unique / total, "max_duplicates": biggest, "lineups": total}


def save_report(report, label, extra=None):
    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    path = os.path.join(BENCHMARK_DIR, f"field_{label}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   **(extra or {}), "report": report}, handle, indent=2, default=str)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Simulate the opponent field and check it against real contest exports.")
    parser.add_argument("--date", default=datetime.today().strftime("%Y-%m-%d"))
    parser.add_argument("--slate")
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--stage")
    parser.add_argument("--contest", choices=sorted(PRESETS), default="twenty-max")
    parser.add_argument("--entries", type=int, help="Override the preset's field size.")
    parser.add_argument("--entry-fee", type=float)
    parser.add_argument("--max-entries", type=int)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--calibrate", action="store_true",
                        help="Recompute the field constants from dk_results/ and stop.")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    if args.calibrate:
        stats = calibrate()
        print(json.dumps(stats, indent=2))
        path = save_report(stats, "calibration")
        print(f"\n  -> {path}")
        return

    with profiler.session("field", date=args.date, slate=args.slate, enabled=args.profile):
        if args.snapshot:
            from .snapshot import load_snapshot
            snap = load_snapshot(args.date, args.slate, stage=args.stage)
            players = snap.players
            print(f"from {snap}")
        else:
            from .slate import build_slate
            players, _, meta = build_slate(args.date, slate=args.slate)
            if players is None or players.empty:
                print(f"No slate data for {args.date}.")
                return
            if not meta.get("has_salary"):
                print("[!] no salaries matched; the field needs prices.")
                return

        config = PRESETS[args.contest]
        config = ContestConfig(
            name=config.name,
            entries=args.entries or config.entries,
            entry_fee=args.entry_fee if args.entry_fee is not None else config.entry_fee,
            max_entries_per_user=args.max_entries or config.max_entries_per_user,
            block_reuse=config.block_reuse,
        )
        started = time.perf_counter()
        report, entries, pool = validate(players, config=config, seed=args.seed)
        elapsed = time.perf_counter() - started

    print(f"\nbuilt in {elapsed:.1f}s ({elapsed / max(len(entries), 1) * 1000:.2f} ms/entry)\n")
    print(format_validation(report, real=real_duplication()))
    path = save_report(report, f"{args.date}_{args.slate or 'main'}_{config.name}",
                       extra={"date": args.date, "slate": args.slate,
                              "seed": args.seed, "elapsed_s": round(elapsed, 2)})
    print(f"\n  -> {path}")


def _opponent_lookup():
    """{normalized pitcher name: opposing team} from every cached slate."""
    from .evaluate import available_dates
    from .slate import build_slate
    mapping = defaultdict(set)
    for date in available_dates():
        try:
            players, _, _ = build_slate(date, check_schedule=False)
        except Exception:
            continue
        if players is None or players.empty:
            continue
        for name, opponent in zip(players["Name"], players["Opp"]):
            mapping[normalize_name(name)].add(canon_team(opponent))
    return {k: sorted(v)[0] for k, v in mapping.items() if len(v) == 1}


if __name__ == "__main__":
    main()
