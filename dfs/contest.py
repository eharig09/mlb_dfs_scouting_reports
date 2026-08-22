"""Contest payouts, duplication, and expected value per candidate lineup.

**What this answers.** A candidate's simulated score distribution says how well it will do.
It does not say what it will *pay*, and those are different questions in three ways:

1. Payouts are a step function of finishing rank, not of score. Going from 140 to 150 points
   can be worth nothing or worth the tournament, depending on where the field lands.
2. First place is not won, it is *split*. Identical lineups tie, and DK divides the prizes
   for the tied positions equally between them. A lineup that wins outright pays ten times
   what the same lineup pays when nineteen other people submitted it.
3. What you are beating is the field, and the field's score distribution is itself random
   and correlated with yours -- you and 3,000 other entries all rostered the same chalk
   stack, so on the night it fails you all fail together.

So EV is computed against `dfs.field`'s simulated opponents on the same simulated slates as
the candidate, never against a static score threshold.

**Duplication needs no separate model here.** Duplicate lineups score identically by
construction, so they tie, so the tie-splitting rule already divides the prize between them.
The duplication *count* is reported because it is diagnostic -- it is the number that tells
you a lineup is unplayable before you enter it -- but the money is handled by the mechanism
that actually causes it.

**Memory.** Scoring 5,000 candidates against 6,000 field entries over 20,000 simulations is
6x10^8 numbers if done naively. The work is chunked over simulations and only the rank
statistics survive each chunk, so peak memory stays in the tens of megabytes.
"""

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .profiling import profiler

BENCHMARK_DIR = os.path.join("docs", "benchmarks")

# Simulations processed at once. Bigger is faster and uses more memory; 512 x 6,000 field
# entries is ~12 MB per chunk, which is the right side of every tradeoff here.
CHUNK = 512

# Points subtracted from every candidate before it is ranked against the simulated field.
#
# **This is an empirical correction for a measured bias, not a modelling choice.** Across
# seven contests whose real standings are on disk, the simulator says our candidate pool
# beats the field by +1.3 points; it actually *lost* to the real field by -5.3. The edge is
# overstated by 6.6 points, and that single gap is what inflated every tournament
# probability (cash 1.5x, top-1% 3.4x, win 4.4x over observed frequencies).
#
# Two things cause it and neither is a bug:
#   * the candidate pool is built for coverage -- it round-robins all 20 teams and jitters
#     the objective -- so its average member is deliberately mediocre;
#   * the simulated field is built from ownership and comes out ~5 points soft at the
#     median against real contests, so the bar it sets is too low.
#
# Correcting the *gap* rather than patching the output probabilities keeps the ranks, the
# tie-splitting and the payout arithmetic mutually consistent. Set to 0.0 to see the
# uncorrected model. Re-derive with the harness described in docs/benchmarks.md.
#
# 9.0, not the 6.6 the edge measurement implies, because 9.0 is what actually calibrates the
# probabilities. The two agreeing to within a third -- one derived from lineup scores, the
# other fitted to observed finishing frequencies -- is the reason to trust either:
#
#   correction   P(cash) ratio   P(top 1%) ratio   P(win) ratio     (1.00 = calibrated)
#          0.0            1.56              3.61           4.63
#          6.6            1.09              1.93           2.16
#          9.0            0.95              1.53           1.63
#         12.0            0.79              1.13           1.13
#
# A single scalar cannot fix both ends: the tail needs more correction than the middle,
# because the candidate pool's upper tail is also slightly too fat. 9.0 makes the cash line
# essentially exact and halves the tail error; 12.0 fixes the tail at the cost of being
# systematically pessimistic about cashing. The residual 1.5x on top-1% is documented rather
# than tuned away, because seven contests will not support a second parameter.
FIELD_EDGE_CORRECTION = 9.0


class ContestError(Exception):
    pass


# ---------------------------------------------------------------------------
# Payout structures
# ---------------------------------------------------------------------------

@dataclass
class Payout:
    """Prize by finishing rank, held as (last rank in tier, prize per entry) steps.

    DK publishes payouts exactly this way -- "1st $10,000, 2nd $5,000, 3rd-5th $1,500" --
    so this stores what the contest page shows rather than a fitted curve.
    """

    name: str
    entry_fee: float
    field_size: int
    tiers: tuple = ()          # ((last_rank, prize), ...) ascending by rank

    def prizes(self):
        """Dense array where prizes[r - 1] is the prize for finishing r-th."""
        out = np.zeros(self.field_size, dtype=float)
        start = 1
        for last_rank, prize in self.tiers:
            last = min(int(last_rank), self.field_size)
            if last >= start:
                out[start - 1:last] = float(prize)
            start = last + 1
            if start > self.field_size:
                break
        return out

    @property
    def total_prizes(self):
        return float(self.prizes().sum())

    @property
    def rake(self):
        """Share of entry fees the house keeps. A sanity check on a hand-entered structure."""
        gross = self.entry_fee * self.field_size
        return 1.0 - self.total_prizes / gross if gross else 0.0

    @property
    def paid_places(self):
        return int((self.prizes() > 0).sum())

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_json(cls, path):
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(name=data.get("name", "custom"),
                   entry_fee=float(data["entry_fee"]),
                   field_size=int(data["field_size"]),
                   tiers=tuple((int(a), float(b)) for a, b in data["tiers"]))


def gpp_payout(field_size, entry_fee, rake=0.15, name="GPP", top_share=0.20,
               paid_share=0.23):
    """A DK-shaped tournament curve: top-heavy, roughly a quarter of the field paid.

    Generated rather than typed because a real structure has 40+ tiers and nobody will enter
    them by hand for a what-if. Pass a real one via `Payout.from_json` when it matters --
    the exact curve moves EV a lot, and this is a stand-in, not a substitute.

    `top_share` is first prize as a fraction of the whole pool; `paid_share` is the fraction
    of the field that cashes. Both are set from DK's published mid-size GPPs.
    """
    pool = entry_fee * field_size * (1.0 - rake)
    paid = max(1, int(round(field_size * paid_share)))

    # Prize weight decays as a power of rank, normalised so first place lands on top_share.
    ranks = np.arange(1, paid + 1, dtype=float)
    weights = ranks ** -1.35
    weights = weights / weights.sum()
    if weights[0] < top_share:
        # Steepen until first place is heavy enough; a flat curve is not a tournament.
        for exponent in np.arange(1.4, 3.01, 0.05):
            weights = ranks ** -exponent
            weights = weights / weights.sum()
            if weights[0] >= top_share:
                break
    prizes = weights * pool

    # Collapse into tiers, rounding to something a contest page would actually print.
    tiers = []
    current = _round_prize(prizes[0])
    for rank in range(2, paid + 1):
        value = _round_prize(prizes[rank - 1])
        if value != current:
            tiers.append((rank - 1, current))
            current = value
    tiers.append((paid, current))
    return Payout(name=name, entry_fee=entry_fee, field_size=field_size,
                  tiers=tuple(tiers))


def _round_prize(value):
    if value >= 1000:
        return round(value, -2)
    if value >= 100:
        return round(value, -1)
    return round(value, 2)


def double_up(field_size, entry_fee, rake=0.10, name="double-up"):
    """Flat payout to the top half -- the cash-game structure.

    Included because it inverts every recommendation a GPP curve makes: here the only thing
    that matters is clearing the line, so variance is a cost rather than the product.
    """
    pool = entry_fee * field_size * (1.0 - rake)
    paid = field_size // 2
    return Payout(name=name, entry_fee=entry_fee, field_size=field_size,
                  tiers=((paid, _round_prize(pool / max(paid, 1))),))


PAYOUT_PRESETS = {
    "single-entry": lambda n, fee: gpp_payout(n or 300, fee or 10.0, name="single-entry"),
    "three-max": lambda n, fee: gpp_payout(n or 1500, fee or 10.0, name="three-max"),
    "twenty-max": lambda n, fee: gpp_payout(n or 6000, fee or 5.0, name="twenty-max"),
    "milly-maker": lambda n, fee: gpp_payout(n or 100000, fee or 20.0, top_share=0.10,
                                             name="milly-maker"),
    "double-up": lambda n, fee: double_up(n or 1000, fee or 10.0),
}


# ---------------------------------------------------------------------------
# Scoring candidates against a field
# ---------------------------------------------------------------------------

def _selection_matrix(entries, n_players):
    """(n_players x n_entries) dense 0/1. Dense on purpose: this is the hot operand."""
    matrix = np.zeros((n_players, len(entries)), dtype=np.float32)
    for column, entry in enumerate(entries):
        matrix[list(entry), column] = 1.0
    return matrix


def duplicate_counts(candidate_players, field_entries, scale=1.0):
    """How many field entries are exactly each candidate.

    `scale` extrapolates a simulated field smaller than the real contest: simulating 6,000
    of a 100,000-entry field and multiplying by 16.7 is far cheaper than simulating all of
    them, and duplication scales linearly in field size for a fixed construction model.
    """
    from collections import Counter
    field = Counter(frozenset(entry) for entry in field_entries)
    return np.array([field.get(frozenset(players), 0) * scale
                     for players in candidate_players], dtype=float)


def evaluate(candidate_pool, simulations, field_entries, payout, field_scale=1.0,
             chunk=CHUNK, cash_line_fraction=0.20,
             edge_correction=FIELD_EDGE_CORRECTION):
    """Contest metrics for every candidate. Returns a DataFrame, one row per candidate.

    `edge_correction` is subtracted from every candidate score before ranking; see
    FIELD_EDGE_CORRECTION for why it exists and how it was measured. Pass 0.0 to disable.

    `simulations` is (n_sims x n_players) from `dfs.simulate`, aligned to
    `candidate_pool.pool`. `field_entries` are player-index lists into the same frame, so
    the candidates and the field are scored on the *same* simulated nights -- which is the
    whole point. Scoring a candidate against a field drawn from different slates would
    delete the correlation that makes chalk dangerous.
    """
    simulations = np.asarray(simulations, dtype=np.float32)
    n_sims, n_players = simulations.shape
    candidates = candidate_pool.lineups
    n_candidates = len(candidates)
    if n_candidates == 0:
        raise ContestError("no candidates to evaluate")
    if not field_entries:
        raise ContestError("no field entries; simulate a field first")

    candidate_matrix = np.asarray(candidate_pool.matrix().todense(), dtype=np.float32)
    field_matrix = _selection_matrix(field_entries, n_players)
    prizes = payout.prizes()
    n_field = len(field_entries)

    # The simulated field stands in for a contest of `payout.field_size` entries.
    scale = field_scale if field_scale != 1.0 else payout.field_size / max(n_field, 1)

    totals = np.zeros(n_candidates)          # summed payout across sims
    wins = np.zeros(n_candidates)
    top01 = np.zeros(n_candidates)
    top1 = np.zeros(n_candidates)
    cashes = np.zeros(n_candidates)
    score_sum = np.zeros(n_candidates)
    score_sq = np.zeros(n_candidates)
    rank_sum = np.zeros(n_candidates)

    dupes = duplicate_counts(candidate_pool.lineups["players"], field_entries, scale)

    with profiler.stage("contest", sims=n_sims, candidates=n_candidates, field=n_field):
        for start in range(0, n_sims, chunk):
            stop = min(start + chunk, n_sims)
            block = simulations[start:stop]
            candidate_scores = block @ candidate_matrix          # (chunk x n_candidates)
            # The measured edge correction. Applied to the candidates rather than to the
            # field so that `sim_mean` still reports the honest simulated score and only the
            # comparison against opponents is adjusted.
            ranked_scores = candidate_scores - edge_correction
            field_scores = block @ field_matrix                  # (chunk x n_field)
            field_sorted = np.sort(field_scores, axis=1)

            score_sum += candidate_scores.sum(axis=0)
            score_sq += (candidate_scores.astype(np.float64) ** 2).sum(axis=0)

            for row in range(stop - start):
                sorted_row = field_sorted[row]
                mine = ranked_scores[row]
                # Field entries strictly worse, and tied, at this candidate's score.
                worse = np.searchsorted(sorted_row, mine, side="left")
                not_better = np.searchsorted(sorted_row, mine, side="right")
                beaten = worse * scale
                tied = (not_better - worse) * scale
                # Rank among the real contest: everyone strictly better, plus one.
                better = (n_field * scale) - beaten - tied
                rank = better + 1.0
                rank_sum += rank

                # Ties share the prizes for the positions they occupy -- DK's rule, and the
                # mechanism by which duplication is actually paid for.
                #
                # `tied` already contains this candidate's duplicates: an identical lineup
                # scores identically, so it lands in the tie count by construction. Adding
                # `dupes` on top would charge for the same copies twice. `dupes` stays a
                # reported diagnostic and does not enter the money.
                sharers = np.maximum(tied + 1.0, 1.0)
                totals += _shared_prize(prizes, rank, sharers)

                wins += rank <= 1.0
                top01 += rank <= max(1.0, payout.field_size * 0.001)
                top1 += rank <= max(1.0, payout.field_size * 0.01)
                cashes += rank <= max(1.0, payout.field_size * cash_line_fraction)

    mean_score = score_sum / n_sims
    variance = np.maximum(score_sq / n_sims - mean_score ** 2, 0.0)
    expected_payout = totals / n_sims

    result = candidates.copy()
    result["sim_mean"] = np.round(mean_score, 2)
    result["sim_sd"] = np.round(np.sqrt(variance), 2)
    result["mean_rank"] = np.round(rank_sum / n_sims, 1)
    result["p_win"] = np.round(wins / n_sims, 6)
    result["p_top01"] = np.round(top01 / n_sims, 5)
    result["p_top1"] = np.round(top1 / n_sims, 4)
    result["p_cash"] = np.round(cashes / n_sims, 4)
    result["exp_dupes"] = np.round(dupes, 2)
    result["p_unique"] = np.round(np.exp(-dupes), 4)
    result["exp_payout"] = np.round(expected_payout, 4)
    result["exp_profit"] = np.round(expected_payout - payout.entry_fee, 4)
    result["roi"] = np.round(
        (expected_payout - payout.entry_fee) / max(payout.entry_fee, 1e-9), 4)
    return result


def _shared_prize(prizes, rank, sharers):
    """Mean prize across the positions a tied group occupies.

    A group of `k` entries tied for rank `r` takes positions r .. r+k-1 and each receives the
    average of those prizes. Treating a tie as "everyone gets the rank-r prize" would value a
    20-times-duplicated winning lineup at twenty times what it pays.
    """
    n = len(prizes)
    start = np.clip(np.floor(rank).astype(np.int64) - 1, 0, n)
    span = np.clip(np.ceil(sharers).astype(np.int64), 1, n)
    end = np.minimum(start + span, n)
    # Cumulative sums make the group average a two-index lookup rather than a loop.
    # Positions past the last paid place contribute zero and still count as sharers, which
    # is correct: a tie straddling the cash line pays its members the average including the
    # zeros, not the prize for the last paying position.
    cumulative = _cumulative(prizes)
    return (cumulative[end] - cumulative[start]) / np.maximum(sharers, 1.0)


_CUMULATIVE_CACHE = {}


def _cumulative(prizes):
    """Cached prefix sums; `evaluate` asks for these once per simulated night."""
    key = id(prizes)
    hit = _CUMULATIVE_CACHE.get(key)
    if hit is None or len(hit) != len(prizes) + 1:
        hit = np.concatenate([[0.0], np.cumsum(prizes)])
        _CUMULATIVE_CACHE.clear()
        _CUMULATIVE_CACHE[key] = hit
    return hit


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

# Printed with every report, because the numbers above it read like precise money and are
# not. Measured on 2026-08-01 main, 1,000 candidates against their real results:
#
#   spearman(sim_mean,   actual) = +0.214      the ranking works
#   spearman(exp_payout, actual) = +0.127      and survives the payout transform
#   actual score by expected-payout decile: 77.4 (worst) rising to 90.5 (best), monotone
#
# But over those same deciles the claimed expected payout runs from $3.10 to $42.30 -- a
# 14x spread produced by a 13-point spread in actual score on a distribution whose SD is 26.
# The cause is compounding: the candidate pool's simulated mean sits ~5 points above what
# it actually scored, and the simulated field's median sits ~4 points below the real one.
# A 9-point error in the candidate-minus-field gap is modest in the middle of the
# distribution and enormous in a payout dominated by first prize.
CALIBRATION_WARNING = (
    "  [i] Probabilities carry an empirical correction fitted on 7 contests with known\n"
    "      standings (FIELD_EDGE_CORRECTION). After it: P(cash) is calibrated to within 5%,\n"
    "      P(top 1%) and P(win) still run about 1.5x high. Treat EV as indicative and the\n"
    "      ordering as the product. See docs/benchmarks.md for the measurements."
)

RANKINGS = {
    "roi": ("roi", "highest expected ROI"),
    "profit": ("exp_profit", "highest expected profit"),
    "win": ("p_win", "highest chance of winning outright"),
    "top1": ("p_top1", "highest chance of a top-1% finish"),
    "cash": ("p_cash", "highest chance of cashing"),
    "mean": ("sim_mean", "highest mean simulated score"),
    "ceiling": ("ceiling", "highest summed ceiling (the old objective)"),
}


def format_summary(result, payout, top=10, by="exp_profit"):
    lines = [
        f"{payout.name}: {payout.field_size:,} entries, ${payout.entry_fee:,.2f} entry, "
        f"${payout.total_prizes:,.0f} in prizes",
        f"  {payout.paid_places:,} places paid ({payout.paid_places / payout.field_size * 100:.0f}%), "
        f"first ${payout.prizes()[0]:,.0f}, rake {payout.rake * 100:.0f}%",
        "",
        f"{len(result):,} candidates ranked by {by}:",
    ]
    columns = [c for c in ("lineup", "proj", "ceiling", "own_sum", "stack_shape",
                           "primary_stack", "sim_mean", "sim_sd", "exp_dupes",
                           "p_win", "p_top1", "p_cash", "exp_payout", "roi")
               if c in result.columns]
    lines.append(result.nlargest(top, by)[columns].to_string(index=False))

    positive = (result["exp_profit"] > 0).sum()
    lines.append("")
    lines.append(CALIBRATION_WARNING)
    lines.append("")
    lines.append(f"  {positive:,} of {len(result):,} candidates have positive expected "
                 f"profit ({positive / len(result) * 100:.0f}%)")
    lines.append(f"  best ROI {result['roi'].max() * 100:+.1f}%, "
                 f"median {result['roi'].median() * 100:+.1f}%, "
                 f"worst {result['roi'].min() * 100:+.1f}%")
    lines.append(f"  duplication: mean {result['exp_dupes'].mean():.1f} expected copies, "
                 f"{(result['exp_dupes'] < 1).mean() * 100:.0f}% of candidates likely unique")

    # The comparison that says whether any of this changed a decision.
    if "ceiling" in result.columns:
        by_ceiling = result.nlargest(1, "ceiling").iloc[0]
        by_profit = result.nlargest(1, "exp_profit").iloc[0]
        lines.append("")
        lines.append("  the old objective vs this one:")
        lines.append(f"    highest summed ceiling  -> lineup {by_ceiling['lineup']}, "
                     f"ROI {by_ceiling['roi'] * 100:+.1f}%, "
                     f"{by_ceiling['exp_dupes']:.1f} expected copies")
        lines.append(f"    highest expected profit -> lineup {by_profit['lineup']}, "
                     f"ROI {by_profit['roi'] * 100:+.1f}%, "
                     f"{by_profit['exp_dupes']:.1f} expected copies")
    return "\n".join(lines)


def save_report(result, payout, label, extra=None):
    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    path = os.path.join(BENCHMARK_DIR, f"contest_{label}.json")
    summary = {
        "payout": payout.to_dict(),
        "candidates": int(len(result)),
        "positive_ev": int((result["exp_profit"] > 0).sum()),
        "roi": {"best": float(result["roi"].max()),
                "median": float(result["roi"].median()),
                "worst": float(result["roi"].min())},
        "duplication": {"mean": float(result["exp_dupes"].mean()),
                        "max": float(result["exp_dupes"].max()),
                        "share_unique": float((result["exp_dupes"] < 1).mean())},
        "best_by": {name: int(result.nlargest(1, column).iloc[0]["lineup"])
                    for name, (column, _label) in
                    (("profit", ("exp_profit", "")), ("win", ("p_win", "")),
                     ("top1", ("p_top1", "")), ("cash", ("p_cash", "")),
                     ("ceiling", ("ceiling", "")))
                    if column in result.columns},
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   **(extra or {}), "summary": summary}, handle, indent=2, default=str)
    return path


# ---------------------------------------------------------------------------
# Shared pipeline
# ---------------------------------------------------------------------------

def _cap_pool_ownership(pool, max_ownership, quiet=False):
    """Drop candidate lineups whose summed projected ownership exceeds the cap.

    Matches the meaning the constraint has inside the solver: a bound on the lineup's total
    Own%, not a per-player limit. Dropping every lineup would leave nothing to select from,
    so that case keeps the pool and says why rather than failing several stages later with
    an unrelated-looking error.
    """
    from .candidates import CandidatePool

    lineups = pool.lineups
    if "own_sum" not in lineups.columns:
        if not quiet:
            print("  [i] --max-ownership ignored: these lineups carry no ownership column.")
        return pool
    own = pd.to_numeric(lineups["own_sum"], errors="coerce")
    keep = own.isna() | (own <= float(max_ownership))
    dropped = int((~keep).sum())
    if not keep.any():
        print(f"  [!] --max-ownership {max_ownership:g} would drop all {len(lineups)} "
              f"lineups (lowest is {own.min():.1f}); ignoring it.")
        return pool
    if dropped and not quiet:
        print(f"  [i] --max-ownership {max_ownership:g}: dropped {dropped} of "
              f"{len(lineups)} lineups, {int(keep.sum())} remain.")
    return CandidatePool(lineups[keep.to_numpy()].reset_index(drop=True), pool.pool,
                         {**pool.meta, "max_ownership": float(max_ownership)})


def build_pipeline(date, slate=None, snapshot=False, stage=None, n_candidates=800,
                   sims=10000, field_entries=None, contest="twenty-max", entry_fee=None,
                   field_size=None, payout_json=None, seed=7, objective="Ceiling",
                   max_ownership=None, from_optimizer=None, allow_started=False,
                   quiet=False):
    """slate -> candidates -> simulations -> field -> payout, ready to evaluate.

    Everything downstream needs the same four objects built against the *same* player frame,
    and getting that wrong is silent: candidates indexed into one frame and simulations into
    another produce plausible numbers that mean nothing. This function is the single place
    that guarantees the alignment.
    """
    from .candidates import from_optimizer as load_optimizer_lineups, generate
    from .field import PRESETS as FIELD_PRESETS, ContestConfig, simulate_field
    from .simulate import simulate_slate
    from .slate import drop_started

    if snapshot:
        from .snapshot import load_snapshot
        snap = load_snapshot(date, slate, stage=stage)
        players, meta = snap.players, snap.meta()
        if not quiet:
            print(f"from {snap}")
    else:
        from .slate import build_slate
        players, _, meta = build_slate(date, slate=slate)
        if players is None or players.empty:
            raise ContestError(f"no slate data for {date}")
        if not meta.get("has_salary"):
            raise ContestError("no salaries matched; contest EV needs prices")

    # Players whose game has already begun cannot be drafted, so they must not reach the
    # candidate pool or the simulated field. This lived only in `dfs.optimize` until it was
    # found missing here -- a portfolio built after first pitch was silently including
    # players who could not score.
    players, started_notes = drop_started(players, meta, date, allow=allow_started)
    if not quiet:
        for note in started_notes:
            print(note)

    if from_optimizer is not None:
        # The label the slate actually resolved to, not the one the caller typed. `build_slate`
        # identifies the DK export by contents and reports it in `slate_label`, and the
        # optimizer files its lineups under that same label -- so a run that omitted --slate
        # would otherwise go looking for `lineups_unknown.csv` and report "no lineups found"
        # about a file sitting right there.
        resolved = slate or meta.get("slate_label")
        pool = load_optimizer_lineups(
            players, path=(from_optimizer if isinstance(from_optimizer, str) else None),
            date=date, slate=resolved)
        if not quiet:
            print(f"  {len(pool)} lineups from {pool.meta['source_path']}")
        # `generate` applies --max-ownership as a solver constraint. Lineups read back from
        # the optimizer were built without it, so it has to be applied here as a filter or
        # the flag silently does nothing on the path the help text calls recommended.
        if max_ownership is not None:
            pool = _cap_pool_ownership(pool, max_ownership, quiet=quiet)
        if not quiet:
            print(f"  {pool.summary()}")
    else:
        pool = generate(players, n_candidates=n_candidates, objective=objective, seed=seed,
                        max_ownership=max_ownership)
        if not quiet:
            print(f"  {pool.summary()}")

    # Simulations are run against the candidate pool's own player frame, because that is
    # what the lineup matrix indexes into.
    simulations = simulate_slate(pool.pool, n_sims=sims, seed=seed)

    base = FIELD_PRESETS[contest]
    field_config = ContestConfig(
        name=base.name,
        entries=field_entries or min(base.entries, 6000),
        entry_fee=entry_fee if entry_fee is not None else base.entry_fee,
        max_entries_per_user=base.max_entries_per_user,
        block_reuse=base.block_reuse,
    )
    entries, field_pool = simulate_field(pool.pool, config=field_config, seed=seed)
    if not quiet:
        print(f"  {len(entries):,} field entries simulated")

    if payout_json:
        payout = Payout.from_json(payout_json)
    else:
        payout = PAYOUT_PRESETS[contest](field_size or base.entries,
                                         field_config.entry_fee)
    return {"players": players, "meta": meta, "candidates": pool,
            "simulations": simulations, "field": entries, "field_pool": field_pool,
            "payout": payout, "field_config": field_config,
            "source_path": pool.meta.get("source_path")}


def main():
    import argparse
    import time
    from datetime import datetime as _dt

    parser = argparse.ArgumentParser(
        description="Score candidate lineups against a simulated field and a payout curve.")
    parser.add_argument("--date", default=_dt.today().strftime("%Y-%m-%d"))
    parser.add_argument("--slate")
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--stage")
    parser.add_argument("--contest", choices=sorted(PAYOUT_PRESETS), default="twenty-max")
    parser.add_argument("--entry-fee", type=float)
    parser.add_argument("--field-size", type=int, help="Real contest size, for the payout.")
    parser.add_argument("--field-entries", type=int, default=4000,
                        help="How many opponents to actually simulate (scaled up to the "
                             "real field size for duplication).")
    parser.add_argument("--payout-json", help="A real payout structure to use instead.")
    parser.add_argument("--candidates", type=int, default=800)
    parser.add_argument("--sims", type=int, default=10000)
    parser.add_argument("--max-ownership", type=float)
    parser.add_argument("--from-optimizer", nargs="?", const=True, metavar="PATH",
                        help="Score the lineups dfs.optimize already generated instead of "
                             "generating new candidates. Bare flag finds the night's "
                             "newest lineups CSV.")
    parser.add_argument("--allow-started", action="store_true",
                        help="Keep players whose game has begun. Analysis only.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--by", default="exp_profit",
                        help="Column to rank the printed table by.")
    parser.add_argument("--out", help="Write the scored candidates to a CSV.")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    with profiler.session("contest", date=args.date, slate=args.slate,
                          enabled=args.profile, sims=args.sims):
        started = time.perf_counter()
        parts = build_pipeline(
            args.date, slate=args.slate, snapshot=args.snapshot, stage=args.stage,
            n_candidates=args.candidates, sims=args.sims,
            field_entries=args.field_entries, contest=args.contest,
            entry_fee=args.entry_fee, field_size=args.field_size,
            payout_json=args.payout_json, seed=args.seed,
            max_ownership=args.max_ownership, from_optimizer=args.from_optimizer,
            allow_started=args.allow_started)
        result = evaluate(parts["candidates"], parts["simulations"], parts["field"],
                          parts["payout"])
        elapsed = time.perf_counter() - started

    print(f"\n{format_summary(result, parts['payout'], top=args.top, by=args.by)}")
    print(f"\n  {elapsed:.1f}s total")
    label = args.slate or parts["meta"].get("slate_label") or "main"
    path = save_report(result, parts["payout"],
                       f"{args.date}_{label}_{args.contest}",
                       extra={"date": args.date, "slate": args.slate, "seed": args.seed,
                              "sims": args.sims, "field_entries": len(parts["field"])})
    print(f"  -> {path}")
    if args.out:
        result.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"  -> {args.out}")

    if args.profile:
        from .profiling import format_report
        print("\n=== pipeline profile ===")
        print(format_report(profiler.last_report))


if __name__ == "__main__":
    main()
