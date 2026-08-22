"""Correlated slate simulation: draw whole nights, not independent players.

**Why independence is the wrong model.** Summing ten players' ceilings assumes their good
outcomes are unrelated. In baseball they are the opposite of unrelated. Four hitters from
one team score in the same innings off the same pitcher; their runs and RBI are literally
the same events counted twice. A pitcher's earned runs *are* the opposing hitters' runs. A
lineup built on that structure has a fatter right tail than the sum of its parts -- which is
why stacking wins tournaments -- and a fatter left tail too, which is why it loses cash
games. `Sigma ceiling` cannot see either.

**The hierarchy.** Every player carries a latent "how did tonight go" level built from
shared shocks:

    slate    one draw per simulation      league-wide scoring environment
    game     one per game                 park, weather, umpire, the game's own total
    team     one per team                 whether this offense showed up
    order    one per adjacent slot block  hitters who bat in the same innings
    player   one per player               individual variation

**Correlation is not imposed, it is inherited.** There is no pitcher-versus-hitter
correlation coefficient anywhere in this module. Instead the opposing team's simulated runs
drive both their hitters' R/RBI and the pitcher's earned runs, so the negative relationship
falls out of the arithmetic at whatever strength the run distribution implies. The same is
true of teammates: team runs are drawn once and then *allocated* across the nine hitters, so
they compete for a shared pool exactly as they do in a real game.

**Components, then scoring.** Nothing here predicts DK points directly. It draws singles,
doubles, walks, innings and earned runs, then calls `dfs.scoring` -- the same function the
projection uses and the same one the review scores against. A scoring-rule change lands in
one place.

**Output.** `simulate_slate` returns `S`, an (n_sims x n_players) float32 array of DK
points aligned to the input frame's row order. Lineup scores are then `S @ L` for a sparse
(n_players x n_lineups) selection matrix -- see `dfs.candidates.CandidatePool.matrix`.

Everything is seeded. The same seed, frame and settings give bit-identical output.
"""

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .profiling import profiler
from .salaries import canon_team
from .scoring import DK_HITTER, DK_PITCHER

BENCHMARK_DIR = os.path.join("docs", "benchmarks")


@dataclass
class SimConfig:
    """Dispersion settings for the hierarchy.

    The sigmas are on a log scale: a shock of s multiplies the affected rate by exp(s). They
    are set from the shape of observed DK outcomes rather than fitted per slate, and they
    are the first thing to tune when simulated distributions do not match history (see
    `validate`).
    """

    # --- shared shocks (log scale) ---
    slate_sigma: float = 0.05      # league-wide night; small, but it moves every lineup together
    game_sigma: float = 0.16       # park/weather/umpire/pace: the single biggest shared term
    # Did this offense show up. FITTED, and far smaller than it looks like it should be:
    # the shared team-run pool already generates most of the teammate correlation on its
    # own, so an explicit team shock on top of it was double-counting. At the old 0.26 the
    # simulator produced a teammate residual correlation of +0.288 against a measured
    # +0.102, and five-stacks came out 1.23x too wide -- which is what inflated every
    # tournament probability. See docs/benchmarks.md.
    team_sigma: float = 0.06
    # Adjacent batting slots sharing innings. Nearly zero, because the effect is not
    # in the data: measured teammate correlation is +0.103 for hitters within two slots of
    # each other and +0.101 for those further apart. The folklore that adjacency matters a
    # great deal is not visible in 2,086 real pairs.
    order_sigma: float = 0.03
    # Individual variation, FITTED against 1,312 finished player-games across ten dates.
    # Raised from 0.22 to absorb the variance removed when `team_sigma` was cut: the total
    # marginal spread has to stay right while the *shared* part of it shrinks.
    #
    # This term is carrying two things at once and it is worth being explicit about it: real
    # game-to-game variation, *and* our own error about the player's true rate. The
    # simulator treats the projection as the truth, so without the second part it would say
    # a lineup optimised on those projections is far better than it is.
    player_sigma: float = 0.48

    # How many adjacent batting slots share an `order` shock. Three is a lineup "turn":
    # 3-4-5 bat together far more often than 1 and 8 do.
    order_block: int = 3

    # --- team runs ---
    # Runs per team-game are overdispersed relative to Poisson: they cluster in innings
    # (one big frame, then nothing). A negative binomial with this dispersion reproduces the
    # observed spread; 1.0 would be Poisson. Fitted alongside player_sigma.
    run_dispersion: float = 4.0
    # Calibration targets these hit, measured over ten dates of finished games:
    #   hitter SD 6.67 (target 6.78)   pitcher SD 9.70 (target 9.89)
    #   five-stack SD 19.8 (target 19.1)
    #   teammate residual r +0.110 (target +0.102)
    #   pitcher-vs-opposing r -0.284 (target -0.281)
    # The one that still misses is P(hitter scores zero): 0.19 against an observed 0.26.
    # Reaching it needs a short-game rate near 0.34, which pushes five-stack SD to 20.7 and
    # re-inflates the tail -- the wrong trade, since stack spread is what the tournament
    # probabilities are actually sensitive to.

    # --- hitter rates ---
    # How hard the latent level moves each event class. Power swings hardest and walks
    # barely move -- the same ordering the projection's ELASTICITY uses, for the same
    # reason: a hot night is mostly a slugging night.
    hr_elasticity: float = 1.7
    hit_elasticity: float = 1.0
    bb_elasticity: float = 0.35
    # Plate appearances follow the team's night: a lineup that scores bats around again.
    pa_elasticity: float = 0.45

    # Chance a starting hitter's night is cut short -- pinch-hit for, defensive replacement,
    # ejection, injury, or a blowout emptying the bench -- and the share of his plate
    # appearances he keeps when it happens.
    #
    # Without this the simulator blanked hitters 16.7% of the time against an observed
    # 26.4%, and its marginal SD came in 7% light. Both are the same hole: the model had no
    # way to produce the one-plate-appearance game, which is where most real zeroes live.
    # Cranking `player_sigma` instead widened the distribution symmetrically, which fixed
    # neither the blank rate nor the shape.
    short_game_rate: float = 0.18
    short_game_share: float = 0.35

    # --- pitcher ---
    ip_sigma: float = 0.20         # length of outing, before the runs-allowed hook
    # How much getting hit shortens the night. Every run above expectation costs this
    # fraction of an inning, which is what turns a bad start into a *short* bad start --
    # the two-sided disaster that makes pitcher variance so wide.
    ip_run_penalty: float = 0.22
    k_dispersion: float = 1.0
    # Share of a team's runs charged to the starter as earned, at his share of the innings.
    # Below 1.0 because unearned runs and bullpen damage are not his.
    er_share: float = 0.86

    # A pitcher whose team scores more is likelier to get the decision. The logistic slope
    # is on the run differential.
    win_slope: float = 0.42

    # --- projection calibration ---
    # The simulator's rates come from the projection, which means it quietly assumes the
    # projection is the truth. It is not. Regressing actual on projected over 2,471 finished
    # hitter-games and 277 starts:
    #
    #     hitters   actual = 0.872 * proj + 0.578
    #     pitchers  actual = 0.667 * proj + 4.862
    #
    # A one-point projection edge is worth 0.87 actual points for a hitter and 0.67 for a
    # pitcher. Without this the simulator overstates the spread between good and bad players
    # by a third at pitcher, and any lineup optimised on those projections looks far better
    # than it is -- the winner's curse, unmodelled.
    #
    # **OFF by default, and that is a measured reversal.** The regression is real, but
    # applying it per player is the wrong transform, because the players a lineup contains
    # are *selected* on high projection and the linear fit over-shrinks exactly there.
    #
    # Checked directly against 17,370 real field lineups from the contest exports, summing
    # each entry's ten players:
    #
    #     raw projection sum   -1.2 mean error   (mean abs 4.5)
    #     calibrated sum       -5.9 mean error   (mean abs 7.5)
    #
    # The raw sum is nearly unbiased at lineup level; the calibration makes it 4.7 points
    # worse. Turning it off halves the simulated field's error against real contests
    # (mean abs 7.5 -> 3.8 across median/p80/p99/winner) and improves pitcher SD as well.
    #
    # The per-player regression stays here because it is a true description of E[actual|proj]
    # in the population, and because a future non-linear version -- the top pitcher bucket
    # needs about a third of the shrinkage the bottom one does -- may well earn its place.
    calibrate: bool = False
    calibration: tuple = (("H", 0.872, 0.578), ("P", 0.667, 4.862))

    def to_dict(self):
        return asdict(self)


DEFAULT_CONFIG = SimConfig()

# Expected-event columns the simulator needs. A frame without them was produced before
# MODEL_VERSION 2 and cannot be simulated from -- better to say so than to invent rates.
HITTER_EVENTS = ["E_PA", "E_1B", "E_2B", "E_3B", "E_HR", "E_BB", "E_HBP", "E_R", "E_RBI", "E_SB"]
PITCHER_EVENTS = ["E_BF", "E_IP", "E_K", "E_BB", "E_H", "E_HBP", "E_ER", "E_W"]


class SimulationError(Exception):
    pass


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def _require_events(players):
    missing = [c for c in HITTER_EVENTS if c not in players.columns]
    if missing:
        raise SimulationError(
            "the slate frame has no expected-event columns "
            f"(missing {', '.join(missing[:4])}...). Rebuild the board with a current "
            "dfs.projections -- simulation draws components, not points."
        )


def _grouping(players):
    """Integer group ids for slate / game / team / order-block, one per player row."""
    games = players.get("Game", pd.Series("", index=players.index)).fillna("").astype(str)
    teams = players["Team"].map(canon_team).fillna("").astype(str)
    game_ids, game_index = pd.factorize(games)
    team_ids, team_index = pd.factorize(teams)
    return game_ids, list(game_index), team_ids, list(team_index)


def _order_blocks(players, team_ids, block):
    """One id per (team, batting-order block). Pitchers get their own singleton block."""
    slot = pd.to_numeric(players.get("Slot"), errors="coerce").fillna(0).astype(int)
    keys = []
    for i, (team, spot, kind) in enumerate(zip(team_ids, slot, players["Type"])):
        keys.append((team, (spot - 1) // block) if kind == "H" and spot >= 1 else ("P", i))
    return pd.factorize(pd.Series(keys, index=players.index))[0]


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_slate(players, n_sims=10000, seed=None, config=DEFAULT_CONFIG,
                   return_components=False):
    """Simulate `n_sims` correlated slates. Returns (n_sims x n_players) DK points.

    Rows align to `players` positionally, so the caller can index straight into the frame
    it passed in.

    With `return_components=True` also returns a dict of the raw draws (team runs, PA,
    hits, innings, ...), which is what `validate` and the tests inspect -- a points total
    can look right while the events underneath are nonsense.
    """
    _require_events(players)
    players = players.reset_index(drop=True)
    n = len(players)
    if n == 0:
        raise SimulationError("empty player frame")

    rng = np.random.default_rng(seed)
    is_hitter = (players["Type"] == "H").to_numpy()
    is_pitcher = ~is_hitter

    game_ids, game_index, team_ids, team_index = _grouping(players)
    block_ids = _order_blocks(players, team_ids, config.order_block)
    n_games, n_teams, n_blocks = len(game_index), len(team_index), block_ids.max() + 1

    with profiler.stage("simulate", sims=n_sims, players=n):
        # --- shared shocks -------------------------------------------------
        slate_shock = rng.normal(0, config.slate_sigma, size=(n_sims, 1))
        game_shock = rng.normal(0, config.game_sigma, size=(n_sims, n_games))
        team_shock = rng.normal(0, config.team_sigma, size=(n_sims, n_teams))
        block_shock = rng.normal(0, config.order_sigma, size=(n_sims, n_blocks))
        player_shock = rng.normal(0, config.player_sigma, size=(n_sims, n))

        # A player's latent level. Every term above them is shared with somebody, which is
        # the entire source of correlation in this model.
        latent = (slate_shock + game_shock[:, game_ids] + team_shock[:, team_ids]
                  + block_shock[:, block_ids] + player_shock)
        # Total variance of the latent level, needed to keep the exponential transforms
        # mean-preserving. exp(X) for X ~ N(0, s^2) has mean exp(s^2/2), not 1, so a rate
        # multiplied by exp(e*X) comes out e^2*s^2/2 too high -- with hr_elasticity 1.7 that
        # inflated home runs by 25% and every hitter's simulated mean with it. Subtracting
        # the term makes each multiplier average exactly 1.0, so the simulator reproduces
        # the projection's mean instead of a systematically hotter slate.
        latent_var = (config.slate_sigma ** 2 + config.game_sigma ** 2
                      + config.team_sigma ** 2 + config.order_sigma ** 2
                      + config.player_sigma ** 2)
        team_var = (config.slate_sigma ** 2 + config.game_sigma ** 2 + config.team_sigma ** 2)
        # The team's own level, for quantities that belong to the team rather than a player.
        team_latent = slate_shock + team_shock
        for t in range(n_teams):
            # Fold in the game each team plays, so two teams in a high-total game both rise.
            members = np.flatnonzero(team_ids == t)
            if len(members):
                team_latent[:, t] = team_latent[:, t] + game_shock[:, game_ids[members[0]]]

        points = np.zeros((n_sims, n), dtype=np.float32)
        components = {}

        # --- team runs -----------------------------------------------------
        # Drawn once per team per simulation and then allocated to hitters, so teammates
        # compete for one pool exactly as they do in a real game. This is what makes a stack
        # correlated, and it is also what the opposing pitcher's earned runs come from.
        expected_runs = np.zeros(n_teams)
        for t in range(n_teams):
            members = np.flatnonzero((team_ids == t) & is_hitter)
            expected_runs[t] = players.loc[members, "E_R"].to_numpy().sum() if len(members) else 0.0
        run_mean = np.maximum(expected_runs, 0.05) * np.exp(team_latent - team_var / 2)
        team_runs = _negative_binomial(rng, run_mean, config.run_dispersion)
        components["team_runs"] = team_runs

        # --- hitters -------------------------------------------------------
        if is_hitter.any():
            hitter_idx = np.flatnonzero(is_hitter)
            points, hit_parts = _simulate_hitters(
                rng, players, hitter_idx, team_ids, latent, team_latent, team_runs,
                points, config, latent_var, team_var)
            components.update(hit_parts)

        # --- pitchers ------------------------------------------------------
        if is_pitcher.any():
            pitcher_idx = np.flatnonzero(is_pitcher)
            points, pitch_parts = _simulate_pitchers(
                rng, players, pitcher_idx, team_index, team_ids, latent, team_runs,
                points, config)
            components.update(pitch_parts)

        # --- projection calibration ----------------------------------------
        # Last, on points, because that is the scale the relationship was measured on.
        if config.calibrate:
            projected = pd.to_numeric(players["Proj"], errors="coerce").fillna(0).to_numpy()
            for kind, slope, intercept in config.calibration:
                mask = (players["Type"] == kind).to_numpy()
                if not mask.any():
                    continue
                shift = (slope - 1.0) * projected[mask] + intercept
                adjusted = points[:, mask] + shift[None, :]
                if kind == "H":
                    # A hitter cannot score below zero, and a blank stays a blank: the
                    # intercept says where the average lands, not that every hitter who did
                    # nothing now collects half a point.
                    adjusted = np.where(points[:, mask] <= 0.0, 0.0,
                                        np.maximum(adjusted, 0.0))
                points[:, mask] = adjusted.astype(np.float32)

    return (points, components) if return_components else points


def _negative_binomial(rng, mean, dispersion):
    """Negative-binomial counts with the given mean and variance = mean * dispersion.

    Runs are not Poisson. They arrive in clusters -- a five-run third and nothing else -- so
    the variance runs well above the mean, and a Poisson draw would produce far too many
    "everyone scored exactly four" nights and far too few blowouts. Dispersion 1.0 collapses
    back to Poisson.
    """
    mean = np.maximum(np.asarray(mean, dtype=float), 1e-6)
    if dispersion <= 1.0 + 1e-9:
        return rng.poisson(mean)
    # variance = mean + mean^2 / r  ->  r = mean / (dispersion - 1)
    r = mean / (dispersion - 1.0)
    p = r / (r + mean)
    return rng.negative_binomial(np.maximum(r, 1e-6), np.clip(p, 1e-9, 1 - 1e-9))


def _simulate_hitters(rng, players, idx, team_ids, latent, team_latent, team_runs,
                      points, config, latent_var, team_var):
    """Draw plate appearances, outcomes, and a share of the team's runs and RBI."""
    n_sims = latent.shape[0]
    sub = players.loc[idx]
    base_pa = sub["E_PA"].to_numpy(dtype=float)

    # Plate appearances follow the team's night: an offense that turns the order over gives
    # everyone another trip, and the bottom of the order gains most.
    #
    # Drawn as floor + Bernoulli(fraction) rather than Poisson. A Poisson around 4.2 has an
    # SD of 2 and hands out zero-PA and nine-PA games regularly; real plate appearances are
    # nearly deterministic given how many times the order turns over, which is a *team*
    # quantity and is already carried by the shock in pa_mean. This keeps the mean exact and
    # puts the variance where it belongs.
    # Short games are mean-neutral by construction. `SLOT_PA` in the projection was fitted
    # as the PA accrued by the player who *starts* in each slot, which already averages over
    # the nights he was pinch-hit for -- so adding explicit early exits without raising the
    # full-game rate would double-count them and quietly lower every hitter's projection.
    # Dividing by the expected retention keeps E[PA] exactly where the projection put it and
    # makes the mechanism a pure change of shape.
    retention = (1.0 - config.short_game_rate
                 + config.short_game_rate * config.short_game_share)
    pa_mean = np.clip(
        base_pa[None, :] / max(retention, 1e-6)
        * np.exp(config.pa_elasticity * team_latent[:, team_ids[idx]]
                 - (config.pa_elasticity ** 2) * team_var / 2),
        0.0, 8.0)
    whole = np.floor(pa_mean)
    pa = (whole + (rng.random(pa_mean.shape) < (pa_mean - whole))).astype(np.int64)

    # Nights that end early: pinch-hit for, defensive replacement, ejection, injury. Applied
    # after the normal draw so the typical game is untouched and only the left tail moves.
    if config.short_game_rate > 0:
        cut = rng.random(pa.shape) < config.short_game_rate
        if cut.any():
            shortened = rng.binomial(np.maximum(pa, 0), config.short_game_share)
            pa = np.where(cut, shortened, pa)

    # Per-PA outcome probabilities from the projection's expected counts.
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = {key: np.where(base_pa > 0, sub[f"E_{key}"].to_numpy(dtype=float) / base_pa, 0.0)
                for key in ("1B", "2B", "3B", "HR", "BB", "HBP")}

    level = latent[:, idx]

    def multiplier(elasticity):
        """exp(e * level), de-biased so it averages exactly 1.0. See `latent_var` above."""
        return np.exp(elasticity * level - (elasticity ** 2) * latent_var / 2)

    scale = {
        "1B": multiplier(config.hit_elasticity),
        "2B": multiplier(config.hit_elasticity),
        "3B": multiplier(config.hit_elasticity),
        "HR": multiplier(config.hr_elasticity),
        "BB": multiplier(config.bb_elasticity),
        "HBP": np.ones_like(level),
    }
    probability = {k: np.clip(rate[k][None, :] * scale[k], 0.0, 1.0) for k in rate}
    total = sum(probability.values())
    # Renormalize only where the scaled rates overflow a plate appearance.
    over = total > 0.98
    if over.any():
        for key in probability:
            probability[key] = np.where(over, probability[key] * 0.98 / np.maximum(total, 1e-9),
                                        probability[key])

    # Sequential binomial thinning over plate appearances. Each PA resolves to exactly one
    # of {HR, 3B, 2B, 1B, BB, HBP, out}, so the conditional probability of the next class is
    # its rate divided by the mass NOT already claimed -- including the out mass. An earlier
    # version divided by the remaining *event* classes only, which omitted the ~70% chance
    # of an out and inflated every hitter's simulated mean from 6.8 to 16.9 points.
    remaining = pa.copy()
    events = {}
    consumed = np.zeros_like(probability["HR"])
    for key in ("HR", "3B", "2B", "1B", "BB", "HBP"):
        share = np.clip(probability[key] / np.maximum(1.0 - consumed, 1e-9), 0.0, 1.0)
        drawn = rng.binomial(np.maximum(remaining, 0), share)
        events[key] = drawn
        remaining = remaining - drawn
        consumed = consumed + probability[key]

    # Runs and RBI are allocated from the team's pool rather than drawn per player. A run
    # scored by the 3-hitter is a run not scored by anyone else, and the RBI that drove it
    # belongs to a teammate -- that shared budget is the mechanical heart of stack
    # correlation, and drawing them independently would destroy it.
    runs = np.zeros_like(pa)
    rbi = np.zeros_like(pa)
    for team in np.unique(team_ids[idx]):
        members = np.flatnonzero(team_ids[idx] == team)
        if not len(members):
            continue
        pool = team_runs[:, team]
        run_weight = sub.iloc[members]["E_R"].to_numpy(dtype=float)
        rbi_weight = sub.iloc[members]["E_RBI"].to_numpy(dtype=float)
        # Weighted by the player's own night: a hitter who homered is likelier to have
        # scored, which keeps R correlated with his own line and not just his team's.
        # No de-biasing needed -- these are shares within a fixed pool, so a common factor
        # cancels and only the relative differences matter.
        #
        # Scaled by how much of the game he was actually in: a hitter pulled after two
        # innings cannot score in the eighth, so his share of the team's runs has to fall
        # with his plate appearances rather than staying at his full-game weight.
        share_of_game = np.divide(pa[:, members],
                                  np.maximum(base_pa[members][None, :], 1e-9))
        boost = np.exp(0.5 * latent[:, idx[members]]) * np.clip(share_of_game, 0.0, 2.0)
        runs[:, members] = _allocate(rng, pool, run_weight[None, :] * boost)
        rbi_pool = rng.binomial(pool, 0.95)
        rbi[:, members] = _allocate(rng, rbi_pool, rbi_weight[None, :] * boost)

    steals = rng.poisson(np.maximum(sub["E_SB"].to_numpy(dtype=float), 0.0)[None, :]
                         * np.ones((n_sims, 1)))

    score = (DK_HITTER["1B"] * events["1B"] + DK_HITTER["2B"] * events["2B"]
             + DK_HITTER["3B"] * events["3B"] + DK_HITTER["HR"] * events["HR"]
             + DK_HITTER["BB"] * events["BB"] + DK_HITTER["HBP"] * events["HBP"]
             + DK_HITTER["R"] * runs + DK_HITTER["RBI"] * rbi + DK_HITTER["SB"] * steals)
    points[:, idx] = score.astype(np.float32)
    return points, {"hitter_pa": pa, "hitter_events": events, "hitter_runs": runs,
                    "hitter_rbi": rbi, "hitter_index": idx}


def _allocate(rng, pool, weights):
    """Split an integer pool across columns in proportion to weights, per simulation.

    Multinomial per row would be the textbook call, but numpy has no vectorised
    multinomial over per-row probability *matrices*. Sequential binomial thinning gives the
    same distribution: take the first column's share of what is left, then the next.
    """
    n_sims, k = weights.shape
    out = np.zeros((n_sims, k), dtype=np.int64)
    remaining = np.asarray(pool, dtype=np.int64).copy()
    tail = weights.sum(axis=1)
    for j in range(k):
        if j == k - 1:
            out[:, j] = remaining
            break
        share = np.divide(weights[:, j], np.maximum(tail, 1e-9))
        drawn = rng.binomial(np.maximum(remaining, 0), np.clip(share, 0.0, 1.0))
        out[:, j] = drawn
        remaining = remaining - drawn
        tail = tail - weights[:, j]
    return out


def _simulate_pitchers(rng, players, idx, team_index, team_ids, latent, team_runs,
                       points, config):
    """Draw the outing: how long it lasted, and what the opposing offense did to it."""
    n_sims = latent.shape[0]
    sub = players.loc[idx]
    position = {team: i for i, team in enumerate(team_index)}

    expected_ip = sub["E_IP"].to_numpy(dtype=float)
    k_rate = sub.get("E_KRATE")
    k_rate = (k_rate.to_numpy(dtype=float) if k_rate is not None
              else np.where(sub["E_BF"] > 0, sub["E_K"] / sub["E_BF"], 0.20))
    bf_per_ip = np.where(expected_ip > 0, sub["E_BF"].to_numpy(dtype=float) / expected_ip, 4.3)
    bb_rate = np.where(sub["E_BF"] > 0, sub["E_BB"].to_numpy(dtype=float) / sub["E_BF"], 0.08)
    h_rate = np.where(sub["E_BF"] > 0, sub["E_H"].to_numpy(dtype=float) / sub["E_BF"], 0.22)
    hbp_rate = np.where(sub["E_BF"] > 0, sub["E_HBP"].to_numpy(dtype=float) / sub["E_BF"], 0.01)
    expected_er = sub["E_ER"].to_numpy(dtype=float)

    # The opposing offense's simulated runs, per pitcher. This single lookup is where the
    # pitcher/opposing-hitter anticorrelation comes from -- no coefficient required.
    opponent = sub["Opp"].map(canon_team).map(position)
    known = opponent.notna().to_numpy()
    opponent_runs = np.zeros((n_sims, len(idx)))
    if known.any():
        columns = opponent[known].astype(int).to_numpy()
        opponent_runs[:, known] = team_runs[:, columns]

    own = sub["Team"].map(canon_team).map(position)
    own_known = own.notna().to_numpy()
    own_runs = np.zeros((n_sims, len(idx)))
    if own_known.any():
        own_runs[:, own_known] = team_runs[:, own[own_known].astype(int).to_numpy()]

    # Expected opposing runs, to measure how badly tonight went against the projection.
    baseline = np.maximum(np.where(known, expected_er / max(config.er_share, 1e-6), 4.4), 0.5)
    run_excess = (opponent_runs - baseline[None, :]) / baseline[None, :]

    # Length of the outing: its own noise, then shortened by getting hit. A starter who is
    # being scored on comes out, which is why a bad pitcher night is short AND bad -- the
    # two losses compound, and modelling IP independently of runs would miss it entirely.
    innings = expected_ip[None, :] * np.exp(rng.normal(0, config.ip_sigma, (n_sims, len(idx))))
    innings = innings * (1.0 - config.ip_run_penalty * np.clip(run_excess, -0.6, 2.0))
    innings = np.clip(innings, 0.0, 9.0)
    outs = np.clip(np.round(innings * 3), 0, 27)
    innings = outs / 3.0

    batters = np.maximum(np.round(innings * bf_per_ip[None, :]), 0).astype(np.int64)
    strikeouts = rng.binomial(batters, np.clip(k_rate, 0.0, 0.6)[None, :] * np.ones((n_sims, 1)))
    walks = rng.binomial(batters, np.clip(bb_rate, 0.0, 0.3)[None, :] * np.ones((n_sims, 1)))
    hbp = rng.binomial(batters, np.clip(hbp_rate, 0.0, 0.05)[None, :] * np.ones((n_sims, 1)))
    hits = rng.binomial(np.maximum(batters - strikeouts - walks - hbp, 0),
                        np.clip(h_rate * 1.35, 0.0, 0.8)[None, :] * np.ones((n_sims, 1)))

    # Earned runs are the opposing team's runs, at the starter's share of the game. Drawn
    # from the same pool the opposing hitters just divided up, which is the whole point.
    share = np.clip(innings / 9.0, 0.0, 1.0)
    earned = rng.binomial(np.maximum(opponent_runs.astype(np.int64), 0),
                          np.clip(share * config.er_share, 0.0, 1.0))

    # A win needs the team ahead and the starter to have gone five.
    differential = own_runs - opponent_runs
    win_probability = 1.0 / (1.0 + np.exp(-config.win_slope * differential))
    qualified = innings >= 5.0
    wins = (rng.random((n_sims, len(idx))) < win_probability * qualified).astype(np.int64)

    score = (DK_PITCHER["IP"] * innings + DK_PITCHER["K"] * strikeouts
             + DK_PITCHER["W"] * wins + DK_PITCHER["ER"] * earned
             + DK_PITCHER["H"] * hits + DK_PITCHER["BB"] * walks
             + DK_PITCHER["HBP"] * hbp)
    points[:, idx] = score.astype(np.float32)
    return points, {"pitcher_ip": innings, "pitcher_k": strikeouts, "pitcher_er": earned,
                    "pitcher_index": idx}


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def player_summary(players, simulations):
    """Simulated marginals per player, next to what the projection claimed."""
    frame = players.reset_index(drop=True)
    return pd.DataFrame({
        "Name": frame["Name"],
        "Team": frame["Team"],
        "Type": frame["Type"],
        "Proj": pd.to_numeric(frame["Proj"], errors="coerce"),
        "Sim Mean": simulations.mean(axis=0).round(2),
        "Sim SD": simulations.std(axis=0).round(2),
        "P(0)": (simulations <= 0).mean(axis=0).round(3),
        "P(<=3)": (simulations <= 3).mean(axis=0).round(3),
        "Sim p90": np.percentile(simulations, 90, axis=0).round(2),
        "Ceiling": pd.to_numeric(frame["Ceiling"], errors="coerce"),
        "Sim p99": np.percentile(simulations, 99, axis=0).round(2),
    })


def correlation_report(players, simulations, min_pairs=3):
    """Mean simulated correlation for the relationships the model is supposed to preserve.

    This is the acceptance test for the hierarchy. Every row has a sign the physics demands;
    a simulator that gets the marginals right and these wrong is worse than useless for
    portfolio work, because it will happily recommend a lineup that cannot happen.
    """
    frame = players.reset_index(drop=True)
    teams = frame["Team"].map(canon_team).to_numpy()
    games = frame.get("Game", pd.Series("", index=frame.index)).astype(str).to_numpy()
    slots = pd.to_numeric(frame.get("Slot"), errors="coerce").fillna(0).to_numpy()
    is_hitter = (frame["Type"] == "H").to_numpy()
    opponents = frame["Opp"].map(canon_team).to_numpy()

    valid = simulations.std(axis=0) > 1e-9
    corr = np.corrcoef(simulations[:, valid], rowvar=False)
    index = {j: k for k, j in enumerate(np.flatnonzero(valid))}

    def mean_of(pairs):
        values = [corr[index[a], index[b]] for a, b in pairs if a in index and b in index]
        return (round(float(np.mean(values)), 4), len(values)) if len(values) >= min_pairs \
            else (float("nan"), len(values))

    n = len(frame)
    same_team, adjacent, distant, opposing, cross_game, same_game_teams = [], [], [], [], [], []
    for a in range(n):
        for b in range(a + 1, n):
            if is_hitter[a] and is_hitter[b]:
                if teams[a] == teams[b]:
                    same_team.append((a, b))
                    (adjacent if abs(slots[a] - slots[b]) <= 2 else distant).append((a, b))
                elif games[a] == games[b]:
                    same_game_teams.append((a, b))
                else:
                    cross_game.append((a, b))
            elif is_hitter[a] != is_hitter[b]:
                pitcher, hitter = (a, b) if not is_hitter[a] else (b, a)
                if opponents[pitcher] == teams[hitter]:
                    opposing.append((pitcher, hitter))

    rows = [
        ("teammate hitters", "positive", *mean_of(same_team)),
        ("  adjacent in the order (<=2 apart)", "most positive", *mean_of(adjacent)),
        ("  distant in the order", "less positive", *mean_of(distant)),
        ("opposing hitters, same game", "slightly positive", *mean_of(same_game_teams)),
        ("pitcher vs the hitters he faces", "NEGATIVE", *mean_of(opposing)),
        ("hitters in different games", "~zero", *mean_of(cross_game)),
    ]
    return pd.DataFrame(rows, columns=["relationship", "expected", "mean r", "pairs"])


def validate(players, simulations, actuals=None):
    """Do the simulated distributions look like baseball? Returns a report dict.

    Marginals are checked against the model's own claims (mean, ceiling) and, when actual
    results are supplied, against what happened. Correlations are checked for sign and
    ordering. A simulator that passes the marginal checks and fails the correlation ones is
    the dangerous case: every player looks right and every *lineup* is wrong.
    """
    summary = player_summary(players, simulations)
    correlations = correlation_report(players, simulations)

    hitters = summary[summary["Type"] == "H"]
    pitchers = summary[summary["Type"] == "P"]
    report = {
        "n_sims": int(simulations.shape[0]),
        "n_players": int(simulations.shape[1]),
        "marginals": {},
        "correlations": correlations.to_dict("records"),
    }
    for label, group in (("hitters", hitters), ("pitchers", pitchers)):
        if group.empty:
            continue
        report["marginals"][label] = {
            "n": int(len(group)),
            "proj_mean": round(float(group["Proj"].mean()), 3),
            "sim_mean": round(float(group["Sim Mean"].mean()), 3),
            "mean_gap": round(float((group["Sim Mean"] - group["Proj"]).mean()), 3),
            "sim_sd_mean": round(float(group["Sim SD"].mean()), 3),
            "p_zero": round(float(group["P(0)"].mean()), 4),
            "p_bust": round(float(group["P(<=3)"].mean()), 4),
            "ceiling_vs_sim_p90": round(float((group["Sim p90"] - group["Ceiling"]).mean()), 3),
        }

    if actuals is not None and len(actuals):
        # The only check that matters in the end: does a simulated night look like a night?
        observed = pd.to_numeric(pd.Series(actuals), errors="coerce").dropna()
        report["observed"] = {
            "n": int(len(observed)),
            "mean": round(float(observed.mean()), 3),
            "sd": round(float(observed.std(ddof=1)), 3),
            "p_zero": round(float((observed <= 0).mean()), 4),
            "p_bust": round(float((observed <= 3).mean()), 4),
            "p90": round(float(observed.quantile(0.90)), 3),
        }
    return report


def format_validation(report):
    lines = [f"{report['n_sims']:,} sims x {report['n_players']} players", ""]
    for label, values in report.get("marginals", {}).items():
        lines.append(f"{label.upper()}  n={values['n']}")
        lines.append(f"  projected mean {values['proj_mean']:6.2f}   "
                     f"simulated mean {values['sim_mean']:6.2f}   "
                     f"gap {values['mean_gap']:+.2f}")
        lines.append(f"  simulated SD   {values['sim_sd_mean']:6.2f}   "
                     f"P(0) {values['p_zero']:.3f}   P(<=3) {values['p_bust']:.3f}")
        lines.append(f"  sim p90 minus stated ceiling: {values['ceiling_vs_sim_p90']:+.2f}")
        lines.append("")
    if "observed" in report:
        o = report["observed"]
        lines.append(f"OBSERVED (actual results, n={o['n']})")
        lines.append(f"  mean {o['mean']:.2f}  sd {o['sd']:.2f}  "
                     f"P(0) {o['p_zero']:.3f}  P(<=3) {o['p_bust']:.3f}  p90 {o['p90']:.2f}")
        lines.append("")
    lines.append("correlation structure:")
    for row in report["correlations"]:
        value = row["mean r"]
        shown = "  n/a" if value != value else f"{value:+.3f}"
        lines.append(f"  {row['relationship']:<38} {shown}   "
                     f"(expected {row['expected']}, {row['pairs']} pairs)")
    return "\n".join(lines)


def save_validation(report, path=None, label="simulate", extra=None):
    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    path = path or os.path.join(BENCHMARK_DIR, f"simulate_{label}.json")
    payload = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               **(extra or {}), "report": report}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    import time

    from .slate import build_slate

    parser = argparse.ArgumentParser(
        description="Simulate correlated slate outcomes and validate the distributions.")
    parser.add_argument("--date", default=datetime.today().strftime("%Y-%m-%d"))
    parser.add_argument("--slate")
    parser.add_argument("--snapshot", action="store_true",
                        help="Simulate the frozen snapshot instead of the live cache.")
    parser.add_argument("--stage")
    parser.add_argument("--sims", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--actuals", action="store_true",
                        help="Also pull final box scores and compare the simulated "
                             "distribution against what happened.")
    parser.add_argument("--candidates", metavar="DIR",
                        help="Score a saved candidate pool against the simulation.")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--label", default=None)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    with profiler.session("simulate", date=args.date, slate=args.slate,
                          enabled=args.profile, sims=args.sims):
        if args.snapshot:
            from .snapshot import load_snapshot
            snap = load_snapshot(args.date, args.slate, stage=args.stage)
            players = snap.players
            print(f"from {snap}")
        else:
            players, _, meta = build_slate(args.date, slate=args.slate)
        if players is None or players.empty:
            print(f"No slate data for {args.date}.")
            return

        started = time.perf_counter()
        simulations = simulate_slate(players, n_sims=args.sims, seed=args.seed)
        elapsed = time.perf_counter() - started

    print(f"\nsimulated {args.sims:,} slates in {elapsed:.2f}s "
          f"({elapsed / args.sims * 1e6:.0f} us/sim, "
          f"{simulations.nbytes / 1e6:.0f} MB)")

    actuals = None
    if args.actuals:
        from .review import attach_actuals, slate_actuals
        scored = attach_actuals(players, slate_actuals(args.date))
        actuals = scored["Actual"].dropna().tolist()

    report = validate(players, simulations, actuals=actuals)
    print()
    print(format_validation(report))

    path = save_validation(report, label=args.label or f"{args.date}_{args.slate or 'main'}",
                           extra={"date": args.date, "slate": args.slate,
                                  "seed": args.seed, "elapsed_s": round(elapsed, 3),
                                  "config": DEFAULT_CONFIG.to_dict()})
    print(f"\n  -> {path}")

    if args.candidates:
        from .candidates import CandidatePool
        pool = CandidatePool.load(args.candidates)
        # Candidate rows index into the candidate pool's own player frame, so the simulation
        # has to be run against that frame rather than the slate's -- the two orderings are
        # not the same once unpriced or filtered players are dropped.
        sims = simulate_slate(pool.pool, n_sims=args.sims, seed=args.seed)
        scores = pool.score(sims)
        frame = pool.lineups.copy()
        frame["sim_mean"] = scores.mean(axis=0).round(2)
        frame["sim_sd"] = scores.std(axis=0).round(2)
        frame["sim_p90"] = np.percentile(scores, 90, axis=0).round(1)
        frame["sim_p99"] = np.percentile(scores, 99, axis=0).round(1)
        print(f"\n=== {len(frame)} candidates scored on {args.sims:,} simulated slates ===")
        print("  ranked by simulated 99th percentile -- the score that wins a tournament,")
        print("  which is not the same ranking as mean or as summed ceiling.\n")
        columns = ["lineup", "proj", "ceiling", "own_sum", "stack_shape", "primary_stack",
                   "sim_mean", "sim_sd", "sim_p90", "sim_p99"]
        print(frame.nlargest(args.top, "sim_p99")[columns].to_string(index=False))
        rank_gap = (frame["ceiling"].rank(ascending=False)
                    - frame["sim_p99"].rank(ascending=False)).abs().mean()
        print(f"\n  mean rank disagreement between summed ceiling and simulated p99: "
              f"{rank_gap:.1f} places (of {len(frame)})")

    if args.profile:
        from .profiling import format_report
        print("\n=== pipeline profile ===")
        print(format_report(profiler.last_report))


if __name__ == "__main__":
    main()
