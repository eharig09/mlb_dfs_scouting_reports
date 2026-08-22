# Simulation and portfolio plan

How the optimizer moves from *summing deterministic player objectives* to *selecting a
portfolio against simulated slates, simulated fields, and a contest payout table*.

Read [`current_architecture.md`](current_architecture.md) first; this document assumes
its measurements, especially §2.2 (the stack bonus costs 7.4× the solve time) and §3.3
(review currently rebuilds slates from mutated caches).

---

## 0. The thing being fixed

Today a lineup's score is `Σ ceiling(player)`, plus a hand-tuned bonus for stacking and a
hand-tuned tax for pitcher-versus-own-stack. That objective cannot express the questions a
tournament actually asks:

- What is the chance this lineup finishes first, given that its eight hitters' outcomes are
  **not independent**?
- If it does finish first, **how many other people submitted it**, and what is the prize
  after the split?
- Given nineteen other lineups I am also entering, does this twentieth one **add** anything,
  or does it lose in exactly the same worlds the others do?

`Σ ceiling` answers none of these. The stack bonus is a proxy for the first, duplication is
entirely absent from the second, and `max_overlap` — a count of shared players — is a crude
proxy for the third. Two lineups can share three players and be near-perfectly correlated
(same game stack, same pitcher); two others can share six and diverge sharply.

The replacement is a pipeline that produces, for every candidate lineup, a **vector of
simulated scores** instead of a scalar, and then selects a subset of candidates whose joint
distribution of outcomes maximizes expected profit.

---

## 1. Target pipeline

```
raw data ─► normalized ─► SNAPSHOT ─► features ─► projections ─► ownership
                             │                        │              │
                             │                        ▼              ▼
                             │                 player sim params   field model
                             │                        │              │
                             │                        ▼              ▼
                             │                  slate simulator   field simulator
                             │                   S (sims×players)  F (field lineups)
                             │                        │              │
                             ├─► candidate pool ──────┤              │
                             │   L (players×lineups)  │              │
                             │                        ▼              ▼
                             │                    scores = S @ L   duplication model
                             │                        │              │
                             │                        └──────┬───────┘
                             │                               ▼
                             │                    contest EV per candidate
                             │                               │
                             │                               ▼
                             │                    joint portfolio selection
                             │                               │
                             └───────────────────────────────┴─► reports / upload
```

Every arrow is a pure function of its inputs plus the snapshot id. That is what makes the
whole thing replayable, and it is why the snapshot comes first.

---

## 2. Phase 1 — measurement and reproducibility

**Status: implemented.** See §9.

### Immutable slate snapshots (`dfs/snapshot.py`)

A snapshot is everything known at one moment, frozen so no later refresh can reach back
into it. `.cache/report_data/*.pkl` is mutated in place by `--refresh-lineups`, so today a
review of last Tuesday reads Tuesday's payload *as it stands now*.

Content-addressed: a snapshot is written under
`.cache/snapshots/<date>/<slate>/<stage>/` with a manifest carrying a sha256 of the
projection table, the salary file, code version, model version, and a data-cutoff
timestamp. Stages are `morning`, `t-2h`, `confirmed`, `final`. Re-taking a stage never
overwrites; it writes the next revision and the manifest records the chain.

Storage is Parquet for the tables, JSON for the manifest. A full 15-game slate snapshot is
~200 KB, so a season of four stages a night is well under a gigabyte.

Consumers (`dfs.review`, `dfs.evaluate`, backtests) prefer a snapshot and fall back to
rebuilding, saying loudly which one they used.

### Expanded evaluation (`dfs/evaluate.py`)

Walk-forward only: for a target date, only data strictly before it may inform the
prediction being scored. `run_walkforward(dates)` iterates in time order and never pools
across the boundary. Random splits of player-game rows are not offered as an option,
because on a slate-structured dataset they leak game state between train and test.

Metrics, all computed per segment:

| Family | Metric |
| --- | --- |
| Point | bias, MAE, RMSE, Pearson, Spearman |
| Ranking | top-decile recall, top/bottom quintile spread |
| Distribution | quantile (pinball) loss at 10/25/50/75/90, PIT histogram + KS, CRPS |
| Probability | Brier score for bust (≤3 pts) and ceiling-threshold exceedance, plus reliability curves |

Segments: hitter/pitcher, salary tier (quintiles), batting-order slot, handedness matchup,
slate size, and confirmed-vs-projected lineup status.

CRPS is computed from the simulated distribution when one exists and from the
floor/mean/ceiling triple otherwise, since a three-point summary still supports a usable
approximation.

### Profiling (`dfs/profiling.py`)

A `stage()` context manager writing nested timings to
`docs/benchmarks/profile_<date>_<slate>.json`. Stages match the pipeline arrows above.
Off by default; `--profile` turns it on. Overhead measured at under 1%.

---

## 3. Phase 2 — projection construction

### Opportunity and rate

Hitters already split PA from rates (`SLOT_PA × _team_pa_multiplier` vs the blended rate
line). The work is widening what feeds PA: pinch-hit and platoon-substitution risk,
catcher rest patterns, doubleheader workload, expected starter duration (a 4-inning starter
means more bullpen PA and a different opponent quality mix), and confirmed-vs-projected
status as a variance term rather than only a mean shift.

Pitchers are the weaker half. `_project_innings` sees recent per-start IP and season IP/GS.
It should see recent pitch counts, days of rest, activation status, opener/bullpen-game
designation, expected game competitiveness (a blowout shortens both starters differently),
and pitcher props where available.

### Component outcomes

The projection already computes 1B/2B/3B/HR/BB/HBP/R/RBI/SB internally and only collapses
to points at the end via `dfs/scoring.py`. That is the right shape and it is what the
simulator needs — the simulator draws **components**, not points, and calls the same
scoring function. Missing today: strikeouts (needed for nothing in DK hitter scoring but
needed to make PA outcomes sum correctly), stolen-base *attempts* separately from
successes, and for pitchers an explicit batters-faced distribution.

### Shrinkage and uncertainty

Per-metric shrinkage constants, not one global `REG` table: HR/PA and platoon splits need
far more regression than K% or BB%. Minor-league translations most of all.

Every mean gets a companion uncertainty. Today `Ceiling = Proj + 1.35 × sqrt(f(Proj))` —
a closed form in the mean — so a rookie with 40 PA and a veteran with 400 PA projected the
same get identical bands. That is wrong in the direction that matters: the rookie is the
tournament play precisely because his distribution is wider. Uncertainty becomes an
explicit column and feeds the simulator's per-player dispersion.

---

## 4. Phase 3 — candidate generation

**Status: implemented as `dfs/candidates.py`.** Separate from selection, by design.

Two stages:

1. **Stack cores.** Enumerate (team, size, member set) combinations, scored and pruned.
   Hard stack constraints, not the soft bonus — architecture §2.2 measured the bonus at
   7.4× the solve cost, and a core is an exact statement of the same intent.
2. **Completion.** Fill pitchers and the remaining hitters around each core.

Output is a Parquet candidate pool: one row per lineup with player ids, salary, projection,
ceiling, ownership sum, stack signature, pitcher pair, and game coverage. Selection reads
this file and never re-solves.

Pruning is deliberately gentle. A dominated player is one strictly worse on *every* axis
used downstream (projection, ceiling, salary, ownership) — a low-owned player is never
dominated by a chalk one at the same price, because low ownership is a payoff term. The
failure mode to avoid is pruning away the 2%-owned bat that wins the tournament.

Reproducible: `--seed` fixes the generation stream. Re-running with the same snapshot and
seed produces a byte-identical pool.

### Scale target

At 0.05 s/lineup (hard stacks, no bonus) 1,000 candidates is ~50 s. Acceptable. If it needs
to be faster, the lever is fewer MILP solves rather than a different solver: complete each
core with a greedy + local-search pass and reserve the MILP for the cores themselves.

---

## 5. Phase 4 — correlated slate simulation

**Status: implemented as `dfs/simulate.py`, and calibrated against actual outcomes.**

Two corrections were needed once real results were compared (benchmarks §7): the per-player
dispersion was too narrow (`player_sigma` 0.22 -> 0.40, fitted so simulated SD matches the
observed 6.67 for hitters and 10.26 for pitchers), and the simulator was treating the
projection as truth. Regressing actual on projected gives `actual = 0.872 * proj + 0.578`
for hitters and `0.667 * proj + 4.862` for pitchers -- a one-point projection edge is worth
well under a point -- and that shrinkage is now applied as a per-player mean shift.

### Hierarchy

```
slate     ε_slate      ~ N(0, σ_slate)          shared scoring environment
 └ game   ε_game       ~ N(0, σ_game)           park, weather, umpire, total
    └ team ε_team      ~ N(0, σ_team)           whether this offense showed up
       │  ε_pitcher    ~ N(0, σ_pitcher)        opposing starter's night, entered negatively
       └ player ε_i    ~ N(0, σ_i)              individual
```

A player's latent "performance level" is a weighted sum of the shocks above them plus their
own. Adjacent batting-order slots get an extra shared term, because slots 3-4-5 score in the
same innings far more often than 1 and 8 do.

The latent level maps to a **PA-count multiplier and a rate multiplier**, then components
are drawn (multinomial over PA outcomes; negative-binomial for pitcher batters faced), then
`dfs/scoring.py` converts to points. Drawing components rather than points is what makes the
correlation structure physical: a team's extra runs arrive as extra times through the order,
which is also extra PA for everyone in it.

Negative pitcher/opposing-hitter correlation is not imposed as a correlation coefficient. It
falls out: the pitcher's earned runs are driven by the same team-offense shock that drives
the hitters' production.

### Interface

`simulate_slate(players, n_sims, seed) -> S` where `S` is `(n_sims, n_players)` float32.
50,000 × 200 float32 is 40 MB — fine in memory.

Lineup scoring is `S @ L` with `L` a sparse `(n_players, n_candidates)` 0/1 matrix. For
50,000 sims × 5,000 candidates that is a 50,000×5,000 result, 1 GB in float32, so it is
chunked over candidates and only summary statistics are retained.

### Validation (required before anything depends on it)

Simulated distributions must be checked against history at four levels:

| Level | Check |
| --- | --- |
| Player | marginal mean, SD, P(0 pts), P(≤3), P(≥ ceiling) vs actual by type and salary tier |
| Team | distribution of 5-hitter stack totals vs actual |
| Game | total runs distribution vs actual |
| Lineup | winning-score distribution vs `dk_results/` `Points` column |

The last is the strongest single test available: five contest exports are already on disk
with full score distributions. A simulator whose 99.9th percentile lineup score does not
resemble the actual winning score is not usable for contest EV no matter how good the
player marginals look.

---

## 6. Phase 5 — ownership and field simulation

### Conditional ownership

The current model is a per-player softmax (architecture §4). It cannot say "given that
someone rostered Judge, how likely is Soto too", which is the quantity duplication depends
on entirely.

Replacement: keep the marginal model (it is fitted and it works — MAE 5.23) and add a
**construction model** on top. A field entry is generated as:

1. Draw a primary stack (team, size) from a distribution over stack popularity
2. Draw a secondary stack or pair conditional on the primary
3. Draw two pitchers from a pitcher-pair distribution, conditioned on not opposing the stack
4. Fill remaining slots by marginal ownership, salary-feasibly
5. Reject and redraw if illegal

Calibration target: the simulated field's per-player roster rate must reproduce the marginal
`%Drafted` from `dk_results/`, and its stack-frequency distribution must match what those
same exports imply from their `Lineup` column. Both are measurable today with the five
files on disk; more will accumulate.

Field size, entry fee, and max-entries-per-user change the construction distribution
materially (a 3-max field stacks differently from a 150-max field), so those are model
inputs, not constants.

### Why field simulation rather than a duplication formula

A closed-form duplication estimate from ownership products is a known-bad approximation: it
assumes independence, which is exactly the assumption that fails. Simulating the field
gives duplication, cash lines, and the score distribution to beat from one object.

---

## 6. Phase 6-7 — duplication and contest EV

**Status: implemented as `dfs/contest.py`.** Duplication turned out to need no separate
model: identical lineups score identically, so they tie, and DK's tie rule already splits
the prizes for the tied positions between them. The duplication count is still reported,
because it is the number that tells you a lineup is unplayable before you enter it, but the
money is handled by the mechanism that actually causes it.

The EV *levels* are not calibrated -- see the open item in §9.

Duplication for a candidate is the number of field entries identical to it. From the
simulated field this is a count; from a fitted model it is a prediction with features:
product of log ownership, primary/secondary stack ownership, pitcher-pair ownership, salary
used and left, number of sub-5% players, and contest size. The fitted model exists so
duplication can be estimated for candidates that never appear in a finite simulated field.

Contest EV per candidate, given simulated lineup scores and a simulated field:

```
for each sim:
    rank  = position of candidate score within the field's scores
    prize = payout_table[rank]
    dupes = expected duplicates of this candidate
    prize = prize / (1 + dupes)          # first place is split, not won outright
EV = mean(prize) - entry_fee
```

Reported per candidate: mean score, score SD, P(1st), P(top 0.1%), P(top 1%), P(cash),
expected payout, expected profit, ROI, expected duplicates, P(unique), and
duplication-adjusted EV.

Contest configuration (entry fee, field size, payout curve, max entries per user) is a JSON
file so a user can paste in a real contest's structure. Presets ship for the common shapes.

**Multiple rankings are preserved.** Highest raw EV is one answer; highest P(1st) is
another; highest EV subject to a floor on cash probability is a third. These genuinely
differ and the right one depends on bankroll and intent, so the tool reports all of them
rather than picking.

---

## 7. Phase 8 — joint portfolio selection

**Status: implemented as `dfs/portfolio.py`.** Measured result in benchmarks §9: ranking
twenty lineups by summed ceiling produces a set worth **2.07 effective independent bets**;
joint selection reaches 5.91 while scoring 12 points higher on best-of-set.

Given `k` entries to submit and `m` candidates with simulated score vectors:

Maximize a portfolio objective that rewards
`E[Σ payout]`, `P(at least one wins)`, `P(at least one top-1%)`, and coverage of distinct
game environments and stacks; and penalizes pairwise **simulated-score correlation** (not
shared-player count), player/stack/pitcher-pair over-exposure, repeated salary
constructions, and duplication risk.

This is a submodular-flavoured selection problem. Plan: greedy forward selection with
marginal-gain evaluation directly on the simulated score matrix — `k` is 1 to 150, `m` is a
few thousand, and greedy on a submodular objective carries a `1 - 1/e` guarantee — then a
local swap-improvement pass. A MILP is not the right tool here: the objective is not linear
in the selection.

`P(at least one lineup wins)` is computed straight off the simulation matrix as the fraction
of sims in which any selected lineup ranks first against the field, which is exactly the
quantity `max_overlap` was always trying to approximate.

User exposure overrides remain hard constraints on top, unchanged in meaning from today.

---

## 8. Delivery order and status

| # | Deliverable | Status |
| --- | --- | --- |
| 1 | `docs/current_architecture.md` | done |
| 2 | `docs/simulation_and_portfolio_plan.md` | this file |
| 3 | Pipeline profiling report | done — `docs/benchmarks/` |
| 4 | Immutable slate snapshots | done — `dfs/snapshot.py` |
| 5 | Walk-forward backtesting + calibration | done — `dfs/evaluate.py` |
| 6 | Candidate-lineup export layer | done — `dfs/candidates.py` |
| 7 | Correlated slate simulation prototype | done — `dfs/simulate.py` |
| 8 | Tests and benchmarks | done — `tests/`, `docs/benchmarks.md` |
| 9 | Field simulation | done — `dfs/field.py`, calibrated on 25,671 real lineups |
| 10 | Duplication model | done — falls out of tie-splitting in `dfs/contest.py` |
| 11 | Contest EV | **ranking works, levels do not** — see benchmarks §8 |
| 12 | Joint portfolio optimization | done — `dfs/portfolio.py` |

### The open item

Deliverable 11 was chased to ground; the investigation is `docs/benchmarks.md` §7A. What
came out of it:

**Fixed, each verified against real data.** The simulator's teammate correlation was 2.7x
too strong and its pitcher-versus-opposing correlation half as negative as reality, so
five-stacks simulated 23% too wide. The batting-order adjacency effect the model carried is
not in the data at all (+0.103 adjacent vs +0.101 distant over 2,086 real pairs). A missing
mechanism -- hitters pulled early -- was added, mean-neutral. The per-player projection
shrinkage was reverted after 17,370 real field lineups showed it made lineup-level error
worse, not better. And `APPEAL_POWER` turned out to be unreachable at import, so every
sweep over it had silently returned the same answer.

**The actual cause of the inflated EV**, once the confounds were removed: the model believed
our candidate pool beat the field by +1.3 points when it had really *lost* to the real field
by -5.3. The pool is generated for coverage, so its average member is deliberately mediocre,
and the simulated field is ~5 points soft -- so the model compared a mediocre pool against a
soft field and saw an edge that was not there. One parameter, `FIELD_EDGE_CORRECTION`,
corrects the gap at the point of ranking.

**Where it lands.** P(cash) is now calibrated to within 5% of observed frequency. P(top 1%)
and P(win) still run ~1.5x high, because a single scalar cannot correct the middle and the
tail at once and seven contests will not support a second parameter.

**Also corrected: an earlier claim in this repo.** The note that simulated mean out-ranks
summed ceiling (+0.214 vs +0.157) was measured on one slate and does not replicate. Across
seven contests, within-contest discrimination is +0.177 for raw projection, +0.170 for
summed ceiling, +0.150 for simulated mean -- all within noise. The simulator earns its keep
by producing *distributions*, which duplication, contest EV and portfolio correlation all
require and a point estimate cannot supply. It does not rank players better.

Next, in order of expected value:

1. **More contests with standings.** Everything above rests on seven, and the top-1% figure
   on ten realised events. This is the binding constraint.
2. **Close the field's remaining ~5-point softness at the median.** Concentration and
   value-weighting were both tried and both failed; heterogeneous entrant skill -- a
   minority of sharp entrants rather than one population -- is the next idea.
3. **Thin the candidate pool's upper tail**: realised scores clear the simulated p90 7.7% of
   the time against an expected 10%.

### Compatibility rules held throughout

- No existing CLI command changes behaviour. New capability arrives as new flags or new
  modules.
- `dfs.optimize`, `dfs.upload`, `dfs.lateswap`, `dfs.cli`, `dfs.review` keep their current
  output formats.
- Snapshots and candidate pools are additive files; nothing reads them unless asked.
- No new hard dependency without a benchmark justifying it. Architecture §2.3 is why the
  solver is not being changed.

### Evidence rules

No change ships as an improvement without out-of-sample evidence. For projections that
means walk-forward metrics on held-out dates, not in-sample fit. For the optimizer it means
scored against actual results across multiple nights, not one. Where a change is neutral on
accuracy but faster, that is stated as a speed result and not dressed up as an accuracy one.
