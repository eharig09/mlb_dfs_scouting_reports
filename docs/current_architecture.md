# Current architecture

A map of what exists today, measured rather than assumed. §1-§7 were written *before* any
change, as a baseline; **§8 describes the pipeline as it now stands** and is the section to
read first if you want to know how the thing works today.

Scope: the `dfs/` package (7,351 lines across 22 modules) plus the parts of
`scouting_report.py` (13,264 lines) that feed it. Measurements are from
**2026-08-01 main** (15 cached games, 198-player priced pool) on this machine, Python
3.14.4 / numpy 2.4.6 / pandas 3.0.3 / scipy 1.17.1.

---

## 1. Data flow

```
                    MLB StatsAPI ─┐
                        pybaseball ├─► scouting_report.py ──► .cache/report_data/
                    Open-Meteo ───┤      (~8 min / game)      <date>_<away>_<home>[_gN].pkl
                    Savant ───────┘                                    │
                                                                       │  pickle
    DK ──► dfs_daily_files/DKSalaries_<date>_<label>.csv                │
             │                                                         ▼
             │                                              dfs.slate.build_slate
             ├──► dfs.adopt   (rename strays by contents)             │
             ├──► dfs.salaries (identify slate by contents)           │
             └──────────────────────────────────────────────────────► │
                                                                       ▼
                            dfs.projections.project_game  (per game, per player)
                                                                       │
                                       dfs.ownership.estimate_ownership│
                                       dfs.results.match_contest ──────┤ (real %Drafted
                                                                       │  when on disk)
                                                                       ▼
                                            players_df, stacks_df, meta
                                        ┌──────────────┼──────────────┐
                                        ▼              ▼              ▼
                                 dfs.board      dfs.optimizer    dfs.pool
                                 (markdown)     (scipy MILP)   (editable CSV)
                                        │              │
                                        │              ▼
                                        │        dfs.exposure ──► exposure_<slate>.csv
                                        │        dfs.upload   ──► upload_<slate>.csv
                                        │        dfs.lateswap ──► swap_<slate>.csv
                                        ▼
                                dfs_boards/<date>/board_<slate>.md
                                                       │
                            dfs.review / dfs.backtest ◄┘  (+ statsapi boxscores,
                                                            dk_results/*.csv)
```

`dfs.highlight` runs the arrow the other way: `scouting_report.py` calls into the DFS
board so player names in a scouting report can be tinted by value.

### Where each concern lives

| Concern | Module | Notes |
| --- | --- | --- |
| Raw acquisition | `scouting_report.py` | StatsAPI, pybaseball, Savant, Open-Meteo. Cached under `.cache/{statcast,statsapi,batting_stats,...}` |
| Game payload cache | `scouting_report.py:3376-3410` | `{"report_args": tuple(27), "advanced_context": dict(~80 keys)}`, pickled |
| Salary-file identification | `dfs/salaries.py` | By **contents**, not filename. `describe_slate`, `slate_games`, `slate_date` |
| Stray-download adoption | `dfs/adopt.py` | Renames `DKSalaries (2).csv` into the convention |
| Output naming / versioning | `dfs/naming.py` | `dfs_boards/<date>/<kind>_<slate>[.rN].<ext>` |
| Normalization | `dfs/slate.py` + `dfs/salaries.py:attach_salaries` | Team alias folding, name normalization, DH resolution |
| Projection | `dfs/projections.py` | Event rates → DK points |
| Scoring table | `dfs/scoring.py` | 52 lines, already centralized |
| Ownership | `dfs/ownership.py` | Softmax over a fitted "field appeal" score |
| Ranking / tiers | `dfs/slate.py` | Percentiles within `Type`, GPP/CASH blends, Tier, Role |
| Optimization | `dfs/optimizer.py` | scipy `milp` (HiGHS), assignment formulation |
| CLI orchestration | `dfs/optimize.py`, `dfs/cli.py` | 607 + 168 lines |
| Exposure reporting | `dfs/exposure.py` | Per-player and per-team |
| Upload / late swap | `dfs/upload.py`, `dfs/lateswap.py` | Both DK layouts, per-game start times |
| Review | `dfs/review.py` | Tier/team scorecard, best buildable, config sweep |
| Backtest | `dfs/backtest.py` | Bias, MAE, correlation, quintiles, ceiling exceedance |
| Contest results | `dfs/results.py` | Parses `dk_results/contest-standings-*.csv` |

---

## 2. Profiling — measured, not assumed

Two runs: `build_slate` on 15 cached games, and `optimize` on the resulting 198-player
pool. Raw numbers in [`docs/benchmarks/profile_2026-08-01.json`](benchmarks/).

### 2.1 Slate build — 2.0 s

| Stage | Time | Share |
| --- | ---: | ---: |
| Unpickle 15 payloads | 0.29 s | 15% |
| `project_game` × 15 | 1.12 s | 56% |
| Salary attach, percentiles, ownership, tiers | ~0.45 s | 23% |
| Schedule API (postponements) | ~0.20 s | 6% |
| **Total `build_slate`** | **1.99 s** | |

Inside the 1.12 s of projection, `cProfile` attributes:

| Call | ncalls | cumtime | What it does |
| --- | ---: | ---: | --- |
| `_arsenal_row` | 270 | 0.37 s | `df[df["Name"] == name]` — full boolean scan **per hitter** |
| `_steal_rate` | 270 | 0.34 s | same pattern |
| `_platoon_row` | 270 | 0.33 s | same pattern, plus a regex per column |
| `_scorecard_row` | 120 | 0.26 s | same pattern |

That is **1.30 s of 1.86 s of profiled projection time spent on four O(rows) lookups
executed once per player.** Each is a dictionary lookup in disguise. This is the single
clearest speed win in the pipeline and it is pure overhead — the results are identical.

`pandas.DataFrame.iterrows` appears 3,118 times, and `players.apply(..., axis=1)` is used
for `Tier`, `Role`, `Why`, `Risks` and `attach_actual_ownership`.

### 2.2 Optimizer — 0.40 s per lineup, and the reason is one feature

```
optimize n=1                0.61 s
optimize n=5                1.53 s   (0.31 s/lineup)
optimize n=20               8.30 s   (0.42 s/lineup)
```

97% of that is inside HiGHS (`scipy.optimize._highspy._core.run`, 7.66 s of 9.00 s).
Model construction is not the problem. Isolating each feature:

| Configuration | 20 lineups | per lineup | mean ceiling |
| --- | ---: | ---: | ---: |
| full defaults (stack bonus + conflict tax) | 7.94 s | 0.397 s | 204.0 |
| no conflict tax | 7.78 s | 0.389 s | 204.1 |
| **no stack bonus** | **1.08 s** | **0.054 s** | 202.1 |
| neither | 0.93 s | 0.047 s | 202.1 |
| full, `--focus-teams` (4 teams) | 2.87 s | 0.143 s | 201.1 |

**The stack-bonus formulation costs 7.4× the solve time for +1.9 mean ceiling points.**
It adds `len(teams) × 4` indicator variables (≈120 on a 15-game slate) whose only job is
to be switched on, so the LP relaxation is weak and HiGHS branches heavily. Confining the
bonus to four focus teams already cuts the cost by 2.8×, which confirms the mechanism.

This matters far more for the target architecture than for today's workflow. Generating
1,000 candidate lineups at 0.40 s each is **6.7 minutes**; at 0.05 s each it is 50
seconds. Phase 3's two-stage design (enumerate stack cores, then complete them) replaces
the soft bonus with hard stack constraints, which are both exact and cheap.

> **Resolved — see `docs/benchmarks.md` §13.** The soft bonus turned out to be strictly
> dominated, not merely expensive: it costs 7.3× the solve time *and* never reaches a
> 5-stack, which the outcome data says is the only size that matters. A hard one-team shape
> constraint (`--stack-shape 5 --stack-teams 0`) is faster than the bonus, hits 5 every time,
> and still names no team. The bonus is now a legacy default rather than the recommended path.

### 2.3 Formulation variants

A stripped model with the same roster/salary/team/game constraints:

| Variant | 20 lineups | per lineup | variables |
| --- | ---: | ---: | ---: |
| dense — player × all 7 slots (current shape) | 0.69 s | 0.034 s | 1,386 |
| dense + `mip_rel_gap` 0.5% | 0.60 s | 0.030 s | 1,386 |
| **sparse — only eligible (player, slot) pairs** | **0.46 s** | **0.023 s** | **219** |
| sparse + `mip_rel_gap` 0.5% | 0.38 s | 0.019 s | 219 |
| sparse + `mip_rel_gap` 2% | 0.34 s | 0.017 s | 219 |

The dense encoding allocates 1,386 binaries where 219 are reachable; HiGHS presolves most
away, so the win is a modest 1.5×, not the 6× the variable count suggests. Worth taking,
but it is not where the time is.

**Conclusion on solvers: do not switch libraries.** HiGHS solves the *base* problem in 23 ms.
The cost is a modeling choice, not the solver. `ortools` is not installed and adding it
cannot be justified from these numbers.

---

## 3. Correctness findings

Recorded here because they affect projection accuracy and any measurement built on top.

### 3.1 The weather factor is inert on 66% of cached games

`_weather_hr_factor` (`dfs/projections.py:210`) reads
`environment["weather"]["temp"]` and `environment["weather"]["wind"]`.

Across the 142 cached payloads:

- **49** have a non-empty `environment["weather"]`
- **118** have `environment["forecast"]` with `temp_f`, `wind`, `wind_gust_mph`, `carry`

The keys also differ (`temp` vs `temp_f`). So for 93 of 142 games the function returns
exactly 1.0 and temperature and wind do not move a single home-run rate — silently, since
a missing key is indistinguishable from neutral weather. Example: `2026-07-30 BOS@ATH`
carries a 96.7°F forecast and gets no carry adjustment at all.

### 3.2 Ownership is estimated per player, independently

`estimate_ownership` is a softmax over a per-player appeal score. It has no notion of a
player being rostered *because a teammate is* — which is exactly the structure that
determines duplication. This is expected for a v1 model and is what Phase 5 replaces.

### 3.3 The review path rebuilds the slate from today's data

`dfs.review.main` calls `build_slate(date, ...)`, which re-reads
`.cache/report_data/*.pkl`. Those payloads are **mutated in place** by
`--refresh-lineups`: `save_report_data_cache` overwrites the same path. So a review of a
past night reads whatever the payload looked like *last time it was refreshed*, not what
was on screen when the lineups were built. Confirmed lineups, late scratches and updated
probables all leak backwards.

Concretely, `dfs.review --entered` compares entered lineups against a `Proj` column that
may never have existed at lock. This is the specific hole immutable snapshots close, and
it is why snapshots are Phase 1 rather than later.

### 3.4 Two rendering passes for one refresh

Documented in `CHEATSHEET.md` step 5: highlighting is rebuilt from the whole slate at
render time while `--refresh-lineups` refreshes one game at a time, so a first pass tints
early games against stale later games. The fix costs a second ~45 s pass. Structurally
this is compute and rendering interleaved; separating them removes the second pass.

---

## 4. Model inventory

### Projection (`dfs/projections.py`)

Hitters — per player:

1. `SLOT_PA[slot] × _team_pa_multiplier(exp_runs, is_home, home_win_prob)` → PA
2. Rate line: season + platoon blended, regressed to `LG` (`REG.season = 200 PA`)
3. Matchup factors, each damped by `FACTOR_DAMPING` — **four of the five are set to 0.0**
   (`arsenal`, `opp_sp`, `bullpen`, `park`); only `form` is live at 1.0
4. Events via `ELASTICITY` exponents on the combined factor
5. R and RBI are **shares of the team's projected runs**, normalized across the nine
6. `hitter_points(events)` → `Proj`; `Ceiling = Proj + 1.35 × sqrt(variance)`;
   `Floor` and `Bust%` from linear fits (`FLOOR_FIT`, `BUST_FIT`)

Pitchers — `_project_innings` blends recent per-start IP with the season rate; K via
`_log5` against the opposing lineup's K%; ER from a 0.75/0.25 FIP/ERA blend scaled by the
opponent's expected runs and park.

**Structural notes.** Opportunity and rate *are* already separated for hitters (PA from
slot/team, rates from skill) — Phase 2's job there is to widen what feeds PA, not to
introduce the split. Pitchers are the weaker half: `_project_innings` sees recent IP and
season IP/GS and nothing about pitch counts, rest days, or bullpen games. Uncertainty is
a deterministic function of the mean (`Ceiling` and `Floor` are both closed-form in
`Proj`), so a sparse-data player and a well-measured one with the same mean get identical
bands. That is the assumption Phase 2's shrinkage work has to break.

### Ownership (`dfs/ownership.py`)

Softmax, temperature 40, over
`0.50 × salary_pct + 0.30 × dk_avg_pct + 0.15 × value_pct + 0.05 × order_appeal`,
normalized to 800 (hitters) / 200 (pitchers), capped at 60% per player with redistribution.
Weights were fitted against five slates (503 player-slates) of measured `%Drafted`:
MAE 7.41 → 5.23, rank correlation 0.51 → 0.63. Deliberately excludes our own projection.

### Optimizer (`dfs/optimizer.py`)

- Variables: `player × slot` binaries, plus stack-level `z` and conflict `y` indicators
- Constraints: one slot per player, exact slot counts, $50k cap, ≤5 hitters/team,
  ≤9 from one game, locks, pins, optional total-ownership cap
- Multiple lineups: **re-solve from scratch**, accumulating one overlap constraint per
  previously produced lineup. Nothing is warm-started or reused
- Diversity: `max_overlap=6` and Gaussian objective jitter `randomness=0.20`
- Exposure: enforced as a *pace* across the set with an urgency-ranked forcing list and
  an infeasibility-driven relaxation loop (`optimizer.py:496-690`)

### Backtest (`dfs/backtest.py`)

Joins projections to StatsAPI boxscores on MLBAM id. Reports bias, MAE, RMSE, Pearson,
Spearman, ceiling exceedance vs a 10% target, top-vs-bottom quintile spread, and a
5-bucket calibration table, split by hitter/pitcher.

Gaps against the Phase 1 target: no quantile loss, no Brier score, no PIT/CRPS, no
breakdown by salary tier / batting order / handedness / slate size / lineup-confirmation
status, and **no walk-forward discipline** — `run_backtest(dates)` scores every date with
today's code and today's payloads, so there is no separation between what was known and
what is being predicted.

---

## 5. Storage and formats

| What | Where | Format |
| --- | --- | --- |
| Game payloads | `.cache/report_data/` | pickle, ~1-3 MB each, 142 on disk |
| Statcast / StatsAPI / weather | `.cache/*/` | pickle + parquet |
| Model calibration features | `.cache/model_calibration_games.csv` | CSV, resumable per-game rows |
| Model calibration coefficients | `.cache/model_calibration.json` | JSON |
| DK salary exports | `dfs_daily_files/` | CSV |
| Boards and lineups | `dfs_boards/<date>/` | Markdown + CSV, `.rN` versioned |
| Contest results | `dk_results/` | CSV (5 files present) |
| Upload templates | `dk_lineups/` | CSV |

Everything internal is pickle or CSV. Nothing is Parquet yet on the DFS side. Pickle of
whole DataFrame tuples is why `pyarrow` is a hard runtime dependency (pandas 3 strings are
Arrow-backed and the payloads cannot unpickle without it).

**Calibration bridge.** Each report payload carries the fingerprint of the game-model
artifact that produced its scorecard. `dfs.slate.build_slate` checks it before calling
`project_game`; stale scorecards are rebuilt from cached inputs and atomically persisted.
The applied fingerprint and any refreshed games are included in slate metadata. Thus the
optimizer does not read the calibration JSON itself, but it cannot project a stale cached
scorecard silently. Missing/invalid calibration artifacts and per-game refresh failures
fail the slate closed, preventing a partial or mixed-calibration player pool.

**Join keys.** MLBAM id is used for backtest/review joins and is correct. Ownership,
contest results and `dfs.pool` merges join on **normalized name** (plus team for pool
merges), because DK exports carry no MLBAM id. `dfs.upload` uses DK's own per-draft-group
player ids, correctly re-read from the template rather than stored.

---

## 6. What is already right, and must not regress

Worth stating explicitly before changing anything, because several of these are non-obvious
and were clearly learned the hard way:

- **Slates are identified by file contents, not filename.** Two exports on one date stop
  the run and list them rather than guessing.
- **Doubleheaders resolve per game, not per team** — everywhere: cache keys (`_g2`),
  slate resolution, started-game filtering, late swap.
- **Late swap pins kept players to their exact column.** DK diffs entries slot by slot.
- **Upload ids come from the template's own player list**, because DK renumbers players
  per draft group.
- **Nothing is overwritten**; re-runs write `.rN` and readers take the newest.
- **Ownership is never modelled from our own projection**, or leverage would be circular.
- **Real `%Drafted` replaces the estimate** wherever `dk_results/` has a matching contest.
- **Pool edits survive re-export**, including for players who have left the slate.

---

## 7. Recommended incremental plan

Ordered by (evidence unlocked) ÷ (risk taken). Each step lands behind a new flag or module
and leaves every existing CLI command byte-identical in behaviour.

**Step 1 — measurement first.** Slate snapshots, walk-forward evaluation, pipeline
profiling. None of it changes a projection or a lineup; all of it is required before any
later claim of improvement can be believed. §3.3 means we currently *cannot* honestly
score a past night.

**Step 2 — free speed.** Index the four O(rows) lookups in §2.1. Identical output,
measured. This is prerequisite work: walk-forward evaluation re-projects every cached game
many times, and a correlated simulator needs the projection path cheap.

**Step 3 — candidate/portfolio split.** Extract lineup generation into a module that
writes a candidate pool with metrics attached, so selection can be re-run without
re-solving. Use hard stack constraints instead of the soft bonus in the candidate path
(§2.2: 7.4× cheaper), keeping the existing `optimize()` behaviour untouched for the
existing CLI.

**Step 4 — correlated simulation prototype.** Hierarchical: slate → game → team → player.
Score lineups as `S @ L`. Validate marginals and correlations against history before
anything depends on it.

**Step 5 — projection construction.** Fix the weather input (§3.1). Widen PA drivers and
pitcher workload. Attach real uncertainty rather than deriving bands from the mean.

**Step 6 onward** — field simulation, duplication, contest EV, joint portfolio selection,
per `docs/simulation_and_portfolio_plan.md`.

---

# 8. The pipeline as it stands now

Sections 1-7 are the baseline. This is what exists after the simulation, field, contest and
portfolio work landed.

## 8.1 Two paths, one of which ships lineups

```
  .cache/report_data/*.pkl  ──►  build_slate  ──►  drop_started  ──►  players
                                                        │
                                                        │  (both paths from here)
                    ┌───────────────────────────────────┴────────────────────────┐
                    ▼                                                            ▼
        PRODUCTION PATH                                              EXPLORATION PATH
                    │                                                            │
        dfs.optimize --n 100                                       dfs.candidates --n 1000
        (pool file, locks, stacks,                                 (stack cores, 20-team
         exposure min/max, boosts,                                  coverage, seeded)
         pins, conflict tax)                                                     │
                    │                                                            │
                    │  lineups_<slate>.csv                                       │
                    ▼                                                            ▼
            candidates.from_optimizer  ──────►  CandidatePool  ◄─────  candidates.generate
                                                     │
                        ┌────────────────────────────┼────────────────────────┐
                        ▼                            ▼                        ▼
                 simulate_slate               simulate_field            (ownership from
                 S: sims × players            F: opponent lineups        the slate build)
                        │                            │
                        └────────────┬───────────────┘
                                     ▼
                          contest.evaluate  (payout curve, tie-split duplication,
                                             FIELD_EDGE_CORRECTION)
                                     │
                                     ▼
                          portfolio.select  (E[max], measured pairwise correlation,
                                             coverage, exposure caps)
                                     │
                        ┌────────────┴────────────┐
                        ▼                         ▼
        candidates.to_optimizer_format      docs/benchmarks/portfolio_*.json
                        │
                        ▼
                lineups_<slate>.rN.csv  ──►  dfs.upload  ──►  DraftKings
```

**The exploration path cannot upload.** `from_optimizer` keeps each lineup's original
optimizer number in `source_lineup`, and `to_optimizer_format` uses it to write the
selection back in the optimizer's own long CSV format. A pool built by `generate` has no
such number, so `to_optimizer_format` refuses rather than inventing one.

## 8.2 Why the bridge exists

The optimizer and the candidate generator are good at different halves of one job.

| | `dfs.optimize` | `dfs.candidates` |
| --- | --- | --- |
| Editable pool file (Lock/Exclude/Boost/Min%/Max%) | yes | no |
| Exposure **minimums**, pace-enforced | yes | no |
| Ranged boosts, redrawn per lineup | yes | no |
| Explicit `--stack "CIN:4,NYY:3"` | yes | `--focus-teams` only |
| Pins (late swap) | yes | no |
| Stack-core enumeration, 20-team coverage | no | yes |
| Speed, 100 lineups | ~40 s | ~5 s |

Rather than growing seven features on the generator, the bridge lets the optimizer generate
and the portfolio select. Measured on 2026-08-01 main, 20 entries chosen from 100:

| source | effective lineups | best-of-set mean | primary stacks |
| --- | ---: | ---: | ---: |
| optimizer 100 → portfolio 20 | **3.73** | 147.8 | 14 |
| optimizer 100 → top 20 by ceiling | 2.25 | 145.8 | 6 |
| candidates 500 → portfolio 20 | 5.91 | 155.6 | 15 |

The bridge captures most of the diversification gain (2.25 → 3.73) while keeping every
constraint. The generated pool still wins outright, because its stack-core coverage reaches
teams the optimizer's jitter never explores — that is the price of the constraints, and it
is visible rather than assumed.

## 8.3 Upload no longer requires a downloaded template

`upload.resolve_template` prefers a real DK download and rebuilds a **bulk** template from
the slate's salary export when there isn't one. DK's bulk template is a fixed frame around
the draft group's player list, and the salary export the board was priced from is that same
list — same ids, same slate. `dfs.optimize --upload` had done this since before this work;
`dfs.upload` gave up, so the standalone path was the one that dead-ended.

Verified on 2026-08-02 early: 20 lineups written, 200 player ids, **0 not in the draft
group and 0 in an ineligible slot.**

Two refusals are deliberate. An explicit `--template` that fails is never substituted, and
the rebuild cannot stand in for an *entries* export — Entry ID, Contest ID and Entry Fee are
issued by DK per entry, so `dfs.lateswap` still needs the real download.

## 8.4 The started-game filter

`drop_started` now lives in `dfs/slate.py`, not `dfs/optimize.py`.

It began as a CLI-level filter, which meant the candidate, field, contest and portfolio
paths all skipped it: **a portfolio built at 8pm would silently include a player whose
7:05 game was in the third inning.** He cannot score, and nothing warned you. It is a
correctness filter, not an optimizer setting, so it belongs beside the slate build.

Both `dfs.optimize` and `contest.build_pipeline` call it; `dfs.optimize` re-exports it for
compatibility. `--allow-started` on either bypasses it for analysis and backtests.

## 8.5 Module inventory, additions

| Module | Lines | What it owns |
| --- | ---: | --- |
| `dfs/profiling.py` | ~200 | Nested stage timing → `docs/benchmarks/profile_*.json` |
| `dfs/snapshot.py` | ~560 | Immutable slate snapshots, four stages, payload-drift detection |
| `dfs/evaluate.py` | ~700 | Walk-forward evaluation; pinball, CRPS, PIT, Brier, segments |
| `dfs/candidates.py` | ~700 | Candidate generation, the optimizer bridge, Parquet pools |
| `dfs/simulate.py` | ~700 | Hierarchical correlated slate simulation |
| `dfs/field.py` | ~900 | Opponent field simulation, calibrated on 25,671 real lineups |
| `dfs/contest.py` | ~570 | Payout curves, tie-split duplication, contest EV |
| `dfs/portfolio.py` | ~570 | Joint selection, effective-lineup maths, baselines |

Nothing in the original inventory (§1) changed behaviour. `dfs/slate.py` gained
`drop_started`; `dfs/projections.py` gained `E_*` expected-event columns, an
`Opp SP Hand` column and `MODEL_VERSION`; `dfs/ownership.py` gained `OWNERSHIP_VERSION`.

## 8.6 Where the calibration constants live

Every fitted number is a module constant with its provenance in the comment above it.

| Constant | Module | Fitted against |
| --- | --- | --- |
| `team_sigma`, `order_sigma`, `player_sigma` | `simulate` | 1,312 player-games, 151 team-games |
| `short_game_rate/share` | `simulate` | observed P(hitter scores 0) |
| `calibrate` (off) | `simulate` | 17,370 real field lineups |
| `PRIMARY_SIZE_SHARE`, `SECONDARY_SIZE_SHARE` | `field` | 25,671 real field lineups |
| `CANONICAL_STACK_RATE/FILL_RATE` | `field` | measured duplication, 10 contests |
| `APPEAL_POWER`, `FIELD_VALUE_WEIGHT` | `field` | 10 matched contest score distributions |
| `FIELD_EDGE_CORRECTION` | `contest` | 7 contests with known standings |

`python -m dfs.field --calibrate` recomputes the field constants from whatever is in
`dk_results/`. The rest are re-derived with the harnesses described in
[`benchmarks.md`](benchmarks.md) §6-§7A.

## 8.7 Stacking is a hard constraint, not an objective term

`docs/benchmarks.md` §13 measured 2,450 candidates against seven contests with real
standings. Stack size has **no effect on mean score** — simulated or actual — and a large
effect on the tail: p_top1 rises 1.65× and expected payout 1.85× from unstacked to 5-stack,
and 68% of each slate's top-10 actual candidates were 5-stacks.

That explains why `STACK_BONUS` was never tunable. Every optimizer objective is a mean-like
quantity, and the bonus is an attempt to encode a tail effect as a linear term in a mean. It
cannot be right at any constant.

The mechanism is `--stack-shape 5 --stack-teams 0`: a hard constraint that some team supplies
5 hitters, with the team left to the solver and 3 hitters plus both pitchers unconstrained.
`--stack-teams` exposes what was previously a hardcoded `top_teams=6` inside `stack_shapes`,
which pre-filtered stack candidates by summed hitter ceiling before salary was considered.
Widening is linear for a one-part shape and permutational for a two-part one, so the default
stays at 6.

`STACK_BONUS` is unchanged and still the default under the correlated objectives — removing
it would alter every existing run. It is now documented as the legacy path.

## 8.8 Shape diversification

`--stack-shape` takes a comma-separated list (`"5-3,5-2,5"`). Combinations are round-robin
interleaved in `stack_shapes`, with shorter shapes cycled rather than exhausted, so any prefix
of the returned list holds the requested mix — callers consume it by cycling, and the shapes
generate very different counts (380 for `5-3` against 20 for `5` on a 20-team slate).

`docs/benchmarks.md` §14 measured the secondary and found the aggregate metrics and the
extreme tail disagree, with the tail favouring a secondary stack and the evidence at ~1.9σ.
The setting that survives either reading is to spread, which is what the list enables.

This also fixed a silent misparse: `_parse_shape` previously kept only the `.isdigit()` parts
after splitting on `-`, so `"5-3,5-2"` was accepted and quietly reinterpreted as `5-2`.

Shape diversity is a generation-side concern. The portfolio's `stack_coverage` weight and
`max_stack_exposure` both key on `primary_stack` — the team — and only *report*
`distinct_stack_shapes`. Nothing downstream will recover a shape the pool never contained.
