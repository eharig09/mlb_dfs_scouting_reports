# Benchmarks and measured results

Every number here was produced on this machine against the repository's own cached data.
Machine-readable copies are in [`benchmarks/`](benchmarks/). Where a change did **not**
help, that is recorded too — the whole point of building the measurement layer first was to
be able to tell.

**Environment.** Windows 11, Python 3.14.4, numpy 2.4.6, pandas 3.0.3, scipy 1.17.1
(HiGHS), pyarrow 24.0.0. Data: 142 cached game payloads across 12 dates
(2026-07-22 → 2026-08-02). Reference slate: **2026-08-01 main**, 15 games, 198-player
priced pool.

---

## 1. Runtime

### 1.1 Projection path — 2.8× faster, bit-identical output

Four per-player lookups (`_arsenal_row`, `_steal_rate`, `_platoon_row`, `_scorecard_row`)
were full boolean DataFrame scans run once per player against the same frame. They are now
built as a dictionary once per frame, with the cache verified against frame identity and
cleared per game.

| | before | after | change |
| --- | ---: | ---: | ---: |
| 142 cached games | 7.78 s | 2.74 s | **2.84× faster** |
| per game | 54.8 ms | 19.3 ms | |

Correctness: **0 of 2,840 player-rows differ** across all 142 payloads, comparing 18 output
fields including the `Supports`/`Cautions` note lists. This is a pure speed change.

### 1.2 Slate build

| stage | before | after |
| --- | ---: | ---: |
| projection (15 games) | 0.94 s | 0.30 s |
| unpickle | 0.24 s | 0.23 s |
| schedule API | 0.31 s | 0.31 s |
| render | 0.07 s | 0.07 s |
| ownership | 0.01 s | 0.01 s |
| **total `dfs.cli`** | **2.19 s** | **1.49 s** |

Reproduce with `python -m dfs.cli --date 2026-08-01 --slate main --profile`.

### 1.3 Where the optimizer's time actually goes

97% of `dfs.optimize` runtime is inside HiGHS. Isolating each feature on the reference
slate, 20 lineups:

| configuration | total | per lineup | mean ceiling |
| --- | ---: | ---: | ---: |
| full defaults (stack bonus + conflict tax) | 7.94 s | 0.397 s | 204.0 |
| no conflict tax | 7.78 s | 0.389 s | 204.1 |
| **no stack bonus** | **1.08 s** | **0.054 s** | 202.1 |
| neither | 0.93 s | 0.047 s | 202.1 |
| full, focus 4 teams | 2.87 s | 0.143 s | 201.1 |

**The stack-bonus formulation costs 7.4× the solve time for +1.9 mean ceiling points.** It
adds ~120 indicator variables whose only role is to be switched on, so the LP relaxation is
weak and HiGHS branches hard. This is a modelling cost, not a solver limitation.

### 1.4 Formulation variants (stripped model, same roster constraints)

| variant | 20 lineups | per lineup | variables |
| --- | ---: | ---: | ---: |
| dense — player × all 7 slots | 0.69 s | 0.034 s | 1,386 |
| dense + `mip_rel_gap` 0.5% | 0.60 s | 0.030 s | 1,386 |
| **sparse — eligible (player, slot) pairs** | **0.46 s** | **0.023 s** | **219** |
| sparse + gap 0.5% | 0.38 s | 0.019 s | 219 |
| sparse + gap 2% | 0.34 s | 0.017 s | 219 |

**Verdict: do not change solver libraries.** HiGHS solves the base problem in 23 ms. The
6× variable reduction only buys 1.5× because presolve already removes most of it. `ortools`
is not installed and nothing here justifies adding it.

### 1.5 Candidate generation — 9-18× faster than re-solving with the optimizer

`dfs.candidates` uses the sparse encoding, a reused base model, hard stack cores instead of
the soft bonus, and a 0.5% optimality gap.

| | candidates | time | per lineup |
| --- | ---: | ---: | ---: |
| `dfs.optimizer` at measured 0.40 s/lineup | 200 | ~80 s (est.) | 0.400 s |
| `dfs.candidates` | 200 | **5.4 s** | **0.027 s** |
| `dfs.candidates` | 1,000 | **44.3 s** | **0.044 s** |

Per-lineup cost rises with pool size because duplicate rejection makes later solves harder,
which is expected.

### 1.6 Simulation

| sims | time | memory | per sim |
| ---: | ---: | ---: | ---: |
| 1,000 | 0.35 s | 0.8 MB | 347 µs |
| 10,000 | 2.01 s | 7.9 MB | 201 µs |
| 50,000 | 15.9 s | 39.6 MB | 317 µs |

Bit-identical for a fixed seed; different seeds differ.

---

## 2. Projection accuracy

### 2.1 Walk-forward baseline

`python -m dfs.evaluate` — 12 dates collected, 9 scored (the first three are consumed as
the training window). Floor and bust bands are refit per target date on strictly earlier
dates only. 2,318 player-games.

| | n | bias | MAE | RMSE | spearman | top-decile recall | CRPS | PIT mean | PIT KS | bust Brier skill | ceiling exceeded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hitters | 2,009 | −0.32 | 5.28 | 6.78 | 0.194 | 0.139 | 3.61 | 0.467 | 0.124 | 0.032 | 0.104 |
| pitchers | 226 | +0.10 | 8.19 | 10.07 | 0.260 | 0.304 | 5.75 | 0.497 | 0.062 | 0.003 | 0.111 |

**The ceiling band is honest.** It is exceeded 10.4% (hitters) and 11.1% (pitchers) of the
time against a 10% target. That was already true and remains true.

Four things this measurement newly exposes:

- **The per-player bust number barely beats the base rate.** Brier skill 0.032 for hitters
  and 0.003 for pitchers. "Everyone busts at the league rate" is nearly as good as the
  model's per-player estimate. `Bust%` is currently a linear function of `Proj`, so this is
  not surprising — but it means the column carries almost no player-specific information.
- **Unconfirmed lineups are systematically over-projected by 1.5 points.** Confirmed
  hitters: bias −0.001. Projected hitters: bias −1.489. The projected-lineup penalty is
  currently a note in the `Why` column, not a number in the projection.
- **Hitter distributions are too narrow at the bottom.** The PIT histogram's lowest decile
  holds 17.6% of the mass against a target of 10%, even after crediting the model with a
  zero atom. Real blanks happen more often than the published bands imply.
- **Rank quality is modest.** Spearman 0.19 for hitters is normal for single-game baseball
  — most of a hitter's outcome is noise — but it caps how much any selection rule can do
  and is worth remembering before blaming the optimizer for a bad night.

### 2.2 By segment (hitters)

| segment | n | bias | MAE | spearman | top-decile recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| min-priced | 235 | −0.84 | 4.41 | 0.205 | 0.125 |
| cheap | 215 | +0.49 | 4.86 | 0.108 | 0.182 |
| mid | 219 | +0.08 | 5.20 | 0.064 | 0.091 |
| expensive | 212 | −1.08 | 5.53 | 0.194 | 0.190 |
| premium | 213 | +0.02 | 6.23 | 0.107 | 0.190 |
| batting 1st | 223 | −0.20 | 6.00 | 0.131 | 0.182 |
| batting 9th | 213 | −0.13 | 4.31 | 0.193 | 0.095 |
| LHB vs LHP | 153 | −0.45 | 5.99 | 0.111 | 0.200 |
| LHB vs RHP | 601 | −0.28 | 5.23 | 0.219 | 0.167 |
| RHB vs LHP | 291 | −0.69 | 5.33 | 0.188 | 0.034 |
| RHB vs RHP | 568 | −0.28 | 5.22 | 0.149 | 0.158 |
| confirmed lineup | 1,576 | **−0.00** | 5.34 | 0.185 | 0.184 |
| projected lineup | 433 | **−1.49** | 5.05 | 0.265 | 0.116 |

### 2.3 The weather fix — correct, but not an improvement

`_weather_hr_factor` read `environment["weather"]`, which is empty on **93 of 142** cached
payloads; the real data sits in `environment["forecast"]` under different key names. So
temperature and wind moved nothing on two thirds of games. Fixed in MODEL_VERSION 2, which
also neutralises enclosed roofs.

Walk-forward A/B over all 12 dates, 2,420 scored player-games:

| metric | v1 (inert) | v2 (fixed) | delta |
| --- | ---: | ---: | ---: |
| hitter MAE | 5.2400 | 5.2410 | +0.0010 |
| hitter spearman | 0.1790 | 0.1790 | 0.0000 |
| hitter CRPS | 3.5802 | 3.5808 | +0.0006 |
| pitcher MAE | 8.1170 | 8.1140 | −0.0030 |
| pitcher CRPS | 5.7169 | 5.7145 | −0.0024 |

**This is a wash.** Deltas are in the fourth decimal place on both sides. The change is kept
because it is a correctness fix — the code now reads the input it was written to read — but
it is **not** an accuracy improvement and is not claimed as one.

Two plausible reasons the effect is negligible, both worth chasing later: the weather HR
factor is clipped to [0.94, 1.10] and only multiplies the home-run rate, which is a small
slice of a hitter's points; and the team's projected runs, computed upstream, may already
absorb the weather signal. Raw data: [`benchmarks/ab_weather_fix.json`](benchmarks/ab_weather_fix.json).

---

## 3. Simulation validity

`python -m dfs.simulate --date 2026-08-01 --slate main --sims 10000 --seed 7 --actuals`

### 3.1 Marginals

| | projected mean | simulated mean | gap | sim SD | P(0) | P(≤3) | sim p90 − stated ceiling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hitters (178) | 6.80 | 7.02 | +0.22 | 6.18 | 0.147 | 0.350 | −0.84 |
| pitchers (20) | 13.43 | 14.34 | +0.91 | 9.04 | 0.054 | 0.117 | +1.26 |
| **observed actuals (187)** | — | 8.18 | — | 7.72 | 0.209 | 0.321 | — |

The simulator reproduces the projection's means and lands its p90 within a point of the
published ceiling. Two honest gaps remain: **simulated SD 6.18 against an observed 7.72**
(under-dispersed) and **P(0) 0.147 against an observed 0.209** (too few outright blanks).
Both point the same way and both match §2.1's PIT finding, which is reassuring — three
independent diagnostics agreeing that the left tail is too thin.

Two bugs were found and fixed during this calibration, both worth recording because both
produced plausible-looking output:

- **Conditional thinning omitted the out probability.** Each plate appearance's event class
  was drawn against the sum of the remaining *event* classes rather than against all
  unclaimed mass, so the ~70% chance of an out was excluded. Hitters simulated at 16.9
  points against a 6.8 projection.
- **The exponential shocks were not mean-preserving.** `exp(X)` for `X ~ N(0, s²)` averages
  `exp(s²/2)`, not 1, so with `hr_elasticity` 1.7 home runs came out 25% high. Each
  multiplier now subtracts `e²s²/2`.

### 3.2 Correlation structure

| relationship | expected | measured r | pairs |
| --- | --- | ---: | ---: |
| teammate hitters | positive | **+0.303** | 704 |
| — adjacent in the order (≤2 apart) | most positive | +0.311 | 295 |
| — distant in the order | less positive | +0.297 | 409 |
| opposing hitters, same game | slightly positive | +0.066 | 792 |
| pitcher vs the hitters he faces | **negative** | **−0.305** | 178 |
| hitters in different games | ~zero | +0.006 | 14,257 |

Every sign and ordering is correct, and the magnitudes sit inside the range published for
MLB DFS (teammates roughly +0.15 to +0.35, pitcher-vs-opposing-bats roughly −0.2 to −0.4).
None of these is imposed as a coefficient: the shared team-run pool produces all of them.

The adjacent-vs-distant separation is weak (+0.311 vs +0.297). `order_sigma` is small
relative to `team_sigma` and the shared run pool dominates both. Worth tuning; noted rather
than papered over.

### 3.3 The simulation disagrees with summed ceiling

1,000 candidates scored on 20,000 simulated slates: **mean rank disagreement between summed
ceiling and simulated 99th percentile is 166 places out of 1,000.** The top of the two
rankings differs materially — one candidate ranked 8th on simulated p99 projects 95.5
against the ceiling-leader's 111.7.

This is the entire thesis of the plan, measured: the current objective and a correlated
simulation of the same slate do not rank lineups the same way.

---

## 4. Candidate pool quality

`python -m dfs.review --date 2026-08-01 --slate main --candidates <dir>`

```
1000 candidates, 13 stack shapes, 20 primary teams, 133 pitcher pairs
best candidate scored 172.8 (4 ARI, proj 85.5, own 68.6)
pool mean 86.3, median 86.3, worst 21.8

  if you had entered one lineup, chosen by:
    highest projection            113.3
    highest ceiling               113.3
    lowest total ownership         61.5
    highest simulated p99         113.3
    random candidate (mean)        86.3
    ex-post best in the pool      172.8

  night's ceiling (hindsight, any legal lineup)   269.6
  best the pool could have given you              172.8   <- generation headroom 96.7
  best any ranking rule actually picked           113.3   <- selection headroom 59.5
```

Generation headroom is largely irreducible — the ex-post best lineup is assembled from
whoever blew up (Hunter Goodman, 8.45 projected, 40.0 actual), and no projection pre-selects
those. **Selection headroom of 59.5 points is the part that was genuinely available**, and
it is what Phases 6-8 exist to capture.

### Pruning safety

Unguarded dominance pruning removed 114 of 198 players on the reference slate and left only
**13 of 20 teams able to field three hitters**, with one team removed outright — meaning no
candidate could ever stack them. Pruning is now off by default, and when enabled it enforces
a per-team hitter floor as well as a per-position one:

| mode | players kept | teams stackable | primary teams in 200 candidates |
| --- | ---: | ---: | ---: |
| unguarded (removed) | 84 / 198 | 13 / 20 | 11 |
| guarded, opt-in `--prune` | 119 / 198 | 20 / 20 | 20 |
| off (default) | 198 / 198 | 20 / 20 | 20 |

Core ordering was also changed to round-robin across teams. Pure score ordering spent the
whole solve budget on the single best offense.

---

## 5. Tests

```
332 passed in 602 s
```

| module | covers |
| --- | --- |
| `test_scoring.py` | DK Classic scoring, pinned to published rules |
| `test_optimizer.py` | roster validity, stacks, exposure, overlap, locks, pins, conflict tax, ownership cap, determinism |
| `test_candidates.py` | legality, diversity, pruning safety, persistence, `S @ L` scoring |
| `test_simulate.py` | reproducibility, marginals, correlation signs and ordering, stack tail, shared run pool |
| `test_field.py` | seat matching, entry legality, teammate lift, stack shares, duplication, contest-export parsing |
| `test_contest.py` | payout expansion, tie splitting, duplicate counting, nested probabilities, EV arithmetic |
| `test_portfolio.py` | effective-lineup maths, selection vs baselines, locks, exposure caps, weight presets |
| `test_snapshot.py` | immutability, revision chains, round-trip, payload-drift detection |
| `test_evaluate.py` | pinball / CRPS / PIT / Brier against analytic answers, walk-forward leakage |
| `test_slate.py` | doubleheaders, started-game filtering, weather inputs, stack building |
| `test_ownership.py` | normalisation to 800/200, capping, non-circularity, real-`%Drafted` override |
| `test_projections.py` | indexed lookups vs the boolean scans they replaced, events-to-points contract |

All randomness is seeded. Tests marked `integration` skip themselves when `.cache/report_data`
is empty, so the suite runs on a clean checkout.

Four real defects were found by writing them:

- `optimize(stacks={"CHW": 4})` did not alias-fold team codes, while `--stack CHW:4` did.
  The library entry point reported "only 0 CHW hitters".
- The field's seat-feasibility check sliced the seat list to the number of players chosen
  so far. Because `HITTER_SEATS` is ordered C/1B/2B/3B/SS/OF/OF/OF, a five-man stack of
  outfielders was tested against seats containing no outfield slot and could never be built
  — most of why five-stacks came out at 45% against a measured 54%.
- The portfolio's local-improvement pass could swap out a locked lineup, so `--lock` worked
  or not depending on whether a better swap happened to exist.
- `FieldPool` raised `AttributeError` instead of its own error message when the slate had
  no ownership column, because `DataFrame.get` returns `None` and `pd.to_numeric(None)`
  hands back a bare numpy nan with no `.isna()`.

And the test fixture itself initially drew `Proj` independently of its event counts, which
is exactly the inconsistency the events-to-points contract test now forbids.

---

## 6. Field simulation (Phase 5)

Calibration set: originally **25,671 real field lineups across 10 contest exports** in
`dk_results/`, contest sizes 98 to 5,945, single-entry through 20-max; **refreshed
2026-08-08 to 53,685 lineups across 19 exports** (see 6.2). Each export's `Lineup` column
holds the full construction of every entry, so field behaviour is directly observable rather
than assumed. The co-occurrence lifts in 6.1 are still the original measurement.

### 6.1 The independence assumption fails, measured

| relationship | lift vs independence |
| --- | ---: |
| **teammate hitters** — P(B \| A) / P(B) | **2.62×** (median 2.31, p90 4.59) |
| hitters on different teams | 0.75× |
| pitcher pairs, against a normalised baseline | 1.01× (median; 51% above, 49% below) |

Teammates are 2.6× more likely to be rostered together than a per-player model implies;
players on other teams are slightly *less* likely, because they compete for the same salary.
Pitcher pairs, once you condition on their marginals, are almost exactly independent — so
the field simulator draws them that way, and that is a measurement rather than a shortcut.

### 6.2 How the field actually builds

Recalibrated **2026-08-08** over **53,685 lineups across 19 exports** (98 to 8,917 entries) —
roughly twice the original set. The 10-export figures are kept alongside because the change
is a change in the *field*, not a correction: five-stacks really did fall away over these
three weeks, and a slate's game count moves the whole distribution.

| primary stack size | 19 exports | 10 exports | | commonest shapes | share |
| --- | ---: | ---: | --- | --- | ---: |
| 5 hitters | **45.0%** | 53.6% | | 5-2 | 22.3% |
| 4 | **28.4%** | 23.4% | | 5-3 | 12.7% |
| 3 | **15.1%** | 12.4% | | 4-2 | 11.3% |
| 2 | **10.2%** | 9.6% | | 5 (no secondary) | 10.1% |
| 1 / none | **1.3%** | 0.9% | | 4-3 | 9.1% |

Also: **0.7%** of real entries roster a pitcher facing 3+ of their own hitters (the field
treats this as very nearly a rule), **6.1%** take both pitchers from one game, and **91.0%**
of entries are unique overall — 81% to 98% by contest, with a maximum duplication of 27.

### 6.3 Simulated field vs the real thing

8,917-entry 20-max field on 2026-08-07 main, scored against that contest's real `%Drafted`
(re-run 2026-08-08 on the refreshed constants):

| | simulated | real |
| --- | ---: | ---: |
| primary 3-stacks | 19.5% | 15.1% |
| primary 4-stacks | 32.2% | 28.4% |
| primary 5-stacks | 39.9% | 45.0% |
| entries unique | 90.9% | 91.0% |
| most-duplicated lineup | 34× | 27× |
| ownership MAE | 1.33 pts | — |
| ownership correlation | 0.915 | — |
| salary used | 98.1% of cap | — |

Duplication and ownership land; the stack ladder is shifted one notch soft — five-stacks
~5pp under target, 3-4 stacks over — because the repair step trades some five-stacks down
when it cannot seat them. That bias survived the recalibration unchanged in size, which is
what says it belongs to the repair step and not to the constants.

The earlier 2026-08-01 run against the 10-export constants, for comparison: 58.3% five-stacks
against a then-measured 53.6%, 94.0% unique against 92.9%, ownership MAE 2.49 / r 0.733
(that slate's ownership was the model's own estimate, not measured `%Drafted`, which is most
of the correlation gap).

And the score distribution, which is the strongest available test because a candidate's EV
is computed against it:

| | winner | p99 | p80 | median |
| --- | ---: | ---: | ---: | ---: |
| simulated field | 184.7 | 153.5 | 108.4 | 86.0 |
| real contests (5 with 1,000+ entries) | 180.4 | 150.0 | 112.1 | 90.1 |

The top of the distribution is close; the middle runs ~4 points light.

### 6.4 What was fitted, and one thing that was not

`CANONICAL_STACK_RATE` and `CANONICAL_FILL_RATE` — the probability an entrant takes the
obvious version of a stack rather than sampling — were fitted by a 4×4 sweep against
measured duplication. With them off the field came back **99.8% unique with a maximum
duplication of 2**, against a real 92.9% and 23; at the fitted 0.25 / 0.08 it lands at
93.1% and 24.

**`FIELD_VALUE_WEIGHT` was tried and rejected.** The hypothesis was that ownership-only
sampling builds a field that scores below the real one because real entrants also optimise
for value. Swept 0.0 → 0.85:

| value weight | winner | p99 | p80 | median | ownership MAE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 184.7 | 153.5 | 108.4 | 86.0 | **2.17** |
| 0.55 | 186.8 | 154.6 | 109.2 | 87.0 | 3.10 |
| 0.85 | 184.8 | 152.0 | 108.9 | 87.5 | 3.89 |
| *real* | *180.4* | *150.0* | *112.1* | *90.1* | — |

It buys 1.5 points of field median and costs 1.7 points of ownership accuracy. Left in as a
knob, defaulted to zero.

### 6.5 The conflict fallback, found by a new slate

`_add_pitchers` excludes pitchers facing the lineup's own stack, and used to relax that rule
whenever no affordable non-conflicting arm was left. On a thin slate that fired constantly:
2026-08-03 main prices only 14 pitchers, and the simulated field rostered a pitcher against
its own stack **4.15% of the time against a measured 0.5%** — eight times too often, purely
because the cheap arms left after a five-stack happened to face it.

It now abandons the entry and lets `draw_entry` redraw with a different stack, keeping the
relaxation only for the final attempt so a genuinely constrained pool still yields a field.

| slate | pitchers priced | before | after |
| --- | ---: | ---: | ---: |
| 2026-08-03 main | 14 | 4.15% | **0.30%** |
| 2026-08-02 early | 17 | — | 0.40% |
| 2026-08-01 main | 20 | — | 0.10% |
| *real field* | | | *0.50%* |

Caught by an integration test that picks the newest cached slate, so a thin slate appearing
in the cache surfaced a weakness the three slates fitted against never exposed. Worth noting
as an argument for keeping that test date-relative rather than pinned.

### 6.6 Speed

| | before | after |
| --- | ---: | ---: |
| per field entry | 13.85 ms | **2.34 ms** |
| 6,000-entry field | 83.1 s | **14.0 s** |

The seat-feasibility check was 10 of 15 seconds. It is now a memoised bitmask matching:
feasibility depends only on the multiset of eligibility patterns, and a slate has very few
distinct patterns, so the same handful of questions recur thousands of times.

---

## 7. Simulator calibration against actual outcomes

Two corrections, both measured, both found while trying to explain implausible EV.

### 7.1 Dispersion

`player_sigma` was 0.22, producing distributions visibly too narrow. Fitted against 753
finished player-games over four dates:

| | simulated | observed |
| --- | ---: | ---: |
| hitter SD | 6.70 | 6.67 |
| pitcher SD | 10.16 | 10.26 |
| hitter mean | 6.50 | 6.42 |

Set to **0.40**, with `run_dispersion` 3.2 → 4.0.

**Residual, unfixed:** simulated P(hitter scores 0) is 0.167 against an observed 0.264.
Pushing `player_sigma` to 0.85 gets it to 0.218 but overshoots SD by 10%. The model has no
mechanism for the sub-population that gets one plate appearance — pinch-hitters, early
exits — and that is where the missing blanks live.

### 7.2 Projections are not truth

Regressing actual on projected over 2,471 finished hitter-games and 277 starts:

```
hitters    actual = 0.872 * proj + 0.578
pitchers   actual = 0.667 * proj + 4.862
```

**A one-point projection edge is worth 0.87 actual points for a hitter and 0.67 for a
pitcher.** The simulator drew rates straight from the projection, so it assumed the
projection was correct and overstated the spread between good and bad players by a third at
pitcher. Applied as a per-player mean *shift*, not a scaling — scaling also multiplies the
spread, which collapsed simulated pitcher SD from 10.16 to 6.80.

---

## 7A. Tracking down the EV inflation

The §8 finding below — EV overstated by roughly an order of magnitude — was chased to
ground. This section is the investigation, because most of it is negative results and those
are the expensive part to rediscover.

### 7A.1 The first comparison was confounded

Every field-vs-real comparison had averaged real contests across nights and compared them to
a simulated field on one night. Slate scoring environments differ enormously — real contest
medians on the ten exports range from **73.0 to 99.4** — so that comparison meant little.

Matching each export to its own slate by player overlap (all 10 matched, overlap 0.70-1.00)
gives six dates. Notably **2026-08-01 main, the slate every earlier benchmark used, has no
contest at all**; it was being compared against other nights.

Per-slate, the field model was much better than it had looked:

| | median | top 20% | top 1% | winner |
| --- | ---: | ---: | ---: | ---: |
| mean error (simulated − real) | −4.2 | −4.6 | **−0.8** | **−1.8** |

The tail — which is what tournament probabilities depend on — was nearly right.

### 7A.2 The correlation was 2.7× too strong

Marginals were already fitted, so the aggregate had to be checked. Comparing simulated
against observed spread over 151 real team-games:

| | simulated / observed SD |
| --- | ---: |
| single hitter | 0.98× |
| two adjacent hitters | 1.01× |
| **five-stack** | **1.23×** |

Singles and pairs right, five-stacks 23% too wide — the signature of excess correlation.
Measuring it directly on residuals (actual − projected) over 10 dates:

| relationship | real | simulated (before) | simulated (after) |
| --- | ---: | ---: | ---: |
| teammate, adjacent in order | +0.103 | +0.288 | +0.114 |
| teammate, distant in order | +0.101 | +0.266 | +0.113 |
| pitcher vs the hitters he faces | −0.281 | −0.145 | −0.232 |
| hitters in different games | −0.000 | −0.002 | +0.005 |

Two errors in opposite directions: teammates 2.8× too correlated, pitcher-vs-opposing only
half as negative as reality.

**The adjacency effect is not in the data.** Real teammate correlation is +0.103 for hitters
within two batting slots and +0.101 for those further apart, over 2,086 pairs. The simulator
had an `order_sigma` producing a spurious one; it is now ~0.

The cause of the excess was double-counting: the shared team-run pool already generates
teammate correlation, and an explicit `team_sigma` of 0.26 was layered on top. Cutting it to
0.06 and raising `player_sigma` to 0.48 (to hold total marginal spread) fixed it.

### 7A.3 The missing mechanism: hitters who get pulled

P(hitter scores zero) was 0.156 simulated against 0.259 observed, and the marginal SD ran 7%
light. Both are one hole: no way to produce the one-plate-appearance game. Raising
`player_sigma` widened the distribution symmetrically and fixed neither.

Added `short_game_rate` (0.18) and `short_game_share` (0.35), **mean-neutral by
construction** — `SLOT_PA` was fitted as PA accrued by the player who *starts*, which
already averages over early exits, so the full-game rate is scaled up by the expected
retention. Without that scaling the mechanism silently lowered every projection.

P(0) improved to 0.195. It does not reach 0.259 without a short-game rate near 0.34, which
pushes five-stack SD to 20.7 and re-inflates the tail. Documented rather than tuned away.

### 7A.4 The projection calibration was reverted

§7.2's shrinkage (`actual = 0.872·proj + 0.578`) is a true description of E[actual|proj] in
the population, and applying it per player made things **worse**, because a lineup contains
ten *selected* players and the linear fit over-shrinks at the top.

Checked directly against **17,370 real field lineups**, summing each entry's ten players:

| | mean error | mean abs |
| --- | ---: | ---: |
| raw projection sum | **−1.2** | 4.5 |
| calibrated sum | −5.9 | 7.5 |

The raw sum is nearly unbiased at lineup level. Turning the calibration off halved the
simulated field's error against real contests:

| | median | p80 | p99 | winner | mean abs |
| --- | ---: | ---: | ---: | ---: | ---: |
| calibration on | −5.3 | −7.2 | −7.4 | −9.9 | 7.5 |
| **calibration off** | **−2.1** | **−3.6** | **−3.5** | **−5.9** | **3.8** |

It also improved pitcher SD (9.49 → 9.99 against 9.9) and five-stack SD. `calibrate` now
defaults to `False`; the constants stay for a future non-linear version, since the top
pitcher bucket needs about a third of the shrinkage the bottom one does.

### 7A.5 A tunable that was not tunable

`APPEAL_POWER` was bound as a default argument, so it became unreachable at import.
A calibration sweep over it returned **identical results at every value** — the worst way
for a knob to fail. Fixed to look up the module constant at call time. Once working, 1.35
was confirmed optimal: raising it lifts the field's median but pushes the tail *down* and
degrades ownership accuracy.

`FIELD_VALUE_WEIGHT` was re-swept under the corrected simulator and **still does not pay**
(0.0 best), reproducing the earlier negative result on different foundations.

### 7A.6 What it actually was

The probability miss is real and not sampling noise. Per night, comparing the observed count
of qualifying candidates against the model's own predicted distribution:

- observed lands inside the model's 10-90 band on 6 of 7 nights — **the spread is right**
- but at the **25th percentile** for cash and the **15th** for top-1%, not the 50th
- with 7 nights that is 2.3σ and 3.2σ — a systematic location bias

The location bias, found by comparing our pool against the **real** field rather than the
simulated one:

| | model says | actually |
| --- | ---: | ---: |
| our pool's edge over the field | **+1.3 pts** | **−5.3 pts** |

**Our candidate pool is worse than the real field**, and the model cannot see it because it
compares against a simulated field built by a similar process. Two causes, neither a bug:
the pool is generated for *coverage* (round-robin over all 20 teams, objective jitter), so
its average member is deliberately mediocre; and the simulated field runs ~5 points soft at
the median.

### 7A.7 The correction

One parameter, `FIELD_EDGE_CORRECTION`, subtracted from candidate scores before ranking —
correcting the gap rather than patching output probabilities, so ranks, tie-splitting and
payout arithmetic stay mutually consistent.

| correction | P(cash) ratio | P(top 1%) ratio | P(win) ratio |
| ---: | ---: | ---: | ---: |
| 0.0 | 1.56 | 3.61 | 4.63 |
| 6.6 *(derived from the edge gap)* | 1.09 | 1.93 | 2.16 |
| **9.0** *(fitted to observed frequencies)* | **0.95** | **1.53** | **1.63** |
| 12.0 | 0.79 | 1.13 | 1.13 |

Set to **9.0**. The derived and fitted values agreeing to within a third — one from lineup
scores, the other from finishing frequencies — is the reason to trust either.

A single scalar cannot fix both ends, because the candidate pool's upper tail is also
slightly too fat (realised scores clear the simulated p90 7.7% of the time against an
expected 10%). The residual 1.5× on top-1% stays documented; seven contests will not
support a second parameter.

---

## 8. Contest EV — after the investigation

**Where it stands.** Probabilities now carry the §7A.7 correction and were re-measured
against the seven contests with known standings:

| | predicted | observed | ratio |
| --- | ---: | ---: | ---: |
| P(cash) | 0.133 | 0.141 | **0.95×** |
| P(top 1%) | 0.0073 | 0.0048 | 1.53× |
| P(win) | 0.00082 | 0.00048 | 1.63× |

The cash line is calibrated. The tail still runs ~1.5× high, for the reason in §7A.7.

**Discrimination, measured within each contest** (pooling across contests is invalid — a
good night lifts every candidate regardless of the model):

| ranking | mean within-contest spearman vs actual |
| --- | ---: |
| raw projection | **+0.177** |
| summed ceiling | +0.170 |
| simulated mean | +0.150 |

An earlier note claimed simulated mean beat summed ceiling (+0.214 vs +0.157). **That was
one slate and it does not replicate.** Across seven contests the three are within noise of
each other, and simulated mean is marginally the worst. The simulation earns its place by
producing *distributions* — which is what duplication, contest EV and portfolio correlation
need, and none of which a point estimate can supply — not by ranking players better.

Discrimination is also wildly uneven: +0.42 and +0.33 on two nights, ~0.00 on three.

**What EV is now good for.** Ordering candidates, and sizing the relative gap between them.
It is not a bankroll tool: the payout curve is synthetic unless you supply a real one with
`--payout-json`, and the tail probabilities remain ~1.5× high.

**What would close the rest**, in order of expected value:

1. **More contests with standings.** Every number here rests on seven, and the top-1%
   estimate on ten realised events. This is the binding constraint on everything else.
2. **Make the simulated field as strong as the real one.** It is ~5 points soft at the
   median. Concentration (§7A.5) and value-weighting (§6.4) were both tried and both failed;
   the next idea is heterogeneous entrant skill — a minority of sharp entrants rather than
   one homogeneous population.
3. **Thin the candidate pool's upper tail.** Realised scores clear the simulated p90 7.7% of
   the time against an expected 10%.

## 9. Joint portfolio selection (Phase 8)

20 lineups from 500 candidates, 2026-08-01 main, 6,000 simulations, 2,500-entry field:

| method | best-of-set mean | best-of-set p99 | **effective lineups** | mean pairwise corr | primary stacks | pitcher pairs | distinct players |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **joint portfolio** | **155.6** | **246.6** | **5.91** | 0.126 | **15** | **20** | 103 |
| highest P(win) | 151.4 | 242.1 | 3.88 | 0.219 | 7 | 17 | 80 |
| highest EV independently | 149.4 | 238.5 | 3.23 | 0.273 | 6 | 16 | 72 |
| random | 146.2 | 223.9 | 6.06 | 0.121 | 11 | 17 | 111 |
| highest projection | 143.9 | 232.3 | 2.23 | 0.419 | 6 | 8 | 70 |
| **highest ceiling** (current default) | 143.4 | 232.3 | **2.07** | 0.456 | 4 | 12 | 67 |

**Ranking 20 lineups by summed ceiling — what the tool does today — produces a set worth
2.07 independent bets.** Joint selection gets 5.91 while also scoring 12 points higher on
best-of-set. Random selection is nearly as diversified (6.06) and 9 points worse, which is
the check that the portfolio objective is doing something other than adding noise.

`effective_lineups = n / (1 + (n-1)r)` is the number worth watching. It is what
`--max-overlap` has always been approximating by counting shared players, computed instead
from the simulated score vectors — so it measures whether two lineups fail in the same
worlds rather than whether they happen to share a catcher.

Unlike §8, this comparison does not depend on the EV *levels* being right — every method is
scored on the same simulated slates, so the ranking between methods is valid even though
the dollar figures are not.

---

## 9A. Ownership model — refit on 3× the data, and where its ceiling is

Fit set: **14 slates, 1,724 player-slates** with measured `%Drafted`, one contest per slate
(the largest field), dated by `dfs.results.contest_night`. The original fit had 5 slates and
503 rows. Raw output in `docs/benchmarks/ownership_refit.json`.

**It held out of sample.** MAE **4.67**, Spearman **0.610**, against 5.23 / 0.63 in-sample.
Pitchers rank far better than hitters (0.784 vs 0.574) and are worse in absolute error
(MAE 7.95 vs 4.30), which is what you would expect when two roster slots carry 200 points of
ownership between ~15 candidates.

**Nothing moved it.** Every knob was swept and the shipped values won:

| what was tried | result |
| --- | --- |
| temperature 10 → 50 | MAE 7.42 / 5.70 / 4.96 / **4.67 at 40** / 4.70 |
| 1,771-point weight simplex × T ∈ {30,40,50} | best 4.652 vs shipped 4.669 — noise |
| z-score or min-max instead of the percentile transform | same MAE-vs-bias frontier |
| `MAX_OWNERSHIP` 40 → 100 | ±0.05 MAE; the cap never binds at T=40 |

**The residual is a ranking limit, not a calibration error.** The model under-calls chalk
badly and over-calls the tail:

| real ownership | n | mean real | mean est | bias |
| --- | ---: | ---: | ---: | ---: |
| <1% | 312 | 0.5 | 3.7 | −3.3 |
| 1–5% | 680 | 2.6 | 5.7 | −3.1 |
| 5–15% | 503 | 8.8 | 9.7 | −0.9 |
| 15–30% | 176 | 20.7 | 14.9 | +5.8 |
| **30%+** | 53 | **42.3** | **23.8** | **+18.5** |

Sharpening is the obvious fix and it fails: it concentrates ownership on whoever
`field_appeal` ranks first, and that ranking recovers only **4.1 of the field's ten
most-owned**. At T=8 the model's top pick averages 64% owned against a real 43% — it is
capable of calling someone chalk, it just does not know who. Spreading the mass is the
correct response to that uncertainty, so the +18.5 is left in place deliberately.

Moving it needs a new **signal**, not a new constant: Vegas implied team runs, a public
consensus projection, or the same player's measured ownership on recent slates. Note the
model deliberately excludes our own projection — see `dfs/ownership.py` — so a *public*
projection is the one that would not make Leverage circular.

---

## 10. What is still not measured

- Portfolio results against **realised** contest outcomes over multiple nights. The
  comparison in §9 is simulated-against-simulated; it shows the objective does what it
  claims, not that it makes money.
- Contest EV levels (§8), which are known to be wrong and are labelled as such.
- Field behaviour on slate sizes outside the 19 contests on hand, all of which are
  medium-to-large MLB slates.
- Whether the 2026-08-08 stack-share shift (five-stacks 53.7% → 45.0%) is the field changing
  or the newer exports being a different mix of contest sizes. Both are consistent with the
  data on hand.
- Anything about showdown, or about FanDuel scoring.

---

## 11. How to set exposures — measured

Four experiments on the seven contests with real standings, scored both in simulation and
against what actually happened. Raw output in `docs/benchmarks/exposure_analysis.json`.

### 11.1 Where the projection actually has signal

Exposure is a bet on being right, so the first question is how often we are.

**Pitchers — real signal at the top.** 12 slates, ~26 priced per slate:

| our rank | mean actual | top-1 | top-3 | top-5 | bust |
| --- | ---: | ---: | ---: | ---: | ---: |
| #1 | **24.1** | 33% | 50% | 67% | **0%** |
| #2 | 20.5 | 25% | 50% | 50% | 8% |
| #3 | 16.8 | 9% | 27% | 45% | 9% |
| #4-5 | 14.6 | 5% | 9% | 18% | 18% |
| #6-10 | 14.6 | 0% | 9% | 22% | 7% |
| *random pick* | *14.0* | | | | |

Our #1 arm beats a random one by **+10.1 points** and captures **50% of the edge available**
between a random pick and the night's best. The signal is gone by #4.

**Hitters and stacks — almost none.** Same 12 slates:

| | mean actual | top-1 | top-3 | share of available edge |
| --- | ---: | ---: | ---: | ---: |
| our #1 hitter | 6.9 | 0% | 0% | **2%** |
| random hitter | 6.5 | | | |
| our #1 five-stack | 37.9 | 0% | 8% | **4%** |
| random five-stack | 36.3 | | | |

`spearman(projected stack, actual stack) = **-0.025**`. We cannot tell which offense will
explode. The #2 projected stack outscored the #1 (49.1 vs 37.9), and the #6-10 hitters
outscored the #1-5.

This is the whole basis for everything below: **concentrate at pitcher, spread at hitter.**

### 11.2 Maximum exposure caps

Portfolio of 20 and 150 from 250 candidates, capping every player:

| cap | best-of-set | eff. lineups | actual max exposure | **realised best** | **real rank** |
| ---: | ---: | ---: | ---: | ---: | ---: |
| *none* | 163.5 | 7.23 | 43% | **160.6** | **23** |
| 60% | 163.5 | 7.27 | 42% | 160.6 | 23 |
| 40% | 163.2 | 7.50 | 37% | 160.2 | 23 |
| 25% | 162.3 | 7.95 | 28% | 159.6 | 27 |
| 15% | 159.8 | 8.53 | 23% | 153.7 | 52 |

*(150 entries; 20 entries shows the same shape more sharply — 15% cap drops realised best
from 135.2 to 124.8 and rank from 201 to 370.)*

**Two findings.** The uncapped portfolio already self-limits to 43% max exposure, because
the correlation penalty in the objective is doing the job — so **a 60% cap is a literal
no-op and 40% barely binds.** And below 25% the caps buy diversification that costs real
points: they trade a better *worst* case for a worse *best* case, which is backwards in a
tournament.

### 11.3 Minimum exposure, by how confident we are

Forcing a player into X% of a 150-entry set:

| forced | #1 pitcher | #3 pitcher | #8 pitcher | #1 five-stack |
| ---: | ---: | ---: | ---: | ---: |
| 0% | 158.4 | 156.1 | **164.4** | **160.6** |
| 15% | **160.2** | 153.3 | 152.1 | 148.9 |
| 25% | **160.2** | 156.8 | 149.9 | 146.5 |
| 40% | 147.9 | **156.9** | 148.7 | 145.3 |
| 60% | 145.7 | 156.9 | 144.4 | 141.7 |
| 85% | 144.1 | 153.8 | 138.8 | 136.1 |

*(realised best lineup; bold marks the optimum)*

**The optimum moves down as confidence falls, exactly as it should** — and it is never
high:

| the bet | edge captured | optimum |
| --- | ---: | --- |
| #1 projected pitcher | 50% | **15-40%** |
| #3 projected pitcher | ~14% | 15-40% |
| #8 projected pitcher | ~0% | **0-15%** |
| #1 projected five-stack | 4% | **0-15%** |

**Every curve is worst at 85%.** Effective lineups collapse from 7.7 to 4.4-6.2 with nothing
gained. On this evidence there is no player and no slate where an 80% minimum is the right
call — not even the arm we are most confident about, on the position where we have the most
signal.

Simulated and realised optima disagree for the #1 pitcher (40% vs 15%). Seven slates is a
small sample for the realised column; the simulated column averages 4,000 draws per slate
and is steadier but inherits the model. Where they disagree, the honest reading is "somewhere
in 15-40%", not either endpoint.

### 11.4 Entry count changes the answer

| | 20 entries | 150 entries |
| --- | --- | --- |
| granularity | 1 lineup = **5%** | 1 lineup = 0.67% |
| uncapped max exposure | 54% | 43% |
| #1 pitcher optimum | 25-50% | 15-40% |
| cost of a 15% cap | −10.4 realised pts | −6.9 realised pts |

At 20 entries you cannot express exposure finer than 5%, and the set is concentrated by
construction — so explicit numbers add little. At 150 the portfolio has room to diversify
and **over-concentration is punished harder**, so the optimum is lower.

### 11.5 What to actually do

1. **If you are using `dfs.portfolio`, set almost nothing.** The correlation penalty already
   caps exposure near 43%, which is where the optimum is. Caps at 40%+ are no-ops; caps
   below 25% cost points.
2. **Use maximum caps as a risk rail, not a tuning knob** — `--max-player-exposure 0.4` to
   stop one player owning the set, and expect it rarely to bind.
3. **Minimums only for pitchers, only the top two or three, and only to 15-40%.**
4. **Do not lock.** 100% exposure was the worst or near-worst setting in every configuration
   tested, including for the best pitcher on the slate.
5. **Never force a stack.** Stack projection has no measurable signal (spearman −0.025);
   forcing the top-projected one was worse than not forcing it in every row.
6. **On the `dfs.optimize` path the pool file's `Min%`/`Max%` matter more**, because that
   path has no correlation-aware objective to self-limit. That is the case exposure controls
   were built for.

---

## 12. Portfolio selection — 177× faster swap pass

Selecting 150 entries from 400 candidates over 20,000 simulations was projected at
**2.6 hours**, essentially all of it in the local swap pass. It now runs the whole pipeline
in **2.5 minutes**.

### 12.1 Where it went

The swap pass tries every candidate in every position: `150 × 250 × 2 = 75,000` objective
evaluations. Each call to `_objective` rebuilt everything from scratch — a `np.corrcoef`
over a 20,000 × 150 block (84 ms of the 93 ms), a full max across 150 columns, and three
pandas `nunique` calls.

### 12.2 Three changes

**A correlation is a dot product.** For Z-scored columns, `corr(i,j) = (Z[:,i]·Z[:,j])/n`.
So for a selection with column sum `S`, the sum over all pairs is `(S·S)/n`, and the mean
pairwise correlation is `((S·S)/n − k) / (k(k−1))` — O(n_sims) to evaluate and O(n_sims) to
update on a swap, with no correlation matrix ever built. The greedy phase uses the same
identity, replacing a matmul that grew with `k` by one matrix-vector product.

**E[max] from the top two.** Keeping the best *and* runner-up per simulation, plus which
position holds the best, makes "max excluding position p" a `np.where` rather than a
recomputation.

**Coverage as counter deltas.** Three `Counter.copy()` calls per trial swap meant 225,000
dict copies. The distinct count only moves when a code hits or leaves zero, which is
arithmetic on the two codes involved.

### 12.3 Measured

| | per trial swap | 75,000 evaluations |
| --- | ---: | ---: |
| original | 93.4 ms | 1.9 hours |
| incremental | 1.71 ms | 129 s |
| \+ coverage deltas | 1.85 ms | 139 s |
| **\+ contiguous transposes** | **0.53 ms** | **39 s** |

The last step was the surprise. `scores` is `(n_sims, n_candidates)` in C order, so a column
is a strided view — correct, and free to *slice*, but arithmetic over it walks 20,000 cache
lines:

| operation | strided | contiguous |
| --- | ---: | ---: |
| `column_sum − Z[:,i] + Z[:,j]` | 403 µs | **32 µs** |
| `np.maximum(without, scores[:,j])` | 118 µs | **36 µs** |

Storing contiguous transposes costs `n_sims × n_candidates × 4` bytes for the standardised
copy — 32 MB at this scale — and paid for itself 3.5×. A first guess that the *slice* was
expensive was wrong: numpy slicing returns a view in 0.4 µs. The cost is entirely in the
arithmetic that follows.

### 12.4 Correctness

`_objective` is kept as the reference implementation and the fast path is pinned against it:
agreement to **2.4e-4** across randomised trial swaps and after applying them, and — the
test that actually matters — **the fast pass selects the identical portfolio** the slow one
does, verified against a literal re-implementation of the old loop.

Timings on the way there: n=15 swap pass 5.21 s → 0.09 s; n=30 17.88 s → 0.37 s; both with
identical selections.

A leftover reference to the old attribute survived in `apply()` and would have crashed on
the first *accepted* swap — the benchmark only ever called `value()`. Found by grep, not by
the benchmark, which is an argument for the equivalence tests exercising `apply` too.

### 12.5 The flag

`--swap-passes` (default 2). Greedy alone already carries the submodular guarantee, so 0 is
a legitimate setting; the passes typically move a handful of lineups. All the exposure
analysis in §11 used `swap_passes=0` because at the time the pass was unaffordable.

---

## 13. How much stacking is optimal — measured

The soft `STACK_BONUS` was the only unguided stacking mechanism, and it was tuned by feel.
Before adding another knob, measure what the target actually is.

### 13.1 Method

Seven contests with real standings (2026-07-27 → 08-02). On each slate, build a pool with
deliberate coverage of *every* primary-stack size: the core enumerator at `sizes=(2,3,4,5)`
for the stacked candidates, plus `optimize(stack_bonus=False, randomness=0.55)` for the
unstacked ones. Score all of them on the same simulations and the same simulated field, then
compare against what actually scored. 2,450 candidates.

### 13.2 Result

| primary stack | n | sim mean | sim p99 | p_top1 | exp EV | ACTUAL | best actual | real rank % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 77 | 92.8 | 154.7 | 0.0046 | 2.61 | 85.7 | 142.9 | 61.7 |
| 2 | 457 | 91.7 | 154.3 | 0.0046 | 2.92 | 88.7 | 151.1 | 56.1 |
| 3 | 107 | 92.9 | 157.4 | 0.0058 | 3.68 | 88.9 | 135.8 | 53.2 |
| 4 | 463 | 92.1 | 159.1 | 0.0063 | 4.10 | 88.0 | 169.5 | 54.4 |
| 5 | 1346 | 92.4 | 161.0 | **0.0076** | **4.83** | 86.7 | **205.2** | 58.0 |

**Stacking does nothing for the mean and everything for the tail.** Mean simulated score is
flat at 91.7–92.9 across every stack size, and mean *actual* score is flat too (85.7–88.9,
with no ordering). But the 99th percentile climbs monotonically, p_top1 rises 1.65×, expected
payout rises 1.85×, and the best score any candidate actually posted goes 142.9 → 205.2.

The decisive number is the top of each slate:

| primary stack | share of each slate's top 10 *actual* candidates |
|---|---:|
| none | 2% |
| 2 | 12% |
| 3 | 0% |
| 4 | 18% |
| 5 | **68%** |

There is no interior optimum. It runs to the DK limit.

Note the mean-actual column is *not* monotone and the per-slate table is noisier still
(2026-07-27 favoured unstacked, 07-30 favoured 3-stacks). That is the finding, not noise
around it: a stack is a bet that one team has a big inning, and most nights it doesn't. The
edge is entirely in what happens when it does.

### 13.3 Why the soft bonus was the wrong instrument

`STACK_BONUS` is a term added to a linear objective, and every objective the optimizer has is
a mean-like quantity. It is trying to encode a *tail* effect into a *mean* — which is why it
needs a magic constant, and why no setting of that constant is right.

Measured on 2026-08-04 main, 30 lineups, `--objective ceiling --max-overlap 6`:

| mechanism | solve time | primary stack sizes | distinct stack teams |
|---|---:|---|---:|
| `--no-stack-bonus` | 3.8s | 4×1, 21×2, 5×3 | 15 |
| default soft bonus | 27.7s | 16×2, 10×3, 4×4 | 14 |
| `--stack-shape 5 --stack-teams 0` | 21.8s | **30×5** | **20** |

The soft bonus is strictly dominated: 7.3× the solve time of no bonus, and it still never
reaches a 5-stack — the only size that matters per §13.2. The hard constraint is *faster*
than the bonus (a fixed stack shrinks the feasible region; a bonus term only makes the
objective harder to bound) and hits the target exactly.

### 13.4 The mechanism

`--stack-shape 5` was already the least restrictive way to require a stack, and was being
mistaken for the restrictive one. It claims 5 of 8 hitter slots, leaves 3 hitters and both
pitchers open, and **does not name a team** — it enumerates one-team combinations and lets
the solver choose. The genuinely restrictive flags are `--focus-teams` (names teams) and
multi-part shapes like `4-3` (fixes the split).

The one real restriction was the undocumented `top_teams=6` inside `stack_shapes`, which
pre-filtered candidates by summed hitter ceiling *before* salary was considered. That is now
`--stack-teams` (default 6, `0` = every team). Widening is cheap for a one-stack shape —
combinations grow linearly, 6 → 20 — but expensive for a two-stack shape, where they grow as
permutations (30 → 380).

Recommendation: `--stack-shape 5 --stack-teams 0`, and leave the soft bonus off.

---

## 14. Does the *secondary* stack matter?

§13 settled the primary at 5. That leaves three hitter slots. Same method — same slates, same
simulations, same field — conditioned on the primary being 5 and split by what the other three
do. 8,903 five-stacks, built by asking the solver for each shape by name across every team
(the enumerator alone leaves 5-3 almost unrepresented).

### 14.1 The two halves disagree

| shape | n | sim mean | sim p99 | p_top1 | exp EV | mean ACTUAL | best actual | rank % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 (scattered) | 1199 | **92.1** | **160.6** | 0.0074 | 4.77 | **85.6** | 176.4 | **58.5** |
| 5-2 | 4055 | 90.6 | 159.4 | 0.0072 | 4.57 | 82.5 | **205.2** | 63.6 |
| 5-3 | 3649 | 89.1 | 158.9 | 0.0071 | 4.60 | 82.0 | 194.4 | 64.0 |

Every aggregate column prefers the scattered 5 — but by very little, and p_top1/exp EV are
flat to three digits. Meanwhile the extreme tail prefers the opposite. Per-slate top-10 share,
corrected for the unequal candidate counts (a shape with 3.4× the candidates should take 3.4×
the top-10 seats for free):

| shape | base rate | top-10 share | lift | seats (of 70) | expected |
|---|---:|---:|---:|---:|---:|
| 5 (scattered) | 13.5% | 6.0% | **0.45×** | 4 | 9.4 |
| 5-2 | 45.5% | 42.0% | 0.92× | 29 | 31.9 |
| 5-3 | 41.0% | 52.0% | **1.27×** | 36 | 28.7 |

Same shape of story as §13, one level down: the secondary costs ~3 points of mean and buys
tail. The best score any candidate posted was a 5-2 at 205.2 and a 5-3 at 194.4, against 176.4
for the best scattered lineup.

### 14.2 How much to believe it

Less than §13. Seventy top-10 seats total, and the scattered shortfall (4 against 9.4) is
about 1.9σ — suggestive, not decisive. The aggregate columns point the other way and are also
within noise of each other. **The honest reading is that no single shape is demonstrably best,
and the secondaries are where the extreme scores came from.**

That argues for spreading across shapes rather than picking one, which is also the setting
that does not require the read to be right.

`exp_dupes` came back 0.00 for every shape, so the "the field crowds 5-2" hypothesis is
**untested**, not refuted — at 2,000 simulated entries against a candidate space this large the
duplication signal is simply absent. Testing it needs a field simulation at real contest size.

### 14.3 What the optimizer was actually producing

Measured, 40 lineups under `--stack-shape 5 --stack-teams 0`:

| shape | optimizer | real field (`SECONDARY_SIZE_SHARE`) |
|---|---:|---:|
| 5 (scattered) | 60% | 20% |
| 5-2 | 37% | 50% |
| 5-3 | **2.5%** | **30%** |

Close to the inverse of the field, and concentrated in the one shape that took 0.45× its share
of top-10 seats. `--stack-shape` now accepts a comma-separated list — `"5-3,5-2,5"` — with the
combinations round-robin interleaved so any prefix holds the mix. That matters because the
shapes generate wildly different counts: on a 20-team slate `5-3` is 380 combinations and `5`
is 20, so concatenating would give the one-stack shape 13% of a 150-lineup run rather than a
third. Shorter shapes cycle rather than exhaust.

### 14.4 A silent misparse, found on the way

`--stack-shape "5-3,5-2"` was already accepted before this change and silently meant `5-2`:
the parser split on `-`, then kept only the parts passing `.isdigit()`, so `'5-3,5-2'` became
`['5', '3,5', '2']` → `[5, 2]` and the middle vanished without a word. Malformed shapes now
raise. `'5,4'` is now valid and means what it looks like: a 5-stack or a 4-stack.
