# NFL / PFF integration log

A running record of how the PFF drop gets into the model: what was built, what was
*measured* rather than assumed, and the traps that cost time. Newest step at the bottom.

Scope: the `nfl/` package only. Nothing here touches the MLB side.

**Reading rule for this file** — a number stated here was checked against real data, and the
check is named. Anything not verified is labelled as an assumption.

---

## Step 1 — The data layer (`nfl/pffdata.py`)

Everything else joins through this, so it went first.

### The problem

The drop is ~90 CSVs across 11 report families, three years deep for the new ones and eight
for the older ones. Two properties of it break joins *silently* — they produce code that
runs clean and matches almost nothing:

1. **A PFF filename carries no season.** Files arrive as `receiving_summary (4).csv`. The
   number is download order: not chronological, and inconsistent between families.
2. **Names are a lossy join key.** Suffixes, accents and team codes all disagree between
   PFF and nflverse.

### What was built

| Piece | What it does |
|---|---|
| `crosswalk()` | PFF `player_id` → nflverse `gsis_id`, pooled over 2017-25 |
| `identify_season(frame)` | Which season an export describes, from its own contents |
| `catalog()` | Every file on disk with its identified season, cached on size+mtime |
| `resolve(family, season)` | The one file to read |
| `load(family, season)` | That file, canonicalised and keyed |

### Finding: PFF and nflverse share an id space

nflverse rosters publish a `pff_id` column, and it is the same id as `player_id` in every
PFF export. **This removes name matching from the primary path entirely.**

Coverage, pooling the crosswalk over 2017-25 (pooling matters — a single season's rosters
fill the column for only about half the players, because a player who carries the id on any
roster row carries it forever):

| family | id coverage |
|---|---|
| `receiving_summary` | 97.2% |
| `slot_coverage` | 99.6% |
| `passing_depth` | 99.2% |
| `offense_blocking` | 66.4% |

The line is the exception, and not by obscurity — Penei Sewell and Joel Bitonio are both
missing. nflverse simply has not filled the column in for most linemen. So name-and-team
matching stays as a **fallback**, and `key_source` on every loaded frame records which key
answered so a name match never quietly passes for an exact one.

### Finding: season is identifiable from roster membership alone

The method: take each file's `(player, team)` pairs and score them against every season's
rosters. **The signal is player movement** — about a quarter of the league changes clubs
each off-season, so the true season matches nearly every row and its neighbours match ~60%.

Measured separation across the whole drop: true season **0.95–0.99**, runner-up **~0.60**,
margin never below 0.27. Decisive on every one of the ~90 files.

Two things this bought that the previous stat-error approach could not:

- **`slot_coverage` is now pinned** (2020-25). It had been unidentifiable and therefore
  unused, because that approach needed a counting stat mapping onto nflverse and slot
  coverage does not have one. It is the substrate for the slot-weakness work.
- **`offense_blocking` is pinned** (2023-25) for the same reason.

It also reproduced every previously hand-verified season without being told them.

### Bug found: `fantasy-stats-passing` was mapped backwards

The hand-built table in `nfl/redzone.py` was correct that `fantasy-stats-receiving` puts the
newest season in the *unnumbered* file — and then assumed passing matched. It does not.
**That family numbers the other way**, so all four seasons were reversed: 2022 filed as
2025, 2025 filed as 2022.

Verified independently of the roster fingerprint, by joining each file's `yds` against real
nflverse passing yards:

| file | vs 2022 | vs 2023 | vs 2024 | vs 2025 |
|---|---|---|---|---|
| `fantasy-stats-passing.csv` | **0** | 1295 | 1333 | 1326 |
| `(1)` | 1327 | **0** | 1164 | 1064 |
| `(2)` | 1273 | 1162 | **0** | 1155 |
| `(3)` | 1248 | 1093 | 978 | **0** |

*(mean absolute error, passing yards, QBs over 1500 yards)*

**Blast radius: none, yet.** The passing map was never loaded — no `load_fantasy_passing`
exists — so nothing reached a report. It was one function call away from putting a
three-year-old quarterback split on a live slate.

The fix is structural rather than a corrected constant: `redzone.py` no longer writes the
season down at all. It calls `pff_path(family, season)`, which resolves through the catalog.

### Bug found: a stat file with no stats

`Receiving Depth/receiving_depth (5).csv` has the right 498 players, the right teams, and
**six columns** — identity only, no statistics. It fingerprints to 2025 perfectly and would
have contributed nothing to a join without raising. The catalog now marks a file unusable
when it carries no columns beyond identity.

### Design decisions worth keeping

- **The season is identified, never written down.** A hand-maintained table was wrong within
  one family of being right, and it looked right.
- **The catalog is the thing you read when a join comes back empty.** It states what the
  loader believes, which is the first thing to check against what you assumed.
- **Cached on size *and* mtime**, so re-downloading the same filename with different contents
  re-fingerprints rather than reusing a stale season.
- **The wider file wins** when two exports cover the same season, since PFF's snap filters
  only ever remove players. Ties break on path, so the choice is stable across runs.

### What is on disk, as identified

| family | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|
| `receiving_summary` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `rushing_summary` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `receiving_scheme` (man/zone) | | | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `defense_coverage_scheme` | | | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `slot_coverage` | | | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `fantasy-stats-receiving` | | | | | ✓ | ✓ | ✓ | ✓ |
| `fantasy-stats-passing` | | | | | ✓ | ✓ | ✓ | ✓ |
| `receiving_depth` | | | | | | ✓ | ✓ | ✓ |
| `receiving_concept` (slot/screen) | | | | | | ✓ | ✓ | ✓ |
| `passing_depth` | | | | | | ✓ | ✓ | ✓ |
| `offense_blocking` | | | | | | ✓ | ✓ | ✓ |

---

## Step 2 — What in this data is actually predictive

Before wiring anything into projections, every candidate signal was tested the same way:
**year-over-year correlation on the same player or team**. A signal that does not correlate
with its own next season cannot forecast one, whatever it explains after the fact.

This pass reshaped the plan, so it is worth reading before the feature work.

### The one-line result

> **PFF's value here is in describing *where and how a player is deployed*, which is close to
> deterministic year to year. It is not in rating *how well anyone performed*, which mostly
> is not — and on defense, barely at all.**

### The ranking (mean year-over-year r)

**Deployment and volume — build on these**

| signal | r | source | n |
|---|---|---|---|
| inline rate (TE alignment) | **+0.95** | `receiving_summary` | ~170 |
| wide rate | **+0.92** | `receiving_summary` | ~170 |
| **average depth of target** | **+0.89** | `receiving_summary` | ~170 |
| route participation rate | **+0.88** | `receiving_summary` | ~170 |
| **red-zone carries / game** | **+0.85** | `fantasy-stats-receiving` | ~271 |
| **slot rate** | **+0.82** | `receiving_summary` | ~170 |
| **red-zone targets / game** | **+0.80** | `fantasy-stats-receiving` | ~271 |
| RB targets / game | +0.77 | `rushing_summary` | ~42 |
| **inside-5 carries / game** | +0.73 | `fantasy-stats-receiving` | ~271 |
| RB carries / game | +0.67 | `rushing_summary` | ~42 |
| targets per route run | +0.62 | `receiving_summary` | ~170 |

**Player skill — usable with shrinkage**

| signal | r | source |
|---|---|---|
| O-line overall grade | +0.62 | `offense_blocking` |
| YPRR | +0.61 | `receiving_summary` |
| O-line pass-block grade | +0.60 | `offense_blocking` |
| route grade | +0.55 | `receiving_summary` |
| O-line run-block grade | +0.53 | `offense_blocking` |
| RB gap/zone carry share | +0.54 | `rushing_summary` |
| elusive rating | +0.53 | `rushing_summary` |

**Noise — do not build matchup adjustments on these**

| signal | r | source |
|---|---|---|
| defensive man rate *(tendency, not quality)* | +0.46 | `defense_coverage_scheme` |
| inside-5 carries as a *share* of carries | +0.33 | `fantasy-stats-receiving` |
| RB yards per carry | +0.27 | `rushing_summary` |
| RB yards after contact | +0.28 | `rushing_summary` |
| receiver man-vs-zone YPRR *gap* | +0.19 | `receiving_scheme` |
| **defense slot YPT allowed** | **+0.10** | `slot_coverage` |
| **defense man YPT allowed** | **+0.05** | `defense_coverage_scheme` |
| **defense zone YPT allowed** | **+0.01** | `defense_coverage_scheme` |
| **defense slot TD rate allowed** | **+0.01** | `slot_coverage` |
| **defense man-vs-zone gap allowed** | **−0.08** | `defense_coverage_scheme` |

### What this means for four of the stated goals

**Man vs zone matchups — real but tiny, and only through volume.**
Two separate findings stack against it. A receiver's man-vs-zone *efficiency* gap barely
persists (r = +0.19), so "this guy is a man-beater" is mostly last season's noise. His
man-vs-zone *target rate* gap does persist (r = +0.37 to +0.51), so volume, not efficiency,
is the channel. Sized end-to-end — shrink last year's gap by its regression slope (0.378),
apply the opponent's man rate against league average (27.6%), multiply by median routes
(20.4/game):

| receiver | vs most man-heavy defense (40%) | vs most zone-heavy (19%) |
|---|---|---|
| 90th-pct man-target-rate | **+0.09 tgt/gm** | −0.06 tgt/gm |
| 10th-pct (zone-leaning) | −0.07 tgt/gm | +0.05 tgt/gm |

About a tenth of a target per game at the two extremes of *both* distributions. That is
worth roughly 0.15 DK points — display and tie-break only, never a projection input.

**Slot weakness hunting — the defensive half does not exist.**
Slot YPT allowed has r = +0.10 year over year and slot TD rate allowed r = +0.01. Last
season's "vulnerable slot defense" is a coin flip. The *offensive* half is one of the most
stable numbers in the data (slot rate, r = +0.82), so the useful question is reframed: not
"which defense is weak in the slot" but "which receivers live there", crossed with defensive
*deployment* rather than defensive results.

**Long-ball threats — the threat side is nearly deterministic, the vulnerability side is not.**
aDOT is r = +0.89, one of the strongest signals available; identifying deep threats is close
to free. Finding "vulnerable deep defenses" from the same style of season-allowed table
should be expected to behave like every other allowed-rate above, and needs a different
source (per-play air yards) before it is worth trusting.

**High-value carries — strongly buildable, with one trap.**
Red-zone carries per game r = +0.85 and inside-5 per game r = +0.73. But inside-5 carries as
a *share* of a back's carries is r = +0.33. **Model high-value work as a per-game volume,
never as a fraction of workload** — the fraction is dominated by how the denominator moved.

**O-line projection — the inputs hold up.**
Individual pass-block (r = +0.60) and run-block (r = +0.53) grades persist about as well as
receiver efficiency, which is what a projection built from returning starters' grades needs.

### Why this is consistent with the rest of the package

It is the same finding the projection model already runs on — opportunity self-correlates
(carries 0.68, targets 0.61), scoring rates do not (TDs 0.03–0.24), so the model projects
volume from a player's own form and takes rates from a positional prior. The PFF data
extends the volume side considerably. It does not rescue the rate side, and defensive
allowed-rates behave worse than anything already in the model.

---

## Step 3 — The deployment layer (`nfl/usage.py`)

The feature module Step 2 pointed at. Five of the seven stated goals read from it.

| function | what it returns |
|---|---|
| `receiver_profile(seasons)` | routes/game, TPRR, slot/wide/inline rate, aDOT, deep target rate, YPRR |
| `rusher_profile(seasons)` | carries and targets per game, gap/zone share, elusive, breakaway |
| `high_value_touches(seasons)` | red-zone and inside-5 carries, red-zone and end-zone targets — **per game** |
| `team_target_distribution(seasons)` | target and route share per player, plus the team's WR/TE/RB mix |

All four accept a single season or a list. A list gives a recency-weighted blend
(`RECENCY_WEIGHTS = 1.0 / 0.45 / 0.20`, not fitted), which is what makes the three years of
depth in the new drop worth having.

### Two rules the module enforces

**Per-game volume, never a share of a moving denominator.** Inside-5 carries per game
self-correlate at +0.73; the same fact as a *share of a back's carries* drops to +0.33,
because the share tracks whatever happened to the denominator — a back whose early-down role
shrank reads as having gained goal-line equity.

**The weights renormalise per player.** A rookie present in one season of three is averaged
over the weight that actually exists for him. Summing over absent seasons would divide his
real usage by three and report a role he does not have.

### Two silent-blank bugs caught while building it

- **PFF has no `RB`.** Backs are `HB`, so `pos == "RB"` matched nothing and *every team's*
  RB target share came back NaN — a whole position missing, with no error anywhere. Fixed
  in `nfl.salaries.canon_position`, which every PFF load now routes through. Sanity check
  after the fix: league mix WR 58.8% / TE 23.9% / RB 17.3%, with SFO the most RB-centric
  passing game in the league. `T`/`G`/`C` deliberately do **not** fold together — where a
  lineman plays is the point of the line work.
- **The fantasy-stats endpoint was not canonicalised.** It writes raw PFF codes, so
  `high_value_touches` had Davante Adams on `LA` and Trey McBride on `ARZ` while every other
  profile said `LAR` and `ARI`. Any join on team would have dropped those clubs without
  raising. This is the third distinct appearance of the team-code trap in this package.

### Tests

`tests/nfl/test_nfl_pffdata.py` and `tests/nfl/test_nfl_usage.py`, 21 tests, **offline** —
`identify_season` takes its fingerprints as an argument precisely so the method can be
tested without nflverse. They cover the refusal-to-guess case (a file of players who never
changed clubs matches every season equally, and the honest answer is "unidentified"), the
name fallback, team-code folding, the `HB` fold, and the blend's renormalisation.

---

## Step 4 — O-line inputs, measured

The stated design was to project lines from *past grades, movement to new teams, returning
starters, and draft capital*. Three of those four were testable directly; here is what each
is worth. (Module not yet written — these are the fitted inputs it will use.)

**Spot assignment is free.** `offense_blocking` carries per-alignment snap counts
(`snap_counts_lt/lg/ce/rg/rt`), so each lineman's position is his max-snap spot rather than
something that has to be inferred. Spot-checked against 2025: Terence Steele RT, Drew Dalman
C, Joe Thuney LG — all correct.

**Past grades: yes, r ≈ 0.60.** Overall grade +0.62, pass block +0.60, run block +0.53 at a
400-snap threshold. Comparable to receiver efficiency, and enough to project from.

**Movement: a grade only half travels.** This is the interesting one.

| | 2023→24 | 2024→25 |
|---|---|---|
| stayed on the same team (n≈137) | **r = +0.61** | **r = +0.60** |
| changed teams (n = 29) | r = +0.27 | r = +0.38 |

A lineman who changes clubs retains roughly **half** the predictive signal of one who stays.
So movement is not a flag to display next to a carried-over grade — it is a reason to shrink
that grade toward the positional mean. *Caveat: 29 movers per season. The direction repeats
across both pairs, but the magnitude is not precise, and it should be refit as seasons
accumulate rather than treated as a constant.*

**Draft capital: yes, and it predicts snaps as well as grade.**
Rookie linemen 2023-25 (n = 68): Spearman(pick, grade) = **−0.405**, fitting

    grade ≈ −5.02 × log(pick) + 76.6

| | n | mean grade | mean snaps |
|---|---|---|---|
| round 1 | 20 | 62.9 | 930 |
| round 2 | 17 | 56.2 | 838 |
| round 3 | 12 | 59.3 | 680 |
| rounds 4-7 | 19 | 50.1 | 499 |

Note round 2 grading *below* round 3 — the same small-n round-boundary inversion that
`nfl/coldstart.py` documents for skill positions, and the same reason to fit on `log(pick)`
rather than on round. Snaps fall monotonically (930 → 499) even where grade does not, so
draft capital answers "does he start at all" more cleanly than "how well does he play".

**Returning starters: not yet measured.** Team-level continuity is the one input of the four
still open, and given that every other *team-level* rate in this data failed to persist
(Step 2), it should be measured before it is trusted rather than assumed to help.

### Bug found: the catalog cache key was not normalised

Wiring `redzone.py` through the catalog made the test suite hang. A caller passing
`root="."` builds the root as `./nfl/pff`, which globs to `./nfl/pff/x.csv` — a different
*string* from the cached `nfl/pff/x.csv` for the same file. Every such call missed the cache
and re-fingerprinted all ~90 exports, several of them 500 columns wide.

Fixed by normalising every path into the cache key, plus per-process memoisation of the
catalog and the roster fingerprints. Lookup went from minutes to **0.03s**, and there is a
regression test on the key.

### The module (`nfl/oline.py`)

    from nfl.oline import projected_line, team_line_strength
    projected_line(2026)                        # five projected starters per club
    team_line_strength(projected_line(2026))    # the unit, and how it was arrived at

**Bottom-up, from the five individuals.** An individual grade self-correlates at +0.60; a
team's *unit* grade at +0.46 and +0.18 over the two pairs on file. The player is the stable
unit and the team aggregate is not, because the personnel underneath it churns — so nothing
carries last season's unit grade forward.

The grade ladder, with `basis` naming which rung answered — the same first-class `basis`
column the skill board uses, so "PFF graded him at 78" and "we inferred it from a
third-round pick" never look alike on a page:

| basis | rule |
|---|---|
| `grade` | graded, staying put — regressed as `mean + 0.60 × (grade − mean)` |
| `grade (moved)` | graded, new club — regressed at 0.33 instead |
| `draft capital` | rookie, from `−5.02 × log(pick) + 76.6` |
| `replacement` | post-rookie with no graded snaps, which is evidence *against* him |

Spots come from PFF's per-alignment snap counts (`snap_counts_lt/lg/ce/rg/rt` — note `ce`
for centre), and the projected starting five from the newest depth-chart snapshot, where
`pos_abb` is LT/LG/C/RG/RT and `pos_rank == 1`. Rosters are no help: they label all five
`OL`. The 2026 charts are a **time series** from March onward, so the newest `dt` is taken —
mixing camp snapshots with current ones would invent lines nobody fields.

The `years_exp == 0` gate on rookie picks is load-bearing for the same reason
`nfl/coldstart.py` documents on the skill side: `draft_number` stays on a roster row forever,
so without it a long-tenured veteran drafted 29th reads as a first-round rookie.

**2026 output, as a sanity check:** 160 starters across 32 clubs — 126 graded, 22 moved,
9 rookies, 3 replacement. Best projected lines LAR / IND / DEN / SFO / PHI, worst GNB / NYJ /
WAS. Philadelphia resolves to Mailata–Dickerson–Jurgens–Steen–Johnson, with Mailata's 87.1
regressed to 77.1.

Tested by `tests/nfl/test_nfl_oline.py` (9 tests, offline), including an explicit test that
two lines with identical projected grades score identically however different their
continuity is.

---

## Step 5 — Stack strategy, measured

Stacking is a bet on *correlation*, so the question is what actually drives the weekly
correlation between a quarterback and his receivers. Measured on 2023-25, 270
team-season-receiver pairs, weekly DK points from `nfl.backtest.offense_actuals`, target
shares from `nfl.usage.team_target_distribution`.

### The WR1 premium does not exist

| receiver, by team target rank | mean QB correlation | median | mean target share |
|---|---|---|---|
| WR1 | **+0.367** | +0.375 | 0.245 |
| WR2 | **+0.359** | +0.379 | 0.180 |
| WR3 | +0.309 | +0.316 | 0.132 |

**A team's second receiver correlates with its quarterback essentially as strongly as its
first** — 0.359 against 0.367, and on the median 0.379 against 0.375, i.e. WR2 slightly
ahead. The drop only arrives
at WR3, and even there it is modest.

That matters because the two are not priced or owned alike: WR2 carries about
three-quarters of WR1's target share, so he is reliably the cheaper and less popular half of
the same correlation. The instinct to pay up for the WR1 in a stack is buying a correlation
that is already available at WR2. *Next step: confirm the salary and ownership gap on real
slates before treating this as a construction rule — the correlation half is measured here,
the price half is not.*

### Target concentration is not the mechanism

The intuitive story — that a concentrated passing game funnels more of the quarterback's
production into one rosterable receiver — does not hold:

    corr(team target HHI, QB-WR1 weekly correlation) = +0.134   (n = 94)
    corr(WR1 target share, QB-WR1 correlation)       = +0.204

and by tercile it is not even monotonic: flat +0.293, middle **+0.432**, concentrated +0.373.

This is the same shape as the MLB stack-identification study, which found contiguity was not
the mechanism behind stack success either. Concentration is a description of a passing game,
not a lever on correlation, and a stack rule built on it would be fitting the middle tercile.

### What this leaves

Correlation is roughly a property of *being in the passing game at all* (+0.31 to +0.37
across the top three), not of rank within it. So stack selection should be driven by the
game environment and by price, with rank as close to a tie-break — which is consistent with
where the MLB side landed, and with `PASS_GAME_POSITIONS` already counting QB/WR/TE alike.

---

## Step 6 — The two new projection sets

Both landed in `nfl/Projections/` mid-build.

**`projections (3).csv` is a fresher pull of the export the slate already uses.** Same
63-column PFF schema as `pff/projections (1).csv`, and **374 of 533 shared players carry
different point totals** (mean absolute delta 1.86), plus one team change.

`nfl/slate.py` had that path pinned by filename, so the newer file would have sat beside it
unread indefinitely. Fixed the same way the season problem was: the filename is no longer
written down. `PFF_PROJECTION_FAMILY = "projections"` resolves through
`pffdata.latest(family)`, which takes the newest by mtime — the right rule here, because a
projection export is *superseded* by its next pull rather than describing a different season
the way a grade file does.

**`projection-set-preseason-all-2026 (1).csv` is a different provider and is catalogued but
not wired in.** 6,701 rows, its own schema, and it covers IDP and kicking as well as offence.
Blending two projection sources is a modelling decision, not a plumbing one — and this
package deliberately does *not* average PFF against its own projections, because they measure
different things in different units (see `nfl/slate.py`'s note on `basis`). Wiring it in
should be a deliberate choice about what question it answers: a second opinion to disagree
with PFF, a fallback where PFF is silent, or a source for the positions PFF's export does not
cover at all.

---

## Reproducing every number in this file

The measurements above are not prose — they are in `nfl/studies.py`, one function per study:

    python -m nfl.studies                 # list them
    python -m nfl.studies persistence     # the central table (Step 2)
    python -m nfl.studies man_zone        # the effect-size sizing
    python -m nfl.studies oline           # grade portability + the draft curve (Step 4)
    python -m nfl.studies stacks          # QB-receiver correlation (Step 5)
    python -m nfl.studies all

They live in the package rather than in a scratch directory for a specific reason: **a
finding that stops being true should be discoverable.** Several constants in `nfl/oline.py`
are fitted numbers — `STAY_RETENTION`, `MOVE_RETENTION`, `DRAFT_SLOPE`, `DRAFT_INTERCEPT` —
and when the 2026 season lands they should be re-derived rather than trusted. `oline`
reprints exactly those four.

Nothing in the package imports from `studies.py`; the studies read data and print tables.
A finding that earns its way into the model is copied into the module that uses it, with
the measurement quoted beside the constant.

### One trap worth recording from writing them up

The draft-capital study joined rookies to grades on `gsis_id` and silently returned **fewer
than 20 rows instead of 68** — nflverse fills `pff_id` for only about two thirds of linemen,
so the id crosswalk drops a third of the sample on that side of the ball. It is the same
fact the data layer already documents, arriving from a different direction: on the offensive
line, name matching is the primary path and the id is the fallback, which is the reverse of
everywhere else.

---

## Step 7 — The operational loop (`nfl/naming.py`, `nfl/pool.py`, `nfl/optimize.py`)

Porting the MLB pipeline's lineup-generation half, so the NFL side can actually produce
entries rather than only describe a slate. Four new pieces plus the missing half of an
existing one.

| module | MLB twin | what it does |
|---|---|---|
| `nfl/naming.py` | `dfs.naming` | `nfl_boards/<date>/<kind>_<slate>.<ext>`, `.rN` versioning |
| `nfl/pool.py` | `dfs.pool` | the editable Lock/Exclude/Boost/Min%/Max% csv |
| `nfl/optimize.py` | `dfs.optimize` | the CLI: board → pool → solve → lineups + exposure |
| `salaries.load_dk_export` | `dfs.salaries` | the DK-file half this module said was missing |

### Football's slate vocabulary is not baseball's

MLB identifies a slate by first pitch, because a baseball night is one evening. Football
spreads across Thursday, Sunday and Monday, runs single-game showdowns most weeks, and its
Sunday blocks **overlap by design**:

    main / early / afternoon / primetime / thursday / monday / showdown-XXYYY

`early` is football's `turbo` — the fragment that **cannot be told from the main slate by
kickoff time**, because both open at 1:00PM ET. Only comparing a week's exports to each
other separates them, and `label_slates` does exactly that: most games wins the plain label,
the smaller block is the fragment, and a block starting after the big slate's last kickoff
is `afternoon` rather than a fragment of it. Verified on a synthetic full Sunday —
main / early / afternoon / showdown resolve correctly — and on the one real export on disk,
which comes back `thursday` (2 games, 8:00PM, 08/20/2026).

### Stack shapes, expressed in existing primitives

`--stack-shape "3-1"` is three pass-game players from an anchor team plus one bring-back.
It needed **no optimizer change**: the anchor's QB is pinned to the QB slot and the counts
go in as `stacks={team: n}`, which the solver already handles and already counts as QB/WR/TE
only. Verified on the real slate — every lineup had exactly three pass-game players from its
anchor with the anchor's QB, and the requested bring-back.

Rank within the passing game is deliberately absent from a shape, per Step 5: WR2 correlates
with the QB as strongly as WR1 at a quarter less target share.

### Three bugs, one of them pre-existing

**`nfl.optimize` collided with a re-exported function.** `nfl/__init__.py` re-exported the
solver as `optimize`, which shadowed the new CLI module — `from nfl import optimize` handed
back a *function*, and every attribute lookup on it failed. `python -m nfl.optimize` still
worked, because runpy loads the file directly, so the collision was invisible until
something imported it. The re-export is gone (`dfs/__init__.py` never had one), with a test
asserting the module is a module.

**An unset flag overrode the solver's default.** The CLI passed `randomness=None` through to
`optimize()`, which *replaces* `DEFAULT_RANDOMNESS` rather than deferring to it, and the
value is then compared against a number. The flag nobody set was the one that crashed the
run. Both `randomness` and `max_overlap` now resolve to the module defaults in `generate()`.

**The pool merge counted every row as hand-edited.** An empty CSV cell reads back as float
NaN; NaN is truthy, so `str(value or "")` yields the string `"nan"` and every guard sees a
non-empty value. A freshly written 66-row pool reported all 66 as carried, which turns the
one number telling you your markup survived into noise. Every cell read now routes through
`_text`. *(The MLB twin does not have this — it guards with `pd.isna` and an explicit
`{"", "nan", "none"}` skip. The bug was in the port, not the original.)*

### One improvement over the MLB original

`dfs.pool` matches saved edits on `(name, team)` and falls back to name alone, so a player
who changed clubs keeps his markup. But the fallback re-opens the hole the pair key closes:
with two same-named players and only one marked, **the unmarked one inherits the other's
lock**. Verified before the guard — marking one of two "Odell Beckham" rows locked both.

Here the fallback requires the name to be unambiguous on *both* sides: once in the saved
edits **and** once on the new slate. The team-change case still works; the collision does
not. `dfs.pool` has the same shape and was left alone, since MLB is live.

Orphaned edits — markup for a player no longer on the slate — are kept as blank rows rather
than deleted, so a decision is visible instead of silently dropped by a rebuild.

---

## Step 8 — The DK upload writer (`nfl/upload.py`)

Completes the entry path: a slate can now go from DK export to a file you upload back.

    python -m nfl.upload --date 2026-09-14 --slate main --list
    python -m nfl.upload --date 2026-09-14 --slate main --lineups 1,3,5-8

Reads the newest `lineups_<slate>.csv`, fills a DK entries export, writes
`upload_<slate>.csv`. **The downloaded template is never modified** — the filled copy is a
new file, so a bad run cannot cost the original. NFL templates live in
`nfl/nfl_dfs/dk_lineups/`, kept separate from MLB's `dk_lineups/` so the two can never be
picked up for each other.

### The two things it refuses

**A template from another week.** This is the failure worth engineering against, because it
does not look like one: names resolve against whatever template is handed over, so a stale
file produces a *full sheet of valid-looking ids* for a contest that has already finished.
The slate date is read out of the template's own embedded player list and a mismatch is an
error, not a warning.

**A player with no DK id.** DK silently ignores a roster cell holding only a name, so an
unresolved player would produce a lineup that looks filled and enters nothing. Ids resolve
against the template's player list first — an id is only valid inside the draftgroup it came
from — with the lineup row's own `DK ID` as the fallback for a bulk template that embeds no
list. A name shared by two players is matched on team **or not at all**.

### Bug found: the lineups file had a blank Slot column

`nfl.optimizer._assemble` names the roster-slot column **`Roster`**; `nfl/optimize.py` was
reading `Slot`. `.get("Slot", "")` returned an empty string for every player rather than
raising, so `lineups_<slate>.csv` shipped a Slot column that was blank all the way down —
and that is precisely the column deciding which DK cell a player is written into. Caught
only because the upload writer needed it.

There is now a test asserting `upload.SLOT_ORDER` and `optimizer.SEATS` describe the same
roster. They are two independent spellings of one fact, and if they drift, every upload is
written into the wrong columns while looking correct.

### Verified end to end

On the real Thursday export, against a synthesized entries template: entry ids preserved,
roster cells filled in DK column order as `Name (ID)`, the unused third entry left blank,
and both guards firing with the message quoted above.

---

## Step 9 — The dashboard page (`dashboards/app_pages/nfl_slate.py`)

`streamlit_app.py` had said "MLB only for now; the NFL slate board is the next page to add"
since the dashboard was built. It is added.

Navigation is now sectioned `{"MLB": ..., "NFL": ...}`, which the entry file was already
shaped for — the MLB pages are untouched and their 110 tests still pass.

**It reads outputs, not sources.** The MLB pages read cached game payloads; this one reads
`nfl_boards/<date>/{pool,lineups,exposure}_<slate>.csv`. Different source, same bargain:
it shows what was actually built rather than rebuilding it, so the page opens instantly and
can never disagree with the file you are about to upload.

| tab | what it shows |
|---|---|
| Lineups | every build, best ceiling first, with a per-lineup player inspector |
| Exposure | per player and per team, as progress bars |
| Pool | what is hand-marked, then the full pool |

Exposure gets a panel rather than a column, for the reason the MLB side learned: it once
found 54% exposure on a punt owned by 1.6% of the field, and nobody goes looking for an
exposure they do not already suspect.

Paths resolve through `nfl.naming` rather than being globbed in the dashboard. That module
already knows `.rN` versioning and that "newest" is by mtime rather than by highest number;
a second implementation would eventually disagree with it about which file is current, and
would do so silently.

### Bug found: the Arrow guard was a no-op on pandas 3

The MLB `arrow_safe` skips any column whose `dtype != object`, because the failure it
exists to fix — Arrow refusing an object column that holds numbers and `""` together, and
Streamlit then rendering **nothing at all** where the table should be — only happens on
object columns.

**pandas 3.0 reads text columns back as its own `str` dtype, not as `object`.** So the
object-only guard skipped every string column and silently did nothing, which looks exactly
like the bug it was written to prevent. The NFL copy asks whether a column is *already*
usable (numeric, boolean, datetime) instead, which covers both spellings. Caught by a test
asserting a blank-carrying `Proj` column comes back numeric; it came back as `str`.

### Testing a Streamlit page

An HTTP health check on the running server answers `200` for a page that raises on every
render, because it does not execute the page script. `streamlit.testing.v1.AppTest` does,
so the page is tested by running it:

    AppTest.from_file(<absolute path>).run()  ->  assert not app.exception

`from_file` resolves a **relative** path against the file that calls it, not the working
directory, so the path has to be built absolute from the test's own location.

14 tests, covering the readers, the empty state a first-time user sees, and the page itself.

---

## Step 10 — Exploring the data (five dashboard pages)

Step 9 added a page over the *optimizer's outputs*. That was downstream of the point: the
PFF drop was brought in to be **explored**, and exploration is what the MLB side is actually
for. So the NFL section now carries five pages over the PFF profiles, plus the Slate page.

| page | answers | source |
|---|---|---|
| Receivers | how every pass-catcher is deployed, on axes you pick | `nfl.usage` |
| Targets | how each club divides its targets — the stack view | `nfl.usage` |
| Ground game | carries, receiving work, red-zone and inside-5 touches | `nfl.usage` |
| Coverage | man/zone tendency, slot, receiver scheme splits | `nfl.pffdata` |
| Trenches | projected starting fives, each with its `basis` | `nfl.oline` |

Shared layer: `dashboards/nfl_pff.py` (cached readers) and `dashboards/nfl_charts.py`
(Altair builders in the house style).

### These are season profiles, and that is the data's shape not a compromise

PFF publishes a season of deployment per player, not a game log, so the question these pages
answer is *"how is this player used, and by whom"* rather than *"is tonight a good spot"*.
That happens to be the half of the data that predicts itself — deployment 0.82–0.95,
efficiency ~0.6, defensive allowed-rates under 0.11 (Step 2).

### The measurement is carried onto the page, not left in this document

The Coverage page **shows** what defences allowed and **warns** on the same panel that those
columns self-correlate at +0.05 (man), +0.01 (zone) and +0.10 (slot). Showing them without
that would invite exactly the matchup call the measurement rules out; hiding them would
throw away a real record of the season. The receiver splits panel points at the TPRR gap
(+0.37 to +0.51) rather than the YPRR gap (+0.19), and says why.

The same principle put `basis` on the Trenches page and continuity in a column that nothing
ranks by.

### Axes are the reader's choice

Receivers and Ground game let you pick both axes. Volume, depth, alignment and efficiency
are four separate readings of the same player, and a fixed pair silently chooses one — the
same reason the MLB matchup page carries view switches.

### Three bugs, all caught by running the pages

**`alt.Scale(domain=None)` is not "no domain".** Altair validates it and refuses, and
Streamlit reports it as a redacted "this app has encountered an error" naming nothing. It
took out four pages at once. The scale is now attached only when a domain is given.

**`idxmax` raises on an all-NaN row** — `ValueError: Encountered all NA values` — so the
alignment label died on any player with no alignment rates at all, rather than leaving one
cell blank. A blend produces such a row easily: a player in the rushing export but not the
receiving one. Rows with nothing to compare are now excluded before `idxmax`, not masked
after it.

**A shared sidebar filter collided with a page's own selector**, raising
`StreamlitDuplicateElementKey`. The Targets board filters inline instead — a sidebar control
there would sit above the club selector and appear to govern it while affecting only one tab.

### Testing pages

`streamlit.testing.v1.AppTest` runs the page script; an HTTP health check does not, and
answers `200` for a page that raises on every render. Note `AppTest` types an
`st.altair_chart` as `UnknownElement` — counting those is how you assert a chart rendered.

33 tests across `tests/test_nfl_dashboard.py` and `tests/test_nfl_pff_pages.py`, including
one that every page renders and one that none reuses a widget key.

---

## Step 11 — Filtering, pricing, stacks, defence and results (six more pages)

Eleven NFL pages now, sharing one filter bar. Step 10 explored the PFF profiles; this step
connects them to a live slate — prices, stacks, matchups and results — and adds the two
files that make that possible: a real 12-game DK salary export and a filled entries export.

| module | what it adds |
|---|---|
| `dashboards/nfl_filters.py` | the one shared bar: slate, position, team, game, price band |
| `dashboards/nfl_slates.py` | slate discovery, pricing, value, the entered-lineups reader |
| `dashboards/nfl_stacking.py` | enumerating and pricing a club's stack combinations |
| `dashboards/nfl_league.py` | defence allowed, offence produced, team results |

New pages: **Value**, **Stacks**, **Defence**, **Teams**, **Entries**.

### The measurement that shaped the Defence page

Before building a "projected allowed" column, DK points allowed per game was tested the same
way everything else here has been. Mean year-over-year correlation, 2021-25:

| position | year over year | within 2025 (wk 1-9 → 10-18) |
|---|---|---|
| RB | **+0.30** | +0.02 |
| TE | +0.19 | +0.23 |
| WR | +0.11 | +0.32 |
| QB | +0.09 | +0.36 |

So allowed-rates are weak everywhere, and the two horizons disagree about *which* position is
least bad. The spread is nonetheless real — WR allowed runs 26.8 to 37.2 DK points a game
between the tenth and ninetieth percentile.

**Both numbers are therefore on the page, side by side.** `Allowed/G` is what happened;
`Projected` is that figure regressed toward the league mean by the position's own measured
reliability. Dallas allowed 40.3 to receivers in 2025 — 8.8 above average — and projects at
32.5, about one point above. The page states the coefficient it used, and the "every defence"
tab shows both tables so the flattening is visible rather than implied.

### Stacks: a club is not one stack

A quarterback with five rosterable pass-catchers offers twenty-five distinct two- and
three-man combinations, differing by thousands in salary and tens of points in captured
target share. `stack_combinations` enumerates them all with price, projection and share;
`team_stack_summary` compares clubs on the *best available* at each size, since a stack is
something you pick rather than an average you receive.

Nothing privileges the WR1, per Step 5. Running backs never enter a stack.

### Bug found: captured share moved with the filter

`Captured share` was computed against the *filtered* shortlist, so raising the minimum-
projection slider inflated every share and a stack of all three survivors read as owning
**100%** of an offence that also throws to four other men. The denominator is now the club's
whole pass game, so the share means the same thing at every filter setting. Verified: the
same three Eagles read 0.775 at a floor of 0.0 and at 5.0.

This is the third appearance of one shape in this project — a rate whose denominator moves
with something the reader controls. The others were inside-5 carries as a share of carries
(Step 2) and the pool's "carried" count (Step 7).

### Value is two numbers

Raw points per $1,000 always favours the cheapest man on the board — every week, every
slate — so a punt at replacement level outranks every real starter. `Band rank` ranks within
a price band instead, which is the question a roster actually asks: *who is the best use of
this slot at this price*. The page shows the boxplot that makes the bias obvious, and says
why quarterbacks own the raw metric.

### Entries: duplication is the headline

Reading the filled DK export showed **7 entries carrying 1 distinct roster**. That does not
read as duplication in a contest list — it reads as seven entries — so the uniqueness count
sits at the top of the page with a warning. The same lineup entered seven times multiplies
the stake, not the number of outcomes you are exposed to.

---

## Step 12 — Exposure, NFL stack shapes, and the groundwork for portfolio

### Exposure was being read and then ignored

`nfl/optimize.py` read `Min%` / `Max%` out of the pool, printed *"N exposure caps"*, and
passed them to `generate()`, which accepted the argument and never used it. `nfl/optimizer.py`
had no exposure machinery at all — its own docstring said so. **That is worse than not having
the feature**: the CLI reported honouring a constraint it was discarding.

The pacing is now lifted from `dfs.optimizer`, which is where its tuning was learned:

- **Caps** bar a player from the next solve once he is at his limit.
- **Minimums are paced, not deferred.** A player wanted in 30 of 50 lineups is in roughly 3
  of the first 5. The MLB comment documents both failure modes this avoids — ceil makes
  every minimum due on lineup one, floor makes them all due on the last — and rounding to
  nearest leaves slack at both ends.
- **State threads across solves.** The shape loop solves one lineup at a time, so running
  counts are passed into every call. Without that each solve believes it is the whole set.

New: `--team-exposure "PHI:30-70%,CIN:0-25%"` and `--stack-at N`. Verified: a cap of 3 in 20
held at exactly 3, a minimum of 10 in 20 hit exactly 10, and a team floor of 6 in 12 stacked
exactly 6.

**Bug found while wiring it:** a team-exposure *minimum* could not be met, because the anchor
rotation walks the top-N clubs by projection and the club with the floor was not among them.
An explicit floor is a decision you made; being outside the projection shortlist is precisely
the reason the rotation would not have reached that club anyway. Floors now reach into every
club the slate can anchor.

### Stack shapes now match how football is actually stacked

    A        pass-game players from the anchor club, quarterback included and pinned
    A-B      ...plus B from the other side of that same game — the bring-back
    A-B+C    ...plus a C-man secondary stack from a *different* game

Three numbers because a football stack is three decisions. `A` is the correlation. `B` is the
hedge that pays in a shootout rather than a blowout. `C` is diversification — a second
correlated block so one game going quiet does not take the lineup with it, which is why it
is forced to come from another game.

Refused rather than silently mangled: a one-man "secondary stack", and any shape whose
players cannot fit alongside the RB slots and a DST.

### Portfolio groundwork: the blocker is ownership, so that is what was built

Portfolio selection cannot exist without a field model — choosing which 20 of 150 lineups to
enter *is* a bet about what everyone else is on. Two new modules, both deliberately honest
about being uncalibrated:

**`nfl/ownership.py`** — per-position softmax over projected points.
`estimate_ownership` verified to sum correctly: QB 100, RB 235, WR 350, TE 115, DST 100,
matching 1/2/3/1/1 roster slots plus a split FLEX. `OWNERSHIP_VERSION = 0` means never
calibrated, and `calibrate()` raises with the fitting order rather than pretending.

**`nfl/results.py`** — reads DK contest-standings exports. Written *before* anything needs
it, because results only accumulate if something checks them from week one; a reader written
in week five discovers weeks one to four were saved wrong. `python -m nfl.results` reports
what is on disk and whether it is enough to calibrate (needs 3+ distinct slates — fitting a
field to one week fits that week's chalk).

### Three bugs in the ownership model, each caught by looking at the output

**Percentiles cannot concentrate a field.** The first version copied MLB's softmax-over-
percentile-appeal. Percentiles are uniform by construction, so with 88 priced quarterbacks
sharing a 100-point pool the top one came out at **2.5%** on a slate where real chalk is
thirty to fifty — and no temperature fixed it, because the input carried no scale. The
softmax now runs on projected points, where the starter-to-backup gap is large in the
exponent. Chase now reads 65%, Gibbs 58%, Taylor 41%.

**A missing projection was filled with the median.** That told the model a player nobody
projects is *averagely* attractive — half a DK slate is third-stringers the field will never
roster. The same "a blank is not an average" trap this project keeps finding.

**A whole position vanished.** PFF's projection export carries no defenses, so keying on
`Proj` alone dropped every DST — a position the field spends a full 100 ownership points on.
DK's own season average stands in where a whole group has no projection.

And leverage was unbounded as ownership approached zero: the board filled with backup
quarterbacks at 0.01% owned scoring 280 on a scale where a real play sits near 0.5. Floored
at 0.5% ownership, which is also true of live contests.

### A testing note worth keeping

`pytest ... 2>&1 | tail` reports **tail's** exit code, not pytest's. A run with 7 failures
reported "exited with code 0". Redirect to a file and check `$?` instead.

---

## Step 13 — Does any of this improve the projection?

Twelve steps of description. This one asks the only question that settles whether it was
worth it, on the model's own held-out harness: **fit on 2022-24, score 2025, compare against
each player's own season average.**

Baseline reproduced exactly: MAE 4.617 against 4.938, rho 0.682 against 0.635.

### Finding 1: the model is only good at players it already knows

MAE by games of history on held-out 2025:

| games | n | model | season average | gain |
|---|---|---|---|---|
| 0-4 | 210 | 4.109 | 4.029 | **−0.081** |
| 4-8 | 358 | 4.067 | 4.043 | **−0.024** |
| 8-16 | 672 | 3.697 | 3.669 | **−0.028** |
| 16+ | 3,939 | 4.851 | 5.284 | **+0.434** |

**Below sixteen games the model loses to a season average**, across a quarter of all rows.
Everything it is worth comes from established players.

### Finding 2: PFF deployment cannot fix that, for a structural reason

The obvious move was to use the PFF deployment signal — the stable half of the data, 0.82 to
0.95 year over year — as a better prior for those players. It cannot be done, and the reason
is worth recording so nobody tries again:

| games of history | rows | with a prior-season PFF profile |
|---|---|---|
| 0-8 | 506 | **0 (0%)** |
| 8-16 | 615 | 54 (8.8%) |
| 16+ | 3,494 | 2,692 (77%) |

**PFF covers 5% of the bucket where the model is weak.** A player with no NFL games has no
prior-season PFF profile either; the two conditions are very nearly mutually exclusive. On
the 54 usable rows a blend moved MAE from 3.981 to 3.913, which on n=54 is nothing.

### Correction: the "QB drift" was an artefact

An earlier bucketed read showed quarterback bias growing +0.286 → +0.887 → +1.057 across
the season and looked like a trend worth chasing. Week by week it oscillates between −2.20
and +2.59 with a slope of **+0.055 points per week** on n≈35 a week. Noise. The apparent
drift was the bucketing.

### Finding 3: cold start was built for this and was never connected

`nfl/coldstart.py` produces a per-player opportunity prior from prior-season usage, draft
capital and depth rank. `project_week` shrinks toward a per-*position* prior. Both are called
"priors", both are passed as `priors=`, they are different shapes, and nothing ever bridged
them — cold start only ever decorated the report board through `nfl/slate.py`.

`coldstart.projection_priors()` is that bridge, and `project_week(player_priors=...)`
overrides **only the three volume terms**. Efficiency stays positional: cold start projects
opportunity, and yards per carry self-correlates at +0.27.

### Result

| games | n | before | after | Δ | vs season average |
|---|---|---|---|---|---|
| 0-4 | 210 | 4.109 | **3.966** | +0.144 | −0.081 → **+0.063** |
| 4-8 | 358 | 4.067 | **3.969** | +0.098 | −0.024 → **+0.074** |
| 8-16 | 672 | 3.697 | **3.638** | +0.059 | −0.028 → **+0.031** |
| 16+ | 3,939 | 4.851 | 4.848 | +0.002 | +0.436 unchanged |

All three weak buckets now beat the baseline; the strong bucket is untouched, which is what
a targeted fix should look like. Overall MAE 4.617 → **4.595**, gain 0.321 → 0.343, bias
0.267 → 0.221.

**Honest limits.** The low buckets are 210, 358 and 672 rows and this is one held-out season.
The overall movement is small because the affected rows are a quarter of the sample.

### Closing the headroom the same session

`roster_universe` returned **438** players for 2025 while **610** actually played. The cause
was not the schema trap I first assumed — it was the status filter.

The roster holds **971** skill players and **609 of the 610** who played. But status is a
season-**end** snapshot, and `startswith(("ACT", "A"))` keeps only the 438 marked `ACT`:

    ACT 438   DEV 164   CUT 151   RES 131   INA 83   RET 5

A player who started eight games and was cut in December reads `CUT`. **197 players who
actually played were filtered out** — a third of everyone who took a snap.

That filter is *right* for a live board and *wrong* for anything historical, so it is now
`active_only=True` by default (the board path in `nfl/cli.py` is unchanged) and
`projection_priors` passes `False`. The asymmetry that makes this safe: `project_week`
builds its own universe from box scores and only *looks up* a prior, so a player in the map
who should not be rostered can never be added to a projection by it — he is simply never
asked about.

### Final result, held-out 2025

| games | n | model | + priors (438) | + priors (971) | vs season average |
|---|---|---|---|---|---|
| 0-4 | 210 | 4.109 | 3.966 | **3.845** | −0.081 → **+0.184** |
| 4-8 | 358 | 4.067 | 3.969 | **3.947** | −0.024 → **+0.096** |
| 8-16 | 672 | 3.697 | 3.638 | **3.620** | −0.028 → **+0.049** |
| 16+ | 3,939 | 4.851 | 4.848 | 4.846 | +0.438 unchanged |

Widening the universe roughly **doubled** the gain in the weakest bucket, which is what the
coverage argument predicted and is the reason to measure coverage rather than assume it.

Overall: MAE **4.617 → 4.584**, gain over a season average 0.321 → **0.354**, rho 0.6815 →
**0.6835**, bias 0.267 → **0.200**.

**Honest limits.** The low buckets are 210, 358 and 672 rows, and this is one held-out
season. The overall movement is modest because the rows it affects are a quarter of the
sample — but the direction is consistent across every bucket and the strong bucket is
untouched, which is what a targeted fix should look like rather than a rebalance.
