# MLB Scouting Report Generator

Generate PDF scouting reports for MLB games using Statcast, MLB StatsAPI, FanGraphs-style leaderboards from `pybaseball`, and local plotting.

## Setup

> **Always call `.\.venv\Scripts\python.exe` explicitly, or activate the venv first.**
> Plain `python` can resolve to a system install ahead of `.venv` (check with
> `Get-Command python -All`). Both entry points now detect this and name the real problem
> rather than failing on whichever dependency happens to import first.

The project uses a single environment, `.venv` (Python 3.14). Create it with the Windows
Python launcher:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If VS Code picks the wrong interpreter, run **Python: Select Interpreter** and choose:

```text
.\.venv\Scripts\python.exe
```

When running without activating first, call the interpreter explicitly —
`.\.venv\Scripts\python.exe ...` — rather than plain `python`, which may resolve to
another Python on the machine. A wrong interpreter typically shows up as
`No module named 'pyarrow'` when reading cached report data.

## Usage

Generate a report by selecting from the schedule interactively:

```powershell
python scouting_report.py --date 2025-06-01
```

Generate a report by schedule number:

```powershell
python scouting_report.py --date 2025-06-01 --game-number 3
```

Generate a report by matchup:

```powershell
python scouting_report.py --date 2025-06-01 --away-team CIN --home-team CHC
```

Markdown is the default output because it is the most repeatable and easiest to inspect.

Optional PDF export:

```powershell
python scouting_report.py --date 2025-06-01 --away-team CIN --home-team CHC --format pdf --include-plots
```

Reports are written to `scouting_reports/`. Plot images are written to `plots/` only when `--include-plots` is used.

## Cache

Report generation caches repeated API responses and DataFrame pulls under `.cache/`.
This makes repeated runs much faster while iterating on the report layout or logic.

To force fresh data for one command:

```powershell
$env:SCOUTING_REPORT_DISABLE_CACHE = "1"
python scouting_report.py --date 2025-06-01 --away-team CIN --home-team CHC
Remove-Item Env:\SCOUTING_REPORT_DISABLE_CACHE
```

To clear cached data, delete the `.cache/` folder.

## DFS Recommender

Turns the cached scouting data into a DraftKings Classic board with a GPP lean. It reads
the payloads the report pipeline already wrote to `.cache/report_data/`, so a full slate
builds in seconds and never re-pulls from an API.

```powershell
python -m dfs.cli --date 2026-07-26
```

Drop the contest's `DKSalaries.csv` into `dfs_daily_files/` first. Salaries are optional —
without them the board still ranks on projection and ceiling, it just can't compute value
per dollar. The salary file also defines the slate: anything DK didn't price is excluded,
and players listed `IL`/`OUT` are dropped.

**Multiple slates in a day.** Exports are identified by their contents, not their name, so
rename them freely — `DKSalaries_2026-07-27_main.csv`, `..._early.csv`. Then
`--list-slates` shows what you have (games, start-time window, player count) and `--slate
main` selects one. If several match and you name none, the tools stop and list them rather
than pricing your board off the wrong contest.

The board cross-checks the DK slate against the report cache and refuses to pretend a
missing game doesn't exist — if a priced game has no cached report, it says so and prints
the command to generate it. Any team without a report is absent from every table, so this
warning should be cleared before trusting the board.

When MLB has not posted a probable pitcher yet, the report bails. DK's salary file often
confirms starters hours earlier, so you can force one:

```powershell
python scouting_report.py --date 2026-07-27 --away-team CHC --home-team STL `
    --home-pitcher "Matthew Liberatore" --format both
```

`--away-pitcher` / `--home-pitcher` accept an MLBAM id or a name.

Outputs land in `dfs_boards/`:

Everything for one night lands in `dfs_boards/<date>/`, named `<kind>_<slate>`:

- `board_<slate>.md` — pitchers, team stacks, hitters by position, tier summary
- `slate_<slate>.csv` — every player with full projection detail
- `stacks_<slate>.csv` — team stacks ranked for GPP

The slate label keeps two contests on the same date apart. Re-running writes `.r2`
rather than overwriting; pass `--overwrite` to replace. `python -m dfs.files --date <date>`
lists a night and explains each file.

### How to read it

- **Proj** is a mean DK-point expectation; **Ceiling** is roughly a 90th-percentile
  outcome and **Floor** the 25th. Both bands are fit from actual results, not assumed.
- **Bust%** is the chance of scoring 3 or fewer points.
- **Value** is points per $1,000. Around 2.0 is the usual break-even for hitters.
- **Edge** is how far the projection ranks above the player's DK season average — the
  gap between tonight's matchup and what the field prices on.
- **Why** lists supporting factors first; anything after `[risk]` is a reason for caution.

**Tier** answers "is this priced well for a tournament": **Core**, **Value**,
**Leverage** (matchup much better than the season line suggests), **Risk** (strong number,
unconfirmed lineup), **Fade**.

**Role** answers a different question — "what kind of play is this": **Ceiling**,
**Floor**, or **Ceiling+Floor**. A player can be a strong ceiling play and a bad floor
play at once, so the two are kept separate and get their own board sections. `GPP` scores
the tournament view (ceiling per dollar); `CASH` scores the floor view (floor per dollar
and bust avoidance). Pitchers and hitters are ranked within type — their scores are not
comparable across types.

### Optimizer

```powershell
python -m dfs.optimize --date 2026-07-27 --n 20 --objective ceiling --stack-shape 5 --stack-teams 0
```

Exact MILP (scipy) over the priced slate. Enforces the DK Classic roster (2 P, C, 1B, 2B,
3B, SS, 3 OF), the $50,000 cap, max 5 hitters per team, and the two-game minimum.
Multi-position eligibility is modelled as a true assignment, so a 2B/3B player fills one
slot rather than satisfying both.

**Defaults are tuned for tournaments, not for maximum projected points.** `--max-overlap 6
--randomness 0.20` yields 63 unique players across 10 lineups (1.4 mean overlap) where the
DK-minimum settings give 24 (6.5 overlap), at the cost of ~8 projected points per lineup.
On the 2026-07-27 review this beat every other configuration tested. That is one slate —
`python -m dfs.review --date <date> --sweep` re-tests it against any night's results. For
a single best lineup, pass `--randomness 0`.

| Flag | Effect |
|---|---|
| `--objective ceiling\|proj\|floor` | ceiling = GPP, floor = cash, proj = balanced |
| `--lock` / `--exclude` | Comma-separated names; accent- and case-insensitive |
| `--stack "CWS:4,HOU:3"` | Explicit team stacks |
| `--stack-shape "5"` | Requires a stack of that shape; the solver picks the team |
| `--stack-teams N` | Teams a shape may stack, best first (default 6; `0` = every team) |
| `--max-overlap N` | Cap shared players between lineups (default 6) |
| `--randomness` | Jitters the objective (default 0.20); `0` for the single best lineup |
| `--min-proj` / `--max-bust` | Prune the pool before solving |
| `--save-config` | Persist settings to `dfs_daily_files/optimizer.json` |

Settings in `dfs_daily_files/optimizer.json` are reused on later runs; command-line flags
always override the file. Lineups are printed and written to `dfs_boards/<date>/lineups_<slate>.csv`.

**Stack correlation.** Summing individual ceilings treats hitters as independent, so an
unconstrained optimizer spreads eight bats across eight games. A bonus scaling with stack
size (applied only to `ceiling`/`proj`, since correlation cuts both ways and never helps a
floor build) makes it choose real 4-4 and 5-3 stacks on its own. `--no-stack-bonus` disables it.

**Editable pool file.** Export the slate, mark it up in Excel, feed it back:

```powershell
python -m dfs.optimize --date 2026-07-27 --write-pool
# edit Lock / Exclude / Boost in dfs_boards/2026-07-27/pool_main.csv
python -m dfs.optimize --date 2026-07-27 --pool --n 20
```

| Column | Accepts | Effect |
|---|---|---|
| `Lock` / `Exclude` | `1`, `y`, `yes`, `x`, `true` | Force in / never use |
| `Boost` | `1.25`, or a range `1.1-1.3` | Multiplies the objective. A range is redrawn per lineup |
| `Min%` / `Max%` | `30`, `30%`, or `0.3` | Share of lineups the player may appear in |

Pool entries combine with `--lock`/`--exclude` rather than replacing them.

**Boost vs exposure — they do different jobs.** A boost only raises a player's value, so a
boosted player tends to appear in *every* lineup; the range varies how strongly they are
favoured, not whether they show up. To actually spread someone across a set, use `Max%`.
Verified on a 10-lineup run: `Max% 40` produced exactly 4 lineups, `Min% 60` exactly 6.

Minimums are met by forcing the shortfall into the remaining lineups, so set them with the
lineup count in mind — `Min% 60` across 5 lineups means 3, not a soft target.

### Uploading lineups to DraftKings

Download the contest's upload file from DK — either the bulk template or the entry export —
and drop it in `dk_lineups/`. Then select which of the generated lineups to send:

```powershell
python -m dfs.upload --date 2026-07-28 --list             # what was generated
python -m dfs.upload --date 2026-07-28 --lineups 1,3,5-8  # write those eight
python -m dfs.upload --date 2026-07-28                    # all of them
```

The result is `dfs_boards/<date>/upload_<slate>.csv`, ready to upload as-is. Your template is
read, never written. The optimizer can do it in the same run with `--upload`, optionally
with a selection:

```powershell
python -m dfs.optimize --date 2026-07-28 --n 20 --upload 1-5
```

Both DK layouts work. The **bulk template** takes one new lineup per row (DK's limit is
500). The **entry export** overwrites lineups on entries you already hold, so the lineup
count is capped by the number of entries, and `--contest "Chin Music"` narrows it to one
contest when the file spans several.

Two things are checked before anything is written, because both produce a file that looks
fine and fails at DK:

- **Player ids are taken from the template's own player list**, not from the ids recorded
  when the lineups were built. DK numbers players per draft group, so a template from a
  different contest renumbers everyone, and DK rejects the upload as a whole without saying
  which player was wrong.
- **The template's slate date must match.** Last night's lineups match tonight's names
  perfectly and would otherwise upload without a complaint.

### Postponed games

The optimizer drops teams whose games are called off. This needs checking on two paths,
because DK's export is a snapshot:

- If you re-download after the postponement, DK writes `Postponed` into `Game Info` while
  leaving the players in the file at full salary — nothing else marks them unplayable.
- If your file predates the postponement, it carries no signal at all. That is the case
  that actually bites, so the live MLB schedule is checked as well.

A team is only dropped when *every* game it has that day is off — on a doubleheader where
the opener was played and the nightcap called, its hitters are still live for the game that
happened. The board and the optimizer both print what was removed. If the schedule lookup
fails, that is reported rather than passed off as "nothing is postponed"; re-download DK's
file, which marks called games itself.

### Re-rendering an already-built report

```powershell
python scouting_report.py --date 2026-07-27 --all-games --format xlsx --from-cache --fast
```

A plain `--from-cache` re-render spends ~14s per game refreshing boxscores, sweeps,
schedule spot and umpire against <1s of actual rendering. `--fast` skips all of it and
renders purely from cache — a 12-game slate takes ~45s instead of several minutes. The
only things that go stale are those four blocks, none of which move pregame.

**A plain re-render will not show confirmed lineups.** It replays whatever lineup was
cached, which is usually a `Fallback` guess made before lineups posted. To pick up
confirmations:

```powershell
python scouting_report.py --date 2026-07-27 --all-games --format xlsx --from-cache --fast --refresh-lineups
```

That re-pulls both lineups (~1s per team) and rebuilds the platoon splits, hot/cold,
offense index and hitter composite. Arsenal, similar-pitcher and BvP columns are carried
over — they need the similarity engine — so a player newly in the lineup simply has no
values there rather than wrong ones.

**Re-running it is cheap.** The posted cards are read before anything is rebuilt, and when
both already match the cache the payload is kept untouched and only the render happens —
~20s per game down to ~1.4s. So there is no cost to running it again at each stage of the
evening; it does the work on the run where the card actually moved and skips the rest.
Anything ambiguous — a partly-posted card, a `Fallback` cache, a doubleheader — rebuilds.
`--force-lineup-refresh` rebuilds unconditionally.

**A game cached before the arsenal K-edge landed needs one non-`--fast` re-render.** The
batter arsenal tables gained a `K Edge` column, which is the only arsenal number the DFS
pitcher projection reads (see `docs/arsenal_study.md`). `--from-cache` without `--fast`
rebuilds it once per cached game; with `--fast` the rebuild is skipped and the factor
simply never fires, which looks the same as a neutral matchup.

**Follow it with a plain re-render to make the highlighting consistent.** DFS tints are
rebuilt from the whole slate at render time, but `--refresh-lineups` refreshes one game at
a time — so the first game rendered is tinted against a pool in which every later game
still holds its pre-lineup guess. A second `--from-cache --fast` pass costs ~45s and
re-tints everything against the finished slate:

```powershell
python scouting_report.py --date 2026-07-27 --all-games --format xlsx --from-cache --fast
```

### Value highlighting in the Excel report

Player names in the `.xlsx` scouting report are tinted on two independent axes,
deliberately outside the report's red-amber-green percentile heat so they read as a
different kind of statement. A legend sits under the title.

| Tint | Axis | Meaning |
|---|---|---|
| **Teal** | Raw projection | Stud / Strong Bat — expensive, but the bat you build around |
| **Indigo** | Points per $1k | Elite Value / Value — the efficiency play that frees salary |
| **Violet** | Both | Top projection *and* priced well |

Points-per-dollar on its own is a cheap-player detector: at a 55th-percentile cut it tinted
names averaging $3,150 and missed 7 of the 10 best-projected hitters. The raw-projection
axis exists to catch the studs that value scoring structurally hides.

Thresholds are high (92nd / 80th percentile) because the tinted set is the *union* of both
axes — a loose cut colours most of a lineup and stops meaning anything. As shipped it lights
roughly a third of a priced slate, 3-5 names per nine-man lineup.

Highlighting turns on automatically when a priced slate exists for the report date;
`--no-dfs-highlight` disables it. Because the index
is built from cached games, generate the slate first and then re-render for full coverage:

```powershell
python -m dfs.cli --date 2026-07-27
python scouting_report.py --date 2026-07-27 --all-games --format xlsx
```

### Historical salary and position analysis

Analyze the archived DFS boards against the matching DraftKings contest results:

```powershell
python -m dfs.price_analysis --from 2026-07-27 --to 2026-08-08
```

The command keeps overlapping slates separate when measuring whether a position was cheap
or weak, but counts each player only once per day for price history and correlations. It
writes a Markdown report plus reusable CSV tables under
`dfs_analysis/price_<from>_<to>/`: player price history and changes, outcome-defined
performance tiers, empirically discovered salary breakpoints, position pay-up economics,
slate-level position depth, weak-position slates, pitcher start-to-start changes, data
coverage, and rank correlations. Missing contest results stay missing rather than becoming
zeroes. Use `--slate main` or `--min-sample 20` to narrow or tune the analysis.

Salary tiers are not fixed $1,000 buckets. A shallow regression tree finds the salary cuts
that best separated actual scoring while requiring meaningful samples on both sides.
Separately, the outcome-origin table starts with Elite/Solid/Useful results and reports the
salary distribution they came from. Position pay-up tables show both absolute points per
$1K and marginal points gained per additional $1K. The depth analysis counts upper-tier
options within each slate and compares shallow, typical and deep pools.

The position tables use DK roster eligibility, so a multi-position hitter contributes to
each slot he could fill. Findings are descriptive; with a short archive, use the reported
sample counts and 95% margins before turning a difference into an optimizer rule.

The same command now runs a full-lineup opportunity-cost backtest. For each hitter slot it
solves two otherwise-unconstrained lineups: one that must use the slate's upper-quartile
salary pool at that exact roster position and one that cannot. The comparison reports what
the extra position salary bought, what was lost across the other nine seats, and the net
projection, ceiling and actual-score change. Shared players cancel in the actual comparison;
every player who differs must have a known result, so missing results are never zero-filled.

To refresh only the strategy layer or choose a cash-style objective:

```powershell
python -m dfs.position_strategy --from 2026-07-27 --to 2026-08-08
python -m dfs.position_strategy --from 2026-07-27 --to 2026-08-08 --objective floor
```

`replacement_scarcity.csv` measures projection-based replacement level, viable alternatives,
and whether the position is deep, weak/empty, or has one standout. These labels are
walk-forward: only earlier slates define the current slate's historical percentiles.
`walkforward_tiers.csv` applies the same discipline to learned salary tiers.
`daily_strategy_card.csv` combines current-slate scarcity, the paired optimizer result, and
prior comparable actual pairs into an actionable pay-up/value/flexible lean.

### Backtest

```powershell
python -c "from dfs.backtest import run_backtest, summarize; print(summarize(run_backtest(['2026-07-24','2026-07-25'])))"
```

Pulls final box scores, converts to DK points, joins on MLBAM id. Reports bias, MAE,
rank correlation, top-vs-bottom quintile separation, and whether the ceiling/floor bands
hit their stated percentiles.

Over 823 hitter-games and 95 starts the model separates top from bottom quintile by
**+3.96 points for hitters** and **+14.18 for pitchers**. Pitchers rank far better
(spearman 0.35) than hitters (0.22), which is expected — a hitter's four plate
appearances are mostly noise.

**The single most useful finding: 27% of hitter-games score zero and 44% score 3 or
fewer, regardless of how good the matchup looks.** There is no such thing as a safe MLB
hitter in one game. Real floor plays are pitchers.

The model is matchup-adjusted expectation, not prediction. League baselines, factor
damping, and the empirical floor/bust fits live at the top of `dfs/projections.py`;
scoring lives in `dfs/scoring.py`.

### Walk-forward evaluation

`dfs.backtest` grades every date with today's code against payloads that have since been
refreshed, which makes it a quick check rather than evidence. `dfs.evaluate` is the honest
version: for each target date the floor/bust calibration is refit on **strictly earlier
dates only**, and the result is scored on that date.

```powershell
python -m dfs.evaluate                    # every cached date
python -m dfs.evaluate --from 2026-07-25  # a window
```

It adds quantile (pinball) loss, CRPS, probability-integral-transform diagnostics, Brier
scores with skill against the base rate, and breakdowns by salary tier, batting order,
handedness matchup, slate size, and confirmed-versus-projected lineup status. Output is
also written as JSON under `docs/benchmarks/`.

What it found on the current model (2,318 player-games over 9 dates):

- The **ceiling band is honest** — exceeded 10.4% of the time against a 10% target.
- The **per-player bust number barely beats the base rate** (Brier skill 0.032). `Bust%` is
  a linear function of `Proj`, so it carries little player-specific information.
- **Unconfirmed lineups are over-projected by 1.5 points.** Confirmed hitters have a bias of
  −0.001; projected ones −1.489.
- **Hitter distributions are too narrow at the bottom** — real blanks happen more often than
  the published bands imply.

## Slate snapshots

`.cache/report_data/*.pkl` is rewritten in place by `--refresh-lineups`, so re-reading a
past night gives you that payload *as it stands now*. Confirmed lineups and late scratches
leak backwards, and a review ends up grading projections that never existed at lock.

```powershell
python -m dfs.snapshot --date 2026-08-01 --stage confirmed
python -m dfs.snapshot --date 2026-08-01 --verify
```

A snapshot freezes the projections, the DK export, per-game weather / probables / batting
orders, model and code versions, a data cutoff, and a sha256 of every payload it came from.
Stages are `morning`, `t-2h`, `confirmed`, `final`; re-taking one writes `.r2` beside it and
never replaces anything. `dfs.review` reads the newest snapshot automatically and says so.

## Candidate pools and correlated simulation

Generation is now separable from selection, and lineups can be scored against simulated
slates rather than summed point projections.

```powershell
python -m dfs.candidates --date 2026-08-01 --slate main --n 1000 --seed 7
python -m dfs.simulate --date 2026-08-01 --slate main --sims 20000 `
    --candidates dfs_boards/2026-08-01/candidates_main
```

`dfs.candidates` builds hundreds or thousands of legal lineups with their metrics attached,
at ~0.03 s each against the optimizer's 0.40 s. `dfs.simulate` draws whole nights through a
hierarchy of shared shocks — slate, game, team, batting-order block, player — where a team's
runs are drawn once and *allocated* across its nine hitters, so teammates compete for one
pool and the opposing starter's earned runs come out of that same pool. Nothing predicts DK
points directly; components are drawn and scored through `dfs/scoring.py`.

Measured correlation structure on a 15-game slate: teammates **+0.30**, pitcher versus the
hitters he faces **−0.31**, different games **+0.01**. None of it is imposed as a
coefficient.

The point of all this: on 2026-08-01, ranking 1,000 candidates by simulated 99th percentile
disagreed with ranking them by summed ceiling **by a mean of 166 places**.

Neither module changes any existing command. `dfs.optimize` behaves exactly as before.

## Opponent field, contest EV, and joint portfolios

The 10 DraftKings contest exports in `dk_results/` carry the full construction of **25,671
real field lineups**, which makes the field directly observable rather than assumed. What
they show is that a per-player ownership model cannot work for duplication:

```
P(teammate B | A rostered) / P(B)  =  2.62x
P(B on another team  | A) / P(B)   =  0.75x
pitcher pairs vs independence      =  1.01x
```

Teammates are 2.6x more likely to be rostered together than independence implies. `dfs.field`
builds opponents the way the exports show real entrants building them — primary stack (54%
are five-stacks), secondary conditional on it (5-2 is 27% of all entries), fill, two
pitchers — and validates itself back against measured stack shapes, ownership, duplication,
and the real score distribution.

**The production shape** — the optimizer generates a wide constrained pool, the portfolio
picks which ones you enter, and upload is unchanged:

```powershell
python -m dfs.optimize  --date 2026-08-01 --n 100 --pool --overwrite
python -m dfs.portfolio --date 2026-08-01 --n 20 --from-optimizer --write-lineups
python -m dfs.upload    --date 2026-08-01
```

`--from-optimizer` keeps everything the optimizer enforced — the editable pool file,
exposure minimums, ranged boosts, explicit stacks, the conflict tax and the started-game
filter — and adds correlation-aware selection on top. `--write-lineups` writes the chosen
set back in the optimizer's own CSV format, so `dfs.upload` cannot tell the difference.

Measured on 2026-08-01, 20 entries chosen from 100:

| source | effective lineups | best-of-set mean |
| --- | ---: | ---: |
| optimizer 100 → portfolio 20 | **3.73** | 147.8 |
| optimizer 100 → top 20 by ceiling | 2.25 | 145.8 |

Or explore the slate without entering anything:

```powershell
python -m dfs.field   --date 2026-08-01 --slate main --contest twenty-max
python -m dfs.contest --date 2026-08-01 --slate main --contest twenty-max
```

`dfs.contest` scores every candidate against that field through a payout curve. Duplication
needs no separate model: identical lineups tie, and DK's tie rule already splits the prizes
for the positions they occupy.

`dfs.portfolio` picks the whole entry set jointly. The headline result:

| method | best-of-set mean | effective lineups | distinct stacks |
| --- | ---: | ---: | ---: |
| joint portfolio | **155.6** | **5.91** | 15 |
| highest P(win) | 151.4 | 3.88 | 7 |
| random | 146.2 | 6.06 | 11 |
| **highest ceiling** (current default) | 143.4 | **2.07** | 4 |

**Ranking twenty lineups by summed ceiling produces a set worth 2.07 independent bets.**
`effective_lineups` is `n / (1 + (n-1)r)` computed from simulated score correlations — what
`--max-overlap` has always been approximating by counting shared players.

> **Calibrated, with a documented residual.** The contest probabilities were traced against
> 7 contests whose real standings are on disk. The model believed our candidate pool beat
> the field by +1.3 points when it had actually *lost* to the real field by −5.3 — the pool
> is generated for coverage so its average member is deliberately mediocre, and the
> simulated field runs ~5 points soft. One parameter (`FIELD_EDGE_CORRECTION`) corrects that
> gap. After it, **P(cash) is calibrated to within 5%**; P(top 1%) and P(win) still run
> ~1.5× high. `docs/benchmarks.md` §7A has the whole investigation, including the several
> things that did not work. Portfolio comparisons are unaffected — every method there is
> scored on the same simulated slates.
>
> An earlier claim here — that simulated mean out-ranks summed ceiling (+0.214 vs +0.157) —
> **was one slate and does not replicate.** Across 7 contests, within-contest discrimination
> is +0.177 for raw projection, +0.170 for summed ceiling and +0.150 for simulated mean, all
> within noise. The simulator earns its place by producing *distributions*, which duplication
> and portfolio correlation require and a point estimate cannot supply — not by ranking
> players better.

None of these modules changes an existing command. `dfs.optimize` behaves exactly as before.

## Documentation

| File | What it holds |
| ---- | ------------- |
| [`CHEATSHEET.md`](CHEATSHEET.md) | Every command and flag |
| [`docs/current_architecture.md`](docs/current_architecture.md) | Module map, data flow, measured profile, known correctness gaps |
| [`docs/simulation_and_portfolio_plan.md`](docs/simulation_and_portfolio_plan.md) | Where the optimizer is going, and in what order |
| [`docs/benchmarks.md`](docs/benchmarks.md) | Every measured result, including the changes that did not help |

## Tests

```powershell
python -m pytest tests/ -q
```

354 tests covering DK scoring, roster validity, stack and exposure and overlap constraints,
doubleheaders, started-game filtering, snapshot immutability, simulation reproducibility and
correlation behaviour, ownership normalisation, field legality and structure,
payout tie-splitting, portfolio diversification, the optimizer→portfolio bridge,
started-game filtering, and the calibration metrics. All seeded;
`integration`-marked tests skip themselves on a clean checkout.

## Current Notes

- The report pipeline is now callable from the command line and no longer auto-runs a hardcoded game when imported.
- Report date is threaded into the major season, lineup, recent-form, transaction, and team-performance lookups.
- Probable pitcher lookup uses the selected game's StatsAPI data first, with the current MLB probable-pitchers page only as a fallback.
- The current Stuff+ score is a local pitch-quality proxy, not a calibrated leaguewide Stuff+ model.
