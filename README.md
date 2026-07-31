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
python -m dfs.optimize --date 2026-07-27 --n 20 --objective ceiling --stack-shape 4-3
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
| `--stack-shape "4-3"` | Explores that shape across the best stacking teams |
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

## Current Notes

- The report pipeline is now callable from the command line and no longer auto-runs a hardcoded game when imported.
- Report date is threaded into the major season, lineup, recent-form, transaction, and team-performance lookups.
- Probable pitcher lookup uses the selected game's StatsAPI data first, with the current MLB probable-pitchers page only as a fallback.
- The current Stuff+ score is a local pitch-quality proxy, not a calibrated leaguewide Stuff+ model.
