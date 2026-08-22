# DFS Operations Cheatsheet

Use this as the operational runbook for daily MLB scouting reports, DraftKings boards,
snapshots, lineup creation, uploads, post-contest review, and the price/tier audits.

## Quick navigation

- [Daily run: start to finish](#daily-run-start-to-finish)
- [Daily price and tier audit](#daily-price-and-tier-audit)
- [Snapshot scheduler](#snapshot-scheduler)
- [Salary files and multiple slates](#salary-files-and-multiple-slates)
- [Scouting reports](#scouting-reports)
- [DFS board](#dfs-board)
- [Optimizer and portfolio](#optimizer-and-portfolio)
- [Upload and late swap](#upload-and-late-swap)
- [StreamFinder priority file](#streamfinder-priority-file)
- [Post-contest review](#post-contest-review)
- [Research and model evaluation](#research-and-model-evaluation)
- [Output locations](#output-locations)
- [Troubleshooting](#troubleshooting)
- [Complete command argument reference](#complete-command-argument-reference)
- [Tests and documentation](#tests-and-documentation)

## Command conventions

Run commands from the repository root:

```text
C:\Users\ehari\Desktop\scouting_report
```

Always use the project interpreter. At the start of a PowerShell session:

```powershell
$PY = ".\.venv\Scripts\python.exe"
$DATE = "2026-08-09"
$SLATE = "main"
$HISTORY_START = "2026-07-27"
```

Commands below use `& $PY`. Alternatively, activate the environment once and replace
`& $PY` with `python`:

```powershell
.\.venv\Scripts\Activate.ps1
```

If `python` reports a missing package such as `sklearn` or `pyarrow`, the wrong interpreter
is probably active. Check with `Get-Command python -All` or use `& $PY` directly.

---

## Daily run: start to finish

### 1. Download and identify DK salary files

Put every `DKSalaries.csv` export in `dfs_daily_files\`. A helpful naming convention is:

```text
DKSalaries_2026-08-09_early.csv
DKSalaries_2026-08-09_main.csv
DKSalaries_2026-08-09_turbo.csv
```

The tools identify exports by contents, not filename. List the slates before doing anything
else:

```powershell
& $PY -m dfs.cli --date $DATE --list-slates
```

If DK left an export in Downloads with a generic name, preview or adopt it:

```powershell
& $PY -m dfs.adopt --date $DATE --dry-run
& $PY -m dfs.adopt --date $DATE
```

### 2. Generate the morning scouting reports

For a new date, populate the report cache and Excel workbooks:

```powershell
& $PY scouting_report.py --date $DATE --all-games --format xlsx
```

Resume an interrupted run without rebuilding completed games:

```powershell
& $PY scouting_report.py --date $DATE --all-games --format xlsx --skip-existing
```

### 3. Preview and launch the snapshot scheduler

Preview the slate-relative times:

```powershell
& $PY -m dfs.autosnap --date $DATE --slate $SLATE --plan
```

Launch it and leave the terminal open:

```powershell
& $PY -m dfs.autosnap --date $DATE --slate $SLATE --adopt `
    --log "logs\autosnap_${DATE}_${SLATE}.log"
```

The scheduler refreshes lineups/reports and snapshots four stages: `morning`, `t-2h`,
`confirmed`, and `final`. See [Snapshot scheduler](#snapshot-scheduler) for multiple slates,
late starts, verification, and Windows Task Scheduler.

### 4. Build the current DFS board

After a scheduler refresh, and especially after the `final` stage, archive the current
board for each slate:

```powershell
& $PY -m dfs.cli --date $DATE --slate $SLATE
```

Stop and fix either of these warnings before trusting the board:

- `INCOMPLETE SLATE`: at least one DK-priced game has no cached scouting report.
- `WRONG SALARY FILE`: the selected export belongs to a different date.

Inspect everything written for the night:

```powershell
& $PY -m dfs.files --date $DATE
```

### 5. Refresh the pre-lock price/tier strategy card

Once the current `slate_<slate>.csv` exists:

```powershell
& $PY -m dfs.price_analysis --from $HISTORY_START --to $DATE
```

This command also runs the tier-level position strategy and produces the current daily
strategy card. Do not run `dfs.position_strategy` separately unless you intentionally want
to refresh only that layer.

The pre-lock run uses today's salaries and projections but only earlier contest results for
historical evidence. It can take several minutes because it builds multiple optimized
lineups for each position tier.

### 6. Generate lineups

Quick 20-lineup GPP build:

```powershell
& $PY -m dfs.optimize --date $DATE --slate $SLATE --n 20 `
    --objective ceiling --stack-shape 5 --stack-teams 0
```

Recommended wider-pool workflow:

```powershell
# First use only: create the editable player pool.
& $PY -m dfs.optimize --date $DATE --slate $SLATE --write-pool

# After editing Lock / Exclude / Boost / Min% / Max% in the pool CSV:
& $PY -m dfs.optimize --date $DATE --slate $SLATE --n 150 --pool `
    --objective ceiling --stack-shape "5-3,5-2,5" --stack-teams 0 --overwrite

# Select the entered set jointly and write it back in optimizer format.
& $PY -m dfs.portfolio --date $DATE --slate $SLATE --n 20 `
    --from-optimizer --write-lineups --contest twenty-max --sims 20000
```

### 7. Create the DK upload

```powershell
& $PY -m dfs.upload --date $DATE --slate $SLATE --list
& $PY -m dfs.upload --date $DATE --slate $SLATE
```

Or select specific optimizer lineup numbers:

```powershell
& $PY -m dfs.upload --date $DATE --slate $SLATE --lineups 1,3,5-8
```

The output is `dfs_boards\<date>\upload_<slate>.csv`. Upload that generated file to DK;
the source template is never modified.

### 8. Write the StreamFinder priority file

Order the live feed by what the night actually rides on, read off the upload:

```powershell
& $PY -m dfs.streamfinder --date $DATE --slate $SLATE
```

This writes `StreamFinder.txt` in the repository root. Run it after step 7, since exposure
is computed from `upload_<slate>.csv` -- the entries actually submitted -- not from the full
optimizer lineup set. Re-run it after a late swap.

### 9. Monitor scratches and postponements

Refresh the cached lineups before checking a scratch:

```powershell
& $PY scouting_report.py --date $DATE --all-games --from-cache `
    --refresh-lineups --format xlsx --dfs-slate $SLATE
```

Preview and execute late swap:

```powershell
& $PY -m dfs.lateswap --date $DATE --slate $SLATE --dry-run
& $PY -m dfs.lateswap --date $DATE --slate $SLATE
```

`dfs.lateswap` preserves started players in their exact DK cells and replaces only broken,
still-editable slots. A lineup that cannot be repaired legally is left unchanged.

### 10. Morning after: results and audits

Download the largest useful `contest-standings-*.csv` for each slate and place it in:

```text
dk_results\
```

Then run:

```powershell
# Score the lineups actually entered.
& $PY -m dfs.review --date $DATE --slate $SLATE --entered

# Rebuild price history, actual tier results, scarcity, and the strategy card.
& $PY -m dfs.price_analysis --from $HISTORY_START --to $DATE

# Compare the real field, top 20%, top 1%, and winners by learned tier.
& $PY -m dfs.tier_audit --from $HISTORY_START --to $DATE --min-match 90
```

The largest-field matching contest is selected automatically. Missing results stay missing;
they are never treated as zero.

---

## Daily price and tier audit

This is the short reference for the audit workflow described above.

### Before lock: decision analysis

Prerequisite: create a current board CSV for every relevant slate:

```powershell
& $PY -m dfs.cli --date $DATE --slate early
& $PY -m dfs.cli --date $DATE --slate main
```

Run:

```powershell
& $PY -m dfs.price_analysis --from $HISTORY_START --to $DATE
```

This writes `dfs_analysis\price_<from>_<to>\` and automatically includes:

- player salary history and start-to-start changes;
- empirical salary bands, not fixed $1,000 buckets;
- elite/solid/useful performance origins;
- position prices, depth, and weak-position days;
- pitcher price changes;
- tier-level optimized lineup sets;
- slate depth versus tier strategy; and
- `daily_strategy_card.csv`.

`dfs.price_analysis` already invokes `dfs.position_strategy`. A separate position command is
only needed for an isolated refresh or a different card objective:

```powershell
& $PY -m dfs.position_strategy --from $HISTORY_START --to $DATE
& $PY -m dfs.position_strategy --from $HISTORY_START --to $DATE --objective floor
```

### After results: actual field audit

Prerequisite: place the DK standings export in `dk_results\`, then rerun the price analysis
so actual points are attached:

```powershell
& $PY -m dfs.price_analysis --from $HISTORY_START --to $DATE
& $PY -m dfs.tier_audit --from $HISTORY_START --to $DATE --min-match 90
```

The tier audit produces:

| File | Use |
| --- | --- |
| `tier_audit.md` | Readable audit report |
| `tier_audit_coverage.csv` | Included, warmup, and excluded contests |
| `tier_historical_summary.csv` | Historical field-to-top-1% leverage by position tier |
| `tier_cohort_summary.csv` | Slate-level field/top-20%/top-1%/winner exposure |
| `player_cohort_summary.csv` | Players inside each tier, such as all Tier 5 first basemen |
| `winning_lineup_tiers.csv` | Every roster slot from winning entries |
| `premium_count_summary.csv` | Results by number of premium hitters and pitchers |
| `premium_pattern_summary.csv` | Results by premium positions, such as `P+3B+SS` |
| `entry_tier_constructions.csv` | Tier signature for every fully matched real lineup |

Important rules:

- Tiers are learned strictly from earlier scored dates.
- Top 20% is a cash proxy; DK standings do not contain the payout table.
- `--min-match 90` excludes contests whose archived board cannot identify at least 90% of
  real roster slots.
- Exact construction tables require known tiers for all ten lineup slots.
- A date with no contest export does not enter the field audit.
- The first historical date can be a warmup because no earlier data exists to define tiers.

Verify the latest result in `tier_audit_coverage.csv`. It should show:

```text
Audit Status = Included
Match% >= 90
Tier-complete Lineups > 0
```

---

## Snapshot scheduler

### Manual launch

Preview:

```powershell
& $PY -m dfs.autosnap --date $DATE --slate $SLATE --plan
```

Launch and leave the terminal open:

```powershell
& $PY -m dfs.autosnap --date $DATE --slate $SLATE --adopt `
    --log "logs\autosnap_${DATE}_${SLATE}.log"
```

If the day has multiple slates, run each in a separate terminal:

```powershell
& $PY -m dfs.autosnap --date $DATE --slate early --adopt `
    --log "logs\autosnap_${DATE}_early.log"

& $PY -m dfs.autosnap --date $DATE --slate main --adopt `
    --log "logs\autosnap_${DATE}_main.log"
```

Launching late and only needing decision snapshots:

```powershell
& $PY -m dfs.autosnap --date $DATE --slate $SLATE `
    --stages confirmed,final --adopt
```

By default, a stage whose time already passed runs immediately. Add `--skip-past` only when
you intentionally want to omit missed stages.

Each normal stage runs the equivalent of:

```powershell
& $PY scouting_report.py --date $DATE --all-games --from-cache `
    --refresh-lineups --format xlsx --dfs-slate $SLATE
```

and then writes an immutable snapshot. Do not use `--no-refresh` for a normal production
run; it can create several differently named snapshots of the same stale morning state.

### Verify snapshots

```powershell
& $PY -m dfs.snapshot --date $DATE --slate $SLATE --list
& $PY -m dfs.snapshot --date $DATE --slate $SLATE --verify
```

Manual emergency snapshot:

```powershell
& $PY -m dfs.snapshot --date $DATE --slate $SLATE --stage final `
    --note "manual final before lock"
```

Snapshots are the only honest record of what the projections knew at lock. Later lineup
refreshes rewrite `.cache\report_data\` in place; `dfs.review` prefers the immutable snapshot.

### Windows Task Scheduler

One daily task can launch the main-slate scheduler and let it derive stage times from DK's
first pitch:

```powershell
$repo = "C:\Users\ehari\Desktop\scouting_report"
$python = Join-Path $repo ".venv\Scripts\python.exe"
$log = Join-Path $repo "logs\autosnap_main.log"

$action = New-ScheduledTaskAction `
    -Execute $python `
    -Argument ('-m dfs.autosnap --slate main --adopt --log "' + $log + '"') `
    -WorkingDirectory $repo

$trigger = New-ScheduledTaskTrigger -Daily -At "09:00"
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 14)

$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive

Register-ScheduledTask -TaskName "DFS autosnap main" `
    -Action $action -Trigger $trigger -Settings $settings `
    -Principal $principal -Force
```

For another slate, register a second task with a different task name, `--slate`, and log.

Check or stop the task:

```powershell
Get-ScheduledTaskInfo -TaskName "DFS autosnap main"
Get-Content logs\autosnap_main.log -Tail 40
Stop-ScheduledTask -TaskName "DFS autosnap main"
```

`-WorkingDirectory` is required because the project uses relative paths. Invoke the venv's
`python.exe` directly rather than wrapping it in `cmd /c`.

---

## Salary files and multiple slates

List the exports discovered for a date:

```powershell
& $PY -m dfs.cli --date $DATE --list-slates
```

Select a slate everywhere when more than one exists:

```powershell
& $PY -m dfs.cli --date $DATE --slate main
& $PY -m dfs.optimize --date $DATE --slate main --n 20
& $PY -m dfs.upload --date $DATE --slate main
```

The slate label becomes part of every filename. Readers use the newest revision. By default,
reruns create `.r2`, `.r3`, and so on; `--overwrite` replaces the current base file.

#### Two slates of the same kind on one night

2026-08-18 ran two turbos. Rename the exports so each carries its own label -- the second
takes a `-2` suffix, matching what the automatic labeller would have chosen:

```text
DKSalaries_2026-08-18_main.csv
DKSalaries_2026-08-18_turbo.csv
DKSalaries_2026-08-18_turbo-2.csv
```

Then select each by its label:

```powershell
& $PY -m dfs.cli --date $DATE --slate turbo
& $PY -m dfs.cli --date $DATE --slate turbo-2
```

`--slate` takes a **label**, and it does double duty: it picks the export and it names
every output file for that slate. An exact label wins over a partial name, so `turbo` finds
the turbo rather than colliding with `turbo-2`. A filename or a full path is also accepted
and selects the same file; the outputs are still filed under the export's own label, not
under the filename you typed.

Passing something that is not any export's label renames the slate. That is the intended
escape hatch (`--slate "turbo night"` files everything under `turbo-night`), but a label
invented at the CLI has no export to resolve back to, so later commands must be given the
same string.

Useful checks:

```powershell
& $PY -m dfs.files --date $DATE
& $PY -m dfs.adopt --date $DATE --dry-run
```

---

## Scouting reports

### Generate or resume

```powershell
& $PY scouting_report.py --date $DATE --all-games --format xlsx
& $PY scouting_report.py --date $DATE --all-games --format xlsx --skip-existing
& $PY scouting_report.py --date $DATE --all-games --format xlsx --upcoming-only
```

One game:

```powershell
& $PY scouting_report.py --date $DATE --away-team CHC --home-team STL --format xlsx
```

### Refresh from cache

Fast re-render without new lineup cards:

```powershell
& $PY scouting_report.py --date $DATE --all-games --from-cache --fast --format xlsx
```

Refresh lineup cards and rebuild affected player sections:

```powershell
& $PY scouting_report.py --date $DATE --all-games --from-cache `
    --refresh-lineups --format xlsx --dfs-slate $SLATE
```

Use `--force-lineup-refresh` only when the posted card should be rebuilt even though it
matches the cache. Use `--dfs-slate` when the same game appears on multiple DK exports and
the Excel DFS highlighting would otherwise be ambiguous.

### Missing probable pitcher

The report no longer skips a game because MLB has not posted a probable. It falls back in
order:

| Rung | Source | Marked provisional |
| --- | --- | --- |
| `--away-pitcher` / `--home-pitcher` | you | no |
| announced | StatsAPI probable, then the probable-pitchers page | no |
| DK salary file | the `Starting` column of every `DKSalaries_<date>_*.csv` for the date | **yes** |
| rotation model | `build_rotation` + `effective_starter` | **yes** |

A provisional starter is labelled on the report — the starter comparison title and the
pitching-tab panel header both carry `⚠ PROJECTED STARTER` and name the source, with the
rotation model's own share and days of rest alongside it. Only when every rung comes up
empty is the game skipped.

Once MLB posts the probable, re-run with `--refresh-lineups`: if the real starter differs
from the projected one the game is regenerated from source (the arsenal, comps, BvP, splits
and hitter composite were all built against the old arm and cannot be patched); if it is the
same arm the label is simply upgraded in place.

Force an MLBAM id or pitcher name when you know better than any of them:

```powershell
& $PY scouting_report.py --date $DATE --away-team CHC --home-team STL `
    --home-pitcher "Matthew Liberatore" --format xlsx
```

The same pattern works with `--away-pitcher`. Use `--strict-probables` to restore the old
skip-unless-announced behaviour — worth it for a date far enough out that a rotation guess is
not worth the generation time.

### Doubleheaders

The report refuses to guess which game is intended. Select it explicitly:

```powershell
& $PY scouting_report.py --date $DATE --away-team CLE --home-team CIN `
    --dh-game 2 --format xlsx
```

Game 2 is cached separately and does not overwrite Game 1.

### Umpire tags and pitcher home-run context

The Excel umpire section automatically matches the assigned home-plate umpire against:

```text
data\raw\Umpire Tags - Sheet1.csv
```

Keep the CSV headers exactly `Umpire,ERA,Rating`. Names are matched case-, accent-, and
punctuation-insensitively, but the full name still needs to identify the same umpire. After
changing the tag file, rerun the report command to regenerate the workbook.

Pitcher summaries include both total home runs allowed (`HR`) and the innings-normalized
rate (`HR/9`). Treat the total as workload-dependent context and use `HR/9` when comparing
pitchers with different innings totals.

### Reading the report heat scale

Stat cells use fixed MLB reference ranges rather than the smallest and largest value in a
single table. The color layer is also bounded:

- very poor results saturate at red;
- very strong results saturate at green; and
- the meaningful middle range receives the full red-to-amber-to-green spectrum.

This prevents one extreme outlier from washing out the differences among the other rows.
The direction is metric-aware: lower ERA, FIP, WHIP, and allowed statistics are good, while
higher offensive production and strikeout ability are good. The Excel report displays the
three-point scale in its second row.

### Maintenance

```powershell
& $PY scouting_report.py --date $DATE --confirmed-lineups
& $PY scouting_report.py --date $DATE --qc-reports
& $PY scouting_report.py --calibrate --calibration-days 30
& $PY scouting_report.py --backtest --backtest-days 30
```

Report-model calibration now builds a reusable regular-season game dataset at
`.cache/model_calibration_games.csv`. MLB schedules are pulled in cached monthly batches;
each row stores the pregame team snapshot plus park, first-pitch temperature/wind, roof,
and the home-plate umpire's prior run tendency. Re-running a date range reuses those rows.
Team records and run rates carry a 20-game neutral prior, with a lighter five-game prior
on L10, so Opening Week results cannot create unsupported extreme win probabilities.
Suspended/resumed games are deduplicated by MLB game id.

For a multi-season fit (latest season is held out for validation):

```powershell
& $PY -m scouting_report --calibrate `
    --calibration-start 2023-03-30 `
    --calibration-end 2026-08-09
```

`--calibration-max-games 0` means the full range and is the default. A positive cap keeps
the newest games, not the oldest games. The saved JSON includes equal-count holdout signal
buckets for win probability, total adjustments, combined context, weather, and umpire
tendency. Each block reports directional accuracy, error changes, confidence intervals,
and magnitude-trend significance.

Every report cache stores the calibration fingerprint that produced its scorecard. When
`dfs.slate` builds a board, it compares that fingerprint with the current
`.cache/model_calibration.json`; a mismatch triggers a local-only scorecard rebuild from
the cached lineups, starters, bullpens, similarity tables, environment, and time-zone
context. The pickle is updated atomically before player projection. Slate metadata records
`calibration_fingerprint`, `calibration_feature_version`, and
`calibration_refreshed_games`, so optimizer provenance is inspectable. A missing artifact,
failed refresh, or mixed fingerprint fails the whole slate closed instead of returning a
partial optimizer pool.

---

## DFS board

Build and optionally print the board:

```powershell
& $PY -m dfs.cli --date $DATE --slate $SLATE
& $PY -m dfs.cli --date $DATE --slate $SLATE --print
& $PY -m dfs.cli --date $DATE --slate $SLATE --top-per-position 12
```

Primary outputs:

- `board_<slate>.md`: readable pitchers, stacks, and hitters by position.
- `slate_<slate>.csv`: full priced player pool and projections.
- `stacks_<slate>.csv`: team stack rankings.

Key columns:

| Column | Meaning |
| --- | --- |
| `Proj` | Mean DK-point expectation |
| `Ceiling` / `Floor` | Approximately 90th- and 25th-percentile outcomes |
| `Bust%` | Probability of a low score |
| `Value` | Projected points per $1,000 |
| `GPP` / `CASH` | Tournament and floor-oriented rankings |
| `Own%` | Projected field ownership |
| `Leverage` | Projection rank relative to ownership rank |
| `Tier` | Board value label: Core, Value, Leverage, Risk, or Fade |
| `Role` | Ceiling, Floor, or Ceiling+Floor |

Board `Tier` is a nightly value label. The historical audit's `Tier 1` through `Tier 5`
are separately learned salary bands; do not confuse the two.

---

## Optimizer and portfolio

### Common optimizer commands

```powershell
# GPP
& $PY -m dfs.optimize --date $DATE --slate $SLATE --n 20 `
    --objective ceiling --stack-shape 5 --stack-teams 0

# Balanced projection
& $PY -m dfs.optimize --date $DATE --slate $SLATE --n 20 --objective proj

# Single-entry/cash
& $PY -m dfs.optimize --date $DATE --slate $SLATE --n 1 `
    --objective floor --randomness 0

# Explicit stack and player decisions
& $PY -m dfs.optimize --date $DATE --slate $SLATE --n 20 `
    --stack "LAD:4,SD:3" --lock "Player Name" --exclude "Other Player"

# Leverage build
& $PY -m dfs.optimize --date $DATE --slate $SLATE --n 20 `
    --objective leverage --max-ownership 120
```

Core options:

| Option | Purpose |
| --- | --- |
| `--objective ceiling|proj|floor|leverage` | Select the lineup objective |
| `--n N` | Number of lineups |
| `--lock` / `--exclude` | Force or remove named players |
| `--stack-shape "5-3,5-2,5"` | Cycle through required stack shapes |
| `--stack-teams 0` | Allow every team as the primary stack |
| `--focus-teams "NYY,LAD,SD"` | Restrict deliberate stacks to selected teams |
| `--max-overlap N` | Maximum shared players across generated lineups |
| `--randomness X` | Objective jitter; use `0` for the deterministic best lineup |
| `--seed N` | Make a randomized set reproducible |
| `--max-ownership PCT` | Cap total projected ownership |
| `--allow-started` | Analysis only; never use for a live build |

The optimizer enforces the DK Classic roster, $50,000 cap, five-hitter team maximum, and
two-game minimum. The default pitcher-versus-own-stack conflict penalty discourages a
starter facing three or more hitters in the same lineup.

### Editable player pool

```powershell
& $PY -m dfs.optimize --date $DATE --slate $SLATE --write-pool
# Edit pool_<slate>.csv in Excel.
& $PY -m dfs.optimize --date $DATE --slate $SLATE --pool --n 20
```

| Column | Effect |
| --- | --- |
| `Lock` / `Exclude` | Hard player decisions |
| `Boost` | Multiplies the objective; does not cap exposure |
| `Min%` / `Max%` | Exact set-level exposure bounds |

Re-running `--write-pool` merges new slate players while preserving prior edits.
`--reset-pool` intentionally discards all edits.

### Joint portfolio selection

Recommended for multi-entry contests:

```powershell
& $PY -m dfs.optimize --date $DATE --slate $SLATE --n 150 --pool `
    --stack-shape "5-3,5-2,5" --stack-teams 0 --overwrite

& $PY -m dfs.portfolio --date $DATE --slate $SLATE --n 20 `
    --from-optimizer --write-lineups --contest twenty-max --sims 20000
```

`--from-optimizer` retains pool edits and optimizer constraints. `--write-lineups` writes
the selected set back into the format used by `dfs.upload`.

Mass multi-entry example:

```powershell
& $PY -m dfs.portfolio --date $DATE --slate $SLATE --n 150 `
    --from-optimizer --write-lineups --contest milly-maker `
    --weights mass-multi --max-player-exposure 0.35 --max-stack-exposure 0.25
```

---

## Upload and late swap

### Upload

Place a DK entries export or bulk upload template in `dk_lineups\`. If no template exists,
the uploader can build a new-entry bulk template from the salary export. Editing already
submitted entries still requires DK's real entries export.

```powershell
& $PY -m dfs.upload --date $DATE --slate $SLATE --list
& $PY -m dfs.upload --date $DATE --slate $SLATE
& $PY -m dfs.upload --date $DATE --slate $SLATE --lineups 1,3,5-8
```

Useful options: `--template`, `--contest`, `--source`, `--out`, and `--overwrite`.

### Late swap

```powershell
& $PY -m dfs.lateswap --date $DATE --slate $SLATE --dry-run
& $PY -m dfs.lateswap --date $DATE --slate $SLATE
```

Or use the optimizer wrapper without generating new lineups:

```powershell
& $PY -m dfs.optimize --date $DATE --slate $SLATE --swap
```

Late swap defaults to the newest `upload_` or `swap_` file. Use `--file` to name one. It
writes a new `swap_<slate>.csv` unless `--in-place` is explicitly requested.

---

## StreamFinder priority file

StreamFinder cuts a live feed to whichever prioritised player is up next, so the list it
wants is small and ordered. After lineups are submitted that order is not an opinion -- it
is exposure. A player in 8 of 9 entries deserves the feed ahead of one in 2, whatever either
was projected for.

```powershell
& $PY -m dfs.streamfinder --date $DATE --slate $SLATE
```

Output goes to `StreamFinder.txt` in the repository root; point StreamFinder at that file.
The printed table shows each player's priority, role, lineup count, and share:

```text
upload_main.csv -> StreamFinder.txt
  30 players prioritised
    #  player                        used  share
    1  Blake Snell              pit     9  100.0%
    2  Framber Valdez           pit     8   88.9%
    3  Randal Grichuk           bat     8   88.9%
  read from 9 lineups
```

Common variations:

```powershell
# More or fewer prioritised players (default 30).
& $PY -m dfs.streamfinder --date $DATE --slate $SLATE --top 20

# Ignore teams globally by MLB team id; repeat the flag per team.
& $PY -m dfs.streamfinder --date $DATE --ignore 116 --ignore 113

# Score a specific upload file, bypassing the date lookup.
& $PY -m dfs.streamfinder --upload dfs_boards\2026-08-17\upload_main.csv

# Override the app's own defaults and the output path.
& $PY -m dfs.streamfinder --date $DATE --on-deck Y --include-cli N `
    --delay 5000 --out StreamFinder.txt
```

Notes:

- Exposure is read from `upload_<slate>.csv`, never `lineups_<slate>.csv`. After a
  `--select` or a hand edit those are different sets, and the lineup file describes a night
  that was not entered.
- `--slate` takes a slate label, not a path -- `--slate turbo`, not
  `--slate dfs_boards\2026-08-18\upload_turbo.csv`. The file to read is `--upload`.
- `--slate` may be omitted when the date has exactly one upload. With more than one the
  command stops and lists the slates rather than guessing.
- The label also selects the salary export that supplies MLBAM ids. If a run resolves the
  upload but prioritises 0 players and lists everyone as "not matched", the label reached
  a different slate's board than the upload came from.
- Ties on lineup count break by projection, so equally-used players still get a stable
  order instead of whatever the optimizer wrote first.
- Player ids are joined from the slate's MLBAM column, not looked up by name. Anyone the
  join misses is listed under "not matched" rather than silently dropped.
- Re-run after `dfs.lateswap`, pointing `--upload` at the new `swap_<slate>.csv` if the
  swapped file is the one submitted.

---

## Post-contest review

After placing standings in `dk_results\`:

```powershell
& $PY -m dfs.results --date $DATE --slate $SLATE
& $PY -m dfs.review --date $DATE --slate $SLATE
& $PY -m dfs.review --date $DATE --slate $SLATE --entered
```

`dfs.results` confirms which contest export matched the slate and displays real ownership.
`dfs.review` scores projections, recommendations, and generated or entered lineups.

Useful diagnostic modes:

```powershell
# Test all shipped optimizer configurations; slow.
& $PY -m dfs.review --date $DATE --slate $SLATE --sweep

# Test one hypothetical configuration.
& $PY -m dfs.review --date $DATE --slate $SLATE `
    --criteria "objective=ceiling,max_overlap=4,randomness=0.3"

# Determine whether a saved candidate pool contained the needed construction.
& $PY -m dfs.review --date $DATE --slate $SLATE --entered `
    --candidates "dfs_boards\$DATE\candidates_$SLATE"
```

`dfs.review` uses the night's snapshot when available. Without one, it warns that the
mutable cache may contain information that arrived after lock.

After a batch of new contest exports, recalibrate and evaluate the field/projection layers:

```powershell
& $PY -m dfs.field --calibrate
& $PY -m dfs.evaluate --label nightly
```

---

## Research and model evaluation

These commands are useful for research and validation; they are not required for the
normal upload workflow.

### Walk-forward evaluation

```powershell
& $PY -m dfs.evaluate
& $PY -m dfs.evaluate --from $HISTORY_START --to $DATE --label nightly
& $PY -m dfs.evaluate --no-recalibrate
```

Use it after changes to projections or probability bands. Walk-forward evaluation trains
only on dates earlier than the date being scored.

### Candidate exploration

```powershell
& $PY -m dfs.candidates --date $DATE --slate $SLATE --n 1000 --seed 7
& $PY -m dfs.candidates --date $DATE --slate $SLATE --snapshot --n 500
```

Candidate generation explores a wider stack/player space but does not carry editable-pool
exposure minimums and cannot directly create an upload unless bridged from the optimizer.

### Correlated simulation

```powershell
& $PY -m dfs.simulate --date $DATE --slate $SLATE --sims 20000 --seed 7
& $PY -m dfs.simulate --date $DATE --slate $SLATE --sims 20000 --actuals
```

Score a saved candidate pool:

```powershell
& $PY -m dfs.simulate --date $DATE --slate $SLATE --sims 20000 `
    --candidates "dfs_boards\$DATE\candidates_$SLATE"
```

### Opponent field and contest EV

```powershell
& $PY -m dfs.field --date $DATE --slate $SLATE --contest twenty-max
& $PY -m dfs.contest --date $DATE --slate $SLATE --contest twenty-max
& $PY -m dfs.contest --date $DATE --slate $SLATE --from-optimizer
```

For a decision involving meaningful money, supply the real payout structure:

```json
{
  "name": "MLB GPP",
  "entry_fee": 20.0,
  "field_size": 11764,
  "tiers": [[1, 10000], [2, 5000], [5, 1500], [10, 600], [2700, 30]]
}
```

```powershell
& $PY -m dfs.contest --date $DATE --slate $SLATE `
    --payout-json contests\mlb_gpp.json --field-size 11764
```

### Quick box-score backtest

```powershell
& $PY -c "from dfs.backtest import run_backtest, summarize; print(summarize(run_backtest(['2026-08-07','2026-08-08'])))"
```

For serious model evaluation, prefer `dfs.evaluate` because it is walk-forward.

---

## Output locations

| Location | Contents |
| --- | --- |
| `scouting_reports\<date>\` | Scouting-report Excel/PDF/Markdown outputs |
| `.cache\report_data\` | Mutable latest per-game report payloads |
| `.cache\snapshots\<date>\<slate>\` | Immutable slate snapshots |
| `dfs_daily_files\` | DK salary exports and optimizer configuration |
| `dfs_boards\<date>\` | Boards, slate CSVs, pools, lineups, uploads, swaps |
| `dk_lineups\` | DK upload templates and entries exports |
| `StreamFinder.txt` | StreamFinder priority file, repository root |
| `dk_results\` | DK contest standings exports |
| `dfs_analysis\price_<from>_<to>\` | Price, tier, scarcity, and real-field audits |
| `docs\benchmarks\` | Evaluation and profiling artifacts |
| `logs\` | Autosnap logs when `--log` is used |

List one night's DFS artifacts:

```powershell
& $PY -m dfs.files --date $DATE
```

Nothing is overwritten by default. Most writers create revision suffixes such as `.r2`;
readers select the newest version. Use `--overwrite` only when replacement is intentional.

---

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Missing Python package | Use `& $PY`; the system Python is not the project environment |
| No salary export found | Put it in `dfs_daily_files\`, run `dfs.adopt`, then `--list-slates` |
| Several matching slates | Pass `--slate early|main|turbo|late` explicitly |
| `INCOMPLETE SLATE` | Generate the missing game report; force a probable pitcher if needed |
| `WRONG SALARY FILE` | Download/select the export for the requested date |
| Excel write error | Close the open workbook and rerun |
| Confirmed lineup missing | Run the cached report refresh with `--refresh-lineups` |
| DFS highlighting ambiguous | Pass `--dfs-slate <label>` to `scouting_report.py` |
| No snapshot | Take one before lock; a later cache rebuild cannot reproduce it honestly |
| Snapshot payload changed | Review from the snapshot, not the rewritten cache |
| Lineups are too similar | Lower `--max-overlap`, widen stack shapes, or use portfolio selection |
| Optimizer infeasible | Relax locks, excludes, ownership cap, exposures, or stack requirements |
| Upload template mismatch | Redownload salary and template files from the same DK draft group |
| Scratch still in late swap | Refresh cached lineups before running late swap |
| Started players appear | Do not use `--allow-started` for a live build |
| Audit has no new date | Add the slate's contest standings export to `dk_results\` |
| Audit status excluded | Inspect `Match%`; archived board coverage is below `--min-match` |
| Tier-complete lineups are zero | Some slots cannot be matched or the date has no earlier tier history |
| No actual points in price audit | Confirm `coverage.csv` has a matching contest and nonzero `Scored` |
| Recursive pytest/temp permission warning | Run focused tests with `-p no:cacheprovider` |

Postponed players are removed automatically when the live schedule confirms the game. If
the schedule lookup fails, verify the DK player pool manually before uploading.

---

## Complete command argument reference

This appendix covers every executable report and DFS command currently shipped in the
repository. It was checked against each parser's current `--help` output. Every command
supports `-h` / `--help`; that universal option is omitted from the individual tables.

Options shown as `[PATH]` accept an optional value: using the bare flag activates automatic
file discovery, while supplying a path selects a specific file. Dates use `YYYY-MM-DD`.
The `dfs.backtest` example elsewhere in this document is a Python API call rather than an
executable CLI, so it has no `--arguments` of its own.

| Area | Commands |
| --- | --- |
| Reports | `scouting_report.py` |
| Files and snapshots | `dfs.cli`, `dfs.adopt`, `dfs.autosnap`, `dfs.snapshot`, `dfs.files` |
| Price strategy | `dfs.price_analysis`, `dfs.position_strategy`, `dfs.tier_audit` |
| Lineups | `dfs.optimize`, `dfs.portfolio`, `dfs.upload`, `dfs.lateswap`, `dfs.review` |
| Live feed | `dfs.streamfinder` |
| Research | `dfs.results`, `dfs.evaluate`, `dfs.candidates`, `dfs.simulate`, `dfs.field`, `dfs.contest` |

### `scouting_report.py`

| Argument | Meaning |
| --- | --- |
| `--date DATE` | Game date; defaults to today |
| `--game-number N` | Select the 1-based game from that date's schedule |
| `--away-team TEAM` | Away-team abbreviation for a single game |
| `--home-team TEAM` | Home-team abbreviation for a single game |
| `--from-cache` | Re-render an existing cached report instead of collecting it again |
| `--fast` | With `--from-cache`, skip every API refresh |
| `--refresh-lineups` | With `--from-cache`, pull current lineup cards and rebuild changed reports |
| `--force-lineup-refresh` | Rebuild even when the posted lineup matches the cache |
| `--no-dfs-highlight` | Disable DFS-value tinting of player-name cells |
| `--dfs-slate SLATE` | Select the DK export used for Excel DFS highlighting |
| `--dh-game {1,2}` | Select a doubleheader game |
| `--away-pitcher ID_OR_NAME` | Override an unposted away probable pitcher |
| `--home-pitcher ID_OR_NAME` | Override an unposted home probable pitcher |
| `--strict-probables` | Skip a game unless MLB has posted a probable, instead of projecting one |
| `--format FORMAT` | `markdown`, `html`, `pdf`, `comparison-pdf`, `xlsx`, or `both`; default `xlsx` |
| `--include-plots` | Generate the slower pitch-location heatmap images |
| `--output-dir DIR` | Base report directory; default `scouting_reports` |
| `--flat-output` | Write directly to `--output-dir` instead of its dated subfolder |
| `--similarity-lookback N` | Seasons used for similar-pitcher and pitcher-type samples |
| `--all-games` | Generate every scheduled game on the date |
| `--skip-existing` | With `--all-games`, skip reports whose expected output already exists |
| `--upcoming-only` | With `--all-games`, omit in-progress and final games |
| `--confirmed-lineups` | List games with both actual nine-man lineups posted; generate no reports |
| `--lineup-check` | Alias for `--confirmed-lineups` |
| `--qc-reports` | Scan the date's Markdown reports and write a QC CSV |
| `--calibrate` | Fit and save report-model calibration coefficients |
| `--calibration-start DATE` | First calibration date |
| `--calibration-end DATE` | Last calibration date; defaults to yesterday |
| `--calibration-days N` | Lookback when `--calibration-start` is omitted |
| `--calibration-max-games N` | Maximum completed games, keeping newest; `0` uses the full range |
| `--calibration-data PATH` | Reusable per-game calibration feature CSV |
| `--backtest` | Write the per-game prediction backtest using current calibration |
| `--backtest-start DATE` | First report-model backtest date |
| `--backtest-end DATE` | Last backtest date; defaults to yesterday |
| `--backtest-days N` | Lookback when `--backtest-start` is omitted |
| `--backtest-max-games N` | Maximum completed games in the backtest |
| `--backtest-output PATH` | Backtest CSV path |

### `dfs.cli`

| Argument | Meaning |
| --- | --- |
| `--date DATE` | Slate date; defaults to today |
| `--salaries PATH` | Explicit DK salary export |
| `--slate SLATE` | Select an export by filename substring |
| `--list-slates` | List matching exports and stop |
| `--data-dir DIR` | Override the cached report-data directory |
| `--output-dir DIR` | Override the board output root |
| `--overwrite` | Replace base outputs instead of writing `.rN` revisions |
| `--downloads` | Also search Downloads for an unfiled DK export |
| `--no-adopt` | Do not rename misnamed DK downloads into the convention |
| `--top-pitchers N` | Pitchers shown in the Markdown board |
| `--top-per-position N` | Hitters shown per position |
| `--print` | Print the board to the terminal |
| `--profile` | Record pipeline timing under `docs\benchmarks` |

### `dfs.adopt`

| Argument | Meaning |
| --- | --- |
| `--date DATE` | Only adopt exports matching this slate date |
| `--dry-run` | Preview file operations without copying or renaming |
| `--downloads` | Search Downloads; matched files are copied, never moved |
| `--dir DIR` | Search an additional directory; repeatable |

### `dfs.autosnap`

| Argument | Meaning |
| --- | --- |
| `--date DATE` | Slate date; defaults to today |
| `--slate SLATE` | Select one DK export |
| `--salaries PATH` | Use an explicit salary export |
| `--stages LIST` | Comma-separated subset of `morning,t-2h,confirmed,final` |
| `--plan` | Print derived stage times and exit |
| `--dry-run` | Walk the schedule and print actions without executing them |
| `--no-refresh` | Snapshot current cache without pulling lineups first |
| `--refresh-cmd COMMAND` | Override refresh command; supports `{python}`, `{date}`, and `{slate}` placeholders |
| `--adopt` | Adopt stray/misnamed DK exports before processing |
| `--log PATH` | Append scheduler output to a log as well as the console |
| `--wait-for-salaries MINUTES` | Poll this long for a missing salary export; `0` fails immediately |
| `--skip-past` | Omit already-passed stages instead of running them immediately |

### `dfs.snapshot`

| Argument | Meaning |
| --- | --- |
| `--date DATE` | Snapshot date |
| `--slate SLATE` | Select one DK export |
| `--stage STAGE` | `morning`, `t-2h`, `confirmed`, or `final`; default `final` |
| `--salaries PATH` | Use an explicit salary export |
| `--note TEXT` | Store free text in the snapshot manifest |
| `--list` | List snapshots and stop |
| `--verify` | Check whether source payloads changed after capture |

### `dfs.files`

| Argument | Meaning |
| --- | --- |
| `--date DATE` | Night to inspect |
| `--root DIR` | Override the DFS board root directory |

### `dfs.price_analysis`

| Argument | Meaning |
| --- | --- |
| `--from DATE` | First slate date |
| `--to DATE` | Last slate date |
| `--slate SLATE` | Restrict the analysis to one slate label |
| `--min-sample N` | Minimum scored tier sample used in written findings; default `10` |
| `--output-root DIR` | Override the analysis output root |

### `dfs.position_strategy`

| Argument | Meaning |
| --- | --- |
| `--from DATE` | First archived date |
| `--to DATE` | Last/current slate date |
| `--slate SLATE` | Restrict to one slate label |
| `--objective OBJECTIVE` | Strategy-card objective: `ceiling`, `floor`, or `proj`; default `ceiling` |
| `--output-root DIR` | Override the analysis output root |

### `dfs.tier_audit`

| Argument | Meaning |
| --- | --- |
| `--from DATE` | First contest date to audit |
| `--to DATE` | Last contest date to audit |
| `--results-dir DIR` | Override the DK standings directory |
| `--output-root DIR` | Override the analysis output root |
| `--min-match PCT` | Minimum matched roster-slot percentage used for strategic rates; default `90` |

### `dfs.optimize`

| Argument | Meaning |
| --- | --- |
| `--date DATE` | Slate date |
| `--salaries PATH` | Use an explicit DK salary export |
| `--slate SLATE` | Select one DK export |
| `--config PATH` | Use a JSON settings file |
| `--n N` | Number of lineups to generate |
| `--objective OBJECTIVE` | `ceiling`, `proj`, `floor`, or `leverage` |
| `--max-ownership PCT` | Maximum summed projected ownership per lineup |
| `--lock PLAYERS` | Comma-separated players required in every lineup |
| `--exclude PLAYERS` | Comma-separated players removed from the pool |
| `--stack STACKS` | Explicit team stacks such as `CWS:4,HOU:3` |
| `--stack-shape SHAPES` | One or more shapes such as `5-3,5-2,5` |
| `--stack-teams N` | Number of top teams allowed as deliberate stacks; `0` allows all |
| `--focus-teams TEAMS` | Comma-separated teams eligible for deliberate stacks |
| `--max-overlap N` | Maximum shared players between generated lineups |
| `--min-proj POINTS` | Remove players projected below this value |
| `--max-bust PCT` | Remove players above this bust probability |
| `--max-from-team N` | Maximum hitters from one team; DK permits five |
| `--randomness FRACTION` | Objective jitter; `0` is deterministic |
| `--seed N` | Reproduce a randomized run |
| `--write-pool` | Write/merge the editable pool and stop |
| `--reset-pool` | With `--write-pool`, discard previous pool edits |
| `--pool [PATH]` | Apply an editable pool; bare flag discovers the standard file |
| `--conflict-penalty MULTIPLIER` | Projection tax on a pitcher facing a prominent own stack |
| `--conflict-min-hitters N` | Opposing hitters required before the conflict tax applies |
| `--no-stack-bonus` | Disable the correlation bonus used by ceiling/projection builds |
| `--save-config` | Save resolved settings to the config file |
| `--exposure-top N` | Rows printed in the exposure summary; `0` prints all |
| `--upload [SELECTION]` | Also build an upload; optional selection such as `1,3,5-8` |
| `--template PATH` | DK upload template used with `--upload` |
| `--allow-started` | Keep started players for analysis/backtests only |
| `--swap` | Repair broken slots in an existing filled upload |
| `--overwrite` | Replace base outputs instead of writing `.rN` revisions |
| `--profile` | Record pipeline timing |

### `dfs.portfolio`

| Argument | Meaning |
| --- | --- |
| `--date DATE` | Slate date |
| `--slate SLATE` | Select one DK export |
| `--snapshot` | Build from a frozen snapshot instead of the live cache |
| `--stage STAGE` | Select the snapshot stage |
| `--n N` | Number of lineups to enter |
| `--contest PRESET` | `single-entry`, `three-max`, `twenty-max`, or `milly-maker` |
| `--weights PRESET` | Objective weights: `cash`, `mass-multi`, `single-entry`, `three-max`, or `twenty-max` |
| `--entry-fee AMOUNT` | Override contest entry fee |
| `--field-size N` | Override real contest field size |
| `--field-entries N` | Number of opponent lineups actually simulated |
| `--payout-json PATH` | Use an exact payout structure |
| `--candidates N` | Candidate lineups generated before joint selection |
| `--sims N` | Correlated outcome simulations |
| `--max-ownership PCT` | Maximum summed projected ownership |
| `--from-optimizer [PATH]` | Select optimizer lineups; bare flag finds the newest set |
| `--write-lineups [PATH]` | Write the chosen set in optimizer format; requires `--from-optimizer` |
| `--allow-started` | Keep started players for analysis only |
| `--max-player-exposure FRACTION` | Hard player exposure cap across the selected set |
| `--max-stack-exposure FRACTION` | Hard primary-stack exposure cap |
| `--seed N` | Reproduce the simulation and selection |
| `--swap-passes N` | Local improvement passes after greedy selection; `0` disables them |
| `--out PATH` | Write selected lineups to this CSV |
| `--profile` | Record pipeline timing |

### `dfs.upload`

| Argument | Meaning |
| --- | --- |
| `--date DATE` | Slate date |
| `--lineups SELECTION` | Lineup numbers such as `1,3,5-8`; default all |
| `--template PATH` | Explicit DK upload/entries template |
| `--contest TEXT` | Fill only entries whose contest name contains this text |
| `--source PATH` | Explicit optimizer lineup CSV |
| `--slate SLATE` | Select one slate's lineups |
| `--out PATH` | Override the generated upload path |
| `--overwrite` | Replace the base upload instead of writing a revision |
| `--no-adopt` | Do not adopt misnamed DK files |
| `--downloads` | Search Downloads and copy a matching template into `dk_lineups` |
| `--list` | Display selected generated lineups without writing |

Template shorthand is date-aware. For example, `--date 2026-08-09 --template turbo`
prefers a `turbo` template whose embedded DK player list is dated `2026-08-09`, even when
older `turbo` templates remain in `dk_lineups\`. An explicitly named file carrying a
different date is rejected rather than uploaded.

### `dfs.lateswap`

| Argument | Meaning |
| --- | --- |
| `--date DATE` | Slate date |
| `--file PATH` | Filled upload/swap to repair; defaults to the newest matching file |
| `--slate SLATE` | Select one slate |
| `--out PATH` | Override the swap output path |
| `--in-place` | Replace the exact input file instead of writing a copy |
| `--overwrite` | Replace the base swap output instead of writing a revision |
| `--objective OBJECTIVE` | Replacement objective: `ceiling`, `proj`, or `floor` |
| `--dry-run` | Report proposed changes without writing |

### `dfs.streamfinder`

| Argument | Meaning |
| --- | --- |
| `--date DATE` | Slate date; defaults to today |
| `--slate SLATE` | Slate label, e.g. `main`; defaults to the date's only upload |
| `--upload PATH` | Explicit upload CSV, bypassing the date lookup |
| `--out PATH` | Output file; default `StreamFinder.txt` |
| `--top N` | How many players to prioritise; default `30` |
| `--ignore TEAM_ID` | MLB team id ignored globally; repeatable |
| `--on-deck {Y,N}` | StreamFinder on-deck setting; default `N` |
| `--include-cli {Y,N}` | StreamFinder include-CLI setting; default `Y` |
| `--delay MS` | Feed delay in milliseconds; default `5000` |
| `--root DIR` | Board output root searched for the upload |

### `dfs.review`

| Argument | Meaning |
| --- | --- |
| `--date DATE` | Contest date |
| `--salaries PATH` | Explicit DK salary export |
| `--slate SLATE` | Select one slate |
| `--entered [PATH]` | Score entered lineups; bare flag finds the newest upload/swap |
| `--usage-top N` | Rows in entered-lineup usage output; `0` shows all |
| `--criteria SETTINGS` | Re-run one comma-separated optimizer configuration |
| `--criteria-lineups N` | Lineups generated for the criteria test |
| `--sweep` | Test all shipped optimizer configurations; slow |
| `--sweep-lineups N` | Lineups generated per sweep configuration |
| `--no-snapshot` | Rebuild from mutable cache instead of the frozen snapshot |
| `--stage STAGE` | Snapshot stage to score against |
| `--candidates DIR` | Grade whether a saved candidate pool contained the needed lineup |

### `dfs.results`

| Argument | Meaning |
| --- | --- |
| `--dir DIR` | Override the DK standings directory |
| `--date DATE` | Match exports to this date |
| `--slate SLATE` | Match one slate label |
| `--top N` | Ownership rows displayed |

### `dfs.evaluate`

| Argument | Meaning |
| --- | --- |
| `--dates LIST` | Explicit comma-separated dates; otherwise uses every cached date |
| `--from DATE` | First inclusive evaluation date |
| `--to DATE` | Last inclusive evaluation date |
| `--data-dir DIR` | Override cached report data |
| `--no-snapshots` | Re-project from mutable cache rather than frozen snapshots |
| `--no-recalibrate` | Use shipped floor/bust constants instead of walk-forward fitting |
| `--min-train-rows N` | Historical player-games required before scoring a date |
| `--min-segment N` | Smallest segment included in the report |
| `--label TEXT` | Output filename suffix |
| `--csv PATH` | Also write collected player-game rows to CSV |

### `dfs.candidates`

| Argument | Meaning |
| --- | --- |
| `--date DATE` | Slate date |
| `--slate SLATE` | Select one DK export |
| `--salaries PATH` | Use an explicit salary export |
| `--snapshot` | Build from a frozen snapshot |
| `--stage STAGE` | Snapshot stage used with `--snapshot` |
| `--n N` | Number of candidate lineups |
| `--objective OBJECTIVE` | `Ceiling`, `Proj`, `Floor`, or `"Lev Score"` |
| `--randomness FRACTION` | Candidate objective jitter |
| `--seed N` | Reproduce candidate generation |
| `--lock PLAYERS` | Comma-separated required players |
| `--exclude PLAYERS` | Comma-separated excluded players |
| `--focus-teams TEAMS` | Restrict deliberate stacks to these teams |
| `--max-ownership PCT` | Maximum summed projected ownership |
| `--min-salary AMOUNT` | Minimum lineup salary |
| `--sizes LIST` | Stack core sizes to enumerate |
| `--prune` | Remove deterministically dominated players before generation |
| `--out DIR` | Candidate-pool output directory |
| `--profile` | Record pipeline timing |

### `dfs.simulate`

| Argument | Meaning |
| --- | --- |
| `--date DATE` | Slate date |
| `--slate SLATE` | Select one DK export |
| `--snapshot` | Simulate a frozen snapshot |
| `--stage STAGE` | Snapshot stage used with `--snapshot` |
| `--sims N` | Number of correlated outcome simulations |
| `--seed N` | Reproduce the simulation |
| `--actuals` | Pull final box scores and validate the distribution |
| `--candidates DIR` | Score a saved candidate pool |
| `--top N` | Rows displayed in ranked output |
| `--label TEXT` | Output label/suffix |
| `--profile` | Record pipeline timing |

### `dfs.field`

| Argument | Meaning |
| --- | --- |
| `--date DATE` | Slate date |
| `--slate SLATE` | Select one DK export |
| `--snapshot` | Use a frozen snapshot |
| `--stage STAGE` | Snapshot stage used with `--snapshot` |
| `--contest PRESET` | `milly-maker`, `single-entry`, `three-max`, or `twenty-max` |
| `--entries N` | Override preset field size |
| `--entry-fee AMOUNT` | Override preset entry fee |
| `--max-entries N` | Override maximum entries per user |
| `--seed N` | Reproduce field generation |
| `--calibrate` | Refit field constants from `dk_results` and stop |
| `--profile` | Record pipeline timing |

### `dfs.contest`

| Argument | Meaning |
| --- | --- |
| `--date DATE` | Slate date |
| `--slate SLATE` | Select one DK export |
| `--snapshot` | Use a frozen snapshot |
| `--stage STAGE` | Snapshot stage used with `--snapshot` |
| `--contest PRESET` | `double-up`, `milly-maker`, `single-entry`, `three-max`, or `twenty-max` |
| `--entry-fee AMOUNT` | Override contest entry fee |
| `--field-size N` | Real contest size used by the payout model |
| `--field-entries N` | Opponent lineups actually simulated and scaled to the field |
| `--payout-json PATH` | Exact payout structure |
| `--candidates N` | Candidate lineups generated for scoring |
| `--sims N` | Contest simulations |
| `--max-ownership PCT` | Maximum summed projected ownership |
| `--from-optimizer [PATH]` | Score optimizer lineups; bare flag finds the newest set |
| `--allow-started` | Keep started players for analysis only |
| `--seed N` | Reproduce the contest simulation |
| `--top N` | Ranked rows displayed |
| `--by COLUMN` | Output column used for ranking |
| `--out PATH` | Write scored candidates to CSV |
| `--profile` | Record pipeline timing |

---

## Tests and documentation

Focused checks are preferred during development:

```powershell
& $PY -m pytest tests\test_tier_audit.py tests\test_position_strategy.py `
    -q -p no:cacheprovider
& $PY -m pytest tests\test_optimizer.py -q -p no:cacheprovider
```

Full suite when appropriate:

```powershell
& $PY -m pytest tests\ -q -p no:cacheprovider
```

Profiling:

```powershell
& $PY -m dfs.cli --date $DATE --slate $SLATE --profile
& $PY -m dfs.optimize --date $DATE --slate $SLATE --n 20 --profile
```

Additional references:

| File | Purpose |
| --- | --- |
| `README.md` | Feature-level usage and examples |
| `docs\current_architecture.md` | Module map, data flow, and correctness boundaries |
| `docs\simulation_and_portfolio_plan.md` | Simulation and portfolio design |
| `docs\benchmarks.md` | Measured results and rejected experiments |
