# CLI Cheatsheet

> **Always call `.\.venv\Scripts\python.exe`, or activate the venv first.**
> Plain `python` can resolve to a system install ahead of `.venv`
> (check with `Get-Command python -All`). Both entry points detect this and say so.
>
> ```powershell
> .\.venv\Scripts\Activate.ps1     # once per session, then plain `python` is safe
> ```
>
> Commands below are written as `python ...` assuming the venv is active.

---

## Daily workflow

```powershell
# 1. Drop the contest's DKSalaries.csv into dfs_daily_files/

# 2. Generate scouting reports for the slate      (~8 min per game, once)
python scouting_report.py --date 2026-07-27 --all-games --format xlsx

# 3. Build the DFS board                          (seconds, reads the cache)
python -m dfs.cli --date 2026-07-27

# 4. Generate lineups
python -m dfs.optimize --date 2026-07-27 --n 20 --objective ceiling --stack-shape 4-3

# 5. After lineups are confirmed, refresh          (~45s for a full slate)
python scouting_report.py --date 2026-07-27 --all-games --format xlsx `
    --from-cache --fast --refresh-lineups
# second pass so every game's DFS highlighting sees the fully-refreshed slate
python scouting_report.py --date 2026-07-27 --all-games --format xlsx --from-cache --fast
python -m dfs.cli --date 2026-07-27

# 6. Re-optimize on the confirmed pool and write the DK upload file
python -m dfs.optimize --date 2026-07-27 --n 20 --objective ceiling --stack-shape 4-3
python -m dfs.upload --date 2026-07-27 --lineups 1,3,5-8
```

Check what's actually posted before step 5:

```powershell
python scouting_report.py --date 2026-07-27 --confirmed-lineups
```

**Step 5 is two passes on purpose.** Highlighting is rebuilt from the whole slate at
render time, but `--refresh-lineups` refreshes one game at a time — so the first games
rendered get tinted against a pool where the later games still hold their pre-lineup
guesses. The second pass costs ~45s and makes the set consistent.

---

## Scouting reports — `scouting_report.py`

### Generating

```powershell
python scouting_report.py --date 2026-07-27 --away-team CHC --home-team STL   # one game
python scouting_report.py --date 2026-07-27 --all-games                       # whole slate
python scouting_report.py --date 2026-07-27 --all-games --skip-existing       # resume
python scouting_report.py --date 2026-07-27 --all-games --upcoming-only       # skip started
```

| Flag                                 | Effect                                                        |
| ------------------------------------ | ------------------------------------------------------------- |
| `--format`                         | `xlsx` (default), `pdf`, `markdown`, `html`, `both` |
| `--game-number N`                  | Pick by schedule position instead of team abbreviations       |
| `--output-dir` / `--flat-output` | Where reports land; flat skips the dated subfolder            |
| `--include-plots`                  | Heatmap images (slow, PDF only)                               |
| `--similarity-lookback N`          | Seasons used for similar-pitcher samples                      |

### Re-rendering from cache

```powershell
# fast: skips ~14s/game of API refresh against <1s of rendering
python scouting_report.py --date 2026-07-27 --all-games --format xlsx --from-cache --fast

# picks up confirmed lineups (a plain re-render will NOT)
python scouting_report.py --date 2026-07-27 --all-games --format xlsx `
    --from-cache --fast --refresh-lineups
```

| Flag                   | Effect                                                                            |
| ---------------------- | --------------------------------------------------------------------------------- |
| `--from-cache`       | Re-render from`.cache/report_data/` instead of pulling fresh                    |
| `--fast`             | Skip every API refresh. Only boxscores, sweeps, schedule spot and umpire go stale |
| `--refresh-lineups`  | Re-pull lineups, then rebuild splits / hot-cold / offense index / composite       |
| `--no-dfs-highlight` | Turn off the value tint on player names                                           |
| `--dfs-slate`        | Which DK export to tint from, when the date has several                           |

**Highlighting picks its own slate.** With two exports on one date (early + main), each
report is tinted from the export that actually prices *its* game — so an early-slate game
is tinted off the early file without being told. Only a game priced on **both** is
ambiguous, and then it says so rather than rendering untinted:

```
⚠️ DFS highlighting off: 2 DK exports match 2026-07-28
   (DKSalaries_2026-07-28_Main.csv, DKSalaries_2026-07-28_late.csv); pass --dfs-slate
```

**A plain `--from-cache` replays the cached lineup**, which is usually a `Fallback` guess
made before lineups posted. Without `--refresh-lineups` it will never show confirmations.

### Doubleheaders

The two games have different starters and lineups, so the tools refuse to guess:

```powershell
python scouting_report.py --date 2026-07-28 --away-team CLE --home-team CIN --format xlsx
# ⚠️ CLE at CIN is a doubleheader on 2026-07-28. Choose with --dh-game:
#      --dh-game 1  2026-07-28T17:40:00Z  gamePk 824490
#      --dh-game 2  2026-07-28T23:10:00Z  gamePk 824489

python scouting_report.py --date 2026-07-28 --away-team CLE --home-team CIN `
    --format xlsx --dh-game 2
```

Game 2 is cached and written as `..._CLE_CIN_g2`, so it never overwrites game 1. On the
DFS side, if both games are cached the board keeps only the one whose first pitch matches
a start time in the DK export — otherwise every player on those teams would appear twice.

Team Performance splits on **`DH game 2`** and **`Day after DH`**, and tonight's row is
highlighted per game (`--dh-game 2` reads as `DH game 2`; the next day reads
`Day after DH`).

### When MLB hasn't posted a probable pitcher

The run bails and caches nothing. DK's salary file often confirms starters hours earlier:

```powershell
python scouting_report.py --date 2026-07-27 --away-team CHC --home-team STL `
    --home-pitcher "Matthew Liberatore" --format xlsx
```

`--away-pitcher` / `--home-pitcher` take an MLBAM id or a name.

### Model maintenance

```powershell
python scouting_report.py --date 2026-07-27 --confirmed-lineups   # who has lineups posted
python scouting_report.py --date 2026-07-27 --qc-reports          # QC scan -> CSV
python scouting_report.py --calibrate --calibration-days 30       # refit win/run model
python scouting_report.py --backtest --backtest-days 30           # per-game prediction CSV
```

---

## DFS board — `python -m dfs.cli`

```powershell
python -m dfs.cli --date 2026-07-27
python -m dfs.cli --date 2026-07-27 --print                    # echo to terminal
python -m dfs.cli --date 2026-07-27 --top-per-position 12
```

| Flag                                        | Effect                                                        |
| ------------------------------------------- | ------------------------------------------------------------- |
| `--salaries PATH`                         | Explicit DK export; otherwise searched in`dfs_daily_files/` |
| `--slate NAME`                            | Pick between multiple exports by filename substring           |
| `--list-slates`                           | Show every DK export found for the date, then stop            |
| `--data-dir`                              | Cached report-data directory                                  |
| `--output-dir`                            | Where the board and CSVs land (default`dfs_boards/`)        |
| `--top-pitchers` / `--top-per-position` | Rows shown per section                                        |

### Multiple DK slates on one day

**Rename exports however you like** — they are identified by contents, not filename. A
useful convention is `DKSalaries_<date>_<label>.csv`:

```text
dfs_daily_files/
  DKSalaries_2026-07-27_main.csv
  DKSalaries_2026-07-27_early.csv
  DKSalaries_2026-07-27_turbo.csv
```

```powershell
python -m dfs.cli --date 2026-07-27 --list-slates      # what's on hand
python -m dfs.cli --date 2026-07-27 --slate main       # pick one
python -m dfs.optimize --date 2026-07-27 --slate main --n 20
```

`--list-slates` fingerprints each file from its own contents:

```text
  DKSalaries_2026-07-27_main.csv
      7 games, 7:10PM-9:45PM ET (643 players)
      ATL@NYM, BOS@ATH, CHC@STL, CLE@CIN, HOU@LAA, MIL@SF, NYY@CWS
  DKSalaries_2026-07-27_turbo.csv
      2 games, 9:40PM-9:45PM ET (189 players)
      BOS@ATH, MIL@SF
```

**If two exports match and you don't name one, the tools stop and list them** rather than
guessing. Silently picking would price the whole board off a contest you aren't entering.
Downloading twice gives you `DKSalaries (1).csv`, so this is easy to hit.

**Outputs** — one folder per night, one set of files per slate:

```
dfs_boards/2026-07-28/
  board_main.md      slate_main.csv     stacks_main.csv
  pool_main.csv      lineups_main.csv   upload_main.csv
  board_late.md      slate_late.csv     ...
```

`python -m dfs.files --date 2026-07-28` lists a night and says what each file is for.

**Nothing is overwritten by default.** Re-running writes `board_main.r2.md` beside
`board_main.md` and says so; pass `--overwrite` to replace in place. Readers (`dfs.upload`,
`dfs.lateswap`) always take the most recently written version.

**The slate label is part of every name**, because a date alone does not identify a contest.
It comes from `--slate`, else the export's filename, else its first-pitch window
(`early` / `main` / `late` / `showdown-<matchup>`).

**Misnamed downloads get filed automatically.** DK hands you `DKSalaries (2).csv` whatever
the file actually is; before each run the tools identify strays *by contents* and rename
them into the convention, printing what moved. `--no-adopt` turns it off, `--downloads`
also searches `~/Downloads` (copying, never moving). Standalone:
`python -m dfs.adopt --date <date> --dry-run`.

**Two warnings it will give you, both worth stopping for:**

- *INCOMPLETE SLATE* — a DK-priced game has no cached report, so that whole team is missing
- *WRONG SALARY FILE* — the export is for a different date; every salary and tier is meaningless

---

## Optimizer — `python -m dfs.optimize`

```powershell
python -m dfs.optimize --date 2026-07-27 --n 20                             # GPP default
python -m dfs.optimize --date 2026-07-27 --n 20 --objective floor           # cash
python -m dfs.optimize --date 2026-07-27 --n 20 --stack-shape 4-3           # explore shapes
python -m dfs.optimize --date 2026-07-27 --n 20 --stack "CWS:4,HOU:3"       # explicit stacks
python -m dfs.optimize --date 2026-07-27 --lock "Max Fried" --exclude "Jo Adell"
```

| Flag                            | Effect                                                                                            |
| ------------------------------- | ------------------------------------------------------------------------------------------------- |
| `--objective`                 | `ceiling` (GPP, default), `floor` (cash), `proj` (balanced)                                 |
| `--n N`                       | Lineups to generate                                                                               |
| `--lock` / `--exclude`      | Comma-separated names, accent- and case-insensitive                                               |
| `--stack "TEAM:N,..."`        | Explicit team stacks                                                                              |
| `--stack-shape "4-3"`         | Explore that shape across the best stacking teams                                                 |
| `--focus-teams "CIN,NYY,SD"`  | Build stacks only from these teams; other teams still fill leftover slots                        |
| `--max-overlap N`             | Max shared players between lineups (**default 6**; 9 = DK minimum, blocks exact dupes only) |
| `--randomness`                | Objective jitter (**default 0.20**); `0` for the single best lineup                       |
| `--seed N`                    | Reproducible randomness — without it, each run differs                                           |
| `--min-proj` / `--max-bust` | Prune the pool before solving                                                                     |
| `--max-from-team N`           | Hitters from one team (DK allows 5)                                                               |
| `--no-stack-bonus`            | Disable the stack correlation bonus                                                               |
| `--conflict-penalty X`        | Tax a pitcher rostered against your own stack (**default 1.0** × his projection); `0` allows it |
| `--conflict-min-hitters N`    | Opposing hitters that make a stack "prominent" enough to tax (**default 3**)                     |
| `--save-config`               | Persist settings to`dfs_daily_files/optimizer.json`                                             |

Enforces the DK Classic roster (2 P, C, 1B, 2B, 3B, SS, 3 OF), the $50,000 cap,
max 5 hitters per team, and the two-game minimum.

**Defaults are tuned for tournaments** — `--max-overlap 6 --randomness 0.20`. Over 10
lineups that gives 63 unique players at 1.4 mean overlap, versus 24 players at 6.5 overlap
under the DK-minimum settings. It projects ~8 points lower per lineup on purpose:
concentration and variance are how you reach a winning score.

For a single best lineup (or cash), turn both off:

```powershell
python -m dfs.optimize --date 2026-07-27 --n 1 --randomness 0 --objective floor
```

Runs are non-deterministic now. Add `--seed N` to reproduce a set.

**Stacking from a hand-picked set of teams.** Pair `--focus-teams` with `--stack-shape` to
say *what* the stacks look like and *who* they come from:

```powershell
python -m dfs.optimize --date 2026-07-28 --n 6 --focus-teams "CIN,NYY,SD" --stack-shape 4-3
```

That walks every ordering of the named teams — CIN 4 / NYY 3, NYY 4 / SD 3, and so on — so
a set of six covers all six pairings of three teams. The shape only claims 7 of the 8 hitter
slots; the rest go to the best value on the board from any team, and **pitchers are never
restricted** (the conflict tax below still keeps them off your own stack).

Team codes are alias-folded, so `CHW`, `OAK`, and `AZ` find CWS, ATH, and ARI. A code with no
hitters on the slate is reported and dropped rather than silently shrinking the stack pool.

`--focus-teams` alone (no shape) is softer: it confines the correlation bonus to those teams,
so the *deliberate* stacks form there, but nothing forces one — a lineup may still come back
built purely on value. Use a shape when you want the stack guaranteed. Note that incidental
pairs from other teams can still appear; they are two good prices that share a jersey, not a
bonus-driven stack.

**Pitcher-against-your-own-stack is taxed.** Rostering a starter opposite 3+ of your own
hitters bets on both sides of one game: the runs your stack scores are the runs charged to
him, so summing the two counts points that can only be collected once. The default penalty
(1.0 × the pitcher's projection) wipes out his value, so any comparable arm takes the slot
and the pairing all but vanishes — but it is a tax, not a ban, so a locked pitcher and a
forced stack still solve. Affected lineups print `[!] <pitcher> pitches against your own
<TEAM> x<N>`. Use `--conflict-min-hitters 1` to also fade single opposing bats, or
`--conflict-penalty 0` to allow the pairing outright.

### Uploading to DraftKings — `python -m dfs.upload`

Drop DK's upload file in `dk_lineups/` — **don't bother renaming it.** It arrives as
`DKSalaries (3).csv` or similar; `dfs.upload` identifies it by contents and files it as
`DKTemplate_<date>_<layout>.csv` on the way through. Templates are matched to `--date` by
the date inside their own player list, so last night's file is never offered as a choice.

Then pick which generated lineups to put in it:

```powershell
python -m dfs.upload --date 2026-07-28 --list             # see what was generated
python -m dfs.upload --date 2026-07-28 --lineups 1,3,5-8  # write those eight
python -m dfs.upload --date 2026-07-28                    # all of them
```

Writes `dfs_boards/<date>/upload_<slate>.csv`, ready to upload. The template is never modified.

| Flag          | Effect                                                                 |
| ------------- | ---------------------------------------------------------------------- |
| `--lineups`   | `1,3,5-8`, or omit for all                                             |
| `--list`      | Show generated lineups with salary/proj/ceiling and stop               |
| `--slate`     | Which slate's lineups to upload, if the night has more than one        |
| `--template`  | Which upload file to fill; a distinctive part of the name is enough    |
| `--no-adopt`  | Don't rename misnamed DK downloads into the convention                 |
| `--downloads` | Also look in `~/Downloads` for the template (copies, never moves)      |
| `--contest`   | Entry-export templates: fill only entries whose contest name matches   |
| `--source`    | A different optimizer lineup CSV                                       |
| `--out`       | Write somewhere else                                                   |
| `--overwrite` | Replace an existing upload file instead of writing `.r2`               |

Or do it in one step from the optimizer:

```powershell
python -m dfs.optimize --date 2026-07-28 --n 20 --upload      # all
python -m dfs.optimize --date 2026-07-28 --n 20 --upload 1-5  # a selection
```

Both DK layouts are handled, detected from the header:

- **Bulk template** (`P,P,C,1B,2B,3B,SS,OF,OF,OF`) — each row is a new lineup, up to 500.
- **Entry export** (`Entry ID,Contest Name,...`) — each row is an entry you already own,
  so you cannot write more lineups than entries. Only the rows being changed are written,
  which is what DK asks for.

Player ids come from the template's own embedded player list, not from the ids stored when
the lineups were built. **DK renumbers every player per draft group**, so a template from a
different contest would otherwise produce a file DK rejects with no indication of which
player was wrong. For the same reason the slate date is checked: last night's lineups match
tonight's names perfectly and would upload without complaint.

### Late swap — `python -m dfs.lateswap`

A game gets postponed, or a hitter is scratched, and the slots those players held are now
dead. This re-reads a file you have **already filled** and refills only the broken slots,
leaving everything else exactly where it was:

```powershell
python -m dfs.lateswap --date 2026-07-28 --dry-run   # what would change
python -m dfs.lateswap --date 2026-07-28             # -> dfs_boards/<date>/swap_<slate>.csv
python -m dfs.lateswap --date 2026-07-28 --in-place  # overwrite the file it read
```

Or from the optimizer, so the evening check is the same command as the morning build:

```powershell
python -m dfs.optimize --date 2026-07-28 --swap             # check and refill, nothing else
python -m dfs.optimize --date 2026-07-28 --n 20 --upload --swap   # build, upload, then check
```

`--swap` on its own **skips generation entirely** — it will not rebuild the lineups behind
entries you have already submitted. It always writes `dfs_boards/<date>/swap_<slate>.csv`,
never over the file it read, so the record of what is actually at DK survives a bad swap.
Re-run it as often as you like through the evening.

```
re-read dfs_boards\2026-07-28\upload_main.csv (bulk)
  postponed: ATL, NYM - Postponed (Inclement Weather)
  locked (game started): CIN, CLE
  row 2: P (slot was empty), OF (slot was empty) -> Taj Bradley [P], Yordan Alvarez [OF]
  row 3: 2B (off the slate) -> Anthony Seigler [2B]
3 lineup(s) swapped, 1 untouched, 0 failed
```

| Flag          | Effect                                                                   |
| ------------- | ------------------------------------------------------------------------ |
| `--file`      | The filled file to re-read. Default: the night's newest `upload_`/`swap_` |
| `--out`       | Where to write. Default `dfs_boards/<date>/swap_<slate>.csv`             |
| `--in-place`  | Overwrite the file that was read                                          |
| `--dry-run`   | Report and write nothing                                                  |
| `--objective` | What to maximize when picking replacements (default `ceiling`)           |

Two DK rules drive the behaviour, and both are enforced:

- **Replacements only come from games that have not started.** Start times come from the
  live MLB schedule (`abstractGameState`), not from parsing DK's `10:10PM ET` strings, so
  there is no timezone maths to get wrong. Resolved **per game, not per team**, so a
  doubleheader nightcap stays fully swappable while the opener is in progress. If the
  lookup fails you get a warning telling you to check before uploading — it does not
  silently assume nothing has started.
- **Kept players never move column.** DK compares an edited entry slot by slot, and there
  are two P columns and three OF columns; a kept player drifting from OF1 to OF2 reads as
  two edited slots, one possibly locked. Keepers are *pinned* to their exact column and
  their original cell is copied across verbatim.

Both layouts work. An entry export stores cells as `Bailey Ober (43695164)` and a bulk
template stores the bare id — replacements are written in whichever form the file already
uses. Entry IDs, contest columns, the embedded player list, and any reservation row with no
lineup yet are all left untouched.

A lineup that cannot be legally refilled is **left exactly as it was** and reported; a
half-swapped entry would be worse than the one you have. Re-running is safe: with nothing
broken it prints `nothing to swap` and writes no file.

> **Doubleheader nightcap projections lean on the opener's starter** until MLB announces
> game 2. The cached report has no other pitcher to use, so the opposing hitters are
> projected against an arm that will not throw. The starter himself is now dropped from the
> pool, but the hitters facing him keep a projection built on the wrong matchup — regenerate
> the `_g2` report once the real starter is posted.

> Scratches are only visible once the game-report cache has seen the new lineup card — the
> slate is built from cached reports, which hold just the nine hitters on each card. Refresh
> with `--refresh-lineups` before swapping on a scratch. Postponements need no refresh; they
> come from the live schedule.

**Optimizing after first pitch.** A player whose game has started cannot be drafted at all,
so the optimizer drops them from the pool:

```
[!] 24 player(s) removed — MIL@KC already under way and cannot be drafted.
```

A second, nastier case is caught too — **the doubleheader opener's starter**:

```
[!] Chase Burns removed — already pitching the earlier game of a doubleheader.
    DK prices the whole staff against the nightcap, so he is listed as available
    and would score nothing.
```

DK prices a team's entire staff against the nightcap's start time, and until MLB names the
game-2 starter the cached report falls back to the game-1 arm. The result looks like a
bargain rather than a mistake: a top-salary ace with a full projection whose game, per DK,
has not started. The live schedule's probable pitchers are the authority on who is actually
throwing. Late swap treats such a player as a **broken slot** rather than a keeper — unlike
a locked player, DK will happily let you move him, and he cannot score.

Start times come from the live MLB schedule, the same source late swap uses, and are
matched **per game**. That matters on a doubleheader: the slate resolves to one half (the
nightcap, for a main slate), and only that half's status counts. Checking at team level
deleted CIN and CLE from a `--slate main` board while the priced 7:10 nightcap had not
thrown a pitch. Three guards worth knowing:

- If **every** game has started, nothing is dropped — that is a review or backtest, not a
  live build, and filtering would just empty the pool. (`dfs.review` calls the optimizer
  directly and never passes through this at all.)
- A pairing the schedule does not know about is never dropped on a guess.
- If the remaining games are fewer than DK's two-game minimum, it says so rather than
  failing with a vague infeasibility.

`--allow-started` keeps them, for analysis and "what could I have built" questions. Expect
projections to fall once the filter bites: on a half-played slate the best arms are usually
already pitching, and those points were never available to you.

### Reviewing a night

```powershell
python -m dfs.review --date 2026-07-27                  # tiers, teams, best buildable lineup
python -m dfs.review --date 2026-07-27 --entered        # how the lineups you uploaded did
python -m dfs.review --date 2026-07-27 --sweep          # test the shipped optimizer settings
python -m dfs.review --date 2026-07-27 `
    --criteria "objective=ceiling,max_overlap=4,randomness=0.3"   # test one ad-hoc config
```

Scores your flags and lineups against actual DK results, and shows the highest-scoring
lineup that was buildable — the night's ceiling, not a target.

`--entered` reads back what you actually uploaded and scores it lineup by lineup:

```
=== lineups you entered: upload_main.csv (bulk) ===
 Lineup  N  Scored  Salary  Proj  Ceiling  Actual  Diff
      1 10      10   49900  96.8    197.4  102.60   5.8
      ...
  10 lineup(s): best 102.6, mean 84.6, worst 52.8  (43% of the night's ceiling)
  best buildable was 238.7; your best left 136.1 on the table.
```

Bare `--entered` finds the night's `upload_<slate>.csv` (or the `swap_` file if you swapped);
pass a path to score a specific one. The gap between your best and the best buildable is the
number that says whether a night was lost on **selection** rather than on variance.

`--criteria` re-runs the optimizer under settings that were never shipped as a preset —
`objective`, `max_overlap`, `randomness`, `stack_bonus`, `min_proj`, `max_bust`,
`max_hitters_per_team`, `min_games`, `conflict_penalty`, `seed`.

| Flag                 | Effect                                                        |
| -------------------- | ------------------------------------------------------------- |
| `--entered [PATH]` | Score the lineups actually uploaded                           |
| `--criteria`       | `name=value,...` optimizer settings to test                 |
| `--sweep`          | Test all eight shipped configs (slow — N solves each)         |
| `--slate`          | Which slate, when the night had more than one                 |

### Leverage and ownership

The board estimates what fraction of the field will roster each player and turns that into
leverage. Ownership is modelled from the DK salary file alone (season average per $1k,
season average, price, batting order) — never from our own projection, or leverage would be
circular.

| Column      | Meaning                                                         |
| ----------- | --------------------------------------------------------------- |
| `Own%`    | Projected field ownership. Sums to 800 across hitters, 200 across pitchers |
| `Own Pct` | That ownership as a percentile within pitchers / hitters        |
| `Leverage`| `Ceiling Pct - Own Pct`. Positive = we like him more than the field |
| `Lev Score`| `Ceiling - 0.12 x Own%`, the points-scale version the solver maximizes |
| `Own Src` | `est` for the model, `DK` once real %Drafted is imported        |

```powershell
python -m dfs.optimize --date 2026-07-30 --n 20 --objective leverage
python -m dfs.optimize --date 2026-07-30 --n 20 --max-ownership 100
```

`--max-ownership` caps the lineup's *total* projected ownership. It is the cleanest way to
force differentiation: unlike banning players or forcing stacks it constrains the lineup's
relationship to the field while leaving every individual pick on merit. Ten chalk plays sum
near 300; a field-beating lineup is usually well under 150.

Measured on 2026-07-30 main (perfect lineup = 215.7):

| build                  | best of 20 | total ownership |
| ---------------------- | ---------- | --------------- |
| `ceiling` (default)  | 128.7      | 152%            |
| `leverage`           | 132.7      | **86%**         |
| `ceiling` + cap 100  | **137.6**  | 94%             |

Same points, roughly half the field overlap — which is the only thing that converts a score
into a finish.

**A caution on stacking flags.** `--focus-teams` only *prefers* stacks: it confines the
correlation bonus, and that bonus scales with the objective's own magnitude, so under
`--objective floor` it can be worth a fraction of a point and lose to individual merit. For a
guaranteed stack use the hard constraint, `--stack "LAD:4,WSH:3"`. The optimizer now says so
when focus teams come back unstacked.

### Editable pool file

```powershell
python -m dfs.optimize --date 2026-07-27 --write-pool
# edit dfs_boards/2026-07-27/pool_main.csv in Excel
python -m dfs.optimize --date 2026-07-27 --pool --n 20
```

| Column                 | Accepts                                | Effect                                               |
| ---------------------- | -------------------------------------- | ---------------------------------------------------- |
| `Lock` / `Exclude` | `1`, `y`, `yes`, `x`, `true` | Force in / never use                                 |
| `Boost`              | `1.25`, or a range `1.1-1.3`       | Multiplies the objective; a range redraws per lineup |
| `Min%` / `Max%`    | `30`, `30%`, `0.3`               | Share of lineups the player may appear in            |

**Boost and exposure do different jobs.** A boost only raises value, so a boosted player
lands in nearly every lineup. To actually spread someone out, use `Max%`. Minimums are
exact, not soft — `Min% 60` over 5 lineups means 3.

Pool entries combine with `--lock`/`--exclude` rather than replacing them.

**Re-exporting merges; it does not overwrite.** A slate is rarely finished when you first
look at it — games cache late, a postponement reshuffles the board — so you can re-run
`--write-pool` at any point and keep everything you have already decided:

```
pool -> dfs_boards\2026-07-28\pool_main.csv
  merged with the existing file: 29 edit(s) kept, 149 player(s) new to the slate.
  [!] 2 edited player(s) are no longer on the slate: Chase Burns, Austin Martin.
```

Players already in the file keep their settings, players new to the slate arrive with
defaults, and **edits for players who have since left the slate are kept as trailing rows**
with the slate columns blank rather than thrown away. That last part is deliberate: a
player can drop out because a game is merely uncached, and losing a decision to a temporary
gap would be worse than a couple of stray rows. The optimizer already reports a lock it
cannot find and carries on, and re-exporting again will not duplicate them.

Matching is on name **plus team**, so two players sharing a name never inherit each other's
settings. `--reset-pool` rebuilds from scratch and discards every edit, for when you
genuinely want a clean sheet.

---

## Backtest

```powershell
python -c "from dfs.backtest import run_backtest, summarize; print(summarize(run_backtest(['2026-07-24','2026-07-25'])))"
```

Pulls final box scores, converts to DK points, joins on MLBAM id. Reports bias, MAE, rank
correlation, top-vs-bottom quintile separation, and whether the ceiling/floor bands hit
their stated percentiles. Run it after any change to `dfs/projections.py`.

---

## Reading the board

| Column                  | Meaning                                                               |
| ----------------------- | --------------------------------------------------------------------- |
| `Proj`                | Mean DK-point expectation                                             |
| `Ceiling` / `Floor` | ~90th / 25th percentile outcomes, fit from actual results             |
| `Bust%`               | Chance of 3 or fewer points                                           |
| `Value`               | Points per $1,000. ~2.0 is break-even for hitters                     |
| `Edge`                | How far the projection ranks above the player's DK season average     |
| `GPP` / `CASH`      | Tournament score (ceiling per dollar) / cash score (floor per dollar) |
| `Tier`                | Is this priced well: Core, Value, Leverage, Risk, Fade                |
| `Role`                | What kind of play: Ceiling, Floor, Ceiling+Floor                      |
| `Why`                 | Supporting factors; anything after`[risk]` is a caution             |

Pitchers and hitters are ranked **within type** — their scores don't compare across types.

**27% of hitter-games score zero and 44% score 3 or fewer, regardless of matchup.** There
is no safe MLB hitter in a single game; real floor plays are pitchers.

---

## Troubleshooting

| Symptom                                       | Cause                                                                                                    |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `No module named 'sklearn'` / `'pyarrow'` | Wrong interpreter — use`.\.venv\Scripts\python.exe`                                                   |
| `INCOMPLETE SLATE`                          | A priced game has no cached report; generate it, using`--home-pitcher` if MLB hasn't posted a probable |
| `WRONG SALARY FILE`                         | The DK export is for another date                                                                        |
| `Could not write ... file open in Excel?`   | Close the workbook and re-run                                                                            |
| Board missing a whole team                    | Same as INCOMPLETE SLATE — check the warning                                                            |
| Confirmed lineups not showing                 | Add`--refresh-lineups` to the cache re-render                                                          |
| Highlighting unchanged after a lineup refresh | Run the plain`--from-cache --fast` second pass (step 5)                                                |
| Lineups all look the same                     | Lower`--max-overlap` (try 6)                                                                           |
| `POSTPONED`                                 | A game was called off; those players are already removed — nothing to do                                |
| `Could not check for postponements`         | StatsAPI unreachable; re-download DK's file, which marks called games itself                             |
| `several upload templates in dk_lineups/`   | Pass`--template`, or delete the stale one                                                              |
| `the template is for <date>`                | You kept last night's upload file; download tonight's                                                    |
| `not in the template's player list`         | Template and salary file are from different contests; re-download both                                   |
