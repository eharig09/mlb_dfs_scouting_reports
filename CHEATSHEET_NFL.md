# NFL DFS Cheatsheet

The NFL half of the project: the slate report, the figures inside it, and the optimizer.
`CHEATSHEET.md` is the MLB twin and covers the daily pipeline, the scheduler, uploads and
the results audits — **none of which exist on the NFL side yet.** What that side has is
listed honestly below, including the gaps, so nothing here is a command that does not run.

## Quick navigation

| | |
|---|---|
| [Conventions](#conventions) | interpreter, working directory |
| [1. Report generator](#1-report-generator) | `python -m nfl.cli` — the one real CLI |
| [2. Visual explorer](#2-visual-explorer) | the dashboard page + the workbook's Visuals tab |
| [3. Optimizer](#3-optimizer) | `python -m nfl.optimize` + the library API |
| [4. DK upload](#4-dk-upload) | `python -m nfl.upload` — fill the entries file |
| [5. PFF data layer](#5-pff-data-layer) | catalog, deployment profiles, O-line |
| [6. Studies](#6-studies) | re-derive every fitted constant |
| [7. Hosting](#7-hosting) | build the snapshot, push, deploy |
| [What does not exist yet](#what-does-not-exist-yet) | read before looking for it |
| [Traps](#traps) | the ones that cost real time |

---

## Conventions

Run from the repository root:

```text
C:\Users\ehari\Desktop\scouting_report
```

Always the project interpreter. At the start of a PowerShell session:

```powershell
$PY = ".\.venv\Scripts\python.exe"
```

### Running multi-line Python

**`& $PY -c @'...'@` does not work.** Windows PowerShell mangles a here-string passed as a
native command argument — double quotes inside it are eaten, so
`print("missing:", missing)` reaches Python as `print(missing:, missing)` and dies on a
SyntaxError that looks like your code is wrong. Two patterns that do work:

```powershell
# one-liner: double quotes outside, SINGLE quotes inside
& $PY -c "from nfl import pffdata; print(pffdata.catalog().head().to_string())"

# multi-line: write the script, then run it
@'
print("this keeps its quotes")
'@ | Out-File -Encoding utf8 .tmp\demo.py
& $PY .tmp\demo.py
```

Every snippet below uses one of those two.

Commands below use `& $PY`. Tests: `& $PY -m pytest tests/nfl/ -q` — **315 passing**; the dashboard suites add ~90 more.
**Check the exit code from a file, not a pipe**: `... -q > out.txt 2>&1; $LASTEXITCODE`. Piping pytest through `tail` reports *tail's* status, and a run with 7 failures came back "exit 0".
Keep the `test_nfl_*` prefix on new test files, and do **not** add a `conftest.py` under
`tests/nfl/`; fixtures live in `tests/nfl/nfl_fixtures.py` and are imported by name. A
second conftest is ambiguous on sys.path and silently ran 77 MLB tests against an NFL slate.

---

## 1. Report generator

The only NFL entry point with a command line.

```powershell
& $PY -m nfl.cli --salaries "nfl/nfl_dfs/nfl_daily_files/DKSalaries (39).csv" --week 1
```

### Flags

| flag | default | what it means |
|---|---|---|
| `--salaries` | *required* | the DK salary export for the slate |
| `--season` | current season | **roster / depth-chart** season, for cold-start priors |
| `--stats-season` | `2025` | the season the **PFF grade and red-zone files** describe |
| `--week` | none | week for strength of schedule; omit for the season average |
| `--output-dir` | `nfl/nfl_reports/<date>` | where the workbook lands |
| `--no-figures` | off | skip the figures, workbook only |

**`--season` and `--stats-season` are different years and both are real.** One says who is on
the roster now; the other says which season of PFF grades to read. Leaving `--stats-season`
at its default while building a 2026 slate is correct today — 2026 has no PFF grades yet.

### Output

```text
nfl/nfl_reports/<date>/slate_<date>_<teams>.xlsx
nfl/nfl_reports/<date>/figures/*.png
```

Both are gitignored. Tabs, in order:

| tab | what it answers |
|---|---|
| **Visuals** | the four figures, embedded |
| **Board** | the main board — proj, PFF PPG, SOS, salary, value |
| **Red Zone** | RZ / EZ / inside-5 workload and the TD equity it implies |
| **Matchup** | positional SOS and the coverage the opponent actually plays |
| **Cold Start** | role, availability, expected opportunity for weeks 1-3 |

### Reading `basis`

Every board row carries a `basis` saying **which source answered**, and it is first class on
purpose: "PFF says 16.7" and "we inferred it from a third-round pick" must never look alike.
On a preseason slate expect roughly `PFF projection` 61 / `prior season` 22 /
`draft capital` 18 / `none` 17 / `replacement` 8 out of 126.

**A missing number is blank, never `0.00`** — a zero would claim the player was projected to
score nothing. Half a preseason board legitimately has no PFF projection.

### Degradation

If nflverse is unreachable the cold-start priors are skipped and the report is still written
(`[!] no roster published for <season>; cold-start priors skipped`). The PFF half of the
board does not depend on nflverse.

---

## 2. Visual explorer

Two of them: the **dashboard page**, and the workbook's **Visuals tab**.

### The dashboard

```powershell
& $PY -m streamlit run streamlit_app.py --server.port 8512
```

Eleven pages under **NFL** in the sidebar, sharing one filter bar (slate, position, team,
game, price band).

**Explore the PFF data**

| page | answers |
|---|---|
| **Receivers** | how every pass-catcher is deployed — pick your own axes |
| **Targets** | how each club divides its targets |
| **Ground game** | carries, receiving work, red-zone and inside-5 touches |
| **Coverage** | man/zone tendency, slot, receiver scheme splits |
| **Trenches** | projected starting fives with a `basis` per man |

**Slate, price and results**

| page | answers |
|---|---|
| **Defence** | what each defence allows by position — *and what that is worth as a forecast* |
| **Value** | PFF per-game projections against DK salary; best of each price band |
| **Stacks** | every QB + pass-catcher combination a club offers, priced |
| **Teams** | records, scoring environment, DK points produced and allowed |
| **Entries** | the lineups you actually submitted, and how alike they are |
| **Slate** | the pool, lineups and exposure the optimizer wrote |

Data lives in `nfl/nfl_dfs/nfl_daily_files/` (DK salary exports) and `nfl/Lineups/` (filled
DK entries exports). Both are picked up automatically and labelled — a Sunday `main` and a
Sunday `early` block are separated by comparing the week's exports to each other, since both
open at 1:00PM.

**Two things the pages are deliberately careful about.**

*Value is two numbers.* Raw points per $1,000 always favours the cheapest man on the board,
so `Band rank` ranks within a price band instead — the question a roster actually asks.

*Defence allowed is not a forecast.* DK points allowed self-correlates year over year at
RB +0.30, TE +0.19, WR +0.11, QB +0.09. The Defence page shows what a club allowed **and**
that figure regressed toward the league mean by exactly that reliability. A defence five
points above average against receivers projects about half a point above average.

**Check the port first.** You run other apps on 850x, and Streamlit exits with
`Port <n> is not available` rather than picking another:

```powershell
& $PY -c "import socket; s=socket.socket(); s.bind(('0.0.0.0', 8512)); print('free')"
```

Exposure has its own panel rather than a column somewhere, for the reason the MLB side
learned: it once found 54% exposure on a punt owned by 1.6% of the field, and nobody goes
looking for an exposure they do not already suspect.

### The workbook's Visuals tab

Four figures, rendered by `nfl/report_visuals.py`:

| caption | slug | shows |
|---|---|---|
| Board | `board` | the slate's projection/salary landscape |
| Touchdown equity | `redzone` | who owns the goal-line and red-zone work |
| Strength of schedule | `schedule` | positional SOS across the slate |
| Coverage fit | `scheme` | man/zone matchup, display only — see [Traps](#traps) |

Rebuild figures alone, without re-running the whole report:

```powershell
@'
from nfl import slate, report_visuals
board = slate.build_slate("path/to/DKSalaries.csv", season=2025, week=1)
made = report_visuals.build_figures(board, "nfl/nfl_reports/scratch/figures", label="test")
print(made)
'@ | Out-File -Encoding utf8 .tmp\figures.py
& $PY .tmp\figures.py
```

`build_figures(board, output_dir, label="slate", dpi=170)` returns `[(caption, path), ...]`
and **renders only the figures that have data** — a figure whose inputs are missing prints
`[!] <caption> figure unavailable: ...` and is skipped rather than aborting the run.

---

## 3. Optimizer

`nfl/optimizer.py` is the solver; **`nfl/optimize.py` is the CLI around it** — the twin of
`dfs.optimize`. There is still no DK upload writer on the NFL side.

**The two names are not interchangeable.** `nfl.optimizer.optimize` is the solver function;
`nfl.optimize` is the pipeline module. `nfl/__init__.py` deliberately does not re-export the
former, because doing so shadowed the latter.

### The operational loop

```powershell
$SAL = "nfl\nfl_dfs\nfl_daily_files\DKSalaries.csv"

# first use only: create the editable pool
& $PY -m nfl.optimize --salaries $SAL --write-pool

# after editing Lock / Exclude / Boost / Min% / Max% in that csv
& $PY -m nfl.optimize --salaries $SAL --n 150 --pool `
    --objective ceiling --stack-shape "3-1,3" --overwrite
```

Writes `nfl_boards\<date>\lineups_<slate>.csv` and `exposure_<slate>.csv`. Nothing is
overwritten by default — a rerun lands as `.r2` unless you pass `--overwrite`.

| flag | notes |
|---|---|
| `--salaries` | *required* — the DK export |
| `--date` / `--slate` | both inferred; `--slate` also picks between exports |
| `--projections` | CSV with `Name, Proj` and optionally `Ceiling, Floor` |
| `--n` | lineups (default 20) |
| `--objective` | `ceiling` *(default)* / `proj` / `floor` / `leverage` |
| `--stack-shape` | e.g. `"3-1+2"` — see below |
| `--team-exposure` | `"PHI:30-70%,CIN:0-25%"` — how many lineups stack each club |
| `--stack-at` | pass-game players that count as a stack (default 3) |
| `--stack` | explicit, e.g. `"BUF:3,KAN:2"` |
| `--stack-teams` | how many anchors to rotate through (default 6) |
| `--lock` / `--exclude` | comma-separated names |
| `--write-pool` / `--reset-pool` / `--pool` | the editable pool |
| `--allow-out` | keep players DK marks OUT/IR (dropped by default) |
| `--min-proj` / `--max-ownership` / `--max-overlap` / `--randomness` / `--seed` | |
| `--overwrite` | replace in place instead of writing `.rN` |

### Stack shapes

    A       pass-game players from the anchor club, QB included and pinned
    A-B     ...plus B from the other side of that same game — the bring-back
    A-B+C   ...plus a C-man secondary stack from a *different* game

Three numbers because a football stack is three decisions. `A` is the correlation you are
buying, `B` is the hedge that pays in a shootout rather than a blowout, `C` is
diversification — forced to another game, since a secondary stack inside the same game is
just a bigger primary one.

A bare `3` asks for three from the anchor; it does **not** forbid a bring-back. Refused:
a one-man "secondary stack", and any shape that cannot fit beside the RB slots and a DST.

Rank within the passing game is deliberately absent: WR2 correlates with the QB as strongly
as WR1 (+0.359 vs +0.367) at about a quarter less target share.

### Exposure

Player caps and minimums come from the pool's `Min%` / `Max%` columns; club-level ones from
`--team-exposure`. Both are properties of the **set**, not of one lineup.

```powershell
& $PY -m nfl.optimize --salaries $SAL --n 150 --pool `
    --stack-shape "3-1+2,4-1" --team-exposure "PHI:30-70%,CIN:0-25%"
```

Minimums are **paced**, not deferred — a player wanted in 30 of 50 is in roughly 3 of the
first 5. A minimum the set ran out of room for is reported, never raised, so an
oversubscribed board still hands back the lineups it built.

An explicit `--team-exposure` floor overrides the `--stack-teams` shortlist: being outside
the projection top-N is exactly why the rotation would not have reached that club.

### If you need the solver directly

### Roster and cap

```text
QB 1 · RB 2 · WR 3 · TE 1 · FLEX 1 · DST 1   = 9 players, $50,000
FLEX is RB/WR/TE
```

### The frame it expects

`nfl.salaries.load_dk_export(path)` builds this from a DK export and returns
`(players, report)`; the report names who was dropped as OUT/IR and who is questionable.

One row per DK entry. Required: **`Name`**, **`Salary`**, and a position in **`DK Pos`**
(falls back to `Pos`), plus whichever objective column you ask for. `Team` is needed for
stacking, `Own%` for the ownership cap, `Boost` is an optional per-player multiplier.

| objective | column read |
|---|---|
| `ceiling` *(default)* | `Ceiling` |
| `proj` | `Proj` |
| `floor` | `Floor` |
| `leverage` | `Lev Score` |

Asking for an objective whose column is absent raises
`OptimizerError: slate has no 'X' column`.

### Worked example

```powershell
@'
import sys; sys.path.insert(0, "tests/nfl")
from nfl_fixtures import make_slate
from nfl import optimizer as opt

slate = make_slate(n_games=5, seed=7)
lineups, pool, missing = opt.optimize(
    slate, n_lineups=3, objective="ceiling", stacks={"BUF": 3}, seed=11)

for n, L in enumerate(lineups, 1):
    print(n, L["salary"], round(L["proj"], 2), round(L["ceiling"], 2), L["teams"])
    print(L["players"][["Name", "Pos", "Team", "Salary", "Proj"]].to_string(index=False))
print("missing:", missing)
'@ | Out-File -Encoding utf8 .tmp\opt_demo.py
& $PY .tmp\opt_demo.py
```

Verified output (first two lineups):

```text
1 50000 113.79 {'BUF': 4, 'BAL': 2, 'SEA': 1, 'MIA': 1}
2 49800 109.84 {'BUF': 4, 'PHI': 2, 'CIN': 1, 'DAL': 1}
```

### Return contract

`optimize()` returns a **3-tuple**: `(lineups, pool, missing)`.

- `lineups` — list of dicts, keys: `players` (9-row frame), `salary`, `proj`, `ceiling`,
  `floor`, `objective`, `conflicts`, `teams`
- `pool` — the filtered playable pool actually solved over
- `missing` — `{"locks": [...], "excludes": [...], "pins": [...]}`, names that matched
  nothing. **Check this**: a misspelled lock is silently not locked otherwise.

### Parameters

| parameter | default | notes |
|---|---|---|
| `n_lineups` | 1 | |
| `objective` | `"ceiling"` | see table above |
| `locks` / `excludes` | none | names, case- and accent-insensitive |
| `pins` | none | `{name: slot}` — forces a roster seat; implies a lock |
| `stacks` | none | `{team: count}` — **counts QB/WR/TE only** |
| `max_overlap` | `5` | players shared with any earlier lineup |
| `randomness` | `0.20` | per-lineup objective jitter |
| `min_proj` | none | pool threshold; never drops a lock |
| `max_players_per_team` | 9 | |
| `min_games` | 2 | DK's two-game rule |
| `max_ownership` | none | needs an `Own%` column |
| `dst_conflict_max` | 2 | own skill players rostered against your own DST |
| `seed` | none | reproducibility |

For the **single best lineup** with no exploration:
`max_overlap=opt.ROSTER_SIZE - 1, randomness=0`.

### Two things about stacks that will surprise you

**`stacks={"BUF": 3}` does not mean "QB plus two."** It is a count over pass-game players
(QB/WR/TE), and the quarterback is not required to be one of them — a legal answer is three
Buffalo receivers with a Seattle QB. Constrain the QB with `locks` or `pins` if you want him.

**`L["teams"]` counts everyone, the constraint counts only QB/WR/TE.** A lineup satisfying
`{"BUF": 3}` can show `teams: {"BUF": 4}` because a Buffalo RB or DST is in there too. The
two numbers are answering different questions and both are correct.

Running backs never count toward their own team's stack, deliberately: a rushing touchdown
is a drive that did *not* end in a passing touchdown, so a back is at best uncorrelated with
his own QB and on the goal line is actively negative.

---

## 4. DK upload

```powershell
& $PY -m nfl.upload --date 2026-09-14 --slate main --list      # see what was built
& $PY -m nfl.upload --date 2026-09-14 --slate main             # fill every entry
& $PY -m nfl.upload --date 2026-09-14 --slate main --lineups 1,3,5-8
```

Reads the newest `lineups_<slate>.csv`, fills a DK **entries** export, and writes
`nfl_boards\<date>\upload_<slate>.csv`. **The template you downloaded is never modified** —
the filled copy is a new file, so a bad run cannot cost you the original.

Put the entries export in `nfl\nfl_dfs\dk_lineups\` — separate from MLB's `dk_lineups\`, so a
baseball template and a football one can never be picked up for each other — or pass
`--template`.

| flag | notes |
|---|---|
| `--list` | print the lineups with salary/proj/ceiling and their stacks, then stop |
| `--lineups` | `1,3,5-8`, or all (default) |
| `--template` / `--template-dir` | the DK entries file; newest in the dir by default |
| `--contest` | entries templates only: fill entries whose contest name contains this |
| `--source` | a specific lineups CSV instead of the newest |
| `--overwrite` | replace in place instead of writing `.rN` |

### Two things it refuses to do

**A template from another week.** Names resolve against whatever template you hand over, so a
stale file happily produces a full sheet of valid-looking ids for a contest that already
finished. The date is checked against the template's own embedded player list, and a mismatch
is an error rather than a warning:

```text
error: the template is for 2026-08-20 but these lineups are for 2026-09-14
```

**A player it cannot resolve to a DK id.** A cell holding only a name is silently not an entry
as far as DK is concerned, so an unresolved player fails the run rather than producing a
lineup that looks filled and enters nothing. Ids come from the template's own player list
first, because an id is only valid inside the draftgroup it came from.

---

## 5. PFF data layer

Added 2026-08-25. Full write-up in `docs/nfl/pff_integration.md`.

### Which file is which season

**Never read a season off a PFF filename.** The numbering is download order and differs per
family — one family puts the newest season in the unnumbered file, another the oldest.

```powershell
# The whole catalog
& $PY -c "from nfl import pffdata; print(pffdata.catalog()[['family','season','confidence','rows','usable','path']].to_string())"

# Only what was rejected, and why
& $PY -c "from nfl import pffdata; c = pffdata.catalog(); print(c[~c.usable][['path','note']].to_string())"
```

`catalog()` is **the thing to read when a join comes back empty** — it states what the loader
believes, which is the first thing to check against what you assumed. Cached on size+mtime;
`catalog(force=True)` re-fingerprints.

```python
from nfl.pffdata import load, resolve, latest
load("receiving_summary", 2025)   # tidy, keyed, canonical teams and positions
resolve("slot_coverage", 2024)    # just the path
latest("projections")             # newest-wins, for exports with no season
```

Families on disk: `receiving_summary`, `rushing_summary` (2018-25); `receiving_scheme`,
`defense_coverage_scheme`, `slot_coverage` (2020-25); `fantasy-stats-receiving`,
`fantasy-stats-passing` (2022-25); `receiving_depth`, `receiving_concept`, `passing_depth`,
`offense_blocking` (2023-25).

### Deployment profiles

```python
from nfl.usage import (receiver_profile, rusher_profile,
                       high_value_touches, team_target_distribution)

receiver_profile(2025)                    # one season
receiver_profile([2025, 2024, 2023])      # recency-weighted blend (1.0 / 0.45 / 0.20)
high_value_touches(2025)                  # RZ + inside-5 work, PER GAME
team_target_distribution(2025)            # target/route share + the team's WR/TE/RB mix
```

All keyed on `gsis_id` except `high_value_touches`, which keys on
`nfl.redzone.name_key` — that export carries no player id at all.

### O-line

```python
from nfl.oline import projected_line, team_line_strength
line = projected_line(2026)          # five projected starters per club, with a basis
team_line_strength(line)             # the unit, ranked
```

`basis` ladder: `grade` (staying, regressed at 0.60) → `grade (moved)` (regressed at 0.33)
→ `draft capital` (`−5.02·log(pick) + 76.6`) → `replacement`.

---

## 6. Studies

Every fitted constant in the NFL package is re-derivable:

```powershell
& $PY -m nfl.studies                  # list them
& $PY -m nfl.studies persistence      # which signals predict themselves
& $PY -m nfl.studies man_zone         # the man/zone effect size
& $PY -m nfl.studies oline            # grade portability + the draft curve
& $PY -m nfl.studies stacks           # QB-receiver correlation
& $PY -m nfl.studies all
```

Run `oline` before trusting `STAY_RETENTION`, `MOVE_RETENTION`, `DRAFT_SLOPE` or
`DRAFT_INTERCEPT` on a new season — it reprints exactly those four.

---

## 7. Hosting

The dashboard runs on Render off **committed data** — it never re-runs the pipeline or calls
an API on page load. The MLB half commits its payloads directly; the NFL half commits a
**precomputed snapshot**, because the raw PFF drop is 14 MB of exports the pages barely read,
and three pages would otherwise call nflverse on every cold start of an ephemeral container.

### The refresh loop

```powershell
& $PY -m nfl.snapshot --build          # ~1.5 MB of parquet into nfl/snapshot/
git add nfl/snapshot nfl/nfl_dfs/nfl_daily_files nfl/Lineups
git commit -m "refresh NFL snapshot" ; git push
```

That is the whole deploy. `& $PY -m nfl.snapshot` with no flags shows what is in it and how
stale.

### What is committed, and what is not

| committed | why |
|---|---|
| `nfl/snapshot/` | every frame the pages read, precomputed |
| `nfl/nfl_dfs/nfl_daily_files/` | DK salary exports — the weekly input |
| `nfl/Lineups/` | filled DK entries exports |

| **not** committed | why |
|---|---|
| `nfl/pff/` | 14 MB raw; the snapshot is derived from it at a tenth the size |
| `.cache/nflverse/` | 14 MB; the league frames are in the snapshot |
| `nfl_boards/` | your local lineup output |

### Two things that will bite if you change them

**Readers try the snapshot first, then fall back to computing live.** So a missing snapshot
is slower locally, never broken — but it also means a *stale* snapshot silently wins over
fresh raw data. Rebuild after dropping in new PFF exports.

**Do not add `scipy` to `requirements-web.txt`.** `nfl/__init__.py` re-exports the solver
lazily (PEP 562) precisely so the hosted app never imports it — ~40 MB of wheel for pages
that never solve a lineup. If you make a page need the solver, that changes.

### Verifying before you push

The condition that matters is the container's, not yours. Move the raw data aside, block the
network, and run every page:

```powershell
Move-Item nfl\pff .tmp\pff_hidden ; Move-Item .cache\nflverse .tmp\nflverse_hidden
& $PY -m pytest tests/test_nfl_snapshot.py -q
Move-Item .tmp\pff_hidden nfl\pff ; Move-Item .tmp\nflverse_hidden .cache\nflverse
```

The first time this was run, **9 of 13 pages failed** — all on `pffdata.catalog()` quietly
reaching for nflverse rosters to answer "which seasons do I have files for".

---

## What does not exist yet

Do not go looking for these; they are MLB-only today.

| missing | notes |
|---|---|
| late swap | `dfs.lateswap` is MLB-only |
| exposure *pacing* | the per-lineup pacing loop; the exposure **report** now exists |
| calibrated ownership | `nfl.ownership` exists but `OWNERSHIP_VERSION = 0` — never fitted |
| portfolio / nightly review | `nfl.results` reads contest exports; nothing selects an entry set yet |
| showdown | a second ROSTER dict with a 1.5× captain multiplier, not a new optimizer |

---

## Traps

| trap | what happens |
|---|---|
| **PFF has no `RB`** | backs are `HB`. A `pos == "RB"` filter empties a whole position, silently. Everything routes through `nfl.salaries.canon_position`. |
| **PFF team codes** | `HST` Houston, plus `ARZ`/`BLT`/`CLV`/`LA`/`SD`. Unfolded, those clubs vanish from a join with no error. Everything routes through `canon_team`. |
| **Canon is PFR-style** | `KC` → `KAN`, `SF` → `SFO`, `GB` → `GNB`, `NO` → `NOR`, `TB` → `TAM`. Filtering a board for `"KC"` returns nothing. |
| **`frame.get(col)`** | returns a **scalar NaN** for a missing column — no index, no `.fillna`, immediate crash. Use `frame.reindex(columns=[col])[col]`. Bitten five times. |
| **SOS: higher = easier** | verified at +0.506 against 2025 WR PPR allowed. Backwards inverts every matchup adjustment. |
| **`spread_line` is positive when HOME is favoured** | reading it backwards inverts every implied total. |
| **Defensive allowed-rates are noise** | slot YPT +0.10, man YPT +0.05, zone YPT +0.01 year over year. The Coverage-fit figure is display and tie-break only — do not build a projection adjustment on it. |
| **High-value work is a per-game volume** | inside-5 carries/game r=0.73; as a *share of carries* r=0.33. The share tracks its denominator. |
| **Overlap and randomness are not fitted** | `DEFAULT_MAX_OVERLAP=5`, `DEFAULT_RANDOMNESS=0.20` are MLB numbers off a 10-man roster, carried over as placeholders. |
| **Yardage bonuses take probabilities, not yards** | they are step functions; `offense_points` will not infer them from a mean. Same for DST points allowed, which takes a *distribution*. |
| **Known projection bias** | passing yards run 1.060× and receiving 1.071× on held-out 2025. Do **not** shrink `YARDAGE_CV` to hide it — that fits dispersion to absorb a mean error. |
