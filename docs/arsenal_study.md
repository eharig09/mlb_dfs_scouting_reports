# Does arsenal fit predict fantasy points?

Two questions, asked of three seasons:

1. When a hitter's numbers against a starter's pitch shapes are good, does he score more
   DK points?
2. When a whole lineup fits an arsenal badly, does the starter score more DK points?

The answers are **no** and **yes**, and the gap between them is the useful part of this
document. The same measurement is worthless per hitter and worth about two DK points per
start when averaged over a lineup, because averaging nine noisy numbers is what makes the
signal readable.

## 1. The dataset

Built from what was already on disk plus one API pull:

| Source | What | Size |
| --- | --- | --- |
| `.cache/statcast/*.pkl` | Pitch-level statcast, 2024 / 2025 / 2026-to-Aug-5 | 1,916,047 pitches, 6,579 games |
| `statsapi` boxscores | Actual DK lines for every player in every game | 194,452 player-games |

From those: **118,422 hitter-vs-starting-pitcher pairs** and **13,158 starts**. Starters
and the lineups they faced are derived from statcast itself rather than from lineup
scraping, so the matchup is exactly the one that happened.

### No-leakage by construction

Every feature is read with `merge_asof(..., allow_exact_matches=False)` against
cumulative-by-date tables, so a row for game date D can only see data from strictly before
D. A rolling window is the difference of two such reads. There is no code path in which
tonight's game informs tonight's feature, which matters because this is exactly the kind
of claim that looks true when it leaks.

### Controls, which are the whole ballgame

Arsenal fit is badly confounded with "is this a good hitter" — a good hitter hits
everything, so a raw correlation between any arsenal score and fantasy points is
guaranteed positive and means nothing. Every number below is the *incremental* effect
after:

- the hitter's own form (DK/game, PA/game, K rate, HR rate, all as-of)
- **his own line against that same hand, over the same window** — the single most
  important control, because without it an arsenal score is just a noisier restatement of
  his platoon split
- lineup slot, park (home-team fixed effects), season, platoon side
- the opposing starter's form (DK/start, K rate, ERA, WHIP, IP/start)
- how much history the fit and the baseline each rest on

Standard errors are clustered on `game_pk` throughout. Nine hitters in a lineup share a
game, a park, a starter and a weather reading; treating them as 118,422 independent
observations would shrink every standard error by roughly √9 and turn noise into
significance.

## 2. Hitter side: nothing usable

Arsenal fit → that hitter's DK points, with the full control set:

| Model | n | coef (DK pts per SD) | t | ΔR² |
| --- | --- | --- | --- | --- |
| Full controls | 100,372 | +0.035 | 1.48 | +0.00002 |
| + batter fixed effects | 100,372 | +0.029 | 1.14 | +0.00001 |
| Only fits with ≥40 PA of history | 34,000 | +0.073 | 1.81 | +0.00010 |

Deciles of arsenal fit, with controls partialled out, are flat — 6.80, 6.82, 6.92, 7.03,
6.87, 6.94, 7.03, 7.05, 6.88, 6.89 DK points. There is no slope to find. Walk-forward
(fit 2024-25, score 2026) changes RMSE by **0.0000** and rank correlation by **−0.0000**.

This independently confirms the `FACTOR_DAMPING["arsenal"] = 0.0` decision already in
`dfs/projections.py`, on 100,372 player-games instead of the 823 that motivated it.

**The matchup effect is real, it is just tiny.** Against the cleaner target of wOBA in
plate appearances against the starter only, the same feature is significant (t = 2.60,
batter fixed effects included) and worth about **+0.003 wOBA per standard deviation**. A
hitter gets four plate appearances, and his DK score is dominated by whether teammates
were on base. The matchup signal is real and it is far below that noise floor.

## 3. Starter side: real, and worth using

The lineup-level aggregate — the mean of the nine hitters' fits — predicts the starter's
DK points, with the full control set including the lineup's own overall quality:

| Model | coef (DK pts per SD) | t | ΔR² |
| --- | --- | --- | --- |
| Baseline controls | −0.86 | −8.09 | +0.00521 |
| + opposing-team fixed effects | −0.89 | −8.21 | +0.00534 |
| **+ pitcher fixed effects** | **−0.55** | **−3.29** | +0.00084 |
| + pitcher and opposing-team FE | −0.55 | −3.17 | +0.00079 |
| Real starts only (IP > 2) | −0.80 | −7.54 | +0.00476 |

The pitcher fixed-effect row is the one that matters. It removes pitcher identity
entirely, so what is left is *within* a pitcher: across his own starts, he does better on
the nights the lineup fits him worse. That rules out "pitchers with unusual arsenals draw
low fit scores and are also good", which is the obvious way this result could have been an
artifact.

Walk-forward, fit on 2024-25 and scored on 2026:

```
RMSE     10.2316 -> 10.1748   (-0.0568)
rank r    0.2750 ->  0.2905   (+0.0155)
```

DK points by decile of the lineup's K edge, controls partialled out:

```
d1  12.92   d2  12.77   d3  13.49   d4  13.40   d5  14.18
d6  13.93   d7  14.36   d8  13.71   d9  14.66   d10 15.85
```

Roughly a **three-point spread** between the lineups that fit an arsenal best and worst.

### Where it comes from

| Component | coef per SD | t | ΔR² |
| --- | --- | --- | --- |
| K | −0.246 | −10.64 | +0.00824 |
| IP | −0.073 | −5.92 | +0.00251 |
| H | +0.067 | +3.10 | +0.00076 |
| ER | +0.057 | +2.85 | +0.00067 |
| BB | −0.014 | −1.09 | +0.00010 |

Strikeouts first, innings second. That is why the shipped version adjusts the K rate and
the innings projection and leaves everything else alone.

### The mean is the right aggregate

Fancier summaries were tested and are worse. The lineup's *best*-fitting hitter carries
much less information than the lineup as a whole, which is the opposite of the intuition
that one dangerous bat decides the night:

| Aggregate | t | ΔR² |
| --- | --- | --- |
| Slot-PA-weighted mean | −8.09 | +0.00521 |
| Unweighted mean | −8.09 | +0.00520 |
| Median | −7.77 | +0.00480 |
| Share of lineup fitting well | −7.76 | +0.00478 |
| Worst-fitting hitter (min) | −7.45 | +0.00439 |
| Mean of the top three fits | −5.89 | +0.00277 |
| Best-fitting hitter (max) | −4.77 | +0.00182 |

Slot weighting makes no measurable difference; it is kept because it costs nothing and is
the defensible choice. A secondary finding worth noting: the *spread* of fits within a
lineup is independently positive (t = +3.58), so a lineup where a few hitters match up
well and the rest do not is better for the pitcher than a uniformly average one.

## 4. Tuning: what actually matters

37 configurations, each scored on the starter-side effect. **Selection used 2024-25 only;
2026 was never looked at until the winner was fixed.** Coordinate sweep from a base of the
report's shipped settings, one axis at a time.

### Metric — the biggest single change

| Metric | t | ΔR² |
| --- | --- | --- |
| **K rate** | **−4.88** | +0.00255 |
| Whiff% | −4.10 | +0.00174 |
| wOBA | −4.01 | +0.00170 |
| OPS *(shipped)* | −3.70 | +0.00145 |
| xwOBA | −3.44 | +0.00129 |
| RV/100 | −2.50 | +0.00067 |

Strikeout rate against the arsenal beats everything, which follows from the decomposition
above: the starter's DK line is mostly strikeouts. **RV/100 is the weakest of the six.** It
is an appealing statistic — it prices every pitch outcome on one scale — but that
generality is the problem here: it averages away the one outcome that dominates pitcher
scoring, and it needs far more pitches to stabilise.

### Shrinkage — the second-biggest, and currently absent

| Prior weight (PA) | t | ΔR² |
| --- | --- | --- |
| 0 *(shipped)* | −3.75 | +0.00157 |
| **25** | **−5.35** | +0.00311 |
| 75 | −5.07 | +0.00279 |
| 150 | −4.86 | +0.00256 |
| 400 | −4.65 | +0.00233 |

A weighted arsenal sample is about **18 plate appearances** at the median. Reporting that
raw, as the report does today, is mostly reporting noise. Pulling it 25 PA toward the
hitter's own rate against that hand is worth more than every other knob except the metric.

### Shape matching — the shipped tolerance is right, and it is load-bearing

| Velocity tolerance | t | median PA of history |
| --- | --- | --- |
| ±1.0 mph | −5.03 | 11 |
| **±2.5 mph** *(shipped)* | **−5.35** | 17 |
| ±4.0 mph | −4.06 | 26 |
| ±6.0 mph | −2.89 | 31 |
| none (pitch type only) | −1.81 | 33 |

Dropping the shape filter and keeping only the pitch-type label destroys the effect
(t = −1.81). This is the "too-loose bucket" failure in numbers: widen the bucket enough and
the split converges on the hitter's overall line, which the baseline control already
absorbs, leaving nothing.

### Everything else barely moves

| Axis | Best | Note |
| --- | --- | --- |
| Pitches counted | top 3 | 2 is worse (−4.96); 4, 5, 6, 8 all ≈ −5.1 |
| Usage weight power | 1.5 | tied with 1.0 (−5.35 both); within noise |
| Batter lookback | 730 days | 365 clearly worse (−4.66); 1095 and all-history identical |
| Arsenal window | 365 days | 45-day is close (−5.23); 90-day worst (−4.97) |
| Same-handed history only | yes | dropping it barely changes the pitcher side, but wrecks the hitter side |
| Minimum usage to count | 5% | no effect |

### Holdout confirmation

Both configurations scored once on 2026, which played no part in selection:

| Configuration | coef | t | ΔR² |
| --- | --- | --- | --- |
| As shipped (top 3, ±2.5 mph, OPS, unshrunk, raw level) | −0.65 | −3.08 | +0.00265 |
| Tuned (top 3, ±2.5 mph, **K rate**, **25 PA shrink**, **residualized**) | −1.29 | −6.00 | **+0.01099** |

**4.1× the explanatory power, out of sample.** The three changes that produced it are the
metric, the shrinkage, and residualizing against the hitter's own line rather than
reporting the level.

## 5. Ranking pitchers within a slate

Predicting DK points and choosing between tonight's arms are different bars. A feature can
lift R² by explaining variation that lives mostly *between* slates and never change which
pitcher gets rostered. So everything in this section is computed strictly inside a slate
(a date with ≥ 6 starts), with models fit walk-forward — 2024 → 2025 and 2024-25 → 2026 —
over **7,737 starts across 311 slates**.

The reference points: the baseline model ranks pairs correctly **59.18%** of the time, gets
a within-slate rank correlation of **+0.262**, and its top pick averages **20.69** DK
against a slate mean of 13.76 and a best-available of 34.13.

| Feature added | pairwise | slate ρ | top-1 | top-3 | top pick changed |
| --- | --- | --- | --- | --- | --- |
| *(baseline)* | 59.18% | +0.262 | 20.69 | 18.95 | — |
| **Lineup arsenal K edge** | **59.47%** | **+0.271** | **21.00** | 18.80 | 22.2% of slates |
| vs type, K rate | 59.09% | +0.261 | 20.48 | 18.86 | 8.7% |
| vs type, wOBA | 59.18% | +0.262 | 20.62 | 18.84 | 3.5% |
| Both vs-type metrics | 59.10% | +0.262 | 20.52 | 18.87 | 9.3% |
| Everything together | 59.35% | +0.268 | 20.95 | 18.79 | 25.1% |

Bootstrapping over slate resamples (slates are the independent unit — every starter appears
in every pair on his own slate):

| Feature | pairwise gain | 95% interval |
| --- | --- | --- |
| **Arsenal K edge** | **+0.29 pts** | **[+0.04, +0.56]** |
| vs type, K rate | −0.09 | [−0.20, +0.01] |
| vs type, wOBA | −0.00 | [−0.07, +0.07] |

And each feature on its own, ranking what the baseline could *not* explain:

| Feature | within-slate ρ vs residual | pairwise |
| --- | --- | --- |
| **Arsenal K edge** | **+0.0913** | **53.19%** of 97,242 pairs |
| vs-type K edge | +0.0108 | 50.38% |
| vs-type wOBA edge | −0.0116 | 49.59% |

**The arsenal K edge reorders a slate; "vs type" does not.** Given two starters the model
otherwise rates equally, the one whose opponent whiffs more against his shapes outscores
the other **53.2%** of the time. That is a small edge, and it is the right size to expect:
it moves the top pick on about a fifth of slates and gains **+1.41 DK** when it does, being
right 54% of the time. It is worth having and it is not worth overriding a projection for.

### Why "vs type" fails, and what it is not

This is a reconstruction of `generate_team_pitcher_type_results`, not the shipped code:
pitchers are vectors of usage, velocity and movement per pitch type, snapshotted monthly,
and comps are the 40 nearest same-handed arms. Median comp sample lands at **266 team PA**
against the report's `TYPE_TARGET_PA = 250`, so it is comparable in the way that matters.

The comp machinery is **not** broken, which is worth establishing before dismissing it. The
vs-type K edge does predict the starter's actual strikeout rate that game (t = +3.06) — it
is just about three times weaker than the arsenal edge (t = +9.17), and once the arsenal
edge is in the model it contributes nothing at all:

| Model | coef | t | ΔR² |
| --- | --- | --- | --- |
| K edge alone | +0.875 | +8.20 | +0.00535 |
| vs-type K edge alone | +0.173 | +1.77 | +0.00025 |
| vs-type wOBA edge alone | +0.031 | +0.32 | +0.00001 |
| K edge, given vs type | +0.893 | +8.05 | +0.00513 |
| **vs type, given K edge** | **−0.059** | **−0.58** | +0.00003 |

The wOBA version — the one closest to the panel's own `OPS Diff` — is the weakest of all, so
this is not an artifact of preferring strikeout rate. Comps buy plate appearances (266 vs
the arsenal slice's ~18) and spend them on the wrong thing: a comp pitcher shares a general
profile but not the specific shapes, and averaging forty of them smooths away the very
thing that turned out to carry the signal.

**This does not say the Type panel is wrong**, only that it should not be used to order
starting pitchers on a slate. It answers a different question — "has this lineup handled
arms like this before" — which is legitimate scouting context. It is simply not a DFS
ranking input, and nothing in the code treats it as one.

## 6. What changed in the code

### `scouting_report.py`

`generate_batter_arsenal_matchups` now also computes each hitter's line against that hand
across *all* pitch types — same source, same window, same hand, differing only in the shape
filter — and reports two new columns:

- `K% vs Hand` — the baseline
- `K Edge` — the shrunk residual, in percentage points. Positive is a pitcher's edge.

`ARSENAL_K_SHRINK_PA = 25.0` is the tuned prior weight. The existing OPS / xwOBA / Arsenal
Score columns are unchanged: the study says they do not predict hitter scoring, but they
remain readable scouting context and nothing depends on churning them.

`render_report_from_cache` rebuilds the arsenal tables for any cached game that predates
`K Edge`, using the same one-time-upgrade pattern as the pitcher-type splits. Without it a
cached game would silently run with the new factor switched off.

### `dfs/projections.py`

`_lineup_arsenal_k_edge` averages `K Edge` across the lineup, weighted by slot plate
appearances, and `project_pitcher` applies it:

```python
ARSENAL_K_PASSTHROUGH = 0.45     # d(K rate) / d(lineup K edge)
ARSENAL_IP_PASSTHROUGH = 2.77    # d(IP) / d(lineup K edge)
ARSENAL_IP_CLIP = 0.25           # innings this factor may move a projection
```

Those are pass-through fractions, not standardized effects: if a lineup strikes out two
points more often against these shapes than it does against this hand generally, about
**0.9 points of that reach the starter's actual K rate**. Both were fitted on 2024-25 and
re-estimated on 2026, which came back *higher* on both (0.68 and 6.54), so the shipped
values are the conservative end.

The hitter-side factor stays damped to zero. Same measurement, opposite verdict, and the
asymmetry is the finding rather than an inconsistency.

Worked example — Cristopher Sánchez vs Washington, 2026-08-06. The lineup strikes out 2.6
points *less* against his shapes than it does against left-handers generally, so his
projection falls from 22.92 to 22.04 DK points (K 6.96 → 6.58, IP 5.96 → 5.88) and the
Why column reads `WSH K -2.6 pts vs his shapes (9 hitters)`.

## 7. Caveats

- **The effect is small in absolute terms.** ΔR² of 0.011 on the holdout is a real edge in
  a game decided by ranking, not a large one. It moves a starter's projection by under a
  point in a typical matchup and about two in the tails.
- **The pass-through constants were fitted against actual outcomes, not against this
  projection's residuals.** The controls in the study resemble what `project_pitcher`
  already knows but are not identical to it, so the true pass-through given everything the
  model already prices could be somewhat lower. The conservative choice of the 2024-25
  values over the larger 2026 ones partly covers this.
- **Openers are tagged, not modelled.** 655 of 13,158 starts went two innings or fewer.
  Excluding them barely changes the result, but the lineup-fit number for an opener
  describes an arsenal the lineup sees twice.
- **2026 is a partial season** (through August 5), which is why it is the holdout rather
  than part of the training set.

## 8. Re-running it

The scripts are in `research/arsenal/`. They write their intermediates (a 54 MB pitch
extract and three seasons of box scores) to whatever directory is passed as the first
argument, which should be somewhere outside the repo. Run from the project root, with the
data directory as `$D`:

```
python research/arsenal/extract_pitches.py  $D     # ~4 min, one process per season
python research/arsenal/fetch_boxscores.py  $D     # ~5 min, 6,579 API calls
python research/arsenal/build_study.py      $D
python research/arsenal/analyze.py          $D
python research/arsenal/robust.py           $D
python research/arsenal/tune.py             $D tuning.csv
python research/arsenal/calibrate.py        $D research/arsenal/tuned.json
```

What each stage does:

1. `extract_pitches.py` — season pickles → compact parquet (one process per season; the
   full frames do not fit in memory together)
2. `fetch_boxscores.py` — StatsAPI boxscores → DK actuals
3. `build_study.py` — pairs and starts with as-of controls
4. `analyze.py` / `robust.py` — the two questions and the fixed-effect checks
5. `tune.py` — the coordinate sweep, selecting on 2024-25
6. `calibrate.py` — pass-through constants in natural units
