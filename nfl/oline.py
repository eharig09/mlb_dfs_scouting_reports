"""Projecting an offensive line from grades, movement and draft capital.

A line is projected **bottom-up, from its five individuals**, and that is a measured choice
rather than a stylistic one. Year over year:

    an individual lineman's grade      r = +0.60   (pass block +0.60, run block +0.53)
    a team's unit grade                r = +0.46 and +0.18 over the two pairs on file

The player is the stable unit; the team aggregate is not, because the personnel underneath
it churns. So nothing here carries last season's *unit* grade forward -- it assembles this
season's five men and grades them.

Three of the four inputs survived testing. The fourth did not:

**Past grade — yes, but regress it.** r = +0.60 means the right carry-forward is
`mean + 0.60 x (grade - mean)`, not the grade itself.

**Movement — a grade only half travels.** Linemen who stayed put correlate at +0.61/+0.60
across the two season pairs; linemen who changed clubs correlate at +0.27/+0.38. So moving
is not a note to display beside a carried-over grade, it is a reason to regress that grade
roughly twice as hard. *Fitted on 29 movers per season -- the direction repeats, the
magnitude does not deserve more than one decimal.*

**Draft capital — yes, and it predicts snaps better than grade.** Rookie linemen 2023-25
(n = 68): Spearman(pick, grade) = -0.405, `grade ~ -5.02 x log(pick) + 76.6`. Round 1
averages 62.9 and 930 snaps, rounds 4-7 average 50.1 and 499. Fitted on `log(pick)` because
the raw round table inverts (round 2 grades below round 3), the same small-n artefact
`nfl/coldstart.py` documents for skill positions.

**Returning starters — no.** Continuity correlates with next season's unit grade (+0.40,
+0.17), but that is confounding, not signal: good lines keep their starters. Against the
*change* in unit grade it is -0.16 and +0.01 -- nothing, with the sign flipping. Lines with
two or fewer returning starters actually improved more than lines with four or more, in both
pairs, which is mean reversion doing the work. **Continuity is reported here as description
and is deliberately not an adjustment**; the grades already carry what it knows.

    from nfl.oline import projected_line
    projected_line(2026)          # five projected starters per club, with a basis

Like the rest of the package, a number this cannot source stays blank rather than becoming
a zero, and `basis` records which rung of the ladder answered.
"""

import numpy as np
import pandas as pd

from nfl import data as nfl_data
from nfl import pffdata
from nfl.salaries import canon_team, normalize_name

# PFF's per-alignment snap counts, which is how a lineman's spot is known rather than
# inferred. `ce` is PFF's spelling for centre.
SPOT_COLUMNS = {
    "snap_counts_lt": "LT", "snap_counts_lg": "LG", "snap_counts_ce": "C",
    "snap_counts_rg": "RG", "snap_counts_rt": "RT",
}
SPOTS = ("LT", "LG", "C", "RG", "RT")

# How much of a graded lineman's distance from the positional mean carries into next season.
# These are the measured year-over-year correlations, which is exactly the regression
# coefficient toward the mean -- not a tuning knob.
STAY_RETENTION = 0.60
MOVE_RETENTION = 0.33

# `grade ~ SLOPE * log(pick) + INTERCEPT`, fitted on 68 rookie lineman seasons 2023-25.
DRAFT_SLOPE = -5.02
DRAFT_INTERCEPT = 76.6
# Undrafted rookies are given the curve's value at a notional pick past the last one, rather
# than left blank: going undrafted is information, and it is not neutral.
UNDRAFTED_PICK = 300

# Minimum snaps at a spot before a season counts toward a lineman's grade. Below this the
# grade is describing a handful of snaps, and PFF's own grades are noisy there.
MIN_SNAPS = 200
# Recency weights for blending a lineman's graded seasons, newest first. Not fitted.
RECENCY_WEIGHTS = (1.0, 0.5, 0.25)


def _num(frame, column):
    """Numeric Series for `column`, safe when the column is absent (`.get` returns a scalar
    NaN with no index -- the recurring crash in this codebase)."""
    return pd.to_numeric(frame.reindex(columns=[column])[column], errors="coerce")


def lineman_season(season, min_snaps=MIN_SNAPS):
    """Every graded lineman in one season, with the spot he actually played.

    The spot is his **max-snap alignment**, taken from PFF's per-alignment counts rather
    than from a position label -- rosters call all five of them `OL`.
    """
    frame = pffdata.load("offense_blocking", season)
    spots = pd.DataFrame({label: _num(frame, column).fillna(0)
                          for column, label in SPOT_COLUMNS.items()})
    out = pd.DataFrame({
        "name_key": frame["name_key"],
        "player": frame["player"],
        "gsis_id": frame["gsis_id"],
        "team": frame["team"],
        "position": frame["position"],
        "snaps": _num(frame, "snap_counts_offense"),
        "grade": _num(frame, "grades_offense"),
        "pass_block": _num(frame, "grades_pass_block"),
        "run_block": _num(frame, "grades_run_block"),
        "pressures": _num(frame, "pressures_allowed"),
        "spot": spots.idxmax(axis=1),
        "spot_snaps": spots.max(axis=1),
    })
    out = out[out["position"].isin(("T", "G", "C")) & (out["snaps"] >= min_snaps)]
    out["season"] = int(season)
    return out.sort_values("snaps", ascending=False).drop_duplicates("name_key")


def lineman_grades(seasons, min_snaps=MIN_SNAPS, weights=RECENCY_WEIGHTS):
    """One row per lineman: his recency-blended grades and the last club he was graded on.

    `last_team` is what makes movement detectable later -- comparing it against where he is
    rostered *now* is the whole test.
    """
    seasons = sorted((int(s) for s in seasons), reverse=True)
    frames = [lineman_season(s, min_snaps) for s in seasons]

    blocks = []
    for frame, weight in zip(frames, list(weights) + [0.0] * len(frames)):
        if weight > 0:
            blocks.append(frame.assign(_w=weight))
    stacked = pd.concat(blocks, ignore_index=True)

    graded = ["grade", "pass_block", "run_block"]
    out = {}
    for column in graded:
        present = stacked[column].notna()
        weight = stacked["_w"].where(present, 0.0)
        numerator = (stacked[column].fillna(0) * weight).groupby(stacked["name_key"]).sum()
        denominator = weight.groupby(stacked["name_key"]).sum()
        out[column] = numerator / denominator.replace(0, np.nan)

    newest = stacked.sort_values("season", ascending=False).drop_duplicates("name_key")
    newest = newest.set_index("name_key")
    result = pd.DataFrame(out)
    result["player"] = newest["player"]
    result["gsis_id"] = newest["gsis_id"]
    result["last_team"] = newest["team"]
    result["last_season"] = newest["season"]
    result["spot"] = newest["spot"]
    result["snaps"] = newest["snaps"]
    return result


def _draft_grade(pick):
    """The grade a rookie's draft slot implies. Undrafted is a slot, not a blank."""
    pick = float(pick) if pd.notna(pick) else UNDRAFTED_PICK
    pick = max(pick, 1.0)
    return DRAFT_SLOPE * np.log(pick) + DRAFT_INTERCEPT


def depth_chart_line(season):
    """The five projected starters per club, from the newest depth-chart snapshot.

    The 2026 depth charts are a **time series** of snapshots from March onward, so the newest
    `dt` is taken -- mixing camp positions with current ones would invent lines nobody fields.
    Rosters are no help here: they label all five spots `OL`.
    """
    charts = nfl_data.load_depth_charts([int(season)])
    if "dt" in charts.columns:
        charts = charts[charts["dt"] == charts["dt"].max()]
    charts = charts[charts.reindex(columns=["pos_abb"])["pos_abb"].isin(SPOTS)]
    rank = pd.to_numeric(charts.reindex(columns=["pos_rank"])["pos_rank"], errors="coerce")
    charts = charts[rank == 1]
    return pd.DataFrame({
        "team": charts["team"].map(canon_team),
        "spot": charts["pos_abb"],
        "gsis_id": charts.reindex(columns=["gsis_id"])["gsis_id"],
        "player": charts.reindex(columns=["player_name"])["player_name"],
        "name_key": charts.reindex(columns=["player_name"])["player_name"].map(normalize_name),
    }).drop_duplicates(["team", "spot"])


def projected_line(season, history=None, min_snaps=MIN_SNAPS):
    """Five projected starters per club, each with a projected grade and a `basis`.

    The ladder, in order, with `basis` naming which rung answered:

        `grade`          graded, and staying put -- regressed at STAY_RETENTION
        `grade (moved)`  graded, on a new club  -- regressed at MOVE_RETENTION
        `draft capital`  a rookie, from his pick
        `replacement`    neither -- a post-rookie lineman with no graded snaps, which is
                         evidence against him rather than an absence of evidence

    A `basis` column is first class here for the same reason it is on the skill board:
    "PFF graded him at 78" and "we inferred it from a third-round pick" must never look
    alike on a page.
    """
    season = int(season)
    history = list(history) if history else [season - 1, season - 2, season - 3]
    grades = lineman_grades(history, min_snaps=min_snaps)
    line = depth_chart_line(season)

    # Grades key on name: nflverse fills `pff_id` for only about two thirds of linemen, so
    # the id crosswalk cannot be the primary path on this side of the ball.
    line = line.merge(grades.reset_index(), on="name_key", how="left",
                      suffixes=("", "_pff"))

    positional_mean = grades["grade"].mean()
    pass_mean = grades["pass_block"].mean()
    run_mean = grades["run_block"].mean()

    rookies = _rookie_picks(season)
    line["draft_number"] = line["gsis_id"].map(rookies)
    moved = line["last_team"].notna() & (line["last_team"] != line["team"])
    has_grade = line["grade"].notna()

    retention = np.where(moved, MOVE_RETENTION, STAY_RETENTION)
    line["proj_grade"] = np.where(
        has_grade, positional_mean + retention * (line["grade"] - positional_mean), np.nan)
    line["proj_pass_block"] = np.where(
        has_grade, pass_mean + retention * (line["pass_block"] - pass_mean), np.nan)
    line["proj_run_block"] = np.where(
        has_grade, run_mean + retention * (line["run_block"] - run_mean), np.nan)

    line["basis"] = np.where(has_grade & moved, "grade (moved)",
                             np.where(has_grade, "grade", "replacement"))

    is_rookie = ~has_grade & line["gsis_id"].isin(rookies.index)
    if is_rookie.any():
        line.loc[is_rookie, "proj_grade"] = line.loc[is_rookie, "draft_number"].map(_draft_grade)
        line.loc[is_rookie, "basis"] = "draft capital"

    replacement = line["basis"] == "replacement"
    line.loc[replacement, "proj_grade"] = _draft_grade(UNDRAFTED_PICK)

    line["moved"] = moved
    return line.sort_values(["team", "spot"]).reset_index(drop=True)


def _rookie_picks(season):
    """gsis_id -> draft pick, for this season's rookies only.

    **The `years_exp == 0` gate is load-bearing.** `draft_number` stays on a roster row
    forever, so without it an eleven-year veteran drafted 29th reads as a first-round rookie
    -- the trap `nfl/coldstart.py` documents on the skill side, and the line is no different.
    """
    rosters = nfl_data.load_rosters([int(season)])
    experience = pd.to_numeric(rosters.reindex(columns=["years_exp"])["years_exp"],
                               errors="coerce")
    rookies = rosters[experience == 0]
    picks = pd.to_numeric(rookies.reindex(columns=["draft_number"])["draft_number"],
                          errors="coerce")
    out = pd.Series(picks.values, index=rookies.reindex(columns=["gsis_id"])["gsis_id"].values)
    return out[~out.index.duplicated()]


def team_line_strength(line):
    """One row per club: the projected unit, and how it was arrived at.

    `continuity` counts starters returning to the same club. It is **reported, not applied**
    -- measured against the change in unit grade it is worth -0.16 and +0.01, and the grades
    it would adjust already contain what it knows. See the module docstring.
    """
    grouped = line.groupby("team")
    out = pd.DataFrame({
        "proj_grade": grouped["proj_grade"].mean(),
        "proj_pass_block": grouped["proj_pass_block"].mean(),
        "proj_run_block": grouped["proj_run_block"].mean(),
        "starters": grouped.size(),
        "continuity": grouped.apply(lambda g: int((g["basis"] == "grade").sum()),
                                    include_groups=False),
        "newcomers": grouped.apply(lambda g: int((g["basis"] == "grade (moved)").sum()),
                                   include_groups=False),
        "rookies": grouped.apply(lambda g: int((g["basis"] == "draft capital").sum()),
                                 include_groups=False),
        "unknown": grouped.apply(lambda g: int((g["basis"] == "replacement").sum()),
                                 include_groups=False),
    })
    return out.sort_values("proj_grade", ascending=False)
