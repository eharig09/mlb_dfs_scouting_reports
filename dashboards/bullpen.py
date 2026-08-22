"""Reliever availability, measured rather than assumed.

The question this answers is "who can actually pitch tonight", and the honest way to answer
it is to look at what relievers *did* rather than at a rule of thumb.

Measured over the 2025 regular season from the Statcast chunks — 20,868 appearances, 281
relievers (20+ games, median outing ≤30 pitches), restricted to days their club actually
played the following day so "did not pitch" is separable from "no game":

    P(pitches the next day)
      after one or more days of rest      26.0%     n = 9,300
      after back-to-back days              5.5%     n = 2,032

    back-to-back, split by the two-day pitch count
      0-25 pitches                         9.7%     n =   638
      25-35                                4.8%     n =   859
      35-45                                1.4%     n =   428
      45+                                  1.9%     n =   107

So a back-to-back reliever is unavailable **94.5%** of the time, and one who threw 35 or more
pitches across the two days is unavailable **98.6%** of the time. Only 111 of 2,032
back-to-backs became a third straight day.

**This is stricter than the report's own rule**, which is why it lives here. The workbook's
`enhance_bullpen_analysis` grades on a three-day pitch total — Taxed at 45+, Monitor at 25+ —
and never looks at consecutive days as such. An arm that threw 15 yesterday and 12 today
totals 27 and grades "Monitor", while the measurement says he is 4.8% likely to appear. The
workbook is untouched; changing a live output mid-season is the user's call, not this
module's.

The tiers below are read straight off that table rather than invented:

    Out        back-to-back, 35+ pitches across the two days   ~1.4% appear
    Doubtful   back-to-back, under 35                          ~5-10% appear
    Monitor    not back-to-back, but heavy recent load
    Available  everything else                                 ~26% appear

The final `Status` is **the stricter of this grading and the report's own**, never this one
alone. The measurement catches what the report misses — consecutive days — but the report
catches what the measurement misses: a single long outing. An arm who threw 51 pitches two
days ago is not back-to-back, so this rule alone would grade him Monitor while the report
calls him Taxed, and a 51-pitch appearance is a multi-inning one that usually costs more than
a day. Combining them can only tighten a board, never loosen it.
"""

import re

import numpy as np
import pandas as pd

# Two-day pitch count at which a back-to-back arm goes from unlikely to effectively out.
HEAVY_TWO_DAY_PITCHES = 35
# Three-day total that marks a rested arm as worked, carried over from the report's own rule.
MONITOR_THREE_DAY_PITCHES = 25

TIERS = ["Out", "Doubtful", "Monitor", "Available"]

#: How hard each grade is, so two gradings can be combined by taking the stricter one.
SEVERITY = {"Available": 0, "Monitor": 1, "Doubtful": 2, "Out": 3}

#: The report's own `Avail` wording, mapped onto the same ladder. "Taxed" is the report's
#: heavy-workload grade and sits alongside Doubtful rather than at Out, because it is a
#: three-day pitch total rather than an observed next-day rate.
REPORT_TIER = {"Available": "Available", "Monitor": "Monitor", "Taxed": "Doubtful"}

#: Observed rate at which each tier's arms pitched the next day, for display.
TIER_APPEARANCE_RATE = {
    "Out": 0.014,
    "Doubtful": 0.062,
    "Monitor": 0.260,
    "Available": 0.260,
}

_DATE_COLUMN = re.compile(r"^\d{2}-\d{2}$")


def date_columns(frame):
    """The per-day pitch-count columns, newest first.

    The bullpen usage table carries one column per recent game date (`08-19`, `08-18`, …)
    in the order the report wrote them, which is already newest-first. Sorting them as
    strings would be wrong across a month boundary, so the file's order is trusted.
    """
    if frame is None or getattr(frame, "empty", True):
        return []
    return [c for c in frame.columns if _DATE_COLUMN.match(str(c))]


def _pitches(value):
    """A day's pitch count. `-` means he did not appear; blank means the same."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    text = str(value).strip()
    if not text or text in {"-", "—", "–"}:
        return 0
    match = re.search(r"\d+", text)
    return int(match.group()) if match else 0


def availability(usage, heavy=HEAVY_TWO_DAY_PITCHES, monitor=MONITOR_THREE_DAY_PITCHES):
    """Add `b2b`, `two_day`, `three_day` and a measured `Status` to a bullpen usage frame.

    The frame is the second element of the report's `{side}_bullpen_l5` payload: one row per
    arm, one column per recent date, plus `Avail` — the report's own grade, which is kept
    alongside rather than overwritten so the two can be compared.
    """
    if usage is None or usage.empty:
        return pd.DataFrame()
    days = date_columns(usage)
    out = usage.copy()
    if not days:
        out["b2b"] = False
        out["two_day"] = 0
        out["three_day"] = 0
        out["Status"] = out.get("Avail", "Available")
        return out

    counts = pd.DataFrame({d: out[d].map(_pitches) for d in days}, index=out.index)
    # Column order is newest-first, so the two most recent games are the first two columns.
    most_recent = counts[days[0]] if len(days) >= 1 else 0
    previous = counts[days[1]] if len(days) >= 2 else 0

    out["b2b"] = (most_recent > 0) & (previous > 0)
    out["two_day"] = most_recent + previous
    out["three_day"] = counts[days[:3]].sum(axis=1)

    measured = np.where(
        out["b2b"] & (out["two_day"] >= heavy), "Out",
        np.where(out["b2b"], "Doubtful",
                 np.where(out["three_day"] >= monitor, "Monitor", "Available")))
    out["measured"] = measured

    # **Take the stricter of the two grades, never just this one.** The back-to-back finding
    # is something the report's rule misses, but the report's rule catches something this one
    # misses: a single long outing. An arm who threw 51 pitches two days ago is not on
    # back-to-back days, so the measurement alone grades him Monitor while the report calls
    # him Taxed — and a 51-pitch appearance is a multi-inning one that usually costs more
    # than a day. Replacing the report's grade would have quietly loosened the board.
    out["Status"] = [max((m, r), key=lambda tier: SEVERITY.get(tier, 0))
                     for m, r in zip(measured, _report_tier(out))]
    out["appear_rate"] = [TIER_APPEARANCE_RATE.get(s, np.nan) for s in out["Status"]]
    return out


def _report_tier(frame):
    """The report's own availability grade, on this module's ladder."""
    if "Avail" not in frame.columns:
        return ["Available"] * len(frame)
    return [REPORT_TIER.get(str(v).strip(), "Available") for v in frame["Avail"]]


def summarise(graded):
    """Counts per tier, in tier order, for a metric row."""
    if graded is None or graded.empty or "Status" not in graded.columns:
        return {tier: 0 for tier in TIERS}
    counts = graded["Status"].value_counts().to_dict()
    return {tier: int(counts.get(tier, 0)) for tier in TIERS}


def disagreements(graded):
    """Arms this rule grades harder than the report's own `Avail` column.

    These are the rows worth looking at: the report calls them usable, the measurement says
    a back-to-back arm appears 5.5% of the time.
    """
    if graded is None or graded.empty or "Avail" not in graded.columns:
        return pd.DataFrame()
    stricter = graded["Status"].isin(["Out", "Doubtful"]) & graded["Avail"].eq("Available")
    return graded[stricter]
