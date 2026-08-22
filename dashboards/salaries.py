"""Join a night's composite scores to what DraftKings charged for them.

Why the residual is the number, not the composite
-------------------------------------------------
Salary and composite matchup score correlate at **+0.73** on a typical slate. The market has
already priced most of what the composite knows, so a chart of raw composite against salary
mostly re-tells you the price — the good matchups are expensive, which is not news.

What is news is the **surplus**: how far a hitter's matchup sits above or below what his own
price bracket normally delivers. A $2,300 bat with a 36.6 composite and a $5,200 bat with a
7.1 are the two ends of that, and neither is visible on the composite alone.

    median composite by price, 2026-08-20
      under $2,600   -11.2
      $2.6-3.2k        4.3
      $3.2-3.8k       13.2
      $3.8-4.4k       14.9
      $4.4k and up    29.2

The expectation curve is the **median within a price band**, not a fitted line. The
relationship is monotone but not straight — it flattens through the middle and steepens at
the top — and a straight fit would systematically overprice the middle of the board.

Slates
------
DraftKings publishes several slates a day (main, early, late, turbo) and a game appears on
more than one. Prices are the same across them — checked, zero discrepancies — so the loader
**unions every slate for the date** and keeps one row per player. Using a single slate
silently drops whoever is not on it: on 2026-08-20 the main slate covers 54 of 162 hitters
and the union covers all 162.
"""

import glob
import os
import re
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st

SALARY_DIR = "dfs_daily_files"
_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?")

#: Price bands for the expectation curve. Quantile bands would move under the chart every
#: time the slate changed shape, so these are fixed and the band medians are what move.
BANDS = [0, 2600, 3200, 3800, 4400, 5200, 100000]
BAND_LABELS = ["<2.6k", "2.6-3.2k", "3.2-3.8k", "3.8-4.4k", "4.4-5.2k", "5.2k+"]

#: A band needs this many hitters before its median is worth drawing a line through.
MIN_BAND = 4

#: Coarser tiers for the *shape* channel on a scatter. Six bands is right for an expectation
#: curve and far too many for shape — four is about what a reader can tell apart at the small
#: end of a size ramp. The cuts are `BANDS` edges, so the two schemes never disagree about
#: which side of a price a hitter is on.
TIER_EDGES = [0, 3200, 4400, 5200, 100000]
TIER_LABELS = ["<3.2k", "3.2-4.4k", "4.4-5.2k", "5.2k+"]

#: Its own category, never folded into the cheapest tier. A hitter the market never priced is
#: a different thing from a cheap one, and the same rule already governs `surplus`.
UNPRICED = "no price"

#: Shape per tier, ordered cheap to expensive. `UNPRICED` takes the open cross so it reads as
#: an absence rather than a rank.
TIER_SHAPES = {
    "<3.2k": "circle",
    "3.2-4.4k": "square",
    "4.4-5.2k": "triangle-up",
    "5.2k+": "diamond",
    UNPRICED: "cross",
}


def price_tier(frame, column="Salary"):
    """A `Price` column: which coarse price tier each hitter sits in, `no price` if none."""
    if frame is None or frame.empty:
        return frame
    out = frame.copy()
    if column not in out.columns:
        out["Price"] = UNPRICED
        return out
    salary = pd.to_numeric(out[column], errors="coerce")
    tier = pd.cut(salary, TIER_EDGES, labels=TIER_LABELS, right=False)
    out["Price"] = tier.astype(object).where(salary.notna(), UNPRICED).fillna(UNPRICED)
    return out


def name_key(value):
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", "", _SUFFIX.sub("", text.lower())).strip()


@st.cache_data(ttl="30m", max_entries=8, show_spinner=False)
def salaries_for_date(date, salary_dir=SALARY_DIR):
    """Every batter priced on any of a date's slates, one row each.

    Pitchers are dropped — `Roster Position` carries `P` for them — because this joins to
    the hitter composite. A player on several slates keeps one row; prices agree.
    """
    frames = []
    for path in glob.glob(os.path.join(salary_dir, f"DKSalaries_{date}_*.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if "Roster Position" not in frame.columns or "Salary" not in frame.columns:
            continue
        slot = frame["Roster Position"].astype(str)
        batters = frame[~slot.str.fullmatch(r"P|SP|RP")]
        if batters.empty:
            continue
        batters = batters.copy()
        batters["key"] = batters["Name"].map(name_key)
        batters["slate"] = os.path.basename(path).rsplit("_", 1)[-1][:-4]
        batters["Salary"] = pd.to_numeric(batters["Salary"], errors="coerce")
        frames.append(batters[["key", "Salary", "Roster Position", "AvgPointsPerGame",
                               "slate"]])
    if not frames:
        return pd.DataFrame(columns=["key", "Salary", "Roster Position",
                                     "AvgPointsPerGame", "slate"])
    union = pd.concat(frames, ignore_index=True).dropna(subset=["Salary"])
    return union.sort_values("Salary", ascending=False).drop_duplicates("key")


def available_dates(salary_dir=SALARY_DIR):
    found = set()
    for path in glob.glob(os.path.join(salary_dir, "DKSalaries_*.csv")):
        match = re.search(r"\d{4}-\d{2}-\d{2}", os.path.basename(path))
        if match:
            found.add(match.group())
    return sorted(found, reverse=True)


def attach_salary(hitters, date, salary_dir=SALARY_DIR):
    """Add `Salary`, `DK Pos` and the price-relative columns to a hitter frame."""
    if hitters is None or hitters.empty:
        return hitters
    prices = salaries_for_date(date, salary_dir)
    frame = hitters.copy()
    frame["key"] = frame["Name"].map(name_key)
    if prices.empty:
        frame["Salary"] = np.nan
        return with_surplus(frame)
    frame = frame.merge(prices.rename(columns={"Roster Position": "DK Pos"}),
                        on="key", how="left")
    return with_surplus(frame)


def with_surplus(frame):
    """Composite against what this hitter's price bracket normally delivers.

    `expected` is the median composite among hitters priced in the same band on the same
    slate; `surplus` is how far above it he sits. A thin band falls back to the slate median
    rather than to a median of two players, which would make one outlier the whole
    expectation for everybody near his price.
    """
    if frame is None or frame.empty:
        return frame
    out = frame.copy()
    salary = pd.to_numeric(out.get("Salary"), errors="coerce")
    composite = pd.to_numeric(out.get("Composite"), errors="coerce")
    out["Salary"] = salary

    out["band"] = pd.cut(salary, BANDS, labels=BAND_LABELS, right=False)
    overall = composite.median()
    counts = out.groupby("band", observed=False)["Composite"].transform("count")
    band_median = out.groupby("band", observed=False)["Composite"].transform("median")
    out["expected"] = np.where(counts >= MIN_BAND, band_median, overall)
    # **A hitter with no salary has no price to be measured against.** Without this he falls
    # through to the slate-wide median and is handed a surplus, which reads as a value call
    # on a player the market never priced.
    unpriced = salary.isna()
    out.loc[unpriced, "expected"] = np.nan
    out["surplus"] = composite - out["expected"]
    # Points per $1k of salary, the plain DFS reading, kept alongside the residual.
    out["per_1k"] = np.where(salary > 0, composite / (salary / 1000.0), np.nan)
    # Imported here, not at module scope: `data` defers its own import of this module the
    # same way, and keeping both deferred means neither has to care which loads first.
    from dashboards import data

    return data.rounded(out)


def band_curve(frame):
    """The expectation curve itself: one row per price band with its median composite."""
    if frame is None or frame.empty or "band" not in frame.columns:
        return pd.DataFrame(columns=["band", "midpoint", "expected", "n"])
    grouped = (frame.groupby("band", observed=True)
               .agg(expected=("Composite", "median"), n=("Composite", "count"),
                    midpoint=("Salary", "median")).reset_index())
    return grouped[grouped["n"] >= MIN_BAND].dropna(subset=["midpoint", "expected"])
