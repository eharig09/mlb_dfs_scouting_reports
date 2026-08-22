"""Rest-of-season projections from Fangraphs, used as the rate estimate for both sides.

The projection model regressed a player's in-season rate line toward a prior weighted by
sample size -- league average for pitchers (`REG_SP`), league-average-times-playing-time for
hitters (`_replacement_baseline`). That is a generic answer to a player-specific question: it
treats every 40-plate-appearance hitter as the same 40-plate-appearance hitter, and every
three-inning starter as the same three-inning starter. Steamer, ZiPS and The Bat already
solve it properly, per player, using minor-league and prior-season history this model never
sees.

**They are used as the estimate, not as a shrinkage target.** A rest-of-season projection is
not a prior -- it is a *posterior* that already contains the player's in-season line. The
first attempt regressed a player's season rate toward his own ROS number, which counts that
season twice, and the walk-forward diff showed exactly that: bias worsened precisely in the
tiers full of established regulars, whose season line and ROS line agree and were being
stacked. Taking the ROS rate directly is both simpler and what the numbers wanted.

**These exports are refreshed by hand.** A file pulled today reflects everything that has
happened up to today, so scoring a *past* date against it leaks -- the estimate has already
seen the games being predicted. `age_days` and `staleness_warning` exist so callers can say
so out loud rather than quietly trusting whatever is on disk.
"""

import glob
import os
import time

import pandas as pd

ROS_DIR = "ROS_Projections"

# A system's row only counts when it projects enough playing time to be a real forecast.
# The deep exports (Steamer reaches ~6,100 pitchers and ~4,800 hitters) run far into the
# minors, where a projection of one or two plate appearances is a placeholder rather than an
# opinion, and averaging those in would drag a composite toward nothing in particular.
MIN_PROJECTED_PA = 20.0
MIN_PROJECTED_IP = 5.0

# How old a manually-refreshed export may get before it is worth complaining about.
STALE_AFTER_DAYS = 7.0

# Rates taken from each file type. These are exactly the fields the model would otherwise
# have had to estimate from a partial season.
HITTER_RATES = ("avg", "iso", "bb_rate", "k_rate", "hr_per_ab", "fpts")
PITCHER_RATES = ("fip", "era", "whip", "k_rate", "bb_rate", "fpts")

# Innings a start is worth, for putting a pitcher's per-inning fantasy rate on the same
# per-appearance footing as a hitter's per-game rate. Only used to scale `fpts`, which is
# consumed as a percentile, so the exact value moves nothing -- it just has to be a constant.
FPTS_INNINGS_PER_START = 5.2

_CACHE = {}


def _kind(frame):
    """'P', 'H' or None -- which export this is, decided by the columns present."""
    columns = set(frame.columns)
    if {"FIP", "WHIP", "IP"} <= columns:
        return "P"
    if {"PA", "AVG", "ISO"} <= columns:
        return "H"
    return None


def _numeric(frame, column):
    """A float Series for `column`, or all-NaN when the export lacks it."""
    if column not in frame.columns:
        return pd.Series(float("nan"), index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def _hitter_rates(frame):
    ab = _numeric(frame, "AB")
    out = pd.DataFrame({
        "sample": _numeric(frame, "PA"),
        "avg": _numeric(frame, "AVG"),
        "iso": _numeric(frame, "ISO"),
        "bb_rate": _numeric(frame, "BB%"),
        "k_rate": _numeric(frame, "K%"),
        # Consensus DK points per game. Not used by the projection model -- it feeds the
        # ownership model, as a stand-in for the public number the field is reading.
        "fpts": _numeric(frame, "FPTS/G"),
        # Per at-bat rather than per plate appearance, because that is the denominator the
        # model uses when it turns a home-run rate into a count.
        "hr_per_ab": _numeric(frame, "HR") / ab.replace(0, float("nan")),
    })
    return out, MIN_PROJECTED_PA


def _pitcher_rates(frame):
    out = pd.DataFrame({
        "sample": _numeric(frame, "IP"),
        "fip": _numeric(frame, "FIP"),
        "era": _numeric(frame, "ERA"),
        "whip": _numeric(frame, "WHIP"),
        "k_rate": _numeric(frame, "K%"),
        "bb_rate": _numeric(frame, "BB%"),
        # Per-inning here, scaled to a start so it is comparable to the hitter rate.
        "fpts": _numeric(frame, "FPTS/IP") * FPTS_INNINGS_PER_START,
    })
    return out, MIN_PROJECTED_IP


def _table(frame, kind):
    """One system's rate lines, indexed by MLBAM id, thin projections dropped."""
    build = _pitcher_rates if kind == "P" else _hitter_rates
    out, minimum = build(frame)
    out.index = pd.to_numeric(frame["MLBAMID"], errors="coerce")
    out = out[out.index.notna()]
    out = out[out["sample"] >= minimum]
    out.index = out.index.astype("int64")
    out = out[~out.index.duplicated(keep="first")]
    return out.drop(columns=["sample"])


def load(directory=ROS_DIR, refresh=False):
    """`({"H": {mlbam: rates}, "P": {mlbam: rates}}, meta)` averaged across every system.

    File type is detected from the columns rather than the filename, so hitter and pitcher
    exports can sit in one folder and be renamed freely. A file matching neither shape is
    skipped rather than parsed as the wrong kind -- reading a pitcher export with hitter
    columns is exactly the failure this guards, and it is not hypothetical.

    Every system present gets an equal vote. Weighting by past accuracy would be better in
    principle and is now possible since the exports are named, but it needs a season of
    scored history per system to fit and cannot be read off these files.
    """
    key = os.path.abspath(directory)
    if not refresh and key in _CACHE:
        return _CACHE[key]

    paths = sorted(glob.glob(os.path.join(directory, "*.csv")))
    meta = {"files": {}, "age_days": None, "stale": False,
            "players": {"H": 0, "P": 0}, "systems": {"H": 0, "P": 0}}
    tables = {"H": [], "P": []}

    for path in paths:
        name = os.path.basename(path)
        try:
            frame = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            meta["files"][name] = "unreadable"
            continue
        kind = _kind(frame)
        if kind is None or "MLBAMID" not in frame.columns:
            meta["files"][name] = "skipped (unrecognised columns)"
            continue
        table = _table(frame, kind)
        meta["files"][name] = f"{kind} {len(table)} players"
        if not table.empty:
            tables[kind].append(table)

    rates = {"H": {}, "P": {}}
    for kind, columns in (("H", HITTER_RATES), ("P", PITCHER_RATES)):
        if not tables[kind]:
            continue
        stacked = pd.concat([t.reindex(columns=list(columns)) for t in tables[kind]])
        composite = stacked.groupby(level=0).mean()
        meta["systems"][kind] = len(tables[kind])
        meta["players"][kind] = int(len(composite))
        rates[kind] = {
            int(k): {c: (None if pd.isna(v[c]) else float(v[c])) for c in columns}
            for k, v in composite.iterrows()
        }

    if paths:
        newest = max(os.path.getmtime(p) for p in paths)
        meta["age_days"] = round((time.time() - newest) / 86400.0, 2)
        meta["stale"] = meta["age_days"] > STALE_AFTER_DAYS

    _CACHE[key] = (rates, meta)
    return _CACHE[key]


def staleness_warning(meta):
    """Human-readable complaint about the hand-refreshed exports, or None when they are fine."""
    if not meta.get("files"):
        return (f"no rest-of-season exports found in {ROS_DIR}/ -- rates will fall back to "
                f"league-average regression")
    missing = [k for k in ("H", "P") if not meta.get("players", {}).get(k)]
    if missing:
        side = " and ".join({"H": "hitter", "P": "pitcher"}[k] for k in missing)
        return f"no {side} rest-of-season exports in {ROS_DIR}/ -- that side falls back to league"
    if meta.get("stale"):
        return (f"rest-of-season exports are {meta['age_days']:.1f} days old "
                f"(>{STALE_AFTER_DAYS:.0f}); refresh {ROS_DIR}/ or the estimates will drift")
    return None
