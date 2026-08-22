"""nflverse data access.

**No new dependency.** `nfl_data_py` is a thin wrapper over a set of parquet files published
as GitHub release assets, and it carries its own pandas pin that has a history of fighting
whatever the host project is on. Reading the parquet directly costs one `requests` call and
a `pd.read_parquet`, both of which this project already has, and it means an nflverse client
release can never break a slate build.

**The cache rule is the interesting part.** A completed season is immutable -- the 2023
weekly stats will never change again -- so it is cached forever and cost is paid once. The
*current* season is republished after every game, so it carries a TTL and is re-fetched.
Getting this backwards in either direction is a real failure: caching the live season
forever freezes a board at whatever it saw first (exactly the bug `_reliever_recent_workload`
had to pass `force=True` to avoid on the MLB side), while expiring completed seasons turns a
five-season backtest into a five-season download every time it runs.

    from nfl.data import load_weekly, load_schedules
    weekly = load_weekly(range(2021, 2026))
"""

import os
import time
from datetime import datetime

import pandas as pd
import requests

CACHE_DIR = os.path.join(".cache", "nflverse")
RELEASE_BASE = "https://github.com/nflverse/nflverse-data/releases/download"

# How stale the in-progress season may get before it is pulled again. Six hours: nflverse
# republishes within a few hours of a game finishing, and nothing in a DFS workflow needs
# last night's box score faster than that.
CURRENT_SEASON_TTL = 6 * 3600
# Schedules include future weeks and are revised for flexed kickoff times, so they expire
# on the same clock as the live season no matter which seasons are being read.
SCHEDULE_TTL = 6 * 3600

# asset -> (release tag, filename template). A template with no {season} is a single file
# covering every season.
ASSETS = {
    "weekly": ("stats_player", "stats_player_week_{season}.parquet"),
    "schedules": ("schedules", "games.parquet"),
    "snap_counts": ("snap_counts", "snap_counts_{season}.parquet"),
    "rosters": ("rosters", "roster_{season}.parquet"),
    "injuries": ("injuries", "injuries_{season}.parquet"),
    "depth_charts": ("depth_charts", "depth_charts_{season}.parquet"),
}


class NflverseError(Exception):
    pass


def current_season(today=None):
    """The season year an NFL date belongs to.

    A season is named for the calendar year it starts in and runs into February, so January
    and February belong to the previous year's season. Reading the calendar year directly
    would file the Super Bowl under the season that has not kicked off yet.
    """
    today = today or datetime.today()
    return today.year if today.month >= 3 else today.year - 1


def _cache_path(asset, season=None):
    name = f"{asset}.parquet" if season is None else f"{asset}_{season}.parquet"
    return os.path.join(CACHE_DIR, name)


def _is_fresh(path, asset, season):
    if not os.path.exists(path):
        return False
    if asset == "schedules":
        return (time.time() - os.path.getmtime(path)) < SCHEDULE_TTL
    if season is not None and season < current_season():
        return True                      # a finished season cannot change
    return (time.time() - os.path.getmtime(path)) < CURRENT_SEASON_TTL


def nflverse_frame(asset, season=None, force=False):
    """One nflverse asset as a DataFrame, disk-cached.

    Returns an empty frame for a season nflverse has not published yet -- an August request
    for the season about to start is a normal thing to do, not an error.
    """
    if asset not in ASSETS:
        raise NflverseError(f"unknown asset '{asset}'; have {sorted(ASSETS)}")
    tag, template = ASSETS[asset]
    if "{season}" in template and season is None:
        raise NflverseError(f"asset '{asset}' is published per season; pass one")

    path = _cache_path(asset, season)
    if not force and _is_fresh(path, asset, season):
        return pd.read_parquet(path)

    url = f"{RELEASE_BASE}/{tag}/{template.format(season=season)}"
    try:
        response = requests.get(url, timeout=120)
    except requests.RequestException as error:
        if os.path.exists(path):
            # A stale copy beats no copy: the caller is mid-build and the alternative is a
            # crash over a season that is very likely unchanged anyway.
            print(f"⚠️ nflverse unreachable ({error}); using cached {os.path.basename(path)}")
            return pd.read_parquet(path)
        raise NflverseError(f"could not reach nflverse for {asset} {season or ''}: {error}")

    if response.status_code == 404:
        return pd.DataFrame()
    if not response.ok:
        if os.path.exists(path):
            print(f"⚠️ nflverse returned {response.status_code}; using cached copy")
            return pd.read_parquet(path)
        raise NflverseError(f"nflverse returned {response.status_code} for {url}")

    os.makedirs(CACHE_DIR, exist_ok=True)
    # Written through a temp file so an interrupted download cannot leave a truncated
    # parquet that every later run then fails to read.
    temporary = path + ".part"
    with open(temporary, "wb") as handle:
        handle.write(response.content)
    os.replace(temporary, path)
    return pd.read_parquet(path)


def _stack(asset, seasons, force=False):
    frames = []
    for season in seasons:
        frame = nflverse_frame(asset, int(season), force=force)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_weekly(seasons, season_type="REG", force=False):
    """Per-player per-week stat lines. The substrate for scoring and for projections.

    Regular season only by default. Postseason rows are real games but they are a different
    population -- different opponents, different rest, a quarter of the league -- and mixing
    them into a rate estimate silently reweights it toward whoever went deep in January.
    """
    frame = _stack("weekly", seasons, force=force)
    if frame.empty:
        return frame
    if season_type and "season_type" in frame.columns:
        frame = frame[frame["season_type"] == season_type]
    return frame.reset_index(drop=True)


def load_schedules(seasons=None, force=False):
    """Every game, past and scheduled: kickoff, teams, scores, spread and total.

    The betting lines are why this matters beyond dates. An implied team total is the single
    best public prior on how many points an offense will score, which is what a projection
    is ultimately trying to divide up among nine players.
    """
    frame = nflverse_frame("schedules", force=force)
    if frame.empty or seasons is None:
        return frame
    wanted = {int(s) for s in seasons}
    return frame[frame["season"].isin(wanted)].reset_index(drop=True)


def load_snap_counts(seasons, force=False):
    """Offensive and defensive snap shares. Opportunity, which is upstream of production."""
    return _stack("snap_counts", seasons, force=force)


def load_rosters(seasons, force=False):
    """Weekly rosters, including the id crosswalk to other sources."""
    return _stack("rosters", seasons, force=force)


def load_injuries(seasons, force=False):
    """Practice participation and game status -- the NFL analogue of a confirmed lineup."""
    return _stack("injuries", seasons, force=force)


def load_depth_charts(seasons, force=False):
    """Positional depth. How a projection knows who is WR1 before the snaps prove it."""
    return _stack("depth_charts", seasons, force=force)


def available_weeks(seasons, season_type="REG"):
    """[(season, week)] that have completed stat lines, oldest first.

    The walk-forward evaluator needs to know what it can score against without downloading
    the whole thing twice, and a season that has started but not finished must report only
    the weeks actually in the data.
    """
    frame = load_weekly(seasons, season_type=season_type)
    if frame.empty:
        return []
    pairs = frame[["season", "week"]].drop_duplicates()
    return sorted((int(s), int(w)) for s, w in pairs.itertuples(index=False))
