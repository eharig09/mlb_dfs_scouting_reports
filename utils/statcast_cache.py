"""Season-scale Statcast pulls assembled from immutable date chunks.

The problem this solves
-----------------------
`pybaseball.statcast()` issues **one HTTP request per day** of the range it is given, so a
season-to-date pull is ~170 requests and ~8 minutes. The old cache keyed that pull on
(start, end); `end` moves every day, so every run missed the cache and re-downloaded the
entire season to gain one day of games -- and left another ~685 MB pickle behind.

The fix is to stop caching the *request* and start caching the *data*. A completed day of
baseball never changes, so it is cached once and reused forever. A range is then assembled
from chunks and only genuinely-new days ever touch the network.

Chunk layout
------------
* **Whole calendar months** inside the range get one pickle each. Six file reads cover a
  season, so assembling a range stays cheap.
* **The trailing partial month** is cached per day, because that is the part that grows.
  Yesterday's pickle is still valid today; only the new day is fetched.
* When a partial month later completes, its day pickles are **rolled up** into the month
  pickle locally rather than re-downloaded. That is what keeps the steady-state marginal
  cost at one day forever instead of resetting to a full month every 30 days.

Immutability caveat
-------------------
Savant occasionally revises past days (re-classified pitch types, corrected batted-ball
data). The old always-re-pull behaviour picked those up silently; chunk caching does not.
`refresh_days()` re-fetches a specific window for that, and `force=True` bypasses chunks
entirely.
"""

import calendar
from datetime import date, datetime, timedelta

import pandas as pd

# Chunks are stored under their own namespace so the legacy monolithic pickles in
# .cache/statcast stay readable (and seedable) rather than being shadowed.
CHUNK_NAMESPACE = "statcast_chunk"


def _to_date(value):
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def _iso(value):
    return _to_date(value).strftime("%Y-%m-%d")


def _month_end(day):
    return day.replace(day=calendar.monthrange(day.year, day.month)[1])


def _next_month(day):
    return (day.replace(day=1) + timedelta(days=32)).replace(day=1)


def plan_chunks(start_date, end_date):
    """Split [start, end] into ("month"|"day", chunk_start, chunk_end) pieces.

    Whole months covered by the range become month chunks; the trailing partial month is
    emitted a day at a time. Returned oldest-first.
    """
    start, end = _to_date(start_date), _to_date(end_date)
    if start > end:
        return []

    chunks = []
    cursor = start
    while cursor <= end:
        m_start = cursor.replace(day=1)
        m_end = _month_end(cursor)
        # A month chunk is only valid when the range covers the month end-to-end; a
        # partial month cached under a month key would be silently short of games.
        if cursor == m_start and m_end <= end:
            chunks.append(("month", cursor, m_end))
            cursor = _next_month(cursor)
        else:
            stop = min(m_end, end)
            day = cursor
            while day <= stop:
                chunks.append(("day", day, day))
                day += timedelta(days=1)
            cursor = stop + timedelta(days=1)
    return chunks


# A day that has no rows *yet* (tonight's games are unplayed, or Savant has not posted
# them) must not be cached as permanently empty, or that day stays missing forever. A day
# this far in the past with no rows is genuinely gameless -- an off-day or the All-Star
# break -- and is safe to remember, which matters because an empty pull is slow (~50s).
_EMPTY_CHUNK_MIN_AGE_DAYS = 3


def _chunk_path(fetcher, start, end):
    import utils.cache as uc
    payload = {"func": getattr(fetcher, "__name__", str(fetcher)),
               "args": (_iso(start), _iso(end)), "kwargs": {}}
    return uc._path(CHUNK_NAMESPACE, uc._cache_key(payload), "pkl")


def _fetch(fetcher, start, end, force=False):
    """Read a chunk from disk, or fetch and store it.

    Persistence is handled here rather than via `cached_dataframe_call` so that an empty
    result at the live edge of the season is not written -- see `_EMPTY_CHUNK_MIN_AGE_DAYS`.
    """
    import pickle
    import utils.cache as uc

    path = _chunk_path(fetcher, start, end)
    if uc.cache_enabled() and not force and path.exists():
        with path.open("rb") as f:
            return pickle.load(f)

    data = fetcher(_iso(start), _iso(end))

    if uc.cache_enabled():
        empty = data is None or getattr(data, "empty", True)
        too_recent = (date.today() - _to_date(end)).days < _EMPTY_CHUNK_MIN_AGE_DAYS
        if not (empty and too_recent):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return data


def _chunk_is_cached(fetcher, start, end):
    """True when this chunk would be served from disk rather than fetched."""
    return _chunk_path(fetcher, start, end).exists()


# How many days may be missing from a just-completed month before it is cheaper to pull
# the month in one go than to fill the gaps a day at a time. A month that was recently the
# live month is missing only the days after the last run, so this is normally 1 or 2; a
# month never touched at all is missing all thirty and should be fetched wholesale.
_ROLLUP_MAX_MISSING_DAYS = 10


def _rollup_month(fetcher, m_start, m_end):
    """Build a completed month from its day chunks instead of re-downloading it.

    Days already on disk are reused and any small gap is filled a day at a time, which is
    the normal case: the month was the live month until yesterday, so only the days since
    the last run are absent. Returns None when too much is missing to be worth it, leaving
    the caller to fetch the month in one request.
    """
    days = []
    day = m_start
    while day <= m_end:
        days.append(day)
        day += timedelta(days=1)

    missing = [d for d in days if not _chunk_is_cached(fetcher, d, d)]
    if len(missing) > _ROLLUP_MAX_MISSING_DAYS:
        return None

    frames = [_fetch(fetcher, d, d) for d in days]
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _write_chunk(fetcher, start, end, frame):
    """Persist an already-assembled frame under a chunk key (no network)."""
    import pickle
    import utils.cache as uc
    if not uc.cache_enabled():
        return
    path = _chunk_path(fetcher, start, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(frame, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_statcast_range(start_date, end_date, fetcher=None, force=False, verbose=False):
    """Pitch-level Statcast for [start_date, end_date], assembled from cached chunks."""
    if fetcher is None:
        from pybaseball import statcast as fetcher

    if force:
        return _fetch(fetcher, start_date, end_date, force=True)

    frames = []
    for kind, c_start, c_end in plan_chunks(start_date, end_date):
        if kind == "month" and not _chunk_is_cached(fetcher, c_start, c_end):
            # The month may already exist as day chunks from when it was the live month.
            rolled = _rollup_month(fetcher, c_start, c_end)
            if rolled is not None:
                _write_chunk(fetcher, c_start, c_end, rolled)
                if verbose:
                    print(f"   statcast: rolled up {_iso(c_start)}..{_iso(c_end)} from day chunks")
                if not rolled.empty:
                    frames.append(rolled)
                continue
        if verbose and not _chunk_is_cached(fetcher, c_start, c_end):
            print(f"   statcast: fetching {_iso(c_start)}..{_iso(c_end)}")
        frame = _fetch(fetcher, c_start, c_end)
        if frame is not None and not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def refresh_days(start_date, end_date, fetcher=None):
    """Re-fetch a window's day chunks, for when Savant revises past data.

    Month chunks covering the window are dropped so they rebuild from the fresh days.
    """
    if fetcher is None:
        from pybaseball import statcast as fetcher

    start, end = _to_date(start_date), _to_date(end_date)
    day = start
    while day <= end:
        _fetch(fetcher, day, day, force=True)
        day += timedelta(days=1)

    cursor = start.replace(day=1)
    while cursor <= end:
        m_end = _month_end(cursor)
        path = _chunk_path(fetcher, cursor, m_end)
        if path.exists():
            path.unlink()
        cursor = _next_month(cursor)
