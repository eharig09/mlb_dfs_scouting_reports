"""Live game status, for catching games that fall off the slate after DK priced them.

A DK export is a snapshot. Download it in the morning, and a game postponed at 4pm still
has its full roster priced in the file at their normal salaries -- nothing in the export
changes. The optimizer will happily build around a starter who is not going to throw a
pitch, and there is no signal in the file to stop it.

DK does mark it once you re-download: `Game Info` reads "Postponed" instead of the usual
"ATL@NYM 07/28/2026 07:10PM ET". That path is handled in salaries.py. This module covers
the case that actually bites, where the file on disk predates the postponement.

Deliberately not cached. Postponement is the one fact about a slate that changes during
the day, and serving it from a cache written this morning would reintroduce the bug.
"""

import re
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

from .salaries import canon_team

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"

# Per-process only, never written to disk. A slate re-render builds the board once per game
# to place the DFS highlights, which would otherwise be one schedule call per game for an
# answer that cannot change inside a single run. A disk cache would be a different thing
# entirely -- it would outlive the postponement and put the bug straight back.
_MEMO = {}

# codedGameState values that mean no game will be played today. 'D' postponed, 'C'
# cancelled. Suspended games are left out on purpose: they resume and DK still scores them.
OFF_STATES = {"D", "C"}
OFF_TEXT = re.compile(r"postponed|cancell?ed", re.IGNORECASE)

# codedGameState is the authority on whether first pitch has happened. 'S' Scheduled,
# 'P' Pre-Game -- and Pre-Game covers **Warmup**, which is the whole reason this is read
# instead of abstractGameState.
#
# abstractGameState does NOT collapse the way it looks like it should: a game in Warmup
# reports abstractGameState 'Live' while codedGameState is still 'P'. Trusting the abstract
# state locked every player of a game roughly half an hour before it started -- observed
# 2026-08-01, STL@TOR flagged "already under way" 18 minutes before first pitch, removing
# 20 draftable players from the pool. Warmup is pre-game; DK takes entries throughout it.
PREGAME_CODES = {"S", "P"}

# Kept only as a fallback for a response with no codedGameState at all.
NOT_STARTED = "Preview"


def _is_off(status):
    return (str(status.get("codedGameState") or "").strip() in OFF_STATES
            or bool(OFF_TEXT.search(str(status.get("detailedState") or ""))))


def _has_started(status):
    coded = str(status.get("codedGameState") or "").strip()
    if coded:
        # Any other known code -- In Progress, Final, Game Over, Suspended -- is past first
        # pitch. Postponed and cancelled also land here and are masked off by `off`.
        return coded not in PREGAME_CODES
    # No coded state: fall back, and err toward started. Refusing to swap a player is
    # recoverable; uploading a lineup DK rejects at the buzzer is not.
    return str(status.get("abstractGameState") or "").strip() != NOT_STARTED


def _start_et(game):
    """First pitch as minutes past midnight ET -- the same scale the cached games use.

    This is what tells the two halves of a doubleheader apart. Both games share a pairing,
    so without a start time there is no way to ask "has *this* game begun" rather than the
    much blunter "has this team played today".
    """
    stamp = game.get("gameDate")
    if not stamp:
        return None
    try:
        moment = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
        eastern = moment.astimezone(ZoneInfo("America/New_York"))
        return eastern.hour * 60 + eastern.minute
    except Exception:
        return None


def game_status(date, timeout=10):
    """[{away, home, off, started, detail}] for every game on `date`.

    Raises requests exceptions; callers decide whether a lookup failure should block.
    """
    response = requests.get(
        SCHEDULE_URL,
        params={"sportId": 1, "date": str(date), "hydrate": "team,probablePitcher"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()

    games = []
    for day in payload.get("dates", []):
        for game in day.get("games", []):
            teams = game.get("teams", {})
            away = canon_team(teams.get("away", {}).get("team", {}).get("abbreviation"))
            home = canon_team(teams.get("home", {}).get("team", {}).get("abbreviation"))
            if not away or not home:
                continue
            status = game.get("status", {})
            detail = str(status.get("detailedState") or "").strip()
            reason = str(status.get("reason") or "").strip()
            off = _is_off(status)
            games.append({
                "away": away, "home": home, "off": off,
                # A called game never counts as started -- its players are gone from the
                # slate entirely, and calling them locked would block the very refill
                # the postponement is asking for.
                "started": _has_started(status) and not off,
                "start": _start_et(game),
                "number": game.get("gameNumber"),
                "away_sp": (teams.get("away", {}).get("probablePitcher") or {}).get("fullName"),
                "home_sp": (teams.get("home", {}).get("probablePitcher") or {}).get("fullName"),
                "detail": f"{detail} ({reason})" if reason else detail,
            })
    return games


def describe_postponed(postponed):
    """'ATL, NYM - Postponed (Inclement Weather)' for a {team: detail} mapping.

    Teams are grouped by reason, which is almost always one for the whole slate, so the
    usual case reads as one clause rather than repeating the reason per team.
    """
    if not postponed:
        return ""
    by_detail = {}
    for team in sorted(postponed):
        by_detail.setdefault(postponed[team], []).append(team)
    return "; ".join(f"{', '.join(teams)} - {detail}" for detail, teams in by_detail.items())


def postponed_teams(date, timeout=10):
    """({team: detail} for teams with no game left today, error or None).

    A team is only counted when *every* game it has today is off. On a doubleheader where
    the opener was played and the nightcap postponed, its hitters are still live for the
    game that happened, and dropping them would be as wrong as keeping them.

    Failure is reported rather than raised: a network problem must not stop a board from
    being built, but it must not pass silently as "nothing is postponed" either.
    """
    if str(date) in _MEMO:
        return dict(_MEMO[str(date)][0]), _MEMO[str(date)][1]
    try:
        games = game_status(date, timeout=timeout)
    except Exception as error:
        # Not memoized: a transient failure should be retried, not fixed for the run.
        return {}, str(error)

    playing, off = set(), {}
    for game in games:
        for team in (game["away"], game["home"]):
            if game["off"]:
                off.setdefault(team, game["detail"])
            else:
                playing.add(team)
    result = {team: detail for team, detail in off.items() if team not in playing}
    _MEMO[str(date)] = (result, None)
    return dict(result), None


def started_pairings(date, timeout=10):
    """({"AWAY@HOME": [{start, started, number, detail}, ...]}, error or None).

    Per *game*, not per team, which is the only granularity that works on a doubleheader:
    the opener can be in the fourth inning while the nightcap is still three hours away and
    fully draftable. Callers match on start time against the game they actually priced.

    Not memoized, deliberately: unlike a postponement this answer changes every few minutes,
    and a stale "not started" is the one that writes a rejected lineup.
    """
    try:
        games = game_status(date, timeout=timeout)
    except Exception as error:
        return {}, str(error)

    found = {}
    for game in games:
        key = f"{game['away']}@{game['home']}"
        found.setdefault(key, []).append({
            "start": game["start"], "started": game["started"],
            "number": game["number"], "detail": game["detail"],
        })
    return found, None


def pitching_elsewhere(date, timeout=10):
    """({normalized starter name: detail} for starters of games already under way, error).

    The doubleheader trap. DK prices a team's whole staff against the nightcap's start time,
    and until MLB names the game-2 starter the cached report falls back to the game-1 arm --
    so the man currently on the mound in the opener shows up as tonight's rosterable ace at
    a top-of-slate salary. He cannot record a single point for you.

    Matching is by name because that is what the cached reports carry; the schedule's own
    probable pitcher is the authority on who is actually throwing.
    """
    from .salaries import normalize_name

    try:
        games = game_status(date, timeout=timeout)
    except Exception as error:
        return {}, str(error)

    throwing = {}
    for game in games:
        if not game["started"]:
            continue
        for name in (game["away_sp"], game["home_sp"]):
            key = normalize_name(name)
            if key:
                throwing[key] = f"{game['away']}@{game['home']} {game['detail']}"
    return throwing, None


def started_teams(date, timeout=10):
    """({team: detail} for teams with no game left to start today, error or None).

    A team only counts when *every* game it has today is under way or over -- the same rule
    `postponed_teams` uses, and for the same reason. On a doubleheader with the opener in
    progress and the nightcap at 7:10, the team's evening roster is completely draftable,
    and calling it started would delete the very game the main slate is built on.

    Prefer `started_pairings` where the specific game is known; this is the team-level
    fallback for callers that only have a team.
    """
    try:
        games = game_status(date, timeout=timeout)
    except Exception as error:
        return {}, str(error)

    pending, started = set(), {}
    for game in games:
        for team in (game["away"], game["home"]):
            if game["started"]:
                started.setdefault(team, game["detail"])
            elif not game["off"]:
                pending.add(team)           # still has a game to come today
    return {team: detail for team, detail in started.items() if team not in pending}, None
