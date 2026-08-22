"""Assemble a full slate: every cached game -> projections -> value -> tiers."""

import glob
import os
import pickle
import tempfile
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from .naming import UNKNOWN_SLATE, slate_for
from .ownership import attach_actual_ownership, estimate_ownership
from .profiling import profiler
from .projections import project_game
from .results import contest_summary, match_contest
from .salaries import (
    AmbiguousSlate, attach_salaries, canon_team, describe_slate, find_salary_file,
    list_salary_files, load_salaries, normalize_name, postponed_in_export, slate_date,
    slate_game_times, slate_games,
)
from .schedule import postponed_teams

REPORT_DATA_DIR = os.path.join(".cache", "report_data")

# GPP lean: ceiling-per-dollar carries most of the weight, mean projection anchors it.
GPP_WEIGHTS = {"proj": 0.30, "ceiling_value": 0.70}
GPP_WEIGHTS_NO_SALARY = {"proj": 0.35, "ceiling": 0.65}

# Cash lean: floor per dollar, with bust risk counted against directly. Deliberately does
# not use ceiling at all -- the whole point is what the play returns when it goes wrong.
CASH_WEIGHTS = {"floor_value": 0.65, "safety": 0.35}


def cached_games(date, data_dir=REPORT_DATA_DIR):
    """Cached payload paths for a slate date, as (path, away, home).

    Doubleheader game 2 is stored as `<date>_<away>_<home>_g2.pkl`; both games are
    returned so a slate containing either one is complete.
    """
    games = []
    for path in sorted(glob.glob(os.path.join(data_dir, f"{date}_*.pkl"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        parts = stem.split("_")
        if len(parts) >= 3:
            games.append((path, parts[1], parts[2]))
    return games


def game_number(path):
    """Doubleheader game number encoded in a cache path (1 when absent)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    tail = stem.split("_")[-1]
    if tail.startswith("g") and tail[1:].isdigit():
        return int(tail[1:])
    return 1


def _load(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _persist_payload(path, payload):
    """Atomically replace a report cache after a local-only calibration refresh."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    handle, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".calibration.tmp", dir=directory
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def refresh_game_calibration(payload, path):
    """Ensure a cached game uses the current report-model artifact before projection."""
    # Lazy import keeps ordinary dfs module imports light and avoids making the large
    # report module part of slate discovery. It is loaded only when a board is built.
    from scouting_report import refresh_payload_model_calibration

    payload, changed, provenance = refresh_payload_model_calibration(payload)
    if changed:
        _persist_payload(path, payload)
    return payload, changed, provenance


def _start_minutes_et(payload):
    """First pitch as minutes past midnight ET, from the cached environment."""
    stamp = (payload.get("advanced_context", {}).get("environment") or {}).get("game_datetime")
    if not stamp:
        return None
    try:
        moment = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
        eastern = moment.astimezone(ZoneInfo("America/New_York"))
        return eastern.hour * 60 + eastern.minute
    except Exception:
        return None


def resolve_doubleheaders(games, salary_file):
    """Drop the doubleheader game that is not on the priced slate.

    Both games of a doubleheader share a pairing, so leaving both cached would list every
    player on that team twice with different starters. The DK export names one start time
    per game, which is the only thing that tells them apart -- so the cached game whose
    first pitch is nearest a priced start time wins.
    """
    from collections import defaultdict
    by_pairing = defaultdict(list)
    for entry in games:
        _, away, home = entry
        by_pairing[f"{canon_team(away)}@{canon_team(home)}"].append(entry)

    duplicates = {k: v for k, v in by_pairing.items() if len(v) > 1}
    if not duplicates:
        return games, []

    wanted = slate_game_times(salary_file) if salary_file else {}
    keep, dropped = [], []
    for pairing, entries in by_pairing.items():
        if len(entries) == 1:
            keep.extend(entries)
            continue
        targets = wanted.get(pairing)
        if not targets:
            # No priced start time to match against: keep the later game, which is the one
            # a main/evening slate almost always uses, and say so.
            entries = sorted(entries, key=lambda e: _start_minutes_et(_load(e[0])) or 0)
            keep.append(entries[-1])
            dropped.extend((e, pairing, "no priced start time; kept the later game")
                           for e in entries[:-1])
            continue
        scored = []
        for entry in entries:
            start = _start_minutes_et(_load(entry[0]))
            gap = min((abs(start - t) for t in targets), default=10 ** 6) if start is not None else 10 ** 6
            scored.append((gap, entry))
        scored.sort(key=lambda item: item[0])
        keep.append(scored[0][1])
        dropped.extend((entry, pairing, "not the priced game") for _, entry in scored[1:])
    return keep, dropped


def started_players(players, meta, date):
    """(locked mask, pitching-elsewhere mask, error or None).

    Two different kinds of unavailable, which must not be conflated:

    * **locked** -- the player's own game has begun. DK will not let you add *or* remove
      them, so late swap has to keep them exactly where they are.
    * **elsewhere** -- the player's DK game has not started, but they are already pitching
      in another one. That is the doubleheader case: DK prices the whole staff against the
      nightcap, so the opener's starter is still listed as swappable and will score zero.
      Unlike a locked player, this one can and should be moved.

    The locked check is resolved per game rather than per team. A doubleheader team has one
    game in progress and one still to come; asking "has this team played today" deletes the
    nightcap, which is exactly the game a main slate is built on.
    """
    from .schedule import pitching_elsewhere, started_pairings

    empty = pd.Series(False, index=players.index)
    if players.empty or "Game" not in players.columns:
        return empty, empty, None

    pairings, error = started_pairings(date)
    if error:
        return empty, empty, error

    starts = meta.get("game_starts") or {}
    verdict = {}
    for pairing in players["Game"].dropna().unique():
        away, _, home = str(pairing).partition("@")
        key = f"{canon_team(away)}@{canon_team(home)}"
        scheduled = pairings.get(key) or []
        if not scheduled:
            verdict[pairing] = False        # unknown to the schedule; do not guess it away
            continue
        ours = starts.get(key)
        if ours is None or len(scheduled) == 1:
            # One game, or no start time to match on: every game of the pairing must be
            # under way before its players are unavailable.
            verdict[pairing] = all(game["started"] for game in scheduled)
            continue
        nearest = min(scheduled, key=lambda game: abs((game["start"] if game["start"]
                                                       is not None else 10 ** 6) - ours))
        verdict[pairing] = bool(nearest["started"])

    locked = players["Game"].map(verdict).fillna(False).astype(bool)

    throwing, _ = pitching_elsewhere(date)
    if throwing:
        keys = players["Name"].map(normalize_name)
        elsewhere = (players["Type"] == "P") & keys.isin(throwing) & ~locked
    else:
        elsewhere = empty
    return locked, elsewhere, None


MIN_GAMES_REPRESENTED = 2


def drop_started(players, meta, date, allow=False):
    """Drop players whose game has already begun -- they cannot be drafted at all.

    The slate removes postponed games but knows nothing about start times, so a run late in
    the evening would otherwise build around a player who is already batting and hand back
    an entry DK will not take.

    Resolved per game, not per team: on a doubleheader the opener can be in progress while
    the nightcap -- the game the main slate is priced on -- is hours away and completely
    draftable.

    A slate where *every* game has started is a review or backtest, not a live build, so
    nothing is dropped there: filtering would empty the pool and fail with a message about
    the wrong thing. (`dfs.review` calls the optimizer directly and never comes through
    here, which is why that path is untouched by any of this.)

    Lives here rather than in `dfs.optimize` because it is a *correctness* filter, not an
    optimizer setting. It used to sit in the CLI, which meant the candidate, field, contest
    and portfolio paths silently skipped it -- a portfolio built at 8pm would happily
    include a player whose 7:05 game was in the third inning, with no warning.
    """
    if allow:
        return players, []
    locked, elsewhere, error = started_players(players, meta, date)
    if error:
        return players, [f"[!] could not check start times: {error}. Players from games "
                         f"already under way may still be in the pool — check before entering."]
    if locked.all():
        return players, []              # nothing live at all: a review or backtest

    notes = []
    if elsewhere.any():
        # Worth naming individually. This one looks like a bargain rather than a mistake --
        # a top-salary arm with a full projection whose game, per DK, has not started.
        for name in players[elsewhere]["Name"]:
            notes.append(f"[!] {name} removed — already pitching the earlier game of a "
                         f"doubleheader. DK prices the whole staff against the nightcap, so "
                         f"he is listed as available and would score nothing.")
    if locked.any():
        gone = sorted(set(players[locked]["Game"].dropna()))
        notes.append(f"[!] {int(locked.sum())} player(s) removed — "
                     f"{', '.join(gone)} already under way and cannot be drafted.")
    drop = locked | elsewhere
    if not drop.any():
        return players, notes

    remaining = players[~drop]
    games = remaining["Game"].dropna().nunique() if "Game" in remaining.columns else 0
    if games < MIN_GAMES_REPRESENTED:
        notes.append(f"    Only {games} game(s) left on the board; DK needs "
                     f"{MIN_GAMES_REPRESENTED}. Use --allow-started to build anyway.")
    return remaining.copy(), notes


def _percentile(series):
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() <= 1:
        return pd.Series(50.0, index=series.index)
    return values.rank(pct=True) * 100


# How far our projection must outrun the player's season baseline to count as leverage.
LEVERAGE_EDGE = 20.0

# Role thresholds. The bust caps come straight from the backtest: 44% of hitter-games
# score 3 or fewer points, so demanding a low bust rate from a hitter would return an
# empty list. Pitchers are held to a much stricter standard because they can actually
# meet it -- the middle projection bucket busted 0% of the time.
CEILING_ROLE_PCT = 75.0
FLOOR_ROLE_PCT = 70.0
FLOOR_MAX_BUST = {"H": 42.0, "P": 22.0}


def _assign_tier(row, has_salary):
    """GPP-oriented buckets.

    Leverage is the one that matters most in tournaments. It is deliberately *not*
    "high ceiling, low projection" -- ceiling is a function of the projection here, so
    that combination never occurs. It is instead "tonight's matchup is much better than
    this player's season line," which is what the field prices and rosters on.
    """
    gpp = row.get("GPP", 0.0)
    proj_pct = row.get("Proj Pct", 50.0)
    ceiling_pct = row.get("Ceiling Pct", 50.0)
    value_pct = row.get("Value Pct", 50.0)
    edge = row.get("Edge")

    if row.get("Lineup") not in (None, "", "Confirmed") and row.get("Type") == "H":
        # Projected/unconfirmed lineups still rank, but never as a core play.
        if gpp >= 80:
            return "Risk"

    if gpp >= 88 and proj_pct >= 78:
        return "Core"
    if has_salary and value_pct >= 85 and proj_pct >= 45:
        return "Value"
    if edge is not None and not pd.isna(edge) and edge >= LEVERAGE_EDGE and ceiling_pct >= 50:
        return "Leverage"
    if gpp <= 18:
        return "Fade"
    return "Neutral"


def _assign_role(row, has_salary):
    """Ceiling play, floor play, both, or neither -- independent of the GPP tier.

    Kept separate from Tier because they answer different questions: Tier is "is this
    priced well for a tournament", Role is "what kind of play is this". A player can
    be a strong ceiling play and a terrible floor play at the same time, and collapsing
    that into one label is what makes most DFS boards useless for cash games.
    """
    ceiling_value_pct = row.get("Ceil Value Pct")
    cash = row.get("CASH")
    bust = row.get("Bust%")
    if not has_salary or ceiling_value_pct is None or pd.isna(ceiling_value_pct):
        return ""

    is_ceiling = ceiling_value_pct >= CEILING_ROLE_PCT
    is_floor = (cash is not None and not pd.isna(cash) and cash >= FLOOR_ROLE_PCT
                and bust is not None and bust <= FLOOR_MAX_BUST.get(row.get("Type"), 100))
    if is_ceiling and is_floor:
        return "Ceiling+Floor"
    if is_ceiling:
        return "Ceiling"
    if is_floor:
        return "Floor"
    return ""


def build_slate(date, salary_path=None, data_dir=REPORT_DATA_DIR, slate=None, check_schedule=True):
    """Build the ranked slate board.

    Returns (players_df, stacks_df, meta). Salaries are optional -- without them the
    board still ranks by projection and ceiling, it just can't show value per dollar.
    """
    games = cached_games(date, data_dir)
    meta = {"date": date, "games": [], "skipped": [], "missing_games": [],
            "salary_file": None, "has_salary": False, "ambiguous_slates": [],
            "slate_label": slate_for(requested=slate) if slate else UNKNOWN_SLATE,
            "environment_error": None, "cached_but_unreadable": 0,
            "doubleheaders": [], "postponed": {}, "postponed_players": 0,
            "schedule_error": None, "game_starts": {},
            "calibration_fingerprint": None, "calibration_created_at": None,
            "calibration_feature_version": None, "calibration_refreshed_games": [],
            "calibration_errors": [], "calibration_consistent": True}

    # The salary file is resolved up front because doubleheader disambiguation needs its
    # start times -- both games of a pairing would otherwise list every player twice.
    try:
        salary_file = find_salary_file(date, salary_path, slate=slate)
    except AmbiguousSlate as choice:
        # Surfaced rather than guessed: picking the wrong contest silently would price
        # every player off a slate you are not entering.
        meta["ambiguous_slates"] = choice.candidates
        salary_file = None

    games, dropped = resolve_doubleheaders(games, salary_file)
    for (path, away, home), pairing, reason in dropped:
        meta["doubleheaders"].append(f"{pairing}: dropped {os.path.basename(path)} ({reason})")

    rows = []
    for path, away, home in games:
        try:
            with profiler.stage("normalize"):
                payload = _load(path)
                try:
                    payload, calibration_changed, calibration = refresh_game_calibration(
                        payload, path
                    )
                except Exception as error:
                    raise RuntimeError(f"calibration refresh failed: {error}") from error
            fingerprint = calibration.get("fingerprint")
            if meta["calibration_fingerprint"] not in (None, fingerprint):
                meta["calibration_consistent"] = False
            meta["calibration_fingerprint"] = (
                fingerprint if meta["calibration_fingerprint"] in (None, fingerprint)
                else "mixed"
            )
            meta["calibration_created_at"] = calibration.get("created_at")
            meta["calibration_feature_version"] = calibration.get("feature_version")
            if calibration_changed:
                meta["calibration_refreshed_games"].append(f"{away}@{home}")
            with profiler.stage("project"):
                game_rows = project_game(payload)
            # Tag the game so the optimizer can enforce DK's two-game minimum.
            for row in game_rows:
                row["Game"] = f"{away}@{home}"
            rows.extend(game_rows)
            meta["games"].append(f"{away}@{home}")
            # First pitch of the game that was actually kept. On a doubleheader the pairing
            # alone cannot say which half is on the slate, so anything asking "has this
            # game started" needs the resolved start time, not just the teams.
            meta["game_starts"][f"{canon_team(away)}@{canon_team(home)}"] = \
                _start_minutes_et(payload)
        except ImportError as error:
            # A missing package fails every game identically. That is an environment
            # problem, not missing data, and must not be reported as "no cached games" --
            # that would send you off to regenerate reports that are already on disk.
            meta["environment_error"] = str(error)
            meta["skipped"].append(f"{away}@{home}: {error}")
        except Exception as error:                      # one bad payload shouldn't kill the slate
            if "calibration" in str(error).casefold():
                meta["calibration_errors"].append(f"{away}@{home}: {error}")
                meta["calibration_consistent"] = False
            meta["skipped"].append(f"{away}@{home}: {error}")

    # A partial board is especially dangerous here: it looks usable but silently omits
    # the game whose cache could not be brought onto the current model artifact.  Fail
    # closed so every consumer (optimizer, late swap, simulations, snapshots) gets the
    # same all-or-nothing calibration guarantee.
    if meta["calibration_errors"] or not meta["calibration_consistent"]:
        return pd.DataFrame(), pd.DataFrame(), meta

    # Collapse the repeated identical message into one line.
    if meta["environment_error"] and len(meta["skipped"]) == len(games):
        meta["skipped"] = []
        meta["cached_but_unreadable"] = len(games)

    if not rows:
        return pd.DataFrame(), pd.DataFrame(), meta

    players = pd.DataFrame(rows)
    for column in ("Supports", "Cautions"):
        players[column] = players[column].apply(lambda r: r if isinstance(r, list) else [])

    profiler.note(games=len(games), players=len(players))
    salaries = load_salaries(salary_file) if salary_file else None
    if salaries is not None and not salaries.empty:
        meta["salary_file"] = salary_file
        # Published so every writer downstream files its output under the same slate label.
        # Derived once here rather than per-CLI: two commands disagreeing about what tonight
        # is called would scatter one night's work across two sets of filenames.
        # Labelled against the night's other exports, not alone: a turbo and a main slate
        # share a start-time bucket, and filing both as "main" overwrites one with the other.
        meta["slate_label"] = slate_for(describe_slate(salary_file), requested=slate,
                                        date=date, peers=list_salary_files(date))
        # A salary file from another date matches enough names to look plausible while
        # pricing the wrong slate entirely, so check it before anything else.
        meta["salary_date"] = slate_date(salary_file)
        meta["stale_salaries"] = bool(meta["salary_date"] and meta["salary_date"] != str(date))
        # A priced game with no cached report would drop out of the board silently,
        # taking its whole player pool with it. Surface it instead.
        cached = {f"{canon_team(away)}@{canon_team(home)}" for _, away, home in games}
        meta["missing_games"] = sorted(slate_games(salary_file) - cached)
    players = attach_salaries(players, salaries)

    # Games called off after the export was downloaded are the one slate change a salary
    # file cannot show, so the live schedule is consulted as well as the file itself.
    postponed, schedule_error = ({}, None)
    if check_schedule:
        with profiler.stage("acquire", source="schedule"):
            postponed, schedule_error = postponed_teams(date)
    if salary_file:
        for team in postponed_in_export(salary_file):
            postponed.setdefault(team, "Postponed")
    meta["schedule_error"] = schedule_error
    if postponed:
        called = players["Team"].map(canon_team).isin(postponed)
        meta["postponed"] = {team: postponed[team] for team in sorted(postponed)}
        meta["postponed_players"] = int(called.sum())
        players = players[~called].copy()
        meta["games"] = [g for g in meta["games"]
                         if not ({canon_team(t) for t in g.split("@")} & set(postponed))]

    salary_values = pd.to_numeric(players["Salary"], errors="coerce")
    has_salary = bool(salary_values.notna().sum() >= 10)
    meta["has_salary"] = has_salary
    meta["matched"] = int(salary_values.notna().sum())
    meta["unmatched"] = int(len(players) - salary_values.notna().sum())

    if has_salary:
        # The salary file is the slate definition: anything DK didn't price isn't playable.
        players = players[salary_values.notna()].copy()
        salary_values = pd.to_numeric(players["Salary"], errors="coerce")
        players["Value"] = (pd.to_numeric(players["Proj"]) / (salary_values / 1000)).round(2)
        players["Ceil Value"] = (pd.to_numeric(players["Ceiling"]) / (salary_values / 1000)).round(2)
        players["Floor Value"] = (pd.to_numeric(players["Floor"]) / (salary_values / 1000)).round(2)
    else:
        players["Value"] = None
        players["Ceil Value"] = None
        players["Floor Value"] = None

    # Rank within pitchers and hitters separately -- their point scales don't compare.
    scored = []
    for _, group in players.groupby("Type", sort=False):
        group = group.copy()
        group["Proj Pct"] = _percentile(group["Proj"]).round(1)
        group["Ceiling Pct"] = _percentile(group["Ceiling"]).round(1)

        # DK's season average is the best available read on what the field expects, and
        # it is what the salary was set from. Where our projection ranks a player well
        # above it, tonight's matchup is doing work the price hasn't accounted for.
        if "DK Avg" in group.columns and pd.to_numeric(group["DK Avg"], errors="coerce").notna().sum() >= 5:
            group["Field Pct"] = _percentile(group["DK Avg"]).round(1)
            group["Edge"] = (group["Proj Pct"] - group["Field Pct"]).round(1)
        else:
            group["Field Pct"] = None
            group["Edge"] = None

        if has_salary:
            group["Value Pct"] = _percentile(group["Value"]).round(1)
            group["Ceil Value Pct"] = _percentile(group["Ceil Value"]).round(1)
            group["GPP"] = (
                GPP_WEIGHTS["proj"] * group["Proj Pct"]
                + GPP_WEIGHTS["ceiling_value"] * group["Ceil Value Pct"]
            ).round(1)
            # Safety is the inverse of bust risk, ranked within the group so it is on the
            # same 0-100 scale as the value percentile it is blended with.
            safety_pct = _percentile(-pd.to_numeric(group["Bust%"], errors="coerce"))
            group["Floor Value Pct"] = _percentile(group["Floor Value"]).round(1)
            group["CASH"] = (
                CASH_WEIGHTS["floor_value"] * group["Floor Value Pct"]
                + CASH_WEIGHTS["safety"] * safety_pct
            ).round(1)
        else:
            group["Value Pct"] = None
            group["Ceil Value Pct"] = None
            group["GPP"] = (
                GPP_WEIGHTS_NO_SALARY["proj"] * group["Proj Pct"]
                + GPP_WEIGHTS_NO_SALARY["ceiling"] * group["Ceiling Pct"]
            ).round(1)
            group["Floor Value Pct"] = None
            group["CASH"] = _percentile(group["Floor"]).round(1)
        scored.append(group)

    players = pd.concat(scored, ignore_index=True)

    # Ownership before tiers: the Leverage tier is defined against what the field will do,
    # so it needs the field model to exist first.
    if has_salary:
        with profiler.stage("ownership"):
            players = estimate_ownership(players)
        # A finished slate has measured ownership sitting in dk_results/. Real %Drafted beats
        # the model outright, so it replaces it wherever it exists -- which is what makes a
        # review of a past night honest rather than a critique of our own guess.
        #
        # The night is passed in, not inferred: every export in dk_results/ is a plausible
        # partial match for every other night, and the date is the only thing that rules the
        # neighbours out.
        label = meta.get("slate_label")
        contest, overlap = match_contest(
            players, date=date, slate=None if label == UNKNOWN_SLATE else label)
        if contest is not None:
            players = attach_actual_ownership(players, contest["ownership"])
            meta["ownership_source"] = os.path.basename(contest["path"])
            meta["ownership_overlap"] = round(overlap, 3)
            meta["contest"] = {"entries": contest["entries"],
                               "summary": contest_summary(contest)}
        # Field Pct now means "how heavily will the field roster this player", replacing a
        # percentile of DK's season average. The old version ignored salary entirely, which
        # is the single biggest driver of ownership -- so it measured "better than his season
        # form", not "cheaper to the field than he should be".
        players["Field Pct"] = players["Own Pct"]
        players["Edge"] = (pd.to_numeric(players["Proj Pct"], errors="coerce")
                           - pd.to_numeric(players["Own Pct"], errors="coerce")).round(1)

    players["Tier"] = players.apply(lambda row: _assign_tier(row, has_salary), axis=1)
    players["Role"] = players.apply(lambda row: _assign_role(row, has_salary), axis=1)
    # Supporting factors lead; the single biggest caution rides along so a flagged play
    # never hides its own risk.
    players["Why"] = players.apply(
        lambda row: "; ".join(row["Supports"][:3]
                              + [f"[risk] {c}" for c in row["Cautions"][:1]]) or "neutral matchup",
        axis=1,
    )
    players["Risks"] = players["Cautions"].apply(lambda r: "; ".join(r[:3]))
    players = players.sort_values(["Type", "GPP"], ascending=[True, False]).reset_index(drop=True)

    return players, build_stacks(players, has_salary), meta


MIN_STACK_HITTERS = 4

# Stack ranking: value leads for a GPP lean, raw ceiling keeps it honest.
STACK_WEIGHTS = {"value": 0.55, "ceiling": 0.45}


def build_stacks(players, has_salary, min_hitters=MIN_STACK_HITTERS):
    """Team stacks, ranked on the top of each lineup -- where GPP stacks actually come from.

    Teams with too few priced hitters are dropped: a "stack" assembled from one or two
    matched players carries a meaningless salary and would outrank real stacks on value.
    """
    hitters = players[players["Type"] == "H"]
    if hitters.empty:
        return pd.DataFrame()

    rows = []
    for team, group in hitters.groupby("Team"):
        if len(group) < min_hitters:
            continue
        top = group.sort_values("Slot").head(5)
        salary = pd.to_numeric(top["Salary"], errors="coerce").sum() if has_salary else None
        ceiling = pd.to_numeric(top["Ceiling"], errors="coerce").sum()
        rows.append({
            "Team": team,
            "Opp": top["Opp"].iloc[0],
            "Opp SP": top["Opp SP"].iloc[0],
            "Team Runs": top["Team Runs"].iloc[0],
            "Top5 Proj": round(pd.to_numeric(top["Proj"], errors="coerce").sum(), 1),
            "Top5 Ceiling": round(ceiling, 1),
            "Hitters": len(group),
            "Top5 Salary": int(salary) if has_salary and pd.notna(salary) else None,
            "Stack Value": round(ceiling / (salary / 1000), 2) if has_salary and salary else None,
            "Avg Matchup": round(pd.to_numeric(top["Matchup"], errors="coerce").mean(), 3),
        })

    stacks = pd.DataFrame(rows)
    if has_salary and stacks["Stack Value"].notna().any():
        # Blend rather than sorting on value alone: ceiling per dollar on its own promotes
        # cheap bad offenses, and a stack in a 3-run game is a bad stack at any price.
        stacks["Stack Score"] = (
            STACK_WEIGHTS["value"] * _percentile(stacks["Stack Value"])
            + STACK_WEIGHTS["ceiling"] * _percentile(stacks["Top5 Ceiling"])
        ).round(1)
        sort_column = "Stack Score"
    else:
        stacks["Stack Score"] = _percentile(stacks["Top5 Ceiling"]).round(1)
        sort_column = "Top5 Ceiling"
    return stacks.sort_values(sort_column, ascending=False).reset_index(drop=True)
