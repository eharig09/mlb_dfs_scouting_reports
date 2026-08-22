"""Open a number and see what it was made of.

Every figure on the dashboard is an aggregate of something the pipeline already cached, and
an aggregate you cannot open is a number you have to take on faith. The report writes all of
this into each game payload — the club's relevant games, the arms an arsenal was compared
against, per-batter pitch-level results, rest splits, head-to-head history — and none of it
was reachable from a chart until now.

This module does no analysis. It **locates and shapes** what the pipeline already computed,
so a click can show its provenance. Anything derived here would defeat the purpose: the
drill-down has to show the same numbers the report used, not a second opinion.

Sides
-----
Payload keys are prefixed `away_`/`home_`, and which one a player belongs to is decided by
his club, not by the frame he was found in. `side_for_team` is the single place that mapping
happens; getting it wrong silently shows a hitter the opposing club's evidence, which reads
as plausible and is entirely wrong.
"""

import pandas as pd

from dashboards import data

#: Payload keys that hold a club's own context, per side.
TEAM_KEYS = {
    "relevant games": "relevant_games",
    "recent detail": "recent_detail",
    "rolling form": "rolling_form",
    "schedule context": "schedule_context",
    "trip form": "trip_form",
    "lineup splits": "lineup_splits",
    "transactions": "transactions",
}


def side_for_team(meta, team):
    """Which half of the payload belongs to this club. None when it is in neither."""
    if team == meta.get("away"):
        return "away"
    if team == meta.get("home"):
        return "home"
    return None


def _frame(payload, key):
    """A DataFrame for a payload key, whatever shape the pipeline stored it in.

    Several context entries are tuples of `(notes, frame, caption)` rather than bare frames,
    so a naive `.get` returns a tuple and the caller renders nothing.
    """
    value = (payload.get("advanced_context") or {}).get(key)
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, tuple):
        for item in value:
            if isinstance(item, pd.DataFrame):
                return item
    return pd.DataFrame()


def team_context(payload, meta, team):
    """Every club-level frame for one side, keyed by a readable name.

    This is the "relevant games" popup: the games the report itself judged most like
    tonight, plus the recent and rolling form those judgements were drawn from.
    """
    side = side_for_team(meta, team)
    if side is None:
        return {}
    found = {}
    for label, suffix in TEAM_KEYS.items():
        frame = _frame(payload, f"{side}_{suffix}")
        if not frame.empty:
            found[label] = frame
    comparable = _frame(payload, f"{side}_comparable_games")
    if not comparable.empty:
        found["comparable games"] = comparable
    return found


def _row_for(frame, name, column="Name"):
    """The one row describing a player, as a tidy label/value frame.

    Transposed on purpose: a single row of eighteen columns is unreadable in a dialog, and
    the reader is scanning for one field rather than comparing across players.
    """
    if frame.empty or column not in frame.columns:
        return pd.DataFrame()
    key = _fold(name)
    match = frame[frame[column].map(_fold) == key]
    if match.empty:
        return pd.DataFrame()
    row = match.iloc[0].drop(labels=[column], errors="ignore")
    return pd.DataFrame({"field": row.index.astype(str),
                         "value": [_clean(v) for v in row.to_numpy()]})


def _fold(value):
    from dashboards import salaries
    return salaries.name_key(value)


def _clean(value):
    if isinstance(value, float) and value != value:
        return ""
    return value


def hitter_evidence(payload, meta, name, team):
    """What the report knows about one hitter tonight, in the order a reader wants it.

    Only sections with something in them are returned — an empty panel labelled
    "Head-to-head" reads as "no history", when the truth is usually "this pitcher has never
    faced him", and those are different claims made in different words.
    """
    side = side_for_team(meta, team)
    if side is None:
        return {}
    sections = {}
    composite = _frame(payload, "hitter_composite")
    row = _row_for(composite, name)
    if not row.empty:
        sections["Composite inputs"] = row
    for label, key in (("Arsenal fit", f"{side}_batter_arsenal"),
                       ("Pitch-level results", f"{side}_arsenal_drilldown"),
                       ("Head-to-head", f"{side}_bvp")):
        row = _row_for(_frame(payload, key), name)
        if not row.empty:
            sections[label] = row
    return sections


def opposing_starter(payload, meta, team):
    """The arm this club faces tonight — the *other* side's starter.

    The inversion is the whole point and the easy bug: a hitter's matchup evidence lives
    under the opponent's prefix.
    """
    side = side_for_team(meta, team)
    if side is None:
        return None, {}
    facing = "home" if side == "away" else "away"
    sections = {}
    for label, key in (("Comparable arms", f"{facing}_pitcher_similar"),
                       ("By days rest", f"{facing}_sp_rest_splits"),
                       ("Arsenal", f"{facing}_arsenal_matchup"),
                       ("Hand splits", f"{facing}_pitcher_hand_splits"),
                       ("Quality", f"{facing}_pitcher_quality")):
        frame = _frame(payload, key)
        if not frame.empty:
            sections[label] = frame
    return starter_name(payload, facing), sections


def pitcher_evidence(payload, meta, side):
    """Comparable arms, rest splits and arsenal for one side's starter.

    `Comparable arms` is the list the arsenal comparison was actually built from — the
    answer to "compared against what?", which is unanswerable from the composite alone.
    """
    sections = {}
    for label, key in (("Comparable arms", f"{side}_pitcher_similar"),
                       ("By days rest", f"{side}_sp_rest_splits"),
                       ("Arsenal", f"{side}_arsenal_matchup"),
                       ("Hand splits", f"{side}_pitcher_hand_splits"),
                       ("Recent form", f"{side}_pitcher_quality"),
                       ("Rest schedule", f"{side}_rest_schedule")):
        frame = _frame(payload, key)
        if not frame.empty:
            sections[label] = frame
    return sections


def payload_for(date, team, cache_dir=data.CACHE_DIR):
    """The cached game a club played on a date, with its metadata. `(payload, meta)`."""
    games = data.list_games(cache_dir)
    if games.empty:
        return None, None
    same_day = games[games["date"] == str(date)]
    match = same_day[(same_day["away"] == team) | (same_day["home"] == team)]
    if match.empty:
        return None, None
    meta = match.iloc[0].to_dict()
    return data.load_payload(meta["path"]), meta


def starter_name(payload, side):
    """The starter's name for a side.

    `*_sp_profile` carries his mix, his batted-ball log and his start dates but **not his
    name**; the hand-splits frame is where the pipeline actually records it.
    """
    splits = _frame(payload, f"{side}_pitcher_hand_splits")
    if splits.empty or "Pitcher" not in splits.columns:
        return None
    named = splits["Pitcher"].dropna()
    return str(named.iloc[0]) if not named.empty else None


def starter_game_log(payload, side):
    """A starter's start-by-start log with days of rest between outings.

    `*_sp_profile` stores three views of the same starts — pitch mix, batted-ball results,
    and the dated start list — that the workbook prints as separate tables. Joined on the
    date they become one game log, and the gap between consecutive dates is the days rest
    the user actually wants to sort by.

    Rest is **arithmetic on the pipeline's own start dates**, not a re-derivation of any
    metric: the oldest start has no predecessor in the window and so gets no rest value
    rather than a guess. `SEASON` totals are dropped — a summary row inside a game log
    would be counted as a game by anything that aggregates it.
    """
    profile = (payload.get("advanced_context") or {}).get(f"{side}_sp_profile") or {}
    starts = profile.get("starts") or []
    if not starts:
        return pd.DataFrame()
    log = pd.DataFrame(starts)
    if "date" not in log.columns:
        return pd.DataFrame()
    log["date"] = pd.to_datetime(log["date"], errors="coerce")
    log = log.dropna(subset=["date"]).sort_values("date", ascending=False)
    # Rest is the gap to the *previous* start, so with dates newest-first it is the
    # difference to the next row down. The oldest start's predecessor is outside the
    # window and stays blank.
    log["Rest"] = (log["date"] - log["date"].shift(-1)).dt.days

    for source, columns in (("mix", None), ("batted", None)):
        frame = profile.get(source)
        if not isinstance(frame, pd.DataFrame) or frame.empty or "Date" not in frame:
            continue
        per_start = frame[frame["Date"].astype(str).str.upper() != "SEASON"].copy()
        if per_start.empty:
            continue
        # The profile frames key on `MM-DD` while the start list carries full dates.
        per_start["_key"] = per_start["Date"].astype(str).str.strip()
        log["_key"] = log["date"].dt.strftime("%m-%d")
        drop = [c for c in ("Opp", "Date") if c in per_start.columns]
        log = log.merge(per_start.drop(columns=drop), on="_key", how="left")
        log = log.drop(columns="_key")

    log["Date"] = log["date"].dt.strftime("%Y-%m-%d")
    ordered = ["Date", "opponent", "Rest", "pitches"]
    rest = [c for c in log.columns if c not in ordered + ["date"]]
    return log.reindex(columns=[c for c in ordered + rest if c in log.columns])


def rest_summary(log):
    """The game log collapsed to one row per rest length.

    The report already ships `sp_rest_splits`, but only as season ERA on 5d against 6d+.
    This is the same question asked of the pitch-level log, so a reader can see the starts
    behind a split rather than two aggregate rows. **Kept separate from the report's own
    split** rather than replacing it — they are computed from different sources and a
    disagreement between them is information, not a bug to paper over.
    """
    if log is None or log.empty or "Rest" not in log.columns:
        return pd.DataFrame()
    data = log.dropna(subset=["Rest"]).copy()
    if data.empty:
        return pd.DataFrame()
    data["Rest"] = data["Rest"].astype(int)
    aggregates = {"Starts": ("Rest", "size")}
    for column, label in (("pitches", "Pitches"), ("HH%", "HH%"), ("GB%", "GB%"),
                          ("FB%", "FB%"), ("Avg EV", "Avg EV"), ("Brl%", "Brl%")):
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
            aggregates[label] = (column, "mean")
    return (data.groupby("Rest", as_index=False).agg(**aggregates)
            .sort_values("Rest").reset_index(drop=True))
