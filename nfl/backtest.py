"""What players actually scored, in DK points, from nflverse box scores.

This is the ground truth everything else is measured against: the projection evaluator
scores against it, the post-week review scores against it, and the simulator is calibrated
to it. It is also the first real check on `nfl.scoring` -- a scoring table is easy to write
and easy to get subtly wrong, and the only way to know is to run it over a few thousand real
stat lines and reconcile against an independent implementation.

`reconcile_with_nflverse` is that check. nflverse publishes its own `fantasy_points_ppr`
under standard scoring, which differs from DraftKings in exactly four ways -- interceptions,
lost fumbles, the yardage bonuses, and offensive fumble-recovery touchdowns. If our number
minus theirs equals those four terms for every row, both implementations are right. If it
does not, one of them has a bug and the residual says where.
"""

import numpy as np
import pandas as pd

from .scoring import DK_DST, dst_points, offense_points, points_allowed_points

# nflverse weekly column -> our event key. Anything absent from a season's schema is treated
# as zero rather than raising, because nflverse has renamed fields across eras (the legacy
# `interceptions` became `passing_interceptions`) and a backtest spanning both should not
# die on the boundary.
OFFENSE_COLUMNS = {
    "passing_yards": "PASS_YD",
    "passing_tds": "PASS_TD",
    "passing_interceptions": "INT",
    "rushing_yards": "RUSH_YD",
    "rushing_tds": "RUSH_TD",
    "receptions": "REC",
    "receiving_yards": "REC_YD",
    "receiving_tds": "REC_TD",
    "fumbles_lost_total": "FUM_LOST",
    "special_teams_tds": "RET_TD",
    "fumble_recovery_tds": "FUM_REC_TD",
}
# Two-point conversions are paid the same wherever they came from, so all three columns
# fold into one event.
TWO_POINT_COLUMNS = ("passing_2pt_conversions", "rushing_2pt_conversions",
                     "receiving_2pt_conversions")

# Yardage bonus thresholds, as (event key, source column, yards needed).
BONUS_THRESHOLDS = (("P_PASS_300", "passing_yards", 300),
                    ("P_RUSH_100", "rushing_yards", 100),
                    ("P_REC_100", "receiving_yards", 100))

FANTASY_POSITIONS = ("QB", "RB", "WR", "TE", "FB", "HB")


def _column(frame, name):
    """A numeric column, or zeros when this season's schema does not carry it."""
    if name not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[name], errors="coerce").fillna(0.0)


def offense_actuals(weekly, positions=FANTASY_POSITIONS):
    """DK points per player-week for offensive players.

    Vectorised rather than a row-wise `offense_points` call: a five-season backtest is
    ~90,000 rows and the per-row version takes minutes. The scoring table is still the
    single source of the values -- they are read out of `nfl.scoring`, not retyped -- so a
    rule change lands here automatically.
    """
    if weekly is None or weekly.empty:
        return pd.DataFrame()
    frame = weekly
    if positions and "position" in frame.columns:
        frame = frame[frame["position"].isin(positions)]
    if frame.empty:
        return pd.DataFrame()

    events = pd.DataFrame(index=frame.index)
    for column, key in OFFENSE_COLUMNS.items():
        events[key] = _column(frame, column)
    events["TWO_PT"] = sum(_column(frame, c) for c in TWO_POINT_COLUMNS)
    # Realised bonuses are 1/0, which is the same arithmetic the projection does with a
    # probability -- the whole reason `offense_points` takes them as probabilities.
    for key, column, threshold in BONUS_THRESHOLDS:
        events[key] = (_column(frame, column) >= threshold).astype(float)

    points = pd.Series(0.0, index=frame.index)
    for key in events.columns:
        points += events[key] * _event_value(key)

    out = pd.DataFrame({
        "season": _column(frame, "season").astype(int),
        "week": _column(frame, "week").astype(int),
        "player_id": frame.get("player_id"),
        "name": frame.get("player_display_name", frame.get("player_name")),
        "position": frame.get("position"),
        "team": frame.get("team", frame.get("recent_team")),
        "opponent": frame.get("opponent_team"),
        "game_id": frame.get("game_id"),
        "dk_points": points.round(2),
    })
    return pd.concat([out, events.add_prefix("E_")], axis=1).reset_index(drop=True)


def _event_value(key):
    """Point value for one event key, read out of the scoring tables."""
    from .scoring import DK_OFFENSE, DK_OFFENSE_BONUS
    if key in DK_OFFENSE:
        return DK_OFFENSE[key]
    return DK_OFFENSE_BONUS[key]


# nflverse defensive column -> DST event. `fumble_recovery_opp` rather than `def_fumbles`:
# DK pays for recovering the OPPONENT's fumble, and a defense recovering its own team's
# fumble is not a takeaway.
DST_COLUMNS = {
    "def_sacks": "SACK",
    "def_interceptions": "INT",
    "fumble_recovery_opp": "FUM_REC",
    "def_safeties": "SAFETY",
}
DST_BLOCK_COLUMNS = ("def_fg_blocks", "def_pat_blocks", "def_punt_blocks")
DST_TD_COLUMNS = ("def_tds", "fumble_recovery_tds", "special_teams_tds")


def dst_actuals(weekly, schedules):
    """DK points per team-week for defense/special teams.

    A DST is not a player, so its line is the sum of its defenders' stats plus a points-
    allowed bucket that only the scoreboard knows. Points allowed comes from the schedule
    rather than from summing anything: DK charges the defense for every point the opposing
    side scored, including ones the defense had nothing to do with, and reconstructing that
    from play-by-play is a good way to be quietly wrong about pick-sixes.
    """
    if weekly is None or weekly.empty or schedules is None or schedules.empty:
        return pd.DataFrame()

    team_column = "team" if "team" in weekly.columns else "recent_team"
    grouped = weekly.groupby(["season", "week", team_column], dropna=True)

    rows = {}
    for key, event in DST_COLUMNS.items():
        rows[event] = grouped.apply(lambda g, k=key: _column(g, k).sum(), include_groups=False)
    rows["BLK"] = grouped.apply(
        lambda g: sum(_column(g, c).sum() for c in DST_BLOCK_COLUMNS), include_groups=False)
    rows["TD"] = grouped.apply(
        lambda g: sum(_column(g, c).sum() for c in DST_TD_COLUMNS), include_groups=False)
    events = pd.DataFrame(rows).reset_index()
    events = events.rename(columns={team_column: "team"})

    allowed = _points_allowed(schedules)
    events = events.merge(allowed, on=["season", "week", "team"], how="left")

    counting = sum(events[event] * value for event, value in DK_DST.items()
                   if event in events.columns)
    bucket = events["points_allowed"].map(
        lambda v: points_allowed_points(v) if pd.notna(v) else np.nan)
    events["dk_points"] = (counting + bucket).round(2)
    return events


def _points_allowed(schedules):
    """(season, week, team) -> points the opposing side scored."""
    played = schedules[schedules["home_score"].notna() & schedules["away_score"].notna()]
    home = played[["season", "week", "home_team", "away_score", "game_id"]].rename(
        columns={"home_team": "team", "away_score": "points_allowed"})
    away = played[["season", "week", "away_team", "home_score", "game_id"]].rename(
        columns={"away_team": "team", "home_score": "points_allowed"})
    return pd.concat([home, away], ignore_index=True)


# Columns nflverse's own standard-scoring formula charges a lost fumble for. Notably NOT
# `fumbles_lost_total`, which is the distinction the reconciliation turned up.
NFLVERSE_FUMBLE_COLUMNS = ("rushing_fumbles_lost", "receiving_fumbles_lost",
                           "sack_fumbles_lost")


def reconcile_with_nflverse(actuals, weekly):
    """Our DK points minus nflverse's PPR, against the difference the rules predict.

    Returns a frame with the residual per row. Every value should be zero; anything else is
    a disagreement between two independent implementations of the same box score, and the
    row it lands on says which rule to look at.

    The five ways DraftKings differs from nflverse's standard scoring:

        interceptions       standard -2, DK -1                         -> +1 per
        offensive fumbles   standard -2, DK -1                         -> +1 per
        RETURN fumbles      standard  0, DK -1                         -> -1 per
        yardage bonuses     standard none, DK +3 at 300/100/100        -> +3 per
        fumble recovery TD  absent from nflverse's formula, DK +6      -> +6 per

    The return-fumble line was found by this function rather than written from the rulebook.
    Running it over 2023-24 left 55 rows short by exactly 2.0, every one a returner with a
    lost fumble and no offensive fumble: nflverse only charges rushing, receiving and sack
    fumbles, while DK charges any lost fumble. Our number is the DraftKings one, so the
    difference belongs here rather than in the scoring table -- but a 55-row exception that
    nobody had accounted for is precisely what this reconciliation exists to surface.
    """
    fumbles = weekly[["season", "week", "player_id"]].copy()
    fumbles["offensive_fumbles_lost"] = sum(
        _column(weekly, column) for column in NFLVERSE_FUMBLE_COLUMNS)

    merged = actuals.merge(
        weekly[["season", "week", "player_id", "fantasy_points_ppr"]],
        on=["season", "week", "player_id"], how="inner")
    merged = merged.merge(fumbles, on=["season", "week", "player_id"], how="left")
    merged["offensive_fumbles_lost"] = merged["offensive_fumbles_lost"].fillna(0.0)
    # Whatever DK charged for that nflverse did not: fumbles lost on a return.
    merged["return_fumbles_lost"] = (
        merged["E_FUM_LOST"] - merged["offensive_fumbles_lost"]).clip(lower=0)

    expected = (
        merged["E_INT"] * 1.0
        + merged["offensive_fumbles_lost"] * 1.0
        + merged["return_fumbles_lost"] * -1.0
        + merged["E_P_PASS_300"] * 3.0
        + merged["E_P_RUSH_100"] * 3.0
        + merged["E_P_REC_100"] * 3.0
        + merged["E_FUM_REC_TD"] * 6.0
    )
    merged["expected_delta"] = expected.round(2)
    merged["actual_delta"] = (merged["dk_points"]
                              - pd.to_numeric(merged["fantasy_points_ppr"],
                                              errors="coerce")).round(2)
    merged["residual"] = (merged["actual_delta"] - merged["expected_delta"]).round(2)
    return merged
