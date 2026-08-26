"""Player deployment profiles: where a player lines up, how deep he is thrown to, and how
much high-value work he gets.

This module exists because of one measured result. Every candidate signal in the PFF drop
was scored on year-over-year self-correlation, and they split cleanly into two groups:

    deployment          inline rate 0.95, wide rate 0.92, aDOT 0.89, route rate 0.88,
                        slot rate 0.82, red-zone carries/game 0.85, RZ targets/game 0.80
    results             YPC 0.27, yards after contact 0.28, man-vs-zone YPRR gap 0.19,
                        and on defense: slot YPT allowed 0.10, man YPT allowed 0.05

**So this file reports deployment and volume, and deliberately does not build ratings out of
allowed-rates.** A receiver's alignment next season is close to knowable; how many yards a
defense will give up in the slot is not, and a matchup edge built on the latter is fitting
last year's variance. See `docs/nfl/pff_integration.md` for the full table.

    from nfl.usage import receiver_profile, high_value_touches, team_target_distribution
    receiver_profile(2025)          # one row per receiver
    receiver_profile([2025, 2024])  # recency-weighted blend across seasons

Every profile is keyed on `gsis_id` and reports **per-game or per-route rates**, never a
share of a moving denominator -- see `high_value_touches` for what that costs.
"""

import numpy as np
import pandas as pd

from nfl import pffdata
from nfl import redzone as rz
from nfl.salaries import canon_position, canon_team

# Weight on each season back from the most recent, for a multi-season blend. Deployment is
# stable enough that older seasons carry real information, but a player who moved into a
# starting role should not be held back by two years of being a backup. Not fitted: chosen
# so the current season carries a clear majority and the third year is a tie-break.
RECENCY_WEIGHTS = (1.0, 0.45, 0.20)

# Minimum routes before a receiver's *rates* are reported. Below this the denominator is
# small enough that a slot rate is describing a handful of snaps.
MIN_ROUTES = 50
# Minimum carries before a rusher's rates are reported.
MIN_CARRIES = 25


def _num(frame, column):
    """A numeric Series for `column`, even when the column is absent.

    `frame.get(col)` returns a **scalar NaN** for a missing column -- no index, no `.fillna`
    -- which is the recurring crash in this codebase. Never reach for `.get` on a frame whose
    schema depends on which export happened to be present.
    """
    return pd.to_numeric(frame.reindex(columns=[column])[column], errors="coerce")


def _rate(numerator, denominator):
    """Elementwise ratio with a zero denominator going to NaN rather than inf.

    A blank means "we cannot say", which is the honest reading of 0/0 and is what the report
    layer prints as an empty cell. An inf would sort to the top of any leaderboard.
    """
    return numerator / denominator.replace(0, np.nan)


def receiver_profile(seasons, min_routes=MIN_ROUTES):
    """Alignment, depth and target rate for every qualifying receiver.

    Columns, all of which self-correlate above +0.6 year to year:

        routes_pg route_rate tprr        volume and participation
        slot_rate wide_rate inline_rate  alignment
        adot deep_tgt_rate               how far downfield he is used
        yprr grade_route                 efficiency, the weakest of these at ~0.6

    `deep_tgt_rate` is the share of a receiver's *routes* that draw a 20+ yard target, taken
    from the `receiving_depth` export -- a rate per route, not per target, so that a player
    who is rarely thrown to at all cannot post a high number off three deep shots.
    """
    seasons = _as_seasons(seasons)
    frames = [_receiver_season(s, min_routes) for s in seasons]
    return _blend(frames, seasons)


def _receiver_season(season, min_routes):
    summary = pffdata.load("receiving_summary", season)
    out = pd.DataFrame({
        "gsis_id": summary["gsis_id"],
        "player": summary["player"],
        "team": summary["team"],
        "pos": summary["pos"],
        "games": _num(summary, "player_game_count"),
        "routes": _num(summary, "routes"),
        "targets": _num(summary, "targets"),
        "route_rate": _num(summary, "route_rate"),
        "slot_rate": _num(summary, "slot_rate"),
        "wide_rate": _num(summary, "wide_rate"),
        "inline_rate": _num(summary, "inline_rate"),
        "adot": _num(summary, "avg_depth_of_target"),
        "yprr": _num(summary, "yprr"),
        "grade_route": _num(summary, "grades_pass_route"),
        "yac_per_rec": _num(summary, "yards_after_catch_per_reception"),
    })
    out = out[out["gsis_id"].notna() & (out["routes"] >= min_routes)]
    out["routes_pg"] = _rate(out["routes"], out["games"])
    out["tprr"] = _rate(out["targets"], out["routes"])

    depth = _depth_season(season)
    out = out.merge(depth, on="gsis_id", how="left")
    return out.set_index("gsis_id")


def _depth_season(season):
    """Deep and intermediate usage per route, from the `receiving_depth` export.

    Falls back to an empty frame when the season has no depth export -- it only goes back to
    2023, while the summaries reach 2018, and a missing column must read as absence rather
    than stop a multi-season blend.
    """
    try:
        depth = pffdata.load("receiving_depth", season)
    except pffdata.PffDataError:
        return pd.DataFrame(columns=["gsis_id", "deep_tgt_rate", "deep_yprr",
                                     "medium_tgt_rate", "behind_los_tgt_rate"])
    routes = _num(depth, "deep_routes").add(_num(depth, "medium_routes"), fill_value=0)
    routes = routes.add(_num(depth, "short_routes"), fill_value=0)
    routes = routes.add(_num(depth, "behind_los_routes"), fill_value=0)
    out = pd.DataFrame({
        "gsis_id": depth["gsis_id"],
        "deep_tgt_rate": _rate(_num(depth, "deep_targets"), routes),
        "deep_yprr": _num(depth, "deep_yprr"),
        "medium_tgt_rate": _rate(_num(depth, "medium_targets"), routes),
        "behind_los_tgt_rate": _rate(_num(depth, "behind_los_targets"), routes),
    })
    return out[out["gsis_id"].notna()].drop_duplicates("gsis_id")


def rusher_profile(seasons, min_carries=MIN_CARRIES):
    """Carry and target volume for every qualifying back, plus the run-scheme split.

    `gap_share` (r = +0.54) is a *scheme* fact -- how much of his work comes on gap concepts
    rather than zone -- and is the one efficiency-adjacent column kept here. Yards per carry
    (r = +0.27) and yards after contact (r = +0.28) are deliberately reported but should be
    treated as descriptive; neither forecasts itself well enough to drive a projection.
    """
    seasons = _as_seasons(seasons)
    frames = [_rusher_season(s, min_carries) for s in seasons]
    return _blend(frames, seasons)


def _rusher_season(season, min_carries):
    rushing = pffdata.load("rushing_summary", season)
    gap = _num(rushing, "gap_attempts")
    zone = _num(rushing, "zone_attempts")
    out = pd.DataFrame({
        "gsis_id": rushing["gsis_id"],
        "player": rushing["player"],
        "team": rushing["team"],
        "pos": rushing["pos"],
        "games": _num(rushing, "player_game_count"),
        "carries": _num(rushing, "attempts"),
        "targets": _num(rushing, "targets"),
        "routes": _num(rushing, "routes"),
        "gap_share": _rate(gap, gap.add(zone, fill_value=0)),
        "elusive": _num(rushing, "elusive_rating"),
        "breakaway_pct": _num(rushing, "breakaway_percent"),
        "grade_run": _num(rushing, "grades_run"),
        "ypc": _num(rushing, "ypa"),
        "yco_per_att": _num(rushing, "yco_attempt"),
    })
    out = out[out["gsis_id"].notna() & (out["carries"] >= min_carries)]
    out["carries_pg"] = _rate(out["carries"], out["games"])
    out["targets_pg"] = _rate(out["targets"], out["games"])
    out["routes_pg"] = _rate(out["routes"], out["games"])
    return out.set_index("gsis_id")


def high_value_touches(seasons, min_games=4):
    """Red-zone and inside-5 work, **per game**.

    The unit is the whole point. Measured year over year:

        red-zone carries / game     +0.85
        red-zone targets / game     +0.80
        inside-5 carries / game     +0.73
        inside-5 carries / carry    +0.33   <- the same fact as a share

    Expressing high-value work as a *share of a back's carries* throws away most of the
    signal, because the share moves with whatever happened to the denominator: a back whose
    early-down role shrank looks like he gained goal-line equity. So every column here is a
    per-game volume, and a caller wanting a share should divide two of them itself and know
    that it is doing so.

    Keyed on `nfl.redzone.name_key`, not `gsis_id`: this comes off PFF's fantasy-stats
    endpoint, which carries no player id at all.
    """
    seasons = _as_seasons(seasons)
    frames = [_high_value_season(s, min_games) for s in seasons]
    return _blend(frames, seasons, key="key")


def _high_value_season(season, min_games):
    frame = pd.read_csv(rz.pff_path("fantasy_receiving", season))
    games = _num(frame, "games")
    out = pd.DataFrame({
        "key": frame["player"].map(rz.name_key),
        "player": frame["player"],
        # PFF's fantasy-stats endpoint writes raw PFF codes -- ARZ, LA, HST -- while every
        # other profile here is already canonical. Unfolded, those clubs quietly vanish from
        # any join on team and nothing raises.
        "team": frame.reindex(columns=["team"])["team"].map(canon_team),
        "pos": frame.reindex(columns=["position"])["position"].map(canon_position),
        "games": games,
        "rz_carries_pg": _rate(_num(frame, "rzRushCarries"), games),
        "i5_carries_pg": _rate(_num(frame, "i5RushCarries"), games),
        "rz_targets_pg": _rate(_num(frame, "rzRecTarg"), games),
        "ez_targets_pg": _rate(_num(frame, "ezRecTarg"), games),
        "rz_rush_td_pg": _rate(_num(frame, "rzRushTds"), games),
        "carries_pg": _rate(_num(frame, "rushCarries"), games),
        "targets_pg": _rate(_num(frame, "recTarg"), games),
    })
    out = out[(out["games"] >= min_games) & out["key"].notna()]
    # High-value touches: goal-line carries and red-zone targets are the two ways a player
    # gets scoring equity that his yardage volume does not already describe.
    out["hv_touches_pg"] = out["i5_carries_pg"].fillna(0) + out["rz_targets_pg"].fillna(0)
    return out.set_index("key")


def team_target_distribution(seasons, min_routes=MIN_ROUTES):
    """How a team's targets divide, by player and by position.

    Returns one row per player with `target_share` (of his team's targets among qualifying
    receivers) and `route_share`, plus the team's position mix attached as `wr_target_share`,
    `te_target_share` and `rb_target_share`.

    **The denominator is qualifying receivers, not the whole team**, so shares sum to 1.0
    within the frame and a club's deep reserves do not dilute the top of it. That makes this
    a comparison *between* the players who actually play, which is the question a stack asks.
    """
    profile = receiver_profile(seasons, min_routes=min_routes).reset_index()
    profile = profile[profile["team"].notna()]

    totals = profile.groupby("team")[["targets", "routes"]].sum()
    profile = profile.merge(totals, on="team", how="left", suffixes=("", "_team"))
    profile["target_share"] = _rate(profile["targets"], profile["targets_team"])
    profile["route_share"] = _rate(profile["routes"], profile["routes_team"])

    by_pos = (profile.assign(_p=profile["pos"].str.upper())
              .pivot_table(index="team", columns="_p", values="target_share", aggfunc="sum"))
    for position in ("WR", "TE", "RB"):
        column = f"{position.lower()}_target_share"
        profile[column] = profile["team"].map(by_pos.reindex(columns=[position])[position])
    return profile.sort_values(["team", "target_share"], ascending=[True, False])


# --- multi-season blending ------------------------------------------------------------------


def _as_seasons(seasons):
    """Accept a single season or an iterable, newest first."""
    if isinstance(seasons, (int, np.integer)):
        return [int(seasons)]
    return [int(s) for s in seasons]


def _blend(frames, seasons, key=None, weights=RECENCY_WEIGHTS):
    """Recency-weighted average of one profile across seasons.

    A single season passes through untouched. With several, numeric columns are averaged
    with `RECENCY_WEIGHTS` **over the seasons that actually have the player** -- the weights
    are renormalised per player, so a rookie with one season is not shrunk toward zero by
    two seasons of absence. Identity columns (name, team, position) come from the most
    recent season that has him, since that is where he plays now.
    """
    if len(frames) == 1:
        return frames[0]

    identity = ["player", "team", "pos"]
    numeric_columns = sorted({c for f in frames for c in f.columns} - set(identity))

    weighted, weight_total, ident = {}, {}, None
    for frame, weight in zip(frames, list(weights) + [0.0] * len(frames)):
        if weight <= 0:
            continue
        numbers = frame.reindex(columns=numeric_columns).apply(pd.to_numeric, errors="coerce")
        present = numbers.notna()
        for column in numeric_columns:
            contribution = numbers[column].fillna(0) * weight
            weighted[column] = contribution if column not in weighted else \
                weighted[column].add(contribution, fill_value=0)
            mass = present[column].astype(float) * weight
            weight_total[column] = mass if column not in weight_total else \
                weight_total[column].add(mass, fill_value=0)
        available = frame.reindex(columns=identity)
        ident = available if ident is None else ident.combine_first(available)

    out = pd.DataFrame({c: weighted[c] / weight_total[c].replace(0, np.nan)
                        for c in numeric_columns})
    out = ident.join(out, how="right") if ident is not None else out
    out.index.name = key or "gsis_id"
    return out
