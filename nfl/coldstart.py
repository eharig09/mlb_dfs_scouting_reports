"""Week 1-3 projections, when nobody has played yet.

The problem
-----------
`nfl.projections.project_week` builds its player universe from *games already played*:

    recent = history[history["season"] >= season - 1]
    latest = recent.groupby("player_id").tail(1)

and then reads each player's team off his most recent box score. Both are fine in week
ten and wrong in week one. Measured against the 2026 rosters:

* **376 of 915 rostered skill players have no 2025 game at all** -- every rookie, plus
  returners. They cannot appear in a projection built from game history, so 41% of the
  pool is invisible exactly when the board needs it.
* **128 of the 539 who did play are on a different team now** -- 24%. Reading team off the
  last box score puts them on their old club, which means the wrong implied total, the
  wrong opponent, and in a single-game slate the wrong game entirely.

The fix is not a second model
-----------------------------
`_shrink(observed, prior, n, k)` already blends a player's own rate toward a prior by
sample size, and at `n = 0` it returns the prior untouched. So cold start is a question of
*what prior each player gets*, not of new projection machinery:

    week 1   n=0   -> the prior is the whole projection
    week 4   n=3   -> own usage starts to dominate
    week 10        -> the prior is irrelevant, which is correct

A veteran's prior is his own prior-season usage, carried across to whatever team now
rosters him. A rookie's prior comes from draft capital. Everyone else gets replacement
level. The season then washes the prior out on its own, which is what "our own tracking"
buys once games start being played.

Draft capital, fitted rather than assumed
-----------------------------------------
Rookie-season outcomes for 2015-2024 skill players, joined to draft slot:

    Spearman(pick, rookie PPR/game) = -0.589
    round 1 mean 11.2 PPG   vs   rounds 4-7 mean 2.95 PPG   (3.8x)

Two curves are fitted per position, on `log(pick)`, and they are deliberately separate:

* **availability** -- games with any touch, of 17, fitted on every drafted rookie;
* **usage per game played** -- fitted only on rookies with three or more touch-games.

Keeping them apart matters. Fitting usage as season-total/17 folds the two together and
systematically understates a rookie who wins a starting job -- a pick-5 running back reads
as 11.8 carries a game instead of 17.5. Availability belongs in the availability term.

`log(pick)` rather than round means: no cliff at a round boundary, full resolution inside
round one, and no small-sample inversions (the raw round table has WR round 5 above round
4, and TE round 4 above round 3, both noise at n ~ 30).

R-squared runs 0.22-0.52. Draft capital is a real but partial signal, and the numbers say
so; anything claiming more would be fitting noise.
"""

import numpy as np
import pandas as pd

POSITIONS = ("QB", "RB", "WR", "TE")

# Games with any offensive touch, out of 17: a + b*log(pick). Fitted on all drafted skill
# rookies, 2015-2024. R^2: QB 0.52, RB 0.22, WR 0.30, TE 0.41.
AVAILABILITY_FIT = {
    "QB": (14.7623, -2.4471),
    "RB": (24.5261, -3.3919),
    "WR": (23.5795, -3.4794),
    "TE": (29.7085, -4.9273),
}

# Per-game-played usage, conditional on playing: a + b*log(pick). Fitted on rookies with
# three or more touch-games. Metrics a position does not accumulate are simply absent --
# an RB's passing attempts fit to zero slope on an all-zero column, which is not a fit.
USAGE_FIT = {
    ("QB", "attempts"): (33.1855, -2.6699),
    ("QB", "carries"): (4.2951, -0.2864),
    ("RB", "carries"): (22.9008, -3.3610),
    ("RB", "targets"): (6.1077, -0.9107),
    ("WR", "targets"): (9.1316, -1.2123),
    ("WR", "carries"): (0.4084, -0.0469),
    ("TE", "targets"): (8.2474, -1.2119),
}

# Where an undrafted free agent sits on the curve. The 2026 draft ran 257 picks, so this is
# a short extrapolation past the last one rather than a separate model. UDFAs who stick are
# real but rare, and the depth chart is what should rescue the ones who win a job.
UNDRAFTED_PICK = 300

# How many games of real usage it takes for a rookie's own form to outweigh his draft
# prior. Deliberately light: draft slot explains a third of the variance at best, so it
# should yield quickly once actual snaps exist.
ROOKIE_PRIOR_WEIGHT = 3.0

# A player on a roster with neither prior-season usage nor draft capital -- a UDFA in his
# second camp, a practice-squad call-up. Not zero, because he is rostered, but low enough
# that he never leads a board on nothing.
REPLACEMENT_USAGE = {"attempts": 0.0, "carries": 0.6, "targets": 0.8}
REPLACEMENT_GAMES = 4.0

SEASON_GAMES = 17.0


def _curve(fit, pick):
    a, b = fit
    return a + b * float(np.log(max(float(pick), 1.0)))


def expected_games(position, pick):
    """Games with any touch a drafted rookie is expected to see, of 17."""
    fit = AVAILABILITY_FIT.get(position)
    if fit is None:
        return 0.0
    return float(np.clip(_curve(fit, pick), 0.0, SEASON_GAMES))


def rookie_usage(position, pick):
    """Per-game-played opportunity for a rookie at this draft slot."""
    usage = {"attempts": 0.0, "carries": 0.0, "targets": 0.0}
    for metric in usage:
        fit = USAGE_FIT.get((position, metric))
        if fit is not None:
            usage[metric] = max(0.0, _curve(fit, pick))
    return usage


def rookie_prior(position, pick=None):
    """A rookie's prior: what he does per game, and how likely he is to play at all.

    `availability` is carried separately rather than multiplied into usage, so a projection
    can price "probably starts, 17 carries" differently from "17 carries if he ever gets
    on the field". Multiplying them here would make those indistinguishable.
    """
    pick = UNDRAFTED_PICK if pick is None or not np.isfinite(pick) else float(pick)
    games = expected_games(position, pick)
    return {
        "source": "draft",
        "pick": pick,
        "games": games,
        "availability": games / SEASON_GAMES,
        "weight": ROOKIE_PRIOR_WEIGHT,
        **rookie_usage(position, pick),
    }


def replacement_prior(position):
    return {
        "source": "replacement",
        "pick": None,
        "games": REPLACEMENT_GAMES,
        "availability": REPLACEMENT_GAMES / SEASON_GAMES,
        "weight": 1.0,
        **REPLACEMENT_USAGE,
    }


def veteran_prior(own_history, halflife=6.0):
    """A returning player's prior: his own usage, weighted toward his recent games.

    A longer half-life than the in-season model uses. Across an offseason the question is
    "what kind of player is this", not "what did his role look like last month", and a
    four-game half-life would hand the whole prior to whatever happened in week 17.
    """
    if own_history is None or len(own_history) == 0:
        return None
    frame = own_history.sort_values(["season", "week"])
    n = len(frame)
    age = np.arange(n - 1, -1, -1, dtype=float)
    weight = 0.5 ** (age / halflife)
    total = weight.sum()

    def wmean(column):
        values = pd.to_numeric(frame.reindex(columns=[column])[column],
                               errors="coerce").fillna(0.0).to_numpy()
        return float((values * weight).sum() / total)

    return {
        "source": "prior_season",
        "pick": None,
        "games": float(n),
        # Deliberately 1.0, not games-played-over-games-available. A veteran's game count
        # is mostly a statement about his *role* -- a backup appears in ten games because
        # he is a backup -- and role is already carried by the depth-rank factor. Deriving
        # availability from it too demotes the same player twice for the same reason.
        # Genuine availability (injury, suspension) belongs to the injuries feed, which is
        # not wired in here yet.
        "availability": 1.0,
        "weight": float(n),
        "attempts": wmean("attempts"),
        "carries": wmean("carries"),
        "targets": wmean("targets"),
    }


def _latest_depth_chart(depth_charts):
    """The most recent depth-chart snapshot, one row per player-slot.

    The file is a time series -- 449,396 rows for 2026 across snapshots from March to
    August -- so reading it without taking the latest `dt` mixes a player's camp position
    with his current one.
    """
    if depth_charts is None or depth_charts.empty or "dt" not in depth_charts.columns:
        return pd.DataFrame()
    latest = depth_charts[depth_charts["dt"] == depth_charts["dt"].max()]
    return latest


def roster_universe(roster, depth_charts=None, positions=POSITIONS):
    """Who is rosterable, and on which team -- from the roster, not from box scores.

    This is the half of cold start that has nothing to do with projection quality: a player
    who changed teams in the offseason is on his new team here, and a player who has never
    taken a snap is present at all.
    """
    if roster is None or roster.empty:
        return pd.DataFrame()
    frame = roster[roster["position"].isin(positions)].copy()
    if "status" in frame.columns:
        # ACT/RES/etc. Anyone not on the active roster is not a DFS option.
        frame = frame[frame["status"].astype(str).str.upper().str.startswith(("ACT", "A"))]

    out = pd.DataFrame({
        "player_id": frame.get("gsis_id"),
        "player_name": frame.get("full_name"),
        "position": frame.get("position"),
        "team": frame.get("team"),
        "years_exp": pd.to_numeric(frame.reindex(columns=["years_exp"])["years_exp"],
                                   errors="coerce"),
        "draft_number": pd.to_numeric(frame.reindex(columns=["draft_number"])["draft_number"],
                                      errors="coerce"),
        "pff_id": frame.get("pff_id"),
    })
    out = out[out["player_id"].notna()]

    depth = _latest_depth_chart(depth_charts)
    if not depth.empty and "gsis_id" in depth.columns:
        rank = (depth.reindex(columns=["gsis_id", "pos_abb", "pos_rank"])
                .dropna(subset=["gsis_id"])
                .sort_values("pos_rank")
                .groupby("gsis_id").head(1)
                .rename(columns={"gsis_id": "player_id",
                                 "pos_abb": "depth_pos", "pos_rank": "depth_rank"}))
        out = out.merge(rank, on="player_id", how="left")

    return out.reset_index(drop=True)


def build_priors(universe, history, positions=POSITIONS):
    """One prior per rostered player: prior-season usage, draft capital, or replacement.

    `history` is completed games from earlier seasons. A player is a rookie here if he has
    no history at all, whatever his listed experience -- what matters to the projection is
    whether there is anything to read, not what the roster file calls him.
    """
    if universe is None or universe.empty:
        return pd.DataFrame()

    by_player = {}
    if history is not None and not history.empty:
        columns = [c for c in ("player_id", "season", "week", "attempts", "carries",
                               "targets") if c in history.columns]
        trimmed = history.reindex(columns=columns)
        by_player = {pid: group for pid, group in trimmed.groupby("player_id", sort=False)}

    rows = []
    for _, player in universe.iterrows():
        position = player.get("position")
        if position not in positions:
            continue
        own = by_player.get(player["player_id"])
        prior = veteran_prior(own)
        if prior is None:
            prior = _prior_without_history(player, position)
        else:
            prior = apply_depth_rank(prior, position, player.get("depth_rank"))
        rows.append({
            "player_id": player["player_id"],
            "player_name": player.get("player_name"),
            "position": position,
            "team": player.get("team"),
            "depth_rank": player.get("depth_rank"),
            **prior,
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    # Usage on this frame is *conditional on playing*. Ranking on it directly puts an
    # undrafted fourth-string quarterback among the starters, because if he played he would
    # throw. The expected_* columns fold availability back in and are what a board should
    # sort on; the conditional numbers stay for reading a role.
    for metric in ("attempts", "carries", "targets"):
        frame[f"expected_{metric}"] = frame[metric] * frame["availability"]
    frame["expected_opportunity"] = (frame["expected_attempts"]
                                     + frame["expected_carries"]
                                     + frame["expected_targets"])
    return frame.sort_values("expected_opportunity", ascending=False).reset_index(drop=True)


# Opportunity multiplier by depth-chart rank, relative to the rank-1 player at the same
# position. Measured over 10,897 player-weeks, 2023-24 regular season, on weeks where the
# player recorded a stat line -- so this is "how much less does the backup get", not
# "does he dress".
#
#     QB  35.7 opp/g at rank 1  vs 17.0 at rank 2   (2.10x)
#     RB  13.3                     7.9              (1.69x)
#     WR   5.4                     2.9              (1.87x)
#     TE   4.6                     2.4              (1.94x)
#
# Rank 3+ collapses further and thins out fast, so it shares one bucket. The QB rank-3
# cell in the raw fit reads 27.1 opp/g on n=40 -- backups who ended up starting, mislabeled
# by a stale chart -- and is deliberately not used.
DEPTH_MULTIPLIER = {
    "QB": {1: 1.00, 2: 0.48, 3: 0.30},
    "RB": {1: 1.00, 2: 0.59, 3: 0.35},
    "WR": {1: 1.00, 2: 0.54, 3: 0.41},
    "TE": {1: 1.00, 2: 0.52, 3: 0.37},
}


def depth_multiplier(position, rank):
    """How much of the rank-1 workload a player at this depth rank tends to see."""
    table = DEPTH_MULTIPLIER.get(position)
    if table is None or rank is None or not np.isfinite(rank):
        return 1.0
    return table.get(int(min(max(rank, 1), 3)), table[3])


def apply_depth_rank(prior, position, rank):
    """Re-scale a *veteran's* prior usage for the role he is listed in now.

    **Why only veterans.** A returning player's prior is what he did last year, which
    encodes the role he held last year. When those disagree the depth chart is the newer
    fact: Aidan O'Connell threw 27.6 times a game in the starts he made, and is listed QB3
    now, so projecting him at a starter's volume in week one is reading a stale role.

    A rookie's draft-capital prior is deliberately left alone. That curve is already an
    average *across* roles for players taken at that pick, so multiplying it by a rank
    factor would count the same information twice. Composing them honestly needs an
    expected-rank-given-pick term, which is not fitted here -- so it is not done here.
    """
    if prior is None or prior.get("source") != "prior_season":
        return prior
    factor = depth_multiplier(position, rank)
    if factor == 1.0:
        return prior
    adjusted = dict(prior)
    for metric in ("attempts", "carries", "targets"):
        adjusted[metric] = prior.get(metric, 0.0) * factor
    adjusted["depth_factor"] = factor
    return adjusted


def _prior_without_history(player, position):
    """The prior for a rostered player with nothing to read.

    **Draft capital is only a rookie prior, and the gate is load-bearing.** `draft_number`
    is on the roster row for the rest of a player's career -- 452 of the 915 rostered 2026
    skill players are veterans carrying one. Reading it without an experience check prices
    an eleven-year veteran off the pick he was taken at in 2015: Phillip Dorsett, pick 29,
    came out of the first build as a rookie WR1 on 5.0 targets a game.

    A player past his rookie year with no recorded usage is not a neutral unknown either.
    A second-year quarterback who has never taken a snap is evidence *against* the draft
    slot, not absent evidence for it, so he drops to replacement rather than keeping the
    prior he failed to earn.
    """
    experience = pd.to_numeric(pd.Series([player.get("years_exp")]),
                               errors="coerce").iloc[0]
    is_rookie = pd.notna(experience) and float(experience) == 0.0
    if not is_rookie or position not in AVAILABILITY_FIT:
        return replacement_prior(position)
    pick = player.get("draft_number")
    return rookie_prior(position, pick if pd.notna(pick) else UNDRAFTED_PICK)


def fit_draft_curves(history, draft_picks, seasons, min_touch_games=3):
    """Refit `AVAILABILITY_FIT` and `USAGE_FIT`. Prints a paste-ready block.

    Hold out the season you intend to score on, the same rule the main model follows.
    """
    frames = []
    for season, weekly in history.items() if isinstance(history, dict) else [(None, history)]:
        frame = weekly[weekly["position"].isin(POSITIONS)].copy()
        if "season_type" in frame.columns:
            frame = frame[frame["season_type"] == "REG"]
        frame["touched"] = frame.reindex(
            columns=["attempts", "carries", "targets"]).fillna(0).sum(axis=1) > 0
        grouped = frame.groupby(["player_id", "position", "season"]).agg(
            games=("touched", "sum"), attempts=("attempts", "sum"),
            carries=("carries", "sum"), targets=("targets", "sum")).reset_index()
        frames.append(grouped)
    produced = pd.concat(frames, ignore_index=True)

    rookies = draft_picks[draft_picks["season"].isin(seasons)
                          & draft_picks["gsis_id"].notna()
                          & draft_picks["position"].isin(POSITIONS)]
    joined = rookies[["season", "gsis_id", "position", "pick"]].merge(
        produced.drop(columns=["position"]), left_on=["gsis_id", "season"],
        right_on=["player_id", "season"], how="left")
    for column in ("games", "attempts", "carries", "targets"):
        joined[column] = joined[column].fillna(0.0)
    joined["logpick"] = np.log(joined["pick"].clip(lower=1))

    availability, usage = {}, {}
    for position in POSITIONS:
        sub = joined[joined["position"] == position]
        if len(sub) < 20:
            continue
        b, a = np.polyfit(sub["logpick"], sub["games"], 1)
        availability[position] = (round(float(a), 4), round(float(b), 4))
        played = sub[sub["games"] >= min_touch_games]
        for metric in ("attempts", "carries", "targets"):
            rate = (played[metric] / played["games"]).to_numpy()
            if not len(rate) or rate.max() <= 0.05:
                continue                    # position does not accumulate this metric
            b, a = np.polyfit(played["logpick"].to_numpy(), rate, 1)
            usage[(position, metric)] = (round(float(a), 4), round(float(b), 4))

    print("AVAILABILITY_FIT = {")
    for key, value in availability.items():
        print(f'    "{key}": {value},')
    print("}\n\nUSAGE_FIT = {")
    for key, value in usage.items():
        print(f'    ("{key[0]}", "{key[1]}"): {value},')
    print("}")
    return availability, usage
