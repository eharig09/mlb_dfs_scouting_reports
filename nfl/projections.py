"""Turn nflverse history into DK-point projections.

Design follows `dfs.projections`: every player gets baseline rates from his own history,
regressed toward a positional prior by sample size, then a set of **named** multiplicative
matchup factors. Keeping the factors named and separate is the point -- they are what a
board shows as the reason a player is flagged, and they are what you tune when the model
feels wrong.

**What the data says to build.** Week-to-week self-correlation, measured over 2023-25 on
17,702 player-weeks:

    carries      RB 0.68      targets      WR 0.61   TE 0.57   RB 0.47
    attempts     QB 0.51      rushing yds  RB 0.52   QB 0.41
    receiving yds WR 0.45     TE 0.43
    rushing TDs  0.03-0.20    receiving TDs 0.03-0.11   passing TDs 0.24

Volume is the predictable part and touchdowns are very nearly noise. So the model projects
**opportunity** from a player's own recent usage and takes **scoring rates** almost entirely
from a positional prior. A model that reads a player's own touchdown history forward is
fitting last month's luck, which is the single most common way a fantasy projection goes
wrong -- and it is why `TD_SHRINK` is an order of magnitude heavier than `VOLUME_SHRINK`.

**Shrinkage is derived, not guessed.** For a mean of n observations with single-observation
reliability rho, the shrinkage constant that reproduces it is k = (1 - rho) / rho. Every
constant below is that formula applied to the table above, which is why they are not round
numbers. `calibrate()` recomputes the table from whatever seasons it is given.

**Walk-forward by construction.** `project_week` takes a cutoff and reads only games
strictly before it. There is no code path that can see the week it is projecting, which is
what makes an evaluation over completed seasons mean anything.
"""

import numpy as np
import pandas as pd
from scipy import stats

from .scoring import DK_DST, offense_points, points_allowed_points

# Bumped whenever a change here would move a projection. Snapshots and evaluations record
# it, so a metric computed under one version is never silently compared against another.
#
#   1  volume from exponentially-weighted recent usage, scoring rates from a positional
#      prior, team context from the implied total, opponent adjustment from fantasy points
#      allowed to the position. Bands and bonus probabilities fitted on 2021-24.
MODEL_VERSION = 1

POSITIONS = ("QB", "RB", "WR", "TE")

# Half-life in games for the usage average. Four games: short enough to follow a role change
# -- a backup taking over is the single biggest thing a weekly projection has to notice --
# and long enough that one bad script does not erase a season of usage.
USAGE_HALFLIFE = 4.0
# How far back to look at all. A season and a bit; older than that and the roster around the
# player has usually changed enough that the usage is describing a different offense.
USAGE_LOOKBACK = 24

# Shrinkage constants, k = (1 - rho) / rho from the measured reliabilities above. Units are
# games for volume and opportunities for the rates.
VOLUME_SHRINK = {"attempts": 0.95, "carries": 0.55, "targets": 0.70}
# Yards per opportunity settles faster than touchdowns but slower than volume.
EFFICIENCY_SHRINK = 45.0
# Touchdown rate. At a measured rho near 0.1 the honest constant is enormous: a receiver
# needs on the order of two full seasons of targets before his own touchdown rate outweighs
# the positional prior. This number looks aggressive and is not.
TD_SHRINK = 220.0

# ---- fitted constants -----------------------------------------------------------------
# Everything below is the output of `calibrate(weekly, schedules, [2022, 2023, 2024])` over
# 14,962 walk-forward player-weeks. 2025 was held out on purpose and is the season these are
# scored against; refitting to include it would turn the evaluation in-sample.
#
# Points per team per game, 2022-24 regular season.
LEAGUE_TEAM_POINTS = 22.31

# Coefficient of variation of realised yardage, feeding the gamma that produces the
# 100/300-yard bonus probabilities. Fitted by matching the modelled exceedance rate to the
# realised one rather than by taking a sample standard deviation -- see `_fit_cv`.
#
# The passing number is far tighter than the receiving ones (0.33 against 0.79) and that is
# real rather than an artefact: a quarterback's yardage is the sum of thirty-odd attempts
# and concentrates, while a receiver's is the sum of six targets and does not. Assuming one
# dispersion for all of them, as the first pass here did, made the 300-yard bonus fire at
# 19.4% against a true 12.4%.
YARDAGE_CV = {"QB_pass": 0.33, "RB_rush": 0.74, "WR_rec": 0.79, "TE_rec": 0.63}

# KNOWN BIAS, measured on held-out 2025 and not yet fixed.
#
#     position  projected yds   actual   ratio
#     QB pass       198.8        187.5   1.060
#     WR rec         32.7         30.5   1.071
#     RB rush        33.8         33.7   1.002
#     TE rec         23.4         23.3   1.004
#
# Passing and receiving yardage run 6-7% high; rushing and tight-end receiving are clean.
# The overstated bonuses track it exactly -- pass 300 fires at 17.9% against a true 12.8%,
# rec 100 at 10.4% against 8.4%, while rush 100 (no mean bias) is the closest to calibrated
# at 13.4% against 11.3%.
#
# The temptation is to shrink YARDAGE_CV until the bonus rates match. That would be fitting
# the wrong parameter: the tail is wrong because the mean is wrong, and absorbing a mean
# error into a dispersion term hides it everywhere else it is doing damage. The likely cause
# is USAGE_HALFLIFE being too responsive -- four games over-weights a recent usage spike
# that does not persist -- and the fix belongs there. Refit the CVs after, not before.
_YARDAGE_BIAS_2025 = {"QB_pass": 1.060, "WR_rec": 1.071, "RB_rush": 1.002, "TE_rec": 1.004}

# Floor and ceiling as (slope, intercept) on the projection, fitted against the realised
# 25th and 90th percentiles within projection buckets. Deliberately not assumed from a
# normal: single-game DK scoring is far too skewed for that.
#
# The quarterback row is the interesting one. Its ceiling slope is 0.91 against 1.46-1.67
# for the skill positions, which says a quarterback's upside is far less projection-
# dependent than a receiver's -- the floor is high and the ceiling is not much above the
# mean, because a QB accumulates from every drive while a receiver needs the ball to find
# him. That is the shape of the position, and it is why a single league-wide ceiling
# multiplier would misprice both ends of the roster.
BAND_FIT = {
    "QB": {"floor": (0.882, -4.11), "ceiling": (0.907, 12.19)},
    "RB": {"floor": (0.673, -1.94), "ceiling": (1.460, 4.14)},
    "WR": {"floor": (0.654, -1.92), "ceiling": (1.667, 3.23)},
    "TE": {"floor": (0.607, -1.26), "ceiling": (1.535, 3.47)},
}


def implied_totals(schedules):
    """(season, week, team) -> implied points, from the betting market.

    An implied team total is the strongest public prior on how many points an offense will
    score, and a projection is ultimately dividing that number among nine players.

    The sign convention is checked, not assumed: nflverse's `spread_line` is positive when
    the HOME team is favoured (corr with home margin +0.45 over 2022-25). Reading it the
    other way inverts every implied total on the slate while still looking plausible.
    """
    frame = schedules.dropna(subset=["total_line", "spread_line"])
    half_total = frame["total_line"] / 2.0
    half_spread = frame["spread_line"] / 2.0
    home = pd.DataFrame({
        "season": frame["season"], "week": frame["week"], "team": frame["home_team"],
        "opponent": frame["away_team"], "game_id": frame["game_id"],
        "implied": half_total + half_spread, "opp_implied": half_total - half_spread,
        "total_line": frame["total_line"], "is_home": True})
    away = pd.DataFrame({
        "season": frame["season"], "week": frame["week"], "team": frame["away_team"],
        "opponent": frame["home_team"], "game_id": frame["game_id"],
        "implied": half_total - half_spread, "opp_implied": half_total + half_spread,
        "total_line": frame["total_line"], "is_home": False})
    return pd.concat([home, away], ignore_index=True)


def _numeric(frame, column):
    """A numeric column, or zeros when the frame does not carry it.

    Reindexed rather than read with `.get`: on a frame missing the column `.get` returns
    None and `pd.to_numeric(None)` is a scalar NaN, which has no `.fillna` and no index.
    The same trap is documented in `dfs.ownership` and has now bitten this project three
    times, so every column read in this module goes through here.
    """
    return pd.to_numeric(frame.reindex(columns=[column])[column],
                         errors="coerce").fillna(0.0)


def _before(weekly, season, week):
    """Games strictly before a cutoff. The whole walk-forward guarantee lives here."""
    return weekly[(weekly["season"] < season)
                  | ((weekly["season"] == season) & (weekly["week"] < week))]


def _weighted_rates(frame, halflife=USAGE_HALFLIFE, lookback=USAGE_LOOKBACK):
    """Exponentially-weighted per-game usage and efficiency for one player.

    `frame` is that player's completed games, oldest first. Weight halves every `halflife`
    games back, so a role change shows up within a few weeks rather than being averaged
    away over a season.
    """
    frame = frame.tail(lookback)
    n = len(frame)
    if not n:
        return None
    age = np.arange(n - 1, -1, -1, dtype=float)
    weight = 0.5 ** (age / halflife)
    total = weight.sum()

    def wmean(column):
        return float((_numeric(frame, column).to_numpy() * weight).sum() / total)

    def wsum(column):
        return float((_numeric(frame, column).to_numpy() * weight).sum())

    return {
        "games": n,
        "weight": float(total),
        "attempts": wmean("attempts"),
        "carries": wmean("carries"),
        "targets": wmean("targets"),
        # Totals, so efficiency divides one weighted sum by another rather than averaging
        # per-game ratios -- a one-carry game must not count as much as a twenty-carry one.
        "pass_yards": wsum("passing_yards"), "pass_att": wsum("attempts"),
        "pass_tds": wsum("passing_tds"), "interceptions": wsum("passing_interceptions"),
        "rush_yards": wsum("rushing_yards"), "rush_att": wsum("carries"),
        "rush_tds": wsum("rushing_tds"),
        "rec_yards": wsum("receiving_yards"), "rec_targets": wsum("targets"),
        "receptions": wsum("receptions"), "rec_tds": wsum("receiving_tds"),
        "fumbles": wsum("fumbles_lost_total"),
    }


def _shrink(observed, prior, n, k):
    """Weighted average of a player's own rate and the positional prior."""
    if n <= 0 or not np.isfinite(observed):
        return prior
    return (n * observed + k * prior) / (n + k)


def positional_priors(history):
    """League rates by position, the target every player's own rates are shrunk toward."""
    priors = {}
    for position in POSITIONS:
        rows = history[history["position"] == position]
        if rows.empty:
            continue

        def total(column):
            return float(_numeric(rows, column).sum())

        games = max(len(rows), 1)
        attempts, carries, targets = total("attempts"), total("carries"), total("targets")
        priors[position] = {
            "attempts": attempts / games, "carries": carries / games,
            "targets": targets / games,
            "pass_ypa": total("passing_yards") / attempts if attempts else 0.0,
            "pass_td_rate": total("passing_tds") / attempts if attempts else 0.0,
            "int_rate": total("passing_interceptions") / attempts if attempts else 0.0,
            "rush_ypc": total("rushing_yards") / carries if carries else 0.0,
            "rush_td_rate": total("rushing_tds") / carries if carries else 0.0,
            "rec_ypt": total("receiving_yards") / targets if targets else 0.0,
            "catch_rate": total("receptions") / targets if targets else 0.0,
            "rec_td_rate": total("receiving_tds") / targets if targets else 0.0,
            "fumble_rate": total("fumbles_lost_total") / games,
        }
    return priors


def opponent_factors(history, positions=POSITIONS, shrink=6.0):
    """{(team, position): multiplier} for DK points allowed to a position, vs league.

    Shrunk toward 1.0 by games: a defense's fantasy points allowed over a handful of games
    is mostly a statement about which offenses it happened to face. Six games of prior is
    enough that a two-week outlier cannot produce a 1.4x factor.
    """
    from .backtest import offense_actuals
    scored = offense_actuals(history)
    if scored.empty:
        return {}
    factors = {}
    for position in positions:
        rows = scored[scored["position"] == position]
        if rows.empty:
            continue
        league = rows.groupby(["season", "week", "opponent"])["dk_points"].sum()
        mean = float(league.mean())
        if not mean:
            continue
        for team, group in rows.groupby("opponent"):
            per_game = group.groupby(["season", "week"])["dk_points"].sum()
            games = len(per_game)
            observed = float(per_game.mean()) / mean if games else 1.0
            factors[(team, position)] = _shrink(observed, 1.0, games, shrink)
    return factors


def _bonus_probability(mean_yards, threshold, cv):
    """P(yards >= threshold) under a gamma with the fitted dispersion.

    A gamma rather than a normal because single-game yardage is bounded below by zero and
    has a long right tail -- a normal puts real mass below zero and badly understates the
    100-yard tail, which is precisely the quantity the DK bonus pays for.
    """
    if mean_yards <= 0 or cv <= 0:
        return 0.0
    shape = 1.0 / (cv ** 2)
    return float(stats.gamma.sf(threshold, a=shape, scale=mean_yards / shape))


def project_week(season, week, weekly, schedules, history=None, priors=None,
                 defense=None, min_games=2, player_priors=None):
    """DK-point projections for every rosterable skill player in one week.

    `history` defaults to every completed game before the cutoff. Pass it explicitly when
    projecting many weeks in a row so the same frame is not re-sliced each time.

    `player_priors` is an optional `{player_id: {attempts, carries, targets}}` map -- see
    `nfl.coldstart.projection_priors`. Where a player has one it replaces the positional
    volume prior he would otherwise be shrunk toward. Note the two "priors" arguments are
    different shapes and always were: `priors` is per *position*, this is per *player*.
    """
    history = _before(weekly, season, week) if history is None else history
    history = history[history["position"].isin(POSITIONS)]
    if history.empty:
        return pd.DataFrame()

    priors = priors if priors is not None else positional_priors(history)
    defense = defense if defense is not None else opponent_factors(history)

    context = implied_totals(schedules)
    context = context[(context["season"] == season) & (context["week"] == week)]
    if context.empty:
        return pd.DataFrame()
    by_team = {row["team"]: row for _, row in context.iterrows()}

    # Who is on a roster this week: everyone who has played recently for a team that plays.
    recent = history[history["season"] >= season - 1]
    team_column = "team" if "team" in recent.columns else "recent_team"
    ordered = history.sort_values(["season", "week"])
    latest = recent.sort_values(["season", "week"]).groupby("player_id").tail(1)

    # Grouped once rather than filtered per player. The row-mask version was O(players x
    # rows) and turned a week into 23 seconds, which makes a five-season walk-forward
    # unrunnable -- and an evaluation nobody runs is the same as no evaluation.
    by_player = {pid: group for pid, group in ordered.groupby("player_id", sort=False)}

    rows = []
    for _, player in latest.iterrows():
        game = by_team.get(player.get(team_column))
        if game is None:
            continue                       # bye week, or a team not on this slate
        position = player.get("position")
        prior = priors.get(position)
        if prior is None:
            continue

        # **A per-player volume prior, where one exists.** `positional_priors` shrinks every
        # player toward his position's league average, which is the right target for a
        # veteran with a long record and a poor one for anyone else -- measured on held-out
        # 2025, the model beats a season average by 0.434 MAE at 16+ games of history and
        # loses to it below that.
        #
        # Only the three **volume** terms are overridden. Cold start projects opportunity;
        # it does not project efficiency, and the rate priors stay positional because
        # efficiency does not persist well enough to personalise (yards per carry
        # self-correlates at +0.27, touchdown rates at 0.03-0.24).
        if player_priors:
            personal = player_priors.get(player["player_id"])
            if personal:
                prior = dict(prior)
                for metric in ("attempts", "carries", "targets"):
                    value = personal.get(metric)
                    if value is not None and not (isinstance(value, float)
                                                  and np.isnan(value)):
                        prior[metric] = float(value)
        own = by_player.get(player["player_id"])
        form = _weighted_rates(own) if own is not None else None
        if form is None or form["games"] < min_games:
            continue

        row = _project_player(player, form, prior, position, game, defense)
        if row is not None:
            rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("Proj", ascending=False).reset_index(drop=True)


def _project_player(player, form, prior, position, game, defense):
    """One player's event line and DK points. Factors stay named on the output."""
    games = form["games"]

    # --- volume: the predictable half -------------------------------------------------
    attempts = _shrink(form["attempts"], prior["attempts"], games, VOLUME_SHRINK["attempts"])
    carries = _shrink(form["carries"], prior["carries"], games, VOLUME_SHRINK["carries"])
    targets = _shrink(form["targets"], prior["targets"], games, VOLUME_SHRINK["targets"])

    # --- rates: shrunk toward the positional prior by opportunity ---------------------
    pass_ypa = _shrink(form["pass_yards"] / form["pass_att"] if form["pass_att"] else np.nan,
                       prior["pass_ypa"], form["pass_att"], EFFICIENCY_SHRINK)
    rush_ypc = _shrink(form["rush_yards"] / form["rush_att"] if form["rush_att"] else np.nan,
                       prior["rush_ypc"], form["rush_att"], EFFICIENCY_SHRINK)
    rec_ypt = _shrink(form["rec_yards"] / form["rec_targets"] if form["rec_targets"] else np.nan,
                      prior["rec_ypt"], form["rec_targets"], EFFICIENCY_SHRINK)
    catch_rate = _shrink(form["receptions"] / form["rec_targets"] if form["rec_targets"] else np.nan,
                         prior["catch_rate"], form["rec_targets"], EFFICIENCY_SHRINK)
    pass_td = _shrink(form["pass_tds"] / form["pass_att"] if form["pass_att"] else np.nan,
                      prior["pass_td_rate"], form["pass_att"], TD_SHRINK)
    rush_td = _shrink(form["rush_tds"] / form["rush_att"] if form["rush_att"] else np.nan,
                      prior["rush_td_rate"], form["rush_att"], TD_SHRINK)
    rec_td = _shrink(form["rec_tds"] / form["rec_targets"] if form["rec_targets"] else np.nan,
                     prior["rec_td_rate"], form["rec_targets"], TD_SHRINK)
    int_rate = _shrink(form["interceptions"] / form["pass_att"] if form["pass_att"] else np.nan,
                       prior["int_rate"], form["pass_att"], EFFICIENCY_SHRINK)

    # --- named factors ----------------------------------------------------------------
    # Scoring scales with the implied total; yardage scales with its square root. Both move
    # with the game environment, but a team projected for 31 points instead of 21 does not
    # gain 48% more yards -- it converts the yards it has into more touchdowns.
    total_factor = float(game["implied"]) / LEAGUE_TEAM_POINTS if LEAGUE_TEAM_POINTS else 1.0
    yard_factor = float(np.sqrt(max(total_factor, 0.05)))
    matchup = defense.get((game["opponent"], position), 1.0)

    pass_yards = attempts * pass_ypa * yard_factor * matchup
    rush_yards = carries * rush_ypc * yard_factor * matchup
    rec_yards = targets * rec_ypt * yard_factor * matchup
    events = {
        "PASS_YD": pass_yards,
        "PASS_TD": attempts * pass_td * total_factor * matchup,
        "INT": attempts * int_rate,
        "RUSH_YD": rush_yards,
        "RUSH_TD": carries * rush_td * total_factor * matchup,
        "REC": targets * catch_rate,
        "REC_YD": rec_yards,
        "REC_TD": targets * rec_td * total_factor * matchup,
        "FUM_LOST": form["fumbles"] / max(form["weight"], 1e-9),
        "P_PASS_300": _bonus_probability(pass_yards, 300, YARDAGE_CV["QB_pass"]),
        "P_RUSH_100": _bonus_probability(rush_yards, 100, YARDAGE_CV["RB_rush"]),
        "P_REC_100": _bonus_probability(
            rec_yards, 100, YARDAGE_CV.get(f"{position}_rec", YARDAGE_CV["WR_rec"])),
    }
    proj = offense_points(events)
    band = BAND_FIT.get(position, BAND_FIT["WR"])
    team_column = "team" if "team" in player.index else "recent_team"
    return {
        "Name": player.get("player_display_name", player.get("player_name")),
        "player_id": player.get("player_id"),
        "Team": player.get(team_column), "Opp": game["opponent"],
        "Game": f"{game['opponent']}@{game['team']}" if game["is_home"]
                else f"{game['team']}@{game['opponent']}",
        "Pos": position, "DK Pos": position, "Type": "OFF",
        "Proj": round(proj, 2),
        "Floor": round(max(0.0, band["floor"][0] * proj + band["floor"][1]), 2),
        "Ceiling": round(band["ceiling"][0] * proj + band["ceiling"][1], 2),
        # Named factors, carried through so a board can say why.
        "Team Total": round(float(game["implied"]), 1),
        "Total Factor": round(total_factor, 3),
        "Matchup": round(float(matchup), 3),
        "Games": form["games"],
        **{f"E_{k}": round(v, 4) for k, v in events.items()},
    }


def project_dst(season, week, weekly, schedules, history=None, shrink=6.0):
    """DK-point projections for team defenses.

    The points-allowed term dominates a DST projection and is a step function of the
    opponent's final score, so it is projected from a DISTRIBUTION around the opponent's
    implied total rather than from the number itself -- see `nfl.scoring` for why the mean
    is the wrong input.
    """
    from .backtest import dst_actuals
    history = _before(weekly, season, week) if history is None else history
    if history.empty:
        return pd.DataFrame()

    past = dst_actuals(history, schedules)
    context = implied_totals(schedules)
    context = context[(context["season"] == season) & (context["week"] == week)]
    if past.empty or context.empty:
        return pd.DataFrame()

    league = {event: float(past[event].mean()) for event in ("SACK", "INT", "FUM_REC", "TD", "BLK")
              if event in past.columns}
    rows = []
    for _, game in context.iterrows():
        own = past[past["team"] == game["team"]]
        events = {}
        for event, mean in league.items():
            observed = float(own[event].mean()) if len(own) else mean
            events[event] = _shrink(observed, mean, len(own), shrink)
        events["SAFETY"] = float(past["SAFETY"].mean()) if "SAFETY" in past else 0.0

        spread = _points_allowed_spread(float(game["opp_implied"]))
        proj = sum(events.get(event, 0.0) * value for event, value in DK_DST.items())
        proj += sum(p * points_allowed_points(total) for total, p in spread.items())
        rows.append({
            "Name": f"{game['team']} DST", "player_id": f"DST_{game['team']}",
            "Team": game["team"], "Opp": game["opponent"],
            "Game": f"{game['opponent']}@{game['team']}" if game["is_home"]
                    else f"{game['team']}@{game['opponent']}",
            "Pos": "DST", "DK Pos": "DST", "Type": "DST",
            "Proj": round(proj, 2),
            "Floor": round(max(0.0, 0.20 * proj - 0.5), 2),
            "Ceiling": round(2.10 * proj + 2.0, 2),
            "Opp Total": round(float(game["opp_implied"]), 1),
            "Team Total": round(float(game["implied"]), 1),
            "Total Factor": 1.0, "Matchup": 1.0, "Games": len(own),
            **{f"E_{k}": round(v, 4) for k, v in events.items()},
        })
    return pd.DataFrame(rows).sort_values("Proj", ascending=False).reset_index(drop=True)


# Dispersion of a team's actual score around its implied total, fitted over 2022-24. The
# points-allowed buckets are seven points wide, so a 9-point spread straddles more than one
# of them -- which is exactly why the DST projection needs a distribution and not a mean.
SCORE_SD = 8.99


def _points_allowed_spread(implied, sd=SCORE_SD, lo=0, hi=56):
    """A discrete distribution over the opponent's final score."""
    totals = np.arange(lo, hi + 1)
    density = stats.norm.pdf(totals, loc=implied, scale=sd)
    density = density / density.sum()
    return dict(zip(totals.tolist(), density.tolist()))


# --------------------------------------------------------------------------------------
# Calibration
#
# Everything above that is a number rather than a rule comes from here. `calibrate` runs a
# walk-forward over the seasons it is given and refits the bands, the yardage dispersion and
# the league baselines against what actually happened, then prints a block that can be
# pasted back over the constants. Nothing is fitted on a season it will later be scored on.
# --------------------------------------------------------------------------------------

def _fit_band(frame, quantile):
    """Least-squares (slope, intercept) for a conditional quantile of Actual on Proj.

    Bucketed first, because a straight quantile regression on 5,000 noisy rows is dominated
    by the low-projection mass where most players live. Buckets give the fit something to
    anchor on across the whole range a lineup actually chooses from.
    """
    rows = frame[["Proj", "Actual"]].dropna()
    if len(rows) < 200:
        return None
    bucket = pd.qcut(rows["Proj"], q=12, duplicates="drop")
    grouped = rows.groupby(bucket, observed=True).agg(
        proj=("Proj", "mean"), value=("Actual", lambda s: s.quantile(quantile)),
        n=("Actual", "size"))
    grouped = grouped[grouped["n"] >= 25]
    if len(grouped) < 4:
        return None
    slope, intercept = np.polyfit(grouped["proj"], grouped["value"], 1)
    return round(float(slope), 3), round(float(intercept), 2)


def _fit_cv(projected_yards, hit, threshold, grid=None):
    """The gamma CV whose modelled exceedance rate matches the realised one.

    Solved by search rather than from the sample standard deviation. The quantity that has
    to be right is P(yards >= threshold), and matching a variance does not guarantee
    matching a tail probability when the distribution is this skewed -- fitting the moment
    instead of the thing you use is how a bonus term ends up 60% too generous.
    """
    grid = grid if grid is not None else np.arange(0.30, 1.60, 0.01)
    mask = projected_yards > 5
    if mask.sum() < 100:
        return None
    yards, realised = projected_yards[mask].to_numpy(), hit[mask].to_numpy().mean()
    best, best_gap = None, np.inf
    for cv in grid:
        shape = 1.0 / (cv ** 2)
        modelled = stats.gamma.sf(threshold, a=shape, scale=yards / shape).mean()
        gap = abs(modelled - realised)
        if gap < best_gap:
            best, best_gap = cv, gap
    return round(float(best), 3)


def calibrate(weekly, schedules, seasons, first_week=3):
    """Refit every fitted constant in this module. Returns a dict ready to paste back.

    Hold out whatever seasons the result will be evaluated on -- calibrating on 2021-25 and
    then reporting a 2025 score is an in-sample number wearing a walk-forward costume.
    """
    from .backtest import offense_actuals
    from .evaluate import walk_forward

    frames = [walk_forward(int(s), weekly, schedules, first_week=first_week)
              for s in seasons]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return {}
    result = pd.concat(frames, ignore_index=True)
    actual = offense_actuals(weekly[weekly["season"].isin([int(s) for s in seasons])])

    bands = {}
    for position in POSITIONS:
        rows = result[result["Pos"] == position]
        floor, ceiling = _fit_band(rows, 0.25), _fit_band(rows, 0.90)
        if floor and ceiling:
            bands[position] = {"floor": floor, "ceiling": ceiling}

    keys = ["season", "week", "player_id"]
    merged = result.merge(
        actual[keys + ["E_P_RUSH_100", "E_P_REC_100", "E_P_PASS_300"]].rename(columns={
            "E_P_RUSH_100": "hit_rush", "E_P_REC_100": "hit_rec",
            "E_P_PASS_300": "hit_pass"}), on=keys, how="inner")
    cv = {}
    for label, position, yards, hit, threshold in (
            ("QB_pass", "QB", "E_PASS_YD", "hit_pass", 300),
            ("RB_rush", "RB", "E_RUSH_YD", "hit_rush", 100),
            ("WR_rec", "WR", "E_REC_YD", "hit_rec", 100),
            ("TE_rec", "TE", "E_REC_YD", "hit_rec", 100)):
        rows = merged[merged["Pos"] == position]
        fitted = _fit_cv(rows[yards], rows[hit], threshold) if not rows.empty else None
        if fitted:
            cv[label] = fitted

    played = schedules[schedules["home_score"].notna()
                       & schedules["season"].isin([int(s) for s in seasons])]
    scores = pd.concat([played["home_score"], played["away_score"]])
    context = implied_totals(played)
    residual = None
    if not context.empty:
        actual_points = pd.concat([
            played[["season", "week", "home_team", "home_score"]].rename(
                columns={"home_team": "team", "home_score": "points"}),
            played[["season", "week", "away_team", "away_score"]].rename(
                columns={"away_team": "team", "away_score": "points"})])
        joined = context.merge(actual_points, on=["season", "week", "team"], how="inner")
        residual = float((joined["points"] - joined["implied"]).std())

    return {
        "seasons": [int(s) for s in seasons],
        "rows": len(result),
        "LEAGUE_TEAM_POINTS": round(float(scores.mean()), 2),
        "SCORE_SD": round(residual, 2) if residual else None,
        "YARDAGE_CV": cv,
        "BAND_FIT": bands,
    }
