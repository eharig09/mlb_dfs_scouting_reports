"""Turn a cached scouting-report payload into DK-point projections.

Design: every player gets baseline event rates from their season line (regressed to
league), then a set of named multiplicative matchup factors. Keeping the factors named
and separate is the point -- they are what the board shows as the reason a player is
flagged, and they are what you tune when the model feels wrong.
"""

import math
import re

import pandas as pd

from .ros import load as load_ros
from .scoring import DK_HITTER, DK_PITCHER, hitter_points, pitcher_points

# Bumped whenever a change here would move a projection. Snapshots and backtests record it,
# so a metric computed under one version is never silently compared against another.
#
#   1  original event-rate model
#   2  weather read from environment["forecast"] as well as ["weather"], enclosed roofs
#      neutralised (the old code was inert on 93 of 142 cached games)
#   3  (unreleased)
#   4  sample-size regression on both sides of the ball. Starter rate stats were read raw
#      off the season line, so a three-inning FIP was treated as true talent; hitter rates
#      regressed to league average regardless of track record, so a 40-PA call-up was
#      treated as an average major leaguer. Both now shrink by sample size -- pitchers
#      toward league via REG_SP, hitters toward a per-player rest-of-season prior (dfs.ros)
#      falling back to league. Team run projections are damped before being split across a
#      lineup (TEAM_RUN_DAMPING).
MODEL_VERSION = 4

# Ceiling settings. CEILING_Z targets roughly a 90th-percentile outcome -- the score you
# need for a tournament, not the score you expect. Backtested at 10.0% exceedance for
# hitters and 12.6% for pitchers, so the band means what it says.
CEILING_Z = 1.35

# Floor and bust bands, fit empirically against 823 hitter-games and 95 starts rather
# than assumed from a normal distribution -- single-game DK scoring is far too skewed for
# that. Floor is the 25th percentile outcome; bust is P(scoring 3 or fewer).
#
# The hitter numbers are the important context here: 27% of hitter-games score zero and
# 44% score 3 or less, no matter how good the spot looks. A genuinely high-floor MLB
# hitter does not exist in a single game -- floor plays live at pitcher.
FLOOR_FIT = {
    "H": {"slope": 0.862, "intercept": -4.847},
    "P": {"slope": 0.796, "intercept": -3.063},
}
BUST_FIT = {
    "H": {"slope": -0.0743, "intercept": 0.9437},
    "P": {"slope": -0.0211, "intercept": 0.4493},
}

# The best-projected hitter bucket in the backtest still busted 32.7% of the time, and the
# best pitcher bucket 5.3%. Projections above the observed range would otherwise
# extrapolate to bust rates nobody has ever posted, which is precisely the wrong place to
# be optimistic -- these are the numbers the "safe play" call rests on.
BUST_FLOOR = {"H": 0.32, "P": 0.05}


def floor_points(points, player_type):
    """25th-percentile outcome for a given projection."""
    fit = FLOOR_FIT[player_type]
    return max(0.0, fit["slope"] * points + fit["intercept"])


def bust_probability(points, player_type):
    """Probability of scoring 3 or fewer DK points, floored at the best observed rate."""
    fit = BUST_FIT[player_type]
    raw = fit["slope"] * points + fit["intercept"]
    return min(0.95, max(BUST_FLOOR[player_type], raw))
HR_POINT_VALUE = 14.0        # HR (10) plus the run and RBI it always carries
HITTER_BASE_VAR = 22.0       # singles, doubles, walks, and the R/RBI the player didn't drive
PITCHER_BASE_VAR = 30.0      # length of outing: the blowup-vs-quality-start swing
ER_OVERDISPERSION = 2.0      # runs cluster in innings; they are not Poisson

# ---------------------------------------------------------------------------
# League baselines. Update these once a season; everything else regresses to them.
# ---------------------------------------------------------------------------
LG = {
    "runs_per_team_game": 4.40,
    "ops": 0.715,
    "avg": 0.246,
    "iso": 0.160,
    "bb_rate": 0.085,
    "hbp_rate": 0.012,
    "k_rate": 0.222,
    "hr_per_ab": 0.033,
    "fip": 4.15,
    "era": 4.15,
    "sp_ip": 5.20,
}

# Run prevention weights FIP heavily -- it is the more predictive of the two -- but not
# exclusively. FIP ignores defense and contact management entirely, and DK scores the
# runs that actually cross the plate, so a pitcher whose ERA has badly trailed his FIP
# should not be projected as though the gap were pure luck.
ERA_BLEND = {"fip": 0.75, "era": 0.25}

# Plate appearances accrued by the player who STARTS in each slot -- not the slot's own
# PA. The difference is pinch hitters and defensive substitutions, which is why these sit
# below the ~38 PA a lineup turns over and why they fall off faster at the bottom of the
# order. Backtested against 823 player-games; the slot curve was ~11% high before.
SLOT_PA = {1: 4.46, 2: 4.38, 3: 4.30, 4: 4.18, 5: 3.98, 6: 3.72, 7: 3.48, 8: 3.30, 9: 3.10}

# Per-factor damping for hitter matchups. Each factor is raised to its exponent before
# entering the product, so 0.0 disables a factor and 1.0 trusts it fully. Backtesting
# showed the raw factors moved projections by 2+ points while actual scoring was nearly
# flat across matchup quintiles -- the spread was noise, not signal. Set from that data.
# Most of these are zero because the team's projected runs -- which drive every hitter's
# R and RBI -- ALREADY price in the opposing starter, the bullpen, and the park. Applying
# them again as rate multipliers counted the same effect twice, and the backtest showed it
# costing both correlation and top-quintile separation. They remain in the Why notes as
# context for your own read; they just no longer move the number.
FACTOR_DAMPING = {
    "arsenal": 0.0,     # 20-70 PA vs a pitch shape: too little signal for one game
    "opp_sp": 0.0,      # already in the team run projection
    "bullpen": 0.0,     # already in the team run projection
    "park": 0.0,        # already in the team run projection (park HR is applied separately)
    "form": 1.0,        # player-level, genuinely independent of the team run projection
}
MATCHUP_DAMPING = 1.0   # global multiplier on the combined factor, after per-factor damping

# Relative share of team runs scored / driven in by slot, before adjusting for player
# skill. Normalized below so a lineup's projected runs and RBI always add back up to the
# team total -- these are shares, and shares that don't sum to one quietly lose offense.
SLOT_R_WEIGHT = {1: 0.125, 2: 0.122, 3: 0.118, 4: 0.113, 5: 0.109, 6: 0.105, 7: 0.104, 8: 0.102, 9: 0.102}
SLOT_RBI_WEIGHT = {1: 0.094, 2: 0.104, 3: 0.119, 4: 0.126, 5: 0.120, 6: 0.110, 7: 0.101, 8: 0.066, 9: 0.060}

# Share of runs that come with an RBI attached (the rest are unearned, wild pitches, etc).
RBI_PER_RUN = 0.95

# Damping on the team run projection before it is split across the lineup.
#
# `Exp Runs` drives every hitter's R and RBI, which is about 2 of a typical 6.7-point
# projection, and it was applied at full strength. Walk-forward residuals say that is too
# much: the correlation between the team run projection and (actual - projected) is negative
# in *every* salary tier (-0.083, -0.125, -0.129, -0.100, -0.101), which is the signature of
# a term the model leans on harder than it deserves. The same pattern shows up on the season
# rate line, so this is the run-environment half of a broader over-extrapolation.
#
# Shrinking toward the league average keeps the ordering of games intact -- a 5.5-run team
# still projects above a 3.5-run team -- while narrowing a spread the results do not support.
# Only the run/RBI split and the lineup-turnover multiplier see the damped number; the
# `Team Runs` field the board and the stacker read stays the raw scorecard value.
TEAM_RUN_DAMPING = 0.65

# The replacement-level prior for hitters; see `_replacement_baseline`. A bat with no track
# record is treated as ~12% below league average, sliding up to league average by the time he
# has a half-season of plate appearances behind him.
REPLACEMENT_SHARE = 0.88
REPLACEMENT_PA_FULL = 300.0

# How a rest-of-season projection is used, when one exists for the hitter.
#
#   "target"  regress his in-season line toward the ROS rate
#   "direct"  take the ROS rate as the estimate and skip the in-season regression
#
# "direct" is the correct reading and the measured one. A ROS projection is not a prior --
# it is a *posterior* that already contains the player's in-season line, plus prior seasons,
# minor-league history and aging that this model never sees. Regressing his season rate
# toward it therefore counts the same season twice, and the walk-forward diff showed exactly
# that signature: bias got worse precisely in the tiers full of established regulars
# (premium -0.253 -> -0.501, expensive -0.019 -> -0.156) whose season line and ROS line agree
# and were being stacked, while the thin-sample tier it was meant to help improved.
ROS_MODE = "direct"

# Regression strength (in PA / batters faced) for each blended source.
REG = {
    "season": 200,
    "platoon": 150,
    "arsenal": 70,
    "sp_ip": 6,        # in starts
    "opp_k": 120,
}

# Regression strength for a STARTER's own rate stats, in season innings pitched.
#
# The hitter path has always regressed its rate line to league by sample size; the pitcher
# path did not, and read FIP / ERA / WHIP / K% / BB% straight off the season line. The `LG`
# defaults below only fire when a value is *missing*, so an arm with three innings of work
# was projected as though its three-inning FIP were true talent. Walk-forward bias by season
# innings showed exactly the regression-to-the-mean failure that implies:
#
#     <20 IP  +4.27      40-80 IP  +0.53
#     20-40   +2.97      80-130    -0.24
#
# and it produced projections no starter can post -- Casey Mize at -9.52 DK points off a
# 3.3-inning line (he scored 17.30), Scherzer at 1.17 off an injury return (18.50). The
# damage lands almost entirely in the cheap tiers, because that is who DK prices off a short
# or ugly line, which is why the min-priced tier was the worst segment in the report.
#
# Ordered by how fast each stat stabilises: strikeout rate is a skill that shows up almost
# immediately, walk rate takes longer, and the run-prevention numbers are slowest because
# they carry batted-ball luck. Values are the usual stabilisation points converted from
# batters faced at ~4.3 BF per inning.
REG_SP = {
    "k": 16,      # ~70 BF
    "bb": 40,     # ~170 BF
    "whip": 70,   # ~300 BF, BABIP-driven and slow
    "fip": 70,    # built from K/BB/HR, so faster than ERA
    "era": 110,   # slowest of all; the ERA/FIP blend leans on FIP for this reason
}

# What an unknown starter regresses toward. A pitcher nobody has seen is not a league-average
# starter -- he is up because someone got hurt -- so run prevention regresses to a shade worse
# than league while the rate skills regress to league proper.
REPLACEMENT_PENALTY = 0.20   # runs of FIP/ERA added to the regression target

# How hard each factor is allowed to push. Matchup edges are real but the tails of
# these ratios are almost always small-sample noise.
CLIP = {
    "platoon": (0.86, 1.18),
    "arsenal": (0.90, 1.12),
    "opp_sp": (0.87, 1.16),
    "bullpen": (0.94, 1.07),
    "park_run": (0.90, 1.12),
    "form": (0.93, 1.09),
    "weather": (0.94, 1.10),
    "total": (0.72, 1.38),
}

# Sensitivity of each event type to the combined matchup factor. Power swings hardest,
# walks barely move.
ELASTICITY = {"hr": 1.70, "hit": 1.00, "bb": 0.35}

# Share of a hitter's PA that come against the bullpen rather than the starter.
BULLPEN_PA_SHARE = 0.32


def _num(value, default=None):
    """Pull a float out of report data that may be a string, a percent, or blank."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return default if (isinstance(value, float) and math.isnan(value)) else float(value)
    text = str(value).strip().replace("%", "").replace(",", "")
    if not text or text in {"-", "--", "N/A", "nan", "None"}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _clip(value, bounds):
    low, high = bounds
    return max(low, min(high, value))


def _shrink(ratio, sample, k):
    """Pull a ratio toward 1.0 based on how much sample backs it."""
    if ratio is None or sample is None or sample <= 0:
        return 1.0
    weight = sample / (sample + k)
    return 1.0 + (ratio - 1.0) * weight


def _log5(pitcher_rate, batter_rate, league_rate):
    """Odds-ratio (log5) combination of a pitcher and batter rate.

    The standard way to combine two rates against a league baseline. Preferred over
    scaling the pitcher's rate by opponent/league because it stays bounded in [0, 1] and
    behaves correctly when both sides are extreme -- an elite strikeout arm against an
    elite contact lineup should not simply multiply through.
    """
    if not 0 < league_rate < 1:
        return pitcher_rate
    pitcher_rate = min(max(pitcher_rate, 1e-6), 1 - 1e-6)
    batter_rate = min(max(batter_rate, 1e-6), 1 - 1e-6)
    odds = (pitcher_rate * batter_rate / league_rate)
    complement = ((1 - pitcher_rate) * (1 - batter_rate) / (1 - league_rate))
    return odds / (odds + complement) if (odds + complement) > 0 else pitcher_rate


def _regress(value, sample, league_value, k):
    if value is None:
        return league_value
    sample = sample or 0
    return (value * sample + league_value * k) / (sample + k)


# ---------------------------------------------------------------------------
# Row lookup
#
# Every per-player lookup here used to be `frame[frame["Name"] == name].iloc[0]`, run once
# per player against the same frame. Profiling a 15-game slate put 1.30s of 1.86s of
# projection time in four of those -- a full boolean scan plus a row materialisation, ~270
# times each, to answer a dictionary question. Building the dictionary once per frame gives
# byte-identical output roughly 4x faster.
#
# The cache is keyed on object identity and *verified* against the frame it was built from,
# because id() is recycled after garbage collection. It is cleared at the top of every
# project_game so it cannot grow with the slate.
# ---------------------------------------------------------------------------

_INDEX_CACHE = {}


def _build_index(frame, key_column, strip_hand=False):
    """{key: {column: value}} for a report frame. First row wins, matching .iloc[0]."""
    if not isinstance(frame, pd.DataFrame) or frame.empty or key_column not in frame.columns:
        return {}
    columns = [str(c) for c in frame.columns]
    if strip_hand:
        # Platoon frames suffix every column with the hand faced ("OPS vs R"). Stripping it
        # can collide two columns onto one name; dict(zip(...)) keeps the last, which is what
        # the original dict comprehension did.
        columns = [re.sub(r"\s+vs\s+[LR]$", "", c) for c in columns]
    values = frame.to_numpy(dtype=object)
    index = {}
    for position, key in enumerate(frame[key_column].astype(str)):
        if key not in index:
            index[key] = dict(zip(columns, values[position]))
    return index


def _indexed(frame, key_column, strip_hand=False):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return {}
    cache_key = (id(frame), key_column, strip_hand)
    cached = _INDEX_CACHE.get(cache_key)
    if cached is not None and cached[0] is frame:
        return cached[1]
    built = _build_index(frame, key_column, strip_hand)
    _INDEX_CACHE[cache_key] = (frame, built)
    return built


# ---------------------------------------------------------------------------
# Game-level context
# ---------------------------------------------------------------------------

def _park_factors(environment):
    park = environment.get("park") or {}
    return {
        "hr": _num(park.get("HR"), 1.0) or 1.0,
        "run": _num(park.get("Runs"), 1.0) or 1.0,
    }


# Roofs that are shut over the field. A closed roof means still air and a controlled
# temperature, so neither term should move -- and "retractable" is not one of them, because
# a retractable roof in July is almost always open.
ENCLOSED_ROOFS = {"cover", "fixed", "closed", "dome"}


def _weather_inputs(environment):
    """(temperature F, wind text, enclosed) from whichever source has it.

    Two sources exist and they are not interchangeable. `environment["weather"]` is
    StatsAPI's live report -- `temp` as a string, wind as "8 mph, L To R" -- and it only
    appears close to first pitch: 49 of the 142 cached payloads carry it. Open-Meteo's
    `environment["forecast"]` carries `temp_f`, `wind` and `roof` on 118 of them.

    This used to read `weather` alone, so on 93 of 142 games the temperature and wind terms
    silently collapsed to 1.0 -- a missing key looks exactly like neutral weather. A 96F
    afternoon in Oakland got no carry adjustment at all. Live reading wins where it exists,
    since it is the measurement rather than the forecast.
    """
    weather = environment.get("weather") or {}
    forecast = environment.get("forecast") or {}

    temp = _num(weather.get("temp"))
    if temp is None:
        temp = _num(forecast.get("temp_f"))

    wind = str(weather.get("wind") or forecast.get("wind") or "")
    enclosed = str(forecast.get("roof") or "").strip().lower() in ENCLOSED_ROOFS
    return temp, wind, enclosed


def _weather_hr_factor(environment):
    """Temperature and wind nudge on home runs. Deliberately small."""
    temp, wind, enclosed = _weather_inputs(environment)
    if enclosed:
        return 1.0

    factor = 1.0
    if temp is not None:
        factor *= 1.0 + (temp - 72.0) * 0.0022      # ~+2% per 10 degrees

    speed = _num((re.match(r"\s*(\d+)", wind) or [None, None])[1] if re.match(r"\s*(\d+)", wind) else None, 0.0) or 0.0
    direction = wind.lower()
    if "out to" in direction:
        factor *= 1.0 + min(speed, 20) * 0.0055
    elif "in from" in direction:
        factor *= 1.0 - min(speed, 20) * 0.0050
    return _clip(factor, CLIP["weather"])


def _scorecard_row(scorecard, team):
    return _indexed(scorecard, "Team").get(str(team), {})


def _win_probability(scorecard_row):
    return (_num(scorecard_row.get("Win Lean"), 50.0) or 50.0) / 100.0


def _bullpen_factor(bullpen_df):
    """Opposing bullpen quality as an offense multiplier (worse pen -> better for bats)."""
    if not isinstance(bullpen_df, pd.DataFrame) or bullpen_df.empty or "FIP" not in bullpen_df:
        return 1.0
    top = bullpen_df.head(6)
    fip = pd.to_numeric(top["FIP"], errors="coerce").mean()
    if pd.isna(fip):
        return 1.0
    return _clip(1.0 + (float(fip) - LG["fip"]) * 0.055, CLIP["bullpen"])


# ---------------------------------------------------------------------------
# Hitters
# ---------------------------------------------------------------------------

def _platoon_row(splits_df, name):
    # Columns are suffixed with the hand faced ("OPS vs R"); stripped for uniform access.
    return _indexed(splits_df, "Name", strip_hand=True).get(str(name), {})


def _arsenal_row(arsenal_df, name):
    return _indexed(arsenal_df, "Name").get(str(name), {})


def _team_pa_multiplier(exp_runs, is_home, home_win_prob):
    """Lineup turnover scales with offense; the home team often skips the 9th."""
    mult = 1.0 + (exp_runs - LG["runs_per_team_game"]) * 0.050
    if is_home:
        mult *= 1.0 - 0.055 * home_win_prob
    return max(0.85, min(1.15, mult))


def _hitter_factors(row, platoon, arsenal, ctx):
    """Named multiplicative matchup factors, split into supporting and cautionary notes.

    Direction matters on a recommender: "park HR 0.92" is a reason to fade, not a reason
    to play, and a Why column that mixes the two is worse than no Why column at all.
    """
    season_ops = _num(row.get("OPS"), LG["ops"]) or LG["ops"]
    factors, supports, cautions = {}, [], []

    def note(factor, text):
        (supports if factor >= 1.0 else cautions).append(text)

    # The platoon split is deliberately NOT a factor here: _hitter_rate_line already
    # blends it into AVG/ISO/BB%/K% per event, which is the more granular application.
    # Multiplying by an OPS ratio on top counted the same effect twice and was the main
    # source of over-projection. Kept as a note only.
    platoon_ops = _num(platoon.get("OPS"))
    platoon_pa = _num(platoon.get("PA"), 0.0) or 0.0
    if platoon_ops and season_ops > 0 and platoon_pa >= 60:
        ratio = platoon_ops / season_ops
        if abs(ratio - 1.0) >= 0.035:
            note(ratio, f"platoon {platoon_ops:.3f} OPS ({int(platoon_pa)} PA)")

    arsenal_ops = _num(arsenal.get("OPS"))
    arsenal_pa = _num(arsenal.get("PA"), 0.0) or 0.0
    if arsenal_ops and season_ops > 0:
        factor = _clip(_shrink(arsenal_ops / season_ops, arsenal_pa, REG["arsenal"]), CLIP["arsenal"])
        factors["arsenal"] = factor
        if arsenal_pa >= 12 and abs(factor - 1.0) >= 0.03:
            fit = str(arsenal.get("Fit") or "").strip()
            note(factor, f"arsenal {arsenal_ops:.3f} OPS/{arsenal_pa:.0f} PA{f' ({fit})' if fit else ''}")

    opp_fip = _num(ctx.get("opp_sp_fip"))
    if opp_fip:
        factor = _clip(1.0 + (opp_fip - LG["fip"]) * 0.070, CLIP["opp_sp"])
        factors["opp_sp"] = factor
        if abs(factor - 1.0) >= 0.04:
            note(factor, f"opp SP {opp_fip:.2f} FIP")

    factors["bullpen"] = 1.0 + (ctx.get("opp_bullpen_factor", 1.0) - 1.0) * BULLPEN_PA_SHARE
    factors["park"] = _clip(ctx.get("park_run", 1.0), CLIP["park_run"])

    form = _num(row.get("Off L28"))
    baseline = _num(row.get("Off Szn"))
    if form is not None and baseline is not None and baseline != 0:
        factor = _clip(_shrink(1.0 + (form - baseline) / 260.0, 60, 60), CLIP["form"])
        factors["form"] = factor
        if abs(factor - 1.0) >= 0.035:
            note(factor, f"L28 form {form:+.0f} vs {baseline:+.0f}")

    total = 1.0
    for name, value in factors.items():
        total *= value ** FACTOR_DAMPING.get(name, 1.0)
    factors["raw_total"] = _clip(total, CLIP["total"])
    factors["total"] = factors["raw_total"] ** MATCHUP_DAMPING
    return factors, supports, cautions


def _replacement_baseline(season_pa):
    """Multiple of league average that a hitter's rate line should regress toward.

    The rate line regressed to league average regardless of how much of a track record
    backed it, which quietly treats a 40-plate-appearance call-up as an average major
    leaguer. He is not: he is up because someone got hurt, and a bat with no history is
    far likelier to be organisational depth than an average regular. Walk-forward bias by
    season PA says exactly that --

        <50 PA   -1.28      120-250  -0.47      400+  +0.41
        50-120   -1.03      250-400  -0.07

    -- monotone across 3,489 hitter-games, with the two thin buckets significant on a
    bootstrap. Regressing toward league was the mechanism: it is the only part of the model
    that treats those hitters as average, and it is applied hardest exactly where the track
    record is thinnest.

    The prior slides with playing time rather than switching at a threshold, so nothing
    jumps at a cutoff, and a settled regular is untouched -- his own line already carries
    almost all the weight, and by `REPLACEMENT_PA_FULL` the target is league average anyway.
    """
    weight = min(1.0, max(0.0, (season_pa or 0.0) / REPLACEMENT_PA_FULL))
    return REPLACEMENT_SHARE + (1.0 - REPLACEMENT_SHARE) * weight


def _hitter_rate_line(row, platoon, ros=None):
    """Blend season and platoon rate stats, regressed toward a per-player prior.

    `ros` is one hitter's rest-of-season composite from `dfs.ros`, or None. When it is
    present it *replaces* the league-average-times-replacement target; the sample-size
    weighting around it is unchanged. A projection system has already done the aging and
    minor-league regression this model cannot, so it is a better answer to "what do we
    believe about a hitter we have barely seen" than a flat constant is.
    """
    season_pa = _num(row.get("PA"), 0.0) or 0.0
    platoon_pa = _num(platoon.get("PA"), 0.0) or 0.0
    weight = platoon_pa / (platoon_pa + REG["platoon"]) if platoon_pa else 0.0
    # Set by the player's full season, not the split: how established he is is a fact about
    # him, not about the hand he happens to be facing tonight. Only used as the fallback
    # prior now, for hitters the projection files do not cover.
    replacement = _replacement_baseline(season_pa)

    def target_for(league_key, ros_key):
        value = (ros or {}).get(ros_key)
        return LG[league_key] * replacement if value is None else value

    def blend(season_key, platoon_key, league_key, ros_key=None):
        season_value = _num(row.get(season_key))
        platoon_value = _num(platoon.get(platoon_key))
        target = target_for(league_key, ros_key)
        ros_value = (ros or {}).get(ros_key) if ros_key else None
        if ros_value is not None and ROS_MODE == "direct":
            # The projection system has already regressed this player; re-regressing his
            # season line toward it would count that season twice.
            base = ros_value
        else:
            base = _regress(season_value, season_pa, target, REG["season"])
        if platoon_value is None:
            return base
        # The platoon split still applies on top: it is the one thing a full-season
        # projection averages away, and it is specific to tonight's opposing hand.
        return base * (1 - weight) + _regress(platoon_value, platoon_pa, target, REG["season"]) * weight

    # Only the bat-quality rates carry the replacement prior. Walk and strikeout rates are
    # left on the league target: they are plate-discipline traits that do not track roster
    # status nearly as cleanly, and moving them would double-count the same adjustment.
    avg = blend("AVG", "AVG", "avg", "avg")
    iso = blend("ISO", "ISO", "iso", "iso")
    bb_rate = _regress(
        (_num(platoon.get("BB%")) or 0) / 100.0 if _num(platoon.get("BB%")) else None,
        platoon_pa, LG["bb_rate"], REG["platoon"],
    )
    k_rate = _regress(
        (_num(platoon.get("K%")) or 0) / 100.0 if _num(platoon.get("K%")) else None,
        platoon_pa, LG["k_rate"], REG["platoon"],
    )

    hr = _num(row.get("HR"), 0.0) or 0.0
    ab = _num(row.get("AB"), 0.0) or 0.0
    ros_hr = (ros or {}).get("hr_per_ab")
    if ros_hr is not None and ROS_MODE == "direct":
        hr_per_ab = ros_hr
    else:
        hr_per_ab = _regress(hr / ab if ab > 0 else None, ab,
                             target_for("hr_per_ab", "hr_per_ab"), 250)

    return {"avg": avg, "iso": iso, "bb_rate": bb_rate, "k_rate": k_rate, "hr_per_ab": hr_per_ab}


def _steal_rate(baserunning_df, name, season_pa):
    """Season steals per PA, regressed. Speed is sticky, so regression is light."""
    row = _indexed(baserunning_df, "Name").get(str(name))
    if not row or not season_pa or season_pa <= 0:
        return 0.0
    steals = _num(row.get("SB"), 0.0) or 0.0
    return steals / (season_pa + 120)


def project_hitters(side, payload):
    """Project every hitter in one lineup. `side` is 'home' or 'away'."""
    args, ctx = payload["report_args"], payload["advanced_context"]
    home = side == "home"

    lineup_df = args[0] if home else args[7]
    if not isinstance(lineup_df, pd.DataFrame) or lineup_df.empty:
        return []

    splits_df = args[25] if home else args[26]
    baserunning_df = args[19] if home else args[20]
    opp_bullpen_df = args[8] if home else args[1]
    team = args[15] if home else args[16]
    opponent = args[16] if home else args[15]

    environment = ctx.get("environment") or {}
    park = _park_factors(environment)
    scorecard = ctx.get("scorecard")
    team_row = _scorecard_row(scorecard, team)
    home_row = _scorecard_row(scorecard, args[15])

    reported_runs = _num(team_row.get("Exp Runs"), LG["runs_per_team_game"]) \
        or LG["runs_per_team_game"]
    # Damped for the math, raw for the board -- see TEAM_RUN_DAMPING.
    exp_runs = LG["runs_per_team_game"] \
        + (reported_runs - LG["runs_per_team_game"]) * TEAM_RUN_DAMPING
    opp_starter = (args[9] if home else args[2]) or {}
    opp_starter_name = str(opp_starter.get("Name") or "TBD")
    # Carried through so evaluation can segment by handedness matchup. The platoon split is
    # already applied to the rate line; this is the label, not a second application of it.
    opp_starter_hand = str(opp_starter.get("Throws") or "")[:1].upper()

    composite = ctx.get("hitter_composite")
    arsenal_df = ctx.get("home_batter_arsenal" if home else "away_batter_arsenal")
    # Cached in dfs.ros, so this is a dict lookup after the first game of a slate. Empty when
    # no exports are on disk, which falls the rate line back to the league-average prior.
    ros_rates = load_ros()[0]["H"]

    game_ctx = {
        "opp_sp_fip": _num(opp_starter.get("FIP")),
        "opp_bullpen_factor": _bullpen_factor(opp_bullpen_df),
        "park_run": park["run"],
    }
    pa_mult = _team_pa_multiplier(exp_runs, home, _win_probability(home_row))
    weather_hr = _weather_hr_factor(environment)
    _, wind_text, _ = _weather_inputs(environment)

    lineup = lineup_df.copy()
    if isinstance(composite, pd.DataFrame) and not composite.empty:
        extra = [c for c in ("Off Szn", "Off L28", "Composite", "Signal") if c in composite.columns]
        if extra:
            lineup = lineup.merge(
                composite[composite["Team"].astype(str) == str(team)][["Name"] + extra],
                on="Name", how="left",
            )

    # Split team runs across the lineup by slot and by skill, then normalize so the nine
    # hitters' projected runs add back up to the team's projected runs.
    obp_values = pd.to_numeric(lineup.get("OBP"), errors="coerce").fillna(0.320)
    slg_values = pd.to_numeric(lineup.get("SLG"), errors="coerce").fillna(0.400)
    obp_mean = obp_values.mean() or 0.320
    slg_mean = slg_values.mean() or 0.400

    slots = [max(1, min(9, int(_num(row.get("Spot"), i + 1) or i + 1)))
             for i, (_, row) in enumerate(lineup.iterrows())]
    run_weights = [SLOT_R_WEIGHT[s] * (float(obp_values.iloc[i]) / obp_mean if obp_mean else 1.0)
                   for i, s in enumerate(slots)]
    rbi_weights = [SLOT_RBI_WEIGHT[s] * (float(slg_values.iloc[i]) / slg_mean if slg_mean else 1.0)
                   for i, s in enumerate(slots)]
    run_total = sum(run_weights) or 1.0
    rbi_total = sum(rbi_weights) or 1.0

    projections = []
    for index, row in lineup.iterrows():
        name = str(row.get("Name") or "").strip()
        if not name:
            continue
        slot = slots[index]

        platoon = _platoon_row(splits_df, name)
        arsenal = _arsenal_row(arsenal_df, name)
        factors, supports, cautions = _hitter_factors(row, platoon, arsenal, game_ctx)
        mlbam = _num(row.get("ID"))
        rates = _hitter_rate_line(row, platoon,
                                  ros_rates.get(int(mlbam)) if mlbam else None)

        total = factors["total"]
        pa = SLOT_PA[slot] * pa_mult
        bb_rate = min(0.30, rates["bb_rate"] * total ** ELASTICITY["bb"])
        hbp_rate = LG["hbp_rate"]
        ab = pa * (1 - bb_rate - hbp_rate)

        hr_rate = rates["hr_per_ab"] * total ** ELASTICITY["hr"] * park["hr"] * weather_hr
        hr = ab * min(0.16, hr_rate)
        hits = ab * min(0.55, rates["avg"] * total ** ELASTICITY["hit"])
        hits = max(hits, hr)

        # Split extra-base hits out of ISO: HR carry 3 extra bases each, and triples run
        # at roughly a tenth of the doubles-plus-triples pool.
        extra_bases = max(0.0, ab * rates["iso"] * total ** ELASTICITY["hit"] - 3 * hr)
        triples = extra_bases / 12.0
        doubles = max(0.0, extra_bases - 2 * triples)
        if doubles + triples + hr > hits:
            scale = max(0.0, hits - hr) / max(1e-6, doubles + triples)
            doubles, triples = doubles * scale, triples * scale
        singles = max(0.0, hits - doubles - triples - hr)

        runs = exp_runs * (run_weights[index] / run_total)
        rbi = exp_runs * RBI_PER_RUN * (rbi_weights[index] / rbi_total)

        steals = _steal_rate(baserunning_df, name, _num(row.get("PA"), 0.0)) * pa

        events = {
            "1B": singles, "2B": doubles, "3B": triples, "HR": hr,
            "R": runs, "RBI": rbi, "BB": pa * bb_rate, "HBP": pa * hbp_rate, "SB": steals,
        }
        points = hitter_points(events)

        # Ceiling drives the GPP view. Built from event variance rather than a flat
        # spread: a home run is worth ~14 points once its own R and RBI are counted, so
        # power is what actually separates ceilings.
        variance = HITTER_BASE_VAR + hr * (HR_POINT_VALUE ** 2) + steals * (DK_HITTER["SB"] ** 2)
        ceiling = points + CEILING_Z * math.sqrt(variance)

        if slot <= 3:
            supports.append(f"bats {slot}{'st' if slot == 1 else 'nd' if slot == 2 else 'rd'}")
        if park["hr"] >= 1.06:
            supports.append(f"park HR {park['hr']:.2f}")
        elif park["hr"] <= 0.94:
            cautions.append(f"park HR {park['hr']:.2f}")
        if weather_hr >= 1.04:
            supports.append(f"wind {wind_text.strip()}" if wind_text.strip() else "warm air")
        elif weather_hr <= 0.97:
            cautions.append("wind holds it in")
        lineup_conf = str(row.get("Lineup Confidence") or "")
        if lineup_conf.startswith("Projected"):
            # A projected card is a real read on the platoon and the roster, not a stale
            # replay -- worth a softer caution, and worth naming how many slots were patched
            # so a board can tell an intact configuration from a heavily-guessed one.
            cautions.append(f"lineup {lineup_conf.lower()}")
        elif lineup_conf not in ("", "Confirmed"):
            cautions.append("lineup not confirmed")

        projections.append({
            "Name": name,
            "Team": team,
            "Opp": opponent,
            "Slot": slot,
            "Pos": str(row.get("Pos") or ""),
            "Bats": str(row.get("Bats") or ""),
            "MLBAM": _num(row.get("ID")),
            "Type": "H",
            "Opp SP": opp_starter_name,
            "Opp SP Hand": opp_starter_hand,
            "Lineup": str(row.get("Lineup Confidence") or ""),
            "PA": round(pa, 2),
            "Team Runs": round(reported_runs, 2),
            "Proj": round(points, 2),
            "Ceiling": round(ceiling, 2),
            "Floor": round(floor_points(points, "H"), 2),
            "Bust%": round(bust_probability(points, "H") * 100, 1),
            "HR": round(hr, 3),
            "SB": round(steals, 3),
            "Matchup": round(total, 3),
            # Expected event counts, published rather than collapsed into Proj. The
            # simulator draws components and converts with dfs.scoring, so it needs the
            # rates the mean was built from; without them it would have to invert a point
            # total, which cannot recover the mix. `E_` prefixed so nothing collides with
            # the DK salary columns or the existing HR/SB summary fields.
            "E_PA": round(pa, 3),
            "E_1B": round(singles, 4),
            "E_2B": round(doubles, 4),
            "E_3B": round(triples, 4),
            "E_HR": round(hr, 4),
            "E_BB": round(pa * bb_rate, 4),
            "E_HBP": round(pa * hbp_rate, 4),
            "E_R": round(runs, 4),
            "E_RBI": round(rbi, 4),
            "E_SB": round(steals, 4),
            "Supports": supports,
            "Cautions": cautions,
        })
    return projections


# ---------------------------------------------------------------------------
# Pitchers
# ---------------------------------------------------------------------------

def _opponent_k_rate(splits_df):
    """Average K% of the lineup the starter faces, using the platoon split vs his hand."""
    if not isinstance(splits_df, pd.DataFrame) or splits_df.empty:
        return LG["k_rate"], 0.0
    columns = [c for c in splits_df.columns if str(c).startswith("K% vs")]
    pa_columns = [c for c in splits_df.columns if str(c).startswith("PA vs")]
    if not columns:
        return LG["k_rate"], 0.0
    rates = pd.to_numeric(splits_df[columns[0]], errors="coerce") / 100.0
    weights = pd.to_numeric(splits_df[pa_columns[0]], errors="coerce") if pa_columns else None
    if weights is not None and weights.sum() > 0:
        combined = pd.concat([rates, weights], axis=1).dropna()
        if not combined.empty:
            value = (combined.iloc[:, 0] * combined.iloc[:, 1]).sum() / combined.iloc[:, 1].sum()
            return float(value), float(combined.iloc[:, 1].sum())
    value = rates.mean()
    return (LG["k_rate"] if pd.isna(value) else float(value)), 0.0


# How much of a lineup's arsenal K edge shows up in the starter's actual strikeout rate,
# and in how deep he goes. Fitted on 2024-25 starts and re-estimated on 2026, which came
# back higher on both (0.68 and 6.5), so these are the conservative end of the range.
#
# Note the asymmetry with the hitter-side arsenal factor above, which is damped to zero.
# The individual hitter number is noise -- 118,422 hitter-games put its effect on DK points
# at t = 1.5 with no walk-forward gain. Averaged over nine hitters that noise cancels and
# what is left predicts the starter's line at t = -8.1, surviving pitcher fixed effects.
# Same measurement, opposite verdict, because the aggregate is the part that is real.
ARSENAL_K_PASSTHROUGH = 0.45     # d(K rate) / d(lineup K edge), both as fractions
ARSENAL_IP_PASSTHROUGH = 2.77    # d(IP) / d(lineup K edge)
ARSENAL_IP_CLIP = 0.25           # innings this factor may move a projection, either way


def lineup_arsenal_k_edge(arsenal_df, lineup_df):
    """Mean arsenal K edge across the lineup the starter faces, weighted by plate appearances.

    Returns (edge, hitters) with the edge as a fraction of PA. `None` when the arsenal
    table is missing or too few hitters have a usable baseline, which is common early in a
    season and for a lineup full of call-ups.
    """
    if not isinstance(arsenal_df, pd.DataFrame) or arsenal_df.empty:
        return None, 0
    if "K Edge" not in arsenal_df.columns:
        return None, 0

    slots = {}
    if isinstance(lineup_df, pd.DataFrame) and "Name" in lineup_df.columns:
        for index, row in lineup_df.reset_index(drop=True).iterrows():
            slot = _num(row.get("Spot"), index + 1) or index + 1
            slots[str(row.get("Name") or "").strip()] = max(1, min(9, int(slot)))

    edges, weights = [], []
    for _, row in arsenal_df.iterrows():
        edge = _num(row.get("K Edge"))
        if edge is None:
            continue
        edges.append(edge / 100.0)   # the table reports percentage points
        weights.append(SLOT_PA[slots.get(str(row.get("Name") or "").strip(), 5)])
    if len(edges) < 5:
        return None, len(edges)
    total = sum(weights) or 1.0
    return sum(e * w for e, w in zip(edges, weights)) / total, len(edges)


def _recent_start_ip(starter, last_n=5):
    """Mean IP over recent starts, and how many starts back it.

    The season `IP` field counts relief innings while `GS` counts only starts, so IP/GS
    blows up for swingmen and injury returns (it had one arm at 11.4 IP per start). The
    per-start log is the honest source.
    """
    starts = starter.get("Last Starts")
    if isinstance(starts, str):
        return None, 0
    if not isinstance(starts, (list, tuple)) or not starts:
        return None, 0
    values = [
        _num(entry.get("IP")) for entry in starts[:last_n]
        if isinstance(entry, dict) and _num(entry.get("IP")) is not None
    ]
    if not values:
        return None, 0
    return sum(values) / len(values), len(values)


def _project_innings(starter):
    """Blend recent per-start workload with the season rate, regressed to league."""
    recent_ip, recent_starts = _recent_start_ip(starter)

    season_ip = None
    innings, starts = _num(starter.get("IP"), 0.0) or 0.0, _num(starter.get("GS"), 0.0) or 0.0
    if starts > 0:
        candidate = innings / starts
        # Only trust the season rate when it is physically plausible for a starter.
        if 3.0 <= candidate <= 7.0:
            season_ip, season_starts = candidate, starts
        else:
            season_starts = 0
    else:
        season_starts = 0

    observations = []
    if recent_ip is not None:
        observations.append((recent_ip, recent_starts * 1.5))   # recent workload leads
    if season_ip is not None:
        observations.append((season_ip, season_starts))

    if not observations:
        return LG["sp_ip"], 0
    weight_total = sum(w for _, w in observations) or 1.0
    blended = sum(v * w for v, w in observations) / weight_total
    regressed = (blended * weight_total + LG["sp_ip"] * REG["sp_ip"]) / (weight_total + REG["sp_ip"])
    return max(3.2, min(6.8, regressed)), weight_total


def _is_opener(opener_profile):
    if isinstance(opener_profile, pd.DataFrame):
        if opener_profile.empty:
            return False
        text = " ".join(str(v) for v in opener_profile.iloc[0].to_dict().values()).lower()
    elif isinstance(opener_profile, dict):
        text = " ".join(str(v) for v in opener_profile.values()).lower()
    else:
        return False
    return "opener" in text


def project_pitcher(side, payload):
    """Project the starting pitcher for one side. Returns None when there is no starter."""
    args, ctx = payload["report_args"], payload["advanced_context"]
    home = side == "home"

    starter = (args[2] if home else args[9]) or {}
    name = str(starter.get("Name") or "").strip()
    if not name or name.upper() == "TBD":
        return None

    team = args[15] if home else args[16]
    opponent = args[16] if home else args[15]
    opp_splits = args[26] if home else args[25]   # opposing lineup vs this starter's hand
    environment = ctx.get("environment") or {}
    park = _park_factors(environment)
    scorecard = ctx.get("scorecard")
    team_row = _scorecard_row(scorecard, team)
    opp_row = _scorecard_row(scorecard, opponent)

    # Season workload is the sample backing every rate below. Relief innings count here on
    # purpose: they are innings the arm actually threw, and they tell us as much about his
    # strikeout rate as a start does.
    season_innings = _num(starter.get("IP"), 0.0) or 0.0
    mlbam = _num(args[5] if home else args[12])
    ros = load_ros()[0]["P"].get(int(mlbam)) if mlbam else None

    def rate(season_key, ros_key, league_value, regression):
        """The starter's rate: the projection system's when it has him, else his own line
        shrunk toward league by how many innings back it."""
        projected = (ros or {}).get(ros_key)
        if projected is not None and ROS_MODE == "direct":
            return projected
        return _regress(_num(starter.get(season_key)), season_innings, league_value, regression)

    fip = rate("FIP", "fip", LG["fip"] + REPLACEMENT_PENALTY, REG_SP["fip"])
    era = rate("ERA", "era", LG["era"] + REPLACEMENT_PENALTY, REG_SP["era"])
    whip = rate("WHIP", "whip", 1.30, REG_SP["whip"])
    k_pct = rate("K%", "k_rate", LG["k_rate"], REG_SP["k"])
    bb_pct = rate("BB%", "bb_rate", 0.080, REG_SP["bb"])

    opener = _is_opener(ctx.get(f"{side}_opener_profile"))
    ip_proj, _ = _project_innings(starter)
    if opener:
        ip_proj = min(ip_proj, 2.0)

    supports, cautions = [], []

    # Strikeouts: the starter's own rate combined with the opposing lineup's K tendency
    # via log5, after regressing the lineup rate toward league for sample size.
    opp_k, opp_k_pa = _opponent_k_rate(opp_splits)
    opp_k_regressed = _regress(opp_k, opp_k_pa, LG["k_rate"], REG["opp_k"])
    k_rate = _clip(_log5(k_pct, opp_k_regressed, LG["k_rate"]), (0.08, 0.42))

    # On top of the lineup's overall K tendency: does it strike out more or less than usual
    # against *these* pitch shapes specifically? This is the one arsenal number that
    # survived a three-season test, and only at lineup level.
    opp_lineup = args[7] if home else args[0]
    opp_arsenal = ctx.get("away_batter_arsenal" if home else "home_batter_arsenal")
    arsenal_edge, arsenal_hitters = lineup_arsenal_k_edge(opp_arsenal, opp_lineup)
    if arsenal_edge:
        k_rate = _clip(k_rate + ARSENAL_K_PASSTHROUGH * arsenal_edge, (0.08, 0.42))
        # A lineup that cannot touch his shapes also lets him go deeper. Small, but it was
        # the second-strongest channel in the study and it compounds with the K rate.
        innings_shift = max(-ARSENAL_IP_CLIP,
                            min(ARSENAL_IP_CLIP, ARSENAL_IP_PASSTHROUGH * arsenal_edge))
        ip_proj = max(3.2, ip_proj + innings_shift)
        if opener:
            ip_proj = min(ip_proj, 2.0)
        if abs(arsenal_edge) >= 0.010:
            text = (f"{opponent} K {arsenal_edge * 100:+.1f} pts vs his shapes "
                    f"({arsenal_hitters} hitters)")
            (supports if arsenal_edge > 0 else cautions).append(text)

    if abs(k_rate - k_pct) >= 0.012:
        if opp_k_regressed > LG["k_rate"]:
            supports.append(f"{opponent} K-prone ({opp_k * 100:.1f}% K)")
        else:
            cautions.append(f"{opponent} contact ({opp_k * 100:.1f}% K)")
    if k_pct >= 0.26:
        supports.append(f"{k_pct * 100:.1f}% K rate")
    elif k_pct <= 0.18:
        cautions.append(f"{k_pct * 100:.1f}% K rate")

    batters_faced = ip_proj * 3 + whip * ip_proj
    strikeouts = batters_faced * k_rate
    walks = batters_faced * bb_pct
    hits = max(0.0, whip * ip_proj - walks)
    hbp = batters_faced * 0.010

    # Earned runs scale the starter's own FIP by the offense he faces and the park.
    opp_runs = _num(opp_row.get("Exp Runs"), LG["runs_per_team_game"]) or LG["runs_per_team_game"]
    offense_factor = _clip(1.0 + (opp_runs - LG["runs_per_team_game"]) * 0.085, (0.82, 1.22))
    run_rate = ERA_BLEND["fip"] * fip + ERA_BLEND["era"] * era
    earned_runs = (run_rate * offense_factor * _clip(park["run"], CLIP["park_run"]) / 9.0) * ip_proj
    if abs(era - fip) >= 1.0:
        gap = f"ERA {era:.2f} vs FIP {fip:.2f}"
        (supports if era < fip else cautions).append(gap)

    # Win: needs the team to win and the starter to go five.
    qualify = 1.0 / (1.0 + math.exp(-(ip_proj - 5.05) * 2.4))
    win_probability = _win_probability(team_row) * qualify * 0.90

    events = {
        "IP": ip_proj, "K": strikeouts, "W": win_probability,
        "ER": earned_runs, "H": hits, "BB": walks, "HBP": hbp,
    }
    points = pitcher_points(events)

    # Pitcher variance is dominated by whether the outing survives -- earned runs swing
    # hardest, strikeouts add the upside on top.
    variance = (
        PITCHER_BASE_VAR
        + strikeouts * (DK_PITCHER["K"] ** 2)
        + earned_runs * (DK_PITCHER["ER"] ** 2) * ER_OVERDISPERSION
        + (DK_PITCHER["W"] ** 2) * win_probability * (1 - win_probability)
    )
    ceiling = points + CEILING_Z * math.sqrt(variance)

    if offense_factor <= 0.95:
        supports.append(f"{opponent} proj {opp_runs:.1f} runs")
    elif offense_factor >= 1.08:
        cautions.append(f"{opponent} proj {opp_runs:.1f} runs")
    if ip_proj >= 6.0:
        supports.append(f"{ip_proj:.1f} IP workload")
    elif ip_proj <= 4.6:
        cautions.append(f"short leash ({ip_proj:.1f} IP)")
    if opener:
        cautions.append("opener — bulk arm follows")
    if park["run"] <= 0.95:
        supports.append(f"park runs {park['run']:.2f}")
    elif park["run"] >= 1.06:
        cautions.append(f"park runs {park['run']:.2f}")

    return {
        "Name": name,
        "Team": team,
        "Opp": opponent,
        "Slot": 0,
        "Pos": "P",
        "Bats": "",
        "MLBAM": _num(args[5] if home else args[12]),
        "Type": "P",
        "Opp SP": "",
        "Opp SP Hand": str(starter.get("Throws") or "")[:1].upper(),   # his own hand
        "Lineup": "Confirmed",
        "PA": round(batters_faced, 1),
        "Team Runs": round(_num(team_row.get("Exp Runs"), LG["runs_per_team_game"]) or 0.0, 2),
        "Proj": round(points, 2),
        "Ceiling": round(ceiling, 2),
        "Floor": round(floor_points(points, "P"), 2),
        "Bust%": round(bust_probability(points, "P") * 100, 1),
        "IP": round(ip_proj, 2),
        "K": round(strikeouts, 2),
        "W%": round(win_probability * 100, 1),
        "ER": round(earned_runs, 2),
        "Matchup": round(offense_factor, 3),
        "Arsenal K Edge": round(arsenal_edge * 100, 2) if arsenal_edge else 0.0,
        # See the hitter block: expected events, for the simulator.
        "E_BF": round(batters_faced, 4),
        "E_IP": round(ip_proj, 4),
        "E_K": round(strikeouts, 4),
        "E_BB": round(walks, 4),
        "E_H": round(hits, 4),
        "E_HBP": round(hbp, 4),
        "E_ER": round(earned_runs, 4),
        "E_W": round(win_probability, 4),
        "E_KRATE": round(k_rate, 4),
        "Supports": supports,
        "Cautions": cautions,
    }


def project_game(payload):
    """All projectable players from one cached game payload."""
    # Row indexes are built per frame and only ever reused within one game, so they are
    # dropped here rather than allowed to accumulate across a slate.
    _INDEX_CACHE.clear()
    rows = []
    for side in ("away", "home"):
        pitcher = project_pitcher(side, payload)
        if pitcher:
            rows.append(pitcher)
        rows.extend(project_hitters(side, payload))
    return rows
