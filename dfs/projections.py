"""Turn a cached scouting-report payload into DK-point projections.

Design: every player gets baseline event rates from their season line (regressed to
league), then a set of named multiplicative matchup factors. Keeping the factors named
and separate is the point -- they are what the board shows as the reason a player is
flagged, and they are what you tune when the model feels wrong.
"""

import math
import re

import pandas as pd

from .scoring import DK_HITTER, DK_PITCHER, hitter_points, pitcher_points

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

# Regression strength (in PA / batters faced) for each blended source.
REG = {
    "season": 200,
    "platoon": 150,
    "arsenal": 70,
    "sp_ip": 6,        # in starts
    "opp_k": 120,
}

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
# Game-level context
# ---------------------------------------------------------------------------

def _park_factors(environment):
    park = environment.get("park") or {}
    return {
        "hr": _num(park.get("HR"), 1.0) or 1.0,
        "run": _num(park.get("Runs"), 1.0) or 1.0,
    }


def _weather_hr_factor(environment):
    """Temperature and wind nudge on home runs. Deliberately small."""
    weather = environment.get("weather") or {}
    factor = 1.0
    temp = _num(weather.get("temp"))
    if temp is not None:
        factor *= 1.0 + (temp - 72.0) * 0.0022      # ~+2% per 10 degrees

    wind = str(weather.get("wind") or "")
    speed = _num((re.match(r"\s*(\d+)", wind) or [None, None])[1] if re.match(r"\s*(\d+)", wind) else None, 0.0) or 0.0
    direction = wind.lower()
    if "out to" in direction:
        factor *= 1.0 + min(speed, 20) * 0.0055
    elif "in from" in direction:
        factor *= 1.0 - min(speed, 20) * 0.0050
    return _clip(factor, CLIP["weather"])


def _scorecard_row(scorecard, team):
    if not isinstance(scorecard, pd.DataFrame) or scorecard.empty:
        return {}
    match = scorecard[scorecard["Team"].astype(str) == str(team)]
    return match.iloc[0].to_dict() if not match.empty else {}


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
    if not isinstance(splits_df, pd.DataFrame) or splits_df.empty or "Name" not in splits_df:
        return {}
    match = splits_df[splits_df["Name"].astype(str) == str(name)]
    if match.empty:
        return {}
    row = match.iloc[0].to_dict()
    # Columns are suffixed with the hand faced ("OPS vs R"); strip it for uniform access.
    return {re.sub(r"\s+vs\s+[LR]$", "", str(key)): value for key, value in row.items()}


def _arsenal_row(arsenal_df, name):
    if not isinstance(arsenal_df, pd.DataFrame) or arsenal_df.empty or "Name" not in arsenal_df:
        return {}
    match = arsenal_df[arsenal_df["Name"].astype(str) == str(name)]
    return match.iloc[0].to_dict() if not match.empty else {}


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


def _hitter_rate_line(row, platoon):
    """Blend season and platoon rate stats, regressed to league."""
    season_pa = _num(row.get("PA"), 0.0) or 0.0
    platoon_pa = _num(platoon.get("PA"), 0.0) or 0.0
    weight = platoon_pa / (platoon_pa + REG["platoon"]) if platoon_pa else 0.0

    def blend(season_key, platoon_key, league_key):
        season_value = _num(row.get(season_key))
        platoon_value = _num(platoon.get(platoon_key))
        base = _regress(season_value, season_pa, LG[league_key], REG["season"])
        if platoon_value is None:
            return base
        return base * (1 - weight) + _regress(platoon_value, platoon_pa, LG[league_key], REG["season"]) * weight

    avg = blend("AVG", "AVG", "avg")
    iso = blend("ISO", "ISO", "iso")
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
    hr_per_ab = _regress(hr / ab if ab > 0 else None, ab, LG["hr_per_ab"], 250)

    return {"avg": avg, "iso": iso, "bb_rate": bb_rate, "k_rate": k_rate, "hr_per_ab": hr_per_ab}


def _steal_rate(baserunning_df, name, season_pa):
    """Season steals per PA, regressed. Speed is sticky, so regression is light."""
    if not isinstance(baserunning_df, pd.DataFrame) or baserunning_df.empty or "Name" not in baserunning_df:
        return 0.0
    match = baserunning_df[baserunning_df["Name"].astype(str) == str(name)]
    if match.empty or not season_pa:
        return 0.0
    steals = _num(match.iloc[0].get("SB"), 0.0) or 0.0
    return (steals / (season_pa + 120)) if season_pa > 0 else 0.0


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

    exp_runs = _num(team_row.get("Exp Runs"), LG["runs_per_team_game"]) or LG["runs_per_team_game"]
    opp_starter = (args[9] if home else args[2]) or {}
    opp_starter_name = str(opp_starter.get("Name") or "TBD")

    composite = ctx.get("hitter_composite")
    arsenal_df = ctx.get("home_batter_arsenal" if home else "away_batter_arsenal")

    game_ctx = {
        "opp_sp_fip": _num(opp_starter.get("FIP")),
        "opp_bullpen_factor": _bullpen_factor(opp_bullpen_df),
        "park_run": park["run"],
    }
    pa_mult = _team_pa_multiplier(exp_runs, home, _win_probability(home_row))
    weather_hr = _weather_hr_factor(environment)

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
        rates = _hitter_rate_line(row, platoon)

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
            supports.append(f"wind {str((environment.get('weather') or {}).get('wind') or '').strip()}")
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
            "Lineup": str(row.get("Lineup Confidence") or ""),
            "PA": round(pa, 2),
            "Team Runs": round(exp_runs, 2),
            "Proj": round(points, 2),
            "Ceiling": round(ceiling, 2),
            "Floor": round(floor_points(points, "H"), 2),
            "Bust%": round(bust_probability(points, "H") * 100, 1),
            "HR": round(hr, 3),
            "SB": round(steals, 3),
            "Matchup": round(total, 3),
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

    fip = _num(starter.get("FIP"), LG["fip"]) or LG["fip"]
    era = _num(starter.get("ERA"), fip) or fip
    whip = _num(starter.get("WHIP"), 1.30) or 1.30
    k_pct = _num(starter.get("K%"), LG["k_rate"]) or LG["k_rate"]
    bb_pct = _num(starter.get("BB%"), 0.080) or 0.080

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
        "Supports": supports,
        "Cautions": cautions,
    }


def project_game(payload):
    """All projectable players from one cached game payload."""
    rows = []
    for side in ("away", "home"):
        pitcher = project_pitcher(side, payload)
        if pitcher:
            rows.append(pitcher)
        rows.extend(project_hitters(side, payload))
    return rows
