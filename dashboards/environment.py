"""Park, weather and defence — the conditions a batted ball lands in.

Three things decide what a fly ball becomes: how far the park plays, what the air is doing,
and who is standing under it. The report caches all three per game and the workbook prints
them in separate tables, so the interaction — the one that matters — was never visible.

Nothing here is a new model. The park factors are the pipeline's, and the weather term is
`dfs.projections._weather_hr_factor`, the same function the projections themselves use. The
only thing this module adds is the **arithmetic of putting them next to a pitcher's contact
profile**, and that arithmetic is stated in the open rather than buried in a score.
"""

import pandas as pd
import streamlit as st

from dashboards import data

#: League batted-ball baselines, the reference the report itself quotes.
LEAGUE_FB = 25.0
LEAGUE_GB = 44.0

#: Below this the environment is called neutral rather than given a direction. The weather
#: term is deliberately small — a couple of percent per ten degrees — so a 1% deviation is
#: not a thing to act on and an arrow drawn for it would overstate the model.
NEUTRAL_BAND = 0.03


def hr_environment(payload):
    """Park and weather as one home-run multiplier, with the parts kept visible.

    Returned separately as well as combined, because they fail differently: the park factor
    is a season-long measurement and the weather term is a small live nudge that goes to
    exactly 1.0 under a closed roof. A combined number alone hides which one is talking.
    """
    from dfs import projections

    environment = (payload.get("advanced_context") or {}).get("environment") or {}
    park = environment.get("park") or {}
    forecast = environment.get("forecast") or {}
    weather = environment.get("weather") or {}

    park_hr = pd.to_numeric(pd.Series([park.get("HR")]), errors="coerce").iloc[0]
    park_runs = pd.to_numeric(pd.Series([park.get("Runs")]), errors="coerce").iloc[0]
    park_hr = 1.0 if pd.isna(park_hr) else float(park_hr)
    park_runs = 1.0 if pd.isna(park_runs) else float(park_runs)

    weather_hr = float(projections._weather_hr_factor(environment))
    temp, wind, enclosed = projections._weather_inputs(environment)

    return {
        "park": environment.get("park_name") or environment.get("venue") or "",
        "park_hr": park_hr,
        "park_runs": park_runs,
        "weather_hr": weather_hr,
        "hr_env": park_hr * weather_hr,
        "temp": temp,
        "wind": wind or (weather.get("wind") or forecast.get("wind") or ""),
        "roof": str(forecast.get("roof") or "").strip(),
        "enclosed": bool(enclosed),
        "condition": str(weather.get("condition") or ""),
    }


def handed_park(payload, hand):
    """The handed park factors, `RHB` or `LHB`, as plain numbers.

    `2B` and `R` are the ones a ground-ball or gap profile lives on, and they diverge from
    the HR factor more than people expect — a park can suppress home runs and inflate
    doubles at the same time, and the single park number hides that entirely.
    """
    environment = (payload.get("advanced_context") or {}).get("environment") or {}
    handed = (environment.get("park_handed") or {}).get(hand) or {}
    out = {}
    for key in ("Park Factor", "R", "HR", "H", "1B", "2B", "3B", "OBP", "SLG"):
        value = pd.to_numeric(pd.Series([handed.get(key)]), errors="coerce").iloc[0]
        # These are published on a 100-is-neutral scale; carried through as-is so the
        # numbers on screen are the ones the source publishes.
        out[key] = None if pd.isna(value) else float(value)
    return out


def direction(value, neutral=NEUTRAL_BAND):
    """`up`, `down` or `neutral` for a multiplier around 1.0.

    A three-state label rather than a signed number, so the arrow and the words agree and
    a 1% wobble is called what it is: nothing.
    """
    if value is None or pd.isna(value):
        return "neutral"
    if value >= 1.0 + neutral:
        return "up"
    if value <= 1.0 - neutral:
        return "down"
    return "neutral"


def hr_leverage(fly_ball_pct, hr_env, league_fb=LEAGUE_FB):
    """How much this arm is *exposed* to the home-run environment.

    ``(FB% / league FB%) × (hr_env − 1)``, in percentage points of the environment's nudge.

    **This is a heuristic, not a projection.** It says a fly-ball arm in a hot, short park
    is more exposed than a sinkerballer in the same park, which is true and is the whole
    reason to put these two numbers on one chart; it does not claim to know by how many
    home runs. The projections model that properly. Read the sign and the ordering, not
    the magnitude.
    """
    if fly_ball_pct is None or pd.isna(fly_ball_pct) or hr_env is None or pd.isna(hr_env):
        return float("nan")
    if not league_fb:
        return float("nan")
    return (float(fly_ball_pct) / league_fb) * (float(hr_env) - 1.0) * 100.0


def _batted_row(payload, side, source):
    if source == "starter":
        _, season = data.starter_batted(payload, side)
        return season or {}
    frame = data.bullpen_batted(payload, side)
    if frame.empty or "is_total" not in frame.columns:
        return {}
    total = frame[frame["is_total"]]
    row = total.iloc[0] if not total.empty else None
    if row is None:
        return {}
    return {key: row.get(key) for key in ("BIP", "GB%", "FB%", "LD%", "HH%", "Brl%")}


@st.cache_data(max_entries=8, show_spinner="Reading cached games…")
def staff_environment(date, cache_dir=data.CACHE_DIR):
    """One row per pitching staff on a slate: contact profile against its conditions.

    Starters and bullpens are both included and labelled, because the question is asked of
    both and a bullpen's fly-ball rate is measured over far more innings than one start.
    """
    games = data.list_games(cache_dir)
    if games.empty:
        return pd.DataFrame()

    rows = []
    for _, game in games[games["date"] == str(date)].iterrows():
        meta = game.to_dict()
        payload = data.load_payload(meta["path"])
        conditions = hr_environment(payload)
        defence_by_side = {}
        for side in ("away", "home"):
            defence = data.context_frame(payload, f"{side}_team_defense")
            defence_by_side[side] = defence.iloc[0].to_dict() if not defence.empty else {}

        for side in ("away", "home"):
            team = meta["away"] if side == "away" else meta["home"]
            opponent = meta["home"] if side == "away" else meta["away"]
            hand = "RHB"        # handed factors are looked up per view, not fixed here
            for source in ("starter", "bullpen"):
                batted = _batted_row(payload, side, source)
                if not batted:
                    continue
                fly = pd.to_numeric(pd.Series([batted.get("FB%")]), errors="coerce").iloc[0]
                ground = pd.to_numeric(pd.Series([batted.get("GB%")]),
                                       errors="coerce").iloc[0]
                line = pd.to_numeric(pd.Series([batted.get("LD%")]), errors="coerce").iloc[0]
                name = (data.context_frame(payload, f"{side}_pitcher_hand_splits")
                        .get("Pitcher", pd.Series(dtype=str)).dropna())
                rows.append({
                    "game": meta["label"], "date": meta["date"], "side": side,
                    "team": team, "opponent": opponent, "unit": source.capitalize(),
                    "name": (str(name.iloc[0]) if source == "starter" and len(name)
                             else f"{team} bullpen"),
                    "park": conditions["park"],
                    "park_hr": conditions["park_hr"],
                    "park_runs": conditions["park_runs"],
                    "weather_hr": conditions["weather_hr"],
                    "hr_env": conditions["hr_env"],
                    "temp": conditions["temp"], "wind": conditions["wind"],
                    "roof": conditions["roof"], "enclosed": conditions["enclosed"],
                    "condition": conditions["condition"],
                    "BIP": batted.get("BIP"),
                    "FB%": fly, "GB%": ground, "LD%": line,
                    "GB+LD%": (ground + line) if pd.notna(ground) and pd.notna(line)
                              else float("nan"),
                    "HH%": batted.get("HH%"), "Brl%": batted.get("Brl%"),
                    "hr_leverage": hr_leverage(fly, conditions["hr_env"]),
                    "impact": direction(conditions["hr_env"]),
                    "park_2b": handed_park(payload, hand).get("2B"),
                    "park_r": handed_park(payload, hand).get("R"),
                    # The defence standing behind this staff is its own club's.
                    "hits_saved": pd.to_numeric(
                        pd.Series([defence_by_side[side].get("Hits Saved/G")]),
                        errors="coerce").iloc[0],
                    "defense_grade": str(defence_by_side[side].get("Grade", "") or ""),
                    "of_saved": pd.to_numeric(
                        pd.Series([defence_by_side[side].get("OF Saved/G")]),
                        errors="coerce").iloc[0],
                    "if_saved": pd.to_numeric(
                        pd.Series([defence_by_side[side].get("IF Saved/G")]),
                        errors="coerce").iloc[0],
                })
    return pd.DataFrame(rows)


def conditions_line(row):
    """A one-line reading of the conditions, in words rather than multipliers."""
    if row is None:
        return ""
    parts = [f"{row.get('park', '')}".strip()]
    if row.get("enclosed"):
        parts.append("roof closed — weather neutral")
    else:
        if pd.notna(row.get("temp")) and row.get("temp") is not None:
            parts.append(f"{float(row['temp']):.0f}°F")
        if row.get("wind"):
            parts.append(str(row["wind"]))
    parts.append(f"park HR {row.get('park_hr', 1):.2f}")
    if not row.get("enclosed"):
        parts.append(f"weather HR {row.get('weather_hr', 1):.2f}")
    return " · ".join(p for p in parts if p)
