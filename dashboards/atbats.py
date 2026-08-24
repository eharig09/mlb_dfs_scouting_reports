"""The plate appearances behind an arsenal number.

`Arsenal OPS` is a line built from Statcast pitches: every pitch a hitter has seen from the
starter's hand, narrowed to the starter's actual pitch *shapes* — the same pitch types, in a
velocity band of ±2.5 mph and a spin band of ±300 rpm. That filter is the whole claim, and
until now it was invisible: a reader saw `.929 in 70 PA` with no way to ask which 70.

Why this imports the pipeline
-----------------------------
The filter is reproduced by **calling `scouting_report`'s own functions**, not by
reimplementing them here. A second implementation would drift, and the failure mode is
specific and bad: a drill-down that shows a *different* set of at-bats than the number was
computed from, which looks like evidence and is the opposite. The import costs ~4s and
happens only when someone actually drills in.

`_filter_to_arsenal_shape` has one behaviour worth knowing: when the shape-matched rows fall
below `min_pa` it **falls back to all pitches of those types**, ignoring velocity and spin.
So a thin sample is silently measured on a looser filter, and this module reports which of
the two happened rather than letting a reader assume the tight one.
"""

import numpy as np
import pandas as pd
import streamlit as st

#: Columns worth showing for a single plate appearance, in reading order.
AT_BAT_COLUMNS = ["game_date", "pitcher_name", "p_throws", "pitch_type", "release_speed",
                  "release_spin_rate", "events", "description", "launch_speed",
                  "launch_angle", "estimated_woba_using_speedangle", "balls", "strikes",
                  "inning"]

#: A plate appearance is the pitch that ended it — the row carrying an `events` value.
#: Every other pitch in the at-bat is context, not an outcome.
PA_MARKER = "events"


@st.cache_data(ttl="12h", max_entries=2, show_spinner="Loading pitch data…")
def _season_pitches(start, end):
    """Every Statcast pitch in a range, from the chunked local cache.

    ~543k rows and 0.62 GB for a season, about 2.6s cold and free thereafter. Cached
    separately from anything else because every drill-down on every page shares it.
    """
    from utils import statcast_cache

    return statcast_cache.load_statcast_range(start, end)


def _card_for(payload, name):
    """The lineup card a hitter is actually on, and his row in it.

    `report_args` holds **both** clubs' cards at slots 0 and 7, and their order is not
    away-first. Resolving the card and the batter id in one place is what keeps them
    consistent: an earlier version looked the id up across both cards but took the lineup
    from whichever card came first, so a Washington hitter was measured against the Texas
    lineup's sample floor and every number silently fell back to a looser filter.
    """
    from dashboards import salaries

    key = salaries.name_key(name)
    args = payload.get("report_args") or ()
    for index in (0, 7):
        if len(args) <= index:
            continue
        card = args[index]
        if not isinstance(card, pd.DataFrame) or card.empty or "ID" not in card.columns:
            continue
        match = card[card["Name"].map(salaries.name_key) == key]
        if not match.empty:
            return card, match.iloc[0]
    return None, None


def _card_ids(card):
    """Every batter id on a card, which is what the shape filter's sample floor counts."""
    if card is None or "ID" not in card.columns:
        return []
    return pd.to_numeric(card["ID"], errors="coerce").dropna().astype(int).tolist()


def arsenal_at_bats(payload, meta, name, team, season_start=None):
    """The plate appearances behind this hitter's arsenal line, plus how they were chosen.

    Returns `(at_bats, context)`. `context` carries the filter that produced them — the
    pitch types and their weights, the hand, the velocity and spin bands, whether the tight
    shape filter survived its sample floor — so a reader can see the claim, not just its
    result.
    """
    # `scouting_report` is imported here rather than at module scope for two reasons. It
    # costs about four seconds and pulls matplotlib, fpdf and pybaseball, so a reader who
    # never drills in never pays for it -- and the hosted build installs only the six
    # packages the dashboard itself needs, so on Render this import is *expected* to fail.
    # Calling the report's own filters is the whole point (a second implementation would
    # drift from the published number), so there is nothing to fall back to: the panel says
    # so and shows nothing rather than showing at-bats it derived some other way.
    try:
        import scouting_report as sr
    except ImportError as error:
        return pd.DataFrame(), {"unavailable": (
            "The at-bat drill-down needs the report pipeline, which is not installed on "
            f"this deployment ({error.name}). It works in the local checkout."
        )}
    from dashboards import drill

    side = drill.side_for_team(meta, team)
    if side is None:
        return pd.DataFrame(), {}
    facing = "home" if side == "away" else "away"

    arsenal = _starter_arsenal(payload, facing)
    if arsenal is None or arsenal.empty:
        return pd.DataFrame(), {}

    card, card_row = _card_for(payload, name)
    if card_row is None:
        return pd.DataFrame(), {}
    batter_id = pd.to_numeric(pd.Series([card_row.get("ID")]), errors="coerce").iloc[0]
    if pd.isna(batter_id):
        return pd.DataFrame(), {}

    starter_arsenal = arsenal
    weights = sr._arsenal_pitch_weights(starter_arsenal, top_n=3)
    if not weights:
        return pd.DataFrame(), {}
    top_pitches = [pitch for pitch, _ in weights]
    traits = sr._arsenal_shape_traits(starter_arsenal, top_n=len(top_pitches))

    throws = drill.starter_name(payload, facing)
    hand = _starter_hand(payload, facing)

    # The window ends the day **before** the game, exactly as `_pregame_end_date` sets it.
    # Including the game date itself adds the at-bats the number was predicting, which is
    # both lookahead and the reason an earlier attempt ran two or three at-bats long.
    season_start = season_start or sr._season_start_date(int(str(meta["date"])[:4]))
    end = sr._pregame_end_date(str(meta["date"])) or str(meta["date"])
    pitches = _season_pitches(season_start, end)
    if pitches is None or pitches.empty or "batter" not in pitches.columns:
        return pd.DataFrame(), {}

    # The pipeline's chain, in its exact order. Three details decide whether the rebuilt
    # rows are the ones that were counted, and getting any of them wrong produced roughly
    # three times too many at-bats on the first attempt:
    #
    # 1. `load_statcast_detail` keeps **only PA-ending rows** (`events` not null). The
    #    arsenal line is built on completed plate appearances, not on every pitch seen.
    # 2. `_filter_to_arsenal_shape` runs on the **whole lineup at once**, and its sample
    #    floor is `max(30, batters * 4)`. Applied per batter, the floor fails every time and
    #    it silently falls back to the loose pitch-type filter.
    # 3. Only then is the individual batter selected out of the filtered frame.
    completed = pitches[pitches[PA_MARKER].notna()]
    lineup_ids = _card_ids(card)

    lineup_rows = completed[completed["batter"].isin(lineup_ids)].copy()
    if hand and "p_throws" in lineup_rows.columns:
        lineup_rows = lineup_rows[lineup_rows["p_throws"].astype(str).str.strip().eq(hand)]
    if lineup_rows.empty:
        return pd.DataFrame(), {}

    shaped_lineup = sr._filter_to_arsenal_shape(
        lineup_rows, starter_arsenal, top_pitches,
        min_pa=max(30, len(lineup_ids) * 4))
    tight = len(shaped_lineup) < len(lineup_rows[lineup_rows["pitch_type"].isin(top_pitches)])

    seen = lineup_rows[lineup_rows["batter"] == int(batter_id)]
    shaped = shaped_lineup[shaped_lineup["batter"] == int(batter_id)].copy()
    if shaped.empty:
        return pd.DataFrame(), {}

    line = sr._events_to_batting_line(shaped)
    metrics = sr._weighted_arsenal_metrics(shaped, weights)

    at_bats = shaped.sort_values("game_date", ascending=False)
    if "pitcher" in at_bats.columns:
        at_bats["pitcher_name"] = _pitcher_names(pitches, at_bats["pitcher"])

    context = {
        "batter_id": int(batter_id),
        "starter": throws,
        "hand": hand,
        "weights": weights,
        "traits": traits,
        "shape_filter": "velocity and spin matched" if tight else
                        "pitch type only (too few shape-matched pitches)",
        "pitches_seen_vs_hand": int(len(seen)),
        "pitches_in_arsenal": int(len(shaped)),
        "plate_appearances": int(line.get("PA") or 0),
        "at_bats": int(line.get("AB") or 0),
        "ops": metrics.get("OPS"),
        "xwoba": metrics.get("xwOBA"),
        "bats": (card_row.get("Bats") if card_row is not None else None),
        "spot": (card_row.get("Spot") if card_row is not None else None),
    }
    return at_bats.reindex(columns=[c for c in AT_BAT_COLUMNS if c in at_bats.columns]), context


#: Where each side's true starter-arsenal table lives in `report_args`. The context frame
#: named `*_arsenal_matchup` looks like the right input and is not: it carries usage and
#: results but **no `release_speed` or `release_spin_rate`**, so `_arsenal_shape_traits`
#: returns traits with no bands, the velocity/spin mask never applies, and the filter
#: degenerates to bare pitch types. That silently produced ~2.7x too many at-bats — a
#: drill-down that looked like evidence for a number it did not belong to.
ARSENAL_SLOT = {"home": 4, "away": 11}


def _starter_arsenal(payload, side):
    """The starter's arsenal with pitch shapes, as the pipeline's own filter expects it."""
    args = payload.get("report_args") or ()
    index = ARSENAL_SLOT.get(side)
    if index is None or len(args) <= index:
        return None
    frame = args[index]
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    if not {"Pitch", "Usage %"}.issubset(frame.columns):
        return None
    return frame


def _starter_hand(payload, side):
    """The starter's throwing hand, from whichever cached frame carries it."""
    from dashboards import data

    similar = data.context_frame(payload, f"{side}_pitcher_similar")
    if not similar.empty and "Throws" in similar.columns:
        hands = similar["Throws"].dropna().astype(str).str.strip().str.upper()
        if not hands.empty:
            return hands.iloc[0]
    return None


def _pitcher_names(pitches, ids):
    """Map pitcher ids to names using the pitch table's own `player_name` column.

    `player_name` is the *pitcher* on a Statcast row, so the lookup is built from rows where
    the id matches rather than from a separate roster call.
    """
    if "player_name" not in pitches.columns or "pitcher" not in pitches.columns:
        return pd.Series("", index=ids.index)
    lookup = (pitches.dropna(subset=["pitcher", "player_name"])
              .drop_duplicates("pitcher").set_index("pitcher")["player_name"])
    return ids.map(lookup).fillna("")


def outcome_summary(at_bats):
    """What the at-bats actually produced, most common first."""
    if at_bats is None or at_bats.empty or PA_MARKER not in at_bats.columns:
        return pd.DataFrame()
    counts = (at_bats[PA_MARKER].value_counts().rename_axis("Result")
              .reset_index(name="Count"))
    counts["Share %"] = counts["Count"] / counts["Count"].sum() * 100
    return counts


def by_pitch_type(at_bats):
    """The same at-bats split by which pitch ended them.

    The arsenal number is a *weighted* blend across pitch types, so seeing it split is how a
    reader finds out whether one pitch is carrying the whole line.
    """
    if at_bats is None or at_bats.empty or "pitch_type" not in at_bats.columns:
        return pd.DataFrame()
    frame = at_bats.copy()
    frame["hit"] = frame[PA_MARKER].isin(["single", "double", "triple", "home_run"])
    frame["strikeout"] = frame[PA_MARKER].astype(str).str.startswith("strikeout")
    aggregates = {"PA": (PA_MARKER, "size"), "Hits": ("hit", "sum"),
                  "K": ("strikeout", "sum")}
    for column, label in (("release_speed", "Velo"), ("launch_speed", "Exit velo"),
                          ("estimated_woba_using_speedangle", "xwOBA")):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            aggregates[label] = (column, "mean")
    out = frame.groupby("pitch_type", as_index=False).agg(**aggregates)
    out["Hit %"] = np.where(out["PA"] > 0, out["Hits"] / out["PA"] * 100, np.nan)
    return out.sort_values("PA", ascending=False)
