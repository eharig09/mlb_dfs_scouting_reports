"""Cached access to the scouting report's own game payloads.

The MLB pipeline already writes a complete payload per game to `.cache/report_data/*.pkl`
— that is what lets `render_report_from_cache` rebuild a workbook in about a second. The
dashboard reads the same files, so it never re-runs the pipeline and never touches the
network.

Two things follow from that and shape this module:

* **The payloads are plain dicts of DataFrames.** Loading one does not import
  `scouting_report`, so the app starts instantly instead of pulling in matplotlib, fpdf and
  pybaseball.
* **There are hundreds of them** — 325 games over 27 days at the time of writing — so the
  app can do the thing the workbook cannot: compare across games and dates. The expensive
  step is reading many pickles, which is why the slate-level loader is cached separately
  from the single-game one.

Caching follows the project rule for cached game data: a completed game's payload never
changes, so entries are held rather than expired, and `max_entries` is what bounds memory.
"""

import glob
import os
import pickle
import re

import pandas as pd
import streamlit as st

CACHE_DIR = os.path.join(".cache", "report_data")

# `2026-08-20_WSH_TEX.pkl`, or `..._g2.pkl` for the second of a doubleheader.
_STEM = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<away>[A-Z]+)_(?P<home>[A-Z]+)(?:_g(?P<game>\d))?$")

SIGNAL_ORDER = ["Priority", "Watch", "Neutral", "Fade"]


def _parse(path):
    match = _STEM.match(os.path.splitext(os.path.basename(path))[0])
    if not match:
        return None
    parts = match.groupdict()
    return {
        "path": path,
        "date": parts["date"],
        "away": parts["away"],
        "home": parts["home"],
        "game": int(parts["game"] or 1),
        "label": f"{parts['away']} @ {parts['home']}"
              + (f" (g{parts['game']})" if parts["game"] else ""),
    }


@st.cache_data(ttl="30m", max_entries=4, show_spinner=False)
def list_games(cache_dir=CACHE_DIR):
    """Every cached game, newest first. Short TTL so a fresh run shows up without a restart."""
    rows = [_parse(p) for p in glob.glob(os.path.join(cache_dir, "*.pkl"))]
    rows = [r for r in rows if r]
    if not rows:
        return pd.DataFrame(columns=["path", "date", "away", "home", "game", "label"])
    return (pd.DataFrame(rows).sort_values(["date", "label"], ascending=[False, True])
            .reset_index(drop=True))


@st.cache_data(max_entries=64, show_spinner=False)
def load_payload(path):
    """One game's payload. A completed game never changes, so this is held, not expired."""
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _numeric(frame, column):
    """A real numeric Series even when the column is absent.

    `frame.get(name)` returns a bare NaN for a missing column — no index, nothing to
    `.fillna` — which is the recurring crash in this codebase.
    """
    if column not in frame.columns:
        return pd.Series(float("nan"), index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


NUMERIC_COMPOSITE = ["Season AB", "Season OPS", "Season HR", "Platoon AB", "Platoon OPS",
                     "Platoon HR", "Arsenal AB", "Arsenal OPS", "Arsenal HR",
                     "Similar AB", "Similar OPS", "Similar HR", "BvP AB", "BvP OPS",
                     "BvP HR", "Composite", "Off Szn", "Off L28"]


def hitters(payload, meta):
    """The hitter composite for one game, typed and tagged with which game it came from."""
    frame = payload.get("advanced_context", {}).get("hitter_composite")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    for column in NUMERIC_COMPOSITE:
        out[column] = _numeric(out, column)
    out["Signal"] = out.get("Signal", pd.Series("", index=out.index)).fillna("").astype(str)
    out["date"] = meta["date"]
    out["game"] = meta["label"]
    out["opponent"] = [meta["home"] if t == meta["away"] else meta["away"]
                       for t in out.get("Team", pd.Series("", index=out.index))]
    return rounded(out)


@st.cache_data(max_entries=8, show_spinner="Reading cached games…")
def hitters_for_date(date, cache_dir=CACHE_DIR):
    """Every hitter on a date's slate, one row per player-game.

    Reading a whole slate is ~15 pickles, which is why it is cached apart from the
    single-game loader — the cross-game views would otherwise re-read them on every widget
    change.
    """
    games = list_games(cache_dir)
    games = games[games["date"] == date]
    frames = []
    for meta in games.to_dict("records"):
        try:
            payload = load_payload(meta["path"])
            board = attach_arsenal_measures(hitters(payload, meta), payload, meta)
            frames.append(attach_card_measures(board, payload, meta))
        except Exception:
            continue
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def scorecard(payload):
    frame = payload.get("advanced_context", {}).get("scorecard")
    return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()


def arrow_safe(frame):
    """Make a payload frame renderable without changing what it says.

    The pipeline formats its context frames for a workbook, where a missing value is the
    empty string. That leaves an object column carrying numbers and `""` together, which
    Arrow refuses outright — `Could not convert '' with type str: tried to convert to
    double` — and Streamlit then renders **nothing at all** where the table should be. It
    affected 55 of the 2,126 cached frames, `sp_rest_splits.ERA` and every column of
    `relevant_games` among them.

    A column is converted only when *every* non-blank value in it parses as a number, so a
    blank becomes a real null and any mixed text column is left exactly as it is. Nothing
    is reformatted or rounded: this changes the type a value is carried in, never the value.
    """
    if frame is None or frame.empty:
        return frame
    out = frame.copy()
    for column in out.columns:
        if out[column].dtype != object:
            continue
        values = out[column]
        text = values.astype("string").fillna("").str.strip()
        filled = values[text != ""]
        if filled.empty:
            continue
        if pd.to_numeric(filled, errors="coerce").notna().all():
            out[column] = pd.to_numeric(values.where(text != ""), errors="coerce")
        else:
            out[column] = text
    return out


#: Baseball quotes two kinds of number and mixing them looks wrong to anyone who reads a
#: box score. **Rate stats carry three decimals** — a .311 average, a .766 OPS — because that
#: is the form they exist in; three decimals on a strikeout rate or a projection is invented
#: precision, and one decimal on an OPS reads as a typo.
RATE_PLACES, OTHER_PLACES = 3, 1

#: The traditional three-decimal family, matched on any word of a column name so the
#: pipeline's prefixes come along for free: `Platoon OPS`, `Season SLG`, `opp_ops`.
RATE_STATS = {"AVG", "OBP", "SLG", "OPS", "ISO", "XWOBA", "WOBA", "BABIP", "XBA", "XSLG"}

#: Columns that *are* OPS figures but whose name ends in a handedness instead of the stat —
#: `Allowed vs L`, `Lineup vs R`, `Edge overall`. Matched on the leading word, and only when
#: the name carries no count word, since `Allowed PA vs L` and `Bats vs L` are counts.
RATE_LEADS = {"ALLOWED", "LINEUP", "EDGE"}
COUNT_WORDS = {"PA", "AB", "BATS", "N", "COUNT", "GAMES", "BIP", "PITCHES"}

#: Park and weather multipliers. These sit within a few points of 1.00, so one decimal
#: collapses a 1.04 park and a 1.00 one into the same number — they are ratios and take the
#: ratio treatment.
FACTORS = {"PARK HR", "PARK RUNS", "PARK FACTOR", "HR ENV", "WEATHER HR", "HR LEVERAGE",
           "ENV", "CARRY"}


def _words(column):
    return [w for w in str(column).replace("_", " ").replace("/", " ").upper().split() if w]


def _places(column):
    """How many decimals a column should carry: three for a ratio, one for everything else."""
    words = _words(column)
    if not words:
        return OTHER_PLACES
    if " ".join(words) in FACTORS:
        return RATE_PLACES
    if set(words) & RATE_STATS:
        return RATE_PLACES
    if words[0] in RATE_LEADS and not (set(words) & COUNT_WORDS):
        return RATE_PLACES
    return OTHER_PLACES


def rounded(frame, places=None):
    """Round every float column for display: three decimals for rate stats, one otherwise.

    Applied where a frame is produced rather than where it is drawn, so the tooltip, the
    table and the chart all quote the same number — a value rounded in one place and not the
    other is how a reader ends up with two different figures for the same thing.

    Pass `places` to force a single precision on everything, which only the rare frame that
    is entirely one kind of number should need.
    """
    if frame is None or getattr(frame, "empty", True):
        return frame
    out = frame.copy()
    for column in out.select_dtypes("float").columns:
        out[column] = out[column].round(places if places is not None else _places(column))
    return out


def context_frame(payload, key):
    """One context frame from a payload, ready to render.

    Every page pulls raw payload frames through here, so this is where the workbook's
    blank-for-missing formatting is turned into real nulls — and, once they are real
    numbers, where the display precision is settled. Rounding here rather than at each
    `st.dataframe` call is what keeps a figure identical between a table, a tooltip and a
    chart axis.
    """
    frame = payload.get("advanced_context", {}).get(key)
    if not isinstance(frame, pd.DataFrame):
        return pd.DataFrame()
    return rounded(arrow_safe(frame))

# --- lineups, pitching and environment -------------------------------------------------
#
# `report_args` is a positional tuple, which is fragile to index into blind. These are the
# slots the report writes, confirmed against the payload: 0/7 are the home and away lineup
# cards (Name, Pos, Bats, Spot), and the frames are what carry a hitter's *position* — the
# composite table does not have one.
HOME_LINEUP, AWAY_LINEUP = 0, 7


def _lineup(payload, index):
    args = payload.get("report_args") or ()
    if index >= len(args):
        return pd.DataFrame()
    frame = args[index]
    return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()


def positions(payload, meta):
    """{player name: (position, batting order spot)} for both clubs.

    Joined by name rather than id because the composite table carries neither an id nor a
    position; both frames come from the same report run, so the spellings agree.
    """
    out = {}
    for index in (HOME_LINEUP, AWAY_LINEUP):
        frame = _lineup(payload, index)
        if frame.empty or "Name" not in frame.columns:
            continue
        for _, row in frame.iterrows():
            out[str(row.get("Name", "")).strip()] = (
                str(row.get("Pos", "") or ""),
                pd.to_numeric(pd.Series([row.get("Spot")]), errors="coerce").iloc[0],
            )
    return out


def with_positions(hitters_frame, payload, meta):
    """Attach `Pos` and lineup `Spot`. Absent for a hitter not in the posted card."""
    if hitters_frame is None or hitters_frame.empty:
        return hitters_frame
    lookup = positions(payload, meta)
    frame = hitters_frame.copy()
    names = frame.get("Name", pd.Series("", index=frame.index)).astype(str).str.strip()
    frame["Pos"] = [lookup.get(n, ("", None))[0] for n in names]
    frame["Spot"] = [lookup.get(n, ("", None))[1] for n in names]
    return frame


def bullpen_usage(payload, side):
    """Per-arm usage grid out of `{side}_bullpen_l5`.

    That payload value is a tuple of (team game log, per-arm usage). The per-arm frame is
    the one with a `Name` column, found by shape rather than index because the tuple order
    is not guaranteed anywhere.
    """
    entry = payload.get("advanced_context", {}).get(f"{side}_bullpen_l5")
    parts = entry if isinstance(entry, (tuple, list)) else [entry]
    for part in parts:
        if isinstance(part, pd.DataFrame) and not part.empty and "Name" in part.columns:
            return arrow_safe(part)
    return pd.DataFrame()


def bullpen_batted(payload, side):
    """Per-arm batted-ball profile, with the BULLPEN total row separated out."""
    entry = payload.get("advanced_context", {}).get(f"{side}_bullpen_batted")
    parts = entry if isinstance(entry, (tuple, list)) else [entry]
    for part in parts:
        if isinstance(part, pd.DataFrame) and not part.empty and "GB%" in part.columns:
            frame = part.copy()
            name = frame.get("Name", pd.Series("", index=frame.index)).astype(str)
            frame["is_total"] = name.str.strip().str.upper().eq("BULLPEN")
            for column in ("BIP", "GB%", "FB%", "LD%", "HH%", "Avg EV", "Max EV", "Brl%"):
                frame[column] = _numeric(frame, column)
            return frame
    return pd.DataFrame()


def starter_batted(payload, side):
    """The starter's season batted-ball line plus his recent starts."""
    profile = payload.get("advanced_context", {}).get(f"{side}_sp_profile")
    if not isinstance(profile, dict):
        return pd.DataFrame(), {}
    frame = profile.get("batted")
    frame = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    if not frame.empty:
        for column in ("BIP", "GB%", "FB%", "LD%", "HH%", "Avg EV", "Max EV", "Brl%"):
            frame[column] = _numeric(frame, column)
    return frame, profile.get("season_batted") or {}


def park(payload):
    """Run and home-run factors plus the handed park factors, or an empty dict."""
    environment = payload.get("advanced_context", {}).get("environment") or {}
    return {
        "name": environment.get("park_name") or environment.get("venue") or "",
        "runs": (environment.get("park") or {}).get("Runs"),
        "hr": (environment.get("park") or {}).get("HR"),
        "profile": (environment.get("park") or {}).get("profile") or "",
        "handed": environment.get("park_handed") or {},
    }


# --- slate-level pitching --------------------------------------------------------------


def _first_row(frame):
    return frame.iloc[0] if isinstance(frame, pd.DataFrame) and not frame.empty else None


def starters(payload, meta):
    """One row per starting pitcher in a game, with the park and defence behind him.

    Side naming is the thing to get right and easy to get wrong: `home_pitcher_hand_splits`
    is the **home starter's** splits, while `home_opp_pitching` describes the arms the home
    club *faces*. The defence behind a starter is his own club's, and the lineup he faces is
    the other one — so `team` and `faces` are always opposites here.
    """
    context = payload.get("advanced_context", {})
    park_info = park(payload)
    watchlist = context.get("pitcher_watchlist")
    watchlist = watchlist if isinstance(watchlist, pd.DataFrame) else pd.DataFrame()

    rows = []
    for side in ("away", "home"):
        team = meta["away"] if side == "away" else meta["home"]
        faces = meta["home"] if side == "away" else meta["away"]
        splits = context.get(f"{side}_pitcher_hand_splits")
        name = ""
        if isinstance(splits, pd.DataFrame) and not splits.empty and "Pitcher" in splits:
            name = str(splits["Pitcher"].dropna().iloc[0]) if splits["Pitcher"].notna().any() else ""
        if not name:
            continue

        _, season = starter_batted(payload, side)
        defence = context.get(f"{side}_team_defense")
        defence_row = _first_row(defence)
        quality = _first_row(context.get(f"{side}_pitcher_quality"))

        entry = {
            "game": meta["label"], "date": meta["date"], "side": side,
            "pitcher": name, "team": team, "faces": faces,
            "park": park_info["name"],
            "park_runs": park_info["runs"], "park_hr": park_info["hr"],
            "gb_rate": season.get("GB%"), "fb_rate": season.get("FB%"),
            "ld_rate": season.get("LD%"), "hh_rate": season.get("HH%"),
            "barrel_rate": season.get("Brl%"), "bip": season.get("BIP"),
            "hits_saved": (pd.to_numeric(pd.Series([defence_row.get("Hits Saved/G")]),
                                         errors="coerce").iloc[0]
                           if defence_row is not None else float("nan")),
            "defense_grade": (str(defence_row.get("Grade", "")) if defence_row is not None
                              else ""),
            "opp_quality": (pd.to_numeric(pd.Series([quality.get("Opp Avg OPS")]),
                                          errors="coerce").iloc[0]
                            if quality is not None else float("nan")),
        }

        # The watchlist is keyed by club, and its `Team` is the pitcher's own club.
        if not watchlist.empty and "Team" in watchlist.columns:
            match = watchlist[watchlist["Team"].astype(str) == str(team)]
            row = _first_row(match)
            if row is not None:
                entry["fip"] = pd.to_numeric(pd.Series([row.get("FIP")]),
                                             errors="coerce").iloc[0]
                entry["k_bb"] = pd.to_numeric(pd.Series([row.get("K-BB")]),
                                              errors="coerce").iloc[0]
                entry["score"] = pd.to_numeric(pd.Series([row.get("Score")]),
                                               errors="coerce").iloc[0]
                entry["why"] = str(row.get("Why", "") or "")
        rows.append(entry)
    return pd.DataFrame(rows)


def _opponent_lineup_ops(payload, meta):
    """Season OPS of each club's posted lineup, weighted by at-bats.

    Weighted, because a nine-man mean treats a 500-at-bat regular and a call-up as equal
    claims about how good the lineup is.
    """
    out = {}
    frame = hitters(payload, meta)
    if frame.empty:
        return out
    for team, group in frame.groupby("Team"):
        weight = pd.to_numeric(group.get("Season AB"), errors="coerce").fillna(0)
        ops = pd.to_numeric(group.get("Season OPS"), errors="coerce")
        usable = ops.notna() & (weight > 0)
        if usable.any():
            out[str(team)] = float((ops[usable] * weight[usable]).sum()
                                   / weight[usable].sum())
    return out


@st.cache_data(max_entries=8, show_spinner="Reading cached games…")
def starters_for_date(date, cache_dir=CACHE_DIR):
    """Every starting pitcher on a date, with the lineup strength he faces attached."""
    games = list_games(cache_dir)
    games = games[games["date"] == date]
    frames = []
    for meta in games.to_dict("records"):
        try:
            payload = load_payload(meta["path"])
            frame = starters(payload, meta)
            if frame.empty:
                continue
            lineup_ops = _opponent_lineup_ops(payload, meta)
            frame["opp_ops"] = [lineup_ops.get(f) for f in frame["faces"]]
            frames.append(frame)
        except Exception:
            continue
    return rounded(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame()


@st.cache_data(max_entries=8, show_spinner="Reading cached games…")
def bullpens_for_date(date, cache_dir=CACHE_DIR):
    """Every rostered reliever on a date, graded for availability."""
    from dashboards import bullpen as bullpen_module

    games = list_games(cache_dir)
    games = games[games["date"] == date]
    frames = []
    for meta in games.to_dict("records"):
        try:
            payload = load_payload(meta["path"])
        except Exception:
            continue
        for side in ("away", "home"):
            graded = bullpen_module.availability(bullpen_usage(payload, side))
            if graded.empty:
                continue
            graded = graded.copy()
            graded["team"] = meta["away"] if side == "away" else meta["home"]
            graded["game"] = meta["label"]
            graded["date"] = meta["date"]
            frames.append(graded)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data(max_entries=8, show_spinner="Projecting the slate…")
def projections_for_date(date, cache_dir=CACHE_DIR):
    """DK point projections for a whole date, from the same cached payloads.

    `dfs.projections.project_game` reads a payload and touches nothing else, so the real
    `Proj`, `Ceiling` and `Floor` the DFS pipeline uses are available here for about 0.8s a
    slate. That matters because a ceiling is not derivable from the composite: it comes
    from event variance, and a home-run bat and a singles hitter with the same mean have
    very different ones.

    Imported inside the function so the dashboard still starts without pulling in the
    projection stack on pages that never ask for it.
    """
    from dfs import projections

    games = list_games(cache_dir)
    if games.empty:
        return pd.DataFrame()
    rows = []
    for _, game in games[games["date"] == str(date)].iterrows():
        try:
            rows.extend(projections.project_game(load_payload(game["path"])))
        except Exception:
            # One malformed payload must not cost the whole slate its projections.
            continue
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    for column in ("Proj", "Ceiling", "Floor", "Bust%", "PA", "HR", "Team Runs"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def attach_projection(hitters_frame, date, cache_dir=CACHE_DIR):
    """Add `Proj`, `Ceiling`, `Floor` and `Bust%` to a hitter frame.

    Joined on the folded name, the same key the salary join uses, so a hitter cannot pick
    up a salary but miss his projection over an accent.
    """
    from dashboards import salaries

    if hitters_frame is None or hitters_frame.empty:
        return hitters_frame
    projected = projections_for_date(date, cache_dir)
    out = hitters_frame.copy()
    columns = ["Proj", "Ceiling", "Floor", "Bust%", "PA", "Team Runs"]
    if projected.empty or "Name" not in projected.columns:
        for column in columns:
            out[column] = float("nan")
        return out
    batters = projected[projected.get("Type", "") != "P"] if "Type" in projected else projected
    keyed = batters.assign(_key=batters["Name"].map(salaries.name_key))
    keyed = keyed.drop_duplicates("_key")
    out["_key"] = out["Name"].map(salaries.name_key)
    merged = out.merge(keyed.reindex(columns=["_key"] + [c for c in columns
                                                         if c in keyed.columns]),
                       on="_key", how="left")
    return merged.drop(columns="_key")



def effective_side(bats, throws):
    """Which side of the plate a hitter will actually bat from tonight.

    `Bats` is `S` for a switch hitter, which matches neither row of a pitcher's hand
    splits. He bats **opposite** the arm, so the side has to be resolved from the pitcher
    before any split can be looked up — skipping this silently drops every switch hitter
    from a matchup panel, and on a typical slate that is ten percent of the board.

    An unknown pitcher hand returns `""` rather than a guess: half the switch hitters would
    be assigned the wrong split, and a wrong split looks exactly as plausible as a right one.
    """
    side = str(bats or "").strip().upper()
    if side in ("L", "R"):
        return side
    return {"R": "L", "L": "R"}.get(str(throws or "").strip().upper(), "")

@st.cache_data(max_entries=8, show_spinner="Reading cached games…")
def hitters_vs_allowed(date, cache_dir=CACHE_DIR):
    """Every hitter on a slate, with what tonight's arm actually allows to his side.

    Why this baseline and not season OPS
    ------------------------------------
    "OPS against this arsenal, compared to his season OPS" answers *is this hitter better or
    worse than usual against these pitches* — a question about the hitter alone. The matchup
    question is different: **does he beat what this particular arm gives up to batters of his
    side?** A .700 bat facing a starter who surrenders .639 to his side is in a worse spot
    than a .680 bat facing one who surrenders .780, and season OPS cannot see that at all.

    The report caches the surrendered line per pitcher and per side in
    `*_pitcher_hand_splits`, with the PA behind it. Joined onto each hitter it becomes a
    per-matchup reference the whole slate can be sorted on.

    Switch hitters
    --------------
    `Bats` is `S` for a switch hitter, which matches neither split row. He bats **opposite**
    the pitcher's throwing hand, so the effective side is resolved from the arm before the
    lookup. Skipping that would silently drop every switch hitter from the panel.
    """
    games = list_games(cache_dir)
    if games.empty:
        return pd.DataFrame()

    # The arm's throwing hand decides which split row applies, and only the projections
    # carry it authoritatively — `pitcher_similar` lists *comparable* arms, whose hand
    # matches the starter's on every one of the 57 games checked but is an inference, not a
    # record.
    #
    # **Keyed by the pitcher, not by (date, club).** Both games of a doubleheader belong to
    # the same club on the same date and can have opposite-handed starters, so a club key
    # silently hands one of the two games the other's hand — and every hitter in it the
    # wrong split.
    from dashboards import salaries as _salaries

    hands = {}
    projected = projections_for_date(date, cache_dir)
    if not projected.empty and {"Opp SP", "Opp SP Hand"} <= set(projected.columns):
        for name, hand in zip(projected["Opp SP"], projected["Opp SP Hand"]):
            if pd.isna(name) or pd.isna(hand):
                continue
            hands.setdefault(_salaries.name_key(name), str(hand))

    frames = []
    for _, game in games[games["date"] == str(date)].iterrows():
        meta = game.to_dict()
        payload = load_payload(meta["path"])
        board = hitters(payload, meta)
        if board.empty:
            continue
        for side in ("away", "home"):
            team = meta["away"] if side == "away" else meta["home"]
            facing = "home" if side == "away" else "away"
            splits = context_frame(payload, f"{facing}_pitcher_hand_splits")
            if splits.empty or "Batter Side" not in splits.columns:
                continue
            pitcher = splits["Pitcher"].dropna()
            pitcher = str(pitcher.iloc[0]) if not pitcher.empty else ""
            throws = str(hands.get(_salaries.name_key(pitcher), "") or "")

            allowed = {}
            for _, split in splits.iterrows():
                allowed[str(split["Batter Side"]).strip().upper()] = split

            club = board[board["Team"] == team].copy()
            if club.empty:
                continue
            bats = club["Bats"].astype("string").fillna("").str.strip().str.upper()
            club["Faces Hand"] = throws.upper()
            club["Effective Side"] = [effective_side(b, throws) for b in bats]

            for column, field in (("Allowed OPS", "OPS"), ("Allowed xwOBA", "xwOBA"),
                                  ("Allowed K%", "K%"), ("Allowed PA", "PA"),
                                  ("Allowed HardHit%", "HardHit%")):
                club[column] = [
                    pd.to_numeric(pd.Series([allowed.get(s, {}).get(field)]),
                                  errors="coerce").iloc[0] if s in allowed else float("nan")
                    for s in club["Effective Side"]]
            club["Opp SP"] = pitcher
            club["Split Tag"] = [str(allowed.get(s, {}).get("Split Tag", "") or "")
                                 for s in club["Effective Side"]]
            frames.append(club)

    if not frames:
        return pd.DataFrame()
    board = pd.concat(frames, ignore_index=True)
    # The edge is the whole point of the frame: how far a hitter's measured line against
    # this arsenal sits above what this arm actually gives up to his side.
    board["Arsenal Edge"] = board["Arsenal OPS"] - board["Allowed OPS"]
    board["Platoon Edge"] = board["Platoon OPS"] - board["Allowed OPS"]
    board["Season Edge"] = board["Season OPS"] - board["Allowed OPS"]
    return board


#: Per-hitter measures the pipeline records in `*_batter_arsenal` but the composite table
#: does not carry. These are what let a view be re-based onto something other than OPS.
ARSENAL_MEASURES = ["xwOBA", "SLG", "HardHit%", "Whiff%", "K% vs Hand", "K Edge",
                    "Arsenal Score", "Fit"]

#: Season rate stats carried on the **lineup card** in `report_args`, which the composite
#: table drops. An earlier version of this module claimed ISO was not derivable from the
#: cache; that was wrong — the card has a full season line per hitter, including ISO.
CARD_MEASURES = ["AVG", "OBP", "SLG", "ISO", "PA", "SB", "RBI"]


def attach_card_measures(frame, payload, meta):
    """Add the lineup card's season rate line to a composite frame.

    The card is the only cached source of per-hitter `AVG`, `OBP` and `ISO`. Its `SLG`
    differs from the arsenal frame's — the card's is season-long, the arsenal frame's is
    against tonight's pitch mix — so the card's is suffixed rather than allowed to collide
    and silently overwrite the matchup number.
    """
    from dashboards import salaries

    if frame is None or frame.empty:
        return frame
    args = payload.get("report_args") or ()
    cards = [args[i] for i in (0, 7)
             if len(args) > i and isinstance(args[i], pd.DataFrame)
             and not args[i].empty and "Name" in args[i].columns]
    out = frame.copy()
    if not cards:
        for column in CARD_MEASURES:
            out[f"Season {column}"] = float("nan")
        return out
    card = pd.concat(cards, ignore_index=True)
    card["_key"] = card["Name"].map(salaries.name_key)
    keep = ["_key"] + [c for c in CARD_MEASURES if c in card.columns]
    card = (card.reindex(columns=keep).drop_duplicates("_key")
            .rename(columns={c: f"Season {c}" for c in CARD_MEASURES}))
    out["_key"] = out["Name"].map(salaries.name_key)
    merged = out.merge(card, on="_key", how="left").drop(columns="_key")
    for column in CARD_MEASURES:
        name = f"Season {column}"
        if name in merged.columns:
            merged[name] = pd.to_numeric(merged[name], errors="coerce")
        else:
            merged[name] = float("nan")
    return merged


def attach_arsenal_measures(frame, payload, meta):
    """Add the per-hitter arsenal measures to a composite frame.

    Joined on folded name within a club, because both sides of a game are in the same
    composite table and two players can share a surname across clubs.
    """
    from dashboards import salaries

    if frame is None or frame.empty:
        return frame
    pieces = []
    for side in ("away", "home"):
        piece = context_frame(payload, f"{side}_batter_arsenal")
        if not piece.empty and "Name" in piece.columns:
            pieces.append(piece)
    out = frame.copy()
    if not pieces:
        for column in ARSENAL_MEASURES:
            out[column] = float("nan")
        return out

    arsenal = pd.concat(pieces, ignore_index=True)
    arsenal["_key"] = arsenal["Name"].map(salaries.name_key)
    keep = ["_key"] + [c for c in ARSENAL_MEASURES if c in arsenal.columns]
    if "Team" in arsenal.columns:
        arsenal["_team"] = arsenal["Team"].astype(str)
        keep.append("_team")
    arsenal = arsenal.reindex(columns=keep).drop_duplicates("_key")

    out["_key"] = out["Name"].map(salaries.name_key)
    merged = out.merge(arsenal.drop(columns=[c for c in ["_team"] if c in arsenal]),
                       on="_key", how="left")
    for column in ARSENAL_MEASURES:
        if column not in merged.columns:
            merged[column] = float("nan")
        elif column != "Fit":
            merged[column] = pd.to_numeric(merged[column], errors="coerce")
    return merged.drop(columns="_key")


@st.cache_data(max_entries=8, show_spinner="Reading cached games…")
def pitcher_vs_lineup(date, cache_dir=CACHE_DIR):
    """Each starter's surrendered OPS against the OPS of the lineup he faces.

    Two lines that the workbook keeps in different tables and never puts on the same scale:

    * what this arm **gives up**, overall and split by the batter's side, from
      `*_pitcher_hand_splits`;
    * what the lineup he faces **hits**, overall and split the same way, weighted by
      at-bats so a 500-AB regular and a call-up are not treated as equal.

    The platoon rows are the ones with teeth. A left-hander who surrenders .654 to lefties
    is only in a good spot if the lineup he draws is actually left-handed, and lineup
    construction is a thing managers change *because* of him — so the count of each handed
    bat is carried alongside, and a switch hitter is counted on the side he will bat from.
    """
    games = list_games(cache_dir)
    if games.empty:
        return pd.DataFrame()

    rows = []
    for _, game in games[games["date"] == str(date)].iterrows():
        meta = game.to_dict()
        payload = load_payload(meta["path"])
        board = hitters(payload, meta)
        if board.empty:
            continue
        for side in ("away", "home"):
            team = meta["away"] if side == "away" else meta["home"]
            faces = meta["home"] if side == "away" else meta["away"]
            splits = context_frame(payload, f"{side}_pitcher_hand_splits")
            if splits.empty or "Batter Side" not in splits.columns:
                continue
            pitcher = splits["Pitcher"].dropna()
            if pitcher.empty:
                continue

            allowed = {}
            for _, split in splits.iterrows():
                allowed[str(split["Batter Side"]).strip().upper()] = split

            lineup = board[board["Team"] == faces]
            if lineup.empty:
                continue
            hand = _starter_hand_from_projection(payload, side, meta, team)
            sides = [effective_side(b, hand) for b in lineup["Bats"]]
            lineup = lineup.assign(_side=sides)

            entry = {
                "game": meta["label"], "date": meta["date"], "side": side,
                "pitcher": str(pitcher.iloc[0]), "team": team, "faces": faces,
                "throws": hand or "",
                "lineup_bats": len(lineup),
                "L bats": int((lineup["_side"] == "L").sum()),
                "R bats": int((lineup["_side"] == "R").sum()),
                "switch bats": int(lineup["Bats"].astype(str).str.upper().eq("S").sum()),
            }
            entry["Allowed OPS"] = _blended_allowed(allowed, lineup)
            entry["Lineup OPS"] = _weighted_ops(lineup, "Season OPS", "Season AB")
            entry["Lineup Platoon OPS"] = _weighted_ops(lineup, "Platoon OPS", "Platoon AB")
            for hand_code, label in (("L", "vs L"), ("R", "vs R")):
                split = allowed.get(hand_code)
                entry[f"Allowed {label}"] = (
                    pd.to_numeric(pd.Series([split.get("OPS")]), errors="coerce").iloc[0]
                    if split is not None else float("nan"))
                entry[f"Allowed PA {label}"] = (
                    pd.to_numeric(pd.Series([split.get("PA")]), errors="coerce").iloc[0]
                    if split is not None else float("nan"))
                subset = lineup[lineup["_side"] == hand_code]
                entry[f"Lineup {label}"] = _weighted_ops(subset, "Season OPS", "Season AB")
                entry[f"Bats {label}"] = int(len(subset))
            # Positive means the lineup hits better than this arm normally allows.
            entry["Edge overall"] = entry["Lineup OPS"] - entry["Allowed OPS"]
            for label in ("vs L", "vs R"):
                entry[f"Edge {label}"] = (entry[f"Lineup {label}"]
                                          - entry[f"Allowed {label}"])
            rows.append(entry)
    return rounded(pd.DataFrame(rows))


def _starter_hand_from_projection(payload, side, meta, team):
    """The starter's own throwing hand.

    The projections record it per hitter as the hand *faced*, so the club that faces this
    starter is the one to read it from.
    """
    faces = meta["home"] if side == "away" else meta["away"]
    projected = projections_for_date(meta["date"])
    if projected.empty or "Opp SP Hand" not in projected.columns:
        return None
    # **Batters only.** The projection frame carries pitcher rows too, and a pitcher's
    # `Opp SP Hand` is not the hand his club's hitters face — taking the first row for a
    # club returned that club's own starter's hand, inverting every matchup on the page.
    batters = projected[projected["Type"] != "P"] if "Type" in projected.columns else projected
    rows = batters[batters.get("Team", "") == faces]
    hands = rows["Opp SP Hand"].dropna().astype(str)
    return hands.iloc[0].strip().upper() if not hands.empty else None


def _weighted_ops(frame, column, weight_column):
    """At-bat weighted OPS. A nine-man mean treats a regular and a call-up as equal."""
    if frame is None or frame.empty or column not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce")
    weights = pd.to_numeric(frame.get(weight_column), errors="coerce")
    if weights is None or weights.isna().all() or weights.fillna(0).sum() <= 0:
        return float(values.mean()) if values.notna().any() else float("nan")
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float("nan")
    return float((values[mask] * weights[mask]).sum() / weights[mask].sum())


def _blended_allowed(allowed, lineup):
    """What this arm gives up, blended by the handedness of the lineup he actually faces.

    Not the mean of his two split rows: a lefty who suppresses lefties but not righties is
    in a very different spot against a right-handed lineup, and averaging the two hides
    exactly that. Weighted by how many bats of each side he will actually see.
    """
    total = weighted = 0.0
    for hand_code in ("L", "R"):
        split = allowed.get(hand_code)
        if split is None:
            continue
        ops = pd.to_numeric(pd.Series([split.get("OPS")]), errors="coerce").iloc[0]
        count = int((lineup["_side"] == hand_code).sum())
        if pd.isna(ops) or count == 0:
            continue
        weighted += float(ops) * count
        total += count
    return weighted / total if total else float("nan")
