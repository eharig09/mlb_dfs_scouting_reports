import pandas as pd
import streamlit as st

from dashboards import bullpen as bullpen_module
from dashboards import charts, data, drill_ui, filters, strikeouts, tables

st.header("Pitching", anchor=False)

games = data.list_games()
if games.empty:
    st.info("No cached games found. Run a report first — the dashboard reads "
            "`.cache/report_data/`.")
    st.stop()

date, slate, in_scope = filters.scope(games)
on_date = in_scope[in_scope["date"] == date]
if on_date.empty:
    st.warning(f"No cached game on the {slate} slate for {date}.")
    st.stop()
with st.sidebar:
    st.subheader("Game", anchor=False)
    label = st.selectbox("Game", list(on_date["label"]), key="pitching_game",
                         persist_state="session")

meta = on_date[on_date["label"] == label].iloc[0].to_dict()
payload = data.load_payload(meta["path"])
starters = data.starters(payload, meta)
park = data.park(payload)

if starters.empty:
    st.warning("No starting-pitcher data in this payload.")
    st.stop()

with st.container(horizontal=True):
    for _, row in starters.iterrows():
        st.metric(f"{row['team']} starter", row["pitcher"],
                  delta=f"FIP {row['fip']:.2f}" if pd.notna(row.get("fip")) else None,
                  delta_color="off", border=True)
    if park.get("runs") is not None:
        st.metric("Park runs", f"{park['runs']:.2f}", delta=park.get("name"),
                  delta_color="off", border=True)

st.subheader("Starters", anchor=False)

left, right = st.columns(2)
for column, (_, row) in zip((left, right), starters.iterrows()):
    side = row["side"]
    with column:
        with st.container(border=True):
            st.markdown(f"**{row['pitcher']}** · {row['team']} vs {row['faces']}")
            splits = data.context_frame(payload, f"{side}_pitcher_hand_splits")
            if not splits.empty:
                st.caption("How each side of the plate has done against him — tinted from "
                           "**his** side, so green is the arm winning that matchup")
                st.dataframe(
                    tables.style(splits, [
                        (tables.by_value, {"column": "Split Tag",
                                           "tints": tables.SPLIT_TAG_TINTS}),
                        (tables.ranked, {"column": "OPS", "best": "low"}),
                        (tables.ranked, {"column": "xwOBA", "best": "low"}),
                        (tables.ranked, {"column": "K%", "best": "high"}),
                    ]),
                    hide_index=True)
            arsenal = data.context_frame(payload, f"{side}_arsenal_matchup")
            if not arsenal.empty:
                st.caption("The lineup he faces, by pitch — right of zero the bats are "
                           "ahead, so a red RV/100 is a pitch they have hit")
                st.dataframe(
                    tables.style(arsenal, [
                        (tables.by_value, {"column": "Edge", "tints": tables.EDGE_TINTS}),
                        # RV/100 is run value from the hitter's side: above zero the bats
                        # gained, which is the bad direction for the arm this panel is about.
                        # Tinted only past the same threshold `Edge` itself uses, so a
                        # mid-band number is left alone rather than dressed up as a read.
                        (tables.signed, {"column": "RV/100", "good_when_negative": True,
                                         "threshold": tables.RV_EDGE_THRESHOLD}),
                        (tables.ranked, {"column": "xwOBA", "best": "low"}),
                        (tables.ranked, {"column": "Whiff%", "best": "high"}),
                    ]),
                    hide_index=True)
            rest = data.context_frame(payload, f"{side}_sp_rest_splits")
            if not rest.empty:
                st.caption("By days of rest — his best and worst bucket marked")
                st.dataframe(
                    tables.highlight(rest, columns={"ERA", "FIP", "WHIP", "K%", "BB%",
                                                    "K-BB", "K-BB%"}),
                    hide_index=True)
            if st.button(f"Open {row['pitcher']}", key=f"open_sp_{side}",
                         width="stretch",
                         help="Start-by-start log with days of rest, and the comparable "
                              "arms his arsenal was scored against"):
                drill_ui.pitcher_dialog(payload, meta, side)

st.subheader("Strikeouts", anchor=False)
st.caption("Whether an arm misses bats is only worth knowing against the bats he draws, so "
           f"all three panels share one reference line at the {strikeouts.LEAGUE_K:.1f}% "
           "league rate.")

# `data.starters` gives the scouting row but not the raw starter-info dict, which is where
# the season K% and the start log live. args[2] is the home starter, args[9] the away one.
_STARTER_INFO = {"home": 2, "away": 9}

for _, row in starters.iterrows():
    side = row["side"]
    args = payload.get("report_args") or ()
    index = _STARTER_INFO[side]
    info = (args[index] or {}) if len(args) > index else {}
    # He faces the OTHER club, so the lineup-side frames are crossed.
    opposing = "away" if side == "home" else "home"
    hand_splits = strikeouts.platoon(payload, side, meta)
    head = strikeouts.headline(info, data.context_frame(payload, f"{opposing}_lineup_splits"),
                               hand_splits)

    with st.container(border=True):
        st.markdown(f"**{row['pitcher']}** · {row['team']} — striking out {row['faces']}")
        with st.container(horizontal=True):
            st.metric("His K%", f"{head['his_k']:.1f}%" if pd.notna(head["his_k"]) else "—",
                      delta=f"{head['his_k'] - head['league']:+.1f} vs league"
                            if pd.notna(head["his_k"]) else None,
                      delta_color="off", border=True)
            st.metric("Their K%",
                      f"{head['lineup_k']:.1f}%" if pd.notna(head["lineup_k"]) else "—",
                      delta="how often this card strikes out", delta_color="off", border=True)
            st.metric("Edge", f"{head['edge']:+.1f}" if pd.notna(head["edge"]) else "—",
                      delta="his rate minus theirs", delta_color="off", border=True)
            if pd.notna(head["split_spread"]):
                st.metric("Platoon spread", f"{head['split_spread']:.1f}",
                          delta="gap between his two sides", delta_color="off", border=True)

        left, right = st.columns([3, 2])
        with left:
            recent = strikeouts.recent_starts(info)
            chart = charts.k_recent_starts(recent, season_k=head["his_k"],
                                           lineup_k=head["lineup_k"],
                                           league=strikeouts.LEAGUE_K)
            if chart is not None:
                st.altair_chart(chart)
                st.caption("Dashes: league, his season rate, and this card's rate. K% per "
                           "start uses batters faced approximated as 3·IP + H + BB — the "
                           "game log carries no TBF — so it runs a shade high on a messy "
                           "start. K/9 in the tooltip is exact.")
            else:
                st.caption("No start log cached for him.")
        with right:
            chart = charts.k_platoon_bars(hand_splits, league=strikeouts.LEAGUE_K)
            if chart is not None:
                st.altair_chart(chart)
                st.caption("The number on each bar is how many bats of that side are in "
                           "tonight's card; switch hitters are counted on the side they "
                           "will actually bat from.")
            else:
                st.caption("No hand splits cached for him.")

        lineup_k = data.attach_hitter_ops(strikeouts.lineup_k(payload, opposing),
                                          payload, meta)
        if not lineup_k.empty:
            chart = charts.k_lineup_bars(lineup_k, league=strikeouts.LEAGUE_K)
            if chart is not None:
                st.altair_chart(chart)
            st.dataframe(
                tables.style(lineup_k.round(1), [
                    (tables.ranked, {"column": "K%", "best": "high"}),
                    (tables.ranked, {"column": "K% vs Hand", "best": "high"}),
                    (tables.signed, {"column": "K Edge"}),
                    (tables.ranked, {"column": "Whiff%", "best": "high"}),
                ]),
                hide_index=True,
                column_config={
                    "K%": st.column_config.NumberColumn("K% vs arsenal", format="%.1f"),
                    "K% vs Hand": st.column_config.NumberColumn("K% vs hand", format="%.1f"),
                    "K Edge": st.column_config.NumberColumn(
                        "K edge", format="%+.1f",
                        help="Arsenal K% minus his rate against this hand. Positive means "
                             "the pitch mix gets him out more than the handedness alone."),
                    "Whiff%": st.column_config.NumberColumn("Whiff%", format="%.1f"),
                })
        else:
            st.caption("No batter-vs-arsenal sample for this lineup.")

with st.container(border=True):
    st.markdown("**Starter batted-ball profile** — league average is roughly 44% ground, "
                "25% fly")
    rows = []
    for _, row in starters.iterrows():
        frame, season = data.starter_batted(payload, row["side"])
        if season:
            rows.append({"Name": row["pitcher"], "BIP": season.get("BIP"),
                         "GB%": season.get("GB%"), "FB%": season.get("FB%"),
                         "LD%": season.get("LD%"), "HH%": season.get("HH%"),
                         "Brl%": season.get("Brl%")})
    season_frame = pd.DataFrame(rows)
    chart = charts.batted_ball_scatter(season_frame, label="Starter")
    if chart is not None:
        st.altair_chart(chart)
    else:
        st.caption("No batted-ball profile cached for these starters.")

with st.container(border=True):
    st.markdown("**What he allows against what they hit** — both on one OPS scale, so the "
                "diagonal is parity: above it the lineup out-hits what this arm normally "
                "surrenders")
    matchups = data.pitcher_vs_lineup(date)
    # Slate-wide panel, so it takes the same exclusion the other slate views take.
    off_slate = filters.off_slate_teams(matchups, date, team_column="team")
    matchups = filters.on_slate(matchups, date, team_column="team")
    if off_slate:
        st.caption(f"Excluded {', '.join(off_slate)} — on no DraftKings slate for {date}.")
    if matchups.empty:
        st.info("No hand splits cached for this slate.")
    else:
        view = st.radio("Split", list(charts.VS_LINEUP_VIEWS), horizontal=True,
                        key="pitching_vs_view",
                        persist_state="session")
        chart = charts.pitcher_vs_lineup_scatter(matchups, view)
        if chart is None:
            st.info("No starter on this slate has that split.")
        else:
            st.altair_chart(chart)
        st.caption(
            "Every starter on the slate, not just this game. Point size is how many bats "
            "the split actually applies to — a favourable-looking platoon number over two "
            "bats is not a matchup. The overall figure blends his two split rows by the "
            "handedness of the card he draws rather than averaging them, because a lefty "
            "who suppresses lefties but not righties is in a very different spot against a "
            "right-handed lineup. Switch hitters are counted on the side they will bat "
            "from.")

    if not matchups.empty:
        st.markdown("**Lineup construction against him** — managers stack the side an arm "
                    "handles worse, and this is where that shows")
        chart = charts.platoon_exposure_bars(matchups)
        if chart is not None:
            st.altair_chart(chart)
        st.caption("Bars to the right mean the card leans toward his weaker split. On "
                   "2026-08-20 Texas drew left-hander Andrew Alvarez with three "
                   "left-handed bats against six right-handed ones.")

st.subheader("Bullpens", anchor=False)
st.caption("Availability is the stricter of the report's own grade and a measured "
           "back-to-back rule: a reliever who pitched on consecutive days appeared the "
           "next day only 5.5% of the time in 2025 (n=2,032), against 26.0% on rest.")

for side in ("away", "home"):
    team = meta["away"] if side == "away" else meta["home"]
    graded = bullpen_module.availability(data.bullpen_usage(payload, side))
    if graded.empty:
        continue
    counts = bullpen_module.summarise(graded)
    with st.container(border=True):
        st.markdown(f"**{team} bullpen**")
        with st.container(horizontal=True):
            for tier in bullpen_module.TIERS:
                st.metric(tier, counts[tier], border=True)

        stricter = bullpen_module.disagreements(graded)
        if not stricter.empty:
            names = ", ".join(stricter["Name"].astype(str))
            st.warning(f"Graded harder than the report: **{names}** — pitched on "
                       f"consecutive days, which the workbook's three-day total does not "
                       f"treat as a separate case.", icon=":material/warning:")

        panel_left, panel_right = st.columns(2)
        with panel_left:
            chart = charts.bullpen_workload_scatter(graded)
            if chart is not None:
                st.altair_chart(chart)
        with panel_right:
            batted = data.bullpen_batted(payload, side)
            arms = batted[~batted["is_total"]] if not batted.empty else batted
            merged = arms
            if not arms.empty and "Name" in graded.columns:
                merged = arms.merge(graded[["Name", "Status"]], on="Name", how="left")
            chart = charts.batted_ball_scatter(merged, label="Reliever") if not merged.empty else None
            if chart is not None:
                st.altair_chart(chart)
            else:
                st.caption("No bullpen batted-ball profile.")

        columns = ["Name", "T", "App", "IP", "Pit", "two_day", "three_day", "b2b",
                   "Avail", "Status", "ERA", "K-BB"]
        shown = graded.reindex(columns=[c for c in columns if c in graded.columns])
        st.dataframe(
            tables.style(shown, [
                (tables.by_value, {"column": "Status", "tints": tables.AVAIL_TINTS}),
                (tables.by_value, {"column": "Avail", "tints": tables.AVAIL_TINTS}),
                (tables.ranked, {"column": "ERA", "best": "low"}),
                (tables.ranked, {"column": "K-BB", "best": "high"}),
            ]),
            hide_index=True,
            column_config={
                "T": st.column_config.TextColumn("Throws"),
                "Pit": st.column_config.NumberColumn("Pitches L5", format="%d"),
                "two_day": st.column_config.NumberColumn("2-day", format="%d"),
                "three_day": st.column_config.NumberColumn("3-day", format="%d"),
                "b2b": st.column_config.CheckboxColumn("Back-to-back"),
                "Avail": st.column_config.TextColumn("Report grade"),
                "Status": st.column_config.TextColumn("Availability"),
            },
        )
