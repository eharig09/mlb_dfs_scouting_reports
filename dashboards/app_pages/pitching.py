import pandas as pd
import streamlit as st

from dashboards import bullpen as bullpen_module
from dashboards import charts, data, drill_ui, filters, tables

st.header("Pitching", anchor=False)

games = data.list_games()
if games.empty:
    st.info("No cached games found. Run a report first — the dashboard reads "
            "`.cache/report_data/`.")
    st.stop()

dates = list(dict.fromkeys(games["date"]))
with st.sidebar:
    st.subheader("Game", anchor=False)
    date = st.selectbox("Date", dates, key="pitching_date",
    persist_state="session")
    on_date = games[games["date"] == date]
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
                    ]),
                    hide_index=True)
            rest = data.context_frame(payload, f"{side}_sp_rest_splits")
            if not rest.empty:
                st.caption("By days of rest — his best and worst bucket marked")
                st.dataframe(
                    tables.style(rest, [(tables.ranked, {"column": "ERA", "best": "low"})]),
                    hide_index=True)
            if st.button(f"Open {row['pitcher']}", key=f"open_sp_{side}",
                         width="stretch",
                         help="Start-by-start log with days of rest, and the comparable "
                              "arms his arsenal was scored against"):
                drill_ui.pitcher_dialog(payload, meta, side)

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
