"""How each club divides its targets — the stack view.

**Shares are of the club's *qualifying* receivers, not of every man who ran a route.** That
makes this a comparison between the players who actually play, which is the question a stack
asks; counting deep reserves in the denominator would dilute the top of every card and make
two clubs with different bench depth look different at the top when they are not.

**The WR1 premium does not exist.** Measured over 2023-25, a club's second receiver
correlates with its quarterback as strongly as its first (+0.359 against +0.367, and higher
on the median) while carrying about a quarter less target share. So the useful reading of
this page is *who else is in the passing game*, not who leads it.
"""

import pandas as pd
import streamlit as st

from dashboards import nfl_charts, nfl_pff

st.header("Targets", anchor=False)
st.caption("Who each club throws to. Shares are within the club's qualifying receivers.")

seasons_available = nfl_pff.available_seasons()
if not seasons_available:
    st.info("No PFF receiving exports found. Drop them under `nfl/pff/` and reload.")
    st.stop()

with st.sidebar:
    st.subheader("Scope", anchor=False)
    seasons = st.multiselect("Seasons", seasons_available, default=seasons_available[:1],
                             key="nfltgt_seasons", persist_state="session")
    if not seasons:
        st.warning("Pick at least one season.")
        st.stop()
    min_routes = st.slider("Minimum routes", 25, 400, 100, step=25, key="nfltgt_routes", persist_state="session")

frame = nfl_pff.target_distribution(tuple(sorted(seasons, reverse=True)),
                                    min_routes=min_routes)
if frame.empty:
    st.warning("Nothing clears that route threshold for those seasons.")
    st.stop()

teams = sorted(frame["Team"].dropna().unique())

mix = (frame.drop_duplicates("Team")[["Team", "WR share", "TE share", "RB share"]]
       .dropna(how="all", subset=["WR share", "TE share", "RB share"]))

tabs = st.tabs(["One club", "League mix", "Every receiver"])

with tabs[0]:
    team = st.selectbox("Club", teams, key="nfltgt_team", persist_state="session")
    chart = nfl_charts.share_bars(frame, team)
    if chart is None:
        st.warning("No qualifying receivers for that club.")
    else:
        st.altair_chart(chart, width="stretch")

    one = frame[frame["Team"] == team]
    columns = [c for c in ("Name", "Pos", "Targets", "Target share", "Route share",
                           "TPRR", "aDOT", "Slot%") if c in one.columns]
    st.dataframe(
        nfl_charts.round_display(
            one[columns].sort_values("Target share", ascending=False)),
        hide_index=True, width="stretch",
        column_config={
            "Target share": st.column_config.ProgressColumn(
                "Target share", min_value=0.0, max_value=1.0, format="%.1f%%"),
            "Route share": st.column_config.NumberColumn("Route share", format="%.3f"),
            "TPRR": st.column_config.NumberColumn("TPRR", format="%.3f"),
        })
    row = mix[mix["Team"] == team]
    if not row.empty:
        wr, te, rb = row.iloc[0][["WR share", "TE share", "RB share"]]
        with st.container(horizontal=True):
            st.metric("To WRs", f"{wr:.0%}" if pd.notna(wr) else "—", border=True)
            st.metric("To TEs", f"{te:.0%}" if pd.notna(te) else "—", border=True)
            st.metric("To RBs", f"{rb:.0%}" if pd.notna(rb) else "—", border=True)

with tabs[1]:
    st.markdown("**Positional mix by club** — how much of the passing game each group owns")
    melted = mix.melt("Team", var_name="Group", value_name="Share").dropna()
    if melted.empty:
        st.warning("No positional mix available on this scope.")
    else:
        melted["Share%"] = melted["Share"] * 100
        import altair as alt
        chart = (alt.Chart(melted).mark_bar().encode(
            y=alt.Y("Team:N", sort=alt.EncodingSortField("Share%", op="max",
                                                         order="descending"), title=None),
            x=alt.X("Share%:Q", title="Share of team targets (%)", stack="normalize"),
            color=alt.Color("Group:N", title="Group",
                            scale=alt.Scale(scheme="tableau10")),
            tooltip=[alt.Tooltip("Team:N"), alt.Tooltip("Group:N"),
                     alt.Tooltip("Share%:Q", format=".1f")])
            .properties(height=nfl_charts.bar_height(melted["Team"].nunique())))
        st.altair_chart(chart, width="stretch")
        st.caption("League means: WR 58.8%, TE 23.9%, RB 17.3% on 2025.")

with tabs[2]:
    # Filtered inline rather than through the shared sidebar bar. A sidebar filter here
    # would sit above the club selector on the first tab and appear to govern it, while
    # actually only affecting this one -- two controls for the same-looking job.
    scoped = frame
    positions = sorted(frame["Pos"].dropna().unique())
    chosen = st.pills("Position", positions, selection_mode="multi", default=positions,
                      key="nfltgt_board_pos", persist_state="session")
    if chosen:
        scoped = scoped[scoped["Pos"].isin(chosen)]
    columns = [c for c in ("Name", "Team", "Pos", "Targets", "Target share",
                           "Route share", "TPRR", "aDOT", "Slot%") if c in scoped.columns]
    st.dataframe(nfl_charts.round_display(
        scoped[columns].sort_values("Target share", ascending=False)),
                 hide_index=True, width="stretch",
                 column_config={"Target share": st.column_config.ProgressColumn(
                     "Target share", min_value=0.0, max_value=1.0, format="%.1f%%")})
