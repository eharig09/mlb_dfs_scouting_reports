"""Clubs: what they scored, what they allowed, and how the market saw it.

Built from schedules and scored box scores, so what the totals market expected and what
actually happened sit in the same row. `vs Total` is the average gap between a club's games
and their closing number — a club consistently over it has been in more scoring than the
market priced, which is a fact about the games rather than a prediction about the next one.

The positional tables are the two halves of the same ledger: `team_offense` is what a club
produced in DK points by position, `defense_allowed` is what it gave up. Same scoring, same
source, so an offence and a defence can be read against each other directly.
"""

import altair as alt
import pandas as pd
import streamlit as st

from dashboards import nfl_charts, nfl_filters, nfl_league, nfl_slates

st.header("Teams", anchor=False)
st.caption("Records, scoring, and DK points produced and allowed by position.")

with st.sidebar:
    st.subheader("Filters", anchor=False)
    seasons = st.multiselect("Seasons", list(range(2025, 2020, -1)), default=[2025],
                             key="nflteam_seasons", persist_state="session")
if not seasons:
    st.warning("Pick at least one season.")
    st.stop()

seasons = tuple(sorted(seasons, reverse=True))
summary = nfl_league.team_summary(seasons)
if summary.empty:
    st.warning("No completed games for those seasons.")
    st.stop()

with st.sidebar:
    path, _ = nfl_filters.slate_picker("nflteam", label="Slate")
    only_slate = False
    if path is not None:
        only_slate = st.toggle("Only clubs on this slate", value=True,
                               key="nflteam_only", persist_state="session")

if path is not None and only_slate:
    teams = set(nfl_slates.teams_on_slate(path))
    summary = summary[summary["Team"].isin(teams)]
    if summary.empty:
        st.warning("No club on that slate has completed games in these seasons.")
        st.stop()

with st.container(horizontal=True):
    st.metric("Clubs", len(summary), border=True)
    st.metric("Best margin", f"{summary['Margin/G'].max():+.1f}", border=True)
    st.metric("Highest scoring", f"{summary['PF/G'].max():.1f}", border=True)
    st.metric("Median combined", f"{summary['Combined/G'].median():.1f}", border=True)

tabs = st.tabs(["Standings", "Scoring environment", "By position", "Game log"])

with tabs[0]:
    st.dataframe(nfl_charts.round_display(summary), hide_index=True,
                 width="stretch",
                 column_config={
                     "PF/G": st.column_config.NumberColumn("PF/G", format="%.1f"),
                     "PA/G": st.column_config.NumberColumn("PA/G", format="%.1f"),
                     "Margin/G": st.column_config.NumberColumn("Margin/G", format="%+.1f"),
                     "Combined/G": st.column_config.NumberColumn("Combined/G", format="%.1f"),
                     "Total line": st.column_config.NumberColumn("Total line", format="%.1f"),
                     "vs Total": st.column_config.NumberColumn("vs Total", format="%+.1f"),
                 })

with tabs[1]:
    st.markdown("**Points for against points allowed**")
    chart = (alt.Chart(summary).mark_point(**nfl_charts.MARK, size=170)
             .encode(
                 x=alt.X("PF/G:Q", title="Points scored per game",
                         scale=alt.Scale(zero=False)),
                 y=alt.Y("PA/G:Q", title="Points allowed per game",
                         scale=alt.Scale(zero=False)),
                 color=alt.Color("Margin/G:Q", title="Margin",
                                 scale=alt.Scale(scheme="blueorange", domainMid=0)),
                 size=alt.Size("Combined/G:Q", title="Combined/G",
                               scale=alt.Scale(range=[80, 400])),
                 tooltip=[alt.Tooltip("Team:N"), alt.Tooltip("W:Q"), alt.Tooltip("L:Q"),
                          alt.Tooltip("PF/G:Q", format=".1f"),
                          alt.Tooltip("PA/G:Q", format=".1f"),
                          alt.Tooltip("Combined/G:Q", format=".1f"),
                          alt.Tooltip("vs Total:Q", format="+.1f")])
             .properties(height=440))
    labels = (alt.Chart(summary).mark_text(align="left", dx=8, fontSize=10)
              .encode(x="PF/G:Q", y="PA/G:Q", text="Team:N"))
    st.altair_chart(chart + labels, width="stretch")
    st.caption("**Top right is the DFS corner**: a club that scores a lot and whose "
               "opponents score a lot too. That is the shootout you want a stack inside, "
               "and it is the opposite of the best *team* on the board — which sits bottom "
               "right. Point size is total scoring in its games.")

    st.markdown("**Against the closing total**")
    st.dataframe(summary[["Team", "Combined/G", "Total line", "vs Total"]]
                 .sort_values("vs Total", ascending=False),
                 hide_index=True, width="stretch",
                 column_config={
                     "Combined/G": st.column_config.NumberColumn(format="%.1f"),
                     "Total line": st.column_config.NumberColumn(format="%.1f"),
                     "vs Total": st.column_config.NumberColumn(format="%+.1f"),
                 })

with tabs[2]:
    produced = nfl_league.team_offense(seasons)
    allowed = nfl_league.defense_allowed(seasons)
    if path is not None and only_slate:
        produced = produced[produced["Team"].isin(teams)]
        allowed = allowed[allowed["Team"].isin(teams)]
    view = st.segmented_control("Side", ["Produced", "Allowed"], default="Produced",
                                key="nflteam_side", persist_state="session") or "Produced"
    frame = produced if view == "Produced" else allowed
    column = "Produced/G" if view == "Produced" else "Allowed/G"
    if frame.empty:
        st.warning("Nothing to show for that scope.")
    else:
        wide = frame.pivot(index="Team", columns="Pos", values=column).reset_index()
        st.markdown(f"**DK points {view.lower()} per game, by position**")
        st.dataframe(wide.sort_values(wide.columns[1], ascending=False), hide_index=True,
                     width="stretch",
                     column_config={p: st.column_config.NumberColumn(p, format="%.1f")
                                    for p in nfl_league.POSITIONS if p in wide.columns})
        if view == "Allowed":
            st.caption("These are descriptive. Year over year they self-correlate at RB "
                       "+0.30, TE +0.19, WR +0.11, QB +0.09 — see the Defence page for the "
                       "regressed version.")

with tabs[3]:
    games = nfl_league.team_results(seasons)
    if path is not None and only_slate:
        games = games[games["Team"].isin(teams)]
    team = st.selectbox("Club", sorted(games["Team"].dropna().unique()), key="nflteam_log", persist_state="session")
    log = games[games["Team"] == team].sort_values(["Season", "Week"])
    st.dataframe(log[["Season", "Week", "Opp", "Home", "Result", "For", "Against",
                      "Margin", "Combined", "Total"]],
                 hide_index=True, width="stretch",
                 column_config={
                     "Margin": st.column_config.NumberColumn("Margin", format="%+d"),
                     "Total": st.column_config.NumberColumn("Total line", format="%.1f"),
                 })
