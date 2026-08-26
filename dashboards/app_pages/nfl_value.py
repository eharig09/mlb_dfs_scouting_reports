"""Projections against price: what everyone costs, and what the projection says they return.

**Two value numbers, not one.** Raw points per $1,000 always favours the cheapest man on the
board — a punt at replacement level outranks every real starter, every week, on every slate.
`Band rank` asks the question a roster actually asks instead: *who is the best use of this
slot at this price*. Read the scatter for shape and the band rank for choices.

**A blank projection is not a zero.** PFF projects about half a DK slate; the rest are third
and fourth stringers it does not carry. They keep their salary and their profile and are
marked as unprojected, because "we have no number for him" and "we expect him to score
nothing" are different claims and only one of them is true.
"""

import altair as alt
import pandas as pd
import streamlit as st

from dashboards import nfl_charts, nfl_filters, nfl_slates

st.header("Value", anchor=False)
st.caption("PFF per-game projections against DraftKings salary.")

slates = nfl_slates.list_slates()
if slates.empty:
    st.info(f"No DK salary exports under `{nfl_slates.SALARY_DIR}`. Add one and reload.")
    st.stop()

with st.sidebar:
    st.subheader("Filters", anchor=False)
    path, row = nfl_filters.slate_picker("nflval", label="Slate")
if path is None:
    path = slates.iloc[0]["Path"]
    row = slates.iloc[0]
    st.caption(f"Showing **{row['Label']}** ({row['Date']}) — pick another in the sidebar.")

players, report = nfl_slates.slate_players(path)
board, source = nfl_slates.attach_projection(players)
board = nfl_slates.add_value(board)

board, _, _ = nfl_filters.sidebar(board, "nflval", show=("position", "team", "game", "band"),
                                  pos_column="Pos")
if board.empty:
    st.warning("Every player was filtered out.")
    st.stop()

projected = board.dropna(subset=["Proj"])

with st.container(horizontal=True):
    st.metric("Priced", len(board), border=True)
    st.metric("Projected", f"{len(projected)}", border=True)
    if not projected.empty:
        st.metric("Median value", f"{projected['Value'].median():.2f}", border=True)
        st.metric("Best value", f"{projected['Value'].max():.2f}", border=True)

if source:
    st.caption(f"Projections from `{source}` — points per game, not per season.")

tabs = st.tabs(["Price vs projection", "Best of each band", "Board", "Unprojected"])

# One control, read by both the scatter and the table below it, so sorting the board
# re-plots the chart rather than leaving the two showing different questions.
SORTS = ["Value", "Proj", "Salary", "Band rank"]
sort_by = st.session_state.get("nflval_sort", "Value")
if sort_by not in SORTS:
    sort_by = "Value"

with tabs[0]:
    if projected.empty:
        st.warning("Nothing on this slate carries a projection.")
    else:
        y_axis = "Value" if sort_by in ("Value", "Band rank") else sort_by
        if y_axis == "Salary":
            y_axis = "Proj"
        chart = nfl_charts.usage_scatter(
            nfl_charts.round_display(projected.rename(columns={"Salary": "Salary$"})),
            "Salary$", y_axis, size=None, color="Pos", labels=True,
            extra_tooltip=("Value", "Proj", "Salary$", "Targets"))
        if chart is not None:
            st.altair_chart(chart, width="stretch")
            st.caption(f"Plotting **{y_axis}** against salary because that is what the "
                       f"board below is sorted by — the two stay in step. Dashed rule is "
                       f"the median of the players drawn, so it moves with your filters.")
        st.markdown("**Value by position**")
        box = (alt.Chart(nfl_charts.round_display(projected))
               .mark_boxplot(extent="min-max").encode(
            x=alt.X("Pos:N", title=None),
            y=alt.Y("Value:Q", title="Points per $1,000"),
            color=alt.Color("Pos:N", legend=None, scale=alt.Scale(scheme="tableau10")))
            .properties(height=300))
        st.altair_chart(box, width="stretch")
        st.caption("Quarterbacks dominate raw points per dollar on every slate — they score "
                   "more and cost less per point than the positions you need three of. "
                   "That is why the next tab ranks inside a band instead.")

with tabs[1]:
    if projected.empty:
        st.warning("Nothing on this slate carries a projection.")
    else:
        st.markdown("**The best use of each price band**")
        for band in nfl_slates.BAND_DISPLAY_ORDER:
            part = projected[projected["Band"] == band]
            if part.empty:
                continue
            st.markdown(f"**{band}**")
            columns = [c for c in ("Name", "Pos", "Team", "Opp", "Salary", "Proj",
                                   "Value", "Targets", "Rush att") if c in part.columns]
            st.dataframe(nfl_charts.round_display(
                part.nlargest(min(8, len(part)), "Value")[columns]),
                         hide_index=True, width="stretch",
                         column_config={
                             "Salary": st.column_config.NumberColumn("Salary", format="$%d"),
                             "Proj": st.column_config.NumberColumn("Proj", format="%.1f"),
                             "Value": st.column_config.NumberColumn("Pts/$1k", format="%.2f"),
                         })

with tabs[2]:
    columns = [c for c in ("Name", "Pos", "Team", "Opp", "Game", "Salary", "Band",
                           "Proj", "Value", "Band rank", "Targets", "Rec", "Rush att",
                           "Pass att", "Status") if c in board.columns]
    sort_by = st.selectbox("Sort by", SORTS, key="nflval_sort", persist_state="session",
                           help="Also sets the y axis of the scatter on the first tab.")
    ascending = sort_by == "Band rank"
    st.dataframe(nfl_charts.round_display(
        board[columns].sort_values(sort_by, ascending=ascending, na_position="last")),
                 hide_index=True, width="stretch",
                 column_config={
                     "Salary": st.column_config.NumberColumn("Salary", format="$%d"),
                     "Proj": st.column_config.NumberColumn("Proj", format="%.1f"),
                     "Value": st.column_config.NumberColumn("Pts/$1k", format="%.2f"),
                 })

with tabs[3]:
    missing = board[board["Proj"].isna()]
    st.caption("Priced by DraftKings, not projected by PFF. Kept rather than dropped — a "
               "blank is 'no number', which is different from 'expected to score nothing'.")
    columns = [c for c in ("Name", "Pos", "Team", "Opp", "Salary", "Band", "Status",
                           "AvgPts") if c in missing.columns]
    st.dataframe(nfl_charts.round_display(
        missing[columns].sort_values("Salary", ascending=False)),
                 hide_index=True, width="stretch",
                 column_config={"Salary": st.column_config.NumberColumn("Salary",
                                                                       format="$%d")})
    if report["questionable"]:
        st.info(f"DK lists {len(report['questionable'])} player(s) as questionable and they "
                f"are still on this board: {', '.join(report['questionable'][:8])}"
                + (" …" if len(report["questionable"]) > 8 else ""),
                icon=":material/help:")
