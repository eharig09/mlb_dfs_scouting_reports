"""Stacks — including the several a single club offers.

"Stack the Eagles" is a decision not yet made. A quarterback with five rosterable
pass-catchers offers twenty-five distinct two- and three-man combinations, and they differ
by thousands in salary and by tens of points in the share of the passing game they own.
This page enumerates them.

**Nothing here privileges the WR1.** Measured over 2023-25, a club's second receiver
correlates with its quarterback as strongly as its first (+0.359 against +0.367, higher on
the median) at about a quarter less target share. Paying up for the WR1 buys a correlation
already available one slot down, so the cheaper combination carrying the same share is the
whole point.

**Captured share is a fact, not a score.** Target concentration does not drive QB-receiver
correlation — +0.13, and not monotonic by tercile — so the page reports what a stack owns
and leaves the judgement to you.
"""

import altair as alt
import pandas as pd
import streamlit as st

from dashboards import nfl_charts, nfl_filters, nfl_slates, nfl_stacking

st.header("Stacks", anchor=False)
st.caption("Every QB + pass-catcher combination a club offers, priced.")

slates = nfl_slates.list_slates()
if slates.empty:
    st.info(f"No DK salary exports under `{nfl_slates.SALARY_DIR}`. Add one and reload.")
    st.stop()

with st.sidebar:
    st.subheader("Filters", anchor=False)
    path, row = nfl_filters.slate_picker("nflstk", label="Slate")
if path is None:
    path = slates.iloc[0]["Path"]
    row = slates.iloc[0]
    st.caption(f"Showing **{row['Label']}** ({row['Date']}) — pick another in the sidebar.")

players, _ = nfl_slates.slate_players(path)
board, _ = nfl_slates.attach_projection(players)
board = nfl_slates.add_value(board)

with st.sidebar:
    min_projection = st.slider("Minimum partner projection", 0.0, 15.0, 5.0, step=0.5,
                               key="nflstk_minproj", persist_state="session",
                               help="Keeps combinations built on players nobody would "
                                    "roster out of the enumeration.")

tabs = st.tabs(["Whole slate", "One club", "Compare clubs", "Bring-backs"])

with tabs[0]:
    st.markdown("**Every combination on the slate, side by side**")
    st.caption("The per-club tab answers *which stack of theirs*; this one answers "
               "*whose stack* — different questions, because a club's best combination can "
               "be the slate's fifth-best at twice the price.")

    everything = nfl_stacking.all_stack_combinations(board, min_projection=min_projection)
    if everything.empty:
        st.warning("No club offers a stack above that projection floor.")
    else:
        with st.container(horizontal=True):
            sizes_all = sorted(everything["Size"].unique())
            picked_all = st.pills("Stack size", sizes_all, selection_mode="multi",
                                  default=sizes_all, key="nflstk_allsize", persist_state="session")
            shortlist = st.toggle("Top per club only", value=True, key="nflstk_shortlist", persist_state="session",
                                  help="Every club represented, so the priciest offences "
                                       "do not crowd out the rest of the slate.")
            per_team = st.number_input("How many per club", 1, 5, 3, key="nflstk_perteam",
                                       disabled=not shortlist)

        scoped = everything[everything["Size"].isin(picked_all)] if picked_all else everything
        if shortlist:
            scoped = nfl_stacking.top_per_team(scoped, per_team=int(per_team))

        with st.container(horizontal=True):
            st.metric("Shown", len(scoped), border=True)
            st.metric("Clubs", scoped["Team"].nunique(), border=True)
            st.metric("Best projection", f"{scoped['Proj'].max():.2f}", border=True)
            st.metric("Best value", f"{scoped['Value'].max():.2f}", border=True)

        chart = (alt.Chart(scoped).mark_point(**nfl_charts.MARK, size=150)
                 .encode(
                     x=alt.X("Salary:Q", title="Combined salary",
                             scale=alt.Scale(zero=False)),
                     y=alt.Y("Proj:Q", title="Combined projection",
                             scale=alt.Scale(zero=False)),
                     color=alt.Color("Team:N", title="Club",
                                     scale=alt.Scale(scheme="category20")),
                     size=alt.Size("Captured share:Q", title="Share of team targets",
                                   scale=alt.Scale(range=[60, 400])),
                     tooltip=[alt.Tooltip("Team:N"), alt.Tooltip("Opp:N"),
                              alt.Tooltip("QB:N"), alt.Tooltip("Partners:N", title="With"),
                              alt.Tooltip("Salary:Q", format=",.0f"),
                              alt.Tooltip("Proj:Q", format=".2f"),
                              alt.Tooltip("Value:Q", title="Pts/$1k", format=".2f"),
                              alt.Tooltip("Captured share:Q", title="Captured",
                                          format=".1%")])
                 .properties(height=470))
        st.altair_chart(chart, width="stretch")
        st.caption("Up and to the left is more projection for less money.")

        columns = ["Team", "Opp", "QB", "Partners", "Size", "Salary", "Proj", "Value",
                   "Captured share"]
        sort_all = st.selectbox("Sort by", ["Proj", "Value", "Captured share", "Salary"],
                                key="nflstk_allsort", persist_state="session")
        st.dataframe(
            nfl_charts.round_display(
                scoped[columns].sort_values(sort_all, ascending=sort_all == "Salary")),
            hide_index=True, width="stretch",
            column_config={
                "Salary": st.column_config.NumberColumn("Salary", format="$%d"),
                "Proj": st.column_config.NumberColumn("Proj", format="%.2f"),
                "Value": st.column_config.NumberColumn("Pts/$1k", format="%.2f"),
                "Captured share": st.column_config.ProgressColumn(
                    "Captured share", min_value=0.0, max_value=1.0, format="%.1f%%"),
            })

with tabs[1]:
    teams = sorted(board.loc[board["Pos"] == "QB", "Team"].dropna().unique())
    if not teams:
        st.warning("No quarterbacks priced on this slate.")
        st.stop()
    team = st.selectbox("Club", teams, key="nflstk_team", persist_state="session")

    combos = nfl_stacking.stack_combinations(board, team, min_projection=min_projection)
    if combos.empty:
        st.warning(f"No stack combinations for {team} above that projection floor.")
    else:
        sizes = sorted(combos["Size"].unique())
        picked = st.pills("Stack size (players, QB included)", sizes,
                          selection_mode="multi", default=sizes, key="nflstk_size", persist_state="session")
        shown = combos[combos["Size"].isin(picked)] if picked else combos

        with st.container(horizontal=True):
            st.metric("Combinations", len(shown), border=True)
            st.metric("Cheapest", f"${shown['Salary'].min():,.0f}", border=True)
            st.metric("Priciest", f"${shown['Salary'].max():,.0f}", border=True)
            st.metric("Best projection", f"{shown['Proj'].max():.2f}", border=True)

        chart = (alt.Chart(shown).mark_point(**nfl_charts.MARK, size=150)
                 .encode(
                     x=alt.X("Salary:Q", title="Combined salary",
                             scale=alt.Scale(zero=False)),
                     y=alt.Y("Proj:Q", title="Combined projection",
                             scale=alt.Scale(zero=False)),
                     color=alt.Color("Size:O", title="Players",
                                     scale=alt.Scale(scheme="tableau10")),
                     size=alt.Size("Captured share:Q", title="Share of team targets",
                                   scale=alt.Scale(range=[60, 420])),
                     tooltip=[alt.Tooltip("Partners:N", title="With"),
                              alt.Tooltip("Salary:Q", format=",.0f"),
                              alt.Tooltip("Proj:Q", format=".1f"),
                              alt.Tooltip("Value:Q", title="Pts/$1k", format=".2f"),
                              alt.Tooltip("Captured share:Q", title="Captured",
                                          format=".1%")])
                 .properties(height=420))
        st.altair_chart(chart, width="stretch")
        st.caption("Up and to the left is more projection for less money. Point size is the "
                   "share of the club's projected targets the stack owns — a fact about "
                   "what you are buying, not a quality score.")

        columns = ["QB", "Partners", "Size", "Positions", "Salary", "Proj", "Value",
                   "Captured targets", "Captured share"]
        sort_by = st.selectbox("Sort by", ["Proj", "Value", "Captured share", "Salary"],
                               key="nflstk_sort", persist_state="session")
        st.dataframe(
            nfl_charts.round_display(
                shown[columns].sort_values(sort_by, ascending=sort_by == "Salary")),
            hide_index=True, width="stretch",
            column_config={
                "Salary": st.column_config.NumberColumn("Salary", format="$%d"),
                "Proj": st.column_config.NumberColumn("Proj", format="%.2f"),
                "Value": st.column_config.NumberColumn("Pts/$1k", format="%.2f"),
                "Captured share": st.column_config.ProgressColumn(
                    "Captured share", min_value=0.0, max_value=1.0, format="%.1f%%"),
            })

with tabs[2]:
    summary = nfl_stacking.team_stack_summary(board, min_projection=min_projection)
    if summary.empty:
        st.warning("No club offers a stack above that projection floor.")
    else:
        st.markdown("**The best stack each club offers, by size**")
        st.caption("The *best available*, not an average — a stack is something you pick, "
                   "so what matters about a club is the ceiling of what it offers.")
        columns = [c for c in ("Team", "Opp", "QB", "Best 2-man", "2-man cost",
                               "Best 3-man", "3-man cost", "Best 4-man", "4-man cost")
                   if c in summary.columns]
        st.dataframe(nfl_charts.round_display(summary[columns]), hide_index=True,
                     width="stretch",
                     column_config={
                         "Best 2-man": st.column_config.NumberColumn(format="%.1f"),
                         "Best 3-man": st.column_config.NumberColumn(format="%.1f"),
                         "Best 4-man": st.column_config.NumberColumn(format="%.1f"),
                         "2-man cost": st.column_config.NumberColumn(format="$%d"),
                         "3-man cost": st.column_config.NumberColumn(format="$%d"),
                         "4-man cost": st.column_config.NumberColumn(format="$%d"),
                     })

with tabs[3]:
    teams = sorted(board.loc[board["Pos"] == "QB", "Team"].dropna().unique())
    team = st.selectbox("Stacking", teams, key="nflstk_bbteam", persist_state="session")
    opponents = board.loc[board["Team"] == team, "Opp"].dropna()
    opponent = opponents.iloc[0] if not opponents.empty else None
    if not opponent:
        st.warning("No opponent found for that club on this slate.")
    else:
        st.markdown(f"**Bringing back from {opponent}**")
        st.caption("What a bring-back buys is correlation with the *opposing* quarterback: "
                   "the shootout that lifts your stack lifts him too. Ranked on projection, "
                   "because the measured spread between a club's own receivers is small "
                   "enough that price and volume decide it.")
        candidates = nfl_stacking.bring_back_candidates(board, team, opponent)
        if candidates.empty:
            st.warning(f"No priced pass-catchers for {opponent}.")
        else:
            st.dataframe(nfl_charts.round_display(candidates), hide_index=True,
                         width="stretch",
                         column_config={
                             "Salary": st.column_config.NumberColumn("Salary", format="$%d"),
                             "Proj": st.column_config.NumberColumn("Proj", format="%.1f"),
                             "Value": st.column_config.NumberColumn("Pts/$1k", format="%.2f"),
                         })
