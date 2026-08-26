"""What each defence allows, and what to actually expect from it.

**Both numbers are here on purpose, and they disagree.** `Allowed/G` is what a defence gave
up — a fact about the season. `Projected` is that figure regressed toward the league mean by
the position's own measured reliability, which is what you should expect next.

Measured over 2021-25, mean year-over-year correlation of DK points allowed per game:

    RB  +0.30      TE  +0.19      WR  +0.11      QB  +0.09

So a defence five points above average against receivers projects about half a point above
average — not five. Reading `Allowed/G` as a forecast is the single most common way to lose
money on a matchup table, which is why the projected column sits next to it rather than
somewhere else.

The spread is nonetheless real: WR allowed runs 26.8 to 37.2 DK points a game between the
tenth and ninetieth percentile, so half a point of shrunk edge is small but not nothing.
"""

import altair as alt
import pandas as pd
import streamlit as st

from dashboards import nfl_charts, nfl_filters, nfl_league, nfl_slates

st.header("Defence", anchor=False)
st.caption("What defences allow by position — and what that is actually worth as a forecast.")

with st.sidebar:
    st.subheader("Filters", anchor=False)
    seasons = st.multiselect("Seasons", list(range(2025, 2020, -1)), default=[2025],
                             key="nfldef_seasons", persist_state="session",
                             help="Several seasons pool into one rate rather than blending.")
if not seasons:
    st.warning("Pick at least one season.")
    st.stop()

allowed = nfl_league.defense_allowed(tuple(sorted(seasons, reverse=True)))
if allowed.empty:
    st.warning("No scored box scores for those seasons.")
    st.stop()

with st.sidebar:
    path, slate_row = nfl_filters.slate_picker("nfldef", label="Slate")

tabs = st.tabs(["This slate's matchups", "Matchup vs stack", "By position",
                "Every defence"])

with tabs[0]:
    if path is None:
        st.info("Pick a slate in the sidebar to see who each offence draws this week.")
    else:
        players, _ = nfl_slates.slate_players(path)
        matchups = nfl_league.matchup_table(allowed, players)
        if matchups.empty:
            st.warning("Could not line this slate's clubs up against the allowed table.")
        else:
            position = st.segmented_control("Position", nfl_league.POSITIONS,
                                            default="WR", key="nfldef_slatepos", persist_state="session")
            part = matchups[matchups["Pos"] == (position or "WR")]
            st.markdown(f"**Offences facing the softest {position or 'WR'} matchups**")
            st.caption("`Rank 1` means that opponent allowed the *most* to the position, so "
                       "a low rank is a good spot for your player.")
            st.dataframe(
                nfl_charts.round_display(
                    part.sort_values("Opp allows/G", ascending=False)), hide_index=True,
                width="stretch",
                column_config={
                    "Opp allows/G": st.column_config.NumberColumn("Opp allows/G",
                                                                  format="%.1f"),
                    "vs League": st.column_config.NumberColumn("vs League", format="%+.1f"),
                    "Projected": st.column_config.NumberColumn("Projected", format="%.1f"),
                })
            chart = (alt.Chart(part).mark_bar(cornerRadiusEnd=3).encode(
                y=alt.Y("Team:N", sort="-x", title=None),
                x=alt.X("Opp allows/G:Q", title=f"What the opponent allows to {position}"),
                color=alt.Color("vs League:Q", title="vs league",
                                scale=alt.Scale(scheme="blueorange", domainMid=0)),
                tooltip=[alt.Tooltip("Team:N", title="Offence"),
                         alt.Tooltip("Opp:N", title="Defence"),
                         alt.Tooltip("Opp allows/G:Q", format=".1f"),
                         alt.Tooltip("Projected:Q", format=".1f"),
                         alt.Tooltip("Rank:Q")])
                .properties(height=nfl_charts.bar_height(len(part))))
            st.altair_chart(chart, width="stretch")

with tabs[1]:
    if path is None:
        st.info("Pick a slate in the sidebar — this panel needs prices to cost a stack.")
    else:
        st.markdown("**Is the matchup soft, and is the stack affordable?**")
        st.caption("The two halves of a stacking decision on the same axes. Right is a "
                   "defence that gives up more to the position; up is a better stack "
                   "facing it.")
        from dashboards import nfl_slates as _slates, nfl_stacking as _stacking

        players_all, _ = _slates.slate_players(path)
        board, _ = _slates.attach_projection(players_all)
        board = _slates.add_value(board)
        summary = _stacking.team_stack_summary(board, min_projection=5.0)

        with st.container(horizontal=True):
            position = st.segmented_control("Position", nfl_league.POSITIONS, default="WR",
                                            key="nfldef_stackpos",
                                            persist_state="session") or "WR"
            size = st.segmented_control("Stack size", ["Best 2-man", "Best 3-man",
                                                       "Best 4-man"],
                                        default="Best 3-man", key="nfldef_stacksize",
                                        persist_state="session") or "Best 3-man"
            measure = st.segmented_control("Y axis", ["Stack proj", "Stack value"],
                                           default="Stack proj", key="nfldef_stackmeasure",
                                           persist_state="session") or "Stack proj"

        paired = nfl_league.allowed_against_stacks(allowed, summary, position, size)
        if paired.empty:
            st.warning("Could not pair this slate's stacks against the allowed table.")
        else:
            chart = nfl_charts.usage_scatter(
                nfl_charts.round_display(paired.rename(columns={"Team": "Name"})
                                         .assign(Team=paired["Team"])),
                "Opp allows/G", measure, size=None, color="Team", labels=True,
                extra_tooltip=("Opp", "QB", "Stack cost", "Projected allowed",
                               "Opp allows rank"))
            if chart is not None:
                st.altair_chart(chart, width="stretch")
                st.caption("Top right is a good stack into a soft matchup — but read the "
                           "horizontal axis with the reliability warning below in mind. "
                           "`Projected allowed` in the tooltip is the same number regressed "
                           "to what it is worth as a forecast.")
            st.dataframe(
                nfl_charts.round_display(paired.sort_values(measure, ascending=False)),
                hide_index=True, width="stretch",
                column_config={
                    "Stack cost": st.column_config.NumberColumn("Stack cost", format="$%d"),
                    "Opp allows/G": st.column_config.NumberColumn(format="%.2f"),
                    "Projected allowed": st.column_config.NumberColumn(format="%.2f"),
                    "Stack proj": st.column_config.NumberColumn(format="%.2f"),
                    "Stack value": st.column_config.NumberColumn("Pts/$1k", format="%.2f"),
                })
            st.info(f"{position} points allowed self-correlates at "
                    f"{nfl_league.RELIABILITY[position]:+.2f} year over year. The vertical "
                    f"axis is solid; the horizontal one is a tie-break.",
                    icon=":material/info:")

with tabs[2]:
    position = st.segmented_control("Position", nfl_league.POSITIONS, default="WR",
                                    key="nfldef_pos", persist_state="session") or "WR"
    part = allowed[allowed["Pos"] == position]
    reliability = nfl_league.RELIABILITY.get(position)
    in_season = nfl_league.IN_SEASON_RELIABILITY.get(position)
    st.warning(
        f"**{position} points allowed self-correlates at r = {reliability:+.2f}** year over "
        f"year (and {in_season:+.2f} within a single season). The `Projected` column is the "
        f"allowed figure regressed toward the league mean by exactly that much — which is "
        f"why it is so much flatter than what actually happened.",
        icon=":material/warning:")
    chart = nfl_charts.tendency_bars(part, "Allowed/G",
                                     f"DK points allowed per game to {position}")
    if chart is not None:
        st.altair_chart(chart, width="stretch")
    st.dataframe(
        nfl_charts.round_display(
            part[["Team", "Allowed/G", "Rank", "vs League", "Projected", "Games"]]),
        hide_index=True, width="stretch",
        column_config={
            "Allowed/G": st.column_config.NumberColumn("Allowed/G", format="%.1f"),
            "vs League": st.column_config.NumberColumn("vs League", format="%+.1f"),
            "Projected": st.column_config.NumberColumn("Projected", format="%.1f"),
        })

with tabs[3]:
    wide = allowed.pivot(index="Team", columns="Pos", values="Allowed/G").reset_index()
    projected = allowed.pivot(index="Team", columns="Pos", values="Projected")
    st.markdown("**Every defence, every position**")
    st.dataframe(wide.sort_values("WR", ascending=False), hide_index=True, width="stretch",
                 column_config={p: st.column_config.NumberColumn(p, format="%.1f")
                                for p in nfl_league.POSITIONS if p in wide.columns})
    st.markdown("**The same table, regressed to what it is worth as a forecast**")
    st.caption("Note how much narrower the spread is. That is the measurement, not a "
               "smoothing choice.")
    st.dataframe(projected.reset_index().sort_values("WR", ascending=False),
                 hide_index=True, width="stretch",
                 column_config={p: st.column_config.NumberColumn(p, format="%.1f")
                                for p in nfl_league.POSITIONS if p in projected.columns})
