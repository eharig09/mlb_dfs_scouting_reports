"""High-value opportunity: the work that scores, separated from the work that accumulates.

A player's yardage volume already describes most of what he does. What it does not describe
is **scoring equity** — the carries from the two-yard line and the targets inside the twenty
that turn into six points at a rate nothing else on the field matches. That is what this
page is for.

**Everything here is per game, never a share.** Measured year over year:

    red-zone carries / game     +0.85      inside-5 carries / game    +0.73
    red-zone targets / game     +0.80      inside-5 as a *share*      +0.33

The share collapses because it tracks whatever happened to the denominator: a back whose
early-down role shrank reads as having *gained* goal-line equity. A per-game volume asks the
only question that matters here — how often does he get the ball where it scores.

**Touchdowns themselves are not on this page as a projection.** Scoring rates self-correlate
at 0.03 to 0.24; the opportunity that precedes them correlates at 0.73 to 0.85. Chase the
opportunity, not last season's finishes.
"""

import altair as alt
import pandas as pd
import streamlit as st

from dashboards import nfl_charts, nfl_filters, nfl_pff, nfl_slates

st.header("High value", anchor=False)
st.caption("Inside-5 carries, red-zone carries and targets — the touches that carry scoring.")

seasons_available = nfl_pff.available_seasons("fantasy-stats-receiving")
if not seasons_available:
    st.info("No PFF red-zone splits found. `fantasy-stats-receiving` covers 2022-25 — drop "
            "the exports under `nfl/pff/` and reload.")
    st.stop()

with st.sidebar:
    st.subheader("Filters", anchor=False)
    seasons = st.multiselect("Seasons", seasons_available, default=seasons_available[:1],
                             key="nflhv_seasons", persist_state="session")
    if not seasons:
        st.warning("Pick at least one season.")
        st.stop()
    min_games = st.slider("Minimum games", 1, 17, 6, key="nflhv_games", persist_state="session")

frame = nfl_pff.high_value(tuple(sorted(seasons, reverse=True)), min_games=min_games)
if frame.empty:
    st.warning("Nothing clears that games threshold.")
    st.stop()

frame, slate_path, _ = nfl_filters.sidebar(frame, "nflhv", show=("slate", "position", "team"),
                                           positions=("RB", "WR", "TE", "QB"))
if slate_path is not None:
    frame = nfl_slates.attach_salary(frame, slate_path)
    frame = nfl_slates.add_value(frame, projection="HV touches/G")
if frame.empty:
    st.warning("Every player was filtered out.")
    st.stop()

with st.container(horizontal=True):
    st.metric("Players", len(frame), border=True)
    st.metric("Median HV/G", f"{frame['HV touches/G'].median():.2f}", border=True)
    st.metric("Most i5 carries/G", f"{frame['i5 car/G'].max():.2f}", border=True)
    st.metric("Most RZ targets/G", f"{frame['RZ tgt/G'].max():.2f}", border=True)

tabs = st.tabs(["All high-value work", "Inside the 5", "Red zone", "Board"])

TOOLTIP_ROUND = {"i5 car/G": "%.2f", "RZ car/G": "%.2f", "RZ tgt/G": "%.2f",
                 "EZ tgt/G": "%.2f", "HV touches/G": "%.2f"}


def _table(part, columns, sort):
    columns = [c for c in columns if c in part.columns]
    st.dataframe(
        nfl_charts.round_display(part[columns].sort_values(sort, ascending=False)),
        hide_index=True, width="stretch",
        column_config={c: st.column_config.NumberColumn(c, format=f)
                       for c, f in TOOLTIP_ROUND.items() if c in columns})


with tabs[0]:
    st.markdown("**Everything that counts as high-value work, per game**")
    st.caption("`HV touches/G` is inside-5 carries plus red-zone targets — the two ways a "
               "player gets scoring equity his yardage volume does not already describe.")
    # **Only men with a claim on both axes are plotted.** A receiver with zero inside-5
    # carries is not a data point on a carries axis, he is a mark stacked on x=0 with two
    # hundred others -- and two hundred of them hide the handful of players the panel exists
    # to find. Everyone is still in the table below.
    both = frame[(frame["i5 car/G"] > 0) & (frame["RZ tgt/G"] > 0)]
    st.caption(f"Plotting the **{len(both)}** players with both a goal-line carry and a "
               f"red-zone target. The rest own one axis or neither, and stack on a zero "
               f"line where they hide the players this panel is for — they are all still "
               f"in the table below.")
    chart = nfl_charts.usage_scatter(both, "i5 car/G", "RZ tgt/G", size="G", color="Pos")
    if chart is not None:
        st.altair_chart(chart, width="stretch")
        st.caption("Backs own the horizontal axis, receivers the vertical. The few who own "
                   "both are the ones worth finding — they hold two independent claims on "
                   "the same end zone.")
    else:
        st.info("Nobody on this scope holds both a goal-line carry and a red-zone target.")
    _table(frame, ["Name", "Team", "Pos", "Opp", "Salary", "G", "HV touches/G",
                   "i5 car/G", "RZ car/G", "RZ tgt/G", "EZ tgt/G", "Carries/G",
                   "Targets/G"], "HV touches/G")

with tabs[1]:
    st.markdown("**Inside the 5** — the highest-leverage carry in football")
    inside = frame[frame["i5 car/G"] > 0]
    if inside.empty:
        st.info("Nobody on this scope has an inside-5 carry.")
    else:
        top = inside.nlargest(min(25, len(inside)), "i5 car/G")
        chart = (alt.Chart(top).mark_bar(cornerRadiusEnd=3).encode(
            y=alt.Y("Name:N", sort="-x", title=None),
            x=alt.X("i5 car/G:Q", title="Inside-5 carries per game"),
            color=alt.Color("Team:N", legend=None,
                            scale=alt.Scale(scheme="category20")),
            tooltip=[alt.Tooltip("Name:N"), alt.Tooltip("Team:N"),
                     alt.Tooltip("i5 car/G:Q", format=".2f"),
                     alt.Tooltip("RZ car/G:Q", format=".2f"),
                     alt.Tooltip("G:Q")])
            .properties(height=nfl_charts.bar_height(len(top))))
        st.altair_chart(chart, width="stretch")
        _table(inside, ["Name", "Team", "Pos", "Opp", "Salary", "G", "i5 car/G",
                        "RZ car/G", "Carries/G", "RZ rush TD/G"], "i5 car/G")
        st.caption("Self-correlates at **+0.73** year over year as a per-game rate — and at "
                   "only +0.33 as a share of a back's carries, which is why it is not "
                   "presented that way.")

with tabs[2]:
    st.markdown("**Red-zone carries and targets**")
    view = st.segmented_control("Show", ["Targets", "Carries", "Both"], default="Both",
                                key="nflhv_rzview", persist_state="session") or "Both"
    if view in ("Targets", "Both"):
        st.markdown("*Red-zone targets per game* — self-correlates at +0.80")
        _table(frame[frame["RZ tgt/G"] > 0],
               ["Name", "Team", "Pos", "Opp", "Salary", "G", "RZ tgt/G", "EZ tgt/G",
                "Targets/G"], "RZ tgt/G")
    if view in ("Carries", "Both"):
        st.markdown("*Red-zone carries per game* — self-correlates at +0.85, the most "
                    "stable number in this data")
        _table(frame[frame["RZ car/G"] > 0],
               ["Name", "Team", "Pos", "Opp", "Salary", "G", "RZ car/G", "i5 car/G",
                "Carries/G"], "RZ car/G")

with tabs[3]:
    if slate_path is not None and "Value" in frame.columns:
        st.caption("`Value` here is **high-value touches per $1,000**, not projected points "
                   "— a different question from the Value page, and the one this page is "
                   "about: who is cheapest per trip to the end zone.")
    columns = ["Name", "Team", "Pos", "Opp", "Salary", "Band", "G", "HV touches/G",
               "i5 car/G", "RZ car/G", "RZ tgt/G", "EZ tgt/G", "RZ rush TD/G",
               "Carries/G", "Targets/G", "Value"]
    sort_by = st.selectbox("Sort by", [c for c in columns if c in frame.columns
                                       and c not in ("Name", "Team", "Pos", "Opp", "Band")],
                           key="nflhv_sort", persist_state="session")
    _table(frame, columns, sort_by)
