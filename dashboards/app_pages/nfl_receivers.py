"""Every pass-catcher, and how he is actually used.

The axes are yours to pick, because "is this player in a good spot" is several different
questions in football too — volume, depth, alignment and efficiency are four separate
readings of the same man, and a fixed pair of axes silently chooses one of them.

**What the defaults choose, and why.** Routes per game against targets per route run: the
first is whether he is on the field, the second is whether the offence looks at him when he
is. Those are the two halves of opportunity, and opportunity is the half of this data that
predicts itself — deployment self-correlates at 0.82–0.95 year over year, efficiency at
about 0.6, and defensive allowed-rates at under 0.11.
"""

import pandas as pd
import streamlit as st

from dashboards import nfl_charts, nfl_pff

st.header("Receivers", anchor=False)
st.caption("PFF deployment profiles. Alignment, depth and route volume are the most stable "
           "numbers in this data; efficiency is weaker and defence-allowed is noise.")

seasons_available = nfl_pff.available_seasons()
if not seasons_available:
    st.info("No PFF receiving exports found. Drop them under `nfl/pff/` and reload.")
    st.stop()

AXES = ["Routes/G", "TPRR", "aDOT", "Slot%", "Wide%", "Inline%", "YPRR",
        "Route grade", "Deep rate", "YAC/rec", "Targets", "Routes"]

with st.sidebar:
    st.subheader("Scope", anchor=False)
    seasons = st.multiselect("Seasons", seasons_available,
                             default=seasons_available[:1], key="nflrec_seasons", persist_state="session",
                             help="Several seasons blend on recency weights 1.0 / 0.45 / "
                                  "0.20 — steadier, and it keeps a one-season sample from "
                                  "reading as a role.")
    if not seasons:
        st.warning("Pick at least one season.")
        st.stop()
    min_routes = st.slider("Minimum routes", 25, 400, 100, step=25, key="nflrec_routes", persist_state="session",
                           help="Below this a slot rate is describing a handful of snaps.")

frame = nfl_pff.receivers(tuple(sorted(seasons, reverse=True)), min_routes=min_routes)
if frame.empty:
    st.warning("Nothing clears that route threshold for those seasons.")
    st.stop()

frame = nfl_pff.scope_sidebar(frame, "nflrec")
if frame.empty:
    st.warning("Every player was filtered out.")
    st.stop()

with st.container(horizontal=True):
    st.metric("Players", len(frame), border=True)
    st.metric("Median routes/G", f"{frame['Routes/G'].median():.1f}", border=True)
    st.metric("Median aDOT", f"{frame['aDOT'].median():.1f}", border=True)
    st.metric("Median TPRR", f"{frame['TPRR'].median():.3f}", border=True)

with st.container(horizontal=True):
    x = st.selectbox("X axis", AXES, index=AXES.index("Routes/G"), key="nflrec_x", persist_state="session")
    y = st.selectbox("Y axis", AXES, index=AXES.index("TPRR"), key="nflrec_y", persist_state="session")
    size = st.selectbox("Size", ["Targets", "Routes", "G", "None"], key="nflrec_size", persist_state="session")

chart = nfl_charts.usage_scatter(frame, x, y, None if size == "None" else size)
if chart is None:
    st.warning(f"No player has both {x} and {y} on this scope.")
else:
    st.altair_chart(chart, width="stretch")
    st.caption(f"Dashed rule is the median **{y}** of the players drawn, so it moves with "
               f"the filters — it is a reading aid, not a league constant. "
               f"Colour is where he lines up most.")

st.subheader("The board", anchor=False)
columns = [c for c in ("Name", "Team", "Pos", "Alignment", "G", "Routes/G", "TPRR",
                       "aDOT", "Deep rate", "Slot%", "Wide%", "Inline%", "YPRR",
                       "Route grade", "YAC/rec", "Targets") if c in frame.columns]
sort_by = st.selectbox("Sort by", [c for c in columns if c not in
                                   ("Name", "Team", "Pos", "Alignment")],
                       key="nflrec_sort", persist_state="session")
st.dataframe(nfl_charts.round_display(
                 frame[columns].sort_values(sort_by, ascending=False)),
             hide_index=True, width="stretch",
             column_config={
                 "Slot%": st.column_config.ProgressColumn(
                     "Slot%", min_value=0, max_value=100, format="%.0f%%"),
                 "Wide%": st.column_config.ProgressColumn(
                     "Wide%", min_value=0, max_value=100, format="%.0f%%"),
                 "Inline%": st.column_config.ProgressColumn(
                     "Inline%", min_value=0, max_value=100, format="%.0f%%"),
                 "Deep rate": st.column_config.NumberColumn("Deep rate", format="%.3f"),
                 "TPRR": st.column_config.NumberColumn("TPRR", format="%.3f"),
             })

if len(seasons) > 1:
    st.caption(f"Blended across {', '.join(str(s) for s in sorted(seasons, reverse=True))} "
               "on recency weights, renormalised per player — a man present in one season "
               "of three is not averaged against two seasons of absence.")
