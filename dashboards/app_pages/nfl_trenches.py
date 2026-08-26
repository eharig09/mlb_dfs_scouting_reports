"""Projected offensive lines, and what each projection is actually standing on.

Built **bottom-up from the five men**, because that is what the data supports: an individual
lineman's grade self-correlates at +0.60 year over year, a team's *unit* grade at +0.46 and
+0.18 across the two season pairs on file. The player is the stable unit; the aggregate
churns underneath it.

`Basis` is first class here for the same reason it is on the skill board — "PFF graded him
at 78" and "we inferred it from a third-round pick" must never look alike on a page.

**Returning starters is deliberately not an input.** Continuity correlates with next
season's unit grade (+0.40, +0.17), but that is confounding rather than signal: good lines
keep their starters. Against the *change* in unit grade it is −0.16 and +0.01, and lines
with two or fewer returning starters improved more than lines with four or more in both
pairs. It is shown as description and never applied.
"""

import streamlit as st

from dashboards import nfl_charts, nfl_pff
from nfl import pffdata

st.header("Trenches", anchor=False)
st.caption("Projected starting fives, regressed from PFF grades, movement and draft capital.")

blocking_seasons = nfl_pff.available_seasons("offense_blocking")
if not blocking_seasons:
    st.info("No PFF blocking exports found. Drop them under `nfl/pff/` and reload.")
    st.stop()

with st.sidebar:
    st.subheader("Scope", anchor=False)
    season = st.number_input("Project season", min_value=min(blocking_seasons),
                             max_value=max(blocking_seasons) + 2,
                             value=max(blocking_seasons) + 1, step=1, key="nflol_season",
                             help="Grades come from the seasons before this one; the "
                                  "starting five comes from that season's depth charts.")

try:
    line, unit = nfl_pff.lines(int(season))
except pffdata.PffDataError as error:
    st.warning(f"No blocking export to project from: {error}")
    st.stop()
except Exception as error:                      # depth charts need nflverse
    st.warning(f"Could not project {season}: {error}\n\n"
               "The starting five comes from nflverse depth charts, so this needs network "
               "the first time a season is read.")
    st.stop()

if line.empty:
    st.warning(f"No projected line for {season}.")
    st.stop()

with st.container(horizontal=True):
    st.metric("Clubs", unit["Team"].nunique(), border=True)
    st.metric("Starters graded", int((line["Basis"] == "grade").sum()), border=True)
    st.metric("On a new club", int((line["Basis"] == "grade (moved)").sum()), border=True)
    st.metric("Rookies", int((line["Basis"] == "draft capital").sum()), border=True)

tabs = st.tabs(["League", "One club", "Every starter"])

with tabs[0]:
    chart = nfl_charts.line_strength(unit)
    if chart is not None:
        st.altair_chart(chart, width="stretch")
        st.caption("Colour is how many of the five carry a real PFF grade — a high bar on "
                   "thin colour is a projection resting on draft capital and replacement "
                   "level, not on evidence.")
    st.dataframe(unit.sort_values("Proj grade", ascending=False), hide_index=True,
                 width="stretch",
                 column_config={
                     "Proj grade": st.column_config.NumberColumn("Proj grade", format="%.1f"),
                     "Proj pass blk": st.column_config.NumberColumn("Pass blk", format="%.1f"),
                     "Proj run blk": st.column_config.NumberColumn("Run blk", format="%.1f"),
                 })
    st.caption("`Returning` is shown but never applied — see the note at the top of this "
               "page. It correlates with next season's grade only because good lines keep "
               "their starters.")

with tabs[1]:
    team = st.selectbox("Club", sorted(line["Team"].dropna().unique()), key="nflol_team", persist_state="session")
    one = line[line["Team"] == team]
    order = {"LT": 0, "LG": 1, "C": 2, "RG": 3, "RT": 4}
    one = one.assign(_o=one["Spot"].map(order)).sort_values("_o").drop(columns="_o")
    columns = [c for c in ("Spot", "Name", "Basis", "Grade", "Proj grade",
                           "Proj pass blk", "Proj run blk", "Last team", "Moved",
                           "Pick") if c in one.columns]
    st.dataframe(one[columns], hide_index=True, width="stretch",
                 column_config={
                     "Grade": st.column_config.NumberColumn("Last grade", format="%.1f"),
                     "Proj grade": st.column_config.NumberColumn("Projected", format="%.1f"),
                 })
    moved = one[one["Basis"] == "grade (moved)"]
    if not moved.empty:
        st.info(f"{len(moved)} starter(s) on a new club. A lineman's grade only half "
                "travels — stayers correlate at +0.60 year over year, movers at +0.27 to "
                "+0.38 — so those grades are regressed roughly twice as hard.",
                icon=":material/swap_horiz:")

with tabs[2]:
    scoped = line
    columns = [c for c in ("Team", "Spot", "Name", "Basis", "Grade", "Proj grade",
                           "Proj pass blk", "Proj run blk", "Last team", "Pick")
               if c in scoped.columns]
    basis = st.multiselect("Basis", sorted(scoped["Basis"].dropna().unique()),
                           key="nflol_basis", persist_state="session", placeholder="Every basis")
    if basis:
        scoped = scoped[scoped["Basis"].isin(basis)]
    st.dataframe(scoped[columns].sort_values("Proj grade", ascending=False),
                 hide_index=True, width="stretch")
