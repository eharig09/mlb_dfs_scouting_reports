"""The run game, and who owns the work that scores.

**High-value touches are per game, never a share.** Inside-5 carries per game self-correlate
at +0.73 year over year; the same fact expressed as a *share of a back's carries* falls to
+0.33, because the share tracks whatever happened to the denominator — a back whose
early-down role shrank reads as having *gained* goal-line equity. Every column here is a
rate per game for that reason.

**Rushing efficiency is on this page but should not be leaned on.** Yards per carry
self-correlates at +0.27 and yards after contact at +0.28. They are worth seeing and worth
not forecasting. Volume (+0.67 carries, +0.77 targets) is the half that predicts itself.
"""

import streamlit as st

from dashboards import nfl_charts, nfl_pff

st.header("Ground game", anchor=False)
st.caption("Carries, receiving work, and the red-zone touches that carry the scoring.")

seasons_available = nfl_pff.available_seasons("rushing_summary")
if not seasons_available:
    st.info("No PFF rushing exports found. Drop them under `nfl/pff/` and reload.")
    st.stop()

with st.sidebar:
    st.subheader("Scope", anchor=False)
    seasons = st.multiselect("Seasons", seasons_available, default=seasons_available[:1],
                             key="nflrun_seasons", persist_state="session")
    if not seasons:
        st.warning("Pick at least one season.")
        st.stop()
    min_carries = st.slider("Minimum carries", 10, 200, 40, step=10, key="nflrun_carries", persist_state="session")

seasons = tuple(sorted(seasons, reverse=True))
frame = nfl_pff.rushers(seasons, min_carries=min_carries)
if frame.empty:
    st.warning("Nothing clears that carry threshold for those seasons.")
    st.stop()

frame = nfl_pff.scope_sidebar(frame, "nflrun", positions=("RB", "QB", "WR"))
if frame.empty:
    st.warning("Every player was filtered out.")
    st.stop()

with st.container(horizontal=True):
    st.metric("Backs", len(frame), border=True)
    st.metric("Median carries/G", f"{frame['Carries/G'].median():.1f}", border=True)
    st.metric("Median targets/G", f"{frame['Targets/G'].median():.1f}", border=True)

tabs = st.tabs(["Workload", "High-value touches", "Board"])

AXES = ["Carries/G", "Targets/G", "Routes/G", "Gap%", "Elusive", "Breakaway%",
        "Run grade", "YPC", "YCO/att"]

with tabs[0]:
    with st.container(horizontal=True):
        x = st.selectbox("X axis", AXES, index=AXES.index("Carries/G"), key="nflrun_x", persist_state="session")
        y = st.selectbox("Y axis", AXES, index=AXES.index("Targets/G"), key="nflrun_y", persist_state="session")
    chart = nfl_charts.usage_scatter(frame, x, y, size="Carries", color="Pos")
    if chart is None:
        st.warning(f"No back has both {x} and {y} on this scope.")
    else:
        st.altair_chart(chart, width="stretch")
        st.caption("Carries and targets are the two workloads a back can own, and they are "
                   "close to independent — the receiving one persists slightly better "
                   "(+0.77 against +0.67).")

with tabs[1]:
    hv = nfl_pff.high_value(seasons)
    if hv.empty:
        st.info("No red-zone splits on this scope — `fantasy-stats-receiving` covers "
                "2022-25.")
    else:
        columns = [c for c in ("Name", "Team", "Pos", "G", "HV touches/G", "i5 car/G",
                               "RZ car/G", "RZ tgt/G", "EZ tgt/G", "Carries/G",
                               "Targets/G") if c in hv.columns]
        st.markdown("**Inside-5 carries and red-zone targets, per game**")
        st.dataframe(nfl_charts.round_display(
            hv[columns].sort_values("HV touches/G", ascending=False).head(60)),
                     hide_index=True, width="stretch")
        # Only men with a claim on both axes -- see the High value page for why.
        both = hv[(hv["i5 car/G"] > 0) & (hv["RZ tgt/G"] > 0)]
        chart = nfl_charts.usage_scatter(
            both.rename(columns={"i5 car/G": "Inside5/G", "RZ tgt/G": "RZtgt/G"}),
            "Inside5/G", "RZtgt/G", size="G", color="Pos", labels=True)
        if chart is not None:
            st.altair_chart(chart, width="stretch")
            st.caption("The two ways a player gets scoring equity his yardage volume does "
                       "not already describe. Backs own the first axis, receivers the "
                       "second, and the few who own both are the ones worth finding.")

with tabs[2]:
    columns = [c for c in ("Name", "Team", "Pos", "G", "Carries/G", "Targets/G",
                           "Routes/G", "Gap%", "Elusive", "Breakaway%", "Run grade",
                           "YPC", "YCO/att") if c in frame.columns]
    sort_by = st.selectbox("Sort by", [c for c in columns if c not in
                                       ("Name", "Team", "Pos")], key="nflrun_sort", persist_state="session")
    st.dataframe(nfl_charts.round_display(
        frame[columns].sort_values(sort_by, ascending=False)),
                 hide_index=True, width="stretch",
                 column_config={"Gap%": st.column_config.ProgressColumn(
                     "Gap%", min_value=0.0, max_value=1.0, format="%.0f%%")})
    st.caption("`Gap%` is the share of his carries on gap concepts rather than zone "
               "(+0.54 year over year) — a scheme fact, and the one efficiency-adjacent "
               "column here worth carrying forward.")
