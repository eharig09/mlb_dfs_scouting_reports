"""Does any of this actually predict anything?

Every other page shows what the report believed before first pitch. This one scores those
beliefs against `dk_results/` — real DK points and real field ownership — and is deliberately
built to make a *weak* answer visible rather than to flatter the model.
"""

import pandas as pd
import streamlit as st

from dashboards import charts, drill_ui, outcomes

#: Candidate axes: the columns worth asking "do usable picks cluster along this?" about.
AXES = {
    "Salary": "DraftKings salary",
    "Proj": "Model projection (DK pts)",
    "Ceiling": "Projected ceiling (DK pts)",
    "Composite": "Composite matchup score",
    "Slot": "Batting order slot",
    "PA": "Expected plate appearances",
    "surplus": "Surplus vs price band",
    "per_1k": "Composite per $1k",
    "Season OPS": "Season OPS",
    "Platoon OPS": "OPS vs this hand",
    "Arsenal OPS": "OPS vs this arsenal",
    "Off L28": "Offense, last 28 days",
    "own": "Actual field ownership %",
}

st.header("Signal", anchor=False)
st.caption("Scored against real contest results. The grid is history, not a projection — "
           "it says what share of past hitters in a region of the board cleared the points "
           "line, and every cell can be opened to see exactly who they were.")

history = outcomes.scored_history()
if history.empty:
    st.info("No contest exports in `dk_results/` could be matched to a cached slate.")
    st.stop()

available = {k: v for k, v in AXES.items() if k in history.columns
             and history[k].notna().sum() > 100}

with st.sidebar:
    st.subheader("Outcome", anchor=False)
    threshold = st.slider("A 'hit' is at least this many DK points", 5.0, 25.0,
                          outcomes.DEFAULT_HIT, step=1.0, key="signal_hit",
                          help="10 is roughly what a mid-priced bat has to clear to have "
                               "been worth its slot.",
                          persist_state="session")
    st.subheader("Axes", anchor=False)
    keys = list(available)
    x = st.selectbox("Horizontal", keys, index=keys.index("Salary"),
                     format_func=lambda k: available[k], key="signal_x",
                     persist_state="session")
    y = st.selectbox("Vertical", keys,
                     index=keys.index("Composite") if "Composite" in keys else 1,
                     format_func=lambda k: available[k], key="signal_y",
                     persist_state="session")
    bins = st.slider("Bins per axis", 3, 8, 6, key="signal_bins",
    persist_state="session")
    min_cell = st.slider("Minimum players per cell", 5, 40, outcomes.MIN_CELL, step=1,
                         key="signal_mincell",
                         help="Cells under this are left blank rather than coloured. A "
                              "thin cell shaded at all reads as knowledge we do not have.",
                         persist_state="session")

    st.subheader("Hold fixed", anchor=False)
    control = st.selectbox("Primary control", list(outcomes.CONTROLS),
                           key="signal_control",
                           format_func=lambda k: outcomes.CONTROLS[k],
                           persist_state="session")
    second_options = ["(none)"] + [c for c in outcomes.CONTROLS if c != control]
    second_choice = st.selectbox(
        "Second control", second_options, key="signal_second",
        index=second_options.index("Slot") if "Slot" in second_options else 0,
        help="Salary alone is not enough. Cheap hitters bat at the bottom of the order, "
             "so a 'cheap hitters beat their price' result is largely 'the score noticed "
             "who was batting eighth'.",
        persist_state="session")
    second = None if second_choice == "(none)" else second_choice

    st.subheader("Filter", anchor=False)
    slates = sorted(history["slate"].dropna().unique())
    chosen_slates = st.multiselect("Slate", slates, default=[], key="signal_slates",
                                   help="Leave empty for every slate.",
                                   persist_state="session")
    teams = sorted(history["Team"].dropna().unique())
    chosen_teams = st.multiselect("Team", teams, default=[], key="signal_teams",
    persist_state="session")
    slots = sorted(s for s in history.get("DK Pos", pd.Series(dtype=str))
                   .dropna().unique() if s)
    chosen_slots = st.multiselect("Roster slot", slots, default=[], key="signal_slots",
    persist_state="session")

scored = history.copy()
scored["hit"] = scored["fpts"] >= threshold
if chosen_slates:
    scored = scored[scored["slate"].isin(chosen_slates)]
if chosen_teams:
    scored = scored[scored["Team"].isin(chosen_teams)]
if chosen_slots and "DK Pos" in scored.columns:
    scored = scored[scored["DK Pos"].isin(chosen_slots)]
if scored.empty:
    st.warning("No scored slates match those filters.")
    st.stop()

base = scored["hit"].mean() * 100
with st.container(horizontal=True):
    st.metric("Scored hitter-slates", f"{len(scored):,}", border=True)
    st.metric("Dates", scored["date"].nunique(), border=True)
    st.metric("Base hit rate", f"{base:.1f}%",
              delta=f"at {threshold:.0f}+ DK points", delta_color="off", border=True)
    st.metric("Mean points", f"{scored['fpts'].mean():.1f}", border=True)

grid = outcomes.hit_grid(scored, x, y, bins, bins, min_cell)

with st.container(border=True):
    st.markdown(f"**Where usable picks came from** — {available[x]} against {available[y]}, "
                f"shaded by the share of past hitters who cleared {threshold:.0f} points. "
                "Click a cell to see them.")
    if grid.empty:
        st.info("No cell holds enough players to be worth colouring. Widen the filters, "
                "lower the minimum, or use fewer bins.")
    else:
        chart = charts.hit_grid_chart(grid, available[x], available[y])
        event = st.altair_chart(chart, on_select="rerun", key="signal_grid")
        picked = (event.get("selection") or {}).get("cell") or []
        st.caption(f"{len(grid)} of {bins * bins} cells hold at least {min_cell} players. "
                   "Blank means too few to say, not zero.")

        if picked:
            cell = picked[0]
            members = outcomes.cell_members(scored, x, y, cell)
            if not members.empty:
                rate = members["hit"].mean() * 100
                st.markdown(
                    f"**{len(members)} hitters** with {available[x]} "
                    f"{cell['x0']:,.1f}–{cell['x1']:,.1f} and {available[y]} "
                    f"{cell['y0']:,.1f}–{cell['y1']:,.1f} — **{rate:.0f}%** hit, "
                    f"{members['fpts'].mean():.1f} points on average "
                    f"(slate base {base:.1f}%)")
                st.dataframe(
                    outcomes.drill_frame(members), hide_index=True, height=320,
                    column_config={
                        "Salary": st.column_config.NumberColumn(format="$%d"),
                        "Composite": st.column_config.NumberColumn(format="%.1f"),
                        "surplus": st.column_config.NumberColumn("Surplus", format="%+.1f"),
                        "own": st.column_config.NumberColumn("Owned", format="%.1f%%"),
                        "fpts": st.column_config.NumberColumn("DK pts", format="%.1f"),
                        "hit": st.column_config.CheckboxColumn("Hit"),
                        "Season OPS": st.column_config.NumberColumn(format="%.3f"),
                        "Platoon OPS": st.column_config.NumberColumn(format="%.3f"),
                        "Arsenal OPS": st.column_config.NumberColumn(format="%.3f"),
                    })
        else:
            st.caption("Click any cell to list the hitters behind it.")

left, right = st.columns(2)
with left.container(border=True):
    st.markdown(f"**Sorting by {available[y]} alone** — the dashed line is the base rate, "
                "so a bar sitting on it means the column told you nothing")
    table = outcomes.lift_table(scored, y)
    chart = charts.lift_bars(table)
    if chart is None:
        st.info("Not enough scored slates to band this column.")
    else:
        st.altair_chart(chart)
        best = table.loc[table["rate_pct"].idxmax()]
        st.caption(f"Top band hits {best['rate_pct']:.0f}% against a {base:.0f}% base — "
                   f"lift {best['lift']:.2f}. Mean salary in that band is "
                   f"${best['salary']:,.0f}, which is most of the story.")

with right.container(border=True):
    held = outcomes.CONTROLS[control].split(" — ")[0]
    if second:
        held += " and " + outcomes.CONTROLS[second].split(" — ")[0]
    st.markdown(f"**{available[y]} holding {held} fixed** — the honest version. Bars are "
                "the spread between the best and worst third of hitters alike on those "
                "controls; the line is two standard errors.")
    controlled = outcomes.controlled_lift(scored, y, control=control, second=second)
    chart = charts.controlled_spread_bars(controlled)
    if chart is None:
        st.info("Not enough scored slates to hold those controls fixed. Two controls cost "
                "cells — try dropping the second one.")
    else:
        st.altair_chart(chart)
        clears = int(controlled["beyond noise"].sum())
        st.caption(
            f"{clears} of {len(controlled)} cells separate beyond noise. "
            "A grey bar is not an edge — it is a spread this sample cannot distinguish "
            "from zero.")
        if second is None:
            st.warning(
                "Salary alone is not a sufficient control. Batting slot moves the outcome "
                "more than the composite does (slots 1-2 average 7.9 DK points and 33% "
                "hits; slots 8-9 average 4.9 and 17%), and cheap hitters bat at the "
                "bottom. Add a second control.", icon=":material/warning:")

with st.container(border=True):
    st.markdown("**Every scored slate** — the points behind all of the above")
    scatter = charts.outcome_scatter(scored, x, y, available[x], available[y])
    point_event = drill_ui.chart_with_drilldown(scatter, scored, None,
                                                "signal_point_pick")
    st.caption("Each point is one past hitter-slate. Click one to open that night's "
               "evidence for him.")

with st.expander("What predicts DK points, measured"):
    numeric = [c for c in available if c in scored.columns]
    ranks = pd.DataFrame({
        "column": [available[c] for c in numeric],
        "pearson": [scored[c].corr(scored["fpts"]) for c in numeric],
        "spearman": [scored[c].corr(scored["fpts"], method="spearman") for c in numeric],
        "n": [int(scored[c].notna().sum()) for c in numeric],
    }).sort_values("pearson", ascending=False)
    st.dataframe(ranks, hide_index=True, column_config={
        "column": st.column_config.TextColumn("Column"),
        "pearson": st.column_config.NumberColumn(format="%+.3f"),
        "spearman": st.column_config.NumberColumn(format="%+.3f"),
        "n": st.column_config.NumberColumn("Rows", format="%d")})
    st.caption(
        "Single-game DK scoring is close to a coin flip dressed up, so these are all small "
        "and that is expected rather than a bug. Ownership is included because the field's "
        "own read is a benchmark worth beating. What matters is not the raw number but "
        "whether anything survives holding salary fixed — the panel above.")

# A point here belongs to a *past* night, so the dialog has to be opened against that row's
# own date rather than a page-level one — the same player appears on many slates.
picked_point = charts.selected(locals().get("point_event"))
if picked_point:
    hit = scored[(scored["Name"].astype(str) == str(picked_point[0].get("Name")))
                 & (scored["Team"].astype(str) == str(picked_point[0].get("Team")))]
    if not hit.empty:
        row = hit.iloc[0]
        drill_ui.open_selection(locals().get("point_event"), scored, row["date"],
                                history=history)
