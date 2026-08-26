"""Quarterbacks, broken out by how far downfield they actually throw.

Built on PFF's `passing_depth` export — 554 columns that had gone untouched until now. Only
the depth half is surfaced: the directional split multiplies the columns by four and divides
the sample by the same, and a quarterback's left-versus-right numbers are mostly a fact about
who he was throwing to rather than about him.

**Depth mix is the thing to read.** A quarterback's average depth of target self-correlates
the way receiver aDOT does — it is a property of the offence, not of the week — while his
completion percentage and touchdown count at any given depth move around far more. So the
attempt *mix* is the stable half of this page and the efficiency inside each bucket is the
volatile half, exactly as everywhere else in this data.

The buckets are PFF's: behind the line of scrimmage (screens and check-downs), short,
medium, and deep at 20+ air yards.
"""

import altair as alt
import pandas as pd
import streamlit as st

from dashboards import nfl_charts, nfl_filters, nfl_pff, nfl_slates

st.header("Quarterbacks", anchor=False)
st.caption("Attempt mix, efficiency and grade by throw depth.")

seasons_available = nfl_pff.available_seasons("passing_depth")
if not seasons_available:
    st.info("No PFF passing-depth exports found. Drop them under `nfl/pff/` and reload.")
    st.stop()

with st.sidebar:
    st.subheader("Filters", anchor=False)
    season = st.selectbox("Season", seasons_available, key="nflqb_season", persist_state="session")
    min_attempts = st.slider("Minimum attempts", 25, 500, 150, step=25,
                             key="nflqb_att", persist_state="session",
                             help="Below this a deep-ball rate is describing a dozen throws.")
    path, _ = nfl_filters.slate_picker("nflqb", label="Slate")

frame = nfl_pff.quarterbacks(season, min_attempts=min_attempts)
if frame.empty:
    st.warning("No quarterback clears that attempt threshold.")
    st.stop()

if path is not None:
    frame = nfl_slates.attach_salary(frame, path)
    frame, _ = nfl_slates.attach_projection(frame)
    frame = nfl_slates.add_value(frame)
    only = st.sidebar.toggle("Only clubs on this slate", value=True, key="nflqb_only", persist_state="session")
    if only:
        frame = frame[frame["On slate"]]

with st.sidebar:
    teams = sorted(frame["Team"].dropna().unique())
    picked = st.multiselect("Team", teams, key="nflqb_team", persist_state="session", placeholder="Every team")
    if picked:
        frame = frame[frame["Team"].isin(picked)]

if frame.empty:
    st.warning("Every quarterback was filtered out.")
    st.stop()

with st.container(horizontal=True):
    st.metric("Quarterbacks", len(frame), border=True)
    st.metric("Median YPA", f"{frame['YPA'].median():.2f}", border=True)
    st.metric("Median aDOT", f"{frame['aDOT est'].median():.2f}", border=True)
    st.metric("Deepest mix", f"{frame['Deep att%'].max():.1f}%", border=True)

long = nfl_pff.quarterback_depth_long(frame)
DEPTH_ORDER = [nfl_pff.DEPTH_LABELS[b] for b in nfl_pff.DEPTH_BUCKETS]

tabs = st.tabs(["Depth mix", "Efficiency by depth", "One quarterback", "Board"])

with tabs[0]:
    st.markdown("**Where each offence looks** — share of attempts by depth")
    st.caption("The stable half of this page. A quarterback's depth mix is a property of the "
               "offence he plays in; what he does inside each bucket moves far more. "
               "**The four buckets cover 86–95% of attempts**, not all of them — the "
               "remainder are throwaways, spikes and batted balls with no meaningful air "
               "yards. `Charted%` on the board says how much of each man's work is here, "
               "and the bars are normalised so the mix is still comparable.")
    chart = (alt.Chart(long.dropna(subset=["Att%"])).mark_bar().encode(
        y=alt.Y("Name:N", title=None,
                sort=alt.EncodingSortField("Att%", op="sum", order="descending")),
        x=alt.X("Att%:Q", title="Share of attempts (%)", stack="normalize"),
        color=alt.Color("Depth:N", title="Depth", sort=DEPTH_ORDER,
                        scale=alt.Scale(scheme="viridis")),
        order=alt.Order("Depth:N", sort="ascending"),
        tooltip=[alt.Tooltip("Name:N"), alt.Tooltip("Team:N"), alt.Tooltip("Depth:N"),
                 alt.Tooltip("Att%:Q", format=".1f"),
                 alt.Tooltip("Attempts:Q", format=".0f"),
                 alt.Tooltip("YPA:Q", format=".2f")])
        .properties(height=nfl_charts.bar_height(frame["Name"].nunique())))
    st.altair_chart(chart, width="stretch")

    st.markdown("**Depth against efficiency**")
    scatter = nfl_charts.usage_scatter(frame, "aDOT est", "YPA", size="Attempts",
                                       color="Team", labels=True)
    if scatter is not None:
        st.altair_chart(scatter, width="stretch")
        st.caption("Right is a deeper passing game; up is more yards per attempt. Deep "
                   "throws pay more per completion and complete less often, so the two are "
                   "not the same axis — a quarterback high and left is efficient without "
                   "the downfield volume, which is a different asset in a stack.")

with tabs[1]:
    measure = st.segmented_control("Measure", ["YPA", "Grade", "Comp%", "TD"],
                                   default="YPA", key="nflqb_measure", persist_state="session") or "YPA"
    depth = st.pills("Depth", DEPTH_ORDER, selection_mode="multi", default=DEPTH_ORDER,
                     key="nflqb_depth", persist_state="session")
    part = long[long["Depth"].isin(depth)] if depth else long
    part = part.dropna(subset=[measure])
    if part.empty:
        st.warning(f"No {measure} available on that scope.")
    else:
        chart = (alt.Chart(part).mark_bar(cornerRadiusEnd=2).encode(
            y=alt.Y("Name:N", title=None,
                    sort=alt.EncodingSortField(measure, op="mean", order="descending")),
            x=alt.X(f"{measure}:Q", title=measure),
            color=alt.Color("Depth:N", title="Depth", sort=DEPTH_ORDER,
                            scale=alt.Scale(scheme="viridis")),
            xOffset=alt.XOffset("Depth:N", sort=DEPTH_ORDER),
            tooltip=[alt.Tooltip("Name:N"), alt.Tooltip("Depth:N"),
                     alt.Tooltip(f"{measure}:Q", format=".2f"),
                     alt.Tooltip("Attempts:Q", format=".0f")])
            .properties(height=nfl_charts.bar_height(part["Name"].nunique() * 1.6)))
        st.altair_chart(chart, width="stretch")
        st.caption("Efficiency inside a bucket is the volatile half — read it as what "
                   "happened, and let the mix on the previous tab carry the expectation.")

with tabs[2]:
    who = st.selectbox("Quarterback", sorted(frame["Name"].unique()), key="nflqb_one", persist_state="session")
    one = frame[frame["Name"] == who].iloc[0]
    with st.container(horizontal=True):
        st.metric("Attempts", f"{one['Attempts']:.0f}", border=True)
        st.metric("Att/G", f"{one['Att/G']:.2f}", border=True)
        st.metric("YPA", f"{one['YPA']:.2f}", border=True)
        st.metric("aDOT", f"{one['aDOT est']:.2f}", border=True)
        st.metric("Charted", f"{one['Charted%']:.1f}%", border=True)
        if "Salary" in frame.columns and pd.notna(one.get("Salary")):
            st.metric("Salary", f"${one['Salary']:,.0f}", border=True)

    rows = []
    for label in DEPTH_ORDER:
        rows.append({
            "Depth": label,
            "Att": one.get(f"{label} att"), "Att%": one.get(f"{label} att%"),
            "Comp%": one.get(f"{label} comp%"), "YPA": one.get(f"{label} YPA"),
            "TD": one.get(f"{label} TD"), "Grade": one.get(f"{label} grade"),
            "BTT%": one.get(f"{label} BTT%"), "TWP%": one.get(f"{label} TWP%"),
        })
    st.dataframe(nfl_charts.round_display(pd.DataFrame(rows)), hide_index=True,
                 width="stretch")
    st.caption("`BTT%` is big-time throws — the deep, tight-window ones. `TWP%` is "
               "turnover-worthy plays. Both are PFF charting rather than outcomes, so they "
               "describe the throws rather than what happened to them.")

with tabs[3]:
    columns = ["Name", "Team", "G", "Attempts", "Att/G", "YPA", "aDOT est", "Charted%"]
    for label in DEPTH_ORDER:
        columns += [f"{label} att%", f"{label} YPA"]
    if "Salary" in frame.columns:
        columns += ["Salary", "Proj", "Value"]
    columns = [c for c in columns if c in frame.columns]
    sort_by = st.selectbox("Sort by", [c for c in columns if c not in ("Name", "Team")],
                           key="nflqb_sort", persist_state="session")
    st.dataframe(
        nfl_charts.round_display(frame[columns].sort_values(sort_by, ascending=False)),
        hide_index=True, width="stretch",
        column_config={"Salary": st.column_config.NumberColumn("Salary", format="$%d")})
