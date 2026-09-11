"""What changed between two immutable slate snapshots."""

import os

import altair as alt
import pandas as pd
import streamlit as st

from dashboards import changes, scales
from dfs.snapshot import STAGES, list_snapshots


st.header("Changes", anchor=False)
st.caption("Compare exactly what the optimizer saw at two saved stages. This reads immutable "
           "snapshots—not today's rewritten report cache—so a lineup confirmation or model "
           "move cannot leak backward into the earlier board.")

snapshots = list_snapshots()
if not snapshots:
    st.info("No DFS snapshots found. Create them with `python -m dfs.snapshot --date "
            "YYYY-MM-DD --stage morning`, then repeat at later stages.")
    st.stop()

dates = sorted({str(s.date) for s in snapshots}, reverse=True)
with st.sidebar:
    st.subheader("Comparison", anchor=False)
    date = st.selectbox("Date", dates, key="changes_date", persist_state="session")
    slates = sorted({str(s.slate) for s in snapshots if str(s.date) == date})
    slate = st.selectbox("Slate", slates, key="changes_slate", persist_state="session")

available = [s for s in snapshots if str(s.date) == date and str(s.slate) == slate]
rank = {stage: i for i, stage in enumerate(STAGES)}
available.sort(key=lambda s: (rank.get(s.stage, -1), s.manifest.get("revision", 1),
                              s.manifest.get("taken_utc") or ""))


def _label(snapshot):
    revision = int(snapshot.manifest.get("revision", 1))
    suffix = "" if revision == 1 else f" r{revision}"
    taken = str(snapshot.manifest.get("taken_utc") or "").replace("T", " ")[:16]
    return f"{snapshot.stage}{suffix} · {taken} UTC"


by_path = {os.path.normpath(s.directory): s for s in available}
paths = list(by_path)
with st.sidebar:
    before_path = st.selectbox("Before", paths, index=0, format_func=lambda p: _label(by_path[p]),
                               key="changes_before", persist_state="session")
    after_path = st.selectbox("After", paths, index=len(paths) - 1,
                              format_func=lambda p: _label(by_path[p]),
                              key="changes_after", persist_state="session")

if before_path == after_path:
    st.warning("Choose two different snapshots.")
    st.stop()

before, after = by_path[before_path], by_path[after_path]
diff = changes.compare(before.players, after.players)
if diff.empty:
    st.info("Neither snapshot contains a player board.")
    st.stop()

delta_columns = [c for c in diff if c.startswith("Δ ")]
measures = [c[2:] for c in delta_columns]
if not measures:
    st.info("These snapshots contain no comparable numeric player measures.")
    st.stop()
default_measure = "Proj" if "Proj" in measures else measures[0]
with st.sidebar:
    player_type = st.radio("Players", ["Hitters", "Pitchers", "All"], horizontal=True,
                           key="changes_type", persist_state="session")
    measure = st.selectbox("Measure", measures, index=measures.index(default_measure),
                           key="changes_measure", persist_state="session")
    minimum = st.number_input("Minimum absolute move", min_value=0.0, value=0.1, step=0.1,
                              key="changes_minimum", persist_state="session")
    include_unchanged = st.toggle("Include unchanged", value=False,
                                  key="changes_unchanged", persist_state="session")

view = diff
if player_type != "All" and "Type" in view:
    wanted = "H" if player_type == "Hitters" else "P"
    view = view[view["Type"].astype(str).eq(wanted)]
delta = f"Δ {measure}"
if not include_unchanged:
    view = view[(view["Status"] != "Unchanged")
                & (view[delta].abs().fillna(float("inf")) >= float(minimum))]

became_confirmed = changes.confirmed_between(diff)
with st.container(horizontal=True):
    st.metric("Changed", int((diff["Status"] == "Changed").sum()), border=True)
    st.metric("Became confirmed", became_confirmed, border=True)
    st.metric("Added", int((diff["Status"] == "Added").sum()), border=True)
    st.metric("Removed", int((diff["Status"] == "Removed").sum()), border=True)

st.caption(f"**{_label(before)}** → **{_label(after)}** · "
           f"{len(before.players)} → {len(after.players)} player rows")

chart_data = view.dropna(subset=[delta]).copy()
if not chart_data.empty:
    chart_data = chart_data.reindex(chart_data[delta].abs().sort_values(ascending=False).index) \
                           .head(24)
    chart_data["Player"] = (chart_data["Name"].astype(str) + " · "
                            + chart_data["Team"].astype(str))
    domain = None
    focus_half = {"Proj": 5.0, "Ceiling": 8.0, "Floor": 5.0, "Own%": 15.0,
                  "Salary": 1500.0, "PA": 1.5, "Team Runs": 1.5}.get(measure)
    if scales.selected_mode(st.session_state) == "focus" and focus_half:
        domain = [-focus_half, focus_half]
    else:
        widest = float(chart_data[delta].abs().max()) or 1.0
        domain = [-widest, widest]
    bars = (alt.Chart(chart_data).mark_bar(cornerRadiusEnd=3)
            .encode(
                x=alt.X(f"{delta}:Q", title=f"Change in {measure}",
                        scale=alt.Scale(domain=domain, clamp=True)),
                y=alt.Y("Player:N", title=None, sort=alt.EncodingSortField(
                    field=delta, op="sum", order="descending")),
                color=alt.condition(f"datum['{delta}'] >= 0", alt.value("#2a9d8f"),
                                    alt.value("#d65a5a")),
                tooltip=[alt.Tooltip("Name:N"), alt.Tooltip("Team:N"),
                         alt.Tooltip(f"{measure} Before:Q", format=".2f"),
                         alt.Tooltip(f"{measure} After:Q", format=".2f"),
                         alt.Tooltip(f"{delta}:Q", format="+.2f"),
                         alt.Tooltip("Details:N")]))
    zero = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(opacity=0.6).encode(x="x:Q")
    st.altair_chart((zero + bars).properties(height=max(280, 24 * len(chart_data))),
                    width="stretch")
    clipped = int((chart_data[delta].abs() > domain[1]).sum())
    if clipped:
        st.caption(f"{clipped} move{'s' if clipped != 1 else ''} reach the Focus boundary; "
                   "the tooltip is exact, or switch Axis range to Full.")

st.subheader("Audit table", anchor=False)
if view.empty:
    st.info("No rows clear those filters.")
else:
    leading = ["Name", "Team", "Type", "Status", f"{measure} Before",
               f"{measure} After", delta, "Details"]
    st.dataframe(view.reindex(columns=[c for c in leading if c in view.columns])
                 .sort_values(delta, key=lambda values: values.abs(), ascending=False),
                 hide_index=True, width="stretch",
                 column_config={delta: st.column_config.NumberColumn(delta, format="%+.2f")})
