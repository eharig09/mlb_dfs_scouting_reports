"""Man, zone and the slot — with an explicit warning about half of it.

**Read the tendency, not the results.** Measured year over year across 2020-25:

    a defense's man rate          +0.46   <- a real, usable fact about how it plays
    man YPT allowed               +0.05
    zone YPT allowed              +0.01
    slot YPT allowed              +0.10
    slot TD rate allowed          +0.01
    man-minus-zone gap allowed    -0.08

So "which defense is weak in the slot" is, on this data, a coin flip dressed as a finding.
The allowed columns are shown anyway — they describe what happened, which is worth seeing —
but they are labelled, and nothing on this page ranks a matchup by them.

The receiver half is better but small. A man-versus-zone *efficiency* gap barely persists
(+0.19); the *target-rate* gap does (+0.37 to +0.51), so volume rather than efficiency is
the only honest channel. Sized end to end it is worth about **0.09 targets a game** at the
extremes of both distributions — a tie-break, not a projection input.
"""

import streamlit as st

from dashboards import nfl_charts, nfl_filters, nfl_pff

st.header("Coverage", anchor=False)
st.caption("Scheme tendency is stable and worth using. What a defence *allowed* is not — "
           "see the note on each panel.")

seasons_available = nfl_pff.available_seasons("defense_coverage_scheme")
if not seasons_available:
    st.info("No PFF coverage exports found. Drop them under `nfl/pff/` and reload.")
    st.stop()

with st.sidebar:
    st.subheader("Scope", anchor=False)
    season = st.selectbox("Season", seasons_available, key="nflcov_season", persist_state="session")

scheme = nfl_pff.defense_scheme(season)
tabs = st.tabs(["Scheme matchups", "Scheme tendency", "What they allowed", "Slot",
                "Receiver splits"])

with tabs[0]:
    st.markdown("**A receiver's scheme numbers against the scheme he will actually see**")
    st.caption("His man numbers matter in proportion to how much man he draws, and how much "
               "man he draws is a property of the defence opposite him. The two have to be "
               "in one row before either says anything about Sunday.")

    with st.sidebar:
        slate_path, slate_row = nfl_filters.slate_picker("nflcov", label="Slate")

    matchups = nfl_pff.scheme_matchup(season, slate_path)
    if matchups.empty:
        st.warning("No receiver scheme splits for that season.")
    elif slate_path is None:
        st.info("Pick a slate in the sidebar to set each receiver against the defence he "
                "draws. Without one there is no opponent to weight by.")
    else:
        with st.container(horizontal=True):
            measure = st.segmented_control("Measure", ["YPRR", "TPRR"], default="TPRR",
                                           key="nflcov_measure", persist_state="session") or "TPRR"
            against = st.segmented_control("Versus", ["Man", "Zone"], default="Man",
                                           key="nflcov_against", persist_state="session") or "Man"
        y = f"{against} {measure}"
        x = "Opp man%" if against == "Man" else "Opp zone%"

        chart = nfl_charts.usage_scatter(matchups, x, y, size="Man routes", color="Pos",
                                         labels=True)
        if chart is not None:
            st.altair_chart(chart, width="stretch")
            st.caption(f"Right is more {against.lower()} coverage faced; up is better "
                       f"{measure} against it. The top-right corner is a receiver who is "
                       f"good against the scheme he is about to see a lot of — which is the "
                       f"whole shape of the question. **Size it honestly**: the TPRR gap "
                       f"behind this persists at +0.37 to +0.51, the YPRR gap only at "
                       f"+0.19, and the end-to-end effect is about 0.09 targets a game.")

        columns = ["Name", "Team", "Pos", "Opp", "Opp man%", "Man TPRR", "Zone TPRR",
                   "TPRR gap", "Man YPRR", "Zone YPRR", "YPRR gap",
                   "Scheme-weighted TPRR", "Scheme-weighted YPRR"]
        columns = [c for c in columns if c in matchups.columns]
        st.dataframe(
            nfl_charts.round_display(matchups[columns].sort_values(y, ascending=False)),
            hide_index=True, width="stretch")

with tabs[1]:
    st.markdown("**How much man each defence plays** — the half that predicts itself")
    chart = nfl_charts.tendency_bars(scheme, "Man%", "Man coverage (% of coverage snaps)")
    if chart is not None:
        st.altair_chart(chart, width="stretch")
    st.caption("Year-over-year **+0.46**, and the league spread runs about 17% to 40% — a "
               "2.3× range. Dashed rule is the league mean for this season.")
    st.dataframe(scheme[["Team", "Man%", "Zone%"]].sort_values("Man%", ascending=False),
                 hide_index=True, width="stretch",
                 column_config={"Man%": st.column_config.ProgressColumn(
                     "Man%", min_value=0, max_value=100, format="%.1f%%")})

with tabs[2]:
    st.warning("These columns do **not** predict next season. Man YPT allowed "
               "self-correlates at +0.05, zone at +0.01, and the man-minus-zone gap at "
               "−0.08. Read them as description of what happened, never as a matchup edge.",
               icon=":material/warning:")
    st.dataframe(
        scheme.sort_values("Scheme gap")[["Team", "Man%", "Man YPT", "Zone YPT",
                                          "Scheme gap"]],
        hide_index=True, width="stretch",
        column_config={
            "Man YPT": st.column_config.NumberColumn("Man YPT", format="%.2f"),
            "Zone YPT": st.column_config.NumberColumn("Zone YPT", format="%.2f"),
            "Scheme gap": st.column_config.NumberColumn("Man − Zone", format="%.2f"),
        })

with tabs[3]:
    slot = nfl_pff.defense_slot(season)
    st.warning("Slot YPT allowed self-correlates at **+0.10** and slot TD rate at "
               "**+0.01**. Last season's vulnerable slot defence is a coin flip; this is "
               "here as a record of the season, not as a target list.",
               icon=":material/warning:")
    chart = nfl_charts.tendency_bars(slot, "Slot YPT", "Slot yards per target allowed")
    if chart is not None:
        st.altair_chart(chart, width="stretch")
    st.dataframe(nfl_charts.round_display(
        slot.sort_values("Slot YPT", ascending=False)), hide_index=True,
                 width="stretch")

with tabs[4]:
    scheme_receivers = nfl_pff.receiver_scheme(season)
    if scheme_receivers.empty:
        st.info("No receiver scheme splits for that season.")
    else:
        st.markdown("**Man versus zone, per receiver**")
        st.caption("The **TPRR gap** is the column with signal (+0.37 to +0.51 year over "
                   "year). The YPRR gap is mostly last season's noise (+0.19), so a "
                   "'man-beater' read off it will not repeat.")
        chart = nfl_charts.usage_scatter(
            scheme_receivers, "Man TPRR", "Zone TPRR", size="Man routes", color="Pos")
        if chart is not None:
            st.altair_chart(chart, width="stretch")
            st.caption("Off the diagonal is a receiver the offence looks at differently by "
                       "scheme. Worth about 0.09 targets a game at the extremes — a "
                       "tie-break between close calls, nothing more.")
        st.dataframe(
            scheme_receivers.sort_values("TPRR gap", ascending=False),
            hide_index=True, width="stretch",
            column_config={
                "Man TPRR": st.column_config.NumberColumn("Man TPRR", format="%.3f"),
                "Zone TPRR": st.column_config.NumberColumn("Zone TPRR", format="%.3f"),
                "TPRR gap": st.column_config.NumberColumn("TPRR gap", format="%.3f"),
                "YPRR gap": st.column_config.NumberColumn("YPRR gap", format="%.2f"),
            })
