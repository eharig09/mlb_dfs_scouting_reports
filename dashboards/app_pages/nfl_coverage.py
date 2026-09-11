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

import pandas as pd
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

        # **Both axes are pinned across the switch.** Man and zone are the two halves of
        # one comparison, and letting each panel rescale to its own data moves every point
        # *and* the axis under it when you flip -- the reader is left comparing two pictures
        # that share nothing. The domains span both schemes so a receiver sitting still
        # means he is genuinely the same.
        y_pair = [f"Man {measure}", f"Zone {measure}"]
        values = pd.concat([matchups[c] for c in y_pair if c in matchups.columns]).dropna()
        y_domain = [float(values.min()), float(values.max())] if len(values) else None
        x_domain = [0.0, 100.0]
        chart = nfl_charts.usage_scatter(matchups, x, y, size="Man routes", color="Pos",
                                         labels=True, x_domain=x_domain, y_domain=y_domain)
        if chart is not None:
            st.altair_chart(chart, width="stretch")
            st.caption(f"Axes are fixed across the Man/Zone switch, so a point that does "
                       f"not move really did not change. Right is more "
                       f"{against.lower()} coverage faced; up is better "
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
    inner = st.segmented_control("Side", ["Matchup", "Receivers", "Defences"],
                                 default="Matchup", key="nflcov_slotside",
                                 persist_state="session") or "Matchup"

    if inner == "Defences":
        st.warning("Slot YPT allowed self-correlates at **+0.10** year over year and slot "
                   "TD rate at **+0.01**. Last season's vulnerable slot defence is a coin "
                   "flip; this is a record of the season, not a target list.",
                   icon=":material/warning:")
        chart = nfl_charts.tendency_bars(slot, "Slot YPT", "Slot yards per target allowed")
        if chart is not None:
            st.altair_chart(chart, width="stretch")
        st.dataframe(nfl_charts.round_display(
            slot.sort_values("Slot YPT", ascending=False)), hide_index=True,
            width="stretch")

    elif inner == "Receivers":
        st.markdown("**Who actually works from the slot, and what he does there**")
        st.caption("`Slot share` is of *routes*, not snaps — a receiver who moves inside on "
                   "third down is a slot receiver for the plays that matter, and a snap "
                   "count flattens that.")
        catchers = nfl_pff.receiver_slot(season)
        if catchers.empty:
            st.info("No slot splits for that season.")
        else:
            chart = nfl_charts.usage_scatter(catchers, "Slot routes/G", "Slot YPRR",
                                             size="Slot targets", color="Pos",
                                             extra_tooltip=("Slot aDOT", "Slot TPRR",
                                                            "Slot grade"))
            if chart is not None:
                st.altair_chart(chart, width="stretch")
                st.caption("Right is more slot volume, up is more production per route "
                           "there. Route grade and YPRR persist around 0.6 — the solid "
                           "half of this page.")
            columns = ["Name", "Team", "Pos", "G", "Slot routes/G", "Slot TPRR",
                       "Slot YPRR", "Slot aDOT", "Slot catch%", "Slot grade",
                       "Slot yards/G", "Slot TD", "Screen YPRR"]
            st.dataframe(nfl_charts.round_display(
                catchers[[c for c in columns if c in catchers.columns]]
                .sort_values("Slot YPRR", ascending=False)),
                hide_index=True, width="stretch")

    else:
        st.markdown("**Slot receivers against the slot defence they draw**")
        if slate_path is None:
            st.info("Pick a slate in the sidebar to set each receiver against his opponent.")
        else:
            paired = nfl_pff.slot_matchup(season, slate_path)
            if paired.empty or "Opp slot YPT" not in paired.columns:
                st.warning("Could not pair slot receivers with slot defences on this slate.")
            else:
                chart = nfl_charts.usage_scatter(
                    paired, "Opp slot YPT", "Slot YPRR", size="Slot targets", color="Pos",
                    extra_tooltip=("Opp", "Slot routes/G", "Slot aDOT"))
                if chart is not None:
                    st.altair_chart(chart, width="stretch")
                st.warning("**Read the vertical axis, not the horizontal one.** What a "
                           "receiver does from the slot is his own — route grade and YPRR "
                           "persist around 0.6. What a defence allowed there does not: slot "
                           "YPT self-correlates at +0.10, slot TD rate at +0.01. Top-right "
                           "is a good slot receiver drawing a defence that *was* leaky, "
                           "which is one real fact and one coin flip.",
                           icon=":material/warning:")
                columns = ["Name", "Team", "Opp", "Pos", "Slot routes/G", "Slot TPRR",
                           "Slot YPRR", "Slot aDOT", "Slot grade", "Opp slot YPT",
                           "Opp slot TD%"]
                st.dataframe(nfl_charts.round_display(
                    paired[[c for c in columns if c in paired.columns]]
                    .sort_values("Slot YPRR", ascending=False)),
                    hide_index=True, width="stretch")

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
