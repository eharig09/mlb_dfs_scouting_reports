"""The lineups you actually entered — exposure, cost, and how alike they are.

Read from a filled DK entries export, so this is what was **submitted**, not what an
optimizer proposed. The two diverge constantly and only one of them is at risk.

**Duplication is the first thing on the page.** Entering the same nine players seven times
is one lineup bought seven times, and it does not read that way in a contest list — it reads
as seven entries. The uniqueness panel says how many distinct rosters are really in there.
"""

import altair as alt
import pandas as pd
import streamlit as st

from dashboards import nfl_charts, nfl_filters, nfl_slates

st.header("Entries", anchor=False)
st.caption("What was actually entered, from a filled DK entries export.")

files = nfl_slates.list_entry_files()
if not files:
    st.info(f"No DK entry exports under `{nfl_slates.ENTRIES_DIR}`. Download the entries "
            "file from a contest and drop it there.")
    st.stop()

with st.sidebar:
    st.subheader("Filters", anchor=False)
    import os
    chosen = st.selectbox("Entries file", files, key="nflent_file", persist_state="session",
                          format_func=os.path.basename)
    path, slate_row = nfl_filters.slate_picker("nflent", label="Price against")

entries = nfl_slates.entered_lineups(chosen)
if entries.empty:
    st.warning("That file has no filled entries — every contest row is empty.")
    st.stop()

players = None
if path is not None:
    players, _ = nfl_slates.slate_players(path)
    board, _ = nfl_slates.attach_projection(players)
    players = nfl_slates.add_value(board)

exposure = nfl_slates.entry_exposure(entries, players)

rosters = (entries.sort_values("Slot").groupby("Entry")["Name"]
           .apply(lambda names: tuple(sorted(names))))
unique = rosters.nunique()
total = entries["Entry"].nunique()

with st.container(horizontal=True):
    st.metric("Entries", total, border=True)
    st.metric("Distinct rosters", unique, border=True)
    st.metric("Contests", entries["Contest"].nunique(), border=True)
    if players is not None:
        priced = entries.merge(players[["key", "Salary"]].drop_duplicates("key"),
                               on="key", how="left")
        spend = priced.groupby("Entry")["Salary"].sum()
        st.metric("Median spend", f"${spend.median():,.0f}", border=True)

if unique < total:
    st.warning(f"**{total} entries, {unique} distinct roster(s).** The same lineup entered "
               f"more than once is one lineup bought more than once — it multiplies the "
               f"stake, not the number of outcomes you are exposed to.",
               icon=":material/content_copy:")

tabs = st.tabs(["Exposure", "Lineups", "Built vs entered", "Stacks entered",
                "Contests"])

with tabs[0]:
    columns = [c for c in ("Name", "Slots", "Team", "Opp", "DK Pos", "Salary", "Band",
                           "Entries", "Exposure%") if c in exposure.columns]
    st.dataframe(nfl_charts.round_display(exposure[columns]), hide_index=True,
                 width="stretch",
                 column_config={
                     "Exposure%": st.column_config.ProgressColumn(
                         "Exposure%", min_value=0, max_value=100, format="%.0f%%"),
                     "Salary": st.column_config.NumberColumn("Salary", format="$%d"),
                 })
    chart = (alt.Chart(exposure.head(25)).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("Name:N", sort="-x", title=None),
        x=alt.X("Exposure%:Q", title="Share of entries (%)"),
        color=alt.Color("Exposure%:Q", legend=None,
                        scale=alt.Scale(scheme="blueorange")),
        tooltip=[alt.Tooltip("Name:N"), alt.Tooltip("Entries:Q"),
                 alt.Tooltip("Exposure%:Q", format=".0f")])
        .properties(height=nfl_charts.bar_height(len(exposure.head(25)))))
    st.altair_chart(chart, width="stretch")

with tabs[1]:
    entry = st.selectbox("Entry", sorted(entries["Entry"].unique()), key="nflent_entry", persist_state="session")
    one = entries[entries["Entry"] == entry]
    order = {slot: i for i, slot in enumerate(
        ["QB", "RB", "WR", "TE", "FLEX", "DST"])}
    one = one.assign(_o=one["Slot"].map(order)).sort_values("_o").drop(columns="_o")
    if players is not None:
        one = one.merge(
            players[["key", "Team", "Opp", "Salary", "Proj", "Value"]].drop_duplicates("key"),
            on="key", how="left")
    columns = [c for c in ("Slot", "Name", "Team", "Opp", "Salary", "Proj", "Value")
               if c in one.columns]
    st.dataframe(nfl_charts.round_display(one[columns]), hide_index=True,
                 width="stretch",
                 column_config={
                     "Salary": st.column_config.NumberColumn("Salary", format="$%d"),
                     "Proj": st.column_config.NumberColumn("Proj", format="%.1f"),
                 })
    if "Salary" in one.columns:
        st.caption(f"Contest: {one['Contest'].iloc[0] if 'Contest' in one else ''} · "
                   f"spend ${one['Salary'].sum():,.0f} of $50,000")

with tabs[2]:
    st.markdown("**What the optimizer built against what you actually entered**")
    st.caption("The two diverge constantly and only one of them is at risk. This reads "
               "`nfl_boards/<date>/lineups_<slate>.csv` — the generated set — and compares "
               "it with the entries file above.")

    from dashboards import nfl_boards

    dates = nfl_boards.list_dates()
    if not dates:
        st.info("Nothing in `nfl_boards/` yet — run `nfl.optimize` to generate a set.")
    else:
        with st.container(horizontal=True):
            built_date = st.selectbox("Date", dates, key="nflent_bdate",
                                      persist_state="session")
            built_slates = nfl_boards.list_slates(built_date)
            built_slate = st.selectbox("Slate", built_slates, key="nflent_bslate",
                                       persist_state="session") if built_slates else None
        built = nfl_boards.read("lineups", built_date, built_slate) if built_slate else None
        if built is None or built.empty:
            st.warning("No generated lineups for that date and slate.")
        else:
            summary = nfl_boards.lineup_summary(built)
            with st.container(horizontal=True):
                st.metric("Built", len(summary), border=True)
                st.metric("Entered", total, border=True)
                st.metric("Distinct entered", unique, border=True)

            built_players = set(built["Name"].dropna())
            entered_players = set(entries["Name"].dropna())
            both = built_players & entered_players
            st.markdown(f"**{len(both)} players appear in both**, "
                        f"{len(entered_players - built_players)} were entered but never "
                        f"built, {len(built_players - entered_players)} were built but "
                        f"never entered.")

            only_entered = sorted(entered_players - built_players)
            if only_entered:
                st.info("Entered without the optimizer ever proposing them: "
                        + ", ".join(only_entered[:12])
                        + (" …" if len(only_entered) > 12 else ""),
                        icon=":material/person_add:")

            # Exposure side by side: the number that says whether the set you entered is
            # the set you designed.
            built_exposure = (built.groupby("Name")["Lineup"].nunique()
                              / max(built["Lineup"].nunique(), 1) * 100)
            side = exposure[["Name", "Exposure%"]].rename(
                columns={"Exposure%": "Entered%"}).copy()
            side["Built%"] = side["Name"].map(built_exposure)
            side["Gap"] = side["Entered%"] - side["Built%"].fillna(0)
            st.markdown("**Exposure gap** — entered minus built")
            st.dataframe(
                nfl_charts.round_display(side.sort_values("Gap", ascending=False)),
                hide_index=True, width="stretch",
                column_config={
                    "Entered%": st.column_config.ProgressColumn(
                        "Entered%", min_value=0, max_value=100, format="%.0f%%"),
                    "Built%": st.column_config.ProgressColumn(
                        "Built%", min_value=0, max_value=100, format="%.0f%%"),
                })

            st.markdown("**The generated set**")
            st.dataframe(nfl_charts.round_display(summary), hide_index=True,
                         width="stretch")

with tabs[3]:
    if players is None:
        st.info("Pick a slate in the sidebar to resolve clubs and see the stacks entered.")
    else:
        joined = entries.merge(players[["key", "Team", "DK Pos"]].drop_duplicates("key"),
                               on="key", how="left")
        pass_game = joined[joined["Slot"].isin(("QB", "WR", "TE"))
                           | joined["DK Pos"].astype(str).str.startswith(("WR", "TE"))]
        counts = (pass_game.dropna(subset=["Team"])
                  .groupby(["Entry", "Team"], as_index=False)
                  .agg(Players=("Name", "nunique")))
        stacks = counts[counts["Players"] >= 2]
        if stacks.empty:
            st.info("No entry carries two or more pass-game players from one club.")
        else:
            shape = (stacks.groupby(["Team", "Players"], as_index=False)
                     .agg(Entries=("Entry", "nunique"))
                     .sort_values(["Entries", "Players"], ascending=False))
            st.markdown("**Stack shapes entered** — pass-game players only (QB/WR/TE)")
            st.dataframe(shape, hide_index=True, width="stretch")
            st.caption("A running back never counts toward his own club's stack: a rushing "
                       "touchdown is a drive that did not end in a passing touchdown.")

with tabs[4]:
    by_contest = (entries.groupby("Contest", as_index=False)
                  .agg(Entries=("Entry", "nunique"),
                       Fee=("Entry fee", lambda s: s.iloc[0] if len(s) else "")))
    st.dataframe(by_contest.sort_values("Entries", ascending=False), hide_index=True,
                 width="stretch")
