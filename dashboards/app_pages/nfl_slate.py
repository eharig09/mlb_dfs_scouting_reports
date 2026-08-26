"""One NFL slate: the pool it was built from, the lineups, and what they are exposed to.

The MLB pages read cached game payloads. This one reads the pipeline's **outputs** --
`pool_<slate>.csv`, `lineups_<slate>.csv`, `exposure_<slate>.csv` -- so it shows what was
actually built rather than rebuilding it, and opens instantly.

**Exposure gets its own panel rather than a column somewhere.** The MLB side found 54%
exposure on a punt owned by 1.6% of the field, and nobody goes looking for an exposure they
do not already suspect. It is the first thing on the page after the lineups themselves.
"""

import pandas as pd
import streamlit as st

from dashboards import nfl_boards

st.header("NFL slate", anchor=False)
st.caption("Reads `nfl_boards/` — the pool, lineups and exposure the pipeline wrote. "
           "It never re-runs the optimizer.")

dates = nfl_boards.list_dates()
if not dates:
    st.info("Nothing in `nfl_boards/` yet. Build a slate first:\n\n"
            "```\npython -m nfl.optimize --salaries \"path/to/DKSalaries.csv\" --write-pool\n"
            "python -m nfl.optimize --salaries \"path/to/DKSalaries.csv\" --n 20\n```")
    st.stop()

with st.sidebar:
    st.subheader("Scope", anchor=False)
    date = st.selectbox("Date", dates, key="nfl_date", persist_state="session")
    slates = nfl_boards.list_slates(date)
    if not slates:
        st.warning("No slates written for that date.")
        st.stop()
    slate = st.selectbox("Slate", slates, key="nfl_slate", persist_state="session")

lineups = nfl_boards.read("lineups", date, slate)
pool = nfl_boards.read("pool", date, slate)
exposure = nfl_boards.read("exposure", date, slate)

if lineups is None and pool is None:
    st.warning(f"Nothing written for **{slate}** on {date} yet.")
    st.stop()

summary = nfl_boards.lineup_summary(lineups)
players, teams = nfl_boards.split_exposure(exposure)

with st.container(horizontal=True):
    st.metric("Lineups", len(summary) if not summary.empty else 0, border=True)
    st.metric("Pool", len(pool) if pool is not None else 0, border=True)
    if not summary.empty:
        st.metric("Best ceiling", f"{summary['Ceiling'].max():.1f}", border=True)
        st.metric("Median salary", f"${summary['Salary'].median():,.0f}", border=True)

tabs = st.tabs(["Lineups", "Exposure", "Pool"])

with tabs[0]:
    if summary.empty:
        st.info("No lineups built for this slate yet — run `nfl.optimize --n 20`.")
    else:
        st.markdown("**Every lineup, best ceiling first**")
        st.dataframe(nfl_boards.arrow_safe(summary), hide_index=True, width="stretch")

        picked = st.selectbox("Inspect a lineup", summary["Lineup"].tolist(),
                              key="nfl_lineup_pick", persist_state="session")
        one = lineups[lineups["Lineup"] == picked]
        st.markdown(f"**Lineup {picked}** — "
                    f"${one['Lineup Salary'].iloc[0]:,.0f}, "
                    f"proj {one['Lineup Proj'].iloc[0]:.1f}, "
                    f"ceiling {one['Lineup Ceiling'].iloc[0]:.1f}")
        columns = [c for c in ("Slot", "Name", "Pos", "Team", "Opp", "Game",
                               "Salary", "Proj", "Ceiling") if c in one.columns]
        st.dataframe(nfl_boards.arrow_safe(one[columns]), hide_index=True, width="stretch")

with tabs[1]:
    if players.empty:
        st.info("No exposure written — it is produced with every lineup set.")
    else:
        st.markdown("**Per player** — how much of the set rides on each name")
        st.dataframe(
            nfl_boards.arrow_safe(players), hide_index=True, width="stretch",
            column_config={"Exposure%": st.column_config.ProgressColumn(
                "Exposure%", min_value=0, max_value=100, format="%.1f%%")})
        if not teams.empty:
            st.markdown("**Per team**")
            st.dataframe(
                nfl_boards.arrow_safe(teams), hide_index=True, width="stretch",
                column_config={"Exposure%": st.column_config.ProgressColumn(
                    "Exposure%", min_value=0, max_value=100, format="%.1f%%")})

with tabs[2]:
    if pool is None:
        st.info("No pool written — run `nfl.optimize --write-pool`.")
    else:
        st.caption("The editable pool as it stands on disk. Edit it in Excel, not here — "
                   "the optimizer reads that file, and a change made on this page would "
                   "not reach it.")
        # `frame.get(col)` returns a **scalar** when the column is absent, not an empty
        # Series -- the recurring trap in this codebase. An all-blank pool reads its edit
        # columns back as float NaN rather than as strings, so both cases are handled by
        # asking whether the column exists before touching it.
        marked = pool.copy()
        flagged = pd.Series(False, index=marked.index)
        for column in ("Lock", "Exclude"):
            if column in marked.columns:
                text = marked[column].astype("string").fillna("").str.strip()
                marked[column] = text
                flagged = flagged | (text != "")
        edited = marked[flagged]
        if not edited.empty:
            st.markdown(f"**{len(edited)} hand-marked**")
            st.dataframe(nfl_boards.arrow_safe(edited), hide_index=True, width="stretch")
        st.markdown("**Full pool**")
        st.dataframe(nfl_boards.arrow_safe(pool), hide_index=True, width="stretch")

path = nfl_boards.path_of("lineups", date, slate) or nfl_boards.path_of("pool", date, slate)
if path:
    st.caption(f"Read from `{path}`")
