"""Clubs as stack units.

A tournament roster is five bats from one club, because their outcomes are correlated — the
innings that produce runs produce them for the whole order at once. So this page aggregates
to the club and asks the roster-construction question, where every other page asks the
scouting one.
"""

import streamlit as st

from dashboards import charts, data, drill, drill_ui, filters, outcomes, salaries
from dashboards import stacks as stacks_module

st.header("Stacks", anchor=False)
st.caption("The top five of each card, priced and in context. The five are taken **by "
           "batting order**, not by projection — a stack is contiguous because that is what "
           "makes the outcomes correlate.")

games = data.list_games()
if games.empty:
    st.info("No cached games found.")
    st.stop()

dates = list(dict.fromkeys(games["date"]))
with st.sidebar:
    st.subheader("Slate", anchor=False)
    date = st.selectbox("Date", dates, key="stacks_date",
    persist_state="session")

table = stacks_module.team_stacks(date)
if table.empty:
    st.warning("No club on this date has enough priced hitters to form a stack.")
    st.stop()

board = salaries.attach_salary(data.hitters_for_date(date), date)
table = stacks_module.add_composite(table, board)

dropped = filters.off_slate_teams(table, date)
with st.sidebar:
    st.subheader("Filter", anchor=False)
    only_slate = st.toggle("Only clubs on a DK slate", value=True, key="stacks_only",
    persist_state="session")
    chosen_teams = st.multiselect("Club", sorted(table["Team"].unique()), default=[],
                                  key="stacks_teams", help="Leave empty for every club.",
                                  persist_state="session")
    st.subheader("Axes", anchor=False)
    axes = {k: v for k, v in charts.STACK_AXES.items() if k in table.columns
            and table[k].notna().any()}
    keys = list(axes)
    x = st.selectbox("Horizontal", keys,
                     index=keys.index("Top5 Salary") if "Top5 Salary" in keys else 0,
                     format_func=lambda k: axes[k], key="stacks_x",
                     persist_state="session")
    y = st.selectbox("Vertical", keys,
                     index=keys.index("Top5 Ceiling") if "Top5 Ceiling" in keys else 1,
                     format_func=lambda k: axes[k], key="stacks_y",
                     persist_state="session")
    size_by = st.selectbox("Size by", keys,
                           index=keys.index("Team Runs") if "Team Runs" in keys else 0,
                           format_func=lambda k: axes[k], key="stacks_size",
                           persist_state="session")

view = filters.on_slate(table, date) if only_slate else table
if chosen_teams:
    view = view[view["Team"].isin(chosen_teams)]
if view.empty:
    st.warning("No club matches those filters.")
    st.stop()

if dropped and only_slate:
    st.info(f"Excluded {', '.join(dropped)} — on no DraftKings slate for {date}.",
            icon=":material/filter_alt:")

if view["Stack Value"].isna().all():
    # A silently absent tile reads as "no stack was good value", which is a different claim
    # from "nothing here is priced".
    st.caption("No DraftKings salaries for this date, so cost-per-point is unavailable — "
               "stacks are ranked on score alone.")

best = view.iloc[0]
# `Stack Value` is ceiling per $1k, so it exists only where the DK salary join landed —
# `project_game` carries no price, and a date with no export leaves the whole column null.
# The old fallback here was `cheapest = best`, which both crashed on the format (None has no
# `.2f`) and, when it did not, labelled the top-*scoring* stack "best per dollar".
priced = view[view["Stack Value"].notna()]
with st.container(horizontal=True):
    st.metric("Stacks", len(view), border=True)
    st.metric("Top stack", str(best["Team"]),
              delta=f"score {best['Stack Score']:.0f} · vs {best['Opp SP']}",
              delta_color="off", border=True)
    if not priced.empty:
        cheapest = priced.loc[priced["Stack Value"].idxmax()]
        st.metric("Best per dollar", str(cheapest["Team"]),
                  delta=f"{cheapest['Stack Value']:.2f} ceiling per $1k",
                  delta_color="off", border=True)
    if view["hr_env"].notna().any():
        hot = view.loc[view["hr_env"].idxmax()]
        st.metric("Best conditions", str(hot["Team"]),
                  delta=f"{hot['hr_env']:.2f} · {hot['park']}", delta_color="off",
                  border=True)

with st.container(border=True):
    st.markdown(f"**{axes[y]} against {axes[x].lower()}** — each club drawn where its top "
                "five sits")
    chart = charts.stack_scatter(view, x, y, size_by)
    if chart is None:
        st.info("That pairing is not available for this slate.")
    else:
        st.altair_chart(chart)
    st.caption("Clubs are named on the chart rather than in a legend — thirty is past what "
               "a legend resolves but well inside what a chart can label. Colour is the "
               "club's own, so the palette is recognition rather than decoding.")

left, right = st.columns(2)
with left.container(border=True):
    st.markdown("**Ranked**")
    rank_by = st.radio("Rank on", ["Stack Score", "Top5 Ceiling", "Stack Value",
                                   "Team Runs"], horizontal=True, key="stacks_rank",
    persist_state="session")
    chart = charts.stack_bars(view, rank_by)
    if chart is not None:
        st.altair_chart(chart)
    st.caption("Stack score blends ceiling per dollar with raw ceiling: value alone "
               "promotes cheap bad offences, and a stack in a three-run game is a bad "
               "stack at any price.")

with right.container(border=True):
    st.markdown("**Open a stack**")
    club = st.selectbox("Club", list(view["Team"]), key="stacks_member_club",
    persist_state="session")
    members = stacks_module.stack_members(date, club)
    if members.empty:
        st.info("No projected hitters cached for that club.")
    else:
        chart = charts.stack_member_bars(members)
        if chart is not None:
            st.altair_chart(chart)
        st.caption(f"The five taken are the top five of the order. {club}'s stack costs "
                   f"${members[members['In stack']]['Salary'].sum():,.0f} of the $50,000 cap.")

with st.container(border=True):
    st.markdown(f"**All {len(view)} stacks** — select a row to open a hitter's evidence")
    columns = ["Team", "Opp", "Opp SP", "Opp SP Hand", "Hitters", "Top5 Salary",
               "Top5 Proj", "Top5 Ceiling", "Stack Value", "Stack Score", "Team Runs",
               "Allowed OPS", "Lineup OPS", "park", "hr_env", "Mean composite",
               "Priority bats"]
    shown = view.reindex(columns=[c for c in columns if c in view.columns])
    st.dataframe(shown, hide_index=True, column_config={
        "Opp SP Hand": st.column_config.TextColumn("Throws"),
        "Top5 Salary": st.column_config.NumberColumn("Top5 $", format="$%d"),
        "Top5 Proj": st.column_config.NumberColumn(format="%.1f"),
        "Top5 Ceiling": st.column_config.NumberColumn(format="%.1f"),
        "Stack Value": st.column_config.NumberColumn("Per $1k", format="%.2f"),
        "Stack Score": st.column_config.NumberColumn("Score", format="%.0f"),
        "Team Runs": st.column_config.NumberColumn("Implied runs", format="%.2f"),
        "Allowed OPS": st.column_config.NumberColumn("SP allows", format="%.3f"),
        "Lineup OPS": st.column_config.NumberColumn("They hit", format="%.3f"),
        "hr_env": st.column_config.NumberColumn("HR env", format="%.2f"),
        "Mean composite": st.column_config.NumberColumn(format="%.1f"),
        "Priority bats": st.column_config.NumberColumn(format="%d"),
    })

with st.container(border=True):
    st.markdown(f"**{club} — every bat on the card**")
    if not members.empty:
        event = st.dataframe(members, hide_index=True, on_select="rerun",
                             selection_mode="single-row", key="stacks_members_table",
                             column_config={
                                 "Slot": st.column_config.NumberColumn("Order",
                                                                       format="%d"),
                                 "Salary": st.column_config.NumberColumn(format="$%d"),
                                 "Proj": st.column_config.NumberColumn(format="%.2f"),
                                 "Ceiling": st.column_config.NumberColumn(format="%.2f"),
                                 "Floor": st.column_config.NumberColumn(format="%.2f"),
                                 "PA": st.column_config.NumberColumn(format="%.2f"),
                                 "In stack": st.column_config.CheckboxColumn("Top five"),
                             })
    else:
        event = None

# Outside the container — Streamlit renders a dialog where it is called from.
chosen = ((event.get("selection") or {}).get("rows") or []) if event is not None else []
if chosen:
    row = members.iloc[chosen[0]]
    payload, meta = drill.payload_for(date, club)
    if payload is None:
        st.warning(f"No cached game found for {club} on {date}.")
    else:
        drill_ui.hitter_dialog(payload, meta, row["Name"], club,
                               outcomes.scored_history())
