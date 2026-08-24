"""Every hitter on a date, across all games — the view the per-game workbook cannot give.

The composite can be read against any of several bases, and the matchup can be read against
what tonight's arm actually surrenders rather than against the hitter's own season. Both
switches exist because "is this a good spot" is several different questions and a fixed pair
of axes silently picks one of them.
"""

import streamlit as st

from dashboards import charts, data, drill, drill_ui, filters, outcomes, salaries, tables

st.header("Slate", anchor=False)
st.caption("Every hitter on a date, across all games. Filters apply to every panel here.")

games = data.list_games()
if games.empty:
    st.info("No cached games found. Run a report first — the dashboard reads "
            "`.cache/report_data/`, it never re-runs the pipeline.")
    st.stop()

date, slate, _scoped = filters.scope(games)

hitters = data.hitters_for_date(date)
if hitters.empty:
    st.warning("No hitter composites cached for that date.")
    st.stop()

# Prices come first so the position and DK-slate filters have something to work with; a
# hitter with no salary row keeps his composite and simply cannot be filtered by slot.
board = salaries.attach_salary(hitters, date)

# **Cached games are not the DK slate.** The dashboard reads every game the report ran,
# and DraftKings puts up a subset — so without this a club nobody can roster is ranked,
# coloured and counted alongside the ones you can. Picking a specific slate in Scope narrows
# it further, to the clubs *that contest* could actually roster.
dropped = filters.off_slate_teams(board, date)
only_slate = True
if slate == filters.ALL_SLATES:
    with st.sidebar:
        only_slate = st.toggle("Only clubs on a DK slate", value=True, key="slate_only",
                               help="Off, nothing is excluded; on, clubs with no priced "
                                    "player that day are dropped.",
                               persist_state="session")
if only_slate:
    board = filters.in_scope(board, date, slate)

view = filters.sidebar(board, "slate", date=date,
                       show=("signal", "slate", "position", "team", "bats", "ab"))

# Every club on the slate, not the filtered subset: colour has to follow the club, so
# narrowing the filter must not repaint whoever survives it.
slate_teams = charts.team_domain(board)
bases = filters.available_bases(board)
if not bases:
    # `available_bases` needs a column with at least 20 non-null rows, so a thin slate can
    # leave it empty. An empty selectbox returns None and every `bases[basis]` below it then
    # raises KeyError — which surfaced as a blank page, not as a message.
    st.warning("No hitting basis on this slate carries enough of a sample to read the "
               "composite against. Pick another date, or loosen the filters.")
    st.stop()
with st.sidebar:
    st.subheader("Read against", anchor=False)
    basis = st.selectbox("Basis", list(bases), key="slate_basis",
                         format_func=lambda k: bases[k],
                         persist_state="session")
    colour_by = st.radio("Colour by", ["Team", "Signal"], horizontal=True,
                         key="slate_colour",
                         help="Team colour with signal as shape is for spotting a club's "
                              "bats clustering in one corner — the stacking question. "
                              "Signal colour reads one player at a time.",
                         persist_state="session")
    size_by = st.selectbox("Size points by", list(charts.SIZE_SCALES), key="slate_size",
                           format_func=lambda k: charts.SIZE_SCALES[k],
                           help="Projection has the widest spread and the strongest link "
                                "to actual points. Ceiling is nearly flat across a slate.",
                           persist_state="session")
    backdrop = st.toggle("Shade by past hit rate", value=False, key="slate_backdrop",
                         help="Historical hit rate from scored contests, behind tonight's "
                              "hitters. See the Signal page for what is behind it.",
                         persist_state="session")

if dropped and only_slate:
    st.info(f"Excluded {', '.join(dropped)} — on no DraftKings slate for {date}, so their "
            "hitters cannot be rostered.", icon=":material/filter_alt:")
elif dropped:
    st.warning(f"{', '.join(dropped)} are on no DraftKings slate for {date}. Their hitters "
               "are ranked below but cannot be rostered.", icon=":material/warning:")

with st.container(horizontal=True):
    st.metric("Games", board["game"].nunique(), border=True)
    st.metric("Hitters", len(board), border=True)
    st.metric("Shown", len(view), border=True)
    st.metric("Priority", int((view["Signal"] == "Priority").sum()), border=True)

if view.empty:
    st.warning("No hitters match those filters.")
    st.stop()

basis_event = None
grid = None
if backdrop:
    history = outcomes.scored_history()
    grid = outcomes.hit_grid(history, "Composite", basis)
    if grid.empty:
        st.caption(f"Not enough scored history to shade composite against {bases[basis]}.")

with st.container(border=True):
    st.markdown(f"**Matchup score against {bases[basis].lower()}** — the rules are the "
                "slate medians, so the top-right quadrant is a good matchup landing on a "
                "bat that is already doing this well")
    if colour_by == "Team":
        chart = charts.team_signal_scatter(
            view, "Composite", basis, "Composite matchup score", bases[basis],
            grid=grid, size_by=size_by, domain=slate_teams)
    else:
        chart = charts.basis_scatter(view, basis, bases[basis], grid, size_by=size_by)
    if chart is None:
        st.info(f"Not enough hitters carry {basis} on this slate.")
    else:
        basis_event = drill_ui.chart_with_drilldown(chart, view, date, "slate_basis_pick")
        note = ("Colour is the club and shape is the signal. "
                if colour_by == "Team" else "")
        if colour_by == "Team" and len(slate_teams) > charts.TEAM_IDENTITY_CAP:
            note += (f"With {len(slate_teams)} clubs on the board no palette can name each "
                     "one — read colour as clustering and use the team filter or the "
                     "tooltip to name a club. ")
        st.caption(note + "Click any point to open that hitter's evidence.")

# --- against what the arm actually allows ---------------------------------------------
allowed = data.hitters_vs_allowed(date)
if not allowed.empty:
    keys = set(view["Name"])
    matchups = allowed[allowed["Name"].isin(keys)]
    if not matchups.empty:
        with st.container(border=True):
            st.markdown("**Against what tonight's arm actually allows** — the diagonal is "
                        "what that pitcher surrenders to batters of his side, so above it "
                        "is a hitter beating the arm's own baseline")
            measure = st.radio(
                "His line", ["Arsenal OPS", "Platoon OPS", "Season OPS"], horizontal=True,
                key="slate_allowed_measure",
                persist_state="session")
            chart = charts.arsenal_vs_allowed_scatter(
                matchups, measure,
                teams=slate_teams if colour_by == "Team" else None)
            if chart is None:
                st.info("No hitter on this slate has a measured line against his arsenal.")
            else:
                allowed_event = drill_ui.chart_with_drilldown(
                    chart, matchups, date, "slate_allowed_pick")
            st.caption(
                "Season OPS answers whether he is better than usual against these pitches — "
                "a question about the hitter alone. This asks the matchup question instead: "
                "a .700 bat facing an arm that surrenders .639 to his side is in a worse "
                "spot than a .680 bat facing one that surrenders .780. Hollow points are "
                "thin samples; switch hitters are placed on the side they will actually "
                "bat from.")

        with st.container(border=True):
            column = {"Arsenal OPS": "Arsenal Edge", "Platoon OPS": "Platoon Edge",
                      "Season OPS": "Season Edge"}[measure]
            st.markdown(f"**Biggest gaps** — {measure} minus what the arm allows his side")
            chart = charts.edge_bars(
                matchups, column,
                domain=slate_teams if colour_by == "Team" else None)
            if chart is None:
                st.info("No hitter clears the sample floor for this measure.")
            else:
                edge_event = drill_ui.chart_with_drilldown(
                    chart, matchups, date, "slate_edge_pick")

left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.markdown("**Top composites**")
        top = (view.sort_values("Composite", ascending=False)
               .reindex(columns=["Name", "Team", "game", "Composite", "Signal"]).head(15))
        st.dataframe(tables.highlight(top), hide_index=True,
                     column_config={"game": st.column_config.TextColumn("Game"),
                                    "Composite": st.column_config.NumberColumn(
                                        format="%.1f")})
with right:
    with st.container(border=True):
        st.markdown("**By club**")
        by_team = (view.groupby("Team", as_index=False)
                   .agg(hitters=("Name", "size"), composite=("Composite", "mean"),
                        priority=("Signal", lambda s: int((s == "Priority").sum())))
                   .sort_values("composite", ascending=False))
        st.dataframe(tables.highlight(by_team), hide_index=True, column_config={
            "hitters": st.column_config.NumberColumn("Hitters", format="%d"),
            "composite": st.column_config.NumberColumn("Mean composite", format="%.1f"),
            "priority": st.column_config.NumberColumn("Priority", format="%d")})

with st.container(border=True):
    st.markdown(f"**All {len(view)} hitters shown** — select a row to open his evidence")
    columns = ["game", "Team", "Name", "DK Pos", "Bats", "Salary", "Composite", basis,
               "Signal", "Season AB"]
    table = (view.reindex(columns=[c for c in dict.fromkeys(columns) if c in view.columns])
                 .sort_values("Composite", ascending=False).reset_index(drop=True))
    event = st.dataframe(
        table, hide_index=True, on_select="rerun", selection_mode="single-row",
        key="slate_board", column_config={
            "game": st.column_config.TextColumn("Game"),
            "DK Pos": st.column_config.TextColumn("Slot"),
            "Salary": st.column_config.NumberColumn(format="$%d"),
            "Composite": st.column_config.NumberColumn(format="%.1f"),
            "Season AB": st.column_config.NumberColumn(format="%d"),
        })

# Every dialog is opened out here, outside the bordered containers: Streamlit renders a
# dialog where it is called from, and nesting it inside one clips the layout it builds.
# Only one may open per run, so the first click that resolves wins.
history = outcomes.scored_history()
opened = False
for source_event, source_frame in ((basis_event, view),
                                   (locals().get("allowed_event"), locals().get("matchups")),
                                   (locals().get("edge_event"), locals().get("matchups"))):
    if source_event is not None and source_frame is not None and not opened:
        opened = drill_ui.open_selection(source_event, source_frame, date,
                                         history=history)

chosen = [] if opened else ((event.get("selection") or {}).get("rows") or [])
if chosen:
    row = table.iloc[chosen[0]]
    payload, meta = drill.payload_for(date, row["Team"])
    if payload is None:
        st.warning(f"No cached game found for {row['Team']} on {date}.")
    else:
        drill_ui.hitter_dialog(payload, meta, row["Name"], row["Team"], history)
