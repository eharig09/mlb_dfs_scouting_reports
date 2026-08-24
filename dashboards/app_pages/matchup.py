import streamlit as st

from dashboards import charts, data, drill_ui, filters, outcomes, salaries, tables

st.header("Matchup", anchor=False)

games = data.list_games()
if games.empty:
    st.info("No cached games found. Run a report first — the dashboard reads "
            "`.cache/report_data/`, it never re-runs the pipeline.")
    st.stop()

date, slate, in_scope = filters.scope(games)
on_date = in_scope[in_scope["date"] == date]
if on_date.empty:
    st.warning(f"No cached game on the {slate} slate for {date}.")
    st.stop()
with st.sidebar:
    st.subheader("Game", anchor=False)
    label = st.selectbox("Game", list(on_date["label"]), key="matchup_game",
                         persist_state="session")

meta = on_date[on_date["label"] == label].iloc[0].to_dict()
payload = data.load_payload(meta["path"])
hitters = data.with_positions(data.hitters(payload, meta), payload, meta)
# Price only feeds the shape channel on the split panels. Attached here rather than inside
# the chart builder so an unpriced date costs one lookup, not one per panel.
hitters = salaries.attach_salary(hitters, date)

if hitters.empty:
    st.warning("This game's payload has no hitter composite.")
    st.stop()

with st.sidebar:
    st.subheader("Filter", anchor=False)
    signals = st.pills("Signal", data.SIGNAL_ORDER, selection_mode="multi",
                       default=data.SIGNAL_ORDER, key="matchup_signals",
                       persist_state="session")
    min_ab = st.slider("Minimum season at-bats", 0, 500, 0, step=25, key="matchup_ab",
    persist_state="session")
    # Positions come from the posted lineup card, so this lists only what is actually in
    # tonight's card rather than every position that exists.
    available_positions = sorted(p for p in hitters["Pos"].dropna().unique() if p)
    positions = st.multiselect("Position", available_positions, default=[],
                               key="matchup_pos",
                               help="Leave empty for the whole card.",
                               persist_state="session")
    bats = st.segmented_control("Bats", ["All", "L", "R", "S"], default="All",
                                key="matchup_bats",
                                persist_state="session")
    # No club or DK-slate control here: this page is one game, so both would be controls
    # that cannot change anything, and an inert filter is worse than an absent one.

view = hitters.copy()
if signals:
    view = view[view["Signal"].isin(signals)]
view = view[view["Season AB"].fillna(0) >= min_ab]
if positions:
    view = view[view["Pos"].isin(positions)]
if bats and bats != "All":
    view = view[view["Bats"].astype(str).str.upper() == bats]

teams = list(dict.fromkeys(hitters["Team"].dropna()))
priority = int((hitters["Signal"] == "Priority").sum())
fades = int((hitters["Signal"] == "Fade").sum())
best = hitters.sort_values("Composite", ascending=False).iloc[0]

with st.container(horizontal=True):
    st.metric("Hitters", len(hitters), border=True)
    st.metric("Priority", priority, border=True)
    st.metric("Fade", fades, border=True)
    st.metric("Top composite", f"{best['Composite']:.0f}",
              delta=str(best["Name"]), delta_color="off", border=True)

if view.empty:
    st.warning("No hitters match those filters.")
    st.stop()

# One control for both split panels: they answer the same question about different
# samples, so reading them on two different axis pairs is a needless translation step.
split_view = charts.SPLIT_VIEWS[
    st.segmented_control("Split panels", list(charts.SPLIT_VIEWS),
                         default=list(charts.SPLIT_VIEWS)[0], key="matchup_split_view",
                         help="How far he is from his own baseline against tonight's "
                              "composite, or the original OPS-against-OPS parity view.",
                         persist_state="session")
    or list(charts.SPLIT_VIEWS)[0]]

left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.markdown("**Platoon** — his OPS against the hand he draws tonight")
        chart = charts.platoon_scatter(view, view=split_view)
        # A statement, not a ternary expression: Streamlit's magic display writes the value
        # of any bare top-level expression to the page, so the ternary form printed a
        # DeltaGenerator repr and its own source line into the app.
        if chart is not None:
            platoon_event = drill_ui.chart_with_drilldown(chart, view, date,
                                                          "matchup_platoon_pick")
        else:
            st.caption("No platoon splits for these hitters.")
with right:
    with st.container(border=True):
        st.markdown("**Matchup against form** — top-right is strong matchup and hot")
        chart = charts.form_scatter(view)
        if chart is not None:
            form_event = drill_ui.chart_with_drilldown(chart, view, date,
                                                       "matchup_form_pick")
        else:
            st.caption("No recent-form index for these hitters.")

with st.container(border=True):
    st.markdown("**Arsenal fit** — his OPS against the pitch mix this arm actually throws")
    chart = charts.arsenal_scatter(view, view=split_view)
    if chart is not None:
        arsenal_event = drill_ui.chart_with_drilldown(chart, view, date,
                                                      "matchup_arsenal_pick")
        st.caption("**Split vs composite** plots the raw split — the number a decision "
                   "gets made on — against tonight's matchup score, with the dashed line at "
                   "the median of the bats shown and shape marking the DK price tier. "
                   "**OPS against OPS** is the same split against his season line, where "
                   "the diagonal shows whether tonight's hand is unusual for him. Click any "
                   "point to open that hitter's evidence; the four furthest from the line "
                   "are named, and the rest are on hover.")
    else:
        st.caption("No batter-vs-arsenal sample for this game.")

with st.container(border=True):
    st.markdown(f"**{len(view)} hitters** · {' / '.join(teams)}")
    columns = ["Team", "Name", "Pos", "Spot", "Bats", "Composite", "Signal",
               "Season OPS", "Season AB",
               "Platoon OPS", "Platoon AB", "Arsenal OPS", "Arsenal AB",
               "Off Szn", "Off L28"]
    table = (view.reindex(columns=[c for c in columns if c in view.columns])
                 .sort_values("Composite", ascending=False).reset_index(drop=True))
    event = st.dataframe(
        table,
        hide_index=True, on_select="rerun", selection_mode="single-row",
        key="matchup_board",
        column_config={
            "Composite": st.column_config.NumberColumn(format="%.1f"),
            "Season OPS": st.column_config.NumberColumn(format="%.3f"),
            "Platoon OPS": st.column_config.NumberColumn(format="%.3f"),
            "Arsenal OPS": st.column_config.NumberColumn(format="%.3f"),
            "Season AB": st.column_config.NumberColumn(format="%d"),
            "Platoon AB": st.column_config.NumberColumn(format="%d"),
            "Arsenal AB": st.column_config.NumberColumn(format="%d"),
            "Off Szn": st.column_config.NumberColumn(format="%d"),
            "Off L28": st.column_config.NumberColumn(format="%d"),
            "Spot": st.column_config.NumberColumn("Order", format="%d"),
        },
    )
    st.caption("Select a row to open what the report used for that hitter.")

# Outside the container: Streamlit renders a dialog where it is called from, and nesting it
# inside one clips the layout it builds.
history = outcomes.scored_history()
opened = False
for source_event in (locals().get("platoon_event"), locals().get("form_event"),
                     locals().get("arsenal_event")):
    if source_event is not None and not opened:
        opened = drill_ui.open_selection(source_event, view, date, history=history)

chosen = [] if opened else ((event.get("selection") or {}).get("rows") or [])
if chosen:
    row = table.iloc[chosen[0]]
    drill_ui.hitter_dialog(payload, meta, row["Name"], row["Team"], history)
