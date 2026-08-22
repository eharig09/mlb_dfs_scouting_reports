import streamlit as st

from dashboards import charts, data, drill, drill_ui, filters, outcomes, salaries

st.header("Value", anchor=False)
st.caption("Each hitter measured against his own price bracket. Salary and composite "
           "correlate at about +0.73, so the raw score mostly restates the market — what is "
           "worth looking at is how far a bat sits from par for what he costs.")

priced_dates = salaries.available_dates()
if not priced_dates:
    st.info("No DraftKings salary files found in `dfs_daily_files/`.")
    st.stop()

games = data.list_games()
cached_dates = set(games["date"]) if not games.empty else set()
usable = [d for d in priced_dates if d in cached_dates]
if not usable:
    st.info("No date has both a cached report and a salary file.")
    st.stop()

with st.sidebar:
    st.subheader("Slate", anchor=False)
    date = st.selectbox("Date", usable, key="value_date",
    persist_state="session")

hitters = data.hitters_for_date(date)
if hitters.empty:
    st.warning("No hitter composites cached for that date.")
    st.stop()

board = data.attach_projection(salaries.attach_salary(hitters, date), date)
# Priced-only already excludes an off-slate club's hitters; the colour domain has to be
# built from the same subset or the legend carries clubs no point can belong to.
priced = board[board["Salary"].notna()]
slate_teams = charts.team_domain(priced)
if priced.empty:
    st.warning("No hitter on this slate matched a salary row.")
    st.stop()

# Signal, DK slate, position, club and at-bats are the shared bar, so a reader who narrows
# the Slate page and clicks here is looking at the same subset. Price is specific to this
# page and stays local.
view = filters.sidebar(priced, "value", date=date,
                       show=("signal", "slate", "position", "team", "bats", "ab"))
with st.sidebar:
    price = st.slider("Salary", int(priced["Salary"].min()), int(priced["Salary"].max()),
                      (int(priced["Salary"].min()), int(priced["Salary"].max())),
                      step=100, key="value_price",
                      persist_state="session")
view = view[view["Salary"].between(*price)]

best = priced.nlargest(1, "surplus").iloc[0]
worst = priced.nsmallest(1, "surplus").iloc[0]
with st.container(horizontal=True):
    st.metric("Hitters priced", len(priced), border=True)
    st.metric("Shown", len(view), border=True)
    st.metric("Best for the price", f"{best['surplus']:+.0f}",
              delta=f"{best['Name']} · ${best['Salary']:,.0f}", delta_color="off",
              border=True)
    st.metric("Worst for the price", f"{worst['surplus']:+.0f}",
              delta=f"{worst['Name']} · ${worst['Salary']:,.0f}", delta_color="off",
              border=True)

if view.empty:
    st.warning("No hitters match those filters.")
    st.stop()

with st.container(border=True):
    st.markdown("**Composite against salary** — the dashed line is par for each price band, "
                "so distance above it is the matchup you are not paying for")
    # The historical layer is opt-in rather than always on. It answers a different question
    # from the points ("where have picks worked?" against "who is on tonight's board?") and
    # a second colour legend on by default makes the slate harder to read, not easier.
    colour_by = st.radio("Colour by", ["Team", "Signal"], horizontal=True,
                         key="value_colour",
                         help="Team colour with signal as shape is for spotting a club's "
                              "bats clustering in one corner — the stacking question.",
                         persist_state="session")
    size_by = st.selectbox("Size points by", list(charts.SIZE_SCALES), key="value_size",
                           format_func=lambda k: charts.SIZE_SCALES[k],
                           help="Projection has the widest spread and the strongest link "
                                "to actual points. Ceiling is nearly flat across a slate.",
                           persist_state="session")
    backdrop = st.toggle("Shade by where past picks have hit", value=False,
                         key="value_backdrop",
                         help="Historical hit rate from scored contests, behind tonight's "
                              "hitters. See the Signal page for what is behind it.",
                         persist_state="session")
    grid = None
    if backdrop:
        history = outcomes.scored_history()
        grid = outcomes.hit_grid(history, "Salary", "Composite")
        if grid.empty:
            st.caption("Not enough scored history to shade this plane yet.")
    if colour_by == "Team":
        chart = charts.team_signal_scatter(
            view, "Salary", "Composite", "DraftKings salary", "Composite matchup score",
            grid=grid if backdrop else None, size_by=size_by, domain=slate_teams,
            x_format="$,.0f")
    else:
        chart = (charts.value_scatter_with_history(view, grid) if backdrop
                 else charts.salary_scatter(view))
    salary_event = drill_ui.chart_with_drilldown(chart, view, date, "value_salary_pick")
    note = "Colour is the club, shape is the signal. " if colour_by == "Team" else ""
    st.caption(note + "Click any point to open that hitter's evidence.")

with st.container(border=True):
    st.markdown("**Projection against price** — salary and composite already correlate at "
                "about +0.73, so this is the pairing that carries what the matchup score "
                "does not: the projection folds in opportunity, and it out-predicts both.")
    # Projection against price, which is the pairing the first panel does *not* show — the
    # two would otherwise be the same chart twice under the team encoding.
    if colour_by == "Team":
        chart = charts.team_signal_scatter(
            view, "Salary", size_by, "DraftKings salary", charts.SIZE_SCALES[size_by],
            size_by=size_by, domain=slate_teams, x_format="$,.0f")
    else:
        chart = charts.ceiling_scatter(view, grid if backdrop else None, size_by=size_by)
    if chart is None:
        st.info("No projections available for this slate.")
    else:
        ceiling_event = drill_ui.chart_with_drilldown(chart, view, date,
                                                      "value_ceiling_pick")

with st.container(border=True):
    st.markdown("**Ceiling for the money** — the dashed line is the going rate for upside "
                "on this slate, so what matters is distance above it")
    chart = charts.ceiling_value_scatter(
        view, domain=slate_teams if colour_by == "Team" else None)
    upside_event = drill_ui.chart_with_drilldown(chart, view, date, "value_upside_pick")

with st.container(border=True):
    st.markdown("**Surplus** — how far each bat beats or misses the going rate for his price")
    chart = charts.surplus_bars(view)
    surplus_event = drill_ui.chart_with_drilldown(chart, view, date, "value_surplus_pick")

with st.container(border=True):
    st.markdown("**Par for the price on this slate** — the curve every hitter above is "
                "measured against")
    curve = salaries.band_curve(view)
    st.dataframe(
        curve.reindex(columns=["band", "n", "midpoint", "expected"]),
        hide_index=True,
        column_config={
            "band": st.column_config.TextColumn("Price band"),
            "n": st.column_config.NumberColumn("Hitters", format="%d"),
            "midpoint": st.column_config.NumberColumn("Median salary", format="$%d"),
            "expected": st.column_config.NumberColumn("Par composite", format="%.1f"),
        },
    )

with st.container(border=True):
    st.markdown(f"**All {len(view)} priced hitters** — select a row to open his evidence")
    columns = ["game", "Team", "Name", "DK Pos", "Bats", "Salary", "Composite",
               "expected", "surplus", "per_1k", "Proj", "Ceiling", "Floor", "Signal",
               "Season OPS", "Season AB"]
    table = (view.reindex(columns=[c for c in columns if c in view.columns])
                 .sort_values("surplus", ascending=False).reset_index(drop=True))
    event = st.dataframe(
        table, hide_index=True, on_select="rerun", selection_mode="single-row",
        key="value_board",
        column_config={
            "game": st.column_config.TextColumn("Game"),
            "DK Pos": st.column_config.TextColumn("Slot"),
            "Salary": st.column_config.NumberColumn(format="$%d"),
            "Composite": st.column_config.NumberColumn(format="%.1f"),
            "expected": st.column_config.NumberColumn("Par", format="%.1f"),
            "surplus": st.column_config.NumberColumn("Surplus", format="%+.1f"),
            "per_1k": st.column_config.NumberColumn("Per $1k", format="%.1f"),
            "Proj": st.column_config.NumberColumn(format="%.1f"),
            "Ceiling": st.column_config.NumberColumn(format="%.1f"),
            "Floor": st.column_config.NumberColumn(format="%.1f"),
            "Season OPS": st.column_config.NumberColumn(format="%.3f"),
            "Season AB": st.column_config.NumberColumn(format="%d"),
        },
    )

# The dialog is opened outside the bordered container: Streamlit renders a dialog wherever
# it is called from, and nesting it inside a container clips the layout it builds.
# Scored history is loaded once here rather than inside each branch — it reads every
# contest export, and every branch below wants the same frame.
history = outcomes.scored_history()
opened = False
for source_event in (salary_event, locals().get("ceiling_event"),
                     locals().get("upside_event"), locals().get("surplus_event")):
    if source_event is not None and not opened:
        opened = drill_ui.open_selection(source_event, view, date, history=history)

chosen = [] if opened else ((event.get("selection") or {}).get("rows") or [])
if chosen:
    row = table.iloc[chosen[0]]
    payload, meta = drill.payload_for(date, row["Team"])
    if payload is None:
        st.warning(f"No cached game found for {row['Team']} on {date}.")
    else:
        drill_ui.hitter_dialog(payload, meta, row["Name"], row["Team"], history)
