"""Park, weather and defence, against the contact profiles that meet them.

Three tables the report already keeps separately. The interaction between them is where the
read is, and it is only visible when they share a plane.
"""

import streamlit as st

from dashboards import charts, data, drill_ui, environment as env, filters, tables

st.header("Conditions", anchor=False)
st.caption("Where a batted ball ends up is decided by the park, the air and the defence. "
           "Every number here is the pipeline's own — the park factors it publishes and the "
           "same weather term the projections use.")

games = data.list_games()
if games.empty:
    st.info("No cached games in `.cache/report_data/`.")
    st.stop()

date, slate, _scoped = filters.scope(games)

staffs = env.staff_environment(date)
off_slate = filters.off_slate_teams(staffs, date, team_column="team")
if staffs.empty:
    st.warning("No batted-ball profiles cached for that date.")
    st.stop()

with st.sidebar:
    st.subheader("Filter", anchor=False)
    units = st.pills("Unit", ["Starter", "Bullpen"], selection_mode="multi",
                     default=["Starter", "Bullpen"], key="conditions_units",
                     persist_state="session")
    teams = st.multiselect("Team", sorted(staffs["team"].unique()), default=[],
                           key="conditions_teams", help="Leave empty for every club.",
                           persist_state="session")
    # Implied once a specific slate is chosen in Scope, so it only appears otherwise.
    only_slate = True
    if slate == filters.ALL_SLATES:
        only_slate = st.toggle("Only clubs on a DK slate", value=True,
                               key="conditions_only",
                               help="Clubs with no priced player that day are dropped.",
                               persist_state="session")
    roofed = st.checkbox("Include closed roofs", value=True, key="conditions_roof",
                         help="A closed roof pins the weather term to exactly 1.00, so "
                              "those games only carry the park factor.",
                         persist_state="session")

view = staffs.copy()
if only_slate:
    view = filters.in_scope(view, date, slate, team_column="team")
if units:
    view = view[view["unit"].isin(units)]
if teams:
    view = view[view["team"].isin(teams)]
if not roofed:
    view = view[~view["enclosed"]]
if view.empty:
    st.warning("No staffs match those filters.")
    st.stop()

if off_slate and only_slate:
    st.info(f"Excluded {', '.join(off_slate)} — on no DraftKings slate for {date}.",
            icon=":material/filter_alt:")

if view.empty:
    st.warning("No staffs match those filters.")
    st.stop()

hottest = view.loc[view["hr_env"].idxmax()]
coldest = view.loc[view["hr_env"].idxmin()]
exposed = view.loc[view["hr_leverage"].idxmax()]
with st.container(horizontal=True):
    st.metric("Staffs", len(view), border=True)
    st.metric("Best HR conditions", f"{hottest['hr_env']:.2f}",
              delta=str(hottest["park"]), delta_color="off", border=True)
    st.metric("Worst HR conditions", f"{coldest['hr_env']:.2f}",
              delta=str(coldest["park"]), delta_color="off", border=True)
    st.metric("Most exposed", f"{exposed['hr_leverage']:+.1f}",
              delta=f"{exposed['name']} · {exposed['FB%']:.0f}% FB",
              delta_color="off", border=True)

with st.container(border=True):
    st.markdown("**Fly-ball rate against the home-run environment** — the rules are league "
                "fly-ball rate and a neutral environment, so the top right quadrant is a "
                "fly-ball staff in conditions that carry")
    chart = charts.hr_exposure_scatter(view)
    exposure_event = drill_ui.chart_with_drilldown(chart, view, date,
                                                   "cond_exposure_pick", kind="staff")
    st.caption("Click any point to open that staff.")
    st.caption("Point size is balls in play. A bullpen's rate is measured over several "
               "hundred and a starter's over a few dozen, and the two are not equally "
               "known.")

with st.container(border=True):
    st.markdown("**Exposure, ranked** — direction and degree for tonight")
    chart = charts.hr_impact_arrows(view)
    arrows_event = drill_ui.chart_with_drilldown(chart, view, date, "cond_arrows_pick",
                                                 kind="staff")
    st.caption("Exposure is a heuristic: fly-ball rate relative to league, scaled by how "
               "far the environment sits from neutral. It says a fly-ball arm in a hot, "
               "short park is more exposed than a sinkerballer in the same park — which is "
               "the reason to put these on one chart. It does **not** claim to know how "
               "many home runs; the projections model that properly. Read the ordering.")

with st.container(border=True):
    st.markdown("**Fieldable contact against the park that rewards it** — a different "
                "question from the home-run factor, and it often points the other way")
    left, right = st.columns(2)
    with left:
        contact = st.radio("Contact measure", ["GB+LD%", "GB%", "LD%"], horizontal=True,
                           key="conditions_contact",
                           persist_state="session")
    with right:
        factor = st.radio("Park factor", list(charts.PARK_AXES), horizontal=True,
                          key="conditions_factor",
                          format_func=lambda k: {"park_2b": "Doubles", "park_r": "Runs",
                                                 "park_hr": "Home runs"}[k],
                          persist_state="session")
    chart = charts.defense_park_scatter(view, x=contact, park_column=factor)
    if chart is None:
        st.info("That pairing is not available for this slate.")
        defense_event = None
    else:
        defense_event = drill_ui.chart_with_drilldown(chart, view, date,
                                                      "cond_defense_pick", kind="staff")
    st.caption("Colour is the sign of hits saved per game by the defence behind that staff "
               "— it swings both ways around zero, so it cannot be a point size.")

with st.container(border=True):
    st.markdown("**Tonight's parks**")
    parks = (view.drop_duplicates("game")
             .reindex(columns=["game", "park", "park_hr", "park_runs", "weather_hr",
                               "hr_env", "temp", "wind", "roof", "condition"])
             .sort_values("hr_env", ascending=False))
    st.dataframe(parks, hide_index=True, column_config={
        "game": st.column_config.TextColumn("Game"),
        "park": st.column_config.TextColumn("Park"),
        "park_hr": st.column_config.NumberColumn("Park HR", format="%.2f"),
        "park_runs": st.column_config.NumberColumn("Park runs", format="%.2f"),
        "weather_hr": st.column_config.NumberColumn("Weather HR", format="%.3f"),
        "hr_env": st.column_config.NumberColumn("Combined", format="%.2f"),
        "temp": st.column_config.NumberColumn("Temp °F", format="%.0f"),
        "wind": st.column_config.TextColumn("Wind"),
        "roof": st.column_config.TextColumn("Roof"),
        "condition": st.column_config.TextColumn("Sky"),
    })
    st.caption("Weather HR is exactly 1.000 under a closed roof — still air and a "
               "controlled temperature mean neither term should move.")

with st.container(border=True):
    st.markdown(f"**All {len(view)} staffs**")
    st.dataframe(
        view.reindex(columns=["game", "team", "unit", "name", "BIP", "GB%", "FB%", "LD%",
                              "GB+LD%", "HH%", "Brl%", "park_hr", "weather_hr", "hr_env",
                              "hr_leverage", "hits_saved", "defense_grade"])
            .sort_values("hr_leverage", ascending=False),
        hide_index=True, column_config={
            "game": st.column_config.TextColumn("Game"),
            "team": st.column_config.TextColumn("Team"),
            "unit": st.column_config.TextColumn("Unit"),
            "name": st.column_config.TextColumn("Staff"),
            "BIP": st.column_config.NumberColumn(format="%d"),
            "park_hr": st.column_config.NumberColumn("Park HR", format="%.2f"),
            "weather_hr": st.column_config.NumberColumn("Weather", format="%.3f"),
            "hr_env": st.column_config.NumberColumn("HR env", format="%.2f"),
            "hr_leverage": st.column_config.NumberColumn("Exposure", format="%+.1f"),
            "hits_saved": st.column_config.NumberColumn("Hits saved/g", format="%+.2f"),
            "defense_grade": st.column_config.TextColumn("Defence"),
        })

# Outside every container, and only one dialog per run — the first resolving click wins.
opened = False
for source_event in (locals().get("exposure_event"), locals().get("arrows_event"),
                     locals().get("defense_event")):
    if source_event is not None and not opened:
        opened = drill_ui.open_selection(source_event, view, date, kind="staff")
