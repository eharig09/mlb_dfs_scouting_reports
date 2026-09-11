"""Scouting dashboard.

    .venv/Scripts/streamlit run streamlit_app.py

Reads the pipeline's own cached game payloads in `.cache/report_data/`, so it starts in
about a second and never re-runs a report or touches the network. Run a report first and
the game appears here.

The NFL section reads `nfl_boards/` instead -- the pool, lineups and exposure that
`nfl.optimize` writes. Different source, same bargain: it shows what was built and never
rebuilds it.
"""

import os
import sys

import streamlit as st

# The pages import `dashboards.*`, so the project root has to be importable whichever
# directory streamlit was launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dashboards import scales  # noqa: E402 - project root is inserted immediately above

st.set_page_config(page_title="Scouting dashboard", page_icon=":material/query_stats:",
                   layout="wide")

with st.sidebar:
    st.segmented_control(
        "Axis range", ["Focus", "Full"], default="Focus", key=scales.SESSION_KEY,
        help="Focus uses stable comparison windows and keeps extremes on the boundary. "
             "Full expands to every value on the unfiltered board.",
        persist_state="session",
    )

mlb_pages = [
    st.Page("dashboards/app_pages/matchup.py", title="Matchup",
            icon=":material/compare_arrows:", default=True),
    st.Page("dashboards/app_pages/pitching.py", title="Pitching",
            icon=":material/sports_baseball:"),
    st.Page("dashboards/app_pages/slate.py", title="Slate",
            icon=":material/grid_view:"),
    st.Page("dashboards/app_pages/value.py", title="Value",
            icon=":material/payments:"),
    st.Page("dashboards/app_pages/stacks.py", title="Stacks",
            icon=":material/layers:"),
    st.Page("dashboards/app_pages/conditions.py", title="Conditions",
            icon=":material/thermostat:"),
]

# Most of the football pages read **season profiles** from the PFF exports, because that is
# the shape PFF publishes -- a season of deployment per player, not a game log. So the
# question they answer is "how is this player used, and by whom" rather than "is tonight a
# good spot", and that is not a compromise: deployment self-correlates at 0.82-0.95 year
# over year while defensive allowed-rates sit under 0.11. The Slate page is the exception --
# it reads `nfl_boards/`, the pipeline's own lineup outputs.
nfl_pages = [
    st.Page("dashboards/app_pages/nfl_qb.py", title="Quarterbacks",
            icon=":material/sports_football:"),
    st.Page("dashboards/app_pages/nfl_receivers.py", title="Receivers",
            icon=":material/directions_run:"),
    st.Page("dashboards/app_pages/nfl_targets.py", title="Targets",
            icon=":material/target:"),
    st.Page("dashboards/app_pages/nfl_ground.py", title="Ground game",
            icon=":material/sprint:"),
    st.Page("dashboards/app_pages/nfl_redzone.py", title="High value",
            icon=":material/local_fire_department:"),
    st.Page("dashboards/app_pages/nfl_coverage.py", title="Coverage",
            icon=":material/shield:"),
    st.Page("dashboards/app_pages/nfl_trenches.py", title="Trenches",
            icon=":material/fitness_center:"),
    st.Page("dashboards/app_pages/nfl_defense.py", title="Defence",
            icon=":material/security:"),
    st.Page("dashboards/app_pages/nfl_value.py", title="Value",
            icon=":material/payments:"),
    st.Page("dashboards/app_pages/nfl_stacks.py", title="Stacks",
            icon=":material/layers:"),
    st.Page("dashboards/app_pages/nfl_teams.py", title="Teams",
            icon=":material/groups:"),
    st.Page("dashboards/app_pages/nfl_entries.py", title="Entries",
            icon=":material/receipt_long:"),
    st.Page("dashboards/app_pages/nfl_slate.py", title="Slate",
            icon=":material/grid_view:"),
]

# The report explorer covers both sports, so it sits outside either section. An empty key
# puts it above the grouped pages, which is where a "start here" page belongs.
shared_pages = [
    st.Page("dashboards/app_pages/reports.py", title="Reports",
            icon=":material/description:"),
    st.Page("dashboards/app_pages/changes.py", title="Changes",
            icon=":material/difference:"),
]

st.navigation({"": shared_pages, "MLB": mlb_pages, "NFL": nfl_pages}).run()
