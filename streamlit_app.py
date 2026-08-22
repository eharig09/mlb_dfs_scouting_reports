"""Scouting dashboard.

    .venv/Scripts/streamlit run streamlit_app.py

Reads the pipeline's own cached game payloads in `.cache/report_data/`, so it starts in
about a second and never re-runs a report or touches the network. Run a report first and
the game appears here.

MLB only for now; the NFL slate board is the next page to add, against `nfl.slate`.
"""

import os
import sys

import streamlit as st

# The pages import `dashboards.*`, so the project root has to be importable whichever
# directory streamlit was launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="Scouting dashboard", page_icon=":material/query_stats:",
                   layout="wide")

pages = [
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
st.navigation({"MLB": pages}).run()
