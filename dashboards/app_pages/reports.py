"""The generated report workbooks, browsable from inside the app.

The pipeline writes Excel — one workbook per game for baseball, one per slate for football —
and those are what you actually read on a given morning. They lived only on the machine that
made them. This page opens them here, and opens the *published* copy when the workbook
itself is not on disk, which is always the case on a host.

**A workbook is a document, not a dataset.** The sheets are laid out for a human: spacer
rows, merged title bars, trailing blank columns. They are cleaned for display and shown as
written, in the workbook's own sheet order — this page does not re-sort, re-rank or
re-interpret anything, because the report already made those decisions and disagreeing with
it silently would be worse than not showing it at all.
"""

import streamlit as st

from dashboards import workbooks

st.header("Reports", anchor=False)
st.caption("Generated scouting workbooks — MLB games and NFL slates.")

modes = workbooks.available()
found = workbooks.list_workbooks()
published = workbooks.published_reports()

if found.empty and not published:
    st.info("No generated workbooks found.\n\n"
            "- MLB: `python scouting_report.py --date <date> --all-games --format xlsx`\n"
            "- NFL: `python -m nfl.cli --salaries <DKSalaries.csv>`\n\n"
            "To read them on a host, publish them first: "
            "`python -m dashboards.publish_reports --date <date>`.")
    st.stop()

# A host has the published bundle and no workbooks. Rebuild the listing from the manifest so
# the page works identically in both places.
if found.empty and published:
    import pandas as pd
    rows = []
    for key, entry in published.items():
        parts = key.split("_")
        date = next((p for p in parts if p[:2] == "20" and len(p) == 10), "")
        rows.append({"Sport": "NFL" if key.startswith("slate") else "MLB",
                     "Date": date, "Report": workbooks._label(key + ".xlsx"),
                     "File": key + ".xlsx", "Path": entry.get("source", key), "KB": 0})
    found = pd.DataFrame(rows).sort_values(["Date", "Report"], ascending=[False, True])

with st.sidebar:
    st.subheader("Report", anchor=False)
    sports = sorted(found["Sport"].unique())
    sport = (st.segmented_control("Sport", sports, default=sports[0], key="rep_sport",
                                  persist_state="session") or sports[0])
    scoped = found[found["Sport"] == sport]

    dates = sorted(scoped["Date"].dropna().unique(), reverse=True)
    date = st.selectbox("Date", dates, key="rep_date", persist_state="session")
    scoped = scoped[scoped["Date"] == date]

    names = sorted(scoped["Report"].unique())
    name = st.selectbox("Game" if sport == "MLB" else "Slate", names, key="rep_name",
                        persist_state="session")

row = scoped[scoped["Report"] == name]
if row.empty:
    st.warning("No report for that selection.")
    st.stop()
path = row.iloc[0]["Path"]

sheets = workbooks.sheet_names(path)
if not sheets:
    st.warning(f"Could not read the sheets for **{name}**.\n\n"
               + ("The workbook is not on this machine and it has not been published — "
                  "run `python -m dashboards.publish_reports` where the file lives."
                  if "xlsx" not in modes else
                  "The file may be open in Excel, or partially written."))
    st.stop()

entry = published.get(workbooks._key(path), {})
with st.container(horizontal=True):
    st.metric("Sheets", len(sheets), border=True)
    st.metric("Date", date, border=True)
    st.metric("Source", "workbook" if "xlsx" in modes and not entry else "published",
              border=True)

skipped = entry.get("skipped") or []
if skipped:
    st.caption(f"Not published: {', '.join(skipped)} — these carry embedded figures rather "
               f"than tables, and a picture does not survive the trip through pandas. Open "
               f"the workbook itself to see them.")

tabs = st.tabs(sheets)
for tab, sheet in zip(tabs, sheets):
    with tab:
        frame = workbooks.clean(workbooks.read_sheet(path, sheet))
        if frame is None or frame.empty:
            st.info(f"**{sheet}** has no tabular data — it is probably a figure sheet.")
            continue
        st.caption(f"{len(frame)} rows × {len(frame.columns)} columns")
        st.dataframe(frame, hide_index=True, width="stretch")

st.caption(f"Read from `{path}`")
