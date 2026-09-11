"""The drill-down popups themselves.

Kept out of the page scripts because the same dialog is reachable from several pages — the
board on Value, the lineup on Matchup, the slate scatter — and a popup that shows different
evidence depending on which chart opened it is worse than no popup.

`dashboards.drill` finds the evidence; this module only decides how it is laid out and in
what order. Nothing here computes a number.
"""

import pandas as pd
import streamlit as st

from dashboards import atbats, charts, data, drill, outcomes

#: Re-exported so the dialogs and the pages share one definition of "renderable".
arrow_safe = data.arrow_safe

#: Formats for columns that recur across the payload's frames. Applied where present so a
#: dialog never has to know which frame it is rendering.
COLUMN_FORMATS = {
    "Salary": st.column_config.NumberColumn(format="$%d"),
    "Composite": st.column_config.NumberColumn(format="%.1f"),
    "surplus": st.column_config.NumberColumn("Surplus", format="%+.1f"),
    "fpts": st.column_config.NumberColumn("DK pts", format="%.1f"),
    "own": st.column_config.NumberColumn("Owned", format="%.1f%%"),
    "hit": st.column_config.CheckboxColumn("Hit"),
    "OPS": st.column_config.NumberColumn(format="%.3f"),
    "Season OPS": st.column_config.NumberColumn(format="%.3f"),
    "Platoon OPS": st.column_config.NumberColumn(format="%.3f"),
    "Arsenal OPS": st.column_config.NumberColumn(format="%.3f"),
    "xwOBA": st.column_config.NumberColumn(format="%.3f"),
    "Similarity": st.column_config.ProgressColumn("Similarity", min_value=0, max_value=100,
                                                  format="%d"),
}


def _table(frame, **kwargs):
    """A frame with whatever of the shared formats apply to it."""
    frame = arrow_safe(frame)
    config = {k: v for k, v in COLUMN_FORMATS.items() if k in frame.columns}
    st.dataframe(frame, hide_index=True, column_config=config, **kwargs)


def _sections(sections, height=None):
    for label, frame in sections.items():
        st.markdown(f"**{label}**")
        _table(frame, **({"height": height} if height else {}))


@st.dialog("Player", width="large")
def hitter_dialog(payload, meta, name, team, history=None):
    """Everything the report used for one hitter, plus how he has actually scored.

    Ordered by what a reader reaches for first: what he is facing, then the club context the
    matchup was judged against, then his own inputs, then his record. The scored history is
    last because it is the only part that is *outcome* rather than input — putting it first
    invites reading a hot streak as the reason to play him.
    """
    st.subheader(f"{name} · {team}", anchor=False)
    st.caption(f"{meta['label']} · {meta['date']}")

    facing, arm = drill.opposing_starter(payload, meta, team)
    if facing:
        st.markdown(f"### Facing {facing}")
        tabs = st.tabs(list(arm))
        for tab, label in zip(tabs, arm):
            with tab:
                _table(arm[label])

    evidence = drill.hitter_evidence(payload, meta, name, team)
    if evidence:
        st.markdown("### His inputs")
        columns = st.columns(min(len(evidence), 3))
        for column, (label, frame) in zip(columns * 3, evidence.items()):
            with column:
                st.markdown(f"**{label}**")
                _table(frame, height=260)

    with st.expander("The at-bats behind the arsenal number", expanded=False):
        _arsenal_evidence(payload, meta, name, team)

    context = drill.team_context(payload, meta, team)
    if context:
        st.markdown(f"### {team} context")
        st.caption("The games the report judged most like tonight, and the form behind that "
                   "judgement.")
        tabs = st.tabs(list(context))
        for tab, label in zip(tabs, context):
            with tab:
                _table(context[label])

    if history is not None and not history.empty:
        scored = outcomes.player_history(history, name)
        if not scored.empty:
            st.markdown("### How he has actually scored")
            hits = int(scored["hit"].sum())
            with st.container(horizontal=True):
                st.metric("Scored slates", len(scored), border=True)
                st.metric("Hit rate", f"{hits / len(scored) * 100:.0f}%", border=True)
                st.metric("Mean DK points", f"{scored['fpts'].mean():.1f}", border=True)
            _table(outcomes.drill_frame(scored))
            st.caption("A record, not a projection. Over this many games it is mostly noise "
                       "— it is here to check the inputs above against reality, not to "
                       "forecast tonight.")


@st.dialog("Starting pitcher", width="large")
def pitcher_dialog(payload, meta, side):
    """A starter's comparables, his arsenal, and his log sorted by days of rest."""
    name = drill.starter_name(payload, side) or "Starter"
    team = meta["away"] if side == "away" else meta["home"]
    st.subheader(f"{name} · {team}", anchor=False)
    st.caption(f"{meta['label']} · {meta['date']}")

    log = drill.starter_game_log(payload, side)
    if not log.empty:
        st.markdown("### Game log")
        st.caption("Rest is the gap to his previous start. The oldest start's predecessor "
                   "is outside the cached window, so it has no rest value rather than a "
                   "guessed one.")
        _table(log)

        summary = drill.rest_summary(log)
        if not summary.empty:
            st.markdown("### By days of rest")
            _table(summary)
            st.caption(f"{int(summary['Starts'].sum())} starts across "
                       f"{len(summary)} rest lengths — small enough that a difference here "
                       "is a thing to look at, not a thing to bet on.")

    sections = drill.pitcher_evidence(payload, meta, side)
    if sections:
        st.markdown("### What he is compared against")
        tabs = st.tabs(list(sections))
        for tab, label in zip(tabs, sections):
            with tab:
                _table(sections[label])
                if label == "Comparable arms":
                    st.caption("The arms behind every 'vs arsenal' number on this slate. "
                               "A composite built on a thin or badly-matched comp set is "
                               "the failure mode this table exists to expose.")


def picker(frame, key, label="Open a player", name_column="Name"):
    """A single-row selection over a table, returning the chosen row or None.

    Streamlit has no click handler on a chart point that can carry a whole row, so the
    table under a chart is the click target. Selection is single-row on purpose: the dialog
    describes one player, and a multi-select that silently shows only the first is a lie
    about what was clicked.
    """
    if frame is None or frame.empty:
        return None
    frame = arrow_safe(frame)
    config = {k: v for k, v in COLUMN_FORMATS.items() if k in frame.columns}
    event = st.dataframe(frame, hide_index=True, on_select="rerun",
                         selection_mode="single-row", key=key, column_config=config)
    rows = (event.get("selection") or {}).get("rows") or []
    if not rows:
        return None
    return frame.iloc[rows[0]]


@st.dialog("Pitching staff", width="large")
def staff_dialog(payload, meta, side, unit, team):
    """A starter or a bullpen, opened from a conditions chart.

    Split by unit because the evidence differs: a starter has a start-by-start log and a
    comp set, a bullpen has per-arm usage and availability. Showing a bullpen the starter's
    panels would produce a dialog full of one man's numbers under a club's name.
    """
    from dashboards import bullpen as bullpen_module

    if unit == "Starter":
        name = drill.starter_name(payload, side) or f"{team} starter"
        st.subheader(f"{name} · {team}", anchor=False)
        st.caption(f"{meta['label']} · {meta['date']}")
        log = drill.starter_game_log(payload, side)
        if not log.empty:
            st.markdown("### Game log")
            _table(log)
            summary = drill.rest_summary(log)
            if not summary.empty:
                st.markdown("### By days of rest")
                _table(summary)
        sections = drill.pitcher_evidence(payload, meta, side)
        if sections:
            st.markdown("### What he is compared against")
            tabs = st.tabs(list(sections))
            for tab, label in zip(tabs, sections):
                with tab:
                    _table(sections[label])
        return

    st.subheader(f"{team} bullpen", anchor=False)
    st.caption(f"{meta['label']} · {meta['date']}")
    usage = data.bullpen_usage(payload, side)
    if not usage.empty:
        st.markdown("### Availability and recent usage")
        graded = bullpen_module.availability(usage)
        _table(graded)
        st.caption("An arm used on consecutive days appeared the next day 5.5% of the time "
                   "over 2,032 measured back-to-backs, against 26.0% on rest — so a "
                   "back-to-back is treated as unavailable regardless of pitch count.")
    batted = data.bullpen_batted(payload, side)
    if not batted.empty:
        st.markdown("### Batted-ball profile")
        _table(batted.drop(columns=[c for c in ["is_total"] if c in batted.columns]))
    notes = (payload.get("advanced_context") or {}).get("bullpen_notes") or []
    if notes:
        st.markdown("### Notes")
        for note in notes:
            st.markdown(f"- {note}")


def open_selection(event, frame, date, kind="hitter", history=None, name="point"):
    """Open the dialog for whatever point was clicked. Does nothing when nothing was.

    Call this **outside** the container the chart lives in: Streamlit renders a dialog where
    it is called from, and nesting it inside a bordered container clips the layout it builds.

    A selection carries only the identity fields, so the full row is recovered from the
    frame that drew the chart. Matching on the identity pair rather than the name alone —
    two clubs can field players who fold to the same name key on one slate.
    """
    picked = charts.selected(event, name)
    if not picked or frame is None or getattr(frame, "empty", True):
        return False
    point = picked[0]

    if kind == "staff":
        who, club = str(point.get("name", "")), str(point.get("team", ""))
        match = frame[(frame["name"].astype(str) == who)
                      & (frame["team"].astype(str) == club)]
        if match.empty:
            return False
        row = match.iloc[0]
        payload, meta = drill.payload_for(date, club)
        if payload is None:
            st.warning(f"No cached game found for {club} on {date}.")
            return False
        staff_dialog(payload, meta, row["side"], row["unit"], club)
        return True

    who, club = str(point.get("Name", "")), str(point.get("Team", ""))
    match = frame[(frame["Name"].astype(str) == who)
                  & (frame["Team"].astype(str) == club)]
    if match.empty:
        return False
    payload, meta = drill.payload_for(date, club)
    if payload is None:
        st.warning(f"No cached game found for {club} on {date}.")
        return False
    hitter_dialog(payload, meta, who, club, history)
    return True


def chart_with_drilldown(chart, frame, date, key, kind="hitter", history=None):
    """Render a clickable chart and return whether a dialog should follow.

    Returns the event rather than opening the dialog itself, because the dialog has to be
    opened outside the caller's container — see `open_selection`.
    """
    if chart is None:
        return None
    # The chart also carries a scale-bound `zoom` interval. Without selection_mode,
    # Streamlit listens to every parameter and a wheel zoom reruns the page, rebuilding the
    # chart at its home domain. Only the point selection is server-side; pan/zoom stays in
    # Vega in the browser and the click-through continues to rerun normally.
    return st.altair_chart(charts.selectable(chart, kind), on_select="rerun",
                           selection_mode="point", key=key)


def _arsenal_evidence(payload, meta, name, team):
    """Every plate appearance the arsenal line was built from, and the filter that chose it.

    Behind an expander because it loads a season of pitch data (~2.6s cold, cached after)
    and imports the pipeline — neither is worth doing for a reader who only wanted the
    summary. Verified to reproduce the published `Arsenal OPS` and at-bat count exactly on
    15 of 15 hitters across two lineups.
    """
    try:
        at_bats, context = atbats.arsenal_at_bats(payload, meta, name, team)
    except Exception as error:                      # noqa: BLE001 - surfaced, not swallowed
        st.caption(f"Could not rebuild the at-bats: {error}")
        return
    if context.get("unavailable"):
        # A different claim from "no at-bats found", and worth keeping apart: one says the
        # sample is empty, the other says nothing was looked at.
        st.info(context["unavailable"], icon=":material/info:")
        return
    if not context or at_bats.empty:
        st.caption("No measured plate appearances against this arsenal.")
        return

    basis = " · ".join(f"{pitch} {weight * 100:.0f}%" for pitch, weight in context["weights"])
    with st.container(horizontal=True):
        st.metric("Plate appearances", context["plate_appearances"], border=True)
        st.metric("At-bats", context["at_bats"], border=True)
        st.metric("OPS", f"{context['ops']:.3f}", border=True)
        st.metric("xwOBA", f"{context['xwoba']:.3f}", border=True)
    st.caption(
        f"Against {context['starter']} ({context['hand']}HP). Weighted on {basis}, "
        f"matched by **{context['shape_filter']}** — the same filter the report used, so "
        "these are the plate appearances the number was computed from, not a re-derivation.")

    traits = [t for t in context.get("traits") or [] if t.get("Velo") is not None]
    if traits:
        st.markdown("**The pitch shapes matched**")
        _table(pd.DataFrame([{
            "Pitch": t["Pitch"], "Usage %": t.get("Usage"),
            "Velo band": f"{t['Velo'] - 2.5:.1f}-{t['Velo'] + 2.5:.1f} mph",
            "Spin band": (f"{t['Spin'] - 300:.0f}-{t['Spin'] + 300:.0f} rpm"
                          if t.get("Spin") is not None else ""),
        } for t in traits]))

    split = atbats.by_pitch_type(at_bats)
    if not split.empty:
        st.markdown("**By pitch** — the line is a weighted blend, so this is where you find "
                    "out if one pitch is carrying it")
        _table(split, height=200)

    outcomes_table = atbats.outcome_summary(at_bats)
    if not outcomes_table.empty:
        st.markdown("**What they produced**")
        _table(outcomes_table, height=200)

    st.markdown(f"**All {len(at_bats)} plate appearances**")
    _table(at_bats, height=320)
