"""Altair builders for the football pages.

Same rules as `dashboards.charts`, and the important one is what is *not* set here:

**Axis and legend colours are left to Streamlit.** It themes Vega charts to match the active
app theme and it knows which theme is live; hard-coded ink does not, and pinning a
light-surface palette makes every legend label invisible in dark mode. Only the *data*
colours are ours, because those carry meaning.

The reference rule on every scatter is the **median of the points actually drawn**, not a
league line. Without it a raw axis is a field of absolute numbers with nothing to sit
against; pretending it were a league constant would be worse, because it moves with the
filter and the caption has to say so.
"""

import altair as alt
import pandas as pd

# Reference-rule grey. Mid-tone on purpose: it is chrome, and it has to stay visible on both
# the light and the dark app surface, which Streamlit does not theme for mark colours.
RULE_GREY = "#9aa0a6"

# Alignment is the most stable thing in this data (0.82-0.95 year over year), so it gets the
# categorical palette rather than team identity, which would repeat across 32 clubs.
ALIGNMENT_COLORS = {"Slot": "#2a78d6", "Wide": "#eb6834", "Inline": "#1baf7a"}
ALIGNMENT_ORDER = ["Wide", "Slot", "Inline"]

BASIS_COLORS = {
    "grade": "#2a78d6",
    "grade (moved)": "#eda100",
    "draft capital": "#1baf7a",
    "replacement": "#9aa0a6",
}
BASIS_ORDER = ["grade", "grade (moved)", "draft capital", "replacement"]

LABEL_LIMIT = 5

# Marks are drawn **without a stroke**. A light outline separates overlapping points, but at
# the sizes these charts use it eats most of a small mark and the colour -- which is the
# channel carrying alignment, position or basis -- stops being readable. Opacity does the
# overlap job instead, and costs nothing that matters.
MARK = {"filled": True, "opacity": 0.8}


def round_display(frame, small=3, large=2):
    """Round float columns for display: `small` decimals for rates, `large` for the rest.

    Split on magnitude rather than on a column list, so a new column is formatted sensibly
    without anyone remembering to add it. A rate that lives under 1.0 needs the extra digit
    to be distinguishable at all -- TPRR runs 0.15 to 0.30 and rounds to a single value at
    two decimals for half the league.
    """
    if frame is None or frame.empty:
        return frame
    out = frame.copy()
    for column in out.columns:
        if not pd.api.types.is_float_dtype(out[column]):
            continue
        peak = out[column].abs().max()
        places = small if pd.notna(peak) and peak < 1.0 else large
        out[column] = out[column].round(places)
    return out


def _styled(chart, height=420):
    """Size the chart and otherwise leave its chrome to Streamlit. See the module docstring.

    Takes the chart, never the frame: these are layered, and passing a composed LayerChart
    back through `alt.Chart(...)` treats it as *data* and fails with the memorably unhelpful
    "`Data` has no parameter named 'layer'".
    """
    return chart.properties(height=height)


def _alignment_scale():
    return alt.Scale(domain=ALIGNMENT_ORDER,
                     range=[ALIGNMENT_COLORS[a] for a in ALIGNMENT_ORDER])


def _rule(value, axis, domain=None):
    """A dashed reference at `value` on the named axis.

    The scale is attached only when a domain is actually given. `alt.Scale(domain=None)` is
    not "no domain" -- Altair validates it and refuses outright with `'None' is an invalid
    value for domain`, which surfaces as a redacted "this app has encountered an error" and
    tells you nothing about which chart did it.
    """
    frame = pd.DataFrame({axis: [value]})
    channel = alt.Y if axis == "y" else alt.X
    encoding = (channel(f"{axis}:Q", scale=alt.Scale(domain=domain)) if domain is not None
                else channel(f"{axis}:Q"))
    return (alt.Chart(frame)
            .mark_rule(color=RULE_GREY, strokeDash=[4, 4], strokeWidth=1, opacity=0.7)
            .encode(**{axis: encoding}))


def _labels(data, x_field, y_field, deviation, color_field, scale, limit=LABEL_LIMIT):
    """Name the few points furthest from the middle. More than a handful is noise."""
    if data.empty or limit <= 0:
        return None
    frame = data.assign(_dev=deviation.abs())
    top = frame.nlargest(min(limit, len(frame)), "_dev")
    if top.empty:
        return None
    return (alt.Chart(top)
            .mark_text(align="left", dx=8, dy=-6, fontSize=11)
            .encode(x=alt.X(x_field), y=alt.Y(y_field),
                    text=alt.Text("Name:N"),
                    color=alt.Color(color_field, scale=scale, legend=None)))


def usage_scatter(frame, x, y, size=None, color="Alignment", height=440, labels=True,
                  extra_tooltip=()):
    """The general two-axis explorer behind the receiver and rushing pages.

    Axes are chosen by the reader rather than fixed, because "is this a good spot" is
    several different questions in football too -- volume, depth, efficiency and alignment
    are four separate readings of the same player and a fixed pair silently picks one.
    """
    # De-duplicated: `color="Team"` would otherwise name Team twice, and selecting a column
    # twice hands Vega a frame with two identically-named columns -- `DuplicateError:
    # Expected unique column names, got 'Team' 2 times`, surfaced as a redacted Streamlit
    # error that names neither the chart nor the column.
    needed = list(dict.fromkeys(
        [c for c in (x, y, size, color, "Name", "Team") if c] + list(extra_tooltip)))
    plotted = frame.dropna(subset=[c for c in (x, y) if c in frame.columns])
    plotted = plotted[[c for c in needed if c in plotted.columns]].copy()
    if plotted.empty:
        return None

    median = float(plotted[y].median())
    scale = _alignment_scale() if color == "Alignment" else alt.Scale(scheme="tableau10")

    tooltip = [alt.Tooltip("Name:N", title="Player"),
               alt.Tooltip("Team:N", title="Team")]
    for column in dict.fromkeys([x, y, size, color, *extra_tooltip]):
        if column and column in plotted.columns and column not in ("Name", "Team"):
            kind = "Q" if pd.api.types.is_numeric_dtype(plotted[column]) else "N"
            spec = {"format": ".2f"} if kind == "Q" else {}
            tooltip.append(alt.Tooltip(f"{column}:{kind}", title=column, **spec))

    encoding = {
        "x": alt.X(f"{x}:Q", title=x, scale=alt.Scale(zero=False)),
        "y": alt.Y(f"{y}:Q", title=y, scale=alt.Scale(zero=False)),
        "color": alt.Color(f"{color}:N", title=color, scale=scale),
        "tooltip": tooltip,
    }
    if size and size in plotted.columns:
        encoding["size"] = alt.Size(f"{size}:Q", title=size,
                                    scale=alt.Scale(range=[60, 460]))

    points = (alt.Chart(plotted)
              .mark_point(**MARK)
              .encode(**encoding))
    layered = _rule(median, "y") + points
    if labels:
        text = _labels(plotted, f"{x}:Q", f"{y}:Q", plotted[y] - median,
                       f"{color}:N", scale)
        if text is not None:
            layered = layered + text
    return _styled(layered, height)


def share_bars(frame, team, height=360):
    """Who a club throws to, as a share of its qualifying receivers' targets."""
    one = frame[frame["Team"] == team].copy()
    if one.empty:
        return None
    one = one.nlargest(min(10, len(one)), "Target share")
    one["Share%"] = one["Target share"] * 100
    return _styled(
        alt.Chart(one).mark_bar(cornerRadiusEnd=3).encode(
            y=alt.Y("Name:N", sort="-x", title=None),
            x=alt.X("Share%:Q", title="Share of team targets (%)"),
            color=alt.Color("Pos:N", title="Position",
                            scale=alt.Scale(scheme="tableau10")),
            tooltip=[alt.Tooltip("Name:N", title="Player"),
                     alt.Tooltip("Pos:N", title="Pos"),
                     alt.Tooltip("Targets:Q", title="Targets", format=".0f"),
                     alt.Tooltip("Share%:Q", title="Share", format=".1f"),
                     alt.Tooltip("aDOT:Q", title="aDOT", format=".1f")]),
        height)


# Pixels per bar. Below about 18 Vega starts dropping axis labels to fit, which is exactly
# what makes a ranked chart useless -- you can see the shape but not read off who is where.
ROW_HEIGHT = 22


def bar_height(rows, minimum=240, maximum=1400):
    """Tall enough that every row keeps its label. See `ROW_HEIGHT`."""
    return int(min(max(rows * ROW_HEIGHT + 50, minimum), maximum))


def tendency_bars(frame, value, title, height=None, descending=True):
    """Every defense on one measure, ranked. The workhorse of the coverage page."""
    plotted = frame.dropna(subset=[value]).copy()
    if plotted.empty:
        return None
    height = height or bar_height(len(plotted))
    order = "-x" if descending else "x"
    league = float(plotted[value].mean())
    bars = (alt.Chart(plotted).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("Team:N", sort=order, title=None),
        x=alt.X(f"{value}:Q", title=title),
        color=alt.Color(f"{value}:Q", title=title,
                        scale=alt.Scale(scheme="blueorange")),
        tooltip=[alt.Tooltip("Team:N", title="Team"),
                 alt.Tooltip(f"{value}:Q", title=title, format=".2f")]))
    return _styled(bars + _rule(league, "x"), height)


def line_strength(unit, height=None):
    """Projected offensive lines, ranked, coloured by how much is actually graded."""
    plotted = unit.dropna(subset=["Proj grade"]).copy()
    if plotted.empty:
        return None
    height = height or bar_height(len(plotted))
    plotted["Graded"] = plotted["Returning"] + plotted["New"]
    league = float(plotted["Proj grade"].mean())
    bars = (alt.Chart(plotted).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("Team:N", sort="-x", title=None),
        x=alt.X("Proj grade:Q", title="Projected unit grade",
                scale=alt.Scale(zero=False)),
        color=alt.Color("Graded:Q", title="Starters with a grade",
                        scale=alt.Scale(scheme="viridis")),
        tooltip=[alt.Tooltip("Team:N", title="Team"),
                 alt.Tooltip("Proj grade:Q", title="Proj grade", format=".1f"),
                 alt.Tooltip("Proj pass blk:Q", title="Pass block", format=".1f"),
                 alt.Tooltip("Proj run blk:Q", title="Run block", format=".1f"),
                 alt.Tooltip("Returning:Q", title="Returning"),
                 alt.Tooltip("New:Q", title="New"),
                 alt.Tooltip("Rookies:Q", title="Rookies")]))
    return _styled(bars + _rule(league, "x"), height)
