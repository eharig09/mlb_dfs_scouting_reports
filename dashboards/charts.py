"""Altair scatter builders shared by the dashboard pages.

Two constraints from the project's visualisation rules drive every choice here, and both
are specific to *scatter*:

* **Scatter is an all-pairs form, so the categorical palette caps at three series.** Every
  point can sit beside every other one, unlike a bar chart where only neighbours touch. Two
  clubs in a matchup view is comfortably inside that; a whole slate is thirty, so those
  views may not colour by team at all and encode team on hover instead.
* **`Signal` is ordered, not categorical** — Priority beats Watch beats Neutral beats Fade.
  It is drawn with the reserved status palette rather than series hues, and the legend
  carries the words, because a status colour must never be the only thing saying what a
  point is.

Reference lines earn their place: a diagonal on a same-unit pair turns "where is this dot"
into "which side of parity is he on", which is the question those panels exist to answer.
"""

import altair as alt
import numpy as np
import pandas as pd

from dashboards import salaries, scales

# Validated categorical pair — the same two the MLB report's figures use, so a club keeps
# its colour between the workbook and the dashboard.
TEAM_SLOTS = ["#2a78d6", "#eb6834"]

# Reserved status palette. Never reused for a series.
SIGNAL_COLORS = {
    "Priority": "#0ca30c",
    "Watch": "#fab219",
    "Neutral": "#9aa0a6",
    "Fade": "#d03b3b",
}
SIGNAL_ORDER = ["Priority", "Watch", "Neutral", "Fade"]

# Reference-rule grey. Mid-tone on purpose: it has to stay visible on both the light and
# the dark app surface, since a rule is chrome and Streamlit does not theme mark colours.
RULE_GREY = "#9aa0a6"

HITTER_OPS_TOOLTIP = [
    alt.Tooltip("Season OPS:Q", title="OPS", format=".3f"),
    alt.Tooltip("L28 OPS:Q", title="L28 OPS", format=".3f"),
    alt.Tooltip("Platoon OPS:Q", title="Platoon OPS", format=".3f"),
    alt.Tooltip("Arsenal OPS:Q", title="Arsenal OPS", format=".3f"),
]

TOOLTIP = [
    alt.Tooltip("Name:N", title="Player"),
    alt.Tooltip("Team:N", title="Team"),
    alt.Tooltip("Bats:N", title="Bats"),
    alt.Tooltip("Composite:Q", title="Composite", format=".1f"),
    alt.Tooltip("Signal:N", title="Signal"),
] + HITTER_OPS_TOOLTIP + [
    alt.Tooltip("Platoon AB:Q", title="Platoon AB", format=".0f"),
]


def _styled(chart, height=380, zoom=False):
    """Size the chart and otherwise leave its chrome to Streamlit.

    **Axis and legend colours are deliberately not set here.** The first version pinned them
    to a light-surface palette (`#0b0b0b` ink, `#7a7975` muted), which rendered the legend
    labels invisible the moment the app was viewed in dark mode. Streamlit themes Vega
    charts to match the active app theme, and it knows which theme is active; hard-coded ink
    does not. Only the *data* colours are ours, because those carry meaning — team identity
    and the reserved status scale — and must not drift with the theme.

    `zoom` attaches the pan/zoom binding, and only the *scatters* ask for it. A ranked bar
    chart has one continuous axis and a categorical one; panning it hides rows without
    telling anyone, which is a worse chart, not a more interactive one.

    Takes the chart, never the frame: these are layered (points over a reference rule), and
    passing a composed LayerChart back through `alt.Chart(...)` treats it as *data*, failing
    with the memorably unhelpful "`Data` has no parameter named 'layer'".
    """
    return _zoomable(chart).properties(height=height) if zoom \
        else chart.properties(height=height)


def _zoomable(chart):
    """Bind pan and zoom to the chart's scales, or hand it back untouched if it will not take.

    The parameter goes on the layer that draws the *marks*, for the same reason the click
    selection does: a scale binding on the reference-rule layer binds the rule's own scale,
    and the points then sit still while the dashed line slides around underneath them.
    """
    if chart is None:
        return None
    try:
        layers = getattr(chart, "layer", None)
        if layers:
            index = _marks_layer(layers)
            layers[index] = layers[index].add_params(scales.binding())
            return chart
        return chart.add_params(scales.binding())
    except Exception:
        # A chart shape that will not take a param is still worth showing.
        return chart


def _ready(frame, columns):
    """Rows with all of `columns` present and non-null, or None if the panel cannot be drawn.

    `frame.dropna(subset=[...])` raises KeyError when the column is missing entirely — which
    is exactly the shape an absent payload section produces — so a chart would crash rather
    than politely decline. Every builder goes through here.

    Also the one place the page's anchored axis bounds are re-attached, so a builder gets
    them without every builder having to remember to.
    """
    if frame is None or getattr(frame, "empty", True):
        return None
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        return None
    data = frame.dropna(subset=list(columns))
    return scales.carry(data, frame) if not data.empty else None


def _qscale(data, field, zero=False, **kw):
    """The scale for one quantitative axis: the page's anchored bounds, or the old default.

    Focus bounds are stable across refreshes and clamp extremes to a visible boundary; Full
    bounds cover the unfiltered board. A frame nobody anchored falls through to the old
    `zero=False` behavior unchanged.
    """
    pinned = scales.of(data, field)
    if pinned:
        # A focus window leaves room for normal differences; clamping keeps a genuine
        # extreme visible on the boundary instead of silently dropping the mark.
        return alt.Scale(domain=list(pinned), clamp=True, **kw)
    return alt.Scale(zero=zero, **kw)


def _same_unit_domain(data, fields, floor=None):
    """One domain for a set of same-unit axes — the parity panels' diagonal needs it.

    Falls back to the frame's own range when the page has not anchored it, which is what the
    parity charts computed for themselves before there were anchors at all.
    """
    pinned = scales.across(data, fields)
    if pinned:
        return [pinned[0], pinned[1]]
    low = float(min(data[f].min() for f in fields)) - 0.05
    high = float(max(data[f].max() for f in fields)) + 0.05
    return [low if floor is None else max(floor, low), high]


def _team_scale(teams):
    ordered = sorted(t for t in teams if isinstance(t, str) and t)
    return alt.Scale(domain=ordered, range=TEAM_SLOTS[: max(len(ordered), 1)])


def _signal_scale():
    return alt.Scale(domain=SIGNAL_ORDER,
                     range=[SIGNAL_COLORS[s] for s in SIGNAL_ORDER])


#: How many points a scatter may name *at rest*. Direct labels work because they are sparing:
#: a name beside every dot is chaos and goes unread, and on an eighteen-hitter panel it would
#: also collide with itself. The extremes are the only ones a reader is going to ask about
#: before they have asked a narrower question.
LABEL_LIMIT = 4

#: How many it may name once the reader has zoomed in. The constraint on labels was never
#: "four is the right number" — it was that a name needs room beside its own point, and
#: zooming is the reader creating that room. So the second tier is generous: the cap is only
#: there to keep the spec from carrying a thousand text marks that will never be drawn.
ZOOM_LABEL_LIMIT = 60

#: Share of the x axis the reader has to zoom inside before the second tier appears. Just
#: over half, so it takes a deliberate zoom rather than an accidental scroll, and so a panel
#: that opens two clubs' worth of bats is already showing them.
ZOOM_LABEL_SPAN = 0.55


#: Name particles that belong to the surname, and suffixes that are not part of it at all.
#: Without both lists a naive "last token" gives `Cruz` for Elly De La Cruz and `Jr.` for
#: Ronald Acuna Jr. -- the second is not a shortened name, it is the wrong word entirely.
NAME_PARTICLES = {"de", "del", "la", "las", "los", "van", "von", "der", "den",
                  "da", "di", "do", "dos", "st", "mc"}
NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _short_name(value):
    """`Aaron Judge` -> `Judge`, `Elly De La Cruz` -> `De La Cruz`, `X Acuna Jr.` -> `Acuna`.

    A label has to fit beside its own point, so it is the surname and nothing else.
    """
    parts = [p for p in str(value or "").split()
             if p.strip(".").lower() not in NAME_SUFFIXES]
    if not parts:
        return ""
    start = len(parts) - 1
    while start > 1 and parts[start - 1].strip(".").lower() in NAME_PARTICLES:
        start -= 1
    return " ".join(parts[start:])


def _text_layer(labelled, x_field, y_field, color_field, scale, *, size, weight,
                opacity=None):
    """One tier of direct labels.

    Labels are coloured on the same scale as their marks rather than given ink of their own.
    Two reasons: it ties the word to the dot without a leader line, and it keeps the rule
    that only *data* colours are ours -- theme ink is Streamlit's to set, and a hard-coded
    label colour is the exact bug that made the legend invisible in dark mode.
    """
    encode = dict(x=alt.X(x_field), y=alt.Y(y_field), text=alt.Text("_label:N"))
    if color_field:
        encode["color"] = alt.Color(color_field, scale=scale, legend=None)
    if opacity is not None:
        encode["opacity"] = opacity
    return (alt.Chart(labelled)
            .mark_text(align="left", baseline="middle", dx=11, dy=-9, fontSize=size,
                       fontWeight=weight)
            .encode(**encode))


def _label_layers(data, x_field, y_field, deviation, color_field=None, scale=alt.Undefined,
                  limit=LABEL_LIMIT, name_column="Name", team=False):
    """Name the extremes always, and everyone else once the reader zooms in.

    Two tiers rather than one, because the old cap of four was answering the wrong question.
    Four is not the number of names a reader wants; it is the number that fits when the whole
    slate is on screen. Zooming *is* the reader asking for the crowded corner to be readable,
    so the second tier fades in when the visible x span drops below `ZOOM_LABEL_SPAN` of the
    axis and fades back out on the double-click that resets it.

    `x_field` arrives as `"Field:Q"`; the predicate needs the bare field name, because that is
    the key the zoom binding reports its extent under.
    """
    if data is None or data.empty or name_column not in data.columns:
        return []
    ranked = data.assign(_dev=deviation.abs()).sort_values("_dev", ascending=False)
    ranked = ranked[ranked["_dev"] > 0]
    if ranked.empty:
        return []

    def labelled(frame):
        text = frame[name_column].map(_short_name)
        if team and "Team" in frame.columns:
            text = text + " " + frame["Team"].astype(str)
        return frame.assign(_label=text)

    layers = [_text_layer(labelled(ranked.head(min(limit, len(ranked)))), x_field, y_field,
                          color_field, scale, size=11, weight=600)]

    field = str(x_field).split(":")[0]
    rest = ranked.iloc[limit:ZOOM_LABEL_LIMIT]
    span = _axis_span(data, field)
    if not rest.empty and span:
        # Lighter than the always-on tier: these are the reader's own follow-up question, and
        # they should not out-shout the four the panel opened with.
        condition = {"condition": {"test": scales.zoomed_in(field, span * ZOOM_LABEL_SPAN),
                                   "value": 0.95}, "value": 0}
        layers.append(_text_layer(labelled(rest), x_field, y_field, color_field, scale,
                                  size=10, weight=500, opacity=condition))
    return layers


def _axis_span(data, field):
    """How wide the x axis is, so "zoomed in" can be a fraction of it rather than a constant.

    The anchored bounds when the page set them -- which is the span the reader actually sees
    at rest -- and the data's own range otherwise.
    """
    pinned = scales.of(data, field)
    if pinned:
        return pinned[1] - pinned[0]
    if field not in getattr(data, "columns", []):
        return None
    values = pd.to_numeric(data[field], errors="coerce").dropna()
    span = float(values.max() - values.min()) if not values.empty else 0.0
    return span or None


def _over(base, layers):
    """Stack `layers` on top of `base`, skipping the ones a builder decided against."""
    for layer in layers:
        if layer is not None:
            base = base + layer
    return base


#: The two ways to read a split panel. `context` is the decision view -- the raw split
#: against tonight's composite; `parity` is the same-unit pair whose diagonal shows the
#: spread against his own season line.
SPLIT_VIEWS = {
    "split vs composite": "context",
    "OPS against OPS": "parity",
}


def _price_scale(values):
    """Shape scale over the price tiers actually present, cheap to expensive."""
    present = [tier for tier in list(salaries.TIER_LABELS) + [salaries.UNPRICED]
               if tier in set(values)]
    return present, alt.Scale(domain=present,
                              range=[salaries.TIER_SHAPES[t] for t in present])


def _split_context_scatter(data, value_column, value_title, size_column, size_title,
                           tooltip, height):
    """Composite on x, the raw split OPS on y, price tier as shape.

    The number on the y axis is the one a decision gets made on, so it is the split itself
    rather than its distance from his season line — a deviation answers "is this unusual for
    him", which is a different and later question than "how well does he hit this".

    The reference is the **median of the unfiltered reference board** when the page supplied
    one, which makes a filter comparison readable without moving the rule underneath it. An
    unanchored chart still uses the points drawn.

    Shape is price tier. It costs nothing already spent — colour is the club, size is the
    sample — and it is the one thing a reader needs that the axes do not carry.
    """
    plotted = scales.carry(salaries.price_tier(data), data)
    median = scales.reference(plotted, value_column)
    median = float(median if median is not None else plotted[value_column].median())
    pinned = scales.of(plotted, value_column)
    domain = (list(pinned) if pinned else
              [max(0.0, float(plotted[value_column].min()) - 0.05),
               float(plotted[value_column].max()) + 0.05])

    rule = (alt.Chart(pd.DataFrame({"y": [median]}))
            .mark_rule(color=RULE_GREY, strokeDash=[4, 4], strokeWidth=1, opacity=0.7)
            .encode(y=alt.Y("y:Q", scale=alt.Scale(domain=domain, clamp=True))))

    tiers, shape_scale = _price_scale(plotted["Price"])
    points = (alt.Chart(plotted)
              # `mark_circle` has no shape channel; a filled point does. The size floor is
              # lifted from 40 because a triangle and a diamond stop being distinguishable
              # long before a circle stops being visible.
              .mark_point(filled=True, opacity=0.85, stroke="#fcfcfb", strokeWidth=1.2)
              .encode(
                  x=alt.X("Composite:Q", title="Composite matchup score",
                          scale=_qscale(plotted, "Composite")),
                  y=alt.Y(f"{value_column}:Q", title=value_title,
                          scale=alt.Scale(domain=domain, clamp=True)),
                  size=_size_channel(plotted, size_column, (90, 520), size_title)[0],
                  color=alt.Color("Team:N", title="Team",
                                  scale=_team_scale(plotted["Team"])),
                  shape=alt.Shape("Price:N", title="Price", scale=shape_scale,
                                  sort=tiers),
                  tooltip=tooltip + [alt.Tooltip("Salary:Q", title="Salary", format=",.0f")]
                  if "Salary" in plotted.columns else tooltip))
    labels = _label_layers(plotted, "Composite:Q", f"{value_column}:Q",
                           plotted[value_column] - median,
                           "Team:N", _team_scale(plotted["Team"]))
    return _styled(_over(rule + points, labels), height, zoom=True)


def platoon_scatter(frame, height=400, view="context"):
    """How he handles the hand he draws tonight, against how he hits in general.

    `view="parity"` is the original: season OPS against the split, same unit on both axes so
    the diagonal is the reading. `view="context"` keeps that reading as a zero rule and
    spends the x axis on the composite instead, which is the only way to get form and
    matchup onto one panel without a second y scale.

    Size is the platoon sample either way, because a 40-point gap on 30 at-bats and on 300
    are not the same claim.
    """
    data = _ready(frame, ["Season OPS", "Platoon OPS"])
    if data is None:
        return None
    if view == "context":
        if "Composite" not in data.columns or data["Composite"].isna().all():
            view = "parity"
        else:
            return _split_context_scatter(
                data.dropna(subset=["Composite"]), "Platoon OPS",
                "OPS vs tonight's hand", "Platoon AB", "Platoon AB", TOOLTIP, height)

    domain = _same_unit_domain(data, ["Season OPS", "Platoon OPS"], floor=0.0)

    parity = (alt.Chart(pd.DataFrame({"x": domain, "y": domain}))
              .mark_line(color=RULE_GREY, strokeDash=[4, 4], strokeWidth=1, opacity=0.7)
              .encode(x=alt.X("x:Q", scale=alt.Scale(domain=domain, clamp=True)),
                      y=alt.Y("y:Q", scale=alt.Scale(domain=domain, clamp=True))))

    points = (alt.Chart(data).mark_circle(opacity=0.85, stroke="#fcfcfb", strokeWidth=1.5)
              .encode(
                  x=alt.X("Season OPS:Q", title="Season OPS",
                          scale=alt.Scale(domain=domain, clamp=True)),
                  y=alt.Y("Platoon OPS:Q", title="OPS vs tonight's hand",
                          scale=alt.Scale(domain=domain, clamp=True)),
                  size=_size_channel(data, "Platoon AB", (40, 520), "Platoon AB")[0],
                  color=alt.Color("Team:N", title="Team", scale=_team_scale(data["Team"])),
                  tooltip=TOOLTIP))
    labels = _label_layers(data, "Season OPS:Q", "Platoon OPS:Q",
                           data["Platoon OPS"] - data["Season OPS"],
                           "Team:N", _team_scale(data["Team"]))
    return _styled(_over(parity + points, labels), height, zoom=True)


def form_scatter(frame, height=400):
    """Tonight's matchup strength against how he has actually been hitting.

    The quadrants are the point: strong matchup and hot is a straightforward play, strong
    matchup and cold is where the model disagrees with the eye test, and that disagreement
    is the only thing on this chart worth arguing about.
    """
    data = _ready(frame, ["Composite", "Off L28"])
    if data is None:
        return None
    league = 100.0
    rule = (alt.Chart(pd.DataFrame({"y": [league]}))
            .mark_rule(color=RULE_GREY, strokeDash=[4, 4], strokeWidth=1, opacity=0.7)
            .encode(y="y:Q"))
    zero = (alt.Chart(pd.DataFrame({"x": [0.0]}))
            .mark_rule(color=RULE_GREY, strokeDash=[4, 4], strokeWidth=1, opacity=0.7)
            .encode(x="x:Q"))
    points = (alt.Chart(data).mark_circle(opacity=0.85, stroke="#fcfcfb", strokeWidth=1.5)
              .encode(
                  x=alt.X("Composite:Q", title="Composite matchup score",
                          scale=_qscale(data, "Composite")),
                  y=alt.Y("Off L28:Q", title="Offense index, last 28 days (100 = league)",
                          scale=_qscale(data, "Off L28")),
                  size=_size_channel(data, "Season AB", (40, 520), "Season AB")[0],
                  color=alt.Color("Signal:N", title="Signal", scale=_signal_scale(),
                                  sort=SIGNAL_ORDER),
                  tooltip=TOOLTIP))
    labels = _label_layers(data, "Composite:Q", "Off L28:Q", data["Off L28"] - league)
    return _styled(_over(rule + zero + points, labels), height, zoom=True)


def slate_scatter(frame, height=460):
    """Every hitter on the slate at once.

    Thirty clubs is far past what colour can carry on an all-pairs form, so team lives in
    the tooltip and the hue is spent on `Signal` — which is ordered and reserved, and the
    only thing here a reader would actually filter on.
    """
    data = _ready(frame, ["Composite", "Season OPS"])
    if data is None:
        return None
    points = (alt.Chart(data).mark_circle(opacity=0.75, stroke="#fcfcfb", strokeWidth=1)
              .encode(
                  x=alt.X("Composite:Q", title="Composite matchup score",
                          scale=_qscale(data, "Composite")),
                  y=alt.Y("Season OPS:Q", title="Season OPS",
                          scale=_qscale(data, "Season OPS")),
                  size=_size_channel(data, "Platoon AB", (30, 400), "Platoon AB")[0],
                  color=alt.Color("Signal:N", title="Signal", scale=_signal_scale(),
                                  sort=SIGNAL_ORDER),
                  tooltip=TOOLTIP + [alt.Tooltip("game:N", title="Game")]))
    labels = _label_layers(data, "Composite:Q", "Season OPS:Q",
                           data["Composite"] - data["Composite"].median(), team=True)
    return _styled(_over(points, labels), height, zoom=True)


def arsenal_scatter(frame, height=400, view="context"):
    """How he hits this arm's pitch mix against how he hits everyone's.

    Below the line means the pitches he is about to see are ones he has handled worse than
    his own baseline — the arsenal-fit read, isolated from whether he is simply good.

    **Conditions cannot be the x axis here, which is worth saying once.** Park, weather and
    defence are properties of the *game*: every hitter on this panel is in the same park in
    the same air, so a conditions term varies only by batter handedness and would draw two
    vertical stripes rather than a spread. `dashboards.environment` also deliberately
    refuses to collapse those three into one score. The composite is the axis that actually
    separates these hitters; conditions belong on the Conditions page, where they vary
    across games.
    """
    data = _ready(frame, ["Season OPS", "Arsenal OPS", "Arsenal AB"])
    if data is None:
        return None
    data = data[data["Arsenal AB"].fillna(0) > 0]
    if data.empty:
        return None
    arsenal_tooltip = TOOLTIP + [alt.Tooltip("Arsenal AB:Q", title="Arsenal AB",
                                             format=".0f")]
    if view == "context" and "Composite" in data.columns and data["Composite"].notna().any():
        return _split_context_scatter(
            data.dropna(subset=["Composite"]), "Arsenal OPS",
            "OPS vs this arsenal", "Arsenal AB", "Arsenal AB",
            arsenal_tooltip, height)
    domain = _same_unit_domain(data, ["Season OPS", "Arsenal OPS"], floor=0.0)
    parity = (alt.Chart(pd.DataFrame({"x": domain, "y": domain}))
              .mark_line(color=RULE_GREY, strokeDash=[4, 4], strokeWidth=1, opacity=0.7)
              .encode(x=alt.X("x:Q", scale=alt.Scale(domain=domain, clamp=True)),
                      y=alt.Y("y:Q", scale=alt.Scale(domain=domain, clamp=True))))
    points = (alt.Chart(data).mark_circle(opacity=0.85, stroke="#fcfcfb", strokeWidth=1.5)
              .encode(
                  x=alt.X("Season OPS:Q", title="Season OPS",
                          scale=alt.Scale(domain=domain, clamp=True)),
                  y=alt.Y("Arsenal OPS:Q", title="OPS vs this arsenal",
                          scale=alt.Scale(domain=domain, clamp=True)),
                  size=_size_channel(data, "Arsenal AB", (40, 520), "Arsenal AB")[0],
                  color=alt.Color("Team:N", title="Team", scale=_team_scale(data["Team"])),
                  tooltip=arsenal_tooltip))
    labels = _label_layers(data, "Season OPS:Q", "Arsenal OPS:Q",
                           data["Arsenal OPS"] - data["Season OPS"],
                           "Team:N", _team_scale(data["Team"]))
    return _styled(_over(parity + points, labels), height, zoom=True)


# ===================================================================================
# Pitching
#
# Batted-ball rates are compositional — GB + FB + LD sum to 100 — so plotting GB against FB
# is honest (two of three parts) while plotting either against the total would not be.
# Park factor against defence is the pairing the report's own "Park/defense fit" row rests
# on, and it is a genuine two-axis question: a fly-ball arm in a park that suppresses
# nothing, behind an outfield that saves nothing, is a different problem from either alone.
# ===================================================================================

AVAIL_COLORS = {
    "Available": "#0ca30c",
    "Monitor": "#fab219",
    "Doubtful": "#ec835a",
    "Out": "#d03b3b",
}
AVAIL_ORDER = ["Available", "Monitor", "Doubtful", "Out"]

# League-average batted-ball rates, for the reference cross on the GB/FB panel.
LEAGUE_GB, LEAGUE_FB = 44.0, 25.0


def _avail_scale():
    return alt.Scale(domain=AVAIL_ORDER, range=[AVAIL_COLORS[a] for a in AVAIL_ORDER])


def _rule(value, axis):
    frame = pd.DataFrame({axis: [value]})
    chart = alt.Chart(frame).mark_rule(color=RULE_GREY, strokeDash=[4, 4],
                                       strokeWidth=1, opacity=0.6)
    return chart.encode(x=f"{axis}:Q") if axis == "x" else chart.encode(y=f"{axis}:Q")


def batted_ball_scatter(frame, label="Pitcher", height=380):
    """Ground-ball rate against fly-ball rate, with the league cross.

    Two of three compositional parts, which is the readable pair: up-and-left is a fly-ball
    arm, down-and-right a ground-ball arm, and the line-drive remainder is what is left over.
    Which quadrant an arm sits in is exactly what park and defence then act on.
    """
    data = _ready(frame, ["GB%", "FB%"])
    if data is None:
        return None
    tooltip = [alt.Tooltip("Name:N", title=label),
               alt.Tooltip("GB%:Q", title="GB%", format=".1f"),
               alt.Tooltip("FB%:Q", title="FB%", format=".1f"),
               alt.Tooltip("LD%:Q", title="LD%", format=".1f"),
               alt.Tooltip("HH%:Q", title="Hard-hit%", format=".1f"),
               alt.Tooltip("Brl%:Q", title="Barrel%", format=".1f"),
               alt.Tooltip("BIP:Q", title="Balls in play", format=".0f")]
    encode = dict(
        x=alt.X("GB%:Q", title="Ground-ball rate", scale=_qscale(data, "GB%")),
        y=alt.Y("FB%:Q", title="Fly-ball rate", scale=_qscale(data, "FB%")),
        size=_size_channel(data, "BIP", (40, 460), "Balls in play")[0],
        tooltip=tooltip)
    if "Status" in data.columns:
        encode["color"] = alt.Color("Status:N", title="Availability",
                                    scale=_avail_scale(), sort=AVAIL_ORDER)
    points = alt.Chart(data).mark_circle(opacity=0.85, stroke="#fcfcfb",
                                         strokeWidth=1.5).encode(**encode)
    labels = _label_layers(data, "GB%:Q", "FB%:Q", data["FB%"] - LEAGUE_FB)
    return _styled(_over(_rule(LEAGUE_GB, "x") + _rule(LEAGUE_FB, "y") + points, labels),
                   height, zoom=True)


def bullpen_workload_scatter(frame, height=380):
    """Two-day pitch count against three-day, coloured by the availability grade.

    The diagonal is where the three-day total is *entirely* the last two days, so points on
    it are arms who worked back-to-back and rested before that. Points well below it carry
    load that is already a day older, and therefore cheaper.
    """
    data = _ready(frame, ["two_day", "three_day"])
    if data is None:
        return None
    top = float(max(data["three_day"].max(), 1)) * 1.15
    domain = [0, top]
    parity = (alt.Chart(pd.DataFrame({"x": domain, "y": domain}))
              .mark_line(color=RULE_GREY, strokeDash=[4, 4], strokeWidth=1, opacity=0.6)
              .encode(x=alt.X("x:Q", scale=alt.Scale(domain=domain)),
                      y=alt.Y("y:Q", scale=alt.Scale(domain=domain))))
    points = (alt.Chart(data).mark_circle(opacity=0.9, stroke="#fcfcfb", strokeWidth=1.5,
                                          size=230)
              .encode(
                  x=alt.X("two_day:Q", title="Pitches, last two games",
                          scale=alt.Scale(domain=domain)),
                  y=alt.Y("three_day:Q", title="Pitches, last three games",
                          scale=alt.Scale(domain=domain)),
                  color=alt.Color("Status:N", title="Availability", scale=_avail_scale(),
                                  sort=AVAIL_ORDER),
                  tooltip=[alt.Tooltip("Name:N", title="Pitcher"),
                           alt.Tooltip("T:N", title="Throws"),
                           alt.Tooltip("two_day:Q", title="Two-day pitches", format=".0f"),
                           alt.Tooltip("three_day:Q", title="Three-day pitches",
                                       format=".0f"),
                           alt.Tooltip("b2b:N", title="Back-to-back"),
                           alt.Tooltip("Avail:N", title="Report grade"),
                           alt.Tooltip("Status:N", title="Measured grade")]))
    labels = _label_layers(data, "two_day:Q", "three_day:Q",
                           data["three_day"] - data["two_day"])
    return _styled(_over(parity + points, labels), height, zoom=True)


def park_defense_scatter(frame, height=420):
    """Park run factor against the defence behind the arm, across the slate.

    Bubble size is the starter's fly-ball rate, so all three parts of the report's own
    "Park/defense fit" question sit on one chart: a big bubble in the top-right is a
    fly-ball arm in a park that adds runs behind a defence that saves none.
    """
    data = _ready(frame, ["park_runs", "hits_saved"])
    if data is None:
        return None
    points = (alt.Chart(data).mark_circle(opacity=0.85, stroke="#fcfcfb", strokeWidth=1.5)
              .encode(
                  x=alt.X("park_runs:Q", title="Park run factor (1.00 = neutral)",
                          scale=_qscale(data, "park_runs")),
                  y=alt.Y("hits_saved:Q",
                          title="Defence behind him: hits saved per game",
                          scale=_qscale(data, "hits_saved")),
                  size=_size_channel(data, "fb_rate", (40, 460), "Starter FB%")[0],
                  tooltip=[alt.Tooltip("pitcher:N", title="Starter"),
                           alt.Tooltip("team:N", title="Pitching for"),
                           alt.Tooltip("game:N", title="Game"),
                           alt.Tooltip("park:N", title="Park"),
                           alt.Tooltip("park_runs:Q", title="Park runs", format=".2f"),
                           alt.Tooltip("hits_saved:Q", title="Hits saved/g", format=".2f"),
                           alt.Tooltip("fb_rate:Q", title="FB%", format=".1f"),
                           alt.Tooltip("gb_rate:Q", title="GB%", format=".1f")]))
    labels = _label_layers(data, "park_runs:Q", "hits_saved:Q", data["hits_saved"],
                           name_column="pitcher")
    return _styled(_over(_rule(1.0, "x") + _rule(0.0, "y") + points, labels), height,
                   zoom=True)


def slate_pitcher_scatter(frame, height=440):
    """Every starter on the slate: how good the arm is against how good the bats are.

    The mirror of the hitter slate view. The y axis is reversed because FIP is a
    lower-is-better scale and an un-reversed axis would put the best arms at the bottom,
    which reads backwards next to every other chart here.
    """
    data = _ready(frame, ["fip", "opp_ops"])
    if data is None:
        return None
    points = (alt.Chart(data).mark_circle(opacity=0.85, stroke="#fcfcfb", strokeWidth=1.5)
              .encode(
                  x=alt.X("opp_ops:Q", title="OPS of the lineup he faces",
                          scale=_qscale(data, "opp_ops")),
                  y=alt.Y("fip:Q", title="FIP  (better arms higher)",
                          scale=_qscale(data, "fip", reverse=True)),
                  size=_size_channel(data, "k_bb", (40, 460), "K-BB%")[0],
                  tooltip=[alt.Tooltip("pitcher:N", title="Starter"),
                           alt.Tooltip("team:N", title="Team"),
                           alt.Tooltip("game:N", title="Game"),
                           alt.Tooltip("fip:Q", title="FIP", format=".2f"),
                           alt.Tooltip("k_bb:Q", title="K-BB%", format=".1f"),
                           alt.Tooltip("opp_ops:Q", title="Opp OPS", format=".3f"),
                           alt.Tooltip("score:Q", title="Watchlist score", format=".1f")]))
    labels = _label_layers(data, "opp_ops:Q", "fip:Q", data["fip"] - data["fip"].median(),
                           name_column="pitcher")
    return _styled(_over(points, labels), height, zoom=True)


# ===================================================================================
# Price
#
# Salary and composite correlate at about +0.73, so a plot of one against the other mostly
# restates the market. The panels below are built around the *residual* instead: how far a
# hitter sits from what his own price bracket normally delivers. That is a signed quantity
# with a meaningful zero, so it gets the diverging treatment — two hues and a neutral
# midpoint — rather than a categorical or sequential one.
# ===================================================================================

# Above or below par: two hues, no ramp between them.
#
# **A continuous surplus ramp was the wrong encoding twice over.** On the bars, surplus is
# already the bar length, and on the scatter it is already the distance from the par line —
# so a colour ramp spends the one free channel restating what the geometry says. It also
# rendered badly: interpolating a saturated red to a light neutral to a saturated blue runs
# the mid-tones through magenta, and on a dark surface the pale midpoint out-shouted both
# ends.
#
# Sign is a two-state fact, so it gets two colours and the magnitude stays with the geometry.
SURPLUS_ABOVE = "#2a78d6"
SURPLUS_BELOW = "#d03b3b"


def _sign_scale():
    return alt.Scale(domain=["above par", "below par"],
                     range=[SURPLUS_ABOVE, SURPLUS_BELOW])


def _with_sign(frame):
    out = frame.copy()
    surplus = pd.to_numeric(out["surplus"], errors="coerce")
    out["vs par"] = np.where(surplus >= 0, "above par", "below par")
    return out


SALARY_TOOLTIP = [
    alt.Tooltip("Name:N", title="Player"),
    alt.Tooltip("Team:N", title="Team"),
    alt.Tooltip("game:N", title="Game"),
    alt.Tooltip("Salary:Q", title="Salary", format="$,.0f"),
    alt.Tooltip("Composite:Q", title="Composite", format=".1f"),
    alt.Tooltip("expected:Q", title="Par for the price", format=".1f"),
    alt.Tooltip("surplus:Q", title="Surplus", format="+.1f"),
    alt.Tooltip("per_1k:Q", title="Composite per $1k", format=".1f"),
    alt.Tooltip("Signal:N", title="Signal"),
] + HITTER_OPS_TOOLTIP


def salary_scatter(frame, height=440):
    """Composite against salary, with the going rate for each price band drawn through it.

    The stepped line is the median composite *within a price band*, which is what "par for
    the price" means here. It is deliberately not a straight fit: the relationship flattens
    through the middle of the board and steepens at the top, so a regression line would
    systematically call the middle underpriced.
    """
    data = _ready(frame, ["Salary", "Composite", "surplus"])
    if data is None:
        return None
    points = (alt.Chart(data).mark_circle(opacity=0.85, stroke="#fcfcfb", strokeWidth=1.2)
              .encode(
                  x=alt.X("Salary:Q", title="DraftKings salary",
                          scale=_qscale(data, "Salary"), axis=alt.Axis(format="$,.0f")),
                  y=alt.Y("Composite:Q", title="Composite matchup score",
                          scale=_qscale(data, "Composite")),
                  # Signal, not surplus: distance from the par line already *is* the
                  # surplus, so the colour channel is free to carry the report's own call.
                  color=alt.Color("Signal:N", title="Signal", scale=_signal_scale(),
                                  sort=SIGNAL_ORDER),
                  size=_size_channel(data, "Season AB", (40, 420), "Season AB")[0],
                  tooltip=SALARY_TOOLTIP))
    labels = _label_layers(data, "Salary:Q", "Composite:Q", data["surplus"], team=True)
    curve = band_curve_chart(data)
    return _styled(_over(points if curve is None else curve + points, labels), height,
                   zoom=True)


def band_curve_chart(data):
    """The price-band expectation, as a line the points can be read against."""
    from dashboards.salaries import band_curve

    curve = band_curve(data)
    if curve is None or curve.empty or len(curve) < 2:
        return None
    return (alt.Chart(curve)
            .mark_line(color=RULE_GREY, strokeWidth=1.5, strokeDash=[5, 4],
                       point=alt.OverlayMarkDef(color=RULE_GREY, size=45))
            .encode(x=alt.X("midpoint:Q"), y=alt.Y("expected:Q"),
                    tooltip=[alt.Tooltip("band:N", title="Price band"),
                             alt.Tooltip("expected:Q", title="Par composite", format=".1f"),
                             alt.Tooltip("n:Q", title="Hitters", format=".0f")]))


def surplus_bars(frame, top_n=18, height=None):
    """Ranked surplus — how far each hitter beats or misses the going rate for his price.

    Both tails, because the overpriced end is as actionable as the underpriced one. Bars grow
    from a zero rule, so the sign is carried by direction and the hue only reinforces it.
    """
    data = _ready(frame, ["Salary", "Composite", "surplus"])
    if data is None:
        return None
    ranked = data.sort_values("surplus", ascending=False)
    half = max(1, top_n // 2)
    picked = (pd.concat([ranked.head(half), ranked.tail(half)])
              .drop_duplicates(subset=["Name", "Team"]) if len(ranked) > top_n else ranked)
    height = height or max(220, 26 * len(picked))
    label = alt.X("surplus:Q", title="Composite above or below par for the price")
    bars = (alt.Chart(_with_sign(picked)).mark_bar(cornerRadiusEnd=3, height=14)
            .encode(
                x=label,
                y=alt.Y("Name:N", title=None, sort="-x"),
                color=alt.Color("vs par:N", title=None, scale=_sign_scale(),
                                legend=None),
                tooltip=SALARY_TOOLTIP))
    rule = (alt.Chart(pd.DataFrame({"x": [0.0]}))
            .mark_rule(color=RULE_GREY, strokeWidth=1).encode(x="x:Q"))
    return _styled(rule + bars, height)


def price_position_scatter(frame, height=400):
    """Composite per $1k against salary, split by the roster slot he fills.

    Position matters to price in a way it does not to matchup: catchers are cheap because
    catchers are cheap, not because their matchups are worse. Faceting by slot keeps a
    comparison inside the pool a lineup actually chooses from.
    """
    data = _ready(frame, ["Salary", "per_1k"])
    if data is None or "DK Pos" not in data.columns:
        return None
    data = data[data["DK Pos"].notna()]
    if data.empty:
        return None
    signed = scales.carry(_with_sign(data), data)
    points = (alt.Chart(signed)
              .mark_circle(opacity=0.8, stroke="#fcfcfb", strokeWidth=1)
              .encode(
                  x=alt.X("Salary:Q", title="Salary", scale=_qscale(data, "Salary"),
                          axis=alt.Axis(format="$,.0f")),
                  y=alt.Y("per_1k:Q", title="Composite per $1k",
                          scale=_qscale(data, "per_1k")),
                  color=alt.Color("vs par:N", title=None, scale=_sign_scale()),
                  tooltip=SALARY_TOOLTIP + [alt.Tooltip("DK Pos:N", title="Slot")]))
    labels = _label_layers(signed, "Salary:Q", "per_1k:Q",
                           signed["per_1k"] - signed["per_1k"].median(),
                           "vs par:N", _sign_scale(), team=True)
    return _styled(_over(points, labels), height, zoom=True)


# ===================================================================================
# Outcome background
#
# A "where do usable picks come from" layer: historical hit rate over a region of the board,
# drawn behind tonight's points.
#
# Sequential, one hue, light to dark — this is magnitude on an ordered scale, and a
# categorical or diverging palette here would invent structure the data does not have. The
# points on top keep their own encoding and get a surface ring so they stay legible over a
# filled cell.
#
# **A thin cell is drawn as nothing rather than pale.** On a grid the eye reads adjacency as
# a trend, so a cell holding eight observations coloured at all is a claim the sample cannot
# support. `outcomes.MIN_CELL` drops them before they reach here.
# ===================================================================================

HIT_SCHEME = "blues"


def hit_background(grid, x_title, y_title, opacity=0.55):
    """The binned hit-rate layer. None when there is nothing solid enough to draw."""
    if grid is None or getattr(grid, "empty", True):
        return None
    return (alt.Chart(grid)
            .mark_rect(opacity=opacity)
            .encode(
                x=alt.X("x0:Q", title=x_title),
                x2="x1:Q",
                y=alt.Y("y0:Q", title=y_title),
                y2="y1:Q",
                color=alt.Color("rate_pct:Q", title="Historical hit rate %",
                                scale=alt.Scale(scheme=HIT_SCHEME)),
                tooltip=[alt.Tooltip("rate_pct:Q", title="Hit rate %", format=".0f"),
                         alt.Tooltip("points:Q", title="Mean DK points", format=".1f"),
                         alt.Tooltip("n:Q", title="Historical players", format=".0f")]))


def with_hit_background(points_chart, grid, x_title, y_title, height=440):
    """Put a scatter over its outcome grid. Falls back to the bare scatter."""
    background = hit_background(grid, x_title, y_title)
    if background is None:
        return points_chart
    return _styled(background + points_chart, height)


def value_scatter_with_history(frame, grid=None, height=460):
    """Tonight's board over the historical hit rate for the same region.

    Salary on x and composite on y, which is the plane the board is actually chosen on. The
    background says what share of hitters in each cell have cleared the points threshold
    historically; the dots are tonight.
    """
    data = _ready(frame, ["Salary", "Composite"])
    if data is None:
        return None
    points = (alt.Chart(data).mark_circle(opacity=0.9, stroke="#fcfcfb", strokeWidth=1.4)
              .encode(
                  x=alt.X("Salary:Q", title="DraftKings salary",
                          scale=_qscale(data, "Salary"), axis=alt.Axis(format="$,.0f")),
                  y=alt.Y("Composite:Q", title="Composite matchup score",
                          scale=_qscale(data, "Composite")),
                  color=alt.Color("Signal:N", title="Signal", scale=_signal_scale(),
                                  sort=SIGNAL_ORDER),
                  size=_size_channel(data, "Season AB", (40, 400), "Season AB")[0],
                  tooltip=SALARY_TOOLTIP))
    labels = _label_layers(data, "Salary:Q", "Composite:Q",
                           data["Composite"] - data["Composite"].median(), team=True)
    if grid is None or getattr(grid, "empty", True):
        return _styled(_over(points, labels), height, zoom=True)
    # The background carries its own colour legend; two legends for two different meanings
    # is correct here, and the rect layer is drawn first so the dots sit on top of it.
    return _styled(_over(hit_background(grid, "DraftKings salary",
                                        "Composite matchup score") + points, labels),
                   height, zoom=True)


def lift_bars(table, height=None):
    """Hit rate by band with the population rate marked, so lift is visible not computed.

    The rule line is the whole point: a set of bars without it invites reading any ordering
    as an edge, when what matters is distance from the base rate.
    """
    if table is None or getattr(table, "empty", True):
        return None
    height = height or max(200, 34 * len(table))
    base = float((table["rate_pct"] * table["n"]).sum() / table["n"].sum())
    bars = (alt.Chart(table).mark_bar(cornerRadiusEnd=3, height=18)
            .encode(
                x=alt.X("rate_pct:Q", title="Hit rate %"),
                y=alt.Y("band:N", title=None, sort=None),
                color=alt.Color("rate_pct:Q", title=None,
                                scale=alt.Scale(scheme=HIT_SCHEME), legend=None),
                tooltip=[alt.Tooltip("band:N", title="Band"),
                         alt.Tooltip("rate_pct:Q", title="Hit rate %", format=".1f"),
                         alt.Tooltip("lift:Q", title="Lift vs base", format=".2f"),
                         alt.Tooltip("points:Q", title="Mean points", format=".2f"),
                         alt.Tooltip("salary:Q", title="Mean salary", format="$,.0f"),
                         alt.Tooltip("n:Q", title="Players", format=".0f")]))
    rule = (alt.Chart(pd.DataFrame({"x": [base]}))
            .mark_rule(color=RULE_GREY, strokeDash=[4, 4], strokeWidth=1.5)
            .encode(x="x:Q",
                    tooltip=[alt.Tooltip("x:Q", title="Base rate %", format=".1f")]))
    return _styled(bars + rule, height)


def controlled_spread_bars(table, height=None):
    """Spread between the best and worst third *within* a price band, against its own noise.

    The error bar is two standard errors of the difference in proportions. A bar that does
    not clear it is not a finding, and drawing them together is the only honest way to show
    a spread this small.
    """
    if table is None or getattr(table, "empty", True):
        return None
    height = height or max(190, 44 * len(table))
    data = table.copy()
    data["lo"] = data["spread"] - 2 * data["se"]
    data["hi"] = data["spread"] + 2 * data["se"]
    bars = (alt.Chart(data).mark_bar(height=14, cornerRadiusEnd=3)
            .encode(
                x=alt.X("spread:Q", title="Hit-rate spread, top third minus bottom (pp)"),
                y=alt.Y("control band:N", title="Salary band", sort=None),
                color=alt.Color("beyond noise:N", title="Clears 2 SE",
                                scale=alt.Scale(domain=[True, False],
                                                range=["#2a78d6", "#9aa0a6"])),
                tooltip=[alt.Tooltip("control band:N", title="Salary band"),
                         alt.Tooltip("low:Q", title="Bottom third hit %", format=".1f"),
                         alt.Tooltip("high:Q", title="Top third hit %", format=".1f"),
                         alt.Tooltip("spread:Q", title="Spread (pp)", format="+.1f"),
                         alt.Tooltip("se:Q", title="Std error (pp)", format=".1f"),
                         alt.Tooltip("n:Q", title="Players", format=".0f")]))
    error = (alt.Chart(data).mark_rule(strokeWidth=1.5, color=RULE_GREY)
             .encode(x="lo:Q", x2="hi:Q", y=alt.Y("control band:N", sort=None)))
    zero = (alt.Chart(pd.DataFrame({"x": [0.0]}))
            .mark_rule(color=RULE_GREY, strokeWidth=1).encode(x="x:Q"))
    return _styled(zero + error + bars, height)


def hit_grid_chart(grid, x_title, y_title, height=460):
    """The outcome grid on its own, clickable.

    A named point selection on the cell bounds, so Streamlit can hand back which cell was
    clicked and the page can show the players behind it. The selection is what makes the
    grid a drill-down rather than a picture — an unexplained dark square is not evidence.
    """
    if grid is None or getattr(grid, "empty", True):
        return None
    picked = alt.selection_point(name="cell", fields=["x0", "x1", "y0", "y1"], empty=True)
    return _styled(
        alt.Chart(grid).mark_rect(stroke="#ffffff", strokeWidth=1)
        .encode(
            x=alt.X("x0:Q", title=x_title), x2="x1:Q",
            y=alt.Y("y0:Q", title=y_title), y2="y1:Q",
            color=alt.Color("rate_pct:Q", title="Hit rate %",
                            scale=alt.Scale(scheme=HIT_SCHEME)),
            opacity=alt.condition(picked, alt.value(0.95), alt.value(0.35)),
            tooltip=[alt.Tooltip("rate_pct:Q", title="Hit rate %", format=".0f"),
                     alt.Tooltip("points:Q", title="Mean DK points", format=".1f"),
                     alt.Tooltip("n:Q", title="Players", format=".0f"),
                     alt.Tooltip("x0:Q", title="From", format=",.0f"),
                     alt.Tooltip("x1:Q", title="To", format=",.0f")])
        .add_params(picked), height)


def outcome_scatter(frame, x, y, x_title, y_title, height=440):
    """Scored history as points: one past hitter-slate each, coloured by whether he hit.

    Two states, not a ramp — the question this answers is "did it clear the bar", and a
    continuous points scale here would bury that in a gradient. Reserved status colours,
    with the miss in neutral grey so the hits are what the eye finds.
    """
    data = _ready(frame, [x, y, "hit"])
    if data is None:
        return None
    plot = data.assign(**{"Result": np.where(data["hit"], "Hit", "Miss")})
    return _styled(
        alt.Chart(plot).mark_circle(size=46, opacity=0.6)
        .encode(
            x=alt.X(f"{x}:Q", title=x_title, scale=_qscale(data, x)),
            y=alt.Y(f"{y}:Q", title=y_title, scale=_qscale(data, y)),
            color=alt.Color("Result:N", title="Result",
                            scale=alt.Scale(domain=["Hit", "Miss"],
                                            range=[SIGNAL_COLORS["Priority"], "#9aa0a6"])),
            tooltip=[alt.Tooltip("Name:N"), alt.Tooltip("date:N", title="Date"),
                     alt.Tooltip("Salary:Q", title="Salary", format="$,.0f"),
                     alt.Tooltip("Composite:Q", title="Composite", format=".1f"),
                     alt.Tooltip("fpts:Q", title="DK points", format=".1f")] +
                    HITTER_OPS_TOOLTIP),
        height, zoom=True)


#: What a point is sized by. Measured across a slate and against outcomes:
#:
#:     column       p90/p10 spread   corr(actual pts)   corr(20+ pt game)
#:     Proj              1.74             +0.159             +0.095
#:     Ceiling           1.45             +0.164             +0.112
#:     Ceiling/Proj      1.27             -0.115             -0.042
#:
#: **Ceiling was the wrong choice and is no longer the default.** Size is an area channel,
#: so what a reader can actually see is the *ratio* between points — and ceiling's runs
#: 11.3 to 20.7 across a whole slate, a p90/p10 of 1.45. Every dot came out nearly the same
#: size. It is also 0.947 correlated with `Proj`, so it was not even a second dimension;
#: after regressing it on Proj the residual carries +0.043 against points.
#:
#: `Proj` has the widest spread of the family and is the strongest single predictor in the
#: dataset — ahead of salary. Ceiling stays selectable for the tournament question
#: specifically, where it does beat Proj on 20+ point games (+0.112 vs +0.095).
SIZE_SCALES = {
    "Proj": "Projection (DK points)",
    "Ceiling": "Ceiling (DK points)",
    "PA": "Expected plate appearances",
    "Season AB": "Season at-bats",
}
DEFAULT_SIZE = "Proj"


def _size_channel(data, column, default_range=(40, 420), title=None):
    """A size encoding for `column`, or a constant when the frame cannot support one.

    Zero is excluded from the domain deliberately. Altair maps value to *area* from zero by
    default, which for a column like ceiling — spanning 11 to 21 — squeezes every point into
    the top third of the range and makes them indistinguishable. Anchoring the domain to the
    data's own range is what restores the contrast the channel is there to provide.
    """
    if column not in data.columns:
        return alt.value(110), None
    values = pd.to_numeric(data[column], errors="coerce").dropna()
    anchored = scales.extent(data, column)
    if values.empty or (values.nunique() < 2 and not anchored):
        return alt.value(110), None
    low, high = anchored or (float(values.min()), float(values.max()))
    return (alt.Size(f"{column}:Q", title=title or SIZE_SCALES.get(column, column),
                     scale=alt.Scale(domain=[low, high], range=list(default_range))),
            column)


def ceiling_scatter(frame, grid=None, height=470, size_by=DEFAULT_SIZE):
    """Matchup score against price, with each bat sized by its projection.

    The third dimension is the point of it. Salary and composite correlate at about +0.73,
    so on those two axes alone the board is close to a diagonal line and every read is
    "expensive bats are good". A projection is not on that line: it folds in opportunity —
    lineup slot, expected plate appearances — that neither axis carries.

    Size, not colour, carries it: this is a magnitude on a ratio scale, and the reserved
    Signal palette is already doing identity work here that a second ramp would fight.
    """
    data = _ready(frame, ["Salary", "Composite", size_by])
    if data is None:
        return None
    plot = data.assign(
        ceiling_per_1k=np.where(data["Salary"] > 0,
                                data["Ceiling"] / (data["Salary"] / 1000.0), np.nan))
    points = (alt.Chart(plot).mark_circle(opacity=0.82, stroke="#fcfcfb", strokeWidth=1.2)
              .encode(
                  x=alt.X("Salary:Q", title="DraftKings salary",
                          scale=_qscale(plot, "Salary"), axis=alt.Axis(format="$,.0f")),
                  y=alt.Y("Composite:Q", title="Composite matchup score",
                          scale=_qscale(plot, "Composite")),
                  size=_size_channel(plot, size_by, (40, 620))[0],
                  color=alt.Color("Signal:N", title="Signal", scale=_signal_scale(),
                                  sort=SIGNAL_ORDER),
                  tooltip=[alt.Tooltip("Name:N"), alt.Tooltip("Team:N"),
                           alt.Tooltip("Salary:Q", format="$,.0f"),
                           alt.Tooltip("Composite:Q", format=".1f"),
                           alt.Tooltip("Proj:Q", title="Projection", format=".1f"),
                           alt.Tooltip("Ceiling:Q", format=".1f"),
                           alt.Tooltip("Floor:Q", format=".1f"),
                           alt.Tooltip("ceiling_per_1k:Q", title="Ceiling per $1k",
                                       format=".2f")] + HITTER_OPS_TOOLTIP))
    labels = _label_layers(plot, "Salary:Q", "Composite:Q",
                           plot["Composite"] - plot["Composite"].median(), team=True)
    background = hit_background(grid, "DraftKings salary", "Composite matchup score")
    return _styled(_over(points if background is None else background + points, labels),
                   height, zoom=True)


def ceiling_value_scatter(frame, height=440, domain=None):
    """Ceiling against price, with the slate's own ceiling-for-the-money line.

    The question a tournament roster actually asks — what upside does this dollar buy —
    which the composite does not answer. The dashed line is the least-squares fit of
    ceiling on salary, i.e. **the going rate for upside on this slate**; the bats worth
    looking at are the ones sitting above it. It is a reference line, not a model: read
    distance from it, not the line itself.
    """
    data = _ready(frame, ["Salary", "Ceiling"])
    if data is None:
        return None
    points = (alt.Chart(data).mark_point(filled=True, size=110, opacity=0.82,
                                         stroke="#fcfcfb", strokeWidth=1.1)
              .encode(
                  x=alt.X("Salary:Q", title="DraftKings salary",
                          scale=_qscale(data, "Salary"), axis=alt.Axis(format="$,.0f")),
                  y=alt.Y("Ceiling:Q", title="Ceiling (DK points)",
                          scale=_qscale(data, "Ceiling")),
                  color=(_team_color(domain) if domain else
                         alt.Color("Signal:N", title="Signal", scale=_signal_scale(),
                                   sort=SIGNAL_ORDER)),
                  shape=(alt.Shape("Signal:N", title="Signal",
                                   scale=_signal_shape_scale(), sort=SIGNAL_ORDER)
                         if domain else alt.Undefined),
                  tooltip=[alt.Tooltip("Name:N"), alt.Tooltip("Team:N"),
                           alt.Tooltip("Salary:Q", format="$,.0f"),
                           alt.Tooltip("Ceiling:Q", format=".1f"),
                           alt.Tooltip("Composite:Q", format=".1f")] + HITTER_OPS_TOOLTIP))
    trend = (alt.Chart(data).transform_regression("Salary", "Ceiling")
             .mark_line(color=RULE_GREY, strokeDash=[5, 4], strokeWidth=1.5)
             .encode(x="Salary:Q", y="Ceiling:Q"))
    labels = _label_layers(data, "Salary:Q", "Ceiling:Q",
                           data["Ceiling"] - data["Ceiling"].median(), team=True)
    return _styled(_over(trend + points, labels), height, zoom=True)


# ===================================================================================
# Conditions
#
# Where a fly ball lands is decided by three things the report keeps in three different
# tables: how far the park plays, what the air is doing, and who is under it. These panels
# put a staff's contact profile on one axis and its conditions on the other, so the
# interaction is a position on a chart rather than a cross-reference in the reader's head.
# ===================================================================================

#: Starter against bullpen. Two series, so the validated pair is enough, and these are
#: units rather than clubs, so no team identity is implied by the hues.
UNIT_COLORS = {"Starter": "#2a78d6", "Bullpen": "#7b4fb5"}

#: Direction glyphs. Shape, not colour, carries direction: the status palette is reserved,
#: and "up" here is good for hitters and bad for pitchers depending on who is reading, so a
#: red/green ramp would be asserting a side the chart has no business taking.
IMPACT_SHAPES = {"up": "triangle-up", "down": "triangle-down", "neutral": "circle"}
IMPACT_ORDER = ["up", "neutral", "down"]


def _unit_scale():
    return alt.Scale(domain=list(UNIT_COLORS), range=list(UNIT_COLORS.values()))


def hr_exposure_scatter(frame, league_fb=25.0, height=470):
    """Fly-ball rate against the home-run environment it will be pitched in.

    The two reference rules are what make this readable: league fly-ball rate on one axis, a
    neutral environment on the other. They cut the plane into quadrants that mean something
    — top right is a fly-ball staff in a park and air that carry, which is the target, and
    bottom left is the opposite.

    Sized by balls in play, because a bullpen's rate is measured over several hundred and a
    starter's over a few dozen, and treating those as equally known is the mistake this
    panel would otherwise invite.
    """
    data = _ready(frame, ["FB%", "hr_env", "unit"])
    if data is None:
        return None
    # A layered chart resolves like-named scales together. If a one-value rule is left on
    # Altair's default quantitative scale, that scale starts at zero and its union with the
    # points silently drags a .94-1.20 environment plot down to 0-1.3. Give rules the exact
    # same domains as the marks so they annotate the plane rather than redefining it.
    neutral_fb = alt.Chart(pd.DataFrame({"x": [league_fb]})).mark_rule(
        color=RULE_GREY, strokeDash=[4, 4]).encode(
            x=alt.X("x:Q", scale=_qscale(data, "FB%")))
    neutral_env = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(
        color=RULE_GREY, strokeDash=[4, 4]).encode(
            y=alt.Y("y:Q", scale=_qscale(data, "hr_env")))
    points = (alt.Chart(data).mark_point(filled=True, opacity=0.85,
                                         stroke="#fcfcfb", strokeWidth=1.1)
              .encode(
                  x=alt.X("FB%:Q", title="Fly-ball rate allowed (%)",
                          scale=_qscale(data, "FB%")),
                  y=alt.Y("hr_env:Q", title="Home-run environment (park x weather)",
                          scale=_qscale(data, "hr_env")),
                  color=alt.Color("unit:N", title="Unit", scale=_unit_scale()),
                  shape=alt.Shape("impact:N", title="Environment",
                                  scale=alt.Scale(domain=IMPACT_ORDER,
                                                  range=[IMPACT_SHAPES[i]
                                                         for i in IMPACT_ORDER]),
                                  sort=IMPACT_ORDER),
                  size=_size_channel(data, "BIP", (60, 500), "Balls in play")[0],
                  tooltip=[alt.Tooltip("name:N", title="Staff"),
                           alt.Tooltip("team:N", title="Team"),
                           alt.Tooltip("game:N", title="Game"),
                           alt.Tooltip("park:N", title="Park"),
                           alt.Tooltip("FB%:Q", format=".1f"),
                           alt.Tooltip("park_hr:Q", title="Park HR", format=".2f"),
                           alt.Tooltip("weather_hr:Q", title="Weather HR", format=".3f"),
                           alt.Tooltip("hr_env:Q", title="Combined", format=".2f"),
                           alt.Tooltip("temp:Q", title="Temp F", format=".0f"),
                           alt.Tooltip("wind:N", title="Wind"),
                           alt.Tooltip("hr_leverage:Q", title="Exposure", format="+.1f")]))
    labels = _label_layers(data, "FB%:Q", "hr_env:Q", data["FB%"] - league_fb,
                           name_column="name")
    return _styled(_over(neutral_fb + neutral_env + points, labels), height, zoom=True)


def hr_impact_arrows(frame, top=16, height=None):
    """Staffs ranked by how exposed they are to tonight's home-run conditions.

    The arrow is the ask: direction and degree at a glance. It is a **heuristic** — fly-ball
    rate relative to league, scaled by the environment's nudge — and the page says so. Read
    the ordering, not the number. A staff in a neutral environment gets a dot rather than an
    arrow, because pointing at nothing is worse than saying nothing.
    """
    data = _ready(frame, ["hr_leverage", "name", "impact"])
    if data is None:
        return None
    ranked = (data.reindex(data["hr_leverage"].abs().sort_values(ascending=False).index)
              .head(top).copy())
    ranked["label"] = ranked["name"].astype(str) + " - " + ranked["unit"].astype(str)
    order = list(ranked.sort_values("hr_leverage", ascending=False)["label"])
    height = height or max(220, 26 * len(ranked))

    bars = (alt.Chart(ranked).mark_bar(height=13, cornerRadiusEnd=3)
            .encode(
                x=alt.X("hr_leverage:Q",
                        title="Exposure to tonight's HR conditions (heuristic)"),
                y=alt.Y("label:N", title=None, sort=order),
                color=alt.Color("unit:N", title="Unit", scale=_unit_scale()),
                tooltip=[alt.Tooltip("name:N", title="Staff"),
                         alt.Tooltip("game:N", title="Game"),
                         alt.Tooltip("park:N", title="Park"),
                         alt.Tooltip("FB%:Q", format=".1f"),
                         alt.Tooltip("hr_env:Q", title="HR environment", format=".2f"),
                         alt.Tooltip("temp:Q", title="Temp F", format=".0f"),
                         alt.Tooltip("wind:N", title="Wind"),
                         alt.Tooltip("hr_leverage:Q", title="Exposure", format="+.1f")]))
    heads = (alt.Chart(ranked).mark_point(filled=True, size=110, opacity=0.95)
             .encode(
                 x=alt.X("hr_leverage:Q"),
                 y=alt.Y("label:N", sort=order),
                 shape=alt.Shape("impact:N", title="Direction",
                                 scale=alt.Scale(domain=IMPACT_ORDER,
                                                 range=["triangle-right", "circle",
                                                        "triangle-left"]),
                                 sort=IMPACT_ORDER),
                 color=alt.Color("unit:N", scale=_unit_scale(), legend=None)))
    zero = (alt.Chart(pd.DataFrame({"x": [0.0]}))
            .mark_rule(color=RULE_GREY, strokeWidth=1).encode(x="x:Q"))
    return _styled(zero + bars + heads, height)


#: Park-factor axes this panel understands, with what each one is asking.
PARK_AXES = {"park_2b": "Park doubles factor (100 = neutral)",
             "park_r": "Park runs factor (100 = neutral)",
             "park_hr": "Park home-run factor (1.00 = neutral)"}


def defense_park_scatter(frame, x="GB+LD%", park_column="park_2b", height=460):
    """Contact a defence has to field, against how the park rewards it getting through.

    Ground balls and line drives are the balls defenders actually decide, and a park's
    **doubles** factor is what a gap costs — a different question from the home-run factor,
    and one that often points the other way. A staff putting a lot of fieldable contact into
    a doubles park behind a defence that costs hits sits bottom-right, which is the read.
    """
    data = _ready(frame, [x, park_column])
    if data is None:
        return None
    neutral_at = 1.0 if park_column == "park_hr" else 100.0
    neutral = alt.Chart(pd.DataFrame({"y": [neutral_at]})).mark_rule(
        color=RULE_GREY, strokeDash=[4, 4]).encode(
            y=alt.Y("y:Q", scale=_qscale(data, park_column)))
    # Hits saved swings both ways around zero, so the size channel cannot carry it — a
    # negative radius is meaningless. Colour takes the sign; the value stays on hover.
    plot = scales.carry(data.assign(defence=np.where(
        pd.to_numeric(data.get("hits_saved"), errors="coerce").fillna(0) >= 0,
        "saves hits", "costs hits")), data)
    points = (alt.Chart(plot).mark_point(filled=True, size=150, opacity=0.85,
                                         stroke="#fcfcfb", strokeWidth=1.1)
              .encode(
                  x=alt.X(f"{x}:Q", title=f"{x} allowed", scale=_qscale(plot, x)),
                  y=alt.Y(f"{park_column}:Q",
                          title=PARK_AXES.get(park_column, park_column),
                          scale=_qscale(plot, park_column)),
                  color=alt.Color("defence:N", title="Defence behind them",
                                  scale=alt.Scale(domain=["saves hits", "costs hits"],
                                                  range=["#2a78d6", "#c96a1e"])),
                  shape=alt.Shape("unit:N", title="Unit"),
                  tooltip=[alt.Tooltip("name:N", title="Staff"),
                           alt.Tooltip("team:N", title="Team"),
                           alt.Tooltip("park:N", title="Park"),
                           alt.Tooltip("GB%:Q", format=".1f"),
                           alt.Tooltip("LD%:Q", format=".1f"),
                           alt.Tooltip("GB+LD%:Q", format=".1f"),
                           alt.Tooltip(f"{park_column}:Q", title="Park factor",
                                       format=".2f"),
                           alt.Tooltip("hits_saved:Q", title="Hits saved/g",
                                       format="+.2f"),
                           alt.Tooltip("defense_grade:N", title="Defence")]))
    labels = _label_layers(plot, f"{x}:Q", f"{park_column}:Q",
                           plot[park_column] - neutral_at, name_column="name")
    return _styled(_over(neutral + points, labels), height, zoom=True)


# ===================================================================================
# Against what this arm actually allows
#
# The arsenal panels elsewhere compare a hitter's line against a pitch mix to his own season
# line, which answers "is he better or worse than usual against these pitches" — a question
# about the hitter alone. These compare it instead to **what this particular arm surrenders
# to batters of his side**, which is the matchup question: a .700 bat facing someone who
# gives up .639 to his side is in a worse spot than a .680 bat facing someone who gives up
# .780, and a season baseline cannot see that at all.
# ===================================================================================

#: Below this many at-bats against the arsenal, the measured line is mostly noise. Points
#: are still drawn — the panel is about a slate, and hiding thin samples hides who is thin —
#: but they are drawn hollow so the eye discounts them without being told to.
ARSENAL_AB_SOLID = 20


def arsenal_vs_allowed_scatter(frame, baseline="Arsenal OPS", height=470, teams=None):
    """A hitter's line against tonight's arsenal, against what that arm gives up to his side.

    The diagonal is the whole chart: on it, the hitter does exactly what this arm allows
    batters like him. Above it he beats the arm's own baseline, which is the only version of
    "good matchup" that has the pitcher in it.

    Hollow points are thin samples. They are shown rather than filtered because a slate view
    that silently drops everyone with few at-bats against the mix is hiding the most common
    case, not cleaning it up.
    """
    data = _ready(frame, [baseline, "Allowed OPS"])
    if data is None:
        return None
    data = data[data.get("Arsenal AB", pd.Series(1, index=data.index)).fillna(0) > 0]
    if data.empty:
        return None

    domain = _same_unit_domain(data, [baseline, "Allowed OPS"], floor=0.0)

    parity = (alt.Chart(pd.DataFrame({"v": domain}))
              .mark_line(color=RULE_GREY, strokeDash=[5, 4], strokeWidth=1.5)
              .encode(x=alt.X("v:Q", scale=alt.Scale(domain=domain, clamp=True)),
                      y=alt.Y("v:Q", scale=alt.Scale(domain=domain, clamp=True))))

    plot = data.assign(sample=np.where(
        data.get("Arsenal AB", pd.Series(0, index=data.index)).fillna(0) >= ARSENAL_AB_SOLID,
        f"{ARSENAL_AB_SOLID}+ AB", f"under {ARSENAL_AB_SOLID} AB"))
    points = (alt.Chart(plot).mark_point(opacity=0.88, strokeWidth=1.6)
              .encode(
                  x=alt.X("Allowed OPS:Q", scale=alt.Scale(domain=domain, clamp=True),
                          title="OPS this arm surrenders to his side"),
                  y=alt.Y(f"{baseline}:Q", scale=alt.Scale(domain=domain, clamp=True),
                          title=f"His {baseline.replace(' OPS', '').lower()} OPS"),
                  color=(_team_color(teams) if teams else
                         alt.Color("Signal:N", title="Signal", scale=_signal_scale(),
                                   sort=SIGNAL_ORDER)),
                  fill=alt.Fill("sample:N", title="Sample",
                                scale=alt.Scale(
                                    domain=[f"{ARSENAL_AB_SOLID}+ AB",
                                            f"under {ARSENAL_AB_SOLID} AB"],
                                    range=["#5b8def", "transparent"])),
                  size=_size_channel(data, "Arsenal AB", (50, 420), "AB vs arsenal")[0],
                  tooltip=[alt.Tooltip("Name:N"), alt.Tooltip("Team:N"),
                           alt.Tooltip("Bats:N", title="Bats"),
                           alt.Tooltip("Effective Side:N", title="Bats tonight"),
                           alt.Tooltip("Opp SP:N", title="Facing"),
                           alt.Tooltip("Faces Hand:N", title="Throws"),
                           alt.Tooltip(f"{baseline}:Q", format=".3f"),
                           alt.Tooltip("Allowed OPS:Q", title="Arm allows", format=".3f"),
                           alt.Tooltip("Allowed PA:Q", title="On PA", format=".0f"),
                           alt.Tooltip("Arsenal AB:Q", title="His AB", format=".0f"),
                           alt.Tooltip("Split Tag:N", title="Arm's split")] +
                          HITTER_OPS_TOOLTIP))
    labels = _label_layers(scales.carry(plot, data), "Allowed OPS:Q", f"{baseline}:Q",
                           plot[baseline] - plot["Allowed OPS"],
                           *(("Team:N", alt.Scale(domain=teams, range=team_colors(teams)))
                             if teams else (None, alt.Undefined)), team=bool(teams))
    return _styled(_over(parity + points, labels), height, zoom=True)


def edge_bars(frame, column="Arsenal Edge", top=20, min_ab=10, height=None,
              domain=None):
    """The biggest gaps between a hitter's line and what tonight's arm allows his side.

    Sorted by the gap, filtered to hitters with at least a token sample. Zero is marked
    because the number only means anything relative to it: a +.100 edge and a -.100 edge are
    the two ends of the same question, and a bar chart without the rule invites reading the
    longest bar as "best" regardless of side.
    """
    data = _ready(frame, [column, "Name"])
    if data is None:
        return None
    data = data[data.get("Arsenal AB", pd.Series(0, index=data.index)).fillna(0) >= min_ab]
    if data.empty:
        return None
    ranked = (data.reindex(data[column].abs().sort_values(ascending=False).index)
              .head(top).copy())
    ranked["label"] = ranked["Name"].astype(str) + " (" + ranked["Team"].astype(str) + ")"
    order = list(ranked.sort_values(column, ascending=False)["label"])
    height = height or max(240, 25 * len(ranked))

    bars = (alt.Chart(ranked).mark_bar(height=14, cornerRadiusEnd=3)
            .encode(
                x=alt.X(f"{column}:Q", title=f"{column} (OPS above what the arm allows)"),
                y=alt.Y("label:N", title=None, sort=order),
                color=(_team_color(domain) if domain else
                       alt.Color("Signal:N", title="Signal", scale=_signal_scale(),
                                 sort=SIGNAL_ORDER)),
                tooltip=[alt.Tooltip("Name:N"), alt.Tooltip("Team:N"),
                         alt.Tooltip("Opp SP:N", title="Facing"),
                         alt.Tooltip("Effective Side:N", title="Bats tonight"),
                         alt.Tooltip("Arsenal OPS:Q", format=".3f"),
                         alt.Tooltip("Allowed OPS:Q", title="Arm allows", format=".3f"),
                         alt.Tooltip(f"{column}:Q", format="+.3f"),
                         alt.Tooltip("Arsenal AB:Q", title="His AB", format=".0f")] +
                        HITTER_OPS_TOOLTIP))
    zero = (alt.Chart(pd.DataFrame({"x": [0.0]}))
            .mark_rule(color=RULE_GREY, strokeWidth=1).encode(x="x:Q"))
    return _styled(zero + bars, height)


def basis_scatter(frame, basis="Season OPS", basis_label=None, grid=None, height=470,
                  size_by=DEFAULT_SIZE):
    """Matchup score against whatever the reader wants to read it against.

    The fixed version of this chart plotted composite against season OPS, which asks "is a
    good matchup landing on a good bat". That is one question of several, and the others are
    not reachable by squinting at it: against **recent** offence it asks whether the matchup
    is landing on someone in form, and against **xwOBA on this arsenal** whether the
    contact quality backs up the OPS the composite is partly built from.

    The medians are drawn as rules rather than a fitted line. The relationship is not
    reliably linear and a slate is 150 points; quadrants answer "which corner is he in",
    which is the actual question, without asserting a functional form.
    """
    label = basis_label or basis
    data = _ready(frame, ["Composite", basis])
    if data is None:
        return None

    ref_x = scales.reference(data, "Composite")
    ref_y = scales.reference(data, basis)
    mid_x = float(ref_x if ref_x is not None else data["Composite"].median())
    mid_y = float(ref_y if ref_y is not None else data[basis].median())
    median_x = (alt.Chart(pd.DataFrame({"x": [mid_x]}))
                .mark_rule(color=RULE_GREY, strokeDash=[4, 4]).encode(x="x:Q"))
    median_y = (alt.Chart(pd.DataFrame({"y": [mid_y]}))
                .mark_rule(color=RULE_GREY, strokeDash=[4, 4]).encode(y="y:Q"))

    tooltip = [alt.Tooltip("Name:N"), alt.Tooltip("Team:N"),
               alt.Tooltip("game:N", title="Game"),
               alt.Tooltip("Bats:N", title="Bats"),
               alt.Tooltip("Composite:Q", format=".1f"),
               alt.Tooltip(f"{basis}:Q", title=label, format=".3f"),
               alt.Tooltip("Signal:N")] + HITTER_OPS_TOOLTIP
    for extra, fmt in (("Salary", "$,.0f"), ("Ceiling", ".1f"), ("Season AB", ".0f")):
        if extra in data.columns:
            tooltip.append(alt.Tooltip(f"{extra}:Q", title=extra, format=fmt))

    size, _ = _size_channel(data, size_by if size_by in data.columns else "Season AB")
    points = (alt.Chart(data).mark_circle(opacity=0.85, stroke="#fcfcfb", strokeWidth=1.1)
              .encode(
                  x=alt.X("Composite:Q", title="Composite matchup score",
                          scale=_qscale(data, "Composite")),
                  y=alt.Y(f"{basis}:Q", title=label, scale=_qscale(data, basis)),
                  color=alt.Color("Signal:N", title="Signal", scale=_signal_scale(),
                                  sort=SIGNAL_ORDER),
                  size=size, tooltip=tooltip))

    background = hit_background(grid, "Composite matchup score", label)
    layers = [background] if background is not None else []
    layers += [median_x, median_y, points]
    composed = layers[0]
    for layer in layers[1:]:
        composed = composed + layer
    names = _label_layers(data, "Composite:Q", f"{basis}:Q", data["Composite"] - mid_x,
                          team=True)
    return _styled(_over(composed, names), height, zoom=True)


# ===================================================================================
# Click-through
#
# Any point on any chart opens its own evidence. Two things make this work and both are
# easy to get wrong:
#
# **The selection must name its fields.** Streamlit's own documentation is explicit: without
# `fields` or `encodings`, "Vega may add an internal row identifier field (vgsid) to your
# data, and selections can then return this identifier instead of your original data
# values." A click would come back as `{"vgsid": 42}` — a number with no way to recover who
# was clicked, and no error to say so.
#
# **The param belongs on the marks layer, not the composition.** These charts are layered
# (points over reference rules, sometimes over a hit-rate grid). Adding the param to the
# composed chart attaches it to the wrong thing; it has to go on the layer that draws the
# points, which is always the last one.
# ===================================================================================

#: What identifies a point, per kind of chart. A pair rather than a name: two clubs can
#: field players with the same surname on the same slate.
IDENTITY = {
    "hitter": ["Name", "Team"],
    "staff": ["name", "team"],
    "cell": ["x0", "x1", "y0", "y1"],
}


def _marks_layer(layers):
    """Index of the layer carrying the data marks, falling back to the topmost one."""
    for index in range(len(layers) - 1, -1, -1):
        mark = getattr(layers[index], "mark", None)
        kind = mark if isinstance(mark, str) else getattr(mark, "type", None)
        if kind in ("circle", "point", "square", "bar"):
            return index
    return len(layers) - 1


def selectable(chart, kind="hitter", fields=None, name="point"):
    """Make a chart's points clickable, returning the chart unchanged if it cannot be.

    Safe to wrap anything: a `None` chart stays `None`, so a page never has to guard the
    call separately from the builder that produced it.
    """
    if chart is None:
        return None
    fields = fields or IDENTITY.get(kind) or IDENTITY["hitter"]
    picked = alt.selection_point(name=name, fields=list(fields), on="click",
                                 clear="dblclick", empty=True)
    try:
        layers = getattr(chart, "layer", None)
        if layers:
            # The param belongs on the *marks* layer. This used to assume "the points are
            # the top layer", which held until direct labels were added on top -- at which
            # point every click resolved against the text layer and the drill-down opened
            # on whichever handful of names happened to be labelled. Find the circles.
            index = _marks_layer(layers)
            layers[index] = layers[index].add_params(picked)
            return chart
        return chart.add_params(picked)
    except Exception:
        # A chart shape that will not take a param is still worth showing.
        return chart


def selected(event, name="point"):
    """The rows a reader clicked, as a list of dicts. Empty when nothing is selected."""
    if not event:
        return []
    selection = event.get("selection") if hasattr(event, "get") else None
    if not selection:
        return []
    picked = selection.get(name)
    if isinstance(picked, dict):
        # A field-less param comes back keyed by field name rather than as rows; treated as
        # nothing selected, because it cannot identify a player.
        return []
    return list(picked or [])


# ===================================================================================
# Team as colour, signal as shape
#
# The default elsewhere is the reverse, and for a reason: scatter is an all-pairs form, so
# colour caps at about three series before hues stop being separable, and a slate is thirty
# clubs. What makes this worth doing anyway is that the question being asked changes. "Which
# club's bats cluster in the good corner" is a *stacking* question, and clustering survives a
# palette that cannot name every member — the eye finds the group without resolving which
# hue is which.
#
# So this is honest about what it delivers: colour for **spotting a cluster**, the team
# filter for naming one, and the tooltip for certainty. Signal moves to shape, where four
# ordered categories fit comfortably.
#
# **Colour follows the club, never its rank.** The domain is every team on the slate, not
# the filtered subset, so narrowing the filter cannot repaint the survivors — a club keeps
# its hue as the view changes, which is the whole basis for recognising it.
# ===================================================================================

#: Signal as shape. Four categories, ordered, and the glyphs get heavier as the call gets
#: stronger so the ordering survives in greyscale and for a colour-blind reader.
SIGNAL_SHAPES = {
    "Priority": "triangle-up",
    "Watch": "square",
    "Neutral": "circle",
    "Fade": "triangle-down",
}

#: Above this many clubs, colour can only carry clustering, not identity.
TEAM_IDENTITY_CAP = 10


def _signal_shape_scale():
    return alt.Scale(domain=SIGNAL_ORDER,
                     range=[SIGNAL_SHAPES[s] for s in SIGNAL_ORDER])


def team_domain(frame, column="Team"):
    """Every club in the frame, in a fixed order.

    Sorted by name rather than by any measure, so the hue assignment is stable across
    reruns and independent of what the filters did.
    """
    if frame is None or column not in getattr(frame, "columns", []):
        return []
    return sorted(str(t) for t in frame[column].dropna().unique())


def _team_color(domain, legend=alt.Undefined):
    """Each club in its own identity colour, over a fixed domain.

    An explicit `range` rather than a scheme, because the point of colouring by team is
    recognition — a club has to keep the colour a reader already associates with it, which
    no generated palette can do.
    """
    return alt.Color("Team:N", title="Team", legend=legend,
                     scale=alt.Scale(domain=domain, range=team_colors(domain)),
                     sort=domain)


def team_signal_scatter(frame, x, y, x_title=None, y_title=None, grid=None,
                        size_by=DEFAULT_SIZE, domain=None, height=470,
                        x_format=None, tooltip=None):
    """A scatter with colour on the club and shape on the signal.

    `domain` should be every club on the slate, passed in by the caller, so filtering to two
    teams does not hand them somebody else's colours.
    """
    data = _ready(frame, [x, y])
    if data is None:
        return None
    domain = domain or team_domain(data)
    axis = alt.Axis(format=x_format) if x_format else alt.Undefined

    tips = tooltip or [
        alt.Tooltip("Name:N"), alt.Tooltip("Team:N"), alt.Tooltip("Signal:N"),
        alt.Tooltip(f"{x}:Q", title=x_title or x, format=",.2f"),
        alt.Tooltip(f"{y}:Q", title=y_title or y, format=",.2f"),
    ] + HITTER_OPS_TOOLTIP
    for extra, fmt in (("Salary", "$,.0f"), ("Proj", ".1f"), ("Ceiling", ".1f"),
                       ("Composite", ".1f"), ("Slot", ".0f")):
        if extra in data.columns and extra not in (x, y):
            tips.append(alt.Tooltip(f"{extra}:Q", title=extra, format=fmt))

    points = (alt.Chart(data).mark_point(filled=True, opacity=0.85,
                                         stroke="#fcfcfb", strokeWidth=1.1)
              .encode(
                  x=alt.X(f"{x}:Q", title=x_title or x, scale=_qscale(data, x),
                          axis=axis),
                  y=alt.Y(f"{y}:Q", title=y_title or y, scale=_qscale(data, y)),
                  color=_team_color(domain),
                  shape=alt.Shape("Signal:N", title="Signal", scale=_signal_shape_scale(),
                                  sort=SIGNAL_ORDER),
                  size=_size_channel(data, size_by, (50, 480))[0],
                  tooltip=tips))

    # Names carry the club too. This is the panel where colour is admittedly only good for
    # spotting a cluster, so the thing a zoom has to deliver is the club, not just the bat.
    labels = _label_layers(data, f"{x}:Q", f"{y}:Q", data[y] - data[y].median(),
                           "Team:N", alt.Scale(domain=domain, range=team_colors(domain)),
                           team=True)
    background = hit_background(grid, x_title or x, y_title or y)
    return _styled(_over(points if background is None else background + points, labels),
                   height, zoom=True)


# ===================================================================================
# What he allows against what they hit
#
# Two OPS figures on one scale, which is the only way the comparison means anything. The
# diagonal is parity: on it, the lineup hits exactly what this arm normally gives up.
#
# The platoon views are where the read is. A left-hander who surrenders .654 to lefties is
# only in a good spot if he actually draws left-handed bats — and managers change lineup
# construction *because* of him. So the point is sized by how many bats of that side are in
# the card, and a small point on a favourable-looking split means the split barely applies.
# ===================================================================================

#: The three views, and which columns each one reads.
VS_LINEUP_VIEWS = {
    "Overall": ("Allowed OPS", "Lineup OPS", "lineup_bats",
                "OPS he surrenders (blended by lineup handedness)",
                "OPS the lineup hits"),
    "vs left-handed bats": ("Allowed vs L", "Lineup vs L", "Bats vs L",
                            "OPS he surrenders to lefties",
                            "OPS the lineup's lefties hit"),
    "vs right-handed bats": ("Allowed vs R", "Lineup vs R", "Bats vs R",
                             "OPS he surrenders to righties",
                             "OPS the lineup's righties hit"),
}

#: Throwing hand. Two series, so the validated pair carries it.
THROWS_COLORS = {"R": "#2a78d6", "L": "#eb6834"}


def pitcher_vs_lineup_scatter(frame, view="Overall", height=480):
    """A starter's surrendered OPS against the OPS of the lineup he draws.

    Above the diagonal the lineup out-hits what this arm normally allows, which is the spot
    worth attacking. Size is the number of bats the split actually applies to — a
    favourable-looking platoon split over two bats is not a matchup.
    """
    x, y, size_column, x_title, y_title = VS_LINEUP_VIEWS.get(
        view, VS_LINEUP_VIEWS["Overall"])
    data = _ready(frame, [x, y])
    if data is None:
        return None
    data = data[pd.to_numeric(data[size_column], errors="coerce").fillna(0) > 0]
    if data.empty:
        return None

    domain = _same_unit_domain(data, [x, y], floor=0.0)

    parity = (alt.Chart(pd.DataFrame({"v": domain}))
              .mark_line(color=RULE_GREY, strokeDash=[5, 4], strokeWidth=1.5)
              .encode(x=alt.X("v:Q", scale=alt.Scale(domain=domain, clamp=True)),
                      y=alt.Y("v:Q", scale=alt.Scale(domain=domain, clamp=True))))

    points = (alt.Chart(data).mark_point(filled=True, opacity=0.85,
                                         stroke="#fcfcfb", strokeWidth=1.2)
              .encode(
                  x=alt.X(f"{x}:Q", title=x_title,
                          scale=alt.Scale(domain=domain, clamp=True)),
                  y=alt.Y(f"{y}:Q", title=y_title,
                          scale=alt.Scale(domain=domain, clamp=True)),
                  color=alt.Color("throws:N", title="Throws",
                                  scale=alt.Scale(domain=list(THROWS_COLORS),
                                                  range=list(THROWS_COLORS.values()))),
                  size=_size_channel(data, size_column, (60, 600),
                                     "Bats it applies to")[0],
                  tooltip=[alt.Tooltip("pitcher:N", title="Starter"),
                           alt.Tooltip("throws:N", title="Throws"),
                           alt.Tooltip("faces:N", title="Faces"),
                           alt.Tooltip("game:N", title="Game"),
                           alt.Tooltip(f"{x}:Q", title="He allows", format=".3f"),
                           alt.Tooltip(f"{y}:Q", title="They hit", format=".3f"),
                           alt.Tooltip("L bats:Q", title="LHB in card", format=".0f"),
                           alt.Tooltip("R bats:Q", title="RHB in card", format=".0f"),
                           alt.Tooltip("switch bats:Q", title="Switch", format=".0f"),
                           alt.Tooltip("Allowed PA vs L:Q", title="His PA vs L",
                                       format=".0f"),
                           alt.Tooltip("Allowed PA vs R:Q", title="His PA vs R",
                                       format=".0f")]))
    labels = _label_layers(data, f"{x}:Q", f"{y}:Q", data[y] - data[x],
                           name_column="pitcher")
    return _styled(_over(parity + points, labels), height, zoom=True)


def platoon_exposure_bars(frame, height=None):
    """How lopsided each lineup is against the hand it faces.

    The lineup-construction question on its own axis: a lefty drawing a card with three
    left-handed bats is being avoided, and that is visible here in a way a scatter of two
    OPS figures cannot show. Positive means the card leans toward the side this arm handles
    *worse*, so the bar direction is already the read.
    """
    data = _ready(frame, ["Bats vs L", "Bats vs R", "Allowed vs L", "Allowed vs R"])
    if data is None:
        return None
    plot = data.copy()
    # Which side he handles worse, and how much of the card is on it.
    worse_is_left = plot["Allowed vs L"] > plot["Allowed vs R"]
    plot["exposed bats"] = np.where(worse_is_left, plot["Bats vs L"], plot["Bats vs R"])
    plot["sheltered bats"] = np.where(worse_is_left, plot["Bats vs R"], plot["Bats vs L"])
    plot["exposure"] = plot["exposed bats"] - plot["sheltered bats"]
    plot["worse side"] = np.where(worse_is_left, "vs L", "vs R")
    plot["label"] = plot["pitcher"].astype(str) + " (" + plot["throws"].astype(str) + ")"
    order = list(plot.sort_values("exposure", ascending=False)["label"])
    height = height or max(240, 25 * len(plot))

    bars = (alt.Chart(plot).mark_bar(height=14, cornerRadiusEnd=3)
            .encode(
                x=alt.X("exposure:Q",
                        title="Bats on his weaker side, minus bats on his stronger side"),
                y=alt.Y("label:N", title=None, sort=order),
                color=alt.Color("throws:N", title="Throws",
                                scale=alt.Scale(domain=list(THROWS_COLORS),
                                                range=list(THROWS_COLORS.values()))),
                tooltip=[alt.Tooltip("pitcher:N", title="Starter"),
                         alt.Tooltip("faces:N", title="Faces"),
                         alt.Tooltip("worse side:N", title="Weaker split"),
                         alt.Tooltip("Allowed vs L:Q", title="Allows vs L", format=".3f"),
                         alt.Tooltip("Allowed vs R:Q", title="Allows vs R", format=".3f"),
                         alt.Tooltip("exposed bats:Q", title="Bats on weaker side",
                                     format=".0f"),
                         alt.Tooltip("sheltered bats:Q", title="Bats on stronger side",
                                     format=".0f")]))
    zero = (alt.Chart(pd.DataFrame({"x": [0.0]}))
            .mark_rule(color=RULE_GREY, strokeWidth=1).encode(x="x:Q"))
    return _styled(zero + bars, height)


# ===================================================================================
# Club colours
#
# Each club's own identity colour, which is what makes a cluster recognisable without
# consulting a legend — the entire reason to colour by team.
#
# **Where a club's primary collides with a division rival's, the secondary is used instead.**
# MLB primaries are heavily navy and heavily red: eight clubs sit within a few ΔE of navy and
# six near the same red, and a scatter cannot distinguish them. So `TEAM_IDENTITY` below is
# the club's *recognisable* colour rather than strictly its first one — Boston keeps red while
# the Angels take their secondary, the Yankees keep navy while Milwaukee takes gold. Each
# substitution is a colour the club genuinely wears.
#
# **Substitution was not nearly enough, and `TEAM_COLORS` is what the charts actually use.**
# Measured in OKLab, the closest pair in the identity map sits 0.006 apart — the Mets and the
# Giants wear, to a scatter, the same orange — and five more pairs sit under 0.03. At that
# distance two clubs are not similar, they are indistinguishable, and the panel is lying about
# having thirty series on it.
#
# So the palette below spends **lightness** on separation and spends nothing else. Each club
# keeps its own hue to within a degree — a red club is red, a navy club is navy, which is all
# the recognition a colour was ever carrying here — and within each hue family the clubs are
# spread across a lightness band by the rank their identity colour already had. The Yankees
# stay the darkest blue; Tampa Bay stays the palest. Chroma is then taken to just under the
# most sRGB can show at that lightness and hue, because a washed-out mark is a third way to
# lose a club.
#
# That moves the closest pair from 0.006 to 0.057 and the fifth-closest from 0.028 to 0.061 —
# roughly a ninefold gain on the worst case. The band is 0.42–0.84, chosen so nothing goes
# black on the dark theme or vanishes into white on the light one.
#
# It still does not pretend to be a validated categorical palette, because none exists at this
# cardinality: thirty series is far beyond what colour separates. What it now delivers is that
# no two clubs read as *the same*. The caption on each page still says colour is for spotting
# a cluster, the tooltip still names the club, and zooming in now names it on the chart.
# ===================================================================================

#: What each club actually wears — the recognition anchor, and the hue every entry in
#: `TEAM_COLORS` is derived from. Kept separate so the derivation stays inspectable.
TEAM_IDENTITY = {
    "ARI": "#A71930",   # Sedona red
    "ATH": "#003831",   # Athletics green
    "ATL": "#CE1141",   # Atlanta red
    "BAL": "#DF4601",   # Orioles orange
    "BOS": "#BD3039",   # Boston red
    "CHC": "#0E3386",    # Cubs blue
    "CIN": "#C6011F",   # Reds red
    "CLE": "#00385D",   # Guardians navy
    "COL": "#33006F",   # Rockies purple
    "CWS": "#27251F",   # White Sox black
    "DET": "#FA4616",   # Tigers orange (secondary; their navy is identical to NYY's)
    "HOU": "#EB6E1F",   # Astros orange
    "KC": "#004687",    # Royals blue
    "LAA": "#BA0021",   # Angels red
    "LAD": "#005A9C",   # Dodger blue
    "MIA": "#00A3E0",   # Marlins blue
    "MIL": "#FFC52F",   # Brewers gold (secondary; navy collides with NYY/DET)
    "MIN": "#002B5C",   # Twins navy
    "NYM": "#FF5910",   # Mets orange
    "NYY": "#0C2340",   # Yankees navy
    "PHI": "#E81828",   # Phillies red
    "PIT": "#FDB827",   # Pirates gold
    "SD": "#2F241D",    # Padres brown
    "SEA": "#0C2C56",   # Mariners navy
    "SF": "#FD5A1E",    # Giants orange
    "STL": "#C41E3A",   # Cardinals red
    "TB": "#8FBCE6",    # Rays light blue (secondary; navy collides)
    "TEX": "#003278",   # Rangers blue
    "TOR": "#134A8E",   # Blue Jays blue
    "WSH": "#AB0003",   # Nationals red
}

#: The drawing palette: `TEAM_IDENTITY` with lightness spread within each hue family. See the
#: block comment above for why, and `tests/test_dashboard_filters.py` for the separation this
#: is required to hold.
TEAM_COLORS = {
    "ARI": "#A30429",   # from #A71930, lightness 0.47 -> 0.45
    "ATH": "#015A50",   # from #003831, lightness 0.31 -> 0.42
    "ATL": "#FB0F50",   # from #CE1141, lightness 0.54 -> 0.63
    "BAL": "#FD6939",   # from #DF4601, lightness 0.61 -> 0.70
    "BOS": "#EA0D35",   # from #BD3039, lightness 0.53 -> 0.59
    "CHC": "#5188FD",   # from #0E3386, lightness 0.36 -> 0.65
    "CIN": "#C70821",   # from #C6011F, lightness 0.52 -> 0.52
    "CLE": "#077EC6",   # from #00385D, lightness 0.33 -> 0.57
    "COL": "#5B05BB",   # from #33006F, lightness 0.29 -> 0.42
    "CWS": "#025F84",   # from #27251F, lightness 0.26 -> 0.46 (hue from their silver)
    "DET": "#FE7F61",   # from #FA4616, lightness 0.65 -> 0.74
    "HOU": "#FE955F",   # from #EB6E1F, lightness 0.68 -> 0.77
    "KC": "#479CFD",    # from #004687, lightness 0.40 -> 0.69
    "LAA": "#B50621",   # from #BA0021, lightness 0.50 -> 0.49
    "LAD": "#72B8FE",   # from #005A9C, lightness 0.46 -> 0.76
    "MIA": "#6BCAFE",   # from #00A3E0, lightness 0.67 -> 0.80
    "MIL": "#FCC015",   # from #FFC52F, lightness 0.85 -> 0.84
    "MIN": "#0360BD",   # from #002B5C, lightness 0.29 -> 0.50
    "NYM": "#FEB7A0",   # from #FF5910, lightness 0.68 -> 0.84
    "NYY": "#004B97",   # from #0C2340, lightness 0.26 -> 0.42
    "PHI": "#FD4847",   # from #E81828, lightness 0.59 -> 0.66
    "PIT": "#B27F08",   # from #FDB827, lightness 0.83 -> 0.63
    "SD": "#624900",    # from #2F241D, lightness 0.27 -> 0.42 (hue from their gold)
    "SEA": "#0468D5",   # from #0C2C56, lightness 0.30 -> 0.53
    "SF": "#FEA68A",    # from #FD5A1E, lightness 0.68 -> 0.81
    "STL": "#D80A3B",   # from #C41E3A, lightness 0.53 -> 0.56
    "TB": "#9FD0FE",    # from #8FBCE6, lightness 0.78 -> 0.84
    "TEX": "#257DFD",   # from #003278, lightness 0.34 -> 0.61
    "TOR": "#69A8FD",   # from #134A8E, lightness 0.41 -> 0.73
    "WSH": "#940304",   # from #AB0003, lightness 0.47 -> 0.42
}

#: Anything the map does not know — an abbreviation that changed, a spring-training club.
#: Grey rather than a generated hue, so an unmapped club is visibly unmapped instead of
#: quietly borrowing somebody else's identity. It is also the one entry with no chroma at
#: all, which is what keeps "we do not know this club" from reading as a thirty-first club.
UNKNOWN_TEAM_COLOR = "#9aa0a6"


def team_colors(domain):
    """Colours for a fixed team domain, in the same order."""
    return [TEAM_COLORS.get(str(team).upper(), UNKNOWN_TEAM_COLOR) for team in domain]


# ===================================================================================
# Stacks
#
# A club is the unit here, not a hitter, so the label goes *on* the mark. Thirty points that
# each need a hover to identify defeats the purpose — at this cardinality the club name fits,
# and a scatter of named clubs is read directly rather than decoded.
# ===================================================================================

#: What a stack can be plotted against. All from the pipeline's own stack row or the
#: conditions join; none of them derived here.
STACK_AXES = {
    "Top5 Salary": "Cost of the top five",
    "Top5 Proj": "Projected points, top five",
    "Top5 Ceiling": "Ceiling, top five",
    "Stack Value": "Ceiling per $1k",
    "Stack Score": "Stack score (value + ceiling)",
    "Team Runs": "Implied team runs",
    "Allowed OPS": "OPS the opposing starter surrenders",
    "Lineup OPS": "OPS this lineup hits",
    "hr_env": "Home-run environment (park × weather)",
    "Mean composite": "Mean composite of the club's bats",
}


def stack_scatter(frame, x="Top5 Salary", y="Top5 Ceiling", size_by="Team Runs",
                  height=520, labels=True):
    """Clubs as stack units, each drawn where its top five sits.

    Labelled rather than legended: thirty clubs is far past what a colour legend can
    resolve, but it is well inside what a chart can *name*, and naming removes the decoding
    step entirely. Colour still carries club identity so a reader who knows the palette does
    not need to read at all.
    """
    data = _ready(frame, [x, y])
    if data is None:
        return None
    domain = team_domain(data)

    tips = [alt.Tooltip("Team:N"), alt.Tooltip("Opp:N", title="Opponent"),
            alt.Tooltip("Opp SP:N", title="Facing"),
            alt.Tooltip("Opp SP Hand:N", title="Throws")]
    for column, fmt in (("Top5 Salary", "$,.0f"), ("Top5 Proj", ".1f"),
                        ("Top5 Ceiling", ".1f"), ("Stack Value", ".2f"),
                        ("Stack Score", ".1f"), ("Team Runs", ".2f"),
                        ("Allowed OPS", ".3f"), ("Lineup OPS", ".3f"),
                        ("hr_env", ".2f"), ("Hitters", ".0f")):
        if column in data.columns:
            tips.append(alt.Tooltip(f"{column}:Q", title=STACK_AXES.get(column, column),
                                    format=fmt))
    if "park" in data.columns:
        tips.append(alt.Tooltip("park:N", title="Park"))

    base = alt.Chart(data).encode(
        x=alt.X(f"{x}:Q", title=STACK_AXES.get(x, x), scale=_qscale(data, x),
                axis=alt.Axis(format="$,.0f") if "Salary" in x else alt.Undefined),
        y=alt.Y(f"{y}:Q", title=STACK_AXES.get(y, y), scale=_qscale(data, y)),
        tooltip=tips)

    points = base.mark_point(filled=True, opacity=0.9, stroke="#fcfcfb",
                             strokeWidth=1.2).encode(
        color=_team_color(domain, legend=None),
        size=_size_channel(data, size_by, (110, 700))[0])

    if not labels:
        return _styled(points, height, zoom=True)
    # Offset so the label sits clear of its own mark rather than on top of it.
    text = base.mark_text(align="left", dx=9, dy=-9, fontSize=11,
                          fontWeight="bold").encode(
        text="Team:N", color=_team_color(domain, legend=None))
    return _styled(points + text, height, zoom=True)


def stack_bars(frame, column="Stack Score", top=None, height=None):
    """Clubs ranked on one stack measure, in their own colours.

    A bar per club is the plain version of the same question, and it is the right form when
    the reader wants an ordering rather than a relationship.
    """
    data = _ready(frame, [column, "Team"])
    if data is None:
        return None
    ranked = data.sort_values(column, ascending=False)
    if top:
        ranked = ranked.head(top)
    domain = team_domain(ranked)
    order = list(ranked["Team"])
    height = height or max(240, 24 * len(ranked))

    tips = [alt.Tooltip("Team:N"), alt.Tooltip("Opp SP:N", title="Facing")]
    for extra, fmt in (("Top5 Salary", "$,.0f"), ("Top5 Ceiling", ".1f"),
                       ("Stack Value", ".2f"), ("Team Runs", ".2f"),
                       ("Allowed OPS", ".3f"), ("hr_env", ".2f")):
        if extra in ranked.columns:
            tips.append(alt.Tooltip(f"{extra}:Q", title=STACK_AXES.get(extra, extra),
                                    format=fmt))

    bars = (alt.Chart(ranked).mark_bar(height=15, cornerRadiusEnd=3)
            .encode(x=alt.X(f"{column}:Q", title=STACK_AXES.get(column, column)),
                    y=alt.Y("Team:N", title=None, sort=order),
                    color=_team_color(domain),
                    tooltip=tips))
    return _styled(bars, height)


def stack_member_bars(frame, height=None):
    """The five bats a stack is made of, against the rest of the card.

    The drill-down for a stack row. Colour is *in the stack or not* rather than club
    identity, because within one club identity is not the question — which bats were taken
    is.
    """
    data = _ready(frame, ["Proj", "Name"])
    if data is None:
        return None
    plot = data.copy()
    plot["Part of stack"] = np.where(plot.get("In stack", False), "Top five", "Rest of card")
    order = list(plot.sort_values("Slot")["Name"])
    height = height or max(240, 26 * len(plot))
    return _styled(
        alt.Chart(plot).mark_bar(height=16, cornerRadiusEnd=3)
        .encode(
            x=alt.X("Proj:Q", title="Projected DK points"),
            y=alt.Y("Name:N", title=None, sort=order),
            color=alt.Color("Part of stack:N", title=None,
                            scale=alt.Scale(domain=["Top five", "Rest of card"],
                                            range=["#2a78d6", "#c9ccd1"])),
            tooltip=[alt.Tooltip("Slot:Q", title="Order", format=".0f"),
                     alt.Tooltip("Name:N"), alt.Tooltip("Pos:N"), alt.Tooltip("Bats:N"),
                     alt.Tooltip("Salary:Q", format="$,.0f"),
                     alt.Tooltip("Proj:Q", format=".2f"),
                     alt.Tooltip("Ceiling:Q", format=".2f"),
                     alt.Tooltip("PA:Q", title="Expected PA", format=".2f")] +
                    HITTER_OPS_TOOLTIP),
        height)


# ===================================================================================
# Strikeouts
#
# Three panels that answer one question — can this arm actually miss these bats — from
# three samples that never share an axis in the workbook. The reference line is league K%
# on all of them, because the interesting range is only about eight points wide and a
# strikeout panel without an anchor makes every pitcher look average.
# ===================================================================================


def k_recent_starts(frame, season_k=None, lineup_k=None, league=22.5, height=260):
    """K% start by start, with his season rate and the card he draws as reference lines.

    A line, not bars: these are ordered in time and the question is which way he is
    trending. Points are kept on the line because five starts is few enough that each one
    is a datum a reader will want to hover.
    """
    data = _ready(frame, ["Date", "K%"])
    if data is None:
        return None
    ordered = data.sort_values("Date")
    base = alt.Chart(ordered)
    line = base.mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("Date:N", title=None, sort=list(ordered["Date"])),
        y=alt.Y("K%:Q", title="K% for the start", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("Date:N"), alt.Tooltip("Opponent:N", title="Opponent"),
                 alt.Tooltip("IP:Q", format=".2f"), alt.Tooltip("SO:Q", title="SO"),
                 alt.Tooltip("K%:Q", format=".1f"), alt.Tooltip("K/9:Q", format=".2f")])
    layers = [line]
    for value, dash in ((league, [2, 4]), (season_k, [6, 3]), (lineup_k, [1, 3])):
        if value is not None and not pd.isna(value):
            layers.append(alt.Chart(pd.DataFrame({"y": [float(value)]}))
                          .mark_rule(color=RULE_GREY, strokeDash=dash, strokeWidth=1,
                                     opacity=0.75)
                          .encode(y="y:Q"))
    return _styled(alt.layer(*layers), height)


def k_platoon_bars(frame, league=22.5, height=200):
    """His K% by side of the plate, sized by how many of those bats are in tonight's card.

    The count is the point. An arm who buries left-handers is not a strikeout play against
    a card with two of them, and a bare split rate cannot say so.
    """
    data = _ready(frame, ["Batter Side", "K%"])
    if data is None:
        return None
    bars = (alt.Chart(data).mark_bar(cornerRadiusEnd=3, height=22)
            .encode(
                y=alt.Y("Batter Side:N", title=None, sort=["L", "R"]),
                x=alt.X("K%:Q", title="K% against that side", scale=alt.Scale(zero=True)),
                # No colour: the y axis already says L or R, so a hue would repeat the
                # label and would have to borrow the team palette for something that is not
                # team identity. Streamlit themes an unstyled mark for us.
                tooltip=[alt.Tooltip("Batter Side:N", title="Bats"),
                         alt.Tooltip("K%:Q", format=".1f"),
                         alt.Tooltip("PA:Q", title="Split PA", format=".0f"),
                         alt.Tooltip("Bats faced:Q", title="In tonight's card")]))
    labels = (alt.Chart(data).mark_text(align="left", dx=6, fontSize=11, fontWeight=600)
              .encode(y=alt.Y("Batter Side:N", sort=["L", "R"]), x=alt.X("K%:Q"),
                      text=alt.Text("Bats faced:Q", format=".0f")))
    rule = (alt.Chart(pd.DataFrame({"x": [league]}))
            .mark_rule(color=RULE_GREY, strokeDash=[2, 4], strokeWidth=1, opacity=0.75)
            .encode(x="x:Q"))
    return _styled(bars + rule + labels, height)


def k_lineup_bars(frame, league=22.5, height=340):
    """Each opposing hitter's K% against this arsenal, worst bat at the top.

    Sorted rather than ordered by lineup slot: the question this panel answers is "who can
    he actually put away", and that is a ranking.
    """
    data = _ready(frame, ["Name", "K%"])
    if data is None:
        return None
    ordered = data.sort_values("K%", ascending=False)
    tooltip = ([alt.Tooltip("Name:N", title="Hitter"),
                alt.Tooltip("K%:Q", format=".1f")] + HITTER_OPS_TOOLTIP)
    for column, title in (("K% vs Hand", "K% vs this hand"), ("K Edge", "K edge"),
                          ("Whiff%", "Whiff%"), ("PA", "PA vs arsenal")):
        if column in ordered.columns:
            tooltip.append(alt.Tooltip(f"{column}:Q", title=title,
                                       format=".0f" if column == "PA" else ".1f"))
    bars = (alt.Chart(ordered).mark_bar(cornerRadiusEnd=3)
            .encode(y=alt.Y("Name:N", title=None, sort=list(ordered["Name"])),
                    # Bar length already *is* the K%. A colour ramp on the same field says
                    # it twice and buys nothing -- the same reason the surplus bars on the
                    # Value page carry no ramp.
                    x=alt.X("K%:Q", title="K% against this arsenal"),
                    tooltip=tooltip))
    rule = (alt.Chart(pd.DataFrame({"x": [league]}))
            .mark_rule(color=RULE_GREY, strokeDash=[2, 4], strokeWidth=1, opacity=0.75)
            .encode(x="x:Q"))
    return _styled(bars + rule, height)
