"""Axis bounds and pan/zoom, shared by both chart modules.

Both halves of this exist because a dashboard is not a picture. A picture is looked at once;
a dashboard is *switched* -- filters narrow, a measure flips, the next game loads -- and what
the reader is doing is comparing the before to the after.

**Bounds.** Every panel used to size its axes to whatever rows survived the filters, which
is the right default for one chart and the wrong one for a sequence of them: narrowing
Signal to Priority redrew the axes underneath the points, so the two things being compared
both moved at once and neither position meant anything. Two rules fix it.

* Bounds come from the **unfiltered** board. A filter changes which points are drawn and
  never where they land. The page is the only thing that knows what the honest reference is
  -- the whole priced board, the whole card, the three measures a control switches between --
  so the page is what calls `anchor`, not the builder.
* In **Focus**, the high-use comparison measures have versioned, fixed windows. Stepping to
  the next refresh therefore cannot move an OPS or Composite axis; an extreme is clamped to
  the boundary and remains exact in its tooltip. In **Full**, bounds are rounded outward to
  a round step and include the entire reference board.

The bounds ride on `DataFrame.attrs`, which is what keeps this from being a twentieth
keyword argument on twenty builders: a page anchors once and everything it feeds downstream
picks them up.

**Zoom.** A fixed axis is only liveable if the reader can open up a crowded corner, so
`binding()` hands back the scale-bound interval that does it -- drag pans, wheel zooms,
double-click resets. `zoomed_in()` is the companion predicate: a Vega expression that turns
true once the visible span has shrunk, which is how a chart knows it now has the room to
name more than its four extremes.

Why that predicate is safe to lean on: a scale-bound interval compiles to a top-level Vega
signal of the selection's own name whose value is `{}` until the reader touches the chart and
`{"<field>": [lo, hi]}` afterwards. `isValid` on the field is therefore both the "has anyone
zoomed yet" test and the guard that keeps a chart whose x field never made it into the
binding quiet rather than broken.
"""

import math

import altair as alt
import pandas as pd

#: Where the scale metadata lives on a frame. These keys deliberately remain ordinary
#: pandas attrs: charts can be tested without a Streamlit runtime and a page only has to
#: anchor its frame once.
ATTRS_KEY = "axis_bounds"
REFERENCE_KEY = "axis_reference"
EXTENTS_KEY = "scale_extents"
MODE_KEY = "axis_mode"
META_KEYS = (ATTRS_KEY, REFERENCE_KEY, EXTENTS_KEY, MODE_KEY)

#: Stable comparison windows for the measures readers switch between most often. They are
#: deliberately broad enough for normal MLB slates and narrow enough that a ten-point
#: Composite move or a .100 OPS gap remains visible. An extreme is clamped to the edge in
#: Focus mode (with its exact value still in the tooltip); Full mode always uses the raw
#: reference range.
FOCUS_DOMAINS = {
    "Composite": (-60.0, 60.0),
    "Season OPS": (0.25, 1.60),
    "Platoon OPS": (0.25, 1.60),
    "Arsenal OPS": (0.25, 1.60),
    "Allowed OPS": (0.25, 1.60),
    "Off L28": (40.0, 180.0),
    "Salary": (2000.0, 7000.0),
}
POLICY_VERSION = 1

#: Page-family overrides. A field name alone is not always enough semantic context: an
#: individual hitter's OPS needs a much wider range than a nine-man lineup aggregate, and
#: three percentages do not belong on one unioned scale merely because they share a unit.
STACK_FOCUS_DOMAINS = {
    "Top5 Salary": (15000.0, 27000.0),
    "Top5 Proj": (30.0, 45.0),
    "Top5 Ceiling": (75.0, 100.0),
    "Stack Value": (3.0, 5.5),
    "Stack Score": (0.0, 100.0),
    "Team Runs": (2.5, 6.5),
    "Allowed OPS": (0.50, 0.95),
    "Lineup OPS": (0.65, 0.85),
    "hr_env": (0.94, 1.20),
    "Mean composite": (-10.0, 30.0),
}

CONDITIONS_FOCUS_DOMAINS = {
    "FB%": (18.0, 38.0),
    "hr_env": (0.94, 1.20),
    "GB+LD%": (55.0, 80.0),
    "GB%": (30.0, 65.0),
    "LD%": (15.0, 30.0),
    "park_2b": (75.0, 130.0),
    "park_r": (85.0, 125.0),
    "park_hr": (0.90, 1.20),
}

#: Zero here means "no salary file joined", not a free five-man stack. It must remain in
#: tables as missing-input evidence but must never determine an axis or size scale.
POSITIVE_ONLY = {"Top5 Salary"}

#: Measures whose real scale cannot go below zero. Applied to Full as well as Focus so a
#: zero observation does not buy a decorative negative band merely because padding is on.
FLOORS = {
    "Season OPS": 0.0, "Platoon OPS": 0.0, "Arsenal OPS": 0.0,
    "Allowed OPS": 0.0, "Off L28": 0.0, "Salary": 0.0,
    "Proj": 0.0, "Ceiling": 0.0, "Floor": 0.0,
}

MODES = ("focus", "full")
SESSION_KEY = "axis_range_mode"

#: Name of the pan/zoom parameter. Charts and the label predicate have to agree on it, so it
#: is spelled once.
ZOOM = "zoom"

#: Fraction of a span to add before snapping outward, so the extreme point does not land on
#: the axis line itself. Small on purpose: snapping to a round step already leaves a margin,
#: and padding twice is what turns a .41-.99 range into a 0.2-1.2 axis with a third of the
#: plot empty.
PAD = 0.02

#: Roughly how many steps a rounded axis should span.
STEPS = 8


def selected_mode(state=None):
    """Normalise the global UI choice without making this pure chart module import Streamlit."""
    value = (state or {}).get(SESSION_KEY, "Focus")
    chosen = str(value).strip().lower()
    return chosen if chosen in MODES else "focus"


def _step(span):
    """A round step of about `span / STEPS` -- one of 1, 2, 2.5 or 5 times a power of ten.

    The *nearest* such step in log space rather than the first one at least that big. Always
    rounding up costs a whole extra decade of empty axis when the ideal step falls just above
    a ladder rung -- a $2,000-$6,500 board wants a $500 tick and was being given $1,000, which
    pushed the axis down to $1,000 and left the cheap third of it empty.
    """
    raw = span / STEPS
    magnitude = 10.0 ** math.floor(math.log10(raw))
    candidates = [factor * magnitude for factor in (1.0, 2.0, 2.5, 5.0, 10.0)]
    return min(candidates, key=lambda step: abs(math.log(step / raw)))


def nice(low, high, pad=PAD):
    """`(low, high)` padded and snapped outward to a round step, or None if it cannot be.

    None rather than a degenerate pair when the values are equal or unusable: a zero-width
    domain is not an axis, and the builder's own `zero=False` default is a better answer
    than a made-up one.
    """
    try:
        low, high = float(low), float(high)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(low) and math.isfinite(high)) or high <= low:
        return None
    span = high - low
    low -= span * pad
    high += span * pad
    step = _step(high - low)
    if not math.isfinite(step) or step <= 0:
        return None
    return (math.floor(low / step) * step, math.ceil(high / step) * step)


def measure(frame, columns=None, pad=PAD):
    """Rounded bounds for every numeric column named, skipping any that cannot carry one."""
    if frame is None or getattr(frame, "empty", True):
        return {}
    columns = list(columns) if columns is not None else list(frame.columns)
    found = {}
    for column in columns:
        if column not in frame.columns:
            continue
        values = _values(frame, column)
        if values.empty:
            continue
        bounds = nice(values.min(), values.max(), pad)
        if bounds:
            found[column] = bounds
    return found


def _values(frame, column):
    """Finite numeric values in one column, shared by reference stats and extents."""
    if frame is None or column not in getattr(frame, "columns", []):
        return pd.Series(dtype="float64")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    values = values[values.map(lambda value: math.isfinite(float(value)))]
    if column in POSITIVE_ONLY:
        values = values[values > 0]
    return values


def _reference_stats(frame, columns=None):
    columns = list(columns) if columns is not None else list(getattr(frame, "columns", []))
    found = {}
    for column in columns:
        values = _values(frame, column)
        if not values.empty:
            found[column] = {"median": float(values.median())}
    return found


def _extents(frame, columns=None):
    columns = list(columns) if columns is not None else list(getattr(frame, "columns", []))
    found = {}
    for column in columns:
        values = _values(frame, column)
        if not values.empty and values.nunique() > 1:
            found[column] = (float(values.min()), float(values.max()))
    return found


def _widen(bounds, group):
    """Give every column in `group` the union of the group's bounds.

    This is what makes a *measure* switch as steady as a filter switch. The three OPS
    readings on the Slate page are the same unit asking three questions, so a control that
    flips between them must not also re-scale the plane they are drawn on.
    """
    present = [c for c in group if c in bounds]
    if len(present) < 2:
        return
    low = min(bounds[c][0] for c in present)
    high = max(bounds[c][1] for c in present)
    for column in present:
        bounds[column] = (low, high)


def anchor(frame, reference=None, *, columns=None, pad=PAD, share=(), mode="focus",
           domains=None):
    """Fix `frame`'s axis bounds to the range `reference` covers, and hand `frame` back.

    `reference` is the unfiltered board; `frame` is whatever survived the filters. Pass only
    one and it is used for both, which is the right call when a page has nothing wider to
    measure against.

    `share` is groups of columns that must end up on one domain -- see `_widen`. `mode` is
    `focus` for stable policy windows or `full` for the rounded raw reference extent.
    `domains` supplies page-family overrides when the same field name has narrower semantics
    in that view, such as aggregate lineup OPS versus individual hitter OPS.
    """
    if frame is None or getattr(frame, "empty", True):
        return frame
    chosen = str(mode or "focus").strip().lower()
    if chosen not in MODES:
        chosen = "focus"
    wider = reference is not None and not getattr(reference, "empty", True)
    source = reference if wider else frame
    bounds = measure(source, columns, pad)
    for column, floor in FLOORS.items():
        if column in bounds:
            bounds[column] = (max(float(floor), bounds[column][0]), bounds[column][1])
    if chosen == "focus":
        # Only replace fields that are actually present and numeric in this reference. A
        # policy entry must never manufacture an axis for a missing payload section.
        policy = {**FOCUS_DOMAINS, **(domains or {})}
        for column, domain in policy.items():
            if column in bounds:
                bounds[column] = domain
    for group in share:
        _widen(bounds, group)
    out = frame.copy(deep=False)
    out.attrs = {
        **getattr(frame, "attrs", {}),
        ATTRS_KEY: bounds,
        REFERENCE_KEY: _reference_stats(source, columns),
        EXTENTS_KEY: _extents(source, columns),
        MODE_KEY: chosen,
    }
    return out


def of(frame, field):
    """The anchored bounds for one field, or None if the page never anchored this frame."""
    bounds = getattr(frame, "attrs", {}).get(ATTRS_KEY) or {}
    found = bounds.get(field)
    return (float(found[0]), float(found[1])) if found else None


def reference(frame, field, statistic="median"):
    """A statistic measured on the unfiltered reference frame, or None when unavailable."""
    found = (getattr(frame, "attrs", {}).get(REFERENCE_KEY) or {}).get(field) or {}
    value = found.get(statistic)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def extent(frame, field):
    """The unpadded min/max used for non-position channels such as point area."""
    found = (getattr(frame, "attrs", {}).get(EXTENTS_KEY) or {}).get(field)
    return (float(found[0]), float(found[1])) if found else None


def mode(frame):
    """The frame's axis policy; unanchored frames have no policy."""
    return getattr(frame, "attrs", {}).get(MODE_KEY)


def overflow(frame, fields=None):
    """Counts below/above the active focus domain for each requested field.

    These are counts, not filtered data: charts clamp the marks to the boundary so no row
    silently disappears, and callers can disclose how many boundary marks are extremes.
    """
    if mode(frame) != "focus" or frame is None or getattr(frame, "empty", True):
        return {}
    fields = list(fields) if fields is not None else list((getattr(frame, "attrs", {})
                                                           .get(ATTRS_KEY) or {}))
    found = {}
    for field in fields:
        domain = of(frame, field)
        values = _values(frame, field)
        if not domain or values.empty:
            continue
        below = int((values < domain[0]).sum())
        above = int((values > domain[1]).sum())
        if below or above:
            found[field] = (below, above)
    return found


def overflow_note(frame, fields=None):
    """Short UI copy explaining boundary marks in Focus mode, or an empty string."""
    found = overflow(frame, fields)
    if not found:
        return ""
    parts = []
    for field, (below, above) in found.items():
        directions = []
        if below:
            directions.append(f"{below} below")
        if above:
            directions.append(f"{above} above")
        parts.append(f"{field}: {' / '.join(directions)}")
    return ("Focus range clamps boundary marks (exact values remain in tooltips): "
            + "; ".join(parts) + ". Switch Axis range to Full to expand the axes.")


def across(frame, fields):
    """One domain covering all of `fields` -- the parity panels' shared scale.

    None unless *every* field is anchored: a diagonal drawn over a domain that only half the
    data was measured against is worse than no fixed domain at all.
    """
    found = [of(frame, field) for field in fields]
    if not found or any(bound is None for bound in found):
        return None
    return (min(b[0] for b in found), max(b[1] for b in found))


def carry(frame, source):
    """Copy `source`'s bounds onto `frame`, for the places a builder rebuilds its data.

    pandas propagates `attrs` through most single-frame operations but not through the ones
    that assemble a new frame, and a builder that loses the bounds halfway silently goes back
    to auto-scaling -- the bug this whole module exists to remove.
    """
    if frame is None or source is None:
        return frame
    source_attrs = getattr(source, "attrs", {})
    carried = {key: source_attrs[key] for key in META_KEYS if key in source_attrs}
    if carried:
        frame.attrs = {**getattr(frame, "attrs", {}), **carried}
    return frame


def binding(name=ZOOM):
    """The pan/zoom parameter: drag to pan, wheel to zoom, double-click back to the anchor.

    Bound to the scales rather than to a brush, so the fixed domain stays the *home* position
    the chart returns to instead of something the reader has to restore by hand.
    """
    return alt.selection_interval(bind="scales", name=name, encodings=["x", "y"])


def zoomed_in(field, span, name=ZOOM):
    """Vega predicate: the reader has zoomed x inside `span`. False before any interaction.

    See the module docstring for why the two `isValid` guards are load-bearing rather than
    defensive noise.
    """
    quoted = str(field).replace("\\", "\\\\").replace("'", "\\'")
    return (f"isValid({name}) && isValid({name}['{quoted}']) && "
            f"abs({name}['{quoted}'][1] - {name}['{quoted}'][0]) < {float(span):.6g}")
