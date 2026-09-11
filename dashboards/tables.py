"""Cell tinting for the dashboard's data tables.

`st.dataframe` takes a pandas Styler, and the division of labour matters: **column_config
formats, the Styler colours.** Formatting through the Styler loses the column's type, which
costs sorting and the numeric alignment the grid does for free.

Two rules shape everything here, and both come from the same place as the chart contract:

* **Tints are translucent, never opaque.** Streamlit themes the grid, so the cell text is
  near-black on the light surface and near-white on the dark one. A solid pastel fill is
  legible under exactly one of those and unreadable under the other — the table equivalent
  of the hard-coded axis ink that made the chart legends vanish in dark mode. An `rgba`
  overlay composites over whichever surface is live and leaves the theme's own ink alone.
* **Colour never carries a value by itself.** Every column tinted here already prints the
  word ("Suppresses", "hitter", "Monitor") or the number. The tint is a second channel on
  something already legible, which is what makes it safe for a colourblind reader and in
  print.

The status hues are the project's reserved palette, the same ones `charts.py` spends on
`Signal` and reliever availability. They are never used for a series.
"""

import pandas as pd

#: Reserved status hues as translucent overlays. Alpha is high enough to read as a tint on
#: white and low enough that white text still clears it on the dark surface.
#: Values are complete CSS declarations, not bare colours: a Styler rejects `rgba(...)` on
#: its own with "Styles supplied as string must follow CSS rule formats".
GOOD = "background-color: rgba(12, 163, 12, 0.22)"
WARN = "background-color: rgba(250, 178, 25, 0.24)"
BAD = "background-color: rgba(208, 59, 59, 0.22)"
MUTED = "background-color: rgba(154, 160, 166, 0.16)"
NONE = ""

#: A pitcher's split against a side of the plate, tinted **from his side**: green means the
#: arm is winning that matchup. The cell says the word too, so the direction cannot be
#: misread as long as the caption states it once.
SPLIT_TAG_TINTS = {
    "Suppresses": GOOD,
    "Neutral": MUTED,
    "Vulnerable": BAD,
    # Not a verdict — an absence of one. It gets the muted tint rather than a colour that
    # would imply the sample said something.
    "Small sample": MUTED,
}

#: What `RV/100` has to reach before it is tinted. The pipeline's own `_edge_tag` calls a
#: pitch for one side at |RV/100| >= 0.7, so anything inside that band is noise it declines
#: to rule on and a tint there would claim more than the number says. Hard-coded rather than
#: imported: reading a payload deliberately does not import `scouting_report` (matplotlib,
#: fpdf and pybaseball ride along with it, for about four seconds).
RV_EDGE_THRESHOLD = 0.7

#: Who is ahead on a pitch, same convention: green is good for the pitcher.
#:
#: `Edge` is an **OR** over two signals — RV/100 *or* xwOBA past its own threshold — so a
#: pitch can be tagged `hitter` on xwOBA alone while its RV/100 sits mid-band and untinted.
#: That is the tag working as designed, not a disagreement to paper over.
EDGE_TINTS = {
    "pitcher": GOOD,
    "even": MUTED,
    "hitter": BAD,
    "small": MUTED,
}

#: Reliever availability. Shares `bullpen.TIERS` and `charts.AVAIL_COLORS` semantics.
AVAIL_TINTS = {
    "Available": GOOD,
    "Monitor": WARN,
    "Doubtful": BAD,
    "Out": BAD,
}


def _blank(frame):
    return pd.DataFrame("", index=frame.index, columns=frame.columns)


def by_value(frame, column, tints):
    """Styler function: tint `column` by looking each cell's text up in `tints`."""
    styles = _blank(frame)
    if column not in frame.columns:
        return styles
    styles[column] = (frame[column].astype(str).str.strip()
                      .map(lambda v: tints.get(v, NONE)).fillna(NONE))
    return styles


def signed(frame, column, good_when_negative=False, threshold=0.0):
    """Tint a column that is anchored at zero, by sign rather than by magnitude.

    Sign gets two colours; magnitude stays in the number itself. A continuous ramp here
    would double-encode a value the cell already prints, and a red→neutral→blue
    interpolation passes through magenta on the way besides.
    """
    styles = _blank(frame)
    if column not in frame.columns:
        return styles
    values = pd.to_numeric(frame[column], errors="coerce")
    positive, negative = (BAD_RGB, GOOD_RGB) if good_when_negative else (GOOD_RGB, BAD_RGB)
    # Depth is how far past the threshold the value sits, scaled by the biggest excursion in
    # the column — so the ramp is calibrated to this table rather than to an absolute that
    # would leave every cell at the same shade.
    excess = (values.abs() - threshold).clip(lower=0)
    span = float(excess.max() or 0)
    styles[column] = [
        NONE if pd.isna(v) or abs(v) <= threshold
        else _ramp(positive if v > 0 else negative, (abs(v) - threshold) / span if span else 1.0)
        for v in values
    ]
    return styles


#: Alpha at the ends of a ramp. The floor is the faintest tint that still reads as one; the
#: ceiling is where white text on the dark surface starts to lose the fill behind it.
RAMP_MIN, RAMP_MAX = 0.06, 0.30

#: Hue per direction, as the rgb triplet the ramp varies alpha over. A ramp is one hue
#: light-to-dark; a diverging pair is two hues with a **neutral midpoint**, which here is the
#: bare surface — no tint at all — rather than a third colour.
GOOD_RGB = "12, 163, 12"
BAD_RGB = "208, 59, 59"


def _ramp(rgb, weight):
    """One step of a single-hue ramp. `weight` in [0, 1]; 0 is no tint at all."""
    if weight <= 0:
        return NONE
    alpha = RAMP_MIN + (RAMP_MAX - RAMP_MIN) * min(weight, 1.0)
    return f"background-color: rgba({rgb}, {alpha:.3f})"


def _spread(values):
    """Each value's position in [0, 1] across the column's own range, NaN-safe."""
    low, high = values.min(), values.max()
    if pd.isna(low) or pd.isna(high) or high == low:
        return pd.Series(0.0, index=values.index)
    return (values - low) / (high - low)


def ranked(frame, column, best="low"):
    """Shade `column` across its own range: deepest green at the best, deepest red at worst.

    A single hue running light-to-dark either side of the midpoint, so the ramp reads as
    magnitude rather than as four unrelated colours. The midpoint is the *bare surface* — a
    value in the middle of the range gets no tint, which is what keeps a diverging scale from
    inventing a third hue for "average".

    The middle of a short table stays nearly clear, which is the honest rendering: a starter
    with three rest buckets has three samples, not a distribution.
    """
    styles = _blank(frame)
    if column not in frame.columns:
        return styles
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.notna().sum() < 2 or values.nunique() < 2:
        return styles
    position = _spread(values)
    if best == "low":
        position = 1.0 - position
    # Centred on the midpoint: above it shades good, below it shades bad, and the distance
    # from centre is what sets the depth.
    column_styles = []
    for value, place in zip(values, position):
        if pd.isna(value):
            column_styles.append(NONE)
            continue
        offset = (place - 0.5) * 2.0
        column_styles.append(_ramp(GOOD_RGB, offset) if offset > 0
                             else _ramp(BAD_RGB, -offset))
    styles[column] = column_styles
    return styles


def style(frame, rules):
    """Apply `rules` — a list of `(function, kwargs)` — to `frame`, returning a Styler.

    Returns the bare frame when nothing applies, because `st.dataframe` is happy with either
    and a Styler carrying no styles is just a slower frame.
    """
    if frame is None or frame.empty or not rules:
        return frame
    styler = frame.style
    applied = False
    for function, kwargs in rules:
        column = kwargs.get("column")
        if column and column not in frame.columns:
            continue
        styler = styler.apply(function, axis=None, **kwargs)
        applied = True
    return styler if applied else frame


# ===================================================================================
# Curated highlighting per view
#
# Ramping every numeric column turns a table into wallpaper: with everything shaded nothing
# is, and the eye has no entry point. So each view names the handful of columns a reader
# actually scans, and the rest stay clear.
#
# Direction is per column and is not guessable — a low ERA is good, a low K% is bad, and a
# low `surplus` means the market charged more than the model thinks he is worth. Getting one
# backwards produces a table that is confidently the wrong colour, which is worse than an
# unshaded one, so they are written out rather than inferred from the name.
# ===================================================================================

#: Columns where a bigger number is the better one, from the reader's point of view.
HIGHER_IS_BETTER = {
    "Composite", "Proj", "Ceiling", "Floor", "surplus", "per_1k", "Stack Score",
    "Stack Value", "Top5 Proj", "Top5 Ceiling", "Season OPS", "Platoon OPS", "Arsenal OPS",
    "L28 OPS", "Off L28", "Off Szn", "PA", "Lineup OPS", "hr_leverage", "hr_env",
    "K%", "K-BB", "K-BB%", "Whiff%",
}

#: ... and where a smaller one is.
LOWER_IS_BETTER = {
    "ERA", "FIP", "WHIP", "BB%", "Bust%", "Salary", "Own%", "Allowed OPS", "opp_ops",
    "Slot", "HH%", "Brl%", "HardHit%", "xwOBA", "RV/100",
}


def highlight(frame, columns=None, extra=None):
    """Shade the columns worth scanning in `frame`, leaving the rest clear.

    `columns` restricts the shading to a named subset — pass it when a frame carries a
    column that is in the vocabulary but is not what this particular table is about.
    `extra` appends rules (categorical tints, signed columns) after the ramps.
    """
    if frame is None or getattr(frame, "empty", True):
        return frame
    present = [c for c in frame.columns
               if (columns is None or c in columns)
               and (c in HIGHER_IS_BETTER or c in LOWER_IS_BETTER)]
    rules = [(ranked, {"column": c,
                       "best": "high" if c in HIGHER_IS_BETTER else "low"})
             for c in present]
    rules.extend(extra or [])
    return style(frame, rules)
