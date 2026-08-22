"""Key-metric visuals for the scouting report.

Four landscape figures, rendered by `build_report_figures` and carried on the workbook's
own **Visuals** tab (and at the top of the PDF), as an at-a-glance read of what the tables
otherwise make you assemble by hand:

* **Offense** -- how each club hits by platoon, how it has hit this *type* of arm against
  its own baseline, and which way recent form is pointing.
* **Starting pitching** -- each starter's platoon split, and per-pitch run value for each
  lineup against the arm it faces.
* **Bullpen availability** -- recent workload per arm, with availability in words.
* **Hitters to target** -- the composite matchup score per bat, carrying the report's own
  signal word.

Design rules this file follows (they are not stylistic preferences -- each one is a
failure mode that shows up in print):

* **Team colour is fixed by side, not by rank or value.** Away is always slot 1 (blue),
  home always slot 2 (orange), in every panel. A reader who learns "orange is the home
  club" in the first panel must not have to relearn it in the third. That also rules out
  shading bars by magnitude, which would burn the colour channel on information the bar
  length already carries.
* **Two series, so a legend is always present.** Identity is never colour-alone.
* **One axis per panel, and paired panels share one scale.** Two clubs drawn side by side
  will be read against each other, so the bullpen and target panels take a shared span --
  independent axes would make a 50-pitch bar look like a 90-pitch one.
* **Polarity is carried by sign, not by a second hue.** A diverging blue-red scale would
  put a conflicting meaning on the same blue the away club owns, so "does this pitch
  favour the arm or the bats" is a bar either side of a zero rule plus a word.
* **Status colour never travels alone** -- availability and signal always ship their word.
* **Text wears ink, never the series colour.** The mark beside a label carries identity;
  the label itself stays legible.
* **Thin marks, hairline solid gridlines, no dashes**, and values only at bar tips.

The palette is the validated categorical pair -- `validate_palette.js "#2a78d6,#eb6834"`
passes every check on the light surface (worst CVD ΔE 24.7, normal-vision 33.6, both
above 3:1 contrast), so the two clubs stay distinguishable in greyscale print and for
colour-vision-deficient readers.

`build_key_metrics_figure` returns a PNG path, or None when there is not enough data to
draw an honest panel -- the report then simply omits the visual rather than showing an
empty frame.
"""

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- design tokens -----------------------------------------------------------------
AWAY_COLOR = "#2a78d6"      # categorical slot 1
HOME_COLOR = "#eb6834"      # categorical slot 2
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#7a7975"
GRID = "#e3e2df"            # one step off the surface
ZERO_RULE = "#b8b7b3"

BAR_THICKNESS = 0.34        # of the category band; leaves the rest as air
BAR_GAP = 0.02              # surface gap between the two touching team bars

# The splits worth drawing, in reading order. "Overall" is the baseline the others are
# read against, so it leads.
_SPLIT_ORDER = ["Overall", "vs LHP", "vs RHP", "L28"]


def _num(value, default=np.nan):
    out = pd.to_numeric(value, errors="coerce")
    if isinstance(out, pd.Series):
        out = out.iloc[0] if len(out) else np.nan
    return default if pd.isna(out) else float(out)


def _split_lookup(splits_df):
    """{clean split name: (OPS, PA, is_tonight)} from a lineup-splits frame.

    The frame marks the split that applies tonight with a trailing '*' on the label
    (e.g. 'vs RHP *'), which is the only place that fact is carried -- so it is parsed
    here rather than re-derived from the starter's hand.
    """
    out = {}
    if not isinstance(splits_df, pd.DataFrame) or splits_df.empty:
        return out
    if "Split" not in splits_df.columns:
        return out
    for _, row in splits_df.iterrows():
        raw = str(row.get("Split", "")).strip()
        tonight = raw.endswith("*")
        name = raw.rstrip("*").strip()
        # 'Home (PF)' / 'Away (PF)' are venue rows; keep them keyed by the bare word.
        if name.startswith(("Home", "Away")):
            name = name.split(" ")[0]
        out[name] = (_num(row.get("OPS")), _num(row.get("PA")), tonight)
    return out


def _panel_platoon(ax, away_team, home_team, away_splits, home_splits):
    """Grouped bars: team OPS by split, with tonight's split called out.

    Absolute OPS rather than a delta from Overall: the reader needs to know that a club
    is a .620 team against lefties, not merely that it is 40 points below its own mean.
    Overall stays on the chart as the row to read the others against.
    """
    away_map, home_map = _split_lookup(away_splits), _split_lookup(home_splits)
    labels, away_vals, home_vals, tonight_for = [], [], [], []
    for split in _SPLIT_ORDER:
        a, h = away_map.get(split), home_map.get(split)
        if a is None and h is None:
            continue
        a_ops = a[0] if a else np.nan
        h_ops = h[0] if h else np.nan
        if np.isnan(a_ops) and np.isnan(h_ops):
            continue
        labels.append(split)
        away_vals.append(a_ops)
        home_vals.append(h_ops)
        # Which *clubs* this split applies to tonight, not merely whether it applies to
        # someone. The two teams face different starters, so in a L-vs-R matchup both the
        # LHP and the RHP row are live -- each for one club. Flagging the row without
        # naming the club read as "both teams face both hands", which is never true.
        tonight_for.append([team for flag, team in ((a and a[2], away_team),
                                                    (h and h[2], home_team)) if flag])
    if not labels:
        return False

    y = np.arange(len(labels))
    # Away sits above home in every band: the y axis is inverted, so away takes the
    # smaller offset. This matches the legend order and the report's away-left convention.
    off = BAR_THICKNESS / 2 + BAR_GAP / 2
    ax.barh(y - off, away_vals, height=BAR_THICKNESS, color=AWAY_COLOR, zorder=3)
    ax.barh(y + off, home_vals, height=BAR_THICKNESS, color=HOME_COLOR, zorder=3)

    for yi, val in zip(y - off, away_vals):
        if not np.isnan(val):
            ax.text(val + 0.012, yi, f"{val:.3f}", va="center", ha="left",
                    fontsize=7, color=INK_SECONDARY, zorder=4)
    for yi, val in zip(y + off, home_vals):
        if not np.isnan(val):
            ax.text(val + 0.012, yi, f"{val:.3f}", va="center", ha="left",
                    fontsize=7, color=INK_SECONDARY, zorder=4)

    # Tonight's split is the one the game actually turns on; mark it in the tick label
    # rather than by re-colouring the bar, which would break team identity.
    tick_labels = [f"{name}  ◄ {'/'.join(teams)}" if teams else name
                   for name, teams in zip(labels, tonight_for)]
    ax.set_yticks(y)
    ax.set_yticklabels(tick_labels, fontsize=8, color=INK_PRIMARY)
    for tick, teams in zip(ax.get_yticklabels(), tonight_for):
        if teams:
            tick.set_color(INK_PRIMARY)
            tick.set_fontweight("bold")
    ax.invert_yaxis()
    top = np.nanmax(away_vals + home_vals)
    ax.set_xlim(0, top * 1.20)
    ax.set_xlabel("OPS", fontsize=8, color=INK_SECONDARY)
    ax.set_title("Platoon & recent splits", fontsize=10, color=INK_PRIMARY,
                 loc="left", pad=8, fontweight="bold")
    return True


_TYPE_CODE_RE = re.compile(r"\b([A-Z]{2,3})\s*\d+%")


def _short_pitcher_type(raw, limit=30):
    """Condense a Pitcher Type label to something that fits beside a dumbbell.

    The field arrives in two shapes: a terse 'R / SI/FF/FC', and a verbose
    'L / FF 36% (91.5 mph/2378 rpm)/FC 32% (...)' which is ~90 characters and ran clean
    across the neighbouring panel. The codes are the part that identifies the arm, so the
    verbose form is reduced to them; anything still over-long is truncated rather than
    left to collide (a label that does not fit is moved or shortened, never clipped).
    """
    text = str(raw or "").strip()
    if not text:
        return ""
    if "%" in text:
        hand = text.split("/", 1)[0].strip()
        codes = _TYPE_CODE_RE.findall(text)
        if codes:
            text = f"{hand} / {'/'.join(codes)}" if hand else "/".join(codes)
    return text if len(text) <= limit else text[: limit - 1].rstrip(" /") + "…"


def _panel_type(ax, away_team, home_team, away_type, home_type):
    """Dumbbell: each club's OPS against this pitcher type vs its own baseline.

    A dumbbell rather than a bar of the difference, because the difference alone hides
    whether a +76 club is good-hitting-better or bad-hitting-less-badly. Both ends are on
    the chart; the arrow between them carries the sign.
    """
    rows = []
    for team, frame, color in ((away_team, away_type, AWAY_COLOR),
                               (home_team, home_type, HOME_COLOR)):
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        row = frame.iloc[0]
        base, typed = _num(row.get("Baseline OPS")), _num(row.get("Type OPS"))
        if np.isnan(base) or np.isnan(typed):
            continue
        rows.append({
            "team": team, "color": color, "baseline": base, "typed": typed,
            "pa": _num(row.get("PA"), 0), "games": _num(row.get("Games"), 0),
            "conf": str(row.get("Confidence", "") or ""),
            "ptype": _short_pitcher_type(row.get("Pitcher Type", "")),
        })
    if not rows:
        return False

    y = np.arange(len(rows))
    for yi, row in zip(y, rows):
        ax.plot([row["baseline"], row["typed"]], [yi, yi],
                color=row["color"], linewidth=2, solid_capstyle="round", zorder=3)
        # Baseline end is hollow, the "vs type" end is filled: one hue, two weights, so
        # the direction of travel is readable without a second colour.
        ax.plot(row["baseline"], yi, "o", markersize=8, markerfacecolor=SURFACE,
                markeredgecolor=row["color"], markeredgewidth=2, zorder=4)
        ax.plot(row["typed"], yi, "o", markersize=9, color=row["color"],
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
        delta = row["typed"] - row["baseline"]
        ax.text(max(row["baseline"], row["typed"]) + 0.018, yi,
                f"{delta*1000:+.0f} pts", va="center", ha="left",
                fontsize=7.5, color=INK_SECONDARY, zorder=6)

    # Team abbreviation only on the axis. Carrying the sample context in the tick label
    # pushed the plot area far to the right -- tick labels are laid out *outside* the
    # axes, so a long one steals width from the data and crowds the neighbouring panel.
    # As in-axes text it sits with the row it describes and costs no plot width.
    ax.set_yticks(y)
    ax.set_yticklabels([r["team"] for r in rows], fontsize=9, color=INK_PRIMARY,
                       fontweight="bold")
    ax.invert_yaxis()
    lo = min(min(r["baseline"], r["typed"]) for r in rows)
    hi = max(max(r["baseline"], r["typed"]) for r in rows)
    pad = max((hi - lo) * 0.35, 0.05)
    ax.set_xlim(lo - pad, hi + pad * 1.9)
    ax.set_ylim(len(rows) - 0.35, -0.75)
    for yi, row in zip(y, rows):
        # Joined from the parts that exist, so a missing confidence grade does not leave
        # a dangling separator.
        parts = [p for p in (row["ptype"],
                             f"{int(row['games'])}g / {int(row['pa'])} PA",
                             row["conf"]) if p]
        ax.text(0.012, yi - 0.32, "  ·  ".join(parts),
                transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=6.8, color=INK_MUTED, zorder=6)
    ax.set_xlabel("OPS", fontsize=8, color=INK_SECONDARY)
    ax.set_title("vs this pitcher type", fontsize=10, color=INK_PRIMARY,
                 loc="left", pad=8, fontweight="bold")
    # The two ends need naming; a colour legend cannot say which dot is which.
    ax.plot([], [], "o", markersize=7, markerfacecolor=SURFACE, markeredgecolor=INK_MUTED,
            markeredgewidth=2, label="baseline (same hand)", linestyle="none")
    ax.plot([], [], "o", markersize=7, color=INK_MUTED, label="vs type", linestyle="none")
    ax.legend(loc="lower right", fontsize=6.5, frameon=False, handletextpad=0.4,
              labelcolor=INK_SECONDARY, borderpad=0.2, ncol=2, columnspacing=1.0,
              bbox_to_anchor=(1.0, -0.02))
    return True


def _panel_form(ax, away_team, home_team, away_form, home_form):
    """Run differential per game across the 7 / 14 / 30-day windows.

    Differential rather than runs scored: it is the one number that says which way a club
    is actually trending, and scored/allowed are both on the chart implicitly through it.
    Bars grow from a zero rule, so above and below the line read immediately.
    """
    def series(frame):
        out = {}
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return out
        for _, row in frame.iterrows():
            window = str(row.get("Window", "")).strip()
            diff = _num(row.get("Run Diff/G"))
            if window and not np.isnan(diff):
                out[window] = diff
        return out

    away_map, home_map = series(away_form), series(home_form)
    windows = [w for w in ("7D", "14D", "30D") if w in away_map or w in home_map]
    if not windows:
        return False

    y = np.arange(len(windows))
    off = BAR_THICKNESS / 2 + BAR_GAP / 2
    away_vals = [away_map.get(w, np.nan) for w in windows]
    home_vals = [home_map.get(w, np.nan) for w in windows]
    ax.barh(y - off, away_vals, height=BAR_THICKNESS, color=AWAY_COLOR, zorder=3)
    ax.barh(y + off, home_vals, height=BAR_THICKNESS, color=HOME_COLOR, zorder=3)

    span = np.nanmax(np.abs(away_vals + home_vals)) or 1.0
    for yi, val in list(zip(y - off, away_vals)) + list(zip(y + off, home_vals)):
        if np.isnan(val):
            continue
        # Label outside the bar end, on whichever side the bar grew.
        pad = span * 0.06
        ax.text(val + (pad if val >= 0 else -pad), yi, f"{val:+.2f}",
                va="center", ha="left" if val >= 0 else "right",
                fontsize=7, color=INK_SECONDARY, zorder=4)

    ax.axvline(0, color=ZERO_RULE, linewidth=1, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"last {w[:-1]} days" for w in windows], fontsize=8,
                       color=INK_PRIMARY)
    ax.invert_yaxis()
    ax.set_xlim(-span * 1.45, span * 1.45)
    ax.set_xlabel("run differential per game", fontsize=8, color=INK_SECONDARY)
    ax.set_title("Recent form", fontsize=10, color=INK_PRIMARY,
                 loc="left", pad=8, fontweight="bold")
    return True


def _style(ax, vertical_grid=True):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["bottom"].set_linewidth(1)
    if vertical_grid:
        # Hairline, solid, recessive -- behind the data, never dashed.
        ax.xaxis.grid(True, color=GRID, linewidth=1, linestyle="-", zorder=0)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=7, colors=INK_MUTED, length=0)
    ax.tick_params(axis="y", length=0)


def build_key_metrics_figure(away_team, home_team, context, output_path,
                             game_date=None, dpi=170):
    """Render the three-panel key-metrics figure. Returns the path, or None.

    `context` is the report's advanced_context dict. Any panel whose data is missing is
    dropped and the remaining ones re-laid out; if none survive, nothing is written and
    the caller omits the section entirely.
    """
    panels = [
        ("platoon", _panel_platoon, (context.get("away_lineup_splits"),
                                     context.get("home_lineup_splits"))),
        ("type", _panel_type, (context.get("away_type_results"),
                               context.get("home_type_results"))),
        ("form", _panel_form, (context.get("away_rolling_form"),
                               context.get("home_rolling_form"))),
    ]

    # Draw into a throwaway figure first to find out which panels actually have data,
    # so the final layout is sized for the panels that survive rather than leaving a
    # blank cell where a table was empty.
    probe_fig, probe_axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3))
    keep = []
    for (name, fn, data), ax in zip(panels, np.atleast_1d(probe_axes)):
        try:
            if fn(ax, away_team, home_team, *data):
                keep.append((name, fn, data))
        except Exception:
            pass
    plt.close(probe_fig)
    if not keep:
        return None

    width = 3.95 * len(keep) + 0.5
    fig, axes = plt.subplots(1, len(keep), figsize=(width, 3.5), dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes)
    for (name, fn, data), ax in zip(keep, axes):
        fn(ax, away_team, home_team, *data)
        _style(ax)

    handles = [
        plt.Line2D([], [], marker="s", markersize=8, linestyle="none",
                   color=AWAY_COLOR, label=f"{away_team} (away)"),
        plt.Line2D([], [], marker="s", markersize=8, linestyle="none",
                   color=HOME_COLOR, label=f"{home_team} (home)"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.008, 0.995),
               ncol=2, frameon=False, fontsize=8.5, labelcolor=INK_PRIMARY,
               handletextpad=0.5, columnspacing=1.6)

    subtitle = f"{away_team} at {home_team}"
    if game_date:
        subtitle += f"  ·  {game_date}"
    fig.text(0.992, 0.978, subtitle, ha="right", va="top", fontsize=7.5,
             color=INK_MUTED)

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight",
                pad_inches=0.16)
    plt.close(fig)
    return output_path


def figure_aspect(path):
    """(width, height) in pixels for a rendered figure.

    `bbox_inches="tight"` trims the canvas, so the final size is not the figsize that was
    asked for -- the embedder has to measure the file to scale it without distortion.
    Read with matplotlib rather than PIL to avoid adding a dependency for two numbers.
    """
    import matplotlib.image as mpimg
    height, width = mpimg.imread(path).shape[:2]
    return width, height


# ===================================================================================
# Pitching, bullpen and hitter-targeting panels
#
# Colour stays bound to the entity throughout: a starter wears his club's colour, an
# arm wears his club's colour, a hitter wears his club's colour. Polarity -- does this
# pitch favour the arm or the bats? -- is carried by the *sign* of the bar off a zero
# rule and by an explicit word at the bar end, not by a second hue. A diverging
# blue-red scale would have put a second, conflicting meaning on the same blue the
# away club already owns.
#
# Availability and signal are the genuine status scales here, and a status colour
# never travels alone: every marker ships with its word.
# ===================================================================================

STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_CRITICAL = "#d03b3b"
NEUTRAL_MARK = "#c3c2b7"

_AVAIL_STATUS = {
    "Available": (STATUS_GOOD, "●"),
    "Monitor": (STATUS_WARNING, "◐"),
    "Taxed": (STATUS_CRITICAL, "○"),
}

_SIGNAL_STATUS = {
    "Priority": (STATUS_GOOD, "▲"),
    "Watch": (STATUS_WARNING, "△"),
    "Neutral": (NEUTRAL_MARK, "·"),
    "Fade": (STATUS_CRITICAL, "▼"),
}

_DOT = "  ·  "


def _first_text(frame, column, default=""):
    if not isinstance(frame, pd.DataFrame) or frame.empty or column not in frame.columns:
        return default
    value = frame[column].dropna()
    return str(value.iloc[0]) if len(value) else default


def _panel_starter_platoon(ax, away_team, home_team, away_hand, home_hand):
    """Each starter's OPS allowed to left- and right-handed bats.

    The actionable question is "which side of the plate does this arm not get out", so
    both ends are drawn and the gap between them *is* the platoon split. A starter wears
    his own club's colour; the L and R ends are told apart by their letters, not by hue.
    """
    rows = []
    for team, frame, color in ((away_team, away_hand, AWAY_COLOR),
                               (home_team, home_hand, HOME_COLOR)):
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        if not {"Batter Side", "OPS"}.issubset(frame.columns):
            continue
        by_side = {}
        for _, row in frame.iterrows():
            side = str(row.get("Batter Side", "")).strip()[:1].upper()
            ops = _num(row.get("OPS"))
            if side in ("L", "R") and not np.isnan(ops):
                by_side[side] = (ops, _num(row.get("PA"), 0),
                                 str(row.get("Split Tag", "") or ""))
        if len(by_side) < 2:
            continue
        rows.append({"team": team, "color": color,
                     "name": _first_text(frame, "Pitcher", team), "sides": by_side})
    if not rows:
        return False

    y = np.arange(len(rows))
    for yi, row in zip(y, rows):
        left, right = row["sides"]["L"][0], row["sides"]["R"][0]
        ax.plot([left, right], [yi, yi], color=row["color"], linewidth=2,
                solid_capstyle="round", zorder=3)
        for side, value in (("L", left), ("R", right)):
            ax.plot(value, yi, "o", markersize=11, color=row["color"],
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
            ax.text(value, yi, side, ha="center", va="center", fontsize=6.5,
                    color=SURFACE, fontweight="bold", zorder=5)
        # Name the side that actually beats him, which is the point of the panel.
        worse = "L" if left >= right else "R"
        tag = row["sides"][worse][2] or "worse side"
        ax.text(max(left, right) + 0.022, yi,
                f"{worse}HB {max(left, right):.3f}{_DOT}{tag}",
                va="center", ha="left", fontsize=7, color=INK_SECONDARY, zorder=6)

    ax.set_yticks(y)
    ax.set_yticklabels([r["name"] for r in rows], fontsize=8, color=INK_PRIMARY,
                       fontweight="bold")
    ax.invert_yaxis()
    lo = min(min(r["sides"]["L"][0], r["sides"]["R"][0]) for r in rows)
    hi = max(max(r["sides"]["L"][0], r["sides"]["R"][0]) for r in rows)
    pad = max((hi - lo) * 0.30, 0.05)
    ax.set_xlim(lo - pad, hi + pad * 3.6)
    ax.set_ylim(len(rows) - 0.4, -0.75)
    ax.set_xlabel("OPS allowed", fontsize=8, color=INK_SECONDARY)
    ax.set_title("Starter platoon split", fontsize=10, color=INK_PRIMARY,
                 loc="left", pad=8, fontweight="bold")
    return True


def _panel_arsenal_edge(ax, title, matchup_df, color):
    """Per-pitch run value from the hitting club's side, on a zero rule.

    Bars to the right are pitches these bats have hit; to the left, pitches that have
    beaten them. Usage rides the label because a pitch worth attacking is only worth
    attacking if he throws it -- a +0.7 RV/100 pitch at 4% usage is a curiosity.
    """
    if not isinstance(matchup_df, pd.DataFrame) or matchup_df.empty:
        return False
    if not {"Pitch", "RV/100"}.issubset(matchup_df.columns):
        return False
    frame = matchup_df.head(8)
    pitches, values, usage, edges = [], [], [], []
    dropped = 0
    for _, row in frame.iterrows():
        rv = _num(row.get("RV/100"))
        if np.isnan(rv):
            continue
        edge = str(row.get("Edge", "") or "")
        # "small" is the report's own word for "fewer than min_pitches seen", i.e. this
        # run value is noise. Left in, it does more than mislead on its own row: one
        # 10%-usage forkball at +6.77 RV/100 stretched the axis to ±20 and squashed every
        # trustworthy bar in the panel to a sliver. Dropped, and the count is disclosed.
        if edge == "small":
            dropped += 1
            continue
        pitches.append(str(row.get("Pitch", "")))
        values.append(rv)
        usage.append(_num(row.get("Usage%"), np.nan))
        edges.append(edge)
    if not pitches:
        return False
    pitches, values, usage, edges = (pitches[:6], values[:6], usage[:6], edges[:6])

    y = np.arange(len(pitches))
    ax.barh(y, values, height=0.42, color=color, zorder=3)
    span = max(abs(v) for v in values) or 1.0
    for yi, value, use in zip(y, values, usage):
        # Value and usage only. The edge word rides the tick label instead: a
        # left-growing bar puts its label in the same space the tick labels occupy, and
        # the longer three-part string collided with the pitch code.
        bits = [f"{value:+.2f}"]
        if not np.isnan(use):
            bits.append(f"{use:.0f}% use")
        ax.text(value + (span * 0.08 if value >= 0 else -span * 0.08), yi,
                _DOT.join(bits), va="center",
                ha="left" if value >= 0 else "right",
                fontsize=6.8, color=INK_SECONDARY, zorder=4)

    ax.axvline(0, color=ZERO_RULE, linewidth=1, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{p}  {e}".strip() for p, e in zip(pitches, edges)],
                       fontsize=8, color=INK_PRIMARY)
    ax.invert_yaxis()
    ax.set_xlim(-span * 3.1, span * 3.1)
    label = "RV/100   (right = bats ahead)"
    if dropped:
        # Say what is missing rather than quietly showing a partial arsenal.
        label += f"   ·   {dropped} low-sample pitch{'es' if dropped > 1 else ''} hidden"
    ax.set_xlabel(label, fontsize=8, color=INK_SECONDARY)
    ax.set_title(title, fontsize=9.5, color=INK_PRIMARY, loc="left", pad=8,
                 fontweight="bold")
    return True


def _panel_bullpen(ax, team, usage_df, color, shared_top=None):
    """Recent workload per arm, with availability stated in words beside it.

    The decision this supports is "who can actually pitch tonight", so the arms are
    ordered by how much they have thrown and every bar carries its status word -- the
    status colour is a second cue, never the only one.
    """
    if not isinstance(usage_df, pd.DataFrame) or usage_df.empty:
        return False
    if not {"Name", "Pit"}.issubset(usage_df.columns):
        return False
    frame = usage_df.copy()
    frame["_pit"] = pd.to_numeric(frame["Pit"], errors="coerce").fillna(0)
    frame = frame[frame["Name"].astype(str).str.strip().ne("")]
    # Display order only -- the bar colour is the club's, not this ranking's.
    frame = frame.sort_values("_pit", ascending=False).head(9)
    if frame.empty:
        return False

    y = np.arange(len(frame))
    ax.barh(y, frame["_pit"].to_numpy(), height=0.42, color=color, zorder=3)
    # Both pens are drawn on one scale. Two panels side by side with independent axes
    # invite exactly the comparison they cannot support -- a 50-pitch bar in one panel
    # would look like a 90-pitch bar in the other.
    top = float(shared_top or frame["_pit"].max()) or 1.0
    for yi, (_, row) in zip(y, frame.iterrows()):
        avail = str(row.get("Avail", "") or "").strip()
        status_color, glyph = _AVAIL_STATUS.get(avail, (NEUTRAL_MARK, "·"))
        pitches = float(row["_pit"])
        ax.text(pitches + top * 0.04, yi, glyph, va="center", ha="left",
                fontsize=8, color=status_color, zorder=4)
        label = f"{int(pitches)}P"
        if avail:
            label = f"{label}  {avail}"
        ax.text(pitches + top * 0.12, yi, label, va="center", ha="left",
                fontsize=6.8, color=INK_SECONDARY, zorder=4)

    hands = frame["T"].astype(str) if "T" in frame.columns else pd.Series([""] * len(frame))
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}  {h}".strip() for n, h in zip(frame["Name"], hands)],
                       fontsize=7.5, color=INK_PRIMARY)
    ax.invert_yaxis()
    ax.set_xlim(0, top * 1.95)
    ax.set_xlabel("pitches, last 5 games", fontsize=8, color=INK_SECONDARY)
    ax.set_title(f"{team} bullpen workload", fontsize=9.5, color=INK_PRIMARY,
                 loc="left", pad=8, fontweight="bold")
    return True


def _panel_hitter_targets(ax, team, composite_df, color, shared_span=None):
    """Composite matchup score per hitter, with the report's own signal word attached.

    The most directly actionable panel: the batting order sorted by how good tonight
    looks for each bat. The signal word is the report's existing classification, repeated
    here rather than re-derived, so the picture and the table cannot disagree.
    """
    if not isinstance(composite_df, pd.DataFrame) or composite_df.empty:
        return False
    if not {"Team", "Name", "Composite"}.issubset(composite_df.columns):
        return False
    frame = composite_df[composite_df["Team"].astype(str) == str(team)].copy()
    if frame.empty:
        return False
    frame["_score"] = pd.to_numeric(frame["Composite"], errors="coerce")
    frame = frame.dropna(subset=["_score"]).head(9)
    if frame.empty:
        return False

    y = np.arange(len(frame))
    ax.barh(y, frame["_score"].to_numpy(), height=0.44, color=color, zorder=3)
    # Shared across both clubs for the same reason as the bullpen panels: these sit side
    # by side and will be read against each other.
    span = float(shared_span or np.nanmax(np.abs(frame["_score"].to_numpy()))) or 1.0
    for yi, (_, row) in zip(y, frame.iterrows()):
        score = float(row["_score"])
        signal = str(row.get("Signal", "") or "").strip()
        status_color, glyph = _SIGNAL_STATUS.get(signal, (NEUTRAL_MARK, "·"))
        step = span * 0.05 if score >= 0 else -span * 0.05
        align = "left" if score >= 0 else "right"
        ax.text(score + step, yi, glyph, va="center", ha=align, fontsize=7.5,
                color=status_color, zorder=4)
        text = f"{score:+.0f}"
        if signal:
            text = f"{text}  {signal}"
        ax.text(score + step * 2.8, yi, text, va="center", ha=align, fontsize=6.8,
                color=INK_SECONDARY, zorder=4)

    ax.axvline(0, color=ZERO_RULE, linewidth=1, zorder=2)
    ax.set_yticks(y)
    bats = frame["Bats"].astype(str) if "Bats" in frame.columns else pd.Series([""] * len(frame))
    ax.set_yticklabels([f"{n}  {b}".strip() for n, b in zip(frame["Name"], bats)],
                       fontsize=7.5, color=INK_PRIMARY)
    ax.invert_yaxis()
    ax.set_xlim(-span * 1.75, span * 1.75)
    ax.set_xlabel("composite matchup score", fontsize=8, color=INK_SECONDARY)
    ax.set_title(f"{team} hitters to target", fontsize=9.5, color=INK_PRIMARY,
                 loc="left", pad=8, fontweight="bold")
    return True


def _bullpen_usage_frame(entry):
    """The per-arm usage table out of a `{side}_bullpen_l5` payload.

    That context value is a tuple of (team game log, per-arm usage); the per-arm frame is
    the one with a Name column, so it is found by shape rather than by index -- the tuple
    order is not guaranteed anywhere.
    """
    parts = entry if isinstance(entry, (tuple, list)) else [entry]
    for part in parts:
        if isinstance(part, pd.DataFrame) and not part.empty and "Name" in part.columns:
            return part
    return None


def _render(panels, output_path, dpi, legend_teams=None, subtitle=None,
            panel_width=3.95, height=3.5):
    """Lay out whichever panels have data and write the PNG. None if none do."""
    probe_fig, probe_axes = plt.subplots(1, max(len(panels), 1),
                                         figsize=(4 * max(len(panels), 1), 3))
    keep = []
    for spec, ax in zip(panels, np.atleast_1d(probe_axes)):
        try:
            if spec["draw"](ax):
                keep.append(spec)
        except Exception:
            pass
    plt.close(probe_fig)
    if not keep:
        return None

    fig, axes = plt.subplots(1, len(keep), figsize=(panel_width * len(keep) + 0.5, height),
                             dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    for spec, ax in zip(keep, np.atleast_1d(axes)):
        spec["draw"](ax)
        _style(ax)

    top = 0.90
    if legend_teams:
        away_team, home_team = legend_teams
        handles = [
            plt.Line2D([], [], marker="s", markersize=8, linestyle="none",
                       color=AWAY_COLOR, label=f"{away_team} (away)"),
            plt.Line2D([], [], marker="s", markersize=8, linestyle="none",
                       color=HOME_COLOR, label=f"{home_team} (home)"),
        ]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.008, 0.995),
                   ncol=2, frameon=False, fontsize=8.5, labelcolor=INK_PRIMARY,
                   handletextpad=0.5, columnspacing=1.6)
    if subtitle:
        fig.text(0.992, 0.978, subtitle, ha="right", va="top", fontsize=7.5,
                 color=INK_MUTED)
    if not legend_teams and not subtitle:
        top = 0.94

    fig.tight_layout(rect=(0, 0, 1, top))
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight",
                pad_inches=0.16)
    plt.close(fig)
    return output_path


def build_pitching_figure(away_team, home_team, context, output_path, game_date=None,
                          dpi=170):
    """Starter platoon splits plus each lineup's per-pitch run value against the arm it faces.

    `{side}_arsenal_matchup` is keyed by the *hitting* club -- `away_arsenal_matchup` is the
    away lineup against the home starter -- so the panels are titled by who is batting,
    matching how the tables downstream read.
    """
    away_sp = _first_text(context.get("home_pitcher_hand_splits"), "Pitcher", home_team)
    home_sp = _first_text(context.get("away_pitcher_hand_splits"), "Pitcher", away_team)
    panels = [
        {"draw": lambda ax: _panel_starter_platoon(
            ax, away_team, home_team,
            context.get("away_pitcher_hand_splits"),
            context.get("home_pitcher_hand_splits"))},
        {"draw": lambda ax: _panel_arsenal_edge(
            ax, f"{away_team} bats vs {away_sp}",
            context.get("away_arsenal_matchup"), AWAY_COLOR)},
        {"draw": lambda ax: _panel_arsenal_edge(
            ax, f"{home_team} bats vs {home_sp}",
            context.get("home_arsenal_matchup"), HOME_COLOR)},
    ]
    subtitle = f"{away_team} at {home_team}"
    if game_date:
        subtitle = f"{subtitle}   {game_date}"
    return _render(panels, output_path, dpi, legend_teams=(away_team, home_team),
                   subtitle=subtitle)


def build_bullpen_figure(away_team, home_team, context, output_path, game_date=None,
                         dpi=170):
    """Who is actually available out of each pen tonight."""
    frames = {side: _bullpen_usage_frame(context.get(f"{side}_bullpen_l5"))
              for side in ("away", "home")}
    tops = [pd.to_numeric(f["Pit"], errors="coerce").max()
            for f in frames.values()
            if isinstance(f, pd.DataFrame) and not f.empty and "Pit" in f.columns]
    shared_top = float(np.nanmax(tops)) if tops else None
    panels = [
        {"draw": lambda ax: _panel_bullpen(ax, away_team, frames["away"], AWAY_COLOR,
                                           shared_top=shared_top)},
        {"draw": lambda ax: _panel_bullpen(ax, home_team, frames["home"], HOME_COLOR,
                                           shared_top=shared_top)},
    ]
    return _render(panels, output_path, dpi, panel_width=4.6, height=3.4)


def build_targets_figure(away_team, home_team, context, output_path, game_date=None,
                         dpi=170):
    """Which bats the report thinks are worth backing tonight, in batting order."""
    composite = context.get("hitter_composite")
    shared_span = None
    if isinstance(composite, pd.DataFrame) and "Composite" in composite.columns:
        scores = pd.to_numeric(composite["Composite"], errors="coerce").abs()
        if scores.notna().any():
            shared_span = float(scores.max())
    panels = [
        {"draw": lambda ax: _panel_hitter_targets(ax, away_team, composite, AWAY_COLOR,
                                                  shared_span=shared_span)},
        {"draw": lambda ax: _panel_hitter_targets(ax, home_team, composite, HOME_COLOR,
                                                  shared_span=shared_span)},
    ]
    return _render(panels, output_path, dpi, panel_width=4.6, height=3.4)


#: Ordered figures for the Visuals tab. Offense first because it frames the game, then the
#: arms, then the pen, then the per-hitter call the rest of it adds up to.
FIGURE_BUILDERS = (
    ("Offense", "offense", build_key_metrics_figure),
    ("Starting pitching", "pitching", build_pitching_figure),
    ("Bullpen availability", "bullpen", build_bullpen_figure),
    ("Hitters to target", "targets", build_targets_figure),
)


def build_report_figures(away_team, home_team, context, output_dir, game_date=None,
                         dh_suffix="", dpi=170):
    """Render every figure that has data. Returns [(caption, path), ...] in tab order."""
    made = []
    for caption, slug, builder in FIGURE_BUILDERS:
        path = os.path.join(
            output_dir, f"{slug}_{game_date}_{away_team}_{home_team}{dh_suffix}.png")
        try:
            if builder(away_team, home_team, context, path, game_date=game_date, dpi=dpi):
                made.append((caption, path))
        except Exception as error:
            print(f"[!] {caption} visual unavailable: {error}")
    return made
