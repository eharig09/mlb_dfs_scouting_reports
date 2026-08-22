"""Figures for the slate workbook's Visuals tab.

Four figures, one per question the board is trying to answer. Same design contract as the
MLB report's visuals, and for the same reasons — each rule below is a failure mode that
showed up in print there:

* **Colour is bound to the club, never to rank or value.** A slate has up to eight teams,
  so the categorical palette is assigned in fixed slot order and a team keeps its hue across
  every figure. Shading a bar by its own magnitude would burn the colour channel on
  information the bar length already carries.
* **Paired or faceted panels share one scale.** Panels side by side get read against each
  other; independent axes invite a comparison they cannot support.
* **Polarity is the sign of the bar off a zero rule plus a word**, never a second hue.
* **Rank on expected values, never on conditional ones** — usage conditional on playing puts
  a fourth-string quarterback among the starters.
* Thin marks, hairline solid gridlines, values at bar tips, text in ink rather than the
  series colour.

The eight-slot categorical order is the validated one; slots 1-3 are safe for all-pairs
forms, which is why the team-coloured figures here cap at the four clubs of a typical
single-game or two-game slate and fall back to a single hue beyond that.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Validated categorical order (worst adjacent CVD dE 9.1 on the light surface).
TEAM_SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
              "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#7a7975"
GRID = "#e3e2df"
RULE = "#b8b7b3"

BAR_H = 0.62


def team_colors(teams):
    """Fixed slot order, so a club keeps its hue across every figure in the workbook."""
    return {team: TEAM_SLOTS[i % len(TEAM_SLOTS)] for i, team in enumerate(sorted(teams))}


def _style(ax, vertical_grid=True):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["bottom"].set_linewidth(1)
    if vertical_grid:
        ax.xaxis.grid(True, color=GRID, linewidth=1, linestyle="-", zorder=0)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=7, colors=MUTED, length=0)
    ax.tick_params(axis="y", length=0)


def _num(frame, column):
    """Column as floats, or all-NaN when absent.

    `frame.get(col)` returns None for a missing column and `pd.to_numeric(None)` is a
    scalar NaN with no index — the recurring crash in this codebase.
    """
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def _finish(fig, path, dpi):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    return path


def _team_legend(fig, colors):
    handles = [plt.Line2D([], [], marker="s", markersize=8, linestyle="none",
                          color=color, label=team) for team, color in colors.items()]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.008, 0.995),
               ncol=min(len(handles), 8), frameon=False, fontsize=8.5,
               labelcolor=INK, handletextpad=0.5, columnspacing=1.4)


def figure_board(board, path, top_n=16, dpi=170):
    """Projected points, the headline ranking."""
    frame = board.dropna(subset=["proj"]).head(top_n)
    if frame.empty:
        return None
    colors = team_colors(frame["team"].dropna().unique())
    fig, ax = plt.subplots(figsize=(9.2, 0.32 * len(frame) + 1.5), dpi=dpi)
    fig.patch.set_facecolor(SURFACE)

    y = np.arange(len(frame))
    values = _num(frame, "proj").to_numpy()
    ax.barh(y, values, height=BAR_H,
            color=[colors.get(t, TEAM_SLOTS[0]) for t in frame["team"]], zorder=3)
    span = np.nanmax(values) or 1.0
    for yi, value, basis in zip(y, values, frame.get("basis", pd.Series("", index=frame.index))):
        # The basis rides the value: "PFF says 16.7" and "we inferred it" must not look alike.
        ax.text(value + span * 0.015, yi, f"{value:.1f}   {basis}", va="center", ha="left",
                fontsize=7, color=INK_2, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}  {p}" for n, p in zip(frame["Name"], frame["pos"])],
                       fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, span * 1.42)
    ax.set_xlabel("projected DK points per game", fontsize=8, color=INK_2)
    ax.set_title("Board", fontsize=11, color=INK, loc="left", pad=8, fontweight="bold")
    _style(ax)
    _team_legend(fig, colors)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _finish(fig, path, dpi)


def figure_red_zone(board, path, top_n=14, dpi=170):
    """Touchdown equity, and who out- or under-scored the work behind it."""
    frame = board.dropna(subset=["xTD_pg"]).sort_values("xTD_pg", ascending=False).head(top_n)
    if frame.empty:
        return None
    colors = team_colors(frame["team"].dropna().unique())
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 0.34 * len(frame) + 1.7), dpi=dpi)
    fig.patch.set_facecolor(SURFACE)

    y = np.arange(len(frame))
    equity = _num(frame, "xTD_pg").to_numpy()
    ax = axes[0]
    ax.barh(y, equity, height=BAR_H,
            color=[colors.get(t, TEAM_SLOTS[0]) for t in frame["team"]], zorder=3)
    span = np.nanmax(equity) or 1.0
    for yi, value in zip(y, equity):
        ax.text(value + span * 0.02, yi, f"{value:.2f}", va="center", ha="left",
                fontsize=7, color=INK_2, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}  {p}" for n, p in zip(frame["Name"], frame["pos"])],
                       fontsize=7.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, span * 1.30)
    ax.set_xlabel("expected TD per game, from red-zone work", fontsize=8, color=INK_2)
    ax.set_title("Touchdown equity", fontsize=10, color=INK, loc="left", pad=8,
                 fontweight="bold")
    _style(ax)

    # Residual: both tails matter, for opposite reasons. Polarity is carried by which side
    # of the zero rule the bar grows on, *not* by a second hue — these are the same players
    # as the left panel and recolouring them green/red there would have one man appear in
    # two colours inside one figure.
    ax = axes[1]
    residual = _num(frame, "TD_oe").to_numpy()
    ax.barh(y, residual, height=BAR_H,
            color=[colors.get(t, TEAM_SLOTS[0]) for t in frame["team"]], zorder=3)
    reach = np.nanmax(np.abs(residual)) or 1.0
    for yi, value in zip(y, residual):
        if not np.isfinite(value):
            continue
        pad = reach * 0.05
        ax.text(value + (pad if value >= 0 else -pad), yi, f"{value:+.1f}", va="center",
                ha="left" if value >= 0 else "right", fontsize=7, color=INK_2, zorder=4)
    ax.axvline(0, color=RULE, linewidth=1, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlim(-reach * 1.5, reach * 1.5)
    ax.set_xlabel("TD over expectation   (left = scored under the work)", fontsize=8,
                  color=INK_2)
    ax.set_title("Regression candidates", fontsize=10, color=INK, loc="left", pad=8,
                 fontweight="bold")
    _style(ax)

    _team_legend(fig, colors)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _finish(fig, path, dpi)


def figure_schedule(board, path, dpi=170):
    """Positional SOS per club — a grid, because it is a small dense matrix.

    Sequential single hue: this is magnitude on an ordered scale, not identity, and a
    categorical palette here would double-encode the number as a colour that means nothing.
    """
    frame = board.dropna(subset=["sos_rating"])
    if frame.empty:
        return None
    grid = (frame.groupby(["team", "pos"])["sos_rating"].first().unstack()
            .reindex(columns=["QB", "RB", "WR", "TE"]))
    grid = grid.dropna(how="all")
    if grid.empty:
        return None

    fig, ax = plt.subplots(figsize=(0.95 * len(grid.columns) + 3.4,
                                    0.5 * len(grid) + 2.0), dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    # Fixed 0-10 domain: the published scale. Auto-scaling would make a slate of uniformly
    # hard matchups look like it contained easy ones.
    mesh = ax.imshow(grid.to_numpy(dtype=float), cmap="Blues", vmin=0, vmax=10, aspect="auto")
    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels(grid.columns, fontsize=9, color=INK)
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels(grid.index, fontsize=9, color=INK, fontweight="bold")
    for i in range(len(grid.index)):
        for j in range(len(grid.columns)):
            value = grid.iloc[i, j]
            if not np.isfinite(value):
                continue
            # Label inside a filled cell: pick the ink that survives the fill behind it.
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8.5,
                    color=SURFACE if value > 6.0 else INK, fontweight="bold")
    ax.set_xticks(np.arange(-.5, len(grid.columns), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(grid.index), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Strength of schedule   ·   higher is easier", fontsize=10, color=INK,
                 loc="left", pad=10, fontweight="bold")
    bar = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.03)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=7, colors=MUTED, length=0)
    fig.tight_layout()
    return _finish(fig, path, dpi)


def figure_scheme_fit(board, path, top_n=12, dpi=170):
    """Man/zone edge signed by what the opposing defense actually plays."""
    ranked = board.dropna(subset=["scheme_fit"]).sort_values("scheme_fit", ascending=False)
    if ranked.empty:
        return None
    # Both tails, not the top N. A zero-ruled chart showing only positive bars wastes half
    # its axis, and the receivers whose strength is the *wrong* one for this defense are as
    # actionable as the ones it suits.
    half = max(1, top_n // 2)
    frame = (pd.concat([ranked.head(half), ranked.tail(half)])
             .drop_duplicates(subset=["Name", "team"]) if len(ranked) > top_n else ranked)
    colors = team_colors(frame["team"].dropna().unique())
    fig, ax = plt.subplots(figsize=(9.6, 0.34 * len(frame) + 1.7), dpi=dpi)
    fig.patch.set_facecolor(SURFACE)

    y = np.arange(len(frame))
    fit = _num(frame, "scheme_fit").to_numpy()
    ax.barh(y, fit, height=BAR_H,
            color=[colors.get(t, TEAM_SLOTS[0]) for t in frame["team"]], zorder=3)
    reach = np.nanmax(np.abs(fit)) or 1.0
    opp_man = _num(frame, "opp_man_rate").to_numpy()
    for yi, value, opp, man in zip(y, fit, frame["opp"], opp_man):
        pad = reach * 0.05
        label = f"{value:+.2f}   vs {opp}"
        if np.isfinite(man):
            label += f" ({man:.0f}% man)"
        ax.text(value + (pad if value >= 0 else -pad), yi, label, va="center",
                ha="left" if value >= 0 else "right", fontsize=7, color=INK_2, zorder=4)
    ax.axvline(0, color=RULE, linewidth=1, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}  {p}" for n, p in zip(frame["Name"], frame["pos"])],
                       fontsize=7.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(-reach * 1.9, reach * 1.9)
    ax.set_xlabel("scheme fit   ·   right = his strength matches what this defense plays",
                  fontsize=8, color=INK_2)
    ax.set_title("Coverage fit", fontsize=10, color=INK, loc="left", pad=8, fontweight="bold")
    _style(ax)
    _team_legend(fig, colors)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _finish(fig, path, dpi)


FIGURES = (
    ("Board", "board", figure_board),
    ("Touchdown equity", "redzone", figure_red_zone),
    ("Strength of schedule", "schedule", figure_schedule),
    ("Coverage fit", "scheme", figure_scheme_fit),
)


def build_figures(board, output_dir, label="slate", dpi=170):
    """Render every figure that has data. Returns [(caption, path), ...]."""
    made = []
    for caption, slug, builder in FIGURES:
        path = os.path.join(output_dir, f"{slug}_{label}.png")
        try:
            if builder(board, path, dpi=dpi):
                made.append((caption, path))
        except Exception as error:
            print(f"[!] {caption} figure unavailable: {error}")
    return made
