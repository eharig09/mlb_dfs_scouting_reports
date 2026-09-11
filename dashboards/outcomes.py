"""What actually happened, joined back to what the report said beforehand.

`dk_results/` holds DraftKings contest standings exports: per-player **actual DK points** and
the field's **real ownership**. `dfs.results.contest_night` ties each export to the slate it
was run on — 46 of 49 files, over 16 dates — which is what lets a night's board be scored
against its own outcome.

Why this module exists
----------------------
Everything else in the dashboard is *descriptive*: it shows what the report believed before
first pitch. This is the only part that can say whether the belief was worth anything.
Measured over 2,731 scored hitter-slates, on rows where every column is present:

    predicting actual DK points        pearson   spearman
      projected ceiling                 +0.164    +0.152
      model projection (Proj)           +0.159    +0.157
      expected plate appearances        +0.142    +0.145
      DK salary                         +0.139    +0.137
      batting order slot                -0.147    -0.150
      composite matchup score           +0.108    +0.108
      surplus vs price band             +0.044    +0.039

    partial correlations with points
      Proj      controlling for salary   +0.084
      Salary    controlling for Proj     +0.030
      Composite controlling for salary   +0.034
      Composite controlling for Proj     +0.002

**The projection beats the market**, and the market adds almost nothing on top of it. And
the composite adds *nothing* on top of the projection — +0.002 — which is what you would
expect from a component of it rather than a rival to it. The composite is worth looking at
as an explanation of where a projection came from, not as a separate opinion. That is a small sample of a very noisy quantity — single-game
baseball scoring is close to a coin flip dressed up — and it is precisely why the binned
views here exist rather than another summary statistic: a weak average can still hide a
region where the score works, and a grid will show it where a correlation will not.

Price is not the only confound
------------------------------
An earlier reading of this module reported a +10.3pp composite spread in the cheapest
salary band as the one surviving edge. **That was substantially opportunity, not matchup.**
Cheap hitters bat at the bottom of the order, and batting slot alone moves the outcome
more than anything the composite says:

    slot 1-2   7.9 mean DK pts   33% hit   avg salary $4,555
    slot 8-9   4.9 mean DK pts   17% hit   avg salary $2,880

Holding salary *and* slot fixed relocates the finding — the composite separates
bottom-of-the-order hitters at **either** price (+8.0pp cheap, beyond noise; +9.2pp
expensive, n too small to call) and does nothing at the top of the order at either price
(+1.5pp, +1.1pp). Which is the more useful statement anyway: it adds information exactly
where lineup slot gives you least.

Running the other way, **expected PA is what separates expensive hitters** (+11.0pp and
+9.4pp within the two dearest bands, both beyond noise) while the composite there does not.

Neither control is lookahead. Salary, batting slot and projected PA are all known before
first pitch; they come from `dfs.projections` reading the same cached payload.

One more caveat this module does not fix: every row here was in a posted or projected
lineup (2,510 Confirmed of 2,733), because the board is built from the report's lineup
card. So none of this describes players who did not play — it describes players who did.

**Nothing here is a projection.** It is a record. The grid says what share of hitters in a
region of the board have historically cleared a points threshold; it does not claim tonight's
hitter will.
"""

import numpy as np
import pandas as pd
import streamlit as st

#: A "hit" for a hitter. 10 DK points is roughly the line a mid-priced bat has to clear to
#: have been worth its slot; the threshold is exposed so a page can move it.
DEFAULT_HIT = 10.0

#: A grid cell needs this many historical observations before its rate is worth showing.
#: Below it the cell is drawn empty rather than in a colour that implies knowledge.
MIN_CELL = 12


@st.cache_data(ttl="6h", max_entries=4, show_spinner="Scoring past slates…")
def scored_history(hit_threshold=DEFAULT_HIT):
    """Every hitter-slate we can both score and describe.

    One row per (date, slate, player): what the report said (composite, its parts), what the
    market charged, what the field did (ownership) and what he actually scored.
    """
    from dashboards import data, salaries
    from dfs import results as results_module

    index = results_module._export_index()
    rows = []
    for contest in results_module.list_contests():
        date, slate = results_module.contest_night(contest, index)
        if not date:
            continue
        points = contest.get("fpts") or {}
        owned = contest.get("ownership") or {}
        for key, scored in points.items():
            rows.append({"date": str(date), "slate": slate, "key": key,
                         "fpts": scored, "own": owned.get(key)})
    if not rows:
        return pd.DataFrame()
    outcomes = pd.DataFrame(rows).drop_duplicates(["date", "slate", "key"])

    frames = []
    for date in sorted(outcomes["date"].unique()):
        hitters = data.hitters_for_date(date)
        if hitters.empty:
            continue
        board = salaries.attach_salary(hitters, date)
        board["key"] = board["Name"].map(salaries.name_key)
        # Both sides carry a `slate`, and they mean different things: the salary file's is
        # which export the price came from, the contest's is which contest was actually
        # played. Merging blind produces slate_x/slate_y and silently loses both names.
        board = board.rename(columns={"slate": "price_slate"})
        # Opportunity, from the report's own pre-game view. Batting slot and expected PA
        # are the confound that matters here and they are *not* lookahead — both are known
        # before first pitch. Without them a "cheap hitters beat their price" finding is
        # mostly "cheap hitters bat eighth".
        board = board.merge(_opportunity(date), on="key", how="left")
        night = outcomes[outcomes["date"] == date][["key", "fpts", "own", "slate"]]
        frames.append(board.merge(night, on="key", how="inner"))
    if not frames:
        return pd.DataFrame()

    joined = pd.concat(frames, ignore_index=True)
    for column in ("fpts", "own", "PA", "Slot", "Proj", "Ceiling"):
        if column in joined.columns:
            joined[column] = pd.to_numeric(joined[column], errors="coerce")
    joined["hit"] = joined["fpts"] >= float(hit_threshold)
    # Points per $1k of salary: the same outcome measured in the units a roster is built
    # in, so a cheap hitter clearing his price is not scored against the same bar as an
    # expensive one who did not.
    joined["pts_per_1k"] = np.where(joined["Salary"] > 0,
                                    joined["fpts"] / (joined["Salary"] / 1000.0), np.nan)
    return joined.dropna(subset=["fpts"])


def _opportunity(date):
    """Batting slot, expected PA and lineup status for a date — all pre-game.

    Comes from `dfs.projections`, which reads the same cached payload, so this adds no
    network and no lookahead: every column here was knowable before first pitch.
    """
    from dashboards import data, salaries

    projected = data.projections_for_date(date)
    if projected.empty or "Name" not in projected.columns:
        return pd.DataFrame(columns=["key", "PA", "Slot", "Lineup"])
    batters = (projected[projected["Type"] != "P"] if "Type" in projected.columns
               else projected).copy()
    batters["key"] = batters["Name"].map(salaries.name_key)
    keep = ["key"] + [c for c in ("PA", "Slot", "Lineup", "Proj", "Ceiling", "Floor")
                      if c in batters.columns]
    return batters.reindex(columns=keep).drop_duplicates("key")


def hit_grid(frame, x, y, x_bins=6, y_bins=6, min_cell=MIN_CELL):
    """Historical hit rate over a 2-D region of the board.

    Returns one row per cell with its bounds, the hit rate, the mean points and the number of
    observations behind it. **Cells under `min_cell` are dropped rather than drawn pale**: a
    thin cell coloured at all reads as knowledge the sample does not support, and on a grid
    the eye reads adjacency as a trend.
    """
    if frame is None or frame.empty:
        return pd.DataFrame()
    if not {x, y, "hit", "fpts"}.issubset(frame.columns):
        return pd.DataFrame()
    data = frame.dropna(subset=[x, y, "fpts"])
    if data.empty:
        return pd.DataFrame()

    # Quantile edges, so each column carries a comparable number of players rather than a
    # comparable width — a fixed-width grid on salary leaves the top bins nearly empty.
    x_edges = np.unique(np.nanquantile(data[x], np.linspace(0, 1, x_bins + 1)))
    y_edges = np.unique(np.nanquantile(data[y], np.linspace(0, 1, y_bins + 1)))
    if len(x_edges) < 3 or len(y_edges) < 3:
        return pd.DataFrame()

    data = data.assign(
        _xi=np.clip(np.digitize(data[x], x_edges[1:-1]), 0, len(x_edges) - 2),
        _yi=np.clip(np.digitize(data[y], y_edges[1:-1]), 0, len(y_edges) - 2))
    grouped = (data.groupby(["_xi", "_yi"], observed=True)
               .agg(rate=("hit", "mean"), points=("fpts", "mean"), n=("hit", "size"))
               .reset_index())
    grouped = grouped[grouped["n"] >= min_cell]
    if grouped.empty:
        return pd.DataFrame()

    grouped["x0"] = x_edges[grouped["_xi"]]
    grouped["x1"] = x_edges[grouped["_xi"] + 1]
    grouped["y0"] = y_edges[grouped["_yi"]]
    grouped["y1"] = y_edges[grouped["_yi"] + 1]
    grouped["rate_pct"] = grouped["rate"] * 100
    return grouped.drop(columns=["_xi", "_yi"])


def lift_table(frame, column, bins=5, min_cell=25):
    """Hit rate by band of one column, with the lift over the population rate.

    The plain version of the same question: does sorting on this number move the hit rate at
    all? `lift` is the ratio to the overall rate, so 1.00 is "this column told you nothing".
    """
    if frame is None or frame.empty or column not in frame.columns:
        return pd.DataFrame()
    data = frame.dropna(subset=[column, "hit"])
    if len(data) < bins * min_cell:
        return pd.DataFrame()
    data = data.copy()
    try:
        data["band"] = pd.qcut(data[column], bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    overall = data["hit"].mean()
    out = (data.groupby("band", observed=True)
           .agg(n=("hit", "size"), rate=("hit", "mean"), points=("fpts", "mean"),
                salary=("Salary", "mean"), low=(column, "min"), high=(column, "max"))
           .reset_index())
    out = out[out["n"] >= min_cell]
    out["lift"] = out["rate"] / overall if overall else np.nan
    out["rate_pct"] = out["rate"] * 100
    out["band"] = out["band"].astype(str)
    return out


def controlled_lift(frame, column, control="Salary", bands=4, splits=3, min_cell=30,
                    second=None, second_bands=2):
    """Does `column` still separate outcomes *inside* a cell of the controls?

    This is the test that matters and the one a raw correlation hides. The composite sorts
    hit rate from 17.5% to 31.2% across its deciles — and mean salary rises from $2,789 to
    $4,749 across those same deciles, so almost all of that ordering is price.

    Controlling for price alone is not enough
    ----------------------------------------
    The first version of this controlled only for salary and reported a +10.3pp spread in
    the cheapest band as the one real edge. That reading was wrong, and the confound is
    **batting-order slot**:

        slot 1-2   7.9 mean pts   33% hit   avg salary $4,555
        slot 8-9   4.9 mean pts   17% hit   avg salary $2,880

    Cheap hitters bat at the bottom of the order, so "cheap hitters beat their price" was
    substantially "the composite noticed who was batting eighth". Adding slot as a second
    control moves the finding somewhere more useful:

        cheap  + slots 1-5   n=454   +1.5pp   within noise
        cheap  + slots 6-9   n=995   +8.0pp   beyond noise
        pricey + slots 1-5   n=1080  +1.1pp   within noise
        pricey + slots 6-9   n=202   +9.2pp   within noise (n too small)

    So the edge is **bottom-of-the-order hitters at any price**, not cheap hitters — the
    composite adds information exactly where lineup slot gives you least. At the top of the
    order it adds nothing at either price.

    Neither control is lookahead: salary and batting slot are both known before first pitch.

    A second control costs cells
    ----------------------------
    Two controls crossed at four bands each is sixteen cells over ~2,700 rows, most of them
    too thin to read. So `second_bands` defaults to 2 and `bands` drops to 2 when a second
    control is given — a coarse split that survives the sample beats a fine one that does not.
    """
    if frame is None or frame.empty:
        return pd.DataFrame()
    controls = [control] + ([second] if second else [])
    if not {column, "hit", *controls}.issubset(frame.columns):
        return pd.DataFrame()
    data = frame.dropna(subset=[column, "hit", *controls]).copy()
    if data.empty:
        return pd.DataFrame()

    keys = []
    primary_bands = 2 if second else bands
    for name, count in zip(controls, [primary_bands, second_bands]):
        try:
            data[f"_c_{name}"] = pd.qcut(data[name], count, duplicates="drop")
        except ValueError:
            return pd.DataFrame()
        keys.append(f"_c_{name}")

    rows = []
    for cell, group in data.groupby(keys, observed=True):
        if len(group) < splits * min_cell:
            continue
        try:
            group = group.assign(_split=pd.qcut(group[column], splits, labels=False,
                                                duplicates="drop"))
        except ValueError:
            continue
        rates = group.groupby("_split", observed=True)["hit"].agg(["mean", "size"])
        if len(rates) < splits:
            continue
        low, high = rates["mean"].iloc[0], rates["mean"].iloc[-1]
        # Standard error of the difference of two proportions, so a spread can be read
        # against the noise it sits in rather than taken at face value.
        n_low, n_high = rates["size"].iloc[0], rates["size"].iloc[-1]
        se = float(np.sqrt(low * (1 - low) / n_low + high * (1 - high) / n_high))
        cell = cell if isinstance(cell, tuple) else (cell,)
        rows.append({"control band": " · ".join(_band_label(c) for c in cell),
                     "n": int(len(group)),
                     "low": low * 100, "high": high * 100,
                     "spread": (high - low) * 100, "se": se * 100,
                     "beyond noise": abs(high - low) > 2 * se})
    return pd.DataFrame(rows)


def _band_label(interval):
    """A readable bound for a qcut interval — `2000-3700` rather than the repr."""
    try:
        low, high = interval.left, interval.right
    except AttributeError:
        return str(interval)
    fmt = (lambda v: f"{v:,.0f}") if abs(high) >= 100 else (lambda v: f"{v:.1f}")
    return f"{fmt(max(low, 0))}-{fmt(high)}"


#: Controls a reader can hold fixed, and what each one is guarding against.
CONTROLS = {
    "Salary": "Salary — what the market already knew",
    "Slot": "Batting order slot — opportunity",
    "PA": "Expected plate appearances — opportunity",
    "own": "Field ownership — what everyone else did",
}


def cell_members(frame, x, y, cell):
    """The historical players behind one grid cell.

    The drill-down the grid exists to support: a shaded cell is a claim, and this is the
    evidence for it.
    """
    if frame is None or frame.empty or not cell:
        return pd.DataFrame()
    data = frame.dropna(subset=[x, y])
    x0, x1, y0, y1 = (float(cell[k]) for k in ("x0", "x1", "y0", "y1"))
    # Half-open on the upper edge, **except for the topmost cell**, which has to take the
    # maximum itself or the single most expensive hitter belongs to no cell at all. The
    # inclusion has to be decided per *edge*, not per row: testing `value == max` row-wise
    # admits the global maximum into every cell in its row or column, and the drill-down
    # then disagrees with the rate the cell is coloured by.
    in_x = (data[x] >= x0) & ((data[x] < x1) | (x1 >= data[x].max()))
    in_y = (data[y] >= y0) & ((data[y] < y1) | (y1 >= data[y].max()))
    return data[in_x & in_y].sort_values("fpts", ascending=False)


def player_history(frame, name):
    """Every scored slate for one hitter, most recent first.

    Backs the click-through: the composite said X, the market charged Y, he scored Z.
    """
    if frame is None or frame.empty or "Name" not in frame.columns:
        return pd.DataFrame()
    from dashboards import salaries as salaries_module
    key = salaries_module.name_key(name)
    return (frame[frame["Name"].map(salaries_module.name_key) == key]
            .sort_values(["date", "slate"], ascending=[False, True]))


#: Columns a scored row can be drilled into, in the order a reader wants them.
#: Opportunity sits beside the score deliberately — batting slot moves the outcome more
#: than the composite does, so a drill-down that omitted it would invite the same wrong
#: reading the controlled panel exists to prevent.
DRILL_COLUMNS = ["date", "slate", "game", "Team", "Name", "Bats", "Slot", "PA", "Salary",
                 "Composite", "surplus", "Proj", "Ceiling", "own", "fpts", "pts_per_1k",
                 "hit", "Signal", "Season OPS", "L28 OPS", "Platoon OPS", "Arsenal OPS",
                 "Off L28"]


def drill_frame(frame):
    """A scored frame reduced to the columns worth reading, without inventing any."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    return frame.reindex(columns=[c for c in DRILL_COLUMNS if c in frame.columns])
