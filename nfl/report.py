"""Write a slate to an Excel workbook.

The workbook is the deliverable, following the MLB side's convention rather than inventing
a new one: every sheet opens at 85% zoom, fits a maximised window without horizontal
scrolling, and shades its numeric columns by league percentile so a board is scannable
before it is read.

Tabs, in the order they are meant to be read:

    Visuals     the four figures, first because it is the at-a-glance pass
    Board       every skill player, ranked, with the basis for each number
    Red Zone    touchdown equity and who out- or under-scored their workload
    Matchup     schedule difficulty and coverage-scheme fit
    Cold Start  players with no history, priced off draft capital

**Heat is direction-aware and only ever applied to columns where high and low genuinely
mean better and worse.** A column like `depth_rank` is ordinal and inverted; a column like
`TD_oe` is a two-sided residual where both extremes are interesting for opposite reasons,
so it gets a diverging scale rather than a good/bad one.

**A missing number is left blank, never zeroed.** Half a preseason board has no PFF
projection, and a blank cell says that where a 0.00 would quietly claim it was projected to
score nothing.
"""

import os

import numpy as np
import pandas as pd

from nfl import slate

# Shared with the MLB workbook: a maximised 1920px window at 85% shows ~288 width units.
WIDTH_BUDGET = 290
MIN_WIDTH, MAX_WIDTH = 4.5, 26.0

# Green -> amber -> red, matching the MLB report's heat so the two read the same way.
_HEAT_GOOD = (99, 190, 123)
_HEAT_MID = (255, 214, 122)
_HEAT_BAD = (243, 129, 129)
_HEAT_NEUTRAL = (222, 226, 230)


def _blend(low, high, t):
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(low, high))


def _heat_rgb(percentile, invert=False):
    """Percentile -> fill. 1.0 is best unless `invert`."""
    if percentile is None or not np.isfinite(percentile):
        return None
    p = float(np.clip(percentile, 0.0, 1.0))
    if invert:
        p = 1.0 - p
    if p >= 0.5:
        return _blend(_HEAT_MID, _HEAT_GOOD, (p - 0.5) * 2)
    return _blend(_HEAT_BAD, _HEAT_MID, p * 2)


def _diverging_rgb(value, scale):
    """Two-sided residual -> fill. Zero is neutral; both tails are coloured."""
    if value is None or not np.isfinite(value) or not scale:
        return None
    t = float(np.clip(value / scale, -1.0, 1.0))
    if t >= 0:
        return _blend(_HEAT_NEUTRAL, _HEAT_GOOD, t)
    return _blend(_HEAT_NEUTRAL, _HEAT_BAD, -t)


#: column -> (heat mode, invert). Anything absent is written plain.
HEAT = {
    "proj": ("pct", False),
    "pff_ppg": ("pct", False),
    "xTD_pg": ("pct", False),
    "xTD": ("pct", False),
    "rz_target_share": ("pct", False),
    "rz_carry_share": ("pct", False),
    "sos_rating": ("pct", False),          # higher = easier, verified
    "expected_opportunity": ("pct", False),
    "snap_avg": ("pct", False),
    "yprr": ("pct", False),
    "grades_pass_route": ("pct", False),
    "elusive_rating": ("pct", False),
    "yco_attempt": ("pct", False),
    "depth_rank": ("pct", True),           # 1 is the starter
    "TD_oe": ("diverge", False),
    "scheme_fit": ("diverge", False),
    "zone_edge": ("diverge", False),
}

_INT_COLUMNS = {"Salary", "depth_rank", "games", "rzRecTarg", "ezRecTarg",
                "rzRushCarries", "i5RushCarries", "TD", "routes", "man_routes",
                "zone_routes", "pick", "snap_avg"}


def _fmt_for(column):
    if column in _INT_COLUMNS:
        return "0"
    if column in ("rz_target_share", "rz_carry_share", "availability"):
        return "0.0%"
    if column in ("sos_mult",):
        return "0.000"
    return "0.00"


def _percentiles(series):
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() < 3:
        return pd.Series(np.nan, index=series.index)
    return values.rank(pct=True)


def _write_sheet(workbook, name, frame, columns, headers=None, note=None, freeze="A3"):
    """One tab. Returns the worksheet."""
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

    sheet = workbook.create_sheet(name)
    sheet.sheet_view.showGridLines = False
    sheet.sheet_view.zoomScale = 85
    sheet.sheet_view.zoomScaleNormal = 85
    sheet.page_setup.orientation = "landscape"
    sheet.page_setup.fitToWidth = 1
    sheet.page_setup.fitToHeight = 0
    sheet.sheet_properties.pageSetUpPr.fitToPage = True

    thin = Side(style="thin", color="D6DAE0")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    header_fill = PatternFill("solid", fgColor="1C2A4A")
    centre = Alignment(horizontal="center", vertical="center", wrap_text=True)
    left = Alignment(horizontal="left", vertical="center")

    row = 1
    if note:
        cell = sheet.cell(row=row, column=1, value=note)
        cell.font = Font(italic=True, size=9, color="4A5568")
        row += 1

    labels = headers or {}
    present = [c for c in columns if c in frame.columns]
    for index, column in enumerate(present, start=1):
        cell = sheet.cell(row=row, column=index, value=labels.get(column, column))
        cell.font = Font(bold=True, size=9, color="FFFFFF")
        cell.fill = header_fill
        cell.alignment = centre
        cell.border = border
    header_row = row

    heat_percentiles = {c: _percentiles(frame[c]) for c in present
                        if HEAT.get(c, ("", False))[0] == "pct"}
    diverge_scale = {}
    for column in present:
        if HEAT.get(column, ("", False))[0] == "diverge":
            values = pd.to_numeric(frame[column], errors="coerce").abs()
            diverge_scale[column] = float(values.quantile(0.9)) if values.notna().any() else None

    for offset, (_, record) in enumerate(frame.iterrows(), start=1):
        for index, column in enumerate(present, start=1):
            value = record.get(column)
            # A blank is a blank. Writing 0 for "no source had this" is a lie the sheet
            # cannot walk back.
            if value is None or (isinstance(value, float) and not np.isfinite(value)) or pd.isna(value):
                cell = sheet.cell(row=header_row + offset, column=index, value=None)
                cell.border = border
                continue
            cell = sheet.cell(row=header_row + offset, column=index, value=value)
            cell.border = border
            cell.font = Font(size=9)
            if isinstance(value, (int, float, np.integer, np.floating)):
                cell.number_format = _fmt_for(column)
                cell.alignment = centre
                mode, invert = HEAT.get(column, (None, False))
                rgb = None
                if mode == "pct":
                    rgb = _heat_rgb(heat_percentiles[column].iloc[offset - 1], invert)
                elif mode == "diverge":
                    rgb = _diverging_rgb(float(value), diverge_scale.get(column))
                if rgb:
                    cell.fill = PatternFill("solid",
                                            fgColor="{:02X}{:02X}{:02X}".format(*rgb))
            else:
                cell.alignment = left

    _fit_widths(sheet, frame, present, labels)
    sheet.freeze_panes = sheet.cell(row=header_row + 1, column=2).coordinate
    return sheet


def _fit_widths(sheet, frame, columns, labels):
    """Width per column, capped so the sheet stays inside the 85%-zoom budget."""
    from openpyxl.utils import get_column_letter

    widths = []
    for column in columns:
        header = str(labels.get(column, column))
        # `.astype(str).replace("nan", "")` looks equivalent and is not: under pandas 3.0
        # replacing the *string* "nan" on a str-dtype Series yields a real float NaN back,
        # so the widths loop then calls len() on a float. Go through the nullable string
        # dtype and fill, which means what it says on every pandas version.
        body = frame[column].astype("string").fillna("").head(200).tolist()
        longest = max([len(header)] + [len(value) for value in body])
        widths.append(min(MAX_WIDTH, max(MIN_WIDTH, longest + 1.5)))

    total = sum(widths)
    if total > WIDTH_BUDGET:
        # Scale back proportionally rather than truncating one column: an over-budget sheet
        # scrolls sideways, which is the specific thing the budget exists to prevent.
        factor = WIDTH_BUDGET / total
        widths = [max(MIN_WIDTH, w * factor) for w in widths]
    for index, width in enumerate(widths, start=1):
        sheet.column_dimensions[get_column_letter(index)].width = round(width, 1)


BOARD_COLUMNS = ["Name", "pos", "team", "opp", "basis", "proj", "pff_ppg", "sos_rating",
                 "depth_rank", "snap_avg", "expected_opportunity", "xTD_pg",
                 "rz_target_share", "yprr", "grades_pass_route", "elusive_rating",
                 "yco_attempt", "Salary"]
BOARD_HEADERS = {"pos": "Pos", "team": "Team", "opp": "Opp", "basis": "Basis",
                 "proj": "Proj", "pff_ppg": "PFF/g", "sos_rating": "SOS",
                 "depth_rank": "Role", "snap_avg": "Snaps 25", "xTD_pg": "xTD/g",
                 "expected_opportunity": "Our opp", "rz_target_share": "RZ tgt sh",
                 "yprr": "YPRR", "grades_pass_route": "Route gr",
                 "elusive_rating": "Elusive", "yco_attempt": "YCO/att"}

RED_ZONE_COLUMNS = ["Name", "pos", "team", "games", "rzRecTarg", "ezRecTarg",
                    "rzRushCarries", "i5RushCarries", "rz_target_share", "rz_carry_share",
                    "xTD_pg", "xTD", "TD", "TD_oe"]
RED_ZONE_HEADERS = {"pos": "Pos", "team": "Team", "games": "G", "rzRecTarg": "RZ tgt",
                    "ezRecTarg": "EZ tgt", "rzRushCarries": "RZ car",
                    "i5RushCarries": "i5 car", "rz_target_share": "RZ tgt sh",
                    "rz_carry_share": "RZ car sh", "xTD_pg": "xTD/g", "TD_oe": "TD oe"}

MATCHUP_COLUMNS = ["Name", "pos", "team", "opp", "sos_rating", "sos_mult",
                   "opp_man_rate", "man_yprr", "zone_yprr", "zone_edge", "scheme_fit",
                   "man_routes", "zone_routes"]
MATCHUP_HEADERS = {"pos": "Pos", "team": "Team", "opp": "Opp", "sos_rating": "SOS",
                   "sos_mult": "SOS x", "opp_man_rate": "Opp man%",
                   "man_yprr": "Man YPRR", "zone_yprr": "Zone YPRR",
                   "zone_edge": "Zone edge", "scheme_fit": "Fit",
                   "man_routes": "Man rt", "zone_routes": "Zone rt"}

COLD_COLUMNS = ["Name", "pos", "team", "opp", "basis", "depth_rank", "pick",
                "availability", "expected_opportunity", "proj"]
COLD_HEADERS = {"pos": "Pos", "team": "Team", "opp": "Opp", "basis": "Basis",
                "depth_rank": "Role", "pick": "Pick", "availability": "Avail",
                "expected_opportunity": "Exp opp", "proj": "Proj"}


def write_workbook(board, output_path, slate_label="", figures=None):
    """Write the slate workbook. `figures` is [(caption, png path), ...] for the Visuals tab."""
    from openpyxl import Workbook

    workbook = Workbook()
    workbook.remove(workbook.active)

    if figures:
        _write_visuals(workbook, figures, slate_label)

    header = f"{slate_label}  ·  {len(board)} skill players" if slate_label else ""
    _write_sheet(workbook, "Board", board, BOARD_COLUMNS, BOARD_HEADERS,
                 note=f"{header}   |   Proj = PFF points per game x schedule; "
                      f"where PFF is silent, our own expected opportunity. "
                      f"Basis names the source — a blank cell means no source had it.")

    red_zone = board[slate._column(board, "xTD_pg").notna()]
    red_zone = red_zone.sort_values("xTD_pg", ascending=False)
    _write_sheet(workbook, "Red Zone", red_zone, RED_ZONE_COLUMNS, RED_ZONE_HEADERS,
                 note="Touchdown equity from where the work happens. An end-zone target is "
                      "worth 0.40 TD, a red-zone target 0.11, an open-field target 0.006. "
                      "TD oe = actual minus expected: positive scored above the workload.")

    matchup = board[slate._column(board, "sos_rating").notna()]
    matchup = matchup.sort_values(["team", "proj"], ascending=[True, False])
    _write_sheet(workbook, "Matchup", matchup, MATCHUP_COLUMNS, MATCHUP_HEADERS,
                 note="SOS: higher is EASIER (verified +0.506 vs WR PPR allowed). "
                      "Fit is the receiver's man/zone edge signed by what this defense "
                      "actually plays; shown only at 50+ routes in each scheme.")

    cold = board[slate._column(board, "basis", "none").isin(["draft capital", "replacement", "none"])]
    cold = cold.sort_values("expected_opportunity", ascending=False)
    _write_sheet(workbook, "Cold Start", cold, COLD_COLUMNS, COLD_HEADERS,
                 note="No PFF projection. Usage is conditional on playing, so rank on "
                      "expected opportunity — availability folded in.")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    workbook.save(output_path)
    return output_path


def _write_visuals(workbook, figures, slate_label):
    from openpyxl.drawing.image import Image as XLImage
    from openpyxl.styles import Font

    sheet = workbook.create_sheet("Visuals")
    sheet.sheet_view.showGridLines = False
    sheet.sheet_view.zoomScale = 85
    sheet.sheet_view.zoomScaleNormal = 85

    row = 1
    if slate_label:
        cell = sheet.cell(row=row, column=1, value=slate_label)
        cell.font = Font(bold=True, size=12, color="1C2A4A")
        row += 2

    for caption, path in figures:
        title = sheet.cell(row=row, column=1, value=caption)
        title.font = Font(bold=True, size=11, color="1C2A4A")
        row += 1
        image = XLImage(path)
        target = 1380
        image.height = int(image.height * target / image.width)
        image.width = target
        sheet.add_image(image, f"A{row}")
        # An anchored image floats over cells rather than occupying them, so the rows under
        # it must be reserved or the next caption lands on top. 17px = a default row.
        row += -(-image.height // 17) + 2
    return sheet
