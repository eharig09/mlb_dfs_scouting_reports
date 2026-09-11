"""Browsing the generated report workbooks from inside the app — including when hosted.

The pipeline writes Excel: `scouting_reports/<date>/scouting_report_<date>_<teams>.xlsx` for
baseball, `nfl/nfl_reports/<date>/slate_<date>_<teams>.xlsx` for football. They are the
thing you actually read on a given morning, and until now they could only be opened on the
machine that made them.

## Why this is not just "open the xlsx"

**The workbooks cannot be committed.** There are 406 of them at 146 MB, on top of the 97 MB
of payloads the MLB pages already need. And most of that weight is not data: a workbook is
443 KB of which the frames are a few thousand cells — the rest is embedded figures and
formatting, neither of which survives a trip through pandas anyway.

So the same bargain the NFL snapshot struck: **publish the sheets as parquet** and commit
that. `publish()` converts whichever reports you choose; the page reads the published bundle
when the workbook itself is not on disk, which is always the case on a host.

**Reading xlsx needs `openpyxl`, and the hosted app does not have it** -- deliberately, the
same way it does not have scipy. Locally the explorer reads workbooks directly; hosted it
reads the bundle. `available()` says which path is live rather than letting an ImportError
decide.
"""

import json
import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

# Where the pipeline writes, in the order a reader thinks about them.
REPORT_ROOTS = (
    ("MLB", "scouting_reports"),
    ("NFL", os.path.join("nfl", "nfl_reports")),
)

PUBLISHED_DIR = os.path.join("reports_published")
MANIFEST = "manifest.json"

# Sheets that carry embedded figures rather than data. Reading them through pandas produces a
# grid of NaN with the odd caption in it, which looks like a broken table rather than a
# picture that did not come along.
IMAGE_SHEETS = {"Visuals"}


def available():
    """Which way of reading a workbook is live: `("xlsx", "published")`, either, or neither."""
    modes = []
    try:
        import openpyxl  # noqa: F401
        modes.append("xlsx")
    except ImportError:
        pass
    if os.path.isdir(PUBLISHED_DIR):
        modes.append("published")
    return tuple(modes)


@st.cache_data(ttl="5m", show_spinner=False)
def list_workbooks(roots=REPORT_ROOTS):
    """Every generated workbook on disk: sport, date, name, path, size."""
    rows = []
    for sport, root in roots:
        if not os.path.isdir(root):
            continue
        for date in sorted(os.listdir(root), reverse=True):
            folder = os.path.join(root, date)
            if not os.path.isdir(folder):
                continue
            for name in sorted(os.listdir(folder)):
                if not name.lower().endswith(".xlsx") or name.startswith("~$"):
                    continue
                path = os.path.join(folder, name)
                rows.append({"Sport": sport, "Date": date, "Report": _label(name),
                             "File": name, "Path": path,
                             "KB": round(os.path.getsize(path) / 1024)})
    return pd.DataFrame(rows)


def _label(name):
    """`scouting_report_2026-08-26_TEX_CWS.xlsx` -> `TEX_CWS`; `slate_..._A-B.xlsx` -> `A-B`."""
    stem = os.path.splitext(name)[0]
    for prefix in ("scouting_report_", "slate_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    parts = stem.split("_", 1)
    return parts[1] if len(parts) > 1 and parts[0][:2] == "20" else stem


@st.cache_data(ttl="10m", max_entries=8, show_spinner=False)
def sheet_names(path):
    """Sheets in a workbook, from the file or from the published bundle."""
    published = _published_manifest().get(_key(path))
    if published:
        return list(published["sheets"])
    if "xlsx" not in available() or not os.path.exists(path):
        return []
    with pd.ExcelFile(path) as book:
        return list(book.sheet_names)


@st.cache_data(ttl="10m", max_entries=64, show_spinner=False)
def read_sheet(path, sheet):
    """One sheet as a frame. Published bundle first, then the workbook itself.

    Published first on purpose: on a host the workbook is not there at all, and locally the
    two are the same data, so preferring the fast path costs nothing and keeps one code
    path warm in both places.
    """
    key = _key(path)
    published = _published_manifest().get(key)
    if published and sheet in published["sheets"]:
        target = os.path.join(PUBLISHED_DIR, key, _safe(sheet) + ".parquet")
        if os.path.exists(target):
            try:
                return pd.read_parquet(target)
            except Exception:
                pass
    if "xlsx" not in available() or not os.path.exists(path):
        return None
    try:
        return pd.read_excel(path, sheet_name=sheet)
    except Exception:
        return None


def _key(path):
    """A stable id for a report, independent of the machine it was generated on."""
    return os.path.splitext(os.path.basename(str(path)))[0]


def _safe(name):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name))


@st.cache_data(ttl="5m", show_spinner=False)
def _published_manifest():
    path = os.path.join(PUBLISHED_DIR, MANIFEST)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle).get("reports", {})


def clean(frame):
    """Make a sheet readable: promote the real header, drop the padding.

    An Excel sheet written for a human is a **document**, and two of its habits break a
    naive read:

    **A merged title bar becomes the header row.** `pd.read_excel` takes row 1 as the
    column names, so a sheet titled "BAL at STL | 2026-08-26 | HP Umpire: ..." comes back
    with that sentence as its first column name and `Unnamed: 1..19` for the rest — the
    actual column names sitting in row 2 as data. Where most of the header is unnamed, the
    first row that fills most of the width is promoted instead.

    **Blank columns cannot all be called "".** Renaming every `Unnamed:` to the empty string
    produced twenty identically-named columns, which pandas tolerates and parquet refuses
    outright — `Duplicate column names found`. They are padded with spaces instead: unique
    as strings, and indistinguishable from blank on the page.
    """
    if frame is None or frame.empty:
        return frame
    out = frame.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if out.empty:
        return out.reset_index(drop=True)

    unnamed = [c for c in out.columns if str(c).startswith("Unnamed:")]
    if len(unnamed) > len(out.columns) * 0.6 and len(out) > 1:
        # The header is somewhere below a title bar. Take the first row that fills most of
        # the width -- a spacer row will not.
        for position in range(min(4, len(out))):
            candidate = out.iloc[position]
            if candidate.notna().sum() >= max(2, len(out.columns) * 0.6):
                out = out.iloc[position + 1:].copy()
                out.columns = [str(v) if pd.notna(v) else "" for v in candidate]
                out = out.dropna(axis=1, how="all").dropna(axis=0, how="all")
                break

    seen, names = {}, []
    for column in out.columns:
        label = "" if str(column).startswith("Unnamed:") else str(column)
        if label in seen:
            seen[label] += 1
            label = label + " " * seen[label] if label == "" else f"{label}.{seen[label]}"
        else:
            seen[label] = 0
        names.append(label)
    out.columns = names
    return out.reset_index(drop=True)


def publish(paths, root=PUBLISHED_DIR, report=print):
    """Convert workbooks to a committed parquet bundle. Returns the manifest written.

    Sheets that are pictures rather than tables are skipped and recorded as skipped, so a
    reader can tell "this sheet had no data" from "this sheet did not come along".
    """
    if "xlsx" not in available():
        raise RuntimeError("publishing needs openpyxl; it is a local-only step")

    os.makedirs(root, exist_ok=True)
    reports, total = {}, 0
    for path in paths:
        key = _key(path)
        try:
            with pd.ExcelFile(path) as book:
                names = list(book.sheet_names)
        except Exception as error:
            report(f"  [skip] {key}: {error.__class__.__name__}")
            continue

        folder = os.path.join(root, key)
        os.makedirs(folder, exist_ok=True)
        kept, skipped = [], []
        for sheet in names:
            if sheet in IMAGE_SHEETS:
                skipped.append(sheet)
                continue
            try:
                frame = clean(pd.read_excel(path, sheet_name=sheet))
            except Exception:
                skipped.append(sheet)
                continue
            if frame is None or frame.empty:
                skipped.append(sheet)
                continue
            # Every column to string: a sheet written for a human mixes numbers, blanks and
            # the odd note in one column, and parquet will not take that as a typed column.
            # The explorer displays them, it does not compute on them.
            target = os.path.join(folder, _safe(sheet) + ".parquet")
            frame.astype("string").to_parquet(target, index=False)
            kept.append(sheet)
            total += os.path.getsize(target)
        reports[key] = {"sheets": kept, "skipped": skipped,
                        "source": path.replace("\\", "/")}
        report(f"  {key}: {len(kept)} sheets")

    manifest = {"built": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "reports": reports, "bytes": total}
    with open(os.path.join(root, MANIFEST), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True)
    return manifest


def published_reports():
    """`{report key: {sheets, skipped, source}}` for whatever has been published."""
    return _published_manifest()
