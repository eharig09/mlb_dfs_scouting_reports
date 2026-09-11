"""Pure comparison helpers for immutable DFS snapshot stages.

The snapshot layer already preserves morning, confirmed and final boards. This module turns
two of those boards into one audit table without rebuilding either stage from today's
mutable payload cache.
"""

import numpy as np
import pandas as pd


NUMERIC = ("Proj", "Ceiling", "Floor", "Own%", "Salary", "PA", "Team Runs", "GPP",
           "CASH", "Leverage")
DETAILS = ("Lineup", "Slot", "Opp SP", "Opp SP Hand", "Team", "Opp", "Role")


def _identity(frame):
    """Stable player-slate key, preferring the contest id over name matching."""
    out = frame.copy()
    game = out.get("Game", pd.Series("", index=out.index)).fillna("").astype(str)
    dk_id = pd.to_numeric(out.get("DK ID", pd.Series(np.nan, index=out.index)),
                          errors="coerce")
    mlbam = pd.to_numeric(out.get("MLBAM", pd.Series(np.nan, index=out.index)),
                          errors="coerce")
    fallback = (out.get("Name", pd.Series("", index=out.index)).fillna("").astype(str)
                .str.casefold().str.replace(r"\W+", "", regex=True)
                + "|" + out.get("Team", pd.Series("", index=out.index)).fillna("")
                .astype(str) + "|" + game)
    keys = fallback
    keys = keys.where(mlbam.isna(), "mlbam:" + mlbam.fillna(0).astype("int64").astype(str)
                      + "|" + game)
    keys = keys.where(dk_id.isna(), "dk:" + dk_id.fillna(0).astype("int64").astype(str))
    out["_key"] = keys
    return out.drop_duplicates("_key", keep="last")


def compare(before, after):
    """Outer-join two snapshot player boards and calculate changes worth auditing."""
    before = _identity(before if isinstance(before, pd.DataFrame) else pd.DataFrame())
    after = _identity(after if isinstance(after, pd.DataFrame) else pd.DataFrame())
    available = list(dict.fromkeys(
        c for c in ("Name", "Team", "Type", "Game", *NUMERIC, *DETAILS)
        if c in before.columns or c in after.columns))
    left = before.reindex(columns=["_key"] + [c for c in available if c in before.columns])
    right = after.reindex(columns=["_key"] + [c for c in available if c in after.columns])
    merged = left.merge(right, on="_key", how="outer", suffixes=(" Before", " After"),
                        indicator=True)

    out = pd.DataFrame(index=merged.index)
    for column in ("Name", "Team", "Type", "Game"):
        old = merged.get(f"{column} Before", pd.Series(np.nan, index=merged.index))
        new = merged.get(f"{column} After", pd.Series(np.nan, index=merged.index))
        out[column] = new.combine_first(old)

    out["Status"] = merged["_merge"].map(
        {"left_only": "Removed", "right_only": "Added", "both": "Present"}).astype(str)

    for column in NUMERIC:
        old = pd.to_numeric(merged.get(f"{column} Before",
                                      pd.Series(np.nan, index=merged.index)), errors="coerce")
        new = pd.to_numeric(merged.get(f"{column} After",
                                      pd.Series(np.nan, index=merged.index)), errors="coerce")
        if old.notna().any() or new.notna().any():
            out[f"{column} Before"] = old
            out[f"{column} After"] = new
            out[f"Δ {column}"] = new - old

    for column in DETAILS:
        old = merged.get(f"{column} Before", pd.Series(np.nan, index=merged.index))
        new = merged.get(f"{column} After", pd.Series(np.nan, index=merged.index))
        if old.notna().any() or new.notna().any():
            out[f"{column} Before"] = old
            out[f"{column} After"] = new

    changed_details = []
    for index in merged.index:
        changed = []
        for column in DETAILS:
            old = merged.at[index, f"{column} Before"] if f"{column} Before" in merged else np.nan
            new = merged.at[index, f"{column} After"] if f"{column} After" in merged else np.nan
            if pd.isna(old) and pd.isna(new):
                continue
            if str(old) != str(new):
                changed.append(f"{column}: {old if pd.notna(old) else '—'} → "
                               f"{new if pd.notna(new) else '—'}")
        changed_details.append("; ".join(changed))
    out["Details"] = changed_details

    numeric_change = [c for c in out.columns if c.startswith("Δ ")]
    moved = (out[numeric_change].abs().fillna(0).gt(1e-9).any(axis=1)
             if numeric_change else pd.Series(False, index=out.index))
    detail_changed = out["Details"].astype(bool)
    present = out["Status"].eq("Present")
    out.loc[present & (moved | detail_changed), "Status"] = "Changed"
    out.loc[present & ~(moved | detail_changed), "Status"] = "Unchanged"
    return out.reset_index(drop=True)


def confirmed_between(frame):
    """How many present players became confirmed between the two stages."""
    before = frame.get("Lineup Before", pd.Series("", index=frame.index)).astype(str)
    after = frame.get("Lineup After", pd.Series("", index=frame.index)).astype(str)
    present = ~frame.get("Status", pd.Series("", index=frame.index)).isin(
        ("Added", "Removed"))
    return int((present & before.ne("Confirmed") & after.eq("Confirmed")).sum())
