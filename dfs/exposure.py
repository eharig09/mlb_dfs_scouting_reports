"""What a set of lineups actually committed to, player by player and team by team.

A board says who is worth playing; the lineups say who you are *actually* playing, and on a
150-lineup set those are not the same statement. One player quietly lands in 70% of the set
because he is cheap and correlated with everything, another that the pool file asked for
never shows up at all, and neither is visible from reading lineups one at a time.

Two things make this more than a tally:

  targets   The pool file's Min%/Max% are requests. Reporting the delivered share beside the
            requested one is the only way to see which requests the roster could not honour
            -- exposure is a property of the whole set, so nothing earlier in the run can
            tell you.
  leverage  Exposure only means something against what the field is doing. A 40% exposure is
            aggressive on a 10%-owned player and a fade on a 60%-owned one, so projected
            ownership is carried through and the gap is reported as Lev.
"""

import numpy as np
import pandas as pd

# Hitters from one team in a single lineup, at or above which the lineup counts as stacking
# that team. Three is where correlation starts paying in MLB GPPs.
STACK_AT = 3


def _numeric(frame, column, default=np.nan):
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def player_exposure(lineups, exposure=None, players=None):
    """One row per player used, with delivered share, requested bounds and leverage.

    `exposure` is the optimizer's resolved {name: (min lineups, max lineups)}; players it
    names are kept even when they never appeared, since a minimum that delivered nothing is
    the single most important row in the report and dropping it would hide it.
    """
    total = len(lineups)
    if not total:
        return pd.DataFrame()

    counts, slots, info = {}, {}, {}
    for lineup in lineups:
        for _, row in lineup["players"].iterrows():
            name = row["Name"]
            counts[name] = counts.get(name, 0) + 1
            slots.setdefault(name, []).append(row.get("Roster"))
            info.setdefault(name, row)

    # A player the pool file demanded who never made a lineup has no row to read details
    # from, so fall back to the slate.
    by_name = {}
    if players is not None and "Name" in getattr(players, "columns", []):
        by_name = {row["Name"]: row for _, row in players.iterrows()}
    for name in exposure or {}:
        if name not in counts:
            counts[name] = 0
            slots[name] = []
            if name in by_name:
                info[name] = by_name[name]

    rows = []
    for name, used in counts.items():
        row = info.get(name)
        if row is None:
            row = by_name.get(name, {})
        seen = [s for s in slots.get(name, []) if s]
        rows.append({
            "Name": name,
            "Team": row.get("Team"),
            "Opp": row.get("Opp"),
            # The slot actually used most, not DK eligibility: on a set this size that is
            # where the player is really being deployed.
            "Roster": max(set(seen), key=seen.count) if seen else row.get("DK Pos"),
            "Salary": row.get("Salary"),
            "Proj": row.get("Proj"),
            "Ceiling": row.get("Ceiling"),
            "Own%": row.get("Own%"),
            "Lineups": used,
            "Exp%": round(100.0 * used / total, 1),
        })

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    limits = exposure or {}
    frame["Min%"] = [round(100.0 * limits[n][0] / total, 1) if n in limits and limits[n][0]
                     else np.nan for n in frame["Name"]]
    frame["Max%"] = [round(100.0 * limits[n][1] / total, 1)
                     if n in limits and limits[n][1] < total else np.nan
                     for n in frame["Name"]]

    # Own% is already a 0-100 percentage from the ownership model (and from a real contest
    # export), so it subtracts from Exp% directly -- no rescaling, which would only ever
    # misfire on a slate where the field happened to be thin everywhere.
    own = _numeric(frame, "Own%")
    frame["Own%"] = own.round(1)
    frame["Lev"] = (frame["Exp%"] - own).round(1)

    def status(row):
        name = row["Name"]
        low, high = limits.get(name, (0, total))
        if row["Lineups"] < low:
            return f"SHORT {low - row['Lineups']}"
        if high < total and row["Lineups"] >= high:
            return "at cap"
        return ""

    frame["Status"] = frame.apply(status, axis=1)
    return frame.sort_values(["Lineups", "Proj"], ascending=False).reset_index(drop=True)


def team_exposure(lineups):
    """One row per team: hitter slots spent, how often it was stacked, and how hard."""
    total = len(lineups)
    if not total:
        return pd.DataFrame()

    teams = {}
    for lineup in lineups:
        frame = lineup["players"]
        hitters = frame[frame["Roster"] != "P"]
        counted = hitters["Team"].value_counts()
        for team, count in counted.items():
            entry = teams.setdefault(team, {"slots": 0, "lineups": 0, "stacked": 0,
                                            "biggest": 0, "sizes": []})
            entry["slots"] += int(count)
            entry["lineups"] += 1
            entry["sizes"].append(int(count))
            entry["biggest"] = max(entry["biggest"], int(count))
            if count >= STACK_AT:
                entry["stacked"] += 1
        for team in frame[frame["Roster"] == "P"]["Team"]:
            teams.setdefault(team, {"slots": 0, "lineups": 0, "stacked": 0,
                                    "biggest": 0, "sizes": []})
            teams[team].setdefault("pitchers", 0)
            teams[team]["pitchers"] = teams[team].get("pitchers", 0) + 1

    rows = []
    for team, entry in teams.items():
        rows.append({
            "Team": team,
            "Hitter Slots": entry["slots"],
            "Lineups": entry["lineups"],
            "In%": round(100.0 * entry["lineups"] / total, 1),
            f"Stacked {STACK_AT}+": entry["stacked"],
            f"Stack%": round(100.0 * entry["stacked"] / total, 1),
            "Biggest": entry["biggest"],
            "Avg When In": round(float(np.mean(entry["sizes"])), 1) if entry["sizes"] else 0.0,
            "P Slots": entry.get("pitchers", 0),
        })
    frame = pd.DataFrame(rows)
    return frame.sort_values(["Hitter Slots", "Lineups"], ascending=False).reset_index(drop=True)


def _cell(value, kind):
    """One value as text. Blank for missing, so an absent Min% reads as absent."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if kind == "int":
        return f"{int(round(float(value))):,}"
    if kind == "num":
        return f"{float(value):,.1f}"
    return str(value)


def _table(frame, spec):
    """Fixed-width text table from [(column, header, width, align, kind)].

    Alignment is per column rather than "first left, rest right": team codes and roster
    slots are labels, and right-aligning them pushes them away from the name they belong to,
    which is exactly the column you scan down.
    """
    spec = [s for s in spec if s[0] in frame.columns]
    head = "  ".join(f"{header:{'<' if align == 'l' else '>'}{width}}"
                     for _, header, width, align, _ in spec).rstrip()
    lines = [head, "  ".join("-" * width for _, _, width, _, _ in spec)]
    for _, row in frame.iterrows():
        cells = []
        for column, _, width, align, kind in spec:
            text = _cell(row.get(column), kind)[:width]
            cells.append(f"{text:{'<' if align == 'l' else '>'}{width}}")
        lines.append("  ".join(cells).rstrip())
    return "\n".join(lines)


PLAYER_SPEC = [
    ("Name",    "Name",   22, "l", "str"),
    ("Team",    "Team",    4, "l", "str"),
    ("Roster",  "Pos",     4, "l", "str"),
    ("Salary",  "Salary",  7, "r", "int"),
    ("Proj",    "Proj",    5, "r", "num"),
    ("Lineups", "N",       4, "r", "int"),
    ("Exp%",    "Exp%",    6, "r", "num"),
    ("Min%",    "Min%",    6, "r", "num"),
    ("Max%",    "Max%",    6, "r", "num"),
    ("Own%",    "Own%",    6, "r", "num"),
    ("Lev",     "Lev",     6, "r", "num"),
    ("Status",  "Status",  8, "l", "str"),
]

TEAM_SPEC = [
    ("Team",         "Team",   4, "l", "str"),
    ("Hitter Slots", "Bats",   6, "r", "int"),
    ("Lineups",      "N",      5, "r", "int"),
    ("In%",          "In%",    6, "r", "num"),
    (f"Stacked {STACK_AT}+", f"Stk{STACK_AT}+", 6, "r", "int"),
    ("Stack%",       "Stk%",   6, "r", "num"),
    ("Biggest",      "Max",    4, "r", "int"),
    ("Avg When In",  "AvgIn",  6, "r", "num"),
]


def format_exposure(players, teams, total, top=20, show_teams=8):
    """Console summary: the heaviest exposures, every unmet request, the stacked teams."""
    out = [f"=== Exposure across {total} lineup(s) ==="]

    if players is None or players.empty:
        return "\n".join(out + ["  (no lineups)"])

    shown = players if not top else players.head(top)
    out.append(_table(shown, PLAYER_SPEC))
    if top and len(players) > top:
        out.append(f"  ... {len(players) - top} more in the csv "
                   f"({len(players)} players used in total)")

    missed = players[players["Status"].str.startswith("SHORT", na=False)]
    if not missed.empty:
        out.append("")
        out.append(f"[!] {len(missed)} Min% request(s) the roster could not meet:")
        for _, row in missed.iterrows():
            # Counts, not percentages: Min% is enforced as a whole number of lineups, and
            # reporting "34.7% vs 35.3%" for a one-lineup miss reads like a rounding fault
            # rather than the single lineup it actually is.
            want = int(round(float(row["Min%"]) * total / 100))
            out.append(f"    {str(row['Name'])[:24]:<24} {row['Lineups']:>4} of {want:<4} "
                       f"lineups ({row['Exp%']:.1f}%)")
        out.append("    Min% is a floor the optimizer paces toward, not a guarantee -- "
                   "minimums can add up to more than the set can seat.")

    capped = players[players["Status"] == "at cap"]
    if not capped.empty:
        out.append("")
        out.append(f"  {len(capped)} player(s) held at their Max%: "
                   + ", ".join(str(n)[:20] for n in capped["Name"].head(8))
                   + (" ..." if len(capped) > 8 else ""))

    if teams is not None and not teams.empty and show_teams:
        out.append("")
        out.append(f"--- Teams (Bats = hitter slots spent; Stk{STACK_AT}+ = lineups with "
                   f"{STACK_AT}+ of that team's bats) ---")
        out.append(_table(teams.head(show_teams), TEAM_SPEC))

    return "\n".join(out)
