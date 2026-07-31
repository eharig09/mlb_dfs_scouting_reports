"""Render the slate as a readable markdown board."""

import pandas as pd

from .schedule import describe_postponed

POSITION_ORDER = ["C", "1B", "2B", "3B", "SS", "OF"]
TIER_MARK = {"Core": "**CORE**", "Value": "VALUE", "Leverage": "LEV", "Risk": "RISK", "Fade": "fade", "Neutral": ""}


def _fmt(value, spec="{:.1f}", blank="-"):
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return blank
    try:
        return spec.format(float(value))
    except (TypeError, ValueError):
        return str(value)


def _salary(value):
    if value is None or pd.isna(value):
        return "-"
    return f"${int(value):,}"


def _table(header, rows):
    if not rows:
        return "_none_\n"
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def _eligible_positions(row):
    raw = str(row.get("DK Pos") or row.get("Pos") or "").strip()
    if not raw or raw.lower() == "none":
        raw = str(row.get("Pos") or "")
    slots = {p.strip().upper() for p in raw.replace(",", "/").split("/") if p.strip()}
    slots = {("LF", "CF", "RF", "OF")[3] if p in {"LF", "CF", "RF"} else p for p in slots}
    slots.discard("DH")
    return {p for p in slots if p in POSITION_ORDER}


def _pitcher_rows(players, has_salary, limit):
    rows = []
    for _, row in players[players["Type"] == "P"].head(limit).iterrows():
        cells = [
            f"{row['Name']} ({row['Team']})",
            f"vs {row['Opp']}",
            _salary(row["Salary"]) if has_salary else "-",
            _fmt(row["Proj"]),
            _fmt(row["Ceiling"]),
            _fmt(row["Value"], "{:.2f}") if has_salary else "-",
            f"{_fmt(row.get('IP'))} / {_fmt(row.get('K'))}",
            _fmt(row["GPP"]),
            _fmt(row.get("Edge"), "{:+.0f}"),
            TIER_MARK.get(row["Tier"], ""),
            row["Why"],
        ]
        rows.append(cells)
    return rows


def _hitter_rows(group, has_salary, limit):
    rows = []
    for _, row in group.head(limit).iterrows():
        rows.append([
            f"{row['Name']} ({row['Team']})",
            int(row["Slot"]),
            f"vs {row['Opp SP']}",
            _salary(row["Salary"]) if has_salary else "-",
            _fmt(row["Proj"]),
            _fmt(row["Ceiling"]),
            _fmt(row["Value"], "{:.2f}") if has_salary else "-",
            _fmt(row["GPP"]),
            _fmt(row.get("Edge"), "{:+.0f}"),
            TIER_MARK.get(row["Tier"], ""),
            row["Why"],
        ])
    return rows


def _role_sections(players, top_pitchers=6, top_hitters=12):
    """Ceiling plays and floor plays, ranked independently of the GPP tier.

    Pitchers and hitters are listed separately throughout: their percentiles are computed
    within type, so a pitcher's score and a hitter's score are not on a comparable scale.
    """
    out = []

    out += ["## Ceiling Plays", "",
            "_Ranked on ceiling per dollar — the tournament view. `Bust%` is the chance of "
            "3 or fewer points, and it is what you are paying for the upside._", ""]
    ceiling = players[players["Role"].isin(["Ceiling", "Ceiling+Floor"])]
    for player_type, label, limit in (("P", "Pitchers", top_pitchers), ("H", "Hitters", top_hitters)):
        group = ceiling[ceiling["Type"] == player_type].sort_values("Ceil Value", ascending=False)
        if group.empty:
            continue
        out += [f"### {label}", ""]
        out.append(_table(
            ["Player", "Pos", "Matchup", "Salary", "Proj", "Ceil", "Ceil/$", "Bust%", "GPP", "Why"],
            [[
                r["Name"], r.get("DK Pos") or r["Pos"], f"{r['Team']} vs {r['Opp']}",
                _salary(r["Salary"]), _fmt(r["Proj"]), _fmt(r["Ceiling"]),
                _fmt(r["Ceil Value"], "{:.2f}"), _fmt(r["Bust%"], "{:.0f}%"),
                _fmt(r["GPP"]), r["Why"],
            ] for _, r in group.head(limit).iterrows()],
        ))

    out += ["## Floor Plays", "",
            "_Ranked on floor per dollar and bust avoidance — the cash view. Floor is the "
            "25th-percentile outcome, fit from actual results rather than assumed._",
            "",
            "> Backtesting found **27% of hitter-games score zero and 44% score 3 or fewer**, "
            "regardless of matchup. A genuinely safe MLB hitter does not exist in a single "
            "game — the floor plays that hold up are pitchers. Weight the hitter list "
            "accordingly.", ""]
    floor = players[players["Role"].isin(["Floor", "Ceiling+Floor"])]
    for player_type, label, limit in (("P", "Pitchers", top_pitchers), ("H", "Hitters", top_hitters)):
        group = floor[floor["Type"] == player_type].sort_values("CASH", ascending=False)
        if group.empty:
            continue
        out += [f"### {label}", ""]
        out.append(_table(
            ["Player", "Pos", "Matchup", "Salary", "Proj", "Floor", "Floor/$", "Bust%", "CASH", "Why"],
            [[
                r["Name"], r.get("DK Pos") or r["Pos"], f"{r['Team']} vs {r['Opp']}",
                _salary(r["Salary"]), _fmt(r["Proj"]), _fmt(r["Floor"]),
                _fmt(r["Floor Value"], "{:.2f}"), _fmt(r["Bust%"], "{:.0f}%"),
                _fmt(r["CASH"]), r["Why"],
            ] for _, r in group.head(limit).iterrows()],
        ))

    both = players[players["Role"] == "Ceiling+Floor"].sort_values("GPP", ascending=False)
    if not both.empty:
        listed = ", ".join(f"{r['Name']} ({r['Team']})" for _, r in both.head(10).iterrows())
        out += [f"**Both ceiling and floor:** {listed}", ""]
    return out


def render_board(players, stacks, meta, top_pitchers=14, top_per_position=8):
    has_salary = meta.get("has_salary", False)
    out = [f"# DFS Board — {meta['date']} (DraftKings Classic, GPP lean)", ""]

    out.append(f"**Games:** {len(meta['games'])} — {', '.join(meta['games'])}" if meta["games"] else "**Games:** none")
    if meta.get("environment_error"):
        # Salaries are never reached when loading fails, so reporting on them here would
        # just be a second, misleading problem to chase.
        pass
    elif has_salary:
        out.append(f"**Salaries:** `{meta['salary_file']}` — {meta['matched']} priced, "
                   f"{meta['unmatched']} projected players not on the slate.")
    else:
        out.append("**Salaries:** none found. Drop `DKSalaries.csv` in `dfs_daily_files/` to get "
                   "value-per-dollar and tiering; the board below ranks on projection and ceiling only.")
    if meta.get("skipped"):
        out.append(f"**Skipped:** {'; '.join(meta['skipped'])}")
    if meta.get("postponed"):
        out += ["",
                f"> **POSTPONED — {describe_postponed(meta['postponed'])}.**",
                f"> {meta['postponed_players']} priced player(s) removed from the pool. DK "
                f"leaves them in the export at full salary, so a board built without this "
                f"check would rank players who are not going to appear."]
    if meta.get("schedule_error"):
        out += ["",
                f"> **Postponement check failed** — `{meta['schedule_error']}`. A game called "
                f"off after the salary file was downloaded would not have been caught."]
    if meta.get("stale_salaries"):
        out += ["",
                f"> **WRONG SALARY FILE — `{meta['salary_file']}` prices the "
                f"{meta['salary_date']} slate, not {meta['date']}.**",
                "> Every salary, value, and tier below is meaningless. Download this "
                "slate's export before using any of it."]
    if meta.get("missing_games") and not meta.get("stale_salaries"):
        missing = ", ".join(meta["missing_games"])
        out += ["",
                f"> **INCOMPLETE SLATE — {missing} priced by DK but not in the report cache.**",
                f"> Those teams are absent from every table below, so this board is not a full "
                f"view of the player pool. Generate them first:",
                ">",
                f"> `python scouting_report.py --date {meta['date']} --away-team AWAY "
                f"--home-team HOME --format both`"]
    out.append("")

    if players.empty:
        if meta.get("environment_error"):
            out += [
                f"> **ENVIRONMENT PROBLEM — not missing data.** "
                f"{meta['cached_but_unreadable']} cached games for {meta['date']} are on "
                f"disk but could not be read: `{meta['environment_error']}`.",
                ">",
                "> Do **not** regenerate the reports — they already exist. This is the "
                "wrong Python interpreter or an incomplete install. Call the project "
                "interpreter explicitly rather than plain `python`:",
                ">",
                "> ```powershell",
                "> .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt",
                "> .\\.venv\\Scripts\\python.exe -m dfs.cli --date " + str(meta["date"]),
                "> ```",
            ]
        else:
            out.append(f"_No cached game data for {meta['date']}. Generate it first: "
                       f"`python scouting_report.py --date {meta['date']} --all-games`._")
        return "\n".join(out)

    out += ["## Pitchers", ""]
    out.append(_table(
        ["Pitcher", "Matchup", "Salary", "Proj", "Ceil", "Val", "IP / K", "GPP", "Edge", "Tier", "Why"],
        _pitcher_rows(players, has_salary, top_pitchers),
    ))

    if not stacks.empty:
        out += ["## Team Stacks", "",
                "_Ranked on the top five lineup spots — ceiling per dollar where priced._", ""]
        out.append(_table(
            ["Team", "Opp SP", "Proj Runs", "Top5 Proj", "Top5 Ceil", "Top5 Salary", "Stack Val", "Score", "Matchup"],
            [[
                f"{r['Team']} (vs {r['Opp']})", r["Opp SP"], _fmt(r["Team Runs"]),
                _fmt(r["Top5 Proj"]), _fmt(r["Top5 Ceiling"]),
                _salary(r["Top5 Salary"]), _fmt(r["Stack Value"], "{:.2f}"),
                _fmt(r.get("Stack Score")), _fmt(r["Avg Matchup"], "{:.3f}"),
            ] for _, r in stacks.iterrows()],
        ))

    hitters = players[players["Type"] == "H"].copy()
    out += ["## Hitters by Position", ""]
    for position in POSITION_ORDER:
        eligible = hitters[hitters.apply(lambda row: position in _eligible_positions(row), axis=1)]
        if eligible.empty:
            continue
        out += [f"### {position}", ""]
        out.append(_table(
            ["Hitter", "Slot", "Opp SP", "Salary", "Proj", "Ceil", "Val", "GPP", "Edge", "Tier", "Why"],
            _hitter_rows(eligible.sort_values("GPP", ascending=False), has_salary, top_per_position),
        ))

    if has_salary:
        out += _role_sections(players)

    out += ["## Tier Summary", ""]
    for tier in ("Core", "Value", "Leverage", "Risk"):
        names = players[players["Tier"] == tier].sort_values("GPP", ascending=False)
        if names.empty:
            continue
        listed = ", ".join(f"{r['Name']} ({r['Team']})" for _, r in names.head(12).iterrows())
        out.append(f"- **{tier}** — {listed}")
    out.append("")
    out.append("_Projections are matchup-adjusted expectations, not predictions. "
               "Tiers are the recommendation; the point totals are the reasoning behind them._")
    return "\n".join(out)
