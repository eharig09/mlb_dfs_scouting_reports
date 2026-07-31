"""Score a night's lineups against what actually happened.

Answers three questions: how did the lineups we built do, what was the best lineup that
was buildable at all, and which optimizer settings would have produced the best set.
"""

import os
import pickle

import numpy as np
import pandas as pd

from .backtest import actual_points
from .optimizer import DEFAULT_MAX_OVERLAP, DEFAULT_RANDOMNESS, optimize
from .slate import build_slate, cached_games
from .salaries import normalize_name


def slate_actuals(date, data_dir=None):
    """{mlbam id: actual DK points} for every finished game on the date."""
    from .slate import REPORT_DATA_DIR
    results = {}
    for path, _, _ in cached_games(date, data_dir or REPORT_DATA_DIR):
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        game_id = (payload["advanced_context"].get("environment") or {}).get("game_id")
        if game_id:
            results.update(actual_points(int(game_id)))
    return results


def attach_actuals(players, actuals):
    """Add an Actual column to a slate frame, matched on MLBAM id."""
    frame = players.copy()
    frame["Actual"] = frame["MLBAM"].map(
        lambda pid: None if pd.isna(pid) else (actuals.get(int(pid)) or {}).get("Actual")
    )
    return frame


def score_lineup(frame, actuals):
    """Total actual DK points for a lineup, plus how many players have a result."""
    total, scored = 0.0, 0
    for _, row in frame.iterrows():
        pid = row.get("MLBAM")
        result = None if pd.isna(pid) else actuals.get(int(pid))
        if result is not None:
            total += result["Actual"]
            scored += 1
    return round(total, 2), scored


def perfect_lineup(players, actuals):
    """The highest-scoring legal lineup in hindsight -- the night's ceiling."""
    frame = attach_actuals(players, actuals)
    frame = frame[frame["Actual"].notna()].copy()
    if frame.empty:
        return None
    # Reuse the optimizer with actual points as the objective. randomness must be 0 here:
    # this is the true maximum, not an exploration, so it must not inherit the tournament
    # default that deliberately perturbs the objective.
    frame["Ceiling"] = frame["Actual"]
    lineups, _, _ = optimize(frame, n_lineups=1, objective="ceiling",
                             stack_bonus=False, randomness=0.0)
    return lineups[0] if lineups else None


def tier_report(players, actuals):
    """How each Tier and Role actually did. The honest scorecard for a night."""
    scored = attach_actuals(players, actuals)
    scored = scored[scored["Actual"].notna()]
    if scored.empty:
        return pd.DataFrame(), pd.DataFrame()

    def summarize(column, order):
        rows = []
        for value in order:
            group = scored[scored[column] == value]
            if group.empty:
                continue
            rows.append({
                column: value, "n": len(group),
                "Proj": round(group["Proj"].mean(), 2),
                "Actual": round(group["Actual"].mean(), 2),
                "Diff": round(group["Actual"].mean() - group["Proj"].mean(), 2),
                "Bust%": round((group["Actual"] <= 3).mean() * 100),
            })
        return pd.DataFrame(rows)

    tiers = summarize("Tier", ["Core", "Value", "Leverage", "Risk", "Neutral", "Fade"])
    roles = summarize("Role", ["Ceiling", "Floor", "Ceiling+Floor"])
    return tiers, roles


def team_report(players, actuals):
    """Projected vs actual hitter points per team -- where stacks were won and lost."""
    scored = attach_actuals(players, actuals)
    scored = scored[(scored["Actual"].notna()) & (scored["Type"] == "H")]
    if scored.empty:
        return pd.DataFrame()
    table = scored.groupby("Team").agg(
        n=("Actual", "size"), Proj=("Proj", "sum"), Actual=("Actual", "sum")).round(1)
    table["Diff"] = (table["Actual"] - table["Proj"]).round(1)
    return table.sort_values("Actual", ascending=False)


def read_entered(path, players):
    """Lineups actually entered at DK -> frames of slate rows, keyed by lineup number.

    An upload file is ten DK ids per row and nothing else, so the slate is what turns those
    back into players. Ids are matched as strings: DK's are opaque numbers and round-tripping
    them through a float silently corrupts the tail digits on a big draft group.
    """
    from .lateswap import read_filled
    from .upload import read_template

    rows, kind, slot_start, _ = read_template(path)
    by_dk = {}
    for _, row in players.iterrows():
        dk_id = str(row.get("DK ID") or "").strip()
        if dk_id.endswith(".0"):
            dk_id = dk_id[:-2]
        if dk_id:
            by_dk[dk_id] = row

    entered, unknown = {}, set()
    for number, (_row_index, _cells, ids) in enumerate(read_filled(rows, kind, slot_start), start=1):
        picked = [by_dk[i] for i in ids if i in by_dk]
        unknown |= {i for i in ids if i and i not in by_dk}
        if picked:
            entered[number] = pd.DataFrame(picked).reset_index(drop=True)
    return entered, sorted(unknown), kind


def score_entered(entered, actuals):
    """Per-lineup actual score for lineups read back out of an upload file."""
    rows = []
    for number, frame in sorted(entered.items()):
        total, scored = score_lineup(frame, actuals)
        rows.append({
            "Lineup": number,
            "N": len(frame),
            "Scored": scored,
            "Salary": int(pd.to_numeric(frame.get("Salary"), errors="coerce").sum() or 0),
            "Proj": round(float(pd.to_numeric(frame["Proj"], errors="coerce").sum()), 1),
            "Ceiling": round(float(pd.to_numeric(frame["Ceiling"], errors="coerce").sum()), 1),
            "Actual": total,
            "Diff": round(total - float(pd.to_numeric(frame["Proj"], errors="coerce").sum()), 1),
        })
    return pd.DataFrame(rows)


def parse_criteria(text):
    """'objective=ceiling,max_overlap=4,randomness=0.2' -> optimizer kwargs.

    Lets a night be re-run under settings that were never shipped as a DEFAULT_CONFIG,
    which is the point: the useful question after a bad night is usually "what would this
    other setting have done", not "how did the eight presets rank".
    """
    from .optimizer import OBJECTIVES

    numeric = {"max_overlap", "min_proj", "max_bust", "randomness", "max_hitters_per_team",
               "min_games", "conflict_penalty", "conflict_min_hitters", "seed"}
    integral = {"max_overlap", "max_hitters_per_team", "min_games", "conflict_min_hitters", "seed"}
    flags = {"stack_bonus"}

    kwargs = {}
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"bad --criteria term '{part}' (expected name=value)")
        name, _, value = part.partition("=")
        name, value = name.strip(), value.strip()
        if name == "objective":
            if value not in OBJECTIVES:
                raise ValueError(f"objective must be one of {sorted(OBJECTIVES)}")
            kwargs[name] = value
        elif name in flags:
            kwargs[name] = value.lower() in {"1", "true", "yes", "y"}
        elif name in numeric:
            kwargs[name] = int(float(value)) if name in integral else float(value)
        else:
            raise ValueError(f"unknown --criteria setting '{name}'")
    return kwargs


def config_sweep(players, actuals, configs, n_lineups=20, seed=7):
    """Run the optimizer under each config and score the resulting set on actual results.

    Reports best/mean/worst lineup so a config that produced one lucky lineup is not
    confused with one that produced a reliably good set.
    """
    rows = []
    for label, kwargs in configs:
        try:
            lineups, _, _ = optimize(players, n_lineups=n_lineups, seed=seed, **kwargs)
        except Exception as error:
            rows.append({"Config": label, "Error": str(error)[:60]})
            continue
        if not lineups:
            rows.append({"Config": label, "Error": "no feasible lineup"})
            continue
        scores = [score_lineup(l["players"], actuals)[0] for l in lineups]
        rows.append({
            "Config": label,
            "N": len(scores),
            "Best": round(max(scores), 1),
            "Mean": round(float(np.mean(scores)), 1),
            "Worst": round(min(scores), 1),
            "Proj": round(float(np.mean([l["proj"] for l in lineups])), 1),
        })
    return pd.DataFrame(rows)


# Every config states its settings explicitly so the comparison stays readable when the
# shipped defaults change. The first row IS the current default.
DEFAULT_CONFIGS = [
    ("DEFAULT: ceiling, overlap 6, rand .20",
     dict(objective="ceiling", max_overlap=DEFAULT_MAX_OVERLAP, randomness=DEFAULT_RANDOMNESS)),
    ("ceiling, overlap 6, no randomness",
     dict(objective="ceiling", max_overlap=6, randomness=0.0)),
    ("ceiling, overlap 9 (DK minimum), no rand",
     dict(objective="ceiling", max_overlap=9, randomness=0.0)),
    ("ceiling, overlap 4, rand .20",
     dict(objective="ceiling", max_overlap=4, randomness=0.20)),
    ("ceiling, rand .40", dict(objective="ceiling", max_overlap=6, randomness=0.40)),
    ("ceiling, no stack bonus",
     dict(objective="ceiling", max_overlap=6, randomness=0.20, stack_bonus=False)),
    ("proj", dict(objective="proj", max_overlap=6, randomness=0.20)),
    ("floor (cash)", dict(objective="floor", max_overlap=6, randomness=0.20)),
]


def main():
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(
        description="Score a night's projections, tiers and lineups against actual results.")
    parser.add_argument("--date", default=datetime.today().strftime("%Y-%m-%d"))
    parser.add_argument("--salaries")
    parser.add_argument("--slate")
    parser.add_argument("--entered", nargs="?", const="auto", metavar="PATH",
                        help="Score the lineups actually entered. Bare flag finds the "
                             "night's upload/swap file; pass a path for a specific one.")
    parser.add_argument("--criteria", metavar="SETTINGS",
                        help="Re-run the optimizer under ad-hoc settings and score the "
                             "result, e.g. 'objective=ceiling,max_overlap=4,randomness=0.2'.")
    parser.add_argument("--criteria-lineups", type=int, default=20)
    parser.add_argument("--sweep", action="store_true",
                        help="Also test the shipped optimizer configurations (slow).")
    parser.add_argument("--sweep-lineups", type=int, default=10)
    args = parser.parse_args()

    players, _, meta = build_slate(args.date, salary_path=args.salaries, slate=args.slate)
    if players.empty:
        print(f"No slate data for {args.date}.")
        return
    actuals = slate_actuals(args.date)
    if not actuals:
        print(f"No finished games for {args.date} yet.")
        return

    # Everything below the team table needs prices: the optimizer's pool is defined by
    # Salary, so an unpriced slate reaches it as an empty pool and fails with a message
    # about roster size that says nothing about the real cause. Name the cause here, and
    # still print the scoring sections that do not depend on salaries.
    if not meta.get("has_salary"):
        candidates = meta.get("ambiguous_slates") or []
        if candidates:
            print(f"NOTE: several DK exports match {args.date}, so none was applied. "
                  f"Re-run with --slate naming one of:")
            for candidate in candidates:
                print(f"        --slate {candidate['name']}   "
                      f"({candidate.get('teams')} teams, {candidate.get('players')} players)")
        else:
            print(f"NOTE: no DK salary file found for {args.date}; pass --salaries to point at one.")
        print("   Tier and team scoring still follow; the buildable-best lineup needs prices.\n")

    pd.set_option("display.width", 200)
    tiers, roles = tier_report(players, actuals)
    print(f"=== {args.date}: how the flags did ===")
    if not tiers.empty:
        print(tiers.to_string(index=False))
    if not roles.empty:
        print()
        print(roles.to_string(index=False))

    print("\n=== team hitter points: projected vs actual ===")
    print(team_report(players, actuals).to_string())

    if not meta.get("has_salary"):
        return

    best = perfect_lineup(players, actuals)
    best_total = None
    if best:
        frame = best["players"].copy()
        frame["Actual"] = [actuals[int(m)]["Actual"] for m in frame["MLBAM"]]
        best_total = float(frame["Actual"].sum())
        print(f"\n=== best lineup that was buildable: {best_total:.1f} pts, "
              f"${best['salary']:,} ===")
        print(frame[["Roster", "Name", "Team", "Salary", "Proj", "Actual"]].to_string(index=False))

    if args.entered:
        report_entered(args, players, actuals, best_total)

    if args.criteria:
        report_criteria(args, players, actuals, best_total)

    if args.sweep:
        print("\n=== shipped optimizer settings vs actual results ===")
        print(config_sweep(players, actuals, DEFAULT_CONFIGS,
                           n_lineups=args.sweep_lineups).to_string(index=False))


def _pct_of_best(value, best_total):
    return f"  ({value / best_total * 100:.0f}% of the night's ceiling)" if best_total else ""


def report_entered(args, players, actuals, best_total):
    """How the lineups actually uploaded to DK did."""
    from .lateswap import SwapError, find_filled

    path = args.entered
    if path == "auto":
        try:
            path = find_filled(args.date, slate=args.slate)
        except SwapError as error:
            print(f"\n[!] no entered lineups to score: {error}")
            return
    try:
        entered, unknown, kind = read_entered(path, players)
    except Exception as error:
        print(f"\n[!] could not read entered lineups from {path}: {error}")
        return
    if not entered:
        print(f"\n[!] {path} has no filled lineups to score.")
        return

    scored = score_entered(entered, actuals)
    print(f"\n=== lineups you entered: {os.path.basename(path)} ({kind}) ===")
    print(scored.to_string(index=False))

    totals = scored["Actual"]
    print(f"\n  {len(scored)} lineup(s): best {totals.max():.1f}, mean {totals.mean():.1f}, "
          f"worst {totals.min():.1f}{_pct_of_best(totals.max(), best_total)}")
    # The gap between what you entered and what was buildable is the only number here that
    # says whether the night was lost on selection rather than on variance.
    if best_total:
        print(f"  best buildable was {best_total:.1f}; your best left "
              f"{best_total - totals.max():.1f} on the table.")
    if unknown:
        print(f"  [!] {len(unknown)} id(s) in the file are not on this slate: "
              f"{', '.join(unknown[:8])}{' ...' if len(unknown) > 8 else ''}")
        print("      Usually the wrong --slate, or a template from another contest.")


def report_criteria(args, players, actuals, best_total):
    """How the optimizer would have done under settings passed on the command line."""
    try:
        kwargs = parse_criteria(args.criteria)
    except ValueError as error:
        print(f"\n[!] {error}")
        return
    settings = ", ".join(f"{k}={v}" for k, v in sorted(kwargs.items())) or "defaults"
    print(f"\n=== optimizer under: {settings} ===")
    table = config_sweep(players, actuals, [(settings, kwargs)],
                         n_lineups=args.criteria_lineups)
    print(table.to_string(index=False))
    if "Best" in table.columns and not table["Best"].isna().all() and best_total:
        print(f"\n  best buildable was {best_total:.1f}"
              f"{_pct_of_best(float(table['Best'].iloc[0]), best_total)}")


if __name__ == "__main__":
    main()
