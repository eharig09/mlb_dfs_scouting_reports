"""Score a night's lineups against what actually happened.

Answers three questions: how did the lineups we built do, what was the best lineup that
was buildable at all, and which optimizer settings would have produced the best set.
"""

import os
import pickle

import numpy as np
import pandas as pd

from .backtest import actual_points
from .exposure import player_exposure
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
    from .upload import SLOT_ORDER, read_template

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
        # The ids arrive in DK's slot order, so the slot each player was entered *at* is
        # known and worth keeping: without it a usage report falls back to DK eligibility
        # and reports a 3B/OF as "3B/OF" rather than the outfield spot he actually filled.
        # Zipped rather than filtered separately, so an unresolvable id cannot shift every
        # slot after it onto the wrong player.
        picked, slots = [], []
        for slot, dk_id in zip(SLOT_ORDER, ids):
            if dk_id in by_dk:
                picked.append(by_dk[dk_id])
                slots.append(slot)
        unknown |= {i for i in ids if i and i not in by_dk}
        if picked:
            frame = pd.DataFrame(picked).reset_index(drop=True)
            frame["Roster"] = slots
            entered[number] = frame
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
            # Cumulative ownership, the same measure reported for the buildable best, so the
            # two are directly comparable: a lineup that scored well on 150% cumulative
            # ownership won a different night than one that scored well on 60%.
            # NaN, not 0, when the slate carries no ownership -- a column of zeroes would
            # read as "nobody rostered these" rather than "we do not know".
            "Own": (round(float(_own_series(frame).sum()), 0)
                    if _own_series(frame).notna().any() else float("nan")),
            "Actual": total,
            "Diff": round(total - float(pd.to_numeric(frame["Proj"], errors="coerce").sum()), 1),
        })
    return pd.DataFrame(rows)


def player_usage(entered, actuals):
    """Your most-rostered players: exposure against field ownership, and what they scored.

    The per-lineup table says how the lineups did; this says *who* did it. Exposure only
    means something next to ownership -- being 50% on a 2%-owned player is the whole bet a
    tournament pays for, and being 50% on a 40%-owned one is just the field with extra steps
    -- so Exp%, Own% and the gap between them sit beside the actual result.

    Reuses the optimizer's own exposure counter so "how often did I roster him" is computed
    one way in this project, not two that can disagree.
    """
    lineups = [{"players": frame} for frame in entered.values()]
    table = player_exposure(lineups)
    if table.empty:
        return table

    # Results are per player, not per lineup, so one pass over the slate rows is enough.
    scored = {}
    for frame in entered.values():
        for _, row in frame.iterrows():
            pid = row.get("MLBAM")
            if pid is None or pd.isna(pid):
                continue
            result = actuals.get(int(pid))
            if result is not None:
                scored[row["Name"]] = result["Actual"]

    table["Actual"] = [scored.get(name) for name in table["Name"]]
    table["Diff"] = (pd.to_numeric(table["Actual"], errors="coerce")
                     - pd.to_numeric(table["Proj"], errors="coerce")).round(1)
    return table


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


def load_slate_for_review(args):
    """(players, meta, one-line description of where the numbers came from).

    A snapshot wins whenever one exists, because the payload cache is mutated in place by
    `--refresh-lineups`: rebuilding a past night re-reads whatever those payloads say *now*,
    so confirmed lineups and late scratches leak backwards into the projections being
    graded. The review then flatters itself with information it did not have.

    When no snapshot exists the rebuild still happens -- there is nothing else to do, and a
    rebuilt review is far better than none -- but the source line says so, and the payload
    digests are checked so a *stale* snapshot is called out too.
    """
    from .snapshot import SnapshotError, load_snapshot

    if not args.no_snapshot:
        try:
            snap = load_snapshot(args.date, args.slate, stage=args.stage)
        except SnapshotError:
            snap = None
        if snap is not None:
            note = (f"scoring the {snap.stage} snapshot taken "
                    f"{snap.manifest.get('taken_utc')} "
                    f"(model v{snap.manifest.get('versions', {}).get('model')}, "
                    f"code {snap.manifest.get('versions', {}).get('code')})")
            stale = snap.stale_payloads()
            if stale:
                note += (f"\n  [i] {len(stale)} game payload(s) have been rewritten since; "
                         f"the snapshot is the only honest copy left.")
            return snap.players, snap.meta(), note

    players, _, meta = build_slate(args.date, salary_path=args.salaries, slate=args.slate)
    note = ("[!] no snapshot for this night, so the slate was rebuilt from the payload "
            "cache.\n    Those payloads are rewritten by --refresh-lineups, so any "
            "confirmed lineup or\n    scratch that arrived after lock is baked into the "
            "projections being graded.\n    Take a snapshot before lock next time: "
            f"python -m dfs.snapshot --date {args.date} --stage final")
    if args.no_snapshot:
        note = "rebuilt from the payload cache (--no-snapshot)"
    return players, meta, note


def candidate_report(directory, actuals, best_total=None):
    """Grade a candidate pool against the night: was the best lineup even reachable?

    This is the diagnostic that separates two failures a points total cannot tell apart. If
    the pool's best lineup scored near the night's ceiling, generation was fine and the loss
    was in *selection* or in variance. If it did not, no selection rule could have saved the
    night -- the winning construction was never on the table.
    """
    from .candidates import CandidatePool

    pool = CandidatePool.load(directory)
    frame = pool.pool
    scored = frame["MLBAM"].map(
        lambda pid: None if pd.isna(pid) else (actuals.get(int(pid)) or {}).get("Actual"))
    points = pd.to_numeric(scored, errors="coerce")

    totals, coverage = [], []
    for players in pool.lineups["players"]:
        values = points.iloc[list(players)]
        totals.append(float(values.fillna(0).sum()))
        coverage.append(int(values.notna().sum()))
    result = pool.lineups.copy()
    result["Actual"] = np.round(totals, 2)
    result["Scored"] = coverage

    lines = [f"\n=== candidate pool: {os.path.basename(directory)} ===",
             f"{len(result)} candidates, {result['stack_shape'].nunique()} stack shapes, "
             f"{result['primary_stack'].nunique()} primary teams, "
             f"{result['pitcher_pair'].nunique()} pitcher pairs"]

    best = result.nlargest(1, "Actual").iloc[0]
    pool_best = float(best["Actual"])
    lines.append(f"best candidate scored {pool_best:.1f} "
                 f"({best['stack_shape']} {best['primary_stack']}, "
                 f"proj {best['proj']:.1f}, own {best['own_sum']})")
    lines.append(f"pool mean {result['Actual'].mean():.1f}, "
                 f"median {result['Actual'].median():.1f}, "
                 f"worst {result['Actual'].min():.1f}")

    # What each ranking rule would have picked, scored on the night. The spread between
    # these and the pool's best is the part selection could have won.
    lines.append("")
    lines.append("  if you had entered one lineup, chosen by:")
    picks = {}
    for column, label, biggest in (("proj", "highest projection", True),
                                   ("ceiling", "highest ceiling", True),
                                   ("own_sum", "lowest total ownership", False),
                                   ("sim_p99", "highest simulated p99", True)):
        if column not in result.columns or result[column].isna().all():
            continue
        pick = (result.nlargest(1, column) if biggest else result.nsmallest(1, column))
        picks[label] = float(pick.iloc[0]["Actual"])
        lines.append(f"    {label:<28} {picks[label]:6.1f}")
    lines.append(f"    {'random candidate (mean)':<28} {result['Actual'].mean():6.1f}")
    lines.append(f"    {'ex-post best in the pool':<28} {pool_best:6.1f}")

    if best_total:
        # Decomposed rather than judged. The best *buildable* lineup is pure hindsight over
        # every legal combination on the slate -- it is built from players who blew up, not
        # from players who projected well, so no projection-driven pool will contain it and
        # "% of the ceiling" is not a pass mark. What the two gaps below separate is which
        # stage a night was actually lost at.
        selection_gap = pool_best - max(picks.values()) if picks else 0.0
        generation_gap = best_total - pool_best
        lines.append("")
        lines.append(f"  night's ceiling (hindsight, any legal lineup) {best_total:7.1f}")
        lines.append(f"  best the pool could have given you            {pool_best:7.1f}"
                     f"   <- generation headroom {generation_gap:.1f}")
        if picks:
            lines.append(f"  best any ranking rule actually picked         "
                         f"{max(picks.values()):7.1f}   <- selection headroom "
                         f"{selection_gap:.1f}")
            lines.append("")
            lines.append("  Generation headroom is mostly irreducible: the ex-post best "
                         "lineup is assembled\n  from whoever happened to blow up, and no "
                         "projection can pre-select them. Selection\n  headroom is the part "
                         "that was genuinely available on the night.")
    return "\n".join(lines)


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
    parser.add_argument("--usage-top", type=int, default=20, metavar="N",
                        help="How many players the entered-lineup usage table shows "
                             "(default 20; 0 for all). Needs --entered.")
    parser.add_argument("--criteria", metavar="SETTINGS",
                        help="Re-run the optimizer under ad-hoc settings and score the "
                             "result, e.g. 'objective=ceiling,max_overlap=4,randomness=0.2'.")
    parser.add_argument("--criteria-lineups", type=int, default=20)
    parser.add_argument("--sweep", action="store_true",
                        help="Also test the shipped optimizer configurations (slow).")
    parser.add_argument("--sweep-lineups", type=int, default=10)
    parser.add_argument("--no-snapshot", action="store_true",
                        help="Rebuild the slate from the payload cache instead of reading "
                             "the night's snapshot. Those payloads are rewritten by "
                             "--refresh-lineups, so a rebuild scores information that was "
                             "not available at lock.")
    parser.add_argument("--stage", help="Which snapshot stage to score against.")
    parser.add_argument("--candidates", metavar="DIR",
                        help="Also grade a saved candidate pool: was the night's best "
                             "lineup reachable from it at all?")
    args = parser.parse_args()

    players, meta, source = load_slate_for_review(args)
    if players.empty:
        print(f"No slate data for {args.date}.")
        return
    print(source)
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
        columns = [c for c in ("Roster", "Name", "Team", "Salary", "Proj", "Own%", "Actual")
                   if c in frame.columns]
        print(frame[columns].to_string(index=False))
        print(ownership_note(frame))

    if args.candidates:
        try:
            print(candidate_report(args.candidates, actuals, best_total))
        except Exception as error:
            print(f"\n[!] could not grade the candidate pool at {args.candidates}: {error}")

    if args.entered:
        report_entered(args, players, actuals, best_total)

    if args.criteria:
        report_criteria(args, players, actuals, best_total)

    if args.sweep:
        print("\n=== shipped optimizer settings vs actual results ===")
        print(config_sweep(players, actuals, DEFAULT_CONFIGS,
                           n_lineups=args.sweep_lineups).to_string(index=False))


# A player under this was effectively unowned by the field.
CONTRARIAN_OWN = 5.0


def _own_series(frame):
    """Own% as a numeric Series, empty when the slate has no ownership at all.

    `frame.get("Own%")` returns None for a missing column and pd.to_numeric then hands back
    a bare numpy nan, which has no .isna() and silently sums to nan -- so an unpriced or
    pre-ownership slate would crash the review rather than skip the section.
    """
    if "Own%" not in getattr(frame, "columns", []):
        return pd.Series(dtype=float)
    return pd.to_numeric(frame["Own%"], errors="coerce")


def ownership_note(frame, label="this lineup"):
    """How owned the field was on a lineup's ten players.

    The number a review is missing without it: whether the night's ceiling was reachable by
    playing chalk or only by being contrarian. Those are opposite lessons -- one says the
    projections were fine and the field got there too, the other says the points were
    sitting somewhere nobody was looking -- and the points total alone cannot tell them
    apart. Cumulative ownership is the GPP convention, so it is reported alongside the mean.
    """
    own = _own_series(frame)
    if own.empty or own.isna().all():
        return "  (no ownership data on this slate — run the board so Own% is estimated)"

    # 'est' is the model's own guess; anything else came from a real contest export. The
    # split is printed per player rather than as one blurred label, because a review run
    # against estimated ownership is measuring the model against itself -- and on a mixed
    # lineup you need to know how much of it is real before drawing a lesson from it.
    counts = frame.get("Own Src", pd.Series(dtype=object)).fillna("").replace("", "est")
    counts = counts.value_counts()
    source = ", ".join(f"{name} {count}/{len(frame)}"
                       for name, count in counts.items()) or "source unknown"
    quiet = int((own < CONTRARIAN_OWN).sum())
    return (f"  field ownership [{source}] on {label}: mean {own.mean():.1f}%, "
            f"cumulative {own.sum():.0f}%, {quiet} of {own.notna().sum()} under "
            f"{CONTRARIAN_OWN:.0f}%")


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
    if "Own" in scored.columns and scored["Own"].notna().any():
        print(f"  cumulative ownership: mean {scored['Own'].mean():.0f}% per lineup, "
              f"range {scored['Own'].min():.0f}-{scored['Own'].max():.0f}%")

    usage = player_usage(entered, actuals)
    if not usage.empty:
        top = args.usage_top
        columns = [c for c in ("Name", "Team", "Roster", "Salary", "Lineups", "Exp%",
                               "Own%", "Lev", "Proj", "Actual", "Diff")
                   if c in usage.columns]
        print(f"\n=== who you were actually on ({len(usage)} players across "
              f"{len(scored)} lineups) ===")
        print(usage.head(top)[columns].to_string(index=False))
        if top and len(usage) > top:
            print(f"  ... {len(usage) - top} more (--usage-top 0 for all)")
        # The line that turns the table into a decision: heavy exposure the field did not
        # share is the bet that was made, whether or not it came in.
        own = pd.to_numeric(usage["Own%"], errors="coerce")
        lev = pd.to_numeric(usage["Lev"], errors="coerce")
        heavy = usage[(pd.to_numeric(usage["Exp%"], errors="coerce") >= 25) & (lev >= 10)]
        if not heavy.empty and own.notna().any():
            hit = heavy[pd.to_numeric(heavy["Diff"], errors="coerce") > 0]
            print(f"\n  {len(heavy)} leveraged bet(s) at 25%+ exposure and 10+ over the "
                  f"field; {len(hit)} beat projection.")
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
