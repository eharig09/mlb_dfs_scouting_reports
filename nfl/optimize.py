"""Generate NFL lineups from the command line.

    python -m nfl.optimize --salaries "path/to/DKSalaries.csv" --n 20

`nfl.optimizer` is the solver; this is the operational wrapper around it -- the football
twin of `dfs.optimize`. It finds the slate, attaches projections, honours the editable pool,
runs the solve, and writes `lineups_<slate>.csv` and `exposure_<slate>.csv` into the dated
folder `nfl.naming` decides.

The recommended loop, mirroring the MLB one:

    # first use only: create the editable pool
    python -m nfl.optimize --salaries DKSalaries.csv --write-pool

    # after editing Lock / Exclude / Boost / Min% / Max% in that csv
    python -m nfl.optimize --salaries DKSalaries.csv --n 150 --pool \
        --objective ceiling --stack-shape "3-1,3" --overwrite

## Stack shapes

`--stack-shape "3-1"` means **three pass-game players from one team plus one bring-back**
from the other side of that same game. The quarterback counts as one of the three and is
pinned, so `3-1` is QB + two of his receivers + one opponent receiver.

It is expressed entirely in `nfl.optimizer`'s existing primitives -- a pin for the QB and
`stacks={team: n}` for the counts -- rather than by adding a constraint. The solver already
knows that a stack counts QB/WR/TE and never a running back.

**Rank within the passing game is deliberately not part of the shape.** Measured over
2023-25, a team's WR2 correlates with its quarterback as strongly as its WR1 (+0.359 against
+0.367, and higher on the median) while costing about a quarter less target share. A shape
that insisted on the WR1 would be paying for a correlation already available one slot down.
See `docs/nfl/pff_integration.md`.
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from nfl import naming, pool as pool_module
from nfl import optimizer as opt
from nfl.projections import BAND_FIT
from nfl.salaries import load_dk_export, normalize_name

# Bands for a defense. `BAND_FIT` is fitted on skill positions only, and letting DST fall
# through to the WR band would claim a fit that was never measured on a defense. These are
# deliberately wide and **explicitly unfitted** -- flagged here rather than buried, in the
# same spirit as the optimizer's carried-over overlap defaults.
DST_BAND = {"floor": (0.30, -1.0), "ceiling": (2.00, 3.0)}

DEFAULT_LINEUPS = 20
# How many teams to consider as the anchor of a stack shape. Lineups rotate through them,
# which is what keeps a 150-lineup set from being 150 builds of the same offense.
DEFAULT_STACK_TEAMS = 6


def _split(value):
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def parse_shape(spec):
    """The football stack grammar. `'3-1+2,4-1,3'` -> [(3, 1, 2), (4, 1, 0), (3, 0, 0)].

        A        pass-game players from the anchor club, **quarterback included**
        A-B      ...plus B from the other side of that same game (the bring-back)
        A-B+C    ...plus a C-man secondary stack from a *different* game

    Three numbers because football stacks are built in three decisions, and they are not
    interchangeable. `A` is the correlation you are buying. `B` is the hedge that pays when
    the game is a shootout rather than a blowout. `C` is diversification -- a second, smaller
    correlated block so a single game going quiet does not take the whole lineup with it.

    The quarterback counts inside `A` and is pinned, so `3-1` is QB + two of his receivers
    + one opponent receiver: the shape most NFL fields are built around.
    """
    shapes = []
    for token in _split(spec):
        head, _, secondary = token.partition("+")
        primary, _, bringback = head.partition("-")

        def _number(text, what):
            try:
                return int(text)
            except ValueError:
                raise SystemExit(f"bad --stack-shape '{token}': {what} '{text}' is not a "
                                 f"number. Shapes look like 3, 3-1 or 3-1+2.")

        count = _number(primary, "anchor size")
        if count < 1:
            raise SystemExit(f"bad --stack-shape '{token}': need at least one player")
        back = _number(bringback, "bring-back") if bringback else 0
        extra = _number(secondary, "secondary stack") if secondary else 0
        if extra == 1:
            raise SystemExit(f"bad --stack-shape '{token}': a one-man secondary stack is "
                             f"not a stack. Use 2 or more, or drop the +.")
        if count + back + extra > opt.ROSTER_SIZE - 1:
            raise SystemExit(f"bad --stack-shape '{token}': {count + back + extra} stacked "
                             f"players will not fit alongside a DST and the RB slots.")
        shapes.append((count, back, extra))
    return shapes


def parse_exposure(spec, n_lineups):
    """`'BUF:20-60%,KAN:0-30%'` -> {team: (min_lineups, max_lineups)}.

    Percentages are of the lineup set and are converted to counts here, so the solver only
    ever deals in whole lineups -- a fraction of a lineup is not a thing it can build.
    """
    out = {}
    for token in _split(spec):
        team, _, span = token.partition(":")
        if not span:
            raise SystemExit(f"bad exposure '{token}': expected TEAM:MIN-MAX")
        low_text, _, high_text = span.partition("-")

        def _share(text):
            text = text.strip().rstrip("%")
            if not text:
                return None
            try:
                value = float(text)
            except ValueError:
                raise SystemExit(f"bad exposure '{token}': '{text}' is not a number")
            fraction = value / 100.0 if value > 1 else value
            return int(round(fraction * n_lineups))

        low, high = _share(low_text), _share(high_text or low_text)
        out[team.strip().upper()] = (low or 0, n_lineups if high is None else high)
    return out


def parse_stacks(spec):
    """'BUF:3,KAN:2' -> {'BUF': 3, 'KAN': 2}."""
    out = {}
    for token in _split(spec):
        team, _, count = token.partition(":")
        if not count:
            raise SystemExit(f"bad --stack '{token}': expected TEAM:COUNT")
        try:
            out[team.strip().upper()] = int(count)
        except ValueError:
            raise SystemExit(f"bad --stack '{token}': '{count}' is not a number")
    return out


def attach_bands(players):
    """Fill `Ceiling` and `Floor` from the fitted per-position bands where absent.

    Only fills; a projection source that supplies its own bands keeps them. The fit is
    `nfl.projections.BAND_FIT`, measured against realised outcomes, so the ceiling a lineup
    is optimised against is the same one the projection model stands behind.
    """
    frame = players.copy()
    proj = pd.to_numeric(frame.reindex(columns=["Proj"])["Proj"], errors="coerce")
    positions = frame.reindex(columns=["Pos"])["Pos"].astype("string").fillna("")

    for column in ("Floor", "Ceiling"):
        if column not in frame.columns:
            frame[column] = np.nan
        current = pd.to_numeric(frame[column], errors="coerce")
        filled = []
        for value, position, points in zip(current, positions, proj):
            if pd.notna(value):
                filled.append(float(value))
                continue
            if pd.isna(points):
                filled.append(np.nan)
                continue
            band = DST_BAND if position == "DST" else BAND_FIT.get(position, BAND_FIT["WR"])
            slope, intercept = band[column.lower()]
            estimate = slope * float(points) + intercept
            filled.append(round(max(0.0, estimate) if column == "Floor" else round(estimate, 2), 2))
        frame[column] = filled
    return frame


def _projection_index(path, name_column="Name", points_column="Proj", games_column=None):
    """{normalised name: {Proj, Ceiling, Floor}} from a projections csv.

    `games_column` divides a season total into a per-game number, which is what the PFF
    export ships and what a one-week slate needs -- a season total silently rewards whoever
    is expected to play seventeen games.
    """
    frame = pd.read_csv(path)
    frame.columns = [str(c).strip() for c in frame.columns]
    if name_column not in frame.columns or points_column not in frame.columns:
        raise SystemExit(f"{path} needs at least {name_column} and {points_column} columns")
    points = pd.to_numeric(frame[points_column], errors="coerce")
    if games_column and games_column in frame.columns:
        games = pd.to_numeric(frame[games_column], errors="coerce").replace(0, np.nan)
        points = points / games

    index = {}
    for position, (_, row) in enumerate(frame.iterrows()):
        key = normalize_name(row.get(name_column))
        if not key:
            continue
        record = {"Proj": points.iloc[position]}
        for column in ("Ceiling", "Floor"):
            if column in frame.columns:
                record[column] = pd.to_numeric(row.get(column), errors="coerce")
        index[key] = record
    return index


def build_board(salaries, projections=None, drop_out=True):
    """DK export + projections -> the frame `nfl.optimizer.optimize` takes.

    Returns `(players, report)`. Players with no projection keep a blank `Proj` rather than
    a zero, and the report counts them: on a real slate a third of the export is fourth
    stringers, and calling that "projected to score nothing" is a different claim from
    "not projected".
    """
    players, report = load_dk_export(salaries, drop_out=drop_out)

    if projections:
        index = _projection_index(projections)
        keys = players["Name"].map(normalize_name)
        for column in ("Proj", "Ceiling", "Floor"):
            values = [index.get(key, {}).get(column, np.nan) for key in keys]
            if any(pd.notna(v) for v in values):
                players[column] = values
    if "Proj" not in players.columns:
        # **Fall back to the PFF projection export before falling back to DK's average.**
        # The dashboard prices every board off that file, and an optimizer quietly using a
        # different number than the pages you decided from is the worst kind of
        # disagreement: both look right and neither is wrong on its own.
        try:
            from nfl import pffdata

            index = _projection_index(pffdata.latest("projections"), name_column="playerName",
                                      points_column="fantasyPoints", games_column="games")
            keys = players["Name"].map(normalize_name)
            values = [index.get(key, {}).get("Proj", np.nan) for key in keys]
            if any(pd.notna(v) for v in values):
                players["Proj"] = values
                report["projection_source"] = "PFF projections (per game)"
        except Exception:
            pass
    if "Proj" not in players.columns:
        # No projection source at all. DK publishes a season average per player, which is a
        # real number and a poor projection -- usable to smoke-test the pipeline, never to
        # submit against. It is labelled so nothing downstream mistakes it for a model.
        players["Proj"] = players["AvgPts"]
        report["projection_source"] = "DK AvgPointsPerGame (NOT a model)"
    report.setdefault("projection_source", projections or "supplied")

    players = attach_bands(players)
    report["unprojected"] = int(pd.to_numeric(players["Proj"], errors="coerce").isna().sum())
    return players, report


def _stack_candidates(players, limit):
    """Teams worth anchoring a stack on, best first, by summed pass-game projection.

    `limit=None` returns every club with a rosterable pass game, which is what an explicit
    exposure floor is allowed to reach into.
    """
    frame = players.copy()
    frame["_proj"] = pd.to_numeric(frame.reindex(columns=["Proj"])["Proj"],
                                   errors="coerce").fillna(0.0)
    pass_game = frame[frame["Pos"].isin(opt.PASS_GAME_POSITIONS)]
    if pass_game.empty:
        return []
    ranked = pass_game.groupby("Team")["_proj"].sum().sort_values(ascending=False)
    return list(ranked.index if limit is None else ranked.index[:limit])


def _team_quarterback(players, team):
    """The highest-projected QB on a team, for pinning as a stack anchor."""
    frame = players[(players["Team"] == team) & (players["Pos"] == "QB")].copy()
    if frame.empty:
        return None
    frame["_proj"] = pd.to_numeric(frame["Proj"], errors="coerce").fillna(0.0)
    return frame.sort_values("_proj", ascending=False).iloc[0]["Name"]


def _opponent_of(players, team):
    opponents = players.loc[players["Team"] == team, "Opp"].dropna().unique()
    return opponents[0] if len(opponents) else None


def generate(players, n_lineups, objective="ceiling", shapes=None, stacks=None,
             locks=None, excludes=None, boosts=None, exposure=None, team_exposure=None,
             stack_at=opt.STACK_EXPOSURE_AT, stack_teams=DEFAULT_STACK_TEAMS,
             max_overlap=None, randomness=None, min_proj=None, max_ownership=None,
             seed=None, report=print):
    """Run the solve, rotating stack shapes and anchor teams across the set.

    With `shapes`, each lineup is solved on its own: a shape and an anchor team are chosen,
    the anchor's quarterback is pinned, and the counts go in as `stacks`. Rotating the
    anchor is what stops a large set from being one offense rebuilt N times.
    """
    # An unset flag means "whatever the optimizer's default is", not "None". Passing None
    # through *overrides* the default rather than deferring to it, and `randomness` is then
    # compared against a number -- so the flag nobody set is the one that crashes the run.
    if randomness is None:
        randomness = opt.DEFAULT_RANDOMNESS
    if max_overlap is None:
        max_overlap = opt.DEFAULT_MAX_OVERLAP

    rng = np.random.default_rng(seed)
    if not shapes:
        return opt.optimize(players, n_lineups=n_lineups, objective=objective,
                            locks=locks, excludes=excludes, stacks=stacks,
                            max_overlap=max_overlap, randomness=randomness,
                            min_proj=min_proj, max_ownership=max_ownership,
                            exposure=exposure, team_exposure=team_exposure,
                            stack_at=stack_at, seed=seed)

    candidates = _stack_candidates(players, stack_teams)
    if not candidates:
        raise SystemExit("no pass-game players on this slate to anchor a stack on")
    # Every club with a quarterback and a pass game — the set an exposure floor may reach
    # into, as opposed to `candidates`, which is the shortlist the rotation walks.
    stackable = _stack_candidates(players, None)

    built, pool_frame, missing = [], None, {"locks": [], "excludes": [], "pins": []}
    # **Exposure is satisfied over the whole set, and this loop solves one lineup at a
    # time.** So the running counts are threaded through every call: without them each solve
    # believes it is the entire set, every minimum looks due immediately, and the first
    # lineup is forced to carry the whole exposure list at once.
    appearances, team_stacks = {}, {}
    attempts = 0
    while len(built) < n_lineups and attempts < n_lineups * 4:
        primary, bringback, secondary = shapes[len(built) % len(shapes)]
        # **A club behind its exposure minimum becomes the anchor.** The shape loop dictates
        # the anchor's count itself, so the solver's own forced-stack pacing steps aside for
        # it -- which means a team minimum can only ever be met by choosing that club here.
        # Left to a plain rotation, `--team-exposure BUF:30-70%` reported "could not meet
        # team stack minimum" on a club that was simply never picked to anchor.
        # Checked against every club the slate *can* anchor, not the top-N shortlist. An
        # explicit `--team-exposure` floor is a decision you made; being outside the
        # projection shortlist is a reason the rotation would not have reached the club
        # anyway, which is exactly the situation the floor exists to override.
        team = _anchor_behind(team_exposure, team_stacks, n_lineups, len(built), stackable)
        if team is None:
            team = candidates[attempts % len(candidates)]
        attempts += 1

        quarterback = _team_quarterback(players, team)
        if quarterback is None:
            continue
        request = {team: primary}
        opponent = _opponent_of(players, team)
        if bringback:
            if opponent is None:
                continue
            request[opponent] = bringback
        if secondary:
            # A second correlated block, deliberately from **another game**: the whole point
            # is that one game going quiet does not take the lineup with it, and a secondary
            # stack inside the same game is just a bigger primary one.
            other = _secondary_team(players, candidates, exclude={team, opponent})
            if other is None:
                continue
            request[other] = secondary

        frame = pool_module.apply_boosts(players, boosts or {}, rng=rng)
        try:
            one, pool_frame, missing = opt.optimize(
                frame, n_lineups=1, objective=objective, locks=locks, excludes=excludes,
                pins={quarterback: "QB"}, stacks=request, max_overlap=max_overlap,
                randomness=randomness, min_proj=min_proj, max_ownership=max_ownership,
                exposure=exposure, team_exposure=team_exposure, stack_at=stack_at,
                prior_appearances=appearances, prior_team_stacks=team_stacks,
                total_lineups=n_lineups, prior_lineups=len(built),
                seed=int(rng.integers(0, 2 ** 31 - 1)))
        except opt.OptimizerError as error:
            report(f"[!] {team} {_shape_label(primary, bringback, secondary)}: {error}")
            continue
        if not one:
            continue
        lineup = one[0]
        lineup["shape"] = _shape_label(primary, bringback, secondary)
        lineup["anchor"] = team
        if any(_same_players(lineup, existing) for existing in built):
            continue
        built.append(lineup)
        appearances = dict(missing.get("appearances") or appearances)
        team_stacks = dict(missing.get("team_stacks") or team_stacks)
    return built, pool_frame, missing


def _anchor_behind(team_exposure, stacked, total, done, candidates):
    """The club furthest behind its stack minimum's pace, or None if all are on track.

    Paced the same way the solver paces a player minimum -- a club wanted in 30 of 50
    lineups should be stacked in roughly 3 of the first 5, not 30 of the last 30. Clubs the
    slate cannot anchor are skipped rather than starved, since `candidates` is already the
    list of teams with a rosterable pass game.
    """
    if not team_exposure:
        return None
    behind = []
    for team, (low, _high) in team_exposure.items():
        if low <= 0 or team not in candidates:
            continue
        used = int(stacked.get(team, 0))
        target = (2 * low * (done + 1) + total) // (2 * max(total, 1))
        if target - used > 0:
            behind.append(((low - used) / max(total - done, 1), team))
    if not behind:
        return None
    behind.sort(reverse=True)
    return behind[0][1]


def _shape_label(primary, bringback, secondary):
    label = str(primary)
    if bringback:
        label += f"-{bringback}"
    if secondary:
        label += f"+{secondary}"
    return label


def _secondary_team(players, candidates, exclude):
    """The best anchor for a secondary stack, from a game the primary one is not in."""
    games = players.dropna(subset=["Team", "Game"]).drop_duplicates("Team")
    blocked_games = set(games.loc[games["Team"].isin(exclude), "Game"])
    for team in candidates:
        if team in exclude:
            continue
        game = games.loc[games["Team"] == team, "Game"]
        if not game.empty and game.iloc[0] not in blocked_games:
            return team
    return None


def _same_players(a, b):
    return set(a["players"]["Name"]) == set(b["players"]["Name"])


def lineup_rows(lineups):
    """One row per player per lineup -- the `lineups_<slate>.csv` layout."""
    rows = []
    for number, lineup in enumerate(lineups, start=1):
        for _, player in lineup["players"].iterrows():
            rows.append({
                "Lineup": number,
                # `_assemble` names this column `Roster`, not `Slot`. Reading the wrong one
                # produced a blank column for every player rather than an error -- and it is
                # the column the upload writer needs, since it is what decides which DK
                # roster cell a player is written into.
                "Slot": player.get("Roster", ""),
                "Name": player.get("Name"),
                "Pos": player.get("Pos"),
                "Team": player.get("Team"),
                "Opp": player.get("Opp"),
                "Game": player.get("Game"),
                "Salary": player.get("Salary"),
                "Proj": player.get("Proj"),
                "Ceiling": player.get("Ceiling"),
                "DK ID": player.get("DK ID"),
                "Lineup Salary": lineup["salary"],
                "Lineup Proj": round(float(lineup["proj"]), 2),
                "Lineup Ceiling": round(float(lineup["ceiling"]), 2),
                "Shape": lineup.get("shape", ""),
                "Anchor": lineup.get("anchor", ""),
            })
    return pd.DataFrame(rows)


def exposure_rows(lineups):
    """Per-player and per-team exposure across the set.

    The MLB twin caught a 54% exposure on a 1.6%-owned punt, which is the whole reason this
    is written every run rather than on request: nobody goes looking for an exposure they do
    not already suspect.
    """
    total = max(len(lineups), 1)
    players, teams = {}, {}
    for lineup in lineups:
        for _, row in lineup["players"].iterrows():
            name = row.get("Name")
            record = players.setdefault(name, {
                "Name": name, "Pos": row.get("Pos"), "Team": row.get("Team"),
                "Salary": row.get("Salary"), "Proj": row.get("Proj"), "Lineups": 0})
            record["Lineups"] += 1
        for team, count in (lineup.get("teams") or {}).items():
            teams[team] = teams.get(team, 0) + 1

    frame = pd.DataFrame(list(players.values()))
    if not frame.empty:
        frame["Exposure%"] = (frame["Lineups"] / total * 100).round(1)
        frame = frame.sort_values("Lineups", ascending=False).reset_index(drop=True)
    team_frame = pd.DataFrame(
        [{"Team": t, "Lineups": c, "Exposure%": round(c / total * 100, 1)}
         for t, c in sorted(teams.items(), key=lambda kv: -kv[1])])
    return frame, team_frame


def _run(args):
    date = args.date or datetime.today().strftime("%Y-%m-%d")
    info = naming.export_info(args.salaries)
    slate = naming.slate_for(info, requested=args.slate, date=date)

    players, report = build_board(args.salaries, projections=args.projections,
                                  drop_out=not args.allow_out)
    print(f"slate {slate}  ({date})  {len(players)} players, "
          f"{len(report['games'])} games")
    print(f"  projections: {report['projection_source']}")
    if report["out"]:
        print(f"  dropped {len(report['out'])} ruled out: {', '.join(report['out'][:6])}"
              + (" ..." if len(report["out"]) > 6 else ""))
    if report["questionable"]:
        print(f"  questionable, kept: {', '.join(report['questionable'][:6])}"
              + (" ..." if len(report["questionable"]) > 6 else ""))
    if report["unprojected"]:
        print(f"  {report['unprojected']} players have no projection (left blank)")

    if args.write_pool:
        path, note, merged = pool_module.write_pool(
            players, date, slate=slate, merge=not args.reset_pool,
            overwrite=args.overwrite, root=args.root)
        if note:
            print(f"  {note}")
        print(f"  -> {path}   ({merged['carried']} hand-marked rows carried)")
        if merged["orphaned"]:
            print(f"  {len(merged['orphaned'])} edits kept for players no longer on the "
                  f"slate: {', '.join(merged['orphaned'][:5])}")
        return 0

    locks, excludes = _split(args.lock), _split(args.exclude)
    boosts, exposure = {}, {}
    if args.pool is not None:
        path = args.pool or naming.latest("pool", date, slate, args.root)
        if not path or not os.path.exists(path):
            raise SystemExit(f"no pool file for {slate} on {date}; run --write-pool first")
        pool_locks, pool_excludes, boosts, exposure = pool_module.read_pool(path)
        locks = list(dict.fromkeys(locks + pool_locks))
        excludes = list(dict.fromkeys(excludes + pool_excludes))
        print(f"  pool {os.path.basename(path)}: {len(pool_locks)} locks, "
              f"{len(pool_excludes)} excludes, {len(boosts)} boosts, "
              f"{len(exposure)} exposure caps")

    resolved = pool_module.resolve_exposure(
        players, exposure, args.n,
        report=lambda names: print(f"  [!] exposure set for {len(names)} player(s) no "
                                   f"longer on the slate; ignored"))
    if resolved:
        print(f"  exposure: {len(resolved)} player caps/minimums in force")
    team_exposure = parse_exposure(args.team_exposure, args.n) if args.team_exposure else None
    if team_exposure:
        print(f"  team exposure: " + ", ".join(
            f"{t} {low}-{high}" for t, (low, high) in team_exposure.items()))

    lineups, _, missing = generate(
        players, n_lineups=args.n, objective=args.objective,
        shapes=parse_shape(args.stack_shape) if args.stack_shape else None,
        stacks=parse_stacks(args.stack) if args.stack else None,
        locks=locks, excludes=excludes, boosts=boosts, exposure=resolved,
        team_exposure=team_exposure, stack_at=args.stack_at,
        stack_teams=args.stack_teams, max_overlap=args.max_overlap,
        randomness=args.randomness, min_proj=args.min_proj,
        max_ownership=args.max_ownership, seed=args.seed)

    for kind in ("locks", "excludes", "pins", "team_exposure"):
        names = (missing or {}).get(kind)
        if names:
            print(f"  [!] {kind} not found on the slate: {', '.join(map(str, names))}")
    # Minimums the pace ran out of room for. Reported rather than raised, so an
    # oversubscribed board still hands back the lineups it did manage to build.
    for kind, label in (("exposure_unmet", "player minimum"),
                        ("team_exposure_unmet", "team stack minimum")):
        names = (missing or {}).get(kind)
        if names:
            print(f"  [!] could not meet {label} for: {', '.join(map(str, names))}")
    if not lineups:
        raise SystemExit("no lineups were produced -- loosen the constraints")
    print(f"  built {len(lineups)} lineups")

    rows = lineup_rows(lineups)
    path, note = naming.resolve("lineups", date, slate, args.root, overwrite=args.overwrite)
    rows.to_csv(path, index=False)
    if note:
        print(f"  {note}")
    print(f"  -> {path}")

    player_exposure, team_exposure = exposure_rows(lineups)
    path, note = naming.resolve("exposure", date, slate, args.root, overwrite=args.overwrite)
    combined = pd.concat(
        [player_exposure.assign(Level="player"),
         team_exposure.assign(Level="team")], ignore_index=True)
    combined.to_csv(path, index=False)
    print(f"  -> {path}")

    top = player_exposure.head(args.exposure_top)
    if not top.empty:
        print("\n  most-used players")
        print(top[["Name", "Pos", "Team", "Lineups", "Exposure%"]]
              .to_string(index=False, justify="left"))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate NFL DK lineups.")
    parser.add_argument("--salaries", required=True, help="DKSalaries CSV for the slate.")
    parser.add_argument("--date", default=None, help="Defaults to today.")
    parser.add_argument("--slate", default=None,
                        help="Slate label. Inferred from the export when omitted.")
    parser.add_argument("--projections", default=None,
                        help="CSV with Name, Proj and optionally Ceiling/Floor.")
    parser.add_argument("--n", type=int, default=DEFAULT_LINEUPS)
    parser.add_argument("--objective", default="ceiling",
                        choices=sorted(opt.OBJECTIVES))
    parser.add_argument("--stack-shape", default=None,
                        help="e.g. '3-1+2'. A = pass-game players from the anchor (QB "
                             "counts, and is pinned), B = bring-back from the same game, "
                             "C = a secondary stack from a different game.")
    parser.add_argument("--stack", default=None, help="Explicit stacks, e.g. 'BUF:3,KAN:2'.")
    parser.add_argument("--stack-teams", type=int, default=DEFAULT_STACK_TEAMS,
                        help="How many teams to rotate stack anchors through.")
    parser.add_argument("--team-exposure", default=None, metavar="SPEC",
                        help="Limit how many lineups stack each club, e.g. "
                             "'BUF:20-60%%,KAN:0-30%%'. Percentages are of the set.")
    parser.add_argument("--stack-at", type=int, default=opt.STACK_EXPOSURE_AT, metavar="N",
                        help=f"Pass-game players from one club that count as a stack for "
                             f"--team-exposure (default {opt.STACK_EXPOSURE_AT}).")
    parser.add_argument("--lock", default=None, help="Comma-separated players who must appear.")
    parser.add_argument("--exclude", default=None, help="Comma-separated players to never use.")
    parser.add_argument("--min-proj", type=float, default=None)
    parser.add_argument("--max-ownership", type=float, default=None, metavar="PCT")
    parser.add_argument("--max-overlap", type=int, default=None,
                        help=f"Players shared with any earlier lineup (default "
                             f"{opt.DEFAULT_MAX_OVERLAP}, not fitted).")
    parser.add_argument("--randomness", type=float, default=None,
                        help=f"Objective jitter per lineup (default "
                             f"{opt.DEFAULT_RANDOMNESS}, not fitted).")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--write-pool", action="store_true",
                        help="Write the editable pool and stop.")
    parser.add_argument("--reset-pool", action="store_true",
                        help="With --write-pool, discard existing markup instead of merging.")
    parser.add_argument("--pool", nargs="?", const="", default=None, metavar="PATH",
                        help="Read locks/excludes/boosts/exposure from the pool file.")
    parser.add_argument("--allow-out", action="store_true",
                        help="Keep players DK marks OUT or IR.")
    parser.add_argument("--exposure-top", type=int, default=20, metavar="N")
    parser.add_argument("--root", default=naming.OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace outputs in place instead of writing .rN versions.")
    return _run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
