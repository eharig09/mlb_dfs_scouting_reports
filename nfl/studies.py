"""The measurements behind `docs/nfl/pff_integration.md`, as runnable code.

Every empirical claim in that document -- which PFF signals persist, how big the man/zone
effect actually is, whether O-line continuity is worth anything, what drives QB-receiver
correlation -- came out of one of these functions. They live here rather than in a scratch
directory so the numbers can be re-derived when new seasons land, and so a claim that stops
being true is discoverable rather than frozen in prose.

    python -m nfl.studies                 # list the studies
    python -m nfl.studies persistence     # run one
    python -m nfl.studies all

**These are studies, not model code.** Nothing in the package imports from here; they read
data, print tables, and return frames. A finding that earns its way into the model gets
copied into the module that uses it, with the measurement quoted next to the constant.
"""

import sys

import numpy as np
import pandas as pd

from nfl import backtest, pffdata, usage
from nfl import data as nfl_data
from nfl.salaries import canon_team, normalize_name


def _num(frame, column):
    return pd.to_numeric(frame.reindex(columns=[column])[column], errors="coerce")


def _yoy(frames, columns, label, index="gsis_id"):
    """Mean year-over-year self-correlation for each column, and the per-pair detail.

    This is the whole method behind the persistence table. **A signal that does not
    correlate with its own next season cannot forecast one**, whatever it explains after
    the fact -- so this is the gate every candidate had to pass before being built on.
    """
    seasons = sorted(frames)
    accumulated = {c: [] for c in columns}
    sizes = []
    for earlier in seasons[:-1]:
        later = earlier + 1
        if later not in frames:
            continue
        joined = frames[earlier][columns].join(
            frames[later][columns], lsuffix="_a", rsuffix="_b", how="inner")
        sizes.append(len(joined))
        for column in columns:
            accumulated[column].append(joined[f"{column}_a"].corr(joined[f"{column}_b"]))

    rows = [{"signal": c, "mean_r": np.mean(v), "pairs": len(v)}
            for c, v in accumulated.items()]
    out = pd.DataFrame(rows).sort_values("mean_r", ascending=False).reset_index(drop=True)
    print(f"\n=== {label}  (n ~ {int(np.mean(sizes)) if sizes else 0} per pair) ===")
    print(out.round(3).to_string(index=False))
    return out


# --- study: which signals persist -----------------------------------------------------------


def _receivers(season, min_routes=150):
    frame = pffdata.load("receiving_summary", season)
    out = pd.DataFrame({
        "gsis_id": frame["gsis_id"],
        "routes": _num(frame, "routes"),
        "slot_rate": _num(frame, "slot_rate"),
        "wide_rate": _num(frame, "wide_rate"),
        "inline_rate": _num(frame, "inline_rate"),
        "route_rate": _num(frame, "route_rate"),
        "adot": _num(frame, "avg_depth_of_target"),
        "yprr": _num(frame, "yprr"),
        "grade_route": _num(frame, "grades_pass_route"),
        "yac_per_rec": _num(frame, "yards_after_catch_per_reception"),
    }).dropna(subset=["gsis_id"])
    out["tprr"] = _num(frame, "targets") / out["routes"]
    return out[out["routes"] >= min_routes].set_index("gsis_id")


def _defense_scheme(season):
    frame = pffdata.load("defense_coverage_scheme", season)
    grouped = pd.DataFrame({
        "team": frame["team"],
        "man_snaps": _num(frame, "man_snap_counts_coverage").fillna(0),
        "zone_snaps": _num(frame, "zone_snap_counts_coverage").fillna(0),
        "man_targets": _num(frame, "man_targets").fillna(0),
        "zone_targets": _num(frame, "zone_targets").fillna(0),
        "man_yards": _num(frame, "man_yards").fillna(0),
        "zone_yards": _num(frame, "zone_yards").fillna(0),
    }).groupby("team").sum()
    grouped["man_rate"] = grouped["man_snaps"] / (grouped["man_snaps"] + grouped["zone_snaps"])
    grouped["man_ypt"] = grouped["man_yards"] / grouped["man_targets"]
    grouped["zone_ypt"] = grouped["zone_yards"] / grouped["zone_targets"]
    grouped["scheme_gap"] = grouped["man_ypt"] - grouped["zone_ypt"]
    return grouped


def _defense_slot(season):
    frame = pffdata.load("slot_coverage", season)
    grouped = pd.DataFrame({
        "team": frame["team"],
        "snaps": _num(frame, "coverage_snaps").fillna(0),
        "targets": _num(frame, "targets").fillna(0),
        "yards": _num(frame, "yards").fillna(0),
        "tds": _num(frame, "touchdowns").fillna(0),
    }).groupby("team").sum()
    grouped["slot_ypt"] = grouped["yards"] / grouped["targets"]
    grouped["slot_yards_per_snap"] = grouped["yards"] / grouped["snaps"]
    grouped["slot_td_rate"] = grouped["tds"] / grouped["targets"]
    return grouped


def persistence(seasons=range(2020, 2026)):
    """The central table: what predicts itself, and what does not.

    The result that reshaped the whole integration is the gap between the two halves --
    deployment sits at 0.8-0.95 while every *defensive allowed-rate* sits under 0.11.
    """
    seasons = list(seasons)
    receivers = {s: _receivers(s) for s in seasons}
    _yoy(receivers, ["inline_rate", "wide_rate", "adot", "route_rate", "slot_rate",
                     "yac_per_rec", "tprr", "yprr", "grade_route"],
         "RECEIVER deployment and rates (>=150 routes)")

    defense = {s: _defense_scheme(s) for s in seasons}
    _yoy(defense, ["man_rate", "man_ypt", "zone_ypt", "scheme_gap"],
         "DEFENSE man/zone -- tendency persists, quality does not", index="team")

    slot = {s: _defense_slot(s) for s in seasons}
    return _yoy(slot, ["slot_ypt", "slot_yards_per_snap", "slot_td_rate"],
                "DEFENSE slot coverage -- allowed-rates are noise", index="team")


# --- study: how big is the man/zone edge ----------------------------------------------------


def _scheme_receivers(season, min_routes=50):
    frame = pffdata.load("receiving_scheme", season)
    out = pd.DataFrame({
        "gsis_id": frame["gsis_id"],
        "games": _num(frame, "player_game_count"),
        "man_routes": _num(frame, "man_routes"),
        "zone_routes": _num(frame, "zone_routes"),
        "man_targets": _num(frame, "man_targets"),
        "zone_targets": _num(frame, "zone_targets"),
        "man_yprr": _num(frame, "man_yprr"),
        "zone_yprr": _num(frame, "zone_yprr"),
    }).dropna(subset=["gsis_id"])
    out = out[(out["man_routes"] >= min_routes) & (out["zone_routes"] >= min_routes)]
    out["yprr_gap"] = out["man_yprr"] - out["zone_yprr"]
    out["target_rate_gap"] = (out["man_targets"] / out["man_routes"]
                              - out["zone_targets"] / out["zone_routes"])
    out["routes_pg"] = (out["man_routes"] + out["zone_routes"]) / out["games"]
    return out.set_index("gsis_id")


def man_zone_effect(seasons=range(2020, 2026)):
    """Sizing the man/zone matchup end to end, in targets per game.

    Two findings stack against it. The *efficiency* gap barely persists, so "man-beater" is
    mostly last season's noise. The *target-rate* gap does persist -- so volume, not
    efficiency, is the only honest channel -- but shrunk by its own regression slope and
    applied across the league's full spread of man rates it is worth about a tenth of a
    target per game. Display and tie-break, never a projection input.
    """
    seasons = list(seasons)
    frames = {s: _scheme_receivers(s) for s in seasons}
    _yoy(frames, ["yprr_gap", "target_rate_gap", "man_yprr", "zone_yprr"],
         "RECEIVER man/zone splits -- efficiency gap vs target-rate gap")

    newest, previous = seasons[-1], seasons[-2]
    joined = frames[previous].join(frames[newest], lsuffix="_a", rsuffix="_b", how="inner")
    slope = np.polyfit(joined["target_rate_gap_a"], joined["target_rate_gap_b"], 1)[0]

    league_man_rate = _defense_scheme(newest)["man_rate"]
    routes = frames[newest]["routes_pg"].median()
    gaps = frames[newest]["target_rate_gap"]

    print(f"\n=== man/zone effect size ({newest}) ===")
    print(f"  regression slope of this year's gap on last year's : {slope:+.3f}")
    print(f"  median routes per game                             : {routes:.1f}")
    print(f"  league man rate  min {league_man_rate.min():.3f} "
          f"mean {league_man_rate.mean():.3f} max {league_man_rate.max():.3f}")
    rows = []
    for label, gap in (("90th pct (man-leaning)", gaps.quantile(0.90)),
                       ("10th pct (zone-leaning)", gaps.quantile(0.10))):
        for opponent, rate in (("most man-heavy", league_man_rate.max()),
                               ("most zone-heavy", league_man_rate.min())):
            delta = gap * slope * (rate - league_man_rate.mean()) * routes
            rows.append({"receiver": label, "opponent": opponent,
                         "targets_per_game": round(delta, 3)})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    return out


# --- study: the O-line inputs ---------------------------------------------------------------


def oline_inputs(seasons=(2023, 2024, 2025)):
    """Do grades travel, does draft capital predict, does continuity matter?

    The third answer is the surprising one: continuity correlates with next season's unit
    grade only because good lines keep their starters. Against the *change* it is nothing.
    """
    from nfl import oline as oline_module

    seasons = list(seasons)
    frames = {s: oline_module.lineman_season(s) for s in seasons}

    print("\n=== does an O-line grade travel with the player? ===")
    rows = []
    for earlier in seasons[:-1]:
        later = earlier + 1
        joined = frames[earlier].set_index("name_key")[["grade", "pass_block", "team"]].join(
            frames[later].set_index("name_key")[["grade", "pass_block", "team"]],
            lsuffix="_a", rsuffix="_b", how="inner")
        stayed = joined[joined["team_a"] == joined["team_b"]]
        moved = joined[joined["team_a"] != joined["team_b"]]
        for label, group in (("stayed", stayed), ("moved", moved)):
            rows.append({"pair": f"{earlier}->{later}", "group": label, "n": len(group),
                         "grade_r": round(group["grade_a"].corr(group["grade_b"]), 3),
                         "pass_block_r": round(
                             group["pass_block_a"].corr(group["pass_block_b"]), 3)})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))

    print("\n=== draft capital -> rookie O-line grade ===")
    rosters = nfl_data.load_rosters(seasons)
    rosters = rosters.assign(
        _exp=pd.to_numeric(rosters.reindex(columns=["years_exp"])["years_exp"],
                           errors="coerce"),
        _pick=pd.to_numeric(rosters.reindex(columns=["draft_number"])["draft_number"],
                            errors="coerce"))
    # The years_exp gate is load-bearing: draft_number stays on a roster row forever.
    #
    # Joined on **name**, not gsis_id: nflverse fills `pff_id` for only about two thirds of
    # linemen, so the id crosswalk drops a third of the sample on this side of the ball.
    # Keying on the id here silently cut 68 rookie seasons to fewer than 20.
    rosters["_name"] = rosters.reindex(columns=["full_name"])["full_name"].map(normalize_name)
    rookies = rosters[rosters["_exp"] == 0][["_name", "season", "_pick"]].dropna()
    rookies = rookies.drop_duplicates(["_name", "season"])
    stacked = pd.concat(frames.values())
    merged = stacked.merge(rookies, left_on=["name_key", "season"],
                           right_on=["_name", "season"], how="inner")
    if len(merged) > 20:
        spearman = merged["_pick"].corr(merged["grade"], method="spearman")
        fit = np.polyfit(np.log(merged["_pick"]), merged["grade"], 1)
        print(f"  n={len(merged)}  spearman(pick, grade) = {spearman:+.3f}")
        print(f"  grade ~ {fit[0]:+.2f} * log(pick) + {fit[1]:.1f}")
    return out


# --- study: what drives a stack -------------------------------------------------------------


def stack_correlation(seasons=(2023, 2024, 2025)):
    """QB-to-receiver weekly DK correlation, by target rank and by team concentration.

    The headline is that WR1 and WR2 correlate with the quarterback equally (+0.367 vs
    +0.365), while WR2 costs meaningfully less target share -- so the WR1 premium in a stack
    is buying a correlation already available one slot down. Concentration, the intuitive
    mechanism, explains almost nothing and is not even monotonic.
    """
    seasons = list(seasons)
    actuals = backtest.offense_actuals(nfl_data.load_weekly(seasons))
    actuals = actuals[actuals["dk_points"].notna()].copy()
    actuals["team"] = actuals["team"].map(canon_team)

    rows = []
    for season in seasons:
        distribution = usage.team_target_distribution(season)
        season_actuals = actuals[actuals["season"] == season]
        quarterbacks = (season_actuals[season_actuals["position"] == "QB"]
                        .groupby(["team", "player_id"]).size().reset_index(name="games"))
        starters = quarterbacks.sort_values("games", ascending=False).drop_duplicates("team")

        for team, group in distribution.groupby("team"):
            receivers = group[group["pos"].isin(("WR", "TE"))].sort_values(
                "target_share", ascending=False)
            starter = starters[starters["team"] == team]
            if len(receivers) < 3 or starter.empty:
                continue
            concentration = float((group["target_share"] ** 2).sum())
            qb_weeks = season_actuals[
                season_actuals["player_id"] == starter.iloc[0]["player_id"]][
                    ["week", "dk_points"]]
            if len(qb_weeks) < 8:
                continue
            for rank, (_, receiver) in enumerate(receivers.head(3).iterrows(), start=1):
                weeks = season_actuals[
                    season_actuals["player_id"] == receiver["gsis_id"]][["week", "dk_points"]]
                joined = qb_weeks.merge(weeks, on="week", suffixes=("_qb", "_rec"))
                if len(joined) < 8:
                    continue
                rows.append({
                    "season": season, "team": team, "rank": rank,
                    "concentration": concentration, "target_share": receiver["target_share"],
                    "correlation": joined["dk_points_qb"].corr(joined["dk_points_rec"]),
                })

    frame = pd.DataFrame(rows)
    by_rank = frame.groupby("rank").agg(
        n=("correlation", "size"), mean_r=("correlation", "mean"),
        median_r=("correlation", "median"), mean_target_share=("target_share", "mean"))
    print("\n=== QB <-> receiver weekly DK correlation, by team target rank ===")
    print(by_rank.round(3).to_string())

    first = frame[frame["rank"] == 1].dropna(subset=["correlation", "concentration"])
    print("\n=== is target concentration the mechanism? (WR1) ===")
    print(f"  corr(team target HHI, QB-WR1 correlation)  = "
          f"{first['concentration'].corr(first['correlation']):+.3f}   n={len(first)}")
    print(f"  corr(WR1 target share, QB-WR1 correlation) = "
          f"{first['target_share'].corr(first['correlation']):+.3f}")
    low, high = first["concentration"].quantile(0.33), first["concentration"].quantile(0.67)
    for label, group in (("flat", first[first["concentration"] <= low]),
                         ("middle", first[(first["concentration"] > low)
                                          & (first["concentration"] < high)]),
                         ("concentrated", first[first["concentration"] >= high])):
        print(f"    {label:14s} n={len(group):3d}  mean r {group['correlation'].mean():+.3f}")
    return by_rank


STUDIES = {
    "persistence": persistence,
    "man_zone": man_zone_effect,
    "oline": oline_inputs,
    "stacks": stack_correlation,
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("studies: " + ", ".join(sorted(STUDIES)) + ", all")
        return 0
    names = sorted(STUDIES) if argv[0] == "all" else argv
    for name in names:
        if name not in STUDIES:
            print(f"unknown study '{name}'; have {sorted(STUDIES)}, or 'all'")
            return 1
        STUDIES[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
