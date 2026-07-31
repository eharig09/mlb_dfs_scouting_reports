"""Score past projections against what actually happened.

Pulls final box scores for every cached game in a date range, converts them to DK points,
and joins to the projections on MLBAM id (no name matching involved). The point is not a
single accuracy number -- it is whether the ranking is any good, and whether the ceiling
and floor bands mean what they claim to mean.
"""

import pickle

import pandas as pd
import statsapi

from .projections import project_game
from .scoring import hitter_points, pitcher_points
from .slate import cached_games


def _ip_to_float(value):
    """MLB innings strings are base-3 after the decimal: '5.2' is five and two thirds."""
    try:
        whole, _, outs = str(value).partition(".")
        return int(whole or 0) + int(outs or 0) / 3.0
    except (TypeError, ValueError):
        return 0.0


def actual_points(game_id):
    """Actual DK points for everyone in a finished game, keyed by MLBAM id."""
    try:
        box = statsapi.boxscore_data(game_id)
    except Exception:
        return {}

    results = {}
    for side in ("away", "home"):
        for entry in (box.get(side, {}).get("players") or {}).values():
            player_id = (entry.get("person") or {}).get("id")
            if not player_id:
                continue
            stats = entry.get("stats") or {}

            batting = stats.get("batting") or {}
            if batting.get("atBats") is not None or batting.get("baseOnBalls"):
                hits = batting.get("hits", 0) or 0
                doubles = batting.get("doubles", 0) or 0
                triples = batting.get("triples", 0) or 0
                homers = batting.get("homeRuns", 0) or 0
                results[int(player_id)] = {
                    "Type": "H",
                    "Actual": hitter_points({
                        "1B": max(0, hits - doubles - triples - homers),
                        "2B": doubles, "3B": triples, "HR": homers,
                        "R": batting.get("runs", 0) or 0,
                        "RBI": batting.get("rbi", 0) or 0,
                        "BB": batting.get("baseOnBalls", 0) or 0,
                        "HBP": batting.get("hitByPitch", 0) or 0,
                        "SB": batting.get("stolenBases", 0) or 0,
                    }),
                    "Actual PA": (batting.get("atBats", 0) or 0) + (batting.get("baseOnBalls", 0) or 0),
                }

            pitching = stats.get("pitching") or {}
            if pitching.get("inningsPitched") is not None:
                innings = _ip_to_float(pitching.get("inningsPitched"))
                results[int(player_id)] = {
                    "Type": "P",
                    "Actual": pitcher_points({
                        "IP": innings,
                        "K": pitching.get("strikeOuts", 0) or 0,
                        "W": pitching.get("wins", 0) or 0,
                        "ER": pitching.get("earnedRuns", 0) or 0,
                        "H": pitching.get("hits", 0) or 0,
                        "BB": pitching.get("baseOnBalls", 0) or 0,
                        "HBP": pitching.get("hitByPitch", 0) or 0,
                    }),
                    "Actual IP": round(innings, 2),
                }
    return results


def run_backtest(dates, data_dir=None):
    """Project every cached game on each date and join actual results.

    Returns a row per projected player. `Played` is False when a projected hitter never
    appeared -- a late scratch -- which is a real DFS risk and is reported separately
    rather than being scored as a miss.
    """
    from .slate import REPORT_DATA_DIR
    data_dir = data_dir or REPORT_DATA_DIR

    rows = []
    for date in dates:
        for path, away, home in cached_games(date, data_dir):
            try:
                with open(path, "rb") as handle:
                    payload = pickle.load(handle)
                projections = project_game(payload)
            except Exception as error:
                print(f"  skip {date} {away}@{home}: {error}")
                continue

            game_id = (payload["advanced_context"].get("environment") or {}).get("game_id")
            if not game_id:
                continue
            outcomes = actual_points(int(game_id))
            if not outcomes:
                continue

            for projection in projections:
                mlbam = projection.get("MLBAM")
                if mlbam is None or pd.isna(mlbam):
                    continue
                outcome = outcomes.get(int(mlbam))
                rows.append({
                    "Date": date,
                    "Game": f"{away}@{home}",
                    "Name": projection["Name"],
                    "Team": projection["Team"],
                    "Type": projection["Type"],
                    "Slot": projection.get("Slot"),
                    "Proj": projection["Proj"],
                    "Ceiling": projection["Ceiling"],
                    "PA": projection.get("PA"),
                    "HR": projection.get("HR"),
                    "Matchup": projection.get("Matchup"),
                    "Team Runs": projection.get("Team Runs"),
                    "IP": projection.get("IP"),
                    "K": projection.get("K"),
                    "Actual PA": (outcome or {}).get("Actual PA"),
                    "Actual IP": (outcome or {}).get("Actual IP"),
                    "Actual": outcome["Actual"] if outcome else None,
                    "Played": bool(outcome),
                })
    return pd.DataFrame(rows)


def _spearman(frame):
    if len(frame) < 3:
        return float("nan")
    return frame["Proj"].corr(frame["Actual"], method="spearman")


def summarize(results):
    """Accuracy, ranking quality, and band calibration, reported per player type."""
    lines = []
    scratched = results[~results["Played"]]
    played = results[results["Played"]].copy()

    lines.append(f"projected {len(results)} player-games, {len(played)} played, "
                 f"{len(scratched)} did not appear "
                 f"({len(scratched) / max(1, len(results)) * 100:.1f}% scratch rate)")

    for player_type, label in (("H", "HITTERS"), ("P", "PITCHERS")):
        group = played[played["Type"] == player_type]
        if group.empty:
            continue
        error = group["Actual"] - group["Proj"]
        # A ceiling claiming to be a ~90th-percentile outcome should be beaten ~10% of
        # the time. Far off in either direction means the band is decorative.
        exceed = (group["Actual"] > group["Ceiling"]).mean() * 100

        lines.append("")
        lines.append(f"{label}  n={len(group)}")
        lines.append(f"  proj mean {group['Proj'].mean():6.2f}   actual mean {group['Actual'].mean():6.2f}"
                     f"   bias {error.mean():+.2f}")
        lines.append(f"  MAE {error.abs().mean():5.2f}   RMSE {(error ** 2).mean() ** 0.5:5.2f}"
                     f"   actual SD {group['Actual'].std():5.2f}")
        lines.append(f"  pearson {group['Proj'].corr(group['Actual']):.3f}"
                     f"   spearman {_spearman(group):.3f}")
        lines.append(f"  ceiling exceeded {exceed:.1f}% of the time (target ~10%)")

        # Does ranking higher actually earn more? This is what a recommender lives on.
        ranked = group.sort_values("Proj", ascending=False)
        size = max(1, len(ranked) // 5)
        top, bottom = ranked.head(size), ranked.tail(size)
        lines.append(f"  top quintile actual {top['Actual'].mean():5.2f}"
                     f"  vs bottom quintile {bottom['Actual'].mean():5.2f}"
                     f"  (spread {top['Actual'].mean() - bottom['Actual'].mean():+.2f})")

        # Calibration curve: within each projection bucket, does actual track projection?
        buckets = pd.qcut(group["Proj"], min(5, group["Proj"].nunique()), duplicates="drop")
        table = group.groupby(buckets, observed=True).agg(
            n=("Actual", "size"), proj=("Proj", "mean"), actual=("Actual", "mean"))
        lines.append("  calibration by projection bucket:")
        for interval, row in table.iterrows():
            lines.append(f"    {str(interval):>18}  n={int(row['n']):4d}  "
                         f"proj {row['proj']:6.2f}  actual {row['actual']:6.2f}  "
                         f"{row['actual'] - row['proj']:+6.2f}")

    return "\n".join(lines)
