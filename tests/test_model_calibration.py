import pandas as pd
import pytest

import scouting_report as report


def _game(game_id, date, home, away, home_score, away_score, umpire="Ump One",
          temp="82", wind="10 mph, Out To CF"):
    return {
        "game_id": game_id,
        "date": date,
        "game_datetime": f"{date}T23:00:00Z",
        "game_number": 1,
        "game_type": "R",
        "home_team": home,
        "away_team": away,
        "home_score": home_score,
        "away_score": away_score,
        "venue": "Yankee Stadium",
        "venue_id": 3313,
        "weather": {"temp": temp, "wind": wind},
        "hp_umpire": umpire,
    }


def test_weather_and_umpire_are_numeric_model_features():
    features = report._weather_calibration_features({
        "park_name": "Yankee Stadium",
        "weather": {"temp": "92", "wind": "12 mph, Out To RF"},
        "forecast": {"roof": "open"},
        "ump_tendency": {"Games": 18, "vs Avg": 0.42},
    })

    assert features["temperature_delta_10f"] == pytest.approx(2.0)
    assert features["wind_out_mph"] == pytest.approx(12.0)
    assert features["weather_available"] == 1.0
    assert features["umpire_run_delta"] == pytest.approx(0.42)
    assert features["umpire_available"] == 1.0


def test_closed_roof_neutralizes_outdoor_weather():
    features = report._weather_calibration_features({
        "weather": {"temp": "101", "wind": "20 mph, Out To CF"},
        "forecast": {"roof": "fixed"},
    })

    assert features["roof_closed"] == 1.0
    assert features["temperature_delta_10f"] == 0.0
    assert features["wind_out_mph"] == 0.0


def test_scorecard_passes_weather_into_calibrated_total(monkeypatch):
    monkeypatch.setattr(report, "load_model_calibration", lambda: {
        "reliability": "medium",
        "validation": {"signal_analysis": {"win": {"magnitude_validated": False}}},
        "win_model": {},
        "total_model": {
            "features": ["raw_total", "temperature_delta_10f"],
            "intercept": 8.8,
            "coef": [0.0, 4.0],
        },
    })
    lineup = pd.DataFrame([{"OPS": 0.700, "ISO": 0.140}])
    similar = pd.DataFrame([{"PA": 20, "OPS_proxy": 0.700}])
    bullpen = pd.DataFrame([{"Bullpen Score": 15.0, "Last3D_Pitches": 0}])
    common = dict(
        home_team="NYY", away_team="BOS",
        home_starter_info={"FIP": 4.2}, away_starter_info={"FIP": 4.2},
        home_lineup_df=lineup, away_lineup_df=lineup,
        home_bullpen_df=bullpen, away_bullpen_df=bullpen,
        home_similarity_df=similar, away_similarity_df=similar,
    )

    _, neutral = report.build_game_scorecard(
        **common,
        environment={"park": {"Runs": 1.0}, "weather": {"temp": "72", "wind": "0 mph"}},
    )
    _, hot = report.build_game_scorecard(
        **common,
        environment={"park": {"Runs": 1.0}, "weather": {"temp": "82", "wind": "0 mph"}},
    )

    assert hot["Total Runs"] > neutral["Total Runs"]


def test_stale_payload_scorecard_is_refreshed_without_losing_columns(monkeypatch):
    calibration = {
        "created_at": "2026-08-10 01:00:00",
        "feature_version": 3,
        "reliability": "medium",
        "validation": {"signal_analysis": {
            "win": {"magnitude_validated": True},
            "context": {"components": {"umpire_tendency": {"magnitude_validated": False}}},
        }},
        "win_model": {},
        "total_model": {
            "features": ["raw_total", "temperature_delta_10f"],
            "intercept": 8.8,
            "coef": [0.0, 4.0],
        },
    }
    monkeypatch.setattr(report, "load_model_calibration", lambda path=report.CALIBRATION_PATH: calibration)
    lineup = pd.DataFrame([{"OPS": 0.700, "ISO": 0.140}])
    similar = pd.DataFrame([{"PA": 20, "OPS_proxy": 0.700}])
    bullpen = pd.DataFrame([{"Bullpen Score": 15.0, "Last3D_Pitches": 0}])
    args = [None] * 17
    args[0], args[1], args[2] = lineup, bullpen, {"FIP": 4.2}
    args[7], args[8], args[9] = lineup, bullpen, {"FIP": 4.2}
    args[15], args[16] = "NYY", "BOS"
    payload = {
        "report_args": tuple(args),
        "advanced_context": {
            "environment": {"park": {"Runs": 1.0}, "weather": {"temp": "82", "wind": "0 mph"}},
            "home_similarity": similar,
            "away_similarity": similar,
            "scorecard": pd.DataFrame([
                {"Team": "BOS", "Win Lean": "50%", "Exp Runs": 4.4, "Keep Me": "yes"},
                {"Team": "NYY", "Win Lean": "50%", "Exp Runs": 4.4, "Keep Me": "yes"},
            ]),
            "projection_summary": {"Calibration Fingerprint": "stale"},
        },
        "model_calibration": {"fingerprint": "stale"},
    }

    refreshed, changed, provenance = report.refresh_payload_model_calibration(payload)

    assert changed
    assert provenance["fingerprint"] != "stale"
    assert refreshed["model_calibration"] == provenance
    assert set(refreshed["advanced_context"]["scorecard"]["Keep Me"]) == {"yes"}
    assert refreshed["advanced_context"]["projection_summary"]["Calibration Fingerprint"] == provenance["fingerprint"]

    _, changed_again, _ = report.refresh_payload_model_calibration(refreshed)
    assert not changed_again


def test_payload_refresh_fails_closed_without_calibration_artifact(monkeypatch):
    monkeypatch.setattr(report, "load_model_calibration", lambda: None)

    with pytest.raises(RuntimeError, match="calibration artifact is unavailable"):
        report.refresh_payload_model_calibration({
            "report_args": tuple([None] * 17),
            "advanced_context": {},
        })


def test_collector_is_pregame_and_capped_to_newest(monkeypatch):
    schedule = [
        _game(1, "2026-04-01", "NYY", "BOS", 5, 3),
        _game(2, "2026-04-02", "BOS", "NYY", 2, 4),
    ]
    monkeypatch.setattr(report, "_calibration_schedule", lambda start, end: schedule)

    frame = report.collect_calibration_dataset(
        "2026-04-01", "2026-04-02", max_games=1,
        data_path=None, checkpoint_every=0,
    )

    assert frame["game_id"].tolist() == [2]
    row = frame.iloc[0]
    # Game 1 is context for game 2, but game 2 has not informed its own snapshot.
    assert row["home_rpg"] == pytest.approx((3 + 4.4 * 20) / 21)
    assert row["home_rapg"] == pytest.approx((5 + 4.4 * 20) / 21)
    assert row["away_rpg"] == pytest.approx((5 + 4.4 * 20) / 21)
    assert row["away_rapg"] == pytest.approx((3 + 4.4 * 20) / 21)
    assert row["umpire_games"] == 1


def test_opening_week_team_rates_are_shrunk_to_priors():
    snapshot = report._shrunk_team_calibration_snapshot(
        wins=1, losses=0, runs=14, allowed=2, recent_results=[1]
    )

    assert snapshot["record_pct"] == pytest.approx(11 / 21)
    assert snapshot["record_pct"] < 0.55
    assert snapshot["rpg"] < 5.0
    assert snapshot["run_diff_per_game"] == pytest.approx(12 / 21)


def test_bulk_schedule_filters_non_regular_games(monkeypatch):
    regular = {
        "gamePk": 10,
        "officialDate": "2025-06-12",
        "gameDate": "2025-06-12T23:00:00Z",
        "gameType": "R",
        "gameNumber": 1,
        "status": {"abstractGameState": "Final", "detailedState": "Final"},
        "teams": {
            "home": {"score": 4, "team": {"id": 147, "name": "New York Yankees"}},
            "away": {"score": 2, "team": {"id": 111, "name": "Boston Red Sox"}},
        },
        "venue": {"id": 3313, "name": "Yankee Stadium"},
        "weather": {"temp": "75", "wind": "6 mph, L To R"},
        "officials": [{
            "officialType": "Home Plate",
            "official": {"fullName": "Ump One"},
        }],
    }
    resumed = {**regular, "gameDate": "2025-06-13T17:00:00Z"}
    postseason = {**regular, "gamePk": 11, "gameType": "F"}
    seen = {}

    def fake_request(url, params, namespace):
        seen.update(params)
        return {"dates": [{"date": "2025-06-12", "games": [regular, resumed, postseason]}]}

    monkeypatch.setattr(report, "cached_json_request", fake_request)
    games = report._calibration_schedule("2025-06-12", "2025-06-12")

    assert [game["game_id"] for game in games] == [10]
    assert games[0]["game_datetime"] == "2025-06-13T17:00:00Z"
    assert seen["gameType"] == "R"
    assert "weather" in seen["hydrate"]
    assert "officials" in seen["hydrate"]


def test_multiseason_fit_uses_latest_season_holdout(monkeypatch):
    rows = []
    for i in range(80):
        season = 2025 if i < 40 else 2026
        rows.append({
            "game_id": i + 1,
            "date": f"{season}-06-{(i % 28) + 1:02d}",
            "game": f"A{i}@H{i}",
            "season": season,
            "home_win": i % 2,
            "home_runs": 3 + (i % 4),
            "away_runs": 2 + ((i + 1) % 4),
            "total_runs": 8 + (i % 5),
            "edge_diff": -0.3 + (i % 7) * 0.1,
            "raw_total": 8.2 + (i % 5) * 0.2,
            "raw_home_runs": 4.2 + (i % 3) * 0.1,
            "raw_away_runs": 4.0 + (i % 3) * 0.1,
            "temperature_delta_10f": (i % 5) - 2,
            "wind_out_mph": (i % 9) - 4,
            "roof_closed": float(i % 10 == 0),
            "weather_available": 1.0,
            "umpire_run_delta": ((i % 5) - 2) * 0.1,
            "umpire_available": float(i % 4 != 0),
        })
    frame = pd.DataFrame(rows)
    monkeypatch.setattr(report, "collect_calibration_dataset", lambda **kwargs: frame.copy())

    payload = report.calibrate_model(
        start_date="2025-01-01", end_date="2026-12-31",
        output_path=None,
        data_path=None,
    )

    assert payload["validation"]["method"] == "latest_season_holdout"
    assert payload["validation"]["train_games"] == 40
    assert payload["validation"]["holdout_games"] == 40
    assert "temperature_delta_10f" in payload["total_model"]["features"]
    assert "umpire_run_delta" in payload["total_model"]["features"]
    analysis = payload["validation"]["signal_analysis"]
    assert len(analysis["win"]["buckets"]) == 5
    assert len(analysis["total_adjustment"]["buckets"]) == 5
    assert "weather" in analysis["context"]["components"]
