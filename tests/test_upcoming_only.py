"""Regression coverage for ``--upcoming-only`` across CLI execution paths."""

import sys

import dfs.slate as slate
import scouting_report as sr


def test_upcoming_status_filter_handles_statsapi_shapes_and_case():
    games = [
        {"status": "Final"},
        {"status": "in progress"},
        {"status": {"detailedState": "Game Over"}},
        {"status": "Scheduled"},
        {"status": "Pre-Game"},
    ]

    assert sr._upcoming_games(games) == games[-2:]


def test_cached_all_games_applies_upcoming_only_before_refresh(monkeypatch):
    monkeypatch.setattr(
        slate,
        "cached_games",
        lambda date: [
            (f"{date}_STL_CHC.pkl", "STL", "CHC"),
            (f"{date}_NYY_BOS.pkl", "NYY", "BOS"),
        ],
    )
    monkeypatch.setattr(
        sr,
        "get_available_games",
        lambda date: [
            {"away_team": "STL", "home_team": "CHC", "game_num": 1, "status": "Final"},
            {"away_team": "NYY", "home_team": "BOS", "game_num": 1, "status": "Scheduled"},
        ],
    )
    rendered = []

    def fake_render(date, away, home, **kwargs):
        rendered.append((away, home, kwargs))
        return []

    monkeypatch.setattr(sr, "render_report_from_cache", fake_render)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scouting_report",
            "--date", "2026-09-01",
            "--upcoming-only",
            "--dfs-slate", "late",
            "--refresh-lineups",
            "--from-cache",
            "--all-games",
        ],
    )

    sr.main()

    assert [(away, home) for away, home, _ in rendered] == [("NYY", "BOS")]
    assert rendered[0][2]["refresh_lineups"] is True
    assert rendered[0][2]["dfs_slate"] == "late"
    assert rendered[0][2]["dh_game"] == 1


def test_cached_upcoming_filter_preserves_doubleheader_game_number(monkeypatch):
    date = "2026-09-01"
    monkeypatch.setattr(
        slate,
        "cached_games",
        lambda requested: [
            (f"{date}_STL_CHC.pkl", "STL", "CHC"),
            (f"{date}_STL_CHC_g2.pkl", "STL", "CHC"),
        ],
    )
    monkeypatch.setattr(
        sr,
        "get_available_games",
        lambda requested: [
            {"away_team": "STL", "home_team": "CHC", "game_num": 1, "status": "Final"},
            {"away_team": "STL", "home_team": "CHC", "game_num": 2, "status": "Pre-Game"},
        ],
    )
    rendered = []
    monkeypatch.setattr(
        sr,
        "render_report_from_cache",
        lambda requested, away, home, **kwargs: rendered.append(kwargs["dh_game"]) or [],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["scouting_report", "--date", date, "--from-cache", "--all-games", "--upcoming-only"],
    )

    sr.main()

    assert rendered == [2]
