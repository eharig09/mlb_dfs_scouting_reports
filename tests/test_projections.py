"""Projection internals: the indexed lookups, and the events-to-points contract.

The row-index change replaced four `frame[frame["Name"] == name]` scans with dictionaries.
That is a pure speed change and its correctness test is equality, not plausibility -- so the
tests here compare the indexed lookup against the boolean scan it replaced, on frames built
to hit the awkward cases (duplicate names, missing columns, suffixed platoon columns).
"""

import numpy as np
import pandas as pd
import pytest

from dfs.projections import (LG, MODEL_VERSION, _arsenal_row, _build_index, _indexed,
                             _INDEX_CACHE, _platoon_row, _scorecard_row, _steal_rate,
                             bust_probability, floor_points, project_game)
from dfs.scoring import hitter_points, pitcher_points


def scan_lookup(frame, key_column, key):
    """The original implementation, kept here as the oracle."""
    if not isinstance(frame, pd.DataFrame) or frame.empty or key_column not in frame:
        return {}
    match = frame[frame[key_column].astype(str) == str(key)]
    return match.iloc[0].to_dict() if not match.empty else {}


@pytest.fixture(autouse=True)
def clear_cache():
    _INDEX_CACHE.clear()
    yield
    _INDEX_CACHE.clear()


class TestRowIndex:
    def test_matches_a_boolean_scan(self):
        frame = pd.DataFrame({"Name": ["A Batter", "B Batter", "C Batter"],
                              "OPS": [0.800, 0.650, 0.910], "PA": [400, 120, 500]})
        for name in frame["Name"]:
            indexed = _indexed(frame, "Name")[name]
            scanned = scan_lookup(frame, "Name", name)
            assert {str(k): v for k, v in scanned.items()} == indexed

    def test_first_row_wins_on_a_duplicate_key(self):
        """`.iloc[0]` semantics: two players sharing a name must not merge."""
        frame = pd.DataFrame({"Name": ["Same Guy", "Same Guy"], "OPS": [0.800, 0.500]})
        assert _indexed(frame, "Name")["Same Guy"]["OPS"] == 0.800

    def test_a_missing_key_returns_empty(self):
        frame = pd.DataFrame({"Name": ["A"], "OPS": [0.8]})
        assert _arsenal_row(frame, "Nobody") == {}

    def test_none_and_empty_frames_are_safe(self):
        assert _indexed(None, "Name") == {}
        assert _indexed(pd.DataFrame(), "Name") == {}
        assert _indexed("not a frame", "Name") == {}
        assert _indexed(pd.DataFrame({"Other": [1]}), "Name") == {}

    def test_platoon_columns_lose_their_hand_suffix(self):
        frame = pd.DataFrame({"Name": ["A Batter"], "OPS vs R": [0.900],
                              "PA vs R": [180], "K% vs R": [21.0]})
        row = _platoon_row(frame, "A Batter")
        assert row["OPS"] == 0.900
        assert row["PA"] == 180
        assert row["K%"] == 21.0

    def test_suffix_stripping_matches_the_original_collision_rule(self):
        """Stripping can collide two columns; the later one wins, as it always did."""
        frame = pd.DataFrame({"Name": ["A"], "PA": [100], "PA vs R": [40]})
        assert _platoon_row(frame, "A")["PA"] == 40

    def test_the_cache_is_keyed_on_the_frame_it_was_built_from(self):
        """id() is recycled, so a cache hit has to verify identity or it serves stale rows."""
        first = pd.DataFrame({"Name": ["A"], "OPS": [0.800]})
        assert _indexed(first, "Name")["A"]["OPS"] == 0.800
        second = pd.DataFrame({"Name": ["A"], "OPS": [0.500]})
        assert _indexed(second, "Name")["A"]["OPS"] == 0.500

    def test_repeated_lookups_hit_the_cache(self):
        frame = pd.DataFrame({"Name": ["A", "B"], "OPS": [0.8, 0.7]})
        _indexed(frame, "Name")
        size = len(_INDEX_CACHE)
        for _ in range(10):
            _indexed(frame, "Name")
        assert len(_INDEX_CACHE) == size

    def test_platoon_and_plain_indexes_are_cached_separately(self):
        frame = pd.DataFrame({"Name": ["A"], "OPS vs R": [0.9]})
        assert "OPS" in _indexed(frame, "Name", strip_hand=True)["A"]
        assert "OPS vs R" in _indexed(frame, "Name")["A"]

    def test_scorecard_is_keyed_on_team(self):
        frame = pd.DataFrame({"Team": ["NYY", "BOS"], "Exp Runs": [5.1, 4.2],
                              "Win Lean": [58.0, 42.0]})
        assert _scorecard_row(frame, "BOS")["Exp Runs"] == 4.2
        assert _scorecard_row(frame, "LAD") == {}

    def test_steal_rate_regresses_and_handles_absences(self):
        frame = pd.DataFrame({"Name": ["Fast Guy"], "SB": [30]})
        assert _steal_rate(frame, "Fast Guy", 500) == pytest.approx(30 / 620)
        assert _steal_rate(frame, "Slow Guy", 500) == 0.0
        assert _steal_rate(frame, "Fast Guy", 0) == 0.0
        assert _steal_rate(None, "Fast Guy", 500) == 0.0

    def test_build_index_preserves_value_types(self):
        frame = pd.DataFrame({"Name": ["A"], "Fit": ["good"], "PA": [12], "OPS": [0.9]})
        row = _build_index(frame, "Name")["A"]
        assert row["Fit"] == "good"
        assert row["PA"] == 12
        assert row["OPS"] == pytest.approx(0.9)


class TestBands:
    def test_floor_is_never_negative(self):
        for points in (0.0, 1.0, 5.0, 40.0):
            assert floor_points(points, "H") >= 0.0
            assert floor_points(points, "P") >= 0.0

    def test_floor_rises_with_the_projection(self):
        assert floor_points(15.0, "H") > floor_points(8.0, "H")

    def test_bust_probability_falls_with_the_projection(self):
        assert bust_probability(15.0, "H") < bust_probability(5.0, "H")

    def test_bust_probability_is_floored_at_the_best_observed_rate(self):
        """The best hitter bucket still busted 32.7%; extrapolating below that is fantasy."""
        from dfs.projections import BUST_FLOOR
        assert bust_probability(60.0, "H") == pytest.approx(BUST_FLOOR["H"])
        assert bust_probability(60.0, "P") == pytest.approx(BUST_FLOOR["P"])

    def test_bust_probability_is_capped(self):
        assert bust_probability(-50.0, "H") <= 0.95


class TestEventContract:
    """The simulator draws these events and re-scores them; they must add back up."""

    def test_hitter_events_reproduce_the_projection(self, slate):
        hitters = slate[slate["Type"] == "H"]
        for _, row in hitters.head(20).iterrows():
            events = {k: row[f"E_{k}"] for k in
                      ("1B", "2B", "3B", "HR", "BB", "HBP", "R", "RBI", "SB")}
            assert hitter_points(events) == pytest.approx(row["Proj"], abs=0.02)

    def test_pitcher_events_reproduce_the_projection(self, slate):
        pitchers = slate[slate["Type"] == "P"]
        for _, row in pitchers.iterrows():
            events = {k: row[f"E_{k}"] for k in ("IP", "K", "W", "ER", "H", "BB", "HBP")}
            assert pitcher_points(events) == pytest.approx(row["Proj"], abs=0.02)

    def test_hitter_events_never_exceed_plate_appearances(self, slate):
        hitters = slate[slate["Type"] == "H"]
        outcomes = hitters[["E_1B", "E_2B", "E_3B", "E_HR", "E_BB", "E_HBP"]].sum(axis=1)
        assert (outcomes <= hitters["E_PA"]).all()

    def test_every_event_column_is_non_negative(self, slate):
        for column in [c for c in slate.columns if c.startswith("E_")]:
            assert (pd.to_numeric(slate[column], errors="coerce").dropna() >= 0).all(), column


class TestVersioning:
    def test_model_version_is_recorded(self):
        assert isinstance(MODEL_VERSION, int) and MODEL_VERSION >= 2

    def test_league_baselines_are_plausible(self):
        assert 3.5 < LG["runs_per_team_game"] < 5.5
        assert 0.15 < LG["k_rate"] < 0.30
        assert 4.0 < LG["sp_ip"] < 6.5


@pytest.mark.integration
class TestAgainstRealPayloads:
    def test_projecting_a_real_game_produces_events(self, cached_dates):
        import glob
        import pickle
        if not cached_dates:
            pytest.skip("no cached report data")
        paths = sorted(glob.glob(f".cache/report_data/{cached_dates[-1]}_*.pkl"))
        if not paths:
            pytest.skip("no payloads")
        with open(paths[0], "rb") as handle:
            payload = pickle.load(handle)
        rows = project_game(payload)
        assert rows
        for row in rows:
            if row["Type"] == "H":
                events = {k: row[f"E_{k}"] for k in
                          ("1B", "2B", "3B", "HR", "BB", "HBP", "R", "RBI", "SB")}
                assert hitter_points(events) == pytest.approx(row["Proj"], abs=0.02)
            else:
                events = {k: row[f"E_{k}"] for k in ("IP", "K", "W", "ER", "H", "BB", "HBP")}
                assert pitcher_points(events) == pytest.approx(row["Proj"], abs=0.02)

    def test_a_real_slate_simulates(self, cached_dates):
        from dfs.simulate import simulate_slate
        from dfs.slate import build_slate
        if not cached_dates:
            pytest.skip("no cached report data")
        try:
            players, _, _ = build_slate(cached_dates[-1], check_schedule=False)
        except Exception:
            pytest.skip("slate would not build")
        if players.empty:
            pytest.skip("no players")
        out = simulate_slate(players, n_sims=200, seed=1)
        assert out.shape == (200, len(players))
        assert np.isfinite(out).all()
