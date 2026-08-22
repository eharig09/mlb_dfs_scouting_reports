"""Walk-forward evaluation and the calibration metrics.

The metrics are tested against distributions whose answers are known analytically, because
a scoring rule that is subtly wrong reports a subtly wrong model and there is nothing in the
output to catch it. The walk-forward machinery is tested for the one property it exists to
provide: no target date may be scored using anything from that date or later.
"""

import numpy as np
import pandas as pd
import pytest

from dfs.evaluate import (CRPS_GRID, REPORT_QUANTILES, ZERO_SHARE_OF_BUST,
                          apply_bands, brier, crps, evaluate, fit_bands, implied_quantiles,
                          metrics, pinball, pit_values, reliability, walk_forward)


class TestPinball:
    def test_zero_when_perfect(self):
        assert pinball(np.array([5.0]), np.array([5.0]), 0.5)[0] == pytest.approx(0.0)

    def test_penalises_under_and_over_prediction_asymmetrically(self):
        # At the 90th percentile, being too low should hurt nine times as much.
        low = pinball(np.array([10.0]), np.array([9.0]), 0.90)[0]
        high = pinball(np.array([8.0]), np.array([9.0]), 0.90)[0]
        assert low == pytest.approx(0.9)
        assert high == pytest.approx(0.1)
        assert low == pytest.approx(9 * high)

    def test_median_loss_is_symmetric(self):
        assert pinball(np.array([12.0]), np.array([10.0]), 0.5)[0] == \
            pytest.approx(pinball(np.array([8.0]), np.array([10.0]), 0.5)[0])

    def test_is_minimised_at_the_true_quantile(self):
        rng = np.random.default_rng(0)
        sample = rng.normal(10, 3, 20000)
        truth = np.quantile(sample, 0.75)
        best = pinball(sample, np.full_like(sample, truth), 0.75).mean()
        for offset in (-1.0, -0.4, 0.4, 1.0):
            worse = pinball(sample, np.full_like(sample, truth + offset), 0.75).mean()
            assert worse > best


class TestImpliedQuantiles:
    def test_is_monotone(self):
        grid = implied_quantiles([8.0], [2.0], [22.0], [0.35], "H", CRPS_GRID)
        assert np.all(np.diff(grid[0]) >= -1e-9)

    def test_stays_monotone_when_bands_conflict(self):
        """At high projections the bust floor can put the 3-point knot above the 25th."""
        grid = implied_quantiles([30.0], [20.0], [45.0], [0.32], "H", CRPS_GRID)
        assert np.all(np.diff(grid[0]) >= -1e-9)

    def test_hitters_cannot_go_negative(self):
        grid = implied_quantiles([6.0], [1.0], [18.0], [0.40], "H", CRPS_GRID)
        assert grid.min() >= 0.0

    def test_pitchers_can_go_negative(self):
        grid = implied_quantiles([12.0], [6.0], [26.0], [0.15], "P", CRPS_GRID)
        assert grid.min() < 0.0

    def test_the_stated_bands_are_reproduced(self):
        floor, ceiling = 2.5, 21.0
        grid = implied_quantiles([8.0], [floor], [ceiling], [0.36], "H",
                                 np.array([0.25, 0.90]))
        assert grid[0, 0] == pytest.approx(floor, abs=0.01)
        assert grid[0, 1] == pytest.approx(ceiling, abs=0.01)

    def test_a_zero_atom_exists_for_hitters(self):
        """DK hitter scoring has a hard atom at zero; without it PIT is unreadable."""
        levels = np.array([0.001, 0.05, 0.10])
        grid = implied_quantiles([8.0], [2.0], [22.0], [0.36], "H", levels)
        assert grid[0, 0] == pytest.approx(0.0)
        assert ZERO_SHARE_OF_BUST["H"] > 0
        assert ZERO_SHARE_OF_BUST["P"] == 0

    def test_vectorises_over_players(self):
        grid = implied_quantiles([5.0, 9.0, 14.0], [1.0, 3.0, 6.0], [15.0, 22.0, 30.0],
                                 [0.5, 0.36, 0.25], "H", REPORT_QUANTILES)
        assert grid.shape == (3, len(REPORT_QUANTILES))
        # Compared from the median up. Below it the zero atom is wider for a bustier player
        # and the monotone fix-up lifts his low quantiles, so a better player can sit lower
        # at the 10th -- which is correct, not a defect: he is likelier to blank outright.
        median_up = slice(2, None)
        assert (grid[2][median_up] >= grid[0][median_up]).all()
        assert (grid[1][median_up] >= grid[0][median_up]).all()


class TestCRPS:
    def test_zero_for_a_perfect_point_forecast(self):
        grid = np.zeros((1, len(CRPS_GRID))) + 7.0
        assert crps(np.array([7.0]), grid)[0] == pytest.approx(0.0, abs=1e-9)

    def test_a_sharp_correct_forecast_beats_a_vague_one(self):
        actual = np.array([10.0])
        sharp = np.linspace(9.0, 11.0, len(CRPS_GRID))[None, :]
        vague = np.linspace(0.0, 20.0, len(CRPS_GRID))[None, :]
        assert crps(actual, sharp)[0] < crps(actual, vague)[0]

    def test_a_sharp_wrong_forecast_loses_to_a_vague_one(self):
        actual = np.array([30.0])
        sharp = np.linspace(9.0, 11.0, len(CRPS_GRID))[None, :]
        vague = np.linspace(0.0, 40.0, len(CRPS_GRID))[None, :]
        assert crps(actual, sharp)[0] > crps(actual, vague)[0]

    def test_matches_the_closed_form_for_a_uniform(self):
        """For U(0,1) with y=0.5, CRPS integrates to 1/12."""
        grid = CRPS_GRID[None, :].copy()
        assert crps(np.array([0.5]), grid)[0] == pytest.approx(1 / 12, abs=2e-3)


class TestPIT:
    def test_uniform_when_the_forecast_is_right(self):
        rng = np.random.default_rng(1)
        n = 4000
        actual = rng.normal(10, 3, n)
        grid = np.tile(10 + 3 * np.sqrt(2) * _erfinv(2 * CRPS_GRID - 1), (n, 1))
        pit = pit_values(actual, grid, CRPS_GRID, rng=rng)
        assert pit.mean() == pytest.approx(0.5, abs=0.03)
        assert pit.std() == pytest.approx(1 / np.sqrt(12), abs=0.03)

    def test_too_narrow_a_forecast_pushes_mass_to_the_edges(self):
        rng = np.random.default_rng(2)
        n = 3000
        actual = rng.normal(10, 6, n)
        grid = np.tile(10 + 1.0 * np.sqrt(2) * _erfinv(2 * CRPS_GRID - 1), (n, 1))
        pit = pit_values(actual, grid, CRPS_GRID, rng=rng)
        edges = ((pit < 0.1) | (pit > 0.9)).mean()
        assert edges > 0.4, "an over-confident forecast must show as U-shaped PIT"

    def test_a_biased_forecast_shifts_the_mean(self):
        rng = np.random.default_rng(3)
        n = 2000
        actual = rng.normal(14, 3, n)
        grid = np.tile(10 + 3 * np.sqrt(2) * _erfinv(2 * CRPS_GRID - 1), (n, 1))
        assert pit_values(actual, grid, CRPS_GRID, rng=rng).mean() > 0.6

    def test_stays_inside_the_unit_interval(self):
        rng = np.random.default_rng(4)
        grid = np.tile(np.linspace(0, 30, len(CRPS_GRID)), (50, 1))
        pit = pit_values(rng.uniform(-5, 40, 50), grid, CRPS_GRID, rng=rng)
        assert (pit >= 0).all() and (pit <= 1).all()


def _erfinv(x):
    from scipy.special import erfinv
    return erfinv(x)


class TestBrier:
    def test_perfect_forecasts_score_zero(self):
        assert brier([1.0, 0.0, 1.0], [1, 0, 1]) == pytest.approx(0.0)

    def test_worst_case_is_one(self):
        assert brier([0.0, 1.0], [1, 0]) == pytest.approx(1.0)

    def test_a_climatological_forecast_scores_the_variance(self):
        outcomes = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        base = outcomes.mean()
        assert brier(np.full(10, base), outcomes) == pytest.approx(base * (1 - base))

    def test_reliability_bins_are_ordered(self):
        rng = np.random.default_rng(5)
        p = rng.uniform(0, 1, 2000)
        y = (rng.uniform(0, 1, 2000) < p).astype(int)
        table = reliability(p, y, bins=5)
        assert len(table) == 5
        assert table["observed"].is_monotonic_increasing
        assert (table["gap"].abs() < 0.10).all()


class TestBandFitting:
    def _history(self, n=800, seed=1):
        rng = np.random.default_rng(seed)
        proj = rng.uniform(2, 18, n)
        actual = np.maximum(0.0, rng.gamma(1.2, proj / 1.2))
        return pd.DataFrame({"Type": "H", "Played": True, "Proj": proj, "Actual": actual})

    def test_fits_when_there_is_enough_history(self):
        fits = fit_bands(self._history())
        assert fits["H"]["fitted"]
        assert "slope" in fits["H"]["floor"]

    def test_falls_back_to_the_shipped_constants_when_thin(self):
        from dfs.projections import FLOOR_FIT
        fits = fit_bands(self._history(n=20))
        assert not fits["H"]["fitted"]
        assert fits["H"]["floor"] == FLOOR_FIT["H"]

    def test_applying_bands_replaces_floor_and_bust(self):
        frame = pd.DataFrame({"Type": ["H", "H"], "Proj": [5.0, 15.0],
                              "Floor": [0.0, 0.0], "Bust%": [50.0, 50.0]})
        fits = {"H": {"floor": {"slope": 0.9, "intercept": -4.0},
                      "bust": {"slope": -0.05, "intercept": 0.9}, "n": 999, "fitted": True}}
        out = apply_bands(frame, fits)
        assert out["Floor"].tolist() == [pytest.approx(0.5), pytest.approx(9.5)]
        assert out["Bust%"].tolist() == [pytest.approx(65.0), pytest.approx(15.0)]

    def test_bust_probability_is_clipped_into_range(self):
        frame = pd.DataFrame({"Type": ["H"], "Proj": [90.0], "Floor": [0.0], "Bust%": [50.0]})
        fits = {"H": {"floor": {"slope": 1.0, "intercept": 0.0},
                      "bust": {"slope": -0.05, "intercept": 0.9}, "n": 999, "fitted": True}}
        assert 0 < apply_bands(frame, fits)["Bust%"].iloc[0] <= 95.0


class TestWalkForward:
    def _frame(self, dates, per_date=300, seed=1):
        rng = np.random.default_rng(seed)
        rows = []
        for date in dates:
            for i in range(per_date):
                proj = rng.uniform(2, 18)
                rows.append({"Date": date, "Type": "H", "Played": True, "Proj": proj,
                             "Ceiling": proj * 2.4, "Floor": max(0.0, proj * 0.8 - 4),
                             "Bust%": 40.0,
                             "Actual": max(0.0, rng.gamma(1.2, proj / 1.2))})
        return pd.DataFrame(rows)

    def test_early_dates_are_dropped_for_want_of_history(self):
        frame = self._frame(["2026-07-01", "2026-07-02", "2026-07-03"], per_date=150)
        scored, schedule = walk_forward(frame, min_train_rows=200)
        assert sorted(scored["Date"].unique()) == ["2026-07-03"]
        assert schedule[0]["train_rows"] >= 200

    def test_no_target_date_informs_its_own_calibration(self):
        """The property the whole module exists for."""
        frame = self._frame(["2026-07-01", "2026-07-02", "2026-07-03", "2026-07-04"])
        _, schedule = walk_forward(frame, min_train_rows=200)
        dates = sorted(frame["Date"].unique())
        for entry in schedule:
            # train_dates counts dates strictly before the target.
            assert entry["train_dates"] == dates.index(entry["date"])
            assert entry["train_rows"] == 300 * dates.index(entry["date"])

    def test_recalibration_actually_changes_the_bands(self):
        frame = self._frame(["2026-07-01", "2026-07-02", "2026-07-03"])
        with_fit, _ = walk_forward(frame, min_train_rows=200, recalibrate=True)
        without, _ = walk_forward(frame, min_train_rows=200, recalibrate=False)
        last = "2026-07-03"
        a = with_fit[with_fit["Date"] == last]["Floor"].to_numpy()
        b = without[without["Date"] == last]["Floor"].to_numpy()
        assert not np.allclose(a, b)

    def test_no_recalibration_keeps_every_date(self):
        frame = self._frame(["2026-07-01", "2026-07-02"], per_date=50)
        scored, schedule = walk_forward(frame, recalibrate=False)
        assert sorted(scored["Date"].unique()) == ["2026-07-01", "2026-07-02"]
        assert schedule == []

    def test_dates_are_processed_in_order_not_file_order(self):
        frame = self._frame(["2026-07-03", "2026-07-01", "2026-07-02"])
        _, schedule = walk_forward(frame, min_train_rows=200)
        assert [e["date"] for e in schedule] == sorted(e["date"] for e in schedule)


class TestMetrics:
    def _played(self, n=400, seed=2):
        rng = np.random.default_rng(seed)
        proj = rng.uniform(3, 16, n)
        return pd.DataFrame({
            "Type": "H", "Played": True, "Proj": proj,
            "Ceiling": proj + 12, "Floor": np.maximum(0, proj - 5), "Bust%": 40.0,
            "Actual": np.maximum(0.0, rng.gamma(1.3, proj / 1.3)),
        })

    def test_all_families_are_reported(self):
        out = metrics(self._played())
        for key in ("bias", "mae", "rmse", "spearman", "top_decile_recall", "crps",
                    "pit_mean", "pit_ks", "bust_brier", "ceiling_exceeded",
                    "pinball_10", "pinball_90", "quintile_spread"):
            assert key in out, key

    def test_small_samples_report_n_and_stop(self):
        out = metrics(self._played(n=3))
        assert out["n"] == 3
        assert "crps" not in out

    def test_scratches_are_counted_not_scored(self):
        frame = self._played(n=100)
        frame.loc[:9, "Played"] = False
        frame.loc[:9, "Actual"] = None
        out = metrics(frame)
        assert out["n"] == 90
        assert out["n_projected"] == 100
        assert out["scratch_rate"] == pytest.approx(0.10)

    def test_bias_sign_is_actual_minus_projected(self):
        frame = pd.DataFrame({"Type": "H", "Played": True, "Proj": [10.0] * 50,
                              "Ceiling": [22.0] * 50, "Floor": [4.0] * 50,
                              "Bust%": [35.0] * 50, "Actual": [14.0] * 50})
        assert metrics(frame)["bias"] == pytest.approx(4.0)

    def test_perfect_ranking_gives_full_recall(self):
        n = 200
        proj = np.linspace(1, 20, n)
        frame = pd.DataFrame({"Type": "H", "Played": True, "Proj": proj,
                              "Ceiling": proj + 10, "Floor": proj - 1,
                              "Bust%": 30.0, "Actual": proj})
        assert metrics(frame)["top_decile_recall"] == pytest.approx(1.0)
        assert metrics(frame)["spearman"] == pytest.approx(1.0)

    def test_evaluate_splits_by_type_and_segment(self, slate):
        frame = slate.copy()
        frame["Played"] = True
        frame["Actual"] = frame["Proj"] * 1.1
        frame["Date"] = "2026-08-01"
        frame["Salary Tier"] = "mid"
        frame["Order"] = frame["Slot"]
        frame["Handedness"] = "RHB vs RHP"
        frame["Slate Size"] = "medium (5-9)"
        frame["Lineup Status"] = "confirmed"
        report = evaluate(frame, min_segment=5)
        assert set(report["overall"]) == {"hitters", "pitchers"}
        assert "batting order" in report["segments"]
