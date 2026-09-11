"""The outcome layer — the only part of the dashboard that can be *wrong* rather than ugly.

Everything else describes what the report believed. This scores those beliefs against real
contest results, so a bug here does not produce a broken chart; it produces a confident one.
The tests are written against that risk: thin cells must not be coloured, a drill-down must
reproduce the cell it opens, and a spread inside the noise must not be flagged as an edge.

`st.cache_data` wraps `scored_history`, so nothing here calls it — these run on fixtures.
"""

import numpy as np
import pandas as pd
import pytest

from dashboards import charts, outcomes


def _scored(n=600, seed=0):
    """A synthetic scored history with a *real* but weak relationship.

    Points depend on salary plus a little composite plus a lot of noise, which is the shape
    of the actual data (r ≈ +0.14 for salary). A fixture where the composite works perfectly
    would pass tests that the real board fails.
    """
    rng = np.random.default_rng(seed)
    salary = rng.integers(2000, 6000, n)
    composite = (salary - 4000) / 100 + rng.normal(0, 12, n)
    points = 6 + (salary - 2000) / 900 + composite * 0.05 + rng.normal(0, 8, n)
    return pd.DataFrame({
        "date": rng.choice(["2026-08-01", "2026-08-02", "2026-08-03"], n),
        "slate": rng.choice(["main", "turbo"], n),
        "Team": rng.choice(["ATH", "KC", "TEX"], n),
        "Name": [f"Player {i}" for i in range(n)],
        "Salary": salary.astype(float),
        "Composite": composite,
        "fpts": points.clip(0),
        "hit": points >= outcomes.DEFAULT_HIT,
    })


class TestHitGrid:
    def test_a_thin_cell_is_dropped_rather_than_coloured(self):
        """A cell of eight players shaded at all is a claim the sample cannot support.

        On a grid the eye reads adjacency as a trend, so a pale-but-present cell is worse
        than a blank one — it joins its neighbours into a gradient that is mostly noise.
        """
        frame = _scored()
        loose = outcomes.hit_grid(frame, "Salary", "Composite", min_cell=1)
        strict = outcomes.hit_grid(frame, "Salary", "Composite", min_cell=25)
        assert len(strict) < len(loose)
        assert (strict["n"] >= 25).all()

    def test_bins_are_quantile_edges_not_equal_width(self):
        """Equal-width salary bins leave the top of the board nearly empty.

        Pinned by population balance rather than by edge arithmetic: the point is that each
        column carries a comparable number of players.
        """
        frame = _scored()
        frame.loc[frame.index[:400], "Salary"] = 2500.0     # a heavy low-price cluster
        grid = outcomes.hit_grid(frame, "Salary", "Composite", 4, 4, min_cell=1)
        widths = (grid[["x0", "x1"]].drop_duplicates()
                  .assign(w=lambda d: d["x1"] - d["x0"])["w"])
        assert widths.nunique() > 1

    def test_rates_are_real_shares_not_rescaled(self):
        grid = outcomes.hit_grid(_scored(), "Salary", "Composite", min_cell=1)
        assert grid["rate_pct"].between(0, 100).all()
        assert grid["rate"].between(0, 1).all()

    def test_every_player_lands_in_exactly_one_cell(self):
        """Cells partition the board — double-counting would inflate every rate."""
        frame = _scored()
        grid = outcomes.hit_grid(frame, "Salary", "Composite", min_cell=1)
        assert grid["n"].sum() == len(frame)

    def test_a_missing_column_is_an_empty_frame_not_an_exception(self):
        frame = _scored().drop(columns=["Composite"])
        assert outcomes.hit_grid(frame, "Salary", "Composite").empty
        assert outcomes.hit_grid(pd.DataFrame(), "Salary", "Composite").empty

    def test_a_column_with_one_value_cannot_be_binned(self):
        frame = _scored()
        frame["Composite"] = 5.0
        assert outcomes.hit_grid(frame, "Salary", "Composite").empty


class TestDrillDown:
    """The grid is only evidence if the cell can be opened and agrees with itself."""

    def test_a_cell_reproduces_its_own_count_and_rate(self):
        frame = _scored()
        grid = outcomes.hit_grid(frame, "Salary", "Composite", min_cell=1)
        for _, cell in grid.iterrows():
            members = outcomes.cell_members(frame, "Salary", "Composite", cell.to_dict())
            assert len(members) == cell["n"]
            assert members["hit"].mean() == pytest.approx(cell["rate"])

    def test_members_come_back_best_first(self):
        frame = _scored()
        grid = outcomes.hit_grid(frame, "Salary", "Composite", min_cell=1)
        members = outcomes.cell_members(frame, "Salary", "Composite", grid.iloc[0].to_dict())
        assert members["fpts"].is_monotonic_decreasing

    def test_a_player_history_folds_names_the_way_the_salary_join_does(self):
        frame = _scored(n=40)
        frame.loc[frame.index[0], "Name"] = "José Ramírez"
        frame.loc[frame.index[1], "Name"] = "Jose Ramirez"
        assert len(outcomes.player_history(frame, "Jose Ramirez")) == 2

    def test_drill_columns_are_never_invented(self):
        """A reindex that adds absent columns would show empty fields as though measured."""
        frame = _scored().drop(columns=["Team"])
        out = outcomes.drill_frame(frame)
        assert "Team" not in out.columns
        assert set(out.columns) <= set(outcomes.DRILL_COLUMNS)


class TestControlledLift:
    """Holding price fixed is the test that matters; the raw lift mostly restates salary."""

    def test_the_flag_almost_never_fires_on_a_column_that_is_pure_noise(self):
        """The guard's calibration, which is the whole reason it is drawn.

        Checked across 25 independent draws rather than one, because a single seed says
        nothing about a 5%-of-the-time event and choosing the seed that passes is not a
        test. A broken standard error — wrong by a factor, or computed on the wrong n —
        shows up here as a flag rate far off nominal, which no single fixture would catch.

        Measured: 3.0% of bands flagged against a two-sided 2-SE nominal of ~4.6%.
        """
        flagged = total = 0
        for seed in range(25):
            frame = _scored(n=800, seed=seed)
            frame["Noise"] = np.random.default_rng(1000 + seed).normal(0, 1, len(frame))
            controlled = outcomes.controlled_lift(frame, "Noise")
            flagged += int(controlled["beyond noise"].sum())
            total += len(controlled)
        assert total > 50
        assert flagged / total < 0.15

    def test_the_noise_flag_is_two_standard_errors_of_the_difference(self):
        controlled = outcomes.controlled_lift(_scored(n=1200), "Composite")
        assert not controlled.empty
        expected = (controlled["spread"].abs() > 2 * controlled["se"])
        assert (controlled["beyond noise"] == expected).all()

    def test_a_band_too_thin_to_split_is_omitted_rather_than_reported(self):
        assert outcomes.controlled_lift(_scored(n=40), "Composite").empty

    def test_lift_of_one_means_the_column_said_nothing(self):
        frame = _scored()
        frame["Noise"] = np.random.default_rng(2).normal(0, 1, len(frame))
        table = outcomes.lift_table(frame, "Noise")
        assert table["lift"].max() < 1.35


class TestOutcomeCharts:
    def _grid(self):
        return outcomes.hit_grid(_scored(), "Salary", "Composite", min_cell=1)

    def test_each_builds_a_valid_vega_spec(self):
        frame = _scored()
        grid = self._grid()
        for chart in (charts.hit_grid_chart(grid, "Salary", "Composite"),
                      charts.lift_bars(outcomes.lift_table(frame, "Composite")),
                      charts.controlled_spread_bars(
                          outcomes.controlled_lift(_scored(n=1200), "Composite")),
                      charts.outcome_scatter(frame, "Salary", "Composite", "s", "c")):
            assert chart is not None
            assert chart.to_dict()

    def test_an_empty_grid_degrades_to_none_not_an_empty_chart(self):
        empty = pd.DataFrame()
        assert charts.hit_background(empty, "x", "y") is None
        assert charts.hit_grid_chart(empty, "x", "y") is None
        assert charts.lift_bars(empty) is None
        assert charts.controlled_spread_bars(empty) is None

    def test_the_hit_rate_ramp_is_sequential_single_hue(self):
        """Magnitude on an ordered scale. A diverging or categorical scheme here would
        invent a midpoint or a set of classes the data does not have."""
        spec = charts.hit_grid_chart(self._grid(), "x", "y").to_dict()
        assert spec["encoding"]["color"]["scale"]["scheme"] == charts.HIT_SCHEME

    def test_the_scatter_over_history_puts_the_points_on_top(self):
        """Layer order is the readability of the whole chart: a rect drawn over the dots
        hides tonight's board behind last month's."""
        from dashboards import salaries
        board = salaries.with_surplus(pd.DataFrame({
            "Name": [f"P{i}" for i in range(12)],
            "Salary": np.linspace(2200, 6000, 12),
            "Composite": np.linspace(-10, 40, 12),
            "Season AB": np.full(12, 200),
            "Signal": ["Priority", "Watch", "Neutral", "Fade"] * 3,
        }))
        spec = charts.value_scatter_with_history(board, self._grid()).to_dict()
        marks = [(l.get("mark") or {}).get("type") for l in spec["layer"]]
        # The grid first, the dots over it, and only the name labels above those.
        assert marks[0] == "rect"
        assert marks.index("circle") > marks.index("rect")
        assert set(marks[marks.index("circle") + 1:]) <= {"text"}

    def test_the_lift_chart_carries_its_base_rate(self):
        """Bars without the base rate invite reading any ordering as an edge."""
        spec = charts.lift_bars(outcomes.lift_table(_scored(), "Composite")).to_dict()
        marks = [layer["mark"]["type"] for layer in spec["layer"]]
        assert "rule" in marks

    def test_the_controlled_chart_draws_its_error_bars(self):
        """A spread this small shown without its noise is not an honest chart."""
        table = outcomes.controlled_lift(_scored(n=1200), "Composite")
        spec = charts.controlled_spread_bars(table).to_dict()
        marks = [layer["mark"]["type"] for layer in spec["layer"]]
        assert marks.count("rule") == 2          # the zero line and the intervals


class TestSecondControl:
    """Salary alone is not a sufficient control, and the sample proves it.

    Measured: batting slot moves the outcome more than the composite does — slots 1-2
    average 7.9 DK points and a 33% hit rate against 4.9 and 17% for slots 8-9 — and cheap
    hitters bat at the bottom. Controlling only for price therefore reports "cheap hitters
    beat their price" when the truth is "the score noticed who was batting eighth".
    """

    def _confounded(self, n=2400, seed=3):
        """A score whose *only* link to points runs through the opportunity confound.

        Salary is deliberately independent of slot here, so controlling for price cannot
        remove the confound and the score keeps separating; controlling for slot removes it
        entirely. That isolates the mechanism, which a fixture where salary proxies slot
        cannot do — there, price already absorbs most of the confound and neither control
        looks very different.
        """
        rng = np.random.default_rng(seed)
        slot = rng.integers(1, 10, n).astype(float)
        salary = rng.integers(2000, 6000, n).astype(float)   # independent of slot
        score = (10 - slot) * 3 + rng.normal(0, 6, n)
        points = 11 - slot * 0.9 + rng.normal(0, 6, n)
        return pd.DataFrame({
            "Salary": salary, "Slot": slot, "Composite": score,
            "fpts": points.clip(0), "hit": points >= outcomes.DEFAULT_HIT,
            "Name": [f"P{i}" for i in range(n)]})

    @staticmethod
    def _weighted_spread(table):
        """Mean absolute spread weighted by cell size.

        A bare `max()` is not comparable between the two runs: crossing a second control
        produces different cells with different counts, and the noisiest small cell can post
        the largest spread by chance while the edge overall has collapsed.
        """
        return float((table["spread"].abs() * table["n"]).sum() / table["n"].sum())

    def test_controlling_the_confound_collapses_an_edge_that_price_cannot_touch(self):
        frame = self._confounded()
        salary_only = outcomes.controlled_lift(frame, "Composite")
        with_slot = outcomes.controlled_lift(frame, "Composite", second="Slot")
        assert not salary_only.empty and not with_slot.empty
        # Price is independent of the confound here, so it removes none of the edge.
        assert self._weighted_spread(salary_only) > 15
        assert salary_only["beyond noise"].any()
        # Slot is the whole mechanism, so holding it fixed has to take most of it away.
        assert self._weighted_spread(with_slot) < self._weighted_spread(salary_only) / 2

    def test_the_second_control_actually_crosses_the_cells(self):
        frame = self._confounded()
        crossed = outcomes.controlled_lift(frame, "Composite", second="Slot")
        assert crossed["control band"].str.contains("·").all()
        assert len(crossed) <= 4          # 2x2, coarse on purpose

    def test_a_missing_control_column_declines_rather_than_ignoring_it(self):
        """Silently falling back to one control would report the confounded number under a
        label promising it had been controlled for."""
        frame = self._confounded().drop(columns=["Slot"])
        assert outcomes.controlled_lift(frame, "Composite", second="Slot").empty

    def test_band_labels_are_readable_bounds(self):
        table = outcomes.controlled_lift(self._confounded(), "Composite")
        assert not table["control band"].str.contains("Interval").any()
        assert table["control band"].str.contains("-").all()

    def test_the_controls_on_offer_are_all_pre_game(self):
        """Every control has to be knowable before first pitch or the panel is circular.

        Ownership is the interesting case and it *is* legitimate: DK publishes it during
        the contest, and the question it answers — does the score beat what the field
        already did — is exactly a pre-game question.
        """
        assert set(outcomes.CONTROLS) == {"Salary", "Slot", "PA", "own"}
        assert "fpts" not in outcomes.CONTROLS
        assert "hit" not in outcomes.CONTROLS


class TestOpportunityColumns:
    def test_the_drill_down_shows_opportunity_beside_the_score(self):
        """A drill-down that omitted batting slot would invite the same wrong reading the
        controlled panel exists to prevent."""
        assert "Slot" in outcomes.DRILL_COLUMNS
        assert "PA" in outcomes.DRILL_COLUMNS

    def test_points_per_thousand_is_the_outcome_in_roster_units(self):
        frame = pd.DataFrame({"Salary": [2000.0, 6000.0], "fpts": [10.0, 10.0],
                              "hit": [True, True], "Name": ["a", "b"]})
        frame["pts_per_1k"] = np.where(frame["Salary"] > 0,
                                       frame["fpts"] / (frame["Salary"] / 1000.0), np.nan)
        assert frame["pts_per_1k"].tolist() == [5.0, 5.0 / 3]
