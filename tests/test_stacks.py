"""Clubs aggregated as stack units.

The risk here is subtle: a stack table that looks right but defines a stack differently from
the optimiser. Then the dashboard recommends one thing and the lineup builder builds another,
and nothing errors.
"""

import numpy as np
import pandas as pd
import pytest

from dashboards import charts, scales, stacks


def _projected(n_teams=3):
    rows = []
    for team_index, team in enumerate(["ATH", "KC", "TEX"][:n_teams]):
        for slot in range(1, 10):
            rows.append({
                "Team": team, "Type": "H", "Slot": slot,
                "Name": f"{team} bat {slot}", "Pos": "OF", "Bats": "R",
                # Projection deliberately *falls* down the order, so a "top five by
                # projection" implementation and a "top five by batting order" one would
                # agree — and then rises again at the bottom, so they disagree.
                "Proj": 9.0 - slot * 0.4 + (3.0 if slot == 9 else 0.0),
                "Ceiling": 20.0 - slot * 0.8 + (6.0 if slot == 9 else 0.0),
                "Opp": "OPP", "Opp SP": "Some Arm", "Opp SP Hand": "R",
                "Team Runs": 4.5 + team_index * 0.2, "Matchup": 1.0,
            })
        rows.append({"Team": team, "Type": "P", "Slot": np.nan, "Name": f"{team} arm",
                     "Proj": 14.0, "Ceiling": 30.0, "Opp": "OPP", "Opp SP": "",
                     "Team Runs": 4.5, "Matchup": 1.0})
    return pd.DataFrame(rows)


class TestStackDefinition:
    """`dfs.slate.build_stacks` owns the definition; this pins that it is what gets used."""

    def test_the_five_are_the_top_of_the_order_not_the_best_projections(self):
        """A stack is contiguous in the lineup because that is what makes outcomes
        correlate. The ninth hitter in this fixture out-projects the third; picking by
        projection would take him and describe a roster whose correlation nobody can build.
        """
        from dfs import slate as slate_module
        frame = _projected()
        frame["Salary"] = 3000
        built = slate_module.build_stacks(frame, True)
        hitters = frame[frame["Type"] == "H"]
        ath = hitters[hitters["Team"] == "ATH"].sort_values("Slot")
        expected = ath.head(5)["Proj"].sum()
        assert built[built["Team"] == "ATH"]["Top5 Proj"].iloc[0] == pytest.approx(
            round(expected, 1))
        # and the ninth hitter, despite projecting higher than several in the stack, is out
        assert ath.iloc[8]["Proj"] > ath.iloc[2]["Proj"]

    def test_pitchers_are_not_part_of_a_stack(self):
        from dfs import slate as slate_module
        frame = _projected()
        frame["Salary"] = 3000
        built = slate_module.build_stacks(frame, True)
        assert (built["Hitters"] == 9).all()

    def test_the_dashboard_does_not_reimplement_it(self):
        """A second definition would drift from the optimiser's, and the two would then
        disagree about what a stack is while both looking correct."""
        import inspect
        source = inspect.getsource(stacks.team_stacks)
        assert "build_stacks" in source


class TestSalaryJoin:
    def test_price_reaches_both_the_table_and_the_members(self, monkeypatch):
        """`project_game` carries no price. Without the join `build_stacks` returns a null
        salary and a null value, and the page renders blanks where the ranking belongs —
        a failure that looks like missing data rather than a missing join."""
        import inspect
        assert "_with_salary" in inspect.getsource(stacks.team_stacks)
        assert "_with_salary" in inspect.getsource(stacks.stack_members)

    def test_a_frame_with_no_prices_still_gets_the_column(self):
        frame = stacks._with_salary(_projected(), "1999-01-01")
        assert "Salary" in frame.columns


class TestStackCharts:
    def _stacks(self):
        return pd.DataFrame({
            "Team": ["ATH", "KC", "TEX"], "Opp": ["KC", "ATH", "WSH"],
            "Opp SP": ["A", "B", "C"], "Opp SP Hand": ["R", "L", "R"],
            "Hitters": [9, 9, 9],
            "Top5 Salary": [17700, 21000, 20400],
            "Top5 Proj": [36.4, 40.7, 39.5],
            "Top5 Ceiling": [82.8, 94.6, 91.2],
            "Stack Value": [4.68, 4.50, 4.47],
            "Stack Score": [60.8, 82.2, 74.2],
            "Team Runs": [4.3, 5.5, 5.2],
            "Allowed OPS": [0.68, 0.70, 0.78],
            "Lineup OPS": [0.70, 0.72, 0.74],
            "park": ["Kauffman", "Great American", "Globe Life"],
            "hr_env": [0.88, 1.25, 1.00],
        })

    def test_every_axis_pairing_builds(self):
        frame = self._stacks()
        available = [k for k in charts.STACK_AXES if k in frame.columns]
        for x in available:
            for y in available:
                chart = charts.stack_scatter(frame, x, y)
                assert chart is not None and chart.to_dict()

    def test_clubs_are_named_on_the_chart_not_only_in_a_legend(self):
        """Thirty clubs is past what a colour legend resolves but well inside what a chart
        can label, and a named mark removes the decoding step entirely."""
        spec = charts.stack_scatter(self._stacks()).to_dict()
        marks = [layer["mark"]["type"] for layer in spec["layer"]]
        assert "text" in marks
        assert spec["layer"][0]["encoding"]["color"]["legend"] is None

    def test_labels_can_be_turned_off_without_breaking_the_chart(self):
        assert charts.stack_scatter(self._stacks(), labels=False).to_dict()

    def test_clubs_keep_their_own_colours(self):
        spec = charts.stack_scatter(self._stacks()).to_dict()
        scale = spec["layer"][0]["encoding"]["color"]["scale"]
        assert scale["range"] == [charts.TEAM_COLORS[t] for t in scale["domain"]]

    def test_focus_spends_the_salary_axis_on_real_stack_costs(self):
        frame = self._stacks()
        frame.loc[0, "Top5 Salary"] = 0  # missing salary join, not a free stack
        anchored = scales.anchor(frame, frame, domains=scales.STACK_FOCUS_DOMAINS)
        spec = charts.stack_scatter(anchored).to_dict()
        assert spec["layer"][0]["encoding"]["x"]["scale"]["domain"] == [15000.0, 27000.0]

    def test_full_salary_range_ignores_the_missing_price_sentinel(self):
        frame = self._stacks()
        frame.loc[0, "Top5 Salary"] = 0
        anchored = scales.anchor(frame, frame, mode="full",
                                 domains=scales.STACK_FOCUS_DOMAINS)
        assert scales.of(anchored, "Top5 Salary")[0] > 0

    def test_the_ranked_bars_build_for_each_measure(self):
        for column in ("Stack Score", "Top5 Ceiling", "Stack Value", "Team Runs"):
            assert charts.stack_bars(self._stacks(), column).to_dict()

    def test_member_bars_separate_the_five_from_the_rest(self):
        members = pd.DataFrame({
            "Slot": [1, 2, 3, 4, 5, 6], "Name": list("abcdef"),
            "Pos": ["OF"] * 6, "Bats": ["R"] * 6, "Salary": [3000] * 6,
            "Proj": [8.4, 9.1, 7.6, 9.0, 8.3, 6.9], "Ceiling": [19.0] * 6,
            "PA": [4.5] * 6, "In stack": [True] * 5 + [False],
        })
        spec = charts.stack_member_bars(members).to_dict()
        assert spec["encoding"]["color"]["scale"]["domain"] == ["Top five",
                                                                "Rest of card"]

    def test_empty_input_degrades_to_none(self):
        empty = self._stacks().iloc[0:0]
        assert charts.stack_scatter(empty) is None
        assert charts.stack_bars(empty) is None
        assert charts.stack_member_bars(pd.DataFrame()) is None
