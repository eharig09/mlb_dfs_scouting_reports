"""Reliever availability in the report itself.

The back-to-back rule was measured before it was written: over the 2025 regular season —
281 relievers, 20,868 appearances, restricted to days the club played again the next day so
"did not pitch" is separable from "no game" — a reliever who worked on consecutive days
appeared the following day **5.5%** of the time (n=2,032) against **26.0%** on rest. Above 35
pitches across the two days it falls to 1.4%.

The three-day pitch total the report already used cannot see this: 15 pitches yesterday plus
12 the day before totals 27 and graded "Monitor", while the arm is effectively out.
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

import scouting_report as sr

CUTOFF = "2026-08-20"


def _log(appearances, cutoff=CUTOFF):
    """A statsapi-shaped game log: [(days_before_cutoff, pitches), ...]."""
    end = datetime.strptime(cutoff, "%Y-%m-%d")
    splits = [{"date": (end - timedelta(days=offset)).strftime("%Y-%m-%d"),
               "stat": {"numberOfPitches": pitches, "inningsPitched": "1.0"}}
              for offset, pitches in appearances]
    return {"stats": [{"splits": splits}]}


@pytest.fixture
def workload(monkeypatch):
    def run(appearances):
        monkeypatch.setattr(sr, "cached_json_request",
                            lambda *a, **k: _log(appearances))
        return sr._reliever_recent_workload(1, 2026, CUTOFF)
    return run


def _grade(w):
    """The report's own grading, mirrored so the test states the rule it is pinning."""
    if w.get("B2B") or w["Last3D_Pitches"] >= 45 or w["Last3D_G"] >= 3:
        return "Taxed"
    if w["Last3D_Pitches"] >= 25:
        return "Monitor"
    return "Available"


class TestBackToBack:
    def test_consecutive_days_into_tonight_are_flagged(self, workload):
        w = workload([(0, 10), (1, 11)])
        assert w["B2B"] is True
        assert w["Last2D_Pitches"] == 21

    def test_a_light_back_to_back_is_now_taxed_where_it_used_to_be_monitor(self, workload):
        """21 pitches over three days used to read Monitor. Measured: ~5% he appears."""
        w = workload([(0, 10), (1, 11)])
        assert w["Last3D_Pitches"] == 21          # under the old 25-pitch Monitor line
        assert _grade(w) == "Taxed"

    def test_consecutive_days_earlier_in_the_week_do_not_count(self, workload):
        """Back-to-back three days ago says nothing about whether he is available tonight."""
        w = workload([(3, 12), (4, 14)])
        assert w["B2B"] is False
        assert _grade(w) == "Available"

    def test_a_single_appearance_yesterday_is_not_back_to_back(self, workload):
        w = workload([(0, 25)])
        assert w["B2B"] is False
        assert _grade(w) == "Monitor"

    def test_two_days_ago_and_today_is_not_consecutive(self, workload):
        w = workload([(0, 12), (2, 12)])
        assert w["B2B"] is False


class TestExistingRulePreserved:
    """The measurement adds a case; it must not remove one.

    A single long outing is not back-to-back, so the new rule alone would have graded it
    Monitor. The report's three-day total is what catches it, and combining the two can only
    ever tighten a board.
    """

    def test_a_long_outing_two_days_ago_is_still_taxed(self, workload):
        w = workload([(2, 51)])
        assert w["B2B"] is False
        assert w["Last3D_Pitches"] == 51
        assert _grade(w) == "Taxed"

    def test_three_appearances_in_three_days_is_still_taxed(self, workload):
        w = workload([(0, 8), (1, 8), (2, 8)])
        assert _grade(w) == "Taxed"

    def test_a_rested_arm_is_still_available(self, workload):
        assert _grade(workload([(5, 9)])) == "Available"


class TestDownstreamContract:
    """New tiers would have been dropped silently by three existing consumers.

    `bullpen_deployment_notes` matches `isin(["Taxed", "Monitor"])`,
    `summarize_bullpen_form` counts those exact strings into fixed columns, and the
    available-only filter compares to "available". Adding "Out" or "Doubtful" would have
    left back-to-back arms uncounted in all three, so the rule reuses "Taxed".
    """

    def test_the_grade_vocabulary_is_unchanged(self, workload):
        grades = {_grade(workload(a)) for a in
                  ([(0, 10), (1, 11)], [(0, 25)], [(5, 9)], [(2, 51)])}
        assert grades <= {"Taxed", "Monitor", "Available"}

    def test_the_evidence_columns_are_additive(self, workload):
        w = workload([(0, 10), (1, 11)])
        for key in ("Last3D_G", "Last3D_Pitches", "Last3D_IP", "Last5D_Pitches",
                    "Last Outing", "L7 Usage", "L7 Pitches"):
            assert key in w, f"{key} was dropped"
        assert {"B2B", "Last2D_Pitches"} <= set(w)

    def test_summarise_still_counts_every_arm(self):
        """The form summary's columns are fixed; every arm has to land in one of them."""
        pen = pd.DataFrame({
            "Name": ["a", "b", "c"], "Throws": ["R", "L", "R"],
            "Availability": ["Taxed", "Monitor", "Available"],
            "Leverage Score": [10.0, 8.0, 6.0], "Last3D_Pitches": [30, 26, 4],
        })
        summary = sr.summarize_bullpen_form("TEX", pen)
        row = summary.iloc[0]
        assert int(row["Available"]) + int(row["Monitor"]) + int(row["Taxed"]) == len(pen)
