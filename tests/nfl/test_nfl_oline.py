"""The O-line projector: the draft curve, the movement penalty, and what is deliberately
*not* an adjustment."""

import numpy as np
import pandas as pd
import pytest

from nfl import oline


class TestDraftCurve:
    """Fitted on 68 rookie lineman seasons 2023-25: grade ~ -5.02 * log(pick) + 76.6."""

    def test_earlier_picks_grade_higher(self):
        assert oline._draft_grade(1) > oline._draft_grade(32) > oline._draft_grade(200)

    def test_it_is_fitted_on_log_pick_so_there_is_no_round_cliff(self):
        """The raw round table inverts -- round 2 grades below round 3 on this sample.

        Fitting on rounds would bake that small-n artefact in as a real step. On log(pick)
        the curve is smooth, so picks 32 and 33 differ by a hair rather than a round.
        """
        step_across_round_boundary = oline._draft_grade(32) - oline._draft_grade(33)
        step_early = oline._draft_grade(1) - oline._draft_grade(2)
        assert abs(step_across_round_boundary) < step_early

    def test_undrafted_is_a_slot_not_a_blank(self):
        """Going undrafted is information, and it is not neutral."""
        assert oline._draft_grade(None) == pytest.approx(
            oline._draft_grade(oline.UNDRAFTED_PICK))
        assert oline._draft_grade(None) < oline._draft_grade(260)

    def test_pick_zero_does_not_explode(self):
        assert np.isfinite(oline._draft_grade(0))


class TestRetention:
    """A grade is regressed toward the mean by its own year-over-year correlation."""

    def test_a_mover_is_regressed_roughly_twice_as_hard(self):
        """Measured: stayers +0.61/+0.60, movers +0.27/+0.38 across the two season pairs.

        So changing clubs is not a note to print beside an unchanged grade -- it halves how
        much of that grade should carry.
        """
        assert oline.MOVE_RETENTION < oline.STAY_RETENTION
        assert oline.MOVE_RETENTION == pytest.approx(oline.STAY_RETENTION / 2, rel=0.2)

    def test_retention_pulls_toward_the_mean_from_both_sides(self):
        mean = 60.0
        for retention in (oline.STAY_RETENTION, oline.MOVE_RETENTION):
            high = mean + retention * (90.0 - mean)
            low = mean + retention * (30.0 - mean)
            assert mean < high < 90.0
            assert 30.0 < low < mean


class TestTeamLineStrength:
    def _line(self, bases):
        return pd.DataFrame({
            "team": ["BUF"] * len(bases),
            "spot": list(oline.SPOTS)[:len(bases)],
            "proj_grade": [70.0] * len(bases),
            "proj_pass_block": [70.0] * len(bases),
            "proj_run_block": [70.0] * len(bases),
            "basis": bases,
        })

    def test_it_counts_how_each_starter_was_arrived_at(self):
        out = oline.team_line_strength(
            self._line(["grade", "grade", "grade (moved)", "draft capital", "replacement"]))
        row = out.loc["BUF"]
        assert row["continuity"] == 2
        assert row["newcomers"] == 1
        assert row["rookies"] == 1
        assert row["unknown"] == 1
        assert row["starters"] == 5

    def test_continuity_is_reported_but_never_applied(self):
        """Measured against the *change* in unit grade it is -0.16 and +0.01 -- nothing,
        with the sign flipping. Its raw +0.40 correlation is confounding: good lines keep
        their starters, and the grades already carry that.

        So two lines of identical projected grades must score identically no matter how
        different their continuity is.
        """
        stable = oline.team_line_strength(self._line(["grade"] * 5))
        churned = oline.team_line_strength(self._line(["grade (moved)"] * 5))
        assert stable.loc["BUF", "proj_grade"] == churned.loc["BUF", "proj_grade"]
        assert stable.loc["BUF", "continuity"] != churned.loc["BUF", "continuity"]


class TestSpotVocabulary:
    def test_the_five_spots_come_from_pff_snap_counts(self):
        """Rosters label all five `OL`, so the alignment has to come from snap counts.

        `ce` is PFF's spelling for centre -- reading it as anything else loses a fifth of
        every line.
        """
        assert set(oline.SPOT_COLUMNS.values()) == set(oline.SPOTS)
        assert oline.SPOT_COLUMNS["snap_counts_ce"] == "C"
