"""The split line shared by the Team Hitting Summary and Opposing Pitching Allowed tables.

Both tables are built from `_split_metrics`, so the two can be read against each other row
by row -- the lineup's K% directly above the K% that staff has allowed. That only holds if
there is one implementation, which is why these tests pin the helper rather than either
table.

The park adjustment is the part most likely to be got wrong later: a park changes how far
the ball carries, not whether the hitter made contact, so it must touch the run-scoring
rates and leave the strikeout and walk rates alone.
"""

import pandas as pd
import pytest

from scouting_report import SPLIT_STAT_COLUMNS, _park_adjusted_split, _split_metrics


def pa_rows(events, **columns):
    """A frame of PA-ending statcast rows, one per event."""
    frame = pd.DataFrame({"events": events})
    for name, values in columns.items():
        frame[name] = values
    return frame


class TestSplitMetrics:
    def test_reports_every_advertised_column(self):
        metrics = _split_metrics(pa_rows(["single", "strikeout", "walk"]))
        assert set(SPLIT_STAT_COLUMNS) <= set(metrics)

    def test_empty_frame_is_zeroed_not_crashing(self):
        metrics = _split_metrics(pd.DataFrame({"events": []}))
        assert metrics["PA"] == 0 and metrics["K%"] == 0.0

    def test_k_rate_is_per_plate_appearance(self):
        metrics = _split_metrics(pa_rows(["strikeout"] * 3 + ["single"] * 7))
        assert metrics["PA"] == 10
        assert metrics["K%"] == pytest.approx(30.0)

    def test_k_rate_counts_the_strikeout_double_play(self):
        """`strikeout_double_play` is a strikeout; missing it undercounts every K rate."""
        metrics = _split_metrics(pa_rows(["strikeout_double_play"] + ["single"] * 9))
        assert metrics["K%"] == pytest.approx(10.0)

    def test_walk_rate_includes_intentional_walks(self):
        metrics = _split_metrics(pa_rows(["walk", "intent_walk"] + ["single"] * 8))
        assert metrics["BB%"] == pytest.approx(20.0)

    def test_walks_are_plate_appearances_but_not_at_bats(self):
        """ISO is per at-bat; counting walks as at-bats would deflate it."""
        walked = _split_metrics(pa_rows(["home_run", "walk"]))
        plain = _split_metrics(pa_rows(["home_run", "field_out"]))
        assert walked["ISO"] > plain["ISO"]

    def test_iso_is_slugging_minus_average(self):
        # Four at-bats, one home run: SLG 1.000, AVG .250, ISO .750.
        metrics = _split_metrics(pa_rows(["home_run"] + ["field_out"] * 3))
        assert metrics["ISO"] == pytest.approx(0.750, abs=0.001)

    def test_iso_is_zero_for_a_singles_only_line(self):
        metrics = _split_metrics(pa_rows(["single"] * 4 + ["field_out"] * 6))
        assert metrics["ISO"] == pytest.approx(0.0)

    def test_hr_is_a_rate_not_a_count(self):
        """Splits carry very different PA totals, so the count is not comparable."""
        small = _split_metrics(pa_rows(["home_run"] + ["field_out"] * 9))
        big = _split_metrics(pa_rows(["home_run"] * 2 + ["field_out"] * 98))
        assert small["HR%"] == pytest.approx(10.0)
        assert big["HR%"] == pytest.approx(2.0)
        assert small["HR%"] > big["HR%"]          # despite half the home runs

    def test_no_whiff_column(self):
        """These frames are PA-ending pitches only; a whiff rate over them would mislead."""
        assert "Whiff%" not in _split_metrics(pa_rows(["single", "strikeout"]))


class TestParkAdjustment:
    def test_a_hitters_park_pulls_the_run_rates_down(self):
        frame = pa_rows(["home_run", "single", "field_out", "field_out"])
        raw = _split_metrics(frame)
        adjusted = _park_adjusted_split(frame, 1.20)
        assert adjusted["OPS"] < raw["OPS"]
        assert adjusted["ISO"] < raw["ISO"]

    def test_a_pitchers_park_pushes_them_up(self):
        frame = pa_rows(["home_run", "single", "field_out", "field_out"])
        raw = _split_metrics(frame)
        adjusted = _park_adjusted_split(frame, 0.85)
        assert adjusted["OPS"] > raw["OPS"]

    def test_it_leaves_the_plate_discipline_rates_alone(self):
        """A park does not change whether a hitter struck out."""
        frame = pa_rows(["strikeout", "walk", "single", "field_out"])
        raw = _split_metrics(frame)
        adjusted = _park_adjusted_split(frame, 1.20)
        assert adjusted["K%"] == raw["K%"]
        assert adjusted["BB%"] == raw["BB%"]
        assert adjusted["HardHit%"] == raw["HardHit%"]

    def test_a_neutral_park_changes_nothing(self):
        frame = pa_rows(["home_run", "single", "field_out"])
        assert _park_adjusted_split(frame, 1.0)["OPS"] == pytest.approx(
            _split_metrics(frame)["OPS"], abs=0.002)
