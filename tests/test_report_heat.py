import math

from scouting_report import (
    _HEAT_BAD_CUTOFF,
    _HEAT_BAD_RGB,
    _HEAT_GOOD_CUTOFF,
    _HEAT_GOOD_RGB,
    _HEAT_MID_RGB,
    _heat_rgb,
    _stat_percentile,
)


def test_heat_colors_saturate_beyond_the_good_and_bad_bounds():
    assert _heat_rgb(-10) == _HEAT_BAD_RGB
    assert _heat_rgb(_HEAT_BAD_CUTOFF) == _HEAT_BAD_RGB
    assert _heat_rgb(_HEAT_GOOD_CUTOFF) == _HEAT_GOOD_RGB
    assert _heat_rgb(10) == _HEAT_GOOD_RGB


def test_heat_colors_keep_a_differentiated_middle_spectrum():
    colors = [_heat_rgb(score) for score in (0.15, 0.30, 0.50, 0.70, 0.85)]
    assert colors[2] == _HEAT_MID_RGB
    assert len(set(colors)) == len(colors)


def test_heat_colors_ignore_missing_and_non_finite_values():
    assert _heat_rgb(None) is None
    assert _heat_rgb("not a number") is None
    assert _heat_rgb(math.nan) is None


def test_stat_reference_bounds_prevent_raw_outliers_from_rescaling_the_table():
    assert _stat_percentile("ops_off", 0.100) == 0.05
    assert _stat_percentile("ops_off", 2.000) == 0.95


def test_lower_is_better_metrics_keep_the_correct_direction():
    assert _stat_percentile("era", 2.50) > _stat_percentile("era", 6.00)
    assert _stat_percentile("k_pct_bat", 12.0) > _stat_percentile("k_pct_bat", 32.0)
