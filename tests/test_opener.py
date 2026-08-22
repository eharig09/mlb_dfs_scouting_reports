"""Opener detection: who started, who followed, and how long each really went.

Three things were wrong and each failed quietly, because a mis-called opener just produces
a slightly odd-looking report rather than an error:

* innings were counted as *distinct innings appeared in*, so 2.1 IP read as three innings
  and an arm who got one out either side of a break read as two;
* the follower was picked by sorting on inning alone, with a non-stable sort, so two
  relievers in the same inning were ordered arbitrarily;
* exhibition games were in the index, where every arm is on a two-inning leash.

Measured against 2026 box scores, those cost 24 false-positive openers, 6 false negatives
and the wrong follower in 3% of half-games. See research/arsenal/opener_audit.py.
"""

import pandas as pd
import pytest

import scouting_report as sr
from scouting_report import (BULK_MIN_INNINGS, OPENER_BULK_REPEAT_RATE,
                             OPENER_MAX_INNINGS, build_opener_index)


@pytest.fixture(autouse=True)
def clear_index_cache():
    sr._OPENER_INDEX_CACHE.clear()
    yield
    sr._OPENER_INDEX_CACHE.clear()


def half(game_pk, pitcher_outs, game_type="R", start_ab=1):
    """Rows for one half-game. `pitcher_outs` is [(pitcher_id, outs)] in appearance order.

    Every out is a separate PA row with an ascending at_bat_number, which is how statcast
    actually presents them.
    """
    rows, ab, inning = [], start_ab, 1
    for pitcher, outs in pitcher_outs:
        for _ in range(outs):
            rows.append({"game_pk": game_pk, "inning": inning, "pitcher": pitcher,
                         "inning_topbot": "Top", "at_bat_number": ab,
                         "events": "field_out", "game_type": game_type})
            ab += 1
            if ab % 3 == 1:
                inning += 1
    return rows


def index_of(rows):
    return build_opener_index(2026, "2026-08-05", pd.DataFrame(rows))


class TestInningsFromOuts:
    def test_innings_come_from_outs_not_from_innings_appeared_in(self):
        """2.1 IP is 2.33, not 3 -- the distinction is the whole opener test."""
        frame = index_of(half(1, [(100, 7), (200, 12)]))
        assert frame["first_inn"].iloc[0] == pytest.approx(7 / 3.0)

    def test_a_two_inning_opener_is_flagged(self):
        frame = index_of(half(1, [(100, 6), (200, 15)]))
        assert bool(frame["is_opener"].iloc[0])

    def test_an_arm_who_goes_two_and_a_third_is_not_an_opener(self):
        """The old distinct-innings count called this three innings and missed it; the
        new count calls it 2.33, which is over the limit -- both agree here, but for
        opposite reasons, so the boundary is worth pinning."""
        frame = index_of(half(1, [(100, 7), (200, 15)]))
        assert frame["first_inn"].iloc[0] > OPENER_MAX_INNINGS
        assert not bool(frame["is_opener"].iloc[0])

    def test_double_plays_count_two_outs(self):
        rows = half(1, [(100, 4), (200, 12)])
        rows[0]["events"] = "grounded_into_double_play"      # 4 rows -> 5 outs
        frame = index_of(rows)
        assert frame["first_inn"].iloc[0] == pytest.approx(5 / 3.0)

    def test_a_non_out_event_adds_no_outs(self):
        rows = half(1, [(100, 6), (200, 12)])
        rows[0]["events"] = "single"
        frame = index_of(rows)
        assert frame["first_inn"].iloc[0] == pytest.approx(5 / 3.0)


class TestFollowerOrdering:
    def test_follower_is_the_second_arm_to_appear(self):
        frame = index_of(half(1, [(100, 3), (200, 12), (300, 6)]))
        assert int(frame["bulk"].iloc[0]) == 200

    def test_two_arms_in_the_same_inning_are_ordered_by_plate_appearance(self):
        """Sorting on inning alone left this tie to a non-stable sort."""
        rows = half(1, [(100, 3), (200, 1), (300, 14)])
        for row in rows:
            row["inning"] = 1                                # force the tie
        frame = index_of(rows)
        assert int(frame["first"].iloc[0]) == 100
        assert int(frame["bulk"].iloc[0]) == 200

    def test_a_complete_game_has_no_follower(self):
        frame = index_of(half(1, [(100, 27)]))
        assert pd.isna(frame["bulk"].iloc[0])
        assert not bool(frame["is_opener"].iloc[0])

    def test_a_short_start_without_a_bulk_arm_is_not_an_opener(self):
        """An opener needs someone behind him who actually carries the game."""
        frame = index_of(half(1, [(100, 6), (200, 3), (300, 3)]))
        assert frame["bulk_inn"].iloc[0] < BULK_MIN_INNINGS
        assert not bool(frame["is_opener"].iloc[0])


class TestGameTypeFilter:
    def test_spring_training_is_excluded(self):
        """Every spring start is opener-shaped; none of it means anything."""
        frame = index_of(half(1, [(100, 6), (200, 15)], game_type="S"))
        assert frame.empty

    def test_regular_season_is_kept(self):
        assert not index_of(half(1, [(100, 6), (200, 15)], game_type="R")).empty

    def test_a_mixed_pull_keeps_only_the_regular_season_games(self):
        rows = half(1, [(100, 6), (200, 15)], game_type="S") + \
               half(2, [(101, 6), (201, 15)], game_type="R")
        frame = index_of(rows)
        assert list(frame["game_pk"]) == [2]


class TestHalves:
    def test_the_two_halves_are_scored_separately(self):
        rows = half(1, [(100, 6), (200, 15)])
        other = half(1, [(300, 21)])
        for row in other:
            row["inning_topbot"] = "Bot"
        frame = index_of(rows + other)
        assert len(frame) == 2
        assert set(frame["is_opener"]) == {True, False}


class TestConfidenceConstant:
    def test_the_repeat_rate_is_reported_as_the_coin_flip_it_is(self):
        """Shipping this above ~0.5 would turn a scouting note back into a prediction."""
        assert 0.2 <= OPENER_BULK_REPEAT_RATE <= 0.45
