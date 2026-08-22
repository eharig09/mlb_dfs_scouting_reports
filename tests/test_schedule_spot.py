"""Schedule-spot bucketing, which used to be computed in UTC.

statsapi reports first pitch in UTC and every rule here is about local time. A 7:10pm
Pacific start is 02:10 UTC the following day, so reading `.hour` and `.date()` off the UTC
stamp called it a day game on the wrong date. Measured across five dates of the 2026
schedule that was 28% of games on the wrong side of day/night and 26% on the wrong date --
enough to corrupt rest gaps, doubleheader detection and get-away days league-wide.

These tests pin the local-time behaviour with fabricated schedule rows, so they run without
a network and they fail if the UTC shortcut ever comes back.
"""

import pytest

from scouting_report import (DAY_GAME_BEFORE_HOUR, SCHEDULE_SPOTS, _local_start,
                             _local_starts, _schedule_spot)

TEAM = 111          # the team under test
OPP_A, OPP_B = 147, 121

# venue_id 3 is Eastern (Fenway), 19 is Mountain (Coors), 1 is Pacific (Angel Stadium).
EASTERN, MOUNTAIN, PACIFIC = 3, 19, 1


def game(date, utc_time, venue=EASTERN, opponent=OPP_A, home=True):
    return {
        "game_date": date,
        "game_datetime": f"{utc_time}Z",
        "venue_id": venue,
        "home_id": TEAM if home else opponent,
        "away_id": opponent if home else TEAM,
        "status": "Final",
    }


def spots(games):
    from collections import Counter
    starts = _local_starts(games)
    per_date = Counter(d for d, _ in starts if d)
    return [_schedule_spot(games, starts, i, TEAM, per_date) for i in range(len(games))]


class TestLocalStart:
    def test_pacific_night_game_stays_on_its_own_date(self):
        """02:10 UTC on the 2nd is 19:10 local on the 1st -- a night game, not a day game."""
        date, hour = _local_start(game("2026-08-01", "2026-08-02T02:10:00", venue=PACIFIC))
        assert str(date) == "2026-08-01"
        assert hour == 19
        assert hour >= DAY_GAME_BEFORE_HOUR      # night

    def test_mountain_night_game_is_not_a_day_game(self):
        _, hour = _local_start(game("2026-08-01", "2026-08-02T00:10:00", venue=MOUNTAIN))
        assert hour == 18 and hour >= DAY_GAME_BEFORE_HOUR

    def test_eastern_day_game(self):
        _, hour = _local_start(game("2026-08-01", "2026-08-01T17:10:00"))
        assert hour == 13 and hour < DAY_GAME_BEFORE_HOUR

    def test_eastern_night_game(self):
        _, hour = _local_start(game("2026-08-01", "2026-08-01T23:10:00"))
        assert hour == 19 and hour >= DAY_GAME_BEFORE_HOUR

    def test_date_comes_from_the_official_field_not_the_stamp(self):
        """MLB's own date is authoritative; a late start must not roll the game forward."""
        date, _ = _local_start(game("2026-08-01", "2026-08-02T02:10:00", venue=PACIFIC))
        assert str(date) == "2026-08-01"

    def test_missing_stamp_still_yields_a_date(self):
        date, hour = _local_start({"game_date": "2026-08-01", "game_datetime": ""})
        assert str(date) == "2026-08-01" and hour is None

    def test_junk_row_is_none(self):
        assert _local_start({}) == (None, None)


class TestScheduleSpot:
    def test_every_label_is_a_known_bucket(self):
        games = [game("2026-08-01", "2026-08-01T23:10:00"),
                 game("2026-08-02", "2026-08-02T17:10:00")]
        assert all(s in SCHEDULE_SPOTS for s in spots(games))

    def test_day_after_night(self):
        games = [game("2026-08-01", "2026-08-01T23:10:00"),
                 game("2026-08-02", "2026-08-02T17:10:00")]
        assert spots(games)[1] == "Day after night"

    def test_day_after_night_fires_for_a_pacific_club_too(self):
        """The case the UTC version could not see: both games look like day games there."""
        games = [game("2026-08-01", "2026-08-02T02:10:00", venue=PACIFIC),
                 game("2026-08-02", "2026-08-02T20:10:00", venue=PACIFIC)]
        assert spots(games)[1] == "Day after night"

    def test_extra_rest(self):
        games = [game("2026-08-01", "2026-08-01T23:10:00"),
                 game("2026-08-04", "2026-08-04T23:10:00")]
        assert spots(games)[1] == "Extra rest (2+d)"

    def test_a_pacific_night_game_is_not_extra_rest_the_next_day(self):
        """UTC dates made a one-day gap look like two whenever the first game was late."""
        games = [game("2026-08-01", "2026-08-02T02:10:00", venue=PACIFIC),
                 game("2026-08-02", "2026-08-03T02:10:00", venue=PACIFIC)]
        assert spots(games)[1] != "Extra rest (2+d)"

    def test_doubleheader_nightcap(self):
        games = [game("2026-08-01", "2026-08-01T17:10:00"),
                 game("2026-08-01", "2026-08-01T23:10:00")]
        assert spots(games)[1] == "DH game 2"

    def test_doubleheader_nightcap_in_the_pacific(self):
        """The nightcap's UTC date used to land on the following day, hiding the DH."""
        games = [game("2026-08-01", "2026-08-01T20:10:00", venue=PACIFIC),
                 game("2026-08-01", "2026-08-02T02:10:00", venue=PACIFIC)]
        assert spots(games)[1] == "DH game 2"

    def test_day_after_a_doubleheader(self):
        games = [game("2026-08-01", "2026-08-01T17:10:00"),
                 game("2026-08-01", "2026-08-01T23:10:00"),
                 game("2026-08-02", "2026-08-02T23:10:00")]
        assert spots(games)[2] == "Day after DH"

    def test_getaway_day_needs_the_series_to_end(self):
        ending = [game("2026-08-01", "2026-08-01T23:10:00", opponent=OPP_A),
                  game("2026-08-02", "2026-08-02T17:10:00", opponent=OPP_A),
                  game("2026-08-03", "2026-08-03T23:10:00", opponent=OPP_B)]
        assert spots(ending)[1] == "Get-away day"

    def test_a_day_game_mid_series_is_not_a_getaway_day(self):
        staying = [game("2026-08-01", "2026-08-01T23:10:00", opponent=OPP_A),
                   game("2026-08-02", "2026-08-02T17:10:00", opponent=OPP_A),
                   game("2026-08-03", "2026-08-03T23:10:00", opponent=OPP_A)]
        assert staying[1] and spots(staying)[1] != "Get-away day"

    def test_a_pacific_night_getaway_is_not_called_a_day_game(self):
        """The single most common misfire: night games out west read as day games."""
        games = [game("2026-08-01", "2026-08-02T02:10:00", venue=PACIFIC, opponent=OPP_A),
                 game("2026-08-02", "2026-08-03T02:10:00", venue=PACIFIC, opponent=OPP_A),
                 game("2026-08-03", "2026-08-03T23:10:00", venue=EASTERN, opponent=OPP_B)]
        assert spots(games)[1] != "Get-away day"

    def test_doubleheader_beats_the_rest_spots(self):
        """Precedence: the more specific bucket has to win or it never appears."""
        games = [game("2026-08-01", "2026-08-01T17:10:00", opponent=OPP_A),
                 game("2026-08-01", "2026-08-01T23:10:00", opponent=OPP_A)]
        assert spots(games)[1] == "DH game 2"

    def test_first_game_of_the_run_is_normal(self):
        assert spots([game("2026-08-01", "2026-08-01T23:10:00")])[0] == "Normal"
