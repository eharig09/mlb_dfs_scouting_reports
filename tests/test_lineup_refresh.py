"""`--refresh-lineups` should not rebuild a report whose lineups have not moved.

The refresh re-pulls hot/cold game logs and season stats for eighteen hitters, loads the
statcast splits and detail tables, and rebuilds the composite and offense index. Autosnap
runs it once per stage, so on a night where the card posts early most of those runs are
rebuilding an identical report. Reading the two cards first is two API calls and settles it.
"""

import pandas as pd
import pytest

import scouting_report as sr


def lineup_frame(names, confidence="Confirmed"):
    return pd.DataFrame({
        "Spot": list(range(1, len(names) + 1)),
        "Name": list(names),
        "Lineup Confidence": [confidence] * len(names),
    })


def card(names):
    """What `get_today_lineup` returns: batting order as 101, 201, ..., 901."""
    return [{"Name": name, "#": (i + 1) * 100 + 1, "Pos": "LF", "Bats": "R", "ID": 100 + i}
            for i, name in enumerate(names)]


NINE = [f"Hitter {i}" for i in range(1, 10)]
OTHER = ["Pinch Hitter"] + NINE[1:]


def payload_for(home, away):
    args = [None] * 27
    args[0], args[7] = home, away
    return {"report_args": tuple(args), "advanced_context": {}}


@pytest.fixture
def posted(monkeypatch):
    """Control what each team's card says. {} means nothing posted."""
    cards = {}

    def fake(team_abbr, game_date=None):
        return cards.get(team_abbr, [])

    monkeypatch.setattr(sr, "get_today_lineup", fake)
    return cards


class TestNothingChanged:
    def test_matching_confirmed_cards_are_current(self, posted):
        posted["CHC"] = card(NINE)
        posted["STL"] = card(OTHER)
        current, _ = sr.cached_lineups_are_current(
            payload_for(lineup_frame(NINE), lineup_frame(OTHER)), "2026-08-07", "STL", "CHC")
        assert current

    def test_order_of_the_cached_rows_does_not_matter(self, posted):
        posted["CHC"] = card(NINE)
        posted["STL"] = card(OTHER)
        shuffled = lineup_frame(NINE).iloc[::-1].reset_index(drop=True)
        current, _ = sr.cached_lineups_are_current(
            payload_for(shuffled, lineup_frame(OTHER)), "2026-08-07", "STL", "CHC")
        assert current

    def test_accents_do_not_count_as_a_change(self, posted):
        accented = ["José Ramírez"] + NINE[1:]
        posted["CHC"] = card(["Jose Ramirez"] + NINE[1:])
        posted["STL"] = card(OTHER)
        current, _ = sr.cached_lineups_are_current(
            payload_for(lineup_frame(accented), lineup_frame(OTHER)), "2026-08-07", "STL", "CHC")
        assert current

    def test_an_unposted_card_against_a_projection_is_current(self, posted):
        """Re-projecting the same date off the same history returns the same nine."""
        current, reason = sr.cached_lineups_are_current(
            payload_for(lineup_frame(NINE, "Projected"), lineup_frame(OTHER, "Projected (8/9)")),
            "2026-08-07", "STL", "CHC")
        assert current, reason


class TestSomethingChanged:
    def test_a_different_name_forces_a_refresh(self, posted):
        posted["CHC"] = card(["Someone Else"] + NINE[1:])
        posted["STL"] = card(OTHER)
        current, reason = sr.cached_lineups_are_current(
            payload_for(lineup_frame(NINE), lineup_frame(OTHER)), "2026-08-07", "STL", "CHC")
        assert not current and "differs" in reason

    def test_a_reordered_lineup_forces_a_refresh(self, posted):
        posted["CHC"] = card(list(reversed(NINE)))
        posted["STL"] = card(OTHER)
        current, _ = sr.cached_lineups_are_current(
            payload_for(lineup_frame(NINE), lineup_frame(OTHER)), "2026-08-07", "STL", "CHC")
        assert not current

    def test_a_projection_that_is_now_posted_is_refreshed_for_the_label(self, posted):
        """Same nine, but the report should say Confirmed rather than Projected."""
        posted["CHC"] = card(NINE)
        posted["STL"] = card(OTHER)
        current, reason = sr.cached_lineups_are_current(
            payload_for(lineup_frame(NINE, "Projected"), lineup_frame(OTHER)),
            "2026-08-07", "STL", "CHC")
        assert not current and "Projected" in reason

    def test_a_fallback_is_always_retried(self, posted):
        current, reason = sr.cached_lineups_are_current(
            payload_for(lineup_frame(NINE, "Fallback"), lineup_frame(OTHER, "Projected")),
            "2026-08-07", "STL", "CHC")
        assert not current and "Fallback" in reason

    def test_a_partial_card_is_not_a_match(self, posted):
        """MLB exposes a few slots early; those are not a lineup."""
        posted["CHC"] = card(NINE[:4])
        current, _ = sr.cached_lineups_are_current(
            payload_for(lineup_frame(NINE), lineup_frame(OTHER)), "2026-08-07", "STL", "CHC")
        assert not current

    def test_an_unreadable_cache_is_refreshed(self, posted):
        posted["CHC"] = card(NINE)
        current, _ = sr.cached_lineups_are_current(
            payload_for(pd.DataFrame(), lineup_frame(OTHER)), "2026-08-07", "STL", "CHC")
        assert not current

    def test_a_doubleheader_is_never_skipped(self, posted):
        """get_today_lineup reads schedule[0], so it may answer about the other half."""
        posted["CHC"] = card(NINE)
        posted["STL"] = card(OTHER)
        current, reason = sr.cached_lineups_are_current(
            payload_for(lineup_frame(NINE), lineup_frame(OTHER)),
            "2026-08-07", "STL", "CHC", dh_game=2)
        assert not current and "doubleheader" in reason

    def test_a_card_that_cannot_be_read_falls_through_to_refreshing(self, monkeypatch):
        def boom(team_abbr, game_date=None):
            raise RuntimeError("statsapi down")

        monkeypatch.setattr(sr, "get_today_lineup", boom)
        current, _ = sr.cached_lineups_are_current(
            payload_for(lineup_frame(NINE), lineup_frame(OTHER)), "2026-08-07", "STL", "CHC")
        assert not current


class TestRefreshHonoursTheCheck:
    def test_an_unchanged_night_returns_the_payload_untouched(self, posted, monkeypatch):
        posted["CHC"] = card(NINE)
        posted["STL"] = card(OTHER)

        def fail(*a, **k):
            raise AssertionError("refresh did work it should have skipped")

        monkeypatch.setattr(sr, "load_statcast_splits", fail)
        monkeypatch.setattr(sr, "generate_team_hitter_report", fail)

        payload = payload_for(lineup_frame(NINE), lineup_frame(OTHER))
        assert sr.refresh_cached_lineups(payload, "2026-08-07", "STL", "CHC") is payload

    def test_force_rebuilds_even_when_the_cards_match(self, posted, monkeypatch):
        posted["CHC"] = card(NINE)
        posted["STL"] = card(OTHER)
        called = []
        monkeypatch.setattr(sr, "load_statcast_splits",
                            lambda **k: called.append("splits") or pd.DataFrame())
        monkeypatch.setattr(sr, "load_statcast_detail",
                            lambda **k: called.append("detail") or pd.DataFrame())

        payload = payload_for(lineup_frame(NINE), lineup_frame(OTHER))
        with pytest.raises(Exception):
            # The rebuild proper needs far more of the pipeline than is stubbed here; that
            # it got as far as loading statcast is what this asserts.
            sr.refresh_cached_lineups(payload, "2026-08-07", "STL", "CHC", force=True)
        assert "splits" in called
