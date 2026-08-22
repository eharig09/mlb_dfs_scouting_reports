"""A game should not be skipped just because MLB has not posted a probable yet.

The hours before a probable posts are exactly when a board is being built, so the report
falls back to the DK salary file (which carries a `Starting` flag, usually populated first)
and then to the rotation model, labels whatever it used, and lets `--refresh-lineups` retire
the guess once a real probable appears.

Everything here is synthetic: no cached payload, no DK export on disk, no network.
"""

import pandas as pd
import pytest

import scouting_report as sr


DK_HEADER = ("Position,Name + ID,Name,ID,Roster Position,Salary,Game Info,"
             "TeamAbbrev,AvgPointsPerGame,Status,Starting")


def dk_export(tmp_path, name, rows, date="08/21/2026", game="TOR@NYY"):
    """A DraftKings salary export with just enough columns to be recognised as one."""
    lines = [DK_HEADER]
    for position, player, team, starting in rows:
        pid = abs(hash(player)) % 90000000
        lines.append(
            f"{position},{player} ({pid}),{player},{pid},P,8000,"
            f"{game} {date} 07:05PM ET,{team},12.0,,{starting}")
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def salary_dir(tmp_path, monkeypatch):
    """Point the DK loader at a temp folder so nothing on disk can leak into a test."""
    from dfs import salaries as dk

    monkeypatch.setattr(dk, "SALARY_DIRS", [str(tmp_path)])
    return tmp_path


@pytest.fixture
def roster(monkeypatch):
    """{team: {name: id}} standing in for the 40-man lookup."""
    people = {}

    def fake(team_abbr, roster_type="active", as_of_date=None):
        return [{"person": {"id": pid, "fullName": name}}
                for name, pid in people.get(team_abbr, {}).items()]

    monkeypatch.setattr(sr, "get_team_roster", fake)
    return people


class TestSalaryFile:
    def test_reads_the_starting_flag(self, salary_dir, roster):
        roster["NYY"] = {"Cam Schlittler": 4001}
        dk_export(salary_dir, "DKSalaries_main.csv", [
            ("SP", "Cam Schlittler", "NYY", "P"),
            ("SP", "Some Other Arm", "NYY", ""),
            ("OF", "Aaron Judge", "NYY", ""),
        ])
        assert sr.starter_from_salary_file("2026-08-21", "NYY") == 4001

    def test_an_opener_listed_at_rp_still_counts(self, salary_dir, roster):
        # DK lists the arm at RP but flags him starting. He is who the top of the order
        # leads off against, which is the whole question the report is answering.
        roster["TOR"] = {"Bullpen Guy": 4002}
        dk_export(salary_dir, "DKSalaries_main.csv", [("RP", "Bullpen Guy", "TOR", "P")])
        assert sr.starter_from_salary_file("2026-08-21", "TOR") == 4002

    def test_every_slate_for_the_date_is_read(self, salary_dir, roster):
        # DK puts up several slates a night and any one game is on a subset of them, so
        # reading only the first file misses most of the board.
        roster["SEA"] = {"Late Arm": 4003}
        dk_export(salary_dir, "DKSalaries_main.csv", [("SP", "Cam Schlittler", "NYY", "P")])
        dk_export(salary_dir, "DKSalaries_late.csv", [("SP", "Late Arm", "SEA", "P")],
                  game="SEA@LAA")
        assert sr.starter_from_salary_file("2026-08-21", "SEA") == 4003

    def test_a_file_for_another_date_is_ignored(self, salary_dir, roster):
        roster["NYY"] = {"Cam Schlittler": 4001}
        dk_export(salary_dir, "DKSalaries_main.csv",
                  [("SP", "Cam Schlittler", "NYY", "P")], date="08/22/2026")
        assert sr.starter_from_salary_file("2026-08-21", "NYY") is None

    def test_accents_and_punctuation_still_join(self, salary_dir, roster):
        roster["SD"] = {"Yu Darvish Jr.": 4004}
        dk_export(salary_dir, "DKSalaries_main.csv", [("SP", "Yu Darvish", "SD", "P")],
                  game="SD@COL")
        assert sr.starter_from_salary_file("2026-08-21", "SD") == 4004

    def test_a_starter_already_committed_to_the_other_game_is_refused(self, salary_dir, roster):
        # A doubleheader: DK shows one starter per team per day, so without the exclusion
        # it hands game one's arm to game two, which is impossible.
        roster["NYY"] = {"Cam Schlittler": 4001}
        dk_export(salary_dir, "DKSalaries_main.csv", [("SP", "Cam Schlittler", "NYY", "P")])
        assert sr.starter_from_salary_file("2026-08-21", "NYY", exclude_ids=[4001]) is None

    def test_no_flagged_starter_is_not_a_guess(self, salary_dir, roster):
        roster["NYY"] = {"Cam Schlittler": 4001}
        dk_export(salary_dir, "DKSalaries_main.csv", [("SP", "Cam Schlittler", "NYY", "")])
        assert sr.starter_from_salary_file("2026-08-21", "NYY") is None

    def test_no_salary_file_at_all(self, salary_dir, roster):
        assert sr.starter_from_salary_file("2026-08-21", "NYY") is None

    def test_a_name_off_the_roster_is_refused(self, salary_dir, roster, monkeypatch):
        # A DK row carrying a traded pitcher's old club must not resolve for that club.
        roster["NYY"] = {"Cam Schlittler": 4001}
        monkeypatch.setattr(sr, "lookup_player_id", lambda name: 9999)
        monkeypatch.setattr(sr, "pitcher_belongs_to_team", lambda pid, team: False)
        dk_export(salary_dir, "DKSalaries_main.csv", [("SP", "Traded Away", "NYY", "P")])
        assert sr.starter_from_salary_file("2026-08-21", "NYY") is None


@pytest.fixture
def ladder(monkeypatch, salary_dir, roster):
    """Every rung of `resolve_starter` under test control. Nothing resolves by default."""
    monkeypatch.setattr(sr, "_resolve_pitcher_override", lambda value, team: None)
    monkeypatch.setattr(sr, "resolve_probable_pitcher_v2",
                        lambda game, side, team, exclude=None, date=None: None)
    monkeypatch.setattr(sr, "starter_from_rotation",
                        lambda *a, **k: (None, None, None, ""))
    monkeypatch.setattr(sr, "_player_name_from_id", lambda pid: f"Pitcher {pid}")
    return monkeypatch


GAME = {"game_id": 1, "home_probable_pitcher": "TBD", "away_probable_pitcher": "TBD"}


class TestLadder:
    def test_an_announced_probable_wins_and_is_not_provisional(self, ladder):
        ladder.setattr(sr, "resolve_probable_pitcher_v2",
                       lambda game, side, team, exclude=None, date=None: 5001)
        pick = sr.resolve_starter(GAME, "home", "NYY", "2026-08-21")
        assert (pick["id"], pick["source"], pick["provisional"]) == (5001, "announced", False)

    def test_an_override_outranks_everything(self, ladder):
        ladder.setattr(sr, "_resolve_pitcher_override", lambda value, team: 5002)
        ladder.setattr(sr, "resolve_probable_pitcher_v2",
                       lambda game, side, team, exclude=None, date=None: 5001)
        pick = sr.resolve_starter(GAME, "home", "NYY", "2026-08-21", override="whoever")
        assert (pick["id"], pick["source"], pick["provisional"]) == (5002, "override", False)

    def test_the_salary_file_fills_in_for_a_missing_probable(self, ladder, salary_dir, roster):
        roster["NYY"] = {"Cam Schlittler": 4001}
        dk_export(salary_dir, "DKSalaries_main.csv", [("SP", "Cam Schlittler", "NYY", "P")])
        pick = sr.resolve_starter(GAME, "home", "NYY", "2026-08-21")
        assert (pick["id"], pick["source"], pick["provisional"]) == (4001, "salary", True)

    def test_the_rotation_model_is_the_last_rung(self, ladder):
        ladder.setattr(sr, "starter_from_rotation",
                       lambda *a, **k: (6001, "Rotation Arm", "rotation", "46% on 5 days"))
        pick = sr.resolve_starter(GAME, "home", "NYY", "2026-08-21")
        assert (pick["id"], pick["source"], pick["provisional"]) == (6001, "rotation", True)
        assert pick["note"] == "46% on 5 days"

    def test_nothing_anywhere_returns_no_id(self, ladder):
        pick = sr.resolve_starter(GAME, "home", "NYY", "2026-08-21")
        assert pick["id"] is None and pick["provisional"]

    def test_strict_mode_stops_after_the_announced_branch(self, ladder, salary_dir, roster):
        roster["NYY"] = {"Cam Schlittler": 4001}
        dk_export(salary_dir, "DKSalaries_main.csv", [("SP", "Cam Schlittler", "NYY", "P")])
        pick = sr.resolve_starter(GAME, "home", "NYY", "2026-08-21", allow_projected=False)
        assert pick["id"] is None

    def test_an_excluded_arm_is_not_offered_by_the_rotation_model(self, ladder, monkeypatch):
        seen = {}

        def fake(team, season, context_end, as_of_date=None, exclude_ids=None):
            seen["exclude"] = set(exclude_ids or [])
            return None, None, None, ""

        monkeypatch.setattr(sr, "starter_from_rotation", fake)
        sr.resolve_starter(GAME, "away", "TOR", "2026-08-21", exclude_ids=[7001])
        assert seen["exclude"] == {7001}


def payload(home_id, away_id, provenance):
    args = [None] * 27
    args[5], args[12] = home_id, away_id
    return {"report_args": tuple(args),
            "advanced_context": {"starter_provenance": provenance}}


def provisional(pid, name, source="rotation"):
    return {"id": pid, "name": name, "source": source, "provisional": True, "note": ""}


def announced(pid, name):
    return {"id": pid, "name": name, "source": "announced", "provisional": False, "note": ""}


@pytest.fixture
def recheck(monkeypatch):
    """`recheck_cached_starters` with the game lookup and the cache write stubbed out."""
    saved = []
    monkeypatch.setattr(sr, "choose_game",
                        lambda date, game_number=None, away_team=None, home_team=None,
                        dh_game=None: dict(GAME))
    monkeypatch.setattr(sr, "save_report_data_cache",
                        lambda *a, **k: saved.append(a))
    return saved


class TestRecheck:
    def test_an_announced_report_is_left_alone(self, recheck, monkeypatch):
        monkeypatch.setattr(sr, "resolve_starter",
                            lambda *a, **k: pytest.fail("should not re-resolve"))
        action, _ = sr.recheck_cached_starters(
            payload(1, 2, {"home": announced(1, "A"), "away": announced(2, "B")}),
            "2026-08-21", "TOR", "NYY")
        assert action == "none"

    def test_a_payload_without_provenance_predates_the_ladder(self, recheck):
        action, _ = sr.recheck_cached_starters(
            payload(1, 2, {}), "2026-08-21", "TOR", "NYY")
        assert action == "none"

    def test_a_different_arm_forces_a_rebuild(self, recheck, monkeypatch):
        monkeypatch.setattr(sr, "resolve_starter",
                            lambda *a, **k: announced(99, "Real Starter"))
        action, detail = sr.recheck_cached_starters(
            payload(1, 2, {"home": provisional(1, "Guessed Arm")}),
            "2026-08-21", "TOR", "NYY")
        assert action == "rebuild"
        assert "Guessed Arm -> Real Starter" in detail

    def test_the_same_arm_confirmed_only_relabels(self, recheck, monkeypatch):
        monkeypatch.setattr(sr, "resolve_starter", lambda *a, **k: announced(1, "Guessed Arm"))
        action, detail = sr.recheck_cached_starters(
            payload(1, 2, {"home": provisional(1, "Guessed Arm")}),
            "2026-08-21", "TOR", "NYY")
        assert action == "relabel"
        assert "confirmed by announced probable" in detail
        assert recheck, "a relabel must persist the new provenance"

    def test_still_nothing_announced_changes_nothing(self, recheck, monkeypatch):
        monkeypatch.setattr(sr, "resolve_starter",
                            lambda *a, **k: provisional(1, "Guessed Arm"))
        action, _ = sr.recheck_cached_starters(
            payload(1, 2, {"home": provisional(1, "Guessed Arm")}),
            "2026-08-21", "TOR", "NYY")
        assert action == "none"
        assert not recheck, "nothing moved, so nothing should have been written"

    def test_a_side_that_cannot_be_resolved_keeps_the_cached_arm(self, recheck, monkeypatch):
        monkeypatch.setattr(sr, "resolve_starter",
                            lambda *a, **k: {"id": None, "name": "TBD",
                                             "source": "unknown", "provisional": True})
        action, _ = sr.recheck_cached_starters(
            payload(1, 2, {"home": provisional(1, "Guessed Arm")}),
            "2026-08-21", "TOR", "NYY")
        assert action == "none"


class TestLabels:
    def test_an_announced_starter_gets_no_marker(self):
        context = {"starter_provenance": {"home": announced(1, "A"), "away": announced(2, "B")}}
        assert sr.cmp_starter_provisional_note(context, "TOR", "NYY") == ""
        assert sr.cmp_starter_provisional_tag(context, "home") == ""

    def test_a_projected_starter_is_named_with_its_source(self):
        context = {"starter_provenance": {"home": provisional(1, "Guessed Arm", "salary"),
                                          "away": announced(2, "B")}}
        note = sr.cmp_starter_provisional_note(context, "TOR", "NYY")
        assert "NYY Guessed Arm" in note and "DraftKings salary file" in note
        assert "TOR" not in note
        assert "PROJECTED STARTER" in sr.cmp_starter_provisional_tag(context, "home")
        assert sr.cmp_starter_provisional_tag(context, "away") == ""

    def test_a_context_predating_the_feature_is_silent(self):
        assert sr.cmp_starter_provisional_note({}, "TOR", "NYY") == ""
        assert sr.cmp_starter_provisional_tag(None, "home") == ""

    def test_the_rotation_models_own_confidence_rides_along(self):
        # The share is renormalised over whoever survives availability gating, so a club
        # whose start order has only ever gone one way returns 100% for an arm on a month's
        # rest. Printing the rest next to the share is the only way a reader can tell that
        # apart from a real 100%.
        pick = provisional(1, "Spencer Miles", "rotation")
        pick["note"] = ("no probable announced; rotation model gives Spencer Miles "
                        "100% on 36 days rest")
        context = {"starter_provenance": {"home": pick}}
        tag = sr.cmp_starter_provisional_tag(context, "home")
        assert "100% on 36 days rest" in tag
        assert "Spencer Miles" not in tag, "the panel header already names him"

    def test_the_salary_source_says_nothing_more_than_its_name(self):
        pick = provisional(1, "Cam Schlittler", "salary")
        pick["note"] = "MLB has not posted a probable; DraftKings lists him as starting"
        tag = sr.cmp_starter_provisional_tag({"starter_provenance": {"home": pick}}, "home")
        assert tag.endswith("DraftKings salary file"), tag


class TestPageFallbackIsToday:
    """The probable-pitchers page has no date parameter -- it is tonight's board.

    Consulted for any other date it answers a different question, and the answer arrives
    labelled `announced`, which is the one label that suppresses every fallback below it.
    """

    def test_a_future_date_never_reaches_the_page(self, monkeypatch):
        monkeypatch.setattr(sr, "get_probable_pitchers_for_game", lambda gid: {})
        monkeypatch.setattr(sr, "get_pitchers_mlb",
                            lambda team: pytest.fail("scraped the page for another date"))
        assert sr.resolve_probable_pitcher_v2(GAME, "home", "DET", date="2099-08-24") is None

    def test_today_still_uses_it(self, monkeypatch):
        import datetime as dt

        monkeypatch.setattr(sr, "get_probable_pitchers_for_game", lambda gid: {})
        monkeypatch.setattr(sr, "get_pitchers_mlb", lambda team: 8001)
        monkeypatch.setattr(sr, "pitcher_belongs_to_team", lambda pid, team: True)
        today = dt.datetime.now().strftime("%Y-%m-%d")
        assert sr.resolve_probable_pitcher_v2(GAME, "home", "DET", date=today) == 8001

    def test_no_date_given_keeps_the_old_behaviour(self, monkeypatch):
        monkeypatch.setattr(sr, "get_probable_pitchers_for_game", lambda gid: {})
        monkeypatch.setattr(sr, "get_pitchers_mlb", lambda team: 8002)
        monkeypatch.setattr(sr, "pitcher_belongs_to_team", lambda pid, team: True)
        assert sr.resolve_probable_pitcher_v2(GAME, "home", "DET") == 8002
