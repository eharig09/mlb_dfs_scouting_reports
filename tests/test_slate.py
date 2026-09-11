"""Slate assembly: doubleheaders, started-game filtering, and lineup confirmation.

Each of these has a documented history of being got wrong in a way that produces a
plausible-looking board. A doubleheader resolved at team level deleted a nightcap that had
not thrown a pitch; a started-game check at team level did the same to a live slate. The
tests below pin the per-*game* behaviour that replaced them.
"""

import os

import pytest

from dfs.projections import _weather_hr_factor, _weather_inputs
from dfs.salaries import canon_team, normalize_name
from dfs.slate import (build_slate, build_stacks, game_number, refresh_game_calibration,
                       resolve_doubleheaders, started_players)


class TestDoubleheaderCacheKeys:
    def test_game_number_defaults_to_one(self):
        assert game_number(os.path.join(".cache", "2026-07-28_CLE_CIN.pkl")) == 1

    def test_game_number_reads_the_suffix(self):
        assert game_number(os.path.join(".cache", "2026-07-28_CLE_CIN_g2.pkl")) == 2

    def test_a_team_name_ending_in_g_is_not_a_game_number(self):
        assert game_number("2026-07-28_CHC_WSH.pkl") == 1


class TestCalibrationBridge:
    def test_changed_payload_is_persisted_and_provenance_returned(self, monkeypatch):
        payload = {"advanced_context": {}}
        updated = {"advanced_context": {"projection_summary": {"Total Runs": 9.1}}}
        provenance = {"fingerprint": "abc123", "feature_version": 3,
                      "created_at": "2026-08-10 01:00:00"}
        monkeypatch.setattr(
            "scouting_report.refresh_payload_model_calibration",
            lambda value: (updated, True, provenance),
        )
        written = {}
        monkeypatch.setattr(
            "dfs.slate._persist_payload",
            lambda path, value: written.update({"path": path, "payload": value}),
        )

        result, changed, applied = refresh_game_calibration(payload, "game.pkl")

        assert changed
        assert result is updated
        assert applied == provenance
        assert written == {"path": "game.pkl", "payload": updated}

    def test_slate_fails_closed_when_one_game_cannot_refresh(self, monkeypatch):
        monkeypatch.setattr(
            "dfs.slate.cached_games",
            lambda date, data_dir: [
                ("good.pkl", "NYY", "BOS"),
                ("bad.pkl", "TOR", "BAL"),
            ],
        )
        monkeypatch.setattr("dfs.slate.find_salary_file", lambda *args, **kwargs: None)
        monkeypatch.setattr("dfs.slate._load", lambda path: {"path": path})

        def refresh(payload, path):
            if path == "bad.pkl":
                raise RuntimeError("artifact mismatch")
            return payload, False, {
                "fingerprint": "abc123",
                "feature_version": 3,
                "created_at": "2026-08-10 01:00:00",
            }

        monkeypatch.setattr("dfs.slate.refresh_game_calibration", refresh)
        monkeypatch.setattr("dfs.slate.project_game", lambda payload: [{"Name": "Player"}])

        players, stacks, meta = build_slate("2026-08-09", check_schedule=False)

        assert players.empty
        assert stacks.empty
        assert not meta["calibration_consistent"]
        assert len(meta["calibration_errors"]) == 1


class TestDoubleheaderResolution:
    def test_a_single_game_pairing_is_untouched(self, tmp_path):
        games = [("a.pkl", "CLE", "CIN"), ("b.pkl", "NYY", "BOS")]
        kept, dropped = resolve_doubleheaders(games, salary_file=None)
        assert len(kept) == 2
        assert dropped == []

    def test_both_halves_would_double_count_every_player(self, tmp_path, monkeypatch):
        """Leaving both would list every CLE and CIN player twice, with different starters."""
        payloads = {}
        for name, hour in (("g1.pkl", "17:40"), ("g2.pkl", "23:10")):
            payloads[name] = {"advanced_context": {"environment": {
                "game_datetime": f"2026-07-28T{hour}:00Z"}}}
        monkeypatch.setattr("dfs.slate._load", lambda path: payloads[path])
        games = [("g1.pkl", "CLE", "CIN"), ("g2.pkl", "CLE", "CIN")]
        kept, dropped = resolve_doubleheaders(games, salary_file=None)
        assert len(kept) == 1
        assert len(dropped) == 1

    def test_without_a_priced_start_time_the_later_game_is_kept(self, monkeypatch):
        """A main slate almost always prices the nightcap; say so rather than guessing."""
        payloads = {
            "g1.pkl": {"advanced_context": {"environment": {
                "game_datetime": "2026-07-28T17:40:00Z"}}},
            "g2.pkl": {"advanced_context": {"environment": {
                "game_datetime": "2026-07-28T23:10:00Z"}}},
        }
        monkeypatch.setattr("dfs.slate._load", lambda path: payloads[path])
        kept, dropped = resolve_doubleheaders(
            [("g1.pkl", "CLE", "CIN"), ("g2.pkl", "CLE", "CIN")], salary_file=None)
        assert kept[0][0] == "g2.pkl"
        assert "later game" in dropped[0][2]

    def test_aliased_team_codes_still_pair_up(self, monkeypatch):
        payloads = {p: {"advanced_context": {"environment": {
            "game_datetime": "2026-07-28T23:10:00Z"}}} for p in ("g1.pkl", "g2.pkl")}
        monkeypatch.setattr("dfs.slate._load", lambda path: payloads[path])
        kept, dropped = resolve_doubleheaders(
            [("g1.pkl", "OAK", "CHW"), ("g2.pkl", "ATH", "CWS")], salary_file=None)
        assert len(kept) == 1, "OAK@CHW and ATH@CWS are the same pairing"


class TestStartedGameFiltering:
    def _slate_with_games(self, slate):
        return slate

    def test_no_schedule_means_nothing_is_dropped(self, slate, monkeypatch):
        """A pairing the schedule does not know about is never dropped on a guess."""
        monkeypatch.setattr("dfs.schedule.started_pairings", lambda date, **kw: ({}, None))
        monkeypatch.setattr("dfs.schedule.pitching_elsewhere", lambda date, **kw: (set(), None))
        locked, elsewhere, error = started_players(slate, {"game_starts": {}}, "2026-08-01")
        assert not locked.any()
        assert not elsewhere.any()

    def test_a_schedule_error_is_surfaced_not_swallowed(self, slate, monkeypatch):
        monkeypatch.setattr("dfs.schedule.started_pairings",
                            lambda date, **kw: ({}, "StatsAPI unreachable"))
        _, _, error = started_players(slate, {}, "2026-08-01")
        assert error == "StatsAPI unreachable"

    def test_a_started_game_locks_only_its_own_players(self, slate, monkeypatch):
        pairing = slate["Game"].iloc[0]
        away, _, home = pairing.partition("@")
        key = f"{canon_team(away)}@{canon_team(home)}"
        monkeypatch.setattr("dfs.schedule.started_pairings",
                            lambda date, **kw: ({key: [{"start": 1200, "started": True}]}, None))
        monkeypatch.setattr("dfs.schedule.pitching_elsewhere", lambda date, **kw: (set(), None))
        locked, _, _ = started_players(slate, {"game_starts": {key: 1200}}, "2026-08-01")
        assert locked[slate["Game"] == pairing].all()
        assert not locked[slate["Game"] != pairing].any()

    def test_a_doubleheader_nightcap_stays_available(self, slate, monkeypatch):
        """Team-level checking deleted CIN and CLE from a slate whose 7:10 had not started."""
        pairing = slate["Game"].iloc[0]
        away, _, home = pairing.partition("@")
        key = f"{canon_team(away)}@{canon_team(home)}"
        scheduled = [{"start": 1060, "started": True},     # the opener, under way
                     {"start": 1390, "started": False}]    # the nightcap, priced
        monkeypatch.setattr("dfs.schedule.started_pairings",
                            lambda date, **kw: ({key: scheduled}, None))
        monkeypatch.setattr("dfs.schedule.pitching_elsewhere", lambda date, **kw: (set(), None))
        locked, _, _ = started_players(slate, {"game_starts": {key: 1390}}, "2026-08-01")
        assert not locked.any(), "the nightcap was deleted by the opener's status"

    def test_a_pitcher_throwing_the_earlier_game_is_flagged_separately(self, slate, monkeypatch):
        """DK prices the whole staff against the nightcap; the opener's arm scores nothing."""
        arm = slate[slate["Type"] == "P"]["Name"].iloc[0]
        monkeypatch.setattr("dfs.schedule.started_pairings", lambda date, **kw: ({}, None))
        monkeypatch.setattr("dfs.schedule.pitching_elsewhere",
                            lambda date, **kw: ({normalize_name(arm)}, None))
        locked, elsewhere, _ = started_players(slate, {"game_starts": {}}, "2026-08-01")
        assert elsewhere[slate["Name"] == arm].all()
        assert not locked.any(), "he is swappable, unlike a locked player"

    def test_hitters_are_never_flagged_as_pitching_elsewhere(self, slate, monkeypatch):
        bat = slate[slate["Type"] == "H"]["Name"].iloc[0]
        monkeypatch.setattr("dfs.schedule.started_pairings", lambda date, **kw: ({}, None))
        monkeypatch.setattr("dfs.schedule.pitching_elsewhere",
                            lambda date, **kw: ({normalize_name(bat)}, None))
        _, elsewhere, _ = started_players(slate, {"game_starts": {}}, "2026-08-01")
        assert not elsewhere.any()


class TestLineupConfirmation:
    def test_confirmed_and_projected_are_distinguishable(self, slate):
        frame = slate.copy()
        frame.loc[frame["Type"] == "H", "Lineup"] = "Projected (2 patched)"
        assert (frame.loc[frame["Type"] == "H", "Lineup"] != "Confirmed").all()

    def test_a_lineup_change_moves_the_projection_and_the_snapshot_digest(self, slate,
                                                                          tmp_path):
        """A refreshed card must be visible as a different snapshot, not absorbed silently."""
        from dfs.snapshot import frame_digest, take_snapshot

        columns = ["Name", "Team", "Type", "Slot", "Salary", "Proj"]
        before = take_snapshot("2026-08-01", slate="main", stage="t-2h", players=slate,
                               meta={"has_salary": True}, root=str(tmp_path),
                               data_dir=str(tmp_path / "none"))

        confirmed = slate.copy()
        swap = confirmed.index[(confirmed["Team"] == "LAD") & (confirmed["Slot"] == 1)]
        confirmed.loc[swap, "Slot"] = 6
        confirmed.loc[swap, "Proj"] = confirmed.loc[swap, "Proj"] * 0.85
        after = take_snapshot("2026-08-01", slate="main", stage="confirmed",
                              players=confirmed, meta={"has_salary": True},
                              root=str(tmp_path), data_dir=str(tmp_path / "none"))

        assert frame_digest(slate, columns) != frame_digest(confirmed, columns)
        assert before.directory != after.directory
        assert before.manifest["digests"]["projections"] != \
            after.manifest["digests"]["projections"]


class TestWeatherInputs:
    def test_live_weather_wins_when_present(self):
        temp, wind, enclosed = _weather_inputs({
            "weather": {"temp": "84", "wind": "8 mph, L To R"},
            "forecast": {"temp_f": 70.0, "wind": "2 mph, calm", "roof": "open"},
        })
        assert temp == 84.0
        assert wind == "8 mph, L To R"
        assert not enclosed

    def test_forecast_is_used_when_live_weather_is_empty(self):
        """93 of 142 cached payloads: the old code read only `weather` and returned 1.0."""
        temp, wind, _ = _weather_inputs({
            "weather": {}, "forecast": {"temp_f": 96.7, "wind": "8 mph out to LF",
                                        "roof": "open"}})
        assert temp == 96.7
        assert "out to" in wind

    def test_a_hot_night_raises_the_home_run_factor(self):
        hot = _weather_hr_factor({"weather": {}, "forecast": {"temp_f": 96.0, "wind": "",
                                                              "roof": "open"}})
        cold = _weather_hr_factor({"weather": {}, "forecast": {"temp_f": 58.0, "wind": "",
                                                               "roof": "open"}})
        assert hot > 1.0 > cold

    def test_wind_out_helps_and_wind_in_hurts(self):
        out = _weather_hr_factor({"weather": {}, "forecast": {
            "temp_f": 72.0, "wind": "15 mph out to CF", "roof": "open"}})
        into = _weather_hr_factor({"weather": {}, "forecast": {
            "temp_f": 72.0, "wind": "15 mph in from CF", "roof": "open"}})
        assert out > 1.0 > into

    def test_a_closed_roof_neutralises_everything(self):
        for roof in ("cover", "fixed", "dome", "closed"):
            factor = _weather_hr_factor({"weather": {}, "forecast": {
                "temp_f": 99.0, "wind": "20 mph out to LF", "roof": roof}})
            assert factor == 1.0, roof

    def test_a_retractable_roof_is_treated_as_open(self):
        """In July it almost always is, and 13 of 142 cached games say retractable."""
        factor = _weather_hr_factor({"weather": {}, "forecast": {
            "temp_f": 95.0, "wind": "12 mph out to LF", "roof": "retractable"}})
        assert factor > 1.0

    def test_no_weather_at_all_is_neutral(self):
        assert _weather_hr_factor({}) == 1.0
        assert _weather_hr_factor({"weather": {}, "forecast": {}}) == 1.0

    def test_the_factor_stays_inside_its_clip(self):
        from dfs.projections import CLIP
        extreme = _weather_hr_factor({"weather": {}, "forecast": {
            "temp_f": 115.0, "wind": "40 mph out to CF", "roof": "open"}})
        assert CLIP["weather"][0] <= extreme <= CLIP["weather"][1]


class TestStacks:
    def test_teams_below_the_minimum_are_dropped(self, slate):
        thin = slate[~((slate["Team"] == "LAD") & (slate["Slot"] > 2))]
        stacks = build_stacks(thin, has_salary=True)
        assert "LAD" not in set(stacks["Team"])

    def test_stack_totals_use_the_top_of_the_order(self, slate):
        stacks = build_stacks(slate, has_salary=True)
        row = stacks.iloc[0]
        team = slate[(slate["Team"] == row["Team"]) & (slate["Type"] == "H")]
        top5 = team.sort_values("Slot").head(5)
        assert row["Top5 Proj"] == pytest.approx(top5["Proj"].sum(), abs=0.05)

    def test_every_team_with_enough_hitters_appears(self, slate):
        stacks = build_stacks(slate, has_salary=True)
        assert len(stacks) == slate[slate["Type"] == "H"]["Team"].nunique()


class TestSalariesNeverCrossTheSlateBoundary:
    """2026-08-25: a six-game contest put Dodgers and Cardinals hitters on the board.

    `attach_salaries` falls back to a bare-name match when name+team misses, which is what
    carries a player through an abbreviation change or a trade. But the fallback did not
    ask whether the export priced that player's team at all, so a projection for the
    Dodgers' Max Muncy took the salary of the Athletics' Max Muncy, and a Cardinals
    infielder named Jose Fermin took the salary of a Cleveland reliever. Both then passed
    the "DK priced it, so it is playable" filter and became draftable.
    """

    import pandas as pd

    SLATE_TEAMS = ["ATH", "CLE", "MIN", "LAA"]

    @staticmethod
    def _salaries(rows):
        import pandas as pd
        from dfs.salaries import _short_key, normalize_name
        frame = pd.DataFrame(rows)
        frame["_key"] = frame["Name"].map(normalize_name)
        frame["_short"] = frame["Name"].map(_short_key)
        frame["DK Name"] = frame["Name"]
        frame["DK Avg"] = 0.0
        frame["DK ID"] = range(len(frame))
        return frame

    @pytest.fixture
    def salaries(self):
        return self._salaries([
            {"Name": "Max Muncy", "DK Team": "ATH", "DK Pos": "3B", "Salary": 3100},
            {"Name": "Jose Fermin", "DK Team": "LAA", "DK Pos": "P", "Salary": 4000},
            {"Name": "Steven Kwan", "DK Team": "CLE", "DK Pos": "OF", "Salary": 4600},
        ])

    @staticmethod
    def _projections(rows):
        import pandas as pd
        return pd.DataFrame(rows)

    def test_an_off_slate_namesake_is_left_unpriced(self, salaries):
        from dfs.salaries import attach_salaries
        players = self._projections([
            {"Name": "Max Muncy", "Team": "LAD", "Type": "H"},
            {"Name": "Max Muncy", "Team": "ATH", "Type": "H"},
        ])
        priced = attach_salaries(players, salaries)
        assert priced.loc[priced["Team"] == "ATH", "Salary"].iloc[0] == 3100
        assert priced.loc[priced["Team"] == "LAD", "Salary"].isna().all()

    def test_a_hitter_never_takes_a_pitchers_price(self, salaries):
        """The Fermin case: same name, different team, and different side of the ball."""
        from dfs.salaries import attach_salaries
        players = self._projections([{"Name": "Jose Fermin", "Team": "STL", "Type": "H"}])
        assert attach_salaries(players, salaries)["Salary"].isna().all()

    def test_a_hitter_on_a_slate_team_still_cannot_take_a_pitchers_price(self, salaries):
        from dfs.salaries import attach_salaries
        players = self._projections([{"Name": "Jose Fermin", "Team": "LAA", "Type": "H"}])
        assert attach_salaries(players, salaries)["Salary"].isna().all()

    def test_the_fallback_still_carries_an_abbreviation_change(self, salaries):
        """Why the bare-name path exists: DK says OAK where the report says ATH."""
        from dfs.salaries import attach_salaries
        players = self._projections([{"Name": "Max Muncy", "Team": "OAK", "Type": "H"}])
        assert attach_salaries(players, salaries)["Salary"].iloc[0] == 3100

    def test_a_pitcher_on_a_slate_team_still_matches(self, salaries):
        from dfs.salaries import attach_salaries
        players = self._projections([{"Name": "Jose Fermin", "Team": "LAA", "Type": "P"}])
        assert attach_salaries(players, salaries)["Salary"].iloc[0] == 4000

    def test_a_projection_with_no_type_is_not_rejected(self, salaries):
        """`Type` is the board's column; nothing else should have to supply it."""
        from dfs.salaries import attach_salaries
        players = self._projections([{"Name": "Steven Kwan", "Team": "CLE"}])
        assert attach_salaries(players, salaries)["Salary"].iloc[0] == 4600
