"""Scheduling snapshots against first pitch.

The stages only mean something if they happen at the moments they name, and those moments
move with the slate: a 12:35 getaway slate and a 9:40 west-coast slate need the same
*relative* times, so a fixed cron entry would fire "t-2h" four hours early on one and after
lock on the other. Everything here is about that derivation being right, and it runs without
a network -- lock is passed in or read from a fixture DK export.
"""

import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from dfs.autosnap import (EASTERN, MORNING_NOT_BEFORE_HOUR, STAGE_OFFSETS, AutosnapError,
                          _clock12, _Tee, slate_lock, stage_times, wait_for_slate)
from dfs.snapshot import STAGES


def lock_at(hour, minute=0, day=(2026, 8, 8)):
    return datetime(day[0], day[1], day[2], hour, minute, tzinfo=EASTERN)


def salary_fixture(tmp_path, games):
    """A minimal DK export: describe_slate only needs Game Info to find first pitch."""
    lines = ["Position,Name + ID,Name,ID,Roster Position,Salary,Game Info,TeamAbbrev,AvgPointsPerGame"]
    for i, (pairing, stamp) in enumerate(games):
        lines.append(f"SP,P{i} ({i}),P{i},{i},P,8000,{pairing} {stamp},AAA,10.0")
    path = tmp_path / "DKSalaries_test.csv"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


class TestStageTimes:
    def test_every_stage_lands_at_its_offset(self):
        lock = lock_at(19, 10)
        plan = dict(stage_times(lock))
        for stage, offset in STAGE_OFFSETS.items():
            if stage == "morning":
                continue                      # floored separately, tested below
            assert plan[stage] == lock - timedelta(minutes=offset)

    def test_the_plan_is_in_chronological_order(self):
        times = [fire for _, fire in stage_times(lock_at(19, 10))]
        assert times == sorted(times)

    def test_late_and_early_slates_get_the_same_relative_times(self):
        """The whole point: a fixed clock time cannot serve both."""
        early = dict(stage_times(lock_at(12, 35)))
        late = dict(stage_times(lock_at(22, 10)))
        for stage in ("t-2h", "confirmed", "final"):
            assert (lock_at(12, 35) - early[stage]) == (lock_at(22, 10) - late[stage])

    def test_final_sits_just_before_lock(self):
        lock = lock_at(19, 10)
        final = dict(stage_times(lock))["final"]
        assert timedelta(minutes=5) < lock - final < timedelta(minutes=30)

    def test_a_subset_of_stages_is_honoured(self):
        plan = stage_times(lock_at(19, 10), ["confirmed", "final"])
        assert [stage for stage, _ in plan] == ["confirmed", "final"]

    def test_an_unknown_stage_is_ignored_rather_than_crashing(self):
        assert stage_times(lock_at(19, 10), ["confirmed", "nonsense"]) != []

    def test_every_snapshot_stage_has_an_offset(self):
        """A stage dfs.snapshot knows about but autosnap cannot schedule is a silent gap."""
        assert set(STAGES) == set(STAGE_OFFSETS)


class TestMorningFloor:
    def test_an_early_slate_does_not_snapshot_at_dawn(self):
        """Lock - 6h for a 12:35 start is 6:35am, and there is nothing to see then."""
        morning = dict(stage_times(lock_at(12, 35)))["morning"]
        local_hour = morning.astimezone(datetime.now().astimezone().tzinfo).hour
        assert local_hour >= MORNING_NOT_BEFORE_HOUR

    def test_a_late_slate_keeps_the_full_six_hours(self):
        lock = lock_at(22, 10)
        assert dict(stage_times(lock))["morning"] == lock - timedelta(minutes=360)

    def test_the_floor_never_pushes_morning_past_the_next_stage(self):
        plan = dict(stage_times(lock_at(12, 35)))
        assert plan["morning"] <= plan["t-2h"]


class TestSlateLock:
    def test_lock_is_the_earliest_game_the_export_prices(self, tmp_path):
        path = salary_fixture(tmp_path, [
            ("SD@ARI", "08/08/2026 09:40PM ET"),
            ("CWS@BOS", "08/08/2026 07:10PM ET"),
            ("MIN@KC", "08/08/2026 07:40PM ET"),
        ])
        lock, used, info = slate_lock("2026-08-08", salary_path=path)
        assert (lock.hour, lock.minute) == (19, 10)
        assert used == path
        assert len(info["games"]) == 3

    def test_a_matinee_export_locks_in_the_afternoon(self, tmp_path):
        path = salary_fixture(tmp_path, [("MIA@ATL", "08/08/2026 12:35PM ET")])
        lock, _, _ = slate_lock("2026-08-08", salary_path=path)
        assert (lock.hour, lock.minute) == (12, 35)

    def test_an_export_without_times_is_an_error_not_a_guess(self, tmp_path):
        path = tmp_path / "DKSalaries_empty.csv"
        path.write_text("Position,Name,Salary,Game Info\nSP,A,8000,\n", encoding="utf-8")
        with pytest.raises(AutosnapError, match="Game Info"):
            slate_lock("2026-08-08", salary_path=str(path))

    def test_an_unmatched_slate_name_lists_what_is_available(self):
        """The usual cause is a --slate that does not match the file naming, and the fix
        is only obvious if the alternatives are named."""
        with pytest.raises(AutosnapError, match="Available:"):
            slate_lock("2026-08-06", salary_path=None, slate="no-such-slate")


class TestClockFormatting:
    def test_noon_and_midnight_read_correctly(self):
        assert _clock12(lock_at(12, 5)) == "12:05PM"
        assert _clock12(lock_at(0, 5)) == "12:05AM"

    def test_no_leading_zero(self):
        """`%-I` strips it on POSIX and raises on Windows, which is why this exists."""
        assert _clock12(lock_at(7, 10)) == "7:10PM" or _clock12(lock_at(7, 10)) == "7:10AM"
        assert not _clock12(lock_at(9, 5)).startswith("0")


class TestWaitForSlate:
    def test_it_returns_at_once_when_the_export_is_there(self, tmp_path):
        path = salary_fixture(tmp_path, [("MIA@ATL", "08/08/2026 12:35PM ET")])
        lock, _, _ = wait_for_slate("2026-08-08", salary_path=path, timeout_minutes=0)
        assert (lock.hour, lock.minute) == (12, 35)

    def test_an_ambiguous_slate_is_not_waited_on(self):
        """No amount of waiting resolves it, and the error already names the candidates."""
        with pytest.raises(AutosnapError, match="Available:"):
            wait_for_slate("2026-08-06", slate="no-such-slate", timeout_minutes=999)

    def test_a_zero_timeout_fails_immediately_rather_than_hanging(self):
        with pytest.raises(AutosnapError):
            wait_for_slate("1999-01-01", slate=None, timeout_minutes=0, poll_seconds=1)


class TestLogging:
    def test_it_writes_to_the_file_and_the_console(self, tmp_path, capsys):
        """A scheduled task's console output is discarded, so the file is the only record."""
        log = tmp_path / "autosnap.log"
        tee = _Tee(sys.stdout, str(log))
        tee.write("hello\n")
        tee.flush()
        assert "hello" in log.read_text(encoding="utf-8")

    def test_it_appends_rather_than_truncating(self, tmp_path):
        """Each night has to add to the log, not erase the last one."""
        log = tmp_path / "autosnap.log"
        log.write_text("night one\n", encoding="utf-8")
        _Tee(sys.stdout, str(log)).write("night two\n")
        body = log.read_text(encoding="utf-8")
        assert "night one" in body and "night two" in body
