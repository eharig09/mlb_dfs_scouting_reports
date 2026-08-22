"""The optimizer -> portfolio bridge, and the started-game filter that guards every path.

The bridge is what makes the new machinery usable in production: `dfs.optimize` carries the
constraints the workflow depends on (pool file, exposure minimums, ranged boosts, explicit
stacks, pins) and `dfs.portfolio` carries correlation-aware selection. Neither has to grow
the other's features if the two can hand work to each other.
"""

import pathlib

import numpy as np
import pandas as pd
import pytest

from dfs.candidates import CandidateError, from_optimizer, to_optimizer_format
from dfs.optimizer import ROSTER_SIZE, optimize
from dfs.slate import drop_started


def write_optimizer_csv(path, lineups):
    """Reproduce exactly what `dfs.optimize` writes, so the reader is tested against the
    real format rather than against a convenient one."""
    rows = []
    for i, lineup in enumerate(lineups, start=1):
        for _, row in lineup["players"].iterrows():
            rows.append({
                "Lineup": i, "Roster": row["Roster"], "Order": row.get("Slot"),
                "Name": row["Name"], "Team": row["Team"], "Opp": row.get("Opp"),
                "Salary": row["Salary"], "Proj": row["Proj"], "Ceiling": row["Ceiling"],
                "Floor": row.get("Floor"), "Tier": row.get("Tier"),
                "Role": row.get("Role"), "DK ID": row.get("DK ID"),
                "Lineup Salary": lineup["salary"], "Lineup Proj": lineup["proj"],
                "Lineup Ceiling": lineup["ceiling"],
            })
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


@pytest.fixture(scope="module")
def optimizer_output(tmp_path_factory):
    from conftest import make_slate
    players = make_slate(n_games=6, seed=1)
    lineups, _pool, _missing = optimize(players, n_lineups=25, seed=3)
    path = write_optimizer_csv(tmp_path_factory.mktemp("bridge") / "lineups_main.csv",
                               lineups)
    return players, lineups, path


class TestFromOptimizer:
    def test_reads_every_lineup(self, optimizer_output):
        players, lineups, path = optimizer_output
        pool = from_optimizer(players, path=path)
        assert len(pool) == len(lineups)

    def test_players_match_the_source(self, optimizer_output):
        players, lineups, path = optimizer_output
        pool = from_optimizer(players, path=path)
        for position, lineup in enumerate(lineups):
            got = set(pool.lineups.iloc[position]["names"])
            assert got == set(lineup["players"]["Name"])

    def test_metrics_match_the_source(self, optimizer_output):
        players, lineups, path = optimizer_output
        pool = from_optimizer(players, path=path)
        for position, lineup in enumerate(lineups):
            row = pool.lineups.iloc[position]
            assert row["salary"] == lineup["salary"]
            assert row["proj"] == pytest.approx(lineup["proj"], abs=0.02)
            assert row["ceiling"] == pytest.approx(lineup["ceiling"], abs=0.02)

    def test_every_lineup_is_complete(self, optimizer_output):
        players, _lineups, path = optimizer_output
        pool = from_optimizer(players, path=path)
        for indices in pool.lineups["players"]:
            assert len(indices) == ROSTER_SIZE
            assert len(set(indices)) == ROSTER_SIZE

    def test_source_lineup_numbers_are_kept(self, optimizer_output):
        """Without these the selection cannot be written back for upload."""
        players, lineups, path = optimizer_output
        pool = from_optimizer(players, path=path)
        assert "source_lineup" in pool.lineups.columns
        assert sorted(pool.lineups["source_lineup"]) == list(range(1, len(lineups) + 1))

    def test_construction_metrics_are_derived(self, optimizer_output):
        players, _lineups, path = optimizer_output
        pool = from_optimizer(players, path=path)
        for column in ("stack_shape", "primary_stack", "pitcher_pair", "own_sum", "games"):
            assert column in pool.lineups.columns

    def test_the_pool_can_be_simulated_and_scored(self, optimizer_output):
        """The whole point: these have to flow into the same machinery as generated ones."""
        from dfs.simulate import simulate_slate
        players, _lineups, path = optimizer_output
        pool = from_optimizer(players, path=path)
        sims = simulate_slate(pool.pool, n_sims=200, seed=1)
        scores = pool.score(sims)
        assert scores.shape == (200, len(pool))
        first = list(pool.lineups["players"].iloc[0])
        assert np.allclose(scores[:, 0], sims[:, first].sum(axis=1), atol=1e-3)

    def test_a_missing_file_says_what_to_run(self, slate, tmp_path):
        with pytest.raises(CandidateError, match="dfs.optimize"):
            from_optimizer(slate, date="2026-08-01", slate=None, root=str(tmp_path))

    def test_finds_the_only_slate_when_none_was_named(self, optimizer_output, tmp_path):
        """A caller who omits --slate looked for `lineups_unknown.csv`. The lineups almost
        always exist under the DK export's own label, and telling them to re-run a
        six-minute solve they have already done is the wrong answer."""
        players, lineups, path = optimizer_output
        night = tmp_path / "2026-08-01"
        night.mkdir()
        (night / "lineups_early.csv").write_bytes(pathlib.Path(path).read_bytes())
        pool = from_optimizer(players, date="2026-08-01", slate=None, root=str(tmp_path))
        assert len(pool) == len(lineups)

    def test_refuses_to_guess_between_several_slates(self, optimizer_output, tmp_path):
        players, _lineups, path = optimizer_output
        night = tmp_path / "2026-08-01"
        night.mkdir()
        for label in ("early", "main"):
            (night / f"lineups_{label}.csv").write_bytes(pathlib.Path(path).read_bytes())
        with pytest.raises(CandidateError, match="pass --slate"):
            from_optimizer(players, date="2026-08-01", slate=None, root=str(tmp_path))

    def test_a_non_optimizer_csv_is_rejected(self, slate, tmp_path):
        path = tmp_path / "wrong.csv"
        pd.DataFrame({"Foo": [1], "Bar": [2]}).to_csv(path, index=False)
        with pytest.raises(CandidateError, match="not an optimizer lineup file"):
            from_optimizer(slate, path=str(path))

    def test_a_slate_that_has_moved_on_is_reported(self, optimizer_output):
        """Rebuild the slate under different names and the mismatch must be named, not
        silently produce an empty or partial pool."""
        players, _lineups, path = optimizer_output
        renamed = players.copy()
        renamed["Name"] = renamed["Name"] + " Different"
        with pytest.raises(CandidateError, match="could be matched to the current slate"):
            from_optimizer(renamed, path=path)

    def test_the_mismatch_names_the_players(self, optimizer_output):
        players, _lineups, path = optimizer_output
        renamed = players.copy()
        renamed["Name"] = renamed["Name"] + " Different"
        with pytest.raises(CandidateError) as caught:
            from_optimizer(renamed, path=path)
        assert "Unmatched players include" in str(caught.value)


class TestToOptimizerFormat:
    def test_round_trips_to_an_uploadable_frame(self, optimizer_output):
        players, _lineups, path = optimizer_output
        pool = from_optimizer(players, path=path)
        selection = [0, 3, 7]
        out = to_optimizer_format(pool, selection, path)
        # dfs.upload requires exactly these two columns to recognise the file.
        assert {"Lineup", "Roster"} <= set(out.columns)
        assert sorted(out["Lineup"].unique()) == [1, 2, 3]
        assert len(out) == len(selection) * ROSTER_SIZE

    def test_renumbers_in_selection_order(self, optimizer_output):
        players, _lineups, path = optimizer_output
        pool = from_optimizer(players, path=path)
        selection = [5, 1]
        out = to_optimizer_format(pool, selection, path)
        first = set(out[out["Lineup"] == 1]["Name"])
        assert first == set(pool.lineups.iloc[5]["names"])

    def test_preserves_the_original_columns(self, optimizer_output):
        players, _lineups, path = optimizer_output
        pool = from_optimizer(players, path=path)
        original = pd.read_csv(path, encoding="utf-8-sig")
        out = to_optimizer_format(pool, [0, 1], path)
        assert list(out.columns) == [c.strip() for c in original.columns]

    def test_a_generated_pool_cannot_be_written_back(self, slate, tmp_path):
        """Only lineups that came from the optimizer carry the numbers needed to do it."""
        from dfs.candidates import generate
        pool = generate(slate, n_candidates=10, seed=1)
        with pytest.raises(CandidateError, match="did not come from dfs.optimize"):
            to_optimizer_format(pool, [0], str(tmp_path / "nothing.csv"))

    def test_dfs_upload_can_read_the_result(self, optimizer_output, tmp_path):
        """The acceptance test for the whole bridge."""
        from dfs.upload import read_lineups
        players, _lineups, path = optimizer_output
        pool = from_optimizer(players, path=path)
        target = tmp_path / "lineups_main.csv"
        to_optimizer_format(pool, [0, 2, 4], path).to_csv(target, index=False,
                                                          encoding="utf-8-sig")
        read_back = read_lineups("2026-08-01", path=str(target))
        assert sorted(read_back) == [1, 2, 3]
        assert all(len(frame) == ROSTER_SIZE for frame in read_back.values())


class TestStartedGameFilter:
    """It lived only in `dfs.optimize`, so the candidate, field, contest and portfolio paths
    all skipped it -- a portfolio built after first pitch could contain players who had
    already batted. These pin the shared behaviour."""

    def test_is_importable_from_both_places(self):
        from dfs.optimize import drop_started as from_cli
        from dfs.slate import drop_started as from_slate
        assert from_cli is from_slate

    def test_removes_players_from_a_started_game(self, slate, monkeypatch):
        from dfs.salaries import canon_team
        pairing = slate["Game"].iloc[0]
        away, _, home = pairing.partition("@")
        key = f"{canon_team(away)}@{canon_team(home)}"
        monkeypatch.setattr("dfs.schedule.started_pairings",
                            lambda date, **kw: ({key: [{"start": 1200, "started": True}]}, None))
        monkeypatch.setattr("dfs.schedule.pitching_elsewhere", lambda date, **kw: (set(), None))
        kept, notes = drop_started(slate, {"game_starts": {key: 1200}}, "2026-08-01")
        assert pairing not in set(kept["Game"])
        assert len(kept) < len(slate)
        assert any("already under way" in n for n in notes)

    def test_allow_started_keeps_everyone(self, slate, monkeypatch):
        from dfs.salaries import canon_team
        pairing = slate["Game"].iloc[0]
        away, _, home = pairing.partition("@")
        key = f"{canon_team(away)}@{canon_team(home)}"
        monkeypatch.setattr("dfs.schedule.started_pairings",
                            lambda date, **kw: ({key: [{"start": 1200, "started": True}]}, None))
        monkeypatch.setattr("dfs.schedule.pitching_elsewhere", lambda date, **kw: (set(), None))
        kept, notes = drop_started(slate, {"game_starts": {key: 1200}}, "2026-08-01",
                                   allow=True)
        assert len(kept) == len(slate)
        assert notes == []

    def test_a_fully_started_slate_is_left_alone(self, slate, monkeypatch):
        """A review or backtest, not a live build -- filtering would empty the pool."""
        from dfs.salaries import canon_team
        pairings = {}
        for pairing in slate["Game"].dropna().unique():
            away, _, home = str(pairing).partition("@")
            pairings[f"{canon_team(away)}@{canon_team(home)}"] = [
                {"start": 1200, "started": True}]
        monkeypatch.setattr("dfs.schedule.started_pairings", lambda date, **kw: (pairings, None))
        monkeypatch.setattr("dfs.schedule.pitching_elsewhere", lambda date, **kw: (set(), None))
        kept, notes = drop_started(slate, {"game_starts": {}}, "2026-08-01")
        assert len(kept) == len(slate)

    def test_a_schedule_failure_warns_and_keeps_everyone(self, slate, monkeypatch):
        monkeypatch.setattr("dfs.schedule.started_pairings",
                            lambda date, **kw: ({}, "StatsAPI unreachable"))
        kept, notes = drop_started(slate, {}, "2026-08-01")
        assert len(kept) == len(slate)
        assert any("could not check start times" in n for n in notes)

    def test_build_pipeline_accepts_the_flag(self):
        """Guards the wiring: the parameter must exist on the shared pipeline."""
        import inspect

        from dfs.contest import build_pipeline
        signature = inspect.signature(build_pipeline)
        assert "allow_started" in signature.parameters
        assert "from_optimizer" in signature.parameters


class TestTemplateFallback:
    """A night whose DK template was never downloaded is not a dead end.

    DK's bulk template is a fixed frame wrapped around the draft group's player list, and
    the salary export the board was priced from *is* that player list -- same ids, same
    slate. `dfs.optimize --upload` has always rebuilt one; `dfs.upload` used to give up.
    """

    def _salary_file(self, tmp_path):
        """A minimal DK salary export: the columns a template needs, and nothing else."""
        rows = [
            "Position,Name + ID,Name,ID,Roster Position,Salary,Game Info,TeamAbbrev,AvgPointsPerGame",
            "SP,Ace One (100),Ace One,100,P,9000,NYY@BOS 07:05PM ET,NYY,18.0",
            "SP,Ace Two (101),Ace Two,101,P,8000,NYY@BOS 07:05PM ET,BOS,17.0",
            "C,Catch Er (102),Catch Er,102,C,4000,NYY@BOS 07:05PM ET,NYY,8.0",
        ]
        path = tmp_path / "DKSalaries_2026-08-02_early.csv"
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return str(path)

    def test_rebuilds_a_bulk_template_from_the_salary_export(self, tmp_path):
        from dfs.upload import parse_template, template_from_salaries
        rows = template_from_salaries(self._salary_file(tmp_path))
        _template, kind, _slot_start, index = parse_template(rows, source="salaries")
        assert kind == "bulk"
        # Every id in the export must be reachable, because that is the whole point: the
        # ids written are DK's own for this draft group, not invented locally.
        found = set()
        for mapping in index:
            if isinstance(mapping, dict):
                found |= {str(v) for v in mapping.values()}
        assert {"100", "101", "102"} <= found

    def test_the_header_is_dks_slot_order(self, tmp_path):
        from dfs.upload import SLOT_ORDER, template_from_salaries
        rows = template_from_salaries(self._salary_file(tmp_path))
        assert rows[0][:len(SLOT_ORDER)] == list(SLOT_ORDER)

    def test_an_export_missing_the_player_list_is_refused(self, tmp_path):
        from dfs.upload import UploadError, template_from_salaries
        path = tmp_path / "bad.csv"
        path.write_text("Position,Name,Salary\nSP,Ace One,9000\n", encoding="utf-8")
        with pytest.raises(UploadError, match="cannot stand in for an upload template"):
            template_from_salaries(str(path))

    def test_an_explicit_template_is_never_substituted(self, tmp_path, monkeypatch):
        """Naming a file that then failed is a mistake worth surfacing, not papering over."""
        from dfs.upload import UploadError, resolve_template
        monkeypatch.setattr("dfs.upload.find_template",
                            lambda *a, **k: (_ for _ in ()).throw(UploadError("nope")))
        with pytest.raises(UploadError, match="nope"):
            resolve_template("2026-08-02", template="dk_lineups/whatever.csv")

    def test_ambiguity_asks_for_a_slate_rather_than_guessing(self, monkeypatch):
        from dfs.salaries import AmbiguousSlate
        from dfs.upload import UploadError, resolve_template

        def no_template(*a, **k):
            raise UploadError("no upload template")

        def ambiguous(*a, **k):
            raise AmbiguousSlate([{"name": "early"}, {"name": "main"}])

        monkeypatch.setattr("dfs.upload.find_template", no_template)
        monkeypatch.setattr("dfs.salaries.find_salary_file", ambiguous)
        with pytest.raises(UploadError, match="Pass --slate"):
            resolve_template("2026-08-02")


class TestExplicitTemplateResolution:
    def test_shorthand_is_narrowed_by_requested_date(self, monkeypatch):
        from dfs.upload import resolve_template_path

        old = "dk_lineups/DKTemplate_2026-08-07_turbo_entries.csv"
        current = "dk_lineups/DKTemplate_2026-08-09_turbo_entries.csv"
        dates = {old: {"2026-08-07"}, current: {"2026-08-09"}}
        monkeypatch.setattr("dfs.upload.os.path.exists", lambda _path: False)
        monkeypatch.setattr("dfs.upload.glob.glob", lambda _pattern: [old, current])
        monkeypatch.setattr("dfs.upload.template_dates_for", lambda path: dates[path])

        resolved = resolve_template_path(
            "turbo", directory="dk_lineups", date="2026-08-09")

        assert resolved == current
        assert resolved != old

    def test_explicit_wrong_date_is_rejected(self, monkeypatch):
        from dfs.upload import UploadError, find_template

        old = "dk_lineups/DKTemplate_2026-08-07_turbo_entries.csv"
        monkeypatch.setattr("dfs.upload.resolve_template_path", lambda *args, **kwargs: old)
        monkeypatch.setattr("dfs.upload.template_dates_for",
                            lambda _path: {"2026-08-07"})

        with pytest.raises(UploadError, match="not the requested date 2026-08-09"):
            find_template(old, directory="dk_lineups", date="2026-08-09")

    def test_identical_downloads_prefer_the_conventional_filename(self, monkeypatch):
        from dfs.upload import _dedupe_template_files

        generic = "dk_lineups/DKEntries (13).csv"
        filed = "dk_lineups/DKTemplate_2026-08-09_turbo_entries.csv"
        monkeypatch.setattr("dfs.upload.filecmp.cmp", lambda *args, **kwargs: True)

        assert _dedupe_template_files([generic, filed]) == [filed]


class TestSwapSlatePlumbing:
    """`--slate` has to reach the swap itself, not just the output filename.

    `run_swap` accepted a slate, used it to name the output, and did not forward it to
    `swap_file`. On a night with two DK exports that meant `--slate early --swap` resolved
    the swap against an ambiguous slate and failed with "several DK exports match" -- the
    exact error the flag was passed to prevent -- while still naming the output `early`.
    """

    def test_run_swap_forwards_the_slate(self, monkeypatch):
        from dfs import optimize as optimize_cli

        seen = {}

        def fake_swap_file(date, path=None, objective="ceiling", slate=None, **kwargs):
            seen["date"] = date
            seen["slate"] = slate
            seen["objective"] = objective
            return None, {}

        monkeypatch.setattr("dfs.lateswap.swap_file", fake_swap_file)
        monkeypatch.setattr("dfs.lateswap.format_report", lambda report: "")
        optimize_cli.run_swap("2026-08-02", "ceiling", slate="early")
        assert seen["slate"] == "early"
        assert seen["date"] == "2026-08-02"

    def test_lateswap_cli_and_run_swap_agree(self):
        """Both entry points must take the same parameter, or one of them is lying."""
        import inspect

        from dfs.lateswap import swap_file
        from dfs.optimize import run_swap
        assert "slate" in inspect.signature(swap_file).parameters
        assert "slate" in inspect.signature(run_swap).parameters
