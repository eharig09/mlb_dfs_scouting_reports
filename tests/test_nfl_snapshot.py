"""The precomputed snapshot the hosted dashboard reads.

The property that matters is **the hosted condition**: no raw PFF exports, no nflverse
cache, no network. A page that works locally and dies on Render is the failure this whole
module exists to prevent, and the first dry run of it failed 9 of 13 pages.
"""

import json
import os

import pandas as pd
import pytest

from nfl import snapshot


class TestReadFallsBackQuietly:
    def test_a_missing_frame_is_none_not_an_error(self, tmp_path):
        """Every reader falls back to computing live, so a missing snapshot degrades to
        'slower locally' rather than 'broken in production'."""
        assert snapshot.read("receivers", 2025, root=str(tmp_path)) is None

    def test_a_corrupt_frame_is_none_too(self, tmp_path):
        path = snapshot.path_for("receivers", 2025, root=str(tmp_path))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("not parquet")
        assert snapshot.read("receivers", 2025, root=str(tmp_path)) is None

    def test_a_round_trip_preserves_awkward_column_names(self, tmp_path):
        """Column names come from the pages: spaces, percent signs and slashes."""
        frame = pd.DataFrame([{"Routes/G": 30.5, "Slot%": 62.1, "HV touches/G": 1.25}])
        snapshot._write(frame, "receivers", 2025, root=str(tmp_path))
        out = snapshot.read("receivers", 2025, root=str(tmp_path))
        assert list(out.columns) == ["Routes/G", "Slot%", "HV touches/G"]
        assert out["Slot%"].iloc[0] == 62.1

    def test_an_empty_frame_is_not_written(self, tmp_path):
        assert snapshot._write(pd.DataFrame(), "receivers", 2025, root=str(tmp_path)) is None

    def test_a_manifest_does_not_serve_an_unlisted_stale_file(self, tmp_path):
        root = str(tmp_path)
        snapshot._write(pd.DataFrame({"value": [1]}), "receivers", 2025, root=root)
        with open(os.path.join(root, snapshot.MANIFEST), "w", encoding="utf-8") as handle:
            json.dump({"files": []}, handle)
        assert snapshot.read("receivers", 2025, root=root) is None

    def test_a_checksum_mismatch_is_not_served(self, tmp_path):
        root = str(tmp_path)
        path = snapshot._write(pd.DataFrame({"value": [1]}), "receivers", 2025, root=root)
        name = os.path.basename(path)
        with open(os.path.join(root, snapshot.MANIFEST), "w", encoding="utf-8") as handle:
            json.dump({"files": [name], "sha256": {name: "not-the-real-digest"}}, handle)
        assert snapshot.read("receivers", 2025, root=root) is None

    def test_a_generation_pointer_selects_only_that_release(self, tmp_path):
        root = str(tmp_path)
        release = os.path.join(root, snapshot.RELEASES, "generation-2")
        snapshot._write(pd.DataFrame({"value": [2]}), "receivers", 2025, root=release)
        with open(os.path.join(release, snapshot.MANIFEST), "w", encoding="utf-8") as handle:
            json.dump({
                "format_version": snapshot.FORMAT_VERSION,
                "generation": "generation-2",
                "files": ["receivers_2025.parquet"],
            }, handle)
        with open(os.path.join(root, snapshot.CURRENT), "w", encoding="utf-8") as handle:
            json.dump({"generation": "generation-2"}, handle)

        assert snapshot.read("receivers", 2025, root=root)["value"].tolist() == [2]

    def test_a_generation_without_a_valid_manifest_is_not_served(self, tmp_path):
        root = str(tmp_path)
        release = os.path.join(root, snapshot.RELEASES, "generation-2")
        snapshot._write(pd.DataFrame({"value": [2]}), "receivers", 2025, root=release)
        with open(os.path.join(root, snapshot.CURRENT), "w", encoding="utf-8") as handle:
            json.dump({"generation": "generation-2"}, handle)

        assert snapshot.read("receivers", 2025, root=root) is None

    def test_a_broken_generation_pointer_does_not_fall_back_to_flat_files(self, tmp_path):
        root = str(tmp_path)
        snapshot._write(pd.DataFrame({"value": [1]}), "receivers", 2025, root=root)
        with open(os.path.join(root, snapshot.CURRENT), "w", encoding="utf-8") as handle:
            json.dump({"generation": "../legacy"}, handle)

        assert snapshot.read("receivers", 2025, root=root) is None


def _stub_snapshot_build(monkeypatch):
    from dashboards import nfl_league, nfl_pff, nfl_slates
    from nfl import pffdata

    frame = lambda *args, **kwargs: pd.DataFrame({"value": [1]})
    catalog = pd.DataFrame([
        {"family": family, "usable": True, "season": 2025}
        for family in ("receiving_summary", "rushing_summary", "defense_coverage_scheme",
                       "slot_coverage", "receiving_scheme", "passing_depth",
                       "offense_blocking", "fantasy-stats-receiving")
    ])
    monkeypatch.setattr(pffdata, "catalog", lambda: catalog)
    for name in ("receivers", "rushers", "high_value", "target_distribution",
                 "defense_scheme", "defense_slot", "receiver_scheme", "quarterbacks"):
        monkeypatch.setattr(nfl_pff, name, frame)
    monkeypatch.setattr(nfl_pff, "lines", lambda season: (frame(), frame()))
    for name in ("defense_allowed", "team_offense", "team_results", "team_summary"):
        monkeypatch.setattr(nfl_league, name, frame)
    monkeypatch.setattr(nfl_slates, "projections", lambda: (frame(), "test"))
    return nfl_slates


class TestGenerationPublication:
    def test_a_complete_build_is_published_as_one_generation(self, tmp_path, monkeypatch):
        _stub_snapshot_build(monkeypatch)
        state = snapshot.build(seasons=[2025], root=str(tmp_path), report=lambda message: None)

        pointer = json.loads((tmp_path / snapshot.CURRENT).read_text(encoding="utf-8"))
        assert pointer["generation"] == state["generation"]
        assert state["format_version"] == snapshot.FORMAT_VERSION
        assert state["sha256"]
        assert snapshot.read("receivers", 2025, root=str(tmp_path)) is not None

    def test_a_failed_build_does_not_replace_the_current_generation(self, tmp_path,
                                                                    monkeypatch):
        nfl_slates = _stub_snapshot_build(monkeypatch)
        first = snapshot.build(seasons=[2025], root=str(tmp_path), report=lambda message: None)
        monkeypatch.setattr(nfl_slates, "projections", lambda: (pd.DataFrame(), "missing"))

        with pytest.raises(snapshot.SnapshotBuildError, match="projections.parquet"):
            snapshot.build(seasons=[2025], root=str(tmp_path), report=lambda message: None)

        assert snapshot.manifest(str(tmp_path))["generation"] == first["generation"]


class TestAvailableSeasons:
    def test_no_snapshot_returns_none_not_empty(self, tmp_path):
        """None and [] mean different things and the caller depends on it: None is 'no
        snapshot, go look at the raw exports'; [] is 'built, and that family had nothing'.

        Collapsing them is what made a hosted page decide there were no PFF exports and
        stop, on a machine that simply had not built a snapshot yet.
        """
        assert snapshot.available_seasons("receiving_summary", root=str(tmp_path)) is None

    def test_a_built_snapshot_answers_without_the_catalog(self, tmp_path):
        root = str(tmp_path)
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, snapshot.MANIFEST), "w", encoding="utf-8") as handle:
            json.dump({"built": "2026-01-01T00:00:00Z",
                       "families": {"receiving_summary": [2025, 2024],
                                    "passing_depth": []}}, handle)
        assert snapshot.available_seasons("receiving_summary", root=root) == [2025, 2024]
        assert snapshot.available_seasons("passing_depth", root=root) == []


class TestDescribe:
    def test_a_missing_snapshot_reports_absent_rather_than_raising(self, tmp_path):
        state = snapshot.describe(root=str(tmp_path))
        assert state["present"] is False and state["files"] == 0


class TestTheRealSnapshotIfBuilt:
    """Against whatever is committed, so a stale or broken snapshot fails here first."""

    def _manifest(self):
        state = snapshot.manifest()
        if not state:
            pytest.skip("no snapshot built")
        return state

    def test_it_records_which_seasons_each_family_covers(self):
        """Without this the hosted app asks `pffdata.catalog()`, which fingerprints raw
        exports and falls through to downloading nflverse rosters — a network call to
        answer 'which seasons do I have files for' on a machine with no files."""
        families = self._manifest().get("families")
        assert families, "manifest carries no family season index"
        assert families.get("receiving_summary")

    def test_the_frames_the_pages_need_are_present(self):
        state = self._manifest()
        newest = max(state["seasons"])
        for name in ("receivers", "target_distribution", "defense_scheme"):
            assert snapshot.read(name, newest) is not None, name
        for name in ("defense_allowed", "team_summary", "projections"):
            assert snapshot.read(name) is not None, name

    def test_it_stays_small_enough_to_commit(self):
        """The whole point is that it is a fraction of the 14 MB raw drop. If this fails,
        something is being snapshotted that should be derived on the page instead."""
        state = snapshot.describe()
        assert state["bytes"] < 20_000_000


class TestTheDashboardDoesNotPullTheSolver:
    def test_importing_dashboard_modules_leaves_scipy_alone(self):
        """`nfl/__init__.py` re-exported the solver eagerly, so `from nfl import naming` --
        just to build a file path -- pulled `scipy.optimize.milp` into the process. That is
        ~40 MB of wheel installed and imported to serve pages that never solve a lineup,
        and `requirements-web.txt` deliberately does not carry it.
        """
        import subprocess
        import sys

        code = (
            "import sys;"
            "import dashboards.nfl_pff, dashboards.nfl_league, dashboards.nfl_slates,"
            " dashboards.nfl_boards, dashboards.nfl_filters;"
            "print(any(m == 'scipy' or m.startswith('scipy.') for m in sys.modules))"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                text=True, timeout=180)
        assert result.stdout.strip().endswith("False"), result.stdout + result.stderr

    def test_the_public_api_still_resolves_lazily(self):
        import nfl
        assert nfl.ROSTER["QB"] == 1
        assert callable(nfl.eligible_positions)
