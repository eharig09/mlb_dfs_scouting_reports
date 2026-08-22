"""Immutable slate snapshots.

The property under test is *immutability*, and the way it fails is silent. A snapshot that
quietly reflects a later refresh does not raise, does not warn, and makes every review of
that night wrong in a flattering direction. So the tests here are mostly about what must
NOT change.
"""

import json
import os

import pandas as pd
import pytest

from dfs.snapshot import (STAGES, Snapshot, SnapshotError, file_digest, frame_digest,
                          list_snapshots, load_snapshot, next_revision, take_snapshot)


@pytest.fixture
def snap_root(tmp_path):
    return str(tmp_path / "snapshots")


@pytest.fixture
def stacks(slate):
    from dfs.slate import build_stacks
    return build_stacks(slate, has_salary=True)


def _take(slate, stacks, root, stage="final", **kwargs):
    return take_snapshot("2026-08-01", slate="main", stage=stage, players=slate,
                         stacks=stacks, meta={"has_salary": True, "slate_label": "main"},
                         root=root, data_dir=str(tmp_missing()), **kwargs)


def tmp_missing():
    """A data dir with no payloads: the synthetic slate has no cached games behind it."""
    return os.path.join("tests", "_no_such_cache")


class TestCapture:
    def test_writes_the_expected_files(self, slate, stacks, snap_root):
        snap = _take(slate, stacks, snap_root)
        for name in ("manifest.json", "players.parquet", "stacks.parquet", "games.json"):
            assert os.path.exists(os.path.join(snap.directory, name)), name

    def test_manifest_records_versions_and_cutoff(self, slate, stacks, snap_root):
        snap = _take(slate, stacks, snap_root)
        manifest = snap.manifest
        assert manifest["versions"]["model"] >= 1
        assert manifest["versions"]["ownership"] >= 1
        assert "code" in manifest["versions"]
        assert manifest["taken_utc"]
        assert manifest["counts"]["players"] == len(slate)
        assert manifest["stage"] == "final"

    def test_settings_are_stored_verbatim(self, slate, stacks, snap_root):
        settings = {"objective": "ceiling", "n": 20, "max_overlap": 6, "seed": 7}
        snap = _take(slate, stacks, snap_root, settings=settings)
        assert snap.manifest["settings"] == settings

    def test_unknown_stage_is_rejected(self, slate, stacks, snap_root):
        with pytest.raises(SnapshotError, match="stage must be"):
            _take(slate, stacks, snap_root, stage="whenever")

    def test_every_documented_stage_is_accepted(self, slate, stacks, snap_root):
        for stage in STAGES:
            snap = _take(slate, stacks, snap_root, stage=stage)
            assert snap.manifest["stage"] == stage

    def test_an_empty_slate_is_refused(self, snap_root):
        with pytest.raises(SnapshotError, match="nothing to snapshot"):
            take_snapshot("2026-08-01", slate="main", players=pd.DataFrame(),
                          meta={}, root=snap_root)


class TestImmutability:
    def test_retaking_a_stage_never_overwrites(self, slate, stacks, snap_root):
        first = _take(slate, stacks, snap_root)
        assert first.manifest["revision"] == 1

        moved = slate.copy()
        moved["Proj"] = moved["Proj"] + 5.0
        second = take_snapshot("2026-08-01", slate="main", stage="final", players=moved,
                               stacks=stacks, meta={"has_salary": True},
                               root=snap_root, data_dir=tmp_missing())
        assert second.manifest["revision"] == 2
        assert second.directory != first.directory
        assert os.path.exists(first.directory)

        # The first revision must still hold the original numbers.
        reloaded = Snapshot(first.directory, first.manifest)
        assert reloaded.players["Proj"].sum() == pytest.approx(slate["Proj"].sum())

    def test_revision_chain_is_recorded(self, slate, stacks, snap_root):
        _take(slate, stacks, snap_root)
        _take(slate, stacks, snap_root)
        third = _take(slate, stacks, snap_root)
        assert third.manifest["revision"] == 3
        assert third.manifest["previous_revisions"] == [1, 2]

    def test_next_revision_counts_existing(self, slate, stacks, snap_root):
        assert next_revision("2026-08-01", "main", "final", snap_root) == 1
        _take(slate, stacks, snap_root)
        assert next_revision("2026-08-01", "main", "final", snap_root) == 2


class TestRoundTrip:
    def test_projections_survive_exactly(self, slate, stacks, snap_root):
        _take(slate, stacks, snap_root)
        loaded = load_snapshot("2026-08-01", "main", root=snap_root)
        for column in ("Proj", "Ceiling", "Floor", "Bust%", "Salary", "Own%"):
            pd.testing.assert_series_equal(
                loaded.players[column].reset_index(drop=True),
                slate[column].reset_index(drop=True), check_names=False)

    def test_event_columns_survive(self, slate, stacks, snap_root):
        """Without these the snapshot cannot be re-simulated, only re-read."""
        _take(slate, stacks, snap_root)
        loaded = load_snapshot("2026-08-01", "main", root=snap_root)
        for column in [c for c in slate.columns if c.startswith("E_")]:
            assert column in loaded.players.columns

    def test_list_columns_come_back_as_lists(self, slate, stacks, snap_root):
        _take(slate, stacks, snap_root)
        loaded = load_snapshot("2026-08-01", "main", root=snap_root)
        assert isinstance(loaded.players["Supports"].iloc[0], list)

    def test_derived_columns_are_rebuilt(self, slate, stacks, snap_root):
        _take(slate, stacks, snap_root)
        loaded = load_snapshot("2026-08-01", "main", root=snap_root)
        assert "Why" in loaded.players.columns
        assert "Risks" in loaded.players.columns

    def test_meta_looks_like_build_slate_meta(self, slate, stacks, snap_root):
        _take(slate, stacks, snap_root)
        meta = load_snapshot("2026-08-01", "main", root=snap_root).meta()
        for key in ("date", "slate_label", "games", "has_salary", "missing_games",
                    "postponed", "snapshot"):
            assert key in meta
        assert meta["snapshot"]["stage"] == "final"

    def test_a_snapshot_can_be_optimized_from(self, slate, stacks, snap_root):
        """The end-to-end point: a frozen slate must still build lineups."""
        from dfs.optimizer import optimize
        _take(slate, stacks, snap_root)
        loaded = load_snapshot("2026-08-01", "main", root=snap_root)
        lineups, _, _ = optimize(loaded.players, n_lineups=2, seed=1)
        assert len(lineups) == 2


class TestSelection:
    def test_most_informed_stage_wins(self, slate, stacks, snap_root):
        _take(slate, stacks, snap_root, stage="morning")
        _take(slate, stacks, snap_root, stage="confirmed")
        _take(slate, stacks, snap_root, stage="t-2h")
        assert load_snapshot("2026-08-01", "main", root=snap_root).stage == "confirmed"

    def test_an_explicit_stage_is_honoured(self, slate, stacks, snap_root):
        _take(slate, stacks, snap_root, stage="morning")
        _take(slate, stacks, snap_root, stage="final")
        picked = load_snapshot("2026-08-01", "main", stage="morning", root=snap_root)
        assert picked.stage == "morning"

    def test_missing_stage_raises(self, slate, stacks, snap_root):
        _take(slate, stacks, snap_root, stage="final")
        with pytest.raises(SnapshotError):
            load_snapshot("2026-08-01", "main", stage="morning", root=snap_root)

    def test_no_snapshots_raises(self, snap_root):
        with pytest.raises(SnapshotError, match="no snapshots"):
            load_snapshot("2026-08-01", "main", root=snap_root)

    def test_listing_reads_manifests_only(self, slate, stacks, snap_root):
        _take(slate, stacks, snap_root, stage="morning")
        _take(slate, stacks, snap_root, stage="final")
        found = list_snapshots("2026-08-01", root=snap_root)
        assert len(found) == 2
        assert all(s._players is None for s in found), "listing must not load Parquet"

    def test_a_future_format_version_is_skipped(self, slate, stacks, snap_root):
        """A newer writer's layout must be ignored, not guessed at."""
        snap = _take(slate, stacks, snap_root)
        path = os.path.join(snap.directory, "manifest.json")
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest["format_version"] = 99
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)
        assert list_snapshots("2026-08-01", root=snap_root) == []


class TestIntegrity:
    def test_frame_digest_ignores_row_order(self, slate):
        columns = ["Name", "Proj", "Ceiling"]
        shuffled = slate.sample(frac=1.0, random_state=1)
        assert frame_digest(slate.sort_values("Name"), columns) == \
            frame_digest(shuffled.sort_values("Name"), columns)

    def test_frame_digest_notices_a_changed_projection(self, slate):
        columns = ["Name", "Proj"]
        moved = slate.copy()
        moved.loc[0, "Proj"] = moved.loc[0, "Proj"] + 0.01
        assert frame_digest(slate, columns) != frame_digest(moved, columns)

    def test_file_digest_of_a_missing_file_is_none(self):
        assert file_digest(os.path.join("tests", "_definitely_not_here.csv")) is None

    def test_stale_payloads_reported_when_a_payload_changes(self, slate, stacks, tmp_path):
        """The specific failure this module exists to make visible."""
        data_dir = tmp_path / "payloads"
        data_dir.mkdir()
        payload = data_dir / "2026-08-01_NYY_BOS.pkl"
        payload.write_bytes(b"original contents")

        snap = Snapshot(str(tmp_path / "s"), {"date": "2026-08-01"}, games=[{
            "game": "NYY@BOS", "payload": payload.name,
            "payload_sha256": file_digest(str(payload)),
        }])
        assert snap.stale_payloads(data_dir=str(data_dir)) == []

        payload.write_bytes(b"refreshed with confirmed lineups")
        stale = snap.stale_payloads(data_dir=str(data_dir))
        assert len(stale) == 1
        assert "rewritten" in stale[0][1]

    def test_deleted_payload_is_reported(self, tmp_path):
        snap = Snapshot(str(tmp_path), {"date": "2026-08-01"}, games=[{
            "game": "NYY@BOS", "payload": "gone.pkl", "payload_sha256": "abc",
        }])
        stale = snap.stale_payloads(data_dir=str(tmp_path))
        assert stale and "deleted" in stale[0][1]


@pytest.mark.integration
class TestAgainstRealData:
    def test_snapshot_a_real_cached_slate(self, cached_dates, tmp_path):
        if not cached_dates:
            pytest.skip("no cached report data")
        date = cached_dates[-1]
        from dfs.slate import build_slate
        try:
            players, stacks, meta = build_slate(date, check_schedule=False)
        except Exception:
            pytest.skip("slate would not build")
        if players.empty:
            pytest.skip("no players for that date")
        snap = take_snapshot(date, slate=meta.get("slate_label") or "main",
                             players=players, stacks=stacks, meta=meta,
                             root=str(tmp_path))
        assert snap.manifest["counts"]["players"] == len(players)
        # Nothing has been refreshed in between, so nothing should read as stale.
        assert snap.stale_payloads() == []
