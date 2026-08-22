"""Chunked Statcast caching: the season is assembled from immutable pieces.

The old cache keyed a season-to-date pull on (start, end). `end` moves every day, so every
run missed and re-downloaded the whole season -- ~170 HTTP requests and ~8 minutes -- to
gain one day of games, and left another ~685 MB pickle behind (34 of them had accumulated,
22 GB in total). Caching the *data* rather than the *request* makes the marginal cost of a
new day one request.

Everything here uses a synthetic fetcher that records its calls, so the suite runs offline
on a clean checkout. What is being tested is the chunking and cache-hit logic, not Savant.
"""

import pandas as pd
import pytest

import utils.cache as uc
from utils import statcast_cache as sc


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(uc, "CACHE_ROOT", tmp_path / ".cache")
    yield


class FakeStatcast:
    """Stands in for `pybaseball.statcast`: one row per day, and it counts its calls.

    `__name__` matters -- the cache key is built from it, so the double must name itself
    the way the real function does or the keys will not line up with production's.
    """
    __name__ = "statcast"

    def __init__(self, available_days=None):
        self.calls = []
        self.available = available_days  # None => every day has a game

    def __call__(self, start, end):
        self.calls.append((start, end))
        days = pd.date_range(start, end, freq="D")
        rows = [{"game_date": d.strftime("%Y-%m-%d"), "pitch": 1}
                for d in days
                if self.available is None or d.strftime("%Y-%m-%d") in self.available]
        return pd.DataFrame(rows, columns=["game_date", "pitch"])


def test_whole_months_chunk_by_month_and_the_tail_by_day():
    plan = sc.plan_chunks("2026-03-01", "2026-05-04")
    assert [(k, str(a), str(b)) for k, a, b in plan][:3] == [
        ("month", "2026-03-01", "2026-03-31"),
        ("month", "2026-04-01", "2026-04-30"),
        ("day", "2026-05-01", "2026-05-01"),
    ]
    assert all(k == "day" for k, _, _ in plan[2:])
    assert len(plan) == 2 + 4


def test_partial_leading_month_is_never_stored_under_a_month_key():
    """A range starting mid-month must not claim to hold the whole month.

    This is the failure that would corrupt the cache rather than merely slow it down: a
    March 15-31 pull filed under "March" would serve two silently missing weeks to every
    later request for the full month.
    """
    plan = sc.plan_chunks("2026-03-15", "2026-04-30")
    assert ("month", "2026-03-01") not in [(k, str(a)) for k, a, _ in plan]
    assert all(k == "day" for k, a, _ in plan if a.month == 3)
    assert ("month", "2026-04-01", "2026-04-30") in [(k, str(a), str(b)) for k, a, b in plan]


def test_assembles_the_full_range_and_fetches_each_chunk_once():
    fake = FakeStatcast()
    out = sc.load_statcast_range("2026-03-01", "2026-05-03", fetcher=fake)
    assert len(out) == 31 + 30 + 3
    assert out["game_date"].tolist() == sorted(out["game_date"].tolist())
    assert len(fake.calls) == 2 + 3          # two months, three days


def test_second_run_touches_the_network_not_at_all():
    fake = FakeStatcast()
    first = sc.load_statcast_range("2026-03-01", "2026-05-03", fetcher=fake)
    fake.calls.clear()
    second = sc.load_statcast_range("2026-03-01", "2026-05-03", fetcher=fake)
    assert fake.calls == []
    pd.testing.assert_frame_equal(first, second)


def test_one_more_day_costs_exactly_one_request():
    """The whole point: yesterday's cache is still good today."""
    fake = FakeStatcast()
    sc.load_statcast_range("2026-03-01", "2026-05-03", fetcher=fake)
    fake.calls.clear()
    out = sc.load_statcast_range("2026-03-01", "2026-05-04", fetcher=fake)
    assert fake.calls == [("2026-05-04", "2026-05-04")]
    assert len(out) == 31 + 30 + 4


def test_completed_month_rolls_up_from_day_chunks_without_refetching():
    """When the live month ends, its days are combined locally rather than re-downloaded.

    Without this the marginal cost resets to a full month every 30 days.
    """
    fake = FakeStatcast()
    sc.load_statcast_range("2026-03-01", "2026-04-29", fetcher=fake)   # April still partial
    fake.calls.clear()
    out = sc.load_statcast_range("2026-03-01", "2026-05-02", fetcher=fake)
    # Only the days April never covered: Apr 30, then May 1-2.
    assert sorted(fake.calls) == [("2026-04-30", "2026-04-30"),
                                  ("2026-05-01", "2026-05-01"),
                                  ("2026-05-02", "2026-05-02")]
    assert len(out) == 31 + 30 + 2
    # And April is now a single month chunk, so the next run reads one file, not thirty.
    assert sc._chunk_is_cached(fake, sc._to_date("2026-04-01"), sc._to_date("2026-04-30"))


def test_a_day_with_no_games_yet_is_not_frozen_empty():
    """The live edge of the season must stay refetchable.

    A day whose games have not posted returns zero rows. Caching that would drop the day
    permanently -- which is exactly what happened to 4,289 real rows during the migration.
    """
    today = pd.Timestamp.today().normalize()
    yesterday = (today - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    fake = FakeStatcast(available_days=set())          # nothing has posted
    sc.load_statcast_range(yesterday, yesterday, fetcher=fake)
    assert not sc._chunk_is_cached(fake, sc._to_date(yesterday), sc._to_date(yesterday))

    fake.available = {yesterday}                        # games post later
    out = sc.load_statcast_range(yesterday, yesterday, fetcher=fake)
    assert len(out) == 1


def test_a_long_past_day_with_no_games_is_remembered():
    """An off-day or the All-Star break is permanently gameless.

    It still has to be cached: an empty Savant pull is slow (~50s measured), so refetching
    every off-day on every run would give back a chunk of what chunking just saved.
    """
    fake = FakeStatcast(available_days=set())
    sc.load_statcast_range("2026-03-10", "2026-03-10", fetcher=fake)
    assert sc._chunk_is_cached(fake, sc._to_date("2026-03-10"), sc._to_date("2026-03-10"))
    fake.calls.clear()
    sc.load_statcast_range("2026-03-10", "2026-03-10", fetcher=fake)
    assert fake.calls == []


def test_refresh_days_repulls_and_drops_the_covering_month():
    """Savant occasionally revises past days; chunk caching would otherwise hide that."""
    fake = FakeStatcast()
    sc.load_statcast_range("2026-03-01", "2026-04-02", fetcher=fake)
    assert sc._chunk_is_cached(fake, sc._to_date("2026-03-01"), sc._to_date("2026-03-31"))

    fake.calls.clear()
    sc.refresh_days("2026-03-05", "2026-03-06", fetcher=fake)
    assert sorted(fake.calls) == [("2026-03-05", "2026-03-05"), ("2026-03-06", "2026-03-06")]
    # March's month chunk is stale now, so it must be gone.
    assert not sc._chunk_is_cached(fake, sc._to_date("2026-03-01"), sc._to_date("2026-03-31"))


def test_force_bypasses_the_chunk_cache_entirely():
    fake = FakeStatcast()
    sc.load_statcast_range("2026-03-01", "2026-03-05", fetcher=fake)
    fake.calls.clear()
    sc.load_statcast_range("2026-03-01", "2026-03-05", fetcher=fake, force=True)
    assert fake.calls == [("2026-03-01", "2026-03-05")]


def test_empty_range_is_empty_not_an_error():
    fake = FakeStatcast()
    assert sc.load_statcast_range("2026-05-05", "2026-05-01", fetcher=fake).empty
    assert fake.calls == []


def test_untouched_month_is_fetched_wholesale_not_day_by_day():
    """The other side of the rollup threshold.

    Backfilling a historical month that has no day chunks at all should cost one request,
    not thirty. Only a month that was recently the *live* month is worth stitching.
    """
    fake = FakeStatcast()
    sc.load_statcast_range("2026-03-01", "2026-03-31", fetcher=fake)
    assert fake.calls == [("2026-03-01", "2026-03-31")]
