"""Opponent field simulation.

Two families of test. The first is *legality* -- every simulated opponent must be a lineup
DK would accept, because an illegal field silently corrupts every rank and every EV computed
against it. The second is *structure* -- the field must reproduce the conditional behaviour
measured in `dk_results/`, since reproducing marginal ownership alone is exactly what the
old model already did and exactly what is not enough.
"""

from collections import Counter

import numpy as np
import pytest

from dfs.field import (HITTER_SEATS, PRIMARY_SIZE_SHARE, ContestConfig, FieldError,
                       FieldPool, _seatable, duplication, field_ownership, field_shapes,
                       parse_contest_lineup, seat_mask, simulate_field)
from dfs.optimizer import MAX_HITTERS_PER_TEAM, ROSTER, ROSTER_SIZE
from dfs.salaries import canon_team
from dfs.scoring import DK_SALARY_CAP


@pytest.fixture(scope="module")
def field(request):
    from conftest import make_slate
    players = make_slate(n_games=6, seed=1)
    config = ContestConfig("test", entries=1200, max_entries_per_user=20)
    entries, pool = simulate_field(players, config=config, seed=7)
    return entries, pool


class TestSeatMatching:
    def test_a_full_legal_set_fits(self):
        masks = [seat_mask({s}) for s in HITTER_SEATS]
        assert _seatable(masks)

    def test_three_shortstops_do_not_fit(self):
        """The whole reason this is a matching and not a count."""
        assert not _seatable([seat_mask({"SS"})] * 3)

    def test_two_shortstops_do_not_fit(self):
        assert not _seatable([seat_mask({"SS"})] * 2)

    def test_four_outfielders_do_not_fit(self):
        assert _seatable([seat_mask({"OF"})] * 3)
        assert not _seatable([seat_mask({"OF"})] * 4)

    def test_multi_position_players_are_placed_flexibly(self):
        # Three players who can each play SS or 2B: two fit, three do not.
        flexible = seat_mask({"SS", "2B"})
        assert _seatable([flexible, flexible])
        assert not _seatable([flexible, flexible, flexible])

    def test_an_all_outfield_stack_is_seatable(self):
        """The bug that suppressed five-stacks: seats were sliced, dropping the OF spots."""
        masks = [seat_mask({"OF"})] * 3 + [seat_mask({"C"}), seat_mask({"1B"})]
        assert _seatable(masks)

    def test_more_than_eight_hitters_never_fits(self):
        assert not _seatable([seat_mask({"OF", "1B", "2B", "3B", "SS", "C"})] * 9)

    def test_memoisation_returns_the_same_answer(self):
        masks = [seat_mask({"OF"}), seat_mask({"SS"})]
        first = _seatable(masks)
        assert _seatable(list(reversed(masks))) == first


class TestParsing:
    def test_reads_dks_run_together_format(self):
        text = ("1B Michael Busch 2B Nico Hoerner 3B Alex Bregman C Carson Kelly "
                "OF Pete Crow-Armstrong OF A.J. Ewing OF Jasson Dominguez "
                "P Payton Tolle P Walbert Urena SS Francisco Lindor")
        parsed = parse_contest_lineup(text)
        assert len(parsed) == 10
        assert ("1B", "Michael Busch") in parsed
        assert sum(1 for slot, _ in parsed if slot == "P") == 2
        assert sum(1 for slot, _ in parsed if slot == "OF") == 3

    def test_hyphenated_and_punctuated_names_survive(self):
        parsed = dict((n, s) for s, n in parse_contest_lineup(
            "OF Pete Crow-Armstrong OF A.J. Ewing"))
        assert "Pete Crow-Armstrong" in parsed
        assert "A.J. Ewing" in parsed

    def test_empty_input(self):
        assert parse_contest_lineup("") == []
        assert parse_contest_lineup(None) == []


class TestLegality:
    def test_every_entry_is_a_legal_dk_lineup(self, field):
        entries, pool = field
        assert len(entries) == 1200
        for entry in entries:
            assert len(entry) == ROSTER_SIZE
            assert len(set(entry)) == ROSTER_SIZE, "a player appears twice"
            assert pool.salary[list(entry)].sum() <= DK_SALARY_CAP

            frame = pool.players.loc[list(entry)]
            hitters = frame[frame["Type"] == "H"]
            pitchers = frame[frame["Type"] == "P"]
            assert len(pitchers) == ROSTER["P"]
            assert len(hitters) == len(HITTER_SEATS)
            assert hitters["Team"].map(canon_team).value_counts().max() <= MAX_HITTERS_PER_TEAM
            assert frame["Game"].nunique() >= 2

    def test_every_entry_can_be_seated(self, field):
        entries, pool = field
        for entry in entries:
            masks = [pool.seat_masks[i] for i in entry if pool.is_hitter[i]]
            assert _seatable(masks)

    def test_entries_spend_most_of_the_cap(self, field):
        """A field that leaves money unspent is not the field you are playing against."""
        entries, pool = field
        spend = np.mean([pool.salary[list(e)].sum() for e in entries])
        assert spend / DK_SALARY_CAP > 0.95


class TestStructure:
    def test_stack_sizes_resemble_the_real_field(self, field):
        entries, pool = field
        _shapes, primary = field_shapes(entries, pool)
        # Five-stacks dominate, and lineups with no stack are rare. Loose bounds: the point
        # is that the generative model produces the right *shape* of distribution.
        assert primary.get(5, 0) > 0.35, "five-stacks should dominate the field"
        assert primary.get(4, 0) > 0.10
        assert primary.get(1, 0) < 0.10, "almost every real entry stacks something"

    def test_teammates_are_correlated_far_above_independence(self, field):
        """The measured 2.6x lift is the thing an independent model cannot produce."""
        entries, pool = field
        n = len(entries)
        membership = {}
        for index, entry in enumerate(entries):
            for player in entry:
                membership.setdefault(player, set()).add(index)

        hitters = [i for i in pool.hitters if len(membership.get(int(i), ())) >= 40]
        same, different = [], []
        for a in hitters:
            for b in hitters:
                if a == b:
                    continue
                marginal = len(membership[int(b)]) / n
                if marginal <= 0:
                    continue
                conditional = len(membership[int(a)] & membership[int(b)]) / \
                    len(membership[int(a)])
                (same if pool.teams[a] == pool.teams[b] else different).append(
                    conditional / marginal)
        assert same and different
        assert np.mean(same) > 1.8, f"teammate lift only {np.mean(same):.2f}"
        assert np.mean(same) > np.mean(different) * 2

    def test_a_pitcher_against_his_own_stack_is_rare(self, field):
        """Loose here on purpose: the fixture prices only 12 pitchers, so the fallback that
        relaxes the rule rather than failing to build a lineup fires far more often than it
        would on a real slate. The tight bound is in the integration test below."""
        entries, pool = field
        conflicted = 0
        for entry in entries:
            counts = Counter(pool.teams[i] for i in entry if pool.is_hitter[i])
            if any(counts.get(pool.opponents[i], 0) >= 3
                   for i in entry if not pool.is_hitter[i]):
                conflicted += 1
        assert conflicted / len(entries) < 0.12

    def test_ownership_is_reproduced(self, field):
        entries, pool = field
        simulated = field_ownership(entries, pool)
        assert np.nanmean(np.abs(simulated - pool.own)) < 6.0

    def test_ownership_sums_to_the_roster_slots(self, field):
        entries, pool = field
        simulated = field_ownership(entries, pool)
        assert simulated[pool.is_hitter].sum() == pytest.approx(800, abs=1.0)
        assert simulated[~pool.is_hitter].sum() == pytest.approx(200, abs=1.0)


class TestDuplication:
    def test_a_multi_entry_field_duplicates_at_all(self, field):
        """A field that never duplicates cannot price duplication risk.

        The share is field-size dependent -- collisions grow with entries -- so the unit
        test only asserts that duplication happens. The calibrated 92-93% figure is checked
        against a real slate at real field size in the integration test.
        """
        entries, _pool = field
        counts = duplication(entries)
        assert max(counts.values()) > 1, "no two entries were ever the same"
        unique_share = sum(1 for v in counts.values() if v == 1) / len(entries)
        assert unique_share > 0.80

    def test_a_single_entry_field_barely_duplicates(self):
        from conftest import make_slate
        players = make_slate(n_games=6, seed=1)
        entries, _pool = simulate_field(
            players, config=ContestConfig("se", entries=400, max_entries_per_user=1,
                                          block_reuse=0.0), seed=3)
        counts = duplication(entries)
        unique_share = sum(1 for v in counts.values() if v == 1) / len(entries)
        assert unique_share > 0.95


class TestReproducibility:
    def test_same_seed_same_field(self, slate):
        config = ContestConfig("t", entries=200, max_entries_per_user=20)
        a, _ = simulate_field(slate, config=config, seed=11)
        b, _ = simulate_field(slate, config=config, seed=11)
        assert [sorted(e) for e in a] == [sorted(e) for e in b]

    def test_different_seeds_differ(self, slate):
        config = ContestConfig("t", entries=200, max_entries_per_user=20)
        a, _ = simulate_field(slate, config=config, seed=11)
        b, _ = simulate_field(slate, config=config, seed=12)
        assert [sorted(e) for e in a] != [sorted(e) for e in b]

    def test_requested_size_is_produced(self, slate):
        for size in (50, 137):
            entries, _ = simulate_field(
                slate, config=ContestConfig("t", entries=size, max_entries_per_user=20),
                seed=1)
            assert len(entries) == size


class TestGuards:
    def test_a_slate_without_ownership_is_refused(self, slate):
        stripped = slate.drop(columns=["Own%"])
        with pytest.raises(FieldError, match="Own%"):
            FieldPool(stripped)

    def test_a_slate_with_no_pitchers_is_refused(self, slate):
        hitters_only = slate[slate["Type"] == "H"]
        with pytest.raises(FieldError, match="pitchers"):
            FieldPool(hitters_only)

    def test_an_empty_slate_is_refused(self, slate):
        with pytest.raises(FieldError):
            FieldPool(slate.head(0))


@pytest.mark.integration
class TestAgainstRealSlates:
    """The tight calibration checks, on a real slate at a real field size."""

    def _real_slate(self, cached_dates):
        """A priced real slate, preferring one whose ownership was actually measured.

        The newest cached date is usually *today*, whose games have not been played, so its
        Own% is the model's own estimate -- and the field simulator reproducing an estimate
        is a much weaker statement than it reproducing DK's %Drafted. It is also a marginal
        one: on an estimated slate the correlation below sits within a seed's noise of the
        threshold (0.553 measured over five seeds), so which side of it a run lands on says
        nothing. On a finished slate the same check comes back at 0.91.
        """
        from dfs.slate import build_slate
        fallback = None
        for date in reversed(cached_dates):
            for label in ("main", None):
                try:
                    players, _, meta = build_slate(date, slate=label, check_schedule=False)
                except (NameError, AttributeError, TypeError, ImportError):
                    # A broken build_slate is a failure, not a date to skip. Swallowing these
                    # turned a NameError in dfs.results into "3 skipped" and a green run.
                    raise
                except Exception:
                    continue
                if players.empty or not meta.get("has_salary") or len(players) <= 120:
                    continue
                if meta.get("ownership_source"):
                    return players
                if fallback is None:
                    fallback = players
        return fallback

    def test_ownership_correlates_strongly(self, cached_dates):
        if not cached_dates:
            pytest.skip("no cached report data")
        players = self._real_slate(cached_dates)
        if players is None:
            pytest.skip("no priced real slate")
        entries, pool = simulate_field(
            players, config=ContestConfig("t", entries=2000, max_entries_per_user=20),
            seed=7)
        simulated = field_ownership(entries, pool)
        assert np.corrcoef(simulated, pool.own)[0, 1] > 0.55
        assert np.nanmean(np.abs(simulated - pool.own)) < 4.0

    def test_conflict_rate_matches_the_measured_one(self, cached_dates):
        if not cached_dates:
            pytest.skip("no cached report data")
        players = self._real_slate(cached_dates)
        if players is None:
            pytest.skip("no priced real slate")
        entries, pool = simulate_field(
            players, config=ContestConfig("t", entries=2000, max_entries_per_user=20),
            seed=7)
        conflicted = 0
        for entry in entries:
            counts = Counter(pool.teams[i] for i in entry if pool.is_hitter[i])
            if any(counts.get(pool.opponents[i], 0) >= 3
                   for i in entry if not pool.is_hitter[i]):
                conflicted += 1
        # Measured across dk_results: 0.5% of real entries.
        assert conflicted / len(entries) < 0.03

    def test_stack_shares_track_the_measured_field(self, cached_dates):
        if not cached_dates:
            pytest.skip("no cached report data")
        players = self._real_slate(cached_dates)
        if players is None:
            pytest.skip("no priced real slate")
        entries, pool = simulate_field(
            players, config=ContestConfig("t", entries=2000, max_entries_per_user=20),
            seed=7)
        _shapes, primary = field_shapes(entries, pool)
        # Checked against the constants, never against a hardcoded floor. PRIMARY_SIZE_SHARE
        # is recalibrated from dk_results/ and moves when it does: the 2026-08-08 refresh
        # took five-stacks from 53.7% to 45.0%, and a bare `> 0.40` then failed while the
        # simulator was tracking its target exactly as well as before.
        #
        # The 0.08 band is not slack, it is a known bias: the repair step trades some
        # five-stacks down, so size 5 lands ~5pp under target and sizes 3-4 a little over.
        # Tighten this only along with a fix to that, not on its own.
        for size in (3, 4, 5):
            assert abs(primary.get(size, 0) - PRIMARY_SIZE_SHARE[size]) < 0.08, (
                f"{size}-stacks simulated at {primary.get(size, 0):.3f} against a measured "
                f"{PRIMARY_SIZE_SHARE[size]:.3f}")
        # Whatever the target, five-stacks must still be the field's commonest shape.
        assert primary.get(5, 0) == max(primary.values())
