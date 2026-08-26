"""Slates, pricing, stacks, defence and the entered-lineups reader."""

import os

import numpy as np
import pandas as pd
import pytest

from dashboards import nfl_league, nfl_slates, nfl_stacking

PAGES = ("nfl_value", "nfl_stacks", "nfl_defense", "nfl_teams", "nfl_entries")


def _page(name):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "dashboards", "app_pages", f"{name}.py")


BOARD = pd.DataFrame([
    {"Name": "QB One", "Pos": "QB", "Team": "PHI", "Opp": "WAS", "Salary": 6600,
     "Proj": 17.0, "Targets": np.nan},
    {"Name": "WR One", "Pos": "WR", "Team": "PHI", "Opp": "WAS", "Salary": 6600,
     "Proj": 12.0, "Targets": 8.0},
    {"Name": "WR Two", "Pos": "WR", "Team": "PHI", "Opp": "WAS", "Salary": 5100,
     "Proj": 9.0, "Targets": 6.0},
    {"Name": "TE One", "Pos": "TE", "Team": "PHI", "Opp": "WAS", "Salary": 4100,
     "Proj": 7.0, "Targets": 5.0},
    {"Name": "RB One", "Pos": "RB", "Team": "PHI", "Opp": "WAS", "Salary": 7000,
     "Proj": 14.0, "Targets": 3.0},
    {"Name": "WR Opp", "Pos": "WR", "Team": "WAS", "Opp": "PHI", "Salary": 5700,
     "Proj": 10.5, "Targets": 7.0},
])


class TestPriceBands:
    def test_a_salary_lands_in_exactly_one_band(self):
        assert nfl_slates.price_band(3400) == "Punt"
        assert nfl_slates.price_band(4800) == "Value"
        assert nfl_slates.price_band(6600) == "Mid"
        assert nfl_slates.price_band(8000) == "Stud"

    def test_a_missing_salary_is_blank_not_a_band(self):
        assert nfl_slates.price_band(np.nan) == ""


class TestValue:
    def test_value_is_points_per_thousand(self):
        out = nfl_slates.add_value(BOARD.assign(Band="Mid"))
        assert out.loc[0, "Value"] == pytest.approx(17.0 / 6.6, rel=1e-3)

    def test_band_rank_is_within_the_band_not_the_board(self):
        """Raw value always favours the cheapest man on the board, so a punt at replacement
        level outranks every real starter. Ranking inside a band asks the question a roster
        actually asks."""
        frame = BOARD.copy()
        frame["Band"] = frame["Salary"].map(nfl_slates.price_band)
        out = nfl_slates.add_value(frame)
        for band, group in out.groupby("Band"):
            assert set(group["Band rank"]) == set(range(1, len(group) + 1))

    def test_a_zero_salary_does_not_divide(self):
        out = nfl_slates.add_value(pd.DataFrame([{"Proj": 10.0, "Salary": 0, "Band": "x"}]))
        assert np.isnan(out.loc[0, "Value"])


class TestStackCombinations:
    def test_it_enumerates_every_size(self):
        combos = nfl_stacking.stack_combinations(BOARD, "PHI", min_projection=0.0)
        # 3 catchers -> 3 one-partner, 3 two-partner, 1 three-partner
        assert len(combos) == 7
        assert sorted(combos["Size"].unique()) == [2, 3, 4]

    def test_a_running_back_never_joins_his_own_stack(self):
        """A rushing touchdown is a drive that did *not* end in a passing touchdown, so a
        back is at best uncorrelated with his own quarterback."""
        combos = nfl_stacking.stack_combinations(BOARD, "PHI", min_projection=0.0)
        assert not any("RB One" in partners for partners in combos["Partners"])

    def test_captured_share_does_not_move_with_the_filter(self):
        """The denominator is the club's whole pass game, not the filtered shortlist.
        Taking it from the survivors made raising the floor inflate every share, and a
        stack of all three qualifiers read as owning 100% of an offence with more mouths.
        """
        loose = nfl_stacking.stack_combinations(BOARD, "PHI", min_projection=0.0)
        tight = nfl_stacking.stack_combinations(BOARD, "PHI", min_projection=8.0)
        biggest_loose = loose.nlargest(1, "Captured targets").iloc[0]
        same = tight[tight["Partners"] == biggest_loose["Partners"]]
        if not same.empty:
            assert same.iloc[0]["Captured share"] == pytest.approx(
                biggest_loose["Captured share"])
        # and it is never the whole pass game unless every catcher is in the stack
        two_man = loose[loose["Size"] == 2]
        assert (two_man["Captured share"] < 1.0).all()

    def test_salary_and_projection_are_the_sum_of_the_men_in_it(self):
        combos = nfl_stacking.stack_combinations(BOARD, "PHI", min_projection=0.0)
        row = combos[combos["Partners"] == "WR One"].iloc[0]
        assert row["Salary"] == 6600 + 6600
        assert row["Proj"] == pytest.approx(17.0 + 12.0)

    def test_a_club_with_no_quarterback_yields_nothing(self):
        assert nfl_stacking.stack_combinations(BOARD, "WAS", min_projection=0.0).empty

    def test_bring_backs_come_from_the_other_side_of_the_game(self):
        candidates = nfl_stacking.bring_back_candidates(BOARD, "PHI", "WAS")
        assert list(candidates["Team"].unique()) == ["WAS"]
        assert "RB One" not in set(candidates["Name"])


class TestDefenceReliability:
    def test_projected_is_the_allowed_figure_regressed_by_its_own_correlation(self):
        """A defence five points above average against receivers projects about half a
        point above average — not five. Reading `Allowed/G` as a forecast is the common way
        to lose money on a matchup table."""
        for position, r in nfl_league.RELIABILITY.items():
            assert 0.0 <= r < 0.5, position

    def test_every_scored_position_has_a_reliability(self):
        assert set(nfl_league.RELIABILITY) == set(nfl_league.POSITIONS)
        assert set(nfl_league.IN_SEASON_RELIABILITY) == set(nfl_league.POSITIONS)


class TestEnteredLineups:
    def _entries_file(self, tmp_path):
        from nfl import upload
        header = (["Entry ID", "Contest Name", "Contest ID", "Entry Fee"]
                  + list(upload.SLOT_ORDER) + ["", "Instructions"])
        filled = ["9001", "NFL $10K", "77", "$5"] + [
            f"P{i} ({1000 + i})" for i in range(len(upload.SLOT_ORDER))] + ["", ""]
        blank = ["9002", "NFL $10K", "77", "$5"] + [""] * len(upload.SLOT_ORDER) + ["", ""]
        path = tmp_path / "DKEntries.csv"
        pd.DataFrame([filled, blank], columns=header).to_csv(path, index=False)
        return str(path)

    def test_it_reads_one_row_per_player_per_entry(self, tmp_path):
        from nfl import upload
        entries = nfl_slates.entered_lineups(self._entries_file(tmp_path))
        assert entries["Entry"].nunique() == 1
        assert len(entries) == len(upload.SLOT_ORDER)

    def test_a_contest_row_with_no_lineup_is_not_an_entry(self, tmp_path):
        """An empty contest row is a slot you own, not a lineup you entered."""
        entries = nfl_slates.entered_lineups(self._entries_file(tmp_path))
        assert "9002" not in set(entries["Entry"])

    def test_exposure_is_out_of_the_entry_count(self, tmp_path):
        entries = nfl_slates.entered_lineups(self._entries_file(tmp_path))
        exposure = nfl_slates.entry_exposure(entries)
        assert (exposure["Exposure%"] == 100.0).all()


class TestRealFilesIfPresent:
    def test_the_slates_on_disk_label_and_price(self):
        slates = nfl_slates.list_slates()
        if slates.empty:
            pytest.skip("no DK salary exports on disk")
        row = slates.iloc[0]
        players, report = nfl_slates.slate_players(row["Path"])
        assert not players.empty
        assert players["Team"].nunique() == row["Games"] * 2
        assert set(players["Band"]) <= {"Punt", "Value", "Mid", "Stud", ""}

    def test_projections_join_by_name_and_stay_per_game(self):
        slates = nfl_slates.list_slates()
        if slates.empty:
            pytest.skip("no DK salary exports on disk")
        players, _ = nfl_slates.slate_players(slates.iloc[0]["Path"])
        board, source = nfl_slates.attach_projection(players)
        if source is None:
            pytest.skip("no projection export on disk")
        projected = board["Proj"].dropna()
        assert not projected.empty
        # Per game, not per season: a season total would be an order of magnitude larger.
        assert projected.max() < 60


class TestPagesRender:
    @pytest.mark.parametrize("page", PAGES)
    def test_it_renders_without_raising(self, page):
        pytest.importorskip("streamlit.testing.v1")
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(_page(page), default_timeout=300).run()
        assert not app.exception, [str(e.value)[:200] for e in app.exception]
        assert app.header


class TestBandDisplayOrder:
    def test_studs_lead_and_punts_trail(self):
        """A board is read from the money down: what am I paying up for, then what am I
        paying for it with. The definition order is low-to-high; the display order is not."""
        assert nfl_slates.BAND_DISPLAY_ORDER[0] == "Stud"
        assert nfl_slates.BAND_DISPLAY_ORDER[-1] == "Punt"
        assert set(nfl_slates.BAND_DISPLAY_ORDER) == {b for _, _, b in nfl_slates.PRICE_BANDS}


class TestAllowedAgainstStacks:
    def _summary(self):
        return pd.DataFrame([
            {"Team": "PHI", "Opp": "WAS", "QB": "QB One", "Best 3-man": 37.1,
             "3-man cost": 18300.0},
            {"Team": "WAS", "Opp": "PHI", "QB": "QB Two", "Best 3-man": 28.0,
             "3-man cost": 15000.0},
        ])

    def _allowed(self):
        return pd.DataFrame([
            {"Team": "WAS", "Pos": "WR", "Allowed/G": 35.9, "Rank": 8,
             "vs League": 4.4, "Projected": 32.0},
            {"Team": "PHI", "Pos": "WR", "Allowed/G": 28.0, "Rank": 25,
             "vs League": -3.5, "Projected": 31.1},
        ])

    def test_it_pairs_each_offence_with_the_defence_it_draws(self):
        """`itertuples` renames any column that is not a valid identifier to a positional
        `_5` — and every column this needs ("Best 3-man", "3-man cost") is that shape. Every
        lookup came back None and the frame came back **empty with no error**.
        """
        out = nfl_league.allowed_against_stacks(self._allowed(), self._summary(), "WR")
        assert len(out) == 2
        philadelphia = out[out["Team"] == "PHI"].iloc[0]
        assert philadelphia["Opp"] == "WAS"
        assert philadelphia["Opp allows/G"] == 35.9
        assert philadelphia["Stack proj"] == 37.1
        assert philadelphia["Stack cost"] == 18300.0

    def test_stack_value_is_points_per_thousand(self):
        out = nfl_league.allowed_against_stacks(self._allowed(), self._summary(), "WR")
        row = out[out["Team"] == "PHI"].iloc[0]
        assert row["Stack value"] == pytest.approx(37.1 / 18.3, rel=1e-3)

    def test_a_size_the_summary_does_not_carry_is_empty_not_a_crash(self):
        out = nfl_league.allowed_against_stacks(self._allowed(), self._summary(), "WR",
                                                size="Best 9-man")
        assert out.empty
