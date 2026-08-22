"""Ownership estimation and normalisation.

The normalisation invariant is the one worth pinning: DK Classic gives the field eight
hitter slots and two pitcher slots, so projected ownership must sum to 800 and 200. A model
that sums elsewhere is not an ownership distribution, and every duplication and leverage
number built on it is on a different scale from the one it claims.
"""

import numpy as np
import pandas as pd
import pytest

from dfs.ownership import (GROUP_SLOTS, LEVERAGE_OWN_SD, LEVERAGE_OWN_SPAN,
                           LEVERAGE_OWN_WEIGHT, MAX_OWNERSHIP, MIN_MEANINGFUL_OWN,
                           _own_weight, attach_actual_ownership, estimate_ownership,
                           field_appeal)


class TestNormalisation:
    def test_hitters_sum_to_eight_hundred(self, slate):
        frame = estimate_ownership(slate)
        total = frame.loc[frame["Type"] == "H", "Own%"].sum()
        assert total == pytest.approx(GROUP_SLOTS["H"] * 100, abs=1.0)

    def test_pitchers_sum_to_two_hundred(self, slate):
        frame = estimate_ownership(slate)
        total = frame.loc[frame["Type"] == "P", "Own%"].sum()
        assert total == pytest.approx(GROUP_SLOTS["P"] * 100, abs=1.0)

    def test_normalisation_holds_on_a_small_slate(self, small_slate):
        frame = estimate_ownership(small_slate)
        for kind, slots in GROUP_SLOTS.items():
            total = frame.loc[frame["Type"] == kind, "Own%"].sum()
            assert total == pytest.approx(slots * 100, abs=1.0)

    def test_no_player_exceeds_the_cap(self, slate):
        frame = estimate_ownership(slate)
        assert frame["Own%"].max() <= MAX_OWNERSHIP + 1e-6

    def test_capping_redistributes_rather_than_truncating(self, slate):
        """Mass a capped player cannot hold has to land somewhere, or the sum breaks."""
        frame = slate.copy()
        # Make one hitter overwhelmingly attractive so the cap must bite.
        target = frame.index[frame["Type"] == "H"][0]
        frame.loc[target, "Salary"] = 6500
        frame.loc[target, "DK Avg"] = 400.0
        out = estimate_ownership(frame)
        assert out.loc[target, "Own%"] <= MAX_OWNERSHIP + 1e-6
        assert out.loc[out["Type"] == "H", "Own%"].sum() == pytest.approx(800, abs=1.0)

    def test_ownership_is_never_negative(self, slate):
        frame = estimate_ownership(slate)
        assert (frame["Own%"].dropna() >= 0).all()


class TestNonCircularity:
    def test_appeal_ignores_our_own_projection(self, slate):
        """If ownership came from our projection, leverage would be structurally zero."""
        frame = slate.copy()
        baseline = field_appeal(frame[frame["Type"] == "H"])
        frame.loc[frame["Type"] == "H", "Proj"] *= 3.0
        frame.loc[frame["Type"] == "H", "Ceiling"] *= 3.0
        moved = field_appeal(frame[frame["Type"] == "H"])
        pd.testing.assert_series_equal(baseline, moved)

    def test_salary_is_the_dominant_term(self, slate):
        """Fitted result, kept honest: price beats value roughly three to one."""
        hitters = slate[slate["Type"] == "H"].copy()
        appeal = field_appeal(hitters)
        salary_rank = hitters["Salary"].rank(pct=True)
        assert appeal.corr(salary_rank) > 0.7


class TestColumns:
    def test_all_derived_columns_are_added(self, slate):
        frame = estimate_ownership(slate)
        for column in ("Own%", "Own Pct", "Leverage", "Lev Score", "Own Src", "Unrostered"):
            assert column in frame.columns

    def test_source_is_marked_as_an_estimate(self, slate):
        frame = estimate_ownership(slate)
        assert set(frame.loc[frame["Own%"].notna(), "Own Src"]) == {"est"}

    def test_lev_score_discounts_ceiling_by_ownership(self, slate):
        """The charge is per group, against that group's own ceiling spread.

        This asserted a flat `LEVERAGE_OWN_WEIGHT` until the weight became per-group. A flat
        points constant means different things to pitchers and hitters and drifts whenever
        the projection scale changes -- which is the whole reason `_own_weight` exists -- so
        the test has to price each row on its own group, not on the fallback constant.
        """
        frame = estimate_ownership(slate)
        rostered = frame[frame["Own%"].notna() & (frame["Own%"] >= MIN_MEANINGFUL_OWN)]
        assert not rostered.empty

        for player_type, group in rostered.groupby("Type", observed=True):
            # Spread is measured over the whole board's group, not the rostered subset.
            weight = _own_weight(frame.loc[frame["Type"] == player_type, "Ceiling"], player_type)
            row = group.iloc[0]
            expected = max(0.0, row["Ceiling"] - weight * row["Own%"])
            assert row["Lev Score"] == pytest.approx(expected, abs=0.02), player_type

    def test_the_ownership_charge_is_set_per_group(self, slate):
        """Pitchers and hitters are traded on their own ceiling scale, not a shared one.

        The flat 0.12 cost a pitcher 0.93 ceiling sd and a hitter 2.54 -- a real asymmetry
        that was invisible. If these ever collapse to one number, that asymmetry has been
        lost again.
        """
        frame = estimate_ownership(slate)
        weights = {t: _own_weight(g["Ceiling"], t) for t, g in frame.groupby("Type", observed=True)}
        assert set(weights) == {"P", "H"}
        assert weights["P"] != weights["H"]
        for player_type, weight in weights.items():
            spread = frame.loc[frame["Type"] == player_type, "Ceiling"].std()
            assert weight == pytest.approx(
                LEVERAGE_OWN_SD[player_type] * spread / LEVERAGE_OWN_SPAN)

    def test_the_flat_weight_is_the_fallback_when_a_group_has_no_spread(self):
        """One player has no ceiling spread to denominate against."""
        assert _own_weight(pd.Series([30.0]), "H") == LEVERAGE_OWN_WEIGHT
        assert _own_weight(pd.Series([], dtype=float), "P") == LEVERAGE_OWN_WEIGHT
        assert _own_weight(pd.Series([30.0, 40.0]), "unknown-type") == LEVERAGE_OWN_WEIGHT

    def test_lev_score_is_never_negative(self, slate):
        """A negative objective value would make the solver prefer an empty slot."""
        frame = estimate_ownership(slate)
        assert (frame["Lev Score"].dropna() >= 0).all()

    def test_an_empty_frame_is_handled(self):
        out = estimate_ownership(pd.DataFrame())
        assert out.empty or out["Own%"].isna().all()


class TestActualOwnership:
    def test_real_percentages_replace_the_estimate(self, slate):
        frame = estimate_ownership(slate)
        name = frame["Name"].iloc[0]
        from dfs.salaries import normalize_name
        out = attach_actual_ownership(frame, {normalize_name(name): 37.5})
        row = out[out["Name"] == name].iloc[0]
        assert row["Own%"] == pytest.approx(37.5)
        assert row["Own Src"] == "DK"

    def test_players_without_real_data_keep_the_estimate(self, slate):
        frame = estimate_ownership(slate)
        from dfs.salaries import normalize_name
        name = frame["Name"].iloc[0]
        before = frame[frame["Name"] != name]["Own%"].tolist()
        out = attach_actual_ownership(frame, {normalize_name(name): 37.5})
        assert out[out["Name"] != name]["Own%"].tolist() == before
        assert set(out[out["Name"] != name]["Own Src"]) == {"est"}

    def test_dk_id_match_wins_over_name(self, slate):
        frame = estimate_ownership(slate)
        row = frame.iloc[0]
        out = attach_actual_ownership(frame, {str(int(row["DK ID"])): 12.0})
        assert out.iloc[0]["Own%"] == pytest.approx(12.0)

    def test_an_unrostered_player_gets_no_leverage(self, slate):
        """A 0.1%-owned player on a finished slate was rested, not missed by the field.

        Left alone he carries a full ceiling against zero ownership and scores as the most
        leveraged play on the board, so the leverage objective would chase scratches.
        """
        frame = estimate_ownership(slate)
        from dfs.salaries import normalize_name
        name = frame["Name"].iloc[0]
        out = attach_actual_ownership(frame, {normalize_name(name): 0.1})
        row = out[out["Name"] == name].iloc[0]
        assert row["Unrostered"]
        assert pd.isna(row["Leverage"])
        assert row["Lev Score"] == 0.0

    def test_the_threshold_is_where_it_says_it_is(self, slate):
        frame = estimate_ownership(slate)
        from dfs.salaries import normalize_name
        name = frame["Name"].iloc[0]
        just_above = attach_actual_ownership(
            frame, {normalize_name(name): MIN_MEANINGFUL_OWN + 0.01})
        assert not just_above[just_above["Name"] == name].iloc[0]["Unrostered"]

    def test_empty_input_is_a_no_op(self, slate):
        frame = estimate_ownership(slate)
        pd.testing.assert_frame_equal(attach_actual_ownership(frame, {}), frame)

    def test_renormalisation_after_real_data(self, slate):
        """Own Pct must be re-ranked, or percentiles describe the old estimate."""
        frame = estimate_ownership(slate)
        from dfs.salaries import normalize_name
        drafted = {normalize_name(n): float(v) for n, v in
                   zip(frame["Name"], np.linspace(1, 50, len(frame)))}
        out = attach_actual_ownership(frame, drafted)
        assert out["Own Pct"].notna().sum() == len(out)
        for kind in GROUP_SLOTS:
            group = out[out["Type"] == kind]
            assert group["Own Pct"].max() == pytest.approx(100.0, abs=1e-6)
