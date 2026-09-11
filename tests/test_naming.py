"""Slate labelling: the turbo/late rule that needs a night's exports read together.

`slug_from_contents` reads one export and buckets it by first pitch. That cannot see a
turbo, because DK runs its short 3-4 game block in the same evening window as the main
slate -- on 2026-08-03, -04, -07 and -08 both came back "main", and only the hand-typed
filenames kept a night's outputs from overwriting each other.
"""

import pytest

from dfs.naming import TURBO_SLATE, label_slates, slug_from_contents


def export(name, games, first_pm, last_pm, players=100):
    """A `describe_slate`-shaped export. Times are PM hours: 19.08 is 7:05PM ET."""
    return {"path": name, "name": name,
            "games": [f"A{i}@B{i}" for i in range(games)],
            "first": int(first_pm * 60), "last": int(last_pm * 60), "players": players}


# A raw DK download carries no slate label in its name; these are what the rule is for.
RAW = "DKSalaries.csv"


def raw(n):
    return f"DKSalaries ({n}).csv"


class TestTurboIsFoundByComparison:
    def test_a_lone_evening_export_is_still_main(self):
        one = export(RAW, 12, 19.08, 22.25)
        assert label_slates([one], "2026-08-07")[RAW] == "main"

    def test_first_pitch_alone_cannot_tell_them_apart(self):
        """The premise: this is why the labels have to be decided as a set."""
        main = export(RAW, 12, 19.08, 22.25)
        turbo = export(raw(1), 3, 18.67, 18.75)
        assert slug_from_contents(main) == slug_from_contents(turbo) == "main"

    def test_the_smaller_block_becomes_the_turbo(self):
        main = export(RAW, 12, 19.08, 22.25)
        turbo = export(raw(1), 3, 18.67, 18.75)
        labels = label_slates([main, turbo], "2026-08-07")
        assert labels[RAW] == "main"
        assert labels[raw(1)] == TURBO_SLATE

    def test_order_of_the_exports_does_not_matter(self):
        main = export(RAW, 12, 19.08, 22.25)
        turbo = export(raw(1), 3, 18.67, 18.75)
        assert (label_slates([main, turbo], "2026-08-07")
                == label_slates([turbo, main], "2026-08-07"))

    def test_two_turbos_get_distinct_labels(self):
        exports = [export(RAW, 11, 19.0, 21.0), export(raw(1), 4, 18.58, 18.67),
                   export(raw(2), 3, 18.08, 18.17)]
        labels = label_slates(exports, "2026-08-07")
        assert labels[RAW] == "main"
        assert {labels[raw(1)], labels[raw(2)]} == {TURBO_SLATE, f"{TURBO_SLATE}-2"}

    def test_equal_game_counts_fall_back_to_player_count(self):
        exports = [export(RAW, 4, 19.0, 21.0, players=400),
                   export(raw(1), 4, 19.0, 21.0, players=380)]
        labels = label_slates(exports, "2026-08-07")
        assert labels[RAW] == "main"
        assert labels[raw(1)] == TURBO_SLATE

    def test_early_main_and_turbo_all_resolve(self):
        exports = [export(RAW, 5, 13.0, 14.5), export(raw(1), 12, 19.08, 22.25),
                   export(raw(2), 3, 18.67, 18.75)]
        labels = label_slates(exports, "2026-08-07")
        assert labels[RAW] == "early"
        assert labels[raw(1)] == "main"
        assert labels[raw(2)] == TURBO_SLATE


class TestLateBeatsTurbo:
    """Being small is not what makes a slate late -- starting after the rest of them is."""

    def test_a_block_after_the_main_slate_is_late_not_turbo(self):
        main = export(RAW, 12, 19.08, 21.67)
        after = export(raw(1), 3, 21.75, 22.0)
        assert label_slates([main, after], "2026-08-07")[raw(1)] == "late"

    def test_a_block_inside_the_main_window_is_a_turbo(self):
        """2026-08-03's turbo started at 8:05PM, inside a 7:05-9:40PM main slate."""
        main = export(RAW, 7, 19.08, 21.67)
        inside = export(raw(1), 3, 20.08, 20.17)
        assert label_slates([main, inside], "2026-08-07")[raw(1)] == TURBO_SLATE


class TestExplicitNamesWin:
    def test_a_filename_label_is_never_overridden(self):
        named = export("DKSalaries_2026-08-07_turbo.csv", 3, 18.0, 18.5)
        main = export(RAW, 10, 19.0, 21.0)
        labels = label_slates([named, main], "2026-08-07")
        assert labels[named["path"]] == "turbo"
        assert labels[RAW] == "main"

    def test_a_taken_label_is_not_handed_out_twice(self):
        """Resolving one collision by creating another would defeat the point."""
        named = export("DKSalaries_2026-08-07_turbo.csv", 3, 18.0, 18.5)
        main = export(RAW, 10, 19.0, 21.0)
        another = export(raw(3), 4, 18.75, 18.83)
        labels = label_slates([named, main, another], "2026-08-07")
        assert labels[another["path"]] == f"{TURBO_SLATE}-2"
        assert len(set(labels.values())) == 3


@pytest.mark.parametrize("exports", [
    [export(RAW, 12, 19.08, 22.25), export(raw(1), 3, 18.67, 18.75)],
    [export(RAW, 12, 19.08, 21.67), export("DKSalaries_2026-08-07_late.csv", 4, 21.67, 22.25),
     export(raw(2), 3, 22.33, 22.5)],
    [export(RAW, 11, 19.0, 21.0), export(raw(1), 4, 18.58, 18.67),
     export(raw(2), 3, 18.08, 18.17)],
    [export(RAW, 5, 13.0, 14.5), export(raw(1), 12, 19.08, 22.25),
     export(raw(2), 3, 18.67, 18.75)],
])
def test_a_night_never_files_two_exports_under_one_label(exports):
    """The property the whole function exists for: date-only naming lost a slate."""
    labels = label_slates(exports, "2026-08-07")
    assert len(set(labels.values())) == len(exports)


class TestTwoSlatesOfTheSameKind:
    """2026-08-18 ran two turbos. Every way of naming one has to reach exactly one file.

    The night's exports were renamed `..._turbo.csv` and `..._turbo-2.csv`, which reads
    unambiguously to a human, and `--slate turbo` still failed: the selector matched by
    substring, "turbo" is inside both names, and the run was rejected as ambiguous. The
    workaround -- passing the whole filename -- selected correctly but then became the
    output label, so the night filed itself under a slug that resolved back to nothing.
    """

    DIR = "dfs_daily_files"
    FILES = ["DKSalaries_2026-08-18_main.csv",
             "DKSalaries_2026-08-18_turbo.csv",
             "DKSalaries_2026-08-18_turbo-2.csv"]

    @pytest.fixture
    def exports(self, tmp_path):
        import os
        folder = tmp_path / self.DIR
        folder.mkdir()
        made = []
        for index, name in enumerate(self.FILES):
            path = str(folder / name)
            open(path, "w").close()
            made.append({"path": path, "name": name,
                         "games": [f"A{i}@B{i}" for i in range(9 - index * 3)],
                         "first": 19 * 60, "last": 20 * 60 + 40, "players": 100})
        return made

    def _pick(self, exports, requested, monkeypatch):
        from dfs import salaries
        monkeypatch.setattr(salaries, "list_salary_files",
                            lambda date, dirs=None: exports)
        return salaries.find_salary_file("2026-08-18", slate=requested)

    @pytest.mark.parametrize("requested,expected", [
        ("turbo", "DKSalaries_2026-08-18_turbo.csv"),
        ("turbo-2", "DKSalaries_2026-08-18_turbo-2.csv"),
        ("main", "DKSalaries_2026-08-18_main.csv"),
        ("DKSalaries_2026-08-18_turbo.csv", "DKSalaries_2026-08-18_turbo.csv"),
        (r"dfs_daily_files\DKSalaries_2026-08-18_turbo-2.csv",
         "DKSalaries_2026-08-18_turbo-2.csv"),
        ("dfs_daily_files/DKSalaries_2026-08-18_main.csv",
         "DKSalaries_2026-08-18_main.csv"),
    ])
    def test_every_way_of_naming_a_slate_selects_one_file(self, exports, monkeypatch,
                                                          requested, expected):
        import os
        assert os.path.basename(self._pick(exports, requested, monkeypatch)) == expected

    def test_an_exact_label_is_not_lost_to_a_longer_name(self, exports, monkeypatch):
        """The failure itself: "turbo" is a substring of "turbo-2"."""
        from dfs.salaries import AmbiguousSlate
        try:
            self._pick(exports, "turbo", monkeypatch)
        except AmbiguousSlate:                       # pragma: no cover - the bug
            pytest.fail("'turbo' was rejected as ambiguous against 'turbo-2'")

    def test_a_partial_name_still_matches_loosely(self, exports, monkeypatch):
        """The substring path is a fallback now, not gone."""
        import os
        picked = self._pick(exports, "_ma", monkeypatch)
        assert os.path.basename(picked) == "DKSalaries_2026-08-18_main.csv"

    @pytest.mark.parametrize("requested", [
        "turbo",
        "DKSalaries_2026-08-18_turbo.csv",
        "DKSalaries_2026-08-18_turbo",
        r"dfs_daily_files\DKSalaries_2026-08-18_turbo.csv",
    ])
    def test_naming_a_file_selects_it_rather_than_relabelling_the_night(self, exports,
                                                                        requested):
        """Whatever picked the export, the night files itself under the export's label."""
        from dfs.naming import slate_for
        chosen = next(e for e in exports if e["name"].endswith("_turbo.csv"))
        assert slate_for(chosen, requested=requested, date="2026-08-18",
                         peers=exports) == "turbo"

    def test_an_explicit_rename_still_wins(self, exports):
        """--slate is still a naming override when it is not just the file's own name."""
        from dfs.naming import slate_for
        chosen = next(e for e in exports if e["name"].endswith("_turbo.csv"))
        assert slate_for(chosen, requested="Turbo Night", date="2026-08-18",
                         peers=exports) == "turbo-night"


class TestATemplateKeepsTheNameItWasGiven:
    """2026-08-25 priced one six-game block and it got downloaded twice.

    The night ended up with `DKSalaries_2026-08-25_late.csv` and `..._turbo.csv` holding
    byte-identical exports. `template_slate` read only the clubs a template priced, so both
    exports matched every template exactly and the first one sorted -- "late" -- claimed
    them all. A template the user had already named `DKTemplate_2026-08-25_turbo.csv` was
    then filed as the late slate's, and collided with the late template already there.
    """

    DATE = "2026-08-25"
    GAMES = ["CHC@ARI", "CIN@SF", "CLE@LAA", "MIN@ATH", "PHI@SEA", "PIT@SD"]

    @pytest.fixture
    def night(self, monkeypatch):
        """The duplicated night: one main export, and one block filed under two labels."""
        from dfs import adopt as adopt_module
        exports = [
            {"path": f"d/DKSalaries_{self.DATE}_main.csv",
             "name": f"DKSalaries_{self.DATE}_main.csv",
             "games": self.GAMES + [f"A{i}@B{i}" for i in range(6)],
             "first": 19 * 60 + 5, "last": 21 * 60 + 45, "players": 1106},
        ] + [
            {"path": f"d/DKSalaries_{self.DATE}_{label}.csv",
             "name": f"DKSalaries_{self.DATE}_{label}.csv",
             "games": list(self.GAMES),
             "first": 21 * 60 + 38, "last": 21 * 60 + 45, "players": 555}
            for label in ("late", "turbo")
        ]
        monkeypatch.setattr(adopt_module, "list_salary_files", lambda date: exports)
        monkeypatch.setattr("dfs.upload.template_teams_for",
                            lambda path: {"ARI", "ATH", "CHC", "CIN", "CLE", "LAA",
                                          "MIN", "PHI", "PIT", "SD", "SEA", "SF"})
        return exports

    @pytest.mark.parametrize("name", ["DKTemplate_2026-08-25_turbo.csv",
                                      "DKTemplate_2026-08-25_turbo_entries.csv"])
    def test_the_filename_decides_when_two_exports_price_the_same_games(self, night, name):
        from dfs.adopt import template_slate
        assert template_slate(f"dk_lineups/{name}", self.DATE) == "turbo"

    def test_a_template_named_for_the_other_block_still_gets_it(self, night):
        from dfs.adopt import template_slate
        assert template_slate(f"dk_lineups/DKTemplate_{self.DATE}_late_entries.csv",
                              self.DATE) == "late"

    def test_an_unnamed_template_still_falls_back_to_what_it_prices(self, night):
        """The name is honoured, not required -- a raw download is matched as before."""
        from dfs.adopt import template_slate
        assert template_slate("dk_lineups/DKEntries (2).csv", self.DATE) in {"late", "turbo"}

    def test_a_label_no_export_carries_is_not_invented(self, night):
        """Only a label the night actually has is trusted; anything else re-derives."""
        from dfs.adopt import template_slate
        assert template_slate(f"dk_lineups/DKTemplate_{self.DATE}_early_entries.csv",
                              self.DATE) in {"late", "turbo"}


class TestTemplateFilenameSlate:
    @pytest.mark.parametrize("name,expected", [
        ("DKTemplate_2026-08-25_turbo.csv", "turbo"),
        ("DKTemplate_2026-08-25_turbo_entries.csv", "turbo"),
        ("DKTemplate_2026-07-30_main_bulk.csv", "main"),
        ("DKTemplate_2026-08-25_showdown-chcari_entries.csv", "showdown-chcari"),
        # 'unknown' is the placeholder for "could not tell", so it is not a name to keep.
        ("DKTemplate_2026-08-25_unknown_entries.csv", ""),
        ("DKTemplate_2026-08-25_entries.csv", ""),
        ("DKTemplate_2026-08-25.csv", ""),
        ("DKTemplate (2).csv", ""),
        # Not our convention: whatever DK called it says nothing about the slate.
        ("DKEntries.csv", ""),
        ("DKEntries (13).csv", ""),
        ("", ""),
    ])
    def test_only_a_slate_somebody_typed_comes_back(self, name, expected):
        from dfs.naming import slug_from_template_name
        assert slug_from_template_name(name, "2026-08-25") == expected
