import json
import pickle

import pandas as pd
import pytest

from artifacts import atomic_write_json, atomic_write_parquet, atomic_write_pickle, staged_path


def test_atomic_writers_round_trip(tmp_path):
    json_path = tmp_path / "state.json"
    pickle_path = tmp_path / "state.pkl"
    parquet_path = tmp_path / "state.parquet"

    atomic_write_json(json_path, {"version": 2}, sort_keys=True)
    atomic_write_pickle(pickle_path, {"players": [1, 2]})
    atomic_write_parquet(parquet_path, pd.DataFrame({"Name": ["A"], "Proj": [8.5]}))

    assert json.loads(json_path.read_text(encoding="utf-8")) == {"version": 2}
    with pickle_path.open("rb") as stream:
        assert pickle.load(stream) == {"players": [1, 2]}
    assert pd.read_parquet(parquet_path).to_dict("records") == [{"Name": "A", "Proj": 8.5}]


def test_a_failed_staged_write_keeps_the_previous_file(tmp_path):
    target = tmp_path / "artifact.txt"
    target.write_text("previous", encoding="utf-8")

    with pytest.raises(RuntimeError):
        with staged_path(target) as temporary:
            temporary.write_text("partial replacement", encoding="utf-8")
            raise RuntimeError("writer failed")

    assert target.read_text(encoding="utf-8") == "previous"
    assert not list(tmp_path.glob(".artifact.txt.*"))
