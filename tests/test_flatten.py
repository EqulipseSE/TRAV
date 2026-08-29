import json
from pathlib import Path

import pytest

from atg_favorites.flatten import (
    _shoe,
    _streck_pct,
    build_dataframe,
    flatten_game,
    is_scratched,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_game.json"


@pytest.fixture()
def sample_game() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_flatten_game_returns_one_row_per_finished_race(sample_game):
    rows = flatten_game(sample_game)
    assert len(rows) == 2


def test_normal_race_favorite_wins(sample_game):
    rows = flatten_game(sample_game)
    row = rows[0]

    assert row["game_type"] == "V75"
    assert row["favorite_horse_name"] == "Favorithästen"
    assert row["favorite_streck_pct"] == pytest.approx(45.0)
    assert row["favorite_won"] is True
    assert row["favorite_top3"] is True
    assert row["favorite_added_distance_m"] == 0
    assert row["num_starters"] == 3
    assert row["num_scratched"] == 0
    assert row["second_favorite_horse_name"] == "Utmanaren"
    assert row["favorite_streck_margin"] == pytest.approx(23.0)
    assert row["winner_horse_name"] == "Favorithästen"
    assert row["winner_is_favorite"] is True
    assert row["favorite_shoes_front"] is False
    assert row["favorite_shoes_back"] is True


def test_favoritfall_race_with_scratching_and_tillagg(sample_game):
    rows = flatten_game(sample_game)
    row = rows[1]

    assert row["favorite_horse_name"] == "StorFavoriten"
    assert row["favorite_streck_pct"] == pytest.approx(52.0)
    assert row["favorite_won"] is False
    assert row["num_starters"] == 4
    assert row["num_scratched"] == 1
    assert row["winner_horse_name"] == "Skräll Sam"
    assert row["winner_is_favorite"] is False
    # back shoes removed ("changed": true, "hasShoe": false) -> barfota + skoombyte
    assert row["favorite_shoes_back"] is False
    assert row["favorite_shoes_changed"] is True


def test_added_distance_reflects_tillagg_for_autostart_group(sample_game):
    rows = flatten_game(sample_game)
    # The third horse in race 1 starts 20m further back (tillägg).
    race = sample_game["races"][0]
    third_start = race["starts"][2]
    assert third_start["distance"] - race["distance"] == 20
    # It isn't the favorite in this race, so this checks the raw fixture
    # invariant that the flatten logic for the favorite relies on.
    assert rows[0]["favorite_added_distance_m"] == 0


def test_is_scratched():
    race = {"result": {"scratchings": [4]}}
    assert is_scratched({"number": 4}, race) is True
    assert is_scratched({"number": 1}, race) is False


def test_streck_pct_none_when_missing():
    assert _streck_pct({"pools": {}}, "V75") is None
    assert _streck_pct({"pools": {"V75": {"betDistribution": 1234}}}, "V75") == pytest.approx(12.34)


def test_shoe_none_when_not_reported():
    horse_not_reported = {"shoes": {"reported": False}}
    horse_reported = {"shoes": {"reported": True, "back": {"hasShoe": True, "changed": False}}}
    assert _shoe(horse_not_reported, "back", "hasShoe") is None
    assert _shoe(horse_reported, "back", "hasShoe") is True


def test_build_dataframe_from_raw_dir(tmp_path, sample_game):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "V75_2024-01-06_1_1.json").write_text(json.dumps(sample_game), encoding="utf-8")

    df = build_dataframe(raw_dir)

    assert len(df) == 2
    assert set(df["game_id"]) == {"V75_2024-01-06_1_1"}
    assert "favorite_streck_pct" in df.columns
    assert df.iloc[0]["source_file"] == "V75_2024-01-06_1_1.json"


def test_build_dataframe_empty_dir(tmp_path):
    df = build_dataframe(tmp_path)
    assert df.empty
