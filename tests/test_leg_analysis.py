import json
from pathlib import Path

import pandas as pd
import pytest

from atg_favorites.leg_analysis import (
    LegNotFoundError,
    build_leg_report,
    current_favorite,
    find_similar_races,
    leg_starters,
    resolve_leg,
    similarity_stats,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_game.json"


class StubClient:
    """A fake AtgClient serving the sample fixture without any network access."""

    def __init__(self, game: dict):
        self.game = game

    def get_calendar_day(self, date_str: str) -> dict:
        return {
            "date": date_str,
            "games": {
                "V75": [{"id": self.game["id"], "status": self.game["status"]}],
            },
        }

    def get_game(self, game_id: str) -> dict:
        assert game_id == self.game["id"]
        return self.game


@pytest.fixture()
def sample_game() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture()
def stub_client(sample_game) -> StubClient:
    return StubClient(sample_game)


def test_resolve_leg_finds_correct_race(stub_client):
    target = resolve_leg(stub_client, "2024-01-06", avd=2, game_type="V75")
    assert target.game_id == "V75_2024-01-06_1_1"
    assert target.race["name"] == "Testlopp 2 (favoritfall)"
    assert target.pool_key == "V75"


def test_resolve_leg_out_of_range_raises(stub_client):
    with pytest.raises(LegNotFoundError):
        resolve_leg(stub_client, "2024-01-06", avd=99, game_type="V75")


def test_resolve_leg_filters_by_track(stub_client):
    target = resolve_leg(stub_client, "2024-01-06", avd=1, game_type="V75", track_name="Testbanan")
    assert target.avd == 1

    with pytest.raises(LegNotFoundError):
        resolve_leg(stub_client, "2024-01-06", avd=1, game_type="V75", track_name="Nowhereville")


def test_leg_starters_and_current_favorite(stub_client):
    target = resolve_leg(stub_client, "2024-01-06", avd=2, game_type="V75")
    starters = leg_starters(target)

    assert len(starters) == 4
    # Sorted by streck_pct descending.
    assert starters.iloc[0]["horse_name"] == "StorFavoriten"

    favorite = current_favorite(starters)
    assert favorite["horse_name"] == "StorFavoriten"
    assert favorite["streck_pct"] == pytest.approx(52.0)
    # The scratched horse must never be picked as favorite even if it had streck data.
    assert favorite["scratched"] is False


@pytest.fixture()
def historical_races() -> pd.DataFrame:
    rows = []
    # 3 races matching everything (same track, distance, start method, field size).
    for i in range(3):
        rows.append(
            {
                "game_id": f"tier1_{i}",
                "game_date": "2024-01-01",
                "track_name": "Bana A",
                "start_method": "volte",
                "race_distance_m": 2140,
                "num_starters": 4,
                "favorite_horse_name": f"Favorit{i}",
                "favorite_streck_pct": 50.0,
                "favorite_won": i == 0,
                "favorite_top3": True,
                "winner_horse_name": f"Vinnare{i}",
            }
        )
    # 6 more races matching distance/start-method/field-size but a different track.
    for i in range(6):
        rows.append(
            {
                "game_id": f"tier2_{i}",
                "game_date": "2024-01-02",
                "track_name": "Bana B",
                "start_method": "volte",
                "race_distance_m": 2100,
                "num_starters": 5,
                "favorite_horse_name": f"Favorit2_{i}",
                "favorite_streck_pct": 48.0,
                "favorite_won": i % 2 == 0,
                "favorite_top3": True,
                "winner_horse_name": f"Vinnare2_{i}",
            }
        )
    # 2 unrelated races (different start method) that should never match.
    for i in range(2):
        rows.append(
            {
                "game_id": f"unrelated_{i}",
                "game_date": "2024-01-03",
                "track_name": "Bana C",
                "start_method": "auto",
                "race_distance_m": 1640,
                "num_starters": 12,
                "favorite_horse_name": f"Unrelated{i}",
                "favorite_streck_pct": 30.0,
                "favorite_won": False,
                "favorite_top3": False,
                "winner_horse_name": f"UnrelatedWinner{i}",
            }
        )
    return pd.DataFrame(rows)


def test_find_similar_races_relaxes_to_tier2_when_tier1_too_small(historical_races):
    target_row = {
        "track_name": "Bana A",
        "start_method": "volte",
        "race_distance_m": 2140,
        "num_starters": 4,
    }
    similar, description = find_similar_races(historical_races, target_row, min_sample_size=8)

    assert len(similar) == 9  # tier1 (3) + tier2 (6), track requirement dropped
    assert "oavsett bana" in description
    assert set(similar["track_name"]) == {"Bana A", "Bana B"}


def test_find_similar_races_keeps_tier1_when_big_enough(historical_races):
    target_row = {
        "track_name": "Bana A",
        "start_method": "volte",
        "race_distance_m": 2140,
        "num_starters": 4,
    }
    similar, description = find_similar_races(historical_races, target_row, min_sample_size=2)

    assert len(similar) == 3
    assert description == "samma bana, distans, startsätt och ungefär samma fältstorlek"


def test_find_similar_races_empty_archive_returns_message():
    similar, description = find_similar_races(pd.DataFrame(), {"track_name": "X"})
    assert similar.empty
    assert "inget historiskt underlag" in description


def test_similarity_stats_basic(historical_races):
    stats = similarity_stats(historical_races, favorite_streck_pct=50.0)
    assert stats["races"] == 11
    assert 0.0 <= stats["favorite_win_rate"] <= 1.0
    assert stats["favoritfall_rate"] == pytest.approx(1 - stats["favorite_win_rate"])


def test_similarity_stats_empty():
    assert similarity_stats(pd.DataFrame(), 40.0) == {"races": 0}


def test_build_leg_report_end_to_end(tmp_path, stub_client, historical_races):
    races_csv = tmp_path / "races.csv"
    historical_races.to_csv(races_csv, index=False)

    report = build_leg_report(
        races_csv,
        date_str="2024-01-06",
        avd=2,
        game_type="V75",
        client=stub_client,
        min_sample_size=2,
    )

    assert report.target.game_id == "V75_2024-01-06_1_1"
    assert report.favorite["horse_name"] == "StorFavoriten"
    assert not report.similar_races.empty
    assert report.similarity_stats["races"] > 0
