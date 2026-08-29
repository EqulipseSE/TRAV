import json
from pathlib import Path

import pandas as pd
import pytest

from atg_favorites.analysis import (
    add_streck_bucket,
    favorite_bucket_analysis,
    favorite_surprises,
    summarize,
)
from atg_favorites.flatten import flatten_game

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_game.json"


@pytest.fixture()
def races_df() -> pd.DataFrame:
    game = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    rows = flatten_game(game)
    return pd.DataFrame(rows)


def test_summarize(races_df):
    summary = summarize(races_df)
    assert summary["races"] == 2
    assert summary["favorite_win_rate"] == pytest.approx(0.5)
    assert summary["favoritfall_rate"] == pytest.approx(0.5)


def test_add_streck_bucket(races_df):
    bucketed = add_streck_bucket(races_df, bucket_edges=(0, 50, 100))
    assert list(bucketed["streck_bucket"]) == ["0-50%", "50-100%"]


def test_favorite_bucket_analysis_columns_and_values(races_df):
    result = favorite_bucket_analysis(races_df, bucket_edges=(0, 50, 100))

    assert list(result.columns) == [
        "streck_bucket",
        "races",
        "avg_streck_pct",
        "implied_win_prob",
        "win_rate",
        "top3_rate",
        "favoritfall_rate",
        "edge_vs_streck",
    ]

    low_bucket = result[result["streck_bucket"] == "0-50%"].iloc[0]
    assert low_bucket["races"] == 1
    assert low_bucket["win_rate"] == pytest.approx(1.0)
    assert low_bucket["favoritfall_rate"] == pytest.approx(0.0)

    high_bucket = result[result["streck_bucket"] == "50-100%"].iloc[0]
    assert high_bucket["races"] == 1
    assert high_bucket["win_rate"] == pytest.approx(0.0)
    assert high_bucket["favoritfall_rate"] == pytest.approx(1.0)
    # The strong (52%) favorite lost, so its edge vs. the streck-implied
    # probability should be strongly negative.
    assert high_bucket["edge_vs_streck"] < -0.4


def test_favorite_bucket_analysis_grouped_by_game_type(races_df):
    races_df["game_type"] = ["V75", "V86"]
    result = favorite_bucket_analysis(races_df, bucket_edges=(0, 100), group_by=["game_type"])
    assert set(result["game_type"]) == {"V75", "V86"}


def test_favorite_surprises_finds_strong_favorite_loss(races_df):
    surprises = favorite_surprises(races_df, min_streck_pct=50.0)
    assert len(surprises) == 1
    assert surprises.iloc[0]["favorite_horse_name"] == "StorFavoriten"


def test_favorite_surprises_empty_when_threshold_too_high(races_df):
    surprises = favorite_surprises(races_df, min_streck_pct=90.0)
    assert surprises.empty
