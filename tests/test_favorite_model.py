import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atg_favorites.favorite_model import (
    ALL_FEATURES,
    build_features_for_leg,
    build_start_dataset,
    calibration_table,
    compute_gap_to_rival,
    evaluate,
    parse_race_class,
    score_leg,
    time_split,
    top_value_non_favorites,
    train_model,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_game.json"


# --- parse_race_class -------------------------------------------------------


def test_parse_race_class_range_and_mare():
    terms = ["3-åriga och äldre ston 300.001 - 1.950.000 kr. Körsvenskrav kat. 1."]
    result = parse_race_class(terms)
    assert result["prize_low"] == pytest.approx(300001)
    assert result["prize_high"] == pytest.approx(1950000)
    assert result["is_mare_race"] is True
    assert result["is_coldblood"] is False


def test_parse_race_class_coldblood_and_lagst():
    terms = ["3-åriga och äldre svenska och norska kallblodiga lägst 150.001 kr."]
    result = parse_race_class(terms)
    assert result["is_coldblood"] is True
    assert result["is_mare_race"] is False
    assert result["prize_low"] == pytest.approx(150001)
    assert result["prize_high"] is None


def test_parse_race_class_hogst():
    terms = ["3-åriga och äldre svenska och norska kallblodiga högst 50.000 kr."]
    result = parse_race_class(terms)
    assert result["prize_high"] == pytest.approx(50000)
    assert result["prize_low"] is None


def test_parse_race_class_no_terms():
    result = parse_race_class(None)
    assert result == {"prize_low": None, "prize_high": None, "is_mare_race": False, "is_coldblood": False}


def test_parse_race_class_does_not_false_positive_on_substring():
    # "Boston" contains "ston" but not as a standalone word.
    terms = ["Lopp för hästar uppfödda i Boston. 100.000 - 200.000 kr."]
    result = parse_race_class(terms)
    assert result["is_mare_race"] is False


# --- compute_gap_to_rival ----------------------------------------------------


def test_compute_gap_to_rival_leader_and_others():
    streck = pd.Series({1: 45.0, 2: 22.0, 3: 9.0})
    gaps = compute_gap_to_rival(streck)
    assert gaps[1] == pytest.approx(23.0)  # leader's margin over runner-up
    assert gaps[2] == pytest.approx(22.0 - 45.0)
    assert gaps[3] == pytest.approx(9.0 - 45.0)


def test_compute_gap_to_rival_single_horse():
    streck = pd.Series({1: 100.0})
    gaps = compute_gap_to_rival(streck)
    assert gaps[1] == 0.0


# --- build_start_dataset ------------------------------------------------


@pytest.fixture()
def sample_game() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_build_start_dataset_from_fixture(tmp_path, sample_game):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "V75_2024-01-06_1_1.json").write_text(json.dumps(sample_game), encoding="utf-8")

    df = build_start_dataset(raw_dir)

    # Race 1 has 3 non-scratched runners, race 2 has 3 (one of four scratched).
    assert len(df) == 6
    assert df["won"].sum() == 2  # exactly one winner per race
    assert set(df["race_id"]) == {"2024-01-06_1_1", "2024-01-06_1_2"}

    favorite_row = df[(df["race_id"] == "2024-01-06_1_1") & (df["horse_name"] == "Favorithästen")].iloc[0]
    assert favorite_row["gap_to_rival"] == pytest.approx(23.0)
    assert bool(favorite_row["won"]) is True

    underdog_row = df[(df["race_id"] == "2024-01-06_1_2") & (df["horse_name"] == "Skräll Sam")].iloc[0]
    assert bool(underdog_row["won"]) is True
    assert underdog_row["gap_to_rival"] < 0


def test_build_start_dataset_empty_dir(tmp_path):
    df = build_start_dataset(tmp_path)
    assert df.empty


# --- time_split ---------------------------------------------------------


def test_time_split_older_vs_last_12_months():
    dates = pd.date_range("2023-01-01", periods=800, freq="D")
    df = pd.DataFrame({"game_date": dates.strftime("%Y-%m-%d"), "won": [False] * len(dates)})

    train, validation = time_split(df, validation_days=365, as_of=dates.max())

    assert len(train) + len(validation) == len(df)
    assert pd.to_datetime(train["game_date"]).max() < pd.to_datetime(validation["game_date"]).min()
    assert pd.to_datetime(validation["game_date"]).min() >= dates.max() - pd.Timedelta(days=365)


# --- synthetic training/evaluation/calibration --------------------------


def _make_synthetic_dataset(n_races: int = 150, seed: int = 0) -> pd.DataFrame:
    """A synthetic per-start dataset with a genuine streck -> win relationship."""
    rng = np.random.default_rng(seed)
    rows = []
    start_date = pd.Timestamp("2024-01-01")
    for race_idx in range(n_races):
        game_date = start_date + pd.Timedelta(days=race_idx * 3)
        n_horses = int(rng.integers(8, 13))
        streck = rng.dirichlet(np.ones(n_horses)) * 100
        win_probs = streck / streck.sum()
        winner_idx = rng.choice(n_horses, p=win_probs)
        gaps = compute_gap_to_rival(pd.Series(streck))

        for i in range(n_horses):
            rows.append(
                {
                    "game_id": f"SIM_{race_idx}",
                    "race_id": f"SIM_{race_idx}_1",
                    "game_date": game_date.date().isoformat(),
                    "game_type": "V75",
                    "track_name": "SimBanan",
                    "horse_id": race_idx * 100 + i,
                    "horse_name": f"Häst{i}",
                    "number": i + 1,
                    "streck_pct": float(streck[i]),
                    "gap_to_rival": float(gaps.iloc[i]),
                    "post_position": i + 1,
                    "start_method": "volte" if race_idx % 2 == 0 else "auto",
                    "added_distance_m": 0,
                    "num_starters": n_horses,
                    "track_condition": "normal",
                    "shoes_changed": float(rng.integers(0, 2)),
                    "barfota_back": float(rng.integers(0, 2)),
                    "driver_win_pct": float(rng.uniform(5, 30)),
                    "prize_low": 100000.0,
                    "prize_high": 300000.0,
                    "is_mare_race": 0.0,
                    "is_coldblood": 0.0,
                    "won": i == winner_idx,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def synthetic_dataset() -> pd.DataFrame:
    return _make_synthetic_dataset()


def test_train_evaluate_and_calibrate_on_synthetic_data(synthetic_dataset):
    train_df, validation_df = time_split(synthetic_dataset, validation_days=365)
    assert not train_df.empty
    assert not validation_df.empty

    model = train_model(train_df)
    metrics = evaluate(model, validation_df)

    assert metrics["n"] == len(validation_df)
    assert 0.0 < metrics["base_rate"] < 1.0
    assert np.isfinite(metrics["log_loss"])
    assert np.isfinite(metrics["brier_score"])
    # Streck is a genuine, strong signal in the synthetic data, so the model
    # should clearly beat random guessing.
    assert metrics["roc_auc"] > 0.6

    table = calibration_table(model, validation_df, n_bins=5)
    assert not table.empty
    assert {"predicted_mean", "actual_rate", "n", "calibration_gap"}.issubset(table.columns)
    assert table["n"].sum() == len(validation_df)


# --- score_leg / top_value_non_favorites ---------------------------------


@pytest.fixture()
def trained_model(synthetic_dataset):
    train_df, _ = time_split(synthetic_dataset, validation_days=365)
    return train_model(train_df)


def _make_starters_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "number": 1,
                "horse_name": "Ledaren",
                "streck_pct": 45.0,
                "post_position": 1,
                "added_distance_m": 0,
                "shoes_back": True,
                "shoes_changed": False,
                "driver_win_pct": 15.0,
                "scratched": False,
            },
            {
                "number": 2,
                "horse_name": "Utmanaren",
                "streck_pct": 8.0,
                "post_position": 2,
                "added_distance_m": 0,
                "shoes_back": False,
                "shoes_changed": True,
                "driver_win_pct": 35.0,
                "scratched": False,
            },
            {
                "number": 3,
                "horse_name": "TredjeHast",
                "streck_pct": 6.0,
                "post_position": 3,
                "added_distance_m": 20,
                "shoes_back": True,
                "shoes_changed": False,
                "driver_win_pct": 10.0,
                "scratched": False,
            },
            {
                "number": 4,
                "horse_name": "Struken",
                "streck_pct": 20.0,
                "post_position": 4,
                "added_distance_m": 0,
                "shoes_back": True,
                "shoes_changed": False,
                "driver_win_pct": 20.0,
                "scratched": True,
            },
        ]
    )


def _make_race_dict() -> dict:
    return {
        "startMethod": "volte",
        "track": {"name": "Testbanan", "condition": "normal"},
        "terms": ["3-åriga och äldre 100.001 - 300.000 kr."],
    }


def test_build_features_for_leg_excludes_scratched():
    features = build_features_for_leg(_make_starters_df(), _make_race_dict())
    assert set(features["number"]) == {1, 2, 3}
    assert set(ALL_FEATURES).issubset(features.columns)
    assert features["prize_low"].iloc[0] == pytest.approx(100001)


def test_score_leg_and_top_value_non_favorites(trained_model):
    scored = score_leg(trained_model, _make_starters_df(), _make_race_dict())

    assert len(scored) == 3  # scratched horse excluded
    assert "model_prob" in scored.columns
    assert "value_ratio" in scored.columns
    # model_prob should be normalised to sum to 1 across the (non-scratched) field.
    assert scored["model_prob"].sum() == pytest.approx(1.0)

    picks = top_value_non_favorites(scored, top_n=2)
    assert len(picks) == 2
    assert 1 not in picks["number"].to_list()  # the favorite (highest streck) is excluded
    # Sorted descending by value_ratio.
    assert picks["value_ratio"].is_monotonic_decreasing


def test_top_value_non_favorites_empty_input():
    assert top_value_non_favorites(pd.DataFrame()).empty
