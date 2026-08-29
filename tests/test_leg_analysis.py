import json
from pathlib import Path

import pandas as pd
import pytest

from atg_favorites.leg_analysis import (
    LegNotFoundError,
    build_comparison_pool,
    build_leg_report,
    class_group,
    current_favorite,
    distance_group,
    find_comparison_races,
    format_report,
    format_widening_summary,
    leg_starters,
    matches_hard_conditions,
    resolve_leg,
    summarize_comparison,
    summarize_widening_for_date,
    wilson_confidence_interval,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_game.json"


class StubClient:
    """A fake AtgClient serving the sample fixture without any network access."""

    def __init__(self, game: dict):
        self.game = game

    def get_calendar_day(self, date_str: str) -> dict:
        num_legs = len(self.game.get("races") or [])
        return {
            "date": date_str,
            "games": {
                "V75": [
                    {
                        "id": self.game["id"],
                        "status": self.game["status"],
                        "races": [f"dummy_{i}" for i in range(num_legs)],
                    }
                ],
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


# --- resolve_leg / leg_starters / current_favorite (unchanged behaviour) ---


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
    assert starters.iloc[0]["horse_name"] == "StorFavoriten"

    favorite = current_favorite(starters)
    assert favorite["horse_name"] == "StorFavoriten"
    assert favorite["streck_pct"] == pytest.approx(52.0)
    assert favorite["scratched"] is False


# --- distance_group / class_group -----------------------------------------


def test_distance_group_boundaries():
    assert distance_group(1899) == "kort (<1900m)"
    assert distance_group(1900) == "medel (1900-2199m)"
    assert distance_group(2140) == "medel (1900-2199m)"
    assert distance_group(2199) == "medel (1900-2199m)"
    assert distance_group(2200) == "lång (>=2200m)"
    assert distance_group(None) == "okänd distans"


def test_class_group_buckets():
    assert class_group(50_000, None) == "<100k kr"
    assert class_group(150_000, 300_000) == "100-300k kr"
    assert class_group(500_000, None) == "300-800k kr"
    assert class_group(1_000_000, None) == ">800k kr"
    assert class_group(None, None) == "okänd klass"
    # Falls back to prize_high when prize_low is missing (e.g. "högst X kr" villkor).
    assert class_group(None, 50_000) == "<100k kr"


# --- wilson_confidence_interval --------------------------------------------


def test_wilson_confidence_interval_zero_n():
    assert wilson_confidence_interval(0, 0) == (0.0, 0.0)


def test_wilson_confidence_interval_reasonable_bounds():
    lo, hi = wilson_confidence_interval(50, 100)
    assert 0.3 < lo < 0.5
    assert 0.5 < hi < 0.7
    assert lo < 0.5 < hi


def test_wilson_confidence_interval_narrows_with_more_data():
    lo_small, hi_small = wilson_confidence_interval(30, 100)
    lo_big, hi_big = wilson_confidence_interval(300, 1000)
    assert (hi_small - lo_small) > (hi_big - lo_big)


# --- matches_hard_conditions -----------------------------------------------


def _pool_row(**overrides) -> dict:
    row = {
        "sport": "trot",
        "is_coldblood": False,
        "is_mare_race": False,
        "track_name": "Bana A",
        "start_method": "volte",
        "distance_group": "medel (1900-2199m)",
        "class_group": "100-300k kr",
        "favorite_streck_pct": 32.0,
        "favorite_won": True,
        "favorite_top3": True,
        "favorite_horse_name": "Favoriten",
        "winner_horse_name": "Favoriten",
        "game_id": "g",
        "game_date": "2024-01-01",
    }
    row.update(overrides)
    return row


def test_matches_hard_conditions_filters_wrong_population():
    pool = pd.DataFrame(
        [
            _pool_row(game_id="ok"),
            _pool_row(game_id="wrong_sport", sport="monté"),
            _pool_row(game_id="wrong_breed", is_coldblood=True),
            _pool_row(game_id="wrong_sex_restriction", is_mare_race=True),
        ]
    )
    target_row = {"sport": "trot", "is_coldblood": False, "is_mare_race": False}
    matched = matches_hard_conditions(pool, target_row)
    assert list(matched["game_id"]) == ["ok"]


# --- find_comparison_races: soft-tier progression + hard-condition floor --


@pytest.fixture()
def comparison_pool() -> pd.DataFrame:
    rows: list[dict] = []

    def add(n: int, **overrides):
        for i in range(n):
            rows.append(_pool_row(game_id=f"{overrides.get('game_id', 'r')}_{i}", **{k: v for k, v in overrides.items() if k != "game_id"}))

    # Tier 0: matches everything (bana A, medel, 100-300k, volte, trot/varmblod/öppet).
    add(3, game_id="tier0")
    # Extra tier-1 matches: different track, everything else the same.
    add(4, game_id="tier1_extra", track_name="Bana B")
    # Extra tier-2 matches: different class too (still same distance/method).
    add(2, game_id="tier2_extra", track_name="Bana B", class_group="300-800k kr")
    # Extra tier-3 matches: different distance too (only start_method left).
    add(2, game_id="tier3_extra", track_name="Bana B", class_group="300-800k kr", distance_group="kort (<1900m)")
    # Never allowed to match: wrong start method (never relaxed).
    add(3, game_id="wrong_method", start_method="auto")
    # Never allowed to match: wrong hard conditions, even though everything else fits.
    add(2, game_id="wrong_mare", is_mare_race=True)
    add(2, game_id="wrong_sport", sport="monté")
    add(2, game_id="wrong_coldblood", is_coldblood=True)
    # Far away on streckprocent - excluded from the streck-bucket baseline fallback.
    # (start_method differs too, so it never leaks into the soft-tier steps either.)
    add(5, game_id="far_streck", favorite_streck_pct=80.0, start_method="auto")

    return pd.DataFrame(rows)


def _target_row(**overrides) -> dict:
    row = {
        "sport": "trot",
        "is_coldblood": False,
        "is_mare_race": False,
        "track_name": "Bana A",
        "start_method": "volte",
        "distance_group": "medel (1900-2199m)",
        "class_group": "100-300k kr",
        "favorite_streck_pct": 32.0,
    }
    row.update(overrides)
    return row


def test_find_comparison_races_step0_when_enough(comparison_pool):
    result = find_comparison_races(comparison_pool, _target_row(), min_n=3)
    assert result.step == 0
    assert result.n == 3
    assert not result.is_baseline
    assert set(result.races["game_id"].str.startswith("tier0")) == {True}


def test_find_comparison_races_relaxes_track_at_step1(comparison_pool):
    result = find_comparison_races(comparison_pool, _target_row(), min_n=4)
    assert result.step == 1
    assert result.n == 7  # tier0 (3) + tier1_extra (4)
    assert "alla banor" in result.description


def test_find_comparison_races_relaxes_class_at_step2(comparison_pool):
    result = find_comparison_races(comparison_pool, _target_row(), min_n=8)
    assert result.step == 2
    assert result.n == 9  # 3 + 4 + 2


def test_find_comparison_races_relaxes_distance_at_step3(comparison_pool):
    result = find_comparison_races(comparison_pool, _target_row(), min_n=10)
    assert result.step == 3
    assert result.n == 11  # 3 + 4 + 2 + 2
    # Wrong start method and wrong hard conditions must never be included, even here.
    assert set(result.races["game_id"]).isdisjoint({"wrong_method", "wrong_mare", "wrong_sport", "wrong_coldblood"})


def test_find_comparison_races_falls_back_to_baseline(comparison_pool):
    result = find_comparison_races(comparison_pool, _target_row(), min_n=999)
    assert result.step == 4
    assert result.is_baseline
    assert "baslinje" in result.description
    # The baseline ignores hard AND soft conditions entirely - everything with a
    # similar streckprocent counts, including the otherwise-excluded groups,
    # except the races with a very different (far away) streckprocent.
    assert result.n == 20  # everything except the 5 "far_streck" rows
    assert "far_streck" not in "".join(result.races["game_id"])


def test_find_comparison_races_empty_pool():
    result = find_comparison_races(pd.DataFrame(), _target_row())
    assert result.n == 0
    assert result.step == -1
    assert "inget historiskt underlag" in result.description


# --- summarize_comparison: n-thresholds ------------------------------------


def _races_with_win_rate(n: int, win_rate: float) -> pd.DataFrame:
    wins = int(round(n * win_rate))
    return pd.DataFrame(
        {
            "favorite_won": [True] * wins + [False] * (n - wins),
            "favorite_top3": [True] * n,
        }
    )


def test_summarize_comparison_insufficient_below_100():
    from atg_favorites.leg_analysis import ComparisonResult

    result = ComparisonResult(_races_with_win_rate(50, 0.4), "desc", step=0, is_baseline=False, n=50)
    stats = summarize_comparison(result)
    assert stats["display"] == "insufficient"
    assert stats["n"] == 50


def test_summarize_comparison_uncertain_between_100_and_300():
    from atg_favorites.leg_analysis import ComparisonResult

    result = ComparisonResult(_races_with_win_rate(150, 0.4), "desc", step=0, is_baseline=False, n=150)
    stats = summarize_comparison(result)
    assert stats["display"] == "uncertain"
    assert stats["win_rate"] == pytest.approx(0.4, abs=0.01)
    assert stats["ci_low"] < stats["win_rate"] < stats["ci_high"]


def test_summarize_comparison_ok_above_300():
    from atg_favorites.leg_analysis import ComparisonResult

    result = ComparisonResult(_races_with_win_rate(500, 0.35), "desc", step=1, is_baseline=False, n=500)
    stats = summarize_comparison(result)
    assert stats["display"] == "ok"


def test_summarize_comparison_zero_races():
    from atg_favorites.leg_analysis import ComparisonResult

    result = ComparisonResult(pd.DataFrame(), "inget historiskt underlag", step=-1, is_baseline=False, n=0)
    stats = summarize_comparison(result)
    assert stats == {"n": 0, "description": "inget historiskt underlag", "step": -1, "is_baseline": False, "display": "insufficient"}


# --- build_comparison_pool (from the real raw-JSON fixture) ---------------


def test_build_comparison_pool_from_fixture(tmp_path, sample_game):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "V75_2024-01-06_1_1.json").write_text(json.dumps(sample_game), encoding="utf-8")

    pool = build_comparison_pool(raw_dir)

    assert len(pool) == 2  # two finished races in the fixture
    assert set(pool["sport"]) == {"trot"}
    assert set(pool["is_mare_race"]) == {False}
    assert set(pool["is_coldblood"]) == {False}
    assert "distance_group" in pool.columns
    assert "class_group" in pool.columns


def test_build_comparison_pool_empty_dir(tmp_path):
    assert build_comparison_pool(tmp_path).empty


# --- build_leg_report / format_report (end to end, injected pool) ---------


def test_build_leg_report_with_injected_pool(stub_client, comparison_pool):
    report = build_leg_report(
        Path("unused"),
        date_str="2024-01-06",
        avd=2,
        game_type="V75",
        client=stub_client,
        min_sample_size=4,
        pool=comparison_pool,
    )

    assert report.target.game_id == "V75_2024-01-06_1_1"
    assert report.favorite["horse_name"] == "StorFavoriten"
    assert report.similarity_stats["n"] == 7
    assert report.similarity_stats["step"] == 1

    text = format_report(report)
    assert "Jämförelselopp" in text
    assert "n=7" in text


def test_build_leg_report_excludes_self_from_pool(stub_client, comparison_pool):
    # Tag one comparison-pool row with the *target* game's id and confirm it's dropped.
    tainted_pool = comparison_pool.copy()
    tainted_pool.loc[tainted_pool.index[0], "game_id"] = "V75_2024-01-06_1_1"

    report = build_leg_report(
        Path("unused"),
        date_str="2024-01-06",
        avd=2,
        game_type="V75",
        client=stub_client,
        min_sample_size=3,
        pool=tainted_pool,
    )
    assert "V75_2024-01-06_1_1" not in report.similar_races.get("game_id", pd.Series(dtype=object)).values


# --- widening summary for a whole day --------------------------------------


def test_summarize_widening_for_date(tmp_path, stub_client):
    summary = summarize_widening_for_date(tmp_path, date_str="2024-01-06", game_type="V75", client=stub_client)

    assert len(summary) == 2  # the fixture game has 2 avdelningar
    assert set(summary["avd"]) == {1, 2}
    # No raw archive at all -> no historical underlag for either leg.
    assert set(summary["step"]) == {-1}
    assert set(summary["step_label"]) == {"Inget historiskt underlag"}

    text = format_widening_summary(summary)
    assert "avd 1" in text
    assert "avd 2" in text


def test_format_widening_summary_empty():
    assert "Inga avdelningar" in format_widening_summary(pd.DataFrame())
