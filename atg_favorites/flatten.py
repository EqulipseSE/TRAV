"""Flatten raw ATG game JSON into one CSV row per finished race (lopp).

Each row describes one race and its pre-race favorite (the horse with the
highest "streckprocent" - the share of the V75/V85/V86 pool staked on it,
``pools.<GAME_TYPE>.betDistribution`` in the raw JSON): its post position
("spår"), any added start distance ("tillägg"), shoeing ("skor"), sulky,
odds, and how it actually finished.

Run as a script::

    python -m atg_favorites.flatten --raw-dir data/raw --out data/processed/races.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from atg_favorites.config import FINISHED_STATUSES, RACES_CSV, RAW_DIR

logger = logging.getLogger(__name__)


def _streck_pct(start: dict[str, Any], pool_key: str) -> float | None:
    pool = (start.get("pools") or {}).get(pool_key) or {}
    bet_distribution = pool.get("betDistribution")
    return None if bet_distribution is None else bet_distribution / 100.0


def _final_odds(start: dict[str, Any]) -> float | None:
    result = start.get("result") or {}
    if result.get("finalOdds") is not None:
        return float(result["finalOdds"])
    vinnare = (start.get("pools") or {}).get("vinnare") or {}
    odds = vinnare.get("odds")
    return None if odds is None else odds / 100.0


def _shoe(horse: dict[str, Any], position: str, key: str) -> bool | None:
    shoes = horse.get("shoes") or {}
    if not shoes.get("reported"):
        return None
    part = shoes.get(position) or {}
    return part.get(key)


def _sulky_type(horse: dict[str, Any]) -> str | None:
    sulky = horse.get("sulky") or {}
    if not sulky.get("reported"):
        return None
    return (sulky.get("type") or {}).get("code")


def _driver_name(start: dict[str, Any]) -> str | None:
    driver = start.get("driver") or {}
    return driver.get("shortName") or _full_name(driver)


def _trainer_name(horse: dict[str, Any]) -> str | None:
    trainer = horse.get("trainer") or {}
    return trainer.get("shortName") or _full_name(trainer)


def _full_name(person: dict[str, Any]) -> str | None:
    first, last = person.get("firstName"), person.get("lastName")
    if not first and not last:
        return None
    return " ".join(part for part in (first, last) if part)


def is_scratched(start: dict[str, Any], race: dict[str, Any]) -> bool:
    scratchings = (race.get("result") or {}).get("scratchings") or []
    return start.get("number") in scratchings


def _start_summary(start: dict[str, Any], race: dict[str, Any], pool_key: str) -> dict[str, Any]:
    horse = start.get("horse") or {}
    result = start.get("result") or {}
    race_distance = race.get("distance")
    start_distance = start.get("distance")
    added_distance = (
        start_distance - race_distance
        if start_distance is not None and race_distance is not None
        else None
    )
    return {
        "number": start.get("number"),
        "post_position": start.get("postPosition"),
        "horse_id": horse.get("id"),
        "horse_name": horse.get("name"),
        "horse_age": horse.get("age"),
        "horse_sex": horse.get("sex"),
        "driver_name": _driver_name(start),
        "trainer_name": _trainer_name(horse),
        "streck_pct": _streck_pct(start, pool_key),
        "added_distance_m": added_distance,
        "shoes_front": _shoe(horse, "front", "hasShoe"),
        "shoes_back": _shoe(horse, "back", "hasShoe"),
        "shoes_changed": _shoe(horse, "front", "changed") or _shoe(horse, "back", "changed"),
        "sulky_type": _sulky_type(horse),
        "final_odds": _final_odds(start),
        "place": result.get("place"),
        "finish_order": result.get("finishOrder"),
        "galloped": result.get("galloped", False),
        "disqualified": result.get("disqualified", False),
        "scratched": is_scratched(start, race),
    }


def flatten_race(race: dict[str, Any], game: dict[str, Any]) -> dict[str, Any] | None:
    """Return one flattened row for ``race``, or ``None`` if it isn't finished/usable."""
    if race.get("status") not in FINISHED_STATUSES:
        return None

    pool_key = game.get("type")
    starts = race.get("starts") or []
    if not starts or not pool_key:
        return None

    summaries = [_start_summary(s, race, pool_key) for s in starts]
    runners = [s for s in summaries if not s["scratched"]]
    with_streck = [s for s in runners if s["streck_pct"] is not None]

    if not with_streck:
        return None

    ranked = sorted(with_streck, key=lambda s: s["streck_pct"], reverse=True)
    favorite = ranked[0]
    second_favorite = ranked[1] if len(ranked) > 1 else None

    winner = next((s for s in summaries if s["place"] == 1), None)

    track = race.get("track") or {}
    row = {
        "game_id": game.get("id"),
        "game_type": pool_key,
        "game_date": race.get("date"),
        "track_id": track.get("id"),
        "track_name": track.get("name"),
        "track_country": track.get("countryCode"),
        "track_condition": track.get("condition"),
        "sport": race.get("sport"),
        "race_id": race.get("id"),
        "race_number": race.get("number"),
        "race_name": race.get("name"),
        "race_distance_m": race.get("distance"),
        "start_method": race.get("startMethod"),
        "scheduled_start_time": race.get("scheduledStartTime"),
        "num_starters": len(summaries),
        "num_scratched": sum(1 for s in summaries if s["scratched"]),
        "favorite_number": favorite["number"],
        "favorite_post_position": favorite["post_position"],
        "favorite_horse_id": favorite["horse_id"],
        "favorite_horse_name": favorite["horse_name"],
        "favorite_horse_age": favorite["horse_age"],
        "favorite_horse_sex": favorite["horse_sex"],
        "favorite_driver": favorite["driver_name"],
        "favorite_trainer": favorite["trainer_name"],
        "favorite_streck_pct": favorite["streck_pct"],
        "favorite_added_distance_m": favorite["added_distance_m"],
        "favorite_shoes_front": favorite["shoes_front"],
        "favorite_shoes_back": favorite["shoes_back"],
        "favorite_shoes_changed": favorite["shoes_changed"],
        "favorite_sulky_type": favorite["sulky_type"],
        "favorite_final_odds": favorite["final_odds"],
        "favorite_place": favorite["place"],
        "favorite_finish_order": favorite["finish_order"],
        "favorite_galloped": favorite["galloped"],
        "favorite_disqualified": favorite["disqualified"],
        "favorite_won": favorite["place"] == 1,
        "favorite_top3": favorite["place"] in (1, 2, 3) if favorite["place"] is not None else False,
        "second_favorite_horse_name": second_favorite["horse_name"] if second_favorite else None,
        "second_favorite_streck_pct": second_favorite["streck_pct"] if second_favorite else None,
        "favorite_streck_margin": (
            favorite["streck_pct"] - second_favorite["streck_pct"] if second_favorite else None
        ),
        "winner_number": winner["number"] if winner else None,
        "winner_horse_name": winner["horse_name"] if winner else None,
        "winner_streck_pct": winner["streck_pct"] if winner else None,
        "winner_is_favorite": bool(winner and winner["number"] == favorite["number"]),
    }
    return row


def flatten_game(game: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten every finished race in ``game`` into a list of row dicts."""
    rows = []
    for race in game.get("races") or []:
        row = flatten_race(race, game)
        if row is not None:
            rows.append(row)
    return rows


def load_raw_games(raw_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Load every ``*.json`` raw game file in ``raw_dir``."""
    games = []
    for path in sorted(raw_dir.glob("*.json")):
        try:
            games.append((path, json.loads(path.read_text(encoding="utf-8"))))
        except json.JSONDecodeError as exc:
            logger.warning("Skipping unreadable raw file %s: %s", path, exc)
    return games


def build_dataframe(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Flatten every raw game file under ``raw_dir`` into a single DataFrame."""
    rows: list[dict[str, Any]] = []
    for path, game in load_raw_games(raw_dir):
        game_rows = flatten_game(game)
        for row in game_rows:
            row["source_file"] = path.name
        rows.extend(game_rows)

    columns = [
        "game_id",
        "game_type",
        "game_date",
        "track_id",
        "track_name",
        "track_country",
        "track_condition",
        "sport",
        "race_id",
        "race_number",
        "race_name",
        "race_distance_m",
        "start_method",
        "scheduled_start_time",
        "num_starters",
        "num_scratched",
        "favorite_number",
        "favorite_post_position",
        "favorite_horse_id",
        "favorite_horse_name",
        "favorite_horse_age",
        "favorite_horse_sex",
        "favorite_driver",
        "favorite_trainer",
        "favorite_streck_pct",
        "favorite_added_distance_m",
        "favorite_shoes_front",
        "favorite_shoes_back",
        "favorite_shoes_changed",
        "favorite_sulky_type",
        "favorite_final_odds",
        "favorite_place",
        "favorite_finish_order",
        "favorite_galloped",
        "favorite_disqualified",
        "favorite_won",
        "favorite_top3",
        "second_favorite_horse_name",
        "second_favorite_streck_pct",
        "favorite_streck_margin",
        "winner_number",
        "winner_horse_name",
        "winner_streck_pct",
        "winner_is_favorite",
        "source_file",
    ]
    df = pd.DataFrame(rows, columns=columns)
    if not df.empty:
        df = df.sort_values(["game_date", "game_id", "race_number"]).reset_index(drop=True)
    return df


def save_races_csv(df: pd.DataFrame, out_path: Path = RACES_CSV) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help=f"Directory with raw JSON (default: {RAW_DIR})")
    parser.add_argument("--out", type=Path, default=RACES_CSV, help=f"CSV output path (default: {RACES_CSV})")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    df = build_dataframe(args.raw_dir)
    if df.empty:
        logger.warning("No finished races found in %s - nothing written.", args.raw_dir)
        return

    out_path = save_races_csv(df, args.out)
    logger.info("Wrote %d rows to %s", len(df), out_path)


if __name__ == "__main__":
    main()
