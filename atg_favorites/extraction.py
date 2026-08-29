"""Shared field-extraction helpers for one ATG "start" (a horse in a race).

Used by both ``flatten.py`` (building the historical ``races.csv``) and
``leg_analysis.py`` (analysing a single upcoming/live leg on demand), so the
exact same logic is used whether the race is finished or still to be run.
"""

from __future__ import annotations

from typing import Any


def streck_pct(start: dict[str, Any], pool_key: str) -> float | None:
    """Share (%) of the V75/V85/V86 pool staked on this horse ("streckprocent")."""
    pool = (start.get("pools") or {}).get(pool_key) or {}
    bet_distribution = pool.get("betDistribution")
    return None if bet_distribution is None else bet_distribution / 100.0


def final_odds(start: dict[str, Any]) -> float | None:
    """Final vinnare-odds if the race has finished, else the current live odds."""
    result = start.get("result") or {}
    if result.get("finalOdds") is not None:
        return float(result["finalOdds"])
    vinnare = (start.get("pools") or {}).get("vinnare") or {}
    odds = vinnare.get("odds")
    return None if odds is None else odds / 100.0


def shoe(horse: dict[str, Any], position: str, key: str) -> bool | None:
    shoes = horse.get("shoes") or {}
    if not shoes.get("reported"):
        return None
    part = shoes.get(position) or {}
    return part.get(key)


def sulky_type(horse: dict[str, Any]) -> str | None:
    sulky = horse.get("sulky") or {}
    if not sulky.get("reported"):
        return None
    return (sulky.get("type") or {}).get("code")


def full_name(person: dict[str, Any]) -> str | None:
    first, last = person.get("firstName"), person.get("lastName")
    if not first and not last:
        return None
    return " ".join(part for part in (first, last) if part)


def driver_name(start: dict[str, Any]) -> str | None:
    driver = start.get("driver") or {}
    return driver.get("shortName") or full_name(driver)


def trainer_name(horse: dict[str, Any]) -> str | None:
    trainer = horse.get("trainer") or {}
    return trainer.get("shortName") or full_name(trainer)


def win_pct(person: dict[str, Any], year: str) -> float | None:
    """A driver's/trainer's win% for ``year`` (e.g. "2026"), if reported."""
    years = (person.get("statistics") or {}).get("years") or {}
    stats = years.get(year) or {}
    pct = stats.get("winPercentage")
    return None if pct is None else pct / 100.0


def record_time_seconds(record: dict[str, Any] | None) -> float | None:
    """A record/km-time dict (``{minutes, seconds, tenths}``) as total seconds."""
    if not record:
        return None
    time = record.get("time") or {}
    minutes, seconds, tenths = time.get("minutes"), time.get("seconds"), time.get("tenths")
    if seconds is None:
        return None
    return (minutes or 0) * 60 + seconds + (tenths or 0) / 10.0


def record_time_str(record: dict[str, Any] | None) -> str | None:
    """Human-readable Swedish trot record string, e.g. ``"1.13,6a"``."""
    if not record:
        return None
    time = record.get("time") or {}
    minutes, seconds, tenths = time.get("minutes"), time.get("seconds"), time.get("tenths")
    if seconds is None:
        return None
    return f"{minutes or 0}.{seconds:02d},{tenths or 0}"


def life_stats(horse: dict[str, Any]) -> dict[str, Any]:
    """Life-time starts/earnings/win% for a horse, from the embedded statistics."""
    life = (horse.get("statistics") or {}).get("life") or {}
    starts = life.get("starts")
    wins = (life.get("placement") or {}).get("1")
    win_percentage = (wins / starts * 100.0) if starts else None
    return {
        "life_starts": starts,
        "life_earnings": life.get("earnings"),
        "life_wins": wins,
        "life_win_pct": win_percentage,
    }


def is_scratched(start: dict[str, Any], race: dict[str, Any]) -> bool:
    scratchings = (race.get("result") or {}).get("scratchings") or []
    return start.get("number") in scratchings


def start_summary(start: dict[str, Any], race: dict[str, Any], pool_key: str) -> dict[str, Any]:
    """Flatten one starter (a horse in a race) into a plain dict.

    Works for both finished races (``result`` populated) and
    upcoming/bettable/ongoing races (``result`` absent), so it is shared
    between the historical flatten pipeline and the live single-leg
    analysis.
    """
    horse = start.get("horse") or {}
    driver = start.get("driver") or {}
    trainer = horse.get("trainer") or {}
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
        "driver_name": driver_name(start),
        "driver_win_pct": win_pct(driver, str(race.get("date", ""))[:4]),
        "trainer_name": trainer_name(horse),
        "trainer_win_pct": win_pct(trainer, str(race.get("date", ""))[:4]),
        "streck_pct": streck_pct(start, pool_key),
        "added_distance_m": added_distance,
        "shoes_front": shoe(horse, "front", "hasShoe"),
        "shoes_back": shoe(horse, "back", "hasShoe"),
        "shoes_changed": shoe(horse, "front", "changed") or shoe(horse, "back", "changed"),
        "sulky_type": sulky_type(horse),
        "final_odds": final_odds(start),
        "record_time": record_time_str(horse.get("record")),
        "record_time_seconds": record_time_seconds(horse.get("record")),
        **life_stats(horse),
        "place": result.get("place"),
        "finish_order": result.get("finishOrder"),
        "galloped": result.get("galloped", False),
        "disqualified": result.get("disqualified", False),
        "scratched": is_scratched(start, race),
    }
