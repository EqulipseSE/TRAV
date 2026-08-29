"""Analysera en enskild avdelning (lopp) - avgjord, pågående eller kommande.

Given a date and an "avdelning" (leg number within a V75/V85/V86 round),
this module:

1. Fetches that leg live from ATG (works whether the round is upcoming,
   bettable, ongoing or already finished).
2. Extracts every starter with its current streckprocent, spår, tillägg,
   skor, sulky, odds and background stats (driver/trainer win%, hästens
   livstidsstatistik och rekord).
3. Finds historical **jämförelselopp** (comparison races) in the local raw
   archive (``data/raw/*.json``) using a two-stage selection:

   * **Hårda villkor** (never relaxed): hästtyp (varmblod/kallblod),
     köns­restriktion (sto-/öppet lopp) and körsätt (sulky/monté) must match
     exactly - a comparison race with the wrong population is worse than
     no comparison race at all.
   * **Mjuka villkor** (relaxed stepwise until there's enough data):
     startmetod + distansgrupp + klassgrupp + bana -> drop bana -> drop
     klassgrupp -> drop distansgrupp (startmetod is never dropped).

   If even the most relaxed step still has under 100 races, no rate is
   shown as "similar-race" statistics; instead a **global baseline** (all
   race types, bucketed only by the favorite's own streckprocent) is used,
   clearly labelled as such.

Run as a script::

    python -m atg_favorites.leg_analysis --date 2026-08-29 --avd 5
    python -m atg_favorites.leg_analysis --date 2026-08-29 --avd 3 --game-type V86
    python -m atg_favorites.leg_analysis --date 2026-08-29   # (no --avd): widening-summary
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from atg_favorites.api_client import AtgApiError, AtgClient
from atg_favorites.config import DEFAULT_BUCKET_EDGES, GAME_TYPES, RAW_DIR
from atg_favorites.extraction import parse_race_class, start_summary
from atg_favorites.flatten import flatten_race, load_raw_games

logger = logging.getLogger(__name__)

#: Under this many jämförelselopp: show NO percentage, just "otillräckligt underlag".
MIN_SAMPLE_SIZE = 100
#: Between MIN_SAMPLE_SIZE and this: show the percentage, but flag it as uncertain.
UNCERTAIN_SAMPLE_SIZE = 300

#: Distansgrupp-gränser (meter): kort < 1900, lång >= 2200, annars medel.
DISTANCE_SHORT_MAX = 1900
DISTANCE_LONG_MIN = 2200

#: Klassgrupp-gränser (kr), grovt bucketat på prissummans undre (eller övre) gräns.
CLASS_GROUP_EDGES = (0, 100_000, 300_000, 800_000, float("inf"))
CLASS_GROUP_LABELS = ("<100k kr", "100-300k kr", "300-800k kr", ">800k kr")

#: De HÅRDA villkoren - kolumner i jämförelsepoolen som aldrig får avvika.
HARD_CONDITION_COLUMNS: tuple[str, ...] = ("sport", "is_coldblood", "is_mare_race")

#: De MJUKA villkoren, strikast till mest uppslappnad. Startmetod ingår i alla
#: (den släpps aldrig); varje steg släpper ytterligare ett villkor.
SOFT_TIERS: tuple[tuple[str, dict[str, bool]], ...] = (
    ("bana+distansgrupp+klassgrupp+startmetod", {"match_track": True, "match_class": True, "match_distance": True}),
    ("distansgrupp+klassgrupp+startmetod (alla banor)", {"match_track": False, "match_class": True, "match_distance": True}),
    ("distansgrupp+startmetod (alla banor/klasser)", {"match_track": False, "match_class": False, "match_distance": True}),
    ("endast startmetod (alla banor/klasser/distanser)", {"match_track": False, "match_class": False, "match_distance": False}),
)

#: Textetiketter för vilket steg (0-3 = mjukt villkor-steg, 4 = baslinje) som användes.
STEP_LABELS: dict[int, str] = {
    0: "Steg 0 (bana+distans+klass+startmetod)",
    1: "Steg 1 (distans+klass+startmetod, alla banor)",
    2: "Steg 2 (distans+startmetod, alla banor/klasser)",
    3: "Steg 3 (endast startmetod)",
    4: "Baslinje (globalt streckintervall, oavsett lopptyp)",
    -1: "Inget historiskt underlag",
}


class LegNotFoundError(RuntimeError):
    """Raised when the requested date/avdelning/game type can't be resolved."""


@dataclass
class LegTarget:
    """The resolved leg (avdelning) to analyse."""

    game_id: str
    game_type: str
    avd: int
    race: dict[str, Any]
    pool_key: str


@dataclass
class ComparisonResult:
    """The outcome of selecting jämförelselopp for one target race."""

    races: pd.DataFrame
    description: str
    step: int  # 0-3 = mjukt-villkor-steg, 4 = baslinje, -1 = inget underlag alls
    is_baseline: bool
    n: int


@dataclass
class LegReport:
    target: LegTarget
    starters: pd.DataFrame
    favorite: dict[str, Any]
    similar_races: pd.DataFrame
    similarity_description: str
    similarity_stats: dict[str, Any] = field(default_factory=dict)
    favoritfall_examples: pd.DataFrame = field(default_factory=pd.DataFrame)


# --- Kalenderuppslag / avdelning ------------------------------------------


def find_candidate_games(
    client: AtgClient,
    date_str: str,
    game_type: str | None = None,
) -> list[dict[str, Any]]:
    """Return every V75/V85/V86 game on ``date_str``, optionally restricted to ``game_type``.

    Calendar entries only carry numeric track ids, not names, so filtering by
    track name happens later in :func:`resolve_leg` once each candidate
    game's full detail (with the real track name) has been fetched.
    """
    calendar = client.get_calendar_day(date_str)
    games_by_type = calendar.get("games", {})

    candidates: list[dict[str, Any]] = []
    types_to_check = (game_type,) if game_type else GAME_TYPES
    for gtype in types_to_check:
        for game in games_by_type.get(gtype, []):
            candidates.append({**game, "type": gtype})
    return candidates


def iter_legs_for_date(
    client: AtgClient,
    date_str: str,
    game_type: str | None = None,
) -> list[tuple[str, int]]:
    """Every ``(game_type, avd)`` pair scheduled on ``date_str``."""
    candidates = find_candidate_games(client, date_str, game_type=game_type)
    legs: list[tuple[str, int]] = []
    for candidate in candidates:
        num_legs = len(candidate.get("races") or [])
        legs.extend((candidate["type"], avd) for avd in range(1, num_legs + 1))
    return legs


def resolve_leg(
    client: AtgClient,
    date_str: str,
    avd: int,
    game_type: str | None = None,
    track_name: str | None = None,
) -> LegTarget:
    """Resolve a date + avdelning (+ optional game type/bana) to a concrete leg."""
    candidates = find_candidate_games(client, date_str, game_type=game_type)
    if not candidates:
        raise LegNotFoundError(
            f"Hittade inga V75/V85/V86-omgångar för {date_str}"
            + (f" (spelform {game_type})" if game_type else "") + "."
        )

    matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for game_summary in candidates:
        try:
            game = client.get_game(game_summary["id"])
        except AtgApiError as exc:
            logger.warning("Kunde inte hämta %s: %s", game_summary["id"], exc)
            continue
        races = game.get("races") or []
        if avd < 1 or avd > len(races):
            continue
        race = races[avd - 1]
        if track_name:
            actual_track = ((race.get("track") or {}).get("name") or "").strip().lower()
            if track_name.strip().lower() not in actual_track:
                continue
        matches.append((game_summary, game))

    if not matches:
        available = ", ".join(f"{c['type']} ({c['id']})" for c in candidates)
        raise LegNotFoundError(
            f"Hittade ingen avdelning {avd} för {date_str} som matchar filtren. "
            f"Tillgängliga omgångar den dagen: {available}."
        )
    if len(matches) > 1:
        ids = ", ".join(g["id"] for _, g in matches)
        raise LegNotFoundError(
            f"Flera omgångar matchar - ange --game-type för att välja en av: {ids}."
        )

    game_summary, game = matches[0]
    race = (game.get("races") or [])[avd - 1]
    return LegTarget(
        game_id=game["id"],
        game_type=game_summary["type"],
        avd=avd,
        race=race,
        pool_key=game_summary["type"],
    )


def leg_starters(target: LegTarget) -> pd.DataFrame:
    """All starters in the target leg as a DataFrame, ranked by streckprocent."""
    starts = target.race.get("starts") or []
    rows = [start_summary(s, target.race, target.pool_key) for s in starts]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("streck_pct", ascending=False, na_position="last").reset_index(drop=True)
    return df


def current_favorite(starters: pd.DataFrame) -> dict[str, Any] | None:
    runners = starters[~starters["scratched"] & starters["streck_pct"].notna()]
    if runners.empty:
        return None
    return runners.iloc[0].to_dict()


# --- Loppklassificering (hårda + mjuka villkor) ---------------------------


def distance_group(distance_m: float | None) -> str:
    """kort (<1900m) / medel / lång (>=2200m)."""
    if distance_m is None:
        return "okänd distans"
    if distance_m < DISTANCE_SHORT_MAX:
        return "kort (<1900m)"
    if distance_m >= DISTANCE_LONG_MIN:
        return "lång (>=2200m)"
    return "medel (1900-2199m)"


def class_group(prize_low: float | None, prize_high: float | None) -> str:
    """Grov bucketering av loppets prissummeklass, för mjuk matchning."""
    reference = prize_low if prize_low is not None else prize_high
    if reference is None:
        return "okänd klass"
    for edge, label in zip(CLASS_GROUP_EDGES[1:], CLASS_GROUP_LABELS):
        if reference < edge:
            return label
    return CLASS_GROUP_LABELS[-1]


def _race_classification(race: dict[str, Any]) -> dict[str, Any]:
    """Hårda + mjuka klassificeringsfält för ett enskilt lopp (rå-JSON)."""
    race_class = parse_race_class(race.get("terms"))
    distance_m = race.get("distance")
    return {
        "sport": race.get("sport"),
        "is_coldblood": race_class["is_coldblood"],
        "is_mare_race": race_class["is_mare_race"],
        "prize_low": race_class["prize_low"],
        "prize_high": race_class["prize_high"],
        "distance_group": distance_group(distance_m),
        "class_group": class_group(race_class["prize_low"], race_class["prize_high"]),
    }


def build_comparison_pool(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Every finished historical race, enriched with hästtyp/köns-/klassfält.

    Bygger på samma per-lopp-rader som ``flatten.py`` producerar (favorit,
    streck, resultat m.m.), men utökat med de fält som krävs för HÅRDA
    villkoren (varmblod/kallblod, sto-/öppet lopp, sulky/monté) och MJUKA
    villkoren (distansgrupp, klassgrupp) i jämförelselopps-urvalet - allt
    parsat direkt ur ``race.terms``/``race.sport``.
    """
    rows: list[dict[str, Any]] = []
    for _path, game in load_raw_games(raw_dir):
        for race in game.get("races") or []:
            row = flatten_race(race, game)
            if row is None:
                continue
            row.update(_race_classification(race))
            rows.append(row)
    return pd.DataFrame(rows)


def _build_target_row(target: LegTarget, favorite: dict[str, Any]) -> dict[str, Any]:
    race = target.race
    track = race.get("track") or {}
    classification = _race_classification(race)
    return {
        "track_name": track.get("name"),
        "start_method": race.get("startMethod"),
        "race_distance_m": race.get("distance"),
        "favorite_streck_pct": favorite.get("streck_pct"),
        **classification,
    }


# --- Urval av jämförelselopp ----------------------------------------------


def matches_hard_conditions(pool: pd.DataFrame, target_row: dict[str, Any]) -> pd.DataFrame:
    """Filtrera till lopp som matchar ALLA hårda villkor (aldrig uppslappnat)."""
    mask = pd.Series(True, index=pool.index)
    for column in HARD_CONDITION_COLUMNS:
        if column in pool.columns:
            mask &= pool[column] == target_row.get(column)
    return pool.loc[mask]


def _apply_soft_filters(
    pool: pd.DataFrame,
    target_row: dict[str, Any],
    match_track: bool,
    match_class: bool,
    match_distance: bool,
) -> pd.DataFrame:
    mask = pd.Series(True, index=pool.index)
    if target_row.get("start_method"):
        mask &= pool["start_method"] == target_row["start_method"]
    if match_track and target_row.get("track_name"):
        mask &= pool["track_name"] == target_row["track_name"]
    if match_distance:
        mask &= pool["distance_group"] == target_row["distance_group"]
    if match_class:
        mask &= pool["class_group"] == target_row["class_group"]
    return pool.loc[mask]


def _hard_condition_labels(target_row: dict[str, Any]) -> list[str]:
    sport = target_row.get("sport")
    return [
        "kallblod" if target_row.get("is_coldblood") else "varmblod",
        "stolopp" if target_row.get("is_mare_race") else "öppet lopp",
        "montélopp" if sport == "monté" else "sulkylopp",
    ]


def _soft_condition_labels(
    target_row: dict[str, Any], match_track: bool, match_class: bool, match_distance: bool
) -> list[str]:
    start_method = target_row.get("start_method")
    method_label = {"auto": "autostart", "volte": "voltstart"}.get(start_method, str(start_method))
    return [
        method_label,
        target_row["distance_group"] if match_distance else "alla distanser",
        target_row["class_group"] if match_class else "alla klasser",
        target_row["track_name"] if (match_track and target_row.get("track_name")) else "alla banor",
    ]


def _streck_bucket_bounds(streck_pct: float, edges: tuple[int, ...] = DEFAULT_BUCKET_EDGES) -> tuple[float, float]:
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= streck_pct < hi:
            return float(lo), float(hi)
    if streck_pct >= edges[-1]:
        return float(edges[-2]), float(edges[-1])
    return float(edges[0]), float(edges[1])


def find_comparison_races(
    pool: pd.DataFrame,
    target_row: dict[str, Any],
    min_n: int = MIN_SAMPLE_SIZE,
) -> ComparisonResult:
    """Select jämförelselopp: hårda villkor alltid, mjuka villkor stegvis uppslappnade.

    Om inte ens det mest uppslappnade mjuka steget ger ``min_n`` lopp inom de
    hårda villkoren, faller urvalet tillbaka på en global baslinje (alla
    lopptyper, bucketerad enbart på favoritens egen streckprocent).
    """
    if pool.empty:
        return ComparisonResult(pool, "inget historiskt underlag inläst än", step=-1, is_baseline=False, n=0)

    hard_pool = matches_hard_conditions(pool, target_row)
    hard_labels = _hard_condition_labels(target_row)

    last_result = hard_pool.iloc[0:0]
    last_description = ""
    for step_idx, (_label, opts) in enumerate(SOFT_TIERS):
        filtered = _apply_soft_filters(hard_pool, target_row, **opts)
        soft_labels = _soft_condition_labels(target_row, **opts)
        description = ", ".join(hard_labels + soft_labels + [f"n={len(filtered)}"])
        last_result, last_description = filtered, description
        if len(filtered) >= min_n:
            return ComparisonResult(filtered, description, step=step_idx, is_baseline=False, n=len(filtered))

    # Even the most relaxed soft tier (still within the hard conditions)
    # didn't reach min_n - fall back to the GLOBAL baseline for the
    # favorite's own streckprocent-intervall, ignoring hard+soft villkor.
    favorite_streck_pct = target_row.get("favorite_streck_pct")
    if favorite_streck_pct is not None:
        lo, hi = _streck_bucket_bounds(favorite_streck_pct)
        baseline = pool.loc[pool["favorite_streck_pct"].between(lo, hi)]
        description = f"baslinje (alla lopptyper, streckintervall {lo:.0f}-{hi:.0f}%), n={len(baseline)}"
        return ComparisonResult(baseline, description, step=4, is_baseline=True, n=len(baseline))

    return ComparisonResult(last_result, last_description, step=3, is_baseline=False, n=len(last_result))


def wilson_confidence_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """95% Wilson score-intervall för en binomial andel (robust även för litet n)."""
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1 + z**2 / n
    center = phat + z**2 / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return (max(0.0, (center - margin) / denom), min(1.0, (center + margin) / denom))


def summarize_comparison(result: ComparisonResult) -> dict[str, Any]:
    """Redovisningsklara mått: n-tröskelbaserad visningsläge + 95% KI.

    ``display`` är ``"insufficient"`` (n<100, ingen procentsiffra visas),
    ``"uncertain"`` (100<=n<300, siffra visas men flaggas osäker) eller
    ``"ok"`` (n>=300).
    """
    n = result.n
    base = {"n": n, "description": result.description, "step": result.step, "is_baseline": result.is_baseline}
    if n == 0:
        return {**base, "display": "insufficient"}

    races = result.races
    wins = int(races["favorite_won"].sum())
    win_rate = wins / n
    ci_low, ci_high = wilson_confidence_interval(wins, n)
    top3_rate = float(races["favorite_top3"].mean()) if "favorite_top3" in races.columns else None

    if n < MIN_SAMPLE_SIZE:
        display = "insufficient"
    elif n < UNCERTAIN_SAMPLE_SIZE:
        display = "uncertain"
    else:
        display = "ok"

    return {
        **base,
        "display": display,
        "win_rate": win_rate,
        "top3_rate": top3_rate,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def build_leg_report(
    raw_dir: Path = RAW_DIR,
    *,
    date_str: str,
    avd: int,
    game_type: str | None = None,
    track_name: str | None = None,
    client: AtgClient | None = None,
    min_sample_size: int = MIN_SAMPLE_SIZE,
    pool: pd.DataFrame | None = None,
) -> LegReport:
    """Bygg en fullständig avdelningsrapport.

    ``pool`` kan skickas in (t.ex. förberäknad/cachad av anroparen) för att
    slippa läsa om ``data/raw`` för varje avdelning - annars byggs den från
    ``raw_dir``.
    """
    client = client or AtgClient()
    target = resolve_leg(client, date_str, avd, game_type=game_type, track_name=track_name)

    starters = leg_starters(target)
    favorite = current_favorite(starters) or {}
    target_row = _build_target_row(target, favorite)

    if pool is None:
        pool = build_comparison_pool(Path(raw_dir)) if Path(raw_dir).exists() else pd.DataFrame()
    if not pool.empty:
        pool = pool.copy()
        pool["favorite_won"] = pool["favorite_won"].astype(bool)
        pool["favorite_top3"] = pool["favorite_top3"].astype(bool)
        # Never "match" a race against itself if it happens to already be
        # in the historical archive (e.g. re-analysing a finished round).
        pool = pool[pool["game_id"] != target.game_id]

    result = find_comparison_races(pool, target_row, min_n=min_sample_size)
    stats = summarize_comparison(result)

    favoritfall_examples = pd.DataFrame()
    if result.n and not result.races.empty:
        favoritfall_examples = (
            result.races.loc[
                ~result.races["favorite_won"],
                [
                    c
                    for c in ("game_date", "track_name", "favorite_horse_name", "favorite_streck_pct", "winner_horse_name")
                    if c in result.races.columns
                ],
            ]
            .sort_values("favorite_streck_pct", ascending=False)
            .head(10)
        )

    return LegReport(
        target=target,
        starters=starters,
        favorite=favorite,
        similar_races=result.races,
        similarity_description=stats["description"],
        similarity_stats=stats,
        favoritfall_examples=favoritfall_examples,
    )


# --- Breddningssammanfattning för en hel dag ------------------------------


def summarize_widening_for_date(
    raw_dir: Path = RAW_DIR,
    *,
    date_str: str,
    game_type: str | None = None,
    client: AtgClient | None = None,
    min_sample_size: int = MIN_SAMPLE_SIZE,
) -> pd.DataFrame:
    """För varje avdelning schemalagd ``date_str``: vilket urvalssteg krävdes?

    En rad per avdelning med kolumnerna ``game_type``, ``avd``,
    ``track_name``, ``step``, ``step_label`` och ``n``.
    """
    client = client or AtgClient()
    pool = build_comparison_pool(Path(raw_dir)) if Path(raw_dir).exists() else pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for gtype, avd in iter_legs_for_date(client, date_str, game_type=game_type):
        try:
            report = build_leg_report(
                raw_dir,
                date_str=date_str,
                avd=avd,
                game_type=gtype,
                client=client,
                min_sample_size=min_sample_size,
                pool=pool,
            )
        except (LegNotFoundError, AtgApiError) as exc:
            logger.warning("Kunde inte analysera %s avd %d: %s", gtype, avd, exc)
            continue

        step = report.similarity_stats.get("step", -1)
        rows.append(
            {
                "game_type": gtype,
                "avd": avd,
                "track_name": (report.target.race.get("track") or {}).get("name"),
                "step": step,
                "step_label": STEP_LABELS.get(step, str(step)),
                "n": report.similarity_stats.get("n", 0),
            }
        )
    return pd.DataFrame(rows)


# --- Textrapportering ------------------------------------------------------


def format_report(report: LegReport) -> str:
    """Render a :class:`LegReport` as a readable plain-text report."""
    race = report.target.race
    track = race.get("track") or {}
    lines = [
        f"{report.target.game_type} avd {report.target.avd} ({report.target.game_id})",
        f"{track.get('name', '?')} - {race.get('name', '')}".strip(" -"),
        f"Distans: {race.get('distance', '?')} m, startsätt: {race.get('startMethod', '?')}, "
        f"bankondition: {track.get('condition', '?')}",
        "",
        "Startande (rankade efter streckprocent):",
    ]
    for _, s in report.starters.iterrows():
        streck = f"{s['streck_pct']:.1f}%" if pd.notna(s.get("streck_pct")) else "–"
        scratched = " (STRUKEN)" if s.get("scratched") else ""
        shoe_bits = []
        if pd.notna(s.get("shoes_front")):
            shoe_bits.append(f"fram {'sko' if s['shoes_front'] else 'barfota'}")
        if pd.notna(s.get("shoes_back")):
            shoe_bits.append(f"bak {'sko' if s['shoes_back'] else 'barfota'}")
        shoes = ", ".join(shoe_bits)
        added = f"+{int(s['added_distance_m'])}m" if pd.notna(s.get("added_distance_m")) and s.get("added_distance_m") else ""
        lines.append(
            f"  {int(s['number']):>2}. {s['horse_name']:<20} streck {streck:>6}  "
            f"spår {s.get('post_position', '?')} {added:<5} {shoes}{scratched}"
        )

    fav = report.favorite
    lines += ["", "Favorit:"]
    if fav:
        lines.append(
            f"  {fav.get('horse_name')} - streck {fav.get('streck_pct', 0):.1f}%, "
            f"kusk {fav.get('driver_name')}, tränare {fav.get('trainer_name')}"
        )
    else:
        lines.append("  Kunde inte avgöra favorit (ingen streckdata).")

    stats = report.similarity_stats
    lines += ["", "Jämförelselopp:"]
    lines.append(f"  Villkor som användes: {stats.get('description', report.similarity_description)}")
    if stats.get("n", 0) == 0:
        lines.append("  Inget historiskt underlag hittades - kör `atg_favorites.fetch` för att bygga upp arkivet.")
    elif stats["display"] == "insufficient":
        lines.append(f"  Otillräckligt underlag (n={stats['n']}) - ingen procentsiffra visas.")
    else:
        warning = "  [OSÄKERT UNDERLAG]" if stats["display"] == "uncertain" else ""
        baseline_note = " (BASLINJE - inte lopptyps-matchat)" if stats.get("is_baseline") else ""
        lines.append(
            f"  Favoritens vinstfrekvens: {stats['win_rate'] * 100:.1f}% "
            f"[95% KI: {stats['ci_low'] * 100:.1f}-{stats['ci_high'] * 100:.1f}%], n={stats['n']}"
            f"{warning}{baseline_note}"
        )
        if stats.get("top3_rate") is not None:
            lines.append(f"  Topp-3-frekvens: {stats['top3_rate'] * 100:.1f}%")

    if not report.favoritfall_examples.empty:
        lines += ["", "  Exempel på tidigare favoritfall bland jämförelselopp:"]
        for _, row in report.favoritfall_examples.iterrows():
            lines.append(
                f"    {row.get('game_date')} {row.get('track_name')}: "
                f"{row.get('favorite_horse_name')} ({row.get('favorite_streck_pct')}%) "
                f"föll mot {row.get('winner_horse_name')}"
            )

    return "\n".join(lines)


def format_widening_summary(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "Inga avdelningar hittades för den dagen."

    lines = [f"Urvalssteg för {len(summary)} avdelning(ar):"]
    for step, group in summary.groupby("step_label", sort=False):
        lines.append(f"  {step}: {len(group)} avdelning(ar)")
    lines.append("")
    lines.append("Per avdelning:")
    for _, row in summary.sort_values(["game_type", "avd"]).iterrows():
        lines.append(f"  {row['game_type']} avd {row['avd']} ({row['track_name']}): {row['step_label']}, n={row['n']}")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--avd",
        type=int,
        default=None,
        help="Avdelningsnummer (1-baserat) inom omgången. Utelämnas: visa breddningssammanfattning för hela dagen.",
    )
    parser.add_argument("--game-type", choices=list(GAME_TYPES), default=None)
    parser.add_argument("--track", default=None, help="Filtrera på bana om flera omgångar matchar.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    if args.avd is None:
        summary = summarize_widening_for_date(args.raw_dir, date_str=args.date, game_type=args.game_type)
        print(format_widening_summary(summary))
        return

    report = build_leg_report(
        args.raw_dir,
        date_str=args.date,
        avd=args.avd,
        game_type=args.game_type,
        track_name=args.track,
    )
    print(format_report(report))


if __name__ == "__main__":
    main()
