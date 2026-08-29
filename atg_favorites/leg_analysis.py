"""Analysera en enskild avdelning (lopp) - avgjord, pågående eller kommande.

Given a date and an "avdelning" (leg number within a V75/V85/V86 round),
this module:

1. Fetches that leg live from ATG (works whether the round is upcoming,
   bettable, ongoing or already finished).
2. Extracts every starter with its current streckprocent, spår, tillägg,
   skor, sulky, odds and background stats (driver/trainer win%, hästens
   livstidsstatistik och rekord).
3. Finds historically *similar* races in the local archive
   (``data/processed/races.csv``, built by ``flatten.py``) - same bana,
   distans, startsätt and fältstorlek when possible, progressively
   relaxing the criteria ("eller annat relevant") if there isn't enough
   data - and summarises how favorites have fared in those similar races.

Run as a script::

    python -m atg_favorites.leg_analysis --date 2026-08-29 --avd 5
    python -m atg_favorites.leg_analysis --date 2026-08-29 --avd 3 --game-type V86
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from atg_favorites.api_client import AtgApiError, AtgClient
from atg_favorites.config import GAME_TYPES, RACES_CSV
from atg_favorites.extraction import start_summary

logger = logging.getLogger(__name__)

#: Similarity tiers tried in order, from strictest to most relaxed. Each is
#: (description, kwargs-overrides-for-_apply_filters).
_SIMILARITY_TIERS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("samma bana, distans, startsätt och ungefär samma fältstorlek", {}),
    ("samma distans, startsätt och ungefär samma fältstorlek (oavsett bana)", {"match_track": False}),
    ("samma startsätt och ungefär samma distans (oavsett bana/fältstorlek)", {"match_track": False, "match_field_size": False}),
    ("alla tidigare inlästa lopp (inget kriterium gav nog underlag)", {"match_track": False, "match_field_size": False, "match_start_method": False}),
)

#: Minimum number of similar races before we stop relaxing the criteria.
MIN_SAMPLE_SIZE = 8


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
class LegReport:
    target: LegTarget
    starters: pd.DataFrame
    favorite: dict[str, Any]
    similar_races: pd.DataFrame
    similarity_description: str
    similarity_stats: dict[str, Any] = field(default_factory=dict)
    favoritfall_examples: pd.DataFrame = field(default_factory=pd.DataFrame)


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


def _apply_filters(
    races: pd.DataFrame,
    target_row: dict[str, Any],
    distance_tolerance: int,
    field_size_tolerance: int,
    match_track: bool = True,
    match_field_size: bool = True,
    match_start_method: bool = True,
) -> pd.DataFrame:
    mask = pd.Series(True, index=races.index)
    if match_track and target_row.get("track_name"):
        mask &= races["track_name"] == target_row["track_name"]
    if match_start_method and target_row.get("start_method"):
        mask &= races["start_method"] == target_row["start_method"]
    if target_row.get("race_distance_m") is not None:
        mask &= races["race_distance_m"].between(
            target_row["race_distance_m"] - distance_tolerance,
            target_row["race_distance_m"] + distance_tolerance,
        )
    if match_field_size and target_row.get("num_starters") is not None:
        mask &= races["num_starters"].between(
            target_row["num_starters"] - field_size_tolerance,
            target_row["num_starters"] + field_size_tolerance,
        )
    return races.loc[mask]


def find_similar_races(
    races: pd.DataFrame,
    target_row: dict[str, Any],
    distance_tolerance: int = 150,
    field_size_tolerance: int = 2,
    min_sample_size: int = MIN_SAMPLE_SIZE,
) -> tuple[pd.DataFrame, str]:
    """Find historically similar races, relaxing criteria until there's enough data.

    Returns ``(similar_races, description)`` where ``description`` explains
    which criteria were actually used (Swedish, human-readable).
    """
    if races.empty:
        return races, "inget historiskt underlag inläst än"

    last_result = races.iloc[0:0]
    last_description = _SIMILARITY_TIERS[-1][0]
    for description, overrides in _SIMILARITY_TIERS:
        filtered = _apply_filters(
            races,
            target_row,
            distance_tolerance=distance_tolerance,
            field_size_tolerance=field_size_tolerance,
            **overrides,
        )
        last_result, last_description = filtered, description
        if len(filtered) >= min_sample_size:
            return filtered, description
    return last_result, last_description


def similarity_stats(similar_races: pd.DataFrame, favorite_streck_pct: float | None) -> dict[str, Any]:
    """Headline stats for the similar-races subset, plus an edge estimate."""
    if similar_races.empty:
        return {"races": 0}

    stats = {
        "races": int(len(similar_races)),
        "avg_favorite_streck_pct": float(similar_races["favorite_streck_pct"].mean()),
        "favorite_win_rate": float(similar_races["favorite_won"].mean()),
        "favorite_top3_rate": float(similar_races["favorite_top3"].mean()),
        "favoritfall_rate": float(1 - similar_races["favorite_won"].mean()),
    }

    if favorite_streck_pct is not None:
        # Restrict further to races whose favorite had a similar streck-%,
        # since a 25% favorite and a 60% favorite aren't really comparable.
        close = similar_races[(similar_races["favorite_streck_pct"] - favorite_streck_pct).abs() <= 10]
        if len(close) >= 5:
            stats["close_streck_band_races"] = int(len(close))
            stats["close_streck_band_win_rate"] = float(close["favorite_won"].mean())
            stats["close_streck_band_edge_vs_streck"] = float(
                close["favorite_won"].mean() - favorite_streck_pct / 100.0
            )

    return stats


def build_leg_report(
    races_csv: Path = RACES_CSV,
    *,
    date_str: str,
    avd: int,
    game_type: str | None = None,
    track_name: str | None = None,
    client: AtgClient | None = None,
    distance_tolerance: int = 150,
    field_size_tolerance: int = 2,
    min_sample_size: int = MIN_SAMPLE_SIZE,
) -> LegReport:
    client = client or AtgClient()
    target = resolve_leg(client, date_str, avd, game_type=game_type, track_name=track_name)

    starters = leg_starters(target)
    favorite = current_favorite(starters) or {}

    track = target.race.get("track") or {}
    target_row = {
        "track_name": track.get("name"),
        "start_method": target.race.get("startMethod"),
        "race_distance_m": target.race.get("distance"),
        "num_starters": len(starters),
    }

    races = pd.read_csv(races_csv) if Path(races_csv).exists() else pd.DataFrame()
    if not races.empty:
        races["favorite_won"] = races["favorite_won"].astype(bool)
        races["favorite_top3"] = races["favorite_top3"].astype(bool)
        # Never "match" a race against itself if it happens to already be
        # in the historical archive (e.g. re-analysing a finished round).
        races = races[races["game_id"] != target.game_id]

    similar, description = find_similar_races(
        races,
        target_row,
        distance_tolerance=distance_tolerance,
        field_size_tolerance=field_size_tolerance,
        min_sample_size=min_sample_size,
    )
    stats = similarity_stats(similar, favorite.get("streck_pct"))

    favoritfall_examples = pd.DataFrame()
    if not similar.empty:
        favoritfall_examples = similar.loc[
            ~similar["favorite_won"],
            [c for c in ("game_date", "track_name", "favorite_horse_name", "favorite_streck_pct", "winner_horse_name") if c in similar.columns],
        ].sort_values("favorite_streck_pct", ascending=False).head(10)

    return LegReport(
        target=target,
        starters=starters,
        favorite=favorite,
        similar_races=similar,
        similarity_description=description,
        similarity_stats=stats,
        favoritfall_examples=favoritfall_examples,
    )


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

    lines += ["", f"Liknande historiska lopp ({report.similarity_description}):"]
    stats = report.similarity_stats
    if stats.get("races"):
        lines.append(
            f"  {stats['races']} liknande lopp: favoriten vann {stats['favorite_win_rate'] * 100:.1f}%, "
            f"topp-3 {stats['favorite_top3_rate'] * 100:.1f}%, favoritfall {stats['favoritfall_rate'] * 100:.1f}%"
        )
        if "close_streck_band_win_rate" in stats:
            lines.append(
                f"  Bland de {stats['close_streck_band_races']} lopp där favoriten hade liknande streck "
                f"(±10 %-enheter): vann {stats['close_streck_band_win_rate'] * 100:.1f}% "
                f"(edge vs. streck: {stats['close_streck_band_edge_vs_streck'] * 100:+.1f} %-enheter)"
            )
    else:
        lines.append("  Inget historiskt underlag hittades - kör `atg_favorites.fetch` för att bygga upp arkivet.")

    if not report.favoritfall_examples.empty:
        lines += ["", "  Exempel på tidigare favoritfall i liknande lopp:"]
        for _, row in report.favoritfall_examples.iterrows():
            lines.append(
                f"    {row.get('game_date')} {row.get('track_name')}: "
                f"{row.get('favorite_horse_name')} ({row.get('favorite_streck_pct')}%) "
                f"föll mot {row.get('winner_horse_name')}"
            )

    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--avd", type=int, required=True, help="Avdelningsnummer (1-baserat) inom omgången.")
    parser.add_argument("--game-type", choices=list(GAME_TYPES), default=None)
    parser.add_argument("--track", default=None, help="Filtrera på bana om flera omgångar matchar.")
    parser.add_argument("--races-csv", type=Path, default=RACES_CSV)
    parser.add_argument("--distance-tolerance", type=int, default=150)
    parser.add_argument("--field-size-tolerance", type=int, default=2)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    report = build_leg_report(
        args.races_csv,
        date_str=args.date,
        avd=args.avd,
        game_type=args.game_type,
        track_name=args.track,
        distance_tolerance=args.distance_tolerance,
        field_size_tolerance=args.field_size_tolerance,
    )
    print(format_report(report))


if __name__ == "__main__":
    main()
