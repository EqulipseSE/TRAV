"""Fetch avgjorda (completed) V75/V85/V86-omgångar and store the raw JSON.

Usage (see README for more examples)::

    python -m atg_favorites.fetch --days-back 14
    python -m atg_favorites.fetch --start-date 2026-01-01 --end-date 2026-01-31
    python -m atg_favorites.fetch --game-types V75 V86 --out-dir data/raw

For every day in the requested range the day's calendar is fetched once,
then every V75/V85/V86 game whose status is ``results`` (i.e. the round has
been fully run/"avgjord") is fetched in full and written to
``data/raw/<game_id>.json``. Already-downloaded games are skipped unless
``--overwrite`` is given, which makes repeated runs cheap and safe to
schedule regularly.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

from atg_favorites.api_client import AtgApiError, AtgClient
from atg_favorites.config import FINISHED_STATUSES, GAME_TYPES, RAW_DIR

logger = logging.getLogger(__name__)


def _date_range(start: date, end: date) -> list[date]:
    if end < start:
        start, end = end, start
    days = (end - start).days
    return [start + timedelta(days=n) for n in range(days + 1)]


def iter_finished_game_ids(
    client: AtgClient,
    days: list[date],
    game_types: tuple[str, ...] = GAME_TYPES,
) -> list[str]:
    """Return the ids of every finished game of the requested types across ``days``."""
    game_ids: list[str] = []
    for day in days:
        day_str = day.isoformat()
        try:
            calendar = client.get_calendar_day(day_str)
        except AtgApiError as exc:
            logger.warning("Could not fetch calendar for %s: %s", day_str, exc)
            continue

        games_by_type = calendar.get("games", {})
        for game_type in game_types:
            for game in games_by_type.get(game_type, []):
                if game.get("status") in FINISHED_STATUSES:
                    game_ids.append(game["id"])
    return game_ids


def fetch_and_save_game(client: AtgClient, game_id: str, out_dir: Path, overwrite: bool) -> Path | None:
    """Fetch one game and save it as ``<out_dir>/<game_id>.json``.

    Returns the path written to, or ``None`` if it was skipped.
    """
    out_path = out_dir / f"{game_id}.json"
    if out_path.exists() and not overwrite:
        logger.info("Skipping %s (already downloaded)", game_id)
        return None

    logger.info("Fetching %s", game_id)
    game = client.get_game(game_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(game, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def run(
    start_date: date,
    end_date: date,
    game_types: tuple[str, ...] = GAME_TYPES,
    out_dir: Path = RAW_DIR,
    overwrite: bool = False,
    request_delay: float = 1.0,
) -> list[Path]:
    client = AtgClient(request_delay=request_delay)
    days = _date_range(start_date, end_date)
    logger.info("Scanning %d day(s) of calendars for %s", len(days), ", ".join(game_types))

    game_ids = iter_finished_game_ids(client, days, game_types)
    logger.info("Found %d finished game(s)", len(game_ids))

    saved: list[Path] = []
    for game_id in game_ids:
        try:
            path = fetch_and_save_game(client, game_id, out_dir, overwrite)
        except AtgApiError as exc:
            logger.warning("Failed to fetch %s: %s", game_id, exc)
            continue
        if path is not None:
            saved.append(path)

    logger.info("Saved %d new raw game file(s) to %s", len(saved), out_dir)
    return saved


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument(
        "--days-back",
        type=int,
        default=7,
        help="Fetch calendars for the last N days including today (default: 7).",
    )
    date_group.add_argument(
        "--start-date",
        type=str,
        help="First date to scan, YYYY-MM-DD. Requires --end-date.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="Last date to scan, YYYY-MM-DD (default: today). Used together with --start-date.",
    )
    parser.add_argument(
        "--game-types",
        nargs="+",
        default=list(GAME_TYPES),
        choices=list(GAME_TYPES),
        help="Which game types to fetch (default: V75 V85 V86).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=RAW_DIR,
        help=f"Directory to store raw JSON in (default: {RAW_DIR}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download games even if a raw JSON file already exists for them.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to sleep between HTTP requests (default: 1.0, be polite to ATG's API).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    today = date.today()
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else today
    else:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else today
        start_date = end_date - timedelta(days=args.days_back - 1)

    run(
        start_date=start_date,
        end_date=end_date,
        game_types=tuple(args.game_types),
        out_dir=args.out_dir,
        overwrite=args.overwrite,
        request_delay=args.delay,
    )


if __name__ == "__main__":
    main()
