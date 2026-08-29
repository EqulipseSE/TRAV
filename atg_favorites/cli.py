"""Unified command line entry point: fetch, flatten and analyse in one place.

    python -m atg_favorites.cli fetch --days-back 14
    python -m atg_favorites.cli flatten
    python -m atg_favorites.cli analyze --by-game-type
    python -m atg_favorites.cli pipeline --days-back 14 --by-game-type

``pipeline`` simply runs fetch -> flatten -> analyze in sequence with shared
defaults, which is the easiest way to go from nothing to a fresh analysis.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

from atg_favorites import analysis, fetch, flatten
from atg_favorites.config import GAME_TYPES, RACES_CSV, RAW_DIR

logger = logging.getLogger(__name__)


def _add_common_fetch_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--days-back", type=int, default=7)
    parser.add_argument("--game-types", nargs="+", default=list(GAME_TYPES), choices=list(GAME_TYPES))
    parser.add_argument("--out-dir", type=str, default=str(RAW_DIR))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--delay", type=float, default=1.0)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="atg-favorites", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch_p = sub.add_parser("fetch", help="Download raw JSON for completed V75/V85/V86 rounds.")
    _add_common_fetch_args(fetch_p)

    flatten_p = sub.add_parser("flatten", help="Flatten raw JSON into races.csv.")
    flatten_p.add_argument("--raw-dir", type=str, default=str(RAW_DIR))
    flatten_p.add_argument("--out", type=str, default=str(RACES_CSV))

    analyze_p = sub.add_parser("analyze", help="Run the streck-adjusted favorite-loss analysis.")
    analyze_p.add_argument("--races-csv", type=str, default=str(RACES_CSV))
    analyze_p.add_argument("--by-game-type", action="store_true")
    analyze_p.add_argument("--min-surprise-streck", type=float, default=35.0)

    pipeline_p = sub.add_parser("pipeline", help="Run fetch, flatten and analyze in sequence.")
    _add_common_fetch_args(pipeline_p)
    pipeline_p.add_argument("--by-game-type", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "fetch":
        today = date.today()
        fetch.run(
            start_date=today - timedelta(days=args.days_back - 1),
            end_date=today,
            game_types=tuple(args.game_types),
            out_dir=Path(args.out_dir),
            overwrite=args.overwrite,
            request_delay=args.delay,
        )
    elif args.command == "flatten":
        df = flatten.build_dataframe(Path(args.raw_dir))
        if df.empty:
            logger.warning("No finished races found - nothing written.")
        else:
            flatten.save_races_csv(df, Path(args.out))
            logger.info("Wrote %d rows to %s", len(df), args.out)
    elif args.command == "analyze":
        df = analysis.load_races(Path(args.races_csv))
        if df.empty:
            logger.warning("No rows in %s - nothing to analyse.", args.races_csv)
        else:
            logger.info("Overall: %s", analysis.summarize(df))
            group_by = ["game_type"] if args.by_game_type else None
            bucket_df = analysis.favorite_bucket_analysis(df, group_by=group_by)
            print(bucket_df.to_string(index=False))
            surprises = analysis.favorite_surprises(df, min_streck_pct=args.min_surprise_streck)
            logger.info("%d favoritfall found (streck >= %.0f%%)", len(surprises), args.min_surprise_streck)
    elif args.command == "pipeline":
        today = date.today()
        fetch.run(
            start_date=today - timedelta(days=args.days_back - 1),
            end_date=today,
            game_types=tuple(args.game_types),
            out_dir=Path(args.out_dir),
            overwrite=args.overwrite,
            request_delay=args.delay,
        )
        df = flatten.build_dataframe(Path(args.out_dir))
        if df.empty:
            logger.warning("No finished races found - nothing to analyse.")
            return
        flatten.save_races_csv(df)
        logger.info("Overall: %s", analysis.summarize(df))
        group_by = ["game_type"] if args.by_game_type else None
        bucket_df = analysis.favorite_bucket_analysis(df, group_by=group_by)
        print(bucket_df.to_string(index=False))
    else:  # pragma: no cover - argparse enforces valid subcommands
        parser.print_help(sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
