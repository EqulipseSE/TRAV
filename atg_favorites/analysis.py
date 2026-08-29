"""Streckjusterad favoritfall-analys.

"Favoritfall" = the case where the pool favorite (highest streckprocent)
does *not* win its race. This module buckets races by the favorite's
streckprocent (the crowd's implied win probability - what everyone
"struck"/staked their money on) and compares the *actual* win/top-3 rate in
each bucket against that implied probability. This tells you whether, once
you adjust ("justerar") for how strongly the crowd backed the favorite,
favorites over- or underperform - the core question behind any "ska jag
gardera favoriten"-strategy for V75/V85/V86.

Run as a script::

    python -m atg_favorites.analysis --races-csv data/processed/races.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from atg_favorites.config import (
    DEFAULT_BUCKET_EDGES,
    FAVORITE_BUCKET_CSV,
    FAVORITE_SURPRISES_CSV,
    RACES_CSV,
)

logger = logging.getLogger(__name__)


def load_races(races_csv: Path = RACES_CSV) -> pd.DataFrame:
    df = pd.read_csv(races_csv)
    for col in ("favorite_won", "favorite_top3", "winner_is_favorite"):
        if col in df.columns:
            df[col] = df[col].astype(bool)
    return df


def add_streck_bucket(
    df: pd.DataFrame,
    bucket_edges: tuple[int, ...] = DEFAULT_BUCKET_EDGES,
    streck_col: str = "favorite_streck_pct",
) -> pd.DataFrame:
    """Add a ``streck_bucket`` categorical column bucketing ``streck_col``."""
    df = df.copy()
    labels = [f"{lo}-{hi}%" for lo, hi in zip(bucket_edges[:-1], bucket_edges[1:])]
    df["streck_bucket"] = pd.cut(
        df[streck_col],
        bins=bucket_edges,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    return df


def favorite_bucket_analysis(
    df: pd.DataFrame,
    bucket_edges: tuple[int, ...] = DEFAULT_BUCKET_EDGES,
    group_by: list[str] | None = None,
) -> pd.DataFrame:
    """Streckjusterad favoritfall-analys grouped by streck-bucket.

    For each streck-% bucket (optionally further split by ``group_by``,
    e.g. ``["game_type"]``) this computes:

    * ``races`` - number of races in the bucket
    * ``avg_streck_pct`` - the average favorite streckprocent in the bucket,
      used as the crowd's *implied* win probability
    * ``win_rate`` - the favorite's *actual* win rate
    * ``top3_rate`` - the favorite's actual top-3 rate
    * ``favoritfall_rate`` - ``1 - win_rate``: how often the favorite loses
    * ``edge_vs_streck`` - ``win_rate - avg_streck_pct/100``: positive means
      favorites in that bucket win *more* than their streck-% implies
      (undervalued by the crowd), negative means they win *less*
      (overvalued/"overbet")
    """
    working = add_streck_bucket(df, bucket_edges)
    group_cols = (group_by or []) + ["streck_bucket"]

    grouped = working.groupby(group_cols, observed=True).agg(
        races=("race_id", "count"),
        avg_streck_pct=("favorite_streck_pct", "mean"),
        win_rate=("favorite_won", "mean"),
        top3_rate=("favorite_top3", "mean"),
    )
    grouped["favoritfall_rate"] = 1 - grouped["win_rate"]
    grouped["implied_win_prob"] = grouped["avg_streck_pct"] / 100.0
    grouped["edge_vs_streck"] = grouped["win_rate"] - grouped["implied_win_prob"]

    grouped = grouped.reset_index().sort_values(group_cols)
    return grouped[
        group_cols
        + [
            "races",
            "avg_streck_pct",
            "implied_win_prob",
            "win_rate",
            "top3_rate",
            "favoritfall_rate",
            "edge_vs_streck",
        ]
    ]


def favorite_surprises(df: pd.DataFrame, min_streck_pct: float = 35.0) -> pd.DataFrame:
    """Races where a strong favorite (>= ``min_streck_pct``) still lost ("favoritfall")."""
    mask = (df["favorite_streck_pct"] >= min_streck_pct) & (~df["favorite_won"])
    columns = [
        "game_id",
        "game_type",
        "game_date",
        "track_name",
        "race_number",
        "favorite_horse_name",
        "favorite_streck_pct",
        "favorite_place",
        "winner_horse_name",
        "winner_streck_pct",
    ]
    existing_columns = [c for c in columns if c in df.columns]
    return df.loc[mask, existing_columns].sort_values("favorite_streck_pct", ascending=False)


def summarize(df: pd.DataFrame) -> dict[str, float]:
    """Overall (non-bucketed) headline numbers."""
    return {
        "races": int(len(df)),
        "avg_favorite_streck_pct": float(df["favorite_streck_pct"].mean()),
        "favorite_win_rate": float(df["favorite_won"].mean()),
        "favorite_top3_rate": float(df["favorite_top3"].mean()),
        "favoritfall_rate": float(1 - df["favorite_won"].mean()),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--races-csv", type=Path, default=RACES_CSV)
    parser.add_argument("--bucket-out", type=Path, default=FAVORITE_BUCKET_CSV)
    parser.add_argument("--surprises-out", type=Path, default=FAVORITE_SURPRISES_CSV)
    parser.add_argument(
        "--by-game-type",
        action="store_true",
        help="Also split the bucket analysis by game_type (V75/V85/V86).",
    )
    parser.add_argument("--min-surprise-streck", type=float, default=35.0)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    df = load_races(args.races_csv)
    if df.empty:
        logger.warning("No rows in %s - nothing to analyse.", args.races_csv)
        return

    logger.info("Overall: %s", summarize(df))

    group_by = ["game_type"] if args.by_game_type else None
    bucket_df = favorite_bucket_analysis(df, group_by=group_by)
    args.bucket_out.parent.mkdir(parents=True, exist_ok=True)
    bucket_df.to_csv(args.bucket_out, index=False)
    logger.info("Wrote streck-bucket analysis (%d rows) to %s", len(bucket_df), args.bucket_out)
    print(bucket_df.to_string(index=False))

    surprises_df = favorite_surprises(df, min_streck_pct=args.min_surprise_streck)
    args.surprises_out.parent.mkdir(parents=True, exist_ok=True)
    surprises_df.to_csv(args.surprises_out, index=False)
    logger.info("Wrote %d favoritfall (favorite-loss) rows to %s", len(surprises_df), args.surprises_out)


if __name__ == "__main__":
    main()
