"""Central configuration: paths, endpoints and constants."""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ATG's public "racinginfo" JSON API. Undocumented but stable in practice;
# used by atg.se itself to render race calendars and game/race pages.
BASE_URL = "https://www.atg.se/services/racinginfo/v1/api"

#: The "big pool" games this project cares about.
GAME_TYPES = ("V75", "V85", "V86")

#: Statuses on a calendar entry / game that mean the round has been fully run.
FINISHED_STATUSES = ("results",)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; atg-favorites-research/0.1; "
        "+https://github.com/) requests"
    ),
    "Accept": "application/json",
}

#: Default CSV output filenames.
RACES_CSV = PROCESSED_DIR / "races.csv"
FAVORITE_BUCKET_CSV = PROCESSED_DIR / "favorite_bucket_analysis.csv"
FAVORITE_SURPRISES_CSV = PROCESSED_DIR / "favorite_surprises.csv"

#: Default streck-% bucket edges used by the favorite analysis (0-100).
DEFAULT_BUCKET_EDGES = (0, 15, 20, 25, 30, 35, 40, 50, 100)
