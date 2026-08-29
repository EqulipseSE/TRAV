"""Thin client around ATG's public racinginfo JSON API.

Two endpoints are used:

* ``GET /calendar/day/{date}`` - the day's tracks and, per game type
  (``V75``/``V85``/``V86``/...), the list of games scheduled/played that day
  together with their status (``upcoming``, ``bettable``, ``ongoing`` or
  ``results``).
* ``GET /games/{game_id}`` - full detail for one game: pools, races and, for
  every race, the full startlist (horse, driver, shoes, sulky, betDistribution
  a.k.a. "streckprocent", odds, and - once run - the result).

Neither endpoint requires authentication. Because this is an undocumented
API meant for atg.se's own frontend, the client is deliberately polite:
requests are throttled and retried with backoff on transient errors.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests
from requests import Response
from requests.exceptions import RequestException

from atg_favorites.config import BASE_URL, DEFAULT_HEADERS

logger = logging.getLogger(__name__)


class AtgApiError(RuntimeError):
    """Raised when the racinginfo API returns an unexpected response."""


class AtgClient:
    """Small HTTP client for the ATG racinginfo API."""

    def __init__(
        self,
        base_url: str = BASE_URL,
        session: requests.Session | None = None,
        request_delay: float = 1.0,
        timeout: float = 15.0,
        max_retries: int = 4,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self.request_delay = request_delay
        self.timeout = timeout
        self.max_retries = max_retries

    def _get(self, path: str) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response: Response = self.session.get(url, timeout=self.timeout)
            except RequestException as exc:  # network hiccup - retry
                last_exc = exc
                logger.warning("Request to %s failed (attempt %d): %s", url, attempt, exc)
            else:
                if response.status_code == 200:
                    time.sleep(self.request_delay)
                    return response.json()
                if response.status_code == 404:
                    raise AtgApiError(f"404 Not Found: {url}")
                if response.status_code in (429, 500, 502, 503, 504):
                    logger.warning(
                        "Transient error %s for %s (attempt %d)",
                        response.status_code,
                        url,
                        attempt,
                    )
                    last_exc = AtgApiError(f"{response.status_code} for {url}")
                else:
                    raise AtgApiError(f"Unexpected status {response.status_code} for {url}")
            time.sleep(min(2**attempt, 30))
        raise AtgApiError(f"Giving up on {url} after {self.max_retries} attempts") from last_exc

    def get_calendar_day(self, date: str) -> dict[str, Any]:
        """Return the calendar (tracks + games) for ``date`` (YYYY-MM-DD)."""
        return self._get(f"calendar/day/{date}")

    def get_game(self, game_id: str) -> dict[str, Any]:
        """Return full detail (pools, races, starts, results) for ``game_id``."""
        return self._get(f"games/{game_id}")
