import logging
import time
from typing import Any

from open_the_valve.config_models import IgdbSourceConfig
from open_the_valve.io_utils.http import RateLimitedClient

logger = logging.getLogger(__name__)

_STEAM_EXTERNAL_SOURCE = 1


class IgdbClient:
    """Client for the IGDB API (Twitch OAuth2 client-credentials flow).

    Rate limit per IGDB docs: 4 req/s, 8 concurrent -- the shared
    RateLimitedClient enforces the request spacing; concurrency is bounded
    naturally since this client is used synchronously.
    """

    def __init__(self, config: IgdbSourceConfig) -> None:
        self._config = config
        self._token: str | None = None
        self._token_expires_at: float = 0.0
        self._auth_client = RateLimitedClient(
            base_url=config.oauth_url.rsplit("/oauth2", 1)[0],
            min_request_interval_seconds=1.0,
        )
        self._client = RateLimitedClient(
            base_url=config.base_url,
            min_request_interval_seconds=1.0 / config.requests_per_second,
            headers={"Client-ID": config.client_id},
        )

    def _ensure_token(self) -> str:
        if self._token and time.monotonic() < self._token_expires_at:
            return self._token

        response = self._auth_client.post(
            "/oauth2/token",
            params={
                "client_id": self._config.client_id,
                "client_secret": self._config.client_secret,
                "grant_type": "client_credentials",
            },
        )
        payload = response.json()
        self._token = payload["access_token"]
        self._token_expires_at = time.monotonic() + payload["expires_in"] - 60
        return self._token

    def _query(self, endpoint: str, apicalypse_body: str) -> list[dict[str, Any]]:
        token = self._ensure_token()
        response = self._client.post(
            f"/{endpoint}",
            content=apicalypse_body,
            headers={"Authorization": f"Bearer {token}"},
        )
        return response.json()

    def lookup_igdb_id_by_steam_appid(self, steam_appid: int) -> int | None:
        results = self._query(
            "external_games",
            f'fields game; where uid = "{steam_appid}" '
            f"& external_game_source = {_STEAM_EXTERNAL_SOURCE}; limit 1;",
        )
        if not results:
            logger.info("no igdb match for steam_appid=%d", steam_appid)
            return None
        return results[0]["game"]

    def get_game_metadata(self, igdb_id: int) -> dict[str, Any] | None:
        results = self._query(
            "games",
            (
                "fields name, genres.name, release_dates.date, platforms.name, "
                "aggregated_rating, involved_companies.company.name; "
                f"where id = {igdb_id};"
            ),
        )
        return results[0] if results else None

    def close(self) -> None:
        self._auth_client.close()
        self._client.close()
