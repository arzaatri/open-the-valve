import logging
from datetime import datetime
from typing import Any

from open_the_valve.config_models import ItadSourceConfig
from open_the_valve.io_utils.http import RateLimitedClient

logger = logging.getLogger(__name__)


class ItadClient:
    """Client for the IsThereAnyDeal API: game lookup, current prices, and
    price/discount history.
    """

    def __init__(self, config: ItadSourceConfig) -> None:
        self._config = config
        self._client = RateLimitedClient(
            base_url=config.base_url,
            min_request_interval_seconds=1.0 / config.requests_per_second,
            headers={"ITAD-API-Key": config.api_key},
        )

    def lookup_by_steam_appid(self, steam_appid: int) -> str | None:
        response = self._client.get("/games/lookup/v1", params={"appid": steam_appid})
        payload = response.json()
        if not payload.get("found"):
            logger.info("no itad match for steam_appid=%d", steam_appid)
            return None
        return payload["game"]["id"]

    def get_prices(self, itad_ids: list[str]) -> list[dict[str, Any]]:
        response = self._client.post("/games/prices/v3", json=itad_ids)
        return response.json()

    def get_price_history(
        self, itad_id: str, since: datetime | None = None
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"id": itad_id}
        if since is not None:
            params["since"] = since.strftime("%Y-%m-%dT%H:%M:%SZ")
        response = self._client.get("/games/history/v2", params=params)
        return response.json()

    def close(self) -> None:
        self._client.close()
