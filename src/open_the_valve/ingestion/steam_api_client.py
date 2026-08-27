import logging

from open_the_valve.config_models import SteamSourceConfig
from open_the_valve.io_utils.http import RateLimitedClient

logger = logging.getLogger(__name__)


class SteamApiClient:
    """Client for the public Steam Web API.

    Only exposes the current concurrent player count per app -- there is no
    official historical endpoint, which is why this client is used for
    going-forward polling rather than backfill.
    """

    def __init__(self, config: SteamSourceConfig) -> None:
        self._config = config
        self._client = RateLimitedClient(
            base_url=config.base_url,
            min_request_interval_seconds=1.0 / config.requests_per_second,
        )

    def get_current_player_count(self, steam_appid: int) -> int | None:
        response = self._client.get(
            "/ISteamUserStats/GetNumberOfCurrentPlayers/v1/",
            params={"appid": steam_appid},
        )
        payload = response.json().get("response", {})
        if payload.get("result") != 1:
            logger.warning("steam api returned no player count for appid=%d", steam_appid)
            return None
        return payload.get("player_count")

    def close(self) -> None:
        self._client.close()
