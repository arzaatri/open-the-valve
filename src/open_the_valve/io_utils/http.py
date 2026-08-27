import logging
import time

import httpx

logger = logging.getLogger(__name__)


class RateLimitedClient:
    """An httpx-backed client enforcing a minimum interval between requests
    and exponential backoff retry on 429/5xx responses.

    Shared by every ingestion client (Steam, ITAD, IGDB, SteamCharts) so
    politeness and retry behavior stay consistent across sources.
    """

    def __init__(
        self,
        base_url: str,
        min_request_interval_seconds: float,
        max_retries: int = 5,
        backoff_base_seconds: float = 2.0,
        headers: dict[str, str] | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url, headers=headers or {}, timeout=timeout_seconds
        )
        self._min_interval = min_request_interval_seconds
        self._max_retries = max_retries
        self._backoff_base = backoff_base_seconds
        self._last_request_at: float | None = None

    def _wait_for_slot(self) -> None:
        if self._last_request_at is None:
            return
        elapsed = time.monotonic() - self._last_request_at
        remaining = self._min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def request(self, method: str, path: str, **kwargs) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            self._wait_for_slot()
            self._last_request_at = time.monotonic()
            try:
                response = self._client.request(method, path, **kwargs)
            except httpx.TransportError as exc:
                last_exc = exc
                logger.warning(
                    "transport error on %s %s (attempt %d): %s", method, path, attempt, exc
                )
                time.sleep(self._backoff_base * (2 ** (attempt - 1)))
                continue

            if response.status_code == 429 or response.status_code >= 500:
                retry_after = response.headers.get("Retry-After")
                delay = (
                    float(retry_after) if retry_after else self._backoff_base * (2 ** (attempt - 1))
                )
                logger.warning(
                    "status %d on %s %s (attempt %d), retrying in %.1fs",
                    response.status_code,
                    method,
                    path,
                    attempt,
                    delay,
                )
                time.sleep(delay)
                continue

            response.raise_for_status()
            return response

        raise RuntimeError(f"exhausted retries for {method} {path}") from last_exc

    def get(self, path: str, **kwargs) -> httpx.Response:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs) -> httpx.Response:
        return self.request("POST", path, **kwargs)

    def close(self) -> None:
        self._client.close()
