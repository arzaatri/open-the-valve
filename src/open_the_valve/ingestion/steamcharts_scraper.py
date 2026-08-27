import logging
from collections import defaultdict
from datetime import UTC, date, datetime

from open_the_valve.config_models import SteamChartsSourceConfig
from open_the_valve.db.models import PlayerCountGranularity
from open_the_valve.io_utils.http import RateLimitedClient

logger = logging.getLogger(__name__)

_MONTHLY_GAP_THRESHOLD_DAYS = 5


class DailyPlayerCountPoint:
    def __init__(
        self, observed_date: date, player_count: int, granularity: PlayerCountGranularity
    ) -> None:
        self.observed_date = observed_date
        self.player_count = player_count
        self.granularity = granularity


class SteamChartsScraper:
    """Politely scrapes SteamCharts' per-app `chart-data.json` endpoint as a
    one-time historical bootstrap for player counts.

    SteamCharts retains hourly resolution for ~1 month, daily for ~3 months,
    and monthly peaks beyond that. This scraper collapses everything to a
    single daily-peak series, tagging older points as `monthly` granularity
    since no finer-grained history exists for them. Going-forward data comes
    from polling the Steam API directly, not from this scraper.
    """

    def __init__(self, config: SteamChartsSourceConfig) -> None:
        self._client = RateLimitedClient(
            base_url=config.base_url,
            min_request_interval_seconds=config.min_request_interval_seconds,
            max_retries=config.max_retries,
            backoff_base_seconds=config.backoff_base_seconds,
            headers={"User-Agent": config.user_agent},
        )

    def fetch_daily_player_counts(self, steam_appid: int) -> list[DailyPlayerCountPoint]:
        response = self._client.get(f"/app/{steam_appid}/chart-data.json")
        raw_points: list[list[int]] = response.json()
        if not raw_points:
            logger.info("no steamcharts data for appid=%d", steam_appid)
            return []

        by_date: dict[date, int] = defaultdict(int)
        monthly_dates: set[date] = set()

        for i, (ts_ms, player_count) in enumerate(raw_points):
            observed_date = datetime.fromtimestamp(ts_ms / 1000, tz=UTC).date()
            by_date[observed_date] = max(by_date[observed_date], player_count)

            if len(raw_points) > 1:
                neighbor_ts_ms = raw_points[i - 1][0] if i > 0 else raw_points[i + 1][0]
                gap_days = abs(ts_ms - neighbor_ts_ms) / (1000 * 60 * 60 * 24)
                if gap_days > _MONTHLY_GAP_THRESHOLD_DAYS:
                    monthly_dates.add(observed_date)

        return [
            DailyPlayerCountPoint(
                observed_date=d,
                player_count=count,
                granularity=(
                    PlayerCountGranularity.MONTHLY
                    if d in monthly_dates
                    else PlayerCountGranularity.DAILY
                ),
            )
            for d, count in sorted(by_date.items())
        ]

    def close(self) -> None:
        self._client.close()
