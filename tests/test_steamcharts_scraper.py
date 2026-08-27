from contextlib import closing
from datetime import UTC, date

import respx
from httpx import Response

from open_the_valve.config_models import SteamChartsSourceConfig
from open_the_valve.db.models import PlayerCountGranularity
from open_the_valve.ingestion.steamcharts_scraper import SteamChartsScraper


def _scraper() -> closing[SteamChartsScraper]:
    config = SteamChartsSourceConfig(
        base_url="https://steamcharts.com",
        user_agent="test-bot/0.1",
        min_request_interval_seconds=0.0,
        max_retries=3,
        backoff_base_seconds=0.01,
    )
    return closing(SteamChartsScraper(config))


def _ts_ms(y: int, m: int, d: int, hour: int = 12) -> int:
    from datetime import datetime

    return int(datetime(y, m, d, hour, tzinfo=UTC).timestamp() * 1000)


@respx.mock
def test_daily_points_collapse_hourly_to_daily_peak():
    raw = [
        [_ts_ms(2026, 1, 1, 0), 100],
        [_ts_ms(2026, 1, 1, 12), 300],
        [_ts_ms(2026, 1, 1, 23), 150],
        [_ts_ms(2026, 1, 2, 0), 200],
    ]
    respx.get("https://steamcharts.com/app/570/chart-data.json").mock(
        return_value=Response(200, json=raw)
    )
    with _scraper() as scraper:
        points = scraper.fetch_daily_player_counts(570)

    assert len(points) == 2
    assert points[0].observed_date == date(2026, 1, 1)
    assert points[0].player_count == 300
    assert points[0].granularity == PlayerCountGranularity.DAILY
    assert points[1].observed_date == date(2026, 1, 2)


@respx.mock
def test_large_gap_marked_as_monthly():
    raw = [
        [_ts_ms(2012, 7, 1), 1000],
        [_ts_ms(2012, 8, 15), 2000],
    ]
    respx.get("https://steamcharts.com/app/570/chart-data.json").mock(
        return_value=Response(200, json=raw)
    )
    with _scraper() as scraper:
        points = scraper.fetch_daily_player_counts(570)

    assert points[0].granularity == PlayerCountGranularity.MONTHLY
    assert points[1].granularity == PlayerCountGranularity.MONTHLY


@respx.mock
def test_empty_response_returns_no_points():
    respx.get("https://steamcharts.com/app/570/chart-data.json").mock(
        return_value=Response(200, json=[])
    )
    with _scraper() as scraper:
        assert scraper.fetch_daily_player_counts(570) == []
