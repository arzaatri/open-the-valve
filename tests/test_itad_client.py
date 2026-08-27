from contextlib import closing
from datetime import UTC, datetime

import respx
from httpx import Response

from open_the_valve.config_models import ItadSourceConfig
from open_the_valve.ingestion.itad_client import ItadClient


def _client() -> closing[ItadClient]:
    config = ItadSourceConfig(
        base_url="https://api.isthereanydeal.com",
        api_key="test-key",
        requests_per_second=1000.0,
        history_since_default_days=365,
    )
    return closing(ItadClient(config))


@respx.mock
def test_lookup_by_steam_appid_found():
    respx.get("https://api.isthereanydeal.com/games/lookup/v1").mock(
        return_value=Response(200, json={"found": True, "game": {"id": "abc123"}})
    )
    with _client() as client:
        assert client.lookup_by_steam_appid(570) == "abc123"


@respx.mock
def test_lookup_by_steam_appid_not_found():
    respx.get("https://api.isthereanydeal.com/games/lookup/v1").mock(
        return_value=Response(200, json={"found": False})
    )
    with _client() as client:
        assert client.lookup_by_steam_appid(999999999) is None


@respx.mock
def test_get_price_history_passes_through():
    history = [
        {
            "shop": {"name": "steam"},
            "timestamp": "2026-01-01T00:00:00+00:00",
            "deal": {"price": {"amount": 9.99}, "cut": 50},
        }
    ]
    respx.get("https://api.isthereanydeal.com/games/history/v2").mock(
        return_value=Response(200, json=history)
    )
    with _client() as client:
        assert client.get_price_history("abc123") == history


@respx.mock
def test_get_price_history_since_uses_rfc3339_z_format():
    route = respx.get("https://api.isthereanydeal.com/games/history/v2").mock(
        return_value=Response(200, json=[])
    )
    with _client() as client:
        client.get_price_history("abc123", since=datetime(2026, 1, 1, 12, 30, 0, tzinfo=UTC))
    since_param = route.calls.last.request.url.params["since"]
    assert since_param == "2026-01-01T12:30:00Z"
