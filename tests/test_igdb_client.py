from contextlib import closing

import respx
from httpx import Response

from open_the_valve.config_models import IgdbSourceConfig
from open_the_valve.ingestion.igdb_client import IgdbClient


def _client() -> closing[IgdbClient]:
    config = IgdbSourceConfig(
        base_url="https://api.igdb.com/v4",
        oauth_url="https://id.twitch.tv/oauth2/token",
        client_id="test-client-id",
        client_secret="test-secret",
        requests_per_second=1000.0,
        max_concurrent_requests=8,
    )
    return closing(IgdbClient(config))


@respx.mock
def test_lookup_igdb_id_by_steam_appid():
    respx.post("https://id.twitch.tv/oauth2/token").mock(
        return_value=Response(200, json={"access_token": "tok", "expires_in": 3600})
    )
    route = respx.post("https://api.igdb.com/v4/external_games").mock(
        return_value=Response(200, json=[{"game": 42}])
    )
    with _client() as client:
        assert client.lookup_igdb_id_by_steam_appid(570) == 42
    assert b"external_game_source" in route.calls.last.request.content


@respx.mock
def test_lookup_igdb_id_by_steam_appid_not_found():
    respx.post("https://id.twitch.tv/oauth2/token").mock(
        return_value=Response(200, json={"access_token": "tok", "expires_in": 3600})
    )
    respx.post("https://api.igdb.com/v4/external_games").mock(return_value=Response(200, json=[]))
    with _client() as client:
        assert client.lookup_igdb_id_by_steam_appid(999999999) is None


@respx.mock
def test_get_game_metadata():
    respx.post("https://id.twitch.tv/oauth2/token").mock(
        return_value=Response(200, json={"access_token": "tok", "expires_in": 3600})
    )
    metadata = {"name": "Test Game", "genres": [{"name": "Shooter"}], "aggregated_rating": 88.5}
    respx.post("https://api.igdb.com/v4/games").mock(return_value=Response(200, json=[metadata]))
    with _client() as client:
        assert client.get_game_metadata(42) == metadata
