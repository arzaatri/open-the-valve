import logging
from contextlib import closing
from datetime import date

from dotenv import load_dotenv

from open_the_valve.config_models import AppConfig
from open_the_valve.db import repo
from open_the_valve.db.models import PlayerCountGranularity, PlayerCountSource
from open_the_valve.db.session import make_engine, session_scope
from open_the_valve.ingestion.steam_api_client import SteamApiClient
from open_the_valve.io_utils.hydra_entrypoint import hydra_entrypoint

load_dotenv()
logger = logging.getLogger(__name__)


def run(config: AppConfig) -> None:
    engine = make_engine(config.db)
    today = date.today()

    with session_scope(engine) as session:
        games = repo.list_games(session)

    with closing(SteamApiClient(config.sources.steam)) as steam_api:
        for game_id, steam_appid in games:
            player_count = steam_api.get_current_player_count(steam_appid)
            if player_count is None:
                continue

            with session_scope(engine) as session:
                existing = repo.get_player_count(
                    session, game_id, today, PlayerCountSource.STEAM_API_POLL
                )
                daily_max = max(player_count, existing) if existing is not None else player_count
                repo.upsert_player_count_row(
                    session,
                    game_id=game_id,
                    observed_date_=today,
                    player_count=daily_max,
                    source=PlayerCountSource.STEAM_API_POLL,
                    granularity=PlayerCountGranularity.DAILY,
                )
            logger.info("polled appid=%d: %d (daily max=%d)", steam_appid, player_count, daily_max)


main = hydra_entrypoint(run)

if __name__ == "__main__":
    main()
