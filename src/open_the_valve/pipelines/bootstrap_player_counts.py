import logging
from contextlib import closing

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from open_the_valve.config_models import AppConfig, SeedGamesFile
from open_the_valve.db import repo
from open_the_valve.db.models import PlayerCountSource
from open_the_valve.db.session import make_engine, session_scope
from open_the_valve.ingestion.steamcharts_scraper import SteamChartsScraper
from open_the_valve.logging_utils import setup_logging

load_dotenv()
logger = logging.getLogger(__name__)

_WATERMARK_SOURCE = "steamcharts_bootstrap"
_DONE_CURSOR = "done"


def run(config: AppConfig) -> None:
    seed_games = SeedGamesFile.model_validate(OmegaConf.load(config.seed_games_file))
    engine = make_engine(config.db)

    with closing(SteamChartsScraper(config.sources.steamcharts)) as scraper:
        for seed_game in seed_games.games:
            with session_scope(engine) as session:
                game_id = repo.upsert_game(session, seed_game.steam_appid, seed_game.name)
                if repo.get_watermark(session, _WATERMARK_SOURCE, game_id) == _DONE_CURSOR:
                    logger.info("skipping %s, already bootstrapped", seed_game.name)
                    continue

            points = scraper.fetch_daily_player_counts(seed_game.steam_appid)
            with session_scope(engine) as session:
                for point in points:
                    repo.upsert_player_count_row(
                        session,
                        game_id=game_id,
                        observed_date_=point.observed_date,
                        player_count=point.player_count,
                        source=PlayerCountSource.STEAMCHARTS_BOOTSTRAP,
                        granularity=point.granularity,
                    )
                repo.set_watermark(session, _WATERMARK_SOURCE, game_id, _DONE_CURSOR)
            logger.info("bootstrapped %d player-count points for %s", len(points), seed_game.name)


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging()
    config = AppConfig.model_validate(OmegaConf.to_container(cfg, resolve=True))
    run(config)


if __name__ == "__main__":
    main()
