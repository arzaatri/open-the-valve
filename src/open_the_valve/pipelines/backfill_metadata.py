import datetime
import logging
from contextlib import closing

from dotenv import load_dotenv
from omegaconf import OmegaConf

from open_the_valve.config_models import AppConfig, SeedGamesFile
from open_the_valve.db import repo
from open_the_valve.db.session import make_engine, session_scope
from open_the_valve.ingestion.igdb_client import IgdbClient
from open_the_valve.io_utils.hydra_entrypoint import hydra_entrypoint

load_dotenv()
logger = logging.getLogger(__name__)


def run(config: AppConfig) -> None:
    seed_games = SeedGamesFile.model_validate(OmegaConf.load(config.seed_games_file))
    engine = make_engine(config.db)

    with closing(IgdbClient(config.sources.igdb)) as igdb:
        for seed_game in seed_games.games:
            with session_scope(engine) as session:
                game_id = repo.upsert_game(session, seed_game.steam_appid, seed_game.name)

            igdb_id = igdb.lookup_igdb_id_by_steam_appid(seed_game.steam_appid)
            if igdb_id is None:
                continue

            metadata = igdb.get_game_metadata(igdb_id)
            if metadata is None:
                continue

            release_date = None
            if metadata.get("release_dates"):
                release_date = datetime.date.fromtimestamp(metadata["release_dates"][0]["date"])

            with session_scope(engine) as session:
                repo.upsert_game(session, seed_game.steam_appid, seed_game.name, igdb_id=igdb_id)
                repo.upsert_game_metadata(
                    session,
                    game_id=game_id,
                    genres=[g["name"] for g in metadata.get("genres", [])],
                    release_date=release_date,
                    platforms=[p["name"] for p in metadata.get("platforms", [])],
                    aggregated_rating=metadata.get("aggregated_rating"),
                    involved_companies=[
                        c["company"]["name"] for c in metadata.get("involved_companies", [])
                    ],
                )
            logger.info("backfilled metadata for %s (igdb_id=%d)", seed_game.name, igdb_id)


main = hydra_entrypoint(run)

if __name__ == "__main__":
    main()
