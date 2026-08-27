import logging
from contextlib import closing
from datetime import UTC, datetime, timedelta

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from open_the_valve.config_models import AppConfig, SeedGamesFile
from open_the_valve.db import repo
from open_the_valve.db.session import make_engine, session_scope
from open_the_valve.ingestion.itad_client import ItadClient
from open_the_valve.logging_utils import setup_logging

load_dotenv()
logger = logging.getLogger(__name__)


def run(config: AppConfig) -> None:
    seed_games = SeedGamesFile.model_validate(OmegaConf.load(config.seed_games_file))
    engine = make_engine(config.db)
    since = datetime.now(UTC) - timedelta(days=config.sources.itad.history_since_default_days)

    with closing(ItadClient(config.sources.itad)) as itad:
        for seed_game in seed_games.games:
            with session_scope(engine) as session:
                game_id = repo.upsert_game(session, seed_game.steam_appid, seed_game.name)

            itad_id = itad.lookup_by_steam_appid(seed_game.steam_appid)
            if itad_id is None:
                continue

            history = itad.get_price_history(itad_id, since=since)
            with session_scope(engine) as session:
                repo.upsert_game(session, seed_game.steam_appid, seed_game.name, itad_id=itad_id)
                for entry in history:
                    deal = entry.get("deal")
                    if deal is None:
                        continue
                    repo.upsert_price_history_row(
                        session,
                        game_id=game_id,
                        store=entry["shop"]["name"],
                        price=deal["price"]["amount"],
                        cut_pct=deal["cut"],
                        observed_at=datetime.fromisoformat(entry["timestamp"]),
                    )
                repo.rebuild_discount_events(session, game_id)
            logger.info("backfilled %d price observations for %s", len(history), seed_game.name)


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging()
    config = AppConfig.model_validate(OmegaConf.to_container(cfg, resolve=True))
    run(config)


if __name__ == "__main__":
    main()
