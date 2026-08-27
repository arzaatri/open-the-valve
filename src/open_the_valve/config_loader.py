from pathlib import Path

from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from open_the_valve.config_models import DbConfig

_CONFIGS_DIR = str((Path(__file__).resolve().parents[2] / "configs").resolve())

load_dotenv()


def load_db_config() -> DbConfig:
    """Compose just the `db` config subtree for Alembic's env.py, which runs
    outside the Hydra CLI and only ever needs the database URL.
    """
    with initialize_config_dir(config_dir=_CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="config")
    return DbConfig.model_validate(OmegaConf.to_container(cfg.db, resolve=True))
