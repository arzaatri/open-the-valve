from collections.abc import Callable
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from open_the_valve.config_models import AppConfig
from open_the_valve.logging_utils import setup_logging

_CONFIG_DIR = str((Path(__file__).resolve().parents[3] / "configs").resolve())


def hydra_entrypoint(run: Callable[[AppConfig], None]) -> Callable[[], None]:
    """Wraps a `run(config: AppConfig) -> None` pipeline function into a
    Hydra-composed CLI entrypoint: compose config -> validate into AppConfig
    -> setup_logging() -> run(config). Every pipeline in pipelines/ follows
    this same sequence, so it lives here once instead of once per file.

    config_path must be absolute here: Hydra resolves a relative config_path
    against the module that literally contains the `@hydra.main` decoration,
    which is this file, not the calling pipeline module.
    """

    @hydra.main(config_path=_CONFIG_DIR, config_name="config", version_base=None)
    def main(cfg: DictConfig) -> None:
        setup_logging()
        config = AppConfig.model_validate(OmegaConf.to_container(cfg, resolve=True))
        run(config)

    return main
