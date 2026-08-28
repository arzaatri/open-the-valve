import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from open_the_valve.config_models import AppConfig
from open_the_valve.features.build_panel import run
from open_the_valve.logging_utils import setup_logging

load_dotenv()


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging()
    config = AppConfig.model_validate(OmegaConf.to_container(cfg, resolve=True))
    run(config)


if __name__ == "__main__":
    main()
