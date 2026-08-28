from pathlib import Path

from pydantic import BaseModel, field_validator

_CONFIGS_DIR = (Path(__file__).resolve().parents[2] / "configs").resolve()


class DbConfig(BaseModel):
    host: str
    port: int
    database: str
    user: str
    password: str

    @property
    def sqlalchemy_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


class SteamSourceConfig(BaseModel):
    base_url: str
    api_key: str | None
    requests_per_second: float
    poll_interval_hours: int


class ItadSourceConfig(BaseModel):
    base_url: str
    api_key: str
    requests_per_second: float
    history_since_default_days: int


class IgdbSourceConfig(BaseModel):
    base_url: str
    oauth_url: str
    client_id: str
    client_secret: str
    requests_per_second: float
    max_concurrent_requests: int


class SteamChartsSourceConfig(BaseModel):
    base_url: str
    user_agent: str
    min_request_interval_seconds: float
    max_retries: int
    backoff_base_seconds: float


class SourcesConfig(BaseModel):
    steam: SteamSourceConfig
    itad: ItadSourceConfig
    igdb: IgdbSourceConfig
    steamcharts: SteamChartsSourceConfig


class SeedGame(BaseModel):
    steam_appid: int
    name: str


class SeedGamesFile(BaseModel):
    games: list[SeedGame]


class PanelConfig(BaseModel):
    start_date: str
    end_date: str
    detrend_window_days: int
    platform_sale_threshold_frac: float
    output_path: str


class ItsConfig(BaseModel):
    pre_window_days: int
    post_window_days: int
    min_pre_period_obs: int


class CateConfig(BaseModel):
    xgb_n_estimators: int
    xgb_max_depth: int
    xgb_learning_rate: float
    cf_n_estimators: int
    cf_min_samples_leaf: int
    estimator_names: list[str]
    slice_dims: list[str]
    exploratory_dims: list[str]
    refutation_num_simulations: int
    refutation_seed: int


class FindingsConfig(BaseModel):
    output_path: str


class CausalConfig(BaseModel):
    panel: PanelConfig
    its: ItsConfig
    cate: CateConfig
    findings: FindingsConfig


class AppConfig(BaseModel):
    db: DbConfig
    sources: SourcesConfig
    seed_games_file: str
    causal: CausalConfig

    @field_validator("seed_games_file")
    @classmethod
    def _resolve_seed_games_file(cls, value: str) -> str:
        path = Path(value)
        return str(path if path.is_absolute() else _CONFIGS_DIR / path)
