import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.engine import Engine

from open_the_valve.config_models import AppConfig, PanelConfig
from open_the_valve.db.models import (
    DiscountEvent,
    Game,
    GameMetadata,
    PlayerCountGranularity,
    PlayerCountHistory,
    PriceHistory,
)
from open_the_valve.db.session import make_engine

load_dotenv()
logger = logging.getLogger(__name__)

_SEASON_BY_MONTH = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "fall",
    10: "fall",
    11: "fall",
}
_GAME_AGE_BUCKET_EDGES = [0, 365, 3 * 365, 7 * 365, np.inf]
_GAME_AGE_BUCKET_LABELS = ["<1y", "1-3y", "3-7y", "7y+"]
_DEPTH_BUCKET_EDGES = [-1, 0, 30, 50, 70, 100]
_DEPTH_BUCKET_LABELS = ["none", "0-30", "30-50", "50-70", "70-100"]
_TREATMENT_STORE = "Steam"


@dataclass
class RawTables:
    games: pd.DataFrame
    metadata: pd.DataFrame
    price_history: pd.DataFrame
    discount_events: pd.DataFrame
    player_counts: pd.DataFrame


def load_raw_tables(engine: Engine) -> RawTables:
    """Pull the full contents of every table Phase 2 needs. Data volume is small
    enough (tens of thousands of rows total) that filtering happens in pandas
    inside build_panel, not via extra WHERE clauses here.
    """
    games = pd.read_sql(select(Game.id, Game.steam_appid, Game.name), engine)
    metadata = pd.read_sql(
        select(
            GameMetadata.game_id,
            GameMetadata.genres,
            GameMetadata.release_date,
            GameMetadata.aggregated_rating,
        ),
        engine,
    )
    price_history = pd.read_sql(
        select(PriceHistory.game_id, PriceHistory.store, PriceHistory.price, PriceHistory.cut_pct),
        engine,
    )
    discount_events = pd.read_sql(
        select(
            DiscountEvent.game_id,
            DiscountEvent.store,
            DiscountEvent.start_at,
            DiscountEvent.end_at,
            DiscountEvent.depth_pct,
        ),
        engine,
    )
    player_counts = pd.read_sql(
        select(
            PlayerCountHistory.game_id,
            PlayerCountHistory.observed_date,
            PlayerCountHistory.player_count,
        ).where(PlayerCountHistory.granularity == PlayerCountGranularity.DAILY),
        engine,
    )
    return RawTables(
        games=games,
        metadata=metadata,
        price_history=price_history,
        discount_events=discount_events,
        player_counts=player_counts,
    )


def _build_date_spine(game_ids: list[int], start_date: date, end_date: date) -> pd.DataFrame:
    dates = pd.date_range(start_date, end_date, freq="D").date
    return pd.DataFrame([(gid, d) for gid in game_ids for d in dates], columns=["game_id", "date"])


def build_treatment_frame(discount_events: pd.DataFrame, date_spine: pd.DataFrame) -> pd.DataFrame:
    """Expands DiscountEvent [start_at, end_at] intervals into per-(game_id, date)
    is_discounted / depth_pct rows. An event with a null end_at is still open as
    of the last ingestion run, so it's treated as active through the spine's max
    date. Where multiple stores/events overlap a game-day, depth_pct is the max
    (deepest discount that day).
    """
    if discount_events.empty:
        spine = date_spine.copy()
        spine["is_discounted"] = False
        spine["depth_pct"] = 0
        return spine

    spine_max_date = date_spine["date"].max()
    events = discount_events.copy()
    events["start_date"] = events["start_at"].dt.tz_localize(None).dt.date
    events["end_date"] = events["end_at"].dt.tz_localize(None).dt.date
    events["end_date"] = events["end_date"].fillna(spine_max_date)

    expanded_rows = []
    for row in events.itertuples(index=False):
        for d in pd.date_range(row.start_date, min(row.end_date, spine_max_date), freq="D").date:
            expanded_rows.append((row.game_id, d, row.depth_pct))
    expanded = pd.DataFrame(expanded_rows, columns=["game_id", "date", "depth_pct"])
    daily_depth = expanded.groupby(["game_id", "date"], as_index=False)["depth_pct"].max()

    merged = date_spine.merge(daily_depth, on=["game_id", "date"], how="left")
    merged["is_discounted"] = merged["depth_pct"].notna()
    merged["depth_pct"] = merged["depth_pct"].fillna(0).astype(int)
    return merged


def compute_detrended_outcome(panel: pd.DataFrame, window_days: int) -> pd.Series:
    """Trailing median of player_count over the last `window_days` NON-discounted
    days, excluding the current day itself (shift(1)) so a day's baseline never
    includes its own outcome. Discounted days are masked to NaN before the
    rolling window so they never enter any baseline. Forward-filled per game so
    a discounted stretch still has a (stale) baseline to compare against; a
    game with no non-discounted history yet in the window is left NaN.
    """
    ordered = panel.sort_values(["game_id", "date"])
    masked = ordered["player_count"].where(~ordered["is_discounted"])
    baseline = masked.groupby(ordered["game_id"]).transform(
        lambda s: s.shift(1).rolling(window_days, min_periods=1).median()
    )
    baseline = baseline.groupby(ordered["game_id"]).ffill()
    detrended = (ordered["player_count"] - baseline).reindex(panel.index)
    return detrended


def flag_platform_sale_windows(panel: pd.DataFrame, threshold_frac: float) -> pd.Series:
    """Per-date fraction of tracked games under active discount; True where that
    fraction meets or exceeds threshold_frac. Derived from the ingested data
    rather than a hand-maintained calendar of known Steam sale dates.
    """
    daily_frac = panel.groupby("date")["is_discounted"].transform("mean")
    return daily_frac >= threshold_frac


def _bucket_price_tier(baseline_price: pd.Series) -> pd.Series:
    return pd.qcut(
        baseline_price, q=4, labels=["budget", "mid", "premium", "AAA"], duplicates="drop"
    )


def _primary_genre(genres: object) -> str:
    if isinstance(genres, list) and genres:
        return genres[0]
    return "Unknown"


def build_panel(raw: RawTables, config: PanelConfig) -> pd.DataFrame:
    """Pure orchestrator: no DB access. Produces one row per (game_id, date) over
    [config.start_date, config.end_date] with outcome, treatment, and covariate
    columns ready for causal/its.py and causal/cate_estimators.py.
    """
    spine = _build_date_spine(raw.games["id"].tolist(), config.start_date, config.end_date)
    # Outcome is Steam's own concurrent-player count, so treatment is scoped to
    # Steam-storefront discounts specifically -- third-party reseller discounts
    # (GOG, Epic, Fanatical, etc.) reach players through a different, indirect
    # channel and would dilute the treatment signal if mixed in.
    steam_events = raw.discount_events[raw.discount_events["store"] == _TREATMENT_STORE]
    panel = build_treatment_frame(steam_events, spine)

    player_counts = raw.player_counts.rename(columns={"observed_date": "date"})
    panel = panel.merge(player_counts, on=["game_id", "date"], how="left")
    panel["log_player_count"] = np.log1p(panel["player_count"])
    panel["detrended_player_count"] = compute_detrended_outcome(panel, config.detrend_window_days)
    panel["is_platform_sale_window"] = flag_platform_sale_windows(
        panel, config.platform_sale_threshold_frac
    )
    panel["day_of_week"] = pd.to_datetime(panel["date"]).dt.day_name()
    panel["season"] = pd.to_datetime(panel["date"]).dt.month.map(_SEASON_BY_MONTH)
    panel["depth_bucket"] = pd.cut(
        panel["depth_pct"], bins=_DEPTH_BUCKET_EDGES, labels=_DEPTH_BUCKET_LABELS
    )

    baseline_price = (
        raw.price_history[
            (raw.price_history["store"] == _TREATMENT_STORE) & (raw.price_history["cut_pct"] == 0)
        ]
        .groupby("game_id")["price"]
        .median()
        .rename("baseline_price")
    )
    panel = panel.merge(baseline_price, on="game_id", how="left")
    panel["price_tier"] = _bucket_price_tier(panel["baseline_price"])

    metadata = raw.metadata.copy()
    metadata["genre"] = metadata["genres"].apply(_primary_genre)
    panel = panel.merge(metadata[["game_id", "genre", "release_date"]], on="game_id", how="left")
    panel["days_since_release"] = (
        pd.to_datetime(panel["date"]) - pd.to_datetime(panel["release_date"])
    ).dt.days
    panel["game_age_bucket"] = pd.cut(
        panel["days_since_release"], bins=_GAME_AGE_BUCKET_EDGES, labels=_GAME_AGE_BUCKET_LABELS
    )

    return panel


def coverage_report(panel: pd.DataFrame) -> pd.DataFrame:
    report = panel.groupby("game_id").agg(
        n_days=("date", "count"),
        n_with_player_count=("player_count", "count"),
        n_discounted_days=("is_discounted", "sum"),
        n_with_detrended_outcome=("detrended_player_count", lambda s: s.notna().sum()),
    )
    report["player_count_coverage_frac"] = report["n_with_player_count"] / report["n_days"]
    return report


def run(config: AppConfig) -> None:
    engine = make_engine(config.db)
    raw = load_raw_tables(engine)
    panel = build_panel(raw, config.causal.panel)

    report = coverage_report(panel)
    logger.info(
        "panel covers %d games x %d days; mean player-count coverage %.1f%%; "
        "total discounted-game-days=%d",
        panel["game_id"].nunique(),
        panel["date"].nunique(),
        report["player_count_coverage_frac"].mean() * 100,
        int(panel["is_discounted"].sum()),
    )

    output_path = config.causal.panel.output_path
    panel.to_parquet(output_path, index=False)
    logger.info("wrote panel with %d rows to %s", len(panel), output_path)
