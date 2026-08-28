from datetime import date, datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from open_the_valve.db.models import (
    CausalRunHistory,
    DiscountEvent,
    Game,
    GameMetadata,
    IngestionWatermark,
    PlayerCountGranularity,
    PlayerCountHistory,
    PlayerCountSource,
    PriceHistory,
)


def upsert_game(
    session: Session,
    steam_appid: int,
    name: str,
    itad_id: str | None = None,
    igdb_id: int | None = None,
) -> int:
    """Insert a game if unseen, or update its known external IDs. Returns games.id."""
    insert_stmt = insert(Game).values(
        steam_appid=steam_appid, name=name, itad_id=itad_id, igdb_id=igdb_id
    )
    upsert_stmt = insert_stmt.on_conflict_do_update(
        index_elements=[Game.steam_appid],
        set_={
            "name": insert_stmt.excluded.name,
            "itad_id": insert_stmt.excluded.itad_id,
            "igdb_id": insert_stmt.excluded.igdb_id,
        },
    ).returning(Game.id)
    return session.execute(upsert_stmt).scalar_one()


def upsert_game_metadata(
    session: Session,
    game_id: int,
    genres: list | None,
    release_date: date | None,
    platforms: list | None,
    aggregated_rating: float | None,
    involved_companies: list | None,
) -> None:
    stmt = insert(GameMetadata).values(
        game_id=game_id,
        genres=genres,
        release_date=release_date,
        platforms=platforms,
        aggregated_rating=aggregated_rating,
        involved_companies=involved_companies,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[GameMetadata.game_id],
        set_={
            "genres": stmt.excluded.genres,
            "release_date": stmt.excluded.release_date,
            "platforms": stmt.excluded.platforms,
            "aggregated_rating": stmt.excluded.aggregated_rating,
            "involved_companies": stmt.excluded.involved_companies,
        },
    )
    session.execute(stmt)


def upsert_price_history_row(
    session: Session,
    game_id: int,
    store: str,
    price: float,
    cut_pct: int,
    observed_at: datetime,
) -> None:
    stmt = insert(PriceHistory).values(
        game_id=game_id, store=store, price=price, cut_pct=cut_pct, observed_at=observed_at
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_price_history_obs",
        set_={"price": stmt.excluded.price, "cut_pct": stmt.excluded.cut_pct},
    )
    session.execute(stmt)


def upsert_player_count_row(
    session: Session,
    game_id: int,
    observed_date_: date,
    player_count: int,
    source: PlayerCountSource,
    granularity: PlayerCountGranularity,
) -> None:
    stmt = insert(PlayerCountHistory).values(
        game_id=game_id,
        observed_date=observed_date_,
        player_count=player_count,
        source=source,
        granularity=granularity,
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_player_count_obs",
        set_={"player_count": stmt.excluded.player_count, "granularity": stmt.excluded.granularity},
    )
    session.execute(stmt)


def list_games(session: Session) -> list[tuple[int, int]]:
    return [(row.id, row.steam_appid) for row in session.execute(select(Game.id, Game.steam_appid))]


def get_player_count(
    session: Session, game_id: int, observed_date_: date, source: PlayerCountSource
) -> int | None:
    return session.execute(
        select(PlayerCountHistory.player_count).where(
            PlayerCountHistory.game_id == game_id,
            PlayerCountHistory.observed_date == observed_date_,
            PlayerCountHistory.source == source,
        )
    ).scalar_one_or_none()


def get_watermark(session: Session, source: str, game_id: int) -> str | None:
    row = session.execute(
        select(IngestionWatermark.cursor).where(
            IngestionWatermark.source == source, IngestionWatermark.game_id == game_id
        )
    ).scalar_one_or_none()
    return row


def set_watermark(session: Session, source: str, game_id: int, cursor: str) -> None:
    stmt = insert(IngestionWatermark).values(source=source, game_id=game_id, cursor=cursor)
    stmt = stmt.on_conflict_do_update(
        constraint="uq_watermark_source_game", set_={"cursor": stmt.excluded.cursor}
    )
    session.execute(stmt)


def rebuild_discount_events(session: Session, game_id: int) -> None:
    """Derive discount_events for a game from its price_history via a
    gaps-and-islands pass: consecutive daily observations at the same store
    with the same nonzero cut_pct collapse into a single event spanning
    their date range.
    """
    rows = session.execute(
        select(PriceHistory.store, PriceHistory.observed_at, PriceHistory.cut_pct)
        .where(PriceHistory.game_id == game_id, PriceHistory.cut_pct > 0)
        .order_by(PriceHistory.store, PriceHistory.observed_at)
    ).all()

    session.query(DiscountEvent).filter(DiscountEvent.game_id == game_id).delete()

    events: list[dict] = []
    open_event: dict | None = None
    for store, observed_at, cut_pct in rows:
        if (
            open_event is not None
            and open_event["store"] == store
            and open_event["depth_pct"] == cut_pct
            and (observed_at - open_event["end_at"]).days <= 1
        ):
            open_event["end_at"] = observed_at
        else:
            if open_event is not None:
                events.append(open_event)
            open_event = {
                "store": store,
                "start_at": observed_at,
                "end_at": observed_at,
                "depth_pct": cut_pct,
            }
    if open_event is not None:
        events.append(open_event)

    for event in events:
        session.execute(insert(DiscountEvent).values(game_id=game_id, **event))


def record_causal_run(
    session: Session,
    panel_start_date: date,
    panel_end_date: date,
    panel_row_count: int,
    n_treated_rows: int,
    mlflow_run_id: str,
) -> None:
    """Appends one row to the causal-analysis run ledger. Not an upsert --
    every run_causal_analysis execution gets its own row, an append-only log
    the retrain gate and drift report read from."""
    session.execute(
        insert(CausalRunHistory).values(
            panel_start_date=panel_start_date,
            panel_end_date=panel_end_date,
            panel_row_count=panel_row_count,
            n_treated_rows=n_treated_rows,
            mlflow_run_id=mlflow_run_id,
        )
    )


def get_latest_causal_run(session: Session) -> CausalRunHistory | None:
    return session.execute(
        select(CausalRunHistory).order_by(CausalRunHistory.id.desc()).limit(1)
    ).scalar_one_or_none()
