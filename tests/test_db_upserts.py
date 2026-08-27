from datetime import UTC, date, datetime

from sqlalchemy import func, select

from open_the_valve.db import repo
from open_the_valve.db.models import Game, PlayerCountGranularity, PlayerCountSource, PriceHistory


def test_upsert_game_is_idempotent(db_session):
    first_id = repo.upsert_game(db_session, steam_appid=999001, name="Test Game")
    second_id = repo.upsert_game(db_session, steam_appid=999001, name="Test Game Renamed")

    assert first_id == second_id
    count = db_session.execute(
        select(func.count()).select_from(Game).where(Game.steam_appid == 999001)
    ).scalar_one()
    assert count == 1

    name = db_session.execute(select(Game.name).where(Game.id == first_id)).scalar_one()
    assert name == "Test Game Renamed"


def test_upsert_price_history_row_is_idempotent(db_session):
    game_id = repo.upsert_game(db_session, steam_appid=999002, name="Price Test Game")
    observed_at = datetime(2026, 1, 1, tzinfo=UTC)

    repo.upsert_price_history_row(db_session, game_id, "steam", 19.99, 0, observed_at)
    repo.upsert_price_history_row(db_session, game_id, "steam", 9.99, 50, observed_at)

    rows = (
        db_session.execute(select(PriceHistory).where(PriceHistory.game_id == game_id))
        .scalars()
        .all()
    )
    assert len(rows) == 1
    assert float(rows[0].price) == 9.99
    assert rows[0].cut_pct == 50


def test_rebuild_discount_events_merges_consecutive_days(db_session):
    game_id = repo.upsert_game(db_session, steam_appid=999003, name="Discount Test Game")
    for day in (1, 2, 3):
        repo.upsert_price_history_row(
            db_session, game_id, "steam", 9.99, 50, datetime(2026, 1, day, tzinfo=UTC)
        )
    repo.upsert_price_history_row(
        db_session, game_id, "steam", 19.99, 0, datetime(2026, 1, 4, tzinfo=UTC)
    )

    repo.rebuild_discount_events(db_session, game_id)

    from open_the_valve.db.models import DiscountEvent

    events = (
        db_session.execute(select(DiscountEvent).where(DiscountEvent.game_id == game_id))
        .scalars()
        .all()
    )
    assert len(events) == 1
    assert events[0].start_at.date() == date(2026, 1, 1)
    assert events[0].end_at.date() == date(2026, 1, 3)
    assert events[0].depth_pct == 50


def test_upsert_player_count_row_is_idempotent(db_session):
    game_id = repo.upsert_game(db_session, steam_appid=999004, name="Player Count Test Game")
    today = date(2026, 1, 1)

    repo.upsert_player_count_row(
        db_session,
        game_id,
        today,
        100,
        PlayerCountSource.STEAM_API_POLL,
        PlayerCountGranularity.DAILY,
    )
    repo.upsert_player_count_row(
        db_session,
        game_id,
        today,
        150,
        PlayerCountSource.STEAM_API_POLL,
        PlayerCountGranularity.DAILY,
    )

    count = repo.get_player_count(db_session, game_id, today, PlayerCountSource.STEAM_API_POLL)
    assert count == 150


def test_watermark_roundtrip(db_session):
    game_id = repo.upsert_game(db_session, steam_appid=999005, name="Watermark Test Game")
    assert repo.get_watermark(db_session, "steamcharts_bootstrap", game_id) is None

    repo.set_watermark(db_session, "steamcharts_bootstrap", game_id, "done")
    assert repo.get_watermark(db_session, "steamcharts_bootstrap", game_id) == "done"
