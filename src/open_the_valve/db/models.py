import enum
from datetime import date, datetime

from sqlalchemy import (
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class PlayerCountSource(enum.StrEnum):
    STEAMCHARTS_BOOTSTRAP = "steamcharts_bootstrap"
    STEAM_API_POLL = "steam_api_poll"


class PlayerCountGranularity(enum.StrEnum):
    DAILY = "daily"
    MONTHLY = "monthly"


class Game(Base):
    __tablename__ = "games"

    id: Mapped[int] = mapped_column(primary_key=True)
    steam_appid: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    itad_id: Mapped[str | None] = mapped_column(String, unique=True)
    igdb_id: Mapped[int | None] = mapped_column(Integer, unique=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    metadata_row: Mapped["GameMetadata | None"] = relationship(
        back_populates="game", uselist=False, cascade="all, delete-orphan"
    )


class GameMetadata(Base):
    __tablename__ = "game_metadata"

    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), primary_key=True)
    genres: Mapped[list | None] = mapped_column(JSONB)
    release_date: Mapped[date | None] = mapped_column(Date)
    platforms: Mapped[list | None] = mapped_column(JSONB)
    aggregated_rating: Mapped[float | None] = mapped_column(Float)
    involved_companies: Mapped[list | None] = mapped_column(JSONB)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    game: Mapped["Game"] = relationship(back_populates="metadata_row")


class PriceHistory(Base):
    __tablename__ = "price_history"
    __table_args__ = (
        UniqueConstraint("game_id", "store", "observed_at", name="uq_price_history_obs"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), nullable=False)
    store: Mapped[str] = mapped_column(String, nullable=False)
    price: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    cut_pct: Mapped[int] = mapped_column(Integer, nullable=False)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class DiscountEvent(Base):
    __tablename__ = "discount_events"
    __table_args__ = (
        UniqueConstraint("game_id", "store", "start_at", name="uq_discount_events_start"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), nullable=False)
    store: Mapped[str] = mapped_column(String, nullable=False)
    start_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    depth_pct: Mapped[int] = mapped_column(Integer, nullable=False)


class PlayerCountHistory(Base):
    __tablename__ = "player_count_history"
    __table_args__ = (
        UniqueConstraint("game_id", "observed_date", "source", name="uq_player_count_obs"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), nullable=False)
    observed_date: Mapped[date] = mapped_column(Date, nullable=False)
    player_count: Mapped[int] = mapped_column(Integer, nullable=False)
    source: Mapped[PlayerCountSource] = mapped_column(
        Enum(PlayerCountSource, name="player_count_source"), nullable=False
    )
    granularity: Mapped[PlayerCountGranularity] = mapped_column(
        Enum(PlayerCountGranularity, name="player_count_granularity"), nullable=False
    )


class IngestionWatermark(Base):
    __tablename__ = "ingestion_watermarks"
    __table_args__ = (UniqueConstraint("source", "game_id", name="uq_watermark_source_game"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String, nullable=False)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), nullable=False)
    cursor: Mapped[str | None] = mapped_column(String)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class CausalRunHistory(Base):
    """One row per `run_causal_analysis` execution -- a pointer to the run's
    MLflow artifacts, not a copy of the panel itself. Read by the Phase 3
    retrain gate (compare current player_count_history size against
    panel_row_count) and the drift report (fetch the previous run's
    cate_predictions.parquet via mlflow_run_id).
    """

    __tablename__ = "causal_run_history"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    panel_start_date: Mapped[date] = mapped_column(Date, nullable=False)
    panel_end_date: Mapped[date] = mapped_column(Date, nullable=False)
    panel_row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    n_treated_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    mlflow_run_id: Mapped[str] = mapped_column(String, nullable=False)
