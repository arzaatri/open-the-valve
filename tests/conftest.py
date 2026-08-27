import os

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from open_the_valve.db.models import Base

os.environ.setdefault("ITAD_API_KEY", "test")
os.environ.setdefault("IGDB_CLIENT_ID", "test")
os.environ.setdefault("IGDB_CLIENT_SECRET", "test")

_TEST_DB_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+psycopg://otv:otv@localhost:5433/open_the_valve"
)


@pytest.fixture(scope="session")
def engine():
    engine = create_engine(_TEST_DB_URL)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(engine) -> Session:
    """A session bound to a transaction + savepoint that always rolls back,
    so each test starts from a clean slate without touching real data.
    """
    connection = engine.connect()
    transaction = connection.begin()
    session = sessionmaker(bind=connection)()
    session.begin_nested()

    yield session

    session.close()
    transaction.rollback()
    connection.close()
