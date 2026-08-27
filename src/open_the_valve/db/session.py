from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from open_the_valve.config_models import DbConfig


def make_engine(db_config: DbConfig) -> Engine:
    return create_engine(db_config.sqlalchemy_url, pool_pre_ping=True)


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    """Yield a transactional session that commits on success and rolls back on error."""
    factory = sessionmaker(bind=engine)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
