"""Test setup. DB tests use a dedicated *_test database that is rebuilt from the Alembic migrations.

Set TEST_DATABASE_URL to override. No external NASA services are contacted.
"""

import os
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+psycopg://marsnav:marsnav@localhost:5434/marsnav_test"
)
# Must be set before the app is imported so the app and Alembic target the test database.
os.environ["DATABASE_URL"] = TEST_DATABASE_URL

from app.core.database import engine as app_engine  # noqa: E402
from app.core.database import get_db  # noqa: E402
from app.main import app  # noqa: E402
from app.models import ExplorationSite, ScienceTarget  # noqa: E402
from app.services import hazard_service, terrain_service  # noqa: E402
from app.services.storage import AssetStorage, get_storage  # noqa: E402
from scripts.seed_dev_data import seed_development_data  # noqa: E402

BACKEND_DIR = Path(__file__).resolve().parents[1]


def _rebuild_test_database() -> str | None:
    """Returns a skip reason if the database server is unreachable."""
    url = make_url(TEST_DATABASE_URL)
    if not (url.database or "").endswith("_test"):
        raise RuntimeError("Refusing to run tests against a database whose name does not end in _test")

    admin = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as connection:
            exists = connection.scalar(
                text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": url.database}
            )
            if not exists:
                connection.execute(text(f'CREATE DATABASE "{url.database}"'))
    except OperationalError as exc:
        return f"PostGIS test database unavailable (start it with `docker compose up -d db`): {exc.orig}"
    finally:
        admin.dispose()

    with app_engine.begin() as connection:
        connection.execute(text("DROP SCHEMA public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))

    config = Config(str(BACKEND_DIR / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    command.upgrade(config, "head")
    return None


@pytest.fixture(scope="session")
def engine() -> Engine:
    reason = _rebuild_test_database()
    if reason:
        pytest.skip(reason)
    assert app_engine.url.database == make_url(TEST_DATABASE_URL).database
    return app_engine


@pytest.fixture
def db_session(engine: Engine) -> Generator[Session, None, None]:
    """Each test runs in a transaction that is rolled back; service commits become savepoints."""
    connection = engine.connect()
    outer = connection.begin()
    session = Session(
        bind=connection,
        join_transaction_mode="create_savepoint",
        autoflush=False,
        expire_on_commit=False,
    )
    try:
        yield session
    finally:
        session.close()
        outer.rollback()
        connection.close()


@pytest.fixture
def client(db_session: Session, storage):
    from fastapi.testclient import TestClient

    def override_get_db() -> Generator[Session, None, None]:
        try:
            yield db_session
        except Exception:
            db_session.rollback()
            raise

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_storage] = lambda: storage
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def _clear_terrain_caches():
    from app.core.limits import route_rate_limiter

    terrain_service.clear_cache()
    hazard_service.clear_cache()
    route_rate_limiter.reset()
    yield
    terrain_service.clear_cache()
    hazard_service.clear_cache()
    route_rate_limiter.reset()


@pytest.fixture
def storage(tmp_path) -> AssetStorage:
    return AssetStorage(tmp_path / "data")


@pytest.fixture
def settings(tmp_path):
    from app.core.config import Settings

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    return Settings(data_dir=tmp_path / "data", ingest_input_dir=inbox)


@pytest.fixture
def seeded(db_session: Session) -> SimpleNamespace:
    seed_development_data(db_session)
    site = db_session.scalars(select(ExplorationSite)).one()
    targets = list(db_session.scalars(select(ScienceTarget).order_by(ScienceTarget.name)))
    return SimpleNamespace(site=site, targets=targets)


@pytest.fixture
def demo_terrain(db_session: Session, storage, settings, seeded):
    """DEMO terrain at 20 m cells (400x400) for the seeded site."""
    from app.services.ingestion_service import process_demo_terrain

    return process_demo_terrain(db_session, storage, settings, seeded.site.slug, resolution_m=20.0)
