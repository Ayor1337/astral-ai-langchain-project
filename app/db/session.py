from functools import lru_cache

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import ConfigurationError, _get_setting_value
from app.db.models import Base


def normalize_database_url(database_url: str) -> str:
    normalized = database_url.strip()
    if normalized.startswith("postgresql://"):
        return normalized.replace("postgresql://", "postgresql+asyncpg://", 1)
    return normalized


def get_database_url() -> str:
    database_url = _get_setting_value("DATABASE_URL")
    if not database_url:
        raise ConfigurationError("DATABASE_URL is not configured")
    normalized = normalize_database_url(database_url)
    if not normalized.startswith(("postgresql://", "postgresql+asyncpg://")):
        raise ConfigurationError(
            "DATABASE_URL must start with postgresql:// or postgresql+asyncpg://"
        )
    return normalized


@lru_cache
def get_engine() -> AsyncEngine:
    return create_async_engine(get_database_url(), future=True)


@lru_cache
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_engine(), expire_on_commit=False)


async def init_db() -> None:
    engine = get_engine()
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        await connection.execute(
            text("ALTER TABLE conversation_messages DROP COLUMN IF EXISTS reasoning_summary")
        )
        await connection.execute(
            text("ALTER TABLE conversation_messages DROP COLUMN IF EXISTS content_blocks")
        )


async def close_db() -> None:
    if get_engine.cache_info().currsize == 0:
        return
    engine = get_engine()
    await engine.dispose()
    get_session_factory.cache_clear()
    get_engine.cache_clear()
