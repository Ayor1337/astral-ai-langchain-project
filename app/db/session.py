from functools import lru_cache

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import ConfigurationError, _get_setting_value
from app.db.models import Base


def normalize_database_url(database_url: str) -> str:
    """将数据库地址归一化为 asyncpg 可用格式。

    Args:
        database_url: 原始数据库地址。

    Returns:
        归一化后的数据库地址。
    """
    normalized = database_url.strip()
    if normalized.startswith("postgresql://"):
        return normalized.replace("postgresql://", "postgresql+asyncpg://", 1)
    return normalized


def get_database_url() -> str:
    """读取并校验数据库地址。

    Returns:
        归一化后的数据库地址。

    Raises:
        ConfigurationError: 数据库地址缺失或格式非法时抛出。
    """
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
    """创建并缓存异步数据库引擎。

    Returns:
        供 SQLAlchemy 使用的异步引擎。
    """
    return create_async_engine(get_database_url(), future=True)


@lru_cache
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """创建并缓存异步会话工厂。

    Returns:
        供服务层使用的会话工厂。
    """
    return async_sessionmaker(get_engine(), expire_on_commit=False)


async def init_db() -> None:
    """初始化数据库结构并清理历史遗留字段。

    该操作是幂等的，适合在应用启动阶段执行。
    """
    engine = get_engine()
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        # 这里保留幂等式清理，兼容旧库结构，避免每次手工迁移。
        await connection.execute(
            text("ALTER TABLE conversation_messages DROP COLUMN IF EXISTS reasoning_summary")
        )
        await connection.execute(
            text("ALTER TABLE conversation_messages DROP COLUMN IF EXISTS content_blocks")
        )


async def close_db() -> None:
    """关闭数据库连接并清理缓存。

    这样可以避免热重载后继续保留失效连接。
    """
    if get_engine.cache_info().currsize == 0:
        return
    engine = get_engine()
    await engine.dispose()
    get_session_factory.cache_clear()
    get_engine.cache_clear()
