from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import User


def utcnow() -> datetime:
    """仓储层统一生成 UTC 时间。

    Returns:
        当前 UTC 时间。
    """
    return datetime.now(timezone.utc)


class UserRepository:
    """封装用户表的最小读写能力。"""

    def __init__(self, session: AsyncSession):
        """初始化用户仓储。

        Args:
            session: 当前异步数据库会话。
        """
        self.session = session

    async def create_user(
        self,
        *,
        username: str,
        nickname: str,
        password_hash: str,
    ) -> User:
        """创建用户并立即 flush。

        Args:
            username: 归一化后的用户名。
            nickname: 展示昵称。
            password_hash: 已哈希密码。

        Returns:
            新创建的用户实体。
        """
        user = User(
            username=username,
            nickname=nickname,
            password_hash=password_hash,
        )
        self.session.add(user)
        await self.session.flush()
        return user

    async def get_by_username(self, username: str) -> User | None:
        """按用户名加载用户。

        Args:
            username: 已归一化用户名。

        Returns:
            匹配的用户实体，不存在时返回 `None`。
        """
        return await self.session.scalar(select(User).where(User.username == username))

    async def get_by_id(self, user_id: UUID | str) -> User | None:
        """按主键加载用户。

        Args:
            user_id: 用户主键。

        Returns:
            匹配的用户实体，不存在时返回 `None`。
        """
        normalized = user_id if isinstance(user_id, UUID) else UUID(str(user_id))
        return await self.session.get(User, normalized)

    async def update_nickname(self, user: User, nickname: str) -> User:
        """更新昵称并刷新更新时间。

        Args:
            user: 待更新用户。
            nickname: 新昵称。

        Returns:
            更新后的用户实体。
        """
        user.nickname = nickname
        user.updated_at = utcnow()
        await self.session.flush()
        return user

    async def update_username(self, user: User, username: str) -> User:
        """更新用户名并刷新更新时间。

        Args:
            user: 待更新用户。
            username: 新用户名。

        Returns:
            更新后的用户实体。
        """
        user.username = username
        user.updated_at = utcnow()
        await self.session.flush()
        return user
