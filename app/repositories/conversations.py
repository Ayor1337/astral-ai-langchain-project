from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation, ConversationMessage


def utcnow() -> datetime:
    """仓储层统一使用 UTC 时间更新审计字段。

    这样所有时间写入都来自同一时区基准。
    """
    return datetime.now(timezone.utc)


class ConversationRepository:
    def __init__(self, session: AsyncSession):
        """初始化会话仓储。

        Args:
            self: 仓储实例本身。
            session: 当前异步数据库会话。
        """
        self.session = session

    async def create_conversation(
        self,
        title: str,
        user_id: str | None = None,
        system_prompt: str | None = None,
    ) -> Conversation:
        """创建会话并立即 flush，确保调用方能拿到主键。

        这样后续消息写入可以直接复用数据库生成的 id。
        """
        conversation = Conversation(
            title=title,
            user_id=user_id,
            system_prompt=system_prompt,
        )
        self.session.add(conversation)
        await self.session.flush()
        return conversation

    async def get_conversation(
        self,
        conversation_id: UUID,
        *,
        user_id: str | None = None,
        include_deleted: bool = False,
    ) -> Conversation | None:
        """按需过滤软删除会话，避免业务层重复拼接条件。

        调用方只需决定是否需要包含已删除数据。

        Args:
            conversation_id: 会话 ID。
            user_id: 当前用户 ID；为空时不追加归属过滤。
            include_deleted: 是否包含已软删除会话。

        Returns:
            匹配的会话实体，不存在时返回 `None`。
        """
        query = select(Conversation).where(Conversation.id == conversation_id)
        if user_id:
            query = query.where(Conversation.user_id == user_id)
        if not include_deleted:
            query = query.where(Conversation.deleted_at.is_(None))
        return await self.session.scalar(query)

    async def list_active_conversations(self, *, user_id: str) -> list[Conversation]:
        """按最近更新时间倒序返回仍可见的会话。

        该方法只负责可见性与排序，不做额外加工。

        Args:
            user_id: 当前用户 ID。

        Returns:
            当前用户的可见会话列表。
        """
        result = await self.session.scalars(
            select(Conversation)
            .where(
                Conversation.deleted_at.is_(None),
                Conversation.user_id == user_id,
            )
            .order_by(desc(Conversation.updated_at))
        )
        return list(result.all())

    async def update_title(self, conversation: Conversation, title: str) -> Conversation:
        """修改标题时同时刷新 updated_at，保持列表排序稳定。

        这样列表接口可以直接按更新时间展示最新改动。
        """
        conversation.title = title
        conversation.updated_at = utcnow()
        await self.session.flush()
        return conversation

    async def soft_delete(self, conversation: Conversation) -> None:
        """通过 deleted_at 标记删除，而不是直接物理删除。

        保留历史数据，便于详情查询和审计。
        """
        now = utcnow()
        conversation.deleted_at = now
        conversation.updated_at = now
        await self.session.flush()

    async def add_message(
        self,
        conversation: Conversation,
        *,
        role: str,
        content: str,
        trace_steps: list[dict[str, object]] | None = None,
    ) -> ConversationMessage:
        """为消息分配单会话递增 sequence，供上下文窗口和展示层复用。

        `sequence` 保证同一会话内消息顺序稳定。
        """
        current_max = await self.session.scalar(
            select(func.coalesce(func.max(ConversationMessage.sequence), 0)).where(
                ConversationMessage.conversation_id == conversation.id
            )
        )
        message = ConversationMessage(
            conversation_id=conversation.id,
            role=role,
            content=content,
            trace_steps=trace_steps,
            sequence=int(current_max or 0) + 1,
        )
        self.session.add(message)
        conversation.updated_at = utcnow()
        await self.session.flush()
        return message

    async def update_message_trace(
        self,
        message: ConversationMessage,
        *,
        trace_steps: list[dict[str, object]] | None,
    ) -> ConversationMessage:
        """在生成结束后补写 trace，避免消息正文和追踪信息的写入时序互相阻塞。

        这样正文先落库，trace 可在后续异步阶段补齐。
        """
        message.trace_steps = trace_steps
        await self.session.flush()
        return message

    async def list_messages(self, conversation_id: UUID) -> list[ConversationMessage]:
        """按自然会话顺序返回全部消息。

        结果顺序可直接用于详情页展示。
        """
        result = await self.session.scalars(
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == conversation_id)
            .order_by(ConversationMessage.sequence)
        )
        return list(result.all())

    async def get_message(self, message_id: int) -> ConversationMessage | None:
        """按主键加载单条消息。

        返回 `None` 表示消息不存在或已被上层条件过滤。
        """
        return await self.session.get(ConversationMessage, message_id)

    async def list_recent_messages(
        self,
        conversation_id: UUID,
        *,
        limit: int,
        before_sequence: int | None = None,
    ) -> list[ConversationMessage]:
        """截取当前消息之前的最近窗口，并在返回前恢复正序。

        用于为模型构建有限长度的上下文。
        """
        query = select(ConversationMessage).where(ConversationMessage.conversation_id == conversation_id)
        if before_sequence is not None:
            query = query.where(ConversationMessage.sequence < before_sequence)
        result = await self.session.scalars(
            query.order_by(desc(ConversationMessage.sequence)).limit(limit)
        )
        messages = list(result.all())
        messages.reverse()
        return messages

    async def count_messages(self, conversation_id: UUID) -> int:
        """统计单会话消息总数，供摘要刷新阈值判断使用。

        只返回数量，不加载实体。
        """
        count = await self.session.scalar(
            select(func.count())
            .select_from(ConversationMessage)
            .where(ConversationMessage.conversation_id == conversation_id)
        )
        return int(count or 0)

    async def list_messages_for_summary(
        self,
        conversation_id: UUID,
        *,
        from_sequence_exclusive: int,
        to_sequence_inclusive: int,
    ) -> list[ConversationMessage]:
        """取出尚未汇总且已脱离上下文窗口的消息区间。

        这样摘要只覆盖真正需要压缩的历史消息。
        """
        result = await self.session.scalars(
            select(ConversationMessage)
            .where(
                ConversationMessage.conversation_id == conversation_id,
                ConversationMessage.sequence > from_sequence_exclusive,
                ConversationMessage.sequence <= to_sequence_inclusive,
            )
            .order_by(ConversationMessage.sequence)
        )
        return list(result.all())

    async def update_summary(
        self,
        conversation: Conversation,
        *,
        summary: str,
        summary_message_count: int,
    ) -> Conversation:
        """更新会话摘要及其已汇总边界。

        摘要写回后可以继续沿用同一窗口策略。
        """
        conversation.summary = summary
        conversation.summary_message_count = summary_message_count
        conversation.updated_at = utcnow()
        await self.session.flush()
        return conversation
