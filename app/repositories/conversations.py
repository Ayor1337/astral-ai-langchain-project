from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation, ConversationMessage


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ConversationRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_conversation(
        self,
        title: str,
        user_id: str | None = None,
        system_prompt: str | None = None,
    ) -> Conversation:
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
        include_deleted: bool = False,
    ) -> Conversation | None:
        query = select(Conversation).where(Conversation.id == conversation_id)
        if not include_deleted:
            query = query.where(Conversation.deleted_at.is_(None))
        return await self.session.scalar(query)

    async def list_active_conversations(self) -> list[Conversation]:
        result = await self.session.scalars(
            select(Conversation)
            .where(Conversation.deleted_at.is_(None))
            .order_by(desc(Conversation.updated_at))
        )
        return list(result.all())

    async def update_title(self, conversation: Conversation, title: str) -> Conversation:
        conversation.title = title
        conversation.updated_at = utcnow()
        await self.session.flush()
        return conversation

    async def soft_delete(self, conversation: Conversation) -> None:
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
        message.trace_steps = trace_steps
        await self.session.flush()
        return message

    async def list_messages(self, conversation_id: UUID) -> list[ConversationMessage]:
        result = await self.session.scalars(
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == conversation_id)
            .order_by(ConversationMessage.sequence)
        )
        return list(result.all())

    async def get_message(self, message_id: int) -> ConversationMessage | None:
        return await self.session.get(ConversationMessage, message_id)

    async def list_recent_messages(
        self,
        conversation_id: UUID,
        *,
        limit: int,
        before_sequence: int | None = None,
    ) -> list[ConversationMessage]:
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
        conversation.summary = summary
        conversation.summary_message_count = summary_message_count
        conversation.updated_at = utcnow()
        await self.session.flush()
        return conversation
