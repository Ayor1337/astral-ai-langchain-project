from uuid import UUID

from app.db.models import Conversation, ConversationMessage
from app.db.session import get_session_factory
from app.repositories.conversations import ConversationRepository
from app.schemas.conversation import ConversationDetail, ConversationListItem, ConversationMessageView
from app.services.exceptions import ConversationNotFoundError

DEFAULT_CONVERSATION_TITLE = "新对话"


def _to_list_item(conversation: Conversation) -> ConversationListItem:
    return ConversationListItem(
        id=conversation.id,
        title=conversation.title,
        summary=conversation.summary,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


def _to_message_view(message: ConversationMessage) -> ConversationMessageView:
    return ConversationMessageView(
        role=message.role,
        content=message.content,
        reasoning_summary=message.reasoning_summary,
        trace_steps=message.trace_steps,
        sequence=message.sequence,
        created_at=message.created_at,
    )


async def create_conversation() -> ConversationListItem:
    session_factory = get_session_factory()
    async with session_factory() as session:
        repository = ConversationRepository(session)
        conversation = await repository.create_conversation(title=DEFAULT_CONVERSATION_TITLE)
        await session.commit()
        return _to_list_item(conversation)


async def list_conversations() -> list[ConversationListItem]:
    session_factory = get_session_factory()
    async with session_factory() as session:
        repository = ConversationRepository(session)
        conversations = await repository.list_active_conversations()
        return [_to_list_item(conversation) for conversation in conversations]


async def get_conversation_detail(conversation_id: UUID) -> ConversationDetail:
    session_factory = get_session_factory()
    async with session_factory() as session:
        repository = ConversationRepository(session)
        conversation = await repository.get_conversation(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError("conversation not found")
        messages = await repository.list_messages(conversation_id)
        return ConversationDetail(
            **_to_list_item(conversation).model_dump(),
            messages=[_to_message_view(message) for message in messages],
        )


async def update_conversation_title(conversation_id: UUID, title: str) -> ConversationListItem:
    session_factory = get_session_factory()
    async with session_factory() as session:
        repository = ConversationRepository(session)
        conversation = await repository.get_conversation(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError("conversation not found")
        conversation = await repository.update_title(conversation, title)
        await session.commit()
        return _to_list_item(conversation)


async def delete_conversation(conversation_id: UUID) -> None:
    session_factory = get_session_factory()
    async with session_factory() as session:
        repository = ConversationRepository(session)
        conversation = await repository.get_conversation(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError("conversation not found")
        await repository.soft_delete(conversation)
        await session.commit()
