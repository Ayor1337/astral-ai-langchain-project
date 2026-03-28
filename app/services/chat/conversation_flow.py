from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from app.schemas.chat import ChatMessage, ChatRequest
from app.services.conversation_service import DEFAULT_CONVERSATION_TITLE
from app.services.exceptions import ConversationNotFoundError
from app.services.memory_service import build_context_messages


@dataclass(slots=True)
class PreparedChatContext:
    conversation: Any
    conversation_title_at_stream_start: str
    llm_messages: list[ChatMessage]
    should_generate_title: bool


async def get_or_create_conversation(
    request: ChatRequest,
    *,
    session_factory: Any,
    repository_factory: Callable[[Any], Any],
) -> Any:
    """加载已有会话，或在旧调用路径下隐式创建会话。

    兼容没有 `conversation_id` 的旧入口。
    """
    async with session_factory() as session:
        repository = repository_factory(session)
        if request.conversation_id is None:
            conversation = await repository.create_conversation(title=DEFAULT_CONVERSATION_TITLE)
        else:
            conversation = await repository.get_conversation(request.conversation_id)
            if conversation is None:
                raise ConversationNotFoundError("conversation not found")
        await session.commit()
    return conversation


async def prepare_chat_context(
    *,
    conversation: Any,
    message: str,
    settings: Any,
    session_factory: Any,
    repository_factory: Callable[[Any], Any],
) -> PreparedChatContext:
    """持久化当前用户消息，并构建发送给模型的上下文消息。

    同时决定当前轮是否需要异步生成标题。
    """
    async with session_factory() as session:
        repository = repository_factory(session)
        current_conversation = await repository.get_conversation(conversation.id)
        if current_conversation is None:
            raise ConversationNotFoundError("conversation not found")

        user_message = await repository.add_message(
            current_conversation,
            role="user",
            content=message,
        )
        recent_messages = await repository.list_recent_messages(
            current_conversation.id,
            limit=settings.memory_window_size,
            before_sequence=user_message.sequence,
        )
        await session.commit()

    llm_messages = build_context_messages(
        system_prompt=current_conversation.system_prompt,
        summary=current_conversation.summary,
        recent_messages=[
            ChatMessage(
                role=item.role,
                content=item.content,
            )
            for item in recent_messages
        ],
        current_message=message,
    )
    return PreparedChatContext(
        conversation=current_conversation,
        conversation_title_at_stream_start=current_conversation.title,
        llm_messages=llm_messages,
        should_generate_title=not recent_messages and current_conversation.title == DEFAULT_CONVERSATION_TITLE,
    )
