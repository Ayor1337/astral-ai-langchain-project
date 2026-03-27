from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from app.core.config import get_settings
from app.db.session import get_session_factory
from app.llm.agents.chat import build_chat_stream
from app.llm.agents.titile import generate_conversation_title
from app.llm.capabilities import validate_chat_capabilities
from app.repositories.conversations import ConversationRepository
from app.schemas.chat import ChatRequest
from app.services.chat.conversation_flow import get_or_create_conversation, prepare_chat_context
from app.services.chat.persistence import (
    persist_chat_completion,
    persist_generated_conversation_title,
    spawn_background_task,
)
from app.services.chat.stream_loop import ChatEvent, build_stream_event_iterator
from app.services.chat_runs import register_chat_run
from app.services.memory_service import refresh_summary_if_needed

SEARCH_SYSTEM_PROMPT = (
    "当且仅当问题需要最新信息、新闻、事实核验或明显依赖联网信息时，才调用 web_search。"
    "如果搜索无结果或搜索失败，请直接基于已有上下文回答，且不要编造来源。"
)


async def stream_chat_events(request: ChatRequest) -> AsyncIterator[ChatEvent]:
    """协调会话读写、模型流式输出、停止控制和最终持久化。

    这是聊天接口的主编排入口，负责把所有异步环节串起来。
    """
    settings = get_settings()
    session_factory = get_session_factory()

    validate_chat_capabilities(
        endpoint=settings.chat_endpoint,
        thinking_enabled=request.thinking_enabled,
        search_enabled=request.search_enabled,
        search=settings.search,
    )

    conversation = await get_or_create_conversation(
        request,
        session_factory=session_factory,
        repository_factory=ConversationRepository,
    )
    run_handle = register_chat_run(conversation.id)
    context = await prepare_chat_context(
        conversation=conversation,
        message=request.message,
        settings=settings,
        session_factory=session_factory,
        repository_factory=ConversationRepository,
    )

    title_task: asyncio.Task[str] | None = None
    if context.should_generate_title:
        title_task = asyncio.create_task(
            generate_conversation_title(
                user_message=request.message,
            )
        )
        spawn_background_task(
            asyncio.create_task(
                persist_generated_conversation_title(
                    session_factory=session_factory,
                    repository_factory=ConversationRepository,
                    conversation_id=context.conversation.id,
                    title_task=title_task,
                )
            )
        )

    stream = await build_chat_stream(
        _build_llm_messages(context.llm_messages, search_enabled=request.search_enabled),
        endpoint=settings.chat_endpoint,
        thinking_enabled=request.thinking_enabled,
        **({"search_enabled": True, "search": settings.search} if request.search_enabled else {}),
    )

    return await build_stream_event_iterator(
        conversation=context.conversation,
        conversation_title_at_stream_start=context.conversation_title_at_stream_start,
        stream=stream,
        run_handle=run_handle,
        use_trace=request.thinking_enabled,
        title_task=title_task,
        session_factory=session_factory,
        repository_factory=ConversationRepository,
        refresh_summary_fn=refresh_summary_if_needed,
        persist_chat_completion_fn=persist_chat_completion,
    )


__all__ = ["ChatEvent", "stream_chat_events"]


def _build_llm_messages(messages, *, search_enabled: bool):
    """按需为模型消息前置搜索系统提示。

    启用联网搜索时，把搜索约束作为第一条 system 消息注入。
    """
    if not search_enabled:
        return messages
    from app.schemas.chat import ChatMessage

    return [ChatMessage(role="system", content=SEARCH_SYSTEM_PROMPT), *messages]
