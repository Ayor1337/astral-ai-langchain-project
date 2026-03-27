from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from app.core.config import ConfigurationError
from app.llm.exceptions import UpstreamServiceError
from app.schemas.trace import TraceStep
from app.services.conversation_service import DEFAULT_CONVERSATION_TITLE
from app.services.exceptions import ConversationNotFoundError

logger = logging.getLogger(__name__)

_BACKGROUND_TASKS: set[asyncio.Task[None]] = set()


def spawn_background_task(task: asyncio.Task[None]) -> None:
    """跟踪后台收尾任务，避免未引用任务在运行中丢失。

    把任务加入全局集合，直到任务结束后自动移除。
    """
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)


async def persist_chat_completion(
    *,
    session_factory: Any,
    repository_factory: Callable[[Any], Any],
    conversation_id: Any,
    assistant_content: str,
    trace_steps: list[TraceStep] | None,
    refresh_summary_fn: Callable[[Any, Any], Any],
) -> None:
    """在后台持久化 assistant 消息、trace 和 summary。

    持久化失败只记录日志，不影响前端流式响应。
    """
    if not assistant_content:
        return

    try:
        async with session_factory() as session:
            repository = repository_factory(session)
            current_conversation = await repository.get_conversation(conversation_id)
            if current_conversation is None:
                raise ConversationNotFoundError("conversation not found")

            await repository.add_message(
                current_conversation,
                role="assistant",
                content=assistant_content,
                trace_steps=trace_steps,
            )
            try:
                await refresh_summary_fn(repository, current_conversation)
            except Exception:
                logger.exception("Failed to refresh conversation summary")

            await session.commit()
    except ConversationNotFoundError:
        logger.warning("Conversation disappeared before chat completion persistence finished")
    except Exception:
        logger.exception("Failed to persist chat completion in background")


async def persist_conversation_title_if_default(
    *,
    session_factory: Any,
    repository_factory: Callable[[Any], Any],
    conversation_id: Any,
    generated_title: str,
) -> None:
    """仅当标题仍为默认值时补写异步生成出的会话标题。

    这样不会覆盖用户后续手动修改的标题。
    """
    try:
        async with session_factory() as session:
            repository = repository_factory(session)
            current_conversation = await repository.get_conversation(conversation_id)
            if current_conversation is None:
                raise ConversationNotFoundError("conversation not found")
            # 这里允许“晚到的标题”补写，但不覆盖用户后续手动改名。
            if current_conversation.title != DEFAULT_CONVERSATION_TITLE:
                return
            await repository.update_title(current_conversation, generated_title)
            await session.commit()
    except ConversationNotFoundError:
        logger.warning("Conversation disappeared before deferred title persistence finished")
    except Exception:
        logger.exception("Failed to persist deferred conversation title")


async def persist_generated_conversation_title(
    *,
    session_factory: Any,
    repository_factory: Callable[[Any], Any],
    conversation_id: Any,
    title_task: asyncio.Task[str],
) -> None:
    """在后台等待标题生成完成，并在成功后补写默认标题。

    标题生成失败会降级为日志告警。
    """
    try:
        generated_title = await title_task
    except (ConfigurationError, UpstreamServiceError):
        logger.warning("Failed to generate conversation title", exc_info=True)
        return
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Failed to generate conversation title")
        return

    await persist_conversation_title_if_default(
        session_factory=session_factory,
        repository_factory=repository_factory,
        conversation_id=conversation_id,
        generated_title=generated_title,
    )
