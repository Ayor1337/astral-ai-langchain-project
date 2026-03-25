from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
import logging
from typing import Any, TypeAlias

from app.core.config import ConfigurationError, get_settings
from app.db.session import get_session_factory
from app.llm.agents.chat import build_chat_stream, validate_chat_capabilities
from app.llm.agents.titile import generate_conversation_title
from app.llm.exceptions import UpstreamServiceError
from app.repositories.conversations import ConversationRepository
from app.schemas.chat import ChatRequest
from app.schemas.trace import TraceStep
from app.services.chat.conversation_flow import get_or_create_conversation, prepare_chat_context
from app.services.chat.persistence import (
    persist_chat_completion,
    persist_generated_conversation_title,
    spawn_background_task,
)
from app.services.chat.trace import (
    build_trace_step_from_block,
    coerce_stream_block,
    ensure_local_thinking_step_id,
    finalize_thinking_step,
    merge_trace_step,
    serialize_trace_steps,
)
from app.services.chat_runs import finish_chat_run, register_chat_run
from app.services.memory_service import refresh_summary_if_needed

logger = logging.getLogger(__name__)

ChatEvent: TypeAlias = tuple[str, dict[str, Any]]


async def stream_chat_events(request: ChatRequest) -> AsyncIterator[ChatEvent]:
    """协调会话读写、模型流式输出、停止控制和最终持久化。"""
    settings = get_settings()
    session_factory = get_session_factory()
    use_trace = request.thinking_enabled

    if use_trace:
        validate_chat_capabilities(
            endpoint=settings.chat_endpoint,
            thinking_enabled=True,
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
        context.llm_messages,
        thinking_enabled=request.thinking_enabled,
    )

    async def iterator() -> AsyncIterator[ChatEvent]:
        """把模型块、标题任务和停止信号编排成统一 SSE 事件流。"""
        stopped = False
        stream_closed = False
        try:
            yield (
                "conversation",
                {
                    "conversation_id": str(context.conversation.id),
                    "title": context.conversation_title_at_stream_start,
                    "run_id": str(run_handle.run_id),
                },
            )

            trace_state: dict[str, TraceStep] = {}
            trace_order = 1
            next_thinking_step_index = 0
            assistant_text_chunks: list[str] = []
            active_thinking_step_id: str | None = None
            emitted_tool_end_step_ids: set[str] = set()
            title_event_emitted = False
            pending_chunk_task: asyncio.Task[object] | None = None

            def trace_event(step_update: TraceStep) -> ChatEvent:
                """把增量更新合并到当前 trace 状态后再返回给前端。"""
                merged = merge_trace_step(trace_state, step_update)
                return ("trace_step", merged)

            def take_tool_end_event(tool_result_step_id: str) -> ChatEvent | None:
                """在 tool_result 之后补一个明确的 tool_end 节点。"""
                nonlocal trace_order
                if not tool_result_step_id or tool_result_step_id in emitted_tool_end_step_ids:
                    return None
                emitted_tool_end_step_ids.add(tool_result_step_id)
                event = trace_event(
                    {
                        "step_id": f"tool-end-{tool_result_step_id}",
                        "parent_step_id": tool_result_step_id,
                        "type": "tool_end",
                        "status": "success",
                        "message": "工具阶段结束。",
                        "order": trace_order,
                    }
                )
                trace_order += 1
                return event

            def take_ready_title_event() -> ChatEvent | None:
                """标题一旦就绪就立即向前端发出，不必等正文流结束。"""
                nonlocal title_event_emitted
                if title_task is None or title_event_emitted or not title_task.done():
                    return None
                title_event_emitted = True
                try:
                    generated_title = title_task.result()
                except (ConfigurationError, UpstreamServiceError, asyncio.CancelledError):
                    return None
                except Exception:
                    return None
                return (
                    "conversation_title",
                    {
                        "conversation_id": str(context.conversation.id),
                        "title": generated_title,
                    },
                )

            async def close_stream() -> None:
                """确保底层模型流只被关闭一次，避免 finally 中重复报错。"""
                nonlocal stream_closed, pending_chunk_task
                if stream_closed:
                    return
                stream_closed = True
                if pending_chunk_task is not None:
                    pending_chunk_task.cancel()
                    with suppress(asyncio.CancelledError, StopAsyncIteration):
                        await pending_chunk_task
                    pending_chunk_task = None
                close_stream_fn = getattr(stream, "aclose", None)
                if callable(close_stream_fn):
                    with suppress(Exception):
                        await close_stream_fn()

            async def next_item_or_stop() -> tuple[str, object | None]:
                """让标题任务、正文块和停止信号一起竞速，避免标题阻塞正文。"""
                nonlocal stopped, pending_chunk_task
                if run_handle.stop_event.is_set():
                    stopped = True
                    return ("stop", None)
                if pending_chunk_task is None:
                    pending_chunk_task = asyncio.create_task(anext(stream))
                stop_task = asyncio.create_task(run_handle.stop_event.wait())
                wait_tasks: set[asyncio.Task[object] | asyncio.Task[bool] | asyncio.Task[str]] = {
                    pending_chunk_task,
                    stop_task,
                }
                if title_task is not None and not title_event_emitted:
                    wait_tasks.add(title_task)
                done, pending = await asyncio.wait(
                    wait_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for pending_task in pending:
                    if pending_task is stop_task:
                        pending_task.cancel()
                for pending_task in pending:
                    if pending_task is stop_task:
                        with suppress(asyncio.CancelledError):
                            await pending_task
                if stop_task in done:
                    stopped = True
                    if pending_chunk_task is not None:
                        pending_chunk_task.cancel()
                        with suppress(asyncio.CancelledError, StopAsyncIteration):
                            await pending_chunk_task
                        pending_chunk_task = None
                    return ("stop", None)
                if title_task is not None and title_task in done:
                    return ("title", None)
                try:
                    chunk = pending_chunk_task.result()
                except StopAsyncIteration:
                    pending_chunk_task = None
                    return ("eof", None)
                pending_chunk_task = None
                return ("chunk", chunk)

            try:
                ready_title_event = take_ready_title_event()
                if ready_title_event is not None:
                    yield ready_title_event

                while True:
                    item_type, chunk = await next_item_or_stop()
                    if item_type == "stop":
                        break
                    if item_type == "title":
                        ready_title_event = take_ready_title_event()
                        if ready_title_event is not None:
                            yield ready_title_event
                        continue
                    if item_type == "eof":
                        break

                    block = coerce_stream_block(chunk)
                    if block is None:
                        continue
                    block_type = block.get("type")
                    if block_type == "text":
                        if use_trace:
                            finalized_thinking = finalize_thinking_step(
                                trace_state,
                                active_thinking_step_id,
                            )
                            if finalized_thinking is not None:
                                active_thinking_step_id = None
                                yield trace_event(finalized_thinking)
                        text = block.get("text")
                        if isinstance(text, str) and text:
                            assistant_text_chunks.append(text)
                            yield ("chunk", {"content": text})
                    elif use_trace:
                        block, next_thinking_step_index = ensure_local_thinking_step_id(
                            block,
                            active_thinking_step_id=active_thinking_step_id,
                            next_thinking_step_index=next_thinking_step_index,
                        )
                        trace_payload = build_trace_step_from_block(
                            block,
                            fallback_order=trace_order,
                        )
                        existing_step = trace_state.get(str(trace_payload.get("step_id", "")))

                        if block_type != "thinking":
                            finalized_thinking = finalize_thinking_step(
                                trace_state,
                                active_thinking_step_id,
                            )
                            if finalized_thinking is not None:
                                active_thinking_step_id = None
                                yield trace_event(finalized_thinking)

                        if existing_step is not None and "order" not in block:
                            existing_order = existing_step.get("order")
                            if isinstance(existing_order, int):
                                trace_payload["order"] = existing_order
                        elif "order" not in block:
                            trace_order += 1
                        else:
                            trace_order = max(trace_order, int(trace_payload.get("order", trace_order)) + 1)

                        if block_type == "thinking":
                            active_thinking_step_id = str(trace_payload.get("step_id", "")) or None

                        yield trace_event(trace_payload)
                        if block_type == "tool_result":
                            tool_result_step_id = str(trace_payload.get("step_id", ""))
                            tool_end_event = take_tool_end_event(tool_result_step_id)
                            if tool_end_event is not None:
                                yield tool_end_event

                if use_trace and not stopped:
                    finalized_thinking = finalize_thinking_step(trace_state, active_thinking_step_id)
                    if finalized_thinking is not None:
                        yield trace_event(finalized_thinking)

                assistant_content = "".join(assistant_text_chunks)
                serialized_trace_steps = (
                    serialize_trace_steps(
                        trace_state,
                        finalize_running=not stopped,
                    )
                    if use_trace and assistant_content
                    else None
                )

                if assistant_content:
                    spawn_background_task(
                        asyncio.create_task(
                            persist_chat_completion(
                                session_factory=session_factory,
                                repository_factory=ConversationRepository,
                                conversation_id=context.conversation.id,
                                assistant_content=assistant_content,
                                trace_steps=serialized_trace_steps,
                                refresh_summary_fn=refresh_summary_if_needed,
                            )
                        )
                    )
                    await asyncio.sleep(0)

                ready_title_event = take_ready_title_event()
                if ready_title_event is not None:
                    yield ready_title_event

                if use_trace:
                    yield ("trace_done", {"status": "stopped" if stopped else "completed"})

                yield (
                    "done",
                    {
                        "status": "stopped" if stopped else "completed",
                        "run_id": str(run_handle.run_id),
                    },
                )
            except UpstreamServiceError as exc:
                logger.exception("Chat stream interrupted by upstream error")
                yield ("error", {"detail": str(exc)})
                return
            except Exception:
                logger.exception("Chat stream interrupted by unexpected error")
                yield ("error", {"detail": "internal server error"})
                return
            finally:
                await close_stream()
        finally:
            finish_chat_run(run_handle.run_id)

    return iterator()


__all__ = ["ChatEvent", "stream_chat_events"]
