from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from typing import Any, TypeAlias

from app.core.config import ConfigurationError
from app.llm.exceptions import UpstreamServiceError
from app.schemas.trace import TraceStep
from app.services.chat.persistence import spawn_background_task
from app.services.chat.trace import (
    build_trace_step_from_block,
    coerce_stream_block,
    ensure_local_thinking_step_id,
    finalize_thinking_step,
    merge_trace_step,
    serialize_trace_steps,
)
from app.services.chat_runs import finish_chat_run

logger = logging.getLogger(__name__)

ChatEvent: TypeAlias = tuple[str, dict[str, Any]]


async def build_stream_event_iterator(
    *,
    conversation: Any,
    conversation_title_at_stream_start: str,
    stream: AsyncIterator[object],
    run_handle: Any,
    use_trace: bool,
    title_task: asyncio.Task[str] | None,
    session_factory: Any,
    repository_factory: Callable[[Any], Any],
    refresh_summary_fn: Callable[[Any, Any], Any],
    persist_chat_completion_fn: Callable[..., Any],
) -> AsyncIterator[ChatEvent]:
    """把模型块、标题任务和停止信号编排成统一 SSE 事件流。

    统一处理流式块、标题回填、停止控制和后台持久化。
    """

    async def iterator() -> AsyncIterator[ChatEvent]:
        """将流式模型输出转换为 SSE 事件序列。

        负责拉取块、维护 trace、处理停止信号和后台持久化。
        """
        stopped = False
        stream_closed = False
        try:
            yield (
                "conversation",
                {
                    "conversation_id": str(conversation.id),
                    "title": conversation_title_at_stream_start,
                    "run_id": str(run_handle.run_id),
                },
            )

            trace_state: dict[str, TraceStep] = {}
            trace_order = 1
            next_thinking_step_index = 0
            assistant_text_chunks: list[str] = []
            collected_sources: list[dict[str, str]] = []
            seen_source_urls: set[str] = set()
            active_thinking_step_id: str | None = None
            title_event_emitted = False
            pending_chunk_task: asyncio.Task[object] | None = None

            def trace_event(step_update: TraceStep) -> ChatEvent:
                """把 trace 更新合并到本地状态并产出事件。

                合并后返回统一的 `trace_step` SSE 事件。
                """
                merged = merge_trace_step(trace_state, step_update)
                return ("trace_step", merged)

            def take_tool_end_event(tool_result_step_id: str) -> ChatEvent | None:
                """为工具执行结果补一个结束事件。

                用于把 `tool_result` 和 `tool_end` 事件串成完整阶段。
                """
                nonlocal trace_order
                if not tool_result_step_id:
                    return None
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
                """在标题任务完成后产出会话标题事件。

                如果标题生成失败或尚未完成，则返回 `None`。
                """
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
                        "conversation_id": str(conversation.id),
                        "title": generated_title,
                    },
                )

            async def close_stream() -> None:
                """关闭底层流并清理挂起的 chunk 任务。

                确保异常和正常结束都不会泄漏异步任务。
                """
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
                """等待下一块流式数据、标题完成或停止信号。

                按优先级把不同异步来源收敛成统一状态。
                """
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
                    if block_type == "search":
                        _collect_search_sources(
                            block,
                            collected_sources=collected_sources,
                            seen_source_urls=seen_source_urls,
                        )
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
                    if use_trace and trace_state and assistant_content
                    else None
                )

                if assistant_content:
                    spawn_background_task(
                        asyncio.create_task(
                            persist_chat_completion_fn(
                                session_factory=session_factory,
                                repository_factory=repository_factory,
                                conversation_id=conversation.id,
                                assistant_content=assistant_content,
                                trace_steps=serialized_trace_steps,
                                refresh_summary_fn=refresh_summary_fn,
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
                        "sources": _finalize_sources(
                            collected_sources=collected_sources,
                            assistant_content=assistant_content,
                        ),
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


def _collect_search_sources(
    block: dict[str, object],
    *,
    collected_sources: list[dict[str, str]],
    seen_source_urls: set[str],
) -> None:
    """从搜索块中收集可公开展示的来源列表。

    只保留成功结果、去重链接和有效文本字段。
    """
    if block.get("status") != "success":
        return
    payload = block.get("payload")
    if not isinstance(payload, dict):
        return
    results = payload.get("results")
    if not isinstance(results, list):
        return
    for item in results:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        url = item.get("url")
        snippet = item.get("snippet")
        if not isinstance(title, str) or not title.strip():
            continue
        if not isinstance(url, str) or not url.strip() or url in seen_source_urls:
            continue
        if not isinstance(snippet, str):
            snippet = ""
        seen_source_urls.add(url)
        collected_sources.append(
            {
                "title": title.strip(),
                "url": url.strip(),
                "snippet": snippet.strip(),
            }
        )


def _finalize_sources(
    *,
    collected_sources: list[dict[str, str]],
    assistant_content: str,
) -> list[dict[str, object]]:
    """根据引用标记裁剪最终来源列表。

    如果正文没有引用，则保留全部来源。
    """
    indexed_sources = [
        {
            "index": index,
            **source,
        }
        for index, source in enumerate(collected_sources, start=1)
    ]
    if not indexed_sources:
        return []

    citation_indexes = _extract_citation_indexes(assistant_content, limit=len(indexed_sources))
    if not citation_indexes:
        return indexed_sources
    return [indexed_sources[index - 1] for index in citation_indexes]


def _extract_citation_indexes(text: str, *, limit: int) -> list[int]:
    """从文本中提取去重后的引用编号。

    只返回落在有效范围内的引用顺序。
    """
    indexes: list[int] = []
    seen: set[int] = set()
    for match in re.finditer(r"\[(\d+)\]", text):
        index = int(match.group(1))
        if index < 1 or index > limit or index in seen:
            continue
        seen.add(index)
        indexes.append(index)
    return indexes
