from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from dataclasses import dataclass, field
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
_SignalResult: TypeAlias = tuple[str, object | None]


@dataclass
class _StreamLoopState:
    """保存流式编排过程中的可变状态。"""

    stopped: bool = False
    stream_closed: bool = False
    trace_state: dict[str, TraceStep] = field(default_factory=dict)
    trace_order: int = 1
    next_thinking_step_index: int = 0
    assistant_text_chunks: list[str] = field(default_factory=list)
    collected_sources: list[dict[str, str]] = field(default_factory=list)
    seen_source_urls: set[str] = field(default_factory=set)
    active_thinking_step_id: str | None = None
    title_event_emitted: bool = False
    pending_chunk_task: asyncio.Task[object] | None = None


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
        state = _StreamLoopState()
        try:
            yield (
                "conversation",
                {
                    "conversation_id": str(conversation.id),
                    "title": conversation_title_at_stream_start,
                    "run_id": str(run_handle.run_id),
                },
            )

            ready_title_event = _take_ready_title_event(
                state,
                title_task=title_task,
                conversation_id=conversation.id,
            )
            if ready_title_event is not None:
                yield ready_title_event

            while True:
                item_type, chunk = await _wait_next_signal(
                    state,
                    stream=stream,
                    stop_event=run_handle.stop_event,
                    title_task=title_task,
                )
                if item_type == "stop":
                    break
                if item_type == "title":
                    ready_title_event = _take_ready_title_event(
                        state,
                        title_task=title_task,
                        conversation_id=conversation.id,
                    )
                    if ready_title_event is not None:
                        yield ready_title_event
                    continue
                if item_type == "eof":
                    break

                block = coerce_stream_block(chunk)
                if block is None:
                    continue
                if block.get("type") == "search":
                    _collect_search_sources(
                        block,
                        collected_sources=state.collected_sources,
                        seen_source_urls=state.seen_source_urls,
                    )
                if block.get("type") == "text":
                    for event in _handle_text_block(block, state, use_trace=use_trace):
                        yield event
                    continue
                if not use_trace:
                    continue
                for event in _handle_trace_block(block, state):
                    yield event

            for event in await _build_completion_events(
                state,
                conversation_id=conversation.id,
                run_id=run_handle.run_id,
                use_trace=use_trace,
                title_task=title_task,
                session_factory=session_factory,
                repository_factory=repository_factory,
                refresh_summary_fn=refresh_summary_fn,
                persist_chat_completion_fn=persist_chat_completion_fn,
            ):
                yield event
        except UpstreamServiceError as exc:
            logger.exception("Chat stream interrupted by upstream error")
            yield ("error", {"detail": str(exc)})
            return
        except Exception:
            logger.exception("Chat stream interrupted by unexpected error")
            yield ("error", {"detail": "internal server error"})
            return
        finally:
            await _close_stream_resources(state, stream)
            finish_chat_run(run_handle.run_id)

    return iterator()


def _take_ready_title_event(
    state: _StreamLoopState,
    *,
    title_task: asyncio.Task[str] | None,
    conversation_id: Any,
) -> ChatEvent | None:
    """在标题任务完成后产出会话标题事件。

    Args:
        state: 流循环状态。
        title_task: 异步标题生成任务。
        conversation_id: 会话 ID。

    Returns:
        可直接发送的标题事件；若标题尚未就绪或不可用则返回 `None`。
    """

    if title_task is None or state.title_event_emitted or not title_task.done():
        return None
    state.title_event_emitted = True
    try:
        generated_title = title_task.result()
    except (ConfigurationError, UpstreamServiceError, asyncio.CancelledError):
        return None
    except Exception:
        return None
    return (
        "conversation_title",
        {
            "conversation_id": str(conversation_id),
            "title": generated_title,
        },
    )


async def _close_stream_resources(
    state: _StreamLoopState,
    stream: AsyncIterator[object],
) -> None:
    """关闭底层流并清理挂起的异步任务。

    Args:
        state: 流循环状态。
        stream: 底层模型流迭代器。
    """

    if state.stream_closed:
        return
    state.stream_closed = True
    if state.pending_chunk_task is not None:
        state.pending_chunk_task.cancel()
        with suppress(asyncio.CancelledError, StopAsyncIteration):
            await state.pending_chunk_task
        state.pending_chunk_task = None
    close_stream_fn = getattr(stream, "aclose", None)
    if callable(close_stream_fn):
        with suppress(Exception):
            await close_stream_fn()


async def _wait_next_signal(
    state: _StreamLoopState,
    *,
    stream: AsyncIterator[object],
    stop_event: asyncio.Event,
    title_task: asyncio.Task[str] | None,
) -> _SignalResult:
    """等待下一块流式数据、标题完成或停止信号。

    Args:
        state: 流循环状态。
        stream: 底层模型流迭代器。
        stop_event: 外部停止信号。
        title_task: 异步标题生成任务。

    Returns:
        信号类型和附带数据的二元组。
    """

    if stop_event.is_set():
        state.stopped = True
        return ("stop", None)
    if state.pending_chunk_task is None:
        state.pending_chunk_task = asyncio.create_task(anext(stream))

    stop_task = asyncio.create_task(stop_event.wait())
    wait_tasks: set[asyncio.Task[Any]] = {state.pending_chunk_task, stop_task}
    if title_task is not None and not state.title_event_emitted:
        wait_tasks.add(title_task)

    done, pending = await asyncio.wait(wait_tasks, return_when=asyncio.FIRST_COMPLETED)
    for pending_task in pending:
        if pending_task is stop_task:
            pending_task.cancel()
            with suppress(asyncio.CancelledError):
                await pending_task

    if stop_task in done:
        state.stopped = True
        if state.pending_chunk_task is not None:
            state.pending_chunk_task.cancel()
            with suppress(asyncio.CancelledError, StopAsyncIteration):
                await state.pending_chunk_task
            state.pending_chunk_task = None
        return ("stop", None)
    if title_task is not None and title_task in done:
        return ("title", None)

    try:
        chunk = state.pending_chunk_task.result()
    except StopAsyncIteration:
        state.pending_chunk_task = None
        return ("eof", None)

    state.pending_chunk_task = None
    return ("chunk", chunk)


def _apply_trace_step(state: _StreamLoopState, step_update: TraceStep) -> ChatEvent:
    """把 trace 更新合并到本地状态并产出事件。

    Args:
        state: 流循环状态。
        step_update: 待合并的 trace 步骤增量。

    Returns:
        合并后的 `trace_step` 事件。
    """

    merged = merge_trace_step(state.trace_state, step_update)
    return ("trace_step", merged)


def _finalize_active_thinking_if_needed(state: _StreamLoopState) -> ChatEvent | None:
    """在文本开始或流结束前关闭当前 thinking 步骤。

    Args:
        state: 流循环状态。

    Returns:
        关闭 thinking 后的 trace 事件；若无需关闭则返回 `None`。
    """

    finalized = finalize_thinking_step(state.trace_state, state.active_thinking_step_id)
    if finalized is None:
        return None
    state.active_thinking_step_id = None
    return _apply_trace_step(state, finalized)


def _handle_text_block(
    block: dict[str, object],
    state: _StreamLoopState,
    *,
    use_trace: bool,
) -> list[ChatEvent]:
    """处理文本块并按需先收口 thinking 状态。

    Args:
        block: 当前文本块。
        state: 流循环状态。
        use_trace: 是否启用 trace 输出。

    Returns:
        由该文本块产生的事件列表。
    """

    events: list[ChatEvent] = []
    if use_trace:
        finalized_thinking_event = _finalize_active_thinking_if_needed(state)
        if finalized_thinking_event is not None:
            events.append(finalized_thinking_event)

    text = block.get("text")
    if isinstance(text, str) and text:
        state.assistant_text_chunks.append(text)
        events.append(("chunk", {"content": text}))
    return events


def _handle_trace_block(
    block: dict[str, object],
    state: _StreamLoopState,
) -> list[ChatEvent]:
    """处理非文本块的 trace 演进与补充事件。

    Args:
        block: 当前结构化块。
        state: 流循环状态。

    Returns:
        由该结构化块产生的 trace 相关事件列表。
    """

    events: list[ChatEvent] = []
    block_type = block.get("type")
    block, state.next_thinking_step_index = ensure_local_thinking_step_id(
        block,
        active_thinking_step_id=state.active_thinking_step_id,
        next_thinking_step_index=state.next_thinking_step_index,
    )
    trace_payload = build_trace_step_from_block(block, fallback_order=state.trace_order)
    existing_step = state.trace_state.get(str(trace_payload.get("step_id", "")))

    if block_type != "thinking":
        finalized_thinking_event = _finalize_active_thinking_if_needed(state)
        if finalized_thinking_event is not None:
            events.append(finalized_thinking_event)

    if existing_step is not None and "order" not in block:
        existing_order = existing_step.get("order")
        if isinstance(existing_order, int):
            trace_payload["order"] = existing_order
    elif "order" not in block:
        state.trace_order += 1
    else:
        state.trace_order = max(state.trace_order, int(trace_payload.get("order", state.trace_order)) + 1)

    if block_type == "thinking":
        state.active_thinking_step_id = str(trace_payload.get("step_id", "")) or None

    events.append(_apply_trace_step(state, trace_payload))
    if block_type == "tool_result":
        tool_end_event = _take_tool_end_event(state, str(trace_payload.get("step_id", "")))
        if tool_end_event is not None:
            events.append(tool_end_event)
    return events


def _take_tool_end_event(
    state: _StreamLoopState,
    tool_result_step_id: str,
) -> ChatEvent | None:
    """为工具结果补一个结束事件。

    Args:
        state: 流循环状态。
        tool_result_step_id: 工具结果步骤 ID。

    Returns:
        工具结束事件；当步骤 ID 为空时返回 `None`。
    """

    if not tool_result_step_id:
        return None
    event = _apply_trace_step(
        state,
        {
            "step_id": f"tool-end-{tool_result_step_id}",
            "parent_step_id": tool_result_step_id,
            "type": "tool_end",
            "status": "success",
            "message": "工具阶段结束。",
            "order": state.trace_order,
        },
    )
    state.trace_order += 1
    return event


async def _build_completion_events(
    state: _StreamLoopState,
    *,
    conversation_id: Any,
    run_id: Any,
    use_trace: bool,
    title_task: asyncio.Task[str] | None,
    session_factory: Any,
    repository_factory: Callable[[Any], Any],
    refresh_summary_fn: Callable[[Any, Any], Any],
    persist_chat_completion_fn: Callable[..., Any],
) -> list[ChatEvent]:
    """生成流结束后的收尾事件，并触发后台持久化。

    Args:
        state: 流循环状态。
        conversation_id: 会话 ID。
        run_id: 当前运行 ID。
        use_trace: 是否启用 trace 输出。
        title_task: 异步标题生成任务。
        session_factory: 会话工厂。
        repository_factory: 仓储工厂。
        refresh_summary_fn: 摘要刷新函数。
        persist_chat_completion_fn: 持久化聊天完成结果的函数。

    Returns:
        流结束阶段需要发送的事件列表。
    """

    events: list[ChatEvent] = []
    if use_trace and not state.stopped:
        finalized_thinking_event = _finalize_active_thinking_if_needed(state)
        if finalized_thinking_event is not None:
            events.append(finalized_thinking_event)

    assistant_content = "".join(state.assistant_text_chunks)
    serialized_trace_steps = (
        serialize_trace_steps(
            state.trace_state,
            finalize_running=not state.stopped,
        )
        if use_trace and state.trace_state and assistant_content
        else None
    )

    if assistant_content:
        spawn_background_task(
            asyncio.create_task(
                persist_chat_completion_fn(
                    session_factory=session_factory,
                    repository_factory=repository_factory,
                    conversation_id=conversation_id,
                    assistant_content=assistant_content,
                    trace_steps=serialized_trace_steps,
                    refresh_summary_fn=refresh_summary_fn,
                )
            )
        )
        await asyncio.sleep(0)

    ready_title_event = _take_ready_title_event(
        state,
        title_task=title_task,
        conversation_id=conversation_id,
    )
    if ready_title_event is not None:
        events.append(ready_title_event)

    status = "stopped" if state.stopped else "completed"
    if use_trace:
        events.append(("trace_done", {"status": status}))

    events.append(
        (
            "done",
            {
                "status": status,
                "run_id": str(run_id),
                "sources": _finalize_sources(
                    collected_sources=state.collected_sources,
                    assistant_content=assistant_content,
                ),
            },
        )
    )
    return events


def _collect_search_sources(
    block: dict[str, object],
    *,
    collected_sources: list[dict[str, str]],
    seen_source_urls: set[str],
) -> None:
    """从搜索块中收集可公开展示的来源列表。

    Args:
        block: 当前搜索块。
        collected_sources: 已收集来源列表，会被原地追加。
        seen_source_urls: 已出现 URL 集合，用于去重。
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

    Args:
        collected_sources: 已收集的来源列表。
        assistant_content: 助手最终回复文本。

    Returns:
        按引用顺序过滤后的来源列表；未引用时返回全量来源。
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

    Args:
        text: 需要解析的文本。
        limit: 允许的最大引用编号。

    Returns:
        按出现顺序返回的去重引用编号列表。
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
