import asyncio
from contextlib import suppress
import logging
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any, TypeAlias

from app.core.config import ConfigurationError, get_settings
from app.db.session import get_session_factory
from app.llm.agents.chat import build_chat_stream, validate_chat_capabilities
from app.llm.agents.titile import generate_conversation_title
from app.llm.exceptions import UpstreamServiceError
from app.repositories.conversations import ConversationRepository
from app.schemas.chat import ChatMessage, ChatRequest
from app.schemas.trace import TraceStep
from app.services.chat_runs import finish_chat_run, register_chat_run
from app.services.conversation_service import DEFAULT_CONVERSATION_TITLE
from app.services.exceptions import ConversationNotFoundError
from app.services.memory_service import build_context_messages, refresh_summary_if_needed

logger = logging.getLogger(__name__)

ChatEvent: TypeAlias = tuple[str, dict[str, Any]]
TRACE_BLOCK_TYPES = {"thinking", "search", "fetch", "tool_call", "tool_result", "retry", "other"}
TRACE_STEP_STATUSES = {"pending", "running", "success", "error", "skipped"}
TITLE_GENERATION_TIMEOUT_SECONDS = 3.0
_BACKGROUND_TASKS: set[asyncio.Task[None]] = set()


def _utcnow_iso() -> str:
    """为 trace 事件统一生成 ISO 格式 UTC 时间戳。"""
    return datetime.now(timezone.utc).isoformat()


def _merge_trace_step(
    trace_state: dict[str, TraceStep],
    step_update: TraceStep,
) -> TraceStep:
    """按 step_id 合并增量 trace，保留流式更新的最后状态。"""
    step_id = str(step_update.get("step_id", ""))
    if not step_id:
        return step_update
    merged = dict(trace_state.get(step_id, {}))
    merged.update(step_update)
    if merged.get("type") == "thinking":
        previous_thinking = trace_state.get(step_id, {}).get("thinking")
        current_thinking = step_update.get("thinking")
        if isinstance(previous_thinking, str) and isinstance(current_thinking, str):
            if current_thinking == previous_thinking:
                merged["thinking"] = previous_thinking
            elif current_thinking.startswith(previous_thinking):
                merged["thinking"] = current_thinking
            elif previous_thinking.endswith(current_thinking):
                merged["thinking"] = previous_thinking
            else:
                merged["thinking"] = previous_thinking + current_thinking
    trace_state[step_id] = merged
    return merged


def _coerce_stream_block(chunk: object) -> dict[str, object] | None:
    """兼容字符串块和结构化块，统一为字典格式继续处理。"""
    if isinstance(chunk, str):
        if not chunk:
            return None
        return {"type": "text", "text": chunk, "index": 0}
    if isinstance(chunk, dict):
        return dict(chunk)
    return None


def _normalize_trace_status(raw_status: object) -> str:
    """将未知状态收敛为 running，避免前端收到不稳定枚举。"""
    if isinstance(raw_status, str) and raw_status in TRACE_STEP_STATUSES:
        return raw_status
    return "running"


def _resolve_trace_step_id(block: dict[str, object], *, step_type: str, fallback_order: int) -> str:
    """尽量复用上游 step_id；缺失时按类型和顺序生成稳定降级 ID。"""
    raw_step_id = block.get("step_id")
    if raw_step_id:
        return str(raw_step_id).strip()
    if step_type == "thinking" and isinstance(block.get("index"), int):
        return f"thinking-{int(block['index'])}"
    return f"{step_type}-{fallback_order}"


def _ensure_local_thinking_step_id(
    block: dict[str, object],
    *,
    active_thinking_step_id: str | None,
    next_thinking_step_index: int,
) -> tuple[dict[str, object], int]:
    """为没有上游 step_id 的 thinking 块分配本地递增 ID。"""
    if block.get("type") != "thinking" or block.get("step_id"):
        return block, next_thinking_step_index

    enriched_block = dict(block)
    if active_thinking_step_id:
        enriched_block["step_id"] = active_thinking_step_id
        return enriched_block, next_thinking_step_index

    enriched_block["step_id"] = f"thinking-{next_thinking_step_index}"
    return enriched_block, next_thinking_step_index + 1


def _build_trace_step_from_block(
    block: dict[str, object],
    *,
    fallback_order: int,
) -> TraceStep:
    """把 provider 返回的块标准化为统一 trace_step 结构。"""
    raw_type = block.get("type")
    step_type = str(raw_type) if isinstance(raw_type, str) else "other"
    if step_type not in TRACE_BLOCK_TYPES:
        step_type = "other"

    trace_step: TraceStep = {
        "step_id": _resolve_trace_step_id(block, step_type=step_type, fallback_order=fallback_order),
        "type": step_type,
        "status": _normalize_trace_status(block.get("status")),
        "timestamp": str(block.get("timestamp") or _utcnow_iso()),
        "order": int(block["order"]) if isinstance(block.get("order"), int) else fallback_order,
    }

    for field_name in (
        "title",
        "message",
        "url",
        "query",
        "error_code",
        "error_message",
        "tool_name",
        "input_json",
        "output_json",
        "retry_of",
        "parent_step_id",
        "kind",
        "thinking",
        "signature",
    ):
        field_value = block.get(field_name)
        if isinstance(field_value, str) and field_value:
            trace_step[field_name] = field_value

    index = block.get("index")
    if isinstance(index, int):
        trace_step["index"] = index

    for int_field in ("result_count", "duration_ms"):
        field_value = block.get(int_field)
        if isinstance(field_value, int):
            trace_step[int_field] = field_value

    payload = block.get("payload")
    if isinstance(payload, dict):
        trace_step["payload"] = payload

    return trace_step


def _serialize_trace_steps(
    trace_state: dict[str, TraceStep],
    *,
    finalize_running: bool,
) -> list[TraceStep] | None:
    """在落库前按顺序序列化 trace，并可将残留 running 步骤收口为 success。"""
    if not trace_state:
        return None

    steps: list[TraceStep] = []
    for step in trace_state.values():
        serialized = dict(step)
        if finalize_running and serialized.get("status") == "running":
            serialized["status"] = "success"
        steps.append(serialized)

    return sorted(
        steps,
        key=lambda step: (
            int(step.get("order", 0)) if isinstance(step.get("order"), int) else 0,
            str(step.get("timestamp", "")),
            str(step.get("step_id", "")),
        ),
    )


def _finalize_thinking_step(
    trace_state: dict[str, TraceStep],
    active_thinking_step_id: str | None,
) -> TraceStep | None:
    """在文本输出开始或结束时关闭当前 thinking 步骤，避免悬挂状态。"""
    if not active_thinking_step_id:
        return None
    current_step = trace_state.get(active_thinking_step_id)
    if current_step is None or current_step.get("type") != "thinking":
        return None
    if current_step.get("status") != "running":
        return None
    return {
        **current_step,
        "status": "success",
        "timestamp": _utcnow_iso(),
    }


def _spawn_background_task(task: asyncio.Task[None]) -> None:
    """跟踪后台收尾任务，避免未引用任务在运行中丢失。"""
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)


async def _persist_chat_completion(
    *,
    session_factory: Any,
    conversation_id: Any,
    assistant_content: str,
    trace_steps: list[TraceStep] | None,
    generated_title: str | None,
) -> None:
    """在后台持久化 assistant 消息、trace、summary 和标题。"""
    if not assistant_content and not generated_title:
        return

    try:
        async with session_factory() as session:
            repository = ConversationRepository(session)
            current_conversation = await repository.get_conversation(conversation_id)
            if current_conversation is None:
                raise ConversationNotFoundError("conversation not found")

            if assistant_content:
                await repository.add_message(
                    current_conversation,
                    role="assistant",
                    content=assistant_content,
                    trace_steps=trace_steps,
                )
                try:
                    await refresh_summary_if_needed(repository, current_conversation)
                except Exception:
                    logger.exception("Failed to refresh conversation summary")

            if generated_title:
                await repository.update_title(current_conversation, generated_title)

            await session.commit()
    except ConversationNotFoundError:
        logger.warning("Conversation disappeared before chat completion persistence finished")
    except Exception:
        logger.exception("Failed to persist chat completion in background")


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

    async with session_factory() as session:
        repository = ConversationRepository(session)
        if request.conversation_id is None:
            # 兼容旧调用路径：未显式建会话时，首轮消息到来时隐式创建。
            conversation = await repository.create_conversation(title=DEFAULT_CONVERSATION_TITLE)
        else:
            conversation = await repository.get_conversation(request.conversation_id)
            if conversation is None:
                raise ConversationNotFoundError("conversation not found")
        await session.commit()
    run_handle = register_chat_run(conversation.id)

    async with session_factory() as session:
        repository = ConversationRepository(session)
        current_conversation = await repository.get_conversation(conversation.id)
        if current_conversation is None:
            raise ConversationNotFoundError("conversation not found")
        user_message = await repository.add_message(
            current_conversation,
            role="user",
            content=request.message,
        )
        recent_messages = await repository.list_recent_messages(
            current_conversation.id,
            limit=settings.memory_window_size,
            before_sequence=user_message.sequence,
        )
        await session.commit()
    should_generate_title = not recent_messages and conversation.title == DEFAULT_CONVERSATION_TITLE

    llm_messages = build_context_messages(
        system_prompt=conversation.system_prompt,
        summary=conversation.summary,
        recent_messages=[
            ChatMessage(
                role=message.role,
                content=message.content,
            )
            for message in recent_messages
        ],
        current_message=request.message,
    )
    stream = await build_chat_stream(
        llm_messages,
        thinking_enabled=request.thinking_enabled,
    )

    async def iterator() -> AsyncIterator[ChatEvent]:
        stopped = False
        stream_closed = False
        try:
            yield (
                "conversation",
                {
                    "conversation_id": str(conversation.id),
                    "title": conversation.title,
                    "run_id": str(run_handle.run_id),
                },
            )

            trace_state: dict[str, TraceStep] = {}
            trace_order = 1
            next_thinking_step_index = 0
            assistant_text_chunks: list[str] = []
            active_thinking_step_id: str | None = None

            def _trace_event(step_update: TraceStep) -> ChatEvent:
                """把增量更新合并到当前 trace 状态后再返回给前端。"""
                merged = _merge_trace_step(trace_state, step_update)
                return ("trace_step", merged)

            async def _close_stream() -> None:
                """确保底层模型流只被关闭一次，避免 finally 中重复报错。"""
                nonlocal stream_closed
                if stream_closed:
                    return
                stream_closed = True
                close_stream = getattr(stream, "aclose", None)
                if callable(close_stream):
                    with suppress(Exception):
                        await close_stream()

            async def _next_chunk_or_stop() -> object | None:
                """在下一块输出和停止信号之间竞速，优先响应停止请求。"""
                nonlocal stopped
                if run_handle.stop_event.is_set():
                    return None
                chunk_task = asyncio.create_task(anext(stream))
                stop_task = asyncio.create_task(run_handle.stop_event.wait())
                done, pending = await asyncio.wait(
                    {chunk_task, stop_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for pending_task in pending:
                    pending_task.cancel()
                for pending_task in pending:
                    with suppress(asyncio.CancelledError):
                        await pending_task
                if stop_task in done:
                    stopped = True
                    chunk_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await chunk_task
                    return None
                with suppress(asyncio.CancelledError):
                    await stop_task
                try:
                    return chunk_task.result()
                except StopAsyncIteration:
                    return None

            try:
                while True:
                    chunk = await _next_chunk_or_stop()
                    if chunk is None:
                        if run_handle.stop_event.is_set():
                            stopped = True
                        break
                    block = _coerce_stream_block(chunk)
                    if block is None:
                        continue
                    block_type = block.get("type")
                    if block_type == "text":
                        if use_trace:
                            # 文本开始输出后，前一个 thinking 步骤应当立即收口。
                            finalized_thinking = _finalize_thinking_step(trace_state, active_thinking_step_id)
                            if finalized_thinking is not None:
                                active_thinking_step_id = None
                                yield _trace_event(finalized_thinking)
                        text = block.get("text")
                        if isinstance(text, str) and text:
                            assistant_text_chunks.append(text)
                            yield ("chunk", {"content": text})
                    elif use_trace:
                        block, next_thinking_step_index = _ensure_local_thinking_step_id(
                            block,
                            active_thinking_step_id=active_thinking_step_id,
                            next_thinking_step_index=next_thinking_step_index,
                        )
                        trace_payload = _build_trace_step_from_block(block, fallback_order=trace_order)
                        existing_step = trace_state.get(str(trace_payload.get("step_id", "")))

                        if block_type != "thinking":
                            finalized_thinking = _finalize_thinking_step(trace_state, active_thinking_step_id)
                            if finalized_thinking is not None:
                                active_thinking_step_id = None
                                yield _trace_event(finalized_thinking)

                        if existing_step is not None and "order" not in block:
                            existing_order = existing_step.get("order")
                            if isinstance(existing_order, int):
                                trace_payload["order"] = existing_order
                        elif "order" not in block:
                            # 未显式提供顺序时，按首次出现顺序递增，保持前端展示稳定。
                            trace_order += 1
                        else:
                            trace_order = max(trace_order, int(trace_payload.get("order", trace_order)) + 1)

                        if block_type == "thinking":
                            active_thinking_step_id = str(trace_payload.get("step_id", "")) or None

                        yield _trace_event(trace_payload)

                if use_trace and not stopped:
                    finalized_thinking = _finalize_thinking_step(trace_state, active_thinking_step_id)
                    if finalized_thinking is not None:
                        active_thinking_step_id = None
                        yield _trace_event(finalized_thinking)

                assistant_content = "".join(assistant_text_chunks)
                serialized_trace_steps = (
                    _serialize_trace_steps(
                        trace_state,
                        finalize_running=not stopped,
                    )
                    if use_trace and assistant_content
                    else None
                )
                generated_title: str | None = None

                if assistant_content and should_generate_title and not stopped:
                    try:
                        generated_title = await asyncio.wait_for(
                            generate_conversation_title(
                                user_message=request.message,
                                assistant_message=assistant_content,
                            ),
                            timeout=TITLE_GENERATION_TIMEOUT_SECONDS,
                        )
                    except (ConfigurationError, UpstreamServiceError, asyncio.TimeoutError):
                        logger.warning("Failed to generate conversation title", exc_info=True)

                if assistant_content or generated_title:
                    _spawn_background_task(
                        asyncio.create_task(
                            _persist_chat_completion(
                                session_factory=session_factory,
                                conversation_id=conversation.id,
                                assistant_content=assistant_content,
                                trace_steps=serialized_trace_steps,
                                generated_title=generated_title,
                            )
                        )
                    )
                    # 让后台任务先获得一次调度机会；不等待其完成。
                    await asyncio.sleep(0)

                if generated_title:
                    yield (
                        "conversation_title",
                        {
                            "conversation_id": str(conversation.id),
                            "title": generated_title,
                        },
                    )

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
                await _close_stream()
        finally:
            finish_chat_run(run_handle.run_id)

    return iterator()
