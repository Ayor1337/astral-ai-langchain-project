import asyncio
from contextlib import suppress
import logging
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any, TypeAlias

from app.core.config import get_settings
from app.db.session import get_session_factory
from app.llm.agents.chat import build_chat_stream, validate_chat_capabilities
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


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _merge_trace_step(
    trace_state: dict[str, TraceStep],
    step_update: TraceStep,
) -> TraceStep:
    step_id = str(step_update.get("step_id", ""))
    if not step_id:
        return step_update
    merged = dict(trace_state.get(step_id, {}))
    merged.update(step_update)
    trace_state[step_id] = merged
    return merged


def _coerce_stream_block(chunk: object) -> dict[str, object] | None:
    if isinstance(chunk, str):
        if not chunk:
            return None
        return {"type": "text", "text": chunk, "index": 0}
    if isinstance(chunk, dict):
        return dict(chunk)
    return None


def _normalize_trace_status(raw_status: object) -> str:
    if isinstance(raw_status, str) and raw_status in TRACE_STEP_STATUSES:
        return raw_status
    return "running"


def _resolve_trace_step_id(block: dict[str, object], *, step_type: str, fallback_order: int) -> str:
    raw_step_id = block.get("step_id")
    if raw_step_id:
        return str(raw_step_id).strip()
    if step_type == "thinking" and isinstance(block.get("index"), int):
        return f"thinking-{int(block['index'])}"
    return f"{step_type}-{fallback_order}"


def _build_trace_step_from_block(
    block: dict[str, object],
    *,
    fallback_order: int,
) -> TraceStep:
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


async def stream_chat_events(request: ChatRequest) -> AsyncIterator[ChatEvent]:
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
            assistant_text_chunks: list[str] = []
            active_thinking_step_id: str | None = None

            def _trace_event(step_update: TraceStep) -> ChatEvent:
                merged = _merge_trace_step(trace_state, step_update)
                return ("trace_step", merged)

            async def _close_stream() -> None:
                nonlocal stream_closed
                if stream_closed:
                    return
                stream_closed = True
                close_stream = getattr(stream, "aclose", None)
                if callable(close_stream):
                    with suppress(Exception):
                        await close_stream()

            async def _next_chunk_or_stop() -> object | None:
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
                            finalized_thinking = _finalize_thinking_step(trace_state, active_thinking_step_id)
                            if finalized_thinking is not None:
                                active_thinking_step_id = None
                                yield _trace_event(finalized_thinking)
                        text = block.get("text")
                        if isinstance(text, str) and text:
                            assistant_text_chunks.append(text)
                            yield ("chunk", {"content": text})
                    elif use_trace:
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
                assistant_message_id: int | None = None

                try:
                    async with session_factory() as session:
                        repository = ConversationRepository(session)
                        current_conversation = await repository.get_conversation(conversation.id)
                        if current_conversation is None:
                            raise ConversationNotFoundError("conversation not found")
                        if assistant_content:
                            assistant_message = await repository.add_message(
                                current_conversation,
                                role="assistant",
                                content=assistant_content,
                            )
                            assistant_message_id = assistant_message.id
                            try:
                                await refresh_summary_if_needed(repository, current_conversation)
                            except Exception:
                                logger.exception("Failed to refresh conversation summary")
                        await session.commit()
                except ConversationNotFoundError as exc:
                    yield ("error", {"detail": str(exc)})
                    return
                except Exception:
                    logger.exception("Failed to persist assistant message")
                    yield ("error", {"detail": "internal server error"})
                    return

                if assistant_message_id is not None and use_trace:
                    try:
                        async with session_factory() as session:
                            repository = ConversationRepository(session)
                            message = await repository.get_message(assistant_message_id)
                            if message is None:
                                raise ConversationNotFoundError("assistant message not found")
                            await repository.update_message_trace(
                                message,
                                trace_steps=_serialize_trace_steps(
                                    trace_state,
                                    finalize_running=not stopped,
                                ),
                            )
                            await session.commit()
                    except Exception:
                        logger.exception("Failed to persist assistant trace")
                        yield ("error", {"detail": "internal server error"})
                        return

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
