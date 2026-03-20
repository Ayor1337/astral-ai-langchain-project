import asyncio
from contextlib import suppress
import logging
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any, TypeAlias

from app.core.config import get_settings
from app.db.session import get_session_factory
from app.llm.base import UpstreamServiceError, build_chat_stream, validate_chat_capabilities
from app.llm.reasoning_agent import generate_thought_steps
from app.llm.planner_agent import plan_execution_route
from app.llm.title_agent import generate_conversation_title
from app.repositories.conversations import ConversationRepository
from app.schemas.chat import ChatMessage, ChatRequest
from app.schemas.trace import TraceStep
from app.services.chat_runs import finish_chat_run, register_chat_run
from app.services.conversation_service import DEFAULT_CONVERSATION_TITLE
from app.services.exceptions import ConversationNotFoundError
from app.services.memory_service import build_context_messages, refresh_summary_if_needed

logger = logging.getLogger(__name__)

ChatEvent: TypeAlias = tuple[str, dict[str, Any]]
TRACE_BLOCK_TYPES = {"search", "fetch", "tool_call", "tool_result", "retry", "other"}
TRACE_STEP_STATUSES = {"pending", "running", "success", "error", "skipped"}
THOUGHT_STEP_TIMEOUT_SECONDS = 99
THOUGHT_ORDER_BASE = -1000
_background_tasks: set[asyncio.Task[Any]] = set()


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


def _merge_content_block(
    content_state: dict[int, dict[str, object]],
    block_update: dict[str, object],
) -> dict[str, object]:
    raw_index = block_update.get("index")
    if isinstance(raw_index, int):
        index = raw_index
    else:
        index = 0 if block_update.get("type") == "text" else len(content_state)

    merged = dict(content_state.get(index, {"index": index}))
    for key, value in block_update.items():
        if key in {"text", "thinking", "signature"} and isinstance(value, str):
            previous = merged.get(key, "")
            merged[key] = (previous if isinstance(previous, str) else "") + value
        else:
            merged[key] = value
    content_state[index] = merged
    return merged


def _sorted_content_blocks(content_state: dict[int, dict[str, object]]) -> list[dict[str, object]]:
    return [content_state[index] for index in sorted(content_state)]


def _extract_text_from_blocks(blocks: list[dict[str, object]]) -> str:
    text_chunks: list[str] = []
    for block in blocks:
        if block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str) and text:
            text_chunks.append(text)
    return "".join(text_chunks)


def _collect_thinking_text(blocks: list[dict[str, object]]) -> str:
    thinking_chunks: list[str] = []
    for block in blocks:
        if block.get("type") != "thinking":
            continue
        thinking = block.get("thinking")
        if isinstance(thinking, str) and thinking:
            thinking_chunks.append(thinking)
    return "".join(thinking_chunks).strip()


def _should_schedule_first_round_title(
    *,
    current_title: str,
    user_message_sequence: int,
    assistant_content: str,
    stopped: bool,
) -> bool:
    return (
        not stopped
        and bool(assistant_content)
        and current_title == DEFAULT_CONVERSATION_TITLE
        and user_message_sequence == 1
    )


def _normalize_trace_status(raw_status: object) -> str:
    if isinstance(raw_status, str) and raw_status in TRACE_STEP_STATUSES:
        return raw_status
    return "running"


def _build_trace_step_from_block(
    block: dict[str, object],
    *,
    fallback_order: int,
) -> TraceStep:
    raw_type = block.get("type")
    step_type = str(raw_type) if isinstance(raw_type, str) else "other"
    if step_type not in TRACE_BLOCK_TYPES:
        step_type = "other"

    raw_step_id = block.get("step_id")
    step_id = str(raw_step_id).strip() if raw_step_id else f"{step_type}-{fallback_order}"
    trace_step: TraceStep = {
        "step_id": step_id,
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
    ):
        field_value = block.get(field_name)
        if isinstance(field_value, str) and field_value:
            trace_step[field_name] = field_value

    for int_field in ("result_count", "duration_ms"):
        field_value = block.get(int_field)
        if isinstance(field_value, int):
            trace_step[int_field] = field_value

    payload = block.get("payload")
    if isinstance(payload, dict):
        trace_step["payload"] = payload

    return trace_step


def _serialize_trace_steps(trace_state: dict[str, TraceStep]) -> list[TraceStep] | None:
    if not trace_state:
        return None
    return sorted(
        (dict(step) for step in trace_state.values()),
        key=lambda step: (
            int(step.get("order", 0)) if isinstance(step.get("order"), int) else 0,
            str(step.get("timestamp", "")),
            str(step.get("step_id", "")),
        ),
    )


def _normalize_thought_candidates(raw_steps: object) -> list[dict[str, str]]:
    if not isinstance(raw_steps, list):
        return []

    normalized_steps: list[dict[str, str]] = []
    for item in raw_steps:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        message = str(item.get("message") or "").strip()
        if not message:
            continue
        normalized_steps.append(
            {
                "title": title or f"思考步骤 {len(normalized_steps) + 1}",
                "message": message,
            }
        )
    return normalized_steps


def _build_generated_thought_step(
    *,
    step_prefix: str,
    step_index: int,
    step_count: int,
    candidate: dict[str, str],
) -> TraceStep:
    return {
        "step_id": f"{step_prefix}-{step_index + 1}",
        "type": "thought",
        "status": "running" if step_index == step_count - 1 else "success",
        "title": candidate["title"],
        "message": candidate["message"],
        "timestamp": _utcnow_iso(),
        "order": THOUGHT_ORDER_BASE + step_index,
    }


def _build_fallback_thought_trace(
    *,
    step_id: str,
    message: str,
    status: str,
    duration_ms: int | None = None,
    signature: str | None = None,
) -> TraceStep:
    trace_payload: TraceStep = {
        "step_id": step_id,
        "type": "thought",
        "kind": "thinking",
        "status": status,
        "title": "模型思考",
        "message": message,
        "timestamp": _utcnow_iso(),
        "order": THOUGHT_ORDER_BASE,
    }
    if isinstance(duration_ms, int):
        trace_payload["duration_ms"] = duration_ms
    if isinstance(signature, str) and signature:
        trace_payload["payload"] = {"signature": signature}
    return trace_payload


async def _generate_title_in_background(
    *,
    session_factory,
    conversation_id,
    first_round_messages: list[ChatMessage],
) -> None:
    try:
        generated_title = await generate_conversation_title(first_round_messages)
    except UpstreamServiceError:
        logger.exception("Failed to generate conversation title")
        return
    except Exception:
        logger.exception("Unexpected error while generating conversation title")
        return

    try:
        async with session_factory() as session:
            repository = ConversationRepository(session)
            conversation = await repository.get_conversation(conversation_id)
            if conversation is None:
                return
            if conversation.title != DEFAULT_CONVERSATION_TITLE:
                return
            if generated_title == conversation.title:
                return
            await repository.update_title(conversation, generated_title)
            await session.commit()
    except Exception:
        logger.exception("Failed to persist generated conversation title")


def _schedule_background_title_generation(
    *,
    session_factory,
    conversation_id,
    first_round_messages: list[ChatMessage],
) -> None:
    task = asyncio.create_task(
        _generate_title_in_background(
            session_factory=session_factory,
            conversation_id=conversation_id,
            first_round_messages=first_round_messages,
        )
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def stream_chat_events(request: ChatRequest) -> AsyncIterator[ChatEvent]:
    settings = get_settings()
    session_factory = get_session_factory()
    if request.thinking_enabled:
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

    planner_result: dict[str, object] | None = None
    route_name: str | None = None
    should_emit_route = False
    use_trace = request.thinking_enabled
    if request.thinking_enabled:
        route_name = "complex"
    else:
        planner_result = await plan_execution_route(message=request.message)
        route_name = str(planner_result.get("route", "")).strip()
        should_emit_route = route_name in {"complex", "agent"}
        use_trace = should_emit_route

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
                content_blocks=message.content_blocks,
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

            if should_emit_route and planner_result is not None:
                yield ("route", planner_result)
                yield ("planner_done", {"status": "completed"})

            trace_state: dict[str, TraceStep] = {}
            trace_order = 1

            def _trace_event(step_update: TraceStep) -> ChatEvent:
                merged = _merge_trace_step(trace_state, step_update)
                return ("trace_step", merged)

            def _thought_event(step_update: TraceStep) -> ChatEvent:
                merged = _merge_trace_step(trace_state, step_update)
                return ("thought_step", merged)

            assistant_content_state: dict[int, dict[str, object]] = {}
            thinking_content_state: dict[int, dict[str, object]] = {}
            generated_thought_prefix = f"assistant-thought-{conversation.id}-{user_message.sequence + 1}"
            fallback_thinking_step_id = f"assistant-thinking-{conversation.id}-{user_message.sequence + 1}"
            thinking_started_at: datetime | None = None
            thought_generation_failed = False
            current_running_generated_thought_id: str | None = None

            def _current_thinking_text() -> str:
                return _collect_thinking_text(_sorted_content_blocks(thinking_content_state))

            def _current_thinking_signature() -> str | None:
                for block in _sorted_content_blocks(thinking_content_state):
                    signature = block.get("signature")
                    if isinstance(signature, str) and signature:
                        return signature
                return None

            async def _emit_generated_thought_updates() -> list[ChatEvent]:
                nonlocal current_running_generated_thought_id, thought_generation_failed
                if not use_trace or thought_generation_failed:
                    return []

                raw_thinking = _current_thinking_text()
                if not raw_thinking:
                    return []

                existing_steps = [
                    dict(step)
                    for step in _serialize_trace_steps(trace_state) or []
                    if step.get("type") == "thought"
                ]
                raw_steps = await asyncio.wait_for(
                    generate_thought_steps(
                        user_message=request.message,
                        raw_thinking=raw_thinking,
                        existing_steps=existing_steps,
                    ),
                    timeout=THOUGHT_STEP_TIMEOUT_SECONDS,
                )
                normalized_steps = _normalize_thought_candidates(raw_steps)
                if not normalized_steps:
                    return []

                thought_events: list[ChatEvent] = []
                new_running_id: str | None = None
                total_steps = len(normalized_steps)
                for index, candidate in enumerate(normalized_steps):
                    update = _build_generated_thought_step(
                        step_prefix=generated_thought_prefix,
                        step_index=index,
                        step_count=total_steps,
                        candidate=candidate,
                    )
                    step_id = str(update["step_id"])
                    existing_step = trace_state.get(step_id)
                    if existing_step is not None and existing_step.get("status") == "success":
                        continue
                    if existing_step is not None and all(existing_step.get(field) == update.get(field) for field in ("status", "title", "message")):
                        if update["status"] == "running":
                            new_running_id = step_id
                        continue
                    thought_events.append(_thought_event(update))
                    if update["status"] == "running":
                        new_running_id = step_id
                current_running_generated_thought_id = new_running_id
                return thought_events

            def _finalize_generated_thought(status: str) -> ChatEvent | None:
                nonlocal current_running_generated_thought_id
                if not current_running_generated_thought_id:
                    return None
                current_step = trace_state.get(current_running_generated_thought_id)
                if current_step is None or current_step.get("status") != "running":
                    current_running_generated_thought_id = None
                    return None
                current_running_generated_thought_id = None
                return _thought_event(
                    {
                        **current_step,
                        "status": status,
                        "timestamp": _utcnow_iso(),
                    }
                )

            def _emit_fallback_thought(status: str, *, duration_ms: int | None = None) -> ChatEvent:
                return _trace_event(
                    _build_fallback_thought_trace(
                        step_id=fallback_thinking_step_id,
                        message=_current_thinking_text() or "正在生成思考轨迹。",
                        status=status,
                        duration_ms=duration_ms,
                        signature=_current_thinking_signature(),
                    )
                )

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
                    if use_trace and not thought_generation_failed:
                        finalized_thought = _finalize_generated_thought("success")
                        if finalized_thought is not None:
                            yield finalized_thought
                    _merge_content_block(assistant_content_state, block)
                    text = block.get("text")
                    if isinstance(text, str) and text:
                        yield ("chunk", {"content": text})
                elif use_trace and block_type == "thinking":
                    _merge_content_block(thinking_content_state, block)
                    if thinking_started_at is None:
                        thinking_started_at = datetime.now(timezone.utc)
                    if thought_generation_failed:
                        yield _emit_fallback_thought("running")
                        continue
                    try:
                        for thought_event in await _emit_generated_thought_updates():
                            yield thought_event
                    except UpstreamServiceError:
                        logger.exception("Failed to generate structured thought steps")
                        thought_generation_failed = True
                        current_running_generated_thought_id = None
                        yield _emit_fallback_thought("running")
                    except TimeoutError:
                        logger.exception("Timed out while generating structured thought steps")
                        thought_generation_failed = True
                        current_running_generated_thought_id = None
                        yield _emit_fallback_thought("running")
                elif use_trace:
                    if not thought_generation_failed:
                        finalized_thought = _finalize_generated_thought("success")
                        if finalized_thought is not None:
                            yield finalized_thought
                    trace_payload = _build_trace_step_from_block(block, fallback_order=trace_order)
                    if "order" not in block:
                        trace_order += 1
                    else:
                        trace_order = max(trace_order, int(trace_payload.get("order", trace_order)) + 1)
                    yield _trace_event(trace_payload)
            try:
                assistant_content_blocks = _sorted_content_blocks(assistant_content_state)
                assistant_content = _extract_text_from_blocks(assistant_content_blocks)
                if use_trace and thinking_started_at is not None:
                    duration_ms = int((datetime.now(timezone.utc) - thinking_started_at).total_seconds() * 1000)
                    if thought_generation_failed:
                        final_status = "skipped" if stopped else "success"
                        yield _emit_fallback_thought(final_status, duration_ms=duration_ms)
                    else:
                        finalized_thought = _finalize_generated_thought("skipped" if stopped else "success")
                        if finalized_thought is not None:
                            yield finalized_thought

                title_task_messages: list[ChatMessage] | None = None
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
                                content_blocks=assistant_content_blocks or None,
                            )
                            assistant_message_id = assistant_message.id
                            try:
                                await refresh_summary_if_needed(repository, current_conversation)
                            except Exception:
                                logger.exception("Failed to refresh conversation summary")
                            if _should_schedule_first_round_title(
                                current_title=current_conversation.title,
                                user_message_sequence=user_message.sequence,
                                assistant_content=assistant_content,
                                stopped=stopped,
                            ):
                                title_task_messages = [
                                    ChatMessage(role="user", content=user_message.content),
                                    ChatMessage(role="assistant", content=assistant_content),
                                ]
                        await session.commit()
                except ConversationNotFoundError as exc:
                    yield ("error", {"detail": str(exc)})
                    return
                except Exception:
                    logger.exception("Failed to persist assistant message")
                    yield ("error", {"detail": "internal server error"})
                    return

                if assistant_message_id is not None:
                    try:
                        async with session_factory() as session:
                            repository = ConversationRepository(session)
                            message = await repository.get_message(assistant_message_id)
                            if message is None:
                                raise ConversationNotFoundError("assistant message not found")
                            await repository.update_message_reasoning(
                                message,
                                reasoning_summary=None,
                                trace_steps=_serialize_trace_steps(trace_state) if use_trace else None,
                            )
                            await session.commit()
                    except Exception:
                        logger.exception("Failed to persist reasoning trace")
                        yield ("error", {"detail": "internal server error"})
                        return

                if title_task_messages is not None:
                    _schedule_background_title_generation(
                        session_factory=session_factory,
                        conversation_id=conversation.id,
                        first_round_messages=title_task_messages,
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
