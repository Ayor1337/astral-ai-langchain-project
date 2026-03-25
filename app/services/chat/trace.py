from __future__ import annotations

from datetime import datetime, timezone

from app.schemas.trace import TraceStep

TRACE_BLOCK_TYPES = {"thinking", "search", "fetch", "tool_call", "tool_result", "retry", "other"}
TRACE_STEP_STATUSES = {"pending", "running", "success", "error", "skipped"}


def utcnow_iso() -> str:
    """为 trace 事件统一生成 ISO 格式 UTC 时间戳。"""
    return datetime.now(timezone.utc).isoformat()


def merge_trace_step(
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


def coerce_stream_block(chunk: object) -> dict[str, object] | None:
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


def ensure_local_thinking_step_id(
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


def build_trace_step_from_block(
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
        "timestamp": str(block.get("timestamp") or utcnow_iso()),
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


def serialize_trace_steps(
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


def finalize_thinking_step(
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
        "timestamp": utcnow_iso(),
    }
