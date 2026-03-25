from app.services.chat.trace import (
    build_trace_step_from_block,
    ensure_local_thinking_step_id,
    merge_trace_step,
    serialize_trace_steps,
)


def test_merge_trace_step_merges_thinking_deltas_without_duplicate_append():
    trace_state = {
        "thinking-0": {
            "step_id": "thinking-0",
            "type": "thinking",
            "status": "running",
            "thinking": "先分析",
        }
    }

    merged = merge_trace_step(
        trace_state,
        {
            "step_id": "thinking-0",
            "type": "thinking",
            "status": "running",
            "thinking": "用户意图。",
        },
    )

    assert merged["thinking"] == "先分析用户意图。"
    assert trace_state["thinking-0"]["thinking"] == "先分析用户意图。"


def test_ensure_local_thinking_step_id_reuses_active_step_before_allocating_new_one():
    block, next_index = ensure_local_thinking_step_id(
        {"type": "thinking", "thinking": "继续思考"},
        active_thinking_step_id="thinking-3",
        next_thinking_step_index=4,
    )

    assert block["step_id"] == "thinking-3"
    assert next_index == 4


def test_serialize_trace_steps_finalizes_running_status_and_keeps_order():
    first_step = build_trace_step_from_block(
        {"type": "search", "query": "hello", "order": 2, "timestamp": "2026-03-18T12:00:01+00:00"},
        fallback_order=2,
    )
    second_step = build_trace_step_from_block(
        {"type": "thinking", "thinking": "先分析", "index": 0, "timestamp": "2026-03-18T12:00:00+00:00"},
        fallback_order=1,
    )

    steps = serialize_trace_steps(
        {
            str(first_step["step_id"]): first_step,
            str(second_step["step_id"]): second_step,
        },
        finalize_running=True,
    )

    assert steps == [
        {
            "step_id": "thinking-0",
            "type": "thinking",
            "status": "success",
            "timestamp": "2026-03-18T12:00:00+00:00",
            "order": 1,
            "thinking": "先分析",
            "index": 0,
        },
        {
            "step_id": "search-2",
            "type": "search",
            "status": "success",
            "timestamp": "2026-03-18T12:00:01+00:00",
            "order": 2,
            "query": "hello",
        },
    ]
