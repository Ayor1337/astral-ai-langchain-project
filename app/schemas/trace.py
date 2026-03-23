from __future__ import annotations

from typing import Literal, TypedDict

TraceStepType = Literal["thinking", "search", "fetch", "tool_call", "tool_result", "retry", "other"]
TraceStepStatus = Literal["pending", "running", "success", "error", "skipped"]


class TraceStep(TypedDict, total=False):
    step_id: str
    parent_step_id: str
    type: TraceStepType
    thinking: str
    signature: str
    index: int
    kind: str
    status: TraceStepStatus
    title: str
    message: str
    url: str
    query: str
    result_count: int
    order: int
    payload: dict[str, object]
    error_code: str
    error_message: str
    timestamp: str
    tool_name: str
    input_json: str
    output_json: str
    duration_ms: int
    retry_of: str


class ToolCallStart(TypedDict, total=False):
    step_id: str
    tool_name: str
    input_json: str
    timestamp: str


class ToolCallDelta(TypedDict, total=False):
    step_id: str
    output_json: str
    timestamp: str


class ToolCallEnd(TypedDict, total=False):
    step_id: str
    status: TraceStepStatus
    duration_ms: int
    error_code: str
    error_message: str
    timestamp: str


class ToolResult(TypedDict, total=False):
    step_id: str
    tool_name: str
    output_json: str
    timestamp: str
