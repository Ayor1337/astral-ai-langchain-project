import inspect
import json
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.core.config import ConfigurationError
from app.llm.base import ThinkingNotSupportedError, UpstreamServiceError
from app.schemas.chat import ChatRequest, ChatRunStopResponse
from app.services.chat_runs import request_stop_chat_run
from app.services.chat_service import stream_chat_events
from app.services.exceptions import ChatRunNotFoundError, ConversationNotFoundError

router = APIRouter(prefix="/api/chat", tags=["chat"])


def _format_sse(event: str, payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {data}\n\n"


async def _resolve_stream(request: ChatRequest) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    stream_or_iterator = stream_chat_events(request)
    if inspect.isawaitable(stream_or_iterator):
        return await stream_or_iterator
    return stream_or_iterator


@router.post(
    "/stream",
    summary="流式聊天",
    description=(
        "接收当前轮输入，并以 SSE 形式返回会话事件和模型输出分片。"
        "推荐流程是先调用 POST /api/conversations 显式创建空会话，再携带 conversation_id 发起聊天；"
        "为兼容旧调用，不传 conversation_id 时仍会在首条消息到达时隐式创建会话。"
        "首个 conversation 事件会返回 run_id，前端可用该 run_id 调用 POST /api/chat/runs/{run_id}/stop 请求终止当前生成。"
        "请求体支持通过 thinking_enabled 控制返回模式："
        "当 thinking_enabled=true 时跳过路由，直接进入复杂执行；原始 thinking 会被整理为逐步追加的 thought_step，"
        "工具轨迹继续通过 trace_step 返回；"
        "当 thinking_enabled=false 时先做 simple/complex/agent 路由，再继续生成回答；"
        "其中 simple 路径只返回 chunk，complex/agent 路径会额外返回 route/planner_done 与链式 trace_step。"
        "首轮会话标题会在后台异步生成并落库，不会阻塞当前 SSE 收尾；前端应通过后续会话列表或详情刷新读取最新标题。"
        "链式执行轨迹统一通过 thought_step、trace_step 与 trace_done 返回。"
    ),
    responses={
        200: {"description": "SSE 流式响应"},
        400: {"description": "当前 provider 不支持请求的能力"},
        404: {"description": "会话不存在"},
        500: {"description": "服务配置错误"},
        502: {"description": "上游模型服务错误"},
    },
)
async def stream_chat(request: ChatRequest) -> StreamingResponse:
    try:
        stream = await _resolve_stream(request)
        first_event = await anext(stream)
    except ConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ThinkingNotSupportedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UpstreamServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except StopAsyncIteration:
        raise HTTPException(status_code=502, detail="chat stream produced no events")

    async def event_stream():
        first_name, first_payload = first_event
        yield _format_sse(first_name, first_payload)
        async for event, payload in stream:
            yield _format_sse(event, payload)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post(
    "/runs/{run_id}/stop",
    response_model=ChatRunStopResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="终止流式聊天",
    description="按 run_id 请求终止当前正在进行的聊天流。run_id 来自 /api/chat/stream 首个 conversation 事件。",
    responses={
        202: {"description": "已接受终止请求"},
        404: {"description": "聊天运行不存在或已结束"},
    },
)
async def stop_chat_run(run_id: UUID) -> ChatRunStopResponse:
    try:
        payload = await request_stop_chat_run(run_id)
    except ChatRunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ChatRunStopResponse.model_validate(payload)
