import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage

from app.core.config import ModelEndpointSettings, SearchSettings
from app.llm.capabilities import validate_chat_capabilities as _validate_chat_capabilities
from app.llm.exceptions import UpstreamServiceError
from app.llm.messages import ContentBlock, normalize_content_blocks, to_langchain_messages
from app.llm.models.factory import create_chat_model
from app.llm.tools import get_chat_tools
from app.services.search_service import TavilySearchService
from app.schemas.chat import ChatMessage


def create_chat_agent(
    *,
    endpoint: ModelEndpointSettings,
    thinking_enabled: bool = False,
    search_enabled: bool = False,
    search: SearchSettings | None = None,
):
    """创建带工具能力的聊天 agent。"""
    model = create_chat_model(
        endpoint=endpoint,
        streaming=True,
        thinking_enabled=thinking_enabled,
    )
    search_fn = None
    if search_enabled and search is not None:
        search_service = TavilySearchService(search)
        search_fn = search_service.search
    return create_agent(
        model=model,
        tools=get_chat_tools(search_fn=search_fn),
        name="chat_agent",
    )


def validate_chat_capabilities(
    *,
    endpoint: ModelEndpointSettings,
    thinking_enabled: bool = False,
    search_enabled: bool = False,
    search: SearchSettings | None = None,
) -> None:
    """在真正创建模型前校验 provider 是否支持请求能力。"""
    _validate_chat_capabilities(
        endpoint=endpoint,
        thinking_enabled=thinking_enabled,
        search_enabled=search_enabled,
        search=search,
    )


def _compact_json(value: object) -> str:
    """输出紧凑 JSON，便于 trace 在网络中传输和前端展示。"""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _tool_result_json(content: object) -> str:
    """工具结果如果已是字符串则原样返回，否则序列化为 JSON。"""
    if isinstance(content, str):
        return content
    return _compact_json(content)


def _parse_json_object(raw: object) -> dict[str, object]:
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, str) or not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def _build_search_call_block(tool_call: dict[str, Any]) -> ContentBlock:
    args = tool_call.get("args", {})
    query = args.get("query") if isinstance(args, dict) else None
    block: ContentBlock = {
        "type": "search",
        "step_id": str(tool_call.get("id", "")) or "search",
        "status": "running",
        "kind": "result_list",
        "message": "正在联网搜索。",
    }
    if isinstance(query, str) and query:
        block["query"] = query
    return block


def _build_search_result_block(message: ToolMessage) -> ContentBlock:
    payload = _parse_json_object(message.content)
    results = payload.get("results")
    query = payload.get("query")
    block: ContentBlock = {
        "type": "search",
        "step_id": message.tool_call_id,
        "status": "success",
        "kind": "result_list",
        "payload": {
            "results": results if isinstance(results, list) else [],
        },
    }
    if isinstance(query, str) and query:
        block["query"] = query
    if isinstance(results, list):
        block["result_count"] = len(results)
    error_message = payload.get("error")
    if isinstance(error_message, str) and error_message:
        block["status"] = "error"
        block["message"] = "联网搜索失败。"
        block["error_message"] = error_message
    return block


def _iter_message_blocks(message: object) -> list[ContentBlock]:
    """从 LangChain 消息中提取统一内容块。"""
    blocks: list[ContentBlock] = []
    if isinstance(message, AIMessage):
        # tool_calls 不在 message.content 中，需要单独转成 trace 块。
        for tool_call in message.tool_calls:
            tool_name = tool_call.get("name")
            if not isinstance(tool_name, str) or not tool_name:
                continue
            if tool_name == "web_search":
                blocks.append(_build_search_call_block(tool_call))
                continue
            blocks.append(
                {
                    "type": "tool_call",
                    "step_id": str(tool_call.get("id", "")) or tool_name,
                    "tool_name": tool_name,
                    "input_json": _compact_json(tool_call.get("args", {})),
                }
            )
        blocks.extend(normalize_content_blocks(message.content))
        return blocks

    if isinstance(message, ToolMessage):
        tool_name = message.name or "tool"
        if tool_name == "web_search":
            return [_build_search_result_block(message)]
        return [
            {
                "type": "tool_result",
                "step_id": message.tool_call_id,
                "tool_name": tool_name,
                "output_json": _tool_result_json(message.content),
            }
        ]

    content = getattr(message, "content", None)
    return normalize_content_blocks(content)


def _iter_update_blocks(update: dict[str, Any]) -> list[ContentBlock]:
    """把 LangChain updates 事件拍平成 AstralAI 自己的块序列。"""
    if not isinstance(update, dict):
        return []
    blocks: list[ContentBlock] = []
    for payload in update.values():
        if not isinstance(payload, dict):
            continue
        messages = payload.get("messages")
        if not isinstance(messages, list):
            continue
        for message in messages:
            blocks.extend(_iter_message_blocks(message))
    return blocks


def _iter_message_stream_blocks(payload: object) -> list[ContentBlock]:
    """从 messages 流模式里提取正文文本与 thinking 增量。"""
    message = payload
    metadata: dict[str, object] | None = None
    if isinstance(payload, tuple) and len(payload) == 2:
        message = payload[0]
        raw_metadata = payload[1]
        if isinstance(raw_metadata, dict):
            metadata = raw_metadata

    if isinstance(message, ToolMessage):
        return []
    if metadata is not None and metadata.get("langgraph_node") not in {None, "model"}:
        return []

    content = getattr(message, "content", message)
    blocks: list[ContentBlock] = []
    for block in normalize_content_blocks(content):
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                blocks.append(block)
            continue
        if block_type == "thinking":
            thinking = block.get("thinking")
            signature = block.get("signature")
            if thinking or signature:
                blocks.append(block)
    return blocks


def _should_yield_update_block(block: ContentBlock) -> bool:
    """updates 只负责结构化步骤，不再重复产出 messages 已覆盖的内容块。"""
    block_type = block.get("type")
    if block_type in {"text", "thinking"}:
        return False
    if block_type == "tool_call":
        tool_name = block.get("tool_name")
        input_json = block.get("input_json")
        return bool(tool_name or input_json)
    if block_type == "tool_result":
        output_json = block.get("output_json")
        tool_name = block.get("tool_name")
        return bool(tool_name or output_json)
    if block_type == "other":
        payload = block.get("payload")
        message = block.get("message")
        return bool(payload or message)
    return True


def _iter_filtered_update_blocks(payload: object) -> list[ContentBlock]:
    """过滤掉 updates 中会与 messages 重叠的文本与 thinking。"""
    blocks: list[ContentBlock] = []
    for block in _iter_update_blocks(payload):
        if _should_yield_update_block(block):
            blocks.append(block)
    return blocks


async def build_chat_stream(
    messages: Sequence[ChatMessage],
    *,
    endpoint: ModelEndpointSettings,
    thinking_enabled: bool = False,
    search_enabled: bool = False,
    search: SearchSettings | None = None,
) -> AsyncIterator[ContentBlock | str]:
    """构建聊天流，并把底层异常统一转换成上游服务异常。"""
    agent = create_chat_agent(
        **(
            {
                "endpoint": endpoint,
                "thinking_enabled": thinking_enabled,
                "search_enabled": True,
                "search": search,
            }
            if search_enabled
            else {
                "endpoint": endpoint,
                "thinking_enabled": thinking_enabled,
                "search_enabled": False,
            }
        ),
    )
    langchain_messages = to_langchain_messages(messages)
    capture_updates = thinking_enabled or search_enabled
    stream_mode: str | list[str] = ["messages", "updates"] if capture_updates else "messages"

    async def iterator() -> AsyncIterator[ContentBlock | str]:
        try:
            async for event in agent.astream(
                {"messages": langchain_messages},
                stream_mode=stream_mode,
            ):
                if capture_updates:
                    if isinstance(event, tuple) and len(event) == 2 and isinstance(event[0], str):
                        mode, payload = event
                    else:
                        mode, payload = ("updates", event) if isinstance(event, dict) else ("messages", event)
                else:
                    mode, payload = "messages", event

                if mode == "messages":
                    for block in _iter_message_stream_blocks(payload):
                        yield block
                    continue

                if mode != "updates":
                    continue

                for block in _iter_filtered_update_blocks(payload):
                    yield block
        except Exception as exc:
            raise UpstreamServiceError(str(exc)) from exc

    return iterator()
