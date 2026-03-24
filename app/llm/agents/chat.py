import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage

from app.core.config import ModelEndpointSettings, get_settings
from app.llm.exceptions import UpstreamServiceError
from app.llm.messages import ContentBlock, normalize_content_blocks, to_langchain_messages
from app.llm.models.factory import create_chat_model
from app.llm.providers import get_provider
from app.llm.tools import get_chat_tools
from app.schemas.chat import ChatMessage


def create_chat_agent(
    *,
    endpoint: ModelEndpointSettings,
    thinking_enabled: bool = False,
):
    """创建带工具能力的聊天 agent。"""
    validate_chat_capabilities(endpoint=endpoint, thinking_enabled=thinking_enabled)
    model = create_chat_model(
        endpoint=endpoint,
        streaming=True,
        thinking_enabled=thinking_enabled,
    )
    return create_agent(
        model=model,
        tools=get_chat_tools(),
        name="chat_agent",
    )


def validate_chat_capabilities(
    *,
    endpoint: ModelEndpointSettings,
    thinking_enabled: bool = False,
) -> None:
    """在真正创建模型前校验 provider 是否支持请求能力。"""
    if not thinking_enabled:
        return
    provider = get_provider(endpoint.provider)
    provider.validate_chat_capabilities(
        endpoint=endpoint,
        thinking_enabled=thinking_enabled,
    )


def _compact_json(value: object) -> str:
    """输出紧凑 JSON，便于 trace 在网络中传输和前端展示。"""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _tool_result_json(content: object) -> str:
    """工具结果如果已是字符串则原样返回，否则序列化为 JSON。"""
    if isinstance(content, str):
        return content
    return _compact_json(content)


def _iter_message_blocks(message: object) -> list[ContentBlock]:
    """从 LangChain 消息中提取统一内容块。"""
    blocks: list[ContentBlock] = []
    if isinstance(message, AIMessage):
        # tool_calls 不在 message.content 中，需要单独转成 trace 块。
        for tool_call in message.tool_calls:
            tool_name = tool_call.get("name")
            if not isinstance(tool_name, str) or not tool_name:
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


async def build_chat_stream(
    messages: Sequence[ChatMessage],
    *,
    thinking_enabled: bool = False,
) -> AsyncIterator[ContentBlock | str]:
    """构建聊天流，并把底层异常统一转换成上游服务异常。"""
    endpoint = get_settings().chat_endpoint
    agent = create_chat_agent(
        endpoint=endpoint,
        thinking_enabled=thinking_enabled,
    )
    langchain_messages = to_langchain_messages(messages)

    async def iterator() -> AsyncIterator[ContentBlock | str]:
        try:
            async for update in agent.astream(
                {"messages": langchain_messages},
                stream_mode="updates",
            ):
                for block in _iter_update_blocks(update):
                    block_type = block.get("type")
                    if block_type == "text":
                        text = block.get("text")
                        if isinstance(text, str) and text:
                            yield block
                    elif block_type == "thinking":
                        # thinking 只在包含可展示内容时向上游透传。
                        thinking = block.get("thinking")
                        signature = block.get("signature")
                        if thinking or signature:
                            yield block
                    else:
                        yield block
        except Exception as exc:
            raise UpstreamServiceError(str(exc)) from exc

    return iterator()
