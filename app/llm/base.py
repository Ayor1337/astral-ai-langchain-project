from collections.abc import AsyncIterator, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.config import ModelEndpointSettings, get_settings
from app.llm.exceptions import ThinkingNotSupportedError, UpstreamServiceError
from app.llm.providers import get_provider
from app.schemas.chat import ChatMessage

ContentBlock = dict[str, object]


def _is_mapping(value: object) -> bool:
    return hasattr(value, "items")


def _to_content_block(value: object) -> ContentBlock | None:
    if _is_mapping(value):
        return dict(value)  # type: ignore[arg-type]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if _is_mapping(dumped):
            return dict(dumped)  # type: ignore[arg-type]
    to_dict = getattr(value, "dict", None)
    if callable(to_dict):
        dumped = to_dict()
        if _is_mapping(dumped):
            return dict(dumped)  # type: ignore[arg-type]
    return None


def normalize_content_blocks(content: object) -> list[ContentBlock]:
    if isinstance(content, str):
        if not content:
            return []
        return [{"type": "text", "text": content, "index": 0}]
    if isinstance(content, (list, tuple)):
        blocks: list[ContentBlock] = []
        for item in content:
            block = _to_content_block(item)
            if block:
                blocks.append(block)
        return blocks
    block = _to_content_block(content)
    if block is not None:
        return [block]
    return []


def extract_text_content(content: object) -> str:
    if isinstance(content, str):
        return content

    text_chunks: list[str] = []
    for block in normalize_content_blocks(content):
        if block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str) and text:
            text_chunks.append(text)
    return "".join(text_chunks)


def _to_langchain_message(message: ChatMessage) -> SystemMessage | HumanMessage | AIMessage:
    if message.role == "system":
        return SystemMessage(content=message.content)
    if message.role == "assistant":
        return AIMessage(content=message.content)
    return HumanMessage(content=message.content)


def create_chat_model(
    *,
    endpoint: ModelEndpointSettings,
    streaming: bool,
    thinking_enabled: bool = False,
):
    provider = get_provider(endpoint.provider)
    return provider.create_chat_model(
        endpoint=endpoint,
        streaming=streaming,
        thinking_enabled=thinking_enabled,
    )


def validate_chat_capabilities(
    *,
    endpoint: ModelEndpointSettings,
    thinking_enabled: bool = False,
) -> None:
    if not thinking_enabled:
        return
    create_chat_model(
        endpoint=endpoint,
        streaming=True,
        thinking_enabled=True,
    )


async def build_chat_stream(
    messages: Sequence[ChatMessage],
    *,
    thinking_enabled: bool = False,
) -> AsyncIterator[ContentBlock | str]:
    model = create_chat_model(
        endpoint=get_settings().chat_endpoint,
        streaming=True,
        thinking_enabled=thinking_enabled,
    )
    langchain_messages = [_to_langchain_message(message) for message in messages]

    async def iterator() -> AsyncIterator[ContentBlock | str]:
        try:
            async for chunk in model.astream(langchain_messages):
                content = getattr(chunk, "content", "")
                if isinstance(content, str):
                    if content:
                        yield {"type": "text", "text": content, "index": 0}
                    continue
                for block in normalize_content_blocks(content):
                    block_type = block.get("type")
                    if block_type == "text":
                        text = block.get("text")
                        if isinstance(text, str) and text:
                            yield block
                    elif block_type == "thinking":
                        thinking = block.get("thinking")
                        signature = block.get("signature")
                        if thinking or signature:
                            yield block
                    else:
                        yield block
        except Exception as exc:
            raise UpstreamServiceError(str(exc)) from exc

    return iterator()


async def generate_summary(
    *,
    previous_summary: str | None,
    messages: Sequence[ChatMessage],
) -> str:
    if not messages:
        return previous_summary or ""

    model = create_chat_model(
        endpoint=get_settings().chat_endpoint,
        streaming=False,
        thinking_enabled=False,
    )
    summary_prompt = [
        SystemMessage(
            content=(
                "你是对话记忆压缩器。请把新增对话压缩成简洁摘要，"
                "保留用户目标、约束、已完成事项和待办，不要编造内容。"
            )
        ),
        HumanMessage(
            content=(
                f"已有摘要：\n{previous_summary or '无'}\n\n"
                "新增消息：\n"
                + "\n".join(f"{message.role}: {message.content}" for message in messages)
                + "\n\n请输出更新后的摘要。"
            )
        ),
    ]
    try:
        response = await model.ainvoke(summary_prompt)
    except Exception as exc:
        raise UpstreamServiceError(str(exc)) from exc

    return extract_text_content(getattr(response, "content", "")).strip()
